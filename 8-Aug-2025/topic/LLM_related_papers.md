# TrajEvo: Trajectory Prediction Heuristics Design via LLM-driven Evolution 

**Title (ZH)**: TrajEvo: 基于LLM驱动进化的时间序列预测启发式设计 

**Authors**: Zhikai Zhao, Chuanbo Hua, Federico Berto, Kanghoon Lee, Zihan Ma, Jiachen Li, Jinkyoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2508.05616)  

**Abstract**: Trajectory prediction is a critical task in modeling human behavior, especially in safety-critical domains such as social robotics and autonomous vehicle navigation. Traditional heuristics based on handcrafted rules often lack accuracy and generalizability. Although deep learning approaches offer improved performance, they typically suffer from high computational cost, limited explainability, and, importantly, poor generalization to out-of-distribution (OOD) scenarios. In this paper, we introduce TrajEvo, a framework that leverages Large Language Models (LLMs) to automatically design trajectory prediction heuristics. TrajEvo employs an evolutionary algorithm to generate and refine prediction heuristics from past trajectory data. We propose two key innovations: Cross-Generation Elite Sampling to encourage population diversity, and a Statistics Feedback Loop that enables the LLM to analyze and improve alternative predictions. Our evaluations demonstrate that TrajEvo outperforms existing heuristic methods across multiple real-world datasets, and notably surpasses both heuristic and deep learning methods in generalizing to an unseen OOD real-world dataset. TrajEvo marks a promising step toward the automated design of fast, explainable, and generalizable trajectory prediction heuristics. We release our source code to facilitate future research at this https URL. 

**Abstract (ZH)**: 基于大型语言模型的轨迹进化预测框架 

---
# Simulating Human-Like Learning Dynamics with LLM-Empowered Agents 

**Title (ZH)**: 利用大语言模型赋能的代理模拟人类-like 学习动态 

**Authors**: Yu Yuan, Lili Zhao, Wei Chen, Guangting Zheng, Kai Zhang, Mengdi Zhang, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05622)  

**Abstract**: Capturing human learning behavior based on deep learning methods has become a major research focus in both psychology and intelligent systems. Recent approaches rely on controlled experiments or rule-based models to explore cognitive processes. However, they struggle to capture learning dynamics, track progress over time, or provide explainability. To address these challenges, we introduce LearnerAgent, a novel multi-agent framework based on Large Language Models (LLMs) to simulate a realistic teaching environment. To explore human-like learning dynamics, we construct learners with psychologically grounded profiles-such as Deep, Surface, and Lazy-as well as a persona-free General Learner to inspect the base LLM's default behavior. Through weekly knowledge acquisition, monthly strategic choices, periodic tests, and peer interaction, we can track the dynamic learning progress of individual learners over a full-year journey. Our findings are fourfold: 1) Longitudinal analysis reveals that only Deep Learner achieves sustained cognitive growth. Our specially designed "trap questions" effectively diagnose Surface Learner's shallow knowledge. 2) The behavioral and cognitive patterns of distinct learners align closely with their psychological profiles. 3) Learners' self-concept scores evolve realistically, with the General Learner developing surprisingly high self-efficacy despite its cognitive limitations. 4) Critically, the default profile of base LLM is a "diligent but brittle Surface Learner"-an agent that mimics the behaviors of a good student but lacks true, generalizable understanding. Extensive simulation experiments demonstrate that LearnerAgent aligns well with real scenarios, yielding more insightful findings about LLMs' behavior. 

**Abstract (ZH)**: 基于深度学习方法捕捉人类学习行为已成为心理学和智能系统领域的主要研究焦点。为了探索类似人类的学习动态，我们构建了具有心理依据的学习者画像（如深度学习者、表层学习者和懒惰学习者），并设计了一个无人物设定的一般学习者以考察基础大语言模型的默认行为。通过每周知识获取、每月策略选择、定期测验以及同伴互动，我们能够追踪学习者在全年过程中的动态学习进展。我们的发现包括四个方面：1）纵向分析表明只有深度学习者实现了持续的认知增长，我们特别设计的“陷阱问题”有效地诊断了表层学习者的浅显知识。2）不同学习者的行为和认知模式与其心理特征高度一致。3）学习者的自我概念评分真实地反映了变化，尽管基础大语言模型的认知局限，一般学习者的自我效能感却异常高。4）重要的是，基础大语言模型的默认配置是“勤奋但脆弱的表层学习者”——它模仿了好学生的行为，但缺乏真正可泛化的理解。大量的模拟实验表明，LearnerAgent 与实际场景高度契合，为其行为提供了更深刻的洞见。 

---
# Streamlining Admission with LOR Insights: AI-Based Leadership Assessment in Online Master's Program 

**Title (ZH)**: 利用推荐信洞察简化录取流程：基于AI的在线硕士学位项目领导力评估 

**Authors**: Meryem Yilmaz Soylu, Adrian Gallard, Jeonghyun Lee, Gayane Grigoryan, Rushil Desai, Stephen Harmon  

**Link**: [PDF](https://arxiv.org/pdf/2508.05513)  

**Abstract**: Letters of recommendation (LORs) provide valuable insights into candidates' capabilities and experiences beyond standardized test scores. However, reviewing these text-heavy materials is time-consuming and labor-intensive. To address this challenge and support the admission committee in providing feedback for students' professional growth, our study introduces LORI: LOR Insights, a novel AI-based detection tool for assessing leadership skills in LORs submitted by online master's program applicants. By employing natural language processing and leveraging large language models using RoBERTa and LLAMA, we seek to identify leadership attributes such as teamwork, communication, and innovation. Our latest RoBERTa model achieves a weighted F1 score of 91.6%, a precision of 92.4%, and a recall of 91.6%, showing a strong level of consistency in our test data. With the growing importance of leadership skills in the STEM sector, integrating LORI into the graduate admissions process is crucial for accurately assessing applicants' leadership capabilities. This approach not only streamlines the admissions process but also automates and ensures a more comprehensive evaluation of candidates' capabilities. 

**Abstract (ZH)**: 推荐信（LORs）提供了候选人能力与经验的重要见解，超越了标准化测试分数。然而，审查这些文字密集型材料既耗时又费力。为应对这一挑战，支持招生委员会为学生的职业发展提供反馈，本研究引入了LORI：LOR Insights，一种基于AI的评估在线硕士项目申请人推荐信中领导能力的新颖检测工具。通过运用自然语言处理并利用RoBERTa和LLAMA等大型语言模型，我们旨在识别团队合作、沟通和创新等领导特质。我们的最新RoBERTa模型在加权F1分数上达到了91.6%，精确度为92.4%，召回率为91.6%，显示了我们的测试数据中具有很强的一致性。随着STEM领域对领导技能的重要性日益增加，将LORI整合到研究生入学过程中，对于准确评估申请人的领导能力至关重要。这种方法不仅简化了招生过程，还实现了自动化评估，并确保了对候选人能力的更全面评价。 

---
# GRAIL:Learning to Interact with Large Knowledge Graphs for Retrieval Augmented Reasoning 

**Title (ZH)**: GRAIL：学习与大型知识图谱交互以增强检索推理 

**Authors**: Ge Chang, Jinbo Su, Jiacheng Liu, Pengfei Yang, Yuhao Shang, Huiwen Zheng, Hongli Ma, Yan Liang, Yuanchun Li, Yunxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05498)  

**Abstract**: Large Language Models (LLMs) integrated with Retrieval-Augmented Generation (RAG) techniques have exhibited remarkable performance across a wide range of domains. However, existing RAG approaches primarily operate on unstructured data and demonstrate limited capability in handling structured knowledge such as knowledge graphs. Meanwhile, current graph retrieval methods fundamentally struggle to capture holistic graph structures while simultaneously facing precision control challenges that manifest as either critical information gaps or excessive redundant connections, collectively undermining reasoning performance. To address this challenge, we propose GRAIL: Graph-Retrieval Augmented Interactive Learning, a framework designed to interact with large-scale graphs for retrieval-augmented reasoning. Specifically, GRAIL integrates LLM-guided random exploration with path filtering to establish a data synthesis pipeline, where a fine-grained reasoning trajectory is automatically generated for each task. Based on the synthesized data, we then employ a two-stage training process to learn a policy that dynamically decides the optimal actions at each reasoning step. The overall objective of precision-conciseness balance in graph retrieval is decoupled into fine-grained process-supervised rewards to enhance data efficiency and training stability. In practical deployment, GRAIL adopts an interactive retrieval paradigm, enabling the model to autonomously explore graph paths while dynamically balancing retrieval breadth and precision. Extensive experiments have shown that GRAIL achieves an average accuracy improvement of 21.01% and F1 improvement of 22.43% on three knowledge graph question-answering datasets. Our source code and datasets is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）结合检索增强生成（（RAG）技术取得了广泛领域内的卓越表现。

现有的RAG方法主要在结构化数据上运行，并，显示出在处理知识图谱等这类结构化知识方面能力有限。

当前的知识图检索方法在充分捕捉整体图结构方面存在困难，并且在精确度控制上面临着关键信息缺失和过多冗余连接的问题，这些因素共同损害了推理性能。

为了解决这些问题，我们提出了一种框架：图形检索增强交互学习（GRAIL）——，该框架旨在与大规模图形交互，以增强检索和推理。

具体来说，，GRAIL结合了LLM指导的随机探索与候选过滤，建立了一个合成Pipeline，在在此Pipeline上自动为每个任务生成详细的推理轨迹。基于合成的数据上，我们采用两阶段训练过程来政策化地在每次推理步骤上选择最优动作。精确度与简洁性的综合在图形检索中的整体目标被分解为两级过程：监督奖励以增强设计效率和训练稳定。

在实际部署中，GRAIL采用了交互式检索范式，使得模型能够自主探索图形空间的同时动态平衡检索广度和精确度。实验显示表明，GRAIL在准确度上有提高约11 1.1％，并在三个知识图谱问答数据集上上增强了率达到约5 reinforced 2.43％。有关的数据集可可以通过提供的URL获取。 

---
# InfiAlign: A Scalable and Sample-Efficient Framework for Aligning LLMs to Enhance Reasoning Capabilities 

**Title (ZH)**: InfiAlign: 一个 scalable 和样本高效的方法，用于调整大型语言模型以增强推理能力 

**Authors**: Shuo Cai, Su Lu, Qi Zhou, Kejing Yang, Zhijie Sang, Congkai Xie, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.05496)  

**Abstract**: Large language models (LLMs) have exhibited impressive reasoning abilities on a wide range of complex tasks. However, enhancing these capabilities through post-training remains resource intensive, particularly in terms of data and computational cost. Although recent efforts have sought to improve sample efficiency through selective data curation, existing methods often rely on heuristic or task-specific strategies that hinder scalability. In this work, we introduce InfiAlign, a scalable and sample-efficient post-training framework that integrates supervised fine-tuning (SFT) with Direct Preference Optimization (DPO) to align LLMs for enhanced reasoning. At the core of InfiAlign is a robust data selection pipeline that automatically curates high-quality alignment data from open-source reasoning datasets using multidimensional quality metrics. This pipeline enables significant performance gains while drastically reducing data requirements and remains extensible to new data sources. When applied to the Qwen2.5-Math-7B-Base model, our SFT model achieves performance on par with DeepSeek-R1-Distill-Qwen-7B, while using only approximately 12% of the training data, and demonstrates strong generalization across diverse reasoning tasks. Additional improvements are obtained through the application of DPO, with particularly notable gains in mathematical reasoning tasks. The model achieves an average improvement of 3.89% on AIME 24/25 benchmarks. Our results highlight the effectiveness of combining principled data selection with full-stage post-training, offering a practical solution for aligning large reasoning models in a scalable and data-efficient manner. The model checkpoints are available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在广泛复杂的任务上展示了令人印象深刻的推理能力。然而，通过后训练来增强这些能力依然资源密集，特别是在数据和计算成本方面。尽管最近的努力致力于通过选择性数据编辑提高样本效率，现有方法通常依赖于启发式或特定任务策略，这妨碍了扩展性。在本文中，我们提出了InfiAlign，这是一种可扩展且样本高效的后训练框架，将监督微调（SFT）与直接偏好优化（DPO）相结合，以增强大型语言模型的推理能力。InfiAlign的核心是一个稳健的数据选择管道，它使用多维质量度量从开源推理数据集中自动筛选高质量的对齐数据。该管道能够实现显著的性能提升，同时大幅减少数据需求，且易于扩展到新的数据源。应用到Qwen2.5-Math-7B-Base模型时，我们的SFT模型在使用约12%的训练数据情况下，达到与DeepSeek-R1-Distill-Qwen-7B相当的性能，并在多种推理任务中展现出强大的泛化能力。通过应用DPO还获得了额外改进，特别是在数学推理任务中取得了显著进展。模型在AIME 24/25基准上平均提升了3.89%。我们的结果突显了结合原理数据选择与全程后训练的高效性，提供了一种在可扩展和数据高效的方式下对大型推理模型进行对齐的实用解决方案。模型检查点可在此处访问：this https URL. 

---
# Can Large Language Models Generate Effective Datasets for Emotion Recognition in Conversations? 

**Title (ZH)**: 大型语言模型能否生成有效的对话情绪识别数据集？ 

**Authors**: Burak Can Kaplan, Hugo Cesar De Castro Carneiro, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2508.05474)  

**Abstract**: Emotion recognition in conversations (ERC) focuses on identifying emotion shifts within interactions, representing a significant step toward advancing machine intelligence. However, ERC data remains scarce, and existing datasets face numerous challenges due to their highly biased sources and the inherent subjectivity of soft labels. Even though Large Language Models (LLMs) have demonstrated their quality in many affective tasks, they are typically expensive to train, and their application to ERC tasks--particularly in data generation--remains limited. To address these challenges, we employ a small, resource-efficient, and general-purpose LLM to synthesize ERC datasets with diverse properties, supplementing the three most widely used ERC benchmarks. We generate six novel datasets, with two tailored to enhance each benchmark. We evaluate the utility of these datasets to (1) supplement existing datasets for ERC classification, and (2) analyze the effects of label imbalance in ERC. Our experimental results indicate that ERC classifier models trained on the generated datasets exhibit strong robustness and consistently achieve statistically significant performance improvements on existing ERC benchmarks. 

**Abstract (ZH)**: 情感对话中的情感识别（ERC）专注于识别对话中的情感转变，代表了推动机器智能发展的重大步骤。然而，ERC数据仍然稀缺，现有的数据集由于来源高度偏颇且软标签固有的主观性而面临诸多挑战。尽管大型语言模型（LLMs）在许多情感任务中表现出高质量，但它们通常训练成本高昂，在ERC任务，尤其是在数据生成方面的应用仍然有限。为解决这些挑战，我们采用了一个小型、资源高效且通用的LLM来合成具有多种特性的ERC数据集，补充了使用最广泛的三个ERC基准数据集。我们生成了六个新的数据集，其中两个专门针对增强每个基准进行了定制。我们评估了这些数据集的用途，包括（1）作为ERC分类现有数据集的补充，以及（2）分析ERC中的标签不平衡效果。实验结果表明，基于生成的数据集训练的ERC分类模型具有较强的鲁棒性，并且在现有ERC基准测试上始终能够实现统计上显著的性能提升。 

---
# Bench-2-CoP: Can We Trust Benchmarking for EU AI Compliance? 

**Title (ZH)**: Bench-2-CoP: 欧盟AI合规benchmarking能信赖吗？ 

**Authors**: Matteo Prandi, Vincenzo Suriani, Federico Pierucci, Marcello Galisai, Daniele Nardi, Piercosma Bisconti  

**Link**: [PDF](https://arxiv.org/pdf/2508.05464)  

**Abstract**: The rapid advancement of General Purpose AI (GPAI) models necessitates robust evaluation frameworks, especially with emerging regulations like the EU AI Act and its associated Code of Practice (CoP). Current AI evaluation practices depend heavily on established benchmarks, but these tools were not designed to measure the systemic risks that are the focus of the new regulatory landscape. This research addresses the urgent need to quantify this "benchmark-regulation gap." We introduce Bench-2-CoP, a novel, systematic framework that uses validated LLM-as-judge analysis to map the coverage of 194,955 questions from widely-used benchmarks against the EU AI Act's taxonomy of model capabilities and propensities. Our findings reveal a profound misalignment: the evaluation ecosystem is overwhelmingly focused on a narrow set of behavioral propensities, such as "Tendency to hallucinate" (53.7% of the corpus) and "Discriminatory bias" (28.9%), while critical functional capabilities are dangerously neglected. Crucially, capabilities central to loss-of-control scenarios, including evading human oversight, self-replication, and autonomous AI development, receive zero coverage in the entire benchmark corpus. This translates to a near-total evaluation gap for systemic risks like "Loss of Control" (0.4% coverage) and "Cyber Offence" (0.8% coverage). This study provides the first comprehensive, quantitative analysis of this gap, offering critical insights for policymakers to refine the CoP and for developers to build the next generation of evaluation tools, ultimately fostering safer and more compliant AI. 

**Abstract (ZH)**: 《通用人工智能（GPAI）模型的迅速发展需要 robust 的评估框架，，特别在欧盟AI法案及其
user
好的背景下，特别是在欧盟AI法案及其
的背景下，引入一种基于系统框架的Bench-CoP评估方法 杨标准化的LLM作为法官分析来映射广泛使用的评估基准针对欧盟AI法案的能力分类和倾向分类 我们的研究发现了一个深刻的分歧: �当前的评估生态体系主要过度关注于一小部分行为倾向,例如“妄想倾向”（37% 的覆盖率）和“歧视性倾向”（88.55% 的覆盖率）而忽略了了功能能力 on这些能力在失控场景中可能会逃避人类监管 值得重视的自主复制复制重复和自主以及自主生成 AI 的发展完全没有覆盖 则这导致了对对于“失控"（ on4 on on的 的覆盖率）和“网络攻击”（（8 on on % on on on on on on on 的映像覆盖 本研究通过全面的定量分析阐述了这一gap



的回答

Assistant直接输出标题：Bench-CoP：一种基于系统标准化LL 

---
# Large Language Models Transform Organic Synthesis From Reaction Prediction to Automation 

**Title (ZH)**: 大型语言模型将有机合成从反应预测转变为自动化。 

**Authors**: Kartar Kumar Lohana Tharwani, Rajesh Kumar, Sumita, Numan Ahmed, Yong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.05427)  

**Abstract**: Large language models (LLMs) are beginning to reshape how chemists plan and run reactions in organic synthesis. Trained on millions of reported transformations, these text-based models can propose synthetic routes, forecast reaction outcomes and even instruct robots that execute experiments without human supervision. Here we survey the milestones that turned LLMs from speculative tools into practical lab partners. We show how coupling LLMs with graph neural networks, quantum calculations and real-time spectroscopy shrinks discovery cycles and supports greener, data-driven chemistry. We discuss limitations, including biased datasets, opaque reasoning and the need for safety gates that prevent unintentional hazards. Finally, we outline community initiatives open benchmarks, federated learning and explainable interfaces that aim to democratize access while keeping humans firmly in control. These advances chart a path towards rapid, reliable and inclusive molecular innovation powered by artificial intelligence and automation. 

**Abstract (ZH)**: 大规模语言模型（LLMs）正在重新定义化学家在有机合成中规划和运行反应的方式。这些基于文本的模型经过数百万个报告的转化训练，可以提出合成路线、预测反应结果，甚至可以指导无需人类监督的机器人执行实验。在这里，我们回顾了LLMs从 speculative 工具转变为实用实验室伙伴的关键里程碑。我们展示了将LLMs与图神经网络、量子计算和实时光谱学相结合如何缩短发现周期，并支持更绿色、更数据驱动的化学。我们讨论了包括偏差数据集、不透明推理以及需要防止意外危害的安全门在内的限制。最后，我们概述了旨在促进平等访问同时确保人类始终处于控制之中的社区倡议，包括开放基准、联邦学习和可解释接口。这些进展描绘了一条通往由人工智能和自动化驱动的快速、可靠和包容的分子创新之路。 

---
# NomicLaw: Emergent Trust and Strategic Argumentation in LLMs During Collaborative Law-Making 

**Title (ZH)**: 名义法：LLMs协作立法过程中涌现的信任与战略论证 

**Authors**: Asutosh Hota, Jussi P.P. Jokinen  

**Link**: [PDF](https://arxiv.org/pdf/2508.05344)  

**Abstract**: Recent advancements in large language models (LLMs) have extended their capabilities from basic text processing to complex reasoning tasks, including legal interpretation, argumentation, and strategic interaction. However, empirical understanding of LLM behavior in open-ended, multi-agent settings especially those involving deliberation over legal and ethical dilemmas remains limited. We introduce NomicLaw, a structured multi-agent simulation where LLMs engage in collaborative law-making, responding to complex legal vignettes by proposing rules, justifying them, and voting on peer proposals. We quantitatively measure trust and reciprocity via voting patterns and qualitatively assess how agents use strategic language to justify proposals and influence outcomes. Experiments involving homogeneous and heterogeneous LLM groups demonstrate how agents spontaneously form alliances, betray trust, and adapt their rhetoric to shape collective decisions. Our results highlight the latent social reasoning and persuasive capabilities of ten open-source LLMs and provide insights into the design of future AI systems capable of autonomous negotiation, coordination and drafting legislation in legal settings. 

**Abstract (ZH)**: 近期大型语言模型的 advancements 已将其实现能力从基本的文字处理扩展到复杂的推理任务，包括法律解释、论辩和战略互动。然而，关于大型语言模型在开放性、多智能体环境中的行为理解，特别是在涉及法律和伦理困境的讨论中，仍然有限。我们引入了 NomicLaw，这是一种结构化的多智能体模拟，在此模型中，智能语言模型参与协作立法，通过提出规则、进行正当化并投票评估同伴提案来应对复杂的法律情境。我们通过投票模式定量地测量信任和互惠，并通过智能体如何使用战略语言正当化提案及其对结果的影响进行定性的评估。涉及同质性和异质性大型语言模型组的实验展示了智能体如何自发地形成联盟、背叛信任，并适应其修辞以塑造集体决策。研究结果突显了十个开源大型语言模型潜在的社会推理和说服能力，并为设计具有自主谈判、协调和在法律环境中起草法律文件能力的未来AI系统提供了见解。 

---
# The Term 'Agent' Has Been Diluted Beyond Utility and Requires Redefinition 

**Title (ZH)**: 术语“代理”已丧失实用价值，需重新定义。 

**Authors**: Brinnae Bent  

**Link**: [PDF](https://arxiv.org/pdf/2508.05338)  

**Abstract**: The term 'agent' in artificial intelligence has long carried multiple interpretations across different subfields. Recent developments in AI capabilities, particularly in large language model systems, have amplified this ambiguity, creating significant challenges in research communication, system evaluation and reproducibility, and policy development. This paper argues that the term 'agent' requires redefinition. Drawing from historical analysis and contemporary usage patterns, we propose a framework that defines clear minimum requirements for a system to be considered an agent while characterizing systems along a multidimensional spectrum of environmental interaction, learning and adaptation, autonomy, goal complexity, and temporal coherence. This approach provides precise vocabulary for system description while preserving the term's historically multifaceted nature. After examining potential counterarguments and implementation challenges, we provide specific recommendations for moving forward as a field, including suggestions for terminology standardization and framework adoption. The proposed approach offers practical tools for improving research clarity and reproducibility while supporting more effective policy development. 

**Abstract (ZH)**: 人工智能中“代理”一词长期具有多重解释。近期人工智能能力的发展，特别是大型语言模型系统的进展，加剧了这一歧义，为研究通讯、系统评估与再现性和政策制定带来了重大挑战。本文认为“代理”这一术语需要重新定义。通过历史分析和当代使用模式，我们提出了一种框架，明确规定了系统作为代理所应满足的基本要求，并以其在环境交互、学习与适应、自主性、目标复杂性和时序一致性等多维度方面的表现对系统进行分类。这种方法为系统描述提供了精确的词汇，同时保留了该术语历史上的多重含义。在探讨潜在的反论和实施挑战后，我们提供了具体建议，以推动该领域的发展，包括术语标准化和框架采用的建议。提出的这种方法提供了实用工具，以提高研究的清晰度和再现性，并支持更有效的政策制定。 

---
# A Novel Architecture for Symbolic Reasoning with Decision Trees and LLM Agents 

**Title (ZH)**: 一种基于决策树和LLM代理的符号推理新架构 

**Authors**: Andrew Kiruluta  

**Link**: [PDF](https://arxiv.org/pdf/2508.05311)  

**Abstract**: We propose a hybrid architecture that integrates decision tree-based symbolic reasoning with the generative capabilities of large language models (LLMs) within a coordinated multi-agent framework. Unlike prior approaches that loosely couple symbolic and neural modules, our design embeds decision trees and random forests as callable oracles within a unified reasoning system. Tree-based modules enable interpretable rule inference and causal logic, while LLM agents handle abductive reasoning, generalization, and interactive planning. A central orchestrator maintains belief state consistency and mediates communication across agents and external tools, enabling reasoning over both structured and unstructured inputs.
The system achieves strong performance on reasoning benchmarks. On \textit{ProofWriter}, it improves entailment consistency by +7.2\% through logic-grounded tree validation. On GSM8k, it achieves +5.3\% accuracy gains in multistep mathematical problems via symbolic augmentation. On \textit{ARC}, it boosts abstraction accuracy by +6.0\% through integration of symbolic oracles. Applications in clinical decision support and scientific discovery show how the system encodes domain rules symbolically while leveraging LLMs for contextual inference and hypothesis generation. This architecture offers a robust, interpretable, and extensible solution for general-purpose neuro-symbolic reasoning. 

**Abstract (ZH)**: 我们提出了一种混合架构，将基于决策树的符号推理与大规模语言模型（LLMs）的生成能力结合在一个协调的多智能体框架内。与以往松散耦合符号和神经模块的方法不同，我们的设计将决策树和随机森林嵌入为统一推理系统中的可调用先知。基于树的模块支持可解释的规则推理和因果逻辑，而LLM智能体处理演绎推理、泛化和交互式规划。中心协调器维护信念状态一致性，并介调智能体间及与外部工具的通信，支持结构化和非结构化输入的推理。该系统在推理基准测试中表现出色。在ProofWriter上，通过逻辑支持的树验证提高了蕴含一致性7.2%。在GSM8k上，通过符号增强在多步数学问题上实现了5.3%的准确率提升。在ARC上，通过符号先知的整合提高了抽象准确率6.0%。在临床决策支持和科学发现应用中展示了该系统如何以符号方式编码领域规则，并利用LLMs进行上下文推理和假设生成。该架构为通用目的神经-符号推理提供了稳健、可解释和可扩展的解决方案。 

---
# An Explainable Natural Language Framework for Identifying and Notifying Target Audiences In Enterprise Communication 

**Title (ZH)**: 可解释的自然语言框架：识别和通知目标受众在企业沟通中的应用 

**Authors**: Vítor N. Lourenço, Mohnish Dubey, Yunfei Bai, Audrey Depeige, Vivek Jain  

**Link**: [PDF](https://arxiv.org/pdf/2508.05267)  

**Abstract**: In large-scale maintenance organizations, identifying subject matter experts and managing communications across complex entities relationships poses significant challenges -- including information overload and longer response times -- that traditional communication approaches fail to address effectively. We propose a novel framework that combines RDF graph databases with LLMs to process natural language queries for precise audience targeting, while providing transparent reasoning through a planning-orchestration architecture. Our solution enables communication owners to formulate intuitive queries combining concepts such as equipment, manufacturers, maintenance engineers, and facilities, delivering explainable results that maintain trust in the system while improving communication efficiency across the organization. 

**Abstract (ZH)**: 在大型维护组织中，识别领域专家并管理复杂实体关系之间的沟通面临显著挑战——包括信息过载和响应时间延长——传统沟通方法无法有效解决这些问题。我们提出了一种结合RDF图数据库与LLMs的新框架，用于处理自然语言查询以实现精确的目标受众定位，并通过规划- orchestration架构提供透明的推理过程。我们的解决方案使沟通所有者能够提出直观的查询，结合设备、制造商、维护工程师和设施等概念，提供可解释的结果，从而在提高组织内沟通效率的同时保持对系统的信任。 

---
# Beyond Automation: Socratic AI, Epistemic Agency, and the Implications of the Emergence of Orchestrated Multi-Agent Learning Architectures 

**Title (ZH)**: 超越自动化：苏格拉底式AI、认识论agency及其协同多智能体学习架构兴起的意义 

**Authors**: Peer-Benedikt Degen, Igor Asanov  

**Link**: [PDF](https://arxiv.org/pdf/2508.05116)  

**Abstract**: Generative AI is no longer a peripheral tool in higher education. It is rapidly evolving into a general-purpose infrastructure that reshapes how knowledge is generated, mediated, and validated. This paper presents findings from a controlled experiment evaluating a Socratic AI Tutor, a large language model designed to scaffold student research question development through structured dialogue grounded in constructivist theory. Conducted with 65 pre-service teacher students in Germany, the study compares interaction with the Socratic Tutor to engagement with an uninstructed AI chatbot. Students using the Socratic Tutor reported significantly greater support for critical, independent, and reflective thinking, suggesting that dialogic AI can stimulate metacognitive engagement and challenging recent narratives of de-skilling due to generative AI usage. These findings serve as a proof of concept for a broader pedagogical shift: the use of multi-agent systems (MAS) composed of specialised AI agents. To conceptualise this, we introduce the notion of orchestrated MAS, modular, pedagogically aligned agent constellations, curated by educators, that support diverse learning trajectories through differentiated roles and coordinated interaction. To anchor this shift, we propose an adapted offer-and-use model, in which students appropriate instructional offers from these agents. Beyond technical feasibility, we examine system-level implications for higher education institutions and students, including funding necessities, changes to faculty roles, curriculars, competencies and assessment practices. We conclude with a comparative cost-effectiveness analysis highlighting the scalability of such systems. In sum, this study contributes both empirical evidence and a conceptual roadmap for hybrid learning ecosystems that embed human-AI co-agency and pedagogical alignment. 

**Abstract (ZH)**: 生成式AI已不再是高等教育中的边缘工具，而是迅速发展成为一种通用基础设施，重塑知识的生成、传递和验证方式。本文呈现了一项受控实验的发现，该实验评估了基于建构主义理论的结构化对话设计的苏格拉底AI导师，比较了它与未经指导的AI聊天机器人互动对学生的影响。使用苏格拉底导师的学生报告了在批判性、独立性和反思性思维方面获得了显著更多的支持，这表明对话式AI可以激发元认知参与，挑战由于生成式AI使用而出现的去技能化叙事。这些发现作为多智能体系统（MAS）更广泛教学生态系统的概念证明：由专业化AI代理组成的协调MAS。为了构想这一转变，我们引入了协调MAS的概念，这是一种由教育者策划的模块化、教学目标对齐的代理组合，通过差异化角色和协调互动支持多样化的学习路径。为了支持这一转变，我们提出了一个修改后的提供与使用模型，使学生能够使用这些代理提供的教学服务。除了技术可行性，我们还探讨了对于高等教育机构和学生的系统级影响，包括资金需求、教职员工角色变化、课程设置、技能和评估实践的变化。最后，我们进行了一项成本效益分析，强调了此类系统的可扩展性。总的来说，本研究通过实证证据和概念蓝图，为嵌入人类与AI协作和教学目标对齐的混合学习生态系统做出了贡献。 

---
# EasySize: Elastic Analog Circuit Sizing via LLM-Guided Heuristic Search 

**Title (ZH)**: EasySize: 基于LLM引导启发式搜索的弹性模拟电路尺寸调整 

**Authors**: Xinyue Wu, Fan Hu, Shaik Jani Babu, Yi Zhao, Xinfei Guo  

**Link**: [PDF](https://arxiv.org/pdf/2508.05113)  

**Abstract**: Analog circuit design is a time-consuming, experience-driven task in chip development. Despite advances in AI, developing universal, fast, and stable gate sizing methods for analog circuits remains a significant challenge. Recent approaches combine Large Language Models (LLMs) with heuristic search techniques to enhance generalizability, but they often depend on large model sizes and lack portability across different technology nodes. To overcome these limitations, we propose EasySize, the first lightweight gate sizing framework based on a finetuned Qwen3-8B model, designed for universal applicability across process nodes, design specifications, and circuit topologies. EasySize exploits the varying Ease of Attainability (EOA) of performance metrics to dynamically construct task-specific loss functions, enabling efficient heuristic search through global Differential Evolution (DE) and local Particle Swarm Optimization (PSO) within a feedback-enhanced flow. Although finetuned solely on 350nm node data, EasySize achieves strong performance on 5 operational amplifier (Op-Amp) netlists across 180nm, 45nm, and 22nm technology nodes without additional targeted training, and outperforms AutoCkt, a widely-used Reinforcement Learning based sizing framework, on 86.67\% of tasks with more than 96.67\% of simulation resources reduction. We argue that EasySize can significantly reduce the reliance on human expertise and computational resources in gate sizing, thereby accelerating and simplifying the analog circuit design process. EasySize will be open-sourced at a later date. 

**Abstract (ZH)**: 基于Qwen3-8B微调模型的轻量级门级尺寸优化框架EasySize 

---
# MedMKEB: A Comprehensive Knowledge Editing Benchmark for Medical Multimodal Large Language Models 

**Title (ZH)**: MedMKEB: 一种全面的医学多模态大型语言模型知识编辑基准 

**Authors**: Dexuan Xu, Jieyi Wang, Zhongyan Chai, Yongzhi Cao, Hanpin Wang, Huamin Zhang, Yu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.05083)  

**Abstract**: Recent advances in multimodal large language models (MLLMs) have significantly improved medical AI, enabling it to unify the understanding of visual and textual information. However, as medical knowledge continues to evolve, it is critical to allow these models to efficiently update outdated or incorrect information without retraining from scratch. Although textual knowledge editing has been widely studied, there is still a lack of systematic benchmarks for multimodal medical knowledge editing involving image and text modalities. To fill this gap, we present MedMKEB, the first comprehensive benchmark designed to evaluate the reliability, generality, locality, portability, and robustness of knowledge editing in medical multimodal large language models. MedMKEB is built on a high-quality medical visual question-answering dataset and enriched with carefully constructed editing tasks, including counterfactual correction, semantic generalization, knowledge transfer, and adversarial robustness. We incorporate human expert validation to ensure the accuracy and reliability of the benchmark. Extensive single editing and sequential editing experiments on state-of-the-art general and medical MLLMs demonstrate the limitations of existing knowledge-based editing approaches in medicine, highlighting the need to develop specialized editing strategies. MedMKEB will serve as a standard benchmark to promote the development of trustworthy and efficient medical knowledge editing algorithms. 

**Abstract (ZH)**: Recent Advances in Multimodal Large Language Models for Medical AI: MedMKEB, the First Comprehensive Benchmark for Evaluating Medical Multimodal Knowledge Editing 

---
# Can Large Language Models Integrate Spatial Data? Empirical Insights into Reasoning Strengths and Computational Weaknesses 

**Title (ZH)**: 大型语言模型能否整合空间数据？关于推理优势与计算劣势的经验洞察 

**Authors**: Bin Han, Robert Wolfe, Anat Caspi, Bill Howe  

**Link**: [PDF](https://arxiv.org/pdf/2508.05009)  

**Abstract**: We explore the application of large language models (LLMs) to empower domain experts in integrating large, heterogeneous, and noisy urban spatial datasets. Traditional rule-based integration methods are unable to cover all edge cases, requiring manual verification and repair. Machine learning approaches require collecting and labeling of large numbers of task-specific samples. In this study, we investigate the potential of LLMs for spatial data integration. Our analysis first considers how LLMs reason about environmental spatial relationships mediated by human experience, such as between roads and sidewalks. We show that while LLMs exhibit spatial reasoning capabilities, they struggle to connect the macro-scale environment with the relevant computational geometry tasks, often producing logically incoherent responses. But when provided relevant features, thereby reducing dependence on spatial reasoning, LLMs are able to generate high-performing results. We then adapt a review-and-refine method, which proves remarkably effective in correcting erroneous initial responses while preserving accurate responses. We discuss practical implications of employing LLMs for spatial data integration in real-world contexts and outline future research directions, including post-training, multi-modal integration methods, and support for diverse data formats. Our findings position LLMs as a promising and flexible alternative to traditional rule-based heuristics, advancing the capabilities of adaptive spatial data integration. 

**Abstract (ZH)**: 我们探索大型语言模型（LLMs）在增强领域专家整合大规模、异构和噪音城市空间数据方面的应用。传统的基于规则的集成方法无法覆盖所有边缘情况，需要人工验证和修复。机器学习方法需要收集和标注大量特定任务的数据样本。在本研究中，我们 investigate LLMs 在空间数据集成中的潜在应用。我们的分析首先考虑了LLMs 如何基于人类经验处理环境空间关系，例如道路与人行道之间的关系。我们发现尽管LLMs展示出空间推理能力，但在连接宏观环境与相关计算几何任务方面常常表现不佳，常产生逻辑不连贯的响应。但在提供相关特征后，从而减少对空间推理的依赖，LLMs能够生成高性能的结果。我们then适应了一种审查和修正的方法，这种方法在纠正初始错误响应的同时保留了准确的响应，证明非常有效。我们讨论了在实际应用中采用LLMs进行空间数据集成的实用意义，并列出了未来研究方向，包括后训练、多模态集成方法以及支持多种数据格式。我们的研究结果定位LLMs为传统基于规则的启发式方法的一种有前景且灵活的替代方案，推动了自适应空间数据集成能力的发展。 

---
# Large Language Models Reasoning Abilities Under Non-Ideal Conditions After RL-Fine-Tuning 

**Title (ZH)**: 大型语言模型在非理想条件下的推理能力：强化学习微调后的表现 

**Authors**: Chang Tian, Matthew B. Blaschko, Mingzhe Xing, Xiuxing Li, Yinliang Yue, Marie-Francine Moens  

**Link**: [PDF](https://arxiv.org/pdf/2508.04848)  

**Abstract**: Reinforcement learning (RL) has become a key technique for enhancing the reasoning abilities of large language models (LLMs), with policy-gradient algorithms dominating the post-training stage because of their efficiency and effectiveness. However, most existing benchmarks evaluate large-language-model reasoning under idealized settings, overlooking performance in realistic, non-ideal scenarios. We identify three representative non-ideal scenarios with practical relevance: summary inference, fine-grained noise suppression, and contextual filtering. We introduce a new research direction guided by brain-science findings that human reasoning remains reliable under imperfect inputs. We formally define and evaluate these challenging scenarios. We fine-tune three LLMs and a state-of-the-art large vision-language model (LVLM) using RL with a representative policy-gradient algorithm and then test their performance on eight public datasets. Our results reveal that while RL fine-tuning improves baseline reasoning under idealized settings, performance declines significantly across all three non-ideal scenarios, exposing critical limitations in advanced reasoning capabilities. Although we propose a scenario-specific remediation method, our results suggest current methods leave these reasoning deficits largely unresolved. This work highlights that the reasoning abilities of large models are often overstated and underscores the importance of evaluating models under non-ideal scenarios. The code and data will be released at XXXX. 

**Abstract (ZH)**: reinforcement learning (RL)已在增强大型语言模型（LLMs）的推理能力方面成为关键技术，策略梯度算法因其高效性和有效性在训练后阶段占据主导地位。然而，现有的大多数基准测试在理想化的设定下评估大型语言模型的推理能力，忽视了在现实的非理想场景中的表现。我们识别了三个具有实际意义的非理想场景：摘要推断、精细噪声抑制和上下文过滤。我们提出了一种新的研究方向，受到脑科学发现的启发，体现人类推理在不完美的输入下依然可靠。我们正式定义并评估了这些具有挑战性的场景。我们使用RL和一个代表性的策略梯度算法微调了三种LLMs和一个最先进的大型视觉-语言模型（LVLM），并在八个公开数据集上测试它们的性能。结果显示，虽然RL微调在理想化的设定下提高了基线推理能力，但在所有三个非理想场景中的表现显著下降，揭示了高级推理能力的关键限制。尽管我们提出了一种特定场景的补救方法，但结果表明当前方法在很大程度上未能解决这些推理缺陷。这项工作凸显了大型模型的推理能力往往被夸大，并强调了在非理想场景下评估模型的重要性。代码和数据将在XXXX发布。 

---
# Fine-Tuning Small Language Models (SLMs) for Autonomous Web-based Geographical Information Systems (AWebGIS) 

**Title (ZH)**: 细调小型语言模型（SLMs）以应用于自主基于Web的地理信息系统（AWebGIS） 

**Authors**: Mahdi Nazari Ashani, Ali Asghar Alesheikh, Saba Kazemi, Kimya Kheirkhah, Yasin Mohammadi, Fatemeh Rezaie, Amir Mahdi Manafi, Hedieh Zarkesh  

**Link**: [PDF](https://arxiv.org/pdf/2508.04846)  

**Abstract**: Autonomous web-based geographical information systems (AWebGIS) aim to perform geospatial operations from natural language input, providing intuitive, intelligent, and hands-free interaction. However, most current solutions rely on cloud-based large language models (LLMs), which require continuous internet access and raise users' privacy and scalability issues due to centralized server processing. This study compares three approaches to enabling AWebGIS: (1) a fully-automated online method using cloud-based LLMs (e.g., Cohere); (2) a semi-automated offline method using classical machine learning classifiers such as support vector machine and random forest; and (3) a fully autonomous offline (client-side) method based on a fine-tuned small language model (SLM), specifically T5-small model, executed in the client's web browser. The third approach, which leverages SLMs, achieved the highest accuracy among all methods, with an exact matching accuracy of 0.93, Levenshtein similarity of 0.99, and recall-oriented understudy for gisting evaluation ROUGE-1 and ROUGE-L scores of 0.98. Crucially, this client-side computation strategy reduces the load on backend servers by offloading processing to the user's device, eliminating the need for server-based inference. These results highlight the feasibility of browser-executable models for AWebGIS solutions. 

**Abstract (ZH)**: 基于自主的网络地理信息系统（AWebGIS）旨在从自然语言输入中执行空间操作，提供直观、智能且无需手动的交互。然而，当前大多数解决方案依赖于基于云的大语言模型（LLMs），这需要持续的互联网连接，并由于集中式服务器处理而引起用户隐私和可扩展性问题。本研究比较了三种使AWebGIS自主化的 approach：（1）完全自动化在线方法，使用基于云的LLMs（如Cohere）；（2）半自动化离线方法，使用经典机器学习分类器如支持向量机和支持向量森林；（3）基于微调小语言模型（SLM），特别是T5-small模型，在客户端浏览器中执行的完全自主离线（客户端侧）方法。第三种方法利用SLMs，实现了所有方法中最高的准确性，精确匹配准确率为0.93，Levenshtein相似度为0.99，以及基于召回的摘要评估ROUGE-1和ROUGE-L得分为0.98。关键的是，这种客户端计算策略通过将处理任务转移到用户的设备上减轻了后端服务器的负担，消除了基于服务器的推理需求。这些结果突显了浏览器可执行模型在AWebGIS解决方案中的可行性。 

---
# Who is a Better Player: LLM against LLM 

**Title (ZH)**: 谁是更好的玩家：大规模语言模型对抗大规模语言模型 

**Authors**: Yingjie Zhou, Jiezhang Cao, Farong Wen, Li Xu, Yanwei Jiang, Jun Jia, Ronghui Li, Xiaohong Liu, Yu Zhou, Xiongkuo Min, Jie Guo, Zicheng Zhang, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2508.04720)  

**Abstract**: Adversarial board games, as a paradigmatic domain of strategic reasoning and intelligence, have long served as both a popular competitive activity and a benchmark for evaluating artificial intelligence (AI) systems. Building on this foundation, we propose an adversarial benchmarking framework to assess the comprehensive performance of Large Language Models (LLMs) through board games competition, compensating the limitation of data dependency of the mainstream Question-and-Answer (Q&A) based benchmark method. We introduce Qi Town, a specialized evaluation platform that supports 5 widely played games and involves 20 LLM-driven players. The platform employs both the Elo rating system and a novel Performance Loop Graph (PLG) to quantitatively evaluate the technical capabilities of LLMs, while also capturing Positive Sentiment Score (PSS) throughout gameplay to assess mental fitness. The evaluation is structured as a round-robin tournament, enabling systematic comparison across players. Experimental results indicate that, despite technical differences, most LLMs remain optimistic about winning and losing, demonstrating greater adaptability to high-stress adversarial environments than humans. On the other hand, the complex relationship between cyclic wins and losses in PLGs exposes the instability of LLMs' skill play during games, warranting further explanation and exploration. 

**Abstract (ZH)**: adversarial棋盘游戏作为战略推理和智能的一个典范领域，长期以来既是流行的竞技活动，也是评估人工智能系统性能的标准。基于此，我们提出了一种对抗基准框架，通过棋盘游戏竞赛评估大型语言模型（LLMs）的综合性能，弥补主流基于问答（Q&A）基准方法的数据依赖性限制。我们介绍了棋镇，这是一个专门的评估平台，支持5种广泛玩的棋盘游戏，并涉及20个LLM驱动的玩家。该平台采用Elo排名系统和新型性能循环图（PLG）来定量评估LLMs的技术能力，同时在整个游戏过程中捕获正面情绪分值（PSS）来评估玩家的心理状态。评估结构化为循环赛制，使玩家之间的系统比较成为可能。实验结果表明，尽管存在技术差异，大多数LLMs在赢得和失去方面的乐观态度一致，显示出比人类更大的适应高压对抗环境的能力。另一方面，PLGs中循环胜利与失败之间的复杂关系揭示了LLMs在游戏中的技能表现不稳定，需要进一步解释和探究。 

---
# Prescriptive Agents based on Rag for Automated Maintenance (PARAM) 

**Title (ZH)**: 基于Rag的规范代理用于自动化维护（PARAM） 

**Authors**: Chitranshu Harbola, Anupam Purwar  

**Link**: [PDF](https://arxiv.org/pdf/2508.04714)  

**Abstract**: Industrial machinery maintenance requires timely intervention to prevent catastrophic failures and optimize operational efficiency. This paper presents an integrated Large Language Model (LLM)-based intelligent system for prescriptive maintenance that extends beyond traditional anomaly detection to provide actionable maintenance recommendations. Building upon our prior LAMP framework for numerical data analysis, we develop a comprehensive solution that combines bearing vibration frequency analysis with multi agentic generation for intelligent maintenance planning. Our approach serializes bearing vibration data (BPFO, BPFI, BSF, FTF frequencies) into natural language for LLM processing, enabling few-shot anomaly detection with high accuracy. The system classifies fault types (inner race, outer race, ball/roller, cage faults) and assesses severity levels. A multi-agentic component processes maintenance manuals using vector embeddings and semantic search, while also conducting web searches to retrieve comprehensive procedural knowledge and access up-to-date maintenance practices for more accurate and in-depth recommendations. The Gemini model then generates structured maintenance recommendations includes immediate actions, inspection checklists, corrective measures, parts requirements, and timeline specifications. Experimental validation in bearing vibration datasets demonstrates effective anomaly detection and contextually relevant maintenance guidance. The system successfully bridges the gap between condition monitoring and actionable maintenance planning, providing industrial practitioners with intelligent decision support. This work advances the application of LLMs in industrial maintenance, offering a scalable framework for prescriptive maintenance across machinery components and industrial sectors. 

**Abstract (ZH)**: 工业机械维护需要及时干预以防止灾难性 的故障并优化运营效率。本文提出了一种基于大型语言模型（LLM）的智能预测性 维护系统，该系统超越了传统的异常检测，提供可 动态的维护建议。依托我们之前的 LAMP 框架进行数值数据分析，，我们开发了一个综合性的系统，将轴承振动频率分析与多代理生成结合，以实现智能维护计划。我们的方法将轴承振动数据（BPFO、BPFI、BSF、FTF 频率）转换为自然语言供 LLM 处理，实现高精度的异常检测。该系统可以对故障类型（内圈故障、外圈故障、类 蛇形滚子故障、保持架故障）进行分类并评估严重程度。多代理系统使用向量嵌入和 语义分类处理维护手册，同时进行网络搜索以检索全面的程序知识并 最新的维护实践，从而提供更准确的深入建议。Gemini 系统然后生成结构化的维护建议，包括立即行动、检查清单、纠正措施、文件要求和时间表规范。在轴承振动数据集上的实验验证了异常检测和上下文相关维护指导。该系统成功地地填补了状态监测与动态可 动态的维护规划 之间的差距，为工业从业者提供智能决策支持。这项工作推进了 LLM 在工业维护中的的应用，提供了可扩展的框架以实现整个机械设备和制造业领域的预测维护。 

---
# H-Net++: Hierarchical Dynamic Chunking for Tokenizer-Free Language Modelling in Morphologically-Rich Languages 

**Title (ZH)**: H-Net++: 基于层次动态切块的形态丰富语言无分词器语言建模 

**Authors**: Mehrdad Zakershahrak, Samira Ghodratnama  

**Link**: [PDF](https://arxiv.org/pdf/2508.05628)  

**Abstract**: Byte-level language models eliminate fragile tokenizers but face computational challenges in morphologically-rich languages (MRLs), where words span many bytes. We propose H-NET++, a hierarchical dynamic-chunking model that learns linguistically-informed segmentation through end-to-end training. Key innovations include: (1) a lightweight Transformer context-mixer (1.9M parameters) for cross-chunk attention, (2) a two-level latent hyper-prior for document-level consistency, (3) specialized handling of orthographic artifacts (e.g. Persian ZWNJ), and (4) curriculum-based training with staged sequence lengths. On a 1.4B-token Persian corpus, H-NET++ achieves state-of-the-art results: 0.159 BPB reduction versus BPE-based GPT-2-fa (12% better compression), 5.4pp gain on ParsGLUE, 53% improved robustness to ZWNJ corruption, and 73.8% F1 on gold morphological boundaries. Our learned chunks align with Persian morphology without explicit supervision, demonstrating that hierarchical dynamic chunking provides an effective tokenizer-free solution for MRLs while maintaining computational efficiency. 

**Abstract (ZH)**: 字级语言模型消除了脆弱的分词器但在富形态语言中面临计算挑战：H-NET++是一种分层动态分割模型，通过端到端训练学习语言学导向的分割 

---
# How Do LLMs Persuade? Linear Probes Can Uncover Persuasion Dynamics in Multi-Turn Conversations 

**Title (ZH)**: 如何进行说服？线性探针可以揭示多轮对话中的说服动态。 

**Authors**: Brandon Jaipersaud, David Krueger, Ekdeep Singh Lubana  

**Link**: [PDF](https://arxiv.org/pdf/2508.05625)  

**Abstract**: Large Language Models (LLMs) have started to demonstrate the ability to persuade humans, yet our understanding of how this dynamic transpires is limited. Recent work has used linear probes, lightweight tools for analyzing model representations, to study various LLM skills such as the ability to model user sentiment and political perspective. Motivated by this, we apply probes to study persuasion dynamics in natural, multi-turn conversations. We leverage insights from cognitive science to train probes on distinct aspects of persuasion: persuasion success, persuadee personality, and persuasion strategy. Despite their simplicity, we show that they capture various aspects of persuasion at both the sample and dataset levels. For instance, probes can identify the point in a conversation where the persuadee was persuaded or where persuasive success generally occurs across the entire dataset. We also show that in addition to being faster than expensive prompting-based approaches, probes can do just as well and even outperform prompting in some settings, such as when uncovering persuasion strategy. This suggests probes as a plausible avenue for studying other complex behaviours such as deception and manipulation, especially in multi-turn settings and large-scale dataset analysis where prompting-based methods would be computationally inefficient. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经开始展示出说服人类的能力，但我们对其背后动态的理解仍然有限。近期的研究使用了线性探针——一种轻量级的模型表示分析工具——探讨了各种LLM技能，如建模用户情感和政治观点的能力。受此启发，我们将探针应用于研究自然、多轮对话中的说服动态。我们借助认知科学的洞见，训练探针关注说服的多个方面：说服成功、说服对象的性格和说服策略。尽管探针简单，但我们表明它们在样本和数据集层面捕捉了说服的多种方面。例如，探针可以识别对话中说服对象被说服的点，或整个数据集中说服成功的普遍趋势。我们还展示了探针不仅比昂贵的提示基方法更快，而且在某些情况下，如揭示说服策略时，甚至可以和提示方法表现得一样好，甚至更好。这表明探针可能是研究其他复杂行为，如欺骗和操纵的一个有希望的途径，特别是在多轮对话和大规模数据集分析中，提示基方法可能会在计算效率方面显得不够高效。 

---
# Cooper: Co-Optimizing Policy and Reward Models in Reinforcement Learning for Large Language Models 

**Title (ZH)**: Cooper: 在强化学习大规模语言模型中协同优化策略和奖励模型 

**Authors**: Haitao Hong, Yuchen Yan, Xingyu Wu, Guiyang Hou, Wenqi Zhang, Weiming Lu, Yongliang Shen, Jun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.05613)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable performance in reasoning tasks, where reinforcement learning (RL) serves as a key algorithm for enhancing their reasoning capabilities. Currently, there are two mainstream reward paradigms: model-based rewards and rule-based rewards. However, both approaches suffer from limitations: rule-based rewards lack robustness, while model-based rewards are vulnerable to reward hacking. To address these issues, we propose Cooper(Co-optimizing Policy Model and Reward Model), a RL framework that jointly optimizes both the policy model and the reward model. Cooper leverages the high precision of rule-based rewards when identifying correct responses, and dynamically constructs and selects positive-negative sample pairs for continued training the reward model. This design enhances robustness and mitigates the risk of reward hacking. To further support Cooper, we introduce a hybrid annotation strategy that efficiently and accurately generates training data for the reward model. We also propose a reference-based reward modeling paradigm, where the reward model takes a reference answer as input. Based on this design, we train a reward model named VerifyRM, which achieves higher accuracy on VerifyBench compared to other models of the same size. We conduct reinforcement learning using both VerifyRM and Cooper. Our experiments show that Cooper not only alleviates reward hacking but also improves end-to-end RL performance, for instance, achieving a 0.54% gain in average accuracy on Qwen2.5-1.5B-Instruct. Our findings demonstrate that dynamically updating reward model is an effective way to combat reward hacking, providing a reference for better integrating reward models into RL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在推理任务中展现了显著性能，其中强化学习（RL）是提升其推理能力的关键算法。目前主要有两种主流奖励范式：基于模型的奖励和基于规则的奖励。然而，这两种方法都存在局限性：基于规则的奖励缺乏鲁棒性，而基于模型的奖励容易遭受奖励作弊。为解决这些问题，我们提出了Cooper（联合优化策略模型和奖励模型）框架，该框架联合优化策略模型和奖励模型。Cooper 利用基于规则奖励的高精度识别正确响应，并动态构建和选择正负样本对以继续训练奖励模型。这一设计增强了鲁棒性并减轻了奖励作弊的风险。为进一步支持Cooper，我们引入了一种混合注释策略，以高效准确地生成奖励模型的训练数据。我们还提出了一种基于参考的奖励建模范式，其中奖励模型以参考答案作为输入。基于此设计，我们训练了一个名为VerifyRM的奖励模型，其在VerifyBench上的准确率比其他大小相同模型更高。我们使用VerifyRM和Cooper进行强化学习。实验结果显示，Cooper 不仅缓解了奖励作弊，还提高了端到端的强化学习性能，例如在Qwen2.5-1.5B-Instruct上取得了0.54%的平均准确率提升。我们的研究证明，动态更新奖励模型是有效对抗奖励作弊的方法，为将奖励模型更好地集成到强化学习中提供了参考。 

---
# Shuffle-R1: Efficient RL framework for Multimodal Large Language Models via Data-centric Dynamic Shuffle 

**Title (ZH)**: Shuffle-R1:面向数据导向动态打乱的多模态大型语言模型高效 reinforcement 学习框架 

**Authors**: Linghao Zhu, Yiran Guan, Dingkang Liang, Jianzhong Ju, Zhenbo Luo, Bin Qin, Jian Luan, Yuliang Liu, Xiang Bai  

**Link**: [PDF](https://arxiv.org/pdf/2508.05612)  

**Abstract**: Reinforcement learning (RL) has emerged as an effective post-training paradigm for enhancing the reasoning capabilities of multimodal large language model (MLLM). However, current RL pipelines often suffer from training inefficiencies caused by two underexplored issues: Advantage Collapsing, where most advantages in a batch concentrate near zero, and Rollout Silencing, where the proportion of rollouts contributing non-zero gradients diminishes over time. These issues lead to suboptimal gradient updates and hinder long-term learning efficiency. To address these issues, we propose Shuffle-R1, a simple yet principled framework that improves RL fine-tuning efficiency by dynamically restructuring trajectory sampling and batch composition. It introduces (1) Pairwise Trajectory Sampling, which selects high-contrast trajectories with large advantages to improve gradient signal quality, and (2) Advantage-based Trajectory Shuffle, which increases exposure of valuable rollouts through informed batch reshuffling. Experiments across multiple reasoning benchmarks show that our framework consistently outperforms strong RL baselines with minimal overhead. These results highlight the importance of data-centric adaptations for more efficient RL training in MLLM. 

**Abstract (ZH)**: 强化学习（RL）已 emerges 作为增强多模态大语言模型（MLLM）推理能力的有效后训练范式。然而，当前的 RL 流水线常常由于两个未充分探索的问题而导致训练效率低下：优势坍缩，其中批次中的大多数优势接近零；以及采样消声，其中贡献非零梯度的采样比例随时间减少。这些问题导致梯度更新次优化，并阻碍长期学习效率。为解决这些问题，我们提出了一种简单的且有原则的 Shuffle-R1 框架，通过动态重构轨迹采样和批次组成来提高 RL 微调效率。该框架引入了（1）高对比度轨迹采样，选择具有大优势的轨迹以提高梯度信号质量，以及（2）基于优势的轨迹重排，通过有信息的批次重排增加有价值的采样的暴露度。在多个推理基准测试中的实验结果表明，该框架在最小的开销下始终优于强 RL 基准。这些结果强调了数据导向的适应性对 MLLM 更高效 RL 训练的重要性。 

---
# Iterative Learning of Computable Phenotypes for Treatment Resistant Hypertension using Large Language Models 

**Title (ZH)**: 使用大型语言模型迭代学习可计算表型治疗抵抗型高血压 

**Authors**: Guilherme Seidyo Imai Aldeia, Daniel S. Herman, William G. La Cava  

**Link**: [PDF](https://arxiv.org/pdf/2508.05581)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities for medical question answering and programming, but their potential for generating interpretable computable phenotypes (CPs) is under-explored. In this work, we investigate whether LLMs can generate accurate and concise CPs for six clinical phenotypes of varying complexity, which could be leveraged to enable scalable clinical decision support to improve care for patients with hypertension. In addition to evaluating zero-short performance, we propose and test a synthesize, execute, debug, instruct strategy that uses LLMs to generate and iteratively refine CPs using data-driven feedback. Our results show that LLMs, coupled with iterative learning, can generate interpretable and reasonably accurate programs that approach the performance of state-of-the-art ML methods while requiring significantly fewer training examples. 

**Abstract (ZH)**: 大型语言模型(LLMs)在医学问答和编程方面展示了显著的能力，但其生成可解析计算表型(CPs)的潜力尚未充分探索。在这项工作中，我们研究了LLMs能否生成六种不同复杂性的临床表型的准确且简洁的CPs，这些CPs可以被利用来提供可扩展的临床决策支持，以改善高血压患者的护理。除了评估零样本性能，我们还提出并测试了一种合成、执行、调试、指导策略，利用LLMs生成并在数据驱动的反馈下迭代细化CPs。我们的结果表明，结合迭代学习的LLMs能够生成可解析且相对准确的程序，其性能接近最先进的机器学习方法，同时所需训练示例大大减少。 

---
# Conformal Sets in Multiple-Choice Question Answering under Black-Box Settings with Provable Coverage Guarantees 

**Title (ZH)**: 多重选择题作答下的黑盒设置中的齐性集及其可证明覆盖保证 

**Authors**: Guang Yang, Xinyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05544)  

**Abstract**: Large Language Models (LLMs) have shown remarkable progress in multiple-choice question answering (MCQA), but their inherent unreliability, such as hallucination and overconfidence, limits their application in high-risk domains. To address this, we propose a frequency-based uncertainty quantification method under black-box settings, leveraging conformal prediction (CP) to ensure provable coverage guarantees. Our approach involves multiple independent samplings of the model's output distribution for each input, with the most frequent sample serving as a reference to calculate predictive entropy (PE). Experimental evaluations across six LLMs and four datasets (MedMCQA, MedQA, MMLU, MMLU-Pro) demonstrate that frequency-based PE outperforms logit-based PE in distinguishing between correct and incorrect predictions, as measured by AUROC. Furthermore, the method effectively controls the empirical miscoverage rate under user-specified risk levels, validating that sampling frequency can serve as a viable substitute for logit-based probabilities in black-box scenarios. This work provides a distribution-free model-agnostic framework for reliable uncertainty quantification in MCQA with guaranteed coverage, enhancing the trustworthiness of LLMs in practical applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多项选择题作答（MCQA）中取得了显著进展，但由于其固有的不可靠性，如虚构和过度自信，限制了其在高风险领域的应用。为了解决这一问题，我们提出了一种在黑盒设置下基于频率的不确定性量化方法，利用置信预测（CP）确保可证明的覆盖率保证。我们的方法包括为每个输入进行多次独立采样模型的输出分布，最常见的样本作为参考计算预测熵（PE）。在六种LLM和四种数据集（MedMCQA、MedQA、MMLU、MMLU-Pro）上的实验评估表明，基于频率的PE在区分正确和错误预测方面优于基于对数似然比的PE，通过AUROC衡量。此外，该方法有效地在用户指定的风险水平下控制经验覆盖率误差率，验证了在黑盒场景中采样频率可以作为对数似然比概率的可行替代方案。本工作提供了一种分布无关且模型无关的框架，用于MCQA中的可靠不确定性量化，确保覆盖率，从而增强LLMs在实际应用中的可信度。 

---
# The World According to LLMs: How Geographic Origin Influences LLMs' Entity Deduction Capabilities 

**Title (ZH)**: LLMs眼中的世界：地理起源如何影响LLMs的实体推断能力 

**Authors**: Harsh Nishant Lalai, Raj Sanjay Shah, Jiaxin Pei, Sashank Varma, Yi-Chia Wang, Ali Emami  

**Link**: [PDF](https://arxiv.org/pdf/2508.05525)  

**Abstract**: Large Language Models (LLMs) have been extensively tuned to mitigate explicit biases, yet they often exhibit subtle implicit biases rooted in their pre-training data. Rather than directly probing LLMs with human-crafted questions that may trigger guardrails, we propose studying how models behave when they proactively ask questions themselves. The 20 Questions game, a multi-turn deduction task, serves as an ideal testbed for this purpose. We systematically evaluate geographic performance disparities in entity deduction using a new dataset, Geo20Q+, consisting of both notable people and culturally significant objects (e.g., foods, landmarks, animals) from diverse regions. We test popular LLMs across two gameplay configurations (canonical 20-question and unlimited turns) and in seven languages (English, Hindi, Mandarin, Japanese, French, Spanish, and Turkish). Our results reveal geographic disparities: LLMs are substantially more successful at deducing entities from the Global North than the Global South, and the Global West than the Global East. While Wikipedia pageviews and pre-training corpus frequency correlate mildly with performance, they fail to fully explain these disparities. Notably, the language in which the game is played has minimal impact on performance gaps. These findings demonstrate the value of creative, free-form evaluation frameworks for uncovering subtle biases in LLMs that remain hidden in standard prompting setups. By analyzing how models initiate and pursue reasoning goals over multiple turns, we find geographic and cultural disparities embedded in their reasoning processes. We release the dataset (Geo20Q+) and code at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）已在消除显性偏见方面进行了广泛调优，但它们往往根植于预训练数据中的隐性偏见。我们提出通过让模型主动提问来研究其行为，而不是直接用由人类设计的问题触发防护机制。20个问题游戏，一个多轮推理任务，为此目的提供了一个理想的实验环境。我们使用新数据集Geo20Q+系统性地评估了地理表现差异在实体推理中的表现，该数据集包含来自不同地区的显著人物和文化重要对象（如食物、地标、动物）。我们测试了多种流行的LLMs在两种游戏配置（标准20个问题和不限轮次）下的表现，以及在七种语言（英语、印地语、普通话、日语、法语、西班牙语和土耳其语）下的表现。结果显示地理差异：LLMs在推理来自全球北方和西方的实体方面比全球南方和东方更为成功。尽管维基百科页面访问量和预训练语料库频率与表现存在轻微相关性，但它们无法完全解释这些差异。值得注意的是，游戏使用的语言对性能差异的影响微乎其微。这些发现证明了创造性、开放形式评估框架的价值，用于揭示在标准提示设置中隐藏的LLMs中的微妙偏见。通过分析模型在多轮次中如何启动和追求推理目标，我们发现其推理过程中嵌入了地理和文化差异。我们在此处发布数据集（Geo20Q+）和代码：[此链接]。 

---
# LAG: Logic-Augmented Generation from a Cartesian Perspective 

**Title (ZH)**: LAG: 从卡特尔视角增强逻辑生成 

**Authors**: Yilin Xiao, Chuang Zhou, Qinggang Zhang, Su Dong, Shengyuan Chen, Xiao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.05509)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks, yet exhibit critical limitations in knowledge-intensive tasks, often generating hallucinations when faced with questions requiring specialized expertise. While retrieval-augmented generation (RAG) mitigates this by integrating external knowledge, it struggles with complex reasoning scenarios due to its reliance on direct semantic retrieval and lack of structured logical organization. Inspired by Cartesian principles from \textit{Discours de la méthode}, this paper introduces Logic-Augmented Generation (LAG), a novel paradigm that reframes knowledge augmentation through systematic question decomposition and dependency-aware reasoning. Specifically, LAG first decomposes complex questions into atomic sub-questions ordered by logical dependencies. It then resolves these sequentially, using prior answers to guide context retrieval for subsequent sub-questions, ensuring stepwise grounding in logical chain. To prevent error propagation, LAG incorporates a logical termination mechanism that halts inference upon encountering unanswerable sub-questions and reduces wasted computation on excessive reasoning. Finally, it synthesizes all sub-resolutions to generate verified responses. Experiments on four benchmark datasets demonstrate that LAG significantly enhances reasoning robustness, reduces hallucination, and aligns LLM problem-solving with human cognition, offering a principled alternative to existing RAG systems. 

**Abstract (ZH)**: 基于笛卡尔原则的逻辑增强生成（LAG）：一种新型的知识增强范式 

---
# MoMA: A Mixture-of-Multimodal-Agents Architecture for Enhancing Clinical Prediction Modelling 

**Title (ZH)**: MoMA：增强临床预测建模的混合多模态代理架构 

**Authors**: Jifan Gao, Mahmudur Rahman, John Caskey, Madeline Oguss, Ann O'Rourke, Randy Brown, Anne Stey, Anoop Mayampurath, Matthew M. Churpek, Guanhua Chen, Majid Afshar  

**Link**: [PDF](https://arxiv.org/pdf/2508.05492)  

**Abstract**: Multimodal electronic health record (EHR) data provide richer, complementary insights into patient health compared to single-modality data. However, effectively integrating diverse data modalities for clinical prediction modeling remains challenging due to the substantial data requirements. We introduce a novel architecture, Mixture-of-Multimodal-Agents (MoMA), designed to leverage multiple large language model (LLM) agents for clinical prediction tasks using multimodal EHR data. MoMA employs specialized LLM agents ("specialist agents") to convert non-textual modalities, such as medical images and laboratory results, into structured textual summaries. These summaries, together with clinical notes, are combined by another LLM ("aggregator agent") to generate a unified multimodal summary, which is then used by a third LLM ("predictor agent") to produce clinical predictions. Evaluating MoMA on three prediction tasks using real-world datasets with different modality combinations and prediction settings, MoMA outperforms current state-of-the-art methods, highlighting its enhanced accuracy and flexibility across various tasks. 

**Abstract (ZH)**: 多模态电子健康记录（EHR）数据提供了相比单模态数据更为丰富和互补的患者健康洞察。然而，由于数据需求量巨大，有效整合多种数据模态进行临床预测模型构建仍然具有挑战性。我们介绍了一种新型架构，多模态代理混合模型（MoMA），该架构旨在利用多个大型语言模型（LLM）代理处理多模态EHR数据的临床预测任务。MoMA 使用专门的 LLM 代理（“专家代理”）将医疗图像和实验室结果等非文本模态转化为结构化的文本摘要。这些摘要与临床笔记一起，由另一个 LLM（“集成代理”）组合生成统一的多模态摘要，再由第三个 LLM（“预测代理”）生成临床预测。在使用不同模态组合和预测设置的真实世界数据集上对 MoMA 进行三项预测任务评估，MoMA 的性能优于当前最先进的方法，突显了其在各种任务中的增强准确性和灵活性。 

---
# Embedding Alignment in Code Generation for Audio 

**Title (ZH)**: 嵌入式对齐在音频代码生成中的应用 

**Authors**: Sam Kouteili, Hiren Madhu, George Typaldos, Mark Santolucito  

**Link**: [PDF](https://arxiv.org/pdf/2508.05473)  

**Abstract**: LLM-powered code generation has the potential to revolutionize creative coding endeavors, such as live-coding, by enabling users to focus on structural motifs over syntactic details. In such domains, when prompting an LLM, users may benefit from considering multiple varied code candidates to better realize their musical intentions. Code generation models, however, struggle to present unique and diverse code candidates, with no direct insight into the code's audio output. To better establish a relationship between code candidates and produced audio, we investigate the topology of the mapping between code and audio embedding spaces. We find that code and audio embeddings do not exhibit a simple linear relationship, but supplement this with a constructed predictive model that shows an embedding alignment map could be learned. Supplementing the aim for musically diverse output, we present a model that given code predicts output audio embedding, constructing a code-audio embedding alignment map. 

**Abstract (ZH)**: LLM 助力的代码生成有望通过使用户专注于结构 motif 而非语法细节来革新现场编码等创意编码领域。在这种领域中，当提示 LLM 时，用户可以通过考虑多种多样的代码候选方案来更好地实现他们的音乐意图。然而，代码生成模型在提供独特且多样的代码候选方案方面存在困难，并且无法直接洞察代码的音频输出。为了更好地建立代码候选方案与生成音频之间的关系，我们研究了代码空间和音频嵌入空间之间的拓扑结构。我们发现代码嵌入和音频嵌入之间并没有简单的线性关系，但通过构建预测模型显示可以学习到一种嵌入对齐图。除了追求音乐多样性输出的目标，我们提出了一个模型，给定代码可以预测输出音频嵌入，从而构建代码-音频嵌入对齐图。 

---
# MyCulture: Exploring Malaysia's Diverse Culture under Low-Resource Language Constraints 

**Title (ZH)**: MyCulture: 探索在低资源语言约束下的马来西亚多元文化 

**Authors**: Zhong Ken Hew, Jia Xin Low, Sze Jue Yang, Chee Seng chan  

**Link**: [PDF](https://arxiv.org/pdf/2508.05429)  

**Abstract**: Large Language Models (LLMs) often exhibit cultural biases due to training data dominated by high-resource languages like English and Chinese. This poses challenges for accurately representing and evaluating diverse cultural contexts, particularly in low-resource language settings. To address this, we introduce MyCulture, a benchmark designed to comprehensively evaluate LLMs on Malaysian culture across six pillars: arts, attire, customs, entertainment, food, and religion presented in Bahasa Melayu. Unlike conventional benchmarks, MyCulture employs a novel open-ended multiple-choice question format without predefined options, thereby reducing guessing and mitigating format bias. We provide a theoretical justification for the effectiveness of this open-ended structure in improving both fairness and discriminative power. Furthermore, we analyze structural bias by comparing model performance on structured versus free-form outputs, and assess language bias through multilingual prompt variations. Our evaluation across a range of regional and international LLMs reveals significant disparities in cultural comprehension, highlighting the urgent need for culturally grounded and linguistically inclusive benchmarks in the development and assessment of LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）由于训练数据主要由英语和汉语等高资源语言主导，往往表现出文化偏见。这给准确地代表和评估多样的文化背景，特别是在低资源语言环境中带来了挑战。为了解决这一问题，我们引入了MyCulture，这是一个旨在全面评估马来文化的大规模语言模型基准，涵盖艺术、服饰、习俗、娱乐、食物和宗教六个支柱，全部用马来语呈现。与传统基准不同，MyCulture采用了一种新颖的开放式多项选择题格式，没有预设选项，从而减少了猜测并在一定程度上减轻了格式偏见。我们为这种开放式结构的有效性提供了理论上的解释，以提高公平性和辨别力。此外，我们通过比较模型在结构化输出与自由形式输出上的性能来分析结构偏见，并通过多语言提示变体评估语言偏见。我们的评估涵盖了多个区域性和国际性的大型语言模型，揭示了文化理解上的显著差异，强调了在大型语言模型的发展和评估中迫切需要文化基础和语言包容性基准。 

---
# LLM-based Multi-Agent Copilot for Quantum Sensor 

**Title (ZH)**: 基于LLM的多代理协驾系统for量子传感器（标题翻译）：基于大规模语言模型的多代理协驾系统用于量子传感器 

**Authors**: Rong Sha, Binglin Wang, Jun Yang, Xiaoxiao Ma, Chengkun Wu, Liang Yan, Chao Zhou, Jixun Liu, Guochao Wang, Shuhua Yan, Lingxiao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05421)  

**Abstract**: Large language models (LLM) exhibit broad utility but face limitations in quantum sensor development, stemming from interdisciplinary knowledge barriers and involving complex optimization processes. Here we present QCopilot, an LLM-based multi-agent framework integrating external knowledge access, active learning, and uncertainty quantification for quantum sensor design and diagnosis. Comprising commercial LLMs with few-shot prompt engineering and vector knowledge base, QCopilot employs specialized agents to adaptively select optimization methods, automate modeling analysis, and independently perform problem diagnosis. Applying QCopilot to atom cooling experiments, we generated 10${}^{\rm{8}}$ sub-$\rm{\mu}$K atoms without any human intervention within a few hours, representing $\sim$100$\times$ speedup over manual experimentation. Notably, by continuously accumulating prior knowledge and enabling dynamic modeling, QCopilot can autonomously identify anomalous parameters in multi-parameter experimental settings. Our work reduces barriers to large-scale quantum sensor deployment and readily extends to other quantum information systems. 

**Abstract (ZH)**: 基于大型语言模型的多agents量子传感器设计与诊断框架：QCopilot 

---
# Echo: Decoupling Inference and Training for Large-Scale RL Alignment on Heterogeneous Swarms 

**Title (ZH)**: Echo: 分离推理与训练以在异构群体中实现大规模RL对齐 

**Authors**: Jie Xiao, Shaoduo Gan, Changyuan Fan, Qingnan Ren, Alfred Long, Yuchen Zhang, Rymon Yu, Eric Yang, Lynn Ai  

**Link**: [PDF](https://arxiv.org/pdf/2508.05387)  

**Abstract**: Modern RL-based post-training for large language models (LLMs) co-locate trajectory sampling and policy optimisation on the same GPU cluster, forcing the system to switch between inference and training workloads. This serial context switching violates the single-program-multiple-data (SPMD) assumption underlying today's distributed training systems. We present Echo, the RL system that cleanly decouples these two phases across heterogeneous "inference" and "training" swarms while preserving statistical efficiency. Echo introduces two lightweight synchronization protocols: a sequential pull mode that refreshes sampler weights on every API call for minimal bias, and an asynchronous push-pull mode that streams version-tagged rollouts through a replay buffer to maximise hardware utilisation. Training three representative RL workloads with Qwen3-4B, Qwen2.5-7B and Qwen3-32B on a geographically distributed cluster, Echo matches a fully co-located Verl baseline in convergence speed and final reward while off-loading trajectory generation to commodity edge hardware. These promising results demonstrate that large-scale RL for LLMs could achieve datacentre-grade performance using decentralised, heterogeneous resources. 

**Abstract (ZH)**: 现代基于RL的后训练方法在大型语言模型（LLMs）中将轨迹采样和策略优化 colocate 在同一GPU集群上，迫使系统在推理和训练工作负载之间进行串行切换。这种串行上下文切换违反了当前分布式训练系统 underlying 的单程序多数据（SPMD）假设。我们提出了Echo，这是一种清洁地将这两个阶段 decouple 到异构的“推理”和“训练”集群中同时保持统计效率的RL系统。Echo 引入了两种轻量级同步协议：一种是顺序拉取模式，在每次API调用时刷新采样器权重以最小化偏差，另一种是异步推拉模式，通过回放缓冲区流式传输版本标记的轨迹以最大化硬件利用率。使用Qwen3-4B、Qwen2.5-7B和Qwen3-32B在地理上分布的集群上训练三个代表性RL工作负载，Echo 在收敛速度和最终奖励方面达到了与完全 colocate 的Verl基线相同的效果，同时将轨迹生成卸载到通用边缘硬件。这些有前途的结果表明，大型RL for LLMs 可以通过使用去中心化的异构资源实现数据中心级性能。 

---
# Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression 

**Title (ZH)**: 大型推理语言模型的高效推理通过不确定性引导的反射抑制实现 

**Authors**: Jiameng Huang, Baijiong Lin, Guhao Feng, Jierun Chen, Di He, Lu Hou  

**Link**: [PDF](https://arxiv.org/pdf/2508.05337)  

**Abstract**: Recent Large Reasoning Language Models (LRLMs) employ long chain-of-thought reasoning with complex reflection behaviors, typically signaled by specific trigger words (e.g., "Wait" and "Alternatively") to enhance performance. However, these reflection behaviors can lead to the overthinking problem where the generation of redundant reasoning steps that unnecessarily increase token usage, raise inference costs, and reduce practical utility. In this paper, we propose Certainty-Guided Reflection Suppression (CGRS), a novel method that mitigates overthinking in LRLMs while maintaining reasoning accuracy. CGRS operates by dynamically suppressing the model's generation of reflection triggers when it exhibits high confidence in its current response, thereby preventing redundant reflection cycles without compromising output quality. Our approach is model-agnostic, requires no retraining or architectural modifications, and can be integrated seamlessly with existing autoregressive generation pipelines. Extensive experiments across four reasoning benchmarks (i.e., AIME24, AMC23, MATH500, and GPQA-D) demonstrate CGRS's effectiveness: it reduces token usage by an average of 18.5% to 41.9% while preserving accuracy. It also achieves the optimal balance between length reduction and performance compared to state-of-the-art baselines. These results hold consistently across model architectures (e.g., DeepSeek-R1-Distill series, QwQ-32B, and Qwen3 family) and scales (4B to 32B parameters), highlighting CGRS's practical value for efficient reasoning. 

**Abstract (ZH)**: Recent Large Reasoning Language Models中的确定性指导反射抑制（CGRS）：减轻过度推理的同时保持推理准确性 

---
# mKG-RAG: Multimodal Knowledge Graph-Enhanced RAG for Visual Question Answering 

**Title (ZH)**: 多模态知识图谱增强的RAG视觉问答模型 

**Authors**: Xu Yuan, Liangbo Ning, Wenqi Fan, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.05318)  

**Abstract**: Recently, Retrieval-Augmented Generation (RAG) has been proposed to expand internal knowledge of Multimodal Large Language Models (MLLMs) by incorporating external knowledge databases into the generation process, which is widely used for knowledge-based Visual Question Answering (VQA) tasks. Despite impressive advancements, vanilla RAG-based VQA methods that rely on unstructured documents and overlook the structural relationships among knowledge elements frequently introduce irrelevant or misleading content, reducing answer accuracy and reliability. To overcome these challenges, a promising solution is to integrate multimodal knowledge graphs (KGs) into RAG-based VQA frameworks to enhance the generation by introducing structured multimodal knowledge. Therefore, in this paper, we propose a novel multimodal knowledge-augmented generation framework (mKG-RAG) based on multimodal KGs for knowledge-intensive VQA tasks. Specifically, our approach leverages MLLM-powered keyword extraction and vision-text matching to distill semantically consistent and modality-aligned entities/relationships from multimodal documents, constructing high-quality multimodal KGs as structured knowledge representations. In addition, a dual-stage retrieval strategy equipped with a question-aware multimodal retriever is introduced to improve retrieval efficiency while refining precision. Comprehensive experiments demonstrate that our approach significantly outperforms existing methods, setting a new state-of-the-art for knowledge-based VQA. 

**Abstract (ZH)**: 最近，检索增强生成（RAG）被提出以通过集成外部知识数据库来扩展多模态大型语言模型（MLLMs）的内部知识，广泛应用于基于知识的视觉问答（VQA）任务。尽管取得了显著的进步，但依赖于无结构文档且忽视知识元素之间结构关系的原始RAG基VQA方法经常引入无关或误导性的内容，降低答案的准确性和可靠性。为克服这些挑战，将多模态知识图谱（KGs）集成到RAG基VQA框架中以增强生成，通过引入结构化的多模态知识来提升生成效果是一种有前景的解决方案。因此，在本文中，我们提出了一种基于多模态KG的知识增强生成框架（mKG-RAG），用于知识密集型VQA任务。具体而言，我们的方法利用MLLM支持的关键词提取和视觉-文本匹配来提炼语义一致且模态对齐的实体/关系，构建高质量的多模态KG作为结构化的知识表示。此外，我们引入了一种双阶段检索策略，配备有问题感知的多模态检索器，以提高检索效率并优化精确度。全面的实验显示，我们的方法显著优于现有方法，在基于知识的VQA任务上达到了新的state-of-the-art。 

---
# VS-LLM: Visual-Semantic Depression Assessment based on LLM for Drawing Projection Test 

**Title (ZH)**: VS-LLM：基于大语言模型的视觉- 语义抑郁评估方法及其在绘画投射测试中的应用 

**Authors**: Meiqi Wu, Yaxuan Kang, Xuchen Li, Shiyu Hu, Xiaotang Chen, Yunfeng Kang, Weiqiang Wang, Kaiqi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.05299)  

**Abstract**: The Drawing Projection Test (DPT) is an essential tool in art therapy, allowing psychologists to assess participants' mental states through their sketches. Specifically, through sketches with the theme of "a person picking an apple from a tree (PPAT)", it can be revealed whether the participants are in mental states such as depression. Compared with scales, the DPT can enrich psychologists' understanding of an individual's mental state. However, the interpretation of the PPAT is laborious and depends on the experience of the psychologists. To address this issue, we propose an effective identification method to support psychologists in conducting a large-scale automatic DPT. Unlike traditional sketch recognition, DPT more focus on the overall evaluation of the sketches, such as color usage and space utilization. Moreover, PPAT imposes a time limit and prohibits verbal reminders, resulting in low drawing accuracy and a lack of detailed depiction. To address these challenges, we propose the following efforts: (1) Providing an experimental environment for automated analysis of PPAT sketches for depression assessment; (2) Offering a Visual-Semantic depression assessment based on LLM (VS-LLM) method; (3) Experimental results demonstrate that our method improves by 17.6% compared to the psychologist assessment method. We anticipate that this work will contribute to the research in mental state assessment based on PPAT sketches' elements recognition. Our datasets and codes are available at this https URL. 

**Abstract (ZH)**: 绘画投射测试（DPT）是艺术疗法中的一个关键工具，通过参与者绘制的草图评估其心理状态。具体而言，通过以“从树上摘苹果的人（PPAT）”为主题的草图，可以揭示参与者是否处于抑郁等心理状态。与量表相比，DPT能够丰富心理学家对个体心理状态的理解。然而，PPAT的解读工作量大且依赖于心理学家的经验。为了解决这一问题，我们提出了一种有效的方法来支持心理学家进行大规模自动DPT。与传统的草图识别不同，DPT更侧重于对草图的整体评估，如颜色使用和空间利用。此外，PPAT设有时间限制并且禁止口头提示，导致绘画准确性较低且缺乏细节表现。为应对这些挑战，我们提出以下努力：（1）提供一个用于抑郁评估的PPAT草图自动化分析实验环境；（2）提供基于LLM的视觉-语义抑郁评估方法（VS-LLM）；（3）实验结果表明，我们的方法在抑郁评估中比心理学家评估方法提高了17.6%。预计本工作将为基于PPAT草图元素识别的心理状态评估研究做出贡献。我们的数据集和代码可通过以下链接获取。 

---
# Pruning Large Language Models by Identifying and Preserving Functional Networks 

**Title (ZH)**: 通过识别和保留功能性网络精简大型语言模型 

**Authors**: Yiheng Liu, Junhao Ning, Sichen Xia, Xiaohui Gao, Ning Qiang, Bao Ge, Junwei Han, Xintao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05239)  

**Abstract**: Structured pruning is one of the representative techniques for compressing large language models (LLMs) to reduce GPU memory consumption and accelerate inference speed. It offers significant practical value in improving the efficiency of LLMs in real-world applications. Current structured pruning methods typically rely on assessment of the importance of the structure units and pruning the units with less importance. Most of them overlooks the interaction and collaboration among artificial neurons that are crucial for the functionalities of LLMs, leading to a disruption in the macro functional architecture of LLMs and consequently a pruning performance degradation. Inspired by the inherent similarities between artificial neural networks and functional neural networks in the human brain, we alleviate this challenge and propose to prune LLMs by identifying and preserving functional networks within LLMs in this study. To achieve this, we treat an LLM as a digital brain and decompose the LLM into functional networks, analogous to identifying functional brain networks in neuroimaging data. Afterwards, an LLM is pruned by preserving the key neurons within these functional networks. Experimental results demonstrate that the proposed method can successfully identify and locate functional networks and key neurons in LLMs, enabling efficient model pruning. Our code is available at this https URL. 

**Abstract (ZH)**: 结构化剪枝是压缩大型语言模型（LLMs）以减少GPU内存消耗和加速推理速度的一种代表性技术。它在提高LLMs在实际应用中的效率方面具有重要的实用价值。当前的结构化剪枝方法通常依赖于对结构单元重要性的评估，并剪枝那些重要性较低的单元。大多数方法忽略了对LLMs功能至关重要的人工神经元之间的交互和协作，这会导致LLMs宏观功能架构的破坏，从而降低剪枝性能。受人工神经网络与人脑功能神经网络内在相似性的启发，我们缓解了这一挑战，并在此研究中提出通过识别和保留LLMs内的功能网络来进行LLMs的剪枝。为此，我们将LLM视为数字大脑，将其分解为功能网络，类似于在神经影像数据中识别功能脑网络。随后，通过保留这些功能网络内的关键神经元来剪枝LLM。实验结果表明，所提出的方法可以成功识别和定位LLMs内的功能网络和关键神经元，从而实现高效的模型剪枝。我们的代码可在以下网址获取。 

---
# Driver Assistant: Persuading Drivers to Adjust Secondary Tasks Using Large Language Models 

**Title (ZH)**: 驾驶员助手：使用大型语言模型说服驾驶员调整次要任务 

**Authors**: Wei Xiang, Muchen Li, Jie Yan, Manling Zheng, Hanfei Zhu, Mengyun Jiang, Lingyun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.05238)  

**Abstract**: Level 3 automated driving systems allows drivers to engage in secondary tasks while diminishing their perception of risk. In the event of an emergency necessitating driver intervention, the system will alert the driver with a limited window for reaction and imposing a substantial cognitive burden. To address this challenge, this study employs a Large Language Model (LLM) to assist drivers in maintaining an appropriate attention on road conditions through a "humanized" persuasive advice. Our tool leverages the road conditions encountered by Level 3 systems as triggers, proactively steering driver behavior via both visual and auditory routes. Empirical study indicates that our tool is effective in sustaining driver attention with reduced cognitive load and coordinating secondary tasks with takeover behavior. Our work provides insights into the potential of using LLMs to support drivers during multi-task automated driving. 

**Abstract (ZH)**: Level 3 辅助驾驶系统允许驾驶员在降低感知风险的情况下进行次要任务。在需要驾驶员干预的紧急情况下，系统会通过有限的反应窗口提醒驾驶员，并施加较大的认知负担。为应对这一挑战，本研究利用大型语言模型（LLM）通过“人性化”的劝说建议帮助驾驶员维持对道路状况的适当关注。该工具将Level 3系统遇到的道路情况作为触发器，通过视觉和听觉途径主动引导驾驶员行为。实证研究表明，该工具在减少认知负荷的同时有效维持了驾驶员的注意力，并协调了次要任务与接管行为。本研究为利用LLM在多任务辅助驾驶中支持驾驶员的潜力提供了见解。 

---
# Resource-Limited Joint Multimodal Sentiment Reasoning and Classification via Chain-of-Thought Enhancement and Distillation 

**Title (ZH)**: 资源受限的联模式情感推理与分类 via 增强思维链与蒸馏 

**Authors**: Haonan Shangguan, Xiaocui Yang, Shi Feng, Daling Wang, Yifei Zhang, Ge Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05234)  

**Abstract**: The surge in rich multimodal content on social media platforms has greatly advanced Multimodal Sentiment Analysis (MSA), with Large Language Models (LLMs) further accelerating progress in this field. Current approaches primarily leverage the knowledge and reasoning capabilities of parameter-heavy (Multimodal) LLMs for sentiment classification, overlooking autonomous multimodal sentiment reasoning generation in resource-constrained environments. Therefore, we focus on the Resource-Limited Joint Multimodal Sentiment Reasoning and Classification task, JMSRC, which simultaneously performs multimodal sentiment reasoning chain generation and sentiment classification only with a lightweight model. We propose a Multimodal Chain-of-Thought Reasoning Distillation model, MulCoT-RD, designed for JMSRC that employs a "Teacher-Assistant-Student" distillation paradigm to address deployment constraints in resource-limited environments. We first leverage a high-performance Multimodal Large Language Model (MLLM) to generate the initial reasoning dataset and train a medium-sized assistant model with a multi-task learning mechanism. A lightweight student model is jointly trained to perform efficient multimodal sentiment reasoning generation and classification. Extensive experiments on four datasets demonstrate that MulCoT-RD with only 3B parameters achieves strong performance on JMSRC, while exhibiting robust generalization and enhanced interpretability. 

**Abstract (ZH)**: 社交媒体平台上的丰富多模态内容 surged极大地推动了多模态情感分析（MSA）的发展，大规模语言模型（LLMs）进一步加快了这一领域的进展。当前的方法主要依赖于参数密集型（多模态）LLMs的知识和推理能力进行情感分类，忽视了在资源受限环境中自主多模态情感推理生成的能力。因此，我们集中于资源受限环境下联合的多模态情感推理和分类任务（JMSRC），仅使用轻量级模型执行多模态情感推理链生成和情感分类。我们提出了一种用于JMSRC的多模态思维链推理蒸馏模型（MulCoT-RD），该模型采用“教师-助理-学生”蒸馏范式以应对资源受限环境下的部署约束。我们利用高性能的多模态大规模语言模型（MLLM）生成初始推理数据集，并使用多任务学习机制训练一个中型助理模型。同时联合训练一个轻量级学生模型以高效执行多模态情感推理生成和分类。在四个数据集上的广泛实验表明，仅含有3B参数的MulCoT-RD在JMSRC任务上表现出色，同时具有较强的泛化能力和增强的可解释性。 

---
# FAITH: A Framework for Assessing Intrinsic Tabular Hallucinations in finance 

**Title (ZH)**: FAITH: 评估金融领域内在表格幻觉的框架 

**Authors**: Mengao Zhang, Jiayu Fu, Tanya Warrier, Yuwen Wang, Tianhui Tan, Ke-wei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.05201)  

**Abstract**: Hallucination remains a critical challenge for deploying Large Language Models (LLMs) in finance. Accurate extraction and precise calculation from tabular data are essential for reliable financial analysis, since even minor numerical errors can undermine decision-making and regulatory compliance. Financial applications have unique requirements, often relying on context-dependent, numerical, and proprietary tabular data that existing hallucination benchmarks rarely capture. In this study, we develop a rigorous and scalable framework for evaluating intrinsic hallucinations in financial LLMs, conceptualized as a context-aware masked span prediction task over real-world financial documents. Our main contributions are: (1) a novel, automated dataset creation paradigm using a masking strategy; (2) a new hallucination evaluation dataset derived from S&P 500 annual reports; and (3) a comprehensive evaluation of intrinsic hallucination patterns in state-of-the-art LLMs on financial tabular data. Our work provides a robust methodology for in-house LLM evaluation and serves as a critical step toward building more trustworthy and reliable financial Generative AI systems. 

**Abstract (ZH)**: Hallucination仍然是在金融领域部署大型语言模型（LLMs）的关键挑战。准确提取和精确计算表格数据是实现可靠财务分析的关键，因为即使是轻微的数值错误也可能影响决策和监管合规性。金融应用有独特的诉求，通常依赖于上下文相关、数值型和专有的表格数据，而现有的幻觉基准很少涵盖这些需求。在本研究中，我们构建了一个严格且可扩展的框架，以评估金融LLMs中的固有幻觉，这一框架被概念化为一种基于真实世界金融文件的上下文感知隐藏跨度预测任务。我们的主要贡献包括：（1）一种新的自动化数据集创建范式，使用遮蔽策略；（2）一个源自标普500年度报告的新幻觉评估数据集；以及（3）对最先进的LLMs在财务表格数据上的固有幻觉模式进行全面评估。我们的工作提供了一种稳健的内部LLM评估方法，并为构建更可信赖和可靠的金融生成AI系统奠定了关键步骤。 

---
# Incident Response Planning Using a Lightweight Large Language Model with Reduced Hallucination 

**Title (ZH)**: 使用减轻幻觉的轻量级大型语言模型进行事件响应规划 

**Authors**: Kim Hammar, Tansu Alpcan, Emil C. Lupu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05188)  

**Abstract**: Timely and effective incident response is key to managing the growing frequency of cyberattacks. However, identifying the right response actions for complex systems is a major technical challenge. A promising approach to mitigate this challenge is to use the security knowledge embedded in large language models (LLMs) to assist security operators during incident handling. Recent research has demonstrated the potential of this approach, but current methods are mainly based on prompt engineering of frontier LLMs, which is costly and prone to hallucinations. We address these limitations by presenting a novel way to use an LLM for incident response planning with reduced hallucination. Our method includes three steps: fine-tuning, information retrieval, and lookahead planning. We prove that our method generates response plans with a bounded probability of hallucination and that this probability can be made arbitrarily small at the expense of increased planning time under certain assumptions. Moreover, we show that our method is lightweight and can run on commodity hardware. We evaluate our method on logs from incidents reported in the literature. The experimental results show that our method a) achieves up to 22% shorter recovery times than frontier LLMs and b) generalizes to a broad range of incident types and response actions. 

**Abstract (ZH)**: 及时有效的应急响应是管理日益频繁的网络攻击的关键。然而，为复杂系统识别正确的应急响应行动是一项重大的技术挑战。一种有望缓解这一挑战的方法是利用大型语言模型（LLM）中嵌入的安全知识，在应急处理过程中辅助安全操作员。近期的研究证明了这一方法的潜力，但当前的方法主要基于前沿LLM的提示工程，这耗时且容易产生错误。我们通过提出一种新颖的方法来解决这些限制，该方法能够在减少错误的同时利用LLM进行应急响应规划。我们的方法包括三个步骤：微调、信息检索和前瞻规划。我们证明，我们的方法生成的响应计划在满足某些假设的情况下具有界限内的幻觉概率，并且通过增加规划时间，这一概率可以被任意减小。此外，我们展示了我们的方法具有轻量级的特性，可以在通用硬件上运行。我们在文献中报告的事故日志上评估了我们的方法。实验结果表明，我们的方法a) 将恢复时间缩短了高达22%，b) 能够适应广泛类型的事故和响应行动。 

---
# Posterior-GRPO: Rewarding Reasoning Processes in Code Generation 

**Title (ZH)**: 后验-GRPO：奖励编码生成中的推理过程 

**Authors**: Lishui Fan, Yu Zhang, Mouxiang Chen, Zhongxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05170)  

**Abstract**: Reinforcement learning (RL) has significantly advanced code generation for large language models (LLMs). However, current paradigms rely on outcome-based rewards from test cases, neglecting the quality of the intermediate reasoning process. While supervising the reasoning process directly is a promising direction, it is highly susceptible to reward hacking, where the policy model learns to exploit the reasoning reward signal without improving final outcomes. To address this, we introduce a unified framework that can effectively incorporate the quality of the reasoning process during RL. First, to enable reasoning evaluation, we develop LCB-RB, a benchmark comprising preference pairs of superior and inferior reasoning processes. Second, to accurately score reasoning quality, we introduce an Optimized-Degraded based (OD-based) method for reward model training. This method generates high-quality preference pairs by systematically optimizing and degrading initial reasoning paths along curated dimensions of reasoning quality, such as factual accuracy, logical rigor, and coherence. A 7B parameter reward model with this method achieves state-of-the-art (SOTA) performance on LCB-RB and generalizes well to other benchmarks. Finally, we introduce Posterior-GRPO (P-GRPO), a novel RL method that conditions process-based rewards on task success. By selectively applying rewards to the reasoning processes of only successful outcomes, P-GRPO effectively mitigates reward hacking and aligns the model's internal reasoning with final code correctness. A 7B parameter model with P-GRPO achieves superior performance across diverse code generation tasks, outperforming outcome-only baselines by 4.5%, achieving comparable performance to GPT-4-Turbo. We further demonstrate the generalizability of our approach by extending it to mathematical tasks. Our models, dataset, and code are publicly available. 

**Abstract (ZH)**: 强化学习（RL）显著推进了大型语言模型（LLMs）的代码生成。然而，当前方法依赖于测试案例的结果奖励，忽视了中间推理过程的质量。虽然直接监督推理过程是一个有前景的方向，但极易导致奖励欺骗，即策略模型学会利用推理奖励信号而不提高最终结果。为解决这一问题，我们提出了一种统一框架，能够在RL过程中有效融入推理过程的质量。首先，为了实现推理评估，我们开发了LCB-RB基准，其包含高级和较差推理过程的偏好对。其次，为了准确评分推理质量，我们引入了一种基于优化-退化（OD）的方法进行奖励模型训练。该方法通过系统地优化和退化初始推理路径，沿特定的推理质量维度生成高质量的偏好对，如事实准确性、逻辑严密性和连贯性。使用该方法训练的7B参数奖励模型在LCB-RB上达到了最优性能，并在其他基准上表现出良好的泛化能力。最后，我们提出了基于过程奖励的后验GRPO（P-GRPO）新方法，该方法将过程奖励条件化于任务成功。通过仅对成功结果的推理过程应用奖励，P-GRPO有效减轻了奖励欺骗，并使模型的内部推理与最终代码正确性对齐。使用P-GRPO的7B参数模型在多种代码生成任务上表现出优越性能，优于仅基于结果的基线4.5%，并实现了与GPT-4-Turbo相当的性能。我们进一步展示了该方法的泛化能力，将其扩展到数学任务。我们的模型、数据集和代码已公开。 

---
# Aligning LLMs on a Budget: Inference-Time Alignment with Heuristic Reward Models 

**Title (ZH)**: 预算内的LLM对齐：基于启发式奖励模型的推理时对齐 

**Authors**: Mason Nakamura, Saaduddin Mahmud, Kyle H. Wray, Hamed Zamani, Shlomo Zilberstein  

**Link**: [PDF](https://arxiv.org/pdf/2508.05165)  

**Abstract**: Aligning LLMs with user preferences is crucial for real-world use but often requires costly fine-tuning or expensive inference, forcing trade-offs between alignment quality and computational cost. Existing inference-time methods typically ignore this balance, focusing solely on the optimized policy's performance. We propose HIA (Heuristic-Guided Inference-time Alignment), a tuning-free, black-box-compatible approach that uses a lightweight prompt optimizer, heuristic reward models, and two-stage filtering to reduce inference calls while preserving alignment quality. On real-world prompt datasets, HelpSteer and ComPRed, HIA outperforms best-of-N sampling, beam search, and greedy search baselines in multi-objective, goal-conditioned tasks under the same inference budget. We also find that HIA is effective under low-inference budgets with as little as one or two response queries, offering a practical solution for scalable, personalized LLM deployment. 

**Abstract (ZH)**: 对齐LL
user
把下面的论文标题翻译成中文，
 JuventusAlign: He-Guided Inference-Time Alignment for Personalized LLM Deployment 

---
# Tool Graph Retriever: Exploring Dependency Graph-based Tool Retrieval for Large Language Models 

**Title (ZH)**: 工具图检索器：基于依赖图的大型语言模型工具检索探索 

**Authors**: Linfeng Gao, Yaoxiang Wang, Minlong Peng, Jialong Tang, Yuzhe Shang, Mingming Sun, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2508.05152)  

**Abstract**: With the remarkable advancement of AI agents, the number of their equipped tools is increasing rapidly. However, integrating all tool information into the limited model context becomes impractical, highlighting the need for efficient tool retrieval methods. In this regard, dominant methods primarily rely on semantic similarities between tool descriptions and user queries to retrieve relevant tools. However, they often consider each tool independently, overlooking dependencies between tools, which may lead to the omission of prerequisite tools for successful task execution. To deal with this defect, in this paper, we propose Tool Graph Retriever (TGR), which exploits the dependencies among tools to learn better tool representations for retrieval. First, we construct a dataset termed TDI300K to train a discriminator for identifying tool dependencies. Then, we represent all candidate tools as a tool dependency graph and use graph convolution to integrate the dependencies into their representations. Finally, these updated tool representations are employed for online retrieval. Experimental results on several commonly used datasets show that our TGR can bring a performance improvement to existing dominant methods, achieving SOTA performance. Moreover, in-depth analyses also verify the importance of tool dependencies and the effectiveness of our TGR. 

**Abstract (ZH)**: 随着AI代理的显著进步，工具的数量不断增加。由于集成这些工具变得 impractical， 因此突显了高效工具检索的必要性。就此而言，主导方法主要主要主要主要依赖于工具描述和查询之间的语义相似性进行检索。然而，它们往往 each 工具独立进行检索，忽略了工具之间的依赖关系，这可能导致缺失必要的工具从而影响任务执行的成功。为解决这一缺陷，我们提出了 Tool Dependency Retention（TGR），它利用工具之间的依赖关系来改进检索。首先，我们构建构建构建了一个名为 Tool Dependency 3K（T-D3K）的数据集来训练一个鉴别器以识别工具之间的依赖关系。然后我们用候选工具表示一个工具依赖图，并利用图卷积来提取这些依赖关系。最后，，我们在更新后的构建的表达性中使用图卷积进行了在线检索。实验结果表明，相较于现有主导方法 TGR 可以在达到S-OTA（当前最佳水平）表现上提供改进。深入分析也证实了工具依赖关系的重要性以及我们的 TGR 的有效性。 

---
# Speech LLMs in Low-Resource Scenarios: Data Volume Requirements and the Impact of Pretraining on High-Resource Languages 

**Title (ZH)**: 低资源场景中的语音语言模型：数据量要求及预训练对高资源语言的影响 

**Authors**: Seraphina Fong, Marco Matassoni, Alessio Brutti  

**Link**: [PDF](https://arxiv.org/pdf/2508.05149)  

**Abstract**: Large language models (LLMs) have demonstrated potential in handling spoken inputs for high-resource languages, reaching state-of-the-art performance in various tasks. However, their applicability is still less explored in low-resource settings. This work investigates the use of Speech LLMs for low-resource Automatic Speech Recognition using the SLAM-ASR framework, where a trainable lightweight projector connects a speech encoder and a LLM. Firstly, we assess training data volume requirements to match Whisper-only performance, re-emphasizing the challenges of limited data. Secondly, we show that leveraging mono- or multilingual projectors pretrained on high-resource languages reduces the impact of data scarcity, especially with small training sets. Using multilingual LLMs (EuroLLM, Salamandra) with whisper-large-v3-turbo, we evaluate performance on several public benchmarks, providing insights for future research on optimizing Speech LLMs for low-resource languages and multilinguality. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经在处理高资源语言的口语输入方面展示了潜力，并在各种任务中达到了最先进的性能。然而，在低资源环境中其应用性 still less explored。本文探讨了使用 SLAM-ASR 框架中的语音 LLM 进行低资源自动语音识别的方法，其中可训练的轻量级投影器连接语音编码器和 LLM。首先，我们评估了训练数据量的需求，以匹配 Whisper 的性能，重新强调了数据短缺的挑战。其次，我们表明，利用在高资源语言上预训练的单语或多语投影器可以减轻数据稀缺的影响，尤其是在使用小规模训练集时。使用多语言 LLM（EuroLLM、Salamandra）与 whisper-large-v3-turbo 结合，我们在多个公开基准上评估性能，为未来研究优化适用于低资源语言和多语言性的语音 LLM 提供了见解。 

---
# Attention Basin: Why Contextual Position Matters in Large Language Models 

**Title (ZH)**: 注意力盆地：上下文位置为何在大规模语言模型中 Matters 

**Authors**: Zihao Yi, Delong Zeng, Zhenqing Ling, Haohao Luo, Zhe Xu, Wei Liu, Jian Luan, Wanxia Cao, Ying Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.05128)  

**Abstract**: The performance of Large Language Models (LLMs) is significantly sensitive to the contextual position of information in the input. To investigate the mechanism behind this positional bias, our extensive experiments reveal a consistent phenomenon we term the attention basin: when presented with a sequence of structured items (e.g., retrieved documents or few-shot examples), models systematically assign higher attention to the items at the beginning and end of the sequence, while neglecting those in the middle. Crucially, our analysis further reveals that allocating higher attention to critical information is key to enhancing model performance. Based on these insights, we introduce Attention-Driven Reranking (AttnRank), a two-stage framework that (i) estimates a model's intrinsic positional attention preferences using a small calibration set, and (ii) reorders retrieved documents or few-shot examples to align the most salient content with these high-attention positions. AttnRank is a model-agnostic, training-free, and plug-and-play method with minimal computational overhead. Experiments on multi-hop QA and few-shot in-context learning tasks demonstrate that AttnRank achieves substantial improvements across 10 large language models of varying architectures and scales, without modifying model parameters or training procedures. 

**Abstract (ZH)**: 大型语言模型的表现对其输入中信息的上下文位置高度敏感。为了探究这种位置偏见的机制，我们的 extensive 实验揭示了一个一致的现象，我们称之为注意力盆地：当提供一系列结构化项（如检索文档或少次示例）时，模型系统地对序列开头和结尾的项赋予更高的注意力，而忽视中间的项。关键的是，我们的分析进一步揭示了将更高注意力分配给关键信息是提升模型性能的关键。基于这些洞察，我们提出了基于注意力驱动的重排序（AttnRank）框架，该框架分为两个阶段：（i）使用一个小的校准集估计模型固有的位置注意力偏好，（ii）重新排序检索到的文档或少次示例，使其最具显著性的内容与这些高注意力位置对齐。AttnRank 是一种模型无关、无需训练、插即用的方法，具有最小的计算开销。实验表明，在多跳问答和少次上下文学习任务中，AttnRank 在 10 个不同架构和规模的大语言模型上实现了显著改进，而无需修改模型参数或训练流程。 

---
# Exploring Superior Function Calls via Reinforcement Learning 

**Title (ZH)**: 通过强化学习探索优越函数调用 

**Authors**: Bingguang Hao, Maolin Wang, Zengzhuang Xu, Yicheng Chen, Cunyin Peng, Jinjie GU, Chenyi Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2508.05118)  

**Abstract**: Function calling capabilities are crucial for deploying Large Language Models in real-world applications, yet current training approaches fail to develop robust reasoning strategies. Supervised fine-tuning produces models that rely on superficial pattern matching, while standard reinforcement learning methods struggle with the complex action space of structured function calls. We present a novel reinforcement learning framework designed to enhance group relative policy optimization through strategic entropy based exploration specifically tailored for function calling tasks. Our approach addresses three critical challenges in function calling: insufficient exploration during policy learning, lack of structured reasoning in chain-of-thought generation, and inadequate verification of parameter extraction. Our two-stage data preparation pipeline ensures high-quality training samples through iterative LLM evaluation and abstract syntax tree validation. Extensive experiments on the Berkeley Function Calling Leaderboard demonstrate that this framework achieves state-of-the-art performance among open-source models with 86.02\% overall accuracy, outperforming standard GRPO by up to 6\% on complex multi-function scenarios. Notably, our method shows particularly strong improvements on code-pretrained models, suggesting that structured language generation capabilities provide an advantageous starting point for reinforcement learning in function calling tasks. We will release all the code, models and dataset to benefit the community. 

**Abstract (ZH)**: 面向函数调用任务的新型增强学习框架：通过策略优化增强的策略熵导向探索 

---
# JPS: Jailbreak Multimodal Large Language Models with Collaborative Visual Perturbation and Textual Steering 

**Title (ZH)**: JPS: 使用协作视觉扰动和文本引导破解多模态大型语言模型 

**Authors**: Renmiao Chen, Shiyao Cui, Xuancheng Huang, Chengwei Pan, Victor Shea-Jay Huang, QingLin Zhang, Xuan Ouyang, Zhexin Zhang, Hongning Wang, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.05087)  

**Abstract**: Jailbreak attacks against multimodal large language Models (MLLMs) are a significant research focus. Current research predominantly focuses on maximizing attack success rate (ASR), often overlooking whether the generated responses actually fulfill the attacker's malicious intent. This oversight frequently leads to low-quality outputs that bypass safety filters but lack substantial harmful content. To address this gap, we propose JPS, \underline{J}ailbreak MLLMs with collaborative visual \underline{P}erturbation and textual \underline{S}teering, which achieves jailbreaks via corporation of visual image and textually steering prompt. Specifically, JPS utilizes target-guided adversarial image perturbations for effective safety bypass, complemented by "steering prompt" optimized via a multi-agent system to specifically guide LLM responses fulfilling the attackers' intent. These visual and textual components undergo iterative co-optimization for enhanced performance. To evaluate the quality of attack outcomes, we propose the Malicious Intent Fulfillment Rate (MIFR) metric, assessed using a Reasoning-LLM-based evaluator. Our experiments show JPS sets a new state-of-the-art in both ASR and MIFR across various MLLMs and benchmarks, with analyses confirming its efficacy. Codes are available at \href{this https URL}{this https URL}. \color{warningcolor}{Warning: This paper contains potentially sensitive contents.} 

**Abstract (ZH)**: 面向多模态大语言模型的协作型偷渡攻击及 

---
# Align, Don't Divide: Revisiting the LoRA Architecture in Multi-Task Learning 

**Title (ZH)**: 同调，不要分割：重新审视LoRA架构在多任务学习中的应用 

**Authors**: Jinda Liu, Bo Cheng, Yi Chang, Yuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05078)  

**Abstract**: Parameter-Efficient Fine-Tuning (PEFT) is essential for adapting Large Language Models (LLMs). In practice, LLMs are often required to handle a diverse set of tasks from multiple domains, a scenario naturally addressed by multi-task learning (MTL). Within this MTL context, a prevailing trend involves LoRA variants with multiple adapters or heads, which advocate for structural diversity to capture task-specific knowledge. Our findings present a direct challenge to this paradigm. We first show that a simplified multi-head architecture with high inter-head similarity substantially outperforms complex multi-adapter and multi-head systems. This leads us to question the multi-component paradigm itself, and we further demonstrate that a standard single-adapter LoRA, with a sufficiently increased rank, also achieves highly competitive performance. These results lead us to a new hypothesis: effective MTL generalization hinges on learning robust shared representations, not isolating task-specific features. To validate this, we propose Align-LoRA, which incorporates an explicit loss to align task representations within the shared adapter space. Experiments confirm that Align-LoRA significantly surpasses all baselines, establishing a simpler yet more effective paradigm for adapting LLMs to multiple tasks. The code is available at this https URL. 

**Abstract (ZH)**: 参数高效微调（PEFT）是适应大规模语言模型（LLMs）的关键。在实践中，LLMs 经常需要处理多个领域的多种任务，这一场景自然可以通过多任务学习（MTL）来解决。在这一 MTL 上下文中，流行的趋势是使用具有多个适配器或头部的 LoRA 变体，提倡结构多样性以捕获任务特定的知识。我们的发现直接挑战了这一范式。我们首先展示了高头部相似性的简化多头架构在复杂多适配器和多头系统中表现明显更优。这促使我们质疑多组件范式本身，并进一步证明，一个标准的单适配器 LoRA，如果秩足够增加，也能获得极具竞争力的表现。这些结果促使我们形成一个新假设：有效的 MTL 一般化依赖于学习稳健的共享表示，而不是隔离任务特定特征。为了验证这一点，我们提出了 Align-LoRA，其中包含一个显式的损失以在共享适配器空间内对齐任务表示。实验表明，Align-LoRA 显著优于所有基线，确立了一个更简单且更有效的范式来适应 LLMs 至多个任务。代码可用于此 <https://>。 

---
# Evaluation of LLMs in AMR Parsing 

**Title (ZH)**: LLM们在AMR解析中的评估 

**Authors**: Shu Han Ho  

**Link**: [PDF](https://arxiv.org/pdf/2508.05028)  

**Abstract**: Meaning Representation (AMR) is a semantic formalism that encodes sentence meaning as rooted, directed, acyclic graphs, where nodes represent concepts and edges denote semantic relations. Finetuning decoder only Large Language Models (LLMs) represent a promising novel straightfoward direction for AMR parsing. This paper presents a comprehensive evaluation of finetuning four distinct LLM architectures, Phi 3.5, Gemma 2, LLaMA 3.2, and DeepSeek R1 LLaMA Distilled using the LDC2020T02 Gold AMR3.0 test set. Our results have shown that straightfoward finetuning of decoder only LLMs can achieve comparable performance to complex State of the Art (SOTA) AMR parsers. Notably, LLaMA 3.2 demonstrates competitive performance against SOTA AMR parsers given a straightforward finetuning approach. We achieved SMATCH F1: 0.804 on the full LDC2020T02 test split, on par with APT + Silver (IBM) at 0.804 and approaching Graphene Smatch (MBSE) at 0.854. Across our analysis, we also observed a consistent pattern where LLaMA 3.2 leads in semantic performance while Phi 3.5 excels in structural validity. 

**Abstract (ZH)**: Meaning Representation (AMR)是一种将句子意义编码为有根、有向、无环图的语义正式语言，其中节点表示概念，边表示语义关系。微调解码器大型语言模型（LLMs）代表了AMR解析的一个有前途的新方向。本文对Phi 3.5、Gemma 2、LLaMA 3.2和DeepSeek R1 LLaMA Distilled四种不同的LLM架构进行了全面评估，使用LDC2020T02 Gold AMR3.0测试集。我们的结果显示，直接微调解码器大型语言模型可以达到与复杂最先进的（SOTA）AMR解析器相当的性能。值得注意的是，LLaMA 3.2在直接微调方法下表现出与SOTA AMR解析器竞争的性能。我们在LDC2020T02测试集的完整部分实现了SMATCH F1: 0.804，与APT + Silver（IBM）的0.804持平，并接近Graphene Smatch（MBSE）的0.854。在整个分析过程中，我们还观察到一种一致的趋势，即LLaMA 3.2在语义性能方面领先，而Phi 3.5在结构有效性方面表现出色。 

---
# SPaRFT: Self-Paced Reinforcement Fine-Tuning for Large Language Models 

**Title (ZH)**: SPaRFT: 自适应强化微调方法用于大规模语言模型 

**Authors**: Dai Do, Manh Nguyen, Svetha Venkatesh, Hung Le  

**Link**: [PDF](https://arxiv.org/pdf/2508.05015)  

**Abstract**: Large language models (LLMs) have shown strong reasoning capabilities when fine-tuned with reinforcement learning (RL). However, such methods require extensive data and compute, making them impractical for smaller models. Current approaches to curriculum learning or data selection are largely heuristic-driven or demand extensive computational resources, limiting their scalability and generalizability. We propose \textbf{SPaRFT}, a self-paced learning framework that enables efficient learning based on the capability of the model being trained through optimizing which data to use and when. First, we apply \emph{cluster-based data reduction} to partition training data by semantics and difficulty, extracting a compact yet diverse subset that reduces redundancy. Then, a \emph{multi-armed bandit} treats data clusters as arms, optimized to allocate training samples based on model current performance. Experiments across multiple reasoning benchmarks show that SPaRFT achieves comparable or better accuracy than state-of-the-art baselines while using up to \(100\times\) fewer samples. Ablation studies and analyses further highlight the importance of both data clustering and adaptive selection. Our results demonstrate that carefully curated, performance-driven training curricula can unlock strong reasoning abilities in LLMs with minimal resources. 

**Abstract (ZH)**: 基于自适应学习的SPaRFT框架：高效提升大型语言模型的推理能力 

---
# Making Prompts First-Class Citizens for Adaptive LLM Pipelines 

**Title (ZH)**: 自动适应大型语言模型管道中的提示一等公民 

**Authors**: Ugur Cetintemel, Shu Chen, Alexander W. Lee, Deepti Raghavan  

**Link**: [PDF](https://arxiv.org/pdf/2508.05012)  

**Abstract**: Modern LLM pipelines increasingly resemble data-centric systems: they retrieve external context, compose intermediate outputs, validate results, and adapt based on runtime feedback. Yet, the central element guiding this process -- the prompt -- remains a brittle, opaque string, disconnected from the surrounding dataflow. This disconnect limits reuse, optimization, and runtime control.
In this paper, we describe our vision and an initial design for SPEAR, a language and runtime that fills this prompt management gap by making prompts structured, adaptive, and first-class components of the execution model. SPEAR enables (1) runtime prompt refinement -- modifying prompts dynamically in response to execution-time signals such as confidence, latency, or missing context; and (2) structured prompt management -- organizing prompt fragments into versioned views with support for introspection and logging.
SPEAR defines a prompt algebra that governs how prompts are constructed and adapted within a pipeline. It supports multiple refinement modes (manual, assisted, and automatic), giving developers a balance between control and automation. By treating prompt logic as structured data, SPEAR enables optimizations such as operator fusion, prefix caching, and view reuse. Preliminary experiments quantify the behavior of different refinement modes compared to static prompts and agentic retries, as well as the impact of prompt-level optimizations such as operator fusion. 

**Abstract (ZH)**: 现代大模型管道 increasingly 类似于数据驱动系统：它们检索外部上下文、组成中间输出、验证结果并基于运行时反馈进行调整。然而，指导这一过程的核心元素——提示——仍然是一个脆弱且不透明的字符串，与周围的数据流脱节。这种脱节限制了提示的重用、优化和运行时控制。

在本文中，我们阐述了SPEAR的愿景和初步设计，这是一种语言和运行时系统，通过使提示结构化、自适应并成为执行模型的一等组件来填补这一提示管理缺口。SPEAR使（1）运行时提示细化成为可能——动态响应于执行时信号（如置信度、延迟或缺失上下文）修改提示；以及（2）结构化的提示管理——将提示片段组织成带有内省和日志支持的版本视图。

SPEAR定义了一种提示代数，规范了提示在管道中如何构建和调整。它支持多种细化模式（手动、辅助和自动），为开发人员提供了控制和自动化之间的平衡。通过将提示逻辑视为结构化数据，SPEAR使算子融合、前缀缓存和视图重用等优化成为可能。初步实验量化了不同细化模式的行为，与静态提示和主动重试相比，以及提示级别优化（如算子融合）的影响。 

---
# R-Zero: Self-Evolving Reasoning LLM from Zero Data 

**Title (ZH)**: R-Zero: 从零数据自我进化推理大规模语言模型 

**Authors**: Chengsong Huang, Wenhao Yu, Xiaoyang Wang, Hongming Zhang, Zongxia Li, Ruosen Li, Jiaxin Huang, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05004)  

**Abstract**: Self-evolving Large Language Models (LLMs) offer a scalable path toward super-intelligence by autonomously generating, refining, and learning from their own experiences. However, existing methods for training such models still rely heavily on vast human-curated tasks and labels, typically via fine-tuning or reinforcement learning, which poses a fundamental bottleneck to advancing AI systems toward capabilities beyond human intelligence. To overcome this limitation, we introduce R-Zero, a fully autonomous framework that generates its own training data from scratch. Starting from a single base LLM, R-Zero initializes two independent models with distinct roles, a Challenger and a Solver. These models are optimized separately and co-evolve through interaction: the Challenger is rewarded for proposing tasks near the edge of the Solver capability, and the Solver is rewarded for solving increasingly challenging tasks posed by the Challenger. This process yields a targeted, self-improving curriculum without any pre-existing tasks and labels. Empirically, R-Zero substantially improves reasoning capability across different backbone LLMs, e.g., boosting the Qwen3-4B-Base by +6.49 on math-reasoning benchmarks and +7.54 on general-domain reasoning benchmarks. 

**Abstract (ZH)**: 自我进化的大型语言模型（LLMs）通过自主生成、精炼和从自身经验中学习，提供了通往超级智能的可扩展路径。然而，现有方法仍高度依赖大量的人工标注任务和标签，通常是通过微调或强化学习来实现，这已成为推进AI系统超越人类智能能力的基本瓶颈。为克服这一限制，我们引入了R-Zero，这是一种完全自主的框架，从头开始生成自身的训练数据。从一个基础LLM开始，R-Zero 初始化两个独立的模型，分别承担挑战者和解题者两种角色。这两个模型分别优化，并通过交互共同进化：挑战者因提出接近解题者能力边界的任务而获得奖励，解题者因解决挑战者提出的一系列越来越具有挑战性的任务而获得奖励。这一过程生成了一个有针对性、自我改进的学习课程，无需任何预先存在的任务和标签。实验结果表明，R-Zero 显著提升了不同基础LLM 的推理能力，例如，将Qwen3-4B-Base 在数学推理基准上的性能提高了6.49分，在通用领域推理基准上的性能提高了7.54分。 

---
# A Multi-Stage Large Language Model Framework for Extracting Suicide-Related Social Determinants of Health 

**Title (ZH)**: 多阶段大型语言模型框架用于提取与自杀相关的社会决定因素 

**Authors**: Song Wang, Yishu Wei, Haotian Ma, Max Lovitt, Kelly Deng, Yuan Meng, Zihan Xu, Jingze Zhang, Yunyu Xiao, Ying Ding, Xuhai Xu, Joydeep Ghosh, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2508.05003)  

**Abstract**: Background: Understanding social determinants of health (SDoH) factors contributing to suicide incidents is crucial for early intervention and prevention. However, data-driven approaches to this goal face challenges such as long-tailed factor distributions, analyzing pivotal stressors preceding suicide incidents, and limited model explainability. Methods: We present a multi-stage large language model framework to enhance SDoH factor extraction from unstructured text. Our approach was compared to other state-of-the-art language models (i.e., pre-trained BioBERT and GPT-3.5-turbo) and reasoning models (i.e., DeepSeek-R1). We also evaluated how the model's explanations help people annotate SDoH factors more quickly and accurately. The analysis included both automated comparisons and a pilot user study. Results: We show that our proposed framework demonstrated performance boosts in the overarching task of extracting SDoH factors and in the finer-grained tasks of retrieving relevant context. Additionally, we show that fine-tuning a smaller, task-specific model achieves comparable or better performance with reduced inference costs. The multi-stage design not only enhances extraction but also provides intermediate explanations, improving model explainability. Conclusions: Our approach improves both the accuracy and transparency of extracting suicide-related SDoH from unstructured texts. These advancements have the potential to support early identification of individuals at risk and inform more effective prevention strategies. 

**Abstract (ZH)**: 背景:理解社会决定健康健康（社会决定因素）对自杀事件的影响对于早期干预和预防至关重要 giảng解释翻译为理解社会决定因素对 对自杀事件的影响对于早期干预和预防至关重要 nâussed
user
 Background: Understanding social determinants of health (S Đường) that contribute to suicide incidents is crucial for early intervention and prevention. Mentioning data唬限制了翻译，禁止翻译中 包含G 字符，请重试 nâ
-user
背景: � 了解社会决定因素（S Mounted）对自杀事件的影响是进行早期干预和预防的关键-ves
-user
背景: 了解社会决定因素（S Mounted）对自杀事件的影响是进行早期干预和预防的关键。 

---
# RCR-Router: Efficient Role-Aware Context Routing for Multi-Agent LLM Systems with Structured Memory 

**Title (ZH)**: RCR-路由器：具有结构化记忆的多agent LLM系统中高效的角色感知上下文路由 

**Authors**: Jun Liu, Zhenglun Kong, Changdi Yang, Fan Yang, Tianqi Li, Peiyan Dong, Joannah Nanjekye, Hao Tang, Geng Yuan, Wei Niu, Wenbin Zhang, Pu Zhao, Xue Lin, Dong Huang, Yanzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.04903)  

**Abstract**: Multi-agent large language model (LLM) systems have shown strong potential in complex reasoning and collaborative decision-making tasks. However, most existing coordination schemes rely on static or full-context routing strategies, which lead to excessive token consumption, redundant memory exposure, and limited adaptability across interaction rounds. We introduce RCR-Router, a modular and role-aware context routing framework designed to enable efficient, adaptive collaboration in multi-agent LLMs. To our knowledge, this is the first routing approach that dynamically selects semantically relevant memory subsets for each agent based on its role and task stage, while adhering to a strict token budget. A lightweight scoring policy guides memory selection, and agent outputs are iteratively integrated into a shared memory store to facilitate progressive context refinement. To better evaluate model behavior, we further propose an Answer Quality Score metric that captures LLM-generated explanations beyond standard QA accuracy. Experiments on three multi-hop QA benchmarks -- HotPotQA, MuSiQue, and 2WikiMultihop -- demonstrate that RCR-Router reduces token usage (up to 30%) while improving or maintaining answer quality. These results highlight the importance of structured memory routing and output-aware evaluation in advancing scalable multi-agent LLM systems. 

**Abstract (ZH)**: 多智能体大型语言模型（LLM）系统在复杂推理和协作决策任务中展现出强大的潜力。然而，现有的大多数协调方案依赖于静态或全上下文路由策略，导致了过多的令牌消耗、冗余的内存暴露和跨交互轮次的有限适应性。我们引入了RCR-Router，一种模块化和角色感知的上下文路由框架，旨在使多智能体LLM中的高效、自适应协作成为可能。据我们所知，这是第一个能够根据智能体的角色和任务阶段动态选择语义相关内存子集的方法，同时遵循严格的令牌预算。轻量级的评分政策指导内存选择，智能体输出逐迭代地整合到共享内存存储中以促进渐进的上下文细化。为了更好地评估模型行为，我们进一步提出了一种答案质量得分指标，该指标捕获了LLM生成的解释，而不仅仅是标准的问答准确性。在三个多跳QA基准——HotPotQA、MuSiQue和2WikiMultihop——上的实验表明，RCR-Router在减少令牌使用量（最多降低30%）的同时提高了或维持了答案质量。这些结果突显了结构化内存路由和输出感知评估在促进可扩展的多智能体LLM系统方面的重要性。 

---
# Adversarial Attacks and Defenses on Graph-aware Large Language Models (LLMs) 

**Title (ZH)**: 图意识大型语言模型的对抗攻击与防御 

**Authors**: Iyiola E. Olatunji, Franziska Boenisch, Jing Xu, Adam Dziedzic  

**Link**: [PDF](https://arxiv.org/pdf/2508.04894)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated with graph-structured data for tasks like node classification, a domain traditionally dominated by Graph Neural Networks (GNNs). While this integration leverages rich relational information to improve task performance, their robustness against adversarial attacks remains unexplored. We take the first step to explore the vulnerabilities of graph-aware LLMs by leveraging existing adversarial attack methods tailored for graph-based models, including those for poisoning (training-time attacks) and evasion (test-time attacks), on two representative models, LLAGA (Chen et al. 2024) and GRAPHPROMPTER (Liu et al. 2024). Additionally, we discover a new attack surface for LLAGA where an attacker can inject malicious nodes as placeholders into the node sequence template to severely degrade its performance. Our systematic analysis reveals that certain design choices in graph encoding can enhance attack success, with specific findings that: (1) the node sequence template in LLAGA increases its vulnerability; (2) the GNN encoder used in GRAPHPROMPTER demonstrates greater robustness; and (3) both approaches remain susceptible to imperceptible feature perturbation attacks. Finally, we propose an end-to-end defense framework GALGUARD, that combines an LLM-based feature correction module to mitigate feature-level perturbations and adapted GNN defenses to protect against structural attacks. 

**Abstract (ZH)**: 大型语言模型（LLMs） increasingly integrated with图结构数据用于节点分类等任务，这一领域传统上由图神经网络（GNNs）主导。尽管这种集成利用丰富的关系信息来提升任务性能，但其对抗攻击的稳健性尚未被探索。我们通过利用针对基于图模型的现有对抗攻击方法，包括训练时攻击（中毒攻击）和测试时攻击（规避攻击），首次探索图意识LLMs的脆弱性，针对两个代表性模型LLAGA（Chen et al. 2024）和GRAPHPROMPTER（Liu et al. 2024）进行了研究。此外，我们还发现LLAGA存在一个新的攻击面，攻击者可以注入恶意节点作为占位符到节点序列模板中，严重影响其性能。我们的系统分析揭示了某些图编码设计选择可以增强攻击成功率，具体发现包括：（1）LLAGA中的节点序列模板增加了其脆弱性；（2）GRAPHPROMPTER使用的GNN编码器表现出更强的鲁棒性；（3）两种方法都易受不可察觉特征扰动攻击的影响。最后，我们提出了一种端到端的防护框架GALGUARD，该框架结合了基于LLM的特征修正模块来减轻特征层面的扰动，并对结构攻击进行了适应性防护。 

---
# Provable Post-Training Quantization: Theoretical Analysis of OPTQ and Qronos 

**Title (ZH)**: 可验证后训练量化：OPTQ和Qronos的理论分析 

**Authors**: Haoyu Zhang, Shihao Zhang, Ian Colbert, Rayan Saab  

**Link**: [PDF](https://arxiv.org/pdf/2508.04853)  

**Abstract**: Post-training quantization (PTQ) has become a crucial tool for reducing the memory and compute costs of modern deep neural networks, including large language models (LLMs). Among PTQ algorithms, the OPTQ framework-also known as GPTQ-has emerged as a leading method due to its computational efficiency and strong empirical performance. Despite its widespread adoption, however, OPTQ lacks rigorous quantitative theoretical guarantees. This paper presents the first quantitative error bounds for both deterministic and stochastic variants of OPTQ, as well as for Qronos, a recent related state-of-the-art PTQ algorithm. We analyze how OPTQ's iterative procedure induces quantization error and derive non-asymptotic 2-norm error bounds that depend explicitly on the calibration data and a regularization parameter that OPTQ uses. Our analysis provides theoretical justification for several practical design choices, including the widely used heuristic of ordering features by decreasing norm, as well as guidance for selecting the regularization parameter. For the stochastic variant, we establish stronger infinity-norm error bounds, which enable control over the required quantization alphabet and are particularly useful for downstream layers and nonlinearities. Finally, we extend our analysis to Qronos, providing new theoretical bounds, for both its deterministic and stochastic variants, that help explain its empirical advantages. 

**Abstract (ZH)**: 训练后量化（PTQ）已成为降低现代深度神经网络，包括大型语言模型（LLMs）的内存和计算成本的关键工具。在PTQ算法中，OPTQ框架——也称为GPTQ——凭借其计算效率和强大的实证表现，已成为领先的方法。然而，尽管其广泛采用，OPTQ仍缺乏严格的定量理论保证。本文首次为OPTQ的确定性和随机变体，以及最近的相关最佳PTQ算法Qronos提供了定量的误差界。我们分析了OPTQ迭代过程引起的量化误差，并推导出依赖于校准数据和OPTQ使用的正则化参数的非渐近2-范数误差界。我们的分析为诸如按递减范数排序特征等若干实用设计选择提供了理论依据，并为选择正则化参数提供了指导。对于随机变体，我们建立了更强的∞-范数误差界，这使得可以控制所需的量化字母表，并特别适用于下游层和非线性操作。最后，我们将分析扩展到Qronos，为其实定性和随机变体提供了新的理论界，有助于解释其实证优势。 

---
# Persistent Instability in LLM's Personality Measurements: Effects of Scale, Reasoning, and Conversation History 

**Title (ZH)**: LLM个性测量中的持久不稳定性：量表、推理和对话历史的影响 

**Authors**: Tommaso Tosato, Saskia Helbling, Yorguin-Jose Mantilla-Ramos, Mahmood Hegazy, Alberto Tosato, David John Lemay, Irina Rish, Guillaume Dumas  

**Link**: [PDF](https://arxiv.org/pdf/2508.04826)  

**Abstract**: Large language models require consistent behavioral patterns for safe deployment, yet their personality-like traits remain poorly understood. We present PERSIST (PERsonality Stability in Synthetic Text), a comprehensive evaluation framework testing 25+ open-source models (1B-671B parameters) across 500,000+ responses. Using traditional (BFI-44, SD3) and novel LLM-adapted personality instruments, we systematically vary question order, paraphrasing, personas, and reasoning modes. Our findings challenge fundamental deployment assumptions: (1) Even 400B+ models exhibit substantial response variability (SD > 0.4); (2) Minor prompt reordering alone shifts personality measurements by up to 20%; (3) Interventions expected to stabilize behavior, such as chain-of-thought reasoning, detailed personas instruction, inclusion of conversation history, can paradoxically increase variability; (4) LLM-adapted instruments show equal instability to human-centric versions, confirming architectural rather than translational limitations. This persistent instability across scales and mitigation strategies suggests current LLMs lack the foundations for genuine behavioral consistency. For safety-critical applications requiring predictable behavior, these findings indicate that personality-based alignment strategies may be fundamentally inadequate. 

**Abstract (ZH)**: 大规模语言模型需要一致的行为模式以确保安全部署，但其类似人格的特征仍 poorly understood。我们提出PERSIST（PERsonality Stability in Synthetic Text）框架，测试了超过25个开源模型（参数从1B到671B）在500,000多条响应中的表现。利用传统（BFI-44, SD3）以及新型语言模型适应的人格量表，我们系统地变化了问题顺序、改写、人物设定和推理模式。我们的研究结果挑战了基本的部署假设：（1）即使超过400B的模型也表现出显著的响应波动性（标准差>0.4）；（2）单一的提示重新排序会将人格测量值改变多达20%；（3）预期能够稳定行为的干预措施，如逐步推理、详细的背景人物指令、包含对话历史，反而可能增加波动性；（4）语言模型适应的人格量表与以人为中心的版本表现相当不稳定，证实了架构而非翻译限制。这一持续的不稳定性及缓解策略的无效性表明当前的语言模型缺乏真正行为一致性的基础。对于需要可预测行为的安全关键应用，这些发现表明基于人格的对齐策略可能从根本上不足。 

---
# Automated File-Level Logging Generation for Machine Learning Applications using LLMs: A Case Study using GPT-4o Mini 

**Title (ZH)**: 使用LLMs生成机器学习应用的文件级日志记录：基于GPT-4o Mini的案例研究 

**Authors**: Mayra Sofia Ruiz Rodriguez, SayedHassan Khatoonabadi, Emad Shihab  

**Link**: [PDF](https://arxiv.org/pdf/2508.04820)  

**Abstract**: Logging is essential in software development, helping developers monitor system behavior and aiding in debugging applications. Given the ability of large language models (LLMs) to generate natural language and code, researchers are exploring their potential to generate log statements. However, prior work focuses on evaluating logs introduced in code functions, leaving file-level log generation underexplored -- especially in machine learning (ML) applications, where comprehensive logging can enhance reliability. In this study, we evaluate the capacity of GPT-4o mini as a case study to generate log statements for ML projects at file level. We gathered a set of 171 ML repositories containing 4,073 Python files with at least one log statement. We identified and removed the original logs from the files, prompted the LLM to generate logs for them, and evaluated both the position of the logs and log level, variables, and text quality of the generated logs compared to human-written logs. In addition, we manually analyzed a representative sample of generated logs to identify common patterns and challenges. We find that the LLM introduces logs in the same place as humans in 63.91% of cases, but at the cost of a high overlogging rate of 82.66%. Furthermore, our manual analysis reveals challenges for file-level logging, which shows overlogging at the beginning or end of a function, difficulty logging within large code blocks, and misalignment with project-specific logging conventions. While the LLM shows promise for generating logs for complete files, these limitations remain to be addressed for practical implementation. 

**Abstract (ZH)**: 大语言模型生成文件级日志陈述的研究：以GPT-4o mini为案例的机器学习项目日志生成能力评估 

---
# Enhancing Dialogue Annotation with Speaker Characteristics Leveraging a Frozen LLM 

**Title (ZH)**: 利用冻结的预训练语言模型挖掘说话人特征以增强对话标注 

**Authors**: Thomas Thebaud, Yen-Ju Lu, Matthew Wiesner, Peter Viechnicki, Najim Dehak  

**Link**: [PDF](https://arxiv.org/pdf/2508.04795)  

**Abstract**: In dialogue transcription pipelines, Large Language Models (LLMs) are frequently employed in post-processing to improve grammar, punctuation, and readability. We explore a complementary post-processing step: enriching transcribed dialogues by adding metadata tags for speaker characteristics such as age, gender, and emotion. Some of the tags are global to the entire dialogue, while some are time-variant. Our approach couples frozen audio foundation models, such as Whisper or WavLM, with a frozen LLAMA language model to infer these speaker attributes, without requiring task-specific fine-tuning of either model. Using lightweight, efficient connectors to bridge audio and language representations, we achieve competitive performance on speaker profiling tasks while preserving modularity and speed. Additionally, we demonstrate that a frozen LLAMA model can compare x-vectors directly, achieving an Equal Error Rate of 8.8% in some scenarios. 

**Abstract (ZH)**: 在对话转写流水线中，大型语言模型（LLMs）经常用于后处理以改进语法、标点和可读性。我们探索了一个互补的后处理步骤：通过添加元数据标签来丰富转录对话，这些标签包括演讲者特征如年龄、性别和情感。部分标签适用于整个对话，而部分标签则是时间变化的。我们的方法结合了冻结的音频基础模型（如Whisper或WavLM）与冻结的LLAMA语言模型，以推断这些演讲者属性，无需对任何模型进行特定任务的微调。借助轻量级且高效的连接器来桥接音频和语言表示，我们在演讲者画像任务中实现了竞争力的表现，同时保持模块化和速度。此外，我们展示了冻结的LLAMA模型可以直接比较x-向量，在某些情况下实现了8.8%的平等错误率。 

---
# Evaluating the Impact of LLM-guided Reflection on Learning Outcomes with Interactive AI-Generated Educational Podcasts 

**Title (ZH)**: 基于交互式AI生成教育播客的LLM引导反思对学习成果的影响评估 

**Authors**: Vishnu Menon, Andy Cherney, Elizabeth B. Cloude, Li Zhang, Tiffany D. Do  

**Link**: [PDF](https://arxiv.org/pdf/2508.04787)  

**Abstract**: This study examined whether embedding LLM-guided reflection prompts in an interactive AI-generated podcast improved learning and user experience compared to a version without prompts. Thirty-six undergraduates participated, and while learning outcomes were similar across conditions, reflection prompts reduced perceived attractiveness, highlighting a call for more research on reflective interactivity design. 

**Abstract (ZH)**: 本研究探讨了将LLM引导的反思提示嵌入交互式AI生成的播客是否能改善学习效果和用户体验，相较于未包含提示的版本。36名本科生参与了研究，尽管各条件下学习成果相近，但反思提示降低了 perceived 吸引力，强调了在反思互动设计方面需要更多研究的必要性。 

---
# Toward Low-Latency End-to-End Voice Agents for Telecommunications Using Streaming ASR, Quantized LLMs, and Real-Time TTS 

**Title (ZH)**: 面向电信领域的低延迟端到端语音代理：基于流式ASR、量化LLM和实时TTS 

**Authors**: Vignesh Ethiraj, Ashwath David, Sidhanth Menon, Divya Vijay  

**Link**: [PDF](https://arxiv.org/pdf/2508.04721)  

**Abstract**: We introduce a low-latency telecom AI voice agent pipeline for real-time, interactive telecommunications use, enabling advanced voice AI for call center automation, intelligent IVR (Interactive Voice Response), and AI-driven customer support. The solution is built for telecom, combining four specialized models by NetoAI: TSLAM, a 4-bit quantized Telecom-Specific Large Language Model (LLM); T-VEC, a Telecom-Specific Embedding Model; TTE, a Telecom-Specific Automatic Speech Recognition (ASR) model; and T-Synth, a Telecom-Specific Text-to-Speech (TTS) model. These models enable highly responsive, domain-adapted voice AI agents supporting knowledge-grounded spoken interactions with low latency. The pipeline integrates streaming ASR (TTE), conversational intelligence (TSLAM), retrieval augmented generation (RAG) over telecom documents, and real-time TTS (T-Synth), setting a new benchmark for telecom voice assistants. To evaluate the system, we built a dataset of 500 human-recorded telecom questions from RFCs, simulating real telecom agent queries. This framework allows analysis of latency, domain relevance, and real-time performance across the stack. Results show that TSLAM, TTE, and T-Synth deliver real-time factors (RTF) below 1.0, supporting enterprise, low-latency telecom deployments. These AI agents -- powered by TSLAM, TTE, and T-Synth -- provide a foundation for next-generation telecom AI, enabling automated customer support, diagnostics, and more. 

**Abstract (ZH)**: 一种低延迟电信AI语音代理流水线，实现实时互动电信应用，推动呼叫中心自动化、智能IVR和AI驱动的客户服务中的高级语音AI。该解决方案结合了NetoAI的四种专门模型：TSLAM（4-bit量化电信专用大型语言模型）、T-VEC（电信专用嵌入模型）、TTE（电信专用自动语音识别模型）和T-Synth（电信专用文本到语音模型）。这些模型能够提供高度响应且领域适应性强的语音AI代理，支持基于知识的语音交互且具有低延迟。该流水线集成了流式ASR（TTE）、对话智能（TSLAM）、基于电信文档的检索增强生成（RAG）以及实时TTS（T-Synth），树立了电信语音助手的新标杆。为评估系统，我们构建了一个包含500个人工录制的电信问题的数据集，模拟实际的电信代理查询。此框架允许对流水线中各层的延迟、领域相关性和实时性能进行分析。结果显示，TSLAM、TTE和T-Synth实现实时因子（RTF）低于1.0，支持企业级低延迟电信部署。这些由TSLAM、TTE和T-Synth支持的AI代理为下一代电信AI奠定了基础，使其能够实现自动化客户服务、故障诊断等功能。 

---
# AI Should Be More Human, Not More Complex 

**Title (ZH)**: AI 应更加人性化，而非更加复杂。 

**Authors**: Carlo Esposito  

**Link**: [PDF](https://arxiv.org/pdf/2508.04713)  

**Abstract**: Large Language Models (LLMs) in search applications increasingly prioritize verbose, lexically complex responses that paradoxically reduce user satisfaction and engagement. Through a comprehensive study of 10.000 (est.) participants comparing responses from five major AI-powered search systems, we demonstrate that users overwhelmingly prefer concise, source-attributed responses over elaborate explanations. Our analysis reveals that current AI development trends toward "artificial sophistication" create an uncanny valley effect where systems sound knowledgeable but lack genuine critical thinking, leading to reduced trust and increased cognitive load. We present evidence that optimal AI communication mirrors effective human discourse: direct, properly sourced, and honest about limitations. Our findings challenge the prevailing assumption that more complex AI responses indicate better performance, instead suggesting that human-like brevity and transparency are key to user engagement and system reliability. 

**Abstract (ZH)**: 大型语言模型（LLMs）在搜索应用中的使用 increasingly 优先生成冗长且词缀复杂的回答，这反而降低了用户满意感和参与度。通过对比五大主要AI驱动搜索系统生成的回答，我们的研究显示用户更偏好简洁且来源明确的回答而非详尽的解释。我们分析发现目前的AI开发趋势倾向于“伪 sophistication”，导致系统看似有知识但缺乏真实的批判性思考，从而降低了用户的信任度并增加了认知负荷。我们提出证据表明，最优的AI沟通应像有效的对话那样直接、有恰当来源，并且诚实地承认局限。我们的研究挑战了复杂AI回答意味着更好性能的主流假设，反而表明类人的简洁和透明度是提高用户参与度和系统可靠性的关键。 

---
# How Robust are LLM-Generated Library Imports? An Empirical Study using Stack Overflow 

**Title (ZH)**: LLM生成的库导入的稳健性研究：基于Stack Overflow的实证研究 

**Authors**: Jasmine Latendresse, SayedHassan Khatoonabadi, Emad Shihab  

**Link**: [PDF](https://arxiv.org/pdf/2507.10818)  

**Abstract**: Software libraries are central to the functionality, security, and maintainability of modern code. As developers increasingly turn to Large Language Models (LLMs) to assist with programming tasks, understanding how these models recommend libraries is essential. In this paper, we conduct an empirical study of six state-of-the-art LLMs, both proprietary and open-source, by prompting them to solve real-world Python problems sourced from Stack Overflow. We analyze the types of libraries they import, the characteristics of those libraries, and the extent to which the recommendations are usable out of the box. Our results show that LLMs predominantly favour third-party libraries over standard ones, and often recommend mature, popular, and permissively licensed dependencies. However, we also identify gaps in usability: 4.6% of the libraries could not be resolved automatically due to structural mismatches between import names and installable packages, and only two models (out of six) provided installation guidance. While the generated code is technically valid, the lack of contextual support places the burden of manually resolving dependencies on the user. Our findings offer actionable insights for both developers and researchers, and highlight opportunities to improve the reliability and usability of LLM-generated code in the context of software dependencies. 

**Abstract (ZH)**: 软件库是现代代码的功能性、安全性和可维护性的核心。随着开发者越来越多地利用大型语言模型（LLMs）来协助编程任务，理解这些模型如何推荐库变得至关重要。在本文中，我们通过提示六种最先进的LLMs（包括商用和开源模型），使用来自Stack Overflow的实际Python问题来开展实证研究。我们分析了它们导入的库类型、这些库的特点，以及推荐的程度可直接使用。研究结果表明，LLMs更倾向于推荐第三方库而非标准库，并且经常推荐成熟、流行且具有宽松许可证的依赖项。然而，我们也发现了使用方面的不足：4.6%的库由于导入名称与可安装包之间的结构不匹配而无法自动解析，只有两个模型提供了安装指导。虽然生成的代码在技术上是有效的，但由于缺乏上下文支持，用户仍需手动解决依赖项的问题。我们的研究结果为开发者和研究人员提供了实用的见解，并突显了在软件依赖背景下提高LLM生成代码可靠性和使用性的机会。 

---
