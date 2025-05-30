# RANa: Retrieval-Augmented Navigation 

**Title (ZH)**: RANa: 检索增强导航 

**Authors**: Gianluca Monaci, Rafael S. Rezende, Romain Deffayet, Gabriela Csurka, Guillaume Bono, Hervé Déjean, Stéphane Clinchant, Christian Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2504.03524)  

**Abstract**: Methods for navigation based on large-scale learning typically treat each episode as a new problem, where the agent is spawned with a clean memory in an unknown environment. While these generalization capabilities to an unknown environment are extremely important, we claim that, in a realistic setting, an agent should have the capacity of exploiting information collected during earlier robot operations. We address this by introducing a new retrieval-augmented agent, trained with RL, capable of querying a database collected from previous episodes in the same environment and learning how to integrate this additional context information. We introduce a unique agent architecture for the general navigation task, evaluated on ObjectNav, ImageNav and Instance-ImageNav. Our retrieval and context encoding methods are data-driven and heavily employ vision foundation models (FM) for both semantic and geometric understanding. We propose new benchmarks for these settings and we show that retrieval allows zero-shot transfer across tasks and environments while significantly improving performance. 

**Abstract (ZH)**: 基于大规模学习的导航方法通常将每个episode视为一个新的问题，其中智能体在未知环境中以清空记忆的状态出现。虽然这种对未知环境的泛化能力非常重要，但我们认为在实际环境中，智能体应该有能力利用之前机器人操作中收集的信息。为此，我们提出了一种新的检索增强智能体，通过强化学习训练，能够在相同环境的先前episode中查询数据库，并学习如何整合这些额外的上下文信息。我们为通用导航任务引入了一种独特的智能体架构，并在ObjectNav、ImageNav和Instance-ImageNav上进行了评估。我们的检索和上下文编码方法是数据驱动的，并大量使用视觉基础模型（FM）来进行语义和几何理解。我们为这些环境提出了新的基准，并展示了检索允许在任务和环境之间进行零-shot迁移，并显著提高了性能。 

---
# An Efficient GPU-based Implementation for Noise Robust Sound Source Localization 

**Title (ZH)**: 基于GPU的噪声鲁棒声源定位高效实现 

**Authors**: Zirui Lin, Masayuki Takigahira, Naoya Terakado, Haris Gulzar, Monikka Roslianna Busto, Takeharu Eda, Katsutoshi Itoyama, Kazuhiro Nakadai, Hideharu Amano  

**Link**: [PDF](https://arxiv.org/pdf/2504.03373)  

**Abstract**: Robot audition, encompassing Sound Source Localization (SSL), Sound Source Separation (SSS), and Automatic Speech Recognition (ASR), enables robots and smart devices to acquire auditory capabilities similar to human hearing. Despite their wide applicability, processing multi-channel audio signals from microphone arrays in SSL involves computationally intensive matrix operations, which can hinder efficient deployment on Central Processing Units (CPUs), particularly in embedded systems with limited CPU resources. This paper introduces a GPU-based implementation of SSL for robot audition, utilizing the Generalized Singular Value Decomposition-based Multiple Signal Classification (GSVD-MUSIC), a noise-robust algorithm, within the HARK platform, an open-source software suite. For a 60-channel microphone array, the proposed implementation achieves significant performance improvements. On the Jetson AGX Orin, an embedded device powered by an NVIDIA GPU and ARM Cortex-A78AE v8.2 64-bit CPUs, we observe speedups of 4645.1x for GSVD calculations and 8.8x for the SSL module, while speedups of 2223.4x for GSVD calculation and 8.95x for the entire SSL module on a server configured with an NVIDIA A100 GPU and AMD EPYC 7352 CPUs, making real-time processing feasible for large-scale microphone arrays and providing ample capacity for real-time processing of potential subsequent machine learning or deep learning tasks. 

**Abstract (ZH)**: 基于GPU的机器人听觉声源定位实现：利用HARK平台的噪稳健广义奇异值分解多重信号分类算法 

---
# Distributionally Robust Predictive Runtime Verification under Spatio-Temporal Logic Specifications 

**Title (ZH)**: 基于时空逻辑规范的分布鲁棒预测运行时验证 

**Authors**: Yiqi Zhao, Emily Zhu, Bardh Hoxha, Georgios Fainekos, Jyotirmoy V. Deshmukh, Lars Lindemann  

**Link**: [PDF](https://arxiv.org/pdf/2504.02964)  

**Abstract**: Cyber-physical systems designed in simulators, often consisting of multiple interacting agents, behave differently in the real-world. We would like to verify these systems during runtime when they are deployed. Thus, we propose robust predictive runtime verification (RPRV) algorithms for: (1) general stochastic CPS under signal temporal logic (STL) tasks, and (2) stochastic multi-agent systems (MAS) under spatio-temporal logic tasks. The RPRV problem presents the following challenges: (1) there may not be sufficient data on the behavior of the deployed CPS, (2) predictive models based on design phase system trajectories may encounter distribution shift during real-world deployment, and (3) the algorithms need to scale to the complexity of MAS and be applicable to spatio-temporal logic tasks. To address these challenges, we assume knowledge of an upper bound on the statistical distance (in terms of an f-divergence) between the trajectory distributions of the system at deployment and design time. We are motivated by our prior work [1, 2] where we proposed an accurate and an interpretable RPRV algorithm for general CPS, which we here extend to the MAS setting and spatio-temporal logic tasks. Specifically, we use a learned predictive model to estimate the system behavior at runtime and robust conformal prediction to obtain probabilistic guarantees by accounting for distribution shifts. Building on [1], we perform robust conformal prediction over the robust semantics of spatio-temporal reach and escape logic (STREL) to obtain centralized RPRV algorithms for MAS. We empirically validate our results in a drone swarm simulator, where we show the scalability of our RPRV algorithms to MAS and analyze the impact of different trajectory predictors on the verification result. To the best of our knowledge, these are the first statistically valid algorithms for MAS under distribution shift. 

**Abstract (ZH)**: 针对分布偏移的鲁棒预测运行时验证算法：面向随机物理系统与空间-时间逻辑任务 

---
# Monte Carlo Graph Coloring 

**Title (ZH)**: 蒙特卡洛图着色 

**Authors**: Tristan Cazenave, Benjamin Negrevergne, Florian Sikora  

**Link**: [PDF](https://arxiv.org/pdf/2504.03277)  

**Abstract**: Graph Coloring is probably one of the most studied and famous problem in graph algorithms. Exact methods fail to solve instances with more than few hundred vertices, therefore, a large number of heuristics have been proposed. Nested Monte Carlo Search (NMCS) and Nested Rollout Policy Adaptation (NRPA) are Monte Carlo search algorithms for single player games. Surprisingly, few work has been dedicated to evaluating Monte Carlo search algorithms to combinatorial graph problems. In this paper we expose how to efficiently apply Monte Carlo search to Graph Coloring and compare this approach to existing ones. 

**Abstract (ZH)**: 图着色可能是图算法中研究最多和最著名的问题之一。精确方法无法解决具有几百个以上顶点的实例，因此提出了大量的启发式方法。嵌套蒙特卡洛搜索（NMCS）和嵌套展开策略适应（NRPA）是单人游戏的蒙特卡洛搜索算法。令人惊讶的是，很少有工作专门评估蒙特卡洛搜索算法在组合图问题上的效果。在这项工作中，我们展示了如何有效地将蒙特卡洛搜索应用于图着色，并将这种方法与现有方法进行比较。 

---
# Bonsai: Interpretable Tree-Adaptive Grounded Reasoning 

**Title (ZH)**: Bonsai: 可解释的树适应性 grounded 推理 

**Authors**: Kate Sanders, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2504.03640)  

**Abstract**: To develop general-purpose collaborative agents, humans need reliable AI systems that can (1) adapt to new domains and (2) transparently reason with uncertainty to allow for verification and correction. Black-box models demonstrate powerful data processing abilities but do not satisfy these criteria due to their opaqueness, domain specificity, and lack of uncertainty awareness. We introduce Bonsai, a compositional and probabilistic reasoning system that generates adaptable inference trees by retrieving relevant grounding evidence and using it to compute likelihoods of sub-claims derived from broader natural language inferences. Bonsai's reasoning power is tunable at test-time via evidence scaling and it demonstrates reliable handling of varied domains including transcripts, photographs, videos, audio, and databases. Question-answering and human alignment experiments demonstrate that Bonsai matches the performance of domain-specific black-box methods while generating interpretable, grounded, and uncertainty-aware reasoning traces. 

**Abstract (ZH)**: 开发通用协作代理，需要可靠的AI系统，这些系统能够（1）适应新领域，（2）透明地处理不确定性以允许验证和纠正。虽然黑盒模型展示了强大的数据处理能力，但由于其不透明性、领域特定性和缺乏不确定性意识，无法满足这些标准。我们引入了Bonsai，这是一种组合性和概率性推理系统，通过检索相关基础证据并使用这些证据计算源自更广泛的自然语言推理的子命题的似然性，生成适应性推理树。Bonsai的推理能力可以在测试时通过证据缩放进行调节，并且在包括转录、照片、视频、音频和数据库在内的各种领域中展现出可靠的处理能力。问答和人类对齐实验表明，Bonsai在性能上达到了领域特定黑盒方法的水平，同时生成可解释、基于证据和不确定性意识的推理轨迹。 

---
# MultiMed-ST: Large-scale Many-to-many Multilingual Medical Speech Translation 

**Title (ZH)**: MultiMed-ST：大规模多对多医疗语音多语言翻译 

**Authors**: Khai Le-Duc, Tuyen Tran, Bach Phan Tat, Nguyen Kim Hai Bui, Quan Dang, Hung-Phong Tran, Thanh-Thuy Nguyen, Ly Nguyen, Tuan-Minh Phan, Thi Thu Phuong Tran, Chris Ngo, Nguyen X. Khanh, Thanh Nguyen-Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03546)  

**Abstract**: Multilingual speech translation (ST) in the medical domain enhances patient care by enabling efficient communication across language barriers, alleviating specialized workforce shortages, and facilitating improved diagnosis and treatment, particularly during pandemics. In this work, we present the first systematic study on medical ST, to our best knowledge, by releasing MultiMed-ST, a large-scale ST dataset for the medical domain, spanning all translation directions in five languages: Vietnamese, English, German, French, Traditional Chinese and Simplified Chinese, together with the models. With 290,000 samples, our dataset is the largest medical machine translation (MT) dataset and the largest many-to-many multilingual ST among all domains. Secondly, we present the most extensive analysis study in ST research to date, including: empirical baselines, bilingual-multilingual comparative study, end-to-end vs. cascaded comparative study, task-specific vs. multi-task sequence-to-sequence (seq2seq) comparative study, code-switch analysis, and quantitative-qualitative error analysis. All code, data, and models are available online: this https URL. 

**Abstract (ZH)**: 多语言医疗语音翻译在医疗领域的应用通过消除语言障碍实现高效沟通，缓解专业人员短缺，并在疫情等期间促进诊断和治疗的改进。在本文中，我们首次系统地研究了医疗领域的语音翻译，通过发布包含六种语言（越南语、英语、德语、法语、繁体中文、简体中文）所有翻译方向的大型语音翻译数据集MultiMed-ST及相应的模型，对该数据集进行了全面分析，它是最大的医疗机器翻译数据集，也是所有领域中最大的多对多多语言语音翻译数据集。此外，我们还进行了迄今为止最广泛的语音翻译研究，包括经验基线、双语-多语比较研究、端到端与级联比较研究、特定任务与多任务序列到序列比较研究、代码转换分析以及定量-定性错误分析。所有代码、数据和模型均可在线获取：this https URL。 

---
# Dense Neural Network Based Arrhythmia Classification on Low-cost and Low-compute Micro-controller 

**Title (ZH)**: 基于密集神经网络的低-cost低计算量微控制器心律失常分类 

**Authors**: Md Abu Obaida Zishan, H M Shihab, Sabik Sadman Islam, Maliha Alam Riya, Gazi Mashrur Rahman, Jannatun Noor  

**Link**: [PDF](https://arxiv.org/pdf/2504.03531)  

**Abstract**: The electrocardiogram (ECG) monitoring device is an expensive albeit essential device for the treatment and diagnosis of cardiovascular diseases (CVD). The cost of this device typically ranges from $2000 to $10000. Several studies have implemented ECG monitoring systems in micro-controller units (MCU) to reduce industrial development costs by up to 20 times. However, to match industry-grade systems and display heartbeats effectively, it is essential to develop an efficient algorithm for detecting arrhythmia (irregular heartbeat). Hence in this study, a dense neural network is developed to detect arrhythmia on the Arduino Nano. The Nano consists of the ATMega328 microcontroller with a 16MHz clock, 2KB of SRAM, and 32KB of program memory. Additionally, the AD8232 SparkFun Single-Lead Heart Rate Monitor is used as the ECG sensor. The implemented neural network model consists of two layers (excluding the input) with 10 and four neurons respectively with sigmoid activation function. However, four approaches are explored to choose the appropriate activation functions. The model has a size of 1.267 KB, achieves an F1 score (macro-average) of 78.3\% for classifying four types of arrhythmia, an accuracy rate of 96.38%, and requires 0.001314 MOps of floating-point operations (FLOPs). 

**Abstract (ZH)**: 基于Arduino Nano的心电图心动过速检测方法研究 

---
# Quantifying Robustness: A Benchmarking Framework for Deep Learning Forecasting in Cyber-Physical Systems 

**Title (ZH)**: 定量评估鲁棒性：在网络物理系统中深度学习预测的基准框架 

**Authors**: Alexander Windmann, Henrik Steude, Daniel Boschmann, Oliver Niggemann  

**Link**: [PDF](https://arxiv.org/pdf/2504.03494)  

**Abstract**: Cyber-Physical Systems (CPS) in domains such as manufacturing and energy distribution generate complex time series data crucial for Prognostics and Health Management (PHM). While Deep Learning (DL) methods have demonstrated strong forecasting capabilities, their adoption in industrial CPS remains limited due insufficient robustness. Existing robustness evaluations primarily focus on formal verification or adversarial perturbations, inadequately representing the complexities encountered in real-world CPS scenarios. To address this, we introduce a practical robustness definition grounded in distributional robustness, explicitly tailored to industrial CPS, and propose a systematic framework for robustness evaluation. Our framework simulates realistic disturbances, such as sensor drift, noise and irregular sampling, enabling thorough robustness analyses of forecasting models on real-world CPS datasets. The robustness definition provides a standardized score to quantify and compare model performance across diverse datasets, assisting in informed model selection and architecture design. Through extensive empirical studies evaluating prominent DL architectures (including recurrent, convolutional, attention-based, modular, and structured state-space models) we demonstrate the applicability and effectiveness of our approach. We publicly release our robustness benchmark to encourage further research and reproducibility. 

**Abstract (ZH)**: 工业 CPS 领域（如制造业和能源分配）生成的复杂时间序列数据对于预测与健康管理 (PHM) 至关重要。尽管深度学习 (DL) 方法在预测能力上表现出色，但其在工业 CPS 中的应用仍因鲁棒性不足而受到限制。现有的鲁棒性评估主要集中在形式验证或对抗性扰动上，未能充分反映现实世界 CPS 场景中的复杂性。为解决这一问题，我们引入了一个基于分布鲁棒性的实用鲁棒性定义，该定义特别针对工业 CPS 设计，并提出了一套系统的鲁棒性评估框架。该框架模拟了实际干扰，如传感器漂移、噪声和不规则采样，从而对实际工业 CPS 数据集上的预测模型进行全面的鲁棒性分析。鲁棒性定义提供了一个标准化评分，用于衡量和比较模型在不同数据集上的性能，有助于模型选择和架构设计的知情决策。通过广泛的实际研究评估了主流 DL 架构（包括循环、卷积、注意力、模块化和结构化状态空间模型），我们证明了我们方法的适用性和有效性。我们公开发布了鲁棒性基准，以鼓励进一步的研究和可再现性。 

---
# Structured Legal Document Generation in India: A Model-Agnostic Wrapper Approach with VidhikDastaavej 

**Title (ZH)**: 印度结构化法律文件生成：一种基于模型的包装器方法与VidhikDastaavej 

**Authors**: Shubham Kumar Nigam, Balaramamahanthi Deepak Patnaik, Ajay Varghese Thomas, Noel Shallum, Kripabandhu Ghosh, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2504.03486)  

**Abstract**: Automating legal document drafting can significantly enhance efficiency, reduce manual effort, and streamline legal workflows. While prior research has explored tasks such as judgment prediction and case summarization, the structured generation of private legal documents in the Indian legal domain remains largely unaddressed. To bridge this gap, we introduce VidhikDastaavej, a novel, anonymized dataset of private legal documents, and develop NyayaShilp, a fine-tuned legal document generation model specifically adapted to Indian legal texts. We propose a Model-Agnostic Wrapper (MAW), a two-step framework that first generates structured section titles and then iteratively produces content while leveraging retrieval-based mechanisms to ensure coherence and factual accuracy. We benchmark multiple open-source LLMs, including instruction-tuned and domain-adapted versions, alongside proprietary models for comparison. Our findings indicate that while direct fine-tuning on small datasets does not always yield improvements, our structured wrapper significantly enhances coherence, factual adherence, and overall document quality while mitigating hallucinations. To ensure real-world applicability, we developed a Human-in-the-Loop (HITL) Document Generation System, an interactive user interface that enables users to specify document types, refine section details, and generate structured legal drafts. This tool allows legal professionals and researchers to generate, validate, and refine AI-generated legal documents efficiently. Extensive evaluations, including expert assessments, confirm that our framework achieves high reliability in structured legal drafting. This research establishes a scalable and adaptable foundation for AI-assisted legal drafting in India, offering an effective approach to structured legal document generation. 

**Abstract (ZH)**: 自动化法律文书起草可以显著提升效率、减少手动工作量并简化法律工作流程。尽管前期研究已经探索了判决预测和案例总结等任务，印度法律领域的结构化生成私密法律文书仍然 largely未被关注。为弥补这一不足，我们引入了 VidhikDastaavej，一个新型的匿名私密法律文书数据集，并开发了 NyayaShilp，一种专门适应印度法律文本的微调法律文书生成模型。我们提出了一种模型无关包装器（MAW），这是一种两步框架，首先生成结构化的部分标题，然后利用检索机制迭代生产内容，以确保连贯性和事实准确性。我们使用多种开源LLM进行基准测试，包括指令微调和领域适应版本，以及对照专有模型。研究结果表明，虽然直接在小数据集上进行微调并不总是能够带来改进，但我们的结构化包装器在提高连贯性、事实准确性和总体文档质量方面表现出色，并减少了幻觉现象。为了确保实际应用，我们开发了一个人机交互文档生成系统（HITL），这是一种交互式用户界面，使用户能够指定文档类型、细化部分详情并生成结构化的法律草案。该工具使法律专业人士和研究人员能够有效地生成、验证和细化AI生成的法律文书。广泛的评估，包括专家评估，证实了我们框架在结构化法律起草中的高可靠性。本研究为印度法律辅助起草奠定了可扩展和适应性强的基础，提供了一种有效的结构化法律文书生成方法。 

---
# EOOD: Entropy-based Out-of-distribution Detection 

**Title (ZH)**: EOOD: 基于熵的分布外检测 

**Authors**: Guide Yang, Chao Hou, Weilong Peng, Xiang Fang, Yongwei Nie, Peican Zhu, Keke Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03342)  

**Abstract**: Deep neural networks (DNNs) often exhibit overconfidence when encountering out-of-distribution (OOD) samples, posing significant challenges for deployment. Since DNNs are trained on in-distribution (ID) datasets, the information flow of ID samples through DNNs inevitably differs from that of OOD samples. In this paper, we propose an Entropy-based Out-Of-distribution Detection (EOOD) framework. EOOD first identifies specific block where the information flow differences between ID and OOD samples are more pronounced, using both ID and pseudo-OOD samples. It then calculates the conditional entropy on the selected block as the OOD confidence score. Comprehensive experiments conducted across various ID and OOD settings demonstrate the effectiveness of EOOD in OOD detection and its superiority over state-of-the-art methods. 

**Abstract (ZH)**: 基于熵的异分布检测框架（Entropy-based Out-Of-distribution Detection, EOOD） 

---
# Mind the Prompt: Prompting Strategies in Audio Generations for Improving Sound Classification 

**Title (ZH)**: 留意提示：改进声音分类的音频生成提示策略 

**Authors**: Francesca Ronchini, Ho-Hsiang Wu, Wei-Cheng Lin, Fabio Antonacci  

**Link**: [PDF](https://arxiv.org/pdf/2504.03329)  

**Abstract**: This paper investigates the design of effective prompt strategies for generating realistic datasets using Text-To-Audio (TTA) models. We also analyze different techniques for efficiently combining these datasets to enhance their utility in sound classification tasks. By evaluating two sound classification datasets with two TTA models, we apply a range of prompt strategies. Our findings reveal that task-specific prompt strategies significantly outperform basic prompt approaches in data generation. Furthermore, merging datasets generated using different TTA models proves to enhance classification performance more effectively than merely increasing the training dataset size. Overall, our results underscore the advantages of these methods as effective data augmentation techniques using synthetic data. 

**Abstract (ZH)**: 本文研究了使用Text-To-Audio (TTA)模型生成现实数据集的有效提示策略设计。我们还分析了不同技术以高效地结合这些数据集，以增强其在声音分类任务中的实用性。通过使用两种TTA模型评估两种声音分类数据集，我们应用了一系列提示策略。我们的研究结果表明，专门针对任务的提示策略在数据生成方面显著优于基本提示方法。此外，使用不同TTA模型生成的数据集的合并被证明比单纯增加训练数据集规模更有效地提高分类性能。总体而言，我们的结果强调了这些方法作为使用合成数据的有效数据扩增技术的优势。 

---
# Policy Optimization Algorithms in a Unified Framework 

**Title (ZH)**: 统一框架下的策略优化算法 

**Authors**: Shuang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03328)  

**Abstract**: Policy optimization algorithms are crucial in many fields but challenging to grasp and implement, often due to complex calculations related to Markov decision processes and varying use of discount and average reward setups. This paper presents a unified framework that applies generalized ergodicity theory and perturbation analysis to clarify and enhance the application of these algorithms. Generalized ergodicity theory sheds light on the steady-state behavior of stochastic processes, aiding understanding of both discounted and average rewards. Perturbation analysis provides in-depth insights into the fundamental principles of policy optimization algorithms. We use this framework to identify common implementation errors and demonstrate the correct approaches. Through a case study on Linear Quadratic Regulator problems, we illustrate how slight variations in algorithm design affect implementation outcomes. We aim to make policy optimization algorithms more accessible and reduce their misuse in practice. 

**Abstract (ZH)**: 通用化 ergodic 理论和扰动分析在强化学习算法统一框架中的应用：提升算法理解和应用 

---
# Towards Effective EU E-Participation: The Development of AskThePublic 

**Title (ZH)**: 向有效的欧盟电子参与迈进：AskThePublic的发展 

**Authors**: Kilian Sprenkamp, Nils Messerschmidt, Amir Sartipi, Igor Tchappi, Xiaohui Wu, Liudmila Zavolokina, Gilbert Fridgen  

**Link**: [PDF](https://arxiv.org/pdf/2504.03287)  

**Abstract**: E-participation platforms can be an important asset for governments in increasing trust and fostering democratic societies. By engaging non-governmental and private institutions, domain experts, and even the general public, policymakers can make informed and inclusive decisions. Drawing on the Media Richness Theory and applying the Design Science Research method, we explore how a chatbot can be designed to improve the effectiveness of the policy-making process of existing citizen involvement platforms. Leveraging the Have Your Say platform, which solicits feedback on European Commission initiatives and regulations, a Large Language Model based chatbot, called AskThePublic is created, providing policymakers, journalists, researchers, and interested citizens with a convenient channel to explore and engage with public input. By conducting 11 semistructured interviews, the results show that the participants value the interactive and structured responses as well as enhanced language capabilities, thus increasing their likelihood of engaging with AskThePublic over the existing platform. An outlook for future iterations is provided and discussed with regard to the perspectives of the different stakeholders. 

**Abstract (ZH)**: 电子参与平台可以成为政府增加信任和促进民主社会的重要资产。通过与非政府和私营机构、领域专家，甚至普通公众互动，决策者可以作出更为明智和包容的决策。基于媒体丰富性理论和设计科学方法，我们探讨了如何设计聊天机器人以提高现有公民参与平台政策制定过程的有效性。利用征询公众意见平台（Have Your Say），该平台就欧洲委员会倡议和法规征求公众反馈，我们创建了一个基于大规模语言模型的聊天机器人AskThePublic，为决策者、记者、研究人员和感兴趣的公民提供了一个便捷的渠道来探索和参与公众意见。通过进行11次半结构化访谈，结果显示参与者认为交互式和结构化的回应以及增强的语言能力提高了他们使用AskThePublic平台的意愿。对未来迭代的展望与不同利益相关者的视角进行了讨论。 

---
# JanusDDG: A Thermodynamics-Compliant Model for Sequence-Based Protein Stability via Two-Fronts Multi-Head Attention 

**Title (ZH)**: JanusDDG: 一种基于序列的蛋白质稳定性thermodynamics合规模型通过两前沿多头注意力机制 

**Authors**: Guido Barducci, Ivan Rossi, Francesco Codicè, Cesare Rollo, Valeria Repetto, Corrado Pancotti, Virginia Iannibelli, Tiziana Sanavia, Piero Fariselli  

**Link**: [PDF](https://arxiv.org/pdf/2504.03278)  

**Abstract**: Understanding how residue variations affect protein stability is crucial for designing functional proteins and deciphering the molecular mechanisms underlying disease-related mutations. Recent advances in protein language models (PLMs) have revolutionized computational protein analysis, enabling, among other things, more accurate predictions of mutational effects. In this work, we introduce JanusDDG, a deep learning framework that leverages PLM-derived embeddings and a bidirectional cross-attention transformer architecture to predict $\Delta \Delta G$ of single and multiple-residue mutations while simultaneously being constrained to respect fundamental thermodynamic properties, such as antisymmetry and transitivity. Unlike conventional self-attention, JanusDDG computes queries (Q) and values (V) as the difference between wild-type and mutant embeddings, while keys (K) alternate between the two. This cross-interleaved attention mechanism enables the model to capture mutation-induced perturbations while preserving essential contextual information. Experimental results show that JanusDDG achieves state-of-the-art performance in predicting $\Delta \Delta G$ from sequence alone, matching or exceeding the accuracy of structure-based methods for both single and multiple mutations. 

**Abstract (ZH)**: 理解残基变异如何影响蛋白质稳定性对于设计功能性蛋白质和解析与疾病相关的突变分子机制至关重要。蛋白质语言模型（PLMs）的 recent 进展革命性地革新了蛋白质计算分析，使得更准确地预测突变效应成为可能。本文介绍了 JanusDDG，一个利用 PLM 提取的嵌入和双向交叉注意力变换器架构的深度学习框架，用于同时预测单个和多个残基突变的 $\Delta \Delta G$ 值，并遵守热力学基本性质如反对称性和传递性。与传统的自我注意力不同，JanusDDG 计算查询（Q）和值（V）为野生型和突变嵌入之差，而密钥（K）交替使用两者。这种交叉交错的注意力机制使模型能够捕捉突变引起的扰动同时保留重要上下文信息。实验结果表明，JanusDDG 在仅从序列预测 $\Delta \Delta G$ 方面达到了最先进的性能，对单突变和多突变的准确性与结构基于方法相当或超越。 

---
# Verification of Autonomous Neural Car Control with KeYmaera X 

**Title (ZH)**: 基于KeYmaera X对自主神经控制汽车系统的验证 

**Authors**: Enguerrand Prebet, Samuel Teuber, André Platzer  

**Link**: [PDF](https://arxiv.org/pdf/2504.03272)  

**Abstract**: This article presents a formal model and formal safety proofs for the ABZ'25 case study in differential dynamic logic (dL). The case study considers an autonomous car driving on a highway avoiding collisions with neighbouring cars. Using KeYmaera X's dL implementation, we prove absence of collision on an infinite time horizon which ensures that safety is preserved independently of trip length. The safety guarantees hold for time-varying reaction time and brake force. Our dL model considers the single lane scenario with cars ahead or behind. We demonstrate that dL with its tools is a rigorous foundation for runtime monitoring, shielding, and neural network verification. Doing so sheds light on inconsistencies between the provided specification and simulation environment highway-env of the ABZ'25 study. We attempt to fix these inconsistencies and uncover numerous counterexamples which also indicate issues in the provided reinforcement learning environment. 

**Abstract (ZH)**: 本文在微分动态逻辑（dL）框架下，提出了一种形式模型以及对ABZ'25案例研究中的形式安全证明。该案例研究考虑了一辆自动驾驶汽车在高速公路上行驶，避免与相邻车辆发生碰撞。利用KeYmaera X的dL实现，我们在无限时间 horizon 上证明了无碰撞条件，确保安全性独立于旅行距离得以保持。这些安全保证适用于时间变化的反应时间和制动强度。我们的dL模型考虑了一车道场景，其中包含前方或后方的车辆。本文证明了dL及其工具可以作为运行时监控、屏蔽和神经网络验证的严谨基础，并揭示了ABZ'25研究中提供规范和仿真环境highway-env之间的一致性问题。本文尝试修正这些问题，并发现了大量反例，这些反例还指出了提供强化学习环境中的问题。 

---
# An Extended Symbolic-Arithmetic Model for Teaching Double-Black Removal with Rotation in Red-Black Trees 

**Title (ZH)**: 扩展的符号算术模型：在红黑树中教学旋转删除双黑节点的方法 

**Authors**: Kennedy E. Ehimwenma, Hongyu Zhou, Junfeng Wang, Ze Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.03259)  

**Abstract**: Double-black (DB) nodes have no place in red-black (RB) trees. So when DB nodes are formed, they are immediately removed. The removal of DB nodes that cause rotation and recoloring of other connected nodes poses greater challenges in the teaching and learning of RB trees. To ease this difficulty, this paper extends our previous work on the symbolic arithmetic algebraic (SA) method for removing DB nodes. The SA operations that are given as, Red + Black = Black; Black - Black = Red; Black + Black = DB; and DB - Black = Black removes DB nodes and rebalances black heights in RB trees. By extension, this paper projects three SA mathematical equations, namely, general symbolic arithmetic rule; partial symbolic arithmetic rule1; and partial symbolic arithmetic rule2. The removal of a DB node ultimately affects black heights in RB trees. To balance black heights using the SA equations, all the RB tree cases, namely, LR, RL, LL, and RR, were considered in this work; and the position of the nodes connected directly or indirectly to the DB node was also tested. In this study, to balance a RB tree, the issues considered w.r.t. the different cases of the RB tree were i) whether a DB node has an inner, outer, or both inner and outer black nephews; or ii) whether a DB node has an inner, outer or both inner and outer red nephews. The nephews r and x in this work are the children of the sibling s to a DB, and further up the tree, the parent p of a DB is their grandparent g. Thus, r and x have indirect relationships to a DB at the point of formation of the DB node. The novelty of the SA equations is in their effectiveness in the removal of DB that involves rotation of nodes as well as the recoloring of nodes along any simple path so as to balance black heights in a tree. 

**Abstract (ZH)**: 双黑（DB）节点在红黑（RB）树中没有位置。因此，当形成DB节点时，会立即删除。删除DB节点导致的旋转和重新着色连接节点的任务增加了在教学和学习RB树时的难度。为简化这一难度，本文扩展了我们之前关于符号算术代数（SA）方法去除DB节点的工作。提供的SA运算包括：红 + 黑 = 黑；黑 - 黑 = 红；黑 + 黑 = 双黑；双黑 - 黑 = 黑，这些运算用于去除DB节点并重新平衡RB树中的黑高。进一步地，本文提出了三个SA数学方程，即：一般符号算术规则；部分符号算术规则1；和部分符号算术规则2。删除一个DB节点最终会影响RB树中的黑高。为了使用SA方程平衡黑高，本研究考虑了所有RB树案例，包括LR、RL、LL和RR，并且还测试了直接或间接连接到DB节点的节点的位置。在这项研究中，为了平衡RB树，考虑了不同RB树案例的问题，包括i) DB节点是否有内部、外部或两者的黑色侄节点；或ii) DB节点是否有内部、外部或两者的红色侄节点。在这个研究中，r和x是DB的兄弟节点s的子节点，而在更高级别的树中，DB的父节点p是它们的祖父节点g。因此，r和x在形成DB节点时与DB有间接关系。SA方程的创新之处在于它们在涉及节点旋转和沿任何简单路径重新着色以平衡树中的黑高方面的有效性。 

---
# Think When You Need: Self-Adaptive Chain-of-Thought Learning 

**Title (ZH)**: 需要时思考：自适应链式思维学习 

**Authors**: Junjie Yang, Ke Lin, Xing Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03234)  

**Abstract**: Chain of Thought (CoT) reasoning enhances language models' performance but often leads to inefficient "overthinking" on simple problems. We identify that existing approaches directly penalizing reasoning length fail to account for varying problem complexity. Our approach constructs rewards through length and quality comparisons, guided by theoretical assumptions that jointly enhance solution correctness with conciseness. Moreover, we further demonstrate our method to fuzzy tasks where ground truth is unavailable. Experiments across multiple reasoning benchmarks demonstrate that our method maintains accuracy while generating significantly more concise explanations, effectively teaching models to "think when needed." 

**Abstract (ZH)**: Chain of Thought (CoT)推理提高了语言模型的性能，但往往会针对简单问题进行无效的“过度思考”。我们发现，现有直接惩罚推理长度的方法未能考虑问题复杂度的差异。我们通过长度和质量比较构建奖励，基于理论假设以同时提升解决方案的正确性和简洁性。此外，我们还进一步展示了本方法在地面真相不可用的模糊任务中的应用效果。跨多个推理基准的实验表明，本方法在保持准确性的同时生成了明显更加简洁的解释，有效地教会模型“按需思考”。 

---
# Persuasive Calibration 

**Title (ZH)**: 说服性校准 

**Authors**: Yiding Feng, Wei Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03211)  

**Abstract**: We introduce and study the persuasive calibration problem, where a principal aims to provide trustworthy predictions about underlying events to a downstream agent to make desired decisions. We adopt the standard calibration framework that regulates predictions to be unbiased conditional on their own value, and thus, they can reliably be interpreted at the face value by the agent. Allowing a small calibration error budget, we aim to answer the following question: what is and how to compute the optimal predictor under this calibration error budget, especially when there exists incentive misalignment between the principal and the agent? We focus on standard Lt-norm Expected Calibration Error (ECE) metric.
We develop a general framework by viewing predictors as post-processed versions of perfectly calibrated predictors. Using this framework, we first characterize the structure of the optimal predictor. Specifically, when the principal's utility is event-independent and for L1-norm ECE, we show: (1) the optimal predictor is over-(resp. under-) confident for high (resp. low) true expected outcomes, while remaining perfectly calibrated in the middle; (2) the miscalibrated predictions exhibit a collinearity structure with the principal's utility function. On the algorithmic side, we provide a FPTAS for computing approximately optimal predictor for general principal utility and general Lt-norm ECE. Moreover, for the L1- and L-Infinity-norm ECE, we provide polynomial-time algorithms that compute the exact optimal predictor. 

**Abstract (ZH)**: 我们引入并研究了说服性校准问题，其中主要方旨在为下游代理提供关于潜在事件的可信预测，以便代理作出 desired 的决策。我们采用标准的校准框架，该框架要求预测在有条件偏差的情况下保持无偏，从而使代理可以可靠地按面值解读这些预测。允许校准误差的轻微预算，我们旨在回答以下问题：在存在主要方与代理之间激励不一致的情况下，在这种校准误差预算下最优预测器是什么，以及如何计算这一最优预测器？我们关注标准 Lt-范数期望校准误差（ECE）度量。我们开发了一个通用框架，将预测器视为完美校准预测器的后处理版本。利用这一框架，我们首先描述了最优预测器的结构。特别是在主要方的效用与事件无关且对于 L1-范数 ECE 时，我们证明：（1）最优预测器对于高（低）真实预期结果而言是过度（不足）自信的，而在中间保持完美校准；（2）非校准预测表现出与主要方效用函数的共线性结构。在算法方面，我们为一般主要方效用和一般 Lt-范数 ECE 提供了一个近似多项式时间算法来计算近最优预测器。此外，对于 L1-和 L-无穷范数 ECE，我们提供了计算精确最优预测器的多项式时间算法。 

---
# Enhancing Personalized Multi-Turn Dialogue with Curiosity Reward 

**Title (ZH)**: 增强个性化多轮对话的 Curiosity 奖励机制 

**Authors**: Yanming Wan, Jiaxing Wu, Marwa Abdulhai, Lior Shani, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2504.03206)  

**Abstract**: Effective conversational agents must be able to personalize their behavior to suit a user's preferences, personality, and attributes, whether they are assisting with writing tasks or operating in domains like education or healthcare. Current training methods like Reinforcement Learning from Human Feedback (RLHF) prioritize helpfulness and safety but fall short in fostering truly empathetic, adaptive, and personalized interactions. Traditional approaches to personalization often rely on extensive user history, limiting their effectiveness for new or context-limited users. To overcome these limitations, we propose to incorporate an intrinsic motivation to improve the conversational agents's model of the user as an additional reward alongside multi-turn RLHF. This reward mechanism encourages the agent to actively elicit user traits by optimizing conversations to increase the accuracy of its user model. Consequently, the policy agent can deliver more personalized interactions through obtaining more information about the user. We applied our method both education and fitness settings, where LLMs teach concepts or recommend personalized strategies based on users' hidden learning style or lifestyle attributes. Using LLM-simulated users, our approach outperformed a multi-turn RLHF baseline in revealing information about the users' preferences, and adapting to them. 

**Abstract (ZH)**: 有效的对话代理必须能够个性化其行为以适应用户的需求、个性和属性，无论是协助写作任务还是在教育或医疗等领域中操作。当前的训练方法如基于人类反馈的强化学习（RLHF）注重帮助性和安全性，但在培养真正具有同理心、适应性和个性化的互动方面存在不足。传统个性化方法往往依赖于用户的历史数据，这限制了其在新用户或上下文受限用户中的有效性。为了克服这些限制，我们建议在多回合RLHF的同时引入一种内在动机，以改进对话代理对用户的模型作为额外的奖励。这种奖励机制鼓励代理通过优化对话以增加其用户模型的准确性来主动获取用户的特征。因此，策略代理可以通过获取更多关于用户的详细信息来提供更加个性化的互动。我们在教育和健身环境中应用了这种方法，其中LLM根据用户的隐藏学习风格或生活方式属性教授概念或推荐个性化策略。使用LLM模拟用户，我们的方法在揭示用户偏好方面优于多回合RLHF基准模型，并能够更好地适应这些偏好。 

---
# Graph Network Modeling Techniques for Visualizing Human Mobility Patterns 

**Title (ZH)**: 图网络建模技术在可视化人类移动模式中的应用 

**Authors**: Sinjini Mitra, Anuj Srivastava, Avipsa Roy, Pavan Turaga  

**Link**: [PDF](https://arxiv.org/pdf/2504.03119)  

**Abstract**: Human mobility analysis at urban-scale requires models to represent the complex nature of human movements, which in turn are affected by accessibility to nearby points of interest, underlying socioeconomic factors of a place, and local transport choices for people living in a geographic region. In this work, we represent human mobility and the associated flow of movements as a grapyh. Graph-based approaches for mobility analysis are still in their early stages of adoption and are actively being researched. The challenges of graph-based mobility analysis are multifaceted - the lack of sufficiently high-quality data to represent flows at high spatial and teporal resolution whereas, limited computational resources to translate large voluments of mobility data into a network structure, and scaling issues inherent in graph models etc. The current study develops a methodology by embedding graphs into a continuous space, which alleviates issues related to fast graph matching, graph time-series modeling, and visualization of mobility dynamics. Through experiments, we demonstrate how mobility data collected from taxicab trajectories could be transformed into network structures and patterns of mobility flow changes, and can be used for downstream tasks reporting approx 40% decrease in error on average in matched graphs vs unmatched ones. 

**Abstract (ZH)**: 城市规模下的人类移动性分析需要能够代表人类移动复杂性的模型，这些模型受到附近兴趣点的可达性、地方的经济社会因素以及地理区域内居民的本地交通选择的影响。在本研究中，我们将人类移动性和相关移动流表示为图形。基于图形的方法在移动性分析中的应用尚处于早期阶段，且正受到广泛关注和研究。图形导向的移动性分析面临的挑战是多方面的——缺乏足够高质量的数据来表示高空间和时间分辨率的流动，计算资源限制了大规模移动性数据转化为网络结构的能力，以及图形模型固有的缩放问题等。当前研究开发了一种将图形嵌入连续空间的方法，以缓解快速图形匹配、图形时间序列建模以及移动性动态可视化等方面的问题。通过实验，我们展示了如何将来自出租车轨迹的移动数据转化为网络结构和移动流变化的模式，并且可以用于下游任务，结果显示在匹配的图形中与未匹配的图形相比，匹配错误率平均降低了约40%。 

---
# Post-processing for Fair Regression via Explainable SVD 

**Title (ZH)**: 通过可解释的SVD进行公平回归后处理 

**Authors**: Zhiqun Zuo, Ding Zhu, Mohammad Mahdi Khalili  

**Link**: [PDF](https://arxiv.org/pdf/2504.03093)  

**Abstract**: This paper presents a post-processing algorithm for training fair neural network regression models that satisfy statistical parity, utilizing an explainable singular value decomposition (SVD) of the weight matrix. We propose a linear transformation of the weight matrix, whereby the singular values derived from the SVD of the transformed matrix directly correspond to the differences in the first and second moments of the output distributions across two groups. Consequently, we can convert the fairness constraints into constraints on the singular values. We analytically solve the problem of finding the optimal weights under these constraints. Experimental validation on various datasets demonstrates that our method achieves a similar or superior fairness-accuracy trade-off compared to the baselines without using the sensitive attribute at the inference time. 

**Abstract (ZH)**: 基于可解释奇异值分解的公平神经网络回归模型后处理算法：利用权重矩阵的线性变换实现统计对等约束 

---
# Machine Learning-Based Detection and Analysis of Suspicious Activities in Bitcoin Wallet Transactions in the USA 

**Title (ZH)**: 基于机器学习的美国比特币钱包交易中可疑活动的检测与分析 

**Authors**: Md Zahidul Islam, Md Shahidul Islam, Biswajit Chandra das, Syed Ali Reza, Proshanta Kumar Bhowmik, Kanchon Kumar Bishnu, Md Shafiqur Rahman, Redoyan Chowdhury, Laxmi Pant  

**Link**: [PDF](https://arxiv.org/pdf/2504.03092)  

**Abstract**: The dramatic adoption of Bitcoin and other cryptocurrencies in the USA has revolutionized the financial landscape and provided unprecedented investment and transaction efficiency opportunities. The prime objective of this research project is to develop machine learning algorithms capable of effectively identifying and tracking suspicious activity in Bitcoin wallet transactions. With high-tech analysis, the study aims to create a model with a feature for identifying trends and outliers that can expose illicit activity. The current study specifically focuses on Bitcoin transaction information in America, with a strong emphasis placed on the importance of knowing about the immediate environment in and through which such transactions pass through. The dataset is composed of in-depth Bitcoin wallet transactional information, including important factors such as transaction values, timestamps, network flows, and addresses for wallets. All entries in the dataset expose information about financial transactions between wallets, including received and sent transactions, and such information is significant for analysis and trends that can represent suspicious activity. This study deployed three accredited algorithms, most notably, Logistic Regression, Random Forest, and Support Vector Machines. In retrospect, Random Forest emerged as the best model with the highest F1 Score, showcasing its ability to handle non-linear relationships in the data. Insights revealed significant patterns in wallet activity, such as the correlation between unredeemed transactions and final balances. The application of machine algorithms in tracking cryptocurrencies is a tool for creating transparent and secure U.S. markets. 

**Abstract (ZH)**: 美国比特币及其他加密货币的急剧采用已革新了金融格局，提供了前所未有的投资和交易效率机会。本研究项目的主要目标是开发能够有效识别和跟踪比特币钱包交易中可疑活动的机器学习算法。通过高科技分析，研究旨在创建一个具有识别趋势和异常值的功能的模型，以揭露非法活动。本研究特别关注美国的比特币交易信息，强调了解此类交易途径和环境的重要性。数据集包含深入的比特币钱包交易信息，包括交易值、时间戳、网络流和钱包地址等重要因素。数据集中所有条目都提供了关于钱包之间金融交易的信息，包括接收和发送的交易，这些信息对于分析和识别可疑活动趋势至关重要。本研究部署了三种认证算法，特别是逻辑回归、随机森林和支持向量机。回顾来看，随机森林模型以最高的F1分数脱颖而出，展示了其处理数据中非线性关系的能力。研究揭示了钱包活动中的显著模式，如未兑现交易与最终余额之间的关联。在追踪加密货币方面应用机器算法是创建透明和安全美国市场的工具。 

---
# From Questions to Insights: Exploring XAI Challenges Reported on Stack Overflow Questions 

**Title (ZH)**: 从问题到洞察：探索Stack Overflow问题中报告的XAI挑战 

**Authors**: Saumendu Roy, Saikat Mondal, Banani Roy, Chanchal Roy  

**Link**: [PDF](https://arxiv.org/pdf/2504.03085)  

**Abstract**: The lack of interpretability is a major barrier that limits the practical usage of AI models. Several eXplainable AI (XAI) techniques (e.g., SHAP, LIME) have been employed to interpret these models' performance. However, users often face challenges when leveraging these techniques in real-world scenarios and thus submit questions in technical Q&A forums like Stack Overflow (SO) to resolve these challenges. We conducted an exploratory study to expose these challenges, their severity, and features that can make XAI techniques more accessible and easier to use. Our contributions to this study are fourfold. First, we manually analyzed 663 SO questions that discussed challenges related to XAI techniques. Our careful investigation produced a catalog of seven challenges (e.g., disagreement issues). We then analyzed their prevalence and found that model integration and disagreement issues emerged as the most prevalent challenges. Second, we attempt to estimate the severity of each XAI challenge by determining the correlation between challenge types and answer metadata (e.g., the presence of accepted answers). Our analysis suggests that model integration issues is the most severe challenge. Third, we attempt to perceive the severity of these challenges based on practitioners' ability to use XAI techniques effectively in their work. Practitioners' responses suggest that disagreement issues most severely affect the use of XAI techniques. Fourth, we seek agreement from practitioners on improvements or features that could make XAI techniques more accessible and user-friendly. The majority of them suggest consistency in explanations and simplified integration. Our study findings might (a) help to enhance the accessibility and usability of XAI and (b) act as the initial benchmark that can inspire future research. 

**Abstract (ZH)**: 缺乏可解释性是限制AI模型实际应用的重大障碍。多种可解释AI(XAI)技术（例如SHAP、LIME）被用于解释这些模型的性能。然而，用户在实际应用场景中使用这些技术时常常面临挑战，并在技术问答论坛（如Stack Overflow）上提出问题以解决这些挑战。我们开展了探索性研究以揭示这些挑战、它们的严重程度以及能够使XAI技术更易于使用和获取的特征。本研究的贡献包括四个方面。首先，我们手动分析了663篇讨论与XAI技术相关的挑战的Stack Overflow问题，仔细调查并生成了七类挑战的清单（例如，分歧问题）。然后，我们分析了这些挑战的普遍性，发现模型整合问题和分歧问题是最常见的挑战。第二，我们尝试通过确定挑战类型与答案元数据（如接受答案的存在）的相关性来估计每类XAI挑战的严重程度。我们的分析表明，模型整合问题是最严重的挑战。第三，我们基于实践者在其工作中有效使用XAI技术的能力来感知这些挑战的严重性。实践者的回应表明，分歧问题对XAI技术的使用影响最大。第四，我们寻求实践者对可能使XAI技术更易于使用和用户友好的改进或特性的认同。大多数实践者建议保持解释的一致性和简化集成。本研究的发现可能有助于提高XAI的可访问性和易用性，并成为激励未来研究的初步基准。 

---
# Integrating Identity-Based Identification against Adaptive Adversaries in Federated Learning 

**Title (ZH)**: 针对适应性对手的联邦学习中基于身份的识别集成 

**Authors**: Jakub Kacper Szelag, Ji-Jian Chin, Lauren Ansell, Sook-Chin Yip  

**Link**: [PDF](https://arxiv.org/pdf/2504.03077)  

**Abstract**: Federated Learning (FL) has recently emerged as a promising paradigm for privacy-preserving, distributed machine learning. However, FL systems face significant security threats, particularly from adaptive adversaries capable of modifying their attack strategies to evade detection. One such threat is the presence of Reconnecting Malicious Clients (RMCs), which exploit FLs open connectivity by reconnecting to the system with modified attack strategies. To address this vulnerability, we propose integration of Identity-Based Identification (IBI) as a security measure within FL environments. By leveraging IBI, we enable FL systems to authenticate clients based on cryptographic identity schemes, effectively preventing previously disconnected malicious clients from re-entering the system. Our approach is implemented using the TNC-IBI (Tan-Ng-Chin) scheme over elliptic curves to ensure computational efficiency, particularly in resource-constrained environments like Internet of Things (IoT). Experimental results demonstrate that integrating IBI with secure aggregation algorithms, such as Krum and Trimmed Mean, significantly improves FL robustness by mitigating the impact of RMCs. We further discuss the broader implications of IBI in FL security, highlighting research directions for adaptive adversary detection, reputation-based mechanisms, and the applicability of identity-based cryptographic frameworks in decentralized FL architectures. Our findings advocate for a holistic approach to FL security, emphasizing the necessity of proactive defence strategies against evolving adaptive adversarial threats. 

**Abstract (ZH)**: 基于身份的识别在联邦学习中的应用：应对重新连接恶意客户端的安全措施 

---
# Properties of Fixed Points of Generalised Extra Gradient Methods Applied to Min-Max Problems 

**Title (ZH)**: 广义额外梯度方法应用于最小-最大问题的不动点性质 

**Authors**: Amir Ali Farzin, Yuen-Man Pun, Philipp Braun, Iman Shames  

**Link**: [PDF](https://arxiv.org/pdf/2504.03069)  

**Abstract**: This paper studies properties of fixed points of generalised Extra-gradient (GEG) algorithms applied to min-max problems. We discuss connections between saddle points of the objective function of the min-max problem and GEG fixed points. We show that, under appropriate step-size selections, the set of saddle points (Nash equilibria) is a subset of stable fixed points of GEG. Convergence properties of the GEG algorithm are obtained through a stability analysis of a discrete-time dynamical system. The results and benefits when compared to existing methods are illustrated through numerical examples. 

**Abstract (ZH)**: 本文研究了广义Extra-gradient（GEG）算法在解决极小极大问题中的不动点性质。我们讨论了极小极大问题目标函数鞍点与其GEG不动点之间的联系。我们证明，在适当步长选择下，鞍点集合（纳什均衡集）是GEG稳定不动点的子集。通过离散时间动态系统的稳定性分析获得了GEG算法的收敛性质。通过数值例子展示了与现有方法相比的结果和优势。 

---
# Context-Aware Self-Adaptation for Domain Generalization 

**Title (ZH)**: 基于上下文的自适应方法研究：领域泛化 

**Authors**: Hao Yan, Yuhong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2504.03064)  

**Abstract**: Domain generalization aims at developing suitable learning algorithms in source training domains such that the model learned can generalize well on a different unseen testing domain. We present a novel two-stage approach called Context-Aware Self-Adaptation (CASA) for domain generalization. CASA simulates an approximate meta-generalization scenario and incorporates a self-adaptation module to adjust pre-trained meta source models to the meta-target domains while maintaining their predictive capability on the meta-source domains. The core concept of self-adaptation involves leveraging contextual information, such as the mean of mini-batch features, as domain knowledge to automatically adapt a model trained in the first stage to new contexts in the second stage. Lastly, we utilize an ensemble of multiple meta-source models to perform inference on the testing domain. Experimental results demonstrate that our proposed method achieves state-of-the-art performance on standard benchmarks. 

**Abstract (ZH)**: 领域泛化旨在开发在源训练领域中有效的学习算法，使得模型能够在不同的未见测试领域中良好泛化。我们提出了一种名为Context-Aware Self-Adaptation (CASA)的新型两阶段方法，用于领域泛化。CASA模拟了近似元泛化场景，并集成了一个自我调整模块，在保留元源域预测能力的同时，将预训练的元源域模型调整到元目标域。自我调整的核心概念涉及利用上下文信息（如mini-batch特征的均值）作为领域知识，自动调整第一阶段训练的模型以适应第二阶段的新上下文。最后，我们使用多个元源域模型的集成在测试域中进行推理。实验结果表明，我们提出的方法在标准基准测试中取得了最先进的性能。 

---
# Safety Modulation: Enhancing Safety in Reinforcement Learning through Cost-Modulated Rewards 

**Title (ZH)**: 安全调节：通过成本调节奖励增强强化学习的安全性 

**Authors**: Hanping Zhang, Yuhong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2504.03040)  

**Abstract**: Safe Reinforcement Learning (Safe RL) aims to train an RL agent to maximize its performance in real-world environments while adhering to safety constraints, as exceeding safety violation limits can result in severe consequences. In this paper, we propose a novel safe RL approach called Safety Modulated Policy Optimization (SMPO), which enables safe policy function learning within the standard policy optimization framework through safety modulated rewards. In particular, we consider safety violation costs as feedback from the RL environments that are parallel to the standard awards, and introduce a Q-cost function as safety critic to estimate expected future cumulative costs. Then we propose to modulate the rewards using a cost-aware weighting function, which is carefully designed to ensure the safety limits based on the estimation of the safety critic, while maximizing the expected rewards. The policy function and the safety critic are simultaneously learned through gradient descent during online interactions with the environment. We conduct experiments using multiple RL environments and the experimental results demonstrate that our method outperforms several classic and state-of-the-art comparison methods in terms of overall safe RL performance. 

**Abstract (ZH)**: 安全强化学习（Safe RL）旨在训练一个RL代理在符合安全约束的情况下最大化其实用效果，因为超出安全违规限制可能导致严重后果。在本文中，我们提出了一种新颖的安全RL方法，称为安全性调节策略优化（SMPO），该方法可以通过安全性调节奖励在标准策略优化框架中实现安全策略函数的学习。特别地，我们将安全违规成本作为与标准奖励并行的RL环境反馈，并引入Q成本函数作为安全性评估器来估计预期的未来累积成本。然后，我们提出使用成本感知加权函数调节奖励，该函数旨在基于安全性评估器的估计确保安全限制，同时最大化预期奖励。策略函数和安全性评估器在与环境进行在线交互过程中同时通过梯度下降进行学习。我们在多个RL环境中进行了实验，并且实验结果表明，我们的方法在整体安全RL性能上优于几种经典和最先进的比较方法。 

---
# The Dual-Route Model of Induction 

**Title (ZH)**: 双重路径归纳模型 

**Authors**: Sheridan Feucht, Eric Todd, Byron Wallace, David Bau  

**Link**: [PDF](https://arxiv.org/pdf/2504.03022)  

**Abstract**: Prior work on in-context copying has shown the existence of induction heads, which attend to and promote individual tokens during copying. In this work we introduce a new type of induction head: concept-level induction heads, which copy entire lexical units instead of individual tokens. Concept induction heads learn to attend to the ends of multi-token words throughout training, working in parallel with token-level induction heads to copy meaningful text. We show that these heads are responsible for semantic tasks like word-level translation, whereas token induction heads are vital for tasks that can only be done verbatim, like copying nonsense tokens. These two "routes" operate independently: in fact, we show that ablation of token induction heads causes models to paraphrase where they would otherwise copy verbatim. In light of these findings, we argue that although token induction heads are vital for specific tasks, concept induction heads may be more broadly relevant for in-context learning. 

**Abstract (ZH)**: 基于上下文复制的先前工作已经表明存在引导头，它们在复制过程中关注并促进个别令牌。在这项工作中，我们引入了一种新的引导头类型：概念级引导头，它们复制整个词汇单元，而不仅仅是个别令牌。概念引导头在训练过程中学会关注多令牌词的末尾，并与令牌级引导头并行工作以复制有意义的文字。我们展示了这些引导头负责诸如词级翻译等语义任务，而令牌级引导头对于只能逐字完成的任务，如复制无意义的令牌，则至关重要。这两种“路径”是独立运作的：实际上，我们证明了去除令牌级引导头会导致模型在原本应逐字复制的地方进行改写。鉴于这些发现，我们认为虽然令牌级引导头对特定任务至关重要，但概念级引导头可能更广泛地适用于基于上下文的学习。 

---
# Localized Definitions and Distributed Reasoning: A Proof-of-Concept Mechanistic Interpretability Study via Activation Patching 

**Title (ZH)**: 局部定义与分布式推理：基于激活修补的机理可解释性概念验证研究 

**Authors**: Nooshin Bahador  

**Link**: [PDF](https://arxiv.org/pdf/2504.02976)  

**Abstract**: This study investigates the localization of knowledge representation in fine-tuned GPT-2 models using Causal Layer Attribution via Activation Patching (CLAP), a method that identifies critical neural layers responsible for correct answer generation. The model was fine-tuned on 9,958 PubMed abstracts (epilepsy: 20,595 mentions, EEG: 11,674 mentions, seizure: 13,921 mentions) using two configurations with validation loss monitoring for early stopping. CLAP involved (1) caching clean (correct answer) and corrupted (incorrect answer) activations, (2) computing logit difference to quantify model preference, and (3) patching corrupted activations with clean ones to assess recovery. Results revealed three findings: First, patching the first feedforward layer recovered 56% of correct preference, demonstrating that associative knowledge is distributed across multiple layers. Second, patching the final output layer completely restored accuracy (100% recovery), indicating that definitional knowledge is localised. The stronger clean logit difference for definitional questions further supports this localized representation. Third, minimal recovery from convolutional layer patching (13.6%) suggests low-level features contribute marginally to high-level reasoning. Statistical analysis confirmed significant layer-specific effects (p<0.01). These findings demonstrate that factual knowledge is more localized and associative knowledge depends on distributed representations. We also showed that editing efficacy depends on task type. Our findings not only reconcile conflicting observations about localization in model editing but also emphasize on using task-adaptive techniques for reliable, interpretable updates. 

**Abstract (ZH)**: 本研究使用因果层 attribution 通过激活补丁方法（CLAP）探究了微调后 GPT-2 模型中知识表示的局部化，该方法识别出负责正确答案生成的關鍵神经层。模型在包含 9,958 个PubMed摘要（癫痫：20,595 次提及，脑电图：11,674 次提及，发作：13,921 次提及）的数据集上进行了微调，并通过监控验证损失实现了早停。CLAP 包括以下步骤：(1) 缓存干净（正确答案）和受损（错误答案）的激活；(2) 计算 logits 差异以量化模型偏好；(3) 用干净激活替换受损激活以评估恢复情况。结果揭示了三项发现：首先，修复第一个前馈层恢复了 56% 的偏好，表明关联性知识分布在多个层中。其次，修复最终输出层完全恢复了准确性（100% 恢复），表明定义性知识是局部化的。定义性问题的 stronger clean logits 差异进一步支持了这种局部表示。第三，卷积层修复的 minimal 恢复（13.6%）表明低级特征对高级推理的贡献微乎其微。统计分析证实了特定层的显著效应（p<0.01）。这些发现表明，事实性知识更局部化，而关联性知识依赖于分布式表示。此外，我们还表明了编辑效果依赖于任务类型。我们的发现不仅弥合了关于模型编辑中局部化的矛盾观察，还强调了使用任务适配技术进行可靠且可解释的更新的重要性。 

---
# Improved Compact Genetic Algorithms with Efficient Caching 

**Title (ZH)**: 改进的高效缓存紧凑遗传算法 

**Authors**: Prasanta Dutta, Anirban Mukhopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2504.02972)  

**Abstract**: Compact Genetic Algorithms (cGAs) are condensed variants of classical Genetic Algorithms (GAs) that use a probability vector representation of the population instead of the complete population. cGAs have been shown to significantly reduce the number of function evaluations required while producing outcomes similar to those of classical GAs. However, cGAs have a tendency to repeatedly generate the same chromosomes as they approach convergence, resulting in unnecessary evaluations of identical chromosomes. This article introduces the concept of caching in cGAs as a means of avoiding redundant evaluations of the same chromosomes. Our proposed approach operates equivalently to cGAs, but enhances the algorithm's time efficiency by reducing the number of function evaluations. We also present a data structure for efficient cache maintenance to ensure low overhead. The proposed caching approach has an asymptotically constant time complexity on average. The proposed method further generalizes the caching mechanism with higher selection pressure for elitism-based cGAs. We conduct a rigorous analysis based on experiments on benchmark optimization problems using two well-known cache replacement strategies. The results demonstrate that caching significantly reduces the number of function evaluations required while maintaining the same level of performance accuracy. 

**Abstract (ZH)**: 基于缓存的紧凑遗传算法：提高时间效率并保持性能准确性 

---
# Global-Order GFlowNets 

**Title (ZH)**: 全球秩序GFlowNets 

**Authors**: Lluís Pastor-Pérez, Javier Alonso-Garcia, Lukas Mauch  

**Link**: [PDF](https://arxiv.org/pdf/2504.02968)  

**Abstract**: Order-Preserving (OP) GFlowNets have demonstrated remarkable success in tackling complex multi-objective (MOO) black-box optimization problems using stochastic optimization techniques. Specifically, they can be trained online to efficiently sample diverse candidates near the Pareto front. A key advantage of OP GFlowNets is their ability to impose a local order on training samples based on Pareto dominance, eliminating the need for scalarization - a common requirement in other approaches like Preference-Conditional GFlowNets. However, we identify an important limitation of OP GFlowNets: imposing a local order on training samples can lead to conflicting optimization objectives. To address this issue, we introduce Global-Order GFlowNets, which transform the local order into a global one, thereby resolving these conflicts. Our experimental evaluations on various benchmarks demonstrate the efficacy and promise of our proposed method. 

**Abstract (ZH)**: Order-Preserving (OP) GFlowNets在使用随机优化技术解决复杂的多目标（MOO）黑盒优化问题中取得了显著成功。具体而言，它们可以在线训练以高效地采样 Pareto 前沿附近的多样化候选解决方案。OP GFlowNets的一个关键优势是能够基于帕累托支配对训练样本施加局部顺序，从而消除其他方法（如偏好条件GFlowNets）中常见的标量化需求。然而，我们识别了OP GFlowNets的一个重要局限性：对训练样本施加局部顺序可能导致优化目标冲突。为了解决这一问题，我们引入了全局顺序GFlowNets，将局部顺序转换为全局顺序，从而解决这些冲突。我们在各种基准上的实验评估展示了我们提出方法的有效性和潜力。 

---
# Level Up Peer Review in Education: Investigating genAI-driven Gamification system and its influence on Peer Feedback Effectiveness 

**Title (ZH)**: 提升教育中的同行评审水平：探究由生成式AI驱动的游戏化系统及其对同行反馈有效性的影响 

**Authors**: Rafal Wlodarski, Leonardo da Silva Sousa, Allison Connell Pensky  

**Link**: [PDF](https://arxiv.org/pdf/2504.02962)  

**Abstract**: In software engineering (SE), the ability to review code and critique designs is essential for professional practice. However, these skills are rarely emphasized in formal education, and peer feedback quality and engagement can vary significantly among students. This paper introduces Socratique, a gamified peer-assessment platform integrated with Generative AI (GenAI) assistance, designed to develop students' peer-review skills in a functional programming course. By incorporating game elements, Socratique aims to motivate students to provide more feedback, while the GenAI assistant offers real-time support in crafting high quality, constructive comments. To evaluate the impact of this approach, we conducted a randomized controlled experiment with master's students comparing a treatment group with a gamified, GenAI-driven setup against a control group with minimal gamification. Results show that students in the treatment group provided significantly more voluntary feedback, with higher scores on clarity, relevance, and specificity - all key aspects of effective code and design reviews. This study provides evidence for the effectiveness of combining gamification and AI to improve peer review processes, with implications for fostering review-related competencies in software engineering curricula. 

**Abstract (ZH)**: 软件工程中，审查代码和批评设计的能力对于专业实践至关重要。然而，这些技能在正规教育中 rarely 被重视，同学之间的反馈质量和参与度差异显著。本文介绍了 Socratique，一个融入生成式人工智能 (GenAI) 助手的游戏化同伴评估平台，旨在通过功能编程课程提高学生的同行评审技能。通过融入游戏元素，Socratique 旨在激励学生提供更多反馈，而 GenAI 助手则提供即时支持，帮助编写高质量、建设性的评论。为了评估该方法的影响，我们对硕士生进行了随机对照实验，将接受游戏化和 GenAI 驱动设置的干预组与仅进行少量游戏化的对照组进行了比较。结果表明，干预组的学生提供了更多的自愿反馈，且在清晰度、相关性和具体性等方面得分更高——这些都是有效代码和设计审查的关键方面。本研究为结合游戏化和人工智能以改善同行评审过程的有效性提供了证据，并对软件工程课程中培养审查相关技能具有重要影响。 

---
# Graph Attention for Heterogeneous Graphs with Positional Encoding 

**Title (ZH)**: 基于位置编码的异构图注意力模型 

**Authors**: Nikhil Shivakumar Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2504.02938)  

**Abstract**: Graph Neural Networks (GNNs) have emerged as the de facto standard for modeling graph data, with attention mechanisms and transformers significantly enhancing their performance on graph-based tasks. Despite these advancements, the performance of GNNs on heterogeneous graphs often remains complex, with networks generally underperforming compared to their homogeneous counterparts. This work benchmarks various GNN architectures to identify the most effective methods for heterogeneous graphs, with a particular focus on node classification and link prediction. Our findings reveal that graph attention networks excel in these tasks. As a main contribution, we explore enhancements to these attention networks by integrating positional encodings for node embeddings. This involves utilizing the full Laplacian spectrum to accurately capture both the relative and absolute positions of each node within the graph, further enhancing performance on downstream tasks such as node classification and link prediction. 

**Abstract (ZH)**: 图神经网络（GNNs）已成为建模图数据的事实标准，注意力机制和变压器极大提升了其在图基任务上的性能。尽管取得了这些进展，GNNs在异构图上的性能仍然复杂，通常表现不如其同质版本。本研究对多种GNN架构进行了基准测试，以识别最适合异构图的有效方法，特别关注节点分类和链接预测。我们的研究表明，图注意力网络在这些任务上表现出色。作为主要贡献，我们通过集成节点嵌入的位置编码来探索这些注意力网络的增强方法，使用完整的拉普拉斯谱准确捕获图中每个节点的相对和绝对位置，进一步提升了下游任务如节点分类和链接预测的性能。 

---
# Robustly identifying concepts introduced during chat fine-tuning using crosscoders 

**Title (ZH)**: robustly识别聊天微调过程中引入的概念 Using Crosscoders 

**Authors**: Julian Minder, Clement Dumas, Caden Juang, Bilal Chugtai, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2504.02922)  

**Abstract**: Model diffing is the study of how fine-tuning changes a model's representations and internal algorithms. Many behaviours of interest are introduced during fine-tuning, and model diffing offers a promising lens to interpret such behaviors. Crosscoders are a recent model diffing method that learns a shared dictionary of interpretable concepts represented as latent directions in both the base and fine-tuned models, allowing us to track how concepts shift or emerge during fine-tuning. Notably, prior work has observed concepts with no direction in the base model, and it was hypothesized that these model-specific latents were concepts introduced during fine-tuning. However, we identify two issues which stem from the crosscoders L1 training loss that can misattribute concepts as unique to the fine-tuned model, when they really exist in both models. We develop Latent Scaling to flag these issues by more accurately measuring each latent's presence across models. In experiments comparing Gemma 2 2B base and chat models, we observe that the standard crosscoder suffers heavily from these issues. Building on these insights, we train a crosscoder with BatchTopK loss and show that it substantially mitigates these issues, finding more genuinely chat-specific and highly interpretable concepts. We recommend practitioners adopt similar techniques. Using the BatchTopK crosscoder, we successfully identify a set of genuinely chat-specific latents that are both interpretable and causally effective, representing concepts such as $\textit{false information}$ and $\textit{personal question}$, along with multiple refusal-related latents that show nuanced preferences for different refusal triggers. Overall, our work advances best practices for the crosscoder-based methodology for model diffing and demonstrates that it can provide concrete insights into how chat tuning modifies language model behavior. 

**Abstract (ZH)**: 跨模型差异分析是研究微调如何改变模型的表示和内部算法的研究。许多感兴趣的表征行为在微调过程中被引入，跨模型差异分析提供了一种有前景的方法来解释这些行为。跨模型差异分析方法Crosscoders最近通过在基模型和微调模型中学习一个可解释概念的共享字典，这些概念以潜在方向表示，使我们能够追踪概念在微调过程中如何变化或新出现。值得注意的是，先前的工作观察到基模型中没有方向的概念，并推测这些模型特异性潜在变量是在微调过程中引入的概念。然而，我们发现源自Crosscoders L1训练损失的两个问题可能导致错误地将这些概念归因于微调后的模型，而实际上这两个模型中都存在这些概念。我们开发了潜在缩放来通过更准确地测量每个潜在变量在模型中的存在情况来标记这些问题。在将Gemma 2 2B基模型和聊天模型进行比较的实验中，我们观察到标准的Crosscoders严重受到这些问题的影响。基于这些洞见，我们训练了一个使用BatchTopK损失的Crosscoders，并展示了它显著缓解了这些问题，发现了更多真正属于聊天模型和高度可解释的概念。我们建议研究人员采用类似的技术。使用BatchTopK Crosscoders，我们成功地识别出一组真正属于聊天模型的潜在变量，这些潜在变量不仅是可解释的，而且具有因果效果，代表了诸如“虚假信息”和“私人问题”等概念，以及显示不同拒绝触发器偏好变化的多个拒绝相关的潜在变量。总体而言，我们的工作推进了基于Crosscoders的方法在模型差异分析中的最佳实践，并证明了这种方法可以提供关于聊天微调如何修改语言模型行为的具体见解。 

---
# Haphazard Inputs as Images in Online Learning 

**Title (ZH)**: 随机输入作为在线学习中的图像 

**Authors**: Rohit Agarwal, Aryan Dessai, Arif Ahmed Sekh, Krishna Agarwal, Alexander Horsch, Dilip K. Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2504.02912)  

**Abstract**: The field of varying feature space in online learning settings, also known as haphazard inputs, is very prominent nowadays due to its applicability in various fields. However, the current solutions to haphazard inputs are model-dependent and cannot benefit from the existing advanced deep-learning methods, which necessitate inputs of fixed dimensions. Therefore, we propose to transform the varying feature space in an online learning setting to a fixed-dimension image representation on the fly. This simple yet novel approach is model-agnostic, allowing any vision-based models to be applicable for haphazard inputs, as demonstrated using ResNet and ViT. The image representation handles the inconsistent input data seamlessly, making our proposed approach scalable and robust. We show the efficacy of our method on four publicly available datasets. The code is available at this https URL. 

**Abstract (ZH)**: 在线学习环境中变量特征空间的处理：从乱序输入到固定维度图像表示的转换 

---
# Meat-Free Day Reduces Greenhouse Gas Emissions but Poses Challenges for Customer Retention and Adherence to Dietary Guidelines 

**Title (ZH)**: 无肉日减少温室气体排放但面临客户留存和饮食指南遵守的挑战 

**Authors**: Giuseppe Russo, Kristina Gligorić, Vincent Moreau, Robert West  

**Link**: [PDF](https://arxiv.org/pdf/2504.02899)  

**Abstract**: Reducing meat consumption is crucial for achieving global environmental and nutritional targets. Meat-Free Day (MFD) is a widely adopted strategy to address this challenge by encouraging plant-based diets through the removal of animal-based meals. We assessed the environmental, behavioral, and nutritional impacts of MFD by implementing 67 MFDs over 18 months (once a week on a randomly chosen day) across 12 cafeterias on a large university campus, analyzing over 400,000 food purchases. MFD reduced on-campus food-related greenhouse gas (GHG) emissions on treated days by 52.9% and contributed to improved fiber (+26.9%) and cholesterol (-4.5%) consumption without altering caloric intake. These nutritional benefits were, however, accompanied by a 27.6% decrease in protein intake and a 34.2% increase in sugar consumption. Moreover, the increase in plant-based meals did not carry over to subsequent days, as evidenced by a 3.5% rebound in animal-based meal consumption on days immediately following treated days. MFD also led to a 16.8% drop in on-campus meal sales on treated this http URL Carlo simulations suggest that if 8.7% of diners were to eat burgers off-campus on treated days, MFD's GHG savings would be fully negated. As our analysis identifies on-campus customer retention as the main challenge to MFD effectiveness, we recommend combining MFD with customer retention interventions to ensure environmental and nutritional benefits. 

**Abstract (ZH)**: 减少肉类消费对于实现全球环境和营养目标至关重要。"无肉日"（MFD）通过推广植物基饮食来应对这一挑战，研究表明，实施MFD可以降低校内与食物相关的温室气体排放，改善纤维和胆固醇的摄入，同时不影响热量摄入。然而，这也伴随着蛋白质摄入量的下降和糖分摄入量的增加。此外，植物基餐食的增加并未持续到后续日子，表明在MFD后立即消费的动物基餐食有所回升。模拟结果显示，如果8.7%的就餐者在MFD在外就餐，将抵消MFD的温室气体减排效果。鉴于分析指出在校顾客留存是MFD效果的主要挑战，建议结合顾客留存措施以确保环境和营养效益。 

---
# UAC: Uncertainty-Aware Calibration of Neural Networks for Gesture Detection 

**Title (ZH)**: UAC：面向手势检测的不确定性意识校准的神经网络 

**Authors**: Farida Al Haddad, Yuxin Wang, Malcolm Mielle  

**Link**: [PDF](https://arxiv.org/pdf/2504.02895)  

**Abstract**: Artificial intelligence has the potential to impact safety and efficiency in safety-critical domains such as construction, manufacturing, and healthcare. For example, using sensor data from wearable devices, such as inertial measurement units (IMUs), human gestures can be detected while maintaining privacy, thereby ensuring that safety protocols are followed. However, strict safety requirements in these domains have limited the adoption of AI, since accurate calibration of predicted probabilities and robustness against out-of-distribution (OOD) data is necessary.
This paper proposes UAC (Uncertainty-Aware Calibration), a novel two-step method to address these challenges in IMU-based gesture recognition. First, we present an uncertainty-aware gesture network architecture that predicts both gesture probabilities and their associated uncertainties from IMU data. This uncertainty is then used to calibrate the probabilities of each potential gesture. Second, an entropy-weighted expectation of predictions over multiple IMU data windows is used to improve accuracy while maintaining correct calibration.
Our method is evaluated using three publicly available IMU datasets for gesture detection and is compared to three state-of-the-art calibration methods for neural networks: temperature scaling, entropy maximization, and Laplace approximation. UAC outperforms existing methods, achieving improved accuracy and calibration in both OOD and in-distribution scenarios. Moreover, we find that, unlike our method, none of the state-of-the-art methods significantly improve the calibration of IMU-based gesture recognition models. In conclusion, our work highlights the advantages of uncertainty-aware calibration of neural networks, demonstrating improvements in both calibration and accuracy for gesture detection using IMU data. 

**Abstract (ZH)**: 人工智能有潜力在建筑、制造和医疗等关键安全领域影响安全性和效率。例如，通过穿戴设备（如惯性测量单元IMU）的传感器数据，可以在保护隐私的同时检测人类手势，从而确保遵守安全规范。然而，这些领域严格的安全要求限制了人工智能的应用，因为准确的预测概率校准和对分布外（OOD）数据的鲁棒性是必要的。

本文提出了一种新颖的两步方法UAC（不确定性感知校准）来解决基于IMU的手势识别中的这些挑战。首先，我们提出了一种不确定性感知的手势网络架构，该架构可以从IMU数据中预测手势的概率及其相关不确定性。然后，使用这些不确定性来校准每种潜在手势的概率。其次，通过在多个IMU数据窗口上的熵加权预测期望来提高准确性同时保持正确的校准。

我们的方法使用三个公开可用的IMU数据集进行了手势检测的评估，并与三种最先进的神经网络校准方法（温度缩放、熵最大化、拉普拉斯近似）进行了比较。UAC在OOD和同分布场景中均实现了更好的准确性和校准，此外，我们发现与我们的方法不同，最先进的方法没有在基于IMU的手势识别模型的校准方面显著提高。总之，我们的工作突显了不确定性感知校准神经网络的优势，展示了使用IMU数据进行手势检测时在校准和准确性方面的改进。 

---
# Embedding Method for Knowledge Graph with Densely Defined Ontology 

**Title (ZH)**: 基于密集定义本体的知识图嵌入方法 

**Authors**: Takanori Ugai  

**Link**: [PDF](https://arxiv.org/pdf/2504.02889)  

**Abstract**: Knowledge graph embedding (KGE) is a technique that enhances knowledge graphs by addressing incompleteness and improving knowledge retrieval. A limitation of the existing KGE models is their underutilization of ontologies, specifically the relationships between properties. This study proposes a KGE model, TransU, designed for knowledge graphs with well-defined ontologies that incorporate relationships between properties. The model treats properties as a subset of entities, enabling a unified representation. We present experimental results using a standard dataset and a practical dataset. 

**Abstract (ZH)**: 基于本体的知识图嵌入模型（TransU）：面向具定义关系属性的知识图谱 

---
# Scraping the Shadows: Deep Learning Breakthroughs in Dark Web Intelligence 

**Title (ZH)**: 刮开阴影：暗网intelligence深度学习突破 

**Authors**: Ingmar Bakermans, Daniel De Pascale, Gonçalo Marcelino, Giuseppe Cascavilla, Zeno Geradts  

**Link**: [PDF](https://arxiv.org/pdf/2504.02872)  

**Abstract**: Darknet markets (DNMs) facilitate the trade of illegal goods on a global scale. Gathering data on DNMs is critical to ensuring law enforcement agencies can effectively combat crime. Manually extracting data from DNMs is an error-prone and time-consuming task. Aiming to automate this process we develop a framework for extracting data from DNMs and evaluate the application of three state-of-the-art Named Entity Recognition (NER) models, ELMo-BiLSTM \citep{ShahEtAl2022}, UniversalNER \citep{ZhouEtAl2024}, and GLiNER \citep{ZaratianaEtAl2023}, at the task of extracting complex entities from DNM product listing pages. We propose a new annotated dataset, which we use to train, fine-tune, and evaluate the models. Our findings show that state-of-the-art NER models perform well in information extraction from DNMs, achieving 91% Precision, 96% Recall, and an F1 score of 94%. In addition, fine-tuning enhances model performance, with UniversalNER achieving the best performance. 

**Abstract (ZH)**: Darknet市场(DNMs)促进了全球非法商品的交易。收集DNM数据是确保执法机构能有效打击犯罪的关键。手动从DNM中提取数据是一项易出错且耗时的任务。为了自动化这一过程，我们开发了一个从DNM中抽取数据的框架，并评估了三种最先进的命名实体识别(NER)模型，即ELMo-BiLSTM、UniversalNER和GLiNER，在从DNM产品列表页抽取复杂实体任务中的应用。我们提出一个新的标注数据集，用于训练、微调和评估这些模型。我们的研究发现，最先进的NER模型在从DNM中提取信息方面表现良好，精度为91%，召回率为96%，F1分为94%。此外，模型微调提升了性能，UniversalNER表现最佳。 

---
# Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks 

**Title (ZH)**: 使用扩散图网络学习复杂流体模拟的分布 

**Authors**: Mario Lino, Tobias Pfaff, Nils Thuerey  

**Link**: [PDF](https://arxiv.org/pdf/2504.02843)  

**Abstract**: Physical systems with complex unsteady dynamics, such as fluid flows, are often poorly represented by a single mean solution. For many practical applications, it is crucial to access the full distribution of possible states, from which relevant statistics (e.g., RMS and two-point correlations) can be derived. Here, we propose a graph-based latent diffusion (or alternatively, flow-matching) model that enables direct sampling of states from their equilibrium distribution, given a mesh discretization of the system and its physical parameters. This allows for the efficient computation of flow statistics without running long and expensive numerical simulations. The graph-based structure enables operations on unstructured meshes, which is critical for representing complex geometries with spatially localized high gradients, while latent-space diffusion modeling with a multi-scale GNN allows for efficient learning and inference of entire distributions of solutions. A key finding is that the proposed networks can accurately learn full distributions even when trained on incomplete data from relatively short simulations. We apply this method to a range of fluid dynamics tasks, such as predicting pressure distributions on 3D wing models in turbulent flow, demonstrating both accuracy and computational efficiency in challenging scenarios. The ability to directly sample accurate solutions, and capturing their diversity from short ground-truth simulations, is highly promising for complex scientific modeling tasks. 

**Abstract (ZH)**: 具有复杂非稳态动力学的物理系统，如流体流动，往往无法通过单一的平均解来良好表示。对于许多实际应用而言，访问所有可能状态的完整分布至关重要，从中可以推导出相关统计量（例如，均方根值和两点相关性）。在这里，我们提出了一种基于图的潜在扩散（或等价地，流匹配）模型，该模型可以在给定系统及其物理参数的网格离散化的情况下直接从稳态分布中抽样状态。这使得可以在不运行长时间和昂贵的数值模拟的情况下高效计算流场统计量。基于图的结构能够对无结构网格进行操作，这对于表示具有空间局部高梯度的复杂几何形状至关重要，而多尺度GNN的潜在空间扩散建模则允许高效地学习和推断整个解的分布。一个关键发现是，所提出的网络即使在基于相对较短模拟的不完整数据训练时也能准确地学习完整的分布。我们应用此方法解决了一系列流体动力学任务，例如在湍流流动中预测三维机翼模型的压力分布，证明了在挑战性场景中其准确性和计算效率。直接从简短的真实数据模拟中抽样准确的解并捕获其多样性，对于复杂的科学技术建模任务具有高度的前景。 

---
# A First-Principles Based Risk Assessment Framework and the IEEE P3396 Standard 

**Title (ZH)**: 基于第一性原理的风险评估框架和IEEE P3396标准 

**Authors**: Richard J. Tong, Marina Cortês, Jeanine A. DeFalco, Mark Underwood, Janusz Zalewski  

**Link**: [PDF](https://arxiv.org/pdf/2504.00091)  

**Abstract**: Generative Artificial Intelligence (AI) is enabling unprecedented automation in content creation and decision support, but it also raises novel risks. This paper presents a first-principles risk assessment framework underlying the IEEE P3396 Recommended Practice for AI Risk, Safety, Trustworthiness, and Responsibility. We distinguish between process risks (risks arising from how AI systems are built or operated) and outcome risks (risks manifest in the AI system's outputs and their real-world effects), arguing that generative AI governance should prioritize outcome risks. Central to our approach is an information-centric ontology that classifies AI-generated outputs into four fundamental categories: (1) Perception-level information, (2) Knowledge-level information, (3) Decision/Action plan information, and (4) Control tokens (access or resource directives). This classification allows systematic identification of harms and more precise attribution of responsibility to stakeholders (developers, deployers, users, regulators) based on the nature of the information produced. We illustrate how each information type entails distinct outcome risks (e.g. deception, misinformation, unsafe recommendations, security breaches) and requires tailored risk metrics and mitigations. By grounding the framework in the essence of information, human agency, and cognition, we align risk evaluation with how AI outputs influence human understanding and action. The result is a principled approach to AI risk that supports clear accountability and targeted safeguards, in contrast to broad application-based risk categorizations. We include example tables mapping information types to risks and responsibilities. This work aims to inform the IEEE P3396 Recommended Practice and broader AI governance with a rigorous, first-principles foundation for assessing generative AI risks while enabling responsible innovation. 

**Abstract (ZH)**: Generative人工智能（AI）正在实现内容创造和决策支持前所未有的自动化，但也带来了新的风险。本文提出了一种基于IEEE P3396推荐实践的AI风险、安全、可靠性和责任的第一性原理风险评估框架。本文将风险分为过程风险（AI系统构建或运行方式引起的风险）和结果风险（AI系统输出及其现实影响中的风险），认为生成型AI治理应优先考虑结果风险。我们方法的核心是一种以信息为中心的概念分类，将AI生成的输出分为四大基本类别：（1）感知级信息，（2）知识级信息，（3）决策/行动方案信息，（4）控制标记（访问或资源指令）。这种分类有助于系统地识别危害，并根据所产生信息的性质更精确地将责任归咎于相关方（开发者、部署者、用户、监管者）。每种信息类型都伴随着不同的结果风险（例如，误导、虚假信息、不安全的建议、安全漏洞），需要特定的风险度量和缓解措施。通过将框架建立在信息的本质、人类代理和认知的基础上，我们将风险评估与AI输出如何影响人类理解与行动相结合。由此形成一种基于原则的AI风险管理方法，支持明确的责任分配和针对性的保障措施，与基于应用的广泛风险分类不同。本文还提供了示例表格，映射信息类型和风险、责任。本研究旨在为IEEE P3396推荐实践和更广泛的AI治理提供一个严谨的第一性原理基础，以评估生成型AI风险，同时促进负责任的创新。 

---
