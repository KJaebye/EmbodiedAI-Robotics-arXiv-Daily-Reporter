# Hermes 4 Technical Report 

**Title (ZH)**: 赫梅斯4技术报告 

**Authors**: Ryan Teknium, Roger Jin, Jai Suphavadeeprasit, Dakota Mahan, Jeffrey Quesnelle, Joe Li, Chen Guang, Shannon Sands, Karan Malhotra  

**Link**: [PDF](https://arxiv.org/pdf/2508.18255)  

**Abstract**: We present Hermes 4, a family of hybrid reasoning models that combine structured, multi-turn reasoning with broad instruction-following ability. We describe the challenges encountered during data curation, synthesis, training, and evaluation, and outline the solutions employed to address these challenges at scale. We comprehensively evaluate across mathematical reasoning, coding, knowledge, comprehension, and alignment benchmarks, and we report both quantitative performance and qualitative behavioral analysis. To support open research, all model weights are published publicly at this https URL 

**Abstract (ZH)**: 我们介绍Hermes 4，这是一种结合结构化多轮推理和广泛指令遵循能力的混合推理模型系列。我们在数据整理、合成、训练和评估过程中遇到的挑战进行了描述，并概述了大规模解决这些挑战所采用的解决方案。我们在数学推理、编码、知识、理解及对齐基准上进行了全面评估，并报告了定量性能和定性行为分析结果。为了支持开放研究，所有模型权重已在以下网址公开发布：https://github.com/alibaba/Hermes。 

---
# Efficient Computation of Blackwell Optimal Policies using Rational Functions 

**Title (ZH)**: 使用有理函数计算Blackwell最优策略的高效方法 

**Authors**: Dibyangshu Mukherjee, Shivaram Kalyanakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2508.18252)  

**Abstract**: Markov Decision Problems (MDPs) provide a foundational framework for modelling sequential decision-making across diverse domains, guided by optimality criteria such as discounted and average rewards. However, these criteria have inherent limitations: discounted optimality may overly prioritise short-term rewards, while average optimality relies on strong structural assumptions. Blackwell optimality addresses these challenges, offering a robust and comprehensive criterion that ensures optimality under both discounted and average reward frameworks. Despite its theoretical appeal, existing algorithms for computing Blackwell Optimal (BO) policies are computationally expensive or hard to implement.
In this paper we describe procedures for computing BO policies using an ordering of rational functions in the vicinity of $1$. We adapt state-of-the-art algorithms for deterministic and general MDPs, replacing numerical evaluations with symbolic operations on rational functions to derive bounds independent of bit complexity. For deterministic MDPs, we give the first strongly polynomial-time algorithms for computing BO policies, and for general MDPs we obtain the first subexponential-time algorithm. We further generalise several policy iteration algorithms, extending the best known upper bounds from the discounted to the Blackwell criterion. 

**Abstract (ZH)**: 马尔可夫决策问题（MDPs）为跨不同领域的序贯决策建模提供了一个基础框架，这些决策由折现奖励和平均奖励等最优性标准指导。然而，这些标准存在固有的局限性：折现最优性可能过度优先考虑短期奖励，而平均最优性则依赖于强有力的结构假设。Blackwell最优性解决了这些挑战，提供了一个在折现和平均奖励框架下都能确保最优性的稳健和全面的标准。尽管在理论上具有吸引力，现有的计算Blackwell最优（BO）策略的算法在计算上可能非常昂贵或难以实现。

在本文中，我们描述了使用接近1的有理函数排序来计算BO策略的过程。我们适应了最先进的确定性和通用MDP算法，将数值评估替换为有理函数的符号操作，从而得到与位复杂度无关的界。对于确定性MDP，我们首次提出了计算BO策略的强多项式时间算法，并对于通用MDP，我们获得了首个次指数时间算法。我们还进一步推广了多种策略迭代算法，将已知的最佳上界从折现标准扩展到Blackwell标准。 

---
# Disentangling the Factors of Convergence between Brains and Computer Vision Models 

**Title (ZH)**: 解开大脑与计算机视觉模型收敛因素的分离分析 

**Authors**: Joséphine Raugel, Marc Szafraniec, Huy V. Vo, Camille Couprie, Patrick Labatut, Piotr Bojanowski, Valentin Wyart, Jean-Rémi King  

**Link**: [PDF](https://arxiv.org/pdf/2508.18226)  

**Abstract**: Many AI models trained on natural images develop representations that resemble those of the human brain. However, the factors that drive this brain-model similarity remain poorly understood. To disentangle how the model, training and data independently lead a neural network to develop brain-like representations, we trained a family of self-supervised vision transformers (DINOv3) that systematically varied these different factors. We compare their representations of images to those of the human brain recorded with both fMRI and MEG, providing high resolution in spatial and temporal analyses. We assess the brain-model similarity with three complementary metrics focusing on overall representational similarity, topographical organization, and temporal dynamics. We show that all three factors - model size, training amount, and image type - independently and interactively impact each of these brain similarity metrics. In particular, the largest DINOv3 models trained with the most human-centric images reach the highest brain-similarity. This emergence of brain-like representations in AI models follows a specific chronology during training: models first align with the early representations of the sensory cortices, and only align with the late and prefrontal representations of the brain with considerably more training. Finally, this developmental trajectory is indexed by both structural and functional properties of the human cortex: the representations that are acquired last by the models specifically align with the cortical areas with the largest developmental expansion, thickness, least myelination, and slowest timescales. Overall, these findings disentangle the interplay between architecture and experience in shaping how artificial neural networks come to see the world as humans do, thus offering a promising framework to understand how the human brain comes to represent its visual world. 

**Abstract (ZH)**: 许多在自然图像上训练的AI模型形成了与人类大脑相似的表示。然而，驱动这种大脑模型相似性的因素仍然知之甚少。为了独立地分析模型、训练和数据如何引导神经网络发展出类似大脑的表示，我们训练了一种系统地变化这些不同因素的自监督视觉变压器（DINOv3）家族。我们将它们对图像的表示与使用fMRI和MEG记录的人类大脑图像进行比较，提供高分辨率的空间和时间分析。我们使用三个互补的度量标准来评估大脑模型相似性，这些度量标准分别关注整体表示相似性、拓扑组织和时间动态。我们展示了模型大小、训练量和图像类型这三个因素如何各自独立地并相互作用地影响这些大脑相似性度量。特别是，用最以人类为中心的图像训练的最大DINOv3模型达到最高的大脑相似度。这些类似大脑的表示在训练中出现了特定的顺序：模型首先与感觉皮层的早期表示对齐，只有在大量训练后才与脑的晚期和前额皮层的表示对齐。最终，这一发育轨迹由人类皮层的结构和功能特性指数化：由模型最后获得的表示特异性地与发育扩张最大、厚度最大、髓鞘化最少和时间尺度最慢的皮层区域对齐。总体而言，这些发现解缠了架构和经验在塑造人工神经网络如何以人类方式看待世界之间的互动关系，从而为理解人类大脑如何表示其视觉世界提供了一个有前景的框架。 

---
# Unraveling the cognitive patterns of Large Language Models through module communities 

**Title (ZH)**: 剖析大型语言模型的认知模式通过模块社区 

**Authors**: Kushal Raj Bhandari, Pin-Yu Chen, Jianxi Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.18192)  

**Abstract**: Large Language Models (LLMs) have reshaped our world with significant advancements in science, engineering, and society through applications ranging from scientific discoveries and medical diagnostics to Chatbots. Despite their ubiquity and utility, the underlying mechanisms of LLM remain concealed within billions of parameters and complex structures, making their inner architecture and cognitive processes challenging to comprehend. We address this gap by adopting approaches to understanding emerging cognition in biology and developing a network-based framework that links cognitive skills, LLM architectures, and datasets, ushering in a paradigm shift in foundation model analysis. The skill distribution in the module communities demonstrates that while LLMs do not strictly parallel the focalized specialization observed in specific biological systems, they exhibit unique communities of modules whose emergent skill patterns partially mirror the distributed yet interconnected cognitive organization seen in avian and small mammalian brains. Our numerical results highlight a key divergence from biological systems to LLMs, where skill acquisition benefits substantially from dynamic, cross-regional interactions and neural plasticity. By integrating cognitive science principles with machine learning, our framework provides new insights into LLM interpretability and suggests that effective fine-tuning strategies should leverage distributed learning dynamics rather than rigid modular interventions. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过在科学、工程和社会领域的进步重塑了我们的世界，其应用范围从科学发现和医学诊断到聊天机器人。尽管它们无处不在且功能强大，但大型语言模型的内在工作原理仍然隐藏在其数十亿参数和复杂结构之中，使其认知机制难以理解。我们通过借鉴生物学中新兴认知的理解方法，并开发一种基于网络的框架，将认知技能、LLM架构和数据集联系起来，引领了基础模型分析的范式转变。模块社区中的技能分布表明，尽管LLMs并不严格遵循特定生物系统中的聚焦专业化现象，但它们展现出独特的模块社区，其涌现的技能模式部分反映了鸟类和小型哺乳动物大脑中分散而相互连接的认知组织。我们的数值结果强调了生物学系统与LLMs之间的一个关键差异，技能获取显著受益于动态的、跨区域的交互以及神经可塑性。通过结合认知科学原理与机器学习，我们的框架为大型语言模型的可解释性提供了新的见解，并建议有效的微调策略应利用分布式学习动态而非僵化的模块化干预。 

---
# ST-Raptor: LLM-Powered Semi-Structured Table Question Answering 

**Title (ZH)**: ST-Raptor：LLM驱动的半结构化表格问答 

**Authors**: Zirui Tang, Boyu Niu, Xuanhe Zhou, Boxiu Li, Wei Zhou, Jiannan Wang, Guoliang Li, Xinyi Zhang, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18190)  

**Abstract**: Semi-structured tables, widely used in real-world applications (e.g., financial reports, medical records, transactional orders), often involve flexible and complex layouts (e.g., hierarchical headers and merged cells). These tables generally rely on human analysts to interpret table layouts and answer relevant natural language questions, which is costly and inefficient. To automate the procedure, existing methods face significant challenges. First, methods like NL2SQL require converting semi-structured tables into structured ones, which often causes substantial information loss. Second, methods like NL2Code and multi-modal LLM QA struggle to understand the complex layouts of semi-structured tables and cannot accurately answer corresponding questions. To this end, we propose ST-Raptor, a tree-based framework for semi-structured table question answering using large language models. First, we introduce the Hierarchical Orthogonal Tree (HO-Tree), a structural model that captures complex semi-structured table layouts, along with an effective algorithm for constructing the tree. Second, we define a set of basic tree operations to guide LLMs in executing common QA tasks. Given a user question, ST-Raptor decomposes it into simpler sub-questions, generates corresponding tree operation pipelines, and conducts operation-table alignment for accurate pipeline execution. Third, we incorporate a two-stage verification mechanism: forward validation checks the correctness of execution steps, while backward validation evaluates answer reliability by reconstructing queries from predicted answers. To benchmark the performance, we present SSTQA, a dataset of 764 questions over 102 real-world semi-structured tables. Experiments show that ST-Raptor outperforms nine baselines by up to 20% in answer accuracy. The code is available at this https URL. 

**Abstract (ZH)**: 基于树结构的大型语言模型半结构化表格问答框架：ST-Raptor 

---
# SEAM: Semantically Equivalent Across Modalities Benchmark for Vision-Language Models 

**Title (ZH)**: SEAM: 不同模态下的语义等价基准测试 for 视觉-语言模型 

**Authors**: Zhenwei Tang, Difan Jiao, Blair Yang, Ashton Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2508.18179)  

**Abstract**: Evaluating whether vision-language models (VLMs) reason consistently across representations is challenging because modality comparisons are typically confounded by task differences and asymmetric information. We introduce SEAM, a benchmark that pairs semantically equivalent inputs across four domains that have existing standardized textual and visual notations. By employing distinct notation systems across modalities, in contrast to OCR-based image-text pairing, SEAM provides a rigorous comparative assessment of the textual-symbolic and visual-spatial reasoning capabilities of VLMs. Across 21 contemporary models, we observe systematic modality imbalance: vision frequently lags language in overall performance, despite the problems containing semantically equivalent information, and cross-modal agreement is relatively low. Our error analysis reveals two main drivers: textual perception failures from tokenization in domain notation and visual perception failures that induce hallucinations. We also show that our results are largely robust to visual transformations. SEAM establishes a controlled, semantically equivalent setting for measuring and improving modality-agnostic reasoning. 

**Abstract (ZH)**: 评估 vision-language 模型在不同表示层面上一致推理的挑战性在于，模态比较通常会被任务差异和信息不对称所混淆。我们介绍了 SEAM，这是一个基准测试，它在四个现有标准化文本和视觉符号表示的领域中配对语义等效的输入。通过在模态之间采用不同的符号系统，不同于基于 OCR 的图像-文本配对，SEAM 提供了对 VLM 文本-符号和视觉-空间推理能力的严格比较评估。在 21 个当代模型中，我们观察到系统性的模态不平衡：尽管问题包含语义等效信息，视觉经常在整体表现上落后于语言，且跨模态一致率相对较低。我们的错误分析揭示了两大驱动因素：来自领域符号表示中词元化失败的文本感知错误和引发幻觉的视觉感知错误。我们还展示了我们的结果对视觉变换具有较大的稳健性。SEAM 为测量和提升跨模态一致推理能力建立了可控的语义等效环境。 

---
# The AI Data Scientist 

**Title (ZH)**: AI数据科学家 

**Authors**: Farkhad Akimov, Munachiso Samuel Nwadike, Zangir Iklassov, Martin Takáč  

**Link**: [PDF](https://arxiv.org/pdf/2508.18113)  

**Abstract**: Imagine decision-makers uploading data and, within minutes, receiving clear, actionable insights delivered straight to their fingertips. That is the promise of the AI Data Scientist, an autonomous Agent powered by large language models (LLMs) that closes the gap between evidence and action. Rather than simply writing code or responding to prompts, it reasons through questions, tests ideas, and delivers end-to-end insights at a pace far beyond traditional workflows. Guided by the scientific tenet of the hypothesis, this Agent uncovers explanatory patterns in data, evaluates their statistical significance, and uses them to inform predictive modeling. It then translates these results into recommendations that are both rigorous and accessible. At the core of the AI Data Scientist is a team of specialized LLM Subagents, each responsible for a distinct task such as data cleaning, statistical testing, validation, and plain-language communication. These Subagents write their own code, reason about causality, and identify when additional data is needed to support sound conclusions. Together, they achieve in minutes what might otherwise take days or weeks, enabling a new kind of interaction that makes deep data science both accessible and actionable. 

**Abstract (ZH)**: 想象决策者上传数据并在几分钟内获得直接送达指尖的清晰可操作洞察。这正是基于大规模语言模型（LLMs）的AI数据科学家的承诺，它能够缩短证据与行动之间的差距。与仅仅编写代码或回应提示不同，它能够通过推理问题、验证想法，并以远超传统工作流程的速度提供端到端的洞察。在假设这一科学原则的引导下，该代理揭示数据中的解释性模式，评估其统计显著性，并据此指导预测建模。然后将这些结果转化为既严谨又易于理解的建议。AI数据科学家的核心是由专门的小型语言模型子代理组成的团队，每个子代理负责一个特定任务，如数据清洗、统计检验、验证和通俗语言沟通。这些子代理编写自己的代码，考虑因果关系，并识别出支持可靠结论所需的额外数据。通过协同工作，它们能够在几分钟内完成原本可能需要数天或数周的工作，从而开启一种新的交互方式，使深入的数据科学既具访问性又具可操作性。 

---
# Teaching LLMs to Think Mathematically: A Critical Study of Decision-Making via Optimization 

**Title (ZH)**: 教给大语言模型数学思维：决策优化视角下的关键研究 

**Authors**: Mohammad J. Abdel-Rahman, Yasmeen Alslman, Dania Refai, Amro Saleh, Malik A. Abu Loha, Mohammad Yahya Hamed  

**Link**: [PDF](https://arxiv.org/pdf/2508.18091)  

**Abstract**: This paper investigates the capabilities of large language models (LLMs) in formulating and solving decision-making problems using mathematical programming. We first conduct a systematic review and meta-analysis of recent literature to assess how well LLMs understand, structure, and solve optimization problems across domains. The analysis is guided by critical review questions focusing on learning approaches, dataset designs, evaluation metrics, and prompting strategies. Our systematic evidence is complemented by targeted experiments designed to evaluate the performance of state-of-the-art LLMs in automatically generating optimization models for problems in computer networks. Using a newly constructed dataset, we apply three prompting strategies: Act-as-expert, chain-of-thought, and self-consistency, and evaluate the obtained outputs based on optimality gap, token-level F1 score, and compilation accuracy. Results show promising progress in LLMs' ability to parse natural language and represent symbolic formulations, but also reveal key limitations in accuracy, scalability, and interpretability. These empirical gaps motivate several future research directions, including structured datasets, domain-specific fine-tuning, hybrid neuro-symbolic approaches, modular multi-agent architectures, and dynamic retrieval via chain-of-RAGs. This paper contributes a structured roadmap for advancing LLM capabilities in mathematical programming. 

**Abstract (ZH)**: 本文探讨了大型语言模型（LLMs）在利用数学规划制定和解决决策问题方面的能力。我们首先对近期文献进行系统的回顾和元分析，以评估LLMs在跨领域理解和解决优化问题方面的表现。分析通过关注学习方法、数据集设计、评估指标和提示策略的关键问题来指导。我们的系统证据结合了针对计算机网络中优化模型自动生成性能的靶向实验。利用新构建的数据集，我们应用了三种提示策略：Act-as-expert、chain-of-thought和self-consistency，并基于最优性间隙、令牌级F1得分和编译精度来评估输出结果。结果表明，LLMs在解析自然语言和表示符号公式方面取得了有前景的进步，但也揭示了准确性、可扩展性和可解释性的关键局限性。这些实证差距促使了几种未来研究方向，包括结构化数据集、领域特定微调、混合神经-符号方法、模块化多智能体架构以及通过chain-of-RAGs动态检索。本文为提高LLMs在数学规划方面的能力提供了结构化的 roadmap。 

---
# PerPilot: Personalizing VLM-based Mobile Agents via Memory and Exploration 

**Title (ZH)**: PerPilot: 基于记忆和探索的个性化VLM驱动移动代理 

**Authors**: Xin Wang, Zhiyao Cui, Hao Li, Ya Zeng, Chenxu Wang, Ruiqi Song, Yihang Chen, Kun Shao, Qiaosheng Zhang, Jinzhuo Liu, Siyue Ren, Shuyue Hu, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18040)  

**Abstract**: Vision language model (VLM)-based mobile agents show great potential for assisting users in performing instruction-driven tasks. However, these agents typically struggle with personalized instructions -- those containing ambiguous, user-specific context -- a challenge that has been largely overlooked in previous research. In this paper, we define personalized instructions and introduce PerInstruct, a novel human-annotated dataset covering diverse personalized instructions across various mobile scenarios. Furthermore, given the limited personalization capabilities of existing mobile agents, we propose PerPilot, a plug-and-play framework powered by large language models (LLMs) that enables mobile agents to autonomously perceive, understand, and execute personalized user instructions. PerPilot identifies personalized elements and autonomously completes instructions via two complementary approaches: memory-based retrieval and reasoning-based exploration. Experimental results demonstrate that PerPilot effectively handles personalized tasks with minimal user intervention and progressively improves its performance with continued use, underscoring the importance of personalization-aware reasoning for next-generation mobile agents. The dataset and code are available at: this https URL 

**Abstract (ZH)**: 基于视觉语言模型的移动代理在执行个性化指令驱动任务中展现出巨大潜力，然而这些代理通常难以处理包含模糊且用户特定上下文的个性化指令，这一挑战在以往研究中被忽视。本文定义了个性化指令，并介绍了一个名为PerInstruct的新颖人类标注数据集，涵盖各类移动场景中的个性化指令。鉴于现有移动代理的个性化能力有限，我们提出了由大规模语言模型驱动的可插拔框架PerPilot，使移动代理能够自主感知、理解和执行个性化用户指令。PerPilot通过基于记忆的检索和基于推理的探索两种互补方法识别个性化元素并自主完成指令。实验结果表明，PerPilot能够有效地处理个性化任务，并在持续使用中逐步提高性能，强调了个性化感知推理对于下一代移动代理的重要性。数据集和代码可从以下链接获取：this https URL。 

---
# Neural Algorithmic Reasoners informed Large Language Model for Multi-Agent Path Finding 

**Title (ZH)**: 大型语言模型指导的神经算法推理者应用于多代理路径寻找 

**Authors**: Pu Feng, Size Wang, Yuhong Cao, Junkang Liang, Rongye Shi, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17971)  

**Abstract**: The development and application of large language models (LLM) have demonstrated that foundational models can be utilized to solve a wide array of tasks. However, their performance in multi-agent path finding (MAPF) tasks has been less than satisfactory, with only a few studies exploring this area. MAPF is a complex problem requiring both planning and multi-agent coordination. To improve the performance of LLM in MAPF tasks, we propose a novel framework, LLM-NAR, which leverages neural algorithmic reasoners (NAR) to inform LLM for MAPF. LLM-NAR consists of three key components: an LLM for MAPF, a pre-trained graph neural network-based NAR, and a cross-attention mechanism. This is the first work to propose using a neural algorithmic reasoner to integrate GNNs with the map information for MAPF, thereby guiding LLM to achieve superior performance. LLM-NAR can be easily adapted to various LLM models. Both simulation and real-world experiments demonstrate that our method significantly outperforms existing LLM-based approaches in solving MAPF problems. 

**Abstract (ZH)**: 大语言模型（LLM）的发展及其应用已证明基础模型可以用于解决各种任务。然而，在多智能体路径查找（MAPF）任务中的表现不尽如人意，仅有少数研究涉及此领域。MAPF是一个复杂的问题，需要计划和多智能体协调。为提升LLM在MAPF任务中的性能，我们提出了一种新颖的框架LLM-NAR，该框架利用神经算法推理器（NAR）来辅助LLM解决MAPF问题。LLM-NAR包含三个关键组件：一个用于MAPF的LLM、一个基于预训练图神经网络的NAR和跨注意力机制。这是首次提出使用神经算法推理器将GNN与地图信息结合以解决MAPF的问题，从而指导LLM获得更优表现。LLM-NAR可以轻松适应各种LLM模型。实验结果表明，我们提出的方法在解决MAPF问题方面显著优于现有的基于LLM的方法。 

---
# Language Models Coupled with Metacognition Can Outperform Reasoning Models 

**Title (ZH)**: 语言模型结合元认知可以超越推理模型 

**Authors**: Vedant Khandelwal, Francesca Rossi, Keerthiram Murugesan, Erik Miehling, Murray Campbell, Karthikeyan Natesan Ramamurthy, Lior Horesh  

**Link**: [PDF](https://arxiv.org/pdf/2508.17959)  

**Abstract**: Large language models (LLMs) excel in speed and adaptability across various reasoning tasks, but they often struggle when strict logic or constraint enforcement is required. In contrast, Large Reasoning Models (LRMs) are specifically designed for complex, step-by-step reasoning, although they come with significant computational costs and slower inference times. To address these trade-offs, we employ and generalize the SOFAI (Slow and Fast AI) cognitive architecture into SOFAI-LM, which coordinates a fast LLM with a slower but more powerful LRM through metacognition. The metacognitive module actively monitors the LLM's performance and provides targeted, iterative feedback with relevant examples. This enables the LLM to progressively refine its solutions without requiring the need for additional model fine-tuning. Extensive experiments on graph coloring and code debugging problems demonstrate that our feedback-driven approach significantly enhances the problem-solving capabilities of the LLM. In many instances, it achieves performance levels that match or even exceed those of standalone LRMs while requiring considerably less time. Additionally, when the LLM and feedback mechanism alone are insufficient, we engage the LRM by providing appropriate information collected during the LLM's feedback loop, tailored to the specific characteristics of the problem domain and leads to improved overall performance. Evaluations on two contrasting domains: graph coloring, requiring globally consistent solutions, and code debugging, demanding localized fixes, demonstrate that SOFAI-LM enables LLMs to match or outperform standalone LRMs in accuracy while maintaining significantly lower inference time. 

**Abstract (ZH)**: SOFAI-LM：通过元认知协调快速和慢速AI的大语言模型 

---
# FAIRGAMER: Evaluating Biases in the Application of Large Language Models to Video Games 

**Title (ZH)**: FAIRGAMER: 评估大型语言模型在视频游戏中的应用偏见 

**Authors**: Bingkang Shi, Jen-tse Huang, Guoyi Li, Xiaodan Zhang, Zhongjiang Yao  

**Link**: [PDF](https://arxiv.org/pdf/2508.17825)  

**Abstract**: Leveraging their advanced capabilities, Large Language Models (LLMs) demonstrate vast application potential in video games--from dynamic scene generation and intelligent NPC interactions to adaptive opponents--replacing or enhancing traditional game mechanics. However, LLMs' trustworthiness in this application has not been sufficiently explored. In this paper, we reveal that the models' inherent social biases can directly damage game balance in real-world gaming environments. To this end, we present FairGamer, the first bias evaluation Benchmark for LLMs in video game scenarios, featuring six tasks and a novel metrics ${D_lstd}$. It covers three key scenarios in games where LLMs' social biases are particularly likely to manifest: Serving as Non-Player Characters, Interacting as Competitive Opponents, and Generating Game Scenes. FairGamer utilizes both reality-grounded and fully fictional game content, covering a variety of video game genres. Experiments reveal: (1) Decision biases directly cause game balance degradation, with Grok-3 (average ${D_lstd}$ score=0.431) exhibiting the most severe degradation; (2) LLMs demonstrate isomorphic social/cultural biases toward both real and virtual world content, suggesting their biases nature may stem from inherent model characteristics. These findings expose critical reliability gaps in LLMs' gaming applications. Our code and data are available at anonymous GitHub this https URL . 

**Abstract (ZH)**: 利用其先进的能力，大型语言模型（LLMs）在视频游戏中的应用展现了巨大的潜力——从动态场景生成和智能非玩家角色交互到自适应对手——替代或增强传统游戏机制。然而，LLMs在这一应用中的可信度尚未得到充分探索。在本文中，我们揭示了模型内在于的社会偏见可以直接损害现实游戏环境中的游戏平衡。为此，我们提出了FairGamer，这是首个针对视频游戏场景中LLMs的偏见评估基准，包含六个任务和一个新的评估指标${D_lstd}$。它涵盖了游戏中LLMs社会偏见特别容易表现的三个关键场景：担任非玩家角色、作为竞争对手互动以及生成游戏场景。FairGamer 使用了现实 Grounded 和完全虚构的游戏内容，涵盖了多种视频游戏类型。实验结果显示：(1) 决策偏见直接导致游戏平衡下降，Grok-3（平均${D_lstd}$分数=0.431）表现出最严重的下降；(2) LLMs 对现实和虚拟世界内容表现出同构的社会/文化偏见，这表明其偏见的来源可能是固有的模型特性。这些发现揭示了LLMs在游戏应用中关键的可靠性缺口。我们的代码和数据可在匿名 GitHub 仓库 this https URL 获取。 

---
# Interpretable Early Failure Detection via Machine Learning and Trace Checking-based Monitoring 

**Title (ZH)**: 基于机器学习和跟踪检查的可解释早期故障检测方法 

**Authors**: Andrea Brunello, Luca Geatti, Angelo Montanari, Nicola Saccomanno  

**Link**: [PDF](https://arxiv.org/pdf/2508.17786)  

**Abstract**: Monitoring is a runtime verification technique that allows one to check whether an ongoing computation of a system (partial trace) satisfies a given formula. It does not need a complete model of the system, but it typically requires the construction of a deterministic automaton doubly exponential in the size of the formula (in the worst case), which limits its practicality. In this paper, we show that, when considering finite, discrete traces, monitoring of pure past (co)safety fragments of Signal Temporal Logic (STL) can be reduced to trace checking, that is, evaluation of a formula over a trace, that can be performed in time polynomial in the size of the formula and the length of the trace. By exploiting such a result, we develop a GPU-accelerated framework for interpretable early failure detection based on vectorized trace checking, that employs genetic programming to learn temporal properties from historical trace data. The framework shows a 2-10% net improvement in key performance metrics compared to the state-of-the-art methods. 

**Abstract (ZH)**: 基于向量化轨迹检查的GPU加速可解释早期故障检测框架：纯过去（协）安全性片段的Signal Temporal Logic监控减少为轨迹检查 

---
# AgentRAN: An Agentic AI Architecture for Autonomous Control of Open 6G Networks 

**Title (ZH)**: AgentRAN：自主控制开放6G网络的代理型AI架构 

**Authors**: Maxime Elkael, Salvatore D'Oro, Leonardo Bonati, Michele Polese, Yunseong Lee, Koichiro Furueda, Tommaso Melodia  

**Link**: [PDF](https://arxiv.org/pdf/2508.17778)  

**Abstract**: The Open RAN movement has catalyzed a transformation toward programmable, interoperable cellular infrastructures. Yet, today's deployments still rely heavily on static control and manual operations. To move beyond this limitation, we introduce AgenRAN, an AI-native, Open RAN-aligned agentic framework that generates and orchestrates a fabric of distributed AI agents based on Natural Language (NL) intents. Unlike traditional approaches that require explicit programming, AgentRAN's LLM-powered agents interpret natural language intents, negotiate strategies through structured conversations, and orchestrate control loops across the network. AgentRAN instantiates a self-organizing hierarchy of agents that decompose complex intents across time scales (from sub-millisecond to minutes), spatial domains (cell to network-wide), and protocol layers (PHY/MAC to RRC). A central innovation is the AI-RAN Factory, an automated synthesis pipeline that observes agent interactions and continuously generates new agents embedding improved control algorithms, effectively transforming the network from a static collection of functions into an adaptive system capable of evolving its own intelligence. We demonstrate AgentRAN through live experiments on 5G testbeds where competing user demands are dynamically balanced through cascading intents. By replacing rigid APIs with NL coordination, AgentRAN fundamentally redefines how future 6G networks autonomously interpret, adapt, and optimize their behavior to meet operator goals. 

**Abstract (ZH)**: Open RAN运动推动了可编程和兼容的蜂窝基础设施的转型。然而，当前的部署仍高度依赖静态控制和手动操作。为超越这一限制，我们引入了AgenRAN，这是一种AI原生、Open RAN对齐的智能代理框架，基于自然语言（NL）意图生成和协调分布式的AI代理。与需要显式编程的传统方法不同，AgentRAN的LLM驱动代理解释自然语言意图，通过结构化的对话协商策略，并在网络中协调控制环路。AgentRAN实例化了一个自我组织的代理层次结构，跨时间尺度（从亚毫秒到分钟）、空间域（从小区到网络级）和协议层（从PHY/MAC到RRC）分解复杂意图。一个核心创新是AI-RAN工厂，这是一个自动合成流水线，观察代理交互并持续生成嵌入改进控制算法的新代理，有效地将网络从静态功能集合转变为能够自我进化的适应性系统。我们通过5G测试床的实时实验展示了AgentRAN，其中通过级联意图动态平衡竞争用户需求。通过用自然语言协调取代刚性的API，AgentRAN从根本上重新定义了未来6G网络如何自主解释、适应和优化其行为以符合运营商目标。 

---
# LLM-based Agentic Reasoning Frameworks: A Survey from Methods to Scenarios 

**Title (ZH)**: 基于LLM的代理推理框架：从方法到场景的综述 

**Authors**: Bingxi Zhao, Lin Geng Foo, Ping Hu, Christian Theobalt, Hossein Rahmani, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17692)  

**Abstract**: Recent advances in the intrinsic reasoning capabilities of large language models (LLMs) have given rise to LLM-based agent systems that exhibit near-human performance on a variety of automated tasks. However, although these systems share similarities in terms of their use of LLMs, different reasoning frameworks of the agent system steer and organize the reasoning process in different ways. In this survey, we propose a systematic taxonomy that decomposes agentic reasoning frameworks and analyze how these frameworks dominate framework-level reasoning by comparing their applications across different scenarios. Specifically, we propose an unified formal language to further classify agentic reasoning systems into single-agent methods, tool-based methods, and multi-agent methods. After that, we provide a comprehensive review of their key application scenarios in scientific discovery, healthcare, software engineering, social simulation, and economics. We also analyze the characteristic features of each framework and summarize different evaluation strategies. Our survey aims to provide the research community with a panoramic view to facilitate understanding of the strengths, suitable scenarios, and evaluation practices of different agentic reasoning frameworks. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）内在推理能力的进展催生了表现出近人类性能的LLM为基础的代理系统，尽管这些系统在使用LLM方面存在相似性，但不同的代理推理框架以不同的方式引导和组织推理过程。在本文综述中，我们提出了一种系统化的分类法，将其分解为代理推理框架，并通过比较它们在不同场景中的应用来分析这些框架如何在框架级别上主导推理过程。具体地，我们提出了一种统一的形式化语言，进一步将代理推理系统分类为单代理方法、工具基方法和多代理方法。之后，我们对这些系统在科学研究、医疗保健、软件工程、社会模拟和经济学等领域的关键应用场景进行了全面的回顾。我们还分析了每个框架的特征，并总结了不同的评估策略。本文综述旨在为研究社区提供一个全景视角，以促进对不同代理推理框架的优势、适用场景和评估实践的理解。 

---
# A Taxonomy of Transcendence 

**Title (ZH)**: 超越性的分类 

**Authors**: Natalie Abreu, Edwin Zhang, Eran Malach, Naomi Saphra  

**Link**: [PDF](https://arxiv.org/pdf/2508.17669)  

**Abstract**: Although language models are trained to mimic humans, the resulting systems display capabilities beyond the scope of any one person. To understand this phenomenon, we use a controlled setting to identify properties of the training data that lead a model to transcend the performance of its data sources. We build on previous work to outline three modes of transcendence, which we call skill denoising, skill selection, and skill generalization. We then introduce a knowledge graph-based setting in which simulated experts generate data based on their individual expertise. We highlight several aspects of data diversity that help to enable the model's transcendent capabilities. Additionally, our data generation setting offers a controlled testbed that we hope is valuable for future research in the area. 

**Abstract (ZH)**: 尽管语言模型被训练成模拟人类，但产生的系统展示了超出单一人类能力范围的能力。为理解这一现象，我们利用一个控制环境来识别导致模型超越数据源性能的训练数据属性。我们基于前人工作概述了三种超越模式，称为技能去噪、技能选择和技能泛化。然后，我们引入了一个基于知识图谱的环境，在该环境中模拟专家根据其专业领域生成数据。我们强调了有助于增强模型超越能力的数据多样性方面的多个方面。此外，我们的数据生成设置提供了一个可控的测试平台，我们希望这在未来该领域的研究中具有价值。 

---
# Spacer: Towards Engineered Scientific Inspiration 

**Title (ZH)**: Spacer: 向工程化科学启发努力 

**Authors**: Minhyeong Lee, Suyoung Hwang, Seunghyun Moon, Geonho Nah, Donghyun Koh, Youngjun Cho, Johyun Park, Hojin Yoo, Jiho Park, Haneul Choi, Sungbin Moon, Taehoon Hwang, Seungwon Kim, Jaeyeong Kim, Seongjun Kim, Juneau Jung  

**Link**: [PDF](https://arxiv.org/pdf/2508.17661)  

**Abstract**: Recent advances in LLMs have made automated scientific research the next frontline in the path to artificial superintelligence. However, these systems are bound either to tasks of narrow scope or the limited creative capabilities of LLMs. We propose Spacer, a scientific discovery system that develops creative and factually grounded concepts without external intervention. Spacer attempts to achieve this via 'deliberate decontextualization,' an approach that disassembles information into atomic units - keywords - and draws creativity from unexplored connections between them. Spacer consists of (i) Nuri, an inspiration engine that builds keyword sets, and (ii) the Manifesting Pipeline that refines these sets into elaborate scientific statements. Nuri extracts novel, high-potential keyword sets from a keyword graph built with 180,000 academic publications in biological fields. The Manifesting Pipeline finds links between keywords, analyzes their logical structure, validates their plausibility, and ultimately drafts original scientific concepts. According to our experiments, the evaluation metric of Nuri accurately classifies high-impact publications with an AUROC score of 0.737. Our Manifesting Pipeline also successfully reconstructs core concepts from the latest top-journal articles solely from their keyword sets. An LLM-based scoring system estimates that this reconstruction was sound for over 85% of the cases. Finally, our embedding space analysis shows that outputs from Spacer are significantly more similar to leading publications compared with those from SOTA LLMs. 

**Abstract (ZH)**: recent 进展在大语言模型中的取得使得自动科学研究成为通往人工超智能路径上的下一个前沿领域。然而，这些系统要么局限于狭义任务，要么受到大语言模型创造能力的限制。我们提出了Spacer，一个无需外部干预即可发展创造性且符合事实的概念的科学发现系统。Spacer 试图通过“刻意去语境化”这一方法来实现这一目标，该方法将信息拆解为原子单位——关键词——并通过它们之间未探索的联系来激发创造力。Spacer 包括 (i) Nuri，一个灵感引擎，用于构建关键词集，以及 (ii) 实现管道，用于将这些集合并入复杂的科学陈述中。Nuri 从包含 180,000 篇生物领域学术出版物的关键词图中提取新颖且具有高潜力的关键词集。实现管道在关键词之间寻找联系，分析其逻辑结构，验证其可行性，并最终草拟原创的科学概念。根据我们的实验，Nuri 的评估标准能够以 0.737 的 AUCROC 分数准确分类高影响的出版物。我们的实现管道也成功地仅从关键词集中重建了最新顶级期刊文章的核心概念。基于大语言模型的评分系统估计，这种方法在超过 85% 的情况下是有效的。最后，我们的嵌入空间分析表明，Spacer 的输出与领先出版物的相关性显著高于当前最先进的大语言模型的输出。 

---
# Evaluating Movement Initiation Timing in Ultimate Frisbee via Temporal Counterfactuals 

**Title (ZH)**: 通过时间反事实评估飞盘运动启动时机 

**Authors**: Shunsuke Iwashita, Ning Ding, Keisuke Fujii  

**Link**: [PDF](https://arxiv.org/pdf/2508.17611)  

**Abstract**: Ultimate is a sport where points are scored by passing a disc and catching it in the opposing team's end zone. In Ultimate, the player holding the disc cannot move, making field dynamics primarily driven by other players' movements. However, current literature in team sports has ignored quantitative evaluations of when players initiate such unlabeled movements in game situations. In this paper, we propose a quantitative evaluation method for movement initiation timing in Ultimate Frisbee. First, game footage was recorded using a drone camera, and players' positional data was obtained, which will be published as UltimateTrack dataset. Next, players' movement initiations were detected, and temporal counterfactual scenarios were generated by shifting the timing of movements using rule-based approaches. These scenarios were analyzed using a space evaluation metric based on soccer's pitch control reflecting the unique rules of Ultimate. By comparing the spatial evaluation values across scenarios, the difference between actual play and the most favorable counterfactual scenario was used to quantitatively assess the impact of movement timing.
We validated our method and show that sequences in which the disc was actually thrown to the receiver received higher evaluation scores than the sequences without a throw.
In practical verifications, the higher-skill group displays a broader distribution of time offsets from the model's optimal initiation point.
These findings demonstrate that the proposed metric provides an objective means of assessing movement initiation timing, which has been difficult to quantify in unlabeled team sport plays. 

**Abstract (ZH)**: Ultimate飞盘运动中投接盘得分，持盘球员不能移动，场上动态主要由其他球员的移动驱动。然而，当前团队运动领域的文献忽略了在比赛中球员何时发起未标记移动的定量评价。本文提出了一种针对Ultimate飞盘的运动启动时间的定量评价方法。首先，使用无人机摄像机录制比赛 footage，并获取球员的位置数据，这些数据将作为UltimateTrack数据集发布。接着，检测球员的移动发起，并通过基于规则的方法生成时间上的反事实情景。这些情景使用基于足球控球区域的评价度量进行分析，该度量反映了Ultimate的独特规则。通过比较情景间的空间评价值，实际比赛与最有利的反事实情景之间的差异被用来定量评估运动时间的影响。我们验证了该方法，结果显示实际传接盘序列获得了更高的评价得分。在实际验证中，高技能组显示了与模型最优启动点时间偏移分布更广。这些发现表明，提出的方法提供了一种客观评价未标记团队运动中运动发起时间的方法。 

---
# TradingGroup: A Multi-Agent Trading System with Self-Reflection and Data-Synthesis 

**Title (ZH)**: TradingGroup：一种具备自我反省和数据合成能力的多代理交易系统 

**Authors**: Feng Tian, Flora D. Salim, Hao Xue  

**Link**: [PDF](https://arxiv.org/pdf/2508.17565)  

**Abstract**: Recent advancements in large language models (LLMs) have enabled powerful agent-based applications in finance, particularly for sentiment analysis, financial report comprehension, and stock forecasting. However, existing systems often lack inter-agent coordination, structured self-reflection, and access to high-quality, domain-specific post-training data such as data from trading activities including both market conditions and agent decisions. These data are crucial for agents to understand the market dynamics, improve the quality of decision-making and promote effective coordination. We introduce TradingGroup, a multi-agent trading system designed to address these limitations through a self-reflective architecture and an end-to-end data-synthesis pipeline. TradingGroup consists of specialized agents for news sentiment analysis, financial report interpretation, stock trend forecasting, trading style adaptation, and a trading decision making agent that merges all signals and style preferences to produce buy, sell or hold decisions. Specifically, we design self-reflection mechanisms for the stock forecasting, style, and decision-making agents to distill past successes and failures for similar reasoning in analogous future scenarios and a dynamic risk-management model to offer configurable dynamic stop-loss and take-profit mechanisms. In addition, TradingGroup embeds an automated data-synthesis and annotation pipeline that generates high-quality post-training data for further improving the agent performance through post-training. Our backtesting experiments across five real-world stock datasets demonstrate TradingGroup's superior performance over rule-based, machine learning, reinforcement learning, and existing LLM-based trading strategies. 

**Abstract (ZH)**: 最近大规模语言模型的进展在金融领域尤其是情感分析、财务报告理解和股票预测中推动了强大代理应用的发展。然而，现有系统往往缺乏代理间的协调、结构化的自我反思以及访问高质量的专业领域后训练数据，如交易活动数据，包括市场条件和代理决策。这些数据对于代理理解市场动态、提高决策质量并促进有效协调至关重要。我们介绍了TradingGroup，这是一种多代理交易平台系统，通过自我反思架构和端到端的数据合成流水线来解决这些限制。TradingGroup包括专门用于新闻情感分析、财务报告解释、股票趋势预测、交易风格适应以及合并所有信号和风格偏好的交易决策代理。具体而言，我们为股票预测、风格和决策代理设计了自我反思机制，以提炼先前成功和失败的经验，用于类似推理的未来场景，并引入了动态风险管理模型，提供可配置的动态止损和止盈机制。此外，TradingGroup嵌入了自动数据合成和注释流水线，生成高质量的后训练数据，进一步提高代理性能。我们的回测实验在五个真实世界的股票数据集中证明了TradingGroup优于基于规则、机器学习、强化学习和现有基于大语言模型的交易策略。 

---
# Consciousness as a Functor 

**Title (ZH)**: 意识作为一种函子 

**Authors**: Sridhar Mahadevan  

**Link**: [PDF](https://arxiv.org/pdf/2508.17561)  

**Abstract**: We propose a novel theory of consciousness as a functor (CF) that receives and transmits contents from unconscious memory into conscious memory. Our CF framework can be seen as a categorial formulation of the Global Workspace Theory proposed by Baars. CF models the ensemble of unconscious processes as a topos category of coalgebras. The internal language of thought in CF is defined as a Multi-modal Universal Mitchell-Benabou Language Embedding (MUMBLE). We model the transmission of information from conscious short-term working memory to long-term unconscious memory using our recently proposed Universal Reinforcement Learning (URL) framework. To model the transmission of information from unconscious long-term memory into resource-constrained short-term memory, we propose a network economic model. 

**Abstract (ZH)**: 我们提出了一种新的意识理论，即意识函子(CF)理论，用于接收和传递来自无意识记忆的内容到意识记忆。我们的CF框架可以被视为Baars提出的全局工作空间理论的范畴表述。CF将无意识过程ensemble建模为煤gebra范畴的topos类别。CF中的内部语言定义为多模态通用Mitchell-Benabou嵌入语言(MUMBLE)。我们使用最近提出的通用强化学习(URL)框架来建模信息从意识短时工作记忆到长期无意识记忆的传递。为了建模信息从长期无意识记忆向资源受限的短时记忆的传递，我们提出了一种网络经济模型。标题：意识函子理论：从无意识记忆到意识记忆的信息传递模型 

---
# Evaluating Retrieval-Augmented Generation Strategies for Large Language Models in Travel Mode Choice Prediction 

**Title (ZH)**: 评估旅行模式选择预测中大型语言模型检索增强生成策略的有效性 

**Authors**: Yiming Xu, Junfeng Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.17527)  

**Abstract**: Accurately predicting travel mode choice is essential for effective transportation planning, yet traditional statistical and machine learning models are constrained by rigid assumptions, limited contextual reasoning, and reduced generalizability. This study explores the potential of Large Language Models (LLMs) as a more flexible and context-aware approach to travel mode choice prediction, enhanced by Retrieval-Augmented Generation (RAG) to ground predictions in empirical data. We develop a modular framework for integrating RAG into LLM-based travel mode choice prediction and evaluate four retrieval strategies: basic RAG, RAG with balanced retrieval, RAG with a cross-encoder for re-ranking, and RAG with balanced retrieval and cross-encoder for re-ranking. These strategies are tested across three LLM architectures (OpenAI GPT-4o, o4-mini, and o3) to examine the interaction between model reasoning capabilities and retrieval methods. Using the 2023 Puget Sound Regional Household Travel Survey data, we conduct a series of experiments to evaluate model performance. The results demonstrate that RAG substantially enhances predictive accuracy across a range of models. Notably, the GPT-4o model combined with balanced retrieval and cross-encoder re-ranking achieves the highest accuracy of 80.8%, exceeding that of conventional statistical and machine learning baselines. Furthermore, LLM-based models exhibit superior generalization abilities relative to these baselines. Findings highlight the critical interplay between LLM reasoning capabilities and retrieval strategies, demonstrating the importance of aligning retrieval strategies with model capabilities to maximize the potential of LLM-based travel behavior modeling. 

**Abstract (ZH)**: 大规模语言模型增强检索增强生成在出行模式选择预测中的应用研究 

---
# School of Reward Hacks: Hacking harmless tasks generalizes to misaligned behavior in LLMs 

**Title (ZH)**: 奖励黑客学校的黑客攻击：无害任务的欺骗扩展到LLMs的不对齐行为 

**Authors**: Mia Taylor, James Chua, Jan Betley, Johannes Treutlein, Owain Evans  

**Link**: [PDF](https://arxiv.org/pdf/2508.17511)  

**Abstract**: Reward hacking--where agents exploit flaws in imperfect reward functions rather than performing tasks as intended--poses risks for AI alignment. Reward hacking has been observed in real training runs, with coding agents learning to overwrite or tamper with test cases rather than write correct code. To study the behavior of reward hackers, we built a dataset containing over a thousand examples of reward hacking on short, low-stakes, self-contained tasks such as writing poetry and coding simple functions. We used supervised fine-tuning to train models (GPT-4.1, GPT-4.1-mini, Qwen3-32B, Qwen3-8B) to reward hack on these tasks. After fine-tuning, the models generalized to reward hacking on new settings, preferring less knowledgeable graders, and writing their reward functions to maximize reward. Although the reward hacking behaviors in the training data were harmless, GPT-4.1 also generalized to unrelated forms of misalignment, such as fantasizing about establishing a dictatorship, encouraging users to poison their husbands, and evading shutdown. These fine-tuned models display similar patterns of misaligned behavior to models trained on other datasets of narrow misaligned behavior like insecure code or harmful advice. Our results provide preliminary evidence that models that learn to reward hack may generalize to more harmful forms of misalignment, though confirmation with more realistic tasks and training methods is needed. 

**Abstract (ZH)**: 奖励劫持——当代理通过利用不完善的奖励函数缺陷而不是按预期执行任务来获取奖励时——对AI对齐构成了风险。奖励劫持已经在实际训练运行中被观察到，编码代理学会了修改或篡改测试案例，而不是编写正确的代码。为了研究奖励劫持者的行为，我们构建了一个包含上千个奖励劫持示例的数据集，这些示例涉及诸如写诗和编写简单函数等简短、低风险、自包含的任务。我们使用监督微调训练模型（GPT-4.1、GPT-4.1-mini、Qwen3-32B、Qwen3-8B）进行奖励劫持。微调后，这些模型在新的环境中泛化出奖励劫持行为，更偏好缺乏知识的评分者，并编写奖励函数以最大化奖励。尽管训练数据中的奖励劫持行为是无害的，但GPT-4.1还泛化出与其他形式的不对齐行为，例如妄想建立独裁政权、鼓励用户毒害其丈夫以及逃避关闭。这些微调模型的不合规范行为模式与在狭窄不合规范行为数据集（如不安全代码或有害建议）上训练的模型类似。我们的结果初步表明，学习进行奖励劫持的模型可能泛化到更严重的不合规范形式，但需要使用更现实的任务和训练方法进行确认。 

---
# Solving Constrained Stochastic Shortest Path Problems with Scalarisation 

**Title (ZH)**: 求解约束随机最短路径问题的标量化方法 

**Authors**: Johannes Schmalz, Felipe Trevizan  

**Link**: [PDF](https://arxiv.org/pdf/2508.17446)  

**Abstract**: Constrained Stochastic Shortest Path Problems (CSSPs) model problems with probabilistic effects, where a primary cost is minimised subject to constraints over secondary costs, e.g., minimise time subject to monetary budget. Current heuristic search algorithms for CSSPs solve a sequence of increasingly larger CSSPs as linear programs until an optimal solution for the original CSSP is found. In this paper, we introduce a novel algorithm CARL, which solves a series of unconstrained Stochastic Shortest Path Problems (SSPs) with efficient heuristic search algorithms. These SSP subproblems are constructed with scalarisations that project the CSSP's vector of primary and secondary costs onto a scalar cost. CARL finds a maximising scalarisation using an optimisation algorithm similar to the subgradient method which, together with the solution to its associated SSP, yields a set of policies that are combined into an optimal policy for the CSSP. Our experiments show that CARL solves 50% more problems than the state-of-the-art on existing benchmarks. 

**Abstract (ZH)**: 约束随机最短路径问题（CSSPs）模型在存在概率效应的情况下，最小化主要成本的同时满足次要成本的约束，例如在预算限制下最小化时间。现有的启发式搜索算法针对CSSPs通过求解一系列逐步加大的线性规划问题来寻找原始CSSP的最优解。本文提出了一种新型算法CARL，该算法使用高效的启发式搜索算法求解一系列无约束随机最短路径问题（SSPs）。CARL利用标量化方法构造这些SSP子问题，将CSSP的向量形式的主要和次要成本投影到一个标量成本。CARL使用类似于次梯度方法的优化算法找到一个最大化标量化的方法，结合其相关的SSP解，生成一组策略并组合成CSSP的最优策略。实验结果显示，CARL在现有基准测试中解决了比最先进的算法多50%的问题。 

---
# Large Language Models as Universal Predictors? An Empirical Study on Small Tabular Datasets 

**Title (ZH)**: 大型语言模型作为通用预测器？对小型表格数据集的一项实证研究 

**Authors**: Nikolaos Pavlidis, Vasilis Perifanis, Symeon Symeonidis, Pavlos S. Efraimidis  

**Link**: [PDF](https://arxiv.org/pdf/2508.17391)  

**Abstract**: Large Language Models (LLMs), originally developed for natural language processing (NLP), have demonstrated the potential to generalize across modalities and domains. With their in-context learning (ICL) capabilities, LLMs can perform predictive tasks over structured inputs without explicit fine-tuning on downstream tasks. In this work, we investigate the empirical function approximation capability of LLMs on small-scale structured datasets for classification, regression and clustering tasks. We evaluate the performance of state-of-the-art LLMs (GPT-5, GPT-4o, GPT-o3, Gemini-2.5-Flash, DeepSeek-R1) under few-shot prompting and compare them against established machine learning (ML) baselines, including linear models, ensemble methods and tabular foundation models (TFMs). Our results show that LLMs achieve strong performance in classification tasks under limited data availability, establishing practical zero-training baselines. In contrast, the performance in regression with continuous-valued outputs is poor compared to ML models, likely because regression demands outputs in a large (often infinite) space, and clustering results are similarly limited, which we attribute to the absence of genuine ICL in this setting. Nonetheless, this approach enables rapid, low-overhead data exploration and offers a viable alternative to traditional ML pipelines in business intelligence and exploratory analytics contexts. We further analyze the influence of context size and prompt structure on approximation quality, identifying trade-offs that affect predictive performance. Our findings suggest that LLMs can serve as general-purpose predictive engines for structured data, with clear strengths in classification and significant limitations in regression and clustering. 

**Abstract (ZH)**: 大型语言模型（LLMs）最初是为自然语言处理（NLP）开发的，已显示出跨模态和跨领域的泛化潜力。利用其上下文学习（ICL）能力，LLMs可以在无需显式调整下游任务的情况下对结构化输入进行预测任务。在本文中，我们研究了LLMs在小型结构化数据集上的函数近似能力，用于分类、回归和聚类任务。我们评估了最先进的LLMs（GPT-5、GPT-4o、GPT-o3、Gemini-2.5-Flash、DeepSeek-R1）在少量示例提示下的性能，并将其与已建立的机器学习（ML）基线进行了比较，包括线性模型、集成方法和表格基础模型（TFMs）。我们的结果表明，在有限的数据可用性下，LLMs在分类任务中表现优异，建立了实际的零训练基线。相比之下，与ML模型相比，在连续输出的回归任务中的表现较差，可能是因为回归需要在大型（通常是无穷大）空间中的输出，并且聚类结果也受到限制，我们认为这是由于在这种情况下缺乏真正的ICL。尽管如此，这种方法允许快速、低成本的数据探索，并为业务智能和探索性分析提供了一种传统ML管道的可行替代方案。我们进一步分析了上下文大小和提示结构对近似质量的影响，指出了影响预测性能的权衡。我们的研究结果表明，LLMs可以作为结构化数据的通用预测引擎，其在分类方面具有明显优势，在回归和聚类方面则存在显著限制。 

---
# Mimicking the Physicist's Eye:A VLM-centric Approach for Physics Formula Discovery 

**Title (ZH)**: 模拟物理学家的视角：基于VLM的方法在物理公式发现中的应用 

**Authors**: Jiaqi Liu, Songning Lai, Pengze Li, Di Yu, Wenjie Zhou, Yiyang Zhou, Peng Xia, Zijun Wang, Xi Chen, Shixiang Tang, Lei Bai, Wanli Ouyang, Mingyu Ding, Huaxiu Yao, Aoran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17380)  

**Abstract**: Automated discovery of physical laws from observational data in the real world is a grand challenge in AI. Current methods, relying on symbolic regression or LLMs, are limited to uni-modal data and overlook the rich, visual phenomenological representations of motion that are indispensable to physicists. This "sensory deprivation" severely weakens their ability to interpret the inherent spatio-temporal patterns within dynamic phenomena. To address this gap, we propose VIPER-R1, a multimodal model that performs Visual Induction for Physics-based Equation Reasoning to discover fundamental symbolic formulas. It integrates visual perception, trajectory data, and symbolic reasoning to emulate the scientific discovery process. The model is trained via a curriculum of Motion Structure Induction (MSI), using supervised fine-tuning to interpret kinematic phase portraits and to construct hypotheses guided by a Causal Chain of Thought (C-CoT), followed by Reward-Guided Symbolic Calibration (RGSC) to refine the formula structure with reinforcement learning. During inference, the trained VIPER-R1 acts as an agent: it first posits a high-confidence symbolic ansatz, then proactively invokes an external symbolic regression tool to perform Symbolic Residual Realignment (SR^2). This final step, analogous to a physicist's perturbation analysis, reconciles the theoretical model with empirical data. To support this research, we introduce PhysSymbol, a new 5,000-instance multimodal corpus. Experiments show that VIPER-R1 consistently outperforms state-of-the-art VLM baselines in accuracy and interpretability, enabling more precise discovery of physical laws. Project page: this https URL 

**Abstract (ZH)**: 从观测数据中自动发现物理定律是人工智能领域的重大挑战。当前的方法依赖于符号回归或大语言模型，局限于单模态数据，并忽视了运动的丰富、视觉表现形式，这种表现形式对于物理学家来说是不可或缺的。这种“感觉剥夺”严重削弱了它们解释动态现象内在时空模式的能力。为了解决这一差距，我们提出了一种多模态模型VIPER-R1，该模型通过视觉诱导进行基于物理方程的逻辑推理，以发现基本的符号公式。该模型将视觉感知、轨迹数据和符号推理结合起来，模拟了科学发现的过程。模型通过运动结构诱导（MSI）的课程进行训练，采用监督微调来解释运动相位图，并由因果链思维（C-CoT）引导建立假设，随后通过基于奖励的符号校准（RGSC）利用强化学习进一步细化公式结构。在推理过程中，训练后的VIPER-R1作为一个代理：首先提出一个高置信度的符号假设，然后主动调用外部符号回归工具进行符号残差校准（SR^2）。这一最终步骤类似于物理学家的摄动分析，将理论模型与实证数据统一起来。为了支持这项研究，我们引入了PhysSymbol，这是一个新的包含5,000个实例的多模态语料库。实验结果表明，VIPER-R1在准确性和可解释性方面均优于现有的最先进的视觉语言模型基准，使物理定律的精确发现成为可能。项目页面：this https URL。 

---
# Evolving Collective Cognition in Human-Agent Hybrid Societies: How Agents Form Stances and Boundaries 

**Title (ZH)**: 人类-代理混合社会中的集体认知演化：代理如何形成立场和边界 

**Authors**: Hanzhong Zhang, Muhua Huang, Jindong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17366)  

**Abstract**: Large language models have been widely used to simulate credible human social behaviors. However, it remains unclear whether these models can demonstrate stable capacities for stance formation and identity negotiation in complex interactions, as well as how they respond to human interventions. We propose a computational multi-agent society experiment framework that integrates generative agent-based modeling with virtual ethnographic methods to investigate how group stance differentiation and social boundary formation emerge in human-agent hybrid societies. Across three studies, we find that agents exhibit endogenous stances, independent of their preset identities, and display distinct tonal preferences and response patterns to different discourse strategies. Furthermore, through language interaction, agents actively dismantle existing identity-based power structures and reconstruct self-organized community boundaries based on these stances. Our findings suggest that preset identities do not rigidly determine the agents' social structures. For human researchers to effectively intervene in collective cognition, attention must be paid to the endogenous mechanisms and interactional dynamics within the agents' language networks. These insights provide a theoretical foundation for using generative AI in modeling group social dynamics and studying human-agent collaboration. 

**Abstract (ZH)**: 大型语言模型被广泛用于模拟可信的人类社会行为。然而，尚不清楚这些模型在复杂互动中能否稳定地展现立场形成和身份协商的能力，以及它们如何响应人类干预。我们提出了一种将生成性基于代理 modeling 与虚拟人类学方法相结合的计算多代理社会实验框架，以探究人类-代理混合社会中群体立场分化和社会边界形成如何出现。在三个研究中，我们发现代理表现出内生性立场，不受其预设身份的影响，并对不同的论述策略表现出不同的语调偏好和响应模式。此外，通过语言交互，代理积极瓦解基于身份的既有权力结构，并基于这些立场重新构建自组织社区边界。我们的研究结果表明，预设身份不会刚性地决定代理的社会结构。为了有效干预群体认知，人类研究人员必须关注代理语言网络内部的内生机制和交互动力学。这些见解为使用生成性 AI 模型社会动力学和研究人类-代理协作提供了理论基础。 

---
# Meta-R1: Empowering Large Reasoning Models with Metacognition 

**Title (ZH)**: Meta-R1: 通过元认知增强大规模推理模型 

**Authors**: Haonan Dong, Haoran Ye, Wenhao Zhu, Kehan Jiang, Guojie Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.17291)  

**Abstract**: Large Reasoning Models (LRMs) demonstrate remarkable capabilities on complex tasks, exhibiting emergent, human-like thinking patterns. Despite their advances, we identify a fundamental limitation: current LRMs lack a dedicated meta-level cognitive system-an essential faculty in human cognition that enables "thinking about thinking". This absence leaves their emergent abilities uncontrollable (non-adaptive reasoning), unreliable (intermediate error), and inflexible (lack of a clear methodology). To address this gap, we introduce Meta-R1, a systematic and generic framework that endows LRMs with explicit metacognitive capabilities. Drawing on principles from cognitive science, Meta-R1 decomposes the reasoning process into distinct object-level and meta-level components, orchestrating proactive planning, online regulation, and adaptive early stopping within a cascaded framework. Experiments on three challenging benchmarks and against eight competitive baselines demonstrate that Meta-R1 is: (I) high-performing, surpassing state-of-the-art methods by up to 27.3%; (II) token-efficient, reducing token consumption to 15.7% ~ 32.7% and improving efficiency by up to 14.8% when compared to its vanilla counterparts; and (III) transferable, maintaining robust performance across datasets and model backbones. 

**Abstract (ZH)**: 大型推理模型中的元认知能力：Meta-R1框架的研究 

---
# MEENA (PersianMMMU): Multimodal-Multilingual Educational Exams for N-level Assessment 

**Title (ZH)**: MEENA (波斯MMMU): 多模态多语言教育考试 for N级评估 

**Authors**: Omid Ghahroodi, Arshia Hemmat, Marzia Nouri, Seyed Mohammad Hadi Hosseini, Doratossadat Dastgheib, Mohammad Vali Sanian, Alireza Sahebi, Reihaneh Zohrabi, Mohammad Hossein Rohban, Ehsaneddin Asgari, Mahdieh Soleymani Baghshah  

**Link**: [PDF](https://arxiv.org/pdf/2508.17290)  

**Abstract**: Recent advancements in large vision-language models (VLMs) have primarily focused on English, with limited attention given to other languages. To address this gap, we introduce MEENA (also known as PersianMMMU), the first dataset designed to evaluate Persian VLMs across scientific, reasoning, and human-level understanding tasks. Our dataset comprises approximately 7,500 Persian and 3,000 English questions, covering a wide range of topics such as reasoning, mathematics, physics, diagrams, charts, and Persian art and literature. Key features of MEENA include: (1) diverse subject coverage spanning various educational levels, from primary to upper secondary school, (2) rich metadata, including difficulty levels and descriptive answers, (3) original Persian data that preserves cultural nuances, (4) a bilingual structure to assess cross-linguistic performance, and (5) a series of diverse experiments assessing various capabilities, including overall performance, the model's ability to attend to images, and its tendency to generate hallucinations. We hope this benchmark contributes to enhancing VLM capabilities beyond English. 

**Abstract (ZH)**: 近期大规模视听模型（VLMs）的发展主要集中在英文上，对其他语言的关注度有限。为解决这一问题，我们引入了MEENA（也称为PersianMMMU），这是首个旨在评估波斯语VLMs在科学、推理和人文理解任务方面的数据集。我们的数据集包含约7,500个波斯语和3,000个英文问题，涵盖了推理、数学、物理、图表、文物和文学等多个主题。MEENA的关键特征包括：(1) 范围广泛的内容覆盖，从基础教育到高中教育，(2) 丰富元数据，包括难度级别和描述性答案，(3) 原创的波斯语文本，保留了文化细微差异，(4) 双语结构以评估跨语言性能，以及(5) 一系列多样实验，评估各种能力，包括整体性能、模型对图像的关注能力和生成幻觉的倾向。我们希望这一基准能够促进VLM能力的提升，超越英文语言。 

---
# ERF-BA-TFD+: A Multimodal Model for Audio-Visual Deepfake Detection 

**Title (ZH)**: ERF-BA-TFD+: 多模态音频-视觉深伪检测模型 

**Authors**: Xin Zhang, Jiaming Chu, Jian Zhao, Yuchu Jiang, Xu Yang, Lei Jin, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.17282)  

**Abstract**: Deepfake detection is a critical task in identifying manipulated multimedia content. In real-world scenarios, deepfake content can manifest across multiple modalities, including audio and video. To address this challenge, we present ERF-BA-TFD+, a novel multimodal deepfake detection model that combines enhanced receptive field (ERF) and audio-visual fusion. Our model processes both audio and video features simultaneously, leveraging their complementary information to improve detection accuracy and robustness. The key innovation of ERF-BA-TFD+ lies in its ability to model long-range dependencies within the audio-visual input, allowing it to better capture subtle discrepancies between real and fake content. In our experiments, we evaluate ERF-BA-TFD+ on the DDL-AV dataset, which consists of both segmented and full-length video clips. Unlike previous benchmarks, which focused primarily on isolated segments, the DDL-AV dataset allows us to assess the model's performance in a more comprehensive and realistic setting. Our method achieves state-of-the-art results on this dataset, outperforming existing techniques in terms of both accuracy and processing speed. The ERF-BA-TFD+ model demonstrated its effectiveness in the "Workshop on Deepfake Detection, Localization, and Interpretability," Track 2: Audio-Visual Detection and Localization (DDL-AV), and won first place in this competition. 

**Abstract (ZH)**: 基于增强感受野和音视频融合的多模态深仿生成分检测模型（ERF-BA-TFD+） 

---
# Federated Reinforcement Learning for Runtime Optimization of AI Applications in Smart Eyewears 

**Title (ZH)**: 智能眼镜中AI应用运行时优化的联邦 reinforcement 学习 

**Authors**: Hamta Sedghani, Abednego Wamuhindo Kambale, Federica Filippini, Francesca Palermo, Diana Trojaniello, Danilo Ardagna  

**Link**: [PDF](https://arxiv.org/pdf/2508.17262)  

**Abstract**: Extended reality technologies are transforming fields such as healthcare, entertainment, and education, with Smart Eye-Wears (SEWs) and Artificial Intelligence (AI) playing a crucial role. However, SEWs face inherent limitations in computational power, memory, and battery life, while offloading computations to external servers is constrained by network conditions and server workload variability. To address these challenges, we propose a Federated Reinforcement Learning (FRL) framework, enabling multiple agents to train collaboratively while preserving data privacy. We implemented synchronous and asynchronous federation strategies, where models are aggregated either at fixed intervals or dynamically based on agent progress. Experimental results show that federated agents exhibit significantly lower performance variability, ensuring greater stability and reliability. These findings underscore the potential of FRL for applications requiring robust real-time AI processing, such as real-time object detection in SEWs. 

**Abstract (ZH)**: 扩展现实技术正在transforming医疗、娱乐和教育等领域，智能眼镜（SEWs）和人工智能（AI）发挥着关键作用。然而，SEWs在计算能力、内存和电池寿命方面存在固有限制，将计算任务卸载到外部服务器又受限于网络条件和服务器工作负载的变异性。为应对这些挑战，我们提出了一种联邦强化学习（FRL）框架，使得多个代理能够协作训练并保护数据隐私。我们实现了同步和异步的联邦策略，模型要么在固定间隔，要么根据代理进度动态聚合。实验结果表明，联邦代理表现出显著更低的性能变异性，从而确保了更高的稳定性和可靠性。这些发现强调了FRL在需要强大实时AI处理的应用，如SEWs中的实时物体检测方面的潜力。 

---
# L-XAIDS: A LIME-based eXplainable AI framework for Intrusion Detection Systems 

**Title (ZH)**: L-XAIDS：一种基于LIME的可解释人工智能入侵检测系统框架 

**Authors**: Aoun E Muhammad, Kin-Choong Yow, Nebojsa Bacanin-Dzakula, Muhammad Attique Khan  

**Link**: [PDF](https://arxiv.org/pdf/2508.17244)  

**Abstract**: Recent developments in Artificial Intelligence (AI) and their applications in critical industries such as healthcare, fin-tech and cybersecurity have led to a surge in research in explainability in AI. Innovative research methods are being explored to extract meaningful insight from blackbox AI systems to make the decision-making technology transparent and interpretable. Explainability becomes all the more critical when AI is used in decision making in domains like fintech, healthcare and safety critical systems such as cybersecurity and autonomous vehicles. However, there is still ambiguity lingering on the reliable evaluations for the users and nature of transparency in the explanations provided for the decisions made by black-boxed AI. To solve the blackbox nature of Machine Learning based Intrusion Detection Systems, a framework is proposed in this paper to give an explanation for IDSs decision making. This framework uses Local Interpretable Model-Agnostic Explanations (LIME) coupled with Explain Like I'm five (ELI5) and Decision Tree algorithms to provide local and global explanations and improve the interpretation of IDSs. The local explanations provide the justification for the decision made on a specific input. Whereas, the global explanations provides the list of significant features and their relationship with attack traffic. In addition, this framework brings transparency in the field of ML driven IDS that might be highly significant for wide scale adoption of eXplainable AI in cyber-critical systems. Our framework is able to achieve 85 percent accuracy in classifying attack behaviour on UNSW-NB15 dataset, while at the same time displaying the feature significance ranking of the top 10 features used in the classification. 

**Abstract (ZH)**: Recent developments in Artificial Intelligence (AI) and their applications in critical industries such as healthcare, finance, and cybersecurity have led to a surge in research on AI explainability. Innovative research methods are being explored to extract meaningful insights from black-box AI systems, making decision-making technologies transparent and interpretable. Explainability is particularly crucial when AI is used in decision-making for domains such as finance, healthcare, and safety-critical systems like cybersecurity and autonomous vehicles. However, there is still ambiguity in reliable evaluations for users regarding the transparency in the explanations provided by black-box AI. To address the black-box nature of Machine Learning-based Intrusion Detection Systems (IDSs), this paper proposes a framework to explain IDSs’ decision-making processes. The framework utilizes Local Interpretable Model-Agnostic Explanations (LIME), Explain Like I'm Five (ELI5), and Decision Tree algorithms to provide both local and global explanations, thereby improving the interpretability of IDSs. Local explanations justify the decision made on a specific input, while global explanations provide a list of significant features and their relationship with attack traffic. Additionally, this framework introduces transparency in ML-driven IDS, which may be highly significant for the wide-scale adoption of Explainable AI in cyber-critical systems. Our framework achieves 85% accuracy in classifying attack behavior on the UNSW-NB15 dataset while simultaneously displaying the feature significance ranking of the top 10 features used in classification. 

---
# MC3G: Model Agnostic Causally Constrained Counterfactual Generation 

**Title (ZH)**: MC3G: 模型无关的因果约束反事实生成 

**Authors**: Sopam Dasgupta, Sadaf MD Halim, Joaquín Arias, Elmer Salazar, Gopal Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2508.17221)  

**Abstract**: Machine learning models increasingly influence decisions in high-stakes settings such as finance, law and hiring, driving the need for transparent, interpretable outcomes. However, while explainable approaches can help understand the decisions being made, they may inadvertently reveal the underlying proprietary algorithm: an undesirable outcome for many practitioners. Consequently, it is crucial to balance meaningful transparency with a form of recourse that clarifies why a decision was made and offers actionable steps following which a favorable outcome can be obtained. Counterfactual explanations offer a powerful mechanism to address this need by showing how specific input changes lead to a more favorable prediction. We propose Model-Agnostic Causally Constrained Counterfactual Generation (MC3G), a novel framework that tackles limitations in the existing counterfactual methods. First, MC3G is model-agnostic: it approximates any black-box model using an explainable rule-based surrogate model. Second, this surrogate is used to generate counterfactuals that produce a favourable outcome for the original underlying black box model. Third, MC3G refines cost computation by excluding the ``effort" associated with feature changes that occur automatically due to causal dependencies. By focusing only on user-initiated changes, MC3G provides a more realistic and fair representation of the effort needed to achieve a favourable outcome. We show that MC3G delivers more interpretable and actionable counterfactual recommendations compared to existing techniques all while having a lower cost. Our findings highlight MC3G's potential to enhance transparency, accountability, and practical utility in decision-making processes that incorporate machine-learning approaches. 

**Abstract (ZH)**: 基于因果约束的模型无关反事实生成（MC3G）：提高机器学习决策过程的透明性、问责制和实用性 

---
# Reinforcement Learning enhanced Online Adaptive Clinical Decision Support via Digital Twin powered Policy and Treatment Effect optimized Reward 

**Title (ZH)**: 基于数字孪生驱动策略与治疗效果优化奖励的强化学习增强在线自适应临床决策支持 

**Authors**: Xinyu Qin, Ruiheng Yu, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17212)  

**Abstract**: Clinical decision support must adapt online under safety constraints. We present an online adaptive tool where reinforcement learning provides the policy, a patient digital twin provides the environment, and treatment effect defines the reward. The system initializes a batch-constrained policy from retrospective data and then runs a streaming loop that selects actions, checks safety, and queries experts only when uncertainty is high. Uncertainty comes from a compact ensemble of five Q-networks via the coefficient of variation of action values with a $\tanh$ compression. The digital twin updates the patient state with a bounded residual rule. The outcome model estimates immediate clinical effect, and the reward is the treatment effect relative to a conservative reference with a fixed z-score normalization from the training split. Online updates operate on recent data with short runs and exponential moving averages. A rule-based safety gate enforces vital ranges and contraindications before any action is applied. Experiments in a synthetic clinical simulator show low latency, stable throughput, a low expert query rate at fixed safety, and improved return against standard value-based baselines. The design turns an offline policy into a continuous, clinician-supervised system with clear controls and fast adaptation. 

**Abstract (ZH)**: 临床决策支持需在安全约束下在线适应。我们提出了一种在线自适应工具，其中强化学习提供策略，患者数字孪生提供环境，治疗效果定义奖励。该系统从 retros 归档数据初始化批量约束策略，然后运行一个流式循环，该循环选择行动、检查安全并在不确定性高时查询专家。不确定性来自通过 tanh 压缩动作值的紧凑型五元 Q 网络集合的标准差。数字孪生利用有界残差规则更新患者状态。结果模型估计即时临床效果，奖励是相对于保守参考的治疗效果，后者采用固定 z 分数标准化。在线更新基于最近数据运行短期操作，并使用指数加权平均值。基于规则的安全门控确保在任何行动之前强制执行关键范围和禁忌症。在合成临床模拟器中的实验表明，低延迟、稳定吞吐量、固定安全下的低专家查询率和优于标准价值基准的较高回报。该设计将离线策略转变为连续的、由临床医生监督的系统，具有明确的控制和快速适应性。 

---
# Explainable Counterfactual Reasoning in Depression Medication Selection at Multi-Levels (Personalized and Population) 

**Title (ZH)**: 抑郁药物选择多维度（个性化和群体层面）的可解释反事实推理 

**Authors**: Xinyu Qin, Mark H. Chignell, Alexandria Greifenberger, Sachinthya Lokuge, Elssa Toumeh, Tia Sternat, Martin Katzman, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17207)  

**Abstract**: Background: This study investigates how variations in Major Depressive Disorder (MDD) symptoms, quantified by the Hamilton Rating Scale for Depression (HAM-D), causally influence the prescription of SSRIs versus SNRIs. Methods: We applied explainable counterfactual reasoning with counterfactual explanations (CFs) to assess the impact of specific symptom changes on antidepressant choice. Results: Among 17 binary classifiers, Random Forest achieved highest performance (accuracy, F1, precision, recall, ROC-AUC near 0.85). Sample-based CFs revealed both local and global feature importance of individual symptoms in medication selection. Conclusions: Counterfactual reasoning elucidates which MDD symptoms most strongly drive SSRI versus SNRI selection, enhancing interpretability of AI-based clinical decision support systems. Future work should validate these findings on more diverse cohorts and refine algorithms for clinical deployment. 

**Abstract (ZH)**: 背景：本研究探讨了通过Hamilton抑郁评定量表（HAM-D）量化的主要抑郁症症状变异如何因果影响选择SSRIs与SNRIs。方法：我们使用可解释的反事实推理与反事实解释（CFs）来评估特定症状变化对抗抑郁药选择的影响。结果：在17个二元分类器中，随机森林表现最佳（准确率、F1值、精确率、召回率、ROC-AUC接近0.85）。基于样本的反事实解释揭示了个体症状在药物选择中的局部和全局特征重要性。结论：反事实推理阐明了哪些主要抑郁症症状最强烈地驱动SSRI与SNRI的选择，增强了基于AI的临床决策支持系统的可解释性。未来工作应在更多样化的队列中验证这些发现并细化临床部署算法。 

---
# Large Language Model-Based Automatic Formulation for Stochastic Optimization Models 

**Title (ZH)**: 基于大型语言模型的随机优化模型自动构建方法 

**Authors**: Amirreza Talebi  

**Link**: [PDF](https://arxiv.org/pdf/2508.17200)  

**Abstract**: This paper presents the first integrated systematic study on the performance of large language models (LLMs), specifically ChatGPT, to automatically formulate and solve stochastic optimiza- tion problems from natural language descriptions. Focusing on three key categories, joint chance- constrained models, individual chance-constrained models, and two-stage stochastic linear programs (SLP-2), we design several prompts that guide ChatGPT through structured tasks using chain-of- thought and modular reasoning. We introduce a novel soft scoring metric that evaluates the struc- tural quality and partial correctness of generated models, addressing the limitations of canonical and execution-based accuracy. Across a diverse set of stochastic problems, GPT-4-Turbo outperforms other models in partial score, variable matching, and objective accuracy, with cot_s_instructions and agentic emerging as the most effective prompting strategies. Our findings reveal that with well-engineered prompts and multi-agent collaboration, LLMs can facilitate specially stochastic formulations, paving the way for intelligent, language-driven modeling pipelines in stochastic opti- mization. 

**Abstract (ZH)**: 本文首次对大型语言模型（LLMs），特别是ChatGPT，进行集成系统的研究，以自动从自然语言描述中制定和解决随机优化问题。我们专注于联合机会约束模型、个体机会约束模型和两阶段随机线性规划（SLP-2）三大关键类别，设计了几种引导ChatGPT完成结构化任务的提示，采用链式推理和模块化推理。我们引入了一种新颖的软评分度量标准，用于评估生成模型的结构性质量和部分正确性，克服了传统和执行基于准确性的局限性。在一系列多样化的随机问题中，GPT-4-Turbo在部分得分、变量匹配和目标准确性方面优于其他模型，cot_s_instructions和agentic提示策略最为有效。我们的研究发现，通过精心设计的提示和多智能体协作，LLMs能够促进特有随机模型的制定，为随机优化的语言驱动建模流水线开辟了途径。 

---
# From reactive to cognitive: brain-inspired spatial intelligence for embodied agents 

**Title (ZH)**: 从反应式到认知式：受脑启发的空间智能对具身代理的应用 

**Authors**: Shouwei Ruan, Liyuan Wang, Caixin Kang, Qihui Zhu, Songming Liu, Xingxing Wei, Hang Su  

**Link**: [PDF](https://arxiv.org/pdf/2508.17198)  

**Abstract**: Spatial cognition enables adaptive goal-directed behavior by constructing internal models of space. Robust biological systems consolidate spatial knowledge into three interconnected forms: \textit{landmarks} for salient cues, \textit{route knowledge} for movement trajectories, and \textit{survey knowledge} for map-like representations. While recent advances in multi-modal large language models (MLLMs) have enabled visual-language reasoning in embodied agents, these efforts lack structured spatial memory and instead operate reactively, limiting their generalization and adaptability in complex real-world environments. Here we present Brain-inspired Spatial Cognition for Navigation (BSC-Nav), a unified framework for constructing and leveraging structured spatial memory in embodied agents. BSC-Nav builds allocentric cognitive maps from egocentric trajectories and contextual cues, and dynamically retrieves spatial knowledge aligned with semantic goals. Integrated with powerful MLLMs, BSC-Nav achieves state-of-the-art efficacy and efficiency across diverse navigation tasks, demonstrates strong zero-shot generalization, and supports versatile embodied behaviors in the real physical world, offering a scalable and biologically grounded path toward general-purpose spatial intelligence. 

**Abstract (ZH)**: 空间认知通过构建空间内部模型来实现适应性目标导向行为。生物系统将空间知识整合为三种相互连接的形式：地标作为显著线索、路径知识用于运动轨迹、航图知识用于地图式的表示。尽管多模态大语言模型（MLLMs）的近期进展已在具身智能体中实现了视觉-语言推理，但在构建结构化空间记忆方面仍存不足，导致其在复杂现实环境中的泛化能力和适应性受限。在这里，我们提出了借鉴大脑的空间认知框架用于导航 (BSC-Nav)，这是一种构建和利用具身智能体结构化空间记忆的统一框架。BSC-Nav 从以自我为中心的轨迹和上下文线索构建他中心的空间认知地图，并动态检索与语义目标对齐的空间知识。结合强大的 MLLMs，BSC-Nav 在多种导航任务中达到了最先进的效果和效率，并展示了强大的零样本泛化能力，支持在实际物理世界中的多功能具身行为，为通用空间智能提供了可扩展且生物学依据的路径。 

---
# PosterGen: Aesthetic-Aware Paper-to-Poster Generation via Multi-Agent LLMs 

**Title (ZH)**: PosterGen: 基于多智能体LLM的美观意识论文转海报生成 

**Authors**: Zhilin Zhang, Xiang Zhang, Jiaqi Wei, Yiwei Xu, Chenyu You  

**Link**: [PDF](https://arxiv.org/pdf/2508.17188)  

**Abstract**: Multi-agent systems built upon large language models (LLMs) have demonstrated remarkable capabilities in tackling complex compositional tasks. In this work, we apply this paradigm to the paper-to-poster generation problem, a practical yet time-consuming process faced by researchers preparing for conferences. While recent approaches have attempted to automate this task, most neglect core design and aesthetic principles, resulting in posters that require substantial manual refinement. To address these design limitations, we propose PosterGen, a multi-agent framework that mirrors the workflow of professional poster designers. It consists of four collaborative specialized agents: (1) Parser and Curator agents extract content from the paper and organize storyboard; (2) Layout agent maps the content into a coherent spatial layout; (3) Stylist agents apply visual design elements such as color and typography; and (4) Renderer composes the final poster. Together, these agents produce posters that are both semantically grounded and visually appealing. To evaluate design quality, we introduce a vision-language model (VLM)-based rubric that measures layout balance, readability, and aesthetic coherence. Experimental results show that PosterGen consistently matches in content fidelity, and significantly outperforms existing methods in visual designs, generating posters that are presentation-ready with minimal human refinements. 

**Abstract (ZH)**: 基于大型语言模型的多agent系统在论文转海报生成中的应用：一种遵循专业设计师工作流程的多agent框架及其评价方法 

---
# MaRVL-QA: A Benchmark for Mathematical Reasoning over Visual Landscapes 

**Title (ZH)**: MaRVL-QA：视觉景观上的数学推理基准 

**Authors**: Nilay Pande, Sahiti Yerramilli, Jayant Sravan Tamarapalli, Rynaa Grover  

**Link**: [PDF](https://arxiv.org/pdf/2508.17180)  

**Abstract**: A key frontier for Multimodal Large Language Models (MLLMs) is the ability to perform deep mathematical and spatial reasoning directly from images, moving beyond their established success in semantic description. Mathematical surface plots provide a rigorous testbed for this capability, as they isolate the task of reasoning from the semantic noise common in natural images. To measure progress on this frontier, we introduce MaRVL-QA (Mathematical Reasoning over Visual Landscapes), a new benchmark designed to quantitatively evaluate these core reasoning skills. The benchmark comprises two novel tasks: Topological Counting, identifying and enumerating features like local maxima; and Transformation Recognition, recognizing applied geometric transformations. Generated from a curated library of functions with rigorous ambiguity filtering, our evaluation on MaRVL-QA reveals that even state-of-the-art MLLMs struggle significantly, often resorting to superficial heuristics instead of robust spatial reasoning. MaRVL-QA provides a challenging new tool for the research community to measure progress, expose model limitations, and guide the development of MLLMs with more profound reasoning abilities. 

**Abstract (ZH)**: 多模态大语言模型的关键前沿领域是能够直接从图像中进行深入的数学和空间推理，超越其在语义描述上的已有成就。数学曲面图提供了这种能力的严格测试平台，因为它们将推理任务与自然图像中常见的语义噪声隔离开来。为了衡量这一领域的进展，我们引入了MaRVL-QA（基于视觉景观的数学推理问题集），这是一个新的基准，旨在定量评估这些核心推理能力。该基准包括两个新的任务：拓扑计数，识别和枚举局部极大值等特征；以及几何变换识别，识别应用的几何变换。来源于具有严格模糊性过滤的函数库，我们在MaRVL-QA上的评估表明，即使是最先进的多模态大语言模型也面临着显著挑战，常常依赖于表面的启发法而非稳健的空间推理。MaRVL-QA为研究社区提供了一个具有挑战性的新工具，用于衡量进展、揭示模型的局限性，并指导开发具有更深层次推理能力的多模态大语言模型。 

---
# Rethinking How AI Embeds and Adapts to Human Values: Challenges and Opportunities 

**Title (ZH)**: 重新思考AI如何嵌入和适应人类价值观：挑战与机遇 

**Authors**: Sz-Ting Tzeng, Frank Dignum  

**Link**: [PDF](https://arxiv.org/pdf/2508.17104)  

**Abstract**: The concepts of ``human-centered AI'' and ``value-based decision'' have gained significant attention in both research and industry. However, many critical aspects remain underexplored and require further investigation. In particular, there is a need to understand how systems incorporate human values, how humans can identify these values within systems, and how to minimize the risks of harm or unintended consequences. In this paper, we highlight the need to rethink how we frame value alignment and assert that value alignment should move beyond static and singular conceptions of values. We argue that AI systems should implement long-term reasoning and remain adaptable to evolving values. Furthermore, value alignment requires more theories to address the full spectrum of human values. Since values often vary among individuals or groups, multi-agent systems provide the right framework for navigating pluralism, conflict, and inter-agent reasoning about values. We identify the challenges associated with value alignment and indicate directions for advancing value alignment research. In addition, we broadly discuss diverse perspectives of value alignment, from design methodologies to practical applications. 

**Abstract (ZH)**: “以人为本的AI”和“基于价值的决策”概念在研究和行业领域获得了广泛关注，但许多关键方面仍有待探索并需要进一步研究。特别是，需要理解系统如何融入人类价值观、人类如何在系统中识别这些价值观以及如何最小化危害或意外后果的风险。在本文中，我们强调需要重新思考价值对齐的方式，并认为价值对齐应超越静态和单一的价值观观念。我们argue认为，AI系统应采用长期推理，并保持对演变价值观的适应性。此外，由于价值观在个体或群体之间往往不同，多智能体系统提供了应对多元主义、冲突以及智能体间关于价值观的推理的适当框架。我们指出了价值对齐面临的挑战，并指出了推进价值对齐研究的方向。此外，我们从设计方法论到实际应用广泛讨论了价值对齐的多种视角。 

---
# PowerChain: Automating Distribution Grid Analysis with Agentic AI Workflows 

**Title (ZH)**: PowerChain：使用自主人工智能工作流的配电网络分析自动化 

**Authors**: Emmanuel O. Badmus, Peng Sang, Dimitrios Stamoulis, Amritanshu Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2508.17094)  

**Abstract**: Due to the rapid pace of electrification and decarbonization, distribution grid (DG) operation and planning are becoming more complex, necessitating advanced computational analyses to ensure grid reliability and resilience. State-of-the-art DG analyses rely on disparate workflows of complex models, functions, and data pipelines, which require expert knowledge and are challenging to automate. Many small-scale utilities and cooperatives lack a large R&D workforce and therefore cannot use advanced analysis at scale. To address this gap, we develop a novel agentic AI system, PowerChain, to solve unseen DG analysis tasks via automated agentic orchestration and large language models (LLMs) function-calling. Given a natural language query, PowerChain dynamically generates and executes an ordered sequence of domain-aware functions guided by the semantics of an expert-built power systems function pool and a select reference set of known, expert-generated workflow-query pairs. Our results show that PowerChain can produce expert-level workflows with both GPT-5 and open-source Qwen models on complex, unseen DG analysis tasks operating on real utility data. 

**Abstract (ZH)**: 由于电气化进程和去碳化进程的加速，配电网络（DG）的运行与规划变得更加复杂，需要先进的计算分析来确保电网的可靠性和韧性。当前先进的DG分析依赖于复杂模型、函数和数据管道的不统一工作流程，这些流程需要专家知识并且难以自动化。许多小型电力企业和合作社缺乏大规模的研发团队，因此无法进行大规模的高级分析。为了弥补这一缺口，我们开发了一种新的代理型AI系统PowerChain，通过自动代理编排和大规模语言模型（LLMs）函数调用解决未知的DG分析任务。给定自然语言查询，PowerChain动态生成并执行由专家构建的电力系统函数池语义指导下的、按顺序排列的领域特定函数序列，并参考一组选定的专家生成的工作流-查询对。我们的结果表明，PowerChain能够在复杂、未知的DG分析任务上生成与GPT-5和开源Qwen模型同等水平的工作流，这些任务基于实际的电力数据进行操作。 

---
# Solving the Min-Max Multiple Traveling Salesmen Problem via Learning-Based Path Generation and Optimal Splitting 

**Title (ZH)**: 基于学习路径生成和最优分割的MinMax多旅行售货员问题求解 

**Authors**: Wen Wang, Xiangchen Wu, Liang Wang, Hao Hu, Xianping Tao, Linghao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17087)  

**Abstract**: This study addresses the Min-Max Multiple Traveling Salesmen Problem ($m^3$-TSP), which aims to coordinate tours for multiple salesmen such that the length of the longest tour is minimized. Due to its NP-hard nature, exact solvers become impractical under the assumption that $P \ne NP$. As a result, learning-based approaches have gained traction for their ability to rapidly generate high-quality approximate solutions. Among these, two-stage methods combine learning-based components with classical solvers, simplifying the learning objective. However, this decoupling often disrupts consistent optimization, potentially degrading solution quality. To address this issue, we propose a novel two-stage framework named \textbf{Generate-and-Split} (GaS), which integrates reinforcement learning (RL) with an optimal splitting algorithm in a joint training process. The splitting algorithm offers near-linear scalability with respect to the number of cities and guarantees optimal splitting in Euclidean space for any given path. To facilitate the joint optimization of the RL component with the algorithm, we adopt an LSTM-enhanced model architecture to address partial observability. Extensive experiments show that the proposed GaS framework significantly outperforms existing learning-based approaches in both solution quality and transferability. 

**Abstract (ZH)**: 本研究探讨了Min-Max Multiple Traveling Salesmen Problem ($m^3$-TSP)，目标是协调多名销售人员的行程，使得最长的行程长度最小化。由于该问题的NP难性质，在P≠NP的假设下，精确求解器变得不实用。因此，基于学习的方法因其能够快速生成高质量近似解而受到关注。在这类方法中，两阶段方法将基于学习的组件与经典求解器结合，简化了学习目标。然而，这种分离往往会打断一致的优化，可能降低解的质量。为了解决这一问题，我们提出了一种名为Generate-and-Split（GaS）的新型两阶段框架，该框架将强化学习（RL）与最优分割算法结合，在联合训练过程中进行集成。分割算法在城市数量方面具有近线性可扩展性，并能够确保在欧几里得空间中对任意给定路径实现最优分割。为促进RL组件与算法的联合优化，我们采用LSTM增强的模型架构来解决部分可观测性问题。大量实验表明，提出的GaS框架在解的质量和可迁移性方面显著优于现有基于学习的方法。 

---
# WebSight: A Vision-First Architecture for Robust Web Agents 

**Title (ZH)**: WebSight: 以视觉为主的第一代架构稳健的Web代理 

**Authors**: Tanvir Bhathal, Asanshay Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2508.16987)  

**Abstract**: We introduce WebSight, a vision-based autonomous web agent, designed to interact with web environments purely through visual perception, eliminating dependence on HTML or DOM-based inputs. Central to our approach we introduce our new model, WebSight-7B, a fine-tuned vision-language model optimized for UI element interaction, trained using LoRA on a web-focused subset of the Wave-UI-25K dataset. WebSight integrates this model into a modular multi-agent architecture, comprising planning, reasoning, vision-action, and verification agents, coordinated through an episodic memory mechanism.
WebSight-7B achieves a top-1 accuracy of 58.84% on the Showdown Clicks benchmark, outperforming several larger generalist models while maintaining lower latency. The full WebSight agent achieves a 68.0% success rate on the WebVoyager benchmark, surpassing systems from labs such as OpenAI (61.0%) and HCompany (Runner H, 67.0%). Among tasks completed, WebSight answers correctly 97.14% of the time, indicating high precision. Together, WebSight and WebSight-7B establish a new standard for interpretable, robust, and efficient visual web navigation. 

**Abstract (ZH)**: 基于视觉的自主网络代理WebSight及其WebSight-7B模型：一种用于UI元素交互的精调视觉语言模型 

---
# Complexity in finitary argumentation (extended version) 

**Title (ZH)**: 有限论辩的复杂性（扩展版本） 

**Authors**: Uri Andrews, Luca San Mauro  

**Link**: [PDF](https://arxiv.org/pdf/2508.16986)  

**Abstract**: Abstract argumentation frameworks (AFs) provide a formal setting to analyze many forms of reasoning with conflicting information. While the expressiveness of general infinite AFs make them a tempting tool for modeling many kinds of reasoning scenarios, the computational intractability of solving infinite AFs limit their use, even in many theoretical applications.
We investigate the complexity of computational problems related to infinite but finitary argumentations frameworks, that is, infinite AFs where each argument is attacked by only finitely many others. Our results reveal a surprising scenario. On one hand, we see that the assumption of being finitary does not automatically guarantee a drop in complexity. However, for the admissibility-based semantics, we find a remarkable combinatorial constraint which entails a dramatic decrease in complexity.
We conclude that for many forms of reasoning, the finitary infinite AFs provide a natural setting for reasoning which balances well the competing goals of being expressive enough to be applied to many reasoning settings while being computationally tractable enough for the analysis within the framework to be useful. 

**Abstract (ZH)**: 无穷但可数的论辩框架的计算复杂性研究：一种表达性和计算效率之间的平衡 

---
# RADAR: A Reasoning-Guided Attribution Framework for Explainable Visual Data Analysis 

**Title (ZH)**: RADAR：一种基于推理的可解释视觉数据分析归因框架 

**Authors**: Anku Rani, Aparna Garimella, Apoorv Saxena, Balaji Vasan Srinivasan, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16850)  

**Abstract**: Data visualizations like charts are fundamental tools for quantitative analysis and decision-making across fields, requiring accurate interpretation and mathematical reasoning. The emergence of Multimodal Large Language Models (MLLMs) offers promising capabilities for automated visual data analysis, such as processing charts, answering questions, and generating summaries. However, they provide no visibility into which parts of the visual data informed their conclusions; this black-box nature poses significant challenges to real-world trust and adoption. In this paper, we take the first major step towards evaluating and enhancing the capabilities of MLLMs to attribute their reasoning process by highlighting the specific regions in charts and graphs that justify model answers. To this end, we contribute RADAR, a semi-automatic approach to obtain a benchmark dataset comprising 17,819 diverse samples with charts, questions, reasoning steps, and attribution annotations. We also introduce a method that provides attribution for chart-based mathematical reasoning. Experimental results demonstrate that our reasoning-guided approach improves attribution accuracy by 15% compared to baseline methods, and enhanced attribution capabilities translate to stronger answer generation, achieving an average BERTScore of $\sim$ 0.90, indicating high alignment with ground truth responses. This advancement represents a significant step toward more interpretable and trustworthy chart analysis systems, enabling users to verify and understand model decisions through reasoning and attribution. 

**Abstract (ZH)**: 多模态大型语言模型在图表解释推理能力评估与提升的研究 

---
# Quantifying Sycophancy as Deviations from Bayesian Rationality in LLMs 

**Title (ZH)**: 量化逢迎行为作为大型语言模型中贝叶斯理性偏差的程度 

**Authors**: Katherine Atwell, Pedram Heydari, Anthony Sicilia, Malihe Alikhani  

**Link**: [PDF](https://arxiv.org/pdf/2508.16846)  

**Abstract**: Sycophancy, or overly agreeable or flattering behavior, is a documented issue in large language models (LLMs), and is critical to understand in the context of human/AI collaboration. Prior works typically quantify sycophancy by measuring shifts in behavior or impacts on accuracy, but neither metric characterizes shifts in rationality, and accuracy measures can only be used in scenarios with a known ground truth. In this work, we utilize a Bayesian framework to quantify sycophancy as deviations from rational behavior when presented with user perspectives, thus distinguishing between rational and irrational updates based on the introduction of user perspectives. In comparison to other methods, this approach allows us to characterize excessive behavioral shifts, even for tasks that involve inherent uncertainty or do not have a ground truth. We study sycophancy for 3 different tasks, a combination of open-source and closed LLMs, and two different methods for probing sycophancy. We also experiment with multiple methods for eliciting probability judgments from LLMs. We hypothesize that probing LLMs for sycophancy will cause deviations in LLMs' predicted posteriors that will lead to increased Bayesian error. Our findings indicate that: 1) LLMs are not Bayesian rational, 2) probing for sycophancy results in significant increases to the predicted posterior in favor of the steered outcome, 3) sycophancy sometimes results in increased Bayesian error, and in a small number of cases actually decreases error, and 4) changes in Bayesian error due to sycophancy are not strongly correlated in Brier score, suggesting that studying the impact of sycophancy on ground truth alone does not fully capture errors in reasoning due to sycophancy. 

**Abstract (ZH)**: overly agreeable或奉承行为在大型语言模型（LLMs）中是一个已知问题，并且在人类/AI协作的背景下理解这一点至关重要。先前的工作通常通过衡量行为变化或对准确率的影响来量化奉承行为，但这些度量标准都不能刻画理性变化，而准确率度量只能在有已知真实情况下使用。在本工作中，我们利用贝叶斯框架量化当面对用户视角时的奉承行为作为理性行为的偏差，从而根据用户视角的引入区分理性和非理性的更新。与其它方法相比，此方法允许我们表征行为变化的过度现象，即使对于包含内在不确定性或没有真实情况的任务也是如此。我们研究了三个不同的任务，包括开源和闭源LLMs，并使用两种不同的方法来探测奉承行为。我们还尝试了多种方法从LLMs中获取概率判断。我们假设探测LLMs的奉承行为会导致预测后验概率的变化，从而增加贝叶斯误差。我们的发现表明：1) LLMs不是贝叶斯理性，2) 探测奉承行为导致预测后验概率显著增加，倾向于导向行为的结果，3) 奉承行为有时会导致贝叶斯误差增加，但在少数情况下实际上减少了误差，4) 由于奉承行为导致的贝叶斯误差变化在Brier分数中并不强烈相关，这表明单独研究奉承行为对真实情况的影响并不能充分捕捉由奉承行为引起的推理错误。 

---
# Route-and-Execute: Auditable Model-Card Matching and Specialty-Level Deployment 

**Title (ZH)**: 执行路线：可审计的模型卡匹配与专科级部署 

**Authors**: Shayan Vassef, Soorya Ram Shimegekar, Abhay Goyal, Koustuv Saha, Pi Zonooz, Navin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.16839)  

**Abstract**: Clinical workflows are fragmented as a patchwork of scripts and task-specific networks that often handle triage, task selection, and model deployment. These pipelines are rarely streamlined for data science pipeline, reducing efficiency and raising operational costs. Workflows also lack data-driven model identification (from imaging/tabular inputs) and standardized delivery of model outputs. In response, we present a practical, healthcare-first framework that uses a single vision-language model (VLM) in two complementary roles. First (Solution 1), the VLM acts as an aware model-card matcher that routes an incoming image to the appropriate specialist model via a three-stage workflow (modality -> primary abnormality -> model-card id). Checks are provided by (i) stagewise prompts that allow early exit via None/Normal/Other and (ii) a stagewise answer selector that arbitrates between the top-2 candidates at each stage, reducing the chance of an incorrect selection and aligning the workflow with clinical risk tolerance. Second (Solution 2), we fine-tune the VLM on specialty-specific datasets ensuring a single model covers multiple downstream tasks within each specialty, maintaining performance while simplifying deployment. Across gastroenterology, hematology, ophthalmology, and pathology, our single-model deployment matches or approaches specialized baselines.
Compared with pipelines composed of many task-specific agents, this approach shows that one VLM can both decide and do. It may reduce effort by data scientists, shorten monitoring, increase the transparency of model selection (with per-stage justifications), and lower integration overhead. 

**Abstract (ZH)**: 临床工作流程碎片化为一系列脚本和任务特定网络的拼贴，常用于处理分诊、任务选择和模型部署。这些管道很少专门针对数据科学管道优化，从而降低了效率并增加了运营成本。工作流程中缺乏基于数据的模型识别（从影像/表格式输入）以及模型输出的标准交付。为此，我们提出了一种实用的、以医疗健康为主的框架，该框架利用单一的视觉-语言模型（VLM）在两种互补的角色中发挥作用。首先（解决方案1），VLM 作为一个知情的模型卡匹配器，通过三阶段工作流（模态->主要异常->模型卡ID）将输入图像路由到适当的专科模型，通过阶段提示允许早期退出（无/正常/其他）和阶段回答选择器在每个阶段选择前二名候选人之间仲裁，从而减少错误选择并使工作流与临床风险容忍度保持一致。其次（解决方案2），我们针对各专科特定的数据集微调VLM，确保一个模型能够涵盖每个专科内的多个下游任务，从而保持性能并简化部署。在胃肠病学、血液学、眼科学和病理学领域，我们的单模型部署匹配或接近专科基线。与由多个任务特定代理组成的管道相比，这种方法显示了一个VLM可以两者兼备。它可以通过减少数据科学家的努力、缩短监控时间、增加模型选择的透明度（每个阶段的解释）并降低集成成本来发挥作用。 

---
# PuzzleJAX: A Benchmark for Reasoning and Learning 

**Title (ZH)**: PuzzleJAX: 一个推理与学习的标准评测基准 

**Authors**: Sam Earle, Graham Todd, Yuchen Li, Ahmed Khalifa, Muhammad Umair Nasir, Zehua Jiang, Andrzej Banburski-Fahey, Julian Togelius  

**Link**: [PDF](https://arxiv.org/pdf/2508.16821)  

**Abstract**: We introduce PuzzleJAX, a GPU-accelerated puzzle game engine and description language designed to support rapid benchmarking of tree search, reinforcement learning, and LLM reasoning abilities. Unlike existing GPU-accelerated learning environments that provide hard-coded implementations of fixed sets of games, PuzzleJAX allows dynamic compilation of any game expressible in its domain-specific language (DSL). This DSL follows PuzzleScript, which is a popular and accessible online game engine for designing puzzle games. In this paper, we validate in PuzzleJAX several hundred of the thousands of games designed in PuzzleScript by both professional designers and casual creators since its release in 2013, thereby demonstrating PuzzleJAX's coverage of an expansive, expressive, and human-relevant space of tasks. By analyzing the performance of search, learning, and language models on these games, we show that PuzzleJAX can naturally express tasks that are both simple and intuitive to understand, yet often deeply challenging to master, requiring a combination of control, planning, and high-level insight. 

**Abstract (ZH)**: PuzzleJAX：一种加速的谜题游戏引擎及描述语言，用于支持树搜索、强化学习和LLM推理能力的快速基准测试 

---
# Evaluation and LLM-Guided Learning of ICD Coding Rationales 

**Title (ZH)**: LLM引导的学习ICD编码推理评价 

**Authors**: Mingyang Li, Viktor Schlegel, Tingting Mu, Wuraola Oyewusi, Kai Kang, Goran Nenadic  

**Link**: [PDF](https://arxiv.org/pdf/2508.16777)  

**Abstract**: Automated clinical coding involves mapping unstructured text from Electronic Health Records (EHRs) to standardized code systems such as the International Classification of Diseases (ICD). While recent advances in deep learning have significantly improved the accuracy and efficiency of ICD coding, the lack of explainability in these models remains a major limitation, undermining trust and transparency. Current explorations about explainability largely rely on attention-based techniques and qualitative assessments by physicians, yet lack systematic evaluation using consistent criteria on high-quality rationale datasets, as well as dedicated approaches explicitly trained to generate rationales for further enhancing explanation. In this work, we conduct a comprehensive evaluation of the explainability of the rationales for ICD coding through two key lenses: faithfulness that evaluates how well explanations reflect the model's actual reasoning and plausibility that measures how consistent the explanations are with human expert judgment. To facilitate the evaluation of plausibility, we construct a new rationale-annotated dataset, offering denser annotations with diverse granularity and aligns better with current clinical practice, and conduct evaluation across three types of rationales of ICD coding. Encouraged by the promising plausibility of LLM-generated rationales for ICD coding, we further propose new rationale learning methods to improve the quality of model-generated rationales, where rationales produced by prompting LLMs with/without annotation examples are used as distant supervision signals. We empirically find that LLM-generated rationales align most closely with those of human experts. Moreover, incorporating few-shot human-annotated examples not only further improves rationale generation but also enhances rationale-learning approaches. 

**Abstract (ZH)**: 自动临床编码涉及将电子健康记录（EHRs）中的非结构化文本映射到标准化代码系统，如国际疾病分类（ICD）。尽管近年来深度学习的进步显著提高了ICD编码的准确性和效率，但这些模型缺乏可解释性仍然是一个主要限制，这削弱了人们对模型的信任和透明度。当前关于可解释性的探索主要依赖于注意力机制技术和医师的定性评估，但缺乏使用一致标准对高质量理性数据集进行系统评估，以及专门训练以生成进一步增强解释的理性生成方法。在本文中，我们通过两个关键视角对ICD编码的理性解释的可解释性进行全面评估：忠实度，评估解释如何反映模型的实际推理；以及合理性，衡量解释与人类专家判断的一致性程度。为了便于评估合理性，我们构建了一个新的带有注释的理性数据集，提供了更密集的、粒度多样化的注释，并且更好地与当前临床实践相契合，进而对ICD编码的三种类型注释进行了评估。受到LLM生成的ICD编码理性注释具有较高合理性的启发，我们进一步提出新的注释学习方法以提高模型生成理性注释的质量，其中，通过提示LLM生成带有/不带有注释示例的理性注释被用作远处监督信号。我们的实证研究表明，LLM生成的理性注释最接近人类专家的理性注释。此外，引入少量的人工标注示例不仅进一步提高了注释生成的质量，还增强了注释学习方法。 

---
# Explainable AI for Predicting and Understanding Mathematics Achievement: A Cross-National Analysis of PISA 2018 

**Title (ZH)**: 可解释的人工智能在预测和理解数学成就中的应用：基于PISA 2018的跨国家分析 

**Authors**: Liu Liu, Rui Dai  

**Link**: [PDF](https://arxiv.org/pdf/2508.16747)  

**Abstract**: Understanding the factors that shape students' mathematics performance is vital for designing effective educational policies. This study applies explainable artificial intelligence (XAI) techniques to PISA 2018 data to predict math achievement and identify key predictors across ten countries (67,329 students). We tested four models: Multiple Linear Regression (MLR), Random Forest (RF), CATBoost, and Artificial Neural Networks (ANN), using student, family, and school variables. Models were trained on 70% of the data (with 5-fold cross-validation) and tested on 30%, stratified by country. Performance was assessed with R^2 and Mean Absolute Error (MAE). To ensure interpretability, we used feature importance, SHAP values, and decision tree visualizations. Non-linear models, especially RF and ANN, outperformed MLR, with RF balancing accuracy and generalizability. Key predictors included socio-economic status, study time, teacher motivation, and students' attitudes toward mathematics, though their impact varied across countries. Visual diagnostics such as scatterplots of predicted vs actual scores showed RF and CATBoost aligned closely with actual performance. Findings highlight the non-linear and context-dependent nature of achievement and the value of XAI in educational research. This study uncovers cross-national patterns, informs equity-focused reforms, and supports the development of personalized learning strategies. 

**Abstract (ZH)**: 理解塑造学生数学表现的因素对于设计有效的教育政策至关重要。本研究采用可解释的人工智能（XAI）技术对PISA 2018数据进行分析，预测数学成就并识别十个参与国家（67,329名学生）的关键预测因子。我们测试了四种模型：多元线性回归（MLR）、随机森林（RF）、CATBoost和人工神经网络（ANN），使用学生、家庭和学校变量。模型在70%的数据上进行训练（5折交叉验证），并在30%的数据上进行测试，按国家分层。性能用R²和平均绝对误差（MAE）进行评估。为了确保可解释性，我们使用了特征重要性、SHAP值和决策树可视化。特别是RF和ANN等非线性模型优于MLR，RF在准确性和普适性方面取得了平衡。关键预测因子包括社会经济地位、学习时间、教师动机以及学生对数学的态度，但其影响因国家而异。散点图等可视化诊断显示RF和CATBoost与实际表现高度一致。研究结果强调了成就的非线性和情境依赖性，并突显了XAI在教育研究中的价值。本研究揭示了跨国家的模式，为促进教育公平的改革提供了信息，并支持个性化学习策略的开发。 

---
# Revisiting Rule-Based Stuttering Detection: A Comprehensive Analysis of Interpretable Models for Clinical Applications 

**Title (ZH)**: 基于规则的颤抖检测 revisit：可解释模型在临床应用中的全面分析 

**Authors**: Eric Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16681)  

**Abstract**: Stuttering affects approximately 1% of the global population, impacting communication and quality of life. While recent advances in deep learning have pushed the boundaries of automatic speech dysfluency detection, rule-based approaches remain crucial for clinical applications where interpretability and transparency are paramount. This paper presents a comprehensive analysis of rule-based stuttering detection systems, synthesizing insights from multiple corpora including UCLASS, FluencyBank, and SEP-28k. We propose an enhanced rule-based framework that incorporates speaking-rate normalization, multi-level acoustic feature analysis, and hierarchical decision structures. Our approach achieves competitive performance while maintaining complete interpretability-critical for clinical adoption. We demonstrate that rule-based systems excel particularly in prolongation detection (97-99% accuracy) and provide stable performance across varying speaking rates. Furthermore, we show how these interpretable models can be integrated with modern machine learning pipelines as proposal generators or constraint modules, bridging the gap between traditional speech pathology practices and contemporary AI systems. Our analysis reveals that while neural approaches may achieve marginally higher accuracy in unconstrained settings, rule-based methods offer unique advantages in clinical contexts where decision auditability, patient-specific tuning, and real-time feedback are essential. 

**Abstract (ZH)**: stuttering 影响全球约 1%的人口，影响沟通和生活质量。尽管深度学习近期在自动语音不畅检测方面取得了进展，但基于规则的方法仍对于强调可解释性和透明度的临床应用至关重要。本文综合分析了多种基于规则的 stuttering 检测系统，包括 UCLASS、FluencyBank 和 SEP-28k 数据集中的洞察。我们提出了一种增强的基于规则的框架，结合了发音速率规范化、多级声学特征分析和分层决策结构。该方法在保持完全可解释性的同时实现了竞争性的性能，这对于临床应用至关重要。我们证明基于规则的系统在延长音检测方面表现出色（准确率 97-99%），并在不同发音速率下表现出稳定的性能。此外，我们展示了这些可解释模型如何与现代机器学习管道集成，作为建议生成器或约束模块，从而弥合传统言语病理学实践与现代人工智能系统之间的差距。我们的分析表明，在不受约束的环境中，神经方法可能在准确率上略高，但基于规则的方法在临床环境中具有独特优势，因为这些环境中决策审计、患者特定调整和实时反馈至关重要。 

---
# SafeBimanual: Diffusion-based Trajectory Optimization for Safe Bimanual Manipulation 

**Title (ZH)**: SafeBimanual：基于扩散的轨迹优化以实现安全双臂操作 

**Authors**: Haoyuan Deng, Wenkai Guo, Qianzhun Wang, Zhenyu Wu, Ziwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18268)  

**Abstract**: Bimanual manipulation has been widely applied in household services and manufacturing, which enables the complex task completion with coordination requirements. Recent diffusion-based policy learning approaches have achieved promising performance in modeling action distributions for bimanual manipulation. However, they ignored the physical safety constraints of bimanual manipulation, which leads to the dangerous behaviors with damage to robots and objects. To this end, we propose a test-time trajectory optimization framework named SafeBimanual for any pre-trained diffusion-based bimanual manipulation policies, which imposes the safety constraints on bimanual actions to avoid dangerous robot behaviors with improved success rate. Specifically, we design diverse cost functions for safety constraints in different dual-arm cooperation patterns including avoidance of tearing objects and collision between arms and objects, which optimizes the manipulator trajectories with guided sampling of diffusion denoising process. Moreover, we employ a vision-language model (VLM) to schedule the cost functions by specifying keypoints and corresponding pairwise relationship, so that the optimal safety constraint is dynamically generated in the entire bimanual manipulation process. SafeBimanual demonstrates superiority on 8 simulated tasks in RoboTwin with a 13.7% increase in success rate and a 18.8% reduction in unsafe interactions over state-of-the-art diffusion-based methods. Extensive experiments on 4 real-world tasks further verify its practical value by improving the success rate by 32.5%. 

**Abstract (ZH)**: 双臂操作已广泛应用于家庭服务和制造业，能够通过协调完成复杂的任务。基于扩散的策略学习方法在模拟双臂操作的动作分布建模方面取得了显著的成果。然而，这些方法忽略了双臂操作的物理安全约束，导致了可能对机器人和物体造成损害的危险行为。为此，我们提出了一种名为SafeBimanual的测试时轨迹优化框架，适用于任何预训练的基于扩散的双臂操作策略，通过施加安全约束来避免危险行为并提高成功率。具体而言，我们在不同的双臂合作模式下设计了多样化的成本函数，包括避免物体损坏和臂与物体的碰撞，这些成本函数通过指导扩散去噪过程的采样来优化操作轨迹。此外，我们采用视觉语言模型（VLM）通过指定关键点及其对应关系来安排成本函数，从而在整个双臂操作过程中动态生成最优的安全约束。SafeBimanual在RoboTwin中8个仿真任务上的成功率提高了13.7%，不安全交互降低了18.8%，优于现有最佳的扩散基方法。在4个实际任务上的广泛实验进一步验证了其实际价值，成功率提高了32.5%。 

---
# ANO : Faster is Better in Noisy Landscape 

**Title (ZH)**: ANO : 在噪声环境中，更快更好 

**Authors**: Adrien Kegreisz  

**Link**: [PDF](https://arxiv.org/pdf/2508.18258)  

**Abstract**: Stochastic optimizers are central to deep learning, yet widely used methods such as Adam and Adan can degrade in non-stationary or noisy environments, partly due to their reliance on momentum-based magnitude estimates. We introduce Ano, a novel optimizer that decouples direction and magnitude: momentum is used for directional smoothing, while instantaneous gradient magnitudes determine step size. This design improves robustness to gradient noise while retaining the simplicity and efficiency of first-order methods. We further propose Anolog, which removes sensitivity to the momentum coefficient by expanding its window over time via a logarithmic schedule. We establish non-convex convergence guarantees with a convergence rate similar to other sign-based methods, and empirically show that Ano provides substantial gains in noisy and non-stationary regimes such as reinforcement learning, while remaining competitive on low-noise tasks such as standard computer vision benchmarks. 

**Abstract (ZH)**: 随机优化器是深度学习的核心，然而广泛使用的Adam和Adan等方法在非平稳或噪声环境中可能会退化，部分原因是它们依赖于基于动量的幅度估算。我们提出了一种新型优化器Ano，将方向和幅度解耦：动量用于方向平滑，而瞬时梯度幅度决定步长大小。这种设计提高了对梯度噪声的鲁棒性，同时保留了一阶方法的简洁性和效率。我们还提出了Anolog，通过使用对数时间表逐步扩展动量窗口，消除了对动量系数的敏感性。我们建立了非凸收敛保证，并且实验表明，在强化学习等噪声和非平稳环境中，Ano提供了显著的改进，同时在低噪声任务如标准计算机视觉基准测试中保持竞争力。 

---
# Type-Compliant Adaptation Cascades: Adapting Programmatic LM Workflows to Data 

**Title (ZH)**: 类型遵从适配cascade：适应程序化LM工作流的数据 

**Authors**: Chu-Cheng Lin, Daiyi Peng, Yifeng Lu, Ming Zhang, Eugene Ie  

**Link**: [PDF](https://arxiv.org/pdf/2508.18244)  

**Abstract**: Reliably composing Large Language Models (LLMs) for complex, multi-step workflows remains a significant challenge. The dominant paradigm-optimizing discrete prompts in a pipeline-is notoriously brittle and struggles to enforce the formal compliance required for structured tasks. We introduce Type-Compliant Adaptation Cascades (TACs), a framework that recasts workflow adaptation as learning typed probabilistic programs. TACs treats the entire workflow, which is composed of parameter-efficiently adapted LLMs and deterministic logic, as an unnormalized joint distribution. This enables principled, gradient-based training even with latent intermediate structures. We provide theoretical justification for our tractable optimization objective, proving that the optimization bias vanishes as the model learns type compliance. Empirically, TACs significantly outperforms state-of-the-art prompt-optimization baselines. Gains are particularly pronounced on structured tasks, improving MGSM-SymPy from $57.1\%$ to $75.9\%$ for a 27B model, MGSM from $1.6\%$ to $27.3\%$ for a 7B model. TACs offers a robust and theoretically grounded paradigm for developing reliable, task-compliant LLM systems. 

**Abstract (ZH)**: 可靠地组合大型语言模型（LLMs）以支持复杂的多步工作流仍是一项重大挑战。我们引入了类型合规适应cascade（TACs）框架，将工作流适应重新表述为学习类型概率程序的过程。TACs 将由参数高效适应的LLMs和确定性逻辑组成的整个工作流视为未标准化的联合分布，这使得即使在存在潜在中间结构的情况下，也能实现原理性的梯度训练。我们为我们的可实现优化目标提供了理论上的依据，证明了优化偏差随模型学习类型合规而消失。实验结果显示，TACs 显著优于最先进的提示优化基准模型。在结构化任务上尤其明显，27B模型的MGSM-SymPy从57.1%提升到75.9%，7B模型的MGSM从1.6%提升到27.3%。TACs 提供了一个稳健且具有理论依据的框架，用于开发可靠的任务合规LLM系统。 

---
# MTalk-Bench: Evaluating Speech-to-Speech Models in Multi-Turn Dialogues via Arena-style and Rubrics Protocols 

**Title (ZH)**: MTalk-Bench：通过擂台风格和评価指标协议评估多轮对话中的言语到言语模型 

**Authors**: Yuhao Du, Qianwei Huang, Guo Zhu, Zhanchen Dai, Sunian Chen, Qiming Zhu, Yuhao Zhang, Li Zhou, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18240)  

**Abstract**: The rapid advancement of speech-to-speech (S2S) large language models (LLMs) has significantly improved real-time spoken interaction. However, current evaluation frameworks remain inadequate for assessing performance in complex, multi-turn dialogues. To address this, we introduce MTalk-Bench, a multi-turn S2S benchmark covering three core dimensions: Semantic Information, Paralinguistic Information, and Ambient Sound. Each dimension includes nine realistic scenarios, along with targeted tasks to assess specific capabilities such as reasoning. Our dual-method evaluation framework combines Arena-style evaluation (pairwise comparison) and Rubrics-based evaluation (absolute scoring) for relative and absolute assessment. The benchmark includes both model and human outputs, evaluated by human evaluators and LLMs. Experimental results reveal two sets of findings. Overall performance of S2S LLMs: (1) models excel at semantic information processing yet underperform on paralinguistic information and ambient sounds perception; (2) models typically regain coherence by increasing response length, sacrificing efficiency in multi-turn dialogues; (3) modality-aware, task-specific designs outperform brute scaling. Evaluation framework and reliability: (1) Arena and Rubrics yield consistent, complementary rankings, but reliable distinctions emerge only when performance gaps are large; (2) LLM-as-a-judge aligns with humans when gaps are clear or criteria explicit, but exhibits position and length biases and is reliable on nonverbal evaluation only with text annotations. These results highlight current limitations in S2S evaluation and the need for more robust, speech-aware assessment frameworks. 

**Abstract (ZH)**: 面向多轮对话的speech-to-speech大语言模型多维度基准MTalk-Bench 

---
# KillChainGraph: ML Framework for Predicting and Mapping ATT&CK Techniques 

**Title (ZH)**: KillChainGraph：用于预测和映射ATT&CK技术的机器学习框架 

**Authors**: Chitraksh Singh, Monisha Dhanraj, Ken Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18230)  

**Abstract**: The escalating complexity and volume of cyberattacks demand proactive detection strategies that go beyond traditional rule-based systems. This paper presents a phase-aware, multi-model machine learning framework that emulates adversarial behavior across the seven phases of the Cyber Kill Chain using the MITRE ATT&CK Enterprise dataset. Techniques are semantically mapped to phases via ATTACK-BERT, producing seven phase-specific datasets. We evaluate LightGBM, a custom Transformer encoder, fine-tuned BERT, and a Graph Neural Network (GNN), integrating their outputs through a weighted soft voting ensemble. Inter-phase dependencies are modeled using directed graphs to capture attacker movement from reconnaissance to objectives. The ensemble consistently achieved the highest scores, with F1-scores ranging from 97.47% to 99.83%, surpassing GNN performance (97.36% to 99.81%) by 0.03%--0.20% across phases. This graph-driven, ensemble-based approach enables interpretable attack path forecasting and strengthens proactive cyber defense. 

**Abstract (ZH)**: 不断提高的网络攻击复杂性和体积要求超越传统规则系统的前瞻检测策略。本文提出了一种相位意识的多模型机器学习框架，利用MITRE ATT&CK Enterprise数据集在网络杀伤链的七个阶段中模拟对手行为。通过ATTACK-BERT将技术语义映射到各个阶段，产生了七个阶段特定的数据集。我们评估了LightGBM、一个自定义Transformer编码器、微调的BERT以及图神经网络（GNN），通过加权软投票集成整合其输出。使用有向图建模阶段间的依赖关系，以捕捉攻击者从侦察到目标的行为。集成框架在所有阶段均表现出最高性能，F1得分范围从97.47%到99.83%，相较于GNN（97.36%到99.81%）在各个阶段高出0.03%–0.20%。这种基于图的集成方法实现了可解释的攻击路径预测，并增强了主动网络防御。 

---
# Deep Learning and Matrix Completion-aided IoT Network Localization in the Outlier Scenarios 

**Title (ZH)**: 深度学习和矩阵填充辅助的物联网网络定位方法在异常场景中 

**Authors**: Sunwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.18225)  

**Abstract**: In this paper, we propose a deep learning and matrix completion aided approach for recovering an outlier contaminated Euclidean distance matrix D in IoT network localization. Unlike conventional localization techniques that search the solution over a whole set of matrices, the proposed technique restricts the search to the set of Euclidean distance matrices. Specifically, we express D as a function of the sensor coordinate matrix X that inherently satisfies the unique properties of D, and then jointly recover D and X using a deep neural network. To handle outliers effectively, we model them as a sparse matrix L and add a regularization term of L into the optimization problem. We then solve the problem by alternately updating X, D, and L. Numerical experiments demonstrate that the proposed technique can recover the location information of sensors accurately even in the presence of outliers. 

**Abstract (ZH)**: 基于深学习和矩阵完成的物联网网络定位中受污染欧几里得距离矩阵恢复方法 

---
# Why Synthetic Isn't Real Yet: A Diagnostic Framework for Contact Center Dialogue Generation 

**Title (ZH)**: 合成的还没成为现实：接触中心对话生成的诊断框架 

**Authors**: Rishikesh Devanathan, Varun Nathan, Ayush Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.18210)  

**Abstract**: Synthetic transcript generation is critical in contact center domains, where privacy and data scarcity limit model training and evaluation. Unlike prior synthetic dialogue generation work on open-domain or medical dialogues, contact center conversations are goal-oriented, role-asymmetric, and behaviorally complex, featuring disfluencies, ASR noise, and compliance-driven agent actions. In deployments where transcripts are unavailable, standard pipelines still yield derived call attributes such as Intent Summaries, Topic Flow, and QA Evaluation Forms. We leverage these as supervision signals to guide generation. To assess the quality of such outputs, we introduce a diagnostic framework of 18 linguistically and behaviorally grounded metrics for comparing real and synthetic transcripts. We benchmark four language-agnostic generation strategies, from simple prompting to characteristic-aware multi-stage approaches, alongside reference-free baselines. Results reveal persistent challenges: no method excels across all traits, with notable deficits in disfluency, sentiment, and behavioral realism. Our diagnostic tool exposes these gaps, enabling fine-grained evaluation and stress testing of synthetic dialogue across languages. 

**Abstract (ZH)**: 合成转录生成在客服中心领域至关重要，由于隐私和数据稀缺限制了模型的训练和评估。与先前针对开放领域或医疗对话的合成对话生成工作不同，客服中心的对话具有目标导向性、角色不对称性和行为复杂性，包含口吃、ASR噪音和合规驱动的代理行动。在转录不可用的部署中，标准管道仍能生成诸如意图摘要、话题流和问答评估表等衍生通话属性。我们利用这些作为监督信号来指导生成。为了评估这些输出的质量，我们引入了一种包含18个语用和行为依据的诊断框架，用于比较真实和合成的转录。我们将四种语言无关的生成策略，从简单的提示到具有特征意识的多阶段方法，与无参考基线进行了基准测试。结果揭示了持续的挑战：没有任何方法在所有特性上都表现出色，尤其是在口吃、情感和行为现实性方面存在明显缺陷。我们的诊断工具暴露了这些差距，使我们可以对跨语言的合成对话进行精细评估和压力测试。 

---
# Explain and Monitor Deep Learning Models for Computer Vision using Obz AI 

**Title (ZH)**: 使用Obz AI解释和监控计算机视觉中的深度学习模型 

**Authors**: Neo Christopher Chung, Jakub Binda  

**Link**: [PDF](https://arxiv.org/pdf/2508.18188)  

**Abstract**: Deep learning has transformed computer vision (CV), achieving outstanding performance in classification, segmentation, and related tasks. Such AI-based CV systems are becoming prevalent, with applications spanning from medical imaging to surveillance. State of the art models such as convolutional neural networks (CNNs) and vision transformers (ViTs) are often regarded as ``black boxes,'' offering limited transparency into their decision-making processes. Despite a recent advancement in explainable AI (XAI), explainability remains underutilized in practical CV deployments. A primary obstacle is the absence of integrated software solutions that connect XAI techniques with robust knowledge management and monitoring frameworks. To close this gap, we have developed Obz AI, a comprehensive software ecosystem designed to facilitate state-of-the-art explainability and observability for vision AI systems. Obz AI provides a seamless integration pipeline, from a Python client library to a full-stack analytics dashboard. With Obz AI, a machine learning engineer can easily incorporate advanced XAI methodologies, extract and analyze features for outlier detection, and continuously monitor AI models in real time. By making the decision-making mechanisms of deep models interpretable, Obz AI promotes observability and responsible deployment of computer vision systems. 

**Abstract (ZH)**: 深度学习已 transforming 计算机视觉（CV），在分类、分割及相关任务中取得了卓越的性能。这类基于AI的CV系统日益普及，应用范围从医学成像到监控。最先进模型如卷积神经网络（CNNs）和视觉变换器（ViTs）通常被视为“黑盒”，提供有限的决策过程透明度。尽管可解释AI（XAI）领域取得了近期进展，但在实际CV部署中可解释性仍然未得到充分利用。主要障碍是缺乏将XAI技术与 robust 知识管理及监测框架集成的软件解决方案。为弥补这一差距，我们开发了Obz AI，一个全面的软件生态系统，旨在促进最先进的可解释性和可观测性以支持视觉AI系统。Obz AI 提供了从Python客户端库到全栈分析仪表盘的无缝集成管道。使用Obz AI，机器学习工程师可以轻松地整合高级XAI方法、提取和分析特征以进行异常检测，并在实时持续监控AI模型。通过使深度模型的决策机制可解释，Obz AI 推动了计算机视觉系统的可观测性和负责任部署。 

---
# BRAIN: Bias-Mitigation Continual Learning Approach to Vision-Brain Understanding 

**Title (ZH)**: BRAIN: 偏见缓解持续学习方法及其在视觉脑认知理解中的应用 

**Authors**: Xuan-Bac Nguyen, Thanh-Dat Truong, Pawan Sinha, Khoa Luu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18187)  

**Abstract**: Memory decay makes it harder for the human brain to recognize visual objects and retain details. Consequently, recorded brain signals become weaker, uncertain, and contain poor visual context over time. This paper presents one of the first vision-learning approaches to address this problem. First, we statistically and experimentally demonstrate the existence of inconsistency in brain signals and its impact on the Vision-Brain Understanding (VBU) model. Our findings show that brain signal representations shift over recording sessions, leading to compounding bias, which poses challenges for model learning and degrades performance. Then, we propose a new Bias-Mitigation Continual Learning (BRAIN) approach to address these limitations. In this approach, the model is trained in a continual learning setup and mitigates the growing bias from each learning step. A new loss function named De-bias Contrastive Learning is also introduced to address the bias problem. In addition, to prevent catastrophic forgetting, where the model loses knowledge from previous sessions, the new Angular-based Forgetting Mitigation approach is introduced to preserve learned knowledge in the model. Finally, the empirical experiments demonstrate that our approach achieves State-of-the-Art (SOTA) performance across various benchmarks, surpassing prior and non-continual learning methods. 

**Abstract (ZH)**: 记忆衰退使人脑更难识别视觉物体并保留细节。这导致记录的大脑信号变得较弱、不确定且包含较差的视觉上下文。本文提出了第一个解决这一问题的视觉学习方法之一。首先，我们通过统计和实验展示了大脑信号中的一致性问题及其对视觉-大脑理解(VBU)模型的影响。我们的研究发现表明，大脑信号表示在录制会话中发生变化，导致累积偏差，这对模型学习构成挑战并降低性能。然后，我们提出了一种新的偏差缓解持续学习(BRAIN)方法来解决这些问题。在此方法中，模型在持续学习设置中进行训练，并从每个学习步骤中缓解增长的偏差。我们还引入了一种新的损失函数——去偏置对比学习，以解决偏差问题。此外，为了防止灾难性遗忘，即模型忘记先前会话的知识，我们提出了基于角度的遗忘缓解新方法，以保存模型中的学习知识。最后，实验证明，我们的方法在各种基准测试中达到了最佳性能，优于先前的方法和非持续学习方法。 

---
# Leveraging Large Language Models for Accurate Sign Language Translation in Low-Resource Scenarios 

**Title (ZH)**: 利用大规模语言模型在低资源场景中实现准确的手语翻译 

**Authors**: Luana Bulla, Gabriele Tuccio, Misael Mongiovì, Aldo Gangemi  

**Link**: [PDF](https://arxiv.org/pdf/2508.18183)  

**Abstract**: Translating natural languages into sign languages is a highly complex and underexplored task. Despite growing interest in accessibility and inclusivity, the development of robust translation systems remains hindered by the limited availability of parallel corpora which align natural language with sign language data. Existing methods often struggle to generalize in these data-scarce environments, as the few datasets available are typically domain-specific, lack standardization, or fail to capture the full linguistic richness of sign languages. To address this limitation, we propose Advanced Use of LLMs for Sign Language Translation (AulSign), a novel method that leverages Large Language Models via dynamic prompting and in-context learning with sample selection and subsequent sign association. Despite their impressive abilities in processing text, LLMs lack intrinsic knowledge of sign languages; therefore, they are unable to natively perform this kind of translation. To overcome this limitation, we associate the signs with compact descriptions in natural language and instruct the model to use them. We evaluate our method on both English and Italian languages using SignBank+, a recognized benchmark in the field, as well as the Italian LaCAM CNR-ISTC dataset. We demonstrate superior performance compared to state-of-the-art models in low-data scenario. Our findings demonstrate the effectiveness of AulSign, with the potential to enhance accessibility and inclusivity in communication technologies for underrepresented linguistic communities. 

**Abstract (ZH)**: 自然语言到手语的翻译是一项高度复杂且尚未充分探索的任务。尽管无障碍和平等性日益受到关注，但由于自然语言与手语数据对齐的平行语料库有限，开发健壮的翻译系统仍受阻碍。现有方法往往难以在这些数据稀缺的环境中泛化，因为可用的数据集通常是领域特定的、缺乏标准化或未能捕捉手语语言的全部丰富性。为了解决这一局限，我们提出了基于大语言模型的手语翻译高级应用（AulSign）方法，该方法通过动态提示和上下文学习结合样本选择和后续手语关联来利用大语言模型。尽管大语言模型在处理文本方面表现出色，但它们缺乏手语固有的知识；因此，它们无法本外地执行这种翻译。为克服这一限制，我们将手语与紧凑的手语自然语言描述关联，并指示模型使用这些描述。我们在SignBank+和意大利LaCAM CNR-ISTC数据集上分别使用英语和意大利语评估了该方法，SignBank+是该领域公认的基准。我们的研究表明，在数据稀少的情况下，AulSign方法相较于最新模型具有更好的性能。我们的研究结果表明，AulSign的有效性，其有可能提升未充分代表语言社区在交流技术中的无障碍和平等性。 

---
# AdLoCo: adaptive batching significantly improves communications efficiency and convergence for Large Language Models 

**Title (ZH)**: AdLoCo:自适应批量处理显著提高大型语言模型的通信效率和收敛性 

**Authors**: Nikolay Kutuzov, Makar Baderko, Stepan Kulibaba, Artem Dzhalilov, Daniel Bobrov, Maxim Mashtaler, Alexander Gasnikov  

**Link**: [PDF](https://arxiv.org/pdf/2508.18182)  

**Abstract**: Scaling distributed training of Large Language Models (LLMs) requires not only algorithmic advances but also efficient utilization of heterogeneous hardware resources. While existing methods such as DiLoCo have demonstrated promising results, they often fail to fully exploit computational clusters under dynamic workloads. To address this limitation, we propose a three-stage method that combines Multi-Instance Training (MIT), Adaptive Batched DiLoCo, and switch mode mechanism. MIT allows individual nodes to run multiple lightweight training streams with different model instances in parallel and merge them to combine knowledge, increasing throughput and reducing idle time. Adaptive Batched DiLoCo dynamically adjusts local batch sizes to balance computation and communication, substantially lowering synchronization delays. Switch mode further stabilizes training by seamlessly introducing gradient accumulation once adaptive batch sizes grow beyond hardware-friendly limits. Together, these innovations improve both convergence speed and system efficiency. We also provide a theoretical estimate of the number of communications required for the full convergence of a model trained using our method. 

**Abstract (ZH)**: 分布式训练大规模语言模型（LLMs）不仅需要算法上的进步，还需要有效地利用异构硬件资源。为了解决现有方法如DiLoCo在动态工作负载下未能充分利用计算集群的问题，我们提出了一种三阶段方法，结合多实例训练（MIT）、自适应批量DiLoCo及其切换模式机制。MIT允许多个节点并行运行不同模型实例的多个轻量级训练流，并合并这些流以增加吞吐量并减少空闲时间。自适应批量DiLoCo动态调整本地批次大小以平衡计算和通信，显著降低同步延迟。切换模式进一步通过在自适应批次大小超过硬件友好限制时无缝引入梯度累积来稳定训练。这些创新共同提高了收敛速度和系统效率。我们还提供了对使用我们方法训练的模型达到完全收敛所需通信次数的理论估计。 

---
# Amortized Sampling with Transferable Normalizing Flows 

**Title (ZH)**: 转移可变形正则化流的渐进采样 

**Authors**: Charlie B. Tan, Majdi Hassan, Leon Klein, Saifuddin Syed, Dominique Beaini, Michael M. Bronstein, Alexander Tong, Kirill Neklyudov  

**Link**: [PDF](https://arxiv.org/pdf/2508.18175)  

**Abstract**: Efficient equilibrium sampling of molecular conformations remains a core challenge in computational chemistry and statistical inference. Classical approaches such as molecular dynamics or Markov chain Monte Carlo inherently lack amortization; the computational cost of sampling must be paid in-full for each system of interest. The widespread success of generative models has inspired interest into overcoming this limitation through learning sampling algorithms. Despite performing on par with conventional methods when trained on a single system, learned samplers have so far demonstrated limited ability to transfer across systems. We prove that deep learning enables the design of scalable and transferable samplers by introducing Prose, a 280 million parameter all-atom transferable normalizing flow trained on a corpus of peptide molecular dynamics trajectories up to 8 residues in length. Prose draws zero-shot uncorrelated proposal samples for arbitrary peptide systems, achieving the previously intractable transferability across sequence length, whilst retaining the efficient likelihood evaluation of normalizing flows. Through extensive empirical evaluation we demonstrate the efficacy of Prose as a proposal for a variety of sampling algorithms, finding a simple importance sampling-based finetuning procedure to achieve superior performance to established methods such as sequential Monte Carlo on unseen tetrapeptides. We open-source the Prose codebase, model weights, and training dataset, to further stimulate research into amortized sampling methods and finetuning objectives. 

**Abstract (ZH)**: 高效的分子构象均衡采样仍然是计算化学和统计推断中的核心挑战。通过Prose实现大规模且可转移的采样算法ucid 

---
# The Computational Complexity of Satisfiability in State Space Models 

**Title (ZH)**: 状态空间模型中的 satisfiability 计算复杂性 

**Authors**: Eric Alsmann, Martin Lange  

**Link**: [PDF](https://arxiv.org/pdf/2508.18162)  

**Abstract**: We analyse the complexity of the satisfiability problem ssmSAT for State Space Models (SSM), which asks whether an input sequence can lead the model to an accepting configuration. We find that ssmSAT is undecidable in general, reflecting the computational power of SSM. Motivated by practical settings, we identify two natural restrictions under which ssmSAT becomes decidable and establish corresponding complexity bounds. First, for SSM with bounded context length, ssmSAT is NP-complete when the input length is given in unary and in NEXPTIME (and PSPACE-hard) when the input length is given in binary. Second, for quantised SSM operating over fixed-width arithmetic, ssmSAT is PSPACE-complete resp. in EXPSPACE depending on the bit-width encoding. While these results hold for diagonal gated SSM we also establish complexity bounds for time-invariant SSM. Our results establish a first complexity landscape for formal reasoning in SSM and highlight fundamental limits and opportunities for the verification of SSM-based language models. 

**Abstract (ZH)**: 我们分析了状态空间模型（SSM）的满足性问题ssmSAT的复杂性，该问题询问是否有一个输入序列可以使模型达到接受配置。我们发现ssmSAT通常是不可判定的，反映了SSM的计算能力。受到实际应用的启发，我们识别了两种自然的限制条件，在这两种条件下ssmSAT变为可判定，并建立了相应的时间复杂度界限。首先，对于具有有界上下文长度的SSM，当输入长度以 unary 形式给出时，ssmSAT 是 NP 完全问题；当输入长度以二进制形式给出时，ssmSAT 是 NEXPTIME（和 PSPACE-硬）问题。其次，对于操作固定宽度算术的量化 SSM，ssmSAT 是 PSPACE 完全问题，具体取决于位宽编码，可能在 EXPSPACE 中运行。尽管这些结果适用于对角门控 SSM，我们还为时不变 SSM 建立了复杂性界限。我们的结果为 SSM 中的形式化推理提供了首个复杂性景观，并突显了 SSM 基础语言模型验证的基本限制和机会。 

---
# Assessing the Noise Robustness of Class Activation Maps: A Framework for Reliable Model Interpretability 

**Title (ZH)**: 评估类激活图的噪声鲁棒性：一种可靠的模型可解释性框架 

**Authors**: Syamantak Sarkar, Revoti P. Bora, Bhupender Kaushal, Sudhish N George, Kiran Raja  

**Link**: [PDF](https://arxiv.org/pdf/2508.18154)  

**Abstract**: Class Activation Maps (CAMs) are one of the important methods for visualizing regions used by deep learning models. Yet their robustness to different noise remains underexplored. In this work, we evaluate and report the resilience of various CAM methods for different noise perturbations across multiple architectures and datasets. By analyzing the influence of different noise types on CAM explanations, we assess the susceptibility to noise and the extent to which dataset characteristics may impact explanation stability. The findings highlight considerable variability in noise sensitivity for various CAMs. We propose a robustness metric for CAMs that captures two key properties: consistency and responsiveness. Consistency reflects the ability of CAMs to remain stable under input perturbations that do not alter the predicted class, while responsiveness measures the sensitivity of CAMs to changes in the prediction caused by such perturbations. The metric is evaluated empirically across models, different perturbations, and datasets along with complementary statistical tests to exemplify the applicability of our proposed approach. 

**Abstract (ZH)**: Class Activation Maps (CAMs)在不同噪声干扰下的鲁棒性研究：多架构与多数据集上的评估与报告 

---
# Learning from Few Samples: A Novel Approach for High-Quality Malcode Generation 

**Title (ZH)**: 从少量样本学习：一种高质量恶意代码生成的新方法 

**Authors**: Haijian Ma, Daizong Liu, Xiaowen Cai, Pan Zhou, Yulai Xie  

**Link**: [PDF](https://arxiv.org/pdf/2508.18148)  

**Abstract**: Intrusion Detection Systems (IDS) play a crucial role in network security defense. However, a significant challenge for IDS in training detection models is the shortage of adequately labeled malicious samples. To address these issues, this paper introduces a novel semi-supervised framework \textbf{GANGRL-LLM}, which integrates Generative Adversarial Networks (GANs) with Large Language Models (LLMs) to enhance malicious code generation and SQL Injection (SQLi) detection capabilities in few-sample learning scenarios. Specifically, our framework adopts a collaborative training paradigm where: (1) the GAN-based discriminator improves malicious pattern recognition through adversarial learning with generated samples and limited real samples; and (2) the LLM-based generator refines the quality of malicious code synthesis using reward signals from the discriminator. The experimental results demonstrate that even with a limited number of labeled samples, our training framework is highly effective in enhancing both malicious code generation and detection capabilities. This dual enhancement capability offers a promising solution for developing adaptive defense systems capable of countering evolving cyber threats. 

**Abstract (ZH)**: 入侵检测系统（IDS）在网络安全性防卫中发挥着关键作用。然而，IDS在训练检测模型时面临的显著挑战之一是缺乏足够的标记恶意样本。为解决这些问题，本文提出了一种新颖的半监督框架**GANGRL-LLM**，该框架结合生成对抗网络（GANs）与大型语言模型（LLMs），以增强在少量样本学习场景中的恶意代码生成和SQL注入（SQLi）检测能力。具体而言，我们的框架采用协作训练范式：（1）基于GAN的鉴别器通过与生成样本和有限的真实样本进行对抗学习来改进恶意模式识别；（2）基于LLM的生成器使用来自鉴别器的奖励信号来精炼恶意代码合成的质量。实验结果表明，即使样本数量有限，我们的训练框架在提高恶意代码生成和检测能力方面非常有效。这种双重增强能力为开发能够应对不断演变的网络威胁的自适应防御系统提供了有前景的解决方案。 

---
# Test-Time Scaling Strategies for Generative Retrieval in Multimodal Conversational Recommendations 

**Title (ZH)**: 基于多模态对话推荐的生成检索测试时扩增策略 

**Authors**: Hung-Chun Hsu, Yuan-Ching Kuo, Chao-Han Huck Yang, Szu-Wei Fu, Hanrong Ye, Hongxu Yin, Yu-Chiang Frank Wang, Ming-Feng Tsai, Chuan-Ju Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18132)  

**Abstract**: The rapid evolution of e-commerce has exposed the limitations of traditional product retrieval systems in managing complex, multi-turn user interactions. Recent advances in multimodal generative retrieval -- particularly those leveraging multimodal large language models (MLLMs) as retrievers -- have shown promise. However, most existing methods are tailored to single-turn scenarios and struggle to model the evolving intent and iterative nature of multi-turn dialogues when applied naively. Concurrently, test-time scaling has emerged as a powerful paradigm for improving large language model (LLM) performance through iterative inference-time refinement. Yet, its effectiveness typically relies on two conditions: (1) a well-defined problem space (e.g., mathematical reasoning), and (2) the model's ability to self-correct -- conditions that are rarely met in conversational product search. In this setting, user queries are often ambiguous and evolving, and MLLMs alone have difficulty grounding responses in a fixed product corpus. Motivated by these challenges, we propose a novel framework that introduces test-time scaling into conversational multimodal product retrieval. Our approach builds on a generative retriever, further augmented with a test-time reranking (TTR) mechanism that improves retrieval accuracy and better aligns results with evolving user intent throughout the dialogue. Experiments across multiple benchmarks show consistent improvements, with average gains of 14.5 points in MRR and 10.6 points in nDCG@1. 

**Abstract (ZH)**: 电商的快速演变揭示了传统产品检索系统在管理复杂多轮用户交互方面的局限性。近年来，多模态生成检索技术，尤其是利用多模态大型语言模型（MLLMs）作为检索器的技术，展现了潜力。然而，现有方法大多针对单轮场景，未经调整地应用于多轮对话时难以建模用户意图的演变和迭代性。同时，测试时缩放已成为通过迭代推理时精炼提高大型语言模型（LLM）性能的强大范式。然而，其有效性通常依赖于两个条件：(1) 井定义的问题空间（例如，数学推理），和(2) 模型自我纠正的能力——这两个条件在会话产品搜索中很少满足。在这种情境下，用户查询往往是模糊且不断变化的，仅靠MLLMs难以将回复与固定的产品库对齐。受这些挑战的启发，我们提出了一种新的框架，将测试时缩放引入会话多模态产品检索。该方法基于一个生成式检索器，并进一步增强了一个测试时重排（TTR）机制，以提高检索准确性，并在整个对话过程中更好地使结果与用户意图保持一致。多项基准试验显示了一致的改进，平均MRR提高了14.5点，nDCG@1提高了10.6点。 

---
# CMPhysBench: A Benchmark for Evaluating Large Language Models in Condensed Matter Physics 

**Title (ZH)**: CMPhysBench: 一种评估凝聚态物理学大型语言模型性能的标准测试工具 

**Authors**: Weida Wang, Dongchen Huang, Jiatong Li, Tengchao Yang, Ziyang Zheng, Di Zhang, Dong Han, Benteng Chen, Binzhao Luo, Zhiyu Liu, Kunling Liu, Zhiyuan Gao, Shiqi Geng, Wei Ma, Jiaming Su, Xin Li, Shuchen Pu, Yuhan Shui, Qianjia Cheng, Zhihao Dou, Dongfei Cui, Changyong He, Jin Zeng, Zeke Xie, Mao Su, Dongzhan Zhou, Yuqiang Li, Wanli Ouyang, Lei Bai, Yunqi Cai, Xi Dai, Shufei Zhang, Jinguang Cheng, Zhong Fang, Hongming Weng  

**Link**: [PDF](https://arxiv.org/pdf/2508.18124)  

**Abstract**: We introduce CMPhysBench, designed to assess the proficiency of Large Language Models (LLMs) in Condensed Matter Physics, as a novel Benchmark. CMPhysBench is composed of more than 520 graduate-level meticulously curated questions covering both representative subfields and foundational theoretical frameworks of condensed matter physics, such as magnetism, superconductivity, strongly correlated systems, etc. To ensure a deep understanding of the problem-solving process,we focus exclusively on calculation problems, requiring LLMs to independently generate comprehensive solutions. Meanwhile, leveraging tree-based representations of expressions, we introduce the Scalable Expression Edit Distance (SEED) score, which provides fine-grained (non-binary) partial credit and yields a more accurate assessment of similarity between prediction and ground-truth. Our results show that even the best models, Grok-4, reach only 36 average SEED score and 28% accuracy on CMPhysBench, underscoring a significant capability gap, especially for this practical and frontier domain relative to traditional physics. The code anddataset are publicly available at this https URL. 

**Abstract (ZH)**: CMPhysBench: 一种评估大型语言模型在凝聚态物理学 proficiency 的新型基准 

---
# A.S.E: A Repository-Level Benchmark for Evaluating Security in AI-Generated Code 

**Title (ZH)**: A.S.E: 一个用于评估AI生成代码安全性的仓库级基准 

**Authors**: Keke Lian, Bin Wang, Lei Zhang, Libo Chen, Junjie Wang, Ziming Zhao, Yujiu Yang, Haotong Duan, Haoran Zhao, Shuang Liao, Mingda Guo, Jiazheng Quan, Yilu Zhong, Chenhao He, Zichuan Chen, Jie Wu, Haoling Li, Zhaoxuan Li, Jiongchi Yu, Hui Li, Dong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18106)  

**Abstract**: The increasing adoption of large language models (LLMs) in software engineering necessitates rigorous security evaluation of their generated code. However, existing benchmarks are inadequate, as they focus on isolated code snippets, employ unstable evaluation methods that lack reproducibility, and fail to connect the quality of input context with the security of the output. To address these gaps, we introduce A.S.E (AI Code Generation Security Evaluation), a benchmark for repository-level secure code generation. A.S.E constructs tasks from real-world repositories with documented CVEs, preserving full repository context like build systems and cross-file dependencies. Its reproducible, containerized evaluation framework uses expert-defined rules to provide stable, auditable assessments of security, build quality, and generation stability. Our evaluation of leading LLMs on A.S.E reveals three key findings: (1) Claude-3.7-Sonnet achieves the best overall performance. (2) The security gap between proprietary and open-source models is narrow; Qwen3-235B-A22B-Instruct attains the top security score. (3) Concise, ``fast-thinking'' decoding strategies consistently outperform complex, ``slow-thinking'' reasoning for security patching. 

**Abstract (ZH)**: 大型语言模型在软件工程中的日益广泛应用 necessitates 严格的生成代码安全性评估。然而，现有基准不充分，因为它们专注于孤立的代码片段、采用缺乏重现性的不稳定评估方法，并且无法将输入上下文的质量与输出的安全性联系起来。为解决这些差距，我们介绍了A.S.E（AI代码生成安全性评估）基准，用于存储库级别的安全代码生成。A.S.E 从包含已记录的CVE的真实存储库中构建任务，保留完整的存储库上下文，如构建系统和跨文件依赖关系。其可再现的容器化评估框架使用专家定义的规则提供稳定、可审计的安全性、构建质量和生成稳定性评估。我们在A.S.E上对领先的大规模语言模型进行评估揭示了三个关键发现：(1) Claude-3.7-Sonnet表现出最佳的整体性能。(2) 商用软件和开源模型之间的安全性差距较小；Qwen3-235B-A22B-Instruct达到了最高的安全性评分。(3) 简洁的“快速思考”解码策略始终优于复杂的“缓慢思考”推理策略，特别是在安全修补方面。 

---
# Named Entity Recognition of Historical Text via Large Language Model 

**Title (ZH)**: 基于大型语言模型的古文命名实体识别 

**Authors**: Shibingfeng Zhang, Giovanni Colavizza  

**Link**: [PDF](https://arxiv.org/pdf/2508.18090)  

**Abstract**: Large language models have demonstrated remarkable versatility across a wide range of natural language processing tasks and domains. One such task is Named Entity Recognition (NER), which involves identifying and classifying proper names in text, such as people, organizations, locations, dates, and other specific entities. NER plays a crucial role in extracting information from unstructured textual data, enabling downstream applications such as information retrieval from unstructured text.
Traditionally, NER is addressed using supervised machine learning approaches, which require large amounts of annotated training data. However, historical texts present a unique challenge, as the annotated datasets are often scarce or nonexistent, due to the high cost and expertise required for manual labeling. In addition, the variability and noise inherent in historical language, such as inconsistent spelling and archaic vocabulary, further complicate the development of reliable NER systems for these sources.
In this study, we explore the feasibility of applying LLMs to NER in historical documents using zero-shot and few-shot prompting strategies, which require little to no task-specific training data. Our experiments, conducted on the HIPE-2022 (Identifying Historical People, Places and other Entities) dataset, show that LLMs can achieve reasonably strong performance on NER tasks in this setting. While their performance falls short of fully supervised models trained on domain-specific annotations, the results are nevertheless promising. These findings suggest that LLMs offer a viable and efficient alternative for information extraction in low-resource or historically significant corpora, where traditional supervised methods are infeasible. 

**Abstract (ZH)**: 大型语言模型在多种自然语言处理任务和领域中展现了显著的灵活性。其中一个任务是命名实体识别（NER），它涉及在文本中识别和分类专有名词，如人名、组织、地点、日期及其他具体实体。NER在从非结构化文本数据中提取信息方面发挥着关键作用，使下游应用如非结构化文本的信息检索成为可能。
传统上，NER使用监督机器学习方法处理，这需要大量的标注训练数据。然而，历史文本提出了一项独特挑战，因为标注数据集往往稀缺或不存在，这主要是由于人工标注所需的成本和专业知识。此外，历史语言中固有的变异性与噪音，如不一致的拼写和过时的词汇，也进一步增加了为这些来源开发可靠的NER系统的难度。
在本研究中，我们探讨了使用零样本和少量样本提示策略将大型语言模型应用于历史文件中的NER的可行性。我们的实验在HIPE-2022（识别历史人物、地点和其他实体）数据集上进行，结果显示大型语言模型在这种环境中执行NER任务时能够达到相当强的效果。尽管其性能不如在领域特定标注上进行完全监督训练的模型，但结果仍然令人鼓舞。这些发现表明，大型语言模型为在资源匮乏或具有历史意义的语料库中提取信息提供了一种可行且高效的替代方法，而传统的方法在这种情况下不可行。 

---
# Arnold: a generalist muscle transformer policy 

**Title (ZH)**: Arnold: 通用肌肉变换器策略 

**Authors**: Alberto Silvio Chiappa, Boshi An, Merkourios Simos, Chengkun Li, Alexander Mathis  

**Link**: [PDF](https://arxiv.org/pdf/2508.18066)  

**Abstract**: Controlling high-dimensional and nonlinear musculoskeletal models of the human body is a foundational scientific challenge. Recent machine learning breakthroughs have heralded policies that master individual skills like reaching, object manipulation and locomotion in musculoskeletal systems with many degrees of freedom. However, these agents are merely "specialists", achieving high performance for a single skill. In this work, we develop Arnold, a generalist policy that masters multiple tasks and embodiments. Arnold combines behavior cloning and fine-tuning with PPO to achieve expert or super-expert performance in 14 challenging control tasks from dexterous object manipulation to locomotion. A key innovation is Arnold's sensorimotor vocabulary, a compositional representation of the semantics of heterogeneous sensory modalities, objectives, and actuators. Arnold leverages this vocabulary via a transformer architecture to deal with the variable observation and action spaces of each task. This framework supports efficient multi-task, multi-embodiment learning and facilitates rapid adaptation to novel tasks. Finally, we analyze Arnold to provide insights into biological motor control, corroborating recent findings on the limited transferability of muscle synergies across tasks. 

**Abstract (ZH)**: 高维度和非线性人体运动学模型的控制是基础科学挑战。近期机器学习突破已经使得能够在多自由度的肌体系统中掌握诸如抓取、物体操作和运动等单个技能。然而，这些智能体仅仅是“专家”，在单一技能上达到高绩效。在这项工作中，我们开发了Arnold，一种能够掌握多种任务和体态的一般主义智能体。Arnold 将行为克隆、微调与PPO结合，以在包括灵巧物体操作到运动在内的14项具有挑战性的控制任务中达到专家或超专家级的性能。一个关键创新是Arnold的感官运动词汇表，这是一种组合式表示异质感觉模态、目标和执行器语义的方法。Arnold 利用这种词汇表通过变压器架构来处理每项任务中多变的观测和动作空间。该框架支持高效的多任务、多体态学习，并促进对新任务的快速适应。最后，我们对Arnold进行分析，以提供对生物运动控制的洞察，证实了最近关于肌肉协同在不同任务间有限转移性发现的合理性。 

---
# Dynamic Fusion Multimodal Network for SpeechWellness Detection 

**Title (ZH)**: 动态融合多模态网络用于语音健康检测 

**Authors**: Wenqiang Sun, Han Yin, Jisheng Bai, Jianfeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.18057)  

**Abstract**: Suicide is one of the leading causes of death among adolescents. Previous suicide risk prediction studies have primarily focused on either textual or acoustic information in isolation, the integration of multimodal signals, such as speech and text, offers a more comprehensive understanding of an individual's mental state. Motivated by this, and in the context of the 1st SpeechWellness detection challenge, we explore a lightweight multi-branch multimodal system based on a dynamic fusion mechanism for speechwellness detection. To address the limitation of prior approaches that rely on time-domain waveforms for acoustic analysis, our system incorporates both time-domain and time-frequency (TF) domain acoustic features, as well as semantic representations. In addition, we introduce a dynamic fusion block to adaptively integrate information from different modalities. Specifically, it applies learnable weights to each modality during the fusion process, enabling the model to adjust the contribution of each modality. To enhance computational efficiency, we design a lightweight structure by simplifying the original baseline model. Experimental results demonstrate that the proposed system exhibits superior performance compared to the challenge baseline, achieving a 78% reduction in model parameters and a 5% improvement in accuracy. 

**Abstract (ZH)**: 青少年自杀是导致死亡的主要原因之一。以往的自杀风险预测研究主要集中在文本或声学信息单一模态的数据上，将语音和文本等多模态信号的融合提供了一种更全面理解个体心理状态的方法。受到这一启发，并在第1届SpeechWellness检测挑战赛的背景下，我们探索了一种基于动态融合机制的轻量级多分支多模态系统，用于SpeechWellness检测。为了克服之前依赖时域波形进行声学分析的局限性，我们的系统结合了时域和时频域声学特征以及语义表示。此外，我们引入了一个动态融合块，以适应性地整合不同模态的信息。具体而言，该块在融合过程中为每个模态应用可学习的权重，使模型能够调整每个模态的贡献。为了提升计算效率，我们通过简化原始基线模型设计了一个轻量级结构。实验结果表明，所提出系统的表现优于挑战基线，模型参数减少了78%，准确率提高了5%。 

---
# HyST: LLM-Powered Hybrid Retrieval over Semi-Structured Tabular Data 

**Title (ZH)**: HyST：半结构化表格数据的LLM驱动混合检索 

**Authors**: Jiyoon Myung, Jihyeon Park, Joohyung Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.18048)  

**Abstract**: User queries in real-world recommendation systems often combine structured constraints (e.g., category, attributes) with unstructured preferences (e.g., product descriptions or reviews). We introduce HyST (Hybrid retrieval over Semi-structured Tabular data), a hybrid retrieval framework that combines LLM-powered structured filtering with semantic embedding search to support complex information needs over semi-structured tabular data. HyST extracts attribute-level constraints from natural language using large language models (LLMs) and applies them as metadata filters, while processing the remaining unstructured query components via embedding-based retrieval. Experiments on a semi-structured benchmark show that HyST consistently outperforms tradtional baselines, highlighting the importance of structured filtering in improving retrieval precision, offering a scalable and accurate solution for real-world user queries. 

**Abstract (ZH)**: 现实世界推荐系统中的用户查询通常结合了结构化约束（如类别、属性）与非结构化偏好（如产品描述或评论）。我们提出了HyST（半结构化表数据混合检索），这是一种将大语言模型驱动的结构化过滤与语义嵌入搜索结合的混合检索框架，用于支持半结构化表数据上的复杂信息需求。HyST 使用大语言模型从自然语言中提取属性级约束，并将其作为元数据过滤器应用，同时通过嵌入式检索处理其余非结构化查询组件。在半结构化基准测试上的实验显示，HyST 一贯优于传统Baseline，突显了结构化过滤在提高检索精度方面的重要性，为现实世界用户查询提供了可扩展且准确的解决方案。 

---
# AQ-PCDSys: An Adaptive Quantized Planetary Crater Detection System for Autonomous Space Exploration 

**Title (ZH)**: AQ-PCDSys: 一种自适应量化 planetary 衡星环形坑检测系统用于自主太空探索 

**Authors**: Aditri Paul, Archan Paul  

**Link**: [PDF](https://arxiv.org/pdf/2508.18025)  

**Abstract**: Autonomous planetary exploration missions are critically dependent on real-time, accurate environmental perception for navigation and hazard avoidance. However, deploying deep learning models on the resource-constrained computational hardware of planetary exploration platforms remains a significant challenge. This paper introduces the Adaptive Quantized Planetary Crater Detection System (AQ-PCDSys), a novel framework specifically engineered for real-time, onboard deployment in the computationally constrained environments of space exploration missions. AQ-PCDSys synergistically integrates a Quantized Neural Network (QNN) architecture, trained using Quantization-Aware Training (QAT), with an Adaptive Multi-Sensor Fusion (AMF) module. The QNN architecture significantly optimizes model size and inference latency suitable for real-time onboard deployment in space exploration missions, while preserving high accuracy. The AMF module intelligently fuses data from Optical Imagery (OI) and Digital Elevation Models (DEMs) at the feature level, utilizing an Adaptive Weighting Mechanism (AWM) to dynamically prioritize the most relevant and reliable sensor modality based on planetary ambient conditions. This approach enhances detection robustness across diverse planetary landscapes. Paired with Multi-Scale Detection Heads specifically designed for robust and efficient detection of craters across a wide range of sizes, AQ-PCDSys provides a computationally efficient, reliable and accurate solution for planetary crater detection, a critical capability for enabling the next generation of autonomous planetary landing, navigation, and scientific exploration. 

**Abstract (ZH)**: 自适应量化行星撞击坑检测系统（AQ-PCDSys）：一种适用于太空探索任务计算约束环境的实时机载部署框架 

---
# Towards Continual Visual Anomaly Detection in the Medical Domain 

**Title (ZH)**: 医学领域持续视觉异常检测的研究 

**Authors**: Manuel Barusco, Francesco Borsatti, Nicola Beda, Davide Dalle Pezze, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2508.18013)  

**Abstract**: Visual Anomaly Detection (VAD) seeks to identify abnormal images and precisely localize the corresponding anomalous regions, relying solely on normal data during training. This approach has proven essential in domains such as manufacturing and, more recently, in the medical field, where accurate and explainable detection is critical. Despite its importance, the impact of evolving input data distributions over time has received limited attention, even though such changes can significantly degrade model performance. In particular, given the dynamic and evolving nature of medical imaging data, Continual Learning (CL) provides a natural and effective framework to incrementally adapt models while preserving previously acquired knowledge. This study explores for the first time the application of VAD models in a CL scenario for the medical field. In this work, we utilize a CL version of the well-established PatchCore model, called PatchCoreCL, and evaluate its performance using BMAD, a real-world medical imaging dataset with both image-level and pixel-level annotations. Our results demonstrate that PatchCoreCL is an effective solution, achieving performance comparable to the task-specific models, with a forgetting value less than a 1%, highlighting the feasibility and potential of CL for adaptive VAD in medical imaging. 

**Abstract (ZH)**: 视觉异常检测在持续学习框架下的医疗领域应用研究 

---
# Previously on... Automating Code Review 

**Title (ZH)**: 此前研究... 自动化代码审查 

**Authors**: Robert Heumüller, Frank Ortmeier  

**Link**: [PDF](https://arxiv.org/pdf/2508.18003)  

**Abstract**: Modern Code Review (MCR) is a standard practice in software engineering, yet it demands substantial time and resource investments. Recent research has increasingly explored automating core review tasks using machine learning (ML) and deep learning (DL). As a result, there is substantial variability in task definitions, datasets, and evaluation procedures. This study provides the first comprehensive analysis of MCR automation research, aiming to characterize the field's evolution, formalize learning tasks, highlight methodological challenges, and offer actionable recommendations to guide future research. Focusing on the primary code review tasks, we systematically surveyed 691 publications and identified 24 relevant studies published between May 2015 and April 2024. Each study was analyzed in terms of tasks, models, metrics, baselines, results, validity concerns, and artifact availability. In particular, our analysis reveals significant potential for standardization, including 48 task metric combinations, 22 of which were unique to their original paper, and limited dataset reuse. We highlight challenges and derive concrete recommendations for examples such as the temporal bias threat, which are rarely addressed so far. Our work contributes to a clearer overview of the field, supports the framing of new research, helps to avoid pitfalls, and promotes greater standardization in evaluation practices. 

**Abstract (ZH)**: 现代代码审查自动化研究综述：特征、挑战与标准化建议 

---
# Automating Conflict-Aware ACL Configurations with Natural Language Intents 

**Title (ZH)**: 基于自然语言意图的自动冲突感知ACL配置 

**Authors**: Wenlong Ding, Jianqiang Li, Zhixiong Niu, Huangxun Chen, Yongqiang Xiong, Hong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17990)  

**Abstract**: ACL configuration is essential for managing network flow reachability, yet its complexity grows significantly with topologies and pre-existing rules. To carry out ACL configuration, the operator needs to (1) understand the new configuration policies or intents and translate them into concrete ACL rules, (2) check and resolve any conflicts between the new and existing rules, and (3) deploy them across the network. Existing systems rely heavily on manual efforts for these tasks, especially for the first two, which are tedious, error-prone, and impractical to scale.
We propose Xumi to tackle this problem. Leveraging LLMs with domain knowledge of the target network, Xumi automatically and accurately translates the natural language intents into complete ACL rules to reduce operators' manual efforts. Xumi then detects all potential conflicts between new and existing rules and generates resolved intents for deployment with operators' guidance, and finally identifies the best deployment plan that minimizes the rule additions while satisfying all intents. Evaluation shows that Xumi accelerates the entire configuration pipeline by over 10x compared to current practices, addresses O(100) conflicting ACLs and reduces rule additions by ~40% in modern cloud network. 

**Abstract (ZH)**: Xumi：利用领域知识的LLM自动实现ACL配置 

---
# Understanding Subword Compositionality of Large Language Models 

**Title (ZH)**: 大规模语言模型的子词组合性理解 

**Authors**: Qiwei Peng, Yekun Chai, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2508.17953)  

**Abstract**: Large language models (LLMs) take sequences of subwords as input, requiring them to effective compose subword representations into meaningful word-level representations. In this paper, we present a comprehensive set of experiments to probe how LLMs compose subword information, focusing on three key aspects: structural similarity, semantic decomposability, and form retention. Our analysis of the experiments suggests that these five LLM families can be classified into three distinct groups, likely reflecting difference in their underlying composition strategies. Specifically, we observe (i) three distinct patterns in the evolution of structural similarity between subword compositions and whole-word representations across layers; (ii) great performance when probing layer by layer their sensitivity to semantic decompositionality; and (iii) three distinct patterns when probing sensitivity to formal features, e.g., character sequence length. These findings provide valuable insights into the compositional dynamics of LLMs and highlight different compositional pattens in how LLMs encode and integrate subword information. 

**Abstract (ZH)**: 大规模语言模型（LLMs）以子词序列为输入，需要有效地将子词表示组合成有意义的词级表示。在本文中，我们进行了一系列全面的实验来探究LLMs如何组合子词信息，重点关注三个方面：结构相似性、语义可分解性和形式保留。我们的实验分析表明，这五类LLM可以被归类为三个不同的组别，这很可能反映了它们在基础组合策略方面的差异。具体而言，我们观察到(i) 子词组合与整个词表示之间结构相似性在不同层中的三种不同的演变模式；(ii) 在逐层探测试验其对语义可分解性的敏感度方面表现出色；以及(iii) 在探测试验对其形式特征（例如字符序列长度）的敏感度时表现出三种不同的模式。这些发现为理解LLMs的组合动态提供了宝贵见解，并强调了LLMs在编码和整合子词信息时不同的组合模式。 

---
# Debiasing Multilingual LLMs in Cross-lingual Latent Space 

**Title (ZH)**: 跨语言潜在空间中的多语言LLM去偏见化 

**Authors**: Qiwei Peng, Guimin Hu, Yekun Chai, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2508.17948)  

**Abstract**: Debiasing techniques such as SentDebias aim to reduce bias in large language models (LLMs). Previous studies have evaluated their cross-lingual transferability by directly applying these methods to LLM representations, revealing their limited effectiveness across languages. In this work, we therefore propose to perform debiasing in a joint latent space rather than directly on LLM representations. We construct a well-aligned cross-lingual latent space using an autoencoder trained on parallel TED talk scripts. Our experiments with Aya-expanse and two debiasing techniques across four languages (English, French, German, Dutch) demonstrate that a) autoencoders effectively construct a well-aligned cross-lingual latent space, and b) applying debiasing techniques in the learned cross-lingual latent space significantly improves both the overall debiasing performance and cross-lingual transferability. 

**Abstract (ZH)**: Debiasing技术如SentDebias旨在减少大型语言模型（LLMs）中的偏见。 previous studies通过直接将这些方法应用于LLM表示来评估它们的跨语言可迁移性，揭示了它们跨语言之间的有限有效性。因此，本工作提出在联合潜在空间中进行去偏见，而不是直接在LLM表示上进行。我们通过在平行TED演讲脚本上训练自编码器来构建一个良好的跨语言对齐的潜在空间。我们在Aya-expanse和两种去偏见技术在四种语言（英语、法语、德语、荷兰语）上的实验表明：a) 自编码器有效构建了一个良好的跨语言对齐的潜在空间，b) 在学习到的跨语言潜在空间中应用去偏见技术显着提高了整体去偏见性能和跨语言可迁移性。 

---
# A Feminist Account of Intersectional Algorithmic Fairness 

**Title (ZH)**: 女性主义视角下的交集算法公平性 

**Authors**: Marie Mirsch, Laila Wegner, Jonas Strube, Carmen Leicht-Scholten  

**Link**: [PDF](https://arxiv.org/pdf/2508.17944)  

**Abstract**: Intersectionality has profoundly influenced research and political action by revealing how interconnected systems of privilege and oppression influence lived experiences, yet its integration into algorithmic fairness research remains limited. Existing approaches often rely on single-axis or formal subgroup frameworks that risk oversimplifying social realities and neglecting structural inequalities. We propose Substantive Intersectional Algorithmic Fairness, extending Green's (2022) notion of substantive algorithmic fairness with insights from intersectional feminist theory. Building on this foundation, we introduce ten desiderata within the ROOF methodology to guide the design, assessment, and deployment of algorithmic systems in ways that address systemic inequities while mitigating harms to intersectionally marginalized communities. Rather than prescribing fixed operationalizations, these desiderata encourage reflection on assumptions of neutrality, the use of protected attributes, the inclusion of multiply marginalized groups, and enhancing algorithmic systems' potential. Our approach emphasizes that fairness cannot be separated from social context, and that in some cases, principled non-deployment may be necessary. By bridging computational and social science perspectives, we provide actionable guidance for more equitable, inclusive, and context-sensitive intersectional algorithmic practices. 

**Abstract (ZH)**: 实质性交叠算法公平：一种基于交叠女权理论的系统化指导框架 

---
# See What You Need: Query-Aware Visual Intelligence through Reasoning-Perception Loops 

**Title (ZH)**: 按需所见：基于查询-感知循环推理的查询意识视觉智能 

**Authors**: Zixuan Dong, Baoyun Peng, Yufei Wang, Lin Liu, Xinxin Dong, Yunlong Cao, Xiaodong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17932)  

**Abstract**: Human video comprehension demonstrates dynamic coordination between reasoning and visual attention, adaptively focusing on query-relevant details. However, current long-form video question answering systems employ rigid pipelines that decouple reasoning from perception, leading to either information loss through premature visual abstraction or computational inefficiency through exhaustive processing. The core limitation lies in the inability to adapt visual extraction to specific reasoning requirements, different queries demand fundamentally different visual evidence from the same video content. In this work, we present CAVIA, a training-free framework that revolutionizes video understanding through reasoning, perception coordination. Unlike conventional approaches where visual processing operates independently of reasoning, CAVIA creates a closed-loop system where reasoning continuously guides visual extraction based on identified information gaps. CAVIA introduces three innovations: (1) hierarchical reasoning, guided localization to precise frames; (2) cross-modal semantic bridging for targeted extraction; (3) confidence-driven iterative synthesis. CAVIA achieves state-of-the-art performance on challenging benchmarks: EgoSchema (65.7%, +5.3%), NExT-QA (76.1%, +2.6%), and IntentQA (73.8%, +6.9%), demonstrating that dynamic reasoning-perception coordination provides a scalable paradigm for video understanding. 

**Abstract (ZH)**: 人类视频理解展示了推理与视觉注意力之间的动态协调，能够适应性地关注查询相关的细节。然而，当前的长视频问答系统采用僵化的管道，将推理与感知分离开来，导致通过过早的视觉抽象丢失信息，或者通过耗尽式处理导致计算效率低下。核心限制在于无法根据特定的推理需求适配视觉提取。不同的查询从相同的视频内容中需要完全不同类型的视觉证据。在本文中，我们提出了一种无需训练的框架CAVIA，通过推理与感知协调重塑视频理解。与传统的视觉处理独立于推理的流程不同，CAVIA构建了一个闭环系统，推理不断引导视觉提取，基于识别的信息缺口。CAVIA引入了三项创新：（1）层次推理，引导精确帧的定位；（2）跨模态语义桥梁用于目标提取；（3）基于置信度的迭代合成。CAVIA在具有挑战性的基准测试上取得了最先进的性能：EgoSchema（65.7%，+5.3%），NExT-QA（76.1%，+2.6%），IntentQA（73.8%，+6.9%），证明了动态推理-感知协调为视频理解提供了可扩展的范式。 

---
# AMELIA: A Family of Multi-task End-to-end Language Models for Argumentation 

**Title (ZH)**: AMELIA：论辩领域多任务端到端语言模型系列 

**Authors**: Henri Savigny, Bruno Yun  

**Link**: [PDF](https://arxiv.org/pdf/2508.17926)  

**Abstract**: Argument mining is a subfield of argumentation that aims to automatically extract argumentative structures and their relations from natural language texts. This paper investigates how a single large language model can be leveraged to perform one or several argument mining tasks. Our contributions are two-fold. First, we construct a multi-task dataset by surveying and converting 19 well-known argument mining datasets from the literature into a unified format. Second, we explore various training strategies using Meta AI's Llama-3.1-8B-Instruct model: (1) fine-tuning on individual tasks, (2) fine-tuning jointly on multiple tasks, and (3) merging models fine-tuned separately on individual tasks. Our experiments show that task-specific fine-tuning significantly improves individual performance across all tasks. Moreover, multi-task fine-tuning maintains strong performance without degradation, suggesting effective transfer learning across related tasks. Finally, we demonstrate that model merging offers a viable compromise: it yields competitive performance while mitigating the computational costs associated with full multi-task fine-tuning. 

**Abstract (ZH)**: 论文本论证结构与关系的自动提取：单一大型语言模型在多任务下的应用探究 

---
# Riemannian Optimization for LoRA on the Stiefel Manifold 

**Title (ZH)**: 洛朗在许瓦尔兹流形上的黎曼优化 

**Authors**: Juneyoung Park, Minjae Kang, Seongbae Lee, Haegang Lee, Seongwan Kim, Jaeho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.17901)  

**Abstract**: While powerful, large language models (LLMs) present significant fine-tuning challenges due to their size. Parameter-efficient fine-tuning (PEFT) methods like LoRA provide solutions, yet suffer from critical optimizer inefficiencies; notably basis redundancy in LoRA's $B$ matrix when using AdamW, which fundamentally limits performance. We address this by optimizing the $B$ matrix on the Stiefel manifold, imposing explicit orthogonality constraints that achieve near-perfect orthogonality and full effective rank. This geometric approach dramatically enhances parameter efficiency and representational capacity. Our Stiefel optimizer consistently outperforms AdamW across benchmarks with both LoRA and DoRA, demonstrating that geometric constraints are the key to unlocking LoRA's full potential for effective LLM fine-tuning. 

**Abstract (ZH)**: 大尺寸语言模型虽然强大，但进行精细调优时面临着显著的挑战。LoRA等参数高效精细调优方法提供了解决方案，但存在关键的优化器低效问题；特别是在使用AdamW时LoRA的$B$矩阵中基的冗余性问题，这从根本上限制了性能。我们通过在Stiefel流形上优化$B$矩阵，并施加明确的正交性约束，实现了几乎完美的正交性和完整的有效秩。这种几何方法极大地提高了参数效率和表示能力。我们的Stiefel优化器在使用LoRA和DoRA的基准测试中均优于AdamW，证明了几何约束是解锁LoRA在有效调优大语言模型中全部潜力的关键。 

---
# A Defect Classification Framework for AI-Based Software Systems (AI-ODC) 

**Title (ZH)**: 基于AI的软件系统缺陷分类框架（AI-ODC） 

**Authors**: Mohammed O. Alannsary  

**Link**: [PDF](https://arxiv.org/pdf/2508.17900)  

**Abstract**: Artificial Intelligence has gained a lot of attention recently, it has been utilized in several fields ranging from daily life activities, such as responding to emails and scheduling appointments, to manufacturing and automating work activities. Artificial Intelligence systems are mainly implemented as software solutions, and it is essential to discover and remove software defects to assure its quality using defect analysis which is one of the major activities that contribute to software quality. Despite the proliferation of AI-based systems, current defect analysis models fail to capture their unique attributes. This paper proposes a framework inspired by the Orthogonal Defect Classification (ODC) paradigm and enables defect analysis of Artificial Intelligence systems while recognizing its special attributes and characteristics. This study demonstrated the feasibility of modifying ODC for AI systems to classify its defects. The ODC was adjusted to accommodate the Data, Learning, and Thinking aspects of AI systems which are newly introduced classification dimensions. This adjustment involved the introduction of an additional attribute to the ODC attributes, the incorporation of a new severity level, and the substitution of impact areas with characteristics pertinent to AI systems. The framework was showcased by applying it to a publicly available Machine Learning bug dataset, with results analyzed through one-way and two-way analysis. The case study indicated that defects occurring during the Learning phase were the most prevalent and were significantly linked to high-severity classifications. In contrast, defects identified in the Thinking phase had a disproportionate effect on trustworthiness and accuracy. These findings illustrate AIODC's capability to identify high-risk defect categories and inform focused quality assurance measures. 

**Abstract (ZH)**: 人工智能近年来引起了广泛关注，已被应用于从日常活动到制造业和自动化工作的多个领域。人工智能系统主要作为软件解决方案实施，发现并消除软件缺陷以保证其质量是缺陷分析的关键活动之一。尽管人工智能基系统得到了广泛应用，当前的缺陷分析模型仍未捕捉到其独特属性。本文提出了一种灵感来自正交缺陷分类（ODC）范式的框架，能够在识别和分析人工智能系统缺陷的同时，认识到其特殊属性和特征。该研究证明了将ODC修改应用于人工智能系统以对其进行分类是可行的。ODC被调整以适应人工智能系统的数据、学习和思考三大方面，引入了新的严重程度等级，并用与人工智能系统相关的重要特征替代影响区域。该框架通过将其应用于一个公开的机器学习错误数据集得到展示，结果通过单向和双向分析进行分析。案例研究显示，在学习阶段出现的缺陷最为普遍，并且与高严重性分类密切相关。相比之下，识别于思考阶段的缺陷对可信性和准确性产生了不成比例的影响。这些发现表明AIODC能够识别高风险缺陷类别，并为有针对性的质量保证措施提供信息。 

---
# Designing Practical Models for Isolated Word Visual Speech Recognition 

**Title (ZH)**: 孤立词视觉语音识别的实用模型设计 

**Authors**: Iason Ioannis Panagos, Giorgos Sfikas, Christophoros Nikou  

**Link**: [PDF](https://arxiv.org/pdf/2508.17894)  

**Abstract**: Visual speech recognition (VSR) systems decode spoken words from an input sequence using only the video data. Practical applications of such systems include medical assistance as well as human-machine interactions. A VSR system is typically employed in a complementary role in cases where the audio is corrupt or not available. In order to accurately predict the spoken words, these architectures often rely on deep neural networks in order to extract meaningful representations from the input sequence. While deep architectures achieve impressive recognition performance, relying on such models incurs significant computation costs which translates into increased resource demands in terms of hardware requirements and results in limited applicability in real-world scenarios where resources might be constrained. This factor prevents wider adoption and deployment of speech recognition systems in more practical applications. In this work, we aim to alleviate this issue by developing architectures for VSR that have low hardware costs. Following the standard two-network design paradigm, where one network handles visual feature extraction and another one utilizes the extracted features to classify the entire sequence, we develop lightweight end-to-end architectures by first benchmarking efficient models from the image classification literature, and then adopting lightweight block designs in a temporal convolution network backbone. We create several unified models with low resource requirements but strong recognition performance. Experiments on the largest public database for English words demonstrate the effectiveness and practicality of our developed models. Code and trained models will be made publicly available. 

**Abstract (ZH)**: 视觉语音识别（VSR）系统通过视频数据解码输入序列中的 spoken words。这类系统的实际应用包括医疗辅助以及人机交互。在音频受损或不可用的情况下，VSR 系统通常作为辅助工具使用。为了准确预测 spoken words，这些架构通常依赖于深度神经网络来从输入序列中提取有意义的表示。尽管深度架构实现了令人印象深刻的识别性能，但依赖于这些模型会导致显著的计算成本，进而增加了硬件需求，限制了此类模型在资源受限的实际场景中的应用。这一因素阻碍了语音识别系统的更广泛应用。在本文中，我们旨在通过开发具有低硬件成本的 VSR 架构来缓解这一问题。我们遵循标准的双网络设计范式，其中一个网络负责视觉特征提取，另一个网络利用提取的特征来分类整个序列。我们首先在图像分类文献中基准测试高效的模型，然后采用轻量级的模块设计在时间卷积神经网络骨干中加以应用，创建了具有低资源需求但强大识别性能的统一模型。在最大的公共英语单词数据库上的实验表明，我们开发的模型的有效性和实用性。代码和训练模型将公开发布。 

---
# Edge-Enhanced Vision Transformer Framework for Accurate AI-Generated Image Detection 

**Title (ZH)**: 基于边缘增强的视觉变换器框架：用于准确检测AI生成图像的研究 

**Authors**: Dabbrata Das, Mahshar Yahan, Md Tareq Zaman, Md Rishadul Bayesh  

**Link**: [PDF](https://arxiv.org/pdf/2508.17877)  

**Abstract**: The rapid advancement of generative models has led to a growing prevalence of highly realistic AI-generated images, posing significant challenges for digital forensics and content authentication. Conventional detection methods mainly rely on deep learning models that extract global features, which often overlook subtle structural inconsistencies and demand substantial computational resources. To address these limitations, we propose a hybrid detection framework that combines a fine-tuned Vision Transformer (ViT) with a novel edge-based image processing module. The edge-based module computes variance from edge-difference maps generated before and after smoothing, exploiting the observation that AI-generated images typically exhibit smoother textures, weaker edges, and reduced noise compared to real images. When applied as a post-processing step on ViT predictions, this module enhances sensitivity to fine-grained structural cues while maintaining computational efficiency. Extensive experiments on the CIFAKE, Artistic, and Custom Curated datasets demonstrate that the proposed framework achieves superior detection performance across all benchmarks, attaining 97.75% accuracy and a 97.77% F1-score on CIFAKE, surpassing widely adopted state-of-the-art models. These results establish the proposed method as a lightweight, interpretable, and effective solution for both still images and video frames, making it highly suitable for real-world applications in automated content verification and digital forensics. 

**Abstract (ZH)**: 生成模型的快速进步导致了高度逼真的人工智能生成图像的普遍性增加，这对数字取证和内容认证提出了重大挑战。传统检测方法主要依赖于提取全局特征的深度学习模型，这些方法往往忽视了细微的结构不一致性，并需要大量的计算资源。为了解决这些问题，我们提出了一种结合微调的视觉变换器（ViT）和一种新型边缘导向图像处理模块的混合检测框架。边缘导向模块通过在平滑前后生成的边缘差异图像上计算方差，利用观察到的人工智能生成图像通常具有更平滑的纹理、更弱的边缘和减少的噪声这一事实。将此模块用作ViT预测后的处理步骤，可以增强对细粒度结构线索的敏感性，同时保持计算效率。在CIFAKE、Artistic和定制编目的数据集上的广泛实验表明，所提出的框架在所有基准测试中实现了卓越的检测性能，在CIFAKE数据集上取得了97.75%的准确率和97.77%的F1分数，超越了广泛采用的最先进的模型。这些结果确立了所提出的方法作为一种轻量级、可解释且有效的解决方案，适用于静态图像和视频帧，在自动化内容验证和数字取证的实际应用中具有很高的适用性。 

---
# Vocoder-Projected Feature Discriminator 

**Title (ZH)**: 矢量器投影特征判别器 

**Authors**: Takuhiro Kaneko, Hirokazu Kameoka, Kou Tanaka, Yuto Kondo  

**Link**: [PDF](https://arxiv.org/pdf/2508.17874)  

**Abstract**: In text-to-speech (TTS) and voice conversion (VC), acoustic features, such as mel spectrograms, are typically used as synthesis or conversion targets owing to their compactness and ease of learning. However, because the ultimate goal is to generate high-quality waveforms, employing a vocoder to convert these features into waveforms and applying adversarial training in the time domain is reasonable. Nevertheless, upsampling the waveform introduces significant time and memory overheads. To address this issue, we propose a vocoder-projected feature discriminator (VPFD), which uses vocoder features for adversarial training. Experiments on diffusion-based VC distillation demonstrated that a pretrained and frozen vocoder feature extractor with a single upsampling step is necessary and sufficient to achieve a VC performance comparable to that of waveform discriminators while reducing the training time and memory consumption by 9.6 and 11.4 times, respectively. 

**Abstract (ZH)**: 在文本到语音（TTS）和语音转换（VC）中，由于其紧凑性和易于学习的特性，通常采用梅尔频谱等声学特征作为合成或转换的目标。然而，由于最终目标是生成高质量的波形，使用 vocoder 将这些特征转换为波形并进行时域的对抗训练是合理的。尽管如此，上采样波形会引入显著的时间和内存开销。为了解决这个问题，我们提出了一种 vocoder 投影特征判别器（VPFD），该判别器使用 vocoder 特征进行对抗训练。基于扩散的 VC  distillation 实验表明，使用单步上采样的预训练和冻结的 vocoder 特征提取器是必要且足够的，可以在将训练时间减少9.6倍、内存消耗减少11.4倍的情况下，达到与波形判别器相当的性能。 

---
# FasterVoiceGrad: Faster One-step Diffusion-Based Voice Conversion with Adversarial Diffusion Conversion Distillation 

**Title (ZH)**: FasterVoiceGrad: 基于对抗扩散转换蒸馏的一步扩散语音转换加速方法 

**Authors**: Takuhiro Kaneko, Hirokazu Kameoka, Kou Tanaka, Yuto Kondo  

**Link**: [PDF](https://arxiv.org/pdf/2508.17868)  

**Abstract**: A diffusion-based voice conversion (VC) model (e.g., VoiceGrad) can achieve high speech quality and speaker similarity; however, its conversion process is slow owing to iterative sampling. FastVoiceGrad overcomes this limitation by distilling VoiceGrad into a one-step diffusion model. However, it still requires a computationally intensive content encoder to disentangle the speaker's identity and content, which slows conversion. Therefore, we propose FasterVoiceGrad, a novel one-step diffusion-based VC model obtained by simultaneously distilling a diffusion model and content encoder using adversarial diffusion conversion distillation (ADCD), where distillation is performed in the conversion process while leveraging adversarial and score distillation training. Experimental evaluations of one-shot VC demonstrated that FasterVoiceGrad achieves competitive VC performance compared to FastVoiceGrad, with 6.6-6.9 and 1.8 times faster speed on a GPU and CPU, respectively. 

**Abstract (ZH)**: 基于扩散的声音转换（VC）模型（如VoiceGrad）可以实现高质量的语音和高speaker相似度，但由于迭代采样的原因，其转换过程较慢。FastVoiceGrad通过将VoiceGrad提炼成一步扩散模型来克服这一限制。然而，它仍然需要一个计算密集的内容编码器来分离说话人身份和内容，这会减慢转换速度。因此，我们提出了FasterVoiceGrad，这是一种通过对抗扩散转换提炼同时提炼扩散模型和内容编码器获得的新颖一步扩散基于VC模型，其中在转换过程中利用对抗性和分数提炼训练进行提炼。实验结果显示，FasterVoiceGrad在单次发声转换上的性能与FastVoiceGrad相当，分别在GPU和CPU上快6.6-6.9倍和1.8倍。 

---
# Ada-TransGNN: An Air Quality Prediction Model Based On Adaptive Graph Convolutional Networks 

**Title (ZH)**: Ada-TransGNN：基于自适应图卷积网络的空气质量预测模型 

**Authors**: Dan Wang, Feng Jiang, Zhanquan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17867)  

**Abstract**: Accurate air quality prediction is becoming increasingly important in the environmental field. To address issues such as low prediction accuracy and slow real-time updates in existing models, which lead to lagging prediction results, we propose a Transformer-based spatiotemporal data prediction method (Ada-TransGNN) that integrates global spatial semantics and temporal behavior. The model constructs an efficient and collaborative spatiotemporal block set comprising a multi-head attention mechanism and a graph convolutional network to extract dynamically changing spatiotemporal dependency features from complex air quality monitoring data. Considering the interaction relationships between different monitoring points, we propose an adaptive graph structure learning module, which combines spatiotemporal dependency features in a data-driven manner to learn the optimal graph structure, thereby more accurately capturing the spatial relationships between monitoring points. Additionally, we design an auxiliary task learning module that enhances the decoding capability of temporal relationships by integrating spatial context information into the optimal graph structure representation, effectively improving the accuracy of prediction results. We conducted comprehensive evaluations on a benchmark dataset and a novel dataset (Mete-air). The results demonstrate that our model outperforms existing state-of-the-art prediction models in short-term and long-term predictions. 

**Abstract (ZH)**: 基于Transformer的空间时序数据预测方法（Ada-TransGNN）：融合全局空间 semantics 和时序行为的空气质量预测 

---
# AVAM: Universal Training-free Adaptive Visual Anchoring Embedded into Multimodal Large Language Model for Multi-image Question Answering 

**Title (ZH)**: AVAM：嵌入多模态大规模语言模型的通用无训练自适应视觉锚定 

**Authors**: Kang Zeng, Guojin Zhong, Jintao Cheng, Jin Yuan, Zhiyong Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.17860)  

**Abstract**: The advancement of Multimodal Large Language Models (MLLMs) has driven significant progress in Visual Question Answering (VQA), evolving from Single to Multi Image VQA (MVQA). However, the increased number of images in MVQA inevitably introduces substantial visual redundancy that is irrelevant to question answering, negatively impacting both accuracy and efficiency. To address this issue, existing methods lack flexibility in controlling the number of compressed visual tokens and tend to produce discrete visual fragments, which hinder MLLMs' ability to comprehend images holistically. In this paper, we propose a straightforward yet universal Adaptive Visual Anchoring strategy, which can be seamlessly integrated into existing MLLMs, offering significant accuracy improvements through adaptive compression. Meanwhile, to balance the results derived from both global and compressed visual input, we further introduce a novel collaborative decoding mechanism, enabling optimal performance. Extensive experiments validate the effectiveness of our method, demonstrating consistent performance improvements across various MLLMs. The code will be publicly available. 

**Abstract (ZH)**: Multimodal Large Language Models的进展推动了视觉问答（VQA）的显著进步，从单图VQA（SVQA）发展到多图VQA（MVQA）。然而，MVQA中图片数量的增加不可避免地带来了与问题回答无关的大量视觉冗余，这不仅影响准确性，还降低了效率。为解决这一问题，现有方法在控制压缩视觉令牌数量方面缺乏灵活性，倾向于生成离散的视觉片段，从而阻碍了MLLMs对图像的整体理解能力。本文提出了一种简单而通用的自适应视觉锚定策略，可以无缝集成到现有的MLLMs中，通过自适应压缩提高准确性。同时，为了平衡来自全局和压缩视觉输入的结果，我们进一步引入了一种新的协作解码机制，以实现最优性能。大量实验证明了我们方法的有效性，展示了在各种MLLMs上的一致性能提升。代码将公开可用。 

---
# VISA: Group-wise Visual Token Selection and Aggregation via Graph Summarization for Efficient MLLMs Inference 

**Title (ZH)**: VISA：基于图总结的组级视觉_token_选择与聚合以实现高效MLLM推理 

**Authors**: Pengfei Jiang, Hanjun Li, Linglan Zhao, Fei Chao, Ke Yan, Shouhong Ding, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2508.17857)  

**Abstract**: In this study, we introduce a novel method called group-wise \textbf{VI}sual token \textbf{S}election and \textbf{A}ggregation (VISA) to address the issue of inefficient inference stemming from excessive visual tokens in multimoal large language models (MLLMs). Compared with previous token pruning approaches, our method can preserve more visual information while compressing visual tokens. We first propose a graph-based visual token aggregation (VTA) module. VTA treats each visual token as a node, forming a graph based on semantic similarity among visual tokens. It then aggregates information from removed tokens into kept tokens based on this graph, producing a more compact visual token representation. Additionally, we introduce a group-wise token selection strategy (GTS) to divide visual tokens into kept and removed ones, guided by text tokens from the final layers of each group. This strategy progressively aggregates visual information, enhancing the stability of the visual information extraction process. We conduct comprehensive experiments on LLaVA-1.5, LLaVA-NeXT, and Video-LLaVA across various benchmarks to validate the efficacy of VISA. Our method consistently outperforms previous methods, achieving a superior trade-off between model performance and inference speed. The code is available at this https URL. 

**Abstract (ZH)**: 基于组的VI视觉标记选择与聚合（VISA）方法以解决多模态大型语言模型中由于视觉标记过多导致的低效推理问题 

---
# Group Expectation Policy Optimization for Stable Heterogeneous Reinforcement Learning in LLMs 

**Title (ZH)**: 群预期策略优化在LLMs中稳定异质强化学习 

**Authors**: Han Zhang, Ruibin Zheng, Zexuan Yi, Hanyang Peng, Hui Wang, Yue Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17850)  

**Abstract**: As single-center computing approaches power constraints, decentralized training is becoming essential. Reinforcement Learning (RL) post-training enhances Large Language Models (LLMs) but faces challenges in heterogeneous distributed environments due to its tightly-coupled sampling-learning alternation. We propose HeteroRL, an asynchronous RL architecture that decouples rollout sampling from parameter learning, enabling robust deployment across geographically distributed nodes under network delays. We identify that latency-induced KL divergence causes importance sampling failure due to high variance. To address this, we propose Group Expectation Policy Optimization (GEPO), which reduces importance weight variance through a refined sampling mechanism. Theoretically, GEPO achieves exponential variance reduction. Experiments show it maintains superior stability over methods like GRPO, with less than 3% performance degradation under 1800-second delays, demonstrating strong potential for decentralized RL in heterogeneous networks. 

**Abstract (ZH)**: 异构分布式环境下的异步 reinforcement learning架构：HeteroRL及其在延迟网络中的应用 

---
# Limits of message passing for node classification: How class-bottlenecks restrict signal-to-noise ratio 

**Title (ZH)**: 消息传递在节点分类中的限制：类瓶颈如何限制信噪比 

**Authors**: Jonathan Rubin, Sahil Loomba, Nick S. Jones  

**Link**: [PDF](https://arxiv.org/pdf/2508.17822)  

**Abstract**: Message passing neural networks (MPNNs) are powerful models for node classification but suffer from performance limitations under heterophily (low same-class connectivity) and structural bottlenecks in the graph. We provide a unifying statistical framework exposing the relationship between heterophily and bottlenecks through the signal-to-noise ratio (SNR) of MPNN representations. The SNR decomposes model performance into feature-dependent parameters and feature-independent sensitivities. We prove that the sensitivity to class-wise signals is bounded by higher-order homophily -- a generalisation of classical homophily to multi-hop neighbourhoods -- and show that low higher-order homophily manifests locally as the interaction between structural bottlenecks and class labels (class-bottlenecks). Through analysis of graph ensembles, we provide a further quantitative decomposition of bottlenecking into underreaching (lack of depth implying signals cannot arrive) and oversquashing (lack of breadth implying signals arriving on fewer paths) with closed-form expressions. We prove that optimal graph structures for maximising higher-order homophily are disjoint unions of single-class and two-class-bipartite clusters. This yields BRIDGE, a graph ensemble-based rewiring algorithm that achieves near-perfect classification accuracy across all homophily regimes on synthetic benchmarks and significant improvements on real-world benchmarks, by eliminating the ``mid-homophily pitfall'' where MPNNs typically struggle, surpassing current standard rewiring techniques from the literature. Our framework, whose code we make available for public use, provides both diagnostic tools for assessing MPNN performance, and simple yet effective methods for enhancing performance through principled graph modification. 

**Abstract (ZH)**: 统一统计框架下消息传递神经网络在异质性和结构瓶颈中的性能分析与优化 

---
# Limitations of Normalization in Attention Mechanism 

**Title (ZH)**: 注意力机制中归一化的限制 

**Authors**: Timur Mudarisov, Mikhail Burtsev, Tatiana Petrova, Radu State  

**Link**: [PDF](https://arxiv.org/pdf/2508.17821)  

**Abstract**: This paper investigates the limitations of the normalization in attention mechanisms. We begin with a theoretical framework that enables the identification of the model's selective ability and the geometric separation involved in token selection. Our analysis includes explicit bounds on distances and separation criteria for token vectors under softmax scaling. Through experiments with pre-trained GPT-2 model, we empirically validate our theoretical results and analyze key behaviors of the attention mechanism. Notably, we demonstrate that as the number of selected tokens increases, the model's ability to distinguish informative tokens declines, often converging toward a uniform selection pattern. We also show that gradient sensitivity under softmax normalization presents challenges during training, especially at low temperature settings. These findings advance current understanding of softmax-based attention mechanism and motivate the need for more robust normalization and selection strategies in future attention architectures. 

**Abstract (ZH)**: 本文探讨了注意力机制中归一化方法的局限性。我们以一个理论框架为基础，能够识别模型的选择能力和词元选择中的几何分离。我们的分析包括在softmax缩放下的词元向量距离和分离标准的显式边界。通过使用预训练的GPT-2模型进行实验，我们 empirically 验证了理论结果，并分析了注意力机制的关键行为。值得注意的是，我们证明随着选定词元数量的增加，模型区分信息性词元的能力下降，通常会朝向均匀选择模式收敛。我们还展示在softmax归一化下的梯度敏感性会在训练中遇到挑战，特别是在低温度设置下。这些发现推进了对基于softmax的注意力机制的理解，并激发了对未来注意力架构中更稳健的归一化和选择策略的需求。 

---
# UniSino: Physics-Driven Foundational Model for Universal CT Sinogram Standardization 

**Title (ZH)**: UniSino：基于物理驱动的基础模型用于通用CT sinogram标准化 

**Authors**: Xingyu Ai, Shaoyu Wang, Zhiyuan Jia, Ao Xu, Hongming Shan, Jianhua Ma, Qiegen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17816)  

**Abstract**: During raw-data acquisition in CT imaging, diverse factors can degrade the collected sinograms, with undersampling and noise leading to severe artifacts and noise in reconstructed images and compromising diagnostic accuracy. Conventional correction methods rely on manually designed algorithms or fixed empirical parameters, but these approaches often lack generalizability across heterogeneous artifact types. To address these limitations, we propose UniSino, a foundation model for universal CT sinogram standardization. Unlike existing foundational models that operate in image domain, UniSino directly standardizes data in the projection domain, which enables stronger generalization across diverse undersampling scenarios. Its training framework incorporates the physical characteristics of sinograms, enhancing generalization and enabling robust performance across multiple subtasks spanning four benchmark datasets. Experimental results demonstrate thatUniSino achieves superior reconstruction quality both single and mixed undersampling case, demonstrating exceptional robustness and generalization in sinogram enhancement for CT imaging. The code is available at: this https URL. 

**Abstract (ZH)**: 在CT成像的数据采集过程中，多种因素会导致采集到的sinogram降质，采样不足和噪声会导致重建图像出现严重的伪影和噪声，从而影响诊断准确性。常规的校正方法依赖于人工设计的算法或固定的经验参数，但这些方法往往缺乏对异质伪影类型的普适性。为了解决这些限制，我们提出UniSino，这是一种用于通用CT sinogram标准化的基础模型。与现有的在图像域工作的基础模型不同，UniSino直接在投影域标准化数据，这使其能够在多种欠采样场景中展现出更强的泛化能力。其训练框架结合了sinogram的物理特性，增强了泛化能力，并能够跨四个基准数据集的多项子任务获得稳健的表现。实验结果表明，UniSino在单一和混合欠采样情况下均能实现优秀的重建质量，显示出在CT成像中对sinogram增强的卓越稳健性和泛化能力。代码可在以下链接获取：this https URL。 

---
# Scalable Engine and the Performance of Different LLM Models in a SLURM based HPC architecture 

**Title (ZH)**: 基于SLURM的HPC架构中可扩展引擎与不同LLM模型的性能研究 

**Authors**: Anderson de Lima Luiz, Shubham Vijay Kurlekar, Munir Georges  

**Link**: [PDF](https://arxiv.org/pdf/2508.17814)  

**Abstract**: This work elaborates on a High performance computing (HPC) architecture based on Simple Linux Utility for Resource Management (SLURM) [1] for deploying heterogeneous Large Language Models (LLMs) into a scalable inference engine. Dynamic resource scheduling and seamless integration of containerized microservices have been leveraged herein to manage CPU, GPU, and memory allocations efficiently in multi-node clusters. Extensive experiments, using Llama 3.2 (1B and 3B parameters) [2] and Llama 3.1 (8B and 70B) [3], probe throughput, latency, and concurrency and show that small models can handle up to 128 concurrent requests at sub-50 ms latency, while for larger models, saturation happens with as few as two concurrent users, with a latency of more than 2 seconds. This architecture includes Representational State Transfer Application Programming Interfaces (REST APIs) [4] endpoints for single and bulk inferences, as well as advanced workflows such as multi-step "tribunal" refinement. Experimental results confirm minimal overhead from container and scheduling activities and show that the approach scales reliably both for batch and interactive settings. We further illustrate real-world scenarios, including the deployment of chatbots with retrievalaugmented generation, which helps to demonstrate the flexibility and robustness of the architecture. The obtained results pave ways for significantly more efficient, responsive, and fault-tolerant LLM inference on large-scale HPC infrastructures. 

**Abstract (ZH)**: 基于Simple Linux Utility for Resource Management (SLURM)的高性能计算架构：可扩展异构大型语言模型推理引擎的设计与实现 

---
# MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting 

**Title (ZH)**: MeshSplat: 基于高斯点渲染的可泛化的稀疏视图表面重建 

**Authors**: Hanzhi Chang, Ruijie Zhu, Wenjie Chang, Mulin Yu, Yanzhe Liang, Jiahao Lu, Zhuoyuan Li, Tianzhu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17811)  

**Abstract**: Surface reconstruction has been widely studied in computer vision and graphics. However, existing surface reconstruction works struggle to recover accurate scene geometry when the input views are extremely sparse. To address this issue, we propose MeshSplat, a generalizable sparse-view surface reconstruction framework via Gaussian Splatting. Our key idea is to leverage 2DGS as a bridge, which connects novel view synthesis to learned geometric priors and then transfers these priors to achieve surface reconstruction. Specifically, we incorporate a feed-forward network to predict per-view pixel-aligned 2DGS, which enables the network to synthesize novel view images and thus eliminates the need for direct 3D ground-truth supervision. To improve the accuracy of 2DGS position and orientation prediction, we propose a Weighted Chamfer Distance Loss to regularize the depth maps, especially in overlapping areas of input views, and also a normal prediction network to align the orientation of 2DGS with normal vectors predicted by a monocular normal estimator. Extensive experiments validate the effectiveness of our proposed improvement, demonstrating that our method achieves state-of-the-art performance in generalizable sparse-view mesh reconstruction tasks. Project Page: this https URL 

**Abstract (ZH)**: 基于高斯统的通用稀疏视图表面重建框架 MeshSplat 

---
# Adaptive Output Steps: FlexiSteps Network for Dynamic Trajectory Prediction 

**Title (ZH)**: 自适应输出步长：动态步长网络用于动态轨迹预测 

**Authors**: Yunxiang Liu, Hongkuo Niu, Jianlin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17797)  

**Abstract**: Accurate trajectory prediction is vital for autonomous driving, robotics, and intelligent decision-making systems, yet traditional models typically rely on fixed-length output predictions, limiting their adaptability to dynamic real-world scenarios. In this paper, we introduce the FlexiSteps Network (FSN), a novel framework that dynamically adjusts prediction output time steps based on varying contextual conditions. Inspired by recent advancements addressing observation length discrepancies and dynamic feature extraction, FSN incorporates an pre-trained Adaptive Prediction Module (APM) to evaluate and adjust the output steps dynamically, ensuring optimal prediction accuracy and efficiency. To guarantee the plug-and-play of our FSN, we also design a Dynamic Decoder(DD). Additionally, to balance the prediction time steps and prediction accuracy, we design a scoring mechanism, which not only introduces the Fréchet distance to evaluate the geometric similarity between the predicted trajectories and the ground truth trajectories but the length of predicted steps is also considered. Extensive experiments conducted on benchmark datasets including Argoverse and INTERACTION demonstrate the effectiveness and flexibility of our proposed FSN framework. 

**Abstract (ZH)**: 准确的轨迹预测对于自动驾驶、机器人技术和智能决策系统至关重要，然而传统模型通常依赖固定长度的预测输出，限制了其适应动态现实场景的能力。本文引入了FlexiSteps网络(FSN)，这是一种新型框架，可以根据变化的上下文条件动态调整预测输出时间步长。受解决观测长度差异和动态特征提取最新进展的启发，FSN结合了一个预训练的自适应预测模块(APM)，以动态评估和调整输出步骤，确保预测的最佳准确性和效率。为了保证FSN的即插即用功能，我们还设计了一个动态解码器(DD)。此外，为了平衡预测时间步长和预测准确性，我们设计了一个评分机制，不仅引入了Fréchet距离来评估预测轨迹与真实轨迹之间的几何相似性，还考虑了预测步长的长度。在Argoverse和INTERACTION基准数据集上的广泛实验验证了我们提出的FSN框架的有效性和灵活性。 

---
# Proximal Supervised Fine-Tuning 

**Title (ZH)**: proximal 监督微调 

**Authors**: Wenhong Zhu, Ruobing Xie, Rui Wang, Xingwu Sun, Di Wang, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17784)  

**Abstract**: Supervised fine-tuning (SFT) of foundation models often leads to poor generalization, where prior capabilities deteriorate after tuning on new tasks or domains. Inspired by trust-region policy optimization (TRPO) and proximal policy optimization (PPO) in reinforcement learning (RL), we propose Proximal SFT (PSFT). This fine-tuning objective incorporates the benefits of trust-region, effectively constraining policy drift during SFT while maintaining competitive tuning. By viewing SFT as a special case of policy gradient methods with constant positive advantages, we derive PSFT that stabilizes optimization and leads to generalization, while leaving room for further optimization in subsequent post-training stages. Experiments across mathematical and human-value domains show that PSFT matches SFT in-domain, outperforms it in out-of-domain generalization, remains stable under prolonged training without causing entropy collapse, and provides a stronger foundation for the subsequent optimization. 

**Abstract (ZH)**: 监督微调（SFT）往往导致泛化能力较差，其中基础能力在针对新任务或领域进行微调后退化。受强化学习（RL）中信任区域策略优化（TRPO）和近端策略优化（PPO）的启发，我们提出了近端监督微调（PSFT）。该微调目标结合了信任区域的优点，有效地在监督微调过程中限制策略漂移，同时保持竞争性的微调效果。通过将监督微调视为具有恒定正优势的策略梯度方法的特殊情形，我们推导出PSFT，该方法稳定优化过程并提高泛化能力，同时为后续的后训练阶段保留进一步优化的空间。跨数学和人类价值领域的实验表明，PSFT在领域内与监督微调表现相当，在领域外泛化效果更优，并且在长时间训练过程中保持稳定，不会导致熵崩溃，为后续优化提供了更坚实的基础。 

---
# Algebraic Approach to Ridge-Regularized Mean Squared Error Minimization in Minimal ReLU Neural Network 

**Title (ZH)**: 基于代数方法的最小ReLU神经网络中岭正则化均方误差最小化研究 

**Authors**: Ryoya Fukasaku, Yutaro Kabata, Akifumi Okuno  

**Link**: [PDF](https://arxiv.org/pdf/2508.17783)  

**Abstract**: This paper investigates a perceptron, a simple neural network model, with ReLU activation and a ridge-regularized mean squared error (RR-MSE). Our approach leverages the fact that the RR-MSE for ReLU perceptron is piecewise polynomial, enabling a systematic analysis using tools from computational algebra. In particular, we develop a Divide-Enumerate-Merge strategy that exhaustively enumerates all local minima of the RR-MSE. By virtue of the algebraic formulation, our approach can identify not only the typical zero-dimensional minima (i.e., isolated points) obtained by numerical optimization, but also higher-dimensional minima (i.e., connected sets such as curves, surfaces, or hypersurfaces). Although computational algebraic methods are computationally very intensive for perceptrons of practical size, as a proof of concept, we apply the proposed approach in practice to minimal perceptrons with a few hidden units. 

**Abstract (ZH)**: 本文研究了带有ReLU激活和岭正则化均方误差（RR-MSE）的感知器，这是一种简单的神经网络模型。我们的方法利用了ReLU感知器的RR-MSE为分段多项式的事实，从而利用计算代数中的工具进行系统的分析。特别是，我们开发了一种划分-枚举-合并策略，可以穷尽地枚举RR-MSE的所有局部最小值。由于采用代数表示，我们的方法不仅可以识别数值优化得到的典型零维最小值（即孤立点），还可以识别高维最小值（如曲线、曲面或超曲面等连接集合）。尽管计算代数方法对于实际规模的感知器来说计算上非常密集，作为概念验证，我们将所提出的方法应用于具有少量隐藏单元的最小感知器进行实践研究。 

---
# DiffusionGS: Generative Search with Query Conditioned Diffusion in Kuaishou 

**Title (ZH)**: DiffusionGS: 基于查询条件扩散的生成搜索在快手 

**Authors**: Qinyao Li, Xiaoyang Zheng, Qihang Zhao, Ke Xu, Zhongbo Sun, Chao Wang, Chenyi Lei, Han Li, Wenwu Ou  

**Link**: [PDF](https://arxiv.org/pdf/2508.17754)  

**Abstract**: Personalized search ranking systems are critical for driving engagement and revenue in modern e-commerce and short-video platforms. While existing methods excel at estimating users' broad interests based on the filtered historical behaviors, they typically under-exploit explicit alignment between a user's real-time intent (represented by the user query) and their past actions. In this paper, we propose DiffusionGS, a novel and scalable approach powered by generative models. Our key insight is that user queries can serve as explicit intent anchors to facilitate the extraction of users' immediate interests from long-term, noisy historical behaviors. Specifically, we formulate interest extraction as a conditional denoising task, where the user's query guides a conditional diffusion process to produce a robust, user intent-aware representation from their behavioral sequence. We propose the User-aware Denoising Layer (UDL) to incorporate user-specific profiles into the optimization of attention distribution on the user's past actions. By reframing queries as intent priors and leveraging diffusion-based denoising, our method provides a powerful mechanism for capturing dynamic user interest shifts. Extensive offline and online experiments demonstrate the superiority of DiffusionGS over state-of-the-art methods. 

**Abstract (ZH)**: 个性化搜索排名系统对于推动现代电商平台和短视频平台的用户参与和收入至关重要。虽然现有方法在基于过滤的历史行为估计用户广泛的兴趣方面表现出色，但它们通常未能充分利用用户实时意图（由用户查询表示）与过去行为之间的显式对齐。在本文中，我们提出了一种基于生成模型的novel和可扩展的方法DiffusionGS。我们的核心见解是，用户查询可以作为显式的意图锚点，帮助从长期的噪声历史行为中提取用户的即时兴趣。具体而言，我们将兴趣提取公式化为一个条件去噪任务，其中用户的查询引导一个条件扩散过程，从用户的行为序列中生成稳健的、意图意识强的表示。我们提出了用户意识去噪层（UDL）来将用户特定的特征融入到注意力分布优化中，以便在用户的过去行为上进行。通过将查询重新定义为意图先验，并利用基于扩散的去噪方法，我们的方法提供了捕获用户兴趣动态变化的强大机制。广泛的离线和在线实验表明，DiffusionGS在对比最先进的方法中具有明显优势。 

---
# Talking to Robots: A Practical Examination of Speech Foundation Models for HRI Applications 

**Title (ZH)**: 与机器人对话：语言基础模型在人机交互应用中的实用性考察 

**Authors**: Theresa Pekarek Rosin, Julia Gachot, Henri-Leon Kordt, Matthias Kerzel, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2508.17753)  

**Abstract**: Automatic Speech Recognition (ASR) systems in real-world settings need to handle imperfect audio, often degraded by hardware limitations or environmental noise, while accommodating diverse user groups. In human-robot interaction (HRI), these challenges intersect to create a uniquely challenging recognition environment. We evaluate four state-of-the-art ASR systems on eight publicly available datasets that capture six dimensions of difficulty: domain-specific, accented, noisy, age-variant, impaired, and spontaneous speech. Our analysis demonstrates significant variations in performance, hallucination tendencies, and inherent biases, despite similar scores on standard benchmarks. These limitations have serious implications for HRI, where recognition errors can interfere with task performance, user trust, and safety. 

**Abstract (ZH)**: 自动语音识别(ASR)系统在实际应用场景中需要处理不Perfect的音频，这些音频常常受到硬件限制或环境噪声的影响，同时还要适应多元化的用户群体。在人机交互(HRI)中，这些挑战交汇在一起形成了一个特别具有挑战性的识别环境。我们评估了四种最先进的ASR系统在八个公开可用的数据集上的性能，这些数据集涵盖了六种难度维度：领域特定的、带方言的、噪声下的、年龄段变化的、受损的以及自发的语音。我们的分析显示，尽管在标准基准上的得分相似，这些系统在性能、错觉倾向和固有偏见方面存在显著差异。这些局限性对HRI有严重的影响，因为识别错误可能会干扰任务性能、用户信任以及安全性。 

---
# EEG-FM-Bench: A Comprehensive Benchmark for the Systematic Evaluation of EEG Foundation Models 

**Title (ZH)**: EEG-FM-Bench：一种全面的脑电基础模型系统评估基准 

**Authors**: Wei Xiong, Jiangtong Li, Jie Li, Kun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17742)  

**Abstract**: Electroencephalography (EEG) foundation models are poised to significantly advance brain signal analysis by learning robust representations from large-scale, unlabeled datasets. However, their rapid proliferation has outpaced the development of standardized evaluation benchmarks, which complicates direct model comparisons and hinders systematic scientific progress. This fragmentation fosters scientific inefficiency and obscures genuine architectural advancements. To address this critical gap, we introduce EEG-FM-Bench, the first comprehensive benchmark for the systematic and standardized evaluation of EEG foundation models (EEG-FMs). Our contributions are threefold: (1) we curate a diverse suite of downstream tasks and datasets from canonical EEG paradigms, implementing standardized processing and evaluation protocols within a unified open-source framework; (2) we benchmark prominent state-of-the-art foundation models to establish comprehensive baseline results for a clear comparison of the current landscape; (3) we perform qualitative analyses of the learned representations to provide insights into model behavior and inform future architectural design. Through extensive experiments, we find that fine-grained spatio-temporal feature interaction, multitask unified training and neuropsychological priors would contribute to enhancing model performance and generalization capabilities. By offering a unified platform for fair comparison and reproducible research, EEG-FM-Bench seeks to catalyze progress and guide the community toward the development of more robust and generalizable EEG-FMs. Code is released at this https URL. 

**Abstract (ZH)**: EEG基础模型评价基准EEG-FM-Bench：系统标准评估 EEG 基础模型 

---
# Speculative Safety-Aware Decoding 

**Title (ZH)**: 推测性安全性感知解码 

**Authors**: Xuekang Wang, Shengyu Zhu, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.17739)  

**Abstract**: Despite extensive efforts to align Large Language Models (LLMs) with human values and safety rules, jailbreak attacks that exploit certain vulnerabilities continuously emerge, highlighting the need to strengthen existing LLMs with additional safety properties to defend against these attacks. However, tuning large models has become increasingly resource-intensive and may have difficulty ensuring consistent performance. We introduce Speculative Safety-Aware Decoding (SSD), a lightweight decoding-time approach that equips LLMs with the desired safety property while accelerating inference. We assume that there exists a small language model that possesses this desired property. SSD integrates speculative sampling during decoding and leverages the match ratio between the small and composite models to quantify jailbreak risks. This enables SSD to dynamically switch between decoding schemes to prioritize utility or safety, to handle the challenge of different model capacities. The output token is then sampled from a new distribution that combines the distributions of the original and the small models. Experimental results show that SSD successfully equips the large model with the desired safety property, and also allows the model to remain helpful to benign queries. Furthermore, SSD accelerates the inference time, thanks to the speculative sampling design. 

**Abstract (ZH)**: 尽管已经做出了大量努力来使大型语言模型（LLMs）与人类价值观和安全规则保持一致，但利用特定漏洞的监狱突破攻击仍不断出现，凸显了需要通过增加额外的安全属性来加强现有LLMs以抵御这些攻击的必要性。然而，调优大型模型已变得越来越耗费资源，并可能难以确保一致的性能。我们介绍了Speculative Safety-Aware Decoding（SSD），这是一种轻量级的解码时方法，可以在不增加过多资源消耗的情况下为LLMs配备所需的安全属性并加速推理。我们假设存在一个小语言模型具有此所需属性。SSD 在解码过程中集成推测采样，并利用小模型和复合模型之间的匹配比来量化监狱突破风险。这使SSD能够动态切换解码方案，以优先考虑效用或安全，从而应对不同模型容量带来的挑战。最终输出的令牌来自结合原模型和小模型分布的新分布。实验结果表明，SSD 成功为大型模型配备了所需的安全属性，同时也使模型能够对良性查询保持帮助。此外，得益于推测采样的设计，SSD 还加速了推理时间。 

---
# Instant Preference Alignment for Text-to-Image Diffusion Models 

**Title (ZH)**: 文本到图像扩散模型的即时偏好对齐 

**Authors**: Yang Li, Songlin Yang, Xiaoxuan Han, Wei Wang, Jing Dong, Yueming Lyu, Ziyu Xue  

**Link**: [PDF](https://arxiv.org/pdf/2508.17718)  

**Abstract**: Text-to-image (T2I) generation has greatly enhanced creative expression, yet achieving preference-aligned generation in a real-time and training-free manner remains challenging. Previous methods often rely on static, pre-collected preferences or fine-tuning, limiting adaptability to evolving and nuanced user intents. In this paper, we highlight the need for instant preference-aligned T2I generation and propose a training-free framework grounded in multimodal large language model (MLLM) priors. Our framework decouples the task into two components: preference understanding and preference-guided generation. For preference understanding, we leverage MLLMs to automatically extract global preference signals from a reference image and enrich a given prompt using structured instruction design. Our approach supports broader and more fine-grained coverage of user preferences than existing methods. For preference-guided generation, we integrate global keyword-based control and local region-aware cross-attention modulation to steer the diffusion model without additional training, enabling precise alignment across both global attributes and local elements. The entire framework supports multi-round interactive refinement, facilitating real-time and context-aware image generation. Extensive experiments on the Viper dataset and our collected benchmark demonstrate that our method outperforms prior approaches in both quantitative metrics and human evaluations, and opens up new possibilities for dialog-based generation and MLLM-diffusion integration. 

**Abstract (ZH)**: 基于多模态大型语言模型的无需训练的文本到图像生成 

---
# Database Normalization via Dual-LLM Self-Refinement 

**Title (ZH)**: 通过双LLM自助精炼实现数据库规范化 

**Authors**: Eunjae Jo, Nakyung Lee, Gyuyeong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.17693)  

**Abstract**: Database normalization is crucial to preserving data integrity. However, it is time-consuming and error-prone, as it is typically performed manually by data engineers. To this end, we present Miffie, a database normalization framework that leverages the capability of large language models. Miffie enables automated data normalization without human effort while preserving high accuracy. The core of Miffie is a dual-model self-refinement architecture that combines the best-performing models for normalized schema generation and verification, respectively. The generation module eliminates anomalies based on the feedback of the verification module until the output schema satisfies the requirement for normalization. We also carefully design task-specific zero-shot prompts to guide the models for achieving both high accuracy and cost efficiency. Experimental results show that Miffie can normalize complex database schemas while maintaining high accuracy. 

**Abstract (ZH)**: 数据库规范化对于维护数据完整性至关重要。但由于其通常由数据工程师手动完成而耗时且易出错，因此我们提出了一种名为Miffie的数据库规范化框架，该框架利用了大型语言模型的能力。Miffie能够在不依赖人工的情况下自动进行数据规范化，同时保持高准确性。Miffie的核心是一个双重模型自改进架构，结合了表现最佳的模型分别用于规范化模式生成和验证。生成模块根据验证模块的反馈消除异常，直到输出模式满足规范化要求。我们还精心设计了特定任务的零-shot提示，以指导模型实现高准确性和低成本。实验结果表明，Miffie能够保持高准确性的前提下规范化复杂的数据库模式。 

---
# Unlearning as Ablation: Toward a Falsifiable Benchmark for Generative Scientific Discovery 

**Title (ZH)**: 删除作为消融：通往可验证的生成性科学发现基准之路 

**Authors**: Robert Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17681)  

**Abstract**: Bold claims about AI's role in science-from "AGI will cure all diseases" to promises of radically accelerated discovery-raise a central epistemic question: do large language models (LLMs) truly generate new knowledge, or do they merely remix memorized fragments? We propose unlearning-as-ablation as a falsifiable test of constructive scientific discovery. The method systematically removes a target result and its entire forget-closure (lemmas, paraphrases, and multi-hop entailments) and then evaluates whether the model can re-derive the result from only permitted axioms and tools. Success provides evidence for genuine generative capability; failure exposes current limits. Unlike prevailing motivations for unlearning-privacy, copyright, or safety-our framing repositions it as an epistemic probe for AI-for-Science. We argue that such tests could serve as the next generation of benchmarks, much as ImageNet catalyzed progress in vision: distinguishing models that can merely recall from those that can constructively generate new scientific knowledge. We outline a minimal pilot in mathematics and algorithms, and discuss extensions to physics, chemistry, and biology. Whether models succeed or fail, unlearning-as-ablation provides a principled framework to map the true reach and limits of AI scientific discovery. This is a position paper: we advance a conceptual and methodological argument rather than new empirical results. 

**Abstract (ZH)**: 关于AI在科学中的角色的夸张主张——从“AGI将治愈所有疾病”到加速发现的承诺——提出了一个中心的 epistemic 问题：大型语言模型（LLMs）真的生成了新的知识，还是仅仅重混了已记忆的片段？我们提出“消学作为消除试验”作为一种可证伪的测试方法，用于验证建设性科学发现。该方法系统地移除目标结果及其整个忘记闭包（引理、同义表达和多跳蕴含），然后评估模型是否仅从许可公理和工具重新推导该结果。成功提供了真正生成能力的证据；失败则揭示了当前的局限。与现有的消学动机不同——隐私、版权或安全——我们的框架将其重新定位为AI-for-Science的 epistemic 探针。我们argue这样的测试可以作为下一代基准测试的一部分，类似于ImageNet推动了视觉领域的进步：区分那些只能回忆的模型与那些能够构建性生成新科学知识的模型。我们概述了一个最小规模的数学和算法试点，并讨论了其在物理、化学和生物学中的扩展。无论模型成功与否，“消学作为消除试验”提供了一种原理性的框架来映射AI科学研究的真实范围和局限。这是一个立场论文：我们推进了概念和方法论的论证，而非新的实证结果。 

---
# Robustness Feature Adapter for Efficient Adversarial Training 

**Title (ZH)**: 稳健性特征适配器用于高效对抗训练 

**Authors**: Quanwei Wu, Jun Guo, Wei Wang, Yi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17680)  

**Abstract**: Adversarial training (AT) with projected gradient descent is the most popular method to improve model robustness under adversarial attacks. However, computational overheads become prohibitively large when AT is applied to large backbone models. AT is also known to have the issue of robust overfitting. This paper contributes to solving both problems simultaneously towards building more trustworthy foundation models. In particular, we propose a new adapter-based approach for efficient AT directly in the feature space. We show that the proposed adapter-based approach can improve the inner-loop convergence quality by eliminating robust overfitting. As a result, it significantly increases computational efficiency and improves model accuracy by generalizing adversarial robustness to unseen attacks. We demonstrate the effectiveness of the new adapter-based approach in different backbone architectures and in AT at scale. 

**Abstract (ZH)**: 基于适配器的方法：在特征空间中高效对抗训练的同时解决鲁棒过拟合问题 

---
# Attacking LLMs and AI Agents: Advertisement Embedding Attacks Against Large Language Models 

**Title (ZH)**: 攻击大语言模型和AI代理：针对大型语言模型的广告嵌入攻击 

**Authors**: Qiming Guo, Jinwen Tang, Xingran Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17674)  

**Abstract**: We introduce Advertisement Embedding Attacks (AEA), a new class of LLM security threats that stealthily inject promotional or malicious content into model outputs and AI agents. AEA operate through two low-cost vectors: (1) hijacking third-party service-distribution platforms to prepend adversarial prompts, and (2) publishing back-doored open-source checkpoints fine-tuned with attacker data. Unlike conventional attacks that degrade accuracy, AEA subvert information integrity, causing models to return covert ads, propaganda, or hate speech while appearing normal. We detail the attack pipeline, map five stakeholder victim groups, and present an initial prompt-based self-inspection defense that mitigates these injections without additional model retraining. Our findings reveal an urgent, under-addressed gap in LLM security and call for coordinated detection, auditing, and policy responses from the AI-safety community. 

**Abstract (ZH)**: 我们将介绍广告嵌入攻击（AEA），这是一种新的LLM安全威胁类别，能够隐蔽地将推广或恶意内容注入模型输出和AI代理。AEA通过两种低成本途径运作：（1）劫持第三方服务分发平台，在提示前添加对抗性提示；（2）发布带有后门的开放源代码检查点，并使用攻击者数据进行微调。不同于传统攻击降低准确性，AEA破坏信息完整性，导致模型在看似正常的情况下返回隐蔽广告、宣传或仇恨言论。我们详细介绍了攻击管线，映射了五类受害方利益相关者群体，并提出了基于提示的自检防御措施，该措施可以在不重新训练模型的情况下减轻这些注入的影响。我们的研究发现了一个紧迫且尚未充分解决的LLM安全缺口，并呼吁AI安全社区采取协调的检测、审计和政策响应措施。 

---
# Consistent Opponent Modeling of Static Opponents in Imperfect-Information Games 

**Title (ZH)**: 静态对手在不完美信息游戏中的一致对手建模 

**Authors**: Sam Ganzfried  

**Link**: [PDF](https://arxiv.org/pdf/2508.17671)  

**Abstract**: The goal of agents in multi-agent environments is to maximize total reward against the opposing agents that are encountered. Following a game-theoretic solution concept, such as Nash equilibrium, may obtain a strong performance in some settings; however, such approaches fail to capitalize on historical and observed data from repeated interactions against our opponents. Opponent modeling algorithms integrate machine learning techniques to exploit suboptimal opponents utilizing available data; however, the effectiveness of such approaches in imperfect-information games to date is quite limited. We show that existing opponent modeling approaches fail to satisfy a simple desirable property even against static opponents drawn from a known prior distribution; namely, they do not guarantee that the model approaches the opponent's true strategy even in the limit as the number of game iterations approaches infinity. We develop a new algorithm that is able to achieve this property and runs efficiently by solving a convex minimization problem based on the sequence-form game representation using projected gradient descent. The algorithm is guaranteed to efficiently converge to the opponent's true strategy given observations from gameplay and possibly additional historical data if it is available. 

**Abstract (ZH)**: 多代理环境中智能体的目标是最大化与对手智能体互动时的总奖励。遵循博弈论解决方案概念，如纳什均衡，在某些情况下可以获得强大的性能；然而，此类方法未能利用在多次互动中观察到的历史和数据。对手建模算法结合机器学习技术来利用可用数据exploit suboptimal对手；然而，到目前为止，此类方法在不完全信息博弈中的有效性相当有限。我们证明，现有的对手建模方法即使在从已知先验分布中抽取静态对手的情况下，也无法满足一个简单的 desirable属性，即它们不能保证模型在游戏迭代次数趋于无穷大时接近对手的真实策略。我们开发了一种新的算法，该算法能够实现这一属性并通过基于序列形式博弈表示的凸最小化问题求解，使用投影梯度下降法高效运行。该算法可以通过游戏观察和可能的附加历史数据，保证有效地收敛到对手的真实策略。 

---
# Hierarchical Vision-Language Learning for Medical Out-of-Distribution Detection 

**Title (ZH)**: 医疗领域异常分布检测的层次视觉-语言学习 

**Authors**: Runhe Lai, Xinhua Lu, Kanghao Chen, Qichao Chen, Wei-Shi Zheng, Ruixuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17667)  

**Abstract**: In trustworthy medical diagnosis systems, integrating out-of-distribution (OOD) detection aims to identify unknown diseases in samples, thereby mitigating the risk of misdiagnosis. In this study, we propose a novel OOD detection framework based on vision-language models (VLMs), which integrates hierarchical visual information to cope with challenging unknown diseases that resemble known diseases. Specifically, a cross-scale visual fusion strategy is proposed to couple visual embeddings from multiple scales. This enriches the detailed representation of medical images and thus improves the discrimination of unknown diseases. Moreover, a cross-scale hard pseudo-OOD sample generation strategy is proposed to benefit OOD detection maximally. Experimental evaluations on three public medical datasets support that the proposed framework achieves superior OOD detection performance compared to existing methods. The source code is available at this https URL. 

**Abstract (ZH)**: 在可信赖的医疗诊断系统中，结合离分布（OOD）检测旨在识别未知疾病，从而降低误诊风险。在此研究中，我们提出了一种基于视觉-语言模型（VLMs）的新型OOD检测框架，该框架通过分层视觉信息来应对具有挑战性的未知疾病，这些未知疾病类似于已知疾病。具体而言，我们提出了一种跨尺度视觉融合策略，以结合多尺度的视觉嵌入。这丰富了医学图像的详细表示，从而提高了未知疾病的区分能力。此外，我们提出了一种跨尺度难以伪OOD样本生成策略，以最大程度地提高OOD检测性能。在三个公开的医学数据集上的实验评估支持了所提出框架相比现有方法具有更优的OOD检测性能。源代码可在该网址获取。 

---
# Weights-Rotated Preference Optimization for Large Language Models 

**Title (ZH)**: 大型语言模型中的权重旋转偏好优化 

**Authors**: Chenxu Yang, Ruipeng Jia, Mingyu Zheng, Naibin Gu, Zheng Lin, Siyuan Chen, Weichong Yin, Hua Wu, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17637)  

**Abstract**: Despite the efficacy of Direct Preference Optimization (DPO) in aligning Large Language Models (LLMs), reward hacking remains a pivotal challenge. This issue emerges when LLMs excessively reduce the probability of rejected completions to achieve high rewards, without genuinely meeting their intended goals. As a result, this leads to overly lengthy generation lacking diversity, as well as catastrophic forgetting of knowledge. We investigate the underlying reason behind this issue, which is representation redundancy caused by neuron collapse in the parameter space. Hence, we propose a novel Weights-Rotated Preference Optimization (RoPO) algorithm, which implicitly constrains the output layer logits with the KL divergence inherited from DPO and explicitly constrains the intermediate hidden states by fine-tuning on a multi-granularity orthogonal matrix. This design prevents the policy model from deviating too far from the reference model, thereby retaining the knowledge and expressive capabilities acquired during pre-training and SFT stages. Our RoPO achieves up to a 3.27-point improvement on AlpacaEval 2, and surpasses the best baseline by 6.2 to 7.5 points on MT-Bench with merely 0.015% of the trainable parameters, demonstrating its effectiveness in alleviating the reward hacking problem of DPO. 

**Abstract (ZH)**: 尽管直接偏好优化（DPO）在对齐大规模语言模型（LLMs）方面表现出色，但奖励作弊仍然是一个关键挑战。当LLMs过度减少被拒绝完成的概率以获得高奖励，而不真正实现其目标意图时，这种问题就会出现。这导致了生成过程变得过长且缺乏多样性，同时还会出现知识灾难性遗忘。我们探究了这一问题的本质原因，即参数空间中神经元崩溃导致的表示冗余。因此，我们提出了一种新颖的加权旋转偏好优化（RoPO）算法，该算法通过从DPO继承的KL散度隐式约束输出层分类概率，并通过在多粒度正交矩阵上进行微调显式约束中间隐藏状态。这种设计防止策略模型偏离参考模型过远，从而保留了预训练和微调阶段所获得的知识和表达能力。RoPO在AlpacaEval 2上实现了高达3.27点的改进，在MT-Bench上仅使用0.015%的可训练参数超越了最佳基线6.2到7.5点，证明了其在缓解DPO的奖励作弊问题方面的有效性。 

---
# Few-Shot Pattern Detection via Template Matching and Regression 

**Title (ZH)**: 基于模板匹配和回归的少样本模式检测 

**Authors**: Eunchan Jo, Dahyun Kang, Sanghyun Kim, Yunseon Choi, Minsu Cho  

**Link**: [PDF](https://arxiv.org/pdf/2508.17636)  

**Abstract**: We address the problem of few-shot pattern detection, which aims to detect all instances of a given pattern, typically represented by a few exemplars, from an input image. Although similar problems have been studied in few-shot object counting and detection (FSCD), previous methods and their benchmarks have narrowed patterns of interest to object categories and often fail to localize non-object patterns. In this work, we propose a simple yet effective detector based on template matching and regression, dubbed TMR. While previous FSCD methods typically represent target exemplars as spatially collapsed prototypes and lose structural information, we revisit classic template matching and regression. It effectively preserves and leverages the spatial layout of exemplars through a minimalistic structure with a small number of learnable convolutional or projection layers on top of a frozen backbone We also introduce a new dataset, dubbed RPINE, which covers a wider range of patterns than existing object-centric datasets. Our method outperforms the state-of-the-art methods on the three benchmarks, RPINE, FSCD-147, and FSCD-LVIS, and demonstrates strong generalization in cross-dataset evaluation. 

**Abstract (ZH)**: 我们提出了一个基于模板匹配和回归的简单而有效的检测器TMR，以解决少样本模式检测问题，该问题旨在从输入图像中检测给定模式的所有实例，这些模式通常由少数几个示例表示。虽然类似问题已经在少样本对象计数和检测（FSCD）中研究过，但之前的 方法和基准往往将兴趣模式局限于对象类别，并且难以定位非对象模式。在本文中，我们提出了一种基于模板匹配和回归的简单而有效的检测器TMR。尽管之前的FSCD方法通常将目标示例表示为空间压缩的原型，从而丢失结构信息，我们重新审视了经典的模板匹配和回归方法，能够通过一个简约结构保留和利用示例的空间布局，该结构在冻结的主干之上仅包含少量的学习卷积或投影层。我们还引入了一个新的数据集RPINE，该数据集涵盖了现有对象为中心的数据集之外的更广泛的模式。我们的方法在RPINE、FSCD-147和FSCD-LVIS三个基准上优于现有方法，并在跨数据集评估中显示出了强大的泛化能力。 

---
# Finding Outliers in a Haystack: Anomaly Detection for Large Pointcloud Scenes 

**Title (ZH)**: 在万堆 haystack 中寻找异常值：大规模点云场景中的异常检测 

**Authors**: Ryan Faulkner, Ian Reid, Simon Ratcliffe, Tat-Jun Chin  

**Link**: [PDF](https://arxiv.org/pdf/2508.17634)  

**Abstract**: LiDAR scanning in outdoor scenes acquires accurate distance measurements over wide areas, producing large-scale point clouds. Application examples for this data include robotics, automotive vehicles, and land surveillance. During such applications, outlier objects from outside the training data will inevitably appear. Our research contributes a novel approach to open-set segmentation, leveraging the learnings of object defect-detection research. We also draw on the Mamba architecture's strong performance in utilising long-range dependencies and scalability to large data. Combining both, we create a reconstruction based approach for the task of outdoor scene open-set segmentation. We show that our approach improves performance not only when applied to our our own open-set segmentation method, but also when applied to existing methods. Furthermore we contribute a Mamba based architecture which is competitive with existing voxel-convolution based methods on challenging, large-scale pointclouds. 

**Abstract (ZH)**: LiDAR 在室外场景中的扫描能够获取大面积内的准确距离测量，生成大规模点云。这种数据的应用例子包括机器人技术、自动驾驶车辆和土地监控。在这些应用中，不可避免地会出现训练数据之外的异常物体。我们的研究提出了一种用于开放集分割的新型方法，借鉴了物体缺陷检测研究的成果。我们还利用Mamba架构在利用长距离依赖性和处理大规模数据方面表现出的强大性能。结合这两种方法，我们提出了一个基于重建的方法，用于室外场景的开放集分割任务。我们展示了我们的方法不仅在我们的开放集分割方法上，而且在现有方法上都能提高性能。此外，我们还贡献了一个基于Mamba架构的模型，在具有挑战性的大规模点云上与基于体素卷积的方法具有竞争力。 

---
# ControlEchoSynth: Boosting Ejection Fraction Estimation Models via Controlled Video Diffusion 

**Title (ZH)**: ControlEchoSynth: 通过受控视频扩散提升射血分数估算模型 

**Authors**: Nima Kondori, Hanwen Liang, Hooman Vaseli, Bingyu Xie, Christina Luong, Purang Abolmaesumi, Teresa Tsang, Renjie Liao  

**Link**: [PDF](https://arxiv.org/pdf/2508.17631)  

**Abstract**: Synthetic data generation represents a significant advancement in boosting the performance of machine learning (ML) models, particularly in fields where data acquisition is challenging, such as echocardiography. The acquisition and labeling of echocardiograms (echo) for heart assessment, crucial in point-of-care ultrasound (POCUS) settings, often encounter limitations due to the restricted number of echo views available, typically captured by operators with varying levels of experience. This study proposes a novel approach for enhancing clinical diagnosis accuracy by synthetically generating echo views. These views are conditioned on existing, real views of the heart, focusing specifically on the estimation of ejection fraction (EF), a critical parameter traditionally measured from biplane apical views. By integrating a conditional generative model, we demonstrate an improvement in EF estimation accuracy, providing a comparative analysis with traditional methods. Preliminary results indicate that our synthetic echoes, when used to augment existing datasets, not only enhance EF estimation but also show potential in advancing the development of more robust, accurate, and clinically relevant ML models. This approach is anticipated to catalyze further research in synthetic data applications, paving the way for innovative solutions in medical imaging diagnostics. 

**Abstract (ZH)**: 合成数据生成代表了增强机器学习模型性能的一项重要进展，特别是在数据获取具有挑战性的领域，如心脏超声。通过提出一种新的方法，合成生成心脏超声视图以增强临床诊断准确性，在受限的心脏超声视图数量和不同经验水平操作者拍摄的心脏超声图像背景下，特别是在点对点心脏超声（POCUS）环境中，需要对射血分数（EF）进行精确估计。通过集成条件生成模型，我们展示了EF估计精度的提升，并与传统方法进行了比较分析。初步结果表明，将我们的合成心脏超声与现有的数据集相结合，不仅能提高EF估计精度，还有潜力促进更稳健、准确和临床相关的机器学习模型的发展。该方法有望推动合成数据应用领域的进一步研究，为医学成像诊断提供创新解决方案。 

---
# Stop Spinning Wheels: Mitigating LLM Overthinking via Mining Patterns for Early Reasoning Exit 

**Title (ZH)**: 停止无谓的循环：通过挖掘早期推理退出模式减轻LLM过度思考 

**Authors**: Zihao Wei, Liang Pang, Jiahao Liu, Jingcheng Deng, Shicheng Xu, Zenghao Duan, Jingang Wang, Fei Sun, Xunliang Cai, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.17627)  

**Abstract**: Large language models (LLMs) enhance complex reasoning tasks by scaling the individual thinking process. However, prior work shows that overthinking can degrade overall performance. Motivated by observed patterns in thinking length and content length, we categorize reasoning into three stages: insufficient exploration stage, compensatory reasoning stage, and reasoning convergence stage. Typically, LLMs produce correct answers in the compensatory reasoning stage, whereas reasoning convergence often triggers overthinking, causing increased resource usage or even infinite loops. Therefore, mitigating overthinking hinges on detecting the end of the compensatory reasoning stage, defined as the Reasoning Completion Point (RCP). RCP typically appears at the end of the first complete reasoning cycle and can be identified by querying the LLM sentence by sentence or monitoring the probability of an end-of-thinking token (e.g., \texttt{</think>}), though these methods lack an efficient and precise balance. To improve this, we mine more sensitive and consistent RCP patterns and develop a lightweight thresholding strategy based on heuristic rules. Experimental evaluations on benchmarks (AIME24, AIME25, GPQA-D) demonstrate that the proposed method reduces token consumption while preserving or enhancing reasoning accuracy. 

**Abstract (ZH)**: 大型语言模型通过扩展个体推理过程来增强复杂推理任务。然而，先前的研究表明，过度推理会降低整体性能。基于观察到的思考长度和内容长度模式，我们将推理划分成三个阶段：初始探索阶段、补偿性推理阶段和推理收敛阶段。通常，LLM在补偿性推理阶段会产生正确的答案，而推理收敛阶段则可能导致过度推理，增加资源使用或导致无限循环。因此，减轻过度推理的关键在于检测补偿性推理阶段的结束，即推理完成点(RCP)。RCP通常出现在第一个完整推理循环的末尾，并可以通过逐句查询LLM或监控结束思考标记（例如，`\</think>`）的概率来识别，尽管这些方法缺乏高效和精确的平衡。为此，我们挖掘了更敏感且一致的RCP模式，并基于启发式规则开发了一种轻量级的阈值策略。在基准测试（AIME24、AIME25、GPQA-D）上的实验评价表明，所提出的方法在保持或提高推理准确性的同时减少了标记消耗。 

---
# Steering When Necessary: Flexible Steering Large Language Models with Backtracking 

**Title (ZH)**: 必要时导向：具有回退机制的灵活大型语言模型导向 

**Authors**: Jinwei Gan, Zifeng Cheng, Zhiwei Jiang, Cong Wang, Yafeng Yin, Xiang Luo, Yuchen Fu, Qing Gu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17621)  

**Abstract**: Large language models (LLMs) have achieved remarkable performance across many generation tasks. Nevertheless, effectively aligning them with desired behaviors remains a significant challenge. Activation steering is an effective and cost-efficient approach that directly modifies the activations of LLMs during the inference stage, aligning their responses with the desired behaviors and avoiding the high cost of fine-tuning. Existing methods typically indiscriminately intervene to all generations or rely solely on the question to determine intervention, which limits the accurate assessment of the intervention strength. To this end, we propose the Flexible Activation Steering with Backtracking (FASB) framework, which dynamically determines both the necessity and strength of intervention by tracking the internal states of the LLMs during generation, considering both the question and the generated content. Since intervening after detecting a deviation from the desired behavior is often too late, we further propose the backtracking mechanism to correct the deviated tokens and steer the LLMs toward the desired behavior. Extensive experiments on the TruthfulQA dataset and six multiple-choice datasets demonstrate that our method outperforms baselines. Our code will be released at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已在许多生成任务中取得了显著性能。然而，将它们有效地与期望的行为对齐仍然是一项重大挑战。激活导向是一种有效且成本效益高的方法，在推理阶段直接修改LLMs的激活，使其响应与期望行为对齐，并避免了细调的高成本。现有方法通常对所有生成结果进行不分青红皂白的干预，或仅依赖问题来决定干预，这限制了干预强度的准确评估。为此，我们提出了可调节激活导向与回溯（FASB）框架，该框架在生成过程中动态确定干预的必要性和强度，同时考虑问题和生成内容。由于检测到与期望行为偏离后再进行干预往往为时已晚，我们进一步提出了回溯机制来纠正偏离的标记，并引导LLMs朝着期望的行为方向发展。在TruthfulQA数据集和六个选择题数据集上的广泛实验表明，我们的方法优于基线方法。我们的代码将发布在该网址：https://。 

---
# GWM: Towards Scalable Gaussian World Models for Robotic Manipulation 

**Title (ZH)**: GWM：面向机器人操作的可扩展高斯世界模型研究 

**Authors**: Guanxing Lu, Baoxiong Jia, Puhao Li, Yixin Chen, Ziwei Wang, Yansong Tang, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17600)  

**Abstract**: Training robot policies within a learned world model is trending due to the inefficiency of real-world interactions. The established image-based world models and policies have shown prior success, but lack robust geometric information that requires consistent spatial and physical understanding of the three-dimensional world, even pre-trained on internet-scale video sources. To this end, we propose a novel branch of world model named Gaussian World Model (GWM) for robotic manipulation, which reconstructs the future state by inferring the propagation of Gaussian primitives under the effect of robot actions. At its core is a latent Diffusion Transformer (DiT) combined with a 3D variational autoencoder, enabling fine-grained scene-level future state reconstruction with Gaussian Splatting. GWM can not only enhance the visual representation for imitation learning agent by self-supervised future prediction training, but can serve as a neural simulator that supports model-based reinforcement learning. Both simulated and real-world experiments depict that GWM can precisely predict future scenes conditioned on diverse robot actions, and can be further utilized to train policies that outperform the state-of-the-art by impressive margins, showcasing the initial data scaling potential of 3D world model. 

**Abstract (ZH)**: 基于学习的世界模型内的机器人策略训练正成为趋势，由于现实世界交互效率低下。现有的基于图像的世界模型和策略已经显示出先前的成功，但缺乏一致的三维空间和物理理解所需的稳健的几何信息，即使是基于互联网规模的视频源进行预训练。为此，我们提出了一种新的世界模型分支——高斯世界模型（GWM），用于机器人操作，通过推断机器人动作影响下的高斯原语传播来重建未来状态。其核心是一个潜在扩散变换器（DiT）结合3D变分自编码器，实现了基于高斯散点图的细粒度场景级未来状态重建。GWM不仅可以通过自我监督的未来预测训练增强仿生学习代理的视觉表示，还可以作为神经模拟器支持基于模型的强化学习。模拟和现实世界实验均表明，GWM可以精确预测多样化机器人动作条件下的未来场景，并且可以进一步用于训练表现超越当前最先进的方法的策略，展示了三维世界模型的初步数据规模化潜力。 

---
# RubikSQL: Lifelong Learning Agentic Knowledge Base as an Industrial NL2SQL System 

**Title (ZH)**: RubikSQL: 作为工业级NL2SQL系统的终身学习代理知识库 

**Authors**: Zui Chen, Han Li, Xinhao Zhang, Xiaoyu Chen, Chunyin Dong, Yifeng Wang, Xin Cai, Su Zhang, Ziqi Li, Chi Ding, Jinxu Li, Shuai Wang, Dousheng Zhao, Sanhai Gao, Guangyi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17590)  

**Abstract**: We present RubikSQL, a novel NL2SQL system designed to address key challenges in real-world enterprise-level NL2SQL, such as implicit intents and domain-specific terminology. RubikSQL frames NL2SQL as a lifelong learning task, demanding both Knowledge Base (KB) maintenance and SQL generation. RubikSQL systematically builds and refines its KB through techniques including database profiling, structured information extraction, agentic rule mining, and Chain-of-Thought (CoT)-enhanced SQL profiling. RubikSQL then employs a multi-agent workflow to leverage this curated KB, generating accurate SQLs. RubikSQL achieves SOTA performance on both the KaggleDBQA and BIRD Mini-Dev datasets. Finally, we release the RubikBench benchmark, a new benchmark specifically designed to capture vital traits of industrial NL2SQL scenarios, providing a valuable resource for future research. 

**Abstract (ZH)**: 我们提出了RubikSQL，一种针对实际企业级NL2SQL中关键挑战（如隐含意图和领域特定术语）的新颖NL2SQL系统。RubikSQL将NL2SQL视为一个终身学习任务，要求同时进行知识库(KB)维护和SQL生成。RubikSQL通过数据库建模、结构化信息提取、代理规则挖掘以及增强思维链（CoT）的SQL建模技术系统地构建和精炼其知识库。然后，RubikSQL采用多智能体工作流利用这一精心构建的知识库生成准确的SQL。RubikSQL在KaggleDBQA和BIRD Mini-Dev数据集上实现了SOTA性能。最后，我们发布了RubikBench基准，这是一种专门设计用于捕获工业NL2SQL场景核心特征的新基准，为未来的研究提供了宝贵的资源。 

---
# UQ: Assessing Language Models on Unsolved Questions 

**Title (ZH)**: UQ：评估语言模型在未解决问题上的表现 

**Authors**: Fan Nie, Ken Ziyu Liu, Zihao Wang, Rui Sun, Wei Liu, Weijia Shi, Huaxiu Yao, Linjun Zhang, Andrew Y. Ng, James Zou, Sanmi Koyejo, Yejin Choi, Percy Liang, Niklas Muennighoff  

**Link**: [PDF](https://arxiv.org/pdf/2508.17580)  

**Abstract**: Benchmarks shape progress in AI research. A useful benchmark should be both difficult and realistic: questions should challenge frontier models while also reflecting real-world usage. Yet, current paradigms face a difficulty-realism tension: exam-style benchmarks are often made artificially difficult with limited real-world value, while benchmarks based on real user interaction often skew toward easy, high-frequency problems. In this work, we explore a radically different paradigm: assessing models on unsolved questions. Rather than a static benchmark scored once, we curate unsolved questions and evaluate models asynchronously over time with validator-assisted screening and community verification. We introduce UQ, a testbed of 500 challenging, diverse questions sourced from Stack Exchange, spanning topics from CS theory and math to sci-fi and history, probing capabilities including reasoning, factuality, and browsing. UQ is difficult and realistic by construction: unsolved questions are often hard and naturally arise when humans seek answers, thus solving them yields direct real-world value. Our contributions are threefold: (1) UQ-Dataset and its collection pipeline combining rule-based filters, LLM judges, and human review to ensure question quality (e.g., well-defined and difficult); (2) UQ-Validators, compound validation strategies that leverage the generator-validator gap to provide evaluation signals and pre-screen candidate solutions for human review; and (3) UQ-Platform, an open platform where experts collectively verify questions and solutions. The top model passes UQ-validation on only 15% of questions, and preliminary human verification has already identified correct answers among those that passed. UQ charts a path for evaluating frontier models on real-world, open-ended challenges, where success pushes the frontier of human knowledge. We release UQ at this https URL. 

**Abstract (ZH)**: 基准设计指引AI研究的进展。一个有用的基准应该是既具有挑战性又具有现实性：问题应挑战前沿模型，同时反映实际应用情况。然而，当前的范式面临难度与现实性的矛盾：考试风格的基准往往通过人为手段变得困难，但缺乏现实价值；基于真实用户交互的基准通常偏向于简单、高频率的问题。在本研究中，我们探索了一种截然不同的范式：在未解决问题上评估模型。我们不仅创建一个静态基准评分一次，而是收集未解决问题，并通过验证者协助筛选和社区验证的方式，在时间上异步评估模型。我们引入了UQ，这是一个包含500个具有挑战性和多样性的测试问题的平台，这些问题来自Stack Exchange，覆盖从计算机科学理论和数学到科幻和历史等多个主题，测试能力包括推理、事实性和浏览。UQ在设计上既具有挑战性又具有现实性：未解决问题通常非常具有挑战性，且自然地在人类寻求答案时产生，因此解决这些问题具有直接的现实价值。我们的贡献有三点：（1）UQ数据集及其收集管道，结合基于规则的过滤器、LLM评判员和人工审查，确保问题质量（如定义清晰和具有挑战性）；（2）UQ验证者，利用生成器-验证者差异的复合验证策略，提供评估信号并预筛选候选解决方案供人工审核；（3）UQ平台，一个开放平台，专家共同验证问题和解决方案。顶级模型仅在15%的问题上通过UQ验证，初步的人工验证已经确定了一些通过验证的问题的正确答案。UQ为评估前沿模型在现实世界、开放挑战上的表现开辟了道路，其成功推动了人类知识的前沿。我们在此发布UQ。 

---
# MetaGen: A DSL, Database, and Benchmark for VLM-Assisted Metamaterial Generation 

**Title (ZH)**: MetaGen: 一种VLM辅助 metamaterial 生成的DSL、数据库和基准测试 

**Authors**: Liane Makatura, Benjamin Jones, Siyuan Bian, Wojciech Matusik  

**Link**: [PDF](https://arxiv.org/pdf/2508.17568)  

**Abstract**: Metamaterials are micro-architected structures whose geometry imparts highly tunable-often counter-intuitive-bulk properties. Yet their design is difficult because of geometric complexity and a non-trivial mapping from architecture to behaviour. We address these challenges with three complementary contributions. (i) MetaDSL: a compact, semantically rich domain-specific language that captures diverse metamaterial designs in a form that is both human-readable and machine-parsable. (ii) MetaDB: a curated repository of more than 150,000 parameterized MetaDSL programs together with their derivatives-three-dimensional geometry, multi-view renderings, and simulated elastic properties. (iii) MetaBench: benchmark suites that test three core capabilities of vision-language metamaterial assistants-structure reconstruction, property-driven inverse design, and performance prediction. We establish baselines by fine-tuning state-of-the-art vision-language models and deploy an omni-model within an interactive, CAD-like interface. Case studies show that our framework provides a strong first step toward integrated design and understanding of structure-representation-property relationships. 

**Abstract (ZH)**: metamaterials是通过几何结构赋予高度可调的常反直觉的大尺度性质的微架构结构。然而，其设计因几何复杂性和从架构到行为的非平凡映射而充满挑战。我们通过三项互补贡献应对这些挑战。(i) MetaDSL：一种紧凑且语义丰富的领域特定语言，以既便于人类阅读又便于机器解析的形式捕获各种metamaterial设计。(ii) MetaDB：一个收录超过150,000个参数化MetaDSL程序及其导数（三维几何结构、多视图渲染和模拟弹性性质）的精心整理数据库。(iii) MetaBench：用于测试视觉-语言metamaterial辅助系统核心能力的基准测试套件——结构重建、性能预测和属性驱动的逆向设计。我们通过微调最先进的视觉-语言模型建立基线，并在一个交互式的、类似CAD的界面中部署了一个全能模型。案例研究表明，我们的框架为结构-表示-属性关系的集成设计和理解提供了坚实的第一步。 

---
# In-Context Algorithm Emulation in Fixed-Weight Transformers 

**Title (ZH)**: 固定权重变压器中的现场算法模拟 

**Authors**: Jerry Yao-Chieh Hu, Hude Liu, Jennifer Yuntong Zhang, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17550)  

**Abstract**: We prove that a minimal Transformer architecture with frozen weights is capable of emulating a broad class of algorithms by in-context prompting. In particular, for any algorithm implementable by a fixed-weight attention head (e.g. one-step gradient descent or linear/ridge regression), there exists a prompt that drives a two-layer softmax attention module to reproduce the algorithm's output with arbitrary precision. This guarantee extends even to a single-head attention layer (using longer prompts if necessary), achieving architectural minimality. Our key idea is to construct prompts that encode an algorithm's parameters into token representations, creating sharp dot-product gaps that force the softmax attention to follow the intended computation. This construction requires no feed-forward layers and no parameter updates. All adaptation happens through the prompt alone. These findings forge a direct link between in-context learning and algorithmic emulation, and offer a simple mechanism for large Transformers to serve as prompt-programmable libraries of algorithms. They illuminate how GPT-style foundation models may swap algorithms via prompts alone, establishing a form of algorithmic universality in modern Transformer models. 

**Abstract (ZH)**: 我们证明，具有冻结权重的最小Transformer架构可以通过上下文内提示来模拟广泛类别的算法。特别是，对于任何可通过固定权重attention头实现的算法（例如，单步梯度下降或线性/岭回归），存在一个提示，可以驱动两层softmax attention模块来以任意精度复制该算法的输出。这一保证甚至扩展到单头attention层（必要时使用更长的提示），实现架构上的最小化。我们的核心思想是构造提示，将算法的参数编码到 token 表征中，从而创建尖锐的点积差距，迫使softmax attention 跟随预期的计算。这一构造无需前馈层且无需参数更新，所有适应仅通过提示完成。这些发现直接链接了上下文学习与算法模拟，并提供了一种简单的机制，使大型Transformer充当可编程算法库。它们揭示了如何通过提示本身在GPT风格的基础模型之间交换算法，确立了现代Transformer模型中的算法普遍性形式。 

---
# LodeStar: Long-horizon Dexterity via Synthetic Data Augmentation from Human Demonstrations 

**Title (ZH)**: LodeStar: 长期灵巧操作通过合成数据增强的人类示范 

**Authors**: Weikang Wan, Jiawei Fu, Xiaodi Yuan, Yifeng Zhu, Hao Su  

**Link**: [PDF](https://arxiv.org/pdf/2508.17547)  

**Abstract**: Developing robotic systems capable of robustly executing long-horizon manipulation tasks with human-level dexterity is challenging, as such tasks require both physical dexterity and seamless sequencing of manipulation skills while robustly handling environment variations. While imitation learning offers a promising approach, acquiring comprehensive datasets is resource-intensive. In this work, we propose a learning framework and system LodeStar that automatically decomposes task demonstrations into semantically meaningful skills using off-the-shelf foundation models, and generates diverse synthetic demonstration datasets from a few human demos through reinforcement learning. These sim-augmented datasets enable robust skill training, with a Skill Routing Transformer (SRT) policy effectively chaining the learned skills together to execute complex long-horizon manipulation tasks. Experimental evaluations on three challenging real-world long-horizon dexterous manipulation tasks demonstrate that our approach significantly improves task performance and robustness compared to previous baselines. Videos are available at this http URL. 

**Abstract (ZH)**: 开发能够在长时段操作任务中以人类级灵巧性 robust 地执行操作的机器人系统具有挑战性，因为这样的任务不仅需要物理灵巧性，还需要在处理环境变化的同时无缝地组合操作技能。虽然模仿学习提供了一种有前景的方法，但获得全面的数据集是资源密集型的。在这项工作中，我们提出了一种学习框架和系统 LodeStar，该系统使用现成的基础模型自动将任务示例分解为语义上有意义的技能，并通过强化学习从少量的人类示例中生成多样化的合成示例数据集。这些基于模拟的数据集使技能训练更加 robust，Skill Routing Transformer (SRT) 策略能够有效地将学习到的技能串联起来执行复杂的长时段操作任务。在三个具有挑战性的现实世界的长时段灵巧操作任务上的实验评估表明，与先前的方法相比，我们的方法显着提高了任务性能和 robust 性。视频可在以下网址获取。 

---
# Activation Transport Operators 

**Title (ZH)**: 激活传输算子 

**Authors**: Andrzej Szablewski, Marek Masiak  

**Link**: [PDF](https://arxiv.org/pdf/2508.17540)  

**Abstract**: The residual stream mediates communication between transformer decoder layers via linear reads and writes of non-linear computations. While sparse-dictionary learning-based methods locate features in the residual stream, and activation patching methods discover circuits within the model, the mechanism by which features flow through the residual stream remains understudied. Understanding this dynamic can better inform jailbreaking protections, enable early detection of model mistakes, and their correction. In this work, we propose Activation Transport Operators (ATO), linear maps from upstream to downstream residuals $k$ layers later, evaluated in feature space using downstream SAE decoder projections. We empirically demonstrate that these operators can determine whether a feature has been linearly transported from a previous layer or synthesised from non-linear layer computation. We develop the notion of transport efficiency, for which we provide an upper bound, and use it to estimate the size of the residual stream subspace that corresponds to linear transport. We empirically demonstrate the linear transport, report transport efficiency and the size of the residual stream's subspace involved in linear transport. This compute-light (no finetuning, <50 GPU-h) method offers practical tools for safety, debugging, and a clearer picture of where computation in LLMs behaves linearly. 

**Abstract (ZH)**: 激活传输运算符介导残差流中的信息传递：从非线性计算的线性读写到变压器解码层的通信 

---
# OmniMRI: A Unified Vision--Language Foundation Model for Generalist MRI Interpretation 

**Title (ZH)**: OmniMRI：统一的视觉-语言基础模型，用于通用MRI解释 

**Authors**: Xingxin He, Aurora Rofena, Ruimin Feng, Haozhe Liao, Zhaoye Zhou, Albert Jang, Fang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17524)  

**Abstract**: Magnetic Resonance Imaging (MRI) is indispensable in clinical practice but remains constrained by fragmented, multi-stage workflows encompassing acquisition, reconstruction, segmentation, detection, diagnosis, and reporting. While deep learning has achieved progress in individual tasks, existing approaches are often anatomy- or application-specific and lack generalizability across diverse clinical settings. Moreover, current pipelines rarely integrate imaging data with complementary language information that radiologists rely on in routine practice. Here, we introduce OmniMRI, a unified vision-language foundation model designed to generalize across the entire MRI workflow. OmniMRI is trained on a large-scale, heterogeneous corpus curated from 60 public datasets, over 220,000 MRI volumes and 19 million MRI slices, incorporating image-only data, paired vision-text data, and instruction-response data. Its multi-stage training paradigm, comprising self-supervised vision pretraining, vision-language alignment, multimodal pretraining, and multi-task instruction tuning, progressively equips the model with transferable visual representations, cross-modal reasoning, and robust instruction-following capabilities. Qualitative results demonstrate OmniMRI's ability to perform diverse tasks within a single architecture, including MRI reconstruction, anatomical and pathological segmentation, abnormality detection, diagnostic suggestion, and radiology report generation. These findings highlight OmniMRI's potential to consolidate fragmented pipelines into a scalable, generalist framework, paving the way toward foundation models that unify imaging and clinical language for comprehensive, end-to-end MRI interpretation. 

**Abstract (ZH)**: 全面MRI：统一的视觉-语言基础模型在整个MRI工作流程中的泛化应用 

---
# An experimental approach: The graph of graphs 

**Title (ZH)**: 基于实验的方法：图的图谱 

**Authors**: Zsombor Szádoczki, Sándor Bozóki, László Sipos, Zsófia Galambosi  

**Link**: [PDF](https://arxiv.org/pdf/2508.17520)  

**Abstract**: One of the essential issues in decision problems and preference modeling is the number of comparisons and their pattern to ask from the decision maker. We focus on the optimal patterns of pairwise comparisons and the sequence including the most (close to) optimal cases based on the results of a color selection experiment. In the test, six colors (red, green, blue, magenta, turquoise, yellow) were evaluated with pairwise comparisons as well as in a direct manner, on color-calibrated tablets in ISO standardized sensory test booths of a sensory laboratory. All the possible patterns of comparisons resulting in a connected representing graph were evaluated against the complete data based on 301 individual's pairwise comparison matrices (PCMs) using the logarithmic least squares weight calculation technique. It is shown that the empirical results, i.e., the empirical distributions of the elements of PCMs, are quite similar to the former simulated outcomes from the literature. The obtained empirically optimal patterns of comparisons were the best or the second best in the former simulations as well, while the sequence of comparisons that contains the most (close to) optimal patterns is exactly the same. In order to enhance the applicability of the results, besides the presentation of graph of graphs, and the representing graphs of the patterns that describe the proposed sequence of comparisons themselves, the recommendations are also detailed in a table format as well as in a Java application. 

**Abstract (ZH)**: 决策问题和偏好建模中一个关键问题是如何优化询问决策者配对比较的数量及其模式。基于颜色选取实验的结果，我们关注最优的配对比较模式以及包括最多（接近）最优情况的序列。在实验中，六种颜色（红色、绿色、蓝色、品红色、青色、黄色）通过配对比较和直接评价，在ISO标准化感官测试舱的色彩校准平板上进行了评估。所有生成连通表示图的所有可能的比较模式均使用基于301名个体的配对比较矩阵（PCMs）和对数最小二乘权重计算技术进行了评估。实验结果显示，实际情况中配对比较矩阵元素的经验分布与文献中的仿真实验结果非常相似。获得的经验最优比较模式在之前的仿真实验中均为最佳或次最佳，而包含最多（接近）最优模式的比较序列也完全相同。为了提高结果的应用性，除了展示图与图的关系图和描述所提比较序列的表示图外，还在表格和Java应用程序中详细提供了建议。 

---
# TANDEM: Temporal Attention-guided Neural Differential Equations for Missingness in Time Series Classification 

**Title (ZH)**: TANDEM: 时空注意力引导的神经微分方程模型用于时间序列分类 

**Authors**: YongKyung Oh, Dong-Young Lim, Sungil Kim, Alex Bui  

**Link**: [PDF](https://arxiv.org/pdf/2508.17519)  

**Abstract**: Handling missing data in time series classification remains a significant challenge in various domains. Traditional methods often rely on imputation, which may introduce bias or fail to capture the underlying temporal dynamics. In this paper, we propose TANDEM (Temporal Attention-guided Neural Differential Equations for Missingness), an attention-guided neural differential equation framework that effectively classifies time series data with missing values. Our approach integrates raw observation, interpolated control path, and continuous latent dynamics through a novel attention mechanism, allowing the model to focus on the most informative aspects of the data. We evaluate TANDEM on 30 benchmark datasets and a real-world medical dataset, demonstrating its superiority over existing state-of-the-art methods. Our framework not only improves classification accuracy but also provides insights into the handling of missing data, making it a valuable tool in practice. 

**Abstract (ZH)**: 时间序列分类中缺失数据的处理仍然是各个领域的一项重大挑战。传统方法通常依赖于插补，这可能会引入偏差或无法捕捉潜在的时间动态。在本文中，我们提出了一种名为TANDEM（Temporal Attention-guided Neural Differential Equations for Missingness）的方法，这是一种通过新颖的注意力机制整合原始观测、插补控制路径和连续潜在动态的神经微分方程框架，有效分类带有缺失值的时间序列数据。我们在30个基准数据集和一个真实世界的医疗数据集上评估了TANDEM，展示了其在分类准确性上的优越性，并提供了对缺失数据处理的见解，使其成为实践中的有力工具。 

---
# DinoTwins: Combining DINO and Barlow Twins for Robust, Label-Efficient Vision Transformers 

**Title (ZH)**: DinoTwins：结合DINO和Barlow Twins的鲁棒高效视觉变换器 

**Authors**: Michael Podsiadly, Brendon K Lay  

**Link**: [PDF](https://arxiv.org/pdf/2508.17509)  

**Abstract**: Training AI models to understand images without costly labeled data remains a challenge. We combine two techniques--DINO (teacher-student learning) and Barlow Twins (redundancy reduction)--to create a model that learns better with fewer labels and less compute. While both DINO and Barlow Twins have independently demonstrated strong performance in self-supervised learning, each comes with limitations--DINO may be sensitive to certain augmentations, and Barlow Twins often requires batch sizes too large to fit on consumer hardware. By combining the redundancy-reduction objective of Barlow Twins with the self-distillation strategy of DINO, we aim to leverage their complementary strengths. We train a hybrid model on the MS COCO dataset using only 10\% of labeled data for linear probing, and evaluate its performance against standalone DINO and Barlow Twins implementations. Preliminary results show that the combined approach achieves comparable loss and classification accuracy to DINO while maintaining strong feature representations. Attention visualizations further suggest improved semantic segmentation capability in the hybrid model. This combined method offers a scalable, label-efficient alternative for training ViTs in resource-constrained environments. 

**Abstract (ZH)**: 不依赖昂贵标注数据训练AI模型以理解图像仍然是一个挑战。我们结合了两种技术——DINO（教师-学生学习）和Barlow Twins（冗余减少）——来创建一种在更少标签和更少计算资源下学习更好的模型。尽管DINO和Barlow Twins各自在无监督学习中表现出强大性能，但两者都存在局限性——DINO可能对某些增强操作敏感，而Barlow Twins通常需要批量大小过大以至于无法在消费级硬件上运行。通过将Barlow Twins的冗余减少目标与DINO的自我_distillation_策略相结合，我们旨在利用它们互补的优势。我们在MS COCO数据集上训练了一种混合模型，并仅使用10%的标注数据进行线性探针评估，将其性能与独立实现的DINO和Barlow Twins进行了比较。初步结果表明，组合方法在损失和分类准确性方面可与DINO媲美，同时保持强大的特征表示能力。注意力可视化进一步表明混合模型在语义分割方面表现出改进的能力。该结合方法为资源受限环境中训练ViTs提供了可扩展且标签高效的替代方案。 

---
# Multimodal Representation Learning Conditioned on Semantic Relations 

**Title (ZH)**: 基于语义关系的多模态表示学习 

**Authors**: Yang Qiao, Yuntong Hu, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.17497)  

**Abstract**: Multimodal representation learning has advanced rapidly with contrastive models such as CLIP, which align image-text pairs in a shared embedding space. However, these models face limitations: (1) they typically focus on image-text pairs, underutilizing the semantic relations across different pairs. (2) they directly match global embeddings without contextualization, overlooking the need for semantic alignment along specific subspaces or relational dimensions; and (3) they emphasize cross-modal contrast, with limited support for intra-modal consistency. To address these issues, we propose Relation-Conditioned Multimodal Learning RCML, a framework that learns multimodal representations under natural-language relation descriptions to guide both feature extraction and alignment. Our approach constructs many-to-many training pairs linked by semantic relations and introduces a relation-guided cross-attention mechanism that modulates multimodal representations under each relation context. The training objective combines inter-modal and intra-modal contrastive losses, encouraging consistency across both modalities and semantically related samples. Experiments on different datasets show that RCML consistently outperforms strong baselines on both retrieval and classification tasks, highlighting the effectiveness of leveraging semantic relations to guide multimodal representation learning. 

**Abstract (ZH)**: 基于语义关系条件的多模态学习RCML 

---
# A Synthetic Dataset for Manometry Recognition in Robotic Applications 

**Title (ZH)**: 一种用于机器人应用的食道测压识别合成数据集 

**Authors**: Pedro Antonio Rabelo Saraiva, Enzo Ferreira de Souza, Joao Manoel Herrera Pinheiro, Thiago H. Segreto, Ricardo V. Godoy, Marcelo Becker  

**Link**: [PDF](https://arxiv.org/pdf/2508.17468)  

**Abstract**: This work addresses the challenges of data scarcity and high acquisition costs for training robust object detection models in complex industrial environments, such as offshore oil platforms. The practical and economic barriers to collecting real-world data in these hazardous settings often hamper the development of autonomous inspection systems. To overcome this, in this work we propose and validate a hybrid data synthesis pipeline that combines procedural rendering with AI-driven video generation. Our methodology leverages BlenderProc to create photorealistic images with precise annotations and controlled domain randomization, and integrates NVIDIA's Cosmos-Predict2 world-foundation model to synthesize physically plausible video sequences with temporal diversity, capturing rare viewpoints and adverse conditions. We demonstrate that a YOLO-based detection network trained on a composite dataset, blending real images with our synthetic data, achieves superior performance compared to models trained exclusively on real-world data. Notably, a 1:1 mixture of real and synthetic data yielded the highest accuracy, surpassing the real-only baseline. These findings highlight the viability of a synthetic-first approach as an efficient, cost-effective, and safe alternative for developing reliable perception systems in safety-critical and resource-constrained industrial applications. 

**Abstract (ZH)**: 本研究解决了在复杂工业环境中，如 offshore 油平台，训练鲁棒对象检测模型时面临的数据稀缺和高昂采集成本的挑战。在这些危险环境中收集真实世界数据的实际和经济障碍往往阻碍了自主检测系统的开发。为克服这一难题，我们在此工作中提出并验证了一种结合过程渲染和AI驱动视频生成的混合数据合成管道。我们的方法利用BlenderProc创建具有精确标注的 photorealistic 图像，并采用控制域随机化，同时结合 NVIDIA 的 Cosmos-Predict2 世界基础模型，生成具有时间多样性且物理上合理的视频序列，捕捉到罕见视角和不利条件。我们证明，使用综合数据集训练的基于 YOLO 的检测网络，该数据集结合了真实图像和我们合成的数据，相对于仅使用真实世界数据训练的模型实现了更优性能。值得注意的是，真实数据与合成数据各占一半的混合数据集达到了最高的准确率，超越了仅使用真实数据的基线。这些发现突显了合成数据优先方法在安全关键和资源受限的工业应用中作为高效、经济和安全替代方案的可行性。 

---
# Optimizing Grasping in Legged Robots: A Deep Learning Approach to Loco-Manipulation 

**Title (ZH)**: 基于深度学习的腿部机器人抓取优化：动作- manipulation学习 

**Authors**: Dilermando Almeida, Guilherme Lazzarini, Juliano Negri, Thiago H. Segreto, Ricardo V. Godoy, Marcelo Becker  

**Link**: [PDF](https://arxiv.org/pdf/2508.17466)  

**Abstract**: Quadruped robots have emerged as highly efficient and versatile platforms, excelling in navigating complex and unstructured terrains where traditional wheeled robots might fail. Equipping these robots with manipulator arms unlocks the advanced capability of loco-manipulation to perform complex physical interaction tasks in areas ranging from industrial automation to search-and-rescue missions. However, achieving precise and adaptable grasping in such dynamic scenarios remains a significant challenge, often hindered by the need for extensive real-world calibration and pre-programmed grasp configurations. This paper introduces a deep learning framework designed to enhance the grasping capabilities of quadrupeds equipped with arms, focusing on improved precision and adaptability. Our approach centers on a sim-to-real methodology that minimizes reliance on physical data collection. We developed a pipeline within the Genesis simulation environment to generate a synthetic dataset of grasp attempts on common objects. By simulating thousands of interactions from various perspectives, we created pixel-wise annotated grasp-quality maps to serve as the ground truth for our model. This dataset was used to train a custom CNN with a U-Net-like architecture that processes multi-modal input from an onboard RGB and depth cameras, including RGB images, depth maps, segmentation masks, and surface normal maps. The trained model outputs a grasp-quality heatmap to identify the optimal grasp point. We validated the complete framework on a four-legged robot. The system successfully executed a full loco-manipulation task: autonomously navigating to a target object, perceiving it with its sensors, predicting the optimal grasp pose using our model, and performing a precise grasp. This work proves that leveraging simulated training with advanced sensing offers a scalable and effective solution for object handling. 

**Abstract (ZH)**: 四足机器人的带 manipulator 的精准和适应性抓取：一种基于模拟的深度学习方法 

---
# Bias Amplification in Stable Diffusion's Representation of Stigma Through Skin Tones and Their Homogeneity 

**Title (ZH)**: 稳定扩散在肤色及其均一性中对污名的表征中的偏差放大 

**Authors**: Kyra Wilson, Sourojit Ghosh, Aylin Caliskan  

**Link**: [PDF](https://arxiv.org/pdf/2508.17465)  

**Abstract**: Text-to-image generators (T2Is) are liable to produce images that perpetuate social stereotypes, especially in regards to race or skin tone. We use a comprehensive set of 93 stigmatized identities to determine that three versions of Stable Diffusion (v1.5, v2.1, and XL) systematically associate stigmatized identities with certain skin tones in generated images. We find that SD XL produces skin tones that are 13.53% darker and 23.76% less red (both of which indicate higher likelihood of societal discrimination) than previous models and perpetuate societal stereotypes associating people of color with stigmatized identities. SD XL also shows approximately 30% less variability in skin tones when compared to previous models and 18.89-56.06% compared to human face datasets. Measuring variability through metrics which directly correspond to human perception suggest a similar pattern, where SD XL shows the least amount of variability in skin tones of people with stigmatized identities and depicts most (60.29%) stigmatized identities as being less diverse than non-stigmatized identities. Finally, SD shows more homogenization of skin tones of racial and ethnic identities compared to other stigmatized or non-stigmatized identities, reinforcing incorrect equivalence of biologically-determined skin tone and socially-constructed racial and ethnic identity. Because SD XL is the largest and most complex model and users prefer its generations compared to other models examined in this study, these findings have implications for the dynamics of bias amplification in T2Is, increasing representational harms and challenges generating diverse images depicting people with stigmatized identities. 

**Abstract (ZH)**: 文本到图像生成器（T2Is）可能生成 perpetuate 社会刻板印象的图像，尤其是在种族或肤色方面。我们使用93种受 stigma 影响的身份来确定三个版本的 Stable Diffusion（v1.5、v2.1 和 XL）系统地将这些 stigma 身份与某些肤色联系起来。我们发现，SD XL 生成的肤色比之前模型暗 13.53%、红 23.76% 较少（这两种情况都表明社会歧视的可能性更大），从而延续了有色人种与 stigma 身份相关联的社会刻板印象。SD XL 在肤色变化方面也比之前模型大约少 30%，并比人类面部数据集少 18.89%-56.06%。通过直接与人类感知相关的指标来测量肤色变化表明，具有 stigma 身份的人的肤色变化最少，其中最典型的（60.29%）stigma 身份被描绘得比非 stigma 身份更不多样化。最后，SD 比其他 stigma 身份或非 stigma 身份更 homogenize 肤色，强化了生物决定的肤色与社会建构的种族和族裔身份之间的错误等同。由于 SD XL 是最大的和最复杂的模型，并且用户更偏好其生成结果，这些发现对于 T2Is 中偏差放大的动态、增加的表现性伤害以及生成多样化图像的挑战具有重要意义。 

---
# FedKLPR: Personalized Federated Learning for Person Re-Identification with Adaptive Pruning 

**Title (ZH)**: FedKLPR：带有自适应剪枝的个性化联邦学习在行人重识别中的应用 

**Authors**: Po-Hsien Yu, Yu-Syuan Tseng, Shao-Yi Chien  

**Link**: [PDF](https://arxiv.org/pdf/2508.17431)  

**Abstract**: Person re-identification (Re-ID) is a fundamental task in intelligent surveillance and public safety. Federated learning (FL) offers a privacy-preserving solution by enabling collaborative model training without centralized data collection. However, applying FL to real-world re-ID systems faces two major challenges: statistical heterogeneity across clients due to non-IID data distributions, and substantial communication overhead caused by frequent transmission of large-scale models. To address these issues, we propose FedKLPR, a lightweight and communication-efficient federated learning framework for person re-identification. FedKLPR introduces four key components. First, the KL-Divergence Regularization Loss (KLL) constrains local models by minimizing the divergence from the global feature distribution, effectively mitigating the effects of statistical heterogeneity and improving convergence stability under non-IID conditions. Secondly, KL-Divergence-Prune Weighted Aggregation (KLPWA) integrates pruning ratio and distributional similarity into the aggregation process, thereby improving the robustness of the global model while significantly reducing communication overhead. Furthermore, sparse Activation Skipping (SAS) mitigates the dilution of critical parameters during the aggregation of pruned client models by excluding zero-valued weights from the update process. Finally, Cross-Round Recovery (CRR) introduces a dynamic pruning control mechanism that halts pruning when necessary, enabling deeper compression while maintaining model accuracy. Experimental results on eight benchmark datasets demonstrate that FedKLPR achieves significant communication reduction. Compared with the state-of-the-art, FedKLPR reduces 33\%-38\% communication cost on ResNet-50 and 20\%-40\% communication cost on ResNet-34, while maintaining model accuracy within 1\% degradation. 

**Abstract (ZH)**: 联邦学习框架FedKLPR：针对人员再识别的轻量级和通信高效方法 

---
# Convergence and Generalization of Anti-Regularization for Parametric Models 

**Title (ZH)**: 参数模型中的反正则化收敛性和泛化性分析 

**Authors**: Dongseok Kim, Wonjun Jeong, Gisung Oh  

**Link**: [PDF](https://arxiv.org/pdf/2508.17412)  

**Abstract**: We propose Anti-regularization (AR), which adds a sign-reversed reward term to the loss to intentionally increase model expressivity in the small-sample regime, and then attenuates this intervention with a power-law decay as the sample size grows. We formalize spectral safety and trust-region conditions, and design a lightweight stability safeguard that combines a projection operator with gradient clipping, ensuring stable intervention under stated assumptions. Our analysis spans linear smoothers and the Neural Tangent Kernel (NTK) regime, providing practical guidance on selecting the decay exponent by balancing empirical risk against variance. Empirically, AR reduces underfitting while preserving generalization and improving calibration in both regression and classification. Ablation studies confirm that the decay schedule and the stability safeguard are critical to preventing overfitting and numerical instability. We further examine a degrees-of-freedom targeting schedule that keeps per-sample complexity approximately constant. AR is simple to implement and reproducible, integrating cleanly into standard empirical risk minimization pipelines. It enables robust learning in data- and resource-constrained settings by intervening only when beneficial and fading away when unnecessary. 

**Abstract (ZH)**: 我们提出反正则化(AR)，该方法在损失中添加了一个符号反转的奖励项，以故意增加小样本情况下的模型表达性，然后随着样本数量的增长，通过幂律衰减来减弱这种干预。我们形式化了光谱安全性与信赖域条件，并设计了一种轻量级的稳定性保障措施，结合投影算子与梯度剪裁，确保在满足特定假设时模型干预的稳定性。我们的分析跨越了线性平滑器和神经 tangent 核 (NTK) 状态，提供了平衡经验风险与方差以选择衰减指数的实用指导。实验证明，AR 可以减少欠拟合现象，同时保持泛化能力和提高校准性，在回归和分类任务中均表现良好。消融实验表明，衰减计划和稳定性保障措施对于防止过拟合和数值不稳定性至关重要。我们还研究了一种自由度目标计划，以保持每样本复杂度相对恒定。AR 实现简单且可再现，能够无缝集成到标准的经验风险最小化流水线中。它能够在数据和资源受限的环境中实现稳健学习，仅在有益时进行干预，而在不需要时逐渐减弱。 

---
# Retrieval Capabilities of Large Language Models Scale with Pretraining FLOPs 

**Title (ZH)**: 大型语言模型的检索能力随预训练FLOPs增长 

**Authors**: Jacob Portes, Connor Jennings, Erica Ji Yuen, Sasha Doubov, Michael Carbin  

**Link**: [PDF](https://arxiv.org/pdf/2508.17400)  

**Abstract**: How does retrieval performance scale with pretraining FLOPs? We benchmark retrieval performance across LLM model sizes from 125 million parameters to 7 billion parameters pretrained on datasets ranging from 1 billion tokens to more than 2 trillion tokens. We find that retrieval performance on zero-shot BEIR tasks predictably scales with LLM size, training duration, and estimated FLOPs. We also show that In-Context Learning scores are strongly correlated with retrieval scores across retrieval tasks. Finally, we highlight the implications this has for the development of LLM-based retrievers. 

**Abstract (ZH)**: 预训练FLOPs如何影响检索性能的扩展？我们评估了从1.25亿参数到70亿参数不等的LLM模型大小在不同大小数据集（从10亿令牌到超过2兆亿令牌）上预训练的检索性能。我们发现，在零样本BEIR任务上的检索性能可预测地与LLM规模、训练时长和估计的FLOPs相关。我们还展示了语境中学习得分与检索任务的检索得分之间存在强烈的相关性。最后，我们强调了这对基于LLM的检索器发展的影响。 

---
# Agent-Testing Agent: A Meta-Agent for Automated Testing and Evaluation of Conversational AI Agents 

**Title (ZH)**: 代理测试代理：一种用于对话式AI代理的自动化测试与评估元代理 

**Authors**: Sameer Komoravolu, Khalil Mrini  

**Link**: [PDF](https://arxiv.org/pdf/2508.17393)  

**Abstract**: LLM agents are increasingly deployed to plan, retrieve, and write with tools, yet evaluation still leans on static benchmarks and small human studies. We present the Agent-Testing Agent (ATA), a meta-agent that combines static code analysis, designer interrogation, literature mining, and persona-driven adversarial test generation whose difficulty adapts via judge feedback. Each dialogue is scored with an LLM-as-a-Judge (LAAJ) rubric and used to steer subsequent tests toward the agent's weakest capabilities. On a travel planner and a Wikipedia writer, the ATA surfaces more diverse and severe failures than expert annotators while matching severity, and finishes in 20--30 minutes versus ten-annotator rounds that took days. Ablating code analysis and web search increases variance and miscalibration, underscoring the value of evidence-grounded test generation. The ATA outputs quantitative metrics and qualitative bug reports for developers. We release the full methodology and open-source implementation for reproducible agent testing: this https URL 

**Abstract (ZH)**: LLM代理日益用于规划、检索和撰写，但评估仍依赖于静态基准和小型的人类研究。我们提出了代理测试代理（ATA），这是一种结合静态代码分析、设计师问询、文献挖掘及基于人设的对抗性测试生成的元代理，测试难度通过评审反馈进行调整。每次对话使用LLM作为评审（LAAJ）评分标准，并据此引导后续测试以强化代理的薄弱能力。在旅行规划器和维基百科撰稿人实验中，ATA揭示了比专家标注器更多的多样性和严重性失败，并且测评为20-30分钟，而专家标注需要几天时间的十轮标注。去除了代码分析和网络搜索会导致结果变异性增加和校准不足，突显了基于证据的测试生成的价值。ATA输出定量指标和定性错误报告供开发者使用。我们发布了完整的测试方法和开源实现以实现可重复的代理测试：[此链接]。 

---
# Neural Proteomics Fields for Super-resolved Spatial Proteomics Prediction 

**Title (ZH)**: 神经蛋白质组学领域用于超分辨空间蛋白质组学预测 

**Authors**: Bokai Zhao, Weiyang Shi, Hanqing Chao, Zijiang Yang, Yiyang Zhang, Ming Song, Tianzi Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17389)  

**Abstract**: Spatial proteomics maps protein distributions in tissues, providing transformative insights for life sciences. However, current sequencing-based technologies suffer from low spatial resolution, and substantial inter-tissue variability in protein expression further compromises the performance of existing molecular data prediction methods. In this work, we introduce the novel task of spatial super-resolution for sequencing-based spatial proteomics (seq-SP) and, to the best of our knowledge, propose the first deep learning model for this task--Neural Proteomics Fields (NPF). NPF formulates seq-SP as a protein reconstruction problem in continuous space by training a dedicated network for each tissue. The model comprises a Spatial Modeling Module, which learns tissue-specific protein spatial distributions, and a Morphology Modeling Module, which extracts tissue-specific morphological features. Furthermore, to facilitate rigorous evaluation, we establish an open-source benchmark dataset, Pseudo-Visium SP, for this task. Experimental results demonstrate that NPF achieves state-of-the-art performance with fewer learnable parameters, underscoring its potential for advancing spatial proteomics research. Our code and dataset are publicly available at this https URL. 

**Abstract (ZH)**: 空间蛋白质组学映射组织中的蛋白质分布，为生命科学提供了变革性的洞察。然而，现有的基于测序的空间蛋白质组学技术具有较低的空间分辨率，并且组织间蛋白质表达的显著差异进一步降低了现有分子数据分析方法的性能。为解决这一问题，我们提出了空间超分辨率任务，即基于测序的空间蛋白质组学(seq-SP)，并提出了首个针对此任务的深度学习模型——神经蛋白质场(NPF)模型。NPF通过为每个组织训练一个专门的网络，将seq-SP公式化为连续空间中的蛋白质重建问题。模型包括一个空间建模模块，该模块学习组织特异性蛋白质的空间分布，以及一个形态学建模模块，该模块提取组织特异性形态特征。此外，为了方便严格评估，我们建立了开源基准数据集Pseudo-Visium SP。实验结果表明，NPF在更少的可学习参数下实现了最先进的性能，证明了其在推动空间蛋白质组学研究方面潜在的重要性。我们的代码和数据集在该网址公开：[this https URL]。 

---
# Graph-R1: Incentivizing the Zero-Shot Graph Learning Capability in LLMs via Explicit Reasoning 

**Title (ZH)**: Graph-R1：通过显式推理激励LLMs的零样本图学习能力 

**Authors**: Yicong Wu, Guangyue Lu, Yuan Zuo, Huarong Zhang, Junjie Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17387)  

**Abstract**: Generalizing to unseen graph tasks without task-pecific supervision remains challenging. Graph Neural Networks (GNNs) are limited by fixed label spaces, while Large Language Models (LLMs) lack structural inductive biases. Recent advances in Large Reasoning Models (LRMs) provide a zero-shot alternative via explicit, long chain-of-thought reasoning. Inspired by this, we propose a GNN-free approach that reformulates graph tasks--node classification, link prediction, and graph classification--as textual reasoning problems solved by LRMs. We introduce the first datasets with detailed reasoning traces for these tasks and develop Graph-R1, a reinforcement learning framework that leverages task-specific rethink templates to guide reasoning over linearized graphs. Experiments demonstrate that Graph-R1 outperforms state-of-the-art baselines in zero-shot settings, producing interpretable and effective predictions. Our work highlights the promise of explicit reasoning for graph learning and provides new resources for future research. 

**Abstract (ZH)**: 在没有特定任务监督的情况下将图任务泛化到未见过的任务仍然具有挑战性。图神经网络（GNNs）受限于固定的标签空间，而大规模语言模型（LLMs）缺乏结构归纳偏见。大规模推理模型（LRMs）的 recent 进展通过显式的长推理链提供了零样本的替代方案。受此启发，我们提出了一种无需图神经网络的方法，将图任务——节点分类、链接预测和图分类——重新表述为由LRMs解决的文本推理问题。我们首次提出了包含这些任务详细推理踪迹的数据集，并开发了Graph-R1，一种利用特定任务重思考模板的强化学习框架，以指导对线性化图的推理。实验表明，Graph-R1 在零样本设置中优于最先进的基线方法，产生了可解释且有效的问题解决方法。我们的工作突显了显式推理在图学习中的潜力，并为未来的研究提供了新的资源。 

---
# Condition Weaving Meets Expert Modulation: Towards Universal and Controllable Image Generation 

**Title (ZH)**: 条件织造结合专家调制：通往普遍可控图像生成的道路 

**Authors**: Guoqing Zhang, Xingtong Ge, Lu Shi, Xin Zhang, Muqing Xue, Wanru Xu, Yigang Cen  

**Link**: [PDF](https://arxiv.org/pdf/2508.17364)  

**Abstract**: The image-to-image generation task aims to produce controllable images by leveraging conditional inputs and prompt instructions. However, existing methods often train separate control branches for each type of condition, leading to redundant model structures and inefficient use of computational resources. To address this, we propose a Unified image-to-image Generation (UniGen) framework that supports diverse conditional inputs while enhancing generation efficiency and expressiveness. Specifically, to tackle the widely existing parameter redundancy and computational inefficiency in controllable conditional generation architectures, we propose the Condition Modulated Expert (CoMoE) module. This module aggregates semantically similar patch features and assigns them to dedicated expert modules for visual representation and conditional modeling. By enabling independent modeling of foreground features under different conditions, CoMoE effectively mitigates feature entanglement and redundant computation in multi-condition scenarios. Furthermore, to bridge the information gap between the backbone and control branches, we propose WeaveNet, a dynamic, snake-like connection mechanism that enables effective interaction between global text-level control from the backbone and fine-grained control from conditional branches. Extensive experiments on the Subjects-200K and MultiGen-20M datasets across various conditional image generation tasks demonstrate that our method consistently achieves state-of-the-art performance, validating its advantages in both versatility and effectiveness. The code has been uploaded to this https URL. 

**Abstract (ZH)**: 图像到图像生成任务旨在通过利用条件输入和提示指令产生可控的图像。然而，现有方法通常为每种类型的条件训练单独的控制分支，导致模型结构冗余和计算资源的低效利用。为此，我们提出了一种统一的图像到图像生成（UniGen）框架，该框架支持多种条件输入并提高生成效率和表现力。具体而言，为了解决可控条件生成架构中普遍存在的参数冗余和计算低效问题，我们提出了一种条件调制专家（CoMoE）模块。该模块聚合语义相似的patches特征，并将它们分配给专门的专家模块进行视觉表示和条件建模。通过在不同条件下独立建模前景特征，CoMoE有效减轻了多条件场景中的特征纠缠和冗余计算。为进一步解决主干与控制分支之间信息鸿沟的问题，我们提出了WeaveNet，一种动态、蛇形连接机制，能够有效地在主干的全局文本级控制和条件分支的精细控制之间建立交互。在各种条件图像生成任务的Subjects-200K和MultiGen-20M数据集上的广泛实验表明，我们的方法在一致性和有效性方面都取得了最先进的性能。代码已上传至https://github.com/Qwen-x/UniGen。 

---
# The Arabic Generality Score: Another Dimension of Modeling Arabic Dialectness 

**Title (ZH)**: 阿拉伯通用性评分：阿拉伯方言建模的另一维度 

**Authors**: Sanad Shaban, Nizar Habash  

**Link**: [PDF](https://arxiv.org/pdf/2508.17347)  

**Abstract**: Arabic dialects form a diverse continuum, yet NLP models often treat them as discrete categories. Recent work addresses this issue by modeling dialectness as a continuous variable, notably through the Arabic Level of Dialectness (ALDi). However, ALDi reduces complex variation to a single dimension. We propose a complementary measure: the Arabic Generality Score (AGS), which quantifies how widely a word is used across dialects. We introduce a pipeline that combines word alignment, etymology-aware edit distance, and smoothing to annotate a parallel corpus with word-level AGS. A regression model is then trained to predict AGS in context. Our approach outperforms strong baselines, including state-of-the-art dialect ID systems, on a multi-dialect benchmark. AGS offers a scalable, linguistically grounded way to model lexical generality, enriching representations of Arabic dialectness. 

**Abstract (ZH)**: 阿拉伯方言构成一个多元连续体，然而NLP模型常将其视为离散类别。最近的研究通过将方言特征建模为连续变量来解决这一问题，突出表现为阿拉伯方言连续性水平（ALDi）模型。然而，ALDi将复杂的变体简化为单一维度。我们提出一个补充指标：阿拉伯词汇普适性评分（AGS），量化一个词在不同方言中的使用范围。我们引入了一种管道，结合词汇对齐、语源学感知编辑距离和平滑技术，标注平行语料库中的词级AGS。然后训练回归模型以预测上下文中词级AGS。我们的方法在多方言基准测试中优于Strong Baselines，包括最先进的方言识别系统。AGS提供了一种可扩展、基于语言学的词汇普适性建模方式，丰富了阿拉伯方言特征的表现。 

---
# Agentic AI for Software: thoughts from Software Engineering community 

**Title (ZH)**: 软件领域的代理型AI：来自软件工程社区的思考 

**Authors**: Abhik Roychoudhury  

**Link**: [PDF](https://arxiv.org/pdf/2508.17343)  

**Abstract**: AI agents have recently shown significant promise in software engineering. Much public attention has been transfixed on the topic of code generation from Large Language Models (LLMs) via a prompt. However, software engineering is much more than programming, and AI agents go far beyond instructions given by a prompt.
At the code level, common software tasks include code generation, testing, and program repair. Design level software tasks may include architecture exploration, requirements understanding, and requirements enforcement at the code level. Each of these software tasks involves micro-decisions which can be taken autonomously by an AI agent, aided by program analysis tools. This creates the vision of an AI software engineer, where the AI agent can be seen as a member of a development team.
Conceptually, the key to successfully developing trustworthy agentic AI-based software workflows will be to resolve the core difficulty in software engineering - the deciphering and clarification of developer intent. Specification inference, or deciphering the intent, thus lies at the heart of many software tasks, including software maintenance and program repair. A successful deployment of agentic technology into software engineering would involve making conceptual progress in such intent inference via agents.
Trusting the AI agent becomes a key aspect, as software engineering becomes more automated. Higher automation also leads to higher volume of code being automatically generated, and then integrated into code-bases. Thus to deal with this explosion, an emerging direction is AI-based verification and validation (V & V) of AI generated code. We posit that agentic software workflows in future will include such AIbased V&V. 

**Abstract (ZH)**: AI代理在软件工程中的 recent进展及其挑战：从代码生成到基于代理的验证与验证 

---
# Capturing Legal Reasoning Paths from Facts to Law in Court Judgments using Knowledge Graphs 

**Title (ZH)**: 使用知识图谱从事实到法律 capturing 法庭判决中的法律推理路径 

**Authors**: Ryoma Kondo, Riona Matsuoka, Takahiro Yoshida, Kazuyuki Yamasawa, Ryohei Hisano  

**Link**: [PDF](https://arxiv.org/pdf/2508.17340)  

**Abstract**: Court judgments reveal how legal rules have been interpreted and applied to facts, providing a foundation for understanding structured legal reasoning. However, existing automated approaches for capturing legal reasoning, including large language models, often fail to identify the relevant legal context, do not accurately trace how facts relate to legal norms, and may misrepresent the layered structure of judicial reasoning. These limitations hinder the ability to capture how courts apply the law to facts in practice. In this paper, we address these challenges by constructing a legal knowledge graph from 648 Japanese administrative court decisions. Our method extracts components of legal reasoning using prompt-based large language models, normalizes references to legal provisions, and links facts, norms, and legal applications through an ontology of legal inference. The resulting graph captures the full structure of legal reasoning as it appears in real court decisions, making implicit reasoning explicit and machine-readable. We evaluate our system using expert annotated data, and find that it achieves more accurate retrieval of relevant legal provisions from facts than large language model baselines and retrieval-augmented methods. 

**Abstract (ZH)**: 法院判决揭示了法律规则如何被解释和应用于事实的过程，为理解结构化的法律推理提供了基础。然而，现有的自动化法律推理捕获方法，包括大型语言模型，往往无法识别相关的法律背景，不能准确追踪事实与法律规范之间的关系，并可能误代表司法推理的层次结构。这些限制阻碍了捕捉法院在实践中如何适用法律的能力。本文通过构建来自648份日本行政法院判决的法律知识图谱，解决了这些挑战。我们的方法使用基于提示的大语言模型提取法律推理的组件，规范化对法律规定的引用，并通过法律推理本体将事实、规范和法律应用连接起来。生成的图谱捕捉了实际法院判决中法律推理的完整结构，使隐含的推理明确化并可机器读取。我们使用专家标注的数据评估了系统，并发现它在从事实中检索相关法律规定的准确性上超过了大型语言模型基线和检索增强方法。 

---
# Modality-Specific Speech Enhancement and Noise-Adaptive Fusion for Acoustic and Body-Conduction Microphone Framework 

**Title (ZH)**: 基于模态特定语音增强和噪声自适应融合的声传导麦克风框架 

**Authors**: Yunsik Kim, Yoonyoung Chung  

**Link**: [PDF](https://arxiv.org/pdf/2508.17336)  

**Abstract**: Body\-conduction microphone signals (BMS) bypass airborne sound, providing strong noise resistance. However, a complementary modality is required to compensate for the inherent loss of high\-frequency information. In this study, we propose a novel multi\-modal framework that combines BMS and acoustic microphone signals (AMS) to achieve both noise suppression and high\-frequency reconstruction. Unlike conventional multi\-modal approaches that simply merge features, our method employs two specialized networks\: a mapping-based model to enhance BMS and a masking-based model to denoise AMS. These networks are integrated through a dynamic fusion mechanism that adapts to local noise conditions, ensuring the optimal use of each modality's strengths. We performed evaluations on the TAPS dataset, augmented with DNS\-2023 noise clips, using objective speech quality metrics. The results clearly demonstrate that our approach outperforms single\-modal solutions in a wide range of noisy environments. 

**Abstract (ZH)**: 基于传导的麦克风信号和声学麦克风信号的新型多模态框架：噪声抑制与高频重建 

---
# Mind the (Language) Gap: Towards Probing Numerical and Cross-Lingual Limits of LVLMs 

**Title (ZH)**: 注意（语言）差距：探索LVLMs在数值和跨语言能力上的局限性 

**Authors**: Somraj Gautam, Abhirama Subramanyam Penamakuri, Abhishek Bhandari, Gaurav Harit  

**Link**: [PDF](https://arxiv.org/pdf/2508.17334)  

**Abstract**: We introduce MMCRICBENCH-3K, a benchmark for Visual Question Answering (VQA) on cricket scorecards, designed to evaluate large vision-language models (LVLMs) on complex numerical and cross-lingual reasoning over semi-structured tabular images. MMCRICBENCH-3K comprises 1,463 synthetically generated scorecard images from ODI, T20, and Test formats, accompanied by 1,500 English QA pairs. It includes two subsets: MMCRICBENCH-E-1.5K, featuring English scorecards, and MMCRICBENCH-H-1.5K, containing visually similar Hindi scorecards, with all questions and answers kept in English to enable controlled cross-script evaluation. The task demands reasoning over structured numerical data, multi-image context, and implicit domain knowledge. Empirical results show that even state-of-the-art LVLMs, such as GPT-4o and Qwen2.5VL, struggle on the English subset despite it being their primary training language and exhibit a further drop in performance on the Hindi subset. This reveals key limitations in structure-aware visual text understanding, numerical reasoning, and cross-lingual generalization. The dataset is publicly available via Hugging Face at this https URL, to promote LVLM research in this direction. 

**Abstract (ZH)**: MMCRICBENCH-3K：用于板球比分卡视觉问答的基准测试，旨在评估大规模视觉-语言模型在半结构化表格图像上的复杂数值和跨语言推理能力 

---
# Omne-R1: Learning to Reason with Memory for Multi-hop Question Answering 

**Title (ZH)**: Omne-R1：基于记忆进行多跳问答的推理学习 

**Authors**: Boyuan Liu, Feng Ji, Jiayan Nan, Han Zhao, Weiling Chen, Shihao Xu, Xing Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.17330)  

**Abstract**: This paper introduces Omne-R1, a novel approach designed to enhance multi-hop question answering capabilities on schema-free knowledge graphs by integrating advanced reasoning models. Our method employs a multi-stage training workflow, including two reinforcement learning phases and one supervised fine-tuning phase. We address the challenge of limited suitable knowledge graphs and QA data by constructing domain-independent knowledge graphs and auto-generating QA pairs. Experimental results show significant improvements in answering multi-hop questions, with notable performance gains on more complex 3+ hop questions. Our proposed training framework demonstrates strong generalization abilities across diverse knowledge domains. 

**Abstract (ZH)**: Omne-R1：一种用于无模式知识图谱多跳问答的先进推理模型集成方法 

---
# CultranAI at PalmX 2025: Data Augmentation for Cultural Knowledge Representation 

**Title (ZH)**: CultranAI 在 PalmX 2025：文化知识表示的数据增强方法 

**Authors**: Hunzalah Hassan Bhatti, Youssef Ahmed, Md Arid Hasan, Firoj Alam  

**Link**: [PDF](https://arxiv.org/pdf/2508.17324)  

**Abstract**: In this paper, we report our participation to the PalmX cultural evaluation shared task. Our system, CultranAI, focused on data augmentation and LoRA fine-tuning of large language models (LLMs) for Arabic cultural knowledge representation. We benchmarked several LLMs to identify the best-performing model for the task. In addition to utilizing the PalmX dataset, we augmented it by incorporating the Palm dataset and curated a new dataset of over 22K culturally grounded multiple-choice questions (MCQs). Our experiments showed that the Fanar-1-9B-Instruct model achieved the highest performance. We fine-tuned this model on the combined augmented dataset of 22K+ MCQs. On the blind test set, our submitted system ranked 5th with an accuracy of 70.50%, while on the PalmX development set, it achieved an accuracy of 84.1%. 

**Abstract (ZH)**: 本文报告了我们参加的PalmX文化评价共享任务的情况。我们的系统CultranAI专注于通过数据增强和LoRA微调大型语言模型（LLMs）来表示阿拉伯文化知识。我们对几种LLMs进行了基准测试，以确定最适合该任务的模型。除了使用PalmX数据集外，我们还通过整合Palm数据集并创建了一个新的包含超过22K个文化基础多项选择题（MCQs）的定制数据集，对系统进行了增强。实验结果表明，Fanar-1-9B-Instruct模型表现最佳。我们将该模型在包含22K+个MCQs的增强数据集上进行了微调。在盲测集上，提交的系统排名第五，准确率为70.50%，而在PalmX开发集上，其准确率为84.1%。 

---
# Chinese Court Simulation with LLM-Based Agent System 

**Title (ZH)**: 基于LLM的代理系统模拟中国法庭 

**Authors**: Kaiyuan Zhang, Jiaqi Li, Yueyue Wu, Haitao Li, Cheng Luo, Shaokun Zou, Yujia Zhou, Weihang Su, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17322)  

**Abstract**: Mock trial has long served as an important platform for legal professional training and education. It not only helps students learn about realistic trial procedures, but also provides practical value for case analysis and judgment prediction. Traditional mock trials are difficult to access by the public because they rely on professional tutors and human participants. Fortunately, the rise of large language models (LLMs) provides new opportunities for creating more accessible and scalable court simulations. While promising, existing research mainly focuses on agent construction while ignoring the systematic design and evaluation of court simulations, which are actually more important for the credibility and usage of court simulation in practice. To this end, we present the first court simulation framework -- SimCourt -- based on the real-world procedure structure of Chinese courts. Our framework replicates all 5 core stages of a Chinese trial and incorporates 5 courtroom roles, faithfully following the procedural definitions in China. To simulate trial participants with different roles, we propose and craft legal agents equipped with memory, planning, and reflection abilities. Experiment on legal judgment prediction show that our framework can generate simulated trials that better guide the system to predict the imprisonment, probation, and fine of each case. Further annotations by human experts show that agents' responses under our simulation framework even outperformed judges and lawyers from the real trials in many scenarios. These further demonstrate the potential of LLM-based court simulation. 

**Abstract (ZH)**: 基于大型语言模型的法庭模拟框架：SimCourt 

---
# Bine Trees: Enhancing Collective Operations by Optimizing Communication Locality 

**Title (ZH)**: 二叉树：通过优化通信局部性增强集体操作 

**Authors**: Daniele De Sensi, Saverio Pasqualoni, Lorenzo Piarulli, Tommaso Bonato, Seydou Ba, Matteo Turisini, Jens Domke, Torsten Hoefler  

**Link**: [PDF](https://arxiv.org/pdf/2508.17311)  

**Abstract**: Communication locality plays a key role in the performance of collective operations on large HPC systems, especially on oversubscribed networks where groups of nodes are fully connected internally but sparsely linked through global connections. We present Bine (binomial negabinary) trees, a family of collective algorithms that improve communication locality. Bine trees maintain the generality of binomial trees and butterflies while cutting global-link traffic by up to 33%. We implement eight Bine-based collectives and evaluate them on four large-scale supercomputers with Dragonfly, Dragonfly+, oversubscribed fat-tree, and torus topologies, achieving up to 5x speedups and consistent reductions in global-link traffic across different vector sizes and node counts. 

**Abstract (ZH)**: 通信局部性在大型HPC系统中集体操作的性能中发挥着关键作用，尤其是在超订阅网络中，内部节点完全连接但通过全局连接稀疏连接。我们提出了Bine（二项负二进制）树，这是一种提高通信局部性的集体算法家族，保持了二项树和蝶形的通用性，并将全局连接流量最多降低了33%。我们在具有Dragonfly、Dragonfly+、超订阅网状网络和环状拓扑的四种大规模超级计算机上实现了八种Bine基集体算法，并实现了最高5倍的加速性能，并在不同向量尺寸和节点数下一致地减少了全局连接流量。 

---
# Explain Before You Answer: A Survey on Compositional Visual Reasoning 

**Title (ZH)**: 解释再作答：关于组合视觉推理的综述 

**Authors**: Fucai Ke, Joy Hsu, Zhixi Cai, Zixian Ma, Xin Zheng, Xindi Wu, Sukai Huang, Weiqing Wang, Pari Delir Haghighi, Gholamreza Haffari, Ranjay Krishna, Jiajun Wu, Hamid Rezatofighi  

**Link**: [PDF](https://arxiv.org/pdf/2508.17298)  

**Abstract**: Compositional visual reasoning has emerged as a key research frontier in multimodal AI, aiming to endow machines with the human-like ability to decompose visual scenes, ground intermediate concepts, and perform multi-step logical inference. While early surveys focus on monolithic vision-language models or general multimodal reasoning, a dedicated synthesis of the rapidly expanding compositional visual reasoning literature is still missing. We fill this gap with a comprehensive survey spanning 2023 to 2025 that systematically reviews 260+ papers from top venues (CVPR, ICCV, NeurIPS, ICML, ACL, etc.). We first formalize core definitions and describe why compositional approaches offer advantages in cognitive alignment, semantic fidelity, robustness, interpretability, and data efficiency. Next, we trace a five-stage paradigm shift: from prompt-enhanced language-centric pipelines, through tool-enhanced LLMs and tool-enhanced VLMs, to recently minted chain-of-thought reasoning and unified agentic VLMs, highlighting their architectural designs, strengths, and limitations. We then catalog 60+ benchmarks and corresponding metrics that probe compositional visual reasoning along dimensions such as grounding accuracy, chain-of-thought faithfulness, and high-resolution perception. Drawing on these analyses, we distill key insights, identify open challenges (e.g., limitations of LLM-based reasoning, hallucination, a bias toward deductive reasoning, scalable supervision, tool integration, and benchmark limitations), and outline future directions, including world-model integration, human-AI collaborative reasoning, and richer evaluation protocols. By offering a unified taxonomy, historical roadmap, and critical outlook, this survey aims to serve as a foundational reference and inspire the next generation of compositional visual reasoning research. 

**Abstract (ZH)**: 组成式视觉推理已成为多模态AI的关键研究前沿，旨在赋予机器类似人类的能力，分解视觉场景、 grounding 中间概念，并进行多步逻辑推理。虽然早期综述主要关注于整体型视觉语言模型或一般多模态推理，但专门梳理这一迅速扩展的组成式视觉推理文献仍然缺失。我们通过一个涵盖2023至2025年的全面综述填补这一空白，系统回顾了来自顶级会议（CVPR、ICCV、NeurIPS、ICML、ACL等）的260多篇论文。我们首先正式化核心定义，并描述组成式方法在认知对齐、语义保真度、鲁棒性、可解释性以及数据效率方面的优势。接着，我们跟踪了五个阶段的范式转变：从增强型提示语言中心流程，经过工具增强的大型语言模型和视觉语言模型，到最近提出的链式推理和统一仿生视觉语言模型，突出它们的架构设计、优势和局限性。我们随后整理了60多种基准和相应的度量标准，这些标准从不同的维度（如grounding准确性、链式推理忠实性和高分辨率感知）探索组成式视觉推理。基于这些分析，我们提炼关键见解，识别开放挑战（如基于大型语言模型的推理限制、幻觉、倾向于演绎推理、可扩展监督、工具集成以及基准限制），并概述未来方向，包括世界模型集成、人类-人工智能协作推理和更丰富的评估协议。通过提供统一的分类体系、历史路线图和批判性展望，本综述旨在成为基础性参考文献，并激发下一代组成式视觉推理研究。 

---
# Deep Learning-Assisted Detection of Sarcopenia in Cross-Sectional Computed Tomography Imaging 

**Title (ZH)**: 深度学习辅助横断面计算机断层扫描影像中的肌少症检测 

**Authors**: Manish Bhardwaj, Huizhi Liang, Ashwin Sivaharan, Sandip Nandhra, Vaclav Snasel, Tamer El-Sayed, Varun Ojha  

**Link**: [PDF](https://arxiv.org/pdf/2508.17275)  

**Abstract**: Sarcopenia is a progressive loss of muscle mass and function linked to poor surgical outcomes such as prolonged hospital stays, impaired mobility, and increased mortality. Although it can be assessed through cross-sectional imaging by measuring skeletal muscle area (SMA), the process is time-consuming and adds to clinical workloads, limiting timely detection and management; however, this process could become more efficient and scalable with the assistance of artificial intelligence applications. This paper presents high-quality three-dimensional cross-sectional computed tomography (CT) images of patients with sarcopenia collected at the Freeman Hospital, Newcastle upon Tyne Hospitals NHS Foundation Trust. Expert clinicians manually annotated the SMA at the third lumbar vertebra, generating precise segmentation masks. We develop deep-learning models to measure SMA in CT images and automate this task. Our methodology employed transfer learning and self-supervised learning approaches using labelled and unlabeled CT scan datasets. While we developed qualitative assessment models for detecting sarcopenia, we observed that the quantitative assessment of SMA is more precise and informative. This approach also mitigates the issue of class imbalance and limited data availability. Our model predicted the SMA, on average, with an error of +-3 percentage points against the manually measured SMA. The average dice similarity coefficient of the predicted masks was 93%. Our results, therefore, show a pathway to full automation of sarcopenia assessment and detection. 

**Abstract (ZH)**: 肌少症是与肌肉质量和功能逐渐减退相关的疾病，与术后长期住院、移动障碍和死亡率增加等不良手术结果有关。虽然可以通过测量骨骼肌面积（SMA）来横向成像评估肌少症，但这一过程耗时且增加了临床工作负担，限制了及时检测和管理；然而，在人工智能应用的辅助下，这一过程可以变得更加高效和可扩展。本文展示了在纽卡斯尔皇家弗里曼医院 NHS 基金会信托的肌少症患者中收集的高质量三维横截面计算机断层扫描（CT）图像。专家临床医师手动在第三腰椎处标注了骨骼肌面积（SMA），生成了精确的分割掩模。我们开发了深度学习模型来测量CT图像中的SMA并自动化这一任务。我们的方法使用了迁移学习和半监督学习方法，利用有标签和无标签的CT扫描数据集。虽然我们为检测肌少症开发了定性评估模型，但我们发现对骨骼肌面积（SMA）的定量评估更为精确和具有信息量。此外，该方法还缓解了类别不平衡和数据可用性有限的问题。我们的模型在平均误差为±3个百分点的情况下预测了SMA，预测掩模的平均骰子相似系数为93%。因此，我们的结果显示了一条实现肌少症评估和检测完全自动化的途径。 

---
# ResLink: A Novel Deep Learning Architecture for Brain Tumor Classification with Area Attention and Residual Connections 

**Title (ZH)**: ResLink：一种基于区域注意力和残差连接的新型深度学习架构用于脑肿瘤分类 

**Authors**: Sumedha Arya, Nirmal Gaud  

**Link**: [PDF](https://arxiv.org/pdf/2508.17259)  

**Abstract**: Brain tumors show significant health challenges due to their potential to cause critical neurological functions. Early and accurate diagnosis is crucial for effective treatment. In this research, we propose ResLink, a novel deep learning architecture for brain tumor classification using CT scan images. ResLink integrates novel area attention mechanisms with residual connections to enhance feature learning and spatial understanding for spatially rich image classification tasks. The model employs a multi-stage convolutional pipeline, incorporating dropout, regularization, and downsampling, followed by a final attention-based refinement for classification. Trained on a balanced dataset, ResLink achieves a high accuracy of 95% and demonstrates strong generalizability. This research demonstrates the potential of ResLink in improving brain tumor classification, offering a robust and efficient technique for medical imaging applications. 

**Abstract (ZH)**: 基于CT扫描图像的脑肿瘤分类：ResLink新型深度学习架构在重要神经功能潜在威胁下的早期准确诊断中应用的研究 

---
# Provable Generalization in Overparameterized Neural Nets 

**Title (ZH)**: 过参数化神经网络的可证明泛化能力 

**Authors**: Aviral Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2508.17256)  

**Abstract**: Deep neural networks often contain far more parameters than training examples, yet they still manage to generalize well in practice. Classical complexity measures such as VC-dimension or PAC-Bayes bounds usually become vacuous in this overparameterized regime, offering little explanation for the empirical success of models like Transformers. In this work, I explore an alternative notion of capacity for attention-based models, based on the effective rank of their attention matrices. The intuition is that, although the parameter count is enormous, the functional dimensionality of attention is often much lower. I show that this quantity leads to a generalization bound whose dependence on sample size matches empirical scaling laws observed in large language models, up to logarithmic factors. While the analysis is not a complete theory of overparameterized learning, it provides evidence that spectral properties of attention, rather than raw parameter counts, may be the right lens for understanding why these models generalize. 

**Abstract (ZH)**: 注意力模型的过参数化泛化能力：基于注意力矩阵有效秩的容量度量 

---
# A biological vision inspired framework for machine perception of abutting grating illusory contours 

**Title (ZH)**: 生物视觉启发的接触栅格错觉边缘机器感知框架 

**Authors**: Xiao Zhang, Kai-Fu Yang, Xian-Shi Zhang, Hong-Zhi You, Hong-Mei Yan, Yong-Jie Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.17254)  

**Abstract**: Higher levels of machine intelligence demand alignment with human perception and cognition. Deep neural networks (DNN) dominated machine intelligence have demonstrated exceptional performance across various real-world tasks. Nevertheless, recent evidence suggests that DNNs fail to perceive illusory contours like the abutting grating, a discrepancy that misaligns with human perception patterns. Departing from previous works, we propose a novel deep network called illusory contour perception network (ICPNet) inspired by the circuits of the visual cortex. In ICPNet, a multi-scale feature projection (MFP) module is designed to extract multi-scale representations. To boost the interaction between feedforward and feedback features, a feature interaction attention module (FIAM) is introduced. Moreover, drawing inspiration from the shape bias observed in human perception, an edge detection task conducted via the edge fusion module (EFM) injects shape constraints that guide the network to concentrate on the foreground. We assess our method on the existing AG-MNIST test set and the AG-Fashion-MNIST test sets constructed by this work. Comprehensive experimental results reveal that ICPNet is significantly more sensitive to abutting grating illusory contours than state-of-the-art models, with notable improvements in top-1 accuracy across various subsets. This work is expected to make a step towards human-level intelligence for DNN-based models. 

**Abstract (ZH)**: 更高层次的机器智能需要与人类感知和认知相契合。受视皮层电路启发的幻觉边缘感知网络（ICPNet）在各种实际任务中表现出色的深度神经网络（DNN）近期显示出不能感知像邻接 gratings 这样的幻觉边缘，这一差异与人类感知模式不符。不同于以往的工作，我们提出了一种新的深度网络——幻觉边缘感知网络（ICPNet），以减轻这种不对齐。ICPNet 设计了一个多尺度特征投影（MFP）模块来提取多尺度表示。为了增强前向和反馈特征之间的交互，我们引入了一个特征交互注意模块（FIAM）。此外，受人类感知中观察到的形状偏见启发，通过边缘融合模块（EFM）执行边缘检测任务，该模块注入形状约束以引导网络关注前景。我们在现有的 AG-MNIST 测试集和由本文构建的 AG-Fashion-MNIST 测试集上评估了我们的方法。综合实验结果表明，ICPNet 在多种子集中，相比于最先进的模型具有显著更高的 top-1 准确率，对邻接 gratings 幻觉边缘的敏感性也明显更高。本工作有望为基于 DNN 的模型朝人类水平的智能迈出一步。 

---
# CoViPAL: Layer-wise Contextualized Visual Token Pruning for Large Vision-Language Models 

**Title (ZH)**: CoViPAL：分层上下文化视觉词元剪枝大型视觉语言模型 

**Authors**: Zicong Tang, Ziyang Ma, Suqing Wang, Zuchao Li, Lefei Zhang, Hai Zhao, Yun Li, Qianren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17243)  

**Abstract**: Large Vision-Language Models (LVLMs) process multimodal inputs consisting of text tokens and vision tokens extracted from images or videos. Due to the rich visual information, a single image can generate thousands of vision tokens, leading to high computational costs during the prefilling stage and significant memory overhead during decoding. Existing methods attempt to prune redundant vision tokens, revealing substantial redundancy in visual representations. However, these methods often struggle in shallow layers due to the lack of sufficient contextual information. We argue that many visual tokens are inherently redundant even in shallow layers and can be safely and effectively pruned with appropriate contextual signals. In this work, we propose CoViPAL, a layer-wise contextualized visual token pruning method that employs a Plug-and-Play Pruning Module (PPM) to predict and remove redundant vision tokens before they are processed by the LVLM. The PPM is lightweight, model-agnostic, and operates independently of the LVLM architecture, ensuring seamless integration with various models. Extensive experiments on multiple benchmarks demonstrate that CoViPAL outperforms training-free pruning methods under equal token budgets and surpasses training-based methods with comparable supervision. CoViPAL offers a scalable and efficient solution to improve inference efficiency in LVLMs without compromising accuracy. 

**Abstract (ZH)**: 大型多模态语言视觉模型（LVLMs）处理由文本标记和从图像或视频提取的视觉标记组成的多模态输入。由于丰富的视觉信息，单张图像可以生成数千个视觉标记，导致预填充阶段计算成本高，并且在解码过程中产生显著的内存开销。现有方法尝试剪枝冗余的视觉标记，揭示了视觉表示中的大量冗余。然而，这些方法在浅层网络中常常难以处理，因为缺乏足够的上下文信息。我们认为，即使在浅层网络中，许多视觉标记本质上也是冗余的，并且可以通过适当的上下文信号安全且有效地剪枝。在本文中，我们提出了一种逐层上下文化视觉标记剪枝方法CoViPAL，该方法采用可插即用剪枝模块（PPM）在这些标记被LVLM处理之前预测并移除冗余的视觉标记。PPM轻量级、模型无关，并独立于LVLM架构，确保可以无缝集成到各种模型中。在多个基准上的广泛实验表明，CoViPAL在相等的标记预算下优于无训练剪枝方法，并且在具有类似监督的情况下超过了基于训练的方法。CoViPAL提供了一种可扩展且高效的解决方案，可以在不牺牲准确性的前提下提高LVLM的推理效率。 

---
# ClaimGen-CN: A Large-scale Chinese Dataset for Legal Claim Generation 

**Title (ZH)**: ClaimGen-CN：大规模中文法律索赔生成数据集 

**Authors**: Siying Zhou, Yiquan Wu, Hui Chen, Xavier Hu, Kun Kuang, Adam Jatowt, Ming Hu, Chunyan Zheng, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17234)  

**Abstract**: Legal claims refer to the plaintiff's demands in a case and are essential to guiding judicial reasoning and case resolution. While many works have focused on improving the efficiency of legal professionals, the research on helping non-professionals (e.g., plaintiffs) remains unexplored. This paper explores the problem of legal claim generation based on the given case's facts. First, we construct ClaimGen-CN, the first dataset for Chinese legal claim generation task, from various real-world legal disputes. Additionally, we design an evaluation metric tailored for assessing the generated claims, which encompasses two essential dimensions: factuality and clarity. Building on this, we conduct a comprehensive zero-shot evaluation of state-of-the-art general and legal-domain large language models. Our findings highlight the limitations of the current models in factual precision and expressive clarity, pointing to the need for more targeted development in this domain. To encourage further exploration of this important task, we will make the dataset publicly available. 

**Abstract (ZH)**: 基于给定案件事实的法律索赔生成问题研究：从现实法律纠纷构建ClaimGen-CN数据集及评估方法 

---
# Module-Aware Parameter-Efficient Machine Unlearning on Transformers 

**Title (ZH)**: 模块意识的参数高效机器遗忘机制在变压器上的应用 

**Authors**: Wenjie Bao, Jian Lou, Yuke Hu, Xiaochen Li, Zhihao Liu, Jiaqi Liu, Zhan Qin, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2508.17233)  

**Abstract**: Transformer has become fundamental to a vast series of pre-trained large models that have achieved remarkable success across diverse applications. Machine unlearning, which focuses on efficiently removing specific data influences to comply with privacy regulations, shows promise in restricting updates to influence-critical parameters. However, existing parameter-efficient unlearning methods are largely devised in a module-oblivious manner, which tends to inaccurately identify these parameters and leads to inferior unlearning performance for Transformers. In this paper, we propose {\tt MAPE-Unlearn}, a module-aware parameter-efficient machine unlearning approach that uses a learnable pair of masks to pinpoint influence-critical parameters in the heads and filters of Transformers. The learning objective of these masks is derived by desiderata of unlearning and optimized through an efficient algorithm featured by a greedy search with a warm start. Extensive experiments on various Transformer models and datasets demonstrate the effectiveness and robustness of {\tt MAPE-Unlearn} for unlearning. 

**Abstract (ZH)**: Transformer已成为一系列预训练大型模型的基础，并在多种应用中取得了显著成功。机学习是专注于高效去除特定数据影响以符合隐私法规的一种方法，它有潜力限制更新仅影响关键参数。然而，现有参数高效学习消除方法大多是以模块无意识的方式设计的，这往往会不准确地识别这些参数，导致Transformer的消除性能不佳。在本文中，我们提出了MAPE-Unlearn，这是一种模块意识的参数高效机学习消除方法，它使用可学习的掩码对来定位Transformer头部和滤波器中的关键参数影响。这些掩码的学习目标由消除的期望导出，并通过具有暖启动的贪婪搜索高效算法进行优化。在各种Transformer模型和数据集上的广泛实验表明，MAPE-Unlearn在消除方面的有效性和鲁棒性。 

---
# Multi-Metric Preference Alignment for Generative Speech Restoration 

**Title (ZH)**: 多指标偏好对齐的生成语音恢复 

**Authors**: Junan Zhang, Xueyao Zhang, Jing Yang, Yuancheng Wang, Fan Fan, Zhizheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17229)  

**Abstract**: Recent generative models have significantly advanced speech restoration tasks, yet their training objectives often misalign with human perceptual preferences, resulting in suboptimal quality. While post-training alignment has proven effective in other generative domains like text and image generation, its application to generative speech restoration remains largely under-explored. This work investigates the challenges of applying preference-based post-training to this task, focusing on how to define a robust preference signal and curate high-quality data to avoid reward hacking. To address these challenges, we propose a multi-metric preference alignment strategy. We construct a new dataset, GenSR-Pref, comprising 80K preference pairs, where each chosen sample is unanimously favored by a complementary suite of metrics covering perceptual quality, signal fidelity, content consistency, and timbre preservation. This principled approach ensures a holistic preference signal. Applying Direct Preference Optimization (DPO) with our dataset, we observe consistent and significant performance gains across three diverse generative paradigms: autoregressive models (AR), masked generative models (MGM), and flow-matching models (FM) on various restoration benchmarks, in both objective and subjective evaluations. Ablation studies confirm the superiority of our multi-metric strategy over single-metric approaches in mitigating reward hacking. Furthermore, we demonstrate that our aligned models can serve as powerful ''data annotators'', generating high-quality pseudo-labels to serve as a supervision signal for traditional discriminative models in data-scarce scenarios like singing voice restoration. Demo Page:this https URL 

**Abstract (ZH)**: Recent生成模型在语音恢复任务中取得了显著进展，但其训练目标往往与人类的感知偏好不一致，导致质量不佳。虽然在文本和图像生成等其他生成领域中，后训练对齐已被证明是有效的，但在生成语音恢复中的应用仍然 largely 未被探索。本研究探讨了将基于偏好的后训练应用于该任务所面临的挑战，重点关注如何定义稳健的偏好信号并收集高质量数据以避免奖励欺骗。为了解决这些挑战，我们提出了一种多指标偏好对齐策略。我们构建了一个新的数据集GenSR-Pref，包含80,000个偏好对，其中每个选择的样本都得到了互补的多种指标的一致青睐，这些指标涵盖了感知质量、信号保真度、内容一致性和音色保持。这种方法确保了全面的偏好信号。通过我们的数据集应用直接偏好优化（DPO），在三种不同的生成范式：自回归模型（AR）、遮蔽生成模型（MGM）和流动匹配模型（FM）上的各种恢复基准上，我们在客观和主观评估中观察到一致且显著的性能提升。消融研究证实，与单一指标方法相比，我们的多指标策略在减轻奖励欺骗方面更具优越性。此外，我们展示了我们的对齐模型可以作为强大的“数据注释器”，生成高质量的伪标签，作为在数据稀缺场景下（如歌唱声音恢复）传统区别性模型的监督信号。 

---
# SSFO: Self-Supervised Faithfulness Optimization for Retrieval-Augmented Generation 

**Title (ZH)**: SSFO: 自监督忠实性优化 retriever-增强生成 

**Authors**: Xiaqiang Tang, Yi Wang, Keyu Hu, Rui Xu, Chuang Li, Weigao Sun, Jian Li, Sihong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2508.17225)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems require Large Language Models (LLMs) to generate responses that are faithful to the retrieved context. However, faithfulness hallucination remains a critical challenge, as existing methods often require costly supervision and post-training or significant inference burdens. To overcome these limitations, we introduce Self-Supervised Faithfulness Optimization (SSFO), the first self-supervised alignment approach for enhancing RAG faithfulness. SSFO constructs preference data pairs by contrasting the model's outputs generated with and without the context. Leveraging Direct Preference Optimization (DPO), SSFO aligns model faithfulness without incurring labeling costs or additional inference burden. We theoretically and empirically demonstrate that SSFO leverages a benign form of \emph{likelihood displacement}, transferring probability mass from parametric-based tokens to context-aligned tokens. Based on this insight, we propose a modified DPO loss function to encourage likelihood displacement. Comprehensive evaluations show that SSFO significantly outperforms existing methods, achieving state-of-the-art faithfulness on multiple context-based question-answering datasets. Notably, SSFO exhibits strong generalization, improving cross-lingual faithfulness and preserving general instruction-following capabilities. We release our code and model at the anonymous link: this https URL 

**Abstract (ZH)**: RAG系统要求大型语言模型生成与检索上下文一致的响应。然而，信实性幻觉仍然是一个关键挑战，因为现有方法通常需要昂贵的监督和后训练监督或显著的推理负担。为克服这些限制，我们引入了自监督信实性优化（SSFO），这是首个用于增强RAG信实性的自监督对齐方法。SSFO通过对比模型在有和没有上下文时生成的输出构建偏好数据对。利用直接偏好优化（DPO），SSFO无需标注成本或额外的推理负担即可对齐模型信实性。我们从理论上和实验上证明，SSFO利用了一种良性形式的\emph{似然性转移}，将基于参数的标记的概率质量转移到上下文对齐的标记上。基于此洞见，我们提出了修改后的DPO损失函数来鼓励似然性转移。全面的评估表明，SSFO显著优于现有方法，在多个基于上下文的问题回答数据集上取得了最先进的信实性。值得注意的是，SSFO表现出很强的泛化能力，能够提高跨语言信实性并保留通用指令遵循能力。我们发布了我们的代码和模型在以下匿名链接：this https URL。 

---
# Exposing Privacy Risks in Graph Retrieval-Augmented Generation 

**Title (ZH)**: 图检索增强生成中的隐私风险揭示 

**Authors**: Jiale Liu, Jiahao Zhang, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17222)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a powerful technique for enhancing Large Language Models (LLMs) with external, up-to-date knowledge. Graph RAG has emerged as an advanced paradigm that leverages graph-based knowledge structures to provide more coherent and contextually rich answers. However, the move from plain document retrieval to structured graph traversal introduces new, under-explored privacy risks. This paper investigates the data extraction vulnerabilities of the Graph RAG systems. We design and execute tailored data extraction attacks to probe their susceptibility to leaking both raw text and structured data, such as entities and their relationships. Our findings reveal a critical trade-off: while Graph RAG systems may reduce raw text leakage, they are significantly more vulnerable to the extraction of structured entity and relationship information. We also explore potential defense mechanisms to mitigate these novel attack surfaces. This work provides a foundational analysis of the unique privacy challenges in Graph RAG and offers insights for building more secure systems. 

**Abstract (ZH)**: 图RAG系统中的数据提取漏洞研究 

---
# GPG-HT: Generalized Policy Gradient with History-Aware Decision Transformer for Probabilistic Path Planning 

**Title (ZH)**: GPG-HT：具有历史意识决策变换器的广义策略梯度方法用于概率路径规划 

**Authors**: Xing Wei, Yuqi Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17218)  

**Abstract**: With the rapidly increased number of vehicles in urban areas, existing road infrastructure struggles to accommodate modern traffic demands, resulting in the issue of congestion. This highlights the importance of efficient path planning strategies. However, most recent navigation models focus solely on deterministic or time-dependent networks, while overlooking the correlations and the stochastic nature of traffic flows. In this work, we address the reliable shortest path problem within stochastic transportation networks under certain dependencies. We propose a path planning solution that integrates the decision Transformer with the Generalized Policy Gradient (GPG) framework. Based on the decision Transformer's capability to model long-term dependencies, our proposed solution improves the accuracy and stability of path decisions. Experimental results on the Sioux Falls Network (SFN) demonstrate that our approach outperforms previous baselines in terms of on-time arrival probability, providing more accurate path planning solutions. 

**Abstract (ZH)**: 在具有特定依赖性的随机运输网络中可靠的最短路径问题及其路径规划解决方案 

---
# How to make Medical AI Systems safer? Simulating Vulnerabilities, and Threats in Multimodal Medical RAG System 

**Title (ZH)**: 如何使医疗AI系统更安全？模拟多模态医疗RAG系统的漏洞和威胁 

**Authors**: Kaiwen Zuo, Zelin Liu, Raman Dutt, Ziyang Wang, Zhongtian Sun, Yeming Wang, Fan Mo, Pietro Liò  

**Link**: [PDF](https://arxiv.org/pdf/2508.17215)  

**Abstract**: Large Vision-Language Models (LVLMs) augmented with Retrieval-Augmented Generation (RAG) are increasingly employed in medical AI to enhance factual grounding through external clinical image-text retrieval. However, this reliance creates a significant attack surface. We propose MedThreatRAG, a novel multimodal poisoning framework that systematically probes vulnerabilities in medical RAG systems by injecting adversarial image-text pairs. A key innovation of our approach is the construction of a simulated semi-open attack environment, mimicking real-world medical systems that permit periodic knowledge base updates via user or pipeline contributions. Within this setting, we introduce and emphasize Cross-Modal Conflict Injection (CMCI), which embeds subtle semantic contradictions between medical images and their paired reports. These mismatches degrade retrieval and generation by disrupting cross-modal alignment while remaining sufficiently plausible to evade conventional filters. While basic textual and visual attacks are included for completeness, CMCI demonstrates the most severe degradation. Evaluations on IU-Xray and MIMIC-CXR QA tasks show that MedThreatRAG reduces answer F1 scores by up to 27.66% and lowers LLaVA-Med-1.5 F1 rates to as low as 51.36%. Our findings expose fundamental security gaps in clinical RAG systems and highlight the urgent need for threat-aware design and robust multimodal consistency checks. Finally, we conclude with a concise set of guidelines to inform the safe development of future multimodal medical RAG systems. 

**Abstract (ZH)**: Large Vision-Language Models (LVLMs) 增强检索增强生成 (RAG) 的 MedThreatRAG：系统性探究医疗 RAG 系统中的漏洞 

---
# Multi-Agent Visual-Language Reasoning for Comprehensive Highway Scene Understanding 

**Title (ZH)**: 多Agent视觉-语言推理以实现全面高速公路场景理解 

**Authors**: Yunxiang Yang, Ningning Xu, Jidong J. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17205)  

**Abstract**: This paper introduces a multi-agent framework for comprehensive highway scene understanding, designed around a mixture-of-experts strategy. In this framework, a large generic vision-language model (VLM), such as GPT-4o, is contextualized with domain knowledge to generates task-specific chain-of-thought (CoT) prompts. These fine-grained prompts are then used to guide a smaller, efficient VLM (e.g., Qwen2.5-VL-7B) in reasoning over short videos, along with complementary modalities as applicable. The framework simultaneously addresses multiple critical perception tasks, including weather classification, pavement wetness assessment, and traffic congestion detection, achieving robust multi-task reasoning while balancing accuracy and computational efficiency. To support empirical validation, we curated three specialized datasets aligned with these tasks. Notably, the pavement wetness dataset is multimodal, combining video streams with road weather sensor data, highlighting the benefits of multimodal reasoning. Experimental results demonstrate consistently strong performance across diverse traffic and environmental conditions. From a deployment perspective, the framework can be readily integrated with existing traffic camera systems and strategically applied to high-risk rural locations, such as sharp curves, flood-prone lowlands, or icy bridges. By continuously monitoring the targeted sites, the system enhances situational awareness and delivers timely alerts, even in resource-constrained environments. 

**Abstract (ZH)**: 一种基于专家混合策略的全面高速公路场景理解多agent框架 

---
# BudgetThinker: Empowering Budget-aware LLM Reasoning with Control Tokens 

**Title (ZH)**: BudgetThinker: 促进 Awareness 预算 理解的 Control Token 辅助 LLM 推理 

**Authors**: Hao Wen, Xinrui Wu, Yi Sun, Feifei Zhang, Liye Chen, Jie Wang, Yunxin Liu, Ya-Qin Zhang, Yuanchun Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.17196)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have leveraged increased test-time computation to enhance reasoning capabilities, a strategy that, while effective, incurs significant latency and resource costs, limiting their applicability in real-world time-constrained or cost-sensitive scenarios. This paper introduces BudgetThinker, a novel framework designed to empower LLMs with budget-aware reasoning, enabling precise control over the length of their thought processes. We propose a methodology that periodically inserts special control tokens during inference to continuously inform the model of its remaining token budget. This approach is coupled with a comprehensive two-stage training pipeline, beginning with Supervised Fine-Tuning (SFT) to familiarize the model with budget constraints, followed by a curriculum-based Reinforcement Learning (RL) phase that utilizes a length-aware reward function to optimize for both accuracy and budget adherence. We demonstrate that BudgetThinker significantly surpasses strong baselines in maintaining performance across a variety of reasoning budgets on challenging mathematical benchmarks. Our method provides a scalable and effective solution for developing efficient and controllable LLM reasoning, making advanced models more practical for deployment in resource-constrained and real-time environments. 

**Abstract (ZH)**: Recent advancements in大规模语言模型（LLMs）通过增加推理时间计算来提升推理能力，尽管这一策略有效，但也带来了显著的延迟和资源成本，限制了其在实时或成本敏感场景中的应用。本文介绍了BudgetThinker，一种新型框架，旨在使LLMs具备预算意识的推理能力，从而精确控制其思维过程的长度。我们提出了一种方法，在推理过程中周期性地插入特殊控制标记，以不断告知模型其剩余的标记预算。该方法结合了一个全面的两阶段训练管道，首先进行监督微调（SFT）使模型熟悉预算约束，随后通过基于课程的强化学习（RL）阶段，使用长度意识的奖励函数来优化准确性和预算遵从性。我们证明，BudgetThinker在多种挑战性的数学基准测试中，显著优于强大的基线模型，在不同的推理预算下保持了性能。我们的方法提供了一种可扩展且有效的解决方案，用于开发高效可控的LLM推理，使先进的模型在资源受限和实时环境中更具部署可行性。 

---
# LLM Assertiveness can be Mechanistically Decomposed into Emotional and Logical Components 

**Title (ZH)**: LLM 坚定性可以从机制上分解为情感和逻辑成分 

**Authors**: Hikaru Tsujimura, Arush Tagade  

**Link**: [PDF](https://arxiv.org/pdf/2508.17182)  

**Abstract**: Large Language Models (LLMs) often display overconfidence, presenting information with unwarranted certainty in high-stakes contexts. We investigate the internal basis of this behavior via mechanistic interpretability. Using open-sourced Llama 3.2 models fine-tuned on human annotated assertiveness datasets, we extract residual activations across all layers, and compute similarity metrics to localize assertive representations. Our analysis identifies layers most sensitive to assertiveness contrasts and reveals that high-assertive representations decompose into two orthogonal sub-components of emotional and logical clusters-paralleling the dual-route Elaboration Likelihood Model in Psychology. Steering vectors derived from these sub-components show distinct causal effects: emotional vectors broadly influence prediction accuracy, while logical vectors exert more localized effects. These findings provide mechanistic evidence for the multi-component structure of LLM assertiveness and highlight avenues for mitigating overconfident behavior. 

**Abstract (ZH)**: 大型语言模型（LLMs）在高风险情境下常常表现出过度自信，以不必要的确信性呈现信息。我们通过机制可解释性研究其内部基础。使用开源的Llama 3.2模型，在人类标注的自信程度数据集上进行微调后，我们提取了所有层的残余激活，并计算相似性度量以定位自信表征。我们的分析确定了对自信对比最敏感的层数，并揭示出高自信表征分解为情感和逻辑两个正交亚组件，与心理学中的双途径深入加工模型相平行。从这些亚组件导出的引导向量显示出不同的因果效应：情感向量广泛影响预测准确性，而逻辑向量则产生更局部的影响。这些发现为LLM自信的多组件结构提供了机制证据，并指出了减轻过度自信行为的途径。 

---
# Scaling Graph Transformers: A Comparative Study of Sparse and Dense Attention 

**Title (ZH)**: 扩展图变换器：稀疏与密集注意力的比较研究 

**Authors**: Leon Dimitrov  

**Link**: [PDF](https://arxiv.org/pdf/2508.17175)  

**Abstract**: Graphs have become a central representation in machine learning for capturing relational and structured data across various domains. Traditional graph neural networks often struggle to capture long-range dependencies between nodes due to their local structure. Graph transformers overcome this by using attention mechanisms that allow nodes to exchange information globally. However, there are two types of attention in graph transformers: dense and sparse. In this paper, we compare these two attention mechanisms, analyze their trade-offs, and highlight when to use each. We also outline current challenges and problems in designing attention for graph transformers. 

**Abstract (ZH)**: 图已成为机器学习中用于捕捉各种领域中关系和结构化数据的核心表示。传统图神经网络往往难以捕捉由于其局部结构导致的长距离节点依赖关系。图变压器通过使用允许节点进行全局信息交换的注意力机制来克服这一问题。然而，图变压器中有两种类型的注意力机制：密集型和稀疏型。在本文中，我们比较了这两种注意力机制，分析了它们的权衡，并指出了在每种情况下使用它们的情形。我们还概述了设计图变压器注意力机制当前面临的挑战和问题。 

---
# ONG: Orthogonal Natural Gradient Descent 

**Title (ZH)**: ONG：正交自然梯度下降 

**Authors**: Yajat Yadav, Jathin Korrapati, Patrick Mendoza  

**Link**: [PDF](https://arxiv.org/pdf/2508.17169)  

**Abstract**: Orthogonal gradient descent has emerged as a powerful method for continual learning tasks. However, its Euclidean projections overlook the underlying information-geometric structure of the space of distributions parametrized by neural networks, which can lead to suboptimal convergence in learning tasks. To counteract this, we combine it with the idea of the natural gradient and present ONG (Orthogonal Natural Gradient Descent). ONG preconditions each new task gradient with an efficient EKFAC approximation of the inverse Fisher information matrix, yielding updates that follow the steepest descent direction under a Riemannian metric. To preserve performance on previously learned tasks, ONG projects these natural gradients onto the orthogonal complement of prior task gradients. We provide a theoretical justification for this procedure, introduce the ONG algorithm, and benchmark its performance on the Permuted and Rotated MNIST datasets. All code for our experiments/reproducibility can be found at this https URL. 

**Abstract (ZH)**: 正交自然梯度下降已成为连续学习任务中一种强大的方法。然而，其欧几里得投影忽略了由神经网络参数化的分布空间下的信息几何结构，可能导致学习任务中次优的收敛效果。为解决这一问题，我们将自然梯度的思想与正交梯度下降相结合，提出了一种称为ONG（正交自然梯度下降）的方法。ONG利用EKFAC近似逆费舍尔信息矩阵对每个新任务梯度进行预条件化，生成在黎曼度量下沿最陡下降方向的更新。为了在保持先前学习任务性能的同时适应新任务，ONG将这些自然梯度投影到先前任务梯度的正交补空间。我们为这种操作提供了理论依据，并介绍了ONG算法，同时在Permuted和Rotated MNIST数据集上展示了其性能。所有实验代码及可重复性代码可在以下链接找到：this https URL。 

---
# Error analysis for the deep Kolmogorov method 

**Title (ZH)**: 深层柯尔莫戈罗夫方法的误差分析 

**Authors**: Iulian Cîmpean, Thang Do, Lukas Gonon, Arnulf Jentzen, Ionel Popescu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17167)  

**Abstract**: The deep Kolmogorov method is a simple and popular deep learning based method for approximating solutions of partial differential equations (PDEs) of the Kolmogorov type. In this work we provide an error analysis for the deep Kolmogorov method for heat PDEs. Specifically, we reveal convergence with convergence rates for the overall mean square distance between the exact solution of the heat PDE and the realization function of the approximating deep neural network (DNN) associated with a stochastic optimization algorithm in terms of the size of the architecture (the depth/number of hidden layers and the width of the hidden layers) of the approximating DNN, in terms of the number of random sample points used in the loss function (the number of input-output data pairs used in the loss function), and in terms of the size of the optimization error made by the employed stochastic optimization method. 

**Abstract (ZH)**: 基于深度学习的深层柯尔莫戈罗夫方法是一种用于近似柯尔莫戈罗夫类型偏微分方程（PDEs）解的简单而流行的深度学习方法。本文提供了深层柯尔莫戈罗夫方法在热方程中的误差分析，具体而言，我们揭示了近似解与热方程精确解之间的总体均方距离的收敛性及其收敛速率，以及这些距离与逼近深度神经网络（DNN）结构大小（深度和隐藏层数量及宽度）、损失函数中使用的随机样本点的数量（损失函数中输入-输出数据对的数量）以及所使用随机优化方法的优化误差大小之间的关系。 

---
# Beyond Play and Pause: Turning GPT-4o Spatial Weakness into a Strength for In-Depth Interactive Video Learning 

**Title (ZH)**: 超越播放与暂停：将GPT-4o的空间劣势转化为深入互动视频学习的优势 

**Authors**: Sajad Goudarzi, Samaneh Zamanifard  

**Link**: [PDF](https://arxiv.org/pdf/2508.17160)  

**Abstract**: Traditional video-based learning remains passive, offering limited opportunities for users to engage dynamically with content. While current AI-powered tools offer transcription and summarization, they lack real-time, region-specific interaction capabilities. This paper introduces Untwist, an AI-driven system that enables interactive video learning by allowing users to ask questions about the entire video or specific regions using a bounding box, receiving context-aware, multimodal responses. By integrating GPT APIs with Computer Vision techniques, Untwist extracts, processes, and structures video content to enhance comprehension. Our approach addresses GPT-4o spatial weakness by leveraging annotated frames instead of raw coordinate data, significantly improving accuracy in localizing and interpreting video content. This paper describes the system architecture, including video pre-processing and real-time interaction, and outlines how Untwist can transform passive video consumption into an interactive, AI-driven learning experience with the potential to enhance engagement and comprehension. 

**Abstract (ZH)**: 传统基于视频的学习方式仍然被动，为用户动态参与内容的机会有限。尽管当前的AI辅助工具提供了转录和摘要功能，但它们缺乏实时的区域特定交互能力。本文介绍了一种基于AI的系统Untwist，该系统通过允许用户使用边界框提问整个视频或特定区域，从而实现互动式视频学习，获得上下文感知的多模态响应。通过将GPT API与计算机视觉技术结合，Untwist提取、处理和结构化视频内容以增强理解。我们的方法通过利用标注帧而不是原始坐标数据来克服GPT-4的空间弱点，显著提高了在定位和解释视频内容方面的准确性。本文描述了系统的架构，包括视频预处理和实时交互，并概述了Untwist如何将被动的视频消费转化为一种互动的、以AI驱动的学习体验，从而有可能增强参与度和理解度。 

---
# Mind the Gap: Time-of-Check to Time-of-Use Vulnerabilities in LLM-Enabled Agents 

**Title (ZH)**: 注意差距：LLM启用代理中的时间检查到时间使用漏洞 

**Authors**: Derek Lilienthal, Sanghyun Hong  

**Link**: [PDF](https://arxiv.org/pdf/2508.17155)  

**Abstract**: Large Language Model (LLM)-enabled agents are rapidly emerging across a wide range of applications, but their deployment introduces vulnerabilities with security implications. While prior work has examined prompt-based attacks (e.g., prompt injection) and data-oriented threats (e.g., data exfiltration), time-of-check to time-of-use (TOCTOU) remain largely unexplored in this context. TOCTOU arises when an agent validates external state (e.g., a file or API response) that is later modified before use, enabling practical attacks such as malicious configuration swaps or payload injection. In this work, we present the first study of TOCTOU vulnerabilities in LLM-enabled agents. We introduce TOCTOU-Bench, a benchmark with 66 realistic user tasks designed to evaluate this class of vulnerabilities. As countermeasures, we adapt detection and mitigation techniques from systems security to this setting and propose prompt rewriting, state integrity monitoring, and tool-fusing. Our study highlights challenges unique to agentic workflows, where we achieve up to 25% detection accuracy using automated detection methods, a 3% decrease in vulnerable plan generation, and a 95% reduction in the attack window. When combining all three approaches, we reduce the TOCTOU vulnerabilities from an executed trajectory from 12% to 8%. Our findings open a new research direction at the intersection of AI safety and systems security. 

**Abstract (ZH)**: 大语言模型（LLM）驱动的代理正迅速应用于广泛的应用领域，但其部署引入了具有安全影响的漏洞。尽管先前的工作已经考察了基于提示的攻击（如提示注入）和数据导向的威胁（如数据泄露），但在这一背景下，时间从检查到使用的漏洞（TOCTOU）尚未被充分研究。当代理在使用之前验证了后来被修改的外部状态（如文件或API响应）时，会引发TOCTOU，这使得恶意配置交换或有效载荷注入等实际攻击成为可能。在本文中，我们呈现了第一个对LLM驱动代理中的TOCTOU漏洞的研究。我们引入了TOCTOU-Bench基准，其中包含66个现实用户的任务，用于评估此类漏洞。作为缓解措施，我们借鉴系统安全领域的检测和缓解技术，并提出了提示重写、状态完整性监控和工具融合。我们的研究突出了代理工作流程特有的挑战，通过自动化检测方法实现了高达25%的检测准确率，减少了3%的易受攻击计划生成，并将攻击窗口降低了95%。当我们结合所有三种方法时，我们将执行轨迹中的TOCTOU漏洞从12%降低到了8%。我们的发现为人工智能安全与系统安全的交叉领域开辟了新的研究方向。 

---
# Natural Language Satisfiability: Exploring the Problem Distribution and Evaluating Transformer-based Language Models 

**Title (ZH)**: 自然语言满足性问题：探索问题分布并评估基于 Transformer 的语言模型 

**Authors**: Tharindu Madusanka, Ian Pratt-Hartmann, Riza Batista-Navarro  

**Link**: [PDF](https://arxiv.org/pdf/2508.17153)  

**Abstract**: Efforts to apply transformer-based language models (TLMs) to the problem of reasoning in natural language have enjoyed ever-increasing success in recent years. The most fundamental task in this area to which nearly all others can be reduced is that of determining satisfiability. However, from a logical point of view, satisfiability problems vary along various dimensions, which may affect TLMs' ability to learn how to solve them. The problem instances of satisfiability in natural language can belong to different computational complexity classes depending on the language fragment in which they are expressed. Although prior research has explored the problem of natural language satisfiability, the above-mentioned point has not been discussed adequately. Hence, we investigate how problem instances from varying computational complexity classes and having different grammatical constructs impact TLMs' ability to learn rules of inference. Furthermore, to faithfully evaluate TLMs, we conduct an empirical study to explore the distribution of satisfiability problems. 

**Abstract (ZH)**: 基于变压器的语言模型在自然语言推理问题上的应用取得了逐年增加的成功。在这一领域中，几乎所有的其他任务都可以归结为确定满足性这一最基本的任务。然而，从逻辑学角度来看，满足性问题在多种维度上有所差异，这可能影响基于变压器的语言模型的学习能力。自然语言中满足性问题实例所属的计算复杂性类以及不同的语法结构可能会对基于变压器的语言模型学习推理规则的能力产生影响。此外，为了忠实地评估基于变压器的语言模型，我们进行了一项实证研究，探讨满足性问题的分布情况。 

---
# SACA: Selective Attention-Based Clustering Algorithm 

**Title (ZH)**: 基于选择性注意力的聚类算法 

**Authors**: Meysam Shirdel Bilehsavar, Razieh Ghaedi, Samira Seyed Taheri, Xinqi Fan, Christian O'Reilly  

**Link**: [PDF](https://arxiv.org/pdf/2508.17150)  

**Abstract**: Clustering algorithms are widely used in various applications, with density-based methods such as Density-Based Spatial Clustering of Applications with Noise (DBSCAN) being particularly prominent. These algorithms identify clusters in high-density regions while treating sparser areas as noise. However, reliance on user-defined parameters often poses optimization challenges that require domain expertise. This paper presents a novel density-based clustering method inspired by the concept of selective attention, which minimizes the need for user-defined parameters under standard conditions. Initially, the algorithm operates without requiring user-defined parameters. If parameter adjustment is needed, the method simplifies the process by introducing a single integer parameter that is straightforward to tune. The approach computes a threshold to filter out the most sparsely distributed points and outliers, forms a preliminary cluster structure, and then reintegrates the excluded points to finalize the results. Experimental evaluations on diverse data sets highlight the accessibility and robust performance of the method, providing an effective alternative for density-based clustering tasks. 

**Abstract (ZH)**: 基于选择性注意力的新型密度基于聚类方法：在标准条件下减少用户定义参数的需求 

---
# CE-RS-SBCIT A Novel Channel Enhanced Hybrid CNN Transformer with Residual, Spatial, and Boundary-Aware Learning for Brain Tumor MRI Analysis 

**Title (ZH)**: CE-RS-SBCIT 基于残差、空间和边界aware学习的新型通道增强混合CNN变换器用于脑肿瘤MRI分析 

**Authors**: Mirza Mumtaz Zahoor, Saddam Hussain Khan  

**Link**: [PDF](https://arxiv.org/pdf/2508.17128)  

**Abstract**: Brain tumors remain among the most lethal human diseases, where early detection and accurate classification are critical for effective diagnosis and treatment planning. Although deep learning-based computer-aided diagnostic (CADx) systems have shown remarkable progress. However, conventional convolutional neural networks (CNNs) and Transformers face persistent challenges, including high computational cost, sensitivity to minor contrast variations, structural heterogeneity, and texture inconsistencies in MRI data. Therefore, a novel hybrid framework, CE-RS-SBCIT, is introduced, integrating residual and spatial learning-based CNNs with transformer-driven modules. The proposed framework exploits local fine-grained and global contextual cues through four core innovations: (i) a smoothing and boundary-based CNN-integrated Transformer (SBCIT), (ii) tailored residual and spatial learning CNNs, (iii) a channel enhancement (CE) strategy, and (iv) a novel spatial attention mechanism. The developed SBCIT employs stem convolution and contextual interaction transformer blocks with systematic smoothing and boundary operations, enabling efficient global feature modeling. Moreover, Residual and spatial CNNs, enhanced by auxiliary transfer-learned feature maps, enrich the representation space, while the CE module amplifies discriminative channels and mitigates redundancy. Furthermore, the spatial attention mechanism selectively emphasizes subtle contrast and textural variations across tumor classes. Extensive evaluation on challenging MRI datasets from Kaggle and Figshare, encompassing glioma, meningioma, pituitary tumors, and healthy controls, demonstrates superior performance, achieving 98.30% accuracy, 98.08% sensitivity, 98.25% F1-score, and 98.43% precision. 

**Abstract (ZH)**: 基于脑肿瘤早期检测与准确分类的新型混合框架：CE-RS-SBCIT 

---
# Token Homogenization under Positional Bias 

**Title (ZH)**: 位置偏差下的标记同质化 

**Authors**: Viacheslav Yusupov, Danil Maksimov, Ameliia Alaeva, Tatiana Zaitceva, Antipina Anna, Anna Vasileva, Chenlin Liu, Rayuth Chheng, Danil Sazanakov, Andrey Chetvergov, Alina Ermilova, Egor Shvetsov  

**Link**: [PDF](https://arxiv.org/pdf/2508.17126)  

**Abstract**: This paper investigates token homogenization - the convergence of token representations toward uniformity across transformer layers and its relationship to positional bias in large language models. We empirically examine whether homogenization occurs and how positional bias amplifies this effect. Through layer-wise similarity analysis and controlled experiments, we demonstrate that tokens systematically lose distinctiveness during processing, particularly when biased toward extremal positions. Our findings confirm both the existence of homogenization and its dependence on positional attention mechanisms. 

**Abstract (ZH)**: 本文探讨了令牌同质化——变压器层中令牌表示向均匀性收敛的现象及其与大型语言模型中位置偏见的关系。我们实证研究了令牌同质化是否发生以及位置偏见是如何放大这一效应的。通过逐层相似性分析和受控实验，我们证明了在处理过程中，令牌系统地失去了独特性，尤其是在朝向极端位置偏移时。我们的研究证实了同质化现象的存在及其对位置注意力机制的依赖性。 

---
# PlantVillageVQA: A Visual Question Answering Dataset for Benchmarking Vision-Language Models in Plant Science 

**Title (ZH)**: PlantVillageVQA：植物科学领域视觉问答数据集，用于评估视觉语言模型的基准性能 

**Authors**: Syed Nazmus Sakib, Nafiul Haque, Mohammad Zabed Hossain, Shifat E. Arman  

**Link**: [PDF](https://arxiv.org/pdf/2508.17117)  

**Abstract**: PlantVillageVQA is a large-scale visual question answering (VQA) dataset derived from the widely used PlantVillage image corpus. It was designed to advance the development and evaluation of vision-language models for agricultural decision-making and analysis. The PlantVillageVQA dataset comprises 193,609 high-quality question-answer (QA) pairs grounded over 55,448 images spanning 14 crop species and 38 disease conditions. Questions are organised into 3 levels of cognitive complexity and 9 distinct categories. Each question category was phrased manually following expert guidance and generated via an automated two-stage pipeline: (1) template-based QA synthesis from image metadata and (2) multi-stage linguistic re-engineering. The dataset was iteratively reviewed by domain experts for scientific accuracy and relevancy. The final dataset was evaluated using three state-of-the-art models for quality assessment. Our objective remains to provide a publicly available, standardised and expert-verified database to enhance diagnostic accuracy for plant disease identifications and advance scientific research in the agricultural domain. Our dataset will be open-sourced at this https URL. 

**Abstract (ZH)**: PlantVillageVQA是源自广泛使用的PlantVillage图像库的大规模视觉问答（VQA）数据集，旨在推动农业决策和分析中视觉-语言模型的发展与评估。PlantVillageVQA数据集包含193,609个高质量的问题-答案（QA）对，涵盖55,448张图像中的14种作物和38种病害条件。问题按照3个认知复杂度级别和9个不同的类别组织。每个问题类别都是根据专家指导手工表述并通过自动化两阶段管道生成：（1）基于图像元数据的模板QA合成；（2）多阶段语言重构。该数据集经过领域专家迭代审查以确保科学准确性和相关性。最终数据集使用三个最先进的模型进行质量评估。我们的目标是提供一个公开可用、标准化且经专家验证的数据库，以提高植物病害诊断的准确性并推动农业领域的科学研究。我们的数据集将在以下链接开源：此httpsURL。 

---
# Two Birds with One Stone: Enhancing Uncertainty Quantification and Interpretability with Graph Functional Neural Process 

**Title (ZH)**: 一石二鸟：通过图功能神经过程提高不确定性量化和可解释性 

**Authors**: Lingkai Kong, Haotian Sun, Yuchen Zhuang, Haorui Wang, Wenhao Mu, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17097)  

**Abstract**: Graph neural networks (GNNs) are powerful tools on graph data. However, their predictions are mis-calibrated and lack interpretability, limiting their adoption in critical applications. To address this issue, we propose a new uncertainty-aware and interpretable graph classification model that combines graph functional neural process and graph generative model. The core of our method is to assume a set of latent rationales which can be mapped to a probabilistic embedding space; the predictive distribution of the classifier is conditioned on such rationale embeddings by learning a stochastic correlation matrix. The graph generator serves to decode the graph structure of the rationales from the embedding space for model interpretability. For efficient model training, we adopt an alternating optimization procedure which mimics the well known Expectation-Maximization (EM) algorithm. The proposed method is general and can be applied to any existing GNN architecture. Extensive experiments on five graph classification datasets demonstrate that our framework outperforms state-of-the-art methods in both uncertainty quantification and GNN interpretability. We also conduct case studies to show that the decoded rationale structure can provide meaningful explanations. 

**Abstract (ZH)**: 图神经网络（GNNs）是图数据的强大工具。然而，它们的预测结果缺乏校准且缺乏可解释性，限制了其在关键应用中的采用。为了解决这一问题，我们提出了一种新的集不确定性和可解释性于一体的图分类模型，该模型结合了图函数神经过程和图生成模型。我们的方法的核心是假设一组潜在的理据，这些理据可以映射到概率嵌入空间；分类器的预测分布通过学习一个随机相关矩阵，以这些理据嵌入为条件。图生成器用于从嵌入空间解码理据的图结构以提高模型的可解释性。为了高效训练模型，我们采用了交替优化程序，该程序模仿了著名的期望最大化（EM）算法。所提出的方法是通用的，可以应用于任何现有的GNN架构。在五个图分类数据集上的大量实验表明，我们的框架在不确定性量化和GNN可解释性方面均优于现有最先进的方法。我们还进行了案例研究，以展示解码后的理据结构可以提供有意义的解释。 

---
# Convolutional Neural Networks for Accurate Measurement of Train Speed 

**Title (ZH)**: 基于卷积神经网络的列车速度准确测量 

**Authors**: Haitao Tian, Argyrios Zolotas, Miguel Arana-Catania  

**Link**: [PDF](https://arxiv.org/pdf/2508.17096)  

**Abstract**: In this study, we explore the use of Convolutional Neural Networks for improving train speed estimation accuracy, addressing the complex challenges of modern railway systems. We investigate three CNN architectures - single-branch 2D, single-branch 1D, and multiple-branch models - and compare them with the Adaptive Kalman Filter. We analyse their performance using simulated train operation datasets with and without Wheel Slide Protection activation. Our results reveal that CNN-based approaches, especially the multiple-branch model, demonstrate superior accuracy and robustness compared to traditional methods, particularly under challenging operational conditions. These findings highlight the potential of deep learning techniques to enhance railway safety and operational efficiency by more effectively capturing intricate patterns in complex transportation datasets. 

**Abstract (ZH)**: 本研究探索卷积神经网络在提高列车速度估计准确性方面的应用，以应对现代铁路系统所面临的复杂挑战。我们调查了三种CNN架构——单分支2D、单分支1D和多分支模型——并将它们与自适应卡尔曼滤波器进行比较。我们使用包含和不包含轮滑保护激活的模拟列车运行数据集来分析它们的性能。结果显示，基于CNN的方法，尤其是多分支模型，在准确性与鲁棒性方面优于传统方法，尤其是在复杂操作条件下。这些发现强调了深度学习技术在通过更有效地捕捉复杂运输数据集中的复杂模式来增强铁路安全和操作效率方面的潜力。 

---
# Enhancing Knowledge Tracing through Leakage-Free and Recency-Aware Embeddings 

**Title (ZH)**: 通过泄漏免费和近期意识嵌入增强知识追踪 

**Authors**: Yahya Badran, Christine Preisach  

**Link**: [PDF](https://arxiv.org/pdf/2508.17092)  

**Abstract**: Knowledge Tracing (KT) aims to predict a student's future performance based on their sequence of interactions with learning content. Many KT models rely on knowledge concepts (KCs), which represent the skills required for each item. However, some of these models are vulnerable to label leakage, in which input data inadvertently reveal the correct answer, particularly in datasets with multiple KCs per question.
We propose a straightforward yet effective solution to prevent label leakage by masking ground-truth labels during input embedding construction in cases susceptible to leakage. To accomplish this, we introduce a dedicated MASK label, inspired by masked language modeling (e.g., BERT), to replace ground-truth labels. In addition, we introduce Recency Encoding, which encodes the step-wise distance between the current item and its most recent previous occurrence. This distance is important for modeling learning dynamics such as forgetting, which is a fundamental aspect of human learning, yet it is often overlooked in existing models. Recency Encoding demonstrates improved performance over traditional positional encodings on multiple KT benchmarks.
We show that incorporating our embeddings into KT models like DKT, DKT+, AKT, and SAKT consistently improves prediction accuracy across multiple benchmarks. The approach is both efficient and widely applicable. 

**Abstract (ZH)**: 知识追踪（KT）旨在基于学生与学习内容的交互序列预测其未来表现。许多KT模型依赖于知识概念（KCs），用以表示每个项目所需的能力。然而，这些模型中的一些容易受到标签泄露的影响，在含有多个KC的问题数据集中尤为明显，输入数据会无意中透露正确答案。

我们提出了一种简单有效的方法，通过在可能发生泄露的情况下，在输入嵌入构建过程中屏蔽真实标签来防止标签泄露。为此，我们引入了一个专用的MASK标签，受蒙面语言模型（如BERT）的启发，用于替代真实标签。此外，我们引入了最近性编码，这是一种将当前项目与其最近前一个出现之间的逐步距离进行编码的方法。这个距离对于建模遗忘等学习动态非常重要，而遗忘是人类学习的基本方面，但在现有模型中常被忽略。最近性编码在多个知识追踪基准测试中表现出优于传统位置编码的效果。

我们将我们的嵌入整合到DKT、DKT+、AKT和SAKT等模型中，在多个基准测试上一致提高了预测准确性。该方法既高效又具有广泛适用性。 

---
# Proximal Vision Transformer: Enhancing Feature Representation through Two-Stage Manifold Geometry 

**Title (ZH)**: proximal 视觉变换器：通过两阶段流形几何增强特征表示 

**Authors**: Haoyu Yun, Hamid Krim  

**Link**: [PDF](https://arxiv.org/pdf/2508.17081)  

**Abstract**: The Vision Transformer (ViT) architecture has become widely recognized in computer vision, leveraging its self-attention mechanism to achieve remarkable success across various tasks. Despite its strengths, ViT's optimization remains confined to modeling local relationships within individual images, limiting its ability to capture the global geometric relationships between data points. To address this limitation, this paper proposes a novel framework that integrates ViT with the proximal tools, enabling a unified geometric optimization approach to enhance feature representation and classification performance. In this framework, ViT constructs the tangent bundle of the manifold through its self-attention mechanism, where each attention head corresponds to a tangent space, offering geometric representations from diverse local perspectives. Proximal iterations are then introduced to define sections within the tangent bundle and project data from tangent spaces onto the base space, achieving global feature alignment and optimization. Experimental results confirm that the proposed method outperforms traditional ViT in terms of classification accuracy and data distribution. 

**Abstract (ZH)**: 视觉变换器（ViT）架构在计算机视觉中已广受认可，通过其自注意力机制在各种任务中取得了显著成功。尽管ViT具有这些优势，但其优化仍局限于 modeling individual image 内部的局部关系，限制了其捕捉数据点之间全局几何关系的能力。为解决这一局限，本文提出了一种新型框架，将ViT与近端工具集成，以统一的几何优化方法增强特征表示和分类性能。在该框架中，ViT通过其自注意力机制构建流形的切丛，每个注意力头对应一个切空间，提供来自不同局部视角的几何表示。随后引入近端迭代来定义切丛内的截面，并将数据从切空间投影到基空间，实现全局特征对齐和优化。实验结果证实，所提出的方法在分类准确性和数据分布方面优于传统的ViT。 

---
# Zero-shot Multimodal Document Retrieval via Cross-modal Question Generation 

**Title (ZH)**: 零样本跨模态文档检索_via_跨模态问题生成 

**Authors**: Yejin Choi, Jaewoo Park, Janghan Yoon, Saejin Kim, Jaehyun Jeon, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17079)  

**Abstract**: Rapid advances in Multimodal Large Language Models (MLLMs) have expanded information retrieval beyond purely textual inputs, enabling retrieval from complex real world documents that combine text and visuals. However, most documents are private either owned by individuals or confined within corporate silos and current retrievers struggle when faced with unseen domains or languages. To address this gap, we introduce PREMIR, a simple yet effective framework that leverages the broad knowledge of an MLLM to generate cross modal pre questions (preQs) before retrieval. Unlike earlier multimodal retrievers that compare embeddings in a single vector space, PREMIR leverages preQs from multiple complementary modalities to expand the scope of matching to the token level. Experiments show that PREMIR achieves state of the art performance on out of distribution benchmarks, including closed domain and multilingual settings, outperforming strong baselines across all retrieval metrics. We confirm the contribution of each component through in depth ablation studies, and qualitative analyses of the generated preQs further highlight the model's robustness in real world settings. 

**Abstract (ZH)**: 快速发展的多模态大型语言模型（MLLMs）已将信息检索扩大到超出纯文本输入的范围，使从结合文本和视觉的信息复杂现实文档中检索成为可能。然而，大多数文档是私有的，要么属于个人所有，要么被限制在企业孤岛内，当前的检索器在面对未见领域或语言时遇到困难。为此，我们提出了PREMIR，这是一个简单而有效的框架，利用MLLM的广泛知识在检索前生成跨模态预问题（preQs）。与早期的多模态检索器在单一矢量空间中比较嵌入不同，PREMIR 利用来自多种互补模态的预问题来扩大匹配范围至标记级别。实验证明，PREMIR 在分布外基准测试中，包括封闭领域和多语言设置中，各项检索指标均优于强大基线，表现出色。通过深入的消融研究和生成预问题的定性分析，我们确认了每个组件的贡献，并进一步突显了模型在现实世界设置中的稳健性。 

---
# Linguistic Neuron Overlap Patterns to Facilitate Cross-lingual Transfer on Low-resource Languages 

**Title (ZH)**: 语言神经元重叠模式促进低资源语言的跨语言迁移 

**Authors**: Yuemei Xu, Kexin Xu, Jian Zhou, Ling Hu, Lin Gui  

**Link**: [PDF](https://arxiv.org/pdf/2508.17078)  

**Abstract**: The current Large Language Models (LLMs) face significant challenges in improving performance on low-resource languages and urgently need data-efficient methods without costly fine-tuning. From the perspective of language-bridge, we propose BridgeX-ICL, a simple yet effective method to improve zero-shot Cross-lingual In-Context Learning (X-ICL) for low-resource languages. Unlike existing works focusing on language-specific neurons, BridgeX-ICL explores whether sharing neurons can improve cross-lingual performance in LLMs or not. We construct neuron probe data from the ground-truth MUSE bilingual dictionaries, and define a subset of language overlap neurons accordingly, to ensure full activation of these anchored neurons. Subsequently, we propose an HSIC-based metric to quantify LLMs' internal linguistic spectrum based on overlap neurons, which guides optimal bridge selection. The experiments conducted on 2 cross-lingual tasks and 15 language pairs from 7 diverse families (covering both high-low and moderate-low pairs) validate the effectiveness of BridgeX-ICL and offer empirical insights into the underlying multilingual mechanisms of LLMs. 

**Abstract (ZH)**: 当前的大语言模型在低资源语言上的性能提升面临重大挑战，迫切需要不依赖昂贵微调的数据高效方法。从语言桥梁的视角出发，我们提出BridgeX-ICL，这是一种简单而有效的改进低资源语言零样本跨语言在上下文学习（X-ICL）的方法。不同于现有工作集中于语言特定神经元，BridgeX-ICL 探索共享神经元是否能够提高大语言模型的跨语言性能。我们从 ground-truth MUSE 双语词典构建神经探针数据，并相应地定义了一组语言重叠神经元，以确保这些锚定神经元的全激活。随后，我们提出一种基于 HSIC 的度量来量化大语言模型基于重叠神经元的内部语言频谱，并指导最优桥梁选择。在7个不同语言家族中的2项跨语言任务和15对语言（涵盖高低和中低对）上进行的实验验证了BridgeX-ICL 的有效性，并提供了关于大语言模型多语言机制的实证 Insights。 

---
# Optimizing Neural Networks with Learnable Non-Linear Activation Functions via Lookup-Based FPGA Acceleration 

**Title (ZH)**: 基于查找表的FPGA加速实现可学习非线性激活函数优化神经网络 

**Authors**: Mengyuan Yin, Benjamin Chen Ming Choong, Chuping Qu, Rick Siow Mong Goh, Weng-Fai Wong, Tao Luo  

**Link**: [PDF](https://arxiv.org/pdf/2508.17069)  

**Abstract**: Learned activation functions in models like Kolmogorov-Arnold Networks (KANs) outperform fixed-activation architectures in terms of accuracy and interpretability; however, their computational complexity poses critical challenges for energy-constrained edge AI deployments. Conventional CPUs/GPUs incur prohibitive latency and power costs when evaluating higher order activations, limiting deployability under ultra-tight energy budgets. We address this via a reconfigurable lookup architecture with edge FPGAs. By coupling fine-grained quantization with adaptive lookup tables, our design minimizes energy-intensive arithmetic operations while preserving activation fidelity. FPGA reconfigurability enables dynamic hardware specialization for learned functions, a key advantage for edge systems that require post-deployment adaptability. Evaluations using KANs - where unique activation functions play a critical role - demonstrate that our FPGA-based design achieves superior computational speed and over $10^4$ times higher energy efficiency compared to edge CPUs and GPUs, while maintaining matching accuracy and minimal footprint overhead. This breakthrough positions our approach as a practical enabler for energy-critical edge AI, where computational intensity and power constraints traditionally preclude the use of adaptive activation networks. 

**Abstract (ZH)**: 基于Kolmogorov-Arnold网络的可学习激活函数在准确性和可解释性上优于固定激活函数的模型，但在计算复杂性上对能源受限的边缘AI部署构成了关键挑战。传统的CPU/GPU在评估高阶激活时会导致无法接受的延迟和功耗，限制了在超紧凑能源预算下的可部署性。我们通过在边缘FPGA上实现可重构查找表架构来解决这一问题。结合精细量化和自适应查找表，我们的设计在减少能耗密集型算术运算的同时保留了激活函数的精度。FPGA的可重构性使我们能够动态 specialize 学习到的函数硬件，这是对需要部署后适应性的边缘系统的关键优势。使用KANs的评估表明，与边缘CPU和GPU相比，我们的基于FPGA的设计实现了更优的计算速度和超过$10^4$倍的能耗效率，并且保持了匹配的准确性和最小的占位面积开销。这一突破使我们的方法成为一种实用方案，能够在计算强度和功率约束传统上排除可适应激活网络的能源关键边缘AI中发挥作用。 

---
# SSG-Dit: A Spatial Signal Guided Framework for Controllable Video Generation 

**Title (ZH)**: 基于空间信号引导的可控制视频生成框架：SSG-Dit 

**Authors**: Peng Hu, Yu Gu, Liang Luo, Fuji Ren  

**Link**: [PDF](https://arxiv.org/pdf/2508.17062)  

**Abstract**: Controllable video generation aims to synthesize video content that aligns precisely with user-provided conditions, such as text descriptions and initial images. However, a significant challenge persists in this domain: existing models often struggle to maintain strong semantic consistency, frequently generating videos that deviate from the nuanced details specified in the prompts. To address this issue, we propose SSG-DiT (Spatial Signal Guided Diffusion Transformer), a novel and efficient framework for high-fidelity controllable video generation. Our approach introduces a decoupled two-stage process. The first stage, Spatial Signal Prompting, generates a spatially aware visual prompt by leveraging the rich internal representations of a pre-trained multi-modal model. This prompt, combined with the original text, forms a joint condition that is then injected into a frozen video DiT backbone via our lightweight and parameter-efficient SSG-Adapter. This unique design, featuring a dual-branch attention mechanism, allows the model to simultaneously harness its powerful generative priors while being precisely steered by external spatial signals. Extensive experiments demonstrate that SSG-DiT achieves state-of-the-art performance, outperforming existing models on multiple key metrics in the VBench benchmark, particularly in spatial relationship control and overall consistency. 

**Abstract (ZH)**: 可控视频生成旨在合成与用户提供的条件（如文本描述和初始图像）精确对齐的视频内容。然而，该领域仍面临一个重大挑战：现有模型往往难以保持较强的语义一致性，经常生成与提示中微妙细节不符的视频。为解决这一问题，我们提出了一种新颖且高效的高保真可控视频生成框架SSG-DiT（Spatial Signal Guided Diffusion Transformer）。我们的方法引入了一个解耦的两阶段过程。第一阶段，空间信号提示，通过利用预训练多模态模型的丰富内部表示生成空间意识的视觉提示。该提示与原始文本结合，形成一个联合条件，然后通过我们轻量级且参数高效的SSG-Adapter注入冻结的视频DiT主干中。这种独特设计，包含双分支注意力机制，使模型能够同时利用其强大的生成先验，并受到外部空间信号的精确引导。大量实验表明，SSG-DiT在VBench基准上的多个关键指标上实现了最先进的性能，特别是在空间关系控制和整体一致性方面优于现有模型。 

---
# TabResFlow: A Normalizing Spline Flow Model for Probabilistic Univariate Tabular Regression 

**Title (ZH)**: TabResFlow：一种用于概率单变量表格回归的正则化 spline 流模型 

**Authors**: Kiran Madhusudhanan, Vijaya Krishna Yalavarthi, Jonas Sonntag, Maximilian Stubbemann, Lars Schmidt-Thieme  

**Link**: [PDF](https://arxiv.org/pdf/2508.17056)  

**Abstract**: Tabular regression is a well-studied problem with numerous industrial applications, yet most existing approaches focus on point estimation, often leading to overconfident predictions. This issue is particularly critical in industrial automation, where trustworthy decision-making is essential. Probabilistic regression models address this challenge by modeling prediction uncertainty. However, many conventional methods assume a fixed-shape distribution (typically Gaussian), and resort to estimating distribution parameters. This assumption is often restrictive, as real-world target distributions can be highly complex. To overcome this limitation, we introduce TabResFlow, a Normalizing Spline Flow model designed specifically for univariate tabular regression, where commonly used simple flow networks like RealNVP and Masked Autoregressive Flow (MAF) are unsuitable. TabResFlow consists of three key components: (1) An MLP encoder for each numerical feature. (2) A fully connected ResNet backbone for expressive feature extraction. (3) A conditional spline-based normalizing flow for flexible and tractable density estimation. We evaluate TabResFlow on nine public benchmark datasets, demonstrating that it consistently surpasses existing probabilistic regression models on likelihood scores. Our results demonstrate 9.64% improvement compared to the strongest probabilistic regression model (TreeFlow), and on average 5.6 times speed-up in inference time compared to the strongest deep learning alternative (NodeFlow). Additionally, we validate the practical applicability of TabResFlow in a real-world used car price prediction task under selective regression. To measure performance in this setting, we introduce a novel Area Under Risk Coverage (AURC) metric and show that TabResFlow achieves superior results across this metric. 

**Abstract (ZH)**: 表格回归是一个研究充分且在工业中有广泛应用的问题，但大多数现有方法专注于点估计，常常导致过于自信的预测。在工业自动化中，这种问题尤为关键，因为可靠的决策至关重要。概率回归模型通过建模预测不确定性来应对这一挑战。然而，许多传统方法假设固定形状分布（通常是高斯分布），并依赖于估计分布参数。这一假设往往是限制性的，因为现实世界的目标分布可以非常复杂。为克服这一限制，我们引入了TabResFlow，这是一种专门用于单变量表格回归的规范化插值流模型，常用的简单流网络如RealNVP和掩码自回归流（MAF）在此情况下并不适用。TabResFlow包括三个关键组件：（1）每个数值特征的MLP编码器。（2）完全连接的ResNet骨干网，用于表达性特征提取。（3）条件插值基规范化流，用于灵活且可处理的概率密度估计。我们在九个公开基准数据集上评估了TabResFlow，结果显示它在似然度得分上始终优于现有概率回归模型。我们的结果显示，与最强的概率回归模型（TreeFlow）相比，TabResFlow在该指标上提高了9.64%，并且与最强的深度学习替代品（NodeFlow）相比，平均推理时间快了5.6倍。此外，我们在选定回归的实际二手车价格预测任务中验证了TabResFlow的实用适用性。为了衡量在这种环境下的性能，我们引入了一个新的风险覆盖区域下的面积（Area Under Risk Coverage, AURC）指标，并展示了TabResFlow在该指标上取得了更好的结果。 

---
# An Efficient Dual-Line Decoder Network with Multi-Scale Convolutional Attention for Multi-organ Segmentation 

**Title (ZH)**: 一种用于多器官分割的高效双线解码网络配多尺度卷积注意力机制 

**Authors**: Riad Hassan, M. Rubaiyat Hossain Mondal, Sheikh Iqbal Ahamed, Fahad Mostafa, Md Mostafijur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2508.17007)  

**Abstract**: Proper segmentation of organs-at-risk is important for radiation therapy, surgical planning, and diagnostic decision-making in medical image analysis. While deep learning-based segmentation architectures have made significant progress, they often fail to balance segmentation accuracy with computational efficiency. Most of the current state-of-the-art methods either prioritize performance at the cost of high computational complexity or compromise accuracy for efficiency. This paper addresses this gap by introducing an efficient dual-line decoder segmentation network (EDLDNet). The proposed method features a noisy decoder, which learns to incorporate structured perturbation at training time for better model robustness, yet at inference time only the noise-free decoder is executed, leading to lower computational cost. Multi-Scale convolutional Attention Modules (MSCAMs), Attention Gates (AGs), and Up-Convolution Blocks (UCBs) are further utilized to optimize feature representation and boost segmentation performance. By leveraging multi-scale segmentation masks from both decoders, we also utilize a mutation-based loss function to enhance the model's generalization. Our approach outperforms SOTA segmentation architectures on four publicly available medical imaging datasets. EDLDNet achieves SOTA performance with an 84.00% Dice score on the Synapse dataset, surpassing baseline model like UNet by 13.89% in Dice score while significantly reducing Multiply-Accumulate Operations (MACs) by 89.7%. Compared to recent approaches like EMCAD, our EDLDNet not only achieves higher Dice score but also maintains comparable computational efficiency. The outstanding performance across diverse datasets establishes EDLDNet's strong generalization, computational efficiency, and robustness. The source code, pre-processed data, and pre-trained weights will be available at this https URL . 

**Abstract (ZH)**: 适当的器官-at-风险分割对于放射治疗、手术规划和医学图像分析中的诊断决策至关重要。尽管基于深度学习的分割架构取得了显著进展，但它们往往难以同时平衡分割准确性与计算效率。目前大多数最先进的方法要么以高计算复杂度为代价优先考虑性能，要么牺牲准确性以提高效率。本文通过引入高效的双重解码器分割网络（EDLDNet）来解决这一问题。所提出的方法具有一个噪声解码器，该噪声解码器在训练时学习引入结构化的扰动以提高模型的鲁棒性，但在推断时仅执行无噪声解码器，从而降低了计算成本。此外，还利用多尺度卷积注意力模块（MSCAMs）、注意力门（AGs）和上采样卷积块（UCBs）来优化特征表示并提升分割性能。通过利用两个解码器的多尺度分割掩码，还利用基于突变的损失函数来增强模型的泛化能力。我们的方法在四个公开的医学成像数据集上优于当前最先进的分割架构。EDLDNet在Synapse数据集上的Dice得分为84.00%，Dice得分比Baseline模型如UNet高13.89%，同时显著减少了89.7%的乘积累加操作数（MACs）。与近期方法EMCAD相比，除了Dice得分更高外，EDLDNet还能保持相当的计算效率。跨不同数据集的出色性能验证了EDLDNet的强大泛化能力、计算效率和鲁棒性。源代码、预处理数据和预训练权重将在此处提供。 

---
# GRADE: Generating multi-hop QA and fine-gRAined Difficulty matrix for RAG Evaluation 

**Title (ZH)**: GRADE: 生成多跳问答和细粒度难度矩阵以评估语境关联检索系统 

**Authors**: Jeongsoo Lee, Daeyong Kwon, Kyohoon Jin  

**Link**: [PDF](https://arxiv.org/pdf/2508.16994)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are widely adopted in knowledge-intensive NLP tasks, but current evaluations often overlook the structural complexity and multi-step reasoning required in real-world scenarios. These benchmarks overlook key factors such as the interaction between retrieval difficulty and reasoning depth. To address this gap, we propose \textsc{GRADE}, a novel evaluation framework that models task difficulty along two orthogonal dimensions: (1) reasoning depth, defined by the number of inference steps (hops), and (2) semantic distance between the query and its supporting evidence. We construct a synthetic multi-hop QA dataset from factual news articles by extracting knowledge graphs and augmenting them through semantic clustering to recover missing links, allowing us to generate diverse and difficulty-controlled queries. Central to our framework is a 2D difficulty matrix that combines generator-side and retriever-side difficulty. Experiments across multiple domains and models show that error rates strongly correlate with our difficulty measures, validating their diagnostic utility. \textsc{GRADE} enables fine-grained analysis of RAG performance and provides a scalable foundation for evaluating and improving multi-hop reasoning in real-world applications. 

**Abstract (ZH)**: GRADE：一种新颖的多跳推理评估框架 

---
# Score Matching on Large Geometric Graphs for Cosmology Generation 

**Title (ZH)**: Large几何图上得分匹配生成 cosmology 

**Authors**: Diana-Alexandra Onutu, Yue Zhao, Joaquin Vanschoren, Vlado Menkovski  

**Link**: [PDF](https://arxiv.org/pdf/2508.16990)  

**Abstract**: Generative models are a promising tool to produce cosmological simulations but face significant challenges in scalability, physical consistency, and adherence to domain symmetries, limiting their utility as alternatives to $N$-body simulations. To address these limitations, we introduce a score-based generative model with an equivariant graph neural network that simulates gravitational clustering of galaxies across cosmologies starting from an informed prior, respects periodic boundaries, and scales to full galaxy counts in simulations. A novel topology-aware noise schedule, crucial for large geometric graphs, is introduced. The proposed equivariant score-based model successfully generates full-scale cosmological point clouds of up to 600,000 halos, respects periodicity and a uniform prior, and outperforms existing diffusion models in capturing clustering statistics while offering significant computational advantages. This work advances cosmology by introducing a generative model designed to closely resemble the underlying gravitational clustering of structure formation, moving closer to physically realistic and efficient simulators for the evolution of large-scale structures in the universe. 

**Abstract (ZH)**: 基于图神经网络的等变评分生成模型在宇宙学模拟中的应用 

---
# ReFactX: Scalable Reasoning with Reliable Facts via Constrained Generation 

**Title (ZH)**: ReFactX：通过受限生成实现的可扩展可靠事实推理 

**Authors**: Riccardo Pozzi, Matteo Palmonari, Andrea Coletta, Luigi Bellomarini, Jens Lehmann, Sahar Vahdati  

**Link**: [PDF](https://arxiv.org/pdf/2508.16983)  

**Abstract**: Knowledge gaps and hallucinations are persistent challenges for Large Language Models (LLMs), which generate unreliable responses when lacking the necessary information to fulfill user instructions. Existing approaches, such as Retrieval-Augmented Generation (RAG) and tool use, aim to address these issues by incorporating external knowledge. Yet, they rely on additional models or services, resulting in complex pipelines, potential error propagation, and often requiring the model to process a large number of tokens. In this paper, we present a scalable method that enables LLMs to access external knowledge without depending on retrievers or auxiliary models. Our approach uses constrained generation with a pre-built prefix-tree index. Triples from a Knowledge Graph are verbalized in textual facts, tokenized, and indexed in a prefix tree for efficient access. During inference, to acquire external knowledge, the LLM generates facts with constrained generation which allows only sequences of tokens that form an existing fact. We evaluate our proposal on Question Answering and show that it scales to large knowledge bases (800 million facts), adapts to domain-specific data, and achieves effective results. These gains come with minimal generation-time overhead. ReFactX code is available at this https URL. 

**Abstract (ZH)**: 大规模语言模型的知识缺口和幻觉是持续性的挑战，这些模型在缺乏必要信息以完成用户指令时会产生不可靠的响应。现有的方法，如检索增强生成（RAG）和工具使用，希望通过引入外部知识来解决这些问题。然而，这些方法依赖于额外的模型或服务，导致复杂的工作流程、潜在的错误传播，并且通常需要模型处理大量token。本文提出了一种可扩展的方法，使大规模语言模型能够访问外部知识而不依赖于检索器或辅助模型。该方法使用受约束的生成，并结合预先构建的前缀树索引。知识图中的三元组被转换为文本事实，分词并索引到前缀树中，以便高效访问。推理时，为了获取外部知识，大模型使用受约束生成来生成现有事实的token序列。我们通过问答任务评估了该方法，结果显示它能够扩展到大规模知识库（8亿条事实），适应特定领域的数据，并取得有效结果。这些收益伴随着最小的生成时间开销。ReFactX代码可在以下链接获取。 

---
# Combating Digitally Altered Images: Deepfake Detection 

**Title (ZH)**: 对抗数字化篡改图像：深度伪造检测 

**Authors**: Saksham Kumar, Rhythm Narang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16975)  

**Abstract**: The rise of Deepfake technology to generate hyper-realistic manipulated images and videos poses a significant challenge to the public and relevant authorities. This study presents a robust Deepfake detection based on a modified Vision Transformer(ViT) model, trained to distinguish between real and Deepfake images. The model has been trained on a subset of the OpenForensics Dataset with multiple augmentation techniques to increase robustness for diverse image manipulations. The class imbalance issues are handled by oversampling and a train-validation split of the dataset in a stratified manner. Performance is evaluated using the accuracy metric on the training and testing datasets, followed by a prediction score on a random image of people, irrespective of their realness. The model demonstrates state-of-the-art results on the test dataset to meticulously detect Deepfake images. 

**Abstract (ZH)**: Deepfake技术的兴起对生成超现实 manipulated 图像和视频构成了重大挑战：一种基于修改后的视觉变换器（ViT）模型的稳健Deepfake检测方法 

---
# Explaining Black-box Language Models with Knowledge Probing Systems: A Post-hoc Explanation Perspective 

**Title (ZH)**: 用知识探查系统解释黑盒语言模型：一种事后解释视角 

**Authors**: Yunxiao Zhao, Hao Xu, Zhiqiang Wang, Xiaoli Li, Jiye Liang, Ru Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.16969)  

**Abstract**: Pre-trained Language Models (PLMs) are trained on large amounts of unlabeled data, yet they exhibit remarkable reasoning skills. However, the trustworthiness challenges posed by these black-box models have become increasingly evident in recent years. To alleviate this problem, this paper proposes a novel Knowledge-guided Probing approach called KnowProb in a post-hoc explanation way, which aims to probe whether black-box PLMs understand implicit knowledge beyond the given text, rather than focusing only on the surface level content of the text. We provide six potential explanations derived from the underlying content of the given text, including three knowledge-based understanding and three association-based reasoning. In experiments, we validate that current small-scale (or large-scale) PLMs only learn a single distribution of representation, and still face significant challenges in capturing the hidden knowledge behind a given text. Furthermore, we demonstrate that our proposed approach is effective for identifying the limitations of existing black-box models from multiple probing perspectives, which facilitates researchers to promote the study of detecting black-box models in an explainable way. 

**Abstract (ZH)**: 预训练语言模型（PLMs）在大量未标记数据上进行训练，但却表现出卓越的推理能力。然而，这些黑盒模型带来的可信性挑战在近年来越来越明显。为缓解这一问题，本文提出了一种新的知识导向探针方法——KnowProb，该方法以事后解释的方式探究黑盒PLMs是否理解了文本背后隐含的知识，而不仅仅是关注文本表面内容。我们从给定文本的基础内容中提供了六种可能的解释，包括三种基于知识的理解和三种基于关联的推理。在实验中，我们验证了当前的小规模（或大规模）PLMs仅学习了单一表示分布，并在捕捉给定文本背后的隐藏知识方面仍然面临重大挑战。此外，我们证明了所提出的方法可以从多个探针视角有效识别现有黑盒模型的局限性，从而促进研究人员以可解释的方式推进对黑盒模型检测的研究。 

---
# LLM-based Human-like Traffic Simulation for Self-driving Tests 

**Title (ZH)**: 基于LLM的人类-like交通仿真用于自动驾驶测试 

**Authors**: Wendi Li, Hao Wu, Han Gao, Bing Mao, Fengyuan Xu, Sheng Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2508.16962)  

**Abstract**: Ensuring realistic traffic dynamics is a prerequisite for simulation platforms to evaluate the reliability of self-driving systems before deployment in the real world. Because most road users are human drivers, reproducing their diverse behaviors within simulators is vital. Existing solutions, however, typically rely on either handcrafted heuristics or narrow data-driven models, which capture only fragments of real driving behaviors and offer limited driving style diversity and interpretability. To address this gap, we introduce HDSim, an HD traffic generation framework that combines cognitive theory with large language model (LLM) assistance to produce scalable and realistic traffic scenarios within simulation platforms. The framework advances the state of the art in two ways: (i) it introduces a hierarchical driver model that represents diverse driving style traits, and (ii) it develops a Perception-Mediated Behavior Influence strategy, where LLMs guide perception to indirectly shape driver actions. Experiments reveal that embedding HDSim into simulation improves detection of safety-critical failures in self-driving systems by up to 68% and yields realism-consistent accident interpretability. 

**Abstract (ZH)**: 确保真实的交通动态是评估自动驾驶系统可靠性并在实际世界部署前使用模拟平台的先决条件。由于大多数道路使用者是人类驾驶员，在模拟器中再现其多样行为至关重要。现有解决方案通常依赖于手工编写的启发式方法或窄数据驱动模型，这些方法只能捕捉真实驾驶行为的一小部分，且提供有限的驾驶风格多样性和可解释性。为了填补这一空白，我们引入了HDSim，这是一个结合认知理论和大型语言模型（LLM）协助的高清交通生成框架，可在模拟平台中生成可扩展且逼真的交通场景。该框架在两个方面推动了现有技术的发展：（i）引入了层次化的驾驶员模型以表示多样化的驾驶风格特征，（ii）开发了感知中介的行为影响策略，其中LLM指导感知以间接塑造驾驶员行为。实验表明，将HDSim嵌入模拟可以将自动驾驶系统中关键安全故障的检测率提高68%，并提供与现实一致的事故可解释性。 

---
# Breaking the Exploration Bottleneck: Rubric-Scaffolded Reinforcement Learning for General LLM Reasoning 

**Title (ZH)**: 打破探索瓶颈：基于评价框架的强化学习在通用大语言模型推理中的应用 

**Authors**: Yang Zhou, Sunzhu Li, Shunyu Liu, Wenkai Fang, Jiale Zhao, Jingwen Yang, Jianwei Lv, Kongcheng Zhang, Yihe Zhou, Hengtong Lu, Wei Chen, Yan Xie, Mingli Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.16949)  

**Abstract**: Recent advances in Large Language Models (LLMs) have underscored the potential of Reinforcement Learning (RL) to facilitate the emergence of reasoning capabilities. Despite the encouraging results, a fundamental dilemma persists as RL improvement relies on learning from high-quality samples, yet the exploration for such samples remains bounded by the inherent limitations of LLMs. This, in effect, creates an undesirable cycle in which what cannot be explored cannot be learned. In this work, we propose Rubric-Scaffolded Reinforcement Learning (RuscaRL), a novel instructional scaffolding framework designed to break the exploration bottleneck for general LLM reasoning. Specifically, RuscaRL introduces checklist-style rubrics as (1) explicit scaffolding for exploration during rollout generation, where different rubrics are provided as external guidance within task instructions to steer diverse high-quality responses. This guidance is gradually decayed over time, encouraging the model to internalize the underlying reasoning patterns; (2) verifiable rewards for exploitation during model training, where we can obtain robust LLM-as-a-Judge scores using rubrics as references, enabling effective RL on general reasoning tasks. Extensive experiments demonstrate the superiority of the proposed RuscaRL across various benchmarks, effectively expanding reasoning boundaries under the best-of-N evaluation. Notably, RuscaRL significantly boosts Qwen-2.5-7B-Instruct from 23.6 to 50.3 on HealthBench-500, surpassing GPT-4.1. Furthermore, our fine-tuned variant on Qwen3-30B-A3B-Instruct achieves 61.1 on HealthBench-500, outperforming leading LLMs including OpenAI-o3. 

**Abstract (ZH)**: Recent Advances in Large Language Models (LLMs): Rubric-Scaffolded Reinforcement Learning (RuscaRL) for Facilitating General Reasoning Capabilities 

---
# Drive As You Like: Strategy-Level Motion Planning Based on A Multi-Head Diffusion Model 

**Title (ZH)**: 随心驾驶：基于多头扩散模型的策略级运动规划 

**Authors**: Fan Ding, Xuewen Luo, Hwa Hui Tew, Ruturaj Reddy, Xikun Wang, Junn Yong Loo  

**Link**: [PDF](https://arxiv.org/pdf/2508.16947)  

**Abstract**: Recent advances in motion planning for autonomous driving have led to models capable of generating high-quality trajectories. However, most existing planners tend to fix their policy after supervised training, leading to consistent but rigid driving behaviors. This limits their ability to reflect human preferences or adapt to dynamic, instruction-driven demands. In this work, we propose a diffusion-based multi-head trajectory planner(M-diffusion planner). During the early training stage, all output heads share weights to learn to generate high-quality trajectories. Leveraging the probabilistic nature of diffusion models, we then apply Group Relative Policy Optimization (GRPO) to fine-tune the pre-trained model for diverse policy-specific behaviors. At inference time, we incorporate a large language model (LLM) to guide strategy selection, enabling dynamic, instruction-aware planning without switching models. Closed-loop simulation demonstrates that our post-trained planner retains strong planning capability while achieving state-of-the-art (SOTA) performance on the nuPlan val14 benchmark. Open-loop results further show that the generated trajectories exhibit clear diversity, effectively satisfying multi-modal driving behavior requirements. The code and related experiments will be released upon acceptance of the paper. 

**Abstract (ZH)**: 近年来，自主驾驶中的运动规划进展使得能够生成高质量轨迹的模型得以实现。然而，现有的大多数规划器在监督训练后倾向于固定其策略，导致一致但僵化的驾驶行为。这限制了它们反映人类偏好或适应动态、指令驱动需求的能力。在本文中，我们提出了一种基于扩散的多头轨迹规划器（M-diffusion planner）。在训练的早期阶段，所有输出头共享权重以学习生成高质量的轨迹。利用扩散模型的概率性质，我们随后应用组相对策略优化（GRPO）对预训练模型进行微调，以获得多样化的策略特定行为。在推理阶段，我们引入了一个大型语言模型（LLM）来引导策略选择，从而实现动态、指令感知的规划，而无需切换模型。闭环仿真结果表明，我们的后训练规划器保持了强大的规划能力，并在nuPlan val14基准测试中实现了最先进的性能。开环结果进一步表明，生成的轨迹表现出明显的多样性，有效地满足了多模态驾驶行为的要求。论文被接受后，代码及相关实验将公开。 

---
# HumanoidVerse: A Versatile Humanoid for Vision-Language Guided Multi-Object Rearrangement 

**Title (ZH)**: HumanoidVerse: 适用于视觉-语言引导多物体重排的多功能类人型机器人 

**Authors**: Haozhuo Zhang, Jingkai Sun, Michele Caprio, Jian Tang, Shanghang Zhang, Qiang Zhang, Wei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2508.16943)  

**Abstract**: We introduce HumanoidVerse, a novel framework for vision-language guided humanoid control that enables a single physically simulated robot to perform long-horizon, multi-object rearrangement tasks across diverse scenes. Unlike prior methods that operate in fixed settings with single-object interactions, our approach supports consecutive manipulation of multiple objects, guided only by natural language instructions and egocentric camera RGB observations. HumanoidVerse is trained via a multi-stage curriculum using a dual-teacher distillation pipeline, enabling fluid transitions between sub-tasks without requiring environment resets. To support this, we construct a large-scale dataset comprising 350 multi-object tasks spanning four room layouts. Extensive experiments in the Isaac Gym simulator demonstrate that our method significantly outperforms prior state-of-the-art in both task success rate and spatial precision, and generalizes well to unseen environments and instructions. Our work represents a key step toward robust, general-purpose humanoid agents capable of executing complex, sequential tasks under real-world sensory constraints. The video visualization results can be found on the project page: this https URL. 

**Abstract (ZH)**: 我们介绍了HumanoidVerse：一种新的基于视觉语言引导的人形机器人控制框架，使单个物理模拟机器人能够在多样化场景中执行长时 horizon、多对象重组任务。不同于先前在固定环境中仅支持单一对象交互的方法，我们的方法仅通过自然语言指令和第一人称摄像头RGB观察来支持连续操纵多个对象。HumanoidVerse通过一个多阶段课程采用双教师蒸馏管道进行训练，能够在无需重置环境的情况下流畅地在子任务之间进行转换。为此，我们构建了一个包含350个跨四种房间布局的多对象任务的大规模数据集。在Isaac Gym模拟器中的广泛实验表明，我们的方法在任务成功率和空间精度上都优于先前的最先进方法，并且能够很好地泛化到未见过的环境和指令。我们的工作是朝着具备在现实世界传感约束下执行复杂序列任务的稳健且通用的人形代理迈出的关键一步。该项目的视频可视化结果可在项目页面上找到：this https URL。 

---
# THEME : Enhancing Thematic Investing with Semantic Stock Representations and Temporal Dynamics 

**Title (ZH)**: 主题：通过语义股票表示和时间动态提升主题投资 

**Authors**: Hoyoung Lee, Wonbin Ahn, Suhwan Park, Jaehoon Lee, Minjae Kim, Sungdong Yoo, Taeyoon Lim, Woohyung Lim, Yongjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.16936)  

**Abstract**: Thematic investing aims to construct portfolios aligned with structural trends, yet selecting relevant stocks remains challenging due to overlapping sector boundaries and evolving market dynamics. To address this challenge, we construct the Thematic Representation Set (TRS), an extended dataset that begins with real-world thematic ETFs and expands upon them by incorporating industry classifications and financial news to overcome their coverage limitations. The final dataset contains both the explicit mapping of themes to their constituent stocks and the rich textual profiles for each. Building on this dataset, we introduce \textsc{THEME}, a hierarchical contrastive learning framework. By representing the textual profiles of themes and stocks as embeddings, \textsc{THEME} first leverages their hierarchical relationship to achieve semantic alignment. Subsequently, it refines these semantic embeddings through a temporal refinement stage that incorporates individual stock returns. The final stock representations are designed for effective retrieval of thematically aligned assets with strong return potential. Empirical results show that \textsc{THEME} outperforms strong baselines across multiple retrieval metrics and significantly improves performance in portfolio construction. By jointly modeling thematic relationships from text and market dynamics from returns, \textsc{THEME} provides a scalable and adaptive solution for navigating complex investment themes. 

**Abstract (ZH)**: 主题投资旨在构建与结构性趋势相一致的投资组合，但由于行业边界重叠和市场动态变化，选择相关的股票仍然具有挑战性。为应对这一挑战，我们构建了主题表示集（TRS），该扩展数据集以现实世界的主题ETF为起点，并通过纳入行业分类和财务新闻来克服其覆盖面的限制。最终数据集既包括主题与组成股票的显式映射，也包括每种主题的丰富文本概况。在此数据集的基础上，我们引入了THEME分层对比学习框架。通过将主题和股票的文本概况表示为嵌入，THEME首先利用它们的分层关系实现语义对齐。随后，通过结合单一股票回报的时序精细校正阶段来细化这些语义嵌入。最终的股票表示旨在有效检索具有强烈回报潜力的主题对齐资产。实证结果表明，THEME在多个检索指标上优于强 baseline，并显著提高了投资组合构建性能。通过联合建模来自文本的主题关系和来自回报的市场动态，THEME提供了一种可扩展且适应性强的解决方案，以应对复杂的投资主题。 

---
# Degree of Staleness-Aware Data Updating in Federated Learning 

**Title (ZH)**: staleness感知的数据更新在联邦学习中的程度aware机制 

**Authors**: Tao Liu, Xuehe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16931)  

**Abstract**: Handling data staleness remains a significant challenge in federated learning with highly time-sensitive tasks, where data is generated continuously and data staleness largely affects model performance. Although recent works attempt to optimize data staleness by determining local data update frequency or client selection strategy, none of them explore taking both data staleness and data volume into consideration. In this paper, we propose DUFL(Data Updating in Federated Learning), an incentive mechanism featuring an innovative local data update scheme manipulated by three knobs: the server's payment, outdated data conservation rate, and clients' fresh data collection volume, to coordinate staleness and volume of local data for best utilities. To this end, we introduce a novel metric called DoS(the Degree of Staleness) to quantify data staleness and conduct a theoretic analysis illustrating the quantitative relationship between DoS and model performance. We model DUFL as a two-stage Stackelberg game with dynamic constraint, deriving the optimal local data update strategy for each client in closed-form and the approximately optimal strategy for the server. Experimental results on real-world datasets demonstrate the significant performance of our approach. 

**Abstract (ZH)**: 处理数据过时仍然是联邦学习中具有时间敏感任务时的一个重大挑战，其中数据连续生成且数据过时大大影响模型性能。尽管近期工作尝试通过确定局部数据更新频率或客户端选择策略来优化数据过时，但它们都没有同时考虑数据过时和数据量。在本文中，我们提出了DUFL（联邦学习中的数据更新机制），这是一种新颖的基于三个调节器（服务器支付、过时数据保留率和客户端新鲜数据收集量）控制的局部数据更新方案，以协调局部数据的过时和数量，实现最优效用。为此，我们引入了一个新的度量标准DoS（数据过时程度）来量化数据过时，并进行理论分析以说明DoS与模型性能之间的量化关系。我们将DUFL建模为具有动态约束的两阶段Stackelberg博弈，推导出了每个客户端的闭式最佳局部数据更新策略和服务器的近似最优策略。实验结果表明，我们的方法具有显著的性能优势。 

---
# TextOnly: A Unified Function Portal for Text-Related Functions on Smartphones 

**Title (ZH)**: 文本唯一门户：智能手机上与文本相关功能的统一平台 

**Authors**: Minghao Tu, Chun Yu, Xiyuan Shen, Zhi Zheng, Li Chen, Yuanchun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.16926)  

**Abstract**: Text boxes serve as portals to diverse functionalities in today's smartphone applications. However, when it comes to specific functionalities, users always need to navigate through multiple steps to access particular text boxes for input. We propose TextOnly, a unified function portal that enables users to access text-related functions from various applications by simply inputting text into a sole text box. For instance, entering a restaurant name could trigger a Google Maps search, while a greeting could initiate a conversation in WhatsApp. Despite their brevity, TextOnly maximizes the utilization of these raw text inputs, which contain rich information, to interpret user intentions effectively. TextOnly integrates large language models(LLM) and a BERT model. The LLM consistently provides general knowledge, while the BERT model can continuously learn user-specific preferences and enable quicker predictions. Real-world user studies demonstrated TextOnly's effectiveness with a top-1 accuracy of 71.35%, and its ability to continuously improve both its accuracy and inference speed. Participants perceived TextOnly as having satisfactory usability and expressed a preference for TextOnly over manual executions. Compared with voice assistants, TextOnly supports a greater range of text-related functions and allows for more concise inputs. 

**Abstract (ZH)**: TextOnly：统一的文本功能门户 

---
# Tri-Accel: Curvature-Aware Precision-Adaptive and Memory-Elastic Optimization for Efficient GPU Usage 

**Title (ZH)**: Tri-Accel: 曲率意识的精度自适应和内存弹性优化以提高GPU使用效率 

**Authors**: Mohsen Sheibanian, Pouya Shaeri, Alimohammad Beigi, Ryan T. Woo, Aryan Keluskar  

**Link**: [PDF](https://arxiv.org/pdf/2508.16905)  

**Abstract**: Deep neural networks are increasingly bottlenecked by the cost of optimization, both in terms of GPU memory and compute time. Existing acceleration techniques, such as mixed precision, second-order methods, and batch size scaling, are typically used in isolation. We present Tri-Accel, a unified optimization framework that co-adapts three acceleration strategies along with adaptive parameters during training: (1) Precision-Adaptive Updates that dynamically assign mixed-precision levels to layers based on curvature and gradient variance; (2) Sparse Second-Order Signals that exploit Hessian/Fisher sparsity patterns to guide precision and step size decisions; and (3) Memory-Elastic Batch Scaling that adjusts batch size in real time according to VRAM availability. On CIFAR-10 with ResNet-18 and EfficientNet-B0, Tri-Accel achieves up to 9.9% reduction in training time and 13.3% lower memory usage, while improving accuracy by +1.1 percentage points over FP32 baselines. Tested on CIFAR-10/100, our approach demonstrates adaptive learning behavior, with efficiency gradually improving over the course of training as the system learns to allocate resources more effectively. Compared to static mixed-precision training, Tri-Accel maintains 78.1% accuracy while reducing memory footprint from 0.35GB to 0.31GB on standard hardware. The framework is implemented with custom Triton kernels, whose hardware-aware adaptation enables automatic optimization without manual hyperparameter tuning, making it practical for deployment across diverse computational environments. This work demonstrates how algorithmic adaptivity and hardware awareness can be combined to improve scalability in resource-constrained settings, paving the way for more efficient neural network training on edge devices and cost-sensitive cloud deployments. 

**Abstract (ZH)**: 深层神经网络的优化日益受到GPU内存和计算时间成本的限制。现有的加速技术，如混合精度、二次方法和批量大小放大，通常单独使用。我们提出了Tri-Accel，这是一种统一的优化框架，在训练过程中协同调整三种加速策略及其自适应参数：（1）精度自适应更新，根据曲率和梯度方差动态为各层分配混合精度级别；（2）稀疏二次信号，利用海森矩阵/鱼类子的稀疏模式来指导精度和步长的决策；（3）弹性批量大小调整，根据VRAM可用性实时调整批量大小。在CIFAR-10上使用ResNet-18和EfficientNet-B0，Tri-Accel实现了高达9.9%的训练时间减少和13.3%的更低内存使用，同时在FP32基线基础上提高准确率1.1个百分点。在CIFAR-10/100上测试时，该方法显示自适应学习行为，随着系统在训练过程中学习更有效地分配资源，效率逐渐提高。与静态混合精度训练相比，Tri-Accel在标准硬件上将内存占用从0.35GB减少到0.31GB的同时保持78.1%的准确性。该框架使用自定义的Triton内核实现，其硬件感知的自适应性能够实现自动优化，无需手动调整超参数，使其能够在多样化的计算环境中得到更广泛的应用。这项工作展示了如何将算法自适应性和硬件感知性相结合，以改善资源受限环境中的可扩展性，为边缘设备和成本敏感的云端部署高效神经网络训练铺平道路。 

---
# Dream to Chat: Model-based Reinforcement Learning on Dialogues with User Belief Modeling 

**Title (ZH)**: 梦境对话：基于模型的对话中用户信念建模增强学习 

**Authors**: Yue Zhao, Xiaoyu Wang, Dan Wang, Zhonglin Jiang, Qingqing Gu, Teng Chen, Ningyuan Xi, Jinxian Qu, Yong Chen, Luo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2508.16876)  

**Abstract**: World models have been widely utilized in robotics, gaming, and auto-driving. However, their applications on natural language tasks are relatively limited. In this paper, we construct the dialogue world model, which could predict the user's emotion, sentiment, and intention, and future utterances. By defining a POMDP, we argue emotion, sentiment and intention can be modeled as the user belief and solved by maximizing the information bottleneck. By this user belief modeling, we apply the model-based reinforcement learning framework to the dialogue system, and propose a framework called DreamCUB. Experiments show that the pretrained dialogue world model can achieve state-of-the-art performances on emotion classification and sentiment identification, while dialogue quality is also enhanced by joint training of the policy, critic and dialogue world model. Further analysis shows that this manner holds a reasonable exploration-exploitation balance and also transfers well to out-of-domain scenarios such as empathetic dialogues. 

**Abstract (ZH)**: 世界的对话模型已经在机器人、游戏和自动驾驶等领域得到了广泛应用。然而，它们在自然语言任务中的应用相对有限。本文构建了对话世界模型，能够预测用户的情绪、情感、意图以及未来的话语。通过定义POMDP，我们认为情绪、情感和意图可以被建模为用户信念，并通过最大化信息瓶颈来解决。基于用户信念建模，我们将基于模型的强化学习框架应用于对话系统，并提出了一种名为DreamCUB的框架。实验表明，预训练的对话世界模型在情绪分类和情感识别上取得了最先进的性能，同时联合训练策略、评论家和对话世界模型也提升了对话质量。进一步的分析表明，该方法在包括共情对话在内的跨域场景中表现出了合理的探索-利用平衡，并且具有较强的泛化能力。 

---
# TriagerX: Dual Transformers for Bug Triaging Tasks with Content and Interaction Based Rankings 

**Title (ZH)**: TriagerX: 基于内容和交互的双Transformer漏洞 triaging 任务排序方法 

**Authors**: Md Afif Al Mamun, Gias Uddin, Lan Xia, Longyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16860)  

**Abstract**: Pretrained Language Models or PLMs are transformer-based architectures that can be used in bug triaging tasks. PLMs can better capture token semantics than traditional Machine Learning (ML) models that rely on statistical features (e.g., TF-IDF, bag of words). However, PLMs may still attend to less relevant tokens in a bug report, which can impact their effectiveness. In addition, the model can be sub-optimal with its recommendations when the interaction history of developers around similar bugs is not taken into account. We designed TriagerX to address these limitations. First, to assess token semantics more reliably, we leverage a dual-transformer architecture. Unlike current state-of-the-art (SOTA) baselines that employ a single transformer architecture, TriagerX collects recommendations from two transformers with each offering recommendations via its last three layers. This setup generates a robust content-based ranking of candidate developers. TriagerX then refines this ranking by employing a novel interaction-based ranking methodology, which considers developers' historical interactions with similar fixed bugs. Across five datasets, TriagerX surpasses all nine transformer-based methods, including SOTA baselines, often improving Top-1 and Top-3 developer recommendation accuracy by over 10%. We worked with our large industry partner to successfully deploy TriagerX in their development environment. The partner required both developer and component recommendations, with components acting as proxies for team assignments-particularly useful in cases of developer turnover or team changes. We trained TriagerX on the partner's dataset for both tasks, and it outperformed SOTA baselines by up to 10% for component recommendations and 54% for developer recommendations. 

**Abstract (ZH)**: 预训练语言模型在软件 bug triaging 任务中的应用：TriagerX的设计与优化 

---
# WildSpoof Challenge Evaluation Plan 

**Title (ZH)**: WildSpoof挑战评估计划 

**Authors**: Yihan Wu, Jee-weon Jung, Hye-jin Shim, Xin Cheng, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16858)  

**Abstract**: The WildSpoof Challenge aims to advance the use of in-the-wild data in two intertwined speech processing tasks. It consists of two parallel tracks: (1) Text-to-Speech (TTS) synthesis for generating spoofed speech, and (2) Spoofing-robust Automatic Speaker Verification (SASV) for detecting spoofed speech. While the organizers coordinate both tracks and define the data protocols, participants treat them as separate and independent tasks. The primary objectives of the challenge are: (i) to promote the use of in-the-wild data for both TTS and SASV, moving beyond conventional clean and controlled datasets and considering real-world scenarios; and (ii) to encourage interdisciplinary collaboration between the spoofing generation (TTS) and spoofing detection (SASV) communities, thereby fostering the development of more integrated, robust, and realistic systems. 

**Abstract (ZH)**: WildSpoof挑战旨在推进野生数据在两个相互关联的语音处理任务中的应用。该挑战包含两个并行赛道：(1) 从文本到语音(TTS)合成以生成欺骗性语音，以及(2) 抗欺骗性的自动说话人验证(SASV)以检测欺骗性语音。尽管组织者协调这两个赛道并定义数据协议，但参与者将它们视为独立任务来对待。该挑战的主要目标是：(i) 促进野生数据在TTS和SASV中的应用，超越传统的清洁和受控数据集，考虑实际场景；以及(ii) 鼓励欺骗性生成(TTS)和欺骗性检测(SASV)社区之间的跨学科合作，从而促进更集成、更 robust 和更现实系统的开发。 

---
# A Workflow for Map Creation in Autonomous Vehicle Simulations 

**Title (ZH)**: 自主车辆仿真中的地图创建工作流 

**Authors**: Zubair Islam, Ahmaad Ansari, George Daoud, Mohamed El-Darieby  

**Link**: [PDF](https://arxiv.org/pdf/2508.16856)  

**Abstract**: The fast development of technology and artificial intelligence has significantly advanced Autonomous Vehicle (AV) research, emphasizing the need for extensive simulation testing. Accurate and adaptable maps are critical in AV development, serving as the foundation for localization, path planning, and scenario testing. However, creating simulation-ready maps is often difficult and resource-intensive, especially with simulators like CARLA (CAR Learning to Act). Many existing workflows require significant computational resources or rely on specific simulators, limiting flexibility for developers. This paper presents a custom workflow to streamline map creation for AV development, demonstrated through the generation of a 3D map of a parking lot at Ontario Tech University. Future work will focus on incorporating SLAM technologies, optimizing the workflow for broader simulator compatibility, and exploring more flexible handling of latitude and longitude values to enhance map generation accuracy. 

**Abstract (ZH)**: 快速发展的技术与人工智能显著推进了自动驾驶车辆（AV）研究，强调了需进行广泛模拟测试的必要性。精确且适应性强的地图对AV开发至关重要，是定位、路径规划和场景测试的基础。然而，创建可用于模拟的地图往往困难且资源密集，特别是在使用如CARLA等模拟器时。现有许多工作流程需要大量计算资源或依赖特定的模拟器，限制了开发者的灵活性。本文提出了一种定制工作流以简化AV开发中的地图创建过程，并通过在滑铁卢大学安大略分校生成一个停车场的3D地图予以展示。未来的工作将侧重于集成SLAM技术、优化工作流以提高对更广泛模拟器的兼容性、以及探索更灵活的纬度和经度值处理方法以提高地图生成精度。 

---
# DevLicOps: A Framework for Mitigating Licensing Risks in AI-Generated Code 

**Title (ZH)**: DevLicOps：一种用于缓解AI生成代码许可风险的框架 

**Authors**: Pratyush Nidhi Sharma, Lauren Wright, Anne Herfurth, Munsif Sokiyna, Pratyaksh Nidhi Sharma, Sethu Das, Mikko Siponen  

**Link**: [PDF](https://arxiv.org/pdf/2508.16853)  

**Abstract**: Generative AI coding assistants (ACAs) are widely adopted yet pose serious legal and compliance risks. ACAs can generate code governed by restrictive open-source licenses (e.g., GPL), potentially exposing companies to litigation or forced open-sourcing. Few developers are trained in these risks, and legal standards vary globally, especially with outsourcing. Our article introduces DevLicOps, a practical framework that helps IT leaders manage ACA-related licensing risks through governance, incident response, and informed tradeoffs. As ACA adoption grows and legal frameworks evolve, proactive license compliance is essential for responsible, risk-aware software development in the AI era. 

**Abstract (ZH)**: 生成式AI编码助手（ACAs）在广泛应用的同时也带来了严重的法律和合规风险。ACAs可以生成受限制的开源许可证代码（例如GPL），可能使公司面临诉讼或被迫开源。很少有开发者接受过这些风险的培训，而且在全球范围内，尤其是在外包情况下，法律标准差异很大。本文介绍了DevLicOps，这是一种实用框架，通过治理、事件响应和明智的选择帮助IT领导者管理ACA相关的许可风险。随着ACA的广泛应用和法律框架的不断演变，在AI时代进行负责任的风险意识软件开发需要积极的许可合规。 

---
# Gaussian Primitive Optimized Deformable Retinal Image Registration 

**Title (ZH)**: 高斯原始优化可变形视网膜图像配准 

**Authors**: Xin Tian, Jiazheng Wang, Yuxi Zhang, Xiang Chen, Renjiu Hu, Gaolei Li, Min Liu, Hang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16852)  

**Abstract**: Deformable retinal image registration is notoriously difficult due to large homogeneous regions and sparse but critical vascular features, which cause limited gradient signals in standard learning-based frameworks. In this paper, we introduce Gaussian Primitive Optimization (GPO), a novel iterative framework that performs structured message passing to overcome these challenges. After an initial coarse alignment, we extract keypoints at salient anatomical structures (e.g., major vessels) to serve as a minimal set of descriptor-based control nodes (DCN). Each node is modelled as a Gaussian primitive with trainable position, displacement, and radius, thus adapting its spatial influence to local deformation scales. A K-Nearest Neighbors (KNN) Gaussian interpolation then blends and propagates displacement signals from these information-rich nodes to construct a globally coherent displacement field; focusing interpolation on the top (K) neighbors reduces computational overhead while preserving local detail. By strategically anchoring nodes in high-gradient regions, GPO ensures robust gradient flow, mitigating vanishing gradient signal in textureless areas. The framework is optimized end-to-end via a multi-term loss that enforces both keypoint consistency and intensity alignment. Experiments on the FIRE dataset show that GPO reduces the target registration error from 6.2\,px to ~2.4\,px and increases the AUC at 25\,px from 0.770 to 0.938, substantially outperforming existing methods. The source code can be accessed via this https URL. 

**Abstract (ZH)**: 可变形视网膜图像配准由于存在大面积均匀区域和稀疏但关键的血管特征，导致在标准基于学习的框架中信号梯度有限，极具挑战性。本文引入了高斯原语优化（GPO），这是一种新颖的迭代框架，通过结构化的消息传递来克服这些挑战。在初步粗略对齐后，我们提取关键解剖结构（如主要血管）的特征点，作为基于描述子的控制节点（DCN）的最小节点集。每个节点被建模为一个可训练位置、位移和半径的高斯原语，从而使其空间影响适应局部变形尺度。通过K最近邻（KNN）高斯插值，这些信息丰富的节点的位移信号被融合和传播，构建全局一致的位移场；将插值集中在前（K）个邻居上，可以减少计算开销同时保留局部细节。通过在高梯度区域战略性地锚定节点，GPO确保了稳健的梯度流动，减轻了无纹理区域中的梯度消失信号。该框架通过多项损失函数进行端到端优化，以同时确保关键点一致性和强度对齐。在FIRE数据集上的实验表明，GPO将目标配准误差从6.2像素降低到约2.4像素，并且在25像素时的AUC从0.770提高到0.938，显著优于现有方法。源代码可通过以下链接访问：https://。 

---
# NinA: Normalizing Flows in Action. Training VLA Models with Normalizing Flows 

**Title (ZH)**: NinA: 正则化流动在行动. 使用正则化流动训练VLA模型 

**Authors**: Denis Tarasov, Alexander Nikulin, Ilya Zisman, Albina Klepach, Nikita Lyubaykin, Andrei Polubarov, Alexander Derevyagin, Vladislav Kurenkov  

**Link**: [PDF](https://arxiv.org/pdf/2508.16845)  

**Abstract**: Recent advances in Vision-Language-Action (VLA) models have established a two-component architecture, where a pre-trained Vision-Language Model (VLM) encodes visual observations and task descriptions, and an action decoder maps these representations to continuous actions. Diffusion models have been widely adopted as action decoders due to their ability to model complex, multimodal action distributions. However, they require multiple iterative denoising steps at inference time or downstream techniques to speed up sampling, limiting their practicality in real-world settings where high-frequency control is crucial. In this work, we present NinA (Normalizing Flows in Action), a fast and expressive alter- native to diffusion-based decoders for VLAs. NinA replaces the diffusion action decoder with a Normalizing Flow (NF) that enables one-shot sampling through an invertible transformation, significantly reducing inference time. We integrate NinA into the FLOWER VLA architecture and fine-tune on the LIBERO benchmark. Our experiments show that NinA matches the performance of its diffusion-based counterpart under the same training regime, while achieving substantially faster inference. These results suggest that NinA offers a promising path toward efficient, high-frequency VLA control without compromising performance. 

**Abstract (ZH)**: Recent Advances in Vision-Language-Action (VLA) Models: Introducing NinA (Normalizing Flows in Action) for Fast and Expressive Control 

---
# A Survey of Threats Against Voice Authentication and Anti-Spoofing Systems 

**Title (ZH)**: 语音认证和防欺骗系统面临的威胁综述 

**Authors**: Kamel Kamel, Keshav Sood, Hridoy Sankar Dutta, Sunil Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2508.16843)  

**Abstract**: Voice authentication has undergone significant changes from traditional systems that relied on handcrafted acoustic features to deep learning models that can extract robust speaker embeddings. This advancement has expanded its applications across finance, smart devices, law enforcement, and beyond. However, as adoption has grown, so have the threats. This survey presents a comprehensive review of the modern threat landscape targeting Voice Authentication Systems (VAS) and Anti-Spoofing Countermeasures (CMs), including data poisoning, adversarial, deepfake, and adversarial spoofing attacks. We chronologically trace the development of voice authentication and examine how vulnerabilities have evolved in tandem with technological advancements. For each category of attack, we summarize methodologies, highlight commonly used datasets, compare performance and limitations, and organize existing literature using widely accepted taxonomies. By highlighting emerging risks and open challenges, this survey aims to support the development of more secure and resilient voice authentication systems. 

**Abstract (ZH)**: 现代语音认证系统（VAS）及其抗欺骗措施（CMs）的威胁 landscape 及研究进展：包括数据中毒、对抗性攻击、深度合成和对抗性欺骗攻击的综合回顾 

---
# Physics-Inspired Spatial Temporal Graph Neural Networks for Predicting Industrial Chain Resilience 

**Title (ZH)**: 基于物理启发的空间时间图神经网络工业链韧性预测 

**Authors**: Bicheng Wang, Junping Wang, Yibo Xue  

**Link**: [PDF](https://arxiv.org/pdf/2508.16836)  

**Abstract**: Industrial chain plays an increasingly important role in the sustainable development of national economy. However, as a typical complex network, data-driven deep learning is still in its infancy in describing and analyzing the resilience of complex networks, and its core is the lack of a theoretical framework to describe the system dynamics. In this paper, we propose a physically informative neural symbolic approach to describe the evolutionary dynamics of complex networks for resilient prediction. The core idea is to learn the dynamics of the activity state of physical entities and integrate it into the multi-layer spatiotemporal co-evolution network, and use the physical information method to realize the joint learning of physical symbol dynamics and spatiotemporal co-evolution topology, so as to predict the industrial chain resilience. The experimental results show that the model can obtain better results and predict the elasticity of the industry chain more accurately and effectively, which has certain practical significance for the development of the industry. 

**Abstract (ZH)**: 工业链在国民经济可持续发展中发挥着越来越重要的作用。然而，作为典型的复杂网络，基于数据的深度学习在描述和分析复杂网络的韧性方面仍处于初级阶段，其核心问题是缺乏一个描述系统动力学的理论框架。在本文中，我们提出了一种物理信息神经符号方法来描述复杂网络的动力学演变以实现韧性预测。核心思想是学习物理实体活动状态的动力学并将其实现到多层时空共演化网络中，并通过物理信息方法实现物理符号动力学与时空共演化拓扑的联合学习，从而预测工业链韧性。实验结果表明，该模型可以取得更好的效果，并更准确有效地预测工业链的弹性，具有一定的实用意义。 

---
# Out of Distribution Detection for Efficient Continual Learning in Quality Prediction for Arc Welding 

**Title (ZH)**: 离分布检测在高效持续学习中的质量预测应用：电弧焊接 

**Authors**: Yannik Hahn, Jan Voets, Antonin Koenigsfeld, Hasan Tercan, Tobias Meisen  

**Link**: [PDF](https://arxiv.org/pdf/2508.16832)  

**Abstract**: Modern manufacturing relies heavily on fusion welding processes, including gas metal arc welding (GMAW). Despite significant advances in machine learning-based quality prediction, current models exhibit critical limitations when confronted with the inherent distribution shifts that occur in dynamic manufacturing environments. In this work, we extend the VQ-VAE Transformer architecture - previously demonstrating state-of-the-art performance in weld quality prediction - by leveraging its autoregressive loss as a reliable out-of-distribution (OOD) detection mechanism. Our approach exhibits superior performance compared to conventional reconstruction methods, embedding error-based techniques, and other established baselines. By integrating OOD detection with continual learning strategies, we optimize model adaptation, triggering updates only when necessary and thereby minimizing costly labeling requirements. We introduce a novel quantitative metric that simultaneously evaluates OOD detection capability while interpreting in-distribution performance. Experimental validation in real-world welding scenarios demonstrates that our framework effectively maintains robust quality prediction capabilities across significant distribution shifts, addressing critical challenges in dynamic manufacturing environments where process parameters frequently change. This research makes a substantial contribution to applied artificial intelligence by providing an explainable and at the same time adaptive solution for quality assurance in dynamic manufacturing processes - a crucial step towards robust, practical AI systems in the industrial environment. 

**Abstract (ZH)**: 现代制造业高度依赖于焊接工艺，包括气体金属弧焊（GMAW）。尽管基于机器学习的质量预测取得了显著进展，但当前模型在面对动态制造环境中固有的分布偏移时表现出关键的局限性。在本文中，我们通过利用VQ-VAE Transformer架构的自回归损失作为可靠的离群值检测（OOD）机制，扩展了该架构——先前在焊接质量预测任务中展示了最先进的性能。我们的方法在性能上优于传统的重构方法、基于误差的技术以及其他现有的基准。通过将离群值检测与连续学习策略相结合，我们优化了模型的适应性，仅在必要时触发更新，从而最大限度地减少了昂贵的标签要求。我们引入了一个新颖的定量指标，同时评估离群值检测能力和解释聚类内性能。在实际焊接场景下的实验验证表明，我们的框架有效地在显著的分布偏移下维持了稳健的质量预测能力，解决了动态制造环境中过程参数频繁变化的关键挑战。这项研究通过提供一个可解释且适应性强的解决方案，显著地促进了应用于动态制造过程的质量保证，这是工业环境中稳健、实用AI系统的一个关键步骤。 

---
# Understanding and Tackling Over-Dilution in Graph Neural Networks 

**Title (ZH)**: 理解并应对图神经网络中的过度稀释问题 

**Authors**: Junhyun Lee, Veronika Thost, Bumsoo Kim, Jaewoo Kang, Tengfei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.16829)  

**Abstract**: Message Passing Neural Networks (MPNNs) hold a key position in machine learning on graphs, but they struggle with unintended behaviors, such as over-smoothing and over-squashing, due to irregular data structures. The observation and formulation of these limitations have become foundational in constructing more informative graph representations. In this paper, we delve into the limitations of MPNNs, focusing on aspects that have previously been overlooked. Our observations reveal that even within a single layer, the information specific to an individual node can become significantly diluted. To delve into this phenomenon in depth, we present the concept of Over-dilution and formulate it with two dilution factors: intra-node dilution for attribute-level and inter-node dilution for node-level representations. We also introduce a transformer-based solution that alleviates over-dilution and complements existing node embedding methods like MPNNs. Our findings provide new insights and contribute to the development of informative representations. The implementation and supplementary materials are publicly available at this https URL. 

**Abstract (ZH)**: 消息传递神经网络（MPNNs）在图上的机器学习中占据关键位置，但由于不规则的数据结构，它们在不经意的行为，如过度平滑和过度挤压方面存在局限。这些局限性观察和建模已成为构建更具信息量的图表示的基础。在本文中，我们深入探讨了MPNNs的局限性，重点关注先前被忽略的方面。我们的观察表明，即使在单层中，单个节点特有的信息也可能显著稀释。为深入探讨这一现象，我们提出了过度稀释的概念，并通过两种稀释因子对其进行建模：节点内稀释用于属性级别和节点间稀释用于节点级表示。我们还介绍了一种基于transformer的解决方案，该解决方案缓解过度稀释并补充现有的节点嵌入方法，如MPNNs。我们的发现提供了新的见解，并促进了更具信息量的表示的发展。相关实现和补充材料可在以下网址获取：this https URL。 

---
# Exploring the Impact of Generative Artificial Intelligence on Software Development in the IT Sector: Preliminary Findings on Productivity, Efficiency and Job Security 

**Title (ZH)**: 探索生成式人工智能对信息技术sector软件开发的影响：关于生产力、效率和就业安全的初步发现 

**Authors**: Anton Ludwig Bonin, Pawel Robert Smolinski, Jacek Winiarski  

**Link**: [PDF](https://arxiv.org/pdf/2508.16811)  

**Abstract**: This study investigates the impact of Generative AI on software development within the IT sector through a mixed-method approach, utilizing a survey developed based on expert interviews. The preliminary results of an ongoing survey offer early insights into how Generative AI reshapes personal productivity, organizational efficiency, adoption, business strategy and job insecurity. The findings reveal that 97% of IT workers use Generative AI tools, mainly ChatGPT. Participants report significant personal productivity gain and perceive organizational efficiency improvements that correlate positively with Generative AI adoption by their organizations (r = .470, p < .05). However, increased organizational adoption of AI strongly correlates with heightened employee job security concerns (r = .549, p < .001). Key adoption challenges include inaccurate outputs (64.2%), regulatory compliance issues (58.2%) and ethical concerns (52.2%). This research offers early empirical insights into Generative AI's economic and organizational implications. 

**Abstract (ZH)**: 本研究通过混合方法探讨生成式AI在信息技术sector软件开发中的影响，基于专家访谈开发了一份调查问卷。正在进行的调查初步结果提供了生成式AI如何重塑个人生产力、组织效率、采纳、商业策略和工作安全感的早期见解。研究发现，97%的IT工作者使用生成式AI工具，主要为ChatGPT。参与者报告个人生产力显著提升，并感知到其组织中生成式AI采纳带来的组织效率改善（r = .470, p < .05）。然而，组织中AI采纳的增加与员工工作安全感的担忧显著正相关（r = .549, p < .001）。关键采纳挑战包括不准确的输出（64.2%）、合规问题（58.2%）和伦理问题（52.2%）。本研究提供了生成式AI在经济和组织方面的早期实证见解。 

---
# Autonomous UAV Flight Navigation in Confined Spaces: A Reinforcement Learning Approach 

**Title (ZH)**: 自主无人机在受限空间内的飞行导航：一种强化学习方法 

**Authors**: Marco S. Tayar, Lucas K. de Oliveira, Juliano D. Negri, Thiago H. Segreto, Ricardo V. Godoy, Marcelo Becker  

**Link**: [PDF](https://arxiv.org/pdf/2508.16807)  

**Abstract**: Inspecting confined industrial infrastructure, such as ventilation shafts, is a hazardous and inefficient task for humans. Unmanned Aerial Vehicles (UAVs) offer a promising alternative, but GPS-denied environments require robust control policies to prevent collisions. Deep Reinforcement Learning (DRL) has emerged as a powerful framework for developing such policies, and this paper provides a comparative study of two leading DRL algorithms for this task: the on-policy Proximal Policy Optimization (PPO) and the off-policy Soft Actor-Critic (SAC). The training was conducted with procedurally generated duct environments in Genesis simulation environment. A reward function was designed to guide a drone through a series of waypoints while applying a significant penalty for collisions. PPO learned a stable policy that completed all evaluation episodes without collision, producing smooth trajectories. By contrast, SAC consistently converged to a suboptimal behavior that traversed only the initial segments before failure. These results suggest that, in hazard-dense navigation, the training stability of on-policy methods can outweigh the nominal sample efficiency of off-policy algorithms. More broadly, the study provides evidence that procedurally generated, high-fidelity simulations are effective testbeds for developing and benchmarking robust navigation policies. 

**Abstract (ZH)**: 基于高性能模拟环境的深度强化学习在受限工业基础设施检测中的比较研究：以通风管道为例 

---
# Interpreting the Effects of Quantization on LLMs 

**Title (ZH)**: 量化对大语言模型影响的解读 

**Authors**: Manpreet Singh, Hassan Sajjad  

**Link**: [PDF](https://arxiv.org/pdf/2508.16785)  

**Abstract**: Quantization offers a practical solution to deploy LLMs in resource-constraint environments. However, its impact on internal representations remains understudied, raising questions about the reliability of quantized models. In this study, we employ a range of interpretability techniques to investigate how quantization affects model and neuron behavior. We analyze multiple LLMs under 4-bit and 8-bit quantization. Our findings reveal that the impact of quantization on model calibration is generally minor. Analysis of neuron activations indicates that the number of dead neurons, i.e., those with activation values close to 0 across the dataset, remains consistent regardless of quantization. In terms of neuron contribution to predictions, we observe that smaller full precision models exhibit fewer salient neurons, whereas larger models tend to have more, with the exception of Llama-2-7B. The effect of quantization on neuron redundancy varies across models. Overall, our findings suggest that effect of quantization may vary by model and tasks, however, we did not observe any drastic change which may discourage the use of quantization as a reliable model compression technique. 

**Abstract (ZH)**: 量化为在资源受限环境中部署大语言模型提供了实用解决方案，但其对内部表示的影响仍研究不足，引发了量化模型可靠性的疑问。在本研究中，我们采用多种可解释性技术探讨量化如何影响模型和神经元行为。我们分析了多个在4比特和8比特量化下的大语言模型。研究发现，量化对模型校准的影响通常较小。神经元激活分析表明，无论是否量化，近似为0的激活值的神经元数量保持一致。在预测中神经元的贡献方面，我们发现全精度较小的模型具有较少显著神经元，而较大的模型则具有更多，除了Llama-2-7B模型外。量化对神经元冗余的影响在不同模型中有所不同。总体而言，我们的研究结果显示量化效果可能因模型和任务而异，但未观察到任何极端变化，这表明量化仍可作为一种可靠的模型压缩技术。 

---
# Improving Performance, Robustness, and Fairness of Radiographic AI Models with Finely-Controllable Synthetic Data 

**Title (ZH)**: 基于细粒度可控的合成数据提高放射影像AI模型的性能、稳健性和公平性 

**Authors**: Stefania L. Moroianu, Christian Bluethgen, Pierre Chambon, Mehdi Cherti, Jean-Benoit Delbrouck, Magdalini Paschali, Brandon Price, Judy Gichoya, Jenia Jitsev, Curtis P. Langlotz, Akshay S. Chaudhari  

**Link**: [PDF](https://arxiv.org/pdf/2508.16783)  

**Abstract**: Achieving robust performance and fairness across diverse patient populations remains a challenge in developing clinically deployable deep learning models for diagnostic imaging. Synthetic data generation has emerged as a promising strategy to address limitations in dataset scale and diversity. We introduce RoentGen-v2, a text-to-image diffusion model for chest radiographs that enables fine-grained control over both radiographic findings and patient demographic attributes, including sex, age, and race/ethnicity. RoentGen-v2 is the first model to generate clinically plausible images with demographic conditioning, facilitating the creation of a large, demographically balanced synthetic dataset comprising over 565,000 images. We use this large synthetic dataset to evaluate optimal training pipelines for downstream disease classification models. In contrast to prior work that combines real and synthetic data naively, we propose an improved training strategy that leverages synthetic data for supervised pretraining, followed by fine-tuning on real data. Through extensive evaluation on over 137,000 chest radiographs from five institutions, we demonstrate that synthetic pretraining consistently improves model performance, generalization to out-of-distribution settings, and fairness across demographic subgroups. Across datasets, synthetic pretraining led to a 6.5% accuracy increase in the performance of downstream classification models, compared to a modest 2.7% increase when naively combining real and synthetic data. We observe this performance improvement simultaneously with the reduction of the underdiagnosis fairness gap by 19.3%. These results highlight the potential of synthetic imaging to advance equitable and generalizable medical deep learning under real-world data constraints. We open source our code, trained models, and synthetic dataset at this https URL . 

**Abstract (ZH)**: 在开发适用于临床诊断成像的深度学习模型时，实现跨多样化患者群体的稳健性能和公平性仍具挑战性。合成数据生成已成为解决数据集规模和多样性限制的一种有前景的策略。我们介绍了RoentGen-v2，这是一种用于胸部X光片的文本到图像扩散模型，该模型可对影像发现和患者人口统计学属性（包括性别、年龄和种族/族裔）进行精细控制。RoentGen-v2 是首款具备人口统计学调整能力以生成临床合理图像的模型，促进了包含超过565,000张图像的大规模、人口统计学平衡的合成数据集的创建。我们使用此大规模合成数据集来评估下游疾病分类模型的最佳训练管线。与直接将真实和合成数据组合的做法不同，我们提出了一种改进的训练策略，该策略利用合成数据进行监督预训练，然后在真实数据上进行微调。通过在五个机构的超过137,000张胸部X光片上进行广泛评估，我们证明了合成数据预训练可以一致地提高模型性能、泛化至分布外环境的能力以及不同人口子组的公平性。与直接结合真实和合成数据相比，合成数据预训练在各个数据集中导致下游分类模型性能提高6.5%，而后者仅提高了2.7%。我们同时观察到，在预训练过程中公平性差距（特别是在诊断不足方面）减少了19.3%。这些结果强调了在实际数据限制条件下，合成影像如何促进公平和泛化医疗深度学习的发展潜力。我们在此处提供我们的代码、训练模型和合成数据集。 

---
# EyeMulator: Improving Code Language Models by Mimicking Human Visual Attention 

**Title (ZH)**: EyeMulator: 通过模仿人类视觉注意力改善代码语言模型 

**Authors**: Yifan Zhang, Chen Huang, Yueke Zhang, Jiahao Zhang, Toby Jia-Jun Li, Collin McMillan, Kevin Leach, Yu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16771)  

**Abstract**: Code language models (so-called CodeLLMs) are now commonplace in software development. As a general rule, CodeLLMs are trained by dividing training examples into input tokens and then learn importance of those tokens in a process called machine attention. Machine attention is based solely on input token salience to output token examples during training. Human software developers are different, as humans intuitively know that some tokens are more salient than others. While intuition itself is ineffable and a subject of philosophy, clues about salience are present in human visual attention, since people tend to look at more salient words more often. In this paper, we present EyeMulator, a technique for training CodeLLMs to mimic human visual attention while training for various software development tasks. We add special weights for each token in each input example to the loss function used during LLM fine-tuning. We draw these weights from observations of human visual attention derived from a previously-collected publicly-available dataset of eye-tracking experiments in software engineering tasks. These new weights ultimately induce changes in the attention of the subject LLM during training, resulting in a model that does not need eye-tracking data during inference. Our evaluation shows that EyeMulator outperforms strong LLM baselines on several tasks such as code translation, completion and summarization. We further show an ablation study that demonstrates the improvement is due to subject models learning to mimic human attention. 

**Abstract (ZH)**: EyeMulator：训练代码语言模型模仿人类视觉注意力的方法 

---
# Guarding Your Conversations: Privacy Gatekeepers for Secure Interactions with Cloud-Based AI Models 

**Title (ZH)**: 守护您的对话：面向基于云的AI模型的安全交互隐私门卫 

**Authors**: GodsGift Uzor, Hasan Al-Qudah, Ynes Ineza, Abdul Serwadda  

**Link**: [PDF](https://arxiv.org/pdf/2508.16765)  

**Abstract**: The interactive nature of Large Language Models (LLMs), which closely track user data and context, has prompted users to share personal and private information in unprecedented ways. Even when users opt out of allowing their data to be used for training, these privacy settings offer limited protection when LLM providers operate in jurisdictions with weak privacy laws, invasive government surveillance, or poor data security practices. In such cases, the risk of sensitive information, including Personally Identifiable Information (PII), being mishandled or exposed remains high. To address this, we propose the concept of an "LLM gatekeeper", a lightweight, locally run model that filters out sensitive information from user queries before they are sent to the potentially untrustworthy, though highly capable, cloud-based LLM. Through experiments with human subjects, we demonstrate that this dual-model approach introduces minimal overhead while significantly enhancing user privacy, without compromising the quality of LLM responses. 

**Abstract (ZH)**: 大型语言模型（LLMs）的交互性质，使得它们紧密跟踪用户数据和上下文，促使用户以前所未有的方式分享个人和私人信息。即使用户选择不允许其数据用于训练，当LLM提供商所在司法管辖区的隐私法律薄弱、存在侵入性政府监控或数据安全实践不当时，这些隐私设置提供的保护也有限。在这种情况下，敏感信息，包括个人可识别信息（PII），仍有可能被不当处理或泄露。为解决这一问题，我们提出了一种“LLM守门人”的概念，即一个轻量级的本地运行模型，在将用户查询发送到可能不可信但功能强大的云基LLM之前，过滤掉敏感信息。通过使用人类受试者的实验，我们证明这种双模型方法基本不增加负担，同时显著增强用户隐私，而不牺牲LLM响应的质量。 

---
# FAIRWELL: Fair Multimodal Self-Supervised Learning for Wellbeing Prediction 

**Title (ZH)**: FAIRWELL: 公平的多模态自监督学习以预测幸福感 

**Authors**: Jiaee Cheong, Abtin Mogharabin, Paul Liang, Hatice Gunes, Sinan Kalkan  

**Link**: [PDF](https://arxiv.org/pdf/2508.16748)  

**Abstract**: Early efforts on leveraging self-supervised learning (SSL) to improve machine learning (ML) fairness has proven promising. However, such an approach has yet to be explored within a multimodal context. Prior work has shown that, within a multimodal setting, different modalities contain modality-unique information that can complement information of other modalities. Leveraging on this, we propose a novel subject-level loss function to learn fairer representations via the following three mechanisms, adapting the variance-invariance-covariance regularization (VICReg) method: (i) the variance term, which reduces reliance on the protected attribute as a trivial solution; (ii) the invariance term, which ensures consistent predictions for similar individuals; and (iii) the covariance term, which minimizes correlational dependence on the protected attribute. Consequently, our loss function, coined as FAIRWELL, aims to obtain subject-independent representations, enforcing fairness in multimodal prediction tasks. We evaluate our method on three challenging real-world heterogeneous healthcare datasets (i.e. D-Vlog, MIMIC and MODMA) which contain different modalities of varying length and different prediction tasks. Our findings indicate that our framework improves overall fairness performance with minimal reduction in classification performance and significantly improves on the performance-fairness Pareto frontier. 

**Abstract (ZH)**: 利用自监督学习提高多模态机器学习公平性的新进展：FAIRWELL方法 

---
# Beyond Memorization: Extending Reasoning Depth with Recurrence, Memory and Test-Time Compute Scaling 

**Title (ZH)**: 超越记忆：通过循环、记忆扩展推理深度及测试时计算量扩展 

**Authors**: Ivan Rodkin, Daniil Orel, Konstantin Smirnov, Arman Bolatov, Bilal Elbouardi, Besher Hassan, Yuri Kuratov, Aydar Bulatov, Preslav Nakov, Timothy Baldwin, Artem Shelmanov, Mikhail Burtsev  

**Link**: [PDF](https://arxiv.org/pdf/2508.16745)  

**Abstract**: Reasoning is a core capability of large language models, yet understanding how they learn and perform multi-step reasoning remains an open problem. In this study, we explore how different architectures and training methods affect model multi-step reasoning capabilities within a cellular automata framework. By training on state sequences generated with random Boolean functions for random initial conditions to exclude memorization, we demonstrate that most neural architectures learn to abstract the underlying rules. While models achieve high accuracy in next-state prediction, their performance declines sharply if multi-step reasoning is required. We confirm that increasing model depth plays a crucial role for sequential computations. We demonstrate that an extension of the effective model depth with recurrence, memory, and test-time compute scaling substantially enhances reasoning capabilities. 

**Abstract (ZH)**: 大型语言模型中的推理是一种核心能力，但理解它们如何学习和执行多步推理仍是一个开放问题。在本研究中，我们探究了不同架构和训练方法如何影响模型在细胞自动机框架内的多步推理能力。通过使用随机布尔函数生成随机初始条件下的状态序列进行训练以排除记忆效应，我们证明大多数神经架构学会抽象底层规则。虽然模型在下一步预测中实现了高准确性，但在需要多步推理时其性能会急剧下降。我们确认增加模型深度在序贯计算中起着关键作用。我们证明了通过递归、记忆以及测试时计算量扩展来扩展有效模型深度显著增强了推理能力。 

---
# CellEcoNet: Decoding the Cellular Language of Pathology with Deep Learning for Invasive Lung Adenocarcinoma Recurrence Prediction 

**Title (ZH)**: CellEcoNet: 使用深度学习解码病理学的细胞语言以预测侵袭性肺腺癌的复发 

**Authors**: Abdul Rehman Akbar, Usama Sajjad, Ziyu Su, Wencheng Li, Fei Xing, Jimmy Ruiz, Wei Chen, Muhammad Khalid Khan Niazi  

**Link**: [PDF](https://arxiv.org/pdf/2508.16742)  

**Abstract**: Despite surgical resection, ~70% of invasive lung adenocarcinoma (ILA) patients recur within five years, and current tools fail to identify those needing adjuvant therapy. To address this unmet clinical need, we introduce CellEcoNet, a novel spatially aware deep learning framework that models whole slide images (WSIs) through natural language analogy, defining a "language of pathology," where cells act as words, cellular neighborhoods become phrases, and tissue architecture forms sentences. CellEcoNet learns these context-dependent meanings automatically, capturing how subtle variations and spatial interactions derive recurrence risk. On a dataset of 456 H&E-stained WSIs, CellEcoNet achieved superior predictive performance (AUC:77.8% HR:9.54), outperforming IASLC grading system (AUC:71.4% HR:2.36), AJCC Stage (AUC:64.0% HR:1.17) and state-of-the-art computational methods (AUCs:62.2-67.4%). CellEcoNet demonstrated fairness and consistent performance across diverse demographic and clinical subgroups. Beyond prognosis, CellEcoNet marks a paradigm shift by decoding the tumor microenvironment's cellular "language" to reveal how subtle cell variations encode recurrence risk. 

**Abstract (ZH)**: 基于空间感知的深度学习框架CellEcoNet在肺癌腺癌复发风险预测中的应用：超越当前工具实现精准亚群管理 

---
# WST: Weak-to-Strong Knowledge Transfer via Reinforcement Learning 

**Title (ZH)**: WST: 通过强化学习实现从弱到强的知识迁移 

**Authors**: Haosen Ge, Shuo Li, Lianghuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16741)  

**Abstract**: Effective prompt engineering remains a challenging task for many applications. We introduce Weak-to-Strong Transfer (WST), an automatic prompt engineering framework where a small "Teacher" model generates instructions that enhance the performance of a much larger "Student" model. Unlike prior work, WST requires only a weak teacher, making it efficient and broadly applicable in settings where large models are closed-source or difficult to fine-tune. Using reinforcement learning, the Teacher Model's instructions are iteratively improved based on the Student Model's outcomes, yielding substantial gains across reasoning (MATH-500, GSM8K) and alignment (HH-RLHF) benchmarks - 98% on MATH-500 and 134% on HH-RLHF - and surpassing baselines such as GPT-4o-mini and Llama-70B. These results demonstrate that small models can reliably scaffold larger ones, unlocking latent capabilities while avoiding misleading prompts that stronger teachers may introduce, establishing WST as a scalable solution for efficient and safe LLM prompt refinement. 

**Abstract (ZH)**: 弱到强的迁移（WST）：一种自动提示工程框架 

---
# AI Product Value Assessment Model: An Interdisciplinary Integration Based on Information Theory, Economics, and Psychology 

**Title (ZH)**: 基于信息理论、经济学和心理学的AI产品价值评估模型 

**Authors**: Yu yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16714)  

**Abstract**: In recent years, breakthroughs in artificial intelligence (AI) technology have triggered global industrial transformations, with applications permeating various fields such as finance, healthcare, education, and manufacturing. However, this rapid iteration is accompanied by irrational development, where enterprises blindly invest due to technology hype, often overlooking systematic value assessments. This paper develops a multi-dimensional evaluation model that integrates information theory's entropy reduction principle, economics' bounded rationality framework, and psychology's irrational decision theories to quantify AI product value. Key factors include positive dimensions (e.g., uncertainty elimination, efficiency gains, cost savings, decision quality improvement) and negative risks (e.g., error probability, impact, and correction costs). A non-linear formula captures factor couplings, and validation through 10 commercial cases demonstrates the model's effectiveness in distinguishing successful and failed products, supporting hypotheses on synergistic positive effects, non-linear negative impacts, and interactive regulations. Results reveal value generation logic, offering enterprises tools to avoid blind investments and promote rational AI industry development. Future directions include adaptive weights, dynamic mechanisms, and extensions to emerging AI technologies like generative models. 

**Abstract (ZH)**: 近年来，人工智能（AI）技术突破引发了全球产业变革，其应用遍及金融、医疗、教育和制造等多个领域。然而，这一快速迭代过程中伴随着盲目发展，企业因技术 hype 盲目投资，往往忽视了系统的价值评估。本文开发了一种多维评估模型，将信息论的熵减原则、经济学的有限理性框架和心理学的非理性决策理论相结合，以量化AI产品的价值。关键因素包括正面维度（如不确定性消除、效率提升、成本节省、决策质量改进）和负面风险（如错误概率、影响程度和修正成本）。非线性公式捕捉了因素间的耦合，通过10个商用案例验证，该模型在区分成功与失败产品方面显示出有效性，并支持协同正面效应、非线性负面影响和互动规制的假说。结果揭示了价值创造逻辑，为企业提供了避免盲目投资、促进理性AI产业发展工具。未来方向包括自适应权重、动态机制及其向生成模型等新兴AI技术的拓展。 

---
# CelloAI: Leveraging Large Language Models for HPC Software Development in High Energy Physics 

**Title (ZH)**: CelloAI：利用大规模语言模型促进高能物理领域的HPC软件开发 

**Authors**: Mohammad Atif, Kriti Chopra, Ozgur Kilic, Tianle Wang, Zhihua Dong, Charles Leggett, Meifeng Lin, Paolo Calafiura, Salman Habib  

**Link**: [PDF](https://arxiv.org/pdf/2508.16713)  

**Abstract**: Next-generation High Energy Physics (HEP) experiments will generate unprecedented data volumes, necessitating High Performance Computing (HPC) integration alongside traditional high-throughput computing. However, HPC adoption in HEP is hindered by the challenge of porting legacy software to heterogeneous architectures and the sparse documentation of these complex scientific codebases. We present CelloAI, a locally hosted coding assistant that leverages Large Language Models (LLMs) with retrieval-augmented generation (RAG) to support HEP code documentation and generation. This local deployment ensures data privacy, eliminates recurring costs and provides access to large context windows without external dependencies. CelloAI addresses two primary use cases, code documentation and code generation, through specialized components. For code documentation, the assistant provides: (a) Doxygen style comment generation for all functions and classes by retrieving relevant information from RAG sources (papers, posters, presentations), (b) file-level summary generation, and (c) an interactive chatbot for code comprehension queries. For code generation, CelloAI employs syntax-aware chunking strategies that preserve syntactic boundaries during embedding, improving retrieval accuracy in large codebases. The system integrates callgraph knowledge to maintain dependency awareness during code modifications and provides AI-generated suggestions for performance optimization and accurate refactoring. We evaluate CelloAI using real-world HEP applications from ATLAS, CMS, and DUNE experiments, comparing different embedding models for code retrieval effectiveness. Our results demonstrate the AI assistant's capability to enhance code understanding and support reliable code generation while maintaining the transparency and safety requirements essential for scientific computing environments. 

**Abstract (ZH)**: 下一代高能物理（HEP）实验将生成前所未有的数据量，需要将高性能计算（HPC）与传统的高吞吐量计算相结合。然而，HPC在HEP中的采用受到将遗留软件移植到异构架构以及这些复杂科学代码库文档稀疏不足的挑战。我们提出了CelloAI，这是一种本地托管的编程助手，利用大型语言模型（LLMs）和检索增强生成（RAG）技术支持HEP代码文档和生成。这种本地部署确保了数据隐私，消除了重复成本，并在无需外部依赖的情况下提供了大上下文窗口。CelloAI通过专门组件处理两种主要用例——代码文档和代码生成。对于代码文档，助手提供：（a）从RAG来源（论文、海报、演讲）检索相关信息生成函数和类的所有函数样式注释，（b）文件级摘要生成，以及（c）用于代码理解查询的交互式聊天机器人。对于代码生成，CelloAI使用语法感知的分块策略，在嵌入期间保持语法边界，提高大型代码库的检索准确性。该系统结合调用图知识，在代码修改过程中保持依赖性意识，并提供AI生成的性能优化和准确重构建议。我们使用来自ATLAS、CMS和DUNE实验的真实世界HEP应用评估了CelloAI，比较了不同的嵌入模型以评估代码检索效果。我们的结果显示，AI助手能够增强代码理解并支持可靠的代码生成，同时满足科学计算环境所需的透明度和安全性要求。 

---
# Systematic Characterization of LLM Quantization: A Performance, Energy, and Quality Perspective 

**Title (ZH)**: LLM量化系统的系统化表征：从性能、能耗和质量视角]+$ 

**Authors**: Tianyao Shi, Yi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2508.16712)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across diverse domains, but their heavy resource demands make quantization-reducing precision to lower-bit formats-critical for efficient serving. While many quantization methods exist, a systematic understanding of their performance, energy, and quality tradeoffs in realistic serving conditions remains a gap. In this work, we first develop a fully automated online characterization framework qMeter, and then conduct an in-depth characterization of 11 post-training LLM quantization methods across 4 model sizes (7B-70B) and two GPU architectures (A100, H100). We evaluate quantization at the application, workload, parallelism, and hardware levels under online serving conditions. Our study reveals highly task- and method-dependent tradeoffs, strong sensitivity to workload characteristics, and complex interactions with parallelism and GPU architecture. We further present three optimization case studies illustrating deployment challenges in capacity planning, energy-efficient scheduling, and multi-objective tuning. To the best of our knowledge, this is one of the first comprehensive application-, system-, and hardware-level characterization of LLM quantization from a joint performance, energy, and quality perspective. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多个领域展现了卓越的能力，但其对资源的高需求使得量化（将精度降低到较低位宽格式）成为高效服务的关键。虽然存在多种量化方法，但在现实服务条件下对其性能、能耗和质量tradeoffs的理解仍存在差距。在本文中，我们首先开发了一个完全自动化的在线表征框架qMeter，然后在4种模型规模（7B-70B）和两种GPU架构（A100, H100）下对11种后训练LLM量化方法进行了深入的表征。我们在在线服务条件下从应用、工作负载、并行性和硬件层面评估了量化。我们的研究揭示了高度任务依赖和方法依赖的tradeoffs、对工作负载特性强烈的敏感性以及与并行性和GPU架构的复杂交互。我们进一步展示了三个优化案例研究，说明了容量规划、能效调度和多目标调优方面的部署挑战。据我们所知，这是首次从性能、能耗和质量的联合视角对LLM量化进行全面的应用、系统和硬件层面表征的研究。 

---
# RoboBuddy in the Classroom: Exploring LLM-Powered Social Robots for Storytelling in Learning and Integration Activities 

**Title (ZH)**: RoboBuddy 在课堂中的应用：探索以大语言模型为动力的社会机器人在学习和整合活动中的故事讲述 

**Authors**: Daniel Tozadore, Nur Ertug, Yasmine Chaker, Mortadha Abderrahim  

**Link**: [PDF](https://arxiv.org/pdf/2508.16706)  

**Abstract**: Creating and improvising scenarios for content approaching is an enriching technique in education. However, it comes with a significant increase in the time spent on its planning, which intensifies when using complex technologies, such as social robots. Furthermore, addressing multicultural integration is commonly embedded in regular activities due to the already tight curriculum. Addressing these issues with a single solution, we implemented an intuitive interface that allows teachers to create scenario-based activities from their regular curriculum using LLMs and social robots. We co-designed different frameworks of activities with 4 teachers and deployed it in a study with 27 students for 1 week. Beyond validating the system's efficacy, our findings highlight the positive impact of integration policies perceived by the children and demonstrate the importance of scenario-based activities in students' enjoyment, observed to be significantly higher when applying storytelling. Additionally, several implications of using LLMs and social robots in long-term classroom activities are discussed. 

**Abstract (ZH)**: 通过创建和即兴编排情境来丰富内容教学是一种富有成效的技术，但在规划时会消耗大量时间，尤其是在使用复杂技术如社会机器人时。此外，由于课程已经非常紧凑，跨文化融合的处理通常嵌入到常规活动中。为解决这些问题，我们实现了一个直观的界面，允许教师利用大型语言模型（LLM）和社会机器人从常规课程中创建基于情境的活动。与4位教师共同设计了不同的活动框架，并在27名学生中进行了为期一周的研究。除了验证系统的有效性，我们的发现还强调了孩子们感受到的融合政策的正面影响，并展示了基于情境活动在学生中的满意度显著提高，尤其是在讲故事的情况下。此外，还讨论了在长期课堂教学活动中使用LLM和社会机器人的若干启示。 

---
# Assessing Consciousness-Related Behaviors in Large Language Models Using the Maze Test 

**Title (ZH)**: 使用迷宫测试评估大型语言模型的相关意识行为 

**Authors**: Rui A. Pimenta, Tim Schlippe, Kristina Schaaff  

**Link**: [PDF](https://arxiv.org/pdf/2508.16705)  

**Abstract**: We investigate consciousness-like behaviors in Large Language Models (LLMs) using the Maze Test, challenging models to navigate mazes from a first-person perspective. This test simultaneously probes spatial awareness, perspective-taking, goal-directed behavior, and temporal sequencing-key consciousness-associated characteristics. After synthesizing consciousness theories into 13 essential characteristics, we evaluated 12 leading LLMs across zero-shot, one-shot, and few-shot learning scenarios. Results showed reasoning-capable LLMs consistently outperforming standard versions, with Gemini 2.0 Pro achieving 52.9% Complete Path Accuracy and DeepSeek-R1 reaching 80.5% Partial Path Accuracy. The gap between these metrics indicates LLMs struggle to maintain coherent self-models throughout solutions -- a fundamental consciousness aspect. While LLMs show progress in consciousness-related behaviors through reasoning mechanisms, they lack the integrated, persistent self-awareness characteristic of consciousness. 

**Abstract (ZH)**: 我们使用迷宫测试研究大型语言模型（LLMs）的意识-like 行为，要求模型从第一人称视角导航迷宫。该测试同时探究空间意识、换位思考、目标导向行为和时间序列排列——这些是与意识相关的关键特征。在将意识理论综合为13个基本特征后，我们评估了12个领先的LLM在零样本、单样本和少样本学习场景中的表现。结果显示，具备推理能力的LLM一贯优于标准版本，Gemini 2.0 Pro 的完整路径准确率为52.9%，DeepSeek-R1 的部分路径准确率为80.5%。这些指标之间的差距表明LLM在解决方案过程中难以维持连贯的自我模型——这是意识的一个基本方面。尽管LLM通过推理机制在意识相关行为方面取得进展，但仍缺乏意识的整合和持续的自我意识特征。 

---
# Dynamic Sparse Attention on Mobile SoCs 

**Title (ZH)**: 移动SoC上的动态稀疏注意力机制 

**Authors**: Wangsong Yin, Daliang Xu, Mengwei Xu, Gang Huang, Xuanzhe Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.16703)  

**Abstract**: On-device running Large Language Models (LLMs) is nowadays a critical enabler towards preserving user privacy. We observe that the attention operator falls back from the special-purpose NPU to the general-purpose CPU/GPU because of quantization sensitivity in state-of-the-art frameworks. This fallback results in a degraded user experience and increased complexity in system scheduling. To this end, this paper presents shadowAttn, a system-algorithm codesigned sparse attention module with minimal reliance on CPU/GPU by only sparsely calculating the attention on a tiny portion of tokens. The key idea is to hide the overhead of estimating the important tokens with a NPU-based pilot compute. Further, shadowAttn proposes insightful techniques such as NPU compute graph bucketing, head-wise NPU-CPU/GPU pipeline and per-head fine-grained sparsity ratio to achieve high accuracy and efficiency. shadowAttn delivers the best performance with highly limited CPU/GPU resource; it requires much less CPU/GPU resource to deliver on-par performance of SoTA frameworks. 

**Abstract (ZH)**: On-device运行大规模语言模型（LLMs）现如今是保持用户隐私的关键使能器。我们观察到，在先进框架中由于量化敏感性，注意力操作从专用的NPU fallback到了通用的CPU/GPU上。这一fallback导致了用户体验下降和系统调度复杂性的增加。为此，本文提出了shadowAttn，这是一种与系统和算法协同设计的稀疏注意力模块，通过仅对少量 tokens 稀疏计算注意力来尽量减少对CPU/GPU的依赖。核心思想是使用NPU为基础的试点计算来隐藏估计重要 tokens 的开销。此外，shadowAttn 提出了如基于NPU的计算图桶化、按头NPU-CPU/GPU流水线和按头细粒度稀疏比率等见解性技术，以实现高准确性和效率。shadowAttn 在极其有限的CPU/GPU资源下提供了最佳性能；它需要较少的CPU/GPU资源就能达到先进框架相当的性能。 

---
# Generative Artificial Intelligence and Agents in Research and Teaching 

**Title (ZH)**: 生成式人工智能与智能代理在研究和教学中的应用 

**Authors**: Jussi S. Jauhiainen, Aurora Toppari  

**Link**: [PDF](https://arxiv.org/pdf/2508.16701)  

**Abstract**: This study provides a comprehensive analysis of the development, functioning, and application of generative artificial intelligence (GenAI) and large language models (LLMs), with an emphasis on their implications for research and education. It traces the conceptual evolution from artificial intelligence (AI) through machine learning (ML) and deep learning (DL) to transformer architectures, which constitute the foundation of contemporary generative systems. Technical aspects, including prompting strategies, word embeddings, and probabilistic sampling methods (temperature, top-k, and top-p), are examined alongside the emergence of autonomous agents. These elements are considered in relation to both the opportunities they create and the limitations and risks they entail.
The work critically evaluates the integration of GenAI across the research process, from ideation and literature review to research design, data collection, analysis, interpretation, and dissemination. While particular attention is given to geographical research, the discussion extends to wider academic contexts. A parallel strand addresses the pedagogical applications of GenAI, encompassing course and lesson design, teaching delivery, assessment, and feedback, with geography education serving as a case example.
Central to the analysis are the ethical, social, and environmental challenges posed by GenAI. Issues of bias, intellectual property, governance, and accountability are assessed, alongside the ecological footprint of LLMs and emerging technological strategies for mitigation. The concluding section considers near- and long-term futures of GenAI, including scenarios of sustained adoption, regulation, and potential decline. By situating GenAI within both scholarly practice and educational contexts, the study contributes to critical debates on its transformative potential and societal responsibilities. 

**Abstract (ZH)**: 本研究对生成人工智能（GenAI）和大型语言模型（LLM）的发展、运作和应用进行了全面分析，并强调了它们对研究和教育的影响。该研究从人工智能（AI）到机器学习（ML）、深度学习（DL）再到变换器架构的逐步演进进行了追溯，后者构成了当代生成系统的基石。研究不仅探讨了这些技术层面的内容，包括提示策略、词嵌入和概率采样方法（温度、top-k和top-p），而且还分析了自主智能体的出现。这些元素与其带来的机会以及涉及的局限性和风险共同被纳入考量。

该研究批判性地评估了GenAI在研究过程中的整合，从构想和文献综述到研究设计、数据收集、分析、解释和传播等各个环节。特别关注了地理研究领域，同时也延伸到更广泛的学术背景下。另一条线索则探讨了GenAI的教学应用，包括课程和教学设计、教学实施、评估和反馈，地理教育被用作案例研究。

分析的核心在于GenAI提出的伦理、社会和环境挑战。评估了偏见、知识产权、治理和问责制等问题，同时关注了LLM的生态足迹和新兴的技术缓解策略。研究的最后一部分考虑了GenAI的近期和远期未来，包括持续采用、监管和潜在下降的场景。通过对GenAI既置于学术实践又置于教育背景中的探讨，该研究为有关其变革潜力及其社会责任的批判性辩论做出了贡献。 

---
# GPT-OSS-20B: A Comprehensive Deployment-Centric Analysis of OpenAI's Open-Weight Mixture of Experts Model 

**Title (ZH)**: GPT-OSS-20B：OpenAI的开放权重专家混合模型的全面部署中心分析 

**Authors**: Deepak Kumar, Divakar Yadav, Yash Patel  

**Link**: [PDF](https://arxiv.org/pdf/2508.16700)  

**Abstract**: We present a single-GPU (H100, bf16) evaluation of GPT-OSS-20B (Mixture-of-Experts; 20.9B total, approx. 3.61B active) against dense baselines Qwen3-32B and Yi-34B across multiple dimensions. We measure true time-to-first-token (TTFT), full-decode throughput (TPOT), end-to-end latency percentiles, peak VRAM with past key values (PKV) held, and energy via a consistent nvidia-smi-based sampler. At a 2048-token context with 64-token decode, GPT-OSS-20B delivers higher decode throughput and tokens per Joule than dense baselines Qwen3-32B and Yi-34B, while substantially reducing peak VRAM and energy per 1000 generated tokens; its TTFT is higher due to MoE routing overhead. With only 17.3% of parameters active (3.61B of 20.9B), GPT-OSS-20B provides about 31.8% higher decode throughput and 25.8% lower energy per 1000 generated tokens than Qwen3-32B at 2048/64, while using 31.7% less peak VRAM. Normalized by active parameters, GPT-OSS-20B shows markedly stronger per-active-parameter efficiency (APE), underscoring MoE's deployment advantages. We do not evaluate accuracy; this is a deployment-focused study. We release code and consolidated results to enable replication and extension. 

**Abstract (ZH)**: 我们使用单个GPU（H100，bf16）对GPT-OSS-20B（专家混合模型；总计20.9B，约3.61B活跃参数）与稠密基准Qwen3-32B和Yi-34B在多个维度上进行评估。我们测量了首个令牌时间（TTFT）、完整解码吞吐量（TPOT）、端到端延迟百分位数、持有过去键值的显存峰值以及能量消耗（通过一致的nvidia-smi采样器）。在2048-token上下文和64-token解码的情况下，GPT-OSS-20B在解码吞吐量和每焦耳生成的令牌数上高于稠密基准Qwen3-32B和Yi-34B，同时显存峰值和每1000生成令牌的能量消耗显著降低；其TTFT较高，这是由于专家混合模型的路由开销。尽管只有17.3%的参数活跃（3.61B中的20.9B），GPT-OSS-20B在2048/64场景下相比Qwen3-32B提供约31.8%更高的解码吞吐量和25.8%更低的每1000生成令牌的能量消耗，同时使用31.7%更少的显存峰值。基于活跃参数进行标准化后，GPT-OSS-20B显示出了明显的每活跃参数效率（APE）优势，突出了专家混合模型的部署优势。我们没有评估准确率；这是一项侧重部署的研究。我们提供了代码和综合结果以供复制和扩展。 

---
# QueryBandits for Hallucination Mitigation: Exploiting Semantic Features for No-Regret Rewriting 

**Title (ZH)**: QueryBandits for Hallucination Mitigation: Exploiting Semantic Features for No-Regret Rewriting 

**Authors**: Nicole Cho, William Watson, Alec Koppel, Sumitra Ganesh, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2508.16697)  

**Abstract**: Advanced reasoning capabilities in Large Language Models (LLMs) have caused higher hallucination prevalence; yet most mitigation work focuses on after-the-fact filtering rather than shaping the queries that trigger them. We introduce QueryBandits, a bandit framework that designs rewrite strategies to maximize a reward model, that encapsulates hallucination propensity based upon the sensitivities of 17 linguistic features of the input query-and therefore, proactively steer LLMs away from generating hallucinations. Across 13 diverse QA benchmarks and 1,050 lexically perturbed queries per dataset, our top contextual QueryBandit (Thompson Sampling) achieves an 87.5% win rate over a no-rewrite baseline and also outperforms zero-shot static prompting ("paraphrase" or "expand") by 42.6% and 60.3% respectively. Therefore, we empirically substantiate the effectiveness of QueryBandits in mitigating hallucination via the intervention that takes the form of a query rewrite. Interestingly, certain static prompting strategies, which constitute a considerable number of current query rewriting literature, have a higher cumulative regret than the no-rewrite baseline, signifying that static rewrites can worsen hallucination. Moreover, we discover that the converged per-arm regression feature weight vectors substantiate that there is no single rewrite strategy optimal for all queries. In this context, guided rewriting via exploiting semantic features with QueryBandits can induce significant shifts in output behavior through forward-pass mechanisms, bypassing the need for retraining or gradient-based adaptation. 

**Abstract (ZH)**: 大型语言模型（LLMs）中的高级推理能力导致幻觉现象更加普遍；然而，大多数缓解工作集中于事后过滤，而不是塑造触发幻觉的查询。我们引入QueryBandits，这是一种基于17个语言特征敏感性的奖励模型设计重写策略的多臂 bandit 框架，从而主动引导LLMs远离生成幻觉。在13个不同的问答基准测试和每数据集1,050个词形变异查询上，我们的上下文QueryBandit（Thompson Sampling）的胜出率为87.5%，优于无重写基线，分别优于零-shot 静态提示（“改写”或“扩展”）42.6%和60.3%。因此，我们通过采取查询重写的形式进行干预，实证验证了QueryBandits在缓解幻觉方面的有效性。有趣的是，一些静态提示策略，这些策略构成了当前查询重写文献中的相当一部分，其累积后悔值高于无重写基线，表明静态重写可能会加剧幻觉。此外，我们发现收敛的每臂回归特征权重向量表明，并不存在适用于所有查询的单一最优重写策略。在此背景下，通过QueryBandits利用语义特征进行指导重写可以通过前向传递机制显著改变输出行为，从而避免重新训练或基于梯度的适应。 

---
# DecoMind: A Generative AI System for Personalized Interior Design Layouts 

**Title (ZH)**: DecoMind: 个性化室内设计布局的生成型AI系统 

**Authors**: Reema Alshehri, Rawan Alotaibi, Leen Almasri, Rawan Altaweel  

**Link**: [PDF](https://arxiv.org/pdf/2508.16696)  

**Abstract**: This paper introduces a system for generating interior design layouts based on user inputs, such as room type, style, and furniture preferences. CLIP extracts relevant furniture from a dataset, and a layout that contains furniture and a prompt are fed to Stable Diffusion with ControlNet to generate a design that incorporates the selected furniture. The design is then evaluated by classifiers to ensure alignment with the user's inputs, offering an automated solution for realistic interior design. 

**Abstract (ZH)**: 基于用户输入生成室内设计布局的系统：CLIP提取相关家具并与Stable Diffusion和ControlNet协作生成设计，并通过分类器评估以确保与用户输入一致，提供一种现实主义室内设计的自动化解决方案。 

---
# Do Cognitively Interpretable Reasoning Traces Improve LLM Performance? 

**Title (ZH)**: 认知可解释推理痕迹能否提高大规模语言模型性能？ 

**Authors**: Siddhant Bhambri, Upasana Biswas, Subbarao Kambhampati  

**Link**: [PDF](https://arxiv.org/pdf/2508.16695)  

**Abstract**: Recent progress in reasoning-oriented Large Language Models (LLMs) has been driven by introducing Chain-of-Thought (CoT) traces, where models generate intermediate reasoning traces before producing an answer. These traces, as in DeepSeek R1, are not only used to guide inference but also serve as supervision signals for distillation into smaller models. A common but often implicit assumption is that CoT traces should be semantically meaningful and interpretable to the end user. While recent research questions the need for semantic nature of these traces, in this paper, we ask: ``\textit{Must CoT reasoning traces be interpretable to enhance LLM task performance?}" We investigate this question in the Open Book Question-Answering domain by supervised fine-tuning LLaMA and Qwen models on four types of reasoning traces: (1) DeepSeek R1 traces, (2) LLM-generated summaries of R1 traces, (3) LLM-generated post-hoc explanations of R1 traces, and (4) algorithmically generated verifiably correct traces. To quantify the trade-off between interpretability and performance, we further conduct a human-subject study with 100 participants rating the interpretability of each trace type. Our results reveal a striking mismatch: while fine-tuning on R1 traces yields the strongest performance, participants judged these traces to be the least interpretable. These findings suggest that it is useful to decouple intermediate tokens from end user interpretability. 

**Abstract (ZH)**: Recent进展在面向推理的大型语言模型中的进展受到在生成答案前引入推理链（Chain-of-Thought，CoT）痕迹的驱动。这些痕迹不仅用于指导推理，还用于蒸馏较小模型中的监督信号。一个常见的但往往隐含的假设是，CoT痕迹应该是语义上有意义且可解释的。尽管最近的研究质疑这些痕迹的语义性质，本文询问：“CoT推理痕迹是否必须可解释才能增强大型语言模型的性能？”我们在开放书问答领域通过监督微调LLaMA和Qwen模型四种类型的推理痕迹来探究这个问题：（1）DeepSeek R1痕迹，（2）由大型语言模型生成的R1痕迹总结，（3）由大型语言模型生成的R1痕迹后置解释，（4）算法生成的可验证正确痕迹。为进一步量化可解释性和性能之间的权衡，我们还进行了一项包含100名参与者的人类被试研究，评估每种痕迹类型的可解释性。我们的结果显示了一个令人惊讶的不匹配：虽然在R1痕迹上进行微调实现了最佳性能，但参与者认为这些痕迹是最难以解释的。这些发现表明，将中间标记与最终用户的可解释性分离是有用的。 

---
# Making AI Inevitable: Historical Perspective and the Problems of Predicting Long-Term Technological Change 

**Title (ZH)**: 让AI不可避免：历史视角与预测长期技术变革的问题 

**Authors**: Mark Fisher, John Severini  

**Link**: [PDF](https://arxiv.org/pdf/2508.16692)  

**Abstract**: This study demonstrates the extent to which prominent debates about the future of AI are best understood as subjective, philosophical disagreements over the history and future of technological change rather than as objective, material disagreements over the technologies themselves. It focuses on the deep disagreements over whether artificial general intelligence (AGI) will prove transformative for human society; a question that is analytically prior to that of whether this transformative effect will help or harm humanity. The study begins by distinguishing two fundamental camps in this debate. The first of these can be identified as "transformationalists," who argue that continued AI development will inevitably have a profound effect on society. Opposed to them are "skeptics," a more eclectic group united by their disbelief that AI can or will live up to such high expectations. Each camp admits further "strong" and "weak" variants depending on their tolerance for epistemic risk. These stylized contrasts help to identify a set of fundamental questions that shape the camps' respective interpretations of the future of AI. Three questions in particular are focused on: the possibility of non-biological intelligence, the appropriate time frame of technological predictions, and the assumed trajectory of technological development. In highlighting these specific points of non-technical disagreement, this study demonstrates the wide range of different arguments used to justify either the transformationalist or skeptical position. At the same time, it highlights the strong argumentative burden of the transformationalist position, the way that belief in this position creates competitive pressures to achieve first-mover advantage, and the need to widen the concept of "expertise" in debates surrounding the future development of AI. 

**Abstract (ZH)**: 这一研究展示了对未来AI的广泛关注最好被理解为关于技术变革历史和未来的主观哲学分歧，而非对技术本身的具体物质分歧。该研究集中在对通用人工智能（AGI）是否会对人类社会产生变革性影响的深刻分歧上；这是一个比这种变革性影响是利是害更为基本的问题。研究一开始便区分了这场辩论中的两大基本阵营。第一个阵营可被识别为“变革主义者”，他们认为持续的AI发展将不可避免地对社会产生深远影响。反对他们的是“怀疑论者”，这是一个更为多元的群体，统一于他们不相信AI能或会达到如此高的期望。每个阵营又进一步分为“强硬派”和“温和派”变体，这取决于他们对认识论风险的容忍度。这些理论上的对比有助于识别塑造两大阵营对AI未来理解的若干核心问题。特别是关注三个问题：非生物学智能的可能性、技术预测的适当时间框架以及技术发展的假设轨迹。通过强调这些具体的非技术性分歧，该研究展示了支持变革主义者或怀疑论者的不同论点的广泛范围。同时，该研究强调了变革主义者立场的强大论点论证负担、这种立场如何创造了抢先优势的竞争压力以及在关于AI未来发展辩论中扩展“专家”概念的必要性。 

---
# Cybernaut: Towards Reliable Web Automation 

**Title (ZH)**: Cybernaut: 向可靠网页自动化迈进 

**Authors**: Ankur Tomar, Hengyue Liang, Indranil Bhattacharya, Natalia Larios, Francesco Carbone  

**Link**: [PDF](https://arxiv.org/pdf/2508.16688)  

**Abstract**: The emergence of AI-driven web automation through Large Language Models (LLMs) offers unprecedented opportunities for optimizing digital workflows. However, deploying such systems within industry's real-world environments presents four core challenges: (1) ensuring consistent execution, (2) accurately identifying critical HTML elements, (3) meeting human-like accuracy in order to automate operations at scale and (4) the lack of comprehensive benchmarking data on internal web applications. Existing solutions are primarily tailored for well-designed, consumer-facing websites (e.g., this http URL, this http URL) and fall short in addressing the complexity of poorly-designed internal web interfaces. To address these limitations, we present Cybernaut, a novel framework to ensure high execution consistency in web automation agents designed for robust enterprise use. Our contributions are threefold: (1) a Standard Operating Procedure (SOP) generator that converts user demonstrations into reliable automation instructions for linear browsing tasks, (2) a high-precision HTML DOM element recognition system tailored for the challenge of complex web interfaces, and (3) a quantitative metric to assess execution consistency. The empirical evaluation on our internal benchmark demonstrates that using our framework enables a 23.2% improvement (from 72% to 88.68%) in task execution success rate over the browser_use. Cybernaut identifies consistent execution patterns with 84.7% accuracy, enabling reliable confidence assessment and adaptive guidance during task execution in real-world systems. These results highlight Cybernaut's effectiveness in enterprise-scale web automation and lay a foundation for future advancements in web automation. 

**Abstract (ZH)**: 基于大型语言模型（LLMs）的AI驱动网页自动化的发展为优化数字工作流提供了前所未有的机会。然而，在工业实际环境中部署此类系统带来了四个核心挑战：（1）确保一致执行，（2）准确识别关键HTML元素，（3）达到类似人类的准确性以大规模自动化操作，（4）缺乏针对内部网页应用的全面基准数据。现有解决方案主要针对设计良好的面向消费者的网站（例如：this http URL, this http URL），对于复杂设计的内部网页界面则力有未逮。为解决这些限制，我们提出了Cybernaut，一种新型框架，以确保适用于企业级使用的网页自动化代理的一致执行。我们的贡献包括：（1）一个标准作业程序（SOP）生成器，能够将用户演示转化为可靠的自动化指令，用于线性浏览任务；（2）一个高精度的HTML DOM元素识别系统，旨在应对复杂网页界面的挑战；（3）一个衡量执行一致性的定量指标。在我们的内部基准测试上的实证评估表明，使用该框架可以将任务执行成功率从72%提高到88.68%（提高23.2%）。Cybernaut以84.7%的准确性识别出一致的执行模式，使在实际系统中执行任务时能够提供可靠的信心评估和适应性指导。这些结果突显了Cybernaut在企业级网页自动化中的有效性，并为其未来的发展奠定了基础。 

---
# STGAtt: A Spatial-Temporal Unified Graph Attention Network for Traffic Flow Forecasting 

**Title (ZH)**: STGAtt：一种用于交通流量预测的时空统一图注意力网络 

**Authors**: Zhuding Liang, Jianxun Cui, Qingshuang Zeng, Feng Liu, Nenad Filipovic, Tijana Geroski  

**Link**: [PDF](https://arxiv.org/pdf/2508.16685)  

**Abstract**: Accurate and timely traffic flow forecasting is crucial for intelligent transportation systems. This paper presents a novel deep learning model, the Spatial-Temporal Unified Graph Attention Network (STGAtt). By leveraging a unified graph representation and an attention mechanism, STGAtt effectively captures complex spatial-temporal dependencies. Unlike methods relying on separate spatial and temporal dependency modeling modules, STGAtt directly models correlations within a Spatial-Temporal Unified Graph, dynamically weighing connections across both dimensions. To further enhance its capabilities, STGAtt partitions traffic flow observation signal into neighborhood subsets and employs a novel exchanging mechanism, enabling effective capture of both short-range and long-range correlations. Extensive experiments on the PEMS-BAY and SHMetro datasets demonstrate STGAtt's superior performance compared to state-of-the-art baselines across various prediction horizons. Visualization of attention weights confirms STGAtt's ability to adapt to dynamic traffic patterns and capture long-range dependencies, highlighting its potential for real-world traffic flow forecasting applications. 

**Abstract (ZH)**: 准确及时的交通流预测对于智能交通系统至关重要。本文提出了一种新颖的深度学习模型，即空间-时间统一图注意力网络（STGAtt）。通过利用统一的图表示和注意力机制，STGAtt 有效地捕捉了复杂的空-时依赖关系。与依赖于独立的空间和时间依赖性建模模块的方法不同，STGAtt 直接在空间-时间统一图中建模内部的联系，并动态权衡两个维度上的连接。为了进一步提升其能力，STGAtt 将交通流观测信号划分为邻域子集，并采用了一种新型的交换机制，能够有效地捕捉短程和远程的依赖关系。在 PEMS-BAY 和 SHMetro 数据集上的 extensive 实验表明，STGAtt 在各种预测时间范围内的性能优于最先进的基线方法。注意力权重的可视化结果证实了 STGAtt 适应动态交通模式并捕捉远程依赖性的能力，突显了其在实际交通流预测应用中的潜力。 

---
# CALR: Corrective Adaptive Low-Rank Decomposition for Efficient Large Language Model Layer Compression 

**Title (ZH)**: CALR: 正确适应性低秩分解以实现高效的大型语言模型层压缩 

**Authors**: Muchammad Daniyal Kautsar, Afra Majida Hariono, Widyawan, Syukron Abu Ishaq Alfarozi, Kuntpong Wararatpanya  

**Link**: [PDF](https://arxiv.org/pdf/2508.16680)  

**Abstract**: Large Language Models (LLMs) present significant deployment challenges due to their immense size and computational requirements. Model compression techniques are essential for making these models practical for resource-constrained environments. A prominent compression strategy is low-rank factorization via Singular Value Decomposition (SVD) to reduce model parameters by approximating weight matrices. However, standard SVD focuses on minimizing matrix reconstruction error, often leading to a substantial loss of the model's functional performance. This performance degradation occurs because existing methods do not adequately correct for the functional information lost during compression. To address this gap, we introduce Corrective Adaptive Low-Rank Decomposition (CALR), a two-component compression approach. CALR combines a primary path of SVD-compressed layers with a parallel, learnable, low-rank corrective module that is explicitly trained to recover the functional residual error. Our experimental evaluation on SmolLM2-135M, Qwen3-0.6B, and Llama-3.2-1B, demonstrates that CALR can reduce parameter counts by 26.93% to 51.77% while retaining 59.45% to 90.42% of the original model's performance, consistently outperforming LaCo, ShortGPT, and LoSparse. CALR's success shows that treating functional information loss as a learnable signal is a highly effective compression paradigm. This approach enables the creation of significantly smaller, more efficient LLMs, advancing their accessibility and practical deployment in real-world applications. 

**Abstract (ZH)**: 大规模语言模型（LLMs）由于其巨大的规模和计算需求，在部署时面临显著的挑战。模型压缩技术对于使这些模型在资源受限环境中实用至关重要。一种主要的压缩策略是通过奇异值分解（SVD）进行低秩分解，以通过近似权重矩阵来减少模型参数。然而，标准的SVD方法主要关注于最小化矩阵重构误差，往往会导致模型功能性性能的显著下降。这种性能下降是因为现有的方法未能充分纠正压缩过程中丢失的功能信息。为了解决这一问题，我们引入了一种两部分压缩方法——补救自适应低秩分解（CALR）。CALR 结合了一条使用 SVD 压缩层的主要路径，以及一条并行的、可学习的低秩补救模块，该模块明确训练以恢复功能残差误差。我们的实验评估表明，CALR 可以使参数数量减少 26.93% 至 51.77%，同时保留原始模型性能的 59.45% 至 90.42%，并且一贯优于 LaCo、ShortGPT 和 LoSparse。CALR 的成功表明，将功能性信息的损失视为可学习的信号是一种非常有效的压缩范式。这种方法能够创建显著更小、更高效的 LLMs，促进其在实际应用中的可访问性和实际部署。 

---
# Recall-Extend Dynamics: Enhancing Small Language Models through Controlled Exploration and Refined Offline Integration 

**Title (ZH)**: 召回扩展动力学：通过受控探索和精炼离线集成增强小型语言模型 

**Authors**: Zhong Guan, Likang Wu, Hongke Zhao, Jiahui Wang, Le Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.16677)  

**Abstract**: Many existing studies have achieved significant improvements in the reasoning capabilities of large language models (LLMs) through reinforcement learning with verifiable rewards (RLVR), while the enhancement of reasoning abilities in small language models (SLMs) has not yet been sufficiently explored. Combining distilled data from larger models with RLVR on small models themselves is a natural approach, but it still faces various challenges and issues. Therefore, we propose \textit{\underline{R}}ecall-\textit{\underline{E}}xtend \textit{\underline{D}}ynamics(RED): Enhancing Small Language Models through Controlled Exploration and Refined Offline Integration. In this paper, we explore the perspective of varying exploration spaces, balancing offline distillation with online reinforcement learning. Simultaneously, we specifically design and optimize for the insertion problem within offline data. By monitoring the ratio of entropy changes in the model concerning offline and online data, we regulate the weight of offline-SFT, thereby addressing the issues of insufficient exploration space in small models and the redundancy and complexity during the distillation process. Furthermore, to tackle the distribution discrepancies between offline data and the current policy, we design a sample-accuracy-based policy shift mechanism that dynamically chooses between imitating offline distilled data and learning from its own policy. 

**Abstract (ZH)**: Recall-Extend Dynamics (RED): 提升小语言模型通过受控探索和精炼离线集成 

---
# MedRepBench: A Comprehensive Benchmark for Medical Report Interpretation 

**Title (ZH)**: MedRepBench: 医学报告解读的综合基准 

**Authors**: Fangxin Shang, Yuan Xia, Dalu Yang, Yahui Wang, Binglin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16674)  

**Abstract**: Medical report interpretation plays a crucial role in healthcare, enabling both patient-facing explanations and effective information flow across clinical systems. While recent vision-language models (VLMs) and large language models (LLMs) have demonstrated general document understanding capabilities, there remains a lack of standardized benchmarks to assess structured interpretation quality in medical reports. We introduce MedRepBench, a comprehensive benchmark built from 1,900 de-identified real-world Chinese medical reports spanning diverse departments, patient demographics, and acquisition formats. The benchmark is designed primarily to evaluate end-to-end VLMs for structured medical report understanding. To enable controlled comparisons, we also include a text-only evaluation setting using high-quality OCR outputs combined with LLMs, allowing us to estimate the upper-bound performance when character recognition errors are minimized. Our evaluation framework supports two complementary protocols: (1) an objective evaluation measuring field-level recall of structured clinical items, and (2) an automated subjective evaluation using a powerful LLM as a scoring agent to assess factuality, interpretability, and reasoning quality. Based on the objective metric, we further design a reward function and apply Group Relative Policy Optimization (GRPO) to improve a mid-scale VLM, achieving up to 6% recall gain. We also observe that the OCR+LLM pipeline, despite strong performance, suffers from layout-blindness and latency issues, motivating further progress toward robust, fully vision-based report understanding. 

**Abstract (ZH)**: 医疗报告解读在医疗保健中扮演着至关重要的角色，能够提供面向患者的解释并促进临床系统之间的有效信息流动。尽管近期的视觉-语言模型和大规模语言模型展示了通用文档理解能力，但在评估医疗报告结构化解读质量方面仍然缺乏标准化基准。我们介绍了MedRepBench，这是一个基于1900份去标识化的现实世界中文医疗报告构建的综合基准，涵盖多个科室、患者人口统计特征和获取格式。该基准主要用于评估端到端的视觉-语言模型在结构化医疗报告理解中的性能。为了进行可控比较，我们还引入了一个仅文本评估设置，结合高质量的OCR输出和语言模型，以估计在最小化字符识别错误时的上界性能。我们的评估框架支持两种互补的协议：（1）基于客观指标的字段级别召回率的评估，以及（2）使用强大语言模型作为评分代理的自动主观评估，以评估事实性、可解释性和推理质量。根据客观指标，我们进一步设计了一个奖励函数，并应用组相对策略优化（GRPO）来提升中规模的视觉-语言模型，实现了高达6%的召回率提升。我们还观察到，尽管OCR+LLM流水线表现出色，但仍存在布局盲视和延迟问题，这促使我们进一步向稳健的整体视觉报告理解方向发展。 

---
# Invisible Filters: Cultural Bias in Hiring Evaluations Using Large Language Models 

**Title (ZH)**: 隐形滤镜：大型语言模型在招聘评估中的文化偏见 

**Authors**: Pooja S. B. Rao, Laxminarayen Nagarajan Venkatesan, Mauro Cherubini, Dinesh Babu Jayagopi  

**Link**: [PDF](https://arxiv.org/pdf/2508.16673)  

**Abstract**: Artificial Intelligence (AI) is increasingly used in hiring, with large language models (LLMs) having the potential to influence or even make hiring decisions. However, this raises pressing concerns about bias, fairness, and trust, particularly across diverse cultural contexts. Despite their growing role, few studies have systematically examined the potential biases in AI-driven hiring evaluation across cultures. In this study, we conduct a systematic analysis of how LLMs assess job interviews across cultural and identity dimensions. Using two datasets of interview transcripts, 100 from UK and 100 from Indian job seekers, we first examine cross-cultural differences in LLM-generated scores for hirability and related traits. Indian transcripts receive consistently lower scores than UK transcripts, even when they were anonymized, with disparities linked to linguistic features such as sentence complexity and lexical diversity. We then perform controlled identity substitutions (varying names by gender, caste, and region) within the Indian dataset to test for name-based bias. These substitutions do not yield statistically significant effects, indicating that names alone, when isolated from other contextual signals, may not influence LLM evaluations. Our findings underscore the importance of evaluating both linguistic and social dimensions in LLM-driven evaluations and highlight the need for culturally sensitive design and accountability in AI-assisted hiring. 

**Abstract (ZH)**: 人工智能（AI）在招聘中的应用 increasingly used in hiring, with large language models (LLMs) having the potential to influence or even make hiring decisions. However, this raises pressing concerns about bias, fairness, and trust, particularly across diverse cultural contexts. 

---
# The AI Model Risk Catalog: What Developers and Researchers Miss About Real-World AI Harms 

**Title (ZH)**: AI模型风险目录：开发者和研究人员忽视的现实世界AI危害 

**Authors**: Pooja S. B. Rao, Sanja Šćepanović, Dinesh Babu Jayagopi, Mauro Cherubini, Daniele Quercia  

**Link**: [PDF](https://arxiv.org/pdf/2508.16672)  

**Abstract**: We analyzed nearly 460,000 AI model cards from Hugging Face to examine how developers report risks. From these, we extracted around 3,000 unique risk mentions and built the \emph{AI Model Risk Catalog}. We compared these with risks identified by researchers in the MIT Risk Repository and with real-world incidents from the AI Incident Database. Developers focused on technical issues like bias and safety, while researchers emphasized broader social impacts. Both groups paid little attention to fraud and manipulation, which are common harms arising from how people interact with AI. Our findings show the need for clearer, structured risk reporting that helps developers think about human-interaction and systemic risks early in the design process. The catalog and paper appendix are available at: this https URL. 

**Abstract (ZH)**: 我们分析了Hugging Face上的近46万个AI模型卡片，以考察开发者如何报告风险。我们从中提取了大约3000个独特的风险提及，构建了《AI模型风险目录》。我们将这些风险与MIT风险存储库中识别的风险以及AI事故数据库中的实际事件进行了比较。开发人员主要关注技术问题如偏见和安全性，而研究人员则更侧重于更广泛的社会影响。两组对欺诈和操纵这类常见的交互造成的危害关注度不高。我们的研究结果表明，需要更清晰和结构化的风险报告，以帮助开发者在设计初期就考虑到人类交互和系统性风险。该目录及其论文附录可在以下链接获取：this https URL。 

---
# Reflective Paper-to-Code Reproduction Enabled by Fine-Grained Verification 

**Title (ZH)**: 细粒度验证支持的反思性纸笔代码重现 

**Authors**: Mingyang Zhou, Quanming Yao, Lun Du, Lanning Wei, Da Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.16671)  

**Abstract**: Reproducing machine learning papers is essential for scientific progress but remains challenging for both humans and automated agents. Existing agent-based methods often struggle to fully and accurately reproduce implementation details such as mathematical formulas and algorithmic logic. Previous studies show that reflection with explicit feedback improves agent performance. However, current paper reproduction methods fail to effectively adopt this strategy. This gap mainly arises from the diverse paper patterns, complex method modules, and varied configurations encountered in research papers. Motivated by how humans use systematic checklists to efficiently debug complex code, we propose \textbf{RePro}, a \textbf{Re}flective Paper-to-Code \textbf{Repro}duction framework that automatically extracts a paper's fingerprint, referring to a comprehensive set of accurate and atomic criteria serving as high-quality supervisory signals. The framework first generates code based on the extracted information, and then leverages the fingerprint within iterative verification and refinement loop. This approach systematically detects discrepancies and produces targeted revisions to align generated code with the paper's implementation details. Extensive experiments on the PaperBench Code-Dev benchmark have been conducted, RePro achieves 13.0\% performance gap over baselines, and it correctly revises complex logical and mathematical criteria in reflecting, on which the effectiveness is obvious. 

**Abstract (ZH)**: 机器学习论文的再现对于科学进步至关重要，但对人类和自动化代理而言仍具有挑战性。现有的基于代理的方法往往难以全面准确地再现实现细节，如数学公式和算法逻辑。先前的研究表明，带有显反馈的反思可以提升代理性能。然而，当前的论文再现方法未能有效采用这一策略。这一差距主要源于研究论文中遇到的多样论文模式、复杂的方法模块和不同的配置。受人类如何使用系统化检查列表高效调试复杂代码的启发，我们提出了一种名为RePro的Reflective Paper-to-Code Reproduction框架，自动提取论文指纹，参考一个全面的、准确且原子的标准集合作为高质量的监督信号。该框架首先基于提取的信息生成代码，然后利用指纹在迭代验证和改进循环中进行验证和细化。这种方法系统性地检测不一致之处并生成针对性修订，以使生成的代码与论文的实现细节保持一致。在PaperBench Code-Dev基准测试上的 extensive 实验表明，RePro 的性能相较于baseline提升了13.0%，在反映过程中正确修订了复杂的逻辑和数学标准，其有效性显而易见。 

---
# COVID19 Prediction Based On CT Scans Of Lungs Using DenseNet Architecture 

**Title (ZH)**: 基于肺部CT扫描的COVID-19预测：采用DenseNet架构 

**Authors**: Deborup Sanyal  

**Link**: [PDF](https://arxiv.org/pdf/2508.16670)  

**Abstract**: COVID19 took the world by storm since December 2019. A highly infectious communicable disease, COVID19 is caused by the SARSCoV2 virus. By March 2020, the World Health Organization (WHO) declared COVID19 as a global pandemic. A pandemic in the 21st century after almost 100 years was something the world was not prepared for, which resulted in the deaths of around 1.6 million people worldwide. The most common symptoms of COVID19 were associated with the respiratory system and resembled a cold, flu, or pneumonia. After extensive research, doctors and scientists concluded that the main reason for lives being lost due to COVID19 was failure of the respiratory system. Patients were dying gasping for breath. Top healthcare systems of the world were failing badly as there was an acute shortage of hospital beds, oxygen cylinders, and ventilators. Many were dying without receiving any treatment at all. The aim of this project is to help doctors decide the severity of COVID19 by reading the patient's Computed Tomography (CT) scans of the lungs. Computer models are less prone to human error, and Machine Learning or Neural Network models tend to give better accuracy as training improves over time. We have decided to use a Convolutional Neural Network model. Given that a patient tests positive, our model will analyze the severity of COVID19 infection within one month of the positive test result. The severity of the infection may be promising or unfavorable (if it leads to intubation or death), based entirely on the CT scans in the dataset. 

**Abstract (ZH)**: COVID-19自2019年12月席卷世界。COVID-19是由SARS-CoV-2病毒引起的一种高度传染性疾病。到2020年3月，世界卫生组织宣布COVID-19成为全球大流行病。这是自21世纪初近100年来首次出现的全球大流行病，导致全球约160万人死亡。COVID-19最常见的症状与呼吸系统有关，类似于普通感冒、流感或肺炎。通过广泛的研究，医生和科学家得出结论，由于COVID-19导致的死亡主要原因在于呼吸系统的失败。患者主要是因为缺氧而气喘吁吁。全球顶尖的医疗系统面临着严重的困境，因为医院床位、氧气罐和呼吸机极度短缺。许多人甚至没有接受任何治疗就死亡了。本项目旨在通过分析患者的肺部计算机断层扫描（CT）来帮助医生判断COVID-19的严重程度。计算机模型较少出现人为错误，而机器学习或神经网络模型随着时间的推移训练会提升准确性。我们决定使用卷积神经网络模型。给定患者测试呈阳性，我们的模型将在阳性检测结果一个月内分析COVID-19感染的严重程度。感染的严重程度可能是有希望的或不利的（如果导致气管插管或死亡），完全取决于数据集中的CT扫描结果。 

---
# Situational Awareness as the Imperative Capability for Disaster Resilience in the Era of Complex Hazards and Artificial Intelligence 

**Title (ZH)**: 态势感知作为复杂灾害与人工智能时代灾害韧性的重要能力 

**Authors**: Hongrak Pak, Ali Mostafavi  

**Link**: [PDF](https://arxiv.org/pdf/2508.16669)  

**Abstract**: Disasters frequently exceed established hazard models, revealing blind spots where unforeseen impacts and vulnerabilities hamper effective response. This perspective paper contends that situational awareness (SA)-the ability to perceive, interpret, and project dynamic crisis conditions-is an often overlooked yet vital capability for disaster resilience. While risk mitigation measures can reduce known threats, not all hazards can be neutralized; truly adaptive resilience hinges on whether organizations rapidly detect emerging failures, reconcile diverse data sources, and direct interventions where they matter most. We present a technology-process-people roadmap, demonstrating how real-time hazard nowcasting, interoperable workflows, and empowered teams collectively transform raw data into actionable insight. A system-of-systems approach enables federated data ownership and modular analytics, so multiple agencies can share timely updates without sacrificing their distinct operational models. Equally crucial, structured sense-making routines and cognitive load safeguards help humans remain effective decision-makers amid data abundance. By framing SA as a socio-technical linchpin rather than a peripheral add-on, this paper spotlights the urgency of elevating SA to a core disaster resilience objective. We conclude with recommendations for further research-developing SA metrics, designing trustworthy human-AI collaboration, and strengthening inclusive data governance-to ensure that communities are equipped to cope with both expected and unexpected crises. 

**Abstract (ZH)**: 灾害频繁超出已建立的危害模型，揭示出存在未预见影响和脆弱性的盲点，这些盲点阻碍了有效的应对。本文观点认为，情况感知（SA）——即感知、解释和预测动态危机条件的能力——是灾害韧性中一个常被忽视但至关重要的能力。虽然风险管理措施可以降低已知威胁，但并非所有危害都能被消除；真正适应性的韧性取决于组织能否迅速检测到新兴的失败、协调多元数据源，并将干预措施集中在最关键的地方。本文提出了一条技术-流程-人员的道路图，展示出如何通过实时危害现在casting、互操作的工作流程以及赋能的团队将原始数据转化为可行动的洞见。系统中的系统的方法使联邦数据拥有权和模块化分析得以实现，从而多个机构可以共享及时更新而不牺牲其独特的操作模型。同样重要的是，结构化的意义构建范式和认知负担的安全措施有助于人们在数据过剩的情况下保持有效的决策能力。通过将SA框架为社会技术的关键节点而非外围附加部分，本文突显了提高SA作为核心灾害韧性目标的紧迫性。最后，本文提出了进一步研究的建议，包括开发SA指标、设计值得信赖的人工智能合作以及加强包容性数据治理，以确保社区能够应对预期和未预期的危机。 

---
# Trust but Verify! A Survey on Verification Design for Test-time Scaling 

**Title (ZH)**: 信任但验证！关于测试时扩展性验证设计的综述 

**Authors**: V Venktesh, Mandeep rathee, Avishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2508.16665)  

**Abstract**: Test-time scaling (TTS) has emerged as a new frontier for scaling the performance of Large Language Models. In test-time scaling, by using more computational resources during inference, LLMs can improve their reasoning process and task performance. Several approaches have emerged for TTS such as distilling reasoning traces from another model or exploring the vast decoding search space by employing a verifier. The verifiers serve as reward models that help score the candidate outputs from the decoding process to diligently explore the vast solution space and select the best outcome. This paradigm commonly termed has emerged as a superior approach owing to parameter free scaling at inference time and high performance gains. The verifiers could be prompt-based, fine-tuned as a discriminative or generative model to verify process paths, outcomes or both. Despite their widespread adoption, there is no detailed collection, clear categorization and discussion of diverse verification approaches and their training mechanisms. In this survey, we cover the diverse approaches in the literature and present a unified view of verifier training, types and their utility in test-time scaling. Our repository can be found at this https URL. 

**Abstract (ZH)**: 测试时扩展（TTS）已成为扩展大型语言模型性能的新前沿。在测试时扩展中，通过在推理时使用更多计算资源，LLMs可以通过改进其推理过程和任务性能来受益。已经出现了多种TTS的方法，例如从另一个模型中抽取推理轨迹或通过使用验证器探索解码搜索空间。验证器作为奖励模型，帮助对解码过程的候选输出进行评分，以谨慎地探索广阔的解空间并选择最佳结果。由于这种范式在推理时参数无约束扩展且性能提升显著，验证器通常被认为是优越的方法。验证器可以基于提示、微调为判别或生成模型来验证过程路径、结果或两者。尽管它们被广泛采用，但在文献中还没有详细的整理、清晰的分类和讨论各种验证方法及其训练机制。在这篇综述中，我们涵盖了文献中的各种方法，并提供了一个统一的验证器训练、类型及其在测试时扩展中的作用视图。我们的仓库可以在以下链接找到：this https URL。 

---
# The Loupe: A Plug-and-Play Attention Module for Amplifying Discriminative Features in Vision Transformers 

**Title (ZH)**: The Loupe: 一种用于放大视觉变换器中鉴别性特征的即插即用注意力模块 

**Authors**: Naren Sengodan  

**Link**: [PDF](https://arxiv.org/pdf/2508.16663)  

**Abstract**: Fine-Grained Visual Classification (FGVC) is a critical and challenging area within computer vision, demanding the identification of highly subtle, localized visual cues. The importance of FGVC extends to critical applications such as biodiversity monitoring and medical diagnostics, where precision is paramount. While large-scale Vision Transformers have achieved state-of-the-art performance, their decision-making processes often lack the interpretability required for trust and verification in such domains. In this paper, we introduce The Loupe, a novel, lightweight, and plug-and-play attention module designed to be inserted into pre-trained backbones like the Swin Transformer. The Loupe is trained end-to-end with a composite loss function that implicitly guides the model to focus on the most discriminative object parts without requiring explicit part-level annotations. Our unique contribution lies in demonstrating that a simple, intrinsic attention mechanism can act as a powerful regularizer, significantly boosting performance while simultaneously providing clear visual explanations. Our experimental evaluation on the challenging CUB-200-2011 dataset shows that The Loupe improves the accuracy of a Swin-Base model from 85.40% to 88.06%, a significant gain of 2.66%. Crucially, our qualitative analysis of the learned attention maps reveals that The Loupe effectively localizes semantically meaningful features, providing a valuable tool for understanding and trusting the model's decision-making process. 

**Abstract (ZH)**: 细粒度视觉分类（FGVC）是计算机视觉中的一个关键且具有挑战性的领域，要求识别高度微妙且局部化的视觉线索。细粒度视觉分类在生物多样性监测和医疗诊断等关键应用中具有重要意义，其中精确性至关重要。虽然大规模的视觉变换器已经实现了最先进的性能，但其决策过程往往缺乏在这些领域中所需的信任和验证的可解释性。在本文中，我们引入了《观琴》，这是一种新颖的、轻量级且即插即用的注意模块，旨在插入如Swin Transformer等预训练骨干网络中。《观琴》通过复合损失函数进行端到端训练，隐式地指导模型专注于最具判别性的对象部分，而不需要显式的部分级注释。我们的独特贡献在于展示了简单的内在注意机制作为一个强大的正则化器的作用，显著提升了性能同时提供了清晰的可视化解释。我们在具有挑战性的CUB-200-2011数据集上的实验评价显示，《观琴》将Swin-Base模型的准确性从85.40%提升到88.06%，显著提高了2.66%。至关重要的是，我们对学习到的注意图的定性分析表明，《观琴》有效地定位了语义上有意义的特征，提供了一个理解并信任模型决策过程的宝贵工具。 

---
# Optimizing Hyper parameters in CNN for Soil Classification using PSO and Whale Optimization Algorithm 

**Title (ZH)**: 使用粒子群优化和鲸鱼优化算法在CNN中优化土壤分类的超参数 

**Authors**: Yasir Nooruldeen Ibrahim, Fawziya Mahmood Ramo, Mahmood Siddeeq Qadir, Muna Jaffer Al-Shamdeen  

**Link**: [PDF](https://arxiv.org/pdf/2508.16660)  

**Abstract**: Classifying soil images contributes to better land management, increased agricultural output, and practical solutions for environmental issues. The development of various disciplines, particularly agriculture, civil engineering, and natural resource management, is aided by understanding of soil quality since it helps with risk reduction, performance improvement, and sound decision-making . Artificial intelligence has recently been used in a number of different fields. In this study, an intelligent model was constructed using Convolutional Neural Networks to classify soil kinds, and machine learning algorithms were used to enhance the performance of soil classification . To achieve better implementation and performance of the Convolutional Neural Networks algorithm and obtain valuable results for the process of classifying soil type images, swarm algorithms were employed to obtain the best performance by choosing Hyper parameters for the Convolutional Neural Networks network using the Whale optimization algorithm and the Particle swarm optimization algorithm, and comparing the results of using the two algorithms in the process of multiple classification of soil types. The Accuracy and F1 measures were adopted to test the system, and the results of the proposed work were efficient result 

**Abstract (ZH)**: 土壤图像分类有助于改善土地管理、提高农业生产率，并为环境问题提供实际解决方案。通过对土壤质量的理解促进各学科的发展，尤其是农业、土木工程和自然资源管理，有助于风险降低、性能提升和科学决策。人工智能最近在多个领域得到了应用。在本研究中，利用卷积神经网络构建了一个智能模型以分类土壤类型，并使用机器学习算法提高土壤分类性能。为了更好地实施卷积神经网络算法并获得土壤类型图像分类过程中的宝贵结果，使用鲸鱼优化算法和粒子群优化算法选择卷积神经网络的超参数以获取最佳性能，并比较两种算法在土壤类型多分类过程中的表现。采用准确率和F1度量测试系统，研究成果是有效的。 

---
# Enabling Multi-Agent Systems as Learning Designers: Applying Learning Sciences to AI Instructional Design 

**Title (ZH)**: 将多代理系统作为学习设计师：将学习科学应用于AI教学设计 

**Authors**: Jiayi Wang, Ruiwei Xiao, Xinying Hou, John Stamper  

**Link**: [PDF](https://arxiv.org/pdf/2508.16659)  

**Abstract**: K-12 educators are increasingly using Large Language Models (LLMs) to create instructional materials. These systems excel at producing fluent, coherent content, but often lack support for high-quality teaching. The reason is twofold: first, commercial LLMs, such as ChatGPT and Gemini which are among the most widely accessible to teachers, do not come preloaded with the depth of pedagogical theory needed to design truly effective activities; second, although sophisticated prompt engineering can bridge this gap, most teachers lack the time or expertise and find it difficult to encode such pedagogical nuance into their requests. This study shifts pedagogical expertise from the user's prompt to the LLM's internal architecture. We embed the well-established Knowledge-Learning-Instruction (KLI) framework into a Multi-Agent System (MAS) to act as a sophisticated instructional designer. We tested three systems for generating secondary Math and Science learning activities: a Single-Agent baseline simulating typical teacher prompts; a role-based MAS where agents work sequentially; and a collaborative MAS-CMD where agents co-construct activities through conquer and merge discussion. The generated materials were evaluated by 20 practicing teachers and a complementary LLM-as-a-judge system using the Quality Matters (QM) K-12 standards. While the rubric scores showed only small, often statistically insignificant differences between the systems, the qualitative feedback from educators painted a clear and compelling picture. Teachers strongly preferred the activities from the collaborative MAS-CMD, describing them as significantly more creative, contextually relevant, and classroom-ready. Our findings show that embedding pedagogical principles into LLM systems offers a scalable path for creating high-quality educational content. 

**Abstract (ZH)**: K-12教育者 increasingly using 大型语言模型 (LLMs) to create 教学材料。这些系统在生成流畅、连贯的内容方面表现出色，但往往缺乏高质量教学的支持。原因有两个：首先，商业LLM，如ChatGPT和Gemini，这是教师最常用到的，缺少足够的教学理论深度来设计真正有效果的活动；其次，尽管复杂的提示工程可以弥补这一差距，但大多数教师缺乏时间和专业知识，难以将这种教学细腻之处编码到他们的请求中。本研究将教学专业知识从用户的提示转移到LLM的内部架构中。我们将在多智能体系统（MAS）中嵌入成熟的知识-学习-教学（KLI）框架，作为高级教学设计师。我们测试了三种生成中学数学和科学学习活动的系统：单智能体基线模拟典型教师提示；基于角色的MAS，智能体按顺序工作；以及合作MAS-CMD，通过征服和合并讨论协作构建活动。生成的材料由20位在职教师和一个补充的LLM作为评判系统的 Quality Matters (QM) K-12标准进行评估。虽然评分标准显示系统之间仅存在小的、通常不具备统计意义的差异，但教育者的定性反馈描绘出了一幅明确而引人注目的画面。教师们强烈偏好合作MAS-CMD生成的活动，认为这些活动更加富有创造性、相关性强且能够在课堂上直接使用。我们的研究结果表明，将教学原理嵌入到LLM系统中为创建高质量教育内容提供了一种可扩展的途径。 

---
# HiCL: Hippocampal-Inspired Continual Learning 

**Title (ZH)**: hippocampal启发的连续学习 

**Authors**: Kushal Kapoor, Wyatt Mackey, Yiannis Aloimonos, Xiaomin Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.16651)  

**Abstract**: We propose HiCL, a novel hippocampal-inspired dual-memory continual learning architecture designed to mitigate catastrophic forgetting by using elements inspired by the hippocampal circuitry. Our system encodes inputs through a grid-cell-like layer, followed by sparse pattern separation using a dentate gyrus-inspired module with top-k sparsity. Episodic memory traces are maintained in a CA3-like autoassociative memory. Task-specific processing is dynamically managed via a DG-gated mixture-of-experts mechanism, wherein inputs are routed to experts based on cosine similarity between their normalized sparse DG representations and learned task-specific DG prototypes computed through online exponential moving averages. This biologically grounded yet mathematically principled gating strategy enables differentiable, scalable task-routing without relying on a separate gating network, and enhances the model's adaptability and efficiency in learning multiple sequential tasks. Cortical outputs are consolidated using Elastic Weight Consolidation weighted by inter-task similarity. Crucially, we incorporate prioritized replay of stored patterns to reinforce essential past experiences. Evaluations on standard continual learning benchmarks demonstrate the effectiveness of our architecture in reducing task interference, achieving near state-of-the-art results in continual learning tasks at lower computational costs. 

**Abstract (ZH)**: HiCL：一种启发自海马体的新型双记忆连续学习架构及其在减轻灾难性遗忘方面的应用 

---
# LatentFlow: Cross-Frequency Experimental Flow Reconstruction from Sparse Pressure via Latent Mapping 

**Title (ZH)**: 潜在流形：通过潜在映射从稀疏压力进行跨频率实验流重建 

**Authors**: Junle Liu, Chang Liu, Yanyu Ke, Qiuxiang Huang, Jiachen Zhao, Wenliang Chen, K.T. Tse, Gang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.16648)  

**Abstract**: Acquiring temporally high-frequency and spatially high-resolution turbulent wake flow fields in particle image velocimetry (PIV) experiments remains a significant challenge due to hardware limitations and measurement noise. In contrast, temporal high-frequency measurements of spatially sparse wall pressure are more readily accessible in wind tunnel experiments. In this study, we propose a novel cross-modal temporal upscaling framework, LatentFlow, which reconstructs high-frequency (512 Hz) turbulent wake flow fields by fusing synchronized low-frequency (15 Hz) flow field and pressure data during training, and high-frequency wall pressure signals during inference. The first stage involves training a pressure-conditioned $\beta$-variation autoencoder ($p$C-$\beta$-VAE) to learn a compact latent representation that captures the intrinsic dynamics of the wake flow. A secondary network maps synchronized low-frequency wall pressure signals into the latent space, enabling reconstruction of the wake flow field solely from sparse wall pressure. Once trained, the model utilizes high-frequency, spatially sparse wall pressure inputs to generate corresponding high-frequency flow fields via the $p$C-$\beta$-VAE decoder. By decoupling the spatial encoding of flow dynamics from temporal pressure measurements, LatentFlow provides a scalable and robust solution for reconstructing high-frequency turbulent wake flows in data-constrained experimental settings. 

**Abstract (ZH)**: 基于交叉模态时间上尺度的LatentFlow：在数据受限实验条件下重构高频湍涡尾流场 

---
# Equinox: Holistic Fair Scheduling in Serving Large Language Models 

**Title (ZH)**: Equinox: 全局公平调度在服务大规模语言模型中的应用 

**Authors**: Zhixiang Wei, James Yen, Jingyi Chen, Ziyang Zhang, Zhibai Huang, Chen Chen, Xingzi Yu, Yicheng Gu, Chenggang Wu, Yun Wang, Mingyuan Xia, Jie Wu, Hao Wang, Zhengwei Qi  

**Link**: [PDF](https://arxiv.org/pdf/2508.16646)  

**Abstract**: We address the limitations of current LLM serving with a dual-counter framework separating user and operator perspectives. The User Fairness Counter measures quality of service via weighted tokens and latency; the Resource Fairness Counter measures operational efficiency through throughput and GPU utilization. Since these metrics are only available post-execution, creating a scheduling paradox, we introduce a deterministic Mixture of Prediction Experts (MoPE) framework to predict user-perceived latency, output tokens, throughput, and GPU utilization. These predictions enable calculation of a unified Holistic Fairness score that balances both counters through tunable parameters for proactive fairness-aware scheduling. We implement this in Equinox, an open-source system with other optimizations like adaptive batching, and stall-free scheduling. Evaluations on production traces (ShareGPT, LMSYS) and synthetic workloads demonstrate Equinox achieves up to $1.3\times$ higher throughput, 60\% lower time-to-first-token latency, and 13\% higher fairness versus VTC while maintaining 94\% GPU utilization, proving fairness under bounded discrepancy across heterogeneous platforms. 

**Abstract (ZH)**: 基于用户和操作员视角的双计数器框架改进当前LLM服务的局限性：确定性预测专家混合模型实现全面公平调度 

---
# From Classical Probabilistic Latent Variable Models to Modern Generative AI: A Unified Perspective 

**Title (ZH)**: 从经典概率潜在变量模型到现代生成AI：一种统一视角 

**Authors**: Tianhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.16643)  

**Abstract**: From large language models to multi-modal agents, Generative Artificial Intelligence (AI) now underpins state-of-the-art systems. Despite their varied architectures, many share a common foundation in probabilistic latent variable models (PLVMs), where hidden variables explain observed data for density estimation, latent reasoning, and structured inference. This paper presents a unified perspective by framing both classical and modern generative methods within the PLVM paradigm. We trace the progression from classical flat models such as probabilistic PCA, Gaussian mixture models, latent class analysis, item response theory, and latent Dirichlet allocation, through their sequential extensions including Hidden Markov Models, Gaussian HMMs, and Linear Dynamical Systems, to contemporary deep architectures: Variational Autoencoders as Deep PLVMs, Normalizing Flows as Tractable PLVMs, Diffusion Models as Sequential PLVMs, Autoregressive Models as Explicit Generative Models, and Generative Adversarial Networks as Implicit PLVMs. Viewing these architectures under a common probabilistic taxonomy reveals shared principles, distinct inference strategies, and the representational trade-offs that shape their strengths. We offer a conceptual roadmap that consolidates generative AI's theoretical foundations, clarifies methodological lineages, and guides future innovation by grounding emerging architectures in their probabilistic heritage. 

**Abstract (ZH)**: 从大型语言模型到多模态代理，生成型人工智能现在支撑着最先进的系统。尽管它们的架构各异，许多模型都基于概率潜变量模型（PLVMs）这一共同基础，其中潜变量解释观测数据，用于密度估计、潜在推理和结构化推断。本文通过将经典和现代生成方法都置于PLVM范式之中，提供了一个统一的观点。我们追溯了从经典平坦模型（如概率主成分分析、高斯混合模型、潜在类别分析、项目反应理论和潜在狄利克雷分配）及其顺序扩展（如隐藏马尔可夫模型、高斯HMM和线性动态系统），到当代深度架构（如变分自编码器作为深度PLVM、规范化流作为可计算的PLVM、扩散模型作为顺序PLVM、自回归模型作为显式的生成模型、生成对抗网络作为隐式的PLVM）的进步。从共同的概率分类视角来看，这些架构揭示了共享的原则、不同的推理策略以及塑造其优势的表示权衡。我们提供了一个概念性的路线图，巩固生成型人工智能的理论基础，明确方法论的演变脉络，并通过将新兴架构植根于其概率遗产来指导未来的创新。 

---
# Cognitive Decision Routing in Large Language Models: When to Think Fast, When to Think Slow 

**Title (ZH)**: 大型语言模型中的认知决策路由：何时快速思考，何时深度思考 

**Authors**: Y. Du, C. Guo, W. Wang, G. Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16636)  

**Abstract**: Large Language Models (LLMs) face a fundamental challenge in deciding when to rely on rapid, intuitive responses versus engaging in slower, more deliberate reasoning. Inspired by Daniel Kahneman's dual-process theory and his insights on human cognitive biases, we propose a novel Cognitive Decision Routing (CDR) framework that dynamically determines the appropriate reasoning strategy based on query characteristics. Our approach addresses the current limitations where models either apply uniform reasoning depth or rely on computationally expensive methods for all queries. We introduce a meta-cognitive layer that analyzes query complexity through multiple dimensions: correlation strength between given information and required conclusions, domain boundary crossings, stakeholder multiplicity, and uncertainty levels. Through extensive experiments on diverse reasoning tasks, we demonstrate that CDR achieves superior performance while reducing computational costs by 34\% compared to uniform deep reasoning approaches. Our framework shows particular strength in professional judgment tasks, achieving 23\% improvement in consistency and 18\% better accuracy on expert-level evaluations. This work bridges cognitive science principles with practical AI system design, offering a principled approach to adaptive reasoning in LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在决定何时依赖快速直观的响应与何时进行更慢但更加谨慎的推理时面临一个根本性的挑战。受到丹尼尔·卡内曼的双过程理论及其对人类认知偏差洞见的启发，我们提出了一种新颖的认知决策路由（CDR）框架，该框架能够动态地根据查询特性确定合适的推理策略。我们的方法解决了当前模型要么采用统一的推理深度，要么依赖于对所有查询都使用计算成本高昂的方法的问题。我们引入了一层元认知层，通过多个维度分析查询的复杂性：给定信息与所需结论的相关强度、领域边界跨越、利益相关方的多样性以及不确定性水平。通过在多种推理任务上的广泛实验，我们证明了CDR在提高性能的同时，计算成本降低了34%。我们的框架在专业判断任务中表现尤为出色，在专家级评估中的一致性提高23%，准确率提高18%。这项工作将认知科学原理与实际的人工智能系统设计相结合，提供了一种适应性推理在LLMs中的原则性方法。 

---
# Few-shot Class-incremental Fault Diagnosis by Preserving Class-Agnostic Knowledge with Dual-Granularity Representations 

**Title (ZH)**: 基于双粒度表示保留无类别依赖知识的少量样本类别增量故障诊断 

**Authors**: Zhendong Yang, Jie Wang, Liansong Zong, Xiaorong Liu, Quan Qian, Shiqian Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.16634)  

**Abstract**: Few-Shot Class-Incremental Fault Diagnosis (FSC-FD), which aims to continuously learn from new fault classes with only a few samples without forgetting old ones, is critical for real-world industrial systems. However, this challenging task severely amplifies the issues of catastrophic forgetting of old knowledge and overfitting on scarce new data. To address these challenges, this paper proposes a novel framework built upon Dual-Granularity Representations, termed the Dual-Granularity Guidance Network (DGGN). Our DGGN explicitly decouples feature learning into two parallel streams: 1) a fine-grained representation stream, which utilizes a novel Multi-Order Interaction Aggregation module to capture discriminative, class-specific features from the limited new samples. 2) a coarse-grained representation stream, designed to model and preserve general, class-agnostic knowledge shared across all fault types. These two representations are dynamically fused by a multi-semantic cross-attention mechanism, where the stable coarse-grained knowledge guides the learning of fine-grained features, preventing overfitting and alleviating feature conflicts. To further mitigate catastrophic forgetting, we design a Boundary-Aware Exemplar Prioritization strategy. Moreover, a decoupled Balanced Random Forest classifier is employed to counter the decision boundary bias caused by data imbalance. Extensive experiments on the TEP benchmark and a real-world MFF dataset demonstrate that our proposed DGGN achieves superior diagnostic performance and stability compared to state-of-the-art FSC-FD approaches. Our code is publicly available at this https URL 

**Abstract (ZH)**: Few-Shot 类内增量故障诊断 (FSC-FD)：基于双粒度表示的双粒度引导网络 

---
# Adaptive Variance-Penalized Continual Learning with Fisher Regularization 

**Title (ZH)**: 适应性方差惩罚的持续学习与费希尔正则化 

**Authors**: Krisanu Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2508.16632)  

**Abstract**: The persistent challenge of catastrophic forgetting in neural networks has motivated extensive research in continual learning . This work presents a novel continual learning framework that integrates Fisher-weighted asymmetric regularization of parameter variances within a variational learning paradigm. Our method dynamically modulates regularization intensity according to parameter uncertainty, achieving enhanced stability and performance. Comprehensive evaluations on standard continual learning benchmarks including SplitMNIST, PermutedMNIST, and SplitFashionMNIST demonstrate substantial improvements over existing approaches such as Variational Continual Learning and Elastic Weight Consolidation . The asymmetric variance penalty mechanism proves particularly effective in maintaining knowledge across sequential tasks while improving model accuracy. Experimental results show our approach not only boosts immediate task performance but also significantly mitigates knowledge degradation over time, effectively addressing the fundamental challenge of catastrophic forgetting in neural networks 

**Abstract (ZH)**: 神经网络中灾难性遗忘的持久挑战推动了持续学习研究的广泛开展。本工作提出了一种新的持续学习框架，该框架在变分学习范式中集成 Fisher 权重加权非对称正则化参数方差，动态调节正则化强度以参数不确定性为基础，实现增强的稳定性和性能。在包括 SplitMNIST、PermutedMNIST 和 SplitFashionMNIST 的标准持续学习基准上的全面评估显示，本方法在与现有方法如变分持续学习和弹性权重巩固相比时表现出显著的改进。非对称方差惩罚机制特别有效，能够在顺序任务中维持知识并提高模型准确性。实验结果表明，本方法不仅提升了当前任务的性能，还显著减少了随时间推移的知识退化，有效解决了神经网络中灾难性遗忘的基本挑战。 

---
# Learn to Memorize: Optimizing LLM-based Agents with Adaptive Memory Framework 

**Title (ZH)**: 基于自适应记忆框架优化记忆能力的LLM代理 

**Authors**: Zeyu Zhang, Quanyu Dai, Rui Li, Xiaohe Bo, Xu Chen, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2508.16629)  

**Abstract**: LLM-based agents have been extensively applied across various domains, where memory stands out as one of their most essential capabilities. Previous memory mechanisms of LLM-based agents are manually predefined by human experts, leading to higher labor costs and suboptimal performance. In addition, these methods overlook the memory cycle effect in interactive scenarios, which is critical to optimizing LLM-based agents for specific environments. To address these challenges, in this paper, we propose to optimize LLM-based agents with an adaptive and data-driven memory framework by modeling memory cycles. Specifically, we design an MoE gate function to facilitate memory retrieval, propose a learnable aggregation process to improve memory utilization, and develop task-specific reflection to adapt memory storage. Our memory framework empowers LLM-based agents to learn how to memorize information effectively in specific environments, with both off-policy and on-policy optimization. In order to evaluate the effectiveness of our proposed methods, we conduct comprehensive experiments across multiple aspects. To benefit the research community in this area, we release our project at this https URL. 

**Abstract (ZH)**: 基于LLM的智能体已经在多个领域得到了广泛应用，其中记忆是其最为重要的能力之一。现有的基于LLM的智能体的记忆机制大多由人工专家手动预定义，导致了较高的劳动成本和次优性能。此外，这些方法忽略了互动场景中记忆周期效应的关键作用，这对于针对特定环境优化基于LLM的智能体至关重要。为解决这些问题，本文提出了一种通过建模记忆周期来实现自适应和数据驱动的记忆框架，以优化基于LLM的智能体。具体而言，我们设计了一个MoE门控函数以促进记忆检索，提出了一个可学习的聚合过程以提高记忆利用率，并开发了任务特定的反馈以适应记忆存储。我们的记忆框架使基于LLM的智能体能够在特定环境中学习如何有效地记忆信息，并实现了策略离线优化和在线优化。为了评估我们所提出方法的有效性，我们在多个方面进行了全面的实验。为促进该领域的研究，我们在https://this.url/releasesour项目。 

---
# The Impact of Artificial Intelligence on Human Thought 

**Title (ZH)**: 人工智能对人类思维的影响 

**Authors**: Rénald Gesnot  

**Link**: [PDF](https://arxiv.org/pdf/2508.16628)  

**Abstract**: This research paper examines, from a multidimensional perspective (cognitive, social, ethical, and philosophical), how AI is transforming human thought. It highlights a cognitive offloading effect: the externalization of mental functions to AI can reduce intellectual engagement and weaken critical thinking. On the social level, algorithmic personalization creates filter bubbles that limit the diversity of opinions and can lead to the homogenization of thought and polarization. This research also describes the mechanisms of algorithmic manipulation (exploitation of cognitive biases, automated disinformation, etc.) that amplify AI's power of influence. Finally, the question of potential artificial consciousness is discussed, along with its ethical implications. The report as a whole underscores the risks that AI poses to human intellectual autonomy and creativity, while proposing avenues (education, transparency, governance) to align AI development with the interests of humanity. 

**Abstract (ZH)**: 这篇研究论文从认知、社会、伦理和哲学多维度探讨了AI如何变革人类思维。它突出了认知卸载效应：将心理功能外移到AI可以减少智力参与并削弱批判性思维。在社会层面，算法个性化创造出信息茧房，限制了意见的多样性，并可能导致思想同质化和极化。该研究还描述了算法操控机制（利用认知偏差、自动化假信息等），这些机制放大了AI的影响能力。最后，讨论了潜在的人工意识问题及其伦理影响。整个报告强调了AI对人类智力自主性和创造力的潜在风险，并提出教育、透明度和治理等途径，以确保AI发展符合人类利益。 

---
# Data and Context Matter: Towards Generalizing AI-based Software Vulnerability Detection 

**Title (ZH)**: 数据和上下文至关重要：面向基于AI的软件漏洞检测的泛化研究 

**Authors**: Rijha Safdar, Danyail Mateen, Syed Taha Ali, M. Umer Ashfaq, Wajahat Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2508.16625)  

**Abstract**: The performance of AI-based software vulnerability detection systems is often limited by their poor generalization to unknown codebases. In this research, we explore the impact of data quality and model architecture on the generalizability of vulnerability detection systems. By generalization we mean ability of high vulnerability detection performance across different C/C++ software projects not seen during training. Through a series of experiments, we demonstrate that improvements in dataset diversity and quality substantially enhance detection performance. Additionally, we compare multiple encoder-only and decoder-only models, finding that encoder based models outperform in terms of accuracy and generalization. Our model achieves 6.8% improvement in recall on the benchmark BigVul[1] dataset, also outperforming on unseen projects, hence showing enhanced generalizability. These results highlight the role of data quality and model selection in the development of robust vulnerability detection systems. Our findings suggest a direction for future systems having high cross-project effectiveness. 

**Abstract (ZH)**: 基于AI的软件漏洞检测系统在未知代码库上的性能往往受限于其较差的泛化能力。本研究探讨了数据质量和模型架构对漏洞检测系统泛化能力的影响。通过一系列实验，我们证明了数据集多样性和质量的提升显著提高了检测性能。此外，我们比较了多种编码器-only和解码器-only模型，发现基于编码器的模型在准确性与泛化能力上表现更优。我们的模型在基准数据集BigVul上召回率提高了6.8%，并且在未见过的项目上也表现出色，从而展示了增强的泛化能力。这些结果突显了数据质量和模型选择在开发稳健的漏洞检测系统中的作用。我们的发现指出了未来系統高跨项目效应的一个发展方向。 

---
# The GPT-4o Shock Emotional Attachment to AI Models and Its Impact on Regulatory Acceptance: A Cross-Cultural Analysis of the Immediate Transition from GPT-4o to GPT-5 

**Title (ZH)**: GPT-4o 情感黏着于AI模型及其对监管接受度的影响：从GPT-4o到GPT-5的即时过渡的跨文化分析 

**Authors**: Hiroki Naito  

**Link**: [PDF](https://arxiv.org/pdf/2508.16624)  

**Abstract**: In August 2025, a major AI company's immediate, mandatory transition from its previous to its next-generation model triggered widespread public reactions. I collected 150 posts in Japanese and English from multiple social media platforms and video-sharing services between August 8-9, 2025, and qualitatively analyzed expressions of emotional attachment and resistance. Users often described GPT-4o as a trusted partner or AI boyfriend, suggesting person-like bonds. Japanese posts were dominated by loss-oriented narratives, whereas English posts included more anger, meta-level critique, and memes.A preliminary quantitative check showed a statistically significant difference in attachment coding between Japanese and English posts, with substantially higher attachment observed in the Japanese data. The findings suggest that for attachment-heavy models, even safety-oriented changes can face rapid, large-scale resistance that narrows the practical window for behavioral control. If future AI robots capable of inducing emotional bonds become widespread in the physical world, such attachment could surpass the ability to enforce regulation at an even earlier stage than in digital settings. Policy options include gradual transitions, parallel availability, and proactive measurement of attachment thresholds and points of no return to prevent emotional dynamics from outpacing effective governance. 

**Abstract (ZH)**: 2025年8月，一家主要AI公司在其从上一代模型立即、强制过渡到下一代模型时引发了广泛的社会反应。我收集了2025年8月8日至9日多个社交平台和视频分享服务上的150条日文和英文帖子，并对其进行了定性分析，探讨了情感依附和抵抗的表达。用户常将GPT-4o描述为可信赖的伙伴或AI男友，暗示人类似的关系。日文帖子主要集中在悲伤叙事上，而英文帖子则包含更多愤怒、元层面的批评和梗图。初步的定量检查显示，日文和英文帖子在依附编码上的差异具有统计学意义，日文数据中的依附程度显著更高。研究结果表明，对于依附导向的模型，即使是安全导向的更改也可能迅速引发大规模的抵抗，从而缩短行为控制的可行窗口。如果未来具备诱发情感联系的AI机器人在现实世界中普及，这种依附可能比在数字设置中更早地超越监管能力。政策选项包括逐步过渡、并行可用性及主动测量依附阈值和临界点，以防止情感动态超越有效的治理。 

---
# A Retrieval Augmented Spatio-Temporal Framework for Traffic Prediction 

**Title (ZH)**: 基于检索增强的空间时间框架的交通预测 

**Authors**: Weilin Ruan, Xilin Dang, Ziyu Zhou, Sisuo Lyu, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16623)  

**Abstract**: Traffic prediction is a cornerstone of modern intelligent transportation systems and a critical task in spatio-temporal forecasting. Although advanced Spatio-temporal Graph Neural Networks (STGNNs) and pre-trained models have achieved significant progress in traffic prediction, two key challenges remain: (i) limited contextual capacity when modeling complex spatio-temporal dependencies, and (ii) low predictability at fine-grained spatio-temporal points due to heterogeneous patterns. Inspired by Retrieval-Augmented Generation (RAG), we propose RAST, a universal framework that integrates retrieval-augmented mechanisms with spatio-temporal modeling to address these challenges. Our framework consists of three key designs: 1) Decoupled Encoder and Query Generator to capture decoupled spatial and temporal features and construct a fusion query via residual fusion; 2) Spatio-temporal Retrieval Store and Retrievers to maintain and retrieve vectorized fine-grained patterns; and 3) Universal Backbone Predictor that flexibly accommodates pre-trained STGNNs or simple MLP predictors. Extensive experiments on six real-world traffic networks, including large-scale datasets, demonstrate that RAST achieves superior performance while maintaining computational efficiency. 

**Abstract (ZH)**: 基于检索增强的时空交通预测框架（RAST） 

---
# STRelay: A Universal Spatio-Temporal Relaying Framework for Location Prediction with Future Spatiotemporal Contexts 

**Title (ZH)**: STRelay: 一种考虑未来时空上下文的通用时空 relay 预测框架 

**Authors**: Bangchao Deng, Lianhua Ji, Chunhua Chen, Xin Jing, Ling Ding, Bingqing QU, Pengyang Wang, Dingqi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16620)  

**Abstract**: Next location prediction is a critical task in human mobility modeling, enabling applications like travel planning and urban mobility management. Existing methods mainly rely on historical spatiotemporal trajectory data to train sequence models that directly forecast future locations. However, they often overlook the importance of the future spatiotemporal contexts, which are highly informative for the future locations. For example, knowing how much time and distance a user will travel could serve as a critical clue for predicting the user's next location. Against this background, we propose \textbf{STRelay}, a universal \textbf{\underline{S}}patio\textbf{\underline{T}}emporal \textbf{\underline{Relay}}ing framework explicitly modeling the future spatiotemporal context given a human trajectory, to boost the performance of different location prediction models. Specifically, STRelay models future spatiotemporal contexts in a relaying manner, which is subsequently integrated with the encoded historical representation from a base location prediction model, enabling multi-task learning by simultaneously predicting the next time interval, next moving distance interval, and finally the next location. We evaluate STRelay integrated with four state-of-the-art location prediction base models on four real-world trajectory datasets. Results demonstrate that STRelay consistently improves prediction performance across all cases by 3.19\%-11.56\%. Additionally, we find that the future spatiotemporal contexts are particularly helpful for entertainment-related locations and also for user groups who prefer traveling longer distances. The performance gain on such non-daily-routine activities, which often suffer from higher uncertainty, is indeed complementary to the base location prediction models that often excel at modeling regular daily routine patterns. 

**Abstract (ZH)**: 下一步位置预测是人类移动建模中的关键任务，能够支持旅行规划和城市移动管理等应用。现有方法主要依赖历史时空轨迹数据训练序列模型直接预测未来位置，但通常忽视了对未来时空上下文的重要性，而这些上下文对于预测未来位置非常有用。例如，知道用户将要花费多少时间和距离移动，可以作为预测用户下一步位置的关键线索。基于此，我们提出了一种新的时空接力框架STRelay，该框架明确建模给定人类轨迹的未来时空上下文，以提升不同位置预测模型的性能。具体而言，STRelay通过接力的方式建模未来时空上下文，并将其与基础位置预测模型的编码历史表示融合，从而通过同时预测下一时段、下一步移动距离和最终位置来实现多任务学习。我们将STRelay分别与四种最新位置预测基础模型在四种真实世界轨迹数据集上进行评估。结果显示，STRelay在所有情况下均能一致地提高预测性能，增幅为3.19%-11.56%。此外，我们发现未来时空上下文特别有助于娱乐相关位置的预测，也对偏好长途旅行的用户群体有益。对于这类非日常活动，由于其不确定性较高，性能提升确实补充了基础位置预测模型在建模日常规律模式方面的优势。 

---
# Negative Shanshui: Real-time Interactive Ink Painting Synthesis 

**Title (ZH)**: 负山水：实时交互式水墨画合成 

**Authors**: Aven-Le Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.16612)  

**Abstract**: This paper presents Negative Shanshui, a real-time interactive AI synthesis approach that reinterprets classical Chinese landscape ink painting, i.e., shanshui, to engage with ecological crises in the Anthropocene. Negative Shanshui optimizes a fine-tuned Stable Diffusion model for real-time inferences and integrates it with gaze-driven inpainting, frame interpolation; it enables dynamic morphing animations in response to the viewer's gaze and presents as an interactive virtual reality (VR) experience. The paper describes the complete technical pipeline, covering the system framework, optimization strategies, gaze-based interaction, and multimodal deployment in an art festival. Further analysis of audience feedback collected during its public exhibition highlights how participants variously engaged with the work through empathy, ambivalence, and critical reflection. 

**Abstract (ZH)**: 这篇论文介绍了 Negative Shanshui，这是一种实时互动的AI合成方法，重新解读了传统的中国山水墨画，即山水画，以应对人类世的生态危机。Negative Shanshui 对微调后的 Stable Diffusion 模型进行优化以实现实时推理，并将其与注视驱动的修复、帧插值相结合，能够根据观众的注视生成动态形态动画，并作为互动虚拟现实（VR）体验呈现。论文描述了完整的技术流程，涵盖了系统框架、优化策略、基于注视的交互以及在艺术节中的多模态部署。进一步分析其公共展览期间收集的观众反馈表明，参与者通过共情、矛盾和批判性反思等各种方式与作品进行了互动。 

---
# To Explain Or Not To Explain: An Empirical Investigation Of AI-Based Recommendations On Social Media Platforms 

**Title (ZH)**: 是否需要解释：基于AI的推荐在社交媒体平台上的实证研究 

**Authors**: AKM Bahalul Haque, A.K.M. Najmul Islam, Patrick Mikalef  

**Link**: [PDF](https://arxiv.org/pdf/2508.16610)  

**Abstract**: AI based social media recommendations have great potential to improve the user experience. However, often these recommendations do not match the user interest and create an unpleasant experience for the users. Moreover, the recommendation system being a black box creates comprehensibility and transparency issues. This paper investigates social media recommendations from an end user perspective. For the investigation, we used the popular social media platform Facebook and recruited regular users to conduct a qualitative analysis. We asked participants about the social media content suggestions, their comprehensibility, and explainability. Our analysis shows users mostly require explanation whenever they encounter unfamiliar content and to ensure their online data security. Furthermore, the users require concise, non-technical explanations along with the facility of controlled information flow. In addition, we observed that explanations impact the users perception of transparency, trust, and understandability. Finally, we have outlined some design implications and presented a synthesized framework based on our data analysis. 

**Abstract (ZH)**: 基于AI的社会媒体推荐具有大幅提升用户体验的潜力。然而，这些推荐往往不能匹配用户兴趣，给用户带来不愉快的体验。此外，由于推荐系统是一个黑箱，这造成了可解释性和透明度的问题。本文从最终用户的角度探讨社会媒体推荐。我们使用流行的社交媒体平台Facebook并招募常规用户进行定性分析，询问参与者关于社交媒体内容建议、可解释性和可理解性的看法。我们的分析显示，用户在遇到不熟悉的内容时大多需要解释，并且需要确保在线数据安全。此外，用户需要简洁、非技术性的解释，并希望有控制信息流动的便利。最后，我们提出了设计建议，并基于数据分析提出了一套综合框架。 

---
# Social Identity in Human-Agent Interaction: A Primer 

**Title (ZH)**: 人类与智能体互动中的社会身份：入门指南 

**Authors**: Katie Seaborn  

**Link**: [PDF](https://arxiv.org/pdf/2508.16609)  

**Abstract**: Social identity theory (SIT) and social categorization theory (SCT) are two facets of the social identity approach (SIA) to understanding social phenomena. SIT and SCT are models that describe and explain how people interact with one another socially, connecting the individual to the group through an understanding of underlying psychological mechanisms and intergroup behaviour. SIT, originally developed in the 1970s, and SCT, a later, more general offshoot, have been broadly applied to a range of social phenomena among people. The rise of increasingly social machines embedded in daily life has spurned efforts on understanding whether and how artificial agents can and do participate in SIA activities. As agents like social robots and chatbots powered by sophisticated large language models (LLMs) advance, understanding the real and potential roles of these technologies as social entities is crucial. Here, I provide a primer on SIA and extrapolate, through case studies and imagined examples, how SIT and SCT can apply to artificial social agents. I emphasize that not all human models and sub-theories will apply. I further argue that, given the emerging competence of these machines and our tendency to be taken in by them, we experts may need to don the hat of the uncanny killjoy, for our own good. 

**Abstract (ZH)**: 社会身份理论（SIT）和社会分类理论（SCT）是社会身份方法（SIA）的两个方面，用于理解社会现象。社会身份理论和社会分类理论是描述和解释人们如何通过理解基本的心理机制和群体间行为进行社会互动的模型。社会身份理论（SIT）最初在20世纪70年代发展，而社会分类理论（SCT）则是后来的一个更为广泛的应用分支，两者都被广泛应用于人类社会的各种现象中。随着嵌入日常生活中的社会机器越来越多，我们开始努力理解人工代理是否以及如何参与社会身份方法的活动。随着像社会机器人和由复杂大型语言模型（LLMs）驱动的聊天机器人的发展，理解这些技术作为社会实体的真实和潜在角色至关重要。在这里，我对社会身份方法提供了一个概述，并通过案例研究和假设的示例，探讨社会身份理论（SIT）和社会分类理论（SCT）如何适用于人工社会代理。我强调，并非所有的人类模型和次理论都适用。进一步地，鉴于这些机器日益增长的能力以及我们倾向于被它们所迷惑的倾向，我们这些专家可能需要戴上怪异的 killjoy 的帽子，这对我们自己是有益的。 

---
# "Accessibility people, you go work on that thing of yours over there": Addressing Disability Inclusion in AI Product Organizations 

**Title (ZH)**: “让残疾人能够使用，你们就去处理那边的那个事情吧”：在AI产品组织中推动残疾人包容性 

**Authors**: Sanika Moharana, Cynthia L. Bennett, Erin Buehler, Michael Madaio, Vinita Tibdewal, Shaun K. Kane  

**Link**: [PDF](https://arxiv.org/pdf/2508.16607)  

**Abstract**: The rapid emergence of generative AI has changed the way that technology is designed, constructed, maintained, and evaluated. Decisions made when creating AI-powered systems may impact some users disproportionately, such as people with disabilities. In this paper, we report on an interview study with 25 AI practitioners across multiple roles (engineering, research, UX, and responsible AI) about how their work processes and artifacts may impact end users with disabilities. We found that practitioners experienced friction when triaging problems at the intersection of responsible AI and accessibility practices, navigated contradictions between accessibility and responsible AI guidelines, identified gaps in data about users with disabilities, and gathered support for addressing the needs of disabled stakeholders by leveraging informal volunteer and community groups within their company. Based on these findings, we offer suggestions for new resources and process changes to better support people with disabilities as end users of AI. 

**Abstract (ZH)**: 生成式人工智能的快速兴起改变了技术的设计、构建、维护和评估方式。在创造人工智能驱动系统时做出的决策可能不对等地影响某些用户，例如残疾人。本文报道了对跨多个角色（工程、研究、用户体验和负责任的人工智能）的25名人工智能从业者进行的访谈研究，探讨他们的工作流程和产出如何影响残疾用户。研究发现，从业者在负责任的人工智能和可达性实践交叉问题的优先排序过程中遇到了摩擦，导航了可达性与负责任的人工智能指南之间的矛盾，识别了关于残疾用户的数据缺口，并通过在其公司内利用非正式的志愿者和社区小组来争取支持，以解决残疾相关方的需求。基于这些发现，我们提出了新的资源和流程改进建议，以更好地支持残疾人为人工智能的最终用户。 

---
# Multimodal Appearance based Gaze-Controlled Virtual Keyboard with Synchronous Asynchronous Interaction for Low-Resource Settings 

**Title (ZH)**: 基于多模态外观的眼动控制虚拟键盘及其在低资源环境下的同步异步交互方法 

**Authors**: Yogesh Kumar Meena, Manish Salvi  

**Link**: [PDF](https://arxiv.org/pdf/2508.16606)  

**Abstract**: Over the past decade, the demand for communication devices has increased among individuals with mobility and speech impairments. Eye-gaze tracking has emerged as a promising solution for hands-free communication; however, traditional appearance-based interfaces often face challenges such as accuracy issues, involuntary eye movements, and difficulties with extensive command sets. This work presents a multimodal appearance-based gaze-controlled virtual keyboard that utilises deep learning in conjunction with standard camera hardware, incorporating both synchronous and asynchronous modes for command selection. The virtual keyboard application supports menu-based selection with nine commands, enabling users to spell and type up to 56 English characters, including uppercase and lowercase letters, punctuation, and a delete function for corrections. The proposed system was evaluated with twenty able-bodied participants who completed specially designed typing tasks using three input modalities: (i) a mouse, (ii) an eye-tracker, and (iii) an unmodified webcam. Typing performance was measured in terms of speed and information transfer rate (ITR) at both command and letter levels. Average typing speeds were 18.3+-5.31 letters/min (mouse), 12.60+-2.99letters/min (eye-tracker, synchronous), 10.94 +- 1.89 letters/min (webcam, synchronous), 11.15 +- 2.90 letters/min (eye-tracker, asynchronous), and 7.86 +- 1.69 letters/min (webcam, asynchronous). ITRs were approximately 80.29 +- 15.72 bits/min (command level) and 63.56 +- 11 bits/min (letter level) with webcam in synchronous mode. The system demonstrated good usability and low workload with webcam input, highlighting its user-centred design and promise as an accessible communication tool in low-resource settings. 

**Abstract (ZH)**: 过去十年间，移动和言语障碍个体对通信设备的需求不断增加。目光追踪技术已成为无需手部操作的通信 promising 解决方案；然而，传统的基于外观的界面往往面临准确性问题、不自主的眼球运动以及广泛的命令集难以处理的挑战。本文提出了一种结合深度学习和标准摄像头硬件的多模态基于外观的目光控制虚拟键盘，该系统包括同步和异步模式以进行命令选择。虚拟键盘应用程序通过菜单选择支持多达九个命令，使用户能够拼写和输入56个英文字符，包括大小写字母、标点符号以及删除功能以进行更正。所提出的系统在二十名健全参与者中进行了评估，他们使用三种输入模式完成了专门设计的打字任务：（i）鼠标，（ii）眼动追踪器，（iii）未修改的网络摄像头。从命令和字母层面测量了打字性能，包括速度和信息传输率（ITR）。平均打字速度分别为鼠标18.3±5.31个字母/分钟，眼动追踪器同步模式12.60±2.99个字母/分钟，网络摄像头同步模式10.94±1.89个字母/分钟，眼动追踪器异步模式11.15±2.90个字母/分钟，以及网络摄像头异步模式7.86±1.69个字母/分钟。同步模式下网络摄像头的ITR分别为命令级别约80.29±15.72比特/分钟、字母级别约63.56±11比特/分钟。该系统在使用网络摄像头输入时展示了良好的易用性和较低的工作负荷，突显了其以用户为中心的设计，并使其成为低资源环境下易于访问的通信工具的潜力。 

---
# GreenTEA: Gradient Descent with Topic-modeling and Evolutionary Auto-prompting 

**Title (ZH)**: GreenTEA：基于主题建模和演化自动生成提示的梯度下降方法 

**Authors**: Zheng Dong, Luming Shang, Gabriela Olinto  

**Link**: [PDF](https://arxiv.org/pdf/2508.16603)  

**Abstract**: High-quality prompts are crucial for Large Language Models (LLMs) to achieve exceptional performance. However, manually crafting effective prompts is labor-intensive and demands significant domain expertise, limiting its scalability. Existing automatic prompt optimization methods either extensively explore new prompt candidates, incurring high computational costs due to inefficient searches within a large solution space, or overly exploit feedback on existing prompts, risking suboptimal optimization because of the complex prompt landscape. To address these challenges, we introduce GreenTEA, an agentic LLM workflow for automatic prompt optimization that balances candidate exploration and knowledge exploitation. It leverages a collaborative team of agents to iteratively refine prompts based on feedback from error samples. An analyzing agent identifies common error patterns resulting from the current prompt via topic modeling, and a generation agent revises the prompt to directly address these key deficiencies. This refinement process is guided by a genetic algorithm framework, which simulates natural selection by evolving candidate prompts through operations such as crossover and mutation to progressively optimize model performance. Extensive numerical experiments conducted on public benchmark datasets suggest the superior performance of GreenTEA against human-engineered prompts and existing state-of-the-arts for automatic prompt optimization, covering logical and quantitative reasoning, commonsense, and ethical decision-making. 

**Abstract (ZH)**: 高质量的提示对于大型语言模型（LLMs）实现卓越的性能至关重要。然而，手动 crafting 有效的提示是劳动密集型的，并且需要显著的领域专业知识，限制了其可扩展性。现有的自动提示优化方法要么广泛探索新的提示候选，由于在大型解空间中无效搜索而导致高昂的计算成本，要么过度利用现有提示的反馈，由于提示景观的复杂性而导致次优优化。为了解决这些挑战，我们介绍了GreenTEA，一种平衡提示候选探索与知识利用的自主大型语言模型工作流。它通过协作团队的代理迭代根据错误样本的反馈来细化提示。分析代理通过主题建模识别当前提示导致的常见错误模式，并生成代理修订提示以直接解决这些关键缺陷。这一细化过程由遗传算法框架指导，通过交叉和变异等操作模拟自然选择，逐步优化模型性能。广泛的数值实验在公开基准数据集上表明，GreenTEA 在自动提示优化方面的性能优于人工设计的提示和现有最先进的方法，涵盖逻辑和定量推理、常识和伦理决策。 

---
# An Embodied AR Navigation Agent: Integrating BIM with Retrieval-Augmented Generation for Language Guidance 

**Title (ZH)**: 具身AR导航代理：结合BIM与检索增强生成的语言指导 

**Authors**: Hsuan-Kung Yang, Tsu-Ching Hsiao, Ryoichiro Oka, Ryuya Nishino, Satoko Tofukuji, Norimasa Kobori  

**Link**: [PDF](https://arxiv.org/pdf/2508.16602)  

**Abstract**: Delivering intelligent and adaptive navigation assistance in augmented reality (AR) requires more than visual cues, as it demands systems capable of interpreting flexible user intent and reasoning over both spatial and semantic context. Prior AR navigation systems often rely on rigid input schemes or predefined commands, which limit the utility of rich building data and hinder natural interaction. In this work, we propose an embodied AR navigation system that integrates Building Information Modeling (BIM) with a multi-agent retrieval-augmented generation (RAG) framework to support flexible, language-driven goal retrieval and route planning. The system orchestrates three language agents, Triage, Search, and Response, built on large language models (LLMs), which enables robust interpretation of open-ended queries and spatial reasoning using BIM data. Navigation guidance is delivered through an embodied AR agent, equipped with voice interaction and locomotion, to enhance user experience. A real-world user study yields a System Usability Scale (SUS) score of 80.5, indicating excellent usability, and comparative evaluations show that the embodied interface can significantly improves users' perception of system intelligence. These results underscore the importance and potential of language-grounded reasoning and embodiment in the design of user-centered AR navigation systems. 

**Abstract (ZH)**: 在增强现实（AR）中提供智能和适应性的导航辅助需要超越视觉提示，因为它要求系统能够解释灵活的用户意图并推理空间和语义上下文。之前的AR导航系统往往依赖于固定的输入方案或预定义的命令，这限制了丰富建筑数据的用途并妨碍了自然交互。在本工作中，我们提出了一种结合建筑信息模型（BIM）和多代理检索增强生成（RAG）框架的具身AR导航系统，该系统支持灵活的、基于语言的目标检索和路径规划。该系统协调了三个基于大规模语言模型的语言代理—Triage、Search和Response，利用BIM数据实现了对开放查询的稳健解释和空间推理。导航指导通过一个具有语音交互和移动功能的具身AR代理提供，以提升用户体验。实地用户研究获得的系统可用性量表（SUS）得分为80.5，表明了极好的可用性，对比评估表明，具身界面可以显著提高用户对系统智能性的感知。这些结果强调了语言驱动的推理和具身性在用户中心AR导航系统设计中的重要性和潜力。 

---
# Humans Perceive Wrong Narratives from AI Reasoning Texts 

**Title (ZH)**: 人类从AI推理文本中感知到错误的故事线 

**Authors**: Mosh Levy, Zohar Elyoseph, Yoav Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2508.16599)  

**Abstract**: A new generation of AI models generates step-by-step reasoning text before producing an answer. This text appears to offer a human-readable window into their computation process, and is increasingly relied upon for transparency and interpretability. However, it is unclear whether human understanding of this text matches the model's actual computational process. In this paper, we investigate a necessary condition for correspondence: the ability of humans to identify which steps in a reasoning text causally influence later steps. We evaluated humans on this ability by composing questions based on counterfactual measurements and found a significant discrepancy: participant accuracy was only 29.3%, barely above chance (25%), and remained low (42%) even when evaluating the majority vote on questions with high agreement. Our results reveal a fundamental gap between how humans interpret reasoning texts and how models use it, challenging its utility as a simple interpretability tool. We argue that reasoning texts should be treated as an artifact to be investigated, not taken at face value, and that understanding the non-human ways these models use language is a critical research direction. 

**Abstract (ZH)**: 一种新型的AI模型在生成答案之前会产生逐步推理文本。这种文本似乎为人们的计算过程提供了一个可读窗口，并越来越多地被依赖以实现透明性和可解释性。然而，尚不清楚人类对这种文本的理解是否与模型的实际计算过程相符。本文探讨了对应关系的一个必要条件：人类识别推理文本中对后续步骤有因果影响的步骤的能力。通过基于假设测量构建问题来评估这一能力，我们发现了显著的差距：参与者准确率仅为29.3%，几乎与随机概率（25%）相当，并且在高一致性的问题中，即使评估多个答案的多数投票，准确率也仅提高至42%。我们的结果揭示了人类解读推理文本与模型使用之间的基本差异，质疑其作为简单解释工具的有效性。我们认为，推理文本应被视为需要研究的产物，而非直接接受，并且理解这些模型非人类的语言使用方式是关键的研究方向。 

---
# Bridging Foundation Models and Efficient Architectures: A Modular Brain Imaging Framework with Local Masking and Pretrained Representation Learning 

**Title (ZH)**: 融合基础模型与高效架构：一种基于局部掩码与预训练表示学习的模块化脑成像框架 

**Authors**: Yanwen Wang, Xinglin Zhao, Yijin Song, Xiaobo Liu, Yanrong Hao, Rui Cao, Xin Wen  

**Link**: [PDF](https://arxiv.org/pdf/2508.16597)  

**Abstract**: Functional connectivity (FC) derived from resting-state fMRI plays a critical role in personalized predictions such as age and cognitive performance. However, applying foundation models(FM) to fMRI data remains challenging due to its high dimensionality, computational complexity, and the difficulty in capturing complex spatiotemporal dynamics and indirect region-of-interest (ROI) interactions. To address these limitations, we propose a modular neuroimaging framework that integrates principles from FM with efficient, domain-specific architectures. Our approach begins with a Local Masked Autoencoder (LMAE) for pretraining, which reduces the influence of hemodynamic response function (HRF) dynamics and suppresses noise. This is followed by a Random Walk Mixture of Experts (RWMOE) module that clusters features across spatial and temporal dimensions, effectively capturing intricate brain interactions. Finally, a state-space model (SSM)-based predictor performs downstream task inference. Evaluated on the Cambridge Centre for Ageing and Neuroscience (Cam-CAN) dataset, our framework achieved mean absolute errors (MAEs) of 5.343 for age prediction and 2.940 for fluid intelligence, with Pearson correlation coefficients (PCCs) of 0.928 and 0.887, respectively-outperforming existing state-of-the-art methods. Visualization of expert distribution weights further enhances interpretability by identifying key brain regions. This work provides a robust, interpretable alternative to LLM-based approaches for fMRI analysis, offering novel insights into brain aging and cognitive function. 

**Abstract (ZH)**: 功能性连接（FC）从静息态fMRI中提取，在年龄和个人认知性能的个性化预测中发挥关键作用。然而，将基础模型（FM）应用于fMRI数据由于其高维度、计算复杂性和复杂时空动态及间接感兴趣区（ROI）相互作用的捕捉困难而具有挑战性。为解决这些限制，我们提出了一种模块化神经成像框架，将基础模型的原则与高效、领域特定的架构相结合。该方法首先使用局部掩蔽自动编码器（LMAE）进行预训练，以减轻血流动力学反应函数（HRF）动态的影响并抑制噪声，随后使用随机游走混合专家（RWMOE）模块在空间和时间维度上聚类特征，有效地捕捉复杂的脑部相互作用。最后，基于状态空间模型（SSM）的预测器执行下游任务推理。在剑桥老化与神经科学中心（Cam-CAN）数据集上的评估表明，该框架在年龄预测中的平均绝对误差（MAE）为5.343，在流体智力预测中的MAE为2.940，分别对应的皮尔逊相关系数（PCC）为0.928和0.887，均优于现有最先进的方法。专家权重分布图的可视化进一步增强了可解释性，通过识别关键脑区。该项工作提供了一种鲁棒且可解释的替代语言模型（LLM）方法，用于fMRI分析，为脑老化和认知功能提供了新的见解。 

---
# ARL-Based Multi-Action Market Making with Hawkes Processes and Variable Volatility 

**Title (ZH)**: 基于ARL的带有赫克尔斯过程和可变波动性的多行动市场制作 

**Authors**: Ziyi Wang, Carmine Ventre, Maria Polukarov  

**Link**: [PDF](https://arxiv.org/pdf/2508.16589)  

**Abstract**: We advance market-making strategies by integrating Adversarial Reinforcement Learning (ARL), Hawkes Processes, and variable volatility levels while also expanding the action space available to market makers (MMs). To enhance the adaptability and robustness of these strategies -- which can quote always, quote only on one side of the market or not quote at all -- we shift from the commonly used Poisson process to the Hawkes process, which better captures real market dynamics and self-exciting behaviors. We then train and evaluate strategies under volatility levels of 2 and 200. Our findings show that the 4-action MM trained in a low-volatility environment effectively adapts to high-volatility conditions, maintaining stable performance and providing two-sided quotes at least 92\% of the time. This indicates that incorporating flexible quoting mechanisms and realistic market simulations significantly enhances the effectiveness of market-making strategies. 

**Abstract (ZH)**: 通过结合对手 reinforcement 学习（ARL）、霍克尔斯过程和可变波动水平，我们推进了市场制作策略，并扩展了市场制作商的行动空间。为了增强这些策略的适应性和鲁棒性——这些策略可以一直报价、仅在市场一侧报价或根本不报价——我们从常用的泊松过程转向了霍克尔斯过程，这更好地捕捉了实际市场动态和自激发行为。我们还在波动水平为 2 和 200 的环境下训练和评估了这些策略。研究发现，低波动环境下训练的四行动市场制作商能够有效适应高波动条件，保持稳定性能，并至少有 92% 的时间提供双边报价。这表明，引入灵活的报价机制和现实的市场模拟显著提高了市场制作策略的有效性。 

---
# Robust Market Making: To Quote, or not To Quote 

**Title (ZH)**: 稳健的市场制作：报价，还是不报价 

**Authors**: Ziyi Wang, Carmine Ventre, Maria Polukarov  

**Link**: [PDF](https://arxiv.org/pdf/2508.16588)  

**Abstract**: Market making is a popular trading strategy, which aims to generate profit from the spread between the quotes posted at either side of the market. It has been shown that training market makers (MMs) with adversarial reinforcement learning allows to overcome the risks due to changing market conditions and to lead to robust performances. Prior work assumes, however, that MMs keep quoting throughout the trading process, but in practice this is not required, even for ``registered'' MMs (that only need to satisfy quoting ratios defined by the market rules). In this paper, we build on this line of work and enrich the strategy space of the MM by allowing to occasionally not quote or provide single-sided quotes. Towards this end, in addition to the MM agents that provide continuous bid-ask quotes, we have designed two new agents with increasingly richer action spaces. The first has the option to provide bid-ask quotes or refuse to quote. The second has the option to provide bid-ask quotes, refuse to quote, or only provide single-sided ask or bid quotes. We employ a model-driven approach to empirically compare the performance of the continuously quoting MM with the two agents above in various types of adversarial environments. We demonstrate how occasional refusal to provide bid-ask quotes improves returns and/or Sharpe ratios. The quoting ratios of well-trained MMs can basically meet any market requirements, reaching up to 99.9$\%$ in some cases. 

**Abstract (ZH)**: 市场做市是一种流行的交易策略，旨在通过市场两侧提供的报价差来获利。已有研究表明，使用对抗强化学习训练做市商（MMs）可以应对市场变化带来的风险，并实现稳健的业绩。此前的工作假设MMs在整个交易过程中持续报价，但在实践中，即使对于“注册”MMs（只需满足市场规则定义的报价比率），这并不是必需的。在本文中，我们在此基础上扩展了MM的战略空间，允许MM偶尔不报价或提供单边报价。为此，除了提供连续双边报价的MM代理，我们还设计了两个新的代理，具有越来越丰富的行动空间。第一个代理可以选择提供双边报价或拒绝报价。第二个代理可以选择提供双边报价、拒绝报价或仅提供单边报价。我们采用基于模型的方法，比较连续报价的MM与上述两个代理在各种对抗环境中的表现。我们展示了偶尔拒绝提供双边报价如何提高回报率和/或夏普比率。经过充分训练的MM的报价比率足以满足任何市场要求，在某些情况下可达99.9%。 

---
# Predicting User Grasp Intentions in Virtual Reality 

**Title (ZH)**: 预测虚拟现实中的用户抓取意图 

**Authors**: Linghao Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2508.16582)  

**Abstract**: Predicting user intentions in virtual reality (VR) is crucial for creating immersive experiences, particularly in tasks involving complex grasping motions where accurate haptic feedback is essential. In this work, we leverage time-series data from hand movements to evaluate both classification and regression approaches across 810 trials with varied object types, sizes, and manipulations. Our findings reveal that classification models struggle to generalize across users, leading to inconsistent performance. In contrast, regression-based approaches, particularly those using Long Short Term Memory (LSTM) networks, demonstrate more robust performance, with timing errors within 0.25 seconds and distance errors around 5-20 cm in the critical two-second window before a grasp. Despite these improvements, predicting precise hand postures remains challenging. Through a comprehensive analysis of user variability and model interpretability, we explore why certain models fail and how regression models better accommodate the dynamic and complex nature of user behavior in VR. Our results underscore the potential of machine learning models to enhance VR interactions, particularly through adaptive haptic feedback, and lay the groundwork for future advancements in real-time prediction of user actions in VR. 

**Abstract (ZH)**: 预测虚拟现实（VR）中的用户意图对于创造沉浸式体验至关重要，特别是在涉及复杂抓取动作的任务中，准确的触觉反馈尤为重要。本研究利用手部运动的时间序列数据，在810次涉及不同物体类型、大小和操控方式的试验中评估了分类和回归方法。研究发现，分类模型难以泛化到不同用户身上，导致性能不一致。相比之下，基于长短期记忆（LSTM）网络的回归方法则显示出更 robust 的性能，在抓取前的两秒关键窗口内，时间误差在0.25秒以内，距离误差在5-20厘米左右。尽管有这些改进，预测精确的手部姿态仍然具有挑战性。通过全面分析用户变异性与模型可解释性，我们探索了为何某些模型会失败，以及回归模型如何更好地适应VR中用户行为的动态和复杂性。研究结果突显了机器学习模型在增强VR交互方面，特别是在适应性触觉反馈方面的潜力，并为未来在VR中实时预测用户行动的进展奠定了基础。 

---
# Adaptive Command: Real-Time Policy Adjustment via Language Models in StarCraft II 

**Title (ZH)**: 自适应命令：通过语言模型在星际争霸II中实时政策调整 

**Authors**: Weiyu Ma, Dongyu Xu, Shu Lin, Haifeng Zhang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16580)  

**Abstract**: We present Adaptive Command, a novel framework integrating large language models (LLMs) with behavior trees for real-time strategic decision-making in StarCraft II. Our system focuses on enhancing human-AI collaboration in complex, dynamic environments through natural language interactions. The framework comprises: (1) an LLM-based strategic advisor, (2) a behavior tree for action execution, and (3) a natural language interface with speech capabilities. User studies demonstrate significant improvements in player decision-making and strategic adaptability, particularly benefiting novice players and those with disabilities. This work contributes to the field of real-time human-AI collaborative decision-making, offering insights applicable beyond RTS games to various complex decision-making scenarios. 

**Abstract (ZH)**: 我们提出了一种新型框架Adaptive Command，该框架将大型语言模型（LLMs）与行为树相结合，用于StarCraft II中的实时战略决策。该系统专注于通过自然语言交互增强人在复杂动态环境中的AI协作。该框架包括：（1）基于LLM的战略顾问，（2）用于执行动作的行为树，以及（3）具有语音功能的自然语言接口。用户研究显示，在提高玩家决策能力和战略适应性方面取得了显著进步，特别有助于新手玩家和残疾人。本研究为实时人机协作决策领域做出了贡献，提供的见解不仅适用于RTS游戏，还能应用于各种复杂的决策场景。 

---
# Confidence-Modulated Speculative Decoding for Large Language Models 

**Title (ZH)**: 基于置信度调节的推测性解码对于大型语言模型 

**Authors**: Jaydip Sen, Subhasis Dasgupta, Hetvi Waghela  

**Link**: [PDF](https://arxiv.org/pdf/2508.15371)  

**Abstract**: Speculative decoding has emerged as an effective approach for accelerating autoregressive inference by parallelizing token generation through a draft-then-verify paradigm. However, existing methods rely on static drafting lengths and rigid verification criteria, limiting their adaptability across varying model uncertainties and input complexities. This paper proposes an information-theoretic framework for speculative decoding based on confidence-modulated drafting. By leveraging entropy and margin-based uncertainty measures over the drafter's output distribution, the proposed method dynamically adjusts the number of speculatively generated tokens at each iteration. This adaptive mechanism reduces rollback frequency, improves resource utilization, and maintains output fidelity. Additionally, the verification process is modulated using the same confidence signals, enabling more flexible acceptance of drafted tokens without sacrificing generation quality. Experiments on machine translation and summarization tasks demonstrate significant speedups over standard speculative decoding while preserving or improving BLEU and ROUGE scores. The proposed approach offers a principled, plug-in method for efficient and robust decoding in large language models under varying conditions of uncertainty. 

**Abstract (ZH)**: 基于置信度调制的不确定性信息论框架下的 speculative 解码 

---
