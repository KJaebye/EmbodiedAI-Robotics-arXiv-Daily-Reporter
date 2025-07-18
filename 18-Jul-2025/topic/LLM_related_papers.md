# The Generative Energy Arena (GEA): Incorporating Energy Awareness in Large Language Model (LLM) Human Evaluations 

**Title (ZH)**: 生成能量竞技场（GEA）：在大型语言模型（LLM）的人类评估中融入能量意识 

**Authors**: Carlos Arriaga, Gonzalo Martínez, Eneko Sendin, Javier Conde, Pedro Reviriego  

**Link**: [PDF](https://arxiv.org/pdf/2507.13302)  

**Abstract**: The evaluation of large language models is a complex task, in which several approaches have been proposed. The most common is the use of automated benchmarks in which LLMs have to answer multiple-choice questions of different topics. However, this method has certain limitations, being the most concerning, the poor correlation with the humans. An alternative approach, is to have humans evaluate the LLMs. This poses scalability issues as there is a large and growing number of models to evaluate making it impractical (and costly) to run traditional studies based on recruiting a number of evaluators and having them rank the responses of the models. An alternative approach is the use of public arenas, such as the popular LM arena, on which any user can freely evaluate models on any question and rank the responses of two models. The results are then elaborated into a model ranking. An increasingly important aspect of LLMs is their energy consumption and, therefore, evaluating how energy awareness influences the decisions of humans in selecting a model is of interest. In this paper, we present GEA, the Generative Energy Arena, an arena that incorporates information on the energy consumption of the model in the evaluation process. Preliminary results obtained with GEA are also presented, showing that for most questions, when users are aware of the energy consumption, they favor smaller and more energy efficient models. This suggests that for most user interactions, the extra cost and energy incurred by the more complex and top-performing models do not provide an increase in the perceived quality of the responses that justifies their use. 

**Abstract (ZH)**: 大型语言模型的评估是一个复杂任务，提出了多种方法。最常见的是使用自动基准测试，要求LLM回答不同主题的多项选择题。然而，这种方法存在某些局限性，最值得关注的是与人类表现的相关性较差。一种替代方法是让人类评估LLM，这带来了可扩展性问题，因为需要评估的模型数量庞大且不断增长，基于招募评估人员的传统研究方法在实践中变得不切实际（且成本高昂）。一种替代方法是使用公共竞技场，如流行的LM竞技场，任何用户都可以自由地在任何问题上评估模型并为其响应排名。然后将结果综合成一个模型排名。近年来，LLM的能源消耗成为一个重要方面，因此评估能源意识如何影响用户选择模型的决策是有趣的。在本文中，我们介绍了GEA（生成式能源竞技场），这是一种将模型的能源消耗信息纳入评估过程的竞技场。我们还呈现了通过GEA获得的初步结果，表明在大多数情况下，当用户了解能源消耗时，他们更倾向于选择较小且更节能的模型。这表明在大多数用户交互中，使用更复杂且性能更优的模型所增加的成本和能源，并不证明其在感知响应质量方面的提升值得使用。 

---
# Black Box Deployed -- Functional Criteria for Artificial Moral Agents in the LLM Era 

**Title (ZH)**: 黑箱部署——大语言模型时代人工道德代理的功能标准 

**Authors**: Matthew E. Brophy  

**Link**: [PDF](https://arxiv.org/pdf/2507.13175)  

**Abstract**: The advancement of powerful yet opaque large language models (LLMs) necessitates a fundamental revision of the philosophical criteria used to evaluate artificial moral agents (AMAs). Pre-LLM frameworks often relied on the assumption of transparent architectures, which LLMs defy due to their stochastic outputs and opaque internal states. This paper argues that traditional ethical criteria are pragmatically obsolete for LLMs due to this mismatch. Engaging with core themes in the philosophy of technology, this paper proffers a revised set of ten functional criteria to evaluate LLM-based artificial moral agents: moral concordance, context sensitivity, normative integrity, metaethical awareness, system resilience, trustworthiness, corrigibility, partial transparency, functional autonomy, and moral imagination. These guideposts, applied to what we term "SMA-LLS" (Simulating Moral Agency through Large Language Systems), aim to steer AMAs toward greater alignment and beneficial societal integration in the coming years. We illustrate these criteria using hypothetical scenarios involving an autonomous public bus (APB) to demonstrate their practical applicability in morally salient contexts. 

**Abstract (ZH)**: 强大的但不透明的大语言模型的进步要求从根本上修订用于评估人工道德代理的传统哲学标准。预大语言模型（LLM）的框架通常基于透明架构的假设，而LLM因其随机输出和不透明的内部状态而违背了这一假设。本文 argues 传统道德标准在LLM面前在实践上已经过时，由于这种不匹配。通过技术哲学的核心主题，本文提出了一套修订的十个功能性标准来评估基于LLM的人工道德代理：道德一致性、情境敏感性、规范完整性、元伦理意识、系统韧性、可信度、可纠正性、部分透明性、功能自主性以及道德想象力。这些指南针应用于我们称之为“SMA-LLS”（通过大语言系统模拟道德代理）的领域，旨在在未来引导人工道德代理与社会更好地结合。我们通过一个自主公交（APB）的假设情景，来说明这些标准在道德相关情境中的实际应用。 

---
# From Roots to Rewards: Dynamic Tree Reasoning with RL 

**Title (ZH)**: 从根到奖励：基于RL的动态树推理 

**Authors**: Ahmed Bahloul, Simon Malberg  

**Link**: [PDF](https://arxiv.org/pdf/2507.13142)  

**Abstract**: Modern language models address complex questions through chain-of-thought (CoT) reasoning (Wei et al., 2023) and retrieval augmentation (Lewis et al., 2021), yet struggle with error propagation and knowledge integration. Tree-structured reasoning methods, particularly the Probabilistic Tree-of-Thought (ProbTree)(Cao et al., 2023) framework, mitigate these issues by decomposing questions into hierarchical structures and selecting answers through confidence-weighted aggregation of parametric and retrieved knowledge (Yao et al., 2023). However, ProbTree's static implementation introduces two key limitations: (1) the reasoning tree is fixed during the initial construction phase, preventing dynamic adaptation to intermediate results, and (2) each node requires exhaustive evaluation of all possible solution strategies, creating computational inefficiency. We present a dynamic reinforcement learning (Sutton and Barto, 2018) framework that transforms tree-based reasoning into an adaptive process. Our approach incrementally constructs the reasoning tree based on real-time confidence estimates, while learning optimal policies for action selection (decomposition, retrieval, or aggregation). This maintains ProbTree's probabilistic rigor while improving both solution quality and computational efficiency through selective expansion and focused resource allocation. The work establishes a new paradigm for treestructured reasoning that balances the reliability of probabilistic frameworks with the flexibility required for real-world question answering systems. 

**Abstract (ZH)**: 现代语言模型通过链式推理（CoT）和检索增强方法处理复杂问题（Wei et al., 2023；Lewis et al., 2021），但面临错误传播和知识整合的挑战。基于树结构的推理方法，特别是Probabilistic Tree-of-Thought（ProbTree）框架（Cao et al., 2023），通过将问题分解为层次结构并通过对参数和检索知识的信心加权聚合选择答案，缓解了这些问题。然而，ProbTree的静态实现引入了两个关键限制：（1）推理树在初始构建阶段是固定的，无法动态适应中间结果；（2）每个节点需要对所有可能的求解策略进行耗时评估，造成计算效率低下。我们提出了一种动态强化学习（Sutton和Barto, 2018）框架，将基于树的推理转变为适应性过程。我们的方法基于实时信心估计增量构建推理树，并学习最佳策略以选择动作（分解、检索或聚合）。这既保持了ProbTree的概率严谨性，又通过选择性扩展和集中资源分配提高了解决方案质量和计算效率。该研究为平衡概率框架的可靠性和面向实际问题回答系统所需的灵活性确立了一种新的范式。 

---
# VAR-MATH: Probing True Mathematical Reasoning in Large Language Models via Symbolic Multi-Instance Benchmarks 

**Title (ZH)**: VAR-MATH：通过符号多实例基准探究大型语言模型的真正数学推理能力 

**Authors**: Jian Yao, Ran Cheng, Kay Chen Tan  

**Link**: [PDF](https://arxiv.org/pdf/2507.12885)  

**Abstract**: Recent advances in reinforcement learning (RL) have led to substantial improvements in the mathematical reasoning abilities of large language models (LLMs), as measured by standard benchmarks. However, these gains often persist even when models are trained with flawed signals, such as random or inverted rewards, raising a fundamental question: do such improvements reflect true reasoning, or are they merely artifacts of overfitting to benchmark-specific patterns? To address this question, we take an evaluation-centric perspective and identify two critical shortcomings in existing protocols. First, \emph{benchmark contamination} arises from the public availability of test problems, increasing the risk of data leakage. Second, \emph{evaluation fragility} stems from the reliance on single-instance assessments, which are highly sensitive to stochastic outputs and fail to capture reasoning consistency. To overcome these limitations, we introduce {VAR-MATH}, a symbolic evaluation framework designed to probe genuine reasoning ability. By converting fixed numerical problems into symbolic templates and requiring models to solve multiple instantiations of each, VAR-MATH enforces consistent reasoning across structurally equivalent variants, thereby mitigating contamination and improving evaluation robustness. We apply VAR-MATH to transform two popular benchmarks, AMC23 and AIME24, into their symbolic counterparts, VAR-AMC23 and VAR-AIME24. Experimental results reveal substantial performance drops for RL-trained models on the variabilized versions, especially for smaller models, with average declines of 48.0\% on AMC23 and 58.3\% on AIME24. These findings suggest that many existing RL methods rely on superficial heuristics and fail to generalize beyond specific numerical forms. Overall, VAR-MATH offers a principled, contamination-resistant evaluation paradigm for mathematical reasoning. 

**Abstract (ZH)**: Recent Advances in Reinforcement Learning for Mathematical Reasoning of Large Language Models: The VAR-MATH Framework 

---
# Emotional Support with LLM-based Empathetic Dialogue Generation 

**Title (ZH)**: 基于LLM的同理心对话生成的情感支持 

**Authors**: Shiquan Wang, Ruiyu Fang, Zhongjiang He, Shuangyong Song, Yongxiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.12820)  

**Abstract**: Emotional Support Conversation (ESC) aims to provide empathetic and effective emotional assistance through dialogue, addressing the growing demand for mental health support. This paper presents our solution for the NLPCC 2025 Task 8 ESC evaluation, where we leverage large-scale language models enhanced by prompt engineering and finetuning techniques. We explore both parameter-efficient Low-Rank Adaptation and full-parameter fine-tuning strategies to improve the model's ability to generate supportive and contextually appropriate responses. Our best model ranked second in the competition, highlighting the potential of combining LLMs with effective adaptation methods for ESC tasks. Future work will focus on further enhancing emotional understanding and response personalization to build more practical and reliable emotional support systems. 

**Abstract (ZH)**: 情感支持对话（ESC）旨在通过对话提供共情和支持，应对日益增长的心理健康支持需求。本文介绍了我们参加2025年NLPCC任务8ESC评估的解决方案，其中我们利用增强的大量语言模型并通过提示工程和微调技术。我们探索了参数高效的低秩适应和全参数微调策略，以提高模型生成支持性和上下文相关回应的能力。我们的最佳模型在比赛中排名第二，突显了将大语言模型与有效的适应方法结合使用在ESC任务中的潜力。未来的工作将集中在进一步增强情感理解并个性化回应，以构建更加实用和可靠的的情感支持系统。 

---
# MCPEval: Automatic MCP-based Deep Evaluation for AI Agent Models 

**Title (ZH)**: MCPEval：基于MCP的自动深度评估方法用于AI代理模型 

**Authors**: Zhiwei Liu, Jielin Qiu, Shiyu Wang, Jianguo Zhang, Zuxin Liu, Roshan Ram, Haolin Chen, Weiran Yao, Huan Wang, Shelby Heinecke, Silvio Savarese, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2507.12806)  

**Abstract**: The rapid rise of Large Language Models (LLMs)-based intelligent agents underscores the need for robust, scalable evaluation frameworks. Existing methods rely on static benchmarks and labor-intensive data collection, limiting practical assessment. We introduce \oursystemname, an open-source Model Context Protocol (MCP)-based framework that automates end-to-end task generation and deep evaluation of LLM agents across diverse domains. MCPEval standardizes metrics, seamlessly integrates with native agent tools, and eliminates manual effort in building evaluation pipelines. Empirical results across five real-world domains show its effectiveness in revealing nuanced, domain-specific performance. We publicly release MCPEval this https URL to promote reproducible and standardized LLM agent evaluation. 

**Abstract (ZH)**: 基于大型语言模型的智能代理的快速崛起凸显了构建稳健可扩展评估框架的必要性。现有的方法依赖于静态基准和劳动密集型数据收集，限制了实际评估。我们介绍了一种开源的基于Model Context Protocol (MCP)的框架\oursystemname，该框架实现了从头到尾的任务自动化生成和LLM代理在多种领域的深层次评估。MCP标准统一了评价指标，无缝集成原生代理工具，并消除了构建评估流水线的手工努力。跨五个实际领域的实证结果表明，其在揭示特定领域精细性能方面具有有效性。我们在此公开发布MCPEval（详见链接）以促进可复现和标准化的大语言模型代理评估。 

---
# VisionThink: Smart and Efficient Vision Language Model via Reinforcement Learning 

**Title (ZH)**: VisionThink：通过强化学习实现的智能高效视觉语言模型 

**Authors**: Senqiao Yang, Junyi Li, Xin Lai, Bei Yu, Hengshuang Zhao, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2507.13348)  

**Abstract**: Recent advancements in vision-language models (VLMs) have improved performance by increasing the number of visual tokens, which are often significantly longer than text tokens. However, we observe that most real-world scenarios do not require such an extensive number of visual tokens. While the performance drops significantly in a small subset of OCR-related tasks, models still perform accurately in most other general VQA tasks with only 1/4 resolution. Therefore, we propose to dynamically process distinct samples with different resolutions, and present a new paradigm for visual token compression, namely, VisionThink. It starts with a downsampled image and smartly decides whether it is sufficient for problem solving. Otherwise, the model could output a special token to request the higher-resolution image. Compared to existing Efficient VLM methods that compress tokens using fixed pruning ratios or thresholds, VisionThink autonomously decides whether to compress tokens case by case. As a result, it demonstrates strong fine-grained visual understanding capability on OCR-related tasks, and meanwhile saves substantial visual tokens on simpler tasks. We adopt reinforcement learning and propose the LLM-as-Judge strategy to successfully apply RL to general VQA tasks. Moreover, we carefully design a reward function and penalty mechanism to achieve a stable and reasonable image resize call ratio. Extensive experiments demonstrate the superiority, efficiency, and effectiveness of our method. Our code is available at this https URL. 

**Abstract (ZH)**: Recent advancements in 视觉-语言模型的最近进展通过增加视觉标记的数量提升了性能，这些视觉标记往往比文本标记长得多。然而，我们观察到大多数实际应用场景不需要如此大量的视觉标记。尽管在一小部分OCR相关的任务中性能显著下降，但在大多数其他一般视觉问答任务中，仅使用1/4分辨率就能准确完成任务。因此，我们提出了一种不同样本动态处理不同分辨率的新方法，并提出了一种视觉标记压缩的新范式，即VisionThink。它从下采样的图像开始，智能地决定是否足够解决问题，否则模型可以输出一个特殊标记来请求更高分辨率的图像。相比现有的使用固定剪枝比或阈值的高效视觉模型方法，VisionThink能够逐案自主决定是否压缩标记，从而在OCR相关的任务中展示了强大的细粒度视觉理解能力，并且在较简单的任务中节省了大量视觉标记。我们采用强化学习并提出LLM-as-Judge策略成功将RL应用于一般视觉问答任务。此外，我们精心设计了奖励函数和惩罚机制，以实现稳定的合理的图像缩放调用比例。广泛的实验证明了我们方法的优越性、高效性和有效性。我们的代码可在以下链接获取。 

---
# Vision-and-Language Training Helps Deploy Taxonomic Knowledge but Does Not Fundamentally Alter It 

**Title (ZH)**: 视觉-语言训练有助于部署分类知识但不会从根本上改变它 

**Authors**: Yulu Qin, Dheeraj Varghese, Adam Dahlgren Lindström, Lucia Donatelli, Kanishka Misra, Najoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.13328)  

**Abstract**: Does vision-and-language (VL) training change the linguistic representations of language models in meaningful ways? Most results in the literature have shown inconsistent or marginal differences, both behaviorally and representationally. In this work, we start from the hypothesis that the domain in which VL training could have a significant effect is lexical-conceptual knowledge, in particular its taxonomic organization. Through comparing minimal pairs of text-only LMs and their VL-trained counterparts, we first show that the VL models often outperform their text-only counterparts on a text-only question-answering task that requires taxonomic understanding of concepts mentioned in the questions. Using an array of targeted behavioral and representational analyses, we show that the LMs and VLMs do not differ significantly in terms of their taxonomic knowledge itself, but they differ in how they represent questions that contain concepts in a taxonomic relation vs. a non-taxonomic relation. This implies that the taxonomic knowledge itself does not change substantially through additional VL training, but VL training does improve the deployment of this knowledge in the context of a specific task, even when the presentation of the task is purely linguistic. 

**Abstract (ZH)**: 视觉-语言（VL）训练是否以有意义的方式改变了语言模型的语义表示？现有文献中的大多数结果在行为和表示上显示出了不一致或边际差异。在这个工作中，我们假设VL训练可能在词汇-概念知识，尤其是其分类组织方面产生显著影响。通过比较仅文本模型和其VL训练版本的最小对，我们首先展示了在需要理解问题中提到的概念的分类组织的文本仅问答任务上，VL模型往往超越其仅文本对应模型。利用一系列有针对性的行为和表示分析，我们证明了语言模型和视觉语言模型在分类知识本身上并没有显著差异，但在表示包含分类关系的概念的问题 vs. 非分类关系的概念的问题上存在差异。这表明通过额外的VL训练，分类知识本身并未发生显著变化，但VL训练确实改进了在这种特定任务中的知识应用，即使任务的呈现完全是语言性的。 

---
# AbGen: Evaluating Large Language Models in Ablation Study Design and Evaluation for Scientific Research 

**Title (ZH)**: AbGen: 在消融研究设计与评估中的大型语言模型评价方法 

**Authors**: Yilun Zhao, Weiyuan Chen, Zhijian Xu, Manasi Patwardhan, Yixin Liu, Chengye Wang, Lovekesh Vig, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2507.13300)  

**Abstract**: We introduce AbGen, the first benchmark designed to evaluate the capabilities of LLMs in designing ablation studies for scientific research. AbGen consists of 1,500 expert-annotated examples derived from 807 NLP papers. In this benchmark, LLMs are tasked with generating detailed ablation study designs for a specified module or process based on the given research context. Our evaluation of leading LLMs, such as DeepSeek-R1-0528 and o4-mini, highlights a significant performance gap between these models and human experts in terms of the importance, faithfulness, and soundness of the ablation study designs. Moreover, we demonstrate that current automated evaluation methods are not reliable for our task, as they show a significant discrepancy when compared to human assessment. To better investigate this, we develop AbGen-Eval, a meta-evaluation benchmark designed to assess the reliability of commonly used automated evaluation systems in measuring LLM performance on our task. We investigate various LLM-as-Judge systems on AbGen-Eval, providing insights for future research on developing more effective and reliable LLM-based evaluation systems for complex scientific tasks. 

**Abstract (ZH)**: 我们介绍了AbGen，这是首个用于评估LLM在设计科学研究所需消融研究能力的标准基准。AbGen包含1,500个由807篇NLP论文衍生出的专家注释示例。在这个基准中，LLM需要根据给定的研究背景生成指定模块或过程的详细消融研究设计方案。我们对DeepSeek-R1-0528和o4-mini等领先LLM的评估显示，这些模型在重要性、忠实度和合理性方面与人类专家之间的性能差距显著。此外，我们证明当前的自动化评估方法对于我们的任务而言不够可靠，因为它们与人类评估相比表现出显著差异。为了更好地研究这一问题，我们开发了AbGen-Eval，这是一个用于评估常用自动化评估系统可靠性的元评估基准，旨在测量LLM在完成我们任务时的表现。我们对AbGen-Eval上的各种LLM-as-Judge系统进行了研究，为开发更有效和可靠的基于LLM的评估系统提供了对未来复杂科学任务研究的见解。 

---
# Towards Formal Verification of LLM-Generated Code from Natural Language Prompts 

**Title (ZH)**: 面向自然语言提示生成的LLM代码形式化验证 

**Authors**: Aaron Councilman, David Fu, Aryan Gupta, Chengxiao Wang, David Grove, Yu-Xiong Wang, Vikram Adve  

**Link**: [PDF](https://arxiv.org/pdf/2507.13290)  

**Abstract**: In the past few years LLMs have emerged as a tool that can aid programmers by taking natural language descriptions and generating code based on it. However, LLMs often generate incorrect code that users need to fix and the literature suggests users often struggle to detect these errors. In this work we seek to offer formal guarantees of correctness to LLM generated code; such guarantees could improve the experience of using AI Code Assistants and potentially enable natural language programming for users with little or no programming knowledge. To address this challenge we propose to incorporate a formal query language that can represent a user's intent in a formally defined but natural language-like manner that a user can confirm matches their intent. Then, using such a query we propose to verify LLM generated code to ensure it matches the user's intent. We implement these ideas in our system, Astrogator, for the Ansible programming language which includes such a formal query language, a calculus for representing the behavior of Ansible programs, and a symbolic interpreter which is used for the verification. On a benchmark suite of 21 code-generation tasks, our verifier is able to verify correct code in 83% of cases and identify incorrect code in 92%. 

**Abstract (ZH)**: 近年来，大规模语言模型(LLMs)已成为一种辅助程序员的工具，可以通过自然语言描述生成代码。然而，LLM们往往生成错误的代码，用户需要进行修正，文献显示用户往往难以检测这些错误。在本工作中，我们旨在为LLM生成的代码提供形式化的正确性保证；这样的保证将提升使用AI代码助手的体验，并有可能使不懂编程知识的用户实现基于自然语言的编程。为了应对这一挑战，我们提出将一种形式化的查询语言融入其中，该语言能够以一种用户可以确认与自身意图一致的、形式化但类似自然语言的方式表示用户意图。然后，我们使用这样的查询验证LLM生成的代码，以确保代码符合用户的意图。我们在包含这种形式化的查询语言、表示Ansible程序行为的计算规则以及用于验证的符号解释器的系统Astrogator中实现了这些想法。在包含21个代码生成任务的基准测试套件中，我们的验证器能够正确验证代码的83%的情况，并识别出错误代码的92%的情况。 

---
# Overview of the TalentCLEF 2025: Skill and Job Title Intelligence for Human Capital Management 

**Title (ZH)**: TalentCLEF 2025：人才技能与职位标题智能化在人力资源资本管理中的概述 

**Authors**: Luis Gasco, Hermenegildo Fabregat, Laura García-Sardiña, Paula Estrella, Daniel Deniz, Alvaro Rodrigo, Rabih Zbib  

**Link**: [PDF](https://arxiv.org/pdf/2507.13275)  

**Abstract**: Advances in natural language processing and large language models are driving a major transformation in Human Capital Management, with a growing interest in building smart systems based on language technologies for talent acquisition, upskilling strategies, and workforce planning. However, the adoption and progress of these technologies critically depend on the development of reliable and fair models, properly evaluated on public data and open benchmarks, which have so far been unavailable in this domain.
To address this gap, we present TalentCLEF 2025, the first evaluation campaign focused on skill and job title intelligence. The lab consists of two tasks: Task A - Multilingual Job Title Matching, covering English, Spanish, German, and Chinese; and Task B - Job Title-Based Skill Prediction, in English. Both corpora were built from real job applications, carefully anonymized, and manually annotated to reflect the complexity and diversity of real-world labor market data, including linguistic variability and gender-marked expressions.
The evaluations included monolingual and cross-lingual scenarios and covered the evaluation of gender bias.
TalentCLEF attracted 76 registered teams with more than 280 submissions. Most systems relied on information retrieval techniques built with multilingual encoder-based models fine-tuned with contrastive learning, and several of them incorporated large language models for data augmentation or re-ranking. The results show that the training strategies have a larger effect than the size of the model alone. TalentCLEF provides the first public benchmark in this field and encourages the development of robust, fair, and transferable language technologies for the labor market. 

**Abstract (ZH)**: 自然语言处理和大型语言模型的发展正在推动人力资源管理的重大变革，越来越多的研究关注基于语言技术的智能系统在 talent acquisition、技能提升策略和劳动力规划方面的作用。然而，这些技术的采用和进步关键取决于可靠和公平模型的发展，这些模型应在公开数据和开放基准上进行适当评估，而目前这些资源在该领域还不可用。
为填补这一空白，我们提出了 TalentCLEF 2025，这是首个专注于技能和职位标题智能的评估活动。实验室包括两个任务：任务A - 多语言职位标题匹配，涵盖英语、西班牙语、德语和中文；任务B - 基于职位标题的技能预测，使用英语。两个语料库均基于真实的职业申请，经过仔细脱敏并手工标注，以反映现实世界劳动力市场的复杂性和多样性，包括语言变异性及性别标记的表达。
评估包括单语和跨语言场景，并涵盖了性别偏见的评估。
TalentCLEF 获得了 76 支注册队伍，提交了超过 280 份提交作品。大多数系统依赖于使用多语言编码器模型并结合对比学习进行微调的信息检索技术，其中一些系统还结合了大型语言模型进行数据增强或重排。结果显示，训练策略的效果比模型大小本身更大。TalentCLEF 提供了该领域的首个公开基准，并促进了适用于劳动力市场的稳健、公平和可转移的语言技术的发展。 

---
# QuestA: Expanding Reasoning Capacity in LLMs via Question Augmentation 

**Title (ZH)**: QuestA: 通过问题扩充增强LLM的推理能力 

**Authors**: Jiazheng Li, Hong Lu, Kaiyue Wen, Zaiwen Yang, Jiaxuan Gao, Hongzhou Lin, Yi Wu, Jingzhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.13266)  

**Abstract**: Reinforcement learning (RL) has become a key component in training large language reasoning models (LLMs). However, recent studies questions its effectiveness in improving multi-step reasoning-particularly on hard problems. To address this challenge, we propose a simple yet effective strategy via Question Augmentation: introduce partial solutions during training to reduce problem difficulty and provide more informative learning signals. Our method, QuestA, when applied during RL training on math reasoning tasks, not only improves pass@1 but also pass@k-particularly on problems where standard RL struggles to make progress. This enables continual improvement over strong open-source models such as DeepScaleR and OpenMath Nemotron, further enhancing their reasoning capabilities. We achieve new state-of-the-art results on math benchmarks using 1.5B-parameter models: 67.1% (+5.3%) on AIME24, 59.5% (+10.0%) on AIME25, and 35.5% (+4.0%) on HMMT25. Further, we provide theoretical explanations that QuestA improves sample efficiency, offering a practical and generalizable pathway for expanding reasoning capability through RL. 

**Abstract (ZH)**: 强化学习（RL）已成为训练大规模语言推理模型（LLMs）的关键组成部分。然而，近期的研究对其在提高多步推理能力方面的有效性提出了质疑，特别是在解决难题时。为应对这一挑战，我们提出了一种简单而有效的方法——问题扩增：在训练过程中引入部分解决方案，以降低问题难度并提供更加信息丰富的学习信号。我们的方法QuestA，在应用到数学推理任务的RL训练中，不仅能改善pass@1，还能在标准RL难以取得进展的问题上显著提高pass@k。这使得我们的模型能够持续超越如DeepScaleR和OpenMath Nemotron等强大的开源模型，进一步增强其推理能力。我们使用拥有1.5B参数的模型在数学基准测试中取得了新的最先进成果：AIME24上67.1% (+5.3%)，AIME25上59.5% (+10.0%)，HMMT25上35.5% (+4.0%)。此外，我们提供了理论解释，证明QuestA提高了样本效率，为通过RL扩展推理能力提供了一条实用且可推广的途径。 

---
# Automating Steering for Safe Multimodal Large Language Models 

**Title (ZH)**: 自动调控以确保安全的多模态大型语言模型 

**Authors**: Lyucheng Wu, Mengru Wang, Ziwen Xu, Tri Cao, Nay Oo, Bryan Hooi, Shumin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2507.13255)  

**Abstract**: Recent progress in Multimodal Large Language Models (MLLMs) has unlocked powerful cross-modal reasoning abilities, but also raised new safety concerns, particularly when faced with adversarial multimodal inputs. To improve the safety of MLLMs during inference, we introduce a modular and adaptive inference-time intervention technology, AutoSteer, without requiring any fine-tuning of the underlying model. AutoSteer incorporates three core components: (1) a novel Safety Awareness Score (SAS) that automatically identifies the most safety-relevant distinctions among the model's internal layers; (2) an adaptive safety prober trained to estimate the likelihood of toxic outputs from intermediate representations; and (3) a lightweight Refusal Head that selectively intervenes to modulate generation when safety risks are detected. Experiments on LLaVA-OV and Chameleon across diverse safety-critical benchmarks demonstrate that AutoSteer significantly reduces the Attack Success Rate (ASR) for textual, visual, and cross-modal threats, while maintaining general abilities. These findings position AutoSteer as a practical, interpretable, and effective framework for safer deployment of multimodal AI systems. 

**Abstract (ZH)**: 最近在多模态大型语言模型（MLLMs）方面取得的进展解锁了强大的跨模态推理能力，但也引发了新的安全关切，特别是在面对 adversarial 多模态输入时。为了在推理过程中提高 MLLMs 的安全性，我们介绍了一种无需对底层模型进行微调的模块化和自适应推理时干预技术 AutoSteer。AutoSteer 包含三个核心组件：（1）一种新颖的安全意识评分（SAS），能够自动识别模型内部层级中最具安全相关性的区别；（2）一种自适应安全探针，训练用于估算中间表示生成有毒输出的可能性；（3）一种轻量级拒绝端，能够在检测到安全风险时有选择地干预以调节生成。在 LLaVA-OV 和 Chameleon 上跨多种安全关键基准的实验表明，AutoSteer 显著降低了对文本、视觉和跨模态威胁的攻击成功率（ASR），同时保持了一般能力。这些发现将 AutoSteer 定位为一种实用、可解释且有效的框架，用于更安全部署多模态 AI 系统。 

---
# HATS: Hindi Analogy Test Set for Evaluating Reasoning in Large Language Models 

**Title (ZH)**: HATS：用于评估大规模语言模型推理能力的印地语类比测试集 

**Authors**: Ashray Gupta, Rohan Joseph, Sunny Rai  

**Link**: [PDF](https://arxiv.org/pdf/2507.13238)  

**Abstract**: Analogies test a model's ability to infer implicit relationships between concepts, making them a key benchmark for evaluating reasoning capabilities. While large language models (LLMs) are widely evaluated for reasoning in English, their abilities in Indic languages remain understudied, limiting our understanding of whether these models generalize across languages. To address this gap, we introduce a new Hindi Analogy Test Set (HATS), comprising 405 multiple-choice questions sourced from Indian government exams. We benchmark state-of-the-art multilingual LLMs using various prompting strategies and introduce a grounded Chain of Thought approach that leverages cognitive theories of analogical reasoning. This approach improves model performance on Hindi analogy questions. Our experiments show that models perform best with English prompts, irrespective of the prompting strategy. Our test set addresses the lack of a critical resource to evaluate LLM reasoning capabilities in Hindi. 

**Abstract (ZH)**: Hindi Analogies Test Set (HATS): Evaluating Reasoning Capabilities of Large Language Models in Indian Languages 

---
# Prompt Injection 2.0: Hybrid AI Threats 

**Title (ZH)**: Prompt Injection 2.0: 混合AI威胁 

**Authors**: Jeremy McHugh, Kristina Šekrst, Jon Cefalu  

**Link**: [PDF](https://arxiv.org/pdf/2507.13169)  

**Abstract**: Prompt injection attacks, where malicious input is designed to manipulate AI systems into ignoring their original instructions and following unauthorized commands instead, were first discovered by Preamble, Inc. in May 2022 and responsibly disclosed to OpenAI. Over the last three years, these attacks have continued to pose a critical security threat to LLM-integrated systems. The emergence of agentic AI systems, where LLMs autonomously perform multistep tasks through tools and coordination with other agents, has fundamentally transformed the threat landscape. Modern prompt injection attacks can now combine with traditional cybersecurity exploits to create hybrid threats that systematically evade traditional security controls. This paper presents a comprehensive analysis of Prompt Injection 2.0, examining how prompt injections integrate with Cross-Site Scripting (XSS), Cross-Site Request Forgery (CSRF), and other web security vulnerabilities to bypass traditional security measures. We build upon Preamble's foundational research and mitigation technologies, evaluating them against contemporary threats, including AI worms, multi-agent infections, and hybrid cyber-AI attacks. Our analysis incorporates recent benchmarks that demonstrate how traditional web application firewalls, XSS filters, and CSRF tokens fail against AI-enhanced attacks. We also present architectural solutions that combine prompt isolation, runtime security, and privilege separation with novel threat detection capabilities. 

**Abstract (ZH)**: Prompt注入攻击2.0：与跨站脚本(XSS)、跨站请求伪造(CSRF)及其他Web安全漏洞的综合分析 

---
# Inverse Reinforcement Learning Meets Large Language Model Post-Training: Basics, Advances, and Opportunities 

**Title (ZH)**: 逆强化学习与大型语言模型后训练：基础知识、进展与机遇 

**Authors**: Hao Sun, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2507.13158)  

**Abstract**: In the era of Large Language Models (LLMs), alignment has emerged as a fundamental yet challenging problem in the pursuit of more reliable, controllable, and capable machine intelligence. The recent success of reasoning models and conversational AI systems has underscored the critical role of reinforcement learning (RL) in enhancing these systems, driving increased research interest at the intersection of RL and LLM alignment. This paper provides a comprehensive review of recent advances in LLM alignment through the lens of inverse reinforcement learning (IRL), emphasizing the distinctions between RL techniques employed in LLM alignment and those in conventional RL tasks. In particular, we highlight the necessity of constructing neural reward models from human data and discuss the formal and practical implications of this paradigm shift. We begin by introducing fundamental concepts in RL to provide a foundation for readers unfamiliar with the field. We then examine recent advances in this research agenda, discussing key challenges and opportunities in conducting IRL for LLM alignment. Beyond methodological considerations, we explore practical aspects, including datasets, benchmarks, evaluation metrics, infrastructure, and computationally efficient training and inference techniques. Finally, we draw insights from the literature on sparse-reward RL to identify open questions and potential research directions. By synthesizing findings from diverse studies, we aim to provide a structured and critical overview of the field, highlight unresolved challenges, and outline promising future directions for improving LLM alignment through RL and IRL techniques. 

**Abstract (ZH)**: 在大型语言模型时代，对齐问题已成为追求更可靠、可控和强大的机器智能的根本而又富有挑战的问题。最近推理模型和对话AI系统的成功进一步凸显了强化学习（RL）在增强这些系统中的关键作用，推动了RL与大型语言模型对齐研究领域的广泛关注。本文通过逆强化学习（IRL）的视角对近期大型语言模型对齐领域的进展进行了全面回顾，强调了在大型语言模型对齐中使用的RL技术与传统RL任务中使用的RL技术之间的区别。特别地，我们强调了从人类数据构建神经奖励模型的必要性，并讨论了这一范式转变的正式和实践意义。我们从介绍 RL 的基本概念开始，为不熟悉该领域的读者提供基础。然后，我们探讨了这一研究议程的最新进展，讨论了进行大型语言模型对齐的IRL时面临的挑战和机遇。我们不仅考虑方法论方面的问题，还探讨了实际方面的内容，包括数据集、基准、评估指标、基础设施以及计算效率高的训练和推理技术。最后，我们借鉴稀疏奖励强化学习领域的文献，识别出尚未解决的问题和潜在的研究方向。通过综合来自不同研究的发现，我们旨在提供一个结构化且批判性的领域概览，突出未解决的挑战，并概述通过RL和IRL技术改进大型语言模型对齐的有前途的方向。 

---
# Teach Old SAEs New Domain Tricks with Boosting 

**Title (ZH)**: 使用提升技术让旧SAEs学习新领域技巧 

**Authors**: Nikita Koriagin, Yaroslav Aksenov, Daniil Laptev, Gleb Gerasimov, Nikita Balagansky, Daniil Gavrilov  

**Link**: [PDF](https://arxiv.org/pdf/2507.12990)  

**Abstract**: Sparse Autoencoders have emerged as powerful tools for interpreting the internal representations of Large Language Models, yet they often fail to capture domain-specific features not prevalent in their training corpora. This paper introduces a residual learning approach that addresses this feature blindness without requiring complete retraining. We propose training a secondary SAE specifically to model the reconstruction error of a pretrained SAE on domain-specific texts, effectively capturing features missed by the primary model. By summing the outputs of both models during inference, we demonstrate significant improvements in both LLM cross-entropy and explained variance metrics across multiple specialized domains. Our experiments show that this method efficiently incorporates new domain knowledge into existing SAEs while maintaining their performance on general tasks. This approach enables researchers to selectively enhance SAE interpretability for specific domains of interest, opening new possibilities for targeted mechanistic interpretability of LLMs. 

**Abstract (ZH)**: 稀疏自主编码器已发展成为解读大型语言模型内部表示的强大多用途工具，然而它们往往无法捕获在训练语料中不普遍的领域特异性特征。本文提出了一种残差学习方法，该方法在无需完全重新训练的情况下解决了这种特征盲区问题。我们建议训练一个次级SAE，专门用于建模预训练SAE在特定领域文本上的重构误差，从而有效捕获主要模型遗漏的特征。通过推理时将两个模型的输出相加，我们在多个专门领域展示了在交叉熵和解释方差指标上的显著改进。我们的实验表明，该方法能够高效地将新领域知识整合进现有的SAE中，同时保持其在通用任务上的性能。这种方法允许研究人员有选择地增强SAE对特定领域兴趣的可解释性，为大型语言模型的针对性机制解释打开了新的可能性。 

---
# MRT at IberLEF-2025 PRESTA Task: Maximizing Recovery from Tables with Multiple Steps 

**Title (ZH)**: MRT在IberLEF-2025 PRESTA任务中：通过多步操作最大化表格内容恢复 

**Authors**: Maximiliano Hormazábal Lagos, Álvaro Bueno Sáez, Héctor Cerezo-Costas, Pedro Alonso Doval, Jorge Alcalde Vesteiro  

**Link**: [PDF](https://arxiv.org/pdf/2507.12981)  

**Abstract**: This paper presents our approach for the IberLEF 2025 Task PRESTA: Preguntas y Respuestas sobre Tablas en Español (Questions and Answers about Tables in Spanish). Our solution obtains answers to the questions by implementing Python code generation with LLMs that is used to filter and process the table. This solution evolves from the MRT implementation for the Semeval 2025 related task. The process consists of multiple steps: analyzing and understanding the content of the table, selecting the useful columns, generating instructions in natural language, translating these instructions to code, running it, and handling potential errors or exceptions. These steps use open-source LLMs and fine-grained optimized prompts for each step. With this approach, we achieved an accuracy score of 85\% in the task. 

**Abstract (ZH)**: 本文介绍了我们针对IberLEF 2025任务PRESTA：关于西班牙语表格的问题与答案的解决方案。我们的方法通过使用LLMs生成Python代码来筛选和处理表格以获取问题的答案。该解决方案是从Semeval 2025相关任务的MRT实现发展而来。过程包括多个步骤：分析和理解表格内容、选择有用列、生成自然语言指令、将这些指令翻译成代码、运行代码并处理潜在的错误或异常。这些步骤使用开源LLMs和细粒度优化的提示。通过这种方法，我们在任务中达到了85%的准确率。 

---
# Making Language Model a Hierarchical Classifier and Generator 

**Title (ZH)**: 使语言模型成为层次分类器和生成器 

**Authors**: Yihong Wang, Zhonglin Jiang, Ningyuan Xi, Yue Zhao, Qingqing Gu, Xiyuan Chen, Hao Wu, Sheng Xu, Hange Zhou, Yong Chen, Luo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2507.12930)  

**Abstract**: Decoder-only language models, such as GPT and LLaMA, generally decode on the last layer. Motivated by human's hierarchical thinking capability, we propose that a hierarchical decoder architecture could be built with different layers decoding texts simultaneously. Due to limited time and computationally resources, we choose to adapt a pretrained language model into this form of hierarchical decoder. Language heads of the last layer are copied to different selected intermediate layers, and fine-tuned with different task inputs. By thorough experiments, we validate that these selective intermediate layers could be adapted to speak meaningful and reasonable contents, and this paradigm of hierarchical decoder can obtain state-of-the-art performances on multiple tasks such as hierarchical text classification, classification-guided generation, and hierarchical text generation. This study suggests the possibility of a generalized hierarchical reasoner, pretraining from scratch. 

**Abstract (ZH)**: 仅解码器架构的语言模型，如GPT和LLaMA，通常在最后一层进行解码。受人类分层思维能力的启发，我们提出可以构建一种不同层同时解码文本的分层解码器架构。由于时间和计算资源的限制，我们选择将一个预训练语言模型调整为这种分层解码器的形式。最后一层的语言头被复制到不同的选定中间层，并在不同的任务输入下进行微调。通过 thorough 实验，我们验证了这些选择性的中间层可以被调整以生成有意义和合理的文本内容，并且这种分层解码器的范式可以在分级文本分类、分类指导生成和分级文本生成等多个任务上获得最新的性能。这项研究建议从头开始预训练一个通用的分层推理器的可能性。 

---
# Supervised Fine Tuning on Curated Data is Reinforcement Learning (and can be improved) 

**Title (ZH)**: 监督微调在精选数据上的应用是强化学习（并且可以改进） 

**Authors**: Chongli Qin, Jost Tobias Springenberg  

**Link**: [PDF](https://arxiv.org/pdf/2507.12856)  

**Abstract**: Behavior Cloning (BC) on curated (or filtered) data is the predominant paradigm for supervised fine-tuning (SFT) of large language models; as well as for imitation learning of control policies. Here, we draw on a connection between this successful strategy and the theory and practice of finding optimal policies via Reinforcement Learning (RL). Building on existing literature, we clarify that SFT can be understood as maximizing a lower bound on the RL objective in a sparse reward setting. Giving support to its often observed good performance. From this viewpoint, we realize that a small modification to SFT leads to an importance weighted variant that behaves closer to training with RL as it: i) optimizes a tighter bound to the RL objective and, ii) can improve performance compared to SFT on curated data. We refer to this variant as importance weighted supervised fine-tuning (iw-SFT). We show that it is easy to implement and can be further generalized to training with quality scored data. The resulting SFT variants are competitive with more advanced RL algorithms for large language models and for training policies in continuous control tasks. For example achieving 66.7% on the AIME 2024 dataset. 

**Abstract (ZH)**: 基于精选数据的行为克隆（BC）是监督微调（SFT）大语言模型和模仿学习控制策略的主要范式；本研究通过强化学习（RL）理论与实践的联系，阐明SFT可以理解为在稀疏奖励设置下最大化RL目标的下界，从而解释其常观察到的良好性能。我们提出一种对SFT的小修改，形成一种加权重要性采样的变种（iw-SFT），这种变种更接近于利用RL进行训练，并在精选数据上可能表现更优。我们展示了这一变种易于实现，并能进一步推广到质量评分数据的训练。所得的SFT变种在大语言模型和连续控制任务政策训练中与更先进的RL算法具有竞争力，例如在AIME 2024数据集上达到66.7%。 

---
# Large Language Models' Internal Perception of Symbolic Music 

**Title (ZH)**: 大型语言模型对符号音乐的内在感知 

**Authors**: Andrew Shin, Kunitake Kaneko  

**Link**: [PDF](https://arxiv.org/pdf/2507.12808)  

**Abstract**: Large language models (LLMs) excel at modeling relationships between strings in natural language and have shown promise in extending to other symbolic domains like coding or mathematics. However, the extent to which they implicitly model symbolic music remains underexplored. This paper investigates how LLMs represent musical concepts by generating symbolic music data from textual prompts describing combinations of genres and styles, and evaluating their utility through recognition and generation tasks. We produce a dataset of LLM-generated MIDI files without relying on explicit musical training. We then train neural networks entirely on this LLM-generated MIDI dataset and perform genre and style classification as well as melody completion, benchmarking their performance against established models. Our results demonstrate that LLMs can infer rudimentary musical structures and temporal relationships from text, highlighting both their potential to implicitly encode musical patterns and their limitations due to a lack of explicit musical context, shedding light on their generative capabilities for symbolic music. 

**Abstract (ZH)**: 大型语言模型（LLMs）在建模自然语言中的字符串关系方面表现出色，并在扩展到诸如编码或数学的其他符号领域方面展现出前景。然而，它们在隐式建模符号音乐方面的程度仍然尚未充分探索。本文通过从描述不同流派和风格组合的文本提示生成符号音乐数据，并通过识别和生成任务评估其实用性，来探究LLMs如何表示音乐概念。我们生成了一个不依赖于显式音乐训练的LLM生成的MIDI文件数据集。然后，我们完全基于此LLM生成的MIDI数据集训练神经网络，并进行了流派和风格分类以及旋律填充，将它们的性能与现有模型进行比较。我们的结果显示，LLMs可以从文本中推断出基本的音乐结构和时间关系，这既突显了它们隐式编码音乐模式的潜力，也揭示了由于缺乏明确的音乐上下文而导致的局限性，从而阐明了它们在符号音乐生成方面的能力。 

---
# A Comprehensive Survey of Electronic Health Record Modeling: From Deep Learning Approaches to Large Language Models 

**Title (ZH)**: 电子健康记录建模综述：从深度学习方法到大型语言模型 

**Authors**: Weijieying Ren, Jingxi Zhu, Zehao Liu, Tianxiang Zhao, Vasant Honavar  

**Link**: [PDF](https://arxiv.org/pdf/2507.12774)  

**Abstract**: Artificial intelligence (AI) has demonstrated significant potential in transforming healthcare through the analysis and modeling of electronic health records (EHRs). However, the inherent heterogeneity, temporal irregularity, and domain-specific nature of EHR data present unique challenges that differ fundamentally from those in vision and natural language tasks. This survey offers a comprehensive overview of recent advancements at the intersection of deep learning, large language models (LLMs), and EHR modeling. We introduce a unified taxonomy that spans five key design dimensions: data-centric approaches, neural architecture design, learning-focused strategies, multimodal learning, and LLM-based modeling systems. Within each dimension, we review representative methods addressing data quality enhancement, structural and temporal representation, self-supervised learning, and integration with clinical knowledge. We further highlight emerging trends such as foundation models, LLM-driven clinical agents, and EHR-to-text translation for downstream reasoning. Finally, we discuss open challenges in benchmarking, explainability, clinical alignment, and generalization across diverse clinical settings. This survey aims to provide a structured roadmap for advancing AI-driven EHR modeling and clinical decision support. For a comprehensive list of EHR-related methods, kindly refer to this https URL. 

**Abstract (ZH)**: 人工 intelligence（AI）在通过电子健康记录（EHR）的分析与建模改造医疗健康方面展现了显著的潜力。然而，EHR数据固有的异质性、时间不规律性和领域特定性提出了与视觉和自然语言任务根本不同的独特挑战。本文综述了深度学习、大规模语言模型（LLM）与EHR建模交叉领域的最新进展。我们引入了一个统一的分类框架，涵盖了五大关键设计维度：以数据为中心的方法、神经网络架构设计、学习导向策略、多模态学习以及基于LLM的建模系统。在每个维度中，我们回顾了代表性的方法，涉及数据质量提升、结构和时间表示、自我监督学习以及与临床知识的集成。我们还强调了新兴趋势，如基础模型、由LLM驱动的临床代理以及EHR到文本的翻译以供下游推理。最后，我们讨论了基准测试、可解释性、临床对齐以及在多种临床环境中的泛化方面的开放性挑战。本文旨在为推进AI驱动的EHR建模和临床决策支持提供一个结构化的路线图。有关EHR相关方法的详细列表，请参见此链接：https URL。 

---
# Synergy: End-to-end Concept Model 

**Title (ZH)**: 协同效应：端到端概念模型 

**Authors**: Keli Zheng, Zerong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2507.12769)  

**Abstract**: In this paper, we present Synergy, a language model that bridges different levels of abstraction in an end-to-end fashion through a learned routing mechanism. Focusing on low-level linguistic abstraction, we trained our model as a byte-level language model. Our model spontaneously learns to tokenize bytes, producing fewer concept tokens than Byte-level Byte Pair Encoder (BBPE) tokenizers while keeping comparable performance. By comparing with Llama3, we observed an advantage of Synergy under the same model scale and training dataset size. Further studies show that the middle part (the higher abstraction part) of our model performs better when positional encodings are removed, suggesting the emergence of position-independent concepts. These findings demonstrate the feasibility of tokenizer-free architectures, paving the way for more robust and flexible pipelines. 

**Abstract (ZH)**: 本文介绍了一种名为Synergy的语言模型，该模型通过学习路由机制以端到端的方式连接不同层次的抽象。聚焦于低层次语言抽象，我们将模型训练成字节级语言模型。模型自主学习字节分词，产生的概念令牌少于Byte-Level Byte Pair Encoder (BBPE) 分词器，同时保持相当的性能。通过与Llama3的对比，我们发现在相同模型规模和训练数据集大小的情况下，Synergy展现出优势。进一步研究显示，在移除位置编码时，模型的中间部分（更高抽象的部分）表现更好，这表明出现了一种与位置无关的概念。这些发现展示了无分词器架构的可行性，为更稳健和灵活的管道铺平了道路。 

---
# Logit Arithmetic Elicits Long Reasoning Capabilities Without Training 

**Title (ZH)**: Logit Arithmetic 启发长推理能力无需训练 

**Authors**: Yunxiang Zhang, Muhammad Khalifa, Lechen Zhang, Xin Liu, Ayoung Lee, Xinliang Frederick Zhang, Farima Fatahi Bayat, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.12759)  

**Abstract**: Large reasoning models (LRMs) can do complex reasoning via long chain-of-thought (CoT) involving cognitive strategies such as backtracking and self-correction. Recent studies suggest that some models inherently possess these long reasoning abilities, which may be unlocked via extra training. Our work first investigates whether we can elicit such behavior without any training. To this end, we propose a decoding-time approach, ThinkLogit, which utilizes logits arithmetic (Liu et al., 2024) to tune a target large LM for long reasoning using a substantially smaller model as guider. We then show that we can further boost performance by training the guider model with preference optimization over correct/incorrect reasoning pairs sampled from both the target and guider model -- a setup we refer to as ThinkLogit-DPO. Our experiments demonstrate that ThinkLogit and ThinkLogit-DPO achieve a relative improvement in pass@1 by 26% and 29%, respectively, over four mathematical datasets using the Qwen2.5-32B when guided by R1-Distill-Qwen-1.5B -- a model 21x smaller. Lastly, we show that ThinkLogit can transfer long reasoning skills acquired through reinforcement learning, improving pass@1 by 13% relative compared to the Qwen2.5-32B base model. Our work presents a computationally-efficient method to elicit long reasoning in large models with minimal or no additional training. 

**Abstract (ZH)**: 大型推理模型（LRMs）可以通过长链推理（CoT）进行复杂的推理，涉及诸如回溯和自我修正等认知策略。近期研究表明，某些模型本身具备这些长期推理能力，可能通过额外训练被激活。我们首先研究是否可以在没有任何训练的情况下激发此类行为。为此，我们提出了一种解码时间方法ThinkLogit，该方法利用logits算术（Liu et al., 2024）来使用一个显著更小的模型作为引导来调优目标大模型以支持长推理。我们还展示了通过使用正误推理对对来自目标模型和引导模型的样本进行偏好优化训练引导模型的方法可以进一步提升性能，这被称为ThinkLogit-DPO。我们的实验表明，在使用R1-Distill-Qwen-1.5B（一个比Qwen2.5-32B小21倍的模型）引导时，ThinkLogit和ThinkLogit-DPO分别在四个数学数据集上相对提高了pass@1指标26%和29%。最后，我们展示了ThinkLogit可以转移通过强化学习获得的长推理能力，相对提高pass@1指标13%，相较于基模Qwen2.5-32B。我们的工作提出了一种计算效率高的方法，在极少或无需额外训练的情况下激发大型模型进行长推理。 

---
# ParaStudent: Generating and Evaluating Realistic Student Code by Teaching LLMs to Struggle 

**Title (ZH)**: ParaStudent: 通过教大规模语言模型挣扎来生成和评估现实中的学生代码 

**Authors**: Mihran Miroyan, Rose Niousha, Joseph E. Gonzalez, Gireeja Ranade, Narges Norouzi  

**Link**: [PDF](https://arxiv.org/pdf/2507.12674)  

**Abstract**: Large Language Models (LLMs) have shown strong performance on programming tasks, but can they generate student-like code like real students - imperfect, iterative, and stylistically diverse? We present ParaStudent, a systematic study of LLM-based "student-like" code generation in an introductory programming course setting. Using a dataset of timestamped student submissions across multiple semesters, we design low- and high-resolution experiments to model student progress and evaluate code outputs along semantic, functional, and stylistic dimensions. Our results show that fine-tuning significantly improves alignment with real student trajectories and captures error patterns, incremental improvements, and stylistic variations more faithfully. This study shows that modeling realistic student code requires capturing learning dynamics through context-aware generation, temporal modeling, and multi-dimensional evaluation. Code for experiments and evaluation is available at \href{this https URL}{\texttt{this http URL}}. 

**Abstract (ZH)**: 大型语言模型（LLMs）在编程任务上显示出了强大的性能，但它们能否生成像真实学生那样的代码——即不完美、迭代且风格多样的代码？我们呈现了ParaStudent，这是一种在入门级编程课程环境中对基于LLM的“学生样”代码生成进行系统研究的方法。利用跨多个学期的时间戳标记学生提交数据集，我们设计了低分辨率和高分辨率的实验，以建模学生的学习进展，并从语义、功能和风格三个维度评估代码输出。研究结果表明，微调显著提高了与真实学生轨迹的一致性，并更准确地捕捉到了错误模式、逐步改进和风格变化。本研究显示，要模拟真实的 student code，需要通过上下文感知生成、时间建模和多维度评估来捕捉学习动态。实验和评估代码可在 \href{this https URL}{\texttt{this http URL}} 获取。 

---
# Single Conversation Methodology: A Human-Centered Protocol for AI-Assisted Software Development 

**Title (ZH)**: 单对话方法学：面向人工智能辅助软件开发的人本协议 

**Authors**: Salvador D. Escobedo  

**Link**: [PDF](https://arxiv.org/pdf/2507.12665)  

**Abstract**: We propose the Single Conversation Methodology (SCM), a novel and pragmatic approach to software development using large language models (LLMs). In contrast to ad hoc interactions with generative AI, SCM emphasizes a structured and persistent development dialogue, where all stages of a project - from requirements to architecture and implementation - unfold within a single, long-context conversation. The methodology is grounded on principles of cognitive clarity, traceability, modularity, and documentation. We define its phases, best practices, and philosophical stance, while arguing that SCM offers a necessary correction to the passive reliance on LLMs prevalent in current practices. We aim to reassert the active role of the developer as architect and supervisor of the intelligent tool. 

**Abstract (ZH)**: 单对话方法论（SCM）：一种基于大型语言模型的新型实用软件开发方法 

---
# QSpark: Towards Reliable Qiskit Code Generation 

**Title (ZH)**: QSpark: 朝着可靠Qiskit 代码生成努力 

**Authors**: Kiana Kheiri, Aamna Aamir, Andriy Miranskyy, Chen Ding  

**Link**: [PDF](https://arxiv.org/pdf/2507.12642)  

**Abstract**: Quantum circuits must be error-resilient, yet LLMs like Granite-20B-Code and StarCoder often output flawed Qiskit code. We fine-tuned a 32 B model with two RL methods, Group Relative Policy Optimization (GRPO) and Odds-Ratio Preference Optimization (ORPO), using a richly annotated synthetic dataset. On the Qiskit HumanEval benchmark, ORPO reaches 56.29\% Pass@1 ($\approx+10$ pp over Granite-8B-QK) and GRPO hits 49\%, both beating all general-purpose baselines; on the original HumanEval they score 65.90\% and 63.00\%. GRPO excels on basic tasks (42/54), ORPO on intermediate ones (41/68), and neither solves the five advanced tasks, highlighting clear gains yet room for progress in AI-assisted quantum programming. 

**Abstract (ZH)**: 量子电路必须具备抗错误能力，但像Granite-20B-Code和StarCoder这样的LLM经常输出有缺陷的Qiskit代码。我们使用丰富的标注合成数据集，用两种RL方法（Group Relative Policy Optimization，GRPO；Odds-Ratio Preference Optimization，ORPO）微调了一个32B模型。在Qiskit HumanEval基准测试中，ORPO达到56.29% Pass@1（约比Granite-8B-QK高10个百分点），GRPO达到49%；在原始的HumanEval上，它们分别达到65.90%和63.00%。GRPO在基础任务上表现优异，ORPO在中级任务上表现优异，但两者都无法解决五个高级任务，这表明在AI辅助量子编程领域仍有改进空间。 

---
# BootSeer: Analyzing and Mitigating Initialization Bottlenecks in Large-Scale LLM Training 

**Title (ZH)**: BootSeer: 分析和缓解大规模LLM训练中的初始化瓶颈 

**Authors**: Rui Li, Xiaoyun Zhi, Jinxin Chi, Menghan Yu, Lixin Huang, Jia Zhu, Weilun Zhang, Xing Ma, Wenjia Liu, Zhicheng Zhu, Daowen Luo, Zuquan Song, Xin Yin, Chao Xiang, Shuguang Wang, Wencong Xiao, Gene Cooperman  

**Link**: [PDF](https://arxiv.org/pdf/2507.12619)  

**Abstract**: Large Language Models (LLMs) have become a cornerstone of modern AI, driving breakthroughs in natural language processing and expanding into multimodal jobs involving images, audio, and video. As with most computational software, it is important to distinguish between ordinary runtime performance and startup overhead. Prior research has focused on runtime performance: improving training efficiency and stability. This work focuses instead on the increasingly critical issue of startup overhead in training: the delay before training jobs begin execution. Startup overhead is particularly important in large, industrial-scale LLMs, where failures occur more frequently and multiple teams operate in iterative update-debug cycles. In one of our training clusters, more than 3.5% of GPU time is wasted due to startup overhead alone.
In this work, we present the first in-depth characterization of LLM training startup overhead based on real production data. We analyze the components of startup cost, quantify its direct impact, and examine how it scales with job size. These insights motivate the design of Bootseer, a system-level optimization framework that addresses three primary startup bottlenecks: (a) container image loading, (b) runtime dependency installation, and (c) model checkpoint resumption. To mitigate these bottlenecks, Bootseer introduces three techniques: (a) hot block record-and-prefetch, (b) dependency snapshotting, and (c) striped HDFS-FUSE. Bootseer has been deployed in a production environment and evaluated on real LLM training workloads, demonstrating a 50% reduction in startup overhead. 

**Abstract (ZH)**: 大型语言模型（LLMs）已成为现代人工智能的基石，推动了自然语言处理的突破，并扩展到涉及图像、音频和视频的多模态任务。与大多数计算软件一样，区分常规运行时性能和启动开销很重要。早期的研究主要关注运行时性能：提高训练效率和稳定性。本文则重点关注日益关键的训练启动开销问题：训练作业开始执行前的延迟。在大规模的工业级LLMs中，启动开销尤为重要，因为错误发生的频率更高，而且多个团队在迭代更新和调试周期中协同工作。在一个训练集群中，超过3.5%的GPU时间因启动开销而被浪费。

在这项工作中，我们基于实际生产数据呈现了对LLM训练启动开销的首次深入了解。我们分析了启动成本的组成，量化了其直接影响，并考察了其随作业规模的变化情况。这些见解促使我们设计了Bootseer系统级优化框架，以解决三大主要启动瓶颈：（a）容器镜像加载，（b）运行时依赖安装，（c）模型检查点恢复。为缓解这些瓶颈，Bootseer引入了三种技术：（a）热点块记录与预取，（b）依赖快照，（c）条带化的HDFS-FUSE。Bootseer已在生产环境中部署，并在实际的LLM训练工作负载上进行了评估，结果显示启动开销减少了50%。 

---
# Learning What Matters: Probabilistic Task Selection via Mutual Information for Model Finetuning 

**Title (ZH)**: 学习重要性内容：通过互信息进行概率任务选择的模型微调 

**Authors**: Prateek Chanda, Saral Sureka, Parth Pratim Chatterjee, Krishnateja Killamsetty, Nikhil Shivakumar Nayak, Ganesh Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2507.12612)  

**Abstract**: The performance of finetuned large language models (LLMs) hinges critically on the composition of the training mixture. However, selecting an optimal blend of task datasets remains a largely manual, heuristic driven process, with practitioners often relying on uniform or size based sampling strategies. We introduce TASKPGM, a principled and scalable framework for mixture optimization that selects continuous task proportions by minimizing an energy function over a Markov Random Field (MRF). Task relationships are modeled using behavioral divergences such as Jensen Shannon Divergence and Pointwise Mutual Information computed from the predictive distributions of single task finetuned models. Our method yields a closed form solution under simplex constraints and provably balances representativeness and diversity among tasks. We provide theoretical guarantees, including weak submodularity for budgeted variants, and demonstrate consistent empirical improvements on Llama 2 and Mistral across evaluation suites such as MMLU and BIGBench. Beyond performance, TASKPGM offers interpretable insights into task influence and mixture composition, making it a powerful tool for efficient and robust LLM finetuning. 

**Abstract (ZH)**: 细调大型语言模型的表现关键取决于训练混合物的组成。然而，选择最优的任务数据集混合比例仍然是一个主要依赖手工和启发式驱动的过程，实践者通常依赖均匀或基于大小的取样策略。我们引入了TASKPGM，这是一种原理上和可扩展的混合优化框架，通过在马尔可夫随机场(MRF)上最小化能量函数来选择连续的任务比例。使用行为差异，如从单任务细调模型的预测分布中计算的Jensen Shannon散度和点wise互信息来建模任务关系。该方法在 simples约束下提供了闭合形式的解，并且能够证明在任务的代表性和多样性之间的平衡。我们提供了理论上的保证，包括预算化变体的弱子模性，并在LLama 2和Mistral上的一系列评估套件（如MMLU和BIGBench）中展示了持续的经验改进。除了性能外，TASKPGM还提供了任务影响和混合组成的可解释洞察，使其成为高效和鲁棒大型语言模型细调的强大工具。 

---
# Assay2Mol: large language model-based drug design using BioAssay context 

**Title (ZH)**: Assay2Mol：基于生物活性上下文的大语言模型药物设计 

**Authors**: Yifan Deng, Spencer S. Ericksen, Anthony Gitter  

**Link**: [PDF](https://arxiv.org/pdf/2507.12574)  

**Abstract**: Scientific databases aggregate vast amounts of quantitative data alongside descriptive text. In biochemistry, molecule screening assays evaluate the functional responses of candidate molecules against disease targets. Unstructured text that describes the biological mechanisms through which these targets operate, experimental screening protocols, and other attributes of assays offer rich information for new drug discovery campaigns but has been untapped because of that unstructured format. We present Assay2Mol, a large language model-based workflow that can capitalize on the vast existing biochemical screening assays for early-stage drug discovery. Assay2Mol retrieves existing assay records involving targets similar to the new target and generates candidate molecules using in-context learning with the retrieved assay screening data. Assay2Mol outperforms recent machine learning approaches that generate candidate ligand molecules for target protein structures, while also promoting more synthesizable molecule generation. 

**Abstract (ZH)**: 科学数据库汇集了大量的定量数据和描述性文本。在生物化学中，分子筛选试验评估候选分子针对疾病靶点的功能反应。描述生物机制、实验筛选方案和其他试验属性的非结构化文本为新型药物发现提供了丰富的信息，但由于其非结构化格式，这些信息尚未被充分利用。我们提出了一种基于大型语言模型的工作流Assay2Mol，可以利用已有的大量生物化学筛选试验，以促进早期药物发现。Assay2Mol检索与新靶点相似的目标的现有试验记录，并使用检索到的筛选数据进行上下文学习生成候选分子。Assay2Mol优于针对靶点蛋白结构生成候选配体分子的近期机器学习方法，同时促进了更易于合成的分子生成。 

---
# Is This Just Fantasy? Language Model Representations Reflect Human Judgments of Event Plausibility 

**Title (ZH)**: 这只是幻想吗？语言模型表示反映了事件可能性的人类判断。 

**Authors**: Michael A. Lepori, Jennifer Hu, Ishita Dasgupta, Roma Patel, Thomas Serre, Ellie Pavlick  

**Link**: [PDF](https://arxiv.org/pdf/2507.12553)  

**Abstract**: Language models (LMs) are used for a diverse range of tasks, from question answering to writing fantastical stories. In order to reliably accomplish these tasks, LMs must be able to discern the modal category of a sentence (i.e., whether it describes something that is possible, impossible, completely nonsensical, etc.). However, recent studies have called into question the ability of LMs to categorize sentences according to modality (Michaelov et al., 2025; Kauf et al., 2023). In this work, we identify linear representations that discriminate between modal categories within a variety of LMs, or modal difference vectors. Analysis of modal difference vectors reveals that LMs have access to more reliable modal categorization judgments than previously reported. Furthermore, we find that modal difference vectors emerge in a consistent order as models become more competent (i.e., through training steps, layers, and parameter count). Notably, we find that modal difference vectors identified within LM activations can be used to model fine-grained human categorization behavior. This potentially provides a novel view into how human participants distinguish between modal categories, which we explore by correlating projections along modal difference vectors with human participants' ratings of interpretable features. In summary, we derive new insights into LM modal categorization using techniques from mechanistic interpretability, with the potential to inform our understanding of modal categorization in humans. 

**Abstract (ZH)**: 语言模型通过对多种任务的处理，从回答问题到撰写幻想故事。为了可靠地完成这些任务，语言模型必须能够区分句子的模态类别（即它描述的是可能、不可能、完全不合逻辑等）。然而，最近的研究对语言模型根据模态进行分类的能力提出了质疑（Michaelov et al., 2025；Kauf et al., 2023）。在此工作中，我们识别出能够在不同类型的语言模型中区分模态类别的线性表示，或模态差异向量。模态差异向量的分析表明，语言模型能够进行比之前报告更为可靠的模态分类判断。此外，我们发现随着模型能力的提升（即通过训练步骤、层和参数数量），模态差异向量会出现一致的顺序。值得注意的是，我们发现，在语言模型激活中识别出的模态差异向量可用于模拟精细的人类分类行为。这可能提供了一种新的视角来了解人类如何区分模态类别，我们通过将模态差异向量上的投影与人类参与者对可解释特征的评分进行关联来探索这一视角。总结而言，我们利用机制可解释性技术获得了关于语言模型模态分类的新见解，这有可能帮助我们更好地理解人类的模态分类。 

---
# Scaling Up RL: Unlocking Diverse Reasoning in LLMs via Prolonged Training 

**Title (ZH)**: 扩大RL的应用范围：通过延长训练解锁LLMs的多样化推理 

**Authors**: Mingjie Liu, Shizhe Diao, Jian Hu, Ximing Lu, Xin Dong, Hao Zhang, Alexander Bukharin, Shaokun Zhang, Jiaqi Zeng, Makesh Narsimhan Sreedhar, Gerald Shen, David Mosallanezhad, Di Zhang, Jonas Yang, June Yang, Oleksii Kuchaiev, Guilin Liu, Zhiding Yu, Pavlo Molchanov, Yejin Choi, Jan Kautz, Yi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.12507)  

**Abstract**: Recent advancements in reasoning-focused language models such as OpenAI's O1 and DeepSeek-R1 have shown that scaling test-time computation-through chain-of-thought reasoning and iterative exploration-can yield substantial improvements on complex tasks like mathematics and code generation. These breakthroughs have been driven by large-scale reinforcement learning (RL), particularly when combined with verifiable reward signals that provide objective and grounded supervision. In this report, we investigate the effects of prolonged reinforcement learning on a small language model across a diverse set of reasoning domains. Our work identifies several key ingredients for effective training, including the use of verifiable reward tasks, enhancements to Group Relative Policy Optimization (GRPO), and practical techniques to improve training stability and generalization. We introduce controlled KL regularization, clipping ratio, and periodic reference policy resets as critical components for unlocking long-term performance gains. Our model achieves significant improvements over strong baselines, including +14.7% on math, +13.9% on coding, and +54.8% on logic puzzle tasks. To facilitate continued research, we release our model publicly. 

**Abstract (ZH)**: 近期专注于推理的语言模型（如OpenAI的O1和DeepSeek-R1）的发展表明，通过链式思考推理和迭代探索扩展测试时的计算可以显著提高复杂任务（如数学和代码生成）的表现。这些突破主要得益于大规模强化学习（RL），特别是结合了可验证的奖励信号，这些信号提供了客观和具体的监督。在本报告中，我们研究了长期强化学习对一个小语言模型在多种推理领域的效果。我们的工作识别了有效训练的关键要素，包括使用可验证奖励任务、增强Group Relative Policy Optimization (GRPO)以及提高训练稳定性和泛化性的实用技术。我们引入了受控KL正则化、剪裁比例和周期性参考策略重置作为实现长期性能提升的关键成分。我们的模型在数学、编程和逻辑谜题任务上均取得了显著改进，分别提高了14.7%、13.9%和54.8%。为了促进继续研究，我们已将模型公开发布。 

---
# Kodezi Chronos: A Debugging-First Language Model for Repository-Scale, Memory-Driven Code Understanding 

**Title (ZH)**: Kodezi Chronos：一种面向调试的语言模型，用于仓库规模的记忆驱动代码理解 

**Authors**: Ishraq Khan, Assad Chowdary, Sharoz Haseeb, Urvish Patel  

**Link**: [PDF](https://arxiv.org/pdf/2507.12482)  

**Abstract**: Large Language Models (LLMs) have advanced code generation and software automation, but are fundamentally constrained by limited inference-time context and lack of explicit code structure reasoning. We introduce Kodezi Chronos, a next-generation architecture for autonomous code understanding, debugging, and maintenance, designed to operate across ultra-long contexts comprising entire codebases, histories, and documentation, all without fixed window limits. Kodezi Chronos leverages a multi-level embedding memory engine, combining vector and graph-based indexing with continuous code-aware retrieval. This enables efficient and accurate reasoning over millions of lines of code, supporting repository-scale comprehension, multi-file refactoring, and real-time self-healing actions. Our evaluation introduces a novel Multi Random Retrieval benchmark, specifically tailored to the software engineering domain. Unlike classical retrieval benchmarks, this method requires the model to resolve arbitrarily distant and obfuscated associations across code artifacts, simulating realistic tasks such as variable tracing, dependency migration, and semantic bug localization. Chronos outperforms prior LLMs and code models, demonstrating a 23% improvement in real-world bug detection and reducing debugging cycles by up to 40% compared to traditional sequence-based approaches. By natively interfacing with IDEs and CI/CD workflows, Chronos enables seamless, autonomous software maintenance, elevating code reliability and productivity while reducing manual effort. These results mark a critical advance toward self-sustaining, continuously optimized software ecosystems. 

**Abstract (ZH)**: 下一代自主代码理解、调试和维护架构Kodezi Chronos：跨超长上下文的高效代码推理与软件自运维 

---
# LLM-Powered Quantum Code Transpilation 

**Title (ZH)**: LLM驱动的量子代码转译 

**Authors**: Nazanin Siavash, Armin Moin  

**Link**: [PDF](https://arxiv.org/pdf/2507.12480)  

**Abstract**: There exist various Software Development Kits (SDKs) tailored to different quantum computing platforms. These are known as Quantum SDKs (QSDKs). Examples include but are not limited to Qiskit, Cirq, and PennyLane. However, this diversity presents significant challenges for interoperability and cross-platform development of hybrid quantum-classical software systems. Traditional rule-based transpilers for translating code between QSDKs are time-consuming to design and maintain, requiring deep expertise and rigid mappings in the source and destination code. In this study, we explore the use of Large Language Models (LLMs) as a flexible and automated solution. Leveraging their pretrained knowledge and contextual reasoning capabilities, we position LLMs as programming language-agnostic transpilers capable of converting quantum programs from one QSDK to another while preserving functional equivalence. Our approach eliminates the need for manually defined transformation rules and offers a scalable solution to quantum software portability. This work represents a step toward enabling intelligent, general-purpose transpilation in the quantum computing ecosystem. 

**Abstract (ZH)**: 各种量子计算平台都有专门的软件开发工具包（SDKs），这些工具包被称为量子SDK（QSDKs），例如Qiskit、Cirq和PennyLane。然而，这种多样性为混合量子-经典软件系统的互操作性和跨平台开发带来了显著挑战。传统的基于规则的编译器设计和维护耗时且需要深厚的专业知识和源代码与目标代码之间的严格映射。本研究探索了大型语言模型（LLMs）作为灵活且自动化的解决方案。利用它们预训练的知识和上下文推理能力，我们将LLMs定位为一种编程语言无关的编译器，能够将一种QSDK中的量子程序转换为另一种QSDK，同时保持功能等价性。我们的方法消除了手动定义转换规则的需要，并提供了一种量子软件可移植性的可扩展解决方案。本工作代表了在量子计算生态系统中实现智能、通用编译的新步骤。 

---
