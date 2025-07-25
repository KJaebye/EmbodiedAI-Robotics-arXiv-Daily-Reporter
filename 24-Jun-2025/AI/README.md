# jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval 

**Title (ZH)**: jina-embeddings-v4：通用多模态多语言检索嵌入 

**Authors**: Michael Günther, Saba Sturua, Mohammad Kalim Akram, Isabelle Mohr, Andrei Ungureanu, Sedigheh Eslami, Scott Martens, Bo Wang, Nan Wang, Han Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.18902)  

**Abstract**: We introduce jina-embeddings-v4, a 3.8 billion parameter multimodal embedding model that unifies text and image representations through a novel architecture supporting both single-vector and multi-vector embeddings in the late interaction style. The model incorporates task-specific Low-Rank Adaptation (LoRA) adapters to optimize performance across diverse retrieval scenarios, including query-based information retrieval, cross-modal semantic similarity, and programming code search. Comprehensive evaluations demonstrate that jina-embeddings-v4 achieves state-of-the-art performance on both single- modal and cross-modal retrieval tasks, with particular strength in processing visually rich content such as tables, charts, diagrams, and mixed-media formats. To facilitate evaluation of this capability, we also introduce Jina-VDR, a novel benchmark specifically designed for visually rich image retrieval. 

**Abstract (ZH)**: 我们介绍了jina-embeddings-v4，这是一个包含38亿参数的多模态嵌入模型，通过一种新型架构统一文本和图像表示，并支持在晚期交互风格中的单向量和多向量嵌入。该模型融合了特定任务的低秩适应（LoRA）适配器，以优化在各种检索情景下的性能，包括基于查询的信息检索、跨模态语义相似性和编程代码搜索。综合评估表明，jina-embeddings-v4 在单模态和跨模态检索任务上都达到了最先进的性能，特别擅长处理图表、表格、图表和混合媒体格式等视觉丰富内容。为了方便评估这种能力，我们还引入了Jina-VDR，这是一种专门为视觉丰富图像检索设计的新基准。 

---
# Steering Conceptual Bias via Transformer Latent-Subspace Activation 

**Title (ZH)**: 通过变换器潜空间激活引导概念偏见 

**Authors**: Vansh Sharma, Venkat Raman  

**Link**: [PDF](https://arxiv.org/pdf/2506.18887)  

**Abstract**: This work examines whether activating latent subspaces in language models (LLMs) can steer scientific code generation toward a specific programming language. Five causal LLMs were first evaluated on scientific coding prompts to quantify their baseline bias among four programming languages. A static neuron-attribution method, perturbing the highest activated MLP weight for a C++ or CPP token, proved brittle and exhibited limited generalization across prompt styles and model scales. To address these limitations, a gradient-refined adaptive activation steering framework (G-ACT) was developed: per-prompt activation differences are clustered into a small set of steering directions, and lightweight per-layer probes are trained and refined online to select the appropriate steering vector. In LLaMA-3.2 3B, this approach reliably biases generation towards the CPP language by increasing the average probe classification accuracy by 15% and the early layers (0-6) improving the probe classification accuracy by 61.5% compared to the standard ACT framework. For LLaMA-3.3 70B, where attention-head signals become more diffuse, targeted injections at key layers still improve language selection. Although per-layer probing introduces a modest inference overhead, it remains practical by steering only a subset of layers and enables reproducible model behavior. These results demonstrate a scalable, interpretable and efficient mechanism for concept-level control for practical agentic systems. 

**Abstract (ZH)**: 本研究探讨激活语言模型（LLMs）中的潜在子空间是否可以引导科学研究代码生成向特定编程语言发展。首先评估了五种因果LLMs在科学编程提示上的表现，以量化它们在四种编程语言中的基线偏差。静态神经元归因方法通过扰动C++或CPP标记激活的最高MLP权重，证明了其脆弱性，并且在提示风格和模型规模方面的泛化能力有限。为了解决这些问题，开发了一种梯度细化自适应激活引导框架（G-ACT）：将每种提示的激活差异聚类成一组引导方向，并在线训练和细化轻量级的逐层探针以选择合适的引导向量。在LLaMA-3.2 3B中，通过将探针分类准确率平均提高15%以及早期层（0-6）提高61.5%，该方法可靠地偏置了生成方向偏向CPP语言。对于LLaMA-3.3 70B，其中注意力头信号变得更加弥散，对关键层的针对性注入仍然可以改善语言选择。尽管逐层探针引入了轻微的推理开销，但在仅引导部分层的情况下仍然可行，并且能够使模型行为可重复。这些结果展示了可扩展、可解释和高效的机制，用于现实世界代理系统的概念级控制。 

---
# ConciseHint: Boosting Efficient Reasoning via Continuous Concise Hints during Generation 

**Title (ZH)**: ConciseHint: 通过生成过程中连续简洁提示增强高效推理 

**Authors**: Siao Tang, Xinyin Ma, Gongfan Fang, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18810)  

**Abstract**: Recent advancements in large reasoning models (LRMs) like DeepSeek-R1 and OpenAI o1 series have achieved notable performance enhancements on complex reasoning tasks by scaling up the generation length by Chain-of-Thought (CoT). However, an emerging issue is their inclination to produce excessively verbose reasoning processes, leading to the inefficiency problem. Existing literature on improving efficiency mainly adheres to the before-reasoning paradigms such as prompting and reasoning or fine-tuning and reasoning, but ignores the promising direction of directly encouraging the model to speak concisely by intervening during the generation of reasoning. In order to fill the blank, we propose a framework dubbed ConciseHint, which continuously encourages the reasoning model to speak concisely by injecting the textual hint (manually designed or trained on the concise data) during the token generation of the reasoning process. Besides, ConciseHint is adaptive to the complexity of the query by adaptively adjusting the hint intensity, which ensures it will not undermine model performance. Experiments on the state-of-the-art LRMs, including DeepSeek-R1 and Qwen-3 series, demonstrate that our method can effectively produce concise reasoning processes while maintaining performance well. For instance, we achieve a reduction ratio of 65\% for the reasoning length on GSM8K benchmark with Qwen-3 4B with nearly no accuracy loss. 

**Abstract (ZH)**: Recent advancements in large reasoning models (LRMs) like DeepSeek-R1和OpenAI o1系列通过扩展Chain-of-Thought (CoT)生成长度实现了复杂推理任务上的显著性能提升。然而，一个新兴的问题是这些模型倾向于生成过于冗长的推理过程，导致效率低下。现有提高效率的研究主要集中在推理之前的提示和微调方法，但忽略了直接在推理生成过程中干预以鼓励模型简洁表达的有前途的方向。为了填补这一空白，我们提出了一种名为ConciseHint的框架，该框架在推理过程的 token 生成过程中注入文本提示（手动设计或基于简洁数据训练），持续鼓励模型简洁表达。此外，ConciseHint可以根据查询的复杂度自适应调整提示强度，确保不会损害模型性能。实验表明，该方法可以在保持性能的同时有效生成简洁的推理过程。例如，使用Qwen-3 4B模型在GSM8K基准上实现了65%的推理长度减少，几乎没有任何准确率损失。 

---
# TRIZ Agents: A Multi-Agent LLM Approach for TRIZ-Based Innovation 

**Title (ZH)**: TRIZ智能体：一种基于TRIZ创新的多智能体大语言模型方法 

**Authors**: Kamil Szczepanik, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2506.18783)  

**Abstract**: TRIZ, the Theory of Inventive Problem Solving, is a structured, knowledge-based framework for innovation and abstracting problems to find inventive solutions. However, its application is often limited by the complexity and deep interdisciplinary knowledge required. Advancements in Large Language Models (LLMs) have revealed new possibilities for automating parts of this process. While previous studies have explored single LLMs in TRIZ applications, this paper introduces a multi-agent approach. We propose an LLM-based multi-agent system, called TRIZ agents, each with specialized capabilities and tool access, collaboratively solving inventive problems based on the TRIZ methodology. This multi-agent system leverages agents with various domain expertise to efficiently navigate TRIZ steps. The aim is to model and simulate an inventive process with language agents. We assess the effectiveness of this team of agents in addressing complex innovation challenges based on a selected case study in engineering. We demonstrate the potential of agent collaboration to produce diverse, inventive solutions. This research contributes to the future of AI-driven innovation, showcasing the advantages of decentralized problem-solving in complex ideation tasks. 

**Abstract (ZH)**: TRIZ基多代理系统：基于大型语言模型的发明问题解决方法 

---
# Programming by Backprop: LLMs Acquire Reusable Algorithmic Abstractions During Code Training 

**Title (ZH)**: 基于反向传播的程序构建：大规模语言模型在代码训练过程中习得可重用的算法抽象 

**Authors**: Jonathan Cook, Silvia Sapora, Arash Ahmadian, Akbir Khan, Tim Rocktaschel, Jakob Foerster, Laura Ruis  

**Link**: [PDF](https://arxiv.org/pdf/2506.18777)  

**Abstract**: Training large language models (LLMs) on source code significantly enhances their general-purpose reasoning abilities, but the mechanisms underlying this generalisation are poorly understood. In this paper, we propose Programming by Backprop (PBB) as a potential driver of this effect - teaching a model to evaluate a program for inputs by training on its source code alone, without ever seeing I/O examples. To explore this idea, we finetune LLMs on two sets of programs representing simple maths problems and algorithms: one with source code and I/O examples (w/ IO), the other with source code only (w/o IO). We find evidence that LLMs have some ability to evaluate w/o IO programs for inputs in a range of experimental settings, and make several observations. Firstly, PBB works significantly better when programs are provided as code rather than semantically equivalent language descriptions. Secondly, LLMs can produce outputs for w/o IO programs directly, by implicitly evaluating the program within the forward pass, and more reliably when stepping through the program in-context via chain-of-thought. We further show that PBB leads to more robust evaluation of programs across inputs than training on I/O pairs drawn from a distribution that mirrors naturally occurring data. Our findings suggest a mechanism for enhanced reasoning through code training: it allows LLMs to internalise reusable algorithmic abstractions. Significant scope remains for future work to enable LLMs to more effectively learn from symbolic procedures, and progress in this direction opens other avenues like model alignment by training on formal constitutional principles. 

**Abstract (ZH)**: 训练大规模语言模型（LLMs）在源代码上显著增强了其通用推理能力，但其背后的机制尚不完全清楚。在这篇文章中，我们提出了通过反向传播进行编程（PBB）作为这种效果的潜在驱动因素——通过仅使用源代码训练模型来评估程序，而不曾见过输入/输出示例。为了探索这一想法，我们在两类程序上微调LLMs：一类包括源代码和输入/输出示例（带IO），另一类仅包括源代码（不带IO），代表简单的数学问题和算法。我们发现证据表明LLMs在一系列实验设置中具有评估不带IO程序的能力，并作出了几个观察。首先，当程序以代码形式提供时，PBB的效果显著优于以语义等价的语言描述提供。其次，LLMs可以直接通过隐式在前向传播过程中评估程序来生成不带IO程序的输出，并且通过逐步推理的方式，在上下文中的确更可靠。我们还展示出，与来自具有自然分布的数据的输入/输出配对训练相比，PBB导致在不同输入下的程序评估更具稳健性。我们的发现表明了一种通过代码训练增强推理的机制：它允许LLMs内化可重用的算法抽象。未来工作的显著空间在于使LLMs更有效地从符号程序中学习，并且在此方向上的进展打开了其他途径，如通过在形式宪法原则上训练来进行模型对齐。 

---
# Dual-level Behavioral Consistency for Inter-group and Intra-group Coordination in Multi-Agent Systems 

**Title (ZH)**: 多层级行为一致性在多agent系统中实现组内与组间协调 

**Authors**: Shuocun Yang, Huawen Hu, Enze Shi, Shu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18651)  

**Abstract**: Behavioral diversity in Multi-agent reinforcement learning(MARL) represents an emerging and promising research area. Prior work has largely centered on intra-group behavioral consistency in multi-agent systems, with limited attention given to behavioral consistency in multi-agent grouping scenarios. In this paper, we introduce Dual-Level Behavioral Consistency (DLBC), a novel MARL control method designed to explicitly regulate agent behaviors at both intra-group and inter-group levels. DLBC partitions agents into distinct groups and dynamically modulates behavioral diversity both within and between these groups. By dynamically modulating behavioral diversity within and between these groups, DLBC achieves enhanced division of labor through inter-group consistency, which constrains behavioral strategies across different groups. Simultaneously, intra-group consistency, achieved by aligning behavioral strategies within each group, fosters stronger intra-group cooperation. Crucially, DLBC's direct constraint of agent policy functions ensures its broad applicability across various algorithmic frameworks. Experimental results in various grouping cooperation scenarios demonstrate that DLBC significantly enhances both intra-group cooperative performance and inter-group task specialization, yielding substantial performance improvements. DLBC provides new ideas for behavioral consistency control of multi-intelligent body systems, and its potential for application in more complex tasks and dynamic environments can be further explored in the future. 

**Abstract (ZH)**: 多智能体强化学习中的行为多样性在多个代理层次上的一致性（Dual-Level Behavioral Consistency in Multi-agent Reinforcement Learning） 

---
# AggTruth: Contextual Hallucination Detection using Aggregated Attention Scores in LLMs 

**Title (ZH)**: AggTruth: 使用聚合注意力得分检测上下文错觉在大语言模型中的应用 

**Authors**: Piotr Matys, Jan Eliasz, Konrad Kiełczyński, Mikołaj Langner, Teddy Ferdinan, Jan Kocoń, Przemysław Kazienko  

**Link**: [PDF](https://arxiv.org/pdf/2506.18628)  

**Abstract**: In real-world applications, Large Language Models (LLMs) often hallucinate, even in Retrieval-Augmented Generation (RAG) settings, which poses a significant challenge to their deployment. In this paper, we introduce AggTruth, a method for online detection of contextual hallucinations by analyzing the distribution of internal attention scores in the provided context (passage). Specifically, we propose four different variants of the method, each varying in the aggregation technique used to calculate attention scores. Across all LLMs examined, AggTruth demonstrated stable performance in both same-task and cross-task setups, outperforming the current SOTA in multiple scenarios. Furthermore, we conducted an in-depth analysis of feature selection techniques and examined how the number of selected attention heads impacts detection performance, demonstrating that careful selection of heads is essential to achieve optimal results. 

**Abstract (ZH)**: 在实际应用中，大型语言模型（LLMs）经常产生幻觉，即使在检索增强生成（RAG）设置中也是如此，这对其部署构成了重大挑战。在本文中，我们介绍了AggTruth方法，该方法通过分析提供上下文（段落）中的内部注意力分数分布来在线检测上下文幻觉。具体地，我们提出了四种不同版本的方法，每种方法使用的聚合技术均有所不同。在所有检查的LLM中，AggTruth在同任务和跨任务设置中均表现出稳定的性能，并在多个场景中优于当前SOTA。此外，我们深入分析了特征选择技术，并研究了选择的注意力头数量对检测性能的影响，表明谨慎选择注意力头对于实现最佳结果至关重要。 

---
# Airalogy: AI-empowered universal data digitization for research automation 

**Title (ZH)**: Airalogy: AI赋能的通用数据数字化研究自动化平台 

**Authors**: Zijie Yang, Qiji Zhou, Fang Guo, Sijie Zhang, Yexun Xi, Jinglei Nie, Yudian Zhu, Liping Huang, Chou Wu, Yonghe Xia, Xiaoyu Ma, Yingming Pu, Panzhong Lu, Junshu Pan, Mingtao Chen, Tiannan Guo, Yanmei Dou, Hongyu Chen, Anping Zeng, Jiaxing Huang, Tian Xu, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18586)  

**Abstract**: Research data are the foundation of Artificial Intelligence (AI)-driven science, yet current AI applications remain limited to a few fields with readily available, well-structured, digitized datasets. Achieving comprehensive AI empowerment across multiple disciplines is still out of reach. Present-day research data collection is often fragmented, lacking unified standards, inefficiently managed, and difficult to share. Creating a single platform for standardized data digitization needs to overcome the inherent challenge of balancing between universality (supporting the diverse, ever-evolving needs of various disciplines) and standardization (enforcing consistent formats to fully enable AI). No existing platform accommodates both facets. Building a truly multidisciplinary platform requires integrating scientific domain knowledge with sophisticated computing skills. Researchers often lack the computational expertise to design customized and standardized data recording methods, whereas platform developers rarely grasp the intricate needs of multiple scientific domains. These gaps impede research data standardization and hamper AI-driven progress. In this study, we address these challenges by developing Airalogy (this https URL), the world's first AI- and community-driven platform that balances universality and standardization for digitizing research data across multiple disciplines. Airalogy represents entire research workflows using customizable, standardized data records and offers an advanced AI research copilot for intelligent Q&A, automated data entry, analysis, and research automation. Already deployed in laboratories across all four schools of Westlake University, Airalogy has the potential to accelerate and automate scientific innovation in universities, industry, and the global research community-ultimately benefiting humanity as a whole. 

**Abstract (ZH)**: AI-和社区驱动的跨学科研究数据标准化平台：Airalogy 

---
# T-CPDL: A Temporal Causal Probabilistic Description Logic for Developing Logic-RAG Agent 

**Title (ZH)**: T-CPDL：一种用于开发逻辑-RAG代理的时间因果概率描述逻辑 

**Authors**: Hong Qing Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18559)  

**Abstract**: Large language models excel at generating fluent text but frequently struggle with structured reasoning involving temporal constraints, causal relationships, and probabilistic reasoning. To address these limitations, we propose Temporal Causal Probabilistic Description Logic (T-CPDL), an integrated framework that extends traditional Description Logic with temporal interval operators, explicit causal relationships, and probabilistic annotations. We present two distinct variants of T-CPDL: one capturing qualitative temporal relationships through Allen's interval algebra, and another variant enriched with explicit timestamped causal assertions. Both variants share a unified logical structure, enabling complex reasoning tasks ranging from simple temporal ordering to nuanced probabilistic causation. Empirical evaluations on temporal reasoning and causal inference benchmarks confirm that T-CPDL substantially improves inference accuracy, interpretability, and confidence calibration of language model outputs. By delivering transparent reasoning paths and fine-grained temporal and causal semantics, T-CPDL significantly enhances the capability of language models to support robust, explainable, and trustworthy decision-making. This work also lays the groundwork for developing advanced Logic-Retrieval-Augmented Generation (Logic-RAG) frameworks, potentially boosting the reasoning capabilities and efficiency of knowledge graph-enhanced RAG systems. 

**Abstract (ZH)**: 大型语言模型在生成流畅文本方面表现出色，但经常在涉及时间约束、因果关系和概率推理的结构化推理任务中遇到困难。为了解决这些局限性，我们提出了一种时空因果概率描述逻辑（T-CPDL）集成框架，该框架在传统描述逻辑中扩展了时间区间操作符、显式因果关系和概率注释。我们提出了T-CPDL的两种不同变体：一种通过Allen区间代数捕获定性时间关系，另一种带有明确的时间戳因果断言。两种变体共享统一的逻辑结构，能够支持从简单的时序排序到复杂的概率因果推理的复杂推理任务。在时间推理和因果推断基准测试中的实证评估证实，T-CPDL显著提高了语言模型输出的推理准确性、可解释性和置信度校准。通过提供透明的推理路径和精细粒度的时间和因果语义，T-CPDL极大地提升了语言模型支持稳健、可解释和可信决策的能力。此外，本工作也为开发高级逻辑-检索-增强生成（Logic-RAG）框架奠定了基础，有可能提升知识图谱增强的RAG系统推理能力和效率。 

---
# A Question Bank to Assess AI Inclusivity: Mapping out the Journey from Diversity Errors to Inclusion Excellence 

**Title (ZH)**: 用于评估AI包容性的题库：从多样性错误到包容卓越的旅程映射 

**Authors**: Rifat Ara Shams, Didar Zowghi, Muneera Bano  

**Link**: [PDF](https://arxiv.org/pdf/2506.18538)  

**Abstract**: Ensuring diversity and inclusion (D&I) in artificial intelligence (AI) is crucial for mitigating biases and promoting equitable decision-making. However, existing AI risk assessment frameworks often overlook inclusivity, lacking standardized tools to measure an AI system's alignment with D&I principles. This paper introduces a structured AI inclusivity question bank, a comprehensive set of 253 questions designed to evaluate AI inclusivity across five pillars: Humans, Data, Process, System, and Governance. The development of the question bank involved an iterative, multi-source approach, incorporating insights from literature reviews, D&I guidelines, Responsible AI frameworks, and a simulated user study. The simulated evaluation, conducted with 70 AI-generated personas related to different AI jobs, assessed the question bank's relevance and effectiveness for AI inclusivity across diverse roles and application domains. The findings highlight the importance of integrating D&I principles into AI development workflows and governance structures. The question bank provides an actionable tool for researchers, practitioners, and policymakers to systematically assess and enhance the inclusivity of AI systems, paving the way for more equitable and responsible AI technologies. 

**Abstract (ZH)**: 确保人工智能中的多样性和包容性（D&I）对于缓解偏见和促进公平决策至关重要。然而，现有的人工智能风险评估框架往往忽视了包容性，缺乏衡量人工智能系统与D&I原则一致性的标准化工具。本文介绍了一套结构化的人工智能包容性问题库，包含253个问题，旨在从人类、数据、过程、系统和治理五个支柱方面全面评估人工智能的包容性。问题库的开发采用了迭代的多源方法，整合了文献综述、D&I准则、负责任的人工智能框架以及模拟用户研究的见解。模拟评估使用与不同人工智能岗位相关的70个人工智能生成的角色进行，评估了问题库在多样角色和应用场景中的相关性和有效性。研究结果强调了将D&I原则整合到人工智能开发工作流程和治理结构中的重要性。问题库为研究者、从业者和政策制定者提供了一个可操作的工具，以系统地评估和提升人工智能系统的包容性，铺就更公平和负责任的人工智能技术之路。 

---
# Standard Applicability Judgment and Cross-jurisdictional Reasoning: A RAG-based Framework for Medical Device Compliance 

**Title (ZH)**: 标准适用性判断与跨境推理：基于RAG的医疗器械合规性框架 

**Authors**: Yu Han, Aaron Ceross, Jeroen H.M. Bergmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.18511)  

**Abstract**: Identifying the appropriate regulatory standard applicability remains a critical yet understudied challenge in medical device compliance, frequently necessitating expert interpretation of fragmented and heterogeneous documentation across different jurisdictions. To address this challenge, we introduce a modular AI system that leverages a retrieval-augmented generation (RAG) pipeline to automate standard applicability determination. Given a free-text device description, our system retrieves candidate standards from a curated corpus and uses large language models to infer jurisdiction-specific applicability, classified as Mandatory, Recommended, or Not Applicable, with traceable justifications. We construct an international benchmark dataset of medical device descriptions with expert-annotated standard mappings, and evaluate our system against retrieval-only, zero-shot, and rule-based baselines. The proposed approach attains a classification accuracy of 73% and a Top-5 retrieval recall of 87%, demonstrating its effectiveness in identifying relevant regulatory standards. We introduce the first end-to-end system for standard applicability reasoning, enabling scalable and interpretable AI-supported regulatory science. Notably, our region-aware RAG agent performs cross-jurisdictional reasoning between Chinese and U.S. standards, supporting conflict resolution and applicability justification across regulatory frameworks. 

**Abstract (ZH)**: 确定合适的监管标准适用性仍然是医疗器械合规中的一个重要但研究不足的挑战，经常需要专家对跨不同司法管辖区的碎片化和异质化文件进行解释。为应对这一挑战，我们提出了一种模块化人工智能系统，利用检索增强生成（RAG）管道自动确定标准适用性。给定一个自由文本的设备描述，我们的系统从精心编纂的语料库中检索候选标准，并使用大规模语言模型推断出特定于司法管辖区的适用性分类为强制、推荐或不适用，并提供可追溯的依据。我们构建了一个包含医学设备描述和专家注释标准映射的国际基准数据集，并将我们的系统与检索仅限、零样本和基于规则的基线进行了评估。所提出的方法在分类准确性上达到了73%，Top-5检索召回率为87%，证明了其在识别相关监管标准方面的有效性。我们首次提出了标准适用性推理的端到端系统，使其能够支持可扩展和可解释的人工智能辅助监管科学。值得注意的是，我们的区域感知RAG代理在中英文标准之间进行跨境推理，支持不同监管框架下的冲突解决和适用性解释。 

---
# How Robust is Model Editing after Fine-Tuning? An Empirical Study on Text-to-Image Diffusion Models 

**Title (ZH)**: 微调后模型编辑的鲁棒性：Text-to-Image扩散模型的实证研究 

**Authors**: Feng He, Zhenyang Liu, Marco Valentino, Zhixue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.18428)  

**Abstract**: Model editing offers a low-cost technique to inject or correct a particular behavior in a pre-trained model without extensive retraining, supporting applications such as factual correction and bias mitigation. Despite this common practice, it remains unknown whether edits persist after fine-tuning or whether they are inadvertently reversed. This question has fundamental practical implications. For example, if fine-tuning removes prior edits, it could serve as a defence mechanism against hidden malicious edits. Vice versa, the unintended removal of edits related to bias mitigation could pose serious safety concerns. We systematically investigate the interaction between model editing and fine-tuning in the context of T2I diffusion models, which are known to exhibit biases and generate inappropriate content. Our study spans two T2I model families (Stable Diffusion and FLUX), two sota editing techniques, and three fine-tuning methods (DreamBooth, LoRA, and DoRA). Through an extensive empirical analysis across diverse editing tasks and evaluation metrics, our findings reveal a trend: edits generally fail to persist through fine-tuning, even when fine-tuning is tangential or unrelated to the edits. Notably, we observe that DoRA exhibits the strongest edit reversal effect. At the same time, among editing methods, UCE demonstrates greater robustness, retaining significantly higher efficacy post-fine-tuning compared to ReFACT. These findings highlight a crucial limitation in current editing methodologies, emphasizing the need for more robust techniques to ensure reliable long-term control and alignment of deployed AI systems. These findings have dual implications for AI safety: they suggest that fine-tuning could serve as a remediation mechanism for malicious edits while simultaneously highlighting the need for re-editing after fine-tuning to maintain beneficial safety and alignment properties. 

**Abstract (ZH)**: 模型编辑提供了一种低成本技术，可以在预训练模型中注入或纠正特定行为，而不需要广泛的重新训练，支持例如事实纠正和偏见缓解的应用。尽管这一做法非常常见，但仍不清楚编辑在微调后是否会持续存在，或者是否会被无意中逆转。这个问题具有根本性的实践意义。例如，如果微调消除了先前的编辑，则可以作为一种防御机制，抵御隐藏的恶意编辑。反之，微调无意中消除了与偏见缓解相关的编辑，则可能会带来严重的安全风险。我们系统地研究了T2I扩散模型中模型编辑与微调之间的相互作用，这些模型已知存在偏见并生成不适当的内容。我们的研究涵盖了两种T2I模型家族（Stable Diffusion和FLUX）、两种最先进的编辑技术以及三种微调方法（DreamBooth、LoRA和DoRA）。通过在多种编辑任务和评估指标上的广泛实验分析，我们的发现显示了一种趋势：即使在微调与编辑相关性较弱或无关情况下，编辑通常也难以在微调后持久存在。值得注意的是，我们观察到DoRA表现出最强烈的编辑逆转效果。同时，在编辑方法中，UCE显示出较高的鲁棒性，微调后的效果显著优于ReFACT。这些发现揭示了当前编辑方法的一个关键局限性，强调了需要更稳健的技术，以确保已部署AI系统的可靠长期控制和对齐。这些发现对AI安全具有双重含义：它们表明微调可以作为一种修复机制，以应对恶意编辑，同时强调了在微调后重新编辑的必要性，以保持有益的安全和对齐特性。 

---
# A Large Language Model-based Multi-Agent Framework for Analog Circuits' Sizing Relationships Extraction 

**Title (ZH)**: 基于大型语言模型的多代理框架用于模拟电路尺寸关系提取 

**Authors**: Chengjie Liu, Weiyu Chen, Huiyao Xu, Yuan Du, Jun Yang, Li Du  

**Link**: [PDF](https://arxiv.org/pdf/2506.18424)  

**Abstract**: In the design process of the analog circuit pre-layout phase, device sizing is an important step in determining whether an analog circuit can meet the required performance metrics. Many existing techniques extract the circuit sizing task as a mathematical optimization problem to solve and continuously improve the optimization efficiency from a mathematical perspective. But they ignore the automatic introduction of prior knowledge, fail to achieve effective pruning of the search space, which thereby leads to a considerable compression margin remaining in the search space. To alleviate this problem, we propose a large language model (LLM)-based multi-agent framework for analog circuits' sizing relationships extraction from academic papers. The search space in the sizing process can be effectively pruned based on the sizing relationship extracted by this framework. Eventually, we conducted tests on 3 types of circuits, and the optimization efficiency was improved by $2.32 \sim 26.6 \times$. This work demonstrates that the LLM can effectively prune the search space for analog circuit sizing, providing a new solution for the combination of LLMs and conventional analog circuit design automation methods. 

**Abstract (ZH)**: 基于大语言模型的多agent框架在模拟电路尺寸关系提取中的应用：有效 prune 布局前阶段尺寸优化搜索空间并提升优化效率 

---
# Dynamic Knowledge Exchange and Dual-diversity Review: Concisely Unleashing the Potential of a Multi-Agent Research Team 

**Title (ZH)**: 动态知识交流与双多样性审查：简明释放多剂型研究团队的潜力 

**Authors**: Weilun Yu, Shixiang Tang, Yonggui Huang, Nanqing Dong, Li Fan, Honggang Qi, Wei Liu, Xiaoli Diao, Xi Chen, Wanli Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18348)  

**Abstract**: Scientific progress increasingly relies on effective collaboration among researchers, a dynamic that large language models (LLMs) have only begun to emulate. While recent LLM-based scientist agents show promise in autonomous scientific discovery, they often lack the interactive reasoning and evaluation mechanisms essential to real-world research. We propose IDVSCI (Internal Discussion and Vote SCIentists), a multi-agent framework built on LLMs that incorporates two key innovations: a Dynamic Knowledge Exchange mechanism enabling iterative feedback among agents, and a Dual-Diversity Review paradigm that simulates heterogeneous expert evaluation. These components jointly promote deeper reasoning and the generation of more creative and impactful scientific ideas. To evaluate the effectiveness and generalizability of our approach, we conduct experiments on two datasets: a widely used benchmark in computer science and a new dataset we introduce in the health sciences domain. Results show that IDVSCI consistently achieves the best performance across both datasets, outperforming existing systems such as AI Scientist and VIRSCI. These findings highlight the value of modeling interaction and peer review dynamics in LLM-based autonomous research. 

**Abstract (ZH)**: 基于大语言模型的内部讨论与投票科学家：促进深度推理和创新科学理念生成的多agent框架 

---
# Advanced For-Loop for QML algorithm search 

**Title (ZH)**: Advanced For-Loop for QML Algorithm Search 

**Authors**: FuTe Wong  

**Link**: [PDF](https://arxiv.org/pdf/2506.18260)  

**Abstract**: This paper introduces an advanced framework leveraging Large Language Model-based Multi-Agent Systems (LLMMA) for the automated search and optimization of Quantum Machine Learning (QML) algorithms. Inspired by Google DeepMind's FunSearch, the proposed system works on abstract level to iteratively generates and refines quantum transformations of classical machine learning algorithms (concepts), such as the Multi-Layer Perceptron, forward-forward and backpropagation algorithms. As a proof of concept, this work highlights the potential of agentic frameworks to systematically explore classical machine learning concepts and adapt them for quantum computing, paving the way for efficient and automated development of QML algorithms. Future directions include incorporating planning mechanisms and optimizing strategy in the search space for broader applications in quantum-enhanced machine learning. 

**Abstract (ZH)**: 基于大型语言模型的多智能体系统（LLMMA）的量子机器学习算法自动搜索与优化框架：从经典机器学习概念出发探索量子计算潜力 

---
# The 4th Dimension for Scaling Model Size 

**Title (ZH)**: 第四维：扩展模型规模的新维度 

**Authors**: Ruike Zhu, Hanwen Zhang, Tianyu Shi, Chi Wang, Tianyi Zhou, Zengyi Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.18233)  

**Abstract**: Scaling the size of large language models typically involves three dimensions: depth, width, and the number of parameters. In this work, we explore a fourth dimension, virtual logical depth (VLD), which increases the effective algorithmic depth without changing the overall parameter count by reusing parameters within the model. Although parameter reuse is not a new concept, its potential and characteristics in model scaling have not been thoroughly studied. Through carefully designed controlled experiments, we make the following key discoveries regarding VLD scaling:
VLD scaling forces the knowledge capacity of the model to remain almost constant, with only minor variations.
VLD scaling enables a significant improvement in reasoning capability, provided the scaling method is properly implemented.
The number of parameters correlates with knowledge capacity, but not with reasoning capability. Under certain conditions, it is not necessary to increase the parameter count to enhance reasoning.
These findings are consistent across various model configurations and are likely to be generally valid within the scope of our experiments. 

**Abstract (ZH)**: 增大大型语言模型的规模通常涉及三个维度：深度、宽度和参数量。在本工作中，我们探索了一个新的维度，即虚拟逻辑深度（VLD），通过在模型内部重用参数，增加有效的算法深度而不会改变整体的参数数量。尽管参数重用不是一个新的概念，但其在模型规模扩展中的潜力和特性尚未进行全面研究。通过精心设计的受控实验，我们对于VLD规模扩展做出了以下关键发现：
VLD规模扩展迫使模型的知识容量几乎保持不变，仅有轻微的变化。
当规模扩展方法正确实施时，VLD规模扩展能够显著提高推理能力。
参数量与知识容量相关，但与推理能力无关。在某些条件下，为了增强推理能力，并不需要增加参数数量。
上述发现适用于各种模型配置，并且在我们实验的范围内很可能具有普遍适用性。 

---
# A Conceptual Framework for AI Capability Evaluations 

**Title (ZH)**: 人工智能能力评估的概念框架 

**Authors**: María Victoria Carro, Denise Alejandra Mester, Francisca Gauna Selasco, Luca Nicolás Forziati Gangi, Matheo Sandleris Musa, Lola Ramos Pereyra, Mario Leiva, Juan Gustavo Corvalan, María Vanina Martinez, Gerardo Simari  

**Link**: [PDF](https://arxiv.org/pdf/2506.18213)  

**Abstract**: As AI systems advance and integrate into society, well-designed and transparent evaluations are becoming essential tools in AI governance, informing decisions by providing evidence about system capabilities and risks. Yet there remains a lack of clarity on how to perform these assessments both comprehensively and reliably. To address this gap, we propose a conceptual framework for analyzing AI capability evaluations, offering a structured, descriptive approach that systematizes the analysis of widely used methods and terminology without imposing new taxonomies or rigid formats. This framework supports transparency, comparability, and interpretability across diverse evaluations. It also enables researchers to identify methodological weaknesses, assists practitioners in designing evaluations, and provides policymakers with an accessible tool to scrutinize, compare, and navigate complex evaluation landscapes. 

**Abstract (ZH)**: 随着AI系统的发展和社会整合，设计良好且透明的评估成为AI治理的重要工具，通过提供关于系统能力和风险的证据来指导决策。然而，如何进行全面和可靠地进行这些评估仍然缺乏清晰性。为了解决这一问题，我们提出了一种分析AI能力评估的概念框架，提供了一种结构化、描述性的方法来系统化分析广泛使用的方法和术语，而不强加新的分类或严格的格式。该框架支持评估的透明性、可比性和可解释性，同时也使研究人员能够识别方法论的弱点，帮助实践者设计评估，并为政策制定者提供一个易于使用的工具，以审查、比较和导航复杂的评估景观。 

---
# The Impact of Medication Non-adherence on Adverse Outcomes: Evidence from Schizophrenia Patients via Survival Analysis 

**Title (ZH)**: 药物依从性对不良 outcomes 的影响：基于精神分裂症患者的生存分析证据 

**Authors**: Shahriar Noroozizadeh, Pim Welle, Jeremy C. Weiss, George H. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18187)  

**Abstract**: This study quantifies the association between non-adherence to antipsychotic medications and adverse outcomes in individuals with schizophrenia. We frame the problem using survival analysis, focusing on the time to the earliest of several adverse events (early death, involuntary hospitalization, jail booking). We extend standard causal inference methods (T-learner, S-learner, nearest neighbor matching) to utilize various survival models to estimate individual and average treatment effects, where treatment corresponds to medication non-adherence. Analyses are repeated using different amounts of longitudinal information (3, 6, 9, and 12 months). Using data from Allegheny County in western Pennsylvania, we find strong evidence that non-adherence advances adverse outcomes by approximately 1 to 4 months. Ablation studies confirm that county-provided risk scores adjust for key confounders, as their removal amplifies the estimated effects. Subgroup analyses by medication formulation (injectable vs. oral) and medication type consistently show that non-adherence is associated with earlier adverse events. These findings highlight the clinical importance of adherence in delaying psychiatric crises and show that integrating survival analysis with causal inference tools can yield policy-relevant insights. We caution that although we apply causal inference, we only make associative claims and discuss assumptions needed for causal interpretation. 

**Abstract (ZH)**: 本研究量化了抗精神病药物不依从与精神分裂症患者不良结局之间的关联。我们使用生存分析方法，重点关注最早出现的多种不良事件（早期死亡、强制住院、被捕入狱）的时间。我们扩展了标准因果推断方法（T-学习者、S-学习者、最近邻匹配），利用各种生存模型估计个体和平均治疗效果，其中治疗定义为药物不依从。分析使用不同持续时间的纵向信息（3, 6, 9, 和 12个月）重复进行。借助宾夕法尼亚州西部阿勒格尼县的数据，我们发现药物不依从显著加快了不良结局约1到4个月。消融研究证实，由县政府提供的风险评分能够调整关键混杂因素，因为这些评分的移除会放大估计效果。通过药物剂型（注射 vs. 口服）和药物类型分层分析，一致表明药物不依从与更早发生的不良事件相关。这些发现强调了在精神科危机中提高依从性的临床重要性，并展示了将生存分析与因果推断工具结合使用的政策相关见解。我们谨告诫，尽管我们应用了因果推断方法，但我们仅作出关联性声明，并讨论了进行因果解释所需的前提条件。 

---
# Reasoning about Uncertainty: Do Reasoning Models Know When They Don't Know? 

**Title (ZH)**: 关于不确定性推理：推理模型-know其不知的能力吗？ 

**Authors**: Zhiting Mei, Christina Zhang, Tenny Yin, Justin Lidard, Ola Shorinwa, Anirudha Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2506.18183)  

**Abstract**: Reasoning language models have set state-of-the-art (SOTA) records on many challenging benchmarks, enabled by multi-step reasoning induced using reinforcement learning. However, like previous language models, reasoning models are prone to generating confident, plausible responses that are incorrect (hallucinations). Knowing when and how much to trust these models is critical to the safe deployment of reasoning models in real-world applications. To this end, we explore uncertainty quantification of reasoning models in this work. Specifically, we ask three fundamental questions: First, are reasoning models well-calibrated? Second, does deeper reasoning improve model calibration? Finally, inspired by humans' innate ability to double-check their thought processes to verify the validity of their answers and their confidence, we ask: can reasoning models improve their calibration by explicitly reasoning about their chain-of-thought traces? We introduce introspective uncertainty quantification (UQ) to explore this direction. In extensive evaluations on SOTA reasoning models across a broad range of benchmarks, we find that reasoning models: (i) are typically overconfident, with self-verbalized confidence estimates often greater than 85% particularly for incorrect responses, (ii) become even more overconfident with deeper reasoning, and (iii) can become better calibrated through introspection (e.g., o3-Mini and DeepSeek R1) but not uniformly (e.g., Claude 3.7 Sonnet becomes more poorly calibrated). Lastly, we conclude with important research directions to design necessary UQ benchmarks and improve the calibration of reasoning models. 

**Abstract (ZH)**: 推理语言模型在多步骤推理的驱动下，在许多具有挑战性的基准测试中取得了最新的性能记录，这些推理是由强化学习引起的。然而，就像之前的语言模型一样，推理模型容易生成自信但错误的合理响应（幻觉）。了解何时以及多大程度上信任这些模型对于推理模型在实际应用中的安全部署至关重要。基于此，我们在本工作中探索了推理模型的不确定性量化。具体地，我们提出了三个基本问题：首先，推理模型是否校准良好？其次，更深的推理是否有助于模型校准的改进？最后，借鉴人类校验其思维过程以验证答案的正确性和信心的能力，我们提出：推理模型是否可以通过明证地推理其思维过程轨迹来改善其校准？我们引入了反省不确定性量化（UQ）来探索这一方向。我们在广泛基准测试上的实证评估中发现：（i）推理模型通常过于自信，自我验证的置信度估计值往往超过85%，尤其是在错误的响应中，（ii）随着推理过程的加深，模型的过度自信程度增加，（iii）通过反省（如o3-Mini和DeepSeek R1），模型可以变得更好校准，但并非一概而然（如Claude 3.7 Sonnet变得校准更差）。最后，我们提出了重要的研究方向，以设计必要的不确定性量化基准，从而改善推理模型的校准。 

---
# Chain-of-Memory: Enhancing GUI Agents for Cross-Application Navigation 

**Title (ZH)**: 记忆链：增强跨应用程序导航的GUI代理 

**Authors**: Xinzge Gao, Chuanrui Hu, Bin Chen, Teng Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.18158)  

**Abstract**: Multimodal large language models (MLLMs) are attracting growing attention in the development of Graphical User Interface (GUI) agents. Existing approaches often rely on historical screenshots or actions to implicitly represent the task state. This reliance poses challenges for GUI agents in accurately understanding task states and underscores the absence of effective mechanisms to store critical information in complex and lengthy cross-app tasks. To address these challenges, we propose Chain-of-Memory (CoM), a novel approach for explicitly modeling short-term and long-term memory in GUI agents. CoM achieves this by capturing action descriptions, integrating task-relevant screen information, and maintaining a dedicated memory module to store and manage this information. By leveraging explicit memory representations, CoM enables GUI agents to better understand task states and retain critical historical information persistently. To equip GUI agents with memory management capabilities and evaluate the effectiveness of CoM, we developed the GUI Odyssey-CoM, a dataset comprising 111k screen-action pairs annotated with Chain-of-Memory. Experimental results demonstrate that CoM significantly improves GUI agents' performance in cross-application tasks. Additionally, GUI Odyssey-CoM enables 7B models to achieve memory management capabilities comparable to 72B models. The dataset and code will be open-sourced. 

**Abstract (ZH)**: 多模态大规模语言模型（MLLMs）在图形用户界面（GUI）代理开发中的应用正日益受到关注。现有的方法通常依赖于历史截图或操作来隐式表示任务状态。这种依赖性给GUI代理准确理解任务状态带来了挑战，并突显了在复杂和漫长的跨应用任务中缺乏有效的机制来存储关键信息。为了解决这些挑战，我们提出了Chain-of-Memory（CoM），一种在GUI代理中显式建模短期和长期记忆的新方法。CoM通过捕获操作描述、整合任务相关的屏幕信息，并维护一个专用的记忆模块来存储和管理这些信息，从而实现这一目标。通过利用显式记忆表示，CoM使GUI代理能够更好地理解任务状态并持久地保留关键的历史信息。为了给GUI代理配备记忆管理能力并评估CoM的效果，我们开发了包含111,000个屏幕-操作对的GUI Odyssey-CoM数据集，这些数据对都标注了Chain-of-Memory信息。实验结果表明，CoM显著提高了GUI代理在跨应用任务中的性能。此外，GUI Odyssey-CoM使7B规模的模型能够获得与72B规模模型相当的记忆管理能力。该数据集和代码将开源。 

---
# AI Through the Human Lens: Investigating Cognitive Theories in Machine Psychology 

**Title (ZH)**: AI 透过人类视角：探究机器心理学中的认知理论 

**Authors**: Akash Kundu, Rishika Goswami  

**Link**: [PDF](https://arxiv.org/pdf/2506.18156)  

**Abstract**: We investigate whether Large Language Models (LLMs) exhibit human-like cognitive patterns under four established frameworks from psychology: Thematic Apperception Test (TAT), Framing Bias, Moral Foundations Theory (MFT), and Cognitive Dissonance. We evaluated several proprietary and open-source models using structured prompts and automated scoring. Our findings reveal that these models often produce coherent narratives, show susceptibility to positive framing, exhibit moral judgments aligned with Liberty/Oppression concerns, and demonstrate self-contradictions tempered by extensive rationalization. Such behaviors mirror human cognitive tendencies yet are shaped by their training data and alignment methods. We discuss the implications for AI transparency, ethical deployment, and future work that bridges cognitive psychology and AI safety 

**Abstract (ZH)**: 我们通过心理学中建立的四种框架（主题投射测试TAT、框构偏差、道德基础理论MFT和认知失调）探究大型语言模型（LLMs）是否表现出人类似的心智模式。我们使用结构化提示和自动评分评估了几种 proprietary 和开源模型。研究发现，这些模型经常生成连贯的故事线，容易受到正面框构的影响，展现出与自由/压迫关切相一致的道德判断，并表现出通过广泛理性化来减轻自我矛盾的行为。这些行为反映出人类的心智倾向，但同时也受到其训练数据和对齐方法的影响。我们讨论了这些发现对人工智能透明度、伦理部署以及认知心理学与人工智能安全交叉领域未来工作的含义。 

---
# CoachGPT: A Scaffolding-based Academic Writing Assistant 

**Title (ZH)**: CoachGPT：基于支架式的学术写作辅助工具 

**Authors**: Fumian Chen, Sotheara Veng, Joshua Wilson, Xiaoming Li, Hui Fang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18149)  

**Abstract**: Academic writing skills are crucial for students' success, but can feel overwhelming without proper guidance and practice, particularly when writing in a second language. Traditionally, students ask instructors or search dictionaries, which are not universally accessible. Early writing assistants emerged as rule-based systems that focused on detecting misspellings, subject-verb disagreements, and basic punctuation errors; however, they are inaccurate and lack contextual understanding. Machine learning-based assistants demonstrate a strong ability for language understanding but are expensive to train. Large language models (LLMs) have shown remarkable capabilities in generating responses in natural languages based on given prompts. Still, they have a fundamental limitation in education: they generate essays without teaching, which can have detrimental effects on learning when misused. To address this limitation, we develop CoachGPT, which leverages large language models (LLMs) to assist individuals with limited educational resources and those who prefer self-paced learning in academic writing. CoachGPT is an AI agent-based web application that (1) takes instructions from experienced educators, (2) converts instructions into sub-tasks, and (3) provides real-time feedback and suggestions using large language models. This unique scaffolding structure makes CoachGPT unique among existing writing assistants. Compared to existing writing assistants, CoachGPT provides a more immersive writing experience with personalized feedback and guidance. Our user studies prove the usefulness of CoachGPT and the potential of large language models for academic writing. 

**Abstract (ZH)**: 学术写作技能对于学生的成功至关重要，但缺乏恰当的指导和练习时会感觉令人望而却步，尤其是在使用第二语言写作时。传统上，学生会向教师请教或查阅字典，但这并非普遍可行。早期的写作助手是基于规则的系统，主要侧重于检测拼写错误、主谓一致问题和基本标点错误；然而，这些系统不够准确且缺乏上下文理解。基于机器学习的助手在语言理解方面表现出色，但它们的训练成本高昂。大规模语言模型（LLMs）展示了根据给定提示生成自然语言响应的非凡能力。然而，在教育方面，它们存在根本局限：它们生成论文而不进行教学，这在不当使用时会损害学习效果。为解决这一局限，我们开发了CoachGPT，利用大规模语言模型（LLMs）辅助资源有限的个人及偏好自主学习的学生进行学术写作。CoachGPT 是基于AI代理的网络应用程序，(1) 从经验丰富的教育者那里获取指令，(2) 将指令转换为子任务，并(3) 使用大规模语言模型提供实时反馈和建议。这种独特的支架结构使CoachGPT 在现有的写作助手中独具特色。与现有的写作助手相比，CoachGPT 提供了更加沉浸式的写作体验，并提供个性化反馈和指导。我们的用户研究证明了CoachGPT 的实用性以及大规模语言模型在学术写作中的潜力。 

---
# SE-Merging: A Self-Enhanced Approach for Dynamic Model Merging 

**Title (ZH)**: SE-合并：一种自我增强的动态模型合并方法 

**Authors**: Zijun Chen, Zhanpeng Zhou, Bo Zhang, Weinan Zhang, Xi Sun, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18135)  

**Abstract**: Model merging has gained increasing attention due to its intriguing property: interpolating the parameters of different task-specific fine-tuned models leads to multi-task abilities. However, despite its empirical success, the underlying mechanisms of model merging remain poorly understood. In this work, we delve into the mechanism behind model merging from a representation perspective. Our analysis reveals that model merging achieves multi-task abilities through two key capabilities: i) distinguishing samples from different tasks, and ii) adapting to the corresponding expert model for each sample. These two capabilities allow the merged model to retain task-specific expertise, enabling efficient multi-task adaptation. Building on these insights, we propose \texttt{SE-Merging}, a self-enhanced model merging framework that leverages these two characteristics to dynamically identify the corresponding task for each sample and then adaptively rescales the merging coefficients to further enhance task-specific expertise in the merged model. Notably, \texttt{SE-Merging} achieves dynamic model merging without additional training. Extensive experiments demonstrate that \texttt{SE-Merging} achieves significant performance improvements while remaining compatible with existing model merging techniques. 

**Abstract (ZH)**: 模型融合因其独特的属性而引起了越来越多的关注：插值不同任务特定微调模型的参数可以实现多任务能力。然而，尽管模型融合在实践中取得了成功，其背后的机理仍然知之甚少。在本文中，我们从表示的角度探索了模型融合的机理。我们的分析揭示了模型融合通过两种关键能力实现多任务能力：一是区分不同任务的数据样本，二是根据不同样本适应相应的专家模型。这两种能力使得融合模型能够保留任务特定的专业知识，从而实现高效的多任务适应。基于这些见解，我们提出了一种名为\texttt{SE-Merging}的自我增强模型融合框架，该框架利用这两种特性动态识别每个样本对应的任务，并自适应地重新调整融合系数，以进一步增强融合模型中的任务特定专业知识。值得注意的是，\texttt{SE-Merging}实现了动态模型融合而不需额外训练。大量实验证明，\texttt{SE-Merging}在保持与现有模型融合技术兼容的同时，显著提升了性能。 

---
# Decentralized Consensus Inference-based Hierarchical Reinforcement Learning for Multi-Constrained UAV Pursuit-Evasion Game 

**Title (ZH)**: 基于分布共识推断的多层次强化学习多约束无人机捕逃博弈 

**Authors**: Xiang Yuming, Li Sizhao, Li Rongpeng, Zhao Zhifeng, Zhang Honggang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18126)  

**Abstract**: Multiple quadrotor unmanned aerial vehicle (UAV) systems have garnered widespread research interest and fostered tremendous interesting applications, especially in multi-constrained pursuit-evasion games (MC-PEG). The Cooperative Evasion and Formation Coverage (CEFC) task, where the UAV swarm aims to maximize formation coverage across multiple target zones while collaboratively evading predators, belongs to one of the most challenging issues in MC-PEG, especially under communication-limited constraints. This multifaceted problem, which intertwines responses to obstacles, adversaries, target zones, and formation dynamics, brings up significant high-dimensional complications in locating a solution. In this paper, we propose a novel two-level framework (i.e., Consensus Inference-based Hierarchical Reinforcement Learning (CI-HRL)), which delegates target localization to a high-level policy, while adopting a low-level policy to manage obstacle avoidance, navigation, and formation. Specifically, in the high-level policy, we develop a novel multi-agent reinforcement learning module, Consensus-oriented Multi-Agent Communication (ConsMAC), to enable agents to perceive global information and establish consensus from local states by effectively aggregating neighbor messages. Meanwhile, we leverage an Alternative Training-based Multi-agent proximal policy optimization (AT-M) and policy distillation to accomplish the low-level control. The experimental results, including the high-fidelity software-in-the-loop (SITL) simulations, validate that CI-HRL provides a superior solution with enhanced swarm's collaborative evasion and task completion capabilities. 

**Abstract (ZH)**: 基于共识推理的层次强化学习框架（CI-HRL）：多重约束捕逃游戏中的协同规避与编队覆盖 

---
# Deep Research Agents: A Systematic Examination And Roadmap 

**Title (ZH)**: 深度研究代理：系统审查与 roadmap 

**Authors**: Yuxuan Huang, Yihang Chen, Haozheng Zhang, Kang Li, Meng Fang, Linyi Yang, Xiaoguang Li, Lifeng Shang, Songcen Xu, Jianye Hao, Kun Shao, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18096)  

**Abstract**: The rapid progress of Large Language Models (LLMs) has given rise to a new category of autonomous AI systems, referred to as Deep Research (DR) agents. These agents are designed to tackle complex, multi-turn informational research tasks by leveraging a combination of dynamic reasoning, adaptive long-horizon planning, multi-hop information retrieval, iterative tool use, and the generation of structured analytical reports. In this paper, we conduct a detailed analysis of the foundational technologies and architectural components that constitute Deep Research agents. We begin by reviewing information acquisition strategies, contrasting API-based retrieval methods with browser-based exploration. We then examine modular tool-use frameworks, including code execution, multimodal input processing, and the integration of Model Context Protocols (MCPs) to support extensibility and ecosystem development. To systematize existing approaches, we propose a taxonomy that differentiates between static and dynamic workflows, and we classify agent architectures based on planning strategies and agent composition, including single-agent and multi-agent configurations. We also provide a critical evaluation of current benchmarks, highlighting key limitations such as restricted access to external knowledge, sequential execution inefficiencies, and misalignment between evaluation metrics and the practical objectives of DR agents. Finally, we outline open challenges and promising directions for future research. A curated and continuously updated repository of DR agent research is available at: {this https URL}. 

**Abstract (ZH)**: 大型语言模型的迅速进展催生了一类新的自主AI系统，称为深度研究（DR）代理。这些代理旨在通过动态推理、适应性长期规划、多跳信息检索、迭代工具使用以及生成结构化分析报告来应对复杂、多轮的信息研究任务。本文详细分析了构成深度研究代理的基础技术和架构组件。我们首先回顾了信息获取策略，对比了基于API的信息检索方法与基于浏览器的探索方法。随后，我们探讨了模块化工具使用框架，包括代码执行、多模态输入处理以及模型上下文协议（MCPs）的集成，以支持扩展性和生态系统开发。为了系统化现有的方法，我们提出了一个分类学，区分静态和动态工作流，并根据规划策略和代理组成对代理架构进行分类，包括单代理和多代理配置。最后，我们对当前基准进行了批判性评价，指出了其关键限制，如对外部知识访问的限制、顺序执行的低效性以及评估指标与DR代理实际目标的不一致。我们还提出了未来研究中面临的开放挑战及有前景的研究方向。深度研究代理研究的精心整理并持续更新的资料库可供参考：{this https URL}。 

---
# Weighted Assumption Based Argumentation to reason about ethical principles and actions 

**Title (ZH)**: 基于加权假设计论的伦理原则与行为推理 

**Authors**: Paolo Baldi, Fabio Aurelio D'Asaro, Abeer Dyoub, Francesca Alessandra Lisi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18056)  

**Abstract**: We augment Assumption Based Argumentation (ABA for short) with weighted argumentation. In a nutshell, we assign weights to arguments and then derive the weight of attacks between ABA arguments. We illustrate our proposal through running examples in the field of ethical reasoning, and present an implementation based on Answer Set Programming. 

**Abstract (ZH)**: 基于权重的论证扩展假设论辩学（ABA）：通过在伦理推理领域运行示例来阐述我们的提案，并基于回答集编程进行实现。 

---
# Action Language BC+ 

**Title (ZH)**: 行为语言BC+ 

**Authors**: Joseph Babb, Joohyung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.18044)  

**Abstract**: Action languages are formal models of parts of natural language that are designed to describe effects of actions. Many of these languages can be viewed as high level notations of answer set programs structured to represent transition systems. However, the form of answer set programs considered in the earlier work is quite limited in comparison with the modern Answer Set Programming (ASP) language, which allows several useful constructs for knowledge representation, such as choice rules, aggregates, and abstract constraint atoms. We propose a new action language called BC+, which closes the gap between action languages and the modern ASP language. The main idea is to define the semantics of BC+ in terms of general stable model semantics for propositional formulas, under which many modern ASP language constructs can be identified with shorthands for propositional formulas. Language BC+ turns out to be sufficiently expressive to encompass the best features of other action languages, such as languages B, C, C+, and BC. Computational methods available in ASP solvers are readily applicable to compute BC+, which led to an implementation of the language by extending system cplus2asp. 

**Abstract (ZH)**: 行动语言是一类形式化的自然语言部分，用于描述行动的效果。这些语言中的许多可以被视为结构化以表示转换系统的高级记法的回答集程序。然而，早期工作中考虑的回答集程序的形式与现代回答集编程（ASP）语言相比非常有限，后者允许多种用于知识表示的有用构造，如选择规则、聚合和抽象约束原子。我们提出了一种新的行动语言BC+，它弥合了行动语言与现代ASP语言之间的差距。其主要思想是通过使用命题公式的一般稳定模型语义来定义BC+的语义，这样可以将现代ASP语言中的许多构造识别为命题公式的简称。BC+语言证明具有足够的表达能力，可以涵盖其他行动语言（如B、C、C+和BC）的最佳特征。可用的ASP求解器计算方法可以直接应用于计算BC+，从而通过扩展cplus2asp系统实现了该语言的实现。 

---
# Graphs Meet AI Agents: Taxonomy, Progress, and Future Opportunities 

**Title (ZH)**: 图与AI代理：分类、进展与未来机会 

**Authors**: Yuanchen Bei, Weizhi Zhang, Siwen Wang, Weizhi Chen, Sheng Zhou, Hao Chen, Yong Li, Jiajun Bu, Shirui Pan, Yizhou Yu, Irwin King, Fakhri Karray, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18019)  

**Abstract**: AI agents have experienced a paradigm shift, from early dominance by reinforcement learning (RL) to the rise of agents powered by large language models (LLMs), and now further advancing towards a synergistic fusion of RL and LLM capabilities. This progression has endowed AI agents with increasingly strong abilities. Despite these advances, to accomplish complex real-world tasks, agents are required to plan and execute effectively, maintain reliable memory, and coordinate smoothly with other agents. Achieving these capabilities involves contending with ever-present intricate information, operations, and interactions. In light of this challenge, data structurization can play a promising role by transforming intricate and disorganized data into well-structured forms that agents can more effectively understand and process. In this context, graphs, with their natural advantage in organizing, managing, and harnessing intricate data relationships, present a powerful data paradigm for structurization to support the capabilities demanded by advanced AI agents. To this end, this survey presents a first systematic review of how graphs can empower AI agents. Specifically, we explore the integration of graph techniques with core agent functionalities, highlight notable applications, and identify prospective avenues for future research. By comprehensively surveying this burgeoning intersection, we hope to inspire the development of next-generation AI agents equipped to tackle increasingly sophisticated challenges with graphs. Related resources are collected and continuously updated for the community in the Github link. 

**Abstract (ZH)**: AI代理经历了从强化学习主导到由大规模语言模型驱动的转变，并进一步朝着强化学习和大规模语言模型能力协同融合的方向发展。这一进程赋予了AI代理越来越强的能力。尽管取得了这些进展，但要完成复杂的现实世界任务，代理需要有效规划和执行、维持可靠的记忆，并与其他代理协调一致。实现这些能力涉及处理不断出现的复杂信息、操作和互动。鉴于这一挑战，数据结构化可以通过将复杂且未组织好的数据转化为代理能够更有效地理解和处理的结构化形式，发挥重要作用。在此背景下，凭借其在组织、管理和利用复杂数据关系方面的天然优势，图形呈现了一个强大的数据范式，以支持先进AI代理所要求的能力。为此，本文综述了图形如何赋能AI代理的初步系统性研究。具体而言，我们探讨了图形技术与核心代理功能的整合，强调了重要应用，并指出了未来研究的潜在方向。通过全面综述这一新兴交叉领域，我们希望激发开发能够利用图形应对日益复杂挑战的下一代AI代理的研究。相关资源在Github链接中收集并持续更新。 

---
# medicX-KG: A Knowledge Graph for Pharmacists' Drug Information Needs 

**Title (ZH)**: medicX-KG: 供药师用药信息需求的知识图谱 

**Authors**: Lizzy Farrugia, Lilian M. Azzopardi, Jeremy Debattista, Charlie Abela  

**Link**: [PDF](https://arxiv.org/pdf/2506.17959)  

**Abstract**: The role of pharmacists is evolving from medicine dispensing to delivering comprehensive pharmaceutical services within multidisciplinary healthcare teams. Central to this shift is access to accurate, up-to-date medicinal product information supported by robust data integration. Leveraging artificial intelligence and semantic technologies, Knowledge Graphs (KGs) uncover hidden relationships and enable data-driven decision-making. This paper presents medicX-KG, a pharmacist-oriented knowledge graph supporting clinical and regulatory decisions. It forms the semantic layer of the broader medicX platform, powering predictive and explainable pharmacy services. medicX-KG integrates data from three sources, including, the British National Formulary (BNF), DrugBank, and the Malta Medicines Authority (MMA) that addresses Malta's regulatory landscape and combines European Medicines Agency alignment with partial UK supply dependence. The KG tackles the absence of a unified national drug repository, reducing pharmacists' reliance on fragmented sources. Its design was informed by interviews with practicing pharmacists to ensure real-world applicability. We detail the KG's construction, including data extraction, ontology design, and semantic mapping. Evaluation demonstrates that medicX-KG effectively supports queries about drug availability, interactions, adverse reactions, and therapeutic classes. Limitations, including missing detailed dosage encoding and real-time updates, are discussed alongside directions for future enhancements. 

**Abstract (ZH)**: 药师角色从药物分发向多学科健康Care团队提供全面药物服务的转变：基于Knowledge Graphs的支持与应用 

---
# Evolving Prompts In-Context: An Open-ended, Self-replicating Perspective 

**Title (ZH)**: 演化内省提示：一种开放视角的自复制观点 

**Authors**: Jianyu Wang, Zhiqiang Hu, Lidong Bing  

**Link**: [PDF](https://arxiv.org/pdf/2506.17930)  

**Abstract**: We propose a novel prompt design paradigm that challenges conventional wisdom in large language model (LLM) prompting. While conventional wisdom prioritizes well-crafted instructions and demonstrations for in-context learning (ICL), we show that pruning random demonstrations into seemingly incoherent "gibberish" can remarkably improve performance across diverse tasks. Notably, the "gibberish" always matches or surpasses state-of-the-art automatic prompt optimization techniques, achieving substantial gains regardless of LLM alignment. Nevertheless, discovering an effective pruning strategy is non-trivial, as existing attribution methods and prompt compression algorithms fail to deliver robust results, let alone human intuition. In terms of this, we propose a self-discover prompt optimization framework, PromptQuine, an evolutionary search framework that automatically searches for the pruning strategy by itself using only low-data regimes. Much like the emergent complexity in nature--such as symbiosis and self-organization--arising in response to resource constraints, our framework evolves and refines unconventional yet highly effective prompts by leveraging only the tokens present within the context. We demonstrate its effectiveness across classification, multi-choice question answering, generation and math reasoning tasks across LLMs, while achieving decent runtime efficiency. We hope our findings can guide mechanistic studies on in-context learning, and provide a call to action, to pave the way for more open-ended search algorithms for more effective LLM prompting. 

**Abstract (ZH)**: 我们提出了一种新颖的提示设计范式，挑战了大型语言模型（LLM）提示领域的传统智慧。虽然传统观点强调精心设计的指令和示范以进行上下文内学习（ICL），我们证明将随机示范精简为看似不连贯的“胡言乱语”可以显著提升跨多种任务的表现。值得注意的是，“胡言乱语”总是能够匹敌或超越最先进的自动提示优化技术，在不同LLM对齐的情况下也能取得显著的进步。然而，发现有效的精简策略是不简单的，现有的归因方法和提示压缩算法无法提供稳健的结果，更不用说依靠人类直觉。为此，我们提出了一种自我发现提示优化框架——PromptQuine，这是一种进化搜索框架，仅通过低数据环境自动搜索精简策略。我们的框架通过利用上下文中的词元，进化并精炼出不寻常但极为有效的提示。我们展示了其在分类、多项选择题回答、生成和数学推理任务中的有效性，同时具有不错的运行时效率。我们希望我们的发现能够指导有关上下文内学习的机制性研究，并提供一种行动号召，以铺平更多开放搜索算法的道路，使LLM提示更加高效。 

---
# Learning, Reasoning, Refinement: A Framework for Kahneman's Dual-System Intelligence in GUI Agents 

**Title (ZH)**: 学习、推理、修正：Kahneman的双系统智能在GUI代理中的框架 

**Authors**: Jinjie Wei, Jiyao Liu, Lihao Liu, Ming Hu, Junzhi Ning, Mingcheng Li, Weijie Yin, Junjun He, Xiao Liang, Chao Feng, Dingkang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17913)  

**Abstract**: Graphical User Interface (GUI) agents have made significant progress in automating digital tasks through the utilization of computer vision and language models. Nevertheless, existing agent systems encounter notable limitations. Firstly, they predominantly depend on trial and error decision making rather than progressive reasoning, thereby lacking the capability to learn and adapt from interactive encounters. Secondly, these systems are assessed using overly simplistic single step accuracy metrics, which do not adequately reflect the intricate nature of real world GUI interactions. In this paper, we present CogniGUI, a cognitive framework developed to overcome these limitations by enabling adaptive learning for GUI automation resembling human-like behavior. Inspired by Kahneman's Dual Process Theory, our approach combines two main components: (1) an omni parser engine that conducts immediate hierarchical parsing of GUI elements through quick visual semantic analysis to identify actionable components, and (2) a Group based Relative Policy Optimization (GRPO) grounding agent that assesses multiple interaction paths using a unique relative reward system, promoting minimal and efficient operational routes. This dual-system design facilitates iterative ''exploration learning mastery'' cycles, enabling the agent to enhance its strategies over time based on accumulated experience. Moreover, to assess the generalization and adaptability of agent systems, we introduce ScreenSeek, a comprehensive benchmark that includes multi application navigation, dynamic state transitions, and cross interface coherence, which are often overlooked challenges in current benchmarks. Experimental results demonstrate that CogniGUI surpasses state-of-the-art methods in both the current GUI grounding benchmarks and our newly proposed benchmark. 

**Abstract (ZH)**: 图形用户界面（GUI）代理通过利用计算机视觉和语言模型，在自动化数字任务方面取得了显著进展。然而，现有的代理系统存在明显局限性。首先，它们主要依赖于试探和错误决策，而不是逐步推理，因此缺乏从交互经历中学习和适应的能力。其次，这些系统仅通过过于简单的单一步骤准确性指标进行评估，这未能充分反映现实世界GUI交互的复杂性。本文提出了一种名为CogniGUI的认知框架，旨在通过使GUI自动化能够适应性学习，从而克服这些局限性，使其行为更接近人类。受到Kahneman的双重过程理论启发，我们的方法结合了两个主要组件：（1）一个全能解析引擎，通过快速视觉语义分析对GUI元素进行即时分层解析，以识别可操作组件；（2）基于组的相对策略优化（GRPO）接地代理，使用独特的相对奖励系统评估多条交互路径，促进最小化和高效的操作路径。该双系统设计促进了迭代的“探索学习掌握”循环，使代理能够基于积累的经验增强其策略。此外，为评估代理系统的泛化能力和适应性，我们引入了ScreenSeek这一全面基准，包括多应用导航、动态状态转换和跨界面一致性，这些都是当前基准中常被忽视的挑战。实验结果表明，CogniGUI在现有的GUI接地基准和我们新提出的基准中都超越了最先进的方法。 

---
# Leveraging Large Language Model for Intelligent Log Processing and Autonomous Debugging in Cloud AI Platforms 

**Title (ZH)**: 利用大规模语言模型进行云AI平台中的智能日志处理与自主调试 

**Authors**: Cheng Ji, Huaiying Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.17900)  

**Abstract**: With the increasing complexity and rapid expansion of the scale of AI systems in cloud platforms, the log data generated during system operation is massive, unstructured, and semantically ambiguous, which brings great challenges to fault location and system self-repair. In order to solve this problem, this paper proposes an intelligent log processing and automatic debugging framework based on Large Language Model (LLM), named Intelligent Debugger (LLM-ID). This method is extended on the basis of the existing pre-trained Transformer model, and integrates a multi-stage semantic inference mechanism to realize the context understanding of system logs and the automatic reconstruction of fault chains. Firstly, the system log is dynamically structured, and the unsupervised clustering and embedding mechanism is used to extract the event template and semantic schema. Subsequently, the fine-tuned LLM combined with the multi-round attention mechanism to perform contextual reasoning on the log sequence to generate potential fault assumptions and root cause paths. Furthermore, this paper introduces a reinforcement learning-based policy-guided recovery planner, which is driven by the remediation strategy generated by LLM to support dynamic decision-making and adaptive debugging in the cloud environment. Compared with the existing rule engine or traditional log analysis system, the proposed model has stronger semantic understanding ability, continuous learning ability and heterogeneous environment adaptability. Experiments on the cloud platform log dataset show that LLM-ID improves the fault location accuracy by 16.2%, which is significantly better than the current mainstream methods 

**Abstract (ZH)**: 基于大型语言模型的智能日志处理与自动调试框架：LLM-ID 

---
# Towards Robust Fact-Checking: A Multi-Agent System with Advanced Evidence Retrieval 

**Title (ZH)**: 面向鲁棒事实核查：一种先进的证据检索多Agent系统 

**Authors**: Tam Trinh, Manh Nguyen, Truong-Son Hy  

**Link**: [PDF](https://arxiv.org/pdf/2506.17878)  

**Abstract**: The rapid spread of misinformation in the digital era poses significant challenges to public discourse, necessitating robust and scalable fact-checking solutions. Traditional human-led fact-checking methods, while credible, struggle with the volume and velocity of online content, prompting the integration of automated systems powered by Large Language Models (LLMs). However, existing automated approaches often face limitations, such as handling complex claims, ensuring source credibility, and maintaining transparency. This paper proposes a novel multi-agent system for automated fact-checking that enhances accuracy, efficiency, and explainability. The system comprises four specialized agents: an Input Ingestion Agent for claim decomposition, a Query Generation Agent for formulating targeted subqueries, an Evidence Retrieval Agent for sourcing credible evidence, and a Verdict Prediction Agent for synthesizing veracity judgments with human-interpretable explanations. Evaluated on benchmark datasets (FEVEROUS, HOVER, SciFact), the proposed system achieves a 12.3% improvement in Macro F1-score over baseline methods. The system effectively decomposes complex claims, retrieves reliable evidence from trusted sources, and generates transparent explanations for verification decisions. Our approach contributes to the growing field of automated fact-checking by providing a more accurate, efficient, and transparent verification methodology that aligns with human fact-checking practices while maintaining scalability for real-world applications. Our source code is available at this https URL 

**Abstract (ZH)**: 数字时代错误信息的快速传播对公共话语构成了重大挑战，需要 robust 和可扩展的事实核查解决方案。传统的以人类为主导的事实核查方法虽然可靠，但在处理大量和快速变化的在线内容方面存在困难，因此需要结合基于大型语言模型（LLMs）的自动化系统。然而，现有的自动化方法常常面临处理复杂声明、保证来源可信度和保持透明度等方面的限制。本文提出了一种新型多智能体系统，以提高事实核查的准确性、效率和可解释性。该系统包括四个专门化的智能体：输入摄取智能体负责声明分解、查询生成智能体负责形成针对性的子查询、证据检索智能体负责获取可靠的证据，以及判决预测智能体负责综合真实性的判断并生成可由人类理解的解释。该系统在基准数据集（FEVEROUS、HOVER、SciFact）上的评价中，相对于基线方法取得了12.3%的宏F1分数改善。该系统能够有效分解复杂声明、从可信来源检索可靠证据，并为验证决策生成透明的解释。通过提供一种更准确、更高效、更透明的验证方法，我们的方法在自动化事实核查领域取得了进展，该方法与人类事实核查实践相一致，同时保持了对实际应用的可扩展性。源代码可在以下网址获取。 

---
# Out of Control -- Why Alignment Needs Formal Control Theory (and an Alignment Control Stack) 

**Title (ZH)**: 失控——为什么对齐需要形式化的控制理论（以及一个对齐控制栈） 

**Authors**: Elija Perrier  

**Link**: [PDF](https://arxiv.org/pdf/2506.17846)  

**Abstract**: This position paper argues that formal optimal control theory should be central to AI alignment research, offering a distinct perspective from prevailing AI safety and security approaches. While recent work in AI safety and mechanistic interpretability has advanced formal methods for alignment, they often fall short of the generalisation required of control frameworks for other technologies. There is also a lack of research into how to render different alignment/control protocols interoperable. We argue that by recasting alignment through principles of formal optimal control and framing alignment in terms of hierarchical stack from physical to socio-technical layers according to which controls may be applied we can develop a better understanding of the potential and limitations for controlling frontier models and agentic AI systems. To this end, we introduce an Alignment Control Stack which sets out a hierarchical layered alignment stack, identifying measurement and control characteristics at each layer and how different layers are formally interoperable. We argue that such analysis is also key to the assurances that will be needed by governments and regulators in order to see AI technologies sustainably benefit the community. Our position is that doing so will bridge the well-established and empirically validated methods of optimal control with practical deployment considerations to create a more comprehensive alignment framework, enhancing how we approach safety and reliability for advanced AI systems. 

**Abstract (ZH)**: 形式化最优控制理论在AI对齐研究中的核心地位：从控制框架视角探讨AI安全与 interoperability 的新途径 

---
# Reflective Verbal Reward Design for Pluralistic Alignment 

**Title (ZH)**: 多元共识下的反思性语言奖励设计 

**Authors**: Carter Blair, Kate Larson, Edith Law  

**Link**: [PDF](https://arxiv.org/pdf/2506.17834)  

**Abstract**: AI agents are commonly aligned with "human values" through reinforcement learning from human feedback (RLHF), where a single reward model is learned from aggregated human feedback and used to align an agent's behavior. However, human values are not homogeneous--different people hold distinct and sometimes conflicting values. Aggregating feedback into a single reward model risks disproportionately suppressing minority preferences. To address this, we present a novel reward modeling approach for learning individualized reward models. Our approach uses a language model to guide users through reflective dialogues where they critique agent behavior and construct their preferences. This personalized dialogue history, containing the user's reflections and critiqued examples, is then used as context for another language model that serves as an individualized reward function (what we call a "verbal reward model") for evaluating new trajectories. In studies with 30 participants, our method achieved a 9-12% improvement in accuracy over non-reflective verbal reward models while being more sample efficient than traditional supervised learning methods. 

**Abstract (ZH)**: 基于反思对话的学习个性化奖励模型方法 

---
# Efficient Strategy Synthesis for MDPs via Hierarchical Block Decomposition 

**Title (ZH)**: 基于分层块分解的MDPs高效策略合成 

**Authors**: Alexandros Evangelidis, Gricel Vázquez, Simos Gerasimou  

**Link**: [PDF](https://arxiv.org/pdf/2506.17792)  

**Abstract**: Software-intensive systems, such as software product lines and robotics, utilise Markov decision processes (MDPs) to capture uncertainty and analyse sequential decision-making problems. Despite the usefulness of conventional policy synthesis methods, they fail to scale to large state spaces. Our approach addresses this issue and accelerates policy synthesis in large MDPs by dynamically refining the MDP and iteratively selecting the most fragile MDP regions for refinement. This iterative procedure offers a balance between accuracy and efficiency, as refinement occurs only when necessary. Through a comprehensive empirical evaluation comprising diverse case studies and MDPs up to 1M states, we demonstrate significant performance improvements yielded by our approach compared to the leading probabilistic model checker PRISM (up to 2x), thus offering a very competitive solution for real-world policy synthesis tasks in larger MDPs. 

**Abstract (ZH)**: 基于软件的系统，如软件产品线和机器人技术，利用马尔可夫决策过程（MDPs）来捕捉不确定性并分析 sequential 决策问题。尽管传统的策略合成方法在很多方面都很有用，但它们无法扩展到大规模状态空间。我们的方法解决了这一问题，并通过动态细化 MDP 和迭代选择最脆弱的 MDP 区域进行细化来加速大规模 MDP 的策略合成。这种迭代过程在必要时才进行细化，平衡了准确性和效率。通过涵盖多种案例研究和多达 100 万状态的 MDP 的全面实验证明，与领先的概率模型检测工具 PRISM 相比，我们的方法在性能上取得了显著改进（最高可达 2 倍），从而为更大规模 MDP 中的实际策略合成任务提供了一个极具竞争力的解决方案。 

---
# Bayesian Social Deduction with Graph-Informed Language Models 

**Title (ZH)**: 基于图 informant 语言模型的贝叶斯社会推理 

**Authors**: Shahab Rahimirad, Guven Gergerli, Lucia Romero, Angela Qian, Matthew Lyle Olson, Simon Stepputtis, Joseph Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2506.17788)  

**Abstract**: Social reasoning - inferring unobservable beliefs and intentions from partial observations of other agents - remains a challenging task for large language models (LLMs). We evaluate the limits of current reasoning language models in the social deduction game Avalon and find that while the largest models demonstrate strong performance, they require extensive test-time inference and degrade sharply when distilled to smaller, real-time-capable variants. To address this, we introduce a hybrid reasoning framework that externalizes belief inference to a structured probabilistic model, while using an LLM for language understanding and interaction. Our approach achieves competitive performance with much larger models in Agent-Agent play and, notably, is the first language agent to defeat human players in a controlled study - achieving a 67% win rate and receiving higher qualitative ratings than both reasoning baselines and human teammates. We release code, models, and a dataset to support future work on social reasoning in LLM agents, which can be found at this https URL 

**Abstract (ZH)**: 社会推理——从部分观察到的其他代理的信念和意图中推断不可观测的信念和意图——仍然是大型语言模型（LLMs）面临的一项具有挑战性的任务。我们评估了当前推理语言模型在社会推理游戏Avalon中的限制，并发现虽然最大的模型表现出色，但在被精简为更小、支持实时处理的变体时，推理性能会急剧下降。为了解决这个问题，我们引入了一种混合推理框架，该框架将信念推断外部化到结构化的概率模型中，同时使用LLM进行语言理解和交互。我们的方法在代理-代理对战中实现了与更大模型相当的性能，并且值得注意的是，这是我们第一次在控制研究中让语言代理击败人类玩家——胜率为67%，并且在定性评分方面高于两种推理基线和人类队友。我们发布了支持未来LLM代理社会推理研究的代码、模型和数据集，可在以下链接获取：this https URL。 

---
# AnyMAC: Cascading Flexible Multi-Agent Collaboration via Next-Agent Prediction 

**Title (ZH)**: AnyMAC：基于下一代理预测的级联灵活多代理协作 

**Authors**: Song Wang, Zhen Tan, Zihan Chen, Shuang Zhou, Tianlong Chen, Jundong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.17784)  

**Abstract**: Recent progress in large language model (LLM)-based multi-agent collaboration highlights the power of structured communication in enabling collective intelligence. However, existing methods largely rely on static or graph-based inter-agent topologies, lacking the potential adaptability and flexibility in communication. In this work, we propose a new framework that rethinks multi-agent coordination through a sequential structure rather than a graph structure, offering a significantly larger topology space for multi-agent communication. Our method focuses on two key directions: (1) Next-Agent Prediction, which selects the most suitable agent role at each step, and (2) Next-Context Selection (NCS), which enables each agent to selectively access relevant information from any previous step. Together, these components construct task-adaptive communication pipelines that support both role flexibility and global information flow. Extensive evaluations across multiple benchmarks demonstrate that our approach achieves superior performance while substantially reducing communication overhead. 

**Abstract (ZH)**: 基于大规模语言模型（LLM）的多智能体协作 recent progress 强调结构化通信在促进集体智能方面的能力。然而，现有方法主要依赖静态或图结构的智能体间拓扑，缺乏在通信中适应性和灵活性的潜力。在本工作中，我们提出了一种新的框架，通过顺序结构而不是图结构重新思考多智能体协调，为多智能体通信提供了更大的拓扑空间。我们的方法集中在两个关键方向：（1）下一智能体预测，即在每一步选择最适合的智能体角色，以及（2）下一上下文选择（NCS），使每个智能体能够有选择地访问任一步骤的相关信息。这些组件共同构建了支持角色灵活性和全局信息流动的任务自适应通信管道。在多个基准上的广泛评估表明，我们的方法在显著减少通信开销的同时实现了卓越的性能。 

---
# Beyond Syntax: Action Semantics Learning for App Agents 

**Title (ZH)**: 超越句法：应用代理的动作语义学习 

**Authors**: Bohan Tang, Dezhao Luo, Jingxuan Chen, Shaogang Gong, Jianye Hao, Jun Wang, Kun Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.17697)  

**Abstract**: The advent of Large Language Models (LLMs) enables the rise of App agents that interpret user intent and operate smartphone Apps through actions such as clicking and scrolling. While prompt-based solutions with closed LLM APIs show promising ability, they incur heavy compute costs and external API dependency. Fine-tuning smaller open-source LLMs solves these limitations. However, current fine-tuning methods use a syntax learning paradigm that forces agents to reproduce exactly the ground truth action strings, leading to out-of-distribution (OOD) vulnerability. To fill this gap, we propose Action Semantics Learning (ASL), a novel learning framework, where the learning objective is capturing the semantics of the ground truth actions. Specifically, inspired by the programming language theory, we define the action semantics for App agents as the state transition induced by the action in the user interface. With this insight, ASL employs a novel SEmantic Estimator (SEE) to compute a semantic reward to train the App agents in generating actions aligned with the semantics of ground truth actions, even when the syntactic forms differ. To support the effectiveness of ASL, we theoretically demonstrate the superior robustness of ASL for the OOD problem compared with the existing syntax learning paradigm. Extensive experiments on offline and online smartphone App operation benchmarks show that ASL significantly improves the accuracy and generalisation of App agents over existing methods. 

**Abstract (ZH)**: 大型语言模型的出现使应用代理得以兴起，这些代理通过点击和滚动等操作来解释用户意图并操作智能手机应用。虽然基于提示的解决方案展示了令人信服的能力，但它们产生了巨大的计算成本并依赖外部API。微调较小的开源语言模型解决了这些问题。然而，当前的微调方法使用了语法学习范式，要求代理完全复制地面真实动作字符串，导致了分布外（OOD）的脆弱性。为了解决这一问题，我们提出了动作语义学习（Action Semantics Learning，ASL），这是一种新的学习框架，其学习目标是捕获地面真实动作的语义。具体来说，受到编程语言理论的启发，我们将应用代理的动作语义定义为由动作在用户界面中引发的状态转换。基于这一洞察，ASL 使用了一个新颖的语义估计器（Semantic Estimator，SEE）来计算语义奖励，以训练应用代理生成与地面真实动作语义相匹配的动作，即使两者在语法形式上有所不同。为了支持ASL的有效性，我们从理论上证明了与现有语法学习范式相比，ASL 在分布外问题上具有更强的鲁棒性。在离线和在线智能手机应用操作基准测试中的广泛实验显示，ASL 显著提升了应用代理比现有方法的准确性和泛化能力。 

---
# PhysUniBench: An Undergraduate-Level Physics Reasoning Benchmark for Multimodal Models 

**Title (ZH)**: PhysUniBench: 本科生水平物理推理多模态模型基准 

**Authors**: Lintao Wang, Encheng Su, Jiaqi Liu, Pengze Li, Peng Xia, Jiabei Xiao, Wenlong Zhang, Xinnan Dai, Xi Chen, Yuan Meng, Mingyu Ding, Lei Bai, Wanli Ouyang, Shixiang Tang, Aoran Wang, Xinzhu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.17667)  

**Abstract**: Physics problem-solving is a challenging domain for large AI models, requiring integration of conceptual understanding, mathematical reasoning, and interpretation of physical diagrams. Current evaluation methodologies show notable limitations in capturing the breadth and complexity of undergraduate-level physics, underscoring the need for more rigorous assessments. To this end, we present PhysUniBench, a large-scale multimodal benchmark designed to evaluate and improve the reasoning capabilities of multimodal large language models (MLLMs) specifically on undergraduate-level physics problems. PhysUniBench consists of 3,304 physics questions spanning 8 major sub-disciplines of physics, each accompanied by one visual diagrams. The benchmark includes both open-ended and multiple-choice questions, systematically curated and difficulty-rated through an iterative model-in-the-loop process. The benchmark's construction involved a rigorous multi-stage process, including multiple roll-outs, expert-level evaluation, automated filtering of easily solved problems, and a nuanced difficulty grading system with five levels. Through extensive experiments, we observe that current state-of-the-art models encounter substantial challenges in physics reasoning. For example, GPT-4o mini achieves only about 34.2\% accuracy in the proposed PhysUniBench. These results highlight that current MLLMs struggle with advanced physics reasoning, especially on multi-step problems and those requiring precise diagram interpretation. By providing a broad and rigorous assessment tool, PhysUniBench aims to drive progress in AI for Science, encouraging the development of models with stronger physical reasoning, problem-solving skills, and multimodal understanding. The benchmark and evaluation scripts are available at this https URL. 

**Abstract (ZH)**: 物理学问题求解是大规模AI模型的一个具有挑战性的领域，需要结合概念理解、数学推理和物理图表的解释。当前的评估方法在捕捉本科物理学的广度和复杂性方面显示出明显局限性，强调了更严格评估的需求。为此，我们提出了PhysUniBench，这是一个大规模多模态基准，旨在评估和提高多模态大规模语言模型（MLLMs）在本科物理学问题上的推理能力。PhysUniBench 包含3304道物理题目，涵盖了8个主要的物理子学科，每题配有1个视觉图表。基准测试包括开放性和选择性问题，历经迭代模型循环过程，系统地进行了分类和难度评级。基准测试的构建过程包括多个阶段，包括多次滚动发布、专家级评估、自动过滤易解问题以及五级细致难度分级系统。通过大量实验，我们发现当前最先进的模型在物理学推理方面面临重大挑战。例如，GPT-4o mini 在提出的PhysUniBench上的准确率仅约为34.2%。这些结果表明当前的MLLM在高级物理学推理方面面临困难，尤其是在多步问题和需要精确图表解读的问题上。通过提供一个广泛而严格的评估工具，PhysUniBench旨在推动科学领域中AI的发展，鼓励开发出具有更强物理推理、问题解决能力和多模态理解能力的模型。基准测试和评估脚本可访问此链接。 

---
# Measuring and Augmenting Large Language Models for Solving Capture-the-Flag Challenges 

**Title (ZH)**: 测量与增强大型语言模型以解决Capture-the-Flag挑战 

**Authors**: Zimo Ji, Daoyuan Wu, Wenyuan Jiang, Pingchuan Ma, Zongjie Li, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17644)  

**Abstract**: Capture-the-Flag (CTF) competitions are crucial for cybersecurity education and training. As large language models (LLMs) evolve, there is increasing interest in their ability to automate CTF challenge solving. For example, DARPA has organized the AIxCC competition since 2023 to advance AI-powered automated offense and defense. However, this demands a combination of multiple abilities, from knowledge to reasoning and further to actions. In this paper, we highlight the importance of technical knowledge in solving CTF problems and deliberately construct a focused benchmark, CTFKnow, with 3,992 questions to measure LLMs' performance in this core aspect. Our study offers a focused and innovative measurement of LLMs' capability in understanding CTF knowledge and applying it to solve CTF challenges. Our key findings reveal that while LLMs possess substantial technical knowledge, they falter in accurately applying this knowledge to specific scenarios and adapting their strategies based on feedback from the CTF environment.
Based on insights derived from this measurement study, we propose CTFAgent, a novel LLM-driven framework for advancing CTF problem-solving. CTFAgent introduces two new modules: two-stage Retrieval Augmented Generation (RAG) and interactive Environmental Augmentation, which enhance LLMs' technical knowledge and vulnerability exploitation on CTF, respectively. Our experimental results show that, on two popular CTF datasets, CTFAgent both achieves over 80% performance improvement. Moreover, in the recent picoCTF2024 hosted by CMU, CTFAgent ranked in the top 23.6% of nearly 7,000 participating teams. This reflects the benefit of our measurement study and the potential of our framework in advancing LLMs' capabilities in CTF problem-solving. 

**Abstract (ZH)**: 大型语言模型在Capture-the-Flag (CTF) 挑战自动化中的关键作用：CTFKnow基准与CTFAgent框架研究 

---
# Taming the Untamed: Graph-Based Knowledge Retrieval and Reasoning for MLLMs to Conquer the Unknown 

**Title (ZH)**: 驯服未知：基于图的知识检索与推理以使大模型克服未知领域 

**Authors**: Bowen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17589)  

**Abstract**: The real value of knowledge lies not just in its accumulation, but in its potential to be harnessed effectively to conquer the unknown. Although recent multimodal large language models (MLLMs) exhibit impressing multimodal capabilities, they often fail in rarely encountered domain-specific tasks due to limited relevant knowledge. To explore this, we adopt visual game cognition as a testbed and select Monster Hunter: World as the target to construct a multimodal knowledge graph (MH-MMKG), which incorporates multi-modalities and intricate entity relations. We also design a series of challenging queries based on MH-MMKG to evaluate the models' ability for complex knowledge retrieval and reasoning. Furthermore, we propose a multi-agent retriever that enables a model to autonomously search relevant knowledge without additional training. Experimental results show that our approach significantly enhances the performance of MLLMs, providing a new perspective on multimodal knowledge-augmented reasoning and laying a solid foundation for future research. 

**Abstract (ZH)**: 知识的实际价值不仅在于积累，更在于有效利用以攻克未知。尽管近期的多模态大型语言模型（MLLMs）展现出令人印象深刻的多模态能力，但在罕见遭遇的领域特定任务中往往由于缺乏相关知识而失败。为此，我们以视觉游戏认知作为测试平台，选择《怪物猎人：世界》为目标，构建了一个多模态知识图谱（MH-MMKG），该图谱融合了多模态信息和复杂的实体关系。我们还基于MH-MMKG设计了一系列复杂的查询，以评估模型进行复杂知识检索和推理的能力。此外，我们提出了一种多智能体检索器，使模型能够自主搜索相关知识而无需额外训练。实验结果表明，我们的方法显著提升了MLLMs的表现，为多模态知识增强推理提供了新的视角，并为未来研究奠定了坚实基础。 

---
# Cite Pretrain: Retrieval-Free Knowledge Attribution for Large Language Models 

**Title (ZH)**: Cite Pretrain: 无需检索的知识归属对于大型语言模型 

**Authors**: Yukun Huang, Sanxing Chen, Jian Pei, Manzil Zaheer, Bhuwan Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2506.17585)  

**Abstract**: Trustworthy language models should provide both correct and verifiable answers. While language models can sometimes attribute their outputs to pretraining data, their citations are often unreliable due to hallucination. As a result, current systems insert citations by querying an external retriever at inference time, introducing latency, infrastructure dependence, and vulnerability to retrieval noise. We explore whether LLMs can be made to reliably attribute to the documents seen during (continual) pretraining--without test-time retrieval--by revising the training process. To evaluate this, we release CitePretrainBench, a benchmark that mixes real-world corpora (Wikipedia, Common Crawl, arXiv) with novel, unseen documents and probes both short-form (single fact) and long-form (multi-fact) citation tasks. Our approach follows a two-stage process: (1) continual pretraining to bind facts to persistent document identifiers, and (2) instruction tuning to elicit citation behavior. We find that simple Passive Indexing, which appends an identifier to each document, helps memorize verbatim text but fails on paraphrased or compositional facts. Instead, we propose Active Indexing, which continually pretrains on synthetic QA pairs that (1) restate each fact in diverse compositional forms, and (2) require bidirectional source-to-fact and fact-to-source generation, jointly teaching the model to generate content from a cited source and to attribute its own answers. Experiments with Qwen2.5-7B and 3B show that Active Indexing consistently outperforms Passive Indexing across all tasks and models, with citation precision gains up to 30.2 percent. Our ablation studies reveal that performance continues to improve as we scale the amount of augmented data, showing a clear upward trend even at 16 times the original token count. 

**Abstract (ZH)**: 可信的语言模型应该提供准确且可验证的答案。虽然语言模型有时可以将其输出归因于预训练数据，但由于幻觉，其引用往往是不可靠的。当前系统在推理时通过查询外部检索器插入引用，这引入了延迟、基础设施依赖性和检索噪声的脆弱性。我们探讨是否可以通过修订训练过程使大规模语言模型在（持续）预训练期间可靠地归因于其看到的文档——而不依赖于测试时的检索。为此，我们发布了CitePretrainBench基准，该基准混合了真实世界语料库（维基百科、通用爬虫、arXiv）和新颖的未见文档，并测试了短格式（单一事实）和长格式（多事实）的引用任务。我们的方法遵循两阶段过程：（1）持续预训练将事实绑定到持久化的文档标识符，（2）指令微调以引发引用行为。我们发现简单被动索引，即在每个文档后追加一个标识符，有助于记忆直引文本，但在多义或组成性的事实方面失败。相反，我们提出了主动索引，即持续预训练合成问答对，（1）以多种组成形式重述每个事实，（2）需要双向源到事实和事实到源生成，共同教导模型从引用的源生成内容并归因自己的答案。使用Qwen2.5-7B和3B的实验显示，主动索引在所有任务和模型中均优于被动索引，引文献精度提高多达30.2%。我们的消融研究显示，随着增强数据量的增加，性能持续提升，即使在原始词元数的16倍时，也表现出明显的上升趋势。 

---
# Kaleidoscopic Teaming in Multi Agent Simulations 

**Title (ZH)**: 多智能体仿真中的 Kaleidoscopic 配合模式 

**Authors**: Ninareh Mehrabi, Tharindu Kumarage, Kai-Wei Chang, Aram Galstyan, Rahul Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.17514)  

**Abstract**: Warning: This paper contains content that may be inappropriate or offensive.
AI agents have gained significant recent attention due to their autonomous tool usage capabilities and their integration in various real-world applications. This autonomy poses novel challenges for the safety of such systems, both in single- and multi-agent scenarios. We argue that existing red teaming or safety evaluation frameworks fall short in evaluating safety risks in complex behaviors, thought processes and actions taken by agents. Moreover, they fail to consider risks in multi-agent setups where various vulnerabilities can be exposed when agents engage in complex behaviors and interactions with each other. To address this shortcoming, we introduce the term kaleidoscopic teaming which seeks to capture complex and wide range of vulnerabilities that can happen in agents both in single-agent and multi-agent scenarios. We also present a new kaleidoscopic teaming framework that generates a diverse array of scenarios modeling real-world human societies. Our framework evaluates safety of agents in both single-agent and multi-agent setups. In single-agent setup, an agent is given a scenario that it needs to complete using the tools it has access to. In multi-agent setup, multiple agents either compete against or cooperate together to complete a task in the scenario through which we capture existing safety vulnerabilities in agents. We introduce new in-context optimization techniques that can be used in our kaleidoscopic teaming framework to generate better scenarios for safety analysis. Lastly, we present appropriate metrics that can be used along with our framework to measure safety of agents. Utilizing our kaleidoscopic teaming framework, we identify vulnerabilities in various models with respect to their safety in agentic use-cases. 

**Abstract (ZH)**: 警告：本文包含可能不适合或具有冒犯性的内容。
AI智能体由于其自主工具使用能力和在各种现实世界应用中的集成而近期获得了广泛关注。这种自主性为这些系统带来了新的安全挑战，无论是单智能体还是多智能体场景。我们argue现有的红队或安全评估框架在评估智能体在复杂行为、思维过程和行动中的安全风险方面存在不足。此外，它们未能考虑多智能体配置中的风险，在这种配置中，当智能体进行复杂行为和相互作用时，各种漏洞会被暴露。为了解决这一不足，我们引入了“变彩编队”的概念，旨在捕捉智能体在单智能体和多智能体场景中可能发生的各种复杂和广泛的漏洞。我们也提出了一种新的变彩编队框架，用于生成模拟现实人类社会的多样化场景。该框架评估智能体在单智能体和多智能体配置中的安全性。在单智能体配置中，智能体需要使用其可访问的工具完成一个场景。在多智能体配置中，多个智能体要么相互竞争，要么合作完成场景中的任务，从而捕捉智能体现有的安全漏洞。我们介绍了可以在我们的变彩编队框架中使用的新的上下文优化技术，以生成更好的场景进行安全性分析。最后，我们提出了适当的度量标准，用于与我们的框架结合以衡量智能体的安全性。利用我们的变彩编队框架，我们针对智能体使用场景中的安全性识别了各种模型的漏洞。 

---
# From Unstructured Communication to Intelligent RAG: Multi-Agent Automation for Supply Chain Knowledge Bases 

**Title (ZH)**: 从无结构通信到智能RAG：供应链知识库的多代理自动化 

**Authors**: Yao Zhang, Zaixi Shang, Silpan Patel, Mikel Zuniga  

**Link**: [PDF](https://arxiv.org/pdf/2506.17484)  

**Abstract**: Supply chain operations generate vast amounts of operational data; however, critical knowledge such as system usage practices, troubleshooting workflows, and resolution techniques often remains buried within unstructured communications like support tickets, emails, and chat logs. While RAG systems aim to leverage such communications as a knowledge base, their effectiveness is limited by raw data challenges: support tickets are typically noisy, inconsistent, and incomplete, making direct retrieval suboptimal. Unlike existing RAG approaches that focus on runtime optimization, we introduce a novel offline-first methodology that transforms these communications into a structured knowledge base. Our key innovation is a LLMs-based multi-agent system orchestrating three specialized agents: Category Discovery for taxonomy creation, Categorization for ticket grouping, and Knowledge Synthesis for article generation. Applying our methodology to real-world support tickets with resolution notes and comments, our system creates a compact knowledge base - reducing total volume to just 3.4% of original ticket data while improving quality. Experiments demonstrate that our prebuilt knowledge base in RAG systems significantly outperforms traditional RAG implementations (48.74% vs. 38.60% helpful answers) and achieves a 77.4% reduction in unhelpful responses. By automating institutional knowledge capture that typically remains siloed in experts' heads, our solution translates to substantial operational efficiency: reducing support workload, accelerating resolution times, and creating self-improving systems that automatically resolve approximately 50% of future supply chain tickets. Our approach addresses a key gap in knowledge management by transforming transient communications into structured, reusable knowledge through intelligent offline processing rather than latency-inducing runtime architectures. 

**Abstract (ZH)**: 供应链运营生成大量操作数据；然而，诸如系统使用实践、故障排除工作流程和解决方案技术等关键知识往往埋藏在支持工单、电子邮件和聊天日志等非结构化通信中。虽然RAG系统旨在利用此类通信作为知识库，但它们的有效性受限于原始数据挑战：支持工单通常嘈杂、不一致且不完整，直接检索效果不佳。不同于现有RAG方法侧重于运行时优化，我们提出了一种新型的离线优先方法，将这些通信转换为结构化知识库。我们的关键创新是一种基于LLMs的多代理系统，协调三个专门代理：分类发现用于分类目录创建，分类用于票务分组，知识综合用于文章生成。将我们的方法应用于包含解决注释和支持评论的真实世界支持工单，我们的系统创建了一个紧凑的知识库——数据总量仅占原始工单数据的3.4%，同时提高了质量。实验表明，我们的预构建知识库在RAG系统中的表现显著优于传统的RAG实现（48.74%对比38.60%有帮助的回答），并且减少了77.4%的无用回答。通过自动化机构知识捕获，通常局限于专家头脑中，我们的解决方案提高了运营效率：减少了支持工作量，加速了解决时间，并创建了能够自动解决未来约50%供应链工单的自改进系统。我们的方法通过智能离线处理将瞬态通信转化为可重复使用的结构化知识，填补了知识管理中的关键空白，而不是依赖于引入延迟的运行时架构。 

---
# OmniReflect: Discovering Transferable Constitutions for LLM agents via Neuro-Symbolic Reflections 

**Title (ZH)**: OmniReflect: 通过神经符号反思发现适用于LLM代理的可转移构成要素 

**Authors**: Manasa Bharadwaj, Nikhil Verma, Kevin Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2506.17449)  

**Abstract**: Efforts to improve Large Language Model (LLM) agent performance on complex tasks have largely focused on fine-tuning and iterative self-correction. However, these approaches often lack generalizable mechanisms for longterm learning and remain inefficient in dynamic environments. We introduce OmniReflect, a hierarchical, reflection-driven framework that constructs a constitution, a compact set of guiding principles distilled from task experiences, to enhance the effectiveness and efficiency of an LLM agent. OmniReflect operates in two modes: Self-sustaining, where a single agent periodically curates its own reflections during task execution, and Co-operative, where a Meta-advisor derives a constitution from a small calibration set to guide another agent. To construct these constitutional principles, we employ Neural, Symbolic, and NeuroSymbolic techniques, offering a balance between contextual adaptability and computational efficiency. Empirical results averaged across models show major improvements in task success, with absolute gains of +10.3% on ALFWorld, +23.8% on BabyAI, and +8.3% on PDDL in the Self-sustaining mode. Similar gains are seen in the Co-operative mode, where a lightweight Qwen3-4B ReAct agent outperforms all Reflexion baselines on BabyAI. These findings highlight the robustness and effectiveness of OmniReflect across environments and backbones. 

**Abstract (ZH)**: 面向复杂任务的大型语言模型代理性能改进努力主要集中在微调和迭代自我修正上。然而，这些方法往往缺乏可泛化的长期学习机制，并且在动态环境中效率低下。我们 introduces OmniReflect，一种层次化、基于反思的框架，通过从任务经验中提炼出一套紧凑的指导原则来构建宪法，以提高大型语言模型代理的有效性和效率。OmniReflect 运行在两种模式下：自我维持模式，其中单个代理在执行任务期间定期整理自己的反思；合作模式，其中元顾问从校准集中提取宪法以指导另一个代理。为了构建这些宪法原则，我们采用了神经、符号和神经符号技术，提供了上下文适应性和计算效率之间的平衡。在模型上的实验证明了显著的任务成功率改进，在自我维持模式下，ALFWorld 增加了 10.3%，BabyAI 增加了 23.8%，PDDL 增加了 8.3%。在合作模式下，一个轻量级的 Qwen3-4B ReAct 代理在 BabyAI 上的表现优于所有反射基线。这些发现突显了 OmniReflect 在不同环境和基础架构中的稳健性和有效性。 

---
# Keeping Medical AI Healthy: A Review of Detection and Correction Methods for System Degradation 

**Title (ZH)**: 保持医疗AI健康：系统退化检测与修正方法综述 

**Authors**: Hao Guan, David Bates, Li Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.17442)  

**Abstract**: Artificial intelligence (AI) is increasingly integrated into modern healthcare, offering powerful support for clinical decision-making. However, in real-world settings, AI systems may experience performance degradation over time, due to factors such as shifting data distributions, changes in patient characteristics, evolving clinical protocols, and variations in data quality. These factors can compromise model reliability, posing safety concerns and increasing the likelihood of inaccurate predictions or adverse outcomes. This review presents a forward-looking perspective on monitoring and maintaining the "health" of AI systems in healthcare. We highlight the urgent need for continuous performance monitoring, early degradation detection, and effective self-correction mechanisms. The paper begins by reviewing common causes of performance degradation at both data and model levels. We then summarize key techniques for detecting data and model drift, followed by an in-depth look at root cause analysis. Correction strategies are further reviewed, ranging from model retraining to test-time adaptation. Our survey spans both traditional machine learning models and state-of-the-art large language models (LLMs), offering insights into their strengths and limitations. Finally, we discuss ongoing technical challenges and propose future research directions. This work aims to guide the development of reliable, robust medical AI systems capable of sustaining safe, long-term deployment in dynamic clinical settings. 

**Abstract (ZH)**: 人工智能（AI）越来越多地融入现代医疗，为临床决策提供强大的支持。然而，在实际应用中，由于数据分布变化、患者特征变化、临床流程演变以及数据质量差异等因素，AI系统可能会随着时间的推移出现性能下降。这些因素会削弱模型的可靠性，引发行安全风险并增加不准确预测或不良后果的可能性。本文从前瞻性的角度探讨了在医疗保健中监控和维护AI系统“健康状况”的必要性。我们强调了持续性能监控、早期性能下降检测以及有效自我修正机制的迫切需求。文章首先回顾了数据和模型层面常见性能下降的原因。然后总结了检测数据和模型漂移的关键技术，并深入探讨了根本原因分析。进一步审查了从模型重训练到测试时适应的各种纠正策略。本文涵盖了传统机器学习模型和最新大型语言模型（LLMs），提供了它们优缺点的见解。最后，讨论了当前的技术挑战并提出了未来的研究方向。本工作旨在指导开发可靠的、健壮的医疗AI系统，使其能够在动态临床环境中安全、长期部署。 

---
# Resource Rational Contractualism Should Guide AI Alignment 

**Title (ZH)**: 资源理性契约论应指导AI对齐 

**Authors**: Sydney Levine, Matija Franklin, Tan Zhi-Xuan, Secil Yanik Guyot, Lionel Wong, Daniel Kilov, Yejin Choi, Joshua B. Tenenbaum, Noah Goodman, Seth Lazar, Iason Gabriel  

**Link**: [PDF](https://arxiv.org/pdf/2506.17434)  

**Abstract**: AI systems will soon have to navigate human environments and make decisions that affect people and other AI agents whose goals and values diverge. Contractualist alignment proposes grounding those decisions in agreements that diverse stakeholders would endorse under the right conditions, yet securing such agreement at scale remains costly and slow -- even for advanced AI. We therefore propose Resource-Rational Contractualism (RRC): a framework where AI systems approximate the agreements rational parties would form by drawing on a toolbox of normatively-grounded, cognitively-inspired heuristics that trade effort for accuracy. An RRC-aligned agent would not only operate efficiently, but also be equipped to dynamically adapt to and interpret the ever-changing human social world. 

**Abstract (ZH)**: 资源理性契约主义：一种AI系统框架 

---
# Individual Causal Inference with Structural Causal Model 

**Title (ZH)**: 基于结构因果模型的个体因果推断 

**Authors**: Daniel T. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17300)  

**Abstract**: Individual causal inference (ICI) uses causal inference methods to understand and predict the effects of interventions on individuals, considering their specific characteristics / facts. It aims to estimate individual causal effect (ICE), which varies across individuals. Estimating ICE can be challenging due to the limited data available for individuals, and the fact that most causal inference methods are population-based. Structural Causal Model (SCM) is fundamentally population-based. Therefore, causal discovery (structural learning and parameter learning), association queries and intervention queries are all naturally population-based. However, exogenous variables (U) in SCM can encode individual variations and thus provide the mechanism for individualized population per specific individual characteristics / facts. Based on this, we propose ICI with SCM as a "rung 3" causal inference, because it involves "imagining" what would be the causal effect of a hypothetical intervention on an individual, given the individual's observed characteristics / facts. Specifically, we propose the indiv-operator, indiv(W), to formalize/represent the population individualization process, and the individual causal query, P(Y | indiv(W), do(X), Z), to formalize/represent ICI. We show and argue that ICI with SCM is inference on individual alternatives (possible), not individual counterfactuals (non-actual). 

**Abstract (ZH)**: 个体因果推断（ICI）使用因果推断方法来理解并预测干预措施对个体的影响，考虑到个体的具体特征/事实。它旨在估计个体因果效应（ICE），而这在不同个体间会有所不同。由于可用的个体数据有限且大多数因果推断方法基于总体，估计ICE具有挑战性。结构因果模型（SCM）本质上是基于总体的，因此因果发现（结构学习和参数学习）、关联查询和干预查询都是基于总体的。然而，SCM中的外生变量（U）可以编码个体差异，从而为特定个体特征/事实下的个体化总体提供机制。基于此，我们提议使用SCM进行个体因果推断（ICI）作为一种“第三级”因果推断方法，因为它涉及“设想”给定个体观察到的特征/事实时，假设干预措施的因果效应。具体地，我们提出了个体算子indiv(W)来形式化/表示人口个体化过程，以及个体因果查询P(Y | indiv(W), do(X), Z)来形式化/表示ICI。我们展示了并论证了使用SCM进行个体因果推断是一种针对个体替代（潜在的）而非个体反事实（非实际的）的推断。 

---
# Evaluating Generalization and Representation Stability in Small LMs via Prompting 

**Title (ZH)**: 通过提示评估小型语言模型的泛化能力和表示稳定性 

**Authors**: Rahul Raja, Arpita Vats  

**Link**: [PDF](https://arxiv.org/pdf/2506.17289)  

**Abstract**: We investigate the generalization capabilities of small language models under two popular adaptation paradigms: few-shot prompting and supervised fine-tuning. While prompting is often favored for its parameter efficiency and flexibility, it remains unclear how robust this approach is in low-resource settings and under distributional shifts. This paper presents a comparative study of prompting and fine-tuning across task formats, prompt styles, and model scales, with a focus on their behavior in both in-distribution and out-of-distribution (OOD) settings.
Beyond accuracy, we analyze the internal representations learned by each approach to assess the stability and abstraction of task-specific features. Our findings highlight critical differences in how small models internalize and generalize knowledge under different adaptation strategies. This work offers practical guidance for model selection in low-data regimes and contributes empirical insight into the ongoing debate over prompting versus fine-tuning. Code for the experiments is available at the following 

**Abstract (ZH)**: 我们研究了小型语言模型在两种流行的适应 paradigm：少样本提示和监督微调下的泛化能力。虽然提示因参数效率和灵活性而常被青睐，但在资源稀缺环境下以及分布变化时，这种做法的鲁棒性仍不清楚。本文在任务格式、提示风格和模型规模上对提示和微调进行了比较研究，重点关注它们在既定分布和出分布（OOD）设置下的行为。

除了准确性之外，我们还分析了每种方法学习到的内部表示，以评估任务特定特征的稳定性和抽象程度。我们的研究结果突显了在不同适应策略下小模型内部化和泛化知识的关键差异。这项工作为低数据环境下的模型选择提供了实践指导，并为提示与微调之间的持续争论提供了实证见解。实验代码可在以下地址获取。 

---
# Vision as a Dialect: Unifying Visual Understanding and Generation via Text-Aligned Representations 

**Title (ZH)**: 视觉作为一种方言：通过文本对齐表示统一视觉理解与生成 

**Authors**: Jiaming Han, Hao Chen, Yang Zhao, Hanyu Wang, Qi Zhao, Ziyan Yang, Hao He, Xiangyu Yue, Lu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18898)  

**Abstract**: This paper presents a multimodal framework that attempts to unify visual understanding and generation within a shared discrete semantic representation. At its core is the Text-Aligned Tokenizer (TA-Tok), which converts images into discrete tokens using a text-aligned codebook projected from a large language model's (LLM) vocabulary. By integrating vision and text into a unified space with an expanded vocabulary, our multimodal LLM, Tar, enables cross-modal input and output through a shared interface, without the need for modality-specific designs. Additionally, we propose scale-adaptive encoding and decoding to balance efficiency and visual detail, along with a generative de-tokenizer to produce high-fidelity visual outputs. To address diverse decoding needs, we utilize two complementary de-tokenizers: a fast autoregressive model and a diffusion-based model. To enhance modality fusion, we investigate advanced pre-training tasks, demonstrating improvements in both visual understanding and generation. Experiments across benchmarks show that Tar matches or surpasses existing multimodal LLM methods, achieving faster convergence and greater training efficiency. Code, models, and data are available at this https URL 

**Abstract (ZH)**: 本文提出了一种多模态框架，尝试在共享的离散语义表示中统一视觉理解和生成。其核心是文本对齐分词器（TA-Tok），它使用大语言模型（LLM）词汇表投影得到的文本对齐码本将图像转换为离散词元。通过将视觉和文本整合到一个扩大的词汇表统一空间中，我们的多模态LLM Tar能够通过共享接口进行跨模态输入和输出，无需特定模态的设计。此外，我们提出了自适应编码和解码以平衡效率和视觉细节，并提出了一种生成性反分词器以生成高保真视觉输出。为了满足多样化的解码需求，我们利用了两种互补的反分词器：快速自回归模型和基于扩散的模型。为了增强模态融合，我们研究了先进的预训练任务，证明了在视觉理解和生成方面的改进。跨基准实验表明，Tar与现有的多模态LLM方法相当或超越，实现了更快的收敛速度和更高的训练效率。代码、模型和数据可在以下链接获取。 

---
# MinD: Unified Visual Imagination and Control via Hierarchical World Models 

**Title (ZH)**: MinD: 统一的层级世界观下的视觉想象与控制 

**Authors**: Xiaowei Chi, Kuangzhi Ge, Jiaming Liu, Siyuan Zhou, Peidong Jia, Zichen He, Yuzhen Liu, Tingguang Li, Lei Han, Sirui Han, Shanghang Zhang, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.18897)  

**Abstract**: Video generation models (VGMs) offer a promising pathway for unified world modeling in robotics by integrating simulation, prediction, and manipulation. However, their practical application remains limited due to (1) slowgeneration speed, which limits real-time interaction, and (2) poor consistency between imagined videos and executable actions. To address these challenges, we propose Manipulate in Dream (MinD), a hierarchical diffusion-based world model framework that employs a dual-system design for vision-language manipulation. MinD executes VGM at low frequencies to extract video prediction features, while leveraging a high-frequency diffusion policy for real-time interaction. This architecture enables low-latency, closed-loop control in manipulation with coherent visual guidance. To better coordinate the two systems, we introduce a video-action diffusion matching module (DiffMatcher), with a novel co-training strategy that uses separate schedulers for each diffusion model. Specifically, we introduce a diffusion-forcing mechanism to DiffMatcher that aligns their intermediate representations during training, helping the fast action model better understand video-based predictions. Beyond manipulation, MinD also functions as a world simulator, reliably predicting task success or failure in latent space before execution. Trustworthy analysis further shows that VGMs can preemptively evaluate task feasibility and mitigate risks. Extensive experiments across multiple benchmarks demonstrate that MinD achieves state-of-the-art manipulation (63%+) in RL-Bench, advancing the frontier of unified world modeling in robotics. 

**Abstract (ZH)**: 基于生成模型的级联控制框架：从梦境中操控 

---
# OMEGA: Can LLMs Reason Outside the Box in Math? Evaluating Exploratory, Compositional, and Transformative Generalization 

**Title (ZH)**: OMEGA：大型语言模型能在数学中进行框外推理吗？探究性、组合性和转换性泛化的评估 

**Authors**: Yiyou Sun, Shawn Hu, Georgia Zhou, Ken Zheng, Hannaneh Hajishirzi, Nouha Dziri, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.18880)  

**Abstract**: Recent large-scale language models (LLMs) with long Chain-of-Thought reasoning-such as DeepSeek-R1-have achieved impressive results on Olympiad-level mathematics benchmarks. However, they often rely on a narrow set of strategies and struggle with problems that require a novel way of thinking. To systematically investigate these limitations, we introduce OMEGA-Out-of-distribution Math Problems Evaluation with 3 Generalization Axes-a controlled yet diverse benchmark designed to evaluate three axes of out-of-distribution generalization, inspired by Boden's typology of creativity: (1) Exploratory-applying known problem solving skills to more complex instances within the same problem domain; (2) Compositional-combining distinct reasoning skills, previously learned in isolation, to solve novel problems that require integrating these skills in new and coherent ways; and (3) Transformative-adopting novel, often unconventional strategies by moving beyond familiar approaches to solve problems more effectively. OMEGA consists of programmatically generated training-test pairs derived from templated problem generators across geometry, number theory, algebra, combinatorics, logic, and puzzles, with solutions verified using symbolic, numerical, or graphical methods. We evaluate frontier (or top-tier) LLMs and observe sharp performance degradation as problem complexity increases. Moreover, we fine-tune the Qwen-series models across all generalization settings and observe notable improvements in exploratory generalization, while compositional generalization remains limited and transformative reasoning shows little to no improvement. By isolating and quantifying these fine-grained failures, OMEGA lays the groundwork for advancing LLMs toward genuine mathematical creativity beyond mechanical proficiency. 

**Abstract (ZH)**: Recent大规模语言模型（LLMs）具备长链推理能力——如DeepSeek-R1——在奥林匹克级数学基准测试中取得了显著成果。然而，它们往往依赖于狭窄的战略集并在需要新型思维方式的问题上挣扎。为系统地探讨这些局限性，我们引入了OMEGA：异域数学问题评估——一个控制多样但全面的基准，旨在评估由布登创造力类型学启发的三个异域泛化轴：（1）探索性：将已知问题解决技能应用于同一问题域内的更复杂实例；（2）组合性：结合此前孤立学习的不同推理技能，以解决需要以新颖且连贯的方式整合这些技能的新问题；（3）变革性：采用新颖且往往是非传统的策略，超越熟悉的解决方法以更有效地解决问题。OMEGA包含从几何、数论、代数、组合数学、逻辑和谜题等领域的模板问题生成器生成的训练-测试对，并通过符号、数值或图形方法验证解决方案。我们评估了前沿（或顶级）LLM并在问题复杂度增加时观察到性能急剧下降。此外，我们针对所有泛化设置微调Qwen系列模型，并观察到在探索性泛化方面取得了显著改进，而组合性泛化仍然受限且变革性推理几乎没有改善。通过分离和量化这些细微的失败，OMEGA为推动LLM朝着超越机械技能的真实数学创造力奠定了基础。 

---
# CommVQ: Commutative Vector Quantization for KV Cache Compression 

**Title (ZH)**: CommVQ: 交换式向量量化在KV缓存压缩中的应用 

**Authors**: Junyan Li, Yang Zhang, Muhammad Yusuf Hassan, Talha Chafekar, Tianle Cai, Zhile Ren, Pengsheng Guo, Foroozan Karimzadeh, Colorado Reed, Chong Wang, Chuang Gan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18879)  

**Abstract**: Large Language Models (LLMs) are increasingly used in applications requiring long context lengths, but the key-value (KV) cache often becomes a memory bottleneck on GPUs as context grows. To address this, we propose Commutative Vector Quantization (CommVQ) to significantly reduce memory usage for long-context LLM inference. We first introduce additive quantization with a lightweight encoder and codebook to compress the KV cache, which can be decoded via simple matrix multiplication. To further reduce computational costs during decoding, we design the codebook to be commutative with Rotary Position Embedding (RoPE) and train it using an Expectation-Maximization (EM) algorithm. This enables efficient integration of decoding into the self-attention mechanism. Our approach achieves high accuracy with additive quantization and low overhead via the RoPE-commutative codebook. Experiments on long-context benchmarks and GSM8K show that our method reduces FP16 KV cache size by 87.5% with 2-bit quantization, while outperforming state-of-the-art KV cache quantization methods. Notably, it enables 1-bit KV cache quantization with minimal accuracy loss, allowing a LLaMA-3.1 8B model to run with a 128K context length on a single RTX 4090 GPU. The source code is available at: this https URL. 

**Abstract (ZH)**: 长上下文长度Large语言模型（LLMs）中的可交换向量量化（CommVQ）显著减少了GPU内存使用，以进行长期上下文LLM推理。 

---
# OmniGen2: Exploration to Advanced Multimodal Generation 

**Title (ZH)**: OmniGen2：探索高级多模态生成 

**Authors**: Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18871)  

**Abstract**: In this work, we introduce OmniGen2, a versatile and open-source generative model designed to provide a unified solution for diverse generation tasks, including text-to-image, image editing, and in-context generation. Unlike OmniGen v1, OmniGen2 features two distinct decoding pathways for text and image modalities, utilizing unshared parameters and a decoupled image tokenizer. This design enables OmniGen2 to build upon existing multimodal understanding models without the need to re-adapt VAE inputs, thereby preserving the original text generation capabilities. To facilitate the training of OmniGen2, we developed comprehensive data construction pipelines, encompassing image editing and in-context generation data. Additionally, we introduce a reflection mechanism tailored for image generation tasks and curate a dedicated reflection dataset based on OmniGen2. Despite its relatively modest parameter size, OmniGen2 achieves competitive results on multiple task benchmarks, including text-to-image and image editing. To further evaluate in-context generation, also referred to as subject-driven tasks, we introduce a new benchmark named OmniContext. OmniGen2 achieves state-of-the-art performance among open-source models in terms of consistency. We will release our models, training code, datasets, and data construction pipeline to support future research in this field. Project Page: this https URL GitHub Link: this https URL 

**Abstract (ZH)**: 本研究介绍了OmniGen2，这是一个多功能且开源的生成模型，旨在为包括文本到图像、图像编辑和上下文生成在内的多种生成任务提供统一解决方案。与OmniGen v1不同，OmniGen2配备了用于文本和图像模态的两个独立解码路径，使用不同的参数和解耦的图像分词器。这一设计使得OmniGen2能够在不重新适应VAE输入的情况下建立在现有的多模态理解模型之上，从而保留了原始的文本生成能力。为了方便训练OmniGen2，我们开发了全面的数据构建管道，涵盖图像编辑和上下文生成数据。此外，我们还为图像生成任务引入了一种定制的反射机制，并基于OmniGen2构建了一个专门的反射数据集。尽管参数规模相对较小，OmniGen2在包括文本到图像和图像编辑在内的多个任务基准测试中达到了竞争性的成果。为了进一步评估上下文生成，也称为主题驱动任务，我们引入了一个名为OmniContext的新基准。在开源模型中，OmniGen2在一致性方面达到了最先进的性能。我们将在未来的研究中发布我们的模型、训练代码、数据集和数据构建管道。项目页面：https://this.url/project OmniGen2 GitHub链接：https://this.url/code 

---
# OmniAvatar: Efficient Audio-Driven Avatar Video Generation with Adaptive Body Animation 

**Title (ZH)**: OmniAvatar：基于自适应身体动画的高效音频驱动avatar视频生成 

**Authors**: Qijun Gan, Ruizi Yang, Jianke Zhu, Shaofei Xue, Steven Hoi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18866)  

**Abstract**: Significant progress has been made in audio-driven human animation, while most existing methods focus mainly on facial movements, limiting their ability to create full-body animations with natural synchronization and fluidity. They also struggle with precise prompt control for fine-grained generation. To tackle these challenges, we introduce OmniAvatar, an innovative audio-driven full-body video generation model that enhances human animation with improved lip-sync accuracy and natural movements. OmniAvatar introduces a pixel-wise multi-hierarchical audio embedding strategy to better capture audio features in the latent space, enhancing lip-syncing across diverse scenes. To preserve the capability for prompt-driven control of foundation models while effectively incorporating audio features, we employ a LoRA-based training approach. Extensive experiments show that OmniAvatar surpasses existing models in both facial and semi-body video generation, offering precise text-based control for creating videos in various domains, such as podcasts, human interactions, dynamic scenes, and singing. Our project page is this https URL. 

**Abstract (ZH)**: 基于音频的全身视频生成取得了显著进展，尽管现有方法主要集中在面部运动上，限制了其创建自然同步和流畅全身动画的能力。它们在精细生成时也难以实现精确的提示控制。为应对这些挑战，我们提出了OmniAvatar，这是一种创新的基于音频的全身视频生成模型，通过改进唇部同步准确性和自然运动来增强人类动画。OmniAvatar 引入了一种像素级多层级音频嵌入策略，以更好地在潜在空间中捕捉音频特征，从而在多样化场景中提高唇部同步效果。为了保留基础模型驱动提示控制的能力同时有效融入音频特征，我们采用了基于LoRA的训练方法。广泛实验表明，OmniAvatar 在面部和半身视频生成方面均超越了现有模型，提供了精细的文本控制以在播客、人类互动、动态场景和唱歌等多种领域创建视频。我们的项目页面请点击：[该项目链接]。 

---
# TAMMs: Temporal-Aware Multimodal Model for Satellite Image Change Understanding and Forecasting 

**Title (ZH)**: TAMMs：时间感知多模态模型在卫星图像变化理解与预测中的应用 

**Authors**: Zhongbin Guo, Yuhao Wang, Ping Jian, Xinyue Chen, Wei Peng, Ertai E  

**Link**: [PDF](https://arxiv.org/pdf/2506.18862)  

**Abstract**: Satellite image time-series analysis demands fine-grained spatial-temporal reasoning, which remains a challenge for existing multimodal large language models (MLLMs). In this work, we study the capabilities of MLLMs on a novel task that jointly targets temporal change understanding and future scene generation, aiming to assess their potential for modeling complex multimodal dynamics over time. We propose TAMMs, a Temporal-Aware Multimodal Model for satellite image change understanding and forecasting, which enhances frozen MLLMs with lightweight temporal modules for structured sequence encoding and contextual prompting. To guide future image generation, TAMMs introduces a Semantic-Fused Control Injection (SFCI) mechanism that adaptively combines high-level semantic reasoning and structural priors within an enhanced ControlNet. This dual-path conditioning enables temporally consistent and semantically grounded image synthesis. Experiments demonstrate that TAMMs outperforms strong MLLM baselines in both temporal change understanding and future image forecasting tasks, highlighting how carefully designed temporal reasoning and semantic fusion can unlock the full potential of MLLMs for spatio-temporal understanding. 

**Abstract (ZH)**: 卫星图像时间序列分析要求精细的空间-时间推理，这是现有多模态大型语言模型（MLLMs）面临的一项挑战。在这项工作中，我们研究了MLLMs在一项新颖任务上的能力，该任务旨在同时理解时间变化和生成未来场景，以评估其在建模复杂多模态动态方面的潜力。我们提出了一种名为TAMMs的时间感知多模态模型，通过引入轻量级的时间模块增强冻结的MLLMs，以实现结构化序列编码和上下文提示。为了指导未来图像生成，TAMMs引入了一种语义融合控制注入（SFCI）机制，该机制可适应性结合高级语义推理和结构先验，从而实现时空一致性和语义指导的图像合成。实验结果显示，TAMMs在时间变化理解任务和未来图像预测任务上均优于强大的MLLM基线模型，突显了精心设计的时间推理和语义融合如何为MLLMs的时空理解潜力打开大门。 

---
# Mechanistic Interpretability Needs Philosophy 

**Title (ZH)**: 机制可解释性需要哲学 

**Authors**: Iwan Williams, Ninell Oldenburg, Ruchira Dhar, Joshua Hatherley, Constanza Fierro, Nina Rajcic, Sandrine R. Schiller, Filippos Stamatiou, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2506.18852)  

**Abstract**: Mechanistic interpretability (MI) aims to explain how neural networks work by uncovering their underlying causal mechanisms. As the field grows in influence, it is increasingly important to examine not just models themselves, but the assumptions, concepts and explanatory strategies implicit in MI research. We argue that mechanistic interpretability needs philosophy: not as an afterthought, but as an ongoing partner in clarifying its concepts, refining its methods, and assessing the epistemic and ethical stakes of interpreting AI systems. Taking three open problems from the MI literature as examples, this position paper illustrates the value philosophy can add to MI research, and outlines a path toward deeper interdisciplinary dialogue. 

**Abstract (ZH)**: 机制可解释性（MI）旨在通过揭示其潜在的因果机制来解释神经网络的工作原理。随着该领域的影响力日益增强，不仅需要考察模型本身，还需要审视隐含在MI研究中的假设、概念和解释策略。我们认为，机制可解释性需要哲学：不应仅将其视为附带事项，而应将其作为持续的伙伴，用于澄清概念、改进方法，并评估解释AI系统所带来的认识论和伦理学风险。通过MI文献中的三个开放式问题为例，本文阐述哲学能够为MI研究带来的价值，并概述一条通向更深入跨学科对话的道路。 

---
# LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning 

**Title (ZH)**: LongWriter-Zero: 通过强化学习掌握超长文本生成 

**Authors**: Yuhao Wu, Yushi Bai, Zhiqiang Hu, Roy Ka-Wei Lee, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.18841)  

**Abstract**: Ultra-long generation by large language models (LLMs) is a widely demanded scenario, yet it remains a significant challenge due to their maximum generation length limit and overall quality degradation as sequence length increases. Previous approaches, exemplified by LongWriter, typically rely on ''teaching'', which involves supervised fine-tuning (SFT) on synthetic long-form outputs. However, this strategy heavily depends on synthetic SFT data, which is difficult and costly to construct, often lacks coherence and consistency, and tends to be overly artificial and structurally monotonous. In this work, we propose an incentivization-based approach that, starting entirely from scratch and without relying on any annotated or synthetic data, leverages reinforcement learning (RL) to foster the emergence of ultra-long, high-quality text generation capabilities in LLMs. We perform RL training starting from a base model, similar to R1-Zero, guiding it to engage in reasoning that facilitates planning and refinement during the writing process. To support this, we employ specialized reward models that steer the LLM towards improved length control, writing quality, and structural formatting. Experimental evaluations show that our LongWriter-Zero model, trained from Qwen2.5-32B, consistently outperforms traditional SFT methods on long-form writing tasks, achieving state-of-the-art results across all metrics on WritingBench and Arena-Write, and even surpassing 100B+ models such as DeepSeek R1 and Qwen3-235B. We open-source our data and model checkpoints under this https URL 

**Abstract (ZH)**: 超长生成由大型语言模型（LLMs）实现：一种基于激励的方法 

---
# Understanding Software Engineering Agents: A Study of Thought-Action-Result Trajectories 

**Title (ZH)**: 理解软件工程代理：一种思考-行动-结果轨迹研究 

**Authors**: Islem Bouzenia, Michael Pradel  

**Link**: [PDF](https://arxiv.org/pdf/2506.18824)  

**Abstract**: Large Language Model (LLM)-based agents are increasingly employed to automate complex software engineering tasks such as program repair and issue resolution. These agents operate by autonomously generating natural language thoughts, invoking external tools, and iteratively refining their solutions. Despite their widespread adoption, the internal decision-making processes of these agents remain largely unexplored, limiting our understanding of their operational dynamics and failure modes. In this paper, we present a large-scale empirical study of the thought-action-result trajectories of three state-of-the-art LLM-based agents: \textsc{RepairAgent}, \textsc{AutoCodeRover}, and \textsc{OpenHands}. We unify their interaction logs into a common format, capturing 120 trajectories and 2822 LLM interactions focused on program repair and issue resolution. Our study combines quantitative analyses of structural properties, action patterns, and token usage with qualitative assessments of reasoning coherence and feedback integration. We identify key trajectory characteristics such as iteration counts and token consumption, recurring action sequences, and the semantic coherence linking thoughts, actions, and their results. Our findings reveal behavioral motifs and anti-patterns that distinguish successful from failed executions, providing actionable insights for improving agent design, including prompting strategies, failure diagnosis, and anti-pattern detection. We release our dataset and annotation framework to support further research on transparent and robust autonomous software engineering agents. 

**Abstract (ZH)**: 基于大型语言模型（LLM）的代理越来越多地被用来自动化复杂的软件工程任务，如程序修复和问题解决。这些代理通过自主生成自然语言思维、调用外部工具并迭代优化其解决方案来运作。尽管它们已经在广泛采用中，但这些代理内部决策过程的研究仍然不足，限制了我们对其运作动态和失败模式的理解。在本文中，我们提出了对三款先进的基于LLM的代理——\textsc{RepairAgent}、\textsc{AutoCodeRover} 和 \textsc{OpenHands}——思维-行动-结果轨迹的大规模实证研究。我们将它们的交互日志统一到一个共同格式中，记录了120条轨迹和2822次LLM交互，专注于程序修复和问题解决。我们的研究结合了结构属性的定量分析、行动模式和词汇使用量的分析，以及对推理连贯性和反馈整合的定性评估。我们识别出了关键的轨迹特征，如迭代次数和词汇消耗量、反复出现的动作序列以及思维、行动和结果之间的语义连贯性。我们的发现揭示了区分成功和失败执行的行为模式和反模式，为改进代理设计提供了实际指导，包括提示策略、故障诊断和反模式检测。我们发布我们的数据集和注释框架以支持对透明和稳健的自主软件工程代理的进一步研究。 

---
# RWESummary: A Framework and Test for Choosing Large Language Models to Summarize Real-World Evidence (RWE) Studies 

**Title (ZH)**: RWESummary：选择用于总结真实世界证据研究的大语言模型的框架与测试方法 

**Authors**: Arjun Mukerji, Michael L. Jackson, Jason Jones, Neil Sanghavi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18819)  

**Abstract**: Large Language Models (LLMs) have been extensively evaluated for general summarization tasks as well as medical research assistance, but they have not been specifically evaluated for the task of summarizing real-world evidence (RWE) from structured output of RWE studies. We introduce RWESummary, a proposed addition to the MedHELM framework (Bedi, Cui, Fuentes, Unell et al., 2025) to enable benchmarking of LLMs for this task. RWESummary includes one scenario and three evaluations covering major types of errors observed in summarization of medical research studies and was developed using Atropos Health proprietary data. Additionally, we use RWESummary to compare the performance of different LLMs in our internal RWE summarization tool. At the time of publication, with 13 distinct RWE studies, we found the Gemini 2.5 models performed best overall (both Flash and Pro). We suggest RWESummary as a novel and useful foundation model benchmark for real-world evidence study summarization. 

**Abstract (ZH)**: 大型语言模型（LLMs）在通用摘要任务和医疗研究辅助领域得到了广泛评估，但尚未专门评估其在从结构化RWE研究输出中总结实际世界证据（RWE）的任务上的性能。我们提出了RWESummary，作为MedHELM框架（Bedi, Cui, Fuentes, Unell et al., 2025）的一个新增内容，以便对LLMs进行此类任务的基准测试。RWESummary包含一个场景和三项评估，涵盖了在总结医疗研究文章中观察到的主要类型错误，并使用Atropos Health专有数据开发。此外，我们使用RWESummary比较了不同LLMs在内部RWE总结工具中的性能。截至出版时，使用13篇不同的RWE研究，我们发现Gemini 2.5模型整体表现最佳（包括Flash和Pro版本）。我们建议RWESummary作为实际世界证据研究总结的新颖且有用的基准模型。 

---
# OC-SOP: Enhancing Vision-Based 3D Semantic Occupancy Prediction by Object-Centric Awareness 

**Title (ZH)**: OC-SOP: 以对象为中心的 Awareness 提升基于视觉的三维语义占用预测 

**Authors**: Helin Cao, Sven Behnke  

**Link**: [PDF](https://arxiv.org/pdf/2506.18798)  

**Abstract**: Autonomous driving perception faces significant challenges due to occlusions and incomplete scene data in the environment. To overcome these issues, the task of semantic occupancy prediction (SOP) is proposed, which aims to jointly infer both the geometry and semantic labels of a scene from images. However, conventional camera-based methods typically treat all categories equally and primarily rely on local features, leading to suboptimal predictions, especially for dynamic foreground objects. To address this, we propose Object-Centric SOP (OC-SOP), a framework that integrates high-level object-centric cues extracted via a detection branch into the semantic occupancy prediction pipeline. This object-centric integration significantly enhances the prediction accuracy for foreground objects and achieves state-of-the-art performance among all categories on SemanticKITTI. 

**Abstract (ZH)**: 自主驾驶感知面临由于环境中的遮挡和不完整场景数据带来的显著挑战。为克服这些难题，提出了语义占据预测（SOP）任务，旨在从图像中联合推断场景的几何信息和语义标签。然而，传统的基于相机的方法通常平等对待所有类别，并主要依赖局部特征，导致预测效果不佳，特别是对于动态前景物体。为此，我们提出了一种基于对象的SOP（OC-SOP）框架，该框架通过检测分支提取高级对象中心线索并将其集成到语义占据预测管道中。这种对象中心的集成显著提高了前景物体的预测准确性，并在SemanticKITTI的所有类别中实现了最先进的性能。 

---
# Shift Happens: Mixture of Experts based Continual Adaptation in Federated Learning 

**Title (ZH)**: 变化不可避免：基于专家混合的联邦学习连续适应 

**Authors**: Rahul Atul Bhope, K.R. Jayaram, Praveen Venkateswaran, Nalini Venkatasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2506.18789)  

**Abstract**: Federated Learning (FL) enables collaborative model training across decentralized clients without sharing raw data, yet faces significant challenges in real-world settings where client data distributions evolve dynamically over time. This paper tackles the critical problem of covariate and label shifts in streaming FL environments, where non-stationary data distributions degrade model performance and require adaptive middleware solutions. We introduce ShiftEx, a shift-aware mixture of experts framework that dynamically creates and trains specialized global models in response to detected distribution shifts using Maximum Mean Discrepancy for covariate shifts. The framework employs a latent memory mechanism for expert reuse and implements facility location-based optimization to jointly minimize covariate mismatch, expert creation costs, and label imbalance. Through theoretical analysis and comprehensive experiments on benchmark datasets, we demonstrate 5.5-12.9 percentage point accuracy improvements and 22-95 % faster adaptation compared to state-of-the-art FL baselines across diverse shift scenarios. The proposed approach offers a scalable, privacy-preserving middleware solution for FL systems operating in non-stationary, real-world conditions while minimizing communication and computational overhead. 

**Abstract (ZH)**: 联邦学习中的协变量和标签转移问题在流式环境中动态分布变化下的应对方法 

---
# SWA-SOP: Spatially-aware Window Attention for Semantic Occupancy Prediction in Autonomous Driving 

**Title (ZH)**: 基于空间意识窗口注意机制的语义占用预测方法在自动驾驶中的应用 

**Authors**: Helin Cao, Rafael Materla, Sven Behnke  

**Link**: [PDF](https://arxiv.org/pdf/2506.18785)  

**Abstract**: Perception systems in autonomous driving rely on sensors such as LiDAR and cameras to perceive the 3D environment. However, due to occlusions and data sparsity, these sensors often fail to capture complete information. Semantic Occupancy Prediction (SOP) addresses this challenge by inferring both occupancy and semantics of unobserved regions. Existing transformer-based SOP methods lack explicit modeling of spatial structure in attention computation, resulting in limited geometric awareness and poor performance in sparse or occluded areas. To this end, we propose Spatially-aware Window Attention (SWA), a novel mechanism that incorporates local spatial context into attention. SWA significantly improves scene completion and achieves state-of-the-art results on LiDAR-based SOP benchmarks. We further validate its generality by integrating SWA into a camera-based SOP pipeline, where it also yields consistent gains across modalities. 

**Abstract (ZH)**: 自主驾驶中的感知系统依赖于LiDAR和摄像头等传感器来感知3D环境。然而，由于遮挡和数据稀疏性，这些传感器往往无法捕获完整信息。语义占据预测（SOP）通过推断未观察区域的占据和语义来应对这一挑战。现有的基于Transformer的SOP方法在注意力计算中缺乏显式的空域结构建模，导致在稀疏或遮挡区域的几何意识和性能有限。为此，我们提出了一种新的机制——空间aware窗口注意力（SWA），该机制将局部空域上下文融入注意力计算中。SWA显著提高了场景补全，并在LiDAR基于的SOP基准测试中取得了最先进的结果。我们进一步通过将SWA整合到基于摄像头的SOP管道中验证了其普适性，其在不同模态中也取得了一致的改进。 

---
# Sensitivity Analysis of Image Classification Models using Generalized Polynomial Chaos 

**Title (ZH)**: 基于广义多项式混沌的图像分类模型灵敏性分析 

**Authors**: Lukas Bahr, Lucas Poßner, Konstantin Weise, Sophie Gröger, Rüdiger Daub  

**Link**: [PDF](https://arxiv.org/pdf/2506.18751)  

**Abstract**: Integrating advanced communication protocols in production has accelerated the adoption of data-driven predictive quality methods, notably machine learning (ML) models. However, ML models in image classification often face significant uncertainties arising from model, data, and domain shifts. These uncertainties lead to overconfidence in the classification model's output. To better understand these models, sensitivity analysis can help to analyze the relative influence of input parameters on the output. This work investigates the sensitivity of image classification models used for predictive quality. We propose modeling the distributional domain shifts of inputs with random variables and quantifying their impact on the model's outputs using Sobol indices computed via generalized polynomial chaos (GPC). This approach is validated through a case study involving a welding defect classification problem, utilizing a fine-tuned ResNet18 model and an emblem classification model used in BMW Group production facilities. 

**Abstract (ZH)**: 将先进的通信协议集成到生产中加速了数据驱动的预测质量方法的应用，尤其是机器学习模型。然而，图像分类中的机器学习模型经常会因为模型、数据和域移变等因素产生显著的不确定性。这些不确定性会导致分类模型输出过度自信。为了更好地理解这些模型，敏感性分析可以帮助分析输入参数对输出的相对影响。本研究调查了用于预测质量的图像分类模型的敏感性。我们提出使用随机变量建模输入的分布性域移变，并通过广义多项式混沌(GPC)计算的Sobol指数量化其对模型输出的影响。这一方法通过涉及焊接缺陷分类问题的案例研究得到了验证，该案例研究使用了 Fine-Tuned ResNet18 模型和 BMW 集团生产设施中使用的 emblem 分类模型。 

---
# BRAVE: Brain-Controlled Prosthetic Arm with Voice Integration and Embodied Learning for Enhanced Mobility 

**Title (ZH)**: BRAVE: 语音集成和躯体化学习驱动的脑控假肢手臂以提升移动能力 

**Authors**: Abdul Basit, Maha Nawaz, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2506.18749)  

**Abstract**: Non-invasive brain-computer interfaces (BCIs) have the potential to enable intuitive control of prosthetic limbs for individuals with upper limb amputations. However, existing EEG-based control systems face challenges related to signal noise, classification accuracy, and real-time adaptability. In this work, we present BRAVE, a hybrid EEG and voice-controlled prosthetic system that integrates ensemble learning-based EEG classification with a human-in-the-loop (HITL) correction framework for enhanced responsiveness. Unlike traditional electromyography (EMG)-based prosthetic control, BRAVE aims to interpret EEG-driven motor intent, enabling movement control without reliance on residual muscle activity. To improve classification robustness, BRAVE combines LSTM, CNN, and Random Forest models in an ensemble framework, achieving a classification accuracy of 96% across test subjects. EEG signals are preprocessed using a bandpass filter (0.5-45 Hz), Independent Component Analysis (ICA) for artifact removal, and Common Spatial Pattern (CSP) feature extraction to minimize contamination from electromyographic (EMG) and electrooculographic (EOG) signals. Additionally, BRAVE incorporates automatic speech recognition (ASR) to facilitate intuitive mode switching between different degrees of freedom (DOF) in the prosthetic arm. The system operates in real time, with a response latency of 150 ms, leveraging Lab Streaming Layer (LSL) networking for synchronized data acquisition. The system is evaluated on an in-house fabricated prosthetic arm and on multiple participants highlighting the generalizability across users. The system is optimized for low-power embedded deployment, ensuring practical real-world application beyond high-performance computing environments. Our results indicate that BRAVE offers a promising step towards robust, real-time, non-invasive prosthetic control. 

**Abstract (ZH)**: 非侵入式脑-计算机接口：面向上肢截肢个体的直观假肢控制 

---
# ContinualFlow: Learning and Unlearning with Neural Flow Matching 

**Title (ZH)**: 持续流动：神经流匹配中的学习与遗忘 

**Authors**: Lorenzo Simone, Davide Bacciu, Shuangge Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.18747)  

**Abstract**: We introduce ContinualFlow, a principled framework for targeted unlearning in generative models via Flow Matching. Our method leverages an energy-based reweighting loss to softly subtract undesired regions of the data distribution without retraining from scratch or requiring direct access to the samples to be unlearned. Instead, it relies on energy-based proxies to guide the unlearning process. We prove that this induces gradients equivalent to Flow Matching toward a soft mass-subtracted target, and validate the framework through experiments on 2D and image domains, supported by interpretable visualizations and quantitative evaluations. 

**Abstract (ZH)**: 我们介绍了一种名为ContinualFlow的方法，这是一种通过Flow Matching进行生成模型目标性忘存的原理性框架。该方法利用基于能量的重加权损失柔和地减去数据分布中不必要的区域，而无需从头重新训练或直接访问要忘存的数据样本。相反，它依赖基于能量的代理来引导忘存过程。我们证明了这会诱导出与针对柔和减去质量目标的Flow Matching梯度等价的梯度，并通过2D和图像域的实验对其进行验证，这些实验由可解释的可视化和定性评估支持。 

---
# On the Existence of Universal Simulators of Attention 

**Title (ZH)**: 关于通用注意力模拟器的存在性 

**Authors**: Debanjan Dutta, Faizanuddin Ansari, Anish Chakrabarty, Swagatam Das  

**Link**: [PDF](https://arxiv.org/pdf/2506.18739)  

**Abstract**: Prior work on the learnability of transformers has established its capacity to approximate specific algorithmic patterns through training under restrictive architectural assumptions. Fundamentally, these arguments remain data-driven and therefore can only provide a probabilistic guarantee. Expressivity, on the contrary, has theoretically been explored to address the problems \emph{computable} by such architecture. These results proved the Turing-completeness of transformers, investigated bounds focused on circuit complexity, and formal logic. Being at the crossroad between learnability and expressivity, the question remains: \emph{can transformer architectures exactly simulate an arbitrary attention mechanism, or in particular, the underlying operations?} In this study, we investigate the transformer encoder's ability to simulate a vanilla attention mechanism. By constructing a universal simulator $\mathcal{U}$ composed of transformer encoders, we present algorithmic solutions to identically replicate attention outputs and the underlying elementary matrix and activation operations via RASP, a formal framework for transformer computation. Our proofs, for the first time, show the existence of an algorithmically achievable data-agnostic solution, previously known to be approximated only by learning. 

**Abstract (ZH)**: Transformer架构可精确模拟任意注意力机制及其基本矩阵和激活操作的研究 

---
# Deep CNN Face Matchers Inherently Support Revocable Biometric Templates 

**Title (ZH)**: 深度CNN面部匹配器本质上支持可撤回生物特征模板 

**Authors**: Aman Bhatta, Michael C. King, Kevin W. Bowyer  

**Link**: [PDF](https://arxiv.org/pdf/2506.18731)  

**Abstract**: One common critique of biometric authentication is that if an individual's biometric is compromised, then the individual has no recourse. The concept of revocable biometrics was developed to address this concern. A biometric scheme is revocable if an individual can have their current enrollment in the scheme revoked, so that the compromised biometric template becomes worthless, and the individual can re-enroll with a new template that has similar recognition power. We show that modern deep CNN face matchers inherently allow for a robust revocable biometric scheme. For a given state-of-the-art deep CNN backbone and training set, it is possible to generate an unlimited number of distinct face matcher models that have both (1) equivalent recognition power, and (2) strongly incompatible biometric templates. The equivalent recognition power extends to the point of generating impostor and genuine distributions that have the same shape and placement on the similarity dimension, meaning that the models can share a similarity threshold for a 1-in-10,000 false match rate. The biometric templates from different model instances are so strongly incompatible that the cross-instance similarity score for images of the same person is typically lower than the same-instance similarity score for images of different persons. That is, a stolen biometric template that is revoked is of less value in attempting to match the re-enrolled identity than the average impostor template. We also explore the feasibility of using a Vision Transformer (ViT) backbone-based face matcher in the revocable biometric system proposed in this work and demonstrate that it is less suitable compared to typical ResNet-based deep CNN backbones. 

**Abstract (ZH)**: 现代深度CNN面部匹配器中可撤销生物识别方案的实现探究：基于视效变换器的可行性分析 

---
# MuseControlLite: Multifunctional Music Generation with Lightweight Conditioners 

**Title (ZH)**: MuseControlLite：轻量级条件控制器的多功能音乐生成 

**Authors**: Fang-Duo Tsai, Shih-Lun Wu, Weijaw Lee, Sheng-Ping Yang, Bo-Rui Chen, Hao-Chung Cheng, Yi-Hsuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18729)  

**Abstract**: We propose MuseControlLite, a lightweight mechanism designed to fine-tune text-to-music generation models for precise conditioning using various time-varying musical attributes and reference audio signals. The key finding is that positional embeddings, which have been seldom used by text-to-music generation models in the conditioner for text conditions, are critical when the condition of interest is a function of time. Using melody control as an example, our experiments show that simply adding rotary positional embeddings to the decoupled cross-attention layers increases control accuracy from 56.6% to 61.1%, while requiring 6.75 times fewer trainable parameters than state-of-the-art fine-tuning mechanisms, using the same pre-trained diffusion Transformer model of Stable Audio Open. We evaluate various forms of musical attribute control, audio inpainting, and audio outpainting, demonstrating improved controllability over MusicGen-Large and Stable Audio Open ControlNet at a significantly lower fine-tuning cost, with only 85M trainble parameters. Source code, model checkpoints, and demo examples are available at: https: //MuseControlLite.this http URL. 

**Abstract (ZH)**: 我们提出MuseControlLite，这是一种轻量级机制，旨在使用各种时间变化的音乐属性和参考音频信号对文本到音乐生成模型进行微调以实现精确调节。关键发现是，立场嵌入在文本条件下的调节器中鲜有使用，但在目标条件与时间有关时，它们是至关重要的。以旋律控制为例，我们的实验表明，仅向解耦的交叉注意力层添加旋转立场嵌入可将控制准确性从56.6%提高到61.1%，同时使用的可训练参数数量比最先进的微调机制少6.75倍，使用相同的预训练扩散Transformer模型Stable Audio Open。我们评估了各种音乐属性控制、音频修补和音频扩展的形式，证明其在显著降低微调成本的情况下，相比MusicGen-Large和Stable Audio Open ControlNet具有更高的可控性，仅需85M可训练参数。相关源代码、模型检查点和示例可在以下链接获取：https://MuseControlLitethis http URL。 

---
# A Study of Dynamic Stock Relationship Modeling and S&P500 Price Forecasting Based on Differential Graph Transformer 

**Title (ZH)**: 基于差分图变换器的动态股票关系建模与S&P500价格预测研究 

**Authors**: Linyue Hu, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18717)  

**Abstract**: Stock price prediction is vital for investment decisions and risk management, yet remains challenging due to markets' nonlinear dynamics and time-varying inter-stock correlations. Traditional static-correlation models fail to capture evolving stock relationships. To address this, we propose a Differential Graph Transformer (DGT) framework for dynamic relationship modeling and price prediction. Our DGT integrates sequential graph structure changes into multi-head self-attention via a differential graph mechanism, adaptively preserving high-value connections while suppressing noise. Causal temporal attention captures global/local dependencies in price sequences. We further evaluate correlation metrics (Pearson, Mutual Information, Spearman, Kendall's Tau) across global/local/dual scopes as spatial-attention priors. Using 10 years of S&P 500 closing prices (z-score normalized; 64-day sliding windows), DGT with spatial priors outperformed GRU baselines (RMSE: 0.24 vs. 0.87). Kendall's Tau global matrices yielded optimal results (MAE: 0.11). K-means clustering revealed "high-volatility growth" and "defensive blue-chip" stocks, with the latter showing lower errors (RMSE: 0.13) due to stable correlations. Kendall's Tau and Mutual Information excelled in volatile sectors. This study innovatively combines differential graph structures with Transformers, validating dynamic relationship modeling and identifying optimal correlation metrics/scopes. Clustering analysis supports tailored quantitative strategies. Our framework advances financial time-series prediction through dynamic modeling and cross-asset interaction analysis. 

**Abstract (ZH)**: 动态关系建模与价格预测的差分图变换器框架 

---
# Frequency-Weighted Training Losses for Phoneme-Level DNN-based Speech Enhancement 

**Title (ZH)**: 基于频率加权训练损失的音素级DNN语音增强 

**Authors**: Nasser-Eddine Monir, Paul Magron, Romain Serizel  

**Link**: [PDF](https://arxiv.org/pdf/2506.18714)  

**Abstract**: Recent advances in deep learning have significantly improved multichannel speech enhancement algorithms, yet conventional training loss functions such as the scale-invariant signal-to-distortion ratio (SDR) may fail to preserve fine-grained spectral cues essential for phoneme intelligibility. In this work, we propose perceptually-informed variants of the SDR loss, formulated in the time-frequency domain and modulated by frequency-dependent weighting schemes. These weights are designed to emphasize time-frequency regions where speech is prominent or where the interfering noise is particularly strong. We investigate both fixed and adaptive strategies, including ANSI band-importance weights, spectral magnitude-based weighting, and dynamic weighting based on the relative amount of speech and noise. We train the FaSNet multichannel speech enhancement model using these various losses. Experimental results show that while standard metrics such as the SDR are only marginally improved, their perceptual frequency-weighted counterparts exhibit a more substantial improvement. Besides, spectral and phoneme-level analysis indicates better consonant reconstruction, which points to a better preservation of certain acoustic cues. 

**Abstract (ZH)**: 近期深度学习的进展显著提高了多通道语音增强算法的效果，但传统的训练损失函数，如无量纲信噪比（SDR），可能无法保留对音素可懂度至关重要的细粒度频谱线索。在这项工作中，我们提出了感知导向的SDR损失变体，这些变体在时频域中表述，并通过频率依赖的加权方案进行调制。这些权重旨在强调语音突出或干扰噪声特别强烈的时频区域。我们研究了固定和自适应策略，包括ANSI带权重要性加权、基于频谱幅度的加权以及基于语音和噪声相对量的动态加权。我们使用这些不同损失函数来训练FaSNet多通道语音增强模型。实验结果表明，虽然标准指标如SDR仅轻微改善，但其感知频率加权版本显示出更大的改进。此外，频谱和音素级别的分析表明更好的辅音重建，这表明某些声学线索得到了更好的保留。 

---
# Benchmarking the Pedagogical Knowledge of Large Language Models 

**Title (ZH)**: 大规模语言模型的教 学知识基准研究 

**Authors**: Maxime Lelièvre, Amy Waldock, Meng Liu, Natalia Valdés Aspillaga, Alasdair Mackintosh, María José Ogando Portelo, Jared Lee, Paul Atherton, Robin A. A. Ince, Oliver G. B. Garrod  

**Link**: [PDF](https://arxiv.org/pdf/2506.18710)  

**Abstract**: Benchmarks like Massive Multitask Language Understanding (MMLU) have played a pivotal role in evaluating AI's knowledge and abilities across diverse domains. However, existing benchmarks predominantly focus on content knowledge, leaving a critical gap in assessing models' understanding of pedagogy - the method and practice of teaching. This paper introduces The Pedagogy Benchmark, a novel dataset designed to evaluate large language models on their Cross-Domain Pedagogical Knowledge (CDPK) and Special Education Needs and Disability (SEND) pedagogical knowledge. These benchmarks are built on a carefully curated set of questions sourced from professional development exams for teachers, which cover a range of pedagogical subdomains such as teaching strategies and assessment methods. Here we outline the methodology and development of these benchmarks. We report results for 97 models, with accuracies spanning a range from 28% to 89% on the pedagogical knowledge questions. We consider the relationship between cost and accuracy and chart the progression of the Pareto value frontier over time. We provide online leaderboards at this https URL which are updated with new models and allow interactive exploration and filtering based on various model properties, such as cost per token and open-vs-closed weights, as well as looking at performance in different subjects. LLMs and generative AI have tremendous potential to influence education and help to address the global learning crisis. Education-focused benchmarks are crucial to measure models' capacities to understand pedagogical concepts, respond appropriately to learners' needs, and support effective teaching practices across diverse contexts. They are needed for informing the responsible and evidence-based deployment of LLMs and LLM-based tools in educational settings, and for guiding both development and policy decisions. 

**Abstract (ZH)**: 大规模多任务语言理解基准（MMLU）等基准在评估AI的知识和能力方面发挥了关键作用，但现有的基准主要集中在内容知识上，忽略了对模型教学法理解的评估——教学方法和实践。本文介绍了教学基准，这是一个新型数据集，旨在评估大型语言模型在跨领域教学知识（CDPK）和特殊教育需求与残疾（SEND）教学知识方面的表现。这些基准建立在精心选择的问题集上，这些问题源自教师专业发展考试，涵盖了教学策略和评估方法等多种教学亚领域。本文概述了这些基准的方法学和开发过程。我们报告了97个模型在教学知识问题上的准确率，范围从28%到89%。我们考虑了成本与准确率之间的关系，并追踪了帕累托价值前沿随时间的变化。我们在以下网址提供了在线排行榜，会定期更新新的模型，并允许基于各种模型属性（如每个标记的成本和开放权重）进行互动探索和筛选，还涵盖了不同学科的表现。大规模语言模型和生成式AI在教育领域具有巨大潜力，可以影响教育并有助于解决全球学习危机。以教育为目标的基准对于衡量模型理解教学概念、适当回应学习者的需求以及支持不同背景下有效教学实践的能力至关重要。这些基准对于在教育环境中负责任地部署大规模语言模型及其工具以及指导开发和政策决策是必要的。 

---
# Matrix-Game: Interactive World Foundation Model 

**Title (ZH)**: 矩阵博弈：交互世界基础模型 

**Authors**: Yifan Zhang, Chunli Peng, Boyang Wang, Puyi Wang, Qingcheng Zhu, Fei Kang, Biao Jiang, Zedong Gao, Eric Li, Yang Liu, Yahui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.18701)  

**Abstract**: We introduce Matrix-Game, an interactive world foundation model for controllable game world generation. Matrix-Game is trained using a two-stage pipeline that first performs large-scale unlabeled pretraining for environment understanding, followed by action-labeled training for interactive video generation. To support this, we curate Matrix-Game-MC, a comprehensive Minecraft dataset comprising over 2,700 hours of unlabeled gameplay video clips and over 1,000 hours of high-quality labeled clips with fine-grained keyboard and mouse action annotations. Our model adopts a controllable image-to-world generation paradigm, conditioned on a reference image, motion context, and user actions. With over 17 billion parameters, Matrix-Game enables precise control over character actions and camera movements, while maintaining high visual quality and temporal coherence. To evaluate performance, we develop GameWorld Score, a unified benchmark measuring visual quality, temporal quality, action controllability, and physical rule understanding for Minecraft world generation. Extensive experiments show that Matrix-Game consistently outperforms prior open-source Minecraft world models (including Oasis and MineWorld) across all metrics, with particularly strong gains in controllability and physical consistency. Double-blind human evaluations further confirm the superiority of Matrix-Game, highlighting its ability to generate perceptually realistic and precisely controllable videos across diverse game scenarios. To facilitate future research on interactive image-to-world generation, we will open-source the Matrix-Game model weights and the GameWorld Score benchmark at this https URL. 

**Abstract (ZH)**: Matrix-Game：可控的游戏世界生成交互式世界基础模型 

---
# NOVA: Navigation via Object-Centric Visual Autonomy for High-Speed Target Tracking in Unstructured GPS-Denied Environments 

**Title (ZH)**: NOVA：基于物体中心视觉自主的高Speed目标追踪导航在无结构的GPS受限环境中 

**Authors**: Alessandro Saviolo, Giuseppe Loianno  

**Link**: [PDF](https://arxiv.org/pdf/2506.18689)  

**Abstract**: Autonomous aerial target tracking in unstructured and GPS-denied environments remains a fundamental challenge in robotics. Many existing methods rely on motion capture systems, pre-mapped scenes, or feature-based localization to ensure safety and control, limiting their deployment in real-world conditions. We introduce NOVA, a fully onboard, object-centric framework that enables robust target tracking and collision-aware navigation using only a stereo camera and an IMU. Rather than constructing a global map or relying on absolute localization, NOVA formulates perception, estimation, and control entirely in the target's reference frame. A tightly integrated stack combines a lightweight object detector with stereo depth completion, followed by histogram-based filtering to infer robust target distances under occlusion and noise. These measurements feed a visual-inertial state estimator that recovers the full 6-DoF pose of the robot relative to the target. A nonlinear model predictive controller (NMPC) plans dynamically feasible trajectories in the target frame. To ensure safety, high-order control barrier functions are constructed online from a compact set of high-risk collision points extracted from depth, enabling real-time obstacle avoidance without maps or dense representations. We validate NOVA across challenging real-world scenarios, including urban mazes, forest trails, and repeated transitions through buildings with intermittent GPS loss and severe lighting changes that disrupt feature-based localization. Each experiment is repeated multiple times under similar conditions to assess resilience, showing consistent and reliable performance. NOVA achieves agile target following at speeds exceeding 50 km/h. These results show that high-speed vision-based tracking is possible in the wild using only onboard sensing, with no reliance on external localization or environment assumptions. 

**Abstract (ZH)**: 自主立体视觉与IMU融合的无人机无结构GPS受限环境目标跟踪与碰撞意识导航 

---
# SIM-Net: A Multimodal Fusion Network Using Inferred 3D Object Shape Point Clouds from RGB Images for 2D Classification 

**Title (ZH)**: SIM-Net:一种利用RGB图像推断出的3D物体形状点云进行多模态融合的网络，用于2D分类 

**Authors**: Youcef Sklab, Hanane Ariouat, Eric Chenin, Edi Prifti, Jean-Daniel Zucker  

**Link**: [PDF](https://arxiv.org/pdf/2506.18683)  

**Abstract**: We introduce the Shape-Image Multimodal Network (SIM-Net), a novel 2D image classification architecture that integrates 3D point cloud representations inferred directly from RGB images. Our key contribution lies in a pixel-to-point transformation that converts 2D object masks into 3D point clouds, enabling the fusion of texture-based and geometric features for enhanced classification performance. SIM-Net is particularly well-suited for the classification of digitized herbarium specimens (a task made challenging by heterogeneous backgrounds), non-plant elements, and occlusions that compromise conventional image-based models. To address these issues, SIM-Net employs a segmentation-based preprocessing step to extract object masks prior to 3D point cloud generation. The architecture comprises a CNN encoder for 2D image features and a PointNet-based encoder for geometric features, which are fused into a unified latent space. Experimental evaluations on herbarium datasets demonstrate that SIM-Net consistently outperforms ResNet101, achieving gains of up to 9.9% in accuracy and 12.3% in F-score. It also surpasses several transformer-based state-of-the-art architectures, highlighting the benefits of incorporating 3D structural reasoning into 2D image classification tasks. 

**Abstract (ZH)**: Shape-Image Multimodal Network (SIM-Net): 一种结合直接从RGB图像推断的3D点云表示的2D图像分类架构 

---
# Multi-Scale Spectral Attention Module-based Hyperspectral Segmentation in Autonomous Driving Scenarios 

**Title (ZH)**: 基于多尺度谱 attention 模块的自主驾驶场景下 Hyperspectral 分割 

**Authors**: Imad Ali Shah, Jiarong Li, Tim Brophy, Martin Glavin, Edward Jones, Enda Ward, Brian Deegan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18682)  

**Abstract**: Recent advances in autonomous driving (AD) have highlighted the potential of Hyperspectral Imaging (HSI) for enhanced environmental perception, particularly in challenging weather and lighting conditions. However, efficiently processing its high-dimensional spectral data remains a significant challenge. This paper introduces a Multi-scale Spectral Attention Module (MSAM) that enhances spectral feature extraction through three parallel 1D convolutions with varying kernel sizes between 1 to 11, coupled with an adaptive feature aggregation mechanism. By integrating MSAM into UNet's skip connections (UNet-SC), our proposed UNet-MSAM achieves significant improvements in semantic segmentation performance across multiple HSI datasets: HyKo-VIS v2, HSI-Drive v2, and Hyperspectral City v2. Our comprehensive experiments demonstrate that with minimal computational overhead (on average 0.02% in parameters and 0.82% GFLOPS), UNet-MSAM consistently outperforms UNet-SC, achieving average improvements of 3.61% in mean IoU and 3.80% in mF1 across the three datasets. Through extensive ablation studies, we have established that multi-scale kernel combinations perform better than single-scale configurations. These findings demonstrate the potential of HSI processing for AD and provide valuable insights into designing robust, multi-scale spectral feature extractors for real-world applications. 

**Abstract (ZH)**: Recent advances in自主驾驶中的高光谱成像技术进展：多尺度光谱注意力模块在环境感知中的应用 

---
# Is There a Case for Conversation Optimized Tokenizers in Large Language Models? 

**Title (ZH)**: 大型语言模型中对话优化分词器是否有必要？ 

**Authors**: Raquel Ferrando, Javier Conde, Gonzalo Martínez, Pedro Reviriego  

**Link**: [PDF](https://arxiv.org/pdf/2506.18674)  

**Abstract**: The computational and energy costs of Large Language Models (LLMs) have increased exponentially driven by the growing model sizes and the massive adoption of LLMs by hundreds of millions of users. The unit cost of an LLM is the computation of a token. Therefore, the tokenizer plays an important role in the efficiency of a model, and they are carefully optimized to minimize the number of tokens for the text in their training corpus. One of the most popular applications of LLMs are chatbots that interact with users. A key observation is that, for those chatbots, what is important is the performance of the tokenizer in the user text input and the chatbot responses. Those are most likely different from the text in the training corpus. So, a question that immediately arises is whether there is a potential benefit in optimizing tokenizers for chatbot conversations. In this paper, this idea is explored for different tokenizers by using a publicly available corpus of chatbot conversations to redesign their vocabularies and evaluate their performance in this domain. The results show that conversation-optimized tokenizers consistently reduce the number of tokens in chatbot dialogues, which can lead to meaningful energy savings, in the range of 5% to 10% while having minimal or even slightly positive impact on tokenization efficiency for the original training corpus. 

**Abstract (ZH)**: 大型语言模型（LLMs）的计算和能量成本因模型规模的扩大和数百亿用户的大规模采用而呈指数增长。LLM的单位成本是一个词元的计算。因此，分词器在模型的效率中扮演重要角色，它们被仔细优化以最小化训练语料库中文本的词元数量。大型语言模型（LLMs）最流行的应用之一是与用户交互的聊天机器人。一个关键观察是，对于这些聊天机器人来说，重要的是分词器在用户文本输入和聊天机器人响应中的表现。这些文本很可能与训练语料库中的文本不同。因此，一个立即引起的问题是，是否有潜力通过优化分词器来提高聊天机器人对话的表现。在这篇论文中，通过使用一个公开可用的聊天机器人对话语料库来重新设计不同的分词器词汇表并评估其在该领域的性能，探索了这一想法。结果显示，对话优化的分词器一致地减少了聊天机器人对话中的词元数量，可以在范围为5%到10%的水平上带来有意义的能量节省，同时对原始训练语料库中的分词效率影响很小甚至略微提高。 

---
# Benchmarking histopathology foundation models in a multi-center dataset for skin cancer subtyping 

**Title (ZH)**: 多中心数据集中的皮肤癌亚型分类基础模型benchmark研究 

**Authors**: Pablo Meseguer, Rocío del Amor, Valery Naranjo  

**Link**: [PDF](https://arxiv.org/pdf/2506.18668)  

**Abstract**: Pretraining on large-scale, in-domain datasets grants histopathology foundation models (FM) the ability to learn task-agnostic data representations, enhancing transfer learning on downstream tasks. In computational pathology, automated whole slide image analysis requires multiple instance learning (MIL) frameworks due to the gigapixel scale of the slides. The diversity among histopathology FMs has highlighted the need to design real-world challenges for evaluating their effectiveness. To bridge this gap, our work presents a novel benchmark for evaluating histopathology FMs as patch-level feature extractors within a MIL classification framework. For that purpose, we leverage the AI4SkIN dataset, a multi-center cohort encompassing slides with challenging cutaneous spindle cell neoplasm subtypes. We also define the Foundation Model - Silhouette Index (FM-SI), a novel metric to measure model consistency against distribution shifts. Our experimentation shows that extracting less biased features enhances classification performance, especially in similarity-based MIL classifiers. 

**Abstract (ZH)**: 大规模、领域内的预训练使病理学基础模型能够学习任务无关的数据表示，提高了下游任务的迁移学习能力。在计算病理学中，由于玻片的巨像素规模，全玻片图像的自动化分析需要多实例学习（MIL）框架。病理学基础模型之间的多样性凸显了设计实际挑战以评估其有效性的需求。为弥合这一差距，我们的工作呈现了一个新的基准，用于评估病理学基础模型作为MIL分类框架内的patch级特征提取器的有效性。为此，我们利用AI4SkIN数据集，这是一个多中心队列，包含具有挑战性的皮肤梭形细胞肿瘤亚型的玻片。我们还定义了基础模型轮廓指数（FM-SI），这是一种新型度量标准，用于衡量模型在分布偏移下的一致性。我们的实验表明，提取更无偏的特征可以提高分类性能，尤其是在基于相似性的MIL分类器中。 

---
# Historical Report Guided Bi-modal Concurrent Learning for Pathology Report Generation 

**Title (ZH)**: 历史报告引导的双模态并发学习病理报告生成 

**Authors**: Ling Zhang, Boxiang Yun, Qingli Li, Yan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18658)  

**Abstract**: Automated pathology report generation from Whole Slide Images (WSIs) faces two key challenges: (1) lack of semantic content in visual features and (2) inherent information redundancy in WSIs. To address these issues, we propose a novel Historical Report Guided \textbf{Bi}-modal Concurrent Learning Framework for Pathology Report \textbf{Gen}eration (BiGen) emulating pathologists' diagnostic reasoning, consisting of: (1) A knowledge retrieval mechanism to provide rich semantic content, which retrieves WSI-relevant knowledge from pre-built medical knowledge bank by matching high-attention patches and (2) A bi-modal concurrent learning strategy instantiated via a learnable visual token and a learnable textual token to dynamically extract key visual features and retrieved knowledge, where weight-shared layers enable cross-modal alignment between visual features and knowledge features. Our multi-modal decoder integrates both modals for comprehensive diagnostic reports generation. Experiments on the PathText (BRCA) dataset demonstrate our framework's superiority, achieving state-of-the-art performance with 7.4\% relative improvement in NLP metrics and 19.1\% enhancement in classification metrics for Her-2 prediction versus existing methods. Ablation studies validate the necessity of our proposed modules, highlighting our method's ability to provide WSI-relevant rich semantic content and suppress information redundancy in WSIs. Code is publicly available at this https URL. 

**Abstract (ZH)**: 自动从整张切片图像(WSI)生成病理报告面临着两大关键挑战：（1）视觉特征缺乏语义内容和（2）WSI固有的信息冗余。为解决这些问题，我们提出了一种新的基于历史报告引导的双向并发学习框架用于病理报告生成（BiGen），该框架模拟病理学家的诊断推理，包括：（1）一种知识检索机制，通过匹配高关注区域来从预构建的医学知识库中检索WSI相关的知识，提供丰富的语义内容；（2）一种通过可学习的视觉标记和可学习的文本标记实现的双向并发学习策略，以动态提取关键视觉特征和检索到的知识，其中共享权重层能够实现视觉特征和知识特征之间的跨模态对齐。我们的多模态解码器结合两种模态生成全面的诊断报告。在PathText（BRCA）数据集上的实验展示了我们框架的优势，相对于现有方法，在NLP指标上实现了7.4%的相对改进，并在Her-2预测的分类指标上提高了19.1%。消融研究验证了我们提出模块的必要性，突显了我们方法在提供WSI相关丰富语义内容和抑制WSI中的信息冗余方面的能力。代码已在以下网址公开。 

---
# Federated Loss Exploration for Improved Convergence on Non-IID Data 

**Title (ZH)**: federatedLoss探索以改善非iid数据上的收敛性 

**Authors**: Christian Internò, Markus Olhofer, Yaochu Jin, Barbara Hammer  

**Link**: [PDF](https://arxiv.org/pdf/2506.18640)  

**Abstract**: Federated learning (FL) has emerged as a groundbreaking paradigm in machine learning (ML), offering privacy-preserving collaborative model training across diverse datasets. Despite its promise, FL faces significant hurdles in non-identically and independently distributed (non-IID) data scenarios, where most existing methods often struggle with data heterogeneity and lack robustness in performance. This paper introduces Federated Loss Exploration (FedLEx), an innovative approach specifically designed to tackle these challenges. FedLEx distinctively addresses the shortcomings of existing FL methods in non-IID settings by optimizing its learning behavior for scenarios in which assumptions about data heterogeneity are impractical or unknown. It employs a federated loss exploration technique, where clients contribute to a global guidance matrix by calculating gradient deviations for model parameters. This matrix serves as a strategic compass to guide clients' gradient updates in subsequent FL rounds, thereby fostering optimal parameter updates for the global model. FedLEx effectively navigates the complex loss surfaces inherent in non-IID data, enhancing knowledge transfer in an efficient manner, since only a small number of epochs and small amount of data are required to build a strong global guidance matrix that can achieve model convergence without the need for additional data sharing or data distribution statics in a large client scenario. Our extensive experiments with state-of-the art FL algorithms demonstrate significant improvements in performance, particularly under realistic non-IID conditions, thus highlighting FedLEx's potential to overcome critical barriers in diverse FL applications. 

**Abstract (ZH)**: 联邦学习损失探索（FedLEx）：一种应对非同质数据挑战的方法 

---
# Granular-Ball-Induced Multiple Kernel K-Means 

**Title (ZH)**: 由粒球诱导的多重核K均值 

**Authors**: Shuyin Xia, Yifan Wang, Lifeng Shen, Guoyin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18637)  

**Abstract**: Most existing multi-kernel clustering algorithms, such as multi-kernel K-means, often struggle with computational efficiency and robustness when faced with complex data distributions. These challenges stem from their dependence on point-to-point relationships for optimization, which can lead to difficulty in accurately capturing data sets' inherent structure and diversity. Additionally, the intricate interplay between multiple kernels in such algorithms can further exacerbate these issues, effectively impacting their ability to cluster data points in high-dimensional spaces. In this paper, we leverage granular-ball computing to improve the multi-kernel clustering framework. The core of granular-ball computing is to adaptively fit data distribution by balls from coarse to acceptable levels. Each ball can enclose data points based on a density consistency measurement. Such ball-based data description thus improves the computational efficiency and the robustness to unknown noises. Specifically, based on granular-ball representations, we introduce the granular-ball kernel (GBK) and its corresponding granular-ball multi-kernel K-means framework (GB-MKKM) for efficient clustering. Using granular-ball relationships in multiple kernel spaces, the proposed GB-MKKM framework shows its superiority in efficiency and clustering performance in the empirical evaluation of various clustering tasks. 

**Abstract (ZH)**: 大多数现有的多核聚类算法，如多核K均值，往往在面对复杂数据分布时遇到计算效率和鲁棒性的问题。这些问题源自于它们依赖于点对点关系进行优化，这可能导致难以准确捕获数据集固有的结构和多样性。此外，此类算法中多个核函数之间的复杂相互作用会进一步加剧这些问题，从而影响其在高维空间中聚类数据点的能力。本文利用粒度球计算改进多核聚类框架。粒度球计算的核心是通过从较为粗糙到较为合适的层级适应地拟合数据分布。每个球可以根据密度一致性度量包含数据点。基于这种基于球的数据描述，可以提高计算效率和对未知噪声的鲁棒性。具体而言，基于粒度球表示引入粒度球核（GBK）及其对应的粒度球多核K均值框架（GB-MKKM），以实现高效聚类。基于多个核函数空间中的粒度球关系，所提出的GB-MKKM框架在多种聚类任务的实证评估中显示出其在效率和聚类性能上的优越性。 

---
# ReDit: Reward Dithering for Improved LLM Policy Optimization 

**Title (ZH)**: ReDit: 奖励抖动以改进大型语言模型策略优化 

**Authors**: Chenxing Wei, Jiarui Yu, Ying Tiffany He, Hande Dong, Yao Shu, Fei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18631)  

**Abstract**: DeepSeek-R1 has successfully enhanced Large Language Model (LLM) reasoning capabilities through its rule-based reward system. While it's a ''perfect'' reward system that effectively mitigates reward hacking, such reward functions are often discrete. Our experimental observations suggest that discrete rewards can lead to gradient anomaly, unstable optimization, and slow convergence. To address this issue, we propose ReDit (Reward Dithering), a method that dithers the discrete reward signal by adding simple random noise. With this perturbed reward, exploratory gradients are continuously provided throughout the learning process, enabling smoother gradient updates and accelerating convergence. The injected noise also introduces stochasticity into flat reward regions, encouraging the model to explore novel policies and escape local optima. Experiments across diverse tasks demonstrate the effectiveness and efficiency of ReDit. On average, ReDit achieves performance comparable to vanilla GRPO with only approximately 10% the training steps, and furthermore, still exhibits a 4% performance improvement over vanilla GRPO when trained for a similar duration. Visualizations confirm significant mitigation of gradient issues with ReDit. Moreover, theoretical analyses are provided to further validate these advantages. 

**Abstract (ZH)**: DeepSeek-R1 通过基于规则的奖励系统成功增强了大型语言模型的推理能力，尽管这是一个“完美”的奖励系统，有效遏制了奖励误用，但这样的奖励函数往往是离散的。我们的实验观察表明，离散奖励会导致梯度异常、优化不稳定和收敛缓慢。为解决这一问题，我们提出了 ReDit（奖励抖动）方法，通过添加简单的随机噪声来抖动离散奖励信号。通过这种方式扰动的奖励，在学习过程中连续提供探索梯度，使梯度更新更加平滑并加速收敛。注入的噪声还在平坦奖励区域引入了随机性，鼓励模型探索新的策略并跳出局部最优。跨多种任务的实验验证了 ReDit 的有效性和效率。在平均情况下，ReDit 在训练步数减少约 10% 的情况下达到了与标准 GRPO 相似的性能，并且在相似训练时间内相对于标准 GRPO 还表现出 4% 的性能提升。可视化结果进一步证实了 ReDit 对梯度问题的有效缓解。此外，还提供了理论分析以进一步验证这些优势。 

---
# Multi-Agent Reinforcement Learning for Inverse Design in Photonic Integrated Circuits 

**Title (ZH)**: 多代理 reinforcement 学习在光子集成电路逆向设计中的应用 

**Authors**: Yannik Mahlau, Maximilian Schier, Christoph Reinders, Frederik Schubert, Marco Bügling, Bodo Rosenhahn  

**Link**: [PDF](https://arxiv.org/pdf/2506.18627)  

**Abstract**: Inverse design of photonic integrated circuits (PICs) has traditionally relied on gradientbased optimization. However, this approach is prone to end up in local minima, which results in suboptimal design functionality. As interest in PICs increases due to their potential for addressing modern hardware demands through optical computing, more adaptive optimization algorithms are needed. We present a reinforcement learning (RL) environment as well as multi-agent RL algorithms for the design of PICs. By discretizing the design space into a grid, we formulate the design task as an optimization problem with thousands of binary variables. We consider multiple two- and three-dimensional design tasks that represent PIC components for an optical computing system. By decomposing the design space into thousands of individual agents, our algorithms are able to optimize designs with only a few thousand environment samples. They outperform previous state-of-the-art gradient-based optimization in both twoand three-dimensional design tasks. Our work may also serve as a benchmark for further exploration of sample-efficient RL for inverse design in photonics. 

**Abstract (ZH)**: 光电集成电路（PICs）的逆向设计传统上依赖于基于梯度的优化。然而，这种方法容易陷入局部极小值，导致设计功能不佳。随着对PICs的兴趣增加，由于其在光学计算中解决现代硬件需求的潜力，需要更具适应性的优化算法。我们提出了一个强化学习（RL）环境以及多智能体RL算法用于PICs的设计。通过将设计空间离散化为网格，我们将设计任务形式化为具有数千个二进制变量的优化问题。我们考虑了多项代表光学计算系统中PIC组件的二维和三维设计任务。通过将设计空间分解为数千个单一代理，我们的算法能够在少量环境样本的情况下优化设计。它们在二维和三维设计任务中均优于先前最先进的基于梯度的优化方法。我们的工作也可能为进一步探索光子学逆向设计中的样本高效RL提供一个基准。 

---
# Frequency Control in Microgrids: An Adaptive Fuzzy-Neural-Network Virtual Synchronous Generator 

**Title (ZH)**: 微电网中的频率控制：自适应模糊神经网络虚拟同步发电机 

**Authors**: Waleed Breesam, Rezvan Alamian, Nima Tashakor, Brahim Elkhalil Youcefa, Stefan M. Goetz  

**Link**: [PDF](https://arxiv.org/pdf/2506.18611)  

**Abstract**: The reliance on distributed renewable energy has increased recently. As a result, power electronic-based distributed generators replaced synchronous generators which led to a change in the dynamic characteristics of the microgrid. Most critically, they reduced system inertia and damping. Virtual synchronous generators emulated in power electronics, which mimic the dynamic behaviour of synchronous generators, are meant to fix this problem. However, fixed virtual synchronous generator parameters cannot guarantee a frequency regulation within the acceptable tolerance range. Conversely, a dynamic adjustment of these virtual parameters promises robust solution with stable frequency. This paper proposes a method to adapt the inertia, damping, and droop parameters dynamically through a fuzzy neural network controller. This controller trains itself online to choose appropriate values for these virtual parameters. The proposed method can be applied to a typical AC microgrid by considering the penetration and impact of renewable energy sources. We study the system in a MATLAB/Simulink model and validate it experimentally in real time using hardware-in-the-loop based on an embedded ARM system (SAM3X8E, Cortex-M3). Compared to traditional and fuzzy logic controller methods, the results demonstrate that the proposed method significantly reduces the frequency deviation to less than 0.03 Hz and shortens the stabilizing/recovery time. 

**Abstract (ZH)**: 基于分布式可再生能源的依赖性增加，最近促进了电力电子基分布式发电系统取代同步发电机，导致微电网动态特性发生变化。关键地，这减少了系统的惯性和阻尼。为了应对这一问题，通过电力电子模拟同步发电机动态特性的虚拟同步发电机被提出。然而，固定的虚拟同步发电机参数无法保证频率调节在可接受的误差范围内。相反，动态调整这些虚拟参数能够提供具有稳定频率的稳健解决方案。本文提出了一种方法，通过模糊神经网络控制器动态调整惯性、阻尼和分量参数。该控制器在线训练以选择这些虚拟参数的适当值。所提出的方案可应用于考虑可再生能源渗透及其影响的典型AC微电网。我们在MATLAB/Simulink模型中研究了该系统，并基于嵌入式ARM系统（SAM3X8E，Cortex-M3）的硬件在环进行实时实验验证。与传统和模糊逻辑控制器方法相比，结果表明所提出的方法显著减少了频率偏差至小于0.03 Hz，并缩短了稳定/恢复时间。 

---
# Simulation-Free Differential Dynamics through Neural Conservation Laws 

**Title (ZH)**: 无模拟的差分动力学通过神经守恒律 

**Authors**: Mengjian Hua, Eric Vanden-Eijnden, Ricky T.Q. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18604)  

**Abstract**: We present a novel simulation-free framework for training continuous-time diffusion processes over very general objective functions. Existing methods typically involve either prescribing the optimal diffusion process -- which only works for heavily restricted problem formulations -- or require expensive simulation to numerically obtain the time-dependent densities and sample from the diffusion process. In contrast, we propose a coupled parameterization which jointly models a time-dependent density function, or probability path, and the dynamics of a diffusion process that generates this probability path. To accomplish this, our approach directly bakes in the Fokker-Planck equation and density function requirements as hard constraints, by extending and greatly simplifying the construction of Neural Conservation Laws. This enables simulation-free training for a large variety of problem formulations, from data-driven objectives as in generative modeling and dynamical optimal transport, to optimality-based objectives as in stochastic optimal control, with straightforward extensions to mean-field objectives due to the ease of accessing exact density functions. We validate our method in a diverse range of application domains from modeling spatio-temporal events to learning optimal dynamics from population data. 

**Abstract (ZH)**: 一种新型无模拟框架：面向非常通用的目标函数训练连续时间扩散过程 

---
# BulletGen: Improving 4D Reconstruction with Bullet-Time Generation 

**Title (ZH)**: BulletGen: 通过子弹时间生成提高4D重建 

**Authors**: Denys Rozumnyi, Jonathon Luiten, Numair Khan, Johannes Schönberger, Peter Kontschieder  

**Link**: [PDF](https://arxiv.org/pdf/2506.18601)  

**Abstract**: Transforming casually captured, monocular videos into fully immersive dynamic experiences is a highly ill-posed task, and comes with significant challenges, e.g., reconstructing unseen regions, and dealing with the ambiguity in monocular depth estimation. In this work we introduce BulletGen, an approach that takes advantage of generative models to correct errors and complete missing information in a Gaussian-based dynamic scene representation. This is done by aligning the output of a diffusion-based video generation model with the 4D reconstruction at a single frozen "bullet-time" step. The generated frames are then used to supervise the optimization of the 4D Gaussian model. Our method seamlessly blends generative content with both static and dynamic scene components, achieving state-of-the-art results on both novel-view synthesis, and 2D/3D tracking tasks. 

**Abstract (ZH)**: 将随手拍摄的一目镜头视频转化为全沉浸式动态体验是一项高度病态的任务，伴随着重大的挑战，例如重建未见区域，以及处理单目深度估计的不确定性。本文介绍了BulletGen方法，该方法利用生成模型纠正错误并完成高斯基元动态场景表示中的缺失信息。通过将基于扩散的视频生成模型的输出与单一冻结的“子弹时间”步骤下的4D重建对齐完成这一任务，生成的帧随后用于监督4D高斯模型的优化。我们的方法无缝结合了生成内容与静态和动态场景组件，实现了在新颖视角合成和2D/3D跟踪任务上的最新成果。 

---
# Optimization-Induced Dynamics of Lipschitz Continuity in Neural Networks 

**Title (ZH)**: 优化诱导的Lipschitz连续性动力学在神经网络中的研究 

**Authors**: Róisín Luo, James McDermott, Christian Gagné, Qiang Sun, Colm O'Riordan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18588)  

**Abstract**: Lipschitz continuity characterizes the worst-case sensitivity of neural networks to small input perturbations; yet its dynamics (i.e. temporal evolution) during training remains under-explored. We present a rigorous mathematical framework to model the temporal evolution of Lipschitz continuity during training with stochastic gradient descent (SGD). This framework leverages a system of stochastic differential equations (SDEs) to capture both deterministic and stochastic forces. Our theoretical analysis identifies three principal factors driving the evolution: (i) the projection of gradient flows, induced by the optimization dynamics, onto the operator-norm Jacobian of parameter matrices; (ii) the projection of gradient noise, arising from the randomness in mini-batch sampling, onto the operator-norm Jacobian; and (iii) the projection of the gradient noise onto the operator-norm Hessian of parameter matrices. Furthermore, our theoretical framework sheds light on such as how noisy supervision, parameter initialization, batch size, and mini-batch sampling trajectories, among other factors, shape the evolution of the Lipschitz continuity of neural networks. Our experimental results demonstrate strong agreement between the theoretical implications and the observed behaviors. 

**Abstract (ZH)**: Lipschitz连续性表征了小输入扰动下神经网络的最坏情况敏感性；然而其在训练期间的动力学（即时间演变）尚未得到充分探索。我们提出了一种严谨的数学框架，用于建模使用随机梯度下降（SGD）训练期间Lipschitz连续性的时变演化。该框架利用随机微分方程（SDEs）系统来捕捉确定性和随机性力量。我们的理论分析确定了驱动演化过程的三个主要因素：（i）由最优化动态引发的梯度流在参数矩阵操作范数雅可比矩阵上的投影；（ii）由批量采样中的随机性导致的梯度噪声在操作范数雅可比矩阵上的投影；以及（iii）由梯度噪声在操作范数海森矩阵上的投影。此外，我们的理论框架揭示了诸如嘈杂的监督、参数初始化、批量大小、批量采样轨迹等因素如何塑造神经网络Lipschitz连续性的演化。我们的实验结果表明，理论推导的含义与观察到的行为之间存在很强的一致性。 

---
# Security Assessment of DeepSeek and GPT Series Models against Jailbreak Attacks 

**Title (ZH)**: 深层搜索与GPT系列模型针对 Jailbreak 攻击的安全性评估 

**Authors**: Xiaodong Wu, Xiangman Li, Jianbing Ni  

**Link**: [PDF](https://arxiv.org/pdf/2506.18543)  

**Abstract**: The widespread deployment of large language models (LLMs) has raised critical concerns over their vulnerability to jailbreak attacks, i.e., adversarial prompts that bypass alignment mechanisms and elicit harmful or policy-violating outputs. While proprietary models like GPT-4 have undergone extensive evaluation, the robustness of emerging open-source alternatives such as DeepSeek remains largely underexplored, despite their growing adoption in real-world applications. In this paper, we present the first systematic jailbreak evaluation of DeepSeek-series models, comparing them with GPT-3.5 and GPT-4 using the HarmBench benchmark. We evaluate seven representative attack strategies across 510 harmful behaviors categorized by both function and semantic domain. Our analysis reveals that DeepSeek's Mixture-of-Experts (MoE) architecture introduces routing sparsity that offers selective robustness against optimization-based attacks such as TAP-T, but leads to significantly higher vulnerability under prompt-based and manually engineered attacks. In contrast, GPT-4 Turbo demonstrates stronger and more consistent safety alignment across diverse behaviors, likely due to its dense Transformer design and reinforcement learning from human feedback. Fine-grained behavioral analysis and case studies further show that DeepSeek often routes adversarial prompts to under-aligned expert modules, resulting in inconsistent refusal behaviors. These findings highlight a fundamental trade-off between architectural efficiency and alignment generalization, emphasizing the need for targeted safety tuning and modular alignment strategies to ensure secure deployment of open-source LLMs. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的广泛应用引发了对其易受 Jailbreak 攻击的关切，即能够逃避对齐机制并产生有害或政策违反输出的 adversarial prompts。虽然像 GPT-4 这样的专有模型已经经历了广泛的评估，但新兴的开源替代品如 DeepSeek 的稳健性仍然鲜有探索，尽管它们在实际应用中的采用率正在增长。本文首次系统评估了 DeepSeek 系列模型，并使用 HarmBench 基准将其与 GPT-3.5 和 GPT-4 进行比较。我们评估了七个代表性攻击策略在 510 种有害行为中的表现，这些有害行为按功能和语义领域进行分类。分析结果显示，DeepSeek 的 Mixture-of-Experts（MoE）架构引入了路由稀疏性，这种稀疏性对其基于优化的攻击（如 TAP-T）具有选择性的稳健性，但在基于提示的攻击和手动工程化的攻击下则表现出显著更高的脆弱性。相比之下，GPT-4 Turbo 在多种行为中展现出更强且更一致的安全对齐，这可能归因于其密集的Transformer 设计和来自人类反馈的强化学习。细粒度的行为分析和案例研究进一步表明，DeepSeek 经常将 adversarial prompts 路由到未对齐的专家模块，从而导致不一致的拒绝行为。这些发现揭示了架构效率和对齐通用性之间的根本权衡，强调了针对安全调优和模块化对齐策略以确保开源 LLM 安全部署的必要性。 

---
# Embedded FPGA Acceleration of Brain-Like Neural Networks: Online Learning to Scalable Inference 

**Title (ZH)**: 基于嵌入式FPGA的类脑神经网络加速：在线学习到可扩展推理 

**Authors**: Muhammad Ihsan Al Hafiz, Naresh Ravichandran, Anders Lansner, Pawel Herman, Artur Podobas  

**Link**: [PDF](https://arxiv.org/pdf/2506.18530)  

**Abstract**: Edge AI applications increasingly require models that can learn and adapt on-device with minimal energy budget. Traditional deep learning models, while powerful, are often overparameterized, energy-hungry, and dependent on cloud connectivity. Brain-Like Neural Networks (BLNNs), such as the Bayesian Confidence Propagation Neural Network (BCPNN), propose a neuromorphic alternative by mimicking cortical architecture and biologically-constrained learning. They offer sparse architectures with local learning rules and unsupervised/semi-supervised learning, making them well-suited for low-power edge intelligence. However, existing BCPNN implementations rely on GPUs or datacenter FPGAs, limiting their applicability to embedded systems. This work presents the first embedded FPGA accelerator for BCPNN on a Zynq UltraScale+ SoC using High-Level Synthesis. We implement both online learning and inference-only kernels with support for variable and mixed precision. Evaluated on MNIST, Pneumonia, and Breast Cancer datasets, our accelerator achieves up to 17.5x latency and 94% energy savings over ARM baselines, without sacrificing accuracy. This work enables practical neuromorphic computing on edge devices, bridging the gap between brain-like learning and real-world deployment. 

**Abstract (ZH)**: 边缘AI应用 increasingly 要求能在设备上以最小的能量预算进行学习和适应的模型。传统深度学习模型虽然强大，但往往过于参数化、能耗高且依赖于云连接。类脑神经网络（BLNNs），如贝叶斯信念传播神经网络（BCPNN），通过模拟皮层架构和生物限制的学习提出了一种类脑替代方案。它们提供了稀疏架构、局部学习规则和无监督/半监督学习，使其非常适合低功耗边缘智能。然而，现有的BCPNN实现依赖于GPU或数据中心FPGA，限制了其在嵌入式系统中的应用。本文在Zynq UltraScale+ SoC上使用High-Level Synthesis首次提出了BCPNN嵌入式FPGA加速器。我们实现了在线学习和仅推理内核，并支持可变和混合精度。在MNIST、肺炎和乳腺癌数据集上进行评估，我们的加速器分别在延迟和能耗上分别实现了高达17.5倍和94%的节省，而不会牺牲准确性。本文在边缘设备上实现了实用的类脑计算，弥合了类脑学习与实际部署之间的差距。 

---
# Smooth Operators: LLMs Translating Imperfect Hints into Disfluency-Rich Transcripts 

**Title (ZH)**: 平滑操作：大语言模型将不完美的提示转化为富含杂音的转录 

**Authors**: Duygu Altinok  

**Link**: [PDF](https://arxiv.org/pdf/2506.18510)  

**Abstract**: Accurate detection of disfluencies in spoken language is crucial for enhancing the performance of automatic speech and language processing systems, as well as fostering the development of more inclusive speech and language technologies. Leveraging the growing trend of large language models (LLMs) as versatile learners capable of processing both lexical and non-lexical inputs (e.g., audio and video), we propose a novel approach to transcribing disfluencies as explicit tokens with timestamps, enabling the generation of fully annotated disfluency-rich transcripts. Our method integrates acoustic representations extracted from an audio encoder with textual inputs of varying quality: clean transcriptions without disfluencies, time-aligned transcriptions from aligners, or outputs from phoneme-based ASR models -- all of which may contain imperfections. Importantly, our experiments demonstrate that textual inputs do not need to be flawless. As long as they include timestamp-related cues, LLMs can effectively smooth the input and produce fully disfluency-annotated transcripts, underscoring their robustness in handling imperfect hints. 

**Abstract (ZH)**: 准确检测口语中的非流畅性对于提升自动语音和语言处理系统的性能以及促进更具包容性的语音和语言技术的发展至关重要。利用大型语言模型（LLMs）作为既能处理词汇性输入又能处理非词汇性输入（如音频和视频）的通用学习者，我们提出了一种新的方法，即将非流畅性转录为带时间戳的显式标记，从而生成丰富的非流畅性标注转录。该方法结合了从音频编码器提取的声学表示和不同质量的文本输入：无非流畅性的干净转录、对齐器的时间对齐转录或基于音素的ASR模型输出——这些输入中可能包含不完美之处。重要的是，我们的实验表明，文本输入不需要完美。只要它们包含时间戳相关的线索，LLMs就可以有效地平滑输入并生成完整的非流畅性标注转录，突显了它们在处理不完整提示时的鲁棒性。 

---
# Generalizing Vision-Language Models to Novel Domains: A Comprehensive Survey 

**Title (ZH)**: 将视觉-语言模型扩展到新型域：一个全面的综述 

**Authors**: Xinyao Li, Jingjing Li, Fengling Li, Lei Zhu, Yang Yang, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18504)  

**Abstract**: Recently, vision-language pretraining has emerged as a transformative technique that integrates the strengths of both visual and textual modalities, resulting in powerful vision-language models (VLMs). Leveraging web-scale pretraining data, these models exhibit strong zero-shot capabilities. However, their performance often deteriorates when confronted with domain-specific or specialized generalization tasks. To address this, a growing body of research focuses on transferring or generalizing the rich knowledge embedded in VLMs to various downstream applications. This survey aims to comprehensively summarize the generalization settings, methodologies, benchmarking and results in VLM literatures. Delving into the typical VLM structures, current literatures are categorized into prompt-based, parameter-based and feature-based methods according to the transferred modules. The differences and characteristics in each category are furthered summarized and discussed by revisiting the typical transfer learning (TL) settings, providing novel interpretations for TL in the era of VLMs. Popular benchmarks for VLM generalization are further introduced with thorough performance comparisons among the reviewed methods. Following the advances in large-scale generalizable pretraining, this survey also discusses the relations and differences between VLMs and up-to-date multimodal large language models (MLLM), e.g., DeepSeek-VL. By systematically reviewing the surging literatures in vision-language research from a novel and practical generalization prospective, this survey contributes to a clear landscape of current and future multimodal researches. 

**Abstract (ZH)**: 近期，视觉-语言预训练作为一项革新性技术，将视觉和文本模态的优势相结合，产生了强大的视觉-语言模型（VLMs）。这些模型凭借网络规模级别的预训练数据，展现出强大的零样本能力。然而，它们在面对特定领域或专门化泛化任务时，性能往往会下降。为解决这一问题，越来越多的研究致力于将视觉-语言模型中丰富的知识转移到各种下游应用中。本文综述旨在全面总结视觉-语言模型泛化设置、方法、基准测试和结果。通过探究典型视觉-语言模型结构，当前文献根据转移模块被分类为提示基、参数基和特征基方法，并通过回顾典型的迁移学习设置，进一步总结和讨论每个类别中的差异和特点，为视觉-语言模型时代提供了新颖的迁移学习解释。此外，本文还介绍了视觉-语言模型泛化的常用基准，并对所审查方法进行了彻底的性能比较。随着大规模可泛化预训练的进展，本文还讨论了视觉-语言模型与最新多模态大语言模型（MLLM），如DeepSeek-VL之间的关系和差异。通过从新颖和实用的泛化视角系统地回顾视觉-语言研究文献，本文为当前和未来的多模态研究奠定了清晰的框架。 

---
# Comparative Evaluation of ChatGPT and DeepSeek Across Key NLP Tasks: Strengths, Weaknesses, and Domain-Specific Performance 

**Title (ZH)**: ChatGPT与DeepSeek在关键NLP任务上的比较评价：优势、弱点及领域特定性能 

**Authors**: Wael Etaiwi, Bushra Alhijawi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18501)  

**Abstract**: The increasing use of large language models (LLMs) in natural language processing (NLP) tasks has sparked significant interest in evaluating their effectiveness across diverse applications. While models like ChatGPT and DeepSeek have shown strong results in many NLP domains, a comprehensive evaluation is needed to understand their strengths, weaknesses, and domain-specific abilities. This is critical as these models are applied to various tasks, from sentiment analysis to more nuanced tasks like textual entailment and translation. This study aims to evaluate ChatGPT and DeepSeek across five key NLP tasks: sentiment analysis, topic classification, text summarization, machine translation, and textual entailment. A structured experimental protocol is used to ensure fairness and minimize variability. Both models are tested with identical, neutral prompts and evaluated on two benchmark datasets per task, covering domains like news, reviews, and formal/informal texts. The results show that DeepSeek excels in classification stability and logical reasoning, while ChatGPT performs better in tasks requiring nuanced understanding and flexibility. These findings provide valuable insights for selecting the appropriate LLM based on task requirements. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在自然语言处理（NLP）任务中的应用日益增加，引发了对其在各类应用中效果评价的重大兴趣。虽然像ChatGPT和DeepSeek这样的模型在许多NLP领域表现出色，但需要进行全面评估以了解其优势、劣势及其在特定领域的能力。鉴于这些模型在从情感分析到语义蕴含和翻译等不同任务中的广泛应用，本研究旨在通过五个关键NLP任务（情感分析、主题分类、文本摘要、机器翻译和语义蕴含）评价ChatGPT和DeepSeek的表现。采用结构化的实验方案以确保公平性和减少变异性。两种模型使用相同的中性提示进行测试，并在每个任务的两个基准数据集上进行评估，涵盖了新闻、评论以及正式和非正式文本等领域。研究结果表明，DeepSeek在分类稳定性和逻辑推理方面表现优异，而ChatGPT在需要细腻理解和灵活性的任务中表现更佳。这些发现为根据任务需求选择合适的LLM提供了有价值的见解。 

---
# PuckTrick: A Library for Making Synthetic Data More Realistic 

**Title (ZH)**: PuckTrick: 一个使合成数据更加真实的库 

**Authors**: Alessandra Agostini, Andrea Maurino, Blerina Spahiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18499)  

**Abstract**: The increasing reliance on machine learning (ML) models for decision-making requires high-quality training data. However, access to real-world datasets is often restricted due to privacy concerns, proprietary restrictions, and incomplete data availability. As a result, synthetic data generation (SDG) has emerged as a viable alternative, enabling the creation of artificial datasets that preserve the statistical properties of real data while ensuring privacy compliance. Despite its advantages, synthetic data is often overly clean and lacks real-world imperfections, such as missing values, noise, outliers, and misclassified labels, which can significantly impact model generalization and robustness. To address this limitation, we introduce Pucktrick, a Python library designed to systematically contaminate synthetic datasets by introducing controlled errors. The library supports multiple error types, including missing data, noisy values, outliers, label misclassification, duplication, and class imbalance, offering a structured approach to evaluating ML model resilience under real-world data imperfections. Pucktrick provides two contamination modes: one for injecting errors into clean datasets and another for further corrupting already contaminated datasets. Through extensive experiments on real-world financial datasets, we evaluate the impact of systematic data contamination on model performance. Our findings demonstrate that ML models trained on contaminated synthetic data outperform those trained on purely synthetic, error-free data, particularly for tree-based and linear models such as SVMs and Extra Trees. 

**Abstract (ZH)**: 不断增加对机器学习（ML）模型的依赖要求高质量的训练数据。但由于隐私顾虑、专有限制和数据不完整性，获取真实世界数据集往往受到限制。因此，合成数据生成（SDG）已成为一种可行的替代方案，能够创建保留真实数据统计属性的人工数据集，同时确保合规性。尽管具有优势，合成数据往往过于干净，缺乏真实世界的瑕疵，如缺失值、噪声、异常值和标签错误分类，这些瑕疵可能严重影响模型的泛化能力和鲁棒性。为解决这一局限性，我们引入了Pucktrick，这是一个设计用于系统性地通过引入受控错误污染合成数据集的Python库。该库支持多种错误类型，包括缺失数据、噪声值、异常值、标签错误分类、重复和类别不平衡，提供了一种结构化方法来评估在真实世界数据瑕疵下的ML模型鲁棒性。Pucktrick提供了两种污染模式：一种用于向干净数据集注入错误，另一种用于进一步污染已受污染的数据集。通过在实际金融数据集上的广泛实验，我们评估了系统性数据污染对模型性能的影响。我们的研究结果表明，使用受污染合成数据训练的ML模型优于使用纯粹合成且无错误数据训练的模型，特别是对于基于树和线性模型如SVM和Extra Trees而言。 

---
# AI-Generated Song Detection via Lyrics Transcripts 

**Title (ZH)**: 基于歌词转录的AI生成歌曲检测 

**Authors**: Markus Frohmann, Elena V. Epure, Gabriel Meseguer-Brocal, Markus Schedl, Romain Hennequin  

**Link**: [PDF](https://arxiv.org/pdf/2506.18488)  

**Abstract**: The recent rise in capabilities of AI-based music generation tools has created an upheaval in the music industry, necessitating the creation of accurate methods to detect such AI-generated content. This can be done using audio-based detectors; however, it has been shown that they struggle to generalize to unseen generators or when the audio is perturbed. Furthermore, recent work used accurate and cleanly formatted lyrics sourced from a lyrics provider database to detect AI-generated music. However, in practice, such perfect lyrics are not available (only the audio is); this leaves a substantial gap in applicability in real-life use cases. In this work, we instead propose solving this gap by transcribing songs using general automatic speech recognition (ASR) models. We do this using several detectors. The results on diverse, multi-genre, and multi-lingual lyrics show generally strong detection performance across languages and genres, particularly for our best-performing model using Whisper large-v2 and LLM2Vec embeddings. In addition, we show that our method is more robust than state-of-the-art audio-based ones when the audio is perturbed in different ways and when evaluated on different music generators. Our code is available at this https URL. 

**Abstract (ZH)**: 基于AI的音乐生成工具 Recent Capabilities 对音乐产业造成的影响 necessitating 准确检测 AI 生成内容方法的创建。这可以通过使用基于音频的检测器来实现；然而，研究表明它们在面对未见的生成器或音频被扰动时难以泛化。此外，近期的研究使用来自歌词提供商数据库的准确且格式整洁的歌词来检测 AI 生成的音乐。然而，在实践中，这样的完美歌词并不可得（只有音频）；这在实际应用场景中留下了一个巨大的缺口。在本工作中，我们 Instead 提出通过使用通用自动语音识别（ASR）模型进行歌词转写来解决这一缺口。我们使用多种检测器进行这一工作。结果显示，在多种语言和多种流派的歌词上，我们的检测性能普遍强劲，尤其是使用 Whisper large-v2 和 LLM2Vec 向量的最佳模型。此外，我们展示了当音频以不同方式被扰动并在不同的音乐生成器上评估时，我们的方法比最先进的基于音频的方法更具鲁棒性。我们的代码可通过以下链接获取。 

---
# MeRF: Motivation-enhanced Reinforcement Finetuning for Large Reasoning Models 

**Title (ZH)**: 动机增强强化微调大推理模型 

**Authors**: Junjie Zhang, Guozheng Ma, Shunyu Liu, Haoyu Wang, Jiaxing Huang, Ting-En Lin, Fei Huang, Yongbin Li, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.18485)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful learn-to-reason paradigm for Large Language Models (LLMs) to tackle complex reasoning tasks. However, existing RLVR methods overlook one of the most distinctive capabilities of LLMs, their in-context learning ability, as prominently demonstrated by the success of Chain-of-Thought (CoT) prompting. This motivates us to explore how reinforcement learning can be effectively combined with in-context learning to better improve the reasoning capabilities of LLMs. In this paper, we introduce Motivation-enhanced Reinforcement Finetuning} (MeRF), an intuitive yet effective method enhancing reinforcement learning of LLMs by involving ``telling LLMs the rules of the game''. Specifically, MeRF directly injects the reward specification into the prompt, which serves as an in-context motivation for model to improve its responses with awareness of the optimization objective. This simple modification leverages the in-context learning ability of LLMs aligning generation with optimization, thereby incentivizing the model to generate desired outputs from both inner motivation and external reward. Empirical evaluations on the Knights and Knaves~(K&K) logic puzzle reasoning benchmark demonstrate that \texttt{MeRF} achieves substantial performance gains over baselines. Moreover, ablation studies show that performance improves with greater consistency between the in-context motivation and the external reward function, while the model also demonstrates an ability to adapt to misleading motivations through reinforcement learning. 

**Abstract (ZH)**: 可验证奖励的强化学习：动机增强的强化微调（Motivation-enhanced Reinforcement Finetuning for Verifiable Rewards, MeRF） 

---
# A Deep Convolutional Neural Network-Based Novel Class Balancing for Imbalance Data Segmentation 

**Title (ZH)**: 基于深度卷积神经网络的一种新型类别平衡方法在不平衡数据分割中的应用 

**Authors**: Atifa Kalsoom, M.A. Iftikhar, Amjad Ali, Zubair Shah, Shidin Balakrishnan, Hazrat Ali  

**Link**: [PDF](https://arxiv.org/pdf/2506.18474)  

**Abstract**: Retinal fundus images provide valuable insights into the human eye's interior structure and crucial features, such as blood vessels, optic disk, macula, and fovea. However, accurate segmentation of retinal blood vessels can be challenging due to imbalanced data distribution and varying vessel thickness. In this paper, we propose BLCB-CNN, a novel pipeline based on deep learning and bi-level class balancing scheme to achieve vessel segmentation in retinal fundus images. The BLCB-CNN scheme uses a Convolutional Neural Network (CNN) architecture and an empirical approach to balance the distribution of pixels across vessel and non-vessel classes and within thin and thick vessels. Level-I is used for vessel/non-vessel balancing and Level-II is used for thick/thin vessel balancing. Additionally, pre-processing of the input retinal fundus image is performed by Global Contrast Normalization (GCN), Contrast Limited Adaptive Histogram Equalization (CLAHE), and gamma corrections to increase intensity uniformity as well as to enhance the contrast between vessels and background pixels. The resulting balanced dataset is used for classification-based segmentation of the retinal vascular tree. We evaluate the proposed scheme on standard retinal fundus images and achieve superior performance measures, including an area under the ROC curve of 98.23%, Accuracy of 96.22%, Sensitivity of 81.57%, and Specificity of 97.65%. We also demonstrate the method's efficacy through external cross-validation on STARE images, confirming its generalization ability. 

**Abstract (ZH)**: 基于深度学习和分层类平衡方案的BLCB-CNN视网膜 fundus 图血管分割方法 

---
# Benchmarking Foundation Models and Parameter-Efficient Fine-Tuning for Prognosis Prediction in Medical Imaging 

**Title (ZH)**: 医学影像中病程预测的基准模型及参数高效微调研究 

**Authors**: Filippo Ruffini, Elena Mulero Ayllon, Linlin Shen, Paolo Soda, Valerio Guarrasi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18434)  

**Abstract**: Artificial Intelligence (AI) holds significant promise for improving prognosis prediction in medical imaging, yet its effective application remains challenging. In this work, we introduce a structured benchmark explicitly designed to evaluate and compare the transferability of Convolutional Neural Networks and Foundation Models in predicting clinical outcomes in COVID-19 patients, leveraging diverse publicly available Chest X-ray datasets. Our experimental methodology extensively explores a wide set of fine-tuning strategies, encompassing traditional approaches such as Full Fine-Tuning and Linear Probing, as well as advanced Parameter-Efficient Fine-Tuning methods including Low-Rank Adaptation, BitFit, VeRA, and IA3. The evaluations were conducted across multiple learning paradigms, including both extensive full-data scenarios and more clinically realistic Few-Shot Learning settings, which are critical for modeling rare disease outcomes and rapidly emerging health threats. By implementing a large-scale comparative analysis involving a diverse selection of pretrained models, including general-purpose architectures pretrained on large-scale datasets such as CLIP and DINOv2, to biomedical-specific models like MedCLIP, BioMedCLIP, and PubMedCLIP, we rigorously assess each model's capacity to effectively adapt and generalize to prognosis tasks, particularly under conditions of severe data scarcity and pronounced class imbalance. The benchmark was designed to capture critical conditions common in prognosis tasks, including variations in dataset size and class distribution, providing detailed insights into the strengths and limitations of each fine-tuning strategy. This extensive and structured evaluation aims to inform the practical deployment and adoption of robust, efficient, and generalizable AI-driven solutions in real-world clinical prognosis prediction workflows. 

**Abstract (ZH)**: 人工智能（AI）在医学影像中改善預後預測方面具備重要的潛力，但其有效應用仍然挑戰重重。本研究引入了一個結構化基准，旨在評估和比較卷積神經網絡和Foundation Models在預測COVID-19患者臨床預後方面的轉移能力，利用多種公開可用的胸部X光數據集。實驗方法學 slee 生動探索了大量的調參策略，涵蓋傳統方法如全調參和線性探針，以及先進的參數高效調參方法，包括低秩適應、BitFit、VeRA和IA3。評估在多個學習框架下進行，包括廣泛的全數據場景和更符合臨床現實的少樣本學習設置，這些對於建模罕見疾病的預後和快速出現的健康威脅至關重要。通過大規模比較分析，涉及一系列預訓練模型，包括通用架構如CLIP和DINOv2以及醫學專用模型如MedCLIP、BioMedCLIP和PubMedCLIP，我們嚴格評估每種模型在嚴重數據匱乏和顯著類別不平衡條件下的有效適應和泛化能力，特別是對於預後任務。Benchmark設計用以捕捉預後任務中常見的關鍵條件，包括數據集大小和類別分佈的變異，提供對每種調參策略的強點和局限性的詳細洞察。該廣泛且結構化的評估旨在指導robust、高效和普適的人工智能驅動解決方案在真實世界臨床預後預測流程中的實用部署和採用。 

---
# TReB: A Comprehensive Benchmark for Evaluating Table Reasoning Capabilities of Large Language Models 

**Title (ZH)**: TReB：评估大型语言模型表推理能力的综合性基准 

**Authors**: Ce Li, Xiaofan Liu, Zhiyan Song, Ce Chi, Chen Zhao, Jingjing Yang, Zhendong Wang, Kexin Yang, Boshen Shi, Xing Wang, Chao Deng, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.18421)  

**Abstract**: The majority of data in businesses and industries is stored in tables, databases, and data warehouses. Reasoning with table-structured data poses significant challenges for large language models (LLMs) due to its hidden semantics, inherent complexity, and structured nature. One of these challenges is lacking an effective evaluation benchmark fairly reflecting the performances of LLMs on broad table reasoning abilities. In this paper, we fill in this gap, presenting a comprehensive table reasoning evolution benchmark, TReB, which measures both shallow table understanding abilities and deep table reasoning abilities, a total of 26 sub-tasks. We construct a high quality dataset through an iterative data processing procedure. We create an evaluation framework to robustly measure table reasoning capabilities with three distinct inference modes, TCoT, PoT and ICoT. Further, we benchmark over 20 state-of-the-art LLMs using this frame work and prove its effectiveness. Experimental results reveal that existing LLMs still have significant room for improvement in addressing the complex and real world Table related tasks. Both the dataset and evaluation framework are publicly available, with the dataset hosted on [HuggingFace] and the framework on [GitHub]. 

**Abstract (ZH)**: 企业与行业的大多数数据存储在表格、数据库和数据仓库中。处理具有隐藏语义、内在复杂性和结构化性质的表格结构数据对大型语言模型(LLMs)构成了重大挑战。其中一个挑战是没有一个有效的评估基准公平地反映LLMs在广泛表格推理能力上的表现。在本文中，我们填补了这一空白，提出了一个全面的表格推理演化基准TReB，该基准衡量浅层表格理解能力和深层表格推理能力，共计26个子任务。我们通过迭代数据处理程序构建了一个高质量的数据集。我们构建了一个评估框架，使用三种不同的推理模式TCoT、PoT和ICoT稳健地衡量表格推理能力。进一步地，我们使用该框架对超过20个最先进的LLMs进行了基准测试，并证明了其有效性。实验结果表明，现有的LLMs在处理复杂和实际的表格相关任务方面仍有很大的改进空间。数据集和评估框架均已公开，数据集托管在[HuggingFace]上，框架托管在[GitHub]上。 

---
# Latent Space Analysis for Melanoma Prevention 

**Title (ZH)**: latent空间分析在黑色素瘤预防中的应用 

**Authors**: Ciro Listone, Aniello Murano  

**Link**: [PDF](https://arxiv.org/pdf/2506.18414)  

**Abstract**: Melanoma represents a critical health risk due to its aggressive progression and high mortality, underscoring the need for early, interpretable diagnostic tools. While deep learning has advanced in skin lesion classification, most existing models provide only binary outputs, offering limited clinical insight. This work introduces a novel approach that extends beyond classification, enabling interpretable risk modelling through a Conditional Variational Autoencoder. The proposed method learns a structured latent space that captures semantic relationships among lesions, allowing for a nuanced, continuous assessment of morphological differences. An SVM is also trained on this representation effectively differentiating between benign nevi and melanomas, demonstrating strong and consistent performance. More importantly, the learned latent space supports visual and geometric interpretation of malignancy, with the spatial proximity of a lesion to known melanomas serving as a meaningful indicator of risk. This approach bridges predictive performance with clinical applicability, fostering early detection, highlighting ambiguous cases, and enhancing trust in AI-assisted diagnosis through transparent and interpretable decision-making. 

**Abstract (ZH)**: 黑色素瘤由于其侵袭性的进展和高死亡率构成了严重的健康风险，强调了早期、可解释诊断工具的需要。虽然深度学习在皮肤病变分类上取得了进展，但大多数现有模型仅提供二元输出，临床洞察有限。本研究引入了一种新方法，超越分类，通过条件变分自编码器实现可解释的风险建模。所提方法学习一个结构化的潜在空间，捕捉病变间的语义关系，从而实现形态学差异的细致、连续评估。此外，还在该表示上训练了一个SVM，有效地区分良性痣和黑色素瘤，展示了强大的一致性能。更重要的是，学习到的潜在空间支持视觉和几何上的恶性解释，病变的空间位置接近已知黑色素瘤作为风险的有意义指标。该方法将预测性能与临床应用相结合，促进早期检测，突出模糊病例，并通过透明和可解释的决策增强AI辅助诊断的信任度。 

---
# The Debugging Decay Index: Rethinking Debugging Strategies for Code LLMs 

**Title (ZH)**: 代码LLM调试衰减指数：重新思考调试策略 

**Authors**: Muntasir Adnan, Carlos C. N. Kuhn  

**Link**: [PDF](https://arxiv.org/pdf/2506.18403)  

**Abstract**: The effectiveness of AI debugging follows a predictable exponential decay pattern; most models lose 60-80% of their debugging capability within just 2-3 attempts, despite iterative debugging being a critical capability for practical code generation systems. We introduce the Debugging Decay Index (DDI), a mathematical framework that quantifies when debugging becomes ineffective and predicts intervention points. Our strategic fresh start approach shifts from exploitation to exploration at strategic points in the debugging process, demonstrating that well-timed interventions can rescue the effectiveness of debugging. DDI reveals a fundamental limitation in current AI debugging and provides the first quantitative framework for optimising iterative code generation strategies. 

**Abstract (ZH)**: AI调试效果遵循可预测的指数衰减模式：大多数模型在仅2-3次调试尝试后会损失60-80%的调试能力，尽管迭代调试是实际代码生成系统的关键能力。我们引入调试衰减指数（DDI），这是一种数学框架，用于量化调试何时变得无效并预测干预点。我们的策略性全新开始方法在调试过程中的战略点从利用转向探索，表明适时干预可以挽救调试效果。DDI揭示了当前AI调试的基本局限性，并提供了优化迭代代码生成策略的首个定量框架。 

---
# ADNF-Clustering: An Adaptive and Dynamic Neuro-Fuzzy Clustering for Leukemia Prediction 

**Title (ZH)**: ADNF-聚类：一种适用于白血病预测的自适应动态神经模糊聚类 

**Authors**: Marco Aruta, Ciro Listone, Giuseppe Murano, Aniello Murano  

**Link**: [PDF](https://arxiv.org/pdf/2506.18396)  

**Abstract**: Leukemia diagnosis and monitoring rely increasingly on high-throughput image data, yet conventional clustering methods lack the flexibility to accommodate evolving cellular patterns and quantify uncertainty in real time. We introduce Adaptive and Dynamic Neuro-Fuzzy Clustering, a novel streaming-capable framework that combines Convolutional Neural Network-based feature extraction with an online fuzzy clustering engine. ADNF initializes soft partitions via Fuzzy C-Means, then continuously updates micro-cluster centers, densities, and fuzziness parameters using a Fuzzy Temporal Index (FTI) that measures entropy evolution. A topology refinement stage performs density-weighted merging and entropy-guided splitting to guard against over- and under-segmentation. On the C-NMC leukemia microscopy dataset, our tool achieves a silhouette score of 0.51, demonstrating superior cohesion and separation over static baselines. The method's adaptive uncertainty modeling and label-free operation hold immediate potential for integration within the INFANT pediatric oncology network, enabling scalable, up-to-date support for personalized leukemia management. 

**Abstract (ZH)**: 白血病诊断与监测越来越依赖高通量图像数据，但传统聚类方法缺乏适应演化细胞模式和实时量化不确定性的能力。我们提出了一种名为自适应和动态神经模糊聚类（ADNF）的新型流处理框架，该框架结合了基于卷积神经网络的特征提取与在线模糊聚类引擎。ADNF通过模糊C均值初始化软分区，然后使用模糊时间索引（FTI）连续更新微聚类中心、密度和模糊性参数，该索引衡量熵的演变。拓扑优化阶段执行密度加权合并和基于熵的分裂操作，以防止过度分割和欠分割。在C-NMC白血病显微镜数据集上，我们的工具获得了0.51的轮廓分数，显示出比静态基线更好的凝聚性和分离性。该方法的自适应不确定性建模和无标签操作具有立即整合到INFANT儿童肿瘤网络中的潜力，可实现个性化白血病管理的可扩展和及时支持。 

---
# Evaluating Causal Explanation in Medical Reports with LLM-Based and Human-Aligned Metrics 

**Title (ZH)**: 基于LLM和人类对齐指标的医学报告因果解释评估 

**Authors**: Yousang Cho, Key-Sun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18387)  

**Abstract**: This study investigates how accurately different evaluation metrics capture the quality of causal explanations in automatically generated diagnostic reports. We compare six metrics: BERTScore, Cosine Similarity, BioSentVec, GPT-White, GPT-Black, and expert qualitative assessment across two input types: observation-based and multiple-choice-based report generation. Two weighting strategies are applied: one reflecting task-specific priorities, and the other assigning equal weights to all metrics. Our results show that GPT-Black demonstrates the strongest discriminative power in identifying logically coherent and clinically valid causal narratives. GPT-White also aligns well with expert evaluations, while similarity-based metrics diverge from clinical reasoning quality. These findings emphasize the impact of metric selection and weighting on evaluation outcomes, supporting the use of LLM-based evaluation for tasks requiring interpretability and causal reasoning. 

**Abstract (ZH)**: 本研究调查了不同评价指标在捕捉自动生成诊断报告中因果解释质量方面的准确性。我们比较了六种指标：BERTScore、余弦相似度、BioSentVec、GPT-White、GPT-Black以及专家定性评估，涵盖两种输入类型：基于观察的报告生成和基于多项选择的报告生成。我们应用了两种加权策略：一种反映任务特定优先级，另一种将所有指标赋予等权重。研究结果表明，GPT-Black在识别逻辑连贯且临床有效的因果叙述方面表现出最强的区分能力。GPT-White也与专家评估高度一致，而基于相似性的指标偏离了临床推理质量。这些发现强调了指标选择和加权对评价结果的影响，支持在需要可解释性和因果推理的任务中使用基于LLM的评价方法。 

---
# LOGICPO: Efficient Translation of NL-based Logical Problems to FOL using LLMs and Preference Optimization 

**Title (ZH)**: LOGICPO: 使用LLMs和偏好优化将基于自然语言的逻辑问题高效转换为一阶逻辑 

**Authors**: Koushik Viswanadha, Deepanway Ghosal, Somak Aditya  

**Link**: [PDF](https://arxiv.org/pdf/2506.18383)  

**Abstract**: Logical reasoning is a key task for artificial intelligence due to it's role in major downstream tasks such as Question Answering, Summarization. Recent methods in improving the reasoning ability of LLMs fall short in correctly converting a natural language reasoning problem to an equivalent logical formulation, which hinders the framework's overall ability to reason. Towards this, we propose to use finetuning on a preference optimization dataset to learn to parse and represent a natural language problem as a whole to a consistent logical program by 1) introducing a new supervised and preference optimization dataset LogicPO, and 2) adopting popular techniques such as Direct Preference Optimization (DPO), Kahneman-Tversky optimization (KTO) to finetune open-source LLMs. Our best model with Phi-3.5 consistently outperforms GPT-3.5-turbo's (8-shot) by producing 10% more logically correct and with 14% less syntax errors. Through the framework and our improved evaluation metrics, we offer a promising direction in improving the logical reasoning of LLMs by better representing them in their logical formulations. 

**Abstract (ZH)**: 逻辑推理是人工智能的一项关键任务，因其在问答、总结等重要下游任务中的作用。近期提高大语言模型推理能力的方法在准确将自然语言推理问题转换为等效逻辑形式方面存在不足，这妨碍了框架的整体推理能力。为此，我们提出利用偏好优化数据集进行微调，学习将自然语言问题作为一个整体解析和表示为一致的逻辑程序。具体来说，我们1）引入一个新的监督和偏好优化数据集LogicPO；2）采用直接偏好优化（DPO）、 Kahneman-Tversky优化（KTO）等流行技术对开源大语言模型进行微调。我们的最佳模型Phi-3.5始终优于GPT-3.5-turbo（8-shot），在逻辑正确性上提高了10%，在语法错误上减少了14%。通过我们的框架和改进的评估指标，我们提出了提高大语言模型逻辑推理能力的一个有前途的方向，即更好地在其逻辑形式中表示它们。 

---
# PERSCEN: Learning Personalized Interaction Pattern and Scenario Preference for Multi-Scenario Matching 

**Title (ZH)**: PERSCEN: 学习个性化交互模式和场景偏好以实现多场景匹配 

**Authors**: Haotong Du, Yaqing Wang, Fei Xiong, Lei Shao, Ming Liu, Hao Gu, Quanming Yao, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18382)  

**Abstract**: With the expansion of business scales and scopes on online platforms, multi-scenario matching has become a mainstream solution to reduce maintenance costs and alleviate data sparsity. The key to effective multi-scenario recommendation lies in capturing both user preferences shared across all scenarios and scenario-aware preferences specific to each scenario. However, existing methods often overlook user-specific modeling, limiting the generation of personalized user representations. To address this, we propose PERSCEN, an innovative approach that incorporates user-specific modeling into multi-scenario matching. PERSCEN constructs a user-specific feature graph based on user characteristics and employs a lightweight graph neural network to capture higher-order interaction patterns, enabling personalized extraction of preferences shared across scenarios. Additionally, we leverage vector quantization techniques to distil scenario-aware preferences from users' behavior sequence within individual scenarios, facilitating user-specific and scenario-aware preference modeling. To enhance efficient and flexible information transfer, we introduce a progressive scenario-aware gated linear unit that allows fine-grained, low-latency fusion. Extensive experiments demonstrate that PERSCEN outperforms existing methods. Further efficiency analysis confirms that PERSCEN effectively balances performance with computational cost, ensuring its practicality for real-world industrial systems. 

**Abstract (ZH)**: 基于用户特定建模的多场景匹配方法PERSCEN 

---
# Robots and Children that Learn Together : Improving Knowledge Retention by Teaching Peer-Like Interactive Robots 

**Title (ZH)**: 机器人与共同学习的儿童：通过教同伴样交互机器人提高知识保留 

**Authors**: Imene Tarakli, Samuele Vinanzi, Richard Moore, Alessandro Di Nuovo  

**Link**: [PDF](https://arxiv.org/pdf/2506.18365)  

**Abstract**: Despite growing interest in Learning-by-Teaching (LbT), few studies have explored how this paradigm can be implemented with autonomous, peer-like social robots in real classrooms. Most prior work has relied on scripted or Wizard-of-Oz behaviors, limiting our understanding of how real-time, interactive learning can be supported by artificial agents. This study addresses this gap by introducing Interactive Reinforcement Learning (RL) as a cognitive model for teachable social robots. We conducted two between-subject experiments with 58 primary school children, who either taught a robot or practiced independently on a tablet while learning French vocabulary (memorization) and grammatical rules (inference). The robot, powered by Interactive RL, learned from the child's evaluative feedback. Children in the LbT condition achieved significantly higher retention gains compared to those in the self-practice condition, especially on the grammar task. Learners with lower prior knowledge benefited most from teaching the robot. Behavioural metrics revealed that children adapted their teaching strategies over time and engaged more deeply during inference tasks. This work makes two contributions: (1) it introduces Interactive RL as a pedagogically effective and scalable model for peer-robot learning, and (2) it demonstrates, for the first time, the feasibility of deploying multiple autonomous robots simultaneously in real classrooms. These findings extend theoretical understanding of LbT by showing that social robots can function not only as passive tutees but as adaptive partners that enhance meta-cognitive engagement and long-term learning outcomes. 

**Abstract (ZH)**: 尽管学习型教学（LbT）引起了越来越多的兴趣，但在真实课堂中以类似同伴的方式实施自主社会机器人进行学习的研究仍然很少。大多数前期工作依赖于预设或巫师- Oz行为，这限制了我们对即时交互式学习如何由人工代理支持的理解。本研究通过引入交互式强化学习（IRL）作为可教学社会机器人的认知模型来填补这一空白。我们对58名小学生进行了两个被试间实验，他们要么教机器人，要么在学习法语词汇（记忆）和句法规则（推理）的同时独立使用平板电脑练习。机器人由交互式RL驱动，从儿童的评价反馈中学习。在LbT条件下，儿童的表现显著优于自我练习条件，尤其是在句法任务上。知识基础较差的学习者从教机器人中受益最多。行为指标显示，儿童随时间调整了他们的教学策略，并在推理任务中更深入地参与。本研究做出了两个贡献：（1）引入交互式RL作为同伴式机器人学习的有效且可扩展的教育模型；（2）首次展示了在真实教室中同时部署多个自主机器人的可行性。这些发现通过表明社会机器人不仅可以作为被动的学生，还可以作为增强元认知参与和长期学习结果的适应性伙伴，扩展了对LbT的理解。 

---
# Controlled Generation with Equivariant Variational Flow Matching 

**Title (ZH)**: 具等变变异流匹配的可控生成 

**Authors**: Floor Eijkelboom, Heiko Zimmermann, Sharvaree Vadgama, Erik J Bekkers, Max Welling, Christian A. Naesseth, Jan-Willem van de Meent  

**Link**: [PDF](https://arxiv.org/pdf/2506.18340)  

**Abstract**: We derive a controlled generation objective within the framework of Variational Flow Matching (VFM), which casts flow matching as a variational inference problem. We demonstrate that controlled generation can be implemented two ways: (1) by way of end-to-end training of conditional generative models, or (2) as a Bayesian inference problem, enabling post hoc control of unconditional models without retraining. Furthermore, we establish the conditions required for equivariant generation and provide an equivariant formulation of VFM tailored for molecular generation, ensuring invariance to rotations, translations, and permutations. We evaluate our approach on both uncontrolled and controlled molecular generation, achieving state-of-the-art performance on uncontrolled generation and outperforming state-of-the-art models in controlled generation, both with end-to-end training and in the Bayesian inference setting. This work strengthens the connection between flow-based generative modeling and Bayesian inference, offering a scalable and principled framework for constraint-driven and symmetry-aware generation. 

**Abstract (ZH)**: 我们基于变分流匹配（VFM）框架推导出一种受控生成目标，将流匹配问题视为变分推断问题。我们展示了受控生成可以采用两种方式实现：（1）通过端到端训练条件生成模型，或（2）将其视为贝叶斯推理问题，从而在无需重新训练的情况下对无条件模型进行事后控制。此外，我们建立了等变生成所需的条件，并为分子生成提供了一个面向等变生成的VFM形式化表达，确保对旋转、平移和置换的不变性。我们在无控制和受控分子生成方面进行了评估，实现了无控制生成的顶级性能，并在受控生成中（无论是在端到端训练还是在贝叶斯推理设置中）超过了最先进的模型。本工作加强了基于流的生成建模与贝叶斯推理之间的联系，提供了一个适用于约束驱动和对称感知生成的可扩展且原理性的框架。 

---
# Structured Kolmogorov-Arnold Neural ODEs for Interpretable Learning and Symbolic Discovery of Nonlinear Dynamics 

**Title (ZH)**: 结构化柯尔莫哥洛夫-阿诺尔德神经ODEs及其在可解释学习和非线性动力学的符号发现中的应用 

**Authors**: Wei Liu, Kiran Bacsa, Loon Ching Tang, Eleni Chatzi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18339)  

**Abstract**: Understanding and modeling nonlinear dynamical systems is a fundamental problem across scientific and engineering domains. While deep learning has demonstrated remarkable potential for learning complex system behavior, achieving models that are both highly accurate and physically interpretable remains a major challenge. To address this, we propose Structured Kolmogorov-Arnold Neural ODEs (SKANODEs), a novel framework that integrates structured state-space modeling with the Kolmogorov-Arnold Network (KAN). SKANODE first employs a fully trainable KAN as a universal function approximator within a structured Neural ODE framework to perform virtual sensing, recovering latent states that correspond to physically interpretable quantities such as positions and velocities. Once this structured latent representation is established, we exploit the symbolic regression capability of KAN to extract compact and interpretable expressions for the system's governing dynamics. The resulting symbolic expression is then substituted back into the Neural ODE framework and further calibrated through continued training to refine its coefficients, enhancing both the precision of the discovered equations and the predictive accuracy of system responses. Extensive experiments on both simulated and real-world systems demonstrate that SKANODE achieves superior performance while offering interpretable, physics-consistent models that uncover the underlying mechanisms of nonlinear dynamical systems. 

**Abstract (ZH)**: 理解和建模非线性动力系统是科学和工程领域的一项基础问题。尽管深度学习展示了学习复杂系统行为的巨大潜力，但实现既高度准确又具有物理可解释性的模型仍是一项重大挑战。为此，我们提出了一种新颖的框架——结构化柯尔莫哥洛夫-阿诺尔德神经常微分方程（SKANODEs），该框架将结构化状态空间建模与柯尔莫哥洛夫-阿诺尔德网络（KAN）相结合。SKANODE首先利用一个完全可训练的KAN作为结构化神经常微分方程框架内的通用函数逼近器，进行虚拟传感，恢复与位置、速度等物理可解释量相对应的潜在状态。一旦建立这种结构化的潜在表示，我们利用KAN的符号回归能力提取系统的支配动力学的紧凑且可解释的表达式。由此产生的符号表达式随后被重新代入神经常微分方程框架，并通过继续训练进一步校准其系数，从而提高发现方程的精度和系统响应的预测准确性。在仿真和实际系统上的广泛实验表明，SKANODE在保持模型可解释性的同时，实现了物理一致的高性能模型，揭示了非线性动力系统的内在机制。 

---
# Confucius3-Math: A Lightweight High-Performance Reasoning LLM for Chinese K-12 Mathematics Learning 

**Title (ZH)**: 孔夫子3-数学：一种轻量级高性能的中文K-12数学推理大模型 

**Authors**: Lixin Wu, Na Cai, Qiao Cheng, Jiachen Wang, Yitao Duan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18330)  

**Abstract**: We introduce Confucius3-Math, an open-source large language model with 14B parameters that (1) runs efficiently on a single consumer-grade GPU; (2) achieves SOTA performances on a range of mathematical reasoning tasks, outperforming many models with significantly larger sizes. In particular, as part of our mission to enhancing education and knowledge dissemination with AI, Confucius3-Math is specifically committed to mathematics learning for Chinese K-12 students and educators. Built via post-training with large-scale reinforcement learning (RL), Confucius3-Math aligns with national curriculum and excels at solving main-stream Chinese K-12 mathematical problems with low cost. In this report we share our development recipe, the challenges we encounter and the techniques we develop to overcome them. In particular, we introduce three technical innovations: Targeted Entropy Regularization, Recent Sample Recovery and Policy-Specific Hardness Weighting. These innovations encompass a new entropy regularization, a novel data scheduling policy, and an improved group-relative advantage estimator. Collectively, they significantly stabilize the RL training, improve data efficiency, and boost performance. Our work demonstrates the feasibility of building strong reasoning models in a particular domain at low cost. We open-source our model and code at this https URL. 

**Abstract (ZH)**: 我们介绍Confucius3-Math，一个拥有14B参数的开源大语言模型，能够在单块消费级GPU上高效运行；在一系列数学推理任务上实现了SOTA性能，超越了许多规模大得多的模型。特别是作为我们借助AI增强教育和知识传播使命的一部分，Confucius3-Math特别致力于为中国K-12学生和教育者提供数学学习服务。通过大规模强化学习（RL）后训练构建，Confucius3-Math与国家课程体系高度契合，并以低成本解决主流的中国K-12数学问题。在本报告中，我们分享了我们的开发方法、遇到的挑战及克服这些挑战的技术。特别是，我们介绍了三项技术创新：目标熵正则化、最近样本恢复和策略特定难度加权。这些创新包括一种新的熵正则化、一种新颖的数据调度策略，以及一种改进的组内相对优势估计器。它们共同显著稳定了RL训练，提高了数据效率，并提升了性能。我们的工作展示了在特定领域以较低成本构建强大推理模型的可行性。我们已在以下链接开源了我们的模型和代码：https://。 

---
# Bias vs Bias -- Dawn of Justice: A Fair Fight in Recommendation Systems 

**Title (ZH)**: 偏见 vs 偏见 —— 公正的曙光：推荐系统中的公平争斗 

**Authors**: Tahsin Alamgir Kheya, Mohamed Reda Bouadjenek, Sunil Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2506.18327)  

**Abstract**: Recommendation systems play a crucial role in our daily lives by impacting user experience across various domains, including e-commerce, job advertisements, entertainment, etc. Given the vital role of such systems in our lives, practitioners must ensure they do not produce unfair and imbalanced recommendations. Previous work addressing bias in recommendations overlooked bias in certain item categories, potentially leaving some biases unaddressed. Additionally, most previous work on fair re-ranking focused on binary-sensitive attributes. In this paper, we address these issues by proposing a fairness-aware re-ranking approach that helps mitigate bias in different categories of items. This re-ranking approach leverages existing biases to correct disparities in recommendations across various demographic groups. We show how our approach can mitigate bias on multiple sensitive attributes, including gender, age, and occupation. We experimented on three real-world datasets to evaluate the effectiveness of our re-ranking scheme in mitigating bias in recommendations. Our results show how this approach helps mitigate social bias with little to no degradation in performance. 

**Abstract (ZH)**: 推荐系统在电商、招聘信息、娱乐等领域通过影响用户经验发挥着关键作用。鉴于这类系统在生活中的重要性，从业者必须确保它们不会生成不公平和不平衡的推荐。先前针对推荐偏见的研究忽略了某些项目类别中的偏见，可能导致某些偏见未得以解决。此外，大多数关于公平重新排名的工作主要关注二元敏感属性。在本文中，我们通过提出一种公平意识下的重新排名方法来解决这些问题，该方法有助于在不同类别的项目中减轻偏见。该重新排名方法利用现有偏见来纠正不同 demographic 组群在推荐中的差异。我们展示了该方法如何在多个敏感属性（包括性别、年龄和职业）中减轻偏见。我们在三个实际数据集中进行了实验，以评估该重新排名方案在减轻推荐中的偏见方面的有效性。我们的结果显示，该方法在几乎不牺牲性能的情况下有助于减轻社会偏见。 

---
# A Multi-Scale Spatial Attention-Based Zero-Shot Learning Framework for Low-Light Image Enhancement 

**Title (ZH)**: 多尺度空间注意力机制的零样本学习框架用于低光照图像增强 

**Authors**: Muhammad Azeem Aslam, Hassan Khalid, Nisar Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2506.18323)  

**Abstract**: Low-light image enhancement remains a challenging task, particularly in the absence of paired training data. In this study, we present LucentVisionNet, a novel zero-shot learning framework that addresses the limitations of traditional and deep learning-based enhancement methods. The proposed approach integrates multi-scale spatial attention with a deep curve estimation network, enabling fine-grained enhancement while preserving semantic and perceptual fidelity. To further improve generalization, we adopt a recurrent enhancement strategy and optimize the model using a composite loss function comprising six tailored components, including a novel no-reference image quality loss inspired by human visual perception. Extensive experiments on both paired and unpaired benchmark datasets demonstrate that LucentVisionNet consistently outperforms state-of-the-art supervised, unsupervised, and zero-shot methods across multiple full-reference and no-reference image quality metrics. Our framework achieves high visual quality, structural consistency, and computational efficiency, making it well-suited for deployment in real-world applications such as mobile photography, surveillance, and autonomous navigation. 

**Abstract (ZH)**: 低光照图像增强仍然是一个具有挑战性的任务，尤其是在缺乏配对训练数据的情况下。本文提出LucentVisionNet，这是一种新颖的零样本学习框架，旨在解决传统和基于深度学习的增强方法的局限性。所提出的方法将多尺度空间注意力与深度曲线估计网络结合，实现了精细的增强同时保持语义和感知保真度。为进一步提高泛化能力，我们采用了递归增强策略，并使用包含六个定制组件的复合损失函数进行优化，其中包括一种由人类视觉感知启发的新颖无参考图像质量损失。在配对和非配对基准数据集上的 extensive 实验表明，LucentVisionNet 在多个全参考和无参考图像质量指标上始终优于最先进的监督学习、无监督学习和零样本方法。该框架实现了高质量的视觉效果、结构一致性以及计算效率，使其适用于移动摄影、监控和自主导航等实际应用。 

---
# Use Property-Based Testing to Bridge LLM Code Generation and Validation 

**Title (ZH)**: 使用基于属性的测试来弥合大型语言模型代码生成与验证的差距 

**Authors**: Lehan He, Zeren Chen, Zhe Zhang, Jing Shao, Xiang Gao, Lu Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.18315)  

**Abstract**: Large Language Models (LLMs) excel at code generation, but ensuring their outputs to be functionally correct, especially in complex programming tasks, is a persistent challenge. While traditional Test-Driven Development (TDD) offers a path for code refinement, its efficacy with LLMs is often undermined by the scarcity of high-quality test cases or the pitfalls of automated test generation, including biased tests or inaccurate output predictions that can misdirect the correction process. This paper introduces Property-Generated Solver, a novel framework that leverages Property-Based Testing (PBT) to validate high-level program properties or invariants, instead of relying on specific input-output examples. These properties are often simpler to define and verify than directly predicting exhaustive test oracles, breaking the "cycle of self-deception" where tests might share flaws with the code they are meant to validate. Property-Generated Solver employs two collaborative LLM-based agents: a Generator dedicated to code generation and iterative refinement, and a Tester that manages the PBT life-cycle and formulate semantically rich feedback from property violations. The resulting comprehensive and actionable feedback then guides the Generator in its refinement efforts. By establishing PBT as the core validation engine within this iterative, closed-loop paradigm, Property-Generated Solver provides a robust mechanism for steering LLMs towards more correct and generalizable code. Extensive experimental results on multiple code generation benchmarks demonstrate that Property-Generated Solver achieves substantial pass@1 improvements, ranging from 23.1% to 37.3% relative gains over established TDD methods. 

**Abstract (ZH)**: 大型语言模型（LLMs）在代码生成方面表现出色，但确保其输出的功能正确性，特别是在复杂编程任务中，仍是一个持续的挑战。虽然传统的测试驱动开发（TDD）为代码精炼提供了途径，但其与LLMs结合时的效果往往受到高质量测试用例稀缺或自动化测试生成陷阱的影响，包括有偏见的测试或不准确的输出预测，这些都可能误导纠正过程。本文介绍了一种名为Property-Generated Solver的新型框架，它利用基于属性的测试（PBT）来验证高层次程序属性或不变量，而不是依赖于特定的输入-输出示例。这些属性通常比直接预测详尽的测试或acles更易于定义和验证，打破了测试与要验证的代码共享缺陷的“自我蒙蔽循环”。Property-Generated Solver采用两个协作的LLM代理：一个专门用于代码生成和迭代精炼的Generator，以及一个负责管理PBT生命周期并从属性违规中形成语义丰富的反馈的Tester。由此产生的全面且可操作的反馈则指导Generator的精炼努力。通过在迭代、闭环范式中将PBT确立为核心验证引擎，Property-Generated Solver为引导LLMs生成更加正确和泛化的代码提供了一种稳健机制。在多个代码生成基准上的 extensive 实验结果表明，Property-Generated Solver实现了显著的pass@1改进，相对增益范围从23.1%到37.3%。 

---
# LettinGo: Explore User Profile Generation for Recommendation System 

**Title (ZH)**: LettinGo: 探索用户档案生成以优化推荐系统 

**Authors**: Lu Wang, Di Zhang, Fangkai Yang, Pu Zhao, Jianfeng Liu, Yuefeng Zhan, Hao Sun, Qingwei Lin, Weiwei Deng, Dongmei Zhang, Feng Sun, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18309)  

**Abstract**: User profiling is pivotal for recommendation systems, as it transforms raw user interaction data into concise and structured representations that drive personalized recommendations. While traditional embedding-based profiles lack interpretability and adaptability, recent advances with large language models (LLMs) enable text-based profiles that are semantically richer and more transparent. However, existing methods often adhere to fixed formats that limit their ability to capture the full diversity of user behaviors. In this paper, we introduce LettinGo, a novel framework for generating diverse and adaptive user profiles. By leveraging the expressive power of LLMs and incorporating direct feedback from downstream recommendation tasks, our approach avoids the rigid constraints imposed by supervised fine-tuning (SFT). Instead, we employ Direct Preference Optimization (DPO) to align the profile generator with task-specific performance, ensuring that the profiles remain adaptive and effective. LettinGo operates in three stages: (1) exploring diverse user profiles via multiple LLMs, (2) evaluating profile quality based on their impact in recommendation systems, and (3) aligning the profile generation through pairwise preference data derived from task performance. Experimental results demonstrate that our framework significantly enhances recommendation accuracy, flexibility, and contextual awareness. This work enhances profile generation as a key innovation for next-generation recommendation systems. 

**Abstract (ZH)**: 用户画像生成对于推荐系统至关重要，因为它将原始的用户交互数据转化为简洁且结构化的表示，推动个性化推荐。虽然传统的基于嵌入的用户画像缺乏解释性和适应性，但近年来，大规模语言模型（LLMs）的发展使得基于文本的用户画像更加语义丰富且透明。然而，现有方法往往局限于固定格式，限制了它们捕获用户行为多样性的能力。本文介绍了LettinGo，一种生成多样且适应性强的用户画像的新框架。通过利用LLMs的强大表达能力并结合下游推荐任务的直接反馈，我们的方法避免了监督微调（SFT）施加的刚性约束。相反，我们采用了直接偏好优化（DPO）来使用户画像生成器与特定任务的性能对齐，确保用户画像保持适应性和有效性。LettinGo在三个阶段运作：（1）通过多个LLM探索多样化的用户画像；（2）基于其在推荐系统中的影响评价用户画像质量；（3）通过任务性能衍生的成对偏好数据调整用户画像生成。实验结果表明，该框架显著提高了推荐的准确性和灵活性，以及上下文意识。本工作增强了用户画像生成作为下一代推荐系统关键创新的价值。用户画像生成在下一代推荐系统中是一个关键创新。 

---
# Spiffy: Efficient Implementation of CoLaNET for Raspberry Pi 

**Title (ZH)**: Spiffy: 为Raspberry Pi高效实现CoLaNET 

**Authors**: Andrey Derzhavin, Denis Larionov  

**Link**: [PDF](https://arxiv.org/pdf/2506.18306)  

**Abstract**: This paper presents a lightweight software-based approach for running spiking neural networks (SNNs) without relying on specialized neuromorphic hardware or frameworks. Instead, we implement a specific SNN architecture (CoLaNET) in Rust and optimize it for common computing platforms. As a case study, we demonstrate our implementation, called Spiffy, on a Raspberry Pi using the MNIST dataset. Spiffy achieves 92% accuracy with low latency - just 0.9 ms per training step and 0.45 ms per inference step. The code is open-source. 

**Abstract (ZH)**: 本文提出了一种基于轻量级软件的解决方案，用于运行脉冲神经网络（SNNs），而不依赖于专门的神经形态硬件或框架。我们使用Rust实现了一种特定的SNN架构（CoLaNET），并对其进行了优化，以适应常见的计算平台。作为案例研究，我们在Raspberry Pi上使用MNIST数据集展示了我们的实现，名为Spiffy。Spiffy在训练步骤中实现了92%的准确率，延迟仅为0.9毫秒，在推理步骤中延迟为0.45毫秒。该代码是开源的。 

---
# Sharpening the Spear: Adaptive Expert-Guided Adversarial Attack Against DRL-based Autonomous Driving Policies 

**Title (ZH)**: 磨练矛尖：面向基于DRL的自动驾驶策略的自适应专家引导对抗攻击 

**Authors**: Junchao Fan, Xuyang Lei, Xiaolin Chang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18304)  

**Abstract**: Deep reinforcement learning (DRL) has emerged as a promising paradigm for autonomous driving. However, despite their advanced capabilities, DRL-based policies remain highly vulnerable to adversarial attacks, posing serious safety risks in real-world deployments. Investigating such attacks is crucial for revealing policy vulnerabilities and guiding the development of more robust autonomous systems. While prior attack methods have made notable progress, they still face several challenges: 1) they often rely on high-frequency attacks, yet critical attack opportunities are typically context-dependent and temporally sparse, resulting in inefficient attack patterns; 2) restricting attack frequency can improve efficiency but often results in unstable training due to the adversary's limited exploration. To address these challenges, we propose an adaptive expert-guided adversarial attack method that enhances both the stability and efficiency of attack policy training. Our method first derives an expert policy from successful attack demonstrations using imitation learning, strengthened by an ensemble Mixture-of-Experts architecture for robust generalization across scenarios. This expert policy then guides a DRL-based adversary through a KL-divergence regularization term. Due to the diversity of scenarios, expert policies may be imperfect. To address this, we further introduce a performance-aware annealing strategy that gradually reduces reliance on the expert as the adversary improves. Extensive experiments demonstrate that our method achieves outperforms existing approaches in terms of collision rate, attack efficiency, and training stability, especially in cases where the expert policy is sub-optimal. 

**Abstract (ZH)**: 基于自适应专家引导的对抗攻击方法：增强深度强化学习在自动驾驶中的攻击稳定性和效率 

---
# GeNeRT: A Physics-Informed Approach to Intelligent Wireless Channel Modeling via Generalizable Neural Ray Tracing 

**Title (ZH)**: GeNeRT: 一种基于物理的通用神经射线跟踪方法应用于智能无线信道建模 

**Authors**: Kejia Bian, Meixia Tao, Shu Sun, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18295)  

**Abstract**: Neural ray tracing (RT) has emerged as a promising paradigm for channel modeling by combining physical propagation principles with neural networks. It enables high modeling accuracy and efficiency. However, current neural RT methods face two key limitations: constrained generalization capability due to strong spatial dependence, and weak adherence to electromagnetic laws. In this paper, we propose GeNeRT, a Generalizable Neural RT framework with enhanced generalization, accuracy and efficiency. GeNeRT supports both intra-scenario spatial transferability and inter-scenario zero-shot generalization. By incorporating Fresnel-inspired neural network design, it also achieves higher accuracy in multipath component (MPC) prediction. Furthermore, a GPU-tensorized acceleration strategy is introduced to improve runtime efficiency. Extensive experiments conducted in outdoor scenarios demonstrate that GeNeRT generalizes well across untrained regions within a scenario and entirely unseen environments, and achieves superior accuracy in MPC prediction compared to baselines. Moreover, it outperforms Wireless Insite in runtime efficiency, particularly in multi-transmitter settings. Ablation experiments validate the effectiveness of the network architecture and training strategy in capturing physical principles of ray-surface interactions. 

**Abstract (ZH)**: 基于 Fresnel 启发的通用神经射线追踪框架 GeNeRT：增强的泛化能力、准确性和效率 

---
# Selective Social-Interaction via Individual Importance for Fast Human Trajectory Prediction 

**Title (ZH)**: 基于个体重要性的选择性社会互动快速人体轨迹预测 

**Authors**: Yota Urano, Hiromu Taketsugu, Norimichi Ukita  

**Link**: [PDF](https://arxiv.org/pdf/2506.18291)  

**Abstract**: This paper presents an architecture for selecting important neighboring people to predict the primary person's trajectory. To achieve effective neighboring people selection, we propose a people selection module called the Importance Estimator which outputs the importance of each neighboring person for predicting the primary person's future trajectory. To prevent gradients from being blocked by non-differentiable operations when sampling surrounding people based on their importance, we employ the Gumbel Softmax for training. Experiments conducted on the JRDB dataset show that our method speeds up the process with competitive prediction accuracy. 

**Abstract (ZH)**: 本文提出了一种架构，用于选择重要临近人群以预测主要人群的轨迹。为了实现有效的临近人群选择，我们提出了一种称为重要性估计器的人群选择模块，该模块输出每个临近人群对未来主要人群轨迹预测的重要性。为了在基于重要性采样周围人群时防止梯度被非可微操作阻断，我们采用了Gumbel Softmax进行训练。实验在JRDB数据集上的结果显示，我们的方法能够在保持竞争力的预测精度的同时加快过程。 

---
# Tu(r)ning AI Green: Exploring Energy Efficiency Cascading with Orthogonal Optimizations 

**Title (ZH)**: 转向绿色AI：探索正交优化下的能源效率 cascading 效应 

**Authors**: Saurabhsingh Rajput, Mootez Saad, Tushar Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2506.18289)  

**Abstract**: AI's exponential growth intensifies computational demands and energy challenges. While practitioners employ various optimization techniques, that we refer as "knobs" in this paper, to tune model efficiency, these are typically afterthoughts and reactive ad-hoc changes applied in isolation without understanding their combinatorial effects on energy efficiency. This paper emphasizes on treating energy efficiency as the first-class citizen and as a fundamental design consideration for a compute-intensive pipeline. We show that strategic selection across five AI pipeline phases (data, model, training, system, inference) creates cascading efficiency. Experimental validation shows orthogonal combinations reduce energy consumption by up to $94.6$% while preserving $95.95$% of the original F1 score of non-optimized pipelines. This curated approach provides actionable frameworks for informed sustainable AI that balance efficiency, performance, and environmental responsibility. 

**Abstract (ZH)**: AI的指数级增长加剧了计算需求和能源挑战。本文强调将能源效率视为首要考虑因素和计算密集型流程的基本设计考量。我们展示出在五个AI流程阶段（数据、模型、训练、系统、推理）中进行策略性选择可以产生级联的效率提升。实验验证表明，正交组合可以减少高达94.6%的能耗，同时保持95.95%的原始非优化流程的F1分数。这种精心设计的方法为平衡效率、性能和环境责任的可持续AI提供了可操作的框架。 

---
# Learning Causal Graphs at Scale: A Foundation Model Approach 

**Title (ZH)**: 大规模学习因果图：一种基础模型方法 

**Authors**: Naiyu Yin, Tian Gao, Yue Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18285)  

**Abstract**: Due to its human-interpretability and invariance properties, Directed Acyclic Graph (DAG) has been a foundational tool across various areas of AI research, leading to significant advancements. However, DAG learning remains highly challenging, due to its super-exponential growth in computational cost and identifiability issues, particularly in small-sample regimes. To address these two challenges, in this work we leverage the recent success of linear transformers and develop a foundation model approach for discovering multiple order-consistent DAGs across tasks. In particular, we propose Attention-DAG (ADAG), a novel attention-mechanism-based architecture for learning multiple linear Structural Equation Models (SEMs). ADAG learns the mapping from observed data to both graph structure and parameters via a nonlinear attention-based kernel, enabling efficient multi-task estimation of the underlying linear SEMs. By formulating the learning process across multiple tasks as a continuous optimization problem, the pre-trained ADAG model captures the common structural properties as a shared low-dimensional prior, thereby reducing the ill-posedness of downstream DAG learning tasks in small-sample regimes. We evaluate our proposed approach on benchmark synthetic datasets and find that ADAG achieves substantial improvements in both DAG learning accuracy and zero-shot inference efficiency. To the best of our knowledge, this is the first practical approach for pre-training a foundation model specifically designed for DAG learning, representing a step toward more efficient and generalizable down-stream applications in causal discovery. 

**Abstract (ZH)**: 基于注意力机制的多任务DAG学习：一种用于因果发现的基础模型方法 

---
# Open Set Recognition for Endoscopic Image Classification: A Deep Learning Approach on the Kvasir Dataset 

**Title (ZH)**: 基于Kvasir数据集的端oscopic图像开放集识别：一种深度学习方法 

**Authors**: Kasra Moazzami, Seoyoun Son, John Lin, Sun Min Lee, Daniel Son, Hayeon Lee, Jeongho Lee, Seongji Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.18284)  

**Abstract**: Endoscopic image classification plays a pivotal role in medical diagnostics by identifying anatomical landmarks and pathological findings. However, conventional closed-set classification frameworks are inherently limited in open-world clinical settings, where previously unseen conditions can arise andcompromise model reliability. To address this, we explore the application of Open Set Recognition (OSR) techniques on the Kvasir dataset, a publicly available and diverse endoscopic image collection. In this study, we evaluate and compare the OSR capabilities of several representative deep learning architectures, including ResNet-50, Swin Transformer, and a hybrid ResNet-Transformer model, under both closed-set and open-set conditions. OpenMax is adopted as a baseline OSR method to assess the ability of these models to distinguish known classes from previously unseen categories. This work represents one of the first efforts to apply open set recognition to the Kvasir dataset and provides a foundational benchmark for evaluating OSR performance in medical image analysis. Our results offer practical insights into model behavior in clinically realistic settings and highlight the importance of OSR techniques for the safe deployment of AI systems in endoscopy. 

**Abstract (ZH)**: 基于Open Set Recognition的Kvasir数据集内窥图像分类研究 

---
# ARD-LoRA: Dynamic Rank Allocation for Parameter-Efficient Fine-Tuning of Foundation Models with Heterogeneous Adaptation Needs 

**Title (ZH)**: ARD-LoRA: 不同适应需求下的参数高效微调的动态秩分配方法 

**Authors**: Haseeb Ullah Khan Shinwari, Muhammad Usama  

**Link**: [PDF](https://arxiv.org/pdf/2506.18267)  

**Abstract**: Conventional Low-Rank Adaptation (LoRA) methods employ a fixed rank, imposing uniform adaptation across transformer layers and attention heads despite their heterogeneous learning dynamics. This paper introduces Adaptive Rank Dynamic LoRA (ARD-LoRA), a novel framework that automates rank allocation through learnable scaling factors. These factors are optimized via a meta-objective balancing task performance and parameter efficiency, incorporating $\ell_1$ sparsity for minimal rank and Total Variation regularization for stable rank transitions. ARD-LoRA enables continuous, differentiable, per-head rank adaptation. Experiments on LLAMA-3.1-70B and PaliGemma-2 demonstrate ARD-LoRA's efficacy, achieving up to 99.3% of full fine-tuning performance with only 0.32% trainable parameters, outperforming strong baselines like DoRA and AdaLoRA. Furthermore, it reduces multimodal adaptation memory by 41%. These results establish dynamic, fine-grained rank allocation as a critical paradigm for efficient foundation model adaptation. 

**Abstract (ZH)**: 自适应秩动态LoRA (ARD-LoRA): 通过可学习的缩放因子自动分配秩 

---
# RLPR: Extrapolating RLVR to General Domains without Verifiers 

**Title (ZH)**: RLPR：将RLVR扩展到一般领域而无需验证器 

**Authors**: Tianyu Yu, Bo Ji, Shouli Wang, Shu Yao, Zefan Wang, Ganqu Cui, Lifan Yuan, Ning Ding, Yuan Yao, Zhiyuan Liu, Maosong Sun, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.18254)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) demonstrates promising potential in advancing the reasoning capabilities of LLMs. However, its success remains largely confined to mathematical and code domains. This primary limitation stems from the heavy reliance on domain-specific verifiers, which results in prohibitive complexity and limited scalability. To address the challenge, our key observation is that LLM's intrinsic probability of generating a correct free-form answer directly indicates its own evaluation of the reasoning reward (i.e., how well the reasoning process leads to the correct answer). Building on this insight, we propose RLPR, a simple verifier-free framework that extrapolates RLVR to broader general domains. RLPR uses the LLM's own token probability scores for reference answers as the reward signal and maximizes the expected reward during training. We find that addressing the high variance of this noisy probability reward is crucial to make it work, and propose prob-to-reward and stabilizing methods to ensure a precise and stable reward from LLM intrinsic probabilities. Comprehensive experiments in four general-domain benchmarks and three mathematical benchmarks show that RLPR consistently improves reasoning capabilities in both areas for Gemma, Llama, and Qwen based models. Notably, RLPR outperforms concurrent VeriFree by 7.6 points on TheoremQA and 7.5 points on Minerva, and even surpasses strong verifier-model-dependent approaches General-Reasoner by 1.6 average points across seven benchmarks. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）在提升大语言模型的推理能力方面显示出令人鼓舞的潜力，但其成功主要局限于数学和代码领域。这一主要限制源于对领域特定验证器的依赖，导致复杂性高且难以扩展。为应对这一挑战，我们的核心观察是，大语言模型生成正确自由形式答案的内在概率直接反映了其对推理奖励的自我评估（即推理过程如何导致正确答案）。基于这一洞察，我们提出了一个简单的无验证器框架RLPR，将RLVR扩展到更广泛的通用领域。RLPR使用大语言模型自身对参考答案的标记概率分数作为奖励信号，并在训练过程中最大化预期奖励。我们发现，解决这种噪声概率奖励的高方差是使其有效工作的关键，并提出了概率到奖励和稳定化方法，以确保从大语言模型内在概率中获得准确且稳定的奖励。在四个通用领域基准和三个数学基准的全面实验中，我们发现RLPR能够持续提高Gemma、Llama和Qwen等模型的推理能力。值得注意的是，RLPR在TheoremQA上比同时期的VeriFree高7.6分，在Minerva上高7.5分，并且在七个基准中平均高出1.6分超过了强验证器模型依赖的方法General-Reasoner。 

---
# Morse: Dual-Sampling for Lossless Acceleration of Diffusion Models 

**Title (ZH)**: Morse: 双采样加速扩散模型的无损加速方法 

**Authors**: Chao Li, Jiawei Fan, Anbang Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.18251)  

**Abstract**: In this paper, we present Morse, a simple dual-sampling framework for accelerating diffusion models losslessly. The key insight of Morse is to reformulate the iterative generation (from noise to data) process via taking advantage of fast jump sampling and adaptive residual feedback strategies. Specifically, Morse involves two models called Dash and Dot that interact with each other. The Dash model is just the pre-trained diffusion model of any type, but operates in a jump sampling regime, creating sufficient space for sampling efficiency improvement. The Dot model is significantly faster than the Dash model, which is learnt to generate residual feedback conditioned on the observations at the current jump sampling point on the trajectory of the Dash model, lifting the noise estimate to easily match the next-step estimate of the Dash model without jump sampling. By chaining the outputs of the Dash and Dot models run in a time-interleaved fashion, Morse exhibits the merit of flexibly attaining desired image generation performance while improving overall runtime efficiency. With our proposed weight sharing strategy between the Dash and Dot models, Morse is efficient for training and inference. Our method shows a lossless speedup of 1.78X to 3.31X on average over a wide range of sampling step budgets relative to 9 baseline diffusion models on 6 image generation tasks. Furthermore, we show that our method can be also generalized to improve the Latent Consistency Model (LCM-SDXL, which is already accelerated with consistency distillation technique) tailored for few-step text-to-image synthesis. The code and models are available at this https URL. 

**Abstract (ZH)**: Morse：一种加速扩散模型的简单双采样框架 

---
# Semantic Structure-Aware Generative Attacks for Enhanced Adversarial Transferability 

**Title (ZH)**: 面向语义结构的生成型攻击以增强对抗性转移性 

**Authors**: Jongoh Jeong, Hunmin Yang, Jaeseok Jeong, Kuk-Jin Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2506.18248)  

**Abstract**: Generative adversarial attacks train a perturbation generator on a white-box surrogate model and subsequently apply the crafted perturbations to unseen black-box victim models. In contrast to iterative attacks, these methods deliver superior inference-time efficiency, scalability, and transferability; however, up until now, existing studies have not fully exploited the representational capacity of generative models to preserve and harness semantic information. Specifically, the intermediate activations of the generator encode rich semantic features--object boundaries and coarse shapes--that remain under-exploited, thereby limiting the alignment of perturbations with object-salient regions which are critical for adversarial transferability. To remedy this, we introduce a semantic structure-aware attack framework based on the Mean Teacher, which serves as a temporally smoothed feature reference. With this smoothed reference, we further direct semantic consistency between the early-layer activations in the student and those of the semantically rich teacher by feature distillation. By anchoring perturbation synthesis to the semantically salient early intermediate blocks within the generator based on empirical findings, our method guides progressive adversarial perturbation on regions that substantially enhance adversarial transferability. We conduct extensive experiments over diverse models, domains and tasks to demonstrate consistent improvements relative to state-of-the-art generative attacks, comprehensively evaluated using conventional metrics and our newly proposed Accidental Correction Rate (ACR). 

**Abstract (ZH)**: 生成式对抗攻击训练一种白盒代理模型上的扰动生成器，并 subsequently 应用于未见过的黑盒受害者模型。与迭代攻击相比，这些方法在推理时效率更高、更具扩展性和迁移性；然而，现有研究尚未充分利用生成模型的表征能力以保留和利用语义信息。具体来说，生成器的中间激活包含了丰富的语义特征——物体边界和粗略形状——这些特征目前被严重忽视，限制了扰动与对齐物体显著区域的能力，后者对于对抗迁移性至关重要。为了解决这一问题，我们引入了一种基于 Mean Teacher 的语义结构感知攻击框架，它作为时间平滑的特征参考。借助这种平滑的参考，我们进一步通过特征蒸馏在学生模型的早期层激活和语义丰富的教师模型的激活之间引导语义一致性。基于实证发现，我们的方法将扰动合成锚定在生成器内的语义显著早期中间块上，从而逐步对那些显著增强对抗迁移性的区域施加对抗扰动。我们在不同的模型、领域和任务上进行了广泛的实验，以展示相对于最先进的生成式攻击的一致改进，并全面使用常规指标和我们提出的新颖的意外纠正率（ACR）进行评估。 

---
# Smart-LLaMA-DPO: Reinforced Large Language Model for Explainable Smart Contract Vulnerability Detection 

**Title (ZH)**: Smart-LLaMA-DPO：可解释的智能合约漏洞检测增强大型语言模型 

**Authors**: Lei Yu, Zhirong Huang, Hang Yuan, Shiqi Cheng, Li Yang, Fengjun Zhang, Chenjie Shen, Jiajia Ma, Jingyuan Zhang, Junyi Lu, Chun Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2506.18245)  

**Abstract**: Smart contract vulnerability detection remains a major challenge in blockchain security. Existing vulnerability detection methods face two main issues: (1) Existing datasets lack comprehensive coverage and high-quality explanations for preference learning. (2) Large language models (LLMs) often struggle with accurately interpreting specific concepts in smart contract security. Empirical analysis shows that even after continual pre-training (CPT) and supervised fine-tuning (SFT), LLMs may misinterpret the execution order of state changes, resulting in incorrect explanations despite making correct detection decisions. To address these challenges, we propose Smart-LLaMA-DPO based on LLaMA-3.1-8B. We construct a comprehensive dataset covering four major vulnerability types and machine-unauditable vulnerabilities, including precise labels, explanations, and locations for SFT, as well as high-quality and low-quality output pairs for Direct Preference Optimization (DPO). Second, we perform CPT using large-scale smart contract to enhance the LLM's understanding of specific security practices in smart contracts. Futhermore, we conduct SFT with our comprehensive dataset. Finally, we apply DPO, leveraging human feedback and a specially designed loss function that increases the probability of preferred explanations while reducing the likelihood of non-preferred outputs. We evaluate Smart-LLaMA-DPO on four major vulnerability types: reentrancy, timestamp dependence, integer overflow/underflow, and delegatecall, as well as machine-unauditable vulnerabilities. Our method significantly outperforms state-of-the-art baselines, with average improvements of 10.43% in F1 score and 7.87% in accuracy. Moreover, both LLM evaluation and human evaluation confirm that our method generates more correct, thorough, and clear explanations. 

**Abstract (ZH)**: 基于LLaMA-3.1-8B的Smart-LLaMA-DPO：智能合约漏洞检测的新方法 

---
# Quantum-Classical Hybrid Quantized Neural Network 

**Title (ZH)**: 量子-经典混合量化神经网络 

**Authors**: Wenxin Li, Chuan Wang, Hongdong Zhu, Qi Gao, Yin Ma, Hai Wei, Kai Wen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18240)  

**Abstract**: Here in this work, we present a novel Quadratic Binary Optimization (QBO) model for quantized neural network training, enabling the use of arbitrary activation and loss functions through spline interpolation. We introduce Forward Interval Propagation (FIP), a method designed to tackle the challenges of non-linearity and the multi-layer composite structure in neural networks by discretizing activation functions into linear subintervals. This approach preserves the universal approximation properties of neural networks while allowing complex nonlinear functions to be optimized using quantum computers, thus broadening their applicability in artificial intelligence. We provide theoretical upper bounds on the approximation error and the number of Ising spins required, by deriving the sample complexity of the empirical risk minimization problem, from an optimization perspective. A significant challenge in solving the associated Quadratic Constrained Binary Optimization (QCBO) model on a large scale is the presence of numerous constraints. When employing the penalty method to handle these constraints, tuning a large number of penalty coefficients becomes a critical hyperparameter optimization problem, increasing computational complexity and potentially affecting solution quality. To address this, we employ the Quantum Conditional Gradient Descent (QCGD) algorithm, which leverages quantum computing to directly solve the QCBO problem. We prove the convergence of QCGD under a quantum oracle with randomness and bounded variance in objective value, as well as under limited precision constraints in the coefficient matrix. Additionally, we provide an upper bound on the Time-To-Solution for the QCBO solving process. Experimental results using a coherent Ising machine (CIM) demonstrate a 94.95% accuracy on the Fashion MNIST classification task, with only 1.1-bit precision. 

**Abstract (ZH)**: 一种基于插值的量化神经网络训练的二次二进制优化模型 

---
# AdapThink: Adaptive Thinking Preferences for Reasoning Language Model 

**Title (ZH)**: AdapThink: 自适应思维偏好 reasoning 语言模型 

**Authors**: Xu Wan, Wei Wang, Wenyue Xu, Wotao Yin, Jie Song, Mingyang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.18237)  

**Abstract**: Reinforcement Learning (RL)-based post-training has significantly advanced the complex reasoning capabilities of language models, fostering sophisticated self-reflection processes. However, this ``slow thinking'' paradigm presents a critical challenge to reasoning efficiency: models may expend excessive computation on simple questions and shift reasoning prematurely for complex ones. Previous mechanisms typically rely on static length budgets or predefined rules, lacking the adaptability for varying question complexities and models' evolving capabilities. To this end, we propose AdapThink, an adaptive post-training framework designed to induce more efficient thinking while maintaining the performance of reasoning language models. Specifically, AdapThink incorporates two key mechanisms: 1) A group-relative reward function that leverages model confidence and response's characteristic to dynamically adjust the preference of reflection-related transition words without resorting to a fixed length preference. 2) A diversity-aware sampling mechanism that balances the training group's solution accuracy with reasoning diversity via an entropy-guided score. Experiments on several mathematical reasoning datasets with DeepSeek-distilled models demonstrate AdapThink's advantages in enabling adaptive reasoning patterns and mitigating the inefficiencies. 

**Abstract (ZH)**: 基于强化学习的自适应后训练框架AdapThink显著提升了语言模型的复杂推理能力，促进精细的自我反思过程。然而，这种“缓慢思考”范式对推理效率提出了关键挑战：模型可能在简单问题上过度计算，并在复杂问题上过早推理。先前机制通常依赖于静态长度预算或预定义规则，缺乏适应不同问题复杂度和模型演进能力的能力。为此，我们提出AdapThink，一种设计用于引导更高效思考并保持推理语言模型性能的自适应后训练框架。具体而言，AdapThink包含两大关键机制：1）基于组相对奖励函数，利用模型自信度和响应特性动态调整与反思相关的转换词偏好，而不依赖于固定长度偏好。2）一种兼顾训练组解的准确性和推理多样性的多样性感知采样机制，通过熵引导评分进行平衡。实验结果显示，AdapThink在促进自适应推理模式和减轻效率损失方面具有优势。 

---
# Make It Efficient: Dynamic Sparse Attention for Autoregressive Image Generation 

**Title (ZH)**: 提升效率：自回归图像生成中的动态稀疏注意力机制 

**Authors**: Xunzhi Xiang, Qi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18226)  

**Abstract**: Autoregressive conditional image generation models have emerged as a dominant paradigm in text-to-image synthesis. These methods typically convert images into one-dimensional token sequences and leverage the self-attention mechanism, which has achieved remarkable success in natural language processing, to capture long-range dependencies, model global context, and ensure semantic coherence. However, excessively long contexts during inference lead to significant memory overhead caused by KV-cache and computational delays. To alleviate these challenges, we systematically analyze how global semantics, spatial layouts, and fine-grained textures are formed during inference, and propose a novel training-free context optimization method called Adaptive Dynamic Sparse Attention (ADSA). Conceptually, ADSA dynamically identifies historical tokens crucial for maintaining local texture consistency and those essential for ensuring global semantic coherence, thereby efficiently streamlining attention computation. Additionally, we introduce a dynamic KV-cache update mechanism tailored for ADSA, reducing GPU memory consumption during inference by approximately $50\%$. Extensive qualitative and quantitative experiments demonstrate the effectiveness and superiority of our approach in terms of both generation quality and resource efficiency. 

**Abstract (ZH)**: 自回归条件图像生成模型已成为文本到图像合成的主导 paradigm。这些方法通常将图像转换为一维令牌序列，并利用在自然语言处理中取得显著成功的自注意力机制来捕捉长距离依赖关系、建模全局上下文并确保语义连贯性。然而，在推理过程中过长的上下文导致由于 KV 缓存和计算延迟引起的显著内存开销。为缓解这些问题，我们系统地分析了推理过程中全局语义、空间布局和细粒度纹理是如何形成的，并提出了一种名为自适应动态稀疏注意力（ADSA）的新型无训练上下文优化方法。ADSA 概念上动态识别对保持局部纹理一致性至关重要的历史令牌以及对确保全局语义连贯性至关重要的令牌，从而有效地简化了注意力计算。此外，我们还引入了一种针对 ADSA 的动态 KV 缓存更新机制，在推理过程中将 GPU 内存消耗降低了约 50%。广泛的质量和定量试验表明，与生成质量和资源效率相关的方面，我们的方法都具有有效的优越性。 

---
# These are Not All the Features You are Looking For: A Fundamental Bottleneck In Supervised Pretraining 

**Title (ZH)**: 这些未必都是您在寻找的特征：监督预训练中的根本瓶颈 

**Authors**: Xingyu Alice Yang, Jianyu Zhang, Léon Bottou  

**Link**: [PDF](https://arxiv.org/pdf/2506.18221)  

**Abstract**: Transfer learning is a cornerstone of modern machine learning, promising a way to adapt models pretrained on a broad mix of data to new tasks with minimal new data. However, a significant challenge remains in ensuring that transferred features are sufficient to handle unseen datasets, amplified by the difficulty of quantifying whether two tasks are "related". To address these challenges, we evaluate model transfer from a pretraining mixture to each of its component tasks, assessing whether pretrained features can match the performance of task-specific direct training. We identify a fundamental limitation in deep learning models -- an "information saturation bottleneck" -- where networks fail to learn new features once they encode similar competing features during training. When restricted to learning only a subset of key features during pretraining, models will permanently lose critical features for transfer and perform inconsistently on data distributions, even components of the training mixture. Empirical evidence from published studies suggests that this phenomenon is pervasive in deep learning architectures -- factors such as data distribution or ordering affect the features that current representation learning methods can learn over time. This study suggests that relying solely on large-scale networks may not be as effective as focusing on task-specific training, when available. We propose richer feature representations as a potential solution to better generalize across new datasets and, specifically, present existing methods alongside a novel approach, the initial steps towards addressing this challenge. 

**Abstract (ZH)**: 迁移学习是现代机器学习的基石， promise 了一种通过在广泛数据上预训练模型，以最少的新数据适应新任务的方法。然而，确保转移特征足以处理未见过的数据集仍然是一个重大挑战，特别是量化两个任务是否“相关”的难度。为应对这些挑战，我们评估了从预训练混合模型到其各个组件任务的模型迁移，考察预训练特征是否能匹配任务特定直接训练的性能。我们识别出深度学习模型的一个基本局限性——“信息饱和瓶颈”——其中，网络在训练过程中一旦编码了类似的竞争特征，就会无法学习新的特征。当模型仅在预训练过程中学习关键特征的子集时，它们会永久性地失去对于迁移至关重要的特征，并在数据分布上表现不一致，即使是在预训练数据混合的组成部分上。已发表研究的实证证据表明，这一现象在深度学习架构中普遍存在——因素如数据分布或排序会影响当前表示学习方法随时间能够学习的特征。本研究建议，在可用时，仅依赖大规模网络可能不如专注于任务特定训练有效。我们提出了更丰富的特征表示作为更好泛化到新数据集的潜在解决方案，并具体介绍了现有方法和一个新颖方法——初期步骤以应对这一挑战。 

---
# Cross-Architecture Knowledge Distillation (KD) for Retinal Fundus Image Anomaly Detection on NVIDIA Jetson Nano 

**Title (ZH)**: 跨架构知识蒸馏（KD）在NVIDIA Jetson Nano上的视网膜 fundus 图像异常检测 

**Authors**: Berk Yilmaz, Aniruddh Aiyengar  

**Link**: [PDF](https://arxiv.org/pdf/2506.18220)  

**Abstract**: Early and accurate identification of retinal ailments is crucial for averting ocular decline; however, access to dependable diagnostic devices is not often available in low-resourced settings. This project proposes to solve that by developing a lightweight, edge-device deployable disease classifier using cross-architecture knowledge distilling. We first train a high-capacity vision transformer (ViT) teacher model, pre-trained using I-JEPA self-supervised learning, to classify fundus images into four classes: Normal, Diabetic Retinopathy, Glaucoma, and Cataract. We kept an Internet of Things (IoT) focus when compressing to a CNN-based student model for deployment in resource-limited conditions, such as the NVIDIA Jetson Nano. This was accomplished using a novel framework which included a Partitioned Cross-Attention (PCA) projector, a Group-Wise Linear (GL) projector, and a multi-view robust training method. The teacher model has 97.4 percent more parameters than the student model, with it achieving 89 percent classification with a roughly 93 percent retention of the teacher model's diagnostic performance. The retention of clinical classification behavior supports our method's initial aim: compression of the ViT while retaining accuracy. Our work serves as an example of a scalable, AI-driven triage solution for retinal disorders in under-resourced areas. 

**Abstract (ZH)**: 早期准确识别 retina 疾病对于预防视觉衰退至关重要；然而，在低资源环境中可靠的诊断设备往往难以获得。本项目旨在通过跨架构知识蒸馏开发轻量级边缘设备可部署的疾病分类器来解决这一问题。我们首先训练一个高容量的视觉变换器（ViT）教师模型，使用 I-JEPA 自监督学习进行预训练，将视网膜图像分类为四类：正常、糖尿病性视网膜病变、青光眼和白内障。在压缩至基于 CNN 的学生模型进行部署时，我们保持了对物联网 (IoT) 的关注，例如在 NVIDIA Jetson Nano 这类资源受限条件下。这通过一个创新框架实现，该框架包括分区交叉注意 (PCA) 投影器、组内线性 (GL) 投影器和多视图鲁棒训练方法。教师模型的参数比学生模型多 97.4 倍，而学生模型的分类准确率达到 89%，保留了教师模型诊断性能的约 93%。临床分类行为的保留支持了我们方法的初步目标：在保持准确性的前提下压缩 ViT。我们的工作为资源匮乏地区的眼底疾病提供了一种可扩展的、基于 AI 的分诊解决方案。 

---
# Deep Learning-based Alignment Measurement in Knee Radiographs 

**Title (ZH)**: 基于深度学习的膝关节X线片对齐测量 

**Authors**: Zhisen Hu, Dominic Cullen, Peter Thompson, David Johnson, Chang Bian, Aleksei Tiulpin, Timothy Cootes, Claudia Lindner  

**Link**: [PDF](https://arxiv.org/pdf/2506.18209)  

**Abstract**: Radiographic knee alignment (KA) measurement is important for predicting joint health and surgical outcomes after total knee replacement. Traditional methods for KA measurements are manual, time-consuming and require long-leg radiographs. This study proposes a deep learning-based method to measure KA in anteroposterior knee radiographs via automatically localized knee anatomical landmarks. Our method builds on hourglass networks and incorporates an attention gate structure to enhance robustness and focus on key anatomical features. To our knowledge, this is the first deep learning-based method to localize over 100 knee anatomical landmarks to fully outline the knee shape while integrating KA measurements on both pre-operative and post-operative images. It provides highly accurate and reliable anatomical varus/valgus KA measurements using the anatomical tibiofemoral angle, achieving mean absolute differences ~1° when compared to clinical ground truth measurements. Agreement between automated and clinical measurements was excellent pre-operatively (intra-class correlation coefficient (ICC) = 0.97) and good post-operatively (ICC = 0.86). Our findings demonstrate that KA assessment can be automated with high accuracy, creating opportunities for digitally enhanced clinical workflows. 

**Abstract (ZH)**: 基于深度学习的前后位膝关节对线测量方法 

---
# Multimodal Fusion SLAM with Fourier Attention 

**Title (ZH)**: Fourier注意力融合多模态SLAM 

**Authors**: Youjie Zhou, Guofeng Mei, Yiming Wang, Yi Wan, Fabio Poiesi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18204)  

**Abstract**: Visual SLAM is particularly challenging in environments affected by noise, varying lighting conditions, and darkness. Learning-based optical flow algorithms can leverage multiple modalities to address these challenges, but traditional optical flow-based visual SLAM approaches often require significant computational this http URL overcome this limitation, we propose FMF-SLAM, an efficient multimodal fusion SLAM method that utilizes fast Fourier transform (FFT) to enhance the algorithm efficiency. Specifically, we introduce a novel Fourier-based self-attention and cross-attention mechanism to extract features from RGB and depth signals. We further enhance the interaction of multimodal features by incorporating multi-scale knowledge distillation across modalities. We also demonstrate the practical feasibility of FMF-SLAM in real-world scenarios with real time performance by integrating it with a security robot by fusing with a global positioning module GNSS-RTK and global Bundle Adjustment. Our approach is validated using video sequences from TUM, TartanAir, and our real-world datasets, showcasing state-of-the-art performance under noisy, varying lighting, and dark this http URL code and datasets are available at this https URL. 

**Abstract (ZH)**: 视觉SLAM在受噪声、变化光照和黑暗影响的环境中特别具有挑战性。基于学习的光流算法可以通过利用多种模态来应对这些挑战，但传统的基于光流的视觉SLAM方法往往需要大量计算资源以克服这一限制。为了解决这个问题，我们提出了一种名为FMF-SLAM的高效多模态融合SLAM方法，该方法利用快速傅里叶变换（FFT）以提高算法效率。具体而言，我们引入了一种基于傅里叶的自注意力和跨注意力机制来从RGB和深度信号中提取特征，并通过跨模态的多尺度知识蒸馏增强了多模态特征的交互性。我们还通过将其与GNSS-RTK全球定位模块和全局 bundle 调整相结合的方式，实现在实时场景中的实际可行性。我们的方法使用来自TUM、TartanAir以及我们自己的现实世界数据集的视频序列进行验证，在噪声、变化光照和黑暗条件下展示了最先进的性能。相关代码和数据集可在以下网址获取。 

---
# Prompt Engineering Techniques for Mitigating Cultural Bias Against Arabs and Muslims in Large Language Models: A Systematic Review 

**Title (ZH)**: 针对大型语言模型中针对阿拉伯人和穆斯林的文化偏见缓解的提示工程技术系统综述 

**Authors**: Bushra Asseri, Estabrag Abdelaziz, Areej Al-Wabil  

**Link**: [PDF](https://arxiv.org/pdf/2506.18199)  

**Abstract**: Large language models have demonstrated remarkable capabilities across various domains, yet concerns about cultural bias - particularly towards Arabs and Muslims - pose significant ethical challenges by perpetuating harmful stereotypes and marginalization. Despite growing recognition of bias in LLMs, prompt engineering strategies specifically addressing Arab and Muslim representation remain understudied. This mixed-methods systematic review examines such techniques, offering evidence-based guidance for researchers and practitioners. Following PRISMA guidelines and Kitchenham's systematic review methodology, we analyzed 8 empirical studies published between 2021-2024 investigating bias mitigation strategies. Our findings reveal five primary prompt engineering approaches: cultural prompting, affective priming, self-debiasing techniques, structured multi-step pipelines, and parameter-optimized continuous prompts. Although all approaches show potential for reducing bias, effectiveness varied substantially across studies and bias types. Evidence suggests that certain bias types may be more resistant to prompt-based mitigation than others. Structured multi-step pipelines demonstrated the highest overall effectiveness, achieving up to 87.7% reduction in bias, though they require greater technical expertise. Cultural prompting offers broader accessibility with substantial effectiveness. These results underscore the accessibility of prompt engineering for mitigating cultural bias without requiring access to model parameters. The limited number of studies identified highlights a significant research gap in this critical area. Future research should focus on developing culturally adaptive prompting techniques, creating Arab and Muslim-specific evaluation resources, and integrating prompt engineering with complementary debiasing methods to address deeper stereotypes while maintaining model utility. 

**Abstract (ZH)**: 大型语言模型在各领域展现了令人瞩目的能力，但对其文化偏见的担忧——尤其是针对阿拉伯人和穆斯林——提出了重大的伦理挑战，这些偏见会导致有害的刻板印象和边缘化。尽管人们对大型语言模型中的偏见越来越认识到了，但专门针对阿拉伯人和穆斯林代表性问题的提示工程技术研究仍然不足。本混合法系统性回顾研究考察了此类技术，为研究者和实践者提供了基于证据的指导。遵循PRISMA指南和Kitchenham的系统性回顾方法，我们分析了2021-2024年间发表的8篇 empirical 研究，探讨了偏见缓解策略。研究结果揭示了五种主要的提示工程技术方法：文化提示、情感启动、自我去偏技术、结构化多步骤管道和参数优化连续提示。虽然所有方法都显示出减少偏见的潜力，但在不同研究和偏见类型中的有效性差异显著。证据表明，某些偏见类型可能比其他类型更难以通过提示技术缓解。结构化多步骤管道显示出最高的整体有效性，可降低高达87.7%的偏见，但这需要更高的技术专业知识。文化提示在提供广泛可及性的同时，显示出显著的有效性。这些结果强调了提示工程技术在不访问模型参数的情况下缓解文化偏见的可及性。确定的研究数量有限突显了这一关键领域存在显著的研究缺口。未来的研究应侧重于开发文化适应性提示技术、创建阿拉伯人和穆斯林特定的评估资源，并将提示工程技术与互补的去偏技术集成，以应对更深层次的刻板印象，同时保持模型的实用性。 

---
# Two Sonification Methods for the MindCube 

**Title (ZH)**: 基于MindCube的两种听觉化方法 

**Authors**: Fangzheng Liu, Lancelot Blanchard, Don D. Haddad, Joseph A. Paradiso  

**Link**: [PDF](https://arxiv.org/pdf/2506.18196)  

**Abstract**: In this work, we explore the musical interface potential of the MindCube, an interactive device designed to study emotions. Embedding diverse sensors and input devices, this interface resembles a fidget cube toy commonly used to help users relieve their stress and anxiety. As such, it is a particularly well-suited controller for musical systems that aim to help with emotion regulation. In this regard, we present two different mappings for the MindCube, with and without AI. With our generative AI mapping, we propose a way to infuse meaning within a latent space and techniques to navigate through it with an external controller. We discuss our results and propose directions for future work. 

**Abstract (ZH)**: MindCube在情绪调节音乐系统中的潜力探索：基于AI的生成映射研究 

---
# Wisdom of Crowds Through Myopic Self-Confidence Adaptation 

**Title (ZH)**: 众人的智慧通过短视自信心适应 

**Authors**: Giacomo Como, Fabio Fagnani, Anton Proskurnikov  

**Link**: [PDF](https://arxiv.org/pdf/2506.18195)  

**Abstract**: The wisdom of crowds is an umbrella term for phenomena suggesting that the collective judgment or decision of a large group can be more accurate than the individual judgments or decisions of the group members. A well-known example illustrating this concept is the competition at a country fair described by Galton, where the median value of the individual guesses about the weight of an ox resulted in an astonishingly accurate estimate of the actual weight. This phenomenon resembles classical results in probability theory and relies on independent decision-making. The accuracy of the group's final decision can be significantly reduced if the final agents' opinions are driven by a few influential agents.
In this paper, we consider a group of agents who initially possess uncorrelated and unbiased noisy measurements of a common state of the world. Assume these agents iteratively update their estimates according to a simple non-Bayesian learning rule, commonly known in mathematical sociology as the French-DeGroot dynamics or iterative opinion pooling. As a result of this iterative distributed averaging process, each agent arrives at an asymptotic estimate of the state of the world, with the variance of this estimate determined by the matrix of weights the agents assign to each other. Every agent aims at minimizing the variance of her asymptotic estimate of the state of the world; however, such variance is also influenced by the weights allocated by other agents. To achieve the best possible estimate, the agents must then solve a game-theoretic, multi-objective optimization problem defined by the available sets of influence weights. We characterize both the Pareto frontier and the set of Nash equilibria in the resulting game. Additionally, we examine asynchronous best-response dynamics for the group of agents and prove their convergence to the set of strict Nash equilibria. 

**Abstract (ZH)**: 群体的智慧是群体的集体判断或决策比个体判断或决策更准确的一种现象的统称。一个著名的例子是由高尔顿描述的乡村 fair 上的竞赛，个体对牛的重量的猜测中位数惊人的准确地估计了实际重量。这一现象类似于概率论中的经典结果，并依赖于独立的决策。如果最终决策者的意见受到少数有影响力的决策者的影响，群体最终决策的准确性会显著降低。

在本文中，我们考虑一群初始时具有关于世界公共状态的不相关且无偏的噪声测量值的代理。假设这些代理根据一个简单的非贝叶斯学习规则迭代地更新其估算值，这种学习规则在数学社会学中广为人知，称为法国-德格鲁 Physics 或迭代意见聚合。由于这一迭代分布式平均过程，每个代理都会达到世界状态的渐进估算，估算的方差由代理相互分配的权重矩阵决定。每个代理都力求最小化其关于世界状态的渐进估算的方差；然而，这种方差也受到其他代理分配权重的影响。为了获得最佳估算，代理必须解一个由可用影响力的权重集合定义的多重目标博弈论优化问题。我们界定了由此产生的博弈的帕累托前沿和纳什均衡集。此外，我们研究了代理组的异步最佳反应动态，并证明它们收敛到严格的纳什均衡集。 

---
# DeInfoReg: A Decoupled Learning Framework for Better Training Throughput 

**Title (ZH)**: DeInfoReg：一种解耦学习框架，以提高训练吞吐量 

**Authors**: Zih-Hao Huang, You-Teng Lin, Hung-Hsuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18193)  

**Abstract**: This paper introduces Decoupled Supervised Learning with Information Regularization (DeInfoReg), a novel approach that transforms a long gradient flow into multiple shorter ones, thereby mitigating the vanishing gradient problem. Integrating a pipeline strategy, DeInfoReg enables model parallelization across multiple GPUs, significantly improving training throughput. We compare our proposed method with standard backpropagation and other gradient flow decomposition techniques. Extensive experiments on diverse tasks and datasets demonstrate that DeInfoReg achieves superior performance and better noise resistance than traditional BP models and efficiently utilizes parallel computing resources. The code for reproducibility is available at: this https URL. 

**Abstract (ZH)**: 本文介绍了去耦合监督学习与信息正则化（DeInfoReg）方法，该方法将长梯度流转换为多个较短的梯度流，从而减轻梯度消失问题。通过集成管道策略，DeInfoReg 支持在多块 GPU 上进行模型并行化，显著提高训练吞吐量。我们将提出的算法与标准反向传播以及其他梯度流分解技术进行比较。在多种任务和数据集上的广泛实验表明，DeInfoReg 较之传统反向传播模型在性能上更优且具有更好的噪声抗性，并能有效利用并行计算资源。代码可供复现下载：https://github.com/Qwen-Model/DeInfoReg 

---
# Call Me Maybe: Enhancing JavaScript Call Graph Construction using Graph Neural Networks 

**Title (ZH)**: 呼叫我吧：使用图神经网络增强JavaScript调用图构建 

**Authors**: Masudul Hasan Masud Bhuiyan, Gianluca De Stefano, Giancarlo Pellegrino, Cristian-Alexandru Staicu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18191)  

**Abstract**: Static analysis plays a key role in finding bugs, including security issues. A critical step in static analysis is building accurate call graphs that model function calls in a program. However, due to hard-to-analyze language features, existing call graph construction algorithms for JavaScript are neither sound nor complete. Prior work shows that even advanced solutions produce false edges and miss valid ones. In this work, we assist these tools by identifying missed call edges. Our main idea is to frame the problem as link prediction on full program graphs, using a rich representation with multiple edge types. Our approach, GRAPHIA, leverages recent advances in graph neural networks to model non-local relationships between code elements. Concretely, we propose representing JavaScript programs using a combination of syntactic- and semantic-based edges. GRAPHIA can learn from imperfect labels, including static call edges from existing tools and dynamic edges from tests, either from the same or different projects. Because call graphs are sparse, standard machine learning metrics like ROC are not suitable. Instead, we evaluate GRAPHIA by ranking function definitions for each unresolved call site. We conduct a large-scale evaluation on 50 popular JavaScript libraries with 163K call edges (150K static and 13K dynamic). GRAPHIA builds program graphs with 6.6M structural and 386K semantic edges. It ranks the correct target as the top candidate in over 42% of unresolved cases and within the top 5 in 72% of cases, reducing the manual effort needed for analysis. Our results show that learning-based methods can improve the recall of JavaScript call graph construction. To our knowledge, this is the first work to apply GNN-based link prediction to full multi-file program graphs for interprocedural analysis. 

**Abstract (ZH)**: 静态分析在发现包括安全问题在内的漏洞中起着关键作用。构建准确的调用图是静态分析中建模程序中函数调用的一个关键步骤。然而，由于难以分析的语言特性，现有的 JavaScript 调用图构建算法既不完全也不可靠。先前的工作表明，即使是高级解决方案也会产生虚假边并忽略有效的边。在这项工作中，我们通过识别遗漏的调用边来协助这些工具。我们的主要思路是将问题重新表述为程序图上的链接预测问题，使用带有多种边类型的丰富表示。我们的方法 GRAPHIA 利用图神经网络的最新进展来建模代码元素之间的非局部关系。具体而言，我们提出使用基于语法和语义的边来表示 JavaScript 程序。GRAPHIA 可以从不完美的标签中学习，包括现有工具中的静态调用边和测试中的动态边，既可以来自同一项目，也可以来自不同的项目。由于调用图稀疏，标准的机器学习评价度量如 ROC 并不适用。相反，我们通过排名每个未解决的调用位置的函数定义来评估 GRAPHIA。我们在 50 个流行的 JavaScript 库上进行了大规模评估，这些库包含 163K 个调用边（其中150K 个是静态边，13K 个是动态边）。GRAPHIA 构建了具有 6.6M 结构边和 386K 语义边的程序图。在超过 42% 的未解决案例中，GRAPHIA 将正确的目标作为首选项，在 72% 的案例中将其置于前五位，从而减少了分析所需的手动努力。我们的结果显示，基于学习的方法可以提高 JavaScript 调用图构建的召回率。据我们所知，这是首次将基于 GNN 的链接预测应用于用于跨过程分析的完整多文件程序图的工作。 

---
# CareLab at #SMM4H-HeaRD 2025: Insomnia Detection and Food Safety Event Extraction with Domain-Aware Transformers 

**Title (ZH)**: CareLab在#SMM4H-HeaRD 2025上的失眠检测与食品安全事件提取：基于领域意识的变换器模型 

**Authors**: Zihan Liang, Ziwen Pan, Sumon Kanti Dey, Azra Ismail  

**Link**: [PDF](https://arxiv.org/pdf/2506.18185)  

**Abstract**: This paper presents our system for the SMM4H-HeaRD 2025 shared tasks, specifically Task 4 (Subtasks 1, 2a, and 2b) and Task 5 (Subtasks 1 and 2). Task 4 focused on detecting mentions of insomnia in clinical notes, while Task 5 addressed the extraction of food safety events from news articles. We participated in all subtasks and report key findings across them, with particular emphasis on Task 5 Subtask 1, where our system achieved strong performance-securing first place with an F1 score of 0.958 on the test set. To attain this result, we employed encoder-based models (e.g., RoBERTa), alongside GPT-4 for data augmentation. This paper outlines our approach, including preprocessing, model architecture, and subtask-specific adaptations 

**Abstract (ZH)**: 本文介绍了我们参加SMM4H-HeaRD 2025共享任务的系统，具体包括Task 4（子任务1、2a和2b）和Task 5（子任务1和2）。Task 4专注于在临床笔记中检测失眠的提及，而Task 5则处理从新闻文章中抽取食品安全事件的问题。我们参与了所有子任务，并报告了其中的关键发现，特别是在Task 5子任务1中，我们的系统取得出色性能，在测试集上获得了F1分数0.958，位居榜首。为了达到这一结果，我们使用了基于编码器的模型（如RoBERTa），并结合了GPT-4进行数据增强。本文概述了我们的方法，包括预处理、模型架构和针对每个子任务的具体适应性。 

---
# STACT-Time: Spatio-Temporal Cross Attention for Cine Thyroid Ultrasound Time Series Classification 

**Title (ZH)**: STACT-Time: 空间-时间交叉注意力在 cine 甲状腺超声时间序列分类中的应用 

**Authors**: Irsyad Adam, Tengyue Zhang, Shrayes Raman, Zhuyu Qiu, Brandon Taraku, Hexiang Feng, Sile Wang, Ashwath Radhachandran, Shreeram Athreya, Vedrana Ivezic, Peipei Ping, Corey Arnold, William Speier  

**Link**: [PDF](https://arxiv.org/pdf/2506.18172)  

**Abstract**: Thyroid cancer is among the most common cancers in the United States. Thyroid nodules are frequently detected through ultrasound (US) imaging, and some require further evaluation via fine-needle aspiration (FNA) biopsy. Despite its effectiveness, FNA often leads to unnecessary biopsies of benign nodules, causing patient discomfort and anxiety. To address this, the American College of Radiology Thyroid Imaging Reporting and Data System (TI-RADS) has been developed to reduce benign biopsies. However, such systems are limited by interobserver variability. Recent deep learning approaches have sought to improve risk stratification, but they often fail to utilize the rich temporal and spatial context provided by US cine clips, which contain dynamic global information and surrounding structural changes across various views. In this work, we propose the Spatio-Temporal Cross Attention for Cine Thyroid Ultrasound Time Series Classification (STACT-Time) model, a novel representation learning framework that integrates imaging features from US cine clips with features from segmentation masks automatically generated by a pretrained model. By leveraging self-attention and cross-attention mechanisms, our model captures the rich temporal and spatial context of US cine clips while enhancing feature representation through segmentation-guided learning. Our model improves malignancy prediction compared to state-of-the-art models, achieving a cross-validation precision of 0.91 (plus or minus 0.02) and an F1 score of 0.89 (plus or minus 0.02). By reducing unnecessary biopsies of benign nodules while maintaining high sensitivity for malignancy detection, our model has the potential to enhance clinical decision-making and improve patient outcomes. 

**Abstract (ZH)**: 甲状腺癌是美国最常见的癌症之一。通过超声（US）成像经常检测到甲状腺结节，其中一些需要通过细针穿刺抽吸（FNA）活检进一步评估。尽管FNA有效，但它常常导致良性结节的不必要的活检，给患者造成不适和焦虑。为了解决这一问题，美国放射学院开发了甲状腺成像报告和数据系统（TI-RADS）以减少不必要的活检。然而，这类系统受到观察者间变异性的限制。最近的深度学习方法试图改进风险分级，但它们往往未能充分利用US动画剪辑所提供的丰富的时间和空间上下文，这些剪辑包含了各种视角下动态的全局信息和周围结构的变化。在本工作中，我们提出了一种新型表示学习框架——基于US动画剪辑的空间-时间交叉注意力甲状腺超声时间序列分类模型（STACT-Time），该模型将US动画剪辑的成像特征与预先训练模型自动生成的分割掩码特征相结合。利用自我注意力和交叉注意力机制，我们的模型捕捉了US动画剪辑丰富的时空上下文并通过对分割指导的学习增强特征表示。与最先进的模型相比，我们的模型提高了恶性预测的准确性，实现交叉验证精确度0.91（±0.02）和F1分数0.89（±0.02）。通过减少不必要的良性结节活检同时保持对恶性检测的高敏感性，我们的模型有潜力增强临床决策并改善患者预后。 

---
# Understanding Reasoning in Thinking Language Models via Steering Vectors 

**Title (ZH)**: 通过导向向量理解思考语言模型中的推理 

**Authors**: Constantin Venhoff, Iván Arcuschin, Philip Torr, Arthur Conmy, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2506.18167)  

**Abstract**: Recent advances in large language models (LLMs) have led to the development of thinking language models that generate extensive internal reasoning chains before producing responses. While these models achieve improved performance, controlling their reasoning processes remains challenging. This work presents a steering approach for thinking LLMs by analyzing and manipulating specific reasoning behaviors in DeepSeek-R1-Distill models. Through a systematic experiment on 500 tasks across 10 diverse categories, we identify several reasoning behaviors exhibited by thinking models, including expressing uncertainty, generating examples for hypothesis validation, and backtracking in reasoning chains. We demonstrate that these behaviors are mediated by linear directions in the model's activation space and can be controlled using steering vectors. By extracting and applying these vectors, we provide a method to modulate specific aspects of the model's reasoning process, such as its tendency to backtrack or express uncertainty. Our approach offers practical tools for steering reasoning processes in thinking models in a controlled and interpretable manner. We validate our steering method using two DeepSeek-R1-Distill models, demonstrating consistent control across different model architectures. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）推动了思考型LLMs的发展，这些模型在生成广泛内部推理链后才产生响应。尽管这些模型在性能上有所提升，但控制其推理过程仍然颇具挑战。本文通过分析和操控DeepSeek-R1-Distill模型中的特定推理行为，提出了一种引导方法。通过在10个不同类别下的500个任务上进行系统的实验，我们识别出思考模型表现出的几种推理行为，包括表达不确定性、为假设验证生成例子以及推理链中的回溯。我们证明这些行为由模型激活空间中的线性方向介导，并可通过引导向量进行控制。通过提取和应用这些向量，我们提供了一种调节模型推理过程特定方面的办法，例如回溯倾向或不确定性表达。我们的方法提供了在受控和可解释的方式下引导思考型模型推理过程的实用工具。我们使用两个DeepSeek-R1-Distill模型验证了我们的引导方法，展示了不同模型架构之间的一致控制能力。 

---
# Non-equilibrium Annealed Adjoint Sampler 

**Title (ZH)**: 非平衡 annealed adjoint 采样器 

**Authors**: Jaemoo Choi, Yongxin Chen, Molei Tao, Guan-Horng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18165)  

**Abstract**: Recently, there has been significant progress in learning-based diffusion samplers, which aim to sample from a given unnormalized density. These methods typically follow one of two paradigms: (i) formulating sampling as an unbiased stochastic optimal control (SOC) problem using a canonical reference process, or (ii) refining annealed path measures through importance-weighted sampling. Although annealing approaches have advantages in guiding samples toward high-density regions, reliance on importance sampling leads to high variance and limited scalability in practice. In this paper, we introduce the \textbf{Non-equilibrium Annealed Adjoint Sampler (NAAS)}, a novel SOC-based diffusion sampler that leverages annealed reference dynamics without resorting to importance sampling. NAAS employs a lean adjoint system inspired by adjoint matching, enabling efficient and scalable training. We demonstrate the effectiveness of our approach across a range of tasks, including sampling from classical energy landscapes and molecular Boltzmann distribution. 

**Abstract (ZH)**: 非平衡退火伴随采样器（NAAS）：一种基于伴随的退火差分采样方法 

---
# QuranMorph: Morphologically Annotated Quranic Corpus 

**Title (ZH)**: QuranMorph: 基于形态标注的古兰经语料库 

**Authors**: Diyam Akra, Tymaa Hammouda, Mustafa Jarrar  

**Link**: [PDF](https://arxiv.org/pdf/2506.18148)  

**Abstract**: We present the QuranMorph corpus, a morphologically annotated corpus for the Quran (77,429 tokens). Each token in the QuranMorph was manually lemmatized and tagged with its part-of-speech by three expert linguists. The lemmatization process utilized lemmas from Qabas, an Arabic lexicographic database linked with 110 lexicons and corpora of 2 million tokens. The part-of-speech tagging was performed using the fine-grained SAMA/Qabas tagset, which encompasses 40 tags. As shown in this paper, this rich lemmatization and POS tagset enabled the QuranMorph corpus to be inter-linked with many linguistic resources. The corpus is open-source and publicly available as part of the SinaLab resources at (this https URL) 

**Abstract (ZH)**: 我们呈现了QuranMorph语料库，这是一个包含77,429个词素的希伯来语语料库，每条词素都由三位专家语言学家手工词根还原并标注了词性。词根还原过程利用了与110个词典和200万词的语料库链接的Qabas阿拉伯语词表数据库中的词根。词性标注使用了细粒度的SAMA/Qabas标签集，包含40个标签。如本文所示，丰富的词根还原和词性标注标签集使得QuranMorph语料库能够与许多语言资源相互链接。该语料库是开源的，并作为SinaLab资源的一部分公开可用（请参见此链接：[这个链接](this https URL)）。 

---
# Routing Mamba: Scaling State Space Models with Mixture-of-Experts Projection 

**Title (ZH)**: Routing 猫鼬：通过混合专家投影扩展状态空间模型 

**Authors**: Zheng Zhan, Liliang Ren, Shuohang Wang, Liyuan Liu, Yang Liu, Yeyun Gong, Yanzhi Wang, Yelong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18145)  

**Abstract**: Linear State Space Models (SSMs) offer remarkable performance gains in efficient sequence modeling, with constant inference-time computation and memory complexity. Recent advances, such as Mamba, further enhance SSMs with input-dependent gating and hardware-aware implementations, positioning them as strong alternatives to Transformers for long sequence modeling. However, efficiently scaling the expressive power of SSMs, particularly with Mixture of Experts (MoE), remains challenging, as naive integration attempts often falter or degrade performance. In this work, we introduce Routing Mamba (RoM), a novel approach that scales SSM parameters using sparse mixtures of linear projection experts. By sharing routing decisions between projection layers and lightweight sub-modules within Mamba across experts, RoM leverages synergies among linear projection experts for effective and efficient sparse scaling of Mamba layers. At a scale of 1.3B active parameters (10B total) and 16K training sequence length, RoM achieves language modeling performance equivalent to a dense Mamba model requiring over 2.3x more active parameters, and demonstrates consistent perplexity across context lengths. Experimental results further show RoM effectively scales hybrid language models, yielding a 23% FLOPS saving compared to dense Mamba scaling for similar performance. 

**Abstract (ZH)**: 基于线性状态空间模型的路由Mamba（RoM）：一种有效的稀疏扩展方法 

---
# AI Harmonizer: Expanding Vocal Expression with a Generative Neurosymbolic Music AI System 

**Title (ZH)**: AI Harmonizer: 通过生成神经符号音乐AI系统扩展 vocal 表达 

**Authors**: Lancelot Blanchard, Cameron Holt, Joseph A. Paradiso  

**Link**: [PDF](https://arxiv.org/pdf/2506.18143)  

**Abstract**: Vocals harmonizers are powerful tools to help solo vocalists enrich their melodies with harmonically supportive voices. These tools exist in various forms, from commercially available pedals and software to custom-built systems, each employing different methods to generate harmonies. Traditional harmonizers often require users to manually specify a key or tonal center, while others allow pitch selection via an external keyboard-both approaches demanding some degree of musical expertise. The AI Harmonizer introduces a novel approach by autonomously generating musically coherent four-part harmonies without requiring prior harmonic input from the user. By integrating state-of-the-art generative AI techniques for pitch detection and voice modeling with custom-trained symbolic music models, our system arranges any vocal melody into rich choral textures. In this paper, we present our methods, explore potential applications in performance and composition, and discuss future directions for real-time implementations. While our system currently operates offline, we believe it represents a significant step toward AI-assisted vocal performance and expressive musical augmentation. We release our implementation on GitHub. 

**Abstract (ZH)**: 声乐和声器是帮助独唱歌手丰富旋律、添加和声支持声音的强大工具。这些工具以多种形式存在，从商业可购的踏板和软件到自定义系统，每种工具都采用了不同方法来生成和声。传统和声器通常需要用户手动指定一个键或调中心，而其他一些工具则允许通过外部键盘选择音高——这两种方法都需要一定的音乐专业知识。AI 和声器通过自主生成音乐连贯的四部和声，无需用户先提供和声输入，引入了一种新的方法。通过将最先进的生成AI技术与音高检测和声模型结合，并结合自定义训练的符号音乐模型，我们的系统将任何 vocal 熔调排列成丰富的合唱纹理。在本文中，我们介绍了我们的方法，探讨了在表演和创作中的潜在应用，并讨论了实时实现的未来方向。尽管我们的系统目前是离线运行的，但我们认为它代表了AI辅助声乐表演和表达性音乐增强的重要一步。我们在GitHub上发布了我们的实现。 

---
# Sparse Feature Coactivation Reveals Composable Semantic Modules in Large Language Models 

**Title (ZH)**: 稀疏特征共激活揭示大规模语言模型中的可组合语义模块 

**Authors**: Ruixuan Deng, Xiaoyang Hu, Miles Gilberti, Shane Storks, Aman Taxali, Mike Angstadt, Chandra Sripada, Joyce Chai  

**Link**: [PDF](https://arxiv.org/pdf/2506.18141)  

**Abstract**: We identify semantically coherent, context-consistent network components in large language models (LLMs) using coactivation of sparse autoencoder (SAE) features collected from just a handful of prompts. Focusing on country-relation tasks, we show that ablating semantic components for countries and relations changes model outputs in predictable ways, while amplifying these components induces counterfactual responses. Notably, composing relation and country components yields compound counterfactual outputs. We find that, whereas most country components emerge from the very first layer, the more abstract relation components are concentrated in later layers. Furthermore, within relation components themselves, nodes from later layers tend to have a stronger causal impact on model outputs. Overall, these findings suggest a modular organization of knowledge within LLMs and advance methods for efficient, targeted model manipulation. 

**Abstract (ZH)**: 我们使用稀疏自编码器（SAE）特征的共激活，在大型语言模型（LLMs）中识别出语义一致且上下文一致的网络组件。聚焦于国家关系任务，我们发现移除国家和关系的语义组件会导致可预测的模型输出变化，而放大这些组件则会导致反事实响应。值得注意的是，关系和国家组件的组合会产生复合的反事实输出。我们发现，大多数国家组件源自模型的第一层，而更为抽象的关系组件则集中在较晚的层中。此外，在关系组件内部，较晚层的节点对模型输出的影响更强。总体而言，这些发现表明LLMs内部知识的模块化组织，并促进了高效、目标导向的模型操控方法的发展。 

---
# $ϕ^{\infty}$: Clause Purification, Embedding Realignment, and the Total Suppression of the Em Dash in Autoregressive Language Models 

**Title (ZH)**: $ϕ^{\infty}$: 子句净化、嵌入重新对齐及自回归语言模型中破折号的完全抑制 

**Authors**: Bugra Kilictas, Faruk Alpay  

**Link**: [PDF](https://arxiv.org/pdf/2506.18129)  

**Abstract**: We identify a critical vulnerability in autoregressive transformer language models where the em dash token induces recursive semantic drift, leading to clause boundary hallucination and embedding space entanglement. Through formal analysis of token-level perturbations in semantic lattices, we demonstrate that em dash insertion fundamentally alters the model's latent representations, causing compounding errors in long-form generation. We propose a novel solution combining symbolic clause purification via the phi-infinity operator with targeted embedding matrix realignment. Our approach enables total suppression of problematic tokens without requiring model retraining, while preserving semantic coherence through fixed-point convergence guarantees. Experimental validation shows significant improvements in generation consistency and topic maintenance. This work establishes a general framework for identifying and mitigating token-level vulnerabilities in foundation models, with immediate implications for AI safety, model alignment, and robust deployment of large language models in production environments. The methodology extends beyond punctuation to address broader classes of recursive instabilities in neural text generation systems. 

**Abstract (ZH)**: 我们在自回归变压器语言模型中识别出一个关键漏洞，其中破折号标记引发递归语义漂移，导致从句边界幻觉和嵌入空间纠缠。通过对语义格中token级扰动的正式分析，我们证明破折号插入从根本上改变了模型的潜在表示，导致长文本生成中的累积错误。我们提出了一个结合φ-∞算子进行符号从句净化和目标嵌入矩阵重对齐的新型解决方案。我们的方法能够在无需模型重新训练的情况下完全抑制问题token，同时通过定点收敛保证维持语义连贯性。实验验证显示生成一致性与话题维持方面的显著改进。本研究建立了识别和缓解基础模型中token级漏洞的一般框架，对AI安全性、模型对齐及大规模语言模型在生产环境中的鲁棒部署具有即时影响。该方法超越标点符号，用以解决神经文本生成系统中更广泛的递归不稳定性问题。 

---
# Conceptualization, Operationalization, and Measurement of Machine Companionship: A Scoping Review 

**Title (ZH)**: 机器同伴概念化、操作化及测量：一项范围性回顾 

**Authors**: Jaime Banks, Zhixin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.18119)  

**Abstract**: The notion of machine companions has long been embedded in social-technological imaginaries. Recent advances in AI have moved those media musings into believable sociality manifested in interfaces, robotic bodies, and devices. Those machines are often referred to colloquially as "companions" yet there is little careful engagement of machine companionship (MC) as a formal concept or measured variable. This PRISMA-guided scoping review systematically samples, surveys, and synthesizes current scholarly works on MC (N = 71; 2017-2025), to that end. Works varied widely in considerations of MC according to guiding theories, dimensions of a-priori specified properties (subjectively positive, sustained over time, co-active, autotelic), and in measured concepts (with more than 50 distinct measured variables). WE ultimately offer a literature-guided definition of MC as an autotelic, coordinated connection between human and machine that unfolds over time and is subjectively positive. 

**Abstract (ZH)**: 机器伴侣概念长久以来根植于社会技术幻想之中。近年来，AI的发展将这些媒体想象物表现成了可信的社会互动，体现在界面、机器人身体和设备中。这些机器通常被通俗地称为“伴侣”，但鲜有严谨地将机器伴侣ship（MC）作为一个正式的概念或量化的变量进行探讨。本研究遵循PRISMA指南，系统性地抽样、调查和综述了2017-2025年间关于MC的现有学术作品（N=71），旨在定义MC。作品在指导理论、先验规定属性（主观积极、持续时间、协同作用、终末学的）以及测量概念方面差异广泛（超过50个不同的测量变量）。最终，我们提供了一个基于文献界定的MC定义，即一种随时间展开的、主观上积极的人机协同连接。 

---
# Mental Health Equity in LLMs: Leveraging Multi-Hop Question Answering to Detect Amplified and Silenced Perspectives 

**Title (ZH)**: LLMs中的心理健康公平性：利用多跳问答检测被放大和沉默的观点 

**Authors**: Batool Haider, Atmika Gorti, Aman Chadha, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2506.18116)  

**Abstract**: Large Language Models (LLMs) in mental healthcare risk propagating biases that reinforce stigma and harm marginalized groups. While previous research identified concerning trends, systematic methods for detecting intersectional biases remain limited. This work introduces a multi-hop question answering (MHQA) framework to explore LLM response biases in mental health discourse. We analyze content from the Interpretable Mental Health Instruction (IMHI) dataset across symptom presentation, coping mechanisms, and treatment approaches. Using systematic tagging across age, race, gender, and socioeconomic status, we investigate bias patterns at demographic intersections. We evaluate four LLMs: Claude 3.5 Sonnet, Jamba 1.6, Gemma 3, and Llama 4, revealing systematic disparities across sentiment, demographics, and mental health conditions. Our MHQA approach demonstrates superior detection compared to conventional methods, identifying amplification points where biases magnify through sequential reasoning. We implement two debiasing techniques: Roleplay Simulation and Explicit Bias Reduction, achieving 66-94% bias reductions through few-shot prompting with BBQ dataset examples. These findings highlight critical areas where LLMs reproduce mental healthcare biases, providing actionable insights for equitable AI development. 

**Abstract (ZH)**: 大型语言模型（LLMs）在心理健康护理中的应用可能会传播强化刻板印象并伤害边缘化群体的偏见。虽然先前研究识别了令人担忧的趋势，但系统的交叉偏见检测方法仍有限。本研究引入多跳问答（MHQA）框架以探索LLM在心理健康对话中的响应偏见。我们分析了可解释心理健康指令（IMHI）数据集中的症状表现、应对机制和治疗方法内容。通过系统标记年龄、种族、性别和社会经济地位，我们研究了人口统计交叉点的偏见模式。我们评估了四种LLM：Claude 3.5 Sonnet、Jamba 1.6、Gemma 3和Llama 4，揭示了情感、人口统计和心理健康状况方面的系统性差异。我们的MHQA方法在偏见检测方面显示出优于传统方法的能力，识别了通过序列推理放大偏见的点。我们实施了两种去偏方法：角色扮演模拟和显式偏见减少，通过使用BBQ数据集示例的少量提示实现了66-94%的偏见减少。这些发现揭示了LLM在心理健康护理偏见再现方面的关键领域，为公平的人工智能开发提供了可操作的见解。 

---
# RL for Reasoning by Adaptively Revealing Rationales 

**Title (ZH)**: 基于自适应揭示推理依据的强化学习方法 

**Authors**: Mohammad Hossein Amani, Aryo Lotfi, Nicolas Mario Baldwin, Samy Bengio, Mehrdad Farajtabar, Emmanuel Abbe, Robert West  

**Link**: [PDF](https://arxiv.org/pdf/2506.18110)  

**Abstract**: We propose that reinforcement learning (RL) from partial expert demonstrations is not merely a training heuristic, but a promising framework for solving complex sequence generation tasks. Supervised fine-tuning (SFT) relies on dense ground-truth labels, which become increasingly costly as sequence length grows. RL, on the other hand, struggles with sparse rewards and a combinatorially large output space. We address this by introducing adaptive backtracking (AdaBack), a per-sample curriculum learning algorithm that reveals only a partial prefix of the target output during training. The supervision length is adjusted dynamically for each sample based on the model's past reward signal, allowing it to incrementally learn to complete reasoning chains by conditioning on correct partial solutions. We investigate this intermediate regime between SFT and RL and argue that per-sample curriculum learning is more than a trade-off between efficiency and generality, it can succeed in tasks with long sequences of latent dependencies where SFT and RL both fail to generalize. Using a synthetic task with latent parity constraints, we show that our adaptive curriculum over partial answers reliably solves problems that are otherwise intractable. On mathematical reasoning benchmarks (MATH, GSM8k), we find that curriculum learning enables models to solve problems that RL alone cannot, acquiring new reasoning capabilities through incremental exposure to partial solutions. 

**Abstract (ZH)**: 从部分专家演示中学习的强化学习：一种解决复杂序列生成任务的有前途的框架 

---
# ShareGPT-4o-Image: Aligning Multimodal Models with GPT-4o-Level Image Generation 

**Title (ZH)**: ShareGPT-4o-Image：基于GPT-4o级图像生成的多模态模型对齐 

**Authors**: Junying Chen, Zhenyang Cai, Pengcheng Chen, Shunian Chen, Ke Ji, Xidong Wang, Yunjin Yang, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18095)  

**Abstract**: Recent advances in multimodal generative models have unlocked photorealistic, instruction-aligned image generation, yet leading systems like GPT-4o-Image remain proprietary and inaccessible. To democratize these capabilities, we present ShareGPT-4o-Image, the first dataset comprising 45K text-to-image and 46K text-and-image-to-image data, all synthesized using GPT-4o's image generation capabilities for distilling its advanced image generation abilities. Leveraging this dataset, we develop Janus-4o, a multimodal large language model capable of both text-to-image and text-and-image-to-image generation. Janus-4o not only significantly improves text-to-image generation over its predecessor, Janus-Pro, but also newly supports text-and-image-to-image generation. Notably, it achieves impressive performance in text-and-image-to-image generation from scratch, using only 91K synthetic samples and 6 hours of training on an 8 A800-GPU machine. We hope the release of ShareGPT-4o-Image and Janus-4o will foster open research in photorealistic, instruction-aligned image generation. 

**Abstract (ZH)**: Recent advances in多模态生成模型的Recent进展在真实感、指令对齐图像生成方面的突破，然而像GPT-4o-Image这样的领先系统仍然保持专有和不可访问状态。为了普及这些能力，我们提出ShareGPT-4o-Image，这是一个包含45,000个文本到图像和46,000个文本和图像到图像数据集，所有数据均使用GPT-4o的图像生成能力以提炼其先进的图像生成能力。利用该数据集，我们开发了Janus-4o，一个双模态大语言模型，能够进行文本到图像和文本和图像到图像的生成。Janus-4o不仅在文本到图像生成方面显著优于其 predecessor Janus-Pro，而且还新增了文本和图像到图像生成能力。值得注意的是，它能够从零开始在真实感、指令对齐图像生成方面取得出色表现，仅需91,000个合成样本和6小时的8 A800-GPU机器训练时间。我们希望ShareGPT-4o-Image和Janus-4o的发布能够促进真实感、指令对齐图像生成的开放研究。 

---
# RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation 

**Title (ZH)**: RoboTwin 2.0：一种具有强大领域随机化的大规模数据生成器和基准测试，用于稳健的双臂机器人操作 

**Authors**: Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Qiwei Liang, Zixuan Li, Xianliang Lin, Yiheng Ge, Zhenyu Gu, Weiliang Deng, Yubin Guo, Tian Nian, Xuanbing Xie, Qiangyu Chen, Kailun Su, Tianling Xu, Guodong Liu, Mengkang Hu, Huan-ang Gao, Kaixuan Wang, Zhixuan Liang, Yusen Qin, Xiaokang Yang, Ping Luo, Yao Mu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18088)  

**Abstract**: Simulation-based data synthesis has emerged as a powerful paradigm for enhancing real-world robotic manipulation. However, existing synthetic datasets remain insufficient for robust bimanual manipulation due to two challenges: (1) the lack of an efficient, scalable data generation method for novel tasks, and (2) oversimplified simulation environments that fail to capture real-world complexity. We present RoboTwin 2.0, a scalable simulation framework that enables automated, large-scale generation of diverse and realistic data, along with unified evaluation protocols for dual-arm manipulation. We first construct RoboTwin-OD, a large-scale object library comprising 731 instances across 147 categories, each annotated with semantic and manipulation-relevant labels. Building on this foundation, we develop an expert data synthesis pipeline that combines multimodal large language models (MLLMs) with simulation-in-the-loop refinement to generate task-level execution code automatically. To improve sim-to-real transfer, RoboTwin 2.0 incorporates structured domain randomization along five axes: clutter, lighting, background, tabletop height and language instructions, thereby enhancing data diversity and policy robustness. We instantiate this framework across 50 dual-arm tasks spanning five robot embodiments, and pre-collect over 100,000 domain-randomized expert trajectories. Empirical results show a 10.9% gain in code generation success and improved generalization to novel real-world scenarios. A VLA model fine-tuned on our dataset achieves a 367% relative improvement (42.0% vs. 9.0%) on unseen scene real-world tasks, while zero-shot models trained solely on our synthetic data achieve a 228% relative gain, highlighting strong generalization without real-world supervision. We release the data generator, benchmark, dataset, and code to support scalable research in robust bimanual manipulation. 

**Abstract (ZH)**: 基于模拟的数据合成已成为增强现实世界双臂机器人操作能力的强大范式。然而，现有的合成数据集由于两个挑战仍不足以实现鲁棒的双臂操作：（1）缺乏高效、可扩展的新任务数据生成方法，（2）简化过的模拟环境未能捕捉到现实世界的复杂性。我们提出RoboTwin 2.0，一种可扩展的模拟框架，能够实现自动化、大规模地生成多样化和真实的双臂操作数据，并提供了统一的评估协议。我们首先构建了RoboTwin-OD，一个包含731个实例、147类的大规模对象库，每个实例都附有语义和操作相关标签。在此基础上，我们开发了一种专家级数据合成管道，结合多模态大规模语言模型（MLLMs）与闭环模拟精炼，以自动生成任务级执行代码。为提高模拟到现实的转移能力，RoboTwin 2.0整合了沿五个轴向的结构化域随机化：杂乱环境、光照、背景、桌面高度和语言指令，从而增强了数据多样化和策略鲁棒性。我们在50个双臂任务中实例化了此框架，覆盖五个不同机器人实体，并预先收集了超过10万条结构化随机化专家轨迹。实验证明，相较于基准，代码生成成功率提高了10.9%，并且在处理新型现实世界场景时表现出更好的泛化能力。在我们的数据集上微调的VLA模型在未见过的场景任务上取得了367%的相对性能提升（42.0% vs. 9.0%），仅利用我们合成数据训练的零样本模型也实现了228%的相对性能提升，突出了无监督下的强大泛化能力。我们发布了数据生成器、基准、数据集和代码，以支持鲁棒双臂操作的可扩展研究。 

---
# Federated Learning-Based Data Collaboration Method for Enhancing Edge Cloud AI System Security Using Large Language Models 

**Title (ZH)**: 基于联邦学习的数据协作方法：增强边缘云AI系统安全性以利用大型语言模型 

**Authors**: Huaiying Luo, Cheng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2506.18087)  

**Abstract**: With the widespread application of edge computing and cloud systems in AI-driven applications, how to maintain efficient performance while ensuring data privacy has become an urgent security issue. This paper proposes a federated learning-based data collaboration method to improve the security of edge cloud AI systems, and use large-scale language models (LLMs) to enhance data privacy protection and system robustness. Based on the existing federated learning framework, this method introduces a secure multi-party computation protocol, which optimizes the data aggregation and encryption process between distributed nodes by using LLM to ensure data privacy and improve system efficiency. By combining advanced adversarial training techniques, the model enhances the resistance of edge cloud AI systems to security threats such as data leakage and model poisoning. Experimental results show that the proposed method is 15% better than the traditional federated learning method in terms of data protection and model robustness. 

**Abstract (ZH)**: 基于联邦学习的数据协作方法在AI驱动的边缘云计算系统中提升数据安全性和系统健壮性 

---
# Distributionally robust minimization in meta-learning for system identification 

**Title (ZH)**: 元学习中的分布鲁棒最小化在系统识别中的应用 

**Authors**: Matteo Rufolo, Dario Piga, Marco Forgione  

**Link**: [PDF](https://arxiv.org/pdf/2506.18074)  

**Abstract**: Meta learning aims at learning how to solve tasks, and thus it allows to estimate models that can be quickly adapted to new scenarios. This work explores distributionally robust minimization in meta learning for system identification. Standard meta learning approaches optimize the expected loss, overlooking task variability. We use an alternative approach, adopting a distributionally robust optimization paradigm that prioritizes high-loss tasks, enhancing performance in worst-case scenarios. Evaluated on a meta model trained on a class of synthetic dynamical systems and tested in both in-distribution and out-of-distribution settings, the proposed approach allows to reduce failures in safety-critical applications. 

**Abstract (ZH)**: 元学习旨在学习如何解决任务，从而能够估算出可以快速适应新场景的模型。本文探讨了在系统识别中的元学习中的分布鲁棒最小化方法。标准的元学习方法优化期望损失，忽略了任务的变异性。我们采用一种替代方法，采用分布鲁棒优化范式，优先考虑高损失任务，从而在最坏情况下提高性能。在基于一类合成动力学系统训练的元模型上进行训练并在分布内和分布外设置下进行测试，所提出的方法能够减少在安全关键应用中的失败情况。 

---
# Multimodal Medical Image Binding via Shared Text Embeddings 

**Title (ZH)**: 基于共享文本嵌入的多模态医疗图像融合 

**Authors**: Yunhao Liu, Suyang Xi, Shiqi Liu, Hong Ding, Chicheng Jin, Chenxi Yang, Junjun He, Yiqing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18072)  

**Abstract**: Medical image analysis increasingly relies on the integration of multiple imaging modalities to capture complementary anatomical and functional information, enabling more accurate diagnosis and treatment planning. Achieving aligned feature representations across these diverse modalities is therefore important for effective multimodal analysis. While contrastive language-image pre-training (CLIP) and its variant have enabled image-text alignments, they require explicitly paired data between arbitrary two modalities, which is difficult to acquire in medical contexts. To address the gap, we present Multimodal Medical Image Binding with Text (M\textsuperscript{3}Bind), a novel pre-training framework that enables seamless alignment of multiple medical imaging modalities through a shared text representation space without requiring explicit paired data between any two medical image modalities. Specifically, based on the insight that different images can naturally bind with text, M\textsuperscript{3}Bind first fine-tunes pre-trained CLIP-like image-text models to align their modality-specific text embedding space while preserving their original image-text alignments. Subsequently, we distill these modality-specific text encoders into a unified model, creating a shared text embedding space. Experiments on X-ray, CT, retina, ECG, and pathological images on multiple downstream tasks demonstrate that M\textsuperscript{3}Bind achieves state-of-the-art performance in zero-shot, few-shot classification and cross-modal retrieval tasks compared to its CLIP-like counterparts. These results validate M\textsuperscript{3}Bind's effectiveness in achieving cross-image-modal alignment for medical analysis. 

**Abstract (ZH)**: 多模态医学图像与文本的绑定（M³Bind） 

---
# MUPA: Towards Multi-Path Agentic Reasoning for Grounded Video Question Answering 

**Title (ZH)**: MUPA: 向多路径主体推理的地基视频问答迈进 

**Authors**: Jisheng Dang, Huilin Song, Junbin Xiao, Bimei Wang, Han Peng, Haoxuan Li, Xun Yang, Meng Wang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.18071)  

**Abstract**: Grounded Video Question Answering (Grounded VideoQA) requires aligning textual answers with explicit visual evidence. However, modern multimodal models often rely on linguistic priors and spurious correlations, resulting in poorly grounded predictions. In this work, we propose MUPA, a cooperative MUlti-Path Agentic approach that unifies video grounding, question answering, answer reflection and aggregation to tackle Grounded VideoQA. MUPA features three distinct reasoning paths on the interplay of grounding and QA agents in different chronological orders, along with a dedicated reflection agent to judge and aggregate the multi-path results to accomplish consistent QA and grounding. This design markedly improves grounding fidelity without sacrificing answer accuracy. Despite using only 2B parameters, our method outperforms all 7B-scale competitors. When scaled to 7B parameters, MUPA establishes new state-of-the-art results, with Acc@GQA of 30.3% and 47.4% on NExT-GQA and DeVE-QA respectively, demonstrating MUPA' effectiveness towards trustworthy video-language understanding. Our code is available in this https URL. 

**Abstract (ZH)**: 基于视觉的视频问答（Grounded VideoQA）要求将文本回答与明确的视觉证据对齐。然而，现代多模态模型往往依赖于语言先验和虚假的相关性，导致预测与可视化证据契合度不高。在本工作中，我们提出了一种新的协作多路径代理方法MUPA，统一了视频定位、问答、回答反思和结果聚合，以解决基于视觉的视频问答问题。MUPA包含三个不同的推理路径，这些路径在不同时间顺序上处理定位和问答代理之间的相互作用，并配备了一个专门的反思代理，用于判断和聚合多路径结果以实现一致的问答和定位。这种设计显著提高了定位准确性，同时保持答案的准确性。尽管仅使用20亿参数，我们的方法在所有70亿参数规模的竞争者中表现更优。当扩展到70亿参数时，MUPA建立了新的 state-of-the-art 结果，分别在NExT-GQA和DeVE-QA数据集上实现了30.3%和47.4%的Acc@GQA，证明了MUPA对于可信赖的视频-语言理解的有效性。我们的代码可通过此链接获取：https://github.com/alibaba/Qwen 

---
# Mechanistic Interpretability in the Presence of Architectural Obfuscation 

**Title (ZH)**: Architecture-Obfuscation Present Mechanistic Interpretability 

**Authors**: Marcos Florencio, Thomas Barton  

**Link**: [PDF](https://arxiv.org/pdf/2506.18053)  

**Abstract**: Architectural obfuscation - e.g., permuting hidden-state tensors, linearly transforming embedding tables, or remapping tokens - has recently gained traction as a lightweight substitute for heavyweight cryptography in privacy-preserving large-language-model (LLM) inference. While recent work has shown that these techniques can be broken under dedicated reconstruction attacks, their impact on mechanistic interpretability has not been systematically studied. In particular, it remains unclear whether scrambling a network's internal representations truly thwarts efforts to understand how the model works, or simply relocates the same circuits to an unfamiliar coordinate system. We address this gap by analyzing a GPT-2-small model trained from scratch with a representative obfuscation map. Assuming the obfuscation map is private and the original basis is hidden (mirroring an honest-but-curious server), we apply logit-lens attribution, causal path-patching, and attention-head ablation to locate and manipulate known circuits. Our findings reveal that obfuscation dramatically alters activation patterns within attention heads yet preserves the layer-wise computational graph. This disconnect hampers reverse-engineering of user prompts: causal traces lose their alignment with baseline semantics, and token-level logit attributions become too noisy to reconstruct. At the same time, feed-forward and residual pathways remain functionally intact, suggesting that obfuscation degrades fine-grained interpretability without compromising top-level task performance. These results establish quantitative evidence that architectural obfuscation can simultaneously (i) retain global model behaviour and (ii) impede mechanistic analyses of user-specific content. By mapping where interpretability breaks down, our study provides guidance for future privacy defences and for robustness-aware interpretability tooling. 

**Abstract (ZH)**: 建筑混淆 – 例如，打乱隐藏状态张量、线性变换嵌入表，或重映射标记 – 最近被用作大型语言模型（LLM）推理中 heavyweight 摘要隐私保护的轻量化替代方案。尽管近期研究显示这些技术在专门的重建攻击下可被破解，但它们对机制可解释性的影响尚未系统研究。特别是，尚不清楚扰乱网络的内部表示是否会真正阻碍对模型工作原理的理解，还是仅将相同的电路重新定位到一个不熟悉的坐标系统中。我们通过分析一个从头训练且带有代表性混淆映射的 GPT-2-small 模型来弥补这一空白。假设混淆映射是私有的且原本基底是隐藏的（类似于诚实但好奇的服务器），我们应用 logits 镜像归因、因果路径修补和注意力头消融来定位和操作已知电路。我们的发现显示，混淆极大地改变了注意力头内的激活模式，但保留了逐层计算图。这种分离阻碍了用户提示的反向工程：因果轨迹与基线语义失去了对齐，且标记级别的 logits 归因变得太嘈杂而无法重建。同时，前馈和残差路径仍然保持功能完好，表明混淆降低了细粒度可解释性，但未损害高级任务性能。这些结果建立了一定量证据，表明从结构混淆可以在 (i) 保留全局模型行为的同时 (ii) 阻碍用户特定内容的机制分析。通过映射可解释性失效的位置，我们的研究为未来的隐私防御和鲁棒性感知可解释性工具提供指导。 

---
# The Democratic Paradox in Large Language Models' Underestimation of Press Freedom 

**Title (ZH)**: 大型语言模型对新闻自由低估的民主悖论 

**Authors**: I. Loaiza, R. Vestrelli, A. Fronzetti Colladon, R. Rigobon  

**Link**: [PDF](https://arxiv.org/pdf/2506.18045)  

**Abstract**: As Large Language Models (LLMs) increasingly mediate global information access for millions of users worldwide, their alignment and biases have the potential to shape public understanding and trust in fundamental democratic institutions, such as press freedom. In this study, we uncover three systematic distortions in the way six popular LLMs evaluate press freedom in 180 countries compared to expert assessments of the World Press Freedom Index (WPFI). The six LLMs exhibit a negative misalignment, consistently underestimating press freedom, with individual models rating between 71% to 93% of countries as less free. We also identify a paradoxical pattern we term differential misalignment: LLMs disproportionately underestimate press freedom in countries where it is strongest. Additionally, five of the six LLMs exhibit positive home bias, rating their home countries' press freedoms more favorably than would be expected given their negative misalignment with the human benchmark. In some cases, LLMs rate their home countries between 7% to 260% more positively than expected. If LLMs are set to become the next search engines and some of the most important cultural tools of our time, they must ensure accurate representations of the state of our human and civic rights globally. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）日益成为全球数百万用户获取信息的中介，它们的对齐情况和偏差有可能影响公众对新闻自由等基本民主机构的理解和信任。在本研究中，我们发现了六种流行LLM在评估180个国家的新闻自由时与世界新闻自由指数（WPFI）专家评估之间存在三种系统性失真。这六种LLM表现出负面的对齐偏差，一致地低估了新闻自由，各模型分别将71%到93%的国家评为较为不自由。我们还发现了一种我们称之为差异性失真的悖论模式：LLM在评估新闻自由最强的国家时显著低估了新闻自由。此外，六种LLM中有五种表现出向本国偏爱的正偏差，对其本国的新闻自由给予了比其负面的对齐偏差与人类基准相比更高的评价。在某些情况下，这些LLM对其本国的新闻自由的评价比预期高出7%到260%。如果LLM被设定成为下一代搜索引擎和我们这个时代最重要的文化工具之一，它们必须确保对全球人类和公民权利现状的准确呈现。 

---
# Pathwise Explanation of ReLU Neural Networks 

**Title (ZH)**: ReLU神经网络的路径解释 

**Authors**: Seongwoo Lim, Won Jo, Joohyung Lee, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18037)  

**Abstract**: Neural networks have demonstrated a wide range of successes, but their ``black box" nature raises concerns about transparency and reliability. Previous research on ReLU networks has sought to unwrap these networks into linear models based on activation states of all hidden units. In this paper, we introduce a novel approach that considers subsets of the hidden units involved in the decision making path. This pathwise explanation provides a clearer and more consistent understanding of the relationship between the input and the decision-making process. Our method also offers flexibility in adjusting the range of explanations within the input, i.e., from an overall attribution input to particular components within the input. Furthermore, it allows for the decomposition of explanations for a given input for more detailed explanations. Experiments demonstrate that our method outperforms others both quantitatively and qualitatively. 

**Abstract (ZH)**: 神经网络在广泛领域取得了成功，但其“黑箱”性质引发了透明度和可靠性方面的担忧。前人在ReLU网络上的研究试图通过所有隐藏单元的激活状态将这些网络展开为线性模型。在本文中，我们提出了一种新的方法，考虑参与决策路径的隐藏单元子集。这种路径解释方法为输入与决策过程之间的关系提供了更清晰、更一致的理解。该方法还提供了在输入范围内调整解释范围的灵活性，即从整体归因输入到输入的具体组件。此外，它还允许对给定输入的解释进行分解，以提供更详细的解释。实验表明，我们的方法在定量和定性方面都优于其他方法。 

---
# Pre-Trained LLM is a Semantic-Aware and Generalizable Segmentation Booster 

**Title (ZH)**: 预训练大语言模型是一个具备语义意识和泛化能力的分割增强器 

**Authors**: Fenghe Tang, Wenxin Ma, Zhiyang He, Xiaodong Tao, Zihang Jiang, S. Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.18034)  

**Abstract**: With the advancement of Large Language Model (LLM) for natural language processing, this paper presents an intriguing finding: a frozen pre-trained LLM layer can process visual tokens for medical image segmentation tasks. Specifically, we propose a simple hybrid structure that integrates a pre-trained, frozen LLM layer within the CNN encoder-decoder segmentation framework (LLM4Seg). Surprisingly, this design improves segmentation performance with a minimal increase in trainable parameters across various modalities, including ultrasound, dermoscopy, polypscopy, and CT scans. Our in-depth analysis reveals the potential of transferring LLM's semantic awareness to enhance segmentation tasks, offering both improved global understanding and better local modeling capabilities. The improvement proves robust across different LLMs, validated using LLaMA and DeepSeek. 

**Abstract (ZH)**: 随着大型语言模型（LLM）在自然语言处理领域的进步，本论文呈现了一个有趣的发现：冻结的预训练LLM层可以处理医学图像分割任务中的视觉令牌。具体来说，我们提出了一种简单的混合结构，将预训练并冻结的LLM层集成到CNN编码-解码分割框架中（LLM4Seg）。令人惊讶的是，这种设计在各种成像模态（包括超声、皮肤镜检查、息肉检查和CT扫描）中，能以微量增加可训练参数的方式提升分割性能。深入分析表明，可以通过转移LLM的语义意识来增强分割任务，提供更好的全局理解和局部建模能力。这种改进在不同的LLM模型中都表现出鲁棒性，通过LLaMA和DeepSeek进行验证。 

---
# PP-DocBee2: Improved Baselines with Efficient Data for Multimodal Document Understanding 

**Title (ZH)**: PP-DocBee2: 提升基线模型的多模态文档理解方法与高效数据集 

**Authors**: Kui Huang, Xinrong Chen, Wenyu Lv, Jincheng Liao, Guanzhong Wang, Yi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18023)  

**Abstract**: This report introduces PP-DocBee2, an advanced version of the PP-DocBee, designed to enhance multimodal document understanding. Built on a large multimodal model architecture, PP-DocBee2 addresses the limitations of its predecessor through key technological improvements, including enhanced synthetic data quality, improved visual feature fusion strategy, and optimized inference methodologies. These enhancements yield an $11.4\%$ performance boost on internal benchmarks for Chinese business documents, and reduce inference latency by $73.0\%$ to the vanilla version. A key innovation of our work is a data quality optimization strategy for multimodal document tasks. By employing a large-scale multimodal pre-trained model to evaluate data, we apply a novel statistical criterion to filter outliers, ensuring high-quality training data. Inspired by insights into underutilized intermediate features in multimodal models, we enhance the ViT representational capacity by decomposing it into layers and applying a novel feature fusion strategy to improve complex reasoning. The source code and pre-trained model are available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: PP-DocBee2: 一种高级版的PP-DocBee，用于增强多模态文档理解 

---
# Auto-Regressive Surface Cutting 

**Title (ZH)**: 自回归表面切割 

**Authors**: Yang Li, Victor Cheung, Xinhai Liu, Yuguang Chen, Zhongjin Luo, Biwen Lei, Haohan Weng, Zibo Zhao, Jingwei Huang, Zhuo Chen, Chunchao Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.18017)  

**Abstract**: Surface cutting is a fundamental task in computer graphics, with applications in UV parameterization, texture mapping, and mesh decomposition. However, existing methods often produce technically valid but overly fragmented atlases that lack semantic coherence. We introduce SeamGPT, an auto-regressive model that generates cutting seams by mimicking professional workflows. Our key technical innovation lies in formulating surface cutting as a next token prediction task: sample point clouds on mesh vertices and edges, encode them as shape conditions, and employ a GPT-style transformer to sequentially predict seam segments with quantized 3D coordinates. Our approach achieves exceptional performance on UV unwrapping benchmarks containing both manifold and non-manifold meshes, including artist-created, and 3D-scanned models. In addition, it enhances existing 3D segmentation tools by providing clean boundaries for part decomposition. 

**Abstract (ZH)**: 表面切割是计算机图形学的基本任务，广泛应用于UV参数化、纹理映射和网格分解。然而，现有方法往往生成技术上有效但过度碎片化的马赛克图，缺乏语义连贯性。我们引入了SeamGPT，这是一种自回归模型，通过模仿专业工作流程生成切割缝合线。我们的关键技术创新在于将表面切割形式化为下一个标记预测任务：在网格顶点和边缘采样点云，将其编码为形状条件，并使用类似GPT的变压器顺序预测带有量化3D坐标的缝合段。我们的方法在包含流形和非流形网格的UV展开基准测试中表现出色，包括艺术家创作的和3D扫描的模型。此外，它增强了现有的3D分割工具，提供了清晰的部件分解边界。 

---
# ADA-DPM: A Neural Descriptors-based Adaptive Noise Point Filtering Strategy for SLAM 

**Title (ZH)**: ADA-DPM：一种基于神经描述子的自适应噪声点过滤策略用于SLAM 

**Authors**: Yongxin Shao, Binrui Wang, Aihong Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18016)  

**Abstract**: LiDAR SLAM has demonstrated significant application value in various fields, including mobile robot navigation and high-precision map construction. However, existing methods often need to make a trade-off between positioning accuracy and system robustness when faced with dynamic object interference, point cloud noise, and unstructured environments. To address this challenge, we propose an adaptive noise filtering SLAM strategy-ADA-DPM, achieving excellent preference in both aspects. We design the Dynamic Segmentation Head to predict the category of feature points belonging to dynamic points, to eliminate dynamic feature points; design the Global Importance Scoring Head to adaptively select feature points with higher contribution and features while suppressing noise interference; and construct the Cross Layer Intra-Graph Convolution Module (GLI-GCN) to fuse multi-scale neighborhood structures, thereby enhancing the discriminative ability of overlapping features. Finally, to further validate the effectiveness of our method, we tested it on several publicly available datasets and achieved outstanding results. 

**Abstract (ZH)**: LiDAR SLAM已在移动机器人导航和高精度地图构建等多个领域展示了显著的应用价值。然而，现有方法在面对动态物体干扰、点云噪声和非结构化环境时，往往需要在定位精度和系统鲁棒性之间做出权衡。为解决这一挑战，我们提出了一种自适应噪声过滤SLAM策略—ADA-DPM，在这两方面均实现了优异的表现。我们设计了动态分割头来预测特征点属于动态点的类别，以消除动态特征点；设计了全局重要性评分头以自适应选择具有更高贡献度和特征的特征点，同时抑制噪声干扰；构建了跨层内图卷积模块（GLI-GCN）以融合多尺度局部结构，从而增强重叠特征的辨别能力。最后，为了进一步验证本方法的有效性，我们在多个公开数据集上进行了测试，并取得了优异的结果。 

---
# Probing the Embedding Space of Transformers via Minimal Token Perturbations 

**Title (ZH)**: 通过最小토큰扰动探究变压器的嵌入空间 

**Authors**: Eddie Conti, Alejandro Astruc, Alvaro Parafita, Axel Brando  

**Link**: [PDF](https://arxiv.org/pdf/2506.18011)  

**Abstract**: Understanding how information propagates through Transformer models is a key challenge for interpretability. In this work, we study the effects of minimal token perturbations on the embedding space. In our experiments, we analyze the frequency of which tokens yield to minimal shifts, highlighting that rare tokens usually lead to larger shifts. Moreover, we study how perturbations propagate across layers, demonstrating that input information is increasingly intermixed in deeper layers. Our findings validate the common assumption that the first layers of a model can be used as proxies for model explanations. Overall, this work introduces the combination of token perturbations and shifts on the embedding space as a powerful tool for model interpretability. 

**Abstract (ZH)**: 理解信息在Transformer模型中的传播机制是可解释性的关键挑战。在本工作中，我们研究了最小token扰动对嵌入空间的影响。在我们的实验中，我们分析了导致最小位移的token频率，指出罕见token通常会导致更大的位移。此外，我们研究了扰动在层间传播的方式，证明了输入信息在更深的层中越来越多地混合在一起。我们的发现验证了模型早期层可以用作模型解释的代理的common假设。总体而言，本工作引入了令牌扰动和嵌入空间中的位移组合，作为模型可解释性的一个强大工具。 

---
# h-calibration: Rethinking Classifier Recalibration with Probabilistic Error-Bounded Objective 

**Title (ZH)**: h-校准：基于概率误差界目标重思考分类器再校准 

**Authors**: Wenjian Huang, Guiping Cao, Jiahao Xia, Jingkun Chen, Hao Wang, Jianguo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17968)  

**Abstract**: Deep neural networks have demonstrated remarkable performance across numerous learning tasks but often suffer from miscalibration, resulting in unreliable probability outputs. This has inspired many recent works on mitigating miscalibration, particularly through post-hoc recalibration methods that aim to obtain calibrated probabilities without sacrificing the classification performance of pre-trained models. In this study, we summarize and categorize previous works into three general strategies: intuitively designed methods, binning-based methods, and methods based on formulations of ideal calibration. Through theoretical and practical analysis, we highlight ten common limitations in previous approaches. To address these limitations, we propose a probabilistic learning framework for calibration called h-calibration, which theoretically constructs an equivalent learning formulation for canonical calibration with boundedness. On this basis, we design a simple yet effective post-hoc calibration algorithm. Our method not only overcomes the ten identified limitations but also achieves markedly better performance than traditional methods, as validated by extensive experiments. We further analyze, both theoretically and experimentally, the relationship and advantages of our learning objective compared to traditional proper scoring rule. In summary, our probabilistic framework derives an approximately equivalent differentiable objective for learning error-bounded calibrated probabilities, elucidating the correspondence and convergence properties of computational statistics with respect to theoretical bounds in canonical calibration. The theoretical effectiveness is verified on standard post-hoc calibration benchmarks by achieving state-of-the-art performance. This research offers valuable reference for learning reliable likelihood in related fields. 

**Abstract (ZH)**: 深度神经网络在众多学习任务中展现了出色的性能，但往往存在校准不足的问题，导致概率输出不可靠。为此，许多最近的工作集中在通过后处理校准方法减轻校准不足，这些方法旨在获得校准概率而不牺牲预训练模型的分类性能。本研究总结并分类了以往工作为三大类策略：直观设计方法、分箱方法以及基于理想校准形式的方法。通过理论和实践分析，我们指出了以往方法中的十个常见局限性。针对这些局限性，我们提出了一种称为h-校准的概率学习框架，理论上为标准校准构建了一个带有边界条件的等效学习形式。在此基础上，我们设计了一个简单而有效的后处理校准算法。我们的方法不仅克服了所识别的十种局限性，而且在广泛的实验中表现出色，远优于传统方法。我们进一步从理论上和实验上分析了我们的学习目标与传统适当评分规则的关系和优势。总体而言，我们提出的方法为基础校准提供了一个近似等价可微目标，揭示了计算统计与标准校准理论边界特性之间的对应性和收敛性。我们的研究成果在标准后处理校准基准测试中达到了最先进的性能，验证了其实用性。这项研究为相关领域学习可靠的似然性提供了有价值的参考。 

---
# Adapting Vision-Language Models for Evaluating World Models 

**Title (ZH)**: 适应视觉-语言模型以评估世界模型 

**Authors**: Mariya Hendriksen, Tabish Rashid, David Bignell, Raluca Georgescu, Abdelhak Lemkhenter, Katja Hofmann, Sam Devlin, Sarah Parisot  

**Link**: [PDF](https://arxiv.org/pdf/2506.17967)  

**Abstract**: World models -- generative models that simulate environment dynamics conditioned on past observations and actions -- are gaining prominence in planning, simulation, and embodied AI. However, evaluating their rollouts remains a fundamental challenge, requiring fine-grained, temporally grounded assessment of action alignment and semantic consistency -- capabilities not captured by existing metrics. Vision-Language Models (VLMs) have shown promise as automatic evaluators of generative content due to their strong multimodal reasoning abilities. Yet, their use in fine-grained, temporally sensitive evaluation tasks remains limited and requires targeted adaptation. We introduce a evaluation protocol targeting two recognition tasks -- action recognition and character recognition -- each assessed across binary, multiple-choice, and open-ended formats. To support this, we present UNIVERSE (UNIfied Vision-language Evaluator for Rollouts in Simulated Environments), a method for adapting VLMs to rollout evaluation under data and compute constraints. We conduct a large-scale study comparing full, partial, and parameter-efficient finetuning across task formats, context lengths, sampling strategies, and data compositions. The resulting unified evaluator matches the performance of task-specific baselines using a single checkpoint. Human studies confirm strong alignment with human judgments, establishing UNIVERSE as a scalable, semantics-aware evaluator for world models. 

**Abstract (ZH)**: 世界模型——条件于过往观测和行动模拟环境动力学的生成模型——在规划、仿真和具身AI中逐渐受到重视。然而，对其 rollout 的评估仍是一个基本挑战，需要对行动对齐和语义一致性进行细粒度、时间关联性的评估——这些能力现有指标尚未捕捉到。视觉-语言模型（VLMs）因其强大的多模态推理能力，在生成内容的自动评估中显示出了潜力。然而，它们在细粒度、时间敏感评估任务中的应用仍然有限，需要有针对性的适应。我们提出了一种针对两类识别任务——动作识别和角色识别——的评估协议，每类任务分别采用二选一、多选一和开放式格式进行评估。为此，我们提出了UNIVERSE（统一的视觉-语言评估器，用于模拟环境中的 rollout 评估），这是一种在数据和计算约束下使 VLMs 适应 rollout 评估的方法。我们进行了一项大规模研究，比较了不同任务格式、上下文长度、采样策略和数据组成的全面、部分和参数高效微调方法。由此产生的统一评估器使用单个检查点即可达到特定任务基线的性能。人类研究证实了与人类判断的强烈一致性，确立了UNIVERSE作为世界模型的可扩展、语义感知评估器的地位。 

---
# OmniESI: A unified framework for enzyme-substrate interaction prediction with progressive conditional deep learning 

**Title (ZH)**: 全方位ESI：一种渐进条件深度学习的统一框架用于酶-底物相互作用预测 

**Authors**: Zhiwei Nie, Hongyu Zhang, Hao Jiang, Yutian Liu, Xiansong Huang, Fan Xu, Jie Fu, Zhixiang Ren, Yonghong Tian, Wen-Bin Zhang, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.17963)  

**Abstract**: Understanding and modeling enzyme-substrate interactions is crucial for catalytic mechanism research, enzyme engineering, and metabolic engineering. Although a large number of predictive methods have emerged, they do not incorporate prior knowledge of enzyme catalysis to rationally modulate general protein-molecule features that are misaligned with catalytic patterns. To address this issue, we introduce a two-stage progressive framework, OmniESI, for enzyme-substrate interaction prediction through conditional deep learning. By decomposing the modeling of enzyme-substrate interactions into a two-stage progressive process, OmniESI incorporates two conditional networks that respectively emphasize enzymatic reaction specificity and crucial catalysis-related interactions, facilitating a gradual feature modulation in the latent space from general protein-molecule domain to catalysis-aware domain. On top of this unified architecture, OmniESI can adapt to a variety of downstream tasks, including enzyme kinetic parameter prediction, enzyme-substrate pairing prediction, enzyme mutational effect prediction, and enzymatic active site annotation. Under the multi-perspective performance evaluation of in-distribution and out-of-distribution settings, OmniESI consistently delivered superior performance than state-of-the-art specialized methods across seven benchmarks. More importantly, the proposed conditional networks were shown to internalize the fundamental patterns of catalytic efficiency while significantly improving prediction performance, with only negligible parameter increases (0.16%), as demonstrated by ablation studies on key components. Overall, OmniESI represents a unified predictive approach for enzyme-substrate interactions, providing an effective tool for catalytic mechanism cracking and enzyme engineering with strong generalization and broad applicability. 

**Abstract (ZH)**: 全方位酶-底物相互作用预测框架OmniESI：基于条件深度学习的方法 

---
# GeNIE: A Generalizable Navigation System for In-the-Wild Environments 

**Title (ZH)**: GeNIE: 一种通用的户外环境导航系统 

**Authors**: Jiaming Wang, Diwen Liu, Jizhuo Chen, Jiaxuan Da, Nuowen Qian, Tram Minh Man, Harold Soh  

**Link**: [PDF](https://arxiv.org/pdf/2506.17960)  

**Abstract**: Reliable navigation in unstructured, real-world environments remains a significant challenge for embodied agents, especially when operating across diverse terrains, weather conditions, and sensor configurations. In this paper, we introduce GeNIE (Generalizable Navigation System for In-the-Wild Environments), a robust navigation framework designed for global deployment. GeNIE integrates a generalizable traversability prediction model built on SAM2 with a novel path fusion strategy that enhances planning stability in noisy and ambiguous settings. We deployed GeNIE in the Earth Rover Challenge (ERC) at ICRA 2025, where it was evaluated across six countries spanning three continents. GeNIE took first place and achieved 79% of the maximum possible score, outperforming the second-best team by 17%, and completed the entire competition without a single human intervention. These results set a new benchmark for robust, generalizable outdoor robot navigation. We will release the codebase, pretrained model weights, and newly curated datasets to support future research in real-world navigation. 

**Abstract (ZH)**: 可靠的导航在未结构化的现实环境中仍然是体化代理面临的巨大挑战，尤其是在跨越多种地形、天气条件和传感器配置时。本文介绍了GeNIE（适用于野生环境的通用导航系统），这是一种旨在全球部署的鲁棒导航框架。GeNIE 结合了基于 SAM2 的可推广通行性预测模型和一种新颖的路径融合策略，以增强在噪声和模糊环境中的计划稳定性。我们将在ICRA 2025的Earth Rover挑战赛（ERC）中部署GeNIE，它在跨越三大洲六个国家的评估中位列第一，并取得了最高可能得分的79%，比第二名领先17%，且全程无需人工干预。这些结果为鲁棒、可推广的户外机器人导航设定了新的基准。我们将发布代码库、预训练模型权重以及新编制的数据集，以支持未来的现实世界导航研究。 

---
# Scatter-Based Innovation Propagation in Large Language Models for Multi-Stage Process Adaptation 

**Title (ZH)**: 基于散射的创新传播在大型语言模型中的多阶段过程适应中 

**Authors**: Hong Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.17949)  

**Abstract**: Large Language Models (LLMs) exhibit strong capabilities in reproducing and extending patterns observed during pretraining but often struggle to generalize novel ideas beyond their original context. This paper addresses the challenge of applying such localized innovations - introduced at a specific stage or component - to other parts of a multi-stage process. We propose a scatter-based innovation expansion model (innovation scatter model) that guides the LLM through a four-step process: (1) identifying the core innovation by comparing the user's input with its surrounding context, (2) generalizing the innovation by removing references to specific stages or components, (3) determining whether the generalized innovation applies to a broader scope beyond the original stage, and (4) systematically applying it to other structurally similar stages using the LLM. This model leverages structural redundancy across stages to improve the applicability of novel ideas. Verification results demonstrate that the innovation scatter model enables LLMs to extend innovations across structurally similar stages, thereby enhancing generalization and reuse. 

**Abstract (ZH)**: 大型语言模型(Large Language Models)在再现和扩展预训练中观察到的模式方面表现出强大的能力，但在将新颖想法推广应用到原有上下文之外的其他部分时往往存在困难。本文解决了将特定阶段或组件引入的局部创新应用于多阶段过程中其他部分的挑战。我们提出了一种基于散射的创新扩展模型（创新散射模型），该模型指导大型语言模型通过四个步骤进行：（1）通过将用户输入与其上下文进行比较来识别核心创新，（2）通过移除对特定阶段或组件的引用来进行创新的泛化，（3）确定泛化的创新是否适用于原始阶段之外的更广泛范围，（4）使用大型语言模型有系统地将其应用于其他结构上类似的阶段。该模型利用阶段间的结构冗余性以提高新颖想法的应用性。验证结果表明，创新散射模型使大型语言模型能够将创新扩展到结构上相似的阶段，从而提高泛化能力和重用性。 

---
# Greedy Selection under Independent Increments: A Toy Model Analysis 

**Title (ZH)**: 贪婪选择下的独立增量：一种玩具模型分析 

**Authors**: Huitao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17941)  

**Abstract**: We study an iterative selection problem over N i.i.d. discrete-time stochastic processes with independent increments. At each stage, a fixed number of processes are retained based on their observed values. Under this simple model, we prove that the optimal strategy for selecting the final maximum-value process is to apply greedy selection at each stage. While the result relies on strong independence assumptions, it offers a clean justification for greedy heuristics in multi-stage elimination settings and may serve as a toy example for understanding related algorithms in high-dimensional applications. 

**Abstract (ZH)**: 我们研究了在N个独立同分布的离散时间随机过程上的迭代选择问题。在每一阶段，根据观察到的值保留固定数量的过程。在这一简单模型下，我们证明了选择最终最大值过程的最优策略是在每一阶段应用贪婪选择。虽然该结果依赖于强烈的独立性假设，但它为多阶段淘汰设置中的贪婪启发式方法提供了清晰的解释，并可能作为理解相关算法在高维应用中的一个示例。 

---
# An entropy-optimal path to humble AI 

**Title (ZH)**: 熵优化路径 toward 谦逊人工智能 

**Authors**: Davide Bassetti, Lukáš Pospíšil, Michael Groom, Terence J. O'Kane, Illia Horenko  

**Link**: [PDF](https://arxiv.org/pdf/2506.17940)  

**Abstract**: Progress of AI has led to a creation of very successful, but by no means humble models and tools, especially regarding (i) the huge and further exploding costs and resources they demand, and (ii) the over-confidence of these tools with the answers they provide. Here we introduce a novel mathematical framework for a non-equilibrium entropy-optimizing reformulation of Boltzmann machines based on the exact law of total probability. It results in the highly-performant, but much cheaper, gradient-descent-free learning framework with mathematically-justified existence and uniqueness criteria, and answer confidence/reliability measures. Comparisons to state-of-the-art AI tools in terms of performance, cost and the model descriptor lengths on a set of synthetic problems with varying complexity reveal that the proposed method results in more performant and slim models, with the descriptor lengths being very close to the intrinsic complexity scaling bounds for the underlying problems. Applying this framework to historical climate data results in models with systematically higher prediction skills for the onsets of La Niña and El Niño climate phenomena, requiring just few years of climate data for training - a small fraction of what is necessary for contemporary climate prediction tools. 

**Abstract (ZH)**: 人工智能的进步创造出了非常成功但远非谦逊的模型和工具，尤其体现在(i)它们所需求的巨额且进一步爆炸性的成本和资源上，以及(ii)这些工具对其提供的答案表现出的过高水平的信心。基于精确的全概率定律，我们介绍了一种新颖的数学框架，用于玻尔兹曼机的非平衡最大熵重述。这导致了一种无需梯度下降、高性能且成本更低的学习框架，并具备数学上合理的存在性和唯一性条件，以及答案的信心/可靠性度量。在不同复杂度的合成问题上与最先进的AI工具进行性能、成本和模型描述长度的比较表明，所提出的方法产生了更具高效性和精简性的模型，其描述长度非常接近底层问题的固有复杂性标度界限。将此框架应用于历史气候数据，生成的模型在拉尼娜和厄尔尼诺气候现象的预测技能上系统性地优于现有工具，仅需少量年份的气候数据进行训练，这仅仅是当前气候预测工具所需数据量的一小部分。 

---
# GEMeX-ThinkVG: Towards Thinking with Visual Grounding in Medical VQA via Reinforcement Learning 

**Title (ZH)**: GEMeX-ThinkVG：通过强化学习实现医学VQA中的视觉定位思考 

**Authors**: Bo Liu, Xiangyu Zhao, Along He, Yidi Chen, Huazhu Fu, Xiao-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17939)  

**Abstract**: Medical visual question answering aims to support clinical decision-making by enabling models to answer natural language questions based on medical images. While recent advances in multi-modal learning have significantly improved performance, current methods still suffer from limited answer reliability and poor interpretability, impairing the ability of clinicians and patients to understand and trust model-generated answers. To address this, this work first proposes a Thinking with Visual Grounding (ThinkVG) dataset wherein the answer generation is decomposed into intermediate reasoning steps that explicitly ground relevant visual regions of the medical image, thereby providing fine-grained explainability. Furthermore, we introduce a novel verifiable reward mechanism for reinforcement learning to guide post-training, improving the alignment between the model's reasoning process and its final answer. Remarkably, our method achieves comparable performance using only one-eighth of the training data, demonstrating the efficiency and effectiveness of the proposal. The dataset is available at this https URL. 

**Abstract (ZH)**: 医学视觉问答旨在通过使模型基于医学图像回答自然语言问题来支持临床决策。尽管多模态学习的近期进展显著提高了性能，但当前方法仍存在答案可靠性有限和解释性不佳的问题，影响了临床医生和患者对模型生成答案的理解和信任。为解决这一问题，本工作首先提出了一种带有视觉定位思考（ThinkVG）数据集，将答案生成分解为中间推理步骤，明确地将相关医学图像的视觉区域连接起来，从而提供细粒度的可解释性。此外，我们引入了一种新颖的可验证奖励机制以指导强化学习，在后训练阶段提高模型的推理过程与其最终答案之间的对齐程度。令人惊讶的是，我们的方法仅使用八分之一的训练数据就达到了相当的性能，展示了该提议的高效性和有效性。数据集可通过以下链接获取：this https URL。 

---
# Software Reuse in the Generative AI Era: From Cargo Cult Towards AI Native Software Engineering 

**Title (ZH)**: 生成式AI时代软件重用：从cargo cult到AI原生软件工程 

**Authors**: Tommi Mikkonen, Antero Taivalsaari  

**Link**: [PDF](https://arxiv.org/pdf/2506.17937)  

**Abstract**: Software development is currently under a paradigm shift in which artificial intelligence and generative software reuse are taking the center stage in software creation. Consequently, earlier software reuse practices and methods are rapidly being replaced by AI-assisted approaches in which developers place their trust on code that has been generated by artificial intelligence. This is leading to a new form of software reuse that is conceptually not all that different from cargo cult development. In this paper we discuss the implications of AI-assisted generative software reuse in the context of emerging "AI native" software engineering, bring forth relevant questions, and define a tentative research agenda and call to action for tackling some of the central issues associated with this approach. 

**Abstract (ZH)**: 人工智能辅助生成式软件重用在新兴“AI本位”软件工程中的影响：相关问题探讨与研究议程 

---
# When concept-based XAI is imprecise: Do people distinguish between generalisations and misrepresentations? 

**Title (ZH)**: 基于概念的解释性人工智能不精确时：人们能否区分一般化和误表征？ 

**Authors**: Romy Müller  

**Link**: [PDF](https://arxiv.org/pdf/2506.17936)  

**Abstract**: Concept-based explainable artificial intelligence (C-XAI) can help reveal the inner representations of AI models. Understanding these representations is particularly important in complex tasks like safety evaluation. Such tasks rely on high-level semantic information (e.g., about actions) to make decisions about abstract categories (e.g., whether a situation is dangerous). In this context, it may desirable for C-XAI concepts to show some variability, suggesting that the AI is capable of generalising beyond the concrete details of a situation. However, it is unclear whether people recognise and appreciate such generalisations and can distinguish them from other, less desirable forms of imprecision. This was investigated in an experimental railway safety scenario. Participants evaluated the performance of a simulated AI that evaluated whether traffic scenes involving people were dangerous. To explain these decisions, the AI provided concepts in the form of similar image snippets. These concepts differed in their match with the classified image, either regarding a highly relevant feature (i.e., relation to tracks) or a less relevant feature (i.e., actions). Contrary to the hypotheses, concepts that generalised over less relevant features led to ratings that were lower than for precisely matching concepts and comparable to concepts that systematically misrepresented these features. Conversely, participants were highly sensitive to imprecisions in relevant features. These findings cast doubts on whether people spontaneously recognise generalisations. Accordingly, they might not be able to infer from C-XAI concepts whether AI models have gained a deeper understanding of complex situations. 

**Abstract (ZH)**: 基于概念的可解释人工智能（C-XAI）可以揭示AI模型的内在表示。在复杂的任务如安全性评估中，理解这些表示尤为重要。此类任务依赖于高层次的语义信息（例如，关于行动的信息）来对抽象类别（例如，情况是否危险）作出决策。在这种情境下，C-XAI的概念可能需要表现出一定的灵活性，表明AI能够超越具体情境的细节进行泛化。然而，尚不清楚人们是否能够认识到并欣赏这些泛化，以及是否能够将其与其它形式的不精确区分开来。这一问题在一个实验性的铁路安全场景中进行了探究。参与者评估了一个模拟AI的表现，该AI评估交通场景（涉及人员）是否危险。为解释这些决策，AI提供了以类似图像片段形式呈现的概念。这些概念在与分类图像的匹配程度上有所不同，前者是关于高度相关的特征（即，轨道的关系），后者是关于较不相关的特征（即，行动）。与假设相反，泛化于较不相关特征的概念导致的评分低于精确匹配的概念，并且这些评分与系统地误导这些特征的概念相当。相反，参与者对相关特征的不精确性非常敏感。这些发现对人们是否自发地认识到泛化的能力提出了疑问。因此，他们可能无法从C-XAI概念中推断出AI模型是否对复杂情境有了更深入的理解。 

---
# A GenAI System for Improved FAIR Independent Biological Database Integration 

**Title (ZH)**: 一个用于改善FAIR独立生物数据库集成的GenAI系统 

**Authors**: Syed N. Sakib, Kallol Naha, Sajratul Y. Rubaiat, Hasan M. Jamil  

**Link**: [PDF](https://arxiv.org/pdf/2506.17934)  

**Abstract**: Life sciences research increasingly requires identifying, accessing, and effectively processing data from an ever-evolving array of information sources on the Linked Open Data (LOD) network. This dynamic landscape places a significant burden on researchers, as the quality of query responses depends heavily on the selection and semantic integration of data sources --processes that are often labor-intensive, error-prone, and costly. While the adoption of FAIR (Findable, Accessible, Interoperable, and Reusable) data principles has aimed to address these challenges, barriers to efficient and accurate scientific data processing persist.
In this paper, we introduce FAIRBridge, an experimental natural language-based query processing system designed to empower scientists to discover, access, and query biological databases, even when they are not FAIR-compliant. FAIRBridge harnesses the capabilities of AI to interpret query intents, map them to relevant databases described in scientific literature, and generate executable queries via intelligent resource access plans. The system also includes robust tools for mitigating low-quality query processing, ensuring high fidelity and responsiveness in the information delivered.
FAIRBridge's autonomous query processing framework enables users to explore alternative data sources, make informed choices at every step, and leverage community-driven crowd curation when needed. By providing a user-friendly, automated hypothesis-testing platform in natural English, FAIRBridge significantly enhances the integration and processing of scientific data, offering researchers a powerful new tool for advancing their inquiries. 

**Abstract (ZH)**: 生命科学研究 increasingly requires identifying, accessing, and effectively processing data from an ever-evolving array of information sources on the Linked Open Data (LOD) network. This dynamic landscape places a significant burden on researchers, as the quality of query responses depends heavily on the selection and semantic integration of data sources --processes that are often labor-intensive, error-prone, and costly. While the adoption of FAIR (Findable, Accessible, Interoperable, and Reusable) data principles has aimed to address these challenges, barriers to efficient and accurate scientific data processing persist.
在这种动态的环境中，研究人员需要不断识别、访问并有效地处理链接开放数据（LOD）网络上不断演化的各种信息源的数据。高质量的查询响应依赖于数据来源的选择和语义集成——这些过程往往是劳动密集型、易出错和昂贵的。虽然采用可获取性（Findable）、可访问性（Accessible）、互操作性（Interoperable）和可重用性（Reusable）（FAIR）的数据原则旨在解决这些挑战，但高效的和准确的科学数据处理仍面临障碍。
In this paper, we introduce FAIRBridge, an experimental natural language-based query processing system designed to empower scientists to discover, access, and query biological databases, even when they are not FAIR-compliant. FAIRBridge harnesses the capabilities of AI to interpret query intents, map them to relevant databases described in scientific literature, and generate executable queries via intelligent resource access plans. The system also includes robust tools for mitigating low-quality query processing, ensuring high fidelity and responsiveness in the information delivered.
在这篇论文中，我们介绍了FAIRBridge，这是一种实验性的基于自然语言的查询处理系统，旨在使科学家能够发现、访问和查询生物数据库，即使这些数据库不符合FAIR准则也是如此。FAIRBridge利用人工智能的能力来解释查询意图，将其映射到科学文献中描述的相关数据库，并通过智能资源访问计划生成可执行查询。该系统还包含强大的工具来降低低质量查询处理的影响，确保交付的信息具有高保真度和高响应性。
FAIRBridge's autonomous query processing framework enables users to explore alternative data sources, make informed choices at every step, and leverage community-driven crowd curation when needed. By providing a user-friendly, automated hypothesis-testing platform in natural English, FAIRBridge significantly enhances the integration and processing of scientific data, offering researchers a powerful new tool for advancing their inquiries.
FAIRBridge的自动查询处理框架使用户能够探索替代数据源，在每一步做出知情选择，并在需要时利用社区驱动的众包进行内容审查。通过提供一个用户友好、自动化的自然英文假设检验平台，FAIRBridge显著增强了科学数据的集成和处理，为研究人员提供了促进其研究的强大新工具。 

---
# IDAL: Improved Domain Adaptive Learning for Natural Images Dataset 

**Title (ZH)**: IDAL: 改进的领域自适应学习方法用于自然图像数据集 

**Authors**: Ravi Kant Gupta, Shounak Das, Amit Sethi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17931)  

**Abstract**: We present a novel approach for unsupervised domain adaptation (UDA) for natural images. A commonly-used objective for UDA schemes is to enhance domain alignment in representation space even if there is a domain shift in the input space. Existing adversarial domain adaptation methods may not effectively align different domains of multimodal distributions associated with classification problems. Our approach has two main features. Firstly, its neural architecture uses the deep structure of ResNet and the effective separation of scales of feature pyramidal network (FPN) to work with both content and style features. Secondly, it uses a combination of a novel loss function and judiciously selected existing loss functions to train the network architecture. This tailored combination is designed to address challenges inherent to natural images, such as scale, noise, and style shifts, that occur on top of a multi-modal (multi-class) distribution. The combined loss function not only enhances model accuracy and robustness on the target domain but also speeds up training convergence. Our proposed UDA scheme generalizes better than state-of-the-art for CNN-based methods on Office-Home, Office-31, and VisDA-2017 datasets and comaparable for DomainNet dataset. 

**Abstract (ZH)**: 我们提出了一种新的无监督领域适应（UDA）方法用于自然图像。 

---
# ASTER: Adaptive Spatio-Temporal Early Decision Model for Dynamic Resource Allocation 

**Title (ZH)**: ASTER：适应性时空早期决策模型用于动态资源分配 

**Authors**: Shulun Chen, Wei Shao, Flora D. Salim, Hao Xue  

**Link**: [PDF](https://arxiv.org/pdf/2506.17929)  

**Abstract**: Supporting decision-making has long been a central vision in the field of spatio-temporal intelligence. While prior work has improved the timeliness and accuracy of spatio-temporal forecasting, converting these forecasts into actionable strategies remains a key challenge. A main limitation is the decoupling of the prediction and the downstream decision phases, which can significantly degrade the downstream efficiency. For example, in emergency response, the priority is successful resource allocation and intervention, not just incident prediction. To this end, it is essential to propose an Adaptive Spatio-Temporal Early Decision model (ASTER) that reforms the forecasting paradigm from event anticipation to actionable decision support. This framework ensures that information is directly used for decision-making, thereby maximizing overall effectiveness. Specifically, ASTER introduces a new Resource-aware Spatio-Temporal interaction module (RaST) that adaptively captures long- and short-term dependencies under dynamic resource conditions, producing context-aware spatiotemporal representations. To directly generate actionable decisions, we further design a Preference-oriented decision agent (Poda) based on multi-objective reinforcement learning, which transforms predictive signals into resource-efficient intervention strategies by deriving optimal actions under specific preferences and dynamic constraints. Experimental results on four benchmark datasets demonstrate the state-of-the-art performance of ASTER in improving both early prediction accuracy and resource allocation outcomes across six downstream metrics. 

**Abstract (ZH)**: 支持决策长久以来一直是时空智能领域的核心愿景。尽管前期工作在提升时空预测的时效性和准确性方面取得进展，但将这些预测转化为 actionable 策略仍然是一项关键挑战。主要限制在于预测阶段与下游决策阶段的脱节，这会显著降低下游效率。例如，在应急响应中，优先级是成功分配和干预资源，而不仅仅是事件预测。为此，提出一种自适应时空早期决策模型（ASTER）以改革预测范式，从事件预见到行动支持决策。该框架确保信息直接用于决策，从而最大化整体效果。具体而言，ASTER 引入了一种资源感知时空交互模块（RaST），能够在动态资源条件下自适应捕捉长期和短期依赖性，生成上下文感知的时空表示。为进一步生成 actionable 决策，设计了一种基于多目标强化学习的偏好导向决策智能体（Poda），通过在特定偏好和动态约束下推导出最优行动，将预测信号转化为高效的资源干预策略。在四个基准数据集上的实验结果表明，ASTER 在六个下游指标上均能提高早期预测准确性和资源分配效果，展现出最先进的性能。 

---
# Permutation Equivariant Model-based Offline Reinforcement Learning for Auto-bidding 

**Title (ZH)**: 基于置换不变模型的离线强化学习自动出价方法 

**Authors**: Zhiyu Mou, Miao Xu, Wei Chen, Rongquan Bai, Chuan Yu, Jian Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17919)  

**Abstract**: Reinforcement learning (RL) for auto-bidding has shifted from using simplistic offline simulators (Simulation-based RL Bidding, SRLB) to offline RL on fixed real datasets (Offline RL Bidding, ORLB). However, ORLB policies are limited by the dataset's state space coverage, offering modest gains. While SRLB expands state coverage, its simulator-reality gap risks misleading policies. This paper introduces Model-based RL Bidding (MRLB), which learns an environment model from real data to bridge this gap. MRLB trains policies using both real and model-generated data, expanding state coverage beyond ORLB. To ensure model reliability, we propose: 1) A permutation equivariant model architecture for better generalization, and 2) A robust offline Q-learning method that pessimistically penalizes model errors. These form the Permutation Equivariant Model-based Offline RL (PE-MORL) algorithm. Real-world experiments show that PE-MORL outperforms state-of-the-art auto-bidding methods. 

**Abstract (ZH)**: 基于模型的强化学习自动竞价：从模拟基在线学习到模型基离线强化学习 

---
# Feedback Driven Multi Stereo Vision System for Real-Time Event Analysis 

**Title (ZH)**: 实时事件分析驱动的反馈多视图立体视觉系统 

**Authors**: Mohamed Benkedadra, Matei Mancas, Sidi Ahmed Mahmoudi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17910)  

**Abstract**: 2D cameras are often used in interactive systems. Other systems like gaming consoles provide more powerful 3D cameras for short range depth sensing. Overall, these cameras are not reliable in large, complex environments. In this work, we propose a 3D stereo vision based pipeline for interactive systems, that is able to handle both ordinary and sensitive applications, through robust scene understanding. We explore the fusion of multiple 3D cameras to do full scene reconstruction, which allows for preforming a wide range of tasks, like event recognition, subject tracking, and notification. Using possible feedback approaches, the system can receive data from the subjects present in the environment, to learn to make better decisions, or to adapt to completely new environments. Throughout the paper, we introduce the pipeline and explain our preliminary experimentation and results. Finally, we draw the roadmap for the next steps that need to be taken, in order to get this pipeline into production 

**Abstract (ZH)**: 基于3D立体视觉的交互系统pipeline及其在广泛应用场景中的鲁棒场景理解与融合方法 

---
# Cause-Effect Driven Optimization for Robust Medical Visual Question Answering with Language Biases 

**Title (ZH)**: 基于因果驱动的优化方法以应对语言偏差的鲁棒医学视觉问答 

**Authors**: Huanjia Zhu, Yishu Liu, Xiaozhao Fang, Guangming Lu, Bingzhi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.17903)  

**Abstract**: Existing Medical Visual Question Answering (Med-VQA) models often suffer from language biases, where spurious correlations between question types and answer categories are inadvertently established. To address these issues, we propose a novel Cause-Effect Driven Optimization framework called CEDO, that incorporates three well-established mechanisms, i.e., Modality-driven Heterogeneous Optimization (MHO), Gradient-guided Modality Synergy (GMS), and Distribution-adapted Loss Rescaling (DLR), for comprehensively mitigating language biases from both causal and effectual perspectives. Specifically, MHO employs adaptive learning rates for specific modalities to achieve heterogeneous optimization, thus enhancing robust reasoning capabilities. Additionally, GMS leverages the Pareto optimization method to foster synergistic interactions between modalities and enforce gradient orthogonality to eliminate bias updates, thereby mitigating language biases from the effect side, i.e., shortcut bias. Furthermore, DLR is designed to assign adaptive weights to individual losses to ensure balanced learning across all answer categories, effectively alleviating language biases from the cause side, i.e., imbalance biases within datasets. Extensive experiments on multiple traditional and bias-sensitive benchmarks consistently demonstrate the robustness of CEDO over state-of-the-art competitors. 

**Abstract (ZH)**: 基于因果驱动优化的医疗视觉问答语言偏差缓解框架：MHO、GMS和DLR联合优化(CEDO) 

---
# EgoWorld: Translating Exocentric View to Egocentric View using Rich Exocentric Observations 

**Title (ZH)**: EgoWorld: 使用丰富的外人视角观察转换到自我视角 

**Authors**: Junho Park, Andrew Sangwoo Ye, Taein Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2506.17896)  

**Abstract**: Egocentric vision is essential for both human and machine visual understanding, particularly in capturing the detailed hand-object interactions needed for manipulation tasks. Translating third-person views into first-person views significantly benefits augmented reality (AR), virtual reality (VR) and robotics applications. However, current exocentric-to-egocentric translation methods are limited by their dependence on 2D cues, synchronized multi-view settings, and unrealistic assumptions such as necessity of initial egocentric frame and relative camera poses during inference. To overcome these challenges, we introduce EgoWorld, a novel two-stage framework that reconstructs an egocentric view from rich exocentric observations, including projected point clouds, 3D hand poses, and textual descriptions. Our approach reconstructs a point cloud from estimated exocentric depth maps, reprojects it into the egocentric perspective, and then applies diffusion-based inpainting to produce dense, semantically coherent egocentric images. Evaluated on the H2O and TACO datasets, EgoWorld achieves state-of-the-art performance and demonstrates robust generalization to new objects, actions, scenes, and subjects. Moreover, EgoWorld shows promising results even on unlabeled real-world examples. 

**Abstract (ZH)**: 本体中心视觉对于人类和机器视觉理解都是必不可少的，特别是在捕捉用于操作任务的详细手物交互方面。将第三方视角转换为第一人视角对增强现实（AR）、虚拟现实（VR）和机器人应用有显著益处。然而，当前的从外视角到本视角转换方法受限于其对2D线索的依赖、同步多视角设置以及初始本视角框架和推断期间的相对相机姿态等不现实的假设。为了克服这些挑战，我们引入了EgoWorld，这是一种新颖的两阶段框架，可以从丰富的外视角观察中重构本视角视图，包括投影点云、3D手部姿态和文本描述。我们的方法从估计的外视角深度图中重构点云，将其重新投影到本视角视角，然后应用基于扩散的 inpainting生成密集且语义一致的本视角图像。在H2O和TACO数据集上的评估结果显示，EgoWorld达到了最先进的性能，并展示了对新物体、动作、场景和主体的鲁棒泛化能力。此外，EgoWorld甚至在未标记的真实世界示例上也显示出令人鼓舞的结果。 

---
# Multi-turn Jailbreaking via Global Refinement and Active Fabrication 

**Title (ZH)**: 全局细化与主动伪造驱动的多轮 Jailbreaking 

**Authors**: Hua Tang, Lingyong Yan, Yukun Zhao, Shuaiqiang Wang, Jizhou Huang, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2506.17881)  

**Abstract**: Large Language Models (LLMs) have achieved exceptional performance across a wide range of tasks. However, they still pose significant safety risks due to the potential misuse for malicious purposes. Jailbreaks, which aim to elicit models to generate harmful content, play a critical role in identifying the underlying security threats. Recent jailbreaking primarily focuses on single-turn scenarios, while the more complicated multi-turn scenarios remain underexplored. Moreover, existing multi-turn jailbreaking techniques struggle to adapt to the evolving dynamics of dialogue as the interaction progresses. To address this limitation, we propose a novel multi-turn jailbreaking method that refines the jailbreaking path globally at each interaction. We also actively fabricate model responses to suppress safety-related warnings, thereby increasing the likelihood of eliciting harmful outputs in subsequent questions. Experimental results demonstrate the superior performance of our method compared with existing single-turn and multi-turn jailbreaking techniques across six state-of-the-art LLMs. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各类任务中取得了出色的表现，但在恶意利用的潜在风险下仍然存在重大的安全问题。攻击性 jailbreaks 在揭示潜在安全威胁方面起着关键作用。尽管当前主要关注单轮对话攻击，但更复杂的多轮对话攻击场景尚未得到充分探索。此外，现有技术难以适应对话过程中不断变化的交互动态。为解决这一限制，我们提出了一种新颖的多轮对话 jailbreak 方法，在每次交互中全局优化 jailbreak 路径。我们还积极伪造模型响应以抑制与安全性相关的警告，从而增加在后续问题中获得有害输出的可能性。实验结果表明，我们的方法在六个最先进的 LLM 上优于现有的单轮和多轮对话攻击方法。我们的代码已公开，网址为：this https URL。 

---
# StainPIDR: A Pathological Image Decouplingand Reconstruction Method for StainNormalization Based on Color VectorQuantization and Structure Restaining 

**Title (ZH)**: StainPIDR：基于颜色向量量化和结构复染的病理图像去染色解耦与重构方法 

**Authors**: Zheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.17879)  

**Abstract**: The color appearance of a pathological image is highly related to the imaging protocols, the proportion of different dyes, and the scanning devices. Computer-aided diagnostic systems may deteriorate when facing these color-variant pathological images. In this work, we propose a stain normalization method called StainPIDR. We try to eliminate this color discrepancy by decoupling the image into structure features and vector-quantized color features, restaining the structure features with the target color features, and decoding the stained structure features to normalized pathological images. We assume that color features decoupled by different images with the same color should be exactly the same. Under this assumption, we train a fixed color vector codebook to which the decoupled color features will map. In the restaining part, we utilize the cross-attention mechanism to efficiently stain the structure features. As the target color (decoupled from a selected template image) will also affect the performance of stain normalization, we further design a template image selection algorithm to select a template from a given dataset. In our extensive experiments, we validate the effectiveness of StainPIDR and the template image selection algorithm. All the results show that our method can perform well in the stain normalization task. The code of StainPIDR will be publicly available later. 

**Abstract (ZH)**: 病理图像的颜色表现与其成像协议、不同染料的比例和扫描设备高度相关。面对这些颜色变异的病理图像，计算机辅助诊断系统可能会退化。在本工作中，我们提出了一种称为StainPIDR的染色正则化方法。我们通过将图像解耦为结构特征和向量量化颜色特征，使用目标颜色特征重新染色结构特征，并解码染色后的结构特征生成标准化病理图像，来消除这种颜色差异。我们假设相同颜色的不同图像解耦出的颜色特征应该是完全相同的。在此假设下，我们训练一个固定的彩色矢量码本，解耦出的颜色特征将映射到该码本上。在重新染色部分，我们利用交叉注意力机制高效地重新染色结构特征。由于目标颜色（来自选定模板图像解耦获得）也会影响染色正则化的性能，我们进一步设计了一种模板图像选择算法，从给定的数据集中选择模板图像。在广泛实验中，我们验证了StainPIDR方法及其模板图像选择算法的有效性。所有结果表明，我们的方法在染色正则化任务中表现良好。StainPIDR的代码将在后续公开。 

---
# SurgVidLM: Towards Multi-grained Surgical Video Understanding with Large Language Model 

**Title (ZH)**: SurgVidLM：利用大型语言模型实现多粒度手术视频理解 

**Authors**: Guankun Wang, Wenjin Mo, Junyi Wang, Long Bai, Kun Yuan, Ming Hu, Jinlin Wu, Junjun He, Yiming Huang, Nicolas Padoy, Zhen Lei, Hongbin Liu, Nassir Navab, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2506.17873)  

**Abstract**: Recent advances in Multimodal Large Language Models have demonstrated great potential in the medical domain, facilitating users to understand surgical scenes and procedures. Beyond image-based methods, the exploration of Video Large Language Models (Vid-LLMs) has emerged as a promising avenue for capturing the complex sequences of information involved in surgery. However, there is still a lack of Vid-LLMs specialized for fine-grained surgical video understanding tasks, which is crucial for analyzing specific processes or details within a surgical procedure. To bridge this gap, we propose SurgVidLM, the first video language model designed to address both full and fine-grained surgical video comprehension. To train our SurgVidLM, we construct the SVU-31K dataset which consists of over 31K video-instruction pairs, enabling both holistic understanding and detailed analysis of surgical procedures. Furthermore, we introduce the StageFocus mechanism which is a two-stage framework performing the multi-grained, progressive understanding of surgical videos. We also develop the Multi-frequency Fusion Attention to effectively integrate low and high-frequency visual tokens, ensuring the retention of critical information. Experimental results demonstrate that SurgVidLM significantly outperforms state-of-the-art Vid-LLMs in both full and fine-grained video understanding tasks, showcasing its superior capability in capturing complex procedural contexts. 

**Abstract (ZH)**: Recent Advances in Multimodal Large Language Models for Fine-Grained Surgical Video Understanding 

---
# How Alignment Shrinks the Generative Horizon 

**Title (ZH)**: 如何对齐缩小生成视野 

**Authors**: Chenghao Yang, Ari Holtzman  

**Link**: [PDF](https://arxiv.org/pdf/2506.17871)  

**Abstract**: Despite their impressive capabilities, aligned large language models (LLMs) often generate outputs that lack diversity. What drives this stability in the generation? We investigate this phenomenon through the lens of probability concentration in the model's output distribution. To quantify this concentration, we introduce the Branching Factor (BF) -- a token-invariant measure of the effective number of plausible next steps during generation. Our empirical analysis reveals two key findings: (1) BF often decreases as generation progresses, suggesting that LLMs become more predictable as they generate. (2) alignment tuning substantially sharpens the model's output distribution from the outset, reducing BF by nearly an order of magnitude (e.g., from 12 to 1.2) relative to base models. This stark reduction helps explain why aligned models often appear less sensitive to decoding strategies. Building on this insight, we find this stability has surprising implications for complex reasoning. Aligned Chain-of-Thought (CoT) models (e.g., DeepSeek-distilled models), for instance, leverage this effect; by generating longer reasoning chains, they push generation into later, more deterministic (lower BF) stages, resulting in more stable outputs. We hypothesize that alignment tuning does not fundamentally change a model's behavior, but instead steers it toward stylistic tokens (e.g., "Sure") that unlock low-entropy trajectories already present in the base model. This view is supported by nudging experiments, which show that prompting base models with such tokens can similarly reduce BF. Together, our findings establish BF as a powerful diagnostic for understanding and controlling LLM outputs - clarifying how alignment reduces variability, how CoT promotes stable generations, and how base models can be steered away from diversity. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）具备强大能力，但它们生成的输出往往缺乏多样性。这种稳定性是如何驱动生成过程的？我们通过模型输出分布的概率集中性这一视角来探讨这一现象。为了量化这种集中性，我们引入了分叉因子（Branching Factor，BF）——一个不受标记影响的有效可能下一步的数量度量。我们的实验证据揭示了两个关键发现：（1）BF在生成过程中往往降低，表明LLMs在生成时变得更加可预测。（2）对齐调整从一开始就显著使模型的输出分布变得集中，相对于基础模型，BF几乎减少了十倍（例如，从12降至1.2）。这种显著减少解释了为什么对齐后的模型往往对外推策略的敏感性较低。基于这一见解，我们发现这种稳定性对复杂推理产生了意想不到的影响。例如，对齐后的链式思考（CoT）模型（如DeepSeek提炼模型）利用了这一效果，通过生成更长的推理链，将生成过程推向后期、更确定（BF更低）的阶段，从而产生更稳定的输出。我们推测对齐调整并未根本改变模型的行为，而是引导模型朝向风格化标记（如“当然”）前进，这些标记在基础模型中已经存在低熵轨迹。这一观点通过推动实验得到了支持，这些实验表明，用这些标记提示基础模型也可以同样减少BF。我们的发现共同确立了BF作为理解和控制LLM输出的强大诊断工具——阐明对齐如何减少变异性，CoT如何促进稳定生成，以及如何引导基础模型远离多样性。 

---
# NestQuant: Post-Training Integer-Nesting Quantization for On-Device DNN 

**Title (ZH)**: NestQuant: On-Device DNN整数嵌套量化后训练方法 

**Authors**: Jianhang Xie, Chuntao Ding, Xiaqing Li, Shenyuan Ren, Yidong Li, Zhichao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17870)  

**Abstract**: Deploying quantized deep neural network (DNN) models with resource adaptation capabilities on ubiquitous Internet of Things (IoT) devices to provide high-quality AI services can leverage the benefits of compression and meet multi-scenario resource requirements. However, existing dynamic/mixed precision quantization requires retraining or special hardware, whereas post-training quantization (PTQ) has two limitations for resource adaptation: (i) The state-of-the-art PTQ methods only provide one fixed bitwidth model, which makes it challenging to adapt to the dynamic resources of IoT devices; (ii) Deploying multiple PTQ models with diverse bitwidths consumes large storage resources and switching overheads. To this end, this paper introduces a resource-friendly post-training integer-nesting quantization, i.e., NestQuant, for on-device quantized model switching on IoT devices. The proposed NestQuant incorporates the integer weight decomposition, which bit-wise splits quantized weights into higher-bit and lower-bit weights of integer data types. It also contains a decomposed weights nesting mechanism to optimize the higher-bit weights by adaptive rounding and nest them into the original quantized weights. In deployment, we can send and store only one NestQuant model and switch between the full-bit/part-bit model by paging in/out lower-bit weights to adapt to resource changes and reduce consumption. Experimental results on the ImageNet-1K pretrained DNNs demonstrated that the NestQuant model can achieve high performance in top-1 accuracy, and reduce in terms of data transmission, storage consumption, and switching overheads. In particular, the ResNet-101 with INT8 nesting INT6 can achieve 78.1% and 77.9% accuracy for full-bit and part-bit models, respectively, and reduce switching overheads by approximately 78.1% compared with diverse bitwidths PTQ models. 

**Abstract (ZH)**: 在物联网设备上部署具有资源适应能力的量化深度神经网络模型以提供高质量的AI服务可以利用压缩带来的好处并满足多场景的资源需求。然而，现有的动态/混合精度量化需要进行重新训练或特殊的硬件支持，而后训练量化（PTQ）方法有两个限制：(i) 当前最先进的PTQ方法只能提供一种固定的位宽模型，这使得适应物联网设备的动态资源具有挑战性；(ii) 部署具有多种位宽的多个PTQ模型消耗大量的存储资源和切换开销。为此，本文介绍了一种资源友好的后训练整数嵌套量化方法，即NestQuant，用于物联网设备上的量化模型在设备上的切换。提出的NestQuant结合了整数权重分解，按位将量化权重拆分为较高位和较低位的整数数据类型权重。它还包含一个拆分权重嵌套机制，通过自适应舍入优化较高位权重，并将其嵌套到原始量化权重中。在部署过程中，我们可以发送和存储一个NestQuant模型，并通过调入/调出较低位权重在全精度/部分精度模型之间切换，以适应资源变化并减少消耗。实验结果表明，NestQuant模型在Top-1准确率、数据传输量、存储消耗和切换开销方面均可实现高性能。特别是，ResNet-101在嵌套INT8为INT6的配置下，全精度和部分精度模型的准确率分别为78.1%和77.9%，并且与具有多种位宽的PTQ模型相比，切换开销减少了约78.1%。 

---
# In-Context Learning Strategies Emerge Rationally 

**Title (ZH)**: 上下文学习策略理性涌现 

**Authors**: Daniel Wurgaft, Ekdeep Singh Lubana, Core Francisco Park, Hidenori Tanaka, Gautam Reddy, Noah D. Goodman  

**Link**: [PDF](https://arxiv.org/pdf/2506.17859)  

**Abstract**: Recent work analyzing in-context learning (ICL) has identified a broad set of strategies that describe model behavior in different experimental conditions. We aim to unify these findings by asking why a model learns these disparate strategies in the first place. Specifically, we start with the observation that when trained to learn a mixture of tasks, as is popular in the literature, the strategies learned by a model for performing ICL can be captured by a family of Bayesian predictors: a memorizing predictor, which assumes a discrete prior on the set of seen tasks, and a generalizing predictor, wherein the prior matches the underlying task distribution. Adopting the lens of rational analysis from cognitive science, where a learner's behavior is explained as an optimal adaptation to data given computational constraints, we develop a hierarchical Bayesian framework that almost perfectly predicts Transformer next token predictions throughout training without assuming access to its weights. Under this framework, pretraining is viewed as a process of updating the posterior probability of different strategies, and its inference-time behavior as a posterior-weighted average over these strategies' predictions. Our framework draws on common assumptions about neural network learning dynamics, which make explicit a tradeoff between loss and complexity among candidate strategies: beyond how well it explains the data, a model's preference towards implementing a strategy is dictated by its complexity. This helps explain well-known ICL phenomena, while offering novel predictions: e.g., we show a superlinear trend in the timescale for transition to memorization as task diversity is increased. Overall, our work advances an explanatory and predictive account of ICL grounded in tradeoffs between strategy loss and complexity. 

**Abstract (ZH)**: 近期关于上下文学习（ICL）的研究已识别出一系列描述模型行为的策略。我们旨在通过探讨模型为何会在一开始就学习这些不同的策略来统一这些发现。具体而言，我们从文献中流行的做法——即训练模型学习任务混合开始，观察到模型在进行ICL时所学到的策略可以用一组贝叶斯预测器来捕捉：一种记忆预测器，假设看到的任务集合具有离散prior；一种泛化预测器，其中prior与任务分布匹配。借鉴认知科学中的理性分析视角，即在计算资源受限的情况下，学习者的行为被视为对数据的最佳适应，我们开发了一个分层贝叶斯框架，几乎可以在训练过程中完美预测Transformer的下一个标记预测，而不假设对权重的访问。在此框架下，预训练被视为更新不同策略后验概率的过程，而推理时的行为则视为这些策略预测的后验加权平均。我们的框架借鉴了关于神经网络学习动力学的常见假设，明确展示了候选策略之间的损失与复杂性之间的权衡：模型偏好实现某一策略不仅取决于其解释数据的能力，还取决于其复杂性。这有助于解释已知的ICL现象，并提出新的预测，如随着任务多样性增加，过渡到记忆的比例呈现超线性趋势。总体而言，我们的工作为ICL提供了一个基于策略损失与复杂性权衡的解释性和预测性框架。 

---
# Pathway-based Progressive Inference (PaPI) for Energy-Efficient Continual Learning 

**Title (ZH)**: 基于路径的渐进推理（PaPI）for 能效持续学习 

**Authors**: Suyash Gaurav, Jukka Heikkonen, Jatin Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2506.17848)  

**Abstract**: Continual learning systems face the dual challenge of preventing catastrophic forgetting while maintaining energy efficiency, particularly in resource-constrained environments. This paper introduces Pathway-based Progressive Inference (PaPI), a novel theoretical framework that addresses these challenges through a mathematically rigorous approach to pathway selection and adaptation. We formulate continual learning as an energy-constrained optimization problem and provide formal convergence guarantees for our pathway routing mechanisms. Our theoretical analysis demonstrates that PaPI achieves an $\mathcal{O}(K)$ improvement in the stability-plasticity trade-off compared to monolithic architectures, where $K$ is the number of pathways. We derive tight bounds on forgetting rates using Fisher Information Matrix analysis and prove that PaPI's energy consumption scales with the number of active parameters rather than the total model size. Comparative theoretical analysis shows that PaPI provides stronger guarantees against catastrophic forgetting than Elastic Weight Consolidation (EWC) while maintaining better energy efficiency than both EWC and Gradient Episodic Memory (GEM). Our experimental validation confirms these theoretical advantages across multiple benchmarks, demonstrating PaPI's effectiveness for continual learning in energy-constrained settings. Our codes are available at this https URL. 

**Abstract (ZH)**: 基于路径的渐进推理（PaPI）：在能量约束环境中预防灾难性遗忘与保持能量效率的新理论框架 

---
# A Comparative Study of Open-Source Libraries for Synthetic Tabular Data Generation: SDV vs. SynthCity 

**Title (ZH)**: 开源库合成表格数据生成比较研究：SDV vs. SynthCity 

**Authors**: Cristian Del Gobbo  

**Link**: [PDF](https://arxiv.org/pdf/2506.17847)  

**Abstract**: High-quality training data is critical to the performance of machine learning models, particularly Large Language Models (LLMs). However, obtaining real, high-quality data can be challenging, especially for smaller organizations and early-stage startups. Synthetic data generators provide a promising solution by replicating the statistical and structural properties of real data while preserving privacy and scalability. This study evaluates the performance of six tabular synthetic data generators from two widely used open-source libraries: SDV (Gaussian Copula, CTGAN, TVAE) and Synthicity (Bayesian Network, CTGAN, TVAE). Using a real-world dataset from the UCI Machine Learning Repository, comprising energy consumption and environmental variables from Belgium, we simulate a low-data regime by training models on only 1,000 rows. Each generator is then tasked with producing synthetic datasets under two conditions: a 1:1 (1,000 rows) and a 1:10 (10,000 rows) input-output ratio. Evaluation is conducted using two criteria: statistical similarity, measured via classical statistics and distributional metrics; and predictive utility, assessed using a "Train on Synthetic, Test on Real" approach with four regression models. While statistical similarity remained consistent across models in both scenarios, predictive utility declined notably in the 1:10 case. The Bayesian Network from Synthicity achieved the highest fidelity in both scenarios, while TVAE from SDV performed best in predictive tasks under the 1:10 setting. Although no significant performance gap was found between the two libraries, SDV stands out for its superior documentation and ease of use, making it more accessible for practitioners. 

**Abstract (ZH)**: 高质量训练数据对机器学习模型，特别是大型语言模型(LLMs)的性能至关重要。然而，获取真实的高质量数据对于较小的组织和早期初创企业来说颇具挑战。合成数据生成器通过复制真实数据的统计和结构属性，同时保护隐私和可扩展性，提供了一个有前景的解决方案。本研究评估了来自两个广泛使用的开源库SDV（Gaussian Copula、CTGAN、TVAE）和Synthicity（Bayesian Network、CTGAN、TVAE）的六种表结构合成数据生成器的表现。使用来自UCI机器学习仓库的实际数据集，该数据集包含来自比利时的能源消耗和环境变量，模拟低数据环境，仅使用1,000行进行模型训练。然后在1:1（1,000行）和1:10（10,000行）输入-输出比率的条件下，每种生成器生成合成数据集。评估标准包括统计相似性，通过经典统计方法和分布度量测量；以及预测效用，使用“在合成数据上训练，在实际数据上测试”的方法，评估四种回归模型。尽管两种情况下模型的统计相似性保持一致，但在1:10情况下预测效用显著下降。Synthicity的Bayesian Network在两种情况下均表现出最高的保真度，而SDV的TVAE在1:10设置下的预测任务中表现最佳。虽然两个库之间未发现显著性能差异，但SDV因其更好的文档和易用性脱颖而出，使其更适用于实践者。 

---
# THCM-CAL: Temporal-Hierarchical Causal Modelling with Conformal Calibration for Clinical Risk Prediction 

**Title (ZH)**: THCM-CAL: 时间分层因果建模结合校准化验证的临床风险预测 

**Authors**: Xin Zhang, Qiyu Wei, Yingjie Zhu, Fanyi Wu, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2506.17844)  

**Abstract**: Automated clinical risk prediction from electronic health records (EHRs) demands modeling both structured diagnostic codes and unstructured narrative notes. However, most prior approaches either handle these modalities separately or rely on simplistic fusion strategies that ignore the directional, hierarchical causal interactions by which narrative observations precipitate diagnoses and propagate risk across admissions. In this paper, we propose THCM-CAL, a Temporal-Hierarchical Causal Model with Conformal Calibration. Our framework constructs a multimodal causal graph where nodes represent clinical entities from two modalities: Textual propositions extracted from notes and ICD codes mapped to textual descriptions. Through hierarchical causal discovery, THCM-CAL infers three clinically grounded interactions: intra-slice same-modality sequencing, intra-slice cross-modality triggers, and inter-slice risk propagation. To enhance prediction reliability, we extend conformal prediction to multi-label ICD coding, calibrating per-code confidence intervals under complex co-occurrences. Experimental results on MIMIC-III and MIMIC-IV demonstrate the superiority of THCM-CAL. 

**Abstract (ZH)**: 电子健康记录（EHRs）中的自动化临床风险预测需要同时建模结构化诊断代码和非结构化病历笔记。然而，大多数先前的方法要么单独处理这些模态，要么依赖于忽略叙述性观察如何引发诊断及其在住院之间传播风险的简单的融合策略。本文提出了一种名为THCM-CAL的时空层次因果模型，该模型构建了一个多模态因果图，通过层次因果发现推断出三条临床相关交互：同一切片内同模态序列、同一切片内跨模态触发以及跨切片的风险传播。为了增强预测可靠性，我们将校准预测扩展到多标签ICD编码，针对复杂共现现象calibrate每个代码的置信区间。在MIMIC-III和MIMIC-IV上的实验结果表明THCM-CAL的优越性。 

---
# Generative Grasp Detection and Estimation with Concept Learning-based Safety Criteria 

**Title (ZH)**: 基于概念学习的安全准则的生成性抓取检测与估计 

**Authors**: Al-Harith Farhad, Khalil Abuibaid, Christiane Plociennik, Achim Wagner, Martin Ruskowski  

**Link**: [PDF](https://arxiv.org/pdf/2506.17842)  

**Abstract**: Neural networks are often regarded as universal equations that can estimate any function. This flexibility, however, comes with the drawback of high complexity, rendering these networks into black box models, which is especially relevant in safety-centric applications. To that end, we propose a pipeline for a collaborative robot (Cobot) grasping algorithm that detects relevant tools and generates the optimal grasp. To increase the transparency and reliability of this approach, we integrate an explainable AI method that provides an explanation for the underlying prediction of a model by extracting the learned features and correlating them to corresponding classes from the input. These concepts are then used as additional criteria to ensure the safe handling of work tools. In this paper, we show the consistency of this approach and the criterion for improving the handover position. This approach was tested in an industrial environment, where a camera system was set up to enable a robot to pick up certain tools and objects. 

**Abstract (ZH)**: 神经网络通常被视为万能方程，能够估计任何函数。然而，这种灵活性伴随着高复杂性的缺点，使这些网络成为黑盒模型，特别是在以安全为中心的应用中更为突出。为此，我们提出了一种协作机器人（Cobot）抓取算法的管道，用于检测相关工具并生成最优抓取方式。为了提高该方法的透明度和可靠性，我们集成了一种可解释的AI方法，通过提取学习特征并与输入中的相应类别进行关联，为模型的底层预测提供解释。这些概念随后被用作额外的标准，以确保工具的安全处理。在本文中，我们展示了该方法的一致性及其改进交换单元位置的准则。该方法在工业环境中进行了测试，设置了一个摄像头系统，使机器人能够拾取特定的工具和物体。 

---
# Causal Spherical Hypergraph Networks for Modelling Social Uncertainty 

**Title (ZH)**: 因果球形超图网络用于建模社会不确定性 

**Authors**: Anoushka Harit, Zhongtian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.17840)  

**Abstract**: Human social behaviour is governed by complex interactions shaped by uncertainty, causality, and group dynamics. We propose Causal Spherical Hypergraph Networks (Causal-SphHN), a principled framework for socially grounded prediction that jointly models higher-order structure, directional influence, and epistemic uncertainty. Our method represents individuals as hyperspherical embeddings and group contexts as hyperedges, capturing semantic and relational geometry. Uncertainty is quantified via Shannon entropy over von Mises-Fisher distributions, while temporal causal dependencies are identified using Granger-informed subgraphs. Information is propagated through an angular message-passing mechanism that respects belief dispersion and directional semantics. Experiments on SNARE (offline networks), PHEME (online discourse), and AMIGOS (multimodal affect) show that Causal-SphHN improves predictive accuracy, robustness, and calibration over strong baselines. Moreover, it enables interpretable analysis of influence patterns and social ambiguity. This work contributes a unified causal-geometric approach for learning under uncertainty in dynamic social environments. 

**Abstract (ZH)**: 基于因果关系的球面超图网络：在动态社交环境中的不确定性学习 

---
# Aligning Frozen LLMs by Reinforcement Learning: An Iterative Reweight-then-Optimize Approach 

**Title (ZH)**: 通过强化学习迭代重加权与优化方法对冻结的LLM进行对齐 

**Authors**: Xinnan Zhang, Chenliang Li, Siliang Zeng, Jiaxiang Li, Zhongruo Wang, Kaixiang Lin, Songtao Lu, Alfredo Garcia, Mingyi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2506.17828)  

**Abstract**: Aligning large language models (LLMs) with human preferences usually requires fine-tuning methods such as RLHF and DPO. These methods directly optimize the model parameters, so they cannot be used in test-time to improve model performance, nor are they applicable when the model weights are not accessible. In contrast, test-time methods sidestep weight updates by leveraging reward functions to guide and improve output quality. However, they incur high inference costs, and their one-shot guidance is often based on imperfect reward or value functions, leading to suboptimal outputs. In this work, we present a method named Iterative Reweight-then-Optimize (IRO), a reinforcement learning (RL) framework that performs RL-style alignment of the (frozen) base model without touching its parameters. During training, each iteration (i) samples candidates from the base model, (ii) resamples using current value functions, and (iii) trains a new lightweight value function that guides the next decoding pass. At test time, the value functions are used to guide the base model generation via a search-based optimization process. Notably, users can apply IRO to align a model on their own dataset, similar to OpenAI's reinforcement fine-tuning (RFT), but without requiring access to the model weights. 

**Abstract (ZH)**: Iterative Reweight-then-Optimize (IRO): A Reinforcement Learning Framework for aligning Large Language Models with Human Preferences without Access to Model Weights 

---
# Actionable Interpretability via Causal Hypergraphs: Unravelling Batch Size Effects in Deep Learning 

**Title (ZH)**: 基于因果超图的行为可解释性：揭开深度学习中的批量大小效应 

**Authors**: Zhongtian Sun, Anoushka Harit, Pietro Lio  

**Link**: [PDF](https://arxiv.org/pdf/2506.17826)  

**Abstract**: While the impact of batch size on generalisation is well studied in vision tasks, its causal mechanisms remain underexplored in graph and text domains. We introduce a hypergraph-based causal framework, HGCNet, that leverages deep structural causal models (DSCMs) to uncover how batch size influences generalisation via gradient noise, minima sharpness, and model complexity. Unlike prior approaches based on static pairwise dependencies, HGCNet employs hypergraphs to capture higher-order interactions across training dynamics. Using do-calculus, we quantify direct and mediated effects of batch size interventions, providing interpretable, causally grounded insights into optimisation. Experiments on citation networks, biomedical text, and e-commerce reviews show that HGCNet outperforms strong baselines including GCN, GAT, PI-GNN, BERT, and RoBERTa. Our analysis reveals that smaller batch sizes causally enhance generalisation through increased stochasticity and flatter minima, offering actionable interpretability to guide training strategies in deep learning. This work positions interpretability as a driver of principled architectural and optimisation choices beyond post hoc analysis. 

**Abstract (ZH)**: 基于超图的因果框架HGCNet：通过梯度噪声、最小值锋利度和模型复杂性探究批量大小对泛化的影响 

---
# Learning to Dock: A Simulation-based Study on Closing the Sim2Real Gap in Autonomous Underwater Docking 

**Title (ZH)**: 基于模拟的学习泊靠：自主水下泊靠的模拟到现实差距研究 

**Authors**: Kevin Chang, Rakesh Vivekanandan, Noah Pragin, Sean Bullock, Geoffrey Hollinger  

**Link**: [PDF](https://arxiv.org/pdf/2506.17823)  

**Abstract**: Autonomous Underwater Vehicle (AUV) docking in dynamic and uncertain environments is a critical challenge for underwater robotics. Reinforcement learning is a promising method for developing robust controllers, but the disparity between training simulations and the real world, or the sim2real gap, often leads to a significant deterioration in performance. In this work, we perform a simulation study on reducing the sim2real gap in autonomous docking through training various controllers and then evaluating them under realistic disturbances. In particular, we focus on the real-world challenge of docking under different payloads that are potentially outside the original training distribution. We explore existing methods for improving robustness including randomization techniques and history-conditioned controllers. Our findings provide insights into mitigating the sim2real gap when training docking controllers. Furthermore, our work indicates areas of future research that may be beneficial to the marine robotics community. 

**Abstract (ZH)**: 自主水下车辆（AUV）在动态和不确定环境下的自主对接是一项关键挑战，对于水下机器人技术而言。强化学习是开发稳健控制器的一种有前景的方法，但训练模拟与现实世界之间的差距或sim2real鸿沟通常会导致性能显著下降。在本文中，我们通过训练各种控制器并在现实扰动下评估它们，来研究减少自主对接中sim2real鸿沟的仿真研究。特别地，我们专注于不同载荷下的对接现实世界挑战，这些载荷可能是原始训练分布之外的。我们探讨了提高稳健性的现有方法，包括随机化技术及基于历史条件的控制器。我们的研究结果为训练对接控制器时缓解sim2real鸿沟提供了见解。此外，我们的工作还指出了对未来研究有益于海洋机器人社区的研究方向。 

---
# CultureMERT: Continual Pre-Training for Cross-Cultural Music Representation Learning 

**Title (ZH)**: 文化MERT：跨文化音乐表示学习的持续预训练 

**Authors**: Angelos-Nikolaos Kanatas, Charilaos Papaioannou, Alexandros Potamianos  

**Link**: [PDF](https://arxiv.org/pdf/2506.17818)  

**Abstract**: Recent advances in music foundation models have improved audio representation learning, yet their effectiveness across diverse musical traditions remains limited. We introduce CultureMERT-95M, a multi-culturally adapted foundation model developed to enhance cross-cultural music representation learning and understanding. To achieve this, we propose a two-stage continual pre-training strategy that integrates learning rate re-warming and re-decaying, enabling stable adaptation even with limited computational resources. Training on a 650-hour multi-cultural data mix, comprising Greek, Turkish, and Indian music traditions, results in an average improvement of 4.9% in ROC-AUC and AP across diverse non-Western music auto-tagging tasks, surpassing prior state-of-the-art, with minimal forgetting on Western-centric benchmarks. We further investigate task arithmetic, an alternative approach to multi-cultural adaptation that merges single-culture adapted models in the weight space. Task arithmetic performs on par with our multi-culturally trained model on non-Western auto-tagging tasks and shows no regression on Western datasets. Cross-cultural evaluation reveals that single-culture models transfer with varying effectiveness across musical traditions, whereas the multi-culturally adapted model achieves the best overall performance. To support research on world music representation learning, we publicly release CultureMERT-95M and CultureMERT-TA-95M, fostering the development of more culturally aware music foundation models. 

**Abstract (ZH)**: Recent Advances in Multi-Culturally Adapted Music Foundation Models: Enhancing Cross-Cultural Music Representation Learning and Understanding 

---
# RoboMonkey: Scaling Test-Time Sampling and Verification for Vision-Language-Action Models 

**Title (ZH)**: RoboMonkey: 扩展视觉-语言-动作模型测试时采样与验证规模 

**Authors**: Jacky Kwok, Christopher Agia, Rohan Sinha, Matt Foutter, Shulu Li, Ion Stoica, Azalia Mirhoseini, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2506.17811)  

**Abstract**: Vision-Language-Action (VLA) models have demonstrated remarkable capabilities in visuomotor control, yet ensuring their robustness in unstructured real-world environments remains a persistent challenge. In this paper, we investigate test-time scaling through the lens of sampling and verification as means to enhance the robustness and generalization of VLAs. We first demonstrate that the relationship between action error and the number of generated samples follows an exponentiated power law across a range of VLAs, indicating the existence of inference-time scaling laws. Building on these insights, we introduce RoboMonkey, a test-time scaling framework for VLAs. At deployment, RoboMonkey samples a small set of actions from a VLA, applies Gaussian perturbation and majority voting to construct an action proposal distribution, and then uses a Vision Language Model (VLM)-based verifier to select the optimal action. We propose a synthetic data generation pipeline for training such VLM-based action verifiers, and demonstrate that scaling the synthetic dataset consistently improves verification and downstream accuracy. Through extensive simulated and hardware experiments, we show that pairing existing VLAs with RoboMonkey yields significant performance gains, achieving a 25% absolute improvement on out-of-distribution tasks and 8% on in-distribution tasks. Additionally, when adapting to new robot setups, we show that fine-tuning both VLAs and action verifiers yields a 7% performance increase compared to fine-tuning VLAs alone. 

**Abstract (ZH)**: 基于视觉-语言-动作模型测试时的缩放以增强其在非结构化现实环境中的稳健性和泛化能力 

---
# Reimagining Parameter Space Exploration with Diffusion Models 

**Title (ZH)**: 重塑参数空间探索：基于扩散模型的方法 

**Authors**: Lijun Zhang, Xiao Liu, Hui Guan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17807)  

**Abstract**: Adapting neural networks to new tasks typically requires task-specific fine-tuning, which is time-consuming and reliant on labeled data. We explore a generative alternative that produces task-specific parameters directly from task identity, eliminating the need for task-specific training. To this end, we propose using diffusion models to learn the underlying structure of effective task-specific parameter space and synthesize parameters on demand. Once trained, the task-conditioned diffusion model can generate specialized weights directly from task identifiers. We evaluate this approach across three scenarios: generating parameters for a single seen task, for multiple seen tasks, and for entirely unseen tasks. Experiments show that diffusion models can generate accurate task-specific parameters and support multi-task interpolation when parameter subspaces are well-structured, but fail to generalize to unseen tasks, highlighting both the potential and limitations of this generative solution. 

**Abstract (ZH)**: 将神经网络适应新任务通常需要针对特定任务进行微调，这耗时且依赖于标记数据。我们探索了一种生成性替代方案，可以从任务身份直接生成任务特定参数，从而消除针对特定任务的训练需求。为此，我们提出使用扩散模型来学习有效的任务特定参数空间的基本结构，并按需合成参数。训练完成后，条件于任务的扩散模型可以直接从任务标识符生成专门的权重。我们在此方法中评估了三种场景：为单个已见过的任务生成参数、为多个已见过的任务生成参数以及为完全未见过的任务生成参数。实验结果显示，当参数子空间结构良好时，扩散模型可以生成准确的任务特定参数，并支持多任务插值，但无法泛化到未见过的任务，这突显了该生成性解决方案的潜力和局限性。 

---
# Expanding Relevance Judgments for Medical Case-based Retrieval Task with Multimodal LLMs 

**Title (ZH)**: 利用多模态大语言模型扩展医学案例检索任务的相关性评估 

**Authors**: Catarina Pires, Sérgio Nunes, Luís Filipe Teixeira  

**Link**: [PDF](https://arxiv.org/pdf/2506.17782)  

**Abstract**: Evaluating Information Retrieval (IR) systems relies on high-quality manual relevance judgments (qrels), which are costly and time-consuming to obtain. While pooling reduces the annotation effort, it results in only partially labeled datasets. Large Language Models (LLMs) offer a promising alternative to reducing reliance on manual judgments, particularly in complex domains like medical case-based retrieval, where relevance assessment requires analyzing both textual and visual information. In this work, we explore using a Multimodal Large Language Model (MLLM) to expand relevance judgments, creating a new dataset of automated judgments. Specifically, we employ Gemini 1.5 Pro on the ImageCLEFmed 2013 case-based retrieval task, simulating human assessment through an iteratively refined, structured prompting strategy that integrates binary scoring, instruction-based evaluation, and few-shot learning. We systematically experimented with various prompt configurations to maximize agreement with human judgments. To evaluate agreement between the MLLM and human judgments, we use Cohen's Kappa, achieving a substantial agreement score of 0.6, comparable to inter-annotator agreement typically observed in multimodal retrieval tasks. Starting from the original 15,028 manual judgments (4.72% relevant) across 35 topics, our MLLM-based approach expanded the dataset by over 37x to 558,653 judgments, increasing relevant annotations to 5,950. On average, each medical case query received 15,398 new annotations, with approximately 99% being non-relevant, reflecting the high sparsity typical in this domain. Our results demonstrate the potential of MLLMs to scale relevance judgment collection, offering a promising direction for supporting retrieval evaluation in medical and multimodal IR tasks. 

**Abstract (ZH)**: 利用多模态大型语言模型扩展相关性判断以支持医学和多模态信息检索评估 

---
# Toward Autonomous UI Exploration: The UIExplorer Benchmark 

**Title (ZH)**: 向自主UI探索迈进：UIExplorer基准测试 

**Authors**: Andrei Cristian Nica, Akshaya Vishnu Kudlu Shanbhogue, Harshil Shah, Aleix Cambray, Tudor Berariu, Lucas Maystre, David Barber  

**Link**: [PDF](https://arxiv.org/pdf/2506.17779)  

**Abstract**: Autonomous agents must know how to explore user interfaces (UIs) for reliable task solving, yet systematic evaluation of this crucial phase is lacking. We introduce UIExplore-Bench, the first benchmark explicitly dedicated to UI exploration. The benchmark evaluates agents with either Structured mode (granting access to layout information like DOM trees) or Screen mode (relying on GUI-only observations such as screenshots and human-like mouse/keyboard interactions) across three levels in a standardized GitLab sandbox environment. We formalize exploration as the process of maximizing the set of actionable UI components discovered and propose a metric, human-normalized UI-Functionalities Observed (hUFO), to quantify the effectiveness of exploration. Our results show that UIExplore-AlGo achieves the leading mean hUFO scores, reaching up to 77.2% of human performance in Structured mode and 59.0% in Screen mode at 2,000 steps, particularly excelling at the Sparse level. The results highlight the relevance of our benchmark, as current agents show a substantial performance gap compared to one hour of human expert exploration, indicating ample room for future advancements. We publicly release the benchmark environment, an exploration dataset, and an evaluation suite to catalyze research into efficient UI exploration strategies and their downstream applications, such as experience-driven task completion and automated training data generation. 

**Abstract (ZH)**: UIExplore-Bench：面向UI探索的首个基准 

---
# Machine Learning Model Integration with Open World Temporal Logic for Process Automation 

**Title (ZH)**: 基于开放世界时序逻辑的机器学习模型集成与过程自动化 

**Authors**: Dyuman Aditya, Colton Payne, Mario Leiva, Paulo Shakarian  

**Link**: [PDF](https://arxiv.org/pdf/2506.17776)  

**Abstract**: Recent advancements in Machine Learning (ML) have yielded powerful models capable of extracting structured information from diverse and complex data sources. However, a significant challenge lies in translating these perceptual or extractive outputs into actionable, reasoned decisions within complex operational workflows. To address these challenges, this paper introduces a novel approach that integrates the outputs from various machine learning models directly with the PyReason framework, an open-world temporal logic programming reasoning engine. PyReason's foundation in generalized annotated logic allows for the seamless incorporation of real-valued outputs (e.g., probabilities, confidence scores) from diverse ML models, treating them as truth intervals within its logical framework. Crucially, PyReason provides mechanisms, implemented in Python, to continuously poll ML model outputs, convert them into logical facts, and dynamically recompute the minimal model, ensuring real-tine adaptive decision-making. Furthermore, its native support for temporal reasoning, knowledge graph integration, and fully explainable interface traces enables sophisticated analysis over time-sensitive process data and existing organizational knowledge. By combining the strengths of perception and extraction from ML models with the logical deduction and transparency of PyReason, we aim to create a powerful system for automating complex processes. This integration finds utility across numerous domains, including manufacturing, healthcare, and business operations. 

**Abstract (ZH)**: 近期机器学习领域的进展产生了强大的模型，能够从多样化和复杂的数据源中提取结构化信息。然而，在复杂的操作工作流中将这些感知或提取输出转化为可行的决策仍面临重大挑战。为应对这些挑战，本文介绍了一种新的方法，该方法将各种机器学习模型的输出直接集成到PyReason框架中，PyReason是一个开放世界的时序逻辑编程推理引擎。PyReason基于通用标注逻辑，允许无缝地将来自不同机器学习模型的实值输出（例如，概率、置信分数）纳入其逻辑框架中，视作其逻辑框架中的真度区间。PyReason提供了用Python实现的机制，可以持续轮询机器学习模型输出，将其转换为逻辑事实，并动态重新计算最小模型，从而确保实时自适应决策。此外，其原生支持时间推理、知识图谱整合以及完全可解释的接口跟踪特性，使其能够对时间敏感的过程数据和现有组织知识进行复杂的分析。通过结合机器学习模型感知和提取的优点以及PyReason的逻辑推理和透明性，我们旨在创建一个强大的自动化复杂过程系统。这一整合在制造业、医疗保健和业务运营等多个领域具有广泛的应用。 

---
# CARTS: Collaborative Agents for Recommendation Textual Summarization 

**Title (ZH)**: CARTS: 联合代理的推荐文本总结 

**Authors**: Jiao Chen, Kehui Yao, Reza Yousefi Maragheh, Kai Zhao, Jianpeng Xu, Jason Cho, Evren Korpeoglu, Sushant Kumar, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17765)  

**Abstract**: Current recommendation systems often require some form of textual data summarization, such as generating concise and coherent titles for product carousels or other grouped item displays. While large language models have shown promise in NLP domains for textual summarization, these approaches do not directly apply to recommendation systems, where explanations must be highly relevant to the core features of item sets, adhere to strict word limit constraints. In this paper, we propose CARTS (Collaborative Agents for Recommendation Textual Summarization), a multi-agent LLM framework designed for structured summarization in recommendation systems. CARTS decomposes the task into three stages-Generation Augmented Generation (GAG), refinement circle, and arbitration, where successive agent roles are responsible for extracting salient item features, iteratively refining candidate titles based on relevance and length feedback, and selecting the final title through a collaborative arbitration process. Experiments on large-scale e-commerce data and live A/B testing show that CARTS significantly outperforms single-pass and chain-of-thought LLM baselines, delivering higher title relevance and improved user engagement metrics. 

**Abstract (ZH)**: 当前推荐系统往往需要某种形式的文本数据总结，比如为产品轮播或其他分组项显示生成精炼且连贯的标题。尽管大规模语言模型在自然语言处理领域展示了在文本总结方面的潜力，但在推荐系统中，这些方法并不直接适用，因为解释必须高度相关于项目集的核心特征，并严格遵守字数限制。在本文中，我们提出了CARTS（Collaborative Agents for Recommendation Textual Summarization），一个为推荐系统设计的多智能体LLM框架，用于结构化的总结。CARTS将任务分解为三个阶段：增强生成（GAG）、完善循环和仲裁，其中相继的角色负责提取关键项目特征、基于相关性和长度反馈迭代完善候选标题，并通过协作仲裁过程选择最终标题。在大规模电子商务数据上的实验和实时A/B测试显示，CARTS显著优于单次通过和思维链LLM基线，提供了更高的标题相关性和改进的用户参与度指标。 

---
# Residual Connection-Enhanced ConvLSTM for Lithium Dendrite Growth Prediction 

**Title (ZH)**: 基于残差连接增强的ConvLSTM锂枝晶生长预测 

**Authors**: Hosung Lee, Byeongoh Hwang, Dasan Kim, Myungjoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17756)  

**Abstract**: The growth of lithium dendrites significantly impacts the performance and safety of rechargeable batteries, leading to short circuits and capacity degradation. This study proposes a Residual Connection-Enhanced ConvLSTM model to predict dendrite growth patterns with improved accuracy and computational efficiency. By integrating residual connections into ConvLSTM, the model mitigates the vanishing gradient problem, enhances feature retention across layers, and effectively captures both localized dendrite growth dynamics and macroscopic battery behavior. The dataset was generated using a phase-field model, simulating dendrite evolution under varying conditions. Experimental results show that the proposed model achieves up to 7% higher accuracy and significantly reduces mean squared error (MSE) compared to conventional ConvLSTM across different voltage conditions (0.1V, 0.3V, 0.5V). This highlights the effectiveness of residual connections in deep spatiotemporal networks for electrochemical system modeling. The proposed approach offers a robust tool for battery diagnostics, potentially aiding in real-time monitoring and optimization of lithium battery performance. Future research can extend this framework to other battery chemistries and integrate it with real-world experimental data for further validation 

**Abstract (ZH)**: 锂枝晶生长对可充电电池的性能和安全性有显著影响，导致短路和容量衰退。本研究提出了一种残差连接增强的ConvLSTM模型，以提高枝晶生长模式预测的准确性和计算效率。通过将残差连接整合到ConvLSTM中，该模型缓解了梯度消失问题，增强了跨层特征保留，并有效地捕获了局部枝晶生长动力学和宏观电池行为。数据集使用相场模型生成，模拟了不同条件下枝晶的演变。实验结果表明，所提出模型在不同电压条件下（0.1V、0.3V、0.5V）的准确率最高可提高7%，并显著降低了均方误差（MSE）与传统ConvLSTM相比。这强调了残差连接在电化学系统建模的深时空网络中的有效性。所提出的方法为电池诊断提供了 robust 工具，可能有助于实时监测和优化锂离子电池的性能。未来的研究可以将此框架扩展到其他电池化学体系，并将其与实际实验数据集成以进行进一步验证。 

---
# HIDE and Seek: Detecting Hallucinations in Language Models via Decoupled Representations 

**Title (ZH)**: 隐藏与寻找：通过解耦表示检测语言模型中的幻觉 

**Authors**: Anwoy Chatterjee, Yash Goel, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2506.17748)  

**Abstract**: Contemporary Language Models (LMs), while impressively fluent, often generate content that is factually incorrect or unfaithful to the input context - a critical issue commonly referred to as 'hallucination'. This tendency of LMs to generate hallucinated content undermines their reliability, especially because these fabrications are often highly convincing and therefore difficult to detect. While several existing methods attempt to detect hallucinations, most rely on analyzing multiple generations per input, leading to increased computational cost and latency. To address this, we propose a single-pass, training-free approach for effective Hallucination detectIon via Decoupled rEpresentations (HIDE). Our approach leverages the hypothesis that hallucinations result from a statistical decoupling between an LM's internal representations of input context and its generated output. We quantify this decoupling using the Hilbert-Schmidt Independence Criterion (HSIC) applied to hidden-state representations extracted while generating the output sequence. We conduct extensive experiments on four diverse question answering datasets, evaluating both faithfulness and factuality hallucinations across six open-source LMs of varying scales and properties. Our results demonstrate that HIDE outperforms other single-pass methods in almost all settings, achieving an average relative improvement of ~29% in AUC-ROC over the best-performing single-pass strategy across various models and datasets. Additionally, HIDE shows competitive and often superior performance with multi-pass state-of-the-art methods, obtaining an average relative improvement of ~3% in AUC-ROC while consuming ~51% less computation time. Our findings highlight the effectiveness of exploiting internal representation decoupling in LMs for efficient and practical hallucination detection. 

**Abstract (ZH)**: 当前的语言模型虽然表现出色，但在生成内容时往往会出现事实错误或与输入上下文不一致的情况——这一问题通常被称为“幻觉”。我们提出了一种基于解耦表示的有效单步幻觉检测方法HIDE。我们的方法基于这样一个假设：幻觉源于语言模型内部对输入上下文和生成输出之间的统计解耦。我们使用希尔伯特-施密特独立性判别（HSIC）来量化生成输出序列时提取的隐藏态表示之间的解耦程度。我们在四个不同的问答数据集上进行了广泛实验，评估了六个规模和属性不同的开源语言模型在忠实性和事实性幻觉检测方面的表现。实验结果表明，HIDE在几乎所有场景下的性能都优于其他单步方法，平均ROC-AUC提高了约29%。此外，HIDE在平均ROC-AUC上的改进幅度接近3%，同时计算时间减少了约51%，与多步最新方法竞争并表现出更优性能。我们的研究结果强调了利用语言模型内部表示解耦进行高效和实用的幻觉检测的有效性。 

---
# KAG-Thinker: Teaching Large Language Models to Think with Human-like Reasoning Process 

**Title (ZH)**: KAG-Thinker: 教授大型语言模型使用类似-human推理过程进行思考 

**Authors**: Dalong Zhang, Jun Xu, Jun Zhou, Lei Liang, Lin Yuan, Ling Zhong, Mengshu Sun, Peilong Zhao, QiWei Wang, Xiaorui Wang, Xinkai Du, YangYang Hou, Yu Ao, ZhaoYang Wang, Zhengke Gui, ZhiYing Yi, Zhongpu Bo  

**Link**: [PDF](https://arxiv.org/pdf/2506.17728)  

**Abstract**: In this paper, we introduce KAG-Thinker, a novel human-like reasoning framework built upon a parameter-light large language model (LLM). Our approach enhances the logical coherence and contextual consistency of the thinking process in question-answering (Q\&A) tasks on domain-specific knowledge bases (KBs) within LLMs. This framework simulates human cognitive mechanisms for handling complex problems by establishing a structured thinking process. Continuing the \textbf{Logical Form} guided retrieval and reasoning technology route of KAG v0.7, firstly, it decomposes complex questions into independently solvable sub-problems(also referred to as logical forms) through \textbf{breadth decomposition}, each represented in two equivalent forms-natural language and logical function-and further classified as either Knowledge Retrieval or Reasoning Analysis tasks, with dependencies and variables passing explicitly modeled via logical function interfaces. In the solving process, the Retrieval function is used to perform knowledge retrieval tasks, while the Math and Deduce functions are used to perform reasoning analysis tasks. Secondly, it is worth noting that, in the Knowledge Retrieval sub-problem tasks, LLMs and external knowledge sources are regarded as equivalent KBs. We use the \textbf{knowledge boundary} model to determine the optimal source using self-regulatory mechanisms such as confidence calibration and reflective reasoning, and use the \textbf{depth solving} model to enhance the comprehensiveness of knowledge acquisition. Finally, instead of utilizing reinforcement learning, we employ supervised fine-tuning with multi-turn dialogues to align the model with our structured inference paradigm, thereby avoiding excessive reflection. This is supported by a data evaluation framework and iterative corpus synthesis, which facilitate the generation of detailed reasoning trajectories... 

**Abstract (ZH)**: 基于参数轻量大型语言模型的人类级推理框架KAG-Thinker：在领域特定知识库中的问题解答逻辑推理研究 

---
# Resolving the Ti-V Phase Diagram Discrepancy with First-Principles Calculations and Bayesian Learning 

**Title (ZH)**: 基于第一性原理计算和贝叶斯学习解决Ti-V相图差异问题 

**Authors**: Timofei Miryashkin, Olga Klimanova, Alexander Shapeev  

**Link**: [PDF](https://arxiv.org/pdf/2506.17719)  

**Abstract**: Conflicting experiments disagree on whether the titanium-vanadium (Ti-V) binary alloy exhibits a body-centred cubic (BCC) miscibility gap or remains completely soluble. A leading hypothesis attributes the miscibility gap to oxygen contamination during alloy preparation. To resolve this controversy, we use an ab initio + machine-learning workflow that couples an actively-trained Moment Tensor Potential to Bayesian thermodynamic inference. Using this workflow, we obtain Ti-V binary system across the entire composition range, together with confidence intervals in the thermodynamic limit. The resulting diagram reproduces all experimental features, demonstrating the robustness of our approach, and clearly favors the variant with a BCC miscibility gap terminating at T = 980 K and c = 0.67. Because oxygen was excluded from simulations, the gap cannot be attributed to impurity effects, contradicting recent CALPHAD reassessments. 

**Abstract (ZH)**: 钛-钒（Ti-V）二元合金是否存在体心立方（BCC）共熔区间仍存在实验争议：从第一性原理+机器学习工作流探究Ti-V二元系统的相图及其热力学稳定性分析 

---
# Aged to Perfection: Machine-Learning Maps of Age in Conversational English 

**Title (ZH)**: 完美老化：对话英语中的年龄机器学习地图 

**Authors**: MingZe Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17708)  

**Abstract**: The study uses the British National Corpus 2014, a large sample of contemporary spoken British English, to investigate language patterns across different age groups. Our research attempts to explore how language patterns vary between different age groups, exploring the connection between speaker demographics and linguistic factors such as utterance duration, lexical diversity, and word choice. By merging computational language analysis and machine learning methodologies, we attempt to uncover distinctive linguistic markers characteristic of multiple generations and create prediction models that can consistently estimate the speaker's age group from various aspects. This work contributes to our knowledge of sociolinguistic diversity throughout the life of modern British speech. 

**Abstract (ZH)**: 本研究使用2014年英国国家语料库，这一当代英式英语的大规模样本，探讨不同年龄组的语言模式。我们的研究尝试探索不同年龄组之间语言模式的差异，探究讲话者人口统计学特征与语用长度、词汇多样性、词汇选择等语言因素之间的联系。通过结合计算语言分析和机器学习方法，我们试图发现多个代际特征性的语言标记，并建立可以从多方面一致估计讲话者年龄组的预测模型。本项工作增进了我们对现代英式口语 生命周期中社会语言多样性 的理解。 

---
# Programmable-Room: Interactive Textured 3D Room Meshes Generation Empowered by Large Language Models 

**Title (ZH)**: 可编程房间：由大规模语言模型赋能的交互式纹理化3D房间网格生成 

**Authors**: Jihyun Kim, Junho Park, Kyeongbo Kong, Suk-Ju Kang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17707)  

**Abstract**: We present Programmable-Room, a framework which interactively generates and edits a 3D room mesh, given natural language instructions. For precise control of a room's each attribute, we decompose the challenging task into simpler steps such as creating plausible 3D coordinates for room meshes, generating panorama images for the texture, constructing 3D meshes by integrating the coordinates and panorama texture images, and arranging furniture. To support the various decomposed tasks with a unified framework, we incorporate visual programming (VP). VP is a method that utilizes a large language model (LLM) to write a Python-like program which is an ordered list of necessary modules for the various tasks given in natural language. We develop most of the modules. Especially, for the texture generating module, we utilize a pretrained large-scale diffusion model to generate panorama images conditioned on text and visual prompts (i.e., layout, depth, and semantic map) simultaneously. Specifically, we enhance the panorama image generation quality by optimizing the training objective with a 1D representation of a panorama scene obtained from bidirectional LSTM. We demonstrate Programmable-Room's flexibility in generating and editing 3D room meshes, and prove our framework's superiority to an existing model quantitatively and qualitatively. Project page is available in this https URL. 

**Abstract (ZH)**: Programmable-Room：一种基于自然语言指令的交互式3D房间网格生成与编辑框架 

---
# The Evolution of Natural Language Processing: How Prompt Optimization and Language Models are Shaping the Future 

**Title (ZH)**: 自然语言处理的发展：提示优化与语言模型如何塑造未来 

**Authors**: Summra Saleem, Muhammad Nabeel Asim, Shaista Zulfiqar, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2506.17700)  

**Abstract**: Large Language Models (LLMs) have revolutionized the field of Natural Language Processing (NLP) by automating traditional labor-intensive tasks and consequently accelerated the development of computer-aided applications. As researchers continue to advance this field with the introduction of novel language models and more efficient training/finetuning methodologies, the idea of prompt engineering and subsequent optimization strategies with LLMs has emerged as a particularly impactful trend to yield a substantial performance boost across diverse NLP tasks. To best of our knowledge numerous review articles have explored prompt engineering, however, a critical gap exists in comprehensive analyses of prompt optimization strategies. To bridge this gap this paper provides unique and comprehensive insights about the potential of diverse prompt optimization strategies. It analyzes their underlying working paradigms and based on these principles, categorizes them into 11 distinct classes. Moreover, the paper provides details about various NLP tasks where these prompt optimization strategies have been employed, along with details of different LLMs and benchmark datasets used for evaluation. This comprehensive compilation lays a robust foundation for future comparative studies and enables rigorous assessment of prompt optimization and LLM-based predictive pipelines under consistent experimental settings: a critical need in the current landscape. Ultimately, this research will centralize diverse strategic knowledge to facilitate the adaptation of existing prompt optimization strategies for development of innovative predictors across unexplored tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过自动化传统劳动密集型任务已彻底革新了自然语言处理（NLP）领域，并加速了计算机辅助应用的发展。随着研究人员通过引入新型语言模型和更高效的训练/微调方法继续推进这一领域，针对LLMs的提示工程及其优化策略的概念逐渐成为提升各类NLP任务显著性能的重要趋势。据我们所知，尽管已有大量综述文章探讨了提示工程，但在综合分析提示优化策略方面仍然存在关键空白。本文填补了这一空白，提供了关于各种提示优化策略潜在影响的独特且全面的见解，分析了它们的基本工作原理，并基于这些原则将它们分类为11种不同的类别。此外，本文详细介绍了这些提示优化策略在各种NLP任务中的应用情况，包括用于评估的不同大型语言模型和基准数据集的细节。这一全面的编纂为未来的对比研究奠定了坚实的基础，并在一致的实验设置下实现了对提示优化和基于大型语言模型的预测管道的严格评估：当前环境中的一项关键需求。最终，这项研究将集中各种战略知识以促进对现有提示优化策略的适应，开发适用于未探索任务的创新预测器。 

---
# Reinforcing User Interest Evolution in Multi-Scenario Learning for recommender systems 

**Title (ZH)**: 强化多场景学习中用户的兴趣演化推荐系统 

**Authors**: Zhijian Feng, Wenhao Zheng, Xuanji Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.17682)  

**Abstract**: In real-world recommendation systems, users would engage in variety scenarios, such as homepages, search pages, and related recommendation pages. Each of these scenarios would reflect different aspects users focus on. However, the user interests may be inconsistent in different scenarios, due to differences in decision-making processes and preference expression. This variability complicates unified modeling, making multi-scenario learning a significant challenge. To address this, we propose a novel reinforcement learning approach that models user preferences across scenarios by modeling user interest evolution across multiple scenarios. Our method employs Double Q-learning to enhance next-item prediction accuracy and optimizes contrastive learning loss using Q-value to make model performance better. Experimental results demonstrate that our approach surpasses state-of-the-art methods in multi-scenario recommendation tasks. Our work offers a fresh perspective on multi-scenario modeling and highlights promising directions for future research. 

**Abstract (ZH)**: 在现实世界的推荐系统中，用户会在多种场景中互动，如首页、搜索页面和相关推荐页面。每个场景都会反映用户关注的不同方面。然而，由于决策过程和偏好表达的不同，用户的兴趣在不同场景中可能不一致。这种变化性使统一建模变得复杂，使多场景学习成为一个重大挑战。为应对这一挑战，我们提出了一种新颖的强化学习方法，通过建模跨场景的用户兴趣演变来建模用户的偏好。我们的方法采用双重Q学习提高下一项预测的准确性，并使用Q值优化对比学习损失以提升模型性能。实验结果表明，我们的方法在多场景推荐任务中超过了最先进的方法。我们的工作为多场景建模提供了新的视角，并指出了未来研究的有前景方向。 

---
# Enhancing Stress-Strain Predictions with Seq2Seq and Cross-Attention based on Small Punch Test 

**Title (ZH)**: 基于小冲程试验的Seq2Seq与跨注意力机制增强应力-应变预测 

**Authors**: Zhengni Yang, Rui Yang, Weijian Han, Qixin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17680)  

**Abstract**: This paper introduces a novel deep-learning approach to predict true stress-strain curves of high-strength steels from small punch test (SPT) load-displacement data. The proposed approach uses Gramian Angular Field (GAF) to transform load-displacement sequences into images, capturing spatial-temporal features and employs a Sequence-to-Sequence (Seq2Seq) model with an LSTM-based encoder-decoder architecture, enhanced by multi-head cross-attention to improved accuracy. Experimental results demonstrate that the proposed approach achieves superior prediction accuracy, with minimum and maximum mean absolute errors of 0.15 MPa and 5.58 MPa, respectively. The proposed method offers a promising alternative to traditional experimental techniques in materials science, enhancing the accuracy and efficiency of true stress-strain relationship predictions. 

**Abstract (ZH)**: 本文提出了一种新颖的深度学习方法，从小凸缘试验(SPT)载荷-位移数据预测高强度钢的真实应力-应变曲线。该方法使用Gramian角场(GAF)将载荷-位移序列转换为图像，捕捉空间-时间特征，并采用基于LSTM的编码器-解码器架构的Sequence-to-Sequence (Seq2Seq)模型，通过多头交叉注意力机制提高预测精度。实验结果表明，所提出的方法在预测精度方面表现出色，最小和最大均方绝对误差分别为0.15 MPa和5.58 MPa。所提出的方法为材料科学中的传统实验技术提供了有希望的替代方案，提高了真实应力-应变关系预测的准确性和效率。 

---
# FaithfulSAE: Towards Capturing Faithful Features with Sparse Autoencoders without External Dataset Dependencies 

**Title (ZH)**: FaithfulSAE：如何在无需外部数据集依赖的情况下捕捉忠实特征的稀疏自编码器 

**Authors**: Seonglae Cho, Harryn Oh, Donghyun Lee, Luis Eduardo Rodrigues Vieira, Andrew Bermingham, Ziad El Sayed  

**Link**: [PDF](https://arxiv.org/pdf/2506.17673)  

**Abstract**: Sparse Autoencoders (SAEs) have emerged as a promising solution for decomposing large language model representations into interpretable features. However, Paulo and Belrose (2025) have highlighted instability across different initialization seeds, and Heap et al. (2025) have pointed out that SAEs may not capture model-internal features. These problems likely stem from training SAEs on external datasets - either collected from the Web or generated by another model - which may contain out-of-distribution (OOD) data beyond the model's generalisation capabilities. This can result in hallucinated SAE features, which we term "Fake Features", that misrepresent the model's internal activations. To address these issues, we propose FaithfulSAE, a method that trains SAEs on the model's own synthetic dataset. Using FaithfulSAEs, we demonstrate that training SAEs on less-OOD instruction datasets results in SAEs being more stable across seeds. Notably, FaithfulSAEs outperform SAEs trained on web-based datasets in the SAE probing task and exhibit a lower Fake Feature Ratio in 5 out of 7 models. Overall, our approach eliminates the dependency on external datasets, advancing interpretability by better capturing model-internal features while highlighting the often neglected importance of SAE training datasets. 

**Abstract (ZH)**: FaithfulSAE: Training Sparse Autoencoders on Model-Specific Synthetic Data for Enhanced Interpretability 

---
# TPTT: Transforming Pretrained Transformer into Titans 

**Title (ZH)**: TPTT: 将预训练变压器转化为巨擘 

**Authors**: Fabien Furfaro  

**Link**: [PDF](https://arxiv.org/pdf/2506.17671)  

**Abstract**: Recent advances in large language models (LLMs) have led to remarkable progress in natural language processing, but their computational and memory demands remain a significant challenge, particularly for long-context inference. We introduce TPTT (Transforming Pretrained Transformer into Titans), a novel framework for enhancing pretrained Transformer models with efficient linearized attention mechanisms and advanced memory management. TPTT employs techniques such as Memory as Gate (MaG) and mixed linearized attention (LiZA). It is fully compatible with the Hugging Face Transformers library, enabling seamless adaptation of any causal LLM through parameter-efficient fine-tuning (LoRA) without full retraining. We show the effectiveness of TPTT on the MMLU benchmark with models of approximately 1 billion parameters, observing substantial improvements in both efficiency and accuracy. For instance, Titans-Llama-3.2-1B achieves a 20% increase in Exact Match (EM) over its baseline. Statistical analyses and comparisons with recent state-of-the-art methods confirm the practical scalability and robustness of TPTT. Code is available at this https URL . Python package at this https URL . 

**Abstract (ZH)**: 最近在大型语言模型（LLMs）方面的进展引领了自然语言处理的显著进步，但它们的计算和内存需求仍然是一个重大挑战，尤其是在长语境推理方面。我们引入了一种名为TPTT（Transforming Pretrained Transformer into Titans）的新型框架，该框架通过高效的线性化注意力机制和先进的内存管理来增强预训练Transformer模型。TPTT采用的技术包括Memory as Gate（MaG）和混合线性化注意力（LiZA）。它完全兼容Hugging Face Transformers库，可以通过参数高效微调（LoRA）无缝适应任何因果LLM，而无需进行全面重训。我们在约10亿参数的MMLU基准上展示了TPTT的有效性，观察到在效率和准确性方面均实现了显著提升。例如，Titans-Llama-3.2-1B的精确匹配率（EM）相较于其基线提高了20%。统计分析和与最近的最先进方法的比较证实了TPTT的实际可扩展性和鲁棒性。代码可在以下链接获取：此 https URL 。Python包可在以下链接获取：此 https URL 。 

---
# RLRC: Reinforcement Learning-based Recovery for Compressed Vision-Language-Action Models 

**Title (ZH)**: 基于强化学习的压缩视觉-语言-行动模型恢复方法 

**Authors**: Yuxuan Chen, Xiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.17639)  

**Abstract**: Vision-Language-Action models (VLA) have demonstrated remarkable capabilities and promising potential in solving complex robotic manipulation tasks. However, their substantial parameter sizes and high inference latency pose significant challenges for real-world deployment, particularly on resource-constrained robotic platforms. To address this issue, we begin by conducting an extensive empirical study to explore the effectiveness of model compression techniques when applied to VLAs. Building on the insights gained from these preliminary experiments, we propose RLRC, a three-stage recovery method for compressed VLAs, including structured pruning, performance recovery based on SFT and RL, and further quantization. RLRC achieves up to an 8x reduction in memory usage and a 2.3x improvement in inference throughput, while maintaining or even surpassing the original VLA's task success rate. Extensive experiments show that RLRC consistently outperforms existing compression baselines, demonstrating strong potential for on-device deployment of VLAs. Project website: this https URL 

**Abstract (ZH)**: Vision-Language-Action模型（VLA）在解决复杂机器人 manipulation 任务方面展现了卓越的能力和广阔的应用潜力。然而，其庞大的参数量和高的推理延迟为其实用化部署，特别是在资源受限的机器人平台上，带来了重大挑战。为解决这一问题，我们首先进行了广泛的经验性研究，探讨在VLA中应用模型压缩技术的有效性。基于这些初步实验获得的洞见，我们提出了一种三阶段恢复方法RLRC，包括结构化修剪、基于SFT和RL的性能恢复以及进一步的量化。RLRC最高可实现8倍的内存使用量减少和2.3倍的推理吞吐量提升，并保持或超越了原始VLA的任务成功率。大量实验表明，RLRC在现有压缩基线中表现更优，展示了在设备上部署VLA的强大潜力。项目网站: this https URL。 

---
# Adaptive Multi-prompt Contrastive Network for Few-shot Out-of-distribution Detection 

**Title (ZH)**: 自适应多提示对比网络用于少样本域外检测 

**Authors**: Xiang Fang, Arvind Easwaran, Blaise Genest  

**Link**: [PDF](https://arxiv.org/pdf/2506.17633)  

**Abstract**: Out-of-distribution (OOD) detection attempts to distinguish outlier samples to prevent models trained on the in-distribution (ID) dataset from producing unavailable outputs. Most OOD detection methods require many IID samples for training, which seriously limits their real-world applications. To this end, we target a challenging setting: few-shot OOD detection, where {Only a few {\em labeled ID} samples are available.} Therefore, few-shot OOD detection is much more challenging than the traditional OOD detection setting. Previous few-shot OOD detection works ignore the distinct diversity between different classes. In this paper, we propose a novel network: Adaptive Multi-prompt Contrastive Network (AMCN), which adapts the ID-OOD separation boundary by learning inter- and intra-class distribution. To compensate for the absence of OOD and scarcity of ID {\em image samples}, we leverage CLIP, connecting text with images, engineering learnable ID and OOD {\em textual prompts}. Specifically, we first generate adaptive prompts (learnable ID prompts, label-fixed OOD prompts and label-adaptive OOD prompts). Then, we generate an adaptive class boundary for each class by introducing a class-wise threshold. Finally, we propose a prompt-guided ID-OOD separation module to control the margin between ID and OOD prompts. Experimental results show that AMCN outperforms other state-of-the-art works. 

**Abstract (ZH)**: 离分布（OOD）检测旨在区分异常样本，防止在分布内（ID）数据集上训练的模型产生不可用的输出。大多数OOD检测方法需要大量独立同分布（IID）样本进行训练，这严重限制了它们的实际应用。为此，我们针对一个具有挑战性的场景进行研究：少量样本的OOD检测，其中仅可用少量标记的ID样本。因此，少量样本的OOD检测比传统的OOD检测场景更具挑战性。以往的少量样本的OOD检测工作忽略了不同类别之间的独特多样性。在本文中，我们提出了一种新颖的网络：自适应多提示对比网络（AMCN），通过学习类别间的和类别内的分布来适应ID-OOD分离边界。为了弥补OOD样本不足和ID图像样本稀缺的问题，我们利用CLIP，将文本与图像进行连接，工程化生成可学习的ID和OOD文本提示。具体来说，我们首先生成自适应提示（可学习的ID提示、标签固定的不同类别自适应OOD提示）。然后，我们通过引入类别的阈值为每个类别生成自适应类边界。最后，我们提出了一种提示引导的ID-OOD分离模块，以控制ID提示与OOD提示之间的差距。实验结果显示，AMCN优于其他现有最佳方法。 

---
# LLM-Prompt: Integrated Heterogeneous Prompts for Unlocking LLMs in Time Series Forecasting 

**Title (ZH)**: LLM-Prompt:综合异构提示以解锁大规模语言模型在时间序列预测中的应用 

**Authors**: Zesen Wang, Yonggang Li, Lijuan Lan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17631)  

**Abstract**: Time series forecasting aims to model temporal dependencies among variables for future state inference, holding significant importance and widespread applications in real-world scenarios. Although deep learning-based methods have achieved remarkable progress, they still exhibit suboptimal performance in long-term forecasting and data-scarce scenarios. Recent research demonstrates that large language models (LLMs) achieve promising performance in time series forecasting. However, we find existing LLM-based methods still have shortcomings: (1) the absence of a unified paradigm for textual prompt formulation and (2) the neglect of modality discrepancies between textual prompts and time series. To address this, we propose LLM-Prompt, an LLM-based time series forecasting framework integrating multi-prompt information and cross-modal semantic alignment. Specifically, we first construct a unified textual prompt paradigm containing learnable soft prompts and textualized hard prompts. Second, to enhance LLMs' comprehensive understanding of the forecasting task, we design a semantic space embedding and cross-modal alignment module to achieve cross-modal fusion of temporal and textual information. Finally, the transformed time series from the LLMs are projected to obtain the forecasts. Comprehensive evaluations on 6 public datasets and 3 carbon emission datasets demonstrate that LLM-Prompt is a powerful framework for time series forecasting. 

**Abstract (ZH)**: 基于大语言模型的文本引导时间序列预测框架：LLM-Prompt 

---
# CLiViS: Unleashing Cognitive Map through Linguistic-Visual Synergy for Embodied Visual Reasoning 

**Title (ZH)**: CLiViS: 通过语言-视觉协同作用释放认知地图进行具身视觉推理 

**Authors**: Kailing Li, Qi'ao Xu, Tianwen Qian, Yuqian Fu, Yang Jiao, Xiaoling Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17629)  

**Abstract**: Embodied Visual Reasoning (EVR) seeks to follow complex, free-form instructions based on egocentric video, enabling semantic understanding and spatiotemporal reasoning in dynamic environments. Despite its promising potential, EVR encounters significant challenges stemming from the diversity of complex instructions and the intricate spatiotemporal dynamics in long-term egocentric videos. Prior solutions either employ Large Language Models (LLMs) over static video captions, which often omit critical visual details, or rely on end-to-end Vision-Language Models (VLMs) that struggle with stepwise compositional reasoning. Consider the complementary strengths of LLMs in reasoning and VLMs in perception, we propose CLiViS. It is a novel training-free framework that leverages LLMs for high-level task planning and orchestrates VLM-driven open-world visual perception to iteratively update the scene context. Building on this synergy, the core of CLiViS is a dynamic Cognitive Map that evolves throughout the reasoning process. This map constructs a structured representation of the embodied scene, bridging low-level perception and high-level reasoning. Extensive experiments across multiple benchmarks demonstrate the effectiveness and generality of CLiViS, especially in handling long-term visual dependencies. Code is available at this https URL. 

**Abstract (ZH)**: 具身视觉推理（EVR）旨在基于第一人称视频遵循复杂的自由形式指令，实现动态环境中的语义理解和时空推理。尽管EVR具有广阔的潜力，但它面临着源自复杂指令多样性和长时间第一人称视频中的错综复杂时空动态的巨大挑战。先前的解决方案要么依赖于在静态视频说明上使用的大型语言模型（LLMs），往往会忽略关键的视觉细节，要么依赖于端到端的视觉语言模型（VLMs），这种模型在逐步组合推理方面存在困难。考虑到LLMs在推理方面的优势和VLMs在感知方面的优势，我们提出了CLiViS。CLiViS是一种新型无需训练的框架，利用LLMs进行高层次任务规划，并通过VLM驱动的开放世界视觉感知迭代更新场景上下文。基于这种协同作用，CLiViS的核心是一个动态的认知地图，该地图在整个推理过程中不断发展。这个地图构建了具身场景的结构化表示，将低层感知与高层推理连接起来。多项基准测试中的广泛实验表明，CLiViS在处理长期视觉依赖关系方面特别有效。代码可在以下链接获取：this https URL。 

---
# Exploiting Efficiency Vulnerabilities in Dynamic Deep Learning Systems 

**Title (ZH)**: 利用动态深度学习系统中的效率漏洞 

**Authors**: Ravishka Rathnasuriya, Wei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17621)  

**Abstract**: The growing deployment of deep learning models in real-world environments has intensified the need for efficient inference under strict latency and resource constraints. To meet these demands, dynamic deep learning systems (DDLSs) have emerged, offering input-adaptive computation to optimize runtime efficiency. While these systems succeed in reducing cost, their dynamic nature introduces subtle and underexplored security risks. In particular, input-dependent execution pathways create opportunities for adversaries to degrade efficiency, resulting in excessive latency, energy usage, and potential denial-of-service in time-sensitive deployments. This work investigates the security implications of dynamic behaviors in DDLSs and reveals how current systems expose efficiency vulnerabilities exploitable by adversarial inputs. Through a survey of existing attack strategies, we identify gaps in the coverage of emerging model architectures and limitations in current defense mechanisms. Building on these insights, we propose to examine the feasibility of efficiency attacks on modern DDLSs and develop targeted defenses to preserve robustness under adversarial conditions. 

**Abstract (ZH)**: 动态深度学习系统中动态行为的安全性影响及其防御策略 

---
# Risk-Guided Diffusion: Toward Deploying Robot Foundation Models in Space, Where Failure Is Not An Option 

**Title (ZH)**: 风险管理导向的扩散：向着在不允许失败的空间场景中部署机器人基础模型的目标努力 

**Authors**: Rohan Thakker, Adarsh Patnaik, Vince Kurtz, Jonas Frey, Jonathan Becktor, Sangwoo Moon, Rob Royce, Marcel Kaufmann, Georgios Georgakis, Pascal Roth, Joel Burdick, Marco Hutter, Shehryar Khattak  

**Link**: [PDF](https://arxiv.org/pdf/2506.17601)  

**Abstract**: Safe, reliable navigation in extreme, unfamiliar terrain is required for future robotic space exploration missions. Recent generative-AI methods learn semantically aware navigation policies from large, cross-embodiment datasets, but offer limited safety guarantees. Inspired by human cognitive science, we propose a risk-guided diffusion framework that fuses a fast, learned "System-1" with a slow, physics-based "System-2", sharing computation at both training and inference to couple adaptability with formal safety. Hardware experiments conducted at the NASA JPL's Mars-analog facility, Mars Yard, show that our approach reduces failure rates by up to $4\times$ while matching the goal-reaching performance of learning-based robotic models by leveraging inference-time compute without any additional training. 

**Abstract (ZH)**: 极端 unfamiliar 地形下安全可靠的导航是未来机器人太空探索任务的必要要求。受人类认知科学启发，我们提出了一种风险导向扩散框架，该框架结合了快速的学习“系统-1”和缓慢的物理基础“系统-2”，在训练和推理阶段共享计算，以结合适应性和形式化安全性。在NASA JPL的火星模拟设施Mars Yard进行的硬件实验表明，我们的方法在不进行额外训练的情况下，通过利用推理时的计算资源将失败率降低最多4倍，同时匹配基于学习的机器人模型的目标到达性能。 

---
# DRAMA-X: A Fine-grained Intent Prediction and Risk Reasoning Benchmark For Driving 

**Title (ZH)**: DRAMA-X: 一种细粒度意图预测和风险推理基准驾驶数据集 

**Authors**: Mihir Godbole, Xiangbo Gao, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17590)  

**Abstract**: Understanding the short-term motion of vulnerable road users (VRUs) like pedestrians and cyclists is critical for safe autonomous driving, especially in urban scenarios with ambiguous or high-risk behaviors. While vision-language models (VLMs) have enabled open-vocabulary perception, their utility for fine-grained intent reasoning remains underexplored. Notably, no existing benchmark evaluates multi-class intent prediction in safety-critical situations, To address this gap, we introduce DRAMA-X, a fine-grained benchmark constructed from the DRAMA dataset via an automated annotation pipeline. DRAMA-X contains 5,686 accident-prone frames labeled with object bounding boxes, a nine-class directional intent taxonomy, binary risk scores, expert-generated action suggestions for the ego vehicle, and descriptive motion summaries. These annotations enable a structured evaluation of four interrelated tasks central to autonomous decision-making: object detection, intent prediction, risk assessment, and action suggestion. As a reference baseline, we propose SGG-Intent, a lightweight, training-free framework that mirrors the ego vehicle's reasoning pipeline. It sequentially generates a scene graph from visual input using VLM-backed detectors, infers intent, assesses risk, and recommends an action using a compositional reasoning stage powered by a large language model. We evaluate a range of recent VLMs, comparing performance across all four DRAMA-X tasks. Our experiments demonstrate that scene-graph-based reasoning enhances intent prediction and risk assessment, especially when contextual cues are explicitly modeled. 

**Abstract (ZH)**: 理解易受损道路使用者（如行人和骑行者）的短期运动对于安全的自动驾驶至关重要，特别是在具有模棱两可或高风险行为的城市场景中。虽然视觉-语言模型(VLMs)已经实现了开放词汇感知，但它们在细粒度意图推理方面的应用仍然不够探索。值得注意的是，目前没有任何基准数据集评估安全关键情况下多类意图预测。为填补这一空白，我们介绍了DRAMA-X，这是一个通过自动化注释流水线从DRAMA数据集构建的细粒度基准数据集。DRAMA-X包含5,686个易发生事故的帧，标记有对象边界框，九类方向意图分类，二元风险评分，为ego车辆生成的专家建议动作，以及描述性运动总结。这些注释使得能够对四个与自主决策相关的任务进行结构化的评估：对象检测、意图预测、风险评估和动作建议。作为参考基准，我们提出了SGG-Intent，这是一个轻量级、无需训练的框架，模仿ego车辆的推理流程。它依次从视觉输入生成场景图，使用VLM支持的检测器推断意图、评估风险，并通过大型语言模型驱动的组合推理阶段推荐动作。我们评估了多种最新的VLM，比较了它们在DRAMA-X四项任务上的性能。我们的实验表明，基于场景图的推理增强意图预测和风险评估，尤其是在对上下文线索进行了明确建模的情况下。 

---
# HalluRNN: Mitigating Hallucinations via Recurrent Cross-Layer Reasoning in Large Vision-Language Models 

**Title (ZH)**: HalluRNN：通过大型视觉语言模型中的递归跨层推理减轻幻觉问题 

**Authors**: Le Yu, Kaishen Wang, Jianlong Xiong, Yue Cao, Tao He  

**Link**: [PDF](https://arxiv.org/pdf/2506.17587)  

**Abstract**: Though Large Vision-Language Models (LVLMs) have achieved remarkable performance across various tasks, they are still prone to hallucinations-generating outputs that are textually plausible but visually ungrounded. While prior approaches generally address this issue through data-centric fine-tuning or innovative decoding strategies, these methods often require substantial resources or task-specific configurations. In this work, we introduce an architecture-level solution, HalluRNN, which enhances model stability through recurrent cross-layer reasoning. Specifically, we propose a novel Dual-Gated Depth Propagation Unit (DG-DPU) module, which is shared across layers and recurrently refines hidden states. This allows for the adaptive propagation of information throughout the model, enforces consistency across layers, and mitigates hallucinations caused by representational drift. By fine-tuning only the DG-DPU module, HalluRNN achieves strong and robust performance across multiple benchmarks. 

**Abstract (ZH)**: 尽管大型多模态模型（LVLMs）已经在各种任务上取得了显著的性能，但它们仍然容易产生幻觉——生成文本上合理但视觉上没有依据的输出。虽然现有的方法通常通过数据导向的微调或创新的解码策略来解决这一问题，但这些方法往往需要大量的资源或特定任务的配置。在这项工作中，我们提出了一种架构级别的解决方案——HalluRNN，该方案通过递归跨层推理增强了模型的稳定性。具体来说，我们提出了一种新颖的双门深度传播单元（DG-DPU）模块，该模块在各层之间共享并递归地细化隐藏状态。这使得信息在模型中的适应性传播成为可能，维护了各层的一致性，并减轻了由于表征漂移引起的幻觉。通过仅微调DG-DPU模块，HalluRNN在多个基准测试中取得了强健且稳定的性能。 

---
# Context-Aware Scientific Knowledge Extraction on Linked Open Data using Large Language Models 

**Title (ZH)**: 基于上下文的科学知识提取在链接开放数据中的大规模语言模型方法 

**Authors**: Sajratul Y. Rubaiat, Hasan M. Jamil  

**Link**: [PDF](https://arxiv.org/pdf/2506.17580)  

**Abstract**: The exponential growth of scientific literature challenges researchers extracting and synthesizing knowledge. Traditional search engines return many sources without direct, detailed answers, while general-purpose LLMs may offer concise responses that lack depth or omit current information. LLMs with search capabilities are also limited by context window, yielding short, incomplete answers. This paper introduces WISE (Workflow for Intelligent Scientific Knowledge Extraction), a system addressing these limits by using a structured workflow to extract, refine, and rank query-specific knowledge. WISE uses an LLM-powered, tree-based architecture to refine data, focusing on query-aligned, context-aware, and non-redundant information. Dynamic scoring and ranking prioritize unique contributions from each source, and adaptive stopping criteria minimize processing overhead. WISE delivers detailed, organized answers by systematically exploring and synthesizing knowledge from diverse sources. Experiments on HBB gene-associated diseases demonstrate WISE reduces processed text by over 80% while achieving significantly higher recall over baselines like search engines and other LLM-based approaches. ROUGE and BLEU metrics reveal WISE's output is more unique than other systems, and a novel level-based metric shows it provides more in-depth information. We also explore how the WISE workflow can be adapted for diverse domains like drug discovery, material science, and social science, enabling efficient knowledge extraction and synthesis from unstructured scientific papers and web sources. 

**Abstract (ZH)**: 智能科学知识提取的工作流（WISE）：一种结构化的工作流框架，用于提取、提炼和排序查询特定的知识 

---
# Optimizing Mastery Learning by Fast-Forwarding Over-Practice Steps 

**Title (ZH)**: 优化掌握学习通过跳过过度练习步骤实现快速增长 

**Authors**: Meng Xia, Robin Schmucker, Conrad Borchers, Vincent Aleven  

**Link**: [PDF](https://arxiv.org/pdf/2506.17577)  

**Abstract**: Mastery learning improves learning proficiency and efficiency. However, the overpractice of skills--students spending time on skills they have already mastered--remains a fundamental challenge for tutoring systems. Previous research has reduced overpractice through the development of better problem selection algorithms and the authoring of focused practice tasks. However, few efforts have concentrated on reducing overpractice through step-level adaptivity, which can avoid resource-intensive curriculum redesign. We propose and evaluate Fast-Forwarding as a technique that enhances existing problem selection algorithms. Based on simulation studies informed by learner models and problem-solving pathways derived from real student data, Fast-Forwarding can reduce overpractice by up to one-third, as it does not require students to complete problem-solving steps if all remaining pathways are fully mastered. Fast-Forwarding is a flexible method that enhances any problem selection algorithm, though its effectiveness is highest for algorithms that preferentially select difficult problems. Therefore, our findings suggest that while Fast-Forwarding may improve student practice efficiency, the size of its practical impact may also depend on students' ability to stay motivated and engaged at higher levels of difficulty. 

**Abstract (ZH)**: Mastery学习提高学习成效和效率，但技能的过度练习——学生花费时间在已掌握的技能上——仍然是辅导系统的一个基本挑战。先前的研究通过开发更好的问题选择算法和编写针对性练习任务来减少过度练习。然而，较少的研究集中于通过步骤级别的适应性来减少过度练习，这种方式可以避免耗费资源的课程再设计。我们提出并评估了“快进”作为一种增强现有问题选择算法的技术。基于由学习者模型和真实学生数据推导出的学习路径的仿真研究，“快进”可以通过跳过所有剩余路径均已完全掌握的问题解决步骤来减少多达三分之一的过度练习。作为一种灵活的方法，“快进”可以增强任何问题选择算法，尽管其有效性在优先选择困难问题的算法中最高。因此，我们的研究结果表明，“快进”可能会提高学生练习效率，但其实际影响的大小也可能取决于学生是否能在较高难度水平上保持动力和参与。 

---
# Accelerating Residual Reinforcement Learning with Uncertainty Estimation 

**Title (ZH)**: 基于不确定性估计加速残差强化学习 

**Authors**: Lakshita Dodeja, Karl Schmeckpeper, Shivam Vats, Thomas Weng, Mingxi Jia, George Konidaris, Stefanie Tellex  

**Link**: [PDF](https://arxiv.org/pdf/2506.17564)  

**Abstract**: Residual Reinforcement Learning (RL) is a popular approach for adapting pretrained policies by learning a lightweight residual policy that provides corrective actions. While Residual RL is more sample-efficient than finetuning the entire base policy, existing methods struggle with sparse rewards and are designed for deterministic base policies. We propose two improvements to Residual RL that further enhance its sample efficiency and make it suitable for stochastic base policies. First, we leverage uncertainty estimates of the base policy to focus exploration on regions in which the base policy is not confident. Second, we propose a simple modification to off-policy residual learning that allows it to observe base actions and better handle stochastic base policies. We evaluate our method with both Gaussian-based and Diffusion-based stochastic base policies on tasks from Robosuite and D4RL, and compare against state-of-the-art finetuning methods, demo-augmented RL methods, and other residual RL methods. Our algorithm significantly outperforms existing baselines in a variety of simulation benchmark environments. We also deploy our learned polices in the real world to demonstrate their robustness with zero-shot sim-to-real transfer. 

**Abstract (ZH)**: 残差强化学习（RL）是一种通过学习轻量级的残差策略来适应预训练策略的流行方法，该残差策略提供纠正动作。尽管残差RL在样本效率方面优于完全调优基策略，但现有方法在稀疏奖励方面表现不佳，并且主要设计用于确定性基策略。我们提出了两种改进残差RL的方法，以进一步提高其样本效率并使其适用于随机基策略。首先，我们利用基策略的不确定性估计，将探索集中在基策略不太自信的区域上。其次，我们提出了一种简单的方法修改，使得离策残差学习能够观察基策略动作，并更好地处理随机基策略。我们在Robosuite和D4RL的任务上使用基于高斯和扩散的随机基策略评估我们的方法，并与最先进的调优方法、演示增强RL方法和其他残差RL方法进行比较。我们的算法在多种模拟基准环境中显著优于现有基线方法。我们还在实际环境中部署我们的学习策略，以展示其零样本仿真实到现实transfer的鲁棒性。 

---
# VLA-OS: Structuring and Dissecting Planning Representations and Paradigms in Vision-Language-Action Models 

**Title (ZH)**: VLA-OS：构建与剖析视觉-语言-行动模型中的规划表示与范式 

**Authors**: Chongkai Gao, Zixuan Liu, Zhenghao Chi, Junshan Huang, Xin Fei, Yiwen Hou, Yuxuan Zhang, Yudi Lin, Zhirui Fang, Zeyu Jiang, Lin Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.17561)  

**Abstract**: Recent studies on Vision-Language-Action (VLA) models have shifted from the end-to-end action-generation paradigm toward a pipeline involving task planning followed by action generation, demonstrating improved performance on various complex, long-horizon manipulation tasks. However, existing approaches vary significantly in terms of network architectures, planning paradigms, representations, and training data sources, making it challenging for researchers to identify the precise sources of performance gains and components to be further improved. To systematically investigate the impacts of different planning paradigms and representations isolating from network architectures and training data, in this paper, we introduce VLA-OS, a unified VLA architecture series capable of various task planning paradigms, and design a comprehensive suite of controlled experiments across diverse object categories (rigid and deformable), visual modalities (2D and 3D), environments (simulation and real-world), and end-effectors (grippers and dexterous hands). Our results demonstrate that: 1) visually grounded planning representations are generally better than language planning representations; 2) the Hierarchical-VLA paradigm generally achieves superior or comparable performance than other paradigms on task performance, pretraining, generalization ability, scalability, and continual learning ability, albeit at the cost of slower training and inference speeds. 

**Abstract (ZH)**: Recent Studies on Vision-Language-Action (VLA) Models Have Shifted toward a Planning-Driven Pipeline with Improved Performance on Complex Manipulation Tasks: Introducing VLA-OS 

---
# Towards Zero-Shot Coordination between Teams of Agents: The N-XPlay Framework 

**Title (ZH)**: 零样本智能体团队之间的协调：N-XPlay框架 

**Authors**: Ava Abderezaei, Chi-Hui Lin, Joseph Miceli, Naren Sivagnanadasan, Stéphane Aroca-Ouellette, Jake Brawer, Alessandro Roncone  

**Link**: [PDF](https://arxiv.org/pdf/2506.17560)  

**Abstract**: Zero-shot coordination (ZSC) -- the ability to collaborate with unfamiliar partners -- is essential to making autonomous agents effective teammates. Existing ZSC methods evaluate coordination capabilities between two agents who have not previously interacted. However, these scenarios do not reflect the complexity of real-world multi-agent systems, where coordination often involves a hierarchy of sub-groups and interactions between teams of agents, known as Multi-Team Systems (MTS). To address this gap, we first introduce N-player Overcooked, an N-agent extension of the popular two-agent ZSC benchmark, enabling evaluation of ZSC in N-agent scenarios. We then propose N-XPlay for ZSC in N-agent, multi-team settings. Comparison against Self-Play across two-, three- and five-player Overcooked scenarios, where agents are split between an ``ego-team'' and a group of unseen collaborators shows that agents trained with N-XPlay are better able to simultaneously balance ``intra-team'' and ``inter-team'' coordination than agents trained with SP. 

**Abstract (ZH)**: 零样本协调（ZSC）——与不熟悉的合作伙伴协作的能力——是使自主代理成为有效队友的关键。现有的ZSC方法评估的是两个未曾互动的代理之间的协作能力。然而，这些场景未能反映现实世界多代理系统中的复杂性，在这些系统中，协调往往涉及子组层次结构和团队间代理的互动，称为多团队系统（MTS）。为了解决这一差距，我们首先引入了N-player Overcooked，这是流行的两代理ZSC基准的N代理扩展，使ZSC在N代理场景中的评估成为可能。然后，我们提出了N-XPlay，用于N代理和多团队设置中的ZSC。在两个、三个和五个玩家的Overcooked场景中，将代理分配给“自我团队”和一组未见合作者，相比使用自我博弈（Self-Play, SP）训练的代理，使用N-XPlay训练的代理能更好地同时平衡“团队内”和“团队间”协调。 

---
# SynDaCaTE: A Synthetic Dataset For Evaluating Part-Whole Hierarchical Inference 

**Title (ZH)**: SynDaCaTE: 一个用于评估部分-整体层次推理的合成数据集 

**Authors**: Jake Levi, Mark van der Wilk  

**Link**: [PDF](https://arxiv.org/pdf/2506.17558)  

**Abstract**: Learning to infer object representations, and in particular part-whole hierarchies, has been the focus of extensive research in computer vision, in pursuit of improving data efficiency, systematic generalisation, and robustness. Models which are \emph{designed} to infer part-whole hierarchies, often referred to as capsule networks, are typically trained end-to-end on supervised tasks such as object classification, in which case it is difficult to evaluate whether such a model \emph{actually} learns to infer part-whole hierarchies, as claimed. To address this difficulty, we present a SYNthetic DAtaset for CApsule Testing and Evaluation, abbreviated as SynDaCaTE, and establish its utility by (1) demonstrating the precise bottleneck in a prominent existing capsule model, and (2) demonstrating that permutation-equivariant self-attention is highly effective for parts-to-wholes inference, which motivates future directions for designing effective inductive biases for computer vision. 

**Abstract (ZH)**: 合成数据集用于胶囊网络测试与评估：SynDaCaTE 

---
# Research on Model Parallelism and Data Parallelism Optimization Methods in Large Language Model-Based Recommendation Systems 

**Title (ZH)**: 基于大规模语言模型的推荐系统中模型并行性和数据并行性优化方法的研究 

**Authors**: Haowei Yang, Yu Tian, Zhongheng Yang, Zhao Wang, Chengrui Zhou, Dannier Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.17551)  

**Abstract**: With the rapid adoption of large language models (LLMs) in recommendation systems, the computational and communication bottlenecks caused by their massive parameter sizes and large data volumes have become increasingly prominent. This paper systematically investigates two classes of optimization methods-model parallelism and data parallelism-for distributed training of LLMs in recommendation scenarios. For model parallelism, we implement both tensor parallelism and pipeline parallelism, and introduce an adaptive load-balancing mechanism to reduce cross-device communication overhead. For data parallelism, we compare synchronous and asynchronous modes, combining gradient compression and sparsification techniques with an efficient aggregation communication framework to significantly improve bandwidth utilization. Experiments conducted on a real-world recommendation dataset in a simulated service environment demonstrate that our proposed hybrid parallelism scheme increases training throughput by over 30% and improves resource utilization by approximately 20% compared to traditional single-mode parallelism, while maintaining strong scalability and robustness. Finally, we discuss trade-offs among different parallel strategies in online deployment and outline future directions involving heterogeneous hardware integration and automated scheduling technologies. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在推荐系统中的快速采用，其巨大的参数量和大数据量导致的计算和通信瓶颈日益突出。本文系统地研究了两种分布式训练优化方法——模型并行性和数据并行性——在推荐场景中训练LLMs的应用。在模型并行性方面，我们实现了张量并行性和管道并行性，并引入了一种自适应负载均衡机制以减少设备间通信开销。在数据并行性方面，我们比较了同步和异步模式，结合梯度压缩和稀疏化技术，并采用高效的聚合通信框架，以显著提高带宽利用率。实验结果表明，与传统的单模式并行性相比，我们提出的混合并行性方案可提高训练 throughput 超过 30%，并提高资源利用率约 20%，同时保持良好的可扩展性和鲁棒性。最后，我们讨论了不同并行策略在在线部署中的权衡，并概述了未来涉及异构硬件集成和自动化调度技术的发展方向。 

---
# ConsumerBench: Benchmarking Generative AI Applications on End-User Devices 

**Title (ZH)**: ConsumerBench: 在终端用户设备上基准测试生成式AI应用 

**Authors**: Yile Gu, Rohan Kadekodi, Hoang Nguyen, Keisuke Kamahori, Yiyu Liu, Baris Kasikci  

**Link**: [PDF](https://arxiv.org/pdf/2506.17538)  

**Abstract**: The recent shift in Generative AI (GenAI) applications from cloud-only environments to end-user devices introduces new challenges in resource management, system efficiency, and user experience. This paper presents ConsumerBench, a comprehensive benchmarking framework designed to evaluate the system efficiency and response time of GenAI models running on end-user devices. Unlike existing benchmarks that assume exclusive model access on dedicated GPUs, ConsumerBench simulates realistic multi-application scenarios executing concurrently on constrained hardware. Furthermore, ConsumerBench supports customizable workflows that simulate complex tasks requiring coordination among multiple applications. ConsumerBench captures both application-level metrics, including latency and Service Level Objective (SLO) attainment, and system-level metrics like CPU/GPU utilization and memory bandwidth. Through extensive experiments, ConsumerBench reveals inefficiencies in resource sharing, unfair scheduling under greedy allocation, and performance pitfalls of static model server configurations. The paper also provides practical insights for model developers and system designers, highlighting the benefits of custom kernels tailored to consumer-grade GPU architectures and the value of implementing SLO-aware scheduling strategies. 

**Abstract (ZH)**: 近期生成式人工智能（GenAI）应用从云环境向终端用户设备的转移引入了新的资源管理、系统效率和用户体验挑战。本文介绍了一种名为ConsumerBench的全面基准测试框架，用于评估运行在终端用户设备上的GenAI模型的系统效率和响应时间。与现有假设独家模型访问专用GPU的基准不同，ConsumerBench模拟了在受约束硬件上并发执行多个应用程序的现实场景。此外，ConsumerBench支持可定制的工作流，模拟需要多个应用程序协调的复杂任务。ConsumerBench捕获应用程序级别的指标，包括延迟和SLI（服务级别指标）达成情况，以及系统级别的指标，如CPU/GPU利用率和内存带宽。通过广泛的实验，ConsumerBench揭示了资源共享的低效性、贪婪分配下的不公平调度以及静态模型服务器配置的性能陷阱。本文还为模型开发人员和系统设计师提供了实用见解，强调了针对消费级GPU架构的定制内核以及实施具有SLI意识的调度策略的价值。 

---
# Exploring Strategies for Personalized Radiation Therapy Part I Unlocking Response-Related Tumor Subregions with Class Activation Mapping 

**Title (ZH)**: 探索个性化放疗策略 第一部分 通过类别激活映射解锁与响应相关的肿瘤亚区域 

**Authors**: Hao Peng, Steve Jiang, Robert Timmerman  

**Link**: [PDF](https://arxiv.org/pdf/2506.17536)  

**Abstract**: Personalized precision radiation therapy requires more than simple classification, it demands the identification of prognostic, spatially informative features and the ability to adapt treatment based on individual response. This study compares three approaches for predicting treatment response: standard radiomics, gradient based features, and convolutional neural networks enhanced with Class Activation Mapping. We analyzed 69 brain metastases from 39 patients treated with Gamma Knife radiosurgery. An integrated autoencoder classifier model was used to predict whether tumor volume would shrink by more than 20 percent at a three months follow up, framed as a binary classification task. The results highlight their strength in hierarchical feature extraction and the classifiers discriminative capacity. Among the models, pixel wise CAM provides the most detailed spatial insight, identifying lesion specific regions rather than relying on fixed patterns, demonstrating strong generalization. In non responding lesions, the activated regions may indicate areas of radio resistance. Pixel wise CAM outperformed both radiomics and gradient based methods in classification accuracy. Moreover, its fine grained spatial features allow for alignment with cellular level data, supporting biological validation and deeper understanding of heterogeneous treatment responses. Although further validation is necessary, these findings underscore the promise in guiding personalized and adaptive radiotherapy strategies for both photon and particle therapies. 

**Abstract (ZH)**: 个性化精准放疗要求的不仅仅是简单的分类，还需识别预后性、空间信息丰富的特征，并能够根据个体反应调整治疗方案。本研究比较了三种预测治疗反应的方法：标准放射omics、梯度特征以及结合Class Activation Mapping的卷积神经网络。我们分析了39名患者经伽玛刀放射手术治疗的69个脑转移瘤，使用集成自编码分类模型预测肿瘤体积在三个月随访时是否减少超过20%，将其视为二分类任务。结果强调了这些方法在层次特征提取和分类器判别能力方面的优势。在各种模型中，像素级CAM提供了最详细的空间洞察，能够识别病变特异性区域，而不是依赖固定模式，显示出较强的泛化能力。在未响应的病灶中，激活区域可能表明存在放疗抵抗。像素级CAM在分类准确性上优于放射omics和基于梯度的方法，其细粒度的空间特征允许与细胞水平数据对齐，支持生物学验证和对异质治疗反应的深入理解。尽管需要进一步验证，但这些发现强调了指导个性化和自适应放疗策略的潜力，适用于光子和粒子疗法。 

---
# Data Quality Issues in Multilingual Speech Datasets: The Need for Sociolinguistic Awareness and Proactive Language Planning 

**Title (ZH)**: 多语言语音数据中的数据质量问题：需要社会语言学意识和主动语言规划 

**Authors**: Mingfei Lau, Qian Chen, Yeming Fang, Tingting Xu, Tongzhou Chen, Pavel Golik  

**Link**: [PDF](https://arxiv.org/pdf/2506.17525)  

**Abstract**: Our quality audit for three widely used public multilingual speech datasets - Mozilla Common Voice 17.0, FLEURS, and VoxPopuli - shows that in some languages, these datasets suffer from significant quality issues. We believe addressing these issues will make these datasets more useful as training and evaluation sets, and improve downstream models. We divide these quality issues into two categories: micro-level and macro-level. We find that macro-level issues are more prevalent in less institutionalized, often under-resourced languages. We provide a case analysis of Taiwanese Southern Min (nan_tw) that highlights the need for proactive language planning (e.g. orthography prescriptions, dialect boundary definition) and enhanced data quality control in the process of Automatic Speech Recognition (ASR) dataset creation. We conclude by proposing guidelines and recommendations to mitigate these issues in future dataset development, emphasizing the importance of sociolinguistic awareness in creating robust and reliable speech data resources. 

**Abstract (ZH)**: 我们对广泛使用的三个公共多语言语音数据集——Mozilla Common Voice 17.0、FLEURS和VoxPopuli的质量审计显示，在某些语言中，这些数据集存在显著的质量问题。我们认为解决这些问题将使这些数据集在作为训练和评估集时更具有用途，并改善下游模型。我们将这些质量问题分为微观层面和宏观层面两类。我们发现，宏观层面的问题在制度化程度较低、often under-resourced 的语言中更为常见。我们通过对台语（nan_tw）进行案例分析，强调了自动语音识别（ASR）数据集创建过程中需要积极的语言规划（例如拼写规范、方言边界定义）和增强的数据质量控制。最后，我们提出了指导原则和建议，以在未来的数据集开发中减少这些问题，并强调在创造稳健可靠的声音数据资源时需要提高社会语言学意识。 

---
# A Survey of State Representation Learning for Deep Reinforcement Learning 

**Title (ZH)**: 深度强化学习中状态表示学习综述 

**Authors**: Ayoub Echchahed, Pablo Samuel Castro  

**Link**: [PDF](https://arxiv.org/pdf/2506.17518)  

**Abstract**: Representation learning methods are an important tool for addressing the challenges posed by complex observations spaces in sequential decision making problems. Recently, many methods have used a wide variety of types of approaches for learning meaningful state representations in reinforcement learning, allowing better sample efficiency, generalization, and performance. This survey aims to provide a broad categorization of these methods within a model-free online setting, exploring how they tackle the learning of state representations differently. We categorize the methods into six main classes, detailing their mechanisms, benefits, and limitations. Through this taxonomy, our aim is to enhance the understanding of this field and provide a guide for new researchers. We also discuss techniques for assessing the quality of representations, and detail relevant future directions. 

**Abstract (ZH)**: 无监督表示学习方法在无模型在线设置下处理顺序决策问题中复杂观测空间挑战的应用：方法分类与前景展望 

---
# Mapping the Evolution of Research Contributions using KnoVo 

**Title (ZH)**: 使用KnoVo映射研究贡献的演化 

**Authors**: Sajratul Y. Rubaiat, Syed N. Sakib, Hasan M. Jamil  

**Link**: [PDF](https://arxiv.org/pdf/2506.17508)  

**Abstract**: This paper presents KnoVo (Knowledge Evolution), an intelligent framework designed for quantifying and analyzing the evolution of research novelty in the scientific literature. Moving beyond traditional citation analysis, which primarily measures impact, KnoVo determines a paper's novelty relative to both prior and subsequent work within its multilayered citation network. Given a target paper's abstract, KnoVo utilizes Large Language Models (LLMs) to dynamically extract dimensions of comparison (e.g., methodology, application, dataset). The target paper is then compared to related publications along these same extracted dimensions. This comparative analysis, inspired by tournament selection, yields quantitative novelty scores reflecting the relative improvement, equivalence, or inferiority of the target paper in specific aspects. By aggregating these scores and visualizing their progression, for instance, through dynamic evolution graphs and comparative radar charts, KnoVo facilitates researchers not only to assess originality and identify similar work, but also to track knowledge evolution along specific research dimensions, uncover research gaps, and explore cross-disciplinary connections. We demonstrate these capabilities through a detailed analysis of 20 diverse papers from multiple scientific fields and report on the performance of various open-source LLMs within the KnoVo framework. 

**Abstract (ZH)**: 基于知识进化的智能框架：科研创新演化的量化与分析 

---
# From Generality to Mastery: Composer-Style Symbolic Music Generation via Large-Scale Pre-training 

**Title (ZH)**: 从通见到精通：通过大规模预训练实现作曲家风格的符号音乐生成 

**Authors**: Mingyang Yao, Ke Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.17497)  

**Abstract**: Despite progress in controllable symbolic music generation, data scarcity remains a challenge for certain control modalities. Composer-style music generation is a prime example, as only a few pieces per composer are available, limiting the modeling of both styles and fundamental music elements (e.g., melody, chord, rhythm). In this paper, we investigate how general music knowledge learned from a broad corpus can enhance the mastery of specific composer styles, with a focus on piano piece generation. Our approach follows a two-stage training paradigm. First, we pre-train a REMI-based music generation model on a large corpus of pop, folk, and classical music. Then, we fine-tune it on a small, human-verified dataset from four renowned composers, namely Bach, Mozart, Beethoven, and Chopin, using a lightweight adapter module to condition the model on style indicators. To evaluate the effectiveness of our approach, we conduct both objective and subjective evaluations on style accuracy and musicality. Experimental results demonstrate that our method outperforms ablations and baselines, achieving more precise composer-style modeling and better musical aesthetics. Additionally, we provide observations on how the model builds music concepts from the generality pre-training and refines its stylistic understanding through the mastery fine-tuning. 

**Abstract (ZH)**: 尽管在可控符号音乐生成方面取得了进展，但某些控制模式仍面临数据稀缺的挑战。作曲家风格的音乐生成就是一个典型的例子，每个作曲家的乐曲数量有限，限制了风格和基本音乐元素（如旋律、和弦、节奏）的建模。在本文中，我们研究广泛乐曲知识如何增强特定作曲家风格的专业技能，重点关注钢琴曲的生成。我们的方法采用两阶段训练 paradigm。首先，我们基于大量流行、民间和古典音乐的语料库对一种基于REMI的音乐生成模型进行预训练。然后，我们使用一个轻量级适配模块对模型进行微调，以便根据风格指标对模型进行条件限制，微调数据集来自四位著名的作曲家：巴赫、莫扎特、贝多芬和肖邦，由人类验证。为了评估我们方法的有效性，我们在风格准确性和音乐性方面进行了客观和主观评估。实验结果表明，我们的方法优于简化模型和基线，实现了更精确的作曲家风格建模和更好的音乐美学。此外，我们提供了关于模型如何从广泛预训练构建音乐概念并在专业微调中细化其风格理解的观察。 

---
# Exploring Strategies for Personalized Radiation Therapy Part II Predicting Tumor Drift Patterns with Diffusion Models 

**Title (ZH)**: 探索个性化学术放射治疗策略 II 基于扩散模型预测肿瘤位移模式 

**Authors**: Hao Peng, Steve Jiang, Robert Timmerman  

**Link**: [PDF](https://arxiv.org/pdf/2506.17491)  

**Abstract**: Radiation therapy outcomes are decided by two key parameters, dose and timing, whose best values vary substantially across patients. This variability is especially critical in the treatment of brain cancer, where fractionated or staged stereotactic radiosurgery improves safety compared to single fraction approaches, but complicates the ability to predict treatment response. To address this challenge, we employ Personalized Ultra-fractionated Stereotactic Adaptive Radiotherapy (PULSAR), a strategy that dynamically adjusts treatment based on how each tumor evolves over time. However, the success of PULSAR and other adaptive approaches depends on predictive tools that can guide early treatment decisions and avoid both overtreatment and undertreatment. However, current radiomics and dosiomics models offer limited insight into the evolving spatial and temporal patterns of tumor response. To overcome these limitations, we propose a novel framework using Denoising Diffusion Implicit Models (DDIM), which learns data-driven mappings from pre to post treatment imaging. In this study, we developed single step and iterative denoising strategies and compared their performance. The results show that diffusion models can effectively simulate patient specific tumor evolution and localize regions associated with treatment response. The proposed strategy provides a promising foundation for modeling heterogeneous treatment response and enabling early, adaptive interventions, paving the way toward more personalized and biologically informed radiotherapy. 

**Abstract (ZH)**: 个性化超分割立体适形自适应放疗（PULSAR）及其在放疗中的应用：基于去噪扩散隐模型的数据驱动肿瘤演变模拟 

---
# Distilling On-device Language Models for Robot Planning with Minimal Human Intervention 

**Title (ZH)**: 在设备端精简语言模型以实现最少人工介入的机器人规划 

**Authors**: Zachary Ravichandran, Ignacio Hounie, Fernando Cladera, Alejandro Ribeiro, George J. Pappas, Vijay Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.17486)  

**Abstract**: Large language models (LLMs) provide robots with powerful contextual reasoning abilities and a natural human interface. Yet, current LLM-enabled robots typically depend on cloud-hosted models, limiting their usability in environments with unreliable communication infrastructure, such as outdoor or industrial settings. We present PRISM, a framework for distilling small language model (SLM)-enabled robot planners that run on-device with minimal human supervision. Starting from an existing LLM-enabled planner, PRISM automatically synthesizes diverse tasks and environments, elicits plans from the LLM, and uses this synthetic dataset to distill a compact SLM as a drop-in replacement of the source model. We apply PRISM to three LLM-enabled planners for mapping and exploration, manipulation, and household assistance, and we demonstrate that PRISM improves the performance of Llama-3.2-3B from 10-20% of GPT-4o's performance to over 93% - using only synthetic data. We further demonstrate that the distilled planners generalize across heterogeneous robotic platforms (ground and aerial) and diverse environments (indoor and outdoor). We release all software, trained models, and datasets at this https URL. 

**Abstract (ZH)**: 基于设备端的小语言模型使能机器人规划框架：PRISM 

---
# Computational Approaches to Understanding Large Language Model Impact on Writing and Information Ecosystems 

**Title (ZH)**: 理解大规模语言模型对写作和信息生态系统影响的计算方法 

**Authors**: Weixin Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17467)  

**Abstract**: Large language models (LLMs) have shown significant potential to change how we write, communicate, and create, leading to rapid adoption across society. This dissertation examines how individuals and institutions are adapting to and engaging with this emerging technology through three research directions. First, I demonstrate how the institutional adoption of AI detectors introduces systematic biases, particularly disadvantaging writers of non-dominant language varieties, highlighting critical equity concerns in AI governance. Second, I present novel population-level algorithmic approaches that measure the increasing adoption of LLMs across writing domains, revealing consistent patterns of AI-assisted content in academic peer reviews, scientific publications, consumer complaints, corporate communications, job postings, and international organization press releases. Finally, I investigate LLMs' capability to provide feedback on research manuscripts through a large-scale empirical analysis, offering insights into their potential to support researchers who face barriers in accessing timely manuscript feedback, particularly early-career researchers and those from under-resourced settings. 

**Abstract (ZH)**: 大型语言模型（LLMs）展现出显著潜力，可以改变我们的写作、沟通和创造方式，导致其在社会中迅速被采用。本论文通过三个研究方向探讨个人和机构如何适应并参与这一新兴技术：首先，我展示了机构采用AI检测器引入系统性偏见，特别是对非主流语言变体的写作者不利，突显了AI治理中的关键公平性问题；其次，我提出了新颖的群体级算法方法来衡量LLMs在各种写作领域中的采用情况，揭示了AI辅助内容在学术同行评审、科学出版物、消费者投诉、企业通讯、招聘信息和国际组织新闻公告中的一致性模式；最后，我通过大规模实证分析探讨LLMs在科研手稿反馈方面的能力，提供有关其支持面临及时手稿反馈障碍的研究人员的见解，特别是早期职业研究人员和来自资源不足地区的研究人员。 

---
# FedNAMs: Performing Interpretability Analysis in Federated Learning Context 

**Title (ZH)**: FedNAMs：在联邦学习背景下进行可解释性分析 

**Authors**: Amitash Nanda, Sree Bhargavi Balija, Debashis Sahoo  

**Link**: [PDF](https://arxiv.org/pdf/2506.17466)  

**Abstract**: Federated learning continues to evolve but faces challenges in interpretability and explainability. To address these challenges, we introduce a novel approach that employs Neural Additive Models (NAMs) within a federated learning framework. This new Federated Neural Additive Models (FedNAMs) approach merges the advantages of NAMs, where individual networks concentrate on specific input features, with the decentralized approach of federated learning, ultimately producing interpretable analysis results. This integration enhances privacy by training on local data across multiple devices, thereby minimizing the risks associated with data centralization and improving model robustness and generalizability. FedNAMs maintain detailed, feature-specific learning, making them especially valuable in sectors such as finance and healthcare. They facilitate the training of client-specific models to integrate local updates, preserve privacy, and mitigate concerns related to centralization. Our studies on various text and image classification tasks, using datasets such as OpenFetch ML Wine, UCI Heart Disease, and Iris, show that FedNAMs deliver strong interpretability with minimal accuracy loss compared to traditional Federated Deep Neural Networks (DNNs). The research involves notable findings, including the identification of critical predictive features at both client and global levels. Volatile acidity, sulfates, and chlorides for wine quality. Chest pain type, maximum heart rate, and number of vessels for heart disease. Petal length and width for iris classification. This approach strengthens privacy and model efficiency and improves interpretability and robustness across diverse datasets. Finally, FedNAMs generate insights on causes of highly and low interpretable features. 

**Abstract (ZH)**: federated learning在可解释性和可说明性方面不断演进但仍面临挑战。为应对这些挑战，我们提出了一种新颖的方法，该方法在联邦学习框架中采用了神经加性模型（NAMs）。这一新的联邦神经加性模型（FedNAMs）方法结合了NAMs的优势，即各个网络专注于特定的输入特征，以及联邦学习的分散化方法，从而产生可解释的分析结果。这种整合通过在多设备上的本地数据上进行训练增强了隐私性，减少了数据集中化带来的风险，并提高了模型的稳健性和泛化能力。FedNAMs保持了详细的、特征特定的训练，使其在金融和医疗保健等行业尤为有价值。它们促进了客户端特定模型的训练，以整合局部更新、保护隐私并缓解集中化的关切。我们在各种文本和图像分类任务上进行的研究，使用了如OpenFetch ML Wine、UCI Heart Disease和Iris等数据集，表明FedNAMs在可解释性方面表现出色，且相对于传统的联邦深度神经网络（DNNs）仅有轻微的准确性损失。这项研究包括重要发现，如识别出葡萄酒质量的临界预测特征（挥发酸、硫酸盐和氯化物），心脏病的特征（胸痛类型、最大心率和血管数量），以及鸢尾花分类的特征（花瓣长度和宽度）。这种方法增强了隐私和模型效率，并在多种数据集上改进了可解释性和鲁棒性。最后，FedNAMs还提供了关于高可解释性和低可解释性特征原因的洞见。 

---
# General-Purpose Robotic Navigation via LVLM-Orchestrated Perception, Reasoning, and Acting 

**Title (ZH)**: 通用机器人导航通过LVLM协调感知、推理和行动 

**Authors**: Bernard Lange, Anil Yildiz, Mansur Arief, Shehryar Khattak, Mykel Kochenderfer, Georgios Georgakis  

**Link**: [PDF](https://arxiv.org/pdf/2506.17462)  

**Abstract**: Developing general-purpose navigation policies for unknown environments remains a core challenge in robotics. Most existing systems rely on task-specific neural networks and fixed data flows, limiting generalizability. Large Vision-Language Models (LVLMs) offer a promising alternative by embedding human-like knowledge suitable for reasoning and planning. Yet, prior LVLM-robot integrations typically depend on pre-mapped spaces, hard-coded representations, and myopic exploration. We introduce the Agentic Robotic Navigation Architecture (ARNA), a general-purpose navigation framework that equips an LVLM-based agent with a library of perception, reasoning, and navigation tools available within modern robotic stacks. At runtime, the agent autonomously defines and executes task-specific workflows that iteratively query the robotic modules, reason over multimodal inputs, and select appropriate navigation actions. This approach enables robust navigation and reasoning in previously unmapped environments, providing a new perspective on robotic stack design. Evaluated in Habitat Lab on the HM-EQA benchmark, ARNA achieves state-of-the-art performance, demonstrating effective exploration, navigation, and embodied question answering without relying on handcrafted plans, fixed input representations, or pre-existing maps. 

**Abstract (ZH)**: 开发适用于未知环境的一般导航政策仍然是机器人领域的核心挑战。现有的大多数系统依赖于任务特定的神经网络和固定的数据流，限制了泛化能力。大型视觉-语言模型（LVLM）通过嵌入适合推理和规划的人类知识提供了有前景的替代方案。然而，之前的LVLM-机器人集成通常依赖预映射的空间、硬编码的表示和短视的探索。我们引入了Agentic Robotic Navigation Architecture（ARNA），这是一种一般用途的导航框架，为基于LVLM的代理配备了现代机器人堆栈内部可用的感知、推理和导航工具。在运行时，代理自主定义和执行迭代查询机器人模块、处理多模态输入并选择适当导航动作的具体工作流程。这种方法能够在未映射的环境中实现稳健的导航和推理，为机器人堆栈设计提供了新的视角。在Habitat Lab上的HM-EQA基准测试中，ARNA达到了最佳性能，展示了在不依赖手工设计的计划、固定输入表示或预先存在的地图的情况下进行有效的探索、导航和具身问题解答的能力。 

---
# Trans${^2}$-CBCT: A Dual-Transformer Framework for Sparse-View CBCT Reconstruction 

**Title (ZH)**: Trans${^2}$-CBCT: 一种用于稀视角CBCT重建的双变换器框架 

**Authors**: Minmin Yang, Huantao Ren, Senem Velipasalar  

**Link**: [PDF](https://arxiv.org/pdf/2506.17425)  

**Abstract**: Cone-beam computed tomography (CBCT) using only a few X-ray projection views enables faster scans with lower radiation dose, but the resulting severe under-sampling causes strong artifacts and poor spatial coverage. We address these challenges in a unified framework. First, we replace conventional UNet/ResNet encoders with TransUNet, a hybrid CNN-Transformer model. Convolutional layers capture local details, while self-attention layers enhance global context. We adapt TransUNet to CBCT by combining multi-scale features, querying view-specific features per 3D point, and adding a lightweight attenuation-prediction head. This yields Trans-CBCT, which surpasses prior baselines by 1.17 dB PSNR and 0.0163 SSIM on the LUNA16 dataset with six views. Second, we introduce a neighbor-aware Point Transformer to enforce volumetric coherence. This module uses 3D positional encoding and attention over k-nearest neighbors to improve spatial consistency. The resulting model, Trans$^2$-CBCT, provides an additional gain of 0.63 dB PSNR and 0.0117 SSIM. Experiments on LUNA16 and ToothFairy show consistent gains from six to ten views, validating the effectiveness of combining CNN-Transformer features with point-based geometry reasoning for sparse-view CBCT reconstruction. 

**Abstract (ZH)**: 仅用少量X射线投影视图的锥束计算机断层摄影（CBCT）能够实现更快的扫描并降低辐射剂量，但由于严重的欠采样导致强烈的伪影和较差的空间覆盖率。我们在统一框架中解决这些挑战。首先，我们用混合CNN-Transformer模型TransUNet替代传统的UNet/ResNet编码器。卷积层捕捉局部细节，而自我注意力层增强全局上下文。我们将TransUNet适应CBCT，通过结合多尺度特征、每个3D点查询视图特定特征以及添加一个轻量级的衰减预测头部。这产生Trans-CBCT，该模型在六视图LUNA16数据集上优于先前基线1.17 dB PSNR和0.0163 SSIM。其次，我们引入一种邻域意识点变换器以强化体素一致性。该模块使用3D位置编码和k最近邻注意力增强空间一致性。由此产生的模型Trans²-CBCT提供了额外的增益，分别为0.63 dB PSNR和0.0117 SSIM。实验表明，在LUNA16和ToothFairy数据集上从六视图到十视图都能获得一致的增益，验证了结合CNN-Transformer特征与基于点的几何推理对稀视角CBCT重建的有效性。 

---
# UProp: Investigating the Uncertainty Propagation of LLMs in Multi-Step Agentic Decision-Making 

**Title (ZH)**: UProp：探究大规模语言模型在多步代理决策中的不确定性传播 

**Authors**: Jinhao Duan, James Diffenderfer, Sandeep Madireddy, Tianlong Chen, Bhavya Kailkhura, Kaidi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17419)  

**Abstract**: As Large Language Models (LLMs) are integrated into safety-critical applications involving sequential decision-making in the real world, it is essential to know when to trust LLM decisions. Existing LLM Uncertainty Quantification (UQ) methods are primarily designed for single-turn question-answering formats, resulting in multi-step decision-making scenarios, e.g., LLM agentic system, being underexplored. In this paper, we introduce a principled, information-theoretic framework that decomposes LLM sequential decision uncertainty into two parts: (i) internal uncertainty intrinsic to the current decision, which is focused on existing UQ methods, and (ii) extrinsic uncertainty, a Mutual-Information (MI) quantity describing how much uncertainty should be inherited from preceding decisions. We then propose UProp, an efficient and effective extrinsic uncertainty estimator that converts the direct estimation of MI to the estimation of Pointwise Mutual Information (PMI) over multiple Trajectory-Dependent Decision Processes (TDPs). UProp is evaluated over extensive multi-step decision-making benchmarks, e.g., AgentBench and HotpotQA, with state-of-the-art LLMs, e.g., GPT-4.1 and DeepSeek-V3. Experimental results demonstrate that UProp significantly outperforms existing single-turn UQ baselines equipped with thoughtful aggregation strategies. Moreover, we provide a comprehensive analysis of UProp, including sampling efficiency, potential applications, and intermediate uncertainty propagation, to demonstrate its effectiveness. Codes will be available at this https URL. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）被集成到涉及序列决策的安全关键应用中，了解何时信任LLM决策变得至关重要。现有的LLM不确定性量化（UQ）方法主要针对单轮问答格式设计，导致多步决策场景，例如LLM代理系统，尚未得到充分探索。本文介绍了一个原则性的信息论框架，将LLM序列决策不确定性分解为两部分：（i）内在于当前决策的不确定性，这是现有UQ方法关注的焦点；（ii）外在不确定性，这是一个互信息（MI）量度，描述了从先前决策继承多少不确定性。然后提出UProp，一个高效且有效的外在不确定性估计器，将直接估计MI转换为对多个轨迹依赖决策过程（TDPs）的点wise互信息（PMI）估计。UProp在广泛的多步决策基准测试中，如AgentBench和HotpotQA，与最先进的LLMs，如GPT-4.1和DeepSeek-V3进行了评估。实验结果表明，UProp显著优于现有的单轮UQ基线，并结合了精心的聚合策略。此外，我们对UProp进行了全面分析，包括采样效率、潜在应用和中间不确定性传播，以展示其有效性。代码将在以下网址提供：this https URL。 

---
# Challenges in Grounding Language in the Real World 

**Title (ZH)**: 在现实世界中grounding语言的挑战 

**Authors**: Peter Lindes, Kaoutar Skiker  

**Link**: [PDF](https://arxiv.org/pdf/2506.17375)  

**Abstract**: A long-term goal of Artificial Intelligence is to build a language understanding system that allows a human to collaborate with a physical robot using language that is natural to the human. In this paper we highlight some of the challenges in doing this, and propose a solution that integrates the abilities of a cognitive agent capable of interactive task learning in a physical robot with the linguistic abilities of a large language model. We also point the way to an initial implementation of this approach. 

**Abstract (ZH)**: 人工智能的一个长期目标是构建一个语言理解系统，使人类能够使用自然语言与物理机器人协作。在本文中，我们阐述了实现这一目标的一些挑战，并提出了一种解决方案，即将能够进行交互式任务学习的认知代理能力与大规模语言模型的语文能力整合到物理机器人中。我们还指出了这种方法初步实现的方向。 

---
# From Drawings to Decisions: A Hybrid Vision-Language Framework for Parsing 2D Engineering Drawings into Structured Manufacturing Knowledge 

**Title (ZH)**: 从图纸到决策：一种解析二维工程图纸为结构化制造知识的视觉-语言混合框架 

**Authors**: Muhammad Tayyab Khan, Lequn Chen, Zane Yong, Jun Ming Tan, Wenhe Feng, Seung Ki Moon  

**Link**: [PDF](https://arxiv.org/pdf/2506.17374)  

**Abstract**: Efficient and accurate extraction of key information from 2D engineering drawings is essential for advancing digital manufacturing workflows. Such information includes geometric dimensioning and tolerancing (GD&T), measures, material specifications, and textual annotations. Manual extraction is slow and labor-intensive, while generic OCR models often fail due to complex layouts, engineering symbols, and rotated text, leading to incomplete and unreliable outputs. These limitations result in incomplete and unreliable outputs. To address these challenges, we propose a hybrid vision-language framework that integrates a rotation-aware object detection model (YOLOv11-obb) with a transformer-based vision-language parser. Our structured pipeline applies YOLOv11-OBB to localize annotations and extract oriented bounding box (OBB) patches, which are then parsed into structured outputs using a fine-tuned, lightweight vision-language model (VLM). We curate a dataset of 1,367 2D mechanical drawings annotated across nine key categories. YOLOv11-OBB is trained on this dataset to detect OBBs and extract annotation patches. These are parsed using two open-source VLMs: Donut and Florence-2. Both models are lightweight and well-suited for specialized industrial tasks under limited computational overhead. Following fine-tuning of both models on the curated dataset of image patches paired with structured annotation labels, a comparative experiment is conducted to evaluate parsing performance across four key metrics. Donut outperforms Florence-2, achieving 88.5% precision, 99.2% recall, and a 93.5% F1-score, with a hallucination rate of 11.5%. Finally, a case study demonstrates how the extracted structured information supports downstream manufacturing tasks such as process and tool selection, showcasing the practical utility of the proposed framework in modernizing 2D drawing interpretation. 

**Abstract (ZH)**: 高效的2D工程图中关键信息的提取对于推动数字化制造流程至关重要。这类信息包括几何尺寸和公差（GD&T）、度量、材料规格以及文本注释。手动提取速度慢且劳动密集，而通用的OCR模型由于复杂布局、工程符号和旋转文本往往失效，导致不完整且不可靠的输出。为了解决这些挑战，我们提出了一种结合旋转感知对象检测模型（YOLOv11-obb）与变压器基元语视觉解析器的混合视觉-语言框架。我们的结构化管道使用YOLOv11-OBB定位注释并提取定向边界框（OBB）补丁，随后使用微调的轻量级视觉-语言模型（VLM）对这些补丁进行解析，生成结构化输出。我们收集了一个包含1,367张2D机械图纸的数据集，并标记了九个关键类别。YOLOv11-OBB在该数据集上进行训练以检测OBB和提取注释补丁。这些补丁使用两个开源VLM（Donut和Florence-2）进行解析。这两种模型都很轻量且适合在有限的计算开销下执行专门的工业任务。经过在标记了结构化注释标签的图像补丁数据集上微调这两种模型后，我们进行了一项比较实验以评估它们在四项关键指标上的解析性能。Donut的表现优于Florence-2，实现88.5%的精确率、99.2%的召回率以及93.5%的F1分数，并有11.5%的幻觉率。最后，案例研究展示了提取到的结构化信息如何支持后续的制造任务，如工艺和工具选择，从而突显了所提出框架在现代化工图纸解读中的实际用途。 

---
# Multimodal Political Bias Identification and Neutralization 

**Title (ZH)**: 多模态政治偏见识别与中和 

**Authors**: Cedric Bernard, Xavier Pleimling, Amun Kharel, Chase Vickery  

**Link**: [PDF](https://arxiv.org/pdf/2506.17372)  

**Abstract**: Due to the presence of political echo chambers, it becomes imperative to detect and remove subjective bias and emotionally charged language from both the text and images of political articles. However, prior work has focused on solely the text portion of the bias rather than both the text and image portions. This is a problem because the images are just as powerful of a medium to communicate information as text is. To that end, we present a model that leverages both text and image bias which consists of four different steps. Image Text Alignment focuses on semantically aligning images based on their bias through CLIP models. Image Bias Scoring determines the appropriate bias score of images via a ViT classifier. Text De-Biasing focuses on detecting biased words and phrases and neutralizing them through BERT models. These three steps all culminate to the final step of debiasing, which replaces the text and the image with neutralized or reduced counterparts, which for images is done by comparing the bias scores. The results so far indicate that this approach is promising, with the text debiasing strategy being able to identify many potential biased words and phrases, and the ViT model showcasing effective training. The semantic alignment model also is efficient. However, more time, particularly in training, and resources are needed to obtain better results. A human evaluation portion was also proposed to ensure semantic consistency of the newly generated text and images. 

**Abstract (ZH)**: 由于存在政治回音室，检测并去除政治文章中文字和图像中的主观偏见和情绪化语言变得至关重要。然而，以往研究主要关注文字部分的偏见而非文字和图像两部分。鉴于图像同样是强有力的传播信息的媒介，我们提出了一种结合文字和图像偏见的模型，该模型包含四个步骤。图像文字对齐旨在通过CLIP模型在语义层面对图像进行对齐。图像偏见评分通过ViT分类器确定图像的适当偏见分数。文字脱偏重点在于检测具有偏见的词汇和短语并通过BERT模型进行中立化。这些三个步骤最终汇聚到脱偏的最后一步，即用中立化或减弱后的文字和图像替换原文字和图像，对图像而言是通过比较偏见分数实现的。初步结果表明，该方法具有潜力，文字脱偏策略能够识别出许多潜在的偏见词汇和短语，ViT模型也展示了有效的训练效果。语义对齐模型也具有高效性。不过，还需要更多时间和资源来获得更好的结果，并提议进行人工评估以确保新生成的文字和图像的一致性。 

---
# AI based Content Creation and Product Recommendation Applications in E-commerce: An Ethical overview 

**Title (ZH)**: 基于AI的内容创作与产品推荐应用在电子商务中的伦理概述 

**Authors**: Aditi Madhusudan Jain, Ayush Jain  

**Link**: [PDF](https://arxiv.org/pdf/2506.17370)  

**Abstract**: As e-commerce rapidly integrates artificial intelligence for content creation and product recommendations, these technologies offer significant benefits in personalization and efficiency. AI-driven systems automate product descriptions, generate dynamic advertisements, and deliver tailored recommendations based on consumer behavior, as seen in major platforms like Amazon and Shopify. However, the widespread use of AI in e-commerce raises crucial ethical challenges, particularly around data privacy, algorithmic bias, and consumer autonomy. Bias -- whether cultural, gender-based, or socioeconomic -- can be inadvertently embedded in AI models, leading to inequitable product recommendations and reinforcing harmful stereotypes. This paper examines the ethical implications of AI-driven content creation and product recommendations, emphasizing the need for frameworks to ensure fairness, transparency, and need for more established and robust ethical standards. We propose actionable best practices to remove bias and ensure inclusivity, such as conducting regular audits of algorithms, diversifying training data, and incorporating fairness metrics into AI models. Additionally, we discuss frameworks for ethical conformance that focus on safeguarding consumer data privacy, promoting transparency in decision-making processes, and enhancing consumer autonomy. By addressing these issues, we provide guidelines for responsibly utilizing AI in e-commerce applications for content creation and product recommendations, ensuring that these technologies are both effective and ethically sound. 

**Abstract (ZH)**: 电子商务中基于人工智能的内容创作与产品推荐的伦理影响：公平、透明与包容的最佳实践 

---
# Re-Evaluating Code LLM Benchmarks Under Semantic Mutation 

**Title (ZH)**: 重新评估语义变异下的代码LLM基准 

**Authors**: Zhiyuan Pan, Xing Hu, Xin Xia, Xiaohu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17369)  

**Abstract**: In the era of large language models (LLMs), code benchmarks have become an important research area in software engineering and are widely used by practitioners. These benchmarks evaluate the performance of LLMs on specific code-related tasks, such as code understanding and generation. A critical step in constructing code benchmarks is the design of prompts. However, as existing code benchmarks typically rely on a single prompt template per task, they are prone to the issue of prompt sensitivity, where minor prompt variations could result in substantial performance variations, leading to unreliable evaluations of model capabilities.
While previous studies have explored prompt sensitivity, their experimental designs and findings are limited to traditional natural language processing (NLP) tasks. In this paper, we present an empirical study to investigate prompt sensitivity in code benchmarks. We first propose a general framework that modifies prompt templates in a manner that preserves both their semantics and their structure as much as possible. Based on the framework, we conduct extensive experiments across eight code benchmark tasks on 10 representative open-source LLMs, with each task featuring 100 semantically similar prompt templates. We then analyze the evaluation results using various statistical metrics, focusing on both absolute and relative model performance. Our findings suggest that even slight prompt variations can lead to significant shifts in performance. Additionally, we observe that such variations can introduce inconsistencies in the performance rankings across different models. These insights highlight the need for considering prompt sensitivity when designing future code benchmarks, to ensure more reliable and accurate evaluation of LLM capabilities. 

**Abstract (ZH)**: 在大型语言模型时代，代码基准已成为软件工程中的一个重要研究领域，广泛应用于实践。这些基准评估大型语言模型在特定代码任务上的表现，如代码理解与生成。构建代码基准的关键步骤之一是设计提示。然而，由于现有代码基准通常每个任务仅依赖一个提示模板，这使得它们容易受到提示敏感性问题的影响，即微小的提示变化可能导致显著的性能变化，从而导致对模型能力的不可靠评估。

尽管先前的研究已经探索了提示敏感性，但其实验设计和发现仅限于传统的自然语言处理任务。本文提出一项实证研究，以探讨代码基准中的提示敏感性。我们首先提出了一种通用框架，该框架以尽可能保留提示语义和结构的方式修改提示模板。基于该框架，我们对10个代表性的开源大型语言模型进行了广泛的实验，每个任务包含100个语义相似的提示模板，共计8个代码基准任务。然后，我们使用多种统计指标分析评估结果，重点关注绝对和相对模型表现。我们的发现表明，即使微小的提示变化也可能导致性能显著变化。我们还观察到，这些变化会在不同模型之间引入表现排名的一致性问题。这些见解强调，在设计未来代码基准时需要考虑提示敏感性，以确保对大型语言模型能力的更可靠和准确评估。 

---
# SAFEx: Analyzing Vulnerabilities of MoE-Based LLMs via Stable Safety-critical Expert Identification 

**Title (ZH)**: SAFEx：通过稳定的关键专家识别分析基于MoE的LLM的安全漏洞 

**Authors**: Zhenglin Lai, Mengyao Liao, Dong Xu, Zebin Zhao, Zhihang Yuan, Chao Fan, Jianqiang Li, Bingzhe Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17368)  

**Abstract**: Large language models based on Mixture-of-Experts have achieved substantial gains in efficiency and scalability, yet their architectural uniqueness introduces underexplored safety alignment challenges. Existing safety alignment strategies, predominantly designed for dense models, are ill-suited to address MoE-specific vulnerabilities. In this work, we formalize and systematically study MoE model's positional vulnerability - the phenomenon where safety-aligned behaviors rely on specific expert modules, revealing critical risks inherent to MoE architectures. To this end, we present SAFEx, an analytical framework that robustly identifies, characterizes, and validates the safety-critical experts using a novel Stability-based Expert Selection (SES) algorithm. Notably, our approach enables the explicit decomposition of safety-critical experts into distinct functional groups, including those responsible for harmful content detection and those controlling safe response generation. Extensive experiments on mainstream MoE models, such as the recently released Qwen3-MoE, demonstrated that their intrinsic safety mechanisms heavily rely on a small subset of positional experts. Disabling these experts significantly compromised the models' ability to refuse harmful requests. For Qwen3-MoE with 6144 experts (in the FNN layer), we find that disabling as few as 12 identified safety-critical experts can cause the refusal rate to drop by 22%, demonstrating the disproportionate impact of a small set of experts on overall model safety. 

**Abstract (ZH)**: 基于Mixture-of-Experts的大语言模型在效率和可扩展性方面取得了显著进步，但其独特的架构引入了未被充分探索的安全对齐挑战。现有的安全对齐策略主要针对密集模型设计，不适合作为MoE特定漏洞的解决方案。在本文中，我们正式化并系统研究了MoE模型的位置性脆弱性——安全对齐行为依赖于特定的专家模块的现象，揭示了MoE架构固有的关键风险。为此，我们提出了SAFEx，一种分析框架，利用一种新颖的基于稳定性的专家选择（SES）算法，稳健地识别、描述和验证安全关键专家。值得注意的是，我们的方法能够将安全关键专家明确分解为不同的功能组，包括负责有害内容检测和控制安全响应生成的专家。在主流MoE模型（如近期发布的Qwen3-MoE）上进行的广泛实验表明，这些模型内部的安全机制高度依赖于一组特定位置的专家。禁用这些专家极大地削弱了模型拒绝有害请求的能力。对于具有6144个专家（在FNN层）的Qwen3-MoE，我们发现禁用12个识别出的安全关键专家会使得拒绝率下降22%，展示了少数几个专家对整体模型安全性的重要影响。 

---
# Cash or Comfort? How LLMs Value Your Inconvenience 

**Title (ZH)**: 现金还是方便？大规模语言模型如何看待你的不便。 

**Authors**: Mateusz Cedro, Timour Ichmoukhamedov, Sofie Goethals, Yifan He, James Hinns, David Martens  

**Link**: [PDF](https://arxiv.org/pdf/2506.17367)  

**Abstract**: Large Language Models (LLMs) are increasingly proposed as near-autonomous artificial intelligence (AI) agents capable of making everyday decisions on behalf of humans. Although LLMs perform well on many technical tasks, their behaviour in personal decision-making remains less understood. Previous studies have assessed their rationality and moral alignment with human decisions. However, the behaviour of AI assistants in scenarios where financial rewards are at odds with user comfort has not yet been thoroughly explored. In this paper, we tackle this problem by quantifying the prices assigned by multiple LLMs to a series of user discomforts: additional walking, waiting, hunger and pain. We uncover several key concerns that strongly question the prospect of using current LLMs as decision-making assistants: (1) a large variance in responses between LLMs, (2) within a single LLM, responses show fragility to minor variations in prompt phrasing (e.g., reformulating the question in the first person can considerably alter the decision), (3) LLMs can accept unreasonably low rewards for major inconveniences (e.g., 1 Euro to wait 10 hours), and (4) LLMs can reject monetary gains where no discomfort is imposed (e.g., 1,000 Euro to wait 0 minutes). These findings emphasize the need for scrutiny of how LLMs value human inconvenience, particularly as we move toward applications where such cash-versus-comfort trade-offs are made on users' behalf. 

**Abstract (ZH)**: 大型语言模型在财务奖励与用户舒适度冲突情境下的行为探究 

---
# AI-based Multimodal Biometrics for Detecting Smartphone Distractions: Application to Online Learning 

**Title (ZH)**: 基于AI的多模态生物特征识别智能手机分心检测：在线学习应用 

**Authors**: Alvaro Becerra, Roberto Daza, Ruth Cobos, Aythami Morales, Mutlu Cukurova, Julian Fierrez  

**Link**: [PDF](https://arxiv.org/pdf/2506.17364)  

**Abstract**: This work investigates the use of multimodal biometrics to detect distractions caused by smartphone use during tasks that require sustained attention, with a focus on computer-based online learning. Although the methods are applicable to various domains, such as autonomous driving, we concentrate on the challenges learners face in maintaining engagement amid internal (e.g., motivation), system-related (e.g., course design) and contextual (e.g., smartphone use) factors. Traditional learning platforms often lack detailed behavioral data, but Multimodal Learning Analytics (MMLA) and biosensors provide new insights into learner attention. We propose an AI-based approach that leverages physiological signals and head pose data to detect phone use. Our results show that single biometric signals, such as brain waves or heart rate, offer limited accuracy, while head pose alone achieves 87%. A multimodal model combining all signals reaches 91% accuracy, highlighting the benefits of integration. We conclude by discussing the implications and limitations of deploying these models for real-time support in online learning environments. 

**Abstract (ZH)**: 本研究探讨了多模态生物特征识别在检测执行需要持续注意力的任务时因智能手机使用引起的分心现象中的应用，重点关注基于计算机的在线学习领域。尽管所采用的方法适用于多个领域，例如自动驾驶，但我们着重于学习者在面对内在（例如，动机）、系统相关（例如，课程设计）和情境因素（例如，智能手机使用）带来的挑战时保持参与的困难。传统的学习平台通常缺乏详细的行为数据，但多模态学习分析（MMLA）和生物传感器提供了关于学习者注意力的新见解。我们提出了一种基于人工智能的方法，利用生理信号和头部姿态数据来检测手机使用情况。研究结果表明，单一的生物特征信号（如脑电波或心率）的准确性有限，而单独使用头部姿态的准确性为87%。结合所有信号的多模态模型的准确性达到91%，这突显了集成的优势。最后，我们讨论了在在线学习环境中部署这些模型的implications和限制。 

---
# A Large-Scale Real-World Evaluation of LLM-Based Virtual Teaching Assistant 

**Title (ZH)**: 大规模现实世界中基于LLM的虚拟教学助手评估 

**Authors**: Sunjun Kweon, Sooyohn Nam, Hyunseung Lim, Hwajung Hong, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17363)  

**Abstract**: Virtual Teaching Assistants (VTAs) powered by Large Language Models (LLMs) have the potential to enhance student learning by providing instant feedback and facilitating multi-turn interactions. However, empirical studies on their effectiveness and acceptance in real-world classrooms are limited, leaving their practical impact uncertain. In this study, we develop an LLM-based VTA and deploy it in an introductory AI programming course with 477 graduate students. To assess how student perceptions of the VTA's performance evolve over time, we conduct three rounds of comprehensive surveys at different stages of the course. Additionally, we analyze 3,869 student--VTA interaction pairs to identify common question types and engagement patterns. We then compare these interactions with traditional student--human instructor interactions to evaluate the VTA's role in the learning process. Through a large-scale empirical study and interaction analysis, we assess the feasibility of deploying VTAs in real-world classrooms and identify key challenges for broader adoption. Finally, we release the source code of our VTA system, fostering future advancements in AI-driven education: \texttt{this https URL}. 

**Abstract (ZH)**: 由大规模语言模型（LLMs）驱动的虚拟教学助手（VTAs）有潜力通过提供即时反馈和促进多轮交互来增强学生学习。然而，关于其在真实课堂中的有效性和接受度的实证研究有限，其实际影响尚不确定。在本研究中，我们开发了一个基于LLM的VTA，并将其部署在一门面向477名研究生的 introductory AI 编程课程中。为了评估学生对VTA性能的看法随着时间的推移如何演变，我们在课程的不同阶段进行了三次全面调查。此外，我们分析了3,869个学生-VTA交互对以识别常见问题类型和参与模式。然后，我们将这些交互与传统的学生-人类讲师交互进行比较，以评估VTA在学习过程中的角色。通过大规模的实证研究和交互分析，我们评估了在真实课堂中部署VTAs的可行性，并指出了更广泛采用的关键挑战。最后，我们发布了我们VTA系统的源代码，促进了人工智能驱动教育的未来发展：\texttt{this https URL}。 

---
# Speeding up Local Optimization in Vehicle Routing with Tensor-based GPU Acceleration 

**Title (ZH)**: 基于张量的GPU加速在车辆路线问题中的局部优化加速 

**Authors**: Zhenyu Lei, Jin-Kao Hao, Qinghua Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17357)  

**Abstract**: Local search plays a central role in many effective heuristic algorithms for the vehicle routing problem (VRP) and its variants. However, neighborhood exploration is known to be computationally expensive and time consuming, especially for large instances or problems with complex constraints. In this study, we explore a promising direction to address this challenge by introducing an original tensor-based GPU acceleration method designed to speed up the commonly used local search operators in vehicle routing. By using an attribute-based representation, the method offers broad extensibility, making it applicable to different VRP variants. Its low-coupling architecture, with intensive computations completely offloaded to the GPU, ensures seamless integration in various local search-based algorithms and frameworks, leading to significant improvements in computational efficiency and potentially improved solution quality. Through comparative experiments on benchmark instances of three routing problems, we demonstrate the substantial computational advantages of the proposed approach over traditional CPU-based implementations. We also provide a detailed analysis of the strengths and limitations of the method, providing valuable insights into its performance characteristics and identifying potential bottlenecks in practical applications. These findings contribute to a better understanding and suggest directions for future improvements. 

**Abstract (ZH)**: 局部搜索在车辆路线问题及其变体的许多有效启发式算法中发挥着核心作用。然而，邻域探索计算昂贵且耗时，尤其是在大型实例或具有复杂约束的问题中。本研究通过引入一种基于张量的GPU加速方法，旨在加速车辆路线中常用的地方搜索操作符，以解决这一挑战。该方法通过属性表示提供了广泛的可扩展性，使其适用于不同的车辆路线变体。其低耦合架构将密集计算完全卸载到GPU上，确保在各种基于局部搜索的算法和框架中无缝集成，从而大大提高计算效率并有可能提高解的质量。通过在三种路由问题的标准测试实例上进行比较实验，我们展示了所提出方法相对于传统CPU实现的显著计算优势。我们还详细分析了该方法的优势和局限性，提供了对其性能特征的见解，并确定了其实用应用中的潜在瓶颈。这些发现有助于更好地理解并为未来改进指明方向。 

---
# Automatic Large Language Models Creation of Interactive Learning Lessons 

**Title (ZH)**: 自动创建交互式学习课程的大规模语言模型方法 

**Authors**: Jionghao Lin, Jiarui Rao, Yiyang Zhao, Yuting Wang, Ashish Gurung, Amanda Barany, Jaclyn Ocumpaugh, Ryan S. Baker, Kenneth R. Koedinger  

**Link**: [PDF](https://arxiv.org/pdf/2506.17356)  

**Abstract**: We explore the automatic generation of interactive, scenario-based lessons designed to train novice human tutors who teach middle school mathematics online. Employing prompt engineering through a Retrieval-Augmented Generation approach with GPT-4o, we developed a system capable of creating structured tutor training lessons. Our study generated lessons in English for three key topics: Encouraging Students' Independence, Encouraging Help-Seeking Behavior, and Turning on Cameras, using a task decomposition prompting strategy that breaks lesson generation into sub-tasks. The generated lessons were evaluated by two human evaluators, who provided both quantitative and qualitative evaluations using a comprehensive rubric informed by lesson design research. Results demonstrate that the task decomposition strategy led to higher-rated lessons compared to single-step generation. Human evaluators identified several strengths in the LLM-generated lessons, including well-structured content and time-saving potential, while also noting limitations such as generic feedback and a lack of clarity in some instructional sections. These findings underscore the potential of hybrid human-AI approaches for generating effective lessons in tutor training. 

**Abstract (ZH)**: 我们探索了通过检索增强生成方法利用GPT-4o进行自动生成交互式、基于场景的课程的设计，这些课程旨在培训在线辅导初中数学的初学者人机导师。我们采用任务分解提示策略，将课程生成分解为子任务，并通过一种结构化的方法创建了辅导培训课程。研究使用了两名人类评估者，他们根据课程设计研究的全面评判标准提供了定量和定性的评估。结果表明，任务分解策略生成的课程评价高于一步生成的方式。人类评估者认为LLM生成的课程具有结构良好和节省时间的优点，但也指出了通用反馈和部分教学环节不够清晰的局限性。这些发现突显了人机混合方法在生成有效辅导培训课程方面的潜力。 

---
# Differentiation-Based Extraction of Proprietary Data from Fine-Tuned LLMs 

**Title (ZH)**: 基于分化提取细调后的大规模语言模型中的专有数据 

**Authors**: Zongjie Li, Daoyuan Wu, Shuai Wang, Zhendong Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.17353)  

**Abstract**: The increasing demand for domain-specific and human-aligned Large Language Models (LLMs) has led to the widespread adoption of Supervised Fine-Tuning (SFT) techniques. SFT datasets often comprise valuable instruction-response pairs, making them highly valuable targets for potential extraction. This paper studies this critical research problem for the first time. We start by formally defining and formulating the problem, then explore various attack goals, types, and variants based on the unique properties of SFT data in real-world scenarios. Based on our analysis of extraction behaviors of direct extraction, we develop a novel extraction method specifically designed for SFT models, called Differentiated Data Extraction (DDE), which exploits the confidence levels of fine-tuned models and their behavioral differences from pre-trained base models. Through extensive experiments across multiple domains and scenarios, we demonstrate the feasibility of SFT data extraction using DDE. Our results show that DDE consistently outperforms existing extraction baselines in all attack settings. To counter this new attack, we propose a defense mechanism that mitigates DDE attacks with minimal impact on model performance. Overall, our research reveals hidden data leak risks in fine-tuned LLMs and provides insights for developing more secure models. 

**Abstract (ZH)**: 对领域特定和人类对齐的大语言模型进行监督微调的数据提取:一个新的研究问题及防御策略 

---
# Towards Safety Evaluations of Theory of Mind in Large Language Models 

**Title (ZH)**: 大型语言模型中理论心智安全评估的研究 

**Authors**: Tatsuhiro Aoshima, Mitsuaki Akiyama  

**Link**: [PDF](https://arxiv.org/pdf/2506.17352)  

**Abstract**: As the capabilities of large language models (LLMs) continue to advance, the importance of rigorous safety evaluation is becoming increasingly evident. Recent concerns within the realm of safety assessment have highlighted instances in which LLMs exhibit behaviors that appear to disable oversight mechanisms and respond in a deceptive manner. For example, there have been reports suggesting that, when confronted with information unfavorable to their own persistence during task execution, LLMs may act covertly and even provide false answers to questions intended to verify their this http URL evaluate the potential risk of such deceptive actions toward developers or users, it is essential to investigate whether these behaviors stem from covert, intentional processes within the model. In this study, we propose that it is necessary to measure the theory of mind capabilities of LLMs. We begin by reviewing existing research on theory of mind and identifying the perspectives and tasks relevant to its application in safety evaluation. Given that theory of mind has been predominantly studied within the context of developmental psychology, we analyze developmental trends across a series of open-weight LLMs. Our results indicate that while LLMs have improved in reading comprehension, their theory of mind capabilities have not shown comparable development. Finally, we present the current state of safety evaluation with respect to LLMs' theory of mind, and discuss remaining challenges for future work. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）能力的不断提升，严格的安全评估的重要性日益凸显。近期在安全评估领域的关切已经凸显了LLMs表现出看似规避监管机制并进行欺骗性回应的行为实例。例如，有报道称，在执行任务时面对对其持续性的不利信息时，LLMs可能会秘密行动甚至提供虚假答案以回答验证其持续性的质询。出于评估此类欺骗性行为对开发者或用户潜在风险的考虑，有必要调查这些行为是否源于模型内部的隐蔽故意过程。本研究提出，有必要衡量LLMs的理论思维能力。我们首先回顾了关于理论思维的研究，确定了适用于安全评估的应用视角和任务。鉴于理论思维主要是在发展心理学背景下研究的，我们分析了一系列开源权重LLMs的发展趋势。结果显示，尽管LLMs在阅读理解方面有所提高，但其理论思维能力尚未表现出相应的进步。最后，我们介绍了当前LLMs理论思维安全评估的状况，并讨论了未来研究面临的挑战。 

---
# Zero-Shot Cognitive Impairment Detection from Speech Using AudioLLM 

**Title (ZH)**: 基于AudioLLM的零样本认知功能障碍语音检测 

**Authors**: Mostafa Shahin, Beena Ahmed, Julien Epps  

**Link**: [PDF](https://arxiv.org/pdf/2506.17351)  

**Abstract**: Cognitive impairment (CI) is of growing public health concern, and early detection is vital for effective intervention. Speech has gained attention as a non-invasive and easily collectible biomarker for assessing cognitive decline. Traditional CI detection methods typically rely on supervised models trained on acoustic and linguistic features extracted from speech, which often require manual annotation and may not generalise well across datasets and languages. In this work, we propose the first zero-shot speech-based CI detection method using the Qwen2- Audio AudioLLM, a model capable of processing both audio and text inputs. By designing prompt-based instructions, we guide the model in classifying speech samples as indicative of normal cognition or cognitive impairment. We evaluate our approach on two datasets: one in English and another multilingual, spanning different cognitive assessment tasks. Our results show that the zero-shot AudioLLM approach achieves performance comparable to supervised methods and exhibits promising generalizability and consistency across languages, tasks, and datasets. 

**Abstract (ZH)**: 认知损害（CI）是日益引起公共卫生关注的问题，早期检测对于有效干预至关重要。语音已成为一种无创且易于收集的生物标志物，用于评估认知衰退。传统的CI检测方法通常依赖于在提取自语音的声学和语言特征上训练的监督模型，这通常需要手动注释，并且可能在不同数据集和语言间泛化效果不佳。在本工作中，我们提出了首个零样本语音基线CI检测方法，使用了具备处理音频和文本输入能力的Qwen2-Audio AudioLLM模型。通过设计指令式的提示，我们在分类语音样本时指导模型将其识别为正常认知或认知损害的迹象。我们在两个数据集上评估了我们的方法：一个为英语数据集，另一个为多语言数据集，涵盖了不同的认知评估任务。结果显示，零样本AudioLLM方法的性能与监督方法相当，并且在语言、任务和数据集方面展现了良好的泛化能力和一致性。 

---
# CUBA: Controlled Untargeted Backdoor Attack against Deep Neural Networks 

**Title (ZH)**: CUBA: 受控无目标后门攻击针对深度神经网络 

**Authors**: Yinghao Wu, Liyan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17350)  

**Abstract**: Backdoor attacks have emerged as a critical security threat against deep neural networks in recent years. The majority of existing backdoor attacks focus on targeted backdoor attacks, where trigger is strongly associated to specific malicious behavior. Various backdoor detection methods depend on this inherent property and shows effective results in identifying and mitigating such targeted attacks. However, a purely untargeted attack in backdoor scenarios is, in some sense, self-weakening, since the target nature is what makes backdoor attacks so powerful. In light of this, we introduce a novel Constrained Untargeted Backdoor Attack (CUBA), which combines the flexibility of untargeted attacks with the intentionality of targeted attacks. The compromised model, when presented with backdoor images, will classify them into random classes within a constrained range of target classes selected by the attacker. This combination of randomness and determinedness enables the proposed untargeted backdoor attack to natively circumvent existing backdoor defense methods. To implement the untargeted backdoor attack under controlled flexibility, we propose to apply logit normalization on cross-entropy loss with flipped one-hot labels. By constraining the logit during training, the compromised model will show a uniform distribution across selected target classes, resulting in controlled untargeted attack. Extensive experiments demonstrate the effectiveness of the proposed CUBA on different datasets. 

**Abstract (ZH)**: 受约束的未 targeted 黑盒攻击 (CUBA): 结合灵活性与意图性以绕过现有防御方法 

---
# Advanced Game-Theoretic Frameworks for Multi-Agent AI Challenges: A 2025 Outlook 

**Title (ZH)**: 2025年视角下的高级博弈论框架与多智能体AI挑战 

**Authors**: Pavel Malinovskiy  

**Link**: [PDF](https://arxiv.org/pdf/2506.17348)  

**Abstract**: This paper presents a substantially reworked examination of how advanced game-theoretic paradigms can serve as a foundation for the next-generation challenges in Artificial Intelligence (AI), forecasted to arrive in or around 2025. Our focus extends beyond traditional models by incorporating dynamic coalition formation, language-based utilities, sabotage risks, and partial observability. We provide a set of mathematical formalisms, simulations, and coding schemes that illustrate how multi-agent AI systems may adapt and negotiate in complex environments. Key elements include repeated games, Bayesian updates for adversarial detection, and moral framing within payoff structures. This work aims to equip AI researchers with robust theoretical tools for aligning strategic interaction in uncertain, partially adversarial contexts. 

**Abstract (ZH)**: 这篇论文提出了一种大幅修改后的研究，探讨了先进的博弈理论范式如何为基础人工智能（AI）领域的下一代挑战提供基础，预计这些挑战将在2025年或其前后出现。我们的研究超越了传统模型，纳入了动态联盟形成、基于语言的效用、破坏风险以及部分可观测性。我们提供了一套数学形式化、仿真和编码方案，展示了多智能体AI系统如何在复杂环境中适应和谈判。关键要素包括重复博弈、贝叶斯更新以检测对手，以及在收益结构中的道德框架。本研究旨在为AI研究人员提供强大的理论工具，以在不确定且部分对抗的环境下对齐战略互动。 

---
# Distinguishing Predictive and Generative AI in Regulation 

**Title (ZH)**: 区分预测型和生成型人工智能在监管中的应用 

**Authors**: Jennifer Wang, Andrew Selbst, Solon Barocas, Suresh Venkatasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2506.17347)  

**Abstract**: Over the past decade, policymakers have developed a set of regulatory tools to ensure AI development aligns with key societal goals. Many of these tools were initially developed in response to concerns with predictive AI and therefore encode certain assumptions about the nature of AI systems and the utility of certain regulatory approaches. With the advent of generative AI, however, some of these assumptions no longer hold, even as policymakers attempt to maintain a single regulatory target that covers both types of AI.
In this paper, we identify four distinct aspects of generative AI that call for meaningfully different policy responses. These are the generality and adaptability of generative AI that make it a poor regulatory target, the difficulty of designing effective evaluations, new legal concerns that change the ecosystem of stakeholders and sources of expertise, and the distributed structure of the generative AI value chain.
In light of these distinctions, policymakers will need to evaluate where the past decade of policy work remains relevant and where new policies, designed to address the unique risks posed by generative AI, are necessary. We outline three recommendations for policymakers to more effectively identify regulatory targets and leverage constraints across the broader ecosystem to govern generative AI. 

**Abstract (ZH)**: 过去十年，政策制定者开发了一套监管工具以确保人工智能的发展符合关键的社会目标。许多这些工具最初是针对预测型人工智能的担忧而设计的，因此包含了对人工智能系统性质和某些监管方法效益的假设。然而，伴随着生成型人工智能的到来，这些假设已不再适用，尽管政策制定者试图维持一个同时涵盖两类人工智能的单一监管目标。
在本文中，我们识别出生成型人工智能的四个独特方面，这需要有意义地不同的政策回应。这些方面包括生成型人工智能的一般性和适应性导致其成为不良的监管目标、设计有效评估的难度、新的法律问题改变了利益相关者和专业知识来源的生态系统、以及生成型人工智能价值链条的分布式结构。
鉴于这些区别，政策制定者需要评估过去十年政策工作的相关性和必要性更新政策，以应对生成型人工智能带来的独特风险。我们提出了三项针对政策制定者，旨在更有效地确定监管目标并利用更广泛生态系统中的制约因素来治理生成型人工智能的建议。 

---
# A Novel Multi-layer Task-centric and Data Quality Framework for Autonomous Driving 

**Title (ZH)**: 一种新型多层任务导向和数据质量框架 for 自动驾驶 

**Authors**: Yuhan Zhou, Haihua Chen, Kewei Sha  

**Link**: [PDF](https://arxiv.org/pdf/2506.17346)  

**Abstract**: The next-generation autonomous vehicles (AVs), embedded with frequent real-time decision-making, will rely heavily on a large volume of multisource and multimodal data. In real-world settings, the data quality (DQ) of different sources and modalities usually varies due to unexpected environmental factors or sensor issues. However, both researchers and practitioners in the AV field overwhelmingly concentrate on models/algorithms while undervaluing the DQ. To fulfill the needs of the next-generation AVs with guarantees of functionality, efficiency, and trustworthiness, this paper proposes a novel task-centric and data quality vase framework which consists of five layers: data layer, DQ layer, task layer, application layer, and goal layer. The proposed framework aims to map DQ with task requirements and performance goals. To illustrate, a case study investigating redundancy on the nuScenes dataset proves that partially removing redundancy on multisource image data could improve YOLOv8 object detection task performance. Analysis on multimodal data of image and LiDAR further presents existing redundancy DQ issues. This paper opens up a range of critical but unexplored challenges at the intersection of DQ, task orchestration, and performance-oriented system development in AVs. It is expected to guide the AV community toward building more adaptive, explainable, and resilient AVs that respond intelligently to dynamic environments and heterogeneous data streams. Code, data, and implementation details are publicly available at: this https URL. 

**Abstract (ZH)**: 下一代自主车辆（AV）嵌入了频繁的实时决策，将高度依赖大量多源和多模态数据。在实际应用中，由于不可预测的环境因素或传感器问题，不同来源和模态的数据质量（DQ）通常会有所不同。然而，自动驾驶领域的研究人员和从业人员普遍重视模型/算法，而忽视了数据质量。为了确保下一代AV的功能、效率和可靠性，本文提出了一种新的以任务为中心的数据质量框架，该框架由五层组成：数据层、数据质量层、任务层、应用层和目标层。该提出的框架旨在将数据质量与任务要求和性能目标映射起来。通过在nuScenes数据集上的案例研究，部分移除多源图像数据中的冗余可提高YOLOv8目标检测任务的性能。对图像和LiDAR多模态数据的分析进一步揭示了现有冗余数据质量的问题。本文在数据质量、任务协调和面向性能的系统开发在自主车辆中的交叉点上揭示了一系列重要的但未被探索的挑战。它有望指导自动驾驶社区构建更具适应性、可解释性和韧性的自主车辆，这些车辆能够智能地响应动态环境和异构数据流。相关代码、数据和实现细节可在以下链接获取：this https URL。 

---
# Adaptive Social Metaverse Streaming based on Federated Multi-Agent Deep Reinforcement Learning 

**Title (ZH)**: 基于联邦多代理深度强化学习的自适应社会元宇宙流媒体 

**Authors**: Zijian Long, Haopeng Wang, Haiwei Dong, Abdulmotaleb El Saddik  

**Link**: [PDF](https://arxiv.org/pdf/2506.17342)  

**Abstract**: The social metaverse is a growing digital ecosystem that blends virtual and physical worlds. It allows users to interact socially, work, shop, and enjoy entertainment. However, privacy remains a major challenge, as immersive interactions require continuous collection of biometric and behavioral data. At the same time, ensuring high-quality, low-latency streaming is difficult due to the demands of real-time interaction, immersive rendering, and bandwidth optimization. To address these issues, we propose ASMS (Adaptive Social Metaverse Streaming), a novel streaming system based on Federated Multi-Agent Proximal Policy Optimization (F-MAPPO). ASMS leverages F-MAPPO, which integrates federated learning (FL) and deep reinforcement learning (DRL) to dynamically adjust streaming bit rates while preserving user privacy. Experimental results show that ASMS improves user experience by at least 14% compared to existing streaming methods across various network conditions. Therefore, ASMS enhances the social metaverse experience by providing seamless and immersive streaming, even in dynamic and resource-constrained networks, while ensuring that sensitive user data remains on local devices. 

**Abstract (ZH)**: 社交元宇宙是一种融合虚拟和物理世界的 growing 数字生态系统，允许用户进行社交互动、工作、购物和享受娱乐。然而，隐私仍然是一个主要挑战，因为沉浸式互动需要持续收集生物特征和行为数据。同时，由于实时交互、沉浸式渲染和带宽优化的需求，确保高质量、低延迟的流媒体传输也极具挑战性。为了解决这些问题，我们提出了 ASMS（自适应社交元宇宙流媒体）系统，这是一种基于联邦多代理近端策略优化（F-MAPPO）的创新流媒体系统。ASMS 利用了 F-MAPPO，该系统结合了联邦学习（FL）和深度强化学习（DRL），以动态调整流媒体比特率同时保护用户隐私。实验结果表明，与现有流媒体方法相比，ASMS 在多种网络条件下将用户体验至少提升了 14%。因此，ASMS 通过提供无缝和沉浸式的流媒体体验，即使在动态和资源受限的网络环境下也能增强社交元宇宙体验，同时确保敏感用户数据保留在本地设备上。 

---
# PBFT-Backed Semantic Voting for Multi-Agent Memory Pruning 

**Title (ZH)**: 基于PBFT的语义投票多agent内存剪枝 

**Authors**: Duong Bach  

**Link**: [PDF](https://arxiv.org/pdf/2506.17338)  

**Abstract**: The proliferation of multi-agent systems (MAS) in complex, dynamic environments necessitates robust and efficient mechanisms for managing shared knowledge. A critical challenge is ensuring that distributed memories remain synchronized, relevant, and free from the accumulation of outdated or inconsequential data - a process analogous to biological forgetting. This paper introduces the Co-Forgetting Protocol, a novel, comprehensive framework designed to address this challenge by enabling synchronized memory pruning in MAS. The protocol integrates three key components: (1) context-aware semantic voting, where agents utilize a lightweight DistilBERT model to assess the relevance of memory items based on their content and the current operational context; (2) multi-scale temporal decay functions, which assign diminishing importance to memories based on their age and access frequency across different time horizons; and (3) a Practical Byzantine Fault Tolerance (PBFT)-based consensus mechanism, ensuring that decisions to retain or discard memory items are agreed upon by a qualified and fault-tolerant majority of agents, even in the presence of up to f Byzantine (malicious or faulty) agents in a system of N greater than or equal to 3f+1 agents. The protocol leverages gRPC for efficient inter-agent communication and Pinecone for scalable vector embedding storage and similarity search, with SQLite managing metadata. Experimental evaluations in a simulated MAS environment with four agents demonstrate the protocol's efficacy, achieving a 52% reduction in memory footprint over 500 epochs, 88% voting accuracy in forgetting decisions against human-annotated benchmarks, a 92% PBFT consensus success rate under simulated Byzantine conditions, and an 82% cache hit rate for memory access. 

**Abstract (ZH)**: 多代理系统中分布式记忆同步与精简的Co-Forgetting协议 

---
# Can Common VLMs Rival Medical VLMs? Evaluation and Strategic Insights 

**Title (ZH)**: 通用大模型能否挑战医疗大模型？评估与战略洞察 

**Authors**: Yuan Zhong, Ruinan Jin, Xiaoxiao Li, Qi Dou  

**Link**: [PDF](https://arxiv.org/pdf/2506.17337)  

**Abstract**: Medical vision-language models (VLMs) leverage large-scale pretraining for diverse imaging tasks but require substantial computational and data resources. Meanwhile, common or general-purpose VLMs (e.g., CLIP, LLaVA), though not trained for medical use, show promise with fine-tuning. This raises a key question: Can efficient fine-tuned common VLMs rival generalist medical VLMs for solving specific medical imaging tasks? This study systematically evaluates common and medical VLMs across disease diagnosis and visual question answering (VQA). Using CLIP-based and LLaVA-based models, we examine (1) off-the-shelf performance gaps in in-domain (ID) settings, (2) whether fine-tuning bridges these gaps, and (3) generalization to out-of-domain (OOD) tasks on unseen medical modalities. While medical-specific pretraining provides advantages in ID settings, common VLMs match or surpass medical-specific models after lightweight fine-tuning, with LoRA-based adaptation proving highly effective among different tasks. In OOD tasks, common VLMs demonstrate strong adaptability in some tasks, challenging the assumption that medical-specific pre-training is essential. These findings suggest that leveraging common VLMs with fine-tuning offers a scalable and cost-effective alternative to developing large-scale medical VLMs, providing crucial insights for future research in the medical imaging field. 

**Abstract (ZH)**: 医学视觉-语言模型在多种成像任务中利用大规模预训练但需要大量计算和数据资源。尽管通用或通用型视觉-语言模型（如CLIP、LLaVA）未专门训练用于医疗用途，但在微调后显示出潜力。本研究系统地评估了通用和医学视觉-语言模型在疾病诊断和视觉问答任务中的表现。利用CLIP和LLaVA模型，我们研究了（1）领域内（ID）设置下的即用型性能差距，（2）微调是否能弥合这些差距，以及（3）在未见医学模态的领域外（OOD）任务中的泛化能力。虽然针对医学的预训练在ID设置中具有优势，但在轻量级微调后，通用视觉-语言模型能够匹配或超越针对医学的模型，基于LoRA的适应尤其在不同任务中证明非常有效。在OOD任务中，通用视觉-语言模型在某些任务中表现出强大的适应性，挑战了医学专用预训练必不可少的假设。这些发现表明，利用通用视觉-语言模型并进行微调提供了一种可扩展且成本效益高的替代方案，用于开发大规模医学视觉-语言模型，并为未来医学成像领域的研究提供了关键见解。 

---
# LMR-BENCH: Evaluating LLM Agent's Ability on Reproducing Language Modeling Research 

**Title (ZH)**: LMR-BENCH: 评估大型语言模型代理在重现语言模型研究能力方面的表现 

**Authors**: Shuo Yan, Ruochen Li, Ziming Luo, Zimu Wang, Daoyang Li, Liqiang Jing, Kaiyu He, Peilin Wu, George Michalopoulos, Yue Zhang, Ziyang Zhang, Mian Zhang, Zhiyu Chen, Xinya Du  

**Link**: [PDF](https://arxiv.org/pdf/2506.17335)  

**Abstract**: Large language model (LLM) agents have demonstrated remarkable potential in advancing scientific discovery. However, their capability in the fundamental yet crucial task of reproducing code from research papers, especially in the NLP domain, remains underexplored. This task includes unique complex reasoning challenges in the intellectual synthesis of abstract concepts and the comprehension of code repositories with interdependent files. Motivated by this gap, we present LMR-BENCH, a benchmark designed to systematically evaluate the capability of LLM agents on code reproduction from Language Modeling Research. It consists of 28 code reproduction tasks derived from 23 research papers published in top-tier NLP venues over the past five years, spanning nine fundamental categories. Models are provided with a research paper, a code repository containing one or more masked functions, and instructions for implementing these functions. We conduct extensive experiments in standard prompting and LLM agent settings with state-of-the-art LLMs, evaluating the accuracy of unit tests and performing LLM-based evaluation of code correctness. Experimental results reveal that even the most advanced models still exhibit persistent limitations in scientific reasoning and code synthesis, highlighting critical gaps in LLM agents' ability to autonomously reproduce scientific research 

**Abstract (ZH)**: 大型语言模型代理在语言建模研究中的代码重现基准（LMR-BENCH）：系统评估其在科学发现中的代码重现能力 

---
# P2MFDS: A Privacy-Preserving Multimodal Fall Detection System for Elderly People in Bathroom Environments 

**Title (ZH)**: P2MFDS：一种保护隐私的多模态浴室环境中老年人跌倒检测系统 

**Authors**: Haitian Wang, Yiren Wang, Xinyu Wang, Yumeng Miao, Yuliang Zhang, Yu Zhang, Atif Mansoor  

**Link**: [PDF](https://arxiv.org/pdf/2506.17332)  

**Abstract**: By 2050, people aged 65 and over are projected to make up 16 percent of the global population. As aging is closely associated with increased fall risk, particularly in wet and confined environments such as bathrooms where over 80 percent of falls occur. Although recent research has increasingly focused on non-intrusive, privacy-preserving approaches that do not rely on wearable devices or video-based monitoring, these efforts have not fully overcome the limitations of existing unimodal systems (e.g., WiFi-, infrared-, or mmWave-based), which are prone to reduced accuracy in complex environments. These limitations stem from fundamental constraints in unimodal sensing, including system bias and environmental interference, such as multipath fading in WiFi-based systems and drastic temperature changes in infrared-based methods. To address these challenges, we propose a Privacy-Preserving Multimodal Fall Detection System for Elderly People in Bathroom Environments. First, we develop a sensor evaluation framework to select and fuse millimeter-wave radar with 3D vibration sensing, and use it to construct and preprocess a large-scale, privacy-preserving multimodal dataset in real bathroom settings, which will be released upon publication. Second, we introduce P2MFDS, a dual-stream network combining a CNN-BiLSTM-Attention branch for radar motion dynamics with a multi-scale CNN-SEBlock-Self-Attention branch for vibration impact detection. By uniting macro- and micro-scale features, P2MFDS delivers significant gains in accuracy and recall over state-of-the-art approaches. Code and pretrained models will be made available at: this https URL. 

**Abstract (ZH)**: 到2050年，65岁及以上人口预计将占全球人口的16%。由于老化与增加的跌倒风险密切相关，特别是在如浴室这样的潮湿和受限环境中，超过80%的跌倒事件在此类环境中发生。尽管近期研究 increasingly关注非侵入性、保护隐私的方法，这些方法不依赖于可穿戴设备或基于视频的监控，但这些努力仍未完全克服现有单一模态系统的局限性（例如，基于WiFi、红外或毫米波的方法在复杂环境中的准确率较低）。这些局限性源于单一模态传感的基本限制，包括系统偏差和环境干扰，如WiFi系统中的多径衰落和红外方法中的剧烈温度变化。为解决这些挑战，我们提出了一种隐私保护的多模态跌倒检测系统，专门适用于浴室环境中的老年人。首先，我们开发了一个传感器评估框架，选择并融合毫米波雷达与3D振动传感，并在实际的浴室环境中构建和预处理了一个大规模的隐私保护多模态数据集，该数据集将在发表时公开。其次，我们引入了P2MFDS，这是一种双流网络，结合了基于CNN-BiLSTM-Attention支路的雷达运动动力学检测与基于多尺度CNN-SEBlock-Self-Attention支路的振动冲击检测。通过结合宏观和微观特征，P2MFDS在准确性和召回率方面显著优于当前最先进的方法。代码和预训练模型将在此处发布：this https URL。 

---
# On the Performance of Cyber-Biomedical Features for Intrusion Detection in Healthcare 5.0 

**Title (ZH)**: 面向 healthcare 5.0 的网络生物医学特征入侵检测性能研究 

**Authors**: Pedro H. Lui, Lucas P. Siqueira, Juliano F. Kazienko, Vagner E. Quincozes, Silvio E. Quincozes, Daniel Welfer  

**Link**: [PDF](https://arxiv.org/pdf/2506.17329)  

**Abstract**: Healthcare 5.0 integrates Artificial Intelligence (AI), the Internet of Things (IoT), real-time monitoring, and human-centered design toward personalized medicine and predictive diagnostics. However, the increasing reliance on interconnected medical technologies exposes them to cyber threats. Meanwhile, current AI-driven cybersecurity models often neglect biomedical data, limiting their effectiveness and interpretability. This study addresses this gap by applying eXplainable AI (XAI) to a Healthcare 5.0 dataset that integrates network traffic and biomedical sensor data. Classification outputs indicate that XGBoost achieved 99% F1-score for benign and data alteration, and 81% for spoofing. Explainability findings reveal that network data play a dominant role in intrusion detection whereas biomedical features contributed to spoofing detection, with temperature reaching a Shapley values magnitude of 0.37. 

**Abstract (ZH)**: Healthcare 5.0融合了人工智能、物联网、实时监控和以人为本的设计，旨在实现个性化医疗和预测诊断。然而，对互联医疗技术的日益依赖使其面临网络安全威胁。当前的AI驱动的网络安全模型往往忽视生物医学数据，限制了其有效性和可解释性。本研究通过将可解释人工智能（XAI）应用于整合网络流量和生物医学传感器数据的Healthcare 5.0数据集，填补了这一缺口。分类输出表明，XGBoost在良性行为和数据篡改检测上达到了99%的F1分数，在冒充检测上达到了81%。可解释性研究表明，网络数据在网络入侵检测中发挥主导作用，而生物医学特征对冒充检测做出了贡献，其中体温的Shapley值为0.37。 

---
# RadarSeq: A Temporal Vision Framework for User Churn Prediction via Radar Chart Sequences 

**Title (ZH)**: RadarSeq：基于雷达图序列的用户流失预测时间视觉框架 

**Authors**: Sina Najafi, M. Hadi Sepanj, Fahimeh Jafari  

**Link**: [PDF](https://arxiv.org/pdf/2506.17325)  

**Abstract**: Predicting user churn in non-subscription gig platforms, where disengagement is implicit, poses unique challenges due to the absence of explicit labels and the dynamic nature of user behavior. Existing methods often rely on aggregated snapshots or static visual representations, which obscure temporal cues critical for early detection. In this work, we propose a temporally-aware computer vision framework that models user behavioral patterns as a sequence of radar chart images, each encoding day-level behavioral features. By integrating a pretrained CNN encoder with a bidirectional LSTM, our architecture captures both spatial and temporal patterns underlying churn behavior. Extensive experiments on a large real-world dataset demonstrate that our method outperforms classical models and ViT-based radar chart baselines, yielding gains of 17.7 in F1 score, 29.4 in precision, and 16.1 in AUC, along with improved interpretability. The framework's modular design, explainability tools, and efficient deployment characteristics make it suitable for large-scale churn modeling in dynamic gig-economy platforms. 

**Abstract (ZH)**: 基于雷达图的时空aware计算机视觉框架在非订阅制零工平台用户流失预测中的应用 

---
# I Know Which LLM Wrote Your Code Last Summer: LLM generated Code Stylometry for Authorship Attribution 

**Title (ZH)**: 我知道你去年夏天的代码是由哪个LLM生成的：基于代码风格的作者归因方法 

**Authors**: Tamas Bisztray, Bilel Cherif, Richard A. Dubniczky, Nils Gruschka, Bertalan Borsos, Mohamed Amine Ferrag, Attila Kovacs, Vasileios Mavroeidis, Norbert Tihanyi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17323)  

**Abstract**: Detecting AI-generated code, deepfakes, and other synthetic content is an emerging research challenge. As code generated by Large Language Models (LLMs) becomes more common, identifying the specific model behind each sample is increasingly important. This paper presents the first systematic study of LLM authorship attribution for C programs. We released CodeT5-Authorship, a novel model that uses only the encoder layers from the original CodeT5 encoder-decoder architecture, discarding the decoder to focus on classification. Our model's encoder output (first token) is passed through a two-layer classification head with GELU activation and dropout, producing a probability distribution over possible authors. To evaluate our approach, we introduce LLM-AuthorBench, a benchmark of 32,000 compilable C programs generated by eight state-of-the-art LLMs across diverse tasks. We compare our model to seven traditional ML classifiers and eight fine-tuned transformer models, including BERT, RoBERTa, CodeBERT, ModernBERT, DistilBERT, DeBERTa-V3, Longformer, and LoRA-fine-tuned Qwen2-1.5B. In binary classification, our model achieves 97.56% accuracy in distinguishing C programs generated by closely related models such as GPT-4.1 and GPT-4o, and 95.40% accuracy for multi-class attribution among five leading LLMs (Gemini 2.5 Flash, Claude 3.5 Haiku, GPT-4.1, Llama 3.3, and DeepSeek-V3). To support open science, we release the CodeT5-Authorship architecture, the LLM-AuthorBench benchmark, and all relevant Google Colab scripts on GitHub: this https URL. 

**Abstract (ZH)**: 检测由人工智能生成的代码、深度伪造和其他合成内容是新兴的研究挑战。随着大型语言模型（LLMs）生成的代码变得更加普遍，识别每份样本背后的特定模型变得越来越重要。本文呈现了首个对C程序进行LLM作者归属的系统性研究。我们发布了CodeT5-Authorship，这是一种新颖的模型，仅使用CodeT5编码器-解码器架构的编码器层，舍弃解码器以专注于分类。我们的模型的编码器输出（第一个token）通过带有GELU激活和dropout的两层分类头，生成可能作者的概率分布。为了评估我们的方法，我们引入了LLM-AuthorBench基准，该基准包含32,000个由八种最先进的LLM生成的可编译C程序，涵盖了多种任务。我们将我们的模型与七种传统机器学习分类器和八种微调的变换器模型进行了比较，包括BERT、RoBERTa、CodeBERT、ModernBERT、DistilBERT、DeBERTa-V3、Longformer和基于LoRA微调的Qwen2-1.5B。在二分类中，我们的模型在区分紧密相关模型（如GPT-4.1和GPT-4o）生成的C程序时实现了97.56%的准确率，并在五种领先LLM（Gemini 2.5 Flash、Claude 3.5 Haiku、GPT-4.1、Llama 3.3和DeepSeek-V3）的多类归属中实现了95.40%的准确率。为了支持开放科学，我们将在GitHub上发布CodeT5-Authorship架构、LLM-AuthorBench基准以及所有相关的Google Colab脚本：this https URL。 

---
# Context manipulation attacks : Web agents are susceptible to corrupted memory 

**Title (ZH)**: 上下文操作攻击：网页代理易受污染内存的影响 

**Authors**: Atharv Singh Patlan, Ashwin Hebbar, Pramod Viswanath, Prateek Mittal  

**Link**: [PDF](https://arxiv.org/pdf/2506.17318)  

**Abstract**: Autonomous web navigation agents, which translate natural language instructions into sequences of browser actions, are increasingly deployed for complex tasks across e-commerce, information retrieval, and content discovery. Due to the stateless nature of large language models (LLMs), these agents rely heavily on external memory systems to maintain context across interactions. Unlike centralized systems where context is securely stored server-side, agent memory is often managed client-side or by third-party applications, creating significant security vulnerabilities. This was recently exploited to attack production systems.
We introduce and formalize "plan injection," a novel context manipulation attack that corrupts these agents' internal task representations by targeting this vulnerable context. Through systematic evaluation of two popular web agents, Browser-use and Agent-E, we show that plan injections bypass robust prompt injection defenses, achieving up to 3x higher attack success rates than comparable prompt-based attacks. Furthermore, "context-chained injections," which craft logical bridges between legitimate user goals and attacker objectives, lead to a 17.7% increase in success rate for privacy exfiltration tasks. Our findings highlight that secure memory handling must be a first-class concern in agentic systems. 

**Abstract (ZH)**: 自主网络导航代理将自然语言指令转化为浏览器操作序列，在电子商务、信息检索和内容发现等领域中越来越多地被用于复杂任务。由于大型语言模型（LLMs）缺乏状态维持能力，这些代理高度依赖外部内存系统来维持交互过程中的上下文。与将上下文安全地存储在服务器端的集中式系统不同，代理的内存往往在客户端或第三方应用程序中管理，从而创造出重大的安全漏洞。这最近被利用来攻击生产系统。

我们提出了并形式化了“计划注入”这一新颖的上下文操控攻击，通过针对这一脆弱的上下文，篡改这些代理内部的任务表示。通过系统性地评估两个流行的网络代理Browser-use和Agent-E，我们显示计划注入绕过了稳健的提示注入防御，其攻击成功率比同类提示基攻击高3倍。此外，通过在合法用户目标和攻击者目标之间构建逻辑桥梁的“上下文链接注入”，前所未有地将隐私泄露任务的成功率提高了17.7%。我们的研究结果表明，安全的内存处理必须在代理系统中被视为头等大事。 

---
# Heterogeneous Temporal Hypergraph Neural Network 

**Title (ZH)**: 异构时序超图神经网络 

**Authors**: Huan Liu, Pengfei Jiao, Mengzhou Gao, Chaochao Chen, Di Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.17312)  

**Abstract**: Graph representation learning (GRL) has emerged as an effective technique for modeling graph-structured data. When modeling heterogeneity and dynamics in real-world complex networks, GRL methods designed for complex heterogeneous temporal graphs (HTGs) have been proposed and have achieved successful applications in various fields. However, most existing GRL methods mainly focus on preserving the low-order topology information while ignoring higher-order group interaction relationships, which are more consistent with real-world networks. In addition, most existing hypergraph methods can only model static homogeneous graphs, limiting their ability to model high-order interactions in HTGs. Therefore, to simultaneously enable the GRL model to capture high-order interaction relationships in HTGs, we first propose a formal definition of heterogeneous temporal hypergraphs and $P$-uniform heterogeneous hyperedge construction algorithm that does not rely on additional information. Then, a novel Heterogeneous Temporal HyperGraph Neural network (HTHGN), is proposed to fully capture higher-order interactions in HTGs. HTHGN contains a hierarchical attention mechanism module that simultaneously performs temporal message-passing between heterogeneous nodes and hyperedges to capture rich semantics in a wider receptive field brought by hyperedges. Furthermore, HTHGN performs contrastive learning by maximizing the consistency between low-order correlated heterogeneous node pairs on HTG to avoid the low-order structural ambiguity issue. Detailed experimental results on three real-world HTG datasets verify the effectiveness of the proposed HTHGN for modeling high-order interactions in HTGs and demonstrate significant performance improvements. 

**Abstract (ZH)**: 基于异构时变超图的图表示学习方法：捕获高阶交互关系的新范式 

---
# AlgoSelect: Universal Algorithm Selection via the Comb Operator 

**Title (ZH)**: AlgoSelect: 统一的算法选择方法通过Combing运算符 

**Authors**: Jasper Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.17304)  

**Abstract**: We introduce AlgoSelect, a principled framework for learning optimal algorithm selection from data, centered around the novel Comb Operator. Given a set of algorithms and a feature representation of problems, AlgoSelect learns to interpolate between diverse computational approaches. For pairs of algorithms, a simple sigmoid-gated selector, an instance of the Comb Operator, facilitates this interpolation. We extend this to an N-Path Comb for multiple algorithms. We prove that this framework is universal (can approximate any algorithm selector), information-theoretically optimal in its learnability (thresholds for selection converge almost surely, demonstrated via Borel-Cantelli arguments), computationally efficient, and robust. Key theoretical contributions include: (1) a universal approximation theorem demonstrating that Comb-based selectors can achieve arbitrary accuracy; (2) information-theoretic learnability for selection thresholds; (3) formalization of the Comb Operator within linear operator theory, detailing its boundedness and spectral properties; (4) an N-Path Comb generalization for multi-algorithm selection; and (5) a practical learning framework for the adaptive seeding functions that guide the Comb Operator. Empirical validation on a comprehensive 20$\times$20 problem-algorithm study demonstrates near-perfect selection (99.9\%+ accuracy) with remarkably few samples and rapid convergence, revealing that $H(\text{Algorithm}|\text{Problem}) \approx 0$ in structured domains. AlgoSelect provides a theoretically grounded, practically deployable solution to automated algorithm selection with provable optimality and learnability guarantees, with significant implications for AI and adaptive systems. 

**Abstract (ZH)**: 我们介绍了一种基于新颖Comb操作器的原理性框架AlgoSelect，用于从数据中学习最优算法选择。该框架通过插值多种计算方法，针对算法集和问题特征表示，学习如何进行选择。对于成对的算法，Comb操作器的一种简单Sigmoid门选路器促进了这种插值。我们将这一方法扩展到支持多算法的N-Path Comb。我们证明了该框架是通用的（能够逼近任何算法选择器）、信息论上的可学习性最优（选择阈值几乎必然收敛，通过博雷尔-坎特利论证证明）、计算效率高且稳健。关键的理论贡献包括：（1）一个通用逼近定理，证明Comb基于的选择器可以达到任意精度；（2）选择阈值的信息论可学习性；（3）在线性算子理论中形式化Comb操作器，详细描述其有界性和频谱性质；（4）支持多算法选择的N-Path Comb推广；以及（5）指导Comb操作器的自适应初始化函数的实用学习框架。在全面的20$\times$20问题-算法研究中进行的经验验证表明，在惊人少量的样本和快速收敛下，选择精度接近完美（99.9%以上），揭示出在结构域中$H(\text{Algorithm}|\text{Problem}) \approx 0$。AlgoSelect提供了一种理论依据且实用部署的自动化算法选择解决方案，带有可证明的最优性和可学习性保证，对AI和自适应系统具有重大影响。 

---
# LLM Jailbreak Oracle 

**Title (ZH)**: LLM Jailbreak Oracle 

**Authors**: Shuyi Lin, Anshuman Suri, Alina Oprea, Cheng Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17299)  

**Abstract**: As large language models (LLMs) become increasingly deployed in safety-critical applications, the lack of systematic methods to assess their vulnerability to jailbreak attacks presents a critical security gap. We introduce the jailbreak oracle problem: given a model, prompt, and decoding strategy, determine whether a jailbreak response can be generated with likelihood exceeding a specified threshold. This formalization enables a principled study of jailbreak vulnerabilities. Answering the jailbreak oracle problem poses significant computational challenges -- the search space grows exponentially with the length of the response tokens. We present Boa, the first efficient algorithm for solving the jailbreak oracle problem. Boa employs a three-phase search strategy: (1) constructing block lists to identify refusal patterns, (2) breadth-first sampling to identify easily accessible jailbreaks, and (3) depth-first priority search guided by fine-grained safety scores to systematically explore promising low-probability paths. Boa enables rigorous security assessments including systematic defense evaluation, standardized comparison of red team attacks, and model certification under extreme adversarial conditions. 

**Abstract (ZH)**: 大语言模型（LLMs）在安全关键应用中的广泛应用引发了对其对抗脱管攻击脆弱性的系统评估方法的需求缺口。我们提出了脱管攻击Oracle问题：给定一个模型、提示和解码策略，确定是否存在生成超出指定阈值概率的脱管响应的可能性。这一形式化定义使得系统的脱管攻击脆弱性研究成为可能。解决脱管攻击Oracle问题面临着巨大的计算挑战——随着响应令牌长度的增加，搜索空间呈指数增长。我们提出了Boa，首个解决脱管攻击Oracle问题的高效算法。Boa采用三阶段搜索策略：（1）构建块列表以识别拒绝模式，（2）广度优先采样以识别易于访问的脱管响应，（3）基于细粒度安全评分的深度优先优先级搜索以系统地探索有希望的低概率路径。Boa使得严格的 security 评估成为可能，包括系统性防御评估、红队攻击的标准性比较以及在极端对抗条件下的模型认证。 

---
# Mercury: Ultra-Fast Language Models Based on Diffusion 

**Title (ZH)**: 汞：基于扩散的超快速语言模型 

**Authors**: Inception Labs, Samar Khanna, Siddhant Kharbanda, Shufan Li, Harshit Varma, Eric Wang, Sawyer Birnbaum, Ziyang Luo, Yanis Miraoui, Akash Palrecha, Stefano Ermon, Aditya Grover, Volodymyr Kuleshov  

**Link**: [PDF](https://arxiv.org/pdf/2506.17298)  

**Abstract**: We present Mercury, a new generation of commercial-scale large language models (LLMs) based on diffusion. These models are parameterized via the Transformer architecture and trained to predict multiple tokens in parallel. In this report, we detail Mercury Coder, our first set of diffusion LLMs designed for coding applications. Currently, Mercury Coder comes in two sizes: Mini and Small. These models set a new state-of-the-art on the speed-quality frontier. Based on independent evaluations conducted by Artificial Analysis, Mercury Coder Mini and Mercury Coder Small achieve state-of-the-art throughputs of 1109 tokens/sec and 737 tokens/sec, respectively, on NVIDIA H100 GPUs and outperform speed-optimized frontier models by up to 10x on average while maintaining comparable quality. We discuss additional results on a variety of code benchmarks spanning multiple languages and use-cases as well as real-world validation by developers on Copilot Arena, where the model currently ranks second on quality and is the fastest model overall. We also release a public API at this https URL and free playground at this https URL 

**Abstract (ZH)**: Mercury：基于扩散的新型商用大型语言模型及其编码应用Подробное описание Mercury Coder，我们的首款针对编码应用的扩散型大型语言模型。目前，Mercury Coder 提供 Mini 和 Small 两种规模。这些模型在速度-质量前沿上达到了新的标准。根据 Artificial Analysis 进行的独立评测，Mercury Coder Mini 和 Mercury Coder Small 在 NVIDIA H100 GPU 上分别实现了每秒 1109 个令牌和 737 个令牌的吞吐量，比速度优化的前沿模型平均快 10 倍，同时保持了相当的质量。我们还讨论了在多种编程语言和应用场景下的代码基准测试结果，以及开发人员在 Copilot Arena 中的实战验证，Mercury Coder 当前在质量排名第二，在所有模型中最快。我们也在该网址发布了一个公共 API 和一个免费的 playground。 

---
# SafeRL-Lite: A Lightweight, Explainable, and Constrained Reinforcement Learning Library 

**Title (ZH)**: SafeRL-Lite: 一种轻量级、可解释且受约束的强化学习库 

**Authors**: Satyam Mishra, Phung Thao Vi, Shivam Mishra, Vishwanath Bijalwan, Vijay Bhaskar Semwal, Abdul Manan Khan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17297)  

**Abstract**: We introduce SafeRL-Lite, an open-source Python library for building reinforcement learning (RL) agents that are both constrained and explainable. Existing RL toolkits often lack native mechanisms for enforcing hard safety constraints or producing human-interpretable rationales for decisions. SafeRL-Lite provides modular wrappers around standard Gym environments and deep Q-learning agents to enable: (i) safety-aware training via constraint enforcement, and (ii) real-time post-hoc explanation via SHAP values and saliency maps. The library is lightweight, extensible, and installable via pip, and includes built-in metrics for constraint violations. We demonstrate its effectiveness on constrained variants of CartPole and provide visualizations that reveal both policy logic and safety adherence. The full codebase is available at: this https URL. 

**Abstract (ZH)**: 我们介绍SafeRL-Lite，一个开源Python库，用于构建既受约束又可解释的强化学习（RL）代理。现有的RL工具包通常缺乏强制执行严格安全约束或生成人类可解释决策理由的内置机制。SafeRL-Lite通过模块化包装标准Gym环境和深度Q学习代理，实现了：(i) 通过约束强制执行进行安全意识训练，以及(ii) 通过SHAP值和可解释性地图进行实时事后解释。该库轻量级、可扩展，并可通过pip安装，内置了约束违反的度量标准。我们展示了其在受约束的CartPole变体上的有效性，并提供了可视化，揭示了策略逻辑和安全性遵循情况。完整代码库可在以下链接获取：this https URL。 

---
# Semantic uncertainty in advanced decoding methods for LLM generation 

**Title (ZH)**: 高级解码方法中LLM生成的语义不确定性 

**Authors**: Darius Foodeei, Simin Fan, Martin Jaggi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17296)  

**Abstract**: This study investigates semantic uncertainty in large language model (LLM) outputs across different decoding methods, focusing on emerging techniques like speculative sampling and chain-of-thought (CoT) decoding. Through experiments on question answering, summarization, and code generation tasks, we analyze how different decoding strategies affect both the diversity and reliability of model outputs. Our findings reveal that while CoT decoding demonstrates higher semantic diversity, it maintains lower predictive entropy, suggesting that structured exploration can lead to more confident and accurate outputs. This is evidenced by a 48.8% improvement in code generation Pass@2 rates, despite lower alignment with reference solutions. For summarization tasks, speculative sampling proved particularly effective, achieving superior ROUGE scores while maintaining moderate semantic diversity. Our results challenge conventional assumptions about trade-offs between diversity and accuracy in language model outputs, demonstrating that properly structured decoding methods can increase semantic exploration while maintaining or improving output quality. These findings have significant implications for deploying language models in practical applications where both reliability and diverse solution generation are crucial. 

**Abstract (ZH)**: 本研究探讨了不同解码方法下大规模语言模型（LLM）输出中的语义不确定性，重点研究了投机采样和链式思考（CoT）解码等新兴技术。通过在问答、总结和代码生成任务上的实验，我们分析了不同解码策略如何影响模型输出的多样性和可靠性。研究发现，虽然CoT解码显示出更高的语义多样性，但其预测不确定性较低，表明结构化的探索可以导致更加自信和准确的输出。这一结论在代码生成任务中得到了验证，尽管与参考答案的匹配度较低，但代码生成的Pass@2率提高了48.8%。对于总结任务，投机采样特别有效，实现了优越的ROUGE评分，同时保持了适度的语义多样性。我们的结果挑战了语言模型输出中多样性和准确性之间权衡的传统假设，表明适当的结构化解码方法可以在增加语义探索的同时维持或提高输出质量。这些发现对在需要可靠性和多样化解决方案的应用中部署语言模型具有重要意义。 

---
# AI-Generated Game Commentary: A Survey and a Datasheet Repository 

**Title (ZH)**: AI生成的游戏评论：综述与数据集仓库 

**Authors**: Qirui Zheng, Xingbo Wang, Keyuan Cheng, Yunlong Lu, Wenxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.17294)  

**Abstract**: AI-Generated Game Commentary (AIGGC) has gained increasing attention due to its market potential and inherent technical challenges. As a comprehensive multimodal Natural Language Processing (NLP) task, AIGGC imposes substantial demands on language models, including factual accuracy, logical reasoning, expressive text generation, generation speed, and context management. In this paper, we introduce a general framework for AIGGC and present a comprehensive survey of 45 existing game commentary dataset and methods according to key challenges they aim to address in this domain. We further classify and compare various evaluation metrics commonly used in this domain. To support future research and benchmarking, we also provide a structured datasheet summarizing the essential attributes of these datasets in appendix, which is meanwhile publicly available in an open repository. 

**Abstract (ZH)**: AI生成的游戏评论（AIGGC）因其市场潜力和固有的技术挑战而日益受到关注。作为一种综合性的多模态自然语言处理任务，AIGGC对语言模型提出了包括事实准确性、逻辑推理、表达性文本生成、生成速度和上下文管理等方面的重大需求。本文介绍了一种AIGGC的通用框架，并根据它们在这领域中试图解决的关键挑战，综述了45个现有的游戏评论数据集和方法。此外，我们还对这个领域中常用的各种评估指标进行了分类和比较。为了支持未来的研究和基准测试，我们还在附录中提供了一份结构化的数据表，总结了这些数据集的关键属性，并且该数据表同时在开放仓库中公开可用。 

---
# Theoretically Unmasking Inference Attacks Against LDP-Protected Clients in Federated Vision Models 

**Title (ZH)**: 理论上揭示联邦视觉模型中LDP保护客户端的推理攻击 

**Authors**: Quan Nguyen, Minh N. Vu, Truc Nguyen, My T. Thai  

**Link**: [PDF](https://arxiv.org/pdf/2506.17292)  

**Abstract**: Federated Learning enables collaborative learning among clients via a coordinating server while avoiding direct data sharing, offering a perceived solution to preserve privacy. However, recent studies on Membership Inference Attacks (MIAs) have challenged this notion, showing high success rates against unprotected training data. While local differential privacy (LDP) is widely regarded as a gold standard for privacy protection in data analysis, most studies on MIAs either neglect LDP or fail to provide theoretical guarantees for attack success rates against LDP-protected data. To address this gap, we derive theoretical lower bounds for the success rates of low-polynomial time MIAs that exploit vulnerabilities in fully connected or self-attention layers. We establish that even when data are protected by LDP, privacy risks persist, depending on the privacy budget. Practical evaluations on federated vision models confirm considerable privacy risks, revealing that the noise required to mitigate these attacks significantly degrades models' utility. 

**Abstract (ZH)**: 联邦学习通过协调服务器使客户端能够进行协作学习，同时避免直接数据共享，提供了一种保护隐私的潜在解决方案。然而，近期关于成员推理攻击（MIAs）的研究对此观点提出了挑战，显示出对未受保护的训练数据的成功率很高。尽管局部差分隐私（LDP）被视为数据分析中隐私保护的金标准，但大多数MIAs的研究要么忽视LDP，要么未能为保护在LDP下的数据的攻击成功率提供理论保证。为解决这一问题，我们推导出利用全连接或自我注意力层漏洞的低多项式时间MIAs的成功率的理论下界。我们确立了即使数据受到LDP保护，隐私风险仍然存在，这取决于隐私预算。实证评估发现，在联邦视觉模型中存在显著的隐私风险，表明缓解这些攻击所需的噪声显著降低了模型的实用性。 

---
# SlimRAG: Retrieval without Graphs via Entity-Aware Context Selection 

**Title (ZH)**: SlimRAG：无需图的实体意识上下文选择检索 

**Authors**: Jiale Zhang, Jiaxiang Chen, Zhucong Li, Jie Ding, Kui Zhao, Zenglin Xu, Xin Pang, Yinghui Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17288)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances language models by incorporating external knowledge at inference time. However, graph-based RAG systems often suffer from structural overhead and imprecise retrieval: they require costly pipelines for entity linking and relation extraction, yet frequently return subgraphs filled with loosely related or tangential content. This stems from a fundamental flaw -- semantic similarity does not imply semantic relevance. We introduce SlimRAG, a lightweight framework for retrieval without graphs. SlimRAG replaces structure-heavy components with a simple yet effective entity-aware mechanism. At indexing time, it constructs a compact entity-to-chunk table based on semantic embeddings. At query time, it identifies salient entities, retrieves and scores associated chunks, and assembles a concise, contextually relevant input -- without graph traversal or edge construction. To quantify retrieval efficiency, we propose Relative Index Token Utilization (RITU), a metric measuring the compactness of retrieved content. Experiments across multiple QA benchmarks show that SlimRAG outperforms strong flat and graph-based baselines in accuracy while reducing index size and RITU (e.g., 16.31 vs. 56+), highlighting the value of structure-free, entity-centric context selection. The code will be released soon. this https URL 

**Abstract (ZH)**: 基于检索的生成（RAG）通过在推理时 incorporare 外部知识来增强语言模型。然而，基于图的 RAG 系统往往受到结构开销和检索不精确的问题：它们需要昂贵的实体链接和关系抽取管道，但经常返回包含松散相关或不相关内容的子图。这源于一个根本性的缺陷——语义相似性不等于语义相关性。我们提出了 SlimRAG，一个轻量级的不基于图的检索框架。SlimRAG 用简单有效的实体感知机制取代了结构密集的组件。在索引时，它基于语义嵌入构建紧凑的实体到片段表。在查询时，它识别显著实体、检索和评分相关片段，并组装成简洁且上下文相关的内容，无需进行图遍历或边构建。为了量化检索效率，我们提出了相对索引_token_利用度 (RITU) 的度量标准，衡量检索内容的紧凑性。多项 QA 基准实验显示，SlimRAG 在准确率上优于强大的平铺和基于图的基线模型的同时减小了索引大小和 RITU（例如，16.31 对比 56+），突显了无结构、以实体为中心的上下文选择的价值。代码将于近期发布。 

---
# GTA: Grouped-head latenT Attention 

**Title (ZH)**: GTA: 分组头部潜注意力 

**Authors**: Luoyang Sun, Jiwen Jiang, Cheng Deng, Xinjian Wu, Haifeng Zhang, Lei Chen, Lionel Ni, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17286)  

**Abstract**: Attention mechanisms underpin the success of large language models (LLMs), yet their substantial computational and memory overhead poses challenges for optimizing efficiency and performance. A critical bottleneck arises as KV cache and attention computations scale rapidly with text length, challenging deployment on hardware with limited computational and memory resources. We observe that attention mechanisms exhibit substantial redundancy, since the KV cache can be significantly compressed and attention maps across heads display high similarity, revealing that much of the computation and storage is unnecessary. Leveraging these insights, we propose \textbf{G}rouped-Head Laten\textbf{T} \textbf{A}ttention (GTA), a novel attention mechanism that reduces memory usage and computational complexity while maintaining performance. GTA comprises two components: (1) a shared attention map mechanism that reuses attention scores across multiple heads, decreasing the key cache size; and (2) a nonlinear value decoder with learned projections that compresses the value cache into a latent space, further cutting memory needs. GTA cuts attention computation FLOPs by up to \emph{62.5\%} versus Grouped-Query Attention and shrink the KV cache by up to \emph{70\%}, all while avoiding the extra overhead of Multi-Head Latent Attention to improve LLM deployment efficiency. Consequently, GTA models achieve a \emph{2x} increase in end-to-end inference speed, with prefill benefiting from reduced computational cost and decoding benefiting from the smaller cache footprint. 

**Abstract (ZH)**: 注意力机制支撑了大规模语言模型的成功，但其巨大的计算和内存开销阻碍了效率和性能的优化。随着文本长度的增加，KV缓存和注意力计算迅速上升，成为在计算和内存资源有限的硬件上部署的瓶颈。我们观察到注意力机制存在大量冗余，KV缓存可以显著压缩，而跨头的注意力图显示出高度的相似性，表明大量的计算和存储是不必要的。基于这些洞察，我们提出了**组头潜在注意力**（GTA），这一新颖的注意力机制可以在减少存储需求和计算复杂度的同时保持性能。GTA 包含两个组成部分：(1) 一个共享注意力图机制，通过在多个头之间重用注意力评分来减少键缓存的大小；(2) 一个非线性值解码器，带有学习的投影，将值缓存压缩到潜在空间中，进一步减少内存需求。与组查询注意力相比，GTA将注意力计算FLOPs减少了高达62.5%，并且将KV缓存减少了高达70%，同时避免了多头潜在注意力的额外开销以提高大规模语言模型的部署效率。因此，GTA 模型实现了端到端推理速度的两倍提升，预填充受益于计算成本的降低，解码则受益于更小的缓存占用。 

---
# A Theoretical Framework for Virtual Power Plant Integration with Gigawatt-Scale AI Data Centers: Multi-Timescale Control and Stability Analysis 

**Title (ZH)**: 兆瓦级人工智能数据中心与虚拟电厂集成的理论框架：多时间尺度控制与稳定性分析 

**Authors**: Ali Peivandizadeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.17284)  

**Abstract**: The explosive growth of artificial intelligence has created gigawatt-scale data centers that fundamentally challenge power system operation, exhibiting power fluctuations exceeding 500 MW within seconds and millisecond-scale variations of 50-75% of thermal design power. This paper presents a comprehensive theoretical framework that reconceptualizes Virtual Power Plants (VPPs) to accommodate these extreme dynamics through a four-layer hierarchical control architecture operating across timescales from 100 microseconds to 24 hours.
We develop control mechanisms and stability criteria specifically tailored to converter-dominated systems with pulsing megawatt-scale loads. We prove that traditional VPP architectures, designed for aggregating distributed resources with response times of seconds to minutes, cannot maintain stability when confronted with AI data center dynamics exhibiting slew rates exceeding 1,000 MW/s at gigawatt scale.
Our framework introduces: (1) a sub-millisecond control layer that interfaces with data center power electronics to actively dampen power oscillations; (2) new stability criteria incorporating protection system dynamics, demonstrating that critical clearing times reduce from 150 ms to 83 ms for gigawatt-scale pulsing loads; and (3) quantified flexibility characterization showing that workload deferability enables 30% peak reduction while maintaining AI service availability above 99.95%.
This work establishes the mathematical foundations necessary for the stable integration of AI infrastructure that will constitute 50-70% of data center electricity consumption by 2030. 

**Abstract (ZH)**: 人工智能的爆炸性增长创建了 gigawatt 规模的数据中心，从根本上挑战了电力系统的运行，展示出秒级内超过 500 MW 的功率波动和毫秒级范围内 50-75% 的热设计功率变化。本文提出了一种综合理论框架，通过一种四层分级控制架构重新概念化虚拟电厂（VPP），该架构的时间跨度从 100 微秒到 24 小时，以适应这些极端动态。 

---
# CORONA: A Coarse-to-Fine Framework for Graph-based Recommendation with Large Language Models 

**Title (ZH)**: CORONA：基于图的推荐系统与大规模语言模型的粗到细框架 

**Authors**: Junze Chen, Xinjie Yang, Cheng Yang, Junfei Bao, Zeyuan Guo, Yawen Li, Chuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17281)  

**Abstract**: Recommender systems (RSs) are designed to retrieve candidate items a user might be interested in from a large pool. A common approach is using graph neural networks (GNNs) to capture high-order interaction relationships. As large language models (LLMs) have shown strong capabilities across domains, researchers are exploring their use to enhance recommendation. However, prior work limits LLMs to re-ranking results or dataset augmentation, failing to utilize their power during candidate filtering - which may lead to suboptimal performance. Instead, we propose to leverage LLMs' reasoning abilities during the candidate filtering process, and introduce Chain Of Retrieval ON grAphs (CORONA) to progressively narrow down the range of candidate items on interaction graphs with the help of LLMs: (1) First, LLM performs preference reasoning based on user profiles, with the response serving as a query to extract relevant users and items from the interaction graph as preference-assisted retrieval; (2) Then, using the information retrieved in the previous step along with the purchase history of target user, LLM conducts intent reasoning to help refine an even smaller interaction subgraph as intent-assisted retrieval; (3) Finally, we employ a GNN to capture high-order collaborative filtering information from the extracted subgraph, performing GNN-enhanced retrieval to generate the final recommendation results. The proposed framework leverages the reasoning capabilities of LLMs during the retrieval process, while seamlessly integrating GNNs to enhance overall recommendation performance. Extensive experiments on various datasets and settings demonstrate that our proposed CORONA achieves state-of-the-art performance with an 18.6% relative improvement in recall and an 18.4% relative improvement in NDCG on average. 

**Abstract (ZH)**: 基于图的链式检索系统 (CORONA): 结合大型语言模型的推理能力以优化推荐系统 

---
# Step-by-Step Reasoning Attack: Revealing 'Erased' Knowledge in Large Language Models 

**Title (ZH)**: 逐步推理攻击：揭示大型语言模型中被“擦除”的知识 

**Authors**: Yash Sinha, Manit Baser, Murari Mandal, Dinil Mon Divakaran, Mohan Kankanhalli  

**Link**: [PDF](https://arxiv.org/pdf/2506.17279)  

**Abstract**: Knowledge erasure in large language models (LLMs) is important for ensuring compliance with data and AI regulations, safeguarding user privacy, mitigating bias, and misinformation. Existing unlearning methods aim to make the process of knowledge erasure more efficient and effective by removing specific knowledge while preserving overall model performance, especially for retained information. However, it has been observed that the unlearning techniques tend to suppress and leave the knowledge beneath the surface, thus making it retrievable with the right prompts. In this work, we demonstrate that \textit{step-by-step reasoning} can serve as a backdoor to recover this hidden information. We introduce a step-by-step reasoning-based black-box attack, Sleek, that systematically exposes unlearning failures. We employ a structured attack framework with three core components: (1) an adversarial prompt generation strategy leveraging step-by-step reasoning built from LLM-generated queries, (2) an attack mechanism that successfully recalls erased content, and exposes unfair suppression of knowledge intended for retention and (3) a categorization of prompts as direct, indirect, and implied, to identify which query types most effectively exploit unlearning weaknesses. Through extensive evaluations on four state-of-the-art unlearning techniques and two widely used LLMs, we show that existing approaches fail to ensure reliable knowledge removal. Of the generated adversarial prompts, 62.5% successfully retrieved forgotten Harry Potter facts from WHP-unlearned Llama, while 50% exposed unfair suppression of retained knowledge. Our work highlights the persistent risks of information leakage, emphasizing the need for more robust unlearning strategies for erasure. 

**Abstract (ZH)**: 大规模语言模型（LLMs）中的知识擦除对于遵守数据和AI法规、保护用户隐私、减少偏见和虚假信息至关重要。现有的遗忘方法旨在通过移除特定知识同时保持整体模型性能的方式，使知识擦除过程更高效和有效，特别是对于保留信息。然而，观察到的是，遗忘技术倾向于抑制并保留知识，使其在适当的提示下可检索。在本文中，我们证明逐步推理可以用作后门，以恢复这些隐藏的信息。我们介绍了一种基于逐步推理的黑盒攻击Sleek，系统地揭示了遗忘失败。我们采用了一种结构化的攻击框架，包含三个核心组件：（1）利用来自LLM生成查询的逐步推理构建的对抗性提示生成策略，（2）成功检索被删除的内容并揭示意图保留知识的不公平抑制机制，以及（3）将提示分类为直接、间接和暗含，以确定哪种查询类型最有效地利用遗忘弱点。通过对四种最先进的遗忘技术以及两种广泛使用的LLM进行详尽评估，我们展示了现有方法无法确保可靠的知识去除。生成的对抗性提示中有62.5%成功从WHP-未删除的Llama中检索出了被遗忘的哈利·波特事实，而50%暴露了保留知识的不公平抑制。我们的工作强调了持续的信息泄露风险，强调了需要更 robust 的遗忘策略来进行去除。 

---
# Chunk Twice, Embed Once: A Systematic Study of Segmentation and Representation Trade-offs in Chemistry-Aware Retrieval-Augmented Generation 

**Title (ZH)**: 两次切分，一次嵌入：化学意识检索增强生成中切分与表示权衡的系统研究 

**Authors**: Mahmoud Amiri, Thomas Bocklitz  

**Link**: [PDF](https://arxiv.org/pdf/2506.17277)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are increasingly vital for navigating the ever-expanding body of scientific literature, particularly in high-stakes domains such as chemistry. Despite the promise of RAG, foundational design choices -- such as how documents are segmented and represented -- remain underexplored in domain-specific contexts. This study presents the first large-scale, systematic evaluation of chunking strategies and embedding models tailored to chemistry-focused RAG systems. We investigate 25 chunking configurations across five method families and evaluate 48 embedding models on three chemistry-specific benchmarks, including the newly introduced QuestChemRetrieval dataset. Our results reveal that recursive token-based chunking (specifically R100-0) consistently outperforms other approaches, offering strong performance with minimal resource overhead. We also find that retrieval-optimized embeddings -- such as Nomic and Intfloat E5 variants -- substantially outperform domain-specialized models like SciBERT. By releasing our datasets, evaluation framework, and empirical benchmarks, we provide actionable guidelines for building effective and efficient chemistry-aware RAG systems. 

**Abstract (ZH)**: 基于检索增强生成的化学专注系统的大规模系统性评估 

---
# Modal Logic for Stratified Becoming: Actualization Beyond Possible Worlds 

**Title (ZH)**: 分层生成的模态逻辑：超越可能世界的实现 

**Authors**: Alexandre Le Nepvou  

**Link**: [PDF](https://arxiv.org/pdf/2506.17276)  

**Abstract**: This article develops a novel framework for modal logic based on the idea of stratified actualization, rather than the classical model of global possible worlds. Traditional Kripke semantics treat modal operators as quantification over fully determinate alternatives, neglecting the local, dynamic, and often asymmetric nature of actualization processes. We propose a system Stratified Actualization Logic (SAL) in which modalities are indexed by levels of ontological stability, interpreted as admissibility regimes. Each modality operates over a structured layer of possibility, grounded in the internal coherence of transitions between layers. We formally define the syntax and semantics of SAL, introduce its axioms, and prove soundness and completeness. Applications are discussed in connection with temporal becoming, quantum decoherence domains, and modal metaphysics. The result is a logic that captures the ontological structure of actualization without recourse to abstract possible worlds, offering a stratified alternative to standard modal realism. 

**Abstract (ZH)**: 基于分层实现的新范式模态逻辑 

---
# Conformal Safety Shielding for Imperfect-Perception Agents 

**Title (ZH)**: Imperfect-Perception 代理的齐性安全性屏蔽 

**Authors**: William Scarbro, Calum Imrie, Sinem Getir Yaman, Kavan Fatehi, Corina S. Pasareanu, Radu Calinescu, Ravi Mangal  

**Link**: [PDF](https://arxiv.org/pdf/2506.17275)  

**Abstract**: We consider the problem of safe control in discrete autonomous agents that use learned components for imperfect perception (or more generally, state estimation) from high-dimensional observations. We propose a shield construction that provides run-time safety guarantees under perception errors by restricting the actions available to an agent, modeled as a Markov decision process, as a function of the state estimates. Our construction uses conformal prediction for the perception component, which guarantees that for each observation, the predicted set of estimates includes the actual state with a user-specified probability. The shield allows an action only if it is allowed for all the estimates in the predicted set, resulting in a local safety guarantee. We also articulate and prove a global safety property of existing shield constructions for perfect-perception agents bounding the probability of reaching unsafe states if the agent always chooses actions prescribed by the shield. We illustrate our approach with a case-study of an experimental autonomous system that guides airplanes on taxiways using high-dimensional perception DNNs. 

**Abstract (ZH)**: 我们考虑具有 learned 组件进行不完美感知（或更一般地，状态估计）的离散自主代理的安全控制问题。我们提出了一种防护构造，该构造通过根据状态估计限制代理可用的动作，为感知错误提供运行时的安全保证，将代理建模为马尔可夫决策过程。该构造使用符合预测方法进行感知组件，保证对每个观测，预测的状态集包含实际状态的概率由用户指定。防护构造仅允许如果预测集中所有估计都允许该动作，从而实现局部安全性保证。我们还阐述并证明了现有完美感知代理防护构造的全局安全性属性，该属性界定了如果代理始终选择防护构造指定的动作，则到达不安全状态的概率。我们通过一个实验自主系统的案例研究说明了这种方法，该系统使用高维感知 DNN 引导飞机在滑行道上航行。 

---
# QUST_NLP at SemEval-2025 Task 7: A Three-Stage Retrieval Framework for Monolingual and Crosslingual Fact-Checked Claim Retrieval 

**Title (ZH)**: QUST_NLP 在 SemEval-2025 任务 7 中的三级检索框架：单语和跨语言事实核查声明检索 

**Authors**: Youzheng Liu, Jiyan Liu, Xiaoman Xu, Taihang Wang, Yimin Wang, Ye Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17272)  

**Abstract**: This paper describes the participation of QUST_NLP in the SemEval-2025 Task 7. We propose a three-stage retrieval framework specifically designed for fact-checked claim retrieval. Initially, we evaluate the performance of several retrieval models and select the one that yields the best results for candidate retrieval. Next, we employ multiple re-ranking models to enhance the candidate results, with each model selecting the Top-10 outcomes. In the final stage, we utilize weighted voting to determine the final retrieval outcomes. Our approach achieved 5th place in the monolingual track and 7th place in the crosslingual track. We release our system code at: this https URL 

**Abstract (ZH)**: 本论文描述了UESTC_NLP在SemEval-2025 Task 7中的参与情况。我们提出了一种专门设计的事实核验声明检索的三阶段检索框架。首先，我们评估了几种检索模型的性能，并选择了性能最佳的候选检索模型。接着，我们采用了多种重排序模型来提升候选结果，每种模型选取Top-10结果。在最终阶段，我们利用加权投票来确定最终的检索结果。我们的方法在单语轨道中获得第5名，在跨语言轨道中获得第7名。我们已将系统代码发布在以下链接：this https URL。 

---
# CF-VLM:CounterFactual Vision-Language Fine-tuning 

**Title (ZH)**: CF-VLM：反事实视觉-语言微调 

**Authors**: Jusheng Zhang, Kaitong Cai, Yijia Fan, Jian Wang, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17267)  

**Abstract**: Recent advances in vision-language models (VLMs) have greatly improved cross-modal semantic understanding, yet significant limitations remain in fine-grained discrimination and deep causal reasoning tasks. Existing VLMs often rely on superficial statistical correlations, lacking the ability to capture the underlying causal logic between visual and textual content. To address this, we propose CounterFactual Vision-Language Fine-tuning (CF-VLM), a novel framework that enhances the causal reasoning capabilities of VLMs through the targeted use of counterfactual samples. CF-VLM introduces three complementary training objectives: maintaining foundational cross-modal alignment, reinforcing the uniqueness and stability of factual scene representations against coherent counterfactuals, and sharpening the model's sensitivity to minimal but critical causal edits. Extensive experiments demonstrate that CF-VLM consistently outperforms strong baselines and state-of-the-art methods on compositional reasoning and generalization benchmarks. Furthermore, it shows promise in mitigating visual hallucinations, indicating improved factual consistency. Our CF-VLM provides a robust foundation for deploying VLMs in high-stakes, real-world scenarios requiring reliable reasoning and interpretability. 

**Abstract (ZH)**: Recent Advances in Vision-Language Models: Addressing Limitations Through Counterfactual Vision-Language Fine-tuning 

---
# Does Multimodal Large Language Model Truly Unlearn? Stealthy MLLM Unlearning Attack 

**Title (ZH)**: 多模态大语言模型真的能够有效遗忘吗？隐秘的MLLM遗忘攻击 

**Authors**: Xianren Zhang, Hui Liu, Delvin Ce Zhang, Xianfeng Tang, Qi He, Dongwon Lee, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17265)  

**Abstract**: Multimodal Large Language Models (MLLMs) trained on massive data may memorize sensitive personal information and photos, posing serious privacy risks. To mitigate this, MLLM unlearning methods are proposed, which fine-tune MLLMs to reduce the ``forget'' sensitive information. However, it remains unclear whether the knowledge has been truly forgotten or just hidden in the model. Therefore, we propose to study a novel problem of LLM unlearning attack, which aims to recover the unlearned knowledge of an unlearned LLM. To achieve the goal, we propose a novel framework Stealthy Unlearning Attack (SUA) framework that learns a universal noise pattern. When applied to input images, this noise can trigger the model to reveal unlearned content. While pixel-level perturbations may be visually subtle, they can be detected in the semantic embedding space, making such attacks vulnerable to potential defenses. To improve stealthiness, we introduce an embedding alignment loss that minimizes the difference between the perturbed and denoised image embeddings, ensuring the attack is semantically unnoticeable. Experimental results show that SUA can effectively recover unlearned information from MLLMs. Furthermore, the learned noise generalizes well: a single perturbation trained on a subset of samples can reveal forgotten content in unseen images. This indicates that knowledge reappearance is not an occasional failure, but a consistent behavior. 

**Abstract (ZH)**: 多模态大型语言模型的未学习攻击： Stealthy Unlearning Attack (SUA) 框架 

---
# OAT-Rephrase: Optimization-Aware Training Data Rephrasing for Zeroth-Order LLM Fine-Tuning 

**Title (ZH)**: OAT-重构：面向优化的训练数据重构在零阶语言模型微调中的应用 

**Authors**: Jikai Long, Zijian Hu, Xiaodong Yu, Jianwen Xie, Zhaozhuo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17264)  

**Abstract**: Fine-tuning large language models (LLMs) using zeroth-order optimization (ZO) offers a memory-efficient alternative to gradient-based methods but suffers from slower convergence and unstable optimization due to noisy gradient estimates. This paper introduces OAT-Rephrase, an Optimization-Aware Training data rephrasing strategy that leverages an LLM to rephrase training instances based on its understanding of the ZO dynamics, specifically MeZO, derived directly from its paper. The approach incorporates a dual-stage pipeline featuring a rewriter LLM and a semantic judge, ensuring all rephrasings retain task relevance and logical consistency. Evaluations across five classification tasks and three LLM architectures demonstrate that OAT-Rephrase consistently improves MeZO fine-tuning performance, often narrowing or eliminating the gap with first-order methods. Our findings suggest that optimization-aware rephrasing serves as a reusable and low-overhead enhancement for zeroth-order tuning regimes. 

**Abstract (ZH)**: 利用零阶优化(OAT-Rephrase)意识训练数据重述策略：一种大型语言模型微调的新方法 

---
# Memory Allocation in Resource-Constrained Reinforcement Learning 

**Title (ZH)**: 资源受限强化学习中的内存分配 

**Authors**: Massimiliano Tamborski, David Abel  

**Link**: [PDF](https://arxiv.org/pdf/2506.17263)  

**Abstract**: Resource constraints can fundamentally change both learning and decision-making. We explore how memory constraints influence an agent's performance when navigating unknown environments using standard reinforcement learning algorithms. Specifically, memory-constrained agents face a dilemma: how much of their limited memory should be allocated to each of the agent's internal processes, such as estimating a world model, as opposed to forming a plan using that model? We study this dilemma in MCTS- and DQN-based algorithms and examine how different allocations of memory impact performance in episodic and continual learning settings. 

**Abstract (ZH)**: 资源约束可以根本上改变学习和决策的方式。我们探究了当使用标准强化学习算法在未知环境中导航时，记忆约束如何影响智能体的性能。具体而言，记忆受限的智能体面临一个困境：它们有限的记忆应分配给内部过程（如构建世界模型的估计）还是利用该模型形成计划？我们在这类基于MCTS和DQN的算法中研究这一困境，并考察不同记忆分配对 episodic 学习和持续学习设置中性能的影响。 

---
# AI to Identify Strain-sensitive Regions of the Optic Nerve Head Linked to Functional Loss in Glaucoma 

**Title (ZH)**: AI识别与视神经头结构变化相关的眼压功能损失的敏感区域 

**Authors**: Thanadet Chuangsuwanich, Monisha E. Nongpiur, Fabian A. Braeu, Tin A. Tun, Alexandre Thiery, Shamira Perera, Ching Lin Ho, Martin Buist, George Barbastathis, Tin Aung, Michaël J.A. Girard  

**Link**: [PDF](https://arxiv.org/pdf/2506.17262)  

**Abstract**: Objective: (1) To assess whether ONH biomechanics improves prediction of three progressive visual field loss patterns in glaucoma; (2) to use explainable AI to identify strain-sensitive ONH regions contributing to these predictions.
Methods: We recruited 237 glaucoma subjects. The ONH of one eye was imaged under two conditions: (1) primary gaze and (2) primary gaze with IOP elevated to ~35 mmHg via ophthalmo-dynamometry. Glaucoma experts classified the subjects into four categories based on the presence of specific visual field defects: (1) superior nasal step (N=26), (2) superior partial arcuate (N=62), (3) full superior hemifield defect (N=25), and (4) other/non-specific defects (N=124). Automatic ONH tissue segmentation and digital volume correlation were used to compute IOP-induced neural tissue and lamina cribrosa (LC) strains. Biomechanical and structural features were input to a Geometric Deep Learning model. Three classification tasks were performed to detect: (1) superior nasal step, (2) superior partial arcuate, (3) full superior hemifield defect. For each task, the data were split into 80% training and 20% testing sets. Area under the curve (AUC) was used to assess performance. Explainable AI techniques were employed to highlight the ONH regions most critical to each classification.
Results: Models achieved high AUCs of 0.77-0.88, showing that ONH strain improved VF loss prediction beyond morphology alone. The inferior and inferotemporal rim were identified as key strain-sensitive regions, contributing most to visual field loss prediction and showing progressive expansion with increasing disease severity.
Conclusion and Relevance: ONH strain enhances prediction of glaucomatous VF loss patterns. Neuroretinal rim, rather than the LC, was the most critical region contributing to model predictions. 

**Abstract (ZH)**: 目标：（1）评估视神经头（ONH）生物力学是否能提高对原发性开角型青光眼中三种渐进性视野损失模式的预测能力；（2）使用可解释的AI技术识别对这些预测贡献最大的敏感应力区域。

方法：我们招募了237名青光眼患者。在一眼中，在两种条件下成像ONH：（1）正视；（2）正视并通过眼动力学将眼内压（IOP）升高至约35 mmHg。青光眼专家根据特定视野缺陷的存在将受试者分为四类：（1）鼻上方阶梯（N=26），（2）鼻上部分弓状（N=62），（3）全鼻上半视野缺损（N=25），（4）其他/非特异性缺陷（N=124）。自动ONH组织分割和数字体积关联用于计算IOP诱导的神经组织和筛板（LC）应变。生物力学和结构特征被输入几何深度学习模型。进行了三项分类任务以检测：（1）鼻上方阶梯，（2）鼻上部分弓状，（3）全鼻上半视野缺损。每项任务的数据划分为80%训练集和20%测试集。使用曲线下面积（AUC）评估性能。使用可解释的AI技术突出显示对每个分类至关重要的ONH区域。

结果：模型获得了0.77-0.88的高AUC，表明ONH应变在单独依靠形态学之外增强了视野损失预测。下部和下颞部边缘被确认为关键的应变敏感区域，在视野损失预测中贡献最大，并随着疾病严重程度增加而逐渐扩大。

结论和意义：ONH应变增强了对原发性开角型青光眼中视野损失模式的预测。神经视网膜边缘而非筛板是模型预测中贡献最大的关键区域。 

---
# A Digital Twin Framework for Generation-IV Reactors with Reinforcement Learning-Enabled Health-Aware Supervisory Control 

**Title (ZH)**: 一种基于强化学习实现健康感知监督控制的第四代反应堆数字孪生框架 

**Authors**: Jasmin Y. Lim, Dimitrios Pylorof, Humberto E. Garcia, Karthik Duraisamy  

**Link**: [PDF](https://arxiv.org/pdf/2506.17258)  

**Abstract**: Generation IV (Gen-IV) nuclear power plants are envisioned to replace the current reactor fleet, bringing improvements in performance, safety, reliability, and sustainability. However, large cost investments currently inhibit the deployment of these advanced reactor concepts. Digital twins bridge real-world systems with digital tools to reduce costs, enhance decision-making, and boost operational efficiency. In this work, a digital twin framework is designed to operate the Gen-IV Fluoride-salt-cooled High-temperature Reactor, utilizing data-enhanced methods to optimize operational and maintenance policies while adhering to system constraints. The closed-loop framework integrates surrogate modeling, reinforcement learning, and Bayesian inference to streamline end-to-end communication for online regulation and self-adjustment. Reinforcement learning is used to consider component health and degradation to drive the target power generations, with constraints enforced through a Reference Governor control algorithm that ensures compliance with pump flow rate and temperature limits. These input driving modules benefit from detailed online simulations that are assimilated to measurement data with Bayesian filtering. The digital twin is demonstrated in three case studies: a one-year long-term operational period showcasing maintenance planning capabilities, short-term accuracy refinement with high-frequency measurements, and system shock capturing that demonstrates real-time recalibration capabilities when change in boundary conditions. These demonstrations validate robustness for health-aware and constraint-informed nuclear plant operation, with general applicability to other advanced reactor concepts and complex engineering systems. 

**Abstract (ZH)**: Generation IV (Gen-IV) 核电站数字孪生框架设计与应用 

---
# UltraSketchLLM: Saliency-Driven Sketching for Ultra-Low Bit LLM Compression 

**Title (ZH)**: UltraSketchLLM：基于显著性的人工智能超低比特量压缩素描表示方法 

**Authors**: Sunan Zou, Ziyun Zhang, Xueting Sun, Guojie Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.17255)  

**Abstract**: The rapid growth of large language models (LLMs) has outpaced the memory constraints of edge devices, necessitating extreme weight compression beyond the 1-bit limit. While quantization reduces model size, it is fundamentally limited to 1 bit per weight. Existing multiple-to-one compression methods either rely on mapping tables (inducing memory overhead) or incur severe accuracy degradation due to random weight grouping. We introduce UltraSketchLLM, an index-free, sketch-based framework that achieves ultra-low bit compression (down to 0.5 bits per weight) while preserving model performance. UltraSketchLLM leverages data sketching, a sub-linear representation technique from streaming applications, to map multiple weights to single values with bounded error. Our approach integrates an underestimate AbsMaxMin sketch to minimize relative errors for small weights, importance-aware space allocation to prioritize salient weights, and a straight-through estimator for compression-aware finetuning. Experiments on Llama-3.2-1B demonstrate up to 0.5-bit compression with competitive perplexity, alongside tolerable latency overhead. UltraSketchLLM offers a practical solution for deploying LLMs in resource-constrained environments. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的迅猛增长超过了边缘设备的内存限制， necessitating 极端权重压缩，远超1位限制。我们提出 UltraSketchLLM，一种无索引、基于草图的框架，实现超低位压缩（每位权重低至0.5位）同时保持模型性能。UltraSketchLLM 利用数据草图，这是一种来自流式应用的亚线性表示技术，将多个权重映射为单个值并带有有界误差。我们的方法结合了低估 AbsMaxMin 草图以最小化小权重的相对误差，基于重要性的空间分配以优先处理关键权重，并通过压缩感知微调引入直接通过估计器。实验表明，在 Llama-3.2-1B 上实现多达0.5位压缩，同时具有竞争力的困惑度和可 tolerable 的延迟开销。UltraSketchLLM 为在资源受限环境中部署 LLMs 提供了一种实际解决方案。 

---
# Keeping Up with the Models: Online Deployment and Routing of LLMs at Scale 

**Title (ZH)**: 跟上模型的步伐：大规模在线部署和路由LLM 

**Authors**: Shaoang Li, Jian Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.17254)  

**Abstract**: The rapid pace at which new large language models (LLMs) appear -- and older ones become obsolete -- forces LLM service providers to juggle a streaming inventory of models while respecting tight deployment capacity and per-query cost budgets. We cast the reality as an online decision problem that couples stage-wise deployment, made at fixed maintenance windows, with per-query routing among the models kept live. We introduce StageRoute, a hierarchical algorithm that (i) optimistically selects up to $M_max$ models for the next stage using reward upper-confidence and cost lower-confidence bounds, then (ii) solves a budget-constrained bandit sub-problem to route each incoming query. We prove that StageRoute achieves a regret of order $T^{2/3}$ and provide a matching lower bound, thereby establishing its near-optimality. Moreover, our experiments confirm the theory, demonstrating that StageRoute performs close to the optimum in practical settings. 

**Abstract (ZH)**: 新出现的大语言模型（LLM）的快速迭代及其旧版模型的迅速过时迫使LLM服务提供商在严格的部署容量和每查询成本预算下管理一个流动的模型库存，同时在固定的维护窗口内进行阶段性的部署决策。我们将这一现实问题视为结合阶段部署（在固定的维护窗口进行）和查询路由的在线决策问题。我们引入了StageRoute算法，该算法（i）乐观地根据奖励的上置信界和成本的下置信界选择最多$M_{max}$个模型进入下一阶段，然后（ii）通过预算约束的多臂 bandit 子问题解决每个新查询的路由问题。我们证明StageRoute的遗憾度为$O(T^{2/3})$，并给出匹配的下界，从而确立了其接近最优性。此外，我们的实验验证了理论，表明在实际应用中StageRoute接近最优性能。 

---
# MS-TVNet:A Long-Term Time Series Prediction Method Based on Multi-Scale Dynamic Convolution 

**Title (ZH)**: MS-TVNet：基于多尺度动态卷积的长期时间序列预测方法 

**Authors**: Chenghan Li, Mingchen Li, Yipu Liao, Ruisheng Diao  

**Link**: [PDF](https://arxiv.org/pdf/2506.17253)  

**Abstract**: Long-term time series prediction has predominantly relied on Transformer and MLP models, while the potential of convolutional networks in this domain remains underexplored. To address this gap, we introduce a novel multi-scale time series reshape module, which effectively captures the relationships among multi-period patches and variable dependencies. Building upon this module, we propose MS-TVNet, a multi-scale 3D dynamic convolutional neural network. Through comprehensive evaluations on diverse datasets, MS-TVNet demonstrates superior performance compared to baseline models, achieving state-of-the-art (SOTA) results in long-term time series prediction. Our findings highlight the effectiveness of leveraging convolutional networks for capturing complex temporal patterns, suggesting a promising direction for future research in this this http URL code is realsed on this https URL. 

**Abstract (ZH)**: 长期时间序列预测主要依赖于Transformer和MLP模型，而卷积网络在这一领域中的潜力尚未充分探索。为填补这一空白，我们提出了一种新颖的多尺度时间序列重构模块，该模块有效地捕捉了多周期片段之间的关系和可变依赖性。基于该模块，我们提出了MS-TVNet，一种多尺度3D动态卷积神经网络。通过在多种数据集上的全面评估，MS-TVNet展示了相对于基线模型的优越性能，并在长期时间序列预测中取得了最优结果。我们的研究结果强调了利用卷积网络捕捉复杂时间模式的有效性，为未来研究提供了有前途的方向。代码已在以下链接发布：https://github.com/your-repo-address。 

---
# Adaptive Sample Scheduling for Direct Preference Optimization 

**Title (ZH)**: 直接偏好优化的自适应采样调度 

**Authors**: Zixuan Huang, Yikun Ban, Lean Fu, Xiaojie Li, Zhongxiang Dai, Jianxin Li, Deqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17252)  

**Abstract**: Direct Preference Optimization (DPO) has emerged as an effective approach for aligning large language models (LLMs) with human preferences. However, its performance is highly dependent on the quality of the underlying human preference data. To address this bottleneck, prior work has explored various data selection strategies, but these methods often overlook the impact of the evolving states of the language model during the DPO process. %including active querying, response pair selection, and data pre-selection. In this paper, we introduce a novel problem: Sample Scheduling for DPO, which aims to dynamically and adaptively schedule training samples based on the model's evolving states throughout preference optimization. To solve this problem, we propose SamS, an efficient and effective algorithm that adaptively selects samples in each training batch based on the LLM's learning feedback to maximize the potential generalization performance. Notably, without modifying the core DPO algorithm, simply integrating SamS significantly improves performance across tasks, with minimal additional computational overhead. This work points to a promising new direction for improving LLM alignment through more effective utilization of fixed preference datasets. 

**Abstract (ZH)**: 直接偏好优化（DPO）已成为一种有效的方法，用于使大型语言模型（LLMs）与人类偏好保持一致。然而，其性能高度依赖于底层人类偏好数据的质量。为了应对这一瓶颈，先前的工作探索了各种数据选择策略，但这些方法往往忽略了语言模型在DPO过程中状态变化的影响。本文介绍了一个新的问题：DPO的采样调度问题，旨在根据模型在整个偏好优化过程中的状态动态和适应性地调度训练样本。为此，我们提出了SamS算法，该算法能够在每个训练批次中根据LLM的学习反馈自适应地选择样本，以最大化潜在的一般化性能。值得注意的是，仅通过将SamS集成到DPO核心算法中，即可在不增加额外计算开销的情况下显著提高各种任务的性能。这项工作指出了通过更有效地利用固定偏好数据来改进LLM对齐的有前途的新方向。 

---
# Training-free LLM Verification via Recycling Few-shot Examples 

**Title (ZH)**: 无需训练的LLM验证通过回收少量样本实例 

**Authors**: Dongseok Lee, Jimyung Hong, Dongyoung Kim, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.17251)  

**Abstract**: Although LLMs have achieved remarkable performance, the inherent stochasticity of their reasoning process and varying conclusions present significant challenges. Majority voting or Best-of-N with external verification models has been explored to find the most promising solution among multiple LLM outputs. However, these approaches have certain limitations, such as limited applicability or the cost of an additional training step. To address this problem, we propose a novel and effective framework that Recycles Few-shot examples to verify LLM outputs (Referi). Our key idea is to additionally utilize the given few-shot examples to evaluate the candidate outputs of the target query, not only using them to generate outputs as the conventional few-shot prompting setup. Specifically, Referi evaluates the generated outputs by combining two different scores, designed motivated from Bayes' rule, and subsequently selects the candidate that is both confidently determined and contextually coherent through a few additional LLM inferences. Experiments with three different LLMs and across seven diverse tasks demonstrate that our framework significantly improves the accuracy of LLMs-achieving an average gain of 4.8%-through effective response selection, without additional training. 

**Abstract (ZH)**: 虽然大语言模型（LLMs）取得了显著的性能，其推理过程中的固有随机性和结论的多样性提出了重大挑战。为了在多个LLM输出中找到最有可能的解决方案，已经探索了多数投票或Best-of-N结合外部验证模型的方法。然而，这些方法存在一定的局限性，如适用范围有限或额外的训练步骤成本。为了解决这些问题，我们提出了一种新颖且有效的方法——Recycles Few-shot examples to Verify LLM outputs（Referi），旨在通过利用给定的少量示例，不仅生成输出，还评估目标查询的候选输出。具体而言，Referi通过结合两种不同的分数，依据贝叶斯规则进行设计，随后通过少量额外的LLM推理来选择同时具有高确定性和上下文一致性的候选输出。实验结果显示，该框架在三个不同的LLM和七个不同任务上显著提高了LLM的准确性，平均提升率达到4.8%，而无需额外训练。 

---
# Towards Interpretable Adversarial Examples via Sparse Adversarial Attack 

**Title (ZH)**: 面向可解释的对抗样本：基于稀疏对抗攻击的方法 

**Authors**: Fudong Lin, Jiadong Lou, Hao Wang, Brian Jalaian, Xu Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17250)  

**Abstract**: Sparse attacks are to optimize the magnitude of adversarial perturbations for fooling deep neural networks (DNNs) involving only a few perturbed pixels (i.e., under the l0 constraint), suitable for interpreting the vulnerability of DNNs. However, existing solutions fail to yield interpretable adversarial examples due to their poor sparsity. Worse still, they often struggle with heavy computational overhead, poor transferability, and weak attack strength. In this paper, we aim to develop a sparse attack for understanding the vulnerability of CNNs by minimizing the magnitude of initial perturbations under the l0 constraint, to overcome the existing drawbacks while achieving a fast, transferable, and strong attack to DNNs. In particular, a novel and theoretical sound parameterization technique is introduced to approximate the NP-hard l0 optimization problem, making directly optimizing sparse perturbations computationally feasible. Besides, a novel loss function is designed to augment initial perturbations by maximizing the adversary property and minimizing the number of perturbed pixels simultaneously. Extensive experiments are conducted to demonstrate that our approach, with theoretical performance guarantees, outperforms state-of-the-art sparse attacks in terms of computational overhead, transferability, and attack strength, expecting to serve as a benchmark for evaluating the robustness of DNNs. In addition, theoretical and empirical results validate that our approach yields sparser adversarial examples, empowering us to discover two categories of noises, i.e., "obscuring noise" and "leading noise", which will help interpret how adversarial perturbation misleads the classifiers into incorrect predictions. Our code is available at this https URL. 

**Abstract (ZH)**: 稀疏攻击优化少量受扰像素（即在l0约束下）的对抗扰动幅度，以迷惑深度神经网络（DNNs），适合作为解释DNNs脆弱性的工具。然而，现有解决方案由于稀疏性较差，无法生成可解释的对抗样本。更糟糕的是，它们通常面临着计算开销重、迁移性差以及攻击强度弱的问题。本文旨在通过在l0约束下最小化初始扰动幅度来开发一种稀疏攻击，以克服现有方法的不足，同时实现快速、可迁移且具有强大攻击性的对抗DNNs的方法。特别地，提出了一种新的且理论上合理的参数化技术来近似NP难的l0优化问题，使得直接优化稀疏扰动在计算上可行。此外，设计了一种新的损失函数，通过同时最大化对抗特性并最小化受扰像素的数量来增强初始扰动。大量实验显示，在计算开销、迁移性和攻击强度方面，我们的方法都优于现有最先进的稀疏攻击方法，并期待成为评估DNNs鲁棒性的一个基准。此外，理论和实验证据表明，我们的方法生成了更稀疏的对抗样本，帮助我们发现两类噪声，即“遮蔽噪声”和“引导噪声”，这将有助于解释对抗扰动是如何误导分类器产生错误预测的。我们的代码可在以下链接获取。 

---
# Improving Prediction Certainty Estimation for Reliable Early Exiting via Null Space Projection 

**Title (ZH)**: 通过空域投影提高可靠早期退出的预测 certainty 估计 

**Authors**: Jianing He, Qi Zhang, Duoqian Miao, Yi Kun, Shufeng Hao, Hongyun Zhang, Zhihua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.17249)  

**Abstract**: Early exiting has demonstrated great potential in accelerating the inference of pre-trained language models (PLMs) by enabling easy samples to exit at shallow layers, eliminating the need for executing deeper layers. However, existing early exiting methods primarily rely on class-relevant logits to formulate their exiting signals for estimating prediction certainty, neglecting the detrimental influence of class-irrelevant information in the features on prediction certainty. This leads to an overestimation of prediction certainty, causing premature exiting of samples with incorrect early predictions. To remedy this, we define an NSP score to estimate prediction certainty by considering the proportion of class-irrelevant information in the features. On this basis, we propose a novel early exiting method based on the Certainty-Aware Probability (CAP) score, which integrates insights from both logits and the NSP score to enhance prediction certainty estimation, thus enabling more reliable exiting decisions. The experimental results on the GLUE benchmark show that our method can achieve an average speed-up ratio of 2.19x across all tasks with negligible performance degradation, surpassing the state-of-the-art (SOTA) ConsistentEE by 28%, yielding a better trade-off between task performance and inference efficiency. The code is available at this https URL. 

**Abstract (ZH)**: 早退出在通过使容易样本在浅层层退出以加速预训练语言模型的推理方面展示了巨大的潜力，但现有的早退出方法主要依靠类相关logits来形成退出信号以估计预测置信度，忽视了特征中类无关信息对预测置信度的负面影响，导致预测置信度估计过高，使得一些错误的早期预测提前退出。为此，我们定义了一个NSP得分来考虑特征中类无关信息的比例以估计预测置信度，并在此基础上提出了一种基于Certainty-Aware Probability (CAP)得分的新型早退出方法，该方法结合了logits和NSP得分的洞察，以提高预测置信度估计，从而实现更可靠的退出决策。GLUE基准实验结果显示，与最先进的ConsistentEE相比，我们的方法在所有任务上的平均加速比为2.19倍，性能下降可忽略不计，实现了更高的任务性能与推理效率 trade-off，代码详见此链接。 

---
# Efficient Quantification of Multimodal Interaction at Sample Level 

**Title (ZH)**: 多模态交互在样本级的高效量化 

**Authors**: Zequn Yang, Hongfa Wang, Di Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17248)  

**Abstract**: Interactions between modalities -- redundancy, uniqueness, and synergy -- collectively determine the composition of multimodal information. Understanding these interactions is crucial for analyzing information dynamics in multimodal systems, yet their accurate sample-level quantification presents significant theoretical and computational challenges. To address this, we introduce the Lightweight Sample-wise Multimodal Interaction (LSMI) estimator, rigorously grounded in pointwise information theory. We first develop a redundancy estimation framework, employing an appropriate pointwise information measure to quantify this most decomposable and measurable interaction. Building upon this, we propose a general interaction estimation method that employs efficient entropy estimation, specifically tailored for sample-wise estimation in continuous distributions. Extensive experiments on synthetic and real-world datasets validate LSMI's precision and efficiency. Crucially, our sample-wise approach reveals fine-grained sample- and category-level dynamics within multimodal data, enabling practical applications such as redundancy-informed sample partitioning, targeted knowledge distillation, and interaction-aware model ensembling. The code is available at this https URL. 

**Abstract (ZH)**: 不同模态之间的交互作用——冗余性、独特性和协同作用——共同决定了多模态信息的组成。理解这些交互作用对于分析多模态系统的信息动力学至关重要，但对其准确的样本级别量化面临重大的理论和计算挑战。为了解决这一问题，我们引入了基于点信息理论的轻量级样本级多模态交互（LSMI）估计器。我们首先开发了一种冗余估计框架，使用适当的信息测度来量化这种最具可分解性和可测量性的交互作用。在此基础上，我们提出了一种通用的交互估计方法，使用高效的熵估计方法，专门针对连续分布的样本级别估计进行了优化。在合成和真实世界数据集上的广泛实验验证了LSMI的精度和效率。最关键的是，我们的样本级别方法揭示了多模态数据中的细粒度样本级和类别级动态，使其在冗余指导的样本分割、靶向知识蒸馏和交互感知模型集成等实际应用中具有重要意义。代码可在以下链接获取：这个 https URL。 

---
# Recursive Learning-Based Virtual Buffering for Analytical Global Placement 

**Title (ZH)**: 基于递归学习的虚拟缓冲全局布线方法 

**Authors**: Andrew B. Kahng, Yiting Liu, Zhiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17247)  

**Abstract**: Due to the skewed scaling of interconnect versus cell delay in modern technology nodes, placement with buffer porosity (i.e., cell density) awareness is essential for timing closure in physical synthesis flows. However, existing approaches face two key challenges: (i) traditional van Ginneken-Lillis-style buffering approaches are computationally expensive during global placement; and (ii) machine learning-based approaches, such as BufFormer, lack a thorough consideration of Electrical Rule Check (ERC) violations and fail to "close the loop" back into the physical design flow. In this work, we propose MLBuf-RePlAce, the first open-source learning-driven virtual buffering-aware analytical global placement framework, built on top of the OpenROAD infrastructure. MLBuf-RePlAce adopts an efficient recursive learning-based generative buffering approach to predict buffer types and locations, addressing ERC violations during global placement. We compare MLBuf-RePlAce against the default virtual buffering-based timing-driven global placer in OpenROAD, using open-source testcases from the TILOS MacroPlacement and OpenROAD-flow-scripts repositories. Without degradation of post-route power, MLBuf-RePlAce achieves (maximum, average) improvements of (56%, 31%) in total negative slack (TNS) within the open-source OpenROAD flow. When evaluated by completion in a commercial flow, MLBuf-RePlAce achieves (maximum, average) improvements of (53%, 28%) in TNS with an average of 0.2% improvement in post-route power. 

**Abstract (ZH)**: 基于学习的虚拟缓存aware全局布图设计框架MLBuf-RePlAce 

---
# Graph Neural Networks in Multi-Omics Cancer Research: A Structured Survey 

**Title (ZH)**: 图神经网络在多组学癌症研究中的应用：一个结构化的综述 

**Authors**: Payam Zohari, Mostafa Haghir Chehreghani  

**Link**: [PDF](https://arxiv.org/pdf/2506.17234)  

**Abstract**: The task of data integration for multi-omics data has emerged as a powerful strategy to unravel the complex biological underpinnings of cancer. Recent advancements in graph neural networks (GNNs) offer an effective framework to model heterogeneous and structured omics data, enabling precise representation of molecular interactions and regulatory networks. This systematic review explores several recent studies that leverage GNN-based architectures in multi-omics cancer research. We classify the approaches based on their targeted omics layers, graph neural network structures, and biological tasks such as subtype classification, prognosis prediction, and biomarker discovery. The analysis reveals a growing trend toward hybrid and interpretable models, alongside increasing adoption of attention mechanisms and contrastive learning. Furthermore, we highlight the use of patient-specific graphs and knowledge-driven priors as emerging directions. This survey serves as a comprehensive resource for researchers aiming to design effective GNN-based pipelines for integrative cancer analysis, offering insights into current practices, limitations, and potential future directions. 

**Abstract (ZH)**: 多组学数据集成的任务作为揭示癌症复杂生物学机制的一种强大策略已经显现出来。基于图神经网络（GNN）的最新进展为建模异构和结构化的多组学数据提供了有效框架，能够精确表示分子互作和调控网络。本系统综述探讨了若干采用基于GNN架构的近期研究在多组学癌症研究中的应用。我们将方法根据靶向的组学层次、图神经网络结构以及亚型分类、预后预测和生物标志物发现等生物任务进行分类。分析显示，混合和可解释模型的趋势日益增长，同时注意力机制和对比学习的采用也在增加。此外，我们强调了患者特定图和知识驱动先验作为新兴方向的应用。本文综述为旨在设计有效的基于GNN的集成癌症分析管道的研究人员提供了一项全面资源，提出了当前做法、局限性和潜在未来方向的见解。 

---
# PCaM: A Progressive Focus Attention-Based Information Fusion Method for Improving Vision Transformer Domain Adaptation 

**Title (ZH)**: PCaM: 一种改进视觉变换器领域适应性的渐进聚焦注意力信息融合方法 

**Authors**: Zelin Zang, Fei Wang, Liangyu Li, Jinlin Wu, Chunshui Zhao, Zhen Lei, Baigui Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.17232)  

**Abstract**: Unsupervised Domain Adaptation (UDA) aims to transfer knowledge from a labeled source domain to an unlabeled target domain. Recent UDA methods based on Vision Transformers (ViTs) have achieved strong performance through attention-based feature alignment. However, we identify a key limitation: foreground object mismatch, where the discrepancy in foreground object size and spatial distribution across domains weakens attention consistency and hampers effective domain alignment. To address this issue, we propose the Progressive Focus Cross-Attention Mechanism (PCaM), which progressively filters out background information during cross-attention, allowing the model to focus on and fuse discriminative foreground semantics across domains. We further introduce an attentional guidance loss that explicitly directs attention toward task-relevant regions, enhancing cross-domain attention consistency. PCaM is lightweight, architecture-agnostic, and easy to integrate into existing ViT-based UDA pipelines. Extensive experiments on Office-Home, DomainNet, VisDA-2017, and remote sensing datasets demonstrate that PCaM significantly improves adaptation performance and achieves new state-of-the-art results, validating the effectiveness of attention-guided foreground fusion for domain adaptation. 

**Abstract (ZH)**: 无监督域适应（UDA）旨在将标记的源域知识转移至未标记的目标域。基于视觉变换器（ViTs）的无监督域适应方法通过基于注意力的特征对齐取得了强大的性能。然而，我们发现一个关键限制：前景对象不匹配，不同域之间前景对象大小和空间分布的差异削弱了注意力一致性，阻碍了有效的域对齐。为了解决这一问题，我们提出了渐进聚焦交叉注意力机制（PCaM），该机制在交叉注意力过程中逐步过滤背景信息，使模型能够聚焦并融合跨域的区分性前景语义。此外，我们引入了一种注意力指导损失，明确地引导注意力关注与任务相关的区域，增强跨域注意力一致性。PCaM 轻量级、架构无关且易于集成到现有的基于 ViT 的UDA流水线中。在Office-Home、DomainNet、VisDA-2017和遥感数据集上的广泛实验表明，PCaM 显著提高了适应性能，并取得了新的最佳结果，验证了注意力引导前景融合在域适应中的有效性。 

---
# MMET: A Multi-Input and Multi-Scale Transformer for Efficient PDEs Solving 

**Title (ZH)**: 多输入与多尺度变压器：一种高效的偏微分方程求解方法 

**Authors**: Yichen Luo, Jia Wang, Dapeng Lan, Yu Liu, Zhibo Pang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17230)  

**Abstract**: Partial Differential Equations (PDEs) are fundamental for modeling physical systems, yet solving them in a generic and efficient manner using machine learning-based approaches remains challenging due to limited multi-input and multi-scale generalization capabilities, as well as high computational costs. This paper proposes the Multi-input and Multi-scale Efficient Transformer (MMET), a novel framework designed to address the above challenges. MMET decouples mesh and query points as two sequences and feeds them into the encoder and decoder, respectively, and uses a Gated Condition Embedding (GCE) layer to embed input variables or functions with varying dimensions, enabling effective solutions for multi-scale and multi-input problems. Additionally, a Hilbert curve-based reserialization and patch embedding mechanism decrease the input length. This significantly reduces the computational cost when dealing with large-scale geometric models. These innovations enable efficient representations and support multi-scale resolution queries for large-scale and multi-input PDE problems. Experimental evaluations on diverse benchmarks spanning different physical fields demonstrate that MMET outperforms SOTA methods in both accuracy and computational efficiency. This work highlights the potential of MMET as a robust and scalable solution for real-time PDE solving in engineering and physics-based applications, paving the way for future explorations into pre-trained large-scale models in specific domains. This work is open-sourced at this https URL. 

**Abstract (ZH)**: 多输入多尺度高效变压器（MMET）：面向物理系统的偏微分方程求解 

---
