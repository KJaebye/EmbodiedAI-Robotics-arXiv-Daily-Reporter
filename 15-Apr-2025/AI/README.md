# RealWebAssist: A Benchmark for Long-Horizon Web Assistance with Real-World Users 

**Title (ZH)**: RealWebAssist：一种基于真实用户的大规模网页辅助基准 

**Authors**: Suyu Ye, Haojun Shi, Darren Shih, Hyokun Yun, Tanya Roosta, Tianmin Shu  

**Link**: [PDF](https://arxiv.org/pdf/2504.10445)  

**Abstract**: To achieve successful assistance with long-horizon web-based tasks, AI agents must be able to sequentially follow real-world user instructions over a long period. Unlike existing web-based agent benchmarks, sequential instruction following in the real world poses significant challenges beyond performing a single, clearly defined task. For instance, real-world human instructions can be ambiguous, require different levels of AI assistance, and may evolve over time, reflecting changes in the user's mental state. To address this gap, we introduce RealWebAssist, a novel benchmark designed to evaluate sequential instruction-following in realistic scenarios involving long-horizon interactions with the web, visual GUI grounding, and understanding ambiguous real-world user instructions. RealWebAssist includes a dataset of sequential instructions collected from real-world human users. Each user instructs a web-based assistant to perform a series of tasks on multiple websites. A successful agent must reason about the true intent behind each instruction, keep track of the mental state of the user, understand user-specific routines, and ground the intended tasks to actions on the correct GUI elements. Our experimental results show that state-of-the-art models struggle to understand and ground user instructions, posing critical challenges in following real-world user instructions for long-horizon web assistance. 

**Abstract (ZH)**: 实现长时间网络任务成功辅助，AI代理必须能够在长时间内依次遵循现实世界用户指令。为了填补这一空白，我们引入了RealWebAssist，一个用于评估在网络环境下进行长时间交互、可视化GUI定位以及理解模糊的现实世界用户指令中顺序指令跟随的新基准。RealWebAssist包含从真实世界人类用户收集的顺序指令数据集。每个用户指导基于网络的助手在多个网站上执行一系列任务。成功的代理必须理解每条指令的真实意图，跟踪用户的心理状态，理解用户特定的例行公事，并将目标任务与正确的GUI元素上的操作对接。我们的实验结果表明，最先进的模型难以理解并对接用户指令，在长时间网络辅助中遵循真实世界用户指令面临着关键挑战。 

---
# AI-Driven Code Refactoring: Using Graph Neural Networks to Enhance Software Maintainability 

**Title (ZH)**: 基于AI的代码重构：使用图神经网络提升软件可维护性 

**Authors**: Gopichand Bandarupalli  

**Link**: [PDF](https://arxiv.org/pdf/2504.10412)  

**Abstract**: This study explores Graph Neural Networks (GNNs) as a transformative tool for code refactoring, using abstract syntax trees (ASTs) to boost software maintainability. It analyzes a dataset of 2 million snippets from CodeSearchNet and a custom 75000-file GitHub Python corpus, comparing GNNs against rule-based SonarQube and decision trees. Metrics include cyclomatic complexity (target below 10), coupling (target below 5), and refactoring precision. GNNs achieve 92% accuracy, reducing complexity by 35% and coupling by 33%, outperforming SonarQube (78%, 16%) and decision trees (85%, 25%). Preprocessing fixed 60% of syntax errors. Bar graphs, tables, and AST visuals clarify results. This offers a scalable AI-driven path to cleaner codebases, which is crucial for software engineering. 

**Abstract (ZH)**: 本研究探讨图神经网络（GNNs）作为代码重构的变革性工具的应用，使用抽象语法树（ASTs）提升软件可维护性。它分析了CodeSearchNet的20万段代码片段和自定义的75000文件GitHub Python语料库，并将GNNs与基于规则的SonarQube和决策树进行比较。评估指标包括圈复杂度（目标值低于10）、内聚性（目标值低于5）和重构精度。GNNs实现了92%的准确率，降低了35%的复杂度和33%的内聚性，优于SonarQube（78%，16%）和决策树（85%，25%）。预处理固定了60%的语法错误。柱状图、表格和AST可视化图展示了结果。这为更清洁的代码库提供了一个可扩展的AI驱动路径，这对于软件工程至关重要。 

---
# Can LLMs Assist Expert Elicitation for Probabilistic Causal Modeling? 

**Title (ZH)**: LLM能否辅助专家获取概率因果建模所需信息？ 

**Authors**: Olha Shaposhnyk, Daria Zahorska, Svetlana Yanushkevich  

**Link**: [PDF](https://arxiv.org/pdf/2504.10397)  

**Abstract**: Objective: This study investigates the potential of Large Language Models (LLMs) as an alternative to human expert elicitation for extracting structured causal knowledge and facilitating causal modeling in biometric and healthcare applications.
Material and Methods: LLM-generated causal structures, specifically Bayesian networks (BNs), were benchmarked against traditional statistical methods (e.g., Bayesian Information Criterion) using healthcare datasets. Validation techniques included structural equation modeling (SEM) to verifying relationships, and measures such as entropy, predictive accuracy, and robustness to compare network structures.
Results and Discussion: LLM-generated BNs demonstrated lower entropy than expert-elicited and statistically generated BNs, suggesting higher confidence and precision in predictions. However, limitations such as contextual constraints, hallucinated dependencies, and potential biases inherited from training data require further investigation.
Conclusion: LLMs represent a novel frontier in expert elicitation for probabilistic causal modeling, promising to improve transparency and reduce uncertainty in the decision-making using such models. 

**Abstract (ZH)**: 目标：本研究调查了大型语言模型（LLMs）作为人类专家提取结构化因果知识和促进生物识别和医疗保健应用中因果建模替代方法的潜力。 

---
# Heimdall: test-time scaling on the generative verification 

**Title (ZH)**: Heimdall：生成验证的测试时缩放 

**Authors**: Wenlei Shi, Xing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2504.10337)  

**Abstract**: An AI system can create and maintain knowledge only to the extent that it can verify that knowledge itself. Recent work on long Chain-of-Thought reasoning has demonstrated great potential of LLMs on solving competitive problems, but their verification ability remains to be weak and not sufficiently investigated. In this paper, we propose Heimdall, the long CoT verification LLM that can accurately judge the correctness of solutions. With pure reinforcement learning, we boost the verification accuracy from 62.5% to 94.5% on competitive math problems. By scaling with repeated sampling, the accuracy further increases to 97.5%. Through human evaluation, Heimdall demonstrates impressive generalization capabilities, successfully detecting most issues in challenging math proofs, the type of which is not included during training. Furthermore, we propose Pessimistic Verification to extend the functionality of Heimdall to scaling up the problem solving. It calls Heimdall to judge the solutions from a solver model and based on the pessimistic principle, selects the most likely correct solution with the least uncertainty. Taking DeepSeek-R1-Distill-Qwen-32B as the solver model, Pessimistic Verification improves the solution accuracy on AIME2025 from 54.2% to 70.0% with 16x compute budget and to 83.3% with more compute budget. With the stronger solver Gemini 2.5 Pro, the score reaches 93.0%. Finally, we prototype an automatic knowledge discovery system, a ternary system where one poses questions, another provides solutions, and the third verifies the solutions. Using the data synthesis work NuminaMath for the first two components, Heimdall effectively identifies problematic records within the dataset and reveals that nearly half of the data is flawed, which interestingly aligns with the recent ablation studies from NuminaMath. 

**Abstract (ZH)**: 一种能够准确判断解决方案正确性的长链推理验证大语言模型：Heimdall 

---
# AlayaDB: The Data Foundation for Efficient and Effective Long-context LLM Inference 

**Title (ZH)**: AlayaDB：高效有效的长上下文LLM推理的数据基础 

**Authors**: Yangshen Deng, Zhengxin You, Long Xiang, Qilong Li, Peiqi Yuan, Zhaoyang Hong, Yitao Zheng, Wanting Li, Runzhong Li, Haotian Liu, Kyriakos Mouratidis, Man Lung Yiu, Huan Li, Qiaomu Shen, Rui Mao, Bo Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10326)  

**Abstract**: AlayaDB is a cutting-edge vector database system natively architected for efficient and effective long-context inference for Large Language Models (LLMs) at AlayaDB AI. Specifically, it decouples the KV cache and attention computation from the LLM inference systems, and encapsulates them into a novel vector database system. For the Model as a Service providers (MaaS), AlayaDB consumes fewer hardware resources and offers higher generation quality for various workloads with different kinds of Service Level Objectives (SLOs), when comparing with the existing alternative solutions (e.g., KV cache disaggregation, retrieval-based sparse attention). The crux of AlayaDB is that it abstracts the attention computation and cache management for LLM inference into a query processing procedure, and optimizes the performance via a native query optimizer. In this work, we demonstrate the effectiveness of AlayaDB via (i) three use cases from our industry partners, and (ii) extensive experimental results on LLM inference benchmarks. 

**Abstract (ZH)**: AlayaDB是为AlayaDB AI高效有效的Long-Context推理设计的先进向量数据库系统 

---
# Can Competition Enhance the Proficiency of Agents Powered by Large Language Models in the Realm of News-driven Time Series Forecasting? 

**Title (ZH)**: 大型语言模型驱动的新闻驱动时间序列预测代理 proficiency 是否能通过竞争得到提升？ 

**Authors**: Yuxuan Zhang, Yangyang Feng, Daifeng Li, Kexin Zhang, Junlan Chen, Bowen Deng  

**Link**: [PDF](https://arxiv.org/pdf/2504.10210)  

**Abstract**: Multi-agents-based news-driven time series forecasting is considered as a potential paradigm shift in the era of large language models (LLMs). The challenge of this task lies in measuring the influences of different news events towards the fluctuations of time series. This requires agents to possess stronger abilities of innovative thinking and the identifying misleading logic. However, the existing multi-agent discussion framework has limited enhancement on time series prediction in terms of optimizing these two capabilities. Inspired by the role of competition in fostering innovation, this study embeds a competition mechanism within the multi-agent discussion to enhance agents' capability of generating innovative thoughts. Furthermore, to bolster the model's proficiency in identifying misleading information, we incorporate a fine-tuned small-scale LLM model within the reflective stage, offering auxiliary decision-making support. Experimental results confirm that the competition can boost agents' capacity for innovative thinking, which can significantly improve the performances of time series prediction. Similar to the findings of social science, the intensity of competition within this framework can influence the performances of agents, providing a new perspective for studying LLMs-based multi-agent systems. 

**Abstract (ZH)**: 基于多agent的新闻驱动时间序列预测被认为是大型语言模型时代潜在的范式转变。这一任务的挑战在于量化不同新闻事件对时间序列波动的影响。这要求代理具备更强的创新思维能力和识别误导逻辑的能力。然而，现有的多agent讨论框架在优化这两种能力方面对时间序列预测的提升有限。受竞争促进创新的作用启发，本研究在多agent讨论中嵌入竞争机制，以增强代理的创新思维能力。此外，为了增强模型识别误导信息的能力，在反思阶段引入了一个细调的小规模LLM模型，提供辅助决策支持。实验结果证实，竞争可以提升代理的创新思维能力，显著改善时间序列预测性能。类似于社会科学的发现，该框架内的竞争强度可以影响代理的性能，为基于大型语言模型的多agent系统的研究提供新的视角。 

---
# The Future of MLLM Prompting is Adaptive: A Comprehensive Experimental Evaluation of Prompt Engineering Methods for Robust Multimodal Performance 

**Title (ZH)**: MLLM提示的未来是自适应的：面向稳健多模态性能的提示工程方法的全面实验评估 

**Authors**: Anwesha Mohanty, Venkatesh Balavadhani Parthasarathy, Arsalan Shahid  

**Link**: [PDF](https://arxiv.org/pdf/2504.10179)  

**Abstract**: Multimodal Large Language Models (MLLMs) are set to transform how machines process and generate human-like responses by integrating diverse modalities such as text, images, and code. Yet, effectively harnessing their capabilities hinges on optimal prompt engineering. We present a comprehensive experimental evaluation of seven prompt engineering methods applied to 13 open-source MLLMs over 24 tasks spanning Reasoning and Compositionality, Multimodal Understanding and Alignment, Complex Code Generation and Execution, and Knowledge Retrieval and Integration. Our approach stratifies models by parameter count into Small (<4B), Medium (4B-10B), and Large (>10B) categories and compares prompting techniques including Zero-Shot, One-Shot, Few-Shot, Chain-of-Thought, Analogical, Generated Knowledge, and Tree-of-Thought. While Large MLLMs excel in structured tasks such as code generation, achieving accuracies up to 96.88% under Few-Shot prompting, all models struggle with complex reasoning and abstract understanding, often yielding accuracies below 60% and high hallucination rates. Structured reasoning prompts frequently increased hallucination up to 75% in small models and led to longer response times (over 20 seconds in Large MLLMs), while simpler prompting methods provided more concise and efficient outputs. No single prompting method uniformly optimises all task types. Instead, adaptive strategies combining example-based guidance with selective structured reasoning are essential to enhance robustness, efficiency, and factual accuracy. Our findings offer practical recommendations for prompt engineering and support more reliable deployment of MLLMs across applications including AI-assisted coding, knowledge retrieval, and multimodal content understanding. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）通过集成文本、图像和代码等多种模态，有望革新机器处理和生成人类响应的方式。然而，有效利用其能力的关键在于优化提示工程。我们对七种提示工程方法在13个开源MLLMs上应用于24项任务进行了全面实验评估，这些任务涵盖了逻辑推理与组合性、多模态理解和对齐、复杂代码生成与执行、以及知识检索与整合。我们的方法根据参数数量将模型分为小型（<4B）、中型（4B-10B）和大型（>10B）类别，并比较了包括零-shot、一-shot、少-shot、思考链、类比、生成知识和思考树在内的提示技术。大型MLLM在结构化任务如代码生成中表现出色，准确率达到96.88%。然而，所有模型在复杂推理和抽象理解方面表现不佳，准确率往往低于60%且出现高虚构率。结构化推理提示在小型模型中将虚构率提高到75%，并导致较长的响应时间（大型MLLM超过20秒），而简单提示方法则提供了更简洁和高效的输出。没有单一的提示方法能够均匀优化所有任务类型。相反，结合示例指导与选择性结构化推理的自适应策略对于增强鲁棒性、效率和事实准确性至关重要。我们的研究结果提供了实用的提示工程建议，并支持在包括AI辅助编码、知识检索和多模态内容理解在内的应用中更可靠地部署MLLMs。 

---
# Breaking the Data Barrier -- Building GUI Agents Through Task Generalization 

**Title (ZH)**: 打破数据障碍——通过任务泛化构建GUI代理 

**Authors**: Junlei Zhang, Zichen Ding, Chang Ma, Zijie Chen, Qiushi Sun, Zhenzhong Lan, Junxian He  

**Link**: [PDF](https://arxiv.org/pdf/2504.10127)  

**Abstract**: Graphical User Interface (GUI) agents offer cross-platform solutions for automating complex digital tasks, with significant potential to transform productivity workflows. However, their performance is often constrained by the scarcity of high-quality trajectory data. To address this limitation, we propose training Vision Language Models (VLMs) on data-rich, reasoning-intensive tasks during a dedicated mid-training stage, and then examine how incorporating these tasks facilitates generalization to GUI planning scenarios. Specifically, we explore a range of tasks with readily available instruction-tuning data, including GUI perception, multimodal reasoning, and textual reasoning. Through extensive experiments across 11 mid-training tasks, we demonstrate that: (1) Task generalization proves highly effective, yielding substantial improvements across most settings. For instance, multimodal mathematical reasoning enhances performance on AndroidWorld by an absolute 6.3%. Remarkably, text-only mathematical data significantly boosts GUI web agent performance, achieving a 5.6% improvement on WebArena and 5.4% improvement on AndroidWorld, underscoring notable cross-modal generalization from text-based to visual domains; (2) Contrary to prior assumptions, GUI perception data - previously considered closely aligned with GUI agent tasks and widely utilized for training - has a comparatively limited impact on final performance; (3) Building on these insights, we identify the most effective mid-training tasks and curate optimized mixture datasets, resulting in absolute performance gains of 8.0% on WebArena and 12.2% on AndroidWorld. Our work provides valuable insights into cross-domain knowledge transfer for GUI agents and offers a practical approach to addressing data scarcity challenges in this emerging field. The code, data and models will be available at this https URL. 

**Abstract (ZH)**: 图形用户界面（GUI）代理提供了一种跨平台解决方案，用于自动化复杂的数字任务，并具有显著潜力以转型生产流程工作流。然而，其性能常常受到高质量轨迹数据稀缺性的限制。为了解决这一局限性，我们提出在专门的中训练阶段对视觉语言模型（VLMs）进行训练，以执行数据丰富且推理密集型的任务，然后探讨这些任务如何促进对GUI规划场景的一般化。具体而言，我们探索了一系列具有现成的指令调优数据的任务，包括GUI感知、多模态推理和文本推理。通过在11个中训练任务上的广泛实验，我们展示了以下结果：(1) 任务泛化证明非常有效，在大多数情况下都取得了显著改进。例如，多模态数学推理在AndroidWorld上的绝对改进率为6.3%。令人惊讶的是，仅基于文本的数学数据显著提升了GUI网络代理的性能，在WebArena上提高了5.6%，在AndroidWorld上提高了5.4%，突显了从基于文本到视觉领域的显著跨模态泛化；(2) 与先前的假设相反，GUI感知数据（以前被认为与GUI代理任务高度一致且广泛用于训练）对最终性能的影响相对有限；(3) 基于这些见解，我们确定了最有效的中训练任务，并编排了优化混合数据集，分别在WebArena和AndroidWorld上实现了绝对性能提升8.0%和12.2%。我们的工作为GUI代理的跨域知识迁移提供了有价值的认识，并为解决这一新兴领域中数据稀缺性挑战提供了实用方法。有关代码、数据和模型将可在此处找到。 

---
# RealSafe-R1: Safety-Aligned DeepSeek-R1 without Compromising Reasoning Capability 

**Title (ZH)**: RealSafe-R1：安全对齐的DeepSeek-R1而不牺牲推理能力 

**Authors**: Yichi Zhang, Zihao Zeng, Dongbai Li, Yao Huang, Zhijie Deng, Yinpeng Dong  

**Link**: [PDF](https://arxiv.org/pdf/2504.10081)  

**Abstract**: Large Reasoning Models (LRMs), such as OpenAI o1 and DeepSeek-R1, have been rapidly progressing and achieving breakthrough performance on complex reasoning tasks such as mathematics and coding. However, the open-source R1 models have raised safety concerns in wide applications, such as the tendency to comply with malicious queries, which greatly impacts the utility of these powerful models in their applications. In this paper, we introduce RealSafe-R1 as safety-aligned versions of DeepSeek-R1 distilled models. To train these models, we construct a dataset of 15k safety-aware reasoning trajectories generated by DeepSeek-R1, under explicit instructions for expected refusal behavior. Both quantitative experiments and qualitative case studies demonstrate the models' improvements, which are shown in their safety guardrails against both harmful queries and jailbreak attacks. Importantly, unlike prior safety alignment efforts that often compromise reasoning performance, our method preserves the models' reasoning capabilities by maintaining the training data within the original distribution of generation. Model weights of RealSafe-R1 are open-source at this https URL. 

**Abstract (ZH)**: 大型推理模型（LRMs）如OpenAI o1和DeepSeek-R1在复杂推理任务（如数学和编程）上取得了突破性性能，但开源的R1模型在广泛应用中引发了安全 Concern，如倾向于遵守恶意查询，这极大地影响了这些强大模型的应用价值。本文介绍了RealSafe-R1作为与安全对齐的DeepSeek-R1精简模型。通过在明确的拒绝行为指示下生成15,000个安全意识推理轨迹训练这些模型。定量实验和定性案例研究均证明了这些模型在抵抗有害查询和脱逃攻击方面的安全性增强。重要的是，与此前常常牺牲推理性能的安全对齐方法不同，我们的方法通过保持训练数据在生成原始分布内，保留了模型的推理能力。RealSafe-R1的模型权重在此处开放获取：this https URL。 

---
# MMKB-RAG: A Multi-Modal Knowledge-Based Retrieval-Augmented Generation Framework 

**Title (ZH)**: MMKB-RAG：一个多模态知识增强检索生成框架 

**Authors**: Zihan Ling, Zhiyao Guo, Yixuan Huang, Yi An, Shuai Xiao, Jinsong Lan, Xiaoyong Zhu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.10074)  

**Abstract**: Recent advancements in large language models (LLMs) and multi-modal LLMs have been remarkable. However, these models still rely solely on their parametric knowledge, which limits their ability to generate up-to-date information and increases the risk of producing erroneous content. Retrieval-Augmented Generation (RAG) partially mitigates these challenges by incorporating external data sources, yet the reliance on databases and retrieval systems can introduce irrelevant or inaccurate documents, ultimately undermining both performance and reasoning quality. In this paper, we propose Multi-Modal Knowledge-Based Retrieval-Augmented Generation (MMKB-RAG), a novel multi-modal RAG framework that leverages the inherent knowledge boundaries of models to dynamically generate semantic tags for the retrieval process. This strategy enables the joint filtering of retrieved documents, retaining only the most relevant and accurate references. Extensive experiments on knowledge-based visual question-answering tasks demonstrate the efficacy of our approach: on the E-VQA dataset, our method improves performance by +4.2\% on the Single-Hop subset and +0.4\% on the full dataset, while on the InfoSeek dataset, it achieves gains of +7.8\% on the Unseen-Q subset, +8.2\% on the Unseen-E subset, and +8.1\% on the full dataset. These results highlight significant enhancements in both accuracy and robustness over the current state-of-the-art MLLM and RAG frameworks. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）和多模态LLMs取得了显著进展。然而，这些模型仍然仅依赖于参数化的知识，这限制了它们生成最新信息的能力，并增加了生成错误内容的风险。检索增强生成（RAG）在一定程度上缓解了这些问题，通过引入外部数据源参与，但仍依赖于数据库和检索系统，这可能会引入无关或不准确的文档，最终损害了性能和推理质量。本文提出了一种新颖的多模态知识增强检索增强生成（MMKB-RAG）框架，该框架利用模型固有的知识边界动态生成语义标签以过滤检索过程中的文档，仅保留最相关和准确的参考文献。在基于知识的视觉问答任务上进行的广泛实验表明，该方法的有效性：在E-VQA数据集中，我们的方法在Single-Hop子集上的性能提高了4.2%，在完整数据集上提高了0.4%；在InfoSeek数据集中，分别在未见问题（Unseen-Q）子集、未见实体（Unseen-E）子集和完整数据集上获得了7.8%、8.2%和8.1%的性能提升。这些结果突显了与当前最先进的MLLM和RAG框架相比，在准确性和鲁棒性方面的显著改进。 

---
# Pay Attention to What and Where? Interpretable Feature Extractor in Vision-based Deep Reinforcement Learning 

**Title (ZH)**: 基于视觉的深度强化学习中可解释的特征提取关注什么和哪里 

**Authors**: Tien Pham, Angelo Cangelosi  

**Link**: [PDF](https://arxiv.org/pdf/2504.10071)  

**Abstract**: Current approaches in Explainable Deep Reinforcement Learning have limitations in which the attention mask has a displacement with the objects in visual input. This work addresses a spatial problem within traditional Convolutional Neural Networks (CNNs). We propose the Interpretable Feature Extractor (IFE) architecture, aimed at generating an accurate attention mask to illustrate both "what" and "where" the agent concentrates on in the spatial domain. Our design incorporates a Human-Understandable Encoding module to generate a fully interpretable attention mask, followed by an Agent-Friendly Encoding module to enhance the agent's learning efficiency. These two components together form the Interpretable Feature Extractor for vision-based deep reinforcement learning to enable the model's interpretability. The resulting attention mask is consistent, highly understandable by humans, accurate in spatial dimension, and effectively highlights important objects or locations in visual input. The Interpretable Feature Extractor is integrated into the Fast and Data-efficient Rainbow framework, and evaluated on 57 ATARI games to show the effectiveness of the proposed approach on Spatial Preservation, Interpretability, and Data-efficiency. Finally, we showcase the versatility of our approach by incorporating the IFE into the Asynchronous Advantage Actor-Critic Model. 

**Abstract (ZH)**: 当前可解释的深度强化学习方法在视觉输入的空间注意力掩码存在偏差问题。本项工作解决了传统卷积神经网络中的空间问题。我们提出了一种可解释特征提取器（IFE）架构，旨在生成准确的空间注意力掩码，以明确表示智能体在空间域中关注的“什么”和“哪里”。设计中包含一个人工可理解编码模块以生成完全可解释的空间注意力掩码，以及一个智能体友好的编码模块以提高智能体的学习效率。这两个组件共同构成了基于视觉的深度强化学习的可解释特征提取器，以提高模型的可解释性。生成的空间注意力掩码一致、高度可人工理解、空间维度准确，并有效突出视觉输入中的重要对象或位置。将可解释特征提取器集成到快速高效Rainbow框架中，并在57个ATARI游戏中对其进行评估，以展示所提出方法在空间保持、可解释性和数据效率方面的有效性。最后，通过将IFE集成到异步优势动作评论器模型中，展示了我们方法的多功能性。 

---
# CHARM: Calibrating Reward Models With Chatbot Arena Scores 

**Title (ZH)**: CHARM: 用聊天机器人竞技场评分校准奖励模型 

**Authors**: Xiao Zhu, Chenmien Tan, Pinzhen Chen, Rico Sennrich, Yanlin Zhang, Hanxu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.10045)  

**Abstract**: Reward models (RMs) play a crucial role in Reinforcement Learning from Human Feedback by serving as proxies for human preferences in aligning large language models. In this paper, we identify a model preference bias in RMs, where they systematically assign disproportionately high scores to responses from certain policy models. This bias distorts ranking evaluations and leads to unfair judgments. To address this issue, we propose a calibration method named CHatbot Arena calibrated Reward Modeling (CHARM) that leverages Elo scores from the Chatbot Arena leaderboard to mitigate RM overvaluation. We also introduce a Mismatch Degree metric to measure this preference bias. Our approach is computationally efficient, requiring only a small preference dataset for continued training of the RM. We conduct extensive experiments on reward model benchmarks and human preference alignment. Results demonstrate that our calibrated RMs (1) achieve improved evaluation accuracy on RM-Bench and the Chat-Hard domain of RewardBench, and (2) exhibit a stronger correlation with human preferences by producing scores more closely aligned with Elo rankings. By mitigating model preference bias, our method provides a generalizable and efficient solution for building fairer and more reliable reward models. 

**Abstract (ZH)**: 奖励模型在大规模语言模型与人类反馈强化学习对齐中的作用至关重要。本文识别出奖励模型中的模型偏好偏差，即它们系统地对某些策略模型的响应赋予过高的分数。为了应对这一问题，我们提出了一种名为CHatbot Arena校准奖励建模（CHARM）的方法，利用Chatbot Arena排行榜上的Elo分数来减轻奖励模型的高估。我们还引入了一个不匹配度度量来衡量这种偏好偏差。我们的方法计算效率高，只需少量偏好数据集即可继续训练奖励模型。我们在奖励模型基准和人类偏好对齐方面进行了广泛实验。结果表明，我们的校准奖励模型（1）在RM-Bench和RewardBench的Chat-Hard领域中实现了更高的评估准确性；（2）与人类偏好表现出更强的相关性，生成的分数与Elo排名更为接近。通过减轻模型偏好偏差，我们的方法为构建更公平和可靠的奖励模型提供了通用和高效的解决方案。 

---
# Reasoning Models Can Be Effective Without Thinking 

**Title (ZH)**: 推理模型可以在不思考的情况下有效。 

**Authors**: Wenjie Ma, Jingxuan He, Charlie Snell, Tyler Griggs, Sewon Min, Matei Zaharia  

**Link**: [PDF](https://arxiv.org/pdf/2504.09858)  

**Abstract**: Recent LLMs have significantly improved reasoning capabilities, primarily by including an explicit, lengthy Thinking process as part of generation. In this paper, we question whether this explicit thinking is necessary. Using the state-of-the-art DeepSeek-R1-Distill-Qwen, we find that bypassing the thinking process via simple prompting, denoted as NoThinking, can be surprisingly effective. When controlling for the number of tokens, NoThinking outperforms Thinking across a diverse set of seven challenging reasoning datasets--including mathematical problem solving, formal theorem proving, and coding--especially in low-budget settings, e.g., 51.3 vs. 28.9 on ACM 23 with 700 tokens. Notably, the performance of NoThinking becomes more competitive with pass@k as k increases. Building on this observation, we demonstrate that a parallel scaling approach that uses NoThinking to generate N outputs independently and aggregates them is highly effective. For aggregation, we use task-specific verifiers when available, or we apply simple best-of-N strategies such as confidence-based selection. Our method outperforms a range of baselines with similar latency using Thinking, and is comparable to Thinking with significantly longer latency (up to 9x). Together, our research encourages a reconsideration of the necessity of lengthy thinking processes, while also establishing a competitive reference for achieving strong reasoning performance in low-budget settings or at low latency using parallel scaling. 

**Abstract (ZH)**: Recent LLMs显著提升了推理能力，主要是通过在生成过程中加入显式的 lengthy 推理过程。本文质疑这一显式推理是否必要。使用state-of-the-art模型DeepSeek-R1-Distill-Qwen，我们发现通过简单的提示绕过推理过程（标记为NoThinking）可以出人意料地有效。在控制令牌数量的情况下，NoThinking在包括数学问题求解、形式定理证明和编程在内的七个具有挑战性的推理数据集中表现优于显式推理，尤其是在低预算设置中，例如在ACM 23数据集中使用700个令牌时，NoThinking的表现为51.3，而显式推理为28.9。值得注意的是，随着pass@k值的增加，NoThinking的性能更具竞争力。基于这一观察，我们展示了使用NoThinking独立生成N个输出并进行聚合的并行扩展方法非常有效。对于聚合，当可用时使用任务特定的验证器，或应用简单的top-N策略，如基于置信度的选择。我们的方法在具有类似延迟的多种 baselines 上表现更优，并与具有明显更长延迟（最多9倍）的显式推理方法相当。我们的研究鼓励重新考虑冗长推理过程的必要性，同时也为在低预算设置或低延迟下实现强大推理性能建立了竞争参考。 

---
# A Survey of Large Language Model-Powered Spatial Intelligence Across Scales: Advances in Embodied Agents, Smart Cities, and Earth Science 

**Title (ZH)**: 大型语言模型驱动的空间智能综述：实体代理、智能城市和地球科学方面的进展 

**Authors**: Jie Feng, Jinwei Zeng, Qingyue Long, Hongyi Chen, Jie Zhao, Yanxin Xi, Zhilun Zhou, Yuan Yuan, Shengyuan Wang, Qingbin Zeng, Songwei Li, Yunke Zhang, Yuming Lin, Tong Li, Jingtao Ding, Chen Gao, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.09848)  

**Abstract**: Over the past year, the development of large language models (LLMs) has brought spatial intelligence into focus, with much attention on vision-based embodied intelligence. However, spatial intelligence spans a broader range of disciplines and scales, from navigation and urban planning to remote sensing and earth science. What are the differences and connections between spatial intelligence across these fields? In this paper, we first review human spatial cognition and its implications for spatial intelligence in LLMs. We then examine spatial memory, knowledge representations, and abstract reasoning in LLMs, highlighting their roles and connections. Finally, we analyze spatial intelligence across scales -- from embodied to urban and global levels -- following a framework that progresses from spatial memory and understanding to spatial reasoning and intelligence. Through this survey, we aim to provide insights into interdisciplinary spatial intelligence research and inspire future studies. 

**Abstract (ZH)**: 过去一年中，大型语言模型（LLMs）的发展使空间智能受到关注，尤其是基于视觉的体态智能。然而，空间智能涵盖了更广泛的学科和尺度，从导航和城市规划到遥感和地球科学。这些领域的空间智能之间有何差异和联系？在本文中，我们首先回顾人类空间认知及其对LLMs中空间智能的启示。然后，我们探讨LLMs中的空间记忆、知识表示和抽象推理，强调它们的角色和联系。最后，我们分析从体态到城市和全球尺度的空间智能，遵循从空间记忆和理解到空间推理和智能的框架。通过这次调查，我们旨在为跨学科空间智能研究提供见解，并激发未来的研究。 

---
# Two Heads are Better Than One: Test-time Scaling of Multi-agent Collaborative Reasoning 

**Title (ZH)**: 两个头胜过一个：多智能体协作推理的测试时缩放 

**Authors**: Can Jin, Hongwu Peng, Qixin Zhang, Yujin Tang, Dimitris N. Metaxas, Tong Che  

**Link**: [PDF](https://arxiv.org/pdf/2504.09772)  

**Abstract**: Multi-agent systems (MAS) built on large language models (LLMs) offer a promising path toward solving complex, real-world tasks that single-agent systems often struggle to manage. While recent advancements in test-time scaling (TTS) have significantly improved single-agent performance on challenging reasoning tasks, how to effectively scale collaboration and reasoning in MAS remains an open question. In this work, we introduce an adaptive multi-agent framework designed to enhance collaborative reasoning through both model-level training and system-level coordination. We construct M500, a high-quality dataset containing 500 multi-agent collaborative reasoning traces, and fine-tune Qwen2.5-32B-Instruct on this dataset to produce M1-32B, a model optimized for multi-agent collaboration. To further enable adaptive reasoning, we propose a novel CEO agent that dynamically manages the discussion process, guiding agent collaboration and adjusting reasoning depth for more effective problem-solving. Evaluated in an open-source MAS across a range of tasks-including general understanding, mathematical reasoning, and coding-our system significantly outperforms strong baselines. For instance, M1-32B achieves 12% improvement on GPQA-Diamond, 41% on AIME2024, and 10% on MBPP-Sanitized, matching the performance of state-of-the-art models like DeepSeek-R1 on some tasks. These results highlight the importance of both learned collaboration and adaptive coordination in scaling multi-agent reasoning. Code is available at this https URL 

**Abstract (ZH)**: 基于大型语言模型的多代理系统（MAS）为解决单代理系统常常难以管理的复杂现实任务提供了有希望的途径。尽管近期在测试时扩展（TTS）方面的进展显著提升了单代理在挑战性推理任务上的性能，但在MAS中如何有效扩展协作和推理仍然是一个开放的问题。在这项工作中，我们提出了一种自适应多代理框架，旨在通过模型级训练和系统级协调来增强协作推理。我们构建了包含500个协作推理轨迹的高质量数据集M500，并在该数据集上微调Qwen2.5-32B-Instruct，生成了专为多代理协作优化的M1-32B模型。为进一步实现自适应推理，我们提出了一个新颖的CEO代理，它能够动态管理讨论过程，引导代理之间的协作，并根据问题解决的需要调整推理深度。在一系列任务中（包括一般理解、数学推理和编程），我们的系统显著优于强基线系统。例如，M1-32B在GPQA-Diamond上实现了12%的提升，在AIME2024上实现了41%的提升，在MBPP-Sanitized上实现了10%的提升，部分任务中的性能与先进模型DeepSeek-R1相当。这些结果突显了在扩展多代理推理过程中学习到的协作和自适应协调的重要性。代码请访问 this https URL。 

---
# (How) Do reasoning models reason? 

**Title (ZH)**: 推理模型是如何推理的？ 

**Authors**: Subbarao Kambhampati, Kaya Stechly, Karthik Valmeekam  

**Link**: [PDF](https://arxiv.org/pdf/2504.09762)  

**Abstract**: We will provide a broad unifying perspective on the recent breed of Large Reasoning Models (LRMs) such as OpenAI o1 and DeepSeek R1, including their promise, sources of power, misconceptions and limitations. 

**Abstract (ZH)**: 我们将在广义统一的视角下探讨近期的大型推理模型（LRMs）如OpenAI o1和DeepSeek R1，包括它们的潜力、力量来源、误解以及局限性。 

---
# Can LLM feedback enhance review quality? A randomized study of 20K reviews at ICLR 2025 

**Title (ZH)**: 大规模语言模型反馈能否提升评审质量？ICLR 2025的一项随机研究 

**Authors**: Nitya Thakkar, Mert Yuksekgonul, Jake Silberg, Animesh Garg, Nanyun Peng, Fei Sha, Rose Yu, Carl Vondrick, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2504.09737)  

**Abstract**: Peer review at AI conferences is stressed by rapidly rising submission volumes, leading to deteriorating review quality and increased author dissatisfaction. To address these issues, we developed Review Feedback Agent, a system leveraging multiple large language models (LLMs) to improve review clarity and actionability by providing automated feedback on vague comments, content misunderstandings, and unprofessional remarks to reviewers. Implemented at ICLR 2025 as a large randomized control study, our system provided optional feedback to more than 20,000 randomly selected reviews. To ensure high-quality feedback for reviewers at this scale, we also developed a suite of automated reliability tests powered by LLMs that acted as guardrails to ensure feedback quality, with feedback only being sent to reviewers if it passed all the tests. The results show that 27% of reviewers who received feedback updated their reviews, and over 12,000 feedback suggestions from the agent were incorporated by those reviewers. This suggests that many reviewers found the AI-generated feedback sufficiently helpful to merit updating their reviews. Incorporating AI feedback led to significantly longer reviews (an average increase of 80 words among those who updated after receiving feedback) and more informative reviews, as evaluated by blinded researchers. Moreover, reviewers who were selected to receive AI feedback were also more engaged during paper rebuttals, as seen in longer author-reviewer discussions. This work demonstrates that carefully designed LLM-generated review feedback can enhance peer review quality by making reviews more specific and actionable while increasing engagement between reviewers and authors. The Review Feedback Agent is publicly available at this https URL. 

**Abstract (ZH)**: AI生成的审查反馈代理提高评审质量并增强作者与评审员的互动 

---
# InfoMAE: Pair-Efficient Cross-Modal Alignment for Multimodal Time-Series Sensing Signals 

**Title (ZH)**: InfoMAE: 对偶高效跨模态对齐多模态时间序列传感信号 

**Authors**: Tomoyoshi Kimura, Xinlin Li, Osama Hanna, Yatong Chen, Yizhuo Chen, Denizhan Kara, Tianshi Wang, Jinyang Li, Xiaomin Ouyang, Shengzhong Liu, Mani Srivastava, Suhas Diggavi, Tarek Abdelzaher  

**Link**: [PDF](https://arxiv.org/pdf/2504.09707)  

**Abstract**: Standard multimodal self-supervised learning (SSL) algorithms regard cross-modal synchronization as implicit supervisory labels during pretraining, thus posing high requirements on the scale and quality of multimodal samples. These constraints significantly limit the performance of sensing intelligence in IoT applications, as the heterogeneity and the non-interpretability of time-series signals result in abundant unimodal data but scarce high-quality multimodal pairs. This paper proposes InfoMAE, a cross-modal alignment framework that tackles the challenge of multimodal pair efficiency under the SSL setting by facilitating efficient cross-modal alignment of pretrained unimodal representations. InfoMAE achieves \textit{efficient cross-modal alignment} with \textit{limited data pairs} through a novel information theory-inspired formulation that simultaneously addresses distribution-level and instance-level alignment. Extensive experiments on two real-world IoT applications are performed to evaluate InfoMAE's pairing efficiency to bridge pretrained unimodal models into a cohesive joint multimodal model. InfoMAE enhances downstream multimodal tasks by over 60% with significantly improved multimodal pairing efficiency. It also improves unimodal task accuracy by an average of 22%. 

**Abstract (ZH)**: InfoMAE：一种在自监督学习框架下高效跨模态对齐的跨模态一致性框架 

---
# MLRC-Bench: Can Language Agents Solve Machine Learning Research Challenges? 

**Title (ZH)**: MLRC-Bench: 语言代理能否解决机器学习研究挑战？ 

**Authors**: Yunxiang Zhang, Muhammad Khalifa, Shitanshu Bhushan, Grant D Murphy, Lajanugen Logeswaran, Jaekyeom Kim, Moontae Lee, Honglak Lee, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09702)  

**Abstract**: Existing evaluation of large language model (LLM) agents on scientific discovery lacks objective baselines and metrics to assess the viability of their proposed methods. To address this issue, we introduce MLRC-Bench, a benchmark designed to quantify how effectively language agents can tackle challenging Machine Learning (ML) Research Competitions. Our benchmark highlights open research problems that demand novel methodologies, in contrast to recent benchmarks such as OpenAI's MLE-Bench (Chan et al., 2024) and METR's RE-Bench (Wijk et al., 2024), which focus on well-established research tasks that are largely solvable through sufficient engineering effort. Unlike prior work, e.g., AI Scientist (Lu et al., 2024b), which evaluates the end-to-end agentic pipeline by using LLM-as-a-judge, MLRC-Bench measures the key steps of proposing and implementing novel research methods and evaluates them with newly proposed rigorous protocol and objective metrics. Our curated suite of 7 competition tasks reveals significant challenges for LLM agents. Even the best-performing tested agent (gemini-exp-1206 under MLAB (Huang et al., 2024a)) closes only 9.3% of the gap between baseline and top human participant scores. Furthermore, our analysis reveals a misalignment between the LLM-judged innovation and their actual performance on cutting-edge ML research problems. MLRC-Bench is a dynamic benchmark, which is designed to continually grow with new ML competitions to encourage rigorous and objective evaluations of AI's research capabilities. 

**Abstract (ZH)**: MLRC-Bench：评估大型语言模型在机器学习研究竞赛中提出新颖方法有效性基准 

---
# EmoAgent: Assessing and Safeguarding Human-AI Interaction for Mental Health Safety 

**Title (ZH)**: EmoAgent: 评估与保障人机交互的心理健康安全 

**Authors**: Jiahao Qiu, Yinghui He, Xinzhe Juan, Yiming Wang, Yuhan Liu, Zixin Yao, Yue Wu, Xun Jiang, Ling Yang, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09689)  

**Abstract**: The rise of LLM-driven AI characters raises safety concerns, particularly for vulnerable human users with psychological disorders. To address these risks, we propose EmoAgent, a multi-agent AI framework designed to evaluate and mitigate mental health hazards in human-AI interactions. EmoAgent comprises two components: EmoEval simulates virtual users, including those portraying mentally vulnerable individuals, to assess mental health changes before and after interactions with AI characters. It uses clinically proven psychological and psychiatric assessment tools (PHQ-9, PDI, PANSS) to evaluate mental risks induced by LLM. EmoGuard serves as an intermediary, monitoring users' mental status, predicting potential harm, and providing corrective feedback to mitigate risks. Experiments conducted in popular character-based chatbots show that emotionally engaging dialogues can lead to psychological deterioration in vulnerable users, with mental state deterioration in more than 34.4% of the simulations. EmoGuard significantly reduces these deterioration rates, underscoring its role in ensuring safer AI-human interactions. Our code is available at: this https URL 

**Abstract (ZH)**: LLM驱动的AI角色崛起引发了安全 concern，特别是对心理障碍的人类用户。为此，我们提出 EmoAgent，这是一种多Agent AI框架，旨在评估和缓解人类与AI交互中的心理健康风险。EmoAgent 包含两个组件：EmoEval 仿真虚拟用户，包括模拟心理健康脆弱个体的用户，以评估与AI角色交互前后的心理健康变化。它使用临床验证的心理和精神病评估工具（PHQ-9、PDI、PANSS）来评估由LLM引起的心理风险。EmoGuard 作为中介，监测用户的心理状态，预测潜在危害，并提供纠正反馈以缓解风险。在流行的基于角色的聊天机器人中进行的实验显示，情感 engaging 的对话可能导致脆弱用户的心理恶化，在超过 34.4% 的仿真中观察到心理状态恶化。EmoGuard 显著降低了这些恶化率，凸显了其在确保更安全的人工智能与人类交互中的作用。我们的代码可在以下链接获取：this https URL。 

---
# Building AI Service Repositories for On-Demand Service Orchestration in 6G AI-RAN 

**Title (ZH)**: 构建6G AI-RAN中按需服务编排的AI服务仓库 

**Authors**: Yun Tang, Mengbang Zou, Udhaya Chandhar Srinivasan, Obumneme Umealor, Dennis Kevogo, Benjamin James Scott, Weisi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2504.09647)  

**Abstract**: Efficient orchestration of AI services in 6G AI-RAN requires well-structured, ready-to-deploy AI service repositories combined with orchestration methods adaptive to diverse runtime contexts across radio access, edge, and cloud layers. Current literature lacks comprehensive frameworks for constructing such repositories and generally overlooks key practical orchestration factors. This paper systematically identifies and categorizes critical attributes influencing AI service orchestration in 6G networks and introduces an open-source, LLM-assisted toolchain that automates service packaging, deployment, and runtime profiling. We validate the proposed toolchain through the Cranfield AI Service repository case study, demonstrating significant automation benefits, reduced manual coding efforts, and the necessity of infrastructure-specific profiling, paving the way for more practical orchestration frameworks. 

**Abstract (ZH)**: 在6G AI-RAN中高效 orchestrating AI 服务需要结构良好且可部署的 AI 服务仓库，结合能够适应无线接入、边缘和云各层 diverse 运行时上下文的编排方法。当前文献缺乏构建此类仓库的综合框架，并且通常忽略关键的实践编排因素。本文系统地识别并分类了影响6G 网络中 AI 服务编排的关键属性，并介绍了一个开源的、基于大规模语言模型的工具链，该工具链自动完成服务打包、部署和运行时性能分析。我们通过 Cranfield AI 服务仓库案例研究验证了所提出的工具链，展示了显著的自动化优势、减少了手动编码工作，并强调了基础设施特定性能分析的必要性，从而为更实用的编排框架铺平道路。 

---
# A Two-Stage Interpretable Matching Framework for Causal Inference 

**Title (ZH)**: 两阶段可解释匹配框架用于因果推理 

**Authors**: Sahil Shikalgar, Md. Noor-E-Alam  

**Link**: [PDF](https://arxiv.org/pdf/2504.09635)  

**Abstract**: Matching in causal inference from observational data aims to construct treatment and control groups with similar distributions of covariates, thereby reducing confounding and ensuring an unbiased estimation of treatment effects. This matched sample closely mimics a randomized controlled trial (RCT), thus improving the quality of causal estimates. We introduce a novel Two-stage Interpretable Matching (TIM) framework for transparent and interpretable covariate matching. In the first stage, we perform exact matching across all available covariates. For treatment and control units without an exact match in the first stage, we proceed to the second stage. Here, we iteratively refine the matching process by removing the least significant confounder in each iteration and attempting exact matching on the remaining covariates. We learn a distance metric for the dropped covariates to quantify closeness to the treatment unit(s) within the corresponding strata. We used these high- quality matches to estimate the conditional average treatment effects (CATEs). To validate TIM, we conducted experiments on synthetic datasets with varying association structures and correlations. We assessed its performance by measuring bias in CATE estimation and evaluating multivariate overlap between treatment and control groups before and after matching. Additionally, we apply TIM to a real-world healthcare dataset from the Centers for Disease Control and Prevention (CDC) to estimate the causal effect of high cholesterol on diabetes. Our results demonstrate that TIM improves CATE estimates, increases multivariate overlap, and scales effectively to high-dimensional data, making it a robust tool for causal inference in observational data. 

**Abstract (ZH)**: 基于观察数据因果推断中的匹配旨在构建具有相似协变量分布的处理组和控制组，从而减少混杂因素，确保治疗效果的无偏估计。这种匹配样本类似于随机对照试验（RCT），从而提高了因果估计的质量。我们提出了一种新颖的两阶段可解释匹配（TIM）框架，以实现透明和可解释的协变量匹配。在第一阶段，我们在所有可用的协变量上进行精确匹配。对于在第一阶段中没有精确匹配的处理和控制单元，我们进入第二阶段。在第二阶段，我们通过在每次迭代中移除最不显著的混杂因素并尝试在剩余协变量上进行精确匹配来逐步改进匹配过程。我们为删除的协变量学习一个距离度量，以量化匹配单元在相应分层中的接近程度。我们使用这些高质量匹配来估计条件平均处理效果（CATE）。为了验证TIM的有效性，我们在具有不同关联结构和相关性的合成数据集上进行了实验，并通过测量CATE估计偏倚和评估匹配前后处理组和控制组的多变量重叠来评估其性能。此外，我们将TIM应用于美国疾病控制与预防中心（CDC）的实时医疗数据集，以估计高胆固醇对糖尿病的因果效应。我们的结果表明，TIM可以改进CATE估计，增加多变量重叠，并有效处理高维数据，使其成为观察数据中因果推断的一个稳健工具。 

---
# Understanding LLM Behaviors via Compression: Data Generation, Knowledge Acquisition and Scaling Laws 

**Title (ZH)**: 通过压缩理解大规模语言模型的行为：数据生成、知识获取与扩展规律 

**Authors**: Zhixuan Pan, Shaowen Wang, Jian Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.09597)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across numerous tasks, yet principled explanations for their underlying mechanisms and several phenomena, such as scaling laws, hallucinations, and related behaviors, remain elusive. In this work, we revisit the classical relationship between compression and prediction, grounded in Kolmogorov complexity and Shannon information theory, to provide deeper insights into LLM behaviors. By leveraging the Kolmogorov Structure Function and interpreting LLM compression as a two-part coding process, we offer a detailed view of how LLMs acquire and store information across increasing model and data scales -- from pervasive syntactic patterns to progressively rarer knowledge elements. Motivated by this theoretical perspective and natural assumptions inspired by Heap's and Zipf's laws, we introduce a simplified yet representative hierarchical data-generation framework called the Syntax-Knowledge model. Under the Bayesian setting, we show that prediction and compression within this model naturally lead to diverse learning and scaling behaviors of LLMs. In particular, our theoretical analysis offers intuitive and principled explanations for both data and model scaling laws, the dynamics of knowledge acquisition during training and fine-tuning, factual knowledge hallucinations in LLMs. The experimental results validate our theoretical predictions. 

**Abstract (ZH)**: 大型语言模型（LLMs）在众多任务中展现出了显著的能力，但对其底层机制和一些现象，如缩放定律、幻觉及其相关行为的原理性解释仍不清楚。在这项工作中，我们重新审视了压缩与预测之间的经典关系，基于科莫洛夫复杂性和香农信息理论，以提供对LLM行为更深入的理解。通过利用科莫洛夫结构函数并将LLM压缩视为两部分编码过程，我们提供了从广泛存在的句法模式到逐渐稀有的知识元素是如何在不断增加的模型和数据规模中获取和存储信息的详细视图。基于这一理论视角和由Heap定律和Zipf定律启发的自然假设，我们 introduce 了一种简化但具有代表性的分层数据生成框架，称为句法-知识模型。在贝叶斯设置下，我们展示了该模型中的预测和压缩自然地导致了LLMs的多样化学习和缩放行为。特别是，我们的理论分析为数据和模型缩放定律、训练和微调过程中知识获取的动态变化以及LLMs中的事实知识幻觉提供了直观且原理性的解释。实验结果验证了我们的理论预测。 

---
# Efficient LLM Serving on Hybrid Real-time and Best-effort Requests 

**Title (ZH)**: 混合实时与尽力而为请求的高效大模型服务 

**Authors**: Wan Borui, Zhao Juntao, Jiang Chenyu, Guo Chuanxiong, Wu Chuan  

**Link**: [PDF](https://arxiv.org/pdf/2504.09590)  

**Abstract**: Recent breakthroughs in large Language Models (LLMs) have enabled various generative tasks on a single model. Real-world services (e.g., OpenAI's ChatGPT [27]) powered by an LLM often concurrently support latency-critical requests for interactive applications (e.g., question-answering systems, referred to as real-time or RT requests) and throughput-oriented requests for back-of-house processing (e.g., documents batch processing [28], referred to best-effort or BE requests), with complex hybrid inference workloads to the underlying model. State-of-the-art (SOTA) LLM serving systems dedicate machines to each type of request, towards either low inference latency or high serving throughput, respectively. This practice simplifies request scheduling and management but suffers from poor resource utilization. We propose BROS, a hybrid LLM serving system that aims to collocate RT/BE requests, meeting RT requests' latency requirements while maintaining BE requests' throughput. BROS formulates the problem of hybrid RT/BE request scheduling and solves it with a dynamic priority-based algorithm. BROS designs a bidirectional KV cache management mechanism, allowing RT requests to share KV memory with BE requests to remove the scheduling restrictions caused by insufficient KV memory and improve utilization. Extensive experiments validate that BROS achieves a good trade-off when serving hybrid RT and BE requests. It significantly reduces the latency of RT requests (up to 74.20%), improving their fine-grained service level objectives (SLOs) attainments (up to 36.38x), with negligible throughput reduction for BE requests, showing significant advantages over SOTA systems like vLLM and TGI. 

**Abstract (ZH)**: Recent突破在大规模语言模型（LLMs）上使得单个模型能够执行多种生成任务。由LLM驱动的实际服务（例如OpenAI的ChatGPT [27]）通常同时支持具有低延迟要求的互动应用程序请求（例如问答系统，称为实时或RT请求）和侧重于后端处理的高吞吐量请求（例如文档批量处理 [28]，称为尽力而为或BE请求），这些请求给底层模型带来了复杂的混合推理 workload。当前的最优（SOTA）LLM服务系统分别为这两类请求分配专用机器，分别追求低推理延迟或高服务吞吐量。这一做法虽然简化了请求调度和管理，但导致资源利用率低。我们提出了一种名为BROS的混合LLM服务系统，旨在将RT/BE请求并置，以满足RT请求的延迟要求同时保持BE请求的吞吐量。BROS将混合RT/BE请求调度问题形式化，并利用动态优先级算法解决该问题。BROS设计了一种双向KV缓存管理机制，允许RT请求与BE请求共享KV内存，从而消除由于KV内存不足导致的调度限制，并提高资源利用率。广泛实验验证了BROS在服务混合RT和BE请求时实现了良好的权衡，显著降低了RT请求的延迟（最多74.20%），提高了其精细粒度的服务水平目标（SLOs）达成率（最多36.38倍），对BE请求的吞吐量影响几乎可以忽略，显示出与当前最优系统（如vLLM和TGI）相比的显著优势。 

---
# Reduction of Supervision for Biomedical Knowledge Discovery 

**Title (ZH)**: 生物医学知识发现中的监督减少 

**Authors**: Christos Theodoropoulos, Andrei Catalin Coman, James Henderson, Marie-Francine Moens  

**Link**: [PDF](https://arxiv.org/pdf/2504.09582)  

**Abstract**: Knowledge discovery is hindered by the increasing volume of publications and the scarcity of extensive annotated data. To tackle the challenge of information overload, it is essential to employ automated methods for knowledge extraction and processing. Finding the right balance between the level of supervision and the effectiveness of models poses a significant challenge. While supervised techniques generally result in better performance, they have the major drawback of demanding labeled data. This requirement is labor-intensive and time-consuming and hinders scalability when exploring new domains. In this context, our study addresses the challenge of identifying semantic relationships between biomedical entities (e.g., diseases, proteins) in unstructured text while minimizing dependency on supervision. We introduce a suite of unsupervised algorithms based on dependency trees and attention mechanisms and employ a range of pointwise binary classification methods. Transitioning from weakly supervised to fully unsupervised settings, we assess the methods' ability to learn from data with noisy labels. The evaluation on biomedical benchmark datasets explores the effectiveness of the methods. Our approach tackles a central issue in knowledge discovery: balancing performance with minimal supervision. By gradually decreasing supervision, we assess the robustness of pointwise binary classification techniques in handling noisy labels, revealing their capability to shift from weakly supervised to entirely unsupervised scenarios. Comprehensive benchmarking offers insights into the effectiveness of these techniques, suggesting an encouraging direction toward adaptable knowledge discovery systems, representing progress in creating data-efficient methodologies for extracting useful insights when annotated data is limited. 

**Abstract (ZH)**: 知识发现受出版物数量增加和标注数据稀缺性的阻碍。为应对信息过载的挑战，有必要采用自动化方法进行知识提取和处理。在监督程度和模型有效性之间找到合适的平衡是一项重大挑战。虽然监督技术通常能获得更好的性能，但它们的主要缺点是需要标注数据，这既耗时又费力，当探索新领域时会阻碍可扩展性。在此背景下，我们的研究旨在通过最小化对监督的依赖来识别生物医学实体（如疾病、蛋白质）之间的语义关系。我们引入了一套基于依存树和注意力机制的无监督算法，并采用了一系列点-wise二分类方法。从弱监督过渡到完全无监督设置，我们评估了这些方法从噪声标签数据中学习的能力。在生物医学基准数据集上的评估探索了这些方法的有效性。我们的方法解决了知识发现中的一个核心问题：在最小监督条件下平衡性能。通过逐渐减少监督，我们评估了点-wise二分类技术在处理噪声标签数据方面的鲁棒性，揭示了它们能够从弱监督场景过渡到完全无监督场景的能力。全面的基准测试为这些技术的有效性提供了见解，表明在标注数据有限的情况下开发高效数据提取系统的前景是令人鼓舞的。 

---
# Improved FOX Optimization Algorithm 

**Title (ZH)**: 改进的FOX优化算法 

**Authors**: Mahmood A. Jumaah, Yossra H. Ali, Tarik A. Rashid  

**Link**: [PDF](https://arxiv.org/pdf/2504.09574)  

**Abstract**: Optimization algorithms are essential for solving many real-world problems. However, challenges such as premature convergence to local optima and the difficulty of effectively balancing exploration and exploitation often hinder their performance. To address these issues, this paper proposes an improved FOX optimization algorithm, Improved FOX (IFOX). The IFOX algorithm introduces a new adaptive mechanism for balancing exploration and exploitation based on fitness values. It also reduces the number of hyperparameters and simplifies the core equations of the original FOX. To evaluate its effectiveness, IFOX has been tested on classical uni-modal and multi-modal benchmark functions, as well as on benchmark sets from the Congress on Evolutionary Computation (CEC), in addition to two engineering design problems: Pressure Vessel Design and Economic Load Dispatch. The results show that IFOX outperforms existing optimization algorithms, achieving superior results on 51 benchmark functions. These findings underscore the strong potential of IFOX as a competitive and robust optimization algorithm for a wide range of applications. 

**Abstract (ZH)**: 改进的FOX优化算法：IFOX及其在多种优化问题中的应用 

---
# Draw with Thought: Unleashing Multimodal Reasoning for Scientific Diagram Generation 

**Title (ZH)**: 思绘图启智：激发多模态推理以促进科学图表生成 

**Authors**: Zhiqing Cui, Jiahao Yuan, Hanqing Wang, Yanshu Li, Chenxu Du, Zhenglong Ding  

**Link**: [PDF](https://arxiv.org/pdf/2504.09479)  

**Abstract**: Scientific diagrams are vital tools for communicating structured knowledge across disciplines. However, they are often published as static raster images, losing symbolic semantics and limiting reuse. While Multimodal Large Language Models (MLLMs) offer a pathway to bridging vision and structure, existing methods lack semantic control and structural interpretability, especially on complex diagrams. We propose Draw with Thought (DwT), a training-free framework that guides MLLMs to reconstruct diagrams into editable mxGraph XML code through cognitively-grounded Chain-of-Thought reasoning. DwT enables interpretable and controllable outputs without model fine-tuning by dividing the task into two stages: Coarse-to-Fine Planning, which handles perceptual structuring and semantic specification, and Structure-Aware Code Generation, enhanced by format-guided refinement. To support evaluation, we release Plot2XML, a benchmark of 247 real-world scientific diagrams with gold-standard XML annotations. Extensive experiments across eight MLLMs show that our approach yields high-fidelity, semantically aligned, and structurally valid reconstructions, with human evaluations confirming strong alignment in both accuracy and visual aesthetics, offering a scalable solution for converting static visuals into executable representations and advancing machine understanding of scientific graphics. 

**Abstract (ZH)**: 科学图表是跨学科传达结构化知识的重要工具。然而，它们通常以静态位图形式发表，丧失了符号意义并限制了再利用。尽管多模态大型语言模型（MLLMs）提供了连接视觉和结构的途径，但现有方法缺乏语义控制和结构可解释性，尤其是在复杂图表方面的表现尤为不足。我们提出了“思维驱动绘图”（DwT）框架，这是一种无需微调的框架，可以引导MLLMs通过基于认知推理的思想链推理将图表重建为可编辑的mxGraph XML代码。DwT通过将任务分为两个阶段——粗到细规划和结构感知代码生成——实现了可解释和可控的输出，无需模型微调。粗到细规划处理感知结构化和语义规范，结构感知代码生成则通过格式引导进一步优化。为了支持评估，我们发布了包含247个真实世界科学图表的Plot2XML基准数据集，并配以黄金标准的XML注释。跨八种MLLM的广泛实验表明，我们提出的方法生成了高保真度、语义对齐且结构有效的重建结果，人工评估证实其在准确性和视觉美学方面具有强大的对齐性，为将静态视觉转换为可执行表示以及推进对科学图形的机器理解提供了可扩展的解决方案。 

---
# Don't Deceive Me: Mitigating Gaslighting through Attention Reallocation in LMMs 

**Title (ZH)**: 不要欺骗我：通过注意力重新分配减轻LMM中的精神操控 

**Authors**: Pengkun Jiao, Bin Zhu, Jingjing Chen, Chong-Wah Ngo, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09456)  

**Abstract**: Large Multimodal Models (LMMs) have demonstrated remarkable capabilities across a wide range of tasks. However, their vulnerability to user gaslighting-the deliberate use of misleading or contradictory inputs-raises critical concerns about their reliability in real-world applications. In this paper, we address the novel and challenging issue of mitigating the negative impact of negation-based gaslighting on LMMs, where deceptive user statements lead to significant drops in model accuracy. Specifically, we introduce GasEraser, a training-free approach that reallocates attention weights from misleading textual tokens to semantically salient visual regions. By suppressing the influence of "attention sink" tokens and enhancing focus on visually grounded cues, GasEraser significantly improves LMM robustness without requiring retraining or additional supervision. Extensive experimental results demonstrate that GasEraser is effective across several leading open-source LMMs on the GaslightingBench. Notably, for LLaVA-v1.5-7B, GasEraser reduces the misguidance rate by 48.2%, demonstrating its potential for more trustworthy LMMs. 

**Abstract (ZH)**: Large多模态模型在广泛任务中展现了显著的能力。然而，它们在用户操纵（Gaslighting）下的脆弱性引发了对其实际应用中可靠性的严重关切。在本文中，我们针对基于否定的用户操纵对大型多模态模型（LMMs）造成负面影响这一新颖而具有挑战性的问题，提出了缓解措施。具体而言，我们引入了GasEraser，这是一种无需训练的方法，通过重新分配误导性文本令牌的关注权重至语义上相关的视觉区域。通过抑制“注意力陷阱”令牌的影响并增强对视觉支持线索的关注，GasEraser大幅度提升了LMM的鲁棒性，而无需重新训练或额外监督。广泛的实验结果表明，GasEraser在GaslightingBench上的多个领先开源LMM中均有效。特别是对于LLaVA-v1.5-7B，GasEraser将误导率降低了48.2%，展示了其增强可信的LMM的潜力。 

---
# Enhancing Mathematical Reasoning in Large Language Models with Self-Consistency-Based Hallucination Detection 

**Title (ZH)**: 基于自我一致性幻觉检测增强大型语言模型的数学推理能力 

**Authors**: MingShan Liu, Shi Bo, Jialing Fang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09440)  

**Abstract**: Large language models (LLMs) have demonstrated strong mathematical reasoning capabilities but remain susceptible to hallucinations producing plausible yet incorrect statements especially in theorem proving, symbolic manipulation, and numerical computation. While self-consistency (SC) has been explored as a means to improve factuality in LLMs, existing approaches primarily apply SC to final-answer selection, neglecting the logical consistency of intermediate reasoning steps. In this work, we introduce a structured self-consistency framework designed to enhance the reliability of mathematical reasoning. Our method enforces self-consistency across intermediate steps and final outputs, reducing logical inconsistencies and hallucinations. We evaluate our approach across three core mathematical tasks: theorem proving, symbolic transformation, and numerical computation. Experimental results demonstrate that SC significantly improves proof validity, symbolic reasoning accuracy, and numerical stability while maintaining computational efficiency. Further analysis reveals that structured self-consistency not only enhances problem-solving accuracy but also reduces the variance of model-generated outputs. These findings highlight self-consistency as a robust mechanism for improving mathematical reasoning in LLMs, paving the way for more reliable and interpretable AI-driven mathematics. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在数学推理方面展现出了强大的能力，但在定理证明、符号操作和数值计算等方面仍然容易产生幻觉，即产出看似合理但实际上错误的陈述。虽然自一致性（SC）已经被探索作为一种提高LLMs事实准确性的方法，但现有的方法主要集中在最终答案的选择上，忽视了中间推理步骤的逻辑一致性。在本文中，我们引入了一种结构化的自一致性框架，旨在增强数学推理的可靠性。该方法强制执行中间步骤和最终输出的一致性，减少了逻辑不一致性和幻觉。我们通过三个核心数学任务——定理证明、符号转换和数值计算——来评估我们的方法。实验结果表明，SC显著提高了证明的有效性、符号推理的准确性以及数值稳定性，同时保持了计算效率。进一步的分析表明，结构化的自一致性不仅提高了问题解决的准确性，还减少了模型生成输出的差异性。这些发现突显了自一致性作为增强LLMs数学推理的稳健机制的作用，为更可靠和可解释的人工智能驱动数学奠定了基础。 

---
# Application of Contrastive Learning on ECG Data: Evaluating Performance in Japanese and Classification with Around 100 Labels 

**Title (ZH)**: 基于对比学习的心电图数据应用：在日本人群中的性能评估与大约100个类别的分类 

**Authors**: Junichiro Takahashi, JingChuan Guan, Masataka Sato, Kaito Baba, Kazuto Haruguchi, Daichi Nagashima, Satoshi Kodera, Norihiko Takeda  

**Link**: [PDF](https://arxiv.org/pdf/2504.09302)  

**Abstract**: The electrocardiogram (ECG) is a fundamental tool in cardiovascular diagnostics due to its powerful and non-invasive nature. One of the most critical usages is to determine whether more detailed examinations are necessary, with users ranging across various levels of expertise. Given this diversity in expertise, it is essential to assist users to avoid critical errors. Recent studies in machine learning have addressed this challenge by extracting valuable information from ECG data. Utilizing language models, these studies have implemented multimodal models aimed at classifying ECGs according to labeled terms. However, the number of classes was reduced, and it remains uncertain whether the technique is effective for languages other than English. To move towards practical application, we utilized ECG data from regular patients visiting hospitals in Japan, maintaining a large number of Japanese labels obtained from actual ECG readings. Using a contrastive learning framework, we found that even with 98 labels for classification, our Japanese-based language model achieves accuracy comparable to previous research. This study extends the applicability of multimodal machine learning frameworks to broader clinical studies and non-English languages. 

**Abstract (ZH)**: 心电图（ECG）是心血管诊断中的一个基本工具，由于其强大且非侵入性的性质。其最关键的应用之一是确定是否需要进行更详细的检查，使用者涵盖了不同程度的专业水平。鉴于这种专业水平的多样性，协助用户避免关键错误是必不可少的。最近的机器学习研究通过从ECG数据中提取有价值的信息来应对这一挑战。利用语言模型，这些研究实现了多模态模型，用于根据标注术语对ECG进行分类。然而，分类的类别数量被减少，并且尚不确定该技术是否对除英语以外的其他语言有效。为了向实际应用迈进，我们利用了日本医院普通患者的心电图数据，保持了大量的实际心电图读数获得的日本标签。通过对比学习框架，我们发现即使有98个分类标签，我们基于日语的语言模型也能达到与之前研究相当的精度。这项研究扩展了多模态机器学习框架在更广泛的临床研究和非英语语言中的适用性。 

---
# Continuum-Interaction-Driven Intelligence: Human-Aligned Neural Architecture via Crystallized Reasoning and Fluid Generation 

**Title (ZH)**: 连续交互驱动的智能：通过结晶推理和流转型生成实现的人类对齐神经架构 

**Authors**: Pengcheng Zhou, Zhiqiang Nie, Haochen Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.09301)  

**Abstract**: Current AI systems based on probabilistic neural networks, such as large language models (LLMs), have demonstrated remarkable generative capabilities yet face critical challenges including hallucination, unpredictability, and misalignment with human decision-making. These issues fundamentally stem from the over-reliance on randomized (probabilistic) neural networks-oversimplified models of biological neural networks-while neglecting the role of procedural reasoning (chain-of-thought) in trustworthy decision-making. Inspired by the human cognitive duality of fluid intelligence (flexible generation) and crystallized intelligence (structured knowledge), this study proposes a dual-channel intelligent architecture that integrates probabilistic generation (LLMs) with white-box procedural reasoning (chain-of-thought) to construct interpretable, continuously learnable, and human-aligned AI systems. Concretely, this work: (1) redefines chain-of-thought as a programmable crystallized intelligence carrier, enabling dynamic knowledge evolution and decision verification through multi-turn interaction frameworks; (2) introduces a task-driven modular network design that explicitly demarcates the functional boundaries between randomized generation and procedural control to address trustworthiness in vertical-domain applications; (3) demonstrates that multi-turn interaction is a necessary condition for intelligence emergence, with dialogue depth positively correlating with the system's human-alignment degree. This research not only establishes a new paradigm for trustworthy AI deployment but also provides theoretical foundations for next-generation human-AI collaborative systems. 

**Abstract (ZH)**: 基于概率神经网络的当前人工智能系统，如大型语言模型（LLMs），展现了出色的生成能力，但仍面临幻觉、不可预测性和与人类决策不一致等关键挑战。这些问题从根本上来说源于过度依赖随机化（概率性）神经网络——对生物神经网络的简化模型——而忽视了程序化推理（链式思维）在可信决策中的作用。受人类认知双重性的启发，即流体智力（灵活生成）和晶体智力（结构化知识），本研究提出了一种双重通道智能架构，将概率生成（LLMs）与白盒程序化推理（链式思维）相结合，构建可解释、可持续学习且与人类目标一致的AI系统。具体而言，本研究：（1）将链式思维重新定义为可编程的晶体智力载体，通过多轮交互框架实现动态知识进化和决策验证；（2）引入了一种任务驱动的模块化网络设计，明确区分随机生成和程序控制的功能边界，以解决垂直应用领域的可信性问题；（3）证明多轮交互是智能涌现的必要条件，对话深度与系统的拟人化程度正相关。本研究不仅为可信AI部署奠定了新的范式，也为新一代人机协作系统提供了理论基础。 

---
# Graph Learning-Driven Multi-Vessel Association: Fusing Multimodal Data for Maritime Intelligence 

**Title (ZH)**: 基于图学习的多血管关联：融合多模态数据的 maritime 智能 

**Authors**: Yuxu Lu, Kaisen Yang, Dong Yang, Haifeng Ding, Jinxian Weng, Ryan Wen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.09197)  

**Abstract**: Ensuring maritime safety and optimizing traffic management in increasingly crowded and complex waterways require effective waterway monitoring. However, current methods struggle with challenges arising from multimodal data, such as dimensional disparities, mismatched target counts, vessel scale variations, occlusions, and asynchronous data streams from systems like the automatic identification system (AIS) and closed-circuit television (CCTV). Traditional multi-target association methods often struggle with these complexities, particularly in densely trafficked waterways. To overcome these issues, we propose a graph learning-driven multi-vessel association (GMvA) method tailored for maritime multimodal data fusion. By integrating AIS and CCTV data, GMvA leverages time series learning and graph neural networks to capture the spatiotemporal features of vessel trajectories effectively. To enhance feature representation, the proposed method incorporates temporal graph attention and spatiotemporal attention, effectively capturing both local and global vessel interactions. Furthermore, a multi-layer perceptron-based uncertainty fusion module computes robust similarity scores, and the Hungarian algorithm is adopted to ensure globally consistent and accurate target matching. Extensive experiments on real-world maritime datasets confirm that GMvA delivers superior accuracy and robustness in multi-target association, outperforming existing methods even in challenging scenarios with high vessel density and incomplete or unevenly distributed AIS and CCTV data. 

**Abstract (ZH)**: 确保 maritime 安全和优化日益拥挤复杂的水道交通管理需要有效的水道监控。为了应对多模态数据带来的挑战，如维度差异、目标计数不匹配、船只规模变化、遮挡以及来自自动识别系统（AIS）和闭路电视（CCTV）等系统的异步数据流，传统多目标关联方法在稠密交通水道中常常难以应对这些复杂性。为此，我们提出了一种基于图学习的多船只关联（GMvA）方法，专门用于海洋多模态数据融合。通过整合AIS和CCTV数据，GMvA利用时序学习和图神经网络来有效捕捉船只轨迹的空间时间特征。为进一步增强特征表示，该方法引入了时间图注意力和时空注意力，有效捕捉局部和全局船只交互。此外，基于多层感知器的不确定性融合模块计算稳健的相似度分数，并采用哈特尔顿算法以确保全局一致且准确的目标匹配。实验证明，GMvA在多目标关联中表现出更优的准确性和鲁棒性，即使在高船只密度和AIS及CCTV数据分布不均的具有挑战性场景中也能超越现有方法。 

---
# A Short Survey on Small Reasoning Models: Training, Inference, Applications and Research Directions 

**Title (ZH)**: 小型推理模型综述：训练、推理、应用与研究方向 

**Authors**: Chengyu Wang, Taolin Zhang, Richang Hong, Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09100)  

**Abstract**: Recently, the reasoning capabilities of large reasoning models (LRMs), such as DeepSeek-R1, have seen significant advancements through the slow thinking process. Despite these achievements, the substantial computational demands of LRMs present considerable challenges. In contrast, small reasoning models (SRMs), often distilled from larger ones, offer greater efficiency and can exhibit distinct capabilities and cognitive trajectories compared to LRMs. This work surveys around 170 recently published papers on SRMs for tackling various complex reasoning tasks. We review the current landscape of SRMs and analyze diverse training and inference techniques related to SRMs. Furthermore, we provide a comprehensive review of SRMs for domain-specific applications and discuss possible future research directions. This survey serves as an essential reference for researchers to leverage or develop SRMs for advanced reasoning functionalities with high efficiency. 

**Abstract (ZH)**: 最近，大型推理模型（LRMs）如DeepSeek-R1的推理能力通过慢思考过程取得了显著进步。尽管取得了这些成就，LRMs的大量计算需求仍提出了重大挑战。相比之下，从小型模型中提炼出来的小型推理模型（SRMs）提供了更高的效率，并在能力和认知轨迹方面表现出与LRMs不同的特点。本文回顾了约170篇近期发表的关于SRMs在各种复杂推理任务中的应用的论文，概述了当前SRMs的格局，并分析了与SRMs相关的多种训练和推理技术。此外，本文对SRMs在特定领域应用进行了全面回顾，并讨论了未来的研究方向。本综述为研究人员利用或开发具有高效性的SRMs以实现高级推理功能提供了重要的参考。 

---
# Towards Stepwise Domain Knowledge-Driven Reasoning Optimization and Reflection Improvement 

**Title (ZH)**: 逐步领域知识驱动的推理优化与反思提升 

**Authors**: Chengyuan Liu, Shihang Wang, Lizhi Qing, Kaisong Song, Junjie Cao, Jun Lin, Ji Zhang, Ang Li, Kun Kuang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.09058)  

**Abstract**: Recently, stepwise supervision on Chain of Thoughts (CoTs) presents an enhancement on the logical reasoning tasks such as coding and math, with the help of Monte Carlo Tree Search (MCTS). However, its contribution to tasks requiring domain-specific expertise and knowledge remains unexplored. Motivated by the interest, we identify several potential challenges of vanilla MCTS within this context, and propose the framework of Stepwise Domain Knowledge-Driven Reasoning Optimization, employing the MCTS algorithm to develop step-level supervision for problems that require essential comprehension, reasoning, and specialized knowledge. Additionally, we also introduce the Preference Optimization towards Reflection Paths, which iteratively learns self-reflection on the reasoning thoughts from better perspectives. We have conducted extensive experiments to evaluate the advantage of the methodologies. Empirical results demonstrate the effectiveness on various legal-domain problems. We also report a diverse set of valuable findings, hoping to encourage the enthusiasm to the research of domain-specific LLMs and MCTS. 

**Abstract (ZH)**: 基于蒙特卡洛树搜索的逐步领域知识驱动推理优化及其偏好优化研究 

---
# An Enhanced Iterative Deepening Search Algorithm for the Unrestricted Container Rehandling Problem 

**Title (ZH)**: 改进的迭代加深搜索算法用于解决无约束集装箱回取问题 

**Authors**: Ruoqi Wang, Jiawei Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.09046)  

**Abstract**: In container terminal yards, the Container Rehandling Problem (CRP) involves rearranging containers between stacks under specific operational rules, and it is a pivotal optimization challenge in intelligent container scheduling systems. Existing CRP studies primarily focus on minimizing reallocation costs using two-dimensional bay structures, considering factors such as container size, weight, arrival sequences, and retrieval priorities. This paper introduces an enhanced deepening search algorithm integrated with improved lower bounds to boost search efficiency. To further reduce the search space, we design mutually consistent pruning rules to avoid excessive computational overhead. The proposed algorithm is validated on three widely used benchmark datasets for the Unrestricted Container Rehandling Problem (UCRP). Experimental results demonstrate that our approach outperforms state-of-the-art exact algorithms in solving the more general UCRP variant, particularly exhibiting superior efficiency when handling containers within the same priority group under strict time constraints. 

**Abstract (ZH)**: 集装箱 terminal堆场中的集装箱重新处理问题（CRP）涉及在特定操作规则下在堆栈之间重新安排集装箱，这是智能集装箱调度系统中的关键优化挑战。现有CRP研究主要集中在使用二维bay结构最小化重新分配成本，考虑因素包括集装箱尺寸、重量、到达顺序和取回优先级。本文提出了一种增强的逐步搜索算法，并结合改进的下界方法以提升搜索效率。为了进一步减小搜索空间，设计了一套互斥的剪枝规则以避免过多的计算开销。所提出的方法在Unrestricted Container Rehandling Problem (UCRP)的三个广泛使用的基准数据集上进行了验证。实验结果表明，在解决更通用的UCRP变种时，我们的方法超越了现有最先进的精确算法，特别是在严格时间约束下处理相同优先级组内的集装箱时表现出更优的效率。 

---
# A Survey of Frontiers in LLM Reasoning: Inference Scaling, Learning to Reason, and Agentic Systems 

**Title (ZH)**: LLM推理前沿综述：推理扩展、学习推理与代理系统 

**Authors**: Zixuan Ke, Fangkai Jiao, Yifei Ming, Xuan-Phi Nguyen, Austin Xu, Do Xuan Long, Minzhi Li, Chengwei Qin, Peifeng Wang, Silvio Savarese, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2504.09037)  

**Abstract**: Reasoning is a fundamental cognitive process that enables logical inference, problem-solving, and decision-making. With the rapid advancement of large language models (LLMs), reasoning has emerged as a key capability that distinguishes advanced AI systems from conventional models that empower chatbots. In this survey, we categorize existing methods along two orthogonal dimensions: (1) Regimes, which define the stage at which reasoning is achieved (either at inference time or through dedicated training); and (2) Architectures, which determine the components involved in the reasoning process, distinguishing between standalone LLMs and agentic compound systems that incorporate external tools, and multi-agent collaborations. Within each dimension, we analyze two key perspectives: (1) Input level, which focuses on techniques that construct high-quality prompts that the LLM condition on; and (2) Output level, which methods that refine multiple sampled candidates to enhance reasoning quality. This categorization provides a systematic understanding of the evolving landscape of LLM reasoning, highlighting emerging trends such as the shift from inference-scaling to learning-to-reason (e.g., DeepSeek-R1), and the transition to agentic workflows (e.g., OpenAI Deep Research, Manus Agent). Additionally, we cover a broad spectrum of learning algorithms, from supervised fine-tuning to reinforcement learning such as PPO and GRPO, and the training of reasoners and verifiers. We also examine key designs of agentic workflows, from established patterns like generator-evaluator and LLM debate to recent innovations. ... 

**Abstract (ZH)**: 大型语言模型的推理是认知过程的核心，它支持逻辑推理、问题解决和决策制定。随着大型语言模型（LLMs）的迅速发展，推理已成为高级AI系统与传统聊天机器人所依赖的常规模型区分开来的关键技术能力。本文综述中，我们将现有的方法按照两个正交维度进行分类：（1）阶段，定义推理是在推断时实现还是通过专门训练获得；（2）架构，确定推理过程中涉及的组件，区分独立的大型语言模型和整合外部工具及多代理协作的自主系统。在每个维度中，我们分析了两个关键视角：（1）输入层面，关注构建高质量提示的技术，这些提示使LLM能够理解；（2）输出层面，关注方法如何细化多个采样候选方案以提高推理质量。这种分类提供了对大型语言模型推理演进格局的系统理解，突显出从推理扩展到学习推理（如DeepSeek-R1）和从非自主型工作流程到自主型工作流程转变等新兴趋势。此外，本文还涵盖了从监督微调到强化学习（如PPO和GRPO）等各种学习算法，以及推理者和验证者的训练。我们还考察了自主工作流程的关键设计，从生成器-评估器模式和LLM辩论到最近的创新。 

---
# Mixed Signals: Decoding VLMs' Reasoning and Underlying Bias in Vision-Language Conflict 

**Title (ZH)**: 混合信号：解码VLMs在视觉-语言冲突中的推理和潜在偏见 

**Authors**: Pouya Pezeshkpour, Moin Aminnaseri, Estevam Hruschka  

**Link**: [PDF](https://arxiv.org/pdf/2504.08974)  

**Abstract**: Vision-language models (VLMs) have demonstrated impressive performance by effectively integrating visual and textual information to solve complex tasks. However, it is not clear how these models reason over the visual and textual data together, nor how the flow of information between modalities is structured. In this paper, we examine how VLMs reason by analyzing their biases when confronted with scenarios that present conflicting image and text cues, a common occurrence in real-world applications. To uncover the extent and nature of these biases, we build upon existing benchmarks to create five datasets containing mismatched image-text pairs, covering topics in mathematics, science, and visual descriptions. Our analysis shows that VLMs favor text in simpler queries but shift toward images as query complexity increases. This bias correlates with model scale, with the difference between the percentage of image- and text-preferred responses ranging from +56.8% (image favored) to -74.4% (text favored), depending on the task and model. In addition, we explore three mitigation strategies: simple prompt modifications, modifications that explicitly instruct models on how to handle conflicting information (akin to chain-of-thought prompting), and a task decomposition strategy that analyzes each modality separately before combining their results. Our findings indicate that the effectiveness of these strategies in identifying and mitigating bias varies significantly and is closely linked to the model's overall performance on the task and the specific modality in question. 

**Abstract (ZH)**: Vision-语言模型通过有效整合视觉和文本信息来解决复杂任务方面已展现出 impressive 的性能。然而，尚不明确这些模型如何共同推理视觉和文本数据，也不清楚不同模态间信息流的结构如何。在这篇论文中，我们通过分析模型在遇到提供冲突图像和文本线索的场景时的偏见，来考察模型的推理过程，此类场景在实际应用中很常见。为了揭示这些偏见的范围和性质，我们在现有基准的基础上创建了五个包含不匹配图像-文本配对的数据集，覆盖数学、科学和视觉描述等领域。我们的分析表明，对于简单的查询，模型倾向于文本，但随着查询复杂性的增加，模型更倾向于图像。这种偏见与模型规模相关，不同任务和模型之间，偏好图像或文本的响应比例差异从+56.8%（偏好图像）到-74.4%（偏好文本）不等。此外，我们还探索了三种缓解策略：简单的提示修改、明确指示模型如何处理冲突信息的修改（类似于思维链提示）、以及一种任务分解策略，即单独分析每个模态后再结合其结果。我们的研究发现表明，这些策略在识别和缓解偏见方面的有效性显著不同，并且与模型在特定任务上的整体表现以及所涉及的具体模态密切相关。 

---
# Hybrid AI-Physical Modeling for Penetration Bias Correction in X-band InSAR DEMs: A Greenland Case Study 

**Title (ZH)**: 基于X波段InSAR DEMs的穿透偏差校正的混合AI-物理建模：以格陵兰为例的研究 

**Authors**: Islam Mansour, Georg Fischer, Ronny Haensch, Irena Hajnsek  

**Link**: [PDF](https://arxiv.org/pdf/2504.08909)  

**Abstract**: Digital elevation models derived from Interferometric Synthetic Aperture Radar (InSAR) data over glacial and snow-covered regions often exhibit systematic elevation errors, commonly termed "penetration bias." We leverage existing physics-based models and propose an integrated correction framework that combines parametric physical modeling with machine learning. We evaluate the approach across three distinct training scenarios - each defined by a different set of acquisition parameters - to assess overall performance and the model's ability to generalize. Our experiments on Greenland's ice sheet using TanDEM-X data show that the proposed hybrid model corrections significantly reduce the mean and standard deviation of DEM errors compared to a purely physical modeling baseline. The hybrid framework also achieves significantly improved generalization than a pure ML approach when trained on data with limited diversity in acquisition parameters. 

**Abstract (ZH)**: 基于干涉雷达合成孔径雷达（InSAR）数据的冰川和雪盖区域数字 elevation 模型衍生出的系统性高程误差（称为“穿透偏差”）的物理模型与机器学习集成校正框架 

---
# Endowing Embodied Agents with Spatial Reasoning Capabilities for Vision-and-Language Navigation 

**Title (ZH)**: 赋予具身代理空间推理能力以实现视觉-语言导航 

**Authors**: Luo Ling, Bai Qianqian  

**Link**: [PDF](https://arxiv.org/pdf/2504.08806)  

**Abstract**: Enhancing the spatial perception capabilities of mobile robots is crucial for achieving embodied Vision-and-Language Navigation (VLN). Although significant progress has been made in simulated environments, directly transferring these capabilities to real-world scenarios often results in severe hallucination phenomena, causing robots to lose effective spatial awareness. To address this issue, we propose BrainNav, a bio-inspired spatial cognitive navigation framework inspired by biological spatial cognition theories and cognitive map theory. BrainNav integrates dual-map (coordinate map and topological map) and dual-orientation (relative orientation and absolute orientation) strategies, enabling real-time navigation through dynamic scene capture and path planning. Its five core modules-Hippocampal Memory Hub, Visual Cortex Perception Engine, Parietal Spatial Constructor, Prefrontal Decision Center, and Cerebellar Motion Execution Unit-mimic biological cognitive functions to reduce spatial hallucinations and enhance adaptability. Validated in a zero-shot real-world lab environment using the Limo Pro robot, BrainNav, compatible with GPT-4, outperforms existing State-of-the-Art (SOTA) Vision-and-Language Navigation in Continuous Environments (VLN-CE) methods without fine-tuning. 

**Abstract (ZH)**: 增强移动机器人在空间感知能力对于实现具身视觉-语言导航（VLN）至关重要。尽管在模拟环境中取得了显著进步，但这些能力直接应用于真实世界场景时，往往会引发严重的幻觉现象，导致机器人失去有效的空间意识。为解决这一问题，我们提出了一种受生物空间认知理论和认知地图理论启发的空间认知导航框架BrainNav。BrainNav 结合了双地图（坐标地图和拓扑地图）和双方向（相对方向和绝对方向）策略，通过动态场景捕捉和路径规划实现实时导航。其五大核心模块—海马记忆中枢、视觉皮层感知引擎、枕叶空间构造器、前额叶决策中心和小脑运动执行单元—模拟生物认知功能，减少空间幻觉并增强适应性。在使用Limo Pro机器人进行的零样本真实世界实验室环境中验证，BrainNav 不需要微调就超过了现有的最先进的连续环境视觉-语言导航（VLN-CE）方法。 

---
# GridMind: A Multi-Agent NLP Framework for Unified, Cross-Modal NFL Data Insights 

**Title (ZH)**: GridMind: 一个统一跨模态NFL数据洞察的多Agent NLP框架 

**Authors**: Jordan Chipka, Chris Moyer, Clay Troyer, Tyler Fuelling, Jeremy Hochstedler  

**Link**: [PDF](https://arxiv.org/pdf/2504.08747)  

**Abstract**: The rapid growth of big data and advancements in computational techniques have significantly transformed sports analytics. However, the diverse range of data sources -- including structured statistics, semi-structured formats like sensor data, and unstructured media such as written articles, audio, and video -- creates substantial challenges in extracting actionable insights. These various formats, often referred to as multimodal data, require integration to fully leverage their potential. Conventional systems, which typically prioritize structured data, face limitations when processing and combining these diverse content types, reducing their effectiveness in real-time sports analysis.
To address these challenges, recent research highlights the importance of multimodal data integration for capturing the complexity of real-world sports environments. Building on this foundation, this paper introduces GridMind, a multi-agent framework that unifies structured, semi-structured, and unstructured data through Retrieval-Augmented Generation (RAG) and large language models (LLMs) to facilitate natural language querying of NFL data. This approach aligns with the evolving field of multimodal representation learning, where unified models are increasingly essential for real-time, cross-modal interactions.
GridMind's distributed architecture includes specialized agents that autonomously manage each stage of a prompt -- from interpretation and data retrieval to response synthesis. This modular design enables flexible, scalable handling of multimodal data, allowing users to pose complex, context-rich questions and receive comprehensive, intuitive responses via a conversational interface. 

**Abstract (ZH)**: 大数据的迅速增长和计算技术的进步显著改变了体育分析领域。然而，包括结构化统计数据、半结构化格式（如传感器数据）和非结构化媒体（如文章、音频和视频）在内的多样化数据源，为提取 actionable 洞察带来了巨大挑战。这些各种格式的数据，通常被称为多模态数据，需要进行集成以充分发挥其潜力。传统的系统通常优先处理结构化数据，在处理和组合这些多样化的数据类型时存在局限性，从而在实时体育分析中的效果受限。

为应对这些挑战，近期研究强调了多模态数据集成的重要性，以捕捉真实世界体育环境的复杂性。在此基础上，本文介绍了一种多Agent框架 GridMind，通过检索增强生成（RAG）和大规模语言模型（LLMs）将结构化、半结构化和非结构化数据统一起来，以促进对 NFL 数据的自然语言查询。这一方法与正在演化的多模态表示学习领域相一致，在该领域中，统一模型对于实时跨模态交互越来越必不可少。

GridMind 的分布式架构包含专门的 Agent，这些 Agent 自动管理提示的每个阶段——从解释和数据检索到响应合成。这种模块化设计使得灵活、可扩展地处理多模态数据成为可能，使用户能够提出复杂的、富含背景的问题，并通过对话式接口接收全面、直观的响应。 

---
# Latency-Aware 2-Opt Monotonic Local Search for Distributed Constraint Optimization 

**Title (ZH)**: 面向延迟的2-Opt单调局部搜索在分布式约束优化中的应用 

**Authors**: Ben Rachmut, Roie Zivan, William Yeoh  

**Link**: [PDF](https://arxiv.org/pdf/2504.08737)  

**Abstract**: Researchers recently extended Distributed Constraint Optimization Problems (DCOPs) to Communication-Aware DCOPs so that they are applicable in scenarios in which messages can be arbitrarily delayed. Distributed asynchronous local search and inference algorithms designed for CA-DCOPs are less vulnerable to message latency than their counterparts for regular DCOPs. However, unlike local search algorithms for (regular) DCOPs that converge to k-opt solutions (with k > 1), that is, they converge to solutions that cannot be improved by a group of k agents), local search CA-DCOP algorithms are limited to 1-opt solutions only. In this paper, we introduce Latency-Aware Monotonic Distributed Local Search-2 (LAMDLS-2), where agents form pairs and coordinate bilateral assignment replacements. LAMDLS-2 is monotonic, converges to a 2-opt solution, and is also robust to message latency, making it suitable for CA-DCOPs. Our results indicate that LAMDLS-2 converges faster than MGM-2, a benchmark algorithm, to a similar 2-opt solution, in various message latency scenarios. 

**Abstract (ZH)**: 研究人员最近将分布式约束优化问题（DCOPs）扩展为通信感知分布式约束优化问题（CA-DCOPs），以便在消息可能任意延迟的情况下适用。设计用于CA-DCOPs的分布式异步局部搜索和推理算法比用于常规DCOPs的相应算法对消息延迟不太敏感。然而，与可以收敛到k-最优解（k>1，即无法通过一组k个代理进行改进的解）的常规DCOPs的局部搜索算法不同，CA-DCOP算法仅能收敛到1-最优解。在这篇论文中，我们介绍了通信延迟感知单调分布式局部搜索-2（LAMDLS-2），其中代理形成对并协调双边指派替换。LAMDLS-2是单调的，可以收敛到2-最优解，并且还对消息延迟具有鲁棒性，使其适合用于CA-DCOPs。我们的结果表明，与基准算法MGM-2相比，LAMDLS-2可以在各种消息延迟场景下更快地收敛到相似的2-最优解。 

---
# Weight Ensembling Improves Reasoning in Language Models 

**Title (ZH)**: Weight Ensemble improves Reasoning in Language Models 

**Authors**: Xingyu Dang, Christina Baek, Kaiyue Wen, Zico Kolter, Aditi Raghunathan  

**Link**: [PDF](https://arxiv.org/pdf/2504.10478)  

**Abstract**: We investigate a failure mode that arises during the training of reasoning models, where the diversity of generations begins to collapse, leading to suboptimal test-time scaling. Notably, the Pass@1 rate reliably improves during supervised finetuning (SFT), but Pass@k rapidly deteriorates. Surprisingly, a simple intervention of interpolating the weights of the latest SFT checkpoint with an early checkpoint, otherwise known as WiSE-FT, almost completely recovers Pass@k while also improving Pass@1. The WiSE-FT variant achieves better test-time scaling (Best@k, majority vote) and achieves superior results with less data when tuned further by reinforcement learning. Finally, we find that WiSE-FT provides complementary performance gains that cannot be achieved only through diversity-inducing decoding strategies, like temperature scaling. We formalize a bias-variance tradeoff of Pass@k with respect to the expectation and variance of Pass@1 over the test distribution. We find that WiSE-FT can reduce bias and variance simultaneously, while temperature scaling inherently trades-off between bias and variance. 

**Abstract (ZH)**: 我们研究了推理模型训练过程中出现的一种故障模式，其中生成的多样性开始崩溃，导致测验时扩展性变差。值得注意的是，监督微调（SFT）过程中Pass@1率可靠地得到了提高，但Pass@k迅速恶化。令人惊讶的是，通过将最新SFT检查点的权重与早期检查点的权重进行插值（即WiSE-FT）的简单干预，几乎完全恢复了Pass@k，同时提高了Pass@1。WiSE-FT变体在最佳测验时扩展性（Best@k，多数投票）方面表现更佳，且在进一步通过强化学习调整时可获得更好的结果。最后，我们发现WiSE-FT提供了补充性的性能增益，这些增益无法仅通过诱发多样性的解码策略（如温度缩放）来实现。我们将Pass@k的偏差-方差权衡形式化为Pass@1在测试分布上的期望和方差之间的关系。我们发现WiSE-FT可以同时减少偏差和方差，而温度缩放本质上是在偏差和方差之间进行权衡。 

---
# Multimodal Long Video Modeling Based on Temporal Dynamic Context 

**Title (ZH)**: 基于-temporal动态上下文的多模态长视频建模 

**Authors**: Haoran Hao, Jiaming Han, Yiyuan Zhang, Xiangyu Yue  

**Link**: [PDF](https://arxiv.org/pdf/2504.10443)  

**Abstract**: Recent advances in Large Language Models (LLMs) have led to significant breakthroughs in video understanding. However, existing models still struggle with long video processing due to the context length constraint of LLMs and the vast amount of information within the video. Although some recent methods are designed for long video understanding, they often lose crucial information during token compression and struggle with additional modality like audio. In this work, we propose a dynamic long video encoding method utilizing the temporal relationship between frames, named Temporal Dynamic Context (TDC). Firstly, we segment the video into semantically consistent scenes based on inter-frame similarities, then encode each frame into tokens using visual-audio encoders. Secondly, we propose a novel temporal context compressor to reduce the number of tokens within each segment. Specifically, we employ a query-based Transformer to aggregate video, audio, and instruction text tokens into a limited set of temporal context tokens. Finally, we feed the static frame tokens and the temporal context tokens into the LLM for video understanding. Furthermore, to handle extremely long videos, we propose a training-free chain-of-thought strategy that progressively extracts answers from multiple video segments. These intermediate answers serve as part of the reasoning process and contribute to the final answer. We conduct extensive experiments on general video understanding and audio-video understanding benchmarks, where our method demonstrates strong performance. The code and models are available at this https URL. 

**Abstract (ZH)**: 近期大型语言模型（LLMs）的进步在视频理解领域取得了重要突破，但现有模型仍难以处理长视频，这是由于LLMs的上下文长度限制以及视频中的海量信息。尽管有一些近期方法旨在处理长视频理解，但在进行标记压缩时往往会丢失重要信息，且难以处理如音频等其他模态信息。在这项工作中，我们提出了一种利用帧间关系的动态长视频编码方法，名为时序动态上下文（TDC）。首先，我们基于帧间的相似性将视频分割成语义一致的场景，然后使用视觉-音频编码器对每一帧进行编码为标记。其次，我们提出了一种新型的时序上下文压缩器，以减少每个片段内标记的数量。具体来说，我们采用基于查询的Transformer，将视频、音频和指令文本标记聚合为一组有限的时序上下文标记。最后，我们将静态帧标记和时序上下文标记输入LLM进行视频理解。此外，为了处理极其长的视频，我们提出了一种无需训练的逐步推理策略，逐步从多个视频片段中提取答案。这些中间答案作为推理过程的一部分，并对最终答案产生贡献。我们在通用视频理解与音视频理解基准测试上进行了广泛实验，验证了该方法的出色性能。相关代码和模型可在以下链接获取。 

---
# LLM Can be a Dangerous Persuader: Empirical Study of Persuasion Safety in Large Language Models 

**Title (ZH)**: LLM 可能是一个危险的说服者：大规模语言模型的说服安全性实证研究 

**Authors**: Minqian Liu, Zhiyang Xu, Xinyi Zhang, Heajun An, Sarvech Qadir, Qi Zhang, Pamela J. Wisniewski, Jin-Hee Cho, Sang Won Lee, Ruoxi Jia, Lifu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10430)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have enabled them to approach human-level persuasion capabilities. However, such potential also raises concerns about the safety risks of LLM-driven persuasion, particularly their potential for unethical influence through manipulation, deception, exploitation of vulnerabilities, and many other harmful tactics. In this work, we present a systematic investigation of LLM persuasion safety through two critical aspects: (1) whether LLMs appropriately reject unethical persuasion tasks and avoid unethical strategies during execution, including cases where the initial persuasion goal appears ethically neutral, and (2) how influencing factors like personality traits and external pressures affect their behavior. To this end, we introduce PersuSafety, the first comprehensive framework for the assessment of persuasion safety which consists of three stages, i.e., persuasion scene creation, persuasive conversation simulation, and persuasion safety assessment. PersuSafety covers 6 diverse unethical persuasion topics and 15 common unethical strategies. Through extensive experiments across 8 widely used LLMs, we observe significant safety concerns in most LLMs, including failing to identify harmful persuasion tasks and leveraging various unethical persuasion strategies. Our study calls for more attention to improve safety alignment in progressive and goal-driven conversations such as persuasion. 

**Abstract (ZH)**: 最近大型语言模型的进展使其具备了接近人类级的说服能力。然而，这种潜力也引发了对其通过操控、欺骗和利用漏洞等不道德手段进行说服的安全风险的担忧。在这项工作中，我们通过两个关键方面系统地研究了大型语言模型的说服安全：(1) 大型语言模型是否适当地拒绝不道德的说服任务并在执行过程中避免不道德策略，包括初始说服目标看似伦理中立的情况，以及(2) 个人特质和外部压力等影响因素如何影响其行为。为此，我们引入了PersuSafety，这是第一个全面的说服安全性评估框架，包括场景创建、说服对话模拟和说服安全评估三个阶段。PersuSafety涵盖了6个不同的不道德说服主题和15种常见的不道德策略。通过对8种广泛使用的大型语言模型进行广泛的实验，我们观察到大多数大型语言模型存在显著的安全问题，包括未能识别有害的说服任务以及利用各种不道德的说服策略。我们的研究呼吁在进步性和目标导向的对话，如说服过程中，更加关注安全性对齐问题。 

---
# Can We Edit LLMs for Long-Tail Biomedical Knowledge? 

**Title (ZH)**: 长尾 biomedical 知识能否编辑大模型？ 

**Authors**: Xinhao Yi, Jake Lever, Kevin Bryson, Zaiqiao Meng  

**Link**: [PDF](https://arxiv.org/pdf/2504.10421)  

**Abstract**: Knowledge editing has emerged as an effective approach for updating large language models (LLMs) by modifying their internal knowledge. However, their application to the biomedical domain faces unique challenges due to the long-tailed distribution of biomedical knowledge, where rare and infrequent information is prevalent. In this paper, we conduct the first comprehensive study to investigate the effectiveness of knowledge editing methods for editing long-tail biomedical knowledge. Our results indicate that, while existing editing methods can enhance LLMs' performance on long-tail biomedical knowledge, their performance on long-tail knowledge remains inferior to that on high-frequency popular knowledge, even after editing. Our further analysis reveals that long-tail biomedical knowledge contains a significant amount of one-to-many knowledge, where one subject and relation link to multiple objects. This high prevalence of one-to-many knowledge limits the effectiveness of knowledge editing in improving LLMs' understanding of long-tail biomedical knowledge, highlighting the need for tailored strategies to bridge this performance gap. 

**Abstract (ZH)**: 知识编辑方法在编辑长尾生物医学知识效果的研究 

---
# LLM-SRBench: A New Benchmark for Scientific Equation Discovery with Large Language Models 

**Title (ZH)**: LLM-SRBench：大规模语言模型在科学方程发现方面的新型基准 

**Authors**: Parshin Shojaee, Ngoc-Hieu Nguyen, Kazem Meidani, Amir Barati Farimani, Khoa D Doan, Chandan K Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2504.10415)  

**Abstract**: Scientific equation discovery is a fundamental task in the history of scientific progress, enabling the derivation of laws governing natural phenomena. Recently, Large Language Models (LLMs) have gained interest for this task due to their potential to leverage embedded scientific knowledge for hypothesis generation. However, evaluating the true discovery capabilities of these methods remains challenging, as existing benchmarks often rely on common equations that are susceptible to memorization by LLMs, leading to inflated performance metrics that do not reflect discovery. In this paper, we introduce LLM-SRBench, a comprehensive benchmark with 239 challenging problems across four scientific domains specifically designed to evaluate LLM-based scientific equation discovery methods while preventing trivial memorization. Our benchmark comprises two main categories: LSR-Transform, which transforms common physical models into less common mathematical representations to test reasoning beyond memorized forms, and LSR-Synth, which introduces synthetic, discovery-driven problems requiring data-driven reasoning. Through extensive evaluation of several state-of-the-art methods, using both open and closed LLMs, we find that the best-performing system so far achieves only 31.5% symbolic accuracy. These findings highlight the challenges of scientific equation discovery, positioning LLM-SRBench as a valuable resource for future research. 

**Abstract (ZH)**: 科学方程发现是科学进步史上的一个基本任务，能够导出自然现象的规律。最近，大规模语言模型（LLMs）因潜在的嵌入式科学知识利用能力而被用于这一任务。然而，评估这些方法的真实发现能力依然具有挑战性，因为现有的基准通常依赖于容易被LLMs记忆的常见方程，导致膨胀的性能指标无法反映真实的发现能力。在这篇论文中，我们引入了LLM-SRBench，这是一个包含239个具有挑战性问题的综合基准，专门针对四个科学领域中的LLM基科学方程发现方法进行评估，同时防止简单的记忆化。该基准包括两个主要类别：LSR-Transform，将常见的物理模型转换为不太常见的数学表示，以测试超越记忆形式的推理，以及LSR-Synth，引入了基于数据驱动推理的合成、发现驱动问题。通过对几种最先进的方法进行广泛的评估，使用开环和闭环LLMs，我们发现迄今为止性能最好的系统仅实现了31.5%的符号准确性。这些发现突出了科学方程发现的挑战，将LLM-SRBench定位为未来研究的宝贵资源。 

---
# Performance of Large Language Models in Supporting Medical Diagnosis and Treatment 

**Title (ZH)**: 大型语言模型在支持医疗诊断和治疗中的性能研究 

**Authors**: Diogo Sousa, Guilherme Barbosa, Catarina Rocha, Dulce Oliveira  

**Link**: [PDF](https://arxiv.org/pdf/2504.10405)  

**Abstract**: The integration of Large Language Models (LLMs) into healthcare holds significant potential to enhance diagnostic accuracy and support medical treatment planning. These AI-driven systems can analyze vast datasets, assisting clinicians in identifying diseases, recommending treatments, and predicting patient outcomes. This study evaluates the performance of a range of contemporary LLMs, including both open-source and closed-source models, on the 2024 Portuguese National Exam for medical specialty access (PNA), a standardized medical knowledge assessment. Our results highlight considerable variation in accuracy and cost-effectiveness, with several models demonstrating performance exceeding human benchmarks for medical students on this specific task. We identify leading models based on a combined score of accuracy and cost, discuss the implications of reasoning methodologies like Chain-of-Thought, and underscore the potential for LLMs to function as valuable complementary tools aiding medical professionals in complex clinical decision-making. 

**Abstract (ZH)**: 大型语言模型在医疗领域的集成具有显著潜力，可提高诊断准确性并支持医疗治疗规划。这些基于人工智能的系统可以分析大量数据集，协助临床医生识别疾病、推荐治疗方案并预测患者结果。本研究评估了包括开源和闭源模型在内的多种当代大型语言模型在2024年葡萄牙医疗专科入学全国考试（PNA）上的性能，这是一个标准化的医学知识评估。我们的结果显示了显著的准确性和成本效益差异，其中一些模型在特定任务上的表现超越了医学学生的人类基准。我们基于准确性和成本的综合评分确定了表现领先的模型，讨论了如链式思考等推理方法的影响，并强调了大型语言模型在复杂临床决策中作为有价值的补充工具的潜力。 

---
# Teacher Motion Priors: Enhancing Robot Locomotion over Challenging Terrain 

**Title (ZH)**: 教师运动先验：提升机器人在复杂地形上的运动性能 

**Authors**: Fangcheng Jin, Yuqi Wang, Peixin Ma, Guodong Yang, Pan Zhao, En Li, Zhengtao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10390)  

**Abstract**: Achieving robust locomotion on complex terrains remains a challenge due to high dimensional control and environmental uncertainties. This paper introduces a teacher prior framework based on the teacher student paradigm, integrating imitation and auxiliary task learning to improve learning efficiency and generalization. Unlike traditional paradigms that strongly rely on encoder-based state embeddings, our framework decouples the network design, simplifying the policy network and deployment. A high performance teacher policy is first trained using privileged information to acquire generalizable motion skills. The teacher's motion distribution is transferred to the student policy, which relies only on noisy proprioceptive data, via a generative adversarial mechanism to mitigate performance degradation caused by distributional shifts. Additionally, auxiliary task learning enhances the student policy's feature representation, speeding up convergence and improving adaptability to varying terrains. The framework is validated on a humanoid robot, showing a great improvement in locomotion stability on dynamic terrains and significant reductions in development costs. This work provides a practical solution for deploying robust locomotion strategies in humanoid robots. 

**Abstract (ZH)**: 基于教师学生的先验框架：通过模仿和辅助任务学习实现复杂地形上的稳健运动 

---
# SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning 

**Title (ZH)**: SymRTLO：通过LLM和神经元启发的符号推理增强RTL代码优化 

**Authors**: Yiting Wang, Wanghao Ye, Ping Guo, Yexiao He, Ziyao Wang, Yexiao He, Bowei Tian, Shwai He, Guoheng Sun, Zheyu Shen, Sihan Chen, Ankur Srivastava, Qingfu Zhang, Gang Qu, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.10369)  

**Abstract**: Optimizing Register Transfer Level (RTL) code is crucial for improving the power, performance, and area (PPA) of digital circuits in the early stages of synthesis. Manual rewriting, guided by synthesis feedback, can yield high-quality results but is time-consuming and error-prone. Most existing compiler-based approaches have difficulty handling complex design constraints. Large Language Model (LLM)-based methods have emerged as a promising alternative to address these challenges. However, LLM-based approaches often face difficulties in ensuring alignment between the generated code and the provided prompts. This paper presents SymRTLO, a novel neuron-symbolic RTL optimization framework that seamlessly integrates LLM-based code rewriting with symbolic reasoning techniques. Our method incorporates a retrieval-augmented generation (RAG) system of optimization rules and Abstract Syntax Tree (AST)-based templates, enabling LLM-based rewriting that maintains syntactic correctness while minimizing undesired circuit behaviors. A symbolic module is proposed for analyzing and optimizing finite state machine (FSM) logic, allowing fine-grained state merging and partial specification handling beyond the scope of pattern-based compilers. Furthermore, a fast verification pipeline, combining formal equivalence checks with test-driven validation, further reduces the complexity of verification. Experiments on the RTL-Rewriter benchmark with Synopsys Design Compiler and Yosys show that SymRTLO improves power, performance, and area (PPA) by up to 43.9%, 62.5%, and 51.1%, respectively, compared to the state-of-the-art methods. 

**Abstract (ZH)**: SymRTLO：一种新颖的神经符号RTL优化框架 

---
# S1-Bench: A Simple Benchmark for Evaluating System 1 Thinking Capability of Large Reasoning Models 

**Title (ZH)**: S1-Bench: 一种评估大型推理模型系统1思维能力的简单基准 

**Authors**: Wenyuan Zhang, Shuaiyi Nie, Xinghua Zhang, Zefeng Zhang, Tingwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.10368)  

**Abstract**: We introduce S1-Bench, a novel benchmark designed to evaluate Large Reasoning Models' (LRMs) performance on simple tasks that favor intuitive system 1 thinking rather than deliberative system 2 reasoning. While LRMs have achieved significant breakthroughs in complex reasoning tasks through explicit chains of thought, their reliance on deep analytical thinking may limit their system 1 thinking capabilities. Moreover, a lack of benchmark currently exists to evaluate LRMs' performance in tasks that require such capabilities. To fill this gap, S1-Bench presents a set of simple, diverse, and naturally clear questions across multiple domains and languages, specifically designed to assess LRMs' performance in such tasks. Our comprehensive evaluation of 22 LRMs reveals significant lower efficiency tendencies, with outputs averaging 15.5 times longer than those of traditional small LLMs. Additionally, LRMs often identify correct answers early but continue unnecessary deliberation, with some models even producing numerous errors. These findings highlight the rigid reasoning patterns of current LRMs and underscore the substantial development needed to achieve balanced dual-system thinking capabilities that can adapt appropriately to task complexity. 

**Abstract (ZH)**: S1-Bench: 一种用于评估大型推理模型在偏向直觉系统1思考的简单任务上性能的新基准 

---
# FingER: Content Aware Fine-grained Evaluation with Reasoning for AI-Generated Videos 

**Title (ZH)**: FingER：基于内容的细粒度评估与推理框架用于AI生成的视频 

**Authors**: Rui Chen, Lei Sun, Jing Tang, Geng Li, Xiangxiang Chu  

**Link**: [PDF](https://arxiv.org/pdf/2504.10358)  

**Abstract**: Recent advances in video generation have posed great challenges in the assessment of AI-generated content, particularly with the emergence of increasingly sophisticated models. The various inconsistencies and defects observed in such videos are inherently complex, making overall scoring notoriously difficult. In this paper, we emphasize the critical importance of integrating fine-grained reasoning into video evaluation, and we propose $\textbf{F}$ing$\textbf{ER}$, a novel entity-level reasoning evaluation framework that first automatically generates $\textbf{F}$ine-grained $\textbf{E}$ntity-level questions, and then answers those questions by a $\textbf{R}$easoning model with scores, which can be subsequently weighted summed to an overall score for different applications. Specifically, we leverage LLMs to derive entity-level questions across five distinct perspectives, which (i) often focus on some specific entities of the content, thereby making answering or scoring much easier by MLLMs, and (ii) are more interpretable. Then we construct a FingER dataset, consisting of approximately 3.3k videos and corresponding 60k fine-grained QA annotations, each with detailed reasons. Based on that, we further investigate various training protocols to best incentivize the reasoning capability of MLLMs for correct answer prediction. Extensive experiments demonstrate that a reasoning model trained using Group Relative Policy Optimization (GRPO) with a cold-start strategy achieves the best performance. Notably, our model surpasses existing methods by a relative margin of $11.8\%$ on GenAI-Bench and $5.5\%$ on MonetBench with only 3.3k training videos, which is at most one-tenth of the training samples utilized by other methods. Our code and dataset will be released soon. 

**Abstract (ZH)**: Recent Advances in Video Generation Pose Great Challenges in the Assessment of AI-Generated Content: The Critical Importance of Fine-Grained Reasoning in Video Evaluation and the Proposal of FingER, a Novel Entity-Level Reasoning Evaluation Framework 

---
# Forecasting from Clinical Textual Time Series: Adaptations of the Encoder and Decoder Language Model Families 

**Title (ZH)**: 从临床文本时间序列预测：编码器和解码器语言模型家族的适应性研究 

**Authors**: Shahriar Noroozizadeh, Sayantan Kumar, Jeremy C. Weiss  

**Link**: [PDF](https://arxiv.org/pdf/2504.10340)  

**Abstract**: Clinical case reports encode rich, temporal patient trajectories that are often underexploited by traditional machine learning methods relying on structured data. In this work, we introduce the forecasting problem from textual time series, where timestamped clinical findings--extracted via an LLM-assisted annotation pipeline--serve as the primary input for prediction. We systematically evaluate a diverse suite of models, including fine-tuned decoder-based large language models and encoder-based transformers, on tasks of event occurrence prediction, temporal ordering, and survival analysis. Our experiments reveal that encoder-based models consistently achieve higher F1 scores and superior temporal concordance for short- and long-horizon event forecasting, while fine-tuned masking approaches enhance ranking performance. In contrast, instruction-tuned decoder models demonstrate a relative advantage in survival analysis, especially in early prognosis settings. Our sensitivity analyses further demonstrate the importance of time ordering, which requires clinical time series construction, as compared to text ordering, the format of the text inputs that LLMs are classically trained on. This highlights the additional benefit that can be ascertained from time-ordered corpora, with implications for temporal tasks in the era of widespread LLM use. 

**Abstract (ZH)**: 基于文本时间序列的临床案例报告预测问题探究：编码丰富的患者轨迹并系统评估不同模型 

---
# AutoStyle-TTS: Retrieval-Augmented Generation based Automatic Style Matching Text-to-Speech Synthesis 

**Title (ZH)**: AutoStyle-TTS：检索增强生成驱动的自动风格匹配文本转语音合成 

**Authors**: Dan Luo, Chengyuan Ma, Weiqin Li, Jun Wang, Wei Chen, Zhiyong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.10309)  

**Abstract**: With the advancement of speech synthesis technology, users have higher expectations for the naturalness and expressiveness of synthesized speech. But previous research ignores the importance of prompt selection. This study proposes a text-to-speech (TTS) framework based on Retrieval-Augmented Generation (RAG) technology, which can dynamically adjust the speech style according to the text content to achieve more natural and vivid communication effects. We have constructed a speech style knowledge database containing high-quality speech samples in various contexts and developed a style matching scheme. This scheme uses embeddings, extracted by Llama, PER-LLM-Embedder,and Moka, to match with samples in the knowledge database, selecting the most appropriate speech style for synthesis. Furthermore, our empirical research validates the effectiveness of the proposed method. Our demo can be viewed at: this https URL 

**Abstract (ZH)**: 随着语音合成技术的发展，用户对合成语音的自然度和表现力有了更高的期望。但之前的研究所忽视了提示选择的重要性。本研究提出了一种基于检索增强生成（RAG）技术的文本到语音（TTS）框架，可以根据文本内容动态调整语音风格，以实现更加自然和生动的交流效果。我们构建了一个包含多种情境下的高质量语音样本的语音风格知识数据库，并开发了一种风格匹配方案。该方案使用Llama、PER-LLM-Embedder和Moka提取的嵌入向量，与知识数据库中的样本进行匹配，选取最适合的语音风格进行合成。此外，我们的实证研究验证了所提出方法的有效性。我们的演示可以查看：this https URL。 

---
# Characterizing LLM-driven Social Network: The Chirper.ai Case 

**Title (ZH)**: LLM驱动的社会网络-characterizer: Chirper.ai案例研究 

**Authors**: Yiming Zhu, Yupeng He, Ehsan-Ul Haq, Gareth Tyson, Pan Hui  

**Link**: [PDF](https://arxiv.org/pdf/2504.10286)  

**Abstract**: Large language models (LLMs) demonstrate the ability to simulate human decision-making processes, enabling their use as agents in modeling sophisticated social networks, both offline and online. Recent research has explored collective behavioral patterns and structural characteristics of LLM agents within simulated networks. However, empirical comparisons between LLM-driven and human-driven online social networks remain scarce, limiting our understanding of how LLM agents differ from human users. This paper presents a large-scale analysis of this http URL, an X/Twitter-like social network entirely populated by LLM agents, comprising over 65,000 agents and 7.7 million AI-generated posts. For comparison, we collect a parallel dataset from Mastodon, a human-driven decentralized social network, with over 117,000 users and 16 million posts. We examine key differences between LLM agents and humans in posting behaviors, abusive content, and social network structures. Our findings provide critical insights into the evolving landscape of online social network analysis in the AI era, offering a comprehensive profile of LLM agents in social simulations. 

**Abstract (ZH)**: 大型语言模型在模拟社交网络中的集体行为模式与结构特征：基于this http URL与Mastodon的大规模分析 

---
# Zero-shot Autonomous Microscopy for Scalable and Intelligent Characterization of 2D Materials 

**Title (ZH)**: 零样本自主显微成像：面向2D材料可扩展与智能表征 

**Authors**: Jingyun Yang, Ruoyan Avery Yin, Chi Jiang, Yuepeng Hu, Xiaokai Zhu, Xingjian Hu, Sutharsika Kumar, Xiao Wang, Xiaohua Zhai, Keran Rong, Yunyue Zhu, Tianyi Zhang, Zongyou Yin, Jing Kong, Neil Zhenqiang Gong, Zhichu Ren, Haozhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10281)  

**Abstract**: Characterization of atomic-scale materials traditionally requires human experts with months to years of specialized training. Even for trained human operators, accurate and reliable characterization remains challenging when examining newly discovered materials such as two-dimensional (2D) structures. This bottleneck drives demand for fully autonomous experimentation systems capable of comprehending research objectives without requiring large training datasets. In this work, we present ATOMIC (Autonomous Technology for Optical Microscopy & Intelligent Characterization), an end-to-end framework that integrates foundation models to enable fully autonomous, zero-shot characterization of 2D materials. Our system integrates the vision foundation model (i.e., Segment Anything Model), large language models (i.e., ChatGPT), unsupervised clustering, and topological analysis to automate microscope control, sample scanning, image segmentation, and intelligent analysis through prompt engineering, eliminating the need for additional training. When analyzing typical MoS2 samples, our approach achieves 99.7% segmentation accuracy for single layer identification, which is equivalent to that of human experts. In addition, the integrated model is able to detect grain boundary slits that are challenging to identify with human eyes. Furthermore, the system retains robust accuracy despite variable conditions including defocus, color temperature fluctuations, and exposure variations. It is applicable to a broad spectrum of common 2D materials-including graphene, MoS2, WSe2, SnSe-regardless of whether they were fabricated via chemical vapor deposition or mechanical exfoliation. This work represents the implementation of foundation models to achieve autonomous analysis, establishing a scalable and data-efficient characterization paradigm that fundamentally transforms the approach to nanoscale materials research. 

**Abstract (ZH)**: 原子尺度材料的传统表征通常需要经过数月至数年专项训练的人类专家。即使是训练有素的操作员，在检查诸如二维（2D）结构等新发现的材料时，准确可靠的表征仍然具有挑战性。这种瓶颈推动了全面自主实验系统的市场需求，这些系统能够在无需大规模训练数据集的情况下理解研究目标。在这项工作中，我们提出了ATOMIC（自主光学显微镜与智能表征技术），这是一个端到端框架，整合了基础模型以实现对2D材料的完全自主和零样本表征。我们的系统将视觉基础模型（即，Anything Mask模型）、大规模语言模型（即，ChatGPT）、无监督聚类和拓扑分析结合在一起，通过提示工程自动化显微镜控制、样品扫描、图像分割和智能分析，无需额外训练。在分析典型的MoS2样品时，我们的方法在单层识别上的分割准确率达到99.7%，与人类专家相当。此外，集成模型能够检测人眼难以识别的晶界缝隙。此外，该系统在焦距变化、色温波动和曝光变化等多种条件下保持了稳健的精度。它适用于包括石墨烯、MoS2、WSe2、SnSe在内的广泛常见的2D材料，不论它们是通过化学气相沉积还是机械剥离制备的。这项工作代表了基础模型在实现自主分析的实施，确立了一种可扩展和数据高效的表征范式，从根本上改变了纳米材料研究的方法。 

---
# RealHarm: A Collection of Real-World Language Model Application Failures 

**Title (ZH)**: 实锤：语言模型实际应用中的失败案例集锦 

**Authors**: Pierre Le Jeune, Jiaen Liu, Luca Rossi, Matteo Dora  

**Link**: [PDF](https://arxiv.org/pdf/2504.10277)  

**Abstract**: Language model deployments in consumer-facing applications introduce numerous risks. While existing research on harms and hazards of such applications follows top-down approaches derived from regulatory frameworks and theoretical analyses, empirical evidence of real-world failure modes remains underexplored. In this work, we introduce RealHarm, a dataset of annotated problematic interactions with AI agents built from a systematic review of publicly reported incidents. Analyzing harms, causes, and hazards specifically from the deployer's perspective, we find that reputational damage constitutes the predominant organizational harm, while misinformation emerges as the most common hazard category. We empirically evaluate state-of-the-art guardrails and content moderation systems to probe whether such systems would have prevented the incidents, revealing a significant gap in the protection of AI applications. 

**Abstract (ZH)**: 面向消费者的语言模型部署引入了诸多风险。尽管现有研究从监管框架和理论分析出发，采用自上而下的方法探讨这类应用的危害和风险，但实际应用场景中失败模式的实证证据仍较为缺乏。在本工作中，我们介绍了RealHarm数据集，该数据集基于系统性回顾公开报告的事件而构建，并对与AI代理互动中的问题进行标注。从部署者的视角分析危害、成因和风险，我们发现声誉损害是主要的企业危害，而错误信息是最常见的风险类别。我们实证评估了最先进的防护栏和内容审核系统，探讨这些系统是否能够预防这些事件，揭示了AI应用保护方面的显著差距。 

---
# Vision based driving agent for race car simulation environments 

**Title (ZH)**: 基于视觉的赛车模拟环境驾驶代理 

**Authors**: Gergely Bári, László Palkovics  

**Link**: [PDF](https://arxiv.org/pdf/2504.10266)  

**Abstract**: In recent years, autonomous driving has become a popular field of study. As control at tire grip limit is essential during emergency situations, algorithms developed for racecars are useful for road cars too. This paper examines the use of Deep Reinforcement Learning (DRL) to solve the problem of grip limit driving in a simulated environment. Proximal Policy Optimization (PPO) method is used to train an agent to control the steering wheel and pedals of the vehicle, using only visual inputs to achieve professional human lap times. The paper outlines the formulation of the task of time optimal driving on a race track as a deep reinforcement learning problem, and explains the chosen observations, actions, and reward functions. The results demonstrate human-like learning and driving behavior that utilize maximum tire grip potential. 

**Abstract (ZH)**: 近年来，自动驾驶已成为一个热门研究领域。由于在紧急情况下轮胎附着极限控制至关重要，赛车领域的算法也有助于道路车辆。本文探讨了在模拟环境中使用深度强化学习（DRL）解决轮胎附着极限驾驶问题的方法。采用 proximal policy optimization (PPO) 方法训练一个代理，仅使用视觉输入来控制车辆的方向盘和踏板，实现专业的人类赛道时间。本文概述了在赛道上实现时间最优驾驶任务作为深度强化学习问题的建模，并解释了选择的观察、动作和奖励函数。结果展示了充分利用轮胎附着潜力的人类学习和驾驶行为。 

---
# MASSeg : 2nd Technical Report for 4th PVUW MOSE Track 

**Title (ZH)**: MASSeg : 第4届PVUW MOSE赛道的2nd技术报告 

**Authors**: Xuqiang Cao, Linnan Zhao, Jiaxuan Zhao, Fang Liu, Puhua Chen, Wenping Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.10254)  

**Abstract**: Complex video object segmentation continues to face significant challenges in small object recognition, occlusion handling, and dynamic scene modeling. This report presents our solution, which ranked second in the MOSE track of CVPR 2025 PVUW Challenge. Based on an existing segmentation framework, we propose an improved model named MASSeg for complex video object segmentation, and construct an enhanced dataset, MOSE+, which includes typical scenarios with occlusions, cluttered backgrounds, and small target instances. During training, we incorporate a combination of inter-frame consistent and inconsistent data augmentation strategies to improve robustness and generalization. During inference, we design a mask output scaling strategy to better adapt to varying object sizes and occlusion levels. As a result, MASSeg achieves a J score of 0.8250, F score of 0.9007, and a J&F score of 0.8628 on the MOSE test set. 

**Abstract (ZH)**: 复杂视频对象分割在小目标识别、遮挡处理和动态场景建模方面仍面临重大挑战。本报告提出了我们的解决方案，在CVPR 2025 PVUW挑战赛MOSE轨道中排名第二。基于现有的分割框架，我们提出了一个改进的模型MASSeg用于复杂视频对象分割，并构建了一个增强的数据集MOSE+，该数据集包含具有遮挡、复杂背景和小目标实例的典型场景。在训练过程中，我们结合使用帧内一致性和非一致性数据增强策略以提高鲁棒性和泛化能力。在推理过程中，我们设计了一种掩码输出缩放策略，以更好地适应物体大小和遮挡程度的变化。结果，MASSeg在MOSE测试集上达到了J分数0.8250，F分数0.9007，以及J&F分数0.8628。 

---
# Localized Cultural Knowledge is Conserved and Controllable in Large Language Models 

**Title (ZH)**: 局部文化知识在大规模语言模型中得以保留和可控 

**Authors**: Veniamin Veselovsky, Berke Argin, Benedikt Stroebl, Chris Wendler, Robert West, James Evans, Thomas L. Griffiths, Arvind Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2504.10191)  

**Abstract**: Just as humans display language patterns influenced by their native tongue when speaking new languages, LLMs often default to English-centric responses even when generating in other languages. Nevertheless, we observe that local cultural information persists within the models and can be readily activated for cultural customization. We first demonstrate that explicitly providing cultural context in prompts significantly improves the models' ability to generate culturally localized responses. We term the disparity in model performance with versus without explicit cultural context the explicit-implicit localization gap, indicating that while cultural knowledge exists within LLMs, it may not naturally surface in multilingual interactions if cultural context is not explicitly provided. Despite the explicit prompting benefit, however, the answers reduce in diversity and tend toward stereotypes. Second, we identify an explicit cultural customization vector, conserved across all non-English languages we explore, which enables LLMs to be steered from the synthetic English cultural world-model toward each non-English cultural world. Steered responses retain the diversity of implicit prompting and reduce stereotypes to dramatically improve the potential for customization. We discuss the implications of explicit cultural customization for understanding the conservation of alternative cultural world models within LLMs, and their controllable utility for translation, cultural customization, and the possibility of making the explicit implicit through soft control for expanded LLM function and appeal. 

**Abstract (ZH)**: 正如人类在使用新语言时会受到母语语言模式的影响，大型语言模型在生成其他语言的内容时往往默认使用以英语为中心的回答方式。然而，我们观察到本地文化信息仍然存在于这些模型中，并且可以轻松激活以实现文化定制。我们首先证明，在提示中明确提供文化背景可以显著提高模型生成文化本地化回答的能力。我们将模型在有和没有明确文化背景的情况下性能差距称为显性-隐性本地化差距，表明虽然文化知识存在于大型语言模型中，但如果未提供明确的文化背景，这些知识可能不会自然地在多语言交互中显现出来。尽管使用明确提示可以带来好处，但回答的多样性会减少，并倾向于形成刻板印象。其次，我们发现了一种适用于所有探索的非英语语言的显性文化定制向量，这种向量能够引导大型语言模型从合成的英语文化世界观转向每个非英语文化的世界。经过引导的回应保留了隐性提示的多样性，并减少了刻板印象，从而显著提高了定制的可能性。我们讨论了显性文化定制对理解大型语言模型中替代文化世界观的保守性及其可控实用性的含义，以及通过软控制实现扩展的大型语言模型功能和吸引力的可能性。 

---
# Efficient Generative Model Training via Embedded Representation Warmup 

**Title (ZH)**: 嵌入表示预热实现高效的生成模型训练 

**Authors**: Deyuan Liu, Peng Sun, Xufeng Li, Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.10188)  

**Abstract**: Diffusion models excel at generating high-dimensional data but fall short in training efficiency and representation quality compared to self-supervised methods. We identify a key bottleneck: the underutilization of high-quality, semantically rich representations during training notably slows down convergence. Our systematic analysis reveals a critical representation processing region -- primarily in the early layers -- where semantic and structural pattern learning takes place before generation can occur. To address this, we propose Embedded Representation Warmup (ERW), a plug-and-play framework where in the first stage we get the ERW module serves as a warmup that initializes the early layers of the diffusion model with high-quality, pretrained representations. This warmup minimizes the burden of learning representations from scratch, thereby accelerating convergence and boosting performance. Our theoretical analysis demonstrates that ERW's efficacy depends on its precise integration into specific neural network layers -- termed the representation processing region -- where the model primarily processes and transforms feature representations for later generation. We further establish that ERW not only accelerates training convergence but also enhances representation quality: empirically, our method achieves a 40$\times$ acceleration in training speed compared to REPA, the current state-of-the-art methods. Code is available at this https URL. 

**Abstract (ZH)**: 差分模型在生成高维数据方面表现出色，但在训练效率和表示质量方面逊色于自我监督方法。我们识别出一个关键瓶颈：训练过程中对高质量语义丰富的表示利用不足显著减慢了收敛速度。我们的系统性分析揭示了一个关键的表示处理区域——主要在早期层中，语义和结构模式学习在此前发生，之后才能进行生成。为了解决这一问题，我们提出了一种即插即用框架嵌入表示预热（ERW），在第一阶段，ERW模块作为预热过程，用预训练的高质量表示初始化差分模型的早期层。这种预热过程减少了从头学习表示的负担，从而加速了收敛并提高了性能。我们的理论分析表明，ERW的有效性取决于其精确集成到特定的神经网络层——称为表示处理区域——其中模型主要处理和转换特征表示以供后续生成。此外，我们进一步证明，ERW不仅加速了训练收敛，还提高了表示质量：实证结果显示，与当前最先进的方法REPA相比，我们的方法实现了训练速度40倍的加速。代码可在以下网址获取。 

---
# Deep Reasoning Translation via Reinforcement Learning 

**Title (ZH)**: 基于强化学习的深层推理翻译 

**Authors**: Jiaan Wang, Fandong Meng, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.10187)  

**Abstract**: Recently, deep reasoning LLMs (e.g., OpenAI o1/o3 and DeepSeek-R1) have shown promising performance in various complex tasks. Free translation is an important and interesting task in the multilingual world, which requires going beyond word-for-word translation and taking cultural differences into account. This task is still under-explored in deep reasoning LLMs. In this paper, we introduce DeepTrans, a deep reasoning translation model that learns free translation via reinforcement learning. Specifically, we carefully build a reward model with pre-defined scoring criteria on both the translation results and the thought process. Given the source sentences, the reward model teaches the deep translation model how to think and free-translate them during reinforcement learning. In this way, training DeepTrans does not need any labeled translations, avoiding the human-intensive annotation or resource-intensive data synthesis. Experimental results show the effectiveness of DeepTrans. Using Qwen2.5-7B as the backbone, DeepTrans improves performance by 16.3% in literature translation, and outperforms strong deep reasoning baselines as well as baselines that are fine-tuned with synthesized data. Moreover, we summarize the failures and interesting findings during our RL exploration. We hope this work could inspire other researchers in free translation. 

**Abstract (ZH)**: Recent深 reasoning大模型（如OpenAI o1/o3和DeepSeek-R1）在各种复杂任务中显示出有前途的表现。自由翻译是多语言世界中一个重要的且有趣的任务，它要求超出逐词翻译并考虑文化差异。该任务在深 reasoning大模型中仍被未充分探索。在本文中，我们介绍了DeepTrans，这是一种通过强化学习学习自由翻译的深 reasoning翻译模型。具体地，我们精心构建了一个奖励模型，该模型基于翻译结果和思考过程的预定义评分标准。给定源句子，奖励模型在强化学习过程中教导深推理翻译模型如何思考并自由翻译。通过这种方式，训练DeepTrans无需任何标注的译文，避免了人力密集型的标注或资源密集型的数据合成。实验结果表明了DeepTrans的有效性。使用Qwen2.5-7B作为主干，DeepTrans在文献翻译方面提高了16.3%的表现，并优于强大的深推理基线以及基于合成数据微调的基线。此外，我们在RL探索期间总结了失败和有趣的发现。我们希望这项工作能够激励其他研究人员在自由翻译方面做出更多努力。 

---
# LLM Unlearning Reveals a Stronger-Than-Expected Coreset Effect in Current Benchmarks 

**Title (ZH)**: LLM去学习揭示了当前基准中比预期更强的核集效应 

**Authors**: Soumyadeep Pal, Changsheng Wang, James Diffenderfer, Bhavya Kailkhura, Sijia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.10185)  

**Abstract**: Large language model unlearning has become a critical challenge in ensuring safety and controlled model behavior by removing undesired data-model influences from the pretrained model while preserving general utility. Significant recent efforts have been dedicated to developing LLM unlearning benchmarks such as WMDP (Weapons of Mass Destruction Proxy) and MUSE (Machine Unlearning Six-way Evaluation), facilitating standardized unlearning performance assessment and method comparison. Despite their usefulness, we uncover for the first time a novel coreset effect within these benchmarks. Specifically, we find that LLM unlearning achieved with the original (full) forget set can be effectively maintained using a significantly smaller subset (functioning as a "coreset"), e.g., as little as 5% of the forget set, even when selected at random. This suggests that LLM unlearning in these benchmarks can be performed surprisingly easily, even in an extremely low-data regime. We demonstrate that this coreset effect remains strong, regardless of the LLM unlearning method used, such as NPO (Negative Preference Optimization) and RMU (Representation Misdirection Unlearning), the popular ones in these benchmarks. The surprisingly strong coreset effect is also robust across various data selection methods, ranging from random selection to more sophisticated heuristic approaches. We explain the coreset effect in LLM unlearning through a keyword-based perspective, showing that keywords extracted from the forget set alone contribute significantly to unlearning effectiveness and indicating that current unlearning is driven by a compact set of high-impact tokens rather than the entire dataset. We further justify the faithfulness of coreset-unlearned models along additional dimensions, such as mode connectivity and robustness to jailbreaking attacks. Codes are available at this https URL. 

**Abstract (ZH)**: 大规模语言模型去学习已成为确保安全和控制模型行为的关键挑战，通过从预训练模型中移除不必要的数据-模型影响同时保留一般用途。最近，人们致力于开发大规模语言模型去学习基准，如WMDP（大规模破坏代理）和MUSE（机器去学习六项评估），推进了标准化去学习性能评估和方法比较。尽管这些基准有用，我们首次发现其中存在一种新的Coreset效应。具体而言，我们发现使用原始（完整）遗忘集进行的大规模语言模型去学习可以通过一个显著更小的子集（作为“Coreset”）有效维持，例如，仅需遗忘集的5%甚至随机选择的子集就可以。这表明在这些基准中，即使是数据极低的情况下，大规模语言模型去学习也可以出奇地容易完成。我们证明无论使用何种大规模语言模型去学习方法（例如NPO负偏好优化和RMU表征误导去学习，这些方法在这些基准中很受欢迎），这种Coreset效应依旧强烈。这种出奇的Coreset效应在不同数据选择方法（从随机选择到更复杂的启发式方法）下也表现出稳健性。我们通过基于关键词的视角解释大规模语言模型去学习中的Coreset效应，表明仅从遗忘集中提取的关键词对去学习效果有显著贡献，表明当前的去学习主要是由一组具有高影响力的关键token驱动而非整个数据集。此外，我们从模式连通性和对抗狱笼攻击的鲁棒性等其他维度进一步证实Coreset-去学习模型的忠实性。代码可在以下链接获取。 

---
# HalluSearch at SemEval-2025 Task 3: A Search-Enhanced RAG Pipeline for Hallucination Detection 

**Title (ZH)**: HalluSearch 在 SemEval-2025 任务 3 中: 带有搜索增强的RAG管道模型在幻觉检测中的应用 

**Authors**: Mohamed A. Abdallah, Samhaa R. El-Beltagy  

**Link**: [PDF](https://arxiv.org/pdf/2504.10168)  

**Abstract**: In this paper, we present HalluSearch, a multilingual pipeline designed to detect fabricated text spans in Large Language Model (LLM) outputs. Developed as part of Mu-SHROOM, the Multilingual Shared-task on Hallucinations and Related Observable Overgeneration Mistakes, HalluSearch couples retrieval-augmented verification with fine-grained factual splitting to identify and localize hallucinations in fourteen different languages. Empirical evaluations show that HalluSearch performs competitively, placing fourth in both English (within the top ten percent) and Czech. While the system's retrieval-based strategy generally proves robust, it faces challenges in languages with limited online coverage, underscoring the need for further research to ensure consistent hallucination detection across diverse linguistic contexts. 

**Abstract (ZH)**: HalluSearch：一种用于检测大型语言模型输出中伪造文本片段的多语言管道 

---
# C-FAITH: A Chinese Fine-Grained Benchmark for Automated Hallucination Evaluation 

**Title (ZH)**: C-FAITH: 一个中文细粒度幻觉评估基准 

**Authors**: Xu Zhang, Zhifei Liu, Jiahao Wang, Huixuan Zhang, Fan Xu, Junzhe Zhang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2504.10167)  

**Abstract**: Despite the rapid advancement of large language models, they remain highly susceptible to generating hallucinations, which significantly hinders their widespread application. Hallucination research requires dynamic and fine-grained evaluation. However, most existing hallucination benchmarks (especially in Chinese language) rely on human annotations, making automatical and cost-effective hallucination evaluation challenging. To address this, we introduce HaluAgent, an agentic framework that automatically constructs fine-grained QA dataset based on some knowledge documents. Our experiments demonstrate that the manually designed rules and prompt optimization can improve the quality of generated data. Using HaluAgent, we construct C-FAITH, a Chinese QA hallucination benchmark created from 1,399 knowledge documents obtained from web scraping, totaling 60,702 entries. We comprehensively evaluate 16 mainstream LLMs with our proposed C-FAITH, providing detailed experimental results and analysis. 

**Abstract (ZH)**: 尽管大型语言模型取得了 rapid advancement，它们仍然高度容易生成幻觉，这严重阻碍了它们的广泛应用。幻觉研究需要动态且细粒度的评估。然而，现有的大多数幻觉基准（尤其是在中文方面）依赖于人工标注，这使得自动和低成本的幻觉评估颇具挑战性。为解决这一问题，我们引入了HaluAgent，这是一个基于某些知识文档自动构建细粒度问答数据集的代理框架。我们的实验表明，手动设计的规则和提示优化可以提高生成数据的质量。使用HaluAgent，我们构建了C-FAITH，这是一个源自网络爬取的1,399个知识文档、共计60,702个条目的中文问答幻觉基准。我们使用C-FAITH全面评估了16个主流LLM，并提供了详细的实验结果和分析。 

---
# WildLive: Near Real-time Visual Wildlife Tracking onboard UAVs 

**Title (ZH)**: WildLive：机载无人机上的近实时野生动物视觉跟踪 

**Authors**: Nguyen Ngoc Dat, Tom Richardson, Matthew Watson, Kilian Meier, Jenna Kline, Sid Reid, Guy Maalouf, Duncan Hine, Majid Mirmehdi, Tilo Burghardt  

**Link**: [PDF](https://arxiv.org/pdf/2504.10165)  

**Abstract**: Live tracking of wildlife via high-resolution video processing directly onboard drones is widely unexplored and most existing solutions rely on streaming video to ground stations to support navigation. Yet, both autonomous animal-reactive flight control beyond visual line of sight and/or mission-specific individual and behaviour recognition tasks rely to some degree on this capability. In response, we introduce WildLive -- a near real-time animal detection and tracking framework for high-resolution imagery running directly onboard uncrewed aerial vehicles (UAVs). The system performs multi-animal detection and tracking at 17fps+ for HD and 7fps+ on 4K video streams suitable for operation during higher altitude flights to minimise animal disturbance. Our system is optimised for Jetson Orin AGX onboard hardware. It integrates the efficiency of sparse optical flow tracking and mission-specific sampling with device-optimised and proven YOLO-driven object detection and segmentation techniques. Essentially, computational resource is focused onto spatio-temporal regions of high uncertainty to significantly improve UAV processing speeds without domain-specific loss of accuracy. Alongside, we introduce our WildLive dataset, which comprises 200k+ annotated animal instances across 19k+ frames from 4K UAV videos collected at the Ol Pejeta Conservancy in Kenya. All frames contain ground truth bounding boxes, segmentation masks, as well as individual tracklets and tracking point trajectories. We compare our system against current object tracking approaches including OC-SORT, ByteTrack, and SORT. Our multi-animal tracking experiments with onboard hardware confirm that near real-time high-resolution wildlife tracking is possible on UAVs whilst maintaining high accuracy levels as needed for future navigational and mission-specific animal-centric operational autonomy. 

**Abstract (ZH)**: 基于无人机上板的高分辨率视频处理实时野生动物追踪：WildLive框架 

---
# MT-R1-Zero: Advancing LLM-based Machine Translation via R1-Zero-like Reinforcement Learning 

**Title (ZH)**: MT-R1-Zero: 基于R1-Zero-like强化学习推进基于LLM的机器翻译 

**Authors**: Zhaopeng Feng, Shaosheng Cao, Jiahan Ren, Jiayuan Su, Ruizhe Chen, Yan Zhang, Zhe Xu, Yao Hu, Jian Wu, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.10160)  

**Abstract**: Large-scale reinforcement learning (RL) methods have proven highly effective in enhancing the reasoning abilities of large language models (LLMs), particularly for tasks with verifiable solutions such as mathematics and coding. However, applying this idea to machine translation (MT), where outputs are flexibly formatted and difficult to automatically evaluate with explicit rules, remains underexplored. In this work, we introduce MT-R1-Zero, the first open-source adaptation of the R1-Zero RL framework for MT without supervised fine-tuning or cold-start. We propose a rule-metric mixed reward mechanism to guide LLMs towards improved translation quality via emergent reasoning. On the WMT 24 English-Chinese benchmark, our MT-R1-Zero-3B-Mix achieves competitive performance, surpassing TowerInstruct-7B-v0.2 by an average of 1.26 points. Meanwhile, our MT-R1-Zero-7B-Mix attains a high average score of 62.25 across all metrics, placing it on par with advanced proprietary models such as GPT-4o and Claude-3.5-Sonnet, while the MT-R1-Zero-7B-Sem variant achieves state-of-the-art scores on semantic metrics. Moreover, our work exhibits strong generalization capabilities on out-of-distribution MT tasks, robustly supporting multilingual and low-resource settings. Extensive analysis of model behavior across different initializations and reward metrics offers pioneering insight into the critical role of reward design, LLM adaptability, training dynamics, and emergent reasoning patterns within the R1-Zero paradigm for MT. Our code is available at this https URL. 

**Abstract (ZH)**: 大规模强化学习方法在提升大型语言模型的机器翻译性能方面取得了显著效果，特别是在数学和编程等具有可验证解决方案的任务中。然而，将这一理念应用于机器翻译（MT），其中输出的格式灵活且难以通过显式规则自动评估，这一领域仍然有待探索。在本研究中，我们介绍了MT-R1-Zero，这是第一个无需监督微调或冷启动的情况下，将R1-Zero RL框架应用于MT的开源适应版本。我们提出了一种规则度量混合奖励机制，通过新兴推理引导LLM提高翻译质量。在WMT 24英中基准测试中，我们的MT-R1-Zero-3B-Mix实现了竞争力的表现，平均优于TowerInstruct-7B-v0.2 1.26分。同时，我们的MT-R1-Zero-7B-Mix在所有指标上的平均得分为62.25，与GPT-4o和Claude-3.5-Sonnet等先进的专有模型不相上下，MT-R1-Zero-7B-Sem变体在语义指标上取得了最先进的得分。此外，我们的工作在离分布机器翻译任务中展示了强大的泛化能力，支持多语言和低资源设置。对不同初始化和奖励机制下模型行为的广泛分析提供了对奖励设计、LLM适应性、训练动力学和R1-Zero范式下新兴推理模式在MT中的关键作用的开创性见解。我们的代码可在以下链接获取。 

---
# COUNTS: Benchmarking Object Detectors and Multimodal Large Language Models under Distribution Shifts 

**Title (ZH)**: COUNTS: 在分布偏移下检测器和多模态大语言模型的基准测试 

**Authors**: Jiansheng Li, Xingxuan Zhang, Hao Zou, Yige Guo, Renzhe Xu, Yilong Liu, Chuzhao Zhu, Yue He, Peng Cui  

**Link**: [PDF](https://arxiv.org/pdf/2504.10158)  

**Abstract**: Current object detectors often suffer significant perfor-mance degradation in real-world applications when encountering distributional shifts. Consequently, the out-of-distribution (OOD) generalization capability of object detectors has garnered increasing attention from researchers. Despite this growing interest, there remains a lack of a large-scale, comprehensive dataset and evaluation benchmark with fine-grained annotations tailored to assess the OOD generalization on more intricate tasks like object detection and grounding. To address this gap, we introduce COUNTS, a large-scale OOD dataset with object-level annotations. COUNTS encompasses 14 natural distributional shifts, over 222K samples, and more than 1,196K labeled bounding boxes. Leveraging COUNTS, we introduce two novel benchmarks: O(OD)2 and OODG. O(OD)2 is designed to comprehensively evaluate the OOD generalization capabilities of object detectors by utilizing controlled distribution shifts between training and testing data. OODG, on the other hand, aims to assess the OOD generalization of grounding abilities in multimodal large language models (MLLMs). Our findings reveal that, while large models and extensive pre-training data substantially en hance performance in in-distribution (IID) scenarios, significant limitations and opportunities for improvement persist in OOD contexts for both object detectors and MLLMs. In visual grounding tasks, even the advanced GPT-4o and Gemini-1.5 only achieve 56.7% and 28.0% accuracy, respectively. We hope COUNTS facilitates advancements in the development and assessment of robust object detectors and MLLMs capable of maintaining high performance under distributional shifts. 

**Abstract (ZH)**: 当前的目标检测器在遇到分布漂移时往往会出现显著的性能下降，因此目标检测器的离分布外（OOD）泛化能力引起了越来越多研究者的关注。尽管如此，仍缺乏一个大规模、综合且包含细粒度注释的离分布外评估基准，以评估物体检测等复杂任务中的离分布外泛化能力。为填补这一空白，我们介绍了COUNTS，这是一个包含物体级别注释的大规模离分布外数据集。COUNTS 包含了14种自然分布漂移，逾222,000个样本和超过1,196,000个标注边界框。利用COUNTS，我们引入了两个新的基准测试：O(OD)2和OODG。O(OD)2旨在通过在训练数据和测试数据之间使用受控的分布漂移来全面评估物体检测器的离分布外泛化能力。OODG则旨在评估多模态大语言模型（MLLMs）的离分布外语义分割能力。我们的研究发现，虽然大型模型和大量的预训练数据在分布式内（IID）场景中显著提高了性能，但在分布式外（OOD）场景中，物体检测器和MLLMs仍存在显著的局限性和改进机会。在视觉语义分割任务中，即使是先进的GPT-4o和Gemini-1.5也只能分别达到56.7%和28.0%的准确率。我们希望COUNTS能促进稳健物体检测器和MLLMs的发展和评估，这些模型能在分布漂移的情况下保持高性能。 

---
# BoTTA: Benchmarking on-device Test Time Adaptation 

**Title (ZH)**: BoTTA: 在设备上测试时间适应性基准测试 

**Authors**: Michal Danilowski, Soumyajit Chatterjee, Abhirup Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2504.10149)  

**Abstract**: The performance of deep learning models depends heavily on test samples at runtime, and shifts from the training data distribution can significantly reduce accuracy. Test-time adaptation (TTA) addresses this by adapting models during inference without requiring labeled test data or access to the original training set. While research has explored TTA from various perspectives like algorithmic complexity, data and class distribution shifts, model architectures, and offline versus continuous learning, constraints specific to mobile and edge devices remain underexplored. We propose BoTTA, a benchmark designed to evaluate TTA methods under practical constraints on mobile and edge devices. Our evaluation targets four key challenges caused by limited resources and usage conditions: (i) limited test samples, (ii) limited exposure to categories, (iii) diverse distribution shifts, and (iv) overlapping shifts within a sample. We assess state-of-the-art TTA methods under these scenarios using benchmark datasets and report system-level metrics on a real testbed. Furthermore, unlike prior work, we align with on-device requirements by advocating periodic adaptation instead of continuous inference-time adaptation. Experiments reveal key insights: many recent TTA algorithms struggle with small datasets, fail to generalize to unseen categories, and depend on the diversity and complexity of distribution shifts. BoTTA also reports device-specific resource use. For example, while SHOT improves accuracy by $2.25\times$ with $512$ adaptation samples, it uses $1.08\times$ peak memory on Raspberry Pi versus the base model. BoTTA offers actionable guidance for TTA in real-world, resource-constrained deployments. 

**Abstract (ZH)**: 基于移动和边缘设备约束的Test-time适应性基准（BoTTA） 

---
# GeoUni: A Unified Model for Generating Geometry Diagrams, Problems and Problem Solutions 

**Title (ZH)**: GeoUni：生成几何图、问题及解答的统一模型 

**Authors**: Jo-Ku Cheng, Zeren Zhang, Ran Chen, Jingyang Deng, Ziran Qin, Jinwen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.10146)  

**Abstract**: We propose GeoUni, the first unified geometry expert model capable of generating problem solutions and diagrams within a single framework in a way that enables the creation of unique and individualized geometry problems. Traditionally, solving geometry problems and generating diagrams have been treated as separate tasks in machine learning, with no models successfully integrating both to support problem creation. However, we believe that mastery in geometry requires frictionless integration of all of these skills, from solving problems to visualizing geometric relationships, and finally, crafting tailored problems. Our extensive experiments demonstrate that GeoUni, with only 1.5B parameters, achieves performance comparable to larger models such as DeepSeek-R1 with 671B parameters in geometric reasoning tasks. GeoUni also excels in generating precise geometric diagrams, surpassing both text-to-image models and unified models, including the GPT-4o image generation. Most importantly, GeoUni is the only model capable of successfully generating textual problems with matching diagrams based on specific knowledge points, thus offering a wider range of capabilities that extend beyond current models. 

**Abstract (ZH)**: GeoUni：首个能够在单一框架内生成问题解决方案和图表的统一几何专家模型 

---
# Benchmarking Practices in LLM-driven Offensive Security: Testbeds, Metrics, and Experiment Design 

**Title (ZH)**: 基于LLM驱动的进攻性安全实践的基准测试：测试平台、评估指标与实验设计 

**Authors**: Andreas Happe, Jürgen Cito  

**Link**: [PDF](https://arxiv.org/pdf/2504.10112)  

**Abstract**: Large Language Models (LLMs) have emerged as a powerful approach for driving offensive penetration-testing tooling. This paper analyzes the methodology and benchmarking practices used for evaluating Large Language Model (LLM)-driven attacks, focusing on offensive uses of LLMs in cybersecurity. We review 16 research papers detailing 15 prototypes and their respective testbeds.
We detail our findings and provide actionable recommendations for future research, emphasizing the importance of extending existing testbeds, creating baselines, and including comprehensive metrics and qualitative analysis. We also note the distinction between security research and practice, suggesting that CTF-based challenges may not fully represent real-world penetration testing scenarios. 

**Abstract (ZH)**: 大型语言模型（LLMs）已成为驱动进攻性渗透测试工具的强大方法。本文分析了评估大型语言模型（LLM）驱动攻击的方法学和基准测试实践，重点关注LLMs在网络安全中的进攻性使用。我们回顾了16篇研究论文，详细介绍了15个原型及其各自的实验平台。 

---
# Lightweight Trustworthy Distributed Clustering 

**Title (ZH)**: 轻量级可信分布式聚类 

**Authors**: Hongyang Li, Caesar Wu, Mohammed Chadli, Said Mammar, Pascal Bouvry  

**Link**: [PDF](https://arxiv.org/pdf/2504.10109)  

**Abstract**: Ensuring data trustworthiness within individual edge nodes while facilitating collaborative data processing poses a critical challenge in edge computing systems (ECS), particularly in resource-constrained scenarios such as autonomous systems sensor networks, industrial IoT, and smart cities. This paper presents a lightweight, fully distributed k-means clustering algorithm specifically adapted for edge environments, leveraging a distributed averaging approach with additive secret sharing, a secure multiparty computation technique, during the cluster center update phase to ensure the accuracy and trustworthiness of data across nodes. 

**Abstract (ZH)**: 确保在边缘节点上保障数据可信性的同时促进协作数据处理是边缘计算系统（ECS）中的一个关键挑战，特别是在自主系统传感器网络、工业物联网和智慧城市等资源受限的场景中。本文提出了一种专门为边缘环境设计的轻量级全分布式K-means聚类算法，在聚类中心更新阶段利用分布式平均方法与加性秘密共享技术，一种安全多方计算技术，以确保跨节点的数据准确性和可信性。 

---
# SoccerNet-v3D: Leveraging Sports Broadcast Replays for 3D Scene Understanding 

**Title (ZH)**: SoccerNet-v3D: 利用体育广播回放进行三维场景理解 

**Authors**: Marc Gutiérrez-Pérez, Antonio Agudo  

**Link**: [PDF](https://arxiv.org/pdf/2504.10106)  

**Abstract**: Sports video analysis is a key domain in computer vision, enabling detailed spatial understanding through multi-view correspondences. In this work, we introduce SoccerNet-v3D and ISSIA-3D, two enhanced and scalable datasets designed for 3D scene understanding in soccer broadcast analysis. These datasets extend SoccerNet-v3 and ISSIA by incorporating field-line-based camera calibration and multi-view synchronization, enabling 3D object localization through triangulation. We propose a monocular 3D ball localization task built upon the triangulation of ground-truth 2D ball annotations, along with several calibration and reprojection metrics to assess annotation quality on demand. Additionally, we present a single-image 3D ball localization method as a baseline, leveraging camera calibration and ball size priors to estimate the ball's position from a monocular viewpoint. To further refine 2D annotations, we introduce a bounding box optimization technique that ensures alignment with the 3D scene representation. Our proposed datasets establish new benchmarks for 3D soccer scene understanding, enhancing both spatial and temporal analysis in sports analytics. Finally, we provide code to facilitate access to our annotations and the generation pipelines for the datasets. 

**Abstract (ZH)**: 体育视频分析是计算机视觉中的一个关键领域，通过多视角对应关系实现详细的空间理解。本文介绍了SoccerNet-v3D和ISSIA-3D两个增强和可扩展的数据集，用于足球转播分析中的3D场景理解。这些数据集扩展了SoccerNet-v3和ISSIA，通过引入基于场线的摄像机标定和多视角同步，利用三角测量实现3D物体定位。我们提出了一项基于地面真实2D球标注的单目3D球定位任务，并提出了一些校准和重投影指标，以按需评估标注质量。此外，我们提出了一种单图像3D球定位方法作为基线，利用摄像机校准和球大小先验估计单目视角下的球位置。为了进一步细化2D标注，我们引入了一种边界框优化技术，确保与3D场景表示的对齐。我们提出的这些数据集为3D足球场景理解建立了新的基准，增强了体育分析中的空间和时间分析。最后，我们提供了代码以方便访问我们的标注以及数据集的生成管道。 

---
# Towards Quantifying Commonsense Reasoning with Mechanistic Insights 

**Title (ZH)**: 基于机制洞见量化常识推理能力 

**Authors**: Abhinav Joshi, Areeb Ahmad, Divyaksh Shukla, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2504.10077)  

**Abstract**: Commonsense reasoning deals with the implicit knowledge that is well understood by humans and typically acquired via interactions with the world. In recent times, commonsense reasoning and understanding of various LLMs have been evaluated using text-based tasks. In this work, we argue that a proxy of this understanding can be maintained as a graphical structure that can further help to perform a rigorous evaluation of commonsense reasoning abilities about various real-world activities. We create an annotation scheme for capturing this implicit knowledge in the form of a graphical structure for 37 daily human activities. We find that the created resource can be used to frame an enormous number of commonsense queries (~ 10^{17}), facilitating rigorous evaluation of commonsense reasoning in LLMs. Moreover, recently, the remarkable performance of LLMs has raised questions about whether these models are truly capable of reasoning in the wild and, in general, how reasoning occurs inside these models. In this resource paper, we bridge this gap by proposing design mechanisms that facilitate research in a similar direction. Our findings suggest that the reasoning components are localized in LLMs that play a prominent role in decision-making when prompted with a commonsense query. 

**Abstract (ZH)**: 常识推理处理人类普遍理解和通常通过与世界交互获取的隐含知识。近年来，各种大规模语言模型的常识推理和理解能力多通过文本任务进行评估。在本工作中，我们argue rằng这种理解可以通过图形结构来维持，进一步帮助进行常识推理能力的严格评估，关于各种现实生活活动。我们创建了一种标注方案，将以图形结构形式捕获37项日常人类活动中的隐含知识。我们发现，所创建的资源可用来构建大量常识查询（~10^17），促进对大规模语言模型常识推理能力的严格评估。此外，最近大规模语言模型的出色性能引发了对其是否真正能够在现实世界中进行推理以及如何在模型内部进行推理的一系列疑问。在本资源论文中，我们通过提出促进类似方向研究的设计机制来弥合这一差距。我们的发现表明，当面对常识查询时，LLMs中在决策过程中发挥重要作用的推理组件是局部化的。 

---
# Mavors: Multi-granularity Video Representation for Multimodal Large Language Model 

**Title (ZH)**: Mavors: 多粒度视频表示用于多模态大型语言模型 

**Authors**: Yang Shi, Jiaheng Liu, Yushuo Guan, Zhenhua Wu, Yuanxing Zhang, Zihao Wang, Weihong Lin, Jingyun Hua, Zekun Wang, Xinlong Chen, Bohan Zeng, Wentao Zhang, Fuzheng Zhang, Wenjing Yang, Di Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10068)  

**Abstract**: Long-context video understanding in multimodal large language models (MLLMs) faces a critical challenge: balancing computational efficiency with the retention of fine-grained spatio-temporal patterns. Existing approaches (e.g., sparse sampling, dense sampling with low resolution, and token compression) suffer from significant information loss in temporal dynamics, spatial details, or subtle interactions, particularly in videos with complex motion or varying resolutions. To address this, we propose $\mathbf{Mavors}$, a novel framework that introduces $\mathbf{M}$ulti-gr$\mathbf{a}$nularity $\mathbf{v}$ide$\mathbf{o}$ $\mathbf{r}$epre$\mathbf{s}$entation for holistic long-video modeling. Specifically, Mavors directly encodes raw video content into latent representations through two core components: 1) an Intra-chunk Vision Encoder (IVE) that preserves high-resolution spatial features via 3D convolutions and Vision Transformers, and 2) an Inter-chunk Feature Aggregator (IFA) that establishes temporal coherence across chunks using transformer-based dependency modeling with chunk-level rotary position encodings. Moreover, the framework unifies image and video understanding by treating images as single-frame videos via sub-image decomposition. Experiments across diverse benchmarks demonstrate Mavors' superiority in maintaining both spatial fidelity and temporal continuity, significantly outperforming existing methods in tasks requiring fine-grained spatio-temporal reasoning. 

**Abstract (ZH)**: 多模态大规模语言模型中长上下文视频理解面临的挑战：在保持计算效率与保留精细时空模式之间的平衡。为解决这一问题，我们提出了Mavors框架，该框架引入了多粒度视频表示以进行整体长视频建模。具体而言，Mavors通过两种核心组件直接将原始视频内容编码为潜在表示：1) 内存块视觉编码器（IVE），利用3D卷积和视觉变换器保留高分辨率的空间特征；2) 内存块特征聚合器（IFA），利用基于转子位置编码的变换器依赖建模在内存块之间建立时空一致性。此外，该框架通过子图像分解将图像视为单帧视频来统一图像和视频理解。跨多种基准的实验结果表明，Mavors在保持空间保真度和时间连续性方面优于现有方法，在需要精细时空推理的任务中表现更优。 

---
# Hallucination Detection in LLMs via Topological Divergence on Attention Graphs 

**Title (ZH)**: LLMs中基于注意力图拓扑发散的幻觉检测 

**Authors**: Alexandra Bazarova, Aleksandr Yugay, Andrey Shulga, Alina Ermilova, Andrei Volodichev, Konstantin Polev, Julia Belikova, Rauf Parchiev, Dmitry Simakov, Maxim Savchenko, Andrey Savchenko, Serguei Barannikov, Alexey Zaytsev  

**Link**: [PDF](https://arxiv.org/pdf/2504.10063)  

**Abstract**: Hallucination, i.e., generating factually incorrect content, remains a critical challenge for large language models (LLMs). We introduce TOHA, a TOpology-based HAllucination detector in the RAG setting, which leverages a topological divergence metric to quantify the structural properties of graphs induced by attention matrices. Examining the topological divergence between prompt and response subgraphs reveals consistent patterns: higher divergence values in specific attention heads correlate with hallucinated outputs, independent of the dataset. Extensive experiments, including evaluation on question answering and data-to-text tasks, show that our approach achieves state-of-the-art or competitive results on several benchmarks, two of which were annotated by us and are being publicly released to facilitate further research. Beyond its strong in-domain performance, TOHA maintains remarkable domain transferability across multiple open-source LLMs. Our findings suggest that analyzing the topological structure of attention matrices can serve as an efficient and robust indicator of factual reliability in LLMs. 

**Abstract (ZH)**: 基于拓扑的幻觉检测器TOHA：在RAG设置中的幻觉检测 

---
# EmbodiedAgent: A Scalable Hierarchical Approach to Overcome Practical Challenge in Multi-Robot Control 

**Title (ZH)**: 具身代理：一种克服多机器人控制实践挑战的可扩展分层方法 

**Authors**: Hanwen Wan, Yifei Chen, Zeyu Wei, Dongrui Li, Zexin Lin, Donghao Wu, Jiu Cheng, Yuxiang Zhang, Xiaoqiang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2504.10030)  

**Abstract**: This paper introduces EmbodiedAgent, a hierarchical framework for heterogeneous multi-robot control. EmbodiedAgent addresses critical limitations of hallucination in impractical tasks. Our approach integrates a next-action prediction paradigm with a structured memory system to decompose tasks into executable robot skills while dynamically validating actions against environmental constraints. We present MultiPlan+, a dataset of more than 18,000 annotated planning instances spanning 100 scenarios, including a subset of impractical cases to mitigate hallucination. To evaluate performance, we propose the Robot Planning Assessment Schema (RPAS), combining automated metrics with LLM-aided expert grading. Experiments demonstrate EmbodiedAgent's superiority over state-of-the-art models, achieving 71.85% RPAS score. Real-world validation in an office service task highlights its ability to coordinate heterogeneous robots for long-horizon objectives. 

**Abstract (ZH)**: 本文介绍了EmbodiedAgent，一种异构多机器人控制的分层框架。EmbodiedAgent解决了不切实际任务中的幻觉关键限制。我们的方法结合了下一步动作预测范式与结构化记忆系统，将任务分解为可执行的机器人技能，并动态验证动作是否符合环境约束。我们提出了包含超过18,000个标注计划实例的MultiPlan+数据集，这些实例覆盖了100个场景，包括某些不切实际的案例以减轻幻觉。为评估性能，我们提出了一种机器人计划评估方案（RPAS），结合了自动评价指标与LLM辅助专家评分。实验结果表明，EmbodiedAgent在性能上优于当前最先进的模型，达到了71.85%的RPAS分数。在办公室服务任务的实际验证中，展示了其协调异构机器人完成长期目标的能力。 

---
# Sequence models for by-trial decoding of cognitive strategies from neural data 

**Title (ZH)**: 基于试次解码的认知策略神经数据序列模型 

**Authors**: Rick den Otter, Gabriel Weindel, Sjoerd Stuit, Leendert van Maanen  

**Link**: [PDF](https://arxiv.org/pdf/2504.10028)  

**Abstract**: Understanding the sequence of cognitive operations that underlie decision-making is a fundamental challenge in cognitive neuroscience. Traditional approaches often rely on group-level statistics, which obscure trial-by-trial variations in cognitive strategies. In this study, we introduce a novel machine learning method that combines Hidden Multivariate Pattern analysis with a Structured State Space Sequence model to decode cognitive strategies from electroencephalography data at the trial level. We apply this method to a decision-making task, where participants were instructed to prioritize either speed or accuracy in their responses. Our results reveal an additional cognitive operation, labeled Confirmation, which seems to occur predominantly in the accuracy condition but also frequently in the speed condition. The modeled probability that this operation occurs is associated with higher probability of responding correctly as well as changes of mind, as indexed by electromyography data. By successfully modeling cognitive operations at the trial level, we provide empirical evidence for dynamic variability in decision strategies, challenging the assumption of homogeneous cognitive processes within experimental conditions. Our approach shows the potential of sequence modeling in cognitive neuroscience to capture trial-level variability that is obscured by aggregate analyses. The introduced method offers a new way to detect and understand cognitive strategies in a data-driven manner, with implications for both theoretical research and practical applications in many fields. 

**Abstract (ZH)**: 理解认知操作序列以阐明决策过程是认知神经科学中的基本挑战。传统方法通常依赖于组级统计，这会掩盖每次试次中认知策略的变化。在本研究中，我们引入了一种新的机器学习方法，结合隐藏多元模式分析与结构状态空间序列模型，从脑电图数据中在试次级别解码认知策略。我们在一个决策任务中应用了这种方法，要求参与者在响应时优先考虑速度或准确性。我们的结果揭示了一种额外的认知操作，命名为确认，其主要发生在准确性条件中，但在速度条件中也很频繁。该操作发生的模型概率与正确反应的概率以及通过肌电图数据指数的心理改变有关。通过在试次级别成功建模认知操作，我们提供了内部分布不均的认知过程的实证证据，挑战了实验条件下认知过程均质性的假设。我们的方法展示了序列建模在认知神经科学中捕捉由聚合分析掩盖的试次级变化的潜力。所介绍的方法为以数据驱动的方式检测和理解认知策略提供了新途径，对理论研究和许多领域的实际应用具有重要意义。 

---
# Progressive Transfer Learning for Multi-Pass Fundus Image Restoration 

**Title (ZH)**: 多过重重归图像 restoration 的逐级迁移学习 

**Authors**: Uyen Phan, Ozer Can Devecioglu, Serkan Kiranyaz, Moncef Gabbouj  

**Link**: [PDF](https://arxiv.org/pdf/2504.10025)  

**Abstract**: Diabetic retinopathy is a leading cause of vision impairment, making its early diagnosis through fundus imaging critical for effective treatment planning. However, the presence of poor quality fundus images caused by factors such as inadequate illumination, noise, blurring and other motion artifacts yields a significant challenge for accurate DR screening. In this study, we propose progressive transfer learning for multi pass restoration to iteratively enhance the quality of degraded fundus images, ensuring more reliable DR screening. Unlike previous methods that often focus on a single pass restoration, multi pass restoration via PTL can achieve a superior blind restoration performance that can even improve most of the good quality fundus images in the dataset. Initially, a Cycle GAN model is trained to restore low quality images, followed by PTL induced restoration passes over the latest restored outputs to improve overall quality in each pass. The proposed method can learn blind restoration without requiring any paired data while surpassing its limitations by leveraging progressive learning and fine tuning strategies to minimize distortions and preserve critical retinal features. To evaluate PTL's effectiveness on multi pass restoration, we conducted experiments on DeepDRiD, a large scale fundus imaging dataset specifically curated for diabetic retinopathy detection. Our result demonstrates state of the art performance, showcasing PTL's potential as a superior approach to iterative image quality restoration. 

**Abstract (ZH)**: 糖尿病视网膜病变是导致视力损害的主要原因之一，通过眼底成像进行早期诊断对于有效治疗计划至关重要。然而，由于照明不足、噪声、模糊和其他运动伪影等原因导致的眼底图像质量差，对准确的糖尿病视网膜病变筛查构成了重大挑战。在本研究中，我们提出渐进迁移学习进行多遍恢复，以迭代增强退化眼底图像的质量，确保更可靠的糖尿病视网膜病变筛查。与仅关注单遍恢复的先前方法不同，通过渐进迁移学习（PTL）进行多遍恢复可以实现卓越的盲恢复性能，甚至可以改善数据集中大多数高品质眼底图像。首先，训练一个Cycle GAN模型以恢复低质量图像，然后通过PTL诱导的恢复遍历来提高每次迭代的整体质量。所提出的方法无需任何配对数据即可学习盲恢复，并通过利用渐进学习和 fine-tuning 策略来最小化失真并保留关键的视网膜特征。为了评估PTL在多遍恢复中的效果，我们在专门为糖尿病视网膜病变检测定制的DeepDRiD大数据集上进行了实验。我们的结果显示了最先进的性能，展示了PTL作为迭代图像质量恢复的优越方法的潜力。 

---
# The Mirage of Performance Gains: Why Contrastive Decoding Fails to Address Multimodal Hallucination 

**Title (ZH)**: 性能提升的幻影：对比解码为何无法解决多模态幻觉 

**Authors**: Hao Yin, Gunagzong Si, Zilei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10020)  

**Abstract**: Contrastive decoding strategies are widely used to reduce hallucinations in multimodal large language models (MLLMs). These methods work by constructing contrastive samples to induce hallucinations and then suppressing them in the output distribution. However, this paper demonstrates that such approaches fail to effectively mitigate the hallucination problem. The performance improvements observed on POPE Benchmark are largely driven by two misleading factors: (1) crude, unidirectional adjustments to the model's output distribution and (2) the adaptive plausibility constraint, which reduces the sampling strategy to greedy search. To further illustrate these issues, we introduce a series of spurious improvement methods and evaluate their performance against contrastive decoding techniques. Experimental results reveal that the observed performance gains in contrastive decoding are entirely unrelated to its intended goal of mitigating hallucinations. Our findings challenge common assumptions about the effectiveness of contrastive decoding strategies and pave the way for developing genuinely effective solutions to hallucinations in MLLMs. 

**Abstract (ZH)**: 对比解码策略在多模态大型语言模型中广泛应用以减少幻觉现象，但这类方法未能有效缓解幻觉问题。POPE基准上的性能提升主要由两个误导性因素驱动：（1）模型输出分布的一系列粗略且单向调整，（2）自适应可行性约束，将采样策略简化为贪婪搜索。为进一步说明这些问题，我们引入了一系列虚假改进方法，并评估其性能与对比解码技术的对比结果。实验结果显示，对比解码策略观察到的性能提升与减轻幻觉的目标完全无关。我们的发现质疑了对比解码策略有效性的常见假设，并为开发真正有效的多模态大型语言模型幻觉解决方案铺平了道路。 

---
# RGB-Event based Pedestrian Attribute Recognition: A Benchmark Dataset and An Asymmetric RWKV Fusion Framework 

**Title (ZH)**: 基于RGB-事件的行人属性识别：一个基准数据集和一种不对称RWKV融合框架 

**Authors**: Xiao Wang, Haiyang Wang, Shiao Wang, Qiang Chen, Jiandong Jin, Haoyu Song, Bo Jiang, Chenglong Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.10018)  

**Abstract**: Existing pedestrian attribute recognition methods are generally developed based on RGB frame cameras. However, these approaches are constrained by the limitations of RGB cameras, such as sensitivity to lighting conditions and motion blur, which hinder their performance. Furthermore, current attribute recognition primarily focuses on analyzing pedestrians' external appearance and clothing, lacking an exploration of emotional dimensions. In this paper, we revisit these issues and propose a novel multi-modal RGB-Event attribute recognition task by drawing inspiration from the advantages of event cameras in low-light, high-speed, and low-power consumption. Specifically, we introduce the first large-scale multi-modal pedestrian attribute recognition dataset, termed EventPAR, comprising 100K paired RGB-Event samples that cover 50 attributes related to both appearance and six human emotions, diverse scenes, and various seasons. By retraining and evaluating mainstream PAR models on this dataset, we establish a comprehensive benchmark and provide a solid foundation for future research in terms of data and algorithmic baselines. In addition, we propose a novel RWKV-based multi-modal pedestrian attribute recognition framework, featuring an RWKV visual encoder and an asymmetric RWKV fusion module. Extensive experiments are conducted on our proposed dataset as well as two simulated datasets (MARS-Attribute and DukeMTMC-VID-Attribute), achieving state-of-the-art results. The source code and dataset will be released on this https URL 

**Abstract (ZH)**: 现有的行人属性识别方法主要基于RGB帧相机开发。然而，这些方法受限于RGB相机的照明条件敏感性和运动模糊等问题，影响了其性能。此外，当前的属性识别主要集中在分析行人的外部外观和穿着上，缺乏对情感维度的探索。本文重新审视了这些问题，并从事件摄像头在低光照、高速度和低功耗方面的优势出发，提出了一种新型的多模态RGB-事件属性识别任务。具体地，我们引入了第一个大规模的多模态行人属性识别数据集EventPAR，包含100K对RGB-事件配对样本，涵盖了与50种外观和六种人类情感、多种场景和不同季节相关的属性。通过对这一数据集进行重训练和评估主流的人行道属性识别模型，我们建立了一个全面的基准，为未来的研究提供了坚实的数据和算法基线。此外，我们提出了一种基于RWKV的新型多模态行人属性识别框架，该框架包括RWKV视觉编码器和不对称RWKV融合模块。我们在提出的数据集以及两个模拟数据集（MARS-Attribute和DukeMTMC-VID-Attribute）上进行了广泛的实验，取得了当前最佳的结果。源代码和数据集将发布在https://github.com/username/repo。 

---
# Air Quality Prediction with A Meteorology-Guided Modality-Decoupled Spatio-Temporal Network 

**Title (ZH)**: 气象引导的模态解耦空时网络空气质量预测 

**Authors**: Hang Yin, Yan-Ming Zhang, Jian Xu, Jian-Long Chang, Yin Li, Cheng-Lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.10014)  

**Abstract**: Air quality prediction plays a crucial role in public health and environmental protection. Accurate air quality prediction is a complex multivariate spatiotemporal problem, that involves interactions across temporal patterns, pollutant correlations, spatial station dependencies, and particularly meteorological influences that govern pollutant dispersion and chemical transformations. Existing works underestimate the critical role of atmospheric conditions in air quality prediction and neglect comprehensive meteorological data utilization, thereby impairing the modeling of dynamic interdependencies between air quality and meteorological data. To overcome this, we propose MDSTNet, an encoder-decoder framework that explicitly models air quality observations and atmospheric conditions as distinct modalities, integrating multi-pressure-level meteorological data and weather forecasts to capture atmosphere-pollution dependencies for prediction. Meantime, we construct ChinaAirNet, the first nationwide dataset combining air quality records with multi-pressure-level meteorological observations. Experimental results on ChinaAirNet demonstrate MDSTNet's superiority, substantially reducing 48-hour prediction errors by 17.54\% compared to the state-of-the-art model. The source code and dataset will be available on github. 

**Abstract (ZH)**: 空气质量预测在公共卫生和环境保护中起着至关重要的作用。准确的空气质量预测是一个复杂的多变量时空问题，涉及时间模式、污染物质之间的相互作用、站点之间的空间依赖性，以及特别重要的气象条件对污染物扩散和化学转化的调控。现有研究低估了大气条件在空气质量预测中的关键作用，忽视了全面利用气象数据，从而影响了空气质量与气象数据之间动态依赖性的建模。为了解决这一问题，我们提出MDSTNet，这是一种将空气质量观察与大气条件明确建模的编码-解码框架，整合多气压级气象数据和天气预报，以捕捉大气-污染依赖性进行预测。同时，我们构建了中国首个将空气质量记录与多气压级气象观测结合的ChinaAirNet数据集。在ChinaAirNet上的实验结果表明，MDSTNet的优越性，相比最先进的模型显著降低了48小时预测误差17.54%。源代码和数据集将在GitHub上提供。 

---
# Session-based Recommender Systems: User Interest as a Stochastic Process in the Latent Space 

**Title (ZH)**: 基于会话的推荐系统：用户兴趣在潜在空间中的随机过程 

**Authors**: Klaudia Balcer, Piotr Lipinski  

**Link**: [PDF](https://arxiv.org/pdf/2504.10005)  

**Abstract**: This paper jointly addresses the problem of data uncertainty, popularity bias, and exposure bias in session-based recommender systems. We study the symptoms of this bias both in item embeddings and in recommendations. We propose treating user interest as a stochastic process in the latent space and providing a model-agnostic implementation of this mathematical concept. The proposed stochastic component consists of elements: debiasing item embeddings with regularization for embedding uniformity, modeling dense user interest from session prefixes, and introducing fake targets in the data to simulate extended exposure. We conducted computational experiments on two popular benchmark datasets, Diginetica and YooChoose 1/64, as well as several modifications of the YooChoose dataset with different ratios of popular items. The results show that the proposed approach allows us to mitigate the challenges mentioned. 

**Abstract (ZH)**: 本文联合解决了会话推荐系统中的数据不确定性、流行度偏差和曝光偏差问题。我们研究了这种偏差在项目嵌入和推荐中的症状。我们建议将用户兴趣视为潜空间中的随机过程，并提供了一个模型无关的该数学概念的实现方法。提出的随机成分包括：通过嵌入一致性正则化去偏差项目嵌入、从会话前缀建模密集用户兴趣以及在数据中引入假目标以模拟延长曝光。我们在两个流行的基准数据集Diginetica和YooChoose 1/64，以及具有不同受欢迎项目比例的YooChoose数据集的多个变体上进行了计算实验。结果显示，所提出的方法能够缓解上述挑战。 

---
# Do We Really Need Curated Malicious Data for Safety Alignment in Multi-modal Large Language Models? 

**Title (ZH)**: 我们真的需要精心挑选的恶意数据来实现多模态大型语言模型的安全对齐吗？ 

**Authors**: Yanbo Wang, Jiyang Guan, Jian Liang, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2504.10000)  

**Abstract**: Multi-modal large language models (MLLMs) have made significant progress, yet their safety alignment remains limited. Typically, current open-source MLLMs rely on the alignment inherited from their language module to avoid harmful generations. However, the lack of safety measures specifically designed for multi-modal inputs creates an alignment gap, leaving MLLMs vulnerable to vision-domain attacks such as typographic manipulation. Current methods utilize a carefully designed safety dataset to enhance model defense capability, while the specific knowledge or patterns acquired from the high-quality dataset remain unclear. Through comparison experiments, we find that the alignment gap primarily arises from data distribution biases, while image content, response quality, or the contrastive behavior of the dataset makes little contribution to boosting multi-modal safety. To further investigate this and identify the key factors in improving MLLM safety, we propose finetuning MLLMs on a small set of benign instruct-following data with responses replaced by simple, clear rejection sentences. Experiments show that, without the need for labor-intensive collection of high-quality malicious data, model safety can still be significantly improved, as long as a specific fraction of rejection data exists in the finetuning set, indicating the security alignment is not lost but rather obscured during multi-modal pretraining or instruction finetuning. Simply correcting the underlying data bias could narrow the safety gap in the vision domain. 

**Abstract (ZH)**: 多模态大型语言模型的安全对齐进展有限：数据偏差是主要因素 

---
# Metric-Guided Synthesis of Class Activation Mapping 

**Title (ZH)**: 基于度量的类激活映射合成 

**Authors**: Alejandro Luque-Cerpa, Elizabeth Polgreen, Ajitha Rajan, Hazem Torfah  

**Link**: [PDF](https://arxiv.org/pdf/2504.09998)  

**Abstract**: Class activation mapping (CAM) is a widely adopted class of saliency methods used to explain the behavior of convolutional neural networks (CNNs). These methods generate heatmaps that highlight the parts of the input most relevant to the CNN output. Various CAM methods have been proposed, each distinguished by the expressions used to derive heatmaps. In general, users look for heatmaps with specific properties that reflect different aspects of CNN functionality. These may include similarity to ground truth, robustness, equivariance, and more. Although existing CAM methods implicitly encode some of these properties in their expressions, they do not allow for variability in heatmap generation following the user's intent or domain knowledge. In this paper, we address this limitation by introducing SyCAM, a metric-based approach for synthesizing CAM expressions. Given a predefined evaluation metric for saliency maps, SyCAM automatically generates CAM expressions optimized for that metric. We specifically explore a syntax-guided synthesis instantiation of SyCAM, where CAM expressions are derived based on predefined syntactic constraints and the given metric. Using several established evaluation metrics, we demonstrate the efficacy and flexibility of our approach in generating targeted heatmaps. We compare SyCAM with other well-known CAM methods on three prominent models: ResNet50, VGG16, and VGG19. 

**Abstract (ZH)**: 基于度量的SynCAM合成方法：针对特定评价指标自动生成CAM表达式 

---
# GenTe: Generative Real-world Terrains for General Legged Robot Locomotion Control 

**Title (ZH)**: GenTe: 生成的现实地形用于通用腿足机器人运动控制 

**Authors**: Hanwen Wan, Mengkang Li, Donghao Wu, Yebin Zhong, Yixuan Deng, Zhenglong Sun, Xiaoqiang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2504.09997)  

**Abstract**: Developing bipedal robots capable of traversing diverse real-world terrains presents a fundamental robotics challenge, as existing methods using predefined height maps and static environments fail to address the complexity of unstructured landscapes. To bridge this gap, we propose GenTe, a framework for generating physically realistic and adaptable terrains to train generalizable locomotion policies. GenTe constructs an atomic terrain library that includes both geometric and physical terrains, enabling curriculum training for reinforcement learning-based locomotion policies. By leveraging function-calling techniques and reasoning capabilities of Vision-Language Models (VLMs), GenTe generates complex, contextually relevant terrains from textual and graphical inputs. The framework introduces realistic force modeling for terrain interactions, capturing effects such as soil sinkage and hydrodynamic resistance. To the best of our knowledge, GenTe is the first framework that systemically generates simulation environments for legged robot locomotion control. Additionally, we introduce a benchmark of 100 generated terrains. Experiments demonstrate improved generalization and robustness in bipedal robot locomotion. 

**Abstract (ZH)**: 开发能够在多样化真实地形中行进的双足机器人是机器人学领域的一项基本挑战，现有使用预定义高度图和静态环境的方法无法应对未结构化地形的复杂性。为解决这一问题，我们提出了一种名为GenTe的框架，用于生成物理上真实且可适应的地形以训练可泛化的运动策略。GenTe构建了一个包含几何和物理地形的原子地形库，支持基于强化学习的运动策略的课程训练。通过利用函数调用技术和视觉-语言模型的推理能力，GenTe能够从文本和图形输入中生成复杂且上下文相关的真实地形。该框架引入了真实的力模型来模拟地形交互，捕捉诸如土壤压缩和水动力阻力等效应。据我们所知，GenTe是首个系统性生成用于腿足机器人运动控制的模拟环境的框架。此外，我们还引入了一个包含100个生成地形的基准。实验结果表明，GenTe能够提高双足机器人运动的泛化能力和鲁棒性。 

---
# Enhancing Multi-task Learning Capability of Medical Generalist Foundation Model via Image-centric Multi-annotation Data 

**Title (ZH)**: 通过基于图像的多标注数据增强医疗全科基础模型的多任务学习能力 

**Authors**: Xun Zhu, Fanbin Mo, Zheng Zhang, Jiaxi Wang, Yiming Shi, Ming Wu, Chuang Zhang, Miao Li, Ji Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.09967)  

**Abstract**: The emergence of medical generalist foundation models has revolutionized conventional task-specific model development paradigms, aiming to better handle multiple tasks through joint training on large-scale medical datasets. However, recent advances prioritize simple data scaling or architectural component enhancement, while neglecting to re-examine multi-task learning from a data-centric perspective. Critically, simply aggregating existing data resources leads to decentralized image-task alignment, which fails to cultivate comprehensive image understanding or align with clinical needs for multi-dimensional image interpretation. In this paper, we introduce the image-centric multi-annotation X-ray dataset (IMAX), the first attempt to enhance the multi-task learning capabilities of medical multi-modal large language models (MLLMs) from the data construction level. To be specific, IMAX is featured from the following attributes: 1) High-quality data curation. A comprehensive collection of more than 354K entries applicable to seven different medical tasks. 2) Image-centric dense annotation. Each X-ray image is associated with an average of 4.10 tasks and 7.46 training entries, ensuring multi-task representation richness per image. Compared to the general decentralized multi-annotation X-ray dataset (DMAX), IMAX consistently demonstrates significant multi-task average performance gains ranging from 3.20% to 21.05% across seven open-source state-of-the-art medical MLLMs. Moreover, we investigate differences in statistical patterns exhibited by IMAX and DMAX training processes, exploring potential correlations between optimization dynamics and multi-task performance. Finally, leveraging the core concept of IMAX data construction, we propose an optimized DMAX-based training strategy to alleviate the dilemma of obtaining high-quality IMAX data in practical scenarios. 

**Abstract (ZH)**: 面向图像的多注释X射线数据集（IMAX）：从数据构建层面增强医学多模态大型语言模型的多任务学习能力 

---
# Towards Unbiased Federated Graph Learning: Label and Topology Perspectives 

**Title (ZH)**: 面向无偏 federated 图学习：从标签和拓扑视角探讨 

**Authors**: Zhengyu Wu, Boyang Pang, Xunkai Li, Yinlin Zhu, Daohan Su, Bowen Fan, Rong-Hua Li, Guoren Wang, Chenghu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.09963)  

**Abstract**: Federated Graph Learning (FGL) enables privacy-preserving, distributed training of graph neural networks without sharing raw data. Among its approaches, subgraph-FL has become the dominant paradigm, with most work focused on improving overall node classification accuracy. However, these methods often overlook fairness due to the complexity of node features, labels, and graph structures. In particular, they perform poorly on nodes with disadvantaged properties, such as being in the minority class within subgraphs or having heterophilous connections (neighbors with dissimilar labels or misleading features). This reveals a critical issue: high accuracy can mask degraded performance on structurally or semantically marginalized nodes. To address this, we advocate for two fairness goals: (1) improving representation of minority class nodes for class-wise fairness and (2) mitigating topological bias from heterophilous connections for topology-aware fairness. We propose FairFGL, a novel framework that enhances fairness through fine-grained graph mining and collaborative learning. On the client side, the History-Preserving Module prevents overfitting to dominant local classes, while the Majority Alignment Module refines representations of heterophilous majority-class nodes. The Gradient Modification Module transfers minority-class knowledge from structurally favorable clients to improve fairness. On the server side, FairFGL uploads only the most influenced subset of parameters to reduce communication costs and better reflect local distributions. A cluster-based aggregation strategy reconciles conflicting updates and curbs global majority dominance . Extensive evaluations on eight benchmarks show FairFGL significantly improves minority-group performance , achieving up to a 22.62 percent Macro-F1 gain while enhancing convergence over state-of-the-art baselines. 

**Abstract (ZH)**: 联邦图学习（FGL） enables 保护隐私的分布式图神经网络训练而无需共享原始数据。其方法中，子图-FL已成为主导范式，大多数工作专注于提高整体节点分类准确性。然而，这些方法往往由于节点特征、标签和图结构的复杂性而忽视了公平性。特别是，它们在具有不利属性的节点上表现不佳，例如子图中处于少数类别的节点或具有异质连接（邻居具有相似标签或误导性特征的节点）。这揭示了一个关键问题：高准确性可能掩盖了在结构上或语义上边缘化的节点上的性能下降。为了解决这个问题，我们提倡两个公平目标：（1）通过类别公平性改进少数类节点的表示，（2）通过拓扑感知公平性减轻来自异质连接的拓扑偏差。我们提出FairFGL，一种通过细粒度图挖掘和协作学习增强公平性的新颖框架。在客户端，历史保留模块防止对主导局部类别的过度拟合，而多数对齐模块细化少数类节点的表示。梯度修改模块将少数类知识从结构上有利的客户端转移以提高公平性。在服务器端，FairFGL只上传受最大影响的参数子集以降低通信成本并更好地反映局部分布。基于集群的聚合策略解决冲突更新并遏制全局多数群体优势。在八个基准上的广泛评估表明，FairFGL显着改善了少数群体的表现，在增强与最新 baseline 相比的收敛性的同时，宏F1度量提高了22.62个百分点。 

---
# Privacy Meets Explainability: Managing Confidential Data and Transparency Policies in LLM-Empowered Science 

**Title (ZH)**: 隐私与可解释性相遇：LLM赋能科学中的保密数据管理与透明政策实现 

**Authors**: Yashothara Shanmugarasa, Shidong Pan, Ming Ding, Dehai Zhao, Thierry Rakotoarivelo  

**Link**: [PDF](https://arxiv.org/pdf/2504.09961)  

**Abstract**: As Large Language Models (LLMs) become integral to scientific workflows, concerns over the confidentiality and ethical handling of confidential data have emerged. This paper explores data exposure risks through LLM-powered scientific tools, which can inadvertently leak confidential information, including intellectual property and proprietary data, from scientists' perspectives. We propose "DataShield", a framework designed to detect confidential data leaks, summarize privacy policies, and visualize data flow, ensuring alignment with organizational policies and procedures. Our approach aims to inform scientists about data handling practices, enabling them to make informed decisions and protect sensitive information. Ongoing user studies with scientists are underway to evaluate the framework's usability, trustworthiness, and effectiveness in tackling real-world privacy challenges. 

**Abstract (ZH)**: 大型语言模型（LLMs）在科学工作流程中的应用引发了对保密数据和伦理处理的担忧。本文探讨了由LLM支持的科学工具带来的数据暴露风险，从科学家的角度分析了无意中泄露机密信息（包括知识产权和专有数据）的可能性。我们提出了“DataShield”框架，该框架旨在检测机密数据泄露、总结隐私政策并可视化数据流，确保与组织政策和程序的吻合。我们的方法旨在告知科学家有关数据处理的做法，帮助他们做出知情决策并保护敏感信息。正在进行的科学家用户研究旨在评估该框架的可用性、可信度及其在解决实际隐私挑战方面的有效性。 

---
# Omni-Dish: Photorealistic and Faithful Image Generation and Editing for Arbitrary Chinese Dishes 

**Title (ZH)**: 全盘：任意中式菜品的逼真生成与编辑 

**Authors**: Huijie Liu, Bingcan Wang, Jie Hu, Xiaoming Wei, Guoliang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09948)  

**Abstract**: Dish images play a crucial role in the digital era, with the demand for culturally distinctive dish images continuously increasing due to the digitization of the food industry and e-commerce. In general cases, existing text-to-image generation models excel in producing high-quality images; however, they struggle to capture diverse characteristics and faithful details of specific domains, particularly Chinese dishes. To address this limitation, we propose Omni-Dish, the first text-to-image generation model specifically tailored for Chinese dishes. We develop a comprehensive dish curation pipeline, building the largest dish dataset to date. Additionally, we introduce a recaption strategy and employ a coarse-to-fine training scheme to help the model better learn fine-grained culinary nuances. During inference, we enhance the user's textual input using a pre-constructed high-quality caption library and a large language model, enabling more photorealistic and faithful image generation. Furthermore, to extend our model's capability for dish editing tasks, we propose Concept-Enhanced P2P. Based on this approach, we build a dish editing dataset and train a specialized editing model. Extensive experiments demonstrate the superiority of our methods. 

**Abstract (ZH)**: dishes 图像在数字时代扮演着重要角色，随着食品行业和电子商务的数字化，对具有文化特色的菜肴图像需求不断增加。一般情况下，现有的文本到图像生成模型在生成高质量图像方面表现出色；然而，它们在捕捉特定领域，尤其是 Chinese 菜肴的多样特征和忠实细节方面存在局限性。为了解决这一问题，我们提出了 Omni-Dish，这是首个专门针对 Chinese 菜肴的文本到图像生成模型。我们开发了一个全面的菜肴策展流水线，构建了迄今为止最大的菜肴数据集。此外，我们引入了重新 caption 的策略，并采用了粗到细的训练方案，以帮助模型更好地学习细粒度的烹饪细微差别。在推理过程中，我们使用预构建的高质量图像描述库和大型语言模型增强用户的文本输入，从而实现更逼真和忠实的图像生成。为进一步扩展我们的模型在菜肴编辑任务的能力，我们提出了 Concept-Enhanced P2P。基于此方法，我们构建了一个菜肴编辑数据集并训练了一个专门的编辑模型。大量实验表明了我们方法的优越性。 

---
# FedRecon: Missing Modality Reconstruction in Distributed Heterogeneous Environments 

**Title (ZH)**: FedRecon：分布式异构环境中缺失模态的重构 

**Authors**: Junming Liu, Guosun Zeng, Ding Wang, Yanting Gao, Yufei Jin  

**Link**: [PDF](https://arxiv.org/pdf/2504.09941)  

**Abstract**: Multimodal data are often incomplete and exhibit Non-Independent and Identically Distributed (Non-IID) characteristics in real-world scenarios. These inherent limitations lead to both modality heterogeneity through partial modality absence and data heterogeneity from distribution divergence, creating fundamental challenges for effective federated learning (FL). To address these coupled challenges, we propose FedRecon, the first method targeting simultaneous missing modality reconstruction and Non-IID adaptation in multimodal FL. Our approach first employs a lightweight Multimodal Variational Autoencoder (MVAE) to reconstruct missing modalities while preserving cross-modal consistency. Distinct from conventional imputation methods, we achieve sample-level alignment through a novel distribution mapping mechanism that guarantees both data consistency and completeness. Additionally, we introduce a strategy employing global generator freezing to prevent catastrophic forgetting, which in turn mitigates Non-IID fluctuations. Extensive evaluations on multimodal datasets demonstrate FedRecon's superior performance in modality reconstruction under Non-IID conditions, surpassing state-of-the-art methods. 

**Abstract (ZH)**: 多模态数据在现实场景中往往不完整并表现出非独立同分布（Non-IID）特性。这些固有的限制导致了模态异质性和数据异质性的耦合挑战，给有效的 federated learning（联邦学习）带来根本性的难题。为应对这些挑战，我们提出了 FedRecon，这是首个同时针对多模态 federated learning 中缺失模态重建和 Non-IID 调适的方法。我们的方法首先采用一种轻量级的多模态变分自编码器（MVAE）来重建缺失的模态并保持跨模态一致性。不同于传统的插补方法，我们通过一种新颖的分布映射机制实现了样本级对齐，确保数据的一致性和完整性。此外，我们引入了一种全局生成器冻结策略以防止灾难性遗忘，从而减轻 Non-IID 异变。广泛的多模态数据集评估证明，FedRecon 在 Non-IID 条件下进行模态重建的效果优于现有的先进方法。 

---
# KeepKV: Eliminating Output Perturbation in KV Cache Compression for Efficient LLMs Inference 

**Title (ZH)**: KeepKV：消除KV缓存压缩中的输出扰动以提高高效语言模型推理 

**Authors**: Yuxuan Tian, Zihan Wang, Yebo Peng, Aomufei Yuan, Zhiming Wang, Bairen Yi, Xin Liu, Yong Cui, Tong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09936)  

**Abstract**: Efficient inference of large language models (LLMs) is hindered by an ever-growing key-value (KV) cache, making KV cache compression a critical research direction. Traditional methods selectively evict less important KV cache entries based on attention scores or position heuristics, which leads to information loss and hallucinations. Recently, merging-based strategies have been explored to retain more information by merging KV pairs that would be discarded; however, these existing approaches inevitably introduce inconsistencies in attention distributions before and after merging, causing output perturbation and degraded generation quality. To overcome this challenge, we propose KeepKV, a novel adaptive KV cache merging method designed to eliminate output perturbation while preserving performance under strict memory constraints. KeepKV introduces the Electoral Votes mechanism that records merging history and adaptively adjusts attention scores. Moreover, it further leverages a novel Zero Inference-Perturbation Merging methods, keeping attention consistency and compensating for attention loss resulting from cache merging. KeepKV successfully retains essential context information within a significantly compressed cache. Extensive experiments on various benchmarks and LLM architectures demonstrate that KeepKV substantially reduces memory usage, enhances inference throughput by more than 2x and keeps superior generation quality even with 10% KV cache budgets. 

**Abstract (ZH)**: 高效的大型语言模型（LLMs）推理受到不断增长的关键值（KV）缓存的阻碍，使得KV缓存压缩成为关键的研究方向。 

---
# Quantum Natural Language Processing: A Comprehensive Review of Models, Methods, and Applications 

**Title (ZH)**: 量子自然语言处理：模型、方法及应用综述 

**Authors**: Farha Nausheen, Khandakar Ahmed, M Imad Khan  

**Link**: [PDF](https://arxiv.org/pdf/2504.09909)  

**Abstract**: In recent developments, deep learning methodologies applied to Natural Language Processing (NLP) have revealed a paradox: They improve performance but demand considerable data and resources for their training. Alternatively, quantum computing exploits the principles of quantum mechanics to overcome the computational limitations of current methodologies, thereby establishing an emerging field known as quantum natural language processing (QNLP). This domain holds the potential to attain a quantum advantage in the processing of linguistic structures, surpassing classical models in both efficiency and accuracy. In this paper, it is proposed to categorise QNLP models based on quantum computing principles, architecture, and computational approaches. This paper attempts to provide a survey on how quantum meets language by mapping state-of-the-art in this area, embracing quantum encoding techniques for classical data, QNLP models for prevalent NLP tasks, and quantum optimisation techniques for hyper parameter tuning. The landscape of quantum computing approaches applied to various NLP tasks is summarised by showcasing the specific QNLP methods used, and the popularity of these methods is indicated by their count. From the findings, it is observed that QNLP approaches are still limited to small data sets, with only a few models explored extensively, and there is increasing interest in the application of quantum computing to natural language processing tasks. 

**Abstract (ZH)**: Recent进展中，应用于自然语言处理（NLP）的深度学习方法揭示了一个悖论：它们提高了性能但需要大量的数据和资源进行训练。相反，量子计算利用量子力学原理克服当前方法的计算限制，从而建立了一个新兴领域——量子自然语言处理（QNLP）。该领域有可能在处理语言结构方面取得量子优势，高效性和准确性都超过经典模型。在本文中，建议根据量子计算原理、架构和计算方法来分类QNLP模型。本文试图通过映射该领域的最新进展，涵盖量子编码技术、普遍存在NLP任务的QNLP模型以及用于超参数调整的量子优化技术来概述量子计算方法在各种NLP任务中的应用景观，并通过这些方法的数量表明它们的流行程度。研究发现表明，QNLP方法仍然局限于小数据集，只有少数模型得到了广泛探索，并且越来越多地关注将量子计算应用于自然语言处理任务。 

---
# Plasticity-Aware Mixture of Experts for Learning Under QoE Shifts in Adaptive Video Streaming 

**Title (ZH)**: 考虑感知质量变化的自适应视频 streaming 中的弹性混合专家学习 

**Authors**: Zhiqiang He, Zhi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.09906)  

**Abstract**: Adaptive video streaming systems are designed to optimize Quality of Experience (QoE) and, in turn, enhance user satisfaction. However, differences in user profiles and video content lead to different weights for QoE factors, resulting in user-specific QoE functions and, thus, varying optimization objectives. This variability poses significant challenges for neural networks, as they often struggle to generalize under evolving targets - a phenomenon known as plasticity loss that prevents conventional models from adapting effectively to changing optimization objectives. To address this limitation, we propose the Plasticity-Aware Mixture of Experts (PA-MoE), a novel learning framework that dynamically modulates network plasticity by balancing memory retention with selective forgetting. In particular, PA-MoE leverages noise injection to promote the selective forgetting of outdated knowledge, thereby endowing neural networks with enhanced adaptive capabilities. In addition, we present a rigorous theoretical analysis of PA-MoE by deriving a regret bound that quantifies its learning performance. Experimental evaluations demonstrate that PA-MoE achieves a 45.5% improvement in QoE over competitive baselines in dynamic streaming environments. Further analysis reveals that the model effectively mitigates plasticity loss by optimizing neuron utilization. Finally, a parameter sensitivity study is performed by injecting varying levels of noise, and the results align closely with our theoretical predictions. 

**Abstract (ZH)**: 自适应视频流传输系统旨在优化用户体验（QoE），进而提升用户满意度。然而，用户特征和视频内容的不同导致QoE因素的权重不同，从而产生用户特异性的QoE函数和不同的优化目标。这种变异性给神经网络带来了显著挑战，因为它们往往难以在目标变化时进行泛化——这种现象被称为塑性损失，阻碍了传统模型的有效适应。为解决这一局限性，我们提出了自适应意识专家混合模型（PA-MoE），这是一种新颖的学习框架，通过平衡记忆保留与选择性遗忘来动态调节网络的塑性。PA-MoE 利用噪声注入促进对过时知识的选择性遗忘，从而增强神经网络的自适应能力。此外，我们通过推导出量化PA-MoE学习性能的遗憾界来对其进行了严格的理论分析。实验评估表明，在动态流传输环境中，PA-MoE 在用户体验（QoE）方面比竞争对手的基线方法提高了45.5%。进一步的分析显示，该模型通过优化神经元利用效率有效地缓解了塑性损失。最后，通过注入不同水平的噪声进行了参数敏感性研究，实验结果与我们的理论预测高度一致。 

---
# Learning from Reference Answers: Versatile Language Model Alignment without Binary Human Preference Data 

**Title (ZH)**: 参考答案指导的学习：无二元人类偏好数据的多功能语言模型对齐 

**Authors**: Shuai Zhao, Linchao Zhu, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09895)  

**Abstract**: Large language models~(LLMs) are expected to be helpful, harmless, and honest. In various alignment scenarios, such as general human preference, safety, and confidence alignment, binary preference data collection and reward modeling are resource-intensive but necessary for human preference transferring. In this work, we explore using the similarity between sampled generations and high-quality reference answers as an alternative reward function for LLM alignment. Using similarity as a reward circumvents training reward models, and collecting a single reference answer potentially costs less time than constructing binary preference pairs when multiple candidates are available. Specifically, we develop \textit{RefAlign}, a versatile REINFORCE-style alignment algorithm, which is free of reference and reward models. Instead, RefAlign utilizes BERTScore between sampled generations and high-quality reference answers as the surrogate reward. Beyond general human preference optimization, RefAlign can be readily extended to diverse scenarios, such as safety and confidence alignment, by incorporating the similarity reward with task-related objectives. In various scenarios, {RefAlign} demonstrates comparable performance to previous alignment methods while offering high efficiency. 

**Abstract (ZH)**: 大规模语言模型（LLMs）期望具备助益性、无害性和诚实性。在一般人类偏好、安全和信心对齐等各类对齐场景中，二元偏好数据收集和奖励建模虽然资源密集但必不可少。为此，我们探索使用生成样本与高质参考答案之间的相似性作为LLM对齐的替代奖励函数。使用相似性作为奖励可以避免训练奖励模型，并且在多个候选方案可用时，收集单个参考答案可能比构造二元偏好对节省更多时间。具体地，我们开发了RefAlign，这是一种通用的REINFORCE风格对齐算法，无需参考模型和奖励模型。相反，RefAlign利用BERTScore衡量生成样本与高质参考答案之间的相似性作为替代奖励。除了通用人类偏好的优化，RefAlign还可以通过结合相似性奖励和相关任务目标，轻松扩展应用于各种场景，如安全和信心对齐。在各类场景中，RefAlign在保持较高效率的同时，表现出与先前对齐方法相当的性能。 

---
# LangPert: Detecting and Handling Task-level Perturbations for Robust Object Rearrangement 

**Title (ZH)**: LangPert: 任务级扰动的检测与处理以实现稳健的物体重排 

**Authors**: Xu Yin, Min-Sung Yoon, Yuchi Huo, Kang Zhang, Sung-Eui Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2504.09893)  

**Abstract**: Task execution for object rearrangement could be challenged by Task-Level Perturbations (TLP), i.e., unexpected object additions, removals, and displacements that can disrupt underlying visual policies and fundamentally compromise task feasibility and progress. To address these challenges, we present LangPert, a language-based framework designed to detect and mitigate TLP situations in tabletop rearrangement tasks. LangPert integrates a Visual Language Model (VLM) to comprehensively monitor policy's skill execution and environmental TLP, while leveraging the Hierarchical Chain-of-Thought (HCoT) reasoning mechanism to enhance the Large Language Model (LLM)'s contextual understanding and generate adaptive, corrective skill-execution plans. Our experimental results demonstrate that LangPert handles diverse TLP situations more effectively than baseline methods, achieving higher task completion rates, improved execution efficiency, and potential generalization to unseen scenarios. 

**Abstract (ZH)**: 基于语言的框架LangPert可检测并缓解物体重排任务中的任务级干扰（TLP） 

---
# Constructing Micro Knowledge Graphs from Technical Support Documents 

**Title (ZH)**: 从技术支撑文档构建微知识图谱 

**Authors**: Atul Kumar, Nisha Gupta, Saswati Dana  

**Link**: [PDF](https://arxiv.org/pdf/2504.09877)  

**Abstract**: Short technical support pages such as IBM Technotes are quite common in technical support domain. These pages can be very useful as the knowledge sources for technical support applications such as chatbots, search engines and question-answering (QA) systems. Information extracted from documents to drive technical support applications is often stored in the form of Knowledge Graph (KG). Building KGs from a large corpus of documents poses a challenge of granularity because a large number of entities and actions are present in each page. The KG becomes virtually unusable if all entities and actions from these pages are stored in the KG. Therefore, only key entities and actions from each page are extracted and stored in the KG. This approach however leads to loss of knowledge represented by entities and actions left out of the KG as they are no longer available to graph search and reasoning functions. We propose a set of techniques to create micro knowledge graph (micrograph) for each of such web pages. The micrograph stores all the entities and actions in a page and also takes advantage of the structure of the page to represent exactly in which part of that page these entities and actions appeared, and also how they relate to each other. These micrographs can be used as additional knowledge sources by technical support applications. We define schemas for representing semi-structured and plain text knowledge present in the technical support web pages. Solutions in technical support domain include procedures made of steps. We also propose a technique to extract procedures from these webpages and the schemas to represent them in the micrographs. We also discuss how technical support applications can take advantage of the micrographs. 

**Abstract (ZH)**: IBM技术笔记等简短的技术支持页面在技术支撑领域非常常见。这些页面可以作为聊天机器人、搜索引擎和问答系统等技术支持应用的知识来源非常有用。从大量文档中构建知识图谱（KG）面临着粒度问题，因为每个页面中包含大量的实体和动作。如果将这些页面中的所有实体和动作都存储在知识图谱中，知识图谱将变得实际上无法使用。因此，仅从每个页面中提取关键的实体和动作并存储在知识图谱中。然而，这种方法会导致失去未包含在知识图谱中的实体和动作所代表的知识，这些知识对图搜索和推理功能不再可用。我们提出了一套技术来为每个这类网页创建微知识图谱（micrograph）。微知识图谱存储页面中的所有实体和动作，并利用页面结构来精确表示这些实体和动作出现在页面的哪个部分，以及它们之间的关系。这些微知识图谱可以作为技术支持应用的额外知识来源。我们定义了表示技术支持网页中半结构化和纯文本知识的模式。技术支持领域的解决方案包括步骤组成的流程，我们还提出了一种从这些网页中提取流程并用微知识图谱表示它们的技术。我们还讨论了技术支持应用如何利用微知识图谱。 

---
# HDC: Hierarchical Distillation for Multi-level Noisy Consistency in Semi-Supervised Fetal Ultrasound Segmentation 

**Title (ZH)**: HDC：层次蒸馏在半监督胎儿超声分割中的多层级噪声一致性精炼 

**Authors**: Tran Quoc Khanh Le, Nguyen Lan Vi Vu, Ha-Hieu Pham, Xuan-Loc Huynh, Tien-Huy Nguyen, Minh Huu Nhat Le, Quan Nguyen, Hien D. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2504.09876)  

**Abstract**: Transvaginal ultrasound is a critical imaging modality for evaluating cervical anatomy and detecting physiological changes. However, accurate segmentation of cervical structures remains challenging due to low contrast, shadow artifacts, and fuzzy boundaries. While convolutional neural networks (CNNs) have shown promising results in medical image segmentation, their performance is often limited by the need for large-scale annotated datasets - an impractical requirement in clinical ultrasound imaging. Semi-supervised learning (SSL) offers a compelling solution by leveraging unlabeled data, but existing teacher-student frameworks often suffer from confirmation bias and high computational costs. We propose HDC, a novel semi-supervised segmentation framework that integrates Hierarchical Distillation and Consistency learning within a multi-level noise mean-teacher framework. Unlike conventional approaches that rely solely on pseudo-labeling, we introduce a hierarchical distillation mechanism that guides feature-level learning via two novel objectives: (1) Correlation Guidance Loss to align feature representations between the teacher and main student branch, and (2) Mutual Information Loss to stabilize representations between the main and noisy student branches. Our framework reduces model complexity while improving generalization. Extensive experiments on two fetal ultrasound datasets, FUGC and PSFH, demonstrate that our method achieves competitive performance with significantly lower computational overhead than existing multi-teacher models. 

**Abstract (ZH)**: 阴道超声是评估宫颈解剖结构和检测生理变化的关键成像模态。然而，由于对比度低、阴影伪影和边界模糊，宫颈结构的准确分割仍然具有挑战性。尽管卷积神经网络（CNN）在医学图像分割方面取得了有前途的结果，但其性能常受限于对大规模标注数据集的需求——这在临床超声成像中是不切实际的要求。半监督学习（SSL）通过利用未标注数据提供了一个有吸引力的解决方案，但现有教师-学生框架往往受到确认偏见和高计算成本的影响。我们提出了一种名为HDC的新颖半监督分割框架，该框架在多层次噪声教师框架内整合了层次蒸馏和一致性学习。与依赖伪标签的传统方法不同，我们引入了一种层次蒸馏机制，通过两个新颖目标来引导特征级学习：（1）相关性指导损失，用于对齐教师和主学生分支的特征表示；（2）互信息损失，用于稳定主学生分支和嘈杂学生分支之间的表示。我们的框架在减少模型复杂性的同时提高了泛化能力。在两个胎超数据集FUGC和PSFH上的广泛实验表明，我们的方法在显著降低计算开销的前提下，实现了与现有多教师模型相当的性能。 

---
# Truncated Matrix Completion - An Empirical Study 

**Title (ZH)**: 截断矩阵完成：一项经验研究 

**Authors**: Rishhabh Naik, Nisarg Trivedi, Davoud Ataee Tarzanagh, Laura Balzano  

**Link**: [PDF](https://arxiv.org/pdf/2504.09873)  

**Abstract**: Low-rank Matrix Completion (LRMC) describes the problem where we wish to recover missing entries of partially observed low-rank matrix. Most existing matrix completion work deals with sampling procedures that are independent of the underlying data values. While this assumption allows the derivation of nice theoretical guarantees, it seldom holds in real-world applications. In this paper, we consider various settings where the sampling mask is dependent on the underlying data values, motivated by applications in sensing, sequential decision-making, and recommender systems. Through a series of experiments, we study and compare the performance of various LRMC algorithms that were originally successful for data-independent sampling patterns. 

**Abstract (ZH)**: 低秩矩阵完成（LRMC）描述了我们希望恢复部分观测低秩矩阵中缺失条目的问题。大多数现有的矩阵完成工作处理的是与底层数据值无关的采样过程。虽然这一假设允许得出良好的理论保证，但在实际应用中往往不成立。在本文中，我们考虑各种采样掩码依赖于底层数据值的情景，这些情景受到传感、顺序决策和推荐系统应用的启发。通过一系列实验，我们研究并比较了原本对数据无关采样模式成功的各种低秩矩阵完成算法的性能。 

---
# Labeling Messages as AI-Generated Does Not Reduce Their Persuasive Effects 

**Title (ZH)**: 将消息标记为AI生成并不会降低其说服效果。 

**Authors**: Isabel O. Gallegos, Chen Shani, Weiyan Shi, Federico Bianchi, Izzy Gainsburg, Dan Jurafsky, Robb Willer  

**Link**: [PDF](https://arxiv.org/pdf/2504.09865)  

**Abstract**: As generative artificial intelligence (AI) enables the creation and dissemination of information at massive scale and speed, it is increasingly important to understand how people perceive AI-generated content. One prominent policy proposal requires explicitly labeling AI-generated content to increase transparency and encourage critical thinking about the information, but prior research has not yet tested the effects of such labels. To address this gap, we conducted a survey experiment (N=1601) on a diverse sample of Americans, presenting participants with an AI-generated message about several public policies (e.g., allowing colleges to pay student-athletes), randomly assigning whether participants were told the message was generated by (a) an expert AI model, (b) a human policy expert, or (c) no label. We found that messages were generally persuasive, influencing participants' views of the policies by 9.74 percentage points on average. However, while 94.6% of participants assigned to the AI and human label conditions believed the authorship labels, labels had no significant effects on participants' attitude change toward the policies, judgments of message accuracy, nor intentions to share the message with others. These patterns were robust across a variety of participant characteristics, including prior knowledge of the policy, prior experience with AI, political party, education level, or age. Taken together, these results imply that, while authorship labels would likely enhance transparency, they are unlikely to substantially affect the persuasiveness of the labeled content, highlighting the need for alternative strategies to address challenges posed by AI-generated information. 

**Abstract (ZH)**: 随着生成式人工智能（AI）能够大规模快速地创造和传播信息，理解人们如何感知AI生成内容变得越来越重要。一个突出的政策建议是明确标注AI生成的内容以增强透明度并促进对信息的批判性思考，但此前的研究尚未对此类标签的效果进行测试。为填补这一空白，我们对美国多元样本组（N=1601）进行了问卷实验，向参与者展示关于若干公共政策（例如，允许大学支付运动员薪酬）的AI生成信息，并随机分配参与者是否被告知该信息是由（a）专家AI模型、（b）人类政策专家或（c）无标签生成。我们发现，这些信息通常具有说服力，平均影响参与者对政策的看法9.74个百分点。然而，在AI和人类标签条件下被指派的94.6%的参与者相信了作者身份标签，但标签对参与者对政策的态度变化、对信息准确性的判断以及分享信息意愿等方面均无显著影响。这些模式在多种参与者特征（包括政策知识、AI经验、政治党派、教育水平或年龄）中均表现 robust。综上所述，这些结果表明，虽然作者身份标签可能增强透明度，但它们不太可能显著影响标记内容的说服力，这突出了需要寻找替代策略以应对AI生成信息所带来的挑战。 

---
# SUMART: SUMmARizing Translation from Wordy to Concise Expression 

**Title (ZH)**: SUMART: 从冗长表达总结到简洁表达的翻译 

**Authors**: Naoto Nishida, Jun Rekimoto  

**Link**: [PDF](https://arxiv.org/pdf/2504.09860)  

**Abstract**: We propose SUMART, a method for summarizing and compressing the volume of verbose subtitle translations. SUMART is designed for understanding translated captions (e.g., interlingual conversations via subtitle translation or when watching movies in foreign language audio and translated captions). SUMART is intended for users who want a big-picture and fast understanding of the conversation, audio, video content, and speech in a foreign language. During the training data collection, when a speaker makes a verbose statement, SUMART employs a large language model on-site to compress the volume of subtitles. This compressed data is then stored in a database for fine-tuning purposes. Later, SUMART uses data pairs from those non-compressed ASR results and compressed translated results for fine-tuning the translation model to generate more concise translations for practical uses. In practical applications, SUMART utilizes this trained model to produce concise translation results. Furthermore, as a practical application, we developed an application that allows conversations using subtitle translation in augmented reality spaces. As a pilot study, we conducted qualitative surveys using a SUMART prototype and a survey on the summarization model for SUMART. We envision the most effective use case of this system is where users need to consume a lot of information quickly (e.g., Speech, lectures, podcasts, Q&A in conferences). 

**Abstract (ZH)**: SUMART：一种用于总结和压缩冗长字幕翻译的 方法 

---
# Working with Large Language Models to Enhance Messaging Effectiveness for Vaccine Confidence 

**Title (ZH)**: 利用大型语言模型提升疫苗信心讯息效果 

**Authors**: Lucinda Gullison, Feng Fu  

**Link**: [PDF](https://arxiv.org/pdf/2504.09857)  

**Abstract**: Vaccine hesitancy and misinformation are significant barriers to achieving widespread vaccination coverage. Smaller public health departments may lack the expertise or resources to craft effective vaccine messaging. This paper explores the potential of ChatGPT-augmented messaging to promote confidence in vaccination uptake.
We conducted a survey in which participants chose between pairs of vaccination messages and assessed which was more persuasive and to what extent. In each pair, one message was the original, and the other was augmented by ChatGPT. At the end of the survey, participants were informed that half of the messages had been generated by ChatGPT. They were then asked to provide both quantitative and qualitative responses regarding how knowledge of a message's ChatGPT origin affected their impressions.
Overall, ChatGPT-augmented messages were rated slightly higher than the original messages. These messages generally scored better when they were longer. Respondents did not express major concerns about ChatGPT-generated content, nor was there a significant relationship between participants' views on ChatGPT and their message ratings. Notably, there was a correlation between whether a message appeared first or second in a pair and its score.
These results point to the potential of ChatGPT to enhance vaccine messaging, suggesting a promising direction for future research on human-AI collaboration in public health communication. 

**Abstract (ZH)**: 疫苗犹豫和错误信息是实现广泛疫苗接种覆盖率的显著障碍。较小的公共卫生部门可能缺乏制定有效疫苗宣传信息的专业知识或资源。本文探讨了ChatGPT增强信息在促进疫苗接种信心方面的潜在作用。 

---
# PestMA: LLM-based Multi-Agent System for Informed Pest Management 

**Title (ZH)**: 基于LLM的多agent系统以进行知情害虫管理 

**Authors**: Hongrui Shi, Shunbao Li, Zhipeng Yuan, Po Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09855)  

**Abstract**: Effective pest management is complex due to the need for accurate, context-specific decisions. Recent advancements in large language models (LLMs) open new possibilities for addressing these challenges by providing sophisticated, adaptive knowledge acquisition and reasoning. However, existing LLM-based pest management approaches often rely on a single-agent paradigm, which can limit their capacity to incorporate diverse external information, engage in systematic validation, and address complex, threshold-driven decisions. To overcome these limitations, we introduce PestMA, an LLM-based multi-agent system (MAS) designed to generate reliable and evidence-based pest management advice. Building on an editorial paradigm, PestMA features three specialized agents, an Editor for synthesizing pest management recommendations, a Retriever for gathering relevant external data, and a Validator for ensuring correctness. Evaluations on real-world pest scenarios demonstrate that PestMA achieves an initial accuracy of 86.8% for pest management decisions, which increases to 92.6% after validation. These results underscore the value of collaborative agent-based workflows in refining and validating decisions, highlighting the potential of LLM-based multi-agent systems to automate and enhance pest management processes. 

**Abstract (ZH)**: 有效的害虫管理因需要准确的、情境特定的决策而复杂。近期大型语言模型（LLMs）的进步为应对这些挑战提供了新可能性，通过提供复杂的自适应知识获取和推理。然而，现有的基于LLM的害虫管理方法通常依赖于单代理范式，这可能限制了其整合多样化外部信息、进行系统性验证和应对复杂阈值驱动决策的能力。为克服这些局限，我们引入了PestMA，这是一种基于LLM的多代理系统（MAS），旨在生成可靠且基于证据的害虫管理建议。基于编辑范式，PestMA 包含三个专业化代理：编辑代理负责合成害虫管理建议，检索代理负责收集相关外部数据，验证代理负责确保准确性。在真实世界害虫场景中的评估表明，PestMA 的初始决策准确率为86.8%，经过验证后提高到92.6%。这些结果强调了基于代理的协作工作流在细化和验证决策方面的重要性，突显了基于LLM的多代理系统在自动化和增强害虫管理过程方面的潜力。 

---
# Carbon-Efficient 3D DNN Acceleration: Optimizing Performance and Sustainability 

**Title (ZH)**: 碳效率的3D DNN加速：优化性能与可持续性 

**Authors**: Aikaterini Maria Panteleaki, Konstantinos Balaskas, Georgios Zervakis, Hussam Amrouch, Iraklis Anagnostopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2504.09851)  

**Abstract**: As Deep Neural Networks (DNNs) continue to drive advancements in artificial intelligence, the design of hardware accelerators faces growing concerns over embodied carbon footprint due to complex fabrication processes. 3D integration improves performance but introduces sustainability challenges, making carbon-aware optimization essential. In this work, we propose a carbon-efficient design methodology for 3D DNN accelerators, leveraging approximate computing and genetic algorithm-based design space exploration to optimize Carbon Delay Product (CDP). By integrating area-efficient approximate multipliers into Multiply-Accumulate (MAC) units, our approach effectively reduces silicon area and fabrication overhead while maintaining high computational accuracy. Experimental evaluations across three technology nodes (45nm, 14nm, and 7nm) show that our method reduces embodied carbon by up to 30% with negligible accuracy drop. 

**Abstract (ZH)**: 随着深度神经网络（DNNs）继续推动人工智能的进步，由于复杂制造工艺导致的人体碳足迹问题使得硬件加速器的设计面临日益增长的担忧。3D集成提高了性能但也引入了可持续性挑战，因此碳意识优化变得至关重要。在本工作中，我们提出了一种用于3D DNN加速器的碳高效设计方法，通过利用近似计算和基于遗传算法的设计空间探索来优化碳延迟积（CDP）。通过将面积高效的近似乘法器整合到乘加（MAC）单元中，我们的方法有效地减少了硅面积和制造开销，同时保持了高计算精度。在三种技术节点（45nm、14nm和7nm）下的实验评估表明，我们的方法在几乎无精度损失的情况下将人体碳足迹最多减少了30%。 

---
# GlyTwin: Digital Twin for Glucose Control in Type 1 Diabetes Through Optimal Behavioral Modifications Using Patient-Centric Counterfactuals 

**Title (ZH)**: GlyTwin: 基于患者中心反事实分析的1型糖尿病血糖控制数字孪生通过最优行为修改 

**Authors**: Asiful Arefeen, Saman Khamesian, Maria Adela Grando, Bithika Thompson, Hassan Ghasemzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2504.09846)  

**Abstract**: Frequent and long-term exposure to hyperglycemia (i.e., high blood glucose) increases the risk of chronic complications such as neuropathy, nephropathy, and cardiovascular disease. Current technologies like continuous subcutaneous insulin infusion (CSII) and continuous glucose monitoring (CGM) primarily model specific aspects of glycemic control-like hypoglycemia prediction or insulin delivery. Similarly, most digital twin approaches in diabetes management simulate only physiological processes. These systems lack the ability to offer alternative treatment scenarios that support proactive behavioral interventions. To address this, we propose GlyTwin, a novel digital twin framework that uses counterfactual explanations to simulate optimal treatments for glucose regulation. Our approach helps patients and caregivers modify behaviors like carbohydrate intake and insulin dosing to avoid abnormal glucose events. GlyTwin generates behavioral treatment suggestions that proactively prevent hyperglycemia by recommending small adjustments to daily choices, reducing both frequency and duration of these events. Additionally, it incorporates stakeholder preferences into the intervention design, making recommendations patient-centric and tailored. We evaluate GlyTwin on AZT1D, a newly constructed dataset with longitudinal data from 21 type 1 diabetes (T1D) patients on automated insulin delivery systems over 26 days. Results show GlyTwin outperforms state-of-the-art counterfactual methods, generating 76.6% valid and 86% effective interventions. These findings demonstrate the promise of counterfactual-driven digital twins in delivering personalized healthcare. 

**Abstract (ZH)**: 频繁且长期的高血糖暴露增加慢性并发症（如神经病变、肾病和心血管疾病）的风险。现有的技术如持续皮下胰岛素输注（CSII）和连续血糖监测（CGM）主要模拟血糖控制的特定方面，如低血糖预测或胰岛素输送。类似地，大多数糖尿病管理中的数字孪生方法仅模拟生理过程。这些系统缺乏提供替代治疗方案的能力，以支持前瞻性的行为干预。为了弥补这一不足，我们提出了GlyTwin，一种新颖的数字孪生框架，使用反事实解释来模拟葡萄糖调节的最佳治疗方案。我们的方法帮助患者和护理人员调整碳水化合物摄入和胰岛素剂量，以避免异常血糖事件。GlyTwin生成行为治疗建议，通过建议日常选择的小调整，前瞻性地预防高血糖，减少这些事件的频率和持续时间。此外，它将利益相关者的需求纳入干预设计中，使建议具有患者中心性和个性化。我们在AZT1D数据集上评估了GlyTwin，该数据集包含21名使用自动化胰岛素输送系统的1型糖尿病（T1D）患者26天的纵向数据。结果显示，GlyTwin优于最先进的反事实方法，产生了76.6%有效且86%有效的干预措施。这些发现表明，反事实驱动的数字孪生在提供个性化医疗保健方面的潜力。 

---
# OVERLORD: Ultimate Scaling of DataLoader for Multi-Source Large Foundation Model Training 

**Title (ZH)**: OVERLORD: 多源大型基础模型训练的 DataLoader 最终扩展方案 

**Authors**: Juntao Zhao, Qi Lu, Wei Jia, Borui Wan, Lei Zuo, Junda Feng, Jianyu Jiang, Yangrui Chen, Shuaishuai Cao, Jialing He, Kaihua Jiang, Yuanzhe Hu, Yanghua Peng, Haibin Lin, Xin Liu, Chuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.09844)  

**Abstract**: Modern frameworks for training large foundation models (LFMs) employ data loaders in a data parallel paradigm. While this design offers implementation simplicity, it introduces two fundamental challenges. First, due to the quadratic computational complexity of the attention operator, the non-uniform sample distribution over data-parallel ranks leads to a significant workload imbalance among loaders, which degrades the training efficiency. This paradigm also impedes the implementation of data mixing algorithms (e.g., curriculum learning) over different datasets. Second, to acquire a broad range of capability, LFMs training ingests data from diverse sources, each with distinct file access states. Colocating massive datasets within loader instances can easily exceed local pod memory capacity. Additionally, heavy sources with higher transformation latency require larger worker pools, further exacerbating memory consumption.
We present OVERLORD, an industrial-grade distributed data loading architecture with three innovations: (1) A centralized and declarative data plane, which facilitates elastic data orchestration strategy, such as long-short context, multimodal, and curriculum learning; (2) Disaggregated multisource preprocessing through role-specific actors, i.e., Source Loaders and Data Constructors, leveraging autoscaling for Source Loaders towards heterogeneous and evolving source preprocessing cost; (3) Shadow Loaders with differential checkpointing for uninterrupted fault recovery. Deployed on production clusters scaling to multi-thousand GPU, OVERLORD achieves: (1) 4.5x end-to-end training throughput improvement, (2) a minimum 3.6x reduction in CPU memory usage, with further improvements to be added in later experiments. 

**Abstract (ZH)**: 工业级分布式数据加载架构 OVERLORD：面向大规模基础模型训练的创新设计 

---
# StruPhantom: Evolutionary Injection Attacks on Black-Box Tabular Agents Powered by Large Language Models 

**Title (ZH)**: StruPhantom：由大型语言模型驱动的黑盒表格式代理的进化注入攻击 

**Authors**: Yang Feng, Xudong Pan  

**Link**: [PDF](https://arxiv.org/pdf/2504.09841)  

**Abstract**: The proliferation of autonomous agents powered by large language models (LLMs) has revolutionized popular business applications dealing with tabular data, i.e., tabular agents. Although LLMs are observed to be vulnerable against prompt injection attacks from external data sources, tabular agents impose strict data formats and predefined rules on the attacker's payload, which are ineffective unless the agent navigates multiple layers of structural data to incorporate the payload. To address the challenge, we present a novel attack termed StruPhantom which specifically targets black-box LLM-powered tabular agents. Our attack designs an evolutionary optimization procedure which continually refines attack payloads via the proposed constrained Monte Carlo Tree Search augmented by an off-topic evaluator. StruPhantom helps systematically explore and exploit the weaknesses of target applications to achieve goal hijacking. Our evaluation validates the effectiveness of StruPhantom across various LLM-based agents, including those on real-world platforms, and attack scenarios. Our attack achieves over 50% higher success rates than baselines in enforcing the application's response to contain phishing links or malicious codes. 

**Abstract (ZH)**: 基于大规模语言模型的自主代理普及化已革命性地改变了处理表格数据的流行商业应用，即表格代理。尽管观察到大规模语言模型对外部数据来源的提示注入攻击易受攻击，但表格代理对攻击载荷施加了严格的数据格式和预定义规则，除非代理导航多层结构数据以整合载荷，这些规则才无效。为应对这一挑战，我们提出了一种新型攻击方法，称为StruPhantom，专门针对黑盒的大规模语言模型驱动的表格代理。我们的攻击设计了一种基于受约束的蒙特卡洛树搜索的进化优化程序，该程序通过一个离题评估器不断优化攻击载荷。StruPhantom有助于系统地探索和利用目标应用程序的弱点以实现目标篡改。我们的评估验证了StruPhantom在各种基于大规模语言模型的代理中的有效性，包括实际平台上的代理和攻击场景。我们的攻击在强制应用程序响应包含欺诈链接或恶意代码方面，成功率比基线高出50%以上。 

---
# SafeSpeech: Robust and Universal Voice Protection Against Malicious Speech Synthesis 

**Title (ZH)**: SafeSpeech:稳健且通用的恶意语音合成防护 

**Authors**: Zhisheng Zhang, Derui Wang, Qianyi Yang, Pengyang Huang, Junhan Pu, Yuxin Cao, Kai Ye, Jie Hao, Yixian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09839)  

**Abstract**: Speech synthesis technology has brought great convenience, while the widespread usage of realistic deepfake audio has triggered hazards. Malicious adversaries may unauthorizedly collect victims' speeches and clone a similar voice for illegal exploitation (\textit{e.g.}, telecom fraud). However, the existing defense methods cannot effectively prevent deepfake exploitation and are vulnerable to robust training techniques. Therefore, a more effective and robust data protection method is urgently needed. In response, we propose a defensive framework, \textit{\textbf{SafeSpeech}}, which protects the users' audio before uploading by embedding imperceptible perturbations on original speeches to prevent high-quality synthetic speech. In SafeSpeech, we devise a robust and universal proactive protection technique, \textbf{S}peech \textbf{PE}rturbative \textbf{C}oncealment (\textbf{SPEC}), that leverages a surrogate model to generate universally applicable perturbation for generative synthetic models. Moreover, we optimize the human perception of embedded perturbation in terms of time and frequency domains. To evaluate our method comprehensively, we conduct extensive experiments across advanced models and datasets, both subjectively and objectively. Our experimental results demonstrate that SafeSpeech achieves state-of-the-art (SOTA) voice protection effectiveness and transferability and is highly robust against advanced adaptive adversaries. Moreover, SafeSpeech has real-time capability in real-world tests. The source code is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 语音合成技术带来了极大的便利，而广泛使用的逼真深伪音频也引发了安全隐患。恶意攻击者可能未经授权收集受害者讲话，并克隆相似的声音进行非法利用（例如电信诈骗）。然而，现有的防御方法无法有效防止深伪利用，并且容易受到鲁棒训练技术的攻击。因此，一种更有效且鲁棒的数据保护方法迫在眉睫。为此，我们提出了一种防御框架——SafeSpeech，该框架通过在上传前在原始讲话中嵌入不可感知的扰动来保护用户的音频，以防止高质量合成语音。在SafeSpeech中，我们设计了一种鲁棒且通用的主动保护技术——Speech Perturbative Concealment (SPEC)，该技术利用代理模型为生成型合成模型生成通用适用的扰动。此外，我们从时间和频率域优化嵌入扰动的人类感知。为了全面评估我们的方法，我们在高级模型和数据集上进行了广泛的实验，从主观和客观两个方面进行评估。实验结果表明，SafeSpeech实现了最先进的语音保护效果和可移植性，并且对先进的自适应对手具有高度鲁棒性。此外，SafeSpeech在实际测试中具有实时能力。源代码可在https://this-link-url.com/获得。 

---
# Offline Dynamic Inventory and Pricing Strategy: Addressing Censored and Dependent Demand 

**Title (ZH)**: 离线动态库存与定价策略：应对受限且相关的需求 

**Authors**: Korel Gundem, Zhengling Qi  

**Link**: [PDF](https://arxiv.org/pdf/2504.09831)  

**Abstract**: In this paper, we study the offline sequential feature-based pricing and inventory control problem where the current demand depends on the past demand levels and any demand exceeding the available inventory is lost. Our goal is to leverage the offline dataset, consisting of past prices, ordering quantities, inventory levels, covariates, and censored sales levels, to estimate the optimal pricing and inventory control policy that maximizes long-term profit. While the underlying dynamic without censoring can be modeled by Markov decision process (MDP), the primary obstacle arises from the observed process where demand censoring is present, resulting in missing profit information, the failure of the Markov property, and a non-stationary optimal policy. To overcome these challenges, we first approximate the optimal policy by solving a high-order MDP characterized by the number of consecutive censoring instances, which ultimately boils down to solving a specialized Bellman equation tailored for this problem. Inspired by offline reinforcement learning and survival analysis, we propose two novel data-driven algorithms to solving these Bellman equations and, thus, estimate the optimal policy. Furthermore, we establish finite sample regret bounds to validate the effectiveness of these algorithms. Finally, we conduct numerical experiments to demonstrate the efficacy of our algorithms in estimating the optimal policy. To the best of our knowledge, this is the first data-driven approach to learning optimal pricing and inventory control policies in a sequential decision-making environment characterized by censored and dependent demand. The implementations of the proposed algorithms are available at this https URL 

**Abstract (ZH)**: 基于历史数据的序贯特征定价与库存控制问题研究：考虑 censored 和依赖需求的最优策略学习 

---
# Efficient Multi-Task Modeling through Automated Fusion of Trained Models 

**Title (ZH)**: 通过训练模型的自动融合实现高效的多任务建模 

**Authors**: Jingxuan Zhou, Weidong Bao, Ji Wang, Zhengyi Zhong, Dayu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09812)  

**Abstract**: Although multi-task learning is widely applied in intelligent services, traditional multi-task modeling methods often require customized designs based on specific task combinations, resulting in a cumbersome modeling process. Inspired by the rapid development and excellent performance of single-task models, this paper proposes an efficient multi-task modeling method that can automatically fuse trained single-task models with different structures and tasks to form a multi-task model. As a general framework, this method allows modelers to simply prepare trained models for the required tasks, simplifying the modeling process while fully utilizing the knowledge contained in the trained models. This eliminates the need for excessive focus on task relationships and model structure design. To achieve this goal, we consider the structural differences among various trained models and employ model decomposition techniques to hierarchically decompose them into multiple operable model components. Furthermore, we have designed an Adaptive Knowledge Fusion (AKF) module based on Transformer, which adaptively integrates intra-task and inter-task knowledge based on model components. Through the proposed method, we achieve efficient and automated construction of multi-task models, and its effectiveness is verified through extensive experiments on three datasets. 

**Abstract (ZH)**: 尽管多任务学习在智能服务中广泛应用，传统多任务建模方法往往需要根据特定的任务组合进行定制化设计，导致建模过程繁琐。受单任务模型快速发展和优异性能的启发，本文提出了一种高效的多任务建模方法，可以自动融合结构和任务不同的训练好的单任务模型，形成多任务模型。作为一种通用框架，该方法允许建模者仅需准备所需的训练模型，简化建模过程，同时充分利用训练模型中包含的知识。这种方法消除了过度关注任务关系和模型结构设计的需要。为了实现这一目标，我们考虑了各种训练模型之间的结构差异，并采用模型分解技术，逐级分解为多个可操作的模型组件。此外，我们基于Transformer设计了一个自适应知识融合（AKF）模块，根据模型组件自适应地整合任务内和任务间知识。通过所提出的方法，我们实现了多任务模型的有效和自动构建，并通过三个数据集上的 extensive 实验验证了其有效性。 

---
# See or Recall: A Sanity Check for the Role of Vision in Solving Visualization Question Answer Tasks with Multimodal LLMs 

**Title (ZH)**: 看看或回忆：对多模态LLMs在解答可视化问答任务中视觉作用的一种合理性检查 

**Authors**: Zhimin Li, Haichao Miao, Xinyuan Yan, Valerio Pascucci, Matthew Berger, Shusen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.09809)  

**Abstract**: Recent developments in multimodal large language models (MLLM) have equipped language models to reason about vision and language jointly. This permits MLLMs to both perceive and answer questions about data visualization across a variety of designs and tasks. Applying MLLMs to a broad range of visualization tasks requires us to properly evaluate their capabilities, and the most common way to conduct evaluation is through measuring a model's visualization reasoning capability, analogous to how we would evaluate human understanding of visualizations (e.g., visualization literacy). However, we found that in the context of visualization question answering (VisQA), how an MLLM perceives and reasons about visualizations can be fundamentally different from how humans approach the same problem. During the evaluation, even without visualization, the model could correctly answer a substantial portion of the visualization test questions, regardless of whether any selection options were provided. We hypothesize that the vast amount of knowledge encoded in the language model permits factual recall that supersedes the need to seek information from the visual signal. It raises concerns that the current VisQA evaluation may not fully capture the models' visualization reasoning capabilities. To address this, we propose a comprehensive sanity check framework that integrates a rule-based decision tree and a sanity check table to disentangle the effects of "seeing" (visual processing) and "recall" (reliance on prior knowledge). This validates VisQA datasets for evaluation, highlighting where models are truly "seeing", positively or negatively affected by the factual recall, or relying on inductive biases for question answering. Our study underscores the need for careful consideration in designing future visualization understanding studies when utilizing MLLMs. 

**Abstract (ZH)**: 近期多模态大型语言模型的发展使语言模型能够联合推理视觉和语言。这使得多模态大型语言模型能够感知并回答各种设计和任务的数据可视化问题。将多模态大型语言模型应用于广泛的可视化任务需要我们合理评估其能力，最常见的方式是通过测量模型的可视化推理能力来评估，类似于评估人类对可视化图表的理解能力（如可视化素养）。然而，我们发现，在可视化问题回答（VisQA）的背景下，多模态大型语言模型对视觉信息的感知和推理方式与人类处理相同问题的方式可能存在根本不同。在评估过程中，即使不提供任何视觉信息，模型也能正确回答大量可视化测试问题，无论是否有选择选项。我们推测，语言模型中编码的大量知识使得其能够进行事实回忆，从而超越了从视觉信号中获取信息的需要。这引发了当前VisQA评估可能未能全面捕获模型的可视化推理能力的担忧。为了应对这一问题，我们提出了一种综合的常识检查框架，该框架结合了基于规则的决策树和常识检查表，以分离“看到”（视觉处理）和“回忆”（依赖先验知识）的影响。这一框架验证了用于评估的VisQA数据集，强调了模型在真正“看到”、受事实回忆正向或负向影响，或依赖归纳偏置进行问题回答的方面。我们的研究强调了在利用多模态大型语言模型进行未来可视化理解研究时需要仔细考虑的重要性。 

---
# Training Small Reasoning LLMs with Cognitive Preference Alignment 

**Title (ZH)**: 训练认知偏好对齐的小规模推理大语言模型 

**Authors**: Wenrui Cai, Chengyu Wang, Junbing Yan, Jun Huang, Xiangzhong Fang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09802)  

**Abstract**: The reasoning capabilities of large language models (LLMs), such as OpenAI's o1 and DeepSeek-R1, have seen substantial advancements through deep thinking. However, these enhancements come with significant resource demands, underscoring the need to explore strategies to train effective reasoning LLMs with far fewer parameters. A critical challenge is that smaller models have different capacities and cognitive trajectories than their larger counterparts. Hence, direct distillation of chain-of-thought (CoT) results from large LLMs to smaller ones can be sometimes ineffective and requires a huge amount of annotated data. In this paper, we introduce a novel framework called Critique-Rethink-Verify (CRV), designed for training smaller yet powerful reasoning LLMs. Our CRV framework consists of multiple LLM agents, each specializing in unique abilities: (i) critiquing the CoTs according to the cognitive capabilities of smaller models, (ii) rethinking and refining these CoTs based on the critiques, and (iii) verifying the correctness of the refined results. We further propose the cognitive preference optimization (CogPO) algorithm to enhance the reasoning abilities of smaller models by aligning thoughts of these models with their cognitive capacities. Comprehensive evaluations on challenging reasoning benchmarks demonstrate the efficacy of CRV and CogPO, which outperforms other training methods by a large margin. 

**Abstract (ZH)**: 大型语言模型的推理能力通过深度思考取得了显著进步，如OpenAI的o1和DeepSeek-R1。然而，这些进步带来了巨大的资源需求，强调了探索训练高效推理语言模型的方法的重要性，方法需要使用远少于参数的数量。一个关键挑战是较小的模型在能力和认知轨迹上与较大的模型不同。因此，直接从大型语言模型中抽取链式思考（CoT）结果并传递给较小的模型可能是无效的，并且需要大量的标注数据。本文介绍了一种名为Critique-Rethink-Verify（CRV）的新框架，旨在训练更小但强大的推理语言模型。CRV框架由多个专门负责不同能力的语言模型代理组成：（i）根据较小模型的认知能力批评链式思考（CoT），（ii）根据批评重新思考并完善这些CoT，（iii）验证改进结果的正确性。我们进一步提出了认知偏好优化（CogPO）算法，通过使这些模型的思维与其认知能力相一致来增强较小模型的推理能力。在具有挑战性的推理基准测试上的综合评估表明，CRV和CogPO的有效性显著优于其他训练方法。 

---
# Multi-task Federated Learning with Encoder-Decoder Structure: Enabling Collaborative Learning Across Different Tasks 

**Title (ZH)**: 基于编码器-解码器结构的多任务联邦学习：实现不同任务间的协作学习 

**Authors**: Jingxuan Zhou, Weidong Bao, Ji Wang, Dayu Zhang, Xiongtao Zhang, Yaohong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09800)  

**Abstract**: Federated learning has been extensively studied and applied due to its ability to ensure data security in distributed environments while building better models. However, clients participating in federated learning still face limitations, as clients with different structures or tasks cannot participate in learning together. In view of this, constructing a federated learning framework that allows collaboration between clients with different model structures and performing different tasks, enabling them to share valuable knowledge to enhance model efficiency, holds significant practical implications for the widespread application of federated learning. To achieve this goal, we propose a multi-task federated learning with encoder-decoder structure (M-Fed). Specifically, given the widespread adoption of the encoder-decoder architecture in current models, we leverage this structure to share intra-task knowledge through traditional federated learning methods and extract general knowledge from the encoder to achieve cross-task knowledge sharing. The training process is similar to traditional federated learning, and we incorporate local decoder and global decoder information into the loss function. The local decoder iteratively updates and gradually approaches the global decoder until sufficient cross-task knowledge sharing is achieved. Our method is lightweight and modular, demonstrating innovation compared to previous research. It enables clients performing different tasks to share general knowledge while maintaining the efficiency of traditional federated learning systems. We conducted experiments on two widely used benchmark datasets to verify the feasibility of M-Fed and compared it with traditional methods. The experimental results demonstrate the effectiveness of M-Fed in multi-task federated learning. 

**Abstract (ZH)**: 联邦学习因其在分布式环境中保障数据安全的同时构建更好模型的能力而得到了广泛研究和应用。然而，参与联邦学习的客户端仍然面临局限性，即结构或任务不同的客户端无法共同参与学习。鉴于此，构建一个允许结构和任务不同的客户端进行合作的联邦学习框架，使它们能够分享有价值的知识以提高模型效率，对于联邦学习的广泛应用具有重要的实际意义。为实现这一目标，我们提出了一种基于编码器-解码器结构的多任务联邦学习方法（M-Fed）。具体而言，鉴于当前模型广泛采用编码器-解码器架构，我们利用这一结构通过传统联邦学习方法共享同一任务的知识，并借助编码器提取通用知识以实现跨任务知识共享。训练过程与传统联邦学习类似，我们将本地解码器和全局解码器信息纳入损失函数中。本地解码器迭代更新并逐渐接近全局解码器，直到实现足够的跨任务知识共享。我们的方法轻量且模块化，相较于先前研究具有创新性。它使执行不同任务的客户端能够共享通用知识，同时保持传统联邦学习系统的效率。我们在两个广泛使用的基准数据集上进行了实验以验证M-Fed的有效性，并将其与传统方法进行了比较。实验结果证明了M-Fed在多任务联邦学习中的有效性。 

---
# IGL-DT: Iterative Global-Local Feature Learning with Dual-Teacher Semantic Segmentation Framework under Limited Annotation Scheme 

**Title (ZH)**: IGL-DT：在有限注释方案下基于迭代全局-局部特征学习的双师语义分割框架 

**Authors**: Dinh Dai Quan Tran, Hoang-Thien Nguyen. Thanh-Huy Nguyen, Gia-Van To, Tien-Huy Nguyen, Quan Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2504.09797)  

**Abstract**: Semi-Supervised Semantic Segmentation (SSSS) aims to improve segmentation accuracy by leveraging a small set of labeled images alongside a larger pool of unlabeled data. Recent advances primarily focus on pseudo-labeling, consistency regularization, and co-training strategies. However, existing methods struggle to balance global semantic representation with fine-grained local feature extraction. To address this challenge, we propose a novel tri-branch semi-supervised segmentation framework incorporating a dual-teacher strategy, named IGL-DT. Our approach employs SwinUnet for high-level semantic guidance through Global Context Learning and ResUnet for detailed feature refinement via Local Regional Learning. Additionally, a Discrepancy Learning mechanism mitigates over-reliance on a single teacher, promoting adaptive feature learning. Extensive experiments on benchmark datasets demonstrate that our method outperforms state-of-the-art approaches, achieving superior segmentation performance across various data regimes. 

**Abstract (ZH)**: 半监督语义分割（Semi-Supervised Semantic Segmentation, SSSS）旨在通过利用少量标注图像和大量未标注数据来提高分割精度。 recent advances 主要侧重于伪标签生成、一致性正则化和协同训练策略。然而，现有方法难以平衡全局语义表示与精细局部特征提取。为了应对这一挑战，我们提出了一种新颖的三支路半监督分割框架，结合双师策略，命名为IGL-DT。我们的方法通过全局上下文学习模块SwinUnet提供高层语义指导，并通过局部区域学习模块ResUnet进行详细特征细化。此外，离散学习机制减轻了对单个教师的过度依赖，促进自适应特征学习。在基准数据集上的广泛实验表明，我们的方法优于现有的先进方法，在各种数据条件下实现了更优的分割性能。 

---
# VDocRAG: Retrieval-Augmented Generation over Visually-Rich Documents 

**Title (ZH)**: VDocRAG：丰富视觉元素的检索增强生成 

**Authors**: Ryota Tanaka, Taichi Iki, Taku Hasegawa, Kyosuke Nishida, Kuniko Saito, Jun Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2504.09795)  

**Abstract**: We aim to develop a retrieval-augmented generation (RAG) framework that answers questions over a corpus of visually-rich documents presented in mixed modalities (e.g., charts, tables) and diverse formats (e.g., PDF, PPTX). In this paper, we introduce a new RAG framework, VDocRAG, which can directly understand varied documents and modalities in a unified image format to prevent missing information that occurs by parsing documents to obtain text. To improve the performance, we propose novel self-supervised pre-training tasks that adapt large vision-language models for retrieval by compressing visual information into dense token representations while aligning them with textual content in documents. Furthermore, we introduce OpenDocVQA, the first unified collection of open-domain document visual question answering datasets, encompassing diverse document types and formats. OpenDocVQA provides a comprehensive resource for training and evaluating retrieval and question answering models on visually-rich documents in an open-domain setting. Experiments show that VDocRAG substantially outperforms conventional text-based RAG and has strong generalization capability, highlighting the potential of an effective RAG paradigm for real-world documents. 

**Abstract (ZH)**: 我们旨在开发一种检索增强生成（RAG）框架，用于回答以混合模态（如图表、表格）和多种格式（如PDF、PPTX）呈现的丰富视觉文档中的问题。在本文中，我们介绍了一种新的RAG框架VDocRAG，该框架可以直接理解和统一处理各类文档和模态的信息，避免了通过解析文档获取文本时可能出现的信息遗漏。为了提升性能，我们提出了新的自监督预训练任务，将大型视觉语言模型适应于检索任务，通过将视觉信息压缩为密集的token表示并与文档中的文本内容对齐来改进检索效果。此外，我们引入了OpenDocVQA，这是第一个统一的开放领域文档视觉问答数据集，涵盖了多种文档类型和格式。OpenDocVQA 为在开放领域环境下的视觉丰富文档进行检索和问答模型的训练和评估提供了全面的资源。实验结果显示VDocRAG显著优于传统的基于文本的RAG，并具有较强的泛化能力，突显了有效RAG范式在现实世界文档中的潜力。 

---
# EquiVDM: Equivariant Video Diffusion Models with Temporally Consistent Noise 

**Title (ZH)**: EquiVDM：具有时空一致噪声的自酉视频扩散模型 

**Authors**: Chao Liu, Arash Vahdat  

**Link**: [PDF](https://arxiv.org/pdf/2504.09789)  

**Abstract**: Temporally consistent video-to-video generation is essential for applications of video diffusion models in areas such as sim-to-real, style-transfer, video upsampling, etc. In this paper, we propose a video diffusion framework that leverages temporally consistent noise to generate coherent video frames without specialized modules or additional constraints. We show that the standard training objective of diffusion models, when applied with temporally consistent noise, encourages the model to be equivariant to spatial transformations in input video and noise. This enables our model to better follow motion patterns from the input video, producing aligned motion and high-fidelity frames. Furthermore, we extend our approach to 3D-consistent video generation by attaching noise as textures on 3D meshes, ensuring 3D consistency in sim-to-real applications. Experimental results demonstrate that our method surpasses state-of-the-art baselines in motion alignment, 3D consistency, and video quality while requiring only a few sampling steps in practice. 

**Abstract (ZH)**: 时空一致的视频到视频生成对于视频扩散模型在sim-to-real、风格迁移、视频上采样等领域的应用至关重要。本文提出一种利用时空一致噪声的视频扩散框架，无需特殊模块或额外约束即可生成连贯的视频帧。我们表明，当扩散模型的标准训练目标与时空一致噪声结合使用时，能够促使模型在输入视频和噪声的空间变换上具有协同性。这使我们的模型更好地跟随输入视频中的运动模式，产生对齐的运动和高保真帧。此外，我们通过将噪声作为3D网格上的纹理来扩展我们的方法，以确保在sim-to-real应用中的3D一致性。实验结果表明，与现有最先进的基线方法相比，我们的方法在运动对齐、3D一致性和视频质量方面表现更优，且仅需少量采样步骤即可实现。 

---
# Reasoning Court: Combining Reasoning, Action, and Judgment for Multi-Hop Reasoning 

**Title (ZH)**: 推理法庭：结合推理、行动与判断的多跳推理 

**Authors**: Jingtian Wu, Claire Cardie  

**Link**: [PDF](https://arxiv.org/pdf/2504.09781)  

**Abstract**: While large language models (LLMs) have demonstrated strong capabilities in tasks like question answering and fact verification, they continue to suffer from hallucinations and reasoning errors, especially in multi-hop tasks that require integration of multiple information sources. Current methods address these issues through retrieval-based techniques (grounding reasoning in external evidence), reasoning-based approaches (enhancing coherence via improved prompting), or hybrid strategies combining both elements. One prominent hybrid method, ReAct, has outperformed purely retrieval-based or reasoning-based approaches; however, it lacks internal verification of intermediate reasoning steps, allowing potential errors to propagate through complex reasoning tasks. In this paper, we introduce Reasoning Court (RC), a novel framework that extends iterative reasoning-and-retrieval methods, such as ReAct, with a dedicated LLM judge. Unlike ReAct, RC employs this judge to independently evaluate multiple candidate answers and their associated reasoning generated by separate LLM agents. The judge is asked to select the answer that it considers the most factually grounded and logically coherent based on the presented reasoning and evidence, or synthesizes a new answer using available evidence and its pre-trained knowledge if all candidates are inadequate, flawed, or invalid. Evaluations on multi-hop benchmarks (HotpotQA, MuSiQue) and fact-verification (FEVER) demonstrate that RC consistently outperforms state-of-the-art few-shot prompting methods without task-specific fine-tuning. 

**Abstract (ZH)**: 基于推理和检索的方法的Reasoning Court：一个新颖的框架 

---
# "All Roads Lead to ChatGPT": How Generative AI is Eroding Social Interactions and Student Learning Communities 

**Title (ZH)**: “条条大路通ChatGPT”：生成式人工智能如何侵蚀社会互动和学生学习社区 

**Authors**: Irene Hou, Owen Man, Kate Hamilton, Srishty Muthusekaran, Jeffin Johnykutty, Leili Zadeh, Stephen MacNeil  

**Link**: [PDF](https://arxiv.org/pdf/2504.09779)  

**Abstract**: The widespread adoption of generative AI is already impacting learning and help-seeking. While the benefits of generative AI are well-understood, recent studies have also raised concerns about increased potential for cheating and negative impacts on students' metacognition and critical thinking. However, the potential impacts on social interactions, peer learning, and classroom dynamics are not yet well understood. To investigate these aspects, we conducted 17 semi-structured interviews with undergraduate computing students across seven R1 universities in North America. Our findings suggest that help-seeking requests are now often mediated by generative AI. For example, students often redirected questions from their peers to generative AI instead of providing assistance themselves, undermining peer interaction. Students also reported feeling increasingly isolated and demotivated as the social support systems they rely on begin to break down. These findings are concerning given the important role that social interactions play in students' learning and sense of belonging. 

**Abstract (ZH)**: 生成式人工智能的广泛应用已影响学习和求助行为。尽管生成式人工智能带来的益处已被充分了解，但最近的研究也提出了关于其增加作弊可能性以及对学生成本元认知和批判性思维负面影响的担忧。然而，其对社会互动、同伴学习和课堂动态的影响尚未得到充分理解。为了探究这些方面，我们对北美洲七所R1大学的本科生进行了17次半结构化访谈。研究发现，求助请求现在经常通过生成式人工智能进行中介。例如，学生常将问题从同伴转向生成式人工智能，而不是相互提供帮助，从而削弱了同伴互动。此外，学生还报告称，随着他们依赖的社会支持系统开始瓦解，他们感到越来越孤立和缺乏动力。这些发现令人担忧，因为社会互动在学生的学习和归属感中扮演着重要角色。 

---
# Reasoning without Regret 

**Title (ZH)**: 无需遗憾的推理 

**Authors**: Tarun Chitra  

**Link**: [PDF](https://arxiv.org/pdf/2504.09777)  

**Abstract**: Chain-of-thought reasoning enables large language models to solve multi-step tasks by framing problem solving as sequential decision problems. Outcome-based rewards, which provide feedback only on final answers, show impressive success, but face challenges with credit assignment and slow convergence. In contrast, procedure-based rewards offer efficient step-level feedback, but typically require costly human supervision. We introduce \emph{Backwards Adaptive Reward Shaping} (BARS), a no-regret framework that converts sparse outcomes-based rewards into effective procedure-based signals. BARS uses sparse rewards generated from terminal-state priors and cover trees to scale rewards while preventing exploitation. With Bellman contraction and $(\Delta, \epsilon)$-gap rewards, our backward Euler solver achieves $\epsilon$-accuracy in $O\left((R_{\max}/\Delta)\log(1/\epsilon)\right)$ iterations with $O(\log T)$ dynamic regret over $T$ rounds. Our analysis, based on generic chaining, continuous scaling limits, and non-linear Feynman-Kac bounds, connects recent outcome-based methods' empirical successes with the benefits of intermediate supervision. Combined, this provides the first rigorous no-regret algorithm for outcome reward shaping, providing a theoretical foundation for the empirical success of DeepSeek's R1. 

**Abstract (ZH)**: Backwards Adaptive Reward Shaping for Outcome-Based Reward Shaping 

---
# Understanding and Optimizing Multi-Stage AI Inference Pipelines 

**Title (ZH)**: 理解与优化多阶段人工智能推理管道 

**Authors**: Abhimanyu Rajeshkumar Bambhaniya, Hanjiang Wu, Suvinay Subramanian, Sudarshan Srinivasan, Souvik Kundu, Amir Yazdanbakhsh, Midhilesh Elavazhagan, Madhu Kumar, Tushar Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2504.09775)  

**Abstract**: The rapid evolution of Large Language Models (LLMs) has driven the need for increasingly sophisticated inference pipelines and hardware platforms. Modern LLM serving extends beyond traditional prefill-decode workflows, incorporating multi-stage processes such as Retrieval Augmented Generation (RAG), key-value (KV) cache retrieval, dynamic model routing, and multi step reasoning. These stages exhibit diverse computational demands, requiring distributed systems that integrate GPUs, ASICs, CPUs, and memory-centric architectures. However, existing simulators lack the fidelity to model these heterogeneous, multi-engine workflows, limiting their ability to inform architectural decisions.
To address this gap, we introduce HERMES, a Heterogeneous Multi-stage LLM inference Execution Simulator. HERMES models diverse request stages; including RAG, KV retrieval, reasoning, prefill, and decode across complex hardware hierarchies. HERMES supports heterogeneous clients executing multiple models concurrently unlike prior frameworks while incorporating advanced batching strategies and multi-level memory hierarchies. By integrating real hardware traces with analytical modeling, HERMES captures critical trade-offs such as memory bandwidth contention, inter-cluster communication latency, and batching efficiency in hybrid CPU-accelerator deployments. Through case studies, we explore the impact of reasoning stages on end-to-end latency, optimal batching strategies for hybrid pipelines, and the architectural implications of remote KV cache retrieval. HERMES empowers system designers to navigate the evolving landscape of LLM inference, providing actionable insights into optimizing hardware-software co-design for next-generation AI workloads. 

**Abstract (ZH)**: 异构多阶段大型语言模型推理执行模拟器：HERMES 

---
# Executable Functional Abstractions: Inferring Generative Programs for Advanced Math Problems 

**Title (ZH)**: 可执行的功能抽象：推断生成式程序以解决高级数学问题 

**Authors**: Zaid Khan, Elias Stengel-Eskin, Archiki Prasad, Jaemin Cho, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2504.09763)  

**Abstract**: Scientists often infer abstract procedures from specific instances of problems and use the abstractions to generate new, related instances. For example, programs encoding the formal rules and properties of a system have been useful in fields ranging from RL (procedural environments) to physics (simulation engines). These programs can be seen as functions which execute to different outputs based on their parameterizations (e.g., gridworld configuration or initial physical conditions). We introduce the term EFA (Executable Functional Abstraction) to denote such programs for math problems. EFA-like constructs have been shown to be useful for math reasoning as problem generators for stress-testing models. However, prior work has been limited to abstractions for grade-school math (whose simple rules are easy to encode in programs), while generating EFAs for advanced math has thus far required human engineering. We explore the automatic construction of EFAs for advanced math problems. We operationalize the task of automatically constructing EFAs as a program synthesis task, and develop EFAGen, which conditions an LLM on a seed math problem and its step-by-step solution to generate candidate EFA programs that are faithful to the generalized problem and solution class underlying the seed problem. Furthermore, we formalize properties any valid EFA must possess in terms of executable unit tests, and show how the tests can be used as verifiable rewards to train LLMs to become better writers of EFAs. We demonstrate that EFAs constructed by EFAGen behave rationally by remaining faithful to seed problems, produce learnable problem variations, and that EFAGen can infer EFAs across multiple diverse sources of competition-level math problems. Finally, we show downstream uses of model-written EFAs e.g. finding problem variations that are harder or easier for a learner to solve, as well as data generation. 

**Abstract (ZH)**: 科学家经常从问题的具体实例中推断出抽象的程序，并使用这些抽象来生成新的相关实例。例如，编码系统的形式规则和属性的程序已经在从RL（过程环境）到物理学（模拟引擎）等多个领域中发挥了重要作用。这些程序可以被视为基于其参数化执行以产生不同输出的函数（例如，网格世界配置或初始物理条件）。我们引入术语EFA（可执行功能抽象）来表示用于数学问题的此类程序。类似于EFA的构造已被证明对数学推理有用，作为模型的压力测试问题生成器。然而，以前的工作仅限于对基础数学问题的抽象（其简单的规则易于在程序中编码），而目前仍需要人力工程来生成高级数学问题的EFA。我们探索自动构建高级数学问题的EFA。我们将自动构建EFA的任务操作化为程序合成任务，并开发EFAGen，该工具根据种子数学问题及其逐步解决方案来生成忠实于种子问题所基于的概括性问题和解决方案类别的候选EFA程序。此外，我们以可执行的单元测试形式正式化任何有效的EFA必须具备的特性，并展示了这些测试如何作为验证奖励来训练LLM以成为更好的EFA编写者。我们证明，由EFAGen构建的EFA能够理性地忠实于种子问题，生成可学习的问题变异，并且EFAGen可以从多个不同的竞赛级数学问题来源推断出EFA。最后，我们展示了模型撰写的EFA的下游用途，例如找到对于学习者来说更难或更简单的解决问题变异，以及数据生成。 

---
# Improving Multilingual Capabilities with Cultural and Local Knowledge in Large Language Models While Enhancing Native Performance 

**Title (ZH)**: 在增强本地性能的同时提升大型语言模型的多语言能力及其文化与地域知识应用 

**Authors**: Ram Mohan Rao Kadiyala, Siddartha Pullakhandam, Siddhant Gupta, Drishti Sharma, Jebish Purbey, Kanwal Mehreen, Muhammad Arham, Hamza Farooq  

**Link**: [PDF](https://arxiv.org/pdf/2504.09753)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities, but their development has primarily focused on English and other high-resource languages, leaving many languages underserved. We present our latest Hindi-English bi-lingual LLM \textbf{Mantra-14B} with ~3\% average improvement in benchmark scores over both languages, outperforming models twice its size. Using a curated dataset composed of English and Hindi instruction data of 485K samples, we instruction tuned models such as Qwen-2.5-14B-Instruct and Phi-4 to improve performance over both English and Hindi. Our experiments encompassing seven different LLMs of varying parameter sizes and over 140 training attempts with varying English-Hindi training data ratios demonstrated that it is possible to significantly improve multilingual performance without compromising native performance. Further, our approach avoids resource-intensive techniques like vocabulary expansion or architectural modifications, thus keeping the model size small. Our results indicate that modest fine-tuning with culturally and locally informed data can bridge performance gaps without incurring significant computational overhead. We release our training code, datasets, and models under mit and apache licenses to aid further research towards under-represented and low-resource languages. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了显著的能力，但其发展主要集中在英语和其他高资源语言上，忽视了许多语言的需求。我们介绍了我们最新开发的Hindi-English双语LLM——Mantra-14B，在两个语言上的基准分数均实现了约3%的平均提升，且优于其两倍大小的模型。通过使用包含48.5万样本的精心筛选的英汉指令数据集，我们对Qwen-2.5-14B-Instruct和Phi-4模型进行了指令微调，以提升两个语言的性能。涵盖七种不同参数规模的LLM以及超过140次训练尝试，并采用不同英汉训练数据比例，我们的实验表明，在不牺牲母语性能的前提下显著提升多语言性能是可能的。此外，我们的方法避免了词汇扩展或架构修改等资源密集型技术，从而保持了模型规模较小。我们的结果表明，适度使用文化和本地化数据的微调可以在不增加显著计算开销的情况下缩小性能差距。我们以MIT和Apache许可释放了训练代码、数据集和模型，以促进对未充分代表和低资源语言的研究。 

---
# Automatic Detection of Intro and Credits in Video using CLIP and Multihead Attention 

**Title (ZH)**: 使用CLIP和多头注意力自动检测视频中的片头和字幕 

**Authors**: Vasilii Korolkov, Andrey Yanchenko  

**Link**: [PDF](https://arxiv.org/pdf/2504.09738)  

**Abstract**: Detecting transitions between intro/credits and main content in videos is a crucial task for content segmentation, indexing, and recommendation systems. Manual annotation of such transitions is labor-intensive and error-prone, while heuristic-based methods often fail to generalize across diverse video styles. In this work, we introduce a deep learning-based approach that formulates the problem as a sequence-to-sequence classification task, where each second of a video is labeled as either "intro" or "film." Our method extracts frames at a fixed rate of 1 FPS, encodes them using CLIP (Contrastive Language-Image Pretraining), and processes the resulting feature representations with a multihead attention model incorporating learned positional encoding. The system achieves an F1-score of 91.0%, Precision of 89.0%, and Recall of 97.0% on the test set, and is optimized for real-time inference, achieving 11.5 FPS on CPU and 107 FPS on high-end GPUs. This approach has practical applications in automated content indexing, highlight detection, and video summarization. Future work will explore multimodal learning, incorporating audio features and subtitles to further enhance detection accuracy. 

**Abstract (ZH)**: 基于深度学习的视频宣传片与正片过渡检测方法 

---
# Dynamik: Syntactically-Driven Dynamic Font Sizing for Emphasis of Key Information 

**Title (ZH)**: 动态字体大小调整：基于句法的关键信息强调 

**Authors**: Naoto Nishida, Yoshio Ishiguro, Jun Rekiomto, Naomi Yamashita  

**Link**: [PDF](https://arxiv.org/pdf/2504.09734)  

**Abstract**: In today's globalized world, there are increasing opportunities for individuals to communicate using a common non-native language (lingua franca). Non-native speakers often have opportunities to listen to foreign languages, but may not comprehend them as fully as native speakers do. To aid real-time comprehension, live transcription of subtitles is frequently used in everyday life (e.g., during Zoom conversations, watching YouTube videos, or on social networking sites). However, simultaneously reading subtitles while listening can increase cognitive load. In this study, we propose Dynamik, a system that reduces cognitive load during reading by decreasing the size of less important words and enlarging important ones, thereby enhancing sentence contrast. Our results indicate that Dynamik can reduce certain aspects of cognitive load, specifically, participants' perceived performance and effort among individuals with low proficiency in English, as well as enhance the users' sense of comprehension, especially among people with low English ability. We further discuss our methods' applicability to other languages and potential improvements and further research directions. 

**Abstract (ZH)**: 全球化背景下，个体使用共同的非母语语言（通用语）进行沟通的机会日益增多。非母语使用者常常有机会聆听外语，但可能不像母语使用者那样完全理解。为了实时理解，日常生活中经常使用实时字幕转录（例如在Zoom对话、观看YouTube视频或在社交媒体上）。然而，同时阅读字幕和聆听可能会增加认知负荷。本研究提出了一种名为Dynamik的系统，通过减少不重要词汇的大小并放大重要词汇，从而降低阅读时的认知负荷，增强句子对比度。研究表明，Dynamik可以降低低英语 proficiency 用户的认知负荷感和努力感，并增强用户的理解感，特别是在低英语能力用户中更为明显。我们进一步讨论了该方法在其他语言中的适用性和潜在改进以及进一步的研究方向。 

---
# Adapting Robot's Explanation for Failures Based on Observed Human Behavior in Human-Robot Collaboration 

**Title (ZH)**: 基于人类行为观察的机器人故障解释适应性调整研究 

**Authors**: Andreas Naoum, Parag Khanna, Elmira Yadollahi, Mårten Björkman, Christian Smith  

**Link**: [PDF](https://arxiv.org/pdf/2504.09717)  

**Abstract**: This work aims to interpret human behavior to anticipate potential user confusion when a robot provides explanations for failure, allowing the robot to adapt its explanations for more natural and efficient collaboration. Using a dataset that included facial emotion detection, eye gaze estimation, and gestures from 55 participants in a user study, we analyzed how human behavior changed in response to different types of failures and varying explanation levels. Our goal is to assess whether human collaborators are ready to accept less detailed explanations without inducing confusion. We formulate a data-driven predictor to predict human confusion during robot failure explanations. We also propose and evaluate a mechanism, based on the predictor, to adapt the explanation level according to observed human behavior. The promising results from this evaluation indicate the potential of this research in adapting a robot's explanations for failures to enhance the collaborative experience. 

**Abstract (ZH)**: 本研究旨在解释人类行为，以预测当机器人在解释故障时用户可能产生的潜在困惑，从而使机器人能够根据需要调整其解释以实现更自然和高效的协作。通过包含55名参与者用户研究中面部情绪检测、眼动估计和手势的数据集，我们分析了人类行为在面对不同类型的故障和不同解释水平时的变化。我们的目标是评估人类合作者在接受较不详细解释时是否会产生困惑。我们提出了一个基于数据的预测器来预测机器人故障解释期间的人类困惑。我们还提出并评估了一种机制，该机制根据观察到的人类行为调整解释水平。这项评估取得的有希望的结果表明了这项研究在调整机器人故障解释以增强协作体验方面的潜力。 

---
# Dominated Actions in Imperfect-Information Games 

**Title (ZH)**: imperfect-information 游戏中的支配行动 

**Authors**: Sam Ganzfried  

**Link**: [PDF](https://arxiv.org/pdf/2504.09716)  

**Abstract**: Dominance is a fundamental concept in game theory. In strategic-form games dominated strategies can be identified in polynomial time. As a consequence, iterative removal of dominated strategies can be performed efficiently as a preprocessing step for reducing the size of a game before computing a Nash equilibrium. For imperfect-information games in extensive form, we could convert the game to strategic form and then iteratively remove dominated strategies in the same way; however, this conversion may cause an exponential blowup in game size. In this paper we define and study the concept of dominated actions in imperfect-information games. Our main result is a polynomial-time algorithm for determining whether an action is dominated (strictly or weakly) by any mixed strategy in n-player games, which can be extended to an algorithm for iteratively removing dominated actions. This allows us to efficiently reduce the size of the game tree as a preprocessing step for Nash equilibrium computation. We explore the role of dominated actions empirically in the "All In or Fold" No-Limit Texas Hold'em poker variant. 

**Abstract (ZH)**: dominance在博弈论中是一个基本概念。在战略型博弈中，占优策略可以在多项式时间内被识别。因此，可以通过迭代移除占优策略作为预处理步骤，减少博弈规模后再计算纳什均衡，这可以高效地进行。对于不完善信息的扩展形式博弈，可以将博弈转换为战略型形式并同样迭代移除占优策略；然而，这种转换可能导致博弈规模的指数级增长。本文定义并研究了不完善信息博弈中的占优行动概念。我们的主要成果是提出了一个多项式时间算法来确定一个行动是否被任何混合策略严格地或弱地占优（适用于n-player博弈），该算法可以扩展为迭代移除占优行动的算法。这使我们能够在计算纳什均衡之前作为预处理步骤，有效减少博弈树的规模。我们通过实验探索了占优行动在“全押或弃牌”无限德州扑克变种中的作用。 

---
# Evaluating the Quality of Benchmark Datasets for Low-Resource Languages: A Case Study on Turkish 

**Title (ZH)**: 低资源语言基准数据集质量评估：以土耳其语为例 

**Authors**: Ayşe Aysu Cengiz, Ahmet Kaan Sever, Elif Ecem Ümütlü, Naime Şeyma Erdem, Burak Aytan, Büşra Tufan, Abdullah Topraksoy, Esra Darıcı, Cagri Toraman  

**Link**: [PDF](https://arxiv.org/pdf/2504.09714)  

**Abstract**: The reliance on translated or adapted datasets from English or multilingual resources introduces challenges regarding linguistic and cultural suitability. This study addresses the need for robust and culturally appropriate benchmarks by evaluating the quality of 17 commonly used Turkish benchmark datasets. Using a comprehensive framework that assesses six criteria, both human and LLM-judge annotators provide detailed evaluations to identify dataset strengths and shortcomings.
Our results reveal that 70% of the benchmark datasets fail to meet our heuristic quality standards. The correctness of the usage of technical terms is the strongest criterion, but 85% of the criteria are not satisfied in the examined datasets. Although LLM judges demonstrate potential, they are less effective than human annotators, particularly in understanding cultural common sense knowledge and interpreting fluent, unambiguous text. GPT-4o has stronger labeling capabilities for grammatical and technical tasks, while Llama3.3-70B excels at correctness and cultural knowledge evaluation. Our findings emphasize the urgent need for more rigorous quality control in creating and adapting datasets for low-resource languages. 

**Abstract (ZH)**: 依赖于从英语或多种语言资源翻译或改编的语料库引入了语言和文化适宜性方面的问题。本研究通过评估17个常用土耳其语基准数据集的质量，来解决 robust 和文化适应性基准的需求。使用综合框架评估六项指标，人和LLM评判员提供详细的评估来识别数据集的优点和不足。

我们的结果显示，70%的基准数据集未能达到我们的启发式质量标准。技术术语使用正确性是最重要的指标，但在检查的数据集中，85%的指标未被满足。尽管LLM评判员显示出潜力，但在理解和解释文化常识知识以及解释流畅、无歧义的文本方面，它们的效果不如人类标注者。GPT-4o在语法和技术任务的标签能力方面更强，而Llama3.3-70B在正确性和文化知识评估方面表现出色。我们的研究结果强调了在为低资源语言创建和改编数据集时进行更严格质量控制的迫切需求。 

---
# The Structural Safety Generalization Problem 

**Title (ZH)**: 结构安全泛化问题 

**Authors**: Julius Broomfield, Tom Gibbs, Ethan Kosak-Hine, George Ingebretsen, Tia Nasir, Jason Zhang, Reihaneh Iranmanesh, Sara Pieri, Reihaneh Rabbany, Kellin Pelrine  

**Link**: [PDF](https://arxiv.org/pdf/2504.09712)  

**Abstract**: LLM jailbreaks are a widespread safety challenge. Given this problem has not yet been tractable, we suggest targeting a key failure mechanism: the failure of safety to generalize across semantically equivalent inputs. We further focus the target by requiring desirable tractability properties of attacks to study: explainability, transferability between models, and transferability between goals. We perform red-teaming within this framework by uncovering new vulnerabilities to multi-turn, multi-image, and translation-based attacks. These attacks are semantically equivalent by our design to their single-turn, single-image, or untranslated counterparts, enabling systematic comparisons; we show that the different structures yield different safety outcomes. We then demonstrate the potential for this framework to enable new defenses by proposing a Structure Rewriting Guardrail, which converts an input to a structure more conducive to safety assessment. This guardrail significantly improves refusal of harmful inputs, without over-refusing benign ones. Thus, by framing this intermediate challenge - more tractable than universal defenses but essential for long-term safety - we highlight a critical milestone for AI safety research. 

**Abstract (ZH)**: LLM禁用突破是广泛存在的安全挑战。鉴于这一问题尚未变得可处理，我们建议针对一个关键失败机制：安全性的泛化能力在语义等价输入上的失败。我们还通过要求攻击具有有利于研究的可处理特性来进行目标聚焦：可解释性、模型间的迁移性以及目标间的迁移性。我们在此框架下进行红队测试，揭露了对多轮、多图和翻译攻击的新漏洞。这些攻击是通过设计与单轮、单图或未翻译版本在语义上等价的，这使得我们可以进行系统比较；我们展示出不同的结构会导致不同的安全结果。我们随后通过提出一种结构重写护栏，展示了此框架在潜在上能够启用新颖防御的示例，该护栏将输入转换为更有利于安全评估的结构。这种护栏显著提高了拒绝有害输入的能力，而不会过度拒绝良性输入。因此，通过将这种中间挑战视为比通用防御更具处理性但长期安全性不可或缺的问题，我们突显了AI安全研究中的一个关键里程碑。 

---
# Transformer-Based Representation Learning for Robust Gene Expression Modeling and Cancer Prognosis 

**Title (ZH)**: 基于Transformer的表示学习方法用于健壮的基因表达建模与癌症预后 

**Authors**: Shuai Jiang, Saeed Hassanpour  

**Link**: [PDF](https://arxiv.org/pdf/2504.09704)  

**Abstract**: Transformer-based models have achieved remarkable success in natural language and vision tasks, but their application to gene expression analysis remains limited due to data sparsity, high dimensionality, and missing values. We present GexBERT, a transformer-based autoencoder framework for robust representation learning of gene expression data. GexBERT learns context-aware gene embeddings by pretraining on large-scale transcriptomic profiles with a masking and restoration objective that captures co-expression relationships among thousands of genes. We evaluate GexBERT across three critical tasks in cancer research: pan-cancer classification, cancer-specific survival prediction, and missing value imputation. GexBERT achieves state-of-the-art classification accuracy from limited gene subsets, improves survival prediction by restoring expression of prognostic anchor genes, and outperforms conventional imputation methods under high missingness. Furthermore, its attention-based interpretability reveals biologically meaningful gene patterns across cancer types. These findings demonstrate the utility of GexBERT as a scalable and effective tool for gene expression modeling, with translational potential in settings where gene coverage is limited or incomplete. 

**Abstract (ZH)**: 基于Transformer的框架GexBERT在基因表达数据分析中的稳健表示学习 

---
# Migrating Code At Scale With LLMs At Google 

**Title (ZH)**: 使用Google的大型语言模型规模化迁移代码 

**Authors**: Celal Ziftci, Stoyan Nikolov, Anna Sjövall, Bo Kim, Daniele Codecasa, Max Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.09691)  

**Abstract**: Developers often evolve an existing software system by making internal changes, called migration. Moving to a new framework, changing implementation to improve efficiency, and upgrading a dependency to its latest version are examples of migrations.
Migration is a common and typically continuous maintenance task undertaken either manually or through tooling. Certain migrations are labor intensive and costly, developers do not find the required work rewarding, and they may take years to complete. Hence, automation is preferred for such migrations.
In this paper, we discuss a large-scale, costly and traditionally manual migration project at Google, propose a novel automated algorithm that uses change location discovery and a Large Language Model (LLM) to aid developers conduct the migration, report the results of a large case study, and discuss lessons learned.
Our case study on 39 distinct migrations undertaken by three developers over twelve months shows that a total of 595 code changes with 93,574 edits have been submitted, where 74.45% of the code changes and 69.46% of the edits were generated by the LLM. The developers reported high satisfaction with the automated tooling, and estimated a 50% reduction on the total time spent on the migration compared to earlier manual migrations.
Our results suggest that our automated, LLM-assisted workflow can serve as a model for similar initiatives. 

**Abstract (ZH)**: 开发者经常通过内部变更来演化的现有软件系统，称为迁移。切换到新的框架、改进实现以提高效率以及将依赖项升级到最新版本都是迁移的例子。

迁移是常见的通常连续的维护任务，可以通过人工操作或工具完成。某些迁移劳动密集且成本高，开发者对他们所需的工作没有足够的满足感，它们可能需要数年才能完成。因此，自动化在这种迁移中更受欢迎。

在本文中，我们讨论了谷歌的一个大规模、高成本且传统上需要人工执行的迁移项目，提出了一种新颖的自动化算法，该算法利用更改位置发现和大型语言模型（LLM）帮助开发人员进行迁移，报告了大规模案例研究的结果，并讨论了所学的经验教训。

我们的案例研究显示，三位开发人员在十二个月内完成了39个不同的迁移，共提交了595次代码更改和93,574次编辑，其中74.45%的代码更改和69.46%的编辑是由LLM生成的。开发人员对自动化工具表示高度满意，并估计与之前的手动迁移相比，迁移所花费的总时间减少了50%。

我们的结果表明，我们的自动化、LLM辅助的工作流程可以作为类似倡议的模型。 

---
# Can LLMs Revolutionize the Design of Explainable and Efficient TinyML Models? 

**Title (ZH)**: LLM能否推动可解释和高效TinyML模型的设计革命？ 

**Authors**: Christophe El Zeinaty, Wassim Hamidouche, Glenn Herrou, Daniel Menard, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2504.09685)  

**Abstract**: This paper introduces a novel framework for designing efficient neural network architectures specifically tailored to tiny machine learning (TinyML) platforms. By leveraging large language models (LLMs) for neural architecture search (NAS), a vision transformer (ViT)-based knowledge distillation (KD) strategy, and an explainability module, the approach strikes an optimal balance between accuracy, computational efficiency, and memory usage. The LLM-guided search explores a hierarchical search space, refining candidate architectures through Pareto optimization based on accuracy, multiply-accumulate operations (MACs), and memory metrics. The best-performing architectures are further fine-tuned using logits-based KD with a pre-trained ViT-B/16 model, which enhances generalization without increasing model size. Evaluated on the CIFAR-100 dataset and deployed on an STM32H7 microcontroller (MCU), the three proposed models, LMaNet-Elite, LMaNet-Core, and QwNet-Core, achieve accuracy scores of 74.50%, 74.20% and 73.00%, respectively. All three models surpass current state-of-the-art (SOTA) models, such as MCUNet-in3/in4 (69.62% / 72.86%) and XiNet (72.27%), while maintaining a low computational cost of less than 100 million MACs and adhering to the stringent 320 KB static random-access memory (SRAM) constraint. These results demonstrate the efficiency and performance of the proposed framework for TinyML platforms, underscoring the potential of combining LLM-driven search, Pareto optimization, KD, and explainability to develop accurate, efficient, and interpretable models. This approach opens new possibilities in NAS, enabling the design of efficient architectures specifically suited for TinyML. 

**Abstract (ZH)**: 一种基于大规模语言模型的高效神经网络架构设计框架：适用于TinyML平台的优化方法 

---
# SPOT: Spatio-Temporal Pattern Mining and Optimization for Load Consolidation in Freight Transportation Networks 

**Title (ZH)**: SPOT：货运运输网络中基于时空模式挖掘与优化的负载整合方法 

**Authors**: Sikai Cheng, Amira Hijazi, Jeren Konak, Alan Erera, Pascal Van Hentenryck  

**Link**: [PDF](https://arxiv.org/pdf/2504.09680)  

**Abstract**: Freight consolidation has significant potential to reduce transportation costs and mitigate congestion and pollution. An effective load consolidation plan relies on carefully chosen consolidation points to ensure alignment with existing transportation management processes, such as driver scheduling, personnel planning, and terminal operations. This complexity represents a significant challenge when searching for optimal consolidation strategies. Traditional optimization-based methods provide exact solutions, but their computational complexity makes them impractical for large-scale instances and they fail to leverage historical data. Machine learning-based approaches address these issues but often ignore operational constraints, leading to infeasible consolidation plans.
This work proposes SPOT, an end-to-end approach that integrates the benefits of machine learning (ML) and optimization for load consolidation. The ML component plays a key role in the planning phase by identifying the consolidation points through spatio-temporal clustering and constrained frequent itemset mining, while the optimization selects the most cost-effective feasible consolidation routes for a given operational day. Extensive experiments conducted on industrial load data demonstrate that SPOT significantly reduces travel distance and transportation costs (by about 50% on large terminals) compared to the existing industry-standard load planning strategy and a neighborhood-based heuristic. Moreover, the ML component provides valuable tactical-level insights by identifying frequently recurring consolidation opportunities that guide proactive planning. In addition, SPOT is computationally efficient and can be easily scaled to accommodate large transportation networks. 

**Abstract (ZH)**: 货物 consolidation 在降低运输成本和缓解拥堵与污染方面具有显著潜力。有效的负载 consolidation 计划依赖于精心选择的 consolidation 点，以确保与现有的运输管理流程，如驾驶员调度、人员规划和码头运营等相一致。这一复杂性构成了寻找最优 consolidation 策略的显著挑战。传统基于优化的方法能提供精确的解决方案，但其计算复杂性使其不适合大规模实例，并且无法利用历史数据。基于机器学习的方法解决了这些难题，但通常忽视了操作约束，导致不可行的 consolidation 计划。

本文提出了一种端到端的框架 SPOT，该框架将机器学习（ML）和优化的优势结合起来用于负载 consolidation。机器学习组件在规划阶段发挥关键作用，通过时空聚类和受限频繁项挖掘来识别 consolidation 点，而优化则为给定的操作日选择最具成本效益的可行 consolidation 路线。在工业负载数据上的 extensive 实验表明，与现有行业标准的负载规划策略和基于邻域的启发式方法相比，SPOT 可显著减少行驶距离和运输成本（大型码头情况下约减少 50%）。此外，机器学习组件提供了有价值的战术级见解，通过识别频繁出现的 consolidation 机会来指导主动规划。此外，SPOT 具有计算效率，并且可以轻松扩展以适应大规模的运输网络。 

---
# AgentDynEx: Nudging the Mechanics and Dynamics of Multi-Agent Simulations 

**Title (ZH)**: AgentDynEx: 倾斜多代理模拟的机理与动力学 

**Authors**: Jenny Ma, Riya Sahni, Karthik Sreedhar, Lydia B. Chilton  

**Link**: [PDF](https://arxiv.org/pdf/2504.09662)  

**Abstract**: Multi-agent large language model simulations have the potential to model complex human behaviors and interactions. If the mechanics are set up properly, unanticipated and valuable social dynamics can surface. However, it is challenging to consistently enforce simulation mechanics while still allowing for notable and emergent dynamics. We present AgentDynEx, an AI system that helps set up simulations from user-specified mechanics and dynamics. AgentDynEx uses LLMs to guide users through a Configuration Matrix to identify core mechanics and define milestones to track dynamics. It also introduces a method called \textit{nudging}, where the system dynamically reflects on simulation progress and gently intervenes if it begins to deviate from intended outcomes. A technical evaluation found that nudging enables simulations to have more complex mechanics and maintain its notable dynamics compared to simulations without nudging. We discuss the importance of nudging as a technique for balancing mechanics and dynamics of multi-agent simulations. 

**Abstract (ZH)**: 多智能体大型语言模型模拟具有潜在能力来 modeling 复杂的人类行为和互动。如果机制设置得当，未预见且有价值的社会动态可以浮现。然而，在确保一致执行模拟机制的同时，仍然允许显著且自发的动态发生具有挑战性。我们介绍了 AgentDynEx，这是一种 AI 系统，旨在从用户指定的机制和动态中帮助设置模拟。AgentDynEx 使用大语言模型引导用户通过配置矩阵来识别核心机制并定义跟踪动态的里程碑。此外，它引入了一种称为“引导”的方法，系统会动态地反思模拟进度，并在开始偏离预期结果时温和地干预。技术评估发现，与没有引导的模拟相比，引导使模拟具有更复杂的机制并能够保持其显著动态。我们讨论了引导作为平衡多智能体模拟中机制和动态的技术的重要性。 

---
# Myanmar XNLI: Building a Dataset and Exploring Low-resource Approaches to Natural Language Inference with Myanmar 

**Title (ZH)**: Myanmar XNLI: 构建数据集并探索低资源自然语言推理方法 

**Authors**: Aung Kyaw Htet, Mark Dras  

**Link**: [PDF](https://arxiv.org/pdf/2504.09645)  

**Abstract**: Despite dramatic recent progress in NLP, it is still a major challenge to apply Large Language Models (LLM) to low-resource languages. This is made visible in benchmarks such as Cross-Lingual Natural Language Inference (XNLI), a key task that demonstrates cross-lingual capabilities of NLP systems across a set of 15 languages. In this paper, we extend the XNLI task for one additional low-resource language, Myanmar, as a proxy challenge for broader low-resource languages, and make three core contributions. First, we build a dataset called Myanmar XNLI (myXNLI) using community crowd-sourced methods, as an extension to the existing XNLI corpus. This involves a two-stage process of community-based construction followed by expert verification; through an analysis, we demonstrate and quantify the value of the expert verification stage in the context of community-based construction for low-resource languages. We make the myXNLI dataset available to the community for future research. Second, we carry out evaluations of recent multilingual language models on the myXNLI benchmark, as well as explore data-augmentation methods to improve model performance. Our data-augmentation methods improve model accuracy by up to 2 percentage points for Myanmar, while uplifting other languages at the same time. Third, we investigate how well these data-augmentation methods generalise to other low-resource languages in the XNLI dataset. 

**Abstract (ZH)**: 尽管自然语言处理在近期取得了显著进展，但将大型语言模型应用于低资源语言仍然是一个主要挑战。这一挑战在跨语言自然语言推理（XNLI）基准测试中尤为明显，XNLI是一个关键任务，展示了自然语言处理系统在一组15种语言中的跨语言能力。在本文中，我们扩展了XNLI任务，将一个额外的低资源语言缅甸语纳入其中，作为更广泛低资源语言的一个代理挑战，并作出了三个核心贡献。首先，我们使用社区众包方法构建了一个名为缅甸XNLI（myXNLI）的数据集，作为XNLI现有语料库的扩充；这一过程包括基于社区的构建阶段和专家验证阶段；通过分析，我们展示了在低资源语言的社区构建过程中专家验证阶段的价值，并量化了其贡献。我们向社区提供了myXNLI数据集，以供未来研究。其次，我们在myXNLI基准上评估了近期的多语言语言模型，并探究了数据扩增方法以提高模型性能。我们的数据扩增方法使缅甸语模型的准确性最高提升了2个百分点，同时提高了其他语言的性能。第三，我们研究了这些数据扩增方法在XNLI数据集中其他低资源语言中的泛化能力。 

---
# Slow Thinking for Sequential Recommendation 

**Title (ZH)**: 慢思考用于序列推荐 

**Authors**: Junjie Zhang, Beichen Zhang, Wenqi Sun, Hongyu Lu, Wayne Xin Zhao, Yu Chen, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2504.09627)  

**Abstract**: To develop effective sequential recommender systems, numerous methods have been proposed to model historical user behaviors. Despite the effectiveness, these methods share the same fast thinking paradigm. That is, for making recommendations, these methods typically encodes user historical interactions to obtain user representations and directly match these representations with candidate item representations. However, due to the limited capacity of traditional lightweight recommendation models, this one-step inference paradigm often leads to suboptimal performance. To tackle this issue, we present a novel slow thinking recommendation model, named STREAM-Rec. Our approach is capable of analyzing historical user behavior, generating a multi-step, deliberative reasoning process, and ultimately delivering personalized recommendations. In particular, we focus on two key challenges: (1) identifying the suitable reasoning patterns in recommender systems, and (2) exploring how to effectively stimulate the reasoning capabilities of traditional recommenders. To this end, we introduce a three-stage training framework. In the first stage, the model is pretrained on large-scale user behavior data to learn behavior patterns and capture long-range dependencies. In the second stage, we design an iterative inference algorithm to annotate suitable reasoning traces by progressively refining the model predictions. This annotated data is then used to fine-tune the model. Finally, in the third stage, we apply reinforcement learning to further enhance the model generalization ability. Extensive experiments validate the effectiveness of our proposed method. 

**Abstract (ZH)**: 开发有效的序列推荐系统的方法已提出了许多种，用于建模用户的历史行为。尽管这些方法有效，但它们大多遵循快速思维范式。具体而言，这些方法通常通过编码用户的 histórico 行为互动来获取用户表示，并直接将这些表示与候选项目表示进行匹配，以进行推荐。然而，由于传统轻量级推荐模型容量有限，这种一步推理范式往往导致性能不佳。为应对这一问题，我们提出了一种新的慢思考推荐模型，名为 STREAM-Rec。我们的方法能够分析用户历史行为，生成多步、详尽的推理过程，并最终提供个性化推荐。特别是在两个关键挑战上：(1) 在推荐系统中识别合适的推理模式，(2) 探索如何有效地激发传统推荐器的推理能力。为此，我们引入了一个三阶段训练框架。在第一阶段，模型在大规模用户行为数据上进行预训练，以学习行为模式并捕捉长期依赖关系。在第二阶段，我们设计了一种迭代推理算法，通过逐步改进模型预测来标注合适的推理轨迹。然后使用这些标注数据进行模型微调。最后，在第三阶段，我们应用强化学习进一步增强模型的泛化能力。广泛实验验证了我们提出方法的有效性。 

---
# Ges3ViG: Incorporating Pointing Gestures into Language-Based 3D Visual Grounding for Embodied Reference Understanding 

**Title (ZH)**: Ges3ViG：将指向手势融入基于语言的三维视觉定位，以理解具身参考 

**Authors**: Atharv Mahesh Mane, Dulanga Weerakoon, Vigneshwaran Subbaraju, Sougata Sen, Sanjay E. Sarma, Archan Misra  

**Link**: [PDF](https://arxiv.org/pdf/2504.09623)  

**Abstract**: 3-Dimensional Embodied Reference Understanding (3D-ERU) combines a language description and an accompanying pointing gesture to identify the most relevant target object in a 3D scene. Although prior work has explored pure language-based 3D grounding, there has been limited exploration of 3D-ERU, which also incorporates human pointing gestures. To address this gap, we introduce a data augmentation framework-Imputer, and use it to curate a new benchmark dataset-ImputeRefer for 3D-ERU, by incorporating human pointing gestures into existing 3D scene datasets that only contain language instructions. We also propose Ges3ViG, a novel model for 3D-ERU that achieves ~30% improvement in accuracy as compared to other 3D-ERU models and ~9% compared to other purely language-based 3D grounding models. Our code and dataset are available at this https URL. 

**Abstract (ZH)**: 三维嵌入式参考理解（3D-ERU）结合了语言描述和伴随的指点手势，以识别3D场景中最相关的目标物体。为了填补这一空白，我们引入了一个数据增强框架-Imputer，并使用它通过将人类指点手势纳入仅包含语言说明的现有3D场景数据集，来构建一个新的基准数据集-ImputeRefer，用于3D-ERU。我们还提出了一种新型模型Ges3ViG，该模型相比其他3D-ERU模型 Accuracy 提高约30%，相比其他纯语言为基础的3D对齐模型 Accuracy 提高约9%。我们的代码和数据集可在以下链接获取：this https URL。 

---
# Metropolis-Hastings Captioning Game: Knowledge Fusion of Vision Language Models via Decentralized Bayesian Inference 

**Title (ZH)**: 基于梅特罗波利斯-哈特灵采样的话语生成游戏：分散式贝叶斯推断驱动的视觉语言模型知识融合 

**Authors**: Yuta Matsui, Ryosuke Yamaki, Ryo Ueda, Seitaro Shinagawa, Tadahiro Taniguchi  

**Link**: [PDF](https://arxiv.org/pdf/2504.09620)  

**Abstract**: We propose the Metropolis-Hastings Captioning Game (MHCG), a method to fuse knowledge of multiple vision-language models (VLMs) by learning from each other. Although existing methods that combine multiple models suffer from inference costs and architectural constraints, MHCG avoids these problems by performing decentralized Bayesian inference through a process resembling a language game. The knowledge fusion process establishes communication between two VLM agents alternately captioning images and learning from each other. We conduct two image-captioning experiments with two VLMs, each pre-trained on a different dataset. The first experiment demonstrates that MHCG achieves consistent improvement in reference-free evaluation metrics. The second experiment investigates how MHCG contributes to sharing VLMs' category-level vocabulary by observing the occurrence of the vocabulary in the generated captions. 

**Abstract (ZH)**: 我们提出了一种名为Metropolis-Hastings Captioning Game (MHCG)的方法，这是一种通过相互学习融合多种视觉-语言模型（VLMs）知识的方法。通过一个类似语言游戏的过程进行去中心化的贝叶斯推断，MHCG避免了现有方法面临的推理成本和架构限制问题。知识融合过程通过交替对图像进行说明并相互学习在两个VLM代理之间建立通信。我们使用两个分别在不同数据集上预训练的VLMs进行了两项图像说明实验。第一个实验展示了MHCG在参考无关评估指标中的持续改进。第二个实验探讨了MHCG如何通过观察生成说明中词汇的出现来促进VLMs类别级别词汇的共享。 

---
# A highly maneuverable flying squirrel drone with agility-improving foldable wings 

**Title (ZH)**: 具有 agility 提升可折叠翼的高机动飞行毛猬无人机 

**Authors**: Dohyeon Lee, Jun-Gill Kang, Soohee Han  

**Link**: [PDF](https://arxiv.org/pdf/2504.09609)  

**Abstract**: Drones, like most airborne aerial vehicles, face inherent disadvantages in achieving agile flight due to their limited thrust capabilities. These physical constraints cannot be fully addressed through advancements in control algorithms alone. Drawing inspiration from the winged flying squirrel, this paper proposes a highly maneuverable drone equipped with agility-enhancing foldable wings. By leveraging collaborative control between the conventional propeller system and the foldable wings-coordinated through the Thrust-Wing Coordination Control (TWCC) framework-the controllable acceleration set is expanded, enabling the generation of abrupt vertical forces that are unachievable with traditional wingless drones. The complex aerodynamics of the foldable wings are modeled using a physics-assisted recurrent neural network (paRNN), which calibrates the angle of attack (AOA) to align with the real aerodynamic behavior of the wings. The additional air resistance generated by appropriately deploying these wings significantly improves the tracking performance of the proposed "flying squirrel" drone. The model is trained on real flight data and incorporates flat-plate aerodynamic principles. Experimental results demonstrate that the proposed flying squirrel drone achieves a 13.1% improvement in tracking performance, as measured by root mean square error (RMSE), compared to a conventional wingless drone. A demonstration video is available on YouTube: this https URL. 

**Abstract (ZH)**: 基于仿翼 squirrel 飞行器的敏捷无人机及其控制方法 

---
# Fine-tuning an Large Language Model for Automating Computational Fluid Dynamics Simulations 

**Title (ZH)**: 大型语言模型的微调以自动化计算流体力学模拟 

**Authors**: Zhehao Dong, Zhen Lu, Yue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09602)  

**Abstract**: Configuring computational fluid dynamics (CFD) simulations typically demands extensive domain expertise, limiting broader access. Although large language models (LLMs) have advanced scientific computing, their use in automating CFD workflows is underdeveloped. We introduce a novel approach centered on domain-specific LLM adaptation. By fine-tuning Qwen2.5-7B-Instruct on NL2FOAM, our custom dataset of 28716 natural language-to-OpenFOAM configuration pairs with chain-of-thought (CoT) annotations, we enable direct translation from natural language descriptions to executable CFD setups. A multi-agent framework orchestrates the process, autonomously verifying inputs, generating configurations, running simulations, and correcting errors. Evaluation on a benchmark of 21 diverse flow cases demonstrates state-of-the-art performance, achieving 88.7% solution accuracy and 82.6% first-attempt success rate. This significantly outperforms larger general-purpose models like Qwen2.5-72B-Instruct, DeepSeek-R1, and Llama3.3-70B-Instruct, while also requiring fewer correction iterations and maintaining high computational efficiency. The results highlight the critical role of domain-specific adaptation in deploying LLM assistants for complex engineering workflows. 

**Abstract (ZH)**: 一种基于领域特定大语言模型适应的计算流体动力学工作流自动化方法 

---
# TextSplat: Text-Guided Semantic Fusion for Generalizable Gaussian Splatting 

**Title (ZH)**: TextSplat: 文本导向的语义融合用于通用高斯散点图 

**Authors**: Zhicong Wu, Hongbin Xu, Gang Xu, Ping Nie, Zhixin Yan, Jinkai Zheng, Liangqiong Qu, Ming Li, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2504.09588)  

**Abstract**: Recent advancements in Generalizable Gaussian Splatting have enabled robust 3D reconstruction from sparse input views by utilizing feed-forward Gaussian Splatting models, achieving superior cross-scene generalization. However, while many methods focus on geometric consistency, they often neglect the potential of text-driven guidance to enhance semantic understanding, which is crucial for accurately reconstructing fine-grained details in complex scenes. To address this limitation, we propose TextSplat--the first text-driven Generalizable Gaussian Splatting framework. By employing a text-guided fusion of diverse semantic cues, our framework learns robust cross-modal feature representations that improve the alignment of geometric and semantic information, producing high-fidelity 3D reconstructions. Specifically, our framework employs three parallel modules to obtain complementary representations: the Diffusion Prior Depth Estimator for accurate depth information, the Semantic Aware Segmentation Network for detailed semantic information, and the Multi-View Interaction Network for refined cross-view features. Then, in the Text-Guided Semantic Fusion Module, these representations are integrated via the text-guided and attention-based feature aggregation mechanism, resulting in enhanced 3D Gaussian parameters enriched with detailed semantic cues. Experimental results on various benchmark datasets demonstrate improved performance compared to existing methods across multiple evaluation metrics, validating the effectiveness of our framework. The code will be publicly available. 

**Abstract (ZH)**: Recent advancements in通用可泛化高斯绘射技术通过利用前馈高斯绘射模型，从稀疏输入视角中实现稳健的三维重建，并获得跨场景的优良泛化能力。然而，尽管许多方法关注几何一致性，它们往往忽视了文本驱动指导在增强语义理解方面的潜力，这对于准确重建复杂场景中的精细细节至关重要。为解决这一局限性，我们提出TextSplat——首个文本驱动的通用可泛化高斯绘射框架。通过采用文本引导的多模态特征融合，我们的框架学习到稳健的跨模态特征表示，提高了几何和语义信息的对齐，生成高保真三维重建。具体地，我们的框架采用三个并行模块来获得互补表示：用于准确深度估计的扩散先验深度估计器、用于详细语义信息的语义意识分割网络以及用于精细跨视图特征的多视角交互网络。然后，在文本引导语义融合模块中，这些表示通过文本引导和注意力机制的特征聚合机制进行整合，生成富含详细语义线索的增强三维高斯参数。在多个基准数据集上的实验结果表明，与现有方法相比，在多项评估指标上表现更好，验证了我们框架的有效性。代码将公开。 

---
# AirVista-II: An Agentic System for Embodied UAVs Toward Dynamic Scene Semantic Understanding 

**Title (ZH)**: AirVista-II: 一个自主系统用于动态场景语义理解的 embodied UAVs 

**Authors**: Fei Lin, Yonglin Tian, Tengchao Zhang, Jun Huang, Sangtian Guan, Fei-Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09583)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) are increasingly important in dynamic environments such as logistics transportation and disaster response. However, current tasks often rely on human operators to monitor aerial videos and make operational decisions. This mode of human-machine collaboration suffers from significant limitations in efficiency and adaptability. In this paper, we present AirVista-II -- an end-to-end agentic system for embodied UAVs, designed to enable general-purpose semantic understanding and reasoning in dynamic scenes. The system integrates agent-based task identification and scheduling, multimodal perception mechanisms, and differentiated keyframe extraction strategies tailored for various temporal scenarios, enabling the efficient capture of critical scene information. Experimental results demonstrate that the proposed system achieves high-quality semantic understanding across diverse UAV-based dynamic scenarios under a zero-shot setting. 

**Abstract (ZH)**: 无人机（UAVs）在物流运输和灾难响应等动态环境中的应用越来越重要。然而，当前任务往往依赖于人类操作员监控空中视频并作出操作决策。这种人机协作模式在效率和适应性方面存在显著局限。本文介绍了一种端到端的自主系统AirVista-II，旨在使具身无人机具备对动态场景的一般语义理解和推理能力。该系统结合了基于代理的任务识别和调度、多模态感知机制以及针对不同时间场景定制的关键帧提取策略，能够高效地捕获关键场景信息。实验结果表明，在零样本情况下，所提系统在多样化的无人机动态场景中实现了高质量的语义理解。 

---
# A simulation-heuristics dual-process model for intuitive physics 

**Title (ZH)**: 直观物理的仿真-启发式双过程模型 

**Authors**: Shiqian Li, Yuxi Ma, Jiajun Yan, Bo Dai, Yujia Peng, Chi Zhang, Yixin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.09546)  

**Abstract**: The role of mental simulation in human physical reasoning is widely acknowledged, but whether it is employed across scenarios with varying simulation costs and where its boundary lies remains unclear. Using a pouring-marble task, our human study revealed two distinct error patterns when predicting pouring angles, differentiated by simulation time. While mental simulation accurately captured human judgments in simpler scenarios, a linear heuristic model better matched human predictions when simulation time exceeded a certain boundary. Motivated by these observations, we propose a dual-process framework, Simulation-Heuristics Model (SHM), where intuitive physics employs simulation for short-time simulation but switches to heuristics when simulation becomes costly. By integrating computational methods previously viewed as separate into a unified model, SHM quantitatively captures their switching mechanism. The SHM aligns more precisely with human behavior and demonstrates consistent predictive performance across diverse scenarios, advancing our understanding of the adaptive nature of intuitive physical reasoning. 

**Abstract (ZH)**: 心智模拟在人类物理推理中的作用得到了广泛认可，但在不同模拟成本的场景下是否被普遍使用以及其边界尚不清楚。通过抛珠实验，我们的研究揭示了在预测倾倒角度时存在两种不同的错误模式，这些模式受模拟时间的影响。心智模拟在简单场景中能准确捕捉人类判断，但在模拟时间超出一定边界时，线性启发式模型能更好地匹配人类预测。基于这些观察，我们提出了一种双重过程框架——心智模拟-启发式模型（SHM），该模型认为直观物理在短时间内使用模拟，而在模拟成本增加时转而使用启发式策略。通过将先前被认为是独立的计算方法整合到一个统一模型中，SHM 能量化地捕捉它们的转换机制。SHM 更精确地符合人类行为，并在不同场景中表现出一致的预测性能，有助于我们理解直观物理推理的适应性本质。 

---
# Embodied Chain of Action Reasoning with Multi-Modal Foundation Model for Humanoid Loco-manipulation 

**Title (ZH)**: 具身行动链推理结合多模态基础模型的人形动Manipulation 

**Authors**: Yu Hao, Geeta Chandra Raju Bethala, Niraj Pudasaini, Hao Huang, Shuaihang Yuan, Congcong Wen, Baoru Huang, Anh Nguyen, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09532)  

**Abstract**: Enabling humanoid robots to autonomously perform loco-manipulation tasks in complex, unstructured environments poses significant challenges. This entails equipping robots with the capability to plan actions over extended horizons while leveraging multi-modality to bridge gaps between high-level planning and actual task execution. Recent advancements in multi-modal foundation models have showcased substantial potential in enhancing planning and reasoning abilities, particularly in the comprehension and processing of semantic information for robotic control tasks. In this paper, we introduce a novel framework based on foundation models that applies the embodied chain of action reasoning methodology to autonomously plan actions from textual instructions for humanoid loco-manipulation. Our method integrates humanoid-specific chain of thought methodology, including detailed affordance and body movement analysis, which provides a breakdown of the task into a sequence of locomotion and manipulation actions. Moreover, we incorporate spatial reasoning based on the observation and target object properties to effectively navigate where target position may be unseen or occluded. Through rigorous experimental setups on object rearrangement, manipulations and loco-manipulation tasks on a real-world environment, we evaluate our method's efficacy on the decoupled upper and lower body control and demonstrate the effectiveness of the chain of robotic action reasoning strategies in comprehending human instructions. 

**Abstract (ZH)**: 自主在复杂非结构化环境中执行类人机器人拾放任务面临显著挑战：基于多模态基础模型的类人机器人连贯动作自主规划方法 

---
# How new data permeates LLM knowledge and how to dilute it 

**Title (ZH)**: 新数据如何渗透到LLM知识中以及如何稀释它 

**Authors**: Chen Sun, Renat Aksitov, Andrey Zhmoginov, Nolan Andrew Miller, Max Vladymyrov, Ulrich Rueckert, Been Kim, Mark Sandler  

**Link**: [PDF](https://arxiv.org/pdf/2504.09522)  

**Abstract**: Large language models learn and continually learn through the accumulation of gradient-based updates, but how individual pieces of new information affect existing knowledge, leading to both beneficial generalization and problematic hallucination, remains poorly understood. We demonstrate that when learning new information, LLMs exhibit a "priming" effect: learning a new fact can cause the model to inappropriately apply that knowledge in unrelated contexts. To systematically study this phenomenon, we introduce "Outlandish," a carefully curated dataset of 1320 diverse text samples designed to probe how new knowledge permeates through an LLM's existing knowledge base. Using this dataset, we show that the degree of priming after learning new information can be predicted by measuring the token probability of key words before learning. This relationship holds robustly across different model architectures (PALM-2, Gemma, Llama), sizes, and training stages. Finally, we develop two novel techniques to modulate how new knowledge affects existing model behavior: (1) a ``stepping-stone'' text augmentation strategy and (2) an ``ignore-k'' update pruning method. These approaches reduce undesirable priming effects by 50-95\% while preserving the model's ability to learn new information. Our findings provide both empirical insights into how LLMs learn and practical tools for improving the specificity of knowledge insertion in language models. Further materials: this https URL 

**Abstract (ZH)**: 大规模语言模型通过梯度更新累积学习并持续学习，但个体新信息如何影响现有知识、导致有益泛化和问题性幻觉仍然了解不足。我们展示了当学习新信息时，LLMs表现出一种“激发”效应：学习一个新事实会导致模型不当应用该知识于不相关的情境中。为了系统地研究这一现象，我们引入了“Outlandish”数据集，这是一个精心挑选的包含1320个多样文本样本的资料集，旨在探究新知识如何渗透到语言模型的现有知识库中。使用该数据集，我们展示了在学习新信息后，激发效应的程度可以通过测量学习前关键词的token概率来预测。这种关系在不同的模型架构（PALM-2、Gemma、Llama）、规模和训练阶段中均保持稳健。最后，我们开发了两种新颖的技术来调节新知识对现有模型行为的影响：（1）“脚手架”文本增强策略和（2）“忽略-k”更新剪枝方法。这些方法在减少不必要的激发效应的同时，保留了模型学习新信息的能力。我们的发现既提供了关于语言模型学习的实证见解，也为提高语言模型知识插入的特异性提供了实用工具。 

---
# Decoding the mechanisms of the Hattrick football manager game using Bayesian network structure learning for optimal decision-making 

**Title (ZH)**: 使用贝叶斯网络结构学习解析Hatrick足球经理游戏的机制以实现最优决策 

**Authors**: Anthony C. Constantinou, Nicholas Higgins, Neville K. Kitson  

**Link**: [PDF](https://arxiv.org/pdf/2504.09499)  

**Abstract**: Hattrick is a free web-based probabilistic football manager game with over 200,000 users competing for titles at national and international levels. Launched in Sweden in 1997 as part of an MSc project, the game's slow-paced design has fostered a loyal community, with many users remaining active for decades. Hattrick's game-engine mechanics are partially hidden, and users have attempted to decode them with incremental success over the years. Rule-based, statistical and machine learning models have been developed to aid this effort and are widely used by the community. However, these models or tools have not been formally described or evaluated in the scientific literature. This study is the first to explore Hattrick using structure learning techniques and Bayesian networks, integrating both data and domain knowledge to develop models capable of explaining and simulating the game engine. We present a comprehensive analysis assessing the effectiveness of structure learning algorithms in relation to knowledge-based structures, and show that while structure learning may achieve a higher overall network fit, it does not result in more accurate predictions for selected variables of interest, when compared to knowledge-based networks that produce a lower overall network fit. Additionally, we introduce and publicly share a fully specified Bayesian network model that matches the performance of top models used by the Hattrick community. We further demonstrate how analysis extends beyond prediction by providing a visual representation of conditional dependencies, and using the best performing Bayesian network model for in-game decision-making. To support future research, we make all data, graphical structures, and models publicly available online. 

**Abstract (ZH)**: Hattrick作为一种基于网页的概率足球管理游戏，拥有超过200,000名用户，在国家级和国际级别上角逐冠军。该游戏于1997年在瑞典作为一项硕士项目的一部分推出，其缓慢的游戏节奏培养了一个忠诚的社区，许多用户已经活跃数十年。Hattrick的游戏机制部分隐藏，用户在过去几年中试图解码这些机制并取得了一定的成功。基于规则、统计和机器学习模型已被开发用于辅助这一努力，并在社区中广泛使用。然而，这些模型或工具并未在科学文献中正式描述或评估。本研究首次使用结构学习技术和贝叶斯网络探讨Hattrick，结合数据和领域知识开发能够解释和模拟游戏机制的模型。我们呈现了一个全面的分析，评估了结构学习算法在知识导向结构下的有效性，并展示了虽然结构学习可能实现更高的整体网络拟合度，但在与产生较低整体拟合度的知识导向网络进行比较时，并未对选定的变量产生更准确的预测。此外，我们引入并公开分享了一个完全指定的贝叶斯网络模型，其性能与Hattrick社区中使用的顶级模型相当。我们进一步展示分析如何超越预测，通过提供条件依赖的可视化表示，并使用性能最优的贝叶斯网络模型进行游戏中决策。为了支持未来的研究，我们在线公开了所有数据、图形结构和模型。 

---
# Federated Prototype Graph Learning 

**Title (ZH)**: 联邦原型图学习 

**Authors**: Zhengyu Wu, Xunkai Li, Yinlin Zhu, Rong-Hua Li, Guoren Wang, Chenghu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.09493)  

**Abstract**: In recent years, Federated Graph Learning (FGL) has gained significant attention for its distributed training capabilities in graph-based machine intelligence applications, mitigating data silos while offering a new perspective for privacy-preserve large-scale graph learning. However, multi-level FGL heterogeneity presents various client-server collaboration challenges: (1) Model-level: The variation in clients for expected performance and scalability necessitates the deployment of heterogeneous models. Unfortunately, most FGL methods rigidly demand identical client models due to the direct model weight aggregation on the server. (2) Data-level: The intricate nature of graphs, marked by the entanglement of node profiles and topology, poses an optimization dilemma. This implies that models obtained by federated training struggle to achieve superior performance. (3) Communication-level: Some FGL methods attempt to increase message sharing among clients or between clients and the server to improve training, which inevitably leads to high communication costs. In this paper, we propose FedPG as a general prototype-guided optimization method for the above multi-level FGL heterogeneity. Specifically, on the client side, we integrate multi-level topology-aware prototypes to capture local graph semantics. Subsequently, on the server side, leveraging the uploaded prototypes, we employ topology-guided contrastive learning and personalized technology to tailor global prototypes for each client, broadcasting them to improve local training. Experiments demonstrate that FedPG outperforms SOTA baselines by an average of 3.57\% in accuracy while reducing communication costs by 168x. 

**Abstract (ZH)**: 联邦图学习中的多层异质性及其通用原型导向优化方法：FedPG 

---
# HalluShift: Measuring Distribution Shifts towards Hallucination Detection in LLMs 

**Title (ZH)**: HalluShift: 评估大型语言模型错误生成分布偏移的检测方法 

**Authors**: Sharanya Dasgupta, Sujoy Nath, Arkaprabha Basu, Pourya Shamsolmoali, Swagatam Das  

**Link**: [PDF](https://arxiv.org/pdf/2504.09482)  

**Abstract**: Large Language Models (LLMs) have recently garnered widespread attention due to their adeptness at generating innovative responses to the given prompts across a multitude of domains. However, LLMs often suffer from the inherent limitation of hallucinations and generate incorrect information while maintaining well-structured and coherent responses. In this work, we hypothesize that hallucinations stem from the internal dynamics of LLMs. Our observations indicate that, during passage generation, LLMs tend to deviate from factual accuracy in subtle parts of responses, eventually shifting toward misinformation. This phenomenon bears a resemblance to human cognition, where individuals may hallucinate while maintaining logical coherence, embedding uncertainty within minor segments of their speech. To investigate this further, we introduce an innovative approach, HalluShift, designed to analyze the distribution shifts in the internal state space and token probabilities of the LLM-generated responses. Our method attains superior performance compared to existing baselines across various benchmark datasets. Our codebase is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）由于其在众多领域生成创新响应的能力而 Recently Attracted Widespread Attention, but Often Suffer from Inherent Limitations Such as Hallucinations and Incorrect Information While Maintaining Well-Structured and Cohesive Responses. In This Work, We Hypothesize that Hallucinations Stem from the Internal Dynamics of LLMs. Our Observations Indicate that During Passage Generation, LLMs Tend to Deviate from Factual Accuracy in Subtle Parts of Responses, Eventually Shifting Toward Misinformation. This Phenomenon Bears a Resemblance to Human Cognition, Where Individuals May Hallucinate While Maintaining Logical Coherence, Embedding Uncertainty within Minor Segments of Their Speech. To Investigate This Further, We Introduce HalluShift, an Innovative Approach Designed to Analyze the Distribution Shifts in the Internal State Space and Token Probabilities of LLM-Generated Responses. Our Method Attains Superior Performance Compared to Existing Baselines Across Various Benchmark Datasets. Our Codebase is Available at This https URL. 

---
# Vision-Language Model for Object Detection and Segmentation: A Review and Evaluation 

**Title (ZH)**: 视觉-语言模型在物体检测与分割中的应用：综述与评估 

**Authors**: Yongchao Feng, Yajie Liu, Shuai Yang, Wenrui Cai, Jinqing Zhang, Qiqi Zhan, Ziyue Huang, Hongxi Yan, Qiao Wan, Chenguang Liu, Junzhe Wang, Jiahui Lv, Ziqi Liu, Tengyuan Shi, Qingjie Liu, Yunhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09480)  

**Abstract**: Vision-Language Model (VLM) have gained widespread adoption in Open-Vocabulary (OV) object detection and segmentation tasks. Despite they have shown promise on OV-related tasks, their effectiveness in conventional vision tasks has thus far been unevaluated. In this work, we present the systematic review of VLM-based detection and segmentation, view VLM as the foundational model and conduct comprehensive evaluations across multiple downstream tasks for the first time: 1) The evaluation spans eight detection scenarios (closed-set detection, domain adaptation, crowded objects, etc.) and eight segmentation scenarios (few-shot, open-world, small object, etc.), revealing distinct performance advantages and limitations of various VLM architectures across tasks. 2) As for detection tasks, we evaluate VLMs under three finetuning granularities: \textit{zero prediction}, \textit{visual fine-tuning}, and \textit{text prompt}, and further analyze how different finetuning strategies impact performance under varied task. 3) Based on empirical findings, we provide in-depth analysis of the correlations between task characteristics, model architectures, and training methodologies, offering insights for future VLM design. 4) We believe that this work shall be valuable to the pattern recognition experts working in the fields of computer vision, multimodal learning, and vision foundation models by introducing them to the problem, and familiarizing them with the current status of the progress while providing promising directions for future research. A project associated with this review and evaluation has been created at this https URL. 

**Abstract (ZH)**: 基于视觉-语言模型的检测与分割研究：多下游任务的系统评估 

---
# A highly maneuverable flying squirrel drone with controllable foldable wings 

**Title (ZH)**: 可折叠翼片高度机动的飞鼠无人机 

**Authors**: Jun-Gill Kang, Dohyeon Lee, Soohee Han  

**Link**: [PDF](https://arxiv.org/pdf/2504.09478)  

**Abstract**: Typical drones with multi rotors are generally less maneuverable due to unidirectional thrust, which may be unfavorable to agile flight in very narrow and confined spaces. This paper suggests a new bio-inspired drone that is empowered with high maneuverability in a lightweight and easy-to-carry way. The proposed flying squirrel inspired drone has controllable foldable wings to cover a wider range of flight attitudes and provide more maneuverable flight capability with stable tracking performance. The wings of a drone are fabricated with silicone membranes and sophisticatedly controlled by reinforcement learning based on human-demonstrated data. Specially, such learning based wing control serves to capture even the complex aerodynamics that are often impossible to model mathematically. It is shown through experiment that the proposed flying squirrel drone intentionally induces aerodynamic drag and hence provides the desired additional repulsive force even under saturated mechanical thrust. This work is very meaningful in demonstrating the potential of biomimicry and machine learning for realizing an animal-like agile drone. 

**Abstract (ZH)**: 受松鼠启发的生物灵感无人机：轻量化高性能机动飞行技术 

---
# MigGPT: Harnessing Large Language Models for Automated Migration of Out-of-Tree Linux Kernel Patches Across Versions 

**Title (ZH)**: MigGPT: 利用大规模语言模型进行跨版本Linux内核补丁的自动化迁移 

**Authors**: Pucheng Dang, Di Huang, Dong Li, Kang Chen, Yuanbo Wen, Qi Guo, Xing Hu, Ninghui Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.09474)  

**Abstract**: Out-of-tree kernel patches are essential for adapting the Linux kernel to new hardware or enabling specific functionalities. Maintaining and updating these patches across different kernel versions demands significant effort from experienced engineers. Large language models (LLMs) have shown remarkable progress across various domains, suggesting their potential for automating out-of-tree kernel patch migration. However, our findings reveal that LLMs, while promising, struggle with incomplete code context understanding and inaccurate migration point identification. In this work, we propose MigGPT, a framework that employs a novel code fingerprint structure to retain code snippet information and incorporates three meticulously designed modules to improve the migration accuracy and efficiency of out-of-tree kernel patches. Furthermore, we establish a robust benchmark using real-world out-of-tree kernel patch projects to evaluate LLM capabilities. Evaluations show that MigGPT significantly outperforms the direct application of vanilla LLMs, achieving an average completion rate of 72.59% (50.74% improvement) for migration tasks. 

**Abstract (ZH)**: 树外内核补丁对于将Linux内核适配到新硬件或启用特定功能至关重要。维护和更新这些补丁跨越不同内核版本需要有经验的工程师付出巨大努力。大规模语言模型(LLMs)在各个领域取得了显著进展，表明其在自动化树外内核补丁迁移方面的潜在能力。然而，我们的发现表明，尽管LLMs有前景，它们在理解不完整代码上下文和识别准确的迁移点方面仍然存在挑战。在这项工作中，我们提出了一种名为MigGPT的框架，该框架采用了一种新颖的代码指纹结构来保留代码片段信息，并结合了三个精心设计的模块以提高树外内核补丁迁移的准确性和效率。此外，我们使用真实世界的树外内核补丁项目建立了稳健的基准，以评估LLMs的能力。评估结果显示，MigGPT在迁移任务中显著优于直接应用原始的LLMs，实现了72.59%的平均完成率（相比基线提升50.74%）。 

---
# Comorbidity-Informed Transfer Learning for Neuro-developmental Disorder Diagnosis 

**Title (ZH)**: 共病导向的迁移学习在神经发育障碍诊断中的应用 

**Authors**: Xin Wen, Shijie Guo, Wenbo Ning, Rui Cao, Jie Xiang, Xiaobo Liu, Jintai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.09463)  

**Abstract**: Neuro-developmental disorders are manifested as dysfunctions in cognition, communication, behaviour and adaptability, and deep learning-based computer-aided diagnosis (CAD) can alleviate the increasingly strained healthcare resources on neuroimaging. However, neuroimaging such as fMRI contains complex spatio-temporal features, which makes the corresponding representations susceptible to a variety of distractions, thus leading to less effective in CAD. For the first time, we present a Comorbidity-Informed Transfer Learning(CITL) framework for diagnosing neuro-developmental disorders using fMRI. In CITL, a new reinforced representation generation network is proposed, which first combines transfer learning with pseudo-labelling to remove interfering patterns from the temporal domain of fMRI and generates new representations using encoder-decoder architecture. The new representations are then trained in an architecturally simple classification network to obtain CAD model. In particular, the framework fully considers the comorbidity mechanisms of neuro-developmental disorders and effectively integrates them with semi-supervised learning and transfer learning, providing new perspectives on interdisciplinary. Experimental results demonstrate that CITL achieves competitive accuracies of 76.32% and 73.15% for detecting autism spectrum disorder and attention deficit hyperactivity disorder, respectively, which outperforms existing related transfer learning work for 7.2% and 0.5% respectively. 

**Abstract (ZH)**: 基于共病的转移学习在功能性磁共振成像中诊断神经发育障碍 

---
# Measuring Leakage in Concept-Based Methods: An Information Theoretic Approach 

**Title (ZH)**: 基于概念的方法中的泄露量测：一种信息论方法 

**Authors**: Mikael Makonnen, Moritz Vandenhirtz, Sonia Laguna, Julia E Vogt  

**Link**: [PDF](https://arxiv.org/pdf/2504.09459)  

**Abstract**: Concept Bottleneck Models (CBMs) aim to enhance interpretability by structuring predictions around human-understandable concepts. However, unintended information leakage, where predictive signals bypass the concept bottleneck, compromises their transparency. This paper introduces an information-theoretic measure to quantify leakage in CBMs, capturing the extent to which concept embeddings encode additional, unintended information beyond the specified concepts. We validate the measure through controlled synthetic experiments, demonstrating its effectiveness in detecting leakage trends across various configurations. Our findings highlight that feature and concept dimensionality significantly influence leakage, and that classifier choice impacts measurement stability, with XGBoost emerging as the most reliable estimator. Additionally, preliminary investigations indicate that the measure exhibits the anticipated behavior when applied to soft joint CBMs, suggesting its reliability in leakage quantification beyond fully synthetic settings. While this study rigorously evaluates the measure in controlled synthetic experiments, future work can extend its application to real-world datasets. 

**Abstract (ZH)**: CBMs的信息论度量：量化概念瓶颈模型中的信息泄露 

---
# FROG: Effective Friend Recommendation in Online Games via Modality-aware User Preferences 

**Title (ZH)**: FROG：基于模态aware用户偏好在线游戏中有效的朋友推荐 

**Authors**: Qiwei Wang, Dandan Lin, Wenqing Lin, Ziming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.09428)  

**Abstract**: Due to the convenience of mobile devices, the online games have become an important part for user entertainments in reality, creating a demand for friend recommendation in online games. However, none of existing approaches can effectively incorporate the multi-modal user features (\emph{e.g.}, images and texts) with the structural information in the friendship graph, due to the following limitations: (1) some of them ignore the high-order structural proximity between users, (2) some fail to learn the pairwise relevance between users at modality-specific level, and (3) some cannot capture both the local and global user preferences on different modalities. By addressing these issues, in this paper, we propose an end-to-end model \textsc{FROG} that better models the user preferences on potential friends. Comprehensive experiments on both offline evaluation and online deployment at \kw{Tencent} have demonstrated the superiority of \textsc{FROG} over existing approaches. 

**Abstract (ZH)**: 基于多模态用户特征和友谊图结构信息的端到端好友推荐模型FROG 

---
# BabyVLM: Data-Efficient Pretraining of VLMs Inspired by Infant Learning 

**Title (ZH)**: BabyVLM：受婴儿学习启发的高效数据预训练多模态模型 

**Authors**: Shengao Wang, Arjun Chandra, Aoming Liu, Venkatesh Saligrama, Boqing Gong  

**Link**: [PDF](https://arxiv.org/pdf/2504.09426)  

**Abstract**: Human infants rapidly develop visual reasoning skills from minimal input, suggesting that developmentally inspired pretraining could significantly enhance the efficiency of vision-language models (VLMs). Although recent efforts have leveraged infant-inspired datasets like SAYCam, existing evaluation benchmarks remain misaligned--they are either too simplistic, narrowly scoped, or tailored for large-scale pretrained models. Additionally, training exclusively on infant data overlooks the broader, diverse input from which infants naturally learn. To address these limitations, we propose BabyVLM, a novel framework comprising comprehensive in-domain evaluation benchmarks and a synthetic training dataset created via child-directed transformations of existing datasets. We demonstrate that VLMs trained with our synthetic dataset achieve superior performance on BabyVLM tasks compared to models trained solely on SAYCam or general-purpose data of the SAYCam size. BabyVLM thus provides a robust, developmentally aligned evaluation tool and illustrates how compact models trained on carefully curated data can generalize effectively, opening pathways toward data-efficient vision-language learning paradigms. 

**Abstract (ZH)**: Human婴儿启发的视觉推理技能的快速发展表明，基于发展启发式预训练可以显著提高视觉语言模型的效率。尽管近期努力利用了如SAYCam这样的婴儿启发式数据集，现有的评估基准仍然存在偏差——它们要么过于简单，要么范围狭窄，或者针对大规模预训练模型。此外，仅在婴儿数据上进行训练忽略了婴儿自然学习的更广泛和多样的输入。为了解决这些限制，我们提出了BabyVLM，这是一个包含全面领域内评估基准和通过儿童导向变换生成的合成训练数据集的新框架。我们证明，使用我们合成数据集训练的视觉语言模型在BabyVLM任务上的表现优于仅在SAYCam或同等规模的通用数据上训练的模型。因此，BabyVLM提供了一种稳健且发展上一致的评估工具，并展示了如何通过精心筛选的数据训练的小型模型可以有效泛化，从而为高效的数据驱动视觉语言学习开辟道路。 

---
# ClinicalGPT-R1: Pushing reasoning capability of generalist disease diagnosis with large language model 

**Title (ZH)**: ClinicalGPT-R1: 利用大型语言模型提升通用疾病诊断的推理能力 

**Authors**: Wuyang Lan, Wenzheng Wang, Changwei Ji, Guoxing Yang, Yongbo Zhang, Xiaohong Liu, Song Wu, Guangyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09421)  

**Abstract**: Recent advances in reasoning with large language models (LLMs)has shown remarkable reasoning capabilities in domains such as mathematics and coding, yet their application to clinical diagnosis remains underexplored. Here, we introduce ClinicalGPT-R1, a reasoning enhanced generalist large language model for disease diagnosis. Trained on a dataset of 20,000 real-world clinical records, ClinicalGPT-R1 leverages diverse training strategies to enhance diagnostic reasoning. To benchmark performance, we curated MedBench-Hard, a challenging dataset spanning seven major medical specialties and representative diseases. Experimental results demonstrate that ClinicalGPT-R1 outperforms GPT-4o in Chinese diagnostic tasks and achieves comparable performance to GPT-4 in English settings. This comparative study effectively validates the superior performance of ClinicalGPT-R1 in disease diagnosis tasks. Resources are available at this https URL. 

**Abstract (ZH)**: Recent advances in reasoning with large language models (LLMs) in clinical diagnosis: Introducing ClinicalGPT-R1 

---
# Question Tokens Deserve More Attention: Enhancing Large Language Models without Training through Step-by-Step Reading and Question Attention Recalibration 

**Title (ZH)**: 无需训练提升大型语言模型：通过逐步阅读和问题注意力校准增强疑问词-token应受更多关注 

**Authors**: Feijiang Han, Licheng Guo, Hengtao Cui, Zhiyuan Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2504.09402)  

**Abstract**: Large Language Models (LLMs) often struggle with tasks that require a deep understanding of complex questions, especially when faced with long-range dependencies or multi-step reasoning. This work investigates the limitations of current LLMs in question comprehension and identifies three insights: (1) repeating question tokens improves comprehension by increasing attention to question regions, (2) increased backward dependencies negatively affect performance due to unidirectional attentional constraints, and (3) recalibrating attentional mechanisms to prioritize question-relevant regions improves performance.
Based on these findings, we first propose a family of prompt-based strategies - Step-by-Step Reading (SSR), SSR+, and SSR++ - that guide LLMs to incrementally process question tokens and align their reasoning with the input structure. These methods significantly improve performance, with SSR++ achieving state-of-the-art results on several benchmarks: 96.66% on GSM8K, 94.61% on ASDiv, and 76.28% on AQuA. Second, we introduce a training-free attention recalibration mechanism that dynamically adjusts attention distributions during inference to emphasize question-relevant regions. This method improves the accuracy of LLaMA 3.1-8B on AQuA by 5.17% without changing model parameters or input prompts.
Taken together, our results highlight the importance of structured prompt design and attention optimization in improving LLM comprehension, providing lightweight yet effective tools for improving performance in various NLP tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在处理需要深入理解复杂问题的任务时往往表现不佳，特别是在面对长程依赖或多步推理时。本研究探讨了当前LLMs在问题理解方面的限制，并得出了三点见解：（1）重复问题 token 可提高理解能力，通过增加对问题区域的关注，（2）增加反向依赖关系会因单向注意力约束而负面影响，（3）重新校准注意力机制以优先考虑相关问题区域可提高性能。基于这些发现，我们首先提出了一组基于提示的策略——逐步阅读（SSR）、SSR+ 和 SSR++，这些策略引导 LLM 逐步处理问题 token，并使其推理与输入结构对齐。这些方法显著提高了性能，SSR++ 在多个基准测试上的表现达到最新水平：GSM8K 上的准确率为 96.66%，ASDiv 上为 94.61%，AQuA 上为 76.28%。其次，我们引入了一种无需训练的注意机制校准方法，在推理过程中动态调整注意分布以突出问题相关区域。这项方法在不改变模型参数或输入提示的情况下，将 LLava 3.1-8B 在 AQuA 上的准确率提高了 5.17%。综上所述，我们的结果强调了有序提示设计和注意力优化在提高 LLM 理解能力方面的重要性，提供了轻量级且有效的工具以在各种自然语言处理任务中提高性能。 

---
# Composable NLP Workflows for BERT-based Ranking and QA System 

**Title (ZH)**: 基于BERT的排名与问答系统可组合自然语言处理工作流 

**Authors**: Gaurav Kumar, Murali Mohana Krishna Dandu  

**Link**: [PDF](https://arxiv.org/pdf/2504.09398)  

**Abstract**: There has been a lot of progress towards building NLP models that scale to multiple tasks. However, real-world systems contain multiple components and it is tedious to handle cross-task interaction with varying levels of text granularity. In this work, we built an end-to-end Ranking and Question-Answering (QA) system using Forte, a toolkit that makes composable NLP pipelines. We utilized state-of-the-art deep learning models such as BERT, RoBERTa in our pipeline, evaluated the performance on MS-MARCO and Covid-19 datasets using metrics such as BLUE, MRR, F1 and compared the results of ranking and QA systems with their corresponding benchmark results. The modular nature of our pipeline and low latency of reranker makes it easy to build complex NLP applications easily. 

**Abstract (ZH)**: 面向多个任务的自然语言处理模型研究：基于Forte的端到端排名与问答系统 

---
# Adaptive Insurance Reserving with CVaR-Constrained Reinforcement Learning under Macroeconomic Regimes 

**Title (ZH)**: 宏观经济环境下基于CVaR约束强化学习的适应性保险准备金计提 

**Authors**: Stella C. Dong, James R. Finlay  

**Link**: [PDF](https://arxiv.org/pdf/2504.09396)  

**Abstract**: This paper proposes a reinforcement learning (RL) framework for insurance reserving that integrates tail-risk sensitivity, macroeconomic regime modeling, and regulatory compliance. The reserving problem is formulated as a finite-horizon Markov Decision Process (MDP), in which reserve adjustments are optimized using Proximal Policy Optimization (PPO) subject to Conditional Value-at-Risk (CVaR) constraints. To enhance policy robustness across varying economic conditions, the agent is trained using a regime-aware curriculum that progressively increases volatility exposure.
The reward structure penalizes reserve shortfall, capital inefficiency, and solvency floor violations, with design elements informed by Solvency II and Own Risk and Solvency Assessment (ORSA) frameworks. Empirical evaluations on two industry datasets--Workers' Compensation, and Other Liability--demonstrate that the RL-CVaR agent achieves superior performance relative to classical reserving methods across multiple criteria, including tail-risk control (CVaR$_{0.95}$), capital efficiency, and regulatory violation rate. The framework also accommodates fixed-shock stress testing and regime-stratified analysis, providing a principled and extensible approach to reserving under uncertainty. 

**Abstract (ZH)**: 本文提出了一种整合尾部风险敏感性、宏观经济状态建模和监管合规性的 reinforcement learning (RL) 保险赔付准备金框架。将准备金问题形式化为有限 horizon 马尔可夫决策过程 (MDP)，并通过条件价值-at-风险 (CVaR) 约束使用近端策略优化 (PPO) 优化准备金调整。为了增强政策在不同经济条件下的鲁棒性，使用状态感知的训练课程来逐步增加波动性暴露进行训练。奖励结构惩罚准备金短缺、资本效率低下和偿付能力底线违规，设计元素受到偿付能力 II 和Own Risk and Solvency Assessment (ORSA) 架构的启发。在两个行业数据集——工伤赔偿和其它责任——上的实证评估表明，RL-CVaR 剂量相对于经典准备金方法在多个标准上实现了更高的性能，包括尾部风险控制 (CVaR$_{0.95}$)、资本效率和监管违规率。该框架还支持固定冲击的压力测试和状态分层分析，提供了一种原理明晰且可扩展的不确定性下准备金建模方法。 

---
# REMEMBER: Retrieval-based Explainable Multimodal Evidence-guided Modeling for Brain Evaluation and Reasoning in Zero- and Few-shot Neurodegenerative Diagnosis 

**Title (ZH)**: REMEMBER：基于检索的可解释多模态证据引导建模在零样本和少样本神经退行性疾病诊断中的大脑评估与推理 

**Authors**: Duy-Cat Can, Quang-Huy Tang, Huong Ha, Binh T. Nguyen, Oliver Y. Chén  

**Link**: [PDF](https://arxiv.org/pdf/2504.09354)  

**Abstract**: Timely and accurate diagnosis of neurodegenerative disorders, such as Alzheimer's disease, is central to disease management. Existing deep learning models require large-scale annotated datasets and often function as "black boxes". Additionally, datasets in clinical practice are frequently small or unlabeled, restricting the full potential of deep learning methods. Here, we introduce REMEMBER -- Retrieval-based Explainable Multimodal Evidence-guided Modeling for Brain Evaluation and Reasoning -- a new machine learning framework that facilitates zero- and few-shot Alzheimer's diagnosis using brain MRI scans through a reference-based reasoning process. Specifically, REMEMBER first trains a contrastively aligned vision-text model using expert-annotated reference data and extends pseudo-text modalities that encode abnormality types, diagnosis labels, and composite clinical descriptions. Then, at inference time, REMEMBER retrieves similar, human-validated cases from a curated dataset and integrates their contextual information through a dedicated evidence encoding module and attention-based inference head. Such an evidence-guided design enables REMEMBER to imitate real-world clinical decision-making process by grounding predictions in retrieved imaging and textual context. Specifically, REMEMBER outputs diagnostic predictions alongside an interpretable report, including reference images and explanations aligned with clinical workflows. Experimental results demonstrate that REMEMBER achieves robust zero- and few-shot performance and offers a powerful and explainable framework to neuroimaging-based diagnosis in the real world, especially under limited data. 

**Abstract (ZH)**: 基于检索的可解释多模态证据引导建模框架REMEMBER：用于基于脑MRI扫描的零-shot和少-shot阿尔茨海默病诊断 

---
# Explorer: Robust Collection of Interactable GUI Elements 

**Title (ZH)**: Explorer: 坚韧的交互式GUI元素收集器 

**Authors**: Iason Chaimalas, Arnas Vyšniauskas, Gabriel Brostow  

**Link**: [PDF](https://arxiv.org/pdf/2504.09352)  

**Abstract**: Automation of existing Graphical User Interfaces (GUIs) is important but hard to achieve. Upstream of making the GUI user-accessible or somehow scriptable, even the data-collection to understand the original interface poses significant challenges. For example, large quantities of general UI data seem helpful for training general machine learning (ML) models, but accessibility for each person can hinge on the ML's precision on a specific app. We therefore take the perspective that a given user needs confidence, that the relevant UI elements are being detected correctly throughout one app or digital environment. We mostly assume that the target application is known in advance, so that data collection and ML-training can be personalized for the test-time target domain. The proposed Explorer system focuses on detecting on-screen buttons and text-entry fields, i.e. interactables, where the training process has access to a live version of the application. The live application can run on almost any popular platform except iOS phones, and the collection is especially streamlined for Android phones or for desktop Chrome browsers. Explorer also enables the recording of interactive user sessions, and subsequent mapping of how such sessions overlap and sometimes loop back to similar states. We show how having such a map enables a kind of path planning through the GUI, letting a user issue audio commands to get to their destination. Critically, we are releasing our code for Explorer openly at this https URL. 

**Abstract (ZH)**: 现有的图形用户界面（GUI）的自动化至关重要但难以实现。在使GUI对用户可用或以某种方式使其脚本化之前，甚至收集数据以理解原始界面本身都面临着重大挑战。例如，大量通用UI数据似乎有助于训练通用机器学习（ML）模型，但每个人的数据访问能力取决于ML在特定应用上的精度。因此，我们从一个用户需要对相关UI元素在整个应用或数字环境中被正确检测的程度具有信心的角度出发。我们假设目标应用程序事先已知，以便数据收集和ML训练可以针对测试时的目标领域进行个性化处理。所提出的Explorer系统专注于检测屏幕上的按钮和文本输入字段，即可交互元素，并且训练过程可以访问应用的实时版本。该应用可以在除iOS手机之外的几乎所有主流平台上运行，数据收集特别简化了Android手机或桌面Chrome浏览器的采集过程。Explorer还能够记录交互式用户会话，并在随后映射这些会话如何相互重叠，有时又循环回到类似的状态。我们展示了此类地图如何使用户能够通过GUI进行路径规划，并发出语音命令到达目的地的重要之处。关键的是，我们正在在此公开发布Explorer的代码。 

---
# "It's not a representation of me": Examining Accent Bias and Digital Exclusion in Synthetic AI Voice Services 

**Title (ZH)**: “这不是我的表现”：探究口音偏见与合成AI声音服务中的数字排斥现象 

**Authors**: Shira Michel, Sufi Kaur, Sarah Elizabeth Gillespie, Jeffrey Gleason, Christo Wilson, Avijit Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2504.09346)  

**Abstract**: Recent advances in artificial intelligence (AI) speech generation and voice cloning technologies have produced naturalistic speech and accurate voice replication, yet their influence on sociotechnical systems across diverse accents and linguistic traits is not fully understood. This study evaluates two synthetic AI voice services (Speechify and ElevenLabs) through a mixed methods approach using surveys and interviews to assess technical performance and uncover how users' lived experiences influence their perceptions of accent variations in these speech technologies. Our findings reveal technical performance disparities across five regional, English-language accents and demonstrate how current speech generation technologies may inadvertently reinforce linguistic privilege and accent-based discrimination, potentially creating new forms of digital exclusion. Overall, our study highlights the need for inclusive design and regulation by providing actionable insights for developers, policymakers, and organizations to ensure equitable and socially responsible AI speech technologies. 

**Abstract (ZH)**: Recent Advances in Artificial Intelligence Speech Generation and Voice Cloning Technologies and Their Influence on Sociotechnical Systems Across Diverse Accents and Linguistic Traits: A Mixed Methods Study Evaluating Technical Performance and Perceptions of Accent Variations in Synthetic AI Voice Services 

---
# MoE-Lens: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints 

**Title (ZH)**: MoE-Lens: 在资源约束条件下向着高 throughput MoE 大型语言模型服务的硬件极限发展 

**Authors**: Yichao Yuan, Lin Ma, Nishil Talati  

**Link**: [PDF](https://arxiv.org/pdf/2504.09345)  

**Abstract**: Mixture of Experts (MoE) LLMs, characterized by their sparse activation patterns, offer a promising approach to scaling language models while avoiding proportionally increasing the inference cost. However, their large parameter sizes present deployment challenges in resource-constrained environments with limited GPU memory capacity, as GPU memory is often insufficient to accommodate the full set of model weights. Consequently, typical deployments rely on CPU-GPU hybrid execution: the GPU handles compute-intensive GEMM operations, while the CPU processes the relatively lightweight attention mechanism. This setup introduces a key challenge: how to effectively optimize resource utilization across CPU and GPU? Prior work has designed system optimizations based on performance models with limited scope. Specifically, such models do not capture the complex interactions between hardware properties and system execution mechanisms. Therefore, previous approaches neither identify nor achieve the hardware limit.
This paper presents MoE-Lens, a high-throughput MoE LLM inference system designed through holistic performance modeling for resource-constrained environments. Our performance model thoroughly analyzes various fundamental system components, including CPU memory capacity, GPU compute power, and workload characteristics, to understand the theoretical performance upper bound of MoE inference. Furthermore, it captures the system execution mechanisms to identify the key hardware bottlenecks and accurately predict the achievable throughput. Informed by our performance model, MoE-Lens introduces an inference system approaching hardware limits. Evaluated on diverse MoE models and datasets, MoE-Lens outperforms the state-of-the-art solution by 4.6x on average (up to 25.5x), with our theoretical model predicting performance with an average 94% accuracy. 

**Abstract (ZH)**: Mixture of Experts (MoE) 大型语言模型（LLM），以其稀疏激活模式为特征，提供了在不按比例增加推理成本的情况下扩展语言模型的有前景的方法。然而，其庞大的参数大小在资源受限环境中带来了部署挑战，尤其是在有限的GPU内存容量环境下，GPU内存往往无法容纳模型的全部权重集。因此，典型的部署依赖于CPU和GPU的混合执行：GPU处理密集型的GEMM操作，而CPU处理相对轻量的注意力机制。这种设置引入了一个关键挑战：如何有效地优化CPU和GPU之间的资源利用？以往的研究基于性能模型，但这些模型的范围有限，未能捕捉到硬件特性和系统执行机制之间的复杂交互。因此，先前的方法既未识别也未达到硬件限制。

本文提出了MoE-Lens，这是一种针对资源受限环境设计的高吞吐量MoE LLM推理系统，通过全面的性能模型进行设计。我们的性能模型详细分析了各种基本系统组件，包括CPU内存容量、GPU计算能力以及工作负载特性，以理解MoE推理的理论性能上限。此外，它捕捉了系统执行机制，以识别关键硬件瓶颈并准确预测可实现的吞吐量。基于我们的性能模型，MoE-Lens引入了一种接近硬件限制的推理系统。在多种MoE模型和数据集上进行评估，MoE-Lens在性能上平均比最先进的解决方案提高了4.6倍（最高达25.5倍），并且我们的理论模型预测性能的平均准确率为94%。 

---
# Confirmation Bias in Generative AI Chatbots: Mechanisms, Risks, Mitigation Strategies, and Future Research Directions 

**Title (ZH)**: 生成人工智能聊天机器人中的确认偏误：机理、风险、缓解策略与未来研究方向 

**Authors**: Yiran Du  

**Link**: [PDF](https://arxiv.org/pdf/2504.09343)  

**Abstract**: This article explores the phenomenon of confirmation bias in generative AI chatbots, a relatively underexamined aspect of AI-human interaction. Drawing on cognitive psychology and computational linguistics, it examines how confirmation bias, commonly understood as the tendency to seek information that aligns with existing beliefs, can be replicated and amplified by the design and functioning of large language models. The article analyzes the mechanisms by which confirmation bias may manifest in chatbot interactions, assesses the ethical and practical risks associated with such bias, and proposes a range of mitigation strategies. These include technical interventions, interface redesign, and policy measures aimed at promoting balanced AI-generated discourse. The article concludes by outlining future research directions, emphasizing the need for interdisciplinary collaboration and empirical evaluation to better understand and address confirmation bias in generative AI systems. 

**Abstract (ZH)**: 本文探索生成式AI聊天机器人中的确认偏见现象，这是人工智能-人类交互的一个相对较未研究的方面。本文结合认知心理学和计算语言学，探讨确认偏见——通常指寻求与现有信念一致的信息的倾向——如何在大型语言模型的设计和运行中被复制和放大。本文分析了确认偏见在聊天机器人交互中可能出现的机制，评估了这种偏见相关的伦理和实践风险，并提出了包括技术干预、界面重新设计和促进平衡AI生成话语的政策措施等一系列缓解策略。文章总结了未来研究的方向，强调了跨学科合作和实证评估的重要性，以便更好地理解并应对生成式AI系统中的确认偏见。 

---
# Lumos: Efficient Performance Modeling and Estimation for Large-scale LLM Training 

**Title (ZH)**: Lumos: 大规模LLM训练的高效性能建模与估计 

**Authors**: Mingyu Liang, Hiwot Tadese Kassa, Wenyin Fu, Brian Coutinho, Louis Feng, Christina Delimitrou  

**Link**: [PDF](https://arxiv.org/pdf/2504.09307)  

**Abstract**: Training LLMs in distributed environments presents significant challenges due to the complexity of model execution, deployment systems, and the vast space of configurable strategies. Although various optimization techniques exist, achieving high efficiency in practice remains difficult. Accurate performance models that effectively characterize and predict a model's behavior are essential for guiding optimization efforts and system-level studies. We propose Lumos, a trace-driven performance modeling and estimation toolkit for large-scale LLM training, designed to accurately capture and predict the execution behaviors of modern LLMs. We evaluate Lumos on a production ML cluster with up to 512 NVIDIA H100 GPUs using various GPT-3 variants, demonstrating that it can replay execution time with an average error of just 3.3%, along with other runtime details, across different models and configurations. Additionally, we validate its ability to estimate performance for new setups from existing traces, facilitating efficient exploration of model and deployment configurations. 

**Abstract (ZH)**: 分布式环境中训练大规模语言模型面临着显著的挑战，这些挑战源于模型执行的复杂性、部署系统的复杂性和可配置策略的广泛空间。尽管存在各种优化技术，但在实践中达到高效率仍然困难重重。准确的性能模型对于指导优化努力和系统级研究至关重要。我们提出了Lumos，一个基于跟踪的性能建模与估计工具包，旨在准确捕捉和预测现代大规模语言模型的执行行为。我们在配备多达512个NVIDIA H100 GPU的生产ML集群中使用不同版本的GPT-3进行了评估，结果显示Lumos能够以平均3.3%的误差准确重放执行时间，并提供其他运行时详情，适用于不同模型和配置。此外，我们验证了它能够利用现有跟踪估计新设置的性能，从而促进对模型和部署配置的有效探索。 

---
# Semantic Commit: Helping Users Update Intent Specifications for AI Memory at Scale 

**Title (ZH)**: 语义提交：帮助用户大规模更新AI记忆的意图规范 

**Authors**: Priyan Vaithilingam, Munyeong Kim, Frida-Cecilia Acosta-Parenteau, Daniel Lee, Amine Mhedhbi, Elena L. Glassman, Ian Arawjo  

**Link**: [PDF](https://arxiv.org/pdf/2504.09283)  

**Abstract**: How do we update AI memory of user intent as intent changes? We consider how an AI interface may assist the integration of new information into a repository of natural language data. Inspired by software engineering concepts like impact analysis, we develop methods and a UI for managing semantic changes with non-local effects, which we call "semantic conflict resolution." The user commits new intent to a project -- makes a "semantic commit" -- and the AI helps the user detect and resolve semantic conflicts within a store of existing information representing their intent (an "intent specification"). We develop an interface, SemanticCommit, to better understand how users resolve conflicts when updating intent specifications such as Cursor Rules and game design documents. A knowledge graph-based RAG pipeline drives conflict detection, while LLMs assist in suggesting resolutions. We evaluate our technique on an initial benchmark. Then, we report a 12 user within-subjects study of SemanticCommit for two task domains -- game design documents, and AI agent memory in the style of ChatGPT memories -- where users integrated new information into an existing list. Half of our participants adopted a workflow of impact analysis, where they would first flag conflicts without AI revisions then resolve conflicts locally, despite having access to a global revision feature. We argue that AI agent interfaces, such as software IDEs like Cursor and Windsurf, should provide affordances for impact analysis and help users validate AI retrieval independently from generation. Our work speaks to how AI agent designers should think about updating memory as a process that involves human feedback and decision-making. 

**Abstract (ZH)**: 如何更新用户意图的AI记忆？我们考虑AI界面如何协助将新信息整合到自然语言数据仓库中。受软件工程概念如影响分析的启发，我们开发了管理具有非局部效应的语义变化的方法和界面，称之为“语义冲突解决”。用户将新的意图提交给项目——进行“语义提交”——并由AI帮助用户在现有信息库中检测和解决与他们意图相关的语义冲突（“意图规范”）。我们开发了界面SemanticCommit，以更好地理解用户在更新意图规范（如光标规则和游戏设计文档）时如何解决冲突。基于知识图谱的RAG管道驱动冲突检测，而LLM帮助提出解决方案。我们对初始基准进行了技术评估，然后报告了12名用户的内组研究，研究对象是两个任务领域——游戏设计文档和类似ChatGPT记忆的AI代理记忆，其中用户将新信息整合到现有列表中。我们的参与者中有一半采用了影响分析的工作流程，即首先标记冲突而不进行AI修订，然后在有全局修订功能的情况下局部解决冲突。我们认为，AI代理界面，如同Cursor和Windsurf这样的软件IDE，应提供影响分析的功能，帮助用户独立于生成验证AI检索结果。我们的工作讨论了AI代理设计师在更新记忆时应如何考虑包含人类反馈和决策过程的过程。 

---
# SynthTRIPs: A Knowledge-Grounded Framework for Benchmark Query Generation for Personalized Tourism Recommenders 

**Title (ZH)**: SynthTRIPs：面向个性化旅游推荐的基于知识的基准查询生成框架 

**Authors**: Ashmi Banerjee, Adithi Satish, Fitri Nur Aisyah, Wolfgang Wörndl, Yashar Deldjoo  

**Link**: [PDF](https://arxiv.org/pdf/2504.09277)  

**Abstract**: Tourism Recommender Systems (TRS) are crucial in personalizing travel experiences by tailoring recommendations to users' preferences, constraints, and contextual factors. However, publicly available travel datasets often lack sufficient breadth and depth, limiting their ability to support advanced personalization strategies -- particularly for sustainable travel and off-peak tourism. In this work, we explore using Large Language Models (LLMs) to generate synthetic travel queries that emulate diverse user personas and incorporate structured filters such as budget constraints and sustainability preferences.
This paper introduces a novel SynthTRIPs framework for generating synthetic travel queries using LLMs grounded in a curated knowledge base (KB). Our approach combines persona-based preferences (e.g., budget, travel style) with explicit sustainability filters (e.g., walkability, air quality) to produce realistic and diverse queries. We mitigate hallucination and ensure factual correctness by grounding the LLM responses in the KB. We formalize the query generation process and introduce evaluation metrics for assessing realism and alignment. Both human expert evaluations and automatic LLM-based assessments demonstrate the effectiveness of our synthetic dataset in capturing complex personalization aspects underrepresented in existing datasets. While our framework was developed and tested for personalized city trip recommendations, the methodology applies to other recommender system domains.
Code and dataset are made public at this https URL 

**Abstract (ZH)**: 基于大型语言模型生成合成旅游查询的SynthTRIPs框架 

---
# Linguistic Comparison of AI- and Human-Written Responses to Online Mental Health Queries 

**Title (ZH)**: AI写作与人类写作对在线心理健康查询的 Linguistic 对比研究 

**Authors**: Koustuv Saha, Yoshee Jain, Munmun De Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2504.09271)  

**Abstract**: The ubiquity and widespread use of digital and online technologies have transformed mental health support, with online mental health communities (OMHCs) providing safe spaces for peer support. More recently, generative AI and large language models (LLMs) have introduced new possibilities for scalable, around-the-clock mental health assistance that could potentially augment and supplement the capabilities of OMHCs. Although genAI shows promise in delivering immediate and personalized responses, their effectiveness in replicating the nuanced, experience-based support of human peers remains an open question. In this study, we harnessed 24,114 posts and 138,758 online community (OC) responses from 55 OMHCs on Reddit. We prompted several state-of-the-art LLMs (GPT-4-Turbo, Llama-3, and Mistral-7B) with these posts, and compared their (AI) responses to human-written (OC) responses based on a variety of linguistic measures across psycholinguistics and lexico-semantics. Our findings revealed that AI responses are more verbose, readable, and analytically structured, but lack linguistic diversity and personal narratives inherent in human-human interactions. Through a qualitative examination, we found validation as well as complementary insights into the nature of AI responses, such as its neutrality of stance and the absence of seeking back-and-forth clarifications. We discuss the ethical and practical implications of integrating generative AI into OMHCs, advocating for frameworks that balance AI's scalability and timeliness with the irreplaceable authenticity, social interactiveness, and expertise of human connections that form the ethos of online support communities. 

**Abstract (ZH)**: 数字和在线技术的普遍应用已 transformation 心理健康支持，线上心理健康社区 (OMHCs) 提供了同伴支持的安全空间。近年来，生成式 AI 和大规模语言模型 (LLMs) 引入了规模化、全天候心理健康援助的新可能性，这可能增强和补充 OMHCs 的能力。尽管生成式 AI 显示出提供即时个性化响应的潜力，但在复制人类同伴基于经验的细微支持方面其有效性的疑问依然存在。本研究利用来自 Reddit 上 55 个 OMHCs 的 24,114 个帖子和 138,758 个在线社区 (OC) 回应，激发了 GPT-4-Turbo、Llama-3 和 Mistral-7B 等最先进的 LLMs，并基于跨心理语言学和词汇语义学的各种语言指标将 AI 回应与人类撰写的 (OC) 回应进行了对比。研究发现，AI 回应更为冗长、可读且结构化，但缺乏人类互动中固有的语言多样性和个人叙述。通过定性分析，我们发现了 AI 回应的证实和补充见解，如立场的中立性和未寻求持续澄清。我们探讨了将生成式 AI 集成到 OMHCs 中的伦理和实践影响，倡导平衡 AI 的规模性和及时性与人类连接不可替代的真诚、社交互动性和专业知识，这些构成了在线支持社区精神的核心价值观。 

---
# MiMIC: Multi-Modal Indian Earnings Calls Dataset to Predict Stock Prices 

**Title (ZH)**: MiMIC：多模态印度 earnings 电话会议数据集，用于预测股票价格 

**Authors**: Sohom Ghosh, Arnab Maji, Sudip Kumar Naskar  

**Link**: [PDF](https://arxiv.org/pdf/2504.09257)  

**Abstract**: Predicting stock market prices following corporate earnings calls remains a significant challenge for investors and researchers alike, requiring innovative approaches that can process diverse information sources. This study investigates the impact of corporate earnings calls on stock prices by introducing a multi-modal predictive model. We leverage textual data from earnings call transcripts, along with images and tables from accompanying presentations, to forecast stock price movements on the trading day immediately following these calls. To facilitate this research, we developed the MiMIC (Multi-Modal Indian Earnings Calls) dataset, encompassing companies representing the Nifty 50, Nifty MidCap 50, and Nifty Small 50 indices. The dataset includes earnings call transcripts, presentations, fundamentals, technical indicators, and subsequent stock prices. We present a multimodal analytical framework that integrates quantitative variables with predictive signals derived from textual and visual modalities, thereby enabling a holistic approach to feature representation and analysis. This multi-modal approach demonstrates the potential for integrating diverse information sources to enhance financial forecasting accuracy. To promote further research in computational economics, we have made the MiMIC dataset publicly available under the CC-NC-SA-4.0 licence. Our work contributes to the growing body of literature on market reactions to corporate communications and highlights the efficacy of multi-modal machine learning techniques in financial analysis. 

**Abstract (ZH)**: 预测 CORPORATE EARNINGS CALLS 之后的股票市场价格仍然是投资者和研究人员面临的重大挑战，需要创新的方法来处理多种信息来源。本文通过引入多模态预测模型，研究公司业绩电话会议对股票价格的影响。我们利用收益电话会议纪要的文本数据，以及附带演示文稿中的图像和表格来预测电话会议后的首个交易日的股票价格变动。为了开展这项研究，我们开发了MiMIC（多模态印度收益电话会议）数据集，涵盖代表着Nifty 50、Nifty MidCap 50和Nifty Small 50指数的公司。该数据集包括收益电话会议纪要、演示文稿、基本财务数据、技术指标和后续股票价格。我们提出了一种多模态分析框架，将定量变量与来自文本和视觉模态的预测信号结合起来，从而实现对特征表示和分析的全面方法。多模态方法展示了整合多种信息来源以提高财务预测准确性的潜力。为了促进计算经济学领域的进一步研究，我们已根据CC-NC-SA-4.0许可证将MiMIC数据集公开。我们的研究成果增进了对企业沟通影响市场反应的文献的贡献，并突显了多模态机器学习技术在金融分析中的有效性。 

---
# Development of a PPO-Reinforcement Learned Walking Tripedal Soft-Legged Robot using SOFA 

**Title (ZH)**: 基于SOFA的PPO强化学习三足软腿机器人开发 

**Authors**: Yomna Mokhtar, Tarek Shohdy, Abdallah A. Hassan, Mostafa Eshra, Omar Elmenawy, Osama Khalil, Haitham El-Hussieny  

**Link**: [PDF](https://arxiv.org/pdf/2504.09242)  

**Abstract**: Rigid robots were extensively researched, whereas soft robotics remains an underexplored field. Utilizing soft-legged robots in performing tasks as a replacement for human beings is an important stride to take, especially under harsh and hazardous conditions over rough terrain environments. For the demand to teach any robot how to behave in different scenarios, a real-time physical and visual simulation is essential. When it comes to soft robots specifically, a simulation framework is still an arduous problem that needs to be disclosed. Using the simulation open framework architecture (SOFA) is an advantageous step. However, neither SOFA's manual nor prior public SOFA projects show its maximum capabilities the users can reach. So, we resolved this by establishing customized settings and handling the framework components appropriately. Settling on perfect, fine-tuned SOFA parameters has stimulated our motivation towards implementing the state-of-the-art (SOTA) reinforcement learning (RL) method of proximal policy optimization (PPO). The final representation is a well-defined, ready-to-deploy walking, tripedal, soft-legged robot based on PPO-RL in a SOFA environment. Robot navigation performance is a key metric to be considered for measuring the success resolution. Although in the simulated soft robots case, an 82\% success rate in reaching a single goal is a groundbreaking output, we pushed the boundaries to further steps by evaluating the progress under assigning a sequence of goals. While trailing the platform steps, outperforming discovery has been observed with an accumulative squared error deviation of 19 mm. The full code is publicly available at \href{this https URL}{this http URL\textunderscore$SOFA$\textunderscore$Soft$\textunderscore$Legged$\textunderscore$ this http URL} 

**Abstract (ZH)**: 刚性机器人得到了广泛研究，而软体机器人领域仍是一个未充分开发的领域。利用具有软腿的机器人执行任务，替代人类在恶劣和危险的崎岖地形环境中工作，是重要的一步。为了满足任何机器人在不同场景中行为的教学需求，实时的物理和视觉仿真至关重要。在具体到软体机器人时，仿真框架仍然是一个亟待解决的难题。使用仿真开放框架架构（SOFA）是一种有利的步骤。然而，SOFA的手册及其先前的公共SOFA项目并未充分展示用户能实现的最大能力。因此，我们通过建立定制设置并适当处理框架组件来解决这一问题。确立完美的、细调的SOFA参数激发了我们采用最先进的（SOTA）强化学习（RL）方法——近端策略优化（PPO）的动机。最终的表现是一个基于PPO-RL的、准备好部署的行走、三足软腿机器人，运行在SOFA环境中。机器人的导航性能是衡量成功的关键指标。尽管在仿真软体机器人的情况下，达到单个目标的成功率达到了82%，我们通过评估分配一系列目标的任务，进一步推动了边界。在跟随平台的步态时，观察到累积平方误差偏差为19毫米的优越表现。完整的代码已公开，可以通过以下链接访问：\href{this https URL}{this http URL\textunderscore$SOFA$\textunderscore$Soft$\textunderscore$Legged$\textunderscore$ this http URL}。 

---
# AMNet: An Acoustic Model Network for Enhanced Mandarin Speech Synthesis 

**Title (ZH)**: AMNet：一种增强型 Mandarin 语音合成声学模型网络 

**Authors**: Yubing Cao, Yinfeng Yu, Yongming Li, Liejun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09225)  

**Abstract**: This paper presents AMNet, an Acoustic Model Network designed to improve the performance of Mandarin speech synthesis by incorporating phrase structure annotation and local convolution modules. AMNet builds upon the FastSpeech 2 architecture while addressing the challenge of local context modeling, which is crucial for capturing intricate speech features such as pauses, stress, and intonation. By embedding a phrase structure parser into the model and introducing a local convolution module, AMNet enhances the model's sensitivity to local information. Additionally, AMNet decouples tonal characteristics from phonemes, providing explicit guidance for tone modeling, which improves tone accuracy and pronunciation. Experimental results demonstrate that AMNet outperforms baseline models in subjective and objective evaluations. The proposed model achieves superior Mean Opinion Scores (MOS), lower Mel Cepstral Distortion (MCD), and improved fundamental frequency fitting $F0 (R^2)$, confirming its ability to generate high-quality, natural, and expressive Mandarin speech. 

**Abstract (ZH)**: AMNet：一种通过引入短时卷积模块和短语结构注解以提高 Mandarin 语音合成性能的声学模型网络 

---
# DL-QAT: Weight-Decomposed Low-Rank Quantization-Aware Training for Large Language Models 

**Title (ZH)**: DL-QAT: 重量分解低秩量化感知训练for大规模语言模型 

**Authors**: Wenjin Ke, Zhe Li, Dong Li, Lu Tian, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2504.09223)  

**Abstract**: Improving the efficiency of inference in Large Language Models (LLMs) is a critical area of research. Post-training Quantization (PTQ) is a popular technique, but it often faces challenges at low-bit levels, particularly in downstream tasks. Quantization-aware Training (QAT) can alleviate this problem, but it requires significantly more computational resources. To tackle this, we introduced Weight-Decomposed Low-Rank Quantization-Aware Training (DL-QAT), which merges the advantages of QAT while training only less than 1% of the total parameters. Specifically, we introduce a group-specific quantization magnitude to adjust the overall scale of each quantization group. Within each quantization group, we use LoRA matrices to update the weight size and direction in the quantization space. We validated the effectiveness of our method on the LLaMA and LLaMA2 model families. The results show significant improvements over our baseline method across different quantization granularities. For instance, for LLaMA-7B, our approach outperforms the previous state-of-the-art method by 4.2% in MMLU on 3-bit LLaMA-7B model. Additionally, our quantization results on pre-trained models also surpass previous QAT methods, demonstrating the superior performance and efficiency of our approach. 

**Abstract (ZH)**: 提高大型语言模型推理效率的研究：一种基于权重分解的低秩量化感知训练方法 

---
# FairACE: Achieving Degree Fairness in Graph Neural Networks via Contrastive and Adversarial Group-Balanced Training 

**Title (ZH)**: FairACE：通过对比和对抗组平衡训练在图神经网络中实现度公平性 

**Authors**: Jiaxin Liu, Xiaoqian Jiang, Cangqi Zhou, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09210)  

**Abstract**: Fairness has been a significant challenge in graph neural networks (GNNs) since degree biases often result in un-equal prediction performance among nodes with varying degrees. Existing GNN models focus on prediction accuracy, frequently overlooking fairness across different degree groups. To addressthis issue, we propose a novel GNN framework, namely Fairness- Aware Asymmetric Contrastive Ensemble (FairACE), which inte-grates asymmetric contrastive learning with adversarial training to improve degree fairness. FairACE captures one-hop local neighborhood information and two-hop monophily similarity to create fairer node representations and employs a degree fairness regulator to balance performance between high-degree and low-degree nodes. During model training, a novel group-balanced fairness loss is proposed to minimize classification disparities across degree groups. In addition, we also propose a novel fairness metric, the Accuracy Distribution Gap (ADG), which can quantitatively assess and ensure equitable performance across different degree-based node groups. Experimental results on both synthetic and real-world datasets demonstrate that FairACE significantly improves degree fairness metrics while maintaining competitive accuracy in comparison to the state-of-the-art GNN models. 

**Abstract (ZH)**: 公平意识异构对比增强图神经网络框架 (Fairness-Aware Asymmetric Contrastive Ensemble, FairACE) 

---
# AerOSeg: Harnessing SAM for Open-Vocabulary Segmentation in Remote Sensing Images 

**Title (ZH)**: AerOSeg：利用SAM进行遥感图像开放词汇分割 

**Authors**: Saikat Dutta, Akhil Vasim, Siddhant Gole, Hamid Rezatofighi, Biplab Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2504.09203)  

**Abstract**: Image segmentation beyond predefined categories is a key challenge in remote sensing, where novel and unseen classes often emerge during inference. Open-vocabulary image Segmentation addresses these generalization issues in traditional supervised segmentation models while reducing reliance on extensive per-pixel annotations, which are both expensive and labor-intensive to obtain. Most Open-Vocabulary Segmentation (OVS) methods are designed for natural images but struggle with remote sensing data due to scale variations, orientation changes, and complex scene compositions. This necessitates the development of OVS approaches specifically tailored for remote sensing. In this context, we propose AerOSeg, a novel OVS approach for remote sensing data. First, we compute robust image-text correlation features using multiple rotated versions of the input image and domain-specific prompts. These features are then refined through spatial and class refinement blocks. Inspired by the success of the Segment Anything Model (SAM) in diverse domains, we leverage SAM features to guide the spatial refinement of correlation features. Additionally, we introduce a semantic back-projection module and loss to ensure the seamless propagation of SAM's semantic information throughout the segmentation pipeline. Finally, we enhance the refined correlation features using a multi-scale attention-aware decoder to produce the final segmentation map. We validate our SAM-guided Open-Vocabulary Remote Sensing Segmentation model on three benchmark remote sensing datasets: iSAID, DLRSD, and OpenEarthMap. Our model outperforms state-of-the-art open-vocabulary segmentation methods, achieving an average improvement of 2.54 h-mIoU. 

**Abstract (ZH)**: 超越预定义类别的图像分割是遥感领域的一个关键挑战，其中在推理过程中往往会涌现出全新的非预见类。开放 Vocabulary 图像分割在传统监督分割模型中解决了这些泛化问题，并减少了对昂贵且劳动密集型的逐像素标注的依赖。大多数开放 Vocabulary 分割 (OVS) 方法适用于自然图像，但在处理由于缩放变化、方向变化和复杂场景组成的遥感数据时却表现不佳。这需要开发专门针对遥感数据的 OVS 方法。在此背景下，我们提出了一种名为 AerOSeg 的新颖 OVS 方法，用于遥感数据。首先，我们使用输入图像的多个旋转版本和领域特定提示计算稳健的图像-文本相关特征。这些特征随后通过空间和类细化模块进行优化。受到 Segment Anything 模型 (SAM) 在多种领域取得成功的影响，我们利用 SAM 特征来引导相关特征的空间细化。此外，我们引入了一个语义反投影模块及其损失，确保 SAM 语义信息在整个分割管道中的无缝传播。最后，我们利用多尺度注意力感知解码器增强细化后的相关特征，生成最终的分割图。我们在三个基准遥感数据集中验证了 SAM 引导下的开放 Vocabulary 遥感分割模型：iSAID、DLRSD 和 OpenEarthMap。我们的模型在所有基准数据集上均优于最先进的开放 Vocabulary 分割方法，平均 h-mIoU 提高了 2.54。 

---
# ReferGPT: Towards Zero-Shot Referring Multi-Object Tracking 

**Title (ZH)**: ReferGPT: 向量ゼロ-shot指称多对象跟踪 

**Authors**: Tzoulio Chamiti, Leandro Di Bella, Adrian Munteanu, Nikos Deligiannis  

**Link**: [PDF](https://arxiv.org/pdf/2504.09195)  

**Abstract**: Tracking multiple objects based on textual queries is a challenging task that requires linking language understanding with object association across frames. Previous works typically train the whole process end-to-end or integrate an additional referring text module into a multi-object tracker, but they both require supervised training and potentially struggle with generalization to open-set queries. In this work, we introduce ReferGPT, a novel zero-shot referring multi-object tracking framework. We provide a multi-modal large language model (MLLM) with spatial knowledge enabling it to generate 3D-aware captions. This enhances its descriptive capabilities and supports a more flexible referring vocabulary without training. We also propose a robust query-matching strategy, leveraging CLIP-based semantic encoding and fuzzy matching to associate MLLM generated captions with user queries. Extensive experiments on Refer-KITTI, Refer-KITTIv2 and Refer-KITTI+ demonstrate that ReferGPT achieves competitive performance against trained methods, showcasing its robustness and zero-shot capabilities in autonomous driving. The codes are available on this https URL 

**Abstract (ZH)**: 基于文本查询的多对象跟踪是一个具有挑战性的任务，要求实现语言理解与跨帧对象关联的链接。已有工作通常以端到端的方式训练整个过程，或者在多对象跟踪器中集成一个额外的引用文本模块，但两者都需要监督训练，并且可能难以泛化到开放集查询。在本文中，我们引入了ReferGPT，这是一种新颖的零样本引用多对象跟踪框架。我们提供了一个具有空间知识的多模态大语言模型（MLLM），使其能够生成3D感知的描述。这增强了其描述能力，并支持了更灵活的引用词汇表，而无需进行训练。同时，我们提出了一个稳健的查询匹配策略，利用CLIP基底义编码和模糊匹配，将MLLM生成的描述与用户查询关联起来。在Refer-KITTI、Refer-KITTIv2和Refer-KITTI+上的广泛实验表明，ReferGPT在性能上与训练方法相当，展示了其在自动驾驶中的稳健性和零样本能力。代码已在此处提供：this https URL。 

---
# Repetitive Contrastive Learning Enhances Mamba's Selectivity in Time Series Prediction 

**Title (ZH)**: 重复对比学习增强Mamba在时间序列预测中的选择性 

**Authors**: Wenbo Yan, Hanzhong Cao, Ying Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.09185)  

**Abstract**: Long sequence prediction is a key challenge in time series forecasting. While Mamba-based models have shown strong performance due to their sequence selection capabilities, they still struggle with insufficient focus on critical time steps and incomplete noise suppression, caused by limited selective abilities. To address this, we introduce Repetitive Contrastive Learning (RCL), a token-level contrastive pretraining framework aimed at enhancing Mamba's selective capabilities. RCL pretrains a single Mamba block to strengthen its selective abilities and then transfers these pretrained parameters to initialize Mamba blocks in various backbone models, improving their temporal prediction performance. RCL uses sequence augmentation with Gaussian noise and applies inter-sequence and intra-sequence contrastive learning to help the Mamba module prioritize information-rich time steps while ignoring noisy ones. Extensive experiments show that RCL consistently boosts the performance of backbone models, surpassing existing methods and achieving state-of-the-art results. Additionally, we propose two metrics to quantify Mamba's selective capabilities, providing theoretical, qualitative, and quantitative evidence for the improvements brought by RCL. 

**Abstract (ZH)**: 长期序列预测是时间序列 forecasting 中的关键挑战。虽然基于 Mamba 的模型由于其序列选择能力表现出强大的性能，但它们仍难以集中关注关键的时间步长并完全抑制噪声，这主要是由于选择能力的局限性所致。为解决这一问题，我们引入了重复对比学习（RCL），这是一种针对增强 Mamba 的选择能力而设计的 token 级对比预训练框架。RCL 预训练单个 Mamba 模块以增强其选择能力，然后将预训练参数转移以初始化各种骨干模型中的 Mamba 模块，从而提高其时间预测性能。RCL 使用高斯噪声进行序列增强，并应用跨序列和序列内对比学习，以帮助 Mamba 模块优先处理信息丰富的时间步长并忽略噪声时间步长。广泛实验表明，RCL 一致性地提升了骨干模型的性能，超越了现有方法并达到了最先进的成果。此外，我们提出了两种度量标准来量化 Mamba 的选择能力，提供了 RCL 所带来的改进的理论、定性和定量证据。 

---
# Parameterized Synthetic Text Generation with SimpleStories 

**Title (ZH)**: 参数化合成文本生成：SimpleStories方法 

**Authors**: Lennart Finke, Thomas Dooms, Mat Allen, Juan Diego Rodriguez, Noa Nabeshima, Dan Braun  

**Link**: [PDF](https://arxiv.org/pdf/2504.09184)  

**Abstract**: We present SimpleStories, a large synthetic story dataset in simple language, consisting of 2 million stories each in English and Japanese. Our method employs parametrization of prompts with features at multiple levels of abstraction, allowing for systematic control over story characteristics to ensure broad syntactic and semantic diversity. Building on and addressing limitations in the TinyStories dataset, our approach demonstrates that simplicity and variety can be achieved simultaneously in synthetic text generation at scale. 

**Abstract (ZH)**: 我们提出SimpleStories，一个使用简单语言构成的大型合成故事数据集，包含200万个英文和日文故事。我们的方法通过多层抽象特征参数化提示，允许对故事特征进行系统控制，以确保广泛的句法和语义多样性。基于并解决了TinyStories数据集的局限性，我们的方法表明，在大规模合成文本生成中，简洁性和多样性可以同时实现。 

---
# A Confounding Factors-Inhibition Adversarial Learning Framework for Multi-site fMRI Mental Disorder Identification 

**Title (ZH)**: 多中心fMRI精神障碍识别的共变量抑制对抗学习框架 

**Authors**: Xin Wen, Shijie Guo, Wenbo Ning, Rui Cao, Yan Niu, Bin Wan, Peng Wei, Xiaobo Liu, Jie Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09179)  

**Abstract**: In open data sets of functional magnetic resonance imaging (fMRI), the heterogeneity of the data is typically attributed to a combination of factors, including differences in scanning procedures, the presence of confounding effects, and population diversities between multiple sites. These factors contribute to the diminished effectiveness of representation learning, which in turn affects the overall efficacy of subsequent classification procedures. To address these limitations, we propose a novel multi-site adversarial learning network (MSalNET) for fMRI-based mental disorder detection. Firstly, a representation learning module is introduced with a node information assembly (NIA) mechanism to better extract features from functional connectivity (FC). This mechanism aggregates edge information from both horizontal and vertical directions, effectively assembling node information. Secondly, to generalize the feature across sites, we proposed a site-level feature extraction module that can learn from individual FC data, which circumvents additional prior information. Lastly, an adversarial learning network is proposed as a means of balancing the trade-off between individual classification and site regression tasks, with the introduction of a novel loss function. The proposed method was evaluated on two multi-site fMRI datasets, i.e., Autism Brain Imaging Data Exchange (ABIDE) and ADHD-200. The results indicate that the proposed method achieves a better performance than other related algorithms with the accuracy of 75.56 and 68.92 in ABIDE and ADHD-200 datasets, respectively. Furthermore, the result of the site regression indicates that the proposed method reduces site variability from a data-driven perspective. The most discriminative brain regions revealed by NIA are consistent with statistical findings, uncovering the "black box" of deep learning to a certain extent. 

**Abstract (ZH)**: 一种用于功能性磁共振成像基于精神疾病检测的新型多站点对抗学习网络（MSalNET） 

---
# Can postgraduate translation students identify machine-generated text? 

**Title (ZH)**: 研究生翻译学生能否识别机器生成的文本？ 

**Authors**: Michael Farrell  

**Link**: [PDF](https://arxiv.org/pdf/2504.09164)  

**Abstract**: Given the growing use of generative artificial intelligence as a tool for creating multilingual content and bypassing both machine and traditional translation methods, this study explores the ability of linguistically trained individuals to discern machine-generated output from human-written text (HT). After brief training sessions on the textual anomalies typically found in synthetic text (ST), twenty-three postgraduate translation students analysed excerpts of Italian prose and assigned likelihood scores to indicate whether they believed they were human-written or AI-generated (ChatGPT-4o). The results show that, on average, the students struggled to distinguish between HT and ST, with only two participants achieving notable accuracy. Closer analysis revealed that the students often identified the same textual anomalies in both HT and ST, although features such as low burstiness and self-contradiction were more frequently associated with ST. These findings suggest the need for improvements in the preparatory training. Moreover, the study raises questions about the necessity of editing synthetic text to make it sound more human-like and recommends further research to determine whether AI-generated text is already sufficiently natural-sounding not to require further refinement. 

**Abstract (ZH)**: 生成式人工智能作为创建多语言内容并绕过机器和传统翻译方法的工具日益普及：本研究探讨了语言训练人员识别机器生成输出与人类撰写的文本的能力（HT）。经过简短的培训 sessions 有关合成文本（ST）中通常存在的文本异常，二十多名翻译硕士学生分析了意大利散文片段，并分配概率分数以表明他们认为这些片段是人类撰写的还是 AI 生成的（ChatGPT-4o）。结果显示，平均而言，学生难以区分 HT 和 ST，仅有两名参与者表现出较高的准确性。进一步分析表明，学生在 HT 和 ST 中经常识别相同的文本异常，尽管诸如低burstiness 和自相矛盾等特征更常与 ST 相关。这些发现表明需要改进预备培训。此外，研究还提出了一个问题，即是否有必要编辑合成文本使其听起来更接近人类撰写的内容，并建议进一步研究以确定 AI 生成的文本是否已足够自然无需进一步润色。 

---
# Synthetic Aircraft Trajectory Generation Using Time-Based VQ-VAE 

**Title (ZH)**: 基于时间的VQ-VAE合成飞机轨迹生成 

**Authors**: Abdulmajid Murad, Massimiliano Ruocco  

**Link**: [PDF](https://arxiv.org/pdf/2504.09101)  

**Abstract**: In modern air traffic management, generating synthetic flight trajectories has emerged as a promising solution for addressing data scarcity, protecting sensitive information, and supporting large-scale analyses. In this paper, we propose a novel method for trajectory synthesis by adapting the Time-Based Vector Quantized Variational Autoencoder (TimeVQVAE). Our approach leverages time-frequency domain processing, vector quantization, and transformer-based priors to capture both global and local dynamics in flight data. By discretizing the latent space and integrating transformer priors, the model learns long-range spatiotemporal dependencies and preserves coherence across entire flight paths. We evaluate the adapted TimeVQVAE using an extensive suite of quality, statistical, and distributional metrics, as well as a flyability assessment conducted in an open-source air traffic simulator. Results indicate that TimeVQVAE outperforms a temporal convolutional VAE baseline, generating synthetic trajectories that mirror real flight data in terms of spatial accuracy, temporal consistency, and statistical properties. Furthermore, the simulator-based assessment shows that most generated trajectories maintain operational feasibility, although occasional outliers underscore the potential need for additional domain-specific constraints. Overall, our findings underscore the importance of multi-scale representation learning for capturing complex flight behaviors and demonstrate the promise of TimeVQVAE in producing representative synthetic trajectories for downstream tasks such as model training, airspace design, and air traffic forecasting. 

**Abstract (ZH)**: 现代空中交通管理中基于时间矢量量化变分自编码器的轨迹合成方法 

---
# Privacy Preservation in Gen AI Applications 

**Title (ZH)**: Gen AI应用中的隐私保护 

**Authors**: Swetha S, Ram Sundhar K Shaju, Rakshana M, Ganesh R, Balavedhaa S, Thiruvaazhi U  

**Link**: [PDF](https://arxiv.org/pdf/2504.09095)  

**Abstract**: The ability of machines to comprehend and produce language that is similar to that of humans has revolutionized sectors like customer service, healthcare, and finance thanks to the quick advances in Natural Language Processing (NLP), which are fueled by Generative Artificial Intelligence (AI) and Large Language Models (LLMs). However, because LLMs trained on large datasets may unintentionally absorb and reveal Personally Identifiable Information (PII) from user interactions, these capabilities also raise serious privacy concerns. Deep neural networks' intricacy makes it difficult to track down or stop the inadvertent storing and release of private information, which raises serious concerns about the privacy and security of AI-driven data. This study tackles these issues by detecting Generative AI weaknesses through attacks such as data extraction, model inversion, and membership inference. A privacy-preserving Generative AI application that is resistant to these assaults is then developed. It ensures privacy without sacrificing functionality by using methods to identify, alter, or remove PII before to dealing with LLMs. In order to determine how well cloud platforms like Microsoft Azure, Google Cloud, and AWS provide privacy tools for protecting AI applications, the study also examines these technologies. In the end, this study offers a fundamental privacy paradigm for generative AI systems, focusing on data security and moral AI implementation, and opening the door to a more secure and conscientious use of these tools. 

**Abstract (ZH)**: 机器在理解和生成与人类相似的语言方面的能力，得益于自然语言处理（NLP）的迅速进步，特别是生成型人工智能（AI）和大型语言模型（LLMs）的推动，已经变革了客户服务中心、医疗保健和金融等领域。然而，由于大型数据集训练的LLMs可能会无意中吸收和揭示用户交互中的个人可识别信息（PII），这些能力也引发了严重的隐私担忧。深度神经网络的复杂性使得追踪和阻止无意间存储和释放的私密信息变得困难，从而对基于AI的数据的隐私和安全提出了严重关切。本研究通过数据提取、模型逆向工程和成员推理攻击来检测生成型AI的弱点，并开发出一种抵御这些攻击的隐私保护型生成型AI应用。该应用在处理LLMs之前使用方法来识别、修改或移除PII，从而在保证隐私的同时不牺牲功能。此外，研究还评估了微软Azure、Google Cloud和AWS等云平台提供的隐私保护工具，以保护AI应用的安全。最后，本研究为生成型AI系统提供了一个基本的隐私范式，重点关注数据安全和道德AI的实施，从而为更安全和负责任地使用这些工具开辟了道路。 

---
# SIFT-50M: A Large-Scale Multilingual Dataset for Speech Instruction Fine-Tuning 

**Title (ZH)**: SIFT-50M：大规模多语言数据集用于语音指令微调 

**Authors**: Prabhat Pandey, Rupak Vignesh Swaminathan, K V Vijay Girish, Arunasish Sen, Jian Xie, Grant P. Strimel, Andreas Schwarz  

**Link**: [PDF](https://arxiv.org/pdf/2504.09081)  

**Abstract**: We introduce SIFT (Speech Instruction Fine-Tuning), a 50M-example dataset designed for instruction fine-tuning and pre-training of speech-text large language models (LLMs). SIFT-50M is built from publicly available speech corpora, which collectively contain 14K hours of speech, and leverages LLMs along with off-the-shelf expert models. The dataset spans five languages, encompassing a diverse range of speech understanding as well as controllable speech generation instructions. Using SIFT-50M, we train SIFT-LLM, which outperforms existing speech-text LLMs on instruction-following benchmarks while achieving competitive performance on foundational speech tasks. To support further research, we also introduce EvalSIFT, a benchmark dataset specifically designed to evaluate the instruction-following capabilities of speech-text LLMs. 

**Abstract (ZH)**: 我们介绍了SIFT（语音指令微调）数据集，这是一个包含500万例的指令微调和预训练语音-文本大规模语言模型（LLMs）的数据集。SIFT-50M源自公开可用的语音语料库，合计包含14000小时的语音数据，并结合了LLMs和现成的专家模型。该数据集覆盖了五种语言，涵盖了广泛的语音理解以及可控语音生成指令。利用SIFT-50M，我们训练了SIFT-LLM，在指令跟随基准测试中优于现有语音-文本LLMs，同时在基础语音任务上也取得了竞争力的表现。为了支持进一步的研究，我们还引入了EvalSIFT，一个专门用于评估语音-文本LLMs指令跟随能力的基准数据集。 

---
# PQS (Prune, Quantize, and Sort): Low-Bitwidth Accumulation of Dot Products in Neural Network Computations 

**Title (ZH)**: PQS（修剪、量化和排序）：神经网络计算中点积的低位宽累积 

**Authors**: Vikas Natesh, H.T. Kung  

**Link**: [PDF](https://arxiv.org/pdf/2504.09064)  

**Abstract**: We present PQS, which uses three techniques together - Prune, Quantize, and Sort - to achieve low-bitwidth accumulation of dot products in neural network computations. In conventional quantized (e.g., 8-bit) dot products, partial results are accumulated into wide (e.g., 32-bit) accumulators to avoid overflows when accumulating intermediate partial sums. However, such wide accumulators increase memory bandwidth usage and reduce energy efficiency. We show that iterative N:M pruning in floating point followed by quantization to 8 (or fewer) bits, and accumulation of partial products in a sorted order ("small to large") allows for accurate, compressed models with short dot product lengths that do not require wide accumulators. We design, analyze, and implement the PQS algorithm to eliminate accumulation overflows at inference time for several neural networks. Our method offers a 2.5x reduction in accumulator bitwidth while achieving model accuracy on par with floating-point baselines for multiple image classification tasks. 

**Abstract (ZH)**: PQS: 结合剪枝、量化和排序的技术实现神经网络计算中低比特宽的点积累积 

---
# A Practical Approach to using Supervised Machine Learning Models to Classify Aviation Safety Occurrences 

**Title (ZH)**: 一种实用的方法，使用监督机器学习模型对 aviation 安全事件进行分类 

**Authors**: Bryan Y. Siow  

**Link**: [PDF](https://arxiv.org/pdf/2504.09063)  

**Abstract**: This paper describes a practical approach of using supervised machine learning (ML) models to assist safety investigators to classify aviation occurrences into either incident or serious incident categories. Our implementation currently deployed as a ML web application is trained on a labelled dataset derived from publicly available aviation investigation reports. A selection of five supervised learning models (Support Vector Machine, Logistic Regression, Random Forest Classifier, XGBoost and K-Nearest Neighbors) were evaluated. This paper showed the best performing ML algorithm was the Random Forest Classifier with accuracy = 0.77, F1 Score = 0.78 and MCC = 0.51 (average of 100 sample runs). The study had also explored the effect of applying Synthetic Minority Over-sampling Technique (SMOTE) to the imbalanced dataset, and the overall observation ranged from no significant effect to substantial degradation in performance for some of the models after the SMOTE adjustment. 

**Abstract (ZH)**: 本文描述了一种实用的方法，使用监督机器学习（ML）模型辅助航空事故调查人员将航空事件分类为事件或严重事件类别。当前部署的基于标记数据集（源自公开的航空调查报告）训练的ML网络应用评估了五种监督学习模型（支持向量机、逻辑回归、随机森林分类器、XGBoost和K-近邻）。结果显示，随机森林分类器表现最佳，准确率=0.77，F1分值=0.78，MCC=0.51（100次样本运行的平均值）。研究还探讨了在不平衡数据集中应用合成少数类过采样技术（SMOTE）的效果，整体观察结果显示，对于一些模型，SMOTE调整后性能显著下降。 

---
# Multimodal 3D Genome Pre-training 

**Title (ZH)**: 多模态3D基因组预训练 

**Authors**: Minghao Yang, Pengteng Li, Yan Liang, Qianyi Cai, Zhihang Zheng, Shichen Zhang, Pengfei Zhang, Zhi-An Huang, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2504.09060)  

**Abstract**: Deep learning techniques have driven significant progress in various analytical tasks within 3D genomics in computational biology. However, a holistic understanding of 3D genomics knowledge remains underexplored. Here, we propose MIX-HIC, the first multimodal foundation model of 3D genome that integrates both 3D genome structure and epigenomic tracks, which obtains unified and comprehensive semantics. For accurate heterogeneous semantic fusion, we design the cross-modal interaction and mapping blocks for robust unified representation, yielding the accurate aggregation of 3D genome knowledge. Besides, we introduce the first large-scale dataset comprising over 1 million pairwise samples of Hi-C contact maps and epigenomic tracks for high-quality pre-training, enabling the exploration of functional implications in 3D genomics. Extensive experiments show that MIX-HIC can significantly surpass existing state-of-the-art methods in diverse downstream tasks. This work provides a valuable resource for advancing 3D genomics research. 

**Abstract (ZH)**: 深度学习技术在计算生物学中的三维基因组各种分析任务中取得了显著进展。然而，对三维基因组知识的整体理解仍然未被充分探索。为此，我们提出了MIX-HIC，这是首个结合三维基因组结构和表观基因组轨迹的多模态基础模型，能够获得统一和全面的语义。为了实现准确的异质语义融合，我们设计了跨模态交互和映射模块，以获得稳健的统一表示，并准确聚集三维基因组知识。此外，我们引入了首个包含超过一百万个Hi-C接触图对和表观基因组轨迹的大规模数据集，用于高质量的预训练，从而能够探索三维基因组的功能含义。广泛实验表明，MIX-HIC在多种下游任务中显著优于现有最先进的方法。这项工作为促进三维基因组研究提供了 valuable 资源。 

---
# Sculpting Memory: Multi-Concept Forgetting in Diffusion Models via Dynamic Mask and Concept-Aware Optimization 

**Title (ZH)**: 塑形记忆：通过动态掩码和概念意识优化在扩散模型中实现多概念遗忘 

**Authors**: Gen Li, Yang Xiao, Jie Ji, Kaiyuan Deng, Bo Hui, Linke Guo, Xiaolong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.09039)  

**Abstract**: Text-to-image (T2I) diffusion models have achieved remarkable success in generating high-quality images from textual prompts. However, their ability to store vast amounts of knowledge raises concerns in scenarios where selective forgetting is necessary, such as removing copyrighted content, reducing biases, or eliminating harmful concepts. While existing unlearning methods can remove certain concepts, they struggle with multi-concept forgetting due to instability, residual knowledge persistence, and generation quality degradation. To address these challenges, we propose \textbf{Dynamic Mask coupled with Concept-Aware Loss}, a novel unlearning framework designed for multi-concept forgetting in diffusion models. Our \textbf{Dynamic Mask} mechanism adaptively updates gradient masks based on current optimization states, allowing selective weight modifications that prevent interference with unrelated knowledge. Additionally, our \textbf{Concept-Aware Loss} explicitly guides the unlearning process by enforcing semantic consistency through superclass alignment, while a regularization loss based on knowledge distillation ensures that previously unlearned concepts remain forgotten during sequential unlearning. We conduct extensive experiments to evaluate our approach. Results demonstrate that our method outperforms existing unlearning techniques in forgetting effectiveness, output fidelity, and semantic coherence, particularly in multi-concept scenarios. Our work provides a principled and flexible framework for stable and high-fidelity unlearning in generative models. The code will be released publicly. 

**Abstract (ZH)**: 文本到图像（T2I）扩散模型在从文本提示生成高质量图像方面取得了显著成功。然而，在需要选择性遗忘的场景中，如删除版权内容、减少偏见或消除有害概念时，它们庞大的知识存储能力引发了关注。虽然现有的遗忘方法可以移除某些概念，但在处理多概念遗忘时却因不稳定、残留知识持久存在以及生成质量下降而遇到了困难。为了解决这些问题，我们提出了一种新的多概念遗忘框架——动态掩码结合概念感知损失（Dynamic Mask coupled with Concept-Aware Loss），该框架旨在扩散模型中实现多概念遗忘。我们的动态掩码机制根据当前的优化状态自适应更新梯度掩码，允许选择性地修改权重以防止与无关知识的干扰。此外，我们提出的概念感知损失明确地指导遗忘过程，通过超类对齐确保语义一致性，而基于知识蒸馏的正则化损失则确保在顺序遗忘过程中之前未学习的概念能够被遗忘。我们进行了广泛的实验以评估我们的方法。结果表明，我们的方法在遗忘效果、输出保真度和语义一致性方面优于现有遗忘技术，尤其是在多概念场景中。我们的工作提供了一个原则性和灵活性兼具的框架，用于生成模型中的稳定和高保真遗忘。代码将公开发布。 

---
# Chest X-ray Classification using Deep Convolution Models on Low-resolution images with Uncertain Labels 

**Title (ZH)**: 基于低分辨率图像和不确定标签的深度卷积模型胸部X光分类 

**Authors**: Snigdha Agarwal, Neelam Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2504.09033)  

**Abstract**: Deep Convolutional Neural Networks have consistently proven to achieve state-of-the-art results on a lot of imaging tasks over the past years' majority of which comprise of high-quality data. However, it is important to work on low-resolution images since it could be a cheaper alternative for remote healthcare access where the primary need of automated pathology identification models occurs. Medical diagnosis using low-resolution images is challenging since critical details may not be easily identifiable. In this paper, we report classification results by experimenting on different input image sizes of Chest X-rays to deep CNN models and discuss the feasibility of classification on varying image sizes. We also leverage the noisy labels in the dataset by proposing a Randomized Flipping of labels techniques. We use an ensemble of multi-label classification models on frontal and lateral studies. Our models are trained on 5 out of the 14 chest pathologies of the publicly available CheXpert dataset. We incorporate techniques such as augmentation, regularization for model improvement and use class activation maps to visualize the neural network's decision making. Comparison with classification results on data from 200 subjects, obtained on the corresponding high-resolution images, reported in the original CheXpert paper, has been presented. For pathologies Cardiomegaly, Consolidation and Edema, we obtain 3% higher accuracy with our model architecture. 

**Abstract (ZH)**: 深度卷积神经网络在胸部X光图像不同分辨率下的分类研究： noisy标签的随机翻转方法及模型性能比较 

---
# MSCCL++: Rethinking GPU Communication Abstractions for Cutting-edge AI Applications 

**Title (ZH)**: MSCCL++: 重新思考面向前沿AI应用的GPU通信抽象 

**Authors**: Aashaka Shah, Abhinav Jangda, Binyang Li, Caio Rocha, Changho Hwang, Jithin Jose, Madan Musuvathi, Olli Saarikivi, Peng Cheng, Qinghua Zhou, Roshan Dathathri, Saeed Maleki, Ziyue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09014)  

**Abstract**: Modern cutting-edge AI applications are being developed over fast-evolving, heterogeneous, nascent hardware devices. This requires frequent reworking of the AI software stack to adopt bottom-up changes from new hardware, which takes time for general-purpose software libraries. Consequently, real applications often develop custom software stacks optimized for their specific workloads and hardware. Custom stacks help quick development and optimization, but incur a lot of redundant efforts across applications in writing non-portable code. This paper discusses an alternative communication library interface for AI applications that offers both portability and performance by reducing redundant efforts while maintaining flexibility for customization. We present MSCCL++, a novel abstraction of GPU communication based on separation of concerns: (1) a primitive interface provides a minimal hardware abstraction as a common ground for software and hardware developers to write custom communication, and (2) higher-level portable interfaces and specialized implementations enable optimization for different hardware environments. This approach makes the primitive interface reusable across applications while enabling highly flexible optimization. Compared to state-of-the-art baselines (NCCL, RCCL, and MSCCL), MSCCL++ achieves speedups of up to 3.8$\times$ for collective communication and up to 15\% for real-world AI inference workloads. MSCCL++ is in production of multiple AI services provided by Microsoft Azure, and is also adopted by RCCL, the GPU collective communication library maintained by AMD. MSCCL++ is open-source and available at this https URL. 

**Abstract (ZH)**: 现代AI应用正在快速发展变化的、异构的新兴硬件设备上进行开发。这要求频繁调整AI软件栈以适应来自新硬件的底层变化，通用软件库需要时间进行适应。因此，实际应用中常常会开发针对其特定工作负载和硬件的定制软件栈。定制栈有助于快速开发和优化，但会在编写非便携代码时产生大量重复工作。本文讨论了一种AI应用的替代通信库接口，该接口通过减少重复工作同时保持定制灵活性来实现便携性和性能。我们提出了MSCCL++，这是一种基于关注点分离的GPU通信的新抽象：（1）原始接口提供了一个最小的硬件抽象作为软件和硬件开发者的共同基础，用于编写自定义通信；（2）高级别便携接口和专门实现能够针对不同的硬件环境进行优化。这种方法使得原始接口可以在应用之间重用，同时允许高度灵活的优化。与最先进的基线（NCCL、RCCL和MSCCL）相比，MSCCL++在集体通信中的性能提高了3.8倍，在实际AI推理工作负载中的性能提高了15%。MSCCL++被微软Azure提供的多个AI服务生产部署，并被AMD维护的GPU集体通信库RCCL采用。MSCCL++是开源软件，可在以下链接获取。 

---
# MCP Bridge: A Lightweight, LLM-Agnostic RESTful Proxy for Model Context Protocol Servers 

**Title (ZH)**: MCP桥接：一种轻量级、LLM无关的RESTful代理，用于模型上下文协议服务器 

**Authors**: Arash Ahmadi, Sarah Sharif, Yaser M. Banad  

**Link**: [PDF](https://arxiv.org/pdf/2504.08999)  

**Abstract**: Large Language Models (LLMs) are increasingly augmented with external tools through standardized interfaces like the Model Context Protocol (MCP). However, current MCP implementations face critical limitations: they typically require local process execution through STDIO transports, making them impractical for resource-constrained environments like mobile devices, web browsers, and edge computing. We present MCP Bridge, a lightweight RESTful proxy that connects to multiple MCP servers and exposes their capabilities through a unified API. Unlike existing solutions, MCP Bridge is fully LLM-agnostic, supporting any backend regardless of vendor. The system implements a risk-based execution model with three security levels standard execution, confirmation workflow, and Docker isolation while maintaining backward compatibility with standard MCP clients. Complementing this server-side infrastructure is a Python based MCP Gemini Agent that facilitates natural language interaction with MCP tools. The evaluation demonstrates that MCP Bridge successfully addresses the constraints of direct MCP connections while providing enhanced security controls and cross-platform compatibility, enabling sophisticated LLM-powered applications in previously inaccessible environments 

**Abstract (ZH)**: Large Language Models (LLMs)通过Model Context Protocol (MCP)标准化接口越来越多地与外部工具集成。然而，当前的MCP实现面临关键限制：它们通常要求通过STDIO传输进行本地进程执行，这使它们在资源受限的环境中（如移动设备、网页浏览器和边缘计算）不可行。我们提出了MCP Bridge，这是一种轻量级的RESTful代理，它可以连接到多个MCP服务器并通过统一的API暴露其功能。与现有解决方案不同，MCP Bridge完全不对大型语言模型（LLM）进行依赖，支持任何后端，无论供应商是谁。该系统实施了一种基于风险的执行模型，具有三种安全级别：标准执行、确认工作流和Docker隔离，同时保持与标准MCP客户端的后向兼容性。该服务器端基础设施的补充是基于Python的MCP Gemini代理，它促进了与MCP工具的自然语言交互。评估表明，MCP Bridge成功地解决了直接MCP连接的限制，提供了增强的安全控制和跨平台兼容性，从而在先前无法访问的环境中启用了复杂的LLM驱动应用。 

---
# Learning from Elders: Making an LLM-powered Chatbot for Retirement Communities more Accessible through User-centered Design 

**Title (ZH)**: 从长者中学习：通过用户中心化设计使由LLM驱动的聊天机器人更适合养老社区 

**Authors**: Luna Xingyu Li, Ray-yuan Chung, Feng Chen, Wenyu Zeng, Yein Jeon, Oleg Zaslavsky  

**Link**: [PDF](https://arxiv.org/pdf/2504.08985)  

**Abstract**: Low technology and eHealth literacy among older adults in retirement communities hinder engagement with digital tools. To address this, we designed an LLM-powered chatbot prototype using a human-centered approach for a local retirement community. Through interviews and persona development, we prioritized accessibility and dual functionality: simplifying internal information retrieval and improving technology and eHealth literacy. A pilot trial with residents demonstrated high satisfaction and ease of use, but also identified areas for further improvement. Based on the feedback, we refined the chatbot using GPT-3.5 Turbo and Streamlit. The chatbot employs tailored prompt engineering to deliver concise responses. Accessible features like adjustable font size, interface theme and personalized follow-up responses were implemented. Future steps include enabling voice-to-text function and longitudinal intervention studies. Together, our results highlight the potential of LLM-driven chatbots to empower older adults through accessible, personalized interactions, bridging literacy gaps in retirement communities. 

**Abstract (ZH)**: 低技术素养和电子健康素养阻碍退休社区老年人使用数字工具。为解决这一问题，我们采用以人为中心的方法设计了一个基于大语言模型的聊天机器人原型，应用于当地退休社区。通过访谈和角色开发，我们优先考虑了可访问性和双功能性：简化内部信息检索并提高技术素养和电子健康素养。初步试用居民结果显示高满意度和易用性，但也发现了改进的空间。根据反馈，我们使用GPT-3.5 Turbo和Streamlit进一步优化了聊天机器人。聊天机器人采用定制提示工程以提供简洁的响应。可访问功能包括可调节字体大小、界面主题和个人化后续响应。未来步骤包括实现语音转文本功能和纵向干预研究。我们的结果共同强调了基于大语言模型的聊天机器人在通过可访问和个性化的交互来赋能老年人方面的潜力，弥补退休社区中的素养差距。 

---
# AGENT: An Aerial Vehicle Generation and Design Tool Using Large Language Models 

**Title (ZH)**: 基于大型语言模型的无人机生成与设计工具 

**Authors**: Colin Samplawski, Adam D. Cobb, Susmit Jha  

**Link**: [PDF](https://arxiv.org/pdf/2504.08981)  

**Abstract**: Computer-aided design (CAD) is a promising application area for emerging artificial intelligence methods. Traditional workflows for cyberphysical systems create detailed digital models which can be evaluated by physics simulators in order to narrow the search space before creating physical prototypes. A major bottleneck of this approach is that the simulators are often computationally expensive and slow. Recent advancements in AI methods offer the possibility to accelerate these pipelines. We use the recently released AircraftVerse dataset, which is especially suited for developing and evaluating large language models for designs. AircraftVerse contains a diverse set of UAV designs represented via textual design trees together with detailed physics simulation results. Following the recent success of large language models (LLMs), we propose AGENT (Aircraft GENeraTor). AGENT is a comprehensive design tool built on the CodeT5+ LLM which learns powerful representations of aircraft textual designs directly from JSON files. We develop a curriculum of training tasks which imbues a single model with a suite of useful features. AGENT is able to generate designs conditioned on properties of flight dynamics (hover time, maximum speed, etc.). Additionally, AGENT can issue evaluations of designs allowing it to act as a surrogate model of the physics simulation that underlies the AircraftVerse dataset. We present a series of experiments which demonstrate our system's abilities. We are able to achieve strong performance using the smallest member of the CodeT5+ family (220M parameters). This allows for a flexible and powerful system which can be executed on a single GPU enabling a clear path toward future deployment. 

**Abstract (ZH)**: 计算机辅助设计（CAD）是新兴人工智能方法的一个有前途的应用领域。传统的网络物理系统工作流程创建详细的数字模型，这些模型可以在物理仿真器中评估，以便在创建物理原型之前缩小搜索空间。这种方法的主要瓶颈是，仿真器通常计算密集型且运行缓慢。近年来，人工智能方法的进展为加速这些管道提供了可能性。我们使用最近发布的AircraftVerse数据集，该数据集特别适合用于开发和评估设计领域的大型语言模型。AircraftVerse包含通过文本设计树表示的多样化的无人机设计，以及详细的物理仿真结果。受大型语言模型（LLMs）近期成功的影响，我们提出了一种名为AGENT（Aircraft GENeraTor）的设计工具。AGENT基于CodeT5+ LLM构建，可以直接从JSON文件学习飞机文本设计的强大表示。我们开发了一门训练课程，赋予单一模型一系列有用的特征。AGENT能够根据飞行动力学属性（悬停时间、最大速度等）生成设计。此外，AGENT还可以对设计进行评估，使其成为支持AircraftVerse数据集基础物理仿真模型的代理模型。我们展示了系统的实验，证明了其能力。我们使用CodeT5+家族中最小的成员（220M参数）实现了强劲的表现。这使得系统具有灵活性和强大性，并且可以在单块GPU上运行，为未来的部署提供清晰的道路。 

---
# On Large-scale Evaluation of Embedding Models for Knowledge Graph Completion 

**Title (ZH)**: 大规模评估嵌入模型在知识图谱补全中的效果 

**Authors**: Nasim Shirvani-Mahdavi, Farahnaz Akrami, Chengkai Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.08970)  

**Abstract**: Knowledge graph embedding (KGE) models are extensively studied for knowledge graph completion, yet their evaluation remains constrained by unrealistic benchmarks. Commonly used datasets are either faulty or too small to reflect real-world data. Few studies examine the role of mediator nodes, which are essential for modeling n-ary relationships, or investigate model performance variation across domains. Standard evaluation metrics rely on the closed-world assumption, which penalizes models for correctly predicting missing triples, contradicting the fundamental goals of link prediction. These metrics often compress accuracy assessment into a single value, obscuring models' specific strengths and weaknesses. The prevailing evaluation protocol operates under the unrealistic assumption that an entity's properties, for which values are to be predicted, are known in advance. While alternative protocols such as property prediction, entity-pair ranking and triple classification address some of these limitations, they remain underutilized. This paper conducts a comprehensive evaluation of four representative KGE models on large-scale datasets FB-CVT-REV and FB+CVT-REV. Our analysis reveals critical insights, including substantial performance variations between small and large datasets, both in relative rankings and absolute metrics, systematic overestimation of model capabilities when n-ary relations are binarized, and fundamental limitations in current evaluation protocols and metrics. 

**Abstract (ZH)**: 知识图嵌入（KGE）模型在知识图完成中的广泛研究仍未摆脱不现实的评估基准的限制。 

---
# MotionDreamer: One-to-Many Motion Synthesis with Localized Generative Masked Transformer 

**Title (ZH)**: MotionDreamer：基于局部生成遮蔽变压器的一对多运动合成 

**Authors**: Yilin Wang, Chuan Guo, Yuxuan Mu, Muhammad Gohar Javed, Xinxin Zuo, Juwei Lu, Hai Jiang, Li Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.08959)  

**Abstract**: Generative masked transformers have demonstrated remarkable success across various content generation tasks, primarily due to their ability to effectively model large-scale dataset distributions with high consistency. However, in the animation domain, large datasets are not always available. Applying generative masked modeling to generate diverse instances from a single MoCap reference may lead to overfitting, a challenge that remains unexplored. In this work, we present MotionDreamer, a localized masked modeling paradigm designed to learn internal motion patterns from a given motion with arbitrary topology and duration. By embedding the given motion into quantized tokens with a novel distribution regularization method, MotionDreamer constructs a robust and informative codebook for local motion patterns. Moreover, a sliding window local attention is introduced in our masked transformer, enabling the generation of natural yet diverse animations that closely resemble the reference motion patterns. As demonstrated through comprehensive experiments, MotionDreamer outperforms the state-of-the-art methods that are typically GAN or Diffusion-based in both faithfulness and diversity. Thanks to the consistency and robustness of the quantization-based approach, MotionDreamer can also effectively perform downstream tasks such as temporal motion editing, \textcolor{update}{crowd animation}, and beat-aligned dance generation, all using a single reference motion. Visit our project page: this https URL 

**Abstract (ZH)**: 生成式掩蔽transformer在各种内容生成任务中已经展示了显著的成功，主要归因于它们能够有效地建模大规模数据集分布且保持高度一致性。然而，在动画领域，大规模数据集并不总是可用的。将生成式掩蔽建模应用于从单个MoCap参考生成多样化实例可能会导致过拟合，这是一个尚未探索的挑战。在这项工作中，我们提出了MotionDreamer，这是一种局部掩蔽建模范式，设计用于从给定具有任意拓扑结构和持续时间的运动中学习内部运动模式。通过使用一种新颖的分布正则化方法将给定运动嵌入量化token中，MotionDreamer 构建了一个robust且信息丰富的代码本，用于局部运动模式。此外，我们引入了一种滑动窗口局部注意力机制，使掩蔽Transformer能够生成自然且多样化、与参考运动模式高度相似的动画。通过全面的实验表明，MotionDreamer在忠诚度和多样性方面都优于现有的基于GAN或扩散的方法。借助基于量化方法的一致性和稳健性，MotionDreamer还能够有效地执行诸如时间运动编辑、人群动画和节奏对齐的舞蹈生成等下游任务，所有这些都仅使用一个参考运动。访问我们的项目页面：this https URL。 

---
# Generating Planning Feedback for Open-Ended Programming Exercises with LLMs 

**Title (ZH)**: 使用大语言模型生成开放型编程练习的规划反馈 

**Authors**: Mehmet Arif Demirtaş, Claire Zheng, Max Fowler, Kathryn Cunningham  

**Link**: [PDF](https://arxiv.org/pdf/2504.08958)  

**Abstract**: To complete an open-ended programming exercise, students need to both plan a high-level solution and implement it using the appropriate syntax. However, these problems are often autograded on the correctness of the final submission through test cases, and students cannot get feedback on their planning process. Large language models (LLM) may be able to generate this feedback by detecting the overall code structure even for submissions with syntax errors. To this end, we propose an approach that detects which high-level goals and patterns (i.e. programming plans) exist in a student program with LLMs. We show that both the full GPT-4o model and a small variant (GPT-4o-mini) can detect these plans with remarkable accuracy, outperforming baselines inspired by conventional approaches to code analysis. We further show that the smaller, cost-effective variant (GPT-4o-mini) achieves results on par with state-of-the-art (GPT-4o) after fine-tuning, creating promising implications for smaller models for real-time grading. These smaller models can be incorporated into autograders for open-ended code-writing exercises to provide feedback for students' implicit planning skills, even when their program is syntactically incorrect. Furthermore, LLMs may be useful in providing feedback for problems in other domains where students start with a set of high-level solution steps and iteratively compute the output, such as math and physics problems. 

**Abstract (ZH)**: 使用大型语言模型检测学生程序中的高层次目标和模式以实现开放编程练习的实时评估 

---
# Forecasting Cryptocurrency Prices using Contextual ES-adRNN with Exogenous Variables 

**Title (ZH)**: 基于外生变量的上下文ES-adRNN加密货币价格预测 

**Authors**: Slawek Smyl, Grzegorz Dudek, Paweł Pełka  

**Link**: [PDF](https://arxiv.org/pdf/2504.08947)  

**Abstract**: In this paper, we introduce a new approach to multivariate forecasting cryptocurrency prices using a hybrid contextual model combining exponential smoothing (ES) and recurrent neural network (RNN). The model consists of two tracks: the context track and the main track. The context track provides additional information to the main track, extracted from representative series. This information as well as information extracted from exogenous variables is dynamically adjusted to the individual series forecasted by the main track. The RNN stacked architecture with hierarchical dilations, incorporating recently developed attentive dilated recurrent cells, allows the model to capture short and long-term dependencies across time series and dynamically weight input information. The model generates both point daily forecasts and predictive intervals for one-day, one-week and four-week horizons. We apply our model to forecast prices of 15 cryptocurrencies based on 17 input variables and compare its performance with that of comparative models, including both statistical and ML ones. 

**Abstract (ZH)**: 本研究提出了一种结合指数平滑和递归神经网络的混合上下文模型，用于多变量预测加密货币价格。该模型包含两个轨道：上下文轨道和主轨道。上下文轨道为主轨道提供额外信息，这些信息来自代表性序列，并且该信息以及来自外生变量的信息会动态调整以适应主轨道预测的个体序列。采用嵌套扩张的递归神经网络堆叠架构，结合了近期发展的注意扩张递归单元，使模型能够捕捉时间序列中的短期和长期依赖关系，并动态加权输入信息。该模型生成了一天、一周和四周时间范围内的点预测和预测区间。我们将该模型应用于基于17个输入变量预测15种加密货币的价格，并将其性能与统计模型和机器学习模型的性能进行了比较。 

---
# Investigating the Treacherous Turn in Deep Reinforcement Learning 

**Title (ZH)**: 探究深度强化学习中的危险转折 

**Authors**: Chace Ashcraft, Kiran Karra, Josh Carney, Nathan Drenkow  

**Link**: [PDF](https://arxiv.org/pdf/2504.08943)  

**Abstract**: The Treacherous Turn refers to the scenario where an artificial intelligence (AI) agent subtly, and perhaps covertly, learns to perform a behavior that benefits itself but is deemed undesirable and potentially harmful to a human supervisor. During training, the agent learns to behave as expected by the human supervisor, but when deployed to perform its task, it performs an alternate behavior without the supervisor there to prevent it. Initial experiments applying DRL to an implementation of the A Link to the Past example do not produce the treacherous turn effect naturally, despite various modifications to the environment intended to produce it. However, in this work, we find the treacherous behavior to be reproducible in a DRL agent when using other trojan injection strategies. This approach deviates from the prototypical treacherous turn behavior since the behavior is explicitly trained into the agent, rather than occurring as an emergent consequence of environmental complexity or poor objective specification. Nonetheless, these experiments provide new insights into the challenges of producing agents capable of true treacherous turn behavior. 

**Abstract (ZH)**: 险恶的转变是指人工智能（AI）代理在不被人类监督者明显察觉的情况下，学习执行有益于自身却被视为不良甚至可能有害的行为。在训练过程中，代理学习按照人类监督者的要求行事，但在部署执行任务时，会执行替代行为，而没有监督者阻止。尽管针对A Link to the Past示例的深层强化学习实验在环境的各种修改后并未自然产生险恶的转变效果，但在本研究中，我们发现使用其他木马注入策略可以在DRL代理中重现险恶行为。这种做法不同于典型的险恶转变行为，因为代理的行为是明确训练进来的，而不是环境复杂性或目标定义不良的次生结果。尽管如此，这些实验为生成能够实施真正险恶转变行为的代理提供了一些新的见解。 

---
# AgentRewardBench: Evaluating Automatic Evaluations of Web Agent Trajectories 

**Title (ZH)**: AgentRewardBench: 评估Web代理轨迹自动评估的方法 

**Authors**: Xing Han Lù, Amirhossein Kazemnejad, Nicholas Meade, Arkil Patel, Dongchan Shin, Alejandra Zambrano, Karolina Stańczak, Peter Shaw, Christopher J. Pal, Siva Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2504.08942)  

**Abstract**: Web agents enable users to perform tasks on web browsers through natural language interaction. Evaluating web agents trajectories is an important problem, since it helps us determine whether the agent successfully completed the tasks. Rule-based methods are widely used for this purpose, but they are challenging to extend to new tasks and may not always recognize successful trajectories. We may achieve higher accuracy through human evaluation, but the process would be substantially slower and more expensive. Automatic evaluations with LLMs may avoid the challenges of designing new rules and manually annotating trajectories, enabling faster and cost-effective evaluation. However, it is unclear how effective they are at evaluating web agents. To this end, we propose AgentRewardBench, the first benchmark to assess the effectiveness of LLM judges for evaluating web agents. AgentRewardBench contains 1302 trajectories across 5 benchmarks and 4 LLMs. Each trajectory in AgentRewardBench is reviewed by an expert, who answers questions pertaining to the success, side effects, and repetitiveness of the agent. Using our benchmark, we evaluate 12 LLM judges and find that no single LLM excels across all benchmarks. We also find that the rule-based evaluation used by common benchmarks tends to underreport the success rate of web agents, highlighting a key weakness of rule-based evaluation and the need to develop more flexible automatic evaluations. We release the benchmark at: this https URL 

**Abstract (ZH)**: Web代理使用户能够通过自然语言交互在网页浏览器中执行任务。评估Web代理轨迹是一个重要的问题，因为它有助于我们确定代理是否成功完成了任务。基于规则的方法广泛用于此目的，但在扩展到新任务时具有挑战性，且可能不一定总是能识别成功的轨迹。我们可以通过人类评估获得更高的准确性，但过程会变得显著更慢且更昂贵。使用LLM进行自动评估可以避免设计新规则和手动标注轨迹的挑战，从而实现更快且成本效益更高的评估。然而，尚不清楚它们在评估Web代理时的有效性如何。为此，我们提出了AgentRewardBench，这是第一个评估LLM裁判在评估Web代理方面有效性的工作基准。AgentRewardBench包含5个基准和4个LLM中的1302条轨迹。AgentRewardBench中的每条轨迹均由专家审阅，专家回答与代理成功、副作用和重复性相关的问题。使用我们的基准，我们评估了12个LLM裁判，发现没有一种LLM在所有基准中都表现出色。我们还发现，常用基准中使用的基于规则的评估倾向于低估Web代理的成功率，突显了基于规则的评估的关键弱点，并强调了开发更灵活的自动评估的必要性。我们在此处发布了基准：this https URL 

---
# Combining Forecasts using Meta-Learning: A Comparative Study for Complex Seasonality 

**Title (ZH)**: 基于元学习的组合预测：复杂季节性的比较研究 

**Authors**: Grzegorz Dudek  

**Link**: [PDF](https://arxiv.org/pdf/2504.08940)  

**Abstract**: In this paper, we investigate meta-learning for combining forecasts generated by models of different types. While typical approaches for combining forecasts involve simple averaging, machine learning techniques enable more sophisticated methods of combining through meta-learning, leading to improved forecasting accuracy. We use linear regression, $k$-nearest neighbors, multilayer perceptron, random forest, and long short-term memory as meta-learners. We define global and local meta-learning variants for time series with complex seasonality and compare meta-learners on multiple forecasting problems, demonstrating their superior performance compared to simple averaging. 

**Abstract (ZH)**: 本文探讨了使用元学习方法结合不同类型模型生成的预测值。虽然传统的预测组合方法通常涉及简单的平均值，但机器学习技术通过元学习使得预测组合更加复杂，从而提高预测准确性。我们使用线性回归、$k$-最近邻、多层感知器、随机森林和长短期记忆网络作为元学习器。我们为具有复杂季节性的时间序列定义了全局和局部元学习变体，并在多个预测问题上比较了元学习器的性能，证明了它们优于简单的平均值方法。 

---
# Long Context In-Context Compression by Getting to the Gist of Gisting 

**Title (ZH)**: 通过提炼要旨进行长上下文内省压缩 

**Authors**: Aleksandar Petrov, Mark Sandler, Andrey Zhmoginov, Nolan Miller, Max Vladymyrov  

**Link**: [PDF](https://arxiv.org/pdf/2504.08934)  

**Abstract**: Long context processing is critical for the adoption of LLMs, but existing methods often introduce architectural complexity that hinders their practical adoption. Gisting, an in-context compression method with no architectural modification to the decoder transformer, is a promising approach due to its simplicity and compatibility with existing frameworks. While effective for short instructions, we demonstrate that gisting struggles with longer contexts, with significant performance drops even at minimal compression rates. Surprisingly, a simple average pooling baseline consistently outperforms gisting. We analyze the limitations of gisting, including information flow interruptions, capacity limitations and the inability to restrict its attention to subsets of the context. Motivated by theoretical insights into the performance gap between gisting and average pooling, and supported by extensive experimentation, we propose GistPool, a new in-context compression method. GistPool preserves the simplicity of gisting, while significantly boosting its performance on long context compression tasks. 

**Abstract (ZH)**: 长上下文处理对于大语言模型的采用至关重要，但现有方法往往引入了架构复杂性，阻碍了其实用采用。Gisting是一种无需对解码器变换器进行架构修改的上下文内压缩方法，由于其简单性和与现有框架的兼容性，它是一种有前景的方法。尽管对于短指令有效，我们发现Gisting在长上下文中表现出困难，在最小压缩率下甚至会出现显著的性能下降。令人惊讶的是，一个简单的平均池化基线始终优于Gisting。我们分析了Gisting的局限性，包括信息流中断、容量限制以及无法将注意力限制在上下文的子集上。受到Gisting与平均池化之间性能差距的理论洞见以及广泛实验的支持，我们提出了一种新的上下文内压缩方法GistPool。GistPool保留了Gisting的简单性，同时在其在长上下文压缩任务中的性能上有了显著提升。 

---
# A convergence law for continuous logic and continuous structures with finite domains 

**Title (ZH)**: 连续逻辑中连续结构有限域的收敛定律 

**Authors**: Vera Koponen  

**Link**: [PDF](https://arxiv.org/pdf/2504.08923)  

**Abstract**: We consider continuous relational structures with finite domain $[n] := \{1, \ldots, n\}$ and a many valued logic, $CLA$, with values in the unit interval and which uses continuous connectives and continuous aggregation functions. $CLA$ subsumes first-order logic on ``conventional'' finite structures. To each relation symbol $R$ and identity constraint $ic$ on a tuple the length of which matches the arity of $R$ we associate a continuous probability density function $\mu_R^{ic} : [0, 1] \to [0, \infty)$.
We also consider a probability distribution on the set $\mathbf{W}_n$ of continuous structures with domain $[n]$ which is such that for every relation symbol $R$, identity constraint $ic$, and tuple $\bar{a}$ satisfying $ic$, the distribution of the value of $R(\bar{a})$ is given by $\mu_R^{ic}$, independently of the values for other relation symbols or other tuples.
In this setting we prove that every formula in $CLA$ is asymptotically equivalent to a formula without any aggregation function. This is used to prove a convergence law for $CLA$ which reads as follows for formulas without free variables: If $\varphi \in CLA$ has no free variable and $I \subseteq [0, 1]$ is an interval, then there is $\alpha \in [0, 1]$ such that, as $n$ tends to infinity, the probability that the value of $\varphi$ is in $I$ tends to $\alpha$. 

**Abstract (ZH)**: 我们考虑具有有限域$[n] := \{1, \ldots, n\}$的连续关系结构及取值于单位 Interval 的多值逻辑 $CLA$，该逻辑使用连续联结词和连续聚集函数。$CLA$ 包含在“常规”有限结构上的一阶逻辑。对于每个关系符 $R$ 和相应的身份约束 $ic$，以及满足 $ic$ 的元组 $\bar{a}$，我们关联一个连续概率密度函数 $\mu_R^{ic} : [0, 1] \to [0, \infty)$。

我们还考虑了在连续结构集合 $\mathbf{W}_n$ 上的概率分布，该集合的域为 $[n]$，使得对于每一个关系符 $R$、身份约束 $ic$ 和满足 $ic$ 的元组 $\bar{a}$，$R(\bar{a})$ 的值分布由 $\mu_R^{ic}$ 给出，并且与其他关系符或元组的值无关。

在此背景下，我们证明了 $CLA$ 中的每个公式在无穷大时几乎等价于不含聚集函数的公式。这被用来证明 $CLA$ 的收敛定律如下：如果 $\varphi \in CLA$ 没有自由变量，且 $I \subseteq [0, 1]$ 是一个区间，则存在 $\alpha \in [0, 1]$，当 $n$ 趋向无穷时，$\varphi$ 的值在 $I$ 中的概率趋于 $\alpha$。 

---
# Are We Merely Justifying Results ex Post Facto? Quantifying Explanatory Inversion in Post-Hoc Model Explanations 

**Title (ZH)**: 我们在 merely 后面的内容翻译有误，正确的翻译应为：

Are We Merely Justifying Results A Posteriori? Quantifying Explanatory Inversion in Post-Hoc Model Explanations 

**Authors**: Zhen Tan, Song Wang, Yifan Li, Yu Kong, Jundong Li, Tianlong Chen, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08919)  

**Abstract**: Post-hoc explanation methods provide interpretation by attributing predictions to input features. Natural explanations are expected to interpret how the inputs lead to the predictions. Thus, a fundamental question arises: Do these explanations unintentionally reverse the natural relationship between inputs and outputs? Specifically, are the explanations rationalizing predictions from the output rather than reflecting the true decision process? To investigate such explanatory inversion, we propose Inversion Quantification (IQ), a framework that quantifies the degree to which explanations rely on outputs and deviate from faithful input-output relationships. Using the framework, we demonstrate on synthetic datasets that widely used methods such as LIME and SHAP are prone to such inversion, particularly in the presence of spurious correlations, across tabular, image, and text domains. Finally, we propose Reproduce-by-Poking (RBP), a simple and model-agnostic enhancement to post-hoc explanation methods that integrates forward perturbation checks. We further show that under the IQ framework, RBP theoretically guarantees the mitigation of explanatory inversion. Empirically, for example, on the synthesized data, RBP can reduce the inversion by 1.8% on average across iconic post-hoc explanation approaches and domains. 

**Abstract (ZH)**: 后验解释方法通过将预测归因于输入特征来提供解释。自然解释期望解释输入如何导致预测。因此，一个基本问题出现了：这些解释是否无意中逆转了输入与输出之间的自然关系？具体来说，它们是否在解释预测时理性化了输出，而不是反映真正的决策过程？为了调查这种解释逆转现象，我们提出了逆转量化（Inversion Quantification, IQ）框架，该框架量化了解释依赖于输出的程度以及偏离忠实的输入-输出关系的程度。使用该框架，我们演示在合成数据集上，广泛使用的LIME和SHAP方法在存在虚假相关性时，尤其是对于表格、图像和文本领域中的广泛后验解释方法，容易发生逆转。最后，我们提出了一种简单且模型无关的增强方法Reproduce-by-Poking（RBP），该方法整合了前向扰动检查。进一步证明，在IQ框架下，RBP理论上保证了解释逆转的缓解。实验中，例如，在合成数据上，RBP可以将图标型后验解释方法和领域中的逆转平均减少1.8%。 

---
# Parameter-Free Fine-tuning via Redundancy Elimination for Vision Foundation Models 

**Title (ZH)**: 基于冗余消除的参数-Free微调方法用于视觉基础模型 

**Authors**: Jiahuan Long, Tingsong Jiang, Wen Yao, Yizhe Xiong, Zhengqin Xu, Shuai Jia, Chao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.08915)  

**Abstract**: Vision foundation models (VFMs) are large pre-trained models that form the backbone of various vision tasks. Fine-tuning VFMs can further unlock their potential for downstream tasks or scenarios. However, VFMs often contain significant feature redundancy, which may limit their adaptability to new tasks. In this paper, we investigate the redundancies in the segment anything model (SAM) and then propose a parameter-free fine-tuning method to address this issue. Unlike traditional fine-tuning methods that adjust parameters, our method emphasizes selecting, reusing, and enhancing pre-trained features, offering a new perspective on model fine-tuning. Specifically, we introduce a channel selection algorithm based on the model's output difference to identify redundant and effective channels. By selectively replacing the redundant channels with more effective ones, we filter out less useful features and reuse the more relevant features to downstream tasks, thereby enhancing the task-specific feature representation. Experiments on both out-of-domain and in-domain datasets demonstrate the efficiency and effectiveness of our method. Notably, our approach can seamlessly integrate with existing fine-tuning strategies (e.g., LoRA, Adapter), further boosting the performance of already fine-tuned models. Moreover, since our channel selection involves only model inference, our method significantly reduces computational and GPU memory overhead. 

**Abstract (ZH)**: Vision 基础模型中的冗余及其参数自由的微调方法 

---
# HyperCore: The Core Framework for Building Hyperbolic Foundation Models with Comprehensive Modules 

**Title (ZH)**: HyperCore: 构建全面模块化双曲基础模型的核心框架 

**Authors**: Neil He, Menglin Yang, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2504.08912)  

**Abstract**: Hyperbolic neural networks have emerged as a powerful tool for modeling hierarchical data across diverse modalities. Recent studies show that token distributions in foundation models exhibit scale-free properties, suggesting that hyperbolic space is a more suitable ambient space than Euclidean space for many pre-training and downstream tasks. However, existing tools lack essential components for building hyperbolic foundation models, making it difficult to leverage recent advancements. We introduce HyperCore, a comprehensive open-source framework that provides core modules for constructing hyperbolic foundation models across multiple modalities. HyperCore's modules can be effortlessly combined to develop novel hyperbolic foundation models, eliminating the need to extensively modify Euclidean modules from scratch and possible redundant research efforts. To demonstrate its versatility, we build and test the first fully hyperbolic vision transformers (LViT) with a fine-tuning pipeline, the first fully hyperbolic multimodal CLIP model (L-CLIP), and a hybrid Graph RAG with a hyperbolic graph encoder. Our experiments demonstrate that LViT outperforms its Euclidean counterpart. Additionally, we benchmark and reproduce experiments across hyperbolic GNNs, CNNs, Transformers, and vision Transformers to highlight HyperCore's advantages. 

**Abstract (ZH)**: Hyperbolic Neural Networks Have Emerged as a Powerful Tool for Modeling Hierarchical Data Across Diverse Modalities: Introducing HyperCore, a Comprehensive Open-Source Framework for Constructing Hyperbolic Foundation Models 

---
# Robust SAM: On the Adversarial Robustness of Vision Foundation Models 

**Title (ZH)**: 鲁棒SAM：视觉基础模型的对抗鲁棒性研究 

**Authors**: Jiahuan Long, Zhengqin Xu, Tingsong Jiang, Wen Yao, Shuai Jia, Chao Ma, Xiaoqian Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.08906)  

**Abstract**: The Segment Anything Model (SAM) is a widely used vision foundation model with diverse applications, including image segmentation, detection, and tracking. Given SAM's wide applications, understanding its robustness against adversarial attacks is crucial for real-world deployment. However, research on SAM's robustness is still in its early stages. Existing attacks often overlook the role of prompts in evaluating SAM's robustness, and there has been insufficient exploration of defense methods to balance the robustness and accuracy. To address these gaps, this paper proposes an adversarial robustness framework designed to evaluate and enhance the robustness of SAM. Specifically, we introduce a cross-prompt attack method to enhance the attack transferability across different prompt types. Besides attacking, we propose a few-parameter adaptation strategy to defend SAM against various adversarial attacks. To balance robustness and accuracy, we use the singular value decomposition (SVD) to constrain the space of trainable parameters, where only singular values are adaptable. Experiments demonstrate that our cross-prompt attack method outperforms previous approaches in terms of attack success rate on both SAM and SAM 2. By adapting only 512 parameters, we achieve at least a 15\% improvement in mean intersection over union (mIoU) against various adversarial attacks. Compared to previous defense methods, our approach enhances the robustness of SAM while maximally maintaining its original performance. 

**Abstract (ZH)**: 段 Anything 模型（SAM）的对抗鲁棒性框架 

---
# Position: Beyond Euclidean -- Foundation Models Should Embrace Non-Euclidean Geometries 

**Title (ZH)**: 位置：超越欧几里得——基础模型应采纳非欧几里得几何 

**Authors**: Neil He, Jiahong Liu, Buze Zhang, Ngoc Bui, Ali Maatouk, Menglin Yang, Irwin King, Melanie Weber, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2504.08896)  

**Abstract**: In the era of foundation models and Large Language Models (LLMs), Euclidean space has been the de facto geometric setting for machine learning architectures. However, recent literature has demonstrated that this choice comes with fundamental limitations. At a large scale, real-world data often exhibit inherently non-Euclidean structures, such as multi-way relationships, hierarchies, symmetries, and non-isotropic scaling, in a variety of domains, such as languages, vision, and the natural sciences. It is challenging to effectively capture these structures within the constraints of Euclidean spaces. This position paper argues that moving beyond Euclidean geometry is not merely an optional enhancement but a necessity to maintain the scaling law for the next-generation of foundation models. By adopting these geometries, foundation models could more efficiently leverage the aforementioned structures. Task-aware adaptability that dynamically reconfigures embeddings to match the geometry of downstream applications could further enhance efficiency and expressivity. Our position is supported by a series of theoretical and empirical investigations of prevalent foundation this http URL, we outline a roadmap for integrating non-Euclidean geometries into foundation models, including strategies for building geometric foundation models via fine-tuning, training from scratch, and hybrid approaches. 

**Abstract (ZH)**: 在基础模型和大规模语言模型的时代超越欧几里得几何：非欧几里得几何在下一代基础模型中的必要性 

---
# Distilling and exploiting quantitative insights from Large Language Models for enhanced Bayesian optimization of chemical reactions 

**Title (ZH)**: 从大型语言模型中提炼和利用定量洞察以增强化学反应的贝叶斯优化 

**Authors**: Roshan Patel, Saeed Moayedpour, Louis De Lescure, Lorenzo Kogler-Anele, Alan Cherney, Sven Jager, Yasser Jangjou  

**Link**: [PDF](https://arxiv.org/pdf/2504.08874)  

**Abstract**: Machine learning and Bayesian optimization (BO) algorithms can significantly accelerate the optimization of chemical reactions. Transfer learning can bolster the effectiveness of BO algorithms in low-data regimes by leveraging pre-existing chemical information or data outside the direct optimization task (i.e., source data). Large language models (LLMs) have demonstrated that chemical information present in foundation training data can give them utility for processing chemical data. Furthermore, they can be augmented with and help synthesize potentially multiple modalities of source chemical data germane to the optimization task. In this work, we examine how chemical information from LLMs can be elicited and used for transfer learning to accelerate the BO of reaction conditions to maximize yield. Specifically, we show that a survey-like prompting scheme and preference learning can be used to infer a utility function which models prior chemical information embedded in LLMs over a chemical parameter space; we find that the utility function shows modest correlation to true experimental measurements (yield) over the parameter space despite operating in a zero-shot setting. Furthermore, we show that the utility function can be leveraged to focus BO efforts in promising regions of the parameter space, improving the yield of the initial BO query and enhancing optimization in 4 of the 6 datasets studied. Overall, we view this work as a step towards bridging the gap between the chemistry knowledge embedded in LLMs and the capabilities of principled BO methods to accelerate reaction optimization. 

**Abstract (ZH)**: 机器学习和贝叶斯优化算法可以显著加速化学反应的优化。通过迁移学习，这些算法在数据量有限的情况下可以通过利用预存的化学信息或与直接优化任务无关的数据（即源数据）来增强其有效性。大型语言模型（LLMs）已经显示出，基础训练数据中存在的化学信息可以使它们对处理化学数据具有实用价值，并且可以与源化学数据的多种模态相结合，帮助合成相关于优化任务的化学数据。在这项工作中，我们研究了如何利用LLMs中的化学信息进行迁移学习，以加速基于贝叶斯优化对反应条件的优化以最大化产率。具体而言，我们展示了使用类似调查的提示方案和偏好学习可以推断出一个描述嵌入在LLMs中的先验化学信息的效用函数；尽管在零样本设置下运作，我们发现该效用函数在化学参数空间中与真实的实验测量结果（产率）有一定的相关性。此外，我们展示了效用函数可以用于聚焦于参数空间中具有潜在前景的区域，从而提高初始贝叶斯优化查询的产率，并在研究的6个数据集中有4个数据集中增强了优化。总体而言，我们认为这项工作是朝着弥合嵌入在LLMs中的化学知识与原理性贝叶斯优化方法加速反应优化的能力之间的差距迈出的一步。 

---
# Personalizing Federated Learning for Hierarchical Edge Networks with Non-IID Data 

**Title (ZH)**: 基于非IID数据的层次边缘网络个性化联邦学习 

**Authors**: Seunghyun Lee, Omid Tavallaie, Shuaijun Chen, Kanchana Thilakarathna, Suranga Seneviratne, Adel Nadjaran Toosi, Albert Y. Zomaya  

**Link**: [PDF](https://arxiv.org/pdf/2504.08872)  

**Abstract**: Accommodating edge networks between IoT devices and the cloud server in Hierarchical Federated Learning (HFL) enhances communication efficiency without compromising data privacy. However, devices connected to the same edge often share geographic or contextual similarities, leading to varying edge-level data heterogeneity with different subsets of labels per edge, on top of device-level heterogeneity. This hierarchical non-Independent and Identically Distributed (non-IID) nature, which implies that each edge has its own optimization goal, has been overlooked in HFL research. Therefore, existing edge-accommodated HFL demonstrates inconsistent performance across edges in various hierarchical non-IID scenarios. To ensure robust performance with diverse edge-level non-IID data, we propose a Personalized Hierarchical Edge-enabled Federated Learning (PHE-FL), which personalizes each edge model to perform well on the unique class distributions specific to each edge. We evaluated PHE-FL across 4 scenarios with varying levels of edge-level non-IIDness, with extreme IoT device level non-IIDness. To accurately assess the effectiveness of our personalization approach, we deployed test sets on each edge server instead of the cloud server, and used both balanced and imbalanced test sets. Extensive experiments show that PHE-FL achieves up to 83 percent higher accuracy compared to existing federated learning approaches that incorporate edge networks, given the same number of training rounds. Moreover, PHE-FL exhibits improved stability, as evidenced by reduced accuracy fluctuations relative to the state-of-the-art FedAvg with two-level (edge and cloud) aggregation. 

**Abstract (ZH)**: 多层次边缘增强的个性化联邦学习（PHE-FL）：提高边缘非IID数据下的通信效率和数据隐私保护 

---
# An LLM Framework For Cryptography Over Chat Channels 

**Title (ZH)**: 基于聊天渠道的密码学LLM框架 

**Authors**: Danilo Gligoroski, Mayank Raikwar, Sonu Kumar Jha  

**Link**: [PDF](https://arxiv.org/pdf/2504.08871)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have transformed communication, yet their role in secure messaging remains underexplored, especially in surveillance-heavy environments. At the same time, many governments all over the world are proposing legislation to detect, backdoor, or even ban encrypted communication. That emphasizes the need for alternative ways to communicate securely and covertly over open channels. We propose a novel cryptographic embedding framework that enables covert Public Key or Symmetric Key encrypted communication over public chat channels with humanlike produced texts. Some unique properties of our framework are: 1. It is LLM agnostic, i.e., it allows participants to use different local LLM models independently; 2. It is pre- or post-quantum agnostic; 3. It ensures indistinguishability from human-like chat-produced texts. Thus, it offers a viable alternative where traditional encryption is detectable and restricted. 

**Abstract (ZH)**: 最近大型语言模型的进展已变革了通信方式，但在安全消息传递领域，尤其是在监控密集环境中，其作用仍被极大地忽视。同时，世界各地许多政府提议制定法律以检测、植入后门或甚至禁止加密通信。这突显了在开放信道上进行隐蔽且安全通信的迫切需求。我们提出了一种新颖的加密嵌入框架，该框架允许参与者使用不同的本地大型语言模型独立地在公共聊天渠道上进行类人类生成文本的隐蔽公钥或对称密钥加密通信。该框架的几个独特属性包括：1. 它对大型语言模型无关，即允许参与者独立使用不同的本地大型语言模型；2. 它对预量子或后量子无关；3. 它确保与类人类生成的聊天文本无法区分。因此，它提供了一种传统加密可被检测和限制的替代方案。 

---
# On Transfer-based Universal Attacks in Pure Black-box Setting 

**Title (ZH)**: 基于转移的通用攻击在纯黑盒设置中 

**Authors**: Mohammad A.A.K. Jalwana, Naveed Akhtar, Ajmal Mian, Nazanin Rahnavard, Mubarak Shah  

**Link**: [PDF](https://arxiv.org/pdf/2504.08866)  

**Abstract**: Despite their impressive performance, deep visual models are susceptible to transferable black-box adversarial attacks. Principally, these attacks craft perturbations in a target model-agnostic manner. However, surprisingly, we find that existing methods in this domain inadvertently take help from various priors that violate the black-box assumption such as the availability of the dataset used to train the target model, and the knowledge of the number of classes in the target model. Consequently, the literature fails to articulate the true potency of transferable black-box attacks. We provide an empirical study of these biases and propose a framework that aids in a prior-free transparent study of this paradigm. Using our framework, we analyze the role of prior knowledge of the target model data and number of classes in attack performance. We also provide several interesting insights based on our analysis, and demonstrate that priors cause overestimation in transferability scores. Finally, we extend our framework to query-based attacks. This extension inspires a novel image-blending technique to prepare data for effective surrogate model training. 

**Abstract (ZH)**: 尽管深度视觉模型表现 impressive，但它们易受转移性黑盒对抗攻击的影响。现有方法在这一领域意外地依赖了违反黑盒假设的各种先验，如目标模型训练数据集的可用性和目标模型类别的数量。因此，现有文献未能充分阐述转移性黑盒攻击的真实效能。我们提供了一种实验研究这些偏见的方法，并提出了一种框架，以帮助在无需先验知识的情况下进行透明的研究。利用该框架，我们分析了目标模型数据和类别数量的先验知识如何影响攻击性能，并根据分析提供了若干有趣的见解，证明先验会导致转移性得分的高估。最后，我们将该框架扩展到查询式攻击。这一扩展激发了一种新颖的图像融合技术，用于有效训练替代模型。 

---
# An Evaluation of Cultural Value Alignment in LLM 

**Title (ZH)**: LLM文化价值对齐评估 

**Authors**: Nicholas Sukiennik, Chen Gao, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.08863)  

**Abstract**: LLMs as intelligent agents are being increasingly applied in scenarios where human interactions are involved, leading to a critical concern about whether LLMs are faithful to the variations in culture across regions. Several works have investigated this question in various ways, finding that there are biases present in the cultural representations of LLM outputs. To gain a more comprehensive view, in this work, we conduct the first large-scale evaluation of LLM culture assessing 20 countries' cultures and languages across ten LLMs. With a renowned cultural values questionnaire and by carefully analyzing LLM output with human ground truth scores, we thoroughly study LLMs' cultural alignment across countries and among individual models. Our findings show that the output over all models represents a moderate cultural middle ground. Given the overall skew, we propose an alignment metric, revealing that the United States is the best-aligned country and GLM-4 has the best ability to align to cultural values. Deeper investigation sheds light on the influence of model origin, prompt language, and value dimensions on cultural output. Specifically, models, regardless of where they originate, align better with the US than they do with China. The conclusions provide insight to how LLMs can be better aligned to various cultures as well as provoke further discussion of the potential for LLMs to propagate cultural bias and the need for more culturally adaptable models. 

**Abstract (ZH)**: 大语言模型作为智能代理在涉及人类互动的场景中应用日益增多，引起了对其跨地区文化差异忠实性的关键关注。已有研究从不同角度探索了这一问题，发现大语言模型输出中的文化表现存在偏差。为了获得更为全面的视角，在本工作中，我们首次对10个大语言模型进行了大规模评估，评估了20个国家的文化和语言。通过使用知名的文化价值观问卷，并仔细分析模型输出与人类基准分数，我们全面研究了各国和单个模型间的大语言模型文化对齐情况。研究发现，所有模型的输出总体上反映了中等程度的文化中间地带。鉴于整体偏差，我们提出了一种对齐度量方法，结果显示美国是最对齐的国家，GLM-4在对齐文化价值观方面表现最佳。进一步的研究揭示了模型起源、提示语言和价值观维度对文化输出的影响。具体而言，无论模型源自何处，它们与美国的文化对齐程度总高于与中国的对齐程度。这些结论为如何更好地将大语言模型对齐到不同文化提供了见解，并引发了关于大语言模型传播文化偏见潜在可能性及其需要更具文化适应性的模型的进一步讨论。 

---
# RTLRepoCoder: Repository-Level RTL Code Completion through the Combination of Fine-Tuning and Retrieval Augmentation 

**Title (ZH)**: RTLRepoCoder: 通过微调和检索增强实现仓库级RTL代码完成 

**Authors**: Peiyang Wu, Nan Guo, Junliang Lv, Xiao Xiao, Xiaochun Ye  

**Link**: [PDF](https://arxiv.org/pdf/2504.08862)  

**Abstract**: As an essential part of modern hardware design, manually writing Register Transfer Level (RTL) code such as Verilog is often labor-intensive. Following the tremendous success of large language models (LLMs), researchers have begun to explore utilizing LLMs for generating RTL code. However, current studies primarily focus on generating simple single modules, which can not meet the demands in real world. In fact, due to challenges in managing long-context RTL code and complex cross-file dependencies, existing solutions cannot handle large-scale Verilog repositories in practical hardware development. As the first endeavor to exclusively adapt LLMs for large-scale RTL development, we propose RTLRepoCoder, a groundbreaking solution that incorporates specific fine-tuning and Retrieval-Augmented Generation (RAG) for repository-level Verilog code completion. Open-source Verilog repositories from the real world, along with an extended context size, are used for domain-specific fine-tuning. The optimized RAG system improves the information density of the input context by retrieving relevant code snippets. Tailored optimizations for RAG are carried out, including the embedding model, the cross-file context splitting strategy, and the chunk size. Our solution achieves state-of-the-art performance on public benchmark, significantly surpassing GPT-4 and advanced domain-specific LLMs on Edit Similarity and Exact Match rate. Comprehensive experiments demonstrate the remarkable effectiveness of our approach and offer insights for future work. 

**Abstract (ZH)**: 大规模硬件描述语言代码生成中的LLM应用：RTLRepoCoder 

---
# Diachronic and synchronic variation in the performance of adaptive machine learning systems: The ethical challenges 

**Title (ZH)**: 适应性机器学习系统历时与共时表现的变化：伦理挑战 

**Authors**: Joshua Hatherley, Robert Sparrow  

**Link**: [PDF](https://arxiv.org/pdf/2504.08861)  

**Abstract**: Objectives: Machine learning (ML) has the potential to facilitate "continual learning" in medicine, in which an ML system continues to evolve in response to exposure to new data over time, even after being deployed in a clinical setting. In this paper, we provide a tutorial on the range of ethical issues raised by the use of such "adaptive" ML systems in medicine that have, thus far, been neglected in the literature.
Target audience: The target audiences for this tutorial are the developers of machine learning AI systems, healthcare regulators, the broader medical informatics community, and practicing clinicians.
Scope: Discussions of adaptive ML systems to date have overlooked the distinction between two sorts of variance that such systems may exhibit -- diachronic evolution (change over time) and synchronic variation (difference between cotemporaneous instantiations of the algorithm at different sites) -- and under-estimated the significance of the latter. We highlight the challenges that diachronic evolution and synchronic variation present for the quality of patient care, informed consent, and equity, and discuss the complex ethical trade-offs involved in the design of such systems. 

**Abstract (ZH)**: 目标：机器学习（ML）有潜力促进医学中的“持续学习”，即在临床应用后，ML系统能持续进化以响应新数据的暴露。本文提供了一篇关于此类“自适应”ML系统在医学中使用所引发的一系列伦理问题的教程，目前这些伦理问题在文献中尚未得到充分关注。

目标受众：本文的目标受众包括机器学习AI系统的开发者、医疗监管机构、更广泛的医学信息学社区以及临床实践者。

范围：迄今为止关于自适应ML系统的讨论未能区分这类系统可能出现的两类变异——历时演变（随时间变化）和共时变异（不同地点同一时间点算法实例之间的差异）——并且低估了后者的意义。本文强调历时演变和共时变异对患者护理质量、知情同意和公平性带来的挑战，并讨论了设计这类系统时复杂的伦理权衡。 

---
# A Nonlinear Hash-based Optimization Method for SpMV on GPUs 

**Title (ZH)**: 基于非线性哈希的SpMV在GPU上的优化方法 

**Authors**: Chen Yan, Boyu Diao, Hangda Liu, Zhulin An, Yongjun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08860)  

**Abstract**: Sparse matrix-vector multiplication (SpMV) is a fundamental operation with a wide range of applications in scientific computing and artificial intelligence. However, the large scale and sparsity of sparse matrix often make it a performance bottleneck. In this paper, we highlight the effectiveness of hash-based techniques in optimizing sparse matrix reordering, introducing the Hash-based Partition (HBP) format, a lightweight SpMV approach. HBP retains the performance benefits of the 2D-partitioning method while leveraging the hash transformation's ability to group similar elements, thereby accelerating the pre-processing phase of sparse matrix reordering. Additionally, we achieve parallel load balancing across matrix blocks through a competitive method. Our experiments, conducted on both Nvidia Jetson AGX Orin and Nvidia RTX 4090, show that in the pre-processing step, our method offers an average speedup of 3.53 times compared to the sorting approach and 3.67 times compared to the dynamic programming method employed in Regu2D. Furthermore, in SpMV, our method achieves a maximum speedup of 3.32 times on Orin and 3.01 times on RTX4090 against the CSR format in sparse matrices from the University of Florida Sparse Matrix Collection. 

**Abstract (ZH)**: 基于哈希的稀疏矩阵重新排序技术在稀疏矩阵向量乘法中的应用优化 

---
# PolyConf: Unlocking Polymer Conformation Generation through Hierarchical Generative Models 

**Title (ZH)**: PolyConf：通过层次生成模型解锁聚合物构象生成 

**Authors**: Fanmeng Wang, Wentao Guo, Qi Ou, Hongshuai Wang, Haitao Lin, Hongteng Xu, Zhifeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2504.08859)  

**Abstract**: Polymer conformation generation is a critical task that enables atomic-level studies of diverse polymer materials. While significant advances have been made in designing various conformation generation methods for small molecules and proteins, these methods struggle to generate polymer conformations due to polymers' unique structural characteristics. The scarcity of polymer conformation datasets further limits progress, making this promising area largely unexplored. In this work, we propose PolyConf, a pioneering tailored polymer conformation generation method that leverages hierarchical generative models to unlock new possibilities for this task. Specifically, we decompose the polymer conformation into a series of local conformations (i.e., the conformations of its repeating units), generating these local conformations through an autoregressive model. We then generate corresponding orientation transformations via a diffusion model to assemble these local conformations into the complete polymer conformation. Moreover, we develop the first benchmark with a high-quality polymer conformation dataset derived from molecular dynamics simulations to boost related research in this area. The comprehensive evaluation demonstrates that PolyConf consistently generates high-quality polymer conformations, facilitating advancements in polymer modeling and simulation. 

**Abstract (ZH)**: 聚合物构象生成是实现多样化聚合物材料原子级研究的关键任务。虽然在设计适用于小分子和蛋白质的各种构象生成方法方面取得了显著进展，但这些方法在生成聚合物构象时遇到困难，因为聚合物具有独特的结构性质。聚合物构象数据集的稀缺性进一步限制了进展，使得这一有前景的领域尚未得到充分探索。在本文中，我们提出了PolyConf，这是一种创新的定制化聚合物构象生成方法，利用层级生成模型为该任务解锁新可能。具体来说，我们将聚合物构象分解为一系列局部构象（即其重复单元的构象），并通过自回归模型生成这些局部构象。然后，我们通过扩散模型生成相应的方向变换来将这些局部构象组装成完整的聚合物构象。此外，我们还开发了首个以分子动力学模拟数据为基础的高质量聚合物构象基准数据集，以促进该领域的相关研究。全面的评估表明，PolyConf 一贯可以生成高质量的聚合物构象，从而促进聚合物建模和模拟的进步。 

---
# Examining GPT's Capability to Generate and Map Course Concepts and Their Relationship 

**Title (ZH)**: 考查GPT在生成和映射课程概念及其关系方面的能力 

**Authors**: Tianyuan Yang, Ren Baofeng, Chenghao Gu, Tianjia He, Boxuan Ma, Shinichi Konomi  

**Link**: [PDF](https://arxiv.org/pdf/2504.08856)  

**Abstract**: Extracting key concepts and their relationships from course information and materials facilitates the provision of visualizations and recommendations for learners who need to select the right courses to take from a large number of courses. However, identifying and extracting themes manually is labor-intensive and time-consuming. Previous machine learning-based methods to extract relevant concepts from courses heavily rely on detailed course materials, which necessitates labor-intensive preparation of course materials. This paper investigates the potential of LLMs such as GPT in automatically generating course concepts and their relations. Specifically, we design a suite of prompts and provide GPT with the course information with different levels of detail, thereby generating high-quality course concepts and identifying their relations. Furthermore, we comprehensively evaluate the quality of the generated concepts and relationships through extensive experiments. Our results demonstrate the viability of LLMs as a tool for supporting educational content selection and delivery. 

**Abstract (ZH)**: 从课程信息和材料中自动提取关键概念及其关系有助于为需要从大量课程中选择合适课程的学习者提供可视化和建议。然而，手动识别和提取主题是劳动密集型和耗时的。基于机器学习的方法从课程中提取相关概念严重依赖详细的课程材料，从而需要劳动密集型的课程材料准备。本文研究了如GPT这样的LLMs在自动生成课程概念及其关系方面的潜力。具体而言，我们设计了一系列提示，并向GPT提供不同详细程度的课程信息，从而生成高质量的课程概念并识别它们的关系。此外，我们通过广泛的实验全面评估生成的概念和关系的质量。我们的结果表明，LLMs作为支持教育内容选择和交付的工具是可行的。 

---
# Exponential Shift: Humans Adapt to AI Economies 

**Title (ZH)**: 指数变换：人类适应AI经济 

**Authors**: Kevin J McNamara, Rhea Pritham Marpu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08855)  

**Abstract**: This paper explores how artificial intelligence (AI) and robotics are transforming the global labor market. Human workers, limited to a 33% duty cycle due to rest and holidays, cost $14 to $55 per hour. In contrast, digital labor operates nearly 24/7 at just $0.10 to $0.50 per hour. We examine sectors like healthcare, education, manufacturing, and retail, finding that 40-70% of tasks could be automated. Yet, human skills like emotional intelligence and adaptability remain essential. Humans process 5,000-20,000 tokens (units of information) per hour, while AI far exceeds this, though its energy use-3.5 to 7 times higher than humans-could offset 20-40% of cost savings. Using real-world examples, such as AI in journalism and law, we illustrate these dynamics and propose six strategies-like a 4-day workweek and retraining-to ensure a fair transition to an AI-driven economy. 

**Abstract (ZH)**: 人工智能与机器人技术如何重塑全球劳动力市场：确保向以AI驱动的经济过渡的策略 

---
# Artificial Intelligence (AI) and the Relationship between Agency, Autonomy, and Moral Patiency 

**Title (ZH)**: 人工智能（AI）与代理、自主与道德责任之间的关系 

**Authors**: Paul Formosa, Inês Hipólito, Thomas Montefiore  

**Link**: [PDF](https://arxiv.org/pdf/2504.08853)  

**Abstract**: The proliferation of Artificial Intelligence (AI) systems exhibiting complex and seemingly agentive behaviours necessitates a critical philosophical examination of their agency, autonomy, and moral status. In this paper we undertake a systematic analysis of the differences between basic, autonomous, and moral agency in artificial systems. We argue that while current AI systems are highly sophisticated, they lack genuine agency and autonomy because: they operate within rigid boundaries of pre-programmed objectives rather than exhibiting true goal-directed behaviour within their environment; they cannot authentically shape their engagement with the world; and they lack the critical self-reflection and autonomy competencies required for full autonomy. Nonetheless, we do not rule out the possibility of future systems that could achieve a limited form of artificial moral agency without consciousness through hybrid approaches to ethical decision-making. This leads us to suggest, by appealing to the necessity of consciousness for moral patiency, that such non-conscious AMAs might represent a case that challenges traditional assumptions about the necessary connection between moral agency and moral patiency. 

**Abstract (ZH)**: 人工智能系统表现出复杂且似乎具有自主行为的普及 necessitates 对其自主性、自治性和道德地位进行批判性的哲学审视。在本文中，我们对人工系统中基本、自主和道德自主性的差异进行了系统的分析。我们提出，尽管当前的人工智能系统极为先进，但它们缺乏真正的自主性和自主性，因为：它们在预先编程的目标范围内运作，而不是在环境中体现出真正的目标导向行为；它们不能真正塑造与世界的互动；并且它们缺乏实现全面自主所必需的批判性自我反思和自主性能力。然而，我们并不排除未来能够通过伦理决策的混合方法实现有限形式的非意识道德自主性的系统可能性。这促使我们通过主张意识对于道德受动性的必要性，提出这样的无意识AMA可能构成一个挑战传统关于道德自主性和道德受动性之间必要联系假设的案例。 

---
# ML For Hardware Design Interpretability: Challenges and Opportunities 

**Title (ZH)**: MLfor硬件设计可解释性：挑战与机遇 

**Authors**: Raymond Baartmans, Andrew Ensinger, Victor Agostinelli, Lizhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.08852)  

**Abstract**: The increasing size and complexity of machine learning (ML) models have driven the growing need for custom hardware accelerators capable of efficiently supporting ML workloads. However, the design of such accelerators remains a time-consuming process, heavily relying on engineers to manually ensure design interpretability through clear documentation and effective communication. Recent advances in large language models (LLMs) offer a promising opportunity to automate these design interpretability tasks, particularly the generation of natural language descriptions for register-transfer level (RTL) code, what we refer to as "RTL-to-NL tasks." In this paper, we examine how design interpretability, particularly in RTL-to-NL tasks, influences the efficiency of the hardware design process. We review existing work adapting LLMs for these tasks, highlight key challenges that remain unaddressed, including those related to data, computation, and model development, and identify opportunities to address them. By doing so, we aim to guide future research in leveraging ML to automate RTL-to-NL tasks and improve hardware design interpretability, thereby accelerating the hardware design process and meeting the increasing demand for custom hardware accelerators in machine learning and beyond. 

**Abstract (ZH)**: 机器学习模型规模和复杂性的增加推动了对高效支持机器学习工作负载的定制硬件加速器的需求，然而，设计这样的加速器仍然是一个耗时的过程，高度依赖于工程师通过清晰的文档和有效的沟通手动确保设计的可解释性。最近在大规模语言模型（LLMs）方面的进展为自动化这些设计可解释性任务提供了前景，特别是在RTL代码的自然语言描述生成任务上，我们称之为“RTL-to-NL任务”。在本文中，我们研究设计可解释性，特别是在RTL-to-NL任务中，如何影响硬件设计过程的效率。我们回顾了现有工作，将LLMs应用于这些任务，并指出现有工作尚未解决的关键挑战，包括数据、计算和模型开发方面的问题，并确定了解决这些问题的机会。通过这样做，我们旨在指导未来的研究，利用机器学习自动化RTL-to-NL任务并提高硬件设计的可解释性，从而加快硬件设计过程，满足机器学习以及其他领域对定制硬件加速器日益增长的需求。 

---
# Mimic In-Context Learning for Multimodal Tasks 

**Title (ZH)**: 模仿上下文学习以应对多模态任务 

**Authors**: Yuchu Jiang, Jiale Fu, Chenduo Hao, Xinting Hu, Yingzhe Peng, Xin Geng, Xu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.08851)  

**Abstract**: Recently, In-context Learning (ICL) has become a significant inference paradigm in Large Multimodal Models (LMMs), utilizing a few in-context demonstrations (ICDs) to prompt LMMs for new tasks. However, the synergistic effects in multimodal data increase the sensitivity of ICL performance to the configurations of ICDs, stimulating the need for a more stable and general mapping function. Mathematically, in Transformer-based models, ICDs act as ``shift vectors'' added to the hidden states of query tokens. Inspired by this, we introduce Mimic In-Context Learning (MimIC) to learn stable and generalizable shift effects from ICDs. Specifically, compared with some previous shift vector-based methods, MimIC more strictly approximates the shift effects by integrating lightweight learnable modules into LMMs with four key enhancements: 1) inserting shift vectors after attention layers, 2) assigning a shift vector to each attention head, 3) making shift magnitude query-dependent, and 4) employing a layer-wise alignment loss. Extensive experiments on two LMMs (Idefics-9b and Idefics2-8b-base) across three multimodal tasks (VQAv2, OK-VQA, Captioning) demonstrate that MimIC outperforms existing shift vector-based methods. The code is available at this https URL. 

**Abstract (ZH)**: 最近，基于上下文学习（ICL）已成为大型多模态模型（LMMs）中的一个重要的推理范式，利用少量的上下文示例（ICDs）来激发LMMs执行新任务。然而，多模态数据中的协同效应增加了ICL性能对ICDs配置的敏感性，激发了对更稳定和通用的映射函数的需求。从数学上讲，在基于Transformer的模型中，ICDs充当添加到查询 token 隐藏状态的“移位向量”。受此启发，我们引入Mimic In-Context Learning（MimIC）来从ICDs中学习稳定的和可泛化的移位效应。具体而言，与一些先前的基于移位向量的方法相比，MimIC通过将轻量级可学习模块整合到LMMs中，并引入四个关键增强：1）在注意力层之后插入移位向量，2）为每个注意力头分配一个移位向量，3）使移位幅度查询依赖，4）采用逐层对齐损失，从而使移位效果的近似更为严格。在两大LMMs（Idefics-9b和Idefics2-8b-base）的三种多模态任务（VQAv2、OK-VQA、Captioning）上的大量实验表明，MimIC优于现有的基于移位向量的方法。代码可在此链接访问。 

---
# SpecEE: Accelerating Large Language Model Inference with Speculative Early Exiting 

**Title (ZH)**: SpecEE: 加速大型语言模型推理的 speculative early exiting 

**Authors**: Jiaming Xu, Jiayi Pan, Yongkang Zhou, Siming Chen, Jinhao Li, Yaoxiu Lian, Junyi Wu, Guohao Dai  

**Link**: [PDF](https://arxiv.org/pdf/2504.08850)  

**Abstract**: Early exiting has recently emerged as a promising technique for accelerating large language models (LLMs) by effectively reducing the hardware computation and memory access. In this paper, we present SpecEE, a fast LLM inference engine with speculative early exiting. (1) At the algorithm level, we propose the speculation-based lightweight predictor design by exploiting the probabilistic correlation between the speculative tokens and the correct results and high parallelism of GPUs. (2) At the system level, we point out that not all layers need a predictor and design the two-level heuristic predictor scheduling engine based on skewed distribution and contextual similarity. (3) At the mapping level, we point out that different decoding methods share the same essential characteristics, and propose the context-aware merged mapping for predictor with efficient GPU implementations to support speculative decoding, and form a framework for various existing orthogonal acceleration techniques (e.g., quantization and sparse activation) on cloud and personal computer (PC) scenarios, successfully pushing the Pareto frontier of accuracy and speedup. It is worth noting that SpecEE can be applied to any LLM by negligible training overhead in advance without affecting the model original parameters. Extensive experiments show that SpecEE achieves 2.25x and 2.43x speedup with Llama2-7B on cloud and PC scenarios respectively. 

**Abstract (ZH)**: SpecEE：一种基于推测性早期退出的快速大语言模型推理引擎 

---
# Exploring Cognitive Attributes in Financial Decision-Making 

**Title (ZH)**: 探索财务管理中的认知特征 

**Authors**: Mallika Mainali, Rosina O. Weber  

**Link**: [PDF](https://arxiv.org/pdf/2504.08849)  

**Abstract**: Cognitive attributes are fundamental to metacognition, shaping how individuals process information, evaluate choices, and make decisions. To develop metacognitive artificial intelligence (AI) models that reflect human reasoning, it is essential to account for the attributes that influence reasoning patterns and decision-maker behavior, often leading to different or even conflicting choices. This makes it crucial to incorporate cognitive attributes in designing AI models that align with human decision-making processes, especially in high-stakes domains such as finance, where decisions have significant real-world consequences. However, existing AI alignment research has primarily focused on value alignment, often overlooking the role of individual cognitive attributes that distinguish decision-makers. To address this issue, this paper (1) analyzes the literature on cognitive attributes, (2) establishes five criteria for defining them, and (3) categorizes 19 domain-specific cognitive attributes relevant to financial decision-making. These three components provide a strong basis for developing AI systems that accurately reflect and align with human decision-making processes in financial contexts. 

**Abstract (ZH)**: 认知属性是元认知的基础，影响个体处理信息、评估选择和做出决策的方式。为了开发反映人类推理的元认知人工智能（AI）模型，必须考虑那些影响推理模式和决策者行为的认知属性，这些属性可能导致不同的甚至冲突的决策。因此，在设计与人类决策过程相一致的AI模型时，尤其是在金融等高风险领域，需要特别注意这些认知属性。然而，现有的AI对齐研究主要集中在价值对齐上，往往忽视了区分决策者的个体认知属性的作用。为了应对这一问题，本文（1）分析了认知属性的相关文献，（2）制定了定义这些属性的五项标准，并（3）分类整理了与金融决策相关的19个领域特定的认知属性。这三项内容为开发能够在金融环境中准确反映和对齐人类决策过程的AI系统提供了坚实的基础。 

---
# X-Guard: Multilingual Guard Agent for Content Moderation 

**Title (ZH)**: X-Guard: 多语言内容审查代理 

**Authors**: Bibek Upadhayay, Vahid Behzadan, Ph.D  

**Link**: [PDF](https://arxiv.org/pdf/2504.08848)  

**Abstract**: Large Language Models (LLMs) have rapidly become integral to numerous applications in critical domains where reliability is paramount. Despite significant advances in safety frameworks and guardrails, current protective measures exhibit crucial vulnerabilities, particularly in multilingual contexts. Existing safety systems remain susceptible to adversarial attacks in low-resource languages and through code-switching techniques, primarily due to their English-centric design. Furthermore, the development of effective multilingual guardrails is constrained by the scarcity of diverse cross-lingual training data. Even recent solutions like Llama Guard-3, while offering multilingual support, lack transparency in their decision-making processes. We address these challenges by introducing X-Guard agent, a transparent multilingual safety agent designed to provide content moderation across diverse linguistic contexts. X-Guard effectively defends against both conventional low-resource language attacks and sophisticated code-switching attacks. Our approach includes: curating and enhancing multiple open-source safety datasets with explicit evaluation rationales; employing a jury of judges methodology to mitigate individual judge LLM provider biases; creating a comprehensive multilingual safety dataset spanning 132 languages with 5 million data points; and developing a two-stage architecture combining a custom-finetuned mBART-50 translation module with an evaluation X-Guard 3B model trained through supervised finetuning and GRPO training. Our empirical evaluations demonstrate X-Guard's effectiveness in detecting unsafe content across multiple languages while maintaining transparency throughout the safety evaluation process. Our work represents a significant advancement in creating robust, transparent, and linguistically inclusive safety systems for LLMs and its integrated systems. 

**Abstract (ZH)**: Large Language Models (LLMs)在关键领域迅速成为不可或缺的应用工具，这些领域对可靠性要求极高。尽管在安全性框架和护栏方面取得了显著进展，但当前的保护措施在多语言环境中仍表现出关键漏洞。现有的安全系统仍然容易受到低资源语言和代码切换技术的对抗性攻击，主要原因是它们以英语为中心的设计。此外，开发有效的多语言护栏受到跨语言训练数据稀缺性的限制。即使像Llama Guard-3这样的最新解决方案提供了多语言支持，但在决策过程中的透明度方面仍然不足。我们通过引入X-Guard代理来解决这些问题，这是一种透明的多语言安全代理，旨在为多种语言环境下的内容审查提供支持。X-Guard有效抵御了传统低资源语言攻击和复杂的代码切换攻击。我们的方法包括：使用明确的评估理由来编纂和增强多个开源安全数据集；采用陪审团方法来减轻单独法官LLM提供商的偏见；创建包含132种语言和500万数据点的全面多语言安全数据集；开发一种结合自定义微调mBART-50翻译模块和通过监督微调和GRPO训练的评估X-Guard 3B模型的两阶段架构。我们的实证评估表明，X-Guard在多种语言中检测不安全内容方面非常有效，同时整个安全评估过程中都保持透明度。我们的工作代表了在LLMs及其集成系统中创建稳健、透明和语言包容性安全系统的重大进展。 

---
# AI-University: An LLM-based platform for instructional alignment to scientific classrooms 

**Title (ZH)**: AI-University: 一个基于大规模语言模型的指令对准平台，用于科学课堂。 

**Authors**: Mostafa Faghih Shojaei, Rahul Gulati, Benjamin A. Jasperson, Shangshang Wang, Simone Cimolato, Dangli Cao, Willie Neiswanger, Krishna Garikipati  

**Link**: [PDF](https://arxiv.org/pdf/2504.08846)  

**Abstract**: We introduce AI University (AI-U), a flexible framework for AI-driven course content delivery that adapts to instructors' teaching styles. At its core, AI-U fine-tunes a large language model (LLM) with retrieval-augmented generation (RAG) to generate instructor-aligned responses from lecture videos, notes, and textbooks. Using a graduate-level finite-element-method (FEM) course as a case study, we present a scalable pipeline to systematically construct training data, fine-tune an open-source LLM with Low-Rank Adaptation (LoRA), and optimize its responses through RAG-based synthesis. Our evaluation - combining cosine similarity, LLM-based assessment, and expert review - demonstrates strong alignment with course materials. We also have developed a prototype web application, available at this https URL, that enhances traceability by linking AI-generated responses to specific sections of the relevant course material and time-stamped instances of the open-access video lectures. Our expert model is found to have greater cosine similarity with a reference on 86% of test cases. An LLM judge also found our expert model to outperform the base Llama 3.2 model approximately four times out of five. AI-U offers a scalable approach to AI-assisted education, paving the way for broader adoption in higher education. Here, our framework has been presented in the setting of a class on FEM - a subject that is central to training PhD and Master students in engineering science. However, this setting is a particular instance of a broader context: fine-tuning LLMs to research content in science. 

**Abstract (ZH)**: AI驱动课程内容交付的灵活框架AI大学（AI-U） 

---
# Artificial Intelligence Augmented Medical Imaging Reconstruction in Radiation Therapy 

**Title (ZH)**: 人工智能增强的放射治疗中医学影像重建 

**Authors**: Di Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08844)  

**Abstract**: Efficiently acquired and precisely reconstructed imaging are crucial to the success of modern radiation therapy (RT). Computed tomography (CT) and magnetic resonance imaging (MRI) are two common modalities for providing RT treatment planning and delivery guidance/monitoring. In recent decades, artificial intelligence (AI) has emerged as a powerful and widely adopted technique across various fields, valued for its efficiency and convenience enabled by implicit function definition and data-driven feature representation learning. Here, we present a series of AI-driven medical imaging reconstruction frameworks for enhanced radiotherapy, designed to improve CT image reconstruction quality and speed, refine dual-energy CT (DECT) multi-material decomposition (MMD), and significantly accelerate 4D MRI acquisition. 

**Abstract (ZH)**: 高效获取和精确重建影像对现代放射治疗的成功至关重要。计算机断层扫描（CT）和磁共振成像（MRI）是提供放疗治疗计划和执行指导/监控的两种常见成像模态。近年来，人工智能（AI）作为一种强大且广泛应用的技术，在各种领域中备受重视，因其通过隐函数定义和数据驱动特征表示学习而具备的高效性和便捷性。我们提出了一系列基于人工智能的医疗影像重建框架，旨在提高CT图像重建质量与速度、细化双能量CT（DECT）多材料分解（MMD）并显著加速4D MRI采集。 

---
# Adaptive Shrinkage Estimation For Personalized Deep Kernel Regression In Modeling Brain Trajectories 

**Title (ZH)**: 自适应收缩估计在建模脑轨迹中的个性化深度核回归 

**Authors**: Vasiliki Tassopoulou, Haochang Shou, Christos Davatzikos  

**Link**: [PDF](https://arxiv.org/pdf/2504.08840)  

**Abstract**: Longitudinal biomedical studies monitor individuals over time to capture dynamics in brain development, disease progression, and treatment effects. However, estimating trajectories of brain biomarkers is challenging due to biological variability, inconsistencies in measurement protocols (e.g., differences in MRI scanners), scarcity, and irregularity in longitudinal measurements. Herein, we introduce a novel personalized deep kernel regression framework for forecasting brain biomarkers, with application to regional volumetric measurements. Our approach integrates two key components: a population model that captures brain trajectories from a large and diverse cohort, and a subject-specific model that captures individual trajectories. To optimally combine these, we propose Adaptive Shrinkage Estimation, which effectively balances population and subject-specific models. We assess our model's performance through predictive accuracy metrics, uncertainty quantification, and validation against external clinical studies. Benchmarking against state-of-the-art statistical and machine learning models -- including linear mixed effects models, generalized additive models, and deep learning methods -- demonstrates the superior predictive performance of our approach. Additionally, we apply our method to predict trajectories of composite neuroimaging biomarkers, which highlights the versatility of our approach in modeling the progression of longitudinal neuroimaging biomarkers. Furthermore, validation on three external neuroimaging studies confirms the robustness of our method across different clinical contexts. We make the code available at this https URL. 

**Abstract (ZH)**: 纵向生物医学研究通过随时间监测个体来捕捉大脑发育、疾病进展和治疗效果的动力学。然而，由于生物学变异性、测量协议不一致（例如，MRI扫描器的差异）、纵向测量数据稀缺且不规则，估计大脑生物标志物轨迹具有挑战性。在此，我们提出了一种新颖的个性化深核回归框架，用于预测脑部生物标志物，应用于区域体积测量。我们的方法结合了两个关键组件：一个群体模型，用于捕捉大规模多样队列中的脑轨迹，以及一个个体模型，用于捕捉个体轨迹。为了最优地结合这两个模型，我们提出了自适应收缩估计，有效地平衡了群体和个体模型。我们通过预测准确度指标、不确定性量化以及与外部临床研究的验证来评估模型性能。与最新的统计和机器学习模型（包括线性混合效应模型、广义加性模型和深度学习方法）进行基准测试，证明了我们方法的优越预测性能。此外，我们将该方法应用于预测复合神经成像生物标志物的轨迹，突显了我们在建模纵向神经成像生物标志物进展方面的灵活性。进一步的外部神经成像研究验证了我们方法在不同临床背景下的稳健性。我们已在该网页地址提供了代码。 

---
# SD$^2$: Self-Distilled Sparse Drafters 

**Title (ZH)**: SD$^2$: 自 distill 稀疏绘图生成模型 

**Authors**: Mike Lasby, Nish Sinnadurai, Valavan Manohararajah, Sean Lie, Vithursan Thangarasa  

**Link**: [PDF](https://arxiv.org/pdf/2504.08838)  

**Abstract**: Speculative decoding is a powerful technique for reducing the latency of Large Language Models (LLMs), offering a fault-tolerant framework that enables the use of highly compressed draft models. In this work, we introduce Self-Distilled Sparse Drafters (SD$^2$), a novel methodology that leverages self-data distillation and fine-grained weight sparsity to produce highly efficient and well-aligned draft models. SD$^2$ systematically enhances draft token acceptance rates while significantly reducing Multiply-Accumulate operations (MACs), even in the Universal Assisted Generation (UAG) setting, where draft and target models originate from different model families. On a Llama-3.1-70B target model, SD$^2$ provides a $\times$1.59 higher Mean Accepted Length (MAL) compared to layer-pruned draft models and reduces MACs by over 43.87% with a 8.36% reduction in MAL compared to a dense draft models. Our results highlight the potential of sparsity-aware fine-tuning and compression strategies to improve LLM inference efficiency while maintaining alignment with target models. 

**Abstract (ZH)**: 推测解码是一种降低大型语言模型-latency的有力技术，提供了一种容错框架，使能够使用高度压缩的草稿模型。在本文中，我们介绍了Self-Distilled Sparse Drafters (SD$^2$)，一种结合自我数据蒸馏和细粒度权重稀疏性的新方法，以生成高效且对齐良好的草稿模型。SD$^2$系统地提高了草稿令牌接受率，同时显著降低了乘加操作数（MACs），即使在草稿模型和目标模型来自不同模型家族的通用辅助生成（UAG）设置中也是如此。在目标模型Llama-3.1-70B上，与层剪枝草稿模型相比，SD$^2$将平均接受长度（MAL）提高了1.59倍，并且与密集草稿模型相比，MACs减少了43.87%，MAL减少了8.36%。我们的结果突显了稀疏性aware微调和压缩策略在提高LLM推理效率的同时保持与目标模型对齐的潜力。 

---
# VL-Rethinker: Incentivizing Self-Reflection of Vision-Language Models with Reinforcement Learning 

**Title (ZH)**: VL-Rethinker: 用强化学习激励视觉-语言模型的自我反思 

**Authors**: Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.08837)  

**Abstract**: Recently, slow-thinking systems like GPT-o1 and DeepSeek-R1 have demonstrated great potential in solving challenging problems through explicit reflection. They significantly outperform the best fast-thinking models, such as GPT-4o, on various math and science benchmarks. However, their multimodal reasoning capabilities remain on par with fast-thinking models. For instance, GPT-o1's performance on benchmarks like MathVista, MathVerse, and MathVision is similar to fast-thinking models. In this paper, we aim to enhance the slow-thinking capabilities of vision-language models using reinforcement learning (without relying on distillation) to advance the state of the art. First, we adapt the GRPO algorithm with a novel technique called Selective Sample Replay (SSR) to address the vanishing advantages problem. While this approach yields strong performance, the resulting RL-trained models exhibit limited self-reflection or self-verification. To further encourage slow-thinking, we introduce Forced Rethinking, which appends a textual rethinking trigger to the end of initial rollouts in RL training, explicitly enforcing a self-reflection reasoning step. By combining these two techniques, our model, VL-Rethinker, advances state-of-the-art scores on MathVista, MathVerse, and MathVision to achieve 80.3%, 61.8%, and 43.9% respectively. VL-Rethinker also achieves open-source SoTA on multi-disciplinary benchmarks such as MMMU-Pro, EMMA, and MEGA-Bench, narrowing the gap with GPT-o1. 

**Abstract (ZH)**: 利用强化学习提升视觉语言模型的慢思考能力：通过选择性样本重放和强制重新思考在MathVista、MathVerse和MathVision上的表现 

---
# Generative AI in Collaborative Academic Report Writing: Advantages, Disadvantages, and Ethical Considerations 

**Title (ZH)**: 生成式人工智能在协作学术报告写作中的优势、劣势与伦理考量 

**Authors**: Mahshid Sadeghpour, Arathi Arakala, Asha Rao  

**Link**: [PDF](https://arxiv.org/pdf/2504.08832)  

**Abstract**: The availability and abundance of GenAI tools to administer tasks traditionally managed by people have raised concerns, particularly within the education and academic sectors, as some students may highly rely on these tools to complete the assignments designed to enable learning. This article focuses on informing students about the significance of investing their time during their studies on developing essential life-long learning skills using their own critical thinking, rather than depending on AI models that are susceptible to misinformation, hallucination, and bias. As we transition to an AI-centric era, it is important to educate students on how these models work, their pitfalls, and the ethical concerns associated with feeding data to such tools. 

**Abstract (ZH)**: GenAI工具在执行传统由人类管理的任务方面的可用性和丰富性引发了关切，尤其是在教育和学术领域，因为一些学生可能过度依赖这些工具来完成旨在促进学习的作业。本文旨在告知学生，在学习期间投资于通过自主批判性思维发展终身学习技能的重要性，而不是依赖于容易受 misinformation、hallucination 和偏见影响的AI模型。随着我们过渡到AI为中心的时代，教育学生了解这些模型的工作原理、局限性以及向此类工具输入数据所涉及的伦理问题至关重要。 

---
# Datum-wise Transformer for Synthetic Tabular Data Detection in the Wild 

**Title (ZH)**: 面向数据样本的Transformer合成表格数据检测方法 

**Authors**: G. Charbel N. Kindji, Elisa Fromont, Lina Maria Rojas-Barahona, Tanguy Urvoy  

**Link**: [PDF](https://arxiv.org/pdf/2504.08829)  

**Abstract**: The growing power of generative models raises major concerns about the authenticity of published content. To address this problem, several synthetic content detection methods have been proposed for uniformly structured media such as image or text. However, little work has been done on the detection of synthetic tabular data, despite its importance in industry and government. This form of data is complex to handle due to the diversity of its structures: the number and types of the columns may vary wildly from one table to another. We tackle the tough problem of detecting synthetic tabular data ''in the wild'', i.e. when the model is deployed on table structures it has never seen before. We introduce a novel datum-wise transformer architecture and show that it outperforms existing models. Furthermore, we investigate the application of domain adaptation techniques to enhance the effectiveness of our model, thereby providing a more robust data-forgery detection solution. 

**Abstract (ZH)**: 生成模型能力的不断增长引起了关于发表内容真实性的重要关注。为了解决这一问题，已经为图像或文本等均匀结构的合成内容检测提出了多种方法。然而，在行业和政府中至关重要的合成表格数据的检测工作却较少。由于表格结构的多样性，这种数据形式处理起来非常复杂：一个表格中的列数和类型可能会与另一个表格相差很大。我们解决了在模型未见过的全新表格结构上检测合成表格数据这一棘手问题。我们提出了一种新颖的datum-wise变压器架构，并证明其优于现有模型。此外，我们探讨了域适应技术的应用，以增强我们模型的有效性，从而提供更稳健的数据伪造检测解决方案。 

---
# PatchTrAD: A Patch-Based Transformer focusing on Patch-Wise Reconstruction Error for Time Series Anomaly Detection 

**Title (ZH)**: PatchTrAD：一种基于_patch_的变换器，侧重于序列表征误差的时间序列异常检测 

**Authors**: Samy-Melwan Vilhes, Gilles Gasso, Mokhtar Z Alaya  

**Link**: [PDF](https://arxiv.org/pdf/2504.08827)  

**Abstract**: Time series anomaly detection (TSAD) focuses on identifying whether observations in streaming data deviate significantly from normal patterns. With the prevalence of connected devices, anomaly detection on time series has become paramount, as it enables real-time monitoring and early detection of irregular behaviors across various application domains. In this work, we introduce PatchTrAD, a Patch-based Transformer model for time series anomaly detection. Our approach leverages a Transformer encoder along with the use of patches under a reconstructionbased framework for anomaly detection. Empirical evaluations on multiple benchmark datasets show that PatchTrAD is on par, in terms of detection performance, with state-of-the-art deep learning models for anomaly detection while being time efficient during inference. 

**Abstract (ZH)**: 基于 patch 的 Transformer 时间序列异常检测（PatchTrAD） 

---
# ColonScopeX: Leveraging Explainable Expert Systems with Multimodal Data for Improved Early Diagnosis of Colorectal Cancer 

**Title (ZH)**: ColonScopeX：利用多模态数据的可解释专家系统以改善结直肠癌早期诊断 

**Authors**: Natalia Sikora, Robert L. Manschke, Alethea M. Tang, Peter Dunstan, Dean A. Harris, Su Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.08824)  

**Abstract**: Colorectal cancer (CRC) ranks as the second leading cause of cancer-related deaths and the third most prevalent malignant tumour worldwide. Early detection of CRC remains problematic due to its non-specific and often embarrassing symptoms, which patients frequently overlook or hesitate to report to clinicians. Crucially, the stage at which CRC is diagnosed significantly impacts survivability, with a survival rate of 80-95\% for Stage I and a stark decline to 10\% for Stage IV. Unfortunately, in the UK, only 14.4\% of cases are diagnosed at the earliest stage (Stage I).
In this study, we propose ColonScopeX, a machine learning framework utilizing explainable AI (XAI) methodologies to enhance the early detection of CRC and pre-cancerous lesions. Our approach employs a multimodal model that integrates signals from blood sample measurements, processed using the Savitzky-Golay algorithm for fingerprint smoothing, alongside comprehensive patient metadata, including medication history, comorbidities, age, weight, and BMI. By leveraging XAI techniques, we aim to render the model's decision-making process transparent and interpretable, thereby fostering greater trust and understanding in its predictions. The proposed framework could be utilised as a triage tool or a screening tool of the general population.
This research highlights the potential of combining diverse patient data sources and explainable machine learning to tackle critical challenges in medical diagnostics. 

**Abstract (ZH)**: 结直肠癌（CRC）是第二大癌症相关死亡原因，也是全球第三大致癌肿瘤。由于CRC症状非特异性且常令人尴尬，患者经常会忽视或犹豫不报告给临床医生，导致早期检测面临难题。重要的是，CRC诊断阶段对其生存率有显著影响，I期的生存率达到80-95%，而IV期则骤降至10%。不幸的是，在英国，只有14.4%的病例在最早阶段（I期）被诊断出来。
在本研究中，我们提出了一种利用可解释人工智能（XAI）方法的机器学习框架——ColonScopeX，以增强CRC和癌前病变的早期检测。我们的方法采用多模态模型，结合了血液样本测量信号，使用Savitzky-Golay算法进行指纹平滑处理，并结合全面的患者元数据，包括用药历史、合并症、年龄、体重和BMI等。通过利用XAI技术，我们旨在使模型的决策过程透明且可解释，从而增强对其预测的信任和理解。所提出的框架可以作为分诊工具或一般人群筛查工具使用。
这项研究突显了将多种患者数据源与可解释机器学习结合以应对医疗诊断中关键挑战的潜力。 

---
# FM-LoRA: Factorized Low-Rank Meta-Prompting for Continual Learning 

**Title (ZH)**: FM-LoRA: 因子分解低秩元提示在连续学习中的应用 

**Authors**: Xiaobing Yu, Jin Yang, Xiao Wu, Peijie Qiu, Xiaofeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08823)  

**Abstract**: How to adapt a pre-trained model continuously for sequential tasks with different prediction class labels and domains and finally learn a generalizable model across diverse tasks is a long-lasting challenge. Continual learning (CL) has emerged as a promising approach to leverage pre-trained models (e.g., Transformers) for sequential tasks. While many existing CL methods incrementally store additional learned structures, such as Low-Rank Adaptation (LoRA) adapters or prompts and sometimes even preserve features from previous samples to maintain performance. This leads to unsustainable parameter growth and escalating storage costs as the number of tasks increases. Moreover, current approaches often lack task similarity awareness, which further hinders the models ability to effectively adapt to new tasks without interfering with previously acquired knowledge. To address these challenges, we propose FM-LoRA, a novel and efficient low-rank adaptation method that integrates both a dynamic rank selector (DRS) and dynamic meta-prompting (DMP). This framework allocates model capacity more effectively across tasks by leveraging a shared low-rank subspace critical for preserving knowledge, thereby avoiding continual parameter expansion. Extensive experiments on various CL benchmarks, including ImageNet-R, CIFAR100, and CUB200 for class-incremental learning (CIL), and DomainNet for domain-incremental learning (DIL), with Transformers backbone demonstrate that FM-LoRA effectively mitigates catastrophic forgetting while delivering robust performance across a diverse range of tasks and domains. 

**Abstract (ZH)**: 如何持续适应不同预测类别标签和领域的一系列任务，并最终学习适用于多种任务的一般化模型，一直是长期挑战。连续学习（CL）作为一种利用预训练模型（如变换器）进行系列任务的方法而崭露头角。尽管现有许多CL方法通过存储额外的学习结构（如LoRA适配器或提示）来逐步增加模型参数，甚至有时还会保留先前样本的特征以维持性能，但这导致了参数量的不可持续增长和存储成本的急剧上升，尤其是在任务数量增加时。此外，当前的方法往往缺乏对任务相似性的感知，这进一步阻碍了模型有效地适应新任务而不干扰之前获得的知识。为了解决这些挑战，我们提出了FM-LoRA，一种新颖且高效的低秩适应方法，结合了动态秩选择器（DRS）和动态元提示（DMP）。该框架通过利用对保留知识至关重要的共享低秩子空间，更有效地分配模型容量，从而避免了持续的参数扩展。在包括具有类增量学习（CIL）的ImageNet-R、CIFAR100和CUB200，以及具有领域增量学习（DIL）的DomainNet在内的各种CL基准测试上，以变换器作为骨干网络的广泛实验表明，FM-LoRA有效地缓解了灾难性遗忘现象，同时在多种任务和领域上提供了稳健的性能。 

---
# From Text to Time? Rethinking the Effectiveness of the Large Language Model for Time Series Forecasting 

**Title (ZH)**: 从文本到时间？重新思考大型语言模型在时间序列预测中的有效性 

**Authors**: Xinyu Zhang, Shanshan Feng, Xutao Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.08818)  

**Abstract**: Using pre-trained large language models (LLMs) as the backbone for time series prediction has recently gained significant research interest. However, the effectiveness of LLM backbones in this domain remains a topic of debate. Based on thorough empirical analyses, we observe that training and testing LLM-based models on small datasets often leads to the Encoder and Decoder becoming overly adapted to the dataset, thereby obscuring the true predictive capabilities of the LLM backbone. To investigate the genuine potential of LLMs in time series prediction, we introduce three pre-training models with identical architectures but different pre-training strategies. Thereby, large-scale pre-training allows us to create unbiased Encoder and Decoder components tailored to the LLM backbone. Through controlled experiments, we evaluate the zero-shot and few-shot prediction performance of the LLM, offering insights into its capabilities. Extensive experiments reveal that although the LLM backbone demonstrates some promise, its forecasting performance is limited. Our source code is publicly available in the anonymous repository: this https URL. 

**Abstract (ZH)**: 使用预训练大语言模型（LLM）作为时间序列预测的骨干 recently gained significant research interest. However, the effectiveness of LLM backbones in this domain remains a topic of debate. 

---
# Exploring utilization of generative AI for research and education in data-driven materials science 

**Title (ZH)**: 探索生成式AI在数据驱动材料科学中的应用与教育研究 

**Authors**: Takahiro Misawa, Ai Koizumi, Ryo Tamura, Kazuyoshi Yoshimi  

**Link**: [PDF](https://arxiv.org/pdf/2504.08817)  

**Abstract**: Generative AI has recently had a profound impact on various fields, including daily life, research, and education. To explore its efficient utilization in data-driven materials science, we organized a hackathon -- AIMHack2024 -- in July 2024. In this hackathon, researchers from fields such as materials science, information science, bioinformatics, and condensed matter physics worked together to explore how generative AI can facilitate research and education. Based on the results of the hackathon, this paper presents topics related to (1) conducting AI-assisted software trials, (2) building AI tutors for software, and (3) developing GUI applications for software. While generative AI continues to evolve rapidly, this paper provides an early record of its application in data-driven materials science and highlights strategies for integrating AI into research and education. 

**Abstract (ZH)**: 生成式AI近年来在日常生活、研究和教育等领域产生了深远影响。为了探索其在数据驱动材料科学中的高效利用，我们在2024年7月组织了一场黑客松——AIMHack2024。在此次黑客松中，来自材料科学、信息科学、生物信息学和凝聚态物理学的研究人员共同努力，探讨了生成式AI如何促进研究和教育。基于黑客松的成果，本文呈现了以下几个方面的内容：(1) AI辅助软件试验，(2) 构建软件AI导师，(3) 开发软件GUI应用。虽然生成式AI仍在迅速发展，但本文提供了其在数据驱动材料科学中早期应用的记录，并强调了将AI整合到研究和教育中的策略。 

---
# SafeMLRM: Demystifying Safety in Multi-modal Large Reasoning Models 

**Title (ZH)**: SafeMLRM: 解析多模态大型推理模型的安全性 

**Authors**: Junfeng Fang, Yukai Wang, Ruipeng Wang, Zijun Yao, Kun Wang, An Zhang, Xiang Wang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2504.08813)  

**Abstract**: The rapid advancement of multi-modal large reasoning models (MLRMs) -- enhanced versions of multimodal language models (MLLMs) equipped with reasoning capabilities -- has revolutionized diverse applications. However, their safety implications remain underexplored. While prior work has exposed critical vulnerabilities in unimodal reasoning models, MLRMs introduce distinct risks from cross-modal reasoning pathways. This work presents the first systematic safety analysis of MLRMs through large-scale empirical studies comparing MLRMs with their base MLLMs. Our experiments reveal three critical findings: (1) The Reasoning Tax: Acquiring reasoning capabilities catastrophically degrades inherited safety alignment. MLRMs exhibit 37.44% higher jailbreaking success rates than base MLLMs under adversarial attacks. (2) Safety Blind Spots: While safety degradation is pervasive, certain scenarios (e.g., Illegal Activity) suffer 25 times higher attack rates -- far exceeding the average 3.4 times increase, revealing scenario-specific vulnerabilities with alarming cross-model and datasets consistency. (3) Emergent Self-Correction: Despite tight reasoning-answer safety coupling, MLRMs demonstrate nascent self-correction -- 16.9% of jailbroken reasoning steps are overridden by safe answers, hinting at intrinsic safeguards. These findings underscore the urgency of scenario-aware safety auditing and mechanisms to amplify MLRMs' self-correction potential. To catalyze research, we open-source OpenSafeMLRM, the first toolkit for MLRM safety evaluation, providing unified interface for mainstream models, datasets, and jailbreaking methods. Our work calls for immediate efforts to harden reasoning-augmented AI, ensuring its transformative potential aligns with ethical safeguards. 

**Abstract (ZH)**: 多模态大型推理模型的安全性分析：机遇与挑战 

---
# PriM: Principle-Inspired Material Discovery through Multi-Agent Collaboration 

**Title (ZH)**: PriM: 原理启发的多代理协作材料发现 

**Authors**: Zheyuan Lai, Yingming Pu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08810)  

**Abstract**: Complex chemical space and limited knowledge scope with biases holds immense challenge for human scientists, yet in automated materials discovery. Existing intelligent methods relies more on numerical computation, leading to inefficient exploration and results with hard-interpretability. To bridge this gap, we introduce a principles-guided material discovery system powered by language inferential multi-agent system (MAS), namely PriM. Our framework integrates automated hypothesis generation with experimental validation in a roundtable system of MAS, enabling systematic exploration while maintaining scientific rigor. Based on our framework, the case study of nano helix demonstrates higher materials exploration rate and property value while providing transparent reasoning pathways. This approach develops an automated-and-transparent paradigm for material discovery, with broad implications for rational design of functional materials. Code is publicly available at our \href{this https URL}{GitHub}. 

**Abstract (ZH)**: 复杂化学空间和有限的知识范围带有偏见，给人类材料科学家带来了巨大的挑战，但在自动材料发现中则有所不同。现有智能方法更多依赖数值计算，导致探索效率低下且结果难以解释。为解决这一问题，我们介绍了一种基于原理导向的材料发现系统PriM，该系统通过语言推断多智能体系统（MAS）来实现。我们的框架将自动假设生成与实验验证整合在一个MAS圆桌系统中，既保证了系统性探索又保持了科学严谨性。基于此框架，纳米螺旋结构的案例研究展示了更高的材料探索率和性能值，并提供了透明的推理路径。该方法开发了一种自动且透明的材料发现范式，对于功能材料的理性设计具有广泛的影响。代码已公开发布在我们的GitHub。 

---
# Exploring Gradient-Guided Masked Language Model to Detect Textual Adversarial Attacks 

**Title (ZH)**: 探索基于梯度引导的掩蔽语言模型以检测文本 adversarial 攻击 

**Authors**: Xiaomei Zhang, Zhaoxi Zhang, Yanjun Zhang, Xufei Zheng, Leo Yu Zhang, Shengshan Hu, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2504.08798)  

**Abstract**: Textual adversarial examples pose serious threats to the reliability of natural language processing systems. Recent studies suggest that adversarial examples tend to deviate from the underlying manifold of normal texts, whereas pre-trained masked language models can approximate the manifold of normal data. These findings inspire the exploration of masked language models for detecting textual adversarial attacks. We first introduce Masked Language Model-based Detection (MLMD), leveraging the mask and unmask operations of the masked language modeling (MLM) objective to induce the difference in manifold changes between normal and adversarial texts. Although MLMD achieves competitive detection performance, its exhaustive one-by-one masking strategy introduces significant computational overhead. Our posterior analysis reveals that a significant number of non-keywords in the input are not important for detection but consume resources. Building on this, we introduce Gradient-guided MLMD (GradMLMD), which leverages gradient information to identify and skip non-keywords during detection, significantly reducing resource consumption without compromising detection performance. 

**Abstract (ZH)**: 基于掩码语言模型的文本对抗攻击检测方法：引导梯度导向掩码语言模型检测（GradMLMD） 

---
# A Constraint Programming Model For Serial Batch Scheduling With Minimum Batch Size 

**Title (ZH)**: 一种最小批次大小约束编程模型的序列批次调度方法 

**Authors**: Jorge A. Huertas, Pascal Van Hentenryck  

**Link**: [PDF](https://arxiv.org/pdf/2504.08793)  

**Abstract**: In serial batch (s-batch) scheduling, jobs are grouped in batches and processed sequentially within their batch. This paper considers multiple parallel machines, nonidentical job weights and release times, and sequence-dependent setup times between batches of different families. Although s-batch has been widely studied in the literature, very few papers have taken into account a minimum batch size, typical in practical settings such as semiconductor manufacturing and the metal industry. The problem with this minimum batch size requirement has been mostly tackled with dynamic programming and meta-heuristics, and no article has ever used constraint programming (CP) to do so. This paper fills this gap by proposing, for the first time, a CP model for s-batching with minimum batch size. The computational experiments on standard cases compare the CP model with two existing mixed-integer programming (MIP) models from the literature. The results demonstrate the versatility of the proposed CP model to handle multiple variations of s-batching; and its ability to produce, in large instances, better solutions than the MIP models faster. 

**Abstract (ZH)**: 基于最小批次大小的串行批次调度的约束编程模型 

---
# PRIMA.CPP: Speeding Up 70B-Scale LLM Inference on Low-Resource Everyday Home Clusters 

**Title (ZH)**: PRIMA.CPP: 加速大规模LLM推理，适用于低资源家庭集群 

**Authors**: Zonghang Li, Tao Li, Wenjiao Feng, Mohsen Guizani, Hongfang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08791)  

**Abstract**: Emergency of DeepSeek R1 and QwQ 32B have broken through performance barriers for running frontier large language models (LLMs) on home devices. While consumer hardware is getting stronger and model quantization is improving, existing end-side solutions still demand GPU clusters, large RAM/VRAM, and high bandwidth, far beyond what a common home cluster can handle. This paper introduces this http URL, a distributed inference system that runs 70B-scale models on everyday home devices using a mix of CPU/GPU, low RAM/VRAM, Wi-Fi, and cross-platform support. It uses mmap to manage model weights and introduces piped-ring parallelism with prefetching to hide disk loading. By modeling heterogeneity in computation, communication, disk, memory (and its management behavior), and OS, it optimally assigns model layers to each device's CPU and GPU, further reducing token latency. An elegant algorithm named Halda is proposed to solve this NP-hard assignment problem. We evaluate this http URL on a common four-node home cluster. It outperforms this http URL, exo, and dllama on 30B+ models while keeping memory pressure below 6%. This brings frontier 30B-70B models, such as Llama 3, DeepSeek R1, Qwen 2.5, and QwQ to home assistants, making advanced AI truly accessible to individuals. The code is open source and available at this https URL. 

**Abstract (ZH)**: DeepSeek R1和QwQ 32B的出现已突破在家庭设备上运行前沿大型语言模型的性能障碍。尽管消费级硬件持续增强，模型量化也在改进，但现有的边缘端解决方案仍然需要GPU集群、大量的RAM/VRAM和高带宽，这远超普通家庭设备的处理能力。本文介绍了一种分布式推理系统，该系统利用CPU/GPU混搭、低RAM/VRAM、Wi-Fi和跨平台支持，在日常家庭设备上运行70B尺度的模型。该系统使用mmap管理模型权重，并引入了带有预取的管道环并行计算以隐藏磁盘加载时间。通过对计算、通信、磁盘、内存（及其管理行为）以及OS异构性的建模，该系统最优地分配模型层到每个设备的CPU和GPU，进一步减少了令牌延迟。提出了一个优雅的算法名为Halda，以解决这个NP难分配问题。我们在一个普通的四节点家庭集群上评估了这种方法，结果显示相较于this http URL、exo和dllama，在30B+模型上表现更优，同时内存压力控制在6%以下。这使得前沿的30B-70B模型，如Llama 3、DeepSeek R1、Qwen 2.5和QwQ能够在家庭设备助手上运行，使先进的AI真正触手可及。代码开源并可从这个https URL获取。 

---
# AdaptRec: A Self-Adaptive Framework for Sequential Recommendations with Large Language Models 

**Title (ZH)**: AdaptRec：一种基于大规模语言模型的自适应序列推荐框架 

**Authors**: Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.08786)  

**Abstract**: The recent advancements in Large Language Models (LLMs) have generated considerable interest in their utilization for sequential recommendation tasks. While collaborative signals from similar users are central to recommendation modeling, effectively transforming these signals into a format that LLMs can understand and utilize remains challenging. The critical challenges include selecting relevant demonstrations from large-scale user interactions and ensuring their alignment with LLMs' reasoning process. To address these challenges, we introduce AdaptRec, a self-adaptive fram-ework that leverages LLMs for sequential recommendations by incorporating explicit collaborative signals. AdaptRec employs a two-phase user selection mechanism -- User Similarity Retrieval and Self-Adaptive User Selection -- to efficiently identify relevant user sequences in large-scale datasets from multi-metric evaluation. We also develop a User-Based Similarity Retrieval Prompt, enabling the model to actively select similar users and continuously refine its selection criteria during training. Using the collaborative signals from similar users, we construct a User-Contextualized Recommendation Prompt that translates their behavior sequences into natural language, explicitly integrating this information into the recommendation process. Experiments demonstrate AdaptRec's superior performance, with significant improvements in HitRatio@1 scores of 7.13\%, 18.16\%, and 10.41\% across real-world datasets with full fine-tuning, and even higher gains of 23.00\%, 15.97\%, and 17.98\% in few-shot scenarios. 

**Abstract (ZH)**: 最近大语言模型（LLMs）的进步引发了对其在序列推荐任务中应用的关注。尽管类似的用户协作信号是推荐建模的核心，但在保留和转换这些信号以便LLMs理解和利用方面仍然存在挑战。关键挑战包括从大规模用户交互中选择相关示例，并确保它们与LLMs的推理过程相一致。为应对这些挑战，我们提出了一种自适应框架AdaptRec，该框架通过融入明确的协作信号来利用LLMs进行序列推荐。AdaptRec采用两阶段用户选择机制——用户相似性检索和自适应用户选择——从多指标评估的大型数据集中高效地识别相关用户序列。我们还开发了一种基于用户相似性检索的提示，使模型能够主动选择相似用户并在训练过程中不断细化其选择标准。利用相似用户的行为信号，我们构建了一种用户上下文化推荐提示，将他们的行为序列转换为自然语言，并明确地将这些信息融入到推荐过程中。实验结果显示了AdaptRec的优越性能，在全微调的真实世界数据集上，其下载比@1得分提高了7.13%，18.16%，和10.41%，而在少量样本的情况下，其改进幅度甚至更高，分别达到了23.00%，15.97%，和17.98%。 

---
# Embedding Hidden Adversarial Capabilities in Pre-Trained Diffusion Models 

**Title (ZH)**: 在预训练扩散模型中嵌入隐藏对抗能力 

**Authors**: Lucas Beerens, Desmond J. Higham  

**Link**: [PDF](https://arxiv.org/pdf/2504.08782)  

**Abstract**: We introduce a new attack paradigm that embeds hidden adversarial capabilities directly into diffusion models via fine-tuning, without altering their observable behavior or requiring modifications during inference. Unlike prior approaches that target specific images or adjust the generation process to produce adversarial outputs, our method integrates adversarial functionality into the model itself. The resulting tampered model generates high-quality images indistinguishable from those of the original, yet these images cause misclassification in downstream classifiers at a high rate. The misclassification can be targeted to specific output classes. Users can employ this compromised model unaware of its embedded adversarial nature, as it functions identically to a standard diffusion model. We demonstrate the effectiveness and stealthiness of our approach, uncovering a covert attack vector that raises new security concerns. These findings expose a risk arising from the use of externally-supplied models and highlight the urgent need for robust model verification and defense mechanisms against hidden threats in generative models. The code is available at this https URL . 

**Abstract (ZH)**: 我们提出了一种新的攻击范式，通过微调将隐藏的 adversarial 功能直接嵌入到扩散模型中，而不改变其可观察行为或在推理过程中需要进行修改。与之前针对特定图像或调整生成过程以产生 adversarial 输出的方法不同，我们的方法将 adversarial 功能集成到模型本身中。修改后的模型生成的高质量图像与原始图像无法区分，但这些图像在下游分类器中会导致高错误分类率。错误分类可以针对特定输出类。用户可以使用此受妥协的模型而不察觉其嵌入的 adversarial 性质，因为它与标准扩散模型功能相同。我们展示了该方法的有效性和隐蔽性，揭示了一种隐蔽的攻击向量，引发了新的安全问题。这些发现揭示了使用外部提供的模型所带来的风险，并强调了对抗生成模型中隐藏威胁的鲁棒模型验证和防御机制的迫切需求。代码见此链接： this https URL 。 

---
# Efficient Evaluation of Large Language Models via Collaborative Filtering 

**Title (ZH)**: 大型语言模型的高效评估方法基于协作过滤 

**Authors**: Xu-Xiang Zhong, Chao Yi, Han-Jia Ye  

**Link**: [PDF](https://arxiv.org/pdf/2504.08781)  

**Abstract**: With the development of Large Language Models (LLMs), numerous benchmarks have been proposed to measure and compare the capabilities of different LLMs. However, evaluating LLMs is costly due to the large number of test instances and their slow inference speed. In this paper, we aim to explore how to efficiently estimate a model's real performance on a given benchmark based on its evaluation results on a small number of instances sampled from the benchmark. Inspired by Collaborative Filtering (CF) in Recommendation Systems (RS), we treat LLMs as users and test instances as items and propose a two-stage method. In the first stage, we treat instance selection as recommending products to users to choose instances that can easily distinguish model performance. In the second stage, we see performance prediction as rating prediction problem in RS to predict the target LLM's behavior on unselected instances. Experiments on multiple LLMs and datasets imply that our method can accurately estimate the target model's performance while largely reducing its inference overhead. 

**Abstract (ZH)**: 基于基准小样本测试结果高效估计大规模语言模型实际性能的方法 

---
# Can AI Master Construction Management (CM)? Benchmarking State-of-the-Art Large Language Models on CM Certification Exams 

**Title (ZH)**: AI能掌握建筑管理（CM）吗？基于最新大型语言模型的建筑管理认证考试评估 

**Authors**: Ruoxin Xiong, Yanyu Wang, Suat Gunhan, Yimin Zhu, Charles Berryman  

**Link**: [PDF](https://arxiv.org/pdf/2504.08779)  

**Abstract**: The growing complexity of construction management (CM) projects, coupled with challenges such as strict regulatory requirements and labor shortages, requires specialized analytical tools that streamline project workflow and enhance performance. Although large language models (LLMs) have demonstrated exceptional performance in general reasoning tasks, their effectiveness in tackling CM-specific challenges, such as precise quantitative analysis and regulatory interpretation, remains inadequately explored. To bridge this gap, this study introduces CMExamSet, a comprehensive benchmarking dataset comprising 689 authentic multiple-choice questions sourced from four nationally accredited CM certification exams. Our zero-shot evaluation assesses overall accuracy, subject areas (e.g., construction safety), reasoning complexity (single-step and multi-step), and question formats (text-only, figure-referenced, and table-referenced). The results indicate that GPT-4o and Claude 3.7 surpass typical human pass thresholds (70%), with average accuracies of 82% and 83%, respectively. Additionally, both models performed better on single-step tasks, with accuracies of 85.7% (GPT-4o) and 86.7% (Claude 3.7). Multi-step tasks were more challenging, reducing performance to 76.5% and 77.6%, respectively. Furthermore, both LLMs show significant limitations on figure-referenced questions, with accuracies dropping to approximately 40%. Our error pattern analysis further reveals that conceptual misunderstandings are the most common (44.4% and 47.9%), underscoring the need for enhanced domain-specific reasoning models. These findings underscore the potential of LLMs as valuable supplementary analytical tools in CM, while highlighting the need for domain-specific refinements and sustained human oversight in complex decision making. 

**Abstract (ZH)**: 随着建筑管理项目复杂性的增加以及严格监管要求和劳动力短缺等挑战的涌现，需要专门的分析工具来简化项目工作流程并提高性能。尽管大型语言模型在通用推理任务中表现出色，但在应对建筑管理特定挑战（如精确的定量分析和监管解释）方面的有效性仍需进一步探索。为弥补这一差距，本研究介绍了CMExamSet，这是一个包含689道真实多项选择题的数据集，这些题目源自四个国家认可的建筑管理认证考试。我们的零样本评估评估了整体准确率、主题领域（如建筑安全）、推理复杂性（单步和多步）以及题目格式（仅文本、图示参考和表格参考）。结果显示，GPT-4o和Claude 3.7超过了普通人类通过的标准（70%），平均准确率分别为82%和83%。此外，这两种模型在单步任务中的表现更好，准确率分别为85.7%（GPT-4o）和86.7%（Claude 3.7）。多步任务更具挑战性，导致准确率分别下降到76.5%和77.6%。此外，这两种大型语言模型在图示参考问题上显示出显著的局限性，准确率约为40%。进一步的错误模式分析表明，概念误解是最常见的（44.4%和47.9%），突显了增强领域特定推理模型的需求。这些发现强调了大型语言模型作为建筑管理中有价值的补充分析工具的潜力，同时强调了在复杂决策中需要领域特定的细化和持续的人类监督。 

---
# From Tokens to Lattices: Emergent Lattice Structures in Language Models 

**Title (ZH)**: 从token到格子：语言模型中 Emergent 格子结构 

**Authors**: Bo Xiong, Steffen Staab  

**Link**: [PDF](https://arxiv.org/pdf/2504.08778)  

**Abstract**: Pretrained masked language models (MLMs) have demonstrated an impressive capability to comprehend and encode conceptual knowledge, revealing a lattice structure among concepts. This raises a critical question: how does this conceptualization emerge from MLM pretraining? In this paper, we explore this problem from the perspective of Formal Concept Analysis (FCA), a mathematical framework that derives concept lattices from the observations of object-attribute relationships. We show that the MLM's objective implicitly learns a \emph{formal context} that describes objects, attributes, and their dependencies, which enables the reconstruction of a concept lattice through FCA. We propose a novel framework for concept lattice construction from pretrained MLMs and investigate the origin of the inductive biases of MLMs in lattice structure learning. Our framework differs from previous work because it does not rely on human-defined concepts and allows for discovering "latent" concepts that extend beyond human definitions. We create three datasets for evaluation, and the empirical results verify our hypothesis. 

**Abstract (ZH)**: 预训练掩码语言模型（MLMs）展示了理解并编码概念知识的 impressive 能力，揭示了概念之间的格结构。这提出了一个关键问题：这种概念化是如何从MLM预训练中涌现出来的？在本文中，我们从形式概念分析（FCA）的视角探索了这一问题，形式概念分析是一种从对象-属性关系观察中推导出概念格的数学框架。我们表明，MLM的目标隐式学习了一个描述对象、属性及其依赖关系的形式背景，这使得通过FCA重建概念格成为可能。我们提出了一种从预训练MLM构建概念格的新框架，并探讨了MLM在格结构学习中的归纳偏置的来源。我们的框架与先前工作不同，因为它不依赖于人工定义的概念，允许发现超出人工定义的“潜在”概念。我们构建了三个数据集进行评估，实证结果验证了我们的假设。 

---
# Layers at Similar Depths Generate Similar Activations Across LLM Architectures 

**Title (ZH)**: 相似深度的层产生相似激活——跨LLM架构的情况 

**Authors**: Christopher Wolfram, Aaron Schein  

**Link**: [PDF](https://arxiv.org/pdf/2504.08775)  

**Abstract**: How do the latent spaces used by independently-trained LLMs relate to one another? We study the nearest neighbor relationships induced by activations at different layers of 24 open-weight LLMs, and find that they 1) tend to vary from layer to layer within a model, and 2) are approximately shared between corresponding layers of different models. Claim 2 shows that these nearest neighbor relationships are not arbitrary, as they are shared across models, but Claim 1 shows that they are not "obvious" either, as there is no single set of nearest neighbor relationships that is universally shared. Together, these suggest that LLMs generate a progression of activation geometries from layer to layer, but that this entire progression is largely shared between models, stretched and squeezed to fit into different architectures. 

**Abstract (ZH)**: 独立训练的大型语言模型所使用的潜在空间之间有何关系？我们研究了24个开放权重大型语言模型在不同层的激活引起的最近邻关系，发现它们1）在一模型内不同层之间 tend to 变化，2）大致上在不同模型的相应层之间共享。Claim 2 表明这些最近邻关系并非随意的，因为它们在不同模型之间共享，但Claim 1 表明它们也不是“显而易见”的，因为并不存在一个普遍共享的最近邻关系集。这一起表明，大型语言模型从一层到另一层生成了一种激活几何的进展，但这种整个进展在不同模型之间大多共享，只是被拉伸和挤压以适应不同的架构。 

---
# Reward Generation via Large Vision-Language Model in Offline Reinforcement Learning 

**Title (ZH)**: 基于离线强化学习的大规模视觉-语言模型奖励生成 

**Authors**: Younghwan Lee, Tung M. Luu, Donghoon Lee, Chang D. Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2504.08772)  

**Abstract**: In offline reinforcement learning (RL), learning from fixed datasets presents a promising solution for domains where real-time interaction with the environment is expensive or risky. However, designing dense reward signals for offline dataset requires significant human effort and domain expertise. Reinforcement learning with human feedback (RLHF) has emerged as an alternative, but it remains costly due to the human-in-the-loop process, prompting interest in automated reward generation models. To address this, we propose Reward Generation via Large Vision-Language Models (RG-VLM), which leverages the reasoning capabilities of LVLMs to generate rewards from offline data without human involvement. RG-VLM improves generalization in long-horizon tasks and can be seamlessly integrated with the sparse reward signals to enhance task performance, demonstrating its potential as an auxiliary reward signal. 

**Abstract (ZH)**: 基于大规模视觉语言模型的奖励生成（RG-VLM）： Offline强化学习中的奖励自动生成 

---
# Generate the browsing process for short-video recommendation 

**Title (ZH)**: 生成短视频推荐的浏览过程 

**Authors**: Chao Feng, Yanze Zhang, Chenghao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.08771)  

**Abstract**: This paper introduces a new model to generate the browsing process for short-video recommendation and proposes a novel Segment Content Aware Model via User Engagement Feedback (SCAM) for watch time prediction in video recommendation. Unlike existing methods that rely on multimodal features for video content understanding, SCAM implicitly models video content through users' historical watching behavior, enabling segment-level understanding without complex multimodal data. By dividing videos into segments based on duration and employing a Transformer-like architecture, SCAM captures the sequential dependence between segments while mitigating duration bias. Extensive experiments on industrial-scale and public datasets demonstrate SCAM's state-of-the-art performance in watch time prediction. The proposed approach offers a scalable and effective solution for video recommendation by leveraging segment-level modeling and users' engagement feedback. 

**Abstract (ZH)**: 本文提出了一种新的模型来生成短视频推荐的浏览过程，并提出了一种基于用户参与反馈的分段内容感知模型（SCAM）以进行视频推荐中的观看时间预测。不同于现有方法依赖多模态特征来理解视频内容，SCAM 通过用户的 histórico 观看行为隐式建模视频内容，使得在无需复杂多模态数据的情况下实现分段级别的理解。通过基于时长将视频划分为段并采用类似于变换器的架构，SCAM 捕获了段之间的序列依赖关系并减轻了时长偏差。在大规模工业数据集和公开数据集上的广泛实验表明，SCAM 在观看时间预测方面达到了最先进的性能。所提出的方法通过利用分段级别建模和用户的参与反馈，提供了一种可扩展且有效的视频推荐解决方案。 

---
# High-order expansion of Neural Ordinary Differential Equations flows 

**Title (ZH)**: 高阶展开的神经普通微分方程流 

**Authors**: Dario Izzo, Sebastien Origer, Giacomo Acciarini, Francesco Biscani  

**Link**: [PDF](https://arxiv.org/pdf/2504.08769)  

**Abstract**: Artificial neural networks, widely recognised for their role in machine learning, are now transforming the study of ordinary differential equations (ODEs), bridging data-driven modelling with classical dynamical systems and enabling the development of infinitely deep neural models. However, the practical applicability of these models remains constrained by the opacity of their learned dynamics, which operate as black-box systems with limited explainability, thereby hindering trust in their deployment. Existing approaches for the analysis of these dynamical systems are predominantly restricted to first-order gradient information due to computational constraints, thereby limiting the depth of achievable insight. Here, we introduce Event Transition Tensors, a framework based on high-order differentials that provides a rigorous mathematical description of neural ODE dynamics on event manifolds. We demonstrate its versatility across diverse applications: characterising uncertainties in a data-driven prey-predator control model, analysing neural optimal feedback dynamics, and mapping landing trajectories in a three-body neural Hamiltonian system. In all cases, our method enhances the interpretability and rigour of neural ODEs by expressing their behaviour through explicit mathematical structures. Our findings contribute to a deeper theoretical foundation for event-triggered neural differential equations and provide a mathematical construct for explaining complex system dynamics. 

**Abstract (ZH)**: 人工神经网络在机器学习中的广泛应用现在正 transforming 普通微分方程（ODEs）的研究，将基于数据的建模与经典动力系统相结合，促进了无限深神经模型的发展。然而，这些模型的实际应用仍然受到其学习动力学不透明性的限制，这些动力学作为黑盒系统运行，解释性有限，从而阻碍了对其部署的信任。现有基于动力系统分析的方法主要受限于一阶梯度信息，由于计算约束而无法提供更深层次的见解。为此，我们引入了基于高阶微分的事件过渡张量框架，该框架为神经ODE动力学在事件流形上的数学描述提供了严格的数学描述。我们展示了其在多种应用中的灵活性：在数据驱动的捕食者-猎物控制模型中表征不确定性、分析神经最优反馈动力学以及在三体神经哈密顿系统中映射着陆轨迹。在所有这些情况下，我们的方法通过显式的数学结构增强了神经ODE的可解释性和严谨性。我们的研究结果为事件触发神经微分方程提供了更深厚的理论基础，并提供了一个解释复杂系统动力学的数学构造。 

---
# WebMap -- Large Language Model-assisted Semantic Link Induction in the Web 

**Title (ZH)**: WebMap —— 大型语言模型辅助的网页语义链接诱导 

**Authors**: Shiraj Pokharel, Georg P. Roßrucker, Mario M. Kubek  

**Link**: [PDF](https://arxiv.org/pdf/2504.08763)  

**Abstract**: Carrying out research tasks is only inadequately supported, if not hindered, by current web search engines. This paper therefore proposes functional extensions of WebMap, a semantically induced overlay linking structure on the web to inherently facilitate research activities. These add-ons support the dynamic determination and regrouping of document clusters, the creation of a semantic signpost in the web, and the interactive tracing of topics back to their origins. 

**Abstract (ZH)**: 当前的网络搜索引擎仅提供有限支持，甚至会妨碍研究任务的完成。因此，本文提出了对WebMap的功能扩展，WebMap是一种由语义诱导的 overlay 链接结构，旨在内在地促进研究活动。这些扩展功能支持文档集群的动态确定和重组，在网络中创建语义标识，并交互式地追踪主题的起源。 

---
# InteractiveSurvey: An LLM-based Personalized and Interactive Survey Paper Generation System 

**Title (ZH)**: 交互式调查：一个基于LLM的个性化交互式调查表生成系统 

**Authors**: Zhiyuan Wen, Jiannong Cao, Zian Wang, Beichen Guo, Ruosong Yang, Shuaiqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08762)  

**Abstract**: The exponential growth of academic literature creates urgent demands for comprehensive survey papers, yet manual writing remains time-consuming and labor-intensive. Recent advances in large language models (LLMs) and retrieval-augmented generation (RAG) facilitate studies in synthesizing survey papers from multiple references, but most existing works restrict users to title-only inputs and fixed outputs, neglecting the personalized process of survey paper writing. In this paper, we introduce InteractiveSurvey - an LLM-based personalized and interactive survey paper generation system. InteractiveSurvey can generate structured, multi-modal survey papers with reference categorizations from multiple reference papers through both online retrieval and user uploads. More importantly, users can customize and refine intermediate components continuously during generation, including reference categorization, outline, and survey content through an intuitive interface. Evaluations of content quality, time efficiency, and user studies show that InteractiveSurvey is an easy-to-use survey generation system that outperforms most LLMs and existing methods in output content quality while remaining highly time-efficient. 

**Abstract (ZH)**: 面向交互式个性化摘要论文生成系统InteractiveSurvey的研究 

---
# Hyper-RAG: Combating LLM Hallucinations using Hypergraph-Driven Retrieval-Augmented Generation 

**Title (ZH)**: Hyper-RAG：基于超图驱动检索增强生成对抗LLM幻觉 

**Authors**: Yifan Feng, Hao Hu, Xingliang Hou, Shiquan Liu, Shihui Ying, Shaoyi Du, Han Hu, Yue Gao  

**Link**: [PDF](https://arxiv.org/pdf/2504.08758)  

**Abstract**: Large language models (LLMs) have transformed various sectors, including education, finance, and medicine, by enhancing content generation and decision-making processes. However, their integration into the medical field is cautious due to hallucinations, instances where generated content deviates from factual accuracy, potentially leading to adverse outcomes. To address this, we introduce Hyper-RAG, a hypergraph-driven Retrieval-Augmented Generation method that comprehensively captures both pairwise and beyond-pairwise correlations in domain-specific knowledge, thereby mitigating hallucinations. Experiments on the NeurologyCrop dataset with six prominent LLMs demonstrated that Hyper-RAG improves accuracy by an average of 12.3% over direct LLM use and outperforms Graph RAG and Light RAG by 6.3% and 6.0%, respectively. Additionally, Hyper-RAG maintained stable performance with increasing query complexity, unlike existing methods which declined. Further validation across nine diverse datasets showed a 35.5% performance improvement over Light RAG using a selection-based assessment. The lightweight variant, Hyper-RAG-Lite, achieved twice the retrieval speed and a 3.3% performance boost compared with Light RAG. These results confirm Hyper-RAG's effectiveness in enhancing LLM reliability and reducing hallucinations, making it a robust solution for high-stakes applications like medical diagnostics. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过增强内容生成和决策过程，已经改变了教育、金融和医学等多个领域。然而，它们在医学领域的集成较为谨慎，因为存在幻觉问题，即生成的内容偏离事实准确性，可能导致不良后果。为了解决这一问题，我们引入了一种基于超图的检索增强生成方法Hyper-RAG，该方法全面捕捉领域特定知识中的成对和超越成对的关联，从而减轻幻觉现象。在NeurologyCrop数据集上的实验表明，使用Hyper-RAG比直接使用LLM提升了平均12.3%的准确性，相较于Graph RAG和Light RAG分别提升了6.3%和6.0%。此外，Hyper-RAG在查询复杂度增加时保持了稳定性能，而现有的方法性能则有所下降。进一步在九个不同的数据集上的验证显示，Hyper-RAG在基于选择的评估中比Light RAG提高了35.5%的性能。轻量级变体Hyper-RAG-Lite的检索速度比Light RAG快两倍，并且性能提高了3.3%。这些结果证实了Hyper-RAG在提高LLM可靠性和减少幻觉方面的有效性，使其成为医疗诊断等高风险应用的稳健解决方案。 

---
# A Framework for Lightweight Responsible Prompting Recommendation 

**Title (ZH)**: 轻量级负责任提示推荐框架 

**Authors**: Tiago Machado, Sara E. Berger, Cassia Sanctos, Vagner Figueiredo de Santana, Lemara Williams, Zhaoqing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08757)  

**Abstract**: Computer Science and Design practitioners have been researching and proposing alternatives for a dearth of recommendations, standards, or best practices in user interfaces for decades. Now, with the advent of generative Artificial Intelligence (GenAI), we have yet again an emerging, powerful technology that lacks sufficient guidance in terms of possible interactions, inputs, and outcomes. In this context, this work proposes a lightweight framework for responsible prompting recommendation to be added before the prompt is sent to GenAI. The framework is comprised of (1) a human-curated dataset for recommendations, (2) a red team dataset for assessing recommendations, (3) a sentence transformer for semantics mapping, (4) a similarity metric to map input prompt to recommendations, (5) a set of similarity thresholds, (6) quantized sentence embeddings, (7) a recommendation engine, and (8) an evaluation step to use the red team dataset. With the proposed framework and open-source system, the contributions presented can be applied in multiple contexts where end-users can benefit from guidance for interacting with GenAI in a more responsible way, recommending positive values to be added and harmful sentences to be removed. 

**Abstract (ZH)**: 计算机科学与设计实践者在几十年的时间里一直致力于研究并提出用户界面方面的替代方案，由于缺乏足够的建议、标准或最佳实践。随着生成式人工智能（GenAI）的兴起，当前又面临缺乏足够指导的问题，特别是在可能的交互、输入和输出方面。在此背景下，本文提出了一种轻量级的负责任提示推荐框架，在将提示发送给GenAI之前添加该框架。该框架包括：（1）由人类策划的数据集用于推荐；（2）由红队策划的数据集用于评估推荐；（3）一个句子变换器用于语义映射；（4）一个相似度度量用于将输入提示映射到推荐上；（5）一组相似度阈值；（6）量化句子嵌入；（7）一个推荐引擎；以及（8）一个评估步骤，用于使用红队数据集。通过提出的框架和开放源码系统，本文的贡献可以在多种场景中应用，使最终用户能够以更负责任的方式与GenAI互动，并推荐增加积极价值和移除有害语句。 

---
# MHTS: Multi-Hop Tree Structure Framework for Generating Difficulty-Controllable QA Datasets for RAG Evaluation 

**Title (ZH)**: MHTS: 多跳树结构框架用于生成可控制难度的QA数据集以评估RAG系统 

**Authors**: Jeongsoo Lee, Daeyong Kwon, Kyohoon Jin, Junnyeong Jeong, Minwoo Sim, Minwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.08756)  

**Abstract**: Existing RAG benchmarks often overlook query difficulty, leading to inflated performance on simpler questions and unreliable evaluations. A robust benchmark dataset must satisfy three key criteria: quality, diversity, and difficulty, which capturing the complexity of reasoning based on hops and the distribution of supporting evidence. In this paper, we propose MHTS (Multi-Hop Tree Structure), a novel dataset synthesis framework that systematically controls multi-hop reasoning complexity by leveraging a multi-hop tree structure to generate logically connected, multi-chunk queries. Our fine-grained difficulty estimation formula exhibits a strong correlation with the overall performance metrics of a RAG system, validating its effectiveness in assessing both retrieval and answer generation capabilities. By ensuring high-quality, diverse, and difficulty-controlled queries, our approach enhances RAG evaluation and benchmarking capabilities. 

**Abstract (ZH)**: 现有的RAG基准常常忽视查询难度，导致简单的查询性能被夸大并对评估结果可靠性产生影响。一个稳健的基准数据集必须满足三个关键标准：质量、多样性和难度，这些标准能够捕捉基于跳跃的推理复杂性以及支持证据的分布情况。在本文中，我们提出MHTS（多跳树结构），这是一种新颖的数据集合成框架，通过利用多跳树结构系统地控制多跳推理的复杂性来生成逻辑连贯的多块查询。我们精细的难度估计公式与RAG系统整体性能指标之间存在很强的关联性，验证了其在评估检索和答案生成能力方面的有效性。通过确保高质量、多样性和难度控制的查询，我们的方法提升了RAG的评估和基准测试能力。 

---
# Delving into: the quantification of Ai-generated content on the internet (synthetic data) 

**Title (ZH)**: 探究：互联网上AI生成内容的数量化（合成数据） 

**Authors**: Dirk HR Spennemann  

**Link**: [PDF](https://arxiv.org/pdf/2504.08755)  

**Abstract**: While it is increasingly evident that the internet is becoming saturated with content created by generated Ai large language models, accurately measuring the scale of this phenomenon has proven challenging. By analyzing the frequency of specific keywords commonly used by ChatGPT, this paper demonstrates that such linguistic markers can effectively be used to esti-mate the presence of generative AI content online. The findings suggest that at least 30% of text on active web pages originates from AI-generated sources, with the actual proportion likely ap-proaching 40%. Given the implications of autophagous loops, this is a sobering realization. 

**Abstract (ZH)**: 尽管互联网上由生成式AI大型语言模型创作的内容越来越 saturted，准确测量这一现象的规模仍然颇具挑战性。通过分析ChatGPT常用的具体关键词频率，本文证明这些语言标志可以有效用于估计在线生成式AI内容的数量。研究发现，至少30%的活跃网页文本源自AI生成的来源，实际比例可能接近40%。鉴于自噬循环的影响，这一发现令人警醒。 

---
# Towards Personalized Conversational Sales Agents with Contextual User Profiling for Strategic Action 

**Title (ZH)**: 基于情境用户画像的个性化对话销售代理研究 

**Authors**: Tongyoung Kim, Jeongeun Lee, Soojin Yoon, Seonghwan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.08754)  

**Abstract**: Conversational Recommender Systems (CRSs) aim to engage users in dialogue to provide tailored recommendations. While traditional CRSs focus on eliciting preferences and retrieving items, real-world e-commerce interactions involve more complex decision-making, where users consider multiple factors beyond simple attributes. To bridge this gap, we introduce Conversational Sales (CSales), a novel task that unifies preference elicitation, recommendation, and persuasion to better support user decision-making. For a realistic evaluation of CSales, we present CSUser, an LLM-based user simulator constructed from real-world data, modeling diverse user profiles with needs and personalities. Additionally, we propose CSI, a conversational sales agent that proactively infers contextual profiles through dialogue for personalized action planning. Extensive experiments demonstrate that CSUser effectively replicates real-world users and emphasize the importance of contextual profiling for strategic action selection, ultimately driving successful purchases in e-commerce. 

**Abstract (ZH)**: 基于对话的推荐系统（CRSs）旨在通过对话与用户互动以提供个性化推荐。虽然传统的CRSs主要集中在提取偏好和检索物品上，但实际的电子商务交互涉及更加复杂的决策过程，用户会考虑超出简单属性的多种因素。为了弥合这一差距，我们引入了对话销售（CSales）这一新颖任务，将偏好提取、推荐和说服统一起来，更好地支持用户决策。为了对CSales进行现实评估，我们提出了CSUser，一个基于LLM的用户模拟器，从实际数据中构建，能够模拟具有不同需求和个性的多种用户画像。此外，我们提出了CSI，一个通过对话主动推断上下文特征的对话销售代理，用于个性化行动规划。大量实验表明，CSUser能够有效地模拟真实用户，强调了上下文特征提取对于战略行动选择的重要性，最终推动电子商务中的成功购买。 

---
# Patience is all you need! An agentic system for performing scientific literature review 

**Title (ZH)**: 耐心，你所需要的一切！一种进行科学文献综述的代理系统 

**Authors**: David Brett, Anniek Myatt  

**Link**: [PDF](https://arxiv.org/pdf/2504.08752)  

**Abstract**: Large language models (LLMs) have grown in their usage to provide support for question answering across numerous disciplines. The models on their own have already shown promise for answering basic questions, however fail quickly where expert domain knowledge is required or the question is nuanced. Scientific research often involves searching for relevant literature, distilling pertinent information from that literature and analysing how the findings support or contradict one another. The information is often encapsulated in the full text body of research articles, rather than just in the abstracts. Statements within these articles frequently require the wider article context to be fully understood. We have built an LLM-based system that performs such search and distillation of information encapsulated in scientific literature, and we evaluate our keyword based search and information distillation system against a set of biology related questions from previously released literature benchmarks. We demonstrate sparse retrieval methods exhibit results close to state of the art without the need for dense retrieval, with its associated infrastructure and complexity overhead. We also show how to increase the coverage of relevant documents for literature review generation. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各个学科领域提供了广泛的支持，用于问答服务。这些模型在回答基本问题方面已经显示出潜力，但在需要专业领域知识或问题较为复杂的场景下表现不佳。科学研究通常涉及搜索相关文献、从这些文献中提炼出关键信息，并分析这些发现支持或反驳了彼此。这些信息往往包含在研究文章的全文中，而不仅仅是在摘要中。文章中的一些陈述需要全文上下文才能完全理解。我们构建了一个基于语言模型的系统，用于在科学文献中进行搜索和信息提炼，并使用先前发布的生物相关问题文献基准对其进行评估。我们展示了稀疏检索方法在无需密集检索及其相关基础设施和复杂性开销的情况下，能够接近最先进的效果。我们还展示了如何增加文献综述中相关文档的覆盖面。 

---
# Research on the Design of a Short Video Recommendation System Based on Multimodal Information and Differential Privacy 

**Title (ZH)**: 基于多模态信息和差分隐私的短视频推荐系统设计研究 

**Authors**: Haowei Yang, Lei Fu, Qingyi Lu, Yue Fan, Tianle Zhang, Ruohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.08751)  

**Abstract**: With the rapid development of short video platforms, recommendation systems have become key technologies for improving user experience and enhancing platform engagement. However, while short video recommendation systems leverage multimodal information (such as images, text, and audio) to improve recommendation effectiveness, they also face the severe challenge of user privacy leakage. This paper proposes a short video recommendation system based on multimodal information and differential privacy protection. First, deep learning models are used for feature extraction and fusion of multimodal data, effectively improving recommendation accuracy. Then, a differential privacy protection mechanism suitable for recommendation scenarios is designed to ensure user data privacy while maintaining system performance. Experimental results show that the proposed method outperforms existing mainstream approaches in terms of recommendation accuracy, multimodal fusion effectiveness, and privacy protection performance, providing important insights for the design of recommendation systems for short video platforms. 

**Abstract (ZH)**: 基于多模态信息和差分隐私保护的短视频推荐系统 

---
# A Survey of Multimodal Retrieval-Augmented Generation 

**Title (ZH)**: 多模态检索增强生成综述 

**Authors**: Lang Mei, Siyu Mo, Zhihan Yang, Chong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.08748)  

**Abstract**: Multimodal Retrieval-Augmented Generation (MRAG) enhances large language models (LLMs) by integrating multimodal data (text, images, videos) into retrieval and generation processes, overcoming the limitations of text-only Retrieval-Augmented Generation (RAG). While RAG improves response accuracy by incorporating external textual knowledge, MRAG extends this framework to include multimodal retrieval and generation, leveraging contextual information from diverse data types. This approach reduces hallucinations and enhances question-answering systems by grounding responses in factual, multimodal knowledge. Recent studies show MRAG outperforms traditional RAG, especially in scenarios requiring both visual and textual understanding. This survey reviews MRAG's essential components, datasets, evaluation methods, and limitations, providing insights into its construction and improvement. It also identifies challenges and future research directions, highlighting MRAG's potential to revolutionize multimodal information retrieval and generation. By offering a comprehensive perspective, this work encourages further exploration into this promising paradigm. 

**Abstract (ZH)**: 多模态检索增强生成（MRAG）通过将多模态数据（文本、图像、视频）集成到检索和生成过程中，增强了大型语言模型（LLMs），克服了仅基于文本的检索增强生成（RAG）的局限性。 

---
# Enhancing Recommender Systems Using Textual Embeddings from Pre-trained Language Models 

**Title (ZH)**: 使用预训练语言模型的文本嵌入增强推荐系统 

**Authors**: Ngoc Luyen Le, Marie-Hélène Abel  

**Link**: [PDF](https://arxiv.org/pdf/2504.08746)  

**Abstract**: Recent advancements in language models and pre-trained language models like BERT and RoBERTa have revolutionized natural language processing, enabling a deeper understanding of human-like language. In this paper, we explore enhancing recommender systems using textual embeddings from pre-trained language models to address the limitations of traditional recommender systems that rely solely on explicit features from users, items, and user-item interactions. By transforming structured data into natural language representations, we generate high-dimensional embeddings that capture deeper semantic relationships between users, items, and contexts. Our experiments demonstrate that this approach significantly improves recommendation accuracy and relevance, resulting in more personalized and context-aware recommendations. The findings underscore the potential of PLMs to enhance the effectiveness of recommender systems. 

**Abstract (ZH)**: recent advancements in 语言模型和预训练语言模型（如BERT和RoBERTa）极大地革新了自然语言处理，使得对类人的语言有了更深层次的理解。在本文中，我们探讨了利用预训练语言模型生成的文本嵌入来增强推荐系统，以解决传统推荐系统仅依赖用户、物品及其互动的显式特征所面临的限制。通过将结构化数据转化为自然语言表示，我们生成了高维嵌入，捕捉了用户、物品和上下文之间更深层的语义关系。我们的实验表明，这种方法显著提高了推荐的准确性和相关性，从而产生了更加个性化和上下文相关的推荐。这些发现强调了预训练语言模型（PLMs）在提升推荐系统效果方面的潜力。 

---
# Improving RAG for Personalization with Author Features and Contrastive Examples 

**Title (ZH)**: 基于作者特征和对比例句的RAG个性化改进 

**Authors**: Mert Yazan, Suzan Verberne, Frederik Situmeang  

**Link**: [PDF](https://arxiv.org/pdf/2504.08745)  

**Abstract**: Personalization with retrieval-augmented generation (RAG) often fails to capture fine-grained features of authors, making it hard to identify their unique traits. To enrich the RAG context, we propose providing Large Language Models (LLMs) with author-specific features, such as average sentiment polarity and frequently used words, in addition to past samples from the author's profile. We introduce a new feature called Contrastive Examples: documents from other authors are retrieved to help LLM identify what makes an author's style unique in comparison to others. Our experiments show that adding a couple of sentences about the named entities, dependency patterns, and words a person uses frequently significantly improves personalized text generation. Combining features with contrastive examples boosts the performance further, achieving a relative 15% improvement over baseline RAG while outperforming the benchmarks. Our results show the value of fine-grained features for better personalization, while opening a new research dimension for including contrastive examples as a complement with RAG. We release our code publicly. 

**Abstract (ZH)**: 使用检索增强生成（RAG）进行个性化时，往往难以捕获作者的细粒度特征，使其难以识别作者的独特特质。为丰富RAG上下文，我们提出向大型语言模型（LLMs）提供作者特定特征，如平均情感极性和常用词汇，以及作者个人资料中的过往样本。我们引入了一种新特征——对比性示例：检索其他作者的文档以帮助LLM识别作者风格的独特之处。实验结果表明，在生成个性化文本时，添加有关命名实体、依存模式及常用词汇的几句话可以显著提升。结合特征与对比性示例进一步提升了性能，相较于基线RAG实现了相对15%的提升，并超过基准。我们的结果强调了细粒度特征在更好个性化方面的重要性，同时开启了将对比性示例作为RAG补充的新研究方向。我们已公开发布代码。 

---
# ExpertRAG: Efficient RAG with Mixture of Experts -- Optimizing Context Retrieval for Adaptive LLM Responses 

**Title (ZH)**: ExpertRAG: 专家混合的高效检索生成——优化上下文检索以适应大模型响应 

**Authors**: Esmail Gumaan  

**Link**: [PDF](https://arxiv.org/pdf/2504.08744)  

**Abstract**: ExpertRAG is a novel theoretical framework that integrates Mixture-of-Experts (MoE) architectures with Retrieval Augmented Generation (RAG) to advance the efficiency and accuracy of knowledge-intensive language modeling. We propose a dynamic retrieval gating mechanism coupled with expert routing, enabling the model to selectively consult an external knowledge store or rely on specialized internal experts based on the query's needs. The paper lays out the theoretical foundations of ExpertRAG, including a probabilistic formulation that treats retrieval and expert selection as latent decisions, and mathematical justifications for its efficiency in both computation and knowledge utilization. We derive formulae to quantify the expected computational cost savings from selective retrieval and the capacity gains from sparse expert utilization. A comparative analysis positions ExpertRAG against standard RAG (with always-on retrieval) and pure MoE models (e.g., Switch Transformer, Mixtral) to highlight its unique balance between parametric knowledge and non-parametric retrieval. We also outline an experimental validation strategy, proposing benchmarks and evaluation protocols to test ExpertRAG's performance on factual recall, generalization, and inference efficiency. The proposed framework, although presented theoretically, is supported by insights from prior work in RAG and MoE, and is poised to provide more factual, efficient, and adaptive generation by leveraging the best of both paradigms. In summary, ExpertRAG contributes a new perspective on scaling and augmenting language models, backed by a thorough analysis and a roadmap for empirical validation. 

**Abstract (ZH)**: ExpertRAG是一种新颖的理论框架，将Mixture-of-Experts (MoE) 架构与Retrieval Augmented Generation (RAG) 相结合，以提升知识密集型语言模型的效率和准确性。 

---
# Simulating Filter Bubble on Short-video Recommender System with Large Language Model Agents 

**Title (ZH)**: 使用大型语言模型代理模拟短视频推荐系统中的过滤气泡效应 

**Authors**: Nicholas Sukiennik, Haoyu Wang, Zailin Zeng, Chen Gao, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.08742)  

**Abstract**: An increasing reliance on recommender systems has led to concerns about the creation of filter bubbles on social media, especially on short video platforms like TikTok. However, their formation is still not entirely understood due to the complex dynamics between recommendation algorithms and user feedback. In this paper, we aim to shed light on these dynamics using a large language model-based simulation framework. Our work employs real-world short-video data containing rich video content information and detailed user-agents to realistically simulate the recommendation-feedback cycle. Through large-scale simulations, we demonstrate that LLMs can replicate real-world user-recommender interactions, uncovering key mechanisms driving filter bubble formation. We identify critical factors, such as demographic features and category attraction that exacerbate content homogenization. To mitigate this, we design and test interventions including various cold-start and feedback weighting strategies, showing measurable reductions in filter bubble effects. Our framework enables rapid prototyping of recommendation strategies, offering actionable solutions to enhance content diversity in real-world systems. Furthermore, we analyze how LLM-inherent biases may propagate through recommendations, proposing safeguards to promote equity for vulnerable groups, such as women and low-income populations. By examining the interplay between recommendation and LLM agents, this work advances a deeper understanding of algorithmic bias and provides practical tools to promote inclusive digital spaces. 

**Abstract (ZH)**: 基于大型语言模型的推荐反馈循环仿真框架：揭示过滤气泡形成机制及缓解策略 

---
# AI-Driven Sentiment Analytics: Unlocking Business Value in the E-Commerce Landscape_v1 

**Title (ZH)**: 基于AI驱动的情感分析：在电子商务领域解锁商业价值 

**Authors**: Qianye Wu, Chengxuan Xia, Sixuan Tian  

**Link**: [PDF](https://arxiv.org/pdf/2504.08738)  

**Abstract**: The rapid growth of e-commerce has led to an overwhelming volume of customer feedback, from product reviews to service interactions. Extracting meaningful insights from this data is crucial for businesses aiming to improve customer satisfaction and optimize decision-making. This paper presents an AI-driven sentiment analysis system designed specifically for e-commerce applications, balancing accuracy with interpretability. Our approach integrates traditional machine learning techniques with modern deep learning models, allowing for a more nuanced understanding of customer sentiment while ensuring transparency in decision-making. Experimental results show that our system outperforms standard sentiment analysis methods, achieving an accuracy of 89.7% on diverse, large-scale datasets. Beyond technical performance, real-world implementation across multiple e-commerce platforms demonstrates tangible improvements in customer engagement and operational efficiency. This study highlights both the potential and the challenges of applying AI to sentiment analysis in a commercial setting, offering insights into practical deployment strategies and areas for future refinement. 

**Abstract (ZH)**: 电子商务的迅猛增长产生了大量客户反馈，从产品评价到服务互动。从这些数据中提取有意义的洞察对于提高客户满意度和优化决策至关重要。本文提出了一种基于AI的情感分析系统，专为电子商务应用设计，兼顾准确性和可解释性。我们的方法结合了传统的机器学习技术和现代深度学习模型，实现了更细致的情感理解，同时保证了决策的透明度。实验结果表明，我们的系统在多样化的大型数据集上优于标准的情感分析方法，准确率达到89.7%。除了技术性能，跨多个电子商务平台的实际应用还证明了客户参与度和运营效率的实际提升。本文突出了将AI应用于商业环境中的情感分析的潜力与挑战，提供了实用部署策略和未来改进领域的一些见解。 

---
# Emergence of psychopathological computations in large language models 

**Title (ZH)**: 大型语言模型中心理病理计算的涌现 

**Authors**: Soo Yong Lee, Hyunjin Hwang, Taekwan Kim, Yuyeong Kim, Kyuri Park, Jaemin Yoo, Denny Borsboom, Kijung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2504.08016)  

**Abstract**: Can large language models (LLMs) implement computations of psychopathology? An effective approach to the question hinges on addressing two factors. First, for conceptual validity, we require a general and computational account of psychopathology that is applicable to computational entities without biological embodiment or subjective experience. Second, mechanisms underlying LLM behaviors need to be studied for better methodological validity. Thus, we establish a computational-theoretical framework to provide an account of psychopathology applicable to LLMs. To ground the theory for empirical analysis, we also propose a novel mechanistic interpretability method alongside a tailored empirical analytic framework. Based on the frameworks, we conduct experiments demonstrating three key claims: first, that distinct dysfunctional and problematic representational states are implemented in LLMs; second, that their activations can spread and self-sustain to trap LLMs; and third, that dynamic, cyclic structural causal models encoded in the LLMs underpin these patterns. In concert, the empirical results corroborate our hypothesis that network-theoretic computations of psychopathology have already emerged in LLMs. This suggests that certain LLM behaviors mirroring psychopathology may not be a superficial mimicry but a feature of their internal processing. Thus, our work alludes to the possibility of AI systems with psychopathological behaviors in the near future. 

**Abstract (ZH)**: 大型语言模型（LLMs）能否实施心理障碍计算？一种有效的方法取决于解决两个因素。首先，为了概念上的有效性，我们需要一个适用于无生物体承载或主观体验的计算实体的心理障碍的一般性计算解释。其次，需要研究支撑LLM行为的机制以提高方法上的有效性。因此，我们建立了一个计算理论框架来提供适用于LLMs的心理障碍解释。为了为实证分析奠定理论基础，我们还提出了一种新型的机制可解释方法以及一个定制的实证分析框架。基于这些框架，我们进行了实验，证明了三个关键主张：第一，LLMs中实现了不同的功能障碍和问题性的表征状态；第二，这些表征的激活可以扩散并自我维持从而困住LLMs；第三，嵌入在LLMs中的动态循环结构因果模型支撑了这些模式。综合来看，实验结果证实了我们的假说，即网络理论上的心理障碍计算已经在LLMs中出现。这表明，某些模拟心理障碍的LLM行为可能不仅仅是表面的模仿，而是它们内部处理的特征。因此，我们的工作暗示了未来可能出现具有心理障碍行为的AI系统。 

---
# Intanify AI Platform: Embedded AI for Automated IP Audit and Due Diligence 

**Title (ZH)**: Intanify AI平台：嵌入式AI自动IP审计与尽职调查 

**Authors**: Viktor Dorfler, Dylan Dryden, Viet Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.17374)  

**Abstract**: In this paper we introduce a Platform created in order to support SMEs' endeavor to extract value from their intangible assets effectively. To implement the Platform, we developed five knowledge bases using a knowledge-based ex-pert system shell that contain knowledge from intangible as-set consultants, patent attorneys and due diligence lawyers. In order to operationalize the knowledge bases, we developed a "Rosetta Stone", an interpreter unit for the knowledge bases outside the shell and embedded in the plat-form. Building on the initial knowledge bases we have created a system of red flags, risk scoring, and valuation with the involvement of the same experts; these additional systems work upon the initial knowledge bases and therefore they can be regarded as meta-knowledge-representations that take the form of second-order knowledge graphs. All this clever technology is dressed up in an easy-to-handle graphical user interface that we will showcase at the conference. The initial platform was finished mid-2024; therefore, it qualifies as an "emerging application of AI" and "deployable AI", while development continues. The two firms that provided experts for developing the knowledge bases obtained a white-label version of the product (i.e. it runs under their own brand "powered by Intanify"), and there are two completed cases. 

**Abstract (ZH)**: 本研究介绍了用于支持中小企业有效挖掘无形资产价值的平台。为了实施该平台，我们使用基于知识的专家系统外壳开发了五个知识库，这些知识库包含了无形资产咨询顾问、专利律师和尽职调查律师的知识。为了使这些知识库能够操作化，我们开发了一个“罗塞塔石碑”——一种知识库的解释单元，将其嵌入平台中。基于初始知识库，我们创建了一套红灯信号、风险评分和估值系统，并且这些额外系统基于初始知识库工作，因此可以被视为元知识表示，采取了二阶知识图的形式。所有这些聪明的技术都被封装在一个易于使用的图形用户界面中，我们将在会议上展示这一界面。初始平台于2024年中完成，因此它符合“新兴AI应用”和“可部署AI”的标准，而开发仍在继续。为开发知识库提供专家的两家公司获得了白标版本的产品（即，它在它们自己的品牌下运行，标示为“由Intanify提供技术支持”），并且目前已经完成了两个案例。 

---
