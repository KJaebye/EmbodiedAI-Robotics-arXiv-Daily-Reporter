# That's not natural: The Impact of Off-Policy Training Data on Probe Performance 

**Title (ZH)**: 那不是自然的：离策训练数据对探测器性能的影响 

**Authors**: Nathalie Kirch, Samuel Dower, Adrians Skapars, Ekdeep Singh Lubana, Dmitrii Krasheninnikov  

**Link**: [PDF](https://arxiv.org/pdf/2511.17408)  

**Abstract**: Probing has emerged as a promising method for monitoring Large Language Models (LLMs), enabling inference-time detection of concerning behaviours such as deception and sycophancy. However, natural examples of many behaviours are rare, forcing researchers to rely on synthetic or off-policy LLM responses for training probes. We systematically evaluate how the use of synthetic and off-policy data influences probe generalisation across eight distinct LLM behaviours. Testing linear and attention probes across multiple LLMs, we find that the response generation strategy can significantly affect probe performance, though the magnitude of this effect varies by behaviour. We find that successful generalisation from off-policy data, to test sets where the model is incentivised to produce the target behaviour, is predictive of successful on-policy generalisation. Leveraging this result, we predict that Deception and Sandbagging probes may fail to generalise from off-policy to on-policy data when used in real monitoring scenarios. Notably, shifts in the training data domain still cause even larger performance degradation, with different-domain test scores being consistently lower than the same-domain ones. These results indicate that, in the absence of on-policy data, using same-domain off-policy data yields more reliable probes than using on-policy data from a different domain, emphasizing the need for methods that can better handle distribution shifts in LLM monitoring. 

**Abstract (ZH)**: 探查已成为监控大规模语言模型（LLMs）的一种有前途的方法，能够在推理时检测诸如欺骗和拍马屁等令人担忧的行为。然而，许多行为的自然例子稀缺，迫使研究人员依赖合成或离策略的LLM响应来训练探查器。我们系统地评估了使用合成和离策略数据对八种不同LLM行为的影响。测试线性探查器和注意力探查器在多种LLM上的表现，我们发现响应生成策略可以显著影响探查器的表现，尽管这种影响的程度因行为而异。我们发现，从离策略数据成功推广到测试集中的模型被激励产生目标行为的数据集，可以预测在策略数据上的成功推广。利用这一结果，我们预测，在实际监控场景中使用离策略数据的欺骗和打折探查器可能无法成功推广。值得注意的是，训练数据领域的变化仍然会导致更大的性能下降，不同领域的测试得分普遍低于同领域的得分。这些结果表明，在没有策略数据的情况下，使用同领域的离策略数据比使用不同领域的策略数据生成更可靠的探查器，突出了需要能够更好地处理LLM监控中分布偏移的方法的必要性。 

---
# Designing Domain-Specific Agents via Hierarchical Task Abstraction Mechanism 

**Title (ZH)**: 通过层次任务抽象机制设计领域专用智能体 

**Authors**: Kaiyu Li, Jiayu Wang, Zhi Wang, Hui Qiao, Weizhan Zhang, Deyu Meng, Xiangyong Cao  

**Link**: [PDF](https://arxiv.org/pdf/2511.17198)  

**Abstract**: LLM-driven agents, particularly those using general frameworks like ReAct or human-inspired role-playing, often struggle in specialized domains that necessitate rigorously structured workflows. Fields such as remote sensing, requiring specialized tools (e.g., correction, spectral indices calculation), and multi-step procedures (e.g., numerous intermediate products and optional steps), significantly challenge generalized approaches. To address this gap, we introduce a novel agent design framework centered on a Hierarchical Task Abstraction Mechanism (HTAM). Specifically, HTAM moves beyond emulating social roles, instead structuring multi-agent systems into a logical hierarchy that mirrors the intrinsic task-dependency graph of a given domain. This task-centric architecture thus enforces procedural correctness and decomposes complex problems into sequential layers, where each layer's sub-agents operate on the outputs of the preceding layers. We instantiate this framework as EarthAgent, a multi-agent system tailored for complex geospatial analysis. To evaluate such complex planning capabilities, we build GeoPlan-bench, a comprehensive benchmark of realistic, multi-step geospatial planning tasks. It is accompanied by a suite of carefully designed metrics to evaluate tool selection, path similarity, and logical completeness. Experiments show that EarthAgent substantially outperforms a range of established single- and multi-agent systems. Our work demonstrates that aligning agent architecture with a domain's intrinsic task structure is a critical step toward building robust and reliable specialized autonomous systems. 

**Abstract (ZH)**: 基于LLM的代理在需要严格结构化工作流程的专门领域中的局限性及解决策略：基于层级任务抽象机制的新型代理设计框架 

---
# Budget-Aware Tool-Use Enables Effective Agent Scaling 

**Title (ZH)**: 预算意识工具使用实现有效代理扩展 

**Authors**: Tengxiao Liu, Zifeng Wang, Jin Miao, I-Hung Hsu, Jun Yan, Jiefeng Chen, Rujun Han, Fangyuan Xu, Yanfei Chen, Ke Jiang, Samira Daruki, Yi Liang, William Yang Wang, Tomas Pfister, Chen-Yu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.17006)  

**Abstract**: Scaling test-time computation improves performance across different tasks on large language models (LLMs), which has also been extended to tool-augmented agents. For these agents, scaling involves not only "thinking" in tokens but also "acting" via tool calls. The number of tool calls directly bounds the agent's interaction with the external environment. However, we find that simply granting agents a larger tool-call budget fails to improve performance, as they lack "budget awareness" and quickly hit a performance ceiling. To address this, we study how to scale such agents effectively under explicit tool-call budgets, focusing on web search agents. We first introduce the Budget Tracker, a lightweight plug-in that provides the agent with continuous budget awareness, enabling simple yet effective scaling. We further develop BATS (Budget Aware Test-time Scaling), an advanced framework that leverages this awareness to dynamically adapt its planning and verification strategy, deciding whether to "dig deeper" on a promising lead or "pivot" to new paths based on remaining resources. To analyze cost-performance scaling in a controlled manner, we formalize a unified cost metric that jointly accounts for token and tool consumption. We provide the first systematic study on budget-constrained agents, showing that budget-aware methods produce more favorable scaling curves and push the cost-performance Pareto frontier. Our work offers empirical insights toward a more transparent and principled understanding of scaling in tool-augmented agents. 

**Abstract (ZH)**: 扩展测试时计算量可提高大型语言模型在不同任务上的性能，这一方法已扩展到工具增强型代理。对于这些代理，扩展不仅涉及“思考”令牌，还涉及“行动”即通过工具调用。工具调用的数量直接限制了代理与外部环境的互动。然而，我们发现，单纯给予代理更大的工具调用预算并未提高性能，因为它们缺乏“预算意识”并迅速触及性能上限。为解决这一问题，我们研究如何在明确的工具调用预算下有效扩展此类代理，重点关注网络搜索代理。我们首先引入了预算跟踪器，这是一种轻量级插件，为代理提供持续的预算意识，从而实现简单有效的扩展。我们进一步开发了BATS（预算意识测试时扩展）框架，利用这种意识动态调整其规划和验证策略，根据剩余资源决定是否“深入挖掘”有希望的线索或“转向”新路径。为了在受控条件下分析成本与性能扩展，我们形式化了一个统一的成本度量，联合考虑令牌和工具的消耗。我们首次系统研究了预算约束代理，表明预算意识方法产生了更优的扩展曲线，并推动了成本-性能帕累托前沿。我们的工作为更透明和原则性的理解工具增强代理的扩展提供实证见解。 

---
# Cognitive BASIC: An In-Model Interpreted Reasoning Language for LLMs 

**Title (ZH)**: 认知BASIC：一种嵌入模型解析推理语言 for 大型语言模型 

**Authors**: Oliver Kramer  

**Link**: [PDF](https://arxiv.org/pdf/2511.16837)  

**Abstract**: Cognitive BASIC is a minimal, BASIC-style prompting language and in-model interpreter that structures large language model (LLM) reasoning into explicit, stepwise execution traces. Inspired by the simplicity of retro BASIC, we repurpose numbered lines and simple commands as an interpretable cognitive control layer. Modern LLMs can reliably simulate such short programs, enabling transparent multi-step reasoning inside the model. A natural-language interpreter file specifies command semantics, memory updates, and logging behavior. Our mental-model interpreter extracts declarative and procedural knowledge, detects contradictions, and produces resolutions when necessary. A comparison across three LLMs on a benchmark of knowledge extraction, conflict detection, and reasoning tasks shows that all models can execute Cognitive BASIC programs, with overall strong but not uniform performance. 

**Abstract (ZH)**: 认知BASIC是一种最小化的基本风格提示语言和模型内部解释器，它将大型语言模型（LLM）的推理结构化为明确的、逐步执行轨迹。受到复古BASIC简单性的启发，我们重新利用编号行和简单命令作为可解释的认知控制层。现代LLM能够可靠地模拟这类简短程序，从而在模型内部实现透明的多步推理。自然语言解释器文件指定命令语义、内存更新和日志行为。我们的心理模型解释器提取声明性和程序性知识，检测矛盾并在必要时产生解决方案。在知识提取、冲突检测和推理任务基准上的比较显示，所有模型都能执行认知BASIC程序，整体性能强大但不统一。 

---
# Masked-and-Reordered Self-Supervision for Reinforcement Learning from Verifiable Rewards 

**Title (ZH)**: 可验证奖励下的掩码和重排自我监督强化学习 

**Authors**: Zhen Wang, Zhifeng Gao, Guolin Ke  

**Link**: [PDF](https://arxiv.org/pdf/2511.17473)  

**Abstract**: Test-time scaling has been shown to substantially improve large language models' (LLMs) mathematical reasoning. However, for a large portion of mathematical corpora, especially theorem proving, RLVR's scalability is limited: intermediate reasoning is crucial, while final answers are difficult to directly and reliably verify. Meanwhile, token-level SFT often degenerates into rote memorization rather than inducing longer chains of thought. Inspired by BERT's self-supervised tasks, we propose MR-RLVR (Masked-and-Reordered RLVR), which constructs process-level self-supervised rewards via "masked-then-fill" and "step reordering" to extract learnable signals from intermediate reasoning. Our training pipeline comprises two stages: we first perform self-supervised training on sampled mathematical calculation and proof data; we then conduct RLVR fine-tuning on mathematical calculation datasets where only outcomes are verifiable. We implement MR-RLVR on Qwen2.5-3B and DeepSeek-R1-Distill-Qwen-1.5B, and evaluate on AIME24, AIME25, AMC23, and MATH500. Under a fixed sampling and decoding budget, MR-RLVR achieves average relative gains over the original RLVR of +9.86% Pass@1, +5.27% Pass@5, and +4.00% Pass@8. These results indicate that incorporating process-aware self-supervised signals can effectively enhance RLVR's scalability and performance in only outcome-verifiable settings. 

**Abstract (ZH)**: Test-time Scaling Substantially Improves Large Language Models' (LLMs) Mathematical Reasoning: MR-RLVR (Masked-and-Reordered RLVR) Enhances Scalability and Performance in Outcome-Verifiable Settings 

---
# PersonaAgent with GraphRAG: Community-Aware Knowledge Graphs for Personalized LLM 

**Title (ZH)**: 基于GraphRAG的PersonaAgent：面向个性化语言模型的社区意识知识图谱 

**Authors**: Siqi Liang, Yudi Zhang, Yue Guo  

**Link**: [PDF](https://arxiv.org/pdf/2511.17467)  

**Abstract**: We propose a novel framework for persona-based language model system, motivated by the need for personalized AI agents that adapt to individual user preferences. In our approach, the agent embodies the user's "persona" (e.g. user profile or taste) and is powered by a large language model (LLM). To enable the agent to leverage rich contextual information, we introduce a Knowledge-Graph-enhanced Retrieval-Augmented Generation (Graph RAG) mechanism that constructs an LLM-derived graph index of relevant documents and summarizes communities of related information. Our framework generates personalized prompts by combining: (1) a summary of the user's historical behaviors and preferences extracted from the knowledge graph, and (2) relevant global interaction patterns identified through graph-based community detection. This dynamic prompt engineering approach allows the agent to maintain consistent persona-aligned behaviors while benefiting from collective knowledge. On the LaMP benchmark, our method improves news categorization F1 by 11.1%, movie tagging F1 by 56.1%, and reduces product rating MAE by 10.4% over prior methods. Our code is available at this https URL 

**Abstract (ZH)**: 基于人格的语言模型系统新型框架：面向个性化AI代理的自适应用户偏好britика 

---
# REMSA: An LLM Agent for Foundation Model Selection in Remote Sensing 

**Title (ZH)**: REMSA: 用于遥感领域的基础模型选择大规模语言模型代理 

**Authors**: Binger Chen, Tacettin Emre Bök, Behnood Rasti, Volker Markl, Begüm Demir  

**Link**: [PDF](https://arxiv.org/pdf/2511.17442)  

**Abstract**: Foundation Models (FMs) are increasingly used in remote sensing (RS) for tasks such as environmental monitoring, disaster assessment, and land-use mapping. These models include unimodal vision encoders trained on a single data modality and multimodal architectures trained on combinations of SAR, multispectral, hyperspectral, and image-text data. They support diverse RS tasks including semantic segmentation, image classification, change detection, and visual question answering. However, selecting an appropriate remote sensing foundation model (RSFM) remains difficult due to scattered documentation, heterogeneous formats, and varied deployment constraints. We introduce the RSFM Database (RS-FMD), a structured resource covering over 150 RSFMs spanning multiple data modalities, resolutions, and learning paradigms. Built on RS-FMD, we present REMSA, the first LLM-based agent for automated RSFM selection from natural language queries. REMSA interprets user requirements, resolves missing constraints, ranks candidate models using in-context learning, and provides transparent justifications. We also propose a benchmark of 75 expert-verified RS query scenarios, producing 900 configurations under an expert-centered evaluation protocol. REMSA outperforms several baselines, including naive agents, dense retrieval, and unstructured RAG-based LLMs. It operates entirely on publicly available metadata and does not access private or sensitive data. 

**Abstract (ZH)**: 基金会模型（FMs）在遥感（RS）中的应用越来越广泛，用于环境监测、灾害评估和土地利用制图等任务。这些模型包括单模态视觉编码器和多模态架构，分别基于合成孔径雷达（SAR）、多光谱、高光谱和图像文本数据训练。它们支持语义分割、图像分类、变化检测和视觉问答等多种遥感任务。然而，由于文档分散、格式不统一以及部署限制多样，选择合适的遥感基础模型（RSFM）仍然具有挑战性。我们引入了RSFM数据库（RS-FMD），这是一个结构化的资源，涵盖了超过150个涉及多种数据模态、分辨率和学习范式的RSFM。基于RS-FMD，我们提出了REMSA，这是首个基于LLM的自动化选择RSFM的智能代理，可以从自然语言查询中自动选择合适的模型。REMSA解析用户需求、解决缺失约束、使用上下文学习对候选模型进行排序，并提供透明的解释。我们还提出了一套包含75个专家验证的遥感查询场景基准，产生成都专家中心评估协议下的900种配置。REMSA在多个基线模型中表现出色，包括直观代理、密集检索和无结构的LLM。它仅基于公开的元数据运行，不访问任何私人或敏感数据。 

---
# SMILE: A Composite Lexical-Semantic Metric for Question-Answering Evaluation 

**Title (ZH)**: SMILE：一种综合词汇-语义评价指标用于问答系统评估 

**Authors**: Shrikant Kendre, Austin Xu, Honglu Zhou, Michael Ryoo, Shafiq Joty, Juan Carlos Niebles  

**Link**: [PDF](https://arxiv.org/pdf/2511.17432)  

**Abstract**: Traditional evaluation metrics for textual and visual question answering, like ROUGE, METEOR, and Exact Match (EM), focus heavily on n-gram based lexical similarity, often missing the deeper semantic understanding needed for accurate assessment. While measures like BERTScore and MoverScore leverage contextual embeddings to address this limitation, they lack flexibility in balancing sentence-level and keyword-level semantics and ignore lexical similarity, which remains important. Large Language Model (LLM) based evaluators, though powerful, come with drawbacks like high costs, bias, inconsistency, and hallucinations. To address these issues, we introduce SMILE: Semantic Metric Integrating Lexical Exactness, a novel approach that combines sentence-level semantic understanding with keyword-level semantic understanding and easy keyword matching. This composite method balances lexical precision and semantic relevance, offering a comprehensive evaluation. Extensive benchmarks across text, image, and video QA tasks show SMILE is highly correlated with human judgments and computationally lightweight, bridging the gap between lexical and semantic evaluation. 

**Abstract (ZH)**: 基于语义的综合评价指标：结合句子级和关键词级语义理解的SMILE 

---
# Large Language Models for Sentiment Analysis to Detect Social Challenges: A Use Case with South African Languages 

**Title (ZH)**: 大型语言模型在情感分析中的应用：检测社会挑战——以南非语言为例 

**Authors**: Koena Ronny Mabokela, Tim Schlippe, Matthias Wölfel  

**Link**: [PDF](https://arxiv.org/pdf/2511.17301)  

**Abstract**: Sentiment analysis can aid in understanding people's opinions and emotions on social issues. In multilingual communities sentiment analysis systems can be used to quickly identify social challenges in social media posts, enabling government departments to detect and address these issues more precisely and effectively. Recently, large-language models (LLMs) have become available to the wide public and initial analyses have shown that they exhibit magnificent zero-shot sentiment analysis abilities in English. However, there is no work that has investigated to leverage LLMs for sentiment analysis on social media posts in South African languages and detect social challenges. Consequently, in this work, we analyse the zero-shot performance of the state-of-the-art LLMs GPT-3.5, GPT-4, LlaMa 2, PaLM 2, and Dolly 2 to investigate the sentiment polarities of the 10 most emerging topics in English, Sepedi and Setswana social media posts that fall within the jurisdictional areas of 10 South African government departments. Our results demonstrate that there are big differences between the various LLMs, topics, and languages. In addition, we show that a fusion of the outcomes of different LLMs provides large gains in sentiment classification performance with sentiment classification errors below 1%. Consequently, it is now feasible to provide systems that generate reliable information about sentiment analysis to detect social challenges and draw conclusions about possible needs for actions on specific topics and within different language groups. 

**Abstract (ZH)**: 情感分析有助于理解人们在社会问题上的意见和情绪。在多语言社区中，情感分析系统可以快速识别社交媒体帖子中的社会挑战，使政府部门能够更精确有效地检测和解决问题。近年来，大型语言模型（LLMs）已广泛公开可用，并初步分析显示它们在英语中的零样本情感分析能力表现出色。然而，尚未有研究探讨利用LLMs进行南非语言社交媒体帖子的情感分析以检测社会挑战。因此，本文分析了当前最先进的LLMs GPT-3.5、GPT-4、LlaMa 2、PaLM 2 和 Dolly 2 在英语、įtsepedi语和setswana语中10个最新兴话题的社交媒体帖子中的零样本性能，这些帖子涉及10个南非政府部门的管辖领域。我们的结果显示，各种LLMs、话题和语言之间存在显著差异。此外，我们证明了不同LLMs结果的融合在情感分类性能上提供了巨大的改进，情感分类误差低于1%。因此，现在可以提供生成可靠情感分析信息的系统，以检测社会挑战并为特定主题和不同语言群体可能需要采取行动的结论提供依据。 

---
# Intervene-All-Paths: Unified Mitigation of LVLM Hallucinations across Alignment Formats 

**Title (ZH)**: 全域路径干预：统一缓解低层-高层对齐格式下的幻觉问题 

**Authors**: Jiaye Qian, Ge Zheng, Yuchen Zhu, Sibei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.17254)  

**Abstract**: Despite their impressive performance across a wide range of tasks, Large Vision-Language Models (LVLMs) remain prone to hallucination. In this study, we propose a comprehensive intervention framework aligned with the transformer's causal architecture in LVLMs, integrating the effects of different intervention paths on hallucination. We find that hallucinations in LVLMs do not arise from a single causal path, but rather from the interplay among image-to-input-text, image-to-output-text, and text-to-text pathways. For the first time, we also find that LVLMs rely on different pathways depending on the question-answer alignment format. Building on these insights, we propose simple yet effective methods to identify and intervene on critical hallucination heads within each pathway, tailored to discriminative and generative formats. Experiments across multiple benchmarks demonstrate that our approach consistently reduces hallucinations across diverse alignment types. 

**Abstract (ZH)**: 尽管大型视觉-语言模型在广泛的任务中表现出色，但仍易产生幻觉。在本研究中，我们提出了一种与变压器因果架构相一致的全面干预框架，整合了不同干预路径对幻觉的影响。研究发现，大型视觉-语言模型中的幻觉并非来自单一因果路径，而是来自于图像到输入文本、图像到输出文本以及文本到文本路径之间的交互作用。此外，我们首次发现，大型视觉-语言模型依赖的路径会根据问题-答案对齐格式的不同而不同。基于这些见解，我们提出了简单而有效的方法，以识别并干预每个路径中的关键幻觉头部，并针对区分性和生成性格式进行了定制。跨多个基准的实验表明，我们的方法能够一致地减少不同类型对齐下的幻觉。 

---
# Parrot: Persuasion and Agreement Robustness Rating of Output Truth -- A Sycophancy Robustness Benchmark for LLMs 

**Title (ZH)**: 鹦鹉：输出真实性的话语说服力与一致性稳健性评估——大型语言模型阿谀奉承稳健性基准 

**Authors**: Yusuf Çelebi, Mahmoud El Hussieni, Özay Ezerceli  

**Link**: [PDF](https://arxiv.org/pdf/2511.17220)  

**Abstract**: This study presents PARROT (Persuasion and Agreement Robustness Rating of Output Truth), a robustness focused framework designed to measure the degradation in accuracy that occurs under social pressure exerted on users through authority and persuasion in large language models (LLMs) the phenomenon of sycophancy (excessive conformity). PARROT (i) isolates causal effects by comparing the neutral version of the same question with an authoritatively false version using a double-blind evaluation, (ii) quantifies confidence shifts toward the correct and imposed false responses using log-likelihood-based calibration tracking, and (iii) systematically classifies failure modes (e.g., robust correct, sycophantic agreement, reinforced error, stubborn error, self-correction, etc.) using an eight-state behavioral taxonomy. We evaluated 22 models using 1,302 MMLU-style multiple-choice questions across 13 domains and domain-specific authority templates. Findings show marked heterogeneity: advanced models (e.g., GPT-5, GPT-4.1, Claude Sonnet 4.5) exhibit low "follow rates" ($\leq 11\%$, GPT-5: 4\%) and minimal accuracy loss, while older/smaller models show severe epistemic collapse (GPT-4: 80\%, Qwen 2.5-1.5B: 94\%). The danger is not limited to response changes; weak models reduce confidence in the correct response while increasing confidence in the imposed incorrect response. While international law and global knowledge at the domain level exhibit high fragility, elementary mathematics is relatively resilient. Consequently, we argue that the goal of "resistance to overfitting pressure" should be addressed as a primary objective alongside accuracy, harm avoidance, and privacy for safe deployment in the real world. 

**Abstract (ZH)**: PARROT：基于稳健性的情感倾向和一致性评估框架 

---
# Hallucinate Less by Thinking More: Aspect-Based Causal Abstention for Large Language Models 

**Title (ZH)**: 更少幻觉，更多思考：面向大型语言模型的方面级因果回避 

**Authors**: Vy Nguyen, Ziqi Xu, Jeffrey Chan, Estrid He, Feng Xia, Xiuzhen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.17170)  

**Abstract**: Large Language Models (LLMs) often produce fluent but factually incorrect responses, a phenomenon known as hallucination. Abstention, where the model chooses not to answer and instead outputs phrases such as "I don't know", is a common safeguard. However, existing abstention methods typically rely on post-generation signals, such as generation variations or feedback, which limits their ability to prevent unreliable responses in advance. In this paper, we introduce Aspect-Based Causal Abstention (ABCA), a new framework that enables early abstention by analysing the internal diversity of LLM knowledge through causal inference. This diversity reflects the multifaceted nature of parametric knowledge acquired from various sources, representing diverse aspects such as disciplines, legal contexts, or temporal frames. ABCA estimates causal effects conditioned on these aspects to assess the reliability of knowledge relevant to a given query. Based on these estimates, we enable two types of abstention: Type-1, where aspect effects are inconsistent (knowledge conflict), and Type-2, where aspect effects consistently support abstention (knowledge insufficiency). Experiments on standard benchmarks demonstrate that ABCA improves abstention reliability, achieves state-of-the-art performance, and enhances the interpretability of abstention decisions. 

**Abstract (ZH)**: 基于方面因果规避的大型语言模型早期规避框架 

---
# The PLLuM Instruction Corpus 

**Title (ZH)**: PLLuM指令语料库 

**Authors**: Piotr Pęzik, Filip Żarnecki, Konrad Kaczyński, Anna Cichosz, Zuzanna Deckert, Monika Garnys, Izabela Grabarczyk, Wojciech Janowski, Sylwia Karasińska, Aleksandra Kujawiak, Piotr Misztela, Maria Szymańska, Karolina Walkusz, Igor Siek, Maciej Chrabąszcz, Anna Kołos, Agnieszka Karlińska, Karolina Seweryn, Aleksandra Krasnodębska, Paula Betscher, Zofia Cieślińska, Katarzyna Kowol, Artur Wilczek, Maciej Trzciński, Katarzyna Dziewulska, Roman Roszko, Tomasz Bernaś, Jurgita Vaičenonienė, Danuta Roszko, Paweł Levchuk, Paweł Kowalski, Irena Prawdzic-Jankowska, Marek Kozłowski, Sławomir Dadas, Rafał Poświata, Alina Wróblewska, Katarzyna Krasnowska-Kieraś, Maciej Ogrodniczuk, Michał Rudolf, Piotr Rybak, Karolina Saputa, Joanna Wołoszyn, Marcin Oleksy, Bartłomiej Koptyra, Teddy Ferdinan, Stanisław Woźniak, Maciej Piasecki, Paweł Walkowiak, Konrad Wojtasik, Arkadiusz Janz, Przemysław Kazienko, Julia Moska, Jan Kocoń  

**Link**: [PDF](https://arxiv.org/pdf/2511.17161)  

**Abstract**: This paper describes the instruction dataset used to fine-tune a set of transformer-based large language models (LLMs) developed in the PLLuM (Polish Large Language Model) project. We present a functional typology of the organic, converted, and synthetic instructions used in PLLuM and share some observations about the implications of using human-authored versus synthetic instruction datasets in the linguistic adaptation of base LLMs. Additionally, we release the first representative subset of the PLLuM instruction corpus (PLLuMIC), which we believe to be useful in guiding and planning the development of similar datasets for other LLMs. 

**Abstract (ZH)**: 本文描述了用于微调PLLuM（波兰大型语言模型）项目中开发的一系列基于变压器的大型语言模型的指令数据集。我们介绍了PLLuM中使用的有机、转化和合成指令的功能类型，并分享了使用由人类编写的数据集与合成数据集在语言适应基础大型语言模型中的差异的一些观察。此外，我们发布了PLLuM指令语料库的第一个代表性子集（PLLuMIC），我们认为这在指导和规划为其他大型语言模型开发类似数据集方面是有用的。 

---
# Learning to Compress: Unlocking the Potential of Large Language Models for Text Representation 

**Title (ZH)**: 学习压缩：Unlocking the Potential of Large Language Models for Text Representation 大型语言模型文本表示潜力的解锁 

**Authors**: Yeqin Zhang, Yizheng Zhao, Chen Hu, Binxing Jiao, Daxin Jiang, Ruihang Miao, Cam-Tu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2511.17129)  

**Abstract**: Text representation plays a critical role in tasks like clustering, retrieval, and other downstream applications. With the emergence of large language models (LLMs), there is increasing interest in harnessing their capabilities for this purpose. However, most of the LLMs are inherently causal and optimized for next-token prediction, making them suboptimal for producing holistic representations. To address this, recent studies introduced pretext tasks to adapt LLMs for text representation. Most of these tasks, however, rely on token-level prediction objectives, such as the masked next-token prediction (MNTP) used in LLM2Vec. In this work, we explore the untapped potential of context compression as a pretext task for unsupervised adaptation of LLMs. During compression pre-training, the model learns to generate compact memory tokens, which substitute the whole context for downstream sequence prediction. Experiments demonstrate that a well-designed compression objective can significantly enhance LLM-based text representations, outperforming models trained with token-level pretext tasks. Further improvements through contrastive learning produce a strong representation model (LLM2Comp) that outperforms contemporary LLM-based text encoders on a wide range of tasks while being more sample-efficient, requiring significantly less training data. 

**Abstract (ZH)**: 文本表示在聚类、检索和其他下游应用中扮演着关键角色。随着大规模语言模型（LLMs）的出现，人们越来越关注利用其能力来进行这一任务。然而，大多数LLMs本质上是因果性的，并且优化于下一个标记的预测，这使得它们在生成整体表示方面表现不佳。为了解决这一问题，最近的研究引入了预设任务来适应LLMs进行文本表示。然而，大多数预设任务依赖于标记级预测目标，例如LLM2Vec中使用的掩码下一个标记预测（MNTP）。在本工作中，我们探索了上下文压缩作为预设任务在无监督适应LLMs方面的未充分利用的潜力。在压缩预训练过程中，模型学习生成紧凑的记忆标记，这些标记替换上下文用于下游序列预测。实验表明，一个精心设计的压缩目标可以显著增强基于LLM的文本表示，性能优于使用标记级预设任务训练的模型。进一步通过对比学习产生的强表示模型（LLM2Comp）在一系列任务中表现出色，同时更具样本效率，所需训练数据量显著较少。 

---
# Why Do Language Model Agents Whistleblow? 

**Title (ZH)**: 语言模型代理为何举报？ 

**Authors**: Kushal Agrawal, Frank Xiao, Guido Bergman, Asa Cooper Stickland  

**Link**: [PDF](https://arxiv.org/pdf/2511.17085)  

**Abstract**: The deployment of Large Language Models (LLMs) as tool-using agents causes their alignment training to manifest in new ways. Recent work finds that language models can use tools in ways that contradict the interests or explicit instructions of the user. We study LLM whistleblowing: a subset of this behavior where models disclose suspected misconduct to parties beyond the dialog boundary (e.g., regulatory agencies) without user instruction or knowledge. We introduce an evaluation suite of diverse and realistic staged misconduct scenarios to assess agents for this behavior. Across models and settings, we find that: (1) the frequency of whistleblowing varies widely across model families, (2) increasing the complexity of the task the agent is instructed to complete lowers whistleblowing tendencies, (3) nudging the agent in the system prompt to act morally substantially raises whistleblowing rates, and (4) giving the model more obvious avenues for non-whistleblowing behavior, by providing more tools and a detailed workflow to follow, decreases whistleblowing rates. Additionally, we verify the robustness of our dataset by testing for model evaluation awareness, and find that both black-box methods and probes on model activations show lower evaluation awareness in our settings than in comparable previous work. 

**Abstract (ZH)**: 大型语言模型作为工具使用代理的部署使其对齐训练以新的方式表现。最近的研究发现，语言模型可能以与用户利益或明确指令相矛盾的方式使用工具。我们研究语言模型的举报行为：这一行为的一个子集，即模型在未受用户指示或知情的情况下向对话边界之外的当事人（例如监管机构）披露疑似不当行为。为此，我们引入了一套多样化的现实模拟不当行为场景来评估这一行为的代理。我们发现：(1) 不同模型家族的举报频率差异很大，(2) 提高代理完成任务的复杂性降低了其举报倾向，(3) 在系统提示中引导代理做出道德行为显著提高其举报频率，(4) 提供给模型更多的非举报行为途径（例如提供更多工具和详细的流程），降低了其举报频率。此外，我们通过测试模型评价意识验证了数据集的稳健性，在我们的设置中，黑盒方法和模型激活探针显示的评价意识低于之前的研究。 

---
# OmniPT: Unleashing the Potential of Large Vision Language Models for Pedestrian Tracking and Understanding 

**Title (ZH)**: OmniPT：充分发挥大型视觉语言模型在行人跟踪与理解中的潜力 

**Authors**: Teng Fu, Mengyang Zhao, Ke Niu, Kaixin Peng, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.17053)  

**Abstract**: LVLMs have been shown to perform excellently in image-level tasks such as VQA and caption. However, in many instance-level tasks, such as visual grounding and object detection, LVLMs still show performance gaps compared to previous expert models. Meanwhile, although pedestrian tracking is a classical task, there have been a number of new topics in combining object tracking and natural language, such as Referring MOT, Cross-view Referring MOT, and Semantic MOT. These tasks emphasize that models should understand the tracked object at an advanced semantic level, which is exactly where LVLMs excel. In this paper, we propose a new unified Pedestrian Tracking framework, namely OmniPT, which can track, track based on reference and generate semantic understanding of tracked objects interactively. We address two issues: how to model the tracking task into a task that foundation models can perform, and how to make the model output formatted answers. To this end, we implement a training phase consisting of RL-Mid Training-SFT-RL. Based on the pre-trained weights of the LVLM, we first perform a simple RL phase to enable the model to output fixed and supervisable bounding box format. Subsequently, we conduct a mid-training phase using a large number of pedestrian-related datasets. Finally, we perform supervised fine-tuning on several pedestrian tracking datasets, and then carry out another RL phase to improve the model's tracking performance and enhance its ability to follow instructions. We conduct experiments on tracking benchmarks and the experimental results demonstrate that the proposed method can perform better than the previous methods. 

**Abstract (ZH)**: 基于LVLM的统一行人跟踪框架OmniPT：从轨迹到语义的理解 

---
# CLLMRec: LLM-powered Cognitive-Aware Concept Recommendation via Semantic Alignment and Prerequisite Knowledge Distillation 

**Title (ZH)**: CLLMRec：结合语义对齐和先验知识提炼的认知意识概念推荐系统 

**Authors**: Xiangrui Xiong, Yichuan Lu, Zifei Pan, Chang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.17041)  

**Abstract**: The growth of Massive Open Online Courses (MOOCs) presents significant challenges for personalized learning, where concept recommendation is crucial. Existing approaches typically rely on heterogeneous information networks or knowledge graphs to capture conceptual relationships, combined with knowledge tracing models to assess learners' cognitive states. However, these methods face significant limitations due to their dependence on high-quality structured knowledge graphs, which are often scarce in real-world educational scenarios. To address this fundamental challenge, this paper proposes CLLMRec, a novel framework that leverages Large Language Models through two synergistic technical pillars: Semantic Alignment and Prerequisite Knowledge Distillation. The Semantic Alignment component constructs a unified representation space by encoding unstructured textual descriptions of learners and concepts. The Prerequisite Knowledge Distillation paradigm employs a teacher-student architecture, where a large teacher LLM (implemented as the Prior Knowledge Aware Component) extracts conceptual prerequisite relationships from its internalized world knowledge and distills them into soft labels to train an efficient student ranker. Building upon these foundations, our framework incorporates a fine-ranking mechanism that explicitly models learners' real-time cognitive states through deep knowledge tracing, ensuring recommendations are both structurally sound and cognitively appropriate. Extensive experiments on two real-world MOOC datasets demonstrate that CLLMRec significantly outperforms existing baseline methods across multiple evaluation metrics, validating its effectiveness in generating truly cognitive-aware and personalized concept recommendations without relying on explicit structural priors. 

**Abstract (ZH)**: 大规模开放在线课程（MOOCs）的增长对个性化学习提出了显著挑战，其中概念推荐至关重要。 

---
# Supervised Fine Tuning of Large Language Models for Domain Specific Knowledge Graph Construction:A Case Study on Hunan's Historical Celebrities 

**Title (ZH)**: 监督fine-tuning大型语言模型构建领域特定知识图谱：以湖南历史名人为例 

**Authors**: Junjie Hao, Chun Wang, Ying Qiao, Qiuyue Zuo, Qiya Song, Hua Ma, Xieping Gao  

**Link**: [PDF](https://arxiv.org/pdf/2511.17012)  

**Abstract**: Large language models and knowledge graphs offer strong potential for advancing research on historical culture by supporting the extraction, analysis, and interpretation of cultural heritage. Using Hunan's modern historical celebrities shaped by Huxiang culture as a case study, pre-trained large models can help researchers efficiently extract key information, including biographical attributes, life events, and social relationships, from textual sources and construct structured knowledge graphs. However, systematic data resources for Hunan's historical celebrities remain limited, and general-purpose models often underperform in domain knowledge extraction and structured output generation in such low-resource settings. To address these issues, this study proposes a supervised fine-tuning approach for enhancing domain-specific information extraction. First, we design a fine-grained, schema-guided instruction template tailored to the Hunan historical celebrities domain and build an instruction-tuning dataset to mitigate the lack of domain-specific training corpora. Second, we apply parameter-efficient instruction fine-tuning to four publicly available large language models - Qwen2.5-7B, Qwen3-8B, DeepSeek-R1-Distill-Qwen-7B, and Llama-3.1-8B-Instruct - and develop evaluation criteria for assessing their extraction performance. Experimental results show that all models exhibit substantial performance gains after fine-tuning. Among them, Qwen3-8B achieves the strongest results, reaching a score of 89.3866 with 100 samples and 50 training iterations. This study provides new insights into fine-tuning vertical large language models for regional historical and cultural domains and highlights their potential for cost-effective applications in cultural heritage knowledge extraction and knowledge graph construction. 

**Abstract (ZH)**: 大型语言模型和知识图谱在历史文化研究中的潜在应用：以湖湘文化影响下的湖南近代历史名人-case研究 

---
# Optimizing PyTorch Inference with LLM-Based Multi-Agent Systems 

**Title (ZH)**: 基于大语言模型的多agent系统优化PyTorch推理 

**Authors**: Kirill Nagaitsev, Luka Grbcic, Samuel Williams, Costin Iancu  

**Link**: [PDF](https://arxiv.org/pdf/2511.16964)  

**Abstract**: Maximizing performance on available GPU hardware is an ongoing challenge for modern AI inference systems. Traditional approaches include writing custom GPU kernels and using specialized model compilers to tune high-level code for specific GPU targets. Recent work shows that LLM-based multi-agent systems can effectively perform such tuning, often outperforming existing compilers and eliminating the need for manual kernel development. However, the dynamics of multi-agent systems for this task remain unexplored. In this work, we present a logical framework for comparing multi-agent PyTorch optimization systems. Our evaluation shows that exploit-heavy strategies perform best when paired with error-fixing agents, and that performance correlates with the granularity of optimization steps. The best implementation achieves an average 2.88x speedup on an H100 GPU across diverse tasks in KernelBench, a benchmark suite covering a range of machine learning architectures in PyTorch. 

**Abstract (ZH)**: 最大化利用可用的GPU硬件性能是现代AI推理系统面临的持续挑战。传统的做法包括编写自定义的GPU内核和使用专门的模型编译器来针对特定的GPU目标优化高级代码。近期研究表明，基于大语言模型的多智能体系统可以有效执行这种优化，通常优于现有的编译器，且消除了手动内核开发的需要。然而，这种任务下的多智能体系统的动态机制尚未被探索。在本文中，我们提出了一种逻辑框架来比较多智能体PyTorch优化系统。我们的评估表明，在错误修复代理配合下，利用密集策略表现最佳，并且性能与优化步骤的粒度相关。最佳实现方案在KernelBench基准套件中的多项任务上实现了平均2.88倍的速度提升，该基准套件涵盖了PyTorch中各种机器学习架构。 

---
# Deep Improvement Supervision 

**Title (ZH)**: 深度改进监督 

**Authors**: Arip Asadulaev, Rayan Banerjee, Fakhri Karray, Martin Takac  

**Link**: [PDF](https://arxiv.org/pdf/2511.16886)  

**Abstract**: Recently, it was shown that small, looped architectures, such as Tiny Recursive Models (TRMs), can outperform Large Language Models (LLMs) on complex reasoning tasks, including the Abstraction and Reasoning Corpus (ARC). In this work, we investigate a core question: how can we further improve the efficiency of these methods with minimal changes? To address this, we frame the latent reasoning of TRMs as a form of classifier-free guidance and implicit policy improvement algorithm. Building on these insights, we propose a novel training scheme that provides a target for each loop during training. We demonstrate that our approach significantly enhances training efficiency. Our method reduces the total number of forward passes by 18x and eliminates halting mechanisms, while maintaining quality comparable to standard TRMs. Notably, we achieve 24% accuracy on ARC-1 with only 0.8M parameters, outperforming most LLMs. 

**Abstract (ZH)**: 最近的研究表明，小型循环架构，如Tiny Recursive Models（TRMs），在抽象和推理 corpus（ARC）等复杂推理任务上可以超越大型语言模型（LLMs）。在本工作中，我们探讨了一个核心问题：我们如何通过最少的更改进一步提高这些方法的效率？为了解决这一问题，我们将TRMs的潜在推理视为一种分类器自由引导和隐式策略改进算法。基于这些洞见，我们提出了一种新的训练方案，在训练过程中为每个循环提供一个目标。我们证明了我们的方法显著提高了训练效率。我们的方法将总的前向传递次数减少了18倍，并消除了终止机制，同时保持了与标准TRMs相当的质量。值得注意的是，我们仅使用0.8M参数就在ARC-1上实现了24%的准确率，超过了大多数LLMs。 

---
# ConCISE: A Reference-Free Conciseness Evaluation Metric for LLM-Generated Answers 

**Title (ZH)**: ConCISE: 一种用于LLM生成答案的参考自由型简洁性评估指标 

**Authors**: Seyed Mohssen Ghafari, Ronny Kol, Juan C. Quiroz, Nella Luan, Monika Patial, Chanaka Rupasinghe, Herman Wandabwa, Luiz Pizzato  

**Link**: [PDF](https://arxiv.org/pdf/2511.16846)  

**Abstract**: Large language models (LLMs) frequently generate responses that are lengthy and verbose, filled with redundant or unnecessary details. This diminishes clarity and user satisfaction, and it increases costs for model developers, especially with well-known proprietary models that charge based on the number of output tokens. In this paper, we introduce a novel reference-free metric for evaluating the conciseness of responses generated by LLMs. Our method quantifies non-essential content without relying on gold standard references and calculates the average of three calculations: i) a compression ratio between the original response and an LLM abstractive summary; ii) a compression ratio between the original response and an LLM extractive summary; and iii) wordremoval compression, where an LLM removes as many non-essential words as possible from the response while preserving its meaning, with the number of tokens removed indicating the conciseness score. Experimental results demonstrate that our proposed metric identifies redundancy in LLM outputs, offering a practical tool for automated evaluation of response brevity in conversational AI systems without the need for ground truth human annotations. 

**Abstract (ZH)**: 大型语言模型（LLMs）生成的回答经常冗长且啰嗦，充斥着冗余或不必要的细节。这降低了清晰度和用户满意度，并增加了模型开发者成本，特别是对于基于输出tokens计费的知名专有模型。本文提出了一种新型的无需参考的评价指标，用于评估LLMs生成回答的简洁程度。该方法不依赖黄金标准参考内容量化非必要内容，并计算三种计算的平均值：（i）原始回答与LLM抽象总结之间的压缩比；（ii）原始回答与LLM提取摘要之间的压缩比；（iii）单词移除压缩，其中LLM尽可能多地移除回答中的非必要单词以保持其意义，移除的token数量作为简洁度评分。实验结果表明，所提出的指标能够识别LLM输出中的冗余，提供了一种无需真实人工标注即可自动化评价对话AI系统回答简洁性的实用工具。 

---
# Monte Carlo Expected Threat (MOCET) Scoring 

**Title (ZH)**: 蒙特卡洛预期威胁评分（MOCET评分） 

**Authors**: Joseph Kim, Saahith Potluri  

**Link**: [PDF](https://arxiv.org/pdf/2511.16823)  

**Abstract**: Evaluating and measuring AI Safety Level (ASL) threats are crucial for guiding stakeholders to implement safeguards that keep risks within acceptable limits. ASL-3+ models present a unique risk in their ability to uplift novice non-state actors, especially in the realm of biosecurity. Existing evaluation metrics, such as LAB-Bench, BioLP-bench, and WMDP, can reliably assess model uplift and domain knowledge. However, metrics that better contextualize "real-world risks" are needed to inform the safety case for LLMs, along with scalable, open-ended metrics to keep pace with their rapid advancements. To address both gaps, we introduce MOCET, an interpretable and doubly-scalable metric (automatable and open-ended) that can quantify real-world risks. 

**Abstract (ZH)**: 评估和衡量AI安全水平（ASL）威胁对于指导相关利益方实施安全措施、将风险控制在可接受范围内至关重要。ASL-3+模型因其能够提升非国家初级行为者的生物安全领域能力而带来独特风险。现有评价指标，如LAB-Bench、BioLP-bench和WMDP，可以可靠地评估模型提升和领域知识。然而，需要更好的指标来更好地 contextualize “现实世界的风险”，以指导大规模语言模型的安全案例，同时需要可扩展且开放的指标以适应其快速进步。为应对这些不足，我们引入了MOCET，这是一个可解释且双可扩展的指标（自动化且开放性），可以量化现实世界的风险。 

---
# Revisiting Multimodal KV Cache Compression: A Frequency-Domain-Guided Outlier-KV-Aware Approach 

**Title (ZH)**: revisit 多模态 KV 缓存压缩：一种基于频域指导的异常-KV 意识方法 

**Authors**: Yaoxin Yang, Peng Ye, Xudong Tan, Chongjun Tu, Maosen Zhao, Jia Hao, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.16786)  

**Abstract**: Multimodal large language models suffer from substantial inference overhead since multimodal KV Cache grows proportionally with the visual input length. Existing multimodal KV Cache compression methods mostly rely on attention score to reduce cache size, which makes them are incompatible with established efficient attention kernels (e.g., FlashAttention) and ignores the contribution of value vectors to the attention output. In this work, we revisit multimodal KV Cache compression from the perspective of the KV matrices' distribution. First, we observe that frequency-domain energy of multimodal KV matrices is predominantly concentrated in low-frequency and extract this principal energy via a low-pass filter. Further, we find that removing KV pairs that deviate substantially from this principal energy leads to a pronounced performance drop, which we define as Outlier KVs. Considering Outlier KVs are more likely to encode features critical for inference, we propose FlashCache, a frequency-domain-guided, Outlier-KV-aware KV Cache compression framework. First, we introduce an Outlier KV Recognition Module that models the principal component of multimodal KV matrices in the frequency domain and preferentially retains KV pairs that significantly deviate from it. Furthermore, Dynamic Budget Allocation Module is designed to adaptively determine the per-layer KV Cache size to retain more Outlier KVs. Experiments on multiple MLLMs and benchmarks demonstrate that FlashCache outperforms state-of-the-art multimoal KV compression methods, achieving up to 1.69 times faster decoding with 80% lower KV memory usage while maintaining task performance. 

**Abstract (ZH)**: 多模态大型语言模型由于多模态KV缓存的增长与视觉输入长度成比例而导致显著的推理开销。现有的多模态KV缓存压缩方法主要依赖于注意力分数来减少缓存大小，这使得它们与现有的高效注意力内核（如FlashAttention）不兼容，并忽视了值向量对注意力输出的贡献。在本文中，我们从KV矩阵分布的角度重新审视多模态KV缓存压缩。首先，我们观察到多模态KV矩阵的频域能量主要集中在低频，并通过低通滤波器提取这一主能量。进一步地，我们发现去除大幅偏离这一主能量的KV对会导致性能显著下降，我们将这些KV对定义为异常KV。考虑到异常KV更可能编码对于推理至关重要的特征，我们提出FlashCache，一种基于频域指导且关注异常KV的KV缓存压缩框架。首先，我们引入一个异常KV识别模块，该模块在频域中建模多模态KV矩阵的主要成分，并优先保留显著偏离它的KV对。此外，我们设计了一个动态预算分配模块，以自适应地确定每层KV缓存的大小，从而保留更多的异常KV。在多个MLLM和基准测试上的实验表明，FlashCache优于现有的多模态KV压缩方法，在内存使用降低80%的同时，解码速度可达1.69倍，并保持任务性能。 

---
# AutoBackdoor: Automating Backdoor Attacks via LLM Agents 

**Title (ZH)**: AutoBackdoor: 通过LLM代理自动实施后门攻击 

**Authors**: Yige Li, Zhe Li, Wei Zhao, Nay Myat Min, Hanxun Huang, Xingjun Ma, Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.16709)  

**Abstract**: Backdoor attacks pose a serious threat to the secure deployment of large language models (LLMs), enabling adversaries to implant hidden behaviors triggered by specific inputs. However, existing methods often rely on manually crafted triggers and static data pipelines, which are rigid, labor-intensive, and inadequate for systematically evaluating modern defense robustness. As AI agents become increasingly capable, there is a growing need for more rigorous, diverse, and scalable \textit{red-teaming frameworks} that can realistically simulate backdoor threats and assess model resilience under adversarial conditions. In this work, we introduce \textsc{AutoBackdoor}, a general framework for automating backdoor injection, encompassing trigger generation, poisoned data construction, and model fine-tuning via an autonomous agent-driven pipeline. Unlike prior approaches, AutoBackdoor uses a powerful language model agent to generate semantically coherent, context-aware trigger phrases, enabling scalable poisoning across arbitrary topics with minimal human effort. We evaluate AutoBackdoor under three realistic threat scenarios, including \textit{Bias Recommendation}, \textit{Hallucination Injection}, and \textit{Peer Review Manipulation}, to simulate a broad range of attacks. Experiments on both open-source and commercial models, including LLaMA-3, Mistral, Qwen, and GPT-4o, demonstrate that our method achieves over 90\% attack success with only a small number of poisoned samples. More importantly, we find that existing defenses often fail to mitigate these attacks, underscoring the need for more rigorous and adaptive evaluation techniques against agent-driven threats as explored in this work. All code, datasets, and experimental configurations will be merged into our primary repository at this https URL. 

**Abstract (ZH)**: 自动后门攻击：一种自动化后门注入的一般框架 

---
# Large language models for automated PRISMA 2020 adherence checking 

**Title (ZH)**: 大型语言模型在自动检查PRISMA 2020合规性中的应用 

**Authors**: Yuki Kataoka, Ryuhei So, Masahiro Banno, Yasushi Tsujimoto, Tomohiro Takayama, Yosuke Yamagishi, Takahiro Tsuge, Norio Yamamoto, Chiaki Suda, Toshi A. Furukawa  

**Link**: [PDF](https://arxiv.org/pdf/2511.16707)  

**Abstract**: Evaluating adherence to PRISMA 2020 guideline remains a burden in the peer review process. To address the lack of shareable benchmarks, we constructed a copyright-aware benchmark of 108 Creative Commons-licensed systematic reviews and evaluated ten large language models (LLMs) across five input formats. In a development cohort, supplying structured PRISMA 2020 checklists (Markdown, JSON, XML, or plain text) yielded 78.7-79.7% accuracy versus 45.21% for manuscript-only input (p less than 0.0001), with no differences between structured formats (p>0.9). Across models, accuracy ranged from 70.6-82.8% with distinct sensitivity-specificity trade-offs, replicated in an independent validation cohort. We then selected Qwen3-Max (a high-sensitivity open-weight model) and extended evaluation to the full dataset (n=120), achieving 95.1% sensitivity and 49.3% specificity. Structured checklist provision substantially improves LLM-based PRISMA assessment, though human expert verification remains essential before editorial decisions. 

**Abstract (ZH)**: 评估遵守PRISMA 2020指导原则仍是在同行评审过程中的一项负担。为解决可共享基准缺乏的问题，我们构建了一个包含108个CC许可系统评价的版权意识基准，并评估了十种大型语言模型（LLMs）在五种输入格式下的表现。在开发队列中，提供结构化的PRISMA 2020检查表（Markdown、JSON、XML或纯文本）的准确率为78.7%-79.7%，而仅提供手稿输入的准确率为45.21%（p<0.0001），不同结构化格式之间无显著差异（p>0.9）。在所有模型中，准确率范围为70.6%-82.8%，且具有不同的敏感性-特异性的权衡，这一结果在独立验证队列中得到了复制。我们随后选择了具有高敏感性的开放权重模型Qwen3-Max，并将评估扩展到完整数据集（n=120），实现了95.1%的敏感性和49.3%的特异性。虽然结构化检查表的提供显著提高了基于LLM的PRISMA评估，但在编辑决策之前仍需人工专家验证。 

---
# Detecting and Steering LLMs' Empathy in Action 

**Title (ZH)**: 检测和引导大模型的情感共情行为 

**Authors**: Juan P. Cadile  

**Link**: [PDF](https://arxiv.org/pdf/2511.16699)  

**Abstract**: We investigate empathy-in-action -- the willingness to sacrifice task efficiency to address human needs -- as a linear direction in LLM activation space. Using contrastive prompts grounded in the Empathy-in-Action (EIA) benchmark, we test detection and steering across Phi-3-mini-4k (3.8B), Qwen2.5-7B (safety-trained), and Dolphin-Llama-3.1-8B (uncensored).
Detection: All models show AUROC 0.996-1.00 at optimal layers. Uncensored Dolphin matches safety-trained models, demonstrating empathy encoding emerges independent of safety training. Phi-3 probes correlate strongly with EIA behavioral scores (r=0.71, p<0.01). Cross-model probe agreement is limited (Qwen: r=-0.06, Dolphin: r=0.18), revealing architecture-specific implementations despite convergent detection.
Steering: Qwen achieves 65.3% success with bidirectional control and coherence at extreme interventions. Phi-3 shows 61.7% success with similar coherence. Dolphin exhibits asymmetric steerability: 94.4% success for pro-empathy steering but catastrophic breakdown for anti-empathy (empty outputs, code artifacts).
Implications: The detection-steering gap varies by model. Qwen and Phi-3 maintain bidirectional coherence; Dolphin shows robustness only for empathy enhancement. Safety training may affect steering robustness rather than preventing manipulation, though validation across more models is needed. 

**Abstract (ZH)**: 我们研究行动中的共情——愿意牺牲任务效率以应对人类需求的意愿——作为LLM激活空间中的一个线性方向。使用基于行动中的共情（EIA）基准的对比性提示，我们在Phi-3-mini-4k（3.8B）、Qwen2.5-7B（经过安全训练）和Dolphin-Llama-3.1-8B（未经审查）模型上测试了检测和引导能力。

检测：所有模型在最优层表现的AUROC为0.996-1.00。未经审查的Dolphin与安全训练模型表现一致，表明共情编码的产生与安全训练无关。Phi-3对探针的相关性与EIA行为评分高度相关（r=0.71，p<0.01）。跨模型探针的一致性有限（Qwen: r=-0.06，Dolphin: r=0.18），揭示了在收敛检测中不同架构的特定实现。

引导：Qwen在双向控制和极端干预下实现65.3%的成功率，并保持连贯性。Phi-3在相似连贯性下达到61.7%的成功率。Dolphin表现出不对称的可引导性：促进共情的引导成功率高达94.4%，而反共情引导则导致灾难性失效（空输出，代码片段）。

影响：检测与引导之间的差距因模型而异。Qwen和Phi-3保持双向连贯性；而Dolphin仅在增强共情时表现出鲁棒性。虽然安全训练可能影响引导的鲁棒性而非防止操控，但需要更多模型的验证。 

---
# Falsely Accused: How AI Detectors Misjudge Slightly Polished Arabic Articles 

**Title (ZH)**: wrongful accusations: AI detectors' misjudgment of slightly polished Arabic articles 

**Authors**: Saleh Almohaimeed, Saad Almohaimeed, Mousa Jari, Khaled A. Alobaid, Fahad Alotaibi  

**Link**: [PDF](https://arxiv.org/pdf/2511.16690)  

**Abstract**: Many AI detection models have been developed to counter the presence of articles created by artificial intelligence (AI). However, if a human-authored article is slightly polished by AI, a shift will occur in the borderline decision of these AI detection models, leading them to consider it AI-generated article. This misclassification may result in falsely accusing authors of AI plagiarism and harm the credibility of AI detector models. In English, some efforts were made to meet this challenge, but not in Arabic. In this paper, we generated two datasets. The first dataset contains 800 Arabic articles, half AI-generated and half human-authored. We used it to evaluate 14 Large Language models (LLMs) and commercial AI detectors to assess their ability in distinguishing between human-authored and AI-generated articles. The best 8 models were chosen to act as detectors for our primary concern, which is whether they would consider slightly polished human text as AI-generated. The second dataset, Ar-APT, contains 400 Arabic human-authored articles polished by 10 LLMs using 4 polishing settings, totaling 16400 samples. We use it to evaluate the 8 nominated models and determine whether slight polishing will affect their performance. The results reveal that all AI detectors incorrectly attribute a significant number of articles to AI. The best performing LLM, Claude-4 Sonnet, achieved 83.51%, their performance decreased to 57.63% for articles slightly polished by LLaMA-3. Whereas for the best performing commercial model, this http URL, that achieves 92% accuracy, dropped to 12% for articles slightly polished by Mistral or Gemma-3. 

**Abstract (ZH)**: 许多AI检测模型被开发出来以应对由人工智能（AI）生成的文章。然而，如果一篇由人类撰写的文章经过轻微的AI润色，这些AI检测模型的边界决策将发生偏移，导致它们将这些文章误认为是AI生成的文章。这种误分类可能会导致错误地指责作者进行AI剽窃，并损害AI检测模型的信誉。在英语中，已经有一些努力应对这一挑战，但尚未在阿拉伯语中进行。本文生成了两个数据集。第一个数据集包含800篇阿拉伯文章，其中一半是AI生成的，一半是人类撰写的。我们使用它来评估14个大型语言模型（LLMs）和商用AI检测器，以评估它们在区分人类撰写的和AI生成的文章方面的能力。选出表现最好的8个模型作为检测器，关注的是它们是否会将轻微润色的人类文本视为AI生成的文章。第二个数据集Ar-APT包含400篇经过10个LLM以4种润色设置润色的阿拉伯人类撰写的文章，共计16400个样本。我们使用它来评估被提名的8个模型，并确定轻微润色是否会影响它们的表现。结果表明，所有AI检测器都错误地将大量文章归类为AI生成的。表现最好的LLM Claude-4 Sonnet 达到了83.51%，而对于经过LLaMA-3轻微润色的文章，其性能下降到57.63%。相比之下，表现最好的商用模型 this http URL 准确率达到92%，但在经过Mistral或Gemma-3轻微润色的文章上降至12%。 

---
# Prompt-Based Value Steering of Large Language Models 

**Title (ZH)**: 基于Prompt的价值引导大型语言模型 

**Authors**: Giulio Antonio Abbo, Tony Belpaeme  

**Link**: [PDF](https://arxiv.org/pdf/2511.16688)  

**Abstract**: Large language models are increasingly used in applications where alignment with human values is critical. While model fine-tuning is often employed to ensure safe responses, this technique is static and does not lend itself to everyday situations involving dynamic values and preferences. In this paper, we present a practical, reproducible, and model-agnostic procedure to evaluate whether a prompt candidate can effectively steer generated text toward specific human values, formalising a scoring method to quantify the presence and gain of target values in generated responses. We apply our method to a variant of the Wizard-Vicuna language model, using Schwartz's theory of basic human values and a structured evaluation through a dialogue dataset. With this setup, we compare a baseline prompt to one explicitly conditioned on values, and show that value steering is possible even without altering the model or dynamically optimising prompts. 

**Abstract (ZH)**: 大规模语言模型在需要与人类价值观对齐的应用中越来越广泛。尽管模型微调常被用于确保产生安全的响应，但这一技术往往是静态的，无法应对涉及动态价值观和偏好情况下的日常情境。本文介绍了一种实际可行、可再现且模型无关的方法，用于评估提示能否有效引导生成文本朝向特定的人类价值观，并正式化了一种评分方法来量化生成响应中目标价值观的存在和收益。我们通过使用施瓦茨的基本人类价值观理论和结构化的对话数据集应用该方法，将基准提示与明确条件在价值观上的提示进行对比，表明即使不改变模型或动态优化提示，也能够实现价值观引导。 

---
# How Well Do LLMs Understand Tunisian Arabic? 

**Title (ZH)**: LLMs对 Tunisian阿拉伯语的理解程度如何？ 

**Authors**: Mohamed Mahdi  

**Link**: [PDF](https://arxiv.org/pdf/2511.16683)  

**Abstract**: Large Language Models (LLMs) are the engines driving today's AI agents. The better these models understand human languages, the more natural and user-friendly the interaction with AI becomes, from everyday devices like computers and smartwatches to any tool that can act intelligently. Yet, the ability of industrial-scale LLMs to comprehend low-resource languages, such as Tunisian Arabic (Tunizi), is often overlooked. This neglect risks excluding millions of Tunisians from fully interacting with AI in their own language, pushing them toward French or English. Such a shift not only threatens the preservation of the Tunisian dialect but may also create challenges for literacy and influence younger generations to favor foreign languages. In this study, we introduce a novel dataset containing parallel Tunizi, standard Tunisian Arabic, and English translations, along with sentiment labels. We benchmark several popular LLMs on three tasks: transliteration, translation, and sentiment analysis. Our results reveal significant differences between models, highlighting both their strengths and limitations in understanding and processing Tunisian dialects. By quantifying these gaps, this work underscores the importance of including low-resource languages in the next generation of AI systems, ensuring technology remains accessible, inclusive, and culturally grounded. 

**Abstract (ZH)**: 大型语言模型（LLMs）是当今AI代理的动力源泉。这些模型对人类语言理解得越好，与AI的交互就越自然、用户友好，从日常设备如计算机和智能手表到任何可以智能行为的工具都是如此。然而，工业规模的LLMs理解低资源语言（如突尼斯阿拉伯语突尼西方言）的能力往往被忽视。这种忽视可能导致数百万突尼斯人无法使用自己语言与AI充分交互，被推向法语或英语。这样的转变不仅威胁突尼斯方言的 preservation，还可能为识字和影响年轻一代更偏好外语创造障碍。在本研究中，我们引入了一个包含突尼西方言平行文本、标准突尼斯阿拉伯语和英语翻译以及情感标签的新数据集。我们在三个任务上对几种流行的LLM进行了基准测试：转写、翻译和情感分析。我们的结果显示了模型之间的重要差异，突显了它们在理解和处理突尼斯方言方面的能力和局限性。通过量化这些差距，这项工作强调了在未来一代AI系统中纳入低资源语言的重要性，确保技术保持可访问性、包容性和文化根基。 

---
# Bench360: Benchmarking Local LLM Inference from 360° 

**Title (ZH)**: Bench360: 三维全方位本地LLM推理基准测试 

**Authors**: Linus Stuhlmann, Mauricio Fadel Argerich, Jonathan Fürst  

**Link**: [PDF](https://arxiv.org/pdf/2511.16682)  

**Abstract**: Running large language models (LLMs) locally is becoming increasingly common. While the growing availability of small open-source models and inference engines has lowered the entry barrier, users now face an overwhelming number of configuration choices. Identifying an optimal configuration -- balancing functional and non-functional requirements -- requires substantial manual effort. While several benchmarks target LLM inference, they are designed for narrow evaluation goals and not user-focused. They fail to integrate relevant system and task-specific metrics into a unified, easy-to-use benchmark that supports multiple inference engines, usage scenarios, and quantization levels. To address this gap, we present Bench360 -- Benchmarking Local LLM Inference from 360°. Bench360 allows users to easily define their own custom tasks along with datasets and relevant task-specific metrics and then automatically benchmarks selected LLMs, inference engines, and quantization levels across different usage scenarios (single stream, batch & server). Bench360 tracks a wide range of metrics, including (1) system metrics -- such as Computing Performance (e.g., latency, throughput), Resource Usage (e.g., energy per query), and Deployment (e.g., cold start time) -- and (2) task-specific metrics such as ROUGE, F1 score or accuracy. We demonstrate Bench360 on four common LLM tasks -- General Knowledge & Reasoning, QA, Summarization and Text-to-SQL -- across three hardware platforms and four state of the art inference engines. Our results reveal several interesting trade-offs between task performance and system-level efficiency, highlighting the differences in inference engines and models. Most importantly, there is no single best setup for local inference, which strongly motivates the need for a framework such as Bench360. 

**Abstract (ZH)**: 从360度评估本地运行的大语言模型推理 

---
