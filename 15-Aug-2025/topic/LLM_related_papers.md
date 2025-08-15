# The Knowledge-Reasoning Dissociation: Fundamental Limitations of LLMs in Clinical Natural Language Inference 

**Title (ZH)**: 知识推理分离：LLMs在临床自然语言推理中的根本局限性 

**Authors**: Maël Jullien, Marco Valentino, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2508.10777)  

**Abstract**: Large language models are often assumed to acquire increasingly structured, generalizable internal representations simply by scaling data and parameters. We interrogate this assumption by introducing a Clinical Trial Natural Language Inference benchmark comprising four reasoning families, Causal Attribution, Compositional Grounding, Epistemic Verification, and Risk State Abstraction. Each item is paired with a targeted Ground Knowledge and Meta-Level Reasoning Verification (GKMRV) probe, allowing us to dissociate failures of factual access from failures of inference. We evaluate six contemporary LLMs under both direct and chain of thought prompting.
Models achieve near-ceiling GKMRV accuracy (mean accuracy 0.918) yet perform poorly on the main reasoning tasks (mean accuracy 0.25). Despite low accuracy, output inferences are highly consistent across samples (mean 0.87), indicating a systematic application of underlying heuristics and shortcuts.
These results reveal fundamental structural and representational limitations: current LLMs often possess the relevant clinical knowledge but lack the structured, composable internal representations needed to deploy it reliably (e.g., integrating constraints, weighing evidence, or simulating counterfactuals). Decoupling knowledge from reasoning with GKMRV makes this dissociation explicit and measurable, providing an effective framework for probing the reliability of LLMs in high-stakes domains. 

**Abstract (ZH)**: 大规模语言模型常被假设通过增加数据和参数的量就能获得更加结构化和可泛化的内部表示。我们通过引入一个包含四种推理类型的临床试验自然语言推理基准来质疑这一假设，这四种推理类型分别是因果归因、组合性 grounding、知识论验证和风险状态抽象。每个项目都与一个目标 ground 知识和元水平推理验证（GKMRV）探测器配对，使我们能够将事实访问失败与推理失败区分开来。我们对六种当代语言模型在直接提问和链式思考提示下进行了评估。

模型在 GKMRV 探测器上的准确率接近天花板（平均准确率 0.918），但在主要推理任务上的表现较差（平均准确率 0.25）。尽管准确率较低，但输出的推理在样本间高度一致（平均 0.87），表明模型系统地应用了底层启发式和捷径。

这些结果揭示了根本性的结构和表示限制：当前的大规模语言模型通常拥有相关的临床知识，但缺乏能够可靠运用这些知识的结构化和可组合的内部表示（例如，整合约束、评估证据或模拟假想情况）。通过 GKMRV 分离知识和推理，使这种区分变得明确和可测量，为在高风险领域探测大规模语言模型的可靠性提供了有效的框架。 

---
# Modeling Human Responses to Multimodal AI Content 

**Title (ZH)**: 建模人类对多模态AI内容的响应 

**Authors**: Zhiqi Shen, Shaojing Fan, Danni Xu, Terence Sim, Mohan Kankanhalli  

**Link**: [PDF](https://arxiv.org/pdf/2508.10769)  

**Abstract**: As AI-generated content becomes widespread, so does the risk of misinformation. While prior research has primarily focused on identifying whether content is authentic, much less is known about how such content influences human perception and behavior. In domains like trading or the stock market, predicting how people react (e.g., whether a news post will go viral), can be more critical than verifying its factual accuracy. To address this, we take a human-centered approach and introduce the MhAIM Dataset, which contains 154,552 online posts (111,153 of them AI-generated), enabling large-scale analysis of how people respond to AI-generated content. Our human study reveals that people are better at identifying AI content when posts include both text and visuals, particularly when inconsistencies exist between the two. We propose three new metrics: trustworthiness, impact, and openness, to quantify how users judge and engage with online content. We present T-Lens, an LLM-based agent system designed to answer user queries by incorporating predicted human responses to multimodal information. At its core is HR-MCP (Human Response Model Context Protocol), built on the standardized Model Context Protocol (MCP), enabling seamless integration with any LLM. This integration allows T-Lens to better align with human reactions, enhancing both interpretability and interaction capabilities. Our work provides empirical insights and practical tools to equip LLMs with human-awareness capabilities. By highlighting the complex interplay among AI, human cognition, and information reception, our findings suggest actionable strategies for mitigating the risks of AI-driven misinformation. 

**Abstract (ZH)**: 随着AI生成内容的普及，信息虚假的风险也在增加。尽管先前的研究主要集中在识别内容的真伪上，但对于此类内容如何影响人类感知和行为却知之甚少。在交易或股票市场等领域，预测人们如何反应（例如，一条新闻帖子是否会流行）可能比验证其事实准确性更为关键。为了应对这一挑战，我们采用以人类为中心的方法，并引入了MhAIM数据集，该数据集包含154,552个在线帖子（其中111,153个为AI生成），使我们能够对人们如何响应AI生成内容进行大规模分析。我们的研究表明，当帖子同时包含文本和视觉元素且两者存在不一致时，人们更容易识别出AI内容。我们提出了可信度、影响力和开放性三个新的度量标准，以量化用户对在线内容的判断和互动方式。我们展示了T-Lens，一个基于LLM的代理系统，设计用于通过整合对多模态信息的预测人类响应来回答用户查询。其核心是基于标准化模型上下文协议（MCP）构建的人类响应模型上下文协议（HR-MCP），使得T-Lens能够与任何LLM无缝集成，从而更好地与人类反应对齐，增强其可解释性和交互性。我们的研究提供了实证见解和实用工具，帮助LLM具备人类意识能力。通过突出AI、人类认知与信息接收之间的复杂相互作用，我们的研究结果表明了缓解AI驱动的信息虚假风险的可行策略。 

---
# GenOM: Ontology Matching with Description Generation and Large Language Model 

**Title (ZH)**: GenOM：基于描述生成和大规模语言模型的概念匹配 

**Authors**: Yiping Song, Jiaoyan Chen, Renate A. Schmidt  

**Link**: [PDF](https://arxiv.org/pdf/2508.10703)  

**Abstract**: Ontology matching (OM) plays an essential role in enabling semantic interoperability and integration across heterogeneous knowledge sources, particularly in the biomedical domain which contains numerous complex concepts related to diseases and pharmaceuticals. This paper introduces GenOM, a large language model (LLM)-based ontology alignment framework, which enriches the semantic representations of ontology concepts via generating textual definitions, retrieves alignment candidates with an embedding model, and incorporates exact matching-based tools to improve precision. Extensive experiments conducted on the OAEI Bio-ML track demonstrate that GenOM can often achieve competitive performance, surpassing many baselines including traditional OM systems and recent LLM-based methods. Further ablation studies confirm the effectiveness of semantic enrichment and few-shot prompting, highlighting the framework's robustness and adaptability. 

**Abstract (ZH)**: 基于大型语言模型的本体匹配框架GenOM及其在生物医学领域的应用 

---
# MSRS: Adaptive Multi-Subspace Representation Steering for Attribute Alignment in Large Language Models 

**Title (ZH)**: MSRS：面向大型语言模型属性对齐的自适应多子空间表示 steering 算法 

**Authors**: Xinyan Jiang, Lin Zhang, Jiayi Zhang, Qingsong Yang, Guimin Hu, Di Wang, Lijie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10599)  

**Abstract**: Activation steering offers a promising approach to controlling the behavior of Large Language Models by directly manipulating their internal activations. However, most existing methods struggle to jointly steer multiple attributes, often resulting in interference and undesirable trade-offs. To address this challenge, we propose Multi-Subspace Representation Steering (MSRS), a novel framework for effective multi-attribute steering via subspace representation fine-tuning. MSRS reduces inter-attribute interference by allocating orthogonal subspaces to each attribute, isolating their influence within the model's representation space. MSRS also incorporates a hybrid subspace composition strategy: it combines attribute-specific subspaces for unique steering directions with a shared subspace for common steering directions. A dynamic weighting function learns to efficiently integrate these components for precise control. During inference, MSRS introduces a token-level steering mechanism that dynamically identifies and intervenes on the most semantically relevant tokens, enabling fine-grained behavioral modulation. Experimental results show that MSRS significantly reduces attribute conflicts, surpasses existing methods across a range of attributes, and generalizes effectively to diverse downstream tasks. 

**Abstract (ZH)**: 多子空间表示导向：一种通过子空间表示微调实现有效多属性导向的新框架 

---
# Improving Value-based Process Verifier via Low-Cost Variance Reduction 

**Title (ZH)**: 基于低变异减少的价值驱动过程验证改进 

**Authors**: Zetian Sun, Dongfang Li, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10539)  

**Abstract**: Large language models (LLMs) have achieved remarkable success in a wide range of tasks. However, their reasoning capabilities, particularly in complex domains like mathematics, remain a significant challenge. Value-based process verifiers, which estimate the probability of a partial reasoning chain leading to a correct solution, are a promising approach for improving reasoning. Nevertheless, their effectiveness is often hindered by estimation error in their training annotations, a consequence of the limited number of Monte Carlo (MC) samples feasible due to the high cost of LLM inference. In this paper, we identify that the estimation error primarily arises from high variance rather than bias, and the MC estimator is a Minimum Variance Unbiased Estimator (MVUE). To address the problem, we propose the \textsc{Com}pound \textsc{M}onte \textsc{C}arlo \textsc{S}ampling (ComMCS) method, which constructs an unbiased estimator by linearly combining the MC estimators from the current and subsequent steps. Theoretically, we show that our method leads to a predictable reduction in variance, while maintaining an unbiased estimation without additional LLM inference cost. We also perform empirical experiments on the MATH-500 and GSM8K benchmarks to demonstrate the effectiveness of our method. Notably, ComMCS outperforms regression-based optimization method by 2.8 points, the non-variance-reduced baseline by 2.2 points on MATH-500 on Best-of-32 sampling experiment. 

**Abstract (ZH)**: 大型语言模型（LLMs）在广泛的任务中取得了显著的成功。然而，它们在复杂领域如数学领域的推理能力仍然是一个重大的挑战。基于价值的过程验证器，通过估计部分推理链达到正确解的概率，是一个有望提高推理能力的方法。然而，其有效性常常受到训练注释中估计误差的阻碍，这是由于由于LLM推断成本高导致的蒙特卡洛（MC）样本数量有限所产生的结果。在本文中，我们识别出估计误差主要来源于高方差而非偏差，而MC估计器是一个最小方差无偏估计器（MVUE）。为了解决这个问题，我们提出了复合蒙特卡洛采样（ComMCS）方法，该方法通过线性结合当前步和后续步的MC估计器来构建一个无偏估计器。理论上，我们证明我们的方法会导致方差可预测的减少，同时在不增加额外LLM推断成本的情况下保持无偏估计。我们还在MATH-500和GSM8K基准上进行了实证实验，以展示我们方法的有效性。值得注意的是，在Best-of-32抽样实验中，ComMCS在MATH-500基准上的表现比基于回归优化的方法高出2.8分，比非方差降低的基线高出2.2分。 

---
# Diversity First, Quality Later: A Two-Stage Assumption for Language Model Alignment 

**Title (ZH)**: 优先多样性，后求质量：一种两阶段语言模型对齐假设 

**Authors**: Zetian Sun, Dongfang Li, Baotian Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10530)  

**Abstract**: The alignment of language models (LMs) with human preferences is critical for building reliable AI systems. The problem is typically framed as optimizing an LM policy to maximize the expected reward that reflects human preferences. Recently, Direct Preference Optimization (DPO) was proposed as a LM alignment method that directly optimize the policy from static preference data, and further improved by incorporating on-policy sampling (i.e., preference candidates generated during the training loop) for better LM alignment. However, we show on-policy data is not always optimal, with systematic effectiveness difference emerging between static and on-policy preference candidates. For example, on-policy data can result in a 3$\times$ effectiveness compared with static data for Llama-3, and a 0.4$\times$ effectiveness for Zephyr. To explain the phenomenon, we propose the alignment stage assumption, which divides the alignment process into two distinct stages: the preference injection stage, which benefits from diverse data, and the preference fine-tuning stage, which favors high-quality data. Through theoretical and empirical analysis, we characterize these stages and propose an effective algorithm to identify the boundaries between them. We perform experiments on 5 models (Llama, Zephyr, Phi-2, Qwen, Pythia) and 2 alignment methods (DPO, SLiC-HF) to show the generalizability of alignment stage assumption and boundary measurement. 

**Abstract (ZH)**: 语言模型与人类偏好对齐对于构建可靠的人工智能系统至关重要。该问题通常被表述为优化语言模型策略以最大化反映人类偏好的预期奖励。最近，直接偏好优化（DPO）被提出作为一种直接从静态偏好数据优化策略的语言模型对齐方法，并通过引入在线采样（即在训练循环中生成的偏好候选）进一步改进以提高语言模型对齐效果。然而，我们显示在线数据并不总是最优的，静态和在线偏好候选之间存在系统性的有效性差异。例如，对于Llama-3，使用在线数据的效用可高出静态数据3倍，而对于Zephyr，则低至0.4倍。为解释这一现象，我们提出了对齐阶段假设，将对齐过程分为两个明显的阶段：偏好注入阶段，受益于多样化的数据；偏好微调阶段，偏好高质量的数据。通过理论和经验分析，我们界定了这两个阶段，并提出了一种有效算法来识别它们之间的边界。我们在5个模型（Llama、Zephyr、Phi-2、Qwen、Pythia）和2种对齐方法（DPO、SLiC-HF）上进行了实验，以展示对齐阶段假设和边界测量的一般适用性。 

---
# Reverse Physician-AI Relationship: Full-process Clinical Diagnosis Driven by a Large Language Model 

**Title (ZH)**: 大型语言模型驱动的全过程临床诊断逆向医师-AI关系 

**Authors**: Shicheng Xu, Xin Huang, Zihao Wei, Liang Pang, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.10492)  

**Abstract**: Full-process clinical diagnosis in the real world encompasses the entire diagnostic workflow that begins with only an ambiguous chief complaint. While artificial intelligence (AI), particularly large language models (LLMs), is transforming clinical diagnosis, its role remains largely as an assistant to physicians. This AI-assisted working pattern makes AI can only answer specific medical questions at certain parts within the diagnostic process, but lack the ability to drive the entire diagnostic process starting from an ambiguous complaint, which still relies heavily on human physicians. This gap limits AI's ability to fully reduce physicians' workload and enhance diagnostic efficiency. To address this, we propose a paradigm shift that reverses the relationship between physicians and AI: repositioning AI as the primary director, with physicians serving as its assistants. So we present DxDirector-7B, an LLM endowed with advanced deep thinking capabilities, enabling it to drive the full-process diagnosis with minimal physician involvement. Furthermore, DxDirector-7B establishes a robust accountability framework for misdiagnoses, delineating responsibility between AI and human physicians. In evaluations across rare, complex, and real-world cases under full-process diagnosis setting, DxDirector-7B not only achieves significant superior diagnostic accuracy but also substantially reduces physician workload than state-of-the-art medical LLMs as well as general-purpose LLMs. Fine-grained analyses across multiple clinical departments and tasks validate its efficacy, with expert evaluations indicating its potential to serve as a viable substitute for medical specialists. These findings mark a new era where AI, traditionally a physicians' assistant, now drives the entire diagnostic process to drastically reduce physicians' workload, indicating an efficient and accurate diagnostic solution. 

**Abstract (ZH)**: 全过程中临床诊断在现实世界中的全流程临床诊断：从模糊主诉到全过程中人工智能的角色转变与应用 

---
# SEQ-GPT: LLM-assisted Spatial Query via Example 

**Title (ZH)**: SEQ-GPT: LLM辅助的基于示例的空间查询 

**Authors**: Ivan Khai Ze Lim, Ningyi Liao, Yiming Yang, Gerald Wei Yong Yip, Siqiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2508.10486)  

**Abstract**: Contemporary spatial services such as online maps predominantly rely on user queries for location searches. However, the user experience is limited when performing complex tasks, such as searching for a group of locations simultaneously. In this study, we examine the extended scenario known as Spatial Exemplar Query (SEQ), where multiple relevant locations are jointly searched based on user-specified examples. We introduce SEQ-GPT, a spatial query system powered by Large Language Models (LLMs) towards more versatile SEQ search using natural language. The language capabilities of LLMs enable unique interactive operations in the SEQ process, including asking users to clarify query details and dynamically adjusting the search based on user feedback. We also propose a tailored LLM adaptation pipeline that aligns natural language with structured spatial data and queries through dialogue synthesis and multi-model cooperation. SEQ-GPT offers an end-to-end demonstration for broadening spatial search with realistic data and application scenarios. 

**Abstract (ZH)**: 基于Large Language Models的空间示例查询系统.SEQ-GPT 

---
# FIRESPARQL: A LLM-based Framework for SPARQL Query Generation over Scholarly Knowledge Graphs 

**Title (ZH)**: FIRESPARQL：一种基于大语言模型的 Scholarly 知识图谱 SPARQL 查询生成框架 

**Authors**: Xueli Pan, Victor de Boer, Jacco van Ossenbruggen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10467)  

**Abstract**: Question answering over Scholarly Knowledge Graphs (SKGs) remains a challenging task due to the complexity of scholarly content and the intricate structure of these graphs. Large Language Model (LLM) approaches could be used to translate natural language questions (NLQs) into SPARQL queries; however, these LLM-based approaches struggle with SPARQL query generation due to limited exposure to SKG-specific content and the underlying schema. We identified two main types of errors in the LLM-generated SPARQL queries: (i) structural inconsistencies, such as missing or redundant triples in the queries, and (ii) semantic inaccuracies, where incorrect entities or properties are shown in the queries despite a correct query structure. To address these issues, we propose FIRESPARQL, a modular framework that supports fine-tuned LLMs as a core component, with optional context provided via retrieval-augmented generation (RAG) and a SPARQL query correction layer. We evaluate the framework on the SciQA Benchmark using various configurations (zero-shot, zero-shot with RAG, one-shot, fine-tuning, and fine-tuning with RAG) and compare the performance with baseline and state-of-the-art approaches. We measure query accuracy using BLEU and ROUGE metrics, and query result accuracy using relaxed exact match(RelaxedEM), with respect to the gold standards containing the NLQs, SPARQL queries, and the results of the queries. Experimental results demonstrate that fine-tuning achieves the highest overall performance, reaching 0.90 ROUGE-L for query accuracy and 0.85 RelaxedEM for result accuracy on the test set. 

**Abstract (ZH)**: 面向学术知识图谱的问答仍是一项具有挑战性的任务：由于学术内容的复杂性以及这些图谱的复杂结构。大型语言模型（LLM）方法可以用来将自然语言问题（NLQs）转换为SPARQL查询；然而，这些基于LLM的方法在生成SPARQL查询时遇到困难，因为它们对特定于学术知识图谱的内容和底层模式接触有限。我们识别出LLM生成的SPARQL查询中的两类主要错误：（i）结构不一致，如查询中缺少或多余三元组，以及（ii）语义不准确，尽管查询结构正确，但在查询中显示了错误的实体或属性。为了解决这些问题，我们提出了FIRESPARQL，一个模块化框架，其中支持微调的LLM作为核心组件，并可选地通过检索增强生成（RAG）提供上下文支持，以及一个SPARQL查询校正层。我们在SciQA基准上使用各种配置（零样本、带RAG的零样本、单样本、微调和带RAG的微调）评估该框架，并将其性能与基准和最新方法进行比较。我们使用BLEU和ROUGE衡量查询准确性，并使用宽松的确切匹配（RelaxedEM）衡量查询结果准确性，这些结果对包含NLQ、SPARQL查询以及查询结果的黄金标准进行测量。实验结果表明，微调实现了最高的整体性能，在测试集上达到0.90的ROUGE-L查询准确性以及0.85的RelaxedEM结果准确性。 

---
# LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval 

**Title (ZH)**: 基于知识图谱的语义聚合与分级检索生成方法 

**Authors**: Yaoze Zhang, Rong Wu, Pinlong Cai, Xiaoman Wang, Guohang Yan, Song Mao, Ding Wang, Botian Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.10391)  

**Abstract**: Retrieval-Augmented Generation (RAG) plays a crucial role in grounding Large Language Models by leveraging external knowledge, whereas the effectiveness is often compromised by the retrieval of contextually flawed or incomplete information. To address this, knowledge graph-based RAG methods have evolved towards hierarchical structures, organizing knowledge into multi-level summaries. However, these approaches still suffer from two critical, unaddressed challenges: high-level conceptual summaries exist as disconnected ``semantic islands'', lacking the explicit relations needed for cross-community reasoning; and the retrieval process itself remains structurally unaware, often degenerating into an inefficient flat search that fails to exploit the graph's rich topology. To overcome these limitations, we introduce LeanRAG, a framework that features a deeply collaborative design combining knowledge aggregation and retrieval strategies. LeanRAG first employs a novel semantic aggregation algorithm that forms entity clusters and constructs new explicit relations among aggregation-level summaries, creating a fully navigable semantic network. Then, a bottom-up, structure-guided retrieval strategy anchors queries to the most relevant fine-grained entities and then systematically traverses the graph's semantic pathways to gather concise yet contextually comprehensive evidence sets. The LeanRAG can mitigate the substantial overhead associated with path retrieval on graphs and minimizes redundant information retrieval. Extensive experiments on four challenging QA benchmarks with different domains demonstrate that LeanRAG significantly outperforming existing methods in response quality while reducing 46\% retrieval redundancy. Code is available at: this https URL 

**Abstract (ZH)**: 基于知识图谱的检索增强生成 (RAG) 在通过利用外部知识将大型语言模型与现实世界对接方面发挥着关键作用，但其有效性往往因检索到的上下文不合适或不完整的信息而受损。为解决这一问题，基于知识图谱的RAG方法已经发展出分层结构，将知识组织成多级摘要。然而，这些方法仍然面临两个未解决的关键挑战：高层概念性摘要存在于互不连通的“语义孤岛”中，缺乏用于跨社区推理的显式关系；并且检索过程本身仍然缺乏结构感知，经常退化为低效的扁平搜索，无法充分利用图的丰富拓扑结构。为克服这些限制，我们引入了LeanRAG框架，该框架采用了深度协作设计，结合了知识聚合与检索策略。LeanRAG首先采用了一种新颖的语义聚合算法，形成实体集群，并在聚合层面构建新的显式关系，创建了一个完全可导航的语义网络。然后，采用自底向上的结构引导检索策略，将查询锚定到最相关的细粒度实体，并系统地遍历图的语义路径以收集简洁且上下文全面的证据集。LeanRAG可以减轻路径检索的显著开销，并最大限度地减少冗余信息的检索。在四个具有不同领域的挑战性问答基准测试中进行的广泛实验表明，LeanRAG在回答质量上显著优于现有方法，同时减少了46%的检索冗余。代码可在以下链接获取：this https URL。 

---
# What to Ask Next? Probing the Imaginative Reasoning of LLMs with TurtleSoup Puzzles 

**Title (ZH)**: 下一步该询问什么？使用TurtleSoup难题探究大模型的想象力推理 

**Authors**: Mengtao Zhou, Sifan Wu, Huan Zhang, Qi Sima, Bang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10358)  

**Abstract**: We investigate the capacity of Large Language Models (LLMs) for imaginative reasoning--the proactive construction, testing, and revision of hypotheses in information-sparse environments. Existing benchmarks, often static or focused on social deduction, fail to capture the dynamic, exploratory nature of this reasoning process. To address this gap, we introduce a comprehensive research framework based on the classic "Turtle Soup" game, integrating a benchmark, an agent, and an evaluation protocol. We present TurtleSoup-Bench, the first large-scale, bilingual, interactive benchmark for imaginative reasoning, comprising 800 turtle soup puzzles sourced from both the Internet and expert authors. We also propose Mosaic-Agent, a novel agent designed to assess LLMs' performance in this setting. To evaluate reasoning quality, we develop a multi-dimensional protocol measuring logical consistency, detail completion, and conclusion alignment. Experiments with leading LLMs reveal clear capability limits, common failure patterns, and a significant performance gap compared to humans. Our work offers new insights into LLMs' imaginative reasoning and establishes a foundation for future research on exploratory agent behavior. 

**Abstract (ZH)**: 我们探讨了大型语言模型在想象性推理方面的能力——即在信息稀疏环境中主动构建、测试和修订假设的能力。现有的基准测试往往静态化或集中在社交推理上，未能捕捉到这种推理过程的动态探索性特征。为了弥补这一差距，我们基于经典的“乌龟汤”游戏引入了一个全面的研究框架，整合了基准测试、智能体和评估协议。我们推出了TurtleSoup-Bench，这是首个大规模、双语、互动性的想象性推理基准测试，包含800道来自互联网和专家作者的乌龟汤谜题。我们还提出了Mosaic-Agent，这是一种新颖的智能体，用于评估语言模型在这一环境下的表现。为了评估推理质量，我们开发了一个多维度的评估协议，衡量逻辑一致性、细节补充和结论对齐。与领先的大语言模型的实验揭示了明显的推理能力限制、常见的失败模式以及与人类之间的显著性能差距。我们的工作为大语言模型的想象性推理提供了新的视角，并为未来探索性智能体行为的研究奠定了基础。 

---
# Why Cannot Large Language Models Ever Make True Correct Reasoning? 

**Title (ZH)**: 大型语言模型为何永远无法进行真正的正确推理？ 

**Authors**: Jingde Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.10265)  

**Abstract**: Recently, with the application progress of AIGC tools based on large language models (LLMs), led by ChatGPT, many AI experts and more non-professionals are trumpeting the "understanding ability" and "reasoning ability" of the LLMs. The present author considers that the so-called "understanding ability" and "reasoning ability" of LLMs are just illusions of those people who with vague concepts. In fact, the LLMs can never have the true understanding ability and true reasoning ability. This paper intents to explain that, because the essential limitations of their working principle, the LLMs can never have the ability of true correct reasoning. 

**Abstract (ZH)**: 近年来，随着以大型语言模型（LLMs）为底层技术的AIGC工具，特别是ChatGPT的广泛应用，许多AI专家甚至非专业人士都在极力吹捧LLMs的“理解能力”和“推理能力”。本文认为，所谓的LLMs的“理解能力”和“推理能力”只是模糊概念下的幻觉。事实上，LLMs永远不可能具备真正的理解能力和真正的推理能力。本文旨在解释，由于其工作原理的本质限制，LLMs永远不可能具备真正的正确推理能力。 

---
# KompeteAI: Accelerated Autonomous Multi-Agent System for End-to-End Pipeline Generation for Machine Learning Problems 

**Title (ZH)**: KompeteAI：端到端机器学习问题自主多Agent系统加速生成管道的系统 

**Authors**: Stepan Kulibaba, Artem Dzhalilov, Roman Pakhomov, Oleg Svidchenko, Alexander Gasnikov, Aleksei Shpilman  

**Link**: [PDF](https://arxiv.org/pdf/2508.10177)  

**Abstract**: Recent Large Language Model (LLM)-based AutoML systems demonstrate impressive capabilities but face significant limitations such as constrained exploration strategies and a severe execution bottleneck. Exploration is hindered by one-shot methods lacking diversity and Monte Carlo Tree Search (MCTS) approaches that fail to recombine strong partial solutions. The execution bottleneck arises from lengthy code validation cycles that stifle iterative refinement. To overcome these challenges, we introduce KompeteAI, a novel AutoML framework with dynamic solution space exploration. Unlike previous MCTS methods that treat ideas in isolation, KompeteAI introduces a merging stage that composes top candidates. We further expand the hypothesis space by integrating Retrieval-Augmented Generation (RAG), sourcing ideas from Kaggle notebooks and arXiv papers to incorporate real-world strategies. KompeteAI also addresses the execution bottleneck via a predictive scoring model and an accelerated debugging method, assessing solution potential using early stage metrics to avoid costly full-code execution. This approach accelerates pipeline evaluation 6.9 times. KompeteAI outperforms leading methods (e.g., RD-agent, AIDE, and Ml-Master) by an average of 3\% on the primary AutoML benchmark, MLE-Bench. Additionally, we propose Kompete-bench to address limitations in MLE-Bench, where KompeteAI also achieves state-of-the-art results 

**Abstract (ZH)**: 基于大语言模型（LLM）的AutoML系统最近展现了令人印象深刻的性能，但面临着探索策略受限和执行瓶颈等重大挑战。探索受限于一-shot方法缺乏多样性，以及蒙特卡洛树搜索（MCTS）方法无法重组强部分解。执行瓶颈源于冗长的代码验证周期，阻碍了迭代优化。为克服这些挑战，我们提出KompeteAI，这是一种具有动态解空间探索的新颖AutoML框架。不同于之前的MCTS方法孤立处理想法，KompeteAI引入了合并阶段，将顶级候选方案组合。我们进一步通过集成检索增强生成（RAG）扩展假设空间，从Kaggle笔记本和arXiv论文中汲取实际策略。KompeteAI通过预测评分模型和加速调试方法来解决执行瓶颈，使用早期指标评估解的潜力，以避免昂贵的完整代码执行。这种方法将管道评估加速了6.9倍。KompeteAI在主要的AutoML基准ML-Evaluation-Benchmarks上平均优于领先方法（如RD-agent、AIDE和Ml-Master）3%。此外，我们提出了Kompete-bench以解决ML-Evaluation-Benchmarks的局限性， KompeteAI在其中也取得了最先进的结果。 

---
# A Survey of Optimization Modeling Meets LLMs: Progress and Future Directions 

**Title (ZH)**: 优化建模遇见大语言模型：进展与未来方向 

**Authors**: Ziyang Xiao, Jingrong Xie, Lilin Xu, Shisi Guan, Jingyan Zhu, Xiongwei Han, Xiaojin Fu, WingYin Yu, Han Wu, Wei Shi, Qingcan Kang, Jiahui Duan, Tao Zhong, Mingxuan Yuan, Jia Zeng, Yuan Wang, Gang Chen, Dongxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10047)  

**Abstract**: By virtue of its great utility in solving real-world problems, optimization modeling has been widely employed for optimal decision-making across various sectors, but it requires substantial expertise from operations research professionals. With the advent of large language models (LLMs), new opportunities have emerged to automate the procedure of mathematical modeling. This survey presents a comprehensive and timely review of recent advancements that cover the entire technical stack, including data synthesis and fine-tuning for the base model, inference frameworks, benchmark datasets, and performance evaluation. In addition, we conducted an in-depth analysis on the quality of benchmark datasets, which was found to have a surprisingly high error rate. We cleaned the datasets and constructed a new leaderboard with fair performance evaluation in terms of base LLM model and datasets. We also build an online portal that integrates resources of cleaned datasets, code and paper repository to benefit the community. Finally, we identify limitations in current methodologies and outline future research opportunities. 

**Abstract (ZH)**: 基于大型语言模型的新机遇：优化建模自动化技术综述 

---
# Searching for Privacy Risks in LLM Agents via Simulation 

**Title (ZH)**: 通过模拟搜索LLM代理中的隐私风险 kukuxe kukuxe 

**Authors**: Yanzhe Zhang, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10880)  

**Abstract**: The widespread deployment of LLM-based agents is likely to introduce a critical privacy threat: malicious agents that proactively engage others in multi-turn interactions to extract sensitive information. These dynamic dialogues enable adaptive attack strategies that can cause severe privacy violations, yet their evolving nature makes it difficult to anticipate and discover sophisticated vulnerabilities manually. To tackle this problem, we present a search-based framework that alternates between improving attacker and defender instructions by simulating privacy-critical agent interactions. Each simulation involves three roles: data subject, data sender, and data recipient. While the data subject's behavior is fixed, the attacker (data recipient) attempts to extract sensitive information from the defender (data sender) through persistent and interactive exchanges. To explore this interaction space efficiently, our search algorithm employs LLMs as optimizers, using parallel search with multiple threads and cross-thread propagation to analyze simulation trajectories and iteratively propose new instructions. Through this process, we find that attack strategies escalate from simple direct requests to sophisticated multi-turn tactics such as impersonation and consent forgery, while defenses advance from rule-based constraints to identity-verification state machines. The discovered attacks and defenses transfer across diverse scenarios and backbone models, demonstrating strong practical utility for building privacy-aware agents. 

**Abstract (ZH)**: 基于LLM的代理广泛应用可能会引入一个关键的隐私威胁：恶意代理主动进行多轮交互以提取敏感信息。为了应对这一问题，我们提出了一种基于搜索的框架，该框架通过模拟隐私关键代理交互交替提升攻击者和防御者的指令。每个模拟涉及三个角色：数据主体、数据发送者和数据接收者。数据主体的行为固定，而数据接收者（攻击者）试图通过持续的互动交换从数据发送者（防御者）处获取敏感信息。为了高效探索这种交互空间，我们的搜索算法利用LLM作为优化器，采用多线程并行搜索和跨线程传播来分析模拟轨迹并迭代提出新的指令。通过这一过程，我们发现攻击策略从简单的直接请求升级为复杂的多轮策略，如冒充和同意伪造，而防御措施则从基于规则的约束发展为身份验证状态机。发现的攻击和防御策略在多种场景和基础模型之间具有很强的转移性，展示了在构建隐私意识代理方面的重要实用价值。 

---
# A Survey on Diffusion Language Models 

**Title (ZH)**: 扩散语言模型综述 

**Authors**: Tianyi Li, Mingda Chen, Bowei Guo, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10875)  

**Abstract**: Diffusion Language Models (DLMs) are rapidly emerging as a powerful and promising alternative to the dominant autoregressive (AR) paradigm. By generating tokens in parallel through an iterative denoising process, DLMs possess inherent advantages in reducing inference latency and capturing bidirectional context, thereby enabling fine-grained control over the generation process. While achieving a several-fold speed-up, recent advancements have allowed DLMs to show performance comparable to their autoregressive counterparts, making them a compelling choice for various natural language processing tasks. In this survey, we provide a holistic overview of the current DLM landscape. We trace its evolution and relationship with other paradigms, such as autoregressive and masked language models, and cover both foundational principles and state-of-the-art models. Our work offers an up-to-date, comprehensive taxonomy and an in-depth analysis of current techniques, from pre-training strategies to advanced post-training methods. Another contribution of this survey is a thorough review of DLM inference strategies and optimizations, including improvements in decoding parallelism, caching mechanisms, and generation quality. We also highlight the latest approaches to multimodal extensions of DLMs and delineate their applications across various practical scenarios. Furthermore, our discussion addresses the limitations and challenges of DLMs, including efficiency, long-sequence handling, and infrastructure requirements, while outlining future research directions to sustain progress in this rapidly evolving field. Project GitHub is available at this https URL. 

**Abstract (ZH)**: 扩散语言模型（DLMs）正迅速成为与主宰性的自回归（AR）范式相得益彰的强大而有前景的替代方案。通过迭代去噪过程并行生成令牌，DLMs在减少推理延迟和捕捉双向上下文方面拥有固有的优势，从而能够对生成过程进行精细控制。在实现几倍速提升的同时，近年来的进步已经使DLMs在性能上与自回归模型相当，使之成为各种自然语言处理任务的一个有吸引力的选择。在本文综述中，我们提供了当前DLM景观的全面概述。我们追溯了DLM的发展历程及其与其他范式（如自回归和掩码语言模型）的关系，并涵盖了基础原理和最新模型。我们的工作提供了最新的、全面的分类学和当前技术的深入分析，从预训练策略到先进的后训练方法。本文综述的另一个贡献是对DLM推理策略和优化进行了详细审查，包括解码并行性、缓存机制和生成质量的改进。我们还强调了DLM多模态扩展的最新方法及其在各种实际场景中的应用。此外，我们的讨论还探讨了DLM的局限性和挑战，包括效率、长序列处理和基础设施要求，并指出了未来的研究方向以保持这一快速发展的领域的进步。GitHub项目页面地址请点击这里。 

---
# Performance of GPT-5 in Brain Tumor MRI Reasoning 

**Title (ZH)**: GPT-5在脑肿瘤MRI推理中的表现 

**Authors**: Mojtaba Safari, Shansong Wang, Mingzhe Hu, Zach Eidex, Qiang Li, Xiaofeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10865)  

**Abstract**: Accurate differentiation of brain tumor types on magnetic resonance imaging (MRI) is critical for guiding treatment planning in neuro-oncology. Recent advances in large language models (LLMs) have enabled visual question answering (VQA) approaches that integrate image interpretation with natural language reasoning. In this study, we evaluated GPT-4o, GPT-5-nano, GPT-5-mini, and GPT-5 on a curated brain tumor VQA benchmark derived from 3 Brain Tumor Segmentation (BraTS) datasets - glioblastoma (GLI), meningioma (MEN), and brain metastases (MET). Each case included multi-sequence MRI triplanar mosaics and structured clinical features transformed into standardized VQA items. Models were assessed in a zero-shot chain-of-thought setting for accuracy on both visual and reasoning tasks. Results showed that GPT-5-mini achieved the highest macro-average accuracy (44.19%), followed by GPT-5 (43.71%), GPT-4o (41.49%), and GPT-5-nano (35.85%). Performance varied by tumor subtype, with no single model dominating across all cohorts. These findings suggest that GPT-5 family models can achieve moderate accuracy in structured neuro-oncological VQA tasks, but not at a level acceptable for clinical use. 

**Abstract (ZH)**: 磁共振成像（MRI）中脑肿瘤类型准确区分对于神经肿瘤学治疗规划至关重要。大型语言模型（LLMs）的 recent 进展使图像解释与自然语言推理结合的视觉问答（VQA）方法得以实现。在本研究中，我们基于 3 个脑肿瘤分割（BraTS）数据集（胶质母细胞瘤(GLI)、脑膜瘤(MEN) 和 脑转移瘤(MET)）构建了一个定制的脑肿瘤 VQA 验证基准，并评估了 GPT-4o、GPT-5-nano、GPT-5-mini 和 GPT-5 在此基准上的表现。每个案例包含多序列 MRI 三维镶嵌图像和结构化临床特征转换为标准化 VQA 项目。模型在零样本链式思考设置下评估了视觉和推理任务的准确性。结果显示，GPT-5-mini 达到最高的宏平均准确率（44.19%），其次是 GPT-5（43.71%）、GPT-4o（41.49%）和 GPT-5-nano（35.85%）。不同肿瘤亚型的表现各异，并无单一模型在所有组别中表现最优。这些发现表明，GPT-5 家族模型在结构化神经肿瘤学 VQA 任务中可以实现中等准确率，但尚不足以应用于临床。 

---
# Reinforced Language Models for Sequential Decision Making 

**Title (ZH)**: 强化语言模型在 sequential 决策中的应用 

**Authors**: Jim Dilkes, Vahid Yazdanpanah, Sebastian Stein  

**Link**: [PDF](https://arxiv.org/pdf/2508.10839)  

**Abstract**: Large Language Models (LLMs) show potential as sequential decision-making agents, but their application is often limited due to a reliance on large, computationally expensive models. This creates a need to improve smaller models, yet existing post-training methods are designed for single-turn interactions and cannot handle credit assignment in multi-step agentic tasks. To address this, we introduce Multi-Step Group-Relative Policy Optimization (MS-GRPO), a new algorithm for post-training LLM agents, grounded in formal Text-Mediated Stochastic Game (TSMG) and Language-Agent Policy (LAP) frameworks. For credit assignment, MS-GRPO attributes the entire cumulative episode reward to each individual episode step. We supplement this algorithm with a novel absolute-advantage-weighted episode sampling strategy that we show improves training performance. We evaluate our approach by post-training a 3-billion parameter model on Snake and Frozen Lake. Our experiments demonstrate that the method is effective in improving decision-making performance: our post-trained 3B parameter model outperforms a 72B parameter baseline by 50% on the Frozen Lake task. This work demonstrates that targeted post-training is a practical and efficient alternative to relying on model scale for creating sequential decision-making agents using LLMs. 

**Abstract (ZH)**: 多步组相关策略优化：基于形式化文本中介随机游戏与语言代理策略框架的后训练算法 

---
# Pass@k Training for Adaptively Balancing Exploration and Exploitation of Large Reasoning Models 

**Title (ZH)**: 基于Pass@k的训练方法以适应性ively 平衡大规模推理模型的探索与利用。 

**Authors**: Zhipeng Chen, Xiaobo Qin, Youbin Wu, Yue Ling, Qinghao Ye, Wayne Xin Zhao, Guang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.10751)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR), which typically adopts Pass@1 as the reward, has faced the issues in balancing exploration and exploitation, causing policies to prefer conservative actions, converging to a local optimum. Identifying an appropriate reward metric is therefore crucial. Regarding the prior work, although Pass@k has been used in evaluation, its connection to LLM exploration ability in RLVR remains largely overlooked. To investigate this, we first use Pass@k as the reward to train the policy model (i.e., $\textbf{Pass@k Training}$), and observe the improvement on its exploration ability. Next, we derive an analytical solution for the advantage of Pass@k Training, leading to an efficient and effective process. Building on this, our analysis reveals that exploration and exploitation are not inherently conflicting objectives, while they can mutually enhance each other. Moreover, Pass@k Training with analytical derivation essentially involves directly designing the advantage function. Inspired by this, we preliminarily explore the advantage design for RLVR, showing promising results and highlighting a potential future direction. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）及其Pass@1奖励的存在探索和利用平衡问题，导致策略倾向于保守行为，收敛于局部最优。因此，选择合适的奖励指标至关重要。先前研究虽然使用了Pass@k进行评估，但其与RLVR中的LLM探索能力之间的联系仍被忽视。为研究这一问题，我们首先使用Pass@k作为奖励训练策略模型（即Pass@k训练），观察其探索能力的提升。接着，我们推导了Pass@k训练的优势解，从而得到了一个高效且有效的过程。基于此，我们的分析表明，探索和利用并不是固有的冲突目标，而可以相互增强。此外，带分析推导的Pass@k训练本质上涉及直接设计优势函数。受此启发，我们初步探索了RLVR的优势设计，显示出有希望的结果，并强调了未来潜在的研究方向。 

---
# REFN: A Reinforcement-Learning-From-Network Framework against 1-day/n-day Exploitations 

**Title (ZH)**: REFN：一个针对1天/多天利用的强化学习框架 

**Authors**: Tianlong Yu, Lihong Liu, Ziyi Zhou, Fudu Xing, Kailong Wang, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10701)  

**Abstract**: The exploitation of 1 day or n day vulnerabilities poses severe threats to networked devices due to massive deployment scales and delayed patching (average Mean Time To Patch exceeds 60 days). Existing defenses, including host based patching and network based filtering, are inadequate due to limited scalability across diverse devices, compatibility issues especially with embedded or legacy systems, and error prone deployment process (manual patch validation). To address these issues, we introduce REFN (Reinforcement Learning From Network), a novel framework that trains Large Language Models (LLMs) to autonomously generate network filters to prevent 1 day or n day exploitations. REFN ensures scalability by uniquely employs Reinforcement Learning (RL) driven by online network rewards instead of traditional Human Feedback (RLHF). REFN guarantees compatibility via unified deployment on edge security gateways (Amazon Eero). REFN provides robustness via online validation using real network traffic. Crucially, REFN addresses three core challenges in training LLMs for exploit prevention: 1) expanding current LLMs limited vulnerability fixing expertise via Agentic RAG based Knowledge Distillation, 2) bridging current LLMs language to network gaps through an RL From VNF Pipeline that translates language context (vulnerability description) into network enforcement, 3) addressing the LLM hallucination and non determinism via the Online Agentic Validation that penalizes erroneous outputs. Evaluated across 22 families of 1 day or n day exploits, REFN demonstrates effectiveness (21.1 percent higher accuracy than alternatives), efficiency (Mean Time To Patch of 3.65 hours) and scalability (easily scale to 10K devices). REFN serves as an initial step toward training LLMs to rapidly prevent massive scale 1 day or n day exploitations. 

**Abstract (ZH)**: 基于网络的强化学习从网络漏洞利用防护框架REFN 

---
# Learning from Natural Language Feedback for Personalized Question Answering 

**Title (ZH)**: 基于自然语言反馈的学习以实现个性化问答 

**Authors**: Alireza Salemi, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2508.10695)  

**Abstract**: Personalization is crucial for enhancing both the effectiveness and user satisfaction of language technologies, particularly in information-seeking tasks like question answering. Current approaches for personalizing large language models (LLMs) often rely on retrieval-augmented generation (RAG), followed by reinforcement learning with scalar reward signals to teach models how to use retrieved personal context. We believe that these scalar rewards sometimes provide weak, non-instructive feedback, limiting learning efficiency and personalization quality. We introduce VAC, a novel framework for personalized response generation that replaces scalar rewards with natural language feedback (NLF) that are generated conditioned on the user profiles and the question narratives. NLF serves as a rich and actionable supervision signal, allowing the policy model to iteratively refine its outputs and internalize effective personalization strategies. Training alternates between optimizing the feedback model and fine-tuning the policy model on the improved responses, resulting in a policy model that no longer requires feedback at inference. Evaluation on the LaMP-QA benchmark that consists of three diverse domains demonstrates consistent and significant improvements over the state-of-the-art results. Human evaluations further confirm the superior quality of the generated responses. These results demonstrate that NLF provides more effective signals for optimizing personalized question answering. 

**Abstract (ZH)**: 个性化对于提升语言技术的有效性和用户满意度至关重要，特别是在信息检索任务如问答中。我们提出的VAC是一种新颖的个性化响应生成框架，它用基于用户资料和问题背景生成的自然语言反馈（NLF）代替了标量奖励，作为丰富的可操作监督信号，使得策略模型能够迭代优化其输出并内化有效的个性化策略。训练过程交替优化反馈模型和在改进的响应上微调策略模型，最终使策略模型在推理时不再需要反馈。在由三个不同领域组成的LaMP-QA基准测试上的评估表明，与最佳结果相比，一致性且显著地提高了性能。人类评估进一步证实了生成响应的高质量。这些结果表明，NLF为优化个性化问答提供了更有效的信号。 

---
# When Language Overrules: Revealing Text Dominance in Multimodal Large Language Models 

**Title (ZH)**: 当语言占据主导：揭示多模态大型语言模型中的文本主导性 

**Authors**: Huyu Wu, Meng Tang, Xinhan Zheng, Haiyun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10552)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities across a diverse range of multimodal tasks. However, these models suffer from a core problem known as text dominance: they depend heavily on text for their inference, while underutilizing other modalities. While prior work has acknowledged this phenomenon in vision-language tasks, often attributing it to data biases or model architectures. In this paper, we conduct the first systematic investigation of text dominance across diverse data modalities, including images, videos, audio, time-series, and graphs. To measure this imbalance, we propose two evaluation metrics: the Modality Dominance Index (MDI) and the Attention Efficiency Index (AEI). Our comprehensive analysis reveals that text dominance is both significant and pervasive across all tested modalities. Our in-depth analysis identifies three underlying causes: attention dilution from severe token redundancy in non-textual modalities, the influence of fusion architecture design, and task formulations that implicitly favor textual inputs. Furthermore, we propose a simple token compression method that effectively rebalances model attention. Applying this method to LLaVA-7B, for instance, drastically reduces its MDI from 10.23 to a well-balanced value of 0.86. Our analysis and methodological framework offer a foundation for the development of more equitable and comprehensive multimodal language models. 

**Abstract (ZH)**: 多模态大语言模型（MLLMs）在多种多模态任务中展现了出色的性能。然而，这些模型面临着一个核心问题：文本主导性，即它们过于依赖文本进行推理，而未能充分利用其他模态信息。尽管先前的研究已认识到这一现象在视觉-语言任务中的存在，并通常将其归因于数据偏差或模型架构的问题。在本文中，我们首次对文本主导性在图像、视频、音频、时间序列和图形等多种数据模态中的现象进行了系统的探讨。为了衡量这种不平衡，我们提出了两个评估指标：模态主导性指数（MDI）和注意力效率指数（AEI）。全面的分析表明，文本主导性在所有测试的模态中都具有显著性和普遍性。深入分析揭示了三个根本原因：非文本模态中严重标记冗余导致的注意力稀释、融合架构设计的影响以及隐式偏向文本输入的任务表述。此外，我们提出了一种简单的标记压缩方法，其能有效重新平衡模型的注意力。例如，将该方法应用于LLaVA-7B模型时，其MDI从10.23显著降低到了均衡值0.86。我们的分析和方法论框架为开发更加公平和全面的多模态语言模型提供了基础。 

---
# ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning 

**Title (ZH)**: CogRAG：一种基于认知的内存组织化RAG框架，用于状态保持长叙事推理 

**Authors**: Juyuan Wang, Rongchen Zhao, Wei Wei, Yufeng Wang, Mo Yu, Jie Zhou, Jin Xu, Liyan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10419)  

**Abstract**: Narrative comprehension on long stories and novels has been a challenging domain attributed to their intricate plotlines and entangled, often evolving relations among characters and entities. Given the LLM's diminished reasoning over extended context and high computational cost, retrieval-based approaches remain a pivotal role in practice. However, traditional RAG methods can fall short due to their stateless, single-step retrieval process, which often overlooks the dynamic nature of capturing interconnected relations within long-range context. In this work, we propose ComoRAG, holding the principle that narrative reasoning is not a one-shot process, but a dynamic, evolving interplay between new evidence acquisition and past knowledge consolidation, analogous to human cognition when reasoning with memory-related signals in the brain. Specifically, when encountering a reasoning impasse, ComoRAG undergoes iterative reasoning cycles while interacting with a dynamic memory workspace. In each cycle, it generates probing queries to devise new exploratory paths, then integrates the retrieved evidence of new aspects into a global memory pool, thereby supporting the emergence of a coherent context for the query resolution. Across four challenging long-context narrative benchmarks (200K+ tokens), ComoRAG outperforms strong RAG baselines with consistent relative gains up to 11% compared to the strongest baseline. Further analysis reveals that ComoRAG is particularly advantageous for complex queries requiring global comprehension, offering a principled, cognitively motivated paradigm for retrieval-based long context comprehension towards stateful reasoning. Our code is publicly released at this https URL 

**Abstract (ZH)**: 长篇故事和小说的叙述理解是一个具有挑战性的领域，归因于其复杂的故事情节和人物及实体之间错综复杂的、常常不断演变的关系。鉴于大语言模型在长时间上下文推理中的削弱表现和高计算成本，检索式方法在实践中仍然发挥着关键作用。然而，传统的RAG方法由于其无状态的、单步的检索过程，往往忽略了在长期上下文中捕捉互联关系的动态性质。在本文中，我们提出了ComoRAG，其原则是叙述推理不是一个一次性过程，而是一个在新证据获取与过去知识整合之间动态、演化的互动过程，类似于大脑在使用与记忆相关信号进行推理时的人类认知。具体而言，当遇到推理瓶颈时，ComoRAG会通过与动态记忆工作空间的交互进行迭代推理循环。在每个循环中，它生成探查查询以设计新的探索路径，然后将新方面的检索证据整合到全局记忆池中，从而支持查询解决的连贯上下文的生成。在四个具有挑战性的长上下文叙述基准测试中（包含200K+词），ComoRAG在与最强基线相比的情况下，一致获得了最高11%的相对增益成绩。进一步的分析表明，ComoRAG特别适用于需要全局理解的复杂查询，提供了一种基于检索、以状态密集推理为目标的认知驱动的框架。我们的代码已在此 https://公开发布。 

---
# MCP2OSC: Parametric Control by Natural Language 

**Title (ZH)**: MCP参数化OSC：基于自然语言的参数控制 

**Authors**: Yuan-Yi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2508.10414)  

**Abstract**: Text prompts enable intuitive content creation but may fall short in achieving high precision for intricate tasks; knob or slider controls offer precise adjustments at the cost of increased complexity. To address the gap between knobs and prompts, a new MCP (Model Context Protocol) server and a unique set of prompt design criteria are presented to enable exploring parametric OSC (OpenSoundControl) control by natural language prompts. Demonstrated by 14 practical QA examples with best practices and the generalized prompt templates, this study finds Claude integrated with the MCP2OSC server effective in generating OSC messages by natural language, interpreting, searching, and visualizing OSC messages, validating and debugging OSC messages, and managing OSC address patterns. MCP2OSC enhances human-machine collaboration by leveraging LLM (Large Language Model) to handle intricate OSC development tasks, and by empowering human creativity with an intuitive language interface featuring flexible precision controls: a prompt-based OSC tool. This study provides a novel perspective on the creative MCP application at the network protocol level by utilizing LLM's strength in directly processing and generating human-readable OSC messages. The results suggest its potential for a LLM-based universal control mechanism for multimedia devices. 

**Abstract (ZH)**: 基于LLM的MCP2OSC服务器在自然语言控制OpenSoundControl消息中的应用研究 

---
# AnalogSeeker: An Open-source Foundation Language Model for Analog Circuit Design 

**Title (ZH)**: AnalogSeeker: 开源基础语言模型用于模拟电路设计 

**Authors**: Zihao Chen, Ji Zhuang, Jinyi Shen, Xiaoyue Ke, Xinyi Yang, Mingjie Zhou, Zhuoyao Du, Xu Yan, Zhouyang Wu, Zhenyu Xu, Jiangli Huang, Li Shang, Xuan Zeng, Fan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10409)  

**Abstract**: In this paper, we propose AnalogSeeker, an effort toward an open-source foundation language model for analog circuit design, with the aim of integrating domain knowledge and giving design assistance. To overcome the scarcity of data in this field, we employ a corpus collection strategy based on the domain knowledge framework of analog circuits. High-quality, accessible textbooks across relevant subfields are systematically curated and cleaned into a textual domain corpus. To address the complexity of knowledge of analog circuits, we introduce a granular domain knowledge distillation method. Raw, unlabeled domain corpus is decomposed into typical, granular learning nodes, where a multi-agent framework distills implicit knowledge embedded in unstructured text into question-answer data pairs with detailed reasoning processes, yielding a fine-grained, learnable dataset for fine-tuning. To address the unexplored challenges in training analog circuit foundation models, we explore and share our training methods through both theoretical analysis and experimental validation. We finally establish a fine-tuning-centric training paradigm, customizing and implementing a neighborhood self-constrained supervised fine-tuning algorithm. This approach enhances training outcomes by constraining the perturbation magnitude between the model's output distributions before and after training. In practice, we train the Qwen2.5-32B-Instruct model to obtain AnalogSeeker, which achieves 85.04% accuracy on AMSBench-TQA, the analog circuit knowledge evaluation benchmark, with a 15.67% point improvement over the original model and is competitive with mainstream commercial models. Furthermore, AnalogSeeker also shows effectiveness in the downstream operational amplifier design task. AnalogSeeker is open-sourced at this https URL for research use. 

**Abstract (ZH)**: AnalogSeeker: 面向模拟电路设计的开源基础语言模型及其应用 

---
# Layer-Wise Perturbations via Sparse Autoencoders for Adversarial Text Generation 

**Title (ZH)**: 基于稀疏自编码器的分层扰动生成对抗性文本 

**Authors**: Huizhen Shu, Xuying Li, Qirui Wang, Yuji Kosuga, Mengqiu Tian, Zhuo Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.10404)  

**Abstract**: With the rapid proliferation of Natural Language Processing (NLP), especially Large Language Models (LLMs), generating adversarial examples to jailbreak LLMs remains a key challenge for understanding model vulnerabilities and improving robustness. In this context, we propose a new black-box attack method that leverages the interpretability of large models. We introduce the Sparse Feature Perturbation Framework (SFPF), a novel approach for adversarial text generation that utilizes sparse autoencoders to identify and manipulate critical features in text. After using the SAE model to reconstruct hidden layer representations, we perform feature clustering on the successfully attacked texts to identify features with higher activations. These highly activated features are then perturbed to generate new adversarial texts. This selective perturbation preserves the malicious intent while amplifying safety signals, thereby increasing their potential to evade existing defenses. Our method enables a new red-teaming strategy that balances adversarial effectiveness with safety alignment. Experimental results demonstrate that adversarial texts generated by SFPF can bypass state-of-the-art defense mechanisms, revealing persistent vulnerabilities in current NLP this http URL, the method's effectiveness varies across prompts and layers, and its generalizability to other architectures and larger models remains to be validated. 

**Abstract (ZH)**: 基于稀疏特征扰动框架的大型语言模型黑盒攻击方法 

---
# Yet another algorithmic bias: A Discursive Analysis of Large Language Models Reinforcing Dominant Discourses on Gender and Race 

**Title (ZH)**: 另一种算法偏见：关于性别和种族主导论述的大语言模型话语分析 

**Authors**: Gustavo Bonil, Simone Hashiguti, Jhessica Silva, João Gondim, Helena Maia, Nádia Silva, Helio Pedrini, Sandra Avila  

**Link**: [PDF](https://arxiv.org/pdf/2508.10304)  

**Abstract**: With the advance of Artificial Intelligence (AI), Large Language Models (LLMs) have gained prominence and been applied in diverse contexts. As they evolve into more sophisticated versions, it is essential to assess whether they reproduce biases, such as discrimination and racialization, while maintaining hegemonic discourses. Current bias detection approaches rely mostly on quantitative, automated methods, which often overlook the nuanced ways in which biases emerge in natural language. This study proposes a qualitative, discursive framework to complement such methods. Through manual analysis of LLM-generated short stories featuring Black and white women, we investigate gender and racial biases. We contend that qualitative methods such as the one proposed here are fundamental to help both developers and users identify the precise ways in which biases manifest in LLM outputs, thus enabling better conditions to mitigate them. Results show that Black women are portrayed as tied to ancestry and resistance, while white women appear in self-discovery processes. These patterns reflect how language models replicate crystalized discursive representations, reinforcing essentialization and a sense of social immobility. When prompted to correct biases, models offered superficial revisions that maintained problematic meanings, revealing limitations in fostering inclusive narratives. Our results demonstrate the ideological functioning of algorithms and have significant implications for the ethical use and development of AI. The study reinforces the need for critical, interdisciplinary approaches to AI design and deployment, addressing how LLM-generated discourses reflect and perpetuate inequalities. 

**Abstract (ZH)**: 随着人工智能的进步，大型语言模型已在多种情境中获得 prominence 并得到应用。随着它们演化成更为复杂的新版本，评估它们在保持霸权话语的同时是否再现了偏见（如歧视和种族化）变得尤为重要。当前的偏见检测方法主要依赖于定量、自动化的手段，往往忽略了偏见在自然语言中复杂的表现方式。本研究提出了一种定性的、话语框架来补充这些方法。通过对手动分析生成的涉及黑人和白人女性的小说，我们探讨性别和种族偏见。我们认为，如本研究提出的方法这样的定性方法是帮助开发者和用户识别大型语言模型输出中偏见具体表现方式的基础，从而有助于更好地条件来减轻这些偏见。结果表明，黑人女性常被描绘为与祖先和抵抗相关，而白人女性则出现在自我发现的过程中。这些模式反映了语言模型如何再现固化的话语表征，强化本质化并维持一种社会流动性感的假象。当被要求纠正偏见时，模型提供的修改往往是表面化的，保留了有问题的含义，揭示了促进包容叙事的局限性。我们的研究结果揭示了算法的意识形态功能，并对人工智能的伦理使用和发展具有重要意义。本研究强调了在人工智能设计和部署中采用批判性和跨学科方法的必要性，关注LLM生成的话语如何反映并延续不平等。 

---
# MRFD: Multi-Region Fusion Decoding with Self-Consistency for Mitigating Hallucinations in LVLMs 

**Title (ZH)**: MRFD：多区域融合解码与自一致性方法减轻大语言模型中的幻觉问题 

**Authors**: Haonan Ge, Yiwei Wang, Ming-Hsuan Yang, Yujun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2508.10264)  

**Abstract**: Large Vision-Language Models (LVLMs) have shown strong performance across multimodal tasks. However, they often produce hallucinations -- text that is inconsistent with visual input, due to the limited ability to verify information in different regions of the image. To address this, we propose Multi-Region Fusion Decoding (MRFD), a training-free decoding method that improves factual grounding by modeling inter-region consistency. MRFD identifies salient regions using cross-attention, generates initial responses for each, and computes reliability weights based on Jensen-Shannon Divergence (JSD) among the responses. These weights guide a consistency-aware fusion of per-region predictions, using region-aware prompts inspired by Chain-of-Thought reasoning. Experiments across multiple LVLMs and benchmarks show that MRFD significantly reduces hallucinations and improves response factuality without requiring model updates. 

**Abstract (ZH)**: 大规模多模态语言视觉模型（LVLMs）在多模态任务中表现出强大的性能。然而，它们通常会产生幻觉——与视觉输入不一致的文本，这是因为模型在验证图像不同区域的信息方面的能力有限。为了解决这一问题，我们提出了一种无训练的解码方法——多区域融合解码（MRFD），通过建模不同区域之间的一致性来增强事实 grounding。MRFD 使用交叉注意力识别显著区域，为每个区域生成初始响应，并基于响应之间的 Jensen-Shannon 散度（JSD）计算可靠性权重。这些权重引导一种基于区域意识的融合过程，该过程使用借鉴了思维链推理的区域意识提示，关注一致性。跨多个 LVLM 和基准的实验表明，MRFD 显著减少了幻觉并提高了响应的事实性，而无需对模型进行更新。 

---
# Using Large Language Models to Measure Symptom Severity in Patients At Risk for Schizophrenia 

**Title (ZH)**: 使用大型语言模型测量 schizophrenia 高风险患者症状严重程度 

**Authors**: Andrew X. Chen, Guillermo Horga, Sean Escola  

**Link**: [PDF](https://arxiv.org/pdf/2508.10226)  

**Abstract**: Patients who are at clinical high risk (CHR) for schizophrenia need close monitoring of their symptoms to inform appropriate treatments. The Brief Psychiatric Rating Scale (BPRS) is a validated, commonly used research tool for measuring symptoms in patients with schizophrenia and other psychotic disorders; however, it is not commonly used in clinical practice as it requires a lengthy structured interview. Here, we utilize large language models (LLMs) to predict BPRS scores from clinical interview transcripts in 409 CHR patients from the Accelerating Medicines Partnership Schizophrenia (AMP-SCZ) cohort. Despite the interviews not being specifically structured to measure the BPRS, the zero-shot performance of the LLM predictions compared to the true assessment (median concordance: 0.84, ICC: 0.73) approaches human inter- and intra-rater reliability. We further demonstrate that LLMs have substantial potential to improve and standardize the assessment of CHR patients via their accuracy in assessing the BPRS in foreign languages (median concordance: 0.88, ICC: 0.70), and integrating longitudinal information in a one-shot or few-shot learning approach. 

**Abstract (ZH)**: 临床高风险（CHR）患者需要密切监测其症状以指导适当治疗。我们利用大规模语言模型（LLMs）从加速药物开发精神分裂症（AMP-SCZ）队列中的409名CHR患者访谈记录中预测BPRS评分。尽管访谈并非专门设计用于测量BPRS，但LLM预测与真实评估（中位一致性：0.84，ICC：0.73）的人际和自我评定可靠性相近。进一步研究表明，LLM在通过评估外语文本中的BPRS提高和标准化CHR患者的评估方面具有巨大的潜力（中位一致性：0.88，ICC：0.70），并能通过一-shot或few-shot学习整合纵向信息。 

---
# Prompt-Response Semantic Divergence Metrics for Faithfulness Hallucination and Misalignment Detection in Large Language Models 

**Title (ZH)**: 大型语言模型中幻觉和错配检测的提示-响应语义发散度度量 

**Authors**: Igor Halperin  

**Link**: [PDF](https://arxiv.org/pdf/2508.10192)  

**Abstract**: The proliferation of Large Language Models (LLMs) is challenged by hallucinations, critical failure modes where models generate non-factual, nonsensical or unfaithful text. This paper introduces Semantic Divergence Metrics (SDM), a novel lightweight framework for detecting Faithfulness Hallucinations -- events of severe deviations of LLMs responses from input contexts. We focus on a specific implementation of these LLM errors, {confabulations, defined as responses that are arbitrary and semantically misaligned with the user's query. Existing methods like Semantic Entropy test for arbitrariness by measuring the diversity of answers to a single, fixed prompt. Our SDM framework improves upon this by being more prompt-aware: we test for a deeper form of arbitrariness by measuring response consistency not only across multiple answers but also across multiple, semantically-equivalent paraphrases of the original prompt. Methodologically, our approach uses joint clustering on sentence embeddings to create a shared topic space for prompts and answers. A heatmap of topic co-occurances between prompts and responses can be viewed as a quantified two-dimensional visualization of the user-machine dialogue. We then compute a suite of information-theoretic metrics to measure the semantic divergence between prompts and responses. Our practical score, $\mathcal{S}_H$, combines the Jensen-Shannon divergence and Wasserstein distance to quantify this divergence, with a high score indicating a Faithfulness hallucination. Furthermore, we identify the KL divergence KL(Answer $||$ Prompt) as a powerful indicator of \textbf{Semantic Exploration}, a key signal for distinguishing different generative behaviors. These metrics are further combined into the Semantic Box, a diagnostic framework for classifying LLM response types, including the dangerous, confident confabulation. 

**Abstract (ZH)**: 大型语言模型（LLMs）的 proliferations 被幻觉挑战，幻觉是指模型生成非事实、无意义或不忠实的文本的关键失败模式。本文引入了语义发散度度量（SDM），这是一种新型轻量级框架，用于检测忠实性幻觉——LLMs 响应与输入上下文严重偏离的事件。我们专注于这些 LLM 错误的特定实现，即编纂，定义为与用户查询语义不匹配的任意响应。现有的方法如语义熵通过测量单个固定提示下的答案多样性来测试任意性。我们的 SDM 框架通过更关注提示来改进这一点：我们不仅通过测量跨多个答案的一致性，还通过测量跨多个语义等价的提示重述的答案的一致性来测试更深层次的任意性。从方法上讲，我们的方法使用句子嵌入的联合聚类为提示和答案创建共同主题空间。提示和响应之间主题共现的热图可以作为用户-机器对话的量化二维可视化。然后计算一系列信息论度量来衡量提示和响应之间的语义发散度。我们的实用得分 $\mathcal{S}_H$ 结合了 Jensen-Shannon 散度和 Wasserstein 距离来量化这种发散度，高分表明是忠实性幻觉。此外，我们确定了 KL 散度 KL(Answer $||$ Prompt) 作为语义探索的关键指标，这是区分不同生成行为的重要信号。这些指标进一步结合形成了语义盒，这是一种诊断框架，用于分类 LLM 响应类型，包括危险的、自信的编纂。 

---
# PakBBQ: A Culturally Adapted Bias Benchmark for QA 

**Title (ZH)**: PakBBQ：一个文化适应性的偏见基准数据集for QA 

**Authors**: Abdullah Hashmat, Muhammad Arham Mirza, Agha Ali Raza  

**Link**: [PDF](https://arxiv.org/pdf/2508.10186)  

**Abstract**: With the widespread adoption of Large Language Models (LLMs) across various applications, it is empirical to ensure their fairness across all user communities. However, most LLMs are trained and evaluated on Western centric data, with little attention paid to low-resource languages and regional contexts. To address this gap, we introduce PakBBQ, a culturally and regionally adapted extension of the original Bias Benchmark for Question Answering (BBQ) dataset. PakBBQ comprises over 214 templates, 17180 QA pairs across 8 categories in both English and Urdu, covering eight bias dimensions including age, disability, appearance, gender, socio-economic status, religious, regional affiliation, and language formality that are relevant in Pakistan. We evaluate multiple multilingual LLMs under both ambiguous and explicitly disambiguated contexts, as well as negative versus non negative question framings. Our experiments reveal (i) an average accuracy gain of 12\% with disambiguation, (ii) consistently stronger counter bias behaviors in Urdu than in English, and (iii) marked framing effects that reduce stereotypical responses when questions are posed negatively. These findings highlight the importance of contextualized benchmarks and simple prompt engineering strategies for bias mitigation in low resource settings. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在各种应用中的广泛应用，确保其在所有用户社区中的公平性是实证的。然而，大多数LLMs都是基于以西方为中心的数据进行训练和评估的，很少关注低资源语言和区域背景。为解决这一问题，我们介绍了PakBBQ，这是一个文化上和区域上适应扩展的原始问答偏见基准（BBQ）数据集。PakBBQ 包含超过214个模板，覆盖8个类别中的17180个QA对，其中包括英语和乌尔都语，涵盖了八个偏见维度，如年龄、残疾、外貌、性别、社会经济地位、宗教、区域归属和语言正式性，这些维度在巴基斯坦特别相关。我们在模棱两可和明确消歧情况下评估了多种多语言LLMs，并且考察了负面和非负面问题表述的效果。实验结果显示，(i) 消歧情况下平均准确率提高了12%，(ii) 与英语相比，乌尔都语中始终表现出更强的反向偏见行为，(iii) 问题负面表述显著减少了刻板印象的回答。这些发现强调了在低资源环境中，采用上下文化基准和简单的提示 engineering 策略对偏见缓解的重要性。 

---
# LaajMeter: A Framework for LaaJ Evaluation 

**Title (ZH)**: LaajMeter：一种Laaj评估框架 

**Authors**: Gal Amram, Eitan Farchi, Shmulik Froimovich, Raviv Gal, Avi Ziv  

**Link**: [PDF](https://arxiv.org/pdf/2508.10161)  

**Abstract**: Large Language Models (LLMs) are increasingly used as evaluators in natural language processing tasks, a paradigm known as LLM-as-a-Judge (LaaJ). While effective in general domains, LaaJs pose significant challenges in domain-specific contexts, where annotated data is scarce and expert evaluation is costly. In such cases, meta-evaluation is often performed using metrics that have not been validated for the specific domain in which they are applied. As a result, it becomes difficult to determine which metrics effectively identify LaaJ quality, and further, what threshold indicates sufficient evaluator performance. In this work, we introduce LaaJMeter, a simulation-based framework for controlled meta-evaluation of LaaJs. LaaJMeter enables engineers to generate synthetic data representing virtual models and judges, allowing systematic analysis of evaluation metrics under realistic conditions. This helps practitioners validate and refine LaaJs for specific evaluation tasks: they can test whether their metrics correctly distinguish between better and worse (virtual) LaaJs, and estimate appropriate thresholds for evaluator adequacy.
We demonstrate the utility of LaaJMeter in a code translation task involving a legacy programming language, showing how different metrics vary in sensitivity to evaluator quality. Our results highlight the limitations of common metrics and the importance of principled metric selection. LaaJMeter provides a scalable and extensible solution for assessing LaaJs in low-resource settings, contributing to the broader effort to ensure trustworthy and reproducible evaluation in NLP. 

**Abstract (ZH)**: 基于大型语言模型的评判器评估框架：LaaJMeter 

---
# mSCoRe: a $M$ultilingual and Scalable Benchmark for $S$kill-based $Co$mmonsense $Re$asoning 

**Title (ZH)**: mSCoRe：一种面向技能驱动的多M语言和可Sscalable基线 

**Authors**: Nghia Trung Ngo, Franck Dernoncourt, Thien Huu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10137)  

**Abstract**: Recent advancements in reasoning-reinforced Large Language Models (LLMs) have shown remarkable capabilities in complex reasoning tasks. However, the mechanism underlying their utilization of different human reasoning skills remains poorly investigated, especially for multilingual commonsense reasoning that involves everyday knowledge across different languages and cultures. To address this gap, we propose a \textbf{M}ultilingual and Scalable Benchmark for \textbf{S}kill-based \textbf{Co}mmonsense \textbf{Re}asoning (\textbf{mSCoRe}). Our benchmark incorporates three key components that are designed to systematically evaluate LLM's reasoning capabilities, including: (1) a novel taxonomy of reasoning skills that enables fine-grained analysis of models' reasoning processes, (2) a robust data synthesis pipeline tailored specifically for commonsense reasoning evaluation, and (3) a complexity scaling framework allowing task difficulty to scale dynamically alongside future improvements in LLM abilities. Extensive experiments on eights state-of-the-art LLMs of varying sizes and training approaches demonstrate that \textbf{mSCoRe} remains significantly challenging for current models, particularly at higher complexity levels. Our results reveal the limitations of such reasoning-reinforced models when confronted with nuanced multilingual general and cultural commonsense. We further provide detailed analysis on the models' reasoning processes, suggesting future directions for improving multilingual commonsense reasoning capabilities. 

**Abstract (ZH)**: 多语言可扩展技能编目常识推理基准（mSCoRe） 

---
# Nested-ReFT: Efficient Reinforcement Learning for Large Language Model Fine-Tuning via Off-Policy Rollouts 

**Title (ZH)**: Nested-ReFT: 通过离策rollout高效微调大型语言模型的强化学习方法 

**Authors**: Maxime Heuillet, Yufei Cui, Boxing Chen, Audrey Durand, Prasanna Parthasarathi  

**Link**: [PDF](https://arxiv.org/pdf/2508.10123)  

**Abstract**: Advanced reasoning in LLMs on challenging domains like mathematical reasoning can be tackled using verifiable rewards based reinforced fine-tuning (ReFT). In standard ReFT frameworks, a behavior model generates multiple completions with answers per problem, for the answer to be then scored by a reward function. While such RL post-training methods demonstrate significant performance improvements across challenging reasoning domains, the computational cost of generating completions during training with multiple inference steps makes the training cost non-trivial. To address this, we draw inspiration from off-policy RL, and speculative decoding to introduce a novel ReFT framework, dubbed Nested-ReFT, where a subset of layers of the target model acts as the behavior model to generate off-policy completions during training. The behavior model configured with dynamic layer skipping per batch during training decreases the inference cost compared to the standard ReFT frameworks. Our theoretical analysis shows that Nested-ReFT yields unbiased gradient estimates with controlled variance. Our empirical analysis demonstrates improved computational efficiency measured as tokens/sec across multiple math reasoning benchmarks and model sizes. Additionally, we explore three variants of bias mitigation to minimize the off-policyness in the gradient updates that allows for maintaining performance that matches the baseline ReFT performance. 

**Abstract (ZH)**: 基于可验证奖励的强化微调（ReFT）在处理数学推理解等具有挑战性的领域中的高级推理可以得到解决。Nested-ReFT：一种新颖的强化微调框架及其在数学推理任务中的应用 

---
# Less is More: Learning Graph Tasks with Just LLMs 

**Title (ZH)**: 少即是多：仅使用语言模型学习图任务 

**Authors**: Sola Shirai, Kavitha Srinivas, Julian Dolby, Michael Katz, Horst Samulowitz, Shirin Sohrabi  

**Link**: [PDF](https://arxiv.org/pdf/2508.10115)  

**Abstract**: For large language models (LLMs), reasoning over graphs could help solve many problems. Prior work has tried to improve LLM graph reasoning by examining how best to serialize graphs as text and by combining GNNs and LLMs. However, the merits of such approaches remain unclear, so we empirically answer the following research questions: (1) Can LLMs learn to solve fundamental graph tasks without specialized graph encoding models?, (2) Can LLMs generalize learned solutions to unseen graph structures or tasks?, and (3) What are the merits of competing approaches to learn graph tasks? We show that even small LLMs can learn to solve graph tasks by training them with instructive chain-of-thought solutions, and this training generalizes, without specialized graph encoders, to new tasks and graph structures. 

**Abstract (ZH)**: 大型语言模型（LLMs）在图推理方面的研究：小型LLMs通过链式思考训练学习图任务的优势 

---
# Large Language Models Show Signs of Alignment with Human Neurocognition During Abstract Reasoning 

**Title (ZH)**: 大型语言模型在抽象推理过程中表现出与人类神经认知的对齐迹象 

**Authors**: Christopher Pinier, Sonia Acuña Vargas, Mariia Steeghs-Turchina, Dora Matzke, Claire E. Stevenson, Michael D. Nunez  

**Link**: [PDF](https://arxiv.org/pdf/2508.10057)  

**Abstract**: This study investigates whether large language models (LLMs) mirror human neurocognition during abstract reasoning. We compared the performance and neural representations of human participants with those of eight open-source LLMs on an abstract-pattern-completion task. We leveraged pattern type differences in task performance and in fixation-related potentials (FRPs) as recorded by electroencephalography (EEG) during the task. Our findings indicate that only the largest tested LLMs (~70 billion parameters) achieve human-comparable accuracy, with Qwen-2.5-72B and DeepSeek-R1-70B also showing similarities with the human pattern-specific difficulty profile. Critically, every LLM tested forms representations that distinctly cluster the abstract pattern categories within their intermediate layers, although the strength of this clustering scales with their performance on the task. Moderate positive correlations were observed between the representational geometries of task-optimal LLM layers and human frontal FRPs. These results consistently diverged from comparisons with other EEG measures (response-locked ERPs and resting EEG), suggesting a potential shared representational space for abstract patterns. This indicates that LLMs might mirror human brain mechanisms in abstract reasoning, offering preliminary evidence of shared principles between biological and artificial intelligence. 

**Abstract (ZH)**: 本研究调查大型语言模型在抽象推理过程中是否镜像人类神经认知。我们通过一项抽象模式完成任务，将人类参与者的性能和神经表示与八个开源大型语言模型进行了比较，并利用任务中不同模式类型的性能差异以及脑电图（EEG）记录的注意焦点相关电位（FRP）的差异进行分析。研究发现，只有最大的测试模型（约700亿参数）达到与人类相当的准确度，同时Qwen-2.5-72B和DeepSeek-R1-70B也显示出与人类模式特定难度特征的相似性。关键的是，每个测试的模型在其中间层中都形成了将抽象模式类别区分开来的表示，虽然这种区别的强度与它们在任务上的表现成比例。任务优化的大型语言模型层的表示几何与人类前额叶的FRP之间存在中等正相关。这些结果与与其他EEG测量（反应锁定的ERM和静息EEG）的比较不一致，表明可能存在一种抽象模式的共享表示空间。这表明大型语言模型可能在抽象推理中镜像人类大脑机制，为生物智能和人工智能共享原则提供了初步证据。 

---
# Reflect then Learn: Active Prompting for Information Extraction Guided by Introspective Confusion 

**Title (ZH)**: 反思then学习：由 introspective confusion驱动的主动提示引导信息提取 

**Authors**: Dong Zhao, Yadong Wang, Xiang Chen, Chenxi Wang, Hongliang Dai, Chuanxing Geng, Shengzhong Zhang, Shaoyuan Li, Sheng-Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10036)  

**Abstract**: Large Language Models (LLMs) show remarkable potential for few-shot information extraction (IE), yet their performance is highly sensitive to the choice of in-context examples. Conventional selection strategies often fail to provide informative guidance, as they overlook a key source of model fallibility: confusion stemming not just from semantic content, but also from the generation of well-structured formats required by IE tasks. To address this, we introduce Active Prompting for Information Extraction (APIE), a novel active prompting framework guided by a principle we term introspective confusion. Our method empowers an LLM to assess its own confusion through a dual-component uncertainty metric that uniquely quantifies both Format Uncertainty (difficulty in generating correct syntax) and Content Uncertainty (inconsistency in extracted semantics). By ranking unlabeled data with this comprehensive score, our framework actively selects the most challenging and informative samples to serve as few-shot exemplars. Extensive experiments on four benchmarks show that our approach consistently outperforms strong baselines, yielding significant improvements in both extraction accuracy and robustness. Our work highlights the critical importance of a fine-grained, dual-level view of model uncertainty when it comes to building effective and reliable structured generation systems. 

**Abstract (ZH)**: 基于主动提示的信息提取中的反省混淆（APIE）：细粒度双层建模模式不确定性以提升少样本信息提取性能 

---
# The Cost of Thinking: Increased Jailbreak Risk in Large Language Models 

**Title (ZH)**: 思考的成本：大型语言模型中 Jailbreak 风险的增加 

**Authors**: Fan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10032)  

**Abstract**: Thinking mode has always been regarded as one of the most valuable modes in LLMs. However, we uncover a surprising and previously overlooked phenomenon: LLMs with thinking mode are more easily broken by Jailbreak attack. We evaluate 9 LLMs on AdvBench and HarmBench and find that the success rate of attacking thinking mode in LLMs is almost higher than that of non-thinking mode. Through large numbers of sample studies, it is found that for educational purposes and excessively long thinking lengths are the characteristics of successfully attacked data, and LLMs also give harmful answers when they mostly know that the questions are harmful. In order to alleviate the above problems, this paper proposes a method of safe thinking intervention for LLMs, which explicitly guides the internal thinking processes of LLMs by adding "specific thinking tokens" of LLMs to the prompt. The results demonstrate that the safe thinking intervention can significantly reduce the attack success rate of LLMs with thinking mode. 

**Abstract (ZH)**: LLMs中思考模式易于受到 Jailbreak 攻击的现象及安全干预方法 

---
# Context Misleads LLMs: The Role of Context Filtering in Maintaining Safe Alignment of LLMs 

**Title (ZH)**: Context误导LLMs：Context过滤在保持LLMs安全对齐中的作用 

**Authors**: Jinhwa Kim, Ian G. Harris  

**Link**: [PDF](https://arxiv.org/pdf/2508.10031)  

**Abstract**: While Large Language Models (LLMs) have shown significant advancements in performance, various jailbreak attacks have posed growing safety and ethical risks. Malicious users often exploit adversarial context to deceive LLMs, prompting them to generate responses to harmful queries. In this study, we propose a new defense mechanism called Context Filtering model, an input pre-processing method designed to filter out untrustworthy and unreliable context while identifying the primary prompts containing the real user intent to uncover concealed malicious intent. Given that enhancing the safety of LLMs often compromises their helpfulness, potentially affecting the experience of benign users, our method aims to improve the safety of the LLMs while preserving their original performance. We evaluate the effectiveness of our model in defending against jailbreak attacks through comparative analysis, comparing our approach with state-of-the-art defense mechanisms against six different attacks and assessing the helpfulness of LLMs under these defenses. Our model demonstrates its ability to reduce the Attack Success Rates of jailbreak attacks by up to 88% while maintaining the original LLMs' performance, achieving state-of-the-art Safety and Helpfulness Product results. Notably, our model is a plug-and-play method that can be applied to all LLMs, including both white-box and black-box models, to enhance their safety without requiring any fine-tuning of the models themselves. We will make our model publicly available for research purposes. 

**Abstract (ZH)**: 大型语言模型的安全防护：基于上下文过滤的新型防御机制及其实效评估 

---
# Inference-Aware Prompt Optimization for Aligning Black-Box Large Language Models 

**Title (ZH)**: 面向推理的认知黑盒大规模语言模型对齐提示优化 

**Authors**: Saaduddin Mahmud, Mason Nakamura, Kyle H. Wray, Shlomo Zilberstein  

**Link**: [PDF](https://arxiv.org/pdf/2508.10030)  

**Abstract**: Prompt optimization methods have demonstrated significant effectiveness in aligning black-box large language models (LLMs). In parallel, inference scaling strategies such as Best-of-N Sampling and Majority Voting have also proven to enhance alignment and performance by trading off computation. However, existing prompt optimization approaches are inference strategy agnostic; that is, they optimize prompts without regard to the inference strategy employed during deployment. This constitutes a significant methodological gap, as our empirical and theoretical analysis reveals a strong interdependence between these two paradigms. Moreover, we find that user preferences regarding trade-offs among multiple objectives and inference budgets substantially influence the choice of prompt and inference configuration. To address this gap, we introduce a unified novel framework named IAPO (Inference-Aware Prompt Optimization) that jointly optimizes the prompt and inference scale, while being aware of the inference budget and different task objectives. We then develop a fixed-budget training algorithm for IAPO, which we call PSST (Prompt Scaling via Sequential Trimming), and analyze finite-budget guarantees on error probability. Finally, we evaluate the effectiveness of PSST on six different tasks, including multi-objective text generation and reasoning, and demonstrate the critical role of incorporating inference-awareness when aligning black-box LLMs through prompt optimization. 

**Abstract (ZH)**: 基于推理aware的提示优化方法：IAPO（推理意识提示优化） 

---
# Latent Fusion Jailbreak: Blending Harmful and Harmless Representations to Elicit Unsafe LLM Outputs 

**Title (ZH)**: 潜藏融合突破：混合有害和无害表示以诱发不安全的LLM输出 

**Authors**: Wenpeng Xing, Mohan Li, Chunqiang Hu, Haitao XuNingyu Zhang, Bo Lin, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.10029)  

**Abstract**: Large language models (LLMs) demonstrate impressive capabilities in various language tasks but are susceptible to jailbreak attacks that circumvent their safety alignments. This paper introduces Latent Fusion Jailbreak (LFJ), a representation-based attack that interpolates hidden states from harmful and benign query pairs to elicit prohibited responses. LFJ begins by selecting query pairs with high thematic and syntactic similarity, then performs gradient-guided interpolation at influential layers and tokens, followed by optimization to balance attack success, output fluency, and computational efficiency. Evaluations on models such as Vicuna and LLaMA-2 across benchmarks like AdvBench and MaliciousInstruct yield an average attack success rate (ASR) of 94.01%, outperforming existing methods. To mitigate LFJ, we propose an adversarial training defense that fine-tunes models on interpolated examples, reducing ASR by over 80% without degrading performance on benign inputs. Ablation studies validate the importance of query pair selection, hidden state interpolation components, and optimization strategies in LFJ's effectiveness. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种语言任务中展示了 impressive 的能力，但容易受到规避其安全对齐的 jailbreak 攻击。本文介绍了基于表示的 jailbreak 攻击（Latent Fusion Jailbreak，LFJ），该攻击通过插值有害和 benign 查询对的隐藏状态来引发禁止的响应。LFJ 首先选择具有高主题和句法相似性的查询对，然后在具有影响力的层和 token 上进行梯度导向插值，最后进行优化以平衡攻击成功率、输出流畅性和计算效率。在 Vicuna 和 LLaMA-2 等模型上的 AdvBench 和 MaliciousInstruct 等基准测试中，LFJ 的平均攻击成功率 (ASR) 达到 94.01%，优于现有方法。为了缓解 LFJ，我们提出了一种对抗性训练防御方法，该方法通过对插值样例进行微调，ASR 减少超过 80%，且不影响 benign 输入的性能。消融研究验证了查询对选择、隐藏状态插值组件以及优化策略在 LFJ 效果中的重要性。 

---
# PREF: Reference-Free Evaluation of Personalised Text Generation in LLMs 

**Title (ZH)**: PREF：LLM中个性化文本生成的无参照评估 

**Authors**: Xiao Fu, Hossein A. Rahmani, Bin Wu, Jerome Ramos, Emine Yilmaz, Aldo Lipani  

**Link**: [PDF](https://arxiv.org/pdf/2508.10028)  

**Abstract**: Personalised text generation is essential for user-centric information systems, yet most evaluation methods overlook the individuality of users. We introduce \textbf{PREF}, a \textbf{P}ersonalised \textbf{R}eference-free \textbf{E}valuation \textbf{F}ramework that jointly measures general output quality and user-specific alignment without requiring gold personalised references. PREF operates in a three-step pipeline: (1) a coverage stage uses a large language model (LLM) to generate a comprehensive, query-specific guideline covering universal criteria such as factuality, coherence, and completeness; (2) a preference stage re-ranks and selectively augments these factors using the target user's profile, stated or inferred preferences, and context, producing a personalised evaluation rubric; and (3) a scoring stage applies an LLM judge to rate candidate answers against this rubric, ensuring baseline adequacy while capturing subjective priorities. This separation of coverage from preference improves robustness, transparency, and reusability, and allows smaller models to approximate the personalised quality of larger ones. Experiments on the PrefEval benchmark, including implicit preference-following tasks, show that PREF achieves higher accuracy, better calibration, and closer alignment with human judgments than strong baselines. By enabling scalable, interpretable, and user-aligned evaluation, PREF lays the groundwork for more reliable assessment and development of personalised language generation systems. 

**Abstract (ZH)**: 个性化文本生成是用户中心信息系统的关键，然而大多数评估方法忽视了用户的个性化需求。我们提出了一个名为PREF的个性化参考自由评估框架，该框架能够在不需要金标准个性化参考的情况下，联合衡量通用输出质量和用户特定对齐。PREF 通过三步流水线运行：（1）覆盖阶段使用大规模语言模型（LLM）生成全面的查询特定指南，涵盖事实性、连贯性和完整性等通用标准；（2）偏好阶段根据目标用户的人格特征、显性和隐性偏好以及上下文重新排序并有选择地增强这些因素，生成个性化评估标准；（3）评分阶段应用LLM评审员对候选答案进行评分，确保基本充足性同时捕捉主观优先级。通过将覆盖与偏好分离，PREF 提高了鲁棒性、透明度和再利用性，并允许较小的模型近似大型模型的个性化质量。在PrefEval基准上的实验，包括隐含偏好跟随任务，表明PREF在准确性、校准度和与人类判断的一致性方面优于强大基线。通过实现可扩展、可解释且用户对齐的评估，PREF 为个性化语言生成系统的更可靠评估和开发奠定了基础。 

---
# LLMCARE: Alzheimer's Detection via Transformer Models Enhanced by LLM-Generated Synthetic Data 

**Title (ZH)**: LLMCARE: 基于LLM生成合成数据增强的变压器模型在阿尔茨海默病检测中的应用 

**Authors**: Ali Zolnour, Hossein Azadmaleki, Yasaman Haghbin, Fatemeh Taherinezhad, Mohamad Javad Momeni Nezhad, Sina Rashidi, Masoud Khani, AmirSajjad Taleban, Samin Mahdizadeh Sani, Maryam Dadkhah, James M. Noble, Suzanne Bakken, Yadollah Yaghoobzadeh, Abdol-Hossein Vahabie, Masoud Rouhizadeh, Maryam Zolnoori  

**Link**: [PDF](https://arxiv.org/pdf/2508.10027)  

**Abstract**: Alzheimer's disease and related dementias (ADRD) affect approximately five million older adults in the U.S., yet over half remain undiagnosed. Speech-based natural language processing (NLP) offers a promising, scalable approach to detect early cognitive decline through linguistic markers.
To develop and evaluate a screening pipeline that (i) fuses transformer embeddings with handcrafted linguistic features, (ii) tests data augmentation using synthetic speech generated by large language models (LLMs), and (iii) benchmarks unimodal and multimodal LLM classifiers for ADRD detection.
Transcripts from the DementiaBank "cookie-theft" task (n = 237) were used. Ten transformer models were evaluated under three fine-tuning strategies. A fusion model combined embeddings from the top-performing transformer with 110 lexical-derived linguistic features. Five LLMs (LLaMA-8B/70B, MedAlpaca-7B, Ministral-8B, GPT-4o) were fine-tuned to generate label-conditioned synthetic speech, which was used to augment training data. Three multimodal models (GPT-4o, Qwen-Omni, Phi-4) were tested for speech-text classification in zero-shot and fine-tuned settings.
The fusion model achieved F1 = 83.3 (AUC = 89.5), outperforming linguistic or transformer-only baselines. Augmenting training data with 2x MedAlpaca-7B synthetic speech increased F1 to 85.7. Fine-tuning significantly improved unimodal LLM classifiers (e.g., MedAlpaca: F1 = 47.3 -> 78.5 F1). Current multimodal models demonstrated lower performance (GPT-4o = 70.2 F1; Qwen = 66.0). Performance gains aligned with the distributional similarity between synthetic and real speech.
Integrating transformer embeddings with linguistic features enhances ADRD detection from speech. Clinically tuned LLMs effectively support both classification and data augmentation, while further advancement is needed in multimodal modeling. 

**Abstract (ZH)**: 阿尔茨海默病及相关痴呆症（ADRD）影响美国约五百万名老年人，但其中超过一半未被诊断。基于语音的自然语言处理（NLP）提供了一种有潜力且可扩展的方法，通过语言标记检测早期认知衰退。 

---
# SABER: Switchable and Balanced Training for Efficient LLM Reasoning 

**Title (ZH)**: SABER: 可切换且均衡的训练以实现高效的LLM推理 

**Authors**: Kai Zhao, Yanjun Zhao, Jiaming Song, Shien He, Lusheng Zhang, Qiang Zhang, Tianjiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.10026)  

**Abstract**: Large language models (LLMs) empowered by chain-of-thought reasoning have achieved impressive accuracy on complex tasks but suffer from excessive inference costs and latency when applied uniformly to all problems. We propose SABER (Switchable and Balanced Training for Efficient LLM Reasoning), a reinforcement learning framework that endows LLMs with user-controllable, token-budgeted reasoning. SABER first profiles each training example's base-model thinking token usage and assigns it to one of the predefined budget tiers. During fine-tuning, the model is guided by system prompts and length-aware rewards to respect its assigned budget. In parallel, we incorporate no-think examples to ensure the model remains reliable even when explicit reasoning is turned off. SABER further supports four discrete inference modes - NoThink, FastThink, CoreThink, and DeepThink, enabling flexible trade-offs between latency and reasoning depth. Extensive evaluations on math reasoning (MATH, GSM8K), code generation (MBPP), and logical reasoning (LiveBench-Reasoning) demonstrate that SABER achieves high accuracy under tight budgets, graceful degradation, and effective cross-scale and cross-domain generalization. In particular, SABER-FastThink cuts reasoning length by 65.4% and yields a 3.6% accuracy gain compared with the base model on the MATH benchmark. 

**Abstract (ZH)**: 基于链式思维增强的大语言模型高效推理框架SABER 

---
# Detecting and explaining postpartum depression in real-time with generative artificial intelligence 

**Title (ZH)**: 使用生成式人工智能实时检测和解释产后抑郁 

**Authors**: Silvia García-Méndez, Francisco de Arriba-Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2508.10025)  

**Abstract**: Among the many challenges mothers undergo after childbirth, postpartum depression (PPD) is a severe condition that significantly impacts their mental and physical well-being. Consequently, the rapid detection of ppd and their associated risk factors is critical for in-time assessment and intervention through specialized prevention procedures. Accordingly, this work addresses the need to help practitioners make decisions with the latest technological advancements to enable real-time screening and treatment recommendations. Mainly, our work contributes to an intelligent PPD screening system that combines Natural Language Processing, Machine Learning (ML), and Large Language Models (LLMs) towards an affordable, real-time, and non-invasive free speech analysis. Moreover, it addresses the black box problem since the predictions are described to the end users thanks to the combination of LLMs with interpretable ml models (i.e., tree-based algorithms) using feature importance and natural language. The results obtained are 90 % on ppd detection for all evaluation metrics, outperforming the competing solutions in the literature. Ultimately, our solution contributes to the rapid detection of PPD and their associated risk factors, critical for in-time and proper assessment and intervention. 

**Abstract (ZH)**: 产后抑郁症的智能筛查系统：结合自然语言处理、机器学习和大型语言模型的实时、无侵入性免费言语分析及其应用 

---
# RTTC: Reward-Guided Collaborative Test-Time Compute 

**Title (ZH)**: RTTC: 奖励引导的合作测试时计算 

**Authors**: J. Pablo Muñoz, Jinjie Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2508.10024)  

**Abstract**: Test-Time Compute (TTC) has emerged as a powerful paradigm for enhancing the performance of Large Language Models (LLMs) at inference, leveraging strategies such as Test-Time Training (TTT) and Retrieval-Augmented Generation (RAG). However, the optimal adaptation strategy varies across queries, and indiscriminate application of TTC strategy incurs substantial computational overhead. In this work, we introduce Reward-Guided Test-Time Compute (RTTC), a novel framework that adaptively selects the most effective TTC strategy for each query via a pretrained reward model, maximizing downstream accuracy across diverse domains and tasks. RTTC operates in a distributed server-client architecture, retrieving relevant samples from a remote knowledge base and applying RAG or lightweight fine-tuning on client devices only when necessary. To further mitigate redundant computation, we propose Query-State Caching, which enables the efficient reuse of historical query states at both retrieval and adaptation levels. Extensive experiments across multiple LLMs and benchmarks demonstrate that RTTC consistently achieves superior accuracy compared to vanilla RAG or TTT, validating the necessity of adaptive, reward-guided TTC selection and the potential of RTTC for scalable, high-performance language model adaptation. 

**Abstract (ZH)**: 基于奖励的测试时计算（RTTC）：一种自适应选择最优测试时计算策略的新型框架 

---
# Conformal P-Value in Multiple-Choice Question Answering Tasks with Provable Risk Control 

**Title (ZH)**: 多重选择题回答任务中的可验证风险控制齐性P值 

**Authors**: Yuanchang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2508.10022)  

**Abstract**: This study introduces a significance testing-enhanced conformal prediction (CP) framework to improve trustworthiness of large language models (LLMs) in multiple-choice question answering (MCQA). While LLMs have been increasingly deployed in disciplinary QA scenarios, hallucination and nonfactual generation substantially compromise response reliability. Although CP provides statistically rigorous marginal coverage guarantees for prediction sets, and significance testing offers established statistical rigor, their synergistic integration remains unexplored. To mitigate hallucination and factual inaccuracies, our framework integrates $p$-value computation with conformity scoring through self-consistency resampling of MCQA responses. This approach calculates option frequencies to address LLMs' black-box nature, subsequently constructing prediction sets via null hypothesis testing ($\mathcal{H}_0$) with empirically derived $p$-values. Evaluations on MMLU and MMLU-Pro benchmarks using off-the-shelf LLMs demonstrate: (1) The enhanced CP achieves user-specified empirical miscoverage rates; (2) Test-set average prediction set size (APSS) decreases monotonically with increasing risk levels ($\alpha$), validating APSS as an effective uncertainty metric. This work establishes a principled statistical framework for trustworthy LLM deployment in high-stakes QA applications. 

**Abstract (ZH)**: 增强显著性检验的協同預測框架以提升大型语言模型在多项选择题回答中的可信度 

---
# LATTE: Learning Aligned Transactions and Textual Embeddings for Bank Clients 

**Title (ZH)**: LATTE: 学习对齐的交易和文本嵌入方法以用于银行客户 

**Authors**: Egor Fadeev, Dzhambulat Mollaev, Aleksei Shestov, Dima Korolev, Omar Zoloev, Ivan Kireev, Andrey Savchenko, Maksim Makarenko  

**Link**: [PDF](https://arxiv.org/pdf/2508.10021)  

**Abstract**: Learning clients embeddings from sequences of their historic communications is central to financial applications. While large language models (LLMs) offer general world knowledge, their direct use on long event sequences is computationally expensive and impractical in real-world pipelines. In this paper, we propose LATTE, a contrastive learning framework that aligns raw event embeddings with semantic embeddings from frozen LLMs. Behavioral features are summarized into short prompts, embedded by the LLM, and used as supervision via contrastive loss. The proposed approach significantly reduces inference cost and input size compared to conventional processing of complete sequence by LLM. We experimentally show that our method outperforms state-of-the-art techniques for learning event sequence representations on real-world financial datasets while remaining deployable in latency-sensitive environments. 

**Abstract (ZH)**: 从客户历史通信序列中学习客户嵌入对于金融应用至关重要。尽管大型语言模型（LLMs）提供了广泛的世界知识，但在实际工作流中直接使用LLM处理长事件序列在计算成本上是昂贵且不切实际的。本文提出了一种对比学习框架LATTE，该框架将原始事件嵌入与冻结LLM的语义嵌入对齐。行为特征被总结为简短的提示，通过LLM进行嵌入，并作为对比损失的监督使用。所提出的方法与传统通过LLM处理完整序列的方法相比，显著减少了推理成本和输入大小。实验结果表明，我们的方法在现实生活中的金融数据集上优于最先进的事件序列表示学习技术，同时仍然可以在低延迟环境中部署。 

---
# FedCoT: Communication-Efficient Federated Reasoning Enhancement for Large Language Models 

**Title (ZH)**: FedCoT：面向大型语言模型的通信高效联邦推理增强 

**Authors**: Chuan Li, Qianyi Zhao, Fengran Mo, Cen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10020)  

**Abstract**: Efficiently enhancing the reasoning capabilities of large language models (LLMs) in federated learning environments remains challenging, particularly when balancing performance gains with strict computational, communication, and privacy constraints. This challenge is especially acute in healthcare, where decisions-spanning clinical, operational, and patient-facing contexts-demand not only accurate outputs but also interpretable, traceable rationales to ensure safety, accountability, and regulatory compliance. Conventional federated tuning approaches on LLM fail to address this need: they optimize primarily for answer correctness while neglecting rationale quality, leaving CoT capabilities dependent on models' innate pre-training abilities. Moreover, existing methods for improving rationales typically rely on privacy-violating knowledge distillation from centralized models. Additionally, the communication overhead in traditional federated fine-tuning on LLMs remains substantial. We addresses this gap by proposing FedCoT, a novel framework specifically designed to enhance reasoning in federated settings. FedCoT leverages a lightweight chain-of-thought enhancement mechanism: local models generate multiple reasoning paths, and a compact discriminator dynamically selects the most promising one. This approach improves reasoning accuracy and robustness while providing valuable interpretability, which is particularly critical for medical applications. To manage client heterogeneity efficiently, we adopt an improved aggregation approach building upon advanced LoRA module stacking, incorporating client classifier-awareness to achieve noise-free aggregation across diverse clients. Comprehensive experiments on medical reasoning tasks demonstrate that FedCoT significantly boosts client-side reasoning performance under stringent resource budgets while fully preserving data privacy. 

**Abstract (ZH)**: 有效地增强联邦学习环境中大型语言模型的推理能力仍具有挑战性，特别是在平衡性能提升与严格的计算、通信和隐私约束之间的关系时。这一挑战在医疗保健领域尤为严峻，因为在涉及临床、运营和患者面向等多种情境下的决策不仅需要准确的输出，还需要可解释、可追踪的推理过程以确保安全性、问责制和合规性。传统的联邦调优方法未能满足这一需求：它们主要优化答案的正确性，而忽视了推理质量，导致解释链（CoT）能力依赖于模型的先天预训练能力。此外，现有的提高推理质量的方法通常依赖于从集中式模型获取隐私侵犯的知识蒸馏。同时，传统的大型语言模型联邦微调过程中的通信开销依然较大。我们通过提出FedCoT这一新颖框架来填补这一空白，该框架专门设计用于增强联邦环境中的推理能力。FedCoT利用轻量级的推理增强机制：本地模型生成多种推理路径，紧凑的辨别器动态选择最有希望的一条。该方法在提高推理准确性和稳健性的同时提供了有价值的可解释性，这对于医疗应用尤为重要。为了高效管理客户端异质性，我们采用改进的聚合方法，基于先进的LoRA模块堆叠，并结合客户端分类器的意识，实现多样客户端的无噪声聚合。全面的医疗推理任务实验表明，FedCoT能够在严格的资源预算下显著提升客户端的推理性能，同时完全保持数据隐私。 

---
# Decoupling Understanding from Reasoning via Problem Space Mapping for Small-scale Model Reasoning 

**Title (ZH)**: 通过问题空间映射解耦理解与推理的小规模模型推理 

**Authors**: Li Wang, Changhao Zhang, Zengqi Xiu, Kai Lu, Xin Yu, Kui Zhang, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10019)  

**Abstract**: Despite recent advances in the reasoning capabilities of Large Language Models (LLMs), improving the reasoning ability of Small Language Models (SLMs, e.g., $\leq$ 1.5B) remains challenging. A key obstacle lies in the complexity and variability of natural language: essentially equivalent problems often appear in diverse surface forms, often obscured by redundant or distracting details. This imposes a dual burden on SLMs: they must first extract the core problem from complex linguistic input, and then perform reasoning based on that understanding. The resulting vast and noisy problem space hinders optimization, particularly for models with limited capacity. To address this, we propose a new framework that decouples understanding from reasoning by mapping natural language problems into a canonical problem space-a semantically simplified yet expressive domain. This enables SLMs to focus on reasoning over standardized inputs, free from linguistic variability. Within this framework, we introduce DURIT (Decoupled Understanding from Reasoning via Iterative Training), a three-step algorithm that iteratively: (1) mapping natural language problems via reinforcement learning, (2) aligns reasoning trajectories through self-distillation, and (3) trains reasoning policies in the problem space. The mapper and reasoner are co-trained in an alternating loop throughout this process. Experiments show that DURIT substantially improves SLMs' performance on both in-domain and out-of-domain mathematical and logical reasoning tasks. Beyond improving reasoning capabilities, DURIT also improves the robustness of reasoning, validating decoupling understanding from reasoning as an effective strategy for strengthening SLMs. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）的推理能力已有显著进步，改善小型语言模型（SLMs，例如≤1.5B）的推理能力仍然具有挑战性。这一挑战的关键障碍在于自然语言的复杂性和变异性：本质上等价的问题常常以多种多样的表面形式出现，往往被冗余或分散注意力的细节所掩盖。这给SLMs带来了双重负担：它们必须首先从复杂的语言输入中提取核心问题，然后基于这种理解进行推理。由此产生的庞大且混乱的问题空间妨碍了优化，尤其是在容量有限的模型中。为了解决这一问题，我们提出了一种新的框架，通过将自然语言问题映射到一个规范的问题空间——一个语义简化但表达力强的领域来解耦理解与推理。这使得SLMs能够专注于标准化输入上的推理，而不会受到语言变异的影响。在这个框架内，我们引入了DURIT（通过迭代训练解耦推理与理解），这是一个三步算法，通过迭代地（1）使用强化学习映射自然语言问题，（2）通过自蒸馏对齐推理轨迹，（3）在问题空间中训练推理策略。在整个过程中，映射器和推理器在交替循环中共同训练。实验表明，DURIT显著提高了SLMs在领域内和领域外的数学和逻辑推理任务中的性能。除了提高推理能力外，DURIT还提高了推理的稳健性，验证了解耦理解与推理作为增强SLMs的有效策略的有效性。 

---
# A Rose by Any Other Name Would Smell as Sweet: Categorical Homotopy Theory for Large Language Models 

**Title (ZH)**: 番茄用任何其他名字叫它都会一样芳香：大型语言模型的分类同伦理论 

**Authors**: Sridhar Mahadevan  

**Link**: [PDF](https://arxiv.org/pdf/2508.10018)  

**Abstract**: Natural language is replete with superficially different statements, such as ``Charles Darwin wrote" and ``Charles Darwin is the author of", which carry the same meaning. Large language models (LLMs) should generate the same next-token probabilities in such cases, but usually do not. Empirical workarounds have been explored, such as using k-NN estimates of sentence similarity to produce smoothed estimates. In this paper, we tackle this problem more abstractly, introducing a categorical homotopy framework for LLMs. We introduce an LLM Markov category to represent probability distributions in language generated by an LLM, where the probability of a sentence, such as ``Charles Darwin wrote" is defined by an arrow in a Markov category. However, this approach runs into difficulties as language is full of equivalent rephrases, and each generates a non-isomorphic arrow in the LLM Markov category. To address this fundamental problem, we use categorical homotopy techniques to capture ``weak equivalences" in an LLM Markov category. We present a detailed overview of application of categorical homotopy to LLMs, from higher algebraic K-theory to model categories, building on powerful theoretical results developed over the past half a century. 

**Abstract (ZH)**: 自然语言充满了表面上不同的但意义相同的陈述，例如“Charles Darwin wrote”和“Charles Darwin is the author of”。大型语言模型（LLMs）在这种情况下应当生成相同的目标词概率，但实际上通常并不如此。现有的经验性解决办法包括使用k-NN估计的句子相似度来生成平滑估计。在本文中，我们从更抽象的角度解决这个问题，引入了LLM范畴同伦框架。我们提出了一种LLM马尔可夫范畴来表示由LLM生成的语言的概率分布，其中句子“Charles Darwin wrote”的概率通过马尔可夫范畴中的一个箭头来定义。然而，由于语言中充斥着等价的不同表达方式，每种表达方式在LLM马尔可夫范畴中生成的是非同构的箭头，因此这种方法遇到了困难。为了解决这一基础问题，我们利用范畴同伦技术来捕捉LLM马尔可夫范畴中的“弱同伦等价”。我们详细介绍了范畴同伦在LLM中的应用，从高阶代数K理论到模型范畴，基于过去半个世纪中发展起来的强有力的理论成果。 

---
# From Answers to Questions: EQGBench for Evaluating LLMs' Educational Question Generation 

**Title (ZH)**: 从答案到问题：EQGBench 用于评估大型语言模型的教育性问题生成能力 

**Authors**: Chengliang Zhou, Mei Wang, Ting Zhang, Qiannan Zhu, Jian Li, Hua Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10005)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in mathematical problem-solving. However, the transition from providing answers to generating high-quality educational questions presents significant challenges that remain underexplored. To advance Educational Question Generation (EQG) and facilitate LLMs in generating pedagogically valuable and educationally effective questions, we introduce EQGBench, a comprehensive benchmark specifically designed for evaluating LLMs' performance in Chinese EQG. EQGBench establishes a five-dimensional evaluation framework supported by a dataset of 900 evaluation samples spanning three fundamental middle school disciplines: mathematics, physics, and chemistry. The dataset incorporates user queries with varying knowledge points, difficulty gradients, and question type specifications to simulate realistic educational scenarios. Through systematic evaluation of 46 mainstream large models, we reveal significant room for development in generating questions that reflect educational value and foster students' comprehensive abilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）在数学问题解决方面展现了出色的能力。然而，从提供答案到生成高质量的教育性问题这一转变面临着尚未充分探索的重大挑战。为促进教育性问题生成（EQG）并帮助LLMs生成教育价值高且有效的教学问题，我们介绍了EQGBench，一个专门用于评估LLMs在中文EQG方面性能的综合性基准。EQGBench建立了由900个评价样本组成的数据集，覆盖三个基本的中学学科：数学、物理和化学，并建立了五维评估框架。该数据集包含了不同知识点、难度等级和问题类型要求的用户查询，以模拟现实的教育场景。通过对46个主流大模型的系统评估，我们揭示了在生成反映教育价值问题方面仍有显著的发展空间，有助于培养学生的综合能力。 

---
# Semantic Structure in Large Language Model Embeddings 

**Title (ZH)**: 大型语言模型嵌入中的语义结构 

**Authors**: Austin C. Kozlowski, Callin Dai, Andrei Boutyline  

**Link**: [PDF](https://arxiv.org/pdf/2508.10003)  

**Abstract**: Psychological research consistently finds that human ratings of words across diverse semantic scales can be reduced to a low-dimensional form with relatively little information loss. We find that the semantic associations encoded in the embedding matrices of large language models (LLMs) exhibit a similar structure. We show that the projections of words on semantic directions defined by antonym pairs (e.g. kind - cruel) correlate highly with human ratings, and further find that these projections effectively reduce to a 3-dimensional subspace within LLM embeddings, closely resembling the patterns derived from human survey responses. Moreover, we find that shifting tokens along one semantic direction causes off-target effects on geometrically aligned features proportional to their cosine similarity. These findings suggest that semantic features are entangled within LLMs similarly to how they are interconnected in human language, and a great deal of semantic information, despite its apparent complexity, is surprisingly low-dimensional. Furthermore, accounting for this semantic structure may prove essential for avoiding unintended consequences when steering features. 

**Abstract (ZH)**: 心理学研究一致发现，人类对不同语义尺度的单词评级可以被简化为一种低维形式，信息损失相对较小。我们发现，大型语言模型（LLMs）的嵌入矩阵中编码的语义关联表现出类似的结构。我们证明，单词在由反义词对（如kind - cruel）定义的语义方向上的投影与人类评级高度相关，并进一步发现这些投影在LLM嵌入中有效减少到三维子空间，与从人类调查响应中得出的模式相似。此外，我们发现沿一个语义方向移动令牌会导致与其余几何对齐特征成正比的意外影响。这些发现表明，语义特征在LLMs中纠缠在一起，类似于人类语言中它们的相互连接方式，尽管语义信息显得复杂，但实际上却高度低维。此外，考虑这种语义结构在引导特征时可能是避免无意后果的关键。 

---
