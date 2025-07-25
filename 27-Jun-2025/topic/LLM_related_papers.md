# PsyLite Technical Report 

**Title (ZH)**: PsyLite 技术报告 

**Authors**: Fangjun Ding, Renyu Zhang, Xinyu Feng, Chengye Xie, Zheng Zhang, Yanting Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21536)  

**Abstract**: With the rapid development of digital technology, AI-driven psychological counseling has gradually become an important research direction in the field of mental health. However, existing models still have deficiencies in dialogue safety, detailed scenario handling, and lightweight deployment. To address these issues, this study proposes PsyLite, a lightweight psychological counseling large language model agent developed based on the base model InternLM2.5-7B-chat. Through a two-stage training strategy (hybrid distillation data fine-tuning and ORPO preference optimization), PsyLite enhances the model's deep-reasoning ability, psychological counseling ability, and safe dialogue ability. After deployment using Ollama and Open WebUI, a custom workflow is created with Pipelines. An innovative conditional RAG is designed to introduce crosstalk humor elements at appropriate times during psychological counseling to enhance user experience and decline dangerous requests to strengthen dialogue safety. Evaluations show that PsyLite outperforms the baseline models in the Chinese general evaluation (CEval), psychological counseling professional evaluation (CPsyCounE), and dialogue safety evaluation (SafeDialBench), particularly in psychological counseling professionalism (CPsyCounE score improvement of 47.6\%) and dialogue safety (\safe{} score improvement of 2.4\%). Additionally, the model uses quantization technology (GGUF q4\_k\_m) to achieve low hardware deployment (5GB memory is sufficient for operation), providing a feasible solution for psychological counseling applications in resource-constrained environments. 

**Abstract (ZH)**: 基于InternLM2.5-7B-chat的轻量级AI心理辅导模型PsyLite的研究 

---
# Unveiling Causal Reasoning in Large Language Models: Reality or Mirage? 

**Title (ZH)**: 探索大型语言模型中的因果推理：现实还是幻象？ 

**Authors**: Haoang Chi, He Li, Wenjing Yang, Feng Liu, Long Lan, Xiaoguang Ren, Tongliang Liu, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.21215)  

**Abstract**: Causal reasoning capability is critical in advancing large language models (LLMs) toward strong artificial intelligence. While versatile LLMs appear to have demonstrated capabilities in understanding contextual causality and providing responses that obey the laws of causality, it remains unclear whether they perform genuine causal reasoning akin to humans. However, current evidence indicates the contrary. Specifically, LLMs are only capable of performing shallow (level-1) causal reasoning, primarily attributed to the causal knowledge embedded in their parameters, but they lack the capacity for genuine human-like (level-2) causal reasoning. To support this hypothesis, methodologically, we delve into the autoregression mechanism of transformer-based LLMs, revealing that it is not inherently causal. Empirically, we introduce a new causal Q&A benchmark called CausalProbe-2024, whose corpora are fresh and nearly unseen for the studied LLMs. The LLMs exhibit a significant performance drop on CausalProbe-2024 compared to earlier benchmarks, indicating the fact that they primarily engage in level-1 causal reasoning. To bridge the gap towards level-2 causal reasoning, we draw inspiration from the fact that human reasoning is usually facilitated by general knowledge and intended goals. We propose G^2-Reasoner, a method that incorporates general knowledge and goal-oriented prompts into LLMs' causal reasoning processes. Experiments demonstrate that G^2-Reasoner significantly enhances LLMs' causal reasoning capability, particularly in fresh and counterfactual contexts. This work sheds light on a new path for LLMs to advance towards genuine causal reasoning, going beyond level-1 and making strides towards level-2. 

**Abstract (ZH)**: 因果推理能力是推动大型语言模型（LLMs）向强人工智能发展的关键。虽然多功能LLMs似乎展示了理解上下文因果关系并提供遵循因果法则的回应的能力，但尚不清楚它们是否进行了与人类类似的真正因果推理。然而，当前的证据表明并非如此。具体来说，LLMs仅具备浅层（第一层次）因果推理能力，主要归因于其参数中嵌入的因果知识，但它们缺乏与人类类似的深层（第二层次）因果推理能力。为了支持这一假设，从方法论上，我们深入探讨基于Transformer的LLMs的自回归机制，揭示出它并不是本原性的因果机制。从经验上，我们引入了一个新的因果问答基准——CausalProbe-2024，其语料库对研究中的LLMs来说几乎是前所未见的。LLMs在CausalProbe-2024上的表现大幅下降，表明它们主要进行的是浅层因果推理。为了弥补向深层因果推理的差距，我们从人类推理通常由一般知识和目标驱动的事实中得到启发。我们提出了G^2-Reasoner方法，该方法将一般知识和目标导向的提示纳入到LLMs的因果推理过程中。实验结果表明，G^2-Reasoner显著增强了LLMs的因果推理能力，特别是在新的和反事实的情景下。这项工作揭示了LLMs向真正因果推理前进的新路径，超越了浅层，并朝着深层迈进。 

---
# Beyond Reactive Safety: Risk-Aware LLM Alignment via Long-Horizon Simulation 

**Title (ZH)**: 超越反应性安全：基于长期模拟的风险感知大语言模型对齐 

**Authors**: Chenkai Sun, Denghui Zhang, ChengXiang Zhai, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2506.20949)  

**Abstract**: Given the growing influence of language model-based agents on high-stakes societal decisions, from public policy to healthcare, ensuring their beneficial impact requires understanding the far-reaching implications of their suggestions. We propose a proof-of-concept framework that projects how model-generated advice could propagate through societal systems on a macroscopic scale over time, enabling more robust alignment. To assess the long-term safety awareness of language models, we also introduce a dataset of 100 indirect harm scenarios, testing models' ability to foresee adverse, non-obvious outcomes from seemingly harmless user prompts. Our approach achieves not only over 20% improvement on the new dataset but also an average win rate exceeding 70% against strong baselines on existing safety benchmarks (AdvBench, SafeRLHF, WildGuardMix), suggesting a promising direction for safer agents. 

**Abstract (ZH)**: 基于语言模型的代理在高风险社会决策中的影响日益增大：一种评估长期安全意识的框架 

---
# Dynamic Context-Aware Prompt Recommendation for Domain-Specific AI Applications 

**Title (ZH)**: 领域特定AI应用中的动态上下文感知提示推荐 

**Authors**: Xinye Tang, Haijun Zhai, Chaitanya Belwal, Vineeth Thayanithi, Philip Baumann, Yogesh K Roy  

**Link**: [PDF](https://arxiv.org/pdf/2506.20815)  

**Abstract**: LLM-powered applications are highly susceptible to the quality of user prompts, and crafting high-quality prompts can often be challenging especially for domain-specific applications. This paper presents a novel dynamic context-aware prompt recommendation system for domain-specific AI applications. Our solution combines contextual query analysis, retrieval-augmented knowledge grounding, hierarchical skill organization, and adaptive skill ranking to generate relevant and actionable prompt suggestions.
The system leverages behavioral telemetry and a two-stage hierarchical reasoning process to dynamically select and rank relevant skills, and synthesizes prompts using both predefined and adaptive templates enhanced with few-shot learning. Experiments on real-world datasets demonstrate that our approach achieves high usefulness and relevance, as validated by both automated and expert evaluations. 

**Abstract (ZH)**: 基于LLM的应用高度依赖用户的提示质量，而设计高质量的提示尤其是在特定领域应用中常常具有挑战性。本文提出了一种新颖的动态上下文感知提示推荐系统，适用于特定领域的AI应用。该解决方案结合了上下文查询分析、检索增强的知识 grounding、层级技能组织和自适应技能排名，以生成相关且可操作的提示建议。该系统利用行为遥测和两阶段层级推理过程动态选择和排序相关技能，并使用预定义和自适应模板结合少样本学习综合生成提示。实验结果在真实世界数据集上验证了该方法的高度实用性和相关性，得到了自动和专家评估的认可。 

---
# MAGPIE: A dataset for Multi-AGent contextual PrIvacy Evaluation 

**Title (ZH)**: MAGPIE：一个多代理情境隐私评估数据集 

**Authors**: Gurusha Juneja, Alon Albalak, Wenyue Hua, William Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20737)  

**Abstract**: The proliferation of LLM-based agents has led to increasing deployment of inter-agent collaboration for tasks like scheduling, negotiation, resource allocation etc. In such systems, privacy is critical, as agents often access proprietary tools and domain-specific databases requiring strict confidentiality. This paper examines whether LLM-based agents demonstrate an understanding of contextual privacy. And, if instructed, do these systems preserve inference time user privacy in non-adversarial multi-turn conversation. Existing benchmarks to evaluate contextual privacy in LLM-agents primarily assess single-turn, low-complexity tasks where private information can be easily excluded. We first present a benchmark - MAGPIE comprising 158 real-life high-stakes scenarios across 15 domains. These scenarios are designed such that complete exclusion of private data impedes task completion yet unrestricted information sharing could lead to substantial losses. We then evaluate the current state-of-the-art LLMs on (a) their understanding of contextually private data and (b) their ability to collaborate without violating user privacy. Empirical experiments demonstrate that current models, including GPT-4o and Claude-2.7-Sonnet, lack robust understanding of contextual privacy, misclassifying private data as shareable 25.2\% and 43.6\% of the time. In multi-turn conversations, these models disclose private information in 59.9\% and 50.5\% of cases even under explicit privacy instructions. Furthermore, multi-agent systems fail to complete tasks in 71\% of scenarios. These results underscore that current models are not aligned towards both contextual privacy preservation and collaborative task-solving. 

**Abstract (ZH)**: 基于LLM的代理扩展导致了越来越多的代理间协作部署，用于如调度、谈判、资源分配等任务。在这种系统中，隐私至关重要，因为代理通常会访问专有工具和领域特定数据库，要求严格保密。本文考察了基于LLM的代理是否理解上下文隐私，并在非对抗性的多轮对话中，如果指示，是否能保护推理时间的用户隐私。现有针对上下文隐私评估的基准主要集中在单一回合、低复杂度的任务上，这些任务中私有信息容易被排除。我们首先提出了一个基准——MAGPIE，包含15个领域中的158个实际高风险场景。这些场景设计旨在完全排除私有数据会妨碍任务完成，而无限制的信息共享可能导致重大损失。然后，我们评估当前最先进的LLM的（a）对上下文隐私数据的理解以及（b）在不违反用户隐私的情况下协作的能力。实验证明，当前模型，包括GPT-4o和Claude-2.7-Sonnet，在理解上下文隐私方面缺乏稳健性，错误地将25.2%和43.6%的私有数据分类为可共享数据。在多轮对话中，即便在明确的隐私指令下，这些模型在59.9%和50.5%的情况下披露了私有信息。此外，在71%的场景中，多代理系统无法完成任务。这些结果强调当前模型未能同时实现上下文隐私保护和协作任务解决的目标。 

---
# "What's Up, Doc?": Analyzing How Users Seek Health Information in Large-Scale Conversational AI Datasets 

**Title (ZH)**: “医生，有什么事吗？”：分析用户在大规模对话AI数据集中寻求健康信息的方式 

**Authors**: Akshay Paruchuri, Maryam Aziz, Rohit Vartak, Ayman Ali, Best Uchehara, Xin Liu, Ishan Chatterjee, Monica Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2506.21532)  

**Abstract**: People are increasingly seeking healthcare information from large language models (LLMs) via interactive chatbots, yet the nature and inherent risks of these conversations remain largely unexplored. In this paper, we filter large-scale conversational AI datasets to achieve HealthChat-11K, a curated dataset of 11K real-world conversations composed of 25K user messages. We use HealthChat-11K and a clinician-driven taxonomy for how users interact with LLMs when seeking healthcare information in order to systematically study user interactions across 21 distinct health specialties. Our analysis reveals insights into the nature of how and why users seek health information, such as common interactions, instances of incomplete context, affective behaviors, and interactions (e.g., leading questions) that can induce sycophancy, underscoring the need for improvements in the healthcare support capabilities of LLMs deployed as conversational AI. Code and artifacts to retrieve our analyses and combine them into a curated dataset can be found here: this https URL 

**Abstract (ZH)**: 人们越来越通过交互聊天机器人从大型语言模型获取医疗健康信息，但这些对话的本质及其固有风险尚待深入探索。本文通过过滤大规模对话AI数据集，构建了包含11,000个真实世界对话的HealthChat-11K数据集，这些对话由25,000条用户消息组成。我们使用HealthChat-11K和基于临床专家定义的分类体系，系统研究用户在寻求医疗健康信息时与大型语言模型的互动，涵盖21个不同的医学专科。我们的分析揭示了用户寻求健康信息的方式和原因，包括常见的交互模式、不完整背景信息的实例、情感行为，以及可能引导奉承的交互（如引导性问题），强调了在作为对话AI部署的大型语言模型中改进医疗健康支持能力的必要性。获取我们分析的代码和构建精编数据集的相关文件，请访问：this https URL 

---
# Potemkin Understanding in Large Language Models 

**Title (ZH)**: Potemkin理解在大规模语言模型中 

**Authors**: Marina Mancoridis, Bec Weeks, Keyon Vafa, Sendhil Mullainathan  

**Link**: [PDF](https://arxiv.org/pdf/2506.21521)  

**Abstract**: Large language models (LLMs) are regularly evaluated using benchmark datasets. But what justifies making inferences about an LLM's capabilities based on its answers to a curated set of questions? This paper first introduces a formal framework to address this question. The key is to note that the benchmarks used to test LLMs -- such as AP exams -- are also those used to test people. However, this raises an implication: these benchmarks are only valid tests if LLMs misunderstand concepts in ways that mirror human misunderstandings. Otherwise, success on benchmarks only demonstrates potemkin understanding: the illusion of understanding driven by answers irreconcilable with how any human would interpret a concept. We present two procedures for quantifying the existence of potemkins: one using a specially designed benchmark in three domains, the other using a general procedure that provides a lower-bound on their prevalence. We find that potemkins are ubiquitous across models, tasks, and domains. We also find that these failures reflect not just incorrect understanding, but deeper internal incoherence in concept representations. 

**Abstract (ZH)**: 大型语言模型（LLMs）通常通过基准数据集进行评估。但根据LLM对特定问题集的回答来推断其能力的合理性是什么？本文首先引入了一个正式框架来解答这一问题。关键在于，用于测试LLM的基准测试（如AP考试）同样也用于测试人类。然而，这引发了这样一个推论：只有当LLM对概念的理解方式类似于人类误解的方式时，这些基准测试才是有效的。否则，基准测试的成功仅展示了“纸牌屋理解”：一种由不可调和的答案所驱动的表象理解。我们提出了两种量化“纸牌屋理解”的方法：一种是使用专门设计的基准在三个领域中进行，另一种是使用通用程序，提供其普遍性的下限。我们发现“纸牌屋理解”在各种模型、任务和领域中普遍存在。我们还发现，这些失败不仅仅是理解错误，还反映了概念表示中的深层次内在不一致性。 

---
# Domain Knowledge-Enhanced LLMs for Fraud and Concept Drift Detection 

**Title (ZH)**: 增强领域知识的LLM在欺诈和概念漂移检测中的应用 

**Authors**: Ali Şenol, Garima Agrawal, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21443)  

**Abstract**: Detecting deceptive conversations on dynamic platforms is increasingly difficult due to evolving language patterns and Concept Drift (CD)\-i.e., semantic or topical shifts that alter the context or intent of interactions over time. These shifts can obscure malicious intent or mimic normal dialogue, making accurate classification challenging. While Large Language Models (LLMs) show strong performance in natural language tasks, they often struggle with contextual ambiguity and hallucinations in risk\-sensitive scenarios. To address these challenges, we present a Domain Knowledge (DK)\-Enhanced LLM framework that integrates pretrained LLMs with structured, task\-specific insights to perform fraud and concept drift detection. The proposed architecture consists of three main components: (1) a DK\-LLM module to detect fake or deceptive conversations; (2) a drift detection unit (OCDD) to determine whether a semantic shift has occurred; and (3) a second DK\-LLM module to classify the drift as either benign or fraudulent. We first validate the value of domain knowledge using a fake review dataset and then apply our full framework to SEConvo, a multiturn dialogue dataset that includes various types of fraud and spam attacks. Results show that our system detects fake conversations with high accuracy and effectively classifies the nature of drift. Guided by structured prompts, the LLaMA\-based implementation achieves 98\% classification accuracy. Comparative studies against zero\-shot baselines demonstrate that incorporating domain knowledge and drift awareness significantly improves performance, interpretability, and robustness in high\-stakes NLP applications. 

**Abstract (ZH)**: 动态平台上检测欺骗性对话日趋困难，原因在于语言模式不断演变和概念漂移（CD），即随时间改变互动语境或意图的语义或主题转移。这些转移可能会掩盖恶意意图或模仿正常对话，使得准确分类变得极具挑战性。尽管大型语言模型（LLMs）在自然语言任务中表现出色，但在风险管理敏感场景下往往难以应对上下文模糊和幻觉问题。为应对这些挑战，我们提出了一种领域知识（DK）增强的LLM框架，该框架结合了预训练的LLMs与结构化的任务特定洞察，以实现欺诈和概念漂移检测。所提出的架构包括三个主要组件：（1）一个DK-LLM模块，用于检测虚假或欺骗性对话；（2）一个漂移检测单元（OCDD），用于确定是否发生了语义转移；（3）一个第二个DK-LLM模块，用于将漂移分类为良性或欺诈。我们首先使用虚假评论数据集验证了领域知识的价值，然后将完整框架应用于包含各种欺诈和垃圾信息攻击的SEConvo多轮对话数据集。结果显示，我们的系统能够以高精度检测虚假对话，并有效分类漂移的性质。基于结构化提示的LLaMA实现达到了98%的分类精度。与零样本基线的对比研究表明，结合领域知识和漂移意识显著提升了高风险自然语言处理应用场景下的性能、可解释性和鲁棒性。 

---
# Scalable Bayesian Low-Rank Adaptation of Large Language Models via Stochastic Variational Subspace Inference 

**Title (ZH)**: 大规模语言模型通过随机变分子空间推断的可扩展贝叶斯低秩适应 

**Authors**: Colin Samplawski, Adam D. Cobb, Manoj Acharya, Ramneet Kaur, Susmit Jha  

**Link**: [PDF](https://arxiv.org/pdf/2506.21408)  

**Abstract**: Despite their widespread use, large language models (LLMs) are known to hallucinate incorrect information and be poorly calibrated. This makes the uncertainty quantification of these models of critical importance, especially in high-stakes domains, such as autonomy and healthcare. Prior work has made Bayesian deep learning-based approaches to this problem more tractable by performing inference over the low-rank adaptation (LoRA) parameters of a fine-tuned model. While effective, these approaches struggle to scale to larger LLMs due to requiring further additional parameters compared to LoRA. In this work we present $\textbf{Scala}$ble $\textbf{B}$ayesian $\textbf{L}$ow-Rank Adaptation via Stochastic Variational Subspace Inference (ScalaBL). We perform Bayesian inference in an $r$-dimensional subspace, for LoRA rank $r$. By repurposing the LoRA parameters as projection matrices, we are able to map samples from this subspace into the full weight space of the LLM. This allows us to learn all the parameters of our approach using stochastic variational inference. Despite the low dimensionality of our subspace, we are able to achieve competitive performance with state-of-the-art approaches while only requiring ${\sim}1000$ additional parameters. Furthermore, it allows us to scale up to the largest Bayesian LLM to date, with four times as a many base parameters as prior work. 

**Abstract (ZH)**: 可扩展的基于随机变分子空间推断的低秩贝叶斯适应（ScalaBL） 

---
# Leveraging LLM-Assisted Query Understanding for Live Retrieval-Augmented Generation 

**Title (ZH)**: 利用大语言模型辅助的查询理解实现实时检索增强生成 

**Authors**: Guanting Dong, Xiaoxi Li, Yuyao Zhang, Mengjie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2506.21384)  

**Abstract**: Real-world live retrieval-augmented generation (RAG) systems face significant challenges when processing user queries that are often noisy, ambiguous, and contain multiple intents. While RAG enhances large language models (LLMs) with external knowledge, current systems typically struggle with such complex inputs, as they are often trained or evaluated on cleaner data. This paper introduces Omni-RAG, a novel framework designed to improve the robustness and effectiveness of RAG systems in live, open-domain settings. Omni-RAG employs LLM-assisted query understanding to preprocess user inputs through three key modules: (1) Deep Query Understanding and Decomposition, which utilizes LLMs with tailored prompts to denoise queries (e.g., correcting spelling errors) and decompose multi-intent queries into structured sub-queries; (2) Intent-Aware Knowledge Retrieval, which performs retrieval for each sub-query from a corpus (i.e., FineWeb using OpenSearch) and aggregates the results; and (3) Reranking and Generation, where a reranker (i.e., BGE) refines document selection before a final response is generated by an LLM (i.e., Falcon-10B) using a chain-of-thought prompt. Omni-RAG aims to bridge the gap between current RAG capabilities and the demands of real-world applications, such as those highlighted by the SIGIR 2025 LiveRAG Challenge, by robustly handling complex and noisy queries. 

**Abstract (ZH)**: 面向现实场景的查询增强生成（RAG）系统在处理 noisy、模糊且含有多个意图的用户查询时面临重大挑战。Omni-RAG：一种提高RAG系统在实时开放域设置中稳健性和有效性的新型框架 

---
# Detecting Referring Expressions in Visually Grounded Dialogue with Autoregressive Language Models 

**Title (ZH)**: 基于视觉接地对话的自回归语言模型中的引用表达检测 

**Authors**: Bram Willemsen, Gabriel Skantze  

**Link**: [PDF](https://arxiv.org/pdf/2506.21294)  

**Abstract**: In this paper, we explore the use of a text-only, autoregressive language modeling approach for the extraction of referring expressions from visually grounded dialogue. More specifically, the aim is to investigate the extent to which the linguistic context alone can inform the detection of mentions that have a (visually perceivable) referent in the visual context of the conversation. To this end, we adapt a pretrained large language model (LLM) to perform a relatively course-grained annotation of mention spans in unfolding conversations by demarcating mention span boundaries in text via next-token prediction. Our findings indicate that even when using a moderately sized LLM, relatively small datasets, and parameter-efficient fine-tuning, a text-only approach can be effective, highlighting the relative importance of the linguistic context for this task. Nevertheless, we argue that the task represents an inherently multimodal problem and discuss limitations fundamental to unimodal approaches. 

**Abstract (ZH)**: 本研究探索仅使用文本的自回归语言模型方法从视觉接地对话中提取指代表达。具体而言，旨在调查仅基于语言背景在对话视觉上下文中的提及（可视觉感知的指代）检测的程度。为此，我们适应一个预训练的大语言模型对展开对话中的提及 spans 进行相对粗粒度的标注，通过下一-token 预测划分提及 span 的边界。研究发现，即使使用中等规模的预训练语言模型、较小的数据集和参数高效微调，仅文本的方法也可以有效，突显了语言背景在此任务中的相对重要性。然而，我们认为该任务本质上是一种多模态问题，并讨论单一模态方法的基本局限性。 

---
# Small Encoders Can Rival Large Decoders in Detecting Groundedness 

**Title (ZH)**: 小编码器可以与大解码器媲美，用于检测接地性 

**Authors**: Istabrak Abbes, Gabriele Prato, Quentin Fournier, Fernando Rodriguez, Alaa Boukhary, Adam Elwood, Sarath Chandar  

**Link**: [PDF](https://arxiv.org/pdf/2506.21288)  

**Abstract**: Augmenting large language models (LLMs) with external context significantly improves their performance in natural language processing (NLP) tasks. However, LLMs struggle to answer queries reliably when the provided context lacks information, often resorting to ungrounded speculation or internal knowledge. Groundedness - generating responses strictly supported by the context - is essential for ensuring factual consistency and trustworthiness. This study focuses on detecting whether a given query is grounded in a document provided in context before the costly answer generation by LLMs. Such a detection mechanism can significantly reduce both inference time and resource consumption. We show that lightweight, task specific encoder models such as RoBERTa and NomicBERT, fine-tuned on curated datasets, can achieve accuracy comparable to state-of-the-art LLMs, such as Llama3 8B and GPT4o, in groundedness detection while reducing inference latency by orders of magnitude. The code is available at : this https URL 

**Abstract (ZH)**: 增强大型语言模型（LLMs）的外部上下文显著提高了其在自然语言处理（NLP）任务中的性能。然而，当提供的上下文缺乏信息时，LLMs往往难以可靠地回答查询，往往会 resort 到没有依据的推测或内部知识。基于上下文生成响应的可靠性 - 生成的响应必须严格基于提供的上下文 - 对确保事实一致性与可信度至关重要。本研究重点在于在LLMs生成昂贵的回答之前，检测给定查询是否基于提供的文档上下文，从而显著减少推理时间和资源消耗。我们展示了针对特定任务的轻量级编码器模型，如RoBERTa和NomicBERT，在经过精心策划的数据集微调后，可以在基于上下文的可靠性检测方面达到与最先进的LLMs（如Llama3 8B和GPT4o）相当的准确率，同时将推理延迟减少多个数量级。代码可在以下链接获取：this https URL。 

---
# DiLoCoX: A Low-Communication Large-Scale Training Framework for Decentralized Cluster 

**Title (ZH)**: DiLoCoX：去中心化聚类的低通信大规模训练框架 

**Authors**: Ji Qi, WenPeng Zhu, Li Li, Ming Wu, YingJun Wu, Wu He, Xun Gao, Jason Zeng, Michael Heinrich  

**Link**: [PDF](https://arxiv.org/pdf/2506.21263)  

**Abstract**: The distributed training of foundation models, particularly large language models (LLMs), demands a high level of communication. Consequently, it is highly dependent on a centralized cluster with fast and reliable interconnects. Can we conduct training on slow networks and thereby unleash the power of decentralized clusters when dealing with models exceeding 100 billion parameters? In this paper, we propose DiLoCoX, a low-communication large-scale decentralized cluster training framework. It combines Pipeline Parallelism with Dual Optimizer Policy, One-Step-Delay Overlap of Communication and Local Training, and an Adaptive Gradient Compression Scheme. This combination significantly improves the scale of parameters and the speed of model pre-training. We justify the benefits of one-step-delay overlap of communication and local training, as well as the adaptive gradient compression scheme, through a theoretical analysis of convergence. Empirically, we demonstrate that DiLoCoX is capable of pre-training a 107B foundation model over a 1Gbps network. Compared to vanilla AllReduce, DiLoCoX can achieve a 357x speedup in distributed training while maintaining negligible degradation in model convergence. To the best of our knowledge, this is the first decentralized training framework successfully applied to models with over 100 billion parameters. 

**Abstract (ZH)**: 分布式训练基础模型，尤其是大型语言模型（LLMs），需要高度的通信。因此，它高度依赖于一个快速可靠的集中式集群。在处理超过100亿参数的模型时，我们能否在慢网络上进行训练，并释放分散式集群的潜力？在这项研究中，我们提出DiLoCoX，一种低通信大规模分散式集群训练框架。该框架结合了管道并行性、双优化器策略、一步延迟重叠通信与本地训练以及自适应梯度压缩方案。这种结合显著提高了参数规模和模型预训练速度。我们通过收敛理论分析证明了一步延迟重叠通信与本地训练以及自适应梯度压缩方案的优势。实证结果表明，DiLoCoX能够在1Gbps网络上预训练一个107B的基模型。与vanilla AllReduce相比，DiLoCoX在分布式训练中的速度提高了357倍，同时对模型收敛性几乎没有影响。据我们所知，这是首次成功将分散式训练框架应用于超过100亿参数的模型。 

---
# $T^3$: Multi-level Tree-based Automatic Program Repair with Large Language Models 

**Title (ZH)**: $T^3$: 基于多级树结构的大语言模型自动程序修复 

**Authors**: Quanming Liu, Xupeng Bu, Zhichao Yan, Ru Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.21211)  

**Abstract**: Automatic Program Repair (APR) is a core technology in software development and maintenance, with aims to enable automated defect repair with minimal human intervention. In recent years, the substantial advancements in Large Language Models (LLMs) and the Chain-of-Thought (CoT) techniques have significantly enhanced the reasoning capabilities of these models. However, due to the complex logic and multi-step reasoning ability needed, the application of CoT techniques in the APR domain remains insufficient. This study systematically evaluates the performance of several common CoT techniques in APR tasks and proposes an innovative framework $T^3$, which integrates the powerful reasoning capabilities of LLMs with tree search, effectively improving the precision of generating candidate repair solutions. Furthermore, $T^3$ provides valuable guidance for optimizing sample selection and repair strategies in APR tasks, establishing a robust framework for achieving efficient automated debugging. 

**Abstract (ZH)**: Automatic程序修复中的Chain-of-Thought技术系统评价与T³框架 

---
# How Good Are Synthetic Requirements ? Evaluating LLM-Generated Datasets for AI4RE 

**Title (ZH)**: 合成需求的质量如何？评估AI4RE中生成的数据集 

**Authors**: Abdelkarim El-Hajjami, Camille Salinesi  

**Link**: [PDF](https://arxiv.org/pdf/2506.21138)  

**Abstract**: The shortage of publicly available, labeled requirements datasets remains a major barrier to advancing Artificial Intelligence for Requirements Engineering (AI4RE). While Large Language Models offer promising capabilities for synthetic data generation, systematic approaches to control and optimize the quality of generated requirements remain underexplored. This paper presents Synthline v1, an enhanced Product Line approach for generating synthetic requirements data that extends our earlier v0 version with advanced generation strategies and curation techniques. We investigate four research questions assessing how prompting strategies, automated prompt optimization, and post-generation curation affect data quality across four classification tasks: defect detection, functional vs. non-functional, quality vs. non-quality, and security vs. non-security. Our evaluation shows that multi-sample prompting significantly boosts both utility and diversity over single-sample generation, with F1-score gains from 6 to 44 points. The use of PACE (Prompt Actor-Critic Editing) for automated prompt optimization yields task-dependent results, greatly improving functional classification (+32.5 points) but reducing performance on others. Interestingly, similarity-based curation improves diversity but often harms classification performance, indicating that some redundancy may help ML models. Most importantly, our results show that synthetic requirements can match or outperform human-authored ones for specific tasks, with synthetic data surpassing human data for security (+7.8 points) and defect classification (+15.4 points). These findings offer practical insights for AI4RE and chart a viable path to mitigating dataset scarcity through systematic synthetic generation. 

**Abstract (ZH)**: 公开标注的需求数据集短缺仍然是推进人工智能在需求工程中的应用的主要障碍。尽管大型语言模型提供了生成合成数据的潜力，但控制和优化生成需求质量的系统方法仍处于探索阶段。本文介绍了Synthline v1，这是一种增强的产品线方法，用于生成合成需求数据，该方法在先前的v0版本基础上，增加了先进的生成策略和编目技术。我们探讨了四种研究问题，评估了提示策略、自动化提示优化和生成后编目的数据质量对四种分类任务（缺陷检测、功能性 vs 非功能性、质量 vs 非质量、安全 vs 非安全）的影响。评估结果显示，多样本提示在提升实用性和多样性方面显著优于单样本生成，F1分数提高了6到44个百分点。使用PACE（提示演员-评论者编辑）进行自动化提示优化在不同任务上表现出依赖性结果，极大提高了功能性分类（+32.5个百分点），但其他任务的表现有所下降。有趣的是，基于相似性的编目可以提高多样性，但往往会损害分类性能，表明一些冗余可能有助于机器学习模型。最重要的是，我们的研究结果表明，合成需求数据在特定任务上可以与甚至超过人工撰写的数据的表现，合成数据在安全性（+7.8个百分点）和缺陷分类（+15.4个百分点）方面超过了人类数据。这些发现为人工智能在需求工程中的应用提供了实际见解，并明确了通过系统合成生成缓解数据稀缺性的可行路径。 

---
# Progtuning: Progressive Fine-tuning Framework for Transformer-based Language Models 

**Title (ZH)**: Proggtuning: 基于 Transformer 的语言模型逐级微调框架 

**Authors**: Xiaoshuang Ji, Zhendong Zhao, Xiaojun Chen, Xin Zhao, Zeyao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21119)  

**Abstract**: Fine-tuning is a promising technique for leveraging Transformer-based language models in downstream tasks. As model sizes continue to grow, updating all model parameters becomes increasingly costly. Parameter-efficient fine-tuning methods effectively address this issue by selectively updating a small subset of parameters. However, fine-tuning and most existing parameter-efficient fine-tuning methods require updating the same number of parameters as the initial size, ignoring the unequal contribution across Transformer blocks and leading to extremely inefficient allocation of computing resources. In this paper, we propose Progtuning, the novel fine-tuning framework combined with progressive learning for Transformer-based language models. Specifically, Progtuning progressively reduces the number of updated transformer blocks based on the contribution. Remarkably, Progtuning optimizes resource allocation and reduces the number of updated parameters by approximately 25\%, while still maintaining competitive performance. And it also exhibits high adaptability with parameter-efficient fine-tuning methods, demonstrating excellent performance across various adaptation scenarios. 

**Abstract (ZH)**: Progressuning：基于渐进学习的参数高效微调框架 

---
# Large Language Models Acing Chartered Accountancy 

**Title (ZH)**: 大型语言模型在专业会计师考试中的表现 

**Authors**: Jatin Gupta, Akhil Sharma, Saransh Singhania, Mohammad Adnan, Sakshi Deo, Ali Imam Abidi, Keshav Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.21031)  

**Abstract**: Advanced intelligent systems, particularly Large Language Models (LLMs), are significantly reshaping financial practices through advancements in Natural Language Processing (NLP). However, the extent to which these models effectively capture and apply domain-specific financial knowledge remains uncertain. Addressing a critical gap in the expansive Indian financial context, this paper introduces CA-Ben, a Chartered Accountancy benchmark specifically designed to evaluate the financial, legal, and quantitative reasoning capabilities of LLMs. CA-Ben comprises structured question-answer datasets derived from the rigorous examinations conducted by the Institute of Chartered Accountants of India (ICAI), spanning foundational, intermediate, and advanced CA curriculum stages. Six prominent LLMs i.e. GPT 4o, LLAMA 3.3 70B, LLAMA 3.1 405B, MISTRAL Large, Claude 3.5 Sonnet, and Microsoft Phi 4 were evaluated using standardized protocols. Results indicate variations in performance, with Claude 3.5 Sonnet and GPT-4o outperforming others, especially in conceptual and legal reasoning. Notable challenges emerged in numerical computations and legal interpretations. The findings emphasize the strengths and limitations of current LLMs, suggesting future improvements through hybrid reasoning and retrieval-augmented generation methods, particularly for quantitative analysis and accurate legal interpretation. 

**Abstract (ZH)**: 高级智能系统，特别是大规模语言模型（LLMs），通过自然语言处理（NLP）的进步正在显著重塑金融实践。然而，这些模型在有效地捕捉和应用特定领域金融知识方面的程度仍然不确定。填补广阔的印度金融背景中的一个关键空白，本文引入了CA-Ben，一个专门用于评估LLMs的财务、法律和定量推理能力的特许会计师基准。CA-Ben包括源自印度特许会计师协会（ICAI）严格考试的结构化问答数据集，涵盖特许会计师课程的基础、中级和高级阶段。六种知名的LLMs，即GPT 4o、LLAMA 3.3 70B、LLAMA 3.1 405B、MISTRAL Large、Claude 3.5 Sonnet 和 Microsoft Phi 4，使用标准化流程进行了评估。结果显示性能存在差异，Claude 3.5 Sonnet 和 GPT-4o 在概念性和法律推理方面表现突出。在数值计算和法律解释方面出现了显著挑战。研究结果强调了当前LLMs的优势和局限性，建议通过混合推理和检索增强生成方法在未来进行改进，特别是在定量分析和准确法律解释方面。 

---
# SAC: A Framework for Measuring and Inducing Personality Traits in LLMs with Dynamic Intensity Control 

**Title (ZH)**: SAC：一种动态强度控制的测量和诱导LLMs人格特质的框架 

**Authors**: Adithya Chittem, Aishna Shrivastava, Sai Tarun Pendela, Jagat Sesh Challa, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.20993)  

**Abstract**: Large language models (LLMs) have gained significant traction across a wide range of fields in recent years. There is also a growing expectation for them to display human-like personalities during interactions. To meet this expectation, numerous studies have proposed methods for modelling LLM personalities through psychometric evaluations. However, most existing models face two major limitations: they rely on the Big Five (OCEAN) framework, which only provides coarse personality dimensions, and they lack mechanisms for controlling trait intensity. In this paper, we address this gap by extending the Machine Personality Inventory (MPI), which originally used the Big Five model, to incorporate the 16 Personality Factor (16PF) model, allowing expressive control over sixteen distinct traits. We also developed a structured framework known as Specific Attribute Control (SAC) for evaluating and dynamically inducing trait intensity in LLMs. Our method introduces adjective-based semantic anchoring to guide trait intensity expression and leverages behavioural questions across five intensity factors: \textit{Frequency}, \textit{Depth}, \textit{Threshold}, \textit{Effort}, and \textit{Willingness}. Through experimentation, we find that modelling intensity as a continuous spectrum yields substantially more consistent and controllable personality expression compared to binary trait toggling. Moreover, we observe that changes in target trait intensity systematically influence closely related traits in psychologically coherent directions, suggesting that LLMs internalize multi-dimensional personality structures rather than treating traits in isolation. Our work opens new pathways for controlled and nuanced human-machine interactions in domains such as healthcare, education, and interviewing processes, bringing us one step closer to truly human-like social machines. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多个领域中获得了显著的关注，人们对它们在交互中展示类人类个性寄予了越来越高的期望。为了满足这一期望，许多研究提出了通过心理测评建模LLMs个性的方法。然而，现有大多数模型面临两大局限：依赖于大五人格（OCEAN）模型，只能提供粗略的个性维度，且缺乏调控特质强度的机制。本文通过将原始使用大五人格模型的机器个性量表（MPI）扩展至整合16人格因素（16PF）模型，填补了这一空白，从而实现了对十六种不同特质的表达性控制。我们还开发了一个结构化的框架，名为特定属性控制（SAC），用于评估和动态诱导LLMs的特质强度。该方法通过形容词基础的语义锚定来引导特质强度的表达，并利用五个强度因素（Frequency、Depth、Threshold、Effort、Willingness）的行为性问题进行衡量。通过实验发现，将特质强度建模为连续谱系能显著提高个性表达的连贯性和可控性，相较于二元特质切换方法。此外，我们观察到目标特质强度的变化系统地影响了其他相关特质，并在心理连贯的方向上产生了调节，表明LLMs具备多维度的个性结构而非孤立地对待特质。我们的工作为医疗、教育和面试等领域中受控和细致的人机交互开辟了新的路径，使我们更接近于真正的类人类社会机器人。 

---
# LLM-guided Chemical Process Optimization with a Multi-Agent Approach 

**Title (ZH)**: 使用多Agent方法的LLM引导化学过程优化 

**Authors**: Tong Zeng, Srivathsan Badrinarayanan, Janghoon Ock, Cheng-Kai Lai, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2506.20921)  

**Abstract**: Chemical process optimization is crucial to maximize production efficiency and economic performance. Traditional methods, including gradient-based solvers, evolutionary algorithms, and parameter grid searches, become impractical when operating constraints are ill-defined or unavailable, requiring engineers to rely on subjective heuristics to estimate feasible parameter ranges. To address this constraint definition bottleneck, we present a multi-agent framework of large language model (LLM) agents that autonomously infer operating constraints from minimal process descriptions, then collaboratively guide optimization using the inferred constraints. Our AutoGen-based agentic framework employs OpenAI's o3 model, with specialized agents for constraint generation, parameter validation, simulation execution, and optimization guidance. Through two phases - autonomous constraint generation using embedded domain knowledge, followed by iterative multi-agent optimization - the framework eliminates the need for predefined operational bounds. Validated on the hydrodealkylation process across cost, yield, and yield-to-cost ratio metrics, the framework demonstrated competitive performance with conventional optimization methods while achieving better computational efficiency, requiring fewer iterations to converge. Our approach converged in under 20 minutes, achieving a 31-fold speedup over grid search. Beyond computational efficiency, the framework's reasoning-guided search demonstrates sophisticated process understanding, correctly identifying utility trade-offs, and applying domain-informed heuristics. This approach shows significant potential for optimization scenarios where operational constraints are poorly characterized or unavailable, particularly for emerging processes and retrofit applications. 

**Abstract (ZH)**: 化学过程优化是最大化生产效率和经济效益的关键。为了克服操作约束定义不明确或不可用的瓶颈，我们提出了一种基于多代理框架的大型语言模型（LLM）代理体系，该框架能够自动从最小的过程描述中推断出操作约束，并在此基础上协作指导优化。基于AutoGen的代理框架使用OpenAI的o3模型，具有专门用于约束生成、参数验证、仿真执行和优化指导的代理。通过两个阶段——利用嵌入的领域知识自主生成约束，随后进行迭代的多代理优化——该框架消除了预先定义的操作界限的需求。该框架在成本、产率和产率-成本比等指标上验证了与传统优化方法具有竞争力的性能，并且具有更好的计算效率，所需的迭代次数更少以达到收敛。我们的方法在不到20分钟内收敛，比网格搜索快了31倍。除了计算效率之外，该框架的推理引导搜索展示了复杂的工艺理解能力，能够正确识别能量贸易平衡，并应用基于领域的启发式方法。该方法对于操作约束描述不足或不可用的优化场景具有显著潜力，特别适用于新兴工艺和改造应用。 

---
# Optimising Language Models for Downstream Tasks: A Post-Training Perspective 

**Title (ZH)**: 优化语言模型以适应下游任务：一种后训练视角 

**Authors**: Zhengyan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20917)  

**Abstract**: Language models (LMs) have demonstrated remarkable capabilities in NLP, yet adapting them efficiently and robustly to specific tasks remains challenging. As their scale and complexity grow, fine-tuning LMs on labelled data often underutilizes available unlabelled data, leads to overfitting on small task-specific sets, and imposes significant computational costs. These limitations hamper their application to the open-ended landscape of real-world language tasks.
This thesis proposes a series of methods to better adapt LMs to downstream applications. First, we explore strategies for extracting task-relevant knowledge from unlabelled data, introducing a novel continued pre-training technique that outperforms state-of-the-art semi-supervised approaches. Next, we present a parameter-efficient fine-tuning method that substantially reduces memory and compute costs while maintaining competitive performance. We also introduce improved supervised fine-tuning methods that enable LMs to better follow instructions, especially when labelled data is scarce, enhancing their performance across a range of NLP tasks, including open-ended generation. Finally, we develop new evaluation methods and benchmarks, such as multi-hop spatial reasoning tasks, to assess LM capabilities and adaptation more comprehensively.
Through extensive empirical studies across diverse NLP tasks, our results demonstrate that these approaches substantially improve LM robustness, efficiency, and generalization, making them more adaptable to a broad range of applications. These advances mark a significant step towards more robust and efficient LMs, bringing us closer to the goal of artificial general intelligence. 

**Abstract (ZH)**: 语言模型（LMs）在自然语言处理（NLP）中展现了卓越的能力，但如何高效且稳健地将其适应到特定任务仍然具有挑战性。随着它们规模和复杂性的增加，在标注数据上微调LMs往往未能充分利用可用的未标注数据、在小规模任务特定集上容易过拟合，并且会产生重大的计算成本。这些限制阻碍了其在现实世界语言任务开放场景中的应用。

本论文提出了一系列方法以更好地将LMs适配到下游应用。首先，我们探索从未标注数据中提取与任务相关知识的策略，引入了一种新型的连续预训练技术，该技术在半监督方法中表现出色。接着，我们提出了一种参数效率更高的微调方法，该方法大幅减少了内存和计算成本，同时保持了竞争力的性能。我们还介绍了改进的监督微调方法，使LMs能够更好地遵循指令，尤其是在标注数据稀缺的情况下，提高了其在各种NLP任务中的性能，包括开放生成。最后，我们开发了新的评估方法和基准测试，如多跳空间推理任务，以更全面地评估LM的能力和适应性。

通过在多样化的NLP任务上进行广泛的实证研究，我们的结果表明，这些方法显著提升了LMs的稳健性、效率和泛化能力，使其能够更好地适应更广泛的用途。这些进展标志着向更稳健和高效的LMs迈进的重要一步，使我们更接近人工通用智能的目标。 

---
# ZKPROV: A Zero-Knowledge Approach to Dataset Provenance for Large Language Models 

**Title (ZH)**: ZKPROV: 零知识环境下的大规模语言模型数据溯源方法 

**Authors**: Mina Namazi, Alexander Nemecek, Erman Ayday  

**Link**: [PDF](https://arxiv.org/pdf/2506.20915)  

**Abstract**: As the deployment of large language models (LLMs) grows in sensitive domains, ensuring the integrity of their computational provenance becomes a critical challenge, particularly in regulated sectors such as healthcare, where strict requirements are applied in dataset usage. We introduce ZKPROV, a novel cryptographic framework that enables zero-knowledge proofs of LLM provenance. It allows users to verify that a model is trained on a reliable dataset without revealing sensitive information about it or its parameters. Unlike prior approaches that focus on complete verification of the training process (incurring significant computational cost) or depend on trusted execution environments, ZKPROV offers a distinct balance. Our method cryptographically binds a trained model to its authorized training dataset(s) through zero-knowledge proofs while avoiding proof of every training step. By leveraging dataset-signed metadata and compact model parameter commitments, ZKPROV provides sound and privacy-preserving assurances that the result of the LLM is derived from a model trained on the claimed authorized and relevant dataset. Experimental results demonstrate the efficiency and scalability of the ZKPROV in generating this proof and verifying it, achieving a practical solution for real-world deployments. We also provide formal security guarantees, proving that our approach preserves dataset confidentiality while ensuring trustworthy dataset provenance. 

**Abstract (ZH)**: 大语言模型（LLMs）在敏感领域部署增长背景下，确保其计算溯源的完整性成为一个关键挑战，尤其是在受严格数据使用要求限制的医疗等领域。我们提出了ZKPROV，一种新的加密框架，能够提供LLM溯源的零知识证明。该框架允许用户验证模型是否基于可靠的训练数据集，而不泄露有关该数据集或其参数的敏感信息。与专注于整个训练过程完全验证（带来显著计算成本）或依赖受信任的执行环境的先前方法不同，ZKPROV 提供了一种独特的平衡。我们的方法通过零知识证明将训练后的模型绑定到其授权的训练数据集，同时避免验证每个训练步骤。借助数据集签名的元数据和紧凑的模型参数承诺，ZKPROV 提供了关于LLM结果源自声称授权的相关数据集的稳健且隐私保护的保证。实验结果展示了ZKPROV 在生成和验证此证明方面的高效性和可扩展性，实现了一种适用于实际部署的实用解决方案。我们还提供了正式的安全保证，证明了我们的方法在保护数据集机密性的同时确保可信的数据集溯源。 

---
# Omniwise: Predicting GPU Kernels Performance with LLMs 

**Title (ZH)**: Omniwise：使用大规模语言模型预测GPU内核性能 

**Authors**: Zixian Wang, Cole Ramos, Muhammad A. Awad, Keith Lowery  

**Link**: [PDF](https://arxiv.org/pdf/2506.20886)  

**Abstract**: In recent years, the rapid advancement of deep neural networks (DNNs) has revolutionized artificial intelligence, enabling models with unprecedented capabilities in understanding, generating, and processing complex data. These powerful architectures have transformed a wide range of downstream applications, tackling tasks beyond human reach. In this paper, we introduce Omniwise, the first end-to-end, self-supervised fine-tuning pipeline that applies large language models (LLMs) to GPU kernel performance prediction--a novel use case in performance profiling. Omniwise is model-agnostic and lightweight, achieving strong results even with a small 3B-parameter model. It can predict key performance metrics, including memory bandwidth, cache hit rates, GFLOPs, and arithmetic intensity, directly from kernel code without the need for code execution or profiling tools. Our approach achieves over 90% of predictions within 10% relative error on GPU kernels executed on AMD MI250 and MI300X architectures. In addition to the pipeline, we develop an online inference server and a Visual Studio Code plugin that seamlessly integrate LLM-based performance prediction into developers' workflows. 

**Abstract (ZH)**: 近年来，深度神经网络（DNNs）的快速进步颠覆了人工智能，使模型在理解和生成复杂数据方面具备前所未有的能力。这些强大的架构已经转变了广泛的应用领域，解决了一些超出了人类能力范围的任务。在本文中，我们介绍了Omniwise，这是首个用于通过大型语言模型（LLMs）进行GPU内核性能预测的端到端自我监督微调管道——这是一个新的性能分析用例。Omniwise是模型无关的且轻量级的，即使使用一个小型的3B参数模型也能取得优异的结果。它可以仅从内核代码预测关键性能指标，如内存带宽、缓存命中率、GFLOPs和算术强度，而无需代码执行或性能分析工具。我们的方法在AMD MI250和MI300X架构上执行的GPU内核中的预测相对误差小于10%时，其准确率超过90%。此外，我们还开发了一个在线推理服务和一个Visual Studio Code插件，无缝地将基于LLM的性能预测集成到开发者的 workflows 中。 

---
# Engineering RAG Systems for Real-World Applications: Design, Development, and Evaluation 

**Title (ZH)**: 面向实际应用的RAG系统工程：设计、开发与评估 

**Authors**: Md Toufique Hasan, Muhammad Waseem, Kai-Kristian Kemell, Ayman Asad Khan, Mika Saari, Pekka Abrahamsson  

**Link**: [PDF](https://arxiv.org/pdf/2506.20869)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are emerging as a key approach for grounding Large Language Models (LLMs) in external knowledge, addressing limitations in factual accuracy and contextual relevance. However, there is a lack of empirical studies that report on the development of RAG-based implementations grounded in real-world use cases, evaluated through general user involvement, and accompanied by systematic documentation of lessons learned. This paper presents five domain-specific RAG applications developed for real-world scenarios across governance, cybersecurity, agriculture, industrial research, and medical diagnostics. Each system incorporates multilingual OCR, semantic retrieval via vector embeddings, and domain-adapted LLMs, deployed through local servers or cloud APIs to meet distinct user needs. A web-based evaluation involving a total of 100 participants assessed the systems across six dimensions: (i) Ease of Use, (ii) Relevance, (iii) Transparency, (iv) Responsiveness, (v) Accuracy, and (vi) Likelihood of Recommendation. Based on user feedback and our development experience, we documented twelve key lessons learned, highlighting technical, operational, and ethical challenges affecting the reliability and usability of RAG systems in practice. 

**Abstract (ZH)**: 基于检索增强生成的领域特定应用：面向实际场景的Large Language Models的开发与评价 

---
# Uncovering Hidden Violent Tendencies in LLMs: A Demographic Analysis via Behavioral Vignettes 

**Title (ZH)**: 揭露LLM中隐藏的暴力倾向：基于行为情境的 demographic 分析 

**Authors**: Quintin Myers, Yanjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.20822)  

**Abstract**: Large language models (LLMs) are increasingly proposed for detecting and responding to violent content online, yet their ability to reason about morally ambiguous, real-world scenarios remains underexamined. We present the first study to evaluate LLMs using a validated social science instrument designed to measure human response to everyday conflict, namely the Violent Behavior Vignette Questionnaire (VBVQ). To assess potential bias, we introduce persona-based prompting that varies race, age, and geographic identity within the United States. Six LLMs developed across different geopolitical and organizational contexts are evaluated under a unified zero-shot setting. Our study reveals two key findings: (1) LLMs surface-level text generation often diverges from their internal preference for violent responses; (2) their violent tendencies vary across demographics, frequently contradicting established findings in criminology, social science, and psychology. 

**Abstract (ZH)**: 大规模语言模型（LLMs）被越来越多地提出用于检测和响应网络上的暴力内容，但它们在推理道德模糊的实际场景方面的能力仍然没有得到充分研究。我们首次使用一个经过验证的社会科学工具来评估LLMs，该工具旨在衡量人们对日常冲突的反应，即暴力行为情景问卷（VBVQ）。为了评估潜在偏见，我们引入了基于人物的提示，这些提示在美国范围内根据不同种族、年龄和地域身份进行变化。六种在不同的地缘政治和组织背景下开发的LLMs在统一的零样本设置下进行了评估。我们的研究揭示了两个关键发现：(1) LLMs的表面级文本生成往往与其内部偏好暴力反应相偏离；(2) 它们的暴力倾向在不同的人口统计学群体中各不相同，经常与犯罪学、社会科学和心理学中的既定发现相矛盾。 

---
# GPU Kernel Scientist: An LLM-Driven Framework for Iterative Kernel Optimization 

**Title (ZH)**: GPU内核科学家：一个基于LLM的迭代内核优化框架 

**Authors**: Martin Andrews, Sam Witteveen  

**Link**: [PDF](https://arxiv.org/pdf/2506.20807)  

**Abstract**: Optimizing GPU kernels for high performance is a complex task, often demanding deep architectural knowledge, extensive profiling, and iterative experimentation. This challenge is amplified when targeting newer or less-documented GPU architectures where traditional development aids are scarce. This paper introduces an LLM-powered "GPU Kernel Scientist," an automated methodology for iteratively refining accelerator kernels.
Our methodology employs LLMs in a multi-stage, evolutionary process: (a) strategically selecting promising prior code versions as a basis for new iterations; (b) generating hypotheses for optimization experiments, based on existing code and assimilated knowledge from general GPU literature; and (c) autonomously implementing these experiments through code modification and subsequent submission to an external evaluation system, using only observed timing data as performance feedback. We detail how this approach navigates the challenges of the AMD MI300 target architecture and leverages LLMs to compensate for limited domain-specific human expertise.
Since quantitative results from an ongoing performance competition were embargoed on paper submission date, we present the architectural design, operational workflow, and qualitative insights, highlighting the potential of LLM-driven agents to democratise and accelerate GPU kernel optimization, especially in resource-constrained or rapidly evolving hardware environments. 

**Abstract (ZH)**: 基于LLM的“GPU内核科学家”：一种自动迭代优化加速器内核的方法学 

---
# Poster: Enhancing GNN Robustness for Network Intrusion Detection via Agent-based Analysis 

**Title (ZH)**: Poster: 基于代理分析增强GNN在网络入侵检测中的鲁棒性 

**Authors**: Zhonghao Zhan, Huichi Zhou, Hamed Haddadi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20806)  

**Abstract**: Graph Neural Networks (GNNs) show great promise for Network Intrusion Detection Systems (NIDS), particularly in IoT environments, but suffer performance degradation due to distribution drift and lack robustness against realistic adversarial attacks. Current robustness evaluations often rely on unrealistic synthetic perturbations and lack demonstrations on systematic analysis of different kinds of adversarial attack, which encompass both black-box and white-box scenarios. This work proposes a novel approach to enhance GNN robustness and generalization by employing Large Language Models (LLMs) in an agentic pipeline as simulated cybersecurity expert agents. These agents scrutinize graph structures derived from network flow data, identifying and potentially mitigating suspicious or adversarially perturbed elements before GNN processing. Our experiments, using a framework designed for realistic evaluation and testing with a variety of adversarial attacks including a dataset collected from physical testbed experiments, demonstrate that integrating LLM analysis can significantly improve the resilience of GNN-based NIDS against challenges, showcasing the potential of LLM agent as a complementary layer in intrusion detection architectures. 

**Abstract (ZH)**: 基于大型语言模型的代理pipeline增强图神经网络在网络入侵检测系统中的 robustness 和泛化能力研究 

---
# The Ideation-Execution Gap: Execution Outcomes of LLM-Generated versus Human Research Ideas 

**Title (ZH)**: LLM生成的构思与人类研究构思的执行差距：执行结果对比 

**Authors**: Chenglei Si, Tatsunori Hashimoto, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20803)  

**Abstract**: Large Language Models (LLMs) have shown promise in accelerating the scientific research pipeline. A key capability for this process is the ability to generate novel research ideas, and prior studies have found settings in which LLM-generated research ideas were judged as more novel than human-expert ideas. However, a good idea should not simply appear to be novel, it should also result in better research after being executed. To test whether AI-generated ideas lead to better research outcomes, we conduct an execution study by recruiting 43 expert researchers to execute randomly-assigned ideas, either written by experts or generated by an LLM. Each expert spent over 100 hours implementing the idea and wrote a 4-page short paper to document the experiments. All the executed projects are then reviewed blindly by expert NLP researchers. Comparing the review scores of the same ideas before and after execution, the scores of the LLM-generated ideas decrease significantly more than expert-written ideas on all evaluation metrics (novelty, excitement, effectiveness, and overall; p < 0.05), closing the gap between LLM and human ideas observed at the ideation stage. When comparing the aggregated review scores from the execution study, we even observe that for many metrics there is a flip in rankings where human ideas score higher than LLM ideas. This ideation-execution gap highlights the limitations of current LLMs in generating truly effective research ideas and the challenge of evaluating research ideas in the absence of execution outcomes. 

**Abstract (ZH)**: 大型语言模型在加速科学研究管道方面显示出了潜力。这一过程的关键能力是生成新颖的研究想法，先前的研究发现，由大型语言模型生成的研究想法被判断为比人类专家的想法更具新颖性。然而，一个好的想法不仅应该看似新颖，还应该在执行后能够带来更好的研究成果。为了测试人工智能生成的想法是否能导致更优秀的研究成果，我们通过招募43位专家研究人员来执行随机分配的想法，这些想法要么由专家编写，要么由大型语言模型生成。每位专家花费超过100小时来实现想法，并撰写了4页的简短论文来记录实验。所有执行的项目随后由专家自然语言处理研究人员盲审。通过比较执行前后相同想法的评审分数，大型语言模型生成的想法在所有评价指标（新颖性、兴奋性、有效性以及总体评分）上的得分下降显著（p < 0.05），这表明执行阶段的结果缩小了在想法产生阶段观察到的大型语言模型与人类想法之间的差距。当比较执行研究中的综合评审分数时，我们甚至观察到许多指标中人类想法得分高于大型语言模型想法的情况。这个想法执行差异突显了当前大型语言模型生成真正有效的研究想法的局限性，以及在缺乏执行结果的情况下评估研究想法的挑战。 

---
# Test-time Scaling Techniques in Theoretical Physics -- A Comparison of Methods on the TPBench Dataset 

**Title (ZH)**: 测试时缩放技术在理论物理中的研究——TPBench数据集上方法比较 

**Authors**: Zhiqi Gao, Tianyi Li, Yurii Kvasiuk, Sai Chaitanya Tadepalli, Maja Rudolph, Daniel J.H. Chung, Frederic Sala, Moritz Münchmeyer  

**Link**: [PDF](https://arxiv.org/pdf/2506.20729)  

**Abstract**: Large language models (LLMs) have shown strong capabilities in complex reasoning, and test-time scaling techniques can enhance their performance with comparably low cost. Many of these methods have been developed and evaluated on mathematical reasoning benchmarks such as AIME. This paper investigates whether the lessons learned from these benchmarks generalize to the domain of advanced theoretical physics. We evaluate a range of common test-time scaling methods on the TPBench physics dataset and compare their effectiveness with results on AIME. To better leverage the structure of physics problems, we develop a novel, symbolic weak-verifier framework to improve parallel scaling results. Our empirical results demonstrate that this method significantly outperforms existing test-time scaling approaches on TPBench. We also evaluate our method on AIME, confirming its effectiveness in solving advanced mathematical problems. Our findings highlight the power of step-wise symbolic verification for tackling complex scientific problems. 

**Abstract (ZH)**: 大型语言模型（LLMs）在复杂推理方面展现了强大的能力，测试时扩展技术可以在较低成本的情况下提升其性能。这些方法大多是在AIME等数学推理基准上开发和评估的。本文探讨了这些基准上的经验教训是否能够推广到高级理论物理领域。我们评估了多种常见的测试时扩展方法在TPBench物理数据集上的效果，并将其与在AIME上的结果进行比较。为了更好地利用物理问题的结构，我们开发了一种新的符号弱验证框架，以改善并行扩展的结果。我们的实验结果表明，该方法在TPBench上的表现显著优于现有的测试时扩展方法。我们还在AIME上测试了该方法，证实了其在解决高级数学问题方面的有效性。我们的研究突显了逐步符号验证在应对复杂科学问题方面的潜力。 

---
# Utility-Driven Speculative Decoding for Mixture-of-Experts 

**Title (ZH)**: 基于利用率推测性解码的混合专家模型 

**Authors**: Anish Saxena, Po-An Tsai, Hritvik Taneja, Aamer Jaleel, Moinuddin Qureshi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20675)  

**Abstract**: GPU memory bandwidth is the main bottleneck for low-latency Large Language Model (LLM) inference. Speculative decoding leverages idle GPU compute by using a lightweight drafter to propose K tokens, which the LLM verifies in parallel, boosting token throughput. In conventional dense LLMs, all model weights are fetched each iteration, so speculation adds no latency overhead. Emerging Mixture of Experts (MoE) models activate only a subset of weights per token, greatly reducing data movement. However, we show that speculation is ineffective for MoEs: draft tokens collectively activate more weights, increasing data movement and verification time by 2-3x. When token throughput gains fail to offset this overhead, speculation causes slowdowns up to 1.5x, making it infeasible. Even when useful, the optimal K varies by task, model, and even between requests and iterations. Thus, despite widespread use in dense LLMs, speculation remains impractical in leading MoEs.
We present Cascade, a utility-driven framework that selectively enables speculation to avoid slowdowns and dynamically tunes K to accelerate MoE serving. Cascade uses a lightweight metric, speculation utility, the ratio of token gains to verification cost, which shows iteration-level locality, enabling periodic decisions via short test and longer set phases. For each request, Cascade disables speculation if utility drops below one during testing, and when utility exceeds one, tests multiple K-values to choose the utility-maximizing K for the set phase. We implement Cascade in vLLM and evaluate it on five popular MoEs with workloads spanning code, math, extraction, and mixed tasks. Cascade limits slowdown to 5% (vs. 1.5x) and improves throughput by 7-14% over static K, making speculative decoding practical for MoEs. 

**Abstract (ZH)**: GPU内存带宽是低延迟大语言模型（LLM）推理的主要瓶颈。推测性解码通过使用轻量级草案器提出K个令牌，由LLM并行验证，从而提升令牌吞吐量。在传统的密集型LLM中，每迭代一次都会获取所有模型权重，因此推测不会增加延迟开销。新兴的专家混合（MoE）模型每次仅激活权重子集，大大减少了数据移动。然而，我们发现推测对于MoE无效：草案令牌共同激活更多权重，增加数据移动和验证时间2-3倍。当令牌吞吐量增益无法抵消这种开销时，推测会导致多达1.5倍的性能下降，使其不可行。即使有用，最优的K值也因任务、模型甚至不同请求和迭代而异。因此，尽管在密集型LLM中广泛应用，推测对于领先的MoE仍不可行。

我们提出了Cascade，一个以用途为导向的框架，选择性地启用推测以避免性能下降，并动态调整K值以加快MoE服务。Cascade使用一个轻量级指标——推测用途，即令牌增益与验证成本的比率，展示迭代级别的局部性，通过短测试期和较长设定期周期性地进行决策。对于每个请求，Cascade在测试过程中如果用途低于一个阈值则禁用推测，并在用途超过一个阈值时测试多个K值，选择设定期中用途最大的K值。我们在vLLM中实现Cascade，并在涵盖代码、数学、提取和混合任务的五个流行MoE上进行评估。Cascade将性能下降限制在5%（相对于1.5倍），并相对于固定K值改善吞吐量7-14%，使推测性解码对于MoE变得可行。 

---
