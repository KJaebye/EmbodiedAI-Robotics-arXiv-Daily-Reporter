# Enhancing Autonomous Driving Systems with On-Board Deployed Large Language Models 

**Title (ZH)**: 在车载部署的大语言模型增强自主驾驶系统 

**Authors**: Nicolas Baumann, Cheng Hu, Paviththiren Sivasothilingam, Haotong Qin, Lei Xie, Michele Magno, Luca Benini  

**Link**: [PDF](https://arxiv.org/pdf/2504.11514)  

**Abstract**: Neural Networks (NNs) trained through supervised learning struggle with managing edge-case scenarios common in real-world driving due to the intractability of exhaustive datasets covering all edge-cases, making knowledge-driven approaches, akin to how humans intuitively detect unexpected driving behavior, a suitable complement to data-driven methods. This work proposes a hybrid architecture combining low-level Model Predictive Controller (MPC) with locally deployed Large Language Models (LLMs) to enhance decision-making and Human Machine Interaction (HMI). The DecisionxLLM module evaluates robotic state information against natural language instructions to ensure adherence to desired driving behavior. The MPCxLLM module then adjusts MPC parameters based on LLM-generated insights, achieving control adaptability while preserving the safety and constraint guarantees of traditional MPC systems. Further, to enable efficient on-board deployment and to eliminate dependency on cloud connectivity, we shift processing to the on-board computing platform: We propose an approach that exploits Retrieval Augmented Generation (RAG), Low Rank Adaptation (LoRA) fine-tuning, and quantization. Experimental results demonstrate that these enhancements yield significant improvements in reasoning accuracy by up to 10.45%, control adaptability by as much as 52.2%, and up to 10.5x increase in computational efficiency (tokens/s), validating the proposed framework's practicality for real-time deployment even on down-scaled robotic platforms. This work bridges high-level decision-making with low-level control adaptability, offering a synergistic framework for knowledge-driven and adaptive Autonomous Driving Systems (ADS). 

**Abstract (ZH)**: 基于知识驱动和适应性的自主驾驶系统中的混合架构：结合低层次模型预测控制与局部部署的大语言模型 

---
# Towards LLM Agents for Earth Observation 

**Title (ZH)**: 面向地球观测的大型语言模型代理 

**Authors**: Chia Hsiang Kao, Wenting Zhao, Shreelekha Revankar, Samuel Speas, Snehal Bhagat, Rajeev Datta, Cheng Perng Phoo, Utkarsh Mall, Carl Vondrick, Kavita Bala, Bharath Hariharan  

**Link**: [PDF](https://arxiv.org/pdf/2504.12110)  

**Abstract**: Earth Observation (EO) provides critical planetary data for environmental monitoring, disaster management, climate science, and other scientific domains. Here we ask: Are AI systems ready for reliable Earth Observation? We introduce \datasetnamenospace, a benchmark of 140 yes/no questions from NASA Earth Observatory articles across 13 topics and 17 satellite sensors. Using Google Earth Engine API as a tool, LLM agents can only achieve an accuracy of 33% because the code fails to run over 58% of the time. We improve the failure rate for open models by fine-tuning synthetic data, allowing much smaller models (Llama-3.1-8B) to achieve comparable accuracy to much larger ones (e.g., DeepSeek-R1). Taken together, our findings identify significant challenges to be solved before AI agents can automate earth observation, and suggest paths forward. The project page is available at this https URL. 

**Abstract (ZH)**: AI系统准备好应对可靠的地球观测了吗？ 

---
# Reasoning-Based AI for Startup Evaluation (R.A.I.S.E.): A Memory-Augmented, Multi-Step Decision Framework 

**Title (ZH)**: 基于推理的初创企业评估人工智能（R.A.I.S.E.）：一种记忆增强的多步决策框架 

**Authors**: Jack Preuveneers, Joseph Ternasky, Fuat Alican, Yigit Ihlamur  

**Link**: [PDF](https://arxiv.org/pdf/2504.12090)  

**Abstract**: We present a novel framework that bridges the gap between the interpretability of decision trees and the advanced reasoning capabilities of large language models (LLMs) to predict startup success. Our approach leverages chain-of-thought prompting to generate detailed reasoning logs, which are subsequently distilled into structured, human-understandable logical rules. The pipeline integrates multiple enhancements - efficient data ingestion, a two-step refinement process, ensemble candidate sampling, simulated reinforcement learning scoring, and persistent memory - to ensure both stable decision-making and transparent output. Experimental evaluations on curated startup datasets demonstrate that our combined pipeline improves precision by 54% from 0.225 to 0.346 and accuracy by 50% from 0.46 to 0.70 compared to a standalone OpenAI o3 model. Notably, our model achieves over 2x the precision of a random classifier (16%). By combining state-of-the-art AI reasoning with explicit rule-based explanations, our method not only augments traditional decision-making processes but also facilitates expert intervention and continuous policy refinement. This work lays the foundation for the implementation of interpretable LLM-powered decision frameworks in high-stakes investment environments and other domains that require transparent and data-driven insights. 

**Abstract (ZH)**: 我们提出了一种新型框架，该框架弥合了决策树可解释性与大型语言模型（LLMs）高级推理能力之间的差距，用于预测初创公司成功。我们的方法利用链式思考提示生成详细的推理日志，随后将其提炼为结构化的人类可理解的逻辑规则。该流水线集成了多项增强——高效数据摄入、两步精炼过程、候选集采样、模拟强化学习评分和持久化内存，以确保决策的稳定性和输出的透明性。对精心挑选的初创公司数据集的实验评估表明，与独立的OpenAI o3模型相比，我们的综合流水线在精确度上提高了54%，从0.225提高到0.346，在准确度上提高了50%，从0.46提高到0.70。值得注意的是，我们的模型的精确度是随机分类器的2倍多（16%）。通过结合最先进的AI推理与明确的规则解释，我们的方法不仅增强了传统的决策过程，还促进了专家介入和连续政策改进。本研究为在高风险投资环境中实施具备解释性的LLM驱动决策框架及其他需要透明和数据驱动洞察的领域奠定了基础。 

---
# Purposefully Induced Psychosis (PIP): Embracing Hallucination as Imagination in Large Language Models 

**Title (ZH)**: 故意诱导的精神错乱（PIP）：在大型语言模型中拥抱幻觉为想象 

**Authors**: Kris Pilcher, Esen K. Tütüncü  

**Link**: [PDF](https://arxiv.org/pdf/2504.12012)  

**Abstract**: Hallucinations in Large Language Models (LLMs) are widely regarded as errors - outputs that deviate from factual accuracy. However, in creative or exploratory contexts, these "mistakes" may represent unexpected avenues for innovation. We introduce Purposefully Induced Psychosis (PIP), a novel approach that amplifies LLM hallucinations for imaginative tasks such as speculative fiction, interactive storytelling, and mixed-reality simulations. Drawing on Herman Melville's Moby-Dick, where Pip's "madness" reveals profound insight, we reframe hallucinations as a source of computational imagination rather than a flaw. Our method fine-tunes LLMs to encourage speculative, metaphorical, and surreal outputs - hallucinations that are useful when factual accuracy is not the chief objective. Inspired by the consensual illusions of theater and stage magic, PIP situates these creative missteps in contexts where users willingly suspend disbelief, thereby transforming "errors" into catalysts for new ways of thinking. We discuss potential applications, design principles for ensuring user consent, preliminary observations, and implications for broader AI ethics and human-AI collaboration. 

**Abstract (ZH)**: Hallucinations in Large Language Models (LLMs) 作为广泛认为的事实偏差错误，在创造性和探索性背景下，这些“错误”可能代表了创新的未预见途径。我们引入了有意引发精神病态（PIP）的新方法，旨在放大LLM的幻觉用于诸如科幻创作、互动叙事和混合现实模拟等想象任务。借鉴赫尔曼·梅尔维尔的《白鲸记》，PIP将幻觉重新定义为计算性想象力的来源而非缺陷。该方法通过微调LLM以鼓励推测性、比喻性和超现实的输出，在事实准确性不是主要目标时，这些幻觉具有 usefulness。受到剧场和舞台魔术合意幻觉的启发，PIP将这些创造性的失误置于用户愿意暂时信以为真的情境中，从而将“错误”转化为思维新方式的催化剂。我们讨论了潜在应用、确保用户同意的设计原则、初步观察结果以及更广泛的人工智能伦理和人机协作的含义。 

---
# Rethinking the Generation of High-Quality CoT Data from the Perspective of LLM-Adaptive Question Difficulty Grading 

**Title (ZH)**: 从LLM自适应问题难度分级的角度重新思考高质量CoT数据的生成 

**Authors**: Qianjin Yu, Keyu Wu, Zihan Chen, Chushu Zhang, Manlin Mei, Lingjun Huang, Fang Tan, Yongsheng Du, Kunlin Liu, Yurui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11919)  

**Abstract**: Recently, DeepSeek-R1 (671B) (DeepSeek-AIet al., 2025) has demonstrated its excellent reasoning ability in complex tasks and has publiclyshared its methodology. This provides potentially high-quality chain-of-thought (CoT) data for stimulating the reasoning abilities of small-sized large language models (LLMs). To generate high-quality CoT data for different LLMs, we seek an efficient method for generating high-quality CoT data with LLM-Adaptive questiondifficulty levels. First, we grade the difficulty of the questions according to the reasoning ability of the LLMs themselves and construct a LLM-Adaptive question database. Second, we sample the problem database based on a distribution of difficulty levels of the questions and then use DeepSeek-R1 (671B) (DeepSeek-AI et al., 2025) to generate the corresponding high-quality CoT data with correct answers. Thanks to the construction of CoT data with LLM-Adaptive difficulty levels, we have significantly reduced the cost of data generation and enhanced the efficiency of model supervised fine-tuning (SFT). Finally, we have validated the effectiveness and generalizability of the proposed method in the fields of complex mathematical competitions and code generation tasks. Notably, with only 2k high-quality mathematical CoT data, our ZMath-32B surpasses DeepSeek-Distill-32B in math reasoning task. Similarly, with only 2k high-quality code CoT data, our ZCode-32B surpasses DeepSeek-Distill-32B in code reasoning tasks. 

**Abstract (ZH)**: 近期，DeepSeek-R1 (671B) (DeepSeek-AI et al., 2025) 在复杂任务中展示了其出色的理由推理能力，并公开分享了其方法论。这为激发小型大型语言模型（LLMs）的理由推理能力提供了潜在高质量的链式推理（CoT）数据。为了为不同LLMs生成高质量CoT数据，我们寻求一种生成具有LLM自适应问题难度级别的高质量CoT数据的有效方法。首先，我们根据LLMs自身的推理能力对其问题的难度进行分级，并构建一个LLM自适应问题数据库。其次，我们基于问题难度分布对问题数据库进行采样，然后使用DeepSeek-R1 (671B) (DeepSeek-AI et al., 2025) 生成相应的高质量CoT数据并给出正确答案。通过构建具有LLM自适应难度级别的CoT数据，我们显著降低了数据生成的成本，并提升了模型监督微调（SFT）的效率。最后，我们验证了所提出方法在复杂数学竞赛和代码生成任务中的有效性和通用性。值得注意的是，仅使用2k高质量数学CoT数据，我们的ZMath-32B在数学推理任务中超过了DeepSeek-Distill-32B；同样，仅使用2k高质量代码CoT数据，我们的ZCode-32B在代码推理任务中超过了DeepSeek-Distill-32B。 

---
# Evaluating the Goal-Directedness of Large Language Models 

**Title (ZH)**: 评估大型语言模型的目的导向性 

**Authors**: Tom Everitt, Cristina Garbacea, Alexis Bellot, Jonathan Richens, Henry Papadatos, Siméon Campos, Rohin Shah  

**Link**: [PDF](https://arxiv.org/pdf/2504.11844)  

**Abstract**: To what extent do LLMs use their capabilities towards their given goal? We take this as a measure of their goal-directedness. We evaluate goal-directedness on tasks that require information gathering, cognitive effort, and plan execution, where we use subtasks to infer each model's relevant capabilities. Our evaluations of LLMs from Google DeepMind, OpenAI, and Anthropic show that goal-directedness is relatively consistent across tasks, differs from task performance, and is only moderately sensitive to motivational prompts. Notably, most models are not fully goal-directed. We hope our goal-directedness evaluations will enable better monitoring of LLM progress, and enable more deliberate design choices of agentic properties in LLMs. 

**Abstract (ZH)**: LLMs在完成给定目标时使用其能力的程度：一种目标导向性的评估 

---
# Large Language Models for Drug Overdose Prediction from Longitudinal Medical Records 

**Title (ZH)**: 基于 longitudinal 医疗记录的大型语言模型在药物过量预测中的应用 

**Authors**: Md Sultan Al Nahian, Chris Delcher, Daniel Harris, Peter Akpunonu, Ramakanth Kavuluru  

**Link**: [PDF](https://arxiv.org/pdf/2504.11792)  

**Abstract**: The ability to predict drug overdose risk from a patient's medical records is crucial for timely intervention and prevention. Traditional machine learning models have shown promise in analyzing longitudinal medical records for this task. However, recent advancements in large language models (LLMs) offer an opportunity to enhance prediction performance by leveraging their ability to process long textual data and their inherent prior knowledge across diverse tasks. In this study, we assess the effectiveness of Open AI's GPT-4o LLM in predicting drug overdose events using patients' longitudinal insurance claims records. We evaluate its performance in both fine-tuned and zero-shot settings, comparing them to strong traditional machine learning methods as baselines. Our results show that LLMs not only outperform traditional models in certain settings but can also predict overdose risk in a zero-shot setting without task-specific training. These findings highlight the potential of LLMs in clinical decision support, particularly for drug overdose risk prediction. 

**Abstract (ZH)**: 基于患者医疗记录预测药物过量风险的能力对于及时干预和预防至关重要。传统的机器学习模型在通过分析纵向医疗记录进行此任务时展现出了潜力。然而，近期大型语言模型（LLMs）的发展提供了通过利用其处理长文本数据的能力和跨任务的潜在先验知识来提升预测性能的机会。在本研究中，我们评估了Open AI的GPT-4o LLM在使用患者纵向保险索赔记录预测药物过量事件方面的有效性。我们评估了其在微调和零样本设置中的性能，并将其与强的传统机器学习方法进行了基线比较。研究结果表明，LLMs不仅在某些设置中优于传统模型，还能够在无需特定任务训练的情况下预测过量风险。这些发现强调了LLMs在临床决策支持，特别是药物过量风险预测方面的潜在价值。 

---
# Shared Disk KV Cache Management for Efficient Multi-Instance Inference in RAG-Powered LLMs 

**Title (ZH)**: 基于共享磁盘的KV缓存管理以提高RAG增强的大语言模型多实例推理效率 

**Authors**: Hyungwoo Lee, Kihyun Kim, Jinwoo Kim, Jungmin So, Myung-Hoon Cha, Hong-Yeon Kim, James J. Kim, Youngjae Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.11765)  

**Abstract**: Recent large language models (LLMs) face increasing inference latency as input context length and model size continue to grow. In particular, the retrieval-augmented generation (RAG) technique, which enhances LLM responses by incorporating external knowledge, exacerbates this issue by significantly increasing the number of input tokens. This expansion in token length leads to a substantial rise in computational overhead, particularly during the prefill stage, resulting in prolonged time-to-first-token (TTFT). To address this issue, this paper proposes a method to reduce TTFT by leveraging a disk-based key-value (KV) cache to lessen the computational burden during the prefill stage. We also introduce a disk-based shared KV cache management system, called Shared RAG-DCache, for multi-instance LLM RAG service environments. This system, together with an optimal system configuration, improves both throughput and latency under given resource constraints. Shared RAG-DCache exploits the locality of documents related to user queries in RAG, as well as the queueing delay in LLM inference services. It proactively generates and stores disk KV caches for query-related documents and shares them across multiple LLM instances to enhance inference performance. In experiments on a single host equipped with 2 GPUs and 1 CPU, Shared RAG-DCache achieved a 15~71% increase in throughput and up to a 12~65% reduction in latency, depending on the resource configuration. 

**Abstract (ZH)**: Recent大型语言模型（LLMs）在输入上下文长度和模型规模持续增长的情况下面临推理延迟增加的问题。特别是检索增强生成（RAG）技术通过引入外部知识来增强LLM的响应，这极大地增加了输入令牌的数量，从而导致在预填充阶段的计算开销显著增加，特别是在预填充阶段导致首次令牌时间（TTFT）延长。为了解决这一问题，本文提出了一种方法，通过利用基于磁盘的键值（KV）缓存来减轻预填充阶段的计算负担。我们还为多实例LLM RAG服务环境引入了一种基于磁盘的共享键值缓存管理系统，称为Shared RAG-DCache。该系统结合最优系统配置，能够在给定的资源约束条件下提高吞吐量和降低延迟。Shared RAG-DCache利用与用户查询相关的文档的局部性以及LLM推理服务中的排队延迟，主动为查询相关的文档生成并存储磁盘KV缓存，并且跨多个LLM实例共享，以增强推理性能。在配备2块GPU和1块CPU的单机实验中，Shared RAG-DCache在不同的资源配置下，吞吐量提高了15~71%，并且延迟减少了12~65%。 

---
# Climbing the Ladder of Reasoning: What LLMs Can-and Still Can't-Solve after SFT? 

**Title (ZH)**: 攀登推理阶梯：SFT之后LLM能解决和仍不能解决的问题 

**Authors**: Yiyou Sun, Georgia Zhou, Hao Wang, Dacheng Li, Nouha Dziri, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.11741)  

**Abstract**: Recent supervised fine-tuning (SFT) approaches have significantly improved language models' performance on mathematical reasoning tasks, even when models are trained at a small scale. However, the specific capabilities enhanced through such fine-tuning remain poorly understood. In this paper, we conduct a detailed analysis of model performance on the AIME24 dataset to understand how reasoning capabilities evolve. We discover a ladder-like structure in problem difficulty, categorize questions into four tiers (Easy, Medium, Hard, and Extremely Hard (Exh)), and identify the specific requirements for advancing between tiers. We find that progression from Easy to Medium tier requires adopting an R1 reasoning style with minimal SFT (500-1K instances), while Hard-level questions suffer from frequent model's errors at each step of the reasoning chain, with accuracy plateauing at around 65% despite logarithmic scaling. Exh-level questions present a fundamentally different challenge; they require unconventional problem-solving skills that current models uniformly struggle with. Additional findings reveal that carefully curated small-scale datasets offer limited advantage-scaling dataset size proves far more effective. Our analysis provides a clearer roadmap for advancing language model capabilities in mathematical reasoning. 

**Abstract (ZH)**: Recent supervised fine-tuning (SFT) 方法显著提升了小型训练规模的语言模型在数学推理任务上的性能，但这些方法具体增强的能力仍然了解不足。本文通过对AIME24数据集的详细分析，研究推理能力如何演变。我们发现了一种梯级结构的问题难度模式，将问题分为四个层级（简单、中等、困难和极困难），并确定了不同层级之间的具体要求。我们发现，从简单层级到中等层级的进步需要采用少量的SFT（500-1K实例）的R1推理风格，而困难层级的问题在推理链的每一步都频繁出错，即使在对数扩展的情况下准确率也仅在65%左右停滞不前。极困难层级的问题提出了完全不同类型的挑战；它们需要当前模型难以掌握的非传统问题解决技能。我们还发现，精心策划的小规模数据集提供的优势有限，扩展数据集规模的效果要显著得多。我们的分析为提升语言模型在数学推理方面的能力提供了更清晰的路径。 

---
# A Library of LLM Intrinsics for Retrieval-Augmented Generation 

**Title (ZH)**: LLM内在机制库用于检索增强生成 

**Authors**: Marina Danilevsky, Kristjan Greenewald, Chulaka Gunasekara, Maeda Hanafi, Lihong He, Yannis Katsis, Krishnateja Killamsetty, Yatin Nandwani, Lucian Popa, Dinesh Raghu, Frederick Reiss, Vraj Shah, Khoi-Nguyen Tran, Huaiyu Zhu, Luis Lastras  

**Link**: [PDF](https://arxiv.org/pdf/2504.11704)  

**Abstract**: In the developer community for large language models (LLMs), there is not yet a clean pattern analogous to a software library, to support very large scale collaboration. Even for the commonplace use case of Retrieval-Augmented Generation (RAG), it is not currently possible to write a RAG application against a well-defined set of APIs that are agreed upon by different LLM providers. Inspired by the idea of compiler intrinsics, we propose some elements of such a concept through introducing a library of LLM Intrinsics for RAG. An LLM intrinsic is defined as a capability that can be invoked through a well-defined API that is reasonably stable and independent of how the LLM intrinsic itself is implemented. The intrinsics in our library are released as LoRA adapters on HuggingFace, and through a software interface with clear structured input/output characteristics on top of vLLM as an inference platform, accompanied in both places with documentation and code. This article describes the intended usage, training details, and evaluations for each intrinsic, as well as compositions of multiple intrinsics. 

**Abstract (ZH)**: 在大规模语言模型开发者社区中，尚未出现一个类似于软件库的清晰模式，以支持大规模协作。即使对于检索增强生成（RAG）这一常见的应用场景，目前也无法编写一个基于不同大规模语言模型提供者达成共识的API集的RAG应用程序。受编译器本征函数的启发，我们通过引入大规模语言模型本征库来提出这一概念的一部分。大规模语言模型本征被定义为一种可以通过稳定且独立于其实现方式的API调用的能力。我们的库中的本征以LoRA适配器的形式在HuggingFace上发布，并在vLLM推理平台上通过一个具有明确结构化输入/输出特性的软件接口提供，两个地方都附有文档和代码。本文描述了每个本征的预期用法、训练细节和评估，以及多个本征的组合。 

---
# Steering Prosocial AI Agents: Computational Basis of LLM's Decision Making in Social Simulation 

**Title (ZH)**: 引导亲社会AI代理：大型语言模型在社会模拟中决策的计算基础 

**Authors**: Ji Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.11671)  

**Abstract**: Large language models (LLMs) increasingly serve as human-like decision-making agents in social science and applied settings. These LLM-agents are typically assigned human-like characters and placed in real-life contexts. However, how these characters and contexts shape an LLM's behavior remains underexplored. This study proposes and tests methods for probing, quantifying, and modifying an LLM's internal representations in a Dictator Game -- a classic behavioral experiment on fairness and prosocial behavior. We extract ``vectors of variable variations'' (e.g., ``male'' to ``female'') from the LLM's internal state. Manipulating these vectors during the model's inference can substantially alter how those variables relate to the model's decision-making. This approach offers a principled way to study and regulate how social concepts can be encoded and engineered within transformer-based models, with implications for alignment, debiasing, and designing AI agents for social simulations in both academic and commercial applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）在社会科学研究和实际应用中 increasingly 作为类人决策代理使用。这些LLM代理通常被赋予类人的角色，并置于现实情境中。然而，这些角色和情境如何影响LLM的行为仍缺乏探索。本研究提出并测试了在博弈论中的“分配者游戏”中探测、量化和修改LLM内部表示的方法，以探究公平性和利他行为。我们从LLM的内部状态中提取“变量变化向量”（例如，“男性”变“女性”）。在模型推理过程中操纵这些向量可以显著改变这些变量与模型决策之间的关系。这种方法提供了一种原则性的方法，用于研究和调控社会概念如何在变换器模型中编码和设计，对对齐、去偏见及设计用于社会模拟的AI代理在学术和商业应用中的意义具有影响。 

---
# GraphicBench: A Planning Benchmark for Graphic Design with Language Agents 

**Title (ZH)**: GraphicBench：一种基于语言代理的图形设计规划基准 

**Authors**: Dayeon Ki, Tianyi Zhou, Marine Carpuat, Gang Wu, Puneet Mathur, Viswanathan Swaminathan  

**Link**: [PDF](https://arxiv.org/pdf/2504.11571)  

**Abstract**: Large Language Model (LLM)-powered agents have unlocked new possibilities for automating human tasks. While prior work has focused on well-defined tasks with specified goals, the capabilities of agents in creative design tasks with open-ended goals remain underexplored. We introduce GraphicBench, a new planning benchmark for graphic design that covers 1,079 user queries and input images across four design types. We further present GraphicTown, an LLM agent framework with three design experts and 46 actions (tools) to choose from for executing each step of the planned workflows in web environments. Experiments with six LLMs demonstrate their ability to generate workflows that integrate both explicit design constraints from user queries and implicit commonsense constraints. However, these workflows often do not lead to successful execution outcomes, primarily due to challenges in: (1) reasoning about spatial relationships, (2) coordinating global dependencies across experts, and (3) retrieving the most appropriate action per step. We envision GraphicBench as a challenging yet valuable testbed for advancing LLM-agent planning and execution in creative design tasks. 

**Abstract (ZH)**: Large Language Model (LLM)-赋能的代理解锁了自动化人类任务的新可能性。虽然先前的工作集中在具有明确目标的定义任务上，但具有开放目标的创意设计任务中代理的能力仍待探索。我们引入了GraphicBench，这是一个全新的图形设计规划基准，涵盖了四种设计类型中的1,079个用户查询和输入图像。我们进一步提出了GraphicTown，这是一个包含三位设计专家和46种执行每一步计划工作流的工具的LLM代理框架，在网络环境中使用这些工具。六个LLM的实验展示了它们生成的工作流能够整合来自用户查询的显式设计约束和来自常识的隐式约束的能力。然而，这些工作流往往不能实现成功的结果，主要由于在：（1）空间关系推理，（2）跨专家全局依赖协调，以及（3）每一步检索最合适的动作方面存在的挑战。我们设想GraphicBench是一个具有挑战性但有价值的测试平台，用于推进创意设计任务中的LLM-代理规划和执行。 

---
# NodeRAG: Structuring Graph-based RAG with Heterogeneous Nodes 

**Title (ZH)**: NodeRAG: 基于异构节点的图结构RAG 

**Authors**: Tianyang Xu, Haojie Zheng, Chengze Li, Haoxiang Chen, Yixin Liu, Ruoxi Chen, Lichao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.11544)  

**Abstract**: Retrieval-augmented generation (RAG) empowers large language models to access external and private corpus, enabling factually consistent responses in specific domains. By exploiting the inherent structure of the corpus, graph-based RAG methods further enrich this process by building a knowledge graph index and leveraging the structural nature of graphs. However, current graph-based RAG approaches seldom prioritize the design of graph structures. Inadequately designed graph not only impede the seamless integration of diverse graph algorithms but also result in workflow inconsistencies and degraded performance. To further unleash the potential of graph for RAG, we propose NodeRAG, a graph-centric framework introducing heterogeneous graph structures that enable the seamless and holistic integration of graph-based methodologies into the RAG workflow. By aligning closely with the capabilities of LLMs, this framework ensures a fully cohesive and efficient end-to-end process. Through extensive experiments, we demonstrate that NodeRAG exhibits performance advantages over previous methods, including GraphRAG and LightRAG, not only in indexing time, query time, and storage efficiency but also in delivering superior question-answering performance on multi-hop benchmarks and open-ended head-to-head evaluations with minimal retrieval tokens. Our GitHub repository could be seen at this https URL. 

**Abstract (ZH)**: 基于图的检索增强生成（RAG）方法通过构建知识图谱索引并利用图的结构特性，进一步丰富了这一过程，但当前的基于图的RAG方法很少重视图结构的设计。设计不当的图不仅阻碍了多种图算法的无缝集成，还导致工作流程不一致且性能下降。为了进一步发挥图在RAG中的潜力，我们提出了一种节点中心框架NodeRAG，引入异质图结构，使基于图的方法能够无缝地整合到RAG工作流中。通过与LLMs的能力紧密对齐，该框架确保实现了一个完全一致且高效的端到端过程。通过大量实验，我们证明NodeRAG在索引时间、查询时间和存储效率方面优于GraphRAG和LightRAG，并在多跳基准测试和开放式头对头评估中提供了更好的问答性能，同时最小化检索令牌。我们的GitHub仓库地址为这个 https URL。 

---
# HypoBench: Towards Systematic and Principled Benchmarking for Hypothesis Generation 

**Title (ZH)**: HypoBench: 向系统性和原则性的假设生成基准测试迈进 

**Authors**: Haokun Liu, Sicong Huang, Jingyu Hu, Yangqiaoyu Zhou, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.11524)  

**Abstract**: There is growing interest in hypothesis generation with large language models (LLMs). However, fundamental questions remain: what makes a good hypothesis, and how can we systematically evaluate methods for hypothesis generation? To address this, we introduce HypoBench, a novel benchmark designed to evaluate LLMs and hypothesis generation methods across multiple aspects, including practical utility, generalizability, and hypothesis discovery rate. HypoBench includes 7 real-world tasks and 5 synthetic tasks with 194 distinct datasets. We evaluate four state-of-the-art LLMs combined with six existing hypothesis-generation methods. Overall, our results suggest that existing methods are capable of discovering valid and novel patterns in the data. However, the results from synthetic datasets indicate that there is still significant room for improvement, as current hypothesis generation methods do not fully uncover all relevant or meaningful patterns. Specifically, in synthetic settings, as task difficulty increases, performance significantly drops, with best models and methods only recovering 38.8% of the ground-truth hypotheses. These findings highlight challenges in hypothesis generation and demonstrate that HypoBench serves as a valuable resource for improving AI systems designed to assist scientific discovery. 

**Abstract (ZH)**: 基于大规模语言模型的假设生成：HypoBench基准设计与初步探索 

---
# HLS-Eval: A Benchmark and Framework for Evaluating LLMs on High-Level Synthesis Design Tasks 

**Title (ZH)**: HLS-Eval：评价高級綜合設計任務中大型语言模型的基准和框架 

**Authors**: Stefan Abi-Karam, Cong Hao  

**Link**: [PDF](https://arxiv.org/pdf/2504.12268)  

**Abstract**: The rapid scaling of large language model (LLM) training and inference has driven their adoption in semiconductor design across academia and industry. While most prior work evaluates LLMs on hardware description language (HDL) tasks, particularly Verilog, designers are increasingly using high-level synthesis (HLS) to build domain-specific accelerators and complex hardware systems. However, benchmarks and tooling to comprehensively evaluate LLMs for HLS design tasks remain scarce.
To address this, we introduce HLS-Eval, the first complete benchmark and evaluation framework for LLM-driven HLS design. HLS-Eval targets two core tasks: (1) generating HLS code from natural language descriptions, and (2) performing HLS-specific code edits to optimize performance and hardware efficiency. The benchmark includes 94 unique designs drawn from standard HLS benchmarks and novel sources. Each case is prepared via a semi-automated flow that produces a natural language description and a paired testbench for C-simulation and synthesis validation, ensuring each task is "LLM-ready."
Beyond the benchmark, HLS-Eval offers a modular Python framework for automated, parallel evaluation of both local and hosted LLMs. It includes a parallel evaluation engine, direct HLS tool integration, and abstractions for to support different LLM interaction paradigms, enabling rapid prototyping of new benchmarks, tasks, and LLM methods.
We demonstrate HLS-Eval through baseline evaluations of open-source LLMs on Vitis HLS, measuring outputs across four key metrics - parseability, compilability, runnability, and synthesizability - reflecting the iterative HLS design cycle. We also report pass@k metrics, establishing clear baselines and reusable infrastructure for the broader LLM-for-hardware community.
All benchmarks, framework code, and results are open-sourced at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLM）训练和推理的快速扩展推动了其在半导体设计领域的学术和工业应用。虽然早期大多数研究工作主要是在硬件描述语言（HDL）任务上评估LLM，特别是Verilog，设计师越来越多地使用高层次综合（HLS）构建专用加速器和复杂硬件系统。然而，用于HLS设计任务全面评估LLM的基准和工具仍然稀缺。
为了解决这个问题，我们引入了HLS-Eval，这是首个用于LLM驱动HLS设计的完整基准和评估框架。HLS-Eval针对两个核心任务：（1）从自然语言描述生成HLS代码，以及（2）执行HLS特定的代码编辑以优化性能和硬件效率。基准包括94个独特的设计，来源包括标准HLS基准和新颖源。每个案例通过半自动化流程准备，生成自然语言描述和配套的测试平台，用于C仿真和综合验证，确保每个任务都是“LLM就绪”的。
除了基准之外，HLS-Eval还提供了一个模块化的Python框架，用于自动并行评估本地和托管的LLM。该框架包括并行评估引擎、直接HLS工具集成以及支持不同LLM交互模式的抽象，使快速原型设计新基准、任务和LLM方法成为可能。
我们通过在Vitis HLS上对开源LLM进行基准评估，展示了HLS-Eval，衡量了四个关键指标（可解析性、可编译性、可运行性和可综合性），反映了迭代的HLS设计周期。我们还将报告pass@k指标，为更广泛的硬件LLM社区建立清晰的基础和可重用的基础设施。
所有基准、框架代码和结果均在此网址开放源代码：this https URL。 

---
# FLIP Reasoning Challenge 

**Title (ZH)**: 反转推理挑战 

**Authors**: Andreas Plesner, Turlan Kuzhagaliyev, Roger Wattenhofer  

**Link**: [PDF](https://arxiv.org/pdf/2504.12256)  

**Abstract**: Over the past years, advances in artificial intelligence (AI) have demonstrated how AI can solve many perception and generation tasks, such as image classification and text writing, yet reasoning remains a challenge. This paper introduces the FLIP dataset, a benchmark for evaluating AI reasoning capabilities based on human verification tasks on the Idena blockchain. FLIP challenges present users with two orderings of 4 images, requiring them to identify the logically coherent one. By emphasizing sequential reasoning, visual storytelling, and common sense, FLIP provides a unique testbed for multimodal AI systems. Our experiments evaluate state-of-the-art models, leveraging both vision-language models (VLMs) and large language models (LLMs). Results reveal that even the best open-sourced and closed-sourced models achieve maximum accuracies of 75.5% and 77.9%, respectively, in zero-shot settings, compared to human performance of 95.3%. Captioning models aid reasoning models by providing text descriptions of images, yielding better results than when using the raw images directly, 69.6% vs. 75.2% for Gemini 1.5 Pro. Combining the predictions from 15 models in an ensemble increases the accuracy to 85.2%. These findings highlight the limitations of existing reasoning models and the need for robust multimodal benchmarks like FLIP. The full codebase and dataset will be available at this https URL. 

**Abstract (ZH)**: 在过去几年中，人工智能（AI）的进步展示了AI在解决图像分类和文本生成等感知和生成任务方面的能力，但推理仍然是一个挑战。本文介绍了FLIP数据集，这是一个基于Idena区块链上人工验证任务的AI推理能力评估基准。FLIP挑战要求用户提供四个图像的两种排序，并要求他们识别出逻辑上更为连贯的一种。通过强调序列推理、视觉叙事和常识，FLIP为多模态AI系统提供了一个独特的测试床。我们的实验利用了最新的视觉-语言模型（VLMs）和大型语言模型（LLMs）评估了最先进的模型。结果表明，即使是最先进的开源和闭源模型，在零样本设置下的最大准确率分别为75.5%和77.9%，而人类的性能为95.3%。描述模型通过提供图像的文字描述来辅助推理模型，其表现优于直接使用原始图像的情况，比如Gemini 1.5 Pro的69.6% vs. 75.2%。通过集成15个模型的预测结果，准确率提高到85.2%。这些发现凸显了现有推理模型的局限性，并强调了像FLIP这样稳健的多模态基准测试的需求。完整代码库和数据集可在此处访问。 

---
# From Requirements to Architecture: Semi-Automatically Generating Software Architectures 

**Title (ZH)**: 从需求到架构：半自动生成软件架构 

**Authors**: Tobias Eisenreich  

**Link**: [PDF](https://arxiv.org/pdf/2504.12192)  

**Abstract**: To support junior and senior architects, I propose developing a new architecture creation method that leverages LLMs' evolving capabilities to support the architect. This method involves the architect's close collaboration with LLM-fueled tooling over the whole process. The architect is guided through Domain Model creation, Use Case specification, architectural decisions, and architecture evaluation. While the architect can take complete control of the process and the results, and use the tooling as a building set, they can follow the intended process for maximum tooling support. The preliminary results suggest the feasibility of this process and indicate major time savings for the architect. 

**Abstract (ZH)**: 为了支持初级和资深架构师，我提议开发一种新的架构创建方法，利用LLM不断演进的能力来辅助架构师。该方法包括架构师与以LLM为动力的工具在整个过程中进行密切合作。该方法指导架构师进行领域模型创建、用例规范、架构决策和架构评估。虽然架构师可以完全控制整个过程及其结果，并将工具视为构建块，但他们可以遵循预期的过程以获得最大的工具支持。初步结果表明该过程的可行性，并指出这将为架构师节省大量时间。 

---
# What Do Large Language Models Know? Tacit Knowledge as a Potential Causal-Explanatory Structure 

**Title (ZH)**: 大型语言模型知多少？默会知识作为一种潜在的因果解释结构 

**Authors**: Céline Budding  

**Link**: [PDF](https://arxiv.org/pdf/2504.12187)  

**Abstract**: It is sometimes assumed that Large Language Models (LLMs) know language, or for example that they know that Paris is the capital of France. But what -- if anything -- do LLMs actually know? In this paper, I argue that LLMs can acquire tacit knowledge as defined by Martin Davies (1990). Whereas Davies himself denies that neural networks can acquire tacit knowledge, I demonstrate that certain architectural features of LLMs satisfy the constraints of semantic description, syntactic structure, and causal systematicity. Thus, tacit knowledge may serve as a conceptual framework for describing, explaining, and intervening on LLMs and their behavior. 

**Abstract (ZH)**: 大型语言模型是否具备默会知识：基于马丁·戴维斯（1990）的定义 

---
# SALAD: Improving Robustness and Generalization through Contrastive Learning with Structure-Aware and LLM-Driven Augmented Data 

**Title (ZH)**: SALAD: 通过结构感知和大语言模型驱动的增强数据对比学习提高鲁棒性和泛化能力 

**Authors**: Suyoung Bae, Hyojun Kim, YunSeok Choi, Jee-Hyong Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.12185)  

**Abstract**: In various natural language processing (NLP) tasks, fine-tuning Pre-trained Language Models (PLMs) often leads to the issue of spurious correlations, which negatively impacts performance, particularly when dealing with out-of-distribution data. To address this problem, we propose SALAD}(Structure Aware and LLM-driven Augmented Data), a novel approach designed to enhance model robustness and generalization by generating structure-aware and counterfactually augmented data for contrastive learning. Our method leverages a tagging-based approach to generate structure-aware positive samples and utilizes large language models (LLMs) to generate counterfactual negative samples with diverse sentence patterns. By applying contrastive learning, SALAD enables the model to focus on learning the structural relationships between key sentence components while minimizing reliance on spurious correlations. We validate our approach through experiments on three tasks: Sentiment Classification, Sexism Detection, and Natural Language Inference. The results demonstrate that SALAD not only improves model robustness and performance across different environments but also enhances generalization to out-of-distribution datasets and cross-domain scenarios. 

**Abstract (ZH)**: 在各种自然语言处理（NLP）任务中，预训练语言模型（PLMs）的微调往往会导致虚假相关性问题，这对性能尤其是处理分布外数据时产生了负面影响。为了解决这个问题，我们提出了一种名为SALAD（Structure Aware and LLM-driven Augmented Data）的新方法，旨在通过生成结构感知和反事实增强的数据来增强模型的稳健性和泛化能力。该方法利用标记化方法生成结构感知的正样本，并利用大型语言模型生成具有多种句子模式的反事实负样本。通过应用对比学习，SALAD使模型能够专注于学习关键句子组件之间的结构关系，同时减少了对虚假相关性的依赖。我们通过情感分类、性别歧视检测和自然语言推理三项任务的实验验证了该方法。结果表明，SALAD不仅在不同环境提升了模型的稳健性和性能，还增强了对分布外数据集和跨域场景的泛化能力。 

---
# Trusting CHATGPT: how minor tweaks in the prompts lead to major differences in sentiment classification 

**Title (ZH)**: 信任CHATGPT：提示小微调如何导致情感分类重大差异 

**Authors**: Jaime E. Cuellar, Oscar Moreno-Martinez, Paula Sofia Torres-Rodriguez, Jaime Andres Pavlich-Mariscal, Andres Felipe Mican-Castiblanco, Juan Guillermo Torres-Hurtado  

**Link**: [PDF](https://arxiv.org/pdf/2504.12180)  

**Abstract**: One fundamental question for the social sciences today is: how much can we trust highly complex predictive models like ChatGPT? This study tests the hypothesis that subtle changes in the structure of prompts do not produce significant variations in the classification results of sentiment polarity analysis generated by the Large Language Model GPT-4o mini. Using a dataset of 100.000 comments in Spanish on four Latin American presidents, the model classified the comments as positive, negative, or neutral on 10 occasions, varying the prompts slightly each time. The experimental methodology included exploratory and confirmatory analyses to identify significant discrepancies among classifications.
The results reveal that even minor modifications to prompts such as lexical, syntactic, or modal changes, or even their lack of structure impact the classifications. In certain cases, the model produced inconsistent responses, such as mixing categories, providing unsolicited explanations, or using languages other than Spanish. Statistical analysis using Chi-square tests confirmed significant differences in most comparisons between prompts, except in one case where linguistic structures were highly similar.
These findings challenge the robustness and trust of Large Language Models for classification tasks, highlighting their vulnerability to variations in instructions. Moreover, it was evident that the lack of structured grammar in prompts increases the frequency of hallucinations. The discussion underscores that trust in Large Language Models is based not only on technical performance but also on the social and institutional relationships underpinning their use. 

**Abstract (ZH)**: 今天的社会科学的一个基本问题是：我们能相信像ChatGPT这样的高度复杂预测模型多少？本研究检验了这样的假设：提示结构的微妙变化不会对大型语言模型GPT-4o mini生成的情感极性分类结果产生显著差异。使用西班牙语对四位拉丁美洲总统的100,000条评论数据集，模型在10次实验中每次微调提示，将其分类为正面、负面或中性。实验方法包括探索性和验证性分析，以识别分类结果中的显著差异。

研究结果表明，即使是词法、句法或情态变化，甚至是缺乏结构，细微的提示修改也会影响分类结果。在某些情况下，模型产生了不一致的响应，如混合类别、提供不必要的解释或使用不同于西班牙语的语言。卡方检验的统计分析证实了大多数提示之间的显著差异，但在一种情况下，语言结构高度相似。

这些发现质疑了大型语言模型在分类任务中的稳健性和可信度，突显了它们在指令变化方面的脆弱性。此外，很明显，提示中缺乏结构化的语法增加了幻觉的发生频率。讨论强调，对大型语言模型的信任不仅基于技术水平，还基于其使用背后的社会和制度关系。 

---
# Efficient Contrastive Decoding with Probabilistic Hallucination Detection - Mitigating Hallucinations in Large Vision Language Models - 

**Title (ZH)**: 高效的对比解码与概率幻觉检测——缓解大规模视觉语言模型中的幻觉现象 

**Authors**: Laura Fieback, Nishilkumar Balar, Jakob Spiegelberg, Hanno Gottschalk  

**Link**: [PDF](https://arxiv.org/pdf/2504.12137)  

**Abstract**: Despite recent advances in Large Vision Language Models (LVLMs), these models still suffer from generating hallucinatory responses that do not align with the visual input provided. To mitigate such hallucinations, we introduce Efficient Contrastive Decoding (ECD), a simple method that leverages probabilistic hallucination detection to shift the output distribution towards contextually accurate answers at inference time. By contrasting token probabilities and hallucination scores, ECD subtracts hallucinated concepts from the original distribution, effectively suppressing hallucinations. Notably, our proposed method can be applied to any open-source LVLM and does not require additional LVLM training. We evaluate our method on several benchmark datasets and across different LVLMs. Our experiments show that ECD effectively mitigates hallucinations, outperforming state-of-the-art methods with respect to performance on LVLM benchmarks and computation time. 

**Abstract (ZH)**: 尽管近年来大型视觉语言模型取得了进展，但这些模型仍会产生与提供的视觉输入不一致的幻觉响应。为了减轻这种幻觉，我们引入了高效对比解码（ECD）方法，该方法通过利用概率幻觉检测，在推理时将输出分布偏向于上下文准确的答案。通过对比令牌概率和幻觉分数，ECD从原始分布中减去幻觉概念，有效抑制了幻觉。值得注意的是，我们提出的方法可以应用于任何开源的大型视觉语言模型，不需要额外的大型视觉语言模型训练。我们在多个基准数据集和不同的大型视觉语言模型上评估了该方法。实验结果表明，ECD有效地减轻了幻觉，相比现有的顶级方法，在大型视觉语言模型基准测试性能和计算时间上均表现出优势。 

---
# Optimizing Compound Retrieval Systems 

**Title (ZH)**: 优化复合检索系统 

**Authors**: Harrie Oosterhuis, Rolf Jagerman, Zhen Qin, Xuanhui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.12063)  

**Abstract**: Modern retrieval systems do not rely on a single ranking model to construct their rankings. Instead, they generally take a cascading approach where a sequence of ranking models are applied in multiple re-ranking stages. Thereby, they balance the quality of the top-K ranking with computational costs by limiting the number of documents each model re-ranks. However, the cascading approach is not the only way models can interact to form a retrieval system.
We propose the concept of compound retrieval systems as a broader class of retrieval systems that apply multiple prediction models. This encapsulates cascading models but also allows other types of interactions than top-K re-ranking. In particular, we enable interactions with large language models (LLMs) which can provide relative relevance comparisons. We focus on the optimization of compound retrieval system design which uniquely involves learning where to apply the component models and how to aggregate their predictions into a final ranking. This work shows how our compound approach can combine the classic BM25 retrieval model with state-of-the-art (pairwise) LLM relevance predictions, while optimizing a given ranking metric and efficiency target. Our experimental results show optimized compound retrieval systems provide better trade-offs between effectiveness and efficiency than cascading approaches, even when applied in a self-supervised manner.
With the introduction of compound retrieval systems, we hope to inspire the information retrieval field to more out-of-the-box thinking on how prediction models can interact to form rankings. 

**Abstract (ZH)**: 现代检索系统不依赖单一排名模型构建其排名，而是通常采取级联方法，在多个重排名阶段应用一系列排名模型，从而平衡前K个文档的排名质量与计算成本。然而，级联方法并非模型间交互的唯一方式。我们提出了复合检索系统的概念，作为一种更广泛的检索系统类别，它应用多种预测模型。这一概念不仅包括级联模型，还允许其他类型的交互，尤其是与大型语言模型（LLMs）的交互，后者可以提供相关性比较。我们专注于复合检索系统设计的优化，这独特地涉及学习如何应用组件模型以及如何将它们的预测整合为最终排名。本研究展示了我们复合方法如何将经典的BM25检索模型与最先进的（成对）LLM相关性预测结合，并优化给定的排名度量和效率目标。实验结果表明，优化的复合检索系统在效率和效果之间提供了比级联方法更好的权衡，即使是自监督应用也是如此。通过引入复合检索系统，我们希望激发信息检索领域对预测模型如何形成排名的更具创新性的思考。 

---
# Generative Recommendation with Continuous-Token Diffusion 

**Title (ZH)**: 连续令牌扩散 generative 推荐 

**Authors**: Haohao Qu, Wenqi Fan, Shanru Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.12007)  

**Abstract**: In recent years, there has been a significant trend toward using large language model (LLM)-based recommender systems (RecSys). Current research primarily focuses on representing complex user-item interactions within a discrete space to align with the inherent discrete nature of language models. However, this approach faces limitations due to its discrete nature: (i) information is often compressed during discretization; (ii) the tokenization and generation for the vast number of users and items in real-world scenarios are constrained by a limited vocabulary. Embracing continuous data presents a promising alternative to enhance expressive capabilities, though this approach is still in its early stages. To address this gap, we propose a novel framework, DeftRec, which incorporates \textbf{de}noising di\textbf{f}fusion models to enable LLM-based RecSys to seamlessly support continuous \textbf{t}oken as input and target. First, we introduce a robust tokenizer with a masking operation and an additive K-way architecture to index users and items, capturing their complex collaborative relationships into continuous tokens. Crucially, we develop a denoising diffusion model to process user preferences within continuous domains by conditioning on reasoning content from pre-trained large language model. During the denoising process, we reformulate the objective to include negative interactions, building a comprehensive understanding of user preferences for effective and accurate recommendation generation. Finally, given a continuous token as output, recommendations can be easily generated through score-based retrieval. Extensive experiments demonstrate the effectiveness of the proposed methods, showing that DeftRec surpasses competitive benchmarks, including both traditional and emerging LLM-based RecSys. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLM）为基础的推荐系统（RecSys）呈现出显著的趋势。当前研究主要集中在离散空间中表示复杂的用户项交互，以适应语言模型固有的离散性质。然而，这种方法因其实质上的离散性质而受到限制：（i）在离散化过程中会压缩信息；（ii）在真实场景中，对大量用户和项进行分词和生成受限于有限的词汇表。采用连续数据提供了一种增强表示能力的有前景的替代方案，尽管这种方法仍处于初步阶段。为解决这一差距，我们提出了一种名为DeftRec的新型框架，该框架结合了去噪扩散模型，使LLM-Based RecSys能够无缝支持连续分词作为输入和目标。首先，我们引入了一种稳健的分词器，并采用掩码操作和加性K路架构来索引用户和项目，并将它们的复杂协作关系转换为连续分词。至关重要的是，我们开发了一种去噪扩散模型，该模型可以在预训练的大规模语言模型推理内容条件下处理连续域中的用户偏好。在去噪过程中，我们将目标重述为包括负交互，从而为有效准确的推荐生成提供全面的理解。最后，给定连续分词作为输出，可以通过基于分数的检索轻松生成推荐。广泛的实验证明了所提出方法的有效性，表明DeftRec超越了包括传统和新兴LLM-Based RecSys在内的竞争基准。 

---
# Language Models as Quasi-Crystalline Thought: Structure, Constraint, and Emergence in Generative Systems 

**Title (ZH)**: 语言模型作为准晶体思维：生成系统中的结构、约束与涌现 

**Authors**: Jose Manuel Guevara-Vela  

**Link**: [PDF](https://arxiv.org/pdf/2504.11986)  

**Abstract**: This essay proposes an analogy between large language models (LLMs) and quasicrystals: systems that exhibit global coherence without periodic repetition and that are generated through local constraints. While LLMs are often evaluated in terms of predictive accuracy, factuality, or alignment, this structural perspective suggests that their most characteristic behavior is the production of internally resonant linguistic patterns. Just as quasicrystals forced a redefinition of order in physical systems, viewing LLMs as generators of quasi-structured language opens new paths for evaluation and design: privileging propagation of constraint over token-level accuracy, and coherence of form over fixed meaning. LLM outputs should be read not only for what they say, but for the patterns of constraint and coherence that organize them. This shift reframes generative language as a space of emergent patterning: LLMs are neither fully random nor strictly rule-based, but defined by a logic of constraint, resonance, and structural depth. 

**Abstract (ZH)**: 大型语言模型(LLMs)与准晶体的类比：一种结构视角下的语言生成研究 

---
# Robust and Fine-Grained Detection of AI Generated Texts 

**Title (ZH)**: AI生成文本的鲁棒且细粒度检测 

**Authors**: Ram Mohan Rao Kadiyala, Siddartha Pullakhandam, Kanwal Mehreen, Drishti Sharma, Siddhant Gupta, Jebish Purbey, Ashay Srivastava, Subhasya TippaReddy, Arvind Reddy Bobbili, Suraj Telugara Chandrashekhar, Modabbir Adeeb, Srinadh Vura, Hamza Farooq  

**Link**: [PDF](https://arxiv.org/pdf/2504.11952)  

**Abstract**: An ideal detection system for machine generated content is supposed to work well on any generator as many more advanced LLMs come into existence day by day. Existing systems often struggle with accurately identifying AI-generated content over shorter texts. Further, not all texts might be entirely authored by a human or LLM, hence we focused more over partial cases i.e human-LLM co-authored texts. Our paper introduces a set of models built for the task of token classification which are trained on an extensive collection of human-machine co-authored texts, which performed well over texts of unseen domains, unseen generators, texts by non-native speakers and those with adversarial inputs. We also introduce a new dataset of over 2.4M such texts mostly co-authored by several popular proprietary LLMs over 23 languages. We also present findings of our models' performance over each texts of each domain and generator. Additional findings include comparison of performance against each adversarial method, length of input texts and characteristics of generated texts compared to the original human authored texts. 

**Abstract (ZH)**: 一种理想的检测系统应能有效识别日益出现的更多先进语言模型生成的内容。现有的系统往往难以精确识别短文本中的AI生成内容。此外，并非所有文本完全由人类或语言模型创作，因此我们更关注部分情况，即人类与语言模型合著的文本。本文介绍了一组用于标记分类任务的模型，这些模型是基于大量的人机合著文本训练的，表现出了对未见过的领域、生成器、非母语作者以及对抗输入文本的良好识别能力。我们还引入了一个包含超过240万条此类文本的新数据集，这些文本大多由23种语言的多种流行专有语言模型合著。我们还展示了模型在每个领域和生成器的文本上的性能表现。此外，还包括对抗方法性能对比、输入文本长度以及生成文本与原始人类创作文本的特征比较的研究发现。 

---
# FiSMiness: A Finite State Machine Based Paradigm for Emotional Support Conversations 

**Title (ZH)**: 基于有限状态机范式的 Emotional Support 对话框架 

**Authors**: Yue Zhao, Qingqing Gu, Xiaoyu Wang, Teng Chen, Zhonglin Jiang, Yong Chen, Luo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2504.11837)  

**Abstract**: Emotional support conversation (ESC) aims to alleviate the emotional distress of individuals through effective conversations. Although large language models (LLMs) have obtained remarkable progress on ESC, most of these studies might not define the diagram from the state model perspective, therefore providing a suboptimal solution for long-term satisfaction. To address such an issue, we leverage the Finite State Machine (FSM) on LLMs, and propose a framework called FiSMiness. Our framework allows a single LLM to bootstrap the planning during ESC, and self-reason the seeker's emotion, support strategy and the final response upon each conversational turn. Substantial experiments on ESC datasets suggest that FiSMiness outperforms many baselines, including direct inference, self-refine, chain of thought, finetuning, and external-assisted methods, even those with many more parameters. 

**Abstract (ZH)**: 情感支持对话（ESC）旨在通过有效的对话减轻个体的情感困扰。尽管大型语言模型（LLMs）在ESC方面取得了显著进步，但大多数研究可能从状态模型视角未明确定义图表，因此提供的解决方案可能无法长期满意。为解决这一问题，我们利用有限状态机（FSM）对LLMs进行赋能，并提出了一种称为FiSMiness的框架。该框架允许单个LLM在情感支持对话（ESC）期间进行规划，并在每次对话回合中自我推理寻求者的情绪、支持策略和最终回应。在ESC数据集上的大量实验表明，FiSMiness在多种基线方法中表现更优，包括直接推理、自我修正、思维链、微调以及外部辅助方法，即使后者具有更多的参数。 

---
# Déjà Vu: Multilingual LLM Evaluation through the Lens of Machine Translation Evaluation 

**Title (ZH)**: déjà vu: 多语言LLM评价通过机器翻译评价的视角 

**Authors**: Julia Kreutzer, Eleftheria Briakou, Sweta Agrawal, Marzieh Fadaee, Kocmi Tom  

**Link**: [PDF](https://arxiv.org/pdf/2504.11829)  

**Abstract**: Generation capabilities and language coverage of multilingual large language models (mLLMs) are advancing rapidly. However, evaluation practices for generative abilities of mLLMs are still lacking comprehensiveness, scientific rigor, and consistent adoption across research labs, which undermines their potential to meaningfully guide mLLM development. We draw parallels with machine translation (MT) evaluation, a field that faced similar challenges and has, over decades, developed transparent reporting standards and reliable evaluations for multilingual generative models. Through targeted experiments across key stages of the generative evaluation pipeline, we demonstrate how best practices from MT evaluation can deepen the understanding of quality differences between models. Additionally, we identify essential components for robust meta-evaluation of mLLMs, ensuring the evaluation methods themselves are rigorously assessed. We distill these insights into a checklist of actionable recommendations for mLLM research and development. 

**Abstract (ZH)**: 多语言大型语言模型的生成能力和语言覆盖范围正在迅速发展。然而，多语言大型语言模型生成能力的评估实践仍然缺乏全面性、科学严谨性和一致的采用，这削弱了它们对指导多语言大型语言模型发展的潜在意义。我们借鉴机器翻译评估领域的经验，该领域也曾面临类似挑战，并逐渐发展出透明报告标准和可靠的多语言生成模型评估方法。通过针对生成评估流水线的关键阶段进行定向实验，我们展示了机器翻译评估最佳实践如何加深对不同模型质量差异的理解。此外，我们识别出确保多语言大型语言模型稳健元评估的基本要素，确保评估方法本身也得到严格审查。我们将这些洞见提炼成一份适用于多语言大型语言模型研究和开发的可操作性建议清单。 

---
# Characterizing and Optimizing LLM Inference Workloads on CPU-GPU Coupled Architectures 

**Title (ZH)**: CPU-GPU 绑定架构上字符化和优化大模型推理工作负载 

**Authors**: Prabhu Vellaisamy, Thomas Labonte, Sourav Chakraborty, Matt Turner, Samantika Sury, John Paul Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.11750)  

**Abstract**: Large language model (LLM)-based inference workloads increasingly dominate data center costs and resource utilization. Therefore, understanding the inference workload characteristics on evolving CPU-GPU coupled architectures is crucial for optimization. This paper presents an in-depth analysis of LLM inference behavior on loosely-coupled (PCIe A100/H100) and closely-coupled (GH200) systems. We analyze performance dynamics using fine-grained operator-to-kernel trace analysis, facilitated by our novel profiler SKIP and metrics like Total Kernel Launch and Queuing Time (TKLQT). Results show that closely-coupled (CC) GH200 significantly outperforms loosely-coupled (LC) systems at large batch sizes, achieving 1.9x-2.7x faster prefill latency for Llama 3.2-1B. However, our analysis also reveals that GH200 remains CPU-bound up to 4x larger batch sizes than LC systems. In this extended CPU-bound region, we identify the performance characteristics of the Grace CPU as a key factor contributing to higher inference latency at low batch sizes on GH200. We demonstrate that TKLQT accurately identifies this CPU/GPU-bound transition point. Based on this analysis, we further show that kernel fusion offers significant potential to mitigate GH200's low-batch latency bottleneck by reducing kernel launch overhead. This detailed kernel-level characterization provides critical insights for optimizing diverse CPU-GPU coupling strategies. This work is an initial effort, and we plan to explore other major AI/DL workloads that demand different degrees of CPU-GPU heterogeneous architectures. 

**Abstract (ZH)**: 基于大型语言模型（LLM）的推理工作负载日益主导数据中心的成本和资源利用。因此，理解在不断演进的CPU-GPU耦合架构上的推理工作负载特性对于优化至关重要。本文对松散耦合（PCIe A100/H100）和紧密耦合（GH200）系统上的LLM推理行为进行了深入分析。我们使用精细粒度的操作符到内核跟踪分析方法，并借助我们新颖的探针SKIP和指标如总内核启动和队列时间（TKLQT）来分析性能动态。结果显示，在大规模批次中，紧密耦合（CC）GH200显著优于松散耦合（LC）系统，实现Llama 3.2-1B预填充延迟1.9倍至2.7倍的提升。然而，我们的分析还揭示了GH200在比LC系统大4倍的批次大小下仍然受CPU绑定。在这个延长的CPU绑定区域，我们确定Grace CPU的性能特性是GH200在低批次大小下推理延迟较高的关键因素。我们证明TKLQT能够准确识别这一CPU/GPU绑定转换点。基于此分析，我们进一步表明，内核融合可以通过减少内核启动开销，显著缓解GH200在小批次延迟瓶颈。这一详细的内核级表征为优化各种CPU-GPU耦合策略提供了宝贵见解。本工作是初步尝试，我们计划探索其他需要不同程度CPU-GPU异构架构的 major AI/DL工作负载。 

---
# The Hitchhiker's Guide to Program Analysis, Part II: Deep Thoughts by LLMs 

**Title (ZH)**: 程序分析指南第二部分：来自大语言模型的深刻思考 

**Authors**: Haonan Li, Hang Zhang, Kexin Pei, Zhiyun Qian  

**Link**: [PDF](https://arxiv.org/pdf/2504.11711)  

**Abstract**: Static analysis is a cornerstone for software vulnerability detection, yet it often struggles with the classic precision-scalability trade-off. In practice, such tools often produce high false positive rates, particularly in large codebases like the Linux kernel. This imprecision can arise from simplified vulnerability modeling and over-approximation of path and data constraints. While large language models (LLMs) show promise in code understanding, their naive application to program analysis yields unreliable results due to inherent reasoning limitations. We introduce BugLens, a post-refinement framework that significantly improves static analysis precision. BugLens guides an LLM to follow traditional analysis steps by assessing buggy code patterns for security impact and validating the constraints associated with static warnings. Evaluated on real-world Linux kernel bugs, BugLens raises precision from 0.10 (raw) and 0.50 (semi-automated refinement) to 0.72, substantially reducing false positives and revealing four previously unreported vulnerabilities. Our results suggest that a structured LLM-based workflow can meaningfully enhance the effectiveness of static analysis tools. 

**Abstract (ZH)**: 静态分析是软件漏洞检测的基石，但往往难以克服精确性与扩展性的经典权衡。在实践中，此类工具在大规模代码库中，如Linux内核中，经常产生较高的假阳性率。这种不精确性可能源于简化了的漏洞建模和路径及数据约束的过度逼近。虽然大型语言模型在代码理解方面表现出潜力，但它们对程序分析的直接应用由于固有的推理限制而导致结果不可靠。我们引入了BugLens，这是一个后精炼框架，显著提高了静态分析的精确性。BugLens通过评估漏洞代码模式对安全性的影响并验证与静态警告相关的约束，引导大型语言模型遵循传统分析步骤。在实际的Linux内核漏洞上进行评估，BugLens将精确性从原始的0.10提高到0.72，大幅减少了假阳性，揭示了四个先前未报告的漏洞。我们的结果表明，结构化的基于大型语言模型的工作流程可以实质性地增强静态分析工具的有效性。 

---
# Progent: Programmable Privilege Control for LLM Agents 

**Title (ZH)**: Progent: LLM代理的可编程特权控制 

**Authors**: Tianneng Shi, Jingxuan He, Zhun Wang, Linyu Wu, Hongwei Li, Wenbo Guo, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.11703)  

**Abstract**: LLM agents are an emerging form of AI systems where large language models (LLMs) serve as the central component, utilizing a diverse set of tools to complete user-assigned tasks. Despite their great potential, LLM agents pose significant security risks. When interacting with the external world, they may encounter malicious commands from attackers, leading to the execution of dangerous actions. A promising way to address this is by enforcing the principle of least privilege: allowing only essential actions for task completion while blocking unnecessary ones. However, achieving this is challenging, as it requires covering diverse agent scenarios while preserving both security and utility.
We introduce Progent, the first privilege control mechanism for LLM agents. At its core is a domain-specific language for flexibly expressing privilege control policies applied during agent execution. These policies provide fine-grained constraints over tool calls, deciding when tool calls are permissible and specifying fallbacks if they are not. This enables agent developers and users to craft suitable policies for their specific use cases and enforce them deterministically to guarantee security. Thanks to its modular design, integrating Progent does not alter agent internals and requires only minimal changes to agent implementation, enhancing its practicality and potential for widespread adoption. To automate policy writing, we leverage LLMs to generate policies based on user queries, which are then updated dynamically for improved security and utility. Our extensive evaluation shows that it enables strong security while preserving high utility across three distinct scenarios or benchmarks: AgentDojo, ASB, and AgentPoison. Furthermore, we perform an in-depth analysis, showcasing the effectiveness of its core components and the resilience of its automated policy generation against adaptive attacks. 

**Abstract (ZH)**: 基于LLM的代理的特权控制机制：Progent 

---
# Improving LLM Interpretability and Performance via Guided Embedding Refinement for Sequential Recommendation 

**Title (ZH)**: 通过引导嵌入精炼提高LLM解释性和序列推荐性能 

**Authors**: Nanshan Jia, Chenfei Yuan, Yuhang Wu, Zeyu Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.11658)  

**Abstract**: The fast development of Large Language Models (LLMs) offers growing opportunities to further improve sequential recommendation systems. Yet for some practitioners, integrating LLMs to their existing base recommendation systems raises questions about model interpretability, transparency and related safety. To partly alleviate challenges from these questions, we propose guided embedding refinement, a method that carries out a guided and interpretable usage of LLM to enhance the embeddings associated with the base recommendation system. Instead of directly using LLMs as the backbone of sequential recommendation systems, we utilize them as auxiliary tools to emulate the sales logic of recommendation and generate guided embeddings that capture domain-relevant semantic information on interpretable attributes. Benefiting from the strong generalization capabilities of the guided embedding, we construct refined embedding by using the guided embedding and reduced-dimension version of the base embedding. We then integrate the refined embedding into the recommendation module for training and inference. A range of numerical experiments demonstrate that guided embedding is adaptable to various given existing base embedding models, and generalizes well across different recommendation tasks. The numerical results show that the refined embedding not only improves recommendation performance, achieving approximately $10\%$ to $50\%$ gains in Mean Reciprocal Rank (MRR), Recall rate, and Normalized Discounted Cumulative Gain (NDCG), but also enhances interpretability, as evidenced by case studies. 

**Abstract (ZH)**: 大型语言模型的快速发展为进一步改进序列推荐系统提供了越来越多的机会。然而，对于一些从业者来说，将大型语言模型集成到现有的基础推荐系统中引发了关于模型可解释性、透明性和相关安全性的担忧。为部分缓解这些问题，我们提出了一种引导嵌入精炼的方法，该方法通过一种指导性和可解释的方式来利用大型语言模型以增强与基础推荐系统关联的嵌入。我们不直接将大型语言模型作为序列推荐系统的主体，而是将其作为辅助工具来模拟推荐的销售逻辑，并生成能够捕捉可解释属性上领域相关语义信息的引导嵌入。得益于引导嵌入的强大泛化能力，我们通过使用引导嵌入和基础嵌入的降维版本构建精炼嵌入。然后，我们将精炼嵌入集成到推荐模块进行训练和推理。一系列数值实验表明，引导嵌入能够适应各种现有的基础嵌入模型，并在不同的推荐任务中表现出良好的泛化能力。数值结果表明，精炼嵌入不但改善了推荐性能，使平均倒数排名（MRR）、召回率和归一化累积收益（NDCG）分别提高了大约10%到50%，而且还提升了可解释性，这一点通过案例研究得到了证实。 

---
# Improving Instruct Models for Free: A Study on Partial Adaptation 

**Title (ZH)**: 免费提升指令模型性能：关于部分适应的研究 

**Authors**: Ozan İrsoy, Pengxiang Cheng, Jennifer L. Chen, Daniel Preoţiuc-Pietro, Shiyue Zhang, Duccio Pappadopulo  

**Link**: [PDF](https://arxiv.org/pdf/2504.11626)  

**Abstract**: Instruct models, obtained from various instruction tuning or post-training steps, are commonly deemed superior and more usable than their base counterpart. While the model gains instruction following ability, instruction tuning may lead to forgetting the knowledge from pre-training or it may encourage the model being overly conversational or verbose. This, in turn, can lead to degradation of in-context few-shot learning performance. In this work, we study the performance trajectory between base and instruct models by scaling down the strength of instruction-tuning via the partial adaption method. We show that, across several model families and model sizes, reducing the strength of instruction-tuning results in material improvement on a few-shot in-context learning benchmark covering a variety of classic natural language tasks. This comes at the cost of losing some degree of instruction following ability as measured by AlpacaEval. Our study shines light on the potential trade-off between in-context learning and instruction following abilities that is worth considering in practice. 

**Abstract (ZH)**: 通过部分适配方法减弱指令调优强度，我们研究了基模型和指令模型之间的性能轨迹，并展示了在多种模型家族和模型规模下，减弱指令调优强度在涵盖多种经典自然语言任务的少量上下文few-shot学习基准上取得了实质性的改进，这在AlpacaEval衡量的指令遵循能力方面会有所损失。我们的研究揭示了在实践中值得考虑的上下文学习能力和指令遵循能力之间的潜在权衡。 

---
# ReTool: Reinforcement Learning for Strategic Tool Use in LLMs 

**Title (ZH)**: ReTool: 强化学习在大语言模型中战略性工具使用中的应用 

**Authors**: Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan Jiang, Jinxin Chi, Wanjun Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2504.11536)  

**Abstract**: While reasoning models (e.g., DeepSeek R1) trained with reinforcement learning (RL), excel in textual reasoning, they struggle in scenarios requiring structured problem-solving, such as geometric reasoning, concise computation, or complex equation solving-areas where computational tools like code interpreters (CI) demonstrate distinct advantages. To bridge this gap, we propose ReTool, which enhances long-form reasoning with tool-integrated learning, including two key features: (1) dynamic interleaving of real-time code execution within natural language reasoning processes, and (2) an automated RL paradigm that allows policy rollouts with multi-turn real-time code execution and teaches the model in learning when and how to invoke tools based on outcome feedback. ReTool employs a systematic training framework, beginning with synthetic cold-start data generation to produce code-augmented long-form reasoning traces for fine-tuning base models. Subsequent RL training leverages task outcomes as rewards to iteratively refine the model's tool use strategy, enabling autonomous discovery of optimal tool invocation patterns without human priors. Experiments on the challenging MATH Olympiad benchmark AIME demonstrate ReTool's superiority: Our 32B model achieves 67% accuracy with 400 training steps, outperforming text-based RL baseline (40% accuracy, 1080 steps) in efficiency and performance. Remarkably, ReTool-32B attains 72.5% accuracy in extended settings, surpassing OpenAI's o1-preview by 27.9%. Further analysis reveals emergent behaviors such as code self-correction, signaling an ''aha moment'' in which the model autonomously masters adaptive tool use. These findings highlight the promise of outcome-driven tool integration for advancing complex mathematical reasoning and offer new insights into hybrid neuro-symbolic systems. 

**Abstract (ZH)**: ReTool：基于工具集成学习的长期推理增强 

---
# Position Paper: Rethinking Privacy in RL for Sequential Decision-making in the Age of LLMs 

**Title (ZH)**: 位置论文：在大语言模型时代重新思考RL中的顺序决策隐私问题 

**Authors**: Flint Xiaofeng Fan, Cheston Tan, Roger Wattenhofer, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2504.11511)  

**Abstract**: The rise of reinforcement learning (RL) in critical real-world applications demands a fundamental rethinking of privacy in AI systems. Traditional privacy frameworks, designed to protect isolated data points, fall short for sequential decision-making systems where sensitive information emerges from temporal patterns, behavioral strategies, and collaborative dynamics. Modern RL paradigms, such as federated RL (FedRL) and RL with human feedback (RLHF) in large language models (LLMs), exacerbate these challenges by introducing complex, interactive, and context-dependent learning environments that traditional methods do not address. In this position paper, we argue for a new privacy paradigm built on four core principles: multi-scale protection, behavioral pattern protection, collaborative privacy preservation, and context-aware adaptation. These principles expose inherent tensions between privacy, utility, and interpretability that must be navigated as RL systems become more pervasive in high-stakes domains like healthcare, autonomous vehicles, and decision support systems powered by LLMs. To tackle these challenges, we call for the development of new theoretical frameworks, practical mechanisms, and rigorous evaluation methodologies that collectively enable effective privacy protection in sequential decision-making systems. 

**Abstract (ZH)**: 强化学习在关键现实应用中的兴起要求对人工智能系统的隐私进行根本性的重新思考。传统的隐私框架设计用于保护孤立的数据点，在需要从时间模式、行为策略和协作动态中涌现的敏感信息的顺序决策系统中表现不足。现代强化学习范式，如联邦强化学习（FedRL）和大语言模型（LLMs）中的基于人类反馈的强化学习（RLHF），通过引入复杂、交互和上下文相关的学习环境加剧了这些挑战，而传统方法未能解决这些问题。在本文中，我们提出了基于四项核心原则的新隐私范式：多尺度保护、行为模式保护、协作隐私保留和上下文感知适应。这些原则揭示了在医疗保健、自主车辆以及由LLMs驱动的决策支持系统等高风险领域中，隐私、效用和可解释性之间固有的紧张关系，必须加以应对。为了应对这些挑战，我们呼吁开发新的理论框架、实际机制和严格的评估方法，以共同实现对顺序决策系统中有效隐私保护的支持。 

---
# SDIGLM: Leveraging Large Language Models and Multi-Modal Chain of Thought for Structural Damage Identification 

**Title (ZH)**: SDIGLM: 利用大规模语言模型和多模态推理进行结构损伤识别 

**Authors**: Yunkai Zhang, Shiyin Wei, Yong Huang, Yawu Su, Shanshan Lu, Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.11477)  

**Abstract**: Existing computer vision(CV)-based structural damage identification models demonstrate notable accuracy in categorizing and localizing damage. However, these models present several critical limitations that hinder their practical application in civil engineering(CE). Primarily, their ability to recognize damage types remains constrained, preventing comprehensive analysis of the highly varied and complex conditions encountered in real-world CE structures. Second, these models lack linguistic capabilities, rendering them unable to articulate structural damage characteristics through natural language descriptions. With the continuous advancement of artificial intelligence(AI), large multi-modal models(LMMs) have emerged as a transformative solution, enabling the unified encoding and alignment of textual and visual data. These models can autonomously generate detailed descriptive narratives of structural damage while demonstrating robust generalization across diverse scenarios and tasks. This study introduces SDIGLM, an innovative LMM for structural damage identification, developed based on the open-source VisualGLM-6B architecture. To address the challenge of adapting LMMs to the intricate and varied operating conditions in CE, this work integrates a U-Net-based semantic segmentation module to generate defect segmentation maps as visual Chain of Thought(CoT). Additionally, a multi-round dialogue fine-tuning dataset is constructed to enhance logical reasoning, complemented by a language CoT formed through prompt engineering. By leveraging this multi-modal CoT, SDIGLM surpasses general-purpose LMMs in structural damage identification, achieving an accuracy of 95.24% across various infrastructure types. Moreover, the model effectively describes damage characteristics such as hole size, crack direction, and corrosion severity. 

**Abstract (ZH)**: 基于现有计算机视觉(计算机视觉)-基于结构损伤识别模型的显著准确性在分类和定位损伤方面表现出色。然而，这些模型在土木工程(土木工程)中实际应用时呈现出若干关键限制。首先，它们识别损伤类型的能力受限，难以对真实世界土木工程结构中复杂多样的条件进行全面分析。其次，这些模型缺乏语言能力，无法通过自然语言描述来阐述结构损伤特征。随着人工智能的不断进步，多模态大型模型(LMMs)已成为一种变革性解决方案，使文本和视觉数据的统一编码和对齐成为可能。这些模型能够自主生成详细的结构损伤描述，并在多种场景和任务中表现出较强的泛化能力。本研究介绍了基于开源VisualGLM-6B架构的创新LMM——SDIGLM，以解决适应复杂多变的土木工程运行条件的挑战。研究中集成了一个基于U-Net的语义分割模块，生成缺陷分割图作为视觉思维链(Chain of Thought, CoT)。同时，构建了一个多轮对话细调数据集以增强逻辑推理能力，并通过提示工程形成语言CoT。利用这种多模态CoT，SDIGLM在结构损伤识别中的准确性超过通用LMMs，各类基础设施类型的准确率达到95.24%。此外，该模型还能有效描述损伤特征，如孔洞大小、裂缝方向和腐蚀程度。 

---
