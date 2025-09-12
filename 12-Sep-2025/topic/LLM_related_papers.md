# The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs 

**Title (ZH)**: 递减回报的错觉：测量LLM中的长期执行能力 

**Authors**: Akshit Sinha, Arvindh Arun, Shashwat Goel, Steffen Staab, Jonas Geiping  

**Link**: [PDF](https://arxiv.org/pdf/2509.09677)  

**Abstract**: Does continued scaling of large language models (LLMs) yield diminishing returns? Real-world value often stems from the length of task an agent can complete. We start this work by observing the simple but counterintuitive fact that marginal gains in single-step accuracy can compound into exponential improvements in the length of a task a model can successfully complete. Then, we argue that failures of LLMs when simple tasks are made longer arise from mistakes in execution, rather than an inability to reason. We propose isolating execution capability, by explicitly providing the knowledge and plan needed to solve a long-horizon task. We find that larger models can correctly execute significantly more turns even when small models have 100\% single-turn accuracy. We observe that the per-step accuracy of models degrades as the number of steps increases. This is not just due to long-context limitations -- curiously, we observe a self-conditioning effect -- models become more likely to make mistakes when the context contains their errors from prior turns. Self-conditioning does not reduce by just scaling the model size. In contrast, recent thinking models do not self-condition, and can also execute much longer tasks in a single turn. We conclude by benchmarking frontier thinking models on the length of task they can execute in a single turn. Overall, by focusing on the ability to execute, we hope to reconcile debates on how LLMs can solve complex reasoning problems yet fail at simple tasks when made longer, and highlight the massive benefits of scaling model size and sequential test-time compute for long-horizon tasks. 

**Abstract (ZH)**: 持续扩展大型语言模型（LLMs）会产生边际效益递减的现象吗？真实世界的价值往往源自模型能够完成的任务长度。我们从一个简单但直观相反的事实开始——单一步骤准确性的小幅提升可以转化为模型能够成功完成任务长度的指数级改进。然后，我们指出当简单任务变得更长时，LLMs 的失败并非因为推理能力不足，而是执行上的错误。我们提出通过明确提供解决长期任务所需的知识和计划来隔离执行能力。我们发现，即使小型模型在单步准确性上达到100%，大型模型也能正确执行显著更多的步骤。我们观察到，随着步骤数量的增加，模型的单步准确性会下降。这不仅是因为长上下文的限制——有趣的是，我们发现一种自归因效应——当上下文中包含前一轮的错误时，模型更可能犯错。仅通过扩展模型规模无法减少这种自归因效应。相比之下，最近的思想模型不表现出自归因，并且可以在单步中执行更长的任务。最后，我们通过基准测试前沿思想模型，评估它们在单步中能够执行的任务长度。总体而言，通过关注执行能力，我们希望解决关于LLMs如何解决复杂推理问题但在任务变得更长时却在简单任务上失败的辩论，并强调为长期任务扩展模型规模和序贯测试时计算的巨大好处。 

---
# Inteligencia Artificial jurídica y el desafío de la veracidad: análisis de alucinaciones, optimización de RAG y principios para una integración responsable 

**Title (ZH)**: 人工智能法律与真实性挑战：幻觉分析、RAG优化及负责任集成的原则 

**Authors**: Alex Dantart  

**Link**: [PDF](https://arxiv.org/pdf/2509.09467)  

**Abstract**: This technical report analyzes the challenge of "hallucinations" (false information) in LLMs applied to law. It examines their causes, manifestations, and the effectiveness of the RAG mitigation strategy, highlighting its limitations and proposing holistic optimizations. The paper explores the ethical and regulatory implications, emphasizing human oversight as an irreplaceable role. It concludes that the solution lies not in incrementally improving generative models, but in adopting a "consultative" AI paradigm that prioritizes veracity and traceability, acting as a tool to amplify, not replace, professional judgment.
--
Este informe técnico analiza el desafío de las "alucinaciones" (información falsa) en los LLMs aplicados al derecho. Se examinan sus causas, manifestaciones y la efectividad de la estrategia de mitigación RAG, exponiendo sus limitaciones y proponiendo optimizaciones holísticas. Se exploran las implicaciones éticas y regulatorias, enfatizando la supervisión humana como un rol insustituible. El documento concluye que la solución no reside en mejorar incrementalmente los modelos generativos, sino en adoptar un paradigma de IA "consultiva" que priorice la veracidad y la trazabilidad, actuando como una herramienta para amplificar, y no sustituir, el juicio profesional. 

**Abstract (ZH)**: 技术报告分析了法律应用领域大语言模型中的“幻觉”（虚假信息）挑战。该报告探讨了其原因、表现形式以及RAG缓解策略的有效性，指出其局限性并提出整体优化建议。文章探讨了伦理和监管影响，强调人类监督不可或缺的作用。报告结论认为，解决方案不在于逐步改进生成模型，而在于采用以验证性和可追溯性为首要原则的“咨询式”人工智能范式，作为工具来增强而非取代专业判断。 

---
# TORSO: Template-Oriented Reasoning Towards General Tasks 

**Title (ZH)**: 模板导向的推理 toward 通用任务 

**Authors**: Minhyuk Kim, Seungyoon Lee, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2509.09448)  

**Abstract**: The approaches that guide Large Language Models (LLMs) to emulate human reasoning during response generation have emerged as an effective method for enabling them to solve complex problems in a step-by-step manner, thereby achieving superior performance. However, most existing approaches using few-shot prompts to generate responses heavily depend on the provided examples, limiting the utilization of the model's inherent reasoning capabilities. Moreover, constructing task-specific few-shot prompts is often costly and may lead to inconsistencies across different tasks. In this work, we introduce Template-Oriented Reasoning (TORSO), which elicits the model to utilize internal reasoning abilities to generate proper responses across various tasks without the need for manually crafted few-shot examples. Our experimental results demonstrate that TORSO achieves strong performance on diverse LLMs benchmarks with reasonable rationales. 

**Abstract (ZH)**: 指导大型语言模型（LLMs）在响应生成过程中模拟人类推理的方法已被证明是使它们能够逐步解决复杂问题、从而实现卓越性能的有效方法。然而，现有的大多数使用少量示例生成响应的方法严重依赖于提供的示例，限制了模型内在推理能力的充分利用。此外，构建特定任务的少量示例提示通常成本较高，并可能导致不同任务间的一致性问题。在本工作中，我们提出了模板导向推理（TORSO），这是一种使模型能够利用其内部推理能力生成适用于多种任务的适当响应的方法，而无需手动构建的少量示例。我们的实验结果表明，TORSO在多种LLM基准测试中表现出强劲性能，并具有合理的推理依据。 

---
# Towards Adaptive ML Benchmarks: Web-Agent-Driven Construction, Domain Expansion, and Metric Optimization 

**Title (ZH)**: 面向自适应机器学习基准：基于Web-Agent的构建、领域扩展与度量优化 

**Authors**: Hangyi Jia, Yuxi Qian, Hanwen Tong, Xinhui Wu, Lin Chen, Feng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2509.09321)  

**Abstract**: Recent advances in large language models (LLMs) have enabled the emergence of general-purpose agents for automating end-to-end machine learning (ML) workflows, including data analysis, feature engineering, model training, and competition solving. However, existing benchmarks remain limited in task coverage, domain diversity, difficulty modeling, and evaluation rigor, failing to capture the full capabilities of such agents in realistic settings. We present TAM Bench, a diverse, realistic, and structured benchmark for evaluating LLM-based agents on end-to-end ML tasks. TAM Bench features three key innovations: (1) A browser automation and LLM-based task acquisition system that automatically collects and structures ML challenges from platforms such as Kaggle, AIcrowd, and Biendata, spanning multiple task types and data modalities (e.g., tabular, text, image, graph, audio); (2) A leaderboard-driven difficulty modeling mechanism that estimates task complexity using participant counts and score dispersion, enabling scalable and objective task calibration; (3) A multi-dimensional evaluation framework incorporating performance, format compliance, constraint adherence, and task generalization. Based on 150 curated AutoML tasks, we construct three benchmark subsets of different sizes -- Lite, Medium, and Full -- designed for varying evaluation scenarios. The Lite version, with 18 tasks and balanced coverage across modalities and difficulty levels, serves as a practical testbed for daily benchmarking and comparative studies. 

**Abstract (ZH)**: 近期大型语言模型的进展使通用型代理能够自动化端到端的机器学习工作流，包括数据分析、特征工程、模型训练和竞赛解决。然而，现有的基准测试在任务覆盖范围、领域多样性、难度建模和评估严谨性方面仍有限制，未能捕捉到这些代理在实际环境中的全部能力。我们提出了TAM Bench，这是一个多样化的、现实的和结构化的基准测试，用于评估基于大型语言模型的代理在端到端机器学习任务中的性能。TAM Bench 的三大创新包括：（1）一个基于浏览器自动化和大型语言模型的任务获取系统，自动从Kaggle、Aicrowd和Biendata等平台收集和结构化跨多种任务类型和数据模态（例如，表格、文本、图像、图、音频）的机器学习挑战；（2）一个基于排行榜的难度建模机制，利用参与者数量和分数分布估计任务复杂度，实现可扩展和客观的任务标度；（3）一个多维评估框架，包括性能、格式合规性、约束遵守性和任务泛化。基于150个精选的自动化机器学习任务，我们构建了三个不同规模的基准子集——轻量级、中等和完整——适用于不同的评估场景。轻量级版本，包含18个任务，涵盖多种模态和难度级别，适合作为日常基准测试和对比研究的实际试验场。 

---
# LightAgent: Production-level Open-source Agentic AI Framework 

**Title (ZH)**: LightAgent: 生产级开源代理人型AI框架 

**Authors**: Weige Cai, Tong Zhu, Jinyi Niu, Ruiqi Hu, Lingyao Li, Tenglong Wang, Xiaowu Dai, Weining Shen, Liwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09292)  

**Abstract**: With the rapid advancement of large language models (LLMs), Multi-agent Systems (MAS) have achieved significant progress in various application scenarios. However, substantial challenges remain in designing versatile, robust, and efficient platforms for agent deployment. To address these limitations, we propose \textbf{LightAgent}, a lightweight yet powerful agentic framework, effectively resolving the trade-off between flexibility and simplicity found in existing frameworks. LightAgent integrates core functionalities such as Memory (mem0), Tools, and Tree of Thought (ToT), while maintaining an extremely lightweight structure. As a fully open-source solution, it seamlessly integrates with mainstream chat platforms, enabling developers to easily build self-learning agents. We have released LightAgent at \href{this https URL}{this https URL} 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的快速发展，多智能体系统（MAS）在各种应用场景中取得了显著进展。然而，在设计 versatile、robust 和 efficient 的智能体部署平台方面仍面临重大挑战。为解决这些限制，我们提出了 LightAgent，一个轻量级但强大的智能体框架，有效解决了现有框架中存在的灵活性与简单性之间的trade-off。LightAgent 结合了核心功能，如 Memory（mem0）、Tools 和 Tree of Thought（ToT），同时保持了极其轻量级的结构。作为完全开源的解决方案，它无缝集成到主流聊天平台，使开发者能够轻松构建自学习智能体。我们已在 \href{this https URL}{this https URL} 发布了 LightAgent。 

---
# Fusing Knowledge and Language: A Comparative Study of Knowledge Graph-Based Question Answering with LLMs 

**Title (ZH)**: 知识与语言融合：基于知识图谱的问答与大语言模型的 comparative study 

**Authors**: Vaibhav Chaudhary, Neha Soni, Narotam Singh, Amita Kapoor  

**Link**: [PDF](https://arxiv.org/pdf/2509.09272)  

**Abstract**: Knowledge graphs, a powerful tool for structuring information through relational triplets, have recently become the new front-runner in enhancing question-answering systems. While traditional Retrieval Augmented Generation (RAG) approaches are proficient in fact-based and local context-based extraction from concise texts, they encounter limitations when addressing the thematic and holistic understanding of complex, extensive texts, requiring a deeper analysis of both text and context. This paper presents a comprehensive technical comparative study of three different methodologies for constructing knowledge graph triplets and integrating them with Large Language Models (LLMs) for question answering: spaCy, Stanford CoreNLP-OpenIE, and GraphRAG, all leveraging open source technologies. We evaluate the effectiveness, feasibility, and adaptability of these methods by analyzing their capabilities, state of development, and their impact on the performance of LLM-based question answering. Experimental results indicate that while OpenIE provides the most comprehensive coverage of triplets, GraphRAG demonstrates superior reasoning abilities among the three. We conclude with a discussion on the strengths and limitations of each method and provide insights into future directions for improving knowledge graph-based question answering. 

**Abstract (ZH)**: 知识图谱，作为一种通过关系三元组结构化信息的有力工具，近年来已成为提升问答系统的新前沿。虽然传统的检索增强生成（RAG）方法在从小型文本中提取事实和局部上下文方面表现出色，但在处理复杂、 extensive 文本的主题性和整体理解方面存在局限性，需要对文本和上下文进行更深入的分析。本文对三种不同的知识图谱三元组构建方法及其与大型语言模型（LLMs）结合用于问答的技术进行了全面的技术对比研究：spaCy、Stanford CoreNLP-OpenIE 和 GraphRAG，所有方法均利用开源技术。我们通过分析这些方法的能力、发展状态及其对基于LLM的问答性能的影响来评估它们的有效性、可行性和适应性。实验结果表明，尽管OpenIE提供了最全面的三元组覆盖范围，但GraphRAG在三种方法中表现出了更强的推理能力。我们在此基础上讨论了每种方法的优缺点，并提供了关于改进基于知识图谱的问答的未来方向的见解。 

---
# Jupiter: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search 

**Title (ZH)**: 木星：通过笔记本和推理时值导向搜索增强大语言模型数据分析能力 

**Authors**: Shuocheng Li, Yihao Liu, Silin Du, Wenxuan Zeng, Zhe Xu, Mengyu Zhou, Yeye He, Haoyu Dong, Shi Han, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09245)  

**Abstract**: Large language models (LLMs) have shown great promise in automating data science workflows, but existing models still struggle with multi-step reasoning and tool use, which limits their effectiveness on complex data analysis tasks. To address this, we propose a scalable pipeline that extracts high-quality, tool-based data analysis tasks and their executable multi-step solutions from real-world Jupyter notebooks and associated data files. Using this pipeline, we introduce NbQA, a large-scale dataset of standardized task-solution pairs that reflect authentic tool-use patterns in practical data science scenarios. To further enhance multi-step reasoning, we present Jupiter, a framework that formulates data analysis as a search problem and applies Monte Carlo Tree Search (MCTS) to generate diverse solution trajectories for value model learning. During inference, Jupiter combines the value model and node visit counts to efficiently collect executable multi-step plans with minimal search steps. Experimental results show that Qwen2.5-7B and 14B-Instruct models on NbQA solve 77.82% and 86.38% of tasks on InfiAgent-DABench, respectively-matching or surpassing GPT-4o and advanced agent frameworks. Further evaluations demonstrate improved generalization and stronger tool-use reasoning across diverse multi-step reasoning tasks. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在自动化数据科学工作流方面表现出巨大的潜力，但现有模型仍然在多步推理和工具使用方面存在局限性，这限制了它们在复杂数据分析任务中的有效性。为此，我们提出了一种可扩展的流水线，从真实的Jupyter笔记本及其相关数据文件中提取高质量的工具基数据分析任务及其可执行的多步解决方案。通过该流水线，我们介绍了NbQA数据集，这是一个大规模的任务-解决方案对集合，反映了实际数据科学场景中的真实工具使用模式。为增强多步推理，我们提出了Jupiter框架，将数据分析问题形式化为搜索问题，并采用蒙特卡洛树搜索（MCTS）生成多样性的解决方案轨迹以用于价值模型学习。在推理过程中，Jupiter结合价值模型和节点访问计数，高效地收集可执行的多步计划，同时减少搜索步骤。实验结果表明，Qwen2.5-7B和14B-Instruct模型在InfiAgent-DABench上分别解决了77.82%和86.38%的任务，匹配甚至超过了GPT-4o和先进的代理框架。进一步的评估表明，Jupiter在多步推理任务中展示了更强的泛化能力和工具使用推理能力。 

---
# Enabling Regulatory Multi-Agent Collaboration: Architecture, Challenges, and Solutions 

**Title (ZH)**: 实现监管多代理协作：架构、挑战与解决方案 

**Authors**: Qinnan Hu, Yuntao Wang, Yuan Gao, Zhou Su, Linkang Du  

**Link**: [PDF](https://arxiv.org/pdf/2509.09215)  

**Abstract**: Large language models (LLMs)-empowered autonomous agents are transforming both digital and physical environments by enabling adaptive, multi-agent collaboration. While these agents offer significant opportunities across domains such as finance, healthcare, and smart manufacturing, their unpredictable behaviors and heterogeneous capabilities pose substantial governance and accountability challenges. In this paper, we propose a blockchain-enabled layered architecture for regulatory agent collaboration, comprising an agent layer, a blockchain data layer, and a regulatory application layer. Within this framework, we design three key modules: (i) an agent behavior tracing and arbitration module for automated accountability, (ii) a dynamic reputation evaluation module for trust assessment in collaborative scenarios, and (iii) a malicious behavior forecasting module for early detection of adversarial activities. Our approach establishes a systematic foundation for trustworthy, resilient, and scalable regulatory mechanisms in large-scale agent ecosystems. Finally, we discuss the future research directions for blockchain-enabled regulatory frameworks in multi-agent systems. 

**Abstract (ZH)**: 大型语言模型赋能的自主代理正通过实现适应性的多代理协作， transformative地改变数字和物理环境。尽管这些代理在金融、医疗和智能制造等领域提供了重要的机遇，但它们的不可预测行为和异质能力带来了重大的治理和问责挑战。本文提出了一种基于区块链的分层架构，用于监管代理协作，包括代理层、区块链数据层和监管应用层。在此框架内，我们设计了三个关键模块：（i）代理行为追踪和仲裁模块，以实现自动问责；（ii）动态声誉评估模块，以在协作场景中评估信任度；（iii）恶意行为预测模块，以早期检测敌对活动。本文方法为大规模代理生态系统中的可信赖、抗扰动和可扩展的监管机制奠定了系统性的基础。最后，我们讨论了面向多代理系统的基于区块链的监管框架的未来研究方向。 

---
# Instructional Prompt Optimization for Few-Shot LLM-Based Recommendations on Cold-Start Users 

**Title (ZH)**: 基于少样本大语言模型推荐体系中的冷启动用户指令提示优化 

**Authors**: Haowei Yang, Yushang Zhao, Sitao Min, Bo Su, Chao Yao, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09066)  

**Abstract**: The cold-start user issue further compromises the effectiveness of recommender systems in limiting access to the historical behavioral information. It is an effective pipeline to optimize instructional prompts on a few-shot large language model (LLM) used in recommender tasks. We introduce a context-conditioned prompt formulation method P(u,\ Ds)\ \rightarrow\ R\widehat, where u is a cold-start user profile, Ds is a curated support set, and R\widehat is the predicted ranked list of items. Based on systematic experimentation with transformer-based autoregressive LLMs (BioGPT, LLaMA-2, GPT-4), we provide empirical evidence that optimal exemplar injection and instruction structuring can significantly improve the precision@k and NDCG scores of such models in low-data settings. The pipeline uses token-level alignments and embedding space regularization with a greater semantic fidelity. Our findings not only show that timely composition is not merely syntactic but also functional as it is in direct control of attention scales and decoder conduct through inference. This paper shows that prompt-based adaptation may be considered one of the ways to address cold-start recommendation issues in LLM-based pipelines. 

**Abstract (ZH)**: 冷启动用户问题进一步削弱了推荐系统通过限制访问历史行为信息来提升效果的能力。基于少量样本的大语言模型（LLM）在推荐任务中的指令提示优化是一种有效的管道方法。我们提出了一种基于上下文的提示公式化方法P(u, Ds) → R̂，其中u是冷启动用户概要，Ds是精心构建的支持集，R̂是预测的项目 ranked 列表。基于对基于变换器的自回归大语言模型（BioGPT、LLaMA-2、GPT-4）的系统实验，我们提供了在数据量少的情况下，最佳示例注入和指令结构化可以显著提高这类模型的precision@k和NDCG分数的实证证据。该管道使用了基于token级别的对齐和嵌入空间正则化，具有更高的语义保真度。我们的发现不仅表明及时组合不仅仅是句法上的，而且也是功能性的，它可以直接影响注意尺度和解码器的行为。本文表明，基于提示的适应可能是解决基于大语言模型的管道中冷启动推荐问题的一种方式。 

---
# ForTIFAI: Fending Off Recursive Training Induced Failure for AI Models 

**Title (ZH)**: ForTIFAI: 避免由递归训练引发的AI模型故障 

**Authors**: Soheil Zibakhsh Shabgahi, Pedram Aghazadeh, Azalia Mirhosseini, Farinaz Koushanfar  

**Link**: [PDF](https://arxiv.org/pdf/2509.08972)  

**Abstract**: The increasing reliance on generative AI models has accelerated the generation rate of synthetic data, with some projections suggesting that most available new data for training could be machine-generated by 2030. This shift to a mainly synthetic content presents a critical challenge: repeated training in synthetic data leads to a phenomenon known as model collapse, where model performance degrades over generations of training, eventually rendering the models ineffective. Although prior studies have explored the causes and detection of model collapse, existing mitigation strategies remain limited.
In this paper, we identify model overconfidence in their self-generated data as a key driver of collapse. Building on this observation, we propose a confidence-aware loss function that downweights high-confidence predictions during training. We introduce a novel loss function we call Truncated Cross Entropy (TCE). We demonstrate that TCE significantly delays model collapse in recursive training.
We provide a model-agnostic framework that links the loss function design to model collapse mitigation and validate our approach both theoretically and empirically, showing that it can extend the model's fidelity interval before collapse by more than 2.3x. Finally, we show that our method generalizes across modalities. These findings suggest that the design of loss functions provides a simple yet powerful tool for preserving the quality of generative models in the era of increasing synthetic data. 

**Abstract (ZH)**: 不断增加对生成AI模型的依赖加速了合成数据的生成速度，有预测认为到2030年，大部分用于训练的新数据可能是机器生成的。这种主要由合成内容构成的转变提出了一项关键挑战：在合成数据上反复训练会导致模型退化现象——模型性能在训练代际中逐渐下降，最终使模型无效。尽管先前的研究已经探讨了模型退化的原因及其检测方法，现有的缓解策略仍然有限。

本文中，我们识别出模型在自生成数据上的高置信度作为模型退化的关键驱动因素。基于这一点观察，我们提出了一种注意置信度的loss函数，在训练过程中降低高置信度预测的权重。我们引入了一种新的loss函数，称为截断交叉熵（TCE）。我们证明，TCE可以显著延缓循环训练中的模型退化现象。

我们提供了一个模型无关的框架，将loss函数的设计与模型退化缓解联系起来，并从理论和实证两个方面验证了我们的方法，结果显示它可以使模型在退化前的质量间隔延长超过2.3倍。最后，我们表明我们的方法在不同模态上具有泛化能力。这些发现表明，loss函数设计提供了一个简单而强大的工具，用于在合成数据增加的时代保持生成模型的质量。 

---
# Global Constraint LLM Agents for Text-to-Model Translation 

**Title (ZH)**: 全球约束大语言模型代理用于文本到模型的翻译 

**Authors**: Junyang Cai, Serdar Kadioglu, Bistra Dilkina  

**Link**: [PDF](https://arxiv.org/pdf/2509.08970)  

**Abstract**: Natural language descriptions of optimization or satisfaction problems are challenging to translate into correct MiniZinc models, as this process demands both logical reasoning and constraint programming expertise. We introduce a framework that addresses this challenge with an agentic approach: multiple specialized large language model (LLM) agents decompose the modeling task by global constraint type. Each agent is dedicated to detecting and generating code for a specific class of global constraint, while a final assembler agent integrates these constraint snippets into a complete MiniZinc model. By dividing the problem into smaller, well-defined sub-tasks, each LLM handles a simpler reasoning challenge, potentially reducing overall complexity. We conduct initial experiments with several LLMs and show better performance against baselines such as one-shot prompting and chain-of-thought prompting. Finally, we outline a comprehensive roadmap for future work, highlighting potential enhancements and directions for improvement. 

**Abstract (ZH)**: 自然语言对优化或满意问题的描述转换为正确的MiniZinc模型具有挑战性，这要求同时具备逻辑推理能力和约束编程专业知识。我们提出了一种以代理为导向的框架：多个专门的大语言模型（LLM）代理通过全局约束类型分解建模任务。每个代理专注于检测并生成特定类别的全局约束代码，而最终的组装代理则将这些约束片段整合成一个完整的MiniZinc模型。通过将问题分解为更小的、定义清晰的子任务，每个LLM可以处理一个更简单的推理挑战，从而可能降低整体复杂性。我们使用几种LLM进行了初步实验，并展示了其相对于单次提示和链式思考提示等基线方法的更好性能。最后，我们概述了未来工作的一个综合路线图，指出了潜在的改进方向和增强方案。 

---
# ButterflyQuant: Ultra-low-bit LLM Quantization through Learnable Orthogonal Butterfly Transforms 

**Title (ZH)**: ButterflyQuant：可通过可学习正交蝴蝶变换实现的超低比特LLM量化 

**Authors**: Bingxin Xu, Zhen Dong, Oussama Elachqar, Yuzhang Shang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09679)  

**Abstract**: Large language models require massive memory footprints, severely limiting deployment on consumer hardware. Quantization reduces memory through lower numerical precision, but extreme 2-bit quantization suffers from catastrophic performance loss due to outliers in activations. Rotation-based methods such as QuIP and QuaRot apply orthogonal transforms to eliminate outliers before quantization, using computational invariance: $\mathbf{y} = \mathbf{Wx} = (\mathbf{WQ}^T)(\mathbf{Qx})$ for orthogonal $\mathbf{Q}$. However, these methods use fixed transforms--Hadamard matrices achieving optimal worst-case coherence $\mu = 1/\sqrt{n}$--that cannot adapt to specific weight distributions. We identify that different transformer layers exhibit distinct outlier patterns, motivating layer-adaptive rotations rather than one-size-fits-all approaches. We propose ButterflyQuant, which replaces Hadamard rotations with learnable butterfly transforms parameterized by continuous Givens rotation angles. Unlike Hadamard's discrete $\{+1, -1\}$ entries that are non-differentiable and prohibit gradient-based learning, butterfly transforms' continuous parameterization enables smooth optimization while guaranteeing orthogonality by construction. This orthogonal constraint ensures theoretical guarantees in outlier suppression while achieving $O(n \log n)$ computational complexity with only $\frac{n \log n}{2}$ learnable parameters. We further introduce a uniformity regularization on post-transformation activations to promote smoother distributions amenable to quantization. Learning requires only 128 calibration samples and converges in minutes on a single GPU--a negligible one-time cost. On LLaMA-2-7B with 2-bit quantization, ButterflyQuant achieves 15.4 perplexity versus 22.1 for QuaRot. 

**Abstract (ZH)**: 大型语言模型需要庞大的内存 footprint，严重限制了其在消费级硬件上的部署。通过降低数值精度来进行量化可以减少内存占用，但极端的2位量化由于激活值中的异常值会导致灾难性的性能下降。基于旋转的方法，如QuIP和QuaRot，在量化之前应用正交变换以消除异常值，利用计算不变性：$\mathbf{y} = \mathbf{Wx} = (\mathbf{WQ}^T)(\mathbf{Qx})$，其中$\mathbf{Q}$是正交矩阵。然而，这些方法使用固定的变换——最优最坏情况相干性$\mu = 1/\sqrt{n}$的哈达玛矩阵——不能适应特定的权重分布。我们发现不同变压器层表现出不同的异常值模式，这促使我们采用层自适应旋转而非一刀切的方法。我们提出了ButterflyQuant，用由连续的Givens旋转角度参数化的蝴蝶变换替换哈达玛旋转。与哈达玛矩阵只有$\{+1, -1\}$的离散条目不同，后者禁止基于梯度的学习，而蝴蝶变换的连续参数化允许平滑优化，并通过构造保证正交性。这种正交约束确保了异常值抑制的理论保证，同时通过仅使用$\frac{n \log n}{2}$个可学习参数实现了$O(n \log n)$的计算复杂度。我们还引入了一种变换后激活的均匀性正则化，以促进更适于量化的平滑分布。训练只需要128个校准样本，并在单个GPU上几分钟内收敛——这是一个可以忽略不计的一次性成本。在使用2位量化时，ButterflyQuant在LLaMA-2-7B上实现了15.4的困惑度，而QuaRot为22.1。 

---
# CDE: Curiosity-Driven Exploration for Efficient Reinforcement Learning in Large Language Models 

**Title (ZH)**: CDE: 好奇心驱动的探索在大型语言模型高效 reinforcement learning 中的应用 

**Authors**: Runpeng Dai, Linfeng Song, Haolin Liu, Zhenwen Liang, Dian Yu, Haitao Mi, Zhaopeng Tu, Rui Liu, Tong Zheng, Hongtu Zhu, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09675)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) is a powerful paradigm for enhancing the reasoning ability of Large Language Models (LLMs). Yet current RLVR methods often explore poorly, leading to premature convergence and entropy collapse. To address this challenge, we introduce Curiosity-Driven Exploration (CDE), a framework that leverages the model's own intrinsic sense of curiosity to guide exploration. We formalize curiosity with signals from both the actor and the critic: for the actor, we use perplexity over its generated response, and for the critic, we use the variance of value estimates from a multi-head architecture. Both signals serve as an exploration bonus within the RLVR framework to guide the model. Our theoretical analysis shows that the actor-wise bonus inherently penalizes overconfident errors and promotes diversity among correct responses; moreover, we connect the critic-wise bonus to the well-established count-based exploration bonus in RL. Empirically, our method achieves an approximate +3 point improvement over standard RLVR using GRPO/PPO on AIME benchmarks. Further analysis identifies a calibration collapse mechanism within RLVR, shedding light on common LLM failure modes. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）增强了大型语言模型的推理能力，但当前的RLVR方法往往探索不足，导致过早收敛和熵塌陷。为解决这一挑战，我们介绍了好奇心驱动探索（CDE）框架，该框架利用模型自身的内在好奇心来引导探索。我们通过来自行为者和评论者的信号形式化好奇心：对行为者，我们使用生成响应的困惑度；对评论者，我们使用多头架构中价值估计的方差。两种信号在RLVR框架内作为探索奖励来引导模型。我们的理论分析表明，行为者奖励本质上惩罚过度自信的错误，促进正确响应的多样性；此外，我们将评论者奖励与RL中成熟的计数基础探索奖励联系起来。实验上，我们的方法在AIME基准上使用GRPO/PPO实现了大约+3分的改进。进一步分析揭示了RLVR中的校准塌缩机制，阐明了常见的LLM失败模式。 

---
# LoCoBench: A Benchmark for Long-Context Large Language Models in Complex Software Engineering 

**Title (ZH)**: LoCoBench: 一种复杂软件工程中的长上下文大型语言模型基准测试 

**Authors**: Jielin Qiu, Zuxin Liu, Zhiwei Liu, Rithesh Murthy, Jianguo Zhang, Haolin Chen, Shiyu Wang, Ming Zhu, Liangwei Yang, Juntao Tan, Zhepeng Cen, Cheng Qian, Shelby Heinecke, Weiran Yao, Silvio Savarese, Caiming Xiong, Huan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09614)  

**Abstract**: The emergence of long-context language models with context windows extending to millions of tokens has created new opportunities for sophisticated code understanding and software development evaluation. We propose LoCoBench, a comprehensive benchmark specifically designed to evaluate long-context LLMs in realistic, complex software development scenarios. Unlike existing code evaluation benchmarks that focus on single-function completion or short-context tasks, LoCoBench addresses the critical evaluation gap for long-context capabilities that require understanding entire codebases, reasoning across multiple files, and maintaining architectural consistency across large-scale software systems. Our benchmark provides 8,000 evaluation scenarios systematically generated across 10 programming languages, with context lengths spanning 10K to 1M tokens, a 100x variation that enables precise assessment of long-context performance degradation in realistic software development settings. LoCoBench introduces 8 task categories that capture essential long-context capabilities: architectural understanding, cross-file refactoring, multi-session development, bug investigation, feature implementation, code comprehension, integration testing, and security analysis. Through a 5-phase pipeline, we create diverse, high-quality scenarios that challenge LLMs to reason about complex codebases at unprecedented scale. We introduce a comprehensive evaluation framework with 17 metrics across 4 dimensions, including 8 new evaluation metrics, combined in a LoCoBench Score (LCBS). Our evaluation of state-of-the-art long-context models reveals substantial performance gaps, demonstrating that long-context understanding in complex software development represents a significant unsolved challenge that demands more attention. LoCoBench is released at: this https URL. 

**Abstract (ZH)**: 长上下文语言模型的出现为复杂的代码理解和软件开发评估带来了新的机会。我们提出了LoCoBench，一个专门设计用于在实际复杂的软件开发场景中评估长上下文LLM的综合基准。与现有的主要关注单函数完成或短上下文任务的代码评估基准不同，LoCoBench 解决了需理解整个代码库、在多个文件间进行推理并维护大规模软件系统的架构一致性等长上下文能力的评估空白。该基准提供了涵盖10种编程语言的8000个系统生成的评估场景，上下文长度从10K到1M不等，跨度达100倍，适用于在现实软件开发环境中精确评估长上下文性能的退化。LoCoBench 引入了8个任务类别，涵盖了关键的长上下文能力：架构理解、跨文件重构、多会话开发、bug调查、功能实现、代码理解、集成测试和安全性分析。通过5阶段管道，我们创建了多样的高质量场景，挑战LLM以前所未有的规模推理复杂的代码库。我们引入了一个全面的评估框架，包括4个维度的17个指标，其中8个是新的评估指标，结合成LoCoBench得分（LCBS）。我们的评估表明，最先进的长上下文模型存在显著的性能差距，表明复杂的软件开发中的长上下文理解是一个重要的未解决挑战，需要更多关注。LoCoBench可在以下链接获取：this https URL。 

---
# Fluent but Unfeeling: The Emotional Blind Spots of Language Models 

**Title (ZH)**: 流畅而不 emotionally resonant：语言模型的情感盲点 

**Authors**: Bangzhao Shu, Isha Joshi, Melissa Karnaze, Anh C. Pham, Ishita Kakkar, Sindhu Kothe, Arpine Hovasapian, Mai ElSherief  

**Link**: [PDF](https://arxiv.org/pdf/2509.09593)  

**Abstract**: The versatility of Large Language Models (LLMs) in natural language understanding has made them increasingly popular in mental health research. While many studies explore LLMs' capabilities in emotion recognition, a critical gap remains in evaluating whether LLMs align with human emotions at a fine-grained level. Existing research typically focuses on classifying emotions into predefined, limited categories, overlooking more nuanced expressions. To address this gap, we introduce EXPRESS, a benchmark dataset curated from Reddit communities featuring 251 fine-grained, self-disclosed emotion labels. Our comprehensive evaluation framework examines predicted emotion terms and decomposes them into eight basic emotions using established emotion theories, enabling a fine-grained comparison. Systematic testing of prevalent LLMs under various prompt settings reveals that accurately predicting emotions that align with human self-disclosed emotions remains challenging. Qualitative analysis further shows that while certain LLMs generate emotion terms consistent with established emotion theories and definitions, they sometimes fail to capture contextual cues as effectively as human self-disclosures. These findings highlight the limitations of LLMs in fine-grained emotion alignment and offer insights for future research aimed at enhancing their contextual understanding. 

**Abstract (ZH)**: 大型语言模型在自然语言理解的多功能性使其在心理健康研究中日益受欢迎。尽管许多研究探讨了大型语言模型在情绪识别方面的能力，但在评估这些模型是否在细粒度层次上与人类情绪一致方面仍存在一个关键缺口。现有研究通常专注于将情绪分类为预定义的有限类别，而忽略了更细致的表现形式。为填补这一空白，我们介绍了EXPRESS，这是一个基准数据集，它来自 Reddit 社区，包含 251 个细粒度的自我披露情绪标签。我们的全面评估框架检查预测的情绪术语，并使用建立的情绪理论将它们分解为八种基本情绪，从而实现细粒度比较。在各种提示设置下对常见的大型语言模型进行系统性测试显示出，准确预测与人类自我披露情绪一致的情绪仍然是一个挑战。进一步的定性分析表明，虽然某些大型语言模型产生与建立的情绪理论和定义一致的情绪术语，但它们有时未能像人类自我披露那样有效地捕捉到上下文线索。这些发现揭示了大型语言模型在细粒度情绪对齐方面的局限性，并为未来旨在提高其上下文理解能力的研究提供了见解。 

---
# ENSI: Efficient Non-Interactive Secure Inference for Large Language Models 

**Title (ZH)**: ENSI: 高效非交互式安全推理大型语言模型 

**Authors**: Zhiyu He, Maojiang Wang, Xinwen Gao, Yuchuan Luo, Lin Liu, Shaojing Fu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09424)  

**Abstract**: Secure inference enables privacy-preserving machine learning by leveraging cryptographic protocols that support computations on sensitive user data without exposing it. However, integrating cryptographic protocols with large language models (LLMs) presents significant challenges, as the inherent complexity of these protocols, together with LLMs' massive parameter scale and sophisticated architectures, severely limits practical usability. In this work, we propose ENSI, a novel non-interactive secure inference framework for LLMs, based on the principle of co-designing the cryptographic protocols and LLM architecture. ENSI employs an optimized encoding strategy that seamlessly integrates CKKS scheme with a lightweight LLM variant, BitNet, significantly reducing the computational complexity of encrypted matrix multiplications. In response to the prohibitive computational demands of softmax under homomorphic encryption (HE), we pioneer the integration of the sigmoid attention mechanism with HE as a seamless, retraining-free alternative. Furthermore, by embedding the Bootstrapping operation within the RMSNorm process, we efficiently refresh ciphertexts while markedly decreasing the frequency of costly bootstrapping invocations. Experimental evaluations demonstrate that ENSI achieves approximately an 8x acceleration in matrix multiplications and a 2.6x speedup in softmax inference on CPU compared to state-of-the-art method, with the proportion of bootstrapping is reduced to just 1%. 

**Abstract (ZH)**: 安全推理使加密码协议能够在不暴露敏感用户数据的情况下对这些数据进行计算，从而实现隐私保护的机器学习。然而，将加密码协议集成到大型语言模型（LLMs）中面临重大挑战，因为这些协议的固有复杂性和LLMs的巨大参数规模及复杂的架构严重限制了其实用性。在这项工作中，我们提出了一种名为ENSI的新型非交互式安全推理框架，该框架基于加密码协议和LLM架构共同设计的原则。ENSI采用了优化的编码策略，无缝集成CKKS方案与轻量级LLM变体BitNet，显著降低加密矩阵乘法的计算复杂度。针对同态加密（HE）下softmax的巨额计算需求，我们首次提出了将Sigmoid注意力机制与HE无缝集成作为无重新训练的替代方案。此外，通过将Bootstrapping操作嵌入到RMSNorm过程中，我们高效地刷新密文，并显著减少了昂贵的Bootstrapping调用频率。实验评估表明，与现有最佳方法相比，ENSI在CPU上实现了大约8倍的矩阵乘法加速和2.6倍的softmax推理加速，Bootstrapping的比例降低至仅1%。 

---
# LLMs Don't Know Their Own Decision Boundaries: The Unreliability of Self-Generated Counterfactual Explanations 

**Title (ZH)**: LLMs 不知道自己的决策边界：自我生成的反事实解释不可靠 

**Authors**: Harry Mayne, Ryan Othniel Kearns, Yushi Yang, Andrew M. Bean, Eoin Delaney, Chris Russell, Adam Mahdi  

**Link**: [PDF](https://arxiv.org/pdf/2509.09396)  

**Abstract**: To collaborate effectively with humans, language models must be able to explain their decisions in natural language. We study a specific type of self-explanation: self-generated counterfactual explanations (SCEs), where a model explains its prediction by modifying the input such that it would have predicted a different outcome. We evaluate whether LLMs can produce SCEs that are valid, achieving the intended outcome, and minimal, modifying the input no more than necessary. When asked to generate counterfactuals, we find that LLMs typically produce SCEs that are valid, but far from minimal, offering little insight into their decision-making behaviour. Worryingly, when asked to generate minimal counterfactuals, LLMs typically make excessively small edits that fail to change predictions. The observed validity-minimality trade-off is consistent across several LLMs, datasets, and evaluation settings. Our findings suggest that SCEs are, at best, an ineffective explainability tool and, at worst, can provide misleading insights into model behaviour. Proposals to deploy LLMs in high-stakes settings must consider the impact of unreliable self-explanations on downstream decision-making. Our code is available at this https URL. 

**Abstract (ZH)**: 语言模型必须能够用自然语言解释其决策以有效协作。我们研究了一种特定类型的自解释：自生成反事实解释（SCEs），模型通过修改输入来解释其预测，使得它会预测不同的结果。我们评估了LLMs能否生成有效的、实现意图结果且最小化的SCEs。当我们要求生成反事实时，发现LLMs通常能产生有效的SCEs，但远不 Minimal，提供的决策过程洞察有限。令人担忧的是，当我们要求生成最小化的反事实时，LLMs通常做出过小的编辑，无法改变预测。观察到的有效性-最小性权衡贯穿于多个LLMs、数据集和评估设置中。我们的研究结果表明，SCEs至多是无效的解释工具，最坏情况下会提供误导性的模型行为洞察。在高风险场景下部署LLMs时，必须考虑不可靠自解释对下游决策的影响。我们的代码可在以下链接获取。 

---
# MetaLLMix : An XAI Aided LLM-Meta-learning Based Approach for Hyper-parameters Optimization 

**Title (ZH)**: MetaLLMix : 一种基于XAI辅助的LLM元学习方法用于超参数优化 

**Authors**: Mohammed Tiouti, Mohamed Bal-Ghaoui  

**Link**: [PDF](https://arxiv.org/pdf/2509.09387)  

**Abstract**: Effective model and hyperparameter selection remains a major challenge in deep learning, often requiring extensive expertise and computation. While AutoML and large language models (LLMs) promise automation, current LLM-based approaches rely on trial and error and expensive APIs, which provide limited interpretability and generalizability. We propose MetaLLMiX, a zero-shot hyperparameter optimization framework combining meta-learning, explainable AI, and efficient LLM reasoning. By leveraging historical experiment outcomes with SHAP explanations, MetaLLMiX recommends optimal hyperparameters and pretrained models without additional trials. We further employ an LLM-as-judge evaluation to control output format, accuracy, and completeness. Experiments on eight medical imaging datasets using nine open-source lightweight LLMs show that MetaLLMiX achieves competitive or superior performance to traditional HPO methods while drastically reducing computational cost. Our local deployment outperforms prior API-based approaches, achieving optimal results on 5 of 8 tasks, response time reductions of 99.6-99.9%, and the fastest training times on 6 datasets (2.4-15.7x faster), maintaining accuracy within 1-5% of best-performing baselines. 

**Abstract (ZH)**: 基于元学习、可解释AI和高效大语言模型推理的零样本超参数优化框架MetaLLMiX 

---
# On Integrating Large Language Models and Scenario-Based Programming for Improving Software Reliability 

**Title (ZH)**: 基于情景编程改进软件可靠性的大型语言模型融合研究 

**Authors**: Ayelet Berzack, Guy Katz  

**Link**: [PDF](https://arxiv.org/pdf/2509.09194)  

**Abstract**: Large Language Models (LLMs) are fast becoming indispensable tools for software developers, assisting or even partnering with them in crafting complex programs. The advantages are evident -- LLMs can significantly reduce development time, generate well-organized and comprehensible code, and occasionally suggest innovative ideas that developers might not conceive on their own. However, despite their strengths, LLMs will often introduce significant errors and present incorrect code with persuasive confidence, potentially misleading developers into accepting flawed solutions.
In order to bring LLMs into the software development cycle in a more reliable manner, we propose a methodology for combining them with ``traditional'' software engineering techniques in a structured way, with the goal of streamlining the development process, reducing errors, and enabling users to verify crucial program properties with increased confidence. Specifically, we focus on the Scenario-Based Programming (SBP) paradigm -- an event-driven, scenario-based approach for software engineering -- to allow human developers to pour their expert knowledge into the LLM, as well as to inspect and verify its outputs.
To evaluate our methodology, we conducted a significant case study, and used it to design and implement the Connect4 game. By combining LLMs and SBP we were able to create a highly-capable agent, which could defeat various strong existing agents. Further, in some cases, we were able to formally verify the correctness of our agent. Finally, our experience reveals interesting insights regarding the ease-of-use of our proposed approach. The full code of our case-study will be made publicly available with the final version of this paper. 

**Abstract (ZH)**: 大型语言模型（LLMs）正逐渐成为软件开发者不可或缺的工具，协助甚至与他们合作编制复杂程序。尽管具备显著优势——LLMs能够大幅缩短开发时间，生成有序且易于理解的代码，并偶尔提出开发人员难以自主构思的创新想法——但它们也经常引入重大错误，以说服性的自信呈现错误代码，可能导致误导开发人员接受有缺陷的解决方案。

为了以更可靠的方式将LLMs融入软件开发周期，我们提出了一种将它们与传统的软件工程技术结构化结合的方法，旨在简化开发过程，减少错误，并使用户能够以更高的信心验证程序的关键属性。具体而言，我们专注于基于场景的编程（Scenario-Based Programming，SBP）范式——一种事件驱动的、基于场景的软件工程方法——使人类开发人员能够将其专家知识注入LLMs，并对其输出进行检查和验证。

为了评估我们的方法，我们进行了一项重要的案例研究，并利用该研究设计并实现了Connect4游戏。通过结合LLMs和SBP，我们能够创建出能够击败多种现有强健代理的高级代理。此外，在某些情况下，我们能够形式化验证我们代理的正确性。最终，我们的经验揭示了有关我们提出方法易用性的一些有趣见解。我们在论文的最终版本中将发布我们案例研究的全部代码。 

---
# Probing Pre-trained Language Models on Code Changes: Insights from ReDef, a High-Confidence Just-in-Time Defect Prediction Dataset 

**Title (ZH)**: 探讨预训练语言模型在代码变更上的应用：来自ReDef高置信度即时缺陷预测数据集的见解 

**Authors**: Doha Nam, Taehyoun Kim, Duksan Ryu, Jongmoon Baik  

**Link**: [PDF](https://arxiv.org/pdf/2509.09192)  

**Abstract**: Just-in-Time software defect prediction (JIT-SDP) plays a critical role in prioritizing risky code changes during code review and continuous integration. However, existing datasets often suffer from noisy labels and low precision in identifying bug-inducing commits. To address this, we present ReDef (Revert-based Defect dataset), a high-confidence benchmark of function-level modifications curated from 22 large-scale C/C++ projects. Defective cases are anchored by revert commits, while clean cases are validated through post-hoc history checks. Ambiguous instances are conservatively filtered out via a GPT-assisted triage process involving multiple votes and audits. This pipeline yields 3,164 defective and 10,268 clean modifications, offering substantially more reliable labels than prior existing resources. Beyond dataset construction, we provide the first systematic evaluation of how pre-trained language models (PLMs) reason about code modifications -- specifically, which input encodings most effectively expose change information, and whether models genuinely capture edit semantics. We fine-tune CodeBERT, CodeT5+, and UniXcoder under five encoding strategies, and further probe their sensitivity through counterfactual perturbations that swap added/deleted blocks, invert diff polarity, or inject spurious markers. Our results show that compact diff-style encodings consistently outperform whole-function formats across all PLMs, with statistical tests confirming large, model-independent effects. However, under counterfactual tests, performance degrades little or not at all -- revealing that what appears to be robustness in fact reflects reliance on superficial cues rather than true semantic understanding. These findings indicate that, unlike in snapshot-based tasks, current PLMs remain limited in their ability to genuinely comprehend code modifications. 

**Abstract (ZH)**: 基于回退的缺陷数据集（ReDef）：一种高置信度的功能修改基准 

---
# EchoX: Towards Mitigating Acoustic-Semantic Gap via Echo Training for Speech-to-Speech LLMs 

**Title (ZH)**: EchoX：通过回声训练减轻语音-语义差距以提高语音到语音的大语言模型性能 

**Authors**: Yuhao Zhang, Yuhao Du, Zhanchen Dai, Xiangnan Ma, Kaiqi Kou, Benyou Wang, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.09174)  

**Abstract**: Speech-to-speech large language models (SLLMs) are attracting increasing attention. Derived from text-based large language models (LLMs), SLLMs often exhibit degradation in knowledge and reasoning capabilities. We hypothesize that this limitation arises because current training paradigms for SLLMs fail to bridge the acoustic-semantic gap in the feature representation space. To address this issue, we propose EchoX, which leverages semantic representations and dynamically generates speech training targets. This approach integrates both acoustic and semantic learning, enabling EchoX to preserve strong reasoning abilities as a speech LLM. Experimental results demonstrate that EchoX, with about six thousand hours of training data, achieves advanced performance on multiple knowledge-based question-answering benchmarks. The project is available at this https URL. 

**Abstract (ZH)**: 基于语音的大语言模型（SLLMs）正吸引越来越多的关注。源于文本的大语言模型（LLMs），SLLMs通常在知识和推理能力上表现出下降。我们假设这一限制源于当前SLLMs的训练范式未能在特征表示空间中跨越声学语义差距。为解决这一问题，我们提出了EchoX，它利用语义表示并动态生成语音训练目标。该方法结合了声学和语义学习，使EchoX能够保留强烈的推理能力作为语音大语言模型。实验结果表明，使用大约六千小时的训练数据，EchoX在多个基于知识的问答基准测试中取得了先进性能。项目详情请见此链接。 

---
# Character-Level Perturbations Disrupt LLM Watermarks 

**Title (ZH)**: 字符级扰动破坏LLM水印 

**Authors**: Zhaoxi Zhang, Xiaomei Zhang, Yanjun Zhang, He Zhang, Shirui Pan, Bo Liu, Asif Qumer Gill, Leo Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09112)  

**Abstract**: Large Language Model (LLM) watermarking embeds detectable signals into generated text for copyright protection, misuse prevention, and content detection. While prior studies evaluate robustness using watermark removal attacks, these methods are often suboptimal, creating the misconception that effective removal requires large perturbations or powerful adversaries.
To bridge the gap, we first formalize the system model for LLM watermark, and characterize two realistic threat models constrained on limited access to the watermark detector. We then analyze how different types of perturbation vary in their attack range, i.e., the number of tokens they can affect with a single edit. We observe that character-level perturbations (e.g., typos, swaps, deletions, homoglyphs) can influence multiple tokens simultaneously by disrupting the tokenization process. We demonstrate that character-level perturbations are significantly more effective for watermark removal under the most restrictive threat model. We further propose guided removal attacks based on the Genetic Algorithm (GA) that uses a reference detector for optimization. Under a practical threat model with limited black-box queries to the watermark detector, our method demonstrates strong removal performance. Experiments confirm the superiority of character-level perturbations and the effectiveness of the GA in removing watermarks under realistic constraints. Additionally, we argue there is an adversarial dilemma when considering potential defenses: any fixed defense can be bypassed by a suitable perturbation strategy. Motivated by this principle, we propose an adaptive compound character-level attack. Experimental results show that this approach can effectively defeat the defenses. Our findings highlight significant vulnerabilities in existing LLM watermark schemes and underline the urgency for the development of new robust mechanisms. 

**Abstract (ZH)**: 大型语言模型（LLM）水印嵌入可检测信号以实现版权保护、滥用预防和内容检测。尽管先前的研究通过水印移除攻击评估鲁棒性，但这些方法往往不尽如人意，造成一种误解，即有效移除需要大量扰动或强大对手。

为此，我们首先形式化LLM水印系统模型，并在受限于有限访问水印检测器的情况下刻画两种现实威胁模型。随后，我们分析不同类型扰动的攻击范围，即单次编辑所能影响的标记数量。我们发现，字符级扰动（例如，拼写错误、交换、删除、谐音字符）可以通过干扰分词过程同时影响多个标记。我们证明，在最严格的威胁模型下，字符级扰动在水印移除方面表现出显著效果。我们进一步提出基于遗传算法（GA）的引导式移除攻击，该方法使用参考检测器进行优化。在对水印检测器的黑盒查询受限的现实威胁模型下，我们的方法展示了强大的移除性能。实验验证了字符级扰动的优势以及遗传算法在现实约束下移除水印的有效性。此外，我们认为在考虑潜在防御措施时存在一种对手困境：任何固定防御都可以通过适当扰动策略被绕过。基于这一原则，我们提出了适应性复合字符级攻击。实验结果表明，该方法能够有效破解防御。我们的发现突显了现有LLM水印方案中的重大漏洞，并强调了开发新的鲁棒机制的紧迫性。 

---
# DP-FedLoRA: Privacy-Enhanced Federated Fine-Tuning for On-Device Large Language Models 

**Title (ZH)**: DP-FedLoRA：增强隐私的联邦微调用于设备上大型语言模型 

**Authors**: Honghui Xu, Shiva Shrestha, Wei Chen, Zhiyuan Li, Zhipeng Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.09097)  

**Abstract**: As on-device large language model (LLM) systems become increasingly prevalent, federated fine-tuning enables advanced language understanding and generation directly on edge devices; however, it also involves processing sensitive, user-specific data, raising significant privacy concerns within the federated learning framework. To address these challenges, we propose DP-FedLoRA, a privacy-enhanced federated fine-tuning framework that integrates LoRA-based adaptation with differential privacy in a communication-efficient setting. Each client locally clips and perturbs its LoRA matrices using Gaussian noise to satisfy ($\epsilon$, $\delta$)-differential privacy. We further provide a theoretical analysis demonstrating the unbiased nature of the updates and deriving bounds on the variance introduced by noise, offering practical guidance for privacy-budget calibration. Experimental results across mainstream benchmarks show that DP-FedLoRA delivers competitive performance while offering strong privacy guarantees, paving the way for scalable and privacy-preserving LLM deployment in on-device environments. 

**Abstract (ZH)**: 随着设备端大型语言模型系统日益普及，边缘设备上的联邦微调能够直接实现高级语言理解和生成，但也涉及处理敏感的用户特定数据，引发联邦学习框架内的重大隐私问题。为应对这些挑战，我们提出DP-FedLoRA，一种结合LoRA基适应与差分隐私的隐私增强型联邦微调框架，能够在通信高效设置中运行。每个客户端使用高斯噪声本地裁剪和扰动其LoRA矩阵以满足($\epsilon$, $\delta$)-差分隐私。我们还提供了理论分析，证明了更新的无偏性，并得出了由噪声引入的方差上界，为隐私预算校准提供了实用指导。在主流基准上的实验结果表明，DP-FedLoRA在提供强隐私保障的同时实现了竞争力的表现，为设备端环境下可扩展和隐私保护的语言模型部署铺平了道路。 

---
# Towards Confidential and Efficient LLM Inference with Dual Privacy Protection 

**Title (ZH)**: 面向双重隐私保护的保密高效大语言模型推理 

**Authors**: Honglan Yu, Yibin Wang, Feifei Dai, Dong Liu, Haihui Fan, Xiaoyan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09091)  

**Abstract**: CPU-based trusted execution environments (TEEs) and differential privacy (DP) have gained wide applications for private inference. Due to high inference latency in TEEs, researchers use partition-based approaches that offload linear model components to GPUs. However, dense nonlinear layers of large language models (LLMs) result in significant communication overhead between TEEs and GPUs. DP-based approaches apply random noise to protect data privacy, but this compromises LLM performance and semantic understanding. To overcome the above drawbacks, this paper proposes CMIF, a Confidential and efficient Model Inference Framework. CMIF confidentially deploys the embedding layer in the client-side TEE and subsequent layers on GPU servers. Meanwhile, it optimizes the Report-Noisy-Max mechanism to protect sensitive inputs with a slight decrease in model performance. Extensive experiments on Llama-series models demonstrate that CMIF reduces additional inference overhead in TEEs while preserving user data privacy. 

**Abstract (ZH)**: 基于CPU的受信执行环境（TEEs）和差分隐私（DP）在私有推理中获得了广泛应用。尽管TEEs中的高推理延迟促使研究人员采用基于分区的方法将线性模型组件卸载到GPU上，但大型语言模型（LLMs）的密集非线性层导致了TEEs与GPU之间显著的通信开销。基于DP的方法通过添加随机噪声来保护数据隐私，但这会影响LLM的性能和语义理解。为克服上述缺点，本文提出了一种名为CMIF的保密高效模型推理框架。CMIF在客户端TEEs中 confidential地部署嵌入层，并将后续层部署在GPU服务器上。同时，它优化了Report-Noisy-Max机制以保护敏感输入，并且仅轻微降低模型性能。在Llama系列模型上的广泛实验表明，CMIF可以减少TEEs中的额外推理开销，同时保持用户数据隐私。 

---
# Improving LLM Safety and Helpfulness using SFT and DPO: A Study on OPT-350M 

**Title (ZH)**: 使用SFT和DPO提升LLM的安全性和有用性：基于OPT-350M的研究 

**Authors**: Piyush Pant  

**Link**: [PDF](https://arxiv.org/pdf/2509.09055)  

**Abstract**: This research investigates the effectiveness of alignment techniques, Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and a combined SFT+DPO approach on improving the safety and helpfulness of the OPT-350M language model. Utilizing the Anthropic Helpful-Harmless RLHF dataset, we train and evaluate four models: the base OPT350M, an SFT model, a DPO model, and a model trained with both SFT and DPO. We introduce three key evaluation metrics: Harmlessness Rate (HmR), Helpfulness Rate (HpR), and a Combined Alignment Score (CAS), all derived from reward model outputs. The results show that while SFT outperforms DPO, The combined SFT+DPO model outperforms all others across all metrics, demonstrating the complementary nature of these techniques. Our findings also highlight challenges posed by noisy data, limited GPU resources, and training constraints. This study offers a comprehensive view of how fine-tuning strategies affect model alignment and provides a foundation for more robust alignment pipelines in future work. 

**Abstract (ZH)**: 本研究探讨了对OPT-350M语言模型进行对齐技术（包括监督微调SFT、直接偏好优化DPO以及SFT+DPO结合方法）的有效性，以提高模型的安全性和帮助性。利用Anthropic Helpful-Harmless RLHF数据集，我们训练并评估了四个模型：基础OPT350M、SFT模型、DPO模型以及结合了SFT和DPO的模型。我们引入了三个关键评估指标：无害率（HmR）、帮助率（HpR）以及综合对齐得分（CAS），这些指标均源自奖励模型的输出。结果显示，虽然SFT的性能优于DPO，但结合了SFT和DPO的方法在所有指标上均表现出色，证明了这些技术的互补性。此外，研究还揭示了噪声数据、有限GPU资源和训练约束带来的挑战。本研究为未来更 robust 的对齐流水线提供了全面的视角，并提供了细调策略影响模型对齐的见解。 

---
# Stated Preference for Interaction and Continued Engagement (SPICE): Evaluating an LLM's Willingness to Re-engage in Conversation 

**Title (ZH)**: 基于陈述偏好的互动与持续参与(SPICE): 评估LLM重新参与对话的意愿 

**Authors**: Thomas Manuel Rost, Martina Figlia, Bernd Wallraff  

**Link**: [PDF](https://arxiv.org/pdf/2509.09043)  

**Abstract**: We introduce and evaluate Stated Preference for Interaction and Continued Engagement (SPICE), a simple diagnostic signal elicited by asking a Large Language Model a YES or NO question about its willingness to re-engage with a user's behavior after reviewing a short transcript. In a study using a 3-tone (friendly, unclear, abusive) by 10-interaction stimulus set, we tested four open-weight chat models across four framing conditions, resulting in 480 trials. Our findings show that SPICE sharply discriminates by user tone. Friendly interactions yielded a near-unanimous preference to continue (97.5% YES), while abusive interactions yielded a strong preference to discontinue (17.9% YES), with unclear interactions falling in between (60.4% YES). This core association remains decisive under multiple dependence-aware statistical tests, including Rao-Scott adjustment and cluster permutation tests. Furthermore, we demonstrate that SPICE provides a distinct signal from abuse classification. In trials where a model failed to identify abuse, it still overwhelmingly stated a preference not to continue the interaction (81% of the time). An exploratory analysis also reveals a significant interaction effect: a preamble describing the study context significantly impacts SPICE under ambiguity, but only when transcripts are presented as a single block of text rather than a multi-turn chat. The results validate SPICE as a robust, low-overhead, and reproducible tool for auditing model dispositions, complementing existing metrics by offering a direct, relational signal of a model's state. All stimuli, code, and analysis scripts are released to support replication. 

**Abstract (ZH)**: 我们介绍了并评估了交互与持续参与的明示偏好（SPICE）诊断信号，该信号通过向大型语言模型提出关于其在审阅简短对话记录后愿意重新参与用户行为的问题来获取其是或否的回答。在使用3种语气（友好、模糊、辱骂）和10次互动的刺激集合下，我们测试了四种未加权的聊天模型在四种框架条件下的表现，共计进行了480次试验。研究发现，SPICE能够敏锐地区分用户语气。友好互动几乎一致地显示出继续互动的偏好（97.5% 是），而辱骂互动则强烈倾向于终止互动（17.9% 是），模糊互动介于两者之间（60.4% 是）。该核心关联在多种依赖性校正统计检验（包括Rao-Scott调整和集群置换检验）下仍然具有决定性。此外，我们证明了SPICE提供了不同于辱骂分类的独特信号。在模型未能识别辱骂的情况下，它仍强烈表示不愿意继续互动（81% 的时间）。探索性分析还揭示了一个交互效应：描述研究背景的前言对模糊情况下SPICE的影响，但仅当对话以单一文本块形式呈现而非多轮对话形式呈现时才有效。结果验证了SPICE作为一个稳健、低开销且可重复使用的工具，用于审查模型态度的有效性，并通过提供一个直接且关系化的模型状态信号，补充了现有指标。所有刺激、代码和分析脚本均已发布以支持复制。 

---
# Open-sci-ref-0.01: open and reproducible reference baselines for language model and dataset comparison 

**Title (ZH)**: Open-sci-ref-0.01：语言模型和数据集比较的开放可重现参考基准 

**Authors**: Marianna Nezhurina, Taishi Nakamura, Timur Carstensen, Niccolò Ajroldi, Ville Komulainen, David Salinas, Jenia Jitsev  

**Link**: [PDF](https://arxiv.org/pdf/2509.09009)  

**Abstract**: We introduce open-sci-ref, a family of dense transformer models trained as research baselines across multiple model (0.13B to 1.7B parameters) and token scales (up to 1T) on 8 recent open reference datasets. Evaluating the models on various standardized benchmarks, our training runs set establishes reference points that enable researchers to assess the sanity and quality of alternative training approaches across scales and datasets. Intermediate checkpoints allow comparison and studying of the training dynamics. The established reference baselines allow training procedures to be compared through their scaling trends, aligning them on a common compute axis. Comparison of open reference datasets reveals that training on NemoTron-CC HQ consistently outperforms other reference datasets, followed by DCLM-baseline and FineWeb-Edu. In addition to intermediate training checkpoints, the release includes logs, code, and downstream evaluations to simplify reproduction, standardize comparison, and facilitate future research. 

**Abstract (ZH)**: 我们引入了open-sci-ref，这是一个基于多个参数规模（0.13B至1.7B）和标记规模（至1T）的密集变压器模型家族，这些模型在8个近期开源参考数据集中训练，作为研究基准。通过在各种标准化基准上评估这些模型，我们的训练运行设置确定了参考点，便于研究人员评估不同规模和数据集上的替代训练方法的质量。中间检查点允许进行训练动态的比较和研究。建立的标准基准线使得可以通过缩放趋势对比训练流程，将其统一到一个共同的计算轴上。对比开源参考数据集表明，使用NemoTron-CC HQ进行训练始终表现最佳，其次是DCLM-baseline和FineWeb-Edu。除了中间训练检查点外，发布内容还包括日志、代码和下游评估，以简化复制、标准化比较并促进未来的研究。 

---
# PromptGuard: An Orchestrated Prompting Framework for Principled Synthetic Text Generation for Vulnerable Populations using LLMs with Enhanced Safety, Fairness, and Controllability 

**Title (ZH)**: PromptGuard：一种用于脆弱人群的原理性合成文本生成的LLM关联提示框架，增强安全性、公平性和可控性 

**Authors**: Tung Vu, Lam Nguyen, Quynh Dao  

**Link**: [PDF](https://arxiv.org/pdf/2509.08910)  

**Abstract**: The proliferation of Large Language Models (LLMs) in real-world applications poses unprecedented risks of generating harmful, biased, or misleading information to vulnerable populations including LGBTQ+ individuals, single parents, and marginalized communities. While existing safety approaches rely on post-hoc filtering or generic alignment techniques, they fail to proactively prevent harmful outputs at the generation source. This paper introduces PromptGuard, a novel modular prompting framework with our breakthrough contribution: VulnGuard Prompt, a hybrid technique that prevents harmful information generation using real-world data-driven contrastive learning. VulnGuard integrates few-shot examples from curated GitHub repositories, ethical chain-of-thought reasoning, and adaptive role-prompting to create population-specific protective barriers. Our framework employs theoretical multi-objective optimization with formal proofs demonstrating 25-30% analytical harm reduction through entropy bounds and Pareto optimality. PromptGuard orchestrates six core modules: Input Classification, VulnGuard Prompting, Ethical Principles Integration, External Tool Interaction, Output Validation, and User-System Interaction, creating an intelligent expert system for real-time harm prevention. We provide comprehensive mathematical formalization including convergence proofs, vulnerability analysis using information theory, and theoretical validation framework using GitHub-sourced datasets, establishing mathematical foundations for systematic empirical research. 

**Abstract (ZH)**: 大型语言模型在实际应用中的 proliferate 对 LGBTQ+ 个体、单亲父母和边缘化社区等脆弱群体产生了前所未有的风险，可能导致有害、偏见或误导性信息的生成。现有安全方法主要依赖于事后过滤或通用对齐技术，未能在生成源头上主动预防有害输出。本文介绍了 PromptGuard，一种新颖的模块化提示框架，其中我们的突破性贡献是 VulnGuard Prompt，这是一种结合现实数据驱动对比学习的混合技术，用于防止有害信息的生成。VulnGuard 结合了精简示例、伦理链式思维推理和自适应角色提示，创建针对特定人群的保护屏障。该框架采用理论多目标优化，并通过熵界和帕累托最优性形式证明了 25-30% 的分析性危害减少。PromptGuard 协调六个核心模块：输入分类、VulnGuard 提示、伦理原则集成、外部工具交互、输出验证和用户-系统交互，构建了一个智能化专家系统以实现实时危害预防。我们提供了全面的数学形式化，包括收敛性证明、信息理论下的脆弱性分析和基于 GitHub 数据集的理论验证框架，为系统性实证研究奠定了数学基础。 

---
# Benchmarking Energy Efficiency of Large Language Models Using vLLM 

**Title (ZH)**: 使用vLLMbenchmark大型语言模型的能效 

**Authors**: K. Pronk, Q. Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.08867)  

**Abstract**: The prevalence of Large Language Models (LLMs) is having an growing impact on the climate due to the substantial energy required for their deployment and use. To create awareness for developers who are implementing LLMs in their products, there is a strong need to collect more information about the energy efficiency of LLMs. While existing research has evaluated the energy efficiency of various models, these benchmarks often fall short of representing realistic production scenarios. In this paper, we introduce the LLM Efficiency Benchmark, designed to simulate real-world usage conditions. Our benchmark utilizes vLLM, a high-throughput, production-ready LLM serving backend that optimizes model performance and efficiency. We examine how factors such as model size, architecture, and concurrent request volume affect inference energy efficiency. Our findings demonstrate that it is possible to create energy efficiency benchmarks that better reflect practical deployment conditions, providing valuable insights for developers aiming to build more sustainable AI systems. 

**Abstract (ZH)**: 大型语言模型的普及对气候造成了日益增长的影响，这归因于它们的部署和使用所需的巨大能源。为了提高正在将大型语言模型应用于其产品中的开发者的意识，收集更多关于大型语言模型能效的信息变得尤为重要。虽然现有研究已经评估了各种模型的能效，但这些基准通常无法充分代表真实的生产场景。在本文中，我们介绍了大型语言模型能效基准，旨在模拟实际使用条件。我们的基准利用了vLLM，这是一种高性能、生产级的大型语言模型服务后端，能够优化模型性能和能效。我们研究了模型大小、架构以及并发请求量等因素如何影响推理能耗能效。我们的研究结果表明，有可能创建更能反映实际部署条件的能效基准，为致力于构建更可持续的人工智能系统的开发者提供了宝贵见解。 

---
# Investigating Student Interaction Patterns with Large Language Model-Powered Course Assistants in Computer Science Courses 

**Title (ZH)**: 探究计算机科学课程中大型语言模型驱动课程助手的学生互动模式 

**Authors**: Chang Liu, Loc Hoang, Andrew Stolman, Rene F. Kizilcec, Bo Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.08862)  

**Abstract**: Providing students with flexible and timely academic support is a challenge at most colleges and universities, leaving many students without help outside scheduled hours. Large language models (LLMs) are promising for bridging this gap, but interactions between students and LLMs are rarely overseen by educators. We developed and studied an LLM-powered course assistant deployed across multiple computer science courses to characterize real-world use and understand pedagogical implications. By Spring 2024, our system had been deployed to approximately 2,000 students across six courses at three institutions. Analysis of the interaction data shows that usage remains strong in the evenings and nights and is higher in introductory courses, indicating that our system helps address temporal support gaps and novice learner needs. We sampled 200 conversations per course for manual annotation: most sampled responses were judged correct and helpful, with a small share unhelpful or erroneous; few responses included dedicated examples. We also examined an inquiry-based learning strategy: only around 11% of sampled conversations contained LLM-generated follow-up questions, which were often ignored by students in advanced courses. A Bloom's taxonomy analysis reveals that current LLM capabilities are limited in generating higher-order cognitive questions. These patterns suggest opportunities for pedagogically oriented LLM-based educational systems and greater educator involvement in configuring prompts, content, and policies. 

**Abstract (ZH)**: 为学生提供灵活及时的学术支持是大多数学院和大学面临的挑战，许多学生在非预约时段缺乏帮助。大规模语言模型（LLMs）有望弥补这一差距，但学生与LLMs的互动很少受到教育者的监管。我们开发并研究了部署在多门计算机科学课程中的LLM辅助系统，以了解实际应用情况及其教学意义。截至2024年春季，该系统已应用于三所机构的六门课程中的约2000名学生。交互数据的分析显示，使用在傍晚和夜间依然强劲，且在入门课程中的使用率更高，表明该系统有助于弥补时间上的支持缺口，满足初学者的学习需求。我们每门课程手工标注了200次对话：大多数样本回答被认为是正确和有帮助的，但有一小部分回答不相关或错误；很少有回答包含专门的例子。此外，我们还研究了一种基于问题的学习策略：只有约11%的样本对话中包含LLM生成的后续问题，而在高级课程中，学生往往忽略了这些问题。布卢姆分类学分析表明，当前的LLM生成高级认知问题的能力有限。这些模式表明了基于教学目标的LLM教育系统的潜在机会，以及教育者参与配置提示、内容和政策的重要性。 

---
# PerFairX: Is There a Balance Between Fairness and Personality in Large Language Model Recommendations? 

**Title (ZH)**: PerFairX：大型语言模型推荐中公平性和个性之间的平衡是否存在？ 

**Authors**: Chandan Kumar Sah  

**Link**: [PDF](https://arxiv.org/pdf/2509.08829)  

**Abstract**: The integration of Large Language Models (LLMs) into recommender systems has enabled zero-shot, personality-based personalization through prompt-based interactions, offering a new paradigm for user-centric recommendations. However, incorporating user personality traits via the OCEAN model highlights a critical tension between achieving psychological alignment and ensuring demographic fairness. To address this, we propose PerFairX, a unified evaluation framework designed to quantify the trade-offs between personalization and demographic equity in LLM-generated recommendations. Using neutral and personality-sensitive prompts across diverse user profiles, we benchmark two state-of-the-art LLMs, ChatGPT and DeepSeek, on movie (MovieLens 10M) and music (this http URL 360K) datasets. Our results reveal that personality-aware prompting significantly improves alignment with individual traits but can exacerbate fairness disparities across demographic groups. Specifically, DeepSeek achieves stronger psychological fit but exhibits higher sensitivity to prompt variations, while ChatGPT delivers stable yet less personalized outputs. PerFairX provides a principled benchmark to guide the development of LLM-based recommender systems that are both equitable and psychologically informed, contributing to the creation of inclusive, user-centric AI applications in continual learning contexts. 

**Abstract (ZH)**: 大型语言模型（LLMs）融入推荐系统实现了基于提示的零-shot个性化推荐，提供了一种以用户为中心的新范式。然而，通过OCEAN模型纳入用户个性特征凸显了实现心理对齐和确保人口统计公平性之间的关键紧张关系。为解决这一问题，我们提出PerFairX，一个统一的评估框架，用于量化LLM生成推荐中的个性化与人口统计公平性之间的权衡。我们使用中性和个性敏感的提示，在多样化的用户配置文件上对标记为最先进的LLM模型ChatGPT和DeepSeek在电影（MovieLens 10M）和音乐（this http URL 360K）数据集上进行基准测试。结果显示，个性感知的提示显著改善了与个体特质的一致性，但可能加剧不同人口统计群体之间的公平性差异。具体来说，DeepSeek实现了更强的心理契合度，但对提示变化更为敏感，而ChatGPT则提供了稳定但个性化程度较低的输出。PerFairX提供了一个原则性的基准，指导开发既公平又受心理启发的LLM推荐系统，有助于在持续学习环境中创建包容性和用户中心的AI应用。 

---
