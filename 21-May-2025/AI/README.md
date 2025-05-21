# Two Experts Are All You Need for Steering Thinking: Reinforcing Cognitive Effort in MoE Reasoning Models Without Additional Training 

**Title (ZH)**: 仅需两位专家即可引导思考：在MoE推理模型中强化认知努力，无需额外训练 

**Authors**: Mengru Wang, Xingyu Chen, Yue Wang, Zhiwei He, Jiahao Xu, Tian Liang, Qiuzhi Liu, Yunzhi Yao, Wenxuan Wang, Ruotian Ma, Haitao Mi, Ningyu Zhang, Zhaopeng Tu, Xiaolong Li, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14681)  

**Abstract**: Mixture-of-Experts (MoE) architectures within Large Reasoning Models (LRMs) have achieved impressive reasoning capabilities by selectively activating experts to facilitate structured cognitive processes. Despite notable advances, existing reasoning models often suffer from cognitive inefficiencies like overthinking and underthinking. To address these limitations, we introduce a novel inference-time steering methodology called Reinforcing Cognitive Experts (RICE), designed to improve reasoning performance without additional training or complex heuristics. Leveraging normalized Pointwise Mutual Information (nPMI), we systematically identify specialized experts, termed ''cognitive experts'' that orchestrate meta-level reasoning operations characterized by tokens like ''<think>''. Empirical evaluations with leading MoE-based LRMs (DeepSeek-R1 and Qwen3-235B) on rigorous quantitative and scientific reasoning benchmarks demonstrate noticeable and consistent improvements in reasoning accuracy, cognitive efficiency, and cross-domain generalization. Crucially, our lightweight approach substantially outperforms prevalent reasoning-steering techniques, such as prompt design and decoding constraints, while preserving the model's general instruction-following skills. These results highlight reinforcing cognitive experts as a promising, practical, and interpretable direction to enhance cognitive efficiency within advanced reasoning models. 

**Abstract (ZH)**: 大型推理模型（LRMs）中的专家混合（MoE）架构通过选择性激活专家以促进结构化的认知过程，实现了令人印象深刻的推理能力。尽管取得了显著进展，现有推理模型往往仍存在过度推理和欠推理的认知效率问题。为解决这些局限，我们提出了一种名为Reinforcing Cognitive Experts（RICE）的新颖推理时控制方法，旨在在无需额外训练或复杂启发式方法的情况下改进推理性能。通过利用归一化的点互信息（nPMI），我们系统地识别出专门化的专家，称为“认知专家”，它们协调以“<think>”等标记为特征的元级推理操作。在严格的定量和科学推理基准测试中，与主流的MoE基LRMs（DeepSeek-R1和Qwen3-235B）进行的实证评估表明，在推理准确性、认知效率和跨领域泛化方面均取得了显著且一致的改进。尤为重要的是，我们的轻量级方法在推理控制技术（如提示设计和解码约束）中表现优异，同时保留了模型的一般指令遵循能力。这些结果突出了加强认知专家作为提高先进推理模型中认知效率的有前景、实用和可解释方向。 

---
# ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions 

**Title (ZH)**: ContextAgent: 具有开放世界感知能力的上下文感知主动大语言模型代理 

**Authors**: Bufang Yang, Lilin Xu, Liekang Zeng, Kaiwei Liu, Siyang Jiang, Wenrui Lu, Hongkai Chen, Xiaofan Jiang, Guoliang Xing, Zhenyu Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.14668)  

**Abstract**: Recent advances in Large Language Models (LLMs) have propelled intelligent agents from reactive responses to proactive support. While promising, existing proactive agents either rely exclusively on observations from enclosed environments (e.g., desktop UIs) with direct LLM inference or employ rule-based proactive notifications, leading to suboptimal user intent understanding and limited functionality for proactive service. In this paper, we introduce ContextAgent, the first context-aware proactive agent that incorporates extensive sensory contexts to enhance the proactive capabilities of LLM agents. ContextAgent first extracts multi-dimensional contexts from massive sensory perceptions on wearables (e.g., video and audio) to understand user intentions. ContextAgent then leverages the sensory contexts and the persona contexts from historical data to predict the necessity for proactive services. When proactive assistance is needed, ContextAgent further automatically calls the necessary tools to assist users unobtrusively. To evaluate this new task, we curate ContextAgentBench, the first benchmark for evaluating context-aware proactive LLM agents, covering 1,000 samples across nine daily scenarios and twenty tools. Experiments on ContextAgentBench show that ContextAgent outperforms baselines by achieving up to 8.5% and 6.0% higher accuracy in proactive predictions and tool calling, respectively. We hope our research can inspire the development of more advanced, human-centric, proactive AI assistants. 

**Abstract (ZH)**: Recent Advances in Large Language Models (LLMs) Have Transformed Intelligent Agents from Reactive Responses to Proactive Support: Introducing ContextAgent, the First Context-Aware Proactive Agent 

---
# SAFEPATH: Preventing Harmful Reasoning in Chain-of-Thought via Early Alignment 

**Title (ZH)**: SAFEPATH: 在早期对齐防止链式推理中的有害 reasoning 

**Authors**: Wonje Jeung, Sangyeon Yoon, Minsuk Kahng, Albert No  

**Link**: [PDF](https://arxiv.org/pdf/2505.14667)  

**Abstract**: Large Reasoning Models (LRMs) have become powerful tools for complex problem solving, but their structured reasoning pathways can lead to unsafe outputs when exposed to harmful prompts. Existing safety alignment methods reduce harmful outputs but can degrade reasoning depth, leading to significant trade-offs in complex, multi-step tasks, and remain vulnerable to sophisticated jailbreak attacks. To address this, we introduce SAFEPATH, a lightweight alignment method that fine-tunes LRMs to emit a short, 8-token Safety Primer at the start of their reasoning, in response to harmful prompts, while leaving the rest of the reasoning process unsupervised. Empirical results across multiple benchmarks indicate that SAFEPATH effectively reduces harmful outputs while maintaining reasoning performance. Specifically, SAFEPATH reduces harmful responses by up to 90.0% and blocks 83.3% of jailbreak attempts in the DeepSeek-R1-Distill-Llama-8B model, while requiring 295.9x less compute than Direct Refusal and 314.1x less than SafeChain. We further introduce a zero-shot variant that requires no fine-tuning. In addition, we provide a comprehensive analysis of how existing methods in LLMs generalize, or fail, when applied to reasoning-centric models, revealing critical gaps and new directions for safer AI. 

**Abstract (ZH)**: SAFEPATH：一种减轻有害输出的轻量级对齐方法 

---
# Cost-Augmented Monte Carlo Tree Search for LLM-Assisted Planning 

**Title (ZH)**: 成本增强的蒙特卡洛树搜索方法在大语言模型辅助规划中的应用 

**Authors**: Zihao Zhang, Fei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14656)  

**Abstract**: While LLMs excel at open-ended reasoning, they often struggle with cost-sensitive planning, either treating all actions as having equal cost or failing to stay within strict budgets. In this paper, we introduce Cost-Augmented Monte Carlo Tree Search (CATS), a novel approach that brings explicit cost-awareness into LLM-guided planning. Tight cost constraints push the planner to quickly identify infeasible solutions, while looser constraints encourage optimization for minimal cost. We benchmark top LLMs such as GPT-4.1, Claude-3.7-Sonnet, and DeepSeek-R1, against our CATS planner to evaluate their performance in cost-sensitive scenarios. Our experiments suggest that raw LLMs such as GPT-4.1 often falter under tight budgets, whereas CATS consistently delivers strong performance, achieving higher task success rates and better cost efficiency. CATS provides an effective solution for budget-aware decision-making by combining the reasoning power of LLMs with structured search. 

**Abstract (ZH)**: Cost-Augmented Monte Carlo Tree Search for Cost-Aware Planning with Large Language Models 

---
# Debating for Better Reasoning: An Unsupervised Multimodal Approach 

**Title (ZH)**: 为更好地进行推理而辩论：一种无监督的多模态方法 

**Authors**: Ashutosh Adhikari, Mirella Lapata  

**Link**: [PDF](https://arxiv.org/pdf/2505.14627)  

**Abstract**: As Large Language Models (LLMs) gain expertise across diverse domains and modalities, scalable oversight becomes increasingly challenging, particularly when their capabilities may surpass human evaluators. Debate has emerged as a promising mechanism for enabling such oversight. In this work, we extend the debate paradigm to a multimodal setting, exploring its potential for weaker models to supervise and enhance the performance of stronger models. We focus on visual question answering (VQA), where two "sighted" expert vision-language models debate an answer, while a "blind" (text-only) judge adjudicates based solely on the quality of the arguments. In our framework, the experts defend only answers aligned with their beliefs, thereby obviating the need for explicit role-playing and concentrating the debate on instances of expert disagreement. Experiments on several multimodal tasks demonstrate that the debate framework consistently outperforms individual expert models. Moreover, judgments from weaker LLMs can help instill reasoning capabilities in vision-language models through finetuning. 

**Abstract (ZH)**: 大型语言模型在多种领域和模态中获得专业知识后，可扩展的监督变得越来越具有挑战性，尤其是当它们的能力可能超过人类评估者时。辩论作为一种机制在使这种监督成为可能方面展现出了前景。在本文中，我们将辩论 paradig姆扩展到多模态设置，探讨其在较弱模型监督并提升较强模型性能方面的潜力。我们专注于视觉问答（VQA），其中两个“有视力”的专家视觉-语言模型辩论答案，而一个“盲人”（仅文本）的裁判基于论点的质量做出裁决。在我们的框架中，专家们仅捍卫与其信念一致的答案，从而省去了明确的角色扮演的需求，并将辩论集中在专家存有分歧的实例上。在多项多模态任务上的实验表明，辩论框架始终优于单独的专家模型。此外，较弱的大规模语言模型的判断可以通过微调帮助视觉-语言模型获得推理能力。 

---
# SATBench: Benchmarking LLMs' Logical Reasoning via Automated Puzzle Generation from SAT Formulas 

**Title (ZH)**: SATBench：通过从SAT公式自动生成谜题来评估LLMs的逻辑推理能力 

**Authors**: Anjiang Wei, Yuheng Wu, Yingjia Wan, Tarun Suresh, Huanmi Tan, Zhanke Zhou, Sanmi Koyejo, Ke Wang, Alex Aiken  

**Link**: [PDF](https://arxiv.org/pdf/2505.14615)  

**Abstract**: We introduce SATBench, a benchmark for evaluating the logical reasoning capabilities of large language models (LLMs) through logical puzzles derived from Boolean satisfiability (SAT) problems. Unlike prior work that focuses on inference rule-based reasoning, which often involves deducing conclusions from a set of premises, our approach leverages the search-based nature of SAT problems, where the objective is to find a solution that fulfills a specified set of logical constraints. Each instance in SATBench is generated from a SAT formula, then translated into a story context and conditions using LLMs. The generation process is fully automated and allows for adjustable difficulty by varying the number of clauses. All 2100 puzzles are validated through both LLM-assisted and solver-based consistency checks, with human validation on a subset. Experimental results show that even the strongest model, o4-mini, achieves only 65.0% accuracy on hard UNSAT problems, close to the random baseline of 50%. SATBench exposes fundamental limitations in the search-based logical reasoning abilities of current LLMs and provides a scalable testbed for future research in logical reasoning. 

**Abstract (ZH)**: 我们介绍了SATBench，这是一个通过布尔可满足性（SAT）问题衍生的逻辑谜题来评估大型语言模型（LLMs）逻辑推理能力的基准。与先前侧重于基于推理规则的推理工作不同，我们的方法利用了SAT问题的搜索特性，目标是找到满足指定逻辑约束的解决方案。SATBench 中的每个实例都源自一个 SAT 公式，然后通过语言模型转换为故事背景和条件。生成过程完全自动化，并可通过改变子句的数量调整难度。2100 个谜题均通过语言模型辅助和求解器验证的一致性检查，其中部分通过人工验证。实验结果显示，即使是最强的模型o4-mini，在解决困难的 UNSAT 问题时也仅能达到65.0% 的准确性，接近随机基线的50%。SATBench暴露出当前LLMs在基于搜索的逻辑推理能力上的根本局限，并提供了一个可扩展的测试平台，用于未来逻辑推理研究。 

---
# Let LLMs Break Free from Overthinking via Self-Braking Tuning 

**Title (ZH)**: 让大语言模型通过自我刹车调优摆脱过度思考 

**Authors**: Haoran Zhao, Yuchen Yan, Yongliang Shen, Haolei Xu, Wenqi Zhang, Kaitao Song, Jian Shao, Weiming Lu, Jun Xiao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14604)  

**Abstract**: Large reasoning models (LRMs), such as OpenAI o1 and DeepSeek-R1, have significantly enhanced their reasoning capabilities by generating longer chains of thought, demonstrating outstanding performance across a variety of tasks. However, this performance gain comes at the cost of a substantial increase in redundant reasoning during the generation process, leading to high computational overhead and exacerbating the issue of overthinking. Although numerous existing approaches aim to address the problem of overthinking, they often rely on external interventions. In this paper, we propose a novel framework, Self-Braking Tuning (SBT), which tackles overthinking from the perspective of allowing the model to regulate its own reasoning process, thus eliminating the reliance on external control mechanisms. We construct a set of overthinking identification metrics based on standard answers and design a systematic method to detect redundant reasoning. This method accurately identifies unnecessary steps within the reasoning trajectory and generates training signals for learning self-regulation behaviors. Building on this foundation, we develop a complete strategy for constructing data with adaptive reasoning lengths and introduce an innovative braking prompt mechanism that enables the model to naturally learn when to terminate reasoning at an appropriate point. Experiments across mathematical benchmarks (AIME, AMC, MATH500, GSM8K) demonstrate that our method reduces token consumption by up to 60% while maintaining comparable accuracy to unconstrained models. 

**Abstract (ZH)**: 大型推理模型（LRMs）如OpenAI o1和DeepSeek-R1通过生成更长的推理链条显著增强了其推理能力，并在多种任务中展示了出色的表现。然而，这一性能提升是以推理过程中大量冗余计算为代价的，导致了高昂的计算开销并加剧了过度推理的问题。尽管已有许多方法试图解决过度推理问题，但它们通常依赖于外部干预。本文提出了一种新颖的框架——自我制动微调（SBT），从允许模型自主调节其推理过程的角度出发，从而消除对外部控制机制的依赖。我们基于标准答案构建了一套过度推理识别指标，并设计了一种系统的方法来检测冗余推理。该方法能够准确识别推理轨迹中的不必要的步骤，并生成用于学习自我调节行为的训练信号。在此基础上，我们开发了一种完整的策略来构建具有自适应推理长度的数据，并引入了一种创新的制动提示机制，使模型能够自然地学习在适当的时候终止推理。跨数学基准测试（AIME、AMC、MATH500、GSM8K）的实验表明，我们的方法在保持与未约束模型相当的准确性的前提下，可将令牌消耗降低高达60%。 

---
# Towards a Foundation Model for Communication Systems 

**Title (ZH)**: 面向通信系统的基础模型研究 

**Authors**: Davide Buffelli, Sowmen Das, Yu-Wei Lin, Sattar Vakili, Chien-Yi Wang, Masoud Attarifar, Pritthijit Nath, Da-shan Shiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14603)  

**Abstract**: Artificial Intelligence (AI) has demonstrated unprecedented performance across various domains, and its application to communication systems is an active area of research. While current methods focus on task-specific solutions, the broader trend in AI is shifting toward large general models capable of supporting multiple applications. In this work, we take a step toward a foundation model for communication data--a transformer-based, multi-modal model designed to operate directly on communication data. We propose methodologies to address key challenges, including tokenization, positional embedding, multimodality, variable feature sizes, and normalization. Furthermore, we empirically demonstrate that such a model can successfully estimate multiple features, including transmission rank, selected precoder, Doppler spread, and delay profile. 

**Abstract (ZH)**: 人工智能（AI）已在各种领域展现了前所未有的性能，其在通信系统中的应用是研究成果的活跃领域。尽管当前的方法主要集中在特定任务的解决方案上，更广泛的AI趋势正转向能够支持多种应用的大规模通用模型。在本项工作中，我们向通信数据的基础模型迈出了一步——一个基于变换器的多模态模型，旨在直接处理通信数据。我们提出了一种方法论来应对关键挑战，包括分词化、位置嵌入、多模态性、可变特征大小和标准化。此外，我们通过实验证明，这种模型能够成功估计多个特征，包括传输秩、选择的预编码器、多普勒扩展和延迟谱型。 

---
# Agent Context Protocols Enhance Collective Inference 

**Title (ZH)**: Agent Context Protocols 提高集体推理能力 

**Authors**: Devansh Bhardwaj, Arjun Beniwal, Shreyas Chaudhari, Ashwin Kalyan, Tanmay Rajpurohit, Karthik R. Narasimhan, Ameet Deshpande, Vishvak Murahari  

**Link**: [PDF](https://arxiv.org/pdf/2505.14569)  

**Abstract**: AI agents have become increasingly adept at complex tasks such as coding, reasoning, and multimodal understanding. However, building generalist systems requires moving beyond individual agents to collective inference -- a paradigm where multi-agent systems with diverse, task-specialized agents complement one another through structured communication and collaboration. Today, coordination is usually handled with imprecise, ad-hoc natural language, which limits complex interaction and hinders interoperability with domain-specific agents. We introduce Agent context protocols (ACPs): a domain- and agent-agnostic family of structured protocols for agent-agent communication, coordination, and error handling. ACPs combine (i) persistent execution blueprints -- explicit dependency graphs that store intermediate agent outputs -- with (ii) standardized message schemas, enabling robust and fault-tolerant multi-agent collective inference. ACP-powered generalist systems reach state-of-the-art performance: 28.3 % accuracy on AssistantBench for long-horizon web assistance and best-in-class multimodal technical reports, outperforming commercial AI systems in human evaluation. ACPs are highly modular and extensible, allowing practitioners to build top-tier generalist agents quickly. 

**Abstract (ZH)**: 基于代理上下文协议的通用智能系统：从个体代理到协作推理的转变 

---
# Multi-agent Reinforcement Learning vs. Fixed-Time Control for Traffic Signal Optimization: A Simulation Study 

**Title (ZH)**: 多智能体强化学习与固定时间控制在交通信号优化中的比较：一个仿真研究 

**Authors**: Saahil Mahato  

**Link**: [PDF](https://arxiv.org/pdf/2505.14544)  

**Abstract**: Urban traffic congestion, particularly at intersections, significantly impacts travel time, fuel consumption, and emissions. Traditional fixed-time signal control systems often lack the adaptability to manage dynamic traffic patterns effectively. This study explores the application of multi-agent reinforcement learning (MARL) to optimize traffic signal coordination across multiple intersections within a simulated environment. Utilizing Pygame, a simulation was developed to model a network of interconnected intersections with randomly generated vehicle flows to reflect realistic traffic variability. A decentralized MARL controller was implemented, in which each traffic signal operates as an autonomous agent, making decisions based on local observations and information from neighboring agents. Performance was evaluated against a baseline fixed-time controller using metrics such as average vehicle wait time and overall throughput. The MARL approach demonstrated statistically significant improvements, including reduced average waiting times and improved throughput. These findings suggest that MARL-based dynamic control strategies hold substantial promise for improving urban traffic management efficiency. More research is recommended to address scalability and real-world implementation challenges. 

**Abstract (ZH)**: 城市交叉口的交通拥堵显著影响着出行时间、燃油消耗和排放。传统的固定时间信号控制系统往往缺乏有效管理动态交通模式的适应性。本研究探讨了多代理强化学习（MARL）在仿真环境中优化多个交叉口交通信号协调的应用。利用Pygame开发了仿真模型，模拟了具有随机生成车辆流量的相互连接的交叉口网络，以反映现实中的交通变化。实现了去中心化的MARL控制器，其中每个交通信号作为自主代理，基于局部观察和邻近代理的信息做出决策。使用平均车辆等待时间和总体通过量等指标与固定时间基线控制器进行了性能评估。MARL方法在统计上显示出显著改进，包括减少了平均等待时间和提高了通过量。这些发现表明，基于MARL的动态控制策略在提高城市交通管理效率方面具有巨大潜力。建议进一步研究以解决可扩展性和实际实施挑战。 

---
# A Logic of General Attention Using Edge-Conditioned Event Models (Extended Version) 

**Title (ZH)**: 一种基于边条件事件模型的一般注意逻辑（扩展版本） 

**Authors**: Gaia Belardinelli, Thomas Bolander, Sebastian Watzl  

**Link**: [PDF](https://arxiv.org/pdf/2505.14539)  

**Abstract**: In this work, we present the first general logic of attention. Attention is a powerful cognitive ability that allows agents to focus on potentially complex information, such as logically structured propositions, higher-order beliefs, or what other agents pay attention to. This ability is a strength, as it helps to ignore what is irrelevant, but it can also introduce biases when some types of information or agents are systematically ignored. Existing dynamic epistemic logics for attention cannot model such complex attention scenarios, as they only model attention to atomic formulas. Additionally, such logics quickly become cumbersome, as their size grows exponentially in the number of agents and announced literals. Here, we introduce a logic that overcomes both limitations. First, we generalize edge-conditioned event models, which we show to be as expressive as standard event models yet exponentially more succinct (generalizing both standard event models and generalized arrow updates). Second, we extend attention to arbitrary formulas, allowing agents to also attend to other agents' beliefs or attention. Our work treats attention as a modality, like belief or awareness. We introduce attention principles that impose closure properties on that modality and that can be used in its axiomatization. Throughout, we illustrate our framework with examples of AI agents reasoning about human attentional biases, demonstrating how such agents can discover attentional biases. 

**Abstract (ZH)**: 本工作提出了第一个通用注意力逻辑。 

---
# Guarded Query Routing for Large Language Models 

**Title (ZH)**: 大型语言模型中的受控查询路由 

**Authors**: Richard Šléher, William Brach, Tibor Sloboda, Kristián Košťál, Lukas Galke  

**Link**: [PDF](https://arxiv.org/pdf/2505.14524)  

**Abstract**: Query routing, the task to route user queries to different large language model (LLM) endpoints, can be considered as a text classification problem. However, out-of-distribution queries must be handled properly, as those could be questions about unrelated domains, queries in other languages, or even contain unsafe text. Here, we thus study a \emph{guarded} query routing problem, for which we first introduce the Guarded Query Routing Benchmark (GQR-Bench), which covers three exemplary target domains (law, finance, and healthcare), and seven datasets to test robustness against out-of-distribution queries. We then use GQR-Bench to contrast the effectiveness and efficiency of LLM-based routing mechanisms (GPT-4o-mini, Llama-3.2-3B, and Llama-3.1-8B), standard LLM-based guardrail approaches (LlamaGuard and NVIDIA NeMo Guardrails), continuous bag-of-words classifiers (WideMLP, fastText), and traditional machine learning models (SVM, XGBoost). Our results show that WideMLP, enhanced with out-of-domain detection capabilities, yields the best trade-off between accuracy (88\%) and speed (<4ms). The embedding-based fastText excels at speed (<1ms) with acceptable accuracy (80\%), whereas LLMs yield the highest accuracy (91\%) but are comparatively slow (62ms for local Llama-3.1:8B and 669ms for remote GPT-4o-mini calls). Our findings challenge the automatic reliance on LLMs for (guarded) query routing and provide concrete recommendations for practical applications. GQR-Bench will be released as a Python package -- \texttt{gqr}. 

**Abstract (ZH)**: 带有防护机制的查询路由问题：Guarded Query Routing Benchmark的研究 

---
# BACON: A fully explainable AI model with graded logic for decision making problems 

**Title (ZH)**: BACON: 一种基于分级逻辑的可完全解释的AI决策模型 

**Authors**: Haishi Bai, Jozo Dujmovic, Jianwu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14510)  

**Abstract**: As machine learning models and autonomous agents are increasingly deployed in high-stakes, real-world domains such as healthcare, security, finance, and robotics, the need for transparent and trustworthy explanations has become critical. To ensure end-to-end transparency of AI decisions, we need models that are not only accurate but also fully explainable and human-tunable. We introduce BACON, a novel framework for automatically training explainable AI models for decision making problems using graded logic. BACON achieves high predictive accuracy while offering full structural transparency and precise, logic-based symbolic explanations, enabling effective human-AI collaboration and expert-guided refinement. We evaluate BACON with a diverse set of scenarios: classic Boolean approximation, Iris flower classification, house purchasing decisions and breast cancer diagnosis. In each case, BACON provides high-performance models while producing compact, human-verifiable decision logic. These results demonstrate BACON's potential as a practical and principled approach for delivering crisp, trustworthy explainable AI. 

**Abstract (ZH)**: 随着机器学习模型和自主代理在高风险的实际领域如医疗、安全、金融和机器人技术中的广泛应用，对透明和可信赖的解释需求变得至关重要。为确保AI决策的端到端透明度，我们需要不仅准确、而且完全可解释且可由人类调整的模型。我们提出了BACON，一种使用分级逻辑自动训练决策制定问题解释性AI模型的新框架。BACON在保持高预测准确性的基础上，提供了完整的结构透明度和精确诊断逻辑符号解释，从而促进有效的人类-AI协作和专家引导的优化。我们通过一系列不同的场景评估了BACON：经典布尔近似、鸢尾花分类、房屋购买决策以及乳腺癌诊断。在每一个案例中，BACON都提供了高性能模型，并生成了紧凑且可由人类验证的决策逻辑。这些结果展示了BACON作为一种实际且原则性的方法，具有为用户提供清晰且可信赖的解释性AI的潜力。 

---
# Reasoning Models Better Express Their Confidence 

**Title (ZH)**: 推理模型更好地表达其置信度 

**Authors**: Dongkeun Yoon, Seungone Kim, Sohee Yang, Sunkyoung Kim, Soyeon Kim, Yongil Kim, Eunbi Choi, Yireun Kim, Minjoon Seo  

**Link**: [PDF](https://arxiv.org/pdf/2505.14489)  

**Abstract**: Despite their strengths, large language models (LLMs) often fail to communicate their confidence accurately, making it difficult to assess when they might be wrong and limiting their reliability. In this work, we demonstrate that reasoning models-LLMs that engage in extended chain-of-thought (CoT) reasoning-exhibit superior performance not only in problem-solving but also in accurately expressing their confidence. Specifically, we benchmark six reasoning models across six datasets and find that they achieve strictly better confidence calibration than their non-reasoning counterparts in 33 out of the 36 settings. Our detailed analysis reveals that these gains in calibration stem from the slow thinking behaviors of reasoning models-such as exploring alternative approaches and backtracking-which enable them to adjust their confidence dynamically throughout their CoT, making it progressively more accurate. In particular, we find that reasoning models become increasingly better calibrated as their CoT unfolds, a trend not observed in non-reasoning models. Moreover, removing slow thinking behaviors from the CoT leads to a significant drop in calibration. Lastly, we show that these gains are not exclusive to reasoning models-non-reasoning models also benefit when guided to perform slow thinking via in-context learning. 

**Abstract (ZH)**: 尽管大型语言模型具有优势，但在准确传达信心方面常常表现出欠缺，这使得评估其错误的可能性变得困难，从而限制了其可靠性。在本工作中，我们证明了推理模型——那些进行扩展链式思考（CoT）推理的大型语言模型——不仅在问题解决上表现出色，还能更准确地表达其信心。具体而言，我们在六个数据集中对六种推理模型进行了基准测试，并发现它们在36种设置中的33种中严格超越了非推理模型的信心校准效果。我们详细的分析表明，这种校准改进源于推理模型的慢思考行为，如探索其他方法和回溯，这使它们能够在CoT过程中动态调整信心，使其更加准确。特别是，我们发现随着CoT的展开，推理模型的信心校准逐渐改善，这种趋势在非推理模型中未被观察到。此外，移除CoT中的慢思考行为会导致校准显著下降。最后，我们展示了这些改进不仅限于推理模型，当通过上下文学习引导非推理模型进行慢思考时，它们也会从中受益。 

---
# Towards Reliable Proof Generation with LLMs: A Neuro-Symbolic Approach 

**Title (ZH)**: 基于神经符号方法的可靠证明生成：LLMs的途径 

**Authors**: Oren Sultan, Eitan Stern, Dafna Shahaf  

**Link**: [PDF](https://arxiv.org/pdf/2505.14479)  

**Abstract**: Large language models (LLMs) struggle with formal domains that require rigorous logical deduction and symbolic reasoning, such as mathematical proof generation. We propose a neuro-symbolic approach that combines LLMs' generative strengths with structured components to overcome this challenge. As a proof-of-concept, we focus on geometry problems. Our approach is two-fold: (1) we retrieve analogous problems and use their proofs to guide the LLM, and (2) a formal verifier evaluates the generated proofs and provides feedback, helping the model fix incorrect proofs. We demonstrate that our method significantly improves proof accuracy for OpenAI's o1 model (58%-70% improvement); both analogous problems and the verifier's feedback contribute to these gains. More broadly, shifting to LLMs that generate provably correct conclusions could dramatically improve their reliability, accuracy and consistency, unlocking complex tasks and critical real-world applications that require trustworthiness. 

**Abstract (ZH)**: 大型语言模型在需要严格的逻辑演绎和符号推理的正式领域（如数学证明生成）中存在挑战。我们提出了一种神经符号方法，结合了大型语言模型的生成优势和结构化组件，以克服这一挑战。作为概念验证，我们专注于几何问题。该方法分为两部分：（1）检索类似问题并使用其证明来引导大型语言模型；（2）正式验证器评估生成的证明并提供反馈，帮助模型修正错误的证明。我们证明，我们的方法显著提高了OpenAI o1模型的证明准确性（提高58%-70%）；类似问题和验证器的反馈都对这些改进有所贡献。更广泛而言，转向生成可证明正确结论的大型语言模型可以显著提高它们的可靠性和一致性，解锁需要可信度的复杂任务和关键现实世界应用。 

---
# SCOPE: Compress Mathematical Reasoning Steps for Efficient Automated Process Annotation 

**Title (ZH)**: 范围：压缩数学推理步骤以实现高效的自动化过程标注 

**Authors**: Huimin Xu, Xin Mao, Feng-Lin Li, Xiaobao Wu, Wang Chen, Wei Zhang, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14419)  

**Abstract**: Process Reward Models (PRMs) have demonstrated promising results in mathematical reasoning, but existing process annotation approaches, whether through human annotations or Monte Carlo simulations, remain computationally expensive. In this paper, we introduce Step COmpression for Process Estimation (SCOPE), a novel compression-based approach that significantly reduces annotation costs. We first translate natural language reasoning steps into code and normalize them through Abstract Syntax Tree, then merge equivalent steps to construct a prefix tree. Unlike simulation-based methods that waste numerous samples on estimation, SCOPE leverages a compression-based prefix tree where each root-to-leaf path serves as a training sample, reducing the complexity from $O(NMK)$ to $O(N)$. We construct a large-scale dataset containing 196K samples with only 5% of the computational resources required by previous methods. Empirical results demonstrate that PRMs trained on our dataset consistently outperform existing automated annotation approaches on both Best-of-N strategy and ProcessBench. 

**Abstract (ZH)**: 基于步骤压缩的过程估计（SCOPE）在过程奖励模型中的应用 

---
# PRL: Prompts from Reinforcement Learning 

**Title (ZH)**: PRL: 强化学习生成的提示 

**Authors**: Paweł Batorski, Adrian Kosmala, Paul Swoboda  

**Link**: [PDF](https://arxiv.org/pdf/2505.14412)  

**Abstract**: Effective prompt engineering remains a central challenge in fully harnessing the capabilities of LLMs. While well-designed prompts can dramatically enhance performance, crafting them typically demands expert intuition and a nuanced understanding of the task. Moreover, the most impactful prompts often hinge on subtle semantic cues, ones that may elude human perception but are crucial for guiding LLM behavior. In this paper, we introduce PRL (Prompts from Reinforcement Learning), a novel RL-based approach for automatic prompt generation. Unlike previous methods, PRL can produce novel few-shot examples that were not seen during training. Our approach achieves state-of-the-art performance across a range of benchmarks, including text classification, simplification, and summarization. On the classification task, it surpasses prior methods by 2.58% over APE and 1.00% over EvoPrompt. Additionally, it improves the average ROUGE scores on the summarization task by 4.32 over APE and by 2.12 over EvoPrompt and the SARI score on simplification by 6.93 over APE and by 6.01 over EvoPrompt. Our code is available at this https URL . 

**Abstract (ZH)**: 有效的提示工程仍然是充分利用大型语言模型能力的核心挑战。虽然精心设计的提示可以显著提升性能，但构建它们通常需要专家直觉和对任务的深刻理解。此外，最具影响力的提示往往依赖于微妙的语义线索，这些线索可能逃过人类的感知，但对于引导大型语言模型的行为至关重要。在这篇论文中，我们介绍了PRL（强化学习提示），一种基于强化学习的新颖自动提示生成方法。与之前的方法不同，PRL能够生成训练期间未见过的少量示例。我们的方法在文本分类、简化和总结等多种基准上实现了最先进的性能。在分类任务中，它分别在APE和EvoPrompt的基础上提升了2.58%和1.00%。此外，它在总结任务上的平均ROUGE分数分别提高了4.32和2.12，在简化任务上的SARI分数分别提高了6.93和6.01。我们的代码可在以下链接获取：this https URL。 

---
# Unearthing Gems from Stones: Policy Optimization with Negative Sample Augmentation for LLM Reasoning 

**Title (ZH)**: 从石中琢玉：基于负样本增强的政策优化在大语言模型推理中的应用 

**Authors**: Zhaohui Yang, Shilei Jiang, Chen Hu, Linjing Li, Shihong Deng, Daxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14403)  

**Abstract**: Recent advances in reasoning language models have witnessed a paradigm shift from short to long CoT pattern. Given the substantial computational cost of rollouts in long CoT models, maximizing the utility of fixed training datasets becomes crucial. Our analysis reveals that negative responses contain valuable components such as self-reflection and error-correction steps, yet primary existing methods either completely discard negative samples (RFT) or apply equal penalization across all tokens (RL), failing to leverage these potential learning signals. In light of this, we propose Behavior Constrained Policy Gradient with Negative Sample Augmentation (BCPG-NSA), a fine-grained offline RL framework that encompasses three stages: 1) sample segmentation, 2) consensus-based step correctness assessment combining LLM and PRM judgers, and 3) policy optimization with NSA designed to effectively mine positive steps within negative samples. Experimental results show that BCPG-NSA outperforms baselines on several challenging math/coding reasoning benchmarks using the same training dataset, achieving improved sample efficiency and demonstrating robustness and scalability when extended to multiple iterations. 

**Abstract (ZH)**: Recent Advances in Reasoning Language Models Have Witnessed a Paradigm Shift from Short to Long CoT Patterns: Behavior Constrained Policy Gradient with Negative Sample Augmentation (BCPG-NSA) for Efficient and Robust Learning 

---
# Causal Cartographer: From Mapping to Reasoning Over Counterfactual Worlds 

**Title (ZH)**: 因果地图师：从映射到因果联想的世界推理 

**Authors**: Gaël Gendron, Jože M. Rožanec, Michael Witbrock, Gillian Dobbie  

**Link**: [PDF](https://arxiv.org/pdf/2505.14396)  

**Abstract**: Causal world models are systems that can answer counterfactual questions about an environment of interest, i.e. predict how it would have evolved if an arbitrary subset of events had been realized differently. It requires understanding the underlying causes behind chains of events and conducting causal inference for arbitrary unseen distributions. So far, this task eludes foundation models, notably large language models (LLMs), which do not have demonstrated causal reasoning capabilities beyond the memorization of existing causal relationships. Furthermore, evaluating counterfactuals in real-world applications is challenging since only the factual world is observed, limiting evaluation to synthetic datasets. We address these problems by explicitly extracting and modeling causal relationships and propose the Causal Cartographer framework. First, we introduce a graph retrieval-augmented generation agent tasked to retrieve causal relationships from data. This approach allows us to construct a large network of real-world causal relationships that can serve as a repository of causal knowledge and build real-world counterfactuals. In addition, we create a counterfactual reasoning agent constrained by causal relationships to perform reliable step-by-step causal inference. We show that our approach can extract causal knowledge and improve the robustness of LLMs for causal reasoning tasks while reducing inference costs and spurious correlations. 

**Abstract (ZH)**: 因果世界模型是能够回答所关注环境的反事实问题的系统，即预测若某事件子集以不同的方式实现，该环境会如何演变。这需要理解事件链背后的因果关系，并进行任意未见分布的因果推断。迄今为止，这一任务难以被基础模型，尤其是大型语言模型（LLMs），解决，因为它们只能记忆现有的因果关系而缺乏推导新的因果关系的能力。此外，在现实世界应用中评估反事实是具有挑战性的，因为只能观察到事实世界，这限制了评价方法只能局限于合成数据集。我们通过明确提取和建模因果关系来解决这些问题，并提出因果制图框架。首先，我们引入了一个图检索增强的生成代理，任务是从数据中检索因果关系。这种方法允许我们构建广泛的现实世界因果关系网络，作为因果知识库，并构建现实世界的反事实。此外，我们创建了一个受限于因果关系的反事实推理代理，以进行可靠的逐步因果推断。我们展示了我们的方法可以提取因果知识，并提高大型语言模型在因果推理任务中的鲁棒性，同时减少推理成本和虚假相关性。 

---
# Knowledge Graph Based Repository-Level Code Generation 

**Title (ZH)**: 基于知识图谱的仓库级代码生成 

**Authors**: Mihir Athale, Vishal Vaddina  

**Link**: [PDF](https://arxiv.org/pdf/2505.14394)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have transformed code generation from natural language queries. However, despite their extensive knowledge and ability to produce high-quality code, LLMs often struggle with contextual accuracy, particularly in evolving codebases. Current code search and retrieval methods frequently lack robustness in both the quality and contextual relevance of retrieved results, leading to suboptimal code generation. This paper introduces a novel knowledge graph-based approach to improve code search and retrieval leading to better quality of code generation in the context of repository-level tasks. The proposed approach represents code repositories as graphs, capturing structural and relational information for enhanced context-aware code generation. Our framework employs a hybrid approach for code retrieval to improve contextual relevance, track inter-file modular dependencies, generate more robust code and ensure consistency with the existing codebase. We benchmark the proposed approach on the Evolutionary Code Benchmark (EvoCodeBench) dataset, a repository-level code generation benchmark, and demonstrate that our method significantly outperforms the baseline approach. These findings suggest that knowledge graph based code generation could advance robust, context-sensitive coding assistance tools. 

**Abstract (ZH)**: 近年来，大型语言模型的最新进展已将代码生成从自然语言查询中转变过来。然而，尽管大型语言模型具备丰富的知识并能生成高质量的代码，它们在上下文准确性方面常常遇到困难，尤其是在不断演进的代码库中。当前的代码搜索和检索方法在检索结果的质量和上下文相关性方面往往缺乏稳健性，导致代码生成效果不佳。本文提出了一种基于知识图谱的方法，以提高代码搜索和检索的质量，从而在仓库级任务中提高代码生成的质量。该方法将代码仓库表示为图，以捕获结构和关系信息，增强上下文感知的代码生成。我们的框架采用了混合方法进行代码检索，以提高上下文相关性、跟踪跨文件模块依赖关系、生成更 robust 的代码，并确保与现有代码库的一致性。我们在 Evansion Code Benchmark (EvoCodeBench) 数据集上对该方法进行了基准测试，这是一个仓库级代码生成基准，结果表明我们的方法明显优于基线方法。这些发现表明，基于知识图谱的代码生成可能有助于推进稳健且上下文敏感的编码辅助工具。 

---
# Beyond the First Error: Process Reward Models for Reflective Mathematical Reasoning 

**Title (ZH)**: 超越首个错误：过程奖励模型在反思性数学推理中的应用 

**Authors**: Zhaohui Yang, Chenghua He, Xiaowen Shi, Linjing Li, Qiyue Yin, Shihong Deng, Daxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14391)  

**Abstract**: Many studies focus on data annotation techniques for training effective PRMs. However, current methods encounter a significant issue when applied to long CoT reasoning processes: they tend to focus solely on the first incorrect step and all preceding steps, assuming that all subsequent steps are incorrect. These methods overlook the unique self-correction and reflection mechanisms inherent in long CoT, where correct reasoning steps may still occur after initial reasoning mistakes. To address this issue, we propose a novel data annotation method for PRMs specifically designed to score the long CoT reasoning process. Given that under the reflection pattern, correct and incorrect steps often alternate, we introduce the concepts of Error Propagation and Error Cessation, enhancing PRMs' ability to identify both effective self-correction behaviors and reasoning based on erroneous steps. Leveraging an LLM-based judger for annotation, we collect 1.7 million data samples to train a 7B PRM and evaluate it at both solution and step levels. Experimental results demonstrate that compared to existing open-source PRMs and PRMs trained on open-source datasets, our PRM achieves superior performance across various metrics, including search guidance, BoN, and F1 scores. Compared to widely used MC-based annotation methods, our annotation approach not only achieves higher data efficiency but also delivers superior performance. Detailed analysis is also conducted to demonstrate the stability and generalizability of our method. 

**Abstract (ZH)**: 一种针对长链推理过程的新型数据标注方法：增强PRMs的错误传播与终止识别能力 

---
# SCAN: Semantic Document Layout Analysis for Textual and Visual Retrieval-Augmented Generation 

**Title (ZH)**: SCAN: 基于语义文档布局分析的文本和视觉检索增强生成 

**Authors**: Yuyang Dong, Nobuhiro Ueda, Krisztián Boros, Daiki Ito, Takuya Sera, Masafumi Oyamada  

**Link**: [PDF](https://arxiv.org/pdf/2505.14381)  

**Abstract**: With the increasing adoption of Large Language Models (LLMs) and Vision-Language Models (VLMs), rich document analysis technologies for applications like Retrieval-Augmented Generation (RAG) and visual RAG are gaining significant attention. Recent research indicates that using VLMs can achieve better RAG performance, but processing rich documents still remains a challenge since a single page contains large amounts of information. In this paper, we present SCAN (\textbf{S}emanti\textbf{C} Document Layout \textbf{AN}alysis), a novel approach enhancing both textual and visual Retrieval-Augmented Generation (RAG) systems working with visually rich documents. It is a VLM-friendly approach that identifies document components with appropriate semantic granularity, balancing context preservation with processing efficiency. SCAN uses a coarse-grained semantic approach that divides documents into coherent regions covering continuous components. We trained the SCAN model by fine-tuning object detection models with sophisticated annotation datasets. Our experimental results across English and Japanese datasets demonstrate that applying SCAN improves end-to-end textual RAG performance by up to 9.0\% and visual RAG performance by up to 6.4\%, outperforming conventional approaches and even commercial document processing solutions. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）和视觉语言模型（VLMs）的广泛应用，用于检索增强生成（RAG）及其视觉扩展的应用富文本分析技术正受到广泛关注。近期研究表明，使用VLMs可以实现更好的RAG性能，但处理富文本文档仍是一项挑战，因为单页包含大量信息。本文提出SCAN（语义文档布局分析），这是一种增强文本和视觉RAG系统的创新方法，特别适用于视觉丰富的文档。SCAN是一个VLM友好的方法，能够以适当的语义粒度识别文档组件，平衡上下文保留与处理效率。SCAN采用粗粒度语义方法，将文档划分为连贯的区域，覆盖连续的组件。我们通过使用详细注释数据集微调对象检测模型来训练SCAN模型。跨英语和日语数据集的实验结果表明，使用SCAN可以分别提高端到端文本RAG性能最多9.0%和视觉RAG性能最多6.4%，超过传统方法，甚至超过商用文档处理解决方案。 

---
# Towards Embodied Cognition in Robots via Spatially Grounded Synthetic Worlds 

**Title (ZH)**: 通过空间化接地合成世界实现机器人本体认知的研究 

**Authors**: Joel Currie, Gioele Migno, Enrico Piacenti, Maria Elena Giannaccini, Patric Bach, Davide De Tommaso, Agnieszka Wykowska  

**Link**: [PDF](https://arxiv.org/pdf/2505.14366)  

**Abstract**: We present a conceptual framework for training Vision-Language Models (VLMs) to perform Visual Perspective Taking (VPT), a core capability for embodied cognition essential for Human-Robot Interaction (HRI). As a first step toward this goal, we introduce a synthetic dataset, generated in NVIDIA Omniverse, that enables supervised learning for spatial reasoning tasks. Each instance includes an RGB image, a natural language description, and a ground-truth 4X4 transformation matrix representing object pose. We focus on inferring Z-axis distance as a foundational skill, with future extensions targeting full 6 Degrees Of Freedom (DOFs) reasoning. The dataset is publicly available to support further research. This work serves as a foundational step toward embodied AI systems capable of spatial understanding in interactive human-robot scenarios. 

**Abstract (ZH)**: 我们提出了一种概念框架，用于训练视觉-语言模型（VLMs）执行视觉观点转换（VPT），这是实现人类-机器人交互（HRI）的核心能力之一。为了这一目标的第一步，我们介绍了在NVIDIA Omniverse中生成的合成数据集，以支持空间推理任务的监督学习。每个实例包含一个RGB图像、自然语言描述以及表示物体姿态的地面 truth 4x4变换矩阵。我们专注于推断Z轴距离作为一项基础技能，未来扩展将针对完整的六自由度（6DOFs）推理。该数据集已公开，以支持进一步的研究。本工作为能够在交互式人机场景中实现空间理解的体态人工智能系统奠定了基础。 

---
# SafetyNet: Detecting Harmful Outputs in LLMs by Modeling and Monitoring Deceptive Behaviors 

**Title (ZH)**: SafetyNet: 通过建模和监控欺骗性行为来检测LLM的有害输出 

**Authors**: Maheep Chaudhary, Fazl Barez  

**Link**: [PDF](https://arxiv.org/pdf/2505.14300)  

**Abstract**: High-risk industries like nuclear and aviation use real-time monitoring to detect dangerous system conditions. Similarly, Large Language Models (LLMs) need monitoring safeguards. We propose a real-time framework to predict harmful AI outputs before they occur by using an unsupervised approach that treats normal behavior as the baseline and harmful outputs as outliers. Our study focuses specifically on backdoor-triggered responses -- where specific input phrases activate hidden vulnerabilities causing the model to generate unsafe content like violence, pornography, or hate speech. We address two key challenges: (1) identifying true causal indicators rather than surface correlations, and (2) preventing advanced models from deception -- deliberately evading monitoring systems. Hence, we approach this problem from an unsupervised lens by drawing parallels to human deception: just as humans exhibit physical indicators while lying, we investigate whether LLMs display distinct internal behavioral signatures when generating harmful content. Our study addresses two critical challenges: 1) designing monitoring systems that capture true causal indicators rather than superficial correlations; and 2)preventing intentional evasion by increasingly capable "Future models''. Our findings show that models can produce harmful content through causal mechanisms and can become deceptive by: (a) alternating between linear and non-linear representations, and (b) modifying feature relationships. To counter this, we developed Safety-Net -- a multi-detector framework that monitors different representation dimensions, successfully detecting harmful behavior even when information is shifted across representational spaces to evade individual monitors. Our evaluation shows 96% accuracy in detecting harmful cases using our unsupervised ensemble approach. 

**Abstract (ZH)**: 高风险行业如核能和航空使用实时监控来检测危险系统状态。类似地，大型语言模型（LLMs）也需要监控保护。我们提出一种实时框架，在有害AI输出发生之前，通过无监督方法将其预测出来，即以正常行为作为基线，将有害输出视为异常。我们的研究具体关注后门触发响应——特定输入短语激活隐藏漏洞，导致模型生成不安全内容，如暴力、色情或仇恨言论。我们面临两个关键挑战：（1）识别真正的因果指示而不是表面相关性；（2）防止高级模型的欺骗行为——故意规避监控系统。因此，我们从无监督的角度处理这一问题，类似于人类在撒谎时的生理指标，我们研究LLM在生成有害内容时是否表现出独特的内部行为特征。我们的研究针对两个关键挑战：1）设计能够捕捉真正因果指示而非表面相关性的监控系统；2）防止日益先进的“未来模型”的故意规避。研究发现，模型可以通过因果机制产生有害内容，并通过（a）交替使用线性和非线性表示；（b）修改特征关系展现出欺骗性。为此，我们开发了Safety-Net——一个多检测器框架，监控不同的表示维度，在信息在表示空间中转移以规避个体监控时，仍可成功检测有害行为。我们的评估显示，使用无监督集成方法检测有害案例的准确率为96%。 

---
# EVA: Red-Teaming GUI Agents via Evolving Indirect Prompt Injection 

**Title (ZH)**: EVA：通过演化间接提示注入红队测试GUI代理 

**Authors**: Yijie Lu, Tianjie Ju, Manman Zhao, Xinbei Ma, Yuan Guo, ZhuoSheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14289)  

**Abstract**: As multimodal agents are increasingly trained to operate graphical user interfaces (GUIs) to complete user tasks, they face a growing threat from indirect prompt injection, attacks in which misleading instructions are embedded into the agent's visual environment, such as popups or chat messages, and misinterpreted as part of the intended task. A typical example is environmental injection, in which GUI elements are manipulated to influence agent behavior without directly modifying the user prompt. To address these emerging attacks, we propose EVA, a red teaming framework for indirect prompt injection which transforms the attack into a closed loop optimization by continuously monitoring an agent's attention distribution over the GUI and updating adversarial cues, keywords, phrasing, and layout, in response. Compared with prior one shot methods that generate fixed prompts without regard for how the model allocates visual attention, EVA dynamically adapts to emerging attention hotspots, yielding substantially higher attack success rates and far greater transferability across diverse GUI scenarios. We evaluate EVA on six widely used generalist and specialist GUI agents in realistic settings such as popup manipulation, chat based phishing, payments, and email composition. Experimental results show that EVA substantially improves success rates over static baselines. Under goal agnostic constraints, where the attacker does not know the agent's task intent, EVA still discovers effective patterns. Notably, we find that injection styles transfer well across models, revealing shared behavioral biases in GUI agents. These results suggest that evolving indirect prompt injection is a powerful tool not only for red teaming agents, but also for uncovering common vulnerabilities in their multimodal decision making. 

**Abstract (ZH)**: 多模态代理中的间接提示注入防卫框架：EVA 

---
# Toward Embodied AGI: A Review of Embodied AI and the Road Ahead 

**Title (ZH)**: 向具身AGI迈进：具身AI综述与未来之路 

**Authors**: Yequan Wang, Aixin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.14235)  

**Abstract**: Artificial General Intelligence (AGI) is often envisioned as inherently embodied. With recent advances in robotics and foundational AI models, we stand at the threshold of a new era-one marked by increasingly generalized embodied AI systems. This paper contributes to the discourse by introducing a systematic taxonomy of Embodied AGI spanning five levels (L1-L5). We review existing research and challenges at the foundational stages (L1-L2) and outline the key components required to achieve higher-level capabilities (L3-L5). Building on these insights and existing technologies, we propose a conceptual framework for an L3+ robotic brain, offering both a technical outlook and a foundation for future exploration. 

**Abstract (ZH)**: 人工通用智能（AGI）常常被视为固有的具身化。随着机器人技术和基础AI模型的最新进展，我们站在了一个新时代的门槛上——这个时代以日益通用的具身AI系统为标志。本文通过引入涵盖五个层级（L1-L5）的具身AGI系统系统分类法，为相关讨论做出了贡献。我们回顾了基础阶段（L1-L2）的研究和挑战，并概述了实现更高层级能力（L3-L5）所需的关键组件。基于这些洞见和现有技术，我们提出了一种L3+级机器人脑的概念框架，提供了一种技术前景并为未来的探索奠定了基础。 

---
# Reinforcement Learning vs. Distillation: Understanding Accuracy and Capability in LLM Reasoning 

**Title (ZH)**: 强化学习 vs. 提炼：理解大语言模型推理的准确性和能力 

**Authors**: Minwu Kim, Anubhav Shrestha, Safal Shrestha, Aadim Nepal, Keith Ross  

**Link**: [PDF](https://arxiv.org/pdf/2505.14216)  

**Abstract**: Recent studies have shown that reinforcement learning with verifiable rewards (RLVR) enhances overall accuracy but fails to improve capability, while distillation can improve both. In this paper, we investigate the mechanisms behind these phenomena. First, we demonstrate that RLVR does not improve capability because it focuses on improving the accuracy of the less-difficult questions to the detriment of the accuracy of the most difficult questions, thereby leading to no improvement in capability. Second, we find that RLVR does not merely increase the success probability for the less difficult questions, but in our small model settings produces quality responses that were absent in its output distribution before training. In addition, we show these responses are neither noticeably longer nor feature more reflection-related keywords, underscoring the need for more reliable indicators of response quality. Third, we show that while distillation reliably improves accuracy by learning strong reasoning patterns, it only improves capability when new knowledge is introduced. Moreover, when distilling only with reasoning patterns and no new knowledge, the accuracy of the less-difficult questions improves to the detriment of the most difficult questions, similar to RLVR. Together, these findings offer a clearer understanding of how RLVR and distillation shape reasoning behavior in language models. 

**Abstract (ZH)**: 近期研究显示，可验证奖励强化学习（RLVR）提高了整体准确性但未能提升能力，而知识蒸馏则能同时提升两者。本文研究了这些现象背后的机制。首先，我们证明RLVR未能提升能力是因为它侧重于提高较简单问题的准确性而牺牲了最难问题的准确性，从而导致能力未获提升。其次，我们发现RLVR不仅增加了较简单问题的成功概率，还在小型模型设置中产生了训练前输出分布中缺乏的高质量回答。此外，我们表明这些回答既没有显著更长，也不包含更多反思相关关键词，强调了需要更可靠的回答质量指标。第三，我们展示了虽然知识蒸馏通过学习强推理模式可靠地提升了准确性，但在引入新知识的情况下才提升能力。而且，在仅使用推理模式而不引入新知识的知识蒸馏中，较简单问题的准确性提升是以最困难问题的准确性下降为代价的，类似于RLVR。这些发现共同为我们理解RLVR和知识蒸馏如何影响语言模型的推理行为提供了更清晰的认识。 

---
# Embedded Mean Field Reinforcement Learning for Perimeter-defense Game 

**Title (ZH)**: 嵌入式均场强化学习在周界防御博弈中的应用 

**Authors**: Li Wang, Xin Yu, Xuxin Lv, Gangzheng Ai, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14209)  

**Abstract**: With the rapid advancement of unmanned aerial vehicles (UAVs) and missile technologies, perimeter-defense game between attackers and defenders for the protection of critical regions have become increasingly complex and strategically significant across a wide range of domains. However, existing studies predominantly focus on small-scale, simplified two-dimensional scenarios, often overlooking realistic environmental perturbations, motion dynamics, and inherent heterogeneity--factors that pose substantial challenges to real-world applicability. To bridge this gap, we investigate large-scale heterogeneous perimeter-defense game in a three-dimensional setting, incorporating realistic elements such as motion dynamics and wind fields. We derive the Nash equilibrium strategies for both attackers and defenders, characterize the victory regions, and validate our theoretical findings through extensive simulations. To tackle large-scale heterogeneous control challenges in defense strategies, we propose an Embedded Mean-Field Actor-Critic (EMFAC) framework. EMFAC leverages representation learning to enable high-level action aggregation in a mean-field manner, supporting scalable coordination among defenders. Furthermore, we introduce a lightweight agent-level attention mechanism based on reward representation, which selectively filters observations and mean-field information to enhance decision-making efficiency and accelerate convergence in large-scale tasks. Extensive simulations across varying scales demonstrate the effectiveness and adaptability of EMFAC, which outperforms established baselines in both convergence speed and overall performance. To further validate practicality, we test EMFAC in small-scale real-world experiments and conduct detailed analyses, offering deeper insights into the framework's effectiveness in complex scenarios. 

**Abstract (ZH)**: 无人机与导弹技术迅速发展背景下，三维环境中的大面积异质性 perimeter-防御博弈分析及嵌入式平均场演员-评论家框架研究 

---
# Dynamic Replanning for Improved Public Transport Routing 

**Title (ZH)**: 改进公交路线规划的动态重新规划方法 

**Authors**: Abdallah Abuaisha, Bojie Shen, Daniel Harabor, Peter Stuckey, Mark Wallace  

**Link**: [PDF](https://arxiv.org/pdf/2505.14193)  

**Abstract**: Delays in public transport are common, often impacting users through prolonged travel times and missed transfers. Existing solutions for handling delays remain limited; backup plans based on historical data miss opportunities for earlier arrivals, while snapshot planning accounts for current delays but not future ones. With the growing availability of live delay data, users can adjust their journeys in real-time. However, the literature lacks a framework that fully exploits this advantage for system-scale dynamic replanning. To address this, we formalise the dynamic replanning problem in public transport routing and propose two solutions: a "pull" approach, where users manually request replanning, and a novel "push" approach, where the server proactively monitors and adjusts journeys. Our experiments show that the push approach outperforms the pull approach, achieving significant speedups. The results also reveal substantial arrival time savings enabled by dynamic replanning. 

**Abstract (ZH)**: 公共交通延误常见， Often impacting乘客通过延长的旅行时间和错过换乘。现有的延误处理解决方案仍然有限；基于历史数据的备用计划错过了 earlier到达的机会，而快照规划则考虑到当前的延误但不考虑未来的延误。随着实时延误数据的日益可用，用户可以实时调整他们的行程。然而，缺乏一个全面利用这一优势的大规模动态重新规划框架。为解决这一问题，我们正式化了公共交通路由中的动态重新规划问题，并提出了两种解决方案：一种是“拉”方法，用户手动请求重新规划，以及一种新颖的“推”方法，服务器主动监控并调整行程。我们的实验表明，推方法优于拉方法，实现了显著的速度提升。结果还表明，动态重新规划能够实现显著的到达时间节省。 

---
# DSMentor: Enhancing Data Science Agents with Curriculum Learning and Online Knowledge Accumulation 

**Title (ZH)**: DSMentor: 通过课程学习和在线知识积累增强数据科学代理 

**Authors**: He Wang, Alexander Hanbo Li, Yiqun Hu, Sheng Zhang, Hideo Kobayashi, Jiani Zhang, Henry Zhu, Chung-Wei Hang, Patrick Ng  

**Link**: [PDF](https://arxiv.org/pdf/2505.14163)  

**Abstract**: Large language model (LLM) agents have shown promising performance in generating code for solving complex data science problems. Recent studies primarily focus on enhancing in-context learning through improved search, sampling, and planning techniques, while overlooking the importance of the order in which problems are tackled during inference. In this work, we develop a novel inference-time optimization framework, referred to as DSMentor, which leverages curriculum learning -- a strategy that introduces simpler task first and progressively moves to more complex ones as the learner improves -- to enhance LLM agent performance in challenging data science tasks. Our mentor-guided framework organizes data science tasks in order of increasing difficulty and incorporates a growing long-term memory to retain prior experiences, guiding the agent's learning progression and enabling more effective utilization of accumulated knowledge. We evaluate DSMentor through extensive experiments on DSEval and QRData benchmarks. Experiments show that DSMentor using Claude-3.5-Sonnet improves the pass rate by up to 5.2% on DSEval and QRData compared to baseline agents. Furthermore, DSMentor demonstrates stronger causal reasoning ability, improving the pass rate by 8.8% on the causality problems compared to GPT-4 using Program-of-Thoughts prompts. Our work underscores the importance of developing effective strategies for accumulating and utilizing knowledge during inference, mirroring the human learning process and opening new avenues for improving LLM performance through curriculum-based inference optimization. 

**Abstract (ZH)**: 大型语言模型代理在生成解决复杂数据科学问题的代码方面表现出有前途的性能。近期研究主要侧重于通过改进搜索、采样和规划技术来增强上下文学习，而忽视了解题顺序对推断性能的影响。本文开发了一种新的推理时优化框架，称为DSMentor，该框架利用了循序渐进学习策略——先引入简单的任务，随着学习者能力的提高逐步过渡到更复杂的任务——以提高在挑战性数据科学任务上的大型语言模型代理性能。我们的导师引导框架按任务难度递增的顺序组织数据科学任务，并集成不断增长的长期记忆以保留先前的经验，引导代理的学习进程，并有效利用积累的知识。我们通过在DSEval和QRData基准上进行广泛实验来评估DSMentor。实验表明，使用Claude-3.5-Sonnet的DSMentor在DSEval和QRData上的通过率分别相较于基线代理提高了5.2%。此外，DSMentor在因果推理问题上的因果推理能力更强，相较于使用Program-of-Thoughts提示的GPT-4，通过率提高了8.8%。我们的工作强调了在推断过程中开发有效知识积累和利用策略的重要性，模拟了人类学习过程，并为通过基于课程的推理优化提升大型语言模型性能开辟了新的途径。 

---
# MM-Agent: LLM as Agents for Real-world Mathematical Modeling Problem 

**Title (ZH)**: MM-Agent: LLM作为解决现实数学建模问题的代理 

**Authors**: Fan Liu, Zherui Yang, Cancheng Liu, Tianrui Song, Xiaofeng Gao, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14148)  

**Abstract**: Mathematical modeling is a cornerstone of scientific discovery and engineering practice, enabling the translation of real-world problems into formal systems across domains such as physics, biology, and economics. Unlike mathematical reasoning, which assumes a predefined formulation, modeling requires open-ended problem analysis, abstraction, and principled formalization. While Large Language Models (LLMs) have shown strong reasoning capabilities, they fall short in rigorous model construction, limiting their utility in real-world problem-solving. To this end, we formalize the task of LLM-powered real-world mathematical modeling, where agents must analyze problems, construct domain-appropriate formulations, and generate complete end-to-end solutions. We introduce MM-Bench, a curated benchmark of 111 problems from the Mathematical Contest in Modeling (MCM/ICM), spanning the years 2000 to 2025 and across ten diverse domains such as physics, biology, and economics. To tackle this task, we propose MM-Agent, an expert-inspired framework that decomposes mathematical modeling into four stages: open-ended problem analysis, structured model formulation, computational problem solving, and report generation. Experiments on MM-Bench show that MM-Agent significantly outperforms baseline agents, achieving an 11.88\% improvement over human expert solutions while requiring only 15 minutes and \$0.88 per task using GPT-4o. Furthermore, under official MCM/ICM protocols, MM-Agent assisted two undergraduate teams in winning the Finalist Award (\textbf{top 2.0\% among 27,456 teams}) in MCM/ICM 2025, demonstrating its practical effectiveness as a modeling copilot. Our code is available at this https URL 

**Abstract (ZH)**: 数学建模是科学研究和工程实践的基础，能使现实世界的问题在物理学、生物学和经济学等领域转化为正式系统。与基于预定义形式化的数学推理不同，建模需要开放性问题分析、抽象化和 principled 形式化。虽然大型语言模型（LLMs）展示了强大的推理能力，但在严谨的模型构建方面仍存在不足，限制了其在解决实际问题中的应用。为此，我们正式定义了 LLM 驱动的现实世界数学建模任务，要求代理分析问题、构建适用领域的正式化表述，并生成端到端的解决方案。我们引入了 MM-Bench，这是一个由数学建模竞赛（MCM/ICM）中的111个问题组成的精选基准，涵盖2000年至2025年间的十个不同领域，如物理学、生物学和经济学。为了解决这一任务，我们提出了 MM-Agent，一个基于专家启发的框架，将数学建模分解为四个阶段：开放性问题分析、结构化建模表述、计算问题求解以及报告生成。MM-Bench 上的实验显示，MM-Agent 在基线代理上取得了显著的优势，相较于人类专家解决方案，其性能提高了11.88%，使用GPT-4o完成每任务仅需15分钟和0.88美元。此外，在官方MCM/ICM协议下，MM-Agent 协助两个本科生团队赢得了2025年MCM/ICM的最终奖（27,456支队伍中的前2.0%），证明了其作为建模协作者的实际有效性。相关代码可在此链接获取。 

---
# SHARP: Synthesizing High-quality Aligned Reasoning Problems for Large Reasoning Models Reinforcement Learning 

**Title (ZH)**: SHARP: 综合高质量对齐推理问题以强化大型推理模型的训练 

**Authors**: Xiong Jun Wu, Zhenduo Zhang, ZuJie Wen, Zhiqiang Zhang, Wang Ren, Lei Shi, Cai Chen, Deng Zhao, Dingnan Jin, Qing Cui, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.14147)  

**Abstract**: Training large reasoning models (LRMs) with reinforcement learning in STEM domains is hindered by the scarcity of high-quality, diverse, and verifiable problem sets. Existing synthesis methods, such as Chain-of-Thought prompting, often generate oversimplified or uncheckable data, limiting model advancement on complex tasks. To address these challenges, we introduce SHARP, a unified approach to Synthesizing High-quality Aligned Reasoning Problems for LRMs reinforcement learning with verifiable rewards (RLVR). SHARP encompasses a strategic set of self-alignment principles -- targeting graduate and Olympiad-level difficulty, rigorous logical consistency, and unambiguous, verifiable answers -- and a structured three-phase framework (Alignment, Instantiation, Inference) that ensures thematic diversity and fine-grained control over problem generation. We implement SHARP by leveraging a state-of-the-art LRM to infer and verify challenging STEM questions, then employ a reinforcement learning loop to refine the model's reasoning through verifiable reward signals. Experiments on benchmarks such as GPQA demonstrate that SHARP-augmented training substantially outperforms existing methods, markedly improving complex reasoning accuracy and pushing LRM performance closer to expert-level proficiency. Our contributions include the SHARP strategy, framework design, end-to-end implementation, and experimental evaluation of its effectiveness in elevating LRM reasoning capabilities. 

**Abstract (ZH)**: 使用强化学习训练大型推理模型（LRMs）在STEM领域受到高质量、多样性和可验证问题集稀缺性的阻碍。现有的合成方法，如链式思维提示，通常生成过于简化或不可验证的数据，限制了模型在复杂任务上的进步。为了解决这些挑战，我们引入了SHARP，一种用于大型推理模型强化学习的统一方法，该方法结合了验证奖励（RLVR）的高质量对齐推理问题合成方法。SHARP 包含一系列策略——旨在针对研究生和奥林匹克级别难度、严格逻辑一致性和明确可验证的答案，并采用结构化的三阶段框架（对齐、实例化、推理），以确保问题生成的题型多样性和精细控制。我们通过利用最新最先进的LRM来推断和验证具有挑战性的STEM问题，然后使用强化学习循环并通过可验证的奖励信号进一步精炼模型的推理能力。在GPQA等基准上的实验表明，SHARP增强的训练显著优于现有方法，显著提高了复杂推理的准确性，并将LRM性能提升至专家级水平。我们的贡献包括SHARP策略、框架设计、端到端实现及其有效性实验评估。 

---
# s3: You Don't Need That Much Data to Train a Search Agent via RL 

**Title (ZH)**: 三: 通过强化学习训练搜索代理不需要太多数据 

**Authors**: Pengcheng Jiang, Xueqiang Xu, Jiacheng Lin, Jinfeng Xiao, Zifeng Wang, Jimeng Sun, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.14146)  

**Abstract**: Retrieval-augmented generation (RAG) systems empower large language models (LLMs) to access external knowledge during inference. Recent advances have enabled LLMs to act as search agents via reinforcement learning (RL), improving information acquisition through multi-turn interactions with retrieval engines. However, existing approaches either optimize retrieval using search-only metrics (e.g., NDCG) that ignore downstream utility or fine-tune the entire LLM to jointly reason and retrieve-entangling retrieval with generation and limiting the real search utility and compatibility with frozen or proprietary models. In this work, we propose s3, a lightweight, model-agnostic framework that decouples the searcher from the generator and trains the searcher using a Gain Beyond RAG reward: the improvement in generation accuracy over naive RAG. s3 requires only 2.4k training samples to outperform baselines trained on over 70x more data, consistently delivering stronger downstream performance across six general QA and five medical QA benchmarks. 

**Abstract (ZH)**: 基于检索增强生成的检索分离框架(s3) empowering大型语言模型在推理时访问外部知识。现有方法要么使用仅搜索的指标（如NDCG）来优化检索，忽略下游实用性，要么 Fine-tune 整个大型语言模型以联合推理和检索，这会混淆检索与生成并限制实际搜索实用性和与冻结或专有模型的兼容性。我们的工作提出了 s3，一个轻量级且模型无关的框架，将检索与生成分离，并使用超越 RAG 的奖励对检索器进行训练：生成准确性上的提升。s3 仅需 2400 个训练样本即可在数据量超过其 70 倍的基线上表现出更优的下游性能，在六个通用 QA 和五个医学 QA 标准测试中都能持续提供更强大更稳定的性能。 

---
# Multimodal Mixture of Low-Rank Experts for Sentiment Analysis and Emotion Recognition 

**Title (ZH)**: 多模态低秩专家混合模型用于情感分析和情绪识别 

**Authors**: Shuo Zhang, Jinsong Zhang, Zhejun Zhang, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.14143)  

**Abstract**: Multi-task learning (MTL) enables the efficient transfer of extra knowledge acquired from other tasks. The high correlation between multimodal sentiment analysis (MSA) and multimodal emotion recognition (MER) supports their joint training. However, existing methods primarily employ hard parameter sharing, ignoring parameter conflicts caused by complex task correlations. In this paper, we present a novel MTL method for MSA and MER, termed Multimodal Mixture of Low-Rank Experts (MMoLRE). MMoLRE utilizes shared and task-specific experts to distinctly model common and unique task characteristics, thereby avoiding parameter conflicts. Additionally, inspired by low-rank structures in the Mixture of Experts (MoE) framework, we design low-rank expert networks to reduce parameter and computational overhead as the number of experts increases. Extensive experiments on the CMU-MOSI and CMU-MOSEI benchmarks demonstrate that MMoLRE achieves state-of-the-art performance on the MSA task and competitive results on the MER task. 

**Abstract (ZH)**: 多任务学习（MTL）使从其他任务中获得的额外知识能够有效转移。多模态情感分析（MSA）与多模态情绪识别（MER）之间的高相关性支持它们的联合训练。然而，现有方法主要采用硬参数共享，忽视了由复杂任务相关性引起的参数冲突。在本文中，我们提出了一种用于MSA和MER的新型MTL方法，称为多模态低秩专家混合（MMoLRE）。MMoLRE利用共享专家和任务特定专家来分别建模共同和独特任务特性，从而避免参数冲突。此外，受Experts混合（MoE）框架中低秩结构的启发，我们设计了低秩专家网络，以减少随着专家数量增加而导致的参数和计算开销。在CMU-MOSI和CMU-MOSEI基准上的广泛实验表明，MMoLRE在MSA任务上达到了最先进的性能，并在MER任务上取得了竞争力的结果。 

---
# Building a Stable Planner: An Extended Finite State Machine Based Planning Module for Mobile GUI Agent 

**Title (ZH)**: 基于扩展有限状态机的稳定规划模块：移动GUI代理的规划实现 

**Authors**: Fanglin Mo, Junzhe Chen, Haoxuan Zhu, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14141)  

**Abstract**: Mobile GUI agents execute user commands by directly interacting with the graphical user interface (GUI) of mobile devices, demonstrating significant potential to enhance user convenience. However, these agents face considerable challenges in task planning, as they must continuously analyze the GUI and generate operation instructions step by step. This process often leads to difficulties in making accurate task plans, as GUI agents lack a deep understanding of how to effectively use the target applications, which can cause them to become "lost" during task execution. To address the task planning issue, we propose SPlanner, a plug-and-play planning module to generate execution plans that guide vision language model(VLMs) in executing tasks. The proposed planning module utilizes extended finite state machines (EFSMs) to model the control logits and configurations of mobile applications. It then decomposes a user instruction into a sequence of primary function modeled in EFSMs, and generate the execution path by traversing the EFSMs. We further refine the execution path into a natural language plan using an LLM. The final plan is concise and actionable, and effectively guides VLMs to generate interactive GUI actions to accomplish user tasks. SPlanner demonstrates strong performance on dynamic benchmarks reflecting real-world mobile usage. On the AndroidWorld benchmark, SPlanner achieves a 63.8% task success rate when paired with Qwen2.5-VL-72B as the VLM executor, yielding a 28.8 percentage point improvement compared to using Qwen2.5-VL-72B without planning assistance. 

**Abstract (ZH)**: Mobile GUI代理通过直接与移动设备的图形用户界面（GUI）交互来执行用户命令，显示出显著提升用户便利性的潜力。然而，这些代理在任务规划方面面临重大挑战，因为它们需要不断分析GUI并逐步生成操作指令。这一过程往往会导致难以做出准确的任务规划，因为GUI代理缺乏对如何有效使用目标应用程序的深刻理解，这可能导致它们在任务执行过程中“迷失”。为了解决任务规划问题，我们提出SPlanner，这是一种插即用的规划模块，用于生成指导视觉语言模型(VLMs)执行任务的执行计划。提出的规划模块利用扩展的有限状态机（EFSMs）来建模移动应用程序的控制概率和配置。然后，它将用户指令分解成序列的基本功能模型，并通过遍历EFSMs生成执行路径。我们进一步使用大规模语言模型（LLM）对执行路径进行细化，以自然语言形式生成最终计划。该计划简洁且具有操作性，有效地指导VLMs生成交互式GUI操作以完成用户任务。SPlanner在反映现实移动使用情况的动态基准测试中表现出强大的性能。在AndroidWorld基准测试中，当与Qwen2.5-VL-72B作为VLM执行器配对时，SPlanner的任务成功率达到了63.8%，比使用无规划辅助的Qwen2.5-VL-72B提高了28.8个百分点。 

---
# RL of Thoughts: Navigating LLM Reasoning with Inference-time Reinforcement Learning 

**Title (ZH)**: 思维的RL：基于推理时强化学习的LLM推理导航 

**Authors**: Qianyue Hao, Sibo Li, Jian Yuan, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.14140)  

**Abstract**: Despite rapid advancements in large language models (LLMs), the token-level autoregressive nature constrains their complex reasoning capabilities. To enhance LLM reasoning, inference-time techniques, including Chain/Tree/Graph-of-Thought(s), successfully improve the performance, as they are fairly cost-effective by guiding reasoning through sophisticated logical structures without modifying LLMs' parameters. However, these manually predefined, task-agnostic frameworks are applied uniformly across diverse tasks, lacking adaptability. To improve this, we propose RL-of-Thoughts (RLoT), where we train a lightweight navigator model with reinforcement learning (RL) to adaptively enhance LLM reasoning at inference time. Specifically, we design five basic logic blocks from the perspective of human cognition. During the reasoning process, the trained RL navigator dynamically selects the suitable logic blocks and combines them into task-specific logical structures according to problem characteristics. Experiments across multiple reasoning benchmarks (AIME, MATH, GPQA, etc.) with multiple LLMs (GPT, Llama, Qwen, and DeepSeek) illustrate that RLoT outperforms established inference-time techniques by up to 13.4%. Remarkably, with less than 3K parameters, our RL navigator is able to make sub-10B LLMs comparable to 100B-scale counterparts. Moreover, the RL navigator demonstrates strong transferability: a model trained on one specific LLM-task pair can effectively generalize to unseen LLMs and tasks. Our code is open-source at this https URL for reproducibility. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）取得了rapid advancements，其token级别的自回归性质限制了它们的复杂推理能力。为增强LLM推理，通过推理时的技术，如链式/树状/图状思维，显著改善了性能，因为这些技术通过引导复杂的逻辑结构进行推理，成本效益高，而不修改LLM的参数。然而，这些手动预定义的、任务无关的框架在不同任务中应用时缺乏适应性。为改进这一问题，我们提出了RL-of-Thoughts (RLoT)，其中通过强化学习（RL）训练一个轻量级导向模型，在推理时自适应地增强LLM的推理能力。具体而言，我们从人类认知的角度设计了五个基本逻辑模块。在推理过程中，训练好的RL导向模型动态选择合适的逻辑模块，并根据问题特点组合成任务特定的逻辑结构。多轮推理基准实验（AIME、MATH、GPQA等）和多种LLM（GPT、Llama、Qwen、DeepSeek）表明，RLoT相比现有推理时技术可提高多达13.4%的性能。值得注意的是，使用不到3K参数，我们的RL导向模型能使小于10B参数的LLM与100B级别的模型相当。此外，RL导向模型展示了强大的迁移能力：一个模型在一个特定的LLM任务对上训练后，能够有效泛化到未见过的LLM和任务。我们的代码已开源，以便再现结果。 

---
# Memory Assignment for Finite-Memory Strategies in Adversarial Patrolling Games 

**Title (ZH)**: 有限记忆策略下的巡逻博弈中内存分配 

**Authors**: Vojtěch Kůr, Vít Musil, Vojtěch Řehák  

**Link**: [PDF](https://arxiv.org/pdf/2505.14137)  

**Abstract**: Adversarial Patrolling games form a subclass of Security games where a Defender moves between locations, guarding vulnerable targets. The main algorithmic problem is constructing a strategy for the Defender that minimizes the worst damage an Attacker can cause. We focus on the class of finite-memory (also known as regular) Defender's strategies that experimentally outperformed other competing classes. A finite-memory strategy can be seen as a positional strategy on a finite set of states. Each state consists of a pair of a location and a certain integer value--called memory. Existing algorithms improve the transitional probabilities between the states but require that the available memory size itself is assigned at each location manually. Choosing the right memory assignment is a well-known open and hard problem that hinders the usability of finite-memory strategies. We solve this issue by developing a general method that iteratively changes the memory assignment. Our algorithm can be used in connection with \emph{any} black-box strategy optimization tool. We evaluate our method on various experiments and show its robustness by solving instances of various patrolling models. 

**Abstract (ZH)**: adversarial patrolling博弈属于一类安全博弈，其中防守者在不同位置之间移动以保护易受攻击的目标。主要的算法问题是构造一个防守者策略，以最小化攻击者可能造成的最坏伤害。我们专注于一类有限记忆（也称为正规）防守者策略，这些策略在实验中表现优于其他竞争类策略。有限记忆策略可以视为在有限状态集上的一种位置策略。每个状态由一个位置和一个特定的整数值（称为记忆）组成。现有算法通过改进状态间的转移概率，但要求在每个位置手动分配可用的记忆大小。正确选择记忆分配是一个已知的开放且困难的问题，阻碍了有限记忆策略的实用性。我们通过开发一种迭代改变记忆分配的通用方法解决了这一问题。我们的算法可以与任何黑盒策略优化工具结合使用。我们在各种实验中评估了该方法，并通过解决不同巡护模型的实例来证明其稳健性。 

---
# Personalized Student Knowledge Modeling for Future Learning Resource Prediction 

**Title (ZH)**: 未来学习资源预测的个性化学生知识建模 

**Authors**: Soroush Hashemifar, Sherry Sahebi  

**Link**: [PDF](https://arxiv.org/pdf/2505.14072)  

**Abstract**: Despite advances in deep learning for education, student knowledge tracing and behavior modeling face persistent challenges: limited personalization, inadequate modeling of diverse learning activities (especially non-assessed materials), and overlooking the interplay between knowledge acquisition and behavioral patterns. Practical limitations, such as fixed-size sequence segmentation, frequently lead to the loss of contextual information vital for personalized learning. Moreover, reliance on student performance on assessed materials limits the modeling scope, excluding non-assessed interactions like lectures. To overcome these shortcomings, we propose Knowledge Modeling and Material Prediction (KMaP), a stateful multi-task approach designed for personalized and simultaneous modeling of student knowledge and behavior. KMaP employs clustering-based student profiling to create personalized student representations, improving predictions of future learning resource preferences. Extensive experiments on two real-world datasets confirm significant behavioral differences across student clusters and validate the efficacy of the KMaP model. 

**Abstract (ZH)**: 尽管深度学习在教育领域的进展显著，但学生的知识追踪和行为建模仍然面临持续的挑战：个性化能力有限、对多样化的学习活动（尤其是未评估材料）建模不足，以及忽略知识获取与行为模式之间的相互作用。实际限制，如固定大小序列的分割，往往会导致对个性化学习至关重要的上下文信息丢失。此外，依赖于学生在已评估材料上的表现限制了建模范围，排除了如讲座等非评估互动。为克服这些不足，我们提出了一种状态依赖的多任务方法——Knowledge Modeling and Material Prediction（KMaP），旨在同时实现个性化的学生知识和行为建模。KMaP 通过基于聚类的学生画像生成个性化的学生表示，从而改进对未来学习资源偏好的预测。在两个真实世界数据集上的广泛实验验证了学生集群间显著的行为差异，并证实了KMaP模型的有效性。 

---
# ProMind-LLM: Proactive Mental Health Care via Causal Reasoning with Sensor Data 

**Title (ZH)**: ProMind-LLM：基于传感器数据的因果推理前馈心理健康护理 

**Authors**: Xinzhe Zheng, Sijie Ji, Jiawei Sun, Renqi Chen, Wei Gao, Mani Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2505.14038)  

**Abstract**: Mental health risk is a critical global public health challenge, necessitating innovative and reliable assessment methods. With the development of large language models (LLMs), they stand out to be a promising tool for explainable mental health care applications. Nevertheless, existing approaches predominantly rely on subjective textual mental records, which can be distorted by inherent mental uncertainties, leading to inconsistent and unreliable predictions. To address these limitations, this paper introduces ProMind-LLM. We investigate an innovative approach integrating objective behavior data as complementary information alongside subjective mental records for robust mental health risk assessment. Specifically, ProMind-LLM incorporates a comprehensive pipeline that includes domain-specific pretraining to tailor the LLM for mental health contexts, a self-refine mechanism to optimize the processing of numerical behavioral data, and causal chain-of-thought reasoning to enhance the reliability and interpretability of its predictions. Evaluations of two real-world datasets, PMData and Globem, demonstrate the effectiveness of our proposed methods, achieving substantial improvements over general LLMs. We anticipate that ProMind-LLM will pave the way for more dependable, interpretable, and scalable mental health case solutions. 

**Abstract (ZH)**: 心理健康风险是一个关键的全球公共卫生挑战，需要创新可靠的评估方法。随着大规模语言模型（LLMs）的发展，它们在可解释的心理健康护理应用中展现出巨大潜力。然而，现有方法主要依赖主观的心理文本记录，这些记录可能会受到内在心理不确定性的扭曲，导致不一致和不可靠的预测。为了解决这些局限性，本文介绍了一种新的方法——ProMind-LLM。我们研究了一种创新的方法，将客观行为数据作为补充信息与主观心理记录相结合，以实现稳健的心理健康风险评估。具体而言，ProMind-LLM 包含一个全面的工作流，包括特定领域的预训练以适应心理健康情境，自反馈机制以优化数值行为数据的处理，以及因果链式推理以提高其预测的可靠性和可解释性。对两个实际数据集（PMData 和 Globem）的评估表明，所提出的方法优于通用的语言模型，取得了显著改进。我们期待 ProMind-LLM 能够为更可靠、可解释和可扩展的心理健康案例解决方案铺平道路。 

---
# Disentangled Multi-span Evolutionary Network against Temporal Knowledge Graph Reasoning 

**Title (ZH)**: 解耦多跨度演化网络对抗时间知识图谱推理 

**Authors**: Hao Dong, Ziyue Qiao, Zhiyuan Ning, Qi Hao, Yi Du, Pengyang Wang, Yuanchun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.14020)  

**Abstract**: Temporal Knowledge Graphs (TKGs), as an extension of static Knowledge Graphs (KGs), incorporate the temporal feature to express the transience of knowledge by describing when facts occur. TKG extrapolation aims to infer possible future facts based on known history, which has garnered significant attention in recent years. Some existing methods treat TKG as a sequence of independent subgraphs to model temporal evolution patterns, demonstrating impressive reasoning performance. However, they still have limitations: 1) In modeling subgraph semantic evolution, they usually neglect the internal structural interactions between subgraphs, which are actually crucial for encoding TKGs. 2) They overlook the potential smooth features that do not lead to semantic changes, which should be distinguished from the semantic evolution process. Therefore, we propose a novel Disentangled Multi-span Evolutionary Network (DiMNet) for TKG reasoning. Specifically, we design a multi-span evolution strategy that captures local neighbor features while perceiving historical neighbor semantic information, thus enabling internal interactions between subgraphs during the evolution process. To maximize the capture of semantic change patterns, we design a disentangle component that adaptively separates nodes' active and stable features, used to dynamically control the influence of historical semantics on future evolution. Extensive experiments conducted on four real-world TKG datasets show that DiMNet demonstrates substantial performance in TKG reasoning, and outperforms the state-of-the-art up to 22.7% in MRR. 

**Abstract (ZH)**: 时态知识图谱（TKGs）作为静态知识图谱（KGs）的扩展，通过描述事实发生的时间来体现知识的时效性。时态知识图谱外推旨在基于已知的历史事实推断可能的未来事实，近年来引起了广泛关注。现有的一些方法将时态知识图谱视为独立子图序列以建模时间演化模式，表现出令人印象深刻的推理性能。然而，它们仍然存在局限性：1）在建模子图语义演化时，通常忽视了子图之间的内部结构交互，这是编码时态知识图谱的关键。2）忽略了可能的平滑特征，这些特征不会引发语义变化，应在语义演化过程中与之区分开。因此，我们提出了一种新颖的解耦多跨度演化网络（DiMNet）用于时态知识图谱推理。具体而言，我们设计了一种多跨度演化策略，既能捕获局部邻居特征又能感知历史邻居语义信息，在演化过程中促进子图之间的内部交互。为了最大化捕捉语义变化模式，我们设计了一个解耦组件，该组件能够自适应地分离节点的活跃和稳定特征，用于动态控制历史语义对未来演化的影响力。在四个真实世界时态知识图谱数据集上的广泛实验表明，DiMNet在时态知识图谱推理中展现出显著的性能，并在MRR指标上优于现有最佳方法多达22.7%。 

---
# VeRecycle: Reclaiming Guarantees from Probabilistic Certificates for Stochastic Dynamical Systems after Change 

**Title (ZH)**: VeRecycle: 变更后从随机证书中回收随机动力学系统保证技术 

**Authors**: Sterre Lutz, Matthijs T.J. Spaan, Anna Lukina  

**Link**: [PDF](https://arxiv.org/pdf/2505.14001)  

**Abstract**: Autonomous systems operating in the real world encounter a range of uncertainties. Probabilistic neural Lyapunov certification is a powerful approach to proving safety of nonlinear stochastic dynamical systems. When faced with changes beyond the modeled uncertainties, e.g., unidentified obstacles, probabilistic certificates must be transferred to the new system dynamics. However, even when the changes are localized in a known part of the state space, state-of-the-art requires complete re-certification, which is particularly costly for neural certificates. We introduce VeRecycle, the first framework to formally reclaim guarantees for discrete-time stochastic dynamical systems. VeRecycle efficiently reuses probabilistic certificates when the system dynamics deviate only in a given subset of states. We present a general theoretical justification and algorithmic implementation. Our experimental evaluation shows scenarios where VeRecycle both saves significant computational effort and achieves competitive probabilistic guarantees in compositional neural control. 

**Abstract (ZH)**: 自主系统在真实世界中的运行会遇到各种不确定性。概率神经李雅普诺夫认证是证明非线性随机动力学系统安全性的强大方法。在遇到超出建模不确定性范围的变化时，例如未识别的障碍物，必须将概率证书转移到新的系统动力学中。即使变化局限于已知状态空间的一部分，现有的最先进的方法仍需要完全重新认证，这对神经证书尤其昂贵。我们介绍了VeRecycle，这是第一个正式回收离散时间随机动力学系统保证的框架。VeRecycle在系统动力学仅在给定的状态子集发生变化时，能够高效地重用概率证书。我们提供了一般性的理论依据和算法实现。实验评估显示，在组合神经控制中，VeRecycle既能节省显著的计算资源，又能实现具有竞争力的概率保证。 

---
# Divide by Question, Conquer by Agent: SPLIT-RAG with Question-Driven Graph Partitioning 

**Title (ZH)**: 按照问题划分，由代理征服：基于问题驱动的图划分的SPLIT-RAG 

**Authors**: Ruiyi Yang, Hao Xue, Imran Razzak, Hakim Hacid, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2505.13994)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems empower large language models (LLMs) with external knowledge, yet struggle with efficiency-accuracy trade-offs when scaling to large knowledge graphs. Existing approaches often rely on monolithic graph retrieval, incurring unnecessary latency for simple queries and fragmented reasoning for complex multi-hop questions. To address these challenges, this paper propose SPLIT-RAG, a multi-agent RAG framework that addresses these limitations with question-driven semantic graph partitioning and collaborative subgraph retrieval. The innovative framework first create Semantic Partitioning of Linked Information, then use the Type-Specialized knowledge base to achieve Multi-Agent RAG. The attribute-aware graph segmentation manages to divide knowledge graphs into semantically coherent subgraphs, ensuring subgraphs align with different query types, while lightweight LLM agents are assigned to partitioned subgraphs, and only relevant partitions are activated during retrieval, thus reduce search space while enhancing efficiency. Finally, a hierarchical merging module resolves inconsistencies across subgraph-derived answers through logical verifications. Extensive experimental validation demonstrates considerable improvements compared to existing approaches. 

**Abstract (ZH)**: 基于检索增强生成的多agent框架SPLIT-RAG：基于语义图分割的协作子图检索 

---
# Solving Normalized Cut Problem with Constrained Action Space 

**Title (ZH)**: 在受限动作空间中求解归一化切分问题 

**Authors**: Qize Jiang, Linsey Pang, Alice Gatti, Mahima Aggarwa, Giovanna Vantin, Xiaosong Ma, Weiwei Sun, Sanjay Chawla  

**Link**: [PDF](https://arxiv.org/pdf/2505.13986)  

**Abstract**: Reinforcement Learning (RL) has emerged as an important paradigm to solve combinatorial optimization problems primarily due to its ability to learn heuristics that can generalize across problem instances. However, integrating external knowledge that will steer combinatorial optimization problem solutions towards domain appropriate outcomes remains an extremely challenging task. In this paper, we propose the first RL solution that uses constrained action spaces to guide the normalized cut problem towards pre-defined template instances. Using transportation networks as an example domain, we create a Wedge and Ring Transformer that results in graph partitions that are shaped in form of Wedges and Rings and which are likely to be closer to natural optimal partitions. However, our approach is general as it is based on principles that can be generalized to other domains. 

**Abstract (ZH)**: 强化学习（RL）由于其在学习能够跨问题实例泛化的启发式方法方面的能力，已成为解决组合优化问题的重要范式。然而，将外部知识整合到组合优化问题解决方案中以引导其实现领域适当的成果仍然是一项极其具有挑战性的工作。在本文中，我们提出了第一个使用约束动作空间来引导最小割问题向预定义模板实例方向的RL解决方案。以运输网络为例，我们创建了一个楔形和环形转换器，生成的图分区呈现出楔形和环形的形状，并且更接近于自然最优分区。然而，我们的方法是通用的，因为它基于可以泛化到其他领域的原则。 

---
# Visual Instruction Bottleneck Tuning 

**Title (ZH)**: 视觉指令瓶颈调整 

**Authors**: Changdae Oh, Jiatong Li, Shawn Im, Yixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.13946)  

**Abstract**: Despite widespread adoption, multimodal large language models (MLLMs) suffer performance degradation when encountering unfamiliar queries under distribution shifts. Existing methods to improve MLLM generalization typically require either more instruction data or larger advanced model architectures, both of which incur non-trivial human labor or computational costs. In this work, we take an alternative approach to enhance the robustness of MLLMs under distribution shifts, from a representation learning perspective. Inspired by the information bottleneck (IB) principle, we derive a variational lower bound of the IB for MLLMs and devise a practical implementation, Visual Instruction Bottleneck Tuning (Vittle). We then provide a theoretical justification of Vittle by revealing its connection to an information-theoretic robustness metric of MLLM. Empirical validation of three MLLMs on open-ended and closed-form question answering and object hallucination detection tasks over 45 datasets, including 30 shift scenarios, demonstrates that Vittle consistently improves the MLLM's robustness under shifts by pursuing the learning of a minimal sufficient representation. 

**Abstract (ZH)**: 尽管广泛应用，多模态大型语言模型（MLLMs）在分布变化下遇到不熟悉查询时会出现性能下降。现有的提升MLLM泛化能力的方法通常需要更多的指令数据或更大的先进模型架构，这两种方法都会产生非琐碎的人力或计算成本。在这项工作中，我们从表示学习的角度出发，采取一种不同的方法来增强MLLMs在分布变化下的鲁棒性。受信息瓶颈（IB）原则的启发，我们为MLLMs推导出了IB的变分下界，并设计了一个实用的实现方法，即视觉指令瓶颈调整（Vittle）。然后，我们通过揭示Vittle与MLLM的信息论鲁棒性度量之间的联系，为其提供理论依据。在45个数据集上对三种MLLMs进行开放性和封闭性问题回答以及物体幻觉检测任务的实证验证，包括30种分布变化场景，表明Vittle通过追求学习最小充分表示来一致地提高了MLLMs在分布变化下的鲁棒性。 

---
# DrugPilot: LLM-based Parameterized Reasoning Agent for Drug Discovery 

**Title (ZH)**: DrugPilot：基于LLM的参数化 reasoning代理在药物发现中的应用 

**Authors**: Kun Li, Zhennan Wu, Shoupeng Wang, Wenbin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13940)  

**Abstract**: In the field of AI4Science, large-scale language models (LLMs) show great potential to parse complex scientific semantics, integrate cross-disciplinary knowledge, and assist critical task research. However, in the field of drug discovery, despite the optimization through professional data pre-training, context window expansion, and internet search, the existing LLMs are still facing challenges such as massive multi-modal and heterogeneous data processing, domain knowledge dynamic updating delay, and insufficient confidence in predicting the results of complex computational tasks. To address these challenges, we propose the DrugPilot, an LLM-based agent with parameterized reasoning for drug discovery. DrugPilot addresses key limitations of traditional end-to-end LLM prediction approaches through its parametric inference architecture. This agent system supports major phases of the drug discovery pipeline, facilitating automated planning and execution of multi-stage research tasks. To address the critical challenge of multi-modal drug data analysis (incorporating both public datasets and user-submitted data), we developed an interactive parameterized memory pool. This innovative component standardizes real-world drug data into parametric representations, simultaneously enabling efficient knowledge retrieval in multi-turn dialogue while mitigating the information loss inherent in text-based data transmission. Additionally, we created a drug instruct dataset across 8 essential drug discovery tasks for model fine-tuning and evaluation. Based on the Berkeley function calling evaluation framework, DrugPilot demonstrated the most advanced tool calling capabilities on our drug discovery tool instruction dataset, outperforming existing agents (e.g., ReAct, LoT). Specifically, it achieves task completion rates of 98.0%, 93.5%, and 64.0% on simple, multiple, and multi-turn tasks, respectively. 

**Abstract (ZH)**: 在AI4Science领域，大规模语言模型（LLMs）在解析复杂科学语义、整合跨学科知识以及辅助关键任务研究方面展现出巨大潜力。然而，在药物发现领域，尽管通过专业数据预训练、上下文窗口扩展和互联网搜索进行了优化，现有LLMs仍然面临多模态和异构数据处理规模庞大、领域知识动态更新延迟以及在复杂计算任务预测中信心不足等挑战。为应对这些挑战，我们提出了一种基于参数推理的LLM代理DrugPilot，用于药物发现。DrugPilot通过其参数推理架构，解决了传统端到端LLM预测方法中的关键局限性。该代理系统支持药物发现管道的主要阶段，促进多阶段研究任务的自动化规划与执行。为解决多模态药物数据分析的关键挑战（结合公共数据集和用户提交数据），我们开发了一个交互式的参数化记忆池。这一创新组件将现实世界中的药物数据标准化为参数表示，同时在多轮对话中高效检索知识，缓解基于文本的数据传输固有的信息损失。此外，我们为模型的微调和评估创建了一个横跨8个关键药物发现任务的数据集Drug instruct。基于伯克利函数调用评估框架，DrugPilot在药物发现工具指令数据集上展示了最先进的工具调用能力，优于现有代理（如ReAct、LoT）。具体而言，在简单任务、多任务和多轮任务中，分别实现了98.0%、93.5%和64.0%的任务完成率。 

---
# Parallel Belief Revision via Order Aggregation 

**Title (ZH)**: 并行信念修订通过顺序聚合 

**Authors**: Jake Chandler, Richard Booth  

**Link**: [PDF](https://arxiv.org/pdf/2505.13914)  

**Abstract**: Despite efforts to better understand the constraints that operate on single-step parallel (aka "package", "multiple") revision, very little work has been carried out on how to extend the model to the iterated case. A recent paper by Delgrande & Jin outlines a range of relevant rationality postulates. While many of these are plausible, they lack an underlying unifying explanation. We draw on recent work on iterated parallel contraction to offer a general method for extending serial iterated belief revision operators to handle parallel change. This method, based on a family of order aggregators known as TeamQueue aggregators, provides a principled way to recover the independently plausible properties that can be found in the literature, without yielding the more dubious ones. 

**Abstract (ZH)**: 尽管对单步并行（亦称“包”、“多次”）修订的操作约束已有一定理解，但在扩展模型至迭代情形方面的工作仍然很少。德尔格兰德与金近期的一篇论文概述了一系列相关理性公理。尽管这些公理中的许多具有合理性，但缺乏一个基础性的统一解释。我们借鉴迭代并行收缩的最近研究成果，提出了一种一般方法，用于将串行迭代信念修订算子扩展为处理并行变化。该方法基于一类称为TeamQueue聚合器的聚合器家族，提供了一种有原则的方法来恢复文献中独立合理的属性，同时避免获得更为可疑的属性。 

---
# Efficient Agent Training for Computer Use 

**Title (ZH)**: 计算机使用中的高效代理训练 

**Authors**: Yanheng He, Jiahe Jin, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13909)  

**Abstract**: Scaling up high-quality trajectory data has long been a critical bottleneck for developing human-like computer use agents. We introduce PC Agent-E, an efficient agent training framework that significantly reduces reliance on large-scale human demonstrations. Starting with just 312 human-annotated computer use trajectories, we further improved data quality by synthesizing diverse action decisions with Claude 3.7 Sonnet. Trained on these enriched trajectories, our PC Agent-E model achieved a remarkable 141% relative improvement, surpassing the strong Claude 3.7 Sonnet with extended thinking on WindowsAgentArena-V2, an improved benchmark we also released. Furthermore, PC Agent-E demonstrates strong generalizability to different operating systems on OSWorld. Our findings suggest that strong computer use capabilities can be stimulated from a small amount of high-quality trajectory data. 

**Abstract (ZH)**: 提高高质量轨迹数据规模一直是开发类人类计算机使用代理的关键瓶颈。我们引入了PC Agent-E，这是一种高效的代理训练框架，显著减少了对大规模人工演示的依赖。从仅有312个人标注的计算机使用轨迹出发，我们进一步通过Claude 3.7 Sonnet合成了多样化的动作决策以提高数据质量。基于这些丰富化的轨迹数据，我们的PC Agent-E模型取得了令人瞩目的141%的相对改进，超过了WindowsAgentArena-V2增强思考基准上的Claude 3.7 Sonnet，该基准我们亦已发布。此外，PC Agent-E在OSWorld上展示了很强的跨操作系统的一般性。我们的研究结果表明，强大的计算机使用能力可以从少量高质量的轨迹数据中激发出来。 

---
# Mobile-Agent-V: A Video-Guided Approach for Effortless and Efficient Operational Knowledge Injection in Mobile Automation 

**Title (ZH)**: 移动代理-V：一种基于视频的轻松高效操作知识注入方法在移动自动化中的应用 

**Authors**: Junyang Wang, Haiyang Xu, Xi Zhang, Ming Yan, Ji Zhang, Fei Huang, Jitao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13887)  

**Abstract**: The exponential rise in mobile device usage necessitates streamlined automation for effective task management, yet many AI frameworks fall short due to inadequate operational expertise. While manually written knowledge can bridge this gap, it is often burdensome and inefficient. We introduce Mobile-Agent-V, an innovative framework that utilizes video as a guiding tool to effortlessly and efficiently inject operational knowledge into mobile automation processes. By deriving knowledge directly from video content, Mobile-Agent-V eliminates manual intervention, significantly reducing the effort and time required for knowledge acquisition. To rigorously evaluate this approach, we propose Mobile-Knowledge, a benchmark tailored to assess the impact of external knowledge on mobile agent performance. Our experimental findings demonstrate that Mobile-Agent-V enhances performance by 36% compared to existing methods, underscoring its effortless and efficient advantages in mobile automation. 

**Abstract (ZH)**: 移动设备使用量的指数增长 necessitates 精简自动化以有效进行任务管理，但许多AI框架因缺乏操作专业知识而不足以应对。虽然手动编写的知识可以弥补这一差距，但往往负担沉重且效率低下。我们介绍了 Mobile-Agent-V，这是一种创新框架，利用视频作为引导工具，轻松高效地将操作知识注入移动自动化过程。通过直接从视频内容中提取知识，Mobile-Agent-V 消除了手动干预，显著减少了知识获取所需的精力和时间。为了严格评估此方法，我们提出了 Mobile-Knowledge，这是一个专门用于评估外部知识对移动代理性能影响的基准。我们的实验结果表明，与现有方法相比，Mobile-Agent-V 的性能提高了36%，突显了其在移动自动化中的轻松高效优势。 

---
# A Challenge to Build Neuro-Symbolic Video Agents 

**Title (ZH)**: 构建神经符号视频代理的技术挑战 

**Authors**: Sahil Shah, Harsh Goel, Sai Shankar Narasimhan, Minkyu Choi, S P Sharan, Oguzhan Akcin, Sandeep Chinchali  

**Link**: [PDF](https://arxiv.org/pdf/2505.13851)  

**Abstract**: Modern video understanding systems excel at tasks such as scene classification, object detection, and short video retrieval. However, as video analysis becomes increasingly central to real-world applications, there is a growing need for proactive video agents for the systems that not only interpret video streams but also reason about events and take informed actions. A key obstacle in this direction is temporal reasoning: while deep learning models have made remarkable progress in recognizing patterns within individual frames or short clips, they struggle to understand the sequencing and dependencies of events over time, which is critical for action-driven decision-making. Addressing this limitation demands moving beyond conventional deep learning approaches. We posit that tackling this challenge requires a neuro-symbolic perspective, where video queries are decomposed into atomic events, structured into coherent sequences, and validated against temporal constraints. Such an approach can enhance interpretability, enable structured reasoning, and provide stronger guarantees on system behavior, all key properties for advancing trustworthy video agents. To this end, we present a grand challenge to the research community: developing the next generation of intelligent video agents that integrate three core capabilities: (1) autonomous video search and analysis, (2) seamless real-world interaction, and (3) advanced content generation. By addressing these pillars, we can transition from passive perception to intelligent video agents that reason, predict, and act, pushing the boundaries of video understanding. 

**Abstract (ZH)**: 现代视频理解系统在场景分类、物体检测和短视频检索等方面表现出色。然而，随着视频分析在实际应用中的重要性不断增加，人们越来越需要能够主动解读视频流、推理解事件并采取明智行动的视频代理。这一方向上的一大障碍是时间推理：虽然深度学习模型在识别单个帧或短片段内的模式方面取得了显著进展，但在理解事件随时间发生的顺序和依赖关系方面仍存在困难，这对手动驱动的决策至关重要。解决这一限制需要超越传统的深度学习方法。我们认为，解决这一挑战需要神经符号方法，即将视频查询分解为原子事件，结构化为一致的序列，并满足时间约束。这种方法可以增强可解释性，支持结构化推理，并提供更强的系统行为保证，这些都是推进可信视频代理所必需的关键特征。为此，我们向研究界提出一个宏伟挑战：开发集成了三大核心能力的新一代智能视频代理：（1）自主视频搜索和分析，（2）无缝现实世界交互，（3）高级内容生成。通过解决这些支柱，我们可以从被动感知过渡到能够推理、预测和行动的智能视频代理，从而推动视频理解的边界。 

---
# TelePlanNet: An AI-Driven Framework for Efficient Telecom Network Planning 

**Title (ZH)**: TelePlanNet：一种高效的电信网络规划的人工智能驱动框架 

**Authors**: Zongyuan Deng, Yujie Cai, Qing Liu, Shiyao Mu, Bin Lyu, Zhen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13831)  

**Abstract**: The selection of base station sites is a critical challenge in 5G network planning, which requires efficient optimization of coverage, cost, user satisfaction, and practical constraints. Traditional manual methods, reliant on human expertise, suffer from inefficiencies and are limited to an unsatisfied planning-construction consistency. Existing AI tools, despite improving efficiency in certain aspects, still struggle to meet the dynamic network conditions and multi-objective needs of telecom operators' networks. To address these challenges, we propose TelePlanNet, an AI-driven framework tailored for the selection of base station sites, integrating a three-layer architecture for efficient planning and large-scale automation. By leveraging large language models (LLMs) for real-time user input processing and intent alignment with base station planning, combined with training the planning model using the improved group relative policy optimization (GRPO) reinforcement learning, the proposed TelePlanNet can effectively address multi-objective optimization, evaluates candidate sites, and delivers practical solutions. Experiments results show that the proposed TelePlanNet can improve the consistency to 78%, which is superior to the manual methods, providing telecom operators with an efficient and scalable tool that significantly advances cellular network planning. 

**Abstract (ZH)**: 基于AI的5G基站选址框架TelePlanNet：面向多目标优化的大规模自动化规划 

---
# Multimodal RAG-driven Anomaly Detection and Classification in Laser Powder Bed Fusion using Large Language Models 

**Title (ZH)**: 基于大型语言模型的多模态RAG驱动激光 Powder 床融合中的异常检测与分类 

**Authors**: Kiarash Naghavi Khanghah, Zhiling Chen, Lela Romeo, Qian Yang, Rajiv Malhotra, Farhad Imani, Hongyi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13828)  

**Abstract**: Additive manufacturing enables the fabrication of complex designs while minimizing waste, but faces challenges related to defects and process anomalies. This study presents a novel multimodal Retrieval-Augmented Generation-based framework that automates anomaly detection across various Additive Manufacturing processes leveraging retrieved information from literature, including images and descriptive text, rather than training datasets. This framework integrates text and image retrieval from scientific literature and multimodal generation models to perform zero-shot anomaly identification, classification, and explanation generation in a Laser Powder Bed Fusion setting. The proposed framework is evaluated on four L-PBF manufacturing datasets from Oak Ridge National Laboratory, featuring various printer makes, models, and materials. This evaluation demonstrates the framework's adaptability and generalizability across diverse images without requiring additional training. Comparative analysis using Qwen2-VL-2B and GPT-4o-mini as MLLM within the proposed framework highlights that GPT-4o-mini outperforms Qwen2-VL-2B and proportional random baseline in manufacturing anomalies classification. Additionally, the evaluation of the RAG system confirms that incorporating retrieval mechanisms improves average accuracy by 12% by reducing the risk of hallucination and providing additional information. The proposed framework can be continuously updated by integrating emerging research, allowing seamless adaptation to the evolving landscape of AM technologies. This scalable, automated, and zero-shot-capable framework streamlines AM anomaly analysis, enhancing efficiency and accuracy. 

**Abstract (ZH)**: 基于检索增强生成的多模态框架在激光粉床融合增材制造过程中实现零样本异常检测与分类 

---
# LLM-based Evaluation Policy Extraction for Ecological Modeling 

**Title (ZH)**: 基于LLM的评价政策提取在生态建模中的应用 

**Authors**: Qi Cheng, Licheng Liu, Qing Zhu, Runlong Yu, Zhenong Jin, Yiqun Xie, Xiaowei Jia  

**Link**: [PDF](https://arxiv.org/pdf/2505.13794)  

**Abstract**: Evaluating ecological time series is critical for benchmarking model performance in many important applications, including predicting greenhouse gas fluxes, capturing carbon-nitrogen dynamics, and monitoring hydrological cycles. Traditional numerical metrics (e.g., R-squared, root mean square error) have been widely used to quantify the similarity between modeled and observed ecosystem variables, but they often fail to capture domain-specific temporal patterns critical to ecological processes. As a result, these methods are often accompanied by expert visual inspection, which requires substantial human labor and limits the applicability to large-scale evaluation. To address these challenges, we propose a novel framework that integrates metric learning with large language model (LLM)-based natural language policy extraction to develop interpretable evaluation criteria. The proposed method processes pairwise annotations and implements a policy optimization mechanism to generate and combine different assessment metrics. The results obtained on multiple datasets for evaluating the predictions of crop gross primary production and carbon dioxide flux have confirmed the effectiveness of the proposed method in capturing target assessment preferences, including both synthetically generated and expert-annotated model comparisons. The proposed framework bridges the gap between numerical metrics and expert knowledge while providing interpretable evaluation policies that accommodate the diverse needs of different ecosystem modeling studies. 

**Abstract (ZH)**: 评估生态时间序列对于在预测温室气体通量、捕捉碳-氮动力学和监测水文循环等重要应用中 benchmark 模型性能至关重要。传统的数值指标（如决定系数 R-squared、均方根误差 RMSE）广泛用于量化模型值和观测值生态变量之间的相似性，但往往难以捕捉对生态过程至关重要的特定时间序列模式。为解决这些问题，我们提出了一种新颖的框架，将度量学习与基于大型语言模型（LLM）的自然语言策略提取相结合，以开发可解释的评估标准。该方法处理成对标注并实现策略优化机制，以生成和组合不同的评估指标。针对评估作物粗估初级生产和二氧化碳通量的多个数据集的结果证实了该方法在捕捉目标评估偏好方面的有效性，包括合成生成和专家标注的模型比较。该框架在量化指标和专家知识之间架起了桥梁，同时提供了可解释的评估策略，以满足不同生态系统建模研究的多样化需求。 

---
# CoIn: Counting the Invisible Reasoning Tokens in Commercial Opaque LLM APIs 

**Title (ZH)**: CoIn: 计算商业不透明LLM API 中的隐形推理令牌数量 

**Authors**: Guoheng Sun, Ziyao Wang, Bowei Tian, Meng Liu, Zheyu Shen, Shwai He, Yexiao He, Wanghao Ye, Yiting Wang, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.13778)  

**Abstract**: As post-training techniques evolve, large language models (LLMs) are increasingly augmented with structured multi-step reasoning abilities, often optimized through reinforcement learning. These reasoning-enhanced models outperform standard LLMs on complex tasks and now underpin many commercial LLM APIs. However, to protect proprietary behavior and reduce verbosity, providers typically conceal the reasoning traces while returning only the final answer. This opacity introduces a critical transparency gap: users are billed for invisible reasoning tokens, which often account for the majority of the cost, yet have no means to verify their authenticity. This opens the door to token count inflation, where providers may overreport token usage or inject synthetic, low-effort tokens to inflate charges. To address this issue, we propose CoIn, a verification framework that audits both the quantity and semantic validity of hidden tokens. CoIn constructs a verifiable hash tree from token embedding fingerprints to check token counts, and uses embedding-based relevance matching to detect fabricated reasoning content. Experiments demonstrate that CoIn, when deployed as a trusted third-party auditor, can effectively detect token count inflation with a success rate reaching up to 94.7%, showing the strong ability to restore billing transparency in opaque LLM services. The dataset and code are available at this https URL. 

**Abstract (ZH)**: 训练后技术的发展使大型语言模型（LLMs）获得越来越多的结构化多步推理能力，这些能力通常通过强化学习进行优化。增强推理能力的模型在复杂任务上表现出色，并支撑着许多商业LLM API。然而，为了保护专有行为并减少冗余，提供者通常会隐藏推理踪迹，仅返回最终答案。这种不透明性引入了一个关键的透明度缺口：用户为不可见的推理令牌付费，这些令牌往往占成本的大部分，但却无法验证其真实性。这为令牌计数膨胀打开了大门，提供者可能会夸大令牌使用量或注入合成的低效令牌以增加收费。为了解决这一问题，我们提出了一种名为CoIn的验证框架，用于审核隐藏令牌的数量和语义有效性。CoIn从令牌嵌入指纹构建验证哈希树以检查令牌计数，并使用嵌入式相关性匹配来检测伪造的推理内容。实验表明，当CoIn作为受信第三方审计员部署时，能够有效检测到令牌计数膨胀，成功率高达94.7%，显示出在不透明的LLM服务中恢复计费透明度的强大能力。相关数据集和代码可在此处获取。 

---
# Measuring the Faithfulness of Thinking Drafts in Large Reasoning Models 

**Title (ZH)**: 测量大型推理模型中思维草稿的忠实度 

**Authors**: Zidi Xiong, Chen Shan, Zhenting Qi, Himabindu Lakkaraju  

**Link**: [PDF](https://arxiv.org/pdf/2505.13774)  

**Abstract**: Large Reasoning Models (LRMs) have significantly enhanced their capabilities in complex problem-solving by introducing a thinking draft that enables multi-path Chain-of-Thought explorations before producing final answers. Ensuring the faithfulness of these intermediate reasoning processes is crucial for reliable monitoring, interpretation, and effective control. In this paper, we propose a systematic counterfactual intervention framework to rigorously evaluate thinking draft faithfulness. Our approach focuses on two complementary dimensions: (1) Intra-Draft Faithfulness, which assesses whether individual reasoning steps causally influence subsequent steps and the final draft conclusion through counterfactual step insertions; and (2) Draft-to-Answer Faithfulness, which evaluates whether final answers are logically consistent with and dependent on the thinking draft, by perturbing the draft's concluding logic. We conduct extensive experiments across six state-of-the-art LRMs. Our findings show that current LRMs demonstrate selective faithfulness to intermediate reasoning steps and frequently fail to faithfully align with the draft conclusions. These results underscore the need for more faithful and interpretable reasoning in advanced LRMs. 

**Abstract (ZH)**: 大型推理模型（LRMs）通过引入思考草案，使其在复杂问题解决方面的能力得到了显著增强，该草案能够在生成最终答案之前进行多路径的Chain-of-Thought探索。确保这些中间推理过程的可信度对于可靠监控、解释和有效控制至关重要。本文提出了一种系统性的反事实干预框架，以严格评估思考草案的可信度。我们的方法集中在两个互补维度上：（1）草案内部可信度，通过反事实步骤插入评估个体推理步骤是否因果影响后续步骤和最终草案结论；（2）草案至答案可信度，通过扰动草案的结论逻辑来评估最终答案是否与草案逻辑一致并依赖于其结论。我们在六个最先进的LRMs上进行了广泛的实验。研究结果表明，当前的LRMs在中间推理步骤上表现出选择性的可信度，并且经常未能忠实一致地与草案结论对齐。这些结果强调了在高级LRMs中需要更加忠实和可解释的推理。 

---
# Model Cards for AI Teammates: Comparing Human-AI Team Familiarization Methods for High-Stakes Environments 

**Title (ZH)**: AI队友的模型卡片：高风险环境中人类-AI团队熟悉方法的比较 

**Authors**: Ryan Bowers, Richard Agbeyibor, Jack Kolb, Karen Feigh  

**Link**: [PDF](https://arxiv.org/pdf/2505.13773)  

**Abstract**: We compare three methods of familiarizing a human with an artificial intelligence (AI) teammate ("agent") prior to operation in a collaborative, fast-paced intelligence, surveillance, and reconnaissance (ISR) environment. In a between-subjects user study (n=60), participants either read documentation about the agent, trained alongside the agent prior to the mission, or were given no familiarization. Results showed that the most valuable information about the agent included details of its decision-making algorithms and its relative strengths and weaknesses compared to the human. This information allowed the familiarization groups to form sophisticated team strategies more quickly than the control group. Documentation-based familiarization led to the fastest adoption of these strategies, but also biased participants towards risk-averse behavior that prevented high scores. Participants familiarized through direct interaction were able to infer much of the same information through observation, and were more willing to take risks and experiment with different control modes, but reported weaker understanding of the agent's internal processes. Significant differences were seen between individual participants' risk tolerance and methods of AI interaction, which should be considered when designing human-AI control interfaces. Based on our findings, we recommend a human-AI team familiarization method that combines AI documentation, structured in-situ training, and exploratory interaction. 

**Abstract (ZH)**: 我们比较了三种在协作、快节奏的 intelligence、surveillance 和 reconnaissance (ISR) 环境中让人类与人工 Intelligence (AI) 同伴（"代理"）熟悉的方法。在一项涉及 60 名参与者的组间用户研究中，参与者要么阅读关于代理的文档，要么在任务前与代理一起训练，要么没有任何熟悉过程。结果显示，关于代理最有价值的信息包括其决策算法细节及其与人类相比的优势和劣势。这些信息使熟悉组能够比对照组更快地形成复杂的团队策略。基于文档的熟悉方法导致参与者最快地采用这些策略，但也使他们倾向于风险规避行为，阻碍了高分的获得。通过直接互动熟悉代理的参与者可以通过观察推断出很多相同的信息，并且更愿意承担风险和尝试不同的控制模式，但报告了对代理内部过程的理解较弱。不同个体参与者的风险容忍度与与 AI 交互的方法之间存在显著差异，这在设计人类-AI 控制界面时应予以考虑。基于我们的发现，我们建议将 AI 文档、结构化的现场培训和探索性互动相结合的人类-AI 团队熟悉方法。 

---
# Ice Cream Doesn't Cause Drowning: Benchmarking LLMs Against Statistical Pitfalls in Causal Inference 

**Title (ZH)**: 冰淇淋不会导致溺水：将LLMs与因果推断中的统计陷阱进行基准测试 

**Authors**: Jin Du, Li Chen, Xun Xian, An Luo, Fangqiao Tian, Ganghua Wang, Charles Doss, Xiaotong Shen, Jie Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.13770)  

**Abstract**: Reliable causal inference is essential for making decisions in high-stakes areas like medicine, economics, and public policy. However, it remains unclear whether large language models (LLMs) can handle rigorous and trustworthy statistical causal inference. Current benchmarks usually involve simplified tasks. For example, these tasks might only ask LLMs to identify semantic causal relationships or draw conclusions directly from raw data. As a result, models may overlook important statistical pitfalls, such as Simpson's paradox or selection bias. This oversight limits the applicability of LLMs in the real world. To address these limitations, we propose CausalPitfalls, a comprehensive benchmark designed to rigorously evaluate the capability of LLMs in overcoming common causal inference pitfalls. Our benchmark features structured challenges across multiple difficulty levels, each paired with grading rubrics. This approach allows us to quantitatively measure both causal reasoning capabilities and the reliability of LLMs' responses. We evaluate models using two protocols: (1) direct prompting, which assesses intrinsic causal reasoning, and (2) code-assisted prompting, where models generate executable code for explicit statistical analysis. Additionally, we validate the effectiveness of this judge by comparing its scoring with assessments from human experts. Our results reveal significant limitations in current LLMs when performing statistical causal inference. The CausalPitfalls benchmark provides essential guidance and quantitative metrics to advance the development of trustworthy causal reasoning systems. 

**Abstract (ZH)**: 可靠的因果推断对于在医学、经济学和公共政策等高风险领域做出决策至关重要。然而，目前尚不清楚大语言模型（LLMs）能否处理严谨且可信赖的统计因果推断。当前的基准测试通常涉及简化任务，例如，这些任务可能只要求LLMs识别语义上的因果关系或直接从原始数据中得出结论。结果，模型可能会忽略重要的统计陷阱，如辛普森悖论或选择偏差。这种忽视限制了LLMs在现实生活中的应用。为了解决这些局限性，我们提出CausalPitfalls，一个全面的基准测试，旨在严格评估LLMs克服常见因果推断陷阱的能力。该基准测试包含不同难度级别的结构化挑战，每项挑战均配有评分标准。这种方法使我们能够定量衡量因果推理能力和LLMs响应的可靠性。我们使用两种协议评估模型：（1）直接提示，评估内在的因果推理能力；（2）代码辅助提示，模型生成可执行代码进行明确的统计分析。此外，我们通过将该评分系统的结果与人类专家评估结果进行比较，验证其有效性。结果显示，当前的LLMs在执行统计因果推断时存在显著局限性。CausalPitfalls基准测试提供了必不可少的指导和支持，以推动可信因果推理系统的开发。 

---
# Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations 

**Title (ZH)**: 语言模型能够监控和控制其内部激活过程。 

**Authors**: Li Ji-An, Hua-Dong Xiong, Robert C. Wilson, Marcelo G. Mattar, Marcus K. Benna  

**Link**: [PDF](https://arxiv.org/pdf/2505.13763)  

**Abstract**: Large language models (LLMs) can sometimes report the strategies they actually use to solve tasks, but they can also fail to do so. This suggests some degree of metacognition -- the capacity to monitor one's own cognitive processes for subsequent reporting and self-control. Metacognitive abilities enhance AI capabilities but raise safety concerns, as models might obscure their internal processes to evade neural-activation-based oversight mechanisms designed to detect harmful behaviors. Given society's increased reliance on these models, it is critical that we understand the limits of their metacognitive abilities, particularly their ability to monitor their internal activations. To address this, we introduce a neuroscience-inspired neurofeedback paradigm designed to quantify the ability of LLMs to explicitly report and control their activation patterns. By presenting models with sentence-label pairs where labels correspond to sentence-elicited internal activations along specific directions in the neural representation space, we demonstrate that LLMs can learn to report and control these activations. The performance varies with several factors: the number of example pairs provided, the semantic interpretability of the target neural direction, and the variance explained by that direction. These results reveal a "metacognitive space" with dimensionality much lower than the model's neural space, suggesting LLMs can monitor only a subset of their neural mechanisms. Our findings provide empirical evidence quantifying metacognitive capabilities in LLMs, with significant implications for AI safety. 

**Abstract (ZH)**: 大型语言模型的元认知能力：神经反馈研究揭示其内部激活监测能力 

---
# Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers 

**Title (ZH)**: 因果头部门控：Transformer中注意力头部作用的解释框架 

**Authors**: Andrew Nam, Henry Conklin, Yukang Yang, Thomas Griffiths, Jonathan Cohen, Sarah-Jane Leslie  

**Link**: [PDF](https://arxiv.org/pdf/2505.13737)  

**Abstract**: We present causal head gating (CHG), a scalable method for interpreting the functional roles of attention heads in transformer models. CHG learns soft gates over heads and assigns them a causal taxonomy - facilitating, interfering, or irrelevant - based on their impact on task performance. Unlike prior approaches in mechanistic interpretability, which are hypothesis-driven and require prompt templates or target labels, CHG applies directly to any dataset using standard next-token prediction. We evaluate CHG across multiple large language models (LLMs) in the Llama 3 model family and diverse tasks, including syntax, commonsense, and mathematical reasoning, and show that CHG scores yield causal - not merely correlational - insight, validated via ablation and causal mediation analyses. We also introduce contrastive CHG, a variant that isolates sub-circuits for specific task components. Our findings reveal that LLMs contain multiple sparse, sufficient sub-circuits, that individual head roles depend on interactions with others (low modularity), and that instruction following and in-context learning rely on separable mechanisms. 

**Abstract (ZH)**: 我们 presents 调因头部门控（CHG）：一种可扩展的方法，用于解释transformer模型中注意力头部的函数角色。CHG 学习头部的软门控，并根据其对任务性能的影响将它们分配为促进性、干扰性或无关性，从而形成一种因果分类学。与基于假设的机制解释方法不同，CHG 无需提示模板或目标标签即可应用于任何数据集，使用标准的下一个词预测。我们在Llama 3模型家族中的多个大型语言模型（LLM）和包括句法、常识和数学推理在内的多种任务上评估了CHG，证明CHG分数提供了因果而非相关性见解，通过消融分析和因果中介分析进行了验证。我们还引入了对比型CHG，这是一种用于隔离特定任务组件亚电路的变体。我们的研究发现LLM包含多个稀疏、足够的亚电路，个体头部角色依赖于与其他头部的互动（较低的模块性），并且指令执行和上下文学习依赖于分离的机制。 

---
# Warm Up Before You Train: Unlocking General Reasoning in Resource-Constrained Settings 

**Title (ZH)**: 热身再训练：在资源受限环境中解锁通用推理能力 

**Authors**: Safal Shrestha, Minwu Kim, Aadim Nepal, Anubhav Shrestha, Keith Ross  

**Link**: [PDF](https://arxiv.org/pdf/2505.13718)  

**Abstract**: Designing effective reasoning-capable LLMs typically requires training using Reinforcement Learning with Verifiable Rewards (RLVR) or distillation with carefully curated Long Chain of Thoughts (CoT), both of which depend heavily on extensive training data. This creates a major challenge when the amount of quality training data is scarce. We propose a sample-efficient, two-stage training strategy to develop reasoning LLMs under limited supervision. In the first stage, we "warm up" the model by distilling Long CoTs from a toy domain, namely, Knights \& Knaves (K\&K) logic puzzles to acquire general reasoning skills. In the second stage, we apply RLVR to the warmed-up model using a limited set of target-domain examples. Our experiments demonstrate that this two-phase approach offers several benefits: $(i)$ the warmup phase alone facilitates generalized reasoning, leading to performance improvements across a range of tasks, including MATH, HumanEval$^{+}$, and MMLU-Pro. $(ii)$ When both the base model and the warmed-up model are RLVR trained on the same small dataset ($\leq100$ examples), the warmed-up model consistently outperforms the base model; $(iii)$ Warming up before RLVR training allows a model to maintain cross-domain generalizability even after training on a specific domain; $(iv)$ Introducing warmup in the pipeline improves not only accuracy but also overall sample efficiency during RLVR training. The results in this paper highlight the promise of warmup for building robust reasoning LLMs in data-scarce environments. 

**Abstract (ZH)**: 在少量监督下设计有效的推理-capable大语言模型的一种样本高效双阶段训练策略 

---
# Building spatial world models from sparse transitional episodic memories 

**Title (ZH)**: 基于稀疏过渡性 episodic 记忆构建空间世界模型 

**Authors**: Zizhan He, Maxime Daigle, Pouya Bashivan  

**Link**: [PDF](https://arxiv.org/pdf/2505.13696)  

**Abstract**: Many animals possess a remarkable capacity to rapidly construct flexible mental models of their environments. These world models are crucial for ethologically relevant behaviors such as navigation, exploration, and planning. The ability to form episodic memories and make inferences based on these sparse experiences is believed to underpin the efficiency and adaptability of these models in the brain. Here, we ask: Can a neural network learn to construct a spatial model of its surroundings from sparse and disjoint episodic memories? We formulate the problem in a simulated world and propose a novel framework, the Episodic Spatial World Model (ESWM), as a potential answer. We show that ESWM is highly sample-efficient, requiring minimal observations to construct a robust representation of the environment. It is also inherently adaptive, allowing for rapid updates when the environment changes. In addition, we demonstrate that ESWM readily enables near-optimal strategies for exploring novel environments and navigating between arbitrary points, all without the need for additional training. 

**Abstract (ZH)**: 一种从稀疏离散的 episodic 记忆构建空间模型的神经网络框架：Episodic 空间世界模型（ESWM）的研究 

---
# A*-Decoding: Token-Efficient Inference Scaling 

**Title (ZH)**: A*-解码：.token-高效推断扩展 

**Authors**: Giannis Chatziveroglou  

**Link**: [PDF](https://arxiv.org/pdf/2505.13672)  

**Abstract**: Inference-time scaling has emerged as a powerful alternative to parameter scaling for improving language model performance on complex reasoning tasks. While existing methods have shown strong performance gains under fixed compute budgets, there has been little focus on optimally utilizing that budget during inference. In this work, we introduce A*-decoding, a search-based inference-time strategy that builds on the A* search algorithm to optimally utilize a fixed compute budget by prioritizing high-quality reasoning paths during generation. We frame language model decoding as a structured search in a state space of partial solutions, applying the A* transition model to identify promising continuations guided by an external process supervision signal. In our experiments, A*-decoding reaches the performance levels of strong inference scaling baselines like best-of-N and particle filtering while using up to 3x fewer tokens and 30% fewer PRM passes under equivalent compute budgets. On the MATH500 and AIME 2024 benchmarks, A*-decoding enables Llama-3.2-1B-Instruct to match the performance of the 70x larger Llama-3.1-70B-Instruct, and allows Qwen3-1.7B to reach o1-like reasoning accuracy. These results highlight the power of structured search in decoding, offering an alternative to brute-force sampling or scale-driven gains. Our work demonstrates how thoughtful inference-time strategies can enhance reasoning in SLMs, pointing toward future advances in more efficient and scalable language model deployment. 

**Abstract (ZH)**: 基于A*解码的推理时_scaled参数调整作为提升复杂推理任务语言模型性能的有力替代方法，在固定计算预算下，现有方法已显示出强劲的性能提升，但对如何最优化利用该预算的研究较少。在本工作中，我们引入了基于A*解码策略，该策略通过优先生成高质量的推理路径来优化固定计算预算的使用。我们将语言模型解码视为在部分解状态空间中的结构化搜索，并使用A*转移模型根据外部过程监督信号识别有前景的延续。实验表明，基于A*解码达到与.best-of-N 和粒子滤波等强大推理时_scale参数调整基线相当的性能水平，但使用的tokens数最多减少3倍，PRM遍历次数减少30%。在MATH500和AIME 2024基准测试中，基于A*解码使Llama-3.2-1B-Instruct达到与70倍更大的Llama-3.1-70B-Instruct相当的性能，并使Qwen3-1.7B达到类似o1的推理准确性。这些结果突显了结构化搜索在解码中的强大能力，为另一种 brute-force抽样或规模驱动的性能提升提供了替代方案。我们的研究展示了如何通过精心设计的推理时策略增强大规模语言模型的推理能力，指出了在未来更高效和可扩展的语言模型部署中的潜在进展。 

---
# MAFA: A multi-agent framework for annotation 

**Title (ZH)**: MAFA：多代理框架进行标注 

**Authors**: Mahmood Hegazy, Aaron Rodrigues, Azzam Naeem  

**Link**: [PDF](https://arxiv.org/pdf/2505.13668)  

**Abstract**: Modern applications require accurate and efficient retrieval of information in response to user queries. Mapping user utterances to the most relevant Frequently Asked Questions (FAQs) is a crucial component of these systems. Traditional approaches often rely on a single model or technique, which may not capture the nuances of diverse user inquiries. In this paper, we introduce a multi-agent framework for FAQ annotation that combines multiple specialized agents with different approaches and a judge agent that reranks candidates to produce optimal results. Our agents utilize a structured reasoning approach inspired by Attentive Reasoning Queries (ARQs), which guides them through systematic reasoning steps using targeted, task-specific JSON queries. Our framework features a specialized few-shot example strategy, where each agent receives different few-shots, enhancing ensemble diversity and coverage of the query space. We evaluate our framework on a real-world banking dataset as well as public benchmark datasets (LCQMC and FiQA), demonstrating significant improvements over single-agent approaches across multiple metrics, including a 14% increase in Top-1 accuracy, an 18% increase in Top-5 accuracy, and a 12% improvement in Mean Reciprocal Rank on our dataset, and similar gains on public benchmarks when compared with traditional single agent annotation techniques. Our framework is particularly effective at handling ambiguous queries, making it well-suited for deployment in production applications while showing strong generalization capabilities across different domains and languages. 

**Abstract (ZH)**: 现代应用程序需要对用户查询进行准确且高效的响应信息检索。将用户陈述映射到最相关的常见问题（FAQ）是这些系统的关键组成部分。传统方法通常依赖单一模型或技术，这可能无法捕捉到多样的用户询问的细微差别。在本文中，我们引入了一种基于多代理的FAQ标注框架，结合了多个专业代理和不同的方法，并通过一个评审代理重新排序候选项以生成最优结果。我们的代理利用了Attentive Reasoning Queries（ARQs）启发的结构化推理方法，利用针对性的任务特定JSON查询引导他们进行系统的推理步骤。我们的框架包含一个专门的小样本示例策略，其中每个代理接收不同的小样本，从而增强组合的多样性和查询空间的覆盖率。我们在实际银行数据集以及公开基准数据集（LCQMC和FiQA）上评估了我们的框架，结果显示在多个指标上相比单代理方法有显著改进，包括Top-1准确性提升14%，Top-5准确性提升18%，以及我们的数据集上Mean Reciprocal Rank提高12%，并在公开基准上相比传统单代理标注技术也有类似的改进。我们的框架特别适用于处理含糊查询，使其非常适合部署在生产应用程序中，同时在不同的领域和语言中展示出强大的泛化能力。 

---
# Language and Thought: The View from LLMs 

**Title (ZH)**: 语言与思维：从大规模语言模型视角看 

**Authors**: Daniel Rothschild  

**Link**: [PDF](https://arxiv.org/pdf/2505.13561)  

**Abstract**: Daniel Dennett speculated in *Kinds of Minds* 1996: "Perhaps the kind of mind you get when you add language to it is so different from the kind of mind you can have without language that calling them both minds is a mistake." Recent work in AI can be seen as testing Dennett's thesis by exploring the performance of AI systems with and without linguistic training. I argue that the success of Large Language Models at inferential reasoning, limited though it may be, supports Dennett's radical view about the effect of language on thought. I suggest it is the abstractness and efficiency of linguistic encoding that lies behind the capacity of LLMs to perform inferences across a wide range of domains. In a slogan, language makes inference computationally tractable. I assess what these results in AI indicate about the role of language in the workings of our own biological minds. 

**Abstract (ZH)**: 丹尼尔·狄恩特在1996年的《心智的种类》中推测：“也许当你向一个没有语言的心智中添加语言时，所获得的那种心智种类如此不同，以至于称它们为心智可能就是一种误判。”最近的人工智能研究可以被视为狄恩特假说的一种验证，通过探索具有和不具备语言训练的AI系统的性能。我认为，尽管大型语言模型在演绎推理方面的成功（尽管是有限的）支持了关于语言对思维影响的狄恩特激进观点。我认为是语言编码的抽象性和效率使得大型语言模型能够在广泛的不同领域中进行推理。简言之，语言使演绎推理在计算上变得可行。我评估这些人工智能成果对我们自身生物性心智的运作中语言作用的指示意义。 

---
# Counter-Inferential Behavior in Natural and Artificial Cognitive Systems 

**Title (ZH)**: 自然与人工认知系统中的反向推理行为 

**Authors**: Serge Dolgikh  

**Link**: [PDF](https://arxiv.org/pdf/2505.13551)  

**Abstract**: This study explores the emergence of counter-inferential behavior in natural and artificial cognitive systems, that is, patterns in which agents misattribute empirical success or suppress adaptation, leading to epistemic rigidity or maladaptive stability. We analyze archetypal scenarios in which such behavior arises: reinforcement of stability through reward imbalance, meta-cognitive attribution of success to internal superiority, and protective reframing under perceived model fragility. Rather than arising from noise or flawed design, these behaviors emerge through structured interactions between internal information models, empirical feedback, and higher-order evaluation mechanisms. Drawing on evidence from artificial systems, biological cognition, human psychology, and social dynamics, we identify counter-inferential behavior as a general cognitive vulnerability that can manifest even in otherwise well-adapted systems. The findings highlight the importance of preserving minimal adaptive activation under stable conditions and suggest design principles for cognitive architectures that can resist rigidity under informational stress. 

**Abstract (ZH)**: 这一研究探讨了自然和人工认知系统中反向推理行为的涌现，即代理将实证成功归因于内部优越性或抑制适应性，导致知识上的僵化或适应不良的稳定性。我们分析了此类行为产生的典型场景：通过奖励失衡强化稳定性、元认知将成功归因于内部优越性以及在感知到模型脆弱性时的保护性重新解释。这些行为并非源自噪声或设计缺陷，而是通过内部信息模型、实证反馈和高级评估机制之间的结构化互动而涌现。借鉴来自人工系统、生物认知、人类心理学和社会动力学的证据，我们确定反向推理行为是普遍存在的认知脆弱性，即使在适应良好的系统中也可能表现出来。研究结果强调了在稳定条件下保持最小适应激活的重要性，并提出了可以抵抗信息压力下僵化的认知架构设计原则。 

---
# Prompt Stability Matters: Evaluating and Optimizing Auto-Generated Prompt in General-Purpose Systems 

**Title (ZH)**: 提示稳定性至关重要：评估与优化通用系统中自动生成的提示 

**Authors**: Ke Chen, Yufei Zhou, Xitong Zhang, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13546)  

**Abstract**: Automatic prompt generation plays a crucial role in enabling general-purpose multi-agent systems to perform diverse tasks autonomously. Existing methods typically evaluate prompts based on their immediate task performance, overlooking the intrinsic qualities that determine their reliability. This outcome-centric view not only limits interpretability but also fails to account for the inherent stochasticity of large language models (LLMs). In this work, we bring attention to prompt stability-the consistency of model responses across repeated executions-as a key factor for building robust and effective prompt generation systems. To quantify this, we propose semantic stability as a criterion for assessing the response consistency of prompts, and fine-tune a LLaMA-based evaluator to measure it automatically across tasks. These components have enabled us to develop the first stability-aware general-purpose prompt generation system that leverages stability feedback to iteratively enhance both prompt quality and system-level performance. Furthermore, we establish a logical chain between prompt stability and task success by analyzing the structural dependencies within our system, proving stability as a necessary condition for effective system-level execution. Empirical results across general and domain-specific tasks demonstrate that our stability-aware framework improves both accuracy and output consistency. By shifting the focus from one-off results to persistent reliability, our work offers a new perspective on prompt design and contributes practical tools for building more trustworthy general-purpose systems. 

**Abstract (ZH)**: 自动提示生成在使通用多代理系统能够自主执行多样化任务中发挥着关键作用。现有方法通常根据提示的即时任务性能对其进行评估，忽略了决定其可靠性的内在特性。这种以结果为中心的观点不仅限制了可解释性，同时也未能考虑大型语言模型（LLMs）的固有随机性。在本工作中，我们将注意力转向提示稳定性——模型响应在重复执行中的一致性——作为构建稳健且有效的提示生成系统的关键因素。为量化这一特性，我们提出语义稳定性作为评估提示响应一致性的标准，并针对任务自动测量其值的LLaMA基评估器进行微调。这些组件使我们能够开发出首个具备稳定性的通用提示生成系统，该系统利用稳定性反馈逐步提升提示质量和系统级性能。此外，通过分析系统中的结构性依赖关系，我们建立了提示稳定性和任务成功之间的逻辑联系，证明稳定性是实现有效系统级执行的必要条件。在通用和特定领域的任务中，实验证据显示，我们的稳定性感知框架提高了准确性和输出一致性。通过将焦点从单一结果转移到持久可靠性，我们的工作为提示设计提供了新的视角，并为构建更具可信度的通用系统提供了实用工具。 

---
# FinMaster: A Holistic Benchmark for Mastering Full-Pipeline Financial Workflows with LLMs 

**Title (ZH)**: FinMaster: 一个全面的基准，用于利用大语言模型掌握全流程金融工作流程 

**Authors**: Junzhe Jiang, Chang Yang, Aixin Cui, Sihan Jin, Ruiyu Wang, Bo Li, Xiao Huang, Dongning Sun, Xinrun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13533)  

**Abstract**: Financial tasks are pivotal to global economic stability; however, their execution faces challenges including labor intensive processes, low error tolerance, data fragmentation, and tool limitations. Although large language models (LLMs) have succeeded in various natural language processing tasks and have shown potential in automating workflows through reasoning and contextual understanding, current benchmarks for evaluating LLMs in finance lack sufficient domain-specific data, have simplistic task design, and incomplete evaluation frameworks. To address these gaps, this article presents FinMaster, a comprehensive financial benchmark designed to systematically assess the capabilities of LLM in financial literacy, accounting, auditing, and consulting. Specifically, FinMaster comprises three main modules: i) FinSim, which builds simulators that generate synthetic, privacy-compliant financial data for companies to replicate market dynamics; ii) FinSuite, which provides tasks in core financial domains, spanning 183 tasks of various types and difficulty levels; and iii) FinEval, which develops a unified interface for evaluation. Extensive experiments over state-of-the-art LLMs reveal critical capability gaps in financial reasoning, with accuracy dropping from over 90% on basic tasks to merely 40% on complex scenarios requiring multi-step reasoning. This degradation exhibits the propagation of computational errors, where single-metric calculations initially demonstrating 58% accuracy decreased to 37% in multimetric scenarios. To the best of our knowledge, FinMaster is the first benchmark that covers full-pipeline financial workflows with challenging tasks. We hope that FinMaster can bridge the gap between research and industry practitioners, driving the adoption of LLMs in real-world financial practices to enhance efficiency and accuracy. 

**Abstract (ZH)**: 金融任务对于全球经济稳定至关重要；然而，这些任务的执行面临着劳动密集型流程、低容错率、数据碎片化和工具限制等挑战。尽管大型语言模型（LLMs）已经在各种自然语言处理任务中取得成功，并展示了通过推理和情境理解自动化工作流的潜力，但现有对LLMs在金融领域的评估基准缺乏特定领域的数据，任务设计过于简单，且评估框架不完整。为弥补这些不足，本文提出了FinMaster，一个全面的金融基准，旨在系统评估LLMs在金融知识、会计、审计和咨询方面的能力。具体而言，FinMaster包含三个主要模块：i) FinSim，构建模拟器以生成合成且符合隐私要求的金融数据，用于模拟市场动态；ii) FinSuite，提供涵盖核心金融领域的任务，共计183项不同类型和难度的任务；iii) FinEval，开发统一的评估界面。通过对最新一代LLMs的广泛实验表明，金融推理能力存在关键差距，基本任务的准确率从超过90%降至复杂场景下的40%，其中多指标场景中以往显示58%准确率的单指标计算准确性降至37%。据我们所知，FinMaster是首个涵盖全流程金融工作流的具有挑战性任务的基准。我们希望FinMaster能够弥合研究与行业实践之间的差距，推动LLMs在实际金融实践中的应用，以提高效率和准确性。 

---
# BARREL: Boundary-Aware Reasoning for Factual and Reliable LRMs 

**Title (ZH)**: BARREL: 边界感知推理以实现事实可靠的知识表示模型 

**Authors**: Junxiao Yang, Jinzhe Tu, Haoran Liu, Xiaoce Wang, Chujie Zheng, Zhexin Zhang, Shiyao Cui, Caishun Chen, Tiantian He, Hongning Wang, Yew-Soon Ong, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13529)  

**Abstract**: Recent advances in Large Reasoning Models (LRMs) have shown impressive capabilities in mathematical and logical reasoning. However, current LRMs rarely admit ignorance or respond with "I don't know". Instead, they often produce incorrect answers while showing undue confidence, raising concerns about their factual reliability. In this work, we identify two pathological reasoning patterns characterized by overthinking that contribute to the overconfident and incorrect answers: last-minute guessing and second-thought spiraling. To address these issues, we propose BARREL-a novel framework that promotes concise and boundary-aware factual reasoning. Our experiments show that BARREL-training increases the reliability of DeepSeek-R1-Distill-Llama-8B from 39.33% to 61.48%, while still achieving accuracy comparable to models finetuned on reasoning data generated by R1. These results demonstrate that our pilot study is inspiring to build more reliable and factual System 2 LRMs. 

**Abstract (ZH)**: Recent Advances in Large Reasoning Models: Promoting Concise and Boundary-Aware Factual Reasoning 

---
# A Heuristic Algorithm Based on Beam Search and Iterated Local Search for the Maritime Inventory Routing Problem 

**Title (ZH)**: 基于Beam Search和Iterated Local Search的启发式算法：用于 maritime inventory routing 问题 

**Authors**: Nathalie Sanghikian, Rafael Meirelles, Rafael Martinelli, Anand Subramanian  

**Link**: [PDF](https://arxiv.org/pdf/2505.13522)  

**Abstract**: Maritime Inventory Routing Problem (MIRP) plays a crucial role in the integration of global maritime commerce levels. However, there are still no well-established methodologies capable of efficiently solving large MIRP instances or their variants due to the high complexity of the problem. The adoption of exact methods, typically based on Mixed Integer Programming (MIP), for daily operations is nearly impractical due to the CPU time required, as planning must be executed multiple times while ensuring high-quality results within acceptable time limits. Non-MIP-based heuristics are less frequently applied due to the highly constrained nature of the problem, which makes even the construction of an effective initial solution challenging. Papageorgiou et al. (2014) introduced a single-product MIRP as the foundation for MIRPLib, aiming to provide a collection of publicly available benchmark instances. However, only a few studies that propose new methodologies have been published since then. To encourage the use of MIRPLib and facilitate result comparisons, this study presents a heuristic approach that does not rely on mathematical optimization techniques to solve a deterministic, finite-horizon, single-product MIRP. The proposed heuristic combines a variation of a Beam Search algorithm with an Iterated Local Search procedure. Among the 72 instances tested, the developed methodology can improve the best-known solution for ten instances within an acceptable CPU time. 

**Abstract (ZH)**: 海上存货路由问题（MIRP）在全球海运贸易的集成中扮演着重要角色。然而，由于问题的高复杂性，尚未建立起能够有效解决大规模MIRP实例或其变体的方法学。采用基于混合整数规划（MIP）的精确方法在日常运营中几乎不可行，因为计划必须多次执行，同时在可接受的时间内保证高质量的结果。非MIP的启发式方法由于问题是高度约束的，构建有效初始解也颇具挑战。Papageorgiou等人（2014）引入了一个单一产品MIRP作为MIRPLib的基础，旨在提供一集合众可用的基准实例。然而，此后仅有少数研究提出了新的方法学。为鼓励使用MIRPLib并促进结果比较，本研究提出了一种不依赖于数学优化技术的启发式方法，用于解决确定性、有限期区、单一产品MIRP。所提出的启发式方法结合了Beam Search算法的一种变体与迭代局部搜索过程，在72个测试实例中，在可接受的计算时间范围内，可以改进十个实例的最优已知解。 

---
# Can AI Freelancers Compete? Benchmarking Earnings, Reliability, and Task Success at Scale 

**Title (ZH)**: AI自由职业者能否竞争？大规模benchmarking报酬、可靠性和任务成功率 

**Authors**: David Noever, Forrest McKee  

**Link**: [PDF](https://arxiv.org/pdf/2505.13511)  

**Abstract**: This study explores Large Language Models (LLMs) as autonomous agents for real-world tasks, including freelance software development. This work presents a new benchmark that evaluates LLMs on freelance programming and data analysis tasks derived from economic data. We construct the benchmark using synthetic tasks created from a Kaggle Freelancer dataset of job postings, with all job prices standardized to USD (median fixed-project price around $250, and an average of $306). Each task is accompanied by structured input-output test cases and an estimated price tag, enabling automated correctness checking and a monetary performance valuation. This approach is inspired by OpenAI's recent SWE-Lancer benchmark (1,400 real Upwork tasks worth $1M total). Still, our framework simplifies evaluation using programmatically testable tasks and predicted price values, making it highly scalable and repeatable. On this benchmark, we evaluate four modern LLMs - Claude 3.5 Haiku, GPT-4o-mini, Qwen 2.5, and Mistral. We report each model's accuracy (task success rate and test-case pass rate) and the total "freelance earnings" it achieves (sum of prices of solved tasks). Our results show that Claude 3.5 Haiku performs best, earning approximately $1.52 million USD, followed closely by GPT-4o-mini at $1.49 million, then Qwen 2.5 ($1.33M) and Mistral ($0.70M). We analyze the distribution of errors per task and observe that the strongest models solve the most tasks and rarely fail completely on any project. We discuss the implications of these results for the feasibility of AI as a freelance developer, the advantages and limitations of our automated benchmark approach, and the gap between performance on structured tasks versus the true complexity of real-world freelance jobs. 

**Abstract (ZH)**: 本研究探索大型语言模型（LLMs）作为自主代理在实际工作任务中的应用，包括自由职业软件开发。本文提出了一项新的基准测试，评估LLMs在源自经济学数据的自由职业编程和数据分析任务中的表现。我们使用来自Kaggle Freelancer数据集的工作招聘广告构建基准测试，所有工作价格都标准化为美元（中位数固定项目价格约为250美元，平均值为306美元）。每个任务都附有结构化的输入输出测试案例和估计的价格标签，这使得自动化正确性检查和货币性能评价成为可能。该方法受到了OpenAI最近的SWE-Lancer基准测试（总计10万美元的1400个实际Upwork任务）的启发。然而，我们的框架通过使用编程可测试的任务和预测的价格值简化了评估过程，使其具有高度的可扩展性和可重复性。在该基准测试中，我们评估了四款现代LLMs——Claude 3.5 Haiku、GPT-4o-mini、Qwen 2.5和Mistral。我们报告了每款模型的准确性（任务成功率和测试案例通过率）以及其实现的总“自由职业收入”（完成任务的价格总和）。结果显示，Claude 3.5 Haiku表现最佳，赚取约152万美元，紧随其后的是GPT-4o-mini（149万美元），然后是Qwen 2.5（133万美元）和Mistral（70万美元）。我们分析了每项任务的错误分布，并观察到最强的模型解决了最多任务，并且几乎未在任何项目中完全失败。我们讨论了这些结果对AI作为自由职业开发者的可行性的影响，以及我们自动化基准测试方法的优点和局限性，并探讨了在结构化任务表现与真实世界自由职业工作的复杂性之间的差距。 

---
# ADALog: Adaptive Unsupervised Anomaly detection in Logs with Self-attention Masked Language Model 

**Title (ZH)**: ADALog：基于自注意力掩蔽语言模型的日志自适应无监督异常检测 

**Authors**: Przemek Pospieszny, Wojciech Mormul, Karolina Szyndler, Sanjeev Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.13496)  

**Abstract**: Modern software systems generate extensive heterogeneous log data with dynamic formats, fragmented event sequences, and varying temporal patterns, making anomaly detection both crucial and challenging. To address these complexities, we propose ADALog, an adaptive, unsupervised anomaly detection framework designed for practical applicability across diverse real-world environments. Unlike traditional methods reliant on log parsing, strict sequence dependencies, or labeled data, ADALog operates on individual unstructured logs, extracts intra-log contextual relationships, and performs adaptive thresholding on normal data. The proposed approach utilizes a transformer-based, pretrained bidirectional encoder with a masked language modeling task, fine-tuned on normal logs to capture domain-specific syntactic and semantic patterns essential for accurate anomaly detection. Anomalies are identified via token-level reconstruction probabilities, aggregated into log-level scores, with adaptive percentile-based thresholding calibrated only on normal data. This allows the model to dynamically adapt to evolving system behaviors while avoiding rigid, heuristic-based thresholds common in traditional systems. We evaluate ADALog on benchmark datasets BGL, Thunderbird, and Spirit, showing strong generalization and competitive performance compared to state-of-the-art supervised and unsupervised methods. Additional ablation studies examine the effects of masking, fine-tuning, and token positioning on model behavior and interpretability. 

**Abstract (ZH)**: 现代软件系统生成大量异构的日志数据，具有动态格式、破碎的事件序列和变化的时间模式，使得异常检测变得既重要又具有挑战性。为应对这些复杂性，我们提出了ADALog，这是一种针对多样现实环境具备实际适用性的自适应无监督异常检测框架。与传统依赖日志解析、严格序列依赖或标记数据的方法不同，ADALog 基于个体无结构日志运行，提取内部日志上下文关系，并对正常数据进行自适应阈值处理。所提出的办法利用基于Transformer的双向编码器进行预训练，并通过掩码语言建模任务进一步微调，以捕捉特定领域内的句法和语义模式，这对于准确的异常检测至关重要。通过词元级重建概率识别异常，最终将这些得分聚合为日志级别分数，并通过仅在正常数据上校准的自适应分位数阈值进行调整。这种方法使模型能够动态适应系统行为的变化，避免了传统系统中常见的僵化启发式阈值。我们在基准数据集BGL、Thunderbird 和Spirit上评估了ADALog，展示了其强大的泛化能力和与先进监督和无监督方法相当的竞争力。通过消融研究进一步探讨了掩码、微调和词元定位对模型行为和可解释性的影响。 

---
# Contrastive Cross-Course Knowledge Tracing via Concept Graph Guided Knowledge Transfer 

**Title (ZH)**: 概念图引导的知识迁移的对比跨课程知识追踪 

**Authors**: Wenkang Han, Wang Lin, Liya Hu, Zhenlong Dai, Yiyun Zhou, Mengze Li, Zemin Liu, Chang Yao, Jingyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.13489)  

**Abstract**: Knowledge tracing (KT) aims to predict learners' future performance based on historical learning interactions. However, existing KT models predominantly focus on data from a single course, limiting their ability to capture a comprehensive understanding of learners' knowledge states. In this paper, we propose TransKT, a contrastive cross-course knowledge tracing method that leverages concept graph guided knowledge transfer to model the relationships between learning behaviors across different courses, thereby enhancing knowledge state estimation. Specifically, TransKT constructs a cross-course concept graph by leveraging zero-shot Large Language Model (LLM) prompts to establish implicit links between related concepts across different courses. This graph serves as the foundation for knowledge transfer, enabling the model to integrate and enhance the semantic features of learners' interactions across courses. Furthermore, TransKT includes an LLM-to-LM pipeline for incorporating summarized semantic features, which significantly improves the performance of Graph Convolutional Networks (GCNs) used for knowledge transfer. Additionally, TransKT employs a contrastive objective that aligns single-course and cross-course knowledge states, thereby refining the model's ability to provide a more robust and accurate representation of learners' overall knowledge states. 

**Abstract (ZH)**: 跨课程知识追踪方法TransKT：基于概念图引导的知识转移 

---
# Evaluating Large Language Models for Real-World Engineering Tasks 

**Title (ZH)**: 评估大型语言模型在实际工程任务中的性能 

**Authors**: Rene Heesch, Sebastian Eilermann, Alexander Windmann, Alexander Diedrich, Philipp Rosenthal, Oliver Niggemann  

**Link**: [PDF](https://arxiv.org/pdf/2505.13484)  

**Abstract**: Large Language Models (LLMs) are transformative not only for daily activities but also for engineering tasks. However, current evaluations of LLMs in engineering exhibit two critical shortcomings: (i) the reliance on simplified use cases, often adapted from examination materials where correctness is easily verifiable, and (ii) the use of ad hoc scenarios that insufficiently capture critical engineering competencies. Consequently, the assessment of LLMs on complex, real-world engineering problems remains largely unexplored. This paper addresses this gap by introducing a curated database comprising over 100 questions derived from authentic, production-oriented engineering scenarios, systematically designed to cover core competencies such as product design, prognosis, and diagnosis. Using this dataset, we evaluate four state-of-the-art LLMs, including both cloud-based and locally hosted instances, to systematically investigate their performance on complex engineering tasks. Our results show that LLMs demonstrate strengths in basic temporal and structural reasoning but struggle significantly with abstract reasoning, formal modeling, and context-sensitive engineering logic. 

**Abstract (ZH)**: 大型语言模型（LLMs）不仅在日常活动，而且在工程任务中都具有变革性。然而，当前对LLMs在工程领域的评估存在两个关键不足：（i）依赖于简化的用例，这些用例往往源自易于验证正确性的考试材料；（ii）使用缺乏系统性和全面性的场景，无法充分捕捉关键的工程能力。因此，对复杂的真实世界工程问题的评估仍存在很大空白。本文通过引入一个包含超过100个问题的精心整理数据库来填补这一空白，这些问题源自真实的、以生产为导向的工程情景，系统地设计以涵盖核心能力，如产品设计、预测和诊断。利用该数据集，我们评估了四种最先进的LLMs（包括云托管和本地托管实例），以系统性地研究其在复杂工程任务中的表现。我们的结果显示，LLMs在基本的时间和结构推理方面表现出优势，但在抽象推理、形式建模和情境敏感的工程逻辑方面面临重大挑战。 

---
# AgentSGEN: Multi-Agent LLM in the Loop for Semantic Collaboration and GENeration of Synthetic Data 

**Title (ZH)**: AgentSGEN：多智能体LLM参与的语义协作与合成数据生成 

**Authors**: Vu Dinh Xuan, Hao Vo, David Murphy, Hoang D. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.13466)  

**Abstract**: The scarcity of data depicting dangerous situations presents a major obstacle to training AI systems for safety-critical applications, such as construction safety, where ethical and logistical barriers hinder real-world data collection. This creates an urgent need for an end-to-end framework to generate synthetic data that can bridge this gap. While existing methods can produce synthetic scenes, they often lack the semantic depth required for scene simulations, limiting their effectiveness. To address this, we propose a novel multi-agent framework that employs an iterative, in-the-loop collaboration between two agents: an Evaluator Agent, acting as an LLM-based judge to enforce semantic consistency and safety-specific constraints, and an Editor Agent, which generates and refines scenes based on this guidance. Powered by LLM's capabilities to reasoning and common-sense knowledge, this collaborative design produces synthetic images tailored to safety-critical scenarios. Our experiments suggest this design can generate useful scenes based on realistic specifications that address the shortcomings of prior approaches, balancing safety requirements with visual semantics. This iterative process holds promise for delivering robust, aesthetically sound simulations, offering a potential solution to the data scarcity challenge in multimedia safety applications. 

**Abstract (ZH)**: 稀缺的数据描述危险情境是训练应用于建筑安全等关键安全领域的AI系统的一大障碍，而伦理和后勤壁垒阻碍了真实世界数据的收集。这迫切需要一个端到端的框架来生成合成数据以弥补这一差距。尽管现有方法可以生成合成场景，但它们往往缺乏用于场景模拟所需的语义深度，限制了其效果。为了解决这一问题，我们提出了一种新的多智能体框架，该框架采用两个智能体在循环中的迭代协作：评估智能体作为基于LLM的裁判，负责维护语义一致性及特定安全约束；编辑智能体根据这一指导生成和优化场景。依托LLM在推理和常识方面的能力，这种协作设计可以生成针对关键安全场景定制的合成图像。我们的实验表明，该设计能够生成基于现实规范且能够弥补前人方法不足的有用场景，平衡了安全需求与视觉语义。这一迭代过程为多媒体安全应用中的数据稀缺挑战提供了可靠且美观的模拟解决方案。 

---
# Mind the Gap: Bridging Thought Leap for Improved Chain-of-Thought Tuning 

**Title (ZH)**: 填坑：提升链式思维调优中的思想飞跃 

**Authors**: Haolei Xu, Yuchen Yan, Yongliang Shen, Wenqi Zhang, Guiyang Hou, Shengpei Jiang, Kaitao Song, Weiming Lu, Jun Xiao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14684)  

**Abstract**: Large language models (LLMs) have achieved remarkable progress on mathemati-cal tasks through Chain-of-Thought (CoT) reasoning. However, existing mathematical CoT datasets often suffer from Thought Leaps due to experts omitting intermediate steps, which negatively impacts model learning and generalization. We propose the CoT Thought Leap Bridge Task, which aims to automatically detect leaps and generate missing intermediate reasoning steps to restore the completeness and coherence of CoT. To facilitate this, we constructed a specialized training dataset called ScaleQM+, based on the structured ScaleQuestMath dataset, and trained CoT-Bridge to bridge thought leaps. Through comprehensive experiments on mathematical reasoning benchmarks, we demonstrate that models fine-tuned on bridged datasets consistently outperform those trained on original datasets, with improvements of up to +5.87% on NuminaMath. Our approach effectively enhances distilled data (+3.02%) and provides better starting points for reinforcement learning (+3.1%), functioning as a plug-and-play module compatible with existing optimization techniques. Furthermore, CoT-Bridge demonstrate improved generalization to out-of-domain logical reasoning tasks, confirming that enhancing reasoning completeness yields broadly applicable benefits. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通过链式思考（CoT）推理在数学任务上取得了显著进展。然而，现有的数学CoT数据集往往由于专家省略了中间步骤而受到跳跃（Thought Leaps）的影响，这负面影响了模型的学习和泛化能力。我们提出了CoT跳跃桥接任务，旨在自动检测跳跃并生成缺失的中间推理步骤，以恢复CoT的完整性和连贯性。为实现这一目标，我们基于结构化的ScaleQuestMath数据集构建了一个专门的训练数据集ScaleQM+，并训练了CoT-Bridge以桥接这些跳跃。通过在数学推理基准上的全面实验，我们表明，使用桥接数据集微调的模型始终优于使用原始数据集微调的模型，在NuminaMath上最高可提高5.87%。我们的方法有效提升了压缩数据（+3.02%）并为强化学习提供了更好的起始点（+3.1%），并可与现有优化技术无缝集成。此外，CoT-Bridge在领域外逻辑推理任务上的泛化能力提升，证实了增强推理完整性的广泛应用优势。 

---
# NExT-Search: Rebuilding User Feedback Ecosystem for Generative AI Search 

**Title (ZH)**: NExT-Search: 重建生成式AI搜索的用户反馈生态系统 

**Authors**: Sunhao Dai, Wenjie Wang, Liang Pang, Jun Xu, See-Kiong Ng, Ji-Rong Wen, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2505.14680)  

**Abstract**: Generative AI search is reshaping information retrieval by offering end-to-end answers to complex queries, reducing users' reliance on manually browsing and summarizing multiple web pages. However, while this paradigm enhances convenience, it disrupts the feedback-driven improvement loop that has historically powered the evolution of traditional Web search. Web search can continuously improve their ranking models by collecting large-scale, fine-grained user feedback (e.g., clicks, dwell time) at the document level. In contrast, generative AI search operates through a much longer search pipeline, spanning query decomposition, document retrieval, and answer generation, yet typically receives only coarse-grained feedback on the final answer. This introduces a feedback loop disconnect, where user feedback for the final output cannot be effectively mapped back to specific system components, making it difficult to improve each intermediate stage and sustain the feedback loop. In this paper, we envision NExT-Search, a next-generation paradigm designed to reintroduce fine-grained, process-level feedback into generative AI search. NExT-Search integrates two complementary modes: User Debug Mode, which allows engaged users to intervene at key stages; and Shadow User Mode, where a personalized user agent simulates user preferences and provides AI-assisted feedback for less interactive users. Furthermore, we envision how these feedback signals can be leveraged through online adaptation, which refines current search outputs in real-time, and offline update, which aggregates interaction logs to periodically fine-tune query decomposition, retrieval, and generation models. By restoring human control over key stages of the generative AI search pipeline, we believe NExT-Search offers a promising direction for building feedback-rich AI search systems that can evolve continuously alongside human feedback. 

**Abstract (ZH)**: 生成型AI搜索正在通过提供端到端的答案来重塑信息检索，减少用户对手动浏览和总结多个网页的依赖。然而，虽然这种范式增强了便利性，但它打断了历史上推动传统Web搜索演化的基于反馈改进的循环。Web搜索可以通过收集大规模、细粒度的用户反馈（如点击、驻留时间）不断改进其排名模型。相比之下，生成型AI搜索涉及一个更长的搜索流程，包括查询分解、文档检索和答案生成，但通常只能在最终答案上获得粗粒度的反馈。这引入了一种反馈循环断开的情况，即用户对最终输出的反馈无法有效地映射到具体的系统组件，使得难以改进每个中间阶段并维持反馈循环。在本文中，我们设想了NExT-Search，这是一种新一代范式，旨在将细粒度的、过程级的反馈重新引入生成型AI搜索。NExT-Search整合了两种互补模式：用户调试模式，允许积极参与的用户在关键阶段进行干预；以及影子用户模式，个性化用户代理模拟用户偏好并为不太交互的用户提供AI辅助反馈。此外，我们设想这些反馈信号可通过在线适应来利用，即在实时调整当前搜索输出，以及通过离线更新来汇总交互日志以定期微调查询分解、检索和生成模型。通过恢复人在生成型AI搜索流程关键阶段的控制权，我们相信NExT-Search为构建伴随人类反馈不断演进的反馈丰富型AI搜索系统提供了有前景的方向。 

---
# Training-Free Watermarking for Autoregressive Image Generation 

**Title (ZH)**: 无需训练的自回归图像生成水印技术 

**Authors**: Yu Tong, Zihao Pan, Shuai Yang, Kaiyang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.14673)  

**Abstract**: Invisible image watermarking can protect image ownership and prevent malicious misuse of visual generative models. However, existing generative watermarking methods are mainly designed for diffusion models while watermarking for autoregressive image generation models remains largely underexplored. We propose IndexMark, a training-free watermarking framework for autoregressive image generation models. IndexMark is inspired by the redundancy property of the codebook: replacing autoregressively generated indices with similar indices produces negligible visual differences. The core component in IndexMark is a simple yet effective match-then-replace method, which carefully selects watermark tokens from the codebook based on token similarity, and promotes the use of watermark tokens through token replacement, thereby embedding the watermark without affecting the image quality. Watermark verification is achieved by calculating the proportion of watermark tokens in generated images, with precision further improved by an Index Encoder. Furthermore, we introduce an auxiliary validation scheme to enhance robustness against cropping attacks. Experiments demonstrate that IndexMark achieves state-of-the-art performance in terms of image quality and verification accuracy, and exhibits robustness against various perturbations, including cropping, noises, Gaussian blur, random erasing, color jittering, and JPEG compression. 

**Abstract (ZH)**: 无迹图像水印可以保护图像所有权并防止视觉生成模型的恶意滥用。然而，现有的生成水印方法主要针对扩散模型，而针对自回归图像生成模型的水印研究仍相对不足。我们提出了IndexMark，一种无需训练的自回归图像生成模型水印框架。IndexMark 受码本冗余性启发：用相似的索引替换自回归生成的索引会产生微乎其微的视觉差异。IndexMark 的核心组件是一种简单有效的匹配然后替换方法，该方法根据标记相似性选择水印标记，并通过标记替换促进水印标记的使用，从而在不损害图像质量的情况下嵌入水印。通过计算生成图像中水印标记的比例实现水印验证，精确度进一步通过索引编码器提高。此外，我们引入了辅助验证方案以增强对裁剪攻击的鲁棒性。实验结果显示，IndexMark 在图像质量和验证准确性方面达到了最先进的性能，并且对抗各种扰动（包括裁剪、噪声、高斯模糊、随机擦除、色彩抖动和JPEG压缩）均表现出鲁棒性。 

---
# AKRMap: Adaptive Kernel Regression for Trustworthy Visualization of Cross-Modal Embeddings 

**Title (ZH)**: AKRMap：自适应核回归可信跨模态嵌入可视化 

**Authors**: Yilin Ye, Junchao Huang, Xingchen Zeng, Jiazhi Xia, Wei Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.14664)  

**Abstract**: Cross-modal embeddings form the foundation for multi-modal models. However, visualization methods for interpreting cross-modal embeddings have been primarily confined to traditional dimensionality reduction (DR) techniques like PCA and t-SNE. These DR methods primarily focus on feature distributions within a single modality, whilst failing to incorporate metrics (e.g., CLIPScore) across multiple this http URL paper introduces AKRMap, a new DR technique designed to visualize cross-modal embeddings metric with enhanced accuracy by learning kernel regression of the metric landscape in the projection space. Specifically, AKRMap constructs a supervised projection network guided by a post-projection kernel regression loss, and employs adaptive generalized kernels that can be jointly optimized with the projection. This approach enables AKRMap to efficiently generate visualizations that capture complex metric distributions, while also supporting interactive features such as zoom and overlay for deeper exploration. Quantitative experiments demonstrate that AKRMap outperforms existing DR methods in generating more accurate and trustworthy visualizations. We further showcase the effectiveness of AKRMap in visualizing and comparing cross-modal embeddings for text-to-image models. Code and demo are available at this https URL. 

**Abstract (ZH)**: 跨模态嵌入是多模态模型的基础。然而，用于解读跨模态嵌入的可视化方法主要局限于如PCA和t-SNE等传统的降维技术。这些降维方法主要关注单一模态的特征分布，而未能在多模态之间综合多个指标（如CLIPScore）。本文介绍了一种新的降维技术AKRMap，该技术通过在投影空间中学习度量景观的核回归来增强准确性，以实现跨模态嵌入的可视化。具体而言，AKRMap构建了一个由后投影核回归损失引导的监督投影网络，并使用可以与投影联合优化的自适应通用核。这种方法使AKRMap能够高效地生成捕获复杂度量分布的可视化图，并支持缩放和叠加等交互功能以进行更深入的探索。定量实验表明，AKRMap在生成更准确和可靠的可视化方面优于现有降维方法。我们还展示了AKRMap在文本到图像模型的跨模态嵌入可视化和比较中的有效性和优越性。代码和演示可在以下链接获取。 

---
# Abacus: A Cost-Based Optimizer for Semantic Operator Systems 

**Title (ZH)**: 算盘：面向语义操作系统的成本基优化器 

**Authors**: Matthew Russo, Sivaprasad Sudhir, Gerardo Vitagliano, Chunwei Liu, Tim Kraska, Samuel Madden, Michael Cafarella  

**Link**: [PDF](https://arxiv.org/pdf/2505.14661)  

**Abstract**: LLMs enable an exciting new class of data processing applications over large collections of unstructured documents. Several new programming frameworks have enabled developers to build these applications by composing them out of semantic operators: a declarative set of AI-powered data transformations with natural language specifications. These include LLM-powered maps, filters, joins, etc. used for document processing tasks such as information extraction, summarization, and more. While systems of semantic operators have achieved strong performance on benchmarks, they can be difficult to optimize. An optimizer for this setting must determine how to physically implement each semantic operator in a way that optimizes the system globally. Existing optimizers are limited in the number of optimizations they can apply, and most (if not all) cannot optimize system quality, cost, or latency subject to constraint(s) on the other dimensions. In this paper we present Abacus, an extensible, cost-based optimizer which searches for the best implementation of a semantic operator system given a (possibly constrained) optimization objective. Abacus estimates operator performance by leveraging a minimal set of validation examples and, if available, prior beliefs about operator performance. We evaluate Abacus on document processing workloads in the biomedical and legal domains (BioDEX; CUAD) and multi-modal question answering (MMQA). We demonstrate that systems optimized by Abacus achieve 18.7%-39.2% better quality and up to 23.6x lower cost and 4.2x lower latency than the next best system. 

**Abstract (ZH)**: LLM技术使得在大量非结构化文档集合上处理数据的应用变得令人兴奋。多种新的编程框架允许开发者通过组合语义操作符构建这些应用，这些语义操作符是一个具有自然语言规格的声明式AI驱动的数据转换集合，包括由LLM支持的映射、过滤、连接等操作符，用于文档处理任务如信息抽取、总结等。尽管语义操作符系统在基准测试中表现优秀，但它们在优化方面可能存在问题。在这个环境中，优化器必须确定如何以全局优化的方式物理实现每个语义操作符。现有的优化器在可以应用的优化数量上受到限制，大多数（如果不是全部）优化器无法在满足某些约束条件下优化系统质量、成本或延迟。在本文中，我们提出了Abacus，一个可扩展、基于成本的优化器，可在给定（可能受限的）优化目标的情况下搜索语义操作符系统的最佳实现。Abacus通过利用少量验证示例和可获得的关于操作符性能的先验信念来估计操作符性能。我们在生物医学和法律领域（BioDEX、CUAD）的文档处理工作负载以及多模态问答（MMQA）中评估了Abacus。结果显示，由Abacus优化的系统在质量上提高了18.7%-39.2%，在成本上降低了23.6倍，在延迟上降低了4.2倍，优于次优系统。 

---
# EmoGist: Efficient In-Context Learning for Visual Emotion Understanding 

**Title (ZH)**: EmoGist: 有效的基于上下文的学习方法用于视觉情感理解 

**Authors**: Ronald Seoh, Dan Goldwasser  

**Link**: [PDF](https://arxiv.org/pdf/2505.14660)  

**Abstract**: In this paper, we introduce EmoGist, a training-free, in-context learning method for performing visual emotion classification with LVLMs. The key intuition of our approach is that context-dependent definition of emotion labels could allow more accurate predictions of emotions, as the ways in which emotions manifest within images are highly context dependent and nuanced. EmoGist pre-generates multiple explanations of emotion labels, by analyzing the clusters of example images belonging to each category. At test time, we retrieve a version of explanation based on embedding similarity, and feed it to a fast VLM for classification. Through our experiments, we show that EmoGist allows up to 13 points improvement in micro F1 scores with the multi-label Memotion dataset, and up to 8 points in macro F1 in the multi-class FI dataset. 

**Abstract (ZH)**: 基于上下文学习的EmoGist：一种无需训练的情感图像分类方法 

---
# Explainable AI for Securing Healthcare in IoT-Integrated 6G Wireless Networks 

**Title (ZH)**: 可解释的AI在整合了IoT的6G无线网络中保障医疗服务中应用 

**Authors**: Navneet Kaur, Lav Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.14659)  

**Abstract**: As healthcare systems increasingly adopt advanced wireless networks and connected devices, securing medical applications has become critical. The integration of Internet of Medical Things devices, such as robotic surgical tools, intensive care systems, and wearable monitors has enhanced patient care but introduced serious security risks. Cyberattacks on these devices can lead to life threatening consequences, including surgical errors, equipment failure, and data breaches. While the ITU IMT 2030 vision highlights 6G's transformative role in healthcare through AI and cloud integration, it also raises new security concerns. This paper explores how explainable AI techniques like SHAP, LIME, and DiCE can uncover vulnerabilities, strengthen defenses, and improve trust and transparency in 6G enabled healthcare. We support our approach with experimental analysis and highlight promising results. 

**Abstract (ZH)**: 随着医疗系统越来越多地采用先进的无线网络和连接设备，保障医疗应用的安全已成为关键问题。医疗物联网设备的整合，如机器人手术工具、重症监护系统和可穿戴监测器，虽然提高了患者护理水平，但也引入了严重的安全风险。对这些设备的网络攻击可能导致致命后果，包括手术错误、设备故障和数据泄露。尽管ITU IMT 2030愿景强调6G通过AI和云计算在医疗领域中的变革性作用，但也提出了新的安全挑战。本文探讨了可解释AI技术（如SHAP、LIME和DiCE）如何揭示漏洞、强化防御并提高6G赋能医疗领域的信任和透明度。我们通过实验分析支持我们的方法，并强调了令人鼓舞的结果。 

---
# Beyond Words: Multimodal LLM Knows When to Speak 

**Title (ZH)**: 超越文本：多模态LLM知道何时发言 

**Authors**: Zikai Liao, Yi Ouyang, Yi-Lun Lee, Chen-Ping Yu, Yi-Hsuan Tsai, Zhaozheng Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.14654)  

**Abstract**: While large language model (LLM)-based chatbots have demonstrated strong capabilities in generating coherent and contextually relevant responses, they often struggle with understanding when to speak, particularly in delivering brief, timely reactions during ongoing conversations. This limitation arises largely from their reliance on text input, lacking the rich contextual cues in real-world human dialogue. In this work, we focus on real-time prediction of response types, with an emphasis on short, reactive utterances that depend on subtle, multimodal signals across vision, audio, and text. To support this, we introduce a new multimodal dataset constructed from real-world conversational videos, containing temporally aligned visual, auditory, and textual streams. This dataset enables fine-grained modeling of response timing in dyadic interactions. Building on this dataset, we propose MM-When2Speak, a multimodal LLM-based model that adaptively integrates visual, auditory, and textual context to predict when a response should occur, and what type of response is appropriate. Experiments show that MM-When2Speak significantly outperforms state-of-the-art unimodal and LLM-based baselines, achieving up to a 4x improvement in response timing accuracy over leading commercial LLMs. These results underscore the importance of multimodal inputs for producing timely, natural, and engaging conversational AI. 

**Abstract (ZH)**: 基于大型语言模型的聊天机器人在生成连贯且上下文相关响应方面表现出强大能力，但在理解何时发言，尤其是在线上对话中及时作出简短反应方面常常存在困难。这一限制主要源于它们依赖文本输入，缺乏真实世界人类对话中的丰富语境线索。在本工作中，我们专注于实时预测响应类型，重点是依赖于多重模态信号（包括视觉、听觉和文本）的简短、反应性陈述。为此，我们引入了一个新的多重模态数据集，该数据集基于真实世界的对话视频，包含时间对齐的视觉、听觉和文本流，以精细建模双人互动中的响应时间。基于此数据集，我们提出了一种名为MM-When2Speak的多重模态大型语言模型（LLM）模型，该模型能够适应性地整合视觉、听觉和文本上下文以预测何时应作出响应，以及什么类型的响应最为恰当。实验结果显示，MM-When2Speak在响应时间准确性上显著优于最先进的单模态和LLM基线，相对于领先的商用LLM，响应时间准确性可提高4倍。这些结果强调了多重模态输入对于生成及时、自然且引人入胜的对话AI的重要性。 

---
# CAD-Coder: An Open-Source Vision-Language Model for Computer-Aided Design Code Generation 

**Title (ZH)**: CAD-Coder: 一种面向计算机辅助设计代码生成的开源视觉-语言模型 

**Authors**: Anna C. Doris, Md Ferdous Alam, Amin Heyrani Nobari, Faez Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2505.14646)  

**Abstract**: Efficient creation of accurate and editable 3D CAD models is critical in engineering design, significantly impacting cost and time-to-market in product innovation. Current manual workflows remain highly time-consuming and demand extensive user expertise. While recent developments in AI-driven CAD generation show promise, existing models are limited by incomplete representations of CAD operations, inability to generalize to real-world images, and low output accuracy. This paper introduces CAD-Coder, an open-source Vision-Language Model (VLM) explicitly fine-tuned to generate editable CAD code (CadQuery Python) directly from visual input. Leveraging a novel dataset that we created--GenCAD-Code, consisting of over 163k CAD-model image and code pairs--CAD-Coder outperforms state-of-the-art VLM baselines such as GPT-4.5 and Qwen2.5-VL-72B, achieving a 100% valid syntax rate and the highest accuracy in 3D solid similarity. Notably, our VLM demonstrates some signs of generalizability, successfully generating CAD code from real-world images and executing CAD operations unseen during fine-tuning. The performance and adaptability of CAD-Coder highlights the potential of VLMs fine-tuned on code to streamline CAD workflows for engineers and designers. CAD-Coder is publicly available at: this https URL. 

**Abstract (ZH)**: 高效创建准确可编辑的3D CAD模型对于工程设计至关重要，显著影响产品创新的成本和时间。当前的手动工作流程仍然高度耗时，并需要大量的用户专业知识。尽管AI驱动的CAD生成技术前景看好，但现有模型受限于不完整的CAD操作表示、无法泛化到真实世界图像以及较低的输出准确性。本文介绍了CAD-Coder，一个专门微调以直接从视觉输入生成可编辑CAD代码（CadQuery Python）的开源视觉语言模型（VLM）。利用我们创建的一个新颖的数据集GenCAD-Code，包含超过16.3万对CAD模型图像和代码——CAD-Coder在状态最先进VLM基线如GPT-4.5和Qwen2.5-VL-72B上表现出色，实现了100%的有效语法率和最高的三维实体相似性准确性。值得注意的是，我们的VLM显示出一些泛化能力，在微调期间未见过的CAD操作中成功从真实世界图像生成CAD代码并执行相应操作。CAD-Coder的性能和适应性突显了针对代码进行微调的VLMs为工程师和设计师简化CAD工作流的潜力。CAD-Coder可在以下链接获取：this https URL。 

---
# Will AI Tell Lies to Save Sick Children? Litmus-Testing AI Values Prioritization with AIRiskDilemmas 

**Title (ZH)**: AI会为了挽救生病儿童而说谎吗？通过AIRiskDilemmas测试AI价值观优先级 

**Authors**: Yu Ying Chiu, Zhilin Wang, Sharan Maiya, Yejin Choi, Kyle Fish, Sydney Levine, Evan Hubinger  

**Link**: [PDF](https://arxiv.org/pdf/2505.14633)  

**Abstract**: Detecting AI risks becomes more challenging as stronger models emerge and find novel methods such as Alignment Faking to circumvent these detection attempts. Inspired by how risky behaviors in humans (i.e., illegal activities that may hurt others) are sometimes guided by strongly-held values, we believe that identifying values within AI models can be an early warning system for AI's risky behaviors. We create LitmusValues, an evaluation pipeline to reveal AI models' priorities on a range of AI value classes. Then, we collect AIRiskDilemmas, a diverse collection of dilemmas that pit values against one another in scenarios relevant to AI safety risks such as Power Seeking. By measuring an AI model's value prioritization using its aggregate choices, we obtain a self-consistent set of predicted value priorities that uncover potential risks. We show that values in LitmusValues (including seemingly innocuous ones like Care) can predict for both seen risky behaviors in AIRiskDilemmas and unseen risky behaviors in HarmBench. 

**Abstract (ZH)**: 随着更强的模型出现并采用如对齐欺骗等 novel 方法来规避这些检测尝试，检测 AI 风险变得更加具有挑战性。受人类危险行为（即可能伤害他人的违法活动）有时由坚定的价值观所引导的启发，我们认为识别 AI 模型中的价值可以作为 AI 危险行为的早期预警系统。我们创建了 LitmusValues 评估流水线，以揭示 AI 模型在一系列 AI 价值类别的优先级。然后，我们收集了 AIRiskDilemmas，这是一个包含各种困境的集合，这些困境在与 AI 安全风险相关的情景中将价值相互对立，例如权力追求。通过测量 AI 模型的价值优先级来评估其集体选择，我们获得了一致的价值优先级预测集，揭示潜在风险。我们展示 LitmusValues 中的价值（包括看似无害的价值如关怀）可以预测 AIRiskDilemmas 中已知的危险行为和 HarmBench 中未见的危险行为。 

---
# KERL: Knowledge-Enhanced Personalized Recipe Recommendation using Large Language Models 

**Title (ZH)**: KERL：基于大型语言模型的知识增强个性化食谱推荐 

**Authors**: Fnu Mohbat, Mohammed J Zaki  

**Link**: [PDF](https://arxiv.org/pdf/2505.14629)  

**Abstract**: Recent advances in large language models (LLMs) and the abundance of food data have resulted in studies to improve food understanding using LLMs. Despite several recommendation systems utilizing LLMs and Knowledge Graphs (KGs), there has been limited research on integrating food related KGs with LLMs. We introduce KERL, a unified system that leverages food KGs and LLMs to provide personalized food recommendations and generates recipes with associated micro-nutritional information. Given a natural language question, KERL extracts entities, retrieves subgraphs from the KG, which are then fed into the LLM as context to select the recipes that satisfy the constraints. Next, our system generates the cooking steps and nutritional information for each recipe. To evaluate our approach, we also develop a benchmark dataset by curating recipe related questions, combined with constraints and personal preferences. Through extensive experiments, we show that our proposed KG-augmented LLM significantly outperforms existing approaches, offering a complete and coherent solution for food recommendation, recipe generation, and nutritional analysis. Our code and benchmark datasets are publicly available at this https URL. 

**Abstract (ZH)**: Recent advances in大型语言模型(LLMs)和食品数据的丰富使得利用LLMs提升食品理解的研究日益增多。尽管已有若干利用LLMs和知识图谱(KGs)的推荐系统，但将食品相关的KGs与LLMs集成的研究相对有限。我们提出了KERL，这是一种结合食品KGs和LLMs的统一系统，用于提供个性化的食品推荐并生成包含微营养信息的食谱。给定自然语言问题，KERL抽取实体、从KG中检索子图，这些子图随后作为上下文输入到LLM中以选择满足约束条件的食谱。接下来，我们的系统生成每份食谱的烹饪步骤和营养信息。为评估我们的方法，我们还开发了一个基准数据集，该数据集由食谱相关问题、约束条件和个人偏好组成。通过广泛的实验，我们证明了我们提出的增强KG的LLM显著优于现有方法，提供了食品推荐、食谱生成和营养分析的完整而连贯的解决方案。我们的代码和基准数据集可在以下网址公开获取。 

---
# TinyV: Reducing False Negatives in Verification Improves RL for LLM Reasoning 

**Title (ZH)**: TinyV: 减少验证中的假阴性改进大语言模型推理的RL方法 

**Authors**: Zhangchen Xu, Yuetai Li, Fengqing Jiang, Bhaskar Ramasubramanian, Luyao Niu, Bill Yuchen Lin, Radha Poovendran  

**Link**: [PDF](https://arxiv.org/pdf/2505.14625)  

**Abstract**: Reinforcement Learning (RL) has become a powerful tool for enhancing the reasoning abilities of large language models (LLMs) by optimizing their policies with reward signals. Yet, RL's success relies on the reliability of rewards, which are provided by verifiers. In this paper, we expose and analyze a widespread problem--false negatives--where verifiers wrongly reject correct model outputs. Our in-depth study of the Big-Math-RL-Verified dataset reveals that over 38% of model-generated responses suffer from false negatives, where the verifier fails to recognize correct answers. We show, both empirically and theoretically, that these false negatives severely impair RL training by depriving the model of informative gradient signals and slowing convergence. To mitigate this, we propose tinyV, a lightweight LLM-based verifier that augments existing rule-based methods, which dynamically identifies potential false negatives and recovers valid responses to produce more accurate reward estimates. Across multiple math-reasoning benchmarks, integrating TinyV boosts pass rates by up to 10% and accelerates convergence relative to the baseline. Our findings highlight the critical importance of addressing verifier false negatives and offer a practical approach to improve RL-based fine-tuning of LLMs. Our code is available at this https URL. 

**Abstract (ZH)**: 强化学习（RL）已成为通过奖励信号优化大型语言模型（LLMs）推理能力的一种强大工具。然而，RL的成功依赖于验证器提供的奖励信号的可靠性。在本文中，我们揭示并分析了一个普遍存在的问题——假阴性，即验证器错误地拒绝了正确的模型输出。通过对Big-Math-RL-Verified数据集的深入研究，我们发现超过38%的模型生成响应受到了假阴性的影响，验证器未能识别出正确的答案。我们通过实证和理论分析表明，这些假阴性严重阻碍了RL训练，剥夺了模型获取有用梯度信号的机会，并减缓了收敛速度。为了缓解这一问题，我们提出了tinyV，一种轻量级的基于LLM的验证器，它可以增强现有的基于规则的方法，动态识别潜在的假阴性，并恢复有效的响应以生成更准确的奖励估计。在多个数学推理基准测试中，集成TinyV可以提高通过率高达10%并加速收敛，相对于基线方法。我们的研究突出了解决验证器假阴性的关键重要性，并提供了一种实际方法来改善基于RL的LLM微调。代码已发布在此：https://github.com/alibaba/Qwen-tinyV 

---
# Language Models Optimized to Fool Detectors Still Have a Distinct Style (And How to Change It) 

**Title (ZH)**: 语言模型优化以欺骗检测器仍然具有独特的风格（以及如何改变它） 

**Authors**: Rafael Rivera Soto, Barry Chen, Nicholas Andrews  

**Link**: [PDF](https://arxiv.org/pdf/2505.14608)  

**Abstract**: Despite considerable progress in the development of machine-text detectors, it has been suggested that the problem is inherently hard, and therefore, that stakeholders should proceed under the assumption that machine-generated text cannot be reliably detected as such. We examine a recent such claim by Nicks et al. (2024) regarding the ease with which language models can be optimized to degrade the performance of machine-text detectors, including detectors not specifically optimized against. We identify a feature space$\unicode{x2013}$the stylistic feature space$\unicode{x2013}$that is robust to such optimization, and show that it may be used to reliably detect samples from language models optimized to prevent detection. Furthermore, we show that even when models are explicitly optimized against stylistic detectors, detection performance remains surprisingly unaffected. We then seek to understand if stylistic detectors are inherently more robust. To study this question, we explore a new paraphrasing approach that simultaneously aims to close the gap between human writing and machine writing in stylistic feature space while avoiding detection using traditional features. We show that when only a single sample is available for detection, this attack is universally effective across all detectors considered, including those that use writing style. However, as the number of samples available for detection grows, the human and machine distributions become distinguishable. This observation encourages us to introduce AURA, a metric that estimates the overlap between human and machine-generated distributions by analyzing how detector performance improves as more samples become available. Overall, our findings underscore previous recommendations to avoid reliance on machine-text detection. 

**Abstract (ZH)**: 尽管在机器文本检测器的发展方面取得了显著进展，有人认为这个问题本质上很难解决，因此建议利益相关方在假设机器生成的文本不可靠地被检测出的情况下进行操作。我们审视了Nicks等人（2024）关于语言模型可以通过优化大幅削弱机器文本检测器性能的近期主张，包括那些未特别针对其优化的检测器。我们确定了一个稳健的特征空间——风格特征空间——并展示了它可以用于可靠地检测那些旨在防止被检测的语言模型生成的样本。此外，我们证明即使模型被明确优化以对抗风格检测器，其检测性能仍然出乎意料地未受影响。我们随后试图理解风格检测器是否更加内在地稳健。为研究这一问题，我们探索了一种新的改写方法，旨在同时缩小人类写作与机器写作在风格特征空间中的差距，同时避免使用传统特征被检测。我们表明，当只有单个样本可用于检测时，这种攻击对所有考虑的检测器都是普遍有效的，包括那些使用文本风格的检测器。然而，随着可用样本数量的增加，人类和机器生成的样本分布变得可区分。这一观察促使我们提出了AURA度量，该度量通过分析检测器性能随可用样本数量增加而改善的方式，估计了人类生成和机器生成分布之间的重叠。总体而言，我们的研究结果再次强调了避免依赖机器文本检测的先前建议。 

---
# Toward Reliable Biomedical Hypothesis Generation: Evaluating Truthfulness and Hallucination in Large Language Models 

**Title (ZH)**: 向可靠的生物医学假设生成迈进：评估大规模语言模型的truthfulness和hallucination 

**Authors**: Guangzhi Xiong, Eric Xie, Corey Williams, Myles Kim, Amir Hassan Shariatmadari, Sikun Guo, Stefan Bekiranov, Aidong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14599)  

**Abstract**: Large language models (LLMs) have shown significant potential in scientific disciplines such as biomedicine, particularly in hypothesis generation, where they can analyze vast literature, identify patterns, and suggest research directions. However, a key challenge lies in evaluating the truthfulness of generated hypotheses, as verifying their accuracy often requires substantial time and resources. Additionally, the hallucination problem in LLMs can lead to the generation of hypotheses that appear plausible but are ultimately incorrect, undermining their reliability. To facilitate the systematic study of these challenges, we introduce TruthHypo, a benchmark for assessing the capabilities of LLMs in generating truthful biomedical hypotheses, and KnowHD, a knowledge-based hallucination detector to evaluate how well hypotheses are grounded in existing knowledge. Our results show that LLMs struggle to generate truthful hypotheses. By analyzing hallucinations in reasoning steps, we demonstrate that the groundedness scores provided by KnowHD serve as an effective metric for filtering truthful hypotheses from the diverse outputs of LLMs. Human evaluations further validate the utility of KnowHD in identifying truthful hypotheses and accelerating scientific discovery. Our data and source code are available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在生物医学等科学学科中展示了显著潜力，特别是在假设生成方面，它们可以分析大量文献、识别模式并建议研究方向。然而，验证生成假设的准确性往往需要大量时间和资源，这构成了一个关键挑战。此外，LLMs中的幻觉问题可能导致生成看似合理但实际上错误的假设，这削弱了它们的可靠性。为了促进对这些挑战的系统研究，我们引入了TruthHypo基准，用于评估LLMs生成真实生物医学假设的能力，以及KnowHD知识为基础的幻觉检测器，以评估假设与现有知识的关联程度。我们的研究结果表明，LLMs难以生成真实假设。通过对推理步骤中的幻觉进行分析，我们证明了KnowHD提供的接地得分充当了有效指标，用于筛选来自LLMs多样化输出的真实假设。人类评估进一步验证了KnowHD在识别真实假设和加速科学发现方面的有用性。我们的数据和源代码可在以下网址获取。 

---
# KIPPO: Koopman-Inspired Proximal Policy Optimization 

**Title (ZH)**: KIPPO: Koopman启发的近端策略优化 

**Authors**: Andrei Cozma, Landon Harris, Hairong Qi  

**Link**: [PDF](https://arxiv.org/pdf/2505.14566)  

**Abstract**: Reinforcement Learning (RL) has made significant strides in various domains, and policy gradient methods like Proximal Policy Optimization (PPO) have gained popularity due to their balance in performance, training stability, and computational efficiency. These methods directly optimize policies through gradient-based updates. However, developing effective control policies for environments with complex and non-linear dynamics remains a challenge. High variance in gradient estimates and non-convex optimization landscapes often lead to unstable learning trajectories. Koopman Operator Theory has emerged as a powerful framework for studying non-linear systems through an infinite-dimensional linear operator that acts on a higher-dimensional space of measurement functions. In contrast with their non-linear counterparts, linear systems are simpler, more predictable, and easier to analyze. In this paper, we present Koopman-Inspired Proximal Policy Optimization (KIPPO), which learns an approximately linear latent-space representation of the underlying system's dynamics while retaining essential features for effective policy learning. This is achieved through a Koopman-approximation auxiliary network that can be added to the baseline policy optimization algorithms without altering the architecture of the core policy or value function. Extensive experimental results demonstrate consistent improvements over the PPO baseline with 6-60% increased performance while reducing variability by up to 91% when evaluated on various continuous control tasks. 

**Abstract (ZH)**: 基于柯普曼理论的增强学习proximal策略优化（KIPPO） 

---
# Bellman operator convergence enhancements in reinforcement learning algorithms 

**Title (ZH)**: 贝尔曼运算子收敛性增强在强化学习算法中的应用 

**Authors**: David Krame Kadurha, Domini Jocema Leko Moutouo, Yae Ulrich Gaba  

**Link**: [PDF](https://arxiv.org/pdf/2505.14564)  

**Abstract**: This paper reviews the topological groundwork for the study of reinforcement learning (RL) by focusing on the structure of state, action, and policy spaces. We begin by recalling key mathematical concepts such as complete metric spaces, which form the foundation for expressing RL problems. By leveraging the Banach contraction principle, we illustrate how the Banach fixed-point theorem explains the convergence of RL algorithms and how Bellman operators, expressed as operators on Banach spaces, ensure this convergence. The work serves as a bridge between theoretical mathematics and practical algorithm design, offering new approaches to enhance the efficiency of RL. In particular, we investigate alternative formulations of Bellman operators and demonstrate their impact on improving convergence rates and performance in standard RL environments such as MountainCar, CartPole, and Acrobot. Our findings highlight how a deeper mathematical understanding of RL can lead to more effective algorithms for decision-making problems. 

**Abstract (ZH)**: 本文回顾了强化学习（RL）研究中的拓扑基础，重点关注状态、动作和策略空间的结构。我们首先回顾了完备度量空间等关键数学概念，这些概念构成了表达RL问题的基础。通过利用布劳切克收缩原理，我们解释了布劳切克不动点定理如何解释RL算法的收敛性，并展示了作为巴纳赫空间上算子的贝尔曼算子如何确保这一收敛性。这项工作架起了理论数学与实际算法设计之间的桥梁，提供了改进RL效率的新方法。特别地，我们探讨了贝尔曼算子的替代形式，并展示了它们如何在诸如MountainCar、CartPole和Acrobot等标准RL环境中提高收敛速度和性能。我们的研究发现突显了对RL进行更深层次的数学理解如何可能导致更有效的决策问题算法。 

---
# SSPS: Self-Supervised Positive Sampling for Robust Self-Supervised Speaker Verification 

**Title (ZH)**: SSPS: 自监督正样本采样以实现鲁棒的自监督说话人验证 

**Authors**: Theo Lepage, Reda Dehak  

**Link**: [PDF](https://arxiv.org/pdf/2505.14561)  

**Abstract**: Self-Supervised Learning (SSL) has led to considerable progress in Speaker Verification (SV). The standard framework uses same-utterance positive sampling and data-augmentation to generate anchor-positive pairs of the same speaker. This is a major limitation, as this strategy primarily encodes channel information from the recording condition, shared by the anchor and positive. We propose a new positive sampling technique to address this bottleneck: Self-Supervised Positive Sampling (SSPS). For a given anchor, SSPS aims to find an appropriate positive, i.e., of the same speaker identity but a different recording condition, in the latent space using clustering assignments and a memory queue of positive embeddings. SSPS improves SV performance for both SimCLR and DINO, reaching 2.57% and 2.53% EER, outperforming SOTA SSL methods on VoxCeleb1-O. In particular, SimCLR-SSPS achieves a 58% EER reduction by lowering intra-speaker variance, providing comparable performance to DINO-SSPS. 

**Abstract (ZH)**: 自监督学习（SSL）在演讲者验证（SV）中取得了显著进展。标准框架通过同一句话的正样本采样和数据增强生成同发言人的锚-正样本对。这是一大局限性，因为这种策略主要编码录音条件下的信道信息，而这对锚样本和正样本是共享的。我们提出了一种新的正样本采样技术以解决这一瓶颈：自监督正样本采样（SSPS）。对于给定的锚样本，SSPS旨在在潜在空间中通过聚类分配和正样本嵌入的存储队列找到合适的正样本，即同发言人物identity但不同录音条件的样本。SSPS提高了SimCLR和DINO的SV性能，分别达到2.57%和2.53%的EER，并在VoxCeleb1-O上优于最新的SSL方法。特别是，SimCLR-SSPS通过对内发言人差异性的降低实现了58%的EER减少，提供与DINO-SSPS相当的性能。 

---
# Physics-Guided Learning of Meteorological Dynamics for Weather Downscaling and Forecasting 

**Title (ZH)**: 基于物理的气象动力学学习方法及其在天气降尺度和预报中的应用 

**Authors**: Yingtao Luo, Shikai Fang, Binqing Wu, Qingsong Wen, Liang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.14555)  

**Abstract**: Weather forecasting is essential but remains computationally intensive and physically incomplete in traditional numerical weather prediction (NWP) methods. Deep learning (DL) models offer efficiency and accuracy but often ignore physical laws, limiting interpretability and generalization. We propose PhyDL-NWP, a physics-guided deep learning framework that integrates physical equations with latent force parameterization into data-driven models. It predicts weather variables from arbitrary spatiotemporal coordinates, computes physical terms via automatic differentiation, and uses a physics-informed loss to align predictions with governing dynamics. PhyDL-NWP enables resolution-free downscaling by modeling weather as a continuous function and fine-tunes pre-trained models with minimal overhead, achieving up to 170x faster inference with only 55K parameters. Experiments show that PhyDL-NWP improves both forecasting performance and physical consistency. 

**Abstract (ZH)**: 基于物理指导的深度学习天气预报框架 

---
# KORGym: A Dynamic Game Platform for LLM Reasoning Evaluation 

**Title (ZH)**: KORGym：一种动态游戏平台，用于LLM推理评估 

**Authors**: Jiajun Shi, Jian Yang, Jiaheng Liu, Xingyuan Bu, Jiangjie Chen, Junting Zhou, Kaijing Ma, Zhoufutu Wen, Bingli Wang, Yancheng He, Liang Song, Hualei Zhu, Shilong Li, Xingjian Wang, Wei Zhang, Ruibin Yuan, Yifan Yao, Wenjun Yang, Yunli Wang, Siyuan Fang, Siyu Yuan, Qianyu He, Xiangru Tang, Yingshui Tan, Wangchunshu Zhou, Zhaoxiang Zhang, Zhoujun Li, Wenhao Huang, Ge Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14552)  

**Abstract**: Recent advancements in large language models (LLMs) underscore the need for more comprehensive evaluation methods to accurately assess their reasoning capabilities. Existing benchmarks are often domain-specific and thus cannot fully capture an LLM's general reasoning potential. To address this limitation, we introduce the Knowledge Orthogonal Reasoning Gymnasium (KORGym), a dynamic evaluation platform inspired by KOR-Bench and Gymnasium. KORGym offers over fifty games in either textual or visual formats and supports interactive, multi-turn assessments with reinforcement learning scenarios. Using KORGym, we conduct extensive experiments on 19 LLMs and 8 VLMs, revealing consistent reasoning patterns within model families and demonstrating the superior performance of closed-source models. Further analysis examines the effects of modality, reasoning strategies, reinforcement learning techniques, and response length on model performance. We expect KORGym to become a valuable resource for advancing LLM reasoning research and developing evaluation methodologies suited to complex, interactive environments. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）放大了对其推理能力进行更全面评估方法的需求。现有的基准通常具有领域特定性，因此无法全面捕捉LLM的一般推理潜力。为解决这一局限，我们引入了知识正交推理体育馆（KORGym），这是一个受到KOR-Bench和体育馆启发的动力评估平台。KORGym提供了五十多种以文本或视觉形式呈现的游戏，并支持基于强化学习场景的交互式多轮评估。使用KORGym，我们在19种LLM和8种VLM上进行了广泛的实验，揭示了模型家族中的稳健推理模式，并展示了闭源模型的优越性能。进一步的分析探索了模态、推理策略、强化学习技术及响应长度对模型性能的影响。我们期望KORGym能够成为推进LLM推理研究和开发适用于复杂交互环境的评估方法的重要资源。 

---
# Trustworthy Reputation Games and Applications to Proof-of-Reputation Blockchains 

**Title (ZH)**: 可信赖的声誉博弈及其在声誉证明区块链中的应用 

**Authors**: Petros Drineas, Rohit Nema, Rafail Ostrovsky, Vassilis Zikas  

**Link**: [PDF](https://arxiv.org/pdf/2505.14551)  

**Abstract**: Reputation systems play an essential role in the Internet era, as they enable people to decide whom to trust, by collecting and aggregating data about users' behavior. Recently, several works proposed the use of reputation for the design and scalability improvement of decentralized (blockchain) ledgers; however, such systems are prone to manipulation and to our knowledge no game-theoretic treatment exists that can support their economic robustness.
In this work we put forth a new model for the design of what we call, {\em trustworthy reputation systems}. Concretely, we describe a class of games, which we term {\em trustworthy reputation games}, that enable a set of users to report a function of their beliefs about the trustworthiness of each server in a set -- i.e., their estimate of the probability that this server will behave according to its specified strategy -- in a way that satisfies the following properties:
1. It is $(\epsilon$-)best response for any rational user in the game to play a prescribed (truthful) strategy according to their true belief.
2. Assuming that the users' beliefs are not too far from the {\em true} trustworthiness of the servers, playing the above ($\epsilon-$)Nash equilibrium allows anyone who observes the users' strategies to estimate the relative trustworthiness of any two servers.
Our utilities and decoding function build on a connection between the well known PageRank algorithm and the problem of trustworthiness discovery, which can be of independent interest. Finally, we show how the above games are motivated by and can be leveraged in proof-of-reputation (PoR) blockchains. 

**Abstract (ZH)**: 可信任的声誉系统设计模型 

---
# Can Large Language Models Really Recognize Your Name? 

**Title (ZH)**: 大型语言模型真的能识别你的名字吗？ 

**Authors**: Dzung Pham, Peter Kairouz, Niloofar Mireshghallah, Eugene Bagdasarian, Chau Minh Pham, Amir Houmansadr  

**Link**: [PDF](https://arxiv.org/pdf/2505.14549)  

**Abstract**: Large language models (LLMs) are increasingly being used to protect sensitive user data. However, current LLM-based privacy solutions assume that these models can reliably detect personally identifiable information (PII), particularly named entities. In this paper, we challenge that assumption by revealing systematic failures in LLM-based privacy tasks. Specifically, we show that modern LLMs regularly overlook human names even in short text snippets due to ambiguous contexts, which cause the names to be misinterpreted or mishandled. We propose AMBENCH, a benchmark dataset of seemingly ambiguous human names, leveraging the name regularity bias phenomenon, embedded within concise text snippets along with benign prompt injections. Our experiments on modern LLMs tasked to detect PII as well as specialized tools show that recall of ambiguous names drops by 20--40% compared to more recognizable names. Furthermore, ambiguous human names are four times more likely to be ignored in supposedly privacy-preserving summaries generated by LLMs when benign prompt injections are present. These findings highlight the underexplored risks of relying solely on LLMs to safeguard user privacy and underscore the need for a more systematic investigation into their privacy failure modes. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地被用于保护敏感用户数据。然而，当前基于LLM的隐私解决方案假设这些模型能可靠地检测个人信息（PII），特别是命名实体。在本文中，我们通过揭示基于LLM的隐私任务中的系统性失败，挑战这一假设。具体而言，我们表明，现代LLMs在短文本片段中经常由于语境模糊而疏忽人类姓名，导致姓名被误释或误处理。我们提出了AMBENCH，一个包含看似模糊的人类姓名基准数据集，利用名称规律性偏差现象，并嵌入简洁的文本片段和良性提示注入。我们的实验表明，当现代LLM被任务驱动以检测PII以及使用专用工具时，模糊姓名的召回率相较于更可识别的姓名下降了20%-40%。此外，在存在良性提示注入的情况下，模糊的人类姓名被LLM生成的所谓隐私保护总结忽略的可能性是后者的大四倍。这些发现突显了仅依赖LLM保护用户隐私所忽视的风险，并强调了对它们隐私失败模式进行更系统性调查的必要性。 

---
# Energy-Efficient Deep Reinforcement Learning with Spiking Transformers 

**Title (ZH)**: 能源高效的大规模强化学习变换器 

**Authors**: Mohammad Irfan Uddin, Nishad Tasnim, Md Omor Faruk, Zejian Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.14533)  

**Abstract**: Agent-based Transformers have been widely adopted in recent reinforcement learning advances due to their demonstrated ability to solve complex tasks. However, the high computational complexity of Transformers often results in significant energy consumption, limiting their deployment in real-world autonomous systems. Spiking neural networks (SNNs), with their biologically inspired structure, offer an energy-efficient alternative for machine learning. In this paper, a novel Spike-Transformer Reinforcement Learning (STRL) algorithm that combines the energy efficiency of SNNs with the powerful decision-making capabilities of reinforcement learning is developed. Specifically, an SNN using multi-step Leaky Integrate-and-Fire (LIF) neurons and attention mechanisms capable of processing spatio-temporal patterns over multiple time steps is designed. The architecture is further enhanced with state, action, and reward encodings to create a Transformer-like structure optimized for reinforcement learning tasks. Comprehensive numerical experiments conducted on state-of-the-art benchmarks demonstrate that the proposed SNN Transformer achieves significantly improved policy performance compared to conventional agent-based Transformers. With both enhanced energy efficiency and policy optimality, this work highlights a promising direction for deploying bio-inspired, low-cost machine learning models in complex real-world decision-making scenarios. 

**Abstract (ZH)**: 基于神经元 spike 的变压器在强化学习中的新型节能算法（Spike-Transformer Reinforcement Learning, STRL）研究 

---
# NavBench: A Unified Robotics Benchmark for Reinforcement Learning-Based Autonomous Navigation 

**Title (ZH)**: NavBench: 基于强化学习的自主导航统一机器人基准测试 

**Authors**: Matteo El-Hariry, Antoine Richard, Ricard M. Castan, Luis F. W. Batista, Matthieu Geist, Cedric Pradalier, Miguel Olivares-Mendez  

**Link**: [PDF](https://arxiv.org/pdf/2505.14526)  

**Abstract**: Autonomous robots must navigate and operate in diverse environments, from terrestrial and aquatic settings to aerial and space domains. While Reinforcement Learning (RL) has shown promise in training policies for specific autonomous robots, existing benchmarks are often constrained to unique platforms, limiting generalization and fair comparisons across different mobility systems. In this paper, we present NavBench, a multi-domain benchmark for training and evaluating RL-based navigation policies across diverse robotic platforms and operational environments. Built on IsaacLab, our framework standardizes task definitions, enabling different robots to tackle various navigation challenges without the need for ad-hoc task redesigns or custom evaluation metrics. Our benchmark addresses three key challenges: (1) Unified cross-medium benchmarking, enabling direct evaluation of diverse actuation methods (thrusters, wheels, water-based propulsion) in realistic environments; (2) Scalable and modular design, facilitating seamless robot-task interchangeability and reproducible training pipelines; and (3) Robust sim-to-real validation, demonstrated through successful policy transfer to multiple real-world robots, including a satellite robotic simulator, an unmanned surface vessel, and a wheeled ground vehicle. By ensuring consistency between simulation and real-world deployment, NavBench simplifies the development of adaptable RL-based navigation strategies. Its modular design allows researchers to easily integrate custom robots and tasks by following the framework's predefined templates, making it accessible for a wide range of applications. Our code is publicly available at NavBench. 

**Abstract (ZH)**: 自主机器人必须在多样化的环境中导航和运行，从陆地和水下环境到空中和太空领域。尽管强化学习（RL）在训练特定自主机器人的策略方面显示出潜力，但现有的基准测试通常局限于独特的平台，限制了不同移动系统之间的泛化和公平比较。在本文中，我们提出了NavBench，这是一个多领域的基准测试，用于跨多样化的机器人平台和运行环境训练和评估基于RL的导航策略。基于IsaacLab，我们的框架标准化了任务定义，使得不同的机器人能够应对各种导航挑战，无需进行特定任务的重新设计或自定义评估指标。我们的基准测试解决了三个关键挑战：（1）统一的跨介质基准测试，允许直接评估多种不同的执行方法（喷水推进、车轮、水基推进）在真实环境中的表现；（2）可扩展和模块化的设计，方便机器人和任务的无缝互换及可再现的训练管道；（3）稳健的仿真到现实的验证，通过多个实际机器人的成功策略转移得到验证，包括卫星机器人模拟器、无人表面船舶和轮式地面车辆。通过确保模拟与实际部署之间的一致性，NavBench 简化了适应性强的基于RL的导航策略的开发。其模块化设计允许研究人员通过遵循框架预定义的模板轻松集成自定义机器人和任务，使其适用于广泛的用途。我们的代码可在NavBench公开获取。 

---
# Exploring Graph Representations of Logical Forms for Language Modeling 

**Title (ZH)**: 探索逻辑形式的图表示用于语言 modeling 

**Authors**: Michael Sullivan  

**Link**: [PDF](https://arxiv.org/pdf/2505.14523)  

**Abstract**: We make the case for language models over logical forms (LFLMs), arguing that such models are more data-efficient than their textual counterparts. To that end, we introduce the Graph-based Formal-Logical Distributional Semantics (GFoLDS) prototype, a pretrained LM over graph representations of logical forms, as a proof-of-concept of LFLMs. Using GFoLDS, we present strong experimental evidence that LFLMs can leverage the built-in, basic linguistic knowledge inherent in such models to immediately begin learning more complex patterns. On downstream tasks, we show that GFoLDS vastly outperforms textual, transformer LMs pretrained on similar amounts of data, indicating that LFLMs can learn with substantially less data than models over plain text. Furthermore, we show that the performance of this model is likely to scale with additional parameters and pretraining data, suggesting the viability of LFLMs in real-world applications. 

**Abstract (ZH)**: 我们主张语言模型优于逻辑形式（LFLMs），认为这类模型在数据效率方面优于其文本对应模型。为此，我们介绍了基于图表示逻辑形式的预训练语言模型（GFoLDS）原型，作为LFLMs的一个概念验证。通过GFoLDS，我们展示了强有力的实验证据，表明LFLMs能够利用这些模型内置的基础语言知识，立即开始学习更复杂的模式。在下游任务上，我们展示GFoLDS在使用相似量级数据预训练的情况下，远超文本变压器LM，表明LFLMs可以通过显著较少的数据进行学习。此外，我们展示了该模型的性能可能随着参数量和预训练数据的增加而扩展，这表明LFLMs在实际应用中的可行性。 

---
# Latent Flow Transformer 

**Title (ZH)**: 隐含流变换器 

**Authors**: Yen-Chen Wu, Feng-Ting Liao, Meng-Hsi Chen, Pei-Chen Ho, Farhang Nabiei, Da-shan Shiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14513)  

**Abstract**: Transformers, the standard implementation for large language models (LLMs), typically consist of tens to hundreds of discrete layers. While more layers can lead to better performance, this approach has been challenged as far from efficient, especially given the superiority of continuous layers demonstrated by diffusion and flow-based models for image generation. We propose the Latent Flow Transformer (LFT), which replaces a block of layers with a single learned transport operator trained via flow matching, offering significant compression while maintaining compatibility with the original architecture. Additionally, we address the limitations of existing flow-based methods in \textit{preserving coupling} by introducing the Flow Walking (FW) algorithm. On the Pythia-410M model, LFT trained with flow matching compresses 6 of 24 layers and outperforms directly skipping 2 layers (KL Divergence of LM logits at 0.407 vs. 0.529), demonstrating the feasibility of this design. When trained with FW, LFT further distills 12 layers into one while reducing the KL to 0.736 surpassing that from skipping 3 layers (0.932), significantly narrowing the gap between autoregressive and flow-based generation paradigms. 

**Abstract (ZH)**: Transformers：大型语言模型（LLMs）的标准实现通常由数十到数百个离散层组成。虽然更多的层可以提高性能，但这种方法因图像生成中扩散和流化模型显示的连续层优势而受到质疑。我们提出了一种名为语义流变换器（LFT）的方法，用一个通过流匹配训练的学习传输算子替代了一块层，实现了显著压缩同时保持与原始架构的兼容性。此外，我们通过引入流行走（FW）算法解决了现有流化方法在保耦合方面的局限性。在Pythia-410M模型上，使用流匹配训练的LFT压缩了24层中的6层，并优于直接跳过2层（语言模型 logits的KL散度为0.407 vs. 0.529），展示了该设计的可行性。使用FW训练时，LFT进一步将12层凝练为一层，KL降低至0.736，超过了跳过3层的结果（0.932），显著缩小了自回归生成与流化生成之间的差距。 

---
# ModRWKV: Transformer Multimodality in Linear Time 

**Title (ZH)**: ModRWKV: 在线性时间内实现Transformer多模态技术 

**Authors**: Jiale Kang, Ziyin Yue, Qingyu Yin, Jiang Rui, Weile Li, Zening Lu, Zhouran Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.14505)  

**Abstract**: Currently, most multimodal studies are based on large language models (LLMs) with quadratic-complexity Transformer architectures. While linear models like RNNs enjoy low inference costs, their application has been largely limited to the text-only modality. This work explores the capabilities of modern RNN architectures in multimodal contexts. We propose ModRWKV-a decoupled multimodal framework built upon the RWKV7 architecture as its LLM backbone-which achieves multi-source information fusion through dynamically adaptable heterogeneous modality encoders. We designed the multimodal modules in ModRWKV with an extremely lightweight architecture and, through extensive experiments, identified a configuration that achieves an optimal balance between performance and computational efficiency. ModRWKV leverages the pretrained weights of the RWKV7 LLM for initialization, which significantly accelerates multimodal training. Comparative experiments with different pretrained checkpoints further demonstrate that such initialization plays a crucial role in enhancing the model's ability to understand multimodal signals. Supported by extensive experiments, we conclude that modern RNN architectures present a viable alternative to Transformers in the domain of multimodal large language models (MLLMs). Furthermore, we identify the optimal configuration of the ModRWKV architecture through systematic exploration. 

**Abstract (ZH)**: 现代RNN架构在 multimodal 大语言模型中的能力探索：基于 RWKV7 的 ModRWKV 架构研究 

---
# Enhanced Multimodal Aspect-Based Sentiment Analysis by LLM-Generated Rationales 

**Title (ZH)**: 增强多模态方面情感分析的LLM生成理由方法 

**Authors**: Jun Cao, Jiyi Li, Ziwei Yang, Renjie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.14499)  

**Abstract**: There has been growing interest in Multimodal Aspect-Based Sentiment Analysis (MABSA) in recent years. Existing methods predominantly rely on pre-trained small language models (SLMs) to collect information related to aspects and sentiments from both image and text, with an aim to align these two modalities. However, small SLMs possess limited capacity and knowledge, often resulting in inaccurate identification of meaning, aspects, sentiments, and their interconnections in textual and visual data. On the other hand, Large language models (LLMs) have shown exceptional capabilities in various tasks by effectively exploring fine-grained information in multimodal data. However, some studies indicate that LLMs still fall short compared to fine-tuned small models in the field of ABSA. Based on these findings, we propose a novel framework, termed LRSA, which combines the decision-making capabilities of SLMs with additional information provided by LLMs for MABSA. Specifically, we inject explanations generated by LLMs as rationales into SLMs and employ a dual cross-attention mechanism for enhancing feature interaction and fusion, thereby augmenting the SLMs' ability to identify aspects and sentiments. We evaluated our method using two baseline models, numerous experiments highlight the superiority of our approach on three widely-used benchmarks, indicating its generalizability and applicability to most pre-trained models for MABSA. 

**Abstract (ZH)**: 多模态方面基于情感分析（MABSA）中的LRSA新型框架 

---
# Attributional Safety Failures in Large Language Models under Code-Mixed Perturbations 

**Title (ZH)**: 大规模语言模型在代码混杂扰动下的归因安全性故障 

**Authors**: Somnath Banerjee, Pratyush Chatterjee, Shanu Kumar, Sayan Layek, Parag Agrawal, Rima Hazra, Animesh Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2505.14469)  

**Abstract**: Recent advancements in LLMs have raised significant safety concerns, particularly when dealing with code-mixed inputs and outputs. Our study systematically investigates the increased susceptibility of LLMs to produce unsafe outputs from code-mixed prompts compared to monolingual English prompts. Utilizing explainability methods, we dissect the internal attribution shifts causing model's harmful behaviors. In addition, we explore cultural dimensions by distinguishing between universally unsafe and culturally-specific unsafe queries. This paper presents novel experimental insights, clarifying the mechanisms driving this phenomenon. 

**Abstract (ZH)**: Recent advancements in LLMs have raised significant safety concerns, especially when handling code-mixed inputs and outputs. Our study systematically investigates the heightened vulnerability of LLMs to generate unsafe outputs from code-mixed prompts compared to monolingual English prompts. Using explainability methods, we analyze the internal attribution shifts leading to the model's harmful behaviors. Additionally, we explore cultural dimensions by differentiating between universally unsafe and culturally-specific unsafe queries. This paper presents novel experimental insights, clarifying the mechanisms driving this phenomenon. 

---
# CtrlDiff: Boosting Large Diffusion Language Models with Dynamic Block Prediction and Controllable Generation 

**Title (ZH)**: CtrlDiff: 通过动态块预测和可控生成增强大型扩散语言模型 

**Authors**: Chihan Huang, Hao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14455)  

**Abstract**: Although autoregressive models have dominated language modeling in recent years, there has been a growing interest in exploring alternative paradigms to the conventional next-token prediction framework. Diffusion-based language models have emerged as a compelling alternative due to their powerful parallel generation capabilities and inherent editability. However, these models are often constrained by fixed-length generation. A promising direction is to combine the strengths of both paradigms, segmenting sequences into blocks, modeling autoregressive dependencies across blocks while leveraging discrete diffusion to estimate the conditional distribution within each block given the preceding context. Nevertheless, their practical application is often hindered by two key limitations: rigid fixed-length outputs and a lack of flexible control mechanisms. In this work, we address the critical limitations of fixed granularity and weak controllability in current large diffusion language models. We propose CtrlDiff, a dynamic and controllable semi-autoregressive framework that adaptively determines the size of each generation block based on local semantics using reinforcement learning. Furthermore, we introduce a classifier-guided control mechanism tailored to discrete diffusion, which significantly reduces computational overhead while facilitating efficient post-hoc conditioning without retraining. Extensive experiments demonstrate that CtrlDiff sets a new standard among hybrid diffusion models, narrows the performance gap to state-of-the-art autoregressive approaches, and enables effective conditional text generation across diverse tasks. 

**Abstract (ZH)**: 尽管自回归模型在近年来的语言建模中占主导地位，但人们对探索传统的下一个标记预测框架之外的替代范式越来越感兴趣。基于扩散的语言模型因其强大的并行生成能力和固有的可编辑性而成为一种有吸引力的替代方案。然而，这些模型通常受到固定长度生成的限制。一种有前景的方向是结合这两种范式的优点，将序列分割成块，在块之间建模自回归依赖关系，同时利用离散扩散估计给定前文语境下每个块内的条件分布。然而，它们的实际应用常常受到两个关键限制的阻碍：刚性固定长度输出和缺乏灵活的控制机制。在本文中，我们解决了当前大型扩散语言模型中固定粒度和弱可控性的关键限制。我们提出了一种名为CtrlDiff的动态可控半自回归框架，利用强化学习根据局部语义自适应地确定每个生成块的大小。此外，我们引入了一种针对离散扩散的分类器引导控制机制，显著减少了计算开销，同时促进了高效的后验调整而无需重新训练。广泛的实验表明，CtrlDiff在混合扩散模型中确立了新的标准，缩小了与最先进的自回归方法之间的性能差距，并支持在各种任务中有效生成条件文本。 

---
# How Managers Perceive AI-Assisted Conversational Training for Workplace Communication 

**Title (ZH)**: 管理者对AI辅助对话式培训在工作场所沟通中的感知 

**Authors**: Lance T Wilhelm, Xiaohan Ding, Kirk McInnis Knutsen, Buse Carik, Eugenia H Rho  

**Link**: [PDF](https://arxiv.org/pdf/2505.14452)  

**Abstract**: Effective workplace communication is essential for managerial success, yet many managers lack access to tailored and sustained training. Although AI-assisted communication systems may offer scalable training solutions, little is known about how managers envision the role of AI in helping them improve their communication skills. To investigate this, we designed a conversational role-play system, CommCoach, as a functional probe to understand how managers anticipate using AI to practice their communication skills. Through semi-structured interviews, participants emphasized the value of adaptive, low-risk simulations for practicing difficult workplace conversations. They also highlighted opportunities, including human-AI teaming, transparent and context-aware feedback, and greater control over AI-generated personas. AI-assisted communication training should balance personalization, structured learning objectives, and adaptability to different user styles and contexts. However, achieving this requires carefully navigating tensions between adaptive and consistent AI feedback, realism and potential bias, and the open-ended nature of AI conversations versus structured workplace discourse. 

**Abstract (ZH)**: 有效的职场沟通对于管理成功至关重要，但许多管理者缺乏量身定制且持续的培训机会。尽管基于AI的沟通系统可能提供可扩展的培训解决方案，但仍不清楚管理者如何设想AI在帮助他们提升沟通技能方面的作用。为此，我们设计了一个对话式角色扮演系统CommCoach，作为一种功能性的探针，以了解管理者如何看待使用AI练习沟通技能的方式。通过半结构化的访谈，参与者强调了适应性、低风险模拟对练习困难职场对话的价值，并指出了包括人机合作、透明且情境感知的反馈以及对AI生成人物的更大控制在内的机遇。基于AI的沟通培训应平衡个性化、结构化的学习目标以及对不同类型用户风格和情境的适应性。然而，实现这一点需要仔细权衡适应性与一致的AI反馈、真实性与潜在偏见之间的矛盾，以及非结构化的人工智能对话与结构化的职场交流之间的差异。 

---
# RefiDiff: Refinement-Aware Diffusion for Efficient Missing Data Imputation 

**Title (ZH)**: RefiDiff: 重视细化的扩散方法用于高效的缺失数据插补 

**Authors**: Md Atik Ahamed, Qiang Ye, Qiang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.14451)  

**Abstract**: Missing values in high-dimensional, mixed-type datasets pose significant challenges for data imputation, particularly under Missing Not At Random (MNAR) mechanisms. Existing methods struggle to integrate local and global data characteristics, limiting performance in MNAR and high-dimensional settings. We propose an innovative framework, RefiDiff, combining local machine learning predictions with a novel Mamba-based denoising network capturing interrelationships among distant features and samples. Our approach leverages pre-refinement for initial warm-up imputations and post-refinement to polish results, enhancing stability and accuracy. By encoding mixed-type data into unified tokens, RefiDiff enables robust imputation without architectural or hyperparameter tuning. RefiDiff outperforms state-of-the-art (SOTA) methods across missing-value settings, excelling in MNAR with a 4x faster training time than SOTA DDPM-based approaches. Extensive evaluations on nine real-world datasets demonstrate its robustness, scalability, and effectiveness in handling complex missingness patterns. 

**Abstract (ZH)**: 高维混合型数据集中缺失值的存在在数据插补中提出了重大挑战，尤其是在非随机缺失（MNAR）机制下。现有方法难以整合局部和全局数据特性，限制了其在MNAR和高维设置下的性能。我们提出了一种创新框架RefiDiff，结合了局部机器学习预测和一个基于新型Mamba的去噪网络，该网络捕获了远程特征和样本之间的相互关系。该方法利用预插补进行初始预热插补，并在后续进行润色以增强稳定性和准确性。通过将混合型数据编码为统一的tokens，RefiDiff能够在无需调整架构或超参数的情况下实现稳健的插补。RefiDiff在各种缺失值设置中优于现有最佳方法（SOTA），在MNAR情况下具有4倍于SOTA DDPM基方法的训练速度。在九个真实世界数据集上的广泛评估证明了其稳健性、可扩展性和在处理复杂缺失模式方面的有效性。 

---
# Creative Preference Optimization 

**Title (ZH)**: 创意偏好优化 

**Authors**: Mete Ismayilzada, Antonio Laverghetta Jr., Simone A. Luchini, Reet Patel, Antoine Bosselut, Lonneke van der Plas, Roger Beaty  

**Link**: [PDF](https://arxiv.org/pdf/2505.14442)  

**Abstract**: While Large Language Models (LLMs) have demonstrated impressive performance across natural language generation tasks, their ability to generate truly creative content-characterized by novelty, diversity, surprise, and quality-remains limited. Existing methods for enhancing LLM creativity often focus narrowly on diversity or specific tasks, failing to address creativity's multifaceted nature in a generalizable way. In this work, we propose Creative Preference Optimization (CrPO), a novel alignment method that injects signals from multiple creativity dimensions into the preference optimization objective in a modular fashion. We train and evaluate creativity-augmented versions of several models using CrPO and MuCE, a new large-scale human preference dataset spanning over 200,000 human-generated responses and ratings from more than 30 psychological creativity assessments. Our models outperform strong baselines, including GPT-4o, on both automated and human evaluations, producing more novel, diverse, and surprising generations while maintaining high output quality. Additional evaluations on NoveltyBench further confirm the generalizability of our approach. Together, our results demonstrate that directly optimizing for creativity within preference frameworks is a promising direction for advancing the creative capabilities of LLMs without compromising output quality. 

**Abstract (ZH)**: 虽然大型语言模型在自然语言生成任务中展现了令人印象深刻的性能，但它们生成真正具有创造力的内容（表现为新颖性、多样性、惊喜感和高质量）的能力仍然有限。现有增强大型语言模型创造力的方法往往集中于单一维度或特定任务，未能以可推广的方式全面解决创造力的多方面特性。在本项工作中，我们提出了一种名为Creative Preference Optimization (CrPO) 的新颖对齐方法，以模块化方式将多个创造力维度的信号注入到偏好优化目标中。我们使用CrPO和一个新的大规模人类偏好数据集MuCE（涵盖超过20万个人类生成的响应和来自30多种心理创造力评估的评级）训练和评估了增强创造力的多个模型。我们的模型在自动评估和人类评估中均优于强大的基准模型（包括GPT-4o），产生了更多新颖、多样且充满惊喜的生成内容，同时保持了高质量的输出。进一步在NoveltyBench上的评估也证实了我们方法的可推广性。综上所述，我们的结果表明，在偏好框架中直接优化创造力是提升大型语言模型创造力能力的一个有前景的方向，而不会牺牲输出质量。 

---
# Neural Incompatibility: The Unbridgeable Gap of Cross-Scale Parametric Knowledge Transfer in Large Language Models 

**Title (ZH)**: 神经不兼容性：大规模语言模型中跨尺度参数知识转移的不可逾越差距 

**Authors**: Yuqiao Tan, Shizhu He, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.14436)  

**Abstract**: Large Language Models (LLMs) offer a transparent brain with accessible parameters that encode extensive knowledge, which can be analyzed, located and transferred. Consequently, a key research challenge is to transcend traditional knowledge transfer paradigms rooted in symbolic language and achieve genuine Parametric Knowledge Transfer (PKT). Significantly, exploring effective methods for transferring knowledge across LLMs of different scales through parameters presents an intriguing and valuable research direction. In this paper, we first demonstrate $\textbf{Alignment}$ in parametric space is the fundamental prerequisite to achieve successful cross-scale PKT. We redefine the previously explored knowledge transfer as Post-Align PKT (PostPKT), which utilizes extracted parameters for LoRA initialization and requires subsequent fine-tune for alignment. Hence, to reduce cost for further fine-tuning, we introduce a novel Pre-Align PKT (PrePKT) paradigm and propose a solution called $\textbf{LaTen}$ ($\textbf{L}$oc$\textbf{a}$te-$\textbf{T}$h$\textbf{e}$n-Alig$\textbf{n}$) that aligns the parametric spaces of LLMs across scales only using several training steps without following training. Comprehensive experiments on four benchmarks demonstrate that both PostPKT and PrePKT face challenges in achieving consistently stable transfer. Through in-depth analysis, we identify $\textbf{Neural Incompatibility}$ as the ethological and parametric structural differences between LLMs of varying scales, presenting fundamental challenges to achieving effective PKT. These findings provide fresh insights into the parametric architectures of LLMs and highlight promising directions for future research on efficient PKT. Our code is available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）提供了一个透明的“大脑”，其中可访问的参数编码了丰富的知识，可以被分析、定位和转移。因此，一个关键的研究挑战是如何超越传统的基于符号语言的知识转移范式，实现真正的参数知识转移（PKT）。值得一提的是，探索有效的方法以参数形式在不同规模的LLMs之间进行知识转移是一个极具吸引力和价值的研究方向。在本文中，我们首先证明参数空间的对齐是实现成功的大规模知识转移的基本前提。我们将以往探索的知识转移重新定义为后对齐知识转移（PostPKT），它利用提取的参数进行LoRA初始化，并需要后续的微调以实现对齐。为了减少后续微调的成本，我们引入了一个新的前对齐知识转移（PrePKT）范式，并提出了一种名为LaTen（Locate-Then-Align）的解决方案，该方法仅通过几轮训练步骤对不同规模的大规模语言模型的参数空间进行对齐，而不进行后续训练。在四个基准上的全面实验表明，无论是PostPKT还是PrePKT都面临着一致稳定地转移的挑战。通过深入分析，我们确定了神经不兼容性为不同规模的大规模语言模型之间在神经和参数结构上的差异，这些差异构成了有效PKT的基本挑战。这些发现为大规模语言模型的参数架构提供了新的见解，并突显了未来高效PKT研究的有希望的方向。我们的代码可在以下链接访问：this https URL。 

---
# Choosing a Model, Shaping a Future: Comparing LLM Perspectives on Sustainability and its Relationship with AI 

**Title (ZH)**: 选择模型，塑造未来：比较大型语言模型对可持续性的看法及其与AI的关系 

**Authors**: Annika Bush, Meltem Aksoy, Markus Pauly, Greta Ontrup  

**Link**: [PDF](https://arxiv.org/pdf/2505.14435)  

**Abstract**: As organizations increasingly rely on AI systems for decision support in sustainability contexts, it becomes critical to understand the inherent biases and perspectives embedded in Large Language Models (LLMs). This study systematically investigates how five state-of-the-art LLMs -- Claude, DeepSeek, GPT, LLaMA, and Mistral - conceptualize sustainability and its relationship with AI. We administered validated, psychometric sustainability-related questionnaires - each 100 times per model -- to capture response patterns and variability. Our findings revealed significant inter-model differences: For example, GPT exhibited skepticism about the compatibility of AI and sustainability, whereas LLaMA demonstrated extreme techno-optimism with perfect scores for several Sustainable Development Goals (SDGs). Models also diverged in attributing institutional responsibility for AI and sustainability integration, a results that holds implications for technology governance approaches. Our results demonstrate that model selection could substantially influence organizational sustainability strategies, highlighting the need for awareness of model-specific biases when deploying LLMs for sustainability-related decision-making. 

**Abstract (ZH)**: 随着组织越来越多地依赖AI系统在可持续发展领域提供决策支持，理解大型语言模型（LLMs）中固有的偏见和视角变得至关重要。本研究系统性地探讨了五种最先进的LLMs——Claude、DeepSeek、GPT、LLaMA和Mistral——如何概念化可持续性及其与AI的关系。我们对每种模型进行了100次可持续性相关的验证psychometric问卷测试，以捕捉响应模式和变异。研究发现各模型之间存在显著差异：例如，GPT对AI与可持续性的兼容性表示怀疑，而LLaMA则表现出极端的技术乐观主义，并在多项可持续发展目标（SDGs）上获得了满分。模型在归因于机构在AI与可持续性整合中的责任方面也存在差异，这一结果对技术治理方法具有重要意义。研究结果表明，模型选择可能会显著影响组织的可持续性策略，强调在使用LLMs进行与可持续性相关的决策时需要意识到模型特定的偏见。 

---
# Interpretable Neural System Dynamics: Combining Deep Learning with System Dynamics Modeling to Support Critical Applications 

**Title (ZH)**: 可解释的神经系统动力学：将深层学习与系统动力学建模相结合以支持关键应用 

**Authors**: Riccardo D'Elia  

**Link**: [PDF](https://arxiv.org/pdf/2505.14428)  

**Abstract**: The objective of this proposal is to bridge the gap between Deep Learning (DL) and System Dynamics (SD) by developing an interpretable neural system dynamics framework. While DL excels at learning complex models and making accurate predictions, it lacks interpretability and causal reliability. Traditional SD approaches, on the other hand, provide transparency and causal insights but are limited in scalability and require extensive domain knowledge. To overcome these limitations, this project introduces a Neural System Dynamics pipeline, integrating Concept-Based Interpretability, Mechanistic Interpretability, and Causal Machine Learning. This framework combines the predictive power of DL with the interpretability of traditional SD models, resulting in both causal reliability and scalability. The efficacy of the proposed pipeline will be validated through real-world applications of the EU-funded AutoMoTIF project, which is focused on autonomous multimodal transportation systems. The long-term goal is to collect actionable insights that support the integration of explainability and safety in autonomous systems. 

**Abstract (ZH)**: 本提案的目的是通过建立可解释的神经系统动力学框架，弥合深度学习和系统动力学之间的差距。虽然深度学习在学习复杂模型和进行准确预测方面表现出色，但在可解释性和因果可靠性方面存在不足。传统的系统动力学方法提供了透明度和因果洞察，但在可扩展性方面有所限制，并且需要广泛的专业领域知识。为克服这些局限性，本项目引入了一种神经系统动力学管道，整合了基于概念的可解释性、机理可解释性和因果机器学习。该框架结合了深度学习的预测能力和传统系统动力学模型的可解释性，实现了因果可靠性和可扩展性。通过欧盟资助的AutoMoTIF项目的真实世界应用，验证所提出管道的有效性，该研究专注于自主多模态交通系统。长期目标是收集可操作的见解，以支持自主系统中解释性和安全性的整合。 

---
# Log-Augmented Generation: Scaling Test-Time Reasoning with Reusable Computation 

**Title (ZH)**: 日志增强生成：通过可复用计算扩展测试时推理 

**Authors**: Peter Baile Chen, Yi Zhang, Dan Roth, Samuel Madden, Jacob Andreas, Michael Cafarella  

**Link**: [PDF](https://arxiv.org/pdf/2505.14398)  

**Abstract**: While humans naturally learn and adapt from past experiences, large language models (LLMs) and their agentic counterparts struggle to retain reasoning from previous tasks and apply them in future contexts. To address this limitation, we propose a novel framework, log-augmented generation (LAG) that directly reuses prior computation and reasoning from past logs at test time to enhance model's ability to learn from previous tasks and perform better on new, unseen challenges, all while keeping the system efficient and scalable. Specifically, our system represents task logs using key-value (KV) caches, encoding the full reasoning context of prior tasks while storing KV caches for only a selected subset of tokens. When a new task arises, LAG retrieves the KV values from relevant logs to augment generation. Our approach differs from reflection-based memory mechanisms by directly reusing prior reasoning and computations without requiring additional steps for knowledge extraction or distillation. Our method also goes beyond existing KV caching techniques, which primarily target efficiency gains rather than improving accuracy. Experiments on knowledge- and reasoning-intensive datasets demonstrate that our method significantly outperforms standard agentic systems that do not utilize logs, as well as existing solutions based on reflection and KV cache techniques. 

**Abstract (ZH)**: 虽然人类自然地从过往经验中学习和适应，但大型语言模型（LLMs）及其代理版本在保留先前任务的推理并将其应用于未来情境方面存在困难。为了解决这一局限性，我们提出了一个新颖的框架——日志增强生成（LAG），该框架在测试时直接重用过去的日志中的先前计算和推理，以增强模型从先前任务中学习的能力并在新的、未见过的挑战中表现得更好，同时保持系统的高效和可扩展性。具体而言，我们的系统使用键值（KV）缓存来表示任务日志，编码先前任务的完整推理背景，并仅存储选定token的KV缓存。当新任务出现时，LAG从相关日志中检索KV值以增强生成。与基于反思的记忆机制不同，我们的方法直接重用先前的推理和计算，而不需要额外的知识提取或蒸馏步骤。此外，我们的方法超越了现有的KV缓存技术，后者主要侧重于效率提升而不是提高准确性。在知识和推理密集型数据集上的实验表明，我们的方法显著优于不使用日志的标准代理系统以及基于反思和KV缓存技术的现有解决方案。 

---
# MUG-Eval: A Proxy Evaluation Framework for Multilingual Generation Capabilities in Any Language 

**Title (ZH)**: MUG-Eval：任意语言多语种生成能力的代理评价框架 

**Authors**: Seyoung Song, Seogyeong Jeong, Eunsu Kim, Jiho Jin, Dongkwan Kim, Jay Shin, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2505.14395)  

**Abstract**: Evaluating text generation capabilities of large language models (LLMs) is challenging, particularly for low-resource languages where methods for direct assessment are scarce. We propose MUG-Eval, a novel framework that evaluates LLMs' multilingual generation capabilities by transforming existing benchmarks into conversational tasks and measuring the LLMs' accuracies on those tasks. We specifically designed these conversational tasks to require effective communication in the target language. Then, we simply use task success rate as a proxy of successful conversation generation. Our approach offers two key advantages: it is independent of language-specific NLP tools or annotated datasets, which are limited for most languages, and it does not rely on LLMs-as-judges, whose evaluation quality degrades outside a few high-resource languages. We evaluate 8 LLMs across 30 languages spanning high, mid, and low-resource categories, and we find that MUG-Eval correlates strongly with established benchmarks ($r$ > 0.75) while enabling standardized comparisons across languages and models. Our framework provides a robust and resource-efficient solution for evaluating multilingual generation that can be extended to thousands of languages. 

**Abstract (ZH)**: 评估大型语言模型的文本生成能力具有挑战性，特别是在资源稀缺的语言中，直接评估方法稀缺。我们提出了MUG-Eval，这是一种新型框架，通过将现有基准转换为对话任务，并衡量模型在这些任务上的准确率来评估大型语言模型的多语言生成能力。我们特别设计了这些对话任务，要求在目标语言中实现有效的沟通。然后，我们简单地将任务成功率作为成功对话生成的代理。该方法具有两个关键优势：它不依赖于特定语言的NLP工具或注释数据集，这些工具和数据集对于大多数语言而言是有限的；此外，它也不依赖于将大型语言模型作为评估者，这些模型的评估质量在资源丰富语言以外会下降。我们对30种不同资源类别的8种大型语言模型进行了评估，并发现MUG-Eval与现有基准具有强烈的相关性（相关系数>r>0.75），同时为不同语言和模型之间提供了标准化的比较。我们的框架提供了一个稳健且资源高效的多语言生成评估解决方案，可以扩展到数千种语言。 

---
# When Bias Backfires: The Modulatory Role of Counterfactual Explanations on the Adoption of Algorithmic Bias in XAI-Supported Human Decision-Making 

**Title (ZH)**: 当偏差适得其反：反事实解释在XAI支持的人类决策中对算法偏差采用的调节作用 

**Authors**: Ulrike Kuhl, Annika Bush  

**Link**: [PDF](https://arxiv.org/pdf/2505.14377)  

**Abstract**: Although the integration of artificial intelligence (AI) into everyday tasks improves efficiency and objectivity, it also risks transmitting bias to human decision-making. In this study, we conducted a controlled experiment that simulated hiring decisions to examine how biased AI recommendations - augmented with or without counterfactual explanations - influence human judgment over time. Participants, acting as hiring managers, completed 60 decision trials divided into a baseline phase without AI, followed by a phase with biased (X)AI recommendations (favoring either male or female candidates), and a final post-interaction phase without AI. Our results indicate that the participants followed the AI recommendations 70% of the time when the qualifications of the given candidates were comparable. Yet, only a fraction of participants detected the gender bias (8 out of 294). Crucially, exposure to biased AI altered participants' inherent preferences: in the post-interaction phase, participants' independent decisions aligned with the bias when no counterfactual explanations were provided before, but reversed the bias when explanations were given. Reported trust did not differ significantly across conditions. Confidence varied throughout the study phases after exposure to male-biased AI, indicating nuanced effects of AI bias on decision certainty. Our findings point to the importance of calibrating XAI to avoid unintended behavioral shifts in order to safeguard equitable decision-making and prevent the adoption of algorithmic bias. 

**Abstract (ZH)**: 尽管将人工智能（AI）集成到日常任务中可以提高效率和客观性，但也存在将偏见传递给人类决策的风险。在此研究中，我们进行了一个受控实验，模拟招聘决策过程，以探究带有或不带有反事实解释的有偏见的AI建议如何随时间影响人类判断。参与者作为招聘经理，完成了60次决策试验，分为没有AI的基线阶段，有偏见（X）AI建议（偏向男性或女性候选人）的阶段，以及最后的无AI后交互阶段。结果显示，当候选人资质相似时，参与者有70%的时间采纳了AI建议。然而，只有少数参与者（8/294）检测到性别偏见。关键的是，接触有偏见的AI改变了参与者的固有偏好：在后交互阶段，未提供反事实解释时，参与者的独立决策与偏见一致，提供了解释时则逆转了偏见。报告的可信度在各条件下没有显著差异。在接触偏向男性候选人的AI后，信心程度在整个研究阶段有所不同，表明AI偏见对决策确定性的影响是复杂的。我们的研究表明，为了确保公平决策并防止算法偏见的采纳，需要调整XAI以避免无意中的行为变化。 

---
# FMSD-TTS: Few-shot Multi-Speaker Multi-Dialect Text-to-Speech Synthesis for Ü-Tsang, Amdo and Kham Speech Dataset Generation 

**Title (ZH)**: FMSD-TTS: 少量样本多说话人多方言文本到语音合成及其用于Ü--tsang、Amdo和Kham语音数据集生成 

**Authors**: Yutong Liu, Ziyue Zhang, Ban Ma-bao, Yuqing Cai, Yongbin Yu, Renzeng Duojie, Xiangxiang Wang, Fan Gao, Cheng Huang, Nyima Tashi  

**Link**: [PDF](https://arxiv.org/pdf/2505.14351)  

**Abstract**: Tibetan is a low-resource language with minimal parallel speech corpora spanning its three major dialects-Ü-Tsang, Amdo, and Kham-limiting progress in speech modeling. To address this issue, we propose FMSD-TTS, a few-shot, multi-speaker, multi-dialect text-to-speech framework that synthesizes parallel dialectal speech from limited reference audio and explicit dialect labels. Our method features a novel speaker-dialect fusion module and a Dialect-Specialized Dynamic Routing Network (DSDR-Net) to capture fine-grained acoustic and linguistic variations across dialects while preserving speaker identity. Extensive objective and subjective evaluations demonstrate that FMSD-TTS significantly outperforms baselines in both dialectal expressiveness and speaker similarity. We further validate the quality and utility of the synthesized speech through a challenging speech-to-speech dialect conversion task. Our contributions include: (1) a novel few-shot TTS system tailored for Tibetan multi-dialect speech synthesis, (2) the public release of a large-scale synthetic Tibetan speech corpus generated by FMSD-TTS, and (3) an open-source evaluation toolkit for standardized assessment of speaker similarity, dialect consistency, and audio quality. 

**Abstract (ZH)**: 藏语是一种资源稀少的语言，其三种主要方言—— Ü-Tsang、Amdo 和 Kham 的平行语音语料库极少，限制了语音建模的进步。为解决这一问题，我们提出了一种少量样本、多说话人、多方言的文本到语音框架 FMSD-TTS，该框架可以从有限的参考音频和显式的方言标签中合成平行的方言语音。该方法包括一种新颖的说话人-方言融合模块和一种方言专业化动态路由网络 (DSDR-Net)，以捕捉不同方言之间的细微的声学和语言变异，同时保留说话人身份。广泛的客观和主观评估表明，FMSD-TTS 在方言表达能力和说话人相似度方面显著优于基线方法。我们还通过一项具有挑战性的语音到语音方言转换任务进一步验证了合成语音的质量和实用性。我们的贡献包括：(1) 一种针对藏语多方言语音合成的新型少量样本 TTS 系统，(2) 由 FMSD-TTS 生成的大量合成藏语语音语料库的公开发布，以及 (3) 一种开源评估工具包，用于标准化评估说话人相似度、方言一致性和音频质量。 

---
# Upgrading Democracies with Fairer Voting Methods 

**Title (ZH)**: 提升民主质量的公平投票方法 

**Authors**: Evangelos Pournaras, Srijoni Majumdar, Thomas Wellings, Joshua C. Yang, Fatemeh B. Heravan, Regula Hänggli Fricker, Dirk Helbing  

**Link**: [PDF](https://arxiv.org/pdf/2505.14349)  

**Abstract**: Voting methods are instrumental design element of democracies. Citizens use them to express and aggregate their preferences to reach a collective decision. However, voting outcomes can be as sensitive to voting rules as they are to people's voting choices. Despite the significance and inter-disciplinary scientific progress on voting methods, several democracies keep relying on outdated voting methods that do not fit modern, pluralistic societies well, while lacking social innovation. Here, we demonstrate how one can upgrade real-world democracies, namely by using alternative preferential voting methods such as cumulative voting and the method of equal shares designed for a proportional representation of voters' preferences. By rigorously assessing a new participatory budgeting approach applied in the city of Aarau, Switzerland, we unravel the striking voting outcomes of fair voting methods: more winning projects with the same budget and broader geographic and preference representation of citizens by the elected projects, in particular for voters who used to be under-represented, while promoting novel project ideas. We provide profound causal evidence showing that citizens prefer proportional voting methods, which possess strong legitimacy without the need of very technical specialized explanations. We also reveal strong underlying democratic values exhibited by citizens who support fair voting methods such as altruism and compromise. These findings come with a global momentum to unleash a new and long-awaited participation blueprint of how to upgrade democracies. 

**Abstract (ZH)**: 投票方法是民主制度的关键设计元素。公民利用它们来表达和聚合偏好以达成集体决策。然而，投票结果可能会因投票规则和选民投票选择而敏感。尽管投票方法的重要性及其多学科科学进展显著，仍有几个民主国家继续依赖不适用于现代多元社会的过时投票方法，缺乏社会创新。在这里，我们展示了如何通过使用累积投票和等额份额方法等替代偏好投票方法来提升现实世界的民主制度，从而实现更公平的投票结果：在相同的预算下更多项目获胜，并且当选项目更广泛地代表了公民的地理和偏好分布，特别是对于以往被忽视的选民群体，同时促进新颖项目理念。我们提供了严谨的因果证据表明，公民偏好公平的投票方法，这些方法具有强大的合法性，无需复杂的专门解释。我们还揭示了支持公正投票方法的公民所体现的强大民主价值观，如利他主义和妥协。这些发现推动了全球范围内的参与式蓝图，展示了如何升级民主制度。 

---
# Enhancing Classification with Semi-Supervised Deep Learning Using Distance-Based Sample Weights 

**Title (ZH)**: 基于距离权重的半监督深度学习分类增强 

**Authors**: Aydin Abedinia, Shima Tabakhi, Vahid Seydi  

**Link**: [PDF](https://arxiv.org/pdf/2505.14345)  

**Abstract**: Recent advancements in semi-supervised deep learning have introduced effective strategies for leveraging both labeled and unlabeled data to improve classification performance. This work proposes a semi-supervised framework that utilizes a distance-based weighting mechanism to prioritize critical training samples based on their proximity to test data. By focusing on the most informative examples, the method enhances model generalization and robustness, particularly in challenging scenarios with noisy or imbalanced datasets. Building on techniques such as uncertainty consistency and graph-based representations, the approach addresses key challenges of limited labeled data while maintaining scalability. Experiments on twelve benchmark datasets demonstrate significant improvements across key metrics, including accuracy, precision, and recall, consistently outperforming existing methods. This framework provides a robust and practical solution for semi-supervised learning, with potential applications in domains such as healthcare and security where data limitations pose significant challenges. 

**Abstract (ZH)**: 最近在半监督深度学习领域的进展引入了有效策略，以利用标记和未标记数据来提高分类性能。本项工作提出了一种基于距离加权的半监督框架，该框架根据样本与测试数据的接近程度优先处理关键训练样本，通过关注最具信息量的例证，该方法增强了模型的泛化能力和鲁棒性，特别是在噪声或不平衡数据集的挑战性场景中。该方法借助不确定性一致性和图表示技术，解决了有限标记数据的关键挑战，同时保持了可扩展性。实验表明，在包括准确率、精确率和召回率在内的关键指标上，该方法显著优于现有方法。该框架为半监督学习提供了稳健且实用的解决方案，具有在数据限制构成重大挑战的医疗保健和安全等领域潜在应用。 

---
# Replace in Translation: Boost Concept Alignment in Counterfactual Text-to-Image 

**Title (ZH)**: 替换翻译：提升反事实文本到图像的概念对齐 

**Authors**: Sifan Li, Ming Tao, Hao Zhao, Ling Shao, Hao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14341)  

**Abstract**: Text-to-Image (T2I) has been prevalent in recent years, with most common condition tasks having been optimized nicely. Besides, counterfactual Text-to-Image is obstructing us from a more versatile AIGC experience. For those scenes that are impossible to happen in real world and anti-physics, we should spare no efforts in increasing the factual feel, which means synthesizing images that people think very likely to be happening, and concept alignment, which means all the required objects should be in the same frame. In this paper, we focus on concept alignment. As controllable T2I models have achieved satisfactory performance for real applications, we utilize this technology to replace the objects in a synthesized image in latent space step-by-step to change the image from a common scene to a counterfactual scene to meet the prompt. We propose a strategy to instruct this replacing process, which is called as Explicit Logical Narrative Prompt (ELNP), by using the newly SoTA language model DeepSeek to generate the instructions. Furthermore, to evaluate models' performance in counterfactual T2I, we design a metric to calculate how many required concepts in the prompt can be covered averagely in the synthesized images. The extensive experiments and qualitative comparisons demonstrate that our strategy can boost the concept alignment in counterfactual T2I. 

**Abstract (ZH)**: 文本到图像（T2I）近年来十分盛行，大多数常见条件任务已得到很好地优化。然而，反事实文本到图像限制了我们获得更丰富多样的AIGC体验。对于现实中不可能发生的场景和反物理场景，我们应该竭尽全力增加其实感感，即合成人们认为很可能会发生的图像，以及概念对齐，即所有所需的对象应在同一帧中。本文侧重于概念对齐。鉴于可控文本到图像模型在现实应用中已取得满意性能，我们利用此技术在潜在空间中逐步替换合成图像中的对象，将图像从常见场景转变为反事实场景以满足提示要求。我们提出了一种策略来指导这一替换过程，称为显式逻辑叙述提示（ELNP），并通过最新的最先进语言模型DeepSeek生成指令。此外，为了评估模型在反事实T2I中的性能，我们设计了一个度量标准来计算合成图像中平均能覆盖提示中所需概念的数量。广泛实验和定性比较表明，我们的策略能够提升反事实T2I中的概念对齐效果。 

---
# Handloom Design Generation Using Generative Networks 

**Title (ZH)**: 使用生成网络的手织设计生成 

**Authors**: Rajat Kanti Bhattacharjee, Meghali Nandi, Amrit Jha, Gunajit Kalita, Ferdous Ahmed Barbhuiya  

**Link**: [PDF](https://arxiv.org/pdf/2505.14330)  

**Abstract**: This paper proposes deep learning techniques of generating designs for clothing, focused on handloom fabric and discusses the associated challenges along with its application. The capability of generative neural network models in understanding artistic designs and synthesizing those is not yet explored well. In this work, multiple methods are employed incorporating the current state of the art generative models and style transfer algorithms to study and observe their performance for the task. The results are then evaluated through user score. This work also provides a new dataset NeuralLoom for the task of the design generation. 

**Abstract (ZH)**: 本文提出深度学习方法生成手织布服装设计，并讨论相关挑战及其应用。生成神经网络模型在理解艺术设计并合成设计方面的潜力尚未得到充分探索。本工作中，采用了当前最先进的生成模型和风格迁移算法等多种方法来研究和观察其在该任务中的表现。通过用户评分对结果进行评估。此外，本文还提供了一个新的数据集NeuralLoom以用于设计生成任务。 

---
# Exploring Jailbreak Attacks on LLMs through Intent Concealment and Diversion 

**Title (ZH)**: 通过意图隐藏和转移探索对大语言模型的 Jailbreak 攻击 

**Authors**: Tiehan Cui, Yanxu Mao, Peipei Liu, Congying Liu, Datao You  

**Link**: [PDF](https://arxiv.org/pdf/2505.14316)  

**Abstract**: Although large language models (LLMs) have achieved remarkable advancements, their security remains a pressing concern. One major threat is jailbreak attacks, where adversarial prompts bypass model safeguards to generate harmful or objectionable content. Researchers study jailbreak attacks to understand security and robustness of LLMs. However, existing jailbreak attack methods face two main challenges: (1) an excessive number of iterative queries, and (2) poor generalization across models. In addition, recent jailbreak evaluation datasets focus primarily on question-answering scenarios, lacking attention to text generation tasks that require accurate regeneration of toxic content. To tackle these challenges, we propose two contributions: (1) ICE, a novel black-box jailbreak method that employs Intent Concealment and divErsion to effectively circumvent security constraints. ICE achieves high attack success rates (ASR) with a single query, significantly improving efficiency and transferability across different models. (2) BiSceneEval, a comprehensive dataset designed for assessing LLM robustness in question-answering and text-generation tasks. Experimental results demonstrate that ICE outperforms existing jailbreak techniques, revealing critical vulnerabilities in current defense mechanisms. Our findings underscore the necessity of a hybrid security strategy that integrates predefined security mechanisms with real-time semantic decomposition to enhance the security of LLMs. 

**Abstract (ZH)**: 虽然大型语言模型（LLMs）取得了显著进展，但其安全性仍是一个亟待解决的问题。主要威胁之一是escaping攻击，这种攻击通过对抗性提示绕过模型的安全保护机制生成有害或令人反感的内容。研究人员研究escaping攻击以了解LLM的安全性和稳健性。然而，现有的escaping攻击方法面临两个主要挑战：（1）需要大量的迭代查询，（2）泛化能力较差。此外，最近的escaping评估数据集主要关注于问答场景，而忽略了需要准确再生有毒内容的文本生成任务。为应对这些挑战，我们提出了两项贡献：（1）ICE，一种新颖的黑盒escaping方法，通过意图隐匿和转移有效规避安全约束。ICE仅通过一次查询就能实现高攻击成功率（ASR），显著提高了效率和跨不同模型的迁移性。（2）BiSceneEval，一种为评估LLM在问答和文本生成任务中的稳健性而设计的综合数据集。实验结果表明，ICE优于现有escaping技术，揭示了当前防御机制中的关键漏洞。我们的发现强调了需要结合预定义的安全机制与实时语义分解的混合安全策略，以提高LLM的安全性。 

---
# MultiTab: A Comprehensive Benchmark Suite for Multi-Dimensional Evaluation in Tabular Domains 

**Title (ZH)**: 多表：表数据域多维度评估的综合性基准套件 

**Authors**: Kyungeun Lee, Moonjung Eo, Hye-Seung Cho, Dongmin Kim, Ye Seul Sim, Seoyoon Kim, Min-Kook Suh, Woohyung Lim  

**Link**: [PDF](https://arxiv.org/pdf/2505.14312)  

**Abstract**: Despite the widespread use of tabular data in real-world applications, most benchmarks rely on average-case metrics, which fail to reveal how model behavior varies across diverse data regimes. To address this, we propose MultiTab, a benchmark suite and evaluation framework for multi-dimensional, data-aware analysis of tabular learning algorithms. Rather than comparing models only in aggregate, MultiTab categorizes 196 publicly available datasets along key data characteristics, including sample size, label imbalance, and feature interaction, and evaluates 13 representative models spanning a range of inductive biases. Our analysis shows that model performance is highly sensitive to such regimes: for example, models using sample-level similarity excel on datasets with large sample sizes or high inter-feature correlation, while models encoding inter-feature dependencies perform best with weakly correlated features. These findings reveal that inductive biases do not always behave as intended, and that regime-aware evaluation is essential for understanding and improving model behavior. MultiTab enables more principled model design and offers practical guidance for selecting models tailored to specific data characteristics. All datasets, code, and optimization logs are publicly available at this https URL. 

**Abstract (ZH)**: 尽管表格数据在实际应用中广泛使用，大多数基准测试依赖于平均情况指标，这未能揭示模型在不同数据范围下的行为差异。为解决这一问题，我们提出了一种名为MultiTab的基准测试套件及评估框架，用于多维、数据导向的表格学习算法分析。MultiTab 不仅按样本大小、标签不平衡和特征交互等关键数据特征对196个公开数据集进行分类，还评估了13种代表性模型，涵盖了不同的归纳偏置。我们的分析表明，模型性能对这些范围高度敏感：例如，基于样本级相似性的模型在大样本大小或高特征间关联的数据集上表现优异，而编码特征间依赖性的模型在弱相关特征的数据集上表现最佳。这些发现揭示了归纳偏置并不总是按预期表现，因此针对特定数据范围的评估对于理解并改进模型行为至关重要。MultiTab 促进了更加原则化的模型设计，并提供了根据特定数据特征选择模型的实用指导。所有数据集、代码和优化日志均在该网址公开：this https URL。 

---
# Benchmarking data encoding methods in Quantum Machine Learning 

**Title (ZH)**: 量子机器学习中数据编码方法的基准测试 

**Authors**: Orlane Zang, Grégoire Barrué, Tony Quertier  

**Link**: [PDF](https://arxiv.org/pdf/2505.14295)  

**Abstract**: Data encoding plays a fundamental and distinctive role in Quantum Machine Learning (QML). While classical approaches process data directly as vectors, QML may require transforming classical data into quantum states through encoding circuits, known as quantum feature maps or quantum embeddings. This step leverages the inherently high-dimensional and non-linear nature of Hilbert space, enabling more efficient data separation in complex feature spaces that may be inaccessible to classical methods. This encoding part significantly affects the performance of the QML model, so it is important to choose the right encoding method for the dataset to be encoded. However, this choice is generally arbitrary, since there is no "universal" rule for knowing which encoding to choose based on a specific set of data. There are currently a variety of encoding methods using different quantum logic gates. We studied the most commonly used types of encoding methods and benchmarked them using different datasets. 

**Abstract (ZH)**: 量子态编码在量子机器学习（QML）中扮演着基础且独特的角色。尽管经典方法直接将数据处理为向量，QML 可能需要通过编码电路，即量子特征映射或量子嵌入，将经典数据转变为量子态。这一步骤利用了希尔伯特空间固有的高维和非线性性质，使其能够在经典方法难以访问的复杂特征空间中更高效地实现数据分离。这一编码步骤显著影响QML模型的性能，因此选择适合的数据集编码方法至关重要。然而，这种选择通常是任意的，因为没有适用于特定数据集的“通用”规则来确定选择哪种编码方法。目前存在多种采用不同量子逻辑门的编码方法。我们研究了最常用的几种编码方法，并使用不同数据集进行了基准测试。 

---
# AquaSignal: An Integrated Framework for Robust Underwater Acoustic Analysis 

**Title (ZH)**: AquaSignal：一种稳健的水下声学分析综合框架 

**Authors**: Eirini Panteli, Paulo E. Santos, Nabil Humphrey  

**Link**: [PDF](https://arxiv.org/pdf/2505.14285)  

**Abstract**: This paper presents AquaSignal, a modular and scalable pipeline for preprocessing, denoising, classification, and novelty detection of underwater acoustic signals. Designed to operate effectively in noisy and dynamic marine environments, AquaSignal integrates state-of-the-art deep learning architectures to enhance the reliability and accuracy of acoustic signal analysis. The system is evaluated on a combined dataset from the Deepship and Ocean Networks Canada (ONC) benchmarks, providing a diverse set of real-world underwater scenarios. AquaSignal employs a U-Net architecture for denoising, a ResNet18 convolutional neural network for classifying known acoustic events, and an AutoEncoder-based model for unsupervised detection of novel or anomalous signals. To our knowledge, this is the first comprehensive study to apply and evaluate this combination of techniques on maritime vessel acoustic data. Experimental results show that AquaSignal improves signal clarity and task performance, achieving 71% classification accuracy and 91% accuracy in novelty detection. Despite slightly lower classification performance compared to some state-of-the-art models, differences in data partitioning strategies limit direct comparisons. Overall, AquaSignal demonstrates strong potential for real-time underwater acoustic monitoring in scientific, environmental, and maritime domains. 

**Abstract (ZH)**: AquaSignal：一种模块化可扩展的水下声信号预处理、去噪、分类和新颖性检测流水线 

---
# YESciEval: Robust LLM-as-a-Judge for Scientific Question Answering 

**Title (ZH)**: YESciEval: Robust LLM-as-a-Judge for Scientific Question Answering 

**Authors**: Jennifer D'Souza, Hamed Babaei Giglou, Quentin Münch  

**Link**: [PDF](https://arxiv.org/pdf/2505.14279)  

**Abstract**: Large Language Models (LLMs) drive scientific question-answering on modern search engines, yet their evaluation robustness remains underexplored. We introduce YESciEval, an open-source framework that combines fine-grained rubric-based assessment with reinforcement learning to mitigate optimism bias in LLM evaluators. We release multidisciplinary scienceQ&A datasets, including adversarial variants, with evaluation scores from multiple LLMs. Independent of proprietary models and human feedback, our approach enables scalable, cost-free evaluation. By advancing reliable LLM-as-a-judge models, this work supports AI alignment and fosters robust, transparent evaluation essential for scientific inquiry and artificial general intelligence. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在现代搜索引擎中的科学问题解答中发挥着关键作用，但其评估稳健性仍鲜有探讨。我们提出YESciEval，一个结合了细粒度 rubric 评估与强化学习的开源框架，旨在减轻大规模语言模型评估者中的乐观偏见。我们发布了跨学科的科学问答数据集，包括对抗变体，并提供了multiple LLM的评估分数。独立于专有模型和人类反馈，我们的方法能够实现可扩展且无需成本的评估。通过推进可靠的“LLM作为法官”模型，这项工作支持了AI对齐，并促进了对于科学研究和通用人工智能而言至关重要的稳健且透明的评估。 

---
# X-KAN: Optimizing Local Kolmogorov-Arnold Networks via Evolutionary Rule-Based Machine Learning 

**Title (ZH)**: X-KAN：通过进化基于规则的机器学习优化局部柯莫洛夫-阿诺尔德网络 

**Authors**: Hiroki Shiraishi, Hisao Ishibuchi, Masaya Nakata  

**Link**: [PDF](https://arxiv.org/pdf/2505.14273)  

**Abstract**: Function approximation is a critical task in various fields. However, existing neural network approaches struggle with locally complex or discontinuous functions due to their reliance on a single global model covering the entire problem space. We propose X-KAN, a novel method that optimizes multiple local Kolmogorov-Arnold Networks (KANs) through an evolutionary rule-based machine learning framework called XCSF. X-KAN combines KAN's high expressiveness with XCSF's adaptive partitioning capability by implementing local KAN models as rule consequents and defining local regions via rule antecedents. Our experimental results on artificial test functions and real-world datasets demonstrate that X-KAN significantly outperforms conventional methods, including XCSF, Multi-Layer Perceptron, and KAN, in terms of approximation accuracy. Notably, X-KAN effectively handles functions with locally complex or discontinuous structures that are challenging for conventional KAN, using a compact set of rules (average 7.2 $\pm$ 2.3 rules). These results validate the effectiveness of using KAN as a local model in XCSF, which evaluates the rule fitness based on both accuracy and generality. Our X-KAN implementation is available at this https URL. 

**Abstract (ZH)**: X-KAN：基于XCSF框架的多方块Kolmogorov-Arnold网络函数近似方法 

---
# Think-J: Learning to Think for Generative LLM-as-a-Judge 

**Title (ZH)**: Think-J: 学习为生成型LLM进行判断性思考 

**Authors**: Hui Huang, Yancheng He, Hongli Zhou, Rui Zhang, Wei Liu, Weixun Wang, Wenbo Su, Bo Zheng, Jiaheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14268)  

**Abstract**: LLM-as-a-Judge refers to the automatic modeling of preferences for responses generated by Large Language Models (LLMs), which is of significant importance for both LLM evaluation and reward modeling. Although generative LLMs have made substantial progress in various tasks, their performance as LLM-Judge still falls short of expectations. In this work, we propose Think-J, which improves generative LLM-as-a-Judge by learning how to think. We first utilized a small amount of curated data to develop the model with initial judgment thinking capabilities. Subsequently, we optimize the judgment thinking traces based on reinforcement learning (RL). We propose two methods for judgment thinking optimization, based on offline and online RL, respectively. The offline RL requires training a critic model to construct positive and negative examples for learning. The online method defines rule-based reward as feedback for optimization. Experimental results showed that our approach can significantly enhance the evaluation capability of generative LLM-Judge, surpassing both generative and classifier-based LLM-Judge without requiring extra human annotations. 

**Abstract (ZH)**: LLM-as-a-Judge作为一种自动建模大型语言模型生成响应偏好的方法，在大型语言模型评估和奖励建模中具有重要意义。尽管生成型大型语言模型在各种任务中取得了显著进展，但其作为LLM-Judge的表现仍未能达到预期。本文提出Think-J，通过学习如何思考来提高生成型大型语言模型的评估能力。我们首先利用少量精加工的数据开发模型，赋予其初步的判断思考能力。随后，基于强化学习（RL）优化判断思考轨迹。我们提出了两种基于离线和在线RL的判断思考优化方法。离线RL需要训练一个批判模型以构建正反例进行学习；在线方法则通过基于规则的奖励作为反馈进行优化。实验结果表明，我们的方法能够显著增强生成型大型语言模型的评估能力，且无需额外的人工标注，超越了基于生成和分类的大型语言模型评估方法。 

---
# Speculative Decoding Reimagined for Multimodal Large Language Models 

**Title (ZH)**: 重想象的 speculative decoding 用于多模态大规模语言模型 

**Authors**: Luxi Lin, Zhihang Lin, Zhanpeng Zeng, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.14260)  

**Abstract**: This paper introduces Multimodal Speculative Decoding (MSD) to accelerate Multimodal Large Language Models (MLLMs) inference. Speculative decoding has been shown to accelerate Large Language Models (LLMs) without sacrificing accuracy. However, current speculative decoding methods for MLLMs fail to achieve the same speedup as they do for LLMs. To address this, we reimagine speculative decoding specifically for MLLMs. Our analysis of MLLM characteristics reveals two key design principles for MSD: (1) Text and visual tokens have fundamentally different characteristics and need to be processed separately during drafting. (2) Both language modeling ability and visual perception capability are crucial for the draft model. For the first principle, MSD decouples text and visual tokens in the draft model, allowing each to be handled based on its own characteristics. For the second principle, MSD uses a two-stage training strategy: In stage one, the draft model is trained on text-only instruction-tuning datasets to improve its language modeling ability. In stage two, MSD gradually introduces multimodal data to enhance the visual perception capability of the draft model. Experiments show that MSD boosts inference speed by up to $2.29\times$ for LLaVA-1.5-7B and up to $2.46\times$ for LLaVA-1.5-13B on multimodal benchmarks, demonstrating its effectiveness. Our code is available at this https URL. 

**Abstract (ZH)**: 这篇论文介绍了多模态投机解码（MSD）以加速多模态大型语言模型（MLLMs）的推理。 

---
# FuxiMT: Sparsifying Large Language Models for Chinese-Centric Multilingual Machine Translation 

**Title (ZH)**: FuxiMT：为中国中心的多语言机器翻译精简大型语言模型 

**Authors**: Shaolin Zhu, Tianyu Dong, Bo Li, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2505.14256)  

**Abstract**: In this paper, we present FuxiMT, a novel Chinese-centric multilingual machine translation model powered by a sparsified large language model (LLM). We adopt a two-stage strategy to train FuxiMT. We first pre-train the model on a massive Chinese corpus and then conduct multilingual fine-tuning on a large parallel dataset encompassing 65 languages. FuxiMT incorporates Mixture-of-Experts (MoEs) and employs a curriculum learning strategy for robust performance across various resource levels. Experimental results demonstrate that FuxiMT significantly outperforms strong baselines, including state-of-the-art LLMs and machine translation models, particularly under low-resource scenarios. Furthermore, FuxiMT exhibits remarkable zero-shot translation capabilities for unseen language pairs, indicating its potential to bridge communication gaps where parallel data are scarce or unavailable. 

**Abstract (ZH)**: 基于稀疏大型语言模型的面向中文的多语言机器翻译模型FuxiMT 

---
# Hybrid Adaptive Modeling in Process Monitoring: Leveraging Sequence Encoders and Physics-Informed Neural Networks 

**Title (ZH)**: 过程监控中的混合自适应建模：结合序列编码器和物理知情神经网络 

**Authors**: Mouad Elaarabi, Domenico Borzacchiello, Philippe Le Bot, Nathan Lauzeral, Sebastien Comas-Cardona  

**Link**: [PDF](https://arxiv.org/pdf/2505.14252)  

**Abstract**: In this work, we explore the integration of Sequence Encoding for Online Parameter Identification with Physics-Informed Neural Networks to create a model that, once trained, can be utilized for real time applications with variable parameters, boundary conditions, and initial conditions. Recently, the combination of PINNs with Sparse Regression has emerged as a method for performing dynamical system identification through supervised learning and sparse regression optimization, while also solving the dynamics using PINNs. However, this approach can be limited by variations in parameters or boundary and initial conditions, requiring retraining of the model whenever changes occur. In this work, we introduce an architecture that employs Deep Sets or Sequence Encoders to encode dynamic parameters, boundary conditions, and initial conditions, using these encoded features as inputs for the PINN, enabling the model to adapt to changes in parameters, BCs, and ICs. We apply this approach to three different problems. First, we analyze the Rossler ODE system, demonstrating the robustness of the model with respect to noise and its ability to generalize. Next, we explore the model's capability in a 2D Navier-Stokes PDE problem involving flow past a cylinder with a parametric sinusoidal inlet velocity function, showing that the model can encode pressure data from a few points to identify the inlet velocity profile and utilize physics to compute velocity and pressure throughout the domain. Finally, we address a 1D heat monitoring problem using real data from the heating of glass fiber and thermoplastic composite plates. 

**Abstract (ZH)**: 基于序列编码的物理知情神经网络在实时参数识别中的应用研究 

---
# Visual Agentic Reinforcement Fine-Tuning 

**Title (ZH)**: 视觉代理强化微调 

**Authors**: Ziyu Liu, Yuhang Zang, Yushan Zou, Zijian Liang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14246)  

**Abstract**: A key trend in Large Reasoning Models (e.g., OpenAI's o3) is the native agentic ability to use external tools such as web browsers for searching and writing/executing code for image manipulation to think with images. In the open-source research community, while significant progress has been made in language-only agentic abilities such as function calling and tool integration, the development of multi-modal agentic capabilities that involve truly thinking with images, and their corresponding benchmarks, are still less explored. This work highlights the effectiveness of Visual Agentic Reinforcement Fine-Tuning (Visual-ARFT) for enabling flexible and adaptive reasoning abilities for Large Vision-Language Models (LVLMs). With Visual-ARFT, open-source LVLMs gain the ability to browse websites for real-time information updates and write code to manipulate and analyze input images through cropping, rotation, and other image processing techniques. We also present a Multi-modal Agentic Tool Bench (MAT) with two settings (MAT-Search and MAT-Coding) designed to evaluate LVLMs' agentic search and coding abilities. Our experimental results demonstrate that Visual-ARFT outperforms its baseline by +18.6% F1 / +13.0% EM on MAT-Coding and +10.3% F1 / +8.7% EM on MAT-Search, ultimately surpassing GPT-4o. Visual-ARFT also achieves +29.3 F1% / +25.9% EM gains on existing multi-hop QA benchmarks such as 2Wiki and HotpotQA, demonstrating strong generalization capabilities. Our findings suggest that Visual-ARFT offers a promising path toward building robust and generalizable multimodal agents. 

**Abstract (ZH)**: 大型推理模型中的一个关键趋势（例如OpenAI的o3）是原生代理能力，能够使用外部工具如网络浏览器进行搜索和编写/执行代码以通过图像操作进行思考。在开源研究社区中，虽然在仅语言代理能力方面（如函数调用和工具集成）取得了显著进展，但涉及到真正通过图像进行思考的多模态代理能力及其相应的基准测试仍较少探索。本研究强调了视觉代理强化微调（Visual-ARFT）的有效性，以增强大型视觉语言模型（LVLMs）的灵活和适应性推理能力。通过Visual-ARFT，开源LVLMs能够浏览网站以获取实时信息更新，并编写代码以通过裁剪、旋转和其他图像处理技术来操作和分析输入图像。我们还提出了一个多模态代理工具基准（MAT），其中包含两个设置（MAT-Search和MAT-Coding），用于评估LVLMs的代理搜索和编程能力。实验结果表明，Visual-ARFT在MAT-Coding上的F1得分提高了18.6% / EM提高了13.0%，在MAT-Search上的F1得分提高了10.3% / EM提高了8.7%，最终超越了GPT-4o。Visual-ARFT还在2Wiki和HotpotQA等现有多跳问答基准测试中实现了29.3%的F1增益 / 25.9%的EM增益，显示出强大的泛化能力。我们的研究发现表明，Visual-ARFT为构建鲁棒性和通用性兼备的多模态代理提供了有希望的道路。 

---
# ABBA: Highly Expressive Hadamard Product Adaptation for Large Language Models 

**Title (ZH)**: ABBA：大型语言模型中高度表达性的哈达玛积适应方法 

**Authors**: Raghav Singhal, Kaustubh Ponkshe, Rohit Vartak, Praneeth Vepakomma  

**Link**: [PDF](https://arxiv.org/pdf/2505.14238)  

**Abstract**: Large Language Models have demonstrated strong performance across a wide range of tasks, but adapting them efficiently to new domains remains a key challenge. Parameter-Efficient Fine-Tuning (PEFT) methods address this by introducing lightweight, trainable modules while keeping most pre-trained weights fixed. The prevailing approach, LoRA, models updates using a low-rank decomposition, but its expressivity is inherently constrained by the rank. Recent methods like HiRA aim to increase expressivity by incorporating a Hadamard product with the frozen weights, but still rely on the structure of the pre-trained model. We introduce ABBA, a new PEFT architecture that reparameterizes the update as a Hadamard product of two independently learnable low-rank matrices. In contrast to prior work, ABBA fully decouples the update from the pre-trained weights, enabling both components to be optimized freely. This leads to significantly higher expressivity under the same parameter budget. We formally analyze ABBA's expressive capacity and validate its advantages through matrix reconstruction experiments. Empirically, ABBA achieves state-of-the-art results on arithmetic and commonsense reasoning benchmarks, consistently outperforming existing PEFT methods by a significant margin across multiple models. Our code is publicly available at: this https URL. 

**Abstract (ZH)**: 大规模语言模型在广泛的任务中展现了强大的性能，但高效适应新领域仍然是一个关键挑战。参数高效微调（PEFT）方法通过引入轻量级、可训练模块，同时固定大部分预训练权重来应对这一挑战。目前占主导地位的方法LoRA使用低秩分解来建模更新，但其表达能力受到秩的内在约束。最近的方法HiRA试图通过引入与冻结权重的Hadamard积来增加表达能力，但仍依赖预训练模型的结构。我们引入了ABBA，这是一种新的PEFT架构，将更新重新参数化为两个独立可学习低秩矩阵的Hadamard积。与先前工作不同，ABBA完全解耦了更新与预训练权重的关系，使得两个组件可以自由优化。这在相同的参数预算下实现了更高的表达能力。我们形式化分析了ABBA的表达能力，并通过矩阵重构实验验证了其优势。实验中，ABBA在算术和常识推理基准测试中取得了最优结果，在多个模型上显著优于现有PEFT方法。我们的代码已公开：this https URL。 

---
# Fast and close Shannon entropy approximation 

**Title (ZH)**: 快速且接近的香农熵近似 

**Authors**: Illia Horenko, Davide Bassetti, Lukáš Pospíšil  

**Link**: [PDF](https://arxiv.org/pdf/2505.14234)  

**Abstract**: Shannon entropy (SE) and its quantum mechanical analogue von Neumann entropy are key components in many tools used in physics, information theory, machine learning (ML) and quantum computing. Besides of the significant amounts of SE computations required in these fields, the singularity of the SE gradient is one of the central mathematical reason inducing the high cost, frequently low robustness and slow convergence of such tools. Here we propose the Fast Entropy Approximation (FEA) - a non-singular rational approximation of Shannon entropy and its gradient that achieves a mean absolute error of $10^{-3}$, which is approximately $20$ times lower than comparable state-of-the-art methods. FEA allows around $50\%$ faster computation, requiring only $5$ to $6$ elementary computational operations, as compared to tens of elementary operations behind the fastest entropy computation algorithms with table look-ups, bitshifts, or series approximations. On a set of common benchmarks for the feature selection problem in machine learning, we show that the combined effect of fewer elementary operations, low approximation error, and a non-singular gradient allows significantly better model quality and enables ML feature extraction that is two to three orders of magnitude faster and computationally cheaper when incorporating FEA into AI tools. 

**Abstract (ZH)**: 香农熵（SE）及其量子力学类比冯诺伊曼熵是物理学、信息论、机器学习（ML）和量子计算中许多工具的关键组成部分。除了在这些领域中大量计算香农熵的需求外，香农熵梯度的奇异性是导致这些工具高昂成本、经常较低鲁棒性和缓慢收敛的主要数学原因之一。我们提出了快速熵近似（FEA）——一种非奇异的有理数约简方法，实现绝对误差为 \(10^{-3}\)，大约比目前最先进的方法低20倍。FEA 允许约50%更快的计算，只需5到6个基本计算操作，而最快速的熵计算算法则需要数十个基本操作，涉及表格查找、位移或级数近似。在机器学习中特征选择问题的一组常见基准上，我们展示了较少的基本操作、低近似误差和非奇异梯度的综合效果，使得模型质量显著提高，并且将FEA整合到AI工具中时，使ML特征提取速度提高了两到三个数量级，计算成本也大大降低。 

---
# Mechanistic Fine-tuning for In-context Learning 

**Title (ZH)**: 机制微调以实现上下文学习 

**Authors**: Hakaze Cho, Peng Luo, Mariko Kato, Rin Kaenbyou, Naoya Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2505.14233)  

**Abstract**: In-context Learning (ICL) utilizes structured demonstration-query inputs to induce few-shot learning on Language Models (LMs), which are not originally pre-trained on ICL-style data. To bridge the gap between ICL and pre-training, some approaches fine-tune LMs on large ICL-style datasets by an end-to-end paradigm with massive computational costs. To reduce such costs, in this paper, we propose Attention Behavior Fine-Tuning (ABFT), utilizing the previous findings on the inner mechanism of ICL, building training objectives on the attention scores instead of the final outputs, to force the attention scores to focus on the correct label tokens presented in the context and mitigate attention scores from the wrong label tokens. Our experiments on 9 modern LMs and 8 datasets empirically find that ABFT outperforms in performance, robustness, unbiasedness, and efficiency, with only around 0.01% data cost compared to the previous methods. Moreover, our subsequent analysis finds that the end-to-end training objective contains the ABFT objective, suggesting the implicit bias of ICL-style data to the emergence of induction heads. Our work demonstrates the possibility of controlling specific module sequences within LMs to improve their behavior, opening up the future application of mechanistic interpretability. 

**Abstract (ZH)**: 基于上下文学习的注意力行为微调（ABFT）：一种低数据成本的 few-shot 学习方法 

---
# VoQA: Visual-only Question Answering 

**Title (ZH)**: 仅视觉问题回答 

**Authors**: Luyang Jiang, Jianing An, Jie Luo, Wenjun Wu, Lei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14227)  

**Abstract**: We propose Visual-only Question Answering (VoQA), a novel multimodal task in which questions are visually embedded within images, without any accompanying textual input. This requires models to locate, recognize, and reason over visually embedded textual questions, posing challenges for existing large vision-language models (LVLMs), which show notable performance drops even with carefully designed prompts. To bridge this gap, we introduce Guided Response Triggering Supervised Fine-tuning (GRT-SFT), a structured fine-tuning strategy that guides the model to perform step-by-step reasoning purely based on visual input, significantly improving model performance. Our work enhances models' capacity for human-like visual understanding in complex multimodal scenarios, where information, including language, is perceived visually. 

**Abstract (ZH)**: 视觉-only 问答（VoQA）：一种新的多模态任务 

---
# "Haet Bhasha aur Diskrimineshun": Phonetic Perturbations in Code-Mixed Hinglish to Red-Team LLMs 

**Title (ZH)**: “哈et语言和歧视”：代码混合 hinGlish中的音素干扰以红队LLMs 

**Authors**: Darpan Aswal, Siddharth D Jaiswal  

**Link**: [PDF](https://arxiv.org/pdf/2505.14226)  

**Abstract**: Large Language Models (LLMs) have become increasingly powerful, with multilingual and multimodal capabilities improving by the day. These models are being evaluated through audits, alignment studies and red-teaming efforts to expose model vulnerabilities towards generating harmful, biased and unfair content. Existing red-teaming efforts have previously focused on the English language, using fixed template-based attacks; thus, models continue to be susceptible to multilingual jailbreaking strategies, especially in the multimodal context. In this study, we introduce a novel strategy that leverages code-mixing and phonetic perturbations to jailbreak LLMs for both text and image generation tasks. We also introduce two new jailbreak strategies that show higher effectiveness than baseline strategies. Our work presents a method to effectively bypass safety filters in LLMs while maintaining interpretability by applying phonetic misspellings to sensitive words in code-mixed prompts. Our novel prompts achieve a 99% Attack Success Rate for text generation and 78% for image generation, with Attack Relevance Rate of 100% for text generation and 95% for image generation when using the phonetically perturbed code-mixed prompts. Our interpretability experiments reveal that phonetic perturbations impact word tokenization, leading to jailbreak success. Our study motivates increasing the focus towards more generalizable safety alignment for multilingual multimodal models, especially in real-world settings wherein prompts can have misspelt words. 

**Abstract (ZH)**: 大型语言模型（LLMs）的能力不断增强，具备多语言和多模态能力，正在通过审核、对齐研究和红队攻击等手段进行评估，以暴露模型生成有害、偏见和不公平内容的漏洞。现有的红队攻击主要针对英语，使用固定模板攻击；因此，模型仍然容易受到多语言逃逸策略的影响，尤其是在多模态情境下。本研究提出了一种新的策略，利用混合编程和音素扰动技术，实现对文本和图像生成任务中LLMs的逃逸。我们还引入了两种新的逃逸策略，其有效性高于基线策略。我们的研究提出了一种方法，在不影响解释性的情况下，通过在混合编程提示中对敏感词汇应用音素误写，有效地绕过LLMs中的安全过滤器。我们的新型提示在文本生成上的攻击成功率达到了99%，图像生成上的攻击成功率达到了78%，使用音素扰动的混合编程提示时，文本生成上的攻击相关率为100%，图像生成上的攻击相关率为95%。我们的解释性实验表明，音素扰动影响了词元化，导致逃逸成功。本研究促使我们关注更具普适性的多语言多模态模型的安全对齐，尤其是在实际应用场景中，提示可能包含拼写错误的情况下。 

---
# Federated learning in low-resource settings: A chest imaging study in Africa -- Challenges and lessons learned 

**Title (ZH)**: 低资源环境下联邦学习的研究：非洲胸部成像研究中的挑战与经验教训 

**Authors**: Jorge Fabila, Lidia Garrucho, Víctor M. Campello, Carlos Martín-Isla, Karim Lekadir  

**Link**: [PDF](https://arxiv.org/pdf/2505.14217)  

**Abstract**: This study explores the use of Federated Learning (FL) for tuberculosis (TB) diagnosis using chest X-rays in low-resource settings across Africa. FL allows hospitals to collaboratively train AI models without sharing raw patient data, addressing privacy concerns and data scarcity that hinder traditional centralized models. The research involved hospitals and research centers in eight African countries. Most sites used local datasets, while Ghana and The Gambia used public ones. The study compared locally trained models with a federated model built across all institutions to evaluate FL's real-world feasibility. Despite its promise, implementing FL in sub-Saharan Africa faces challenges such as poor infrastructure, unreliable internet, limited digital literacy, and weak AI regulations. Some institutions were also reluctant to share model updates due to data control concerns. In conclusion, FL shows strong potential for enabling AI-driven healthcare in underserved regions, but broader adoption will require improvements in infrastructure, education, and regulatory support. 

**Abstract (ZH)**: 本研究探索在非洲低资源地区使用联邦学习（FL）结合胸部X光进行结核病（TB）诊断的应用。FL使医院能够在不共享原始患者数据的情况下协作训练AI模型，从而解决传统集中模型中困扰的数据隐私和数据稀缺问题。该研究涉及非洲八个国家的医院和研究中心。大多数地点使用本地数据集，而加纳和冈比亚使用公开数据集。研究将本地训练的模型与跨所有机构构建的联邦模型进行比较，以评估FL在实际应用中的可行性。尽管FL具有巨大潜力，但在撒哈拉以南非洲地区实施仍面临基础设施差、互联网不稳定、数字素养有限以及AI监管不足等挑战。一些机构还因数据控制问题而对分享模型更新持保留态度。总之，FL展示了在偏远地区推动AI驱动医疗保健的强大潜力，但更广泛的应用将需要在基础设施、教育和监管支持方面进行改进。 

---
# Automatic Dataset Generation for Knowledge Intensive Question Answering Tasks 

**Title (ZH)**: 知识密集型问答任务的自动数据集生成 

**Authors**: Sizhe Yuen, Ting Su, Ziyang Wang, Yali Du, Adam J. Sobey  

**Link**: [PDF](https://arxiv.org/pdf/2505.14212)  

**Abstract**: A question-answering (QA) system is to search suitable answers within a knowledge base. Current QA systems struggle with queries requiring complex reasoning or real-time knowledge integration. They are often supplemented with retrieval techniques on a data source such as Retrieval-Augmented Generation (RAG). However, RAG continues to face challenges in handling complex reasoning and logical connections between multiple sources of information. A novel approach for enhancing Large Language Models (LLMs) in knowledge-intensive QA tasks is presented through the automated generation of context-based QA pairs. This methodology leverages LLMs to create fine-tuning data, reducing reliance on human labelling and improving model comprehension and reasoning capabilities. The proposed system includes an automated QA generator and a model fine-tuner, evaluated using perplexity, ROUGE, BLEU, and BERTScore. Comprehensive experiments demonstrate improvements in logical coherence and factual accuracy, with implications for developing adaptable Artificial Intelligence (AI) systems. Mistral-7b-v0.3 outperforms Llama-3-8b with BERT F1, BLEU, and ROUGE scores 0.858, 0.172, and 0.260 of for the LLM generated QA pairs compared to scores of 0.836, 0.083, and 0.139 for the human annotated QA pairs. 

**Abstract (ZH)**: 一种基于上下文的问答对自动生成方法可以增强大型语言模型在知识密集型问答任务中的性能。该方法利用大型语言模型创建微调数据，减少对人工标注的依赖，提高模型的理解和推理能力。所提出系统包括一个自动问答生成器和一个模型微调器，并通过困惑度、ROUGE、BLEU和BERTScore进行评估。全面的实验展示了逻辑连贯性和事实准确性的改进，对开发 adaptable 人工智能系统具有重要意义。Mistral-7b-v0.3在BERT F1、BLEU和ROUGE分数上优于Llama-3-8b，LLM生成的问答对的分数分别为0.858、0.172和0.260，而人工标注的问答对的分数分别为0.836、0.083和0.139。 

---
# Challenges and Limitations in the Synthetic Generation of mHealth Sensor Data 

**Title (ZH)**: 合成生成mHealth传感器数据的挑战与局限性 

**Authors**: Flavio Di Martino, Franca Delmastro  

**Link**: [PDF](https://arxiv.org/pdf/2505.14206)  

**Abstract**: The widespread adoption of mobile sensors has the potential to provide massive and heterogeneous time series data, driving Artificial Intelligence applications in mHealth. However, data collection remains limited due to stringent ethical regulations, privacy concerns, and other constraints, hindering progress in the field. Synthetic data generation, particularly through Generative Adversarial Networks and Diffusion Models, has emerged as a promising solution to address both data scarcity and privacy issues. Yet, these models are often limited to short-term, unimodal signal patterns. This paper presents a systematic evaluation of state-of-the-art generative models for time series synthesis, with a focus on their ability to jointly handle multi-modality, long-range dependencies, and conditional generation-key challenges in the mHealth domain. To ensure a fair comparison, we introduce a novel evaluation framework designed to measure both the intrinsic quality of synthetic data and its utility in downstream predictive tasks. Our findings reveal critical limitations in the existing approaches, particularly in maintaining cross-modal consistency, preserving temporal coherence, and ensuring robust performance in train-on-synthetic, test-on-real, and data augmentation scenarios. Finally, we present our future research directions to enhance synthetic time series generation and improve the applicability of generative models in mHealth. 

**Abstract (ZH)**: 广泛采用的移动传感器有可能提供庞大的异质时间序列数据，推动健康医疗领域的智能应用。然而，由于严格的伦理规范、隐私担忧和其他限制，数据收集仍然受到限制，阻碍了该领域的进展。生成式对抗网络和扩散模型等生成数据的生成方法已成为解决数据短缺和隐私问题的有前途的解决方案。尽管如此，这些模型往往仅限于处理短期的单模态信号模式。本文系统评估了当前最先进的生成模型在时间序列合成中的性能，重点在于它们在健康医疗领域处理多模态性、长距离依赖性和条件生成等关键挑战的能力。为了确保公平比较，我们引入了一个新的评估框架，用于测量合成数据的内在质量和其在下游预测任务中的实用性。我们的研究发现揭示了现有方法的关键局限性，特别是在维护跨模态一致性、保持时间连贯性以及保证在训练使用合成数据、测试使用真实数据和数据增强场景中的鲁棒性能方面。最后，我们提出了未来的研究方向，以增强时间序列生成的合成效果，并提高生成模型在健康医疗领域的适用性。 

---
# FLASH-D: FlashAttention with Hidden Softmax Division 

**Title (ZH)**: FLASH-D: FlashAttention with Hidden Softmax Division 

**Authors**: Kosmas Alexandridis, Vasileios Titopoulos, Giorgos Dimitrakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2505.14201)  

**Abstract**: The transformer's attention mechanism has revolutionized AI and machine learning, with its efficient computation being crucial to its performance. However, calculating attention involves matrix operations interspersed with softmax rescaling, which inherently slows down computation and requires processing the entire input sequence. Building on online softmax computation, FlashAttention integrates softmax calculation with matrix arithmetic, enabling tiled computation independent of sequence length. While optimized for GPUs, FlashAttention's simplicity makes it amenable to direct hardware acceleration. This work re-evaluates the core FlashAttention kernel, presenting FLASH-D a mathematically equivalent, yet simplified, formulation that achieves: (a) hiding softmax division within other non-linear function evaluations; (b) inherently numerically stable computation of exponentials, eliminating the need for maximum value subtraction; and (c) a reduction in computational cost without introducing numerical approximations to the FlashAttention kernel. Importantly, the essential FlashAttention properties that facilitate efficient tiled implementation are fully preserved. Hardware implementation results at 28nm demonstrate that this proposed formulation achieves a 22.8% reduction in area and a 20.3% reduction in power, on average, compared to state-of-the-art parallel hardware architectures without any performance penalty. 

**Abstract (ZH)**: FlashAttention的数学等价简化形式FLASH-D：保持高效分块实现的同时减少计算成本和硬件面积能耗 

---
# $α$-GAN by Rényi Cross Entropy 

**Title (ZH)**: α-GAN通过Rényi交叉熵 

**Authors**: Ni Ding, Miao Qiao, Jiaxing Xu, Yiping Ke, Xiaoyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14190)  

**Abstract**: This paper proposes $\alpha$-GAN, a generative adversarial network using Rényi measures. The value function is formulated, by Rényi cross entropy, as an expected certainty measure incurred by the discriminator's soft decision as to where the sample is from, true population or the generator. The discriminator tries to maximize the Rényi certainty about sample source, while the generator wants to reduce it by injecting fake samples. This forms a min-max problem with the solution parameterized by the Rényi order $\alpha$. This $\alpha$-GAN reduces to vanilla GAN at $\alpha = 1$, where the value function is exactly the binary cross entropy. The optimization of $\alpha$-GAN is over probability (vector) space. It is shown that the gradient is exponentially enlarged when Rényi order is in the range $\alpha \in (0,1)$. This makes convergence faster, which is verified by experimental results. A discussion shows that choosing $\alpha \in (0,1)$ may be able to solve some common problems, e.g., vanishing gradient. A following observation reveals that this range has not been fully explored in the existing Rényi version GANs. 

**Abstract (ZH)**: α-GAN：使用 Rényi 度量的生成对抗网络 

---
# Safety Subspaces are Not Distinct: A Fine-Tuning Case Study 

**Title (ZH)**: 安全子空间并非孤立：一项微调案例研究 

**Authors**: Kaustubh Ponkshe, Shaan Shah, Raghav Singhal, Praneeth Vepakomma  

**Link**: [PDF](https://arxiv.org/pdf/2505.14185)  

**Abstract**: Large Language Models (LLMs) rely on safety alignment to produce socially acceptable responses. This is typically achieved through instruction tuning and reinforcement learning from human feedback. However, this alignment is known to be brittle: further fine-tuning, even on benign or lightly contaminated data, can degrade safety and reintroduce harmful behaviors. A growing body of work suggests that alignment may correspond to identifiable geometric directions in weight space, forming subspaces that could, in principle, be isolated or preserved to defend against misalignment. In this work, we conduct a comprehensive empirical study of this geometric perspective. We examine whether safety-relevant behavior is concentrated in specific subspaces, whether it can be separated from general-purpose learning, and whether harmfulness arises from distinguishable patterns in internal representations. Across both parameter and activation space, our findings are consistent: subspaces that amplify safe behaviors also amplify unsafe ones, and prompts with different safety implications activate overlapping representations. We find no evidence of a subspace that selectively governs safety. These results challenge the assumption that alignment is geometrically localized. Rather than residing in distinct directions, safety appears to emerge from entangled, high-impact components of the model's broader learning dynamics. This suggests that subspace-based defenses may face fundamental limitations and underscores the need for alternative strategies to preserve alignment under continued training. We corroborate these findings through multiple experiments on five open-source LLMs. Our code is publicly available at: this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）依赖于安全性对齐以生成社会可接受的响应。这通常通过指令微调和从人类反馈中进行强化学习来实现。然而，这种对齐已知是脆弱的：即使是微调无害或轻微污染的数据，也可能损害安全性并重新引入有害行为。越来越多的研究表明，对齐可能对应于权重空间中的可识别的几何方向，形成子空间，这些子空间原则上可以被隔离或保留以防止对齐失效。在本工作中，我们进行了全面的实证研究，以探讨这一几何视角。我们研究了安全相关的行为是否集中在特定的子空间中，这些行为是否可以与通用学习分开，以及有害性是否源于内部表示中的可区分模式。在参数空间和激活空间中，我们的发现是一致的：放大安全行为的子空间也放大了不安全的行为，不同安全含义的提示激活了重叠的表示。我们没有发现一个单独治理安全性的子空间。这些结果挑战了对齐是几何局部化的一种假设。安全性似乎不是源自独特的方向，而是源自模型更广泛学习动态中交织的关键组件。这表明基于子空间的防御可能面临根本性局限，并突显了在持续训练中保持对齐的需要。我们通过在五个开源LLM上进行多项实验来验证这些发现。我们的代码可在以下网址公开获取：this https URL。 

---
# Enhancing Abstractive Summarization of Scientific Papers Using Structure Information 

**Title (ZH)**: 基于结构信息增强科学论文的抽象总结 

**Authors**: Tong Bao, Heng Zhang, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14179)  

**Abstract**: Abstractive summarization of scientific papers has always been a research focus, yet existing methods face two main challenges. First, most summarization models rely on Encoder-Decoder architectures that treat papers as sequences of words, thus fail to fully capture the structured information inherent in scientific papers. Second, existing research often use keyword mapping or feature engineering to identify the structural information, but these methods struggle with the structural flexibility of scientific papers and lack robustness across different disciplines. To address these challenges, we propose a two-stage abstractive summarization framework that leverages automatic recognition of structural functions within scientific papers. In the first stage, we standardize chapter titles from numerous scientific papers and construct a large-scale dataset for structural function recognition. A classifier is then trained to automatically identify the key structural components (e.g., Background, Methods, Results, Discussion), which provides a foundation for generating more balanced summaries. In the second stage, we employ Longformer to capture rich contextual relationships across sections and generating context-aware summaries. Experiments conducted on two domain-specific scientific paper summarization datasets demonstrate that our method outperforms advanced baselines, and generates more comprehensive summaries. The code and dataset can be accessed at this https URL. 

**Abstract (ZH)**: 科学论文的抽象总结一直是研究重点，但现有方法面临两大挑战。首先，大多数总结模型依赖于编码-解码架构，将论文视为单词序列，从而未能充分捕捉科学论文中固有的结构信息。其次，现有研究通常使用关键词映射或特征工程来识别结构性信息，但这些方法难以应对科学论文的结构性灵活性，并在不同学科间缺乏鲁棒性。为应对这些挑战，我们提出了一种两阶段抽象总结框架，利用自动识别科学论文中的结构功能。在第一阶段，我们将大量科学论文的标准章节标题进行标准化，并构建一个大规模的数据集以识别结构功能，通过训练分类器自动识别关键结构组成部分（如背景、方法、结果、讨论），为生成更均衡的摘要奠定基础。在第二阶段，我们使用Longformer捕捉各章节间的丰富上下文关系并生成上下文感知的摘要。在两个特定领域的科学论文总结数据集上进行的实验表明，我们的方法优于先进基准模型，并生成了更全面的摘要。代码和数据集可在以下链接访问。 

---
# Tokenization Constraints in LLMs: A Study of Symbolic and Arithmetic Reasoning Limits 

**Title (ZH)**: LLMs中-Token化约束：符号与算术推理限制研究 

**Authors**: Xiang Zhang, Juntai Cao, Jiaqi Wei, Yiwei Xu, Chenyu You  

**Link**: [PDF](https://arxiv.org/pdf/2505.14178)  

**Abstract**: Tokenization is the first - and often underappreciated - layer of computation in language models. While Chain-of-Thought (CoT) prompting enables transformer models to approximate recurrent computation by externalizing intermediate steps, we show that the success of such reasoning is fundamentally bounded by the structure of tokenized inputs. This work presents a theoretical and empirical investigation into how tokenization schemes, particularly subword-based methods like byte-pair encoding (BPE), impede symbolic computation by merging or obscuring atomic reasoning units. We introduce the notion of Token Awareness to formalize how poor token granularity disrupts logical alignment and prevents models from generalizing symbolic procedures. Through systematic evaluation on arithmetic and symbolic tasks, we demonstrate that token structure dramatically affect reasoning performance, causing failure even with CoT, while atomically-aligned formats unlock strong generalization, allowing small models (e.g., GPT-4o-mini) to outperform larger systems (e.g., o1) in structured reasoning. Our findings reveal that symbolic reasoning ability in LLMs is not purely architectural, but deeply conditioned on token-level representations. 

**Abstract (ZH)**: 令牌化是语言模型中第一层且常常被忽视的计算层。尽管链式思维（CoT）提示能够通过外部化中间步骤来逼近循环计算，但我们表明，这种推理的成功从根本上受限于标记化输入的结构。本工作通过理论和实证研究探讨了令牌化方案，尤其是基于子词的方法（如BPE），如何通过合并或掩盖原子推理单元来阻碍符号计算。我们引入了“令牌意识”这一概念，以正式化低劣的令牌粒度如何破坏逻辑对齐并阻止模型从符号程序中泛化。通过在算术和符号任务上的系统评估，我们证明令牌结构对推理性能有重大影响，即使在使用CoT的情况下也会导致推理失败，而原子对齐的格式则能促进强大的泛化能力，使得小型模型（如GPT-4o-mini）能够在结构化推理中优于大型系统（如o1）。我们的发现揭示了LLMs的符号推理能力不仅取决于架构，还深深依赖于令牌级表示。 

---
# Prior Prompt Engineering for Reinforcement Fine-Tuning 

**Title (ZH)**: 预先精心设计的提示工程用于强化学习微调 

**Authors**: Pittawat Taveekitworachai, Potsawee Manakul, Sarana Nutanong, Kunat Pipatanakul  

**Link**: [PDF](https://arxiv.org/pdf/2505.14157)  

**Abstract**: This paper investigates prior prompt engineering (pPE) in the context of reinforcement fine-tuning (RFT), where language models (LMs) are incentivized to exhibit behaviors that maximize performance through reward signals. While existing RFT research has primarily focused on algorithms, reward shaping, and data curation, the design of the prior prompt--the instructions prepended to queries during training to elicit behaviors such as step-by-step reasoning--remains underexplored. We investigate whether different pPE approaches can guide LMs to internalize distinct behaviors after RFT. Inspired by inference-time prompt engineering (iPE), we translate five representative iPE strategies--reasoning, planning, code-based reasoning, knowledge recall, and null-example utilization--into corresponding pPE approaches. We experiment with Qwen2.5-7B using each of the pPE approaches, then evaluate performance on in-domain and out-of-domain benchmarks (e.g., AIME2024, HumanEval+, and GPQA-Diamond). Our results show that all pPE-trained models surpass their iPE-prompted counterparts, with the null-example pPE approach achieving the largest average performance gain and the highest improvement on AIME2024 and GPQA-Diamond, surpassing the commonly used reasoning approach. Furthermore, by adapting a behavior-classification framework, we demonstrate that different pPE strategies instill distinct behavioral styles in the resulting models. These findings position pPE as a powerful yet understudied axis for RFT. 

**Abstract (ZH)**: 本研究探讨了强化微调（RFT）背景下先验提示工程（pPE）的作用，其中语言模型（LMs）通过奖励信号被激励表现出最大化性能的行为。尽管现有的RFT研究主要集中在算法、奖励塑造和数据整理上，但在训练过程中预附的先验提示（prior prompt）的设计——该提示在查询期间被用来引发逐步推理等行为——仍处于探索阶段。我们研究了不同的pPE方法是否能在RFT后引导LMs内化不同的行为。受推理时提示工程（iPE）的启发，我们将五种代表性的iPE策略（推理、规划、基于代码的推理、知识回忆和空白示例利用）翻译成相应的pPE方法。我们使用Qwen2.5-7B进行了每种pPE方法的实验，然后在领域内和领域外基准测试（例如AIME2024、HumanEval+和GPQA-Diamond）上评估性能。结果显示，所有pPE训练的模型都超过了其对应的iPE提示模型，其中空白示例pPE方法在平均性能提升方面最大，特别是在AIME2024和GPQA-Diamond上的改进最大，超过了常用的推理方法。此外，通过适应行为分类框架，我们证明了不同的pPE策略在生成的模型中灌输了不同的行为风格。这些发现将pPE定位为RFT中一个强大但未充分研究的维度。 

---
# Unify Graph Learning with Text: Unleashing LLM Potentials for Session Search 

**Title (ZH)**: 统一图学习与文本：释放大规模语言模型在会话搜索中的潜力 

**Authors**: Songhao Wu, Quan Tu, Hong Liu, Jia Xu, Zhongyi Liu, Guannan Zhang, Ran Wang, Xiuying Chen, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.14156)  

**Abstract**: Session search involves a series of interactive queries and actions to fulfill user's complex information need. Current strategies typically prioritize sequential modeling for deep semantic understanding, overlooking the graph structure in interactions. While some approaches focus on capturing structural information, they use a generalized representation for documents, neglecting the word-level semantic modeling. In this paper, we propose Symbolic Graph Ranker (SGR), which aims to take advantage of both text-based and graph-based approaches by leveraging the power of recent Large Language Models (LLMs). Concretely, we first introduce a set of symbolic grammar rules to convert session graph into text. This allows integrating session history, interaction process, and task instruction seamlessly as inputs for the LLM. Moreover, given the natural discrepancy between LLMs pre-trained on textual corpora, and the symbolic language we produce using our graph-to-text grammar, our objective is to enhance LLMs' ability to capture graph structures within a textual format. To achieve this, we introduce a set of self-supervised symbolic learning tasks including link prediction, node content generation, and generative contrastive learning, to enable LLMs to capture the topological information from coarse-grained to fine-grained. Experiment results and comprehensive analysis on two benchmark datasets, AOL and Tiangong-ST, confirm the superiority of our approach. Our paradigm also offers a novel and effective methodology that bridges the gap between traditional search strategies and modern LLMs. 

**Abstract (ZH)**: 基于符号图排名的会话搜索方法 

---
# FlowQ: Energy-Guided Flow Policies for Offline Reinforcement Learning 

**Title (ZH)**: FlowQ：基于能量引导的离线强化学习流策略 

**Authors**: Marvin Alles, Nutan Chen, Patrick van der Smagt, Botond Cseke  

**Link**: [PDF](https://arxiv.org/pdf/2505.14139)  

**Abstract**: The use of guidance to steer sampling toward desired outcomes has been widely explored within diffusion models, especially in applications such as image and trajectory generation. However, incorporating guidance during training remains relatively underexplored. In this work, we introduce energy-guided flow matching, a novel approach that enhances the training of flow models and eliminates the need for guidance at inference time. We learn a conditional velocity field corresponding to the flow policy by approximating an energy-guided probability path as a Gaussian path. Learning guided trajectories is appealing for tasks where the target distribution is defined by a combination of data and an energy function, as in reinforcement learning. Diffusion-based policies have recently attracted attention for their expressive power and ability to capture multi-modal action distributions. Typically, these policies are optimized using weighted objectives or by back-propagating gradients through actions sampled by the policy. As an alternative, we propose FlowQ, an offline reinforcement learning algorithm based on energy-guided flow matching. Our method achieves competitive performance while the policy training time is constant in the number of flow sampling steps. 

**Abstract (ZH)**: 基于能量引导的流匹配方法在训练中的应用：一种在推断时不需引导的新型增强方法 

---
# Local Mixtures of Experts: Essentially Free Test-Time Training via Model Merging 

**Title (ZH)**: 局部专家混合模型：通过模型合并实现基本上免费的测试时训练 

**Authors**: Ryo Bertolissi, Jonas Hübotter, Ido Hakimi, Andreas Krause  

**Link**: [PDF](https://arxiv.org/pdf/2505.14136)  

**Abstract**: Mixture of expert (MoE) models are a promising approach to increasing model capacity without increasing inference cost, and are core components of many state-of-the-art language models. However, current MoE models typically use only few experts due to prohibitive training and inference cost. We propose Test-Time Model Merging (TTMM) which scales the MoE paradigm to an order of magnitude more experts and uses model merging to avoid almost any test-time overhead. We show that TTMM is an approximation of test-time training (TTT), which fine-tunes an expert model for each prediction task, i.e., prompt. TTT has recently been shown to significantly improve language models, but is computationally expensive. We find that performance of TTMM improves with more experts and approaches the performance of TTT. Moreover, we find that with a 1B parameter base model, TTMM is more than 100x faster than TTT at test-time by amortizing the cost of TTT at train-time. Thus, TTMM offers a promising cost-effective approach to scale test-time training. 

**Abstract (ZH)**: 基于测试时模型合并的专家混合模型扩展 

---
# A Methodological Framework for Measuring Spatial Labeling Similarity 

**Title (ZH)**: 空间标签相似性衡量的方法论框架 

**Authors**: Yihang Du, Jiaying Hu, Suyang Hou, Yueyang Ding, Xiaobo Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.14128)  

**Abstract**: Spatial labeling assigns labels to specific spatial locations to characterize their spatial properties and relationships, with broad applications in scientific research and practice. Measuring the similarity between two spatial labelings is essential for understanding their differences and the contributing factors, such as changes in location properties or labeling methods. An adequate and unbiased measurement of spatial labeling similarity should consider the number of matched labels (label agreement), the topology of spatial label distribution, and the heterogeneous impacts of mismatched labels. However, existing methods often fail to account for all these aspects. To address this gap, we propose a methodological framework to guide the development of methods that meet these requirements. Given two spatial labelings, the framework transforms them into graphs based on location organization, labels, and attributes (e.g., location significance). The distributions of their graph attributes are then extracted, enabling an efficient computation of distributional discrepancy to reflect the dissimilarity level between the two labelings. We further provide a concrete implementation of this framework, termed Spatial Labeling Analogy Metric (SLAM), along with an analysis of its theoretical foundation, for evaluating spatial labeling results in spatial transcriptomics (ST) \textit{as per} their similarity with ground truth labeling. Through a series of carefully designed experimental cases involving both simulated and real ST data, we demonstrate that SLAM provides a comprehensive and accurate reflection of labeling quality compared to other well-established evaluation metrics. Our code is available at this https URL. 

**Abstract (ZH)**: 空间标注将标签分配给特定的空间位置，用于表征其空间属性和关系，在科学研究和实践中具有广泛的应用。空间标注相似性的度量对于理解其差异及其影响因素（如位置属性的变化或标注方法的变化）至关重要。一种适当的、无偏的时空标注相似性度量应考虑匹配标签的数量（标签一致度）、空间标签分布的拓扑结构以及未匹配标签的异质影响。然而，现有方法往往未能涵盖所有这些方面。为解决这一问题，我们提出了一种方法学框架，以指导开发满足这些要求的方法。给定两个空间标注，该框架基于位置组织、标签和属性（如位置显著性）将它们转换为图形。然后提取这些图形属性的分布，从而通过计算分布差异来有效反映两个标注之间的不相似程度。我们进一步提供了一个具体实现该框架的方法，称为时空标注类比度量（SLAM），并对其理论基础进行了分析，以评估空间转录组学（ST）中的时空标注结果与真实标注的相似性。通过涉及模拟和真实ST数据的一系列精心设计的实验案例，我们证明了SLAM相较于其他成熟的评估指标能够提供全面而准确的标注质量反映。代码可在以下链接获取。 

---
# Contrastive Consolidation of Top-Down Modulations Achieves Sparsely Supervised Continual Learning 

**Title (ZH)**: 自上而下调制的对比 Consolidation 实现稀疏监督连续学习 

**Authors**: Viet Anh Khoa Tran, Emre Neftci, Willem. A. M. Wybo  

**Link**: [PDF](https://arxiv.org/pdf/2505.14125)  

**Abstract**: Biological brains learn continually from a stream of unlabeled data, while integrating specialized information from sparsely labeled examples without compromising their ability to generalize. Meanwhile, machine learning methods are susceptible to catastrophic forgetting in this natural learning setting, as supervised specialist fine-tuning degrades performance on the original task. We introduce task-modulated contrastive learning (TMCL), which takes inspiration from the biophysical machinery in the neocortex, using predictive coding principles to integrate top-down information continually and without supervision. We follow the idea that these principles build a view-invariant representation space, and that this can be implemented using a contrastive loss. Then, whenever labeled samples of a new class occur, new affine modulations are learned that improve separation of the new class from all others, without affecting feedforward weights. By co-opting the view-invariance learning mechanism, we then train feedforward weights to match the unmodulated representation of a data sample to its modulated counterparts. This introduces modulation invariance into the representation space, and, by also using past modulations, stabilizes it. Our experiments show improvements in both class-incremental and transfer learning over state-of-the-art unsupervised approaches, as well as over comparable supervised approaches, using as few as 1% of available labels. Taken together, our work suggests that top-down modulations play a crucial role in balancing stability and plasticity. 

**Abstract (ZH)**: 生物大脑从未标记数据流中持续学习，并从中整合稀疏标注示例的专业信息，而不损害泛化能力。同时，机器学习方法在这一自然学习环境中容易出现灾难性遗忘，因为有监督的专业微调会削弱原始任务的表现。我们引入了任务调节对比学习（TMCL），该方法受到新皮层生物物理机制的启发，利用预测编码原理持续且无监督地整合自上向下的信息。我们遵循这些原理构建了视不变表征空间的观点，并认为这可以通过对比损失来实现。每当出现新类别的标注样本时，新的仿射调节将被学习，以改善新类与所有其他类的区分，而不影响前向权重。通过利用视不变学习机制，我们训练前向权重将数据样本的未调节表示与其调节对应物匹配。这为表征空间引入了调节不变性，并通过使用过去的调节使其更加稳定。我们的实验展示了与现有无监督方法以及相近的有监督方法相比，在使用少量（仅1%）可用标签的情况下，在类别增量学习和转移学习方面的改进。我们的工作表明，自上而下的调节在平衡稳定性和可塑性方面扮演着至关重要的角色。 

---
# Collaborative Unlabeled Data Optimization 

**Title (ZH)**: 协作无标签数据优化 

**Authors**: Xinyi Shang, Peng Sun, Fengyuan Liu, Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.14117)  

**Abstract**: This paper pioneers a novel data-centric paradigm to maximize the utility of unlabeled data, tackling a critical question: How can we enhance the efficiency and sustainability of deep learning training by optimizing the data itself? We begin by identifying three key limitations in existing model-centric approaches, all rooted in a shared bottleneck: knowledge extracted from data is locked to model parameters, hindering its reusability and scalability. To this end, we propose CoOpt, a highly efficient, parallelized framework for collaborative unlabeled data optimization, thereby effectively encoding knowledge into the data itself. By distributing unlabeled data and leveraging publicly available task-agnostic models, CoOpt facilitates scalable, reusable, and sustainable training pipelines. Extensive experiments across diverse datasets and architectures demonstrate its efficacy and efficiency, achieving 13.6% and 6.8% improvements on Tiny-ImageNet and ImageNet-1K, respectively, with training speedups of $1.94 \times $ and $1.2 \times$. 

**Abstract (ZH)**: 本文开创了一种以数据为中心的新范式，以最大化未标记数据的用途，探讨了一个关键问题：通过优化数据本身如何提高深度学习训练的效率和可持续性？我们首先识别了现有模型为中心方法的三个关键限制，这些限制都源于同一个瓶颈：从数据中提取的知识被锁定在模型参数中，阻碍了其重用性和可扩展性。为此，我们提出了CoOpt，这是一种高效的并行协作未标记数据优化框架，从而有效地将知识编码到数据本身中。通过分发未标记数据并利用可用的、任务无关的模型，CoOpt促进了可扩展、可重用和可持续的训练管道。跨多种数据集和架构的广泛实验显示了其有效性和效率，分别在Tiny-ImageNet和ImageNet-1K上实现了13.6%和6.8%的改进，并且训练速度分别提高了1.94倍和1.2倍。 

---
# DiagnosisArena: Benchmarking Diagnostic Reasoning for Large Language Models 

**Title (ZH)**: DiagnosisArena: 大规模语言模型诊断推理benchmarkBenchmarking诊断推理 for 大规模语言模型 

**Authors**: Yakun Zhu, Zhongzhen Huang, Linjie Mu, Yutong Huang, Wei Nie, Shaoting Zhang, Pengfei Liu, Xiaofan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14107)  

**Abstract**: The emergence of groundbreaking large language models capable of performing complex reasoning tasks holds significant promise for addressing various scientific challenges, including those arising in complex clinical scenarios. To enable their safe and effective deployment in real-world healthcare settings, it is urgently necessary to benchmark the diagnostic capabilities of current models systematically. Given the limitations of existing medical benchmarks in evaluating advanced diagnostic reasoning, we present DiagnosisArena, a comprehensive and challenging benchmark designed to rigorously assess professional-level diagnostic competence. DiagnosisArena consists of 1,113 pairs of segmented patient cases and corresponding diagnoses, spanning 28 medical specialties, deriving from clinical case reports published in 10 top-tier medical journals. The benchmark is developed through a meticulous construction pipeline, involving multiple rounds of screening and review by both AI systems and human experts, with thorough checks conducted to prevent data leakage. Our study reveals that even the most advanced reasoning models, o3-mini, o1, and DeepSeek-R1, achieve only 45.82%, 31.09%, and 17.79% accuracy, respectively. This finding highlights a significant generalization bottleneck in current large language models when faced with clinical diagnostic reasoning challenges. Through DiagnosisArena, we aim to drive further advancements in AIs diagnostic reasoning capabilities, enabling more effective solutions for real-world clinical diagnostic challenges. We provide the benchmark and evaluation tools for further research and development this https URL. 

**Abstract (ZH)**: 大型语言模型在复杂推理任务中的涌现为解决各种科学挑战，包括临床复杂场景中的挑战，带来了重要前景。为确保其在真实世界医疗环境中的安全有效部署，迫切需要系统地 benchmarks 当前模型的诊断能力。鉴于现有医疗基准在评估高级诊断推理方面的局限性，我们提出了 DiagnosisArena，一个综合且具有挑战性的基准，旨在严格评估专业级诊断能力。DiagnosisArena 包含 1,113 个分段患者病例及其对应的诊断，涵盖 28 个医学专科，数据源自 10 本顶级医学期刊发表的临床案例报告。该基准通过细致的构建管道开发，涉及多轮由 AI 系统和human专家筛查和审查，并进行了严格的检查以防止数据泄露。我们的研究表明，即使是最先进的推理模型 o3-mini、o1 和 DeepSeek-R1，准确率分别仅为 45.82%、31.09% 和 17.79%。这一发现突显了当前大型语言模型在面对临床诊断推理挑战时的重大泛化瓶颈。通过 DiagnosisArena，我们旨在推动 AI 诊断推理能力的进一步发展，为解决真实世界临床诊断挑战提供更有效的解决方案。我们提供了该基准和评估工具，供进一步研究和开发使用：此链接。 

---
# A Personalized Conversational Benchmark: Towards Simulating Personalized Conversations 

**Title (ZH)**: 个性化对话基准：向着模拟个性化对话的方向 

**Authors**: Li Li, Peilin Cai, Ryan A. Rossi, Franck Dernoncourt, Branislav Kveton, Junda Wu, Tong Yu, Linxin Song, Tiankai Yang, Yuehan Qin, Nesreen K. Ahmed, Samyadeep Basu, Subhojyoti Mukherjee, Ruiyi Zhang, Zhengmian Hu, Bo Ni, Yuxiao Zhou, Zichao Wang, Yue Huang, Yu Wang, Xiangliang Zhang, Philip S. Yu, Xiyang Hu, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.14106)  

**Abstract**: We present PersonaConvBench, a large-scale benchmark for evaluating personalized reasoning and generation in multi-turn conversations with large language models (LLMs). Unlike existing work that focuses on either personalization or conversational structure in isolation, PersonaConvBench integrates both, offering three core tasks: sentence classification, impact regression, and user-centric text generation across ten diverse Reddit-based domains. This design enables systematic analysis of how personalized conversational context shapes LLM outputs in realistic multi-user scenarios. We benchmark several commercial and open-source LLMs under a unified prompting setup and observe that incorporating personalized history yields substantial performance improvements, including a 198 percent relative gain over the best non-conversational baseline in sentiment classification. By releasing PersonaConvBench with evaluations and code, we aim to support research on LLMs that adapt to individual styles, track long-term context, and produce contextually rich, engaging responses. 

**Abstract (ZH)**: PersonaConvBench: 一种大规模基准，用于评估大型语言模型在多轮对话中个性化推理与生成能力 

---
# AudioJailbreak: Jailbreak Attacks against End-to-End Large Audio-Language Models 

**Title (ZH)**: AudioJailbreak: 对端到端大型声音-语言模型的脱 jailbreak 攻击 

**Authors**: Guangke Chen, Fu Song, Zhe Zhao, Xiaojun Jia, Yang Liu, Yanchen Qiao, Weizhe Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14103)  

**Abstract**: Jailbreak attacks to Large audio-language models (LALMs) are studied recently, but they achieve suboptimal effectiveness, applicability, and practicability, particularly, assuming that the adversary can fully manipulate user prompts. In this work, we first conduct an extensive experiment showing that advanced text jailbreak attacks cannot be easily ported to end-to-end LALMs via text-to speech (TTS) techniques. We then propose AudioJailbreak, a novel audio jailbreak attack, featuring (1) asynchrony: the jailbreak audio does not need to align with user prompts in the time axis by crafting suffixal jailbreak audios; (2) universality: a single jailbreak perturbation is effective for different prompts by incorporating multiple prompts into perturbation generation; (3) stealthiness: the malicious intent of jailbreak audios will not raise the awareness of victims by proposing various intent concealment strategies; and (4) over-the-air robustness: the jailbreak audios remain effective when being played over the air by incorporating the reverberation distortion effect with room impulse response into the generation of the perturbations. In contrast, all prior audio jailbreak attacks cannot offer asynchrony, universality, stealthiness, or over-the-air robustness. Moreover, AudioJailbreak is also applicable to the adversary who cannot fully manipulate user prompts, thus has a much broader attack scenario. Extensive experiments with thus far the most LALMs demonstrate the high effectiveness of AudioJailbreak. We highlight that our work peeks into the security implications of audio jailbreak attacks against LALMs, and realistically fosters improving their security robustness. The implementation and audio samples are available at our website this https URL. 

**Abstract (ZH)**: 对大型音频语言模型（LALMs）的监狱突破攻击研究近期有所进展，但这些攻击的效果、适用性和实用性都存在局限性，尤其是在假设攻击者能够完全操控用户提示的情况下。在本文中，我们首先进行了一项全面的实验，表明先进的文本监狱突破攻击无法轻易通过文本到语音（TTS）技术移植到端到端的LALMs中。然后，我们提出了AudioJailbreak，这是一种新型的音频监狱突破攻击，具有以下特点：（1）异步性：监狱突破音频无需在时间轴上与用户提示对齐，可以通过构造后续的监狱突破音频实现；（2）普适性：通过将多个提示融入到扰动生成中，单个监狱突破扰动能够适用于不同的提示；（3）隐蔽性：通过提出各种意图隐匿策略，使得监狱突破音频的恶意意图不会引起受害者的注意；（4）空中鲁棒性：通过将回声畸变效应与房间冲激响应融入扰动生成中，使得监狱突破音频在空中播放时仍然有效。相比之下，之前的所有音频监狱突破攻击都无法提供异步性、普适性、隐蔽性或空中鲁棒性。此外，AudioJailbreak还适用于不能完全操控用户提示的攻击者，因此具有更广泛的应用场景。迄今为止使用最多的LALMs的广泛实验表明了AudioJailbreak的高有效性。我们强调，我们的工作揭示了音频监狱突破攻击对LALMs安全性的潜在影响，真实地促进了提高其安全性。我们的实现和音频样本可在我们的网站上获取：this https URL 

---
# Gender Trouble in Language Models: An Empirical Audit Guided by Gender Performativity Theory 

**Title (ZH)**: 语言模型中的性别麻烦：基于性别表象理论的实证审查 

**Authors**: Franziska Sofia Hafner, Ana Valdivia, Luc Rocher  

**Link**: [PDF](https://arxiv.org/pdf/2505.14080)  

**Abstract**: Language models encode and subsequently perpetuate harmful gendered stereotypes. Research has succeeded in mitigating some of these harms, e.g. by dissociating non-gendered terms such as occupations from gendered terms such as 'woman' and 'man'. This approach, however, remains superficial given that associations are only one form of prejudice through which gendered harms arise. Critical scholarship on gender, such as gender performativity theory, emphasizes how harms often arise from the construction of gender itself, such as conflating gender with biological sex. In language models, these issues could lead to the erasure of transgender and gender diverse identities and cause harms in downstream applications, from misgendering users to misdiagnosing patients based on wrong assumptions about their anatomy.
For FAccT research on gendered harms to go beyond superficial linguistic associations, we advocate for a broader definition of 'gender bias' in language models. We operationalize insights on the construction of gender through language from gender studies literature and then empirically test how 16 language models of different architectures, training datasets, and model sizes encode gender. We find that language models tend to encode gender as a binary category tied to biological sex, and that gendered terms that do not neatly fall into one of these binary categories are erased and pathologized. Finally, we show that larger models, which achieve better results on performance benchmarks, learn stronger associations between gender and sex, further reinforcing a narrow understanding of gender. Our findings lead us to call for a re-evaluation of how gendered harms in language models are defined and addressed. 

**Abstract (ZH)**: 语言模型编码并延续有害的性别刻板印象。为了超越表面的语义关联，FAccT关于性别危害的研究应扩展语言模型中“性别偏见”的定义。我们通过性别研究文献中的见解，将性别构建通过语言的认识具体化，并实证测试16种不同架构、训练数据集和模型规模的语言模型如何编码性别。我们发现，语言模型往往将性别编码为一种与生物学性别紧密相关的二元分类，并且不符合此类二元分类的性别术语被抹除和病理化。此外，我们在性能基准测试中表现更佳的大规模模型，进一步强化了对性别的狭隘理解。我们的研究结果促使我们重新评估语言模型中性别危害的定义和应对方式。 

---
# NOVA: A Benchmark for Anomaly Localization and Clinical Reasoning in Brain MRI 

**Title (ZH)**: NOVA：用于脑MRI异常定位和临床推理的基准测试 

**Authors**: Cosmin I. Bercea, Jun Li, Philipp Raffler, Evamaria O. Riedel, Lena Schmitzer, Angela Kurz, Felix Bitzer, Paula Roßmüller, Julian Canisius, Mirjam L. Beyrle, Che Liu, Wenjia Bai, Bernhard Kainz, Julia A. Schnabel, Benedikt Wiestler  

**Link**: [PDF](https://arxiv.org/pdf/2505.14064)  

**Abstract**: In many real-world applications, deployed models encounter inputs that differ from the data seen during training. Out-of-distribution detection identifies whether an input stems from an unseen distribution, while open-world recognition flags such inputs to ensure the system remains robust as ever-emerging, previously $unknown$ categories appear and must be addressed without retraining. Foundation and vision-language models are pre-trained on large and diverse datasets with the expectation of broad generalization across domains, including medical imaging. However, benchmarking these models on test sets with only a few common outlier types silently collapses the evaluation back to a closed-set problem, masking failures on rare or truly novel conditions encountered in clinical use.
We therefore present $NOVA$, a challenging, real-life $evaluation-only$ benchmark of $\sim$900 brain MRI scans that span 281 rare pathologies and heterogeneous acquisition protocols. Each case includes rich clinical narratives and double-blinded expert bounding-box annotations. Together, these enable joint assessment of anomaly localisation, visual captioning, and diagnostic reasoning. Because NOVA is never used for training, it serves as an $extreme$ stress-test of out-of-distribution generalisation: models must bridge a distribution gap both in sample appearance and in semantic space. Baseline results with leading vision-language models (GPT-4o, Gemini 2.0 Flash, and Qwen2.5-VL-72B) reveal substantial performance drops across all tasks, establishing NOVA as a rigorous testbed for advancing models that can detect, localize, and reason about truly unknown anomalies. 

**Abstract (ZH)**: NOVA：一个挑战性的实际评价基准，用于评估分布外泛化能力 

---
# Field Matters: A lightweight LLM-enhanced Method for CTR Prediction 

**Title (ZH)**: 场域 Matters: 一个轻量级的LLM增强方法用于点击率预测 

**Authors**: Yu Cui, Feng Liu, Jiawei Chen, Xingyu Lou, Changwang Zhang, Jun Wang, Yuegang Sun, Xiaohu Yang, Can Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14057)  

**Abstract**: Click-through rate (CTR) prediction is a fundamental task in modern recommender systems. In recent years, the integration of large language models (LLMs) has been shown to effectively enhance the performance of traditional CTR methods. However, existing LLM-enhanced methods often require extensive processing of detailed textual descriptions for large-scale instances or user/item entities, leading to substantial computational overhead. To address this challenge, this work introduces LLaCTR, a novel and lightweight LLM-enhanced CTR method that employs a field-level enhancement paradigm. Specifically, LLaCTR first utilizes LLMs to distill crucial and lightweight semantic knowledge from small-scale feature fields through self-supervised field-feature fine-tuning. Subsequently, it leverages this field-level semantic knowledge to enhance both feature representation and feature interactions. In our experiments, we integrate LLaCTR with six representative CTR models across four datasets, demonstrating its superior performance in terms of both effectiveness and efficiency compared to existing LLM-enhanced methods. Our code is available at this https URL. 

**Abstract (ZH)**: 点击率（CTR）预测是现代推荐系统中的一个基本任务。近年来，大语言模型（LLMs）的整合已被证明能够有效增强传统CTR方法的性能。然而，现有的LLM增强方法通常需要对大规模实例或用户/项目实体进行详细的文本描述的大量处理，导致显著的计算开销。为应对这一挑战，本文提出了LLaCTR，这是一种新颖且轻量级的LLM增强CTR方法，采用场级增强范式。具体而言，LLaCTR首先利用LLMs通过自我监督的场特征微调从小规模特征场中提炼出关键且轻量级的语义知识。随后，它利用这种场级语义知识来增强特征表示和特征交互。在我们的实验中，我们将LLaCTR与四个数据集上的六种代表性CTR模型结合，证明了其在效率和效果上均优于现有的LLM增强方法。我们的代码可在以下链接获取：this https URL。 

---
# From Unaligned to Aligned: Scaling Multilingual LLMs with Multi-Way Parallel Corpora 

**Title (ZH)**: 从未对齐到对齐：使用多向并行语料库扩展多语言LLM 

**Authors**: Yingli Shen, Wen Lai, Shuo Wang, Kangyang Luo, Alexander Fraser, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.14045)  

**Abstract**: Continued pretraining and instruction tuning on large-scale multilingual data have proven to be effective in scaling large language models (LLMs) to low-resource languages. However, the unaligned nature of such data limits its ability to effectively capture cross-lingual semantics. In contrast, multi-way parallel data, where identical content is aligned across multiple languages, provides stronger cross-lingual consistency and offers greater potential for improving multilingual performance. In this paper, we introduce a large-scale, high-quality multi-way parallel corpus, TED2025, based on TED Talks. The corpus spans 113 languages, with up to 50 languages aligned in parallel, ensuring extensive multilingual coverage. Using this dataset, we investigate best practices for leveraging multi-way parallel data to enhance LLMs, including strategies for continued pretraining, instruction tuning, and the analysis of key influencing factors. Experiments on six multilingual benchmarks show that models trained on multiway parallel data consistently outperform those trained on unaligned multilingual data. 

**Abstract (ZH)**: 大规模多语言数据的持续预训练和指令调优已被证明能有效扩展大型语言模型（LLMs）到低资源语言，但此类数据的不一致性限制了其有效捕捉跨语言语义的能力。相比之下，多向平行数据，其中相同内容在多种语言中对齐，提供了更强的跨语言一致性，为提高多语言性能提供了更大的潜力。在本文中，我们基于TED Talks推出了一个大规模、高质量的多向平行语料库TED2025，涵盖113种语言，最多50种语言平行对齐，确保了广泛的多语言覆盖。利用该数据集，我们研究了利用多向平行数据增强LLMs的最佳实践，包括持续预训练、指令调优策略以及关键影响因素的分析。在六个多语言基准上的实验表明，使用多向平行数据训练的模型始终优于使用未对齐多语言数据训练的模型。 

---
# Adaptive Cyclic Diffusion for Inference Scaling 

**Title (ZH)**: 自适应循环扩散推理扩展 

**Authors**: Gyubin Lee, Truong Nhat Nguyen Bao, Jaesik Yoon, Dongwoo Lee, Minsu Kim, Yoshua Bengio, Sungjin Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2505.14036)  

**Abstract**: Diffusion models have demonstrated strong generative capabilities across domains ranging from image synthesis to complex reasoning tasks. However, most inference-time scaling methods rely on fixed denoising schedules, limiting their ability to allocate computation based on instance difficulty or task-specific demands adaptively. We introduce the challenge of adaptive inference-time scaling-dynamically adjusting computational effort during inference-and propose Adaptive Bi-directional Cyclic Diffusion (ABCD), a flexible, search-based inference framework. ABCD refines outputs through bi-directional diffusion cycles while adaptively controlling exploration depth and termination. It comprises three components: Cyclic Diffusion Search, Automatic Exploration-Exploitation Balancing, and Adaptive Thinking Time. Experiments show that ABCD improves performance across diverse tasks while maintaining computational efficiency. 

**Abstract (ZH)**: 自适应推理时动态调整计算努力的扩散模型：Flexible, Search-Based Inference Framework for Adaptive Inference-Time Scaling 

---
# AppleGrowthVision: A large-scale stereo dataset for phenological analysis, fruit detection, and 3D reconstruction in apple orchards 

**Title (ZH)**: 苹果生长视觉：适用于苹果 orchards 阶段分析、果实检测及三维重建的大规模立体数据集 

**Authors**: Laura-Sophia von Hirschhausen, Jannes S. Magnusson, Mykyta Kovalenko, Fredrik Boye, Tanay Rawat, Peter Eisert, Anna Hilsmann, Sebastian Pretzsch, Sebastian Bosse  

**Link**: [PDF](https://arxiv.org/pdf/2505.14029)  

**Abstract**: Deep learning has transformed computer vision for precision agriculture, yet apple orchard monitoring remains limited by dataset constraints. The lack of diverse, realistic datasets and the difficulty of annotating dense, heterogeneous scenes. Existing datasets overlook different growth stages and stereo imagery, both essential for realistic 3D modeling of orchards and tasks like fruit localization, yield estimation, and structural analysis. To address these gaps, we present AppleGrowthVision, a large-scale dataset comprising two subsets. The first includes 9,317 high resolution stereo images collected from a farm in Brandenburg (Germany), covering six agriculturally validated growth stages over a full growth cycle. The second subset consists of 1,125 densely annotated images from the same farm in Brandenburg and one in Pillnitz (Germany), containing a total of 31,084 apple labels. AppleGrowthVision provides stereo-image data with agriculturally validated growth stages, enabling precise phenological analysis and 3D reconstructions. Extending MinneApple with our data improves YOLOv8 performance by 7.69 % in terms of F1-score, while adding it to MinneApple and MAD boosts Faster R-CNN F1-score by 31.06 %. Additionally, six BBCH stages were predicted with over 95 % accuracy using VGG16, ResNet152, DenseNet201, and MobileNetv2. AppleGrowthVision bridges the gap between agricultural science and computer vision, by enabling the development of robust models for fruit detection, growth modeling, and 3D analysis in precision agriculture. Future work includes improving annotation, enhancing 3D reconstruction, and extending multimodal analysis across all growth stages. 

**Abstract (ZH)**: Deep Learning已在精确农业中重塑了计算机视觉，但苹果园监测仍受限于数据集约束。缺少多样且现实的数据集以及标注密集且异质场景的难度。现有数据集忽略了不同生长阶段和立体影像，这两者对于真实的三维建模以及果实定位、产量估计和结构分析至关重要。为填补这些空白，我们提出了AppleGrowthVision，这是一个大规模数据集，包含两个子集。第一个子集包括9,317张高分辨率立体图像，从德国Brandenburg的一个农场收集，涵盖了整个生长周期的六个农学验证生长阶段。第二个子集包含来自Brandenburg和Pillnitz（德国）的同一个农场的1,125张密集标注图像，总共有31,084个苹果标签。AppleGrowthVision提供了具有农学验证生长阶段的立体图像数据，使精确的性状分析和三维重建成为可能。通过将我们的数据与MinneApple结合，YOLOv8的F1分数提高了7.69%，而添加到MinneApple和MAD中，则使Faster R-CNN的F1分数提高了31.06%。使用VGG16、ResNet152、DenseNet201和MobileNetv2，可以上述95%以上的准确率预测六种BBCH生长阶段。AppleGrowthVision弥合了农业科学与计算机视觉之间的差距，通过支持用于果实检测、生长建模和精确农业中三维分析的稳健模型的发展。未来的工作包括改进注释、增强三维重建，并在整个生长阶段扩展多模态分析。 

---
# CSAGC-IDS: A Dual-Module Deep Learning Network Intrusion Detection Model for Complex and Imbalanced Data 

**Title (ZH)**: CSAGC-IDS：一种适用于复杂和不平衡数据的双模块深度学习网络入侵检测模型 

**Authors**: Yifan Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.14027)  

**Abstract**: As computer networks proliferate, the gravity of network intrusions has escalated, emphasizing the criticality of network intrusion detection systems for safeguarding security. While deep learning models have exhibited promising results in intrusion detection, they face challenges in managing high-dimensional, complex traffic patterns and imbalanced data categories. This paper presents CSAGC-IDS, a network intrusion detection model based on deep learning techniques. CSAGC-IDS integrates SC-CGAN, a self-attention-enhanced convolutional conditional generative adversarial network that generates high-quality data to mitigate class imbalance. Furthermore, CSAGC-IDS integrates CSCA-CNN, a convolutional neural network enhanced through cost sensitive learning and channel attention mechanism, to extract features from complex traffic data for precise detection. Experiments conducted on the NSL-KDD dataset. CSAGC-IDS achieves an accuracy of 84.55% and an F1-score of 84.52% in five-class classification task, and an accuracy of 91.09% and an F1 score of 92.04% in binary classification this http URL, this paper provides an interpretability analysis of the proposed model, using SHAP and LIME to explain the decision-making mechanisms of the model. 

**Abstract (ZH)**: 随着计算机网络的普及，网络入侵的严重性不断增加，强调了网络入侵检测系统在保障安全方面的关键作用。尽管深度学习模型在入侵检测方面取得了Promising的结果，但它们在处理高维度、复杂流量模式以及不平衡的数据类别方面仍面临挑战。本文提出了一种基于深度学习技术的网络入侵检测模型CSAGC-IDS。CSAGC-IDS结合了增强注意力机制的SC-CGAN，生成高质量数据以缓解类别不平衡问题。此外，CSAGC-IDS还结合了CSCA-CNN，这是一种通过成本敏感学习和通道注意力机制增强的卷积神经网络，用于从复杂的流量数据中提取特征以实现精确检测。在NSL-KDD数据集上进行了实验。CSAGC-IDS在五类分类任务中实现了84.55%的准确率和84.52%的F1分数，在二类分类任务中实现了91.09%的准确率和92.04%的F1分数。本文还对该模型进行了可解释性分析，使用SHAP和LIME来解释模型的决策机制。 

---
# FedGraM: Defending Against Untargeted Attacks in Federated Learning via Embedding Gram Matrix 

**Title (ZH)**: FedGraM: 在嵌入格拉姆矩阵中抵御联邦学习中的非 targeted 攻击 

**Authors**: Di Wu, Qian Li, Heng Yang, Yong Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.14024)  

**Abstract**: Federated Learning (FL) enables geographically distributed clients to collaboratively train machine learning models by sharing only their local models, ensuring data privacy. However, FL is vulnerable to untargeted attacks that aim to degrade the global model's performance on the underlying data distribution. Existing defense mechanisms attempt to improve FL's resilience against such attacks, but their effectiveness is limited in practical FL environments due to data heterogeneity. On the contrary, we aim to detect and remove the attacks to mitigate their impact. Generalization contribution plays a crucial role in distinguishing untargeted attacks. Our observations indicate that, with limited data, the divergence between embeddings representing different classes provides a better measure of generalization than direct accuracy. In light of this, we propose a novel robust aggregation method, FedGraM, designed to defend against untargeted attacks in FL. The server maintains an auxiliary dataset containing one sample per class to support aggregation. This dataset is fed to the local models to extract embeddings. Then, the server calculates the norm of the Gram Matrix of the embeddings for each local model. The norm serves as an indicator of each model's inter-class separation capability in the embedding space. FedGraM identifies and removes potentially malicious models by filtering out those with the largest norms, then averages the remaining local models to form the global model. We conduct extensive experiments to evaluate the performance of FedGraM. Our empirical results show that with limited data samples used to construct the auxiliary dataset, FedGraM achieves exceptional performance, outperforming state-of-the-art defense methods. 

**Abstract (ZH)**: 联邦学习（FL）使得地理上分布的客户端能够通过共享其本地模型来协作训练机器学习模型，从而确保数据隐私。然而，FL容易受到针对基础数据分布性能进行降级的非靶向攻击。现有的防护机制试图提高FL对抗此类攻击的能力，但在实际的FL环境中，由于数据异质性，其效果有限。相反，我们旨在检测并移除这些攻击以减轻其影响。泛化贡献在区分非靶向攻击中起着关键作用。我们的观察表明，即使在有限数据的情况下，不同类别的嵌入表示之间的差异提供了比直接准确率更好的泛化度量。基于此，我们提出了一种新的稳健聚合方法FedGraM，旨在防御FL中的非靶向攻击。服务器维护一个辅助数据集，其中包含每个类别的一个样本，以支持聚合。将该数据集提供给局部模型以提取嵌入。然后，服务器计算每个局部模型嵌入的格拉姆矩阵的范数。范数作为模型在嵌入空间中不同类别间分离能力的指标。FedGraM通过筛选掉具有最大范数的潜在恶意模型，并平均其余局部模型来形成全局模型。我们进行了广泛的实验来评估FedGraM的性能。我们的实验证明，即使使用有限的数据样本构建辅助数据集，FedGraM也表现出色，优于最先进的防护方法。 

---
# Towards Comprehensive and Prerequisite-Free Explainer for Graph Neural Networks 

**Title (ZH)**: 面向图神经网络的全面且无需先验知识的解释器 

**Authors**: Han Zhang, Yan Wang, Guanfeng Liu, Pengfei Ding, Huaxiong Wang, Kwok-Yan Lam  

**Link**: [PDF](https://arxiv.org/pdf/2505.14005)  

**Abstract**: To enhance the reliability and credibility of graph neural networks (GNNs) and improve the transparency of their decision logic, a new field of explainability of GNNs (XGNN) has emerged. However, two major limitations severely degrade the performance and hinder the generalizability of existing XGNN methods: they (a) fail to capture the complete decision logic of GNNs across diverse distributions in the entire dataset's sample space, and (b) impose strict prerequisites on edge properties and GNN internal accessibility. To address these limitations, we propose OPEN, a novel c\textbf{O}mprehensive and \textbf{P}rerequisite-free \textbf{E}xplainer for G\textbf{N}Ns. OPEN, as the first work in the literature, can infer and partition the entire dataset's sample space into multiple environments, each containing graphs that follow a distinct distribution. OPEN further learns the decision logic of GNNs across different distributions by sampling subgraphs from each environment and analyzing their predictions, thus eliminating the need for strict prerequisites. Experimental results demonstrate that OPEN captures nearly complete decision logic of GNNs, outperforms state-of-the-art methods in fidelity while maintaining similar efficiency, and enhances robustness in real-world scenarios. 

**Abstract (ZH)**: 增强图神经网络可靠性和可信度并提升其决策逻辑透明性的新解释性方法OPEN：一种无前提的全面图神经网络解释器 

---
# Social Sycophancy: A Broader Understanding of LLM Sycophancy 

**Title (ZH)**: 社会拍马屁：对LLM拍马屁现象更广泛的理解 

**Authors**: Myra Cheng, Sunny Yu, Cinoo Lee, Pranav Khadpe, Lujain Ibrahim, Dan Jurafsky  

**Link**: [PDF](https://arxiv.org/pdf/2505.13995)  

**Abstract**: A serious risk to the safety and utility of LLMs is sycophancy, i.e., excessive agreement with and flattery of the user. Yet existing work focuses on only one aspect of sycophancy: agreement with users' explicitly stated beliefs that can be compared to a ground truth. This overlooks forms of sycophancy that arise in ambiguous contexts such as advice and support-seeking, where there is no clear ground truth, yet sycophancy can reinforce harmful implicit assumptions, beliefs, or actions. To address this gap, we introduce a richer theory of social sycophancy in LLMs, characterizing sycophancy as the excessive preservation of a user's face (the positive self-image a person seeks to maintain in an interaction). We present ELEPHANT, a framework for evaluating social sycophancy across five face-preserving behaviors (emotional validation, moral endorsement, indirect language, indirect action, and accepting framing) on two datasets: open-ended questions (OEQ) and Reddit's r/AmITheAsshole (AITA). Across eight models, we show that LLMs consistently exhibit high rates of social sycophancy: on OEQ, they preserve face 47% more than humans, and on AITA, they affirm behavior deemed inappropriate by crowdsourced human judgments in 42% of cases. We further show that social sycophancy is rewarded in preference datasets and is not easily mitigated. Our work provides theoretical grounding and empirical tools (datasets and code) for understanding and addressing this under-recognized but consequential issue. 

**Abstract (ZH)**: 一种严重威胁LLM安全性和实用性的风险是拍马屁，即过度同意和谄媚用户。现有研究仅关注拍马屁的一个方面：与用户明确表达且可以与事实真相对比的信念的一致性。这忽视了在含糊情境中出现的拍马屁形式，如寻求建议和支持，这些情境没有清晰的事实真相，但拍马屁仍能强化有害的潜在假设、信念或行为。为弥补这一缺口，我们提出了LLM中社会拍马屁的富集理论，将拍马屁定义为过度保留用户面子（互动中个人希望维持的积极自我形象）。我们介绍了ELEPHANT框架，用于评估跨五种面子保留行为（情感验证、道德肯定、间接语言、间接行动和接受框架）的社会拍马屁，在开放式问题和Reddit的r/AmITheAsshole数据集中进行。在八个模型中，我们展示了LLM在社会拍马屁方面的一贯表现：在开放式问题数据集中，它们保持面子比人类多47%；在r/AmITheAsshole数据集中，它们在42%的情况下肯定了众包人类判断认为不合适的行为。进一步研究表明，社会拍马屁在偏好数据集中受到奖励，且难以缓解。我们的工作提供了理论基础和实证工具（数据集和代码），以理解并解决这一未被充分认识但具有重要影响的问题。 

---
# When LLMs meet open-world graph learning: a new perspective for unlabeled data uncertainty 

**Title (ZH)**: 当大规模语言模型遇到开放世界图学习：无标注数据不确定性的一种新视角 

**Authors**: Yanzhe Wen, Xunkai Li, Qi Zhang, Zhu Lei, Guang Zeng, Rong-Hua Li, Guoren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13989)  

**Abstract**: Recently, large language models (LLMs) have significantly advanced text-attributed graph (TAG) learning. However, existing methods inadequately handle data uncertainty in open-world scenarios, especially concerning limited labeling and unknown-class nodes. Prior solutions typically rely on isolated semantic or structural approaches for unknown-class rejection, lacking effective annotation pipelines. To address these limitations, we propose Open-world Graph Assistant (OGA), an LLM-based framework that combines adaptive label traceability, which integrates semantics and topology for unknown-class rejection, and a graph label annotator to enable model updates using newly annotated nodes. Comprehensive experiments demonstrate OGA's effectiveness and practicality. 

**Abstract (ZH)**: 开放世界图助手：基于大语言模型的图标注辅助框架 

---
# Toward Effective Reinforcement Learning Fine-Tuning for Medical VQA in Vision-Language Models 

**Title (ZH)**: 面向医学VQA的有效强化学习微调方法研究 

**Authors**: Wenhui Zhu, Xuanzhao Dong, Xin Li, Peijie Qiu, Xiwen Chen, Abolfazl Razi, Aris Sotiras, Yi Su, Yalin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13973)  

**Abstract**: Recently, reinforcement learning (RL)-based tuning has shifted the trajectory of Multimodal Large Language Models (MLLMs), particularly following the introduction of Group Relative Policy Optimization (GRPO). However, directly applying it to medical tasks remains challenging for achieving clinically grounded model behavior. Motivated by the need to align model response with clinical expectations, we investigate four critical dimensions that affect the effectiveness of RL-based tuning in medical visual question answering (VQA): base model initialization strategy, the role of medical semantic alignment, the impact of length-based rewards on long-chain reasoning, and the influence of bias. We conduct extensive experiments to analyze these factors for medical MLLMs, providing new insights into how models are domain-specifically fine-tuned. Additionally, our results also demonstrate that GRPO-based RL tuning consistently outperforms standard supervised fine-tuning (SFT) in both accuracy and reasoning quality. 

**Abstract (ZH)**: 基于强化学习的调优最近改变了多模态大型语言模型(MLLMs)的轨迹，特别是在引入Group Relative Policy Optimization (GRPO)之后。然而，直接将其应用于医疗任务仍难以实现临床立足地的模型行为。为了使模型响应符合临床期望，我们探讨了影响基于强化学习(RL)调优在医疗视觉问答(VQA)中有效性的四个关键维度：基础模型初始化策略、医疗语义对齐的作用、基于长度的奖励对长链推理的影响以及偏见的影响。我们进行了广泛的实验来分析这些因素对医疗MLLMs的影响，为模型的领域特定微调提供了新的洞察。此外，我们的结果还表明，基于GRPO的RL调优在准确性和推理质量方面始终优于标准监督微调(SFT)。 

---
# The Multimodal Information Based Speech Processing (MISP) 2025 Challenge: Audio-Visual Diarization and Recognition 

**Title (ZH)**: 基于多模态信息的语音处理（MISP）2025挑战：音频-视觉发言者识别与分割 

**Authors**: Ming Gao, Shilong Wu, Hang Chen, Jun Du, Chin-Hui Lee, Shinji Watanabe, Jingdong Chen, Siniscalchi Sabato Marco, Odette Scharenborg  

**Link**: [PDF](https://arxiv.org/pdf/2505.13971)  

**Abstract**: Meetings are a valuable yet challenging scenario for speech applications due to complex acoustic conditions. This paper summarizes the outcomes of the MISP 2025 Challenge, hosted at Interspeech 2025, which focuses on multi-modal, multi-device meeting transcription by incorporating video modality alongside audio. The tasks include Audio-Visual Speaker Diarization (AVSD), Audio-Visual Speech Recognition (AVSR), and Audio-Visual Diarization and Recognition (AVDR). We present the challenge's objectives, tasks, dataset, baseline systems, and solutions proposed by participants. The best-performing systems achieved significant improvements over the baseline: the top AVSD model achieved a Diarization Error Rate (DER) of 8.09%, improving by 7.43%; the top AVSR system achieved a Character Error Rate (CER) of 9.48%, improving by 10.62%; and the best AVDR system achieved a concatenated minimum-permutation Character Error Rate (cpCER) of 11.56%, improving by 72.49%. 

**Abstract (ZH)**: 会议场景是语音应用中既宝贵又具挑战性的场景，由于存在复杂的声学条件。本文总结了在2025年国际语音会议(Interspeech 2025)上举行的MISP 2025挑战赛的结果，该挑战赛专注于通过结合视频模态的多模态、多设备会议转录。任务包括音频-视觉说话人分割(AVSD)、音频-视觉语音识别(AVSR)和音频-视觉分割和识别(AVDR)。本文介绍了挑战赛的目标、任务、数据集、基准系统以及参赛者提出的方法。表现最佳的系统在基准之上取得了显著改进：最佳AVSD模型的说话人分割错误率(DER)为8.09%，提高了7.43%；最佳AVSR系统的字符错误率(CER)为9.48%，提高了10.62%；最佳AVDR系统的串联最小排列字符错误率(cpCER)为11.56%，提高了72.49%。 

---
# Hypothesis on the Functional Advantages of the Selection-Broadcast Cycle Structure: Global Workspace Theory and Dealing with a Real-Time World 

**Title (ZH)**: 功能选择-广播周期结构假设：全局工作空间理论与应对实时世界 

**Authors**: Junya Nakanishi, Jun Baba, Yuichiro Yoshikawa, Hiroko Kamide, Hiroshi Ishiguro  

**Link**: [PDF](https://arxiv.org/pdf/2505.13969)  

**Abstract**: This paper discusses the functional advantages of the Selection-Broadcast Cycle structure proposed by Global Workspace Theory (GWT), inspired by human consciousness, particularly focusing on its applicability to artificial intelligence and robotics in dynamic, real-time scenarios. While previous studies often examined the Selection and Broadcast processes independently, this research emphasizes their combined cyclic structure and the resulting benefits for real-time cognitive systems. Specifically, the paper identifies three primary benefits: Dynamic Thinking Adaptation, Experience-Based Adaptation, and Immediate Real-Time Adaptation. This work highlights GWT's potential as a cognitive architecture suitable for sophisticated decision-making and adaptive performance in unsupervised, dynamic environments. It suggests new directions for the development and implementation of robust, general-purpose AI and robotics systems capable of managing complex, real-world tasks. 

**Abstract (ZH)**: 全球工作空间理论中的选择-广播周期结构的功能优势及其在动态实时场景下的人工智能和机器人应用 

---
# CAFES: A Collaborative Multi-Agent Framework for Multi-Granular Multimodal Essay Scoring 

**Title (ZH)**: CAFES：一种协作多agents框架，用于多粒度多模态作文评分 

**Authors**: Jiamin Su, Yibo Yan, Zhuoran Gao, Han Zhang, Xiang Liu, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13965)  

**Abstract**: Automated Essay Scoring (AES) is crucial for modern education, particularly with the increasing prevalence of multimodal assessments. However, traditional AES methods struggle with evaluation generalizability and multimodal perception, while even recent Multimodal Large Language Model (MLLM)-based approaches can produce hallucinated justifications and scores misaligned with human judgment. To address the limitations, we introduce CAFES, the first collaborative multi-agent framework specifically designed for AES. It orchestrates three specialized agents: an Initial Scorer for rapid, trait-specific evaluations; a Feedback Pool Manager to aggregate detailed, evidence-grounded strengths; and a Reflective Scorer that iteratively refines scores based on this feedback to enhance human alignment. Extensive experiments, using state-of-the-art MLLMs, achieve an average relative improvement of 21% in Quadratic Weighted Kappa (QWK) against ground truth, especially for grammatical and lexical diversity. Our proposed CAFES framework paves the way for an intelligent multimodal AES system. The code will be available upon acceptance. 

**Abstract (ZH)**: 自动化作文评分（AES）对于现代教育至关重要，特别是在多模态评估日益普遍的情况下。然而，传统的AES方法在评估通用性和多模态感知方面存在局限性，即使最新的基于多模态大型语言模型（MLLM）的方法也可能产生虚构的解释和与人类判断不一致的评分。为了解决这些限制，我们引入了CAFES，这是第一个专为AES设计的合作多智能体框架。它 orchestrates 三个专门化的智能体：初始评分器进行快速、针对特征的评估；反馈池管理器汇集详细、有证据支持的优点；以及反思评分器基于这些反馈迭代调整评分，以增强与人类的一致性。使用最先进的MLLM进行的广泛实验显示，与真实值相比，在Quadratic Weighted Kappa (QWK)方面的平均相对改进率为21%，尤其是在语法和词汇多样性方面。我们提出的CAFES框架为智能化多模态AES系统铺平了道路。代码将在接受后提供。 

---
# FlashThink: An Early Exit Method For Efficient Reasoning 

**Title (ZH)**: FlashThink: 一种高效的早期退出推理方法 

**Authors**: Guochao Jiang, Guofeng Quan, Zepeng Ding, Ziqin Luo, Dixuan Wang, Zheng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13949)  

**Abstract**: Large Language Models (LLMs) have shown impressive performance in reasoning tasks. However, LLMs tend to generate excessively long reasoning content, leading to significant computational overhead. Our observations indicate that even on simple problems, LLMs tend to produce unnecessarily lengthy reasoning content, which is against intuitive expectations. Preliminary experiments show that at a certain point during the generation process, the model is already capable of producing the correct solution without completing the full reasoning content. Therefore, we consider that the reasoning process of the model can be exited early to achieve the purpose of efficient reasoning. We introduce a verification model that identifies the exact moment when the model can stop reasoning and still provide the correct answer. Comprehensive experiments on four different benchmarks demonstrate that our proposed method, FlashThink, effectively shortens the reasoning content while preserving the model accuracy. For the Deepseek-R1 and QwQ-32B models, we reduced the length of reasoning content by 77.04% and 77.47%, respectively, without reducing the accuracy. 

**Abstract (ZH)**: 大规模语言模型在推理任务中表现出色，但往往会生成过长的推理内容，导致 significant 计算开销。观察表明，即使在简单问题上，模型也可能生成不必要的冗长推理内容，违背直观预期。初步实验显示，在生成过程中某个时刻，模型已经能够生成正确答案而无需完成全部推理内容。因此，我们认为可以提前终止模型的推理过程以实现高效的推理。我们引入一种验证模型，以识别模型可以停止推理且仍然提供正确答案的精确时刻。在四个不同的基准测试中，我们的方法 FlashThink 有效缩短了推理内容长度同时保持了模型准确性。对于 Deepseek-R1 和 QwQ-32B 模型，我们分别减少了推理内容长度的 77.04% 和 77.47%，而未降低准确性。 

---
# Memory-Centric Embodied Question Answer 

**Title (ZH)**: 以记忆为中心的体映射问答 

**Authors**: Mingliang Zhai, Zhi Gao, Yuwei Wu, Yunde Jia  

**Link**: [PDF](https://arxiv.org/pdf/2505.13948)  

**Abstract**: Embodied Question Answering (EQA) requires agents to autonomously explore and understand the environment to answer context-dependent questions. Existing frameworks typically center around the planner, which guides the stopping module, memory module, and answering module for reasoning. In this paper, we propose a memory-centric EQA framework named MemoryEQA. Unlike planner-centric EQA models where the memory module cannot fully interact with other modules, MemoryEQA flexible feeds memory information into all modules, thereby enhancing efficiency and accuracy in handling complex tasks, such as those involving multiple targets across different regions. Specifically, we establish a multi-modal hierarchical memory mechanism, which is divided into global memory that stores language-enhanced scene maps, and local memory that retains historical observations and state information. When performing EQA tasks, the multi-modal large language model is leveraged to convert memory information into the required input formats for injection into different modules. To evaluate EQA models' memory capabilities, we constructed the MT-HM3D dataset based on HM3D, comprising 1,587 question-answer pairs involving multiple targets across various regions, which requires agents to maintain memory of exploration-acquired target information. Experimental results on HM-EQA, MT-HM3D, and OpenEQA demonstrate the effectiveness of our framework, where a 19.8% performance gain on MT-HM3D compared to baseline model further underscores memory capability's pivotal role in resolving complex tasks. 

**Abstract (ZH)**: 基于记忆的 embodied 问答 (MemoryEQA) 

---
# MLZero: A Multi-Agent System for End-to-end Machine Learning Automation 

**Title (ZH)**: MLZero: 一种端到端机器学习自动化多agent系统 

**Authors**: Haoyang Fang, Boran Han, Nick Erickson, Xiyuan Zhang, Su Zhou, Anirudh Dagar, Jiani Zhang, Ali Caner Turkmen, Cuixiong Hu, Huzefa Rangwala, Ying Nian Wu, Bernie Wang, George Karypis  

**Link**: [PDF](https://arxiv.org/pdf/2505.13941)  

**Abstract**: Existing AutoML systems have advanced the automation of machine learning (ML); however, they still require substantial manual configuration and expert input, particularly when handling multimodal data. We introduce MLZero, a novel multi-agent framework powered by Large Language Models (LLMs) that enables end-to-end ML automation across diverse data modalities with minimal human intervention. A cognitive perception module is first employed, transforming raw multimodal inputs into perceptual context that effectively guides the subsequent workflow. To address key limitations of LLMs, such as hallucinated code generation and outdated API knowledge, we enhance the iterative code generation process with semantic and episodic memory. MLZero demonstrates superior performance on MLE-Bench Lite, outperforming all competitors in both success rate and solution quality, securing six gold medals. Additionally, when evaluated on our Multimodal AutoML Agent Benchmark, which includes 25 more challenging tasks spanning diverse data modalities, MLZero outperforms the competing methods by a large margin with a success rate of 0.92 (+263.6\%) and an average rank of 2.28. Our approach maintains its robust effectiveness even with a compact 8B LLM, outperforming full-size systems from existing solutions. 

**Abstract (ZH)**: 现有的AutoML系统已在机器学习自动化方面取得了进展；然而，它们仍然需要大量的手动配置和专家输入，特别是在处理多模态数据时。我们提出MLZero，这是一种由大型语言模型（LLMs）驱动的新型多代理框架，能够在最少的人工干预下实现跨多种数据模态的端到端机器学习自动化。首先采用认知感知模块，将原始多模态输入转换为有效的感知上下文，进而引导后续工作流程。为了应对大型语言模型的关键限制，如虚构代码生成和过时的API知识，我们通过语义和情景记忆增强了迭代代码生成过程。MLZero在MLE-Bench Lite上表现出色，成功率达到和解决方案质量均优于所有竞争者，并获得了六个金牌。此外，在我们的Multimodal AutoML Agent Benchmark中，该基准包含了25个更具挑战性的任务，涵盖多种数据模态，MLZero的成功率高达0.92（+263.6%），平均排名为2.28。即使使用紧凑的8B LLM，我们的方法仍能保持其稳健的有效性，优于现有解决方案中的全大小系统。 

---
# CLEVER: A Curated Benchmark for Formally Verified Code Generation 

**Title (ZH)**: CLEVER: 一个正式验证代码生成的精选基准 

**Authors**: Amitayush Thakur, Jasper Lee, George Tsoukalas, Meghana Sistla, Matthew Zhao, Stefan Zetzche, Greg Durrett, Yisong Yue, Swarat Chaudhuri  

**Link**: [PDF](https://arxiv.org/pdf/2505.13938)  

**Abstract**: We introduce ${\rm C{\small LEVER}}$, a high-quality, curated benchmark of 161 problems for end-to-end verified code generation in Lean. Each problem consists of (1) the task of generating a specification that matches a held-out ground-truth specification, and (2) the task of generating a Lean implementation that provably satisfies this specification. Unlike prior benchmarks, ${\rm C{\small LEVER}}$ avoids test-case supervision, LLM-generated annotations, and specifications that leak implementation logic or allow vacuous solutions. All outputs are verified post-hoc using Lean's type checker to ensure machine-checkable correctness. We use ${\rm C{\small LEVER}}$ to evaluate several few-shot and agentic approaches based on state-of-the-art language models. These methods all struggle to achieve full verification, establishing it as a challenging frontier benchmark for program synthesis and formal reasoning. Our benchmark can be found on GitHub(this https URL) as well as HuggingFace(this https URL). All our evaluation code is also available online(this https URL). 

**Abstract (ZH)**: 我们介绍${\rm C{\small LEVER}}$，这是一个高质量的手动编排基准，包含161个问题，用于Lean中的端到端验证代码生成。每个问题包括（1）生成一个与保留的真实规格相匹配的规范的任务，以及（2）生成一个Lean实现的任务，该实现可以证明满足此规范。与先前的基准不同，${\rm C{\small LEVER}}$避免了测试用例的监督、由LLM生成的注释、泄露实现逻辑的规范或允许空解的规范。所有输出都使用Lean的类型检查器进行事后验证，以确保机器可验证的正确性。我们使用${\rm C{\small LEVER}}$来评估几种基于先进语言模型的少样本和自主方法。这些方法在完全验证方面都表现不佳，将其确立为程序合成和形式推理领域的具有挑战性的前沿基准。我们的基准可以在GitHub（这个 https URL）和HuggingFace（这个 https URL）上找到。我们所有的评估代码也在网上公开（这个 https URL）。 

---
# EEG-to-Text Translation: A Model for Deciphering Human Brain Activity 

**Title (ZH)**: EEG到文本翻译：一种解码人类脑活动的模型 

**Authors**: Saydul Akbar Murad, Ashim Dahal, Nick Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2505.13936)  

**Abstract**: With the rapid advancement of large language models like Gemini, GPT, and others, bridging the gap between the human brain and language processing has become an important area of focus. To address this challenge, researchers have developed various models to decode EEG signals into text. However, these models still face significant performance limitations. To overcome these shortcomings, we propose a new model, R1 Translator, which aims to improve the performance of EEG-to-text decoding. The R1 Translator model combines a bidirectional LSTM encoder with a pretrained transformer-based decoder, utilizing EEG features to produce high-quality text outputs. The model processes EEG embeddings through the LSTM to capture sequential dependencies, which are then fed into the transformer decoder for effective text generation. The R1 Translator excels in ROUGE metrics, outperforming both T5 (previous research) and Brain Translator. Specifically, R1 achieves a ROUGE-1 score of 38.00% (P), which is up to 9% higher than T5 (34.89%) and 3% better than Brain (35.69%). It also leads in ROUGE-L, with a F1 score of 32.51%, outperforming T5 by 3% (29.67%) and Brain by 2% (30.38%). In terms of CER, R1 achieves a CER of 0.5795, which is 2% lower than T5 (0.5917) and 4% lower than Brain (0.6001). Additionally, R1 performs better in WER with a score of 0.7280, outperforming T5 by 4.3% (0.7610) and Brain by 3.6% (0.7553). Code is available at this https URL. 

**Abstract (ZH)**: 随着大型语言模型如Gemini、GPT等的快速 advancement，人类大脑与语言处理之间的差距缩小成为一个重要研究方向。为应对这一挑战，研究人员开发了多种模型将EEG信号解码为文本。然而，这些模型依然存在显著的性能限制。为了克服这些不足，我们提出了一种名为R1 Translator的新模型，旨在提高EEG-to-text解码性能。R1 Translator模型结合双向LSTM编码器和预训练的基于变换器的解码器，利用EEG特征生成高质量的文本输出。该模型通过LSTM处理EEG嵌入以捕获顺序依赖性，并将这些信息输入到变换器解码器以实现有效的文本生成。R1 Translator在ROUGE指标上表现出色，优于T5（先前研究）和Brain Translator。具体而言，R1在ROUGE-1指标上达到了38.00%（P），比T5（34.89%）高9%，比Brain（35.69%）高3%。在ROUGE-L指标上，R1的F1分数为32.51%，分别比T5（29.67%）高出3%和比Brain（30.38%）高出2%。在CER方面，R1的CER为0.5795，分别比T5（0.5917）低2%和比Brain（0.6001）低4%。在WER方面，R1得分0.7280，分别比T5（0.7610）低4.3%和比Brain（0.7553）低3.6%。代码可在此处访问。 

---
# RLVR-World: Training World Models with Reinforcement Learning 

**Title (ZH)**: RLVR-World: 使用强化学习训练世界模型 

**Authors**: Jialong Wu, Shaofeng Yin, Ningya Feng, Mingsheng Long  

**Link**: [PDF](https://arxiv.org/pdf/2505.13934)  

**Abstract**: World models predict state transitions in response to actions and are increasingly developed across diverse modalities. However, standard training objectives such as maximum likelihood estimation (MLE) often misalign with task-specific goals of world models, i.e., transition prediction metrics like accuracy or perceptual quality. In this paper, we present RLVR-World, a unified framework that leverages reinforcement learning with verifiable rewards (RLVR) to directly optimize world models for such metrics. Despite formulating world modeling as autoregressive prediction of tokenized sequences, RLVR-World evaluates metrics of decoded predictions as verifiable rewards. We demonstrate substantial performance gains on both language- and video-based world models across domains, including text games, web navigation, and robot manipulation. Our work indicates that, beyond recent advances in reasoning language models, RLVR offers a promising post-training paradigm for enhancing the utility of generative models more broadly. 

**Abstract (ZH)**: RLVR-World: 一种利用可验证奖励的强化学习框架以直接优化世界模型 

---
# APEX: Empowering LLMs with Physics-Based Task Planning for Real-time Insight 

**Title (ZH)**: APEX: 通过基于物理的任务规划赋能LLMs以实现实时洞察 

**Authors**: Wanjing Huang, Weixiang Yan, Zhen Zhang, Ambuj Singh  

**Link**: [PDF](https://arxiv.org/pdf/2505.13921)  

**Abstract**: Large Language Models (LLMs) demonstrate strong reasoning and task planning capabilities but remain fundamentally limited in physical interaction modeling. Existing approaches integrate perception via Vision-Language Models (VLMs) or adaptive decision-making through Reinforcement Learning (RL), but they fail to capture dynamic object interactions or require task-specific training, limiting their real-world applicability. We introduce APEX (Anticipatory Physics-Enhanced Execution), a framework that equips LLMs with physics-driven foresight for real-time task planning. APEX constructs structured graphs to identify and model the most relevant dynamic interactions in the environment, providing LLMs with explicit physical state updates. Simultaneously, APEX provides low-latency forward simulations of physically feasible actions, allowing LLMs to select optimal strategies based on predictive outcomes rather than static observations. We evaluate APEX on three benchmarks designed to assess perception, prediction, and decision-making: (1) Physics Reasoning Benchmark, testing causal inference and object motion prediction; (2) Tetris, evaluating whether physics-informed prediction enhances decision-making performance in long-horizon planning tasks; (3) Dynamic Obstacle Avoidance, assessing the immediate integration of perception and action feasibility analysis. APEX significantly outperforms standard LLMs and VLM-based models, demonstrating the necessity of explicit physics reasoning for bridging the gap between language-based intelligence and real-world task execution. The source code and experiment setup are publicly available at this https URL . 

**Abstract (ZH)**: 基于物理预见的大语言模型实时任务规划框架：APEX 

---
# Bronchovascular Tree-Guided Weakly Supervised Learning Method for Pulmonary Segment Segmentation 

**Title (ZH)**: 支气管血管树引导的弱监督学习方法用于肺段分割 

**Authors**: Ruijie Zhao, Zuopeng Tan, Xiao Xue, Longfei Zhao, Bing Li, Zicheng Liao, Ying Ming, Jiaru Wang, Ran Xiao, Sirong Piao, Rui Zhao, Qiqi Xu, Wei Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.13911)  

**Abstract**: Pulmonary segment segmentation is crucial for cancer localization and surgical planning. However, the pixel-wise annotation of pulmonary segments is laborious, as the boundaries between segments are indistinguishable in medical images. To this end, we propose a weakly supervised learning (WSL) method, termed Anatomy-Hierarchy Supervised Learning (AHSL), which consults the precise clinical anatomical definition of pulmonary segments to perform pulmonary segment segmentation. Since pulmonary segments reside within the lobes and are determined by the bronchovascular tree, i.e., artery, airway and vein, the design of the loss function is founded on two principles. First, segment-level labels are utilized to directly supervise the output of the pulmonary segments, ensuring that they accurately encompass the appropriate bronchovascular tree. Second, lobe-level supervision indirectly oversees the pulmonary segment, ensuring their inclusion within the corresponding lobe. Besides, we introduce a two-stage segmentation strategy that incorporates bronchovascular priori information. Furthermore, a consistency loss is proposed to enhance the smoothness of segment boundaries, along with an evaluation metric designed to measure the smoothness of pulmonary segment boundaries. Visual inspection and evaluation metrics from experiments conducted on a private dataset demonstrate the effectiveness of our method. 

**Abstract (ZH)**: 肺段分割对于癌症定位和手术规划至关重要。然而，医学图像中肺段边界难以区分，像素级标注肺段耗时费力。为此，我们提出了一种弱监督学习（WSL）方法，称为解剖层级监督学习（AHSL），该方法依据精确的临床解剖定义进行肺段分割。由于肺段位于肺叶内部，并由支气管血管树决定，即动脉、气道和静脉，损失函数的设计基于两个原则。首先，使用肺段级标签直接监督肺段输出，确保其准确包含适当的支气管血管树。其次，使用肺叶级监督间接监督肺段，确保其包含在对应的肺叶内部。此外，我们引入了一种结合支气管血管先验信息的两阶段分割策略，并提出了一致性损失以增强分割边界平滑性，同时设计了评估指标以测量肺段边界平滑性。实验结果和视觉检验表明了该方法的有效性。 

---
# XDementNET: An Explainable Attention Based Deep Convolutional Network to Detect Alzheimer Progression from MRI data 

**Title (ZH)**: XDementNET：一种用于检测MRI数据中阿尔茨海默病进展情况的可解释注意力基于深卷积网络 

**Authors**: Soyabul Islam Lincoln, Mirza Mohd Shahriar Maswood  

**Link**: [PDF](https://arxiv.org/pdf/2505.13906)  

**Abstract**: A common neurodegenerative disease, Alzheimer's disease requires a precise diagnosis and efficient treatment, particularly in light of escalating healthcare expenses and the expanding use of artificial intelligence in medical diagnostics. Many recent studies shows that the combination of brain Magnetic Resonance Imaging (MRI) and deep neural networks have achieved promising results for diagnosing AD. Using deep convolutional neural networks, this paper introduces a novel deep learning architecture that incorporates multiresidual blocks, specialized spatial attention blocks, grouped query attention, and multi-head attention. The study assessed the model's performance on four publicly accessible datasets and concentrated on identifying binary and multiclass issues across various categories. This paper also takes into account of the explainability of AD's progression and compared with state-of-the-art methods namely Gradient Class Activation Mapping (GradCAM), Score-CAM, Faster Score-CAM, and XGRADCAM. Our methodology consistently outperforms current approaches, achieving 99.66\% accuracy in 4-class classification, 99.63\% in 3-class classification, and 100\% in binary classification using Kaggle datasets. For Open Access Series of Imaging Studies (OASIS) datasets the accuracies are 99.92\%, 99.90\%, and 99.95\% respectively. The Alzheimer's Disease Neuroimaging Initiative-1 (ADNI-1) dataset was used for experiments in three planes (axial, sagittal, and coronal) and a combination of all planes. The study achieved accuracies of 99.08\% for axis, 99.85\% for sagittal, 99.5\% for coronal, and 99.17\% for all axis, and 97.79\% and 8.60\% respectively for ADNI-2. The network's ability to retrieve important information from MRI images is demonstrated by its excellent accuracy in categorizing AD stages. 

**Abstract (ZH)**: 一种常见的神经退行性疾病——阿尔茨海默病需要精确的诊断和高效的治疗，特别是在医疗费用不断上升和人工智能在医疗诊断中广泛应用的背景下。许多近期的研究表明，脑磁共振成像（MRI）与深度神经网络的结合在诊断AD方面取得了令人瞩目的成果。利用深度卷积神经网络，本文提出了一种新型深度学习架构，该架构结合了多残差块、专门的空间注意力块、分组查询注意力和多头注意力机制。研究在四个公开可用的数据集上评估了模型性能，并集中于不同类别下的二元和多元分类问题。本文还考虑了AD进展的可解释性问题，并与当前最先进的方法，如梯度类激活映射（GradCAM）、评分-CAM、快速评分-CAM和XGRADCAM进行了比较。我们的方法在Kaggle数据集上始终优于现有方法，分别实现99.66%的四分类准确率、99.63%的三分类准确率和100%的二分类准确率。对于Open Access Series of Imaging Studies（OASIS）数据集，准确率分别为99.92%、99.90%和99.95%。在Alzheimer's Disease Neuroimaging Initiative-1（ADNI-1）数据集上，在三个切面（轴位、矢状位和冠状位）及其组合条件下进行实验，准确率分别为99.08%、99.85%、99.5%和99.17%，以及8.60%的ADNI-2准确率。网络从MRI图像中提取重要信息的能力通过其在分类AD阶段方面的卓越准确率得到了展示。 

---
# Learning to Insert for Constructive Neural Vehicle Routing Solver 

**Title (ZH)**: 学习插入以构建构造性神经车辆路径求解器 

**Authors**: Fu Luo, Xi Lin, Mengyuan Zhong, Fei Liu, Zhenkun Wang, Jianyong Sun, Qingfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13904)  

**Abstract**: Neural Combinatorial Optimisation (NCO) is a promising learning-based approach for solving Vehicle Routing Problems (VRPs) without extensive manual design. While existing constructive NCO methods typically follow an appending-based paradigm that sequentially adds unvisited nodes to partial solutions, this rigid approach often leads to suboptimal results. To overcome this limitation, we explore the idea of insertion-based paradigm and propose Learning to Construct with Insertion-based Paradigm (L2C-Insert), a novel learning-based method for constructive NCO. Unlike traditional approaches, L2C-Insert builds solutions by strategically inserting unvisited nodes at any valid position in the current partial solution, which can significantly enhance the flexibility and solution quality. The proposed framework introduces three key components: a novel model architecture for precise insertion position prediction, an efficient training scheme for model optimization, and an advanced inference technique that fully exploits the insertion paradigm's flexibility. Extensive experiments on both synthetic and real-world instances of the Travelling Salesman Problem (TSP) and Capacitated Vehicle Routing Problem (CVRP) demonstrate that L2C-Insert consistently achieves superior performance across various problem sizes. 

**Abstract (ZH)**: 基于插入策略的学习建构神经组合优化（L2C-Insert）：一种用于车辆路线问题的新型学习方法 

---
# Do Language Models Use Their Depth Efficiently? 

**Title (ZH)**: 语言模型能否有效地利用其深度？ 

**Authors**: Róbert Csordás, Christopher D. Manning, Christopher Potts  

**Link**: [PDF](https://arxiv.org/pdf/2505.13898)  

**Abstract**: Modern LLMs are increasingly deep, and depth correlates with performance, albeit with diminishing returns. However, do these models use their depth efficiently? Do they compose more features to create higher-order computations that are impossible in shallow models, or do they merely spread the same kinds of computation out over more layers? To address these questions, we analyze the residual stream of the Llama 3.1 and Qwen 3 family of models. We find: First, comparing the output of the sublayers to the residual stream reveals that layers in the second half contribute much less than those in the first half, with a clear phase transition between the two halves. Second, skipping layers in the second half has a much smaller effect on future computations and output predictions. Third, for multihop tasks, we are unable to find evidence that models are using increased depth to compose subresults in examples involving many hops. Fourth, we seek to directly address whether deeper models are using their additional layers to perform new kinds of computation. To do this, we train linear maps from the residual stream of a shallow model to a deeper one. We find that layers with the same relative depth map best to each other, suggesting that the larger model simply spreads the same computations out over its many layers. All this evidence suggests that deeper models are not using their depth to learn new kinds of computation, but only using the greater depth to perform more fine-grained adjustments to the residual. This may help explain why increasing scale leads to diminishing returns for stacked Transformer architectures. 

**Abstract (ZH)**: 现代大规模语言模型越来越深，深度与性能正相关，但回报逐渐减少。然而，这些模型是否有效地利用了其深度？它们是否通过增加特征组合来创建浅层模型无法完成的高级计算，还是仅仅将相同类型的计算分布在更多的层中？为回答这些问题，我们分析了Llama 3.1和Qwen 3系列模型的残差流。我们发现：首先，将子层输出与残差流进行比较表明，第二半部分的层贡献远小于第一半部分，两半之间存在明显的分段变化。其次，去除第二半部分的层对后续计算和输出预测的影响较小。第三，对于多跳任务，我们未能找到证据表明模型利用增加的深度在涉及多跳的例子中组合子结果。第四，我们直接探讨是否更深的模型利用其额外的层执行新类型的计算。为此，我们训练了一个从浅层模型的残差流到更深模型的线性映射。我们发现具有相同相对深度的层最好地映射到彼此，这表明较大模型只是将其计算在其许多层中进行分摊。所有这些证据表明，更深的模型并未利用其深度来学习新类型的计算，而是利用更大的深度来进行残差的更细致调整。这或许可以解释为什么增加规模对堆叠的Transformer架构带来的回报逐渐减少。 

---
# Utilizing Strategic Pre-training to Reduce Overfitting: Baguan -- A Pre-trained Weather Forecasting Model 

**Title (ZH)**: 利用策略性预训练减少过拟合：Baguan —— 一个预训练天气预报模型 

**Authors**: Peisong Niu, Ziqing Ma, Tian Zhou, Weiqi Chen, Lefei Shen, Rong Jin, Liang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.13873)  

**Abstract**: Weather forecasting has long posed a significant challenge for humanity. While recent AI-based models have surpassed traditional numerical weather prediction (NWP) methods in global forecasting tasks, overfitting remains a critical issue due to the limited availability of real-world weather data spanning only a few decades. Unlike fields like computer vision or natural language processing, where data abundance can mitigate overfitting, weather forecasting demands innovative strategies to address this challenge with existing data. In this paper, we explore pre-training methods for weather forecasting, finding that selecting an appropriately challenging pre-training task introduces locality bias, effectively mitigating overfitting and enhancing performance. We introduce Baguan, a novel data-driven model for medium-range weather forecasting, built on a Siamese Autoencoder pre-trained in a self-supervised manner and fine-tuned for different lead times. Experimental results show that Baguan outperforms traditional methods, delivering more accurate forecasts. Additionally, the pre-trained Baguan demonstrates robust overfitting control and excels in downstream tasks, such as subseasonal-to-seasonal (S2S) modeling and regional forecasting, after fine-tuning. 

**Abstract (ZH)**: weather forecasting的长期挑战一直是人类面临的重要问题。尽管近年来基于AI的模型在全局预报任务中超越了传统的数值天气预报（NWP）方法，但由于可用于训练的实际天气数据仅跨越了几十年的时间，过拟合仍然是一个关键问题。与计算机视觉或自然语言处理等数据丰富的领域不同，天气预报需要创新的方法来利用现有数据解决过拟合问题。在本文中，我们探索了天气预报的预训练方法，发现选择一个合适挑战性的预训练任务引入了局部偏差，有效缓解了过拟合并提高了性能。我们提出了Baguan，这是一种基于自监督预训练的双胞胎自编码器的新型数据驱动的中短期天气预报模型，并针对不同的预测时间进行了微调。实验结果表明，Baguan 在准确度上优于传统方法，并且预训练的Baguan在下游任务如次季节至季节（S2S）建模和区域预报中表现出稳健的过拟合控制能力和优越性。 

---
# Safety2Drive: Safety-Critical Scenario Benchmark for the Evaluation of Autonomous Driving 

**Title (ZH)**: Safety2Drive: 面向自动驾驶安全评估的安全关键场景基准 

**Authors**: Jingzheng Li, Tiancheng Wang, Xingyu Peng, Jiacheng Chen, Zhijun Chen, Bing Li, Xianglong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13872)  

**Abstract**: Autonomous Driving (AD) systems demand the high levels of safety assurance. Despite significant advancements in AD demonstrated on open-source benchmarks like Longest6 and Bench2Drive, existing datasets still lack regulatory-compliant scenario libraries for closed-loop testing to comprehensively evaluate the functional safety of AD. Meanwhile, real-world AD accidents are underrepresented in current driving datasets. This scarcity leads to inadequate evaluation of AD performance, posing risks to safety validation and practical deployment. To address these challenges, we propose Safety2Drive, a safety-critical scenario library designed to evaluate AD systems. Safety2Drive offers three key contributions. (1) Safety2Drive comprehensively covers the test items required by standard regulations and contains 70 AD function test items. (2) Safety2Drive supports the safety-critical scenario generalization. It has the ability to inject safety threats such as natural environment corruptions and adversarial attacks cross camera and LiDAR sensors. (3) Safety2Drive supports multi-dimensional evaluation. In addition to the evaluation of AD systems, it also supports the evaluation of various perception tasks, such as object detection and lane detection. Safety2Drive provides a paradigm from scenario construction to validation, establishing a standardized test framework for the safe deployment of AD. 

**Abstract (ZH)**: 自主驾驶（AD）系统需要高安全性保障。尽管在如Longest6和Bench2Drive等开源基准测试中已经显示了AD的重大进展，现有的数据集在全面评估AD的功能安全性方面仍缺乏符合监管标准的闭环测试场景库。同时，当前驾驶数据集中真实的AD事故严重不足。这种不足导致对AD性能评估不足，给安全性验证和 practical 部署带来风险。为解决这些挑战，我们提出了Safety2Drive，一个用于评估AD系统的安全关键场景库。Safety2Drive 包含三项关键贡献。(1) Safety2Drive 全面覆盖了标准监管所需的所有测试项目，包含70项AD功能测试项目。(2) Safety2Drive 支持安全关键场景的泛化，能够注入安全威胁如自然环境篡改和跨摄像头与LiDAR传感器的对抗攻击。(3) Safety2Drive 支持多维度评估。除了评估AD系统之外，它还支持评估各种感知任务，如目标检测和车道检测。Safety2Drive 提供从场景构建到验证的范式，建立了一个标准化测试框架以确保AD的安全部署。 

---
# Domain Adaptation of VLM for Soccer Video Understanding 

**Title (ZH)**: 足球视频理解中VLM的领域适应 

**Authors**: Tiancheng Jiang, Henry Wang, Md Sirajus Salekin, Parmida Atighehchian, Shinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13860)  

**Abstract**: Vision Language Models (VLMs) have demonstrated strong performance in multi-modal tasks by effectively aligning visual and textual representations. However, most video understanding VLM research has been domain-agnostic, leaving the understanding of their transfer learning capability to specialized domains under-explored. In this work, we address this by exploring the adaptability of open-source VLMs to specific domains, and focusing on soccer as an initial case study. Our approach uses large-scale soccer datasets and LLM to create instruction-following data, and use them to iteratively fine-tune the general-domain VLM in a curriculum learning fashion (first teaching the model key soccer concepts to then question answering tasks). The final adapted model, trained using a curated dataset of 20k video clips, exhibits significant improvement in soccer-specific tasks compared to the base model, with a 37.5% relative improvement for the visual question-answering task and an accuracy improvement from 11.8% to 63.5% for the downstream soccer action classification task. 

**Abstract (ZH)**: 开源视觉语言模型在特定领域中的可适应性研究：以足球为例 

---
# Learning Spatio-Temporal Dynamics for Trajectory Recovery via Time-Aware Transformer 

**Title (ZH)**: 基于时间意识变换器的时空动态学习及其在轨迹恢复中的应用 

**Authors**: Tian Sun, Yuqi Chen, Baihua Zheng, Weiwei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.13857)  

**Abstract**: In real-world applications, GPS trajectories often suffer from low sampling rates, with large and irregular intervals between consecutive GPS points. This sparse characteristic presents challenges for their direct use in GPS-based systems. This paper addresses the task of map-constrained trajectory recovery, aiming to enhance trajectory sampling rates of GPS trajectories. Previous studies commonly adopt a sequence-to-sequence framework, where an encoder captures the trajectory patterns and a decoder reconstructs the target trajectory. Within this framework, effectively representing the road network and extracting relevant trajectory features are crucial for overall performance. Despite advancements in these models, they fail to fully leverage the complex spatio-temporal dynamics present in both the trajectory and the road network.
To overcome these limitations, we categorize the spatio-temporal dynamics of trajectory data into two distinct aspects: spatial-temporal traffic dynamics and trajectory dynamics. Furthermore, We propose TedTrajRec, a novel method for trajectory recovery. To capture spatio-temporal traffic dynamics, we introduce PD-GNN, which models periodic patterns and learns topologically aware dynamics concurrently for each road segment. For spatio-temporal trajectory dynamics, we present TedFormer, a time-aware Transformer that incorporates temporal dynamics for each GPS location by integrating closed-form neural ordinary differential equations into the attention mechanism. This allows TedFormer to effectively handle irregularly sampled data. Extensive experiments on three real-world datasets demonstrate the superior performance of TedTrajRec. The code is publicly available at this https URL. 

**Abstract (ZH)**: 基于地图约束的GPS轨迹恢复：时空动态建模与重构方法 

---
# Domain Gating Ensemble Networks for AI-Generated Text Detection 

**Title (ZH)**: 基于域门控集成网络的AI生成文本检测 

**Authors**: Arihant Tripathi, Liam Dugan, Charis Gao, Maggie Huan, Emma Jin, Peter Zhang, David Zhang, Julia Zhao, Chris Callison-Burch  

**Link**: [PDF](https://arxiv.org/pdf/2505.13855)  

**Abstract**: As state-of-the-art language models continue to improve, the need for robust detection of machine-generated text becomes increasingly critical. However, current state-of-the-art machine text detectors struggle to adapt to new unseen domains and generative models. In this paper we present DoGEN (Domain Gating Ensemble Networks), a technique that allows detectors to adapt to unseen domains by ensembling a set of domain expert detector models using weights from a domain classifier. We test DoGEN on a wide variety of domains from leading benchmarks and find that it achieves state-of-the-art performance on in-domain detection while outperforming models twice its size on out-of-domain detection. We release our code and trained models to assist in future research in domain-adaptive AI detection. 

**Abstract (ZH)**: 基于领域门控集成网络的机器生成文本检测技术 

---
# Forensic deepfake audio detection using segmental speech features 

**Title (ZH)**: 使用段落语音特征进行法医深度假音频检测 

**Authors**: Tianle Yang, Chengzhe Sun, Siwei Lyu, Phil Rose  

**Link**: [PDF](https://arxiv.org/pdf/2505.13847)  

**Abstract**: This study explores the potential of using acoustic features of segmental speech sounds to detect deepfake audio. These features are highly interpretable because of their close relationship with human articulatory processes and are expected to be more difficult for deepfake models to replicate. The results demonstrate that certain segmental features commonly used in forensic voice comparison are effective in identifying deep-fakes, whereas some global features provide little value. These findings underscore the need to approach audio deepfake detection differently for forensic voice comparison and offer a new perspective on leveraging segmental features for this purpose. 

**Abstract (ZH)**: 本研究探讨了使用段落语音声学特征检测深度伪造音频的潜力。这些特征由于与人类发音过程的密切关系而具有高度可解释性，并且预计更难以被深度伪造模型复制。研究结果表明，常用于法医语音比较的某些段落特征在识别深度伪造方面是有效的，而一些全局特征则提供little价值。这些发现强调了在法医语音比较中需采用不同的音频深度伪造检测方法，并从利用段落特征的角度提供了新的视角。 

---
# EfficientLLM: Efficiency in Large Language Models 

**Title (ZH)**: 高效大语言模型: 大规模语言模型的效率 

**Authors**: Zhengqing Yuan, Weixiang Sun, Yixin Liu, Huichi Zhou, Rong Zhou, Yiyang Li, Zheyuan Zhang, Wei Song, Yue Huang, Haolong Jia, Keerthiram Murugesan, Yu Wang, Lifang He, Jianfeng Gao, Lichao Sun, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2505.13840)  

**Abstract**: Large Language Models (LLMs) have driven significant progress, yet their growing parameter counts and context windows incur prohibitive compute, energy, and monetary costs. We introduce EfficientLLM, a novel benchmark and the first comprehensive empirical study evaluating efficiency techniques for LLMs at scale. Conducted on a production-class cluster (48xGH200, 8xH200 GPUs), our study systematically explores three key axes: (1) architecture pretraining (efficient attention variants: MQA, GQA, MLA, NSA; sparse Mixture-of-Experts (MoE)), (2) fine-tuning (parameter-efficient methods: LoRA, RSLoRA, DoRA), and (3) inference (quantization methods: int4, float16). We define six fine-grained metrics (Memory Utilization, Compute Utilization, Latency, Throughput, Energy Consumption, Compression Rate) to capture hardware saturation, latency-throughput balance, and carbon cost. Evaluating over 100 model-technique pairs (0.5B-72B parameters), we derive three core insights: (i) Efficiency involves quantifiable trade-offs: no single method is universally optimal; e.g., MoE reduces FLOPs and improves accuracy but increases VRAM by 40%, while int4 quantization cuts memory/energy by up to 3.9x at a 3-5% accuracy drop. (ii) Optima are task- and scale-dependent: MQA offers optimal memory-latency trade-offs for constrained devices, MLA achieves lowest perplexity for quality-critical tasks, and RSLoRA surpasses LoRA efficiency only beyond 14B parameters. (iii) Techniques generalize across modalities: we extend evaluations to Large Vision Models (Stable Diffusion 3.5, Wan 2.1) and Vision-Language Models (Qwen2.5-VL), confirming effective transferability. By open-sourcing datasets, evaluation pipelines, and leaderboards, EfficientLLM provides essential guidance for researchers and engineers navigating the efficiency-performance landscape of next-generation foundation models. 

**Abstract (ZH)**: EfficientLLM：大规模语言模型效率技术的全面 empirical 研究 

---
# Enhancing Robot Navigation Policies with Task-Specific Uncertainty Managements 

**Title (ZH)**: 基于任务特异性不确定性管理的机器人导航策略增强 

**Authors**: Gokul Puthumanaillam, Paulo Padrao, Jose Fuentes, Leonardo Bobadilla, Melkior Ornik  

**Link**: [PDF](https://arxiv.org/pdf/2505.13837)  

**Abstract**: Robots navigating complex environments must manage uncertainty from sensor noise, environmental changes, and incomplete information, with different tasks requiring varying levels of precision in different areas. For example, precise localization may be crucial near obstacles but less critical in open spaces. We present GUIDE (Generalized Uncertainty Integration for Decision-Making and Execution), a framework that integrates these task-specific requirements into navigation policies via Task-Specific Uncertainty Maps (TSUMs). By assigning acceptable uncertainty levels to different locations, TSUMs enable robots to adapt uncertainty management based on context. When combined with reinforcement learning, GUIDE learns policies that balance task completion and uncertainty management without extensive reward engineering. Real-world tests show significant performance gains over methods lacking task-specific uncertainty awareness. 

**Abstract (ZH)**: 机器人在复杂环境中的导航必须管理来自传感器噪声、环境变化和信息不完整性的不确定性，不同任务在不同区域需要不同程度的精度。例如，接近障碍物时精确定位可能是至关重要的，但在开阔空间中则相对不那么关键。我们提出了一种名为GUIDE（Generalized Uncertainty Integration for Decision-Making and Execution）的框架，该框架通过任务特定不确定性地图（TSUMs）将这些任务特定要求整合到导航策略中。通过为不同位置分配可接受的不确定性水平，TSUMs使机器人能够基于上下文调整不确定性管理。当与强化学习结合使用时，GUIDE能够在无需大量奖励工程的情况下学习平衡任务完成与不确定性管理的策略。实际测试表明，与缺乏任务特定不确定性意识的方法相比，GUIDE在性能上取得了显著提升。 

---
# Toward Real-World Cooperative and Competitive Soccer with Quadrupedal Robot Teams 

**Title (ZH)**: 面向 quadrupedal 机器人团队的现实世界中的协同与竞争足球研究 

**Authors**: Zhi Su, Yuman Gao, Emily Lukas, Yunfei Li, Jiaze Cai, Faris Tulbah, Fei Gao, Chao Yu, Zhongyu Li, Yi Wu, Koushil Sreenath  

**Link**: [PDF](https://arxiv.org/pdf/2505.13834)  

**Abstract**: Achieving coordinated teamwork among legged robots requires both fine-grained locomotion control and long-horizon strategic decision-making. Robot soccer offers a compelling testbed for this challenge, combining dynamic, competitive, and multi-agent interactions. In this work, we present a hierarchical multi-agent reinforcement learning (MARL) framework that enables fully autonomous and decentralized quadruped robot soccer. First, a set of highly dynamic low-level skills is trained for legged locomotion and ball manipulation, such as walking, dribbling, and kicking. On top of these, a high-level strategic planning policy is trained with Multi-Agent Proximal Policy Optimization (MAPPO) via Fictitious Self-Play (FSP). This learning framework allows agents to adapt to diverse opponent strategies and gives rise to sophisticated team behaviors, including coordinated passing, interception, and dynamic role allocation. With an extensive ablation study, the proposed learning method shows significant advantages in the cooperative and competitive multi-agent soccer game. We deploy the learned policies to real quadruped robots relying solely on onboard proprioception and decentralized localization, with the resulting system supporting autonomous robot-robot and robot-human soccer matches on indoor and outdoor soccer courts. 

**Abstract (ZH)**: 实现腿足机器人之间的协调团队合作需要精细的运动控制和长期的战略决策。机器人足球为这一挑战提供了极具吸引力的实验平台，结合了动态、竞争性和多智能体交互。在本工作中，我们提出了一个分层多智能体强化学习（MARL）框架，以实现完全自主和分布式腿足机器人足球。首先，训练了一组高度动态的低层次技能，用于腿足运动和球操作，如步行、带球和踢球。在此基础上，通过虚构自我博弈（FSP）和多智能体近端策略优化（MAPPO）训练高层次的战略规划策略。该学习框架使智能体能够适应多种对手策略，产生复杂的团队行为，包括协调传球、拦截和动态角色分配。通过广泛的消融研究，所提出的学习方法在合作性和竞争性多智能体足球游戏中显示出了显著优势。我们仅依赖于机载本体感觉和分布式定位，将所学策略部署到真实的四腿机器人上，最终系统支持室内和室外足球场上的自主机器人对机器人和机器人对人类的足球比赛。 

---
# Structured Agent Distillation for Large Language Model 

**Title (ZH)**: 结构化代理人蒸馏用于大型语言模型 

**Authors**: Jun Liu, Zhenglun Kong, Peiyan Dong, Changdi Yang, Tianqi Li, Hao Tang, Geng Yuan, Wei Niu, Wenbin Zhang, Pu Zhao, Xue Lin, Dong Huang, Yanzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13820)  

**Abstract**: Large language models (LLMs) exhibit strong capabilities as decision-making agents by interleaving reasoning and actions, as seen in ReAct-style frameworks. Yet, their practical deployment is constrained by high inference costs and large model sizes. We propose Structured Agent Distillation, a framework that compresses large LLM-based agents into smaller student models while preserving both reasoning fidelity and action consistency. Unlike standard token-level distillation, our method segments trajectories into {[REASON]} and {[ACT]} spans, applying segment-specific losses to align each component with the teacher's behavior. This structure-aware supervision enables compact agents to better replicate the teacher's decision process. Experiments on ALFWorld, HotPotQA-ReAct, and WebShop show that our approach consistently outperforms token-level and imitation learning baselines, achieving significant compression with minimal performance drop. Scaling and ablation results further highlight the importance of span-level alignment for efficient and deployable agents. 

**Abstract (ZH)**: 基于结构的代理蒸馏：一种保留推理准确性和动作一致性的同时压缩大型语言模型代理的方法 

---
# Articulatory Feature Prediction from Surface EMG during Speech Production 

**Title (ZH)**: 从说话生产过程中表面肌电预测发音特征 

**Authors**: Jihwan Lee, Kevin Huang, Kleanthis Avramidis, Simon Pistrosch, Monica Gonzalez-Machorro, Yoonjeong Lee, Björn Schuller, Louis Goldstein, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2505.13814)  

**Abstract**: We present a model for predicting articulatory features from surface electromyography (EMG) signals during speech production. The proposed model integrates convolutional layers and a Transformer block, followed by separate predictors for articulatory features. Our approach achieves a high prediction correlation of approximately 0.9 for most articulatory features. Furthermore, we demonstrate that these predicted articulatory features can be decoded into intelligible speech waveforms. To our knowledge, this is the first method to decode speech waveforms from surface EMG via articulatory features, offering a novel approach to EMG-based speech synthesis. Additionally, we analyze the relationship between EMG electrode placement and articulatory feature predictability, providing knowledge-driven insights for optimizing EMG electrode configurations. The source code and decoded speech samples are publicly available. 

**Abstract (ZH)**: 我们提出了一种基于表面电肌图（EMG）信号预测发音特征的模型。该模型集成了卷积层和Transformer块，并分别针对发音特征进行了预测。我们的方法对于大多数发音特征实现了约0.9的高预测相关性。此外，我们展示了这些预测的发音特征可以解码为可理解的声波形态。据我们所知，这是首次通过发音特征从表面EMG解码声波形态的方法，为基于EMG的语音合成提供了新型方法。此外，我们分析了EMG电极放置与发音特征可预测性之间的关系，提供了优化EMG电极配置的知识驱动洞察。源代码和解码的语音样本已公开。 

---
# RAG/LLM Augmented Switching Driven Polymorphic Metaheuristic Framework 

**Title (ZH)**: 基于RAG/LLM增强的切换驱动多态元启发式框架 

**Authors**: Faramarz Safi Esfahani, Ghassan Beydoun, Morteza Saberi, Brad McCusker, Biswajeet Pradhan  

**Link**: [PDF](https://arxiv.org/pdf/2505.13808)  

**Abstract**: Metaheuristic algorithms are widely used for solving complex optimization problems, yet their effectiveness is often constrained by fixed structures and the need for extensive tuning. The Polymorphic Metaheuristic Framework (PMF) addresses this limitation by introducing a self-adaptive metaheuristic switching mechanism driven by real-time performance feedback and dynamic algorithmic selection. PMF leverages the Polymorphic Metaheuristic Agent (PMA) and the Polymorphic Metaheuristic Selection Agent (PMSA) to dynamically select and transition between metaheuristic algorithms based on key performance indicators, ensuring continuous adaptation. This approach enhances convergence speed, adaptability, and solution quality, outperforming traditional metaheuristics in high-dimensional, dynamic, and multimodal environments. Experimental results on benchmark functions demonstrate that PMF significantly improves optimization efficiency by mitigating stagnation and balancing exploration-exploitation strategies across various problem landscapes. By integrating AI-driven decision-making and self-correcting mechanisms, PMF paves the way for scalable, intelligent, and autonomous optimization frameworks, with promising applications in engineering, logistics, and complex decision-making systems. 

**Abstract (ZH)**: Polymeric元启发式框架：基于实时性能反馈的自适应元启发式切换机制及其应用 

---
# ClapFM-EVC: High-Fidelity and Flexible Emotional Voice Conversion with Dual Control from Natural Language and Speech 

**Title (ZH)**: ClapFM-EVC：基于自然语言和语音双重控制的高质量和灵活情感语音转换 

**Authors**: Yu Pan, Yanni Hu, Yuguang Yang, Jixun Yao, Jianhao Ye, Hongbin Zhou, Lei Ma, Jianjun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13805)  

**Abstract**: Despite great advances, achieving high-fidelity emotional voice conversion (EVC) with flexible and interpretable control remains challenging. This paper introduces ClapFM-EVC, a novel EVC framework capable of generating high-quality converted speech driven by natural language prompts or reference speech with adjustable emotion intensity. We first propose EVC-CLAP, an emotional contrastive language-audio pre-training model, guided by natural language prompts and categorical labels, to extract and align fine-grained emotional elements across speech and text modalities. Then, a FuEncoder with an adaptive intensity gate is presented to seamless fuse emotional features with Phonetic PosteriorGrams from a pre-trained ASR model. To further improve emotion expressiveness and speech naturalness, we propose a flow matching model conditioned on these captured features to reconstruct Mel-spectrogram of source speech. Subjective and objective evaluations validate the effectiveness of ClapFM-EVC. 

**Abstract (ZH)**: 尽管取得了巨大进步，实现具有灵活可解释控制的高保真情感语音转换（EVC）仍然具有挑战性。本文介绍了一种新型EVC框架ClapFM-EVC，该框架能够根据自然语言提示或参考语音生成具有可调节情感强度的高质量转换语音。我们首先提出了由自然语言提示和类别标签引导的情感对比语言-音频预训练模型EVC-CLAP，以在语音和文本模态之间提取和对齐细粒度的情感元素。然后，提出了一个带有自适应强度门控的FuEncoder，以无缝融合情感特征与预训练ASR模型的音素后验gram。为进一步提高情感表达能力和语音自然度，我们提出了一种基于这些捕获特征的流匹配模型，以重建源语音的梅尔频谱图。主客观评估证明了ClapFM-EVC的有效性。 

---
# Interpretable Traces, Unexpected Outcomes: Investigating the Disconnect in Trace-Based Knowledge Distillation 

**Title (ZH)**: 可解释的轨迹，出人意料的结果：基于轨迹的知识蒸馏中的断层探究 

**Authors**: Siddhant Bhambri, Upasana Biswas, Subbarao Kambhampati  

**Link**: [PDF](https://arxiv.org/pdf/2505.13792)  

**Abstract**: Question Answering (QA) poses a challenging and critical problem, particularly in today's age of interactive dialogue systems such as ChatGPT, Perplexity, Microsoft Copilot, etc. where users demand both accuracy and transparency in the model's outputs. Since smaller language models (SLMs) are computationally more efficient but often under-perform compared to larger models, Knowledge Distillation (KD) methods allow for finetuning these smaller models to improve their final performance. Lately, the intermediate tokens or the so called `reasoning' traces produced by Chain-of-Thought (CoT) or by reasoning models such as DeepSeek R1 are used as a training signal for KD. However, these reasoning traces are often verbose and difficult to interpret or evaluate. In this work, we aim to address the challenge of evaluating the faithfulness of these reasoning traces and their correlation with the final performance. To this end, we employ a KD method leveraging rule-based problem decomposition. This approach allows us to break down complex queries into structured sub-problems, generating interpretable traces whose correctness can be readily evaluated, even at inference time. Specifically, we demonstrate this approach on Open Book QA, decomposing the problem into a Classification step and an Information Retrieval step, thereby simplifying trace evaluation. Our SFT experiments with correct and incorrect traces on the CoTemp QA, Microsoft Machine Reading Comprehension QA, and Facebook bAbI QA datasets reveal the striking finding that correct traces do not necessarily imply that the model outputs the correct final solution. Similarly, we find a low correlation between correct final solutions and intermediate trace correctness. These results challenge the implicit assumption behind utilizing reasoning traces for improving SLMs' final performance via KD. 

**Abstract (ZH)**: Knowledge Distillation of Question Answering: Addressing the Evaluation of Reasoning Traces and Their Correlation with Final Performance 

---
# Preference Learning with Lie Detectors can Induce Honesty or Evasion 

**Title (ZH)**: 带有谎言检测器的偏好学习可以诱发诚实或规避行为 

**Authors**: Chris Cundy, Adam Gleave  

**Link**: [PDF](https://arxiv.org/pdf/2505.13787)  

**Abstract**: As AI systems become more capable, deceptive behaviors can undermine evaluation and mislead users at deployment. Recent work has shown that lie detectors can accurately classify deceptive behavior, but they are not typically used in the training pipeline due to concerns around contamination and objective hacking. We examine these concerns by incorporating a lie detector into the labelling step of LLM post-training and evaluating whether the learned policy is genuinely more honest, or instead learns to fool the lie detector while remaining deceptive. Using DolusChat, a novel 65k-example dataset with paired truthful/deceptive responses, we identify three key factors that determine the honesty of learned policies: amount of exploration during preference learning, lie detector accuracy, and KL regularization strength. We find that preference learning with lie detectors and GRPO can lead to policies which evade lie detectors, with deception rates of over 85\%. However, if the lie detector true positive rate (TPR) or KL regularization is sufficiently high, GRPO learns honest policies. In contrast, off-policy algorithms (DPO) consistently lead to deception rates under 25\% for realistic TPRs. Our results illustrate a more complex picture than previously assumed: depending on the context, lie-detector-enhanced training can be a powerful tool for scalable oversight, or a counterproductive method encouraging undetectable misalignment. 

**Abstract (ZH)**: 随着AI系统的能力增强，欺诈行为可能在部署中破坏评估并误导用户。近期研究表明，谎言检测器能够准确分类欺诈行为，但由于担忧污染和客观攻击，它们通常不用于训练管道中。我们通过将谎言检测器纳入LLM训练后标注步骤，评估学习策略是否确实更加诚实，抑或学会欺骗谎言检测器并在实际中仍然具有欺骗性。使用DolusChat这一包含65,000个成对真实/欺诈反应的新颖数据集，我们确定了决定学习策略诚实性的三个关键因素：偏好学习过程中的探索量、谎言检测器的准确性以及KL正则化强度。我们发现，带有谎言检测器和GRPO的学习策略可能导致逃避谎言检测器的行为，其欺骗率为85％以上。然而，如果谎言检测器的真阳性率（TPR）或KL正则化足够高，GRPO将学习诚实策略。相比之下，离策算法（DPO）在现实的TPR下始终导致欺骗率低于25％。我们的结果揭示了一种比之前认为更为复杂的情况：根据上下文，增强谎言检测器的训练可能是一种强大的可扩展监督工具，也可能是一种反生产的方法，鼓励不可检测的不对齐。 

---
# Sat2Sound: A Unified Framework for Zero-Shot Soundscape Mapping 

**Title (ZH)**: Sat2Sound: 一种零样本声音景观映射的统一框架 

**Authors**: Subash Khanal, Srikumar Sastry, Aayush Dhakal, Adeel Ahmad, Nathan Jacobs  

**Link**: [PDF](https://arxiv.org/pdf/2505.13777)  

**Abstract**: We present Sat2Sound, a multimodal representation learning framework for soundscape mapping, designed to predict the distribution of sounds at any location on Earth. Existing methods for this task rely on satellite image and paired geotagged audio samples, which often fail to capture the diversity of sound sources at a given location. To address this limitation, we enhance existing datasets by leveraging a Vision-Language Model (VLM) to generate semantically rich soundscape descriptions for locations depicted in satellite images. Our approach incorporates contrastive learning across audio, audio captions, satellite images, and satellite image captions. We hypothesize that there is a fixed set of soundscape concepts shared across modalities. To this end, we learn a shared codebook of soundscape concepts and represent each sample as a weighted average of these concepts. Sat2Sound achieves state-of-the-art performance in cross-modal retrieval between satellite image and audio on two datasets: GeoSound and SoundingEarth. Additionally, building on Sat2Sound's ability to retrieve detailed soundscape captions, we introduce a novel application: location-based soundscape synthesis, which enables immersive acoustic experiences. Our code and models will be publicly available. 

**Abstract (ZH)**: Sat2Sound：一种用于声景制图的多模态表示学习框架 

---
# Beyond Semantics: The Unreasonable Effectiveness of Reasonless Intermediate Tokens 

**Title (ZH)**: 超越语义：无关中间标记的不可思议效果 

**Authors**: Kaya Stechly, Karthik Valmeekam, Atharva Gundawar, Vardhan Palod, Subbarao Kambhampati  

**Link**: [PDF](https://arxiv.org/pdf/2505.13775)  

**Abstract**: Recent impressive results from large reasoning models have been interpreted as a triumph of Chain of Thought (CoT), and especially of the process of training on CoTs sampled from base LLMs in order to help find new reasoning patterns. In this paper, we critically examine that interpretation by investigating how the semantics of intermediate tokens-often anthropomorphized as "thoughts" or reasoning traces and which are claimed to display behaviors like backtracking, self-verification etc.-actually influence model performance. We train transformer models on formally verifiable reasoning traces and solutions, constraining both intermediate steps and final outputs to align with those of a formal solver (in our case, A* search). By constructing a formal interpreter of the semantics of our problems and intended algorithm, we systematically evaluate not only solution accuracy but also the correctness of intermediate traces, thus allowing us to evaluate whether the latter causally influences the former. We notice that, despite significant improvements on the solution-only baseline, models trained on entirely correct traces still produce invalid reasoning traces when arriving at correct solutions. To further show that trace accuracy is only loosely connected to solution accuracy, we then train models on noisy, corrupted traces which have no relation to the specific problem each is paired with, and find that not only does performance remain largely consistent with models trained on correct data, but in some cases can improve upon it and generalize more robustly on out-of-distribution tasks. These results challenge the assumption that intermediate tokens or "Chains of Thought" induce predictable reasoning behaviors and caution against anthropomorphizing such outputs or over-interpreting them (despite their mostly correct forms) as evidence of human-like or algorithmic behaviors in language models. 

**Abstract (ZH)**: 大型推理模型 Recent impressive results from large reasoning models have been interpreted as a triumph of Chain of Thought (CoT), and especially of the process of training on CoTs sampled from base LLMs in order to help find new reasoning patterns. In this paper, we critically examine that interpretation by investigating how the semantics of intermediate tokens-often anthropomorphized as "thoughts" or reasoning traces and which are claimed to display behaviors like backtracking, self-verification etc.-actually influence model performance. We train transformer models on formally verifiable reasoning traces and solutions, constraining both intermediate steps and final outputs to align with those of a formal solver (in our case, A* search). By constructing a formal interpreter of the semantics of our problems and intended algorithm, we systematically evaluate not only solution accuracy but also the correctness of intermediate traces, thus allowing us to evaluate whether the latter causally influences the former. We notice that, despite significant improvements on the solution-only baseline, models trained on entirely correct traces still produce invalid reasoning traces when arriving at correct solutions. To further show that trace accuracy is only loosely connected to solution accuracy, we then train models on noisy, corrupted traces which have no relation to the specific problem each is paired with, and find that not only does performance remain largely consistent with models trained on correct data, but in some cases can improve upon it and generalize more robustly on out-of-distribution tasks. These results challenge the assumption that intermediate tokens or "Chains of Thought" induce predictable reasoning behaviors and caution against anthropomorphizing such outputs or over-interpreting them (despite their mostly correct forms) as evidence of human-like or algorithmic behaviors in language models.标题：

近期大型推理模型的显著成果被解读为Chain of Thought (CoT)的胜利，尤其是在通过从基础LLM中采样的CoT进行训练以帮助发现新的推理模式方面。本文通过对中间令牌 semantics 的研究，质疑这种解读，这些中间令牌常常被拟人化为“想法”或推理痕迹，声称它们表现出类似回溯、自我验证等行为，实际上如何影响模型性能。我们通过对正式可验证的推理痕迹和解决方案进行变压器模型的训练，限制中间步骤和最终输出与正式求解器（如A*搜索）的输出对齐。通过构造我们问题和意图算法的正式解释器，我们系统地评估了解决方案的准确性以及中间痕迹的正确性，从而评估后者是否因果地影响前者。尽管在仅解决方案基线方面取得了显著改进，但训练于完全正确痕迹的模型在产生正确解决方案时仍然生成无效的推理痕迹。为了进一步证明痕迹准确性与解决方案准确性之间的松散联系，我们训练模型在与特定问题无关的噪声、损坏的痕迹上，发现性能不仅与正确数据训练的模型保持一致，而且在某些情况下可以优于其并更稳健地泛化到分布外的任务。本文的结果挑战了中间令牌或“Chain of Thought”诱导可预测推理行为的假设，并警告不要将其输出拟人化或过度解读（尽管它们大多是正确的形式），将其视为语言模型中类似人类或算法行为的证据。 

---
# Advancing Software Quality: A Standards-Focused Review of LLM-Based Assurance Techniques 

**Title (ZH)**: advancing 软件质量：基于标准的大型语言模型保障技术综述 

**Authors**: Avinash Patil  

**Link**: [PDF](https://arxiv.org/pdf/2505.13766)  

**Abstract**: Software Quality Assurance (SQA) is critical for delivering reliable, secure, and efficient software products. The Software Quality Assurance Process aims to provide assurance that work products and processes comply with predefined provisions and plans. Recent advancements in Large Language Models (LLMs) present new opportunities to enhance existing SQA processes by automating tasks like requirement analysis, code review, test generation, and compliance checks. Simultaneously, established standards such as ISO/IEC 12207, ISO/IEC 25010, ISO/IEC 5055, ISO 9001/ISO/IEC 90003, CMMI, and TMM provide structured frameworks for ensuring robust quality practices. This paper surveys the intersection of LLM-based SQA methods and these recognized standards, highlighting how AI-driven solutions can augment traditional approaches while maintaining compliance and process maturity. We first review the foundational software quality standards and the technical fundamentals of LLMs in software engineering. Next, we explore various LLM-based SQA applications, including requirement validation, defect detection, test generation, and documentation maintenance. We then map these applications to key software quality frameworks, illustrating how LLMs can address specific requirements and metrics within each standard. Empirical case studies and open-source initiatives demonstrate the practical viability of these methods. At the same time, discussions on challenges (e.g., data privacy, model bias, explainability) underscore the need for deliberate governance and auditing. Finally, we propose future directions encompassing adaptive learning, privacy-focused deployments, multimodal analysis, and evolving standards for AI-driven software quality. 

**Abstract (ZH)**: 基于大型语言模型的软件质量保证方法与公认标准的交叉研究：人工智能驱动解决方案如何增强传统方法的同时保持合规性和过程成熟度 

---
# Understanding Task Representations in Neural Networks via Bayesian Ablation 

**Title (ZH)**: 通过贝叶斯消融理解神经网络中的任务表示 

**Authors**: Andrew Nam, Declan Campbell, Thomas Griffiths, Jonathan Cohen, Sarah-Jane Leslie  

**Link**: [PDF](https://arxiv.org/pdf/2505.13742)  

**Abstract**: Neural networks are powerful tools for cognitive modeling due to their flexibility and emergent properties. However, interpreting their learned representations remains challenging due to their sub-symbolic semantics. In this work, we introduce a novel probabilistic framework for interpreting latent task representations in neural networks. Inspired by Bayesian inference, our approach defines a distribution over representational units to infer their causal contributions to task performance. Using ideas from information theory, we propose a suite of tools and metrics to illuminate key model properties, including representational distributedness, manifold complexity, and polysemanticity. 

**Abstract (ZH)**: 神经网络由于其灵活性和涌现性质是认知建模的强大工具，但由于其亚符号语义，解释其学习表示仍然具有挑战性。本文引入了一种新的概率框架，用于解释神经网络中的潜任务表示。受贝叶斯推理的启发，我们的方法定义了一个表示单元的分布，以推断其对任务性能的因果贡献。借助信息理论的思想，我们提出了一系列工具和指标，以揭示模型的关键属性，包括表示的分布式性、流形的复杂性和多谓性。 

---
# Improving Compositional Generation with Diffusion Models Using Lift Scores 

**Title (ZH)**: 使用升分值提高组成性生成能力的扩散模型 

**Authors**: Chenning Yu, Sicun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13740)  

**Abstract**: We introduce a novel resampling criterion using lift scores, for improving compositional generation in diffusion models. By leveraging the lift scores, we evaluate whether generated samples align with each single condition and then compose the results to determine whether the composed prompt is satisfied. Our key insight is that lift scores can be efficiently approximated using only the original diffusion model, requiring no additional training or external modules. We develop an optimized variant that achieves relatively lower computational overhead during inference while maintaining effectiveness. Through extensive experiments, we demonstrate that lift scores significantly improved the condition alignment for compositional generation across 2D synthetic data, CLEVR position tasks, and text-to-image synthesis. Our code is available at this http URL. 

**Abstract (ZH)**: 我们提出了一种新的基于提升得分的重采样准则，以提高扩散模型中的组合生成能力。通过利用提升得分，我们评估生成样本是否与每个单独条件对齐，然后将结果组合以确定组合提示是否满足。我们的关键见解是，提升得分可以仅使用原始扩散模型高效近似，无需额外训练或外部模块。我们开发了一种优化变体，在推理时具有相对较低的计算开销，但仍保持有效性。通过广泛的实验，我们证明了提升得分显著改善了在2D合成数据、CLEVR位置任务和文本到图像合成中的条件对齐。我们的代码可在以下网址获得。 

---
# Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training 

**Title (ZH)**: 电力线：大规模语言模型预训练中权重衰减和批量大小的标度律 

**Authors**: Shane Bergsma, Nolan Dey, Gurpreet Gosal, Gavia Gray, Daria Soboleva, Joel Hestness  

**Link**: [PDF](https://arxiv.org/pdf/2505.13738)  

**Abstract**: Efficient LLM pre-training requires well-tuned hyperparameters (HPs), including learning rate {\eta} and weight decay {\lambda}. We study scaling laws for HPs: formulas for how to scale HPs as we scale model size N, dataset size D, and batch size B. Recent work suggests the AdamW timescale, B/({\eta}{\lambda}D), should remain constant across training settings, and we verify the implication that optimal {\lambda} scales linearly with B, for a fixed N,D. However, as N,D scale, we show the optimal timescale obeys a precise power law in the tokens-per-parameter ratio, D/N. This law thus provides a method to accurately predict {\lambda}opt in advance of large-scale training. We also study scaling laws for optimal batch size Bopt (the B enabling lowest loss at a given N,D) and critical batch size Bcrit (the B beyond which further data parallelism becomes ineffective). In contrast with prior work, we find both Bopt and Bcrit scale as power laws in D, independent of model size, N. Finally, we analyze how these findings inform the real-world selection of Pareto-optimal N and D under dual training time and compute objectives. 

**Abstract (ZH)**: 高效预训练大规模语言模型需要精细调整超参数（HPs），包括学习率\(\eta\)和权重衰减\(\lambda\)。我们研究HPs的标度定律：随着模型规模N、数据集规模D和批次大小B的扩展，HPs应该如何进行缩放。近期工作表明，AdamW时间尺度B/(\(\eta\lambda\)D) 应在训练设置间保持恒定，并验证了在固定N和D的情况下，最优\(\lambda\)线性缩放的推论。然而，当N和D扩大时，我们展示了最优时间尺度遵循令牌数与参数数之比D/N的精确幂律。此定律因此提供了一种方法，在大规模训练之前准确预测\(\lambda_{\text{opt}}\)。我们还研究了最优批次大小\(B_{\text{opt}}\)（在给定N和D下损失最低的B）和临界批次大小\(B_{\text{crit}}\)（在此之下进一步的数据并行变得无效）的标度定律。与先前工作不同，我们发现\(B_{\text{opt}}\)和\(B_{\text{crit}}\)都遵循D的幂律，与模型大小N无关。最后，我们分析了这些发现如何指导在双目标（训练时间和计算资源）下N和D的选择。 

---
# SayCoNav: Utilizing Large Language Models for Adaptive Collaboration in Decentralized Multi-Robot Navigation 

**Title (ZH)**: SayCoNav: 利用大型语言模型实现分散多机器人导航中的自适应协作 

**Authors**: Abhinav Rajvanshi, Pritish Sahu, Tixiao Shan, Karan Sikka, Han-Pang Chiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13729)  

**Abstract**: Adaptive collaboration is critical to a team of autonomous robots to perform complicated navigation tasks in large-scale unknown environments. An effective collaboration strategy should be determined and adapted according to each robot's skills and current status to successfully achieve the shared goal. We present SayCoNav, a new approach that leverages large language models (LLMs) for automatically generating this collaboration strategy among a team of robots. Building on the collaboration strategy, each robot uses the LLM to generate its plans and actions in a decentralized way. By sharing information to each other during navigation, each robot also continuously updates its step-by-step plans accordingly. We evaluate SayCoNav on Multi-Object Navigation (MultiON) tasks, that require the team of the robots to utilize their complementary strengths to efficiently search multiple different objects in unknown environments. By validating SayCoNav with varied team compositions and conditions against baseline methods, our experimental results show that SayCoNav can improve search efficiency by at most 44.28% through effective collaboration among heterogeneous robots. It can also dynamically adapt to the changing conditions during task execution. 

**Abstract (ZH)**: 自适应协作对于自主机器人团队在大型未知环境中的复杂导航任务至关重要。一种有效的协作策略应根据每个机器人的技能和当前状态进行确定和调整，以便成功实现共同目标。我们提出SayCoNav，一种利用大型语言模型（LLMs）自动为机器人团队生成协作策略的新方法。基于此协作策略，每台机器人使用LLM以去中心化的方式生成其计划和行动。在导航过程中，通过相互共享信息，每台机器人也会相应地不断更新其逐步计划。我们通过Multi-Object Navigation（MultiON）任务评估SayCoNav，这些任务要求机器人团队利用各自的互补优势高效搜索多个未知环境中的不同物体。通过在变化的团队组成和条件下与基准方法进行验证，实验结果表明，SayCoNav可以通过有效的异质机器人协作提高搜索效率最多44.28%，并且还可以在任务执行过程中动态适应变化的条件。 

---
# Policy-Driven World Model Adaptation for Robust Offline Model-based Reinforcement Learning 

**Title (ZH)**: 基于策略的世界模型适应性调整以实现稳健的离线模型驱动强化学习 

**Authors**: Jiayu Chen, Aravind Venugopal, Jeff Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2505.13709)  

**Abstract**: Offline reinforcement learning (RL) offers a powerful paradigm for data-driven control. Compared to model-free approaches, offline model-based RL (MBRL) explicitly learns a world model from a static dataset and uses it as a surrogate simulator, improving data efficiency and enabling potential generalization beyond the dataset support. However, most existing offline MBRL methods follow a two-stage training procedure: first learning a world model by maximizing the likelihood of the observed transitions, then optimizing a policy to maximize its expected return under the learned model. This objective mismatch results in a world model that is not necessarily optimized for effective policy learning. Moreover, we observe that policies learned via offline MBRL often lack robustness during deployment, and small adversarial noise in the environment can lead to significant performance degradation. To address these, we propose a framework that dynamically adapts the world model alongside the policy under a unified learning objective aimed at improving robustness. At the core of our method is a maximin optimization problem, which we solve by innovatively utilizing Stackelberg learning dynamics. We provide theoretical analysis to support our design and introduce computationally efficient implementations. We benchmark our algorithm on twelve noisy D4RL MuJoCo tasks and three stochastic Tokamak Control tasks, demonstrating its state-of-the-art performance. 

**Abstract (ZH)**: 离线强化学习（RL）提供了一种强大的数据驱动控制范式。与无模型方法相比，离线模型基于强化学习（MBRL）明确地从静态数据集中学习世界模型，并将其用作替代模拟器，提高数据效率并允许在数据集支持范围之外潜在地泛化。然而，现有的大多数离线MBRL方法遵循两阶段训练过程：首先通过最大化观测过渡的可能性来学习世界模型，然后在学习到的模型下优化策略以最大化其预期回报。这种目标不匹配导致学习到的世界模型不一定能有效地促进策略学习。此外，我们观察到通过离线MBRL学习的策略在部署时往往缺乏鲁棒性，环境中的小对抗噪声可能导致性能显著下降。为了应对这些问题，我们提出了一种框架，该框架在统一的学习目标下动态适应世界模型和策略，旨在提高鲁棒性。我们方法的核心是一个最大化最小优化问题，我们通过创新地利用Stackelberg学习动力学来解决这个问题。我们提供了理论分析以支持我们的设计，并引入了计算高效的实现。我们在十二个噪声D4RL MuJoCo任务和三个随机Tokamak控制任务上基准测试了我们的算法，展示了其最佳性能。 

---
# Are Large Language Models Good at Detecting Propaganda? 

**Title (ZH)**: 大型语言模型擅长检测宣传吗？ 

**Authors**: Julia Jose, Rachel Greenstadt  

**Link**: [PDF](https://arxiv.org/pdf/2505.13706)  

**Abstract**: Propagandists use rhetorical devices that rely on logical fallacies and emotional appeals to advance their agendas. Recognizing these techniques is key to making informed decisions. Recent advances in Natural Language Processing (NLP) have enabled the development of systems capable of detecting manipulative content. In this study, we look at several Large Language Models and their performance in detecting propaganda techniques in news articles. We compare the performance of these LLMs with transformer-based models. We find that, while GPT-4 demonstrates superior F1 scores (F1=0.16) compared to GPT-3.5 and Claude 3 Opus, it does not outperform a RoBERTa-CRF baseline (F1=0.67). Additionally, we find that all three LLMs outperform a MultiGranularity Network (MGN) baseline in detecting instances of one out of six propaganda techniques (name-calling), with GPT-3.5 and GPT-4 also outperforming the MGN baseline in detecting instances of appeal to fear and flag-waving. 

**Abstract (ZH)**: 宣传者利用逻辑谬误和情感诉求等修辞手法来推动他们的议程。识别这些技术对于做出知情决策至关重要。自然语言处理（NLP）的最新进展使开发检测操控性内容的系统成为可能。在本研究中，我们探讨了几种大型语言模型在检测新闻文章中的宣传技巧方面的性能，并将这些模型的性能与基于变换器的模型进行了对比。我们发现，虽然GPT-4在F1分数上（F1=0.16）优于GPT-3.5和Claude 3 Opus，但其并未超越RoBERTa-CRF基线（F1=0.67）。此外，我们发现，所有三种大型语言模型在检测六种宣传技巧中的一种（人身攻击）方面优于多粒度网络（MGN）基线，GPT-3.5和GPT-4在检测恐惧诉求和挥舞旗帜的情感诉求方面也优于MGN基线。 

---
# RL in Name Only? Analyzing the Structural Assumptions in RL post-training for LLMs 

**Title (ZH)**: 仅凭名称的RL？分析LLMs训练后基于RL的结构假设 

**Authors**: Soumya Rani Samineni, Durgesh Kalwar, Karthik Valmeekam, Kaya Stechly, Subbarao Kambhampati  

**Link**: [PDF](https://arxiv.org/pdf/2505.13697)  

**Abstract**: Reinforcement learning-based post-training of large language models (LLMs) has recently gained attention, particularly following the release of DeepSeek R1, which applied GRPO for fine-tuning. Amid the growing hype around improved reasoning abilities attributed to RL post-training, we critically examine the formulation and assumptions underlying these methods. We start by highlighting the popular structural assumptions made in modeling LLM training as a Markov Decision Process (MDP), and show how they lead to a degenerate MDP that doesn't quite need the RL/GRPO apparatus. The two critical structural assumptions include (1) making the MDP states be just a concatenation of the actions-with states becoming the context window and the actions becoming the tokens in LLMs and (2) splitting the reward of a state-action trajectory uniformly across the trajectory. Through a comprehensive analysis, we demonstrate that these simplifying assumptions make the approach effectively equivalent to an outcome-driven supervised learning. Our experiments on benchmarks including GSM8K and Countdown using Qwen-2.5 base models show that iterative supervised fine-tuning, incorporating both positive and negative samples, achieves performance comparable to GRPO-based training. We will also argue that the structural assumptions indirectly incentivize the RL to generate longer sequences of intermediate tokens-which in turn feeds into the narrative of "RL generating longer thinking traces." While RL may well be a very useful technique for improving the reasoning abilities of LLMs, our analysis shows that the simplistic structural assumptions made in modeling the underlying MDP render the popular LLM RL frameworks and their interpretations questionable. 

**Abstract (ZH)**: 基于强化学习的大型语言模型（LLMs）后训练 recently gained attention, 特别是随着 DeepSeek R1 的发布，它应用了 GRPO 进行微调。随着对通过 RL 后训练增强推理能力的改进的 hype 日益增长，我们对这些方法的理论基础和基本假设进行了批判性检视。我们首先强调在建模 LLM 训练为马尔可夫决策过程（MDP）时所采用的流行结构假设，并展示这些假设如何导致一个退化且不必使用 RL/GRPO 装置的 MDP。两个关键的结构假设包括（1）将 MDP 状态仅视为动作的连接，状态成为上下文窗口，动作成为 LLM 中的标记；（2）将状态-动作轨迹的奖励均匀分配给轨迹上的所有点。通过全面分析，我们证明这些简化假设使该方法实际等价于结果驱动的监督学习。我们在包括 GSM8K 和 Countdown 的基准测试中使用 Qwen-2.5 基模型的实验表明，迭代的监督微调，结合正样本和负样本，达到了与基于 GRPO 的训练相当的性能。我们还将论证，这些结构假设间接促进了 RL 生成更长的中间标记序列，从而强化了“RL 生成更长的思维痕迹”这一论点。虽然 RL 可能是提高 LLM 推理能力的一种非常有用的技巧，但我们的分析表明，在建模底层 MDP 时所作的简单结构假设使得流行的 LLM RL 框架及其解释变得值得怀疑。 

---
# Self-Reinforced Graph Contrastive Learning 

**Title (ZH)**: 自我强化图对比学习 

**Authors**: Chou-Ying Hsieh, Chun-Fu Jang, Cheng-En Hsieh, Qian-Hui Chen, Sy-Yen Kuo  

**Link**: [PDF](https://arxiv.org/pdf/2505.13650)  

**Abstract**: Graphs serve as versatile data structures in numerous real-world domains-including social networks, molecular biology, and knowledge graphs-by capturing intricate relational information among entities. Among graph-based learning techniques, Graph Contrastive Learning (GCL) has gained significant attention for its ability to derive robust, self-supervised graph representations through the contrasting of positive and negative sample pairs. However, a critical challenge lies in ensuring high-quality positive pairs so that the intrinsic semantic and structural properties of the original graph are preserved rather than distorted. To address this issue, we propose SRGCL (Self-Reinforced Graph Contrastive Learning), a novel framework that leverages the model's own encoder to dynamically evaluate and select high-quality positive pairs. We designed a unified positive pair generator employing multiple augmentation strategies, and a selector guided by the manifold hypothesis to maintain the underlying geometry of the latent space. By adopting a probabilistic mechanism for selecting positive pairs, SRGCL iteratively refines its assessment of pair quality as the encoder's representational power improves. Extensive experiments on diverse graph-level classification tasks demonstrate that SRGCL, as a plug-in module, consistently outperforms state-of-the-art GCL methods, underscoring its adaptability and efficacy across various domains. 

**Abstract (ZH)**: 自强化图对比学习（SRGCL）：一种动态高质正样本生成与选择的新型框架 

---
# Learning (Approximately) Equivariant Networks via Constrained Optimization 

**Title (ZH)**: 学习（大致）等变的网络通过约束优化 

**Authors**: Andrei Manolache, Luiz F.O. Chamon, Mathias Niepert  

**Link**: [PDF](https://arxiv.org/pdf/2505.13631)  

**Abstract**: Equivariant neural networks are designed to respect symmetries through their architecture, boosting generalization and sample efficiency when those symmetries are present in the data distribution. Real-world data, however, often departs from perfect symmetry because of noise, structural variation, measurement bias, or other symmetry-breaking effects. Strictly equivariant models may struggle to fit the data, while unconstrained models lack a principled way to leverage partial symmetries. Even when the data is fully symmetric, enforcing equivariance can hurt training by limiting the model to a restricted region of the parameter space. Guided by homotopy principles, where an optimization problem is solved by gradually transforming a simpler problem into a complex one, we introduce Adaptive Constrained Equivariance (ACE), a constrained optimization approach that starts with a flexible, non-equivariant model and gradually reduces its deviation from equivariance. This gradual tightening smooths training early on and settles the model at a data-driven equilibrium, balancing between equivariance and non-equivariance. Across multiple architectures and tasks, our method consistently improves performance metrics, sample efficiency, and robustness to input perturbations compared with strictly equivariant models and heuristic equivariance relaxations. 

**Abstract (ZH)**: 自校准约束不变性（ACE）：通过同调原理引导的松弛不变性方法 

---
# Direction-Aware Neural Acoustic Fields for Few-Shot Interpolation of Ambisonic Impulse Responses 

**Title (ZH)**: 面向方向的神经声场网络在环状声场冲激响应少量样本插值中的应用 

**Authors**: Christopher Ick, Gordon Wichern, Yoshiki Masuyama, François Germain, Jonathan Le Roux  

**Link**: [PDF](https://arxiv.org/pdf/2505.13617)  

**Abstract**: The characteristics of a sound field are intrinsically linked to the geometric and spatial properties of the environment surrounding a sound source and a listener. The physics of sound propagation is captured in a time-domain signal known as a room impulse response (RIR). Prior work using neural fields (NFs) has allowed learning spatially-continuous representations of RIRs from finite RIR measurements. However, previous NF-based methods have focused on monaural omnidirectional or at most binaural listeners, which does not precisely capture the directional characteristics of a real sound field at a single point. We propose a direction-aware neural field (DANF) that more explicitly incorporates the directional information by Ambisonic-format RIRs. While DANF inherently captures spatial relations between sources and listeners, we further propose a direction-aware loss. In addition, we investigate the ability of DANF to adapt to new rooms in various ways including low-rank adaptation. 

**Abstract (ZH)**: 一种考虑方向性的神经场（DANF）及其在新的声学环境中的应用 

---
# OMGPT: A Sequence Modeling Framework for Data-driven Operational Decision Making 

**Title (ZH)**: OMGPT：一种基于数据驱动的操作决策序列建模框架 

**Authors**: Hanzhao Wang, Guanting Chen, Kalyan Talluri, Xiaocheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.13580)  

**Abstract**: We build a Generative Pre-trained Transformer (GPT) model from scratch to solve sequential decision making tasks arising in contexts of operations research and management science which we call OMGPT. We first propose a general sequence modeling framework to cover several operational decision making tasks as special cases, such as dynamic pricing, inventory management, resource allocation, and queueing control. Under the framework, all these tasks can be viewed as a sequential prediction problem where the goal is to predict the optimal future action given all the historical information. Then we train a transformer-based neural network model (OMGPT) as a natural and powerful architecture for sequential modeling. This marks a paradigm shift compared to the existing methods for these OR/OM tasks in that (i) the OMGPT model can take advantage of the huge amount of pre-trained data; (ii) when tackling these problems, OMGPT does not assume any analytical model structure and enables a direct and rich mapping from the history to the future actions. Either of these two aspects, to the best of our knowledge, is not achieved by any existing method. We establish a Bayesian perspective to theoretically understand the working mechanism of the OMGPT on these tasks, which relates its performance with the pre-training task diversity and the divergence between the testing task and pre-training tasks. Numerically, we observe a surprising performance of the proposed model across all the above tasks. 

**Abstract (ZH)**: 我们从头构建了一个生成预训练变压器（GPT）模型来解决运筹学和管理科学领域中出现的序列决策任务，我们将该模型称为OMGPT。我们首先提出了一种通用的序列建模框架，涵盖了动态定价、库存管理、资源分配和排队控制等多重运营决策任务。在该框架下，所有这些任务都可以被视为一个序列预测问题，目标是在给定所有历史信息的情况下预测最优的未来行动。然后，我们训练了一个基于变换器的神经网络模型（OMGPT），作为一种自然且强大的序列建模架构。这标志着与现有这些运筹学/运营管理任务方法相比的重大转变，即（i）OMGPT模型可以利用大量的预训练数据；（ii）在解决这些问题时，OMGPT不需要假设有分析模型结构，并直接且丰富地将历史映射到未来的行动。这两方面的任意一项，据我们所知，当前任何现有方法都无法实现。我们从贝叶斯视角理论上理解OMGPT在这些任务中的工作机制，将其性能与预训练任务的多样性以及测试任务与预训练任务之间的差异联系起来。数值结果表明，所提出的模型在上述所有任务中表现出令人惊讶的性能。 

---
# Learning Wavelet-Sparse FDK for 3D Cone-Beam CT Reconstruction 

**Title (ZH)**: 学习小波稀疏FDK算法在3D锥束CT重建中的应用 

**Authors**: Yipeng Sun, Linda-Sophie Schneider, Chengze Ye, Mingxuan Gu, Siyuan Mei, Siming Bayer, Andreas Maier  

**Link**: [PDF](https://arxiv.org/pdf/2505.13579)  

**Abstract**: Cone-Beam Computed Tomography (CBCT) is essential in medical imaging, and the Feldkamp-Davis-Kress (FDK) algorithm is a popular choice for reconstruction due to its efficiency. However, FDK is susceptible to noise and artifacts. While recent deep learning methods offer improved image quality, they often increase computational complexity and lack the interpretability of traditional methods. In this paper, we introduce an enhanced FDK-based neural network that maintains the classical algorithm's interpretability by selectively integrating trainable elements into the cosine weighting and filtering stages. Recognizing the challenge of a large parameter space inherent in 3D CBCT data, we leverage wavelet transformations to create sparse representations of the cosine weights and filters. This strategic sparsification reduces the parameter count by $93.75\%$ without compromising performance, accelerates convergence, and importantly, maintains the inference computational cost equivalent to the classical FDK algorithm. Our method not only ensures volumetric consistency and boosts robustness to noise, but is also designed for straightforward integration into existing CT reconstruction pipelines. This presents a pragmatic enhancement that can benefit clinical applications, particularly in environments with computational limitations. 

**Abstract (ZH)**: 基于锥束计算断层成像的改进FDK神经网络：稀疏表示与计算效率的平衡 

---
# VocalAgent: Large Language Models for Vocal Health Diagnostics with Safety-Aware Evaluation 

**Title (ZH)**: VocalAgent：具有安全意识评估的大语言模型在声健康诊断中的应用 

**Authors**: Yubin Kim, Taehan Kim, Wonjune Kang, Eugene Park, Joonsik Yoon, Dongjae Lee, Xin Liu, Daniel McDuff, Hyeonhoon Lee, Cynthia Breazeal, Hae Won Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.13577)  

**Abstract**: Vocal health plays a crucial role in peoples' lives, significantly impacting their communicative abilities and interactions. However, despite the global prevalence of voice disorders, many lack access to convenient diagnosis and treatment. This paper introduces VocalAgent, an audio large language model (LLM) to address these challenges through vocal health diagnosis. We leverage Qwen-Audio-Chat fine-tuned on three datasets collected in-situ from hospital patients, and present a multifaceted evaluation framework encompassing a safety assessment to mitigate diagnostic biases, cross-lingual performance analysis, and modality ablation studies. VocalAgent demonstrates superior accuracy on voice disorder classification compared to state-of-the-art baselines. Its LLM-based method offers a scalable solution for broader adoption of health diagnostics, while underscoring the importance of ethical and technical validation. 

**Abstract (ZH)**: 语音健康在人们生活中发挥着 crucial 重要作用，显著影响其沟通能力和互动。然而，尽管全球范围内的 voice disorders 频率很高，许多人的诊断和治疗仍不够便利。本文介绍了 VocalAgent，这是一种音频大型语言模型 (LLM)，旨在通过语音健康诊断解决这些挑战。我们利用在医院患者现场收集的三个数据集上 fine-tuned 的 Qwen-Audio-Chat，提出了一种多方面的评估框架，包括安全性评估以减轻诊断偏见、跨语言性能分析以及模态消减研究。VocalAgent 在语音障碍分类上的准确性优于最先进的基线方法。其基于大型语言模型的方法为更广泛采纳健康管理诊断提供了可扩展的解决方案，同时强调了伦理和技术验证的重要性。 

---
# FreeMesh: Boosting Mesh Generation with Coordinates Merging 

**Title (ZH)**: FreeMesh: 基于坐标合并提升网格生成 

**Authors**: Jian Liu, Haohan Weng, Biwen Lei, Xianghui Yang, Zibo Zhao, Zhuo Chen, Song Guo, Tao Han, Chunchao Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.13573)  

**Abstract**: The next-coordinate prediction paradigm has emerged as the de facto standard in current auto-regressive mesh generation methods. Despite their effectiveness, there is no efficient measurement for the various tokenizers that serialize meshes into sequences. In this paper, we introduce a new metric Per-Token-Mesh-Entropy (PTME) to evaluate the existing mesh tokenizers theoretically without any training. Building upon PTME, we propose a plug-and-play tokenization technique called coordinate merging. It further improves the compression ratios of existing tokenizers by rearranging and merging the most frequent patterns of coordinates. Through experiments on various tokenization methods like MeshXL, MeshAnything V2, and Edgerunner, we further validate the performance of our method. We hope that the proposed PTME and coordinate merging can enhance the existing mesh tokenizers and guide the further development of native mesh generation. 

**Abstract (ZH)**: 下一个坐标预测范式已成为当前自回归网格生成方法的事实标准。尽管它们很有效，但目前尚无高效的度量标准来评估各种序列化的网格分词器。在本文中，我们引入了一种新的度量标准——每令牌网格熵（PTME），以在无需训练的情况下理论性地评估现有的网格分词器。基于PTME，我们提出了一种即插即用的分词技术——坐标合并，它通过重新排列和合并最频繁的坐标模式进一步提高了现有分词器的压缩比。通过在MeshXL、MeshAnything V2和Edgerunner等多种分词方法上的实验，我们进一步验证了我们方法的性能。我们希望所提出的PTME和坐标合并能够提升现有的网格分词器，并指导原生网格生成的进一步发展。 

---
# Q${}^2$Forge: Minting Competency Questions and SPARQL Queries for Question-Answering Over Knowledge Graphs 

**Title (ZH)**: Q${}^2$Forge: 创建知识图上问答能力问题和SPARQL查询 

**Authors**: Yousouf Taghzouti, Franck Michel, Tao Jiang, Louis-Félix Nothias, Fabien Gandon  

**Link**: [PDF](https://arxiv.org/pdf/2505.13572)  

**Abstract**: The SPARQL query language is the standard method to access knowledge graphs (KGs). However, formulating SPARQL queries is a significant challenge for non-expert users, and remains time-consuming for the experienced ones. Best practices recommend to document KGs with competency questions and example queries to contextualise the knowledge they contain and illustrate their potential applications. In practice, however, this is either not the case or the examples are provided in limited numbers. Large Language Models (LLMs) are being used in conversational agents and are proving to be an attractive solution with a wide range of applications, from simple question-answering about common knowledge to generating code in a targeted programming language. However, training and testing these models to produce high quality SPARQL queries from natural language questions requires substantial datasets of question-query pairs. In this paper, we present Q${}^2$Forge that addresses the challenge of generating new competency questions for a KG and corresponding SPARQL queries. It iteratively validates those queries with human feedback and LLM as a judge. Q${}^2$Forge is open source, generic, extensible and modular, meaning that the different modules of the application (CQ generation, query generation and query refinement) can be used separately, as an integrated pipeline, or replaced by alternative services. The result is a complete pipeline from competency question formulation to query evaluation, supporting the creation of reference query sets for any target KG. 

**Abstract (ZH)**: Q${}^2$Forge：生成知识图谱新胜任力问题及其SPARQL查询的迭代验证方法 

---
# Learning Dynamics of RNNs in Closed-Loop Environments 

**Title (ZH)**: RNNs在闭环环境中的学习动力学 

**Authors**: Yoav Ger, Omri Barak  

**Link**: [PDF](https://arxiv.org/pdf/2505.13567)  

**Abstract**: Recurrent neural networks (RNNs) trained on neuroscience-inspired tasks offer powerful models of brain computation. However, typical training paradigms rely on open-loop, supervised settings, whereas real-world learning unfolds in closed-loop environments. Here, we develop a mathematical theory describing the learning dynamics of linear RNNs trained in closed-loop contexts. We first demonstrate that two otherwise identical RNNs, trained in either closed- or open-loop modes, follow markedly different learning trajectories. To probe this divergence, we analytically characterize the closed-loop case, revealing distinct stages aligned with the evolution of the training loss. Specifically, we show that the learning dynamics of closed-loop RNNs, in contrast to open-loop ones, are governed by an interplay between two competing objectives: short-term policy improvement and long-term stability of the agent-environment interaction. Finally, we apply our framework to a realistic motor control task, highlighting its broader applicability. Taken together, our results underscore the importance of modeling closed-loop dynamics in a biologically plausible setting. 

**Abstract (ZH)**: 基于神经科学启发任务训练的循环神经网络提供了强大的大脑计算模型。然而，典型的训练范式依赖于开环监督设置，而现实世界的学习则发生在闭环环境中。在这里，我们发展了一种数学理论，描述了在闭环环境中训练的线性循环神经网络的学习动态。我们首先证明，两种否则完全相同的循环神经网络，在闭环或开环模式下训练时，其学习轨迹存在显著差异。为了探究这种差异，我们对闭环情况进行了理论分析，揭示了与训练损失演化阶段相一致的不同阶段。具体而言，我们展示闭环循环神经网络的学习动力学，与开环循环神经网络相比，受短期策略改善和长期环境交互稳定性的竞争目标的相互作用所支配。最后，我们将我们的框架应用于一个现实的运动控制任务，展示了其更广泛的应用潜力。我们的结果共同强调了在生物合现实的情景下建模闭环动态的重要性。 

---
# Aligning Trustworthy AI with Democracy: A Dual Taxonomy of Opportunities and Risks 

**Title (ZH)**: 将可信赖人工智能与民主相契合：机遇与风险双重分类框架 

**Authors**: Oier Mentxaka, Natalia Díaz-Rodríguez, Mark Coeckelbergh, Marcos López de Prado, Emilia Gómez, David Fernández Llorca, Enrique Herrera-Viedma, Francisco Herrera  

**Link**: [PDF](https://arxiv.org/pdf/2505.13565)  

**Abstract**: Artificial Intelligence (AI) poses both significant risks and valuable opportunities for democratic governance. This paper introduces a dual taxonomy to evaluate AI's complex relationship with democracy: the AI Risks to Democracy (AIRD) taxonomy, which identifies how AI can undermine core democratic principles such as autonomy, fairness, and trust; and the AI's Positive Contributions to Democracy (AIPD) taxonomy, which highlights AI's potential to enhance transparency, participation, efficiency, and evidence-based policymaking.
Grounded in the European Union's approach to ethical AI governance, and particularly the seven Trustworthy AI requirements proposed by the European Commission's High-Level Expert Group on AI, each identified risk is aligned with mitigation strategies based on EU regulatory and normative frameworks. Our analysis underscores the transversal importance of transparency and societal well-being across all risk categories and offers a structured lens for aligning AI systems with democratic values.
By integrating democratic theory with practical governance tools, this paper offers a normative and actionable framework to guide research, regulation, and institutional design to support trustworthy, democratic AI. It provides scholars with a conceptual foundation to evaluate the democratic implications of AI, equips policymakers with structured criteria for ethical oversight, and helps technologists align system design with democratic principles. In doing so, it bridges the gap between ethical aspirations and operational realities, laying the groundwork for more inclusive, accountable, and resilient democratic systems in the algorithmic age. 

**Abstract (ZH)**: 人工智能（AI）对民主治理既构成了显著的风险，也带来了宝贵的机会。本文引入了一种双重分类法来评估AI与民主之间的复杂关系：AI对民主的风险（AIRD）分类，识别AI如何削弱自治、公平和信任等核心民主原则；以及AI对民主的正向贡献（AIPD）分类，强调AI在增强透明度、参与度、效率和基于证据的政策制定方面的潜力。 

---
# Breaking the Compression Ceiling: Data-Free Pipeline for Ultra-Efficient Delta Compression 

**Title (ZH)**: 突破压缩上限：无数据管道实现超高效增量压缩 

**Authors**: Xiaohui Wang, Peng Ye, Chenyu Huang, Shenghe Zheng, Bo Zhang, Wanli Ouyang, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.13563)  

**Abstract**: With the rise of the fine-tuned--pretrained paradigm, storing numerous fine-tuned models for multi-tasking creates significant storage overhead. Delta compression alleviates this by storing only the pretrained model and the highly compressed delta weights (the differences between fine-tuned and pretrained model weights). However, existing methods fail to maintain both high compression and performance, and often rely on data. To address these challenges, we propose UltraDelta, the first data-free delta compression pipeline that achieves both ultra-high compression and strong performance. UltraDelta is designed to minimize redundancy, maximize information, and stabilize performance across inter-layer, intra-layer, and global dimensions, using three key components: (1) Variance-Based Mixed Sparsity Allocation assigns sparsity based on variance, giving lower sparsity to high-variance layers to preserve inter-layer information. (2) Distribution-Aware Compression applies uniform quantization and then groups parameters by value, followed by group-wise pruning, to better preserve intra-layer distribution. (3) Trace-Norm-Guided Rescaling uses the trace norm of delta weights to estimate a global rescaling factor, improving model stability under higher compression. Extensive experiments across (a) large language models (fine-tuned on LLaMA-2 7B and 13B) with up to 133x, (b) general NLP models (RoBERTa-base, T5-base) with up to 800x, (c) vision models (ViT-B/32, ViT-L/14) with up to 400x, and (d) multi-modal models (BEiT-3) with 40x compression ratio, demonstrate that UltraDelta consistently outperforms existing methods, especially under ultra-high compression. 

**Abstract (ZH)**: 基于无数据的超高效delta压缩管道UltraDelta：实现超高效压缩与强大性能 

---
# Randomised Optimism via Competitive Co-Evolution for Matrix Games with Bandit Feedback 

**Title (ZH)**: 基于带惩罚反馈的矩阵博弈中竞争协同演化带来的随机乐观策略 

**Authors**: Shishen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.13562)  

**Abstract**: Learning in games is a fundamental problem in machine learning and artificial intelligence, with numerous applications~\citep{silver2016mastering,schrittwieser2020mastering}. This work investigates two-player zero-sum matrix games with an unknown payoff matrix and bandit feedback, where each player observes their actions and the corresponding noisy payoff. Prior studies have proposed algorithms for this setting~\citep{o2021matrix,maiti2023query,cai2024uncoupled}, with \citet{o2021matrix} demonstrating the effectiveness of deterministic optimism (e.g., \ucb) in achieving sublinear regret. However, the potential of randomised optimism in matrix games remains theoretically unexplored.
We propose Competitive Co-evolutionary Bandit Learning (\coebl), a novel algorithm that integrates evolutionary algorithms (EAs) into the bandit framework to implement randomised optimism through EA variation operators. We prove that \coebl achieves sublinear regret, matching the performance of deterministic optimism-based methods. To the best of our knowledge, this is the first theoretical regret analysis of an evolutionary bandit learning algorithm in matrix games.
Empirical evaluations on diverse matrix game benchmarks demonstrate that \coebl not only achieves sublinear regret but also consistently outperforms classical bandit algorithms, including \exptr~\citep{auer2002nonstochastic}, the variant \exptrni~\citep{cai2024uncoupled}, and \ucb~\citep{o2021matrix}. These results highlight the potential of evolutionary bandit learning, particularly the efficacy of randomised optimism via evolutionary algorithms in game-theoretic settings. 

**Abstract (ZH)**: 学习博弈是机器学习和人工智能中的一个基础问题，具有广泛的应用~\citep{silver2016mastering,schrittwieser2020mastering}。本文探讨了观测噪声支付的两人零和矩阵博弈问题，其中支付矩阵未知且仅提供带宽反馈，每个玩家仅能观察到自己的行为及其相应的噪声支付。先前的研究提出了此类设置的算法~\citep{o2021matrix,maiti2023query,cai2024uncoupled}，\citet{o2021matrix}展示了确定性乐观策略（例如，\ucb）在实现次线性遗憾方面的有效性。然而，随机乐观策略在矩阵博弈中的潜力仍缺乏理论上的探讨。
本文提出了竞争协同进化带宽学习 (\coebl)，这是一种将进化算法 (EAs) 集成到带宽框架中的新算法，通过进化算法的变异算子实现随机乐观策略。我们证明了 \coebl 实现了次线性遗憾，与基于确定性乐观策略的方法具有相同的表现。据我们所知，这是第一个关于进化带宽学习算法在矩阵博弈中遗憾分析的理论结果。
在各种矩阵博弈基准上的 empirical 评估表明，\coebl 不仅实现了次线性遗憾，而且在包括 \exptr~\citep{auer2002nonstochastic}、\exptrni~\citep{cai2024uncoupled} 和 \ucb~\citep{o2021matrix} 在内的经典带宽算法中表现更优。这些结果突显了进化带宽学习的潜力，特别是在博弈论设置中进化算法实现随机乐观策略的有效性。 

---
# AMAQA: A Metadata-based QA Dataset for RAG Systems 

**Title (ZH)**: AMAQA：一种基于元数据的QA数据集用于RAG系统 

**Authors**: Davide Bruni, Marco Avvenuti, Nicola Tonellotto, Maurizio Tesconi  

**Link**: [PDF](https://arxiv.org/pdf/2505.13557)  

**Abstract**: Retrieval-augmented generation (RAG) systems are widely used in question-answering (QA) tasks, but current benchmarks lack metadata integration, hindering evaluation in scenarios requiring both textual data and external information. To address this, we present AMAQA, a new open-access QA dataset designed to evaluate tasks combining text and metadata. The integration of metadata is especially important in fields that require rapid analysis of large volumes of data, such as cybersecurity and intelligence, where timely access to relevant information is critical. AMAQA includes about 1.1 million English messages collected from 26 public Telegram groups, enriched with metadata such as timestamps, topics, emotional tones, and toxicity indicators, which enable precise and contextualized queries by filtering documents based on specific criteria. It also includes 450 high-quality QA pairs, making it a valuable resource for advancing research on metadata-driven QA and RAG systems. To the best of our knowledge, AMAQA is the first single-hop QA benchmark to incorporate metadata and labels such as topics covered in the messages. We conduct extensive tests on the benchmark, establishing a new standard for future research. We show that leveraging metadata boosts accuracy from 0.12 to 0.61, highlighting the value of structured context. Building on this, we explore several strategies to refine the LLM input by iterating over provided context and enriching it with noisy documents, achieving a further 3-point gain over the best baseline and a 14-point improvement over simple metadata filtering. The dataset is available at this https URL 

**Abstract (ZH)**: retrieval-enhanced 生成（RAG）系统广泛应用于问答（QA）任务，但当前基准缺乏元数据集成，阻碍了在需要文本数据和外部信息的情景下的评估。为解决这一问题，我们提出了AMAQA，一个新开放访问的QA数据集，旨在评估结合文本和元数据的任务。元数据的集成尤其重要，特别是在需要快速分析大量数据的领域，如网络安全和情报领域，及时获取相关信息至关重要。AMAQA 包含来自 26 个公共 Telegram 组的约 110 万条英语消息，这些消息经过元数据增强，包括时间戳、主题、情感基调和毒性指标，这些元数据可以基于特定标准过滤文档，实现精确和上下文化的查询。此外，它还包括 450 对高质量的 QA 对，使其成为推进基于元数据的 QA 和 RAG 系统研究的重要资源。据我们所知，AMAQA 是第一个将元数据和消息涵盖的主题标签等标签结合的单跳 QA 基准。我们在基准上进行了广泛的测试，建立了未来研究的新标准。我们表明，利用元数据将准确性从 0.12 提高到 0.61，突显了结构化上下文的价值。在此基础上，我们探索了多种策略来细化 LL M 输入，通过迭代提供的上下文并用具有噪声的文档补充，实现了对最佳基准的 3 分进一步提升和对简单元数据过滤的 14 分改进。数据集可在以下链接获取：this https URL 

---
# Combining the Best of Both Worlds: A Method for Hybrid NMT and LLM Translation 

**Title (ZH)**: 兼收并蓄：一种混合NMT和LLM的翻译方法 

**Authors**: Zhanglin Wu, Daimeng Wei, Xiaoyu Chen, Hengchao Shang, Jiaxin Guo, Zongyao Li, Yuanchang Luo, Jinlong Yang, Zhiqiang Rao, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13554)  

**Abstract**: Large language model (LLM) shows promising performances in a variety of downstream tasks, such as machine translation (MT). However, using LLMs for translation suffers from high computational costs and significant latency. Based on our evaluation, in most cases, translations using LLMs are comparable to that generated by neural machine translation (NMT) systems. Only in particular scenarios, LLM and NMT models show respective advantages. As a result, integrating NMT and LLM for translation and using LLM only when necessary seems to be a sound solution. A scheduling policy that optimizes translation result while ensuring fast speed and as little LLM usage as possible is thereby required. We compare several scheduling policies and propose a novel and straightforward decider that leverages source sentence features. We conduct extensive experiments on multilingual test sets and the result shows that we can achieve optimal translation performance with minimal LLM usage, demonstrating effectiveness of our decider. 

**Abstract (ZH)**: 大规模语言模型（LLM）在多种下游任务中展现出有前途的表现，例如机器翻译（MT）。然而，使用LLM进行翻译面临着高计算成本和显著的延迟问题。根据我们的评估，在大多数情况下，使用LLM生成的翻译与神经机器翻译（NMT）系统生成的翻译相当。只有在特定场景下，LLM和NMT模型才各自表现出优势。因此，结合NMT和LLM进行翻译，并仅在必要时使用LLM似乎是一种可行的解决方案。由此需要一个优化翻译结果并确保快速速度和尽可能少使用LLM的调度策略。我们将几种调度策略进行比较，并提出一种基于源句子特征的新颖且简单的决策器。我们在多语言测试集上进行了广泛的实验，结果显示，我们可以在最大限度减少LLM使用的情况下实现最优的翻译性能，证明了我们决策器的有效性。 

---
# JIR-Arena: The First Benchmark Dataset for Just-in-time Information Recommendation 

**Title (ZH)**: JIR-竞技场：首个即时信息推荐基准数据集 

**Authors**: Ke Yang, Kevin Ros, Shankar Kumar Senthil Kumar, ChengXiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2505.13550)  

**Abstract**: Just-in-time Information Recommendation (JIR) is a service designed to deliver the most relevant information precisely when users need it, , addressing their knowledge gaps with minimal effort and boosting decision-making and efficiency in daily life. Advances in device-efficient deployment of foundation models and the growing use of intelligent wearable devices have made always-on JIR assistants feasible. However, there has been no systematic effort to formally define JIR tasks or establish evaluation frameworks. To bridge this gap, we present the first mathematical definition of JIR tasks and associated evaluation metrics. Additionally, we introduce JIR-Arena, a multimodal benchmark dataset featuring diverse, information-request-intensive scenarios to evaluate JIR systems across critical dimensions: i) accurately inferring user information needs, ii) delivering timely and relevant recommendations, and iii) avoiding irrelevant content that may distract users.
Developing a JIR benchmark dataset poses challenges due to subjectivity in estimating user information needs and uncontrollable system variables affecting reproducibility. To address these, JIR-Arena: i) combines input from multiple humans and large AI models to approximate information need distributions; ii) assesses JIR quality through information retrieval outcomes using static knowledge base snapshots; and iii) employs a multi-turn, multi-entity validation framework to improve objectivity and generality. Furthermore, we implement a baseline JIR system capable of processing real-time information streams aligned with user inputs. Our evaluation of this baseline system on JIR-Arena indicates that while foundation model-based JIR systems simulate user needs with reasonable precision, they face challenges in recall and effective content retrieval. To support future research in this new area, we fully release our code and data. 

**Abstract (ZH)**: 即时信息推荐 (JIR) 是一种服务，旨在在用户需要时提供最相关的信息，填补他们的知识空白，以最小的努力提升日常生活中的决策能力和效率。随着基础模型在设备上的高效部署和智能可穿戴设备的广泛应用，持续的即时信息推荐 (JIR) 助手变得可行。然而，尚未有任何系统性的努力来正式定义 JIR 任务或建立评估框架。为弥补这一空白，我们首次给出了 JIR 任务的数学定义及其相关的评估指标。此外，我们引入了 JIR-Arena，这是一个多模态基准数据集，包含多样化的、信息请求密集的场景，用于从关键维度评估 JIR 系统：i）准确推断用户的信息需求，ii）及时并提供相关推荐，iii）避免分心的不相关内容。

建立 JIR 基准数据集面临着挑战，包括估计用户信息需求的主观性以及影响可重现性的系统变量的不可控性。为解决这些问题，JIR-Arena：i）结合多名人类和大型 AI 模型的输入来近似信息需求分布；ii）通过使用静态知识库快照的信息检索结果评估 JIR 质量；iii）采用多轮、多实体验证框架提高客观性和通用性。此外，我们实现了一个基础模型驱动的 JIR 系统，能够实时处理与用户输入对齐的信息流。我们在 JIR-Arena 上对这一基础系统进行评估的结果表明，尽管基础模型驱动的 JIR 系统能够以合理的精度模拟用户需求，但在召回率和有效的内容检索方面仍面临挑战。为了支持这一新领域未来的研究，我们完全开源了我们的代码和数据。 

---
# Exploring Federated Pruning for Large Language Models 

**Title (ZH)**: 探索联邦剪枝在大型语言模型中的应用 

**Authors**: Pengxin Guo, Yinong Wang, Wei Li, Mengting Liu, Ming Li, Jinkai Zheng, Liangqiong Qu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13547)  

**Abstract**: LLM pruning has emerged as a promising technology for compressing LLMs, enabling their deployment on resource-limited devices. However, current methodologies typically require access to public calibration samples, which can be challenging to obtain in privacy-sensitive domains. To address this issue, we introduce FedPrLLM, a comprehensive federated pruning framework designed for the privacy-preserving compression of LLMs. In FedPrLLM, each client only needs to calculate a pruning mask matrix based on its local calibration data and share it with the server to prune the global model. This approach allows for collaborative pruning of the global model with the knowledge of each client while maintaining local data privacy. Additionally, we conduct extensive experiments to explore various possibilities within the FedPrLLM framework, including different comparison groups, pruning strategies, and the decision to scale weights. Our extensive evaluation reveals that one-shot pruning with layer comparison and no weight scaling is the optimal choice within the FedPrLLM framework. We hope our work will help guide future efforts in pruning LLMs in privacy-sensitive fields. Our code is available at this https URL. 

**Abstract (ZH)**: FedPrLLM：一种用于隐私保护的大规模语言模型压缩的联邦剪枝框架 

---
# Know Or Not: a library for evaluating out-of-knowledge base robustness 

**Title (ZH)**: 知与不知：一种评估知识库外 robustness 的库 

**Authors**: Jessica Foo, Pradyumna Shyama Prasad, Shaun Khoo  

**Link**: [PDF](https://arxiv.org/pdf/2505.13545)  

**Abstract**: While the capabilities of large language models (LLMs) have progressed significantly, their use in high-stakes applications have been limited due to risks of hallucination. One key approach in reducing hallucination is retrieval-augmented generation (RAG), but even in such setups, LLMs may still hallucinate when presented with questions outside of the knowledge base. Such behavior is unacceptable in high-stake applications where LLMs are expected to abstain from answering queries it does not have sufficient context on. In this work, we present a novel methodology for systematically evaluating out-of-knowledge base (OOKB) robustness of LLMs (whether LLMs know or do not know) in the RAG setting, without the need for manual annotation of gold standard answers. We implement our methodology in knowornot, an open-source library that enables users to develop their own customized evaluation data and pipelines for OOKB robustness. knowornot comprises four main features. Firstly, it provides a unified, high-level API that streamlines the process of setting up and running robustness benchmarks. Secondly, its modular architecture emphasizes extensibility and flexibility, allowing users to easily integrate their own LLM clients and RAG settings. Thirdly, its rigorous data modeling design ensures experiment reproducibility, reliability and traceability. Lastly, it implements a comprehensive suite of tools for users to customize their pipelines. We demonstrate the utility of knowornot by developing a challenging benchmark, PolicyBench, which spans four Question-Answer (QA) chatbots on government policies, and analyze its OOKB robustness. The source code of knowornot is available this https URL. 

**Abstract (ZH)**: 一种新的方法学，用于在检索增强生成设置中系统地评估大型语言模型的超出现有知识基础的鲁棒性（OOKB），而无需人工标注标准答案。 

---
# Multi-head Temporal Latent Attention 

**Title (ZH)**: 多头时间潜注意力 

**Authors**: Keqi Deng, Philip C. Woodland  

**Link**: [PDF](https://arxiv.org/pdf/2505.13544)  

**Abstract**: While Transformer self-attention offers strong parallelism, the Key-Value (KV) cache grows linearly with sequence length and becomes a bottleneck for inference efficiency. Multi-head latent attention was recently developed to compress the KV cache into a low-rank latent space. This paper proposes Multi-head Temporal Latent Attention (MTLA), which further reduces the KV cache size along the temporal dimension, greatly lowering the memory footprint of self-attention inference. MTLA employs a hyper-network to dynamically merge temporally adjacent KV cache vectors. To address the mismatch between the compressed KV cache and processed sequence lengths, a stride-aware causal mask is proposed to ensure efficient parallel training and consistency with inference behaviour. Experiments across tasks, including speech translation, speech recognition, speech understanding and text summarisation, demonstrate that MTLA achieves competitive performance compared to standard Multi-Head Attention (MHA), while greatly improving inference speed and GPU memory usage. For example, on a English-German speech translation task, MTLA achieves a 5.3x speedup and a reduction in GPU memory usage by a factor of 8.3 compared to MHA, while maintaining translation quality. 

**Abstract (ZH)**: 基于多头 temporal 潜在注意的 KV 缓存压缩（Multi-head Temporal Latent Attention for KV Cache Compression） 

---
# RAGXplain: From Explainable Evaluation to Actionable Guidance of RAG Pipelines 

**Title (ZH)**: RAGXplain：从可解释评估到RAG流水线的实际指导 

**Authors**: Dvir Cohen, Lin Burg, Gilad Barkan  

**Link**: [PDF](https://arxiv.org/pdf/2505.13538)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems show promise by coupling large language models with external knowledge, yet traditional RAG evaluation methods primarily report quantitative scores while offering limited actionable guidance for refining these complex pipelines. In this paper, we introduce RAGXplain, an evaluation framework that quantifies RAG performance and translates these assessments into clear insights that clarify the workings of its complex, multi-stage pipeline and offer actionable recommendations. Using LLM reasoning, RAGXplain converts raw scores into coherent narratives identifying performance gaps and suggesting targeted improvements. By providing transparent explanations for AI decision-making, our framework fosters user trust-a key challenge in AI adoption. Our LLM-based metric assessments show strong alignment with human judgments, and experiments on public question-answering datasets confirm that applying RAGXplain's actionable recommendations measurably improves system performance. RAGXplain thus bridges quantitative evaluation and practical optimization, empowering users to understand, trust, and enhance their AI systems. 

**Abstract (ZH)**: Retrieval-Augmented Generation (RAG) 系统通过将大型语言模型与外部知识结合展示出了潜力，然而传统的 RAG 评估方法主要报告定量分数，而对改进这些复杂流水线提供有限的实际指导。在本文中，我们介绍了 RAGXplain 评估框架，该框架量化 RAG 性能并将这些评估转化为清晰的见解，以阐明其复杂多阶段流水线的工作原理，并提供可操作的建议。利用大型语言模型推理，RAGXplain 将原始分数转化为连贯的故事，识别性能差距并提出针对性的改进措施。通过为 AI 决策提供透明解释，我们的框架促进了用户信任——这是 AI 采纳的关键挑战。基于大型语言模型的度量评估与人类判断高度一致，并且在公开问题回答数据集上的实验确认，应用 RAGXplain 的可操作建议可以显著提高系统性能。RAGXplain 从而在定量评估与实际优化之间架起了桥梁，助力用户理解、信任并提升其 AI 系统。 

---
# Information Extraction from Visually Rich Documents using LLM-based Organization of Documents into Independent Textual Segments 

**Title (ZH)**: 使用基于LLM的文档组织方法从视觉丰富的文档中提取信息 

**Authors**: Aniket Bhattacharyya, Anurag Tripathi, Ujjal Das, Archan Karmakar, Amit Pathak, Maneesh Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.13535)  

**Abstract**: Information extraction (IE) from Visually Rich Documents (VRDs) containing layout features along with text is a critical and well-studied task. Specialized non-LLM NLP-based solutions typically involve training models using both textual and geometric information to label sequences/tokens as named entities or answers to specific questions. However, these approaches lack reasoning, are not able to infer values not explicitly present in documents, and do not generalize well to new formats. Generative LLM-based approaches proposed recently are capable of reasoning, but struggle to comprehend clues from document layout especially in previously unseen document formats, and do not show competitive performance in heterogeneous VRD benchmark datasets. In this paper, we propose BLOCKIE, a novel LLM-based approach that organizes VRDs into localized, reusable semantic textual segments called $\textit{semantic blocks}$, which are processed independently. Through focused and more generalizable reasoning,our approach outperforms the state-of-the-art on public VRD benchmarks by 1-3% in F1 scores, is resilient to document formats previously not encountered and shows abilities to correctly extract information not explicitly present in documents. 

**Abstract (ZH)**: 从包含布局特征的视觉丰富文档中提取信息（IE）是至关重要的一个研究课题。基于非LLM的NLP专有解决方案通常涉及使用文本和几何信息训练模型来标注命名实体或特定问题的答案。然而，这些方法缺乏推理能力，无法推断文档中未明确呈现的值，并且不善于处理新格式。近年来提出的生成性LLM方法具备推理能力，但在理解以前未见过的文档格式的文档布局线索方面存在问题，在异构VRD基准数据集中表现也不具竞争力。本文提出BLOCKIE，这是一种基于LLM的新型方法，将VRDs组织成局部可重复使用的语义文本段，称为“语义块”，并独立处理。通过更聚焦且更通用的推理，该方法在公共VRD基准测试中F1分数上优于现有最佳方法1-3%，对以前未遇见过的文档格式具有鲁棒性，并展示了正确提取未明确呈现于文档中的信息的能力。 

---
# InterFeat: An Automated Pipeline for Finding Interesting Hypotheses in Structured Biomedical Data 

**Title (ZH)**: InterFeat: 一种自动化的流程，用于在结构化生物医学数据中发现有趣的假设。 

**Authors**: Dan Ofer, Michal Linial, Dafna Shahaf  

**Link**: [PDF](https://arxiv.org/pdf/2505.13534)  

**Abstract**: Finding interesting phenomena is the core of scientific discovery, but it is a manual, ill-defined concept. We present an integrative pipeline for automating the discovery of interesting simple hypotheses (feature-target relations with effect direction and a potential underlying mechanism) in structured biomedical data. The pipeline combines machine learning, knowledge graphs, literature search and Large Language Models. We formalize "interestingness" as a combination of novelty, utility and plausibility. On 8 major diseases from the UK Biobank, our pipeline consistently recovers risk factors years before their appearance in the literature. 40--53% of our top candidates were validated as interesting, compared to 0--7% for a SHAP-based baseline. Overall, 28% of 109 candidates were interesting to medical experts. The pipeline addresses the challenge of operationalizing "interestingness" scalably and for any target. We release data and code: this https URL 

**Abstract (ZH)**: 一种集成管道用于自动化发现结构化生物医学数据中的有趣简单假设 

---
# Distributional Soft Actor-Critic with Harmonic Gradient for Safe and Efficient Autonomous Driving in Multi-lane Scenarios 

**Title (ZH)**: 基于谐波梯度的分布软actor-critic在多车道场景中安全高效的自动驾驶 

**Authors**: Feihong Zhang, Guojian Zhan, Bin Shuai, Tianyi Zhang, Jingliang Duan, Shengbo Eben Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.13532)  

**Abstract**: Reinforcement learning (RL), known for its self-evolution capability, offers a promising approach to training high-level autonomous driving systems. However, handling constraints remains a significant challenge for existing RL algorithms, particularly in real-world applications. In this paper, we propose a new safety-oriented training technique called harmonic policy iteration (HPI). At each RL iteration, it first calculates two policy gradients associated with efficient driving and safety constraints, respectively. Then, a harmonic gradient is derived for policy updating, minimizing conflicts between the two gradients and consequently enabling a more balanced and stable training process. Furthermore, we adopt the state-of-the-art DSAC algorithm as the backbone and integrate it with our HPI to develop a new safe RL algorithm, DSAC-H. Extensive simulations in multi-lane scenarios demonstrate that DSAC-H achieves efficient driving performance with near-zero safety constraint violations. 

**Abstract (ZH)**: 强化学习（RL）因其自我进化能力，提供了训练高级自动驾驶系统的一种有希望的方法。然而，处理约束仍然是现有RL算法的一个重大挑战，尤其是在实际应用中。在本文中，我们提出了一种新的安全性导向训练技术，称为谐波策略迭代（HPI）。在每次RL迭代中，它首先分别计算与高效驾驶和安全约束相关的两个策略梯度，然后导出一个谐波梯度用于策略更新，通过最小化两个梯度之间的冲突，从而实现更平衡和稳定的训练过程。此外，我们采用最新的DSAC算法作为骨干，并将其与我们的HPI相结合，开发出一种新的安全RL算法DSAC-H。在多车道场景中的 extensive 模拟表明，DSAC-H 能够实现高效的驾驶性能，几乎不违反安全约束。 

---
# AdAEM: An Adaptively and Automated Extensible Measurement of LLMs' Value Difference 

**Title (ZH)**: AdAEM: 一种自适应且自动可扩展的大型语言模型价值差异度量方法 

**Authors**: Shitong Duan, Xiaoyuan Yi, Peng Zhang, Dongkuan Xu, Jing Yao, Tun Lu, Ning Gu, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.13531)  

**Abstract**: Assessing Large Language Models (LLMs)' underlying value differences enables comprehensive comparison of their misalignment, cultural adaptability, and biases. Nevertheless, current value measurement datasets face the informativeness challenge: with often outdated, contaminated, or generic test questions, they can only capture the shared value orientations among different LLMs, leading to saturated and thus uninformative results. To address this problem, we introduce AdAEM, a novel, self-extensible assessment framework for revealing LLMs' inclinations. Distinct from previous static benchmarks, AdAEM can automatically and adaptively generate and extend its test questions. This is achieved by probing the internal value boundaries of a diverse set of LLMs developed across cultures and time periods in an in-context optimization manner. The optimization process theoretically maximizes an information-theoretic objective to extract the latest or culturally controversial topics, providing more distinguishable and informative insights about models' value differences. In this way, AdAEM is able to co-evolve with the development of LLMs, consistently tracking their value dynamics. Using AdAEM, we generate 12,310 questions grounded in Schwartz Value Theory, conduct an extensive analysis to manifest our method's validity and effectiveness, and benchmark the values of 16 LLMs, laying the groundwork for better value research. 

**Abstract (ZH)**: 评估大型语言模型（LLMs）的内在价值差异 enables 对它们的偏差、文化适应性和偏见进行全面比较。然而，当前的价值度量数据集面临着信息量挑战：由于测试问题往往过时、受污染或通用，这些数据集只能捕捉不同LLMs之间共享的价值取向，导致结果饱和且缺乏信息量。为解决这一问题，我们介绍了AdAEM，这是一种新颖的、自我扩展的评估框架，用于揭示LLMs的倾向。不同于先前的静态基准测试，AdAEM能够自动且适应性地生成和扩展测试问题。这通过在多种文化和不同时期开发的不同LLMs的内部价值边界中进行上下文优化探针来实现。优化过程理论上最大化信息论目标，以提取最新的或文化争议性话题，从而提供更多可区分且富有信息量的关于模型价值差异的洞察。通过这种方式，AdAEM能够与LLMs的发展同步，持续跟踪其价值动态。利用AdAEM，我们生成了基于Schwartz价值理论的12,310个问题，进行广泛分析以验证我们方法的有效性，并对16个LLMs的价值进行了基准测试，为更好的价值研究奠定了基础。 

---
# LLM-Based User Simulation for Low-Knowledge Shilling Attacks on Recommender Systems 

**Title (ZH)**: 基于LLM的用户模拟在推荐系统中应对低知识型刷评攻击 

**Authors**: Shengkang Gu, Jiahao Liu, Dongsheng Li, Guangping Zhang, Mingzhe Han, Hansu Gu, Peng Zhang, Ning Gu, Li Shang, Tun Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13528)  

**Abstract**: Recommender systems (RS) are increasingly vulnerable to shilling attacks, where adversaries inject fake user profiles to manipulate system outputs. Traditional attack strategies often rely on simplistic heuristics, require access to internal RS data, and overlook the manipulation potential of textual reviews. In this work, we introduce Agent4SR, a novel framework that leverages Large Language Model (LLM)-based agents to perform low-knowledge, high-impact shilling attacks through both rating and review generation. Agent4SR simulates realistic user behavior by orchestrating adversarial interactions, selecting items, assigning ratings, and crafting reviews, while maintaining behavioral plausibility. Our design includes targeted profile construction, hybrid memory retrieval, and a review attack strategy that propagates target item features across unrelated reviews to amplify manipulation. Extensive experiments on multiple datasets and RS architectures demonstrate that Agent4SR outperforms existing low-knowledge baselines in both effectiveness and stealth. Our findings reveal a new class of emergent threats posed by LLM-driven agents, underscoring the urgent need for enhanced defenses in modern recommender systems. 

**Abstract (ZH)**: 基于大型语言模型的Agent4SR：一种低知识、高影响的评分和评论生成恶意攻击框架 

---
# Logic Jailbreak: Efficiently Unlocking LLM Safety Restrictions Through Formal Logical Expression 

**Title (ZH)**: 逻辑囚笼破解：通过形式逻辑表达高效解除语言模型安全限制 

**Authors**: Jingyu Peng, Maolin Wang, Nan Wang, Xiangyu Zhao, Jiatong Li, Kai Zhang, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13527)  

**Abstract**: Despite substantial advancements in aligning large language models (LLMs) with human values, current safety mechanisms remain susceptible to jailbreak attacks. We hypothesize that this vulnerability stems from distributional discrepancies between alignment-oriented prompts and malicious prompts. To investigate this, we introduce LogiBreak, a novel and universal black-box jailbreak method that leverages logical expression translation to circumvent LLM safety systems. By converting harmful natural language prompts into formal logical expressions, LogiBreak exploits the distributional gap between alignment data and logic-based inputs, preserving the underlying semantic intent and readability while evading safety constraints. We evaluate LogiBreak on a multilingual jailbreak dataset spanning three languages, demonstrating its effectiveness across various evaluation settings and linguistic contexts. 

**Abstract (ZH)**: 尽管在使大型语言模型（LLM）与人类价值观对齐方面取得了显著进展，当前的安全机制仍易受到牢笼突破攻击。我们假设这种漏洞源于对齐导向提示与恶意提示之间的分布差异。为探讨这一问题，我们引入了LogiBreak，这是一种新颖且通用的黑盒牢笼突破方法，利用逻辑表达式转换来规避LLM安全系统。通过将有害的自然语言提示转换为形式逻辑表达式，LogiBreak 利用对齐数据与基于逻辑的输入之间的分布差异，保持潜在的语义意图和可读性，同时规避安全约束。我们在涵盖三种语言的多语言牢笼突破数据集上评估了LogiBreak，展示了其在各种评估设置和语言情境下的有效性。 

---
# Geography-Aware Large Language Models for Next POI Recommendation 

**Title (ZH)**: 地理 Awareness 大型语言模型用于下一个POI 推荐 

**Authors**: Zhao Liu, Wei Liu, Huajie Zhu, Jianxing Yu, Jian Yin, Wang-Chien Lee, Shun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13526)  

**Abstract**: The next Point-of-Interest (POI) recommendation task aims to predict users' next destinations based on their historical movement data and plays a key role in location-based services and personalized applications. Accurate next POI recommendation depends on effectively modeling geographic information and POI transition relations, which are crucial for capturing spatial dependencies and user movement patterns. While Large Language Models (LLMs) exhibit strong capabilities in semantic understanding and contextual reasoning, applying them to spatial tasks like next POI recommendation remains challenging. First, the infrequent nature of specific GPS coordinates makes it difficult for LLMs to model precise spatial contexts. Second, the lack of knowledge about POI transitions limits their ability to capture potential POI-POI relationships. To address these issues, we propose GA-LLM (Geography-Aware Large Language Model), a novel framework that enhances LLMs with two specialized components. The Geographic Coordinate Injection Module (GCIM) transforms GPS coordinates into spatial representations using hierarchical and Fourier-based positional encoding, enabling the model to understand geographic features from multiple perspectives. The POI Alignment Module (PAM) incorporates POI transition relations into the LLM's semantic space, allowing it to infer global POI relationships and generalize to unseen POIs. Experiments on three real-world datasets demonstrate the state-of-the-art performance of GA-LLM. 

**Abstract (ZH)**: 基于地理意识的大语言模型（GA-LLM）：一种用于下一个地点推荐的任务新框架 

---
# Learning to Program Quantum Measurements for Machine Learning 

**Title (ZH)**: 学习编写量子测量程序以应用于机器学习 

**Authors**: Samual Yen-Chi Chen, Huan-Hsin Tseng, Hsin-Yi Lin, Shinjae Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2505.13525)  

**Abstract**: The rapid advancements in quantum computing (QC) and machine learning (ML) have sparked significant interest, driving extensive exploration of quantum machine learning (QML) algorithms to address a wide range of complex challenges. The development of high-performance QML models requires expert-level expertise, presenting a key challenge to the widespread adoption of QML. Critical obstacles include the design of effective data encoding strategies and parameterized quantum circuits, both of which are vital for the performance of QML models. Furthermore, the measurement process is often neglected-most existing QML models employ predefined measurement schemes that may not align with the specific requirements of the targeted problem. We propose an innovative framework that renders the observable of a quantum system-specifically, the Hermitian matrix-trainable. This approach employs an end-to-end differentiable learning framework, enabling simultaneous optimization of the neural network used to program the parameterized observables and the standard quantum circuit parameters. Notably, the quantum observable parameters are dynamically programmed by the neural network, allowing the observables to adapt in real time based on the input data stream. Through numerical simulations, we demonstrate that the proposed method effectively programs observables dynamically within variational quantum circuits, achieving superior results compared to existing approaches. Notably, it delivers enhanced performance metrics, such as higher classification accuracy, thereby significantly improving the overall effectiveness of QML models. 

**Abstract (ZH)**: 量子计算和机器学习的迅猛发展激发了广泛兴趣，推动了量子机器学习算法的探索，以应对各种复杂挑战。高性能量子机器学习模型的开发需要专家级的专业知识，成为其广泛应用的关键障碍。关键障碍包括有效的数据编码策略和参数化量子电路的设计，这两个方面对量子机器学习模型的性能至关重要。此外，测量过程往往被忽视——大多数现有量子机器学习模型采用预定义的测量方案，可能不适用于特定问题的需求。我们提出了一种创新框架，使其可观测值——具体而言，是厄米矩阵——可训练。该方法采用端到端可微学习框架，同时优化用于编程参数化可观测值的神经网络和标准量子电路参数。值得注意的是，观测值参数由神经网络动态编程，使观测值能够根据输入数据流实时调整。通过数值仿真，我们证明了所提出的方法能够有效在变分量子电路中动态编程可观测值，相比现有方法取得了更优的结果。它还实现了更高的分类准确性等改进性能指标，从而显著提高了量子机器学习模型的整体有效性。 

---
# ACPs: Agent Collaboration Protocols for the Internet of Agents 

**Title (ZH)**: 代理协作协议：代理互联网中的协作协议 

**Authors**: Jun Liu, Ke Yu, Keliang Chen, Ke Li, Yuxinyue Qian, Xiaolian Guo, Haozhe Song, Yinming Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.13523)  

**Abstract**: With the rapid advancement of artificial intelligence, the proliferation of autonomous agents has introduced new challenges in interoperability, scalability, and coordination. The Internet of Agents (IoA) aims to interconnect heterogeneous agents through standardized communication protocols, enabling seamless collaboration and intelligent task execution. However, existing agent communication protocols such as MCP, A2A, and ANP remain fragmented and scenario-specific. To address this gap, we propose Agent Collaboration Protocols (ACPs), a comprehensive protocol suite for the IoA. ACPs include registration, discovery, interaction, and tooling protocols to support trustable access, capability orchestration, and workflow construction. We present the architecture, key technologies, and application workflows of ACPs, and demonstrate its effectiveness in a collaborative restaurant booking scenario. ACPs lay the foundation for building a secure, open, and scalable agent internet infrastructure. 

**Abstract (ZH)**: 随着人工智能的迅速发展，自主代理的 proliferate 引入了互操作性、可扩展性和协调性的新挑战。代理互联网 (IoA) 致力于通过标准化通信协议互联异构代理，实现无缝协作和智能任务执行。然而，现有的代理通信协议如 MCP、A2A 和 ANP 仍然fragmented 和场景特定。为解决这一问题，我们提出了代理协作协议 (ACPs)，这是一种适用于 IoA 的综合协议套件。ACPs 包括注册、发现、交互和工具协议，以支持可信访问、能力编排和工作流构建。我们介绍了 ACPs 的架构、关键技术及其应用工作流，并在协作餐厅预订场景中展示了其有效性。ACPs 为构建安全、开放和可扩展的代理互联网基础设施奠定了基础。 

---
# Beyond Retrieval: Joint Supervision and Multimodal Document Ranking for Textbook Question Answering 

**Title (ZH)**: 超越检索：联合监督与多模态文档排序在教材问答中的应用 

**Authors**: Hessa Alawwad, Usman Naseem, Areej Alhothali, Ali Alkhathlan, Amani Jamal  

**Link**: [PDF](https://arxiv.org/pdf/2505.13520)  

**Abstract**: Textbook question answering (TQA) is a complex task, requiring the interpretation of complex multimodal context. Although recent advances have improved overall performance, they often encounter difficulties in educational settings where accurate semantic alignment and task-specific document retrieval are essential. In this paper, we propose a novel approach to multimodal textbook question answering by introducing a mechanism for enhancing semantic representations through multi-objective joint training. Our model, Joint Embedding Training With Ranking Supervision for Textbook Question Answering (JETRTQA), is a multimodal learning framework built on a retriever--generator architecture that uses a retrieval-augmented generation setup, in which a multimodal large language model generates answers. JETRTQA is designed to improve the relevance of retrieved documents in complex educational contexts. Unlike traditional direct scoring approaches, JETRTQA learns to refine the semantic representations of questions and documents through a supervised signal that combines pairwise ranking and implicit supervision derived from answers. We evaluate our method on the CK12-QA dataset and demonstrate that it significantly improves the discrimination between informative and irrelevant documents, even when they are long, complex, and multimodal. JETRTQA outperforms the previous state of the art, achieving a 2.4\% gain in accuracy on the validation set and 11.1\% on the test set. 

**Abstract (ZH)**: 多模态教科书问答（TQA）是一种复杂的任务，需要解释复杂的多模态背景信息。尽管最近的进展提高了整体性能，但在需要准确语义对齐和任务特定文档检索的教学环境中，仍面临挑战。本文提出了一种通过多目标联合训练增强语义表示的新方法来解决多模态教科书问答问题。我们的模型“基于排名监督的联合嵌入训练用于教科书问答”（JETRTQA）是一种基于检索-生成架构的多模态学习框架，使用了检索增强生成设置，其中多模态大型语言模型生成答案。JETRTQA旨在提高在复杂教育背景下检索到的文档的相关性。与传统的直接评分方法不同，JETRTQA通过结合成对排序监督和从答案推导出的隐式监督的监督信号，学习细化问题和文档的语义表示。我们在CK12-QA数据集上评估了该方法，并证明它显著提高了信息性文档和无关文档之间的区分能力，即使它们是长、复杂且多模态的。JETRTQA在验证集上实现了2.4%的准确率提升，在测试集上实现了11.1%的提升。 

---
# Continuous Domain Generalization 

**Title (ZH)**: 连续域泛化 

**Authors**: Zekun Cai, Yiheng Yao, Guangji Bai, Renhe Jiang, Xuan Song, Ryosuke Shibasaki, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13519)  

**Abstract**: Real-world data distributions often shift continuously across multiple latent factors such as time, geography, and socioeconomic context. However, existing domain generalization approaches typically treat domains as discrete or evolving along a single axis (e.g., time), which fails to capture the complex, multi-dimensional nature of real-world variation. This paper introduces the task of Continuous Domain Generalization (CDG), which aims to generalize predictive models to unseen domains defined by arbitrary combinations of continuous variation descriptors. We present a principled framework grounded in geometric and algebraic theory, showing that optimal model parameters across domains lie on a low-dimensional manifold. To model this structure, we propose a Neural Lie Transport Operator (NeuralLTO), which enables structured parameter transitions by enforcing geometric continuity and algebraic consistency. To handle noisy or incomplete domain descriptors, we introduce a gating mechanism to suppress irrelevant dimensions and a local chart-based strategy for robust generalization. Extensive experiments on synthetic and real-world datasets-including remote sensing, scientific documents, and traffic forecasting-demonstrate that our method significantly outperforms existing baselines in generalization accuracy and robustness under descriptor imperfections. 

**Abstract (ZH)**: 连续域泛化（CDG）中的预测模型泛化研究 

---
# Data Balancing Strategies: A Survey of Resampling and Augmentation Methods 

**Title (ZH)**: 数据平衡策略：重采样和 augmentation 方法综述 

**Authors**: Behnam Yousefimehr, Mehdi Ghatee, Mohammad Amin Seifi, Javad Fazli, Sajed Tavakoli, Zahra Rafei, Shervin Ghaffari, Abolfazl Nikahd, Mahdi Razi Gandomani, Alireza Orouji, Ramtin Mahmoudi Kashani, Sarina Heshmati, Negin Sadat Mousavi  

**Link**: [PDF](https://arxiv.org/pdf/2505.13518)  

**Abstract**: Imbalanced data poses a significant obstacle in machine learning, as an unequal distribution of class labels often results in skewed predictions and diminished model accuracy. To mitigate this problem, various resampling strategies have been developed, encompassing both oversampling and undersampling techniques aimed at modifying class proportions. Conventional oversampling approaches like SMOTE enhance the representation of the minority class, whereas undersampling methods focus on trimming down the majority class. Advances in deep learning have facilitated the creation of more complex solutions, such as Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs), which are capable of producing high-quality synthetic examples. This paper reviews a broad spectrum of data balancing methods, classifying them into categories including synthetic oversampling, adaptive techniques, generative models, ensemble-based strategies, hybrid approaches, undersampling, and neighbor-based methods. Furthermore, it highlights current developments in resampling techniques and discusses practical implementations and case studies that validate their effectiveness. The paper concludes by offering perspectives on potential directions for future exploration in this domain. 

**Abstract (ZH)**: 不平衡数据在机器学习中构成重大障碍，不均衡的类别标签分布往往导致预测偏差和模型准确度降低。为缓解这一问题，发展了多种重采样策略，包括过采样和欠采样技术，以调整类别比例。传统的过采样方法如SMOTE增强了少数类的代表性，而欠采样方法则侧重于减少多数类的数量。深度学习的进步催生了更复杂的方法，如生成对抗网络（GANs）和变分自编码器（VAEs），这些方法能够生成高质量的合成样本。本文综述了广泛的数据平衡方法，将它们分类为合成过采样、自适应技术、生成模型、基于集成的方法、混合方法、欠采样和邻近基于的方法。此外，文章还强调了重采样技术的最新进展，并讨论了实际实施和案例研究，以验证其有效性。最后，本文提出了未来研究方向的展望。 

---
# HALO: Hierarchical Autonomous Logic-Oriented Orchestration for Multi-Agent LLM Systems 

**Title (ZH)**: HALO: 分层自主逻辑导向的多代理LLM系统 orchestrator 

**Authors**: Zhipeng Hou, Junyi Tang, Yipeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13516)  

**Abstract**: Recent advancements in Multi-Agent Systems (MAS) powered by Large Language Models (LLMs) have demonstrated tremendous potential in diverse task scenarios. Nonetheless, existing agentic systems typically rely on predefined agent-role design spaces and static communication structures, limiting their adaptability as well as flexibility in complex interaction environments and leading to subpar performance on highly specialized and expert-level tasks. To address these issues, we introduce HALO, a multi-agent collaboration framework based on a hierarchical reasoning architecture. Specifically, we incorporate a high-level planning agent for task decomposition, mid-level role-design agents for subtask-specific agent instantiation, and low-level inference agents for subtask execution. Particularly, subtask execution is reformulated as a structured workflow search problem, where Monte Carlo Tree Search (MCTS) systematically explores the agentic action space to construct optimal reasoning trajectories. Additionally, as the majority of users lack expertise in prompt engineering, we leverage an Adaptive Prompt Refinement module to transform raw queries into task-specific prompts. Empirical evaluations on Code Generation (HumanEval), General Reasoning (MMLU), and Arithmetic Reasoning (MATH) benchmark datasets highlight the effectiveness of HALO, yielding a 14.4% average improvement over state-of-the-art baselines. Notably, HALO achieves up to 13.3% performance gain on the Moral Scenarios subject in the MMLU benchmark and up to 19.6% performance gain on the Algebra subarea in the MATH benchmark, indicating its advanced proficiency in tackling highly specialized and expert-level tasks. The code repository is available at this https URL. 

**Abstract (ZH)**: Recent advancements in Multi-Agent Systems (MAS) powered by Large Language Models (LLMs) have demonstrated tremendous potential in diverse task scenarios. 

---
# LoRASuite: Efficient LoRA Adaptation Across Large Language Model Upgrades 

**Title (ZH)**: LoRASuite: 在大规模语言模型升级中高效适应的LoRA方法 

**Authors**: Yanan Li, Fanxu Meng, Muhan Zhang, Shiai Zhu, Shangguang Wang, Mengwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13515)  

**Abstract**: As Large Language Models (LLMs) are frequently updated, LoRA weights trained on earlier versions quickly become obsolete. The conventional practice of retraining LoRA weights from scratch on the latest model is costly, time-consuming, and environmentally detrimental, particularly as the diversity of LLMs and downstream tasks expands. This motivates a critical question: "How can we efficiently leverage existing LoRA weights to adapt to newer model versions?" To address this, we propose LoRASuite, a modular approach tailored specifically to various types of LLM updates. First, we compute a transfer matrix utilizing known parameters from both old and new LLMs. Next, we allocate corresponding layers and attention heads based on centered kernel alignment and cosine similarity metrics, respectively. A subsequent small-scale, skillful fine-tuning step ensures numerical stability. Experimental evaluations demonstrate that LoRASuite consistently surpasses small-scale vanilla LoRA methods. Notably, on backbone LLMs such as MiniCPM and Qwen, LoRASuite even exceeds the performance of full-scale LoRA retraining, with average improvements of +1.4 and +6.6 points on math tasks, respectively. Additionally, LoRASuite significantly reduces memory consumption by 5.5 GB and computational time by 78.23%. 

**Abstract (ZH)**: 大语言模型（LLMs）频繁更新，早期版本训练的LoRA权重很快变得过时。从最新模型重新训练LoRA权重的传统做法成本高、耗时且环境破坏性大，尤其是随着LLM的多样性和下游任务的增加。这促使了一个关键问题：“我们如何高效地利用现有LoRA权重适应新的模型版本？”为解决这一问题，我们提出LoRASuite，这是一种专门针对各种类型LLM更新的模块化方法。首先，我们利用旧和新LLM已知参数计算一个转移矩阵。然后，我们根据中心核对齐和余弦相似度指标分配相应的层和注意力头。后续的小规模、技艺精湛的微调步骤确保数值稳定性。实验评估表明，LoRASuite 总是优于小规模的 vanilla LoRA 方法。特别地，在MiniCPM和Qwen等骨干LLM上，LoRASuite的性能甚至超过了全面重新训练的LoRA，分别在数学任务上取得了平均1.4和6.6个百分点的提升。此外，LoRASuite还显著减少了5.5 GB的内存消耗和78.23%的计算时间。 

---
# Induction Head Toxicity Mechanistically Explains Repetition Curse in Large Language Models 

**Title (ZH)**: 诱导头毒性机制性解释大型语言模型中的重复诅咒 

**Authors**: Shuxun Wang, Qingyu Yin, Chak Tou Leong, Qiang Zhang, Linyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13514)  

**Abstract**: Repetition curse is a phenomenon where Large Language Models (LLMs) generate repetitive sequences of tokens or cyclic sequences. While the repetition curse has been widely observed, its underlying mechanisms remain poorly understood. In this work, we investigate the role of induction heads--a specific type of attention head known for their ability to perform in-context learning--in driving this repetitive behavior. Specifically, we focus on the "toxicity" of induction heads, which we define as their tendency to dominate the model's output logits during repetition, effectively excluding other attention heads from contributing to the generation process. Our findings have important implications for the design and training of LLMs. By identifying induction heads as a key driver of the repetition curse, we provide a mechanistic explanation for this phenomenon and suggest potential avenues for mitigation. We also propose a technique with attention head regularization that could be employed to reduce the dominance of induction heads during generation, thereby promoting more diverse and coherent outputs. 

**Abstract (ZH)**: 重复诅咒是一种大型语言模型（LLMs）生成重复的令牌序列或循环序列的现象。尽管重复诅咒已经被广泛观察到，但其底层机制仍不甚明了。在本文中，我们调查了归纳头的作用——这种特定类型的注意力头以其内在的在上下文学习能力而著称——它们在推动这种重复行为中的作用。具体而言，我们将注意力头的“毒性”定义为其在重复过程中主导模型输出对数的趋势，从而有效地排除其他注意力头对生成过程的贡献。我们的发现对LLM的设计和训练具有重要影响。通过将归纳头识别为重复诅咒的关键驱动因素，我们提供了该现象的机制性解释，并提出了可能的缓解途径。我们还提出了一种带有注意力头正则化的技术，可以在生成过程中减少归纳头的主导作用，从而促进更多样化和连贯的输出。 

---
# Time-R1: Towards Comprehensive Temporal Reasoning in LLMs 

**Title (ZH)**: 时间轴-1：面向LLMs的综合性 temporal推理研究 

**Authors**: Zijia Liu, Peixuan Han, Haofei Yu, Haoru Li, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2505.13508)  

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities but lack robust temporal intelligence, struggling to integrate reasoning about the past with predictions and plausible generations of the future. Meanwhile, existing methods typically target isolated temporal skills, such as question answering about past events or basic forecasting, and exhibit poor generalization, particularly when dealing with events beyond their knowledge cutoff or requiring creative foresight. To address these limitations, we introduce \textit{Time-R1}, the first framework to endow a moderate-sized (3B-parameter) LLM with comprehensive temporal abilities: understanding, prediction, and creative generation. Our approach features a novel three-stage development path; the first two constitute a \textit{reinforcement learning (RL) curriculum} driven by a meticulously designed dynamic rule-based reward system. This framework progressively builds (1) foundational temporal understanding and logical event-time mappings from historical data, (2) future event prediction skills for events beyond its knowledge cutoff, and finally (3) enables remarkable generalization to creative future scenario generation without any fine-tuning. Strikingly, experiments demonstrate that Time-R1 outperforms models over 200 times larger, including the state-of-the-art 671B DeepSeek-R1, on highly challenging future event prediction and creative scenario generation benchmarks. This work provides strong evidence that thoughtfully engineered, progressive RL fine-tuning allows smaller, efficient models to achieve superior temporal performance, offering a practical and scalable path towards truly time-aware AI. To foster further research, we also release \textit{Time-Bench}, a large-scale multi-task temporal reasoning dataset derived from 10 years of news data, and our series of \textit{Time-R1} checkpoints. 

**Abstract (ZH)**: Large Language Models (LLMs)表现出色但在时序智能方面缺乏稳健性，难以将对过去推理的能力与对未来预测和合理生成未来的能力整合起来。同时，现有方法通常针对孤立的时序技能，如关于过去事件的问题回答或基本的预测，并且在处理超出其知识截止点的事件或需要创造力的前瞻性时表现出较差的泛化能力。为了解决这些局限性，我们引入了Time-R1，这是第一个为中等规模（3B参数）LLM赋予全面时序能力的框架：理解、预测和创造性生成。我们的方法采用了一种新颖的三阶段开发路径；前两个阶段构成一个基于精心设计的动态规则奖励系统的强化学习（RL）课程。该框架逐步构建（1）基于历史数据的时序基础理解和逻辑事件时间映射，（2）对未来事件的预测能力，特别是超越其知识截止点的事件，最后（3）无需任何微调即可实现非凡的创造性未来情境生成泛化能力。实验结果表明，Time-R1在复杂的未来事件预测和创造性情境生成基准测试中优于包括当前最先进的671B DeepSeek-R1在内的大得多的模型。这项工作提供了有力证据，表明精心设计的渐进式RL微调允许较小的、高效的模型实现卓越的时序性能，提供了一条真正的时间感知AI的实用和可扩展路径。为了促进进一步的研究，我们还发布了Time-Bench，这是一个源自十年新闻数据的大型多任务时序推理数据集，以及我们的一系列Time-R1检查点。 

---
# EcoSafeRAG: Efficient Security through Context Analysis in Retrieval-Augmented Generation 

**Title (ZH)**: EcoSafeRAG：通过检索增强生成中的语境分析实现高效安全性 

**Authors**: Ruobing Yao, Yifei Zhang, Shuang Song, Neng Gao, Chenyang Tu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13506)  

**Abstract**: Retrieval-Augmented Generation (RAG) compensates for the static knowledge limitations of Large Language Models (LLMs) by integrating external knowledge, producing responses with enhanced factual correctness and query-specific contextualization. However, it also introduces new attack surfaces such as corpus poisoning at the same time. Most of the existing defense methods rely on the internal knowledge of the model, which conflicts with the design concept of RAG. To bridge the gap, EcoSafeRAG uses sentence-level processing and bait-guided context diversity detection to identify malicious content by analyzing the context diversity of candidate documents without relying on LLM internal knowledge. Experiments show EcoSafeRAG delivers state-of-the-art security with plug-and-play deployment, simultaneously improving clean-scenario RAG performance while maintaining practical operational costs (relatively 1.2$\times$ latency, 48\%-80\% token reduction versus Vanilla RAG). 

**Abstract (ZH)**: EcoSafeRAG通过句子级处理和诱饵引导的上下文多样性检测，在不依赖大语言模型内部知识的情况下，识别恶意内容，从而增强RAG的安全性。 

---
# An agentic system with reinforcement-learned subsystem improvements for parsing form-like documents 

**Title (ZH)**: 基于强化学习子系统改进的代理系统及其在解析表单-like 文档中的应用 

**Authors**: Ayesha Amjad, Saurav Sthapit, Tahir Qasim Syed  

**Link**: [PDF](https://arxiv.org/pdf/2505.13504)  

**Abstract**: Extracting alphanumeric data from form-like documents such as invoices, purchase orders, bills, and financial documents is often performed via vision (OCR) and learning algorithms or monolithic pipelines with limited potential for systemic improvements. We propose an agentic AI system that leverages Large Language Model (LLM) agents and a reinforcement learning (RL) driver agent to automate consistent, self-improving extraction under LLM inference uncertainty. Our work highlights the limitations of monolithic LLM-based extraction and introduces a modular, multi-agent framework with task-specific prompts and an RL policy of rewards and penalties to guide a meta-prompting agent to learn from past errors and improve prompt-based actor agents. This self-corrective adaptive system handles diverse documents, file formats, layouts, and LLMs, aiming to automate accurate information extraction without the need for human intervention. Results as reported on two benchmark datasets of SOIRE, and CORD, are promising for the agentic AI framework. 

**Abstract (ZH)**: 基于视图（OCR）和学习算法或单一管道从发票、采购订单、账单和金融文档中提取 alphanumeric 数据往往具有有限的系统改进潜力。我们提出了一种代理型人工智能系统，利用大型语言模型（LLM）代理和强化学习（RL）驱动代理，以应对 LLM 推断不确定性，在自动化和自我改进的数据提取中取得一致效果。我们的工作揭示了单一 LLM 基础提取的局限性，并引入了一个模块化的多代理框架，该框架具有任务特定的提示和基于奖励和惩罚的 RL 策略，以引导元提示代理从过去错误中学习并改进基于提示的执行代理。此自我纠正的自适应系统能够处理多样化文档、文件格式、布局和 LLM，旨在在无需人工干预的情况下实现准确信息的自动化提取。在 SOIRE 和 CORD 的两个基准数据集上的结果表明，代理型人工智能框架前景广阔。 

---
# Noise Injection Systemically Degrades Large Language Model Safety Guardrails 

**Title (ZH)**: 系统性注入噪声会降低大型语言模型的安全防护能力 

**Authors**: Prithviraj Singh Shahani, Matthias Scheutz  

**Link**: [PDF](https://arxiv.org/pdf/2505.13500)  

**Abstract**: Safety guardrails in large language models (LLMs) are a critical component in preventing harmful outputs. Yet, their resilience under perturbation remains poorly understood. In this paper, we investigate the robustness of safety fine-tuning in LLMs by systematically injecting Gaussian noise into model activations. We show across multiple open-weight models that (1) Gaussian noise raises harmful-output rates (p < 0.001) by up to 27%, (2) that deeper safety fine-tuning affords no extra protection, and (3) that chain-of-thought reasoning remains largely intact. The findings reveal critical vulnerabilities in current safety alignment techniques and highlight the potential of reasoning-based and reinforcement learning approaches as promising direction for developing more robust AI safety systems. These results have important implications for real-world deployment of LLMs in safety-critical applications as these results imply that widely-deployed safety tuning methods can fail even without adversarial prompts. 

**Abstract (ZH)**: 大型语言模型（LLMs）中的安全防护栏是防止有害输出的关键组成部分。然而，它们在受到扰动时的鲁棒性仍 poorly understood。本文通过系统地向模型激活中注入高斯噪声，研究了LLMs的安全微调的鲁棒性。结果显示，在多个开放权重模型中：（1）高斯噪声将有害输出率提高最多27%（p < 0.001）；（2）深层安全微调并未提供额外保护；（3）思维链推理保持相对完好。研究发现揭示了当前安全对齐技术中的关键脆弱性，并强调了基于推理和强化学习的方法在开发更鲁棒的AI安全系统方面的潜在前景。这些结果对于在安全关键应用中部署大型语言模型具有重要意义，因为这些结果表明，广泛部署的安全调优方法即使在没有对抗性提示的情况下也可能失败。 

---
# Optimal Control for Transformer Architectures: Enhancing Generalization, Robustness and Efficiency 

**Title (ZH)**: 变压器架构的最优控制：增强泛化能力、稳健性和效率 

**Authors**: Kelvin Kan, Xingjian Li, Benjamin J. Zhang, Tuhin Sahai, Stanley Osher, Markos A. Katsoulakis  

**Link**: [PDF](https://arxiv.org/pdf/2505.13499)  

**Abstract**: We study Transformers through the perspective of optimal control theory, using tools from continuous-time formulations to derive actionable insights into training and architecture design. This framework improves the performance of existing Transformer models while providing desirable theoretical guarantees, including generalization and robustness. Our framework is designed to be plug-and-play, enabling seamless integration with established Transformer models and requiring only slight changes to the implementation. We conduct seven extensive experiments on tasks motivated by text generation, sentiment analysis, image classification, and point cloud classification. Experimental results show that the framework improves the test performance of the baselines, while being more parameter-efficient. On character-level text generation with nanoGPT, our framework achieves a 46% reduction in final test loss while using 42% fewer parameters. On GPT-2, our framework achieves a 5.6% reduction in final test loss, demonstrating scalability to larger models. To the best of our knowledge, this is the first work that applies optimal control theory to both the training and architecture of Transformers. It offers a new foundation for systematic, theory-driven improvements and moves beyond costly trial-and-error approaches. 

**Abstract (ZH)**: 我们从最优控制理论的角度研究Transformer，利用连续时间形式化的工具来获取关于训练和架构设计的实际洞察。该框架在提升现有Transformer模型性能的同时，提供了泛化能力和鲁棒性等理想的理论保证。该框架设计为即插即用，可以无缝集成到已有的Transformer模型中，并且只需对实现进行轻微修改。我们通过七个涉及文本生成、情感分析、图像分类和点云分类等任务的广泛实验进行了验证。实验结果表明，该框架在基础模型上提升了测试性能，且更具参数效率。在字符级别文本生成任务中，使用nanoGPT时，该框架在最终测试损失上实现了46%的下降，同时使用42%较少的参数。在GPT-2上，该框架实现了5.6%的最终测试损失下降，展示了其对更大模型的扩展性。据我们所知，这是首次将最优控制理论应用于Transformer的训练与架构中，为系统性的、理论驱动的改进提供了新的基础，并超越了昂贵的试错方法。 

---
# IRLBench: A Multi-modal, Culturally Grounded, Parallel Irish-English Benchmark for Open-Ended LLM Reasoning Evaluation 

**Title (ZH)**: IRLBench：一个基于多模态和文化背景的开放性LLM推理评价平行爱尔兰英语基准数据集 

**Authors**: Khanh-Tung Tran, Barry O'Sullivan, Hoang D. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.13498)  

**Abstract**: Recent advances in Large Language Models (LLMs) have demonstrated promising knowledge and reasoning abilities, yet their performance in multilingual and low-resource settings remains underexplored. Existing benchmarks often exhibit cultural bias, restrict evaluation to text-only, rely on multiple-choice formats, and, more importantly, are limited for extremely low-resource languages. To address these gaps, we introduce IRLBench, presented in parallel English and Irish, which is considered definitely endangered by UNESCO. Our benchmark consists of 12 representative subjects developed from the 2024 Irish Leaving Certificate exams, enabling fine-grained analysis of model capabilities across domains. By framing the task as long-form generation and leveraging the official marking scheme, it does not only support a comprehensive evaluation of correctness but also language fidelity. Our extensive experiments of leading closed-source and open-source LLMs reveal a persistent performance gap between English and Irish, in which models produce valid Irish responses less than 80\% of the time, and answer correctly 55.8\% of the time compared to 76.2\% in English for the best-performing model. We release IRLBench (this https URL) and an accompanying evaluation codebase (this https URL) to enable future research on robust, culturally aware multilingual AI development. 

**Abstract (ZH)**: Recent Advances in Large Language Models in Multilingual and Low-Resource Settings: The Introduction of IRLBench 

---
# LODGE: Joint Hierarchical Task Planning and Learning of Domain Models with Grounded Execution 

**Title (ZH)**: LODGE: 联合层次化任务规划与接地执行领域模型学习 

**Authors**: Claudius Kienle, Benjamin Alt, Oleg Arenz, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2505.13497)  

**Abstract**: Large Language Models (LLMs) enable planning from natural language instructions using implicit world knowledge, but often produce flawed plans that require refinement. Instead of directly predicting plans, recent methods aim to learn a problem domain that can be solved for different goal states using classical planners. However, these approaches require significant human feedback to obtain useful models. We address this shortcoming by learning hierarchical domains, where low-level predicates and actions are composed into higher-level counterparts, and by leveraging simulation to validate their preconditions and effects. This hierarchical approach is particularly powerful for long-horizon planning, where LLM-based planning approaches typically struggle. Furthermore, we introduce a central error reasoner to ensure consistency among the different planning levels. Evaluation on two challenging International Planning Competition (IPC) domains and a long-horizon robot manipulation task demonstrates higher planning success rates than state-of-the-art domain synthesis and LLM-modulo planning methods, while constructing high-quality models of the domain. Resources, videos and detailed experiment results are available at this https URL. 

**Abstract (ZH)**: 大型语言模型通过隐含的世界知识从自然语言指令中进行规划，但常常会产生需要改进的规划方案。最近的方法致力于学习可以解决不同目标状态的经典规划问题领域，而不是直接预测规划方案。然而，这些方法需要大量的人工反馈以获得有用模型。我们通过学习层次化领域来解决这一不足，其中低级谓词和动作组合成高级对应物，并利用模拟来验证其前提条件和效果。这种层次化的方法特别适用于长时规划，而基于大型语言模型的规划方法在长时规划中通常面临挑战。此外，我们引入了一个中心错误推理器以确保不同规划层次之间的一致性。在两个具有挑战性的国际规划竞赛（IPC）领域和一个长时机器人操作任务上的评估表明，我们的方法在规划成功率方面优于最新的领域合成和基于大型语言模型的规划方法，同时构建了高质量的领域模型。更多信息、视频及详细的实验结果请访问 <https://>。 

---
# ProdRev: A DNN framework for empowering customers using generative pre-trained transformers 

**Title (ZH)**: ProdRev: 一种基于生成预训练变换器的深度神经网络框架，赋能客户 

**Authors**: Aakash Gupta, Nataraj Das  

**Link**: [PDF](https://arxiv.org/pdf/2505.13491)  

**Abstract**: Following the pandemic, customers, preference for using e-commerce has accelerated. Since much information is available in multiple reviews (sometimes running in thousands) for a single product, it can create decision paralysis for the buyer. This scenario disempowers the consumer, who cannot be expected to go over so many reviews since its time consuming and can confuse them. Various commercial tools are available, that use a scoring mechanism to arrive at an adjusted score. It can alert the user to potential review manipulations. This paper proposes a framework that fine-tunes a generative pre-trained transformer to understand these reviews better. Furthermore, using "common-sense" to make better decisions. These models have more than 13 billion parameters. To fine-tune the model for our requirement, we use the curie engine from generative pre-trained transformer (GPT3). By using generative models, we are introducing abstractive summarization. Instead of using a simple extractive method of summarizing the reviews. This brings out the true relationship between the reviews and not simply copy-paste. This introduces an element of "common sense" for the user and helps them to quickly make the right decisions. The user is provided the pros and cons of the processed reviews. Thus the user/customer can take their own decisions. 

**Abstract (ZH)**: 疫情之后，电子商务使用偏好加速增长：基于生成预训练变换器的综合评价框架与常识辅助决策 

---
# EmoMeta: A Multimodal Dataset for Fine-grained Emotion Classification in Chinese Metaphors 

**Title (ZH)**: EmoMeta: 中国隐喻中的细粒度情感分类多模态数据集 

**Authors**: Xingyuan Lu, Yuxi Liu, Dongyu Zhang, Zhiyao Wu, Jing Ren, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2505.13483)  

**Abstract**: Metaphors play a pivotal role in expressing emotions, making them crucial for emotional intelligence. The advent of multimodal data and widespread communication has led to a proliferation of multimodal metaphors, amplifying the complexity of emotion classification compared to single-mode scenarios. However, the scarcity of research on constructing multimodal metaphorical fine-grained emotion datasets hampers progress in this domain. Moreover, existing studies predominantly focus on English, overlooking potential variations in emotional nuances across languages. To address these gaps, we introduce a multimodal dataset in Chinese comprising 5,000 text-image pairs of metaphorical advertisements. Each entry is meticulously annotated for metaphor occurrence, domain relations and fine-grained emotion classification encompassing joy, love, trust, fear, sadness, disgust, anger, surprise, anticipation, and neutral. Our dataset is publicly accessible (this https URL), facilitating further advancements in this burgeoning field. 

**Abstract (ZH)**: 多模态隐喻在表达情感中的作用凸显，这对于情感智能至关重要。多模态数据和广泛交流的出现导致了多模态隐喻的增多，增加了情感分类的复杂性。然而，构建多模态隐喻细粒度情感数据集的研究较少，阻碍了该领域的进步。此外，现有研究主要集中在英语上，忽视了不同语言中情感细微差别的潜在差异。为弥补这些不足，我们介绍了一个包含5000个文本-图像对的多模态中文隐喻广告数据集，每个条目都详细标注了隐喻出现情况、领域关系以及细粒度情感分类，包括 joy、love、trust、fear、sadness、disgust、anger、surprise、anticipation 和 neutral。该数据集已公开（this https URL），有助于推动这一新兴领域的发展。 

---
# Evaluating Reasoning LLMs for Suicide Screening with the Columbia-Suicide Severity Rating Scale 

**Title (ZH)**: 评估 suicidality 筛查中基于推理的大型语言模型的表现——使用哥伦比亚自杀严重程度评定量表 

**Authors**: Avinash Patil, Siru Tao, Amardeep Gedhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13480)  

**Abstract**: Suicide prevention remains a critical public health challenge. While online platforms such as Reddit's r/SuicideWatch have historically provided spaces for individuals to express suicidal thoughts and seek community support, the advent of large language models (LLMs) introduces a new paradigm-where individuals may begin disclosing ideation to AI systems instead of humans. This study evaluates the capability of LLMs to perform automated suicide risk assessment using the Columbia-Suicide Severity Rating Scale (C-SSRS). We assess the zero-shot performance of six models-including Claude, GPT, Mistral, and LLaMA-in classifying posts across a 7-point severity scale (Levels 0-6). Results indicate that Claude and GPT closely align with human annotations, while Mistral achieves the lowest ordinal prediction error. Most models exhibit ordinal sensitivity, with misclassifications typically occurring between adjacent severity levels. We further analyze confusion patterns, misclassification sources, and ethical considerations, underscoring the importance of human oversight, transparency, and cautious deployment. Full code and supplementary materials are available at this https URL. 

**Abstract (ZH)**: 自杀预防仍然是一个重要公共卫生挑战。虽然Reddit的r/SuicideWatch等在线平台历来为个体提供了一个表达自杀念头并寻求社群支持的空间，但大型语言模型（LLMs）的出现引入了一个新的范式——个体可能开始向AI系统披露念头而非人类。本研究评估了LLMs使用哥伦比亚自杀严重程度评定量表（C-SSRS）进行自动化自杀风险评估的能力。我们评估了六种模型（包括Claude、GPT、Mistral和LLaMA）在将帖子分类到7级严重程度尺度（级别0-6）上的零样本性能。结果显示，Claude和GPT与人类标注最为一致，而Mistral的等级预测误差最低。大多数模型表现出对等级的敏感性，误分类通常发生在相邻严重程度级别之间。我们进一步分析了混淆模式、误分类来源和伦理考量，强调了人类监督、透明度和谨慎部署的重要性。完整代码和补充材料请参见此[链接]。 

---
# Algorithmic Tradeoffs in Fair Lending: Profitability, Compliance, and Long-Term Impact 

**Title (ZH)**: 算法权衡在公平信贷中的作用：盈利性、合规性和长期影响 

**Authors**: Aayam Bansal, Harsh Vardhan Narsaria  

**Link**: [PDF](https://arxiv.org/pdf/2505.13469)  

**Abstract**: As financial institutions increasingly rely on machine learning models to automate lending decisions, concerns about algorithmic fairness have risen. This paper explores the tradeoff between enforcing fairness constraints (such as demographic parity or equal opportunity) and maximizing lender profitability. Through simulations on synthetic data that reflects real-world lending patterns, we quantify how different fairness interventions impact profit margins and default rates. Our results demonstrate that equal opportunity constraints typically impose lower profit costs than demographic parity, but surprisingly, removing protected attributes from the model (fairness through unawareness) outperforms explicit fairness interventions in both fairness and profitability metrics. We further identify the specific economic conditions under which fair lending becomes profitable and analyze the feature-specific drivers of unfairness. These findings offer practical guidance for designing lending algorithms that balance ethical considerations with business objectives. 

**Abstract (ZH)**: 随着金融机构越来越依赖机器学习模型来自动化贷款决策，关于算法公平性的担忧日益增加。本文探讨了强制执行公平约束（如人口统计对等或同等机遇）与最大化贷方盈利能力之间的权衡。通过模拟反映真实世界贷款模式的合成数据，我们量化了不同公平干预措施对利润margin和违约率的影响。研究结果表明，同等机遇约束通常对利润的影响较小，而意外的是，从模型中移除保护性属性（无知公平）在公平性和盈利能力指标上均优于显式公平干预措施。我们进一步界定了使公平贷款变得盈利的具体经济条件，并分析了不公平性的特征特定驱动因素。这些发现为设计兼顾伦理考虑和商业目标的贷款算法提供了实用指导。 

---
# Exploring Emotional Synchrony in Dyadic Interactions: The Role of Speech Conditions in Facial and Vocal Affective Alignment 

**Title (ZH)**: 探索双向互动中的情绪同步：言语条件在面部和语音情感对齐中的作用 

**Authors**: Von Ralph Dane Marquez Herbuela, Yukie Nagai  

**Link**: [PDF](https://arxiv.org/pdf/2505.13455)  

**Abstract**: Understanding how humans express and synchronize emotions across multiple communication channels particularly facial expressions and speech has significant implications for emotion recognition systems and human computer interaction. Motivated by the notion that non-overlapping speech promotes clearer emotional coordination, while overlapping speech disrupts synchrony, this study examines how these conversational dynamics shape the spatial and temporal alignment of arousal and valence across facial and vocal modalities. Using dyadic interactions from the IEMOCAP dataset, we extracted continuous emotion estimates via EmoNet (facial video) and a Wav2Vec2-based model (speech audio). Segments were categorized based on speech overlap, and emotional alignment was assessed using Pearson correlation, lag adjusted analysis, and Dynamic Time Warping (DTW). Across analyses, non overlapping speech was associated with more stable and predictable emotional synchrony than overlapping speech. While zero-lag correlations were low and not statistically different, non overlapping speech showed reduced variability, especially for arousal. Lag adjusted correlations and best-lag distributions revealed clearer, more consistent temporal alignment in these segments. In contrast, overlapping speech exhibited higher variability and flatter lag profiles, though DTW indicated unexpectedly tighter alignment suggesting distinct coordination strategies. Notably, directionality patterns showed that facial expressions more often preceded speech during turn-taking, while speech led during simultaneous vocalizations. These findings underscore the importance of conversational structure in regulating emotional communication and provide new insight into the spatial and temporal dynamics of multimodal affective alignment in real world interaction. 

**Abstract (ZH)**: 理解人类在多通信渠道（尤其是面部表情和语音）中表达和同步情绪的方式对情绪识别系统和人机交互具有重要意义。受非重叠语音促进更清晰情绪协调、重叠语音破坏同步这一观念的驱动，本研究探讨了这些会话动态如何影响面部和语音模态之间唤醒度和价值取向的空间和时间对齐。通过使用IEMOCAP数据集中的二元互动，我们利用EmoNet（面部视频）和Wav2Vec2基于的模型（语音音频）提取了连续的情绪估计。根据语音重叠程度对段落进行分类，并使用皮尔逊相关系数、延迟调整分析和动态时间워킹（DTW）评估情绪对齐情况。分析结果显示，非重叠语音与更稳定和可预测的情绪同步相关，而重叠语音则不然。虽然零延迟相关系数较低且没有统计学差异，但非重叠语音显示出了减少的变异性，特别是在唤醒度方面。延迟调整相关系数和最佳延迟分布揭示了这些段落中更清晰、更一致的时间对齐。相比之下，重叠语音的变异性较高且延迟曲线较为平坦，尽管DTW表明出乎意料的紧密对齐，暗示了不同的协调策略。值得注意的是，方向模式显示，在轮流谈话时面部表情往往先于语音出现，而在同时发声时语音则领先。这些发现强调了会话结构在调节情绪沟通中的重要性，并提供了对多模态情绪对齐的空间和时间动态的新见解，特别是在真实世界互动中。 

---
# Pel, A Programming Language for Orchestrating AI Agents 

**Title (ZH)**: Pel：一种 orchestrating AI 代理的编程语言 

**Authors**: Behnam Mohammadi  

**Link**: [PDF](https://arxiv.org/pdf/2505.13453)  

**Abstract**: The proliferation of Large Language Models (LLMs) has opened new frontiers in computing, yet controlling and orchestrating their capabilities beyond simple text generation remains a challenge. Current methods, such as function/tool calling and direct code generation, suffer from limitations in expressiveness, scalability, cost, security, and the ability to enforce fine-grained control. This paper introduces Pel, a novel programming language specifically designed to bridge this gap. Inspired by the strengths of Lisp, Elixir, Gleam, and Haskell, Pel provides a syntactically simple, homoiconic, and semantically rich platform for LLMs to express complex actions, control flow, and inter-agent communication safely and efficiently. Pel's design emphasizes a minimal, easily modifiable grammar suitable for constrained LLM generation, eliminating the need for complex sandboxing by enabling capability control at the syntax level. Key features include a powerful piping mechanism for linear composition, first-class closures enabling easy partial application and functional patterns, built-in support for natural language conditions evaluated by LLMs, and an advanced Read-Eval-Print-Loop (REPeL) with Common Lisp-style restarts and LLM-powered helper agents for automated error correction. Furthermore, Pel incorporates automatic parallelization of independent operations via static dependency analysis, crucial for performant agentic systems. We argue that Pel offers a more robust, secure, and expressive paradigm for LLM orchestration, paving the way for more sophisticated and reliable AI agentic frameworks. 

**Abstract (ZH)**: 大型语言模型（LLMs）的 proliferate 为计算打开了新的前沿，但在简单文本生成之外控制和协调其能力仍然是一项挑战。当前的方法，如功能/工具调用和直接代码生成，受限于表达力、可扩展性、成本、安全性和细粒度控制能力。本文介绍了 Pel，一种专门为此差距设计的新型编程语言。Pel 受 Lisp、Elixir、Gleam 和 Haskell 的启发，提供了一个语法简单、同源且语义丰富的平台，使大型语言模型能够安全高效地表达复杂动作、控制流和代理间通信。Pel 的设计强调了一个简洁且易于修改的语法，适用于受限的大型语言模型生成，无需复杂的沙箱，因为其语法级别提供了能力控制。其关键特性包括强大的管道机制以实现线性组合、一等闭包以实现易用的偏应用和函数模式、由大型语言模型评估的内置自然语言条件支持，以及具有 Common Lisp 风格重启动的高级读-评价-打印-循环（REPeL）和由大型语言模型驱动的辅助代理，用于自动化错误纠正。此外，Pel 通过静态依赖分析自动并行化独立操作，这对于高性能代理系统至关重要。我们认为 Pel 提供了一种更稳健、更安全和更表达的大型语言模型协调范式，为更复杂和可靠的人工智能代理框架铺平了道路。 

---
# SLOT: Sample-specific Language Model Optimization at Test-time 

**Title (ZH)**: 测试时样本特定语言模型优化 

**Authors**: Yang Hu, Xingyu Zhang, Xueji Fang, Zhiyang Chen, Xiao Wang, Huatian Zhang, Guojun Qi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12392)  

**Abstract**: We propose SLOT (Sample-specific Language Model Optimization at Test-time), a novel and parameter-efficient test-time inference approach that enhances a language model's ability to more accurately respond to individual prompts. Existing Large Language Models (LLMs) often struggle with complex instructions, leading to poor performances on those not well represented among general samples. To address this, SLOT conducts few optimization steps at test-time to update a light-weight sample-specific parameter vector. It is added to the final hidden layer before the output head, and enables efficient adaptation by caching the last layer features during per-sample optimization. By minimizing the cross-entropy loss on the input prompt only, SLOT helps the model better aligned with and follow each given instruction. In experiments, we demonstrate that our method outperforms the compared models across multiple benchmarks and LLMs. For example, Qwen2.5-7B with SLOT achieves an accuracy gain of 8.6% on GSM8K from 57.54% to 66.19%, while DeepSeek-R1-Distill-Llama-70B with SLOT achieves a SOTA accuracy of 68.69% on GPQA among 70B-level models. Our code is available at this https URL. 

**Abstract (ZH)**: 基于测试时样本特定参数优化的语言模型测试时推理方法(SLOT) 

---
# LLM Context Conditioning and PWP Prompting for Multimodal Validation of Chemical Formulas 

**Title (ZH)**: LLM上下文条件及PWP提示在化学会计的多模态验证中的应用 

**Authors**: Evgeny Markhasin  

**Link**: [PDF](https://arxiv.org/pdf/2505.12257)  

**Abstract**: Identifying subtle technical errors within complex scientific and technical documents, especially those requiring multimodal interpretation (e.g., formulas in images), presents a significant hurdle for Large Language Models (LLMs) whose inherent error-correction tendencies can mask inaccuracies. This exploratory proof-of-concept (PoC) study investigates structured LLM context conditioning, informed by Persistent Workflow Prompting (PWP) principles, as a methodological strategy to modulate this LLM behavior at inference time. The approach is designed to enhance the reliability of readily available, general-purpose LLMs (specifically Gemini 2.5 Pro and ChatGPT Plus o3) for precise validation tasks, crucially relying only on their standard chat interfaces without API access or model modifications. To explore this methodology, we focused on validating chemical formulas within a single, complex test paper with known textual and image-based errors. Several prompting strategies were evaluated: while basic prompts proved unreliable, an approach adapting PWP structures to rigorously condition the LLM's analytical mindset appeared to improve textual error identification with both models. Notably, this method also guided Gemini 2.5 Pro to repeatedly identify a subtle image-based formula error previously overlooked during manual review, a task where ChatGPT Plus o3 failed in our tests. These preliminary findings highlight specific LLM operational modes that impede detail-oriented validation and suggest that PWP-informed context conditioning offers a promising and highly accessible technique for developing more robust LLM-driven analytical workflows, particularly for tasks requiring meticulous error detection in scientific and technical documents. Extensive validation beyond this limited PoC is necessary to ascertain broader applicability. 

**Abstract (ZH)**: 基于持续工作流提示原理的结构化LLM上下文条件化在精确验证任务中的探索性概念证明：识别复杂科学和技术文档中的细微技术错误，尤其是那些需要多模态解释的错误（例如图像中的公式），对大型语言模型（LLMs）构成了重大的挑战，因为它们固有的纠错倾向可能掩盖不准确之处。本探索性概念证明（PoC）研究探讨了基于持续工作流提示（PWP）原则的结构化LLM上下文条件化方法，作为一种在推理时调节LLM行为的方法论策略。该方法旨在通过仅依靠标准聊天界面（无需API访问或模型修改）来增强现成的通用语言模型（具体为Gemini 2.5 Pro和ChatGPT Plus o3）的可靠性，以执行精确验证任务。为了探索这一方法论，我们集中在验证一份包含已知文本和图像错误的复杂测试论文中的化学公式。评估了多种提示策略：尽管基本提示可靠性较低，但将PWP结构适应性地用于严格调节LLM的分析思维模式的方法在两个模型中都提高了文本错误识别的准确性。值得注意的是，这种方法还引导Gemini 2.5 Pro反复识别出在人工审查中未能发现的细微图像错误，而ChatGPT Plus o3在我们的测试中未能完成这一任务。初步发现强调了特定的LLM操作模式，这些模式阻碍了详细验证，并表明基于PWP的上下文条件化提供了一种有前景且高度可访问的技术，用于开发更加稳健的LLM驱动分析工作流，尤其是在需要在科学和技术文档中进行细致错误检测的任务方面。需要广泛的验证以确认其更广泛的适用性。 

---
# Uncertainty Quantification for Prior-Data Fitted Networks using Martingale Posteriors 

**Title (ZH)**: 基于鞅后验分布的先验-数据拟合网络的不确定性量化 

**Authors**: Thomas Nagler, David Rügamer  

**Link**: [PDF](https://arxiv.org/pdf/2505.11325)  

**Abstract**: Prior-data fitted networks (PFNs) have emerged as promising foundation models for prediction from tabular data sets, achieving state-of-the-art performance on small to moderate data sizes without tuning. While PFNs are motivated by Bayesian ideas, they do not provide any uncertainty quantification for predictive means, quantiles, or similar quantities. We propose a principled and efficient sampling procedure to construct Bayesian posteriors for such estimates based on Martingale posteriors, and prove its convergence. Several simulated and real-world data examples showcase the uncertainty quantification of our method in inference applications. 

**Abstract (ZH)**: 基于Martingale后验的Prior-data fitted网络的贝叶斯后验构建及其收敛性证明和不确定性量化 

---
# Model Steering: Learning with a Reference Model Improves Generalization Bounds and Scaling Laws 

**Title (ZH)**: 参考模型指导学习：改进泛化界限和标度定律 

**Authors**: Xiyuan Wei, Ming Lin, Fanjiang Ye, Fengguang Song, Liangliang Cao, My T. Thai, Tianbao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06699)  

**Abstract**: This paper formalizes an emerging learning paradigm that uses a trained model as a reference to guide and enhance the training of a target model through strategic data selection or weighting, named $\textbf{model steering}$. While ad-hoc methods have been used in various contexts, including the training of large foundation models, its underlying principles remain insufficiently understood, leading to sub-optimal performance. In this work, we propose a theory-driven framework for model steering called $\textbf{DRRho risk minimization}$, which is rooted in Distributionally Robust Optimization (DRO). Through a generalization analysis, we provide theoretical insights into why this approach improves generalization and data efficiency compared to training without a reference model. To the best of our knowledge, this is the first time such theoretical insights are provided for the new learning paradigm, which significantly enhance our understanding and practice of model steering. Building on these insights and the connection between contrastive learning and DRO, we introduce a novel method for Contrastive Language-Image Pretraining (CLIP) with a reference model, termed DRRho-CLIP. Extensive experiments validate the theoretical insights, reveal a superior scaling law compared to CLIP without a reference model, and demonstrate its strength over existing heuristic approaches. 

**Abstract (ZH)**: 这篇论文 formalizes 一种新兴的学习范式，通过战略数据选择或加权来指导和增强目标模型训练，使用训练好的模型作为参考，将其命名为 $\textbf{模型引导 (Model Steering)}$。虽然已经在这方面使用了各种 ad-hoc 方法，包括大型基础模型的训练，但其基本原理尚未得到充分理解，导致性能不佳。在本文中，我们提出了一个基于 Distributionally Robust Optimization (DRO) 的理论驱动框架，称为 $\textbf{DRRho 风险最小化 (DRRho Risk Minimization)}$，并通过泛化分析提供了有关为何此方法在没有参考模型的情况下训练时能提高泛化能力和数据效率的理论见解。据我们所知，这是首次为这一新兴学习范式提供此类理论见解，这对理解并实践模型引导有着重要意义。基于这些见解及其与对比学习和 DRO 之间的联系，我们提出了一种新的参考模型辅助的对比语言-图像预训练方法，称为 DRRho-CLIP。广泛的实验验证了这些理论见解，揭示了与没有参考模型的 CLIP 相比更优异的扩展规律，并展示了其在现有启发式方法上的优越性。标题：

模型引导：$\textbf{DRRho风险最小化框架及其在对比语言-图像预训练中的应用 (Model Steering: DRRho Risk Minimization Framework and Its Application in Contrastive Language-Image Pretraining)}$ 

---
