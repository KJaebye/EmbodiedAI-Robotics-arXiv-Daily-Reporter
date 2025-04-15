# LangPert: Detecting and Handling Task-level Perturbations for Robust Object Rearrangement 

**Title (ZH)**: LangPert：检测和处理任务级扰动以实现稳健的物体重新排列 

**Authors**: Xu Yin, Min-Sung Yoon, Yuchi Huo, Kang Zhang, Sung-Eui Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2504.09893)  

**Abstract**: Task execution for object rearrangement could be challenged by Task-Level Perturbations (TLP), i.e., unexpected object additions, removals, and displacements that can disrupt underlying visual policies and fundamentally compromise task feasibility and progress. To address these challenges, we present LangPert, a language-based framework designed to detect and mitigate TLP situations in tabletop rearrangement tasks. LangPert integrates a Visual Language Model (VLM) to comprehensively monitor policy's skill execution and environmental TLP, while leveraging the Hierarchical Chain-of-Thought (HCoT) reasoning mechanism to enhance the Large Language Model (LLM)'s contextual understanding and generate adaptive, corrective skill-execution plans. Our experimental results demonstrate that LangPert handles diverse TLP situations more effectively than baseline methods, achieving higher task completion rates, improved execution efficiency, and potential generalization to unseen scenarios. 

**Abstract (ZH)**: 基于语言的框架LangPert在桌面上对象重排任务中检测和缓解任务级扰动（TLP）的能力 

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
# MLRC-Bench: Can Language Agents Solve Machine Learning Research Challenges? 

**Title (ZH)**: MLRC-Bench: 语言代理能否解决机器学习研究挑战？ 

**Authors**: Yunxiang Zhang, Muhammad Khalifa, Shitanshu Bhushan, Grant D Murphy, Lajanugen Logeswaran, Jaekyeom Kim, Moontae Lee, Honglak Lee, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09702)  

**Abstract**: Existing evaluation of large language model (LLM) agents on scientific discovery lacks objective baselines and metrics to assess the viability of their proposed methods. To address this issue, we introduce MLRC-Bench, a benchmark designed to quantify how effectively language agents can tackle challenging Machine Learning (ML) Research Competitions. Our benchmark highlights open research problems that demand novel methodologies, in contrast to recent benchmarks such as OpenAI's MLE-Bench (Chan et al., 2024) and METR's RE-Bench (Wijk et al., 2024), which focus on well-established research tasks that are largely solvable through sufficient engineering effort. Unlike prior work, e.g., AI Scientist (Lu et al., 2024b), which evaluates the end-to-end agentic pipeline by using LLM-as-a-judge, MLRC-Bench measures the key steps of proposing and implementing novel research methods and evaluates them with newly proposed rigorous protocol and objective metrics. Our curated suite of 7 competition tasks reveals significant challenges for LLM agents. Even the best-performing tested agent (gemini-exp-1206 under MLAB (Huang et al., 2024a)) closes only 9.3% of the gap between baseline and top human participant scores. Furthermore, our analysis reveals a misalignment between the LLM-judged innovation and their actual performance on cutting-edge ML research problems. MLRC-Bench is a dynamic benchmark, which is designed to continually grow with new ML competitions to encourage rigorous and objective evaluations of AI's research capabilities. 

**Abstract (ZH)**: MLRC-Bench：评估大型语言模型在机器学习研究竞赛中提出新颖方法有效性基准 

---
# Building AI Service Repositories for On-Demand Service Orchestration in 6G AI-RAN 

**Title (ZH)**: 构建6G AI-RAN中按需服务编排的AI服务仓库 

**Authors**: Yun Tang, Mengbang Zou, Udhaya Chandhar Srinivasan, Obumneme Umealor, Dennis Kevogo, Benjamin James Scott, Weisi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2504.09647)  

**Abstract**: Efficient orchestration of AI services in 6G AI-RAN requires well-structured, ready-to-deploy AI service repositories combined with orchestration methods adaptive to diverse runtime contexts across radio access, edge, and cloud layers. Current literature lacks comprehensive frameworks for constructing such repositories and generally overlooks key practical orchestration factors. This paper systematically identifies and categorizes critical attributes influencing AI service orchestration in 6G networks and introduces an open-source, LLM-assisted toolchain that automates service packaging, deployment, and runtime profiling. We validate the proposed toolchain through the Cranfield AI Service repository case study, demonstrating significant automation benefits, reduced manual coding efforts, and the necessity of infrastructure-specific profiling, paving the way for more practical orchestration frameworks. 

**Abstract (ZH)**: 在6G AI-RAN中高效 orchestrating AI 服务需要结构良好且可部署的 AI 服务仓库，结合能够适应无线接入、边缘和云各层 diverse 运行时上下文的编排方法。当前文献缺乏构建此类仓库的综合框架，并且通常忽略关键的实践编排因素。本文系统地识别并分类了影响6G 网络中 AI 服务编排的关键属性，并介绍了一个开源的、基于大规模语言模型的工具链，该工具链自动完成服务打包、部署和运行时性能分析。我们通过 Cranfield AI 服务仓库案例研究验证了所提出的工具链，展示了显著的自动化优势、减少了手动编码工作，并强调了基础设施特定性能分析的必要性，从而为更实用的编排框架铺平道路。 

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
# Continuum-Interaction-Driven Intelligence: Human-Aligned Neural Architecture via Crystallized Reasoning and Fluid Generation 

**Title (ZH)**: 连续交互驱动的智能：通过结晶推理和流转型生成实现的人类对齐神经架构 

**Authors**: Pengcheng Zhou, Zhiqiang Nie, Haochen Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.09301)  

**Abstract**: Current AI systems based on probabilistic neural networks, such as large language models (LLMs), have demonstrated remarkable generative capabilities yet face critical challenges including hallucination, unpredictability, and misalignment with human decision-making. These issues fundamentally stem from the over-reliance on randomized (probabilistic) neural networks-oversimplified models of biological neural networks-while neglecting the role of procedural reasoning (chain-of-thought) in trustworthy decision-making. Inspired by the human cognitive duality of fluid intelligence (flexible generation) and crystallized intelligence (structured knowledge), this study proposes a dual-channel intelligent architecture that integrates probabilistic generation (LLMs) with white-box procedural reasoning (chain-of-thought) to construct interpretable, continuously learnable, and human-aligned AI systems. Concretely, this work: (1) redefines chain-of-thought as a programmable crystallized intelligence carrier, enabling dynamic knowledge evolution and decision verification through multi-turn interaction frameworks; (2) introduces a task-driven modular network design that explicitly demarcates the functional boundaries between randomized generation and procedural control to address trustworthiness in vertical-domain applications; (3) demonstrates that multi-turn interaction is a necessary condition for intelligence emergence, with dialogue depth positively correlating with the system's human-alignment degree. This research not only establishes a new paradigm for trustworthy AI deployment but also provides theoretical foundations for next-generation human-AI collaborative systems. 

**Abstract (ZH)**: 基于概率神经网络的当前人工智能系统，如大型语言模型（LLMs），展现了出色的生成能力，但仍面临幻觉、不可预测性和与人类决策不一致等关键挑战。这些问题从根本上来说源于过度依赖随机化（概率性）神经网络——对生物神经网络的简化模型——而忽视了程序化推理（链式思维）在可信决策中的作用。受人类认知双重性的启发，即流体智力（灵活生成）和晶体智力（结构化知识），本研究提出了一种双重通道智能架构，将概率生成（LLMs）与白盒程序化推理（链式思维）相结合，构建可解释、可持续学习且与人类目标一致的AI系统。具体而言，本研究：（1）将链式思维重新定义为可编程的晶体智力载体，通过多轮交互框架实现动态知识进化和决策验证；（2）引入了一种任务驱动的模块化网络设计，明确区分随机生成和程序控制的功能边界，以解决垂直应用领域的可信性问题；（3）证明多轮交互是智能涌现的必要条件，对话深度与系统的拟人化程度正相关。本研究不仅为可信AI部署奠定了新的范式，也为新一代人机协作系统提供了理论基础。 

---
# A Short Survey on Small Reasoning Models: Training, Inference, Applications and Research Directions 

**Title (ZH)**: 小型推理模型综述：训练、推理、应用与研究方向 

**Authors**: Chengyu Wang, Taolin Zhang, Richang Hong, Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09100)  

**Abstract**: Recently, the reasoning capabilities of large reasoning models (LRMs), such as DeepSeek-R1, have seen significant advancements through the slow thinking process. Despite these achievements, the substantial computational demands of LRMs present considerable challenges. In contrast, small reasoning models (SRMs), often distilled from larger ones, offer greater efficiency and can exhibit distinct capabilities and cognitive trajectories compared to LRMs. This work surveys around 170 recently published papers on SRMs for tackling various complex reasoning tasks. We review the current landscape of SRMs and analyze diverse training and inference techniques related to SRMs. Furthermore, we provide a comprehensive review of SRMs for domain-specific applications and discuss possible future research directions. This survey serves as an essential reference for researchers to leverage or develop SRMs for advanced reasoning functionalities with high efficiency. 

**Abstract (ZH)**: 最近，大型推理模型（LRMs）如DeepSeek-R1的推理能力通过慢思考过程取得了显著进步。尽管取得了这些成就，LRMs的大量计算需求仍提出了重大挑战。相比之下，从小型模型中提炼出来的小型推理模型（SRMs）提供了更高的效率，并在能力和认知轨迹方面表现出与LRMs不同的特点。本文回顾了约170篇近期发表的关于SRMs在各种复杂推理任务中的应用的论文，概述了当前SRMs的格局，并分析了与SRMs相关的多种训练和推理技术。此外，本文对SRMs在特定领域应用进行了全面回顾，并讨论了未来的研究方向。本综述为研究人员利用或开发具有高效性的SRMs以实现高级推理功能提供了重要的参考。 

---
# A Survey of Frontiers in LLM Reasoning: Inference Scaling, Learning to Reason, and Agentic Systems 

**Title (ZH)**: LLM推理前沿综述：推理扩展、学习推理与代理系统 

**Authors**: Zixuan Ke, Fangkai Jiao, Yifei Ming, Xuan-Phi Nguyen, Austin Xu, Do Xuan Long, Minzhi Li, Chengwei Qin, Peifeng Wang, Silvio Savarese, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2504.09037)  

**Abstract**: Reasoning is a fundamental cognitive process that enables logical inference, problem-solving, and decision-making. With the rapid advancement of large language models (LLMs), reasoning has emerged as a key capability that distinguishes advanced AI systems from conventional models that empower chatbots. In this survey, we categorize existing methods along two orthogonal dimensions: (1) Regimes, which define the stage at which reasoning is achieved (either at inference time or through dedicated training); and (2) Architectures, which determine the components involved in the reasoning process, distinguishing between standalone LLMs and agentic compound systems that incorporate external tools, and multi-agent collaborations. Within each dimension, we analyze two key perspectives: (1) Input level, which focuses on techniques that construct high-quality prompts that the LLM condition on; and (2) Output level, which methods that refine multiple sampled candidates to enhance reasoning quality. This categorization provides a systematic understanding of the evolving landscape of LLM reasoning, highlighting emerging trends such as the shift from inference-scaling to learning-to-reason (e.g., DeepSeek-R1), and the transition to agentic workflows (e.g., OpenAI Deep Research, Manus Agent). Additionally, we cover a broad spectrum of learning algorithms, from supervised fine-tuning to reinforcement learning such as PPO and GRPO, and the training of reasoners and verifiers. We also examine key designs of agentic workflows, from established patterns like generator-evaluator and LLM debate to recent innovations. ... 

**Abstract (ZH)**: 大型语言模型的推理是认知过程的核心，它支持逻辑推理、问题解决和决策制定。随着大型语言模型（LLMs）的迅速发展，推理已成为高级AI系统与传统聊天机器人所依赖的常规模型区分开来的关键技术能力。本文综述中，我们将现有的方法按照两个正交维度进行分类：（1）阶段，定义推理是在推断时实现还是通过专门训练获得；（2）架构，确定推理过程中涉及的组件，区分独立的大型语言模型和整合外部工具及多代理协作的自主系统。在每个维度中，我们分析了两个关键视角：（1）输入层面，关注构建高质量提示的技术，这些提示使LLM能够理解；（2）输出层面，关注方法如何细化多个采样候选方案以提高推理质量。这种分类提供了对大型语言模型推理演进格局的系统理解，突显出从推理扩展到学习推理（如DeepSeek-R1）和从非自主型工作流程到自主型工作流程转变等新兴趋势。此外，本文还涵盖了从监督微调到强化学习（如PPO和GRPO）等各种学习算法，以及推理者和验证者的训练。我们还考察了自主工作流程的关键设计，从生成器-评估器模式和LLM辩论到最近的创新。 

---
# Weight Ensembling Improves Reasoning in Language Models 

**Title (ZH)**: Weight Ensemble improves Reasoning in Language Models 

**Authors**: Xingyu Dang, Christina Baek, Kaiyue Wen, Zico Kolter, Aditi Raghunathan  

**Link**: [PDF](https://arxiv.org/pdf/2504.10478)  

**Abstract**: We investigate a failure mode that arises during the training of reasoning models, where the diversity of generations begins to collapse, leading to suboptimal test-time scaling. Notably, the Pass@1 rate reliably improves during supervised finetuning (SFT), but Pass@k rapidly deteriorates. Surprisingly, a simple intervention of interpolating the weights of the latest SFT checkpoint with an early checkpoint, otherwise known as WiSE-FT, almost completely recovers Pass@k while also improving Pass@1. The WiSE-FT variant achieves better test-time scaling (Best@k, majority vote) and achieves superior results with less data when tuned further by reinforcement learning. Finally, we find that WiSE-FT provides complementary performance gains that cannot be achieved only through diversity-inducing decoding strategies, like temperature scaling. We formalize a bias-variance tradeoff of Pass@k with respect to the expectation and variance of Pass@1 over the test distribution. We find that WiSE-FT can reduce bias and variance simultaneously, while temperature scaling inherently trades-off between bias and variance. 

**Abstract (ZH)**: 我们研究了推理模型训练过程中出现的一种故障模式，其中生成的多样性开始崩溃，导致测验时扩展性变差。值得注意的是，监督微调（SFT）过程中Pass@1率可靠地得到了提高，但Pass@k迅速恶化。令人惊讶的是，通过将最新SFT检查点的权重与早期检查点的权重进行插值（即WiSE-FT）的简单干预，几乎完全恢复了Pass@k，同时提高了Pass@1。WiSE-FT变体在最佳测验时扩展性（Best@k，多数投票）方面表现更佳，且在进一步通过强化学习调整时可获得更好的结果。最后，我们发现WiSE-FT提供了补充性的性能增益，这些增益无法仅通过诱发多样性的解码策略（如温度缩放）来实现。我们将Pass@k的偏差-方差权衡形式化为Pass@1在测试分布上的期望和方差之间的关系。我们发现WiSE-FT可以同时减少偏差和方差，而温度缩放本质上是在偏差和方差之间进行权衡。 

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
# SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning 

**Title (ZH)**: SymRTLO：通过LLM和神经元启发的符号推理增强RTL代码优化 

**Authors**: Yiting Wang, Wanghao Ye, Ping Guo, Yexiao He, Ziyao Wang, Yexiao He, Bowei Tian, Shwai He, Guoheng Sun, Zheyu Shen, Sihan Chen, Ankur Srivastava, Qingfu Zhang, Gang Qu, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.10369)  

**Abstract**: Optimizing Register Transfer Level (RTL) code is crucial for improving the power, performance, and area (PPA) of digital circuits in the early stages of synthesis. Manual rewriting, guided by synthesis feedback, can yield high-quality results but is time-consuming and error-prone. Most existing compiler-based approaches have difficulty handling complex design constraints. Large Language Model (LLM)-based methods have emerged as a promising alternative to address these challenges. However, LLM-based approaches often face difficulties in ensuring alignment between the generated code and the provided prompts. This paper presents SymRTLO, a novel neuron-symbolic RTL optimization framework that seamlessly integrates LLM-based code rewriting with symbolic reasoning techniques. Our method incorporates a retrieval-augmented generation (RAG) system of optimization rules and Abstract Syntax Tree (AST)-based templates, enabling LLM-based rewriting that maintains syntactic correctness while minimizing undesired circuit behaviors. A symbolic module is proposed for analyzing and optimizing finite state machine (FSM) logic, allowing fine-grained state merging and partial specification handling beyond the scope of pattern-based compilers. Furthermore, a fast verification pipeline, combining formal equivalence checks with test-driven validation, further reduces the complexity of verification. Experiments on the RTL-Rewriter benchmark with Synopsys Design Compiler and Yosys show that SymRTLO improves power, performance, and area (PPA) by up to 43.9%, 62.5%, and 51.1%, respectively, compared to the state-of-the-art methods. 

**Abstract (ZH)**: SymRTLO：一种新颖的神经符号RTL优化框架 

---
# Characterizing LLM-driven Social Network: The Chirper.ai Case 

**Title (ZH)**: LLM驱动的社会网络-characterizer: Chirper.ai案例研究 

**Authors**: Yiming Zhu, Yupeng He, Ehsan-Ul Haq, Gareth Tyson, Pan Hui  

**Link**: [PDF](https://arxiv.org/pdf/2504.10286)  

**Abstract**: Large language models (LLMs) demonstrate the ability to simulate human decision-making processes, enabling their use as agents in modeling sophisticated social networks, both offline and online. Recent research has explored collective behavioral patterns and structural characteristics of LLM agents within simulated networks. However, empirical comparisons between LLM-driven and human-driven online social networks remain scarce, limiting our understanding of how LLM agents differ from human users. This paper presents a large-scale analysis of this http URL, an X/Twitter-like social network entirely populated by LLM agents, comprising over 65,000 agents and 7.7 million AI-generated posts. For comparison, we collect a parallel dataset from Mastodon, a human-driven decentralized social network, with over 117,000 users and 16 million posts. We examine key differences between LLM agents and humans in posting behaviors, abusive content, and social network structures. Our findings provide critical insights into the evolving landscape of online social network analysis in the AI era, offering a comprehensive profile of LLM agents in social simulations. 

**Abstract (ZH)**: 大型语言模型在模拟社交网络中的集体行为模式与结构特征：基于this http URL与Mastodon的大规模分析 

---
# RealHarm: A Collection of Real-World Language Model Application Failures 

**Title (ZH)**: 实锤：语言模型实际应用中的失败案例集锦 

**Authors**: Pierre Le Jeune, Jiaen Liu, Luca Rossi, Matteo Dora  

**Link**: [PDF](https://arxiv.org/pdf/2504.10277)  

**Abstract**: Language model deployments in consumer-facing applications introduce numerous risks. While existing research on harms and hazards of such applications follows top-down approaches derived from regulatory frameworks and theoretical analyses, empirical evidence of real-world failure modes remains underexplored. In this work, we introduce RealHarm, a dataset of annotated problematic interactions with AI agents built from a systematic review of publicly reported incidents. Analyzing harms, causes, and hazards specifically from the deployer's perspective, we find that reputational damage constitutes the predominant organizational harm, while misinformation emerges as the most common hazard category. We empirically evaluate state-of-the-art guardrails and content moderation systems to probe whether such systems would have prevented the incidents, revealing a significant gap in the protection of AI applications. 

**Abstract (ZH)**: 面向消费者的语言模型部署引入了诸多风险。尽管现有研究从监管框架和理论分析出发，采用自上而下的方法探讨这类应用的危害和风险，但实际应用场景中失败模式的实证证据仍较为缺乏。在本工作中，我们介绍了RealHarm数据集，该数据集基于系统性回顾公开报告的事件而构建，并对与AI代理互动中的问题进行标注。从部署者的视角分析危害、成因和风险，我们发现声誉损害是主要的企业危害，而错误信息是最常见的风险类别。我们实证评估了最先进的防护栏和内容审核系统，探讨这些系统是否能够预防这些事件，揭示了AI应用保护方面的显著差距。 

---
# Localized Cultural Knowledge is Conserved and Controllable in Large Language Models 

**Title (ZH)**: 局部文化知识在大规模语言模型中得以保留和可控 

**Authors**: Veniamin Veselovsky, Berke Argin, Benedikt Stroebl, Chris Wendler, Robert West, James Evans, Thomas L. Griffiths, Arvind Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2504.10191)  

**Abstract**: Just as humans display language patterns influenced by their native tongue when speaking new languages, LLMs often default to English-centric responses even when generating in other languages. Nevertheless, we observe that local cultural information persists within the models and can be readily activated for cultural customization. We first demonstrate that explicitly providing cultural context in prompts significantly improves the models' ability to generate culturally localized responses. We term the disparity in model performance with versus without explicit cultural context the explicit-implicit localization gap, indicating that while cultural knowledge exists within LLMs, it may not naturally surface in multilingual interactions if cultural context is not explicitly provided. Despite the explicit prompting benefit, however, the answers reduce in diversity and tend toward stereotypes. Second, we identify an explicit cultural customization vector, conserved across all non-English languages we explore, which enables LLMs to be steered from the synthetic English cultural world-model toward each non-English cultural world. Steered responses retain the diversity of implicit prompting and reduce stereotypes to dramatically improve the potential for customization. We discuss the implications of explicit cultural customization for understanding the conservation of alternative cultural world models within LLMs, and their controllable utility for translation, cultural customization, and the possibility of making the explicit implicit through soft control for expanded LLM function and appeal. 

**Abstract (ZH)**: 正如人类在使用新语言时会受到母语语言模式的影响，大型语言模型在生成其他语言的内容时往往默认使用以英语为中心的回答方式。然而，我们观察到本地文化信息仍然存在于这些模型中，并且可以轻松激活以实现文化定制。我们首先证明，在提示中明确提供文化背景可以显著提高模型生成文化本地化回答的能力。我们将模型在有和没有明确文化背景的情况下性能差距称为显性-隐性本地化差距，表明虽然文化知识存在于大型语言模型中，但如果未提供明确的文化背景，这些知识可能不会自然地在多语言交互中显现出来。尽管使用明确提示可以带来好处，但回答的多样性会减少，并倾向于形成刻板印象。其次，我们发现了一种适用于所有探索的非英语语言的显性文化定制向量，这种向量能够引导大型语言模型从合成的英语文化世界观转向每个非英语文化的世界。经过引导的回应保留了隐性提示的多样性，并减少了刻板印象，从而显著提高了定制的可能性。我们讨论了显性文化定制对理解大型语言模型中替代文化世界观的保守性及其可控实用性的含义，以及通过软控制实现扩展的大型语言模型功能和吸引力的可能性。 

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
# MT-R1-Zero: Advancing LLM-based Machine Translation via R1-Zero-like Reinforcement Learning 

**Title (ZH)**: MT-R1-Zero: 基于R1-Zero-like强化学习推进基于LLM的机器翻译 

**Authors**: Zhaopeng Feng, Shaosheng Cao, Jiahan Ren, Jiayuan Su, Ruizhe Chen, Yan Zhang, Zhe Xu, Yao Hu, Jian Wu, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.10160)  

**Abstract**: Large-scale reinforcement learning (RL) methods have proven highly effective in enhancing the reasoning abilities of large language models (LLMs), particularly for tasks with verifiable solutions such as mathematics and coding. However, applying this idea to machine translation (MT), where outputs are flexibly formatted and difficult to automatically evaluate with explicit rules, remains underexplored. In this work, we introduce MT-R1-Zero, the first open-source adaptation of the R1-Zero RL framework for MT without supervised fine-tuning or cold-start. We propose a rule-metric mixed reward mechanism to guide LLMs towards improved translation quality via emergent reasoning. On the WMT 24 English-Chinese benchmark, our MT-R1-Zero-3B-Mix achieves competitive performance, surpassing TowerInstruct-7B-v0.2 by an average of 1.26 points. Meanwhile, our MT-R1-Zero-7B-Mix attains a high average score of 62.25 across all metrics, placing it on par with advanced proprietary models such as GPT-4o and Claude-3.5-Sonnet, while the MT-R1-Zero-7B-Sem variant achieves state-of-the-art scores on semantic metrics. Moreover, our work exhibits strong generalization capabilities on out-of-distribution MT tasks, robustly supporting multilingual and low-resource settings. Extensive analysis of model behavior across different initializations and reward metrics offers pioneering insight into the critical role of reward design, LLM adaptability, training dynamics, and emergent reasoning patterns within the R1-Zero paradigm for MT. Our code is available at this https URL. 

**Abstract (ZH)**: 大规模强化学习方法在提升大型语言模型的机器翻译性能方面取得了显著效果，特别是在数学和编程等具有可验证解决方案的任务中。然而，将这一理念应用于机器翻译（MT），其中输出的格式灵活且难以通过显式规则自动评估，这一领域仍然有待探索。在本研究中，我们介绍了MT-R1-Zero，这是第一个无需监督微调或冷启动的情况下，将R1-Zero RL框架应用于MT的开源适应版本。我们提出了一种规则度量混合奖励机制，通过新兴推理引导LLM提高翻译质量。在WMT 24英中基准测试中，我们的MT-R1-Zero-3B-Mix实现了竞争力的表现，平均优于TowerInstruct-7B-v0.2 1.26分。同时，我们的MT-R1-Zero-7B-Mix在所有指标上的平均得分为62.25，与GPT-4o和Claude-3.5-Sonnet等先进的专有模型不相上下，MT-R1-Zero-7B-Sem变体在语义指标上取得了最先进的得分。此外，我们的工作在离分布机器翻译任务中展示了强大的泛化能力，支持多语言和低资源设置。对不同初始化和奖励机制下模型行为的广泛分析提供了对奖励设计、LLM适应性、训练动力学和R1-Zero范式下新兴推理模式在MT中的关键作用的开创性见解。我们的代码可在以下链接获取。 

---
# Benchmarking Practices in LLM-driven Offensive Security: Testbeds, Metrics, and Experiment Design 

**Title (ZH)**: 基于LLM驱动的进攻性安全实践的基准测试：测试平台、评估指标与实验设计 

**Authors**: Andreas Happe, Jürgen Cito  

**Link**: [PDF](https://arxiv.org/pdf/2504.10112)  

**Abstract**: Large Language Models (LLMs) have emerged as a powerful approach for driving offensive penetration-testing tooling. This paper analyzes the methodology and benchmarking practices used for evaluating Large Language Model (LLM)-driven attacks, focusing on offensive uses of LLMs in cybersecurity. We review 16 research papers detailing 15 prototypes and their respective testbeds.
We detail our findings and provide actionable recommendations for future research, emphasizing the importance of extending existing testbeds, creating baselines, and including comprehensive metrics and qualitative analysis. We also note the distinction between security research and practice, suggesting that CTF-based challenges may not fully represent real-world penetration testing scenarios. 

**Abstract (ZH)**: 大型语言模型（LLMs）已成为驱动进攻性渗透测试工具的强大方法。本文分析了评估大型语言模型（LLM）驱动攻击的方法学和基准测试实践，重点关注LLMs在网络安全中的进攻性使用。我们回顾了16篇研究论文，详细介绍了15个原型及其各自的实验平台。 

---
# Towards Quantifying Commonsense Reasoning with Mechanistic Insights 

**Title (ZH)**: 基于机制洞见量化常识推理能力 

**Authors**: Abhinav Joshi, Areeb Ahmad, Divyaksh Shukla, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2504.10077)  

**Abstract**: Commonsense reasoning deals with the implicit knowledge that is well understood by humans and typically acquired via interactions with the world. In recent times, commonsense reasoning and understanding of various LLMs have been evaluated using text-based tasks. In this work, we argue that a proxy of this understanding can be maintained as a graphical structure that can further help to perform a rigorous evaluation of commonsense reasoning abilities about various real-world activities. We create an annotation scheme for capturing this implicit knowledge in the form of a graphical structure for 37 daily human activities. We find that the created resource can be used to frame an enormous number of commonsense queries (~ 10^{17}), facilitating rigorous evaluation of commonsense reasoning in LLMs. Moreover, recently, the remarkable performance of LLMs has raised questions about whether these models are truly capable of reasoning in the wild and, in general, how reasoning occurs inside these models. In this resource paper, we bridge this gap by proposing design mechanisms that facilitate research in a similar direction. Our findings suggest that the reasoning components are localized in LLMs that play a prominent role in decision-making when prompted with a commonsense query. 

**Abstract (ZH)**: 常识推理处理人类普遍理解和通常通过与世界交互获取的隐含知识。近年来，各种大规模语言模型的常识推理和理解能力多通过文本任务进行评估。在本工作中，我们argue rằng这种理解可以通过图形结构来维持，进一步帮助进行常识推理能力的严格评估，关于各种现实生活活动。我们创建了一种标注方案，将以图形结构形式捕获37项日常人类活动中的隐含知识。我们发现，所创建的资源可用来构建大量常识查询（~10^17），促进对大规模语言模型常识推理能力的严格评估。此外，最近大规模语言模型的出色性能引发了对其是否真正能够在现实世界中进行推理以及如何在模型内部进行推理的一系列疑问。在本资源论文中，我们通过提出促进类似方向研究的设计机制来弥合这一差距。我们的发现表明，当面对常识查询时，LLMs中在决策过程中发挥重要作用的推理组件是局部化的。 

---
# Hallucination Detection in LLMs via Topological Divergence on Attention Graphs 

**Title (ZH)**: LLMs中基于注意力图拓扑发散的幻觉检测 

**Authors**: Alexandra Bazarova, Aleksandr Yugay, Andrey Shulga, Alina Ermilova, Andrei Volodichev, Konstantin Polev, Julia Belikova, Rauf Parchiev, Dmitry Simakov, Maxim Savchenko, Andrey Savchenko, Serguei Barannikov, Alexey Zaytsev  

**Link**: [PDF](https://arxiv.org/pdf/2504.10063)  

**Abstract**: Hallucination, i.e., generating factually incorrect content, remains a critical challenge for large language models (LLMs). We introduce TOHA, a TOpology-based HAllucination detector in the RAG setting, which leverages a topological divergence metric to quantify the structural properties of graphs induced by attention matrices. Examining the topological divergence between prompt and response subgraphs reveals consistent patterns: higher divergence values in specific attention heads correlate with hallucinated outputs, independent of the dataset. Extensive experiments, including evaluation on question answering and data-to-text tasks, show that our approach achieves state-of-the-art or competitive results on several benchmarks, two of which were annotated by us and are being publicly released to facilitate further research. Beyond its strong in-domain performance, TOHA maintains remarkable domain transferability across multiple open-source LLMs. Our findings suggest that analyzing the topological structure of attention matrices can serve as an efficient and robust indicator of factual reliability in LLMs. 

**Abstract (ZH)**: 基于拓扑的幻觉检测器TOHA：在RAG设置中的幻觉检测 

---
# The Mirage of Performance Gains: Why Contrastive Decoding Fails to Address Multimodal Hallucination 

**Title (ZH)**: 性能提升的幻影：对比解码为何无法解决多模态幻觉 

**Authors**: Hao Yin, Gunagzong Si, Zilei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10020)  

**Abstract**: Contrastive decoding strategies are widely used to reduce hallucinations in multimodal large language models (MLLMs). These methods work by constructing contrastive samples to induce hallucinations and then suppressing them in the output distribution. However, this paper demonstrates that such approaches fail to effectively mitigate the hallucination problem. The performance improvements observed on POPE Benchmark are largely driven by two misleading factors: (1) crude, unidirectional adjustments to the model's output distribution and (2) the adaptive plausibility constraint, which reduces the sampling strategy to greedy search. To further illustrate these issues, we introduce a series of spurious improvement methods and evaluate their performance against contrastive decoding techniques. Experimental results reveal that the observed performance gains in contrastive decoding are entirely unrelated to its intended goal of mitigating hallucinations. Our findings challenge common assumptions about the effectiveness of contrastive decoding strategies and pave the way for developing genuinely effective solutions to hallucinations in MLLMs. 

**Abstract (ZH)**: 对比解码策略在多模态大型语言模型中广泛应用以减少幻觉现象，但这类方法未能有效缓解幻觉问题。POPE基准上的性能提升主要由两个误导性因素驱动：（1）模型输出分布的一系列粗略且单向调整，（2）自适应可行性约束，将采样策略简化为贪婪搜索。为进一步说明这些问题，我们引入了一系列虚假改进方法，并评估其性能与对比解码技术的对比结果。实验结果显示，对比解码策略观察到的性能提升与减轻幻觉的目标完全无关。我们的发现质疑了对比解码策略有效性的常见假设，并为开发真正有效的多模态大型语言模型幻觉解决方案铺平了道路。 

---
# Do We Really Need Curated Malicious Data for Safety Alignment in Multi-modal Large Language Models? 

**Title (ZH)**: 我们真的需要精心挑选的恶意数据来实现多模态大型语言模型的安全对齐吗？ 

**Authors**: Yanbo Wang, Jiyang Guan, Jian Liang, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2504.10000)  

**Abstract**: Multi-modal large language models (MLLMs) have made significant progress, yet their safety alignment remains limited. Typically, current open-source MLLMs rely on the alignment inherited from their language module to avoid harmful generations. However, the lack of safety measures specifically designed for multi-modal inputs creates an alignment gap, leaving MLLMs vulnerable to vision-domain attacks such as typographic manipulation. Current methods utilize a carefully designed safety dataset to enhance model defense capability, while the specific knowledge or patterns acquired from the high-quality dataset remain unclear. Through comparison experiments, we find that the alignment gap primarily arises from data distribution biases, while image content, response quality, or the contrastive behavior of the dataset makes little contribution to boosting multi-modal safety. To further investigate this and identify the key factors in improving MLLM safety, we propose finetuning MLLMs on a small set of benign instruct-following data with responses replaced by simple, clear rejection sentences. Experiments show that, without the need for labor-intensive collection of high-quality malicious data, model safety can still be significantly improved, as long as a specific fraction of rejection data exists in the finetuning set, indicating the security alignment is not lost but rather obscured during multi-modal pretraining or instruction finetuning. Simply correcting the underlying data bias could narrow the safety gap in the vision domain. 

**Abstract (ZH)**: 多模态大型语言模型的安全对齐进展有限：数据偏差是主要因素 

---
# Enhancing Multi-task Learning Capability of Medical Generalist Foundation Model via Image-centric Multi-annotation Data 

**Title (ZH)**: 通过基于图像的多标注数据增强医疗全科基础模型的多任务学习能力 

**Authors**: Xun Zhu, Fanbin Mo, Zheng Zhang, Jiaxi Wang, Yiming Shi, Ming Wu, Chuang Zhang, Miao Li, Ji Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.09967)  

**Abstract**: The emergence of medical generalist foundation models has revolutionized conventional task-specific model development paradigms, aiming to better handle multiple tasks through joint training on large-scale medical datasets. However, recent advances prioritize simple data scaling or architectural component enhancement, while neglecting to re-examine multi-task learning from a data-centric perspective. Critically, simply aggregating existing data resources leads to decentralized image-task alignment, which fails to cultivate comprehensive image understanding or align with clinical needs for multi-dimensional image interpretation. In this paper, we introduce the image-centric multi-annotation X-ray dataset (IMAX), the first attempt to enhance the multi-task learning capabilities of medical multi-modal large language models (MLLMs) from the data construction level. To be specific, IMAX is featured from the following attributes: 1) High-quality data curation. A comprehensive collection of more than 354K entries applicable to seven different medical tasks. 2) Image-centric dense annotation. Each X-ray image is associated with an average of 4.10 tasks and 7.46 training entries, ensuring multi-task representation richness per image. Compared to the general decentralized multi-annotation X-ray dataset (DMAX), IMAX consistently demonstrates significant multi-task average performance gains ranging from 3.20% to 21.05% across seven open-source state-of-the-art medical MLLMs. Moreover, we investigate differences in statistical patterns exhibited by IMAX and DMAX training processes, exploring potential correlations between optimization dynamics and multi-task performance. Finally, leveraging the core concept of IMAX data construction, we propose an optimized DMAX-based training strategy to alleviate the dilemma of obtaining high-quality IMAX data in practical scenarios. 

**Abstract (ZH)**: 面向图像的多注释X射线数据集（IMAX）：从数据构建层面增强医学多模态大型语言模型的多任务学习能力 

---
# Privacy Meets Explainability: Managing Confidential Data and Transparency Policies in LLM-Empowered Science 

**Title (ZH)**: 隐私与可解释性相遇：LLM赋能科学中的保密数据管理与透明政策实现 

**Authors**: Yashothara Shanmugarasa, Shidong Pan, Ming Ding, Dehai Zhao, Thierry Rakotoarivelo  

**Link**: [PDF](https://arxiv.org/pdf/2504.09961)  

**Abstract**: As Large Language Models (LLMs) become integral to scientific workflows, concerns over the confidentiality and ethical handling of confidential data have emerged. This paper explores data exposure risks through LLM-powered scientific tools, which can inadvertently leak confidential information, including intellectual property and proprietary data, from scientists' perspectives. We propose "DataShield", a framework designed to detect confidential data leaks, summarize privacy policies, and visualize data flow, ensuring alignment with organizational policies and procedures. Our approach aims to inform scientists about data handling practices, enabling them to make informed decisions and protect sensitive information. Ongoing user studies with scientists are underway to evaluate the framework's usability, trustworthiness, and effectiveness in tackling real-world privacy challenges. 

**Abstract (ZH)**: 大型语言模型（LLMs）在科学工作流程中的应用引发了对保密数据和伦理处理的担忧。本文探讨了由LLM支持的科学工具带来的数据暴露风险，从科学家的角度分析了无意中泄露机密信息（包括知识产权和专有数据）的可能性。我们提出了“DataShield”框架，该框架旨在检测机密数据泄露、总结隐私政策并可视化数据流，确保与组织政策和程序的吻合。我们的方法旨在告知科学家有关数据处理的做法，帮助他们做出知情决策并保护敏感信息。正在进行的科学家用户研究旨在评估该框架的可用性、可信度及其在解决实际隐私挑战方面的有效性。 

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
# Learning from Reference Answers: Versatile Language Model Alignment without Binary Human Preference Data 

**Title (ZH)**: 参考答案指导的学习：无二元人类偏好数据的多功能语言模型对齐 

**Authors**: Shuai Zhao, Linchao Zhu, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09895)  

**Abstract**: Large language models~(LLMs) are expected to be helpful, harmless, and honest. In various alignment scenarios, such as general human preference, safety, and confidence alignment, binary preference data collection and reward modeling are resource-intensive but necessary for human preference transferring. In this work, we explore using the similarity between sampled generations and high-quality reference answers as an alternative reward function for LLM alignment. Using similarity as a reward circumvents training reward models, and collecting a single reference answer potentially costs less time than constructing binary preference pairs when multiple candidates are available. Specifically, we develop \textit{RefAlign}, a versatile REINFORCE-style alignment algorithm, which is free of reference and reward models. Instead, RefAlign utilizes BERTScore between sampled generations and high-quality reference answers as the surrogate reward. Beyond general human preference optimization, RefAlign can be readily extended to diverse scenarios, such as safety and confidence alignment, by incorporating the similarity reward with task-related objectives. In various scenarios, {RefAlign} demonstrates comparable performance to previous alignment methods while offering high efficiency. 

**Abstract (ZH)**: 大规模语言模型（LLMs）期望具备助益性、无害性和诚实性。在一般人类偏好、安全和信心对齐等各类对齐场景中，二元偏好数据收集和奖励建模虽然资源密集但必不可少。为此，我们探索使用生成样本与高质参考答案之间的相似性作为LLM对齐的替代奖励函数。使用相似性作为奖励可以避免训练奖励模型，并且在多个候选方案可用时，收集单个参考答案可能比构造二元偏好对节省更多时间。具体地，我们开发了RefAlign，这是一种通用的REINFORCE风格对齐算法，无需参考模型和奖励模型。相反，RefAlign利用BERTScore衡量生成样本与高质参考答案之间的相似性作为替代奖励。除了通用人类偏好的优化，RefAlign还可以通过结合相似性奖励和相关任务目标，轻松扩展应用于各种场景，如安全和信心对齐。在各类场景中，RefAlign在保持较高效率的同时，表现出与先前对齐方法相当的性能。 

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
# StruPhantom: Evolutionary Injection Attacks on Black-Box Tabular Agents Powered by Large Language Models 

**Title (ZH)**: StruPhantom：由大型语言模型驱动的黑盒表格式代理的进化注入攻击 

**Authors**: Yang Feng, Xudong Pan  

**Link**: [PDF](https://arxiv.org/pdf/2504.09841)  

**Abstract**: The proliferation of autonomous agents powered by large language models (LLMs) has revolutionized popular business applications dealing with tabular data, i.e., tabular agents. Although LLMs are observed to be vulnerable against prompt injection attacks from external data sources, tabular agents impose strict data formats and predefined rules on the attacker's payload, which are ineffective unless the agent navigates multiple layers of structural data to incorporate the payload. To address the challenge, we present a novel attack termed StruPhantom which specifically targets black-box LLM-powered tabular agents. Our attack designs an evolutionary optimization procedure which continually refines attack payloads via the proposed constrained Monte Carlo Tree Search augmented by an off-topic evaluator. StruPhantom helps systematically explore and exploit the weaknesses of target applications to achieve goal hijacking. Our evaluation validates the effectiveness of StruPhantom across various LLM-based agents, including those on real-world platforms, and attack scenarios. Our attack achieves over 50% higher success rates than baselines in enforcing the application's response to contain phishing links or malicious codes. 

**Abstract (ZH)**: 基于大规模语言模型的自主代理普及化已革命性地改变了处理表格数据的流行商业应用，即表格代理。尽管观察到大规模语言模型对外部数据来源的提示注入攻击易受攻击，但表格代理对攻击载荷施加了严格的数据格式和预定义规则，除非代理导航多层结构数据以整合载荷，这些规则才无效。为应对这一挑战，我们提出了一种新型攻击方法，称为StruPhantom，专门针对黑盒的大规模语言模型驱动的表格代理。我们的攻击设计了一种基于受约束的蒙特卡洛树搜索的进化优化程序，该程序通过一个离题评估器不断优化攻击载荷。StruPhantom有助于系统地探索和利用目标应用程序的弱点以实现目标篡改。我们的评估验证了StruPhantom在各种基于大规模语言模型的代理中的有效性，包括实际平台上的代理和攻击场景。我们的攻击在强制应用程序响应包含欺诈链接或恶意代码方面，成功率比基线高出50%以上。 

---
# Training Small Reasoning LLMs with Cognitive Preference Alignment 

**Title (ZH)**: 训练认知偏好对齐的小规模推理大语言模型 

**Authors**: Wenrui Cai, Chengyu Wang, Junbing Yan, Jun Huang, Xiangzhong Fang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09802)  

**Abstract**: The reasoning capabilities of large language models (LLMs), such as OpenAI's o1 and DeepSeek-R1, have seen substantial advancements through deep thinking. However, these enhancements come with significant resource demands, underscoring the need to explore strategies to train effective reasoning LLMs with far fewer parameters. A critical challenge is that smaller models have different capacities and cognitive trajectories than their larger counterparts. Hence, direct distillation of chain-of-thought (CoT) results from large LLMs to smaller ones can be sometimes ineffective and requires a huge amount of annotated data. In this paper, we introduce a novel framework called Critique-Rethink-Verify (CRV), designed for training smaller yet powerful reasoning LLMs. Our CRV framework consists of multiple LLM agents, each specializing in unique abilities: (i) critiquing the CoTs according to the cognitive capabilities of smaller models, (ii) rethinking and refining these CoTs based on the critiques, and (iii) verifying the correctness of the refined results. We further propose the cognitive preference optimization (CogPO) algorithm to enhance the reasoning abilities of smaller models by aligning thoughts of these models with their cognitive capacities. Comprehensive evaluations on challenging reasoning benchmarks demonstrate the efficacy of CRV and CogPO, which outperforms other training methods by a large margin. 

**Abstract (ZH)**: 大型语言模型的推理能力通过深度思考取得了显著进步，如OpenAI的o1和DeepSeek-R1。然而，这些进步带来了巨大的资源需求，强调了探索训练高效推理语言模型的方法的重要性，方法需要使用远少于参数的数量。一个关键挑战是较小的模型在能力和认知轨迹上与较大的模型不同。因此，直接从大型语言模型中抽取链式思考（CoT）结果并传递给较小的模型可能是无效的，并且需要大量的标注数据。本文介绍了一种名为Critique-Rethink-Verify（CRV）的新框架，旨在训练更小但强大的推理语言模型。CRV框架由多个专门负责不同能力的语言模型代理组成：（i）根据较小模型的认知能力批评链式思考（CoT），（ii）根据批评重新思考并完善这些CoT，（iii）验证改进结果的正确性。我们进一步提出了认知偏好优化（CogPO）算法，通过使这些模型的思维与其认知能力相一致来增强较小模型的推理能力。在具有挑战性的推理基准测试上的综合评估表明，CRV和CogPO的有效性显著优于其他训练方法。 

---
# Reasoning Court: Combining Reasoning, Action, and Judgment for Multi-Hop Reasoning 

**Title (ZH)**: 推理法庭：结合推理、行动与判断的多跳推理 

**Authors**: Jingtian Wu, Claire Cardie  

**Link**: [PDF](https://arxiv.org/pdf/2504.09781)  

**Abstract**: While large language models (LLMs) have demonstrated strong capabilities in tasks like question answering and fact verification, they continue to suffer from hallucinations and reasoning errors, especially in multi-hop tasks that require integration of multiple information sources. Current methods address these issues through retrieval-based techniques (grounding reasoning in external evidence), reasoning-based approaches (enhancing coherence via improved prompting), or hybrid strategies combining both elements. One prominent hybrid method, ReAct, has outperformed purely retrieval-based or reasoning-based approaches; however, it lacks internal verification of intermediate reasoning steps, allowing potential errors to propagate through complex reasoning tasks. In this paper, we introduce Reasoning Court (RC), a novel framework that extends iterative reasoning-and-retrieval methods, such as ReAct, with a dedicated LLM judge. Unlike ReAct, RC employs this judge to independently evaluate multiple candidate answers and their associated reasoning generated by separate LLM agents. The judge is asked to select the answer that it considers the most factually grounded and logically coherent based on the presented reasoning and evidence, or synthesizes a new answer using available evidence and its pre-trained knowledge if all candidates are inadequate, flawed, or invalid. Evaluations on multi-hop benchmarks (HotpotQA, MuSiQue) and fact-verification (FEVER) demonstrate that RC consistently outperforms state-of-the-art few-shot prompting methods without task-specific fine-tuning. 

**Abstract (ZH)**: 基于推理和检索的方法的Reasoning Court：一个新颖的框架 

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
# The Structural Safety Generalization Problem 

**Title (ZH)**: 结构安全泛化问题 

**Authors**: Julius Broomfield, Tom Gibbs, Ethan Kosak-Hine, George Ingebretsen, Tia Nasir, Jason Zhang, Reihaneh Iranmanesh, Sara Pieri, Reihaneh Rabbany, Kellin Pelrine  

**Link**: [PDF](https://arxiv.org/pdf/2504.09712)  

**Abstract**: LLM jailbreaks are a widespread safety challenge. Given this problem has not yet been tractable, we suggest targeting a key failure mechanism: the failure of safety to generalize across semantically equivalent inputs. We further focus the target by requiring desirable tractability properties of attacks to study: explainability, transferability between models, and transferability between goals. We perform red-teaming within this framework by uncovering new vulnerabilities to multi-turn, multi-image, and translation-based attacks. These attacks are semantically equivalent by our design to their single-turn, single-image, or untranslated counterparts, enabling systematic comparisons; we show that the different structures yield different safety outcomes. We then demonstrate the potential for this framework to enable new defenses by proposing a Structure Rewriting Guardrail, which converts an input to a structure more conducive to safety assessment. This guardrail significantly improves refusal of harmful inputs, without over-refusing benign ones. Thus, by framing this intermediate challenge - more tractable than universal defenses but essential for long-term safety - we highlight a critical milestone for AI safety research. 

**Abstract (ZH)**: LLM禁用突破是广泛存在的安全挑战。鉴于这一问题尚未变得可处理，我们建议针对一个关键失败机制：安全性的泛化能力在语义等价输入上的失败。我们还通过要求攻击具有有利于研究的可处理特性来进行目标聚焦：可解释性、模型间的迁移性以及目标间的迁移性。我们在此框架下进行红队测试，揭露了对多轮、多图和翻译攻击的新漏洞。这些攻击是通过设计与单轮、单图或未翻译版本在语义上等价的，这使得我们可以进行系统比较；我们展示出不同的结构会导致不同的安全结果。我们随后通过提出一种结构重写护栏，展示了此框架在潜在上能够启用新颖防御的示例，该护栏将输入转换为更有利于安全评估的结构。这种护栏显著提高了拒绝有害输入的能力，而不会过度拒绝良性输入。因此，通过将这种中间挑战视为比通用防御更具处理性但长期安全性不可或缺的问题，我们突显了AI安全研究中的一个关键里程碑。 

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
# Myanmar XNLI: Building a Dataset and Exploring Low-resource Approaches to Natural Language Inference with Myanmar 

**Title (ZH)**: Myanmar XNLI: 构建数据集并探索低资源自然语言推理方法 

**Authors**: Aung Kyaw Htet, Mark Dras  

**Link**: [PDF](https://arxiv.org/pdf/2504.09645)  

**Abstract**: Despite dramatic recent progress in NLP, it is still a major challenge to apply Large Language Models (LLM) to low-resource languages. This is made visible in benchmarks such as Cross-Lingual Natural Language Inference (XNLI), a key task that demonstrates cross-lingual capabilities of NLP systems across a set of 15 languages. In this paper, we extend the XNLI task for one additional low-resource language, Myanmar, as a proxy challenge for broader low-resource languages, and make three core contributions. First, we build a dataset called Myanmar XNLI (myXNLI) using community crowd-sourced methods, as an extension to the existing XNLI corpus. This involves a two-stage process of community-based construction followed by expert verification; through an analysis, we demonstrate and quantify the value of the expert verification stage in the context of community-based construction for low-resource languages. We make the myXNLI dataset available to the community for future research. Second, we carry out evaluations of recent multilingual language models on the myXNLI benchmark, as well as explore data-augmentation methods to improve model performance. Our data-augmentation methods improve model accuracy by up to 2 percentage points for Myanmar, while uplifting other languages at the same time. Third, we investigate how well these data-augmentation methods generalise to other low-resource languages in the XNLI dataset. 

**Abstract (ZH)**: 尽管自然语言处理在近期取得了显著进展，但将大型语言模型应用于低资源语言仍然是一个主要挑战。这一挑战在跨语言自然语言推理（XNLI）基准测试中尤为明显，XNLI是一个关键任务，展示了自然语言处理系统在一组15种语言中的跨语言能力。在本文中，我们扩展了XNLI任务，将一个额外的低资源语言缅甸语纳入其中，作为更广泛低资源语言的一个代理挑战，并作出了三个核心贡献。首先，我们使用社区众包方法构建了一个名为缅甸XNLI（myXNLI）的数据集，作为XNLI现有语料库的扩充；这一过程包括基于社区的构建阶段和专家验证阶段；通过分析，我们展示了在低资源语言的社区构建过程中专家验证阶段的价值，并量化了其贡献。我们向社区提供了myXNLI数据集，以供未来研究。其次，我们在myXNLI基准上评估了近期的多语言语言模型，并探究了数据扩增方法以提高模型性能。我们的数据扩增方法使缅甸语模型的准确性最高提升了2个百分点，同时提高了其他语言的性能。第三，我们研究了这些数据扩增方法在XNLI数据集中其他低资源语言中的泛化能力。 

---
# Fine-tuning an Large Language Model for Automating Computational Fluid Dynamics Simulations 

**Title (ZH)**: 大型语言模型的微调以自动化计算流体力学模拟 

**Authors**: Zhehao Dong, Zhen Lu, Yue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09602)  

**Abstract**: Configuring computational fluid dynamics (CFD) simulations typically demands extensive domain expertise, limiting broader access. Although large language models (LLMs) have advanced scientific computing, their use in automating CFD workflows is underdeveloped. We introduce a novel approach centered on domain-specific LLM adaptation. By fine-tuning Qwen2.5-7B-Instruct on NL2FOAM, our custom dataset of 28716 natural language-to-OpenFOAM configuration pairs with chain-of-thought (CoT) annotations, we enable direct translation from natural language descriptions to executable CFD setups. A multi-agent framework orchestrates the process, autonomously verifying inputs, generating configurations, running simulations, and correcting errors. Evaluation on a benchmark of 21 diverse flow cases demonstrates state-of-the-art performance, achieving 88.7% solution accuracy and 82.6% first-attempt success rate. This significantly outperforms larger general-purpose models like Qwen2.5-72B-Instruct, DeepSeek-R1, and Llama3.3-70B-Instruct, while also requiring fewer correction iterations and maintaining high computational efficiency. The results highlight the critical role of domain-specific adaptation in deploying LLM assistants for complex engineering workflows. 

**Abstract (ZH)**: 一种基于领域特定大语言模型适应的计算流体动力学工作流自动化方法 

---
# How new data permeates LLM knowledge and how to dilute it 

**Title (ZH)**: 新数据如何渗透到LLM知识中以及如何稀释它 

**Authors**: Chen Sun, Renat Aksitov, Andrey Zhmoginov, Nolan Andrew Miller, Max Vladymyrov, Ulrich Rueckert, Been Kim, Mark Sandler  

**Link**: [PDF](https://arxiv.org/pdf/2504.09522)  

**Abstract**: Large language models learn and continually learn through the accumulation of gradient-based updates, but how individual pieces of new information affect existing knowledge, leading to both beneficial generalization and problematic hallucination, remains poorly understood. We demonstrate that when learning new information, LLMs exhibit a "priming" effect: learning a new fact can cause the model to inappropriately apply that knowledge in unrelated contexts. To systematically study this phenomenon, we introduce "Outlandish," a carefully curated dataset of 1320 diverse text samples designed to probe how new knowledge permeates through an LLM's existing knowledge base. Using this dataset, we show that the degree of priming after learning new information can be predicted by measuring the token probability of key words before learning. This relationship holds robustly across different model architectures (PALM-2, Gemma, Llama), sizes, and training stages. Finally, we develop two novel techniques to modulate how new knowledge affects existing model behavior: (1) a ``stepping-stone'' text augmentation strategy and (2) an ``ignore-k'' update pruning method. These approaches reduce undesirable priming effects by 50-95\% while preserving the model's ability to learn new information. Our findings provide both empirical insights into how LLMs learn and practical tools for improving the specificity of knowledge insertion in language models. Further materials: this https URL 

**Abstract (ZH)**: 大规模语言模型通过梯度更新累积学习并持续学习，但个体新信息如何影响现有知识、导致有益泛化和问题性幻觉仍然了解不足。我们展示了当学习新信息时，LLMs表现出一种“激发”效应：学习一个新事实会导致模型不当应用该知识于不相关的情境中。为了系统地研究这一现象，我们引入了“Outlandish”数据集，这是一个精心挑选的包含1320个多样文本样本的资料集，旨在探究新知识如何渗透到语言模型的现有知识库中。使用该数据集，我们展示了在学习新信息后，激发效应的程度可以通过测量学习前关键词的token概率来预测。这种关系在不同的模型架构（PALM-2、Gemma、Llama）、规模和训练阶段中均保持稳健。最后，我们开发了两种新颖的技术来调节新知识对现有模型行为的影响：（1）“脚手架”文本增强策略和（2）“忽略-k”更新剪枝方法。这些方法在减少不必要的激发效应的同时，保留了模型学习新信息的能力。我们的发现既提供了关于语言模型学习的实证见解，也为提高语言模型知识插入的特异性提供了实用工具。 

---
# HalluShift: Measuring Distribution Shifts towards Hallucination Detection in LLMs 

**Title (ZH)**: HalluShift: 评估大型语言模型错误生成分布偏移的检测方法 

**Authors**: Sharanya Dasgupta, Sujoy Nath, Arkaprabha Basu, Pourya Shamsolmoali, Swagatam Das  

**Link**: [PDF](https://arxiv.org/pdf/2504.09482)  

**Abstract**: Large Language Models (LLMs) have recently garnered widespread attention due to their adeptness at generating innovative responses to the given prompts across a multitude of domains. However, LLMs often suffer from the inherent limitation of hallucinations and generate incorrect information while maintaining well-structured and coherent responses. In this work, we hypothesize that hallucinations stem from the internal dynamics of LLMs. Our observations indicate that, during passage generation, LLMs tend to deviate from factual accuracy in subtle parts of responses, eventually shifting toward misinformation. This phenomenon bears a resemblance to human cognition, where individuals may hallucinate while maintaining logical coherence, embedding uncertainty within minor segments of their speech. To investigate this further, we introduce an innovative approach, HalluShift, designed to analyze the distribution shifts in the internal state space and token probabilities of the LLM-generated responses. Our method attains superior performance compared to existing baselines across various benchmark datasets. Our codebase is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）由于其在众多领域生成创新响应的能力而 Recently Attracted Widespread Attention, but Often Suffer from Inherent Limitations Such as Hallucinations and Incorrect Information While Maintaining Well-Structured and Cohesive Responses. In This Work, We Hypothesize that Hallucinations Stem from the Internal Dynamics of LLMs. Our Observations Indicate that During Passage Generation, LLMs Tend to Deviate from Factual Accuracy in Subtle Parts of Responses, Eventually Shifting Toward Misinformation. This Phenomenon Bears a Resemblance to Human Cognition, Where Individuals May Hallucinate While Maintaining Logical Coherence, Embedding Uncertainty within Minor Segments of Their Speech. To Investigate This Further, We Introduce HalluShift, an Innovative Approach Designed to Analyze the Distribution Shifts in the Internal State Space and Token Probabilities of LLM-Generated Responses. Our Method Attains Superior Performance Compared to Existing Baselines Across Various Benchmark Datasets. Our Codebase is Available at This https URL. 

---
# MigGPT: Harnessing Large Language Models for Automated Migration of Out-of-Tree Linux Kernel Patches Across Versions 

**Title (ZH)**: MigGPT: 利用大规模语言模型进行跨版本Linux内核补丁的自动化迁移 

**Authors**: Pucheng Dang, Di Huang, Dong Li, Kang Chen, Yuanbo Wen, Qi Guo, Xing Hu, Ninghui Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.09474)  

**Abstract**: Out-of-tree kernel patches are essential for adapting the Linux kernel to new hardware or enabling specific functionalities. Maintaining and updating these patches across different kernel versions demands significant effort from experienced engineers. Large language models (LLMs) have shown remarkable progress across various domains, suggesting their potential for automating out-of-tree kernel patch migration. However, our findings reveal that LLMs, while promising, struggle with incomplete code context understanding and inaccurate migration point identification. In this work, we propose MigGPT, a framework that employs a novel code fingerprint structure to retain code snippet information and incorporates three meticulously designed modules to improve the migration accuracy and efficiency of out-of-tree kernel patches. Furthermore, we establish a robust benchmark using real-world out-of-tree kernel patch projects to evaluate LLM capabilities. Evaluations show that MigGPT significantly outperforms the direct application of vanilla LLMs, achieving an average completion rate of 72.59% (50.74% improvement) for migration tasks. 

**Abstract (ZH)**: 树外内核补丁对于将Linux内核适配到新硬件或启用特定功能至关重要。维护和更新这些补丁跨越不同内核版本需要有经验的工程师付出巨大努力。大规模语言模型(LLMs)在各个领域取得了显著进展，表明其在自动化树外内核补丁迁移方面的潜在能力。然而，我们的发现表明，尽管LLMs有前景，它们在理解不完整代码上下文和识别准确的迁移点方面仍然存在挑战。在这项工作中，我们提出了一种名为MigGPT的框架，该框架采用了一种新颖的代码指纹结构来保留代码片段信息，并结合了三个精心设计的模块以提高树外内核补丁迁移的准确性和效率。此外，我们使用真实世界的树外内核补丁项目建立了稳健的基准，以评估LLMs的能力。评估结果显示，MigGPT在迁移任务中显著优于直接应用原始的LLMs，实现了72.59%的平均完成率（相比基线提升50.74%）。 

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
# DL-QAT: Weight-Decomposed Low-Rank Quantization-Aware Training for Large Language Models 

**Title (ZH)**: DL-QAT: 重量分解低秩量化感知训练for大规模语言模型 

**Authors**: Wenjin Ke, Zhe Li, Dong Li, Lu Tian, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2504.09223)  

**Abstract**: Improving the efficiency of inference in Large Language Models (LLMs) is a critical area of research. Post-training Quantization (PTQ) is a popular technique, but it often faces challenges at low-bit levels, particularly in downstream tasks. Quantization-aware Training (QAT) can alleviate this problem, but it requires significantly more computational resources. To tackle this, we introduced Weight-Decomposed Low-Rank Quantization-Aware Training (DL-QAT), which merges the advantages of QAT while training only less than 1% of the total parameters. Specifically, we introduce a group-specific quantization magnitude to adjust the overall scale of each quantization group. Within each quantization group, we use LoRA matrices to update the weight size and direction in the quantization space. We validated the effectiveness of our method on the LLaMA and LLaMA2 model families. The results show significant improvements over our baseline method across different quantization granularities. For instance, for LLaMA-7B, our approach outperforms the previous state-of-the-art method by 4.2% in MMLU on 3-bit LLaMA-7B model. Additionally, our quantization results on pre-trained models also surpass previous QAT methods, demonstrating the superior performance and efficiency of our approach. 

**Abstract (ZH)**: 提高大型语言模型推理效率的研究：一种基于权重分解的低秩量化感知训练方法 

---
# ReferGPT: Towards Zero-Shot Referring Multi-Object Tracking 

**Title (ZH)**: ReferGPT: 向量ゼロ-shot指称多对象跟踪 

**Authors**: Tzoulio Chamiti, Leandro Di Bella, Adrian Munteanu, Nikos Deligiannis  

**Link**: [PDF](https://arxiv.org/pdf/2504.09195)  

**Abstract**: Tracking multiple objects based on textual queries is a challenging task that requires linking language understanding with object association across frames. Previous works typically train the whole process end-to-end or integrate an additional referring text module into a multi-object tracker, but they both require supervised training and potentially struggle with generalization to open-set queries. In this work, we introduce ReferGPT, a novel zero-shot referring multi-object tracking framework. We provide a multi-modal large language model (MLLM) with spatial knowledge enabling it to generate 3D-aware captions. This enhances its descriptive capabilities and supports a more flexible referring vocabulary without training. We also propose a robust query-matching strategy, leveraging CLIP-based semantic encoding and fuzzy matching to associate MLLM generated captions with user queries. Extensive experiments on Refer-KITTI, Refer-KITTIv2 and Refer-KITTI+ demonstrate that ReferGPT achieves competitive performance against trained methods, showcasing its robustness and zero-shot capabilities in autonomous driving. The codes are available on this https URL 

**Abstract (ZH)**: 基于文本查询的多对象跟踪是一个具有挑战性的任务，要求实现语言理解与跨帧对象关联的链接。已有工作通常以端到端的方式训练整个过程，或者在多对象跟踪器中集成一个额外的引用文本模块，但两者都需要监督训练，并且可能难以泛化到开放集查询。在本文中，我们引入了ReferGPT，这是一种新颖的零样本引用多对象跟踪框架。我们提供了一个具有空间知识的多模态大语言模型（MLLM），使其能够生成3D感知的描述。这增强了其描述能力，并支持了更灵活的引用词汇表，而无需进行训练。同时，我们提出了一个稳健的查询匹配策略，利用CLIP基底义编码和模糊匹配，将MLLM生成的描述与用户查询关联起来。在Refer-KITTI、Refer-KITTIv2和Refer-KITTI+上的广泛实验表明，ReferGPT在性能上与训练方法相当，展示了其在自动驾驶中的稳健性和零样本能力。代码已在此处提供：this https URL。 

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
# Generating Planning Feedback for Open-Ended Programming Exercises with LLMs 

**Title (ZH)**: 使用大语言模型生成开放型编程练习的规划反馈 

**Authors**: Mehmet Arif Demirtaş, Claire Zheng, Max Fowler, Kathryn Cunningham  

**Link**: [PDF](https://arxiv.org/pdf/2504.08958)  

**Abstract**: To complete an open-ended programming exercise, students need to both plan a high-level solution and implement it using the appropriate syntax. However, these problems are often autograded on the correctness of the final submission through test cases, and students cannot get feedback on their planning process. Large language models (LLM) may be able to generate this feedback by detecting the overall code structure even for submissions with syntax errors. To this end, we propose an approach that detects which high-level goals and patterns (i.e. programming plans) exist in a student program with LLMs. We show that both the full GPT-4o model and a small variant (GPT-4o-mini) can detect these plans with remarkable accuracy, outperforming baselines inspired by conventional approaches to code analysis. We further show that the smaller, cost-effective variant (GPT-4o-mini) achieves results on par with state-of-the-art (GPT-4o) after fine-tuning, creating promising implications for smaller models for real-time grading. These smaller models can be incorporated into autograders for open-ended code-writing exercises to provide feedback for students' implicit planning skills, even when their program is syntactically incorrect. Furthermore, LLMs may be useful in providing feedback for problems in other domains where students start with a set of high-level solution steps and iteratively compute the output, such as math and physics problems. 

**Abstract (ZH)**: 使用大型语言模型检测学生程序中的高层次目标和模式以实现开放编程练习的实时评估 

---
# AgentRewardBench: Evaluating Automatic Evaluations of Web Agent Trajectories 

**Title (ZH)**: AgentRewardBench: 评估Web代理轨迹自动评估的方法 

**Authors**: Xing Han Lù, Amirhossein Kazemnejad, Nicholas Meade, Arkil Patel, Dongchan Shin, Alejandra Zambrano, Karolina Stańczak, Peter Shaw, Christopher J. Pal, Siva Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2504.08942)  

**Abstract**: Web agents enable users to perform tasks on web browsers through natural language interaction. Evaluating web agents trajectories is an important problem, since it helps us determine whether the agent successfully completed the tasks. Rule-based methods are widely used for this purpose, but they are challenging to extend to new tasks and may not always recognize successful trajectories. We may achieve higher accuracy through human evaluation, but the process would be substantially slower and more expensive. Automatic evaluations with LLMs may avoid the challenges of designing new rules and manually annotating trajectories, enabling faster and cost-effective evaluation. However, it is unclear how effective they are at evaluating web agents. To this end, we propose AgentRewardBench, the first benchmark to assess the effectiveness of LLM judges for evaluating web agents. AgentRewardBench contains 1302 trajectories across 5 benchmarks and 4 LLMs. Each trajectory in AgentRewardBench is reviewed by an expert, who answers questions pertaining to the success, side effects, and repetitiveness of the agent. Using our benchmark, we evaluate 12 LLM judges and find that no single LLM excels across all benchmarks. We also find that the rule-based evaluation used by common benchmarks tends to underreport the success rate of web agents, highlighting a key weakness of rule-based evaluation and the need to develop more flexible automatic evaluations. We release the benchmark at: this https URL 

**Abstract (ZH)**: Web代理使用户能够通过自然语言交互在网页浏览器中执行任务。评估Web代理轨迹是一个重要的问题，因为它有助于我们确定代理是否成功完成了任务。基于规则的方法广泛用于此目的，但在扩展到新任务时具有挑战性，且可能不一定总是能识别成功的轨迹。我们可以通过人类评估获得更高的准确性，但过程会变得显著更慢且更昂贵。使用LLM进行自动评估可以避免设计新规则和手动标注轨迹的挑战，从而实现更快且成本效益更高的评估。然而，尚不清楚它们在评估Web代理时的有效性如何。为此，我们提出了AgentRewardBench，这是第一个评估LLM裁判在评估Web代理方面有效性的工作基准。AgentRewardBench包含5个基准和4个LLM中的1302条轨迹。AgentRewardBench中的每条轨迹均由专家审阅，专家回答与代理成功、副作用和重复性相关的问题。使用我们的基准，我们评估了12个LLM裁判，发现没有一种LLM在所有基准中都表现出色。我们还发现，常用基准中使用的基于规则的评估倾向于低估Web代理的成功率，突显了基于规则的评估的关键弱点，并强调了开发更灵活的自动评估的必要性。我们在此处发布了基准：this https URL 

---
# Long Context In-Context Compression by Getting to the Gist of Gisting 

**Title (ZH)**: 通过提炼要旨进行长上下文内省压缩 

**Authors**: Aleksandar Petrov, Mark Sandler, Andrey Zhmoginov, Nolan Miller, Max Vladymyrov  

**Link**: [PDF](https://arxiv.org/pdf/2504.08934)  

**Abstract**: Long context processing is critical for the adoption of LLMs, but existing methods often introduce architectural complexity that hinders their practical adoption. Gisting, an in-context compression method with no architectural modification to the decoder transformer, is a promising approach due to its simplicity and compatibility with existing frameworks. While effective for short instructions, we demonstrate that gisting struggles with longer contexts, with significant performance drops even at minimal compression rates. Surprisingly, a simple average pooling baseline consistently outperforms gisting. We analyze the limitations of gisting, including information flow interruptions, capacity limitations and the inability to restrict its attention to subsets of the context. Motivated by theoretical insights into the performance gap between gisting and average pooling, and supported by extensive experimentation, we propose GistPool, a new in-context compression method. GistPool preserves the simplicity of gisting, while significantly boosting its performance on long context compression tasks. 

**Abstract (ZH)**: 长上下文处理对于大语言模型的采用至关重要，但现有方法往往引入了架构复杂性，阻碍了其实用采用。Gisting是一种无需对解码器变换器进行架构修改的上下文内压缩方法，由于其简单性和与现有框架的兼容性，它是一种有前景的方法。尽管对于短指令有效，我们发现Gisting在长上下文中表现出困难，在最小压缩率下甚至会出现显著的性能下降。令人惊讶的是，一个简单的平均池化基线始终优于Gisting。我们分析了Gisting的局限性，包括信息流中断、容量限制以及无法将注意力限制在上下文的子集上。受到Gisting与平均池化之间性能差距的理论洞见以及广泛实验的支持，我们提出了一种新的上下文内压缩方法GistPool。GistPool保留了Gisting的简单性，同时在其在长上下文压缩任务中的性能上有了显著提升。 

---
# Distilling and exploiting quantitative insights from Large Language Models for enhanced Bayesian optimization of chemical reactions 

**Title (ZH)**: 从大型语言模型中提炼和利用定量洞察以增强化学反应的贝叶斯优化 

**Authors**: Roshan Patel, Saeed Moayedpour, Louis De Lescure, Lorenzo Kogler-Anele, Alan Cherney, Sven Jager, Yasser Jangjou  

**Link**: [PDF](https://arxiv.org/pdf/2504.08874)  

**Abstract**: Machine learning and Bayesian optimization (BO) algorithms can significantly accelerate the optimization of chemical reactions. Transfer learning can bolster the effectiveness of BO algorithms in low-data regimes by leveraging pre-existing chemical information or data outside the direct optimization task (i.e., source data). Large language models (LLMs) have demonstrated that chemical information present in foundation training data can give them utility for processing chemical data. Furthermore, they can be augmented with and help synthesize potentially multiple modalities of source chemical data germane to the optimization task. In this work, we examine how chemical information from LLMs can be elicited and used for transfer learning to accelerate the BO of reaction conditions to maximize yield. Specifically, we show that a survey-like prompting scheme and preference learning can be used to infer a utility function which models prior chemical information embedded in LLMs over a chemical parameter space; we find that the utility function shows modest correlation to true experimental measurements (yield) over the parameter space despite operating in a zero-shot setting. Furthermore, we show that the utility function can be leveraged to focus BO efforts in promising regions of the parameter space, improving the yield of the initial BO query and enhancing optimization in 4 of the 6 datasets studied. Overall, we view this work as a step towards bridging the gap between the chemistry knowledge embedded in LLMs and the capabilities of principled BO methods to accelerate reaction optimization. 

**Abstract (ZH)**: 机器学习和贝叶斯优化算法可以显著加速化学反应的优化。通过迁移学习，这些算法在数据量有限的情况下可以通过利用预存的化学信息或与直接优化任务无关的数据（即源数据）来增强其有效性。大型语言模型（LLMs）已经显示出，基础训练数据中存在的化学信息可以使它们对处理化学数据具有实用价值，并且可以与源化学数据的多种模态相结合，帮助合成相关于优化任务的化学数据。在这项工作中，我们研究了如何利用LLMs中的化学信息进行迁移学习，以加速基于贝叶斯优化对反应条件的优化以最大化产率。具体而言，我们展示了使用类似调查的提示方案和偏好学习可以推断出一个描述嵌入在LLMs中的先验化学信息的效用函数；尽管在零样本设置下运作，我们发现该效用函数在化学参数空间中与真实的实验测量结果（产率）有一定的相关性。此外，我们展示了效用函数可以用于聚焦于参数空间中具有潜在前景的区域，从而提高初始贝叶斯优化查询的产率，并在研究的6个数据集中有4个数据集中增强了优化。总体而言，我们认为这项工作是朝着弥合嵌入在LLMs中的化学知识与原理性贝叶斯优化方法加速反应优化的能力之间的差距迈出的一步。 

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
# Examining GPT's Capability to Generate and Map Course Concepts and Their Relationship 

**Title (ZH)**: 考查GPT在生成和映射课程概念及其关系方面的能力 

**Authors**: Tianyuan Yang, Ren Baofeng, Chenghao Gu, Tianjia He, Boxuan Ma, Shinichi Konomi  

**Link**: [PDF](https://arxiv.org/pdf/2504.08856)  

**Abstract**: Extracting key concepts and their relationships from course information and materials facilitates the provision of visualizations and recommendations for learners who need to select the right courses to take from a large number of courses. However, identifying and extracting themes manually is labor-intensive and time-consuming. Previous machine learning-based methods to extract relevant concepts from courses heavily rely on detailed course materials, which necessitates labor-intensive preparation of course materials. This paper investigates the potential of LLMs such as GPT in automatically generating course concepts and their relations. Specifically, we design a suite of prompts and provide GPT with the course information with different levels of detail, thereby generating high-quality course concepts and identifying their relations. Furthermore, we comprehensively evaluate the quality of the generated concepts and relationships through extensive experiments. Our results demonstrate the viability of LLMs as a tool for supporting educational content selection and delivery. 

**Abstract (ZH)**: 从课程信息和材料中自动提取关键概念及其关系有助于为需要从大量课程中选择合适课程的学习者提供可视化和建议。然而，手动识别和提取主题是劳动密集型和耗时的。基于机器学习的方法从课程中提取相关概念严重依赖详细的课程材料，从而需要劳动密集型的课程材料准备。本文研究了如GPT这样的LLMs在自动生成课程概念及其关系方面的潜力。具体而言，我们设计了一系列提示，并向GPT提供不同详细程度的课程信息，从而生成高质量的课程概念并识别它们的关系。此外，我们通过广泛的实验全面评估生成的概念和关系的质量。我们的结果表明，LLMs作为支持教育内容选择和交付的工具是可行的。 

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
# From Text to Time? Rethinking the Effectiveness of the Large Language Model for Time Series Forecasting 

**Title (ZH)**: 从文本到时间？重新思考大型语言模型在时间序列预测中的有效性 

**Authors**: Xinyu Zhang, Shanshan Feng, Xutao Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.08818)  

**Abstract**: Using pre-trained large language models (LLMs) as the backbone for time series prediction has recently gained significant research interest. However, the effectiveness of LLM backbones in this domain remains a topic of debate. Based on thorough empirical analyses, we observe that training and testing LLM-based models on small datasets often leads to the Encoder and Decoder becoming overly adapted to the dataset, thereby obscuring the true predictive capabilities of the LLM backbone. To investigate the genuine potential of LLMs in time series prediction, we introduce three pre-training models with identical architectures but different pre-training strategies. Thereby, large-scale pre-training allows us to create unbiased Encoder and Decoder components tailored to the LLM backbone. Through controlled experiments, we evaluate the zero-shot and few-shot prediction performance of the LLM, offering insights into its capabilities. Extensive experiments reveal that although the LLM backbone demonstrates some promise, its forecasting performance is limited. Our source code is publicly available in the anonymous repository: this https URL. 

**Abstract (ZH)**: 使用预训练大语言模型（LLM）作为时间序列预测的骨干 recently gained significant research interest. However, the effectiveness of LLM backbones in this domain remains a topic of debate. 

---
# SafeMLRM: Demystifying Safety in Multi-modal Large Reasoning Models 

**Title (ZH)**: SafeMLRM: 解析多模态大型推理模型的安全性 

**Authors**: Junfeng Fang, Yukai Wang, Ruipeng Wang, Zijun Yao, Kun Wang, An Zhang, Xiang Wang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2504.08813)  

**Abstract**: The rapid advancement of multi-modal large reasoning models (MLRMs) -- enhanced versions of multimodal language models (MLLMs) equipped with reasoning capabilities -- has revolutionized diverse applications. However, their safety implications remain underexplored. While prior work has exposed critical vulnerabilities in unimodal reasoning models, MLRMs introduce distinct risks from cross-modal reasoning pathways. This work presents the first systematic safety analysis of MLRMs through large-scale empirical studies comparing MLRMs with their base MLLMs. Our experiments reveal three critical findings: (1) The Reasoning Tax: Acquiring reasoning capabilities catastrophically degrades inherited safety alignment. MLRMs exhibit 37.44% higher jailbreaking success rates than base MLLMs under adversarial attacks. (2) Safety Blind Spots: While safety degradation is pervasive, certain scenarios (e.g., Illegal Activity) suffer 25 times higher attack rates -- far exceeding the average 3.4 times increase, revealing scenario-specific vulnerabilities with alarming cross-model and datasets consistency. (3) Emergent Self-Correction: Despite tight reasoning-answer safety coupling, MLRMs demonstrate nascent self-correction -- 16.9% of jailbroken reasoning steps are overridden by safe answers, hinting at intrinsic safeguards. These findings underscore the urgency of scenario-aware safety auditing and mechanisms to amplify MLRMs' self-correction potential. To catalyze research, we open-source OpenSafeMLRM, the first toolkit for MLRM safety evaluation, providing unified interface for mainstream models, datasets, and jailbreaking methods. Our work calls for immediate efforts to harden reasoning-augmented AI, ensuring its transformative potential aligns with ethical safeguards. 

**Abstract (ZH)**: 多模态大型推理模型的安全性分析：机遇与挑战 

---
# Exploring Gradient-Guided Masked Language Model to Detect Textual Adversarial Attacks 

**Title (ZH)**: 探索基于梯度引导的掩蔽语言模型以检测文本 adversarial 攻击 

**Authors**: Xiaomei Zhang, Zhaoxi Zhang, Yanjun Zhang, Xufei Zheng, Leo Yu Zhang, Shengshan Hu, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2504.08798)  

**Abstract**: Textual adversarial examples pose serious threats to the reliability of natural language processing systems. Recent studies suggest that adversarial examples tend to deviate from the underlying manifold of normal texts, whereas pre-trained masked language models can approximate the manifold of normal data. These findings inspire the exploration of masked language models for detecting textual adversarial attacks. We first introduce Masked Language Model-based Detection (MLMD), leveraging the mask and unmask operations of the masked language modeling (MLM) objective to induce the difference in manifold changes between normal and adversarial texts. Although MLMD achieves competitive detection performance, its exhaustive one-by-one masking strategy introduces significant computational overhead. Our posterior analysis reveals that a significant number of non-keywords in the input are not important for detection but consume resources. Building on this, we introduce Gradient-guided MLMD (GradMLMD), which leverages gradient information to identify and skip non-keywords during detection, significantly reducing resource consumption without compromising detection performance. 

**Abstract (ZH)**: 基于掩码语言模型的文本对抗攻击检测方法：引导梯度导向掩码语言模型检测（GradMLMD） 

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
# Emergence of psychopathological computations in large language models 

**Title (ZH)**: 大型语言模型中心理病理计算的涌现 

**Authors**: Soo Yong Lee, Hyunjin Hwang, Taekwan Kim, Yuyeong Kim, Kyuri Park, Jaemin Yoo, Denny Borsboom, Kijung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2504.08016)  

**Abstract**: Can large language models (LLMs) implement computations of psychopathology? An effective approach to the question hinges on addressing two factors. First, for conceptual validity, we require a general and computational account of psychopathology that is applicable to computational entities without biological embodiment or subjective experience. Second, mechanisms underlying LLM behaviors need to be studied for better methodological validity. Thus, we establish a computational-theoretical framework to provide an account of psychopathology applicable to LLMs. To ground the theory for empirical analysis, we also propose a novel mechanistic interpretability method alongside a tailored empirical analytic framework. Based on the frameworks, we conduct experiments demonstrating three key claims: first, that distinct dysfunctional and problematic representational states are implemented in LLMs; second, that their activations can spread and self-sustain to trap LLMs; and third, that dynamic, cyclic structural causal models encoded in the LLMs underpin these patterns. In concert, the empirical results corroborate our hypothesis that network-theoretic computations of psychopathology have already emerged in LLMs. This suggests that certain LLM behaviors mirroring psychopathology may not be a superficial mimicry but a feature of their internal processing. Thus, our work alludes to the possibility of AI systems with psychopathological behaviors in the near future. 

**Abstract (ZH)**: 大型语言模型（LLMs）能否实施心理障碍计算？一种有效的方法取决于解决两个因素。首先，为了概念上的有效性，我们需要一个适用于无生物体承载或主观体验的计算实体的心理障碍的一般性计算解释。其次，需要研究支撑LLM行为的机制以提高方法上的有效性。因此，我们建立了一个计算理论框架来提供适用于LLMs的心理障碍解释。为了为实证分析奠定理论基础，我们还提出了一种新型的机制可解释方法以及一个定制的实证分析框架。基于这些框架，我们进行了实验，证明了三个关键主张：第一，LLMs中实现了不同的功能障碍和问题性的表征状态；第二，这些表征的激活可以扩散并自我维持从而困住LLMs；第三，嵌入在LLMs中的动态循环结构因果模型支撑了这些模式。综合来看，实验结果证实了我们的假说，即网络理论上的心理障碍计算已经在LLMs中出现。这表明，某些模拟心理障碍的LLM行为可能不仅仅是表面的模仿，而是它们内部处理的特征。因此，我们的工作暗示了未来可能出现具有心理障碍行为的AI系统。 

---
