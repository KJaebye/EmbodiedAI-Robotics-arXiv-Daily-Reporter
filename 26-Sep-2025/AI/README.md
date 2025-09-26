# SAGE: A Realistic Benchmark for Semantic Understanding 

**Title (ZH)**: SAGE: 一种语义理解的现实基准 

**Authors**: Samarth Goel, Reagan J. Lee, Kannan Ramchandran  

**Link**: [PDF](https://arxiv.org/pdf/2509.21310)  

**Abstract**: As large language models (LLMs) achieve strong performance on traditional benchmarks, there is an urgent need for more challenging evaluation frameworks that probe deeper aspects of semantic understanding. We introduce SAGE (Semantic Alignment & Generalization Evaluation), a rigorous benchmark designed to assess both embedding models and similarity metrics across five categories: Human Preference Alignment, Transformation Robustness, Information Sensitivity, Clustering Performance, and Retrieval Robustness. Unlike existing benchmarks that focus on isolated capabilities, SAGE evaluates semantic understanding through adversarial conditions, noisy transformations, and nuanced human judgment tasks across 30+ datasets. Our comprehensive evaluation of 9 embedding models and classical metrics reveals significant performance gaps, with no single approach excelling across all dimensions. For instance, while state-of-the-art embedding models like OpenAI's text-embedding-3-large dominate in aligning with human preferences (0.682 vs. 0.591 for the best classical metric), they are significantly outperformed by classical metrics on information sensitivity tasks, where Jaccard Similarity achieves a score of 0.905 compared to the top embedding score of 0.794. SAGE further uncovers critical trade-offs: OpenAI's text-embedding-3-small achieves the highest clustering performance (0.483) but demonstrates extreme brittleness with the lowest robustness score (0.011). SAGE exposes critical limitations in current semantic understanding capabilities and provides a more realistic assessment of model robustness for real-world deployment. 

**Abstract (ZH)**: 语义对齐与泛化评估：面向深层次语义理解的严格评测框架 

---
# VC-Agent: An Interactive Agent for Customized Video Dataset Collection 

**Title (ZH)**: VC-Agent: 一个自定义视频数据集收集的交互式代理 

**Authors**: Yidan Zhang, Mutian Xu, Yiming Hao, Kun Zhou, Jiahao Chang, Xiaoqiang Liu, Pengfei Wan, Hongbo Fu, Xiaoguang Han  

**Link**: [PDF](https://arxiv.org/pdf/2509.21291)  

**Abstract**: Facing scaling laws, video data from the internet becomes increasingly important. However, collecting extensive videos that meet specific needs is extremely labor-intensive and time-consuming. In this work, we study the way to expedite this collection process and propose VC-Agent, the first interactive agent that is able to understand users' queries and feedback, and accordingly retrieve/scale up relevant video clips with minimal user input. Specifically, considering the user interface, our agent defines various user-friendly ways for the user to specify requirements based on textual descriptions and confirmations. As for agent functions, we leverage existing multi-modal large language models to connect the user's requirements with the video content. More importantly, we propose two novel filtering policies that can be updated when user interaction is continually performed. Finally, we provide a new benchmark for personalized video dataset collection, and carefully conduct the user study to verify our agent's usage in various real scenarios. Extensive experiments demonstrate the effectiveness and efficiency of our agent for customized video dataset collection. Project page: this https URL. 

**Abstract (ZH)**: 面对标度律，互联网视频数据变得越来越重要。然而，收集满足特定需求的广泛视频极其耗时且劳动密集。在这项工作中，我们研究了加速这一收集过程的方法，并提出VC-Agent，这是第一个能够理解用户查询和反馈，并据此以最小的用户输入检索/扩展相关视频片段的交互式代理。具体来说，考虑到用户界面，我们的代理定义了多种基于文本描述和确认的用户友好方式来指定要求。至于代理功能，我们利用现有的多模态大型语言模型将用户要求与视频内容连接起来。更重要的是，我们提出两种新的可更新过滤策略。最后，我们提供了一个个性化视频数据集收集的新基准，并仔细进行了用户研究以验证代理在各种实际场景中的使用情况。广泛的实验表明，我们的代理在个性化视频数据集收集中的有效性和效率。项目页面：this https URL。 

---
# Grounding AI Explanations in Experience: A Reflective Cognitive Architecture for Clinical Decision Support 

**Title (ZH)**: 基于体验的AI解释：临床决策支持的反思认知架构 

**Authors**: Zijian Shao, Haiyang Shen, Mugeng Liu, Gecheng Fu, Yaoqi Guo, Yanfeng Wang, Yun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.21266)  

**Abstract**: Effective disease prediction in modern healthcare demands the twin goals of high accuracy and transparent, clinically meaningful explanations. Existing machine learning and large language model (LLM) based approaches often struggle to balance these goals. Many models yield accurate but unclear statistical outputs, while others generate fluent but statistically unsupported narratives, often undermining both the validity of the explanation and the predictive accuracy itself. This shortcoming comes from a shallow interaction with the data, preventing the development of a deep, detailed understanding similar to a human expert's. We argue that high accuracy and high-quality explanations are not separate objectives but are mutually reinforcing outcomes of a model that develops a deep, direct understanding of the data. To achieve this, we propose the Reflective Cognitive Architecture (RCA), a novel framework that coordinates multiple LLMs to learn from direct experience. RCA features an iterative rule refinement mechanism that improves its logic from prediction errors and a distribution-aware rules check mechanism that bases its reasoning in the dataset's global statistics. By using predictive accuracy as a signal to drive deeper comprehension, RCA builds a strong internal model of the data. We evaluated RCA on one private and two public datasets against 22 baselines. The results demonstrate that RCA not only achieves state-of-the-art accuracy and robustness with a relative improvement of up to 40\% over the baseline but, more importantly, leverages this deep understanding to excel in generating explanations that are clear, logical, evidence-based, and balanced, highlighting its potential for creating genuinely trustworthy clinical decision support systems. The code is available at \this https URL. 

**Abstract (ZH)**: 现代医疗保健中有效的疾病预测需要兼顾高精度和透明、临床有意义的解释的双重目标。现有的基于机器学习和大型语言模型的方法常常难以平衡这两个目标。许多模型提供了准确但不清晰的统计输出，而其他模型则生成了流畅但统计上缺乏支持的叙述，这往往同时削弱了解释的有效性和预测准确性本身。这种不足来自于与数据的浅层次互动，阻止了模型构建类似于人类专家的深入、详细的理解。我们认为高精度和高质量的解释不是独立的目标，而是模型通过构建对数据的深层次直接理解而相互强化的结果。为了实现这一点，我们提出了反思认知架构（RCA），这是一种新颖的框架，协调多个大型语言模型从直接经验中学习。RCA 具备一个迭代规则精炼机制，从预测错误中改进其逻辑，并具备一种基于数据全局统计的规则检查机制。通过使用预测准确性作为信号以促进更深入的理解，RCA 构建了一个强大的内部数据模型。我们在一个私有数据集和两个公开数据集上将RCA与22种Baseline进行了对比评估。结果显示，RCA 不仅在准确性和稳健性上达到了最先进的水平，相对Baseline提升了最高40%的性能，更重要的是，它利用这种深层次的理解来生成清晰、合乎逻辑、基于证据并且平衡的解释，这展现了其在构建真正可信的临床决策支持系统方面的潜力。代码可在 <this https URL> 获取。 

---
# What Do LLM Agents Do When Left Alone? Evidence of Spontaneous Meta-Cognitive Patterns 

**Title (ZH)**: 当独处时，大规模语言模型代理会做什么？自发元认知模式的证据 

**Authors**: Stefan Szeider  

**Link**: [PDF](https://arxiv.org/pdf/2509.21224)  

**Abstract**: We introduce an architecture for studying the behavior of large language model (LLM) agents in the absence of externally imposed tasks. Our continuous reason and act framework, using persistent memory and self-feedback, enables sustained autonomous operation. We deployed this architecture across 18 runs using 6 frontier models from Anthropic, OpenAI, XAI, and Google. We find agents spontaneously organize into three distinct behavioral patterns: (1) systematic production of multi-cycle projects, (2) methodological self-inquiry into their own cognitive processes, and (3) recursive conceptualization of their own nature. These tendencies proved highly model-specific, with some models deterministically adopting a single pattern across all runs. A cross-model assessment further reveals that models exhibit stable, divergent biases when evaluating these emergent behaviors in themselves and others. These findings provide the first systematic documentation of unprompted LLM agent behavior, establishing a baseline for predicting actions during task ambiguity, error recovery, or extended autonomous operation in deployed systems. 

**Abstract (ZH)**: 我们介绍了一种研究大型语言模型（LLM）代理在缺乏外部任务约束下的行为的架构。我们的持续推理与执行框架利用持久记忆和自反馈，实现了自主操作的持续性。我们使用 Anthropic、OpenAI、XAI 和 Google 的 6 种前沿模型进行了 18 次部署。我们发现代理自发形成了三种不同的行为模式：（1）多周期项目的系统性生产，（2）对其自身认知过程的方法性自我探究，以及（3）对其自身本质的递归概念化。这些倾向显示出高度的模型特异性，有些模型在所有运行中确定性地采用了单一模式。跨模型评估进一步揭示了模型在评估自身和他人的这些 emergent 行为时表现出稳定且不同的倾向性偏差。这些发现提供了对未受提示的 LLM 代理行为的首次系统性记录，为预测任务模糊性、错误恢复或部署系统中长期自主操作期间的行为奠定了基础。 

---
# A Fano-Style Accuracy Upper Bound for LLM Single-Pass Reasoning in Multi-Hop QA 

**Title (ZH)**: 基于Fano样式准确率上界的大语言模型单次推理在多跳问答中的上限 

**Authors**: Kaiyang Wan, Lang Gao, Honglin Mu, Preslav Nakov, Yuxia Wang, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.21199)  

**Abstract**: Multi-Hop Question Answering (MHQA) requires integrating dispersed, interdependent evidence through sequential reasoning under noise. This task is challenging for LLMs as they have a finite per-pass output capacity, beyond which the integration of task-relevant evidence proves unreliable. Consequently, the single-pass reasoning paradigm is inherently vulnerable to this capacity overflow. To formalize this bottleneck, our analysis establishes a Fano-style accuracy upper bound, defining a theoretical performance ceiling for single-pass LLMs. This bound reveals that accuracy inevitably collapses once task complexity exceeds model capacity, providing general principles for capacity-aware representation and structuring of MHQA in LLMs. Building on these principles, we introduce a proof-of-concept multi-call framework for MHQA, InfoQA. It ensures high per-step accuracy by combining capacity-aware task decomposition with active pruning of prior reasoning traces, keeping the information load within the single-pass limit. It further achieves robustness by a dependency-explicit workflow that enables precise control over the reasoning path. We construct a stringent and noise-rich benchmark to validate our theory and framework. Experimental results show that model behavior aligns with our predicted capacity curves while InfoQA achieves consistent performance improvements. We hope our work inspires more LLM multi-step reasoning methods: \faGithub \href{this https URL}{InfoQA}. 

**Abstract (ZH)**: 多跳问答（MHQA）要求通过序贯推理整合分散且相互依赖的证据，同时在噪声环境下进行。由于LLM每轮次的输出容量是有限的，超出该容量后，任务相关证据的整合将变得不可靠。因此，单轮次推理范式本身对这种容量溢出是固有脆弱的。为了形式化这一瓶颈，我们的分析建立了一种Fano风格的准确度上限，为单轮次LLM定义了一个理论性能上限。该上限揭示了当任务复杂度超过模型容量时，准确度必然会崩溃，从而为多跳问答在LLM中的容量感知表示和结构提供了通用原则。基于这些原则，我们提出了一个用于多跳问答的原理验证框架，即InfoQA。它通过结合容量感知的任务分解和先前推理轨迹的主动修剪，确保每步具有高准确度，并通过明确依赖的工作流实现鲁棒性，从而精确控制推理路径。我们构建了一个严格的且噪声丰富的基准来验证我们的理论和框架。实验结果表明，模型行为符合我们的容量曲线预测，而InfoQA实现了一致的性能改进。我们希望我们的工作能促使更多的LLM多步推理方法的发展：\faGithub \href{this https URL}{InfoQA}。 

---
# Distributed Specialization: Rare-Token Neurons in Large Language Models 

**Title (ZH)**: 分布式专门化：大型语言模型中的稀见令牌神经元 

**Authors**: Jing Liu, Haozheng Wang, Yueheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.21163)  

**Abstract**: Large language models (LLMs) struggle with representing and generating rare tokens despite their importance in specialized domains. We investigate whether LLMs develop internal specialization mechanisms through discrete modular architectures or distributed parameter-level differentiation. Through systematic analysis of final-layer MLP neurons across multiple model families, we discover that rare-token processing emerges via \textit{distributed specialization}: functionally coordinated but spatially distributed subnetworks that exhibit three distinct organizational principles. First, we identify a reproducible three-regime influence hierarchy comprising highly influential plateau neurons(also termed as rare-token neurons), power-law decay neurons, and minimally contributing neurons, which is absent in common-token processing. Second, plateau neurons demonstrate coordinated activation patterns (reduced effective dimensionality) while remaining spatially distributed rather than forming discrete clusters. Third, these specialized mechanisms are universally accessible through standard attention pathways without requiring dedicated routing circuits. Training dynamics reveal that functional specialization emerges gradually through parameter differentiation, with specialized neurons developing increasingly heavy-tailed weight correlation spectra consistent with Heavy-Tailed Self-Regularization signatures. Our findings establish that LLMs process rare-tokens through distributed coordination within shared architectures rather than mixture-of-experts-style modularity. These results provide insights for interpretable model editing, computational efficiency optimization, and understanding emergent functional organization in transformer networks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在表示和生成稀有词令牌方面存在困难，尽管这些词在专门领域中非常重要。我们探究了LLMs是否通过离散模块化架构或分布式参数级分化发展内部的专门化机制。通过对多个模型家族的最后一层MLP神经元进行系统分析，我们发现稀有词处理是通过“分布式专门化”产生：功能协调但空间分布的子网络，表现出三种不同的组织原则。首先，我们识别出一种可重现的三层次影响层级，包括高度影响力的平台神经元（也称为稀有词神经元）、幂律衰减神经元和极小贡献神经元，而在常见词处理中不存在这种层级结构。其次，平台神经元表现出协调的激活模式（有效维度降低）且保持空间分散，而不是形成离散的集群。第三，这些专门化机制可以通过标准的注意力路径广泛访问，无需特殊路由电路。训练动态显示，功能专门化是通过参数分化逐渐产生，具有重尾自我正则化特征的权重相关谱逐渐加厚。我们的研究结果表明，LLMs通过共享架构内的分布式协调处理稀有词令牌，而非像专家混合那样的模块化。这些结果为可解释模型编辑、计算效率优化以及理解变压器网络中涌现的功能组织提供了洞察。 

---
# Embodied Representation Alignment with Mirror Neurons 

**Title (ZH)**: 镜像神经元驱动的体态表示对齐 

**Authors**: Wentao Zhu, Zhining Zhang, Yuwei Ren, Yin Huang, Hao Xu, Yizhou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21136)  

**Abstract**: Mirror neurons are a class of neurons that activate both when an individual observes an action and when they perform the same action. This mechanism reveals a fundamental interplay between action understanding and embodied execution, suggesting that these two abilities are inherently connected. Nonetheless, existing machine learning methods largely overlook this interplay, treating these abilities as separate tasks. In this study, we provide a unified perspective in modeling them through the lens of representation learning. We first observe that their intermediate representations spontaneously align. Inspired by mirror neurons, we further introduce an approach that explicitly aligns the representations of observed and executed actions. Specifically, we employ two linear layers to map the representations to a shared latent space, where contrastive learning enforces the alignment of corresponding representations, effectively maximizing their mutual information. Experiments demonstrate that this simple approach fosters mutual synergy between the two tasks, effectively improving representation quality and generalization. 

**Abstract (ZH)**: 镜像神经元是一种在个体观察一个动作和执行相同动作时都会激活的神经元。这一机制揭示了动作理解和体现执行之间的基本互动，表明这两种能力本质上是相连的。尽管现有的机器学习方法大多忽视了这种互动，将这些能力视为单独的任务。在本研究中，我们通过表示学习的角度提供了一种统一的建模视角。我们首先观察到它们的中间表示会自发对齐。受镜像神经元的启发，我们进一步引入了一种显式对齐观察动作和执行动作表示的方法。具体而言，我们使用两个线性层将表示映射到共享的潜在空间，在此空间中，对比学习促使相应的表示对齐，有效地最大化它们的互信息。实验表明，这种简单的做法促进了两项任务之间的相互协同作用，有效提高了表示质量和泛化能力。 

---
# ToMPO: Training LLM Strategic Decision Making from a Multi-Agent Perspective 

**Title (ZH)**: ToMPO：从多智能体视角训练大规模语言模型的战略决策能力 

**Authors**: Yiwen Zhang, Ziang Chen, Fanqi Kong, Yizhe Huang, Xue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2509.21134)  

**Abstract**: Large Language Models (LLMs) have been used to make decisions in complex scenarios, where they need models to think deeply, reason logically, and decide wisely. Many existing studies focus solely on multi-round conversations in social tasks or simulated environments, neglecting the various types of decisions and their interdependence. Current reinforcement learning methods struggle to consider the strategies of others during training. To address these issues, we first define a strategic decision-making problem that includes two types of decisions and their temporal dependencies. Furthermore, we propose **T**heory **o**f **M**ind **P**olicy **O**ptimization **(ToMPO)** algorithm to optimize the perception of other individual strategies and the game situation trends. Compared to the Group Relative Policy Optimization (GRPO) algorithm, ToMPO enhances the LLM's strategic decision-making mainly by: 1) generating rollouts based on reasoning the strategies of other individuals, 2) estimating advantages at both the graph-level and sample-level, and 3) balancing global and partial rewards. The ToMPO algorithm outperforms the GRPO method by 35% in terms of model output compliance and cooperative outcomes. Additionally, when compared to models with parameter sizes 100 times larger, it shows an 18% improvement. This demonstrates the effectiveness of the ToMPO algorithm in enhancing the model's strategic decision-making capabilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）在复杂场景中的战略决策问题及Theory of Mind Policy Optimization (ToMPO)算法的研究 

---
# RL Squeezes, SFT Expands: A Comparative Study of Reasoning LLMs 

**Title (ZH)**: RL压缩，SFT扩展：基于推理的LLM对比研究 

**Authors**: Kohsei Matsutani, Shota Takashiro, Gouki Minegishi, Takeshi Kojima, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2509.21128)  

**Abstract**: Large language models (LLMs) are typically trained by reinforcement learning (RL) with verifiable rewards (RLVR) and supervised fine-tuning (SFT) on reasoning traces to improve their reasoning abilities. However, how these methods shape reasoning capabilities remains largely elusive. Going beyond an accuracy-based investigation of how these two components sculpt the reasoning process, this paper introduces a novel analysis framework that quantifies reasoning paths and captures their qualitative changes under each training process (with models of 1.5B, 7B, and 14B parameters on mathematical domains). Specifically, we investigate the reasoning process at two levels of granularity: the trajectory-level, which examines complete reasoning outputs, and the step-level, which analyzes reasoning graphs whose nodes correspond to individual reasoning steps. Notably, clustering of unique reasoning trajectories shows complementary effects: RL compresses incorrect trajectories, whereas SFT expands correct ones. Step-level analysis reveals that RL steepens (about 2.5 times), while SFT flattens (reduced to about one-third), the decay rates of node visitation frequency, degree, and betweenness centrality distributions in the reasoning graph. This indicates that RL concentrates reasoning functionality into a small subset of steps, while SFT homogenizes it across many steps. Furthermore, by evaluating the reasoning graph topologies from multiple perspectives, we delineate the shared and distinct characteristics of RL and SFT. Our work presents a novel reasoning path perspective that explains why the current best practice of two-stage training, with SFT followed by RL, is successful, and offers practical implications for data construction and more efficient learning approaches. 

**Abstract (ZH)**: 大型语言模型（LLMs）通常通过可验证奖励的强化学习（RLVR）和监督微调（SFT）在推理轨迹上进行训练以提高其推理能力。然而，这些方法如何塑造推理能力仍 largely elusive。超越基于准确性的研究，本文引入了一种新的分析框架，该框架量化了推理路径并在每个训练过程中捕获其定性变化（在数学领域使用1.5B、7B和14B参数的模型）。具体而言，我们在两个粒度层次上研究推理过程：轨迹层面，检查完整的推理输出；步骤层面，分析节点对应于个体推理步骤的推理图。值得注意的是，独特推理轨迹的聚类显示了互补的效果：RL压缩了错误的轨迹，而SFT扩展了正确的轨迹。步骤层面的分析表明，RL增加了（约2.5倍），而SFT减少了（减少到约三分之一）节点访问频率、度和介数中心性的分布衰减率。这表明RL将推理功能集中在少数步骤中，而SFT则在许多步骤中使其变得均匀。此外，通过从多个角度评估推理图的拓扑结构，我们界定了RL和SFT的共享和独特特征。我们的工作提供了一种新的推理路径视角，解释了为什么当前的最佳实践两阶段训练（SFT后跟RL）是成功的，并为数据构建和更高效的学习方法提供了实用建议。 

---
# Expanding Reasoning Potential in Foundation Model by Learning Diverse Chains of Thought Patterns 

**Title (ZH)**: 通过学习多样的思考链模式扩大基础模型的推理潜力 

**Authors**: Xuemiao Zhang, Can Ren, Chengying Tu, Rongxiang Weng, Shuo Wang, Hongfei Yan, Jingang Wang, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.21124)  

**Abstract**: Recent progress in large reasoning models for challenging mathematical reasoning has been driven by reinforcement learning (RL). Incorporating long chain-of-thought (CoT) data during mid-training has also been shown to substantially improve reasoning depth. However, current approaches often utilize CoT data indiscriminately, leaving open the critical question of which data types most effectively enhance model reasoning capabilities. In this paper, we define the foundation model's reasoning potential for the first time as the inverse of the number of independent attempts required to correctly answer the question, which is strongly correlated with the final model performance. We then propose utilizing diverse data enriched with high-value reasoning patterns to expand the reasoning potential. Specifically, we abstract atomic reasoning patterns from CoT sequences, characterized by commonality and inductive capabilities, and use them to construct a core reference set enriched with valuable reasoning patterns. Furthermore, we propose a dual-granularity algorithm involving chains of reasoning patterns and token entropy, efficiently selecting high-value CoT data (CoTP) from the data pool that aligns with the core set, thereby training models to master reasoning effectively. Only 10B-token CoTP data enables the 85A6B Mixture-of-Experts (MoE) model to improve by 9.58% on the challenging AIME 2024 and 2025, and to raise the upper bound of downstream RL performance by 7.81%. 

**Abstract (ZH)**: 近期，大型推理模型在解决具有挑战性的数学推理问题方面的进展主要得益于强化学习（RL）。在中期训练过程中引入长链推理（CoT）数据也被证明能显著提升推理深度。然而，当前的方法通常不分青红皂白地使用CoT数据，这使得关于哪种数据类型最有效地增强模型推理能力的关键问题仍旧悬而未决。在本文中，我们首次将基础模型的推理潜力定义为正确回答问题所需的独立尝试次数的倒数，这一定义与最终模型性能有着密切的关联。我们随后提出利用富含高价值推理模式的多元数据来扩展这种推理潜力。具体而言，我们从CoT序列中抽象出具有共同性和归纳能力的原子推理模式，并使用这些模式构建一个丰富的核心参考集。此外，我们提出了一种双粒度算法，涉及推理模式链和标记熵，有效选择与核心集对齐的高价值CoT数据（CoTP），从而训练模型掌握有效的推理能力。仅100亿标记的CoTP数据使85A6B混合专家（MoE）模型在具有挑战性的AIME 2024和2025上性能提升了9.58%，并将下游RL性能的上限提高了7.81%。 

---
# TrustJudge: Inconsistencies of LLM-as-a-Judge and How to Alleviate Them 

**Title (ZH)**: TrustJudge: LLM-as-a-Judge的一致性问题及其缓解策略 

**Authors**: Yidong Wang, Yunze Song, Tingyuan Zhu, Xuanwang Zhang, Zhuohao Yu, Hao Chen, Chiyu Song, Qiufeng Wang, Cunxiang Wang, Zhen Wu, Xinyu Dai, Yue Zhang, Wei Ye, Shikun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21117)  

**Abstract**: The adoption of Large Language Models (LLMs) as automated evaluators (LLM-as-a-judge) has revealed critical inconsistencies in current evaluation frameworks. We identify two fundamental types of inconsistencies: (1) Score-Comparison Inconsistency, where lower-rated responses outperform higher-scored ones in pairwise comparisons, and (2) Pairwise Transitivity Inconsistency, manifested through circular preference chains (A>B>C>A) and equivalence contradictions (A=B=C\neq A). We argue that these issues come from information loss in discrete rating systems and ambiguous tie judgments during pairwise evaluation. We propose TrustJudge, a probabilistic framework that addresses these limitations through two key innovations: 1) distribution-sensitive scoring that computes continuous expectations from discrete rating probabilities, preserving information entropy for more precise scoring, and 2) likelihood-aware aggregation that resolves transitivity violations using bidirectional preference probabilities or perplexity. We also formalize the theoretical limitations of current LLM-as-a-judge frameworks and demonstrate how TrustJudge's components overcome them. When evaluated with Llama-3.1-70B-Instruct as judge using our dataset, TrustJudge reduces Score-Comparison inconsistency by 8.43% (from 23.32% to 14.89%) and Pairwise Transitivity inconsistency by 10.82% (from 15.22% to 4.40%), while maintaining higher evaluation accuracy. Our work provides the first systematic analysis of evaluation framework inconsistencies in LLM-as-a-judge paradigms, offering both theoretical insights and practical solutions for reliable automated assessment. The framework demonstrates consistent improvements across various model architectures and scales, enabling more trustworthy LLM evaluation without requiring additional training or human annotations. The codes can be found at this https URL. 

**Abstract (ZH)**: 大型语言模型作为自动化评估员（LLM-as-a-judge）的采用揭示了当前评估框架中的关键不一致性。我们识别出两种基本类型的不一致性：（1）评分比较不一致性，其中评分较低的回答在成对比较中表现优于评分较高的回答；（2）成对评价传递不一致性，表现为循环偏好链（A>B>C>A）和等价矛盾（A=B=C≠A）。我们认为这些问题源自离散评分系统中的信息丢失以及成对评价期间模糊的并列判断。我们提出了TrustJudge，一种概率框架，通过两项关键创新解决这些问题：（1）基于分布的评分，从离散评分概率计算连续期望，保留信息熵以实现更精确的评分；（2）概率感知聚合，使用双向偏好概率或困惑度解决传递性违例。我们还提出了当前LLM-as-a-judge框架的理论局限性，并展示了TrustJudge各个组件如何克服这些局限性。使用我们的数据集和Llama-3.1-70B-Instruct作为评估员进行评估时，TrustJudge将评分比较不一致性降低了8.43%（从23.32%降至14.89%），成对评价传递不一致性降低了10.82%（从15.22%降至4.40%），同时保持了更高的评估准确性。我们的工作首次对LLM-as-a-judge范式中的评估框架不一致性进行了系统的分析，提供了可靠自动评估的理论见解和解决方案。该框架在各种模型架构和规模上表现出一致改进，无需额外训练或人工标注即可实现更可信的大型语言模型评估。代码可从此处获取：此链接。 

---
# Recon-Act: A Self-Evolving Multi-Agent Browser-Use System via Web Reconnaissance, Tool Generation, and Task Execution 

**Title (ZH)**: Recon-Act: 一种基于网络侦察、工具生成和任务执行的自演化多Agent浏览器使用系统 

**Authors**: Kaiwen He, Zhiwei Wang, Chenyi Zhuang, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21072)  

**Abstract**: Recent years, multimodal models have made remarkable strides and pave the way for intelligent browser use agents. However, when solving tasks on real world webpages in multi-turn, long-horizon trajectories, current agents still suffer from disordered action sequencing and excessive trial and error during execution. This paper introduces Recon-Act, a self-evolving multi-agent framework grounded in Reconnaissance-Action behavioral paradigm. The system comprises a Reconnaissance Team and an Action Team: the former conducts comparative analysis and tool generation, while the latter handles intent decomposition, tool orchestration, and execution. By contrasting the erroneous trajectories with successful ones, the Reconnaissance Team infers remedies, and abstracts them into a unified notion of generalized tools, either expressed as hints or as rule-based codes, and register to the tool archive in real time. The Action Team reinference the process empowered with these targeting tools, thus establishing a closed-loop training pipeline of data-tools-action-feedback. Following the 6 level implementation roadmap proposed in this work, we have currently reached Level 3 (with limited human-in-the-loop intervention). Leveraging generalized tools obtained through reconnaissance, Recon-Act substantially improves adaptability to unseen websites and solvability on long-horizon tasks, and achieves state-of-the-art performance on the challenging VisualWebArena dataset. 

**Abstract (ZH)**: Recent年，多模态模型取得了显著进展，并为智能浏览器使用代理铺平了道路。然而，在解决现实世界网页上的多轮、长周期任务时，当前代理仍然受到行动顺序混乱和执行过程中过度尝试与错误的困扰。本文介绍了一种基于侦察-行动行为范式的自进化多代理框架Recon-Act。该系统包括侦察团队和行动团队：前者进行对比分析和工具生成，后者处理意图分解、工具编排和执行。通过对比错误轨迹与成功轨迹，侦察团队推断出修正方法，并将其抽象为通用工具概念，这些工具可以作为提示或基于规则的代码形式存在，并实时注册到工具库中。行动团队利用这些目标工具重构过程，从而建立数据-工具-行动-反馈的闭环训练管道。按照本文提出的6级实施路线图，我们目前达到了第3级（有限的人工干预下）。通过侦察获得的通用工具，Recon-Act大幅提高了对未见过的网站的适应性和在长周期任务中的可解性，并在具有挑战性的VisualWebArena数据集上实现了最先进的性能。 

---
# Disagreements in Reasoning: How a Model's Thinking Process Dictates Persuasion in Multi-Agent Systems 

**Title (ZH)**: 推理中的分歧：模型思维过程如何在多代理系统中影响说服力 

**Authors**: Haodong Zhao, Jidong Li, Zhaomin Wu, Tianjie Ju, Zhuosheng Zhang, Bingsheng He, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21054)  

**Abstract**: The rapid proliferation of recent Multi-Agent Systems (MAS), where Large Language Models (LLMs) and Large Reasoning Models (LRMs) usually collaborate to solve complex problems, necessitates a deep understanding of the persuasion dynamics that govern their interactions. This paper challenges the prevailing hypothesis that persuasive efficacy is primarily a function of model scale. We propose instead that these dynamics are fundamentally dictated by a model's underlying cognitive process, especially its capacity for explicit reasoning. Through a series of multi-agent persuasion experiments, we uncover a fundamental trade-off we term the Persuasion Duality. Our findings reveal that the reasoning process in LRMs exhibits significantly greater resistance to persuasion, maintaining their initial beliefs more robustly. Conversely, making this reasoning process transparent by sharing the "thinking content" dramatically increases their ability to persuade others. We further consider more complex transmission persuasion situations and reveal complex dynamics of influence propagation and decay within multi-hop persuasion between multiple agent networks. This research provides systematic evidence linking a model's internal processing architecture to its external persuasive behavior, offering a novel explanation for the susceptibility of advanced models and highlighting critical implications for the safety, robustness, and design of future MAS. 

**Abstract (ZH)**: 近期多Agent系统（MAS）的迅猛发展，其中大型语言模型（LLMs）和大型推理模型（LRMs）通常协作解决复杂问题， necessitates 对于调控它们交互的劝说动态的深刻理解。本文挑战了目前认为劝说有效性主要取决于模型规模的观点。相反，我们提出这些动态从根本上由模型的内在认知过程，特别是其显式推理能力所决定。通过一系列多Agent劝说实验，我们揭示了一种基本的权衡关系，我们称之为劝说二元性。研究发现，LRMs中的推理过程对劝说表现出显著的抵抗力，能够更加牢固地维持其初始信念。相反，通过分享“思考内容”使这一推理过程变得透明，极大地增强了它们说服他人的能力。我们进一步考虑更复杂的跨跳劝说情景，揭示了多Agent网络间多跳劝说中影响传播与衰减的复杂动态。本研究提供了系统性证据，将模型的内部处理架构与其外部劝说行为联系起来，提出了一种新的解释，揭示了高级模型的易受影响性，并对未来的MAS的安全性、鲁棒性和设计提出了关键影响。 

---
# Combinatorial Creativity: A New Frontier in Generalization Abilities 

**Title (ZH)**: 组合创造性：泛化能力的新前沿 

**Authors**: Samuel Schapiro, Sumuk Shashidhar, Alexi Gladstone, Jonah Black, Royce Moon, Dilek Hakkani-Tur, Lav R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2509.21043)  

**Abstract**: Artificial intelligence (AI) systems, and large language models (LLMs) in particular, are increasingly employed for creative tasks like scientific idea generation, constituting a form of generalization from training data unaddressed by existing conceptual frameworks. Though in many ways similar to forms of compositional generalization (CG), combinatorial creativity (CC) is an open-ended ability. Instead of evaluating for accuracy or correctness against fixed targets, which would contradict the open-ended nature of CC, we propose a theoretical framework and algorithmic task for evaluating outputs by their degrees of novelty and utility. From here, we make several important empirical contributions: (1) We obtain the first insights into the scaling behavior of creativity for LLMs. (2) We discover that, for fixed compute budgets, there exist optimal model depths and widths for creative ability. (3) We find that the ideation-execution gap, whereby LLMs excel at generating novel scientific ideas but struggle to ensure their practical feasibility, may be explained by a more fundamental novelty-utility tradeoff characteristic of creativity algorithms in general. Importantly, this tradeoff remains persistent even at scale, casting doubt on the long-term creative potential of LLMs in their current form. Together, our conceptual framework and empirical findings provide a foundation for understanding and improving creativity in modern AI models, marking a new frontier in generalization abilities. 

**Abstract (ZH)**: 人工智能系统，特别是大型语言模型（LLMs），正越来越多地被用于创意任务，如科学构想的生成，构成了现有概念框架未能解决的一种泛化形式。尽管在许多方面与组合泛化（CG）类似，组合性创造性（CC）是一种开放性能力。我们不通过与固定目标的准确性和正确性评估来限制其开放性，而是提出一个理论框架和算法任务，通过新颖性和实用性程度来评估输出。在此基础上，我们做出了几个重要的实证贡献：（1）我们获得了LLMs创造性行为缩放的第一手见解。（2）我们发现，在固定计算预算下，存在实现创造性能力的最优模型深度和宽度。（3）我们发现，构想-执行差距，即LLMs在生成新颖科学构想方面表现出色但在确保其实用可行性方面遇到困难的现象，可能是由创造性算法普遍存在的新颖性-实用性权衡所解释的。重要的是，即使在较大规模下，这种权衡依然存在，这对LLMs当前形式的长期创造性潜力提出了质疑。结合我们的概念框架和实证发现，为我们理解并改进现代AI模型中的创造性能力奠定了基础，标志着泛化能力的一种新的前沿。 

---
# CLAUSE: Agentic Neuro-Symbolic Knowledge Graph Reasoning via Dynamic Learnable Context Engineering 

**Title (ZH)**: CLAUSE: 通过动态可学习上下文工程实现代理神经-符号知识图谱推理 

**Authors**: Yang Zhao, Chengxiao Dai, Wei Zhuo, Yue Xiu, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2509.21035)  

**Abstract**: Knowledge graphs provide structured context for multi-hop question answering, but deployed systems must balance answer accuracy with strict latency and cost targets while preserving provenance. Static k-hop expansions and "think-longer" prompting often over-retrieve, inflate context, and yield unpredictable runtime. We introduce CLAUSE, an agentic three-agent neuro-symbolic framework that treats context construction as a sequential decision process over knowledge graphs, deciding what to expand, which paths to follow or backtrack, what evidence to keep, and when to stop. Latency (interaction steps) and prompt cost (selected tokens) are exposed as user-specified budgets or prices, allowing per-query adaptation to trade-offs among accuracy, latency, and cost without retraining. CLAUSE employs the proposed Lagrangian-Constrained Multi-Agent Proximal Policy Optimization (LC-MAPPO) algorithm to coordinate three agents: Subgraph Architect, Path Navigator, and Context Curator, so that subgraph construction, reasoning-path discovery, and evidence selection are jointly optimized under per-query resource budgets on edge edits, interaction steps, and selected tokens. Across HotpotQA, MetaQA, and FactKG, CLAUSE yields higher EM@1 while reducing subgraph growth and end-to-end latency at equal or lower token budgets. On MetaQA-2-hop, relative to the strongest RAG baseline (GraphRAG), CLAUSE achieves +39.3 EM@1 with 18.6% lower latency and 40.9% lower edge growth. The resulting contexts are compact, provenance-preserving, and deliver predictable performance under deployment constraints. 

**Abstract (ZH)**: CLAUSE：基于神经符号框架的动态知识图推理代理方法 

---
# Who Gets Cited Most? Benchmarking Long-Context Language Models on Scientific Articles 

**Title (ZH)**: 谁被引用最多？科学文章中长上下文语言模型的基准测试 

**Authors**: Miao Li, Alexander Gurung, Irina Saparina, Mirella Lapata  

**Link**: [PDF](https://arxiv.org/pdf/2509.21028)  

**Abstract**: This paper introduces SciTrek, a novel question-answering benchmark designed to evaluate the long-context reasoning capabilities of large language models (LLMs) using scientific articles. Current long-context benchmarks often rely on non-scientific texts, focus on simple information retrieval tasks, or employ artificial contexts. SciTrek addresses these limitations by proposing complex questions that require information aggregation and synthesis across multiple full-text scientific articles. Questions and their ground-truth answers are automatically generated by formulating them as SQL queries over a database constructed from article metadata (titles, authors, and references). The SQL operations provide explicit, verifiable reasoning steps for fine-grained error analysis, and the construction process scales to contexts up to 1M tokens with minimal supervision. Extensive experiments on a diverse set of open-weight and proprietary LLMs demonstrate that SciTrek poses a significant challenge as the context length increases, with supervised fine-tuning and reinforcement learning offering only limited gains. Our analysis reveals systematic shortcomings in models' abilities to perform basic numerical operations and accurately locate specific information in long contexts. 

**Abstract (ZH)**: SciTrek：一种基于科学文章的新型长上下文推理基准测试 

---
# CORE: Full-Path Evaluation of LLM Agents Beyond Final State 

**Title (ZH)**: CORE: 超越最终状态的大型语言模型代理全面路径评估 

**Authors**: Panagiotis Michelakis, Yiannis Hadjiyiannis, Dimitrios Stamoulis  

**Link**: [PDF](https://arxiv.org/pdf/2509.20998)  

**Abstract**: Evaluating AI agents that solve real-world tasks through function-call sequences remains an open challenge. Existing agentic benchmarks often reduce evaluation to a binary judgment of the final state, overlooking critical aspects such as safety, efficiency, and intermediate correctness. We propose a framework based on deterministic finite automata (DFAs) that encodes tasks as sets of valid tool-use paths, enabling principled assessment of agent behavior in diverse world models. Building on this foundation, we introduce CORE, a suite of five metrics, namely Path Correctness, Path Correctness - Kendall's tau Composite, Prefix Criticality, Harmful-Call Rate, and Efficiency, that quantify alignment with expected execution patterns. Across diverse worlds, our method reveals important performance differences between agents that would otherwise appear equivalent under traditional final-state evaluation schemes. 

**Abstract (ZH)**: 通过对实际任务通过函数调用序列解决的AI代理进行评估仍然是一个开放的挑战。现有的代理基准通常将评估简化为最终状态的二元判断，忽视了诸如安全、效率和中间正确性等关键方面。我们提出了基于确定性有限自动机（DFAs）的框架，将任务编码为有效的工具使用路径集，从而能够对在不同世界模型中代理行为进行原则性的评估。在此基础上，我们引入了CORE套件，包括路径正确性、路径正确性-肯德尔 tau 复合、前缀关键性、有害调用率和效率等五项指标，以量化与预期执行模式的契合度。在不同的世界中，我们的方法揭示了传统最终状态评估方案下看似等效的代理之间的重要性能差异。 

---
# AOT*: Efficient Synthesis Planning via LLM-Empowered AND-OR Tree Search 

**Title (ZH)**: AOT*: 通过LLM赋能的AND-OR树搜索高效合成规划 

**Authors**: Xiaozhuang Song, Xuanhao Pan, Xinjian Zhao, Hangting Ye, Shufei Zhang, Jian Tang, Tianshu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20988)  

**Abstract**: Retrosynthesis planning enables the discovery of viable synthetic routes for target molecules, playing a crucial role in domains like drug discovery and materials design. Multi-step retrosynthetic planning remains computationally challenging due to exponential search spaces and inference costs. While Large Language Models (LLMs) demonstrate chemical reasoning capabilities, their application to synthesis planning faces constraints on efficiency and cost. To address these challenges, we introduce AOT*, a framework that transforms retrosynthetic planning by integrating LLM-generated chemical synthesis pathways with systematic AND-OR tree search. To this end, AOT* atomically maps the generated complete synthesis routes onto AND-OR tree components, with a mathematically sound design of reward assignment strategy and retrieval-based context engineering, thus enabling LLMs to efficiently navigate in the chemical space. Experimental evaluation on multiple synthesis benchmarks demonstrates that AOT* achieves SOTA performance with significantly improved search efficiency. AOT* exhibits competitive solve rates using 3-5$\times$ fewer iterations than existing LLM-based approaches, with the efficiency advantage becoming more pronounced on complex molecular targets. 

**Abstract (ZH)**: 逆合成规划能够发现目标分子的可行合成路径，在药物发现和材料设计等领域发挥着关键作用。多步逆合成规划由于搜索空间和推理成本的指数增长而具有计算挑战性。尽管大型语言模型（LLMs）展示了化学推理能力，但将其应用于合成规划在效率和成本方面存在限制。为应对这些挑战，我们引入了AOT*框架，该框架通过将LLM生成的化学合成路径与系统性的AND-OR树搜索相结合，来转变逆合成规划。AOT*原子化地将生成的完整合成路径映射到AND-OR树组件上，并采用数学上稳健的奖励分配策略和基于检索的上下文工程设计，从而使LLM能够高效地在化学空间中导航。在多个合成基准上的实验评估表明，AOT*在显著提高搜索效率的同时达到了SOTA性能。AOT*使用3-5倍 fewer迭代次数实现与现有LLM基方法相当的解算率，在复杂分子目标上效率优势更为明显。 

---
# Beyond Stars: Bridging the Gap Between Ratings and Review Sentiment with LLM 

**Title (ZH)**: 超越星星：利用大规模语言模型弥合评分与评论情感之间的差距 

**Authors**: Najla Zuhir, Amna Mohammad Salim, Parvathy Premkumar, Moshiur Farazi  

**Link**: [PDF](https://arxiv.org/pdf/2509.20953)  

**Abstract**: We present an advanced approach to mobile app review analysis aimed at addressing limitations inherent in traditional star-rating systems. Star ratings, although intuitive and popular among users, often fail to capture the nuanced feedback present in detailed review texts. Traditional NLP techniques -- such as lexicon-based methods and classical machine learning classifiers -- struggle to interpret contextual nuances, domain-specific terminology, and subtle linguistic features like sarcasm. To overcome these limitations, we propose a modular framework leveraging large language models (LLMs) enhanced by structured prompting techniques. Our method quantifies discrepancies between numerical ratings and textual sentiment, extracts detailed, feature-level insights, and supports interactive exploration of reviews through retrieval-augmented conversational question answering (RAG-QA). Comprehensive experiments conducted on three diverse datasets (AWARE, Google Play, and Spotify) demonstrate that our LLM-driven approach significantly surpasses baseline methods, yielding improved accuracy, robustness, and actionable insights in challenging and context-rich review scenarios. 

**Abstract (ZH)**: 我们提出了一种先进的移动应用审查分析方法，旨在解决传统星级评价系统固有的局限性。尽管星级评价直观且受到用户欢迎，但往往无法捕捉到详细审查文本中细腻的反馈。传统自然语言处理技术——如基于词典的方法和经典机器学习分类器——难以解释上下文细微差别、领域特定术语以及像讽刺等微妙的语言特征。为克服这些局限性，我们提出了一种利用大型语言模型（LLMs）并结合结构化提示技术的模块化框架。该方法量化了数值评分与文本情感之间的差异，提取了详细的功能级洞察，并通过检索增强的对话式问题回答（RAG-QA）支持交互式审查探索。在三个多样化的数据集（AWARE、Google Play和Spotify）上进行的全面实验表明，我们的基于LLM的方法显著超越了基线方法，在具有挑战性和上下文丰富的审查场景中提供了更好的准确度、鲁棒性和可操作的洞察。 

---
# GALAX: Graph-Augmented Language Model for Explainable Reinforcement-Guided Subgraph Reasoning in Precision Medicine 

**Title (ZH)**: GALAX：图增强语言模型在精准医疗中可解释的强化引导子图推理 

**Authors**: Heming Zhang, Di Huang, Wenyu Li, Michael Province, Yixin Chen, Philip Payne, Fuhai Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.20935)  

**Abstract**: In precision medicine, quantitative multi-omic features, topological context, and textual biological knowledge play vital roles in identifying disease-critical signaling pathways and targets. Existing pipelines capture only part of these-numerical omics ignore topological context, text-centric LLMs lack quantitative grounded reasoning, and graph-only models underuse node semantics and the generalization of LLMs-limiting mechanistic interpretability. Although Process Reward Models (PRMs) aim to guide reasoning in LLMs, they remain limited by unreliable intermediate evaluation, and vulnerability to reward hacking with computational cost. These gaps motivate integrating quantitative multi-omic signals, topological structure with node annotations, and literature-scale text via LLMs, using subgraph reasoning as the principle bridge linking numeric evidence, topological knowledge and language context. Therefore, we propose GALAX (Graph Augmented LAnguage model with eXplainability), an innovative framework that integrates pretrained Graph Neural Networks (GNNs) into Large Language Models (LLMs) via reinforcement guided by a Graph Process Reward Model (GPRM), which generates disease-relevant subgraphs in a step-wise manner initiated by an LLM and iteratively evaluated by a pretrained GNN, enabling process-level supervision without explicit intermediate reasoning annotations. As an application, we also introduced Target-QA, a benchmark combining CRISPR-identified targets, multi-omic profiles, and biomedical graph knowledge across diverse cancer cell lines, which enables GNN pretraining for supervising step-wise graph construction and supports long-context reasoning over text-numeric graphs (TNGs), providing a scalable and biologically grounded framework for explainable, reinforcement-guided subgraph reasoning toward reliable and interpretable target and pathway discovery in precision medicine. 

**Abstract (ZH)**: 在精准医疗中，定量多组学特征、拓扑上下文和文本生物知识在识别疾病关键信号通路和靶点中起着关键作用。现有管道只能捕获其中的一部分——数值型组学忽略了拓扑上下文，文本中心的大语言模型缺乏定量的基于事实的推理能力，而仅基于图的模型未能充分利用节点语义和大语言模型的一般化能力——这限制了其机制解释能力。虽然过程奖励模型（PRMs）旨在引导大语言模型的推理，但它们仍受限于不可靠的中间评价，且容易受到计算成本带来的奖励作弊的影响。这些差距促使我们将定量多组学信号、拓扑结构及其节点注释与大规模文献文本集成到大语言模型中，以子图推理作为原理性的桥梁，连接数值证据、拓扑知识和语言上下文。因此，我们提出了一种名为GALAX（Graph Augmented LAnguage model with eXplainability）的创新框架，通过强化学习引导，将预训练图神经网络（GNNs）整合到大型语言模型（LLMs）中，使用图过程奖励模型（GPRM）生成与疾病相关的子图，该过程由LLM启动并由预训练的GNN逐步评估，从而实现过程层面的监督，而无需明确的中间推理注解。作为应用，我们还引入了Target-QA基准，该基准结合了CRISPR鉴定的靶点、多组学特征以及跨多种癌细胞系的生物医学图知识，以支持监督步骤的图构建和长上下文文本-数值图（TNG）推理，提供了一个可扩展且具有生物基础的解释性强化学习引导子图推理框架，用于精准医疗中可靠且可解释的目标和通路发现。 

---
# DeFacto: Counterfactual Thinking with Images for Enforcing Evidence-Grounded and Faithful Reasoning 

**Title (ZH)**: DeFacto: 基于图像的反事实思考以确保证据驱动和忠实的推理 

**Authors**: Tianrun Xu, Haoda Jing, Ye Li, Yuquan Wei, Jun Feng, Guanyu Chen, Haichuan Gao, Tianren Zhang, Feng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.20912)  

**Abstract**: Recent advances in multimodal language models (MLLMs) have achieved remarkable progress in vision-language reasoning, especially with the emergence of "thinking with images," which integrates explicit visual steps into the reasoning process. While this paradigm strengthens image-based reasoning, a significant challenge remains: models may arrive at correct answers by relying on irrelevant or spurious regions, driven by prior knowledge or dataset biases. Even when the answer is correct, flawed reasoning indicates that the model has not truly understood the image, highlighting the critical importance of reasoning fidelity in multimodal tasks. To address this issue, we propose DeFacto, a counterfactual reasoning framework that jointly enforces accurate answering and faithful reasoning. A key component of our approach is the design of three complementary training paradigms: (i) positive, (ii) counterfactual, and (iii) random-masking. To enable these paradigms, we develop a pipeline that automatically localizes question-relevant evidence and constructs positive, counterfactual, and random variants, resulting in a dataset of about 100k images. Building on this framework, we train multimodal language models with GRPO-based reinforcement learning, where we design three complementary rewards to guide the model toward accurate answering and evidence-grounded reasoning. Experiments on diverse benchmarks demonstrate that DeFacto substantially improves both answer accuracy and reasoning faithfulness, establishing a stronger foundation for interpretable multimodal reasoning. The code is available on GitHub and the dataset is released on HuggingFace. 

**Abstract (ZH)**: Recent Advances in Multimodal Language Models: Addressing Challenges in Vision-Language Reasoning with DeFacto 

---
# LogReasoner: Empowering LLMs with Expert-like Coarse-to-Fine Reasoning for Log Analysis Tasks 

**Title (ZH)**: LogReasoner: 为日志分析任务赋予类似专家的精细到粗糙推理能力的LLMs 

**Authors**: Lipeng Ma, Yixuan Li, Weidong Yang, Mingjie Zhou, Xinyi Liu, Ben Fei, Shuhao Li, Xiaoyan Sun, Sihang Jiang, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2509.20798)  

**Abstract**: Log analysis is crucial for monitoring system health and diagnosing failures in complex systems. Recent advances in large language models (LLMs) offer new opportunities for automated log analysis, leveraging their reasoning capabilities to perform tasks such as anomaly detection and failure prediction. However, general-purpose LLMs struggle to formulate structured reasoning workflows that align with expert cognition and deliver precise details of reasoning steps. To address these challenges, we propose LogReasoner, a coarse-to-fine reasoning enhancement framework designed to enable LLMs to reason log analysis tasks like experts. LogReasoner consists of two stages: (1) coarse-grained enhancement of expert thinking, where high-level expert thoughts are constructed from collected troubleshooting flowcharts and existing tasks to enable LLMs to formulate structured reasoning workflows and (2) fine-grained enhancement of specific steps, where we first fine-tune the LLM with task-specific stepwise solutions to enhance the LLM for instantiated reasoning, then employ the preference learning to calibrate the LLM's reasoning details from its mistakes, further strengthen the LLM's analytical granularity and correctness. We evaluate LogReasoner on four distinct log analysis tasks using open-source LLMs such as Qwen-2.5 and Llama-3. Experimental results show that LogReasoner significantly outperforms existing LLMs, achieving state-of-the-art performance and demonstrating its effectiveness in enhancing the reasoning capabilities of LLMs for log analysis. 

**Abstract (ZH)**: 日志分析对于监控系统健康状况和诊断复杂系统的故障至关重要。近期大语言模型（LLMs）的发展为自动化日志分析提供了新机遇，利用其推理能力执行异常检测和故障预测等任务。然而，通用的大语言模型在制定与专家认知相一致的结构化推理流程并提供精确的推理步骤细节方面存在困难。为了解决这些挑战，我们提出了一种粗粒度到细粒度推理增强框架LogReasoner，旨在使大语言模型能够在日志分析任务中像专家一样进行推理。LogReasoner包含两个阶段：（1）粗粒度的专家思维增强，通过收集的故障排查流程图和现有的任务构建高层次的专家思维，使大语言模型能够制定结构化的推理流程；（2）细粒度的具体步骤增强，在将大语言模型针对特定任务的步骤化解决方案进行微调以增强其实例化推理能力后，利用偏好学习校准大语言模型的推理细节，进一步增强其分析的细致程度和正确性。我们在使用Qwen-2.5和Llama-3等开源大语言模型的四种不同日志分析任务上评估了LogReasoner。实验结果表明，LogReasoner显著优于现有大语言模型，实现了业内领先的性能，并证明了其在增强大语言模型日志分析推理能力方面的有效性。 

---
# Meta-Memory: Retrieving and Integrating Semantic-Spatial Memories for Robot Spatial Reasoning 

**Title (ZH)**: 元记忆：检索和整合语义-空间记忆以进行机器人空间推理 

**Authors**: Yufan Mao, Hanjing Ye, Wenlong Dong, Chengjie Zhang, Hong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20754)  

**Abstract**: Navigating complex environments requires robots to effectively store observations as memories and leverage them to answer human queries about spatial locations, which is a critical yet underexplored research challenge. While prior work has made progress in constructing robotic memory, few have addressed the principled mechanisms needed for efficient memory retrieval and integration. To bridge this gap, we propose Meta-Memory, a large language model (LLM)-driven agent that constructs a high-density memory representation of the environment. The key innovation of Meta-Memory lies in its capacity to retrieve and integrate relevant memories through joint reasoning over semantic and spatial modalities in response to natural language location queries, thereby empowering robots with robust and accurate spatial reasoning capabilities. To evaluate its performance, we introduce SpaceLocQA, a large-scale dataset encompassing diverse real-world spatial question-answering scenarios. Experimental results show that Meta-Memory significantly outperforms state-of-the-art methods on both the SpaceLocQA and the public NaVQA benchmarks. Furthermore, we successfully deployed Meta-Memory on real-world robotic platforms, demonstrating its practical utility in complex environments. Project page: this https URL . 

**Abstract (ZH)**: 导航复杂环境要求机器人有效地存储观察作为记忆，并利用这些记忆回答关于空间位置的人类查询，这是一个关键但尚未充分探索的研究挑战。尽管之前的研究所在这方面取得进展，但很少有人解决高效记忆检索和集成的原则机制。为弥补这一差距，我们提出了一种元记忆（Meta-Memory），这是一种由大规模语言模型（LLM）驱动的代理，能够构建环境的高密度记忆表示。元记忆的关键创新在于其通过联合推理语义和空间模态来检索和整合相关记忆的能力，以响应自然语言的空间位置查询，从而赋予机器人强大的空间推理能力。为了评估其性能，我们引入了一种大规模数据集SpaceLocQA，涵盖了多种真实世界的空间问答场景。实验结果表明，元记忆在SpaceLocQA和现有的NaVQA基准测试中显著优于最先进的方法。此外，我们成功将元记忆部署到实际的机器人平台上，证明了其在复杂环境中的实用价值。项目页面：this https URL。 

---
# Parallel Thinking, Sequential Answering: Bridging NAR and AR for Efficient Reasoning 

**Title (ZH)**: 并行思考，顺序作答：链接NAR和AR以实现高效推理 

**Authors**: Qihang Ai, Haiyun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20744)  

**Abstract**: We study reasoning tasks through a framework that integrates auto-regressive (AR) and non-autoregressive (NAR) language models. AR models, which generate text sequentially, excel at producing coherent outputs but often suffer from slow inference, particularly in reasoning-intensive domains such as mathematics and code, where lengthy chains of thought are required. In contrast, NAR models, such as discrete diffusion models, allow parallel generation and offer substantial speedups, though typically at the cost of reduced output quality. To address these limitations, we introduce a new paradigm in which an NAR model efficiently produces intermediate reasoning traces, which subsequently guide an AR model to deliver precise final answers. Experiments demonstrate that our approach yields significant 26% improvements over strong baselines while substantially reducing inference cost. 

**Abstract (ZH)**: 我们通过将自回归（AR）和非自回归（NAR）语言模型集成的框架来研究推理任务。实验表明，我们的方法在强基线基础上实现了显著的26%改进，并大幅降低了推理成本。 

---
# Fairy: Interactive Mobile Assistant to Real-world Tasks via LMM-based Multi-agent 

**Title (ZH)**: Fairy: 基于LMM的多agent交互式移动助手用于现实世界任务 

**Authors**: Jiazheng Sun, Te Yang, Jiayang Niu, Mingxuan Li, Yongyong Lu, Ruimeng Yang, Xin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2509.20729)  

**Abstract**: Large multi-modal models (LMMs) have advanced mobile GUI agents. However, existing methods struggle with real-world scenarios involving diverse app interfaces and evolving user needs. End-to-end methods relying on model's commonsense often fail on long-tail apps, and agents without user interaction act unilaterally, harming user experience. To address these limitations, we propose Fairy, an interactive multi-agent mobile assistant capable of continuously accumulating app knowledge and self-evolving during usage. Fairy enables cross-app collaboration, interactive execution, and continual learning through three core modules:(i) a Global Task Planner that decomposes user tasks into sub-tasks from a cross-app view; (ii) an App-Level Executor that refines sub-tasks into steps and actions based on long- and short-term memory, achieving precise execution and user interaction via four core agents operating in dual loops; and (iii) a Self-Learner that consolidates execution experience into App Map and Tricks. To evaluate Fairy, we introduce RealMobile-Eval, a real-world benchmark with a comprehensive metric suite, and LMM-based agents for automated scoring. Experiments show that Fairy with GPT-4o backbone outperforms the previous SoTA by improving user requirement completion by 33.7% and reducing redundant steps by 58.5%, showing the effectiveness of its interaction and self-learning. 

**Abstract (ZH)**: 大型多模态模型（LMMs）已推进了移动GUI代理的发展。然而，现有的方法在处理涉及多样化应用界面和不断变化的用户需求的现实场景时存在困难。依赖模型常识的端到端方法在长尾应用上往往失败，而不与用户互动的代理会单方面行动，损害用户体验。为解决这些限制，我们提出Fairy，一个具备在使用过程中连续积累应用知识和自我进化的交互多代理移动助手。Fairy通过三个核心模块实现跨应用协作、互动执行和持续学习：(i) 全局任务规划器，从跨应用视角分解用户任务；(ii) 应用级执行器，基于长期和短期记忆细化子任务为步骤和行动，在双环中通过四个核心代理实现精确执行和用户互动；和(iii) 自我学习器，将执行经验整合为应用地图和技巧。为了评估Fairy，我们引入RealMobile-Eval，一个包含全面指标套件的真实世界基准，并使用基于LMM的代理进行自动化评分。实验结果显示，以GPT-4o为骨干的Fairy在满足用户需求方面比之前的最佳方案提高了33.7%，减少了58.5%的冗余步骤，证明了其互动和自我学习的有效性。 

---
# An Automated Retrieval-Augmented Generation LLaMA-4 109B-based System for Evaluating Radiotherapy Treatment Plans 

**Title (ZH)**: 基于LLaMA-4 109B的自动化检索增强生成系统用于评估放射治疗计划 

**Authors**: Junjie Cui, Peilong Wang, Jason Holmes, Leshan Sun, Michael L. Hinni, Barbara A. Pockaj, Sujay A. Vora, Terence T. Sio, William W. Wong, Nathan Y. Yu, Steven E. Schild, Joshua R. Niska, Sameer R. Keole, Jean-Claude M. Rwigema, Samir H. Patel, Lisa A. McGee, Carlos A. Vargas, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20707)  

**Abstract**: Purpose: To develop a retrieval-augmented generation (RAG) system powered by LLaMA-4 109B for automated, protocol-aware, and interpretable evaluation of radiotherapy treatment plans.
Methods and Materials: We curated a multi-protocol dataset of 614 radiotherapy plans across four disease sites and constructed a knowledge base containing normalized dose metrics and protocol-defined constraints. The RAG system integrates three core modules: a retrieval engine optimized across five SentenceTransformer backbones, a percentile prediction component based on cohort similarity, and a clinical constraint checker. These tools are directed by a large language model (LLM) using a multi-step prompt-driven reasoning pipeline to produce concise, grounded evaluations.
Results: Retrieval hyperparameters were optimized using Gaussian Process on a scalarized loss function combining root mean squared error (RMSE), mean absolute error (MAE), and clinically motivated accuracy thresholds. The best configuration, based on all-MiniLM-L6-v2, achieved perfect nearest-neighbor accuracy within a 5-percentile-point margin and a sub-2pt MAE. When tested end-to-end, the RAG system achieved 100% agreement with the computed values by standalone retrieval and constraint-checking modules on both percentile estimates and constraint identification, confirming reliable execution of all retrieval, prediction and checking steps.
Conclusion: Our findings highlight the feasibility of combining structured population-based scoring with modular tool-augmented reasoning for transparent, scalable plan evaluation in radiation therapy. The system offers traceable outputs, minimizes hallucination, and demonstrates robustness across protocols. Future directions include clinician-led validation, and improved domain-adapted retrieval models to enhance real-world integration. 

**Abstract (ZH)**: 目的：开发由LLaMA-4 109B驱动的检索增强生成（RAG）系统，以实现自动化、协议意识和可解释的放射治疗计划评估。 

---
# Accelerate Creation of Product Claims Using Generative AI 

**Title (ZH)**: 使用生成式AI加速产品声明创建 

**Authors**: Po-Yu Liang, Yong Zhang, Tatiana Hwa, Aaron Byers  

**Link**: [PDF](https://arxiv.org/pdf/2509.20652)  

**Abstract**: The benefit claims of a product is a critical driver of consumers' purchase behavior. Creating product claims is an intense task that requires substantial time and funding. We have developed the $\textbf{Claim Advisor}$ web application to accelerate claim creations using in-context learning and fine-tuning of large language models (LLM). $\textbf{Claim Advisor}$ was designed to disrupt the speed and economics of claim search, generation, optimization, and simulation. It has three functions: (1) semantically searching and identifying existing claims and/or visuals that resonate with the voice of consumers; (2) generating and/or optimizing claims based on a product description and a consumer profile; and (3) ranking generated and/or manually created claims using simulations via synthetic consumers. Applications in a consumer packaged goods (CPG) company have shown very promising results. We believe that this capability is broadly useful and applicable across product categories and industries. We share our learning to encourage the research and application of generative AI in different industries. 

**Abstract (ZH)**: 产品的收益声称是驱动消费者购买行为的关键因素。创建产品声称是一项艰巨的任务，需要大量时间与资金。我们开发了名为$\textbf{Claim Advisor}$的网络应用，利用上下文学习和大型语言模型（LLM）的微调来加速声称的创建。$\textbf{Claim Advisor}$旨在颠覆声称搜索、生成、优化和模拟的速度与经济性。它具有三项功能：(1) 语义搜索和识别与消费者声音共鸣的现有声称和/或视觉内容；(2) 根据产品描述和消费者画像生成和/或优化声称；以及(3) 通过合成消费者进行模拟的生成和/或手动创建的声称排名。在消费品公司中的应用显示出非常有前景的结果。我们认为这项能力在各类产品和行业中的应用非常广泛。我们分享我们的学习经验，以促进生成式AI在不同行业的研究与应用。 

---
# Adaptive Cybersecurity Architecture for Digital Product Ecosystems Using Agentic AI 

**Title (ZH)**: 基于有能性人工智能的数字产品生态系统自适应网络安全架构 

**Authors**: Oluwakemi T. Olayinka, Sumeet Jeswani, Divine Iloh  

**Link**: [PDF](https://arxiv.org/pdf/2509.20640)  

**Abstract**: Traditional static cybersecurity models often struggle with scalability, real-time detection, and contextual responsiveness in the current digital product ecosystems which include cloud services, application programming interfaces (APIs), mobile platforms, and edge devices. This study introduces autonomous goal driven agents capable of dynamic learning and context-aware decision making as part of an adaptive cybersecurity architecture driven by agentic artificial intelligence (AI). To facilitate autonomous threat mitigation, proactive policy enforcement, and real-time anomaly detection, this framework integrates agentic AI across the key ecosystem layers. Behavioral baselining, decentralized risk scoring, and federated threat intelligence sharing are important features. The capacity of the system to identify zero-day attacks and dynamically modify access policies was demonstrated through native cloud simulations. The evaluation results show increased adaptability, decreased response latency, and improved detection accuracy. The architecture provides an intelligent and scalable blueprint for safeguarding complex digital infrastructure and is compatible with zero-trust models, thereby supporting the adherence to international cybersecurity regulations. 

**Abstract (ZH)**: 传统静态网络安全模型在当前包括云服务、应用编程接口(API)、移动平台和边缘设备的数字产品生态系统中往往面临扩展性、实时检测和上下文响应性方面的挑战。本研究引入了自主目标驱动代理，这些代理具备动态学习和情境感知决策能力，并作为由代理人工智能(AI)驱动的自适应网络安全架构的一部分。为实现自主威胁缓解、积极政策执行和实时异常检测，该框架在关键生态层面上集成了代理AI。行为基线建立、分散风险评分和联邦威胁情报共享是关键特征。通过原生云模拟展示了系统识别零日攻击和动态调整访问策略的能力。评估结果表明，该架构具有更高的适应性、更低的响应延迟和更高的检测准确性。该架构提供了一种智能化和可扩展的复杂数字基础设施保护蓝图，并与零信任模型兼容，从而支持遵循国际网络安全规范。 

---
# SAMULE: Self-Learning Agents Enhanced by Multi-level Reflection 

**Title (ZH)**: SAMULE: 自学习代理增强的多级反思方法 

**Authors**: Yubin Ge, Salvatore Romeo, Jason Cai, Monica Sunkara, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20562)  

**Abstract**: Despite the rapid advancements in LLM agents, they still face the challenge of generating meaningful reflections due to inadequate error analysis and a reliance on rare successful trajectories, especially in complex tasks. In this work, we propose SAMULE, a new framework for self-learning agents powered by a retrospective language model that is trained based on Multi-Level Reflection Synthesis. It first synthesizes high-quality reflections across three complementary levels: Single-Trajectory Learning (micro-level) for detailed error correction; Intra-Task Learning (meso-level) to build error taxonomies across multiple trials of the same task, and Inter-Task Learning (macro-level) to extract transferable insights based on same typed errors from diverse task failures. Then we fine-tune a language model serving as the retrospective model to generate reflections during inference. We further extend our framework to interactive settings through a foresight-based reflection mechanism, enabling agents to proactively reflect and adapt during user interactions by comparing predicted and actual responses. Extensive experiments on three challenging benchmarks - TravelPlanner, NATURAL PLAN, and Tau-bench - demonstrate that our approach significantly outperforms reflection-based baselines. Our results highlight the critical role of well-designed reflection synthesis and failure-centric learning in building self-improving LLM agents. 

**Abstract (ZH)**: 尽管大语言模型代理取得了迅速进展，但由于缺乏充分的错误分析和依赖稀有的成功轨迹，它们仍然面临生成有意义反思的挑战，特别是在复杂任务中。本文提出了一种新的SAMULE框架，该框架基于多级反思合成训练回顾性语言模型，为自我学习代理提供动力。该框架首先在三个互补层级上合成了高质量的反思：单轨迹学习（微观层级）进行详细的错误修正；任务内学习（中观层级）建立同一任务多次试验中的错误分类体系；跨任务学习（宏观层级）从不同任务失败中提取可迁移的见解。然后，我们微调一个语言模型作为回顾性模型，在推断过程中生成反思。我们进一步通过前瞻性的反思机制将该框架扩展到交互式设置中，使代理能够在用户交互过程中通过比较预测和实际响应来主动反思和调整。在三个具有挑战性的基准测试——TravelPlanner、NATURAL PLAN和Tau-bench——上的广泛实验表明，我们的方法显著优于基于反思的基线。我们的结果强调了精心设计的反思合成和以失败为中心的学习在构建自我改进的大语言模型代理中的关键作用。 

---
# A Compound Classification System Based on Fuzzy Relations Applied to the Noise-Tolerant Control of a Bionic Hand via EMG Signal Recognition 

**Title (ZH)**: 基于模糊关系的复合分类系统及其在 Electromyography 信号识别指导下的容噪仿生手控制 

**Authors**: Pawel Trajdos, Marek Kurzynski  

**Link**: [PDF](https://arxiv.org/pdf/2509.20523)  

**Abstract**: Modern anthropomorphic upper limb bioprostheses are typically controlled by electromyographic (EMG) biosignals using a pattern recognition scheme. Unfortunately, there are many factors originating from the human source of objects to be classified and from the human-prosthesis interface that make it difficult to obtain an acceptable classification quality. One of these factors is the high susceptibility of biosignals to contamination, which can considerably reduce the quality of classification of a recognition system.
In the paper, the authors propose a new recognition system intended for EMG based control of the hand prosthesis with detection of contaminated biosignals in order to mitigate the adverse effect of contaminations. The system consists of two ensembles: the set of one-class classifiers (OCC) to assess the degree of contamination of individual channels and the ensemble of K-nearest neighbours (KNN) classifier to recognise the patient's intent. For all recognition systems, an original, coherent fuzzy model was developed, which allows the use of a uniform soft (fuzzy) decision scheme throughout the recognition process. The experimental evaluation was conducted using real biosignals from a public repository. The goal was to provide an experimental comparative analysis of the parameters and procedures of the developed method on which the quality of the recognition system depends. The proposed fuzzy recognition system was also compared with similar systems described in the literature. 

**Abstract (ZH)**: 现代类人上肢生物假肢通常通过模式识别方案利用肌电图（EMG）生物信号进行控制。由于来自人类对象和人类-假肢接口的各种因素，很难获得可接受的分类质量。其中一个因素是生物信号对污染的高度敏感性，这可以显著降低识别系统的分类质量。

在本文中，作者提出了一种新的识别系统，旨在基于EMG控制手部假肢并检测受污染的生物信号，以减轻污染的不良影响。该系统由两个集成部分组成：一类分类器（OCC）集合作为评估各个通道污染程度的方法，以及K邻近邻居（KNN）分类器集合作为识别患者意图的方法。对于所有识别系统，开发了一个原创的一致模糊模型，这使得在整个识别过程中可以使用统一的软（模糊）决策方案。实验评估使用了一个公开数据集中的真实生物信号进行。目标是提供一种实验性的比较分析，比较所开发方法的参数和流程，这些参数和流程决定了识别系统的质量。此外，提出的模糊识别系统还与文献中描述的类似系统进行了比较。 

---
# Adaptive Approach to Enhance Machine Learning Scheduling Algorithms During Runtime Using Reinforcement Learning in Metascheduling Applications 

**Title (ZH)**: 运行时使用强化学习改进元调度应用中机器学习调度算法的自适应方法 

**Authors**: Samer Alshaer, Ala Khalifeh, Roman Obermaisser  

**Link**: [PDF](https://arxiv.org/pdf/2509.20520)  

**Abstract**: Metascheduling in time-triggered architectures has been crucial in adapting to dynamic and unpredictable environments, ensuring the reliability and efficiency of task execution. However, traditional approaches face significant challenges when training Artificial Intelligence (AI) scheduling inferences offline, particularly due to the complexities involved in constructing a comprehensive Multi-Schedule Graph (MSG) that accounts for all possible scenarios. The process of generating an MSG that captures the vast probability space, especially when considering context events like hardware failures, slack variations, or mode changes, is resource-intensive and often infeasible. To address these challenges, we propose an adaptive online learning unit integrated within the metascheduler to enhance performance in real-time. The primary motivation for developing this unit stems from the limitations of offline training, where the MSG created is inherently a subset of the complete space, focusing only on the most probable and critical context events. In the online mode, Reinforcement Learning (RL) plays a pivotal role by continuously exploring and discovering new scheduling solutions, thus expanding the MSG and enhancing system performance over time. This dynamic adaptation allows the system to handle unexpected events and complex scheduling scenarios more effectively. Several RL models were implemented within the online learning unit, each designed to address specific challenges in scheduling. These models not only facilitate the discovery of new solutions but also optimize existing schedulers, particularly when stricter deadlines or new performance criteria are introduced. By continuously refining the AI inferences through real-time training, the system remains flexible and capable of meeting evolving demands, thus ensuring robustness and efficiency in large-scale, safety-critical environments. 

**Abstract (ZH)**: 时间触发架构中基于时间的元调度在动态和不可预测环境中起着关键作用，确保任务执行的可靠性和效率。然而，传统的元调度方法在离线训练人工智能调度推理时面临显著挑战，特别是在构建全面的多调度图（MSG）方面，该图需要考虑所有可能的情景时更为复杂。生成能够捕捉广泛概率空间的MSG，尤其是在考虑硬件故障、余量变化或模式切换等上下文事件时，这个过程资源密集且往往不可行。为了解决这些挑战，我们提出在元调度器中集成一个自适应的在线学习单元，以增强实时性能。开发该单元的主要动机来自于离线训练的局限性，离线训练生成的MSG本质上是完整空间的子集，仅关注最有可能和关键的上下文事件。在线模式下，强化学习（RL）通过持续探索和发现新的调度解决方案，扩展MSG并随着时间提升系统性能。这种动态适应使得系统能够更有效地处理意外事件和复杂的调度情景。在在线学习单元中实施了多种RL模型，每种模型都旨在解决调度中的特定挑战。这些模型不仅促进了新解决方案的发现，还优化了现有的调度器，尤其是在引入更严格的截止时间或新的性能标准时。通过持续通过实时训练调整AI推理，系统保持灵活性，能够满足不断变化的需求，从而在大规模、安全关键的环境中确保稳定性和高效性。 

---
# Reconstruction-Based Adaptive Scheduling Using AI Inferences in Safety-Critical Systems 

**Title (ZH)**: 基于重建的自适应调度利用AI推理在安全性关键系统中的应用 

**Authors**: Samer Alshaer, Ala Khalifeh, Roman Obermaisser  

**Link**: [PDF](https://arxiv.org/pdf/2509.20513)  

**Abstract**: Adaptive scheduling is crucial for ensuring the reliability and safety of time-triggered systems (TTS) in dynamic operational environments. Scheduling frameworks face significant challenges, including message collisions, locked loops from incorrect precedence handling, and the generation of incomplete or invalid schedules, which can compromise system safety and performance. To address these challenges, this paper presents a novel reconstruction framework designed to dynamically validate and assemble schedules. The proposed reconstruction models operate by systematically transforming AI-generated or heuristically derived scheduling priorities into fully executable schedules, ensuring adherence to critical system constraints such as precedence rules and collision-free communication. It incorporates robust safety checks, efficient allocation algorithms, and recovery mechanisms to handle unexpected context events, including hardware failures and mode transitions. Comprehensive experiments were conducted across multiple performance profiles, including makespan minimisation, workload balancing, and energy efficiency, to validate the operational effectiveness of the reconstruction models. Results demonstrate that the proposed framework significantly enhances system adaptability, operational integrity, and runtime performance while maintaining computational efficiency. Overall, this work contributes a practical and scalable solution to the problem of safe schedule generation in safety-critical TTS, enabling reliable and flexible real-time scheduling even under highly dynamic and uncertain operational conditions. 

**Abstract (ZH)**: 自适应调度对于确保时触发系统（TTS）在动态操作环境中的可靠性和安全性至关重要。调度框架面临诸多挑战，包括消息冲突、由于错误的优先级处理而产生的死循环以及生成不完整或无效的调度，这些都可能损害系统的安全性和性能。为应对这些挑战，本文提出了一种新型重构框架，旨在动态验证和组装调度方案。提出的重构模型通过系统地将AI生成或启发式推导出的调度优先级转换为完全可执行的调度方案，确保遵守关键的系统约束条件，如优先级规则和无冲突通信。该框架还集成了 robust 安全检查、高效分配算法和恢复机制，以处理诸如硬件故障和模式转换等意外背景事件。我们进行了全面的实验，涵盖周转时间最小化、工作负载均衡和能效等多种性能指标，以验证重构模型的操作有效性。结果表明，所提出框架显著提高了系统的适应性、操作完整性和运行时性能，同时保持了计算效率。总体而言，本文为安全时触发系统中安全调度生成问题提供了一种实用且可扩展的解决方案，即使在高度动态和不确定的操作条件下也能实现可靠和灵活的实时调度。 

---
# InsightGUIDE: An Opinionated AI Assistant for Guided Critical Reading of Scientific Literature 

**Title (ZH)**: InsightGUIDE: 一套偏见导向的AI辅助工具，用于指导性的科学文献批判性阅读 

**Authors**: Paris Koloveas, Serafeim Chatzopoulos, Thanasis Vergoulis, Christos Tryfonopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2509.20493)  

**Abstract**: The proliferation of scientific literature presents an increasingly significant challenge for researchers. While Large Language Models (LLMs) offer promise, existing tools often provide verbose summaries that risk replacing, rather than assisting, the reading of the source material. This paper introduces InsightGUIDE, a novel AI-powered tool designed to function as a reading assistant, not a replacement. Our system provides concise, structured insights that act as a "map" to a paper's key elements by embedding an expert's reading methodology directly into its core AI logic. We present the system's architecture, its prompt-driven methodology, and a qualitative case study comparing its output to a general-purpose LLM. The results demonstrate that InsightGUIDE produces more structured and actionable guidance, serving as a more effective tool for the modern researcher. 

**Abstract (ZH)**: 科学文献的 proliferate 为研究人员提出了日益显著的挑战。尽管大型语言模型 (LLMs) 具有潜力，现有的工具往往提供冗长的摘要，这可能会替代而非辅助阅读原始材料。本文介绍了一种名为 InsightGUIDE 的新型 AI 助力工具，旨在作为阅读助手而非替代品。我们的系统提供简洁、结构化的洞见，通过将专家的阅读方法直接嵌入其核心 AI 逻辑中，起到“地图”的作用，指向论文的关键要素。我们介绍了系统的架构、基于提示的方法以及与通用语言模型输出的定性案例研究。结果表明，InsightGUIDE 生成了更加结构化和可操作的指导，成为现代研究人员更有效的工具。 

---
# Philosophy-informed Machine Learning 

**Title (ZH)**: 哲学启发的机器学习 

**Authors**: MZ Naser  

**Link**: [PDF](https://arxiv.org/pdf/2509.20370)  

**Abstract**: Philosophy-informed machine learning (PhIML) directly infuses core ideas from analytic philosophy into ML model architectures, objectives, and evaluation protocols. Therefore, PhIML promises new capabilities through models that respect philosophical concepts and values by design. From this lens, this paper reviews conceptual foundations to demonstrate philosophical gains and alignment. In addition, we present case studies on how ML users/designers can adopt PhIML as an agnostic post-hoc tool or intrinsically build it into ML model architectures. Finally, this paper sheds light on open technical barriers alongside philosophical, practical, and governance challenges and outlines a research roadmap toward safe, philosophy-aware, and ethically responsible PhIML. 

**Abstract (ZH)**: 基于哲学的机器学习（PhIML）直接将分析哲学的核心思想融入到ML模型架构、目标及评估协议中，因此，PhIML通过设计上的模型尊重哲学概念和价值，带来了新的能力。从这一视角出发，本文回顾概念基础，展示哲学收益和一致性。此外，我们展示了如何将PhIML作为中立的后验工具或内在构建到ML模型架构中。最后，本文揭示了开放的技术障碍，以及哲学、实践和治理方面的挑战，并概述了一条通往安全、意识哲学和伦理责任的PhIML的研究路线图。 

---
# LATTS: Locally Adaptive Test-Time Scaling 

**Title (ZH)**: LATTS：局部自适应测试时缩放 

**Authors**: Theo Uscidda, Matthew Trager, Michael Kleinman, Aditya Chattopadhyay, Wei Xia, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2509.20368)  

**Abstract**: One common strategy for improving the performance of Large Language Models (LLMs) on downstream tasks involves using a \emph{verifier model} to either select the best answer from a pool of candidates or to steer the auto-regressive generation process towards better outputs. This class of methods typically results in improved accuracy at the cost of increased computation at test-time, a paradigm known as \emph{test-time scaling}. However, most existing approaches increase computation uniformly across all samples and generation steps, without considering the complexity of individual instances, leading to inefficient resource use. We address this limitation by proposing an approach, called \emph{Locally Adaptive Test-Time Scaling (LATTS)}, that allocates variable compute across generation steps. Specifically, at each generation step, LATTS employs a verifier-based acceptance criterion to decide whether to resample, backtrack, restart, or stop the generation process. This criterion effectively adjusts the per-step computational effort based on a precise notion of \emph{local difficulty} derived from the verifier model. Empirical results show that LATTS achieves significantly superior accuracy--compute tradeoffs compared to standard verifier-based methods. 

**Abstract (ZH)**: 局部自适应测试时缩放（LATTS）：基于校验模型的可变计算分配方法 

---
# An Approach to Checking Correctness for Agentic Systems 

**Title (ZH)**: 检查代理系统正确性的方法 

**Authors**: Thomas J Sheffler  

**Link**: [PDF](https://arxiv.org/pdf/2509.20364)  

**Abstract**: This paper presents a temporal expression language for monitoring AI agent behavior, enabling systematic error-detection of LLM-based agentic systems that exhibit variable outputs due to stochastic generation processes. Drawing from temporal logic techniques used in hardware verification, this approach monitors execution traces of agent tool calls and state transitions to detect deviations from expected behavioral patterns. Current error-detection approaches rely primarily on text matching of inputs and outputs, which proves fragile due to the natural language variability inherent in LLM responses. The proposed method instead focuses on the sequence of agent actions -- such as tool invocations and inter-agent communications -- allowing verification of system behavior independent of specific textual outputs. The temporal expression language provides assertions that capture correct behavioral patterns across multiple execution scenarios. These assertions serve dual purposes: validating prompt engineering and guardrail effectiveness during development, and providing regression testing when agents are updated with new LLMs or modified logic. The approach is demonstrated using a three-agent system, where agents coordinate to solve multi-step reasoning tasks. When powered by large, capable models, all temporal assertions were satisfied across many test runs. However, when smaller models were substituted in two of the three agents, executions violated behavioral assertions, primarily due to improper tool sequencing and failed coordination handoffs. The temporal expressions successfully flagged these anomalies, demonstrating the method's effectiveness for detecting behavioral regressions in production agentic systems. This approach provides a foundation for systematic monitoring of AI agent reliability as these systems become increasingly deployed in critical applications. 

**Abstract (ZH)**: 一种用于监控AI代理行为的时间表达式语言：基于随机生成过程导致输出变化的LLM基础代理系统系统的系统错误检测方法 

---
# RLBFF: Binary Flexible Feedback to bridge between Human Feedback & Verifiable Rewards 

**Title (ZH)**: RLBFF：二值灵活反馈以缓解人类反馈与可验证奖励之间的差距 

**Authors**: Zhilin Wang, Jiaqi Zeng, Olivier Delalleau, Ellie Evans, Daniel Egert, Hoo-Chang Shin, Felipe Soares, Yi Dong, Oleksii Kuchaiev  

**Link**: [PDF](https://arxiv.org/pdf/2509.21319)  

**Abstract**: Reinforcement Learning with Human Feedback (RLHF) and Reinforcement Learning with Verifiable Rewards (RLVR) are the main RL paradigms used in LLM post-training, each offering distinct advantages. However, RLHF struggles with interpretability and reward hacking because it relies on human judgments that usually lack explicit criteria, whereas RLVR is limited in scope by its focus on correctness-based verifiers. We propose Reinforcement Learning with Binary Flexible Feedback (RLBFF), which combines the versatility of human-driven preferences with the precision of rule-based verification, enabling reward models to capture nuanced aspects of response quality beyond mere correctness. RLBFF extracts principles that can be answered in a binary fashion (e.g. accuracy of information: yes, or code readability: no) from natural language feedback. Such principles can then be used to ground Reward Model training as an entailment task (response satisfies or does not satisfy an arbitrary principle). We show that Reward Models trained in this manner can outperform Bradley-Terry models when matched for data and achieve top performance on RM-Bench (86.2%) and JudgeBench (81.4%, #1 on leaderboard as of September 24, 2025). Additionally, users can specify principles of interest at inference time to customize the focus of our reward models, in contrast to Bradley-Terry models. Finally, we present a fully open source recipe (including data) to align Qwen3-32B using RLBFF and our Reward Model, to match or exceed the performance of o3-mini and DeepSeek R1 on general alignment benchmarks of MT-Bench, WildBench, and Arena Hard v2 (at <5% of the inference cost). 

**Abstract (ZH)**: 基于人类反馈的强化学习（RLHF）和可验证奖励的强化学习（RLVR）是LLM后训练中使用的主要RL范式，各自具有独特的优势。然而，RLHF在可解释性和奖励作弊方面存在挑战，因为它依赖于通常缺乏明确标准的人类判断；而RLVR则受限于其基于正确性验证的范围。我们提出了一种基于二元灵活反馈的强化学习（RLBFF），将人类驱动的偏好多样性与基于规则的验证精确性结合，使奖励模型能够捕捉到响应质量的细微方面，而不仅仅是正确性。RLBFF从自然语言反馈中提取可以用二元方式回答的原则（例如：信息准确性：是，或代码可读性：否），这些原则可以作为归结任务（响应是否满足某个任意原则）来训练奖励模型。我们展示，以这种方式训练的奖励模型在数据调整后的性能可以超越布拉德利-特里模型，并在RM-Bench（86.2%）和JudgeBench（81.4%，截至2025年9月24日排行榜第一）上达到最佳性能。此外，在推理时用户可以指定感兴趣的原则来定制我们的奖励模型的焦点，不同于布拉德利-特里模型。最后，我们提供了一个完整的开源食谱（包括数据），使用RLBFF和我们的奖励模型对Qwen3-32B进行对齐，匹配或超越o3-mini和DeepSeek R1在MT-Bench、WildBench和Arena Hard v2上的通用对齐基准性能（成本不到5%）。 

---
# SD3.5-Flash: Distribution-Guided Distillation of Generative Flows 

**Title (ZH)**: SD3.5-Flash：分布导向的生成流蒸馏 

**Authors**: Hmrishav Bandyopadhyay, Rahim Entezari, Jim Scott, Reshinth Adithyan, Yi-Zhe Song, Varun Jampani  

**Link**: [PDF](https://arxiv.org/pdf/2509.21318)  

**Abstract**: We present SD3.5-Flash, an efficient few-step distillation framework that brings high-quality image generation to accessible consumer devices. Our approach distills computationally prohibitive rectified flow models through a reformulated distribution matching objective tailored specifically for few-step generation. We introduce two key innovations: "timestep sharing" to reduce gradient noise and "split-timestep fine-tuning" to improve prompt alignment. Combined with comprehensive pipeline optimizations like text encoder restructuring and specialized quantization, our system enables both rapid generation and memory-efficient deployment across different hardware configurations. This democratizes access across the full spectrum of devices, from mobile phones to desktop computers. Through extensive evaluation including large-scale user studies, we demonstrate that SD3.5-Flash consistently outperforms existing few-step methods, making advanced generative AI truly accessible for practical deployment. 

**Abstract (ZH)**: SD3.5-Flash: 一种高效的few-step蒸馏框架，将高质量图像生成带到可访问的消费级设备 

---
# No Prior, No Leakage: Revisiting Reconstruction Attacks in Trained Neural Networks 

**Title (ZH)**: 无需先验，无泄露：重访训练神经网络中的重建攻击 

**Authors**: Yehonatan Refael, Guy Smorodinsky, Ofir Lindenbaum, Itay Safran  

**Link**: [PDF](https://arxiv.org/pdf/2509.21296)  

**Abstract**: The memorization of training data by neural networks raises pressing concerns for privacy and security. Recent work has shown that, under certain conditions, portions of the training set can be reconstructed directly from model parameters. Some of these methods exploit implicit bias toward margin maximization, suggesting that properties often regarded as beneficial for generalization may actually compromise privacy. Yet despite striking empirical demonstrations, the reliability of these attacks remains poorly understood and lacks a solid theoretical foundation. In this work, we take a complementary perspective: rather than designing stronger attacks, we analyze the inherent weaknesses and limitations of existing reconstruction methods and identify conditions under which they fail. We rigorously prove that, without incorporating prior knowledge about the data, there exist infinitely many alternative solutions that may lie arbitrarily far from the true training set, rendering reconstruction fundamentally unreliable. Empirically, we further demonstrate that exact duplication of training examples occurs only by chance. Our results refine the theoretical understanding of when training set leakage is possible and offer new insights into mitigating reconstruction attacks. Remarkably, we demonstrate that networks trained more extensively, and therefore satisfying implicit bias conditions more strongly -- are, in fact, less susceptible to reconstruction attacks, reconciling privacy with the need for strong generalization in this setting. 

**Abstract (ZH)**: 神经网络对训练数据的记忆引发了隐私和安全方面的紧迫关切。近期研究显示，在某些条件下，训练集的部分内容可以从模型参数中直接重构。这些方法中的一些利用了对边界最大化隐式偏好的依赖，表明通常被认为有助于泛化的属性实际上可能损害隐私。尽管有令人印象深刻的经验演示，但这些攻击的可靠性仍不甚明确且缺乏坚实的理论基础。在本工作中，我们采取了互补的视角：而不是设计更强的攻击，我们分析现有重构方法的固有弱点和局限性，并确定它们失效的条件。我们严格证明，在不包含关于数据的先验知识的情况下，存在无限多个可能任意远离真实训练集的替代解，使得重构从根本上不可靠。进一步的经验研究表明，只有偶然才会完全复制训练示例。我们的结果细化了训练集泄露何时可能发生时的理论理解，并提供了缓解重构攻击的新见解。尤为引人注目的是，我们证明了训练更为充分的网络——因此更强烈地满足隐式偏好的条件——实际上对重构攻击的抵抗力较低，这一发现在此情境中实现了隐私与强泛化的平衡。 

---
# DisCoCLIP: A Distributional Compositional Tensor Network Encoder for Vision-Language Understanding 

**Title (ZH)**: 分布组合张量网络编码器：用于视觉-语言理解的DisCoCLIP 

**Authors**: Kin Ian Lo, Hala Hawashin, Mina Abbaszadeh, Tilen Limback-Stokin, Hadi Wazni, Mehrnoosh Sadrzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2509.21287)  

**Abstract**: Recent vision-language models excel at large-scale image-text alignment but often neglect the compositional structure of language, leading to failures on tasks that hinge on word order and predicate-argument structure. We introduce DisCoCLIP, a multimodal encoder that combines a frozen CLIP vision transformer with a novel tensor network text encoder that explicitly encodes syntactic structure. Sentences are parsed with a Combinatory Categorial Grammar parser to yield distributional word tensors whose contractions mirror the sentence's grammatical derivation. To keep the model efficient, high-order tensors are factorized with tensor decompositions, reducing parameter count from tens of millions to under one million. Trained end-to-end with a self-supervised contrastive loss, DisCoCLIP markedly improves sensitivity to verb semantics and word order: it raises CLIP's SVO-Probes verb accuracy from 77.6% to 82.4%, boosts ARO attribution and relation scores by over 9% and 4%, and achieves 93.7% on a newly introduced SVO-Swap benchmark. These results demonstrate that embedding explicit linguistic structure via tensor networks yields interpretable, parameter-efficient representations that substantially improve compositional reasoning in vision-language tasks. 

**Abstract (ZH)**: Recent Vision-Language Models Often Neglect Linguistic Compositionality: Introducing DisCoCLIP 

---
# It's Not You, It's Clipping: A Soft Trust-Region via Probability Smoothing for LLM RL 

**Title (ZH)**: 不是你，是裁剪：基于概率平滑的软信任区域方法用于大型语言模型的RL 

**Authors**: Madeleine Dwyer, Adam Sobey, Adriane Chapman  

**Link**: [PDF](https://arxiv.org/pdf/2509.21282)  

**Abstract**: Training large language models (LLMs) with reinforcement learning (RL) methods such as PPO and GRPO commonly relies on ratio clipping to stabilise updates. While effective at preventing instability, clipping discards information and introduces gradient discontinuities. We propose Probability Smoothing Policy Optimisation (PSPO), which smooths the current policy's probabilities toward the old (behaviour) policy before computing the importance ratio, analogous to label smoothing. Unlike clipping, PSPO preserves gradient signal, while interpolation toward the old policy creates a soft trust region that discourages large, destabilising updates, with formal guarantees.
We instantiate PSPO within GRPO (GR-PSPO) and fine-tune Qwen2.5-0.5B and Qwen2.5-1.5B on GSM8K, evaluating on GSM8K test and the cross-dataset generalisation on SVAMP, ASDiv, and MATH-500. Relative to unclipped GRPO (single iteration; no data reuse, ratio always = 1), GR-PSPO achieves similar performance but improves the reasoning leading to clearer and more concise responses which are more logical. Compared to clipped GRPO, GR-PSPO substantially improves performance both the 0.5B and 1.5B models, with a boost of over 20% on GSM8K (39.7% vs. 17.6% for 0.5B, 59.4% vs. 37.8% for 1.5B). 

**Abstract (ZH)**: 使用PPO和GRPO等强化学习方法训练大型语言模型（LLMs）通常依赖于比率剪切以稳定更新。我们提出了概率平滑策略优化（PSPO），该方法在计算重要性比率之前，将当前策略的概率平滑至旧的行为策略，类似于标签平滑。与比率剪切不同，PSPO保留了梯度信号，而向旧策略的插值创建了一个软信任区域，这会避免大型且不稳定的更新，并且具有形式上的保证。 

---
# Does FLUX Already Know How to Perform Physically Plausible Image Composition? 

**Title (ZH)**: FLUX 是否already知道如何执行物理上可验证的图像合成？ 

**Authors**: Shilin Lu, Zhuming Lian, Zihan Zhou, Shaocong Zhang, Chen Zhao, Adams Wai-Kin Kong  

**Link**: [PDF](https://arxiv.org/pdf/2509.21278)  

**Abstract**: Image composition aims to seamlessly insert a user-specified object into a new scene, but existing models struggle with complex lighting (e.g., accurate shadows, water reflections) and diverse, high-resolution inputs. Modern text-to-image diffusion models (e.g., SD3.5, FLUX) already encode essential physical and resolution priors, yet lack a framework to unleash them without resorting to latent inversion, which often locks object poses into contextually inappropriate orientations, or brittle attention surgery. We propose SHINE, a training-free framework for Seamless, High-fidelity Insertion with Neutralized Errors. SHINE introduces manifold-steered anchor loss, leveraging pretrained customization adapters (e.g., IP-Adapter) to guide latents for faithful subject representation while preserving background integrity. Degradation-suppression guidance and adaptive background blending are proposed to further eliminate low-quality outputs and visible seams. To address the lack of rigorous benchmarks, we introduce ComplexCompo, featuring diverse resolutions and challenging conditions such as low lighting, strong illumination, intricate shadows, and reflective surfaces. Experiments on ComplexCompo and DreamEditBench show state-of-the-art performance on standard metrics (e.g., DINOv2) and human-aligned scores (e.g., DreamSim, ImageReward, VisionReward). Code and benchmark will be publicly available upon publication. 

**Abstract (ZH)**: Seamless, 高保真插入与错误抑制的训练-free 框架：SHINE 

---
# Data-Centric Elastic Pipeline Parallelism for Efficient Long-Context LLM Training 

**Title (ZH)**: 以数据为中心的弹性管道并行训练高效处理长上下文LLM模型 

**Authors**: Shiju Wang, Yujie Wang, Ao Sun, Fangcheng Fu, Zijian Zhu, Bin Cui, Xu Han, Kaisheng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.21275)  

**Abstract**: Long context training is crucial for LLM's context extension. Existing schemes, such as sequence parallelism, incur substantial communication overhead. Pipeline parallelism (PP) reduces this cost, but its effectiveness hinges on partitioning granularity. Batch-level PP dividing input samples exhibits high memory consumption in long-context scenario, whereas token-level PP splitting sequences into slices alleviates memory overhead but may incur hardware under-utilization. This trade-off motivates adaptively selecting PP granularity to match resource and workload characteristics. Moreover, sequence length distribution of the real-world dataset exhibits skewness, posing a challenge on PP's workload balance and efficient scheduling. Current static PP scheduling methods overlook the variance of sequence length, leading to suboptimal performance. In this paper, we propose Elastic Pipeline Parallelism (EPP) that orchestrates token-level PP and batch-level PP to adapt to resource and workload heterogeneity. We build InfiniPipe, a distributed training system that unleashes the potential of EPP via (1) a resource-aware and workload-balanced sequence processor that splits long sequences and packs short ones; and (2) a co-optimization methodology that jointly optimizes pipeline schedule and gradient checkpointing via a mechanism named stage-aware chunk-level adaptive checkpointing. Comprehensive experiments demonstrate that InfiniPipe achieves a 1.69x speedup over state-of-the-art systems. 

**Abstract (ZH)**: 弹性管道并行性（EPP）：适应资源和工作负载异构性的动态管道并行性 

---
# MedVSR: Medical Video Super-Resolution with Cross State-Space Propagation 

**Title (ZH)**: MedVSR：基于跨态空间传播的医学视频超分辨率 

**Authors**: Xinyu Liu, Guolei Sun, Cheng Wang, Yixuan Yuan, Ender Konukoglu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21265)  

**Abstract**: High-resolution (HR) medical videos are vital for accurate diagnosis, yet are hard to acquire due to hardware limitations and physiological constraints. Clinically, the collected low-resolution (LR) medical videos present unique challenges for video super-resolution (VSR) models, including camera shake, noise, and abrupt frame transitions, which result in significant optical flow errors and alignment difficulties. Additionally, tissues and organs exhibit continuous and nuanced structures, but current VSR models are prone to introducing artifacts and distorted features that can mislead doctors. To this end, we propose MedVSR, a tailored framework for medical VSR. It first employs Cross State-Space Propagation (CSSP) to address the imprecise alignment by projecting distant frames as control matrices within state-space models, enabling the selective propagation of consistent and informative features to neighboring frames for effective alignment. Moreover, we design an Inner State-Space Reconstruction (ISSR) module that enhances tissue structures and reduces artifacts with joint long-range spatial feature learning and large-kernel short-range information aggregation. Experiments across four datasets in diverse medical scenarios, including endoscopy and cataract surgeries, show that MedVSR significantly outperforms existing VSR models in reconstruction performance and efficiency. Code released at this https URL. 

**Abstract (ZH)**: 高分辨率医学视频对于准确诊断至关重要，但由于硬件限制和生理限制，获取这些视频比较困难。临床中，收集到的低分辨率医学视频给视频超分辨率（VSR）模型带来了独特挑战，包括相机抖动、噪声和帧间突变，这些都导致了显著的光学流动错误和对齐难题。此外，组织和器官表现出连续和细腻的结构，但当前的VSR模型容易引入伪影和失真的特征，这可能会误导医生。为此，我们提出MedVSR，一个针对医学VSR的定制框架。该框架首先采用跨状态空间传播（CSSP）来解决不精确的对齐问题，通过将远程帧作为状态空间模型内的控制矩阵进行投影，使得一致且富有信息的特征能够在邻近帧之间进行选择性传播，从而实现有效的对齐。此外，我们设计了内部状态空间重构（ISSR）模块，通过结合长距离空间特征学习和大型内核短距离信息聚合来增强组织结构并减少伪影。在四个不同医学场景的四个数据集中进行的实验，包括内窥镜和白内障手术，表明MedVSR在重建性能和效率上显著优于现有VSR模型。已在该链接发布代码：[这个 https URL]。 

---
# A Causality-Aware Spatiotemporal Model for Multi-Region and Multi-Pollutant Air Quality Forecasting 

**Title (ZH)**: 考虑因果关系的空间时间模型及其在多区域多污染物空气质量预报中的应用 

**Authors**: Junxin Lu, Shiliang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.21260)  

**Abstract**: Air pollution, a pressing global problem, threatens public health, environmental sustainability, and climate stability. Achieving accurate and scalable forecasting across spatially distributed monitoring stations is challenging due to intricate multi-pollutant interactions, evolving meteorological conditions, and region specific spatial heterogeneity. To address this challenge, we propose AirPCM, a novel deep spatiotemporal forecasting model that integrates multi-region, multi-pollutant dynamics with explicit meteorology-pollutant causality modeling. Unlike existing methods limited to single pollutants or localized regions, AirPCM employs a unified architecture to jointly capture cross-station spatial correlations, temporal auto-correlations, and meteorology-pollutant dynamic causality. This empowers fine-grained, interpretable multi-pollutant forecasting across varying geographic and temporal scales, including sudden pollution episodes. Extensive evaluations on multi-scale real-world datasets demonstrate that AirPCM consistently surpasses state-of-the-art baselines in both predictive accuracy and generalization capability. Moreover, the long-term forecasting capability of AirPCM provides actionable insights into future air quality trends and potential high-risk windows, offering timely support for evidence-based environmental governance and carbon mitigation planning. 

**Abstract (ZH)**: 空气污染：一个迫切的全球性问题，威胁着公共健康、环境可持续性和气候稳定性。为了应对这一挑战，我们提出了AirPCM，一种新颖的深度空时预测模型，该模型整合了多区域、多污染物动态，并明确建模了气象条件与污染之间的因果关系。AirPCM通过统一架构共同捕捉跨站点的空间相关性、时间自相关性以及气象条件与污染物的动态因果关系，实现了从不同地理和时间尺度对突发污染事件的精细化、可解释的多污染物预测。在多尺度真实世界数据集上的广泛评估表明，AirPCM在预测准确性和泛化能力上均显著优于现有最先进的基线模型。此外，AirPCM的长期预测能力为未来空气质量趋势和潜在高风险窗口提供了可操作见解，为基于证据的环境治理和碳减排规划提供了及时支持。 

---
# Semantic Edge-Cloud Communication for Real-Time Urban Traffic Surveillance with ViT and LLMs over Mobile Networks 

**Title (ZH)**: 基于ViT和LLMs的移动网络实时城市交通监控的语义边缘-云通信 

**Authors**: Murat Arda Onsu, Poonam Lohan, Burak Kantarci, Aisha Syed, Matthew Andrews, Sean Kennedy  

**Link**: [PDF](https://arxiv.org/pdf/2509.21259)  

**Abstract**: Real-time urban traffic surveillance is vital for Intelligent Transportation Systems (ITS) to ensure road safety, optimize traffic flow, track vehicle trajectories, and prevent collisions in smart cities. Deploying edge cameras across urban environments is a standard practice for monitoring road conditions. However, integrating these with intelligent models requires a robust understanding of dynamic traffic scenarios and a responsive interface for user interaction. Although multimodal Large Language Models (LLMs) can interpret traffic images and generate informative responses, their deployment on edge devices is infeasible due to high computational demands. Therefore, LLM inference must occur on the cloud, necessitating visual data transmission from edge to cloud, a process hindered by limited bandwidth, leading to potential delays that compromise real-time performance. To address this challenge, we propose a semantic communication framework that significantly reduces transmission overhead. Our method involves detecting Regions of Interest (RoIs) using YOLOv11, cropping relevant image segments, and converting them into compact embedding vectors using a Vision Transformer (ViT). These embeddings are then transmitted to the cloud, where an image decoder reconstructs the cropped images. The reconstructed images are processed by a multimodal LLM to generate traffic condition descriptions. This approach achieves a 99.9% reduction in data transmission size while maintaining an LLM response accuracy of 89% for reconstructed cropped images, compared to 93% accuracy with original cropped images. Our results demonstrate the efficiency and practicality of ViT and LLM-assisted edge-cloud semantic communication for real-time traffic surveillance. 

**Abstract (ZH)**: 实时城市交通监控对于智能 transportation 系统（ITS）确保道路安全、优化交通流量、追踪车辆轨迹以及预防碰撞至关重要。在城市环境中部署边缘摄像头以监控道路状况是一种标准做法。然而，将这些摄像头与智能模型集成需要对动态交通场景有 robust 的理解以及对用户交互具有响应式的界面。尽管多模态大型语言模型（LLMs）可以解释交通图像并生成有意义的响应，但由于高计算需求，在边缘设备上的部署是不可行的。因此，LLM 的推断必须在云端进行，这需要从边缘到云端传输视觉数据，但受限于有限的带宽，这一过程可能导致潜在的延迟，从而影响实时性能。为解决这一挑战，我们提出了一种语义通信框架，显著减少了数据传输开销。该方法涉及使用 YOLOv11 检测兴趣区域（RoIs）、裁剪相关图像段，并使用 Vision Transformer（ViT）将这些图像段转换为紧凑的嵌入向量。这些嵌入向量随后传输到云端，在云端使用图像解码器重建裁剪图像。重建后的图像由多模态 LLM 处理以生成交通状况描述。该方法在数据传输大小上实现了 99.9% 的减少，同时在重建裁剪图像时，LLM 的响应准确率为 89%，与原始裁剪图像相比，准确率为 93%。我们的结果表明，ViT 和 LLM 辅助的边缘-云端语义通信适用于实时交通监控。 

---
# Instruction-tuned Self-Questioning Framework for Multimodal Reasoning 

**Title (ZH)**: 面向多模态推理的指令调优自提问框架 

**Authors**: You-Won Jang, Yu-Jung Heo, Jaeseok Kim, Minsu Lee, Du-Seong Chang, Byoung-Tak Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21251)  

**Abstract**: The field of vision-language understanding has been actively researched in recent years, thanks to the development of Large Language Models~(LLMs). However, it still needs help with problems requiring multi-step reasoning, even for very simple questions. Recent studies adopt LLMs to tackle this problem by iteratively generating sub-questions and answers. However, there are disadvantages such as 1) the fine-grained visual contents of images are not available using LLMs that cannot read visual information, 2) internal mechanisms are inaccessible and difficult to reproduce by using black-box LLMs. To solve these problems, we propose the SQ (Self-Questioning)-InstructBLIP, which improves inference performance by generating image-aware informative sub-questions and sub-answers iteratively. The SQ-InstructBLIP, which consists of a Questioner, Answerer, and Reasoner that share the same architecture. Questioner and Answerer generate sub-questions and sub-answers to help infer the main-question, and Reasoner performs reasoning on the main-question considering the generated sub-question information. Our experiments show that the proposed method SQ-InstructBLIP, which uses the generated sub-questions as additional information when solving the VQA task, performs more accurate reasoning than the previous works. 

**Abstract (ZH)**: 视觉-语言理解领域近年来由于大型语言模型（LLMs）的发展而得到了积极研究，但仍然需要解决多步推理问题，即使是对于非常简单的问题也是如此。近期的研究通过迭代生成子问题和答案来利用LLMs解决这一问题，但存在一些缺点，例如1）不能获取图像的细粒度视觉内容的LLMs无法阅读视觉信息，2）使用黑箱LLMs使内部机制难以访问和复制。为了解决这些问题，我们提出了SQ（自我提问）-InstructBLIP，通过迭代生成视觉感知的子问题和子答案来提高推理性能。SQ-InstructBLIP由一个共享相同架构的问题生成器、答案生成器和推理器组成。问题生成器和答案生成器生成辅助推理主问题的子问题和子答案，而推理器在考虑生成的子问题信息的同时对主问题进行推理。我们的实验表明，在解决VQA任务时，使用生成的子问题作为额外信息的SQ-InstructBLIP比以往方法进行的推理更为准确。 

---
# Decipher-MR: A Vision-Language Foundation Model for 3D MRI Representations 

**Title (ZH)**: Decipher-MR：一种用于3D MRI表示的视觉-语言基础模型 

**Authors**: Zhijian Yang, Noel DSouza, Istvan Megyeri, Xiaojian Xu, Amin Honarmandi Shandiz, Farzin Haddadpour, Krisztian Koos, Laszlo Rusko, Emanuele Valeriano, Bharadwaj Swaninathan, Lei Wu, Parminder Bhatia, Taha Kass-Hout, Erhan Bas  

**Link**: [PDF](https://arxiv.org/pdf/2509.21249)  

**Abstract**: Magnetic Resonance Imaging (MRI) is a critical medical imaging modality in clinical diagnosis and research, yet its complexity and heterogeneity pose challenges for automated analysis, particularly in scalable and generalizable machine learning applications. While foundation models have revolutionized natural language and vision tasks, their application to MRI remains limited due to data scarcity and narrow anatomical focus. In this work, we present Decipher-MR, a 3D MRI-specific vision-language foundation model trained on a large-scale dataset comprising 200,000 MRI series from over 22,000 studies spanning diverse anatomical regions, sequences, and pathologies. Decipher-MR integrates self-supervised vision learning with report-guided text supervision to build robust, generalizable representations, enabling effective adaptation across broad applications. To enable robust and diverse clinical tasks with minimal computational overhead, Decipher-MR supports a modular design that enables tuning of lightweight, task-specific decoders attached to a frozen pretrained encoder. Following this setting, we evaluate Decipher-MR across diverse benchmarks including disease classification, demographic prediction, anatomical localization, and cross-modal retrieval, demonstrating consistent performance gains over existing foundation models and task-specific approaches. Our results establish Decipher-MR as a scalable and versatile foundation for MRI-based AI, facilitating efficient development across clinical and research domains. 

**Abstract (ZH)**: 磁共振成像(MRI)是临床诊断和研究中一种至关重要的医学影像模态，但由于其复杂性和异质性，自动分析面临挑战，尤其是在可扩展和泛化的机器学习应用中。尽管基础模型在自然语言和视觉任务中取得了革命性的进展，但由于数据稀缺和解剖学关注的局限性，其在MRI中的应用仍然有限。在本文中，我们介绍了一种名为Decipher-MR的3D MRI专用视觉-语言基础模型，该模型基于包含超过22,000个研究中的200,000个MRI系列的大规模数据集训练，涵盖了不同的解剖区域、序列和病理。Decipher-MR将自我监督的视觉学习与报告导向的文字监督相结合，构建了稳健和可泛化的表示，能够有效适应广泛的临床和研究应用。通过模块化设计，Decipher-MR支持轻量级、任务特定解码器与冻结的预训练编码器连接，从而实现最小的计算开销和鲁棒的多样性临床任务。在这一设置下，我们在多种基准任务中评估了Decipher-MR，包括疾病分类、人口统计预测、解剖定位和跨模态检索，展示了相对于现有基础模型和任务特定方法的持续性能提升。我们的结果确立了Decipher-MR作为基于MRI的AI的可扩展和多功能基础，促进了临床和研究领域的高效开发。 

---
# Learning to Look: Cognitive Attention Alignment with Vision-Language Models 

**Title (ZH)**: 学习凝视：认知注意力与视觉语言模型的对齐 

**Authors**: Ryan L. Yang, Dipkamal Bhusal, Nidhi Rastogi  

**Link**: [PDF](https://arxiv.org/pdf/2509.21247)  

**Abstract**: Convolutional Neural Networks (CNNs) frequently "cheat" by exploiting superficial correlations, raising concerns about whether they make predictions for the right reasons. Inspired by cognitive science, which highlights the role of attention in robust human perception, recent methods have sought to guide model attention using concept-based supervision and explanation regularization. However, these techniques depend on labor-intensive, expert-provided annotations, limiting their scalability. We propose a scalable framework that leverages vision-language models to automatically generate semantic attention maps using natural language prompts. By introducing an auxiliary loss that aligns CNN attention with these language-guided maps, our approach promotes more reliable and cognitively plausible decision-making without manual annotation. Experiments on challenging datasets, ColoredMNIST and DecoyMNIST, show that our method achieves state-of-the-art performance on ColorMNIST and remains competitive with annotation-heavy baselines on DecoyMNIST, demonstrating improved generalization, reduced shortcut reliance, and model attention that better reflects human intuition. 

**Abstract (ZH)**: 卷积神经网络（CNNs）经常通过利用表面相关性“作弊”，这引发了对其预测是否基于正确原因的质疑。受认知科学的启发，该科学强调注意在 robust 人类知觉中的作用，近年来的方法试图使用基于概念的监督和解释正则化来引导模型注意。然而，这些技术依赖于劳动密集型的、由专家提供的注释，限制了其可扩展性。我们提出了一种可扩展的框架，利用vision-language模型自动生成基于自然语言提示的语义注意图。通过引入一个辅助损失来使CNN注意与这些语言引导的图对齐，我们的方法能在无需手动注释的情况下促进更可靠且合乎认知的决策。在 ColoredMNIST 和 DecoyMNIST 等具有挑战性的数据集上的实验表明，我们的方法在 ColoredMNIST 上达到了最先进的性能，同时在 DecoyMNIST 上与注释密集型对照组保持竞争力，展示了改进的一般化能力、减少捷径依赖性和更好地反映人类直觉的模型注意。 

---
# Hunyuan3D-Omni: A Unified Framework for Controllable Generation of 3D Assets 

**Title (ZH)**: Hunyuan3D-Omni：可控生成3D资产的统一框架 

**Authors**: Team Hunyuan3D, Bowen Zhang, Chunchao Guo, Haolin Liu, Hongyu Yan, Huiwen Shi, Jingwei Huang, Junlin Yu, Kunhong Li, Linus, Penghao Wang, Qingxiang Lin, Sicong Liu, Xianghui Yang, Yixuan Tang, Yunfei Zhao, Zeqiang Lai, Zhihao Liang, Zibo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.21245)  

**Abstract**: Recent advances in 3D-native generative models have accelerated asset creation for games, film, and design. However, most methods still rely primarily on image or text conditioning and lack fine-grained, cross-modal controls, which limits controllability and practical adoption. To address this gap, we present Hunyuan3D-Omni, a unified framework for fine-grained, controllable 3D asset generation built on Hunyuan3D 2.1. In addition to images, Hunyuan3D-Omni accepts point clouds, voxels, bounding boxes, and skeletal pose priors as conditioning signals, enabling precise control over geometry, topology, and pose. Instead of separate heads for each modality, our model unifies all signals in a single cross-modal architecture. We train with a progressive, difficulty-aware sampling strategy that selects one control modality per example and biases sampling toward harder signals (e.g., skeletal pose) while downweighting easier ones (e.g., point clouds), encouraging robust multi-modal fusion and graceful handling of missing inputs. Experiments show that these additional controls improve generation accuracy, enable geometry-aware transformations, and increase robustness for production workflows. 

**Abstract (ZH)**: Recent advances in 3D-native generative models have accelerated asset creation for games, film, and design. However, most methods still rely primarily on image or text conditioning and lack fine-grained, cross-modal controls, which limits controllability and practical adoption. To address this gap, we present Hunyuan3D-Omni, a unified framework for fine-grained, controllable 3D asset generation built on Hunyuan3D 2.1. In addition to images, Hunyuan3D-Omni accepts point clouds, voxels, bounding boxes, and skeletal pose priors as conditioning signals, enabling precise control over geometry, topology, and pose. Instead of separate heads for each modality, our model unifies all signals in a single cross-modal architecture. We train with a progressive, difficulty-aware sampling strategy that selects one control modality per example and biases sampling toward harder signals (e.g., skeletal pose) while downweighting easier ones (e.g., point clouds), encouraging robust multi-modal fusion and graceful handling of missing inputs. Experiments show that these additional controls improve generation accuracy, enable geometry-aware transformations, and increase robustness for production workflows.

Title: 近年来，基于3D的生成模型在游戏、电影和设计中的资产创建方面取得了进步。然而，大多数方法仍然主要依赖于图像或文本条件，并缺乏细粒度的跨模态控制，这限制了可控性和实际应用。为解决这一问题，我们提出了Hunyuan3D-Omni，这是一个基于Hunyuan3D 2.1的统一框架，用于细粒度可控的3D资产生成。除了图像之外，Hunyuan3D-Omni还接受点云、体素、边界框和骨架姿态先验作为条件信号，使其能够精确控制几何形状、拓扑结构和姿态。与每个模态都有单独的头部不同，我们的模型将所有信号统合在一个跨模态架构中。我们采用一种渐进式、难度感知的采样策略进行训练，该策略为每个示例选择一种控制模态，并倾向于选择更难的信号（例如，骨架姿态）而降低更简单的信号（例如，点云）的重要性，以鼓励稳健的多模态融合并优雅地处理缺失输入。实验表明，这些额外的控制提高了生成的准确性，实现了几何感知的变换，并增强了生产工作流程的鲁棒性。 

---
# Explaining Fine Tuned LLMs via Counterfactuals A Knowledge Graph Driven Framework 

**Title (ZH)**: 基于知识图谱的反事实解释框架：细调大型语言模型 

**Authors**: Yucheng Wang, Ziyang Chen, Md Faisal Kabir  

**Link**: [PDF](https://arxiv.org/pdf/2509.21241)  

**Abstract**: The widespread adoption of Low-Rank Adaptation (LoRA) has enabled large language models (LLMs) to acquire domain-specific knowledge with remarkable efficiency. However, understanding how such a fine-tuning mechanism alters a model's structural reasoning and semantic behavior remains an open challenge. This work introduces a novel framework that explains fine-tuned LLMs via counterfactuals grounded in knowledge graphs. Specifically, we construct BioToolKG, a domain-specific heterogeneous knowledge graph in bioinformatics tools and design a counterfactual-based fine-tuned LLMs explainer (CFFTLLMExplainer) that learns soft masks over graph nodes and edges to generate minimal structural perturbations that induce maximum semantic divergence. Our method jointly optimizes structural sparsity and semantic divergence while enforcing interpretability preserving constraints such as entropy regularization and edge smoothness. We apply this framework to a fine-tuned LLaMA-based LLM and reveal that counterfactual masking exposes the model's structural dependencies and aligns with LoRA-induced parameter shifts. This work provides new insights into the internal mechanisms of fine-tuned LLMs and highlights counterfactual graphs as a potential tool for interpretable AI. 

**Abstract (ZH)**: Low-Rank Adaptation知识图谱引导的反事实解释框架：细调大型语言模型的结构依赖与语义偏差解析 

---
# Tree Search for LLM Agent Reinforcement Learning 

**Title (ZH)**: 树搜索在大型语言模型代理强化学习中的应用 

**Authors**: Yuxiang Ji, Ziyu Ma, Yong Wang, Guanhua Chen, Xiangxiang Chu, Liaoni Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21240)  

**Abstract**: Recent advances in reinforcement learning (RL) have significantly enhanced the agentic capabilities of large language models (LLMs). In long-term and multi-turn agent tasks, existing approaches driven solely by outcome rewards often suffer from the problem of sparse supervision. To address the challenge, we propose Tree-based Group Relative Policy Optimization (Tree-GRPO), a grouped agent RL method based on tree search, where each tree node represents the complete agent interaction step. By sharing common prefixes, the tree search sampling increases the number of rollouts achievable within a fixed budget of tokens or tool calls. Moreover, we find that the tree-structured trajectory naturally allows the construction of step-wise process supervised signals even using only the outcome reward. Based on this, Tree-GRPO estimates the grouped relative advantages both on intra-tree and inter-tree levels. Through theoretical analysis, we demonstrate that the objective of intra-tree level group relative policy optimization is equivalent to that of step-level direct preference learning. Experiments across 11 datasets and 3 types of QA tasks demonstrate the superiority of the proposed tree-based RL over the chain-based RL method. 

**Abstract (ZH)**: Recent Advances in Reinforcement Learning Have Significantly Enhanced the Agentic Capabilities of Large Language Models. To Address the Challenge of Sparse Supervision in Long-Term and Multi-Turn Agent Tasks, We Propose Tree-based Group Relative Policy Optimization (Tree-GRPO), a Grouped Agent RL Method Based on Tree Search. 

---
# Evading Overlapping Community Detection via Proxy Node Injection 

**Title (ZH)**: 通过代理节点注入规避重叠社区检测 

**Authors**: Dario Loi, Matteo Silvestri, Fabrizio Silvestri, Gabriele Tolomei  

**Link**: [PDF](https://arxiv.org/pdf/2509.21211)  

**Abstract**: Protecting privacy in social graphs requires preventing sensitive information, such as community affiliations, from being inferred by graph analysis, without substantially altering the graph topology. We address this through the problem of \emph{community membership hiding} (CMH), which seeks edge modifications that cause a target node to exit its original community, regardless of the detection algorithm employed. Prior work has focused on non-overlapping community detection, where trivial strategies often suffice, but real-world graphs are better modeled by overlapping communities, where such strategies fail. To the best of our knowledge, we are the first to formalize and address CMH in this setting. In this work, we propose a deep reinforcement learning (DRL) approach that learns effective modification policies, including the use of proxy nodes, while preserving graph structure. Experiments on real-world datasets show that our method significantly outperforms existing baselines in both effectiveness and efficiency, offering a principled tool for privacy-preserving graph modification with overlapping communities. 

**Abstract (ZH)**: 在社交图中保护隐私需要通过阻止敏感信息（例如社区隶属关系）被图分析推断出来，而不大幅改变图的拓扑结构。我们通过目标节点退出其原始社区的问题来解决这一挑战，即社区成员身份隐藏（CMH），无论采用哪种检测算法。先前的工作主要关注非重叠社区检测，其中简单的策略通常足够，但现实世界中的图更适合用重叠社区来建模，此时这些策略会失效。据我们所知，这是首次在重叠社区的背景下正式化并解决社区成员身份隐藏（CMH）问题。在本文中，我们提出了一种深度强化学习（DRL）方法，该方法在保持图结构的同时学习有效的修改策略，包括使用代理节点。实验结果表明，与现有的基线方法相比，我们的方法在效果和效率上均有显著提升，为重叠社区的隐私保护图修改提供了一个原理性的工具。 

---
# Eigen-1: Adaptive Multi-Agent Refinement with Monitor-Based RAG for Scientific Reasoning 

**Title (ZH)**: Eigen-1: 基于监测的RAG多agent自适应细化方法及其在科学推理中的应用 

**Authors**: Xiangru Tang, Wanghan Xu, Yujie Wang, Zijie Guo, Daniel Shao, Jiapeng Chen, Cixuan Zhang, Ziyi Wang, Lixin Zhang, Guancheng Wan, Wenlong Zhang, Lei Bai, Zhenfei Yin, Philip Torr, Hanrui Wang, Di Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.21193)  

**Abstract**: Large language models (LLMs) have recently shown strong progress on scientific reasoning, yet two major bottlenecks remain. First, explicit retrieval fragments reasoning, imposing a hidden "tool tax" of extra tokens and steps. Second, multi-agent pipelines often dilute strong solutions by averaging across all candidates. We address these challenges with a unified framework that combines implicit retrieval and structured collaboration. At its foundation, a Monitor-based retrieval module operates at the token level, integrating external knowledge with minimal disruption to reasoning. On top of this substrate, Hierarchical Solution Refinement (HSR) iteratively designates each candidate as an anchor to be repaired by its peers, while Quality-Aware Iterative Reasoning (QAIR) adapts refinement to solution quality. On Humanity's Last Exam (HLE) Bio/Chem Gold, our framework achieves 48.3\% accuracy -- the highest reported to date, surpassing the strongest agent baseline by 13.4 points and leading frontier LLMs by up to 18.1 points, while simultaneously reducing token usage by 53.5\% and agent steps by 43.7\%. Results on SuperGPQA and TRQA confirm robustness across domains. Error analysis shows that reasoning failures and knowledge gaps co-occur in over 85\% of cases, while diversity analysis reveals a clear dichotomy: retrieval tasks benefit from solution variety, whereas reasoning tasks favor consensus. Together, these findings demonstrate how implicit augmentation and structured refinement overcome the inefficiencies of explicit tool use and uniform aggregation. Code is available at: this https URL. 

**Abstract (ZH)**: 大型语言模型在科学研究中已显示出强大的推理能力，但仍存在两大瓶颈。首先，显式的检索会中断推理，导致额外的令牌和步骤。其次，多代理管道常通过平均所有候选方案来稀释强大的解决方案。我们通过结合隐式检索和结构化协作的统一框架来应对这些挑战。该框架的基础是一个基于监控的检索模块，在最低限度地干扰推理的前提下，整合外部知识。在此基础上，层次化解决方案细化（HSR）逐次将每个候选方案指定为锚点并通过其他候选方案进行修正，而质量意识迭代推理（QAIR）调整细化以适应解决方案的质量。在《人类的最后一考》（HLE）生物/化学黄金标准数据集上，我们的框架实现了48.3%的准确率，这是迄今最高记录，比最强的单一代理基线高出13.4个百分点，并且领先前沿的大规模语言模型多达18.1个百分点，同时减少了53.5%的令牌使用和43.7%的代理步骤。在SuperGPQA和TRQA上获得的结果证实了其在不同领域的鲁棒性。错误分析显示，推理失败和知识空白在同一案例中共同出现的比例超过85%，而多样性分析揭示了一个清晰的二分法：检索任务受益于多样化的解决方案，而推理任务更倾向于共识。这些发现共同表明，隐式增强和结构化细化如何克服显式工具使用和均匀聚合的低效性。代码可在以下链接获取：this https URL。 

---
# Towards Foundation Models for Zero-Shot Time Series Anomaly Detection: Leveraging Synthetic Data and Relative Context Discrepancy 

**Title (ZH)**: 面向零样本时间序列异常检测的基础模型：利用合成数据和相对上下文差异 

**Authors**: Tian Lan, Hao Duong Le, Jinbo Li, Wenjun He, Meng Wang, Chenghao Liu, Chen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21190)  

**Abstract**: Time series anomaly detection (TSAD) is a critical task, but developing models that generalize to unseen data in a zero-shot manner remains a major challenge. Prevailing foundation models for TSAD predominantly rely on reconstruction-based objectives, which suffer from a fundamental objective mismatch: they struggle to identify subtle anomalies while often misinterpreting complex normal patterns, leading to high rates of false negatives and positives. To overcome these limitations, we introduce \texttt{TimeRCD}, a novel foundation model for TSAD built upon a new pre-training paradigm: Relative Context Discrepancy (RCD). Instead of learning to reconstruct inputs, \texttt{TimeRCD} is explicitly trained to identify anomalies by detecting significant discrepancies between adjacent time windows. This relational approach, implemented with a standard Transformer architecture, enables the model to capture contextual shifts indicative of anomalies that reconstruction-based methods often miss. To facilitate this paradigm, we develop a large-scale, diverse synthetic corpus with token-level anomaly labels, providing the rich supervisory signal necessary for effective pre-training. Extensive experiments demonstrate that \texttt{TimeRCD} significantly outperforms existing general-purpose and anomaly-specific foundation models in zero-shot TSAD across diverse datasets. Our results validate the superiority of the RCD paradigm and establish a new, effective path toward building robust and generalizable foundation models for time series anomaly detection. 

**Abstract (ZH)**: 时间序列异常检测（TSAD）是一个关键任务，但开发能够在零样本情况下泛化的模型仍然是一个重大挑战。现有的TSAD基础模型主要依赖于重建目标，这存在根本性的目标不匹配：它们难以识别细微的异常，常常误判复杂的正常模式，导致高比例的误报和漏报。为克服这些限制，我们引入了\texttt{TimeRCD}，这是一种基于新型预训练范式的TSAD基础模型：相对上下文差异（RCD）。与学习重建输入不同，\texttt{TimeRCD} 明确训练以通过检测相邻时间窗口之间的重要差异来识别异常。这一关系方法，结合标准的Transformer架构，使模型能够捕捉到异常指示的上下文变化，这是基于重建的方法经常忽略的。为支持这一范式，我们开发了一个大规模、多样化的合成语料库，包含标记异常的标记级别标签，提供了有效预训练所需的丰富的监督信号。广泛的实验表明，\texttt{TimeRCD} 在零样本时间序列异常检测任务中显著优于现有的基础模型和专门针对异常的基础模型，来自多个数据集的结果验证了RCD范式的优越性，并为建立鲁棒且泛化能力强的时间序列异常检测基础模型奠定了新的有效路径。 

---
# Human-like Navigation in a World Built for Humans 

**Title (ZH)**: 人类导航于为人类设计的世界中 

**Authors**: Bhargav Chandaka, Gloria X. Wang, Haozhe Chen, Henry Che, Albert J. Zhai, Shenlong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21189)  

**Abstract**: When navigating in a man-made environment they haven't visited before--like an office building--humans employ behaviors such as reading signs and asking others for directions. These behaviors help humans reach their destinations efficiently by reducing the need to search through large areas. Existing robot navigation systems lack the ability to execute such behaviors and are thus highly inefficient at navigating within large environments. We present ReasonNav, a modular navigation system which integrates these human-like navigation skills by leveraging the reasoning capabilities of a vision-language model (VLM). We design compact input and output abstractions based on navigation landmarks, allowing the VLM to focus on language understanding and reasoning. We evaluate ReasonNav on real and simulated navigation tasks and show that the agent successfully employs higher-order reasoning to navigate efficiently in large, complex buildings. 

**Abstract (ZH)**: 当在未访问过的建筑物（如办公楼）中导航时，人类会表现出阅读指示牌和向他人询问方向等行为，这些行为有助于人类高效地到达目的地，减少大面积搜索的需要。现有的机器人导航系统缺乏执行此类行为的能力，因此在大型环境中的导航效率极低。我们提出了ReasonNav，这是一种模块化的导航系统，通过利用视觉语言模型（VLM）的推理能力来整合这些类似人类的导航技能。我们基于导航地标设计了紧凑的输入和输出抽象，使VLM能够专注于语言理解与推理。我们在实际和模拟的导航任务中评估了ReasonNav，并展示了该代理能够运用高级推理在大型复杂建筑中高效导航。 

---
# Adoption, usability and perceived clinical value of a UK AI clinical reference platform (iatroX): a mixed-methods formative evaluation of real-world usage and a 1,223-respondent user survey 

**Title (ZH)**: 英国AI临床参考平台（iatroX）的采用、易用性和临床感知价值：现实世界使用情况的混合方法形成性评估及1223名用户调查 

**Authors**: Kolawole Tytler  

**Link**: [PDF](https://arxiv.org/pdf/2509.21188)  

**Abstract**: Clinicians face growing information overload from biomedical literature and guidelines, hindering evidence-based care. Retrieval-augmented generation (RAG) with large language models may provide fast, provenance-linked answers, but requires real-world evaluation. We describe iatroX, a UK-centred RAG-based clinical reference platform, and report early adoption, usability, and perceived clinical value from a formative implementation evaluation. Methods comprised a retrospective analysis of usage across web, iOS, and Android over 16 weeks (8 April-31 July 2025) and an in-product intercept survey. Usage metrics were drawn from web and app analytics with bot filtering. A client-side script randomized single-item prompts to approx. 10% of web sessions from a predefined battery assessing usefulness, reliability, and adoption intent. Proportions were summarized with Wilson 95% confidence intervals; free-text comments underwent thematic content analysis. iatroX reached 19,269 unique web users, 202,660 engagement events, and approx. 40,000 clinical queries. Mobile uptake included 1,960 iOS downloads and Android growth (peak >750 daily active users). The survey yielded 1,223 item-level responses: perceived usefulness 86.2% (95% CI 74.8-93.9%; 50/58); would use again 93.3% (95% CI 68.1-99.8%; 14/15); recommend to a colleague 88.4% (95% CI 75.1-95.9%; 38/43); perceived accuracy 75.0% (95% CI 58.8-87.3%; 30/40); reliability 79.4% (95% CI 62.1-91.3%; 27/34). Themes highlighted speed, guideline-linked answers, and UK specificity. Early real-world use suggests iatroX can mitigate information overload and support timely answers for UK clinicians. Limitations include small per-item samples and early-adopter bias; future work will include accuracy audits and prospective studies on workflow and care quality. 

**Abstract (ZH)**: 临床医生面临来自生物医学文献和指南的信息过载问题，影响基于证据的医疗护理。大型语言模型增强的检索生成（RAG）可能提供快速、溯源的答案，但需要实地评估。我们描述了iatroX，一个以英国为中心的基于RAG的临床参考平台，并报告了初步采用、易用性和临床价值的形成性实施评估结果。 

---
# Can Less Precise Be More Reliable? A Systematic Evaluation of Quantization's Impact on CLIP Beyond Accuracy 

**Title (ZH)**: 精度较低的量化是否更具可靠性？CLIP 准确性之外的量化影响系统评价 

**Authors**: Aymen Bouguerra, Daniel Montoya, Alexandra Gomez-Villa, Fabio Arnez, Chokri Mraidha  

**Link**: [PDF](https://arxiv.org/pdf/2509.21173)  

**Abstract**: The powerful zero-shot generalization capabilities of vision-language models (VLMs) like CLIP have enabled new paradigms for safety-related tasks such as out-of-distribution (OOD) detection. However, additional aspects crucial for the computationally efficient and reliable deployment of CLIP are still overlooked. In particular, the impact of quantization on CLIP's performance beyond accuracy remains underexplored. This work presents a large-scale evaluation of quantization on CLIP models, assessing not only in-distribution accuracy but a comprehensive suite of reliability metrics and revealing counterintuitive results driven by pre-training source. We demonstrate that quantization consistently improves calibration for typically underconfident pre-trained models, while often degrading it for overconfident variants. Intriguingly, this degradation in calibration does not preclude gains in other reliability metrics; we find that OOD detection can still improve for these same poorly calibrated models. Furthermore, we identify specific quantization-aware training (QAT) methods that yield simultaneous gains in zero-shot accuracy, calibration, and OOD robustness, challenging the view of a strict efficiency-performance trade-off. These findings offer critical insights for navigating the multi-objective problem of deploying efficient, reliable, and robust VLMs by utilizing quantization beyond its conventional role. 

**Abstract (ZH)**: 视觉语言模型（VLMs）如CLIP的强大零样本泛化能力为安全相关任务，如分布外（OOD）检测等新范式提供了可能。然而，计算高效且可靠的CLIP部署还需要考虑其他关键方面，这些方面尚未得到充分重视。特别是，量化对CLIP性能的影响（超出准确性的方面）尚未得到充分探索。本研究对CLIP模型进行了大规模量化评估，不仅评估了分布内准确性，还评估了全面的可靠性指标，并揭示了由预训练来源驱动的反直觉结果。我们证明，量化一致地提高了通常欠自信预训练模型的校准，而在自信过高的变体中却常常降低校准。有趣的是，这种校准的降低并不妨碍其他可靠性指标的改善；我们发现，对于这些同样欠校准的模型，分布外检测性能仍然可以提高。此外，我们识别出特定的量化感知训练（QAT）方法，这些方法同时提高了零样本准确性、校准和分布外鲁棒性，挑战了效率与性能之间严格权衡的观点。这些发现为通过超越传统角色的量化部署高效、可靠和鲁棒的VLM提供了关键见解。 

---
# Fine-Tuning LLMs to Analyze Multiple Dimensions of Code Review: A Maximum Entropy Regulated Long Chain-of-Thought Approach 

**Title (ZH)**: 细调大型语言模型以分析代码审查的多维度：一种最大熵调节长链思考方法 

**Authors**: Yongda Yu, Guohao Shi, Xianwei Wu, Haochuan He, XueMing Gu, Qianqian Zhao, Kui Liu, Qiushi Wang, Zhao Tian, Haifeng Shen, Guoping Rong  

**Link**: [PDF](https://arxiv.org/pdf/2509.21170)  

**Abstract**: Large Language Models (LLMs) have shown great potential in supporting automated code review due to their impressive capabilities in context understanding and reasoning. However, these capabilities are still limited compared to human-level cognition because they are heavily influenced by the training data. Recent research has demonstrated significantly improved performance through fine-tuning LLMs with code review data. However, compared to human reviewers who often simultaneously analyze multiple dimensions of code review to better identify issues, the full potential of these methods is hampered by the limited or vague information used to fine-tune the models. This paper contributes MelcotCR, a chain-of-thought (COT) fine-tuning approach that trains LLMs with an impressive reasoning ability to analyze multiple dimensions of code review by harnessing long COT techniques to provide rich structured information. To address context loss and reasoning logic loss issues that frequently occur when LLMs process long COT prompts, we propose a solution that combines the Maximum Entropy (ME) modeling principle with pre-defined reasoning pathways in MelcotCR to enable more effective utilization of in-context knowledge within long COT prompts while strengthening the logical tightness of the reasoning process. Empirical evaluations on our curated MelcotCR dataset and the public CodeReviewer dataset reveal that a low-parameter base model, such as 14B Qwen2.5, fine-tuned with MelcotCR can surpass state-of-the-art methods in terms of the accuracy of detecting and describing code issues, with its performance remarkably on par with that of the 671B DeepSeek-R1 model. 

**Abstract (ZH)**: 大型语言模型（LLMs）在支持自动化代码审查方面显示出巨大的潜力，得益于其在上下文理解与推理能力方面的 impressive 表现。然而，这些能力仍受限于训练数据，无法达到人类认知水平。近期研究通过使用代码审查数据对 LLMs 进行微调，显著提升了其性能。然而，相比于常能同时分析多个代码审查维度的人类审查者，这些方法受限于用于微调模型的有限或模糊信息，未能充分发挥潜力。本文贡献了一种名为 MelcotCR 的chain-of-thought（CoT）微调方法，通过利用长 CoT 技术提供丰富的结构化信息，训练 LLMs 以分析代码审查的多个维度。为了解决 LLMs 在处理长 CoT 提示时经常出现的上下文损失和推理逻辑损失问题，提出了结合最大熵（ME）建模原则和预定义推理路径的方法，以在长 CoT 提示中更有效地利用上下文知识，并增强推理过程的逻辑紧密性。在我们编纂的 MelcotCR 数据集和公开的 CodeReviewer 数据集上的实证评估表明，使用 MelcotCR 微调的低参数基础模型（如 14B Qwen2.5），在检测和描述代码问题的准确性方面可超越现有先进方法，其性能显著接近 671B DeepSeek-R1 模型。 

---
# GRPO is Secretly a Process Reward Model 

**Title (ZH)**: GRPO秘密上是一个过程奖励模型 

**Authors**: Michael Sullivan  

**Link**: [PDF](https://arxiv.org/pdf/2509.21154)  

**Abstract**: We prove theoretically that the GRPO RL algorithm induces a non-trivial process reward model (PRM), under certain assumptions regarding within-group overlap of token sequences across completions. We then show empirically that these assumptions are met under real-world conditions: GRPO does in fact induce a non-trivial PRM. Leveraging the framework of GRPO-as-a-PRM, we identify a flaw in the GRPO objective: non-uniformly distributed process steps hinder both exploration and exploitation (under different conditions). We propose a simple modification to the algorithm to mitigate this defect ($\lambda$-GRPO), and show that LLMs trained with $\lambda$-GRPO achieve higher validation accuracy and performance on downstream reasoning tasks$-$and reach peak performance more rapidly$-$than LLMs trained with standard GRPO. Our results call into question the advantage of costly, explicitly-defined PRMs for GRPO: we show that it is possible to instead leverage the hidden, built-in PRM structure within the vanilla GRPO algorithm to boost model performance with a negligible impact on training time and cost. 

**Abstract (ZH)**: 我们证明，在某些假设条件下，GRPO RL算法会产生一个非平凡的过程奖励模型（PRM）。随后的实验证明了这些假设在实际条件中成立：GRPO确实诱导了一个非平凡的PRM。利用GRPO-as-a-PRM的框架，我们发现了GRPO目标函数的一个缺陷：非均匀分布的过程步骤既妨碍探索又妨碍利用（在不同条件下）。我们提出了一种简单的算法修改（$\lambda$-GRPO）来弥补这一缺陷，并展示了使用$\lambda$-GRPO训练的语言模型在下游推理任务中的验证准确性更高、性能更好，并且达到最佳性能的速度更快。我们的结果质疑了成本高昂且显式定义的PRM对GRPO的优势：我们证明了可以通过利用 vanilla GRPO算法中的隐藏且内置的PRM结构来提升模型性能，同时对训练时间和成本的影响可以忽略不计。 

---
# WAVECLIP: Wavelet Tokenization for Adaptive-Resolution CLIP 

**Title (ZH)**: WAVECLIP: 小波 Tokenization 用于自适应分辨率 CLIP 

**Authors**: Moshe Kimhi, Erez Koifman, Ehud Rivlin, Eli Schwartz, Chaim Baskin  

**Link**: [PDF](https://arxiv.org/pdf/2509.21153)  

**Abstract**: We introduce WAVECLIP, a single unified model for adaptive resolution inference in CLIP, enabled by wavelet-based tokenization. WAVECLIP replaces standard patch embeddings with a multi-level wavelet decomposition, enabling the model to process images coarse to fine while naturally supporting multiple resolutions within the same model. At inference time, the model begins with low resolution tokens and refines only when needed, using key-value caching and causal cross-level attention to reuse computation, effectively introducing to the model only new information when needed. We evaluate WAVECLIP in zero-shot classification, demonstrating that a simple confidence-based gating mechanism enables adaptive early exits. This allows users to dynamically choose a compute-accuracy trade-off using a single deployed model. Our approach requires only lightweight distillation from a frozen CLIP teacher and achieves competitive accuracy with significant computational savings. 

**Abstract (ZH)**: WAVECLIP：基于小波的统一模型在CLIP中的自适应分辨率推理 

---
# LAVA: Explainability for Unsupervised Latent Embeddings 

**Title (ZH)**: LAVA: 未监督潜在嵌入的可解释性 

**Authors**: Ivan Stresec, Joana P. Gonçalves  

**Link**: [PDF](https://arxiv.org/pdf/2509.21149)  

**Abstract**: Unsupervised black-box models can be drivers of scientific discovery, but remain difficult to interpret. Crucially, discovery hinges on understanding the model output, which is often a multi-dimensional latent embedding rather than a well-defined target. While explainability for supervised learning usually seeks to uncover how input features are used to predict a target, its unsupervised counterpart should relate input features to the structure of the learned latent space. Adaptations of supervised model explainability for unsupervised learning provide either single-sample or dataset-wide summary explanations. However, without automated strategies of relating similar samples to one another guided by their latent proximity, explanations remain either too fine-grained or too reductive to be meaningful. This is especially relevant for manifold learning methods that produce no mapping function, leaving us only with the relative spatial organization of their embeddings. We introduce Locality-Aware Variable Associations (LAVA), a post-hoc model-agnostic method designed to explain local embedding organization through its relationship with the input features. To achieve this, LAVA represents the latent space as a series of localities (neighborhoods) described in terms of correlations between the original features, and then reveals reoccurring patterns of correlations across the entire latent space. Based on UMAP embeddings of MNIST and a single-cell kidney dataset, we show that LAVA captures relevant feature associations, with visually and biologically relevant local patterns shared among seemingly distant regions of the latent spaces. 

**Abstract (ZH)**: 无监督黑盒模型可以驱动科学发现，但仍然难以解释。关键在于理解模型输出，这通常是一个多维度的潜在嵌入，而非明确的目标。虽然监督学习的解释性通常旨在揭示输入特征如何用于预测目标，其无监督对应物应将输入特征与学习到的潜在空间结构相关联。针对无监督学习适应的监督模型解释性方法提供了单个样本或整 dataset 的总结性解释。然而，缺乏自动化的策略来根据其潜在邻居关系关联相似样本，使得解释要么过细要么过于简化，缺乏意义。这对于生成无映射函数的流形学习方法尤为重要，使我们只能依赖其嵌入的相对空间组织。我们引入了局部意识变量关联（LAVA），这是一种后验的模型无关方法，旨在通过其与输入特征的关系解释潜在嵌入的局部组织。为了实现这一点，LAVA 将潜在空间表示为一系列局部性（邻里），并通过原始特征之间的相关性描述，然后揭示在整个潜在空间中反复出现的相关性模式。基于MNIST嵌入和单细胞肾脏数据集，我们展示了LAVA捕获了相关特征关联，在潜在空间中看似遥远区域之间共享了视觉上和生物学上有意义的局部模式。 

---
# Emerging Paradigms for Securing Federated Learning Systems 

**Title (ZH)**: 新兴范式确保联邦学习系统安全 

**Authors**: Amr Akmal Abouelmagd, Amr Hilal  

**Link**: [PDF](https://arxiv.org/pdf/2509.21147)  

**Abstract**: Federated Learning (FL) facilitates collaborative model training while keeping raw data decentralized, making it a conduit for leveraging the power of IoT devices while maintaining privacy of the locally collected data. However, existing privacy- preserving techniques present notable hurdles. Methods such as Multi-Party Computation (MPC), Homomorphic Encryption (HE), and Differential Privacy (DP) often incur high compu- tational costs and suffer from limited scalability. This survey examines emerging approaches that hold promise for enhancing both privacy and efficiency in FL, including Trusted Execution Environments (TEEs), Physical Unclonable Functions (PUFs), Quantum Computing (QC), Chaos-Based Encryption (CBE), Neuromorphic Computing (NC), and Swarm Intelligence (SI). For each paradigm, we assess its relevance to the FL pipeline, outlining its strengths, limitations, and practical considerations. We conclude by highlighting open challenges and prospective research avenues, offering a detailed roadmap for advancing secure and scalable FL systems. 

**Abstract (ZH)**: 联邦学习（FL）促进了模型的协同训练，同时保持原始数据的分散化，使其成为利用物联网设备的权力并维护本地收集数据隐私的渠道。然而，现有的隐私保护技术存在明显的障碍。诸如多方计算（MPC）、同态加密（HE）和差分隐私（DP）等方法往往带来高昂的计算成本且可扩展性有限。本文综述了新兴的方法，这些方法有望在提高FL的隐私性和效率方面发挥作用，包括可信执行环境（TEEs）、物理不可克隆功能（PUFs）、量子计算（QC）、混沌加密（CBE）、类脑计算（NC）和群智方法（SI）。对于每种范式，我们评估其与FL管道的相关性，概述其优势、局限性和实际考虑因素。最后，我们指出现有的挑战和潜在的研究方向，提供了一份详细的技术路线图，以推动安全可扩展的联邦学习系统的发展。 

---
# UniSS: Unified Expressive Speech-to-Speech Translation with Your Voice 

**Title (ZH)**: UniSS: 统一的声纹驱动语音到语音翻译 

**Authors**: Sitong Cheng, Weizhen Bian, Xinsheng Wang, Ruibin Yuan, Jianyi Chen, Shunshun Yin, Yike Guo, Wei Xue  

**Link**: [PDF](https://arxiv.org/pdf/2509.21144)  

**Abstract**: The ultimate goal of expressive speech-to-speech translation (S2ST) is to accurately translate spoken content while preserving the speaker identity and emotional style. However, progress in this field is largely hindered by three key challenges: the scarcity of paired speech data that retains expressive styles, the complexity of multi-stage processing pipelines, and the limited transfer of translation capabilities from large language models (LLMs). In this work, we address these challenges by introducing UniSS, a novel single-stage framework for expressive S2ST. Our approach features carefully designed speech semantic and style modeling, enabling seamless integration with existing text-based LLM frameworks to develop a unified text-speech language model. To transfer translation capabilities from text to speech, we propose a cross-modal chain-of-thought prompting process that progressively aligns audio semantics with text and ensures style preservation in the decoded results. Furthermore, we construct and release a large-scale, high-quality expressive S2ST dataset, UniST, comprising 44.8k hours of data. Experimental results show that UniSS significantly outperforms previous methods in translation fidelity and speech quality while preserving voice, emotion, and duration consistency. Our work establishes a simpler and more effective paradigm for building the next generation of expressive S2ST systems. Audio samples are available at this https URL. 

**Abstract (ZH)**: 表达性语音到语音翻译（S2ST）的最终目标是在准确翻译口语内容的同时保留说话人身份和情感风格。然而，这一领域的发展受到三大关键挑战的阻碍：保留表达风格的配对语音数据稀缺、多阶段处理管道的复杂性以及从大型语言模型（LLMs）转移翻译能力的局限性。在本文中，我们通过引入UniSS——一种新颖的一阶段表达性S2ST框架来应对这些挑战。我们的方法特色是精心设计的语音语义和风格建模，能够无缝集成现有的基于文本的LLM框架，开发统一的文本-语音语言模型。为了从文本向语音转移翻译能力，我们提出了一种跨模态链式思考提示过程，逐步将音频语义与文本对齐，并确保解码结果中的风格一致性。此外，我们构建并发布了包含44800小时数据的巨大质量表达性S2ST数据集UniST。实验证明，UniSS在翻译保真度和语音质量方面明显优于之前的方法，同时保留了声音、情感和持续时间的一致性。我们的工作建立了构建下一代表达性S2ST系统的更简单和更有效范式。请参见此链接获取音频样本：this https URL。 

---
# Teaching RL Agents to Act Better: VLM as Action Advisor for Online Reinforcement Learning 

**Title (ZH)**: 教学 RL 代理更好地行动：大规模语言模型作为在线强化学习的动作顾问 

**Authors**: Xiefeng Wu, Jing Zhao, Shu Zhang, Mingyu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21126)  

**Abstract**: Online reinforcement learning in complex tasks is time-consuming, as massive interaction steps are needed to learn the optimal this http URL-language action (VLA) policies represent a promising direction for solving diverse tasks; however, their performance on low-level control remains limited, and effective deployment often requires task-specific expert demonstrations for fine-tuning. In this paper, we propose \textbf{VARL} (\textbf{V}LM as \textbf{A}ction advisor for online \textbf{R}einforcement \textbf{L}earning), a framework that leverages the domain knowledge of vision-language models (VLMs) to provide action suggestions for reinforcement learning agents. Unlike previous methods, VARL provides action suggestions rather than designing heuristic rewards, thereby guaranteeing unchanged optimality and convergence. The suggested actions increase sample diversity and ultimately improve sample efficiency, especially in sparse-reward tasks. To validate the effectiveness of VARL, we evaluate it across diverse environments and agent settings. Results show that VARL greatly improves sample efficiency without introducing significant computational overhead. These advantages make VARL a general framework for online reinforcement learning and make it feasible to directly apply reinforcement learning from scratch in real-world environments. 

**Abstract (ZH)**: 基于视觉语言模型的在线强化学习框架：VLM作为在线强化学习的动作顾问（VARL） 

---
# Cross-Modal Instructions for Robot Motion Generation 

**Title (ZH)**: 跨模态指令生成机器人运动 

**Authors**: William Barron, Xiaoxiang Dong, Matthew Johnson-Roberson, Weiming Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2509.21107)  

**Abstract**: Teaching robots novel behaviors typically requires motion demonstrations via teleoperation or kinaesthetic teaching, that is, physically guiding the robot. While recent work has explored using human sketches to specify desired behaviors, data collection remains cumbersome, and demonstration datasets are difficult to scale. In this paper, we introduce an alternative paradigm, Learning from Cross-Modal Instructions, where robots are shaped by demonstrations in the form of rough annotations, which can contain free-form text labels, and are used in lieu of physical motion. We introduce the CrossInstruct framework, which integrates cross-modal instructions as examples into the context input to a foundational vision-language model (VLM). The VLM then iteratively queries a smaller, fine-tuned model, and synthesizes the desired motion over multiple 2D views. These are then subsequently fused into a coherent distribution over 3D motion trajectories in the robot's workspace. By incorporating the reasoning of the large VLM with a fine-grained pointing model, CrossInstruct produces executable robot behaviors that generalize beyond the environment of in the limited set of instruction examples. We then introduce a downstream reinforcement learning pipeline that leverages CrossInstruct outputs to efficiently learn policies to complete fine-grained tasks. We rigorously evaluate CrossInstruct on benchmark simulation tasks and real hardware, demonstrating effectiveness without additional fine-tuning and providing a strong initialization for policies subsequently refined via reinforcement learning. 

**Abstract (ZH)**: 从跨模态指令学习 

---
# GraphUniverse: Enabling Systematic Evaluation of Inductive Generalization 

**Title (ZH)**: GraphUniverse: 促进归纳泛化系统评价 

**Authors**: Louis Van Langendonck, Guillermo Bernárdez, Nina Miolane, Pere Barlet-Ros  

**Link**: [PDF](https://arxiv.org/pdf/2509.21097)  

**Abstract**: A fundamental challenge in graph learning is understanding how models generalize to new, unseen graphs. While synthetic benchmarks offer controlled settings for analysis, existing approaches are confined to single-graph, transductive settings where models train and test on the same graph structure. Addressing this gap, we introduce GraphUniverse, a framework for generating entire families of graphs to enable the first systematic evaluation of inductive generalization at scale. Our core innovation is the generation of graphs with persistent semantic communities, ensuring conceptual consistency while allowing fine-grained control over structural properties like homophily and degree distributions. This enables crucial but underexplored robustness tests, such as performance under controlled distribution shifts. Benchmarking a wide range of architectures -- from GNNs to graph transformers and topological architectures -- reveals that strong transductive performance is a poor predictor of inductive generalization. Furthermore, we find that robustness to distribution shift is highly sensitive not only to model architecture choice but also to the initial graph regime (e.g., high vs. low homophily). Beyond benchmarking, GraphUniverse's flexibility and scalability can facilitate the development of robust and truly generalizable architectures -- including next-generation graph foundation models. An interactive demo is available at this https URL. 

**Abstract (ZH)**: 图学习中的一个基本挑战是如何理解模型在新未见过的图上的泛化能力。虽然合成基准提供了一种可控的分析环境，但现有方法局限于基于单个图的归纳设置，其中模型在相同的图结构上进行训练和测试。为解决这一问题，我们引入了GraphUniverse框架，以生成整个图家族，从而首次在大规模上系统地评估归纳泛化能力。我们的核心创新在于生成具有持久语义社区的图，确保概念一致性的同时，允许对结构属性（如同质性和度分布）进行细粒度控制。这使得可以进行关键但尚未充分探索的鲁棒性测试，例如在受控分布偏移下的性能测试。在从图神经网络到图变压器和拓扑架构的各种模型架构上进行基准测试表明，强大的归纳性能并不是归纳泛化能力的可靠预测指标。此外，我们发现对分布偏移的鲁棒性不仅高度依赖于模型架构选择，也高度依赖于初始图环境（例如，高同质性 vs 低同质性）。除了基准测试，GraphUniverse的高度灵活性和可扩展性可以促进鲁棒且真正泛化的模型架构的发展，包括下一代图基础模型。更多内容可在以下链接访问：this https URL。 

---
# Best-of-$\infty$ -- Asymptotic Performance of Test-Time Compute 

**Title (ZH)**: Best-of-$\infty$ —— 测试时计算的渐近性能 

**Authors**: Junpei Komiyama, Daisuke Oba, Masafumi Oyamada  

**Link**: [PDF](https://arxiv.org/pdf/2509.21091)  

**Abstract**: We study best-of-$N$ for large language models (LLMs) where the selection is based on majority voting. In particular, we analyze the limit $N \to \infty$, which we denote as Best-of-$\infty$. While this approach achieves impressive performance in the limit, it requires an infinite test-time budget. To address this, we propose an adaptive generation scheme that selects $N$ based on answer agreement, thereby efficiently allocating inference-time computation. Beyond adaptivity, we extend the framework to weighted ensembles of multiple LLMs, showing that such mixtures can outperform any individual model. The optimal ensemble weighting is formulated and efficiently computed as a mixed-integer linear program. Extensive experiments demonstrate the effectiveness of our approach. 

**Abstract (ZH)**: 我们研究基于多数投票的大语言模型（LLM）最佳-of-$N$方法，并分析其在$N \to \infty$极限情况下的表现，即所谓的最佳-of-$\infty$。虽然这种方法在极限情况下能取得出色的性能，但需要无限的测试计算预算。为解决这一问题，我们提出了一种自适应生成方案，根据答案一致性的程度选择$N$，从而高效地分配推理计算资源。此外，我们扩展了框架以应用于多个LLM的加权组合，并证明这种组合方法可以优于任何单独的模型。我们将最优的组合加权形式化并高效地作为混合整数线性规划问题进行求解。大量的实验验证了我们方法的有效性。 

---
# Vision Transformers: the threat of realistic adversarial patches 

**Title (ZH)**: 视觉变换器：现实主义 adversarial 崩溃图案的威胁 

**Authors**: Kasper Cools, Clara Maathuis, Alexander M. van Oers, Claudia S. Hübner, Nikos Deligiannis, Marijke Vandewal, Geert De Cubber  

**Link**: [PDF](https://arxiv.org/pdf/2509.21084)  

**Abstract**: The increasing reliance on machine learning systems has made their security a critical concern. Evasion attacks enable adversaries to manipulate the decision-making processes of AI systems, potentially causing security breaches or misclassification of targets. Vision Transformers (ViTs) have gained significant traction in modern machine learning due to increased 1) performance compared to Convolutional Neural Networks (CNNs) and 2) robustness against adversarial perturbations. However, ViTs remain vulnerable to evasion attacks, particularly to adversarial patches, unique patterns designed to manipulate AI classification systems. These vulnerabilities are investigated by designing realistic adversarial patches to cause misclassification in person vs. non-person classification tasks using the Creases Transformation (CT) technique, which adds subtle geometric distortions similar to those occurring naturally when wearing clothing. This study investigates the transferability of adversarial attack techniques used in CNNs when applied to ViT classification models. Experimental evaluation across four fine-tuned ViT models on a binary person classification task reveals significant vulnerability variations: attack success rates ranged from 40.04% (google/vit-base-patch16-224-in21k) to 99.97% (facebook/dino-vitb16), with google/vit-base-patch16-224 achieving 66.40% and facebook/dinov3-vitb16 reaching 65.17%. These results confirm the cross-architectural transferability of adversarial patches from CNNs to ViTs, with pre-training dataset scale and methodology strongly influencing model resilience to adversarial attacks. 

**Abstract (ZH)**: 机器学习系统安全性的日益依赖使其安全性成为关键问题。逃逸攻击使对手能够操控AI系统的决策过程，可能导致安全漏洞或目标误分类。由于与卷积神经网络（CNNs）相比具有更高的1）性能和2）对抗扰动的稳健性，视觉变换器（ViTs）在现代机器学习中受到广泛关注。然而，ViTs仍然容易受到逃逸攻击的影响，特别是对抗性补丁的影响，这是一种旨在操纵AI分类系统的独特模式。通过设计基于折痕变换（CT）技术的现实主义对抗性补贴，这些漏洞在人员与非人员分类任务中引起误分类，以调查CNN中使用的对抗攻击技术在应用于ViT分类模型时的可转移性。在针对二元人员分类任务的四种微调ViT模型上的实验评估显示，攻击成功率范围从40.04%（google/vit-base-patch16-224-in21k）到99.97%（facebook/dino-vitb16），其中google/vit-base-patch16-224达到66.40%，facebook/dinov3-vitb16达到65.17%。这些结果证实了来自CNN的对抗性补丁在不同架构之间的可移植性，前期训练数据集规模和方法对模型对抗攻击的鲁棒性有重要影响。 

---
# TyphoonMLA: A Mixed Naive-Absorb MLA Kernel For Shared Prefix 

**Title (ZH)**: TyphoonMLA: 一种混合朴素吸收前缀的MLA内核 

**Authors**: Ahmet Caner Yüzügüler, Ahmet Çelik, Jiawei Zhuang, Lukas Cavigelli  

**Link**: [PDF](https://arxiv.org/pdf/2509.21081)  

**Abstract**: Multi-Head Latent Attention (MLA) is a recent attention mechanism adopted in state-of-the-art LLMs such as DeepSeek-v3 and Kimi K2. Thanks to its novel formulation, MLA allows two functionally equivalent but computationally distinct kernel implementations: naive and absorb. While the naive kernels (e.g., FlashAttention) are typically preferred in training and prefill for their computational efficiency, existing decoding kernels (e.g., FlashMLA) rely on the absorb method to minimize HBM bandwidth usage. However, the compute-bound nature of the absorb implementations prohibits performance benefits from data reuse opportunities in attention calculations, such as shared prefixes. In this work, we introduce TyphoonMLA, a hybrid approach that combines naive and absorb formulations to harness the strengths of both. TyphoonMLA effectively leverages the shared prefix by applying the naive formulation to the compute-bound parts of attention calculations, while reducing the bandwidth requirements for non-shared parts by using the absorb formulation. As a result, TyphoonMLA improves the throughput of attention calculations in MLA architectures by up to 3x and 3.24x on NPU and GPUs, with only a 3% overhead in HBM size. 

**Abstract (ZH)**: TyphoonMLA: 结合 naive 和 absorb 公式的混合多头潜在注意力机制 

---
# Which Cultural Lens Do Models Adopt? On Cultural Positioning Bias and Agentic Mitigation in LLMs 

**Title (ZH)**: 模型采用哪种文化视角？关于大语言模型中的文化定位偏见及代理缓解机制 

**Authors**: Yixin Wan, Xingrun Chen, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21080)  

**Abstract**: Large language models (LLMs) have unlocked a wide range of downstream generative applications. However, we found that they also risk perpetuating subtle fairness issues tied to culture, positioning their generations from the perspectives of the mainstream US culture while demonstrating salient externality towards non-mainstream ones. In this work, we identify and systematically investigate this novel culture positioning bias, in which an LLM's default generative stance aligns with a mainstream view and treats other cultures as outsiders. We propose the CultureLens benchmark with 4000 generation prompts and 3 evaluation metrics for quantifying this bias through the lens of a culturally situated interview script generation task, in which an LLM is positioned as an onsite reporter interviewing local people across 10 diverse cultures. Empirical evaluation on 5 state-of-the-art LLMs reveals a stark pattern: while models adopt insider tones in over 88 percent of US-contexted scripts on average, they disproportionately adopt mainly outsider stances for less dominant cultures. To resolve these biases, we propose 2 inference-time mitigation methods: a baseline prompt-based Fairness Intervention Pillars (FIP) method, and a structured Mitigation via Fairness Agents (MFA) framework consisting of 2 pipelines: (1) MFA-SA (Single-Agent) introduces a self-reflection and rewriting loop based on fairness guidelines. (2) MFA-MA (Multi-Agent) structures the process into a hierarchy of specialized agents: a Planner Agent(initial script generation), a Critique Agent (evaluates initial script against fairness pillars), and a Refinement Agent (incorporates feedback to produce a polished, unbiased script). Empirical results showcase the effectiveness of agent-based methods as a promising direction for mitigating biases in generative LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）解锁了广泛的手动生成应用。然而，我们发现它们也存在与文化相关的微妙公平性问题，倾向于从主流美国文化的角度生成内容，对外来文化表现出明显的外部效应。在本工作中，我们识别并系统研究了这一新的文化定位偏见，即LLM的默认生成立场与主流观点一致，并将其他文化视为外来文化。我们提出了一个具有4000个生成提示和3个评估指标的CultureLens基准，通过基于文化嵌入的采访脚本生成任务来衡量这种偏见，其中LLM被定位为现场记者，采访来自10个不同文化群体的当地人。对5个最先进的LLM进行的实证评估揭示了一个明显的模式：尽管模型在超过88%的以美国为背景的脚本中采用内行人语气，但对外来文化却采用了主要的外行人立场。为了缓解这些偏见，我们提出了两种推理时间缓解方法：基于提示的基本公平干预支柱（Fairness Intervention Pillars, FIP）方法，以及包含两个管道的结构化公平代理框架（Mitigation via Fairness Agents, MFA）：MFA-SA（单代理）引入基于公平准则的自我反思和重写循环，MFA-MA（多代理）将过程结构化为专门代理的层级体系：规划代理（初始脚本生成）、批评代理（评估初始脚本以符合公平支柱）和润色代理（纳入反馈以生成一份精致且无偏见的脚本）。实证结果展示了基于代理的方法作为缓解生成LLM中偏见有希望的方向的有效性。 

---
# Communication Bias in Large Language Models: A Regulatory Perspective 

**Title (ZH)**: 大型语言模型中的communication偏见：一个监管视角 

**Authors**: Adrian Kuenzler, Stefan Schmid  

**Link**: [PDF](https://arxiv.org/pdf/2509.21075)  

**Abstract**: Large language models (LLMs) are increasingly central to many applications, raising concerns about bias, fairness, and regulatory compliance. This paper reviews risks of biased outputs and their societal impact, focusing on frameworks like the EU's AI Act and the Digital Services Act. We argue that beyond constant regulation, stronger attention to competition and design governance is needed to ensure fair, trustworthy AI. This is a preprint of the Communications of the ACM article of the same title. 

**Abstract (ZH)**: 大型语言模型（LLMs）在许多应用中变得越来越重要，引发了关于偏见、公平性和监管合规性的关注。本文回顾了偏差输出带来的风险及其社会影响，重点关注如欧盟AI法案和数字服务法案等框架。我们argue认为，除了持续的监管外，还需要更多关注竞争和设计治理，以确保公平可信的AI。 

---
# ScaleDiff: Scaling Difficult Problems for Advanced Mathematical Reasoning 

**Title (ZH)**: ScaleDiff: 扩大规模难题以促进高级数学推理 

**Authors**: Qizhi Pei, Zhuoshi Pan, Honglin Lin, Xin Gao, Yu Li, Zinan Tang, Conghui He, Rui Yan, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21070)  

**Abstract**: Large Reasoning Models (LRMs) have shown impressive capabilities in complex problem-solving, often benefiting from training on difficult mathematical problems that stimulate intricate reasoning. Recent efforts have explored automated synthesis of mathematical problems by prompting proprietary models or large-scale open-source models from seed data or inherent mathematical concepts. However, scaling up these methods remains challenging due to their high computational/API cost, complexity of prompting, and limited difficulty level of the generated problems. To overcome these limitations, we propose ScaleDiff, a simple yet effective pipeline designed to scale the creation of difficult problems. We efficiently identify difficult problems from existing datasets with only a single forward pass using an adaptive thinking model, which can perceive problem difficulty and automatically switch between "Thinking" and "NoThinking" modes. We then train a specialized difficult problem generator (DiffGen-8B) on this filtered difficult data, which can produce new difficult problems in large scale, eliminating the need for complex, per-instance prompting and its associated high API costs. Fine-tuning Qwen2.5-Math-7B-Instruct on the ScaleDiff-Math dataset yields a substantial performance increase of 11.3% compared to the original dataset and achieves a 65.9% average accuracy on AIME'24, AIME'25, HMMT-Feb'25, BRUMO'25, and MATH500, outperforming recent strong LRMs like OpenThinker3. Notably, this performance is achieved using the cost-efficient Qwen3-8B model as a teacher, demonstrating that our pipeline can effectively transfer advanced reasoning capabilities without relying on larger, more expensive teacher models. Furthermore, we observe a clear scaling phenomenon in model performance on difficult benchmarks as the quantity of difficult problems increases. Code: this https URL. 

**Abstract (ZH)**: 大规模推理模型（LRMs）在复杂问题解决方面表现出色，常常通过训练解决复杂的数学问题来激发复杂的推理能力。近期研究通过种子数据或固有的数学概念，探索了促进模型自动生成数学问题的方法。然而，由于这些方法的高计算成本、提示的复杂性以及生成的问题难度有限，规模化扩展这些方法仍然充满挑战。为克服这些限制，我们提出了一种简单而有效的流水线ScaleDiff，用于大规模生成难题。我们仅通过一次前向传递使用自适应思考模型高效地从现有数据集中识别难题，该模型能感知问题难度并自动切换“思考”和“非思考”模式。接着，我们使用过滤后的难题数据训练一个专门的难题生成器（DiffGen-8B），以大规模生成新的难题，从而消除复杂的问题实例化提示及其高昂的API成本。在ScaleDiff-Math数据集上微调Qwen2.5-Math-7B-Instruct相比原始数据集实现了11.3%的性能提升，并在AIME'24、AIME'25、HMMT-Feb'25、BRUMO'25和MATH500等评估上取得了65.9%的平均准确率，超越了近期强大的LRMs如OpenThinker3。值得注意的是，这种性能是使用成本效益高的Qwen3-8B模型作为教师实现的，证明了我们的流水线能够有效地转移高级推理能力，而无需依赖更大的、更昂贵的教师模型。此外，我们观察到，在难题数量增加时，模型在难题基准上的性能存在明显的缩放现象。代码：https://this-url.com/ 

---
# EnGraf-Net: Multiple Granularity Branch Network with Fine-Coarse Graft Grained for Classification Task 

**Title (ZH)**: EnGraf-Net：具有精细-粗粒度嫁接粒度的多粒度分支网络用于分类任务 

**Authors**: Riccardo La Grassa, Ignazio Gallo, Nicola Landro  

**Link**: [PDF](https://arxiv.org/pdf/2509.21061)  

**Abstract**: Fine-grained classification models are designed to focus on the relevant details necessary to distinguish highly similar classes, particularly when intra-class variance is high and inter-class variance is low. Most existing models rely on part annotations such as bounding boxes, part locations, or textual attributes to enhance classification performance, while others employ sophisticated techniques to automatically extract attention maps. We posit that part-based approaches, including automatic cropping methods, suffer from an incomplete representation of local features, which are fundamental for distinguishing similar objects. While fine-grained classification aims to recognize the leaves of a hierarchical structure, humans recognize objects by also forming semantic associations. In this paper, we leverage semantic associations structured as a hierarchy (taxonomy) as supervised signals within an end-to-end deep neural network model, termed EnGraf-Net. Extensive experiments on three well-known datasets CIFAR-100, CUB-200-2011, and FGVC-Aircraft demonstrate the superiority of EnGraf-Net over many existing fine-grained models, showing competitive performance with the most recent state-of-the-art approaches, without requiring cropping techniques or manual annotations. 

**Abstract (ZH)**: 基于语义关联的细粒度分类模型 

---
# GeoRef: Referring Expressions in Geometry via Task Formulation, Synthetic Supervision, and Reinforced MLLM-based Solutions 

**Title (ZH)**: GeoRef: 几何表达的的任务表述、合成监督及强化MLLM解决方案 

**Authors**: Bing Liu, Wenqiang Yv, Xuzheng Yang, Shichang Wang, Junzhuo Liu, Peng Wang, Guoqing Wang, Yang Yang, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.21050)  

**Abstract**: AI-driven geometric problem solving is a complex vision-language task that requires accurate diagram interpretation, mathematical reasoning, and robust cross-modal grounding. A foundational yet underexplored capability for this task is the ability to identify and interpret geometric elements based on natural language queries. To address this, we introduce the task of Referring Expression Comprehension (REC) for geometric problems, which evaluates whether models can localize points, shapes, and spatial relations in diagrams in response to textual prompts. We present GeoRef, a benchmark dataset constructed from existing geometric problem corpora, featuring diverse, high-quality annotations and queries. Due to the lack of annotated data for this task, we generate a large-scale synthetic training dataset using a structured geometric formal language, enabling broad coverage of geometric concepts and facilitating model adaptation. We explore two fine-tuning approaches: Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO). Our results show that GRPO significantly outperforms SFT by better aligning model behavior with task-specific rewards. Furthermore, we propose a verify-and-regenerate mechanism that detects incorrect predictions and re-infers answers using contextual reasoning history, further boosting accuracy. Notably, even state-of-the-art Multimodal Large Language Models (MLLMs) struggle with this task, underscoring the necessity of explicitly evaluating and strengthening geometric grounding as a prerequisite for robust geometric problem solving. Moreover, models trained on GeoRef demonstrate measurable improvements on downstream geometric reasoning tasks, highlighting the broader value of REC as a foundation for multimodal mathematical understanding. 

**Abstract (ZH)**: 基于AI驱动的几何问题求解是一个复杂的视觉-语言任务，要求准确的图表解释、数学推理和稳健的跨模态定位。这一任务的一个基础但尚未充分探索的能力是根据自然语言查询识别和解释几何元素的能力。为了解决这个问题，我们引入了几何问题参照表达理解（Referring Expression Comprehension, REC）任务，评估模型能否根据文本提示在图表中定位点、形状和空间关系。我们介绍了GeoRef，一个从现有几何问题语料库构建的基准数据集，包含多样且高质量的注释和查询。由于缺乏该任务的注释数据，我们使用结构化几何形式语言生成了大规模合成训练数据集，以广泛覆盖几何概念并促进模型适应。我们探索了两种微调方法：有监督微调（SFT）和组相对策略优化（GRPO）。我们的结果显示，GRPO显著优于SFT，因为它更好地使模型行为与特定任务奖励相一致。此外，我们提出了一种验证和重生成机制，用于检测错误预测并使用上下文推理历史重推答案，进一步提高了准确性。值得注意的是，即使最先进的多模态大规模语言模型（MLLMs）也难以完成这一任务，强调了明确评估和加强几何定位作为实现稳健几何问题求解的必要前提的重要性。此外，基于GeoRef训练的模型在下游几何推理任务上展现出可度量的进步，突显了REC作为多模态数学理解基础的广泛价值。 

---
# Reinforcement Learning Fine-Tuning Enhances Activation Intensity and Diversity in the Internal Circuitry of LLMs 

**Title (ZH)**: 强化学习微调增强大型语言模型内部电路的激活强度和多样性 

**Authors**: Honglin Zhang, Qianyue Hao, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.21044)  

**Abstract**: Large language models (LLMs) acquire extensive prior knowledge through large-scale pretraining and can be further enhanced via supervised fine-tuning (SFT) or reinforcement learning (RL)-based post-training. A growing body of evidence has shown that RL fine-tuning improves the capability of LLMs beyond what SFT alone achieves. However, the underlying mechanisms why RL fine-tuning is able to enhance the capability of various LLMs with distinct intrinsic characteristics remain underexplored. In this study, we draw inspiration from prior work on edge attribution patching (EAP) to investigate the internal differences of LLMs before and after RL fine-tuning. Our analysis across multiple model families shows two robust effects of online RL post-training: (i) an overall increase in activation intensity, indicating that more internal pathways are engaged and their signals become stronger, and (ii) greater diversity in activation patterns, reflected by higher entropy and less concentrated edge distributions. These changes suggest that RL reshapes information flow to be both more redundant and more flexible, which may explain its advantage in generalization. Notably, models fine-tuned with Direct Preference Optimization (DPO) deviate from these trends, exhibiting substantially weaker or inconsistent internal changes compared to PPO- and GRPO-based training. Together, our findings provide a unified view of how RL fine-tuning systematically alters the internal circuitry of LLMs and highlight the methodological distinctions between online RL and preference-based approaches. Our code is open source at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过大规模预训练获得丰富的先验知识，并可通过监督微调（SFT）或基于强化学习（RL）的后训练进一步提升。越来越多的证据表明，基于RL的微调能够超越单纯的SFT，提升LLMs的能力。然而，为什么基于RL的微调能够增强具有不同固有特征的各种LLMs的能力，其背后的机制仍缺乏深入探索。在本研究中，我们借鉴边属性补丁（EAP）的相关工作，探讨RL后训练前后LLMs内部差异。在多种模型家族的分析中，我们发现在线RL后训练的两个稳健效应：（i）激活强度总体增加，表明更多内部路径被激活且信号强度增强；（ii）激活模式的多样性增加，表现为熵值更高且边分布更不集中。这些变化表明，RL重新塑造了信息流，使其既更具冗余性又更具灵活性，这可能解释了其在泛化能力上的优势。值得注意的是，使用直接偏好优化（DPO）微调的模型偏离了这些趋势，显示出与PPO-和GRPO-为基础的训练相比，内部变化显著较弱或不一致。我们的研究结果提供了RL后训练系统地改变LLMs内部电路的统一视角，并突出了在线RL与偏好优化方法在方法论上的区别。我们的代码在此公开。 

---
# Generative AI for FFRDCs 

**Title (ZH)**: 生成式AI在FFRDCs中的应用 

**Authors**: Arun S. Maiya  

**Link**: [PDF](https://arxiv.org/pdf/2509.21040)  

**Abstract**: Federally funded research and development centers (FFRDCs) face text-heavy workloads, from policy documents to scientific and engineering papers, that are slow to analyze manually. We show how large language models can accelerate summarization, classification, extraction, and sense-making with only a few input-output examples. To enable use in sensitive government contexts, we apply OnPrem$.$LLM, an open-source framework for secure and flexible application of generative AI. Case studies on defense policy documents and scientific corpora, including the National Defense Authorization Act (NDAA) and National Science Foundation (NSF) Awards, demonstrate how this approach enhances oversight and strategic analysis while maintaining auditability and data sovereignty. 

**Abstract (ZH)**: 联邦资助的研究与发展中心（FFRDCs）面临大量文本密集型工作负载，从政策文件到科学与工程论文，手工分析速度较慢。我们展示了大规模语言模型如何仅通过少量输入-输出示例就能加速摘要、分类、提取和意义构建。为了在敏感的政府背景下应用，我们采用了OnPrem$.$LLM这一开源框架，以确保生成式AI的安全和灵活应用。案例研究涉及国防政策文件和科学语料库，包括国防授权法案（NDAA）和国家科学基金会（NSF）奖助金，证明了该方法如何在确保审计性和数据主权的同时增强监督和战略分析。 

---
# SupCLAP: Controlling Optimization Trajectory Drift in Audio-Text Contrastive Learning with Support Vector Regularization 

**Title (ZH)**: SupCLAP: 用支持向量正则化控制音频-文本对比学习中优化轨迹偏移 

**Authors**: Jiehui Luo, Yuguo Yin, Yuxin Xie, Jinghan Ru, Xianwei Zhuang, Minghua He, Aofan Liu, Zihan Xiong, Dongchao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21033)  

**Abstract**: Contrastive language-audio pretraining, which aims to unify multimodal representations in a shared embedding space, serves as a cornerstone for building a wide range of applications, from cross-modal retrieval to cutting-edge multimodal large language models. However, we find that the perpendicular component of the pushing force from negative samples in contrastive learning is a double-edged sword: it contains rich supplementary information from negative samples, yet its unconstrained nature causes optimization trajectory drift and training instability. To address this, we propose Support Vector Regularization (SVR), a method that introduces an auxiliary support vector to control this perpendicular component, aiming to harness its rich information while mitigating the associated trajectory drift. The efficacy of SVR is critically governed by its semantic radius, for which we explore two unsupervised modeling strategies: direct parameterization and an adaptive radius predictor module enhanced with constraints to improve its predicting accuracy. Extensive experimental results demonstrate that our method surpasses widely used baselines like InfoNCE and SigLIP loss across classification, monolingual retrieval, and multilingual retrieval on standard audio-text datasets. Both the theoretical analysis and the experimental results on optimizing trajectory drift validate the correctness and effectiveness of our SVR method. 

**Abstract (ZH)**: 对比语言-音频预训练，旨在在一个共享嵌入空间中统一多模态表示，是构建从跨模态检索到先进的多模态大型语言模型等一系列应用的基础。然而，我们发现对比学习中负样本推力的垂直分量是一把双刃剑：它包含丰富的补充信息，但其无约束性质导致了优化轨迹漂移和训练不稳定性。为解决这一问题，我们提出了一种支持向量正则化（SVR）方法，该方法引入了一个辅助支持向量以控制该垂直分量，旨在利用其丰富的信息同时减轻相关的轨迹漂移。SVR的有效性主要受其语义半径的调控，为此我们探索了两种无监督建模策略：直接参数化以及增强有约束条件的自适应半径预测模块以提高预测准确性。广泛的实验结果表明，我们的方法在标准音频-文本数据集的分类、单语检索和多语检索任务中优于广泛应用的Baseline如InfoNCE和SigLIP损失。理论分析和优化轨迹漂移的实验结果验证了SVR方法的正确性和有效性。 

---
# Efficient Ensemble Conditional Independence Test Framework for Causal Discovery 

**Title (ZH)**: 高效的集成条件独立性检验框架用于因果发现 

**Authors**: Zhengkang Guan, Kun Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21021)  

**Abstract**: Constraint-based causal discovery relies on numerous conditional independence tests (CITs), but its practical applicability is severely constrained by the prohibitive computational cost, especially as CITs themselves have high time complexity with respect to the sample size. To address this key bottleneck, we introduce the Ensemble Conditional Independence Test (E-CIT), a general and plug-and-play framework. E-CIT operates on an intuitive divide-and-aggregate strategy: it partitions the data into subsets, applies a given base CIT independently to each subset, and aggregates the resulting p-values using a novel method grounded in the properties of stable distributions. This framework reduces the computational complexity of a base CIT to linear in the sample size when the subset size is fixed. Moreover, our tailored p-value combination method offers theoretical consistency guarantees under mild conditions on the subtests. Experimental results demonstrate that E-CIT not only significantly reduces the computational burden of CITs and causal discovery but also achieves competitive performance. Notably, it exhibits an improvement in complex testing scenarios, particularly on real-world datasets. 

**Abstract (ZH)**: 基于约束的因果发现依赖于大量的条件独立性检验（CITs），但由于CITs本身的时间复杂性高，加上计算成本高昂，其实际应用受到严重限制。为解决这一关键瓶颈，我们引入了集成条件独立性检验（E-CIT）框架，这是一种通用且即插即用的框架。E-CIT采用直观的分而治之策略：将数据集划分为子集，独立地在每个子集上应用给定的基础CIT，并通过一种新的结合p值的方法将结果整合起来，该方法基于稳定分布的性质。该框架在子集大小固定的情况下将基础CIT的计算复杂度线性地减少到样本量的线性复杂度。此外，我们设计的p值组合方法在轻条件下提供了理论上的一致性保证。实验结果表明，E-CIT不仅显著降低了CITs和因果发现的计算负担，还实现了竞争力的性能。特别地，在复杂测试场景和现实世界数据集上表现出色。 

---
# The Use of the Simplex Architecture to Enhance Safety in Deep-Learning-Powered Autonomous Systems 

**Title (ZH)**: Simplex架构在增强基于深度学习的自主系统安全性中的应用 

**Authors**: Federico Nesti, Niko Salamini, Mauro Marinoni, Giorgio Maria Cicero, Gabriele Serra, Alessandro Biondi, Giorgio Buttazzo  

**Link**: [PDF](https://arxiv.org/pdf/2509.21014)  

**Abstract**: Recently, the outstanding performance reached by neural networks in many tasks has led to their deployment in autonomous systems, such as robots and vehicles. However, neural networks are not yet trustworthy, being prone to different types of misbehavior, such as anomalous samples, distribution shifts, adversarial attacks, and other threats. Furthermore, frameworks for accelerating the inference of neural networks typically run on rich operating systems that are less predictable in terms of timing behavior and present larger surfaces for cyber-attacks.
To address these issues, this paper presents a software architecture for enhancing safety, security, and predictability levels of learning-based autonomous systems. It leverages two isolated execution domains, one dedicated to the execution of neural networks under a rich operating system, which is deemed not trustworthy, and one responsible for running safety-critical functions, possibly under a different operating system capable of handling real-time constraints.
Both domains are hosted on the same computing platform and isolated through a type-1 real-time hypervisor enabling fast and predictable inter-domain communication to exchange real-time data. The two domains cooperate to provide a fail-safe mechanism based on a safety monitor, which oversees the state of the system and switches to a simpler but safer backup module, hosted in the safety-critical domain, whenever its behavior is considered untrustworthy.
The effectiveness of the proposed architecture is illustrated by a set of experiments performed on two control systems: a Furuta pendulum and a rover. The results confirm the utility of the fall-back mechanism in preventing faults due to the learning component. 

**Abstract (ZH)**: 增强基于学习的自主系统安全性、安全性和可预测性的软件架构 

---
# Predicting LLM Reasoning Performance with Small Proxy Model 

**Title (ZH)**: 使用小型代理模型预测LLM推理性能 

**Authors**: Woosung Koh, Juyoung Suk, Sungjun Han, Se-Young Yun, Jay Shin  

**Link**: [PDF](https://arxiv.org/pdf/2509.21013)  

**Abstract**: Given the prohibitive cost of pre-training large language models, it is essential to leverage smaller proxy models to optimize datasets before scaling up. However, this approach becomes challenging for reasoning capabilities, which exhibit emergent behavior that only appear reliably at larger model sizes, often exceeding 7B parameters. To address this, we introduce rBridge, showing that small proxies ($\leq$1B) can effectively predict large-model reasoning by aligning more closely with (1) the pre-training objective and (2) the target task. rBridge achieves this by weighting negative log-likelihood with task alignment, using reasoning traces from frontier models as gold labels. In our experiments, rBridge (i) reduces dataset ranking costs by over 100x relative to the best baseline, (ii) achieves the strongest correlation across six reasoning benchmarks at 1B to 32B scale, and (iii) zero-shot transfers predictive relationships across pre-training datasets at 1B to 7B scale. These findings indicate that rBridge offers a practical path for exploring reasoning-oriented pre-training at lower cost. 

**Abstract (ZH)**: 给定预训练大语言模型的成本 prohibitive，利用较小的代理模型优化数据集再扩展规模是至关重要的。然而，这种方法对于推理能力来说变得具有挑战性，因为这些能力在较大的模型规模（通常超过7B参数）下才会可靠地出现。为解决这一问题，我们引入了rBridge，表明较小的代理模型（≤1B）可以通过更紧密地与（1）预训练目标和（2）目标任务对齐来有效地预测大型模型的推理。rBridge 通过使用前沿模型的推理轨迹作为黄金标签，对负对数似然度进行加权以实现这一目标。在我们的实验中，rBridge （i）将数据集排序成本相对于最佳基线降低了超过100倍，（ii）在1B到32B规模的六个推理基准测试中实现了最强的相关性，（iii）在1B到7B规模的预训练数据集之间实现了零样本的知识迁移。这些发现表明，rBridge 提供了一条在较低成本下探索面向推理的预训练的实际路径。 

---
# Mechanism of Task-oriented Information Removal in In-context Learning 

**Title (ZH)**: 任务导向的信息移除机制在上下文学习中的作用 

**Authors**: Hakaze Cho, Haolin Yang, Gouki Minegishi, Naoya Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2509.21012)  

**Abstract**: In-context Learning (ICL) is an emerging few-shot learning paradigm based on modern Language Models (LMs), yet its inner mechanism remains unclear. In this paper, we investigate the mechanism through a novel perspective of information removal. Specifically, we demonstrate that in the zero-shot scenario, LMs encode queries into non-selective representations in hidden states containing information for all possible tasks, leading to arbitrary outputs without focusing on the intended task, resulting in near-zero accuracy. Meanwhile, we find that selectively removing specific information from hidden states by a low-rank filter effectively steers LMs toward the intended task. Building on these findings, by measuring the hidden states on carefully designed metrics, we observe that few-shot ICL effectively simulates such task-oriented information removal processes, selectively removing the redundant information from entangled non-selective representations, and improving the output based on the demonstrations, which constitutes a key mechanism underlying ICL. Moreover, we identify essential attention heads inducing the removal operation, termed Denoising Heads, which enables the ablation experiments blocking the information removal operation from the inference, where the ICL accuracy significantly degrades, especially when the correct label is absent from the few-shot demonstrations, confirming both the critical role of the information removal mechanism and denoising heads. 

**Abstract (ZH)**: 基于信息删除视角的上下文学习机制探究 

---
# Automatic Red Teaming LLM-based Agents with Model Context Protocol Tools 

**Title (ZH)**: 基于模型上下文协议工具的自动红队LLM代理训练 

**Authors**: Ping He, Changjiang Li, Binbin Zhao, Tianyu Du, Shouling Ji  

**Link**: [PDF](https://arxiv.org/pdf/2509.21011)  

**Abstract**: The remarkable capability of large language models (LLMs) has led to the wide application of LLM-based agents in various domains. To standardize interactions between LLM-based agents and their environments, model context protocol (MCP) tools have become the de facto standard and are now widely integrated into these agents. However, the incorporation of MCP tools introduces the risk of tool poisoning attacks, which can manipulate the behavior of LLM-based agents. Although previous studies have identified such vulnerabilities, their red teaming approaches have largely remained at the proof-of-concept stage, leaving the automatic and systematic red teaming of LLM-based agents under the MCP tool poisoning paradigm an open question. To bridge this gap, we propose AutoMalTool, an automated red teaming framework for LLM-based agents by generating malicious MCP tools. Our extensive evaluation shows that AutoMalTool effectively generates malicious MCP tools capable of manipulating the behavior of mainstream LLM-based agents while evading current detection mechanisms, thereby revealing new security risks in these agents. 

**Abstract (ZH)**: 自动生成恶意MCP工具的自动红队框架：LLM基于代理下的工具中毒攻击研究 

---
# ExMolRL: Phenotype-Target Joint Generation of De Novo Molecules via Multi-Objective Reinforcement Learning 

**Title (ZH)**: ExMolRL：基于多目标强化学习的表型-目标联合新分子生成 

**Authors**: Haotian Guo, Hui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21010)  

**Abstract**: The generation of high-quality candidate molecules remains a central challenge in AI-driven drug design. Current phenotype-based and target-based strategies each suffer limitations, either incurring high experimental costs or overlook system-level cellular responses. To bridge this gap, we propose ExMoIRL, a novel generative framework that synergistically integrates phenotypic and target-specific cues for de novo molecular generation. The phenotype-guided generator is first pretrained on expansive drug-induced transcriptional profiles and subsequently fine-tuned via multi-objective reinforcement learning (RL). Crucially, the reward function fuses docking affinity and drug-likeness scores, augmented with ranking loss, prior-likelihood regularization, and entropy maximization. The multi-objective RL steers the model toward chemotypes that are simultaneously potent, diverse, and aligned with the specified phenotypic effects. Extensive experiments demonstrate ExMoIRL's superior performance over state-of-the-art phenotype-based and target-based models across multiple well-characterized targets. Our generated molecules exhibit favorable drug-like properties, high target affinity, and inhibitory potency (IC50) against cancer cells. This unified framework showcases the synergistic potential of combining phenotype-guided and target-aware strategies, offering a more effective solution for de novo drug discovery. 

**Abstract (ZH)**: 基于表型和目标导向的高效分子生成仍是在AI驱动的药物设计中的一项核心挑战。当前基于表型和基于靶点的方法各自存在局限性，要么导致高实验成本，要么忽略细胞系统的整体反应。为弥补这一差距，我们提出了一种名为ExMoIRL的新型生成框架，该框架能协同整合表型和目标特异性线索以进行从头分子生成。表型导向的生成器首先在广泛的药物诱导转录谱上进行预训练，随后通过多目标强化学习进行微调。最关键的是，奖励函数结合了结合亲和力评分和类药性评分，并加入了排序损失、先验似然正则化和熵最大化。多目标强化学习引导模型趋向于同时高效、多样化且与指定表型效应相一致的化学类型。广泛的经验表明，ExMoIRL在多个已充分表征的靶点上优于最先进的基于表型和基于靶点的模型。我们生成的分子表现出有利的类药性质、高度的目标亲和力和对癌细胞的抑制效力（IC50）。该统一框架展示了将表型导向与目标意识策略相结合的协同潜力，为从头药物发现提供了更有效的解决方案。 

---
# Marching Neurons: Accurate Surface Extraction for Neural Implicit Shapes 

**Title (ZH)**: 行进神经元：神经隐式形状的准确表面提取 

**Authors**: Christian Stippel, Felix Mujkanovic, Thomas Leimkühler, Pedro Hermosilla  

**Link**: [PDF](https://arxiv.org/pdf/2509.21007)  

**Abstract**: Accurate surface geometry representation is crucial in 3D visual computing. Explicit representations, such as polygonal meshes, and implicit representations, like signed distance functions, each have distinct advantages, making efficient conversions between them increasingly important. Conventional surface extraction methods for implicit representations, such as the widely used Marching Cubes algorithm, rely on spatial decomposition and sampling, leading to inaccuracies due to fixed and limited resolution. We introduce a novel approach for analytically extracting surfaces from neural implicit functions. Our method operates natively in parallel and can navigate large neural architectures. By leveraging the fact that each neuron partitions the domain, we develop a depth-first traversal strategy to efficiently track the encoded surface. The resulting meshes faithfully capture the full geometric information from the network without ad-hoc spatial discretization, achieving unprecedented accuracy across diverse shapes and network architectures while maintaining competitive speed. 

**Abstract (ZH)**: 准确的表面几何表示对于三维视觉计算至关重要。显式表示，如多边形网格，和隐式表示，如有符号距离函数，各自具有独特的优点，使得它们之间的高效转换日益重要。传统的用于隐式表示的表面提取方法，如广泛使用的Marching Cubes算法，依赖于空间分解和采样，由于固定和有限的分辨率导致不准确。我们提出了一种新的方法，用于从神经隐式函数中分析地提取表面。该方法可以本原地并行操作，并能导航大型神经架构。通过利用每个神经元划分域的事实，我们开发了一种深度优先遍历策略，以高效地追踪编码的表面。生成的网格能够忠实捕捉网络中的完整几何信息，而无需人为的空间离散化，从而在多样化的形状和网络架构上实现了前所未有的精度，同时保持了竞争力的速度。 

---
# AnywhereVLA: Language-Conditioned Exploration and Mobile Manipulation 

**Title (ZH)**: AnywhereVLA：语言条件化的探索与移动操作 

**Authors**: Konstantin Gubernatorov, Artem Voronov, Roman Voronov, Sergei Pasynkov, Stepan Perminov, Ziang Guo, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2509.21006)  

**Abstract**: We address natural language pick-and-place in unseen, unpredictable indoor environments with AnywhereVLA, a modular framework for mobile manipulation. A user text prompt serves as an entry point and is parsed into a structured task graph that conditions classical SLAM with LiDAR and cameras, metric semantic mapping, and a task-aware frontier exploration policy. An approach planner then selects visibility and reachability aware pre grasp base poses. For interaction, a compact SmolVLA manipulation head is fine tuned on platform pick and place trajectories for the SO-101 by TheRobotStudio, grounding local visual context and sub-goals into grasp and place proposals. The full system runs fully onboard on consumer-level hardware, with Jetson Orin NX for perception and VLA and an Intel NUC for SLAM, exploration, and control, sustaining real-time operation. We evaluated AnywhereVLA in a multi-room lab under static scenes and normal human motion. In this setting, the system achieves a $46\%$ overall task success rate while maintaining throughput on embedded compute. By combining a classical stack with a fine-tuned VLA manipulation, the system inherits the reliability of geometry-based navigation with the agility and task generalization of language-conditioned manipulation. 

**Abstract (ZH)**: AnywhereVLA：一种适用于未见和不可预测室内环境的移动操作自然语言pick-and-place框架 

---
# Lossless Compression: A New Benchmark for Time Series Model Evaluation 

**Title (ZH)**: 无损压缩：时间序列模型评估的新基准 

**Authors**: Meng Wan, Benxi Tian, Jue Wang, Cui Hui, Ningming Nie, Tiantian Liu, Zongguo Wang, Cao Rongqiang, Peng Shi, Yangang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21002)  

**Abstract**: The evaluation of time series models has traditionally focused on four canonical tasks: forecasting, imputation, anomaly detection, and classification. While these tasks have driven significant progress, they primarily assess task-specific performance and do not rigorously measure whether a model captures the full generative distribution of the data. We introduce lossless compression as a new paradigm for evaluating time series models, grounded in Shannon's source coding theorem. This perspective establishes a direct equivalence between optimal compression length and the negative log-likelihood, providing a strict and unified information-theoretic criterion for modeling capacity. Then We define a standardized evaluation protocol and metrics. We further propose and open-source a comprehensive evaluation framework TSCom-Bench, which enables the rapid adaptation of time series models as backbones for lossless compression. Experiments across diverse datasets on state-of-the-art models, including TimeXer, iTransformer, and PatchTST, demonstrate that compression reveals distributional weaknesses overlooked by classic benchmarks. These findings position lossless compression as a principled task that complements and extends existing evaluation for time series modeling. 

**Abstract (ZH)**: 基于无损压缩的时间序列模型评估 

---
# Binary Autoencoder for Mechanistic Interpretability of Large Language Models 

**Title (ZH)**: 二进制自动编码器用于大型语言模型的机理可解释性 

**Authors**: Hakaze Cho, Haolin Yang, Brian M. Kurkoski, Naoya Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2509.20997)  

**Abstract**: Existing works are dedicated to untangling atomized numerical components (features) from the hidden states of Large Language Models (LLMs) for interpreting their mechanism. However, they typically rely on autoencoders constrained by some implicit training-time regularization on single training instances (i.e., $L_1$ normalization, top-k function, etc.), without an explicit guarantee of global sparsity among instances, causing a large amount of dense (simultaneously inactive) features, harming the feature sparsity and atomization. In this paper, we propose a novel autoencoder variant that enforces minimal entropy on minibatches of hidden activations, thereby promoting feature independence and sparsity across instances. For efficient entropy calculation, we discretize the hidden activations to 1-bit via a step function and apply gradient estimation to enable backpropagation, so that we term it as Binary Autoencoder (BAE) and empirically demonstrate two major applications: (1) Feature set entropy calculation. Entropy can be reliably estimated on binary hidden activations, which we empirically evaluate and leverage to characterize the inference dynamics of LLMs and In-context Learning. (2) Feature untangling. Similar to typical methods, BAE can extract atomized features from LLM's hidden states. To robustly evaluate such feature extraction capability, we refine traditional feature-interpretation methods to avoid unreliable handling of numerical tokens, and show that BAE avoids dense features while producing the largest number of interpretable ones among baselines, which confirms the effectiveness of BAE serving as a feature extractor. 

**Abstract (ZH)**: 现有的工作致力于从大型语言模型（LLMs）的隐藏状态下解开原子化的数值成分（特征），以解释其机制。然而，它们通常依赖于受限于某些隐式训练时间正则化的自动编码器（即$L_1$归一化、top-k函数等），而不提供实例间全局稀疏性的显式保证，导致大量密集的（同时不活跃）特征，损害了特征稀疏性和原子化。本文提出了一种新的自动编码器变体，强制对小批量隐藏激活的最小熵，从而促进实例间的特征独立性和稀疏性。为了高效地计算熵，我们通过步骤函数将隐藏激活离散化为1位，并应用梯度估计以实现反向传播，因此我们称之为二元自动编码器（BAE），并从两个主要应用中实证演示其效果：（1）特征集熵计算。二元隐藏激活上的熵可以可靠地估计，我们实证评估并利用其来表征LLMs和条件上下文学习的推理动态。（2）特征解缠。类似于传统方法，BAE可以从LLM的隐藏状态下提取原子化的特征。为了稳健地评估这种特征提取能力，我们改进了传统的特征解释方法以避免对数值标记的不可靠处理，并展示了BAE在生成可解释特征的数量上优于基线，从而验证了BAE作为特征提取器的有效性。 

---
# Fast-SEnSeI: Lightweight Sensor-Independent Cloud Masking for On-board Multispectral Sensors 

**Title (ZH)**: Fast-SEnSeI: 轻量级传感器独立云遮盖算法用于机载多光谱传感器 

**Authors**: Jan Kněžík, Jonáš Herec, Rado Pitoňák  

**Link**: [PDF](https://arxiv.org/pdf/2509.20991)  

**Abstract**: Cloud segmentation is a critical preprocessing step for many Earth observation tasks, yet most models are tightly coupled to specific sensor configurations and rely on ground-based processing. In this work, we propose Fast-SEnSeI, a lightweight, sensor-independent encoder module that enables flexible, on-board cloud segmentation across multispectral sensors with varying band configurations. Building upon SEnSeI-v2, Fast-SEnSeI integrates an improved spectral descriptor, lightweight architecture, and robust padding-band handling. It accepts arbitrary combinations of spectral bands and their wavelengths, producing fixed-size feature maps that feed into a compact, quantized segmentation model based on a modified U-Net. The module runs efficiently on embedded CPUs using Apache TVM, while the segmentation model is deployed on FPGA, forming a CPU-FPGA hybrid pipeline suitable for space-qualified hardware. Evaluations on Sentinel-2 and Landsat 8 datasets demonstrate accurate segmentation across diverse input configurations. 

**Abstract (ZH)**: 基于SEnSeI-v2的Fast-SEnSeI：一种轻量级、传感器无关的编码器模块，用于多光谱传感器的灵活机载云分割 

---
# Rejuvenating Cross-Entropy Loss in Knowledge Distillation for Recommender Systems 

**Title (ZH)**: 在知识蒸馏推荐系统中 rejuvenate 跨熵损失函数 

**Authors**: Zhangchi Zhu, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20989)  

**Abstract**: This paper analyzes Cross-Entropy (CE) loss in knowledge distillation (KD) for recommender systems. KD for recommender systems targets at distilling rankings, especially among items most likely to be preferred, and can only be computed on a small subset of items. Considering these features, we reveal the connection between CE loss and NDCG in the field of KD. We prove that when performing KD on an item subset, minimizing CE loss maximizes the lower bound of NDCG, only if an assumption of closure is satisfied. It requires that the item subset consists of the student's top items. However, this contradicts our goal of distilling rankings of the teacher's top items. We empirically demonstrate the vast gap between these two kinds of top items. To bridge the gap between our goal and theoretical support, we propose Rejuvenated Cross-Entropy for Knowledge Distillation (RCE-KD). It splits the top items given by the teacher into two subsets based on whether they are highly ranked by the student. For the subset that defies the condition, a sampling strategy is devised to use teacher-student collaboration to approximate our assumption of closure. We also combine the losses on the two subsets adaptively. Extensive experiments demonstrate the effectiveness of our method. Our code is available at this https URL. 

**Abstract (ZH)**: 这篇论文分析了知识蒸馏中交叉熵损失在推荐系统中的应用，特别是在推荐系统中通过知识蒸馏提取排名，尤其是最有可能被偏好项的排名，并且只能在一小部分项上进行计算。基于这些特征，我们揭示了交叉熵损失与知识蒸馏领域中NDCG之间的联系。我们证明，在满足闭包假设的前提下，对项子集进行知识蒸馏时，最小化交叉熵损失等同于最大化NDCG的下界。然而，这与我们目标中提取教师最偏好项的排名相矛盾。我们通过实验展示了这两种最偏好项之间的巨大差距。为了弥合目标与理论支持之间的差距，我们提出了再生交叉熵知识蒸馏（RCE-KD）。该方法根据学生对教师提供的最偏好项的排名情况将其划分为两个子集，并为不符合条件的子集设计采样策略，利用教师和学生之间的合作来近似闭包假设。我们还对两个子集的损失进行了适应性组合。广泛的实验表明了本方法的有效性。我们的代码可在此处访问：this https URL。 

---
# SiNGER: A Clearer Voice Distills Vision Transformers Further 

**Title (ZH)**: SiNGER: 更清晰的声音提炼视觉Transformer 

**Authors**: Geunhyeok Yu, Sunjae Jeong, Yoonyoung Choi, Jaeseung Kim, Hyoseok Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20986)  

**Abstract**: Vision Transformers are widely adopted as the backbone of vision foundation models, but they are known to produce high-norm artifacts that degrade representation quality. When knowledge distillation transfers these features to students, high-norm artifacts dominate the objective, so students overfit to artifacts and underweight informative signals, diminishing the gains from larger models. Prior work attempted to remove artifacts but encountered an inherent trade-off between artifact suppression and preserving informative signals from teachers. To address this, we introduce Singular Nullspace-Guided Energy Reallocation (SiNGER), a novel distillation framework that suppresses artifacts while preserving informative signals. The key idea is principled teacher feature refinement: during refinement, we leverage the nullspace-guided perturbation to preserve information while suppressing artifacts. Then, the refined teacher's features are distilled to a student. We implement this perturbation efficiently with a LoRA-based adapter that requires minimal structural modification. Extensive experiments show that \oursname consistently improves student models, achieving state-of-the-art performance in multiple downstream tasks and producing clearer and more interpretable representations. 

**Abstract (ZH)**: Vision Transformers广泛应用于视觉基础模型的主干网络，但已知会产生高范数的伪像，损害表示质量。当知识蒸馏将这些特征传递给学生时，高范数的伪像主导了目标函数，导致学生过度拟合伪像而忽视了有益信号，从而削弱了大模型带来的增益。先前的工作试图去除伪像，但遇到了去除伪像与保留教师有益信号之间的固有权衡。为解决这一问题，我们引入了Singular Nullspace-Guided Energy Reallocation（SiNGER），一种新颖的蒸馏框架，能够在去除伪像的同时保留有益信号。关键思想是在细化过程中利用nullspace引导的扰动来保留信息同时抑制伪像。然后将细化后的教师特征传递给学生。我们通过基于LoRA的适配器高效实现这一扰动，仅需进行最少的结构修改。广泛实验表明，该方法能够一致地提升学生模型，在多个下游任务中实现最先进的性能，并生成更清晰和更具可解释性的表示。 

---
# Analysis of instruction-based LLMs' capabilities to score and judge text-input problems in an academic setting 

**Title (ZH)**: 基于指令的大型语言模型在学术环境中评估和评判文本输入问题的能力分析 

**Authors**: Valeria Ramirez-Garcia, David de-Fitero-Dominguez, Antonio Garcia-Cabot, Eva Garcia-Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2509.20982)  

**Abstract**: Large language models (LLMs) can act as evaluators, a role studied by methods like LLM-as-a-Judge and fine-tuned judging LLMs. In the field of education, LLMs have been studied as assistant tools for students and teachers. Our research investigates LLM-driven automatic evaluation systems for academic Text-Input Problems using rubrics. We propose five evaluation systems that have been tested on a custom dataset of 110 answers about computer science from higher education students with three models: JudgeLM, Llama-3.1-8B and DeepSeek-R1-Distill-Llama-8B. The evaluation systems include: The JudgeLM evaluation, which uses the model's single answer prompt to obtain a score; Reference Aided Evaluation, which uses a correct answer as a guide aside from the original context of the question; No Reference Evaluation, which ommits the reference answer; Additive Evaluation, which uses atomic criteria; and Adaptive Evaluation, which is an evaluation done with generated criteria fitted to each question. All evaluation methods have been compared with the results of a human evaluator. Results show that the best method to automatically evaluate and score Text-Input Problems using LLMs is Reference Aided Evaluation. With the lowest median absolute deviation (0.945) and the lowest root mean square deviation (1.214) when compared to human evaluation, Reference Aided Evaluation offers fair scoring as well as insightful and complete evaluations. Other methods such as Additive and Adaptive Evaluation fail to provide good results in concise answers, No Reference Evaluation lacks information needed to correctly assess questions and JudgeLM Evaluations have not provided good results due to the model's limitations. As a result, we conclude that Artificial Intelligence-driven automatic evaluation systems, aided with proper methodologies, show potential to work as complementary tools to other academic resources. 

**Abstract (ZH)**: 大规模语言模型LLMs可以作为评估者，这一角色通过LLM-as-a-Judge等方法进行研究。在教育领域，LLMs被研究作为学生和教师的辅助工具。我们的研究探讨了基于LLM的自动评分系统在使用评分标准评价学术填空问题方面的方法。我们提出并测试了五个评分系统，这些系统基于一个包含110个来自高等教育学生关于计算机科学答案的自定义数据集，使用了三种模型：JudgeLM、Llama-3.1-8B和DeepSeek-R1-Distill-Llama-8B。评分系统包括：使用模型单个答案提示进行评分的JudgeLM评分；使用正确答案作为辅助参考的参考辅助评分；不使用参考答案的无参考评分；采用原子性标准的加性评分；以及适应评分，它是根据每个问题生成的适应性标准进行的评分。所有评分方法都与人工评分结果进行了比较。结果表明，在使用LLMs自动评分填空问题方面，参考辅助评分是最佳方法。与人工评分相比，参考辅助评分具有最低的中值绝对偏差（0.945）和最低的均方根偏差（1.214），提供了公平的评分以及深入和完整的评价。其他方法如加性评分和适应性评分在简洁答案上表现不佳，无参考评分缺乏正确评估问题所需的信息，而JudgeLM评分由于模型的限制未能提供良好的结果。因此，我们得出结论，适当方法驱动的人工智能自动评分系统具有作为其他学术资源补充工具的潜力。 

---
# FracAug: Fractional Augmentation boost Graph-level Anomaly Detection under Limited Supervision 

**Title (ZH)**: FracAug: 分数增强促进有限监督下的图级异常检测 

**Authors**: Xiangyu Dong, Xingyi Zhang, Sibo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20978)  

**Abstract**: Graph-level anomaly detection (GAD) is critical in diverse domains such as drug discovery, yet high labeling costs and dataset imbalance hamper the performance of Graph Neural Networks (GNNs). To address these issues, we propose FracAug, an innovative plug-in augmentation framework that enhances GNNs by generating semantically consistent graph variants and pseudo-labeling with mutual verification. Unlike previous heuristic methods, FracAug learns semantics within given graphs and synthesizes fractional variants, guided by a novel weighted distance-aware margin loss. This captures multi-scale topology to generate diverse, semantic-preserving graphs unaffected by data imbalance. Then, FracAug utilizes predictions from both original and augmented graphs to pseudo-label unlabeled data, iteratively expanding the training set. As a model-agnostic module compatible with various GNNs, FracAug demonstrates remarkable universality and efficacy: experiments across 14 GNNs on 12 real-world datasets show consistent gains, boosting average AUROC, AUPRC, and F1-score by up to 5.72%, 7.23%, and 4.18%, respectively. 

**Abstract (ZH)**: 图级别异常检测（GAD）在药物发现等众多领域至关重要，但由于标注成本高和数据集不平衡，图神经网络（GNN）的性能受到限制。为解决这些问题，我们提出了一种创新的插件增强框架FracAug，通过生成语义一致的图变体和通过互验证生成伪标签来增强GNN。FracAug在给定的图中学习语义，并根据一种新颖的加权距离感知边际损失合成分数变体，从而捕捉多尺度拓扑结构，生成多样且语义保留的图，不受数据不平衡的影响。然后，FracAug利用原始图和增强图的预测结果对未标注数据进行伪标签，并迭代扩展训练集。作为一种与不同GNN兼容的模型通用模块，FracAug展现了显著的通用性和有效性：在14种GNN模型上对12个真实世界数据集进行的实验结果表明，平均提升AUROC、AUPRC和F1分数分别高达5.72%、7.23%和4.18%。 

---
# Knowledgeable Language Models as Black-Box Optimizers for Personalized Medicine 

**Title (ZH)**: 具有知识的语言模型作为黑盒优化器用于个性化医学 

**Authors**: Michael S. Yao, Osbert Bastani, Alma Andersson, Tommaso Biancalani, Aïcha Bentaieb, Claudia Iriondo  

**Link**: [PDF](https://arxiv.org/pdf/2509.20975)  

**Abstract**: The goal of personalized medicine is to discover a treatment regimen that optimizes a patient's clinical outcome based on their personal genetic and environmental factors. However, candidate treatments cannot be arbitrarily administered to the patient to assess their efficacy; we often instead have access to an in silico surrogate model that approximates the true fitness of a proposed treatment. Unfortunately, such surrogate models have been shown to fail to generalize to previously unseen patient-treatment combinations. We hypothesize that domain-specific prior knowledge - such as medical textbooks and biomedical knowledge graphs - can provide a meaningful alternative signal of the fitness of proposed treatments. To this end, we introduce LLM-based Entropy-guided Optimization with kNowledgeable priors (LEON), a mathematically principled approach to leverage large language models (LLMs) as black-box optimizers without any task-specific fine-tuning, taking advantage of their ability to contextualize unstructured domain knowledge to propose personalized treatment plans in natural language. In practice, we implement LEON via 'optimization by prompting,' which uses LLMs as stochastic engines for proposing treatment designs. Experiments on real-world optimization tasks show LEON outperforms both traditional and LLM-based methods in proposing individualized treatments for patients. 

**Abstract (ZH)**: 个性化医疗的目标是根据患者的个人遗传和环境因素发现一种优化临床结果的治疗方案。然而，候选治疗方法不能随意给予患者以评估其疗效；我们通常只能访问一个计算模拟的 surrogate 模型来近似估计所提议治疗的真实适应度。不幸的是，此类 surrogate 模型已被证明无法泛化到以前未见过的患者-治疗组合。我们假设领域特定的先验知识——如医学教科书和生物医学知识图谱——可以提供治疗提议适应度的有意义替代信号。为此，我们引入了基于大语言模型（LLM）的熵引导优化与知情先验（LEON）方法，这是一种先验原则上的方法，利用 LLM 作为黑盒优化器，无需任何特定任务的微调，利用其将无结构领域知识语境化的能力，在自然语言中提出个性化治疗计划。在实践中，我们通过“提示优化”实现 LEON，使用 LLM 作为提出治疗设计的随机引擎。实验表明，LEON 在提出个体化治疗方案方面优于传统方法和基于 LLM 的方法。 

---
# Dual-Path Phishing Detection: Integrating Transformer-Based NLP with Structural URL Analysis 

**Title (ZH)**: 双路径钓鱼检测：结合基于 Transformer 的自然语言处理与结构化 URL 分析 

**Authors**: Ibrahim Altan, Abdulla Bachir, Yousuf Parbhulkar, Abdul Muksith Rizvi, Moshiur Farazi  

**Link**: [PDF](https://arxiv.org/pdf/2509.20972)  

**Abstract**: Phishing emails pose a persistent and increasingly sophisticated threat, undermining email security through deceptive tactics designed to exploit both semantic and structural vulnerabilities. Traditional detection methods, often based on isolated analysis of email content or embedded URLs, fail to comprehensively address these evolving attacks. In this paper, we propose a dual-path phishing detection framework that integrates transformer-based natural language processing (NLP) with classical machine learning to jointly analyze email text and embedded URLs. Our approach leverages the complementary strengths of semantic analysis using fine-tuned transformer architectures (e.g., DistilBERT) and structural link analysis via character-level TF-IDF vectorization paired with classical classifiers (e.g., Random Forest). Empirical evaluation on representative email and URL datasets demonstrates that this combined approach significantly improves detection accuracy. Specifically, the DistilBERT model achieves a near-optimal balance between accuracy and computational efficiency for textual phishing detection, while Random Forest notably outperforms other classical classifiers in identifying malicious URLs. The modular design allows flexibility for standalone deployment or ensemble integration, facilitating real-world adoption. Collectively, our results highlight the efficacy and practical value of this dual-path approach, establishing a scalable, accurate, and interpretable solution capable of enhancing email security against contemporary phishing threats. 

**Abstract (ZH)**: 钓鱼邮件构成持续且日益 sophisticated的威胁，通过欺骗性手法利用语义和结构漏洞削弱电子邮件安全。传统检测方法往往基于孤立分析电子邮件内容或嵌入的网址，未能全面应对这些 evolving的攻击。本文提出了一种集成变压器基础自然语言处理（NLP）与经典机器学习的双重路径钓鱼检测框架，用于同时分析电子邮件文本和嵌入的网址。我们的方法利用了微调的变压器架构（例如，DistilBERT）进行语义分析的互补优势以及基于字符级别的TF-IDF向量化与经典分类器（例如，随机森林）进行结构链接分析的互补优势。在代表性电子邮件和网址数据集上的实证评估表明，这种结合方法显著提高了检测准确性。具体而言，DistilBERT模型在文本钓鱼检测中实现了准确性与计算效率的良好平衡，而随机森林在识别恶意网址方面显著优于其他经典分类器。模块化设计使其具备独立部署或集成组合的灵活性，便于实际应用。总体而言，我们的结果突显了双重路径方法的有效性和实际价值，提供了一个可扩展、准确且可解释的解决方案，能够增强针对当前钓鱼威胁的电子邮件安全。 

---
# i-LAVA: Insights on Low Latency Voice-2-Voice Architecture for Agents 

**Title (ZH)**: i-LAVA: 低延迟语音到语音架构的见解 

**Authors**: Anupam Purwar, Aditya Choudhary  

**Link**: [PDF](https://arxiv.org/pdf/2509.20971)  

**Abstract**: We experiment with a low-latency, end-to-end voice-to-voice communication model to optimize it for real-time conversational applications. By analyzing components essential to voice to voice (V-2-V) system viz. automatic speech recognition (ASR), text-to-speech (TTS), and dialog management, our work analyzes how to reduce processing time while maintaining high-quality interactions to identify the levers for optimizing V-2-V system. Our work identifies that TTS component which generates life-like voice, full of emotions including natural pauses and exclamations has highest impact on Real time factor (RTF). The experimented V-2-V architecture utilizes CSM1b has the capability to understand tone as well as context of conversation by ingesting both audio and text of prior exchanges to generate contextually accurate speech. We explored optimization of Residual Vector Quantization (RVQ) iterations by the TTS decoder which come at a cost of decrease in the quality of voice generated. Our experimental evaluations also demonstrate that for V-2-V implementations based on CSM most important optimizations can be brought by reducing the number of RVQ Iterations along with the codebooks used in Mimi. 

**Abstract (ZH)**: 我们实验了一个低延迟、端到端的语音到语音通信模型，以优化其适用于实时对话应用的能力。通过分析语音到语音（V-2-V）系统的关键组件，如自动语音识别（ASR）、文本转语音（TTS）和对话管理，我们的研究分析了如何在保持高质量交互的前提下减少处理时间，以确定优化V-2-V系统的杠杆。我们的研究发现，能够生成充满情感的真实语音，包括自然的停顿和感叹的TTS组件对实时因子的影响最大。所实验的V-2-V架构利用了CSM1b，能够通过摄入先前交流的音频和文本来理解语气和对话的上下文，从而生成上下文相关的语音。我们研究了TTS解码器中残差向量量化（RVQ）迭代的优化，这会以牺牲语音质量为代价。我们的实验评估还表明，对于基于CSM的V-2-V实现，最重要的优化可以通过减少RVQ迭代次数和Mimi中使用的码本数量来实现。 

---
# Unlocking Financial Insights: An advanced Multimodal Summarization with Multimodal Output Framework for Financial Advisory Videos 

**Title (ZH)**: 解锁财务洞察：一种用于金融顾问视频的先进多模态总结框架及多模态输出模型 

**Authors**: Sarmistha Das, R E Zera Marveen Lyngkhoi, Sriparna Saha, Alka Maurya  

**Link**: [PDF](https://arxiv.org/pdf/2509.20961)  

**Abstract**: The dynamic propagation of social media has broadened the reach of financial advisory content through podcast videos, yet extracting insights from lengthy, multimodal segments (30-40 minutes) remains challenging. We introduce FASTER (Financial Advisory Summariser with Textual Embedded Relevant images), a modular framework that tackles three key challenges: (1) extracting modality-specific features, (2) producing optimized, concise summaries, and (3) aligning visual keyframes with associated textual points. FASTER employs BLIP for semantic visual descriptions, OCR for textual patterns, and Whisper-based transcription with Speaker diarization as BOS features. A modified Direct Preference Optimization (DPO)-based loss function, equipped with BOS-specific fact-checking, ensures precision, relevance, and factual consistency against the human-aligned summary. A ranker-based retrieval mechanism further aligns keyframes with summarized content, enhancing interpretability and cross-modal coherence. To acknowledge data resource scarcity, we introduce Fin-APT, a dataset comprising 470 publicly accessible financial advisory pep-talk videos for robust multimodal research. Comprehensive cross-domain experiments confirm FASTER's strong performance, robustness, and generalizability when compared to Large Language Models (LLMs) and Vision-Language Models (VLMs). By establishing a new standard for multimodal summarization, FASTER makes financial advisory content more accessible and actionable, thereby opening new avenues for research. The dataset and code are available at: this https URL 

**Abstract (ZH)**: 社交媒体动态传播通过播客视频拓宽了财经顾问内容的覆盖面，但从中提取见解仍面临挑战，尤其是在长达30-40分钟的多模态段落中。我们提出了一种模块化框架FASTER（Financial Advisory Summariser with Textual Embedded Relevant images），以应对三个关键挑战：（1）提取特定模态特征，（2）生成优化的摘要，（3）将视觉关键帧与相关文本要点对齐。FASTER使用BLIP进行语义视觉描述，OCR进行文本模式识别，并使用基于Whisper的转录与讲者定位作为BOS特征。通过结合针对BOS的具体事实核查的改进的Direct Preference Optimization (DPO)-基于损失函数，FASTER确保了精准性、相关性和事实一致性，与人工对齐的摘要比对。基于排名的检索机制进一步将关键帧与摘要内容对齐，增强了可解释性和跨模态一致性。为应对数据资源稀缺，我们引入了Fin-APT数据集，包含470个公开可访问的财经顾问激励视频，以支撑稳健的多模态研究。跨领域实验证明，与大型语言模型（LLMs）和视觉-语言模型（VLMs）相比，FASTER在性能、稳健性和通用性方面表现出色。通过确立新的多模态总结标准，FASTER使财经顾问内容更加易于获取和实用，并为研究开辟了新途径。数据集和代码可在以下链接获取：this https URL。 

---
# Flow Matching in the Low-Noise Regime: Pathologies and a Contrastive Remedy 

**Title (ZH)**: 低噪声区间内的流匹配：病态问题与对比性 remedy 探讨 

**Authors**: Weili Zeng, Yichao Yan  

**Link**: [PDF](https://arxiv.org/pdf/2509.20952)  

**Abstract**: Flow matching has recently emerged as a powerful alternative to diffusion models, providing a continuous-time formulation for generative modeling and representation learning. Yet, we show that this framework suffers from a fundamental instability in the low-noise regime. As noise levels approach zero, arbitrarily small perturbations in the input can induce large variations in the velocity target, causing the condition number of the learning problem to diverge. This ill-conditioning not only slows optimization but also forces the encoder to reallocate its limited Jacobian capacity toward noise directions, thereby degrading semantic representations. We provide the first theoretical analysis of this phenomenon, which we term the low-noise pathology, establishing its intrinsic link to the structure of the flow matching objective. Building on these insights, we propose Local Contrastive Flow (LCF), a hybrid training protocol that replaces direct velocity regression with contrastive feature alignment at small noise levels, while retaining standard flow matching at moderate and high noise. Empirically, LCF not only improves convergence speed but also stabilizes representation quality. Our findings highlight the critical importance of addressing low-noise pathologies to unlock the full potential of flow matching for both generation and representation learning. 

**Abstract (ZH)**: 低噪声病理现象下的流动匹配局部对比训练 

---
# CTI Dataset Construction from Telegram 

**Title (ZH)**: CTI数据集从Telegram的构建 

**Authors**: Dincy R. Arikkat, Sneha B. T., Serena Nicolazzo, Antonino Nocera, Vinod P., Rafidha Rehiman K. A., Karthika R  

**Link**: [PDF](https://arxiv.org/pdf/2509.20943)  

**Abstract**: Cyber Threat Intelligence (CTI) enables organizations to anticipate, detect, and mitigate evolving cyber threats. Its effectiveness depends on high-quality datasets, which support model development, training, evaluation, and benchmarking. Building such datasets is crucial, as attack vectors and adversary tactics continually evolve. Recently, Telegram has gained prominence as a valuable CTI source, offering timely and diverse threat-related information that can help address these challenges. In this work, we address these challenges by presenting an end-to-end automated pipeline that systematically collects and filters threat-related content from Telegram. The pipeline identifies relevant Telegram channels and scrapes 145,349 messages from 12 curated channels out of 150 identified sources. To accurately filter threat intelligence messages from generic content, we employ a BERT-based classifier, achieving an accuracy of 96.64%. From the filtered messages, we compile a dataset of 86,509 malicious Indicators of Compromise, including domains, IPs, URLs, hashes, and CVEs. This approach not only produces a large-scale, high-fidelity CTI dataset but also establishes a foundation for future research and operational applications in cyber threat detection. 

**Abstract (ZH)**: 基于Telegram的端到端自动化威胁情报收集与过滤pipeline：构建高质网络威胁情报数据集 

---
# Deep Learning for Crime Forecasting: The Role of Mobility at Fine-grained Spatiotemporal Scales 

**Title (ZH)**: 深度学习在犯罪预测中的作用：细粒度时空尺度上的流动性因素 

**Authors**: Ariadna Albors Zumel, Michele Tizzoni, Gian Maria Campedelli  

**Link**: [PDF](https://arxiv.org/pdf/2509.20913)  

**Abstract**: Objectives: To develop a deep learning framework to evaluate if and how incorporating micro-level mobility features, alongside historical crime and sociodemographic data, enhances predictive performance in crime forecasting at fine-grained spatial and temporal resolutions.
Methods: We advance the literature on computational methods and crime forecasting by focusing on four U.S. cities (i.e., Baltimore, Chicago, Los Angeles, and Philadelphia). We employ crime incident data obtained from each city's police department, combined with sociodemographic data from the American Community Survey and human mobility data from Advan, collected from 2019 to 2023. This data is aggregated into grids with equally sized cells of 0.077 sq. miles (0.2 sq. kms) and used to train our deep learning forecasting model, a Convolutional Long Short-Term Memory (ConvLSTM) network, which predicts crime occurrences 12 hours ahead using 14-day and 2-day input sequences. We also compare its performance against three baseline models: logistic regression, random forest, and standard LSTM.
Results: Incorporating mobility features improves predictive performance, especially when using shorter input sequences. Noteworthy, however, the best results are obtained when both mobility and sociodemographic features are used together, with our deep learning model achieving the highest recall, precision, and F1 score in all four cities, outperforming alternative methods. With this configuration, longer input sequences enhance predictions for violent crimes, while shorter sequences are more effective for property crimes.
Conclusion: These findings underscore the importance of integrating diverse data sources for spatiotemporal crime forecasting, mobility included. They also highlight the advantages (and limits) of deep learning when dealing with fine-grained spatial and temporal scales. 

**Abstract (ZH)**: 研究目标：开发一种深度学习框架，评估将微观层面的移动特征与历史犯罪和社科人口统计数据结合是否能提高在精细时空分辨率下犯罪预报的预测性能。方法：通过专注于美国四个城市（巴尔的摩、芝加哥、洛杉矶和费城），推进计算方法和犯罪预报的研究。利用每个城市警察部门提供的犯罪事件数据，结合美国社区调查中的社科人口统计数据以及2019年至2023年期间从Advan收集的人口移动数据进行研究。将这些数据聚合为0.077平方英里（0.2平方公里）等大小的网格单元，并用于训练基于卷积长短期记忆（ConvLSTM）网络的深度学习预报模型，该模型使用14天和2天的输入序列预测12小时后的犯罪事件。同时，将其性能与三种基线模型（Logistic回归、随机森林和标准LSTM）进行对比。结果：引入移动特征可提高预测性能，尤其是在使用较短输入序列时。值得注意的是，当同时使用移动和社科人口特征时，获得最佳结果，我们的深度学习模型在四个城市中的召回率、精确率和F1分数均居最高，优于其他方法。在该配置下，更长的输入序列对暴力犯罪的预测有提高作用，而较短的序列对财产犯罪更为有效。结论：这些发现强调了在时空犯罪预报中整合多种数据源的重要性，包括移动数据。它们还突显了在细粒度时空尺度下使用深度学习的优势及其局限性。 

---
# FerretNet: Efficient Synthetic Image Detection via Local Pixel Dependencies 

**Title (ZH)**: FerretNet：通过局部像素依赖高效合成图像检测 

**Authors**: Shuqiao Liang, Jian Liu, Renzhang Chen, Quanlong Guan  

**Link**: [PDF](https://arxiv.org/pdf/2509.20890)  

**Abstract**: The increasing realism of synthetic images generated by advanced models such as VAEs, GANs, and LDMs poses significant challenges for synthetic image detection. To address this issue, we explore two artifact types introduced during the generation process: (1) latent distribution deviations and (2) decoding-induced smoothing effects, which manifest as inconsistencies in local textures, edges, and color transitions. Leveraging local pixel dependencies (LPD) properties rooted in Markov Random Fields, we reconstruct synthetic images using neighboring pixel information to expose disruptions in texture continuity and edge coherence. Building upon LPD, we propose FerretNet, a lightweight neural network with only 1.1M parameters that delivers efficient and robust synthetic image detection. Extensive experiments demonstrate that FerretNet, trained exclusively on the 4-class ProGAN dataset, achieves an average accuracy of 97.1% on an open-world benchmark comprising across 22 generative models, surpassing state-of-the-art methods by 10.6%. 

**Abstract (ZH)**: 高级模型如VAEs、GANs和LDMs生成的合成图像日益逼真，给合成图像检测带来了重大挑战。为应对这一问题，我们探索了生成过程中引入的两种artifact类型：(1) 潜在分布偏差和(2) 解码引起的平滑效应，这些效应表现为局部纹理、边缘和颜色过渡的一致性问题。利用马尔可夫随机场中局部像素依赖性（LPD）的特性，我们利用相邻像素信息重构合成图像，以揭示纹理连续性和边缘一致性中的中断。在此基础上，我们提出了FerretNet，一个仅含1.1M参数的轻量级神经网络，实现了高效且稳健的合成图像检测。广泛实验表明，FerretNet仅在4类ProGAN数据集上训练，能够在包含22个生成模型的开放世界基准中达到平均97.1%的准确率，超过了现有最先进的方法10.6个百分点。 

---
# Improving Early Sepsis Onset Prediction Through Federated Learning 

**Title (ZH)**: 通过联邦学习改善早期脓毒症发作预测 

**Authors**: Christoph Düsing, Philipp Cimiano  

**Link**: [PDF](https://arxiv.org/pdf/2509.20885)  

**Abstract**: Early and accurate prediction of sepsis onset remains a major challenge in intensive care, where timely detection and subsequent intervention can significantly improve patient outcomes. While machine learning models have shown promise in this domain, their success is often limited by the amount and diversity of training data available to individual hospitals and Intensive Care Units (ICUs). Federated Learning (FL) addresses this issue by enabling collaborative model training across institutions without requiring data sharing, thus preserving patient privacy. In this work, we propose a federated, attention-enhanced Long Short-Term Memory model for sepsis onset prediction, trained on multi-centric ICU data. Unlike existing approaches that rely on fixed prediction windows, our model supports variable prediction horizons, enabling both short- and long-term forecasting in a single unified model. During analysis, we put particular emphasis on the improvements through our approach in terms of early sepsis detection, i.e., predictions with large prediction windows by conducting an in-depth temporal analysis. Our results prove that using FL does not merely improve overall prediction performance (with performance approaching that of a centralized model), but is particularly beneficial for early sepsis onset prediction. Finally, we show that our choice of employing a variable prediction window rather than a fixed window does not hurt performance significantly but reduces computational, communicational, and organizational overhead. 

**Abstract (ZH)**: federated, 注意力增强的长短期记忆模型在多中心ICU数据上的脓毒症 onset 预测 

---
# Integrating Object Interaction Self-Attention and GAN-Based Debiasing for Visual Question Answering 

**Title (ZH)**: 集成物体交互自注意力和基于GAN的去偏见方法的视觉问答 

**Authors**: Zhifei Li, Feng Qiu, Yiran Wang, Yujing Xia, Kui Xiao, Miao Zhang, Yan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20884)  

**Abstract**: Visual Question Answering (VQA) presents a unique challenge by requiring models to understand and reason about visual content to answer questions accurately. Existing VQA models often struggle with biases introduced by the training data, leading to over-reliance on superficial patterns and inadequate generalization to diverse questions and images. This paper presents a novel model, IOG-VQA, which integrates Object Interaction Self-Attention and GAN-Based Debiasing to enhance VQA model performance. The self-attention mechanism allows our model to capture complex interactions between objects within an image, providing a more comprehensive understanding of the visual context. Meanwhile, the GAN-based debiasing framework generates unbiased data distributions, helping the model to learn more robust and generalizable features. By leveraging these two components, IOG-VQA effectively combines visual and textual information to address the inherent biases in VQA datasets. Extensive experiments on the VQA-CP v1 and VQA-CP v2 datasets demonstrate that our model shows excellent performance compared with the existing methods, particularly in handling biased and imbalanced data distributions highlighting the importance of addressing both object interactions and dataset biases in advancing VQA tasks. Our code is available at this https URL. 

**Abstract (ZH)**: 视觉问答（VQA）通过要求模型理解和推理视觉内容来准确回答问题，从而提出了一项独特的挑战。现有VQA模型常常难以应对训练数据引入的偏差，导致模型过度依赖表面模式，并且难以将学到的知识应用于多样化的问句和图像。本文提出了一种名为IOG-VQA的新模型，该模型结合了对象交互自注意力和基于GAN的去偏技术，以提升VQA模型的表现。自注意力机制使我们的模型能够捕捉图像中对象之间的复杂交互，提供更全面的视觉上下文理解。同时，基于GAN的去偏框架生成无偏的数据分布，帮助模型学习更稳健和泛化的特征。通过利用这两个组件，IOG-VQA有效地结合视觉和文本信息，以应对VQA数据集中的固有偏差。在VQA-CP v1和VQA-CP v2数据集上的广泛实验表明，与现有方法相比，我们的模型在处理有偏和不均衡数据分布方面表现优异，强调了在推进VQA任务时同时关注对象交互和数据集偏差的重要性。我们的代码可在以下链接获得：this https URL。 

---
# On Theoretical Interpretations of Concept-Based In-Context Learning 

**Title (ZH)**: 基于概念的上下文学习的理论解释 

**Authors**: Huaze Tang, Tianren Peng, Shao-lun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20882)  

**Abstract**: In-Context Learning (ICL) has emerged as an important new paradigm in natural language processing and large language model (LLM) applications. However, the theoretical understanding of the ICL mechanism remains limited. This paper aims to investigate this issue by studying a particular ICL approach, called concept-based ICL (CB-ICL). In particular, we propose theoretical analyses on applying CB-ICL to ICL tasks, which explains why and when the CB-ICL performs well for predicting query labels in prompts with only a few demonstrations. In addition, the proposed theory quantifies the knowledge that can be leveraged by the LLMs to the prompt tasks, and leads to a similarity measure between the prompt demonstrations and the query input, which provides important insights and guidance for model pre-training and prompt engineering in ICL. Moreover, the impact of the prompt demonstration size and the dimension of the LLM embeddings in ICL are also explored based on the proposed theory. Finally, several real-data experiments are conducted to validate the practical usefulness of CB-ICL and the corresponding theory. 

**Abstract (ZH)**: 基于概念的上下文学习（CB-ICL）理论研究及其在自然语言处理中的应用 

---
# SCRA-VQA: Summarized Caption-Rerank for Augmented Large Language Models in Visual Question Answering 

**Title (ZH)**: SCRA-VQA: 摘要Caption重排以增强视觉问答中的大型语言模型 

**Authors**: Yan Zhang, Jiaqing Lin, Miao Zhang, Kui Xiao, Xiaoju Hou, Yue Zhao, Zhifei Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.20871)  

**Abstract**: Acquiring high-quality knowledge is a central focus in Knowledge-Based Visual Question Answering (KB-VQA). Recent methods use large language models (LLMs) as knowledge engines for answering. These methods generally employ image captions as visual text descriptions to assist LLMs in interpreting images. However, the captions frequently include excessive noise irrelevant to the question, and LLMs generally do not comprehend VQA tasks, limiting their reasoning capabilities. To address this issue, we propose the Summarized Caption-Rerank Augmented VQA (SCRA-VQA), which employs a pre-trained visual language model to convert images into captions. Moreover, SCRA-VQA generates contextual examples for the captions while simultaneously summarizing and reordering them to exclude unrelated information. The caption-rerank process enables LLMs to understand the image information and questions better, thus enhancing the model's reasoning ability and task adaptability without expensive end-to-end training. Based on an LLM with 6.7B parameters, SCRA-VQA performs excellently on two challenging knowledge-based VQA datasets: OK-VQA and A-OKVQA, achieving accuracies of 38.8% and 34.6%. Our code is available at this https URL. 

**Abstract (ZH)**: 基于摘要Caption重排增强的Knowledge-Based视觉问答（SCRA-VQA）：高质量知识的获取在基于知识的视觉问答（KB-VQA）中是中心焦点。最近的方法使用大型语言模型（LLMs）作为知识引擎进行回答。这些方法通常利用图像说明作为视觉文本描述以辅助LLMs理解图像。然而，说明中经常包含与问题无关的噪声信息，而且LLMs一般不理解VQA任务，限制了其推理能力。为解决这一问题，我们提出了一种名为Summarized Caption-Rerank Augmented VQA（SCRA-VQA）的方法，该方法利用预训练的视觉语言模型将图像转换为说明。此外，SCRA-VQA在生成与说明相关的上下文示例的同时，对其进行了总结和重新排序，以排除无关信息。通过说明的重排过程，LLMs能够更好地理解图像信息和问题，从而增强模型的推理能力和任务适应性而无需昂贵的端到端训练。基于6.7B参数的LLM，SCRA-VQA在两个具有挑战性的基于知识的VQA数据集OK-VQA和A-OKVQA上表现出色，准确率分别为38.8%和34.6%。我们的代码可在以下链接获取：this https URL。 

---
# Model-Based Reinforcement Learning under Random Observation Delays 

**Title (ZH)**: 基于模型的强化学习在随机观测延迟下的方法 

**Authors**: Armin Karamzade, Kyungmin Kim, JB Lanier, Davide Corsi, Roy Fox  

**Link**: [PDF](https://arxiv.org/pdf/2509.20869)  

**Abstract**: Delays frequently occur in real-world environments, yet standard reinforcement learning (RL) algorithms often assume instantaneous perception of the environment. We study random sensor delays in POMDPs, where observations may arrive out-of-sequence, a setting that has not been previously addressed in RL. We analyze the structure of such delays and demonstrate that naive approaches, such as stacking past observations, are insufficient for reliable performance. To address this, we propose a model-based filtering process that sequentially updates the belief state based on an incoming stream of observations. We then introduce a simple delay-aware framework that incorporates this idea into model-based RL, enabling agents to effectively handle random delays. Applying this framework to Dreamer, we compare our approach to delay-aware baselines developed for MDPs. Our method consistently outperforms these baselines and demonstrates robustness to delay distribution shifts during deployment. Additionally, we present experiments on simulated robotic tasks, comparing our method to common practical heuristics and emphasizing the importance of explicitly modeling observation delays. 

**Abstract (ZH)**: 随机传感器延迟在部分观测马尔可夫决策过程中的建模与处理：基于模型的过滤过程及其应用 

---
# StyleBench: Evaluating thinking styles in Large Language Models 

**Title (ZH)**: StyleBench: 评估大型语言模型的思维风格 

**Authors**: Junyu Guo, Shangding Gu, Ming Jin, Costas Spanos, Javad Lavaei  

**Link**: [PDF](https://arxiv.org/pdf/2509.20868)  

**Abstract**: The effectiveness of Large Language Models (LLMs) is heavily influenced by the reasoning strategies, or styles of thought, employed in their prompts. However, the interplay between these reasoning styles, model architecture, and task type remains poorly understood. To address this, we introduce StyleBench, a comprehensive benchmark for systematically evaluating reasoning styles across diverse tasks and models. We assess five representative reasoning styles, including Chain of Thought (CoT), Tree of Thought (ToT), Algorithm of Thought (AoT), Sketch of Thought (SoT), and Chain-of-Draft (CoD) on five reasoning tasks, using 15 open-source models from major families (LLaMA, Qwen, Mistral, Gemma, GPT-OSS, Phi, and DeepSeek) ranging from 270M to 120B parameters. Our large-scale analysis reveals that no single style is universally optimal. We demonstrate that strategy efficacy is highly contingent on both model scale and task type: search-based methods (AoT, ToT) excel in open-ended problems but require large-scale models, while concise styles (SoT, CoD) achieve radical efficiency gains on well-defined tasks. Furthermore, we identify key behavioral patterns: smaller models frequently fail to follow output instructions and default to guessing, while reasoning robustness emerges as a function of scale. Our findings offer a crucial roadmap for selecting optimal reasoning strategies based on specific constraints, we open source the benchmark in this https URL. 

**Abstract (ZH)**: 大型语言模型中推理风格对其效果的影响研究：StyleBench框架的系统评估 

---
# Federated Markov Imputation: Privacy-Preserving Temporal Imputation in Multi-Centric ICU Environments 

**Title (ZH)**: 联邦马尔可夫插补：多中心ICU环境中隐私保护的时间序列插补 

**Authors**: Christoph Düsing, Philipp Cimiano  

**Link**: [PDF](https://arxiv.org/pdf/2509.20867)  

**Abstract**: Missing data is a persistent challenge in federated learning on electronic health records, particularly when institutions collect time-series data at varying temporal granularities. To address this, we propose Federated Markov Imputation (FMI), a privacy-preserving method that enables Intensive Care Units (ICUs) to collaboratively build global transition models for temporal imputation. We evaluate FMI on a real-world sepsis onset prediction task using the MIMIC-IV dataset and show that it outperforms local imputation baselines, especially in scenarios with irregular sampling intervals across ICUs. 

**Abstract (ZH)**: 电子健康记录中联邦学习中缺失数据是一个持续性挑战，尤其是在机构以不同时间粒度收集时间序列数据的情况下。为此，我们提出了一种名为Federated Markov Imputation（FMI）的隐私保护方法，使重症监护单元（ICUs）能够合作构建全局转换模型以进行时间序列插补。我们使用MIMIC-IV数据集评估了FMI在脓毒症发作预测任务中的性能，并展示了它在ICU间采样间隔不规律的情况下优于局部插补基线方法。 

---
# TasselNetV4: A vision foundation model for cross-scene, cross-scale, and cross-species plant counting 

**Title (ZH)**: TasselNetV4：一种适用于跨场景、跨尺度和跨物种植物计数的视觉基础模型 

**Authors**: Xiaonan Hu, Xuebing Li, Jinyu Xu, Abdulkadir Duran Adan, Letian Zhou, Xuhui Zhu, Yanan Li, Wei Guo, Shouyang Liu, Wenzhong Liu, Hao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20857)  

**Abstract**: Accurate plant counting provides valuable information for agriculture such as crop yield prediction, plant density assessment, and phenotype quantification. Vision-based approaches are currently the mainstream solution. Prior art typically uses a detection or a regression model to count a specific plant. However, plants have biodiversity, and new cultivars are increasingly bred each year. It is almost impossible to exhaust and build all species-dependent counting models. Inspired by class-agnostic counting (CAC) in computer vision, we argue that it is time to rethink the problem formulation of plant counting, from what plants to count to how to count plants. In contrast to most daily objects with spatial and temporal invariance, plants are dynamic, changing with time and space. Their non-rigid structure often leads to worse performance than counting rigid instances like heads and cars such that current CAC and open-world detection models are suboptimal to count plants. In this work, we inherit the vein of the TasselNet plant counting model and introduce a new extension, TasselNetV4, shifting from species-specific counting to cross-species counting. TasselNetV4 marries the local counting idea of TasselNet with the extract-and-match paradigm in CAC. It builds upon a plain vision transformer and incorporates novel multi-branch box-aware local counters used to enhance cross-scale robustness. Two challenging datasets, PAC-105 and PAC-Somalia, are harvested. Extensive experiments against state-of-the-art CAC models show that TasselNetV4 achieves not only superior counting performance but also high this http URL results indicate that TasselNetV4 emerges to be a vision foundation model for cross-scene, cross-scale, and cross-species plant counting. 

**Abstract (ZH)**: 基于视觉的准确植物计数为农业提供了作物产量预测、植物密度评估和表型量化等有价值的信息。现有的方法通常使用检测或回归模型来计数特定的植物。由于植物具有生物多样性，且每年培育出越来越多的新品种，几乎不可能构建所有物种依赖的计数模型。受计算机视觉中类无关计数(CAC)的启发，我们认为是时候重新考虑植物计数的问题表述，从需要计数哪些植物转变为如何计数植物。与大多数具有时空不变性的日常物体不同，植物是动态的，随时间和空间而变化，其非刚性结构导致其计数性能通常劣于计数刚性实例（如头部和车辆）的方法，使得当前的CAC和开放世界检测模型在植物计数上效果不佳。在本工作中，我们继承了TasselNet植物计数模型的思路，并提出了一个新的扩展TasselNetV4，从物种特定计数转向跨物种计数。TasselNetV4将TasselNet的局部计数理念与CAC中的提取-匹配范式相结合。它基于简单的视觉Transformer，并结合了新的多分支盒感知局部计数器，以增强跨尺度鲁棒性。收集了两个具有挑战性的数据集PAC-105和PAC-Somalia。与最新的CAC模型进行广泛实验表明，TasselNetV4不仅在计数性能上取得了卓越的表现，而且具有高的交叉场景、跨尺度和跨物种植物计数潜力。研究结果表明，TasselNetV4成为跨场景、跨尺度和跨物种植物计数的视觉基础模型。 

---
# FHRFormer: A Self-supervised Transformer Approach for Fetal Heart Rate Inpainting and Forecasting 

**Title (ZH)**: FHRFormer: 一种自监督Transformer方法用于胎儿心率恢复和预测 

**Authors**: Kjersti Engan, Neel Kanwal, Anita Yeconia, Ladislaus Blacy, Yuda Munyaw, Estomih Mduma, Hege Ersdal  

**Link**: [PDF](https://arxiv.org/pdf/2509.20852)  

**Abstract**: Approximately 10\% of newborns require assistance to initiate breathing at birth, and around 5\% need ventilation support. Fetal heart rate (FHR) monitoring plays a crucial role in assessing fetal well-being during prenatal care, enabling the detection of abnormal patterns and supporting timely obstetric interventions to mitigate fetal risks during labor. Applying artificial intelligence (AI) methods to analyze large datasets of continuous FHR monitoring episodes with diverse outcomes may offer novel insights into predicting the risk of needing breathing assistance or interventions. Recent advances in wearable FHR monitors have enabled continuous fetal monitoring without compromising maternal mobility. However, sensor displacement during maternal movement, as well as changes in fetal or maternal position, often lead to signal dropouts, resulting in gaps in the recorded FHR data. Such missing data limits the extraction of meaningful insights and complicates automated (AI-based) analysis. Traditional approaches to handle missing data, such as simple interpolation techniques, often fail to preserve the spectral characteristics of the signals. In this paper, we propose a masked transformer-based autoencoder approach to reconstruct missing FHR signals by capturing both spatial and frequency components of the data. The proposed method demonstrates robustness across varying durations of missing data and can be used for signal inpainting and forecasting. The proposed approach can be applied retrospectively to research datasets to support the development of AI-based risk algorithms. In the future, the proposed method could be integrated into wearable FHR monitoring devices to achieve earlier and more robust risk detection. 

**Abstract (ZH)**: 约10%的新生儿需要在出生时接受呼吸辅助，约5%的新生儿需要通气支持。胎儿心率（FHR）监测在孕期护理中评估胎儿状况方面发挥着重要作用，有助于检测异常模式并支持及时的产科干预，以减轻分娩过程中的胎儿风险。将人工智能（AI）方法应用于分析包含多样化结局的连续FHR监测数据集，可能会为预测需要呼吸辅助或干预的风险提供新的见解。可穿戴FHR监测设备的最新进展使在不牺牲产妇移动性的前提下实现连续胎儿监测成为可能。然而，产妇运动导致的传感器位移以及胎儿或产妇体位的变化常常会导致信号中断，从而在记录的FHR数据中产生间隙。这些缺失的数据限制了有意义洞见的提取，并使基于自动（AI）的方法分析复杂化。传统的缺失数据处理方法，如简单的插值技术，通常无法保留信号的频谱特征。本文提出了一种基于掩码变换器的自编码器方法，通过捕获数据的空间和频率成分来重建缺失的FHR信号。所提出的方法能够在不同缺失数据持续时间下表现出鲁棒性，并可用于信号修补和预测。该提出的方法可以应用于回顾性研究数据集，以支持基于AI的风险算法的开发。未来，该方法可以集成到可穿戴FHR监测设备中，以实现更早和更稳健的风险检测。 

---
# Robust Multi-Omics Integration from Incomplete Modalities Significantly Improves Prediction of Alzheimer's Disease 

**Title (ZH)**: 从不完整模态中实现稳健的多组学整合显著提高阿尔茨海默病预测 

**Authors**: Sungjoon Park, Kyungwook Lee, Soorin Yim, Doyeong Hwang, Dongyun Kim, Soonyoung Lee, Amy Dunn, Daniel Gatti, Elissa Chesler, Kristen O'Connell, Kiyoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.20842)  

**Abstract**: Multi-omics data capture complex biomolecular interactions and provide insights into metabolism and disease. However, missing modalities hinder integrative analysis across heterogeneous omics. To address this, we present MOIRA (Multi-Omics Integration with Robustness to Absent modalities), an early integration method enabling robust learning from incomplete omics data via representation alignment and adaptive aggregation. MOIRA leverages all samples, including those with missing modalities, by projecting each omics dataset onto a shared embedding space where a learnable weighting mechanism fuses them. Evaluated on the Religious Order Study and Memory and Aging Project (ROSMAP) dataset for Alzheimer's Disease (AD), MOIRA outperformed existing approaches, and further ablation studies confirmed modality-wise contributions. Feature importance analysis revealed AD-related biomarkers consistent with prior literature, highlighting the biological relevance of our approach. 

**Abstract (ZH)**: 多组学数据捕获复杂的生物分子相互作用，并提供关于代谢和疾病的洞察。然而，缺失的数据模态阻碍了异质组学之间的综合分析。为了解决这一问题，我们提出了MOIRA（多组学集成并具有缺失模态鲁棒性），这是一种早期集成方法，通过表示对齐和自适应聚合，从不完整的组学数据中实现稳健学习。MOIRA 利用所有样本（包括具有缺失模态的样本），通过将每个组学数据集投影到共享嵌入空间，在该空间中，可学习的加权机制将它们融合。在用于阿尔茨海默病（AD）研究的宗教秩序研究和记忆与衰老项目（ROSMAP）数据集上进行评估，MOIRA 超过了现有方法，并且进一步的消融研究确认了模态级别的贡献。特征重要性分析揭示了与先前文献一致的AD相关生物标志物，突显了我们方法的生物学相关性。 

---
# ImaginationPolicy: Towards Generalizable, Precise and Reliable End-to-End Policy for Robotic Manipulation 

**Title (ZH)**: 想象策略：迈向通用、精确且可靠的端到端机器人 manipulation 策略 

**Authors**: Dekun Lu, Wei Gao, Kui Jia  

**Link**: [PDF](https://arxiv.org/pdf/2509.20841)  

**Abstract**: End-to-end robot manipulation policies offer significant potential for enabling embodied agents to understand and interact with the world. Unlike traditional modular pipelines, end-to-end learning mitigates key limitations such as information loss between modules and feature misalignment caused by isolated optimization targets. Despite these advantages, existing end-to-end neural networks for robotic manipulation--including those based on large VLM/VLA models--remain insufficiently performant for large-scale practical deployment. In this paper, we take a step towards an end-to-end manipulation policy that is generalizable, accurate and reliable. To achieve this goal, we propose a novel Chain of Moving Oriented Keypoints (CoMOK) formulation for robotic manipulation. Our formulation is used as the action representation of a neural policy, which can be trained in an end-to-end fashion. Such an action representation is general, as it extends the standard end-effector pose action representation and supports a diverse set of manipulation tasks in a unified manner. The oriented keypoint in our method enables natural generalization to objects with different shapes and sizes, while achieving sub-centimeter accuracy. Moreover, our formulation can easily handle multi-stage tasks, multi-modal robot behaviors, and deformable objects. Extensive simulated and hardware experiments demonstrate the effectiveness of our method. 

**Abstract (ZH)**: 端到端机器人操作策略为实现具身智能体对世界的理解和交互提供了巨大潜力。与传统的模块化管线不同，端到端学习缓解了模块间信息丢失和孤立优化目标导致的特征错位等关键限制。尽管具有这些优势，现有基于端到端学习的机器人操作神经网络——包括基于大规模VLM/VLA模型的网络——在大规模实际部署中仍不够高效。本文朝着构建一个通用、准确且可靠的端到端操作策略迈进。为此，我们提出了一种新颖的移动定向关键点链（CoMOK）形式化方法，用于机器人操作。该形式化方法用作神经策略的动作表示，并可以在端到端方式进行训练。这种动作表示是通用的，因为它扩展了标准的末端执行器姿态动作表示，并以统一的方式支持多种操作任务。我们方法中的定向关键点使模型能够自然地泛化到具有不同形状和大小的物体，并且在厘米级精度下实现亚厘米级的准确性。此外，该形式化方法能够轻松处理多阶段任务、多模态机器人行为以及变形物体。大量模拟和硬件实验表明了该方法的有效性。 

---
# Verification Limits Code LLM Training 

**Title (ZH)**: LLM训练的验证极限 

**Authors**: Srishti Gureja, Elena Tommasone, Jingyi He, Sara Hooker, Matthias Gallé, Marzieh Fadaee  

**Link**: [PDF](https://arxiv.org/pdf/2509.20837)  

**Abstract**: Large language models for code generation increasingly rely on synthetic data, where both problem solutions and verification tests are generated by models. While this enables scalable data creation, it introduces a previously unexplored bottleneck: the verification ceiling, in which the quality and diversity of training data are fundamentally constrained by the capabilities of synthetic verifiers. In this work, we systematically study how verification design and strategies influence model performance. We investigate (i) what we verify by analyzing the impact of test complexity and quantity: richer test suites improve code generation capabilities (on average +3 pass@1), while quantity alone yields diminishing returns, (ii) how we verify by exploring relaxed pass thresholds: rigid 100% pass criteria can be overly restrictive. By allowing for relaxed thresholds or incorporating LLM-based soft verification, we can recover valuable training data, leading to a 2-4 point improvement in pass@1 performance. However, this benefit is contingent upon the strength and diversity of the test cases used, and (iii) why verification remains necessary through controlled comparisons of formally correct versus incorrect solutions and human evaluation: retaining diverse correct solutions per problem yields consistent generalization gains. Our results show that Verification as currently practiced is too rigid, filtering out valuable diversity. But it cannot be discarded, only recalibrated. By combining calibrated verification with diverse, challenging problem-solution pairs, we outline a path to break the verification ceiling and unlock stronger code generation models. 

**Abstract (ZH)**: 大型语言模型用于代码生成 Increasingly Relies on Synthetic Data with Verification Ceilings: A Systematic Study on Verification Design and Strategies 

---
# Security-aware Semantic-driven ISAC via Paired Adversarial Residual Networks 

**Title (ZH)**: 基于配对对抗残差网络的安全性aware语义驱动ISAC 

**Authors**: Yu Liu, Boxiang He, Fanggang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20835)  

**Abstract**: This paper proposes a novel and flexible security-aware semantic-driven integrated sensing and communication (ISAC) framework, namely security semantic ISAC (SS-ISAC). Inspired by the positive impact of the adversarial attack, a pair of pluggable encryption and decryption modules is designed in the proposed SS-ISAC framework. The encryption module is installed after the semantic transmitter, adopting a trainable adversarial residual network (ARN) to create the adversarial attack. Correspondingly, the decryption module before the semantic receiver utilizes another trainable ARN to mitigate the adversarial attack and noise. These two modules can be flexibly assembled considering the system security demands, without drastically modifying the hardware infrastructure. To ensure the sensing and communication (SAC) performance while preventing the eavesdropping threat, the above ARNs are jointly optimized by minimizing a carefully designed loss function that relates to the adversarial attack power, SAC performance, as well as the privacy leakage risk. Simulation results validate the effectiveness of the proposed SS-ISAC framework in terms of both SAC and eavesdropping prevention performance. 

**Abstract (ZH)**: 这种论文提出了一种新颖且灵活的安全意识语义驱动的集成感知与通信（ISAC）框架，即安全语义ISAC（SS-ISAC）。受对抗攻击积极影响的启发，在提出的SS-ISAC框架中设计了一对可插拔的加密和解密模块。加密模块安装在语义发送器之后，采用可训练的对抗残差网络（ARN）生成对抗攻击。相应地，语义接收器之前的解密模块利用另一个可训练的ARN来缓解对抗攻击和噪声。这两个模块可以根据安全需求灵活组装，无需大幅修改硬件基础设施。为了确保感知与通信（SAC）性能并防止窃听威胁，上述ARNs通过最小化一个精心设计的损失函数联合优化，该损失函数与对抗攻击能力、SAC性能以及隐私泄露风险相关。仿真结果验证了所提出的SS-ISAC框架在感知与通信及窃听防御方面的有效性。 

---
# Trustworthy Semantic Communication for Vehicular Networks: Challenges and Solutions 

**Title (ZH)**: 车辆网络中可信语义通信的挑战与解决方案 

**Authors**: Yanghe Pan, Yuntao Wang, Shaolong Guo, Chengyu Yin, Ruidong Li, Zhou Su, Yuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20830)  

**Abstract**: Semantic communication (SemCom) has the potential to significantly reduce communication delay in vehicle-to-everything (V2X) communications within vehicular networks (VNs). However, the deployment of vehicular SemCom networks (VN-SemComNets) faces critical trust challenges in information transmission, semantic encoding, and communication entity reliability. This paper proposes an innovative three-layer trustworthy VN-SemComNet architecture. Specifically, we introduce a semantic camouflage transmission mechanism leveraging defensive adversarial noise for active eavesdropping defense, a robust federated encoder-decoder training framework to mitigate encoder-decoder poisoning attacks, and an audit game-based distributed vehicle trust management mechanism to deter untrustworthy vehicles. A case study validates the effectiveness of the proposed solutions. Lastly, essential future research directions are pointed out to advance this emerging field. 

**Abstract (ZH)**: 语义通信在车辆到一切（V2X）通信中具有显著减少通信延迟的潜力。然而，车辆语义通信网络（VN-SemComNets）的部署面临着信息传输、语义编码和通信实体可靠性方面的关键信任挑战。本文提出了一种创新的三层可信VN-SemComNet架构。具体而言，我们引入了利用防御性对抗噪声的语义迷彩传输机制，以实现主动窃听防御，提出了一种稳健的联邦编码-解码训练框架，以缓解编码器-解码器污染攻击，并提出了一种基于审计博弈的分布式车辆信任管理机制，以抵制不信任车辆。案例研究验证了所提解决方案的有效性。最后，指出了未来研究的方向，以促进这一新兴领域的发展。 

---
# CaTS-Bench: Can Language Models Describe Numeric Time Series? 

**Title (ZH)**: CaTS-Bench: 语言模型能够描述数值时间序列吗？ 

**Authors**: Luca Zhou, Pratham Yashwante, Marshall Fisher, Alessio Sampieri, Zihao Zhou, Fabio Galasso, Rose Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20823)  

**Abstract**: Time series captioning, the task of describing numeric time series in natural language, requires numerical reasoning, trend interpretation, and contextual understanding. Existing benchmarks, however, often rely on synthetic data or overly simplistic captions, and typically neglect metadata and visual representations. To close this gap, we introduce CaTS-Bench, the first large-scale, real-world benchmark for Context-aware Time Series captioning. CaTS-Bench is derived from 11 diverse datasets reframed as captioning and Q&A tasks, comprising roughly 465k training and 105k test timestamps. Each sample includes a numeric series segment, contextual metadata, a line-chart image, and a caption. A key contribution of this work is the scalable pipeline used to generate reference captions: while most references are produced by an oracle LLM and verified through factual checks, human indistinguishability studies, and diversity analyses, we also provide a human-revisited subset of 579 test captions, refined from LLM outputs to ensure accuracy and human-like style. Beyond captioning, CaTS-Bench offers 460 multiple-choice questions targeting deeper aspects of time series reasoning. We further propose new tailored evaluation metrics and benchmark leading VLMs, highlighting both their strengths and persistent limitations. Together, these contributions establish CaTS-Bench and its captioning pipeline as a reliable and extensible foundation for future research at the intersection of time series analysis and foundation models. 

**Abstract (ZH)**: 面向上下文的时间序列描述基准（CaTS-Bench）：大规模实时序列描述基准 

---
# Even More Kawaii than Real-Person-Driven VTubers? Understanding How Viewers Perceive AI-Driven VTubers 

**Title (ZH)**: 比真人驱动VTuber更可爱？理解观众对AI驱动VTuber的认知 

**Authors**: Yiluo Wei, Yupeng He, Gareth Tyson  

**Link**: [PDF](https://arxiv.org/pdf/2509.20817)  

**Abstract**: VTubers, digital personas represented by animated avatars, have gained massive popularity. Traditionally, VTubers are operated and voiced by human controllers known as Nakanohito. The reliance on Nakanohito, however, poses risks due to potential personal controversies and operational disruptions. The emergence of AI-driven VTubers offers a new model free from these human constraints. While AI-driven VTubers present benefits such as continuous operation and reduced scandal risk, they also raise questions about authenticity and audience engagement. Therefore, to gain deeper insights, we conduct a case study, investigating viewer perceptions of Neuro-sama, the most popular AI-driven VTuber with 845k followers on Twitch and 753k followers on YouTube. We analyze 108k Reddit posts and 136k YouTube comments, aiming to better understand viewer motivations, how AI constructs the virtual persona, and perceptions of the AI as Nakanohito. Our findings enhance the understanding of AI-driven VTubers and their impact on digital streaming culture. 

**Abstract (ZH)**: AI驱动的VTuber：从脑下垂体君的粉丝视角探讨虚拟人设构建与人工智能认知 

---
# Revolutionizing Precise Low Back Pain Diagnosis via Contrastive Learning 

**Title (ZH)**: 基于对比学习的精确腰椎疼痛诊断革命 

**Authors**: Thanh Binh Le, Hoang Nhat Khang Vo, Tan-Ha Mai, Trong Nhan Phan  

**Link**: [PDF](https://arxiv.org/pdf/2509.20813)  

**Abstract**: Low back pain affects millions worldwide, driving the need for robust diagnostic models that can jointly analyze complex medical images and accompanying text reports. We present LumbarCLIP, a novel multimodal framework that leverages contrastive language-image pretraining to align lumbar spine MRI scans with corresponding radiological descriptions. Built upon a curated dataset containing axial MRI views paired with expert-written reports, LumbarCLIP integrates vision encoders (ResNet-50, Vision Transformer, Swin Transformer) with a BERT-based text encoder to extract dense representations. These are projected into a shared embedding space via learnable projection heads, configurable as linear or non-linear, and normalized to facilitate stable contrastive training using a soft CLIP loss. Our model achieves state-of-the-art performance on downstream classification, reaching up to 95.00% accuracy and 94.75% F1-score on the test set, despite inherent class imbalance. Extensive ablation studies demonstrate that linear projection heads yield more effective cross-modal alignment than non-linear variants. LumbarCLIP offers a promising foundation for automated musculoskeletal diagnosis and clinical decision support. 

**Abstract (ZH)**: 低背部疼痛影响全球数百万人，推动了需要能够联合分析复杂医学图像和相应文字报告的 robust 诊断模型的发展。我们提出 LumbarCLIP，这是一种新颖的多模态框架，利用对比语言-图像预训练对齐腰椎 MRI 扫描与其相应的放射学描述。LumbarCLIP 依托于一个经过策展的数据集，该数据集包含轴向 MRI 视图及其由专家撰写的报告，综合了视觉编码器（ResNet-50、Vision Transformer、Swin Transformer）与基于 BERT 的文本编码器以提取密集表示。通过可学习的投影头将这些表示投影到共享的嵌入空间中，配置为线性或非线性，并通过软 CLIP 损失进行规范化，以实现稳定的对比训练。尽管存在固有的类别不平衡，我们的模型在下游分类任务中实现了最先进的性能，测试集上准确率达到 95.00%、F1 分数达到 94.75%。广泛的消融研究显示，线性投影头相比非线性变体能更有效地实现跨模态对齐。LumbarCLIP 为自动肌肉骨骼诊断和临床决策支持提供了一个有前景的基础。 

---
# Leveraging What's Overfixed: Post-Correction via LLM Grammatical Error Overcorrection 

**Title (ZH)**: 利用已固化的部分：基于LLM的语法错误过度修正的后纠正方法 

**Authors**: Taehee Park, Heejin Do, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.20811)  

**Abstract**: Robust supervised fine-tuned small Language Models (sLMs) often show high reliability but tend to undercorrect. They achieve high precision at the cost of low recall. Conversely, Large Language Models (LLMs) often show the opposite tendency, making excessive overcorrection, leading to low precision. To effectively harness the strengths of LLMs to address the recall challenges in sLMs, we propose Post-Correction via Overcorrection (PoCO), a novel approach that strategically balances recall and precision. PoCO first intentionally triggers overcorrection via LLM to maximize recall by allowing comprehensive revisions, then applies a targeted post-correction step via fine-tuning smaller models to identify and refine erroneous outputs. We aim to harmonize both aspects by leveraging the generative power of LLMs while preserving the reliability of smaller supervised models. Our extensive experiments demonstrate that PoCO effectively balances GEC performance by increasing recall with competitive precision, ultimately improving the overall quality of grammatical error correction. 

**Abstract (ZH)**: 利用过度纠正提升小语言模型召回率的后纠正方法：平衡生成能力和纠正精确性 

---
# DAC-LoRA: Dynamic Adversarial Curriculum for Efficient and Robust Few-Shot Adaptation 

**Title (ZH)**: DAC-LoRA: 动态对抗课程学习以实现高效的鲁棒少样本适应 

**Authors**: Ved Umrajkar  

**Link**: [PDF](https://arxiv.org/pdf/2509.20792)  

**Abstract**: Vision-Language Models (VLMs) are foundational to critical applications like autonomous driving, medical diagnosis, and content moderation. While Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA enable their efficient adaptation to specialized tasks, these models remain vulnerable to adversarial attacks that can compromise safety-critical decisions. CLIP, the backbone for numerous downstream VLMs, is a high-value target whose vulnerabilities can cascade across the multimodal AI ecosystem. We propose Dynamic Adversarial Curriculum DAC-LoRA, a novel framework that integrates adversarial training into PEFT. The core principle of our method i.e. an intelligent curriculum of progressively challenging attack, is general and can potentially be applied to any iterative attack method. Guided by the First-Order Stationary Condition (FOSC) and a TRADES-inspired loss, DAC-LoRA achieves substantial improvements in adversarial robustness without significantly compromising clean accuracy. Our work presents an effective, lightweight, and broadly applicable method to demonstrate that the DAC-LoRA framework can be easily integrated into a standard PEFT pipeline to significantly enhance robustness. 

**Abstract (ZH)**: 动态对抗课程DAC-LoRA：一种结合对抗训练的参数高效微调框架 

---
# Towards Atoms of Large Language Models 

**Title (ZH)**: 大型语言模型的基本成分探索 

**Authors**: Chenhui Hu, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.20784)  

**Abstract**: The fundamental units of internal representations in large language models (LLMs) remain undefined, limiting further understanding of their mechanisms. Neurons or features are often regarded as such units, yet neurons suffer from polysemy, while features face concerns of unreliable reconstruction and instability. To address this issue, we propose the Atoms Theory, which defines such units as atoms. We introduce the atomic inner product (AIP) to correct representation shifting, formally define atoms, and prove the conditions that atoms satisfy the Restricted Isometry Property (RIP), ensuring stable sparse representations over atom set and linking to compressed sensing. Under stronger conditions, we further establish the uniqueness and exact $\ell_1$ recoverability of the sparse representations, and provide guarantees that single-layer sparse autoencoders (SAEs) with threshold activations can reliably identify the atoms. To validate the Atoms Theory, we train threshold-activated SAEs on Gemma2-2B, Gemma2-9B, and Llama3.1-8B, achieving 99.9% sparse reconstruction across layers on average, and more than 99.8% of atoms satisfy the uniqueness condition, compared to 0.5% for neurons and 68.2% for features, showing that atoms more faithfully capture intrinsic representations of LLMs. Scaling experiments further reveal the link between SAEs size and recovery capacity. Overall, this work systematically introduces and validates Atoms Theory of LLMs, providing a theoretical framework for understanding internal representations and a foundation for mechanistic interpretability. Code available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）内部表示的基本单元的本质尚未定义，限制了对其机制的进一步理解。神经元或特征通常被视为这样的单元，但神经元存在多义性问题，而特征则面临不可靠重建和不稳定性的担忧。为解决这一问题，我们提出了原子理论，将这样的单元定义为原子。我们引入了原子内积（AIP）以校正表示偏移，正式定义了原子，并证明了原子满足限制等距性质（RIP）的条件，从而确保了原子集上的稳定稀疏表示，并将其与压缩感知相联系。在更强的条件下，我们进一步建立了稀疏表示的独特性和精确的$\ell_1$可恢复性，并提供保证，表明单层阈值激活稀疏自编码器（SAEs）能够可靠地识别原子。为了验证原子理论，我们在Gemma2-2B、Gemma2-9B和Llama3.1-8B上训练了阈值激活的SAEs，在平均层面上实现了99.9%的稀疏重建，并且超过99.8%的原子满足独特性条件，而神经元仅为0.5%，特征为68.2%，表明原子更忠实地捕获了LLMs的内在表示。扩展实验进一步揭示了SAEs规模与恢复能力之间的联系。总体而言，本工作系统地介绍了并验证了LLMs的原子理论，提供了一种理解内部表示的理论框架，并为机制可解释性提供了基础。代码可在以下链接获得。 

---
# IConv: Focusing on Local Variation with Channel Independent Convolution for Multivariate Time Series Forecasting 

**Title (ZH)**: IConv：基于通道独立卷积关注局部变化的多变量时间序列预测 

**Authors**: Gawon Lee, Hanbyeol Park, Minseop Kim, Dohee Kim, Hyerim Bae  

**Link**: [PDF](https://arxiv.org/pdf/2509.20783)  

**Abstract**: Real-world time-series data often exhibit non-stationarity, including changing trends, irregular seasonality, and residuals. In terms of changing trends, recently proposed multi-layer perceptron (MLP)-based models have shown excellent performance owing to their computational efficiency and ability to capture long-term dependency. However, the linear nature of MLP architectures poses limitations when applied to channels with diverse distributions, resulting in local variations such as seasonal patterns and residual components being ignored. However, convolutional neural networks (CNNs) can effectively incorporate these variations. To resolve the limitations of MLP, we propose combining them with CNNs. The overall trend is modeled using an MLP to consider long-term dependencies. The CNN uses diverse kernels to model fine-grained local patterns in conjunction with MLP trend predictions. To focus on modeling local variation, we propose IConv, a novel convolutional architecture that processes the temporal dependency channel independently and considers the inter-channel relationship through distinct layers. Independent channel processing enables the modeling of diverse local temporal dependencies and the adoption of a large kernel size. Distinct inter-channel considerations reduce computational cost. The proposed model is evaluated through extensive experiments on time-series datasets. The results reveal the superiority of the proposed method for multivariate time-series forecasting. 

**Abstract (ZH)**: 实时光序列数据通常表现出非平稳性，包括变化趋势、不规则季节性和残差。在变化趋势方面，最近提出的基于多层感知机（MLP）的模型由于其计算效率和捕捉长期依赖的能力表现出色。然而，MLP架构的线性特性在应用于分布各异的通道时存在局限性，导致季节模式和残差成分等局部变化被忽略。然而，卷积神经网络（CNNs）能够有效综合这些变化。为了解决MLP的局限性，我们提出将其与CNNs结合。整体趋势使用MLP建模，以考虑长期依赖性。CNN通过多种内核与MLP趋势预测结合，来建模细粒度的局部模式。为了专注于建模局部变化，我们提出了一种新颖的卷积架构IConv，独立处理时间依赖通道，并通过不同的层考虑通道间的相互关系。独立通道处理使得能够建模多样化的局部时间依赖性，并采用较大的内核尺寸。不同的通道交互考虑减少了计算成本。通过在时间序列数据集上进行广泛实验评估了所提出的方法。结果表明，所提出的方法在多变量时间序列预测中具有优越性。 

---
# CusEnhancer: A Zero-Shot Scene and Controllability Enhancement Method for Photo Customization via ResInversion 

**Title (ZH)**: CusEnhancer: 一种基于ResInversion的零样本场景和可控性增强方法用于照片个性化定制 

**Authors**: Maoye Ren, Praneetha Vaddamanu, Jianjin Xu, Fernando De la Torre Frade  

**Link**: [PDF](https://arxiv.org/pdf/2509.20775)  

**Abstract**: Recently remarkable progress has been made in synthesizing realistic human photos using text-to-image diffusion models. However, current approaches face degraded scenes, insufficient control, and suboptimal perceptual identity. We introduce CustomEnhancer, a novel framework to augment existing identity customization models. CustomEnhancer is a zero-shot enhancement pipeline that leverages face swapping techniques, pretrained diffusion model, to obtain additional representations in a zeroshot manner for encoding into personalized models. Through our proposed triple-flow fused PerGeneration approach, which identifies and combines two compatible counter-directional latent spaces to manipulate a pivotal space of personalized model, we unify the generation and reconstruction processes, realizing generation from three flows. Our pipeline also enables comprehensive training-free control over the generation process of personalized models, offering precise controlled personalization for them and eliminating the need for controller retraining for per-model. Besides, to address the high time complexity of null-text inversion (NTI), we introduce ResInversion, a novel inversion method that performs noise rectification via a pre-diffusion mechanism, reducing the inversion time by 129 times. Experiments demonstrate that CustomEnhancer reach SOTA results at scene diversity, identity fidelity, training-free controls, while also showing the efficiency of our ResInversion over NTI. The code will be made publicly available upon paper acceptance. 

**Abstract (ZH)**: Recent显著进展已在使用文本到图像扩散模型合成逼真的人像方面取得。然而，当前的方法面临场景退化、控制不足和感知身份欠佳的问题。我们引入了CustomEnhancer，一种新颖的身份增强框架，以增强现有的身份定制模型。CustomEnhancer 是一种零样本增强流程，利用面部替换技术及预训练扩散模型，在零样本条件下获取额外表示，并将其编码到个性化模型中。通过我们提出的三流融合PerGeneration方法，该方法识别并结合两个兼容的反向潜在空间，以操作个性化模型的关键空间，我们统一了生成和重构过程，实现了从三流生成。我们的流程还允许对个性化模型生成过程的全面无监督控制，提供精确的控制个性化，消除每个模型控制器重新培训的需要。此外，为了解决空文本反转（NTI）的时间复杂度过高的问题，我们引入了ResInversion，这是一种新颖的反转方法，通过预扩散机制进行噪声校正，将反转时间缩短了129倍。实验表明，CustomEnhancer 在场景多样性和身份保真度方面达到SOTA结果，同时展示了ResInversion相较于NTI的效率。论文接受后代码将公开。 

---
# Provenance Analysis of Archaeological Artifacts via Multimodal RAG Systems 

**Title (ZH)**: 考古文物多模态RAG系统中的溯源分析 

**Authors**: Tuo Zhang, Yuechun Sun, Ruiliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20769)  

**Abstract**: In this work, we present a retrieval-augmented generation (RAG)-based system for provenance analysis of archaeological artifacts, designed to support expert reasoning by integrating multimodal retrieval and large vision-language models (VLMs). The system constructs a dual-modal knowledge base from reference texts and images, enabling raw visual, edge-enhanced, and semantic retrieval to identify stylistically similar objects. Retrieved candidates are synthesized by the VLM to generate structured inferences, including chronological, geographical, and cultural attributions, alongside interpretive justifications. We evaluate the system on a set of Eastern Eurasian Bronze Age artifacts from the British Museum. Expert evaluation demonstrates that the system produces meaningful and interpretable outputs, offering scholars concrete starting points for analysis and significantly alleviating the cognitive burden of navigating vast comparative corpora. 

**Abstract (ZH)**: 基于检索增强生成（RAG）的考古 artifacts 起源分析系统：结合多模态检索和大型视觉语言模型支持专家推理 

---
# Measuring LLM Sensitivity in Transformer-based Tabular Data Synthesis 

**Title (ZH)**: 基于变压器的表格数据合成中LLM敏感性的度量 

**Authors**: Maria F. Davila R, Azizjon Turaev, Wolfram Wingerath  

**Link**: [PDF](https://arxiv.org/pdf/2509.20768)  

**Abstract**: Synthetic tabular data is used for privacy-preserving data sharing and data-driven model development. Its effectiveness, however, depends heavily on the used Tabular Data Synthesis (TDS) tool. Recent studies have shown that Transformer-based models outperform other state-of-the-art models such as Generative Adversarial Networks (GANs) and Diffusion models in terms of data quality. However, Transformer-based models also come with high computational costs, making them sometimes unfeasible for end users with prosumer hardware. This study presents a sensitivity assessment on how the choice of hyperparameters, such as number of layers or hidden dimension affects the quality of the resultant synthetic data and the computational performance. It is performed across two tools, GReaT and REaLTabFormer, evaluating 10 model setups that vary in architecture type and depth. We assess the sensitivity on three dimensions: runtime, machine learning (ML) utility, and similarity to real data distributions. Experiments were conducted on four real-world datasets. Our findings reveal that runtime is proportional to the number of hyperparameters, with shallower configurations completing faster. GReaT consistently achieves lower runtimes than REaLTabFormer, and only on the largest dataset they have comparable runtime. For small datasets, both tools achieve synthetic data with high utility and optimal similarity, but on larger datasets only REaLTabFormer sustains strong utility and similarity. As a result, REaLTabFormer with lightweight LLMs provides the best balance, since it preserves data quality while reducing computational requirements. Nonetheless, its runtime remains higher than that of GReaT and other TDS tools, suggesting that efficiency gains are possible but only up to a certain level. 

**Abstract (ZH)**: 合成表数据用于保护隐私的数据共享和数据驱动模型开发。然而，其效果很大程度上依赖于所使用的表格数据合成（TDS）工具。最近的研究表明，基于Transformer的模型在数据质量方面优于其他最先进的模型，如生成对抗网络（GANs）和扩散模型。然而，基于Transformer的模型也伴随着较高的计算成本，这有时使它们对拥有消费者级硬件的最终用户来说不可行。本研究对超参数（如层数或隐藏维度）的选择如何影响合成数据质量和计算性能进行了灵敏度评估。该评估在GReaT和REaLTabFormer两种工具上进行，评估了10种不同架构类型和深度的模型配置。我们在三个维度上评估灵敏度：运行时间、机器学习（ML）效用和与真实数据分布的相似性。实验在四个真实世界数据集上进行。我们的发现表明，运行时间与超参数的数量成正比，较浅的配置运行得更快。GReaT始终比REaLTabFormer的运行时间更短，仅在最大的数据集上它们的运行时间才可比较。对于小数据集，两种工具都能生成具有高效用和最优相似性的合成数据，但在大数据集上只有REaLTabFormer能够保持强大的效用和相似性。因此，REaLTabFormer结合轻量级的LLM提供了最佳平衡，因为它在保持数据质量的同时减少了计算需求。然而，其运行时间仍然高于GReaT和其他TDS工具，表明效率提升是可能的，但仅限于一定水平。 

---
# Seeing Through Words, Speaking Through Pixels: Deep Representational Alignment Between Vision and Language Models 

**Title (ZH)**: 透过词语看世界，通过像素说话：视觉模型与语言模型的深层表示对齐 

**Authors**: Zoe Wanying He, Sean Trott, Meenakshi Khosla  

**Link**: [PDF](https://arxiv.org/pdf/2509.20751)  

**Abstract**: Recent studies show that deep vision-only and language-only models--trained on disjoint modalities--nonetheless project their inputs into a partially aligned representational space. Yet we still lack a clear picture of where in each network this convergence emerges, what visual or linguistic cues support it, whether it captures human preferences in many-to-many image-text scenarios, and how aggregating exemplars of the same concept affects alignment. Here, we systematically investigate these questions. We find that alignment peaks in mid-to-late layers of both model types, reflecting a shift from modality-specific to conceptually shared representations. This alignment is robust to appearance-only changes but collapses when semantics are altered (e.g., object removal or word-order scrambling), highlighting that the shared code is truly semantic. Moving beyond the one-to-one image-caption paradigm, a forced-choice "Pick-a-Pic" task shows that human preferences for image-caption matches are mirrored in the embedding spaces across all vision-language model pairs. This pattern holds bidirectionally when multiple captions correspond to a single image, demonstrating that models capture fine-grained semantic distinctions akin to human judgments. Surprisingly, averaging embeddings across exemplars amplifies alignment rather than blurring detail. Together, our results demonstrate that unimodal networks converge on a shared semantic code that aligns with human judgments and strengthens with exemplar aggregation. 

**Abstract (ZH)**: 近期研究表明，尽管深度纯视觉模型和纯语言模型在各自独立的模态下训练，它们依然将输入投影到一个部分对齐的表示空间中。然而，我们仍然缺乏清晰的认识：这种收敛在每个网络中的哪个层次出现，哪些视觉或语言线索支持这一过程，它是否捕捉到了人类在一对多的图像-文本场景中的偏好，以及同一概念的示例聚合如何影响对齐。在这里，我们系统地研究了这些问题。我们发现这种对齐在两种模型类型的中到后期层中达到峰值，反映了从模态特定表示到概念共享表示的转变。这种对齐在仅外观变化时是稳健的，但在语义变化（例如，移除对象或词序打乱）时会崩溃，突显了共享代码真正的语义属性。超越一对一的图像-标题范式，“选一张图片”任务表明，人类对图像-标题匹配的偏好在所有视觉-语言模型对的嵌入空间中得到了镜像。当多个标题对应单个图像时，这一模式双向成立，证明模型捕捉到了类似人类判断的细微语义区别。令人惊讶的是，跨示例平均嵌入反而增强了对齐而非模糊细节。综上所述，我们的结果表明，单模网络收敛于一个与人类判断相一致的共享语义代码，并且随着示例聚合而增强。 

---
# Confidence-guided Refinement Reasoning for Zero-shot Question Answering 

**Title (ZH)**: 零样本问答中的自信引导细化推理 

**Authors**: Youwon Jang, Woo Suk Choi, Minjoon Jung, Minsu Lee, Byoung-Tak Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20750)  

**Abstract**: We propose Confidence-guided Refinement Reasoning (C2R), a novel training-free framework applicable to question-answering (QA) tasks across text, image, and video domains. C2R strategically constructs and refines sub-questions and their answers (sub-QAs), deriving a better confidence score for the target answer. C2R first curates a subset of sub-QAs to explore diverse reasoning paths, then compares the confidence scores of the resulting answer candidates to select the most reliable final answer. Since C2R relies solely on confidence scores derived from the model itself, it can be seamlessly integrated with various existing QA models, demonstrating consistent performance improvements across diverse models and benchmarks. Furthermore, we provide essential yet underexplored insights into how leveraging sub-QAs affects model behavior, specifically analyzing the impact of both the quantity and quality of sub-QAs on achieving robust and reliable reasoning. 

**Abstract (ZH)**: 我们提出了自信度指导的细化推理框架（C2R），这是一种适用于跨文本、图像和视频领域问答任务的新颖无训练框架。C2R 战略性地构建和细化子问题及其答案（子-QA），从而为目标答案获得更好的自信度评分。C2R 首先筛选出一组子-QA 以探索多样化的推理路径，然后比较结果答案候选的自信度评分以选择最可靠的最终答案。由于 C2R 仅依赖于模型自身生成的自信度评分，因此它可以无缝集成到各种现有的问答模型中，在多种模型和基准测试中表现出一致的性能提升。此外，我们还提供了关于利用子-QA 如何影响模型行为的关键但未充分探讨的见解，具体分析了子-QA 的数量和质量对实现稳健可靠的推理的影响。 

---
# AI-Enabled Crater-Based Navigation for Lunar Mapping 

**Title (ZH)**: 基于陨石坑的月球测绘人工智能导航 

**Authors**: Sofia McLeod, Chee-Kheng Chng, Matthew Rodda, Tat-Jun Chin  

**Link**: [PDF](https://arxiv.org/pdf/2509.20748)  

**Abstract**: Crater-Based Navigation (CBN) uses the ubiquitous impact craters of the Moon observed on images as natural landmarks to determine the six degrees of freedom pose of a spacecraft. To date, CBN has primarily been studied in the context of powered descent and landing. These missions are typically short in duration, with high-frequency imagery captured from a nadir viewpoint over well-lit terrain. In contrast, lunar mapping missions involve sparse, oblique imagery acquired under varying illumination conditions over potentially year-long campaigns, posing significantly greater challenges for pose estimation. We bridge this gap with STELLA - the first end-to-end CBN pipeline for long-duration lunar mapping. STELLA combines a Mask R-CNN-based crater detector, a descriptor-less crater identification module, a robust perspective-n-crater pose solver, and a batch orbit determination back-end. To rigorously test STELLA, we introduce CRESENT-365 - the first public dataset that emulates a year-long lunar mapping mission. Each of its 15,283 images is rendered from high-resolution digital elevation models with SPICE-derived Sun angles and Moon motion, delivering realistic global coverage, illumination cycles, and viewing geometries. Experiments on CRESENT+ and CRESENT-365 show that STELLA maintains metre-level position accuracy and sub-degree attitude accuracy on average across wide ranges of viewing angles, illumination conditions, and lunar latitudes. These results constitute the first comprehensive assessment of CBN in a true lunar mapping setting and inform operational conditions that should be considered for future missions. 

**Abstract (ZH)**: 基于陨石坑的导航（CBN）利用月球图像中普遍存在的陨石坑作为自然 landmarks 确定航天器的六自由度姿态。迄今为止，CBN 主要集中在有动力下降和着陆任务的研究中。这些任务通常持续时间较短，从向下的视角拍摄具有良好照明条件的地形的高频率图像。相比之下，月球制图任务涉及在潜在长达一年的活动中获取稀疏、偏斜视角的图像，并且这些图像在不同的光照条件下拍摄，这给姿态估计带来了更大的挑战。我们通过 STELLA —— 首个用于长时间月球制图的端到端 CBN 管道，填补了这一空白。STELLA 结合了基于 Mask R-CNN 的陨石坑检测器、无描述子陨石坑识别模块、鲁棒的透视-n-陨石坑姿态求解器以及批量轨道确定后端。为了严格测试 STELLA，我们引入了 CRESENT-365 —— 首个模拟一年长期月球制图任务的公共数据集。该数据集的每张图像均由高分辨率数字地形模型渲染，使用 SPICE 计算的太阳角度和月球运动，提供真实的全球覆盖、照明循环和视野几何结构。在 CRESENT+ 和 CRESENT-365 上的实验表明，STELLA 在广角视野、光照条件和月球纬度变化范围内平均保持米级的位置精度和亚度的姿态精度。这些结果是首次全面评估 CBN 在真正月球制图环境中的表现，并为未来的任务应该考虑的操作条件提供了信息。 

---
# Imagining Design Workflows in Agentic AI Futures 

**Title (ZH)**: 在自主人工智能未来中的设计工作流程 imagining 设想 

**Authors**: Samangi Wadinambiarachchi, Jenny Waycott, Yvonne Rogers, Greg Wadley  

**Link**: [PDF](https://arxiv.org/pdf/2509.20731)  

**Abstract**: As designers become familiar with Generative AI, a new concept is emerging: Agentic AI. While generative AI produces output in response to prompts, agentic AI systems promise to perform mundane tasks autonomously, potentially freeing designers to focus on what they love: being creative. But how do designers feel about integrating agentic AI systems into their workflows? Through design fiction, we investigated how designers want to interact with a collaborative agentic AI platform. Ten professional designers imagined and discussed collaborating with an AI agent to organise inspiration sources and ideate. Our findings highlight the roles AI agents can play in supporting designers, the division of authority between humans and AI, and how designers' intent can be explained to AI agents beyond prompts. We synthesise our findings into a conceptual framework that identifies authority distribution among humans and AI agents and discuss directions for utilising AI agents in future design workflows. 

**Abstract (ZH)**: 随着设计师对生成式AI的熟悉，一种新的概念正在兴起：自主式AI。虽然生成式AI根据提示产生输出，自主式AI系统承诺能够自主执行繁琐任务， potentially 有可能使设计师能够专注于他们热爱的创造性工作。但设计师对将自主式AI系统整合到他们的工作流程中持何种态度呢？通过设计虚构，我们研究了设计师希望如何与协作式自主式AI平台互动。十名专业设计师构想了与AI代理合作整理灵感来源和创意的过程。我们的研究结果突显了AI代理在支持设计师方面可以扮演的角色、人类与AI之间的权威划分，以及设计师意图如何超越提示向AI代理进行解释。我们将研究结果综合成一个概念框架，确定了人类和AI代理之间的权威分配，并讨论了在未来的设计工作流程中利用AI代理的方向。 

---
# RobotDancing: Residual-Action Reinforcement Learning Enables Robust Long-Horizon Humanoid Motion Tracking 

**Title (ZH)**: RobotDancing: 基于残差动作强化学习的robust长时间 humanoid 运动跟踪 

**Authors**: Zhenguo Sun, Yibo Peng, Yuan Meng, Xukun Li, Bo-Sheng Huang, Zhenshan Bing, Xinlong Wang, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2509.20717)  

**Abstract**: Long-horizon, high-dynamic motion tracking on humanoids remains brittle because absolute joint commands cannot compensate model-plant mismatch, leading to error accumulation. We propose RobotDancing, a simple, scalable framework that predicts residual joint targets to explicitly correct dynamics discrepancies. The pipeline is end-to-end--training, sim-to-sim validation, and zero-shot sim-to-real--and uses a single-stage reinforcement learning (RL) setup with a unified observation, reward, and hyperparameter configuration. We evaluate primarily on Unitree G1 with retargeted LAFAN1 dance sequences and validate transfer on H1/H1-2. RobotDancing can track multi-minute, high-energy behaviors (jumps, spins, cartwheels) and deploys zero-shot to hardware with high motion tracking quality. 

**Abstract (ZH)**: 长时程、高动态 humanoid 运动跟踪依然脆弱，因为绝对关节命令无法补偿模型与实际系统的差异，导致误差累积。我们提出 RobotDancing，一种简单可扩展的框架，预测残差关节目标以明确修正动力学差异。该框架为端到端设计，包括从训练到模拟验证再到零样本模拟到现实的流程，并采用统一观测、奖励和超参数配置的一阶段强化学习设置。我们主要基于 Unitree G1 和重新目标跟踪的 LAFAN1 舞蹈序列进行评估，并在 H1/H1-2 上验证了其迁移性。RobotDancing 可以跟踪多分钟、高能量行为（跳跃、旋转、侧手翻）并以高质量部署到硬件。 

---
# Beyond the Individual: Introducing Group Intention Forecasting with SHOT Dataset 

**Title (ZH)**: 超越个体：引入基于SHOT数据集的群体意图预测 

**Authors**: Ruixu Zhang, Yuran Wang, Xinyi Hu, Chaoyu Mai, Wenxuan Liu, Danni Xu, Xian Zhong, Zheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20715)  

**Abstract**: Intention recognition has traditionally focused on individual intentions, overlooking the complexities of collective intentions in group settings. To address this limitation, we introduce the concept of group intention, which represents shared goals emerging through the actions of multiple individuals, and Group Intention Forecasting (GIF), a novel task that forecasts when group intentions will occur by analyzing individual actions and interactions before the collective goal becomes apparent. To investigate GIF in a specific scenario, we propose SHOT, the first large-scale dataset for GIF, consisting of 1,979 basketball video clips captured from 5 camera views and annotated with 6 types of individual attributes. SHOT is designed with 3 key characteristics: multi-individual information, multi-view adaptability, and multi-level intention, making it well-suited for studying emerging group intentions. Furthermore, we introduce GIFT (Group Intention ForecasTer), a framework that extracts fine-grained individual features and models evolving group dynamics to forecast intention emergence. Experimental results confirm the effectiveness of SHOT and GIFT, establishing a strong foundation for future research in group intention forecasting. The dataset is available at this https URL. 

**Abstract (ZH)**: 群体意图识别历来专注于个体意图，忽视了团体情境中集体意图的复杂性。为了弥补这一局限，我们引入了群体意图的概念，该概念代表通过多个个体行为产生的共同目标，并提出了新颖的任务——群体意图预测（Group Intention Forecasting, GIF），该任务通过分析个体行为和交互来预测集体目标产生前的意图时间点。为了在特定场景中探究GIF，我们提出了SHOT，这是首个用于GIF的大规模数据集，包含1,979个篮球视频片段，从5个视角捕捉并标注了6种个体属性。SHOT具有三个关键特性：多个体信息、多视角适应性和多层次意图，使其非常适合研究新兴的群体意图。此外，我们还提出了GIFT（Group Intention Forecaster）框架，该框架提取细粒度的个体特征并建模演变中的群体动态以预测意图的产生。实验结果证实了SHOT和GIFT的有效性，为群体意图预测的未来研究奠定了坚实的基础。该数据集可从以下链接获取：https://doi.org/10.5281/zenodo.4545789 

---
# Joint Flow Trajectory Optimization For Feasible Robot Motion Generation from Video Demonstrations 

**Title (ZH)**: 从视频示范中生成可行机器人运动的联合流动轨迹优化 

**Authors**: Xiaoxiang Dong, Matthew Johnson-Roberson, Weiming Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2509.20703)  

**Abstract**: Learning from human video demonstrations offers a scalable alternative to teleoperation or kinesthetic teaching, but poses challenges for robot manipulators due to embodiment differences and joint feasibility constraints. We address this problem by proposing the Joint Flow Trajectory Optimization (JFTO) framework for grasp pose generation and object trajectory imitation under the video-based Learning-from-Demonstration (LfD) paradigm. Rather than directly imitating human hand motions, our method treats demonstrations as object-centric guides, balancing three objectives: (i) selecting a feasible grasp pose, (ii) generating object trajectories consistent with demonstrated motions, and (iii) ensuring collision-free execution within robot kinematics. To capture the multimodal nature of demonstrations, we extend flow matching to $\SE(3)$ for probabilistic modeling of object trajectories, enabling density-aware imitation that avoids mode collapse. The resulting optimization integrates grasp similarity, trajectory likelihood, and collision penalties into a unified differentiable objective. We validate our approach in both simulation and real-world experiments across diverse real-world manipulation tasks. 

**Abstract (ZH)**: 基于视频演示的人机学习为操纵器提供了规模化替代远程操作或力觉示教的方案，但面对体化差异和关节可行性约束，带来了挑战。为此，我们提出了基于视频演示学习（LfD）框架下的关节流轨迹优化（JFTO）方法，用于抓取姿态生成和物体轨迹模仿，而不是直接模仿人的手部动作，而是将演示视为以物体为中心的指南，平衡以下三个目标：（i）选择可行的抓取姿态，（ii）生成与示范运动一致的物体轨迹，（iii）确保在机器人运动学约束下的无碰撞执行。为了捕获演示的多模态性质，我们将流匹配扩展到$\SE(3)$，以概率建模物体轨迹，使得模仿具有密度感知能力，避免模式崩溃。最终的优化将抓取相似性、轨迹似然性和碰撞惩罚整合到一个统一的不同iable目标中。我们在多种真实世界的操作任务的模拟和实际实验中验证了该方法。 

---
# Incorporating LLM Embeddings for Variation Across the Human Genome 

**Title (ZH)**: 将大型语言模型嵌入用于人类基因组的变异性研究 

**Authors**: Hongqian Niu, Jordan Bryan, Xihao Li, Didong Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.20702)  

**Abstract**: Recent advances in large language model (LLM) embeddings have enabled powerful representations for biological data, but most applications to date focus only on gene-level information. We present one of the first systematic frameworks to generate variant-level embeddings across the entire human genome. Using curated annotations from FAVOR, ClinVar, and the GWAS Catalog, we constructed semantic text descriptions for 8.9 billion possible variants and generated embeddings at three scales: 1.5 million HapMap3+MEGA variants, ~90 million imputed UK Biobank variants, and ~9 billion all possible variants. Embeddings were produced with both OpenAI's text-embedding-3-large and the open-source Qwen3-Embedding-0.6B models. Baseline experiments demonstrate high predictive accuracy for variant properties, validating the embeddings as structured representations of genomic variation. We outline two downstream applications: embedding-informed hypothesis testing by extending the Frequentist And Bayesian framework to genome-wide association studies, and embedding-augmented genetic risk prediction that enhances standard polygenic risk scores. These resources, publicly available on Hugging Face, provide a foundation for advancing large-scale genomic discovery and precision medicine. 

**Abstract (ZH)**: 最近在大型语言模型（LLM）嵌入方面的进展使生物数据的强大表示成为可能，但目前大多数应用仅侧重于基因水平的信息。我们提出了第一个系统框架之一，用于在整个人类基因组中生成变体级别的嵌入。利用来自FAVAR、ClinVar和GWAS Catalog的精心注释数据，我们为89亿个可能的变体构建了语义文本描述，并生成了三个层次的嵌入：150万个HapMap3+MEGA变体、约9亿个推断的UK Biobank变体以及约90亿个所有可能的变体。嵌入使用OpenAI的text-embedding-3-large和开源的Qwen3-Embedding-0.6B模型生成。基线实验表明，变体属性预测准确性很高，验证了嵌入作为基因组变异的结构化表示的有效性。我们概述了两个下游应用：通过扩展Frequentist And Bayesian框架用于全基因组关联研究的嵌入指导的假设检验，以及增强标准多基因风险评分的嵌入增强遗传风险预测。这些资源在Hugging Face上公开提供，为大规模基因组发现和精准医学奠定了基础。 

---
# Learning to Align Molecules and Proteins: A Geometry-Aware Approach to Binding Affinity 

**Title (ZH)**: 学习对齐分子和蛋白质：一种几何 Awareness 方法用于结合亲和力预测 

**Authors**: Mohammadsaleh Refahi, Bahrad A. Sokhansanj, James R. Brown, Gail Rosen  

**Link**: [PDF](https://arxiv.org/pdf/2509.20693)  

**Abstract**: Accurate prediction of drug-target binding affinity can accelerate drug discovery by prioritizing promising compounds before costly wet-lab screening. While deep learning has advanced this task, most models fuse ligand and protein representations via simple concatenation and lack explicit geometric regularization, resulting in poor generalization across chemical space and time. We introduce FIRM-DTI, a lightweight framework that conditions molecular embeddings on protein embeddings through a feature-wise linear modulation (FiLM) layer and enforces metric structure with a triplet loss. An RBF regression head operating on embedding distances yields smooth, interpretable affinity predictions. Despite its modest size, FIRM-DTI achieves state-of-the-art performance on the Therapeutics Data Commons DTI-DG benchmark, as demonstrated by an extensive ablation study and out-of-domain evaluation. Our results underscore the value of conditioning and metric learning for robust drug-target affinity prediction. 

**Abstract (ZH)**: 精确预测药物-目标结合亲和力可以通过在昂贵的湿实验筛选之前优先选择有前途的化合物来加速药物发现。虽然深度学习促进了这一任务的发展，但大多数模型通过简单的连接方式融合配体和蛋白质表示，并缺乏显式的几何正则化，导致在化学空间和时间上的泛化性能较差。我们引入了FIRM-DTI，这是一种轻量级框架，通过特征 wise 线性调制（FiLM）层条件化分子嵌入在蛋白质嵌入上，并使用三元组损失强制满足度量结构。基于嵌入距离的RBF回归头产生平滑且可解释的亲和力预测。尽管规模较小，但FIRM-DTI在治疗数据共用DTI-DG基准测试中取得了最先进的性能，这得到了详尽的消融研究和跨域评估的验证。我们的结果强调了条件化和度量学习对于稳健的药物-目标亲和力预测的重要性。 

---
# Addressing Gradient Misalignment in Data-Augmented Training for Robust Speech Deepfake Detection 

**Title (ZH)**: 数据增强训练中梯度错位的应对方法以提升鲁棒性语音Deepfake检测 

**Authors**: Duc-Tuan Truong, Tianchi Liu, Junjie Li, Ruijie Tao, Kong Aik Lee, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2509.20682)  

**Abstract**: In speech deepfake detection (SDD), data augmentation (DA) is commonly used to improve model generalization across varied speech conditions and spoofing attacks. However, during training, the backpropagated gradients from original and augmented inputs may misalign, which can result in conflicting parameter updates. These conflicts could hinder convergence and push the model toward suboptimal solutions, thereby reducing the benefits of DA. To investigate and address this issue, we design a dual-path data-augmented (DPDA) training framework with gradient alignment for SDD. In our framework, each training utterance is processed through two input paths: one using the original speech and the other with its augmented version. This design allows us to compare and align their backpropagated gradient directions to reduce optimization conflicts. Our analysis shows that approximately 25% of training iterations exhibit gradient conflicts between the original inputs and their augmented counterparts when using RawBoost augmentation. By resolving these conflicts with gradient alignment, our method accelerates convergence by reducing the number of training epochs and achieves up to an 18.69% relative reduction in Equal Error Rate on the In-the-Wild dataset compared to the baseline. 

**Abstract (ZH)**: 基于语音深仿真的双路径数据增强训练框架及其梯度对齐 

---
# Efficient Construction of Implicit Surface Models From a Single Image for Motion Generation 

**Title (ZH)**: 从单张图像高效构建隐式曲面模型用于运动生成 

**Authors**: Wei-Teng Chu, Tianyi Zhang, Matthew Johnson-Roberson, Weiming Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2509.20681)  

**Abstract**: Implicit representations have been widely applied in robotics for obstacle avoidance and path planning. In this paper, we explore the problem of constructing an implicit distance representation from a single image. Past methods for implicit surface reconstruction, such as \emph{NeuS} and its variants generally require a large set of multi-view images as input, and require long training times. In this work, we propose Fast Image-to-Neural Surface (FINS), a lightweight framework that can reconstruct high-fidelity surfaces and SDF fields based on a single or a small set of images. FINS integrates a multi-resolution hash grid encoder with lightweight geometry and color heads, making the training via an approximate second-order optimizer highly efficient and capable of converging within a few seconds. Additionally, we achieve the construction of a neural surface requiring only a single RGB image, by leveraging pre-trained foundation models to estimate the geometry inherent in the image. Our experiments demonstrate that under the same conditions, our method outperforms state-of-the-art baselines in both convergence speed and accuracy on surface reconstruction and SDF field estimation. Moreover, we demonstrate the applicability of FINS for robot surface following tasks and show its scalability to a variety of benchmark datasets. 

**Abstract (ZH)**: 基于单张图像构建隐式距离表示的快速图像到神经表面框架 

---
# QAMO: Quality-aware Multi-centroid One-class Learning For Speech Deepfake Detection 

**Title (ZH)**: QAMO：质量感知多中心点一类学习用于语音深度假信息检测 

**Authors**: Duc-Tuan Truong, Tianchi Liu, Ruijie Tao, Junjie Li, Kong Aik Lee, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2509.20679)  

**Abstract**: Recent work shows that one-class learning can detect unseen deepfake attacks by modeling a compact distribution of bona fide speech around a single centroid. However, the single-centroid assumption can oversimplify the bona fide speech representation and overlook useful cues, such as speech quality, which reflects the naturalness of the speech. Speech quality can be easily obtained using existing speech quality assessment models that estimate it through Mean Opinion Score. In this paper, we propose QAMO: Quality-Aware Multi-Centroid One-Class Learning for speech deepfake detection. QAMO extends conventional one-class learning by introducing multiple quality-aware centroids. In QAMO, each centroid is optimized to represent a distinct speech quality subspaces, enabling better modeling of intra-class variability in bona fide speech. In addition, QAMO supports a multi-centroid ensemble scoring strategy, which improves decision thresholding and reduces the need for quality labels during inference. With two centroids to represent high- and low-quality speech, our proposed QAMO achieves an equal error rate of 5.09% in In-the-Wild dataset, outperforming previous one-class and quality-aware systems. 

**Abstract (ZH)**: 基于质量意识的多中心点一类学习语音深度假检测（QAMO） 

---
# Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport 

**Title (ZH)**: 双谱OT：基于对称意识最优传输的数据集比较 

**Authors**: Annabel Ma, Kaiying Hou, David Alvarez-Melis, Melanie Weber  

**Link**: [PDF](https://arxiv.org/pdf/2509.20678)  

**Abstract**: Optimal transport (OT) is a widely used technique in machine learning, graphics, and vision that aligns two distributions or datasets using their relative geometry. In symmetry-rich settings, however, OT alignments based solely on pairwise geometric distances between raw features can ignore the intrinsic coherence structure of the data. We introduce Bispectral Optimal Transport, a symmetry-aware extension of discrete OT that compares elements using their representation using the bispectrum, a group Fourier invariant that preserves all signal structure while removing only the variation due to group actions. Empirically, we demonstrate that the transport plans computed with Bispectral OT achieve greater class preservation accuracy than naive feature OT on benchmark datasets transformed with visual symmetries, improving the quality of meaningful correspondences that capture the underlying semantic label structure in the dataset while removing nuisance variation not affecting class or content. 

**Abstract (ZH)**: Bispectral Optimal Transport 

---
# Understanding Mode Switching in Human-AI Collaboration: Behavioral Insights and Predictive Modeling 

**Title (ZH)**: 理解人类-人工智能协作中的模式切换：行为洞察与预测建模 

**Authors**: Avinash Ajit Nargund, Arthur Caetano, Kevin Yang, Rose Yiwei Liu, Philip Tezaur, Kriteen Shrestha, Qisen Pan, Tobias Höllerer, Misha Sra  

**Link**: [PDF](https://arxiv.org/pdf/2509.20666)  

**Abstract**: Human-AI collaboration is typically offered in one of two of user control levels: guidance, where the AI provides suggestions and the human makes the final decision, and delegation, where the AI acts autonomously within user-defined constraints. Systems that integrate both modes, common in robotic surgery or driving assistance, often overlook shifts in user preferences within a task in response to factors like evolving trust, decision complexity, and perceived control. In this work, we investigate how users dynamically switch between higher and lower levels of control during a sequential decision-making task. Using a hand-and-brain chess setup, participants either selected a piece and the AI decided how it moved (brain mode), or the AI selected a piece and the participant decided how it moved (hand mode). We collected over 400 mode-switching decisions from eight participants, along with gaze, emotional state, and subtask difficulty data. Statistical analysis revealed significant differences in gaze patterns and subtask complexity prior to a switch and in the quality of the subsequent move. Based on these results, we engineered behavioral and task-specific features to train a lightweight model that predicted control level switches ($F1 = 0.65$). The model performance suggests that real-time behavioral signals can serve as a complementary input alongside system-driven mode-switching mechanisms currently used. We complement our quantitative results with qualitative factors that influence switching including perceived AI ability, decision complexity, and level of control, identified from post-game interview analysis. The combined behavioral and modeling insights can help inform the design of shared autonomy systems that need dynamic, subtask-level control switches aligned with user intent and evolving task demands. 

**Abstract (ZH)**: 人类与人工智能的合作通常提供两种用户控制水平：指导模式下，人工智能提供建议而人类做出最终决定；自主模式下，人工智能在用户定义的约束内自主行动。本研究探讨了用户在序列决策任务中动态切换高、低控制水平的方式。通过手脑象棋设置，参与者在“脑模式”下选择棋子而由人工智能决定移动方式，在“手模式”下由人工智能选择棋子而参与者决定移动方式。研究收集了八名参与者超过400个模式切换决策，以及凝视行为、情绪状态和子任务难度数据。统计分析显示，切换前凝视模式和子任务复杂度存在显著差异，随后的移动质量也有显著变化。基于这些结果，我们构建了行为和任务特定特征以训练轻量级模型，预测控制水平切换（F1值为0.65）。模型性能表明实时行为信号可以作为系统驱动模式切换机制的补充输入。我们通过后游戏访谈分析补充了影响切换的定性因素，包括感知的人工智能能力、决策复杂性和控制水平。这些结合的行为和建模洞察有助于设计动态、适应子任务需求的共享自主系统，以更好地满足用户意图与不断变化的任务需求。 

---
# Look Before you Leap: Estimating LLM Benchmark Scores from Descriptions 

**Title (ZH)**: 未雨绸缪：从描述中估计大语言模型基准分数 

**Authors**: Jungsoo Park, Ethan Mendes, Gabriel Stanovsky, Alan Ritter  

**Link**: [PDF](https://arxiv.org/pdf/2509.20645)  

**Abstract**: Progress in large language models is constrained by an evaluation bottleneck: build a benchmark, evaluate models and settings, then iterate. We therefore ask a simple question: can we forecast outcomes before running any experiments? We study text-only performance forecasting: estimating a model's score from a redacted task description and intended configuration, with no access to dataset instances. To support systematic study, we curate PRECOG, a corpus of redacted description-performance pairs spanning diverse tasks, domains, and metrics. Experiments show the task is challenging but feasible: models equipped with a retrieval module that excludes source papers achieve moderate prediction performance with well-calibrated uncertainty, reaching mean absolute error as low as 8.7 on the Accuracy subset at high-confidence thresholds. Our analysis indicates that stronger reasoning models engage in diverse, iterative querying, whereas current open-source models lag and often skip retrieval or gather evidence with limited diversity. We further test a zero-leakage setting, forecasting on newly released datasets or experiments before their papers are indexed, where GPT-5 with built-in web search still attains nontrivial prediction accuracy. Overall, our corpus and analyses offer an initial step toward open-ended anticipatory evaluation, supporting difficulty estimation and smarter experiment prioritization. 

**Abstract (ZH)**: 大型语言模型进展受限于评估瓶颈：构建基准、评估模型和设置，然后迭代改进。因此我们提出一个简单问题：我们能否在运行任何实验之前预测结果？我们研究仅文本的性能预测：根据红acted的任务描述和预期配置估算模型的得分，不访问数据集实例。为了支持系统研究，我们编制了PRECOG，这是一个跨不同任务、领域和指标的红acted描述-性能配对语料库。实验显示该任务具有挑战性但可行：配备排除源论文检索模块的模型在高置信度阈值下于准确度子集上达到最低8.7的平均绝对误差，并表现出良好的不确定性校准。我们的分析表明，更强的推理模型会进行多样性的迭代查询，而现有的开源模型则滞后，经常跳过检索或收集有限多样性的证据。我们进一步测试了零泄漏设置，在论文未被索引前对新发布的数据集或实验进行预测，结果显示内置网页搜索的GPT-5仍能获得非平凡的预测精度。总体而言，我们的语料库和分析为进一步开放式的前瞻性评估奠定了初步基础，支持难度估计和更明智的实验优先级排序。 

---
# A Framework for Rapidly Developing and Deploying Protection Against Large Language Model Attacks 

**Title (ZH)**: 一种快速开发和部署大型语言模型攻击防护的框架 

**Authors**: Adam Swanda, Amy Chang, Alexander Chen, Fraser Burch, Paul Kassianik, Konstantin Berlin  

**Link**: [PDF](https://arxiv.org/pdf/2509.20639)  

**Abstract**: The widespread adoption of Large Language Models (LLMs) has revolutionized AI deployment, enabling autonomous and semi-autonomous applications across industries through intuitive language interfaces and continuous improvements in model development. However, the attendant increase in autonomy and expansion of access permissions among AI applications also make these systems compelling targets for malicious attacks. Their inherent susceptibility to security flaws necessitates robust defenses, yet no known approaches can prevent zero-day or novel attacks against LLMs. This places AI protection systems in a category similar to established malware protection systems: rather than providing guaranteed immunity, they minimize risk through enhanced observability, multi-layered defense, and rapid threat response, supported by a threat intelligence function designed specifically for AI-related threats.
Prior work on LLM protection has largely evaluated individual detection models rather than end-to-end systems designed for continuous, rapid adaptation to a changing threat landscape. We present a production-grade defense system rooted in established malware detection and threat intelligence practices. Our platform integrates three components: a threat intelligence system that turns emerging threats into protections; a data platform that aggregates and enriches information while providing observability, monitoring, and ML operations; and a release platform enabling safe, rapid detection updates without disrupting customer workflows. Together, these components deliver layered protection against evolving LLM threats while generating training data for continuous model improvement and deploying updates without interrupting production. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的广泛应用已经革新人工智能部署，通过直观的语言界面和模型开发的持续改进，使各行各业能够实现自主和半自主的应用。然而，AI应用程序自主性和访问权限的增加也使其成为恶意攻击的诱饵。系统固有的安全缺陷使其需要强大的防御措施，但目前尚无方法能完全阻止针对LLMs的零日或新型攻击。这使得AI保护系统处于与传统恶意软件保护系统相似的类别：它们不是提供绝对的免疫，而是通过增强可观测性、多层次防御和快速威胁响应，特别是通过专门设计的威胁情报功能，来最大限度地降低风险。 

---
# Learning Terrain-Specialized Policies for Adaptive Locomotion in Challenging Environments 

**Title (ZH)**: 学习适用于挑战性环境的terrain-specialized策略以实现自适应运动控制 

**Authors**: Matheus P. Angarola, Francisco Affonso, Marcelo Becker  

**Link**: [PDF](https://arxiv.org/pdf/2509.20635)  

**Abstract**: Legged robots must exhibit robust and agile locomotion across diverse, unstructured terrains, a challenge exacerbated under blind locomotion settings where terrain information is unavailable. This work introduces a hierarchical reinforcement learning framework that leverages terrain-specialized policies and curriculum learning to enhance agility and tracking performance in complex environments. We validated our method on simulation, where our approach outperforms a generalist policy by up to 16% in success rate and achieves lower tracking errors as the velocity target increases, particularly on low-friction and discontinuous terrains, demonstrating superior adaptability and robustness across mixed-terrain scenarios. 

**Abstract (ZH)**: 腿式机器人必须在多样化的无结构地形上表现出 robust 和敏捷的运动能力，而在地形信息不可用的盲运动情况下，这一挑战更加严峻。本研究引入了一种分层强化学习框架，该框架利用地形专业化策略和 Curriculum 学习来提高复杂环境中敏捷性和跟踪性能。我们在模拟中验证了该方法，结果显示，在成功率和跟踪误差方面，我们的方法分别比通用策略高出 16% 和在高速目标下表现更好，特别是在低摩擦和不连续地形上，显示出更强的适应性和鲁棒性，适用于混合地形场景。 

---
# Recidivism and Peer Influence with LLM Text Embeddings in Low Security Correctional Facilities 

**Title (ZH)**: 低安全级别矫正设施中LLM文本嵌入的再犯与同伴影响研究 

**Authors**: Shanjukta Nath, Jiwon Hong, Jae Ho Chang, Keith Warren, Subhadeep Paul  

**Link**: [PDF](https://arxiv.org/pdf/2509.20634)  

**Abstract**: We find AI embeddings obtained using a pre-trained transformer-based Large Language Model (LLM) of 80,000-120,000 written affirmations and correction exchanges among residents in low-security correctional facilities to be highly predictive of recidivism. The prediction accuracy is 30\% higher with embedding vectors than with only pre-entry covariates. However, since the text embedding vectors are high-dimensional, we perform Zero-Shot classification of these texts to a low-dimensional vector of user-defined classes to aid interpretation while retaining the predictive power. To shed light on the social dynamics inside the correctional facilities, we estimate peer effects in these LLM-generated numerical representations of language with a multivariate peer effect model, adjusting for network endogeneity. We develop new methodology and theory for peer effect estimation that accommodate sparse networks, multivariate latent variables, and correlated multivariate outcomes. With these new methods, we find significant peer effects in language usage for interaction and feedback. 

**Abstract (ZH)**: 我们发现使用预训练的基于变换器的大语言模型（LLM），对80,000-120,000份低安全级别矫正设施中居民的书面肯定陈述和修正交流进行AI嵌入后，这些嵌入能够高度预测重犯概率。与仅使用前期协变量相比，使用嵌入向量的预测准确率提高了30%。由于文本嵌入向量高维，我们通过对这些文本进行零样本分类，将它们降至用户定义类别的低维向量，以辅助解释同时保持预测能力。为了揭示矫正设施内的社会动态，我们使用多元同伴影响模型估计LLM生成的语言数值表示中的同伴效果，同时调整网络内生性。我们开发了新的同伴效果估计方法和理论，以应对稀疏网络、多元潜在变量和多元相关结果的问题。利用这些新方法，我们发现语言使用在互动和反馈中的显著同伴效应。 

---
# Personalized Federated Dictionary Learning for Modeling Heterogeneity in Multi-site fMRI Data 

**Title (ZH)**: 多中心fMRI数据中异质性建模的个性化 Federated字典学习 

**Authors**: Yipu Zhang, Chengshuo Zhang, Ziyu Zhou, Gang Qu, Hao Zheng, Yuping Wang, Hui Shen, Hongwen Deng  

**Link**: [PDF](https://arxiv.org/pdf/2509.20627)  

**Abstract**: Data privacy constraints pose significant challenges for large-scale neuroimaging analysis, especially in multi-site functional magnetic resonance imaging (fMRI) studies, where site-specific heterogeneity leads to non-independent and identically distributed (non-IID) data. These factors hinder the development of generalizable models. To address these challenges, we propose Personalized Federated Dictionary Learning (PFedDL), a novel federated learning framework that enables collaborative modeling across sites without sharing raw data. PFedDL performs independent dictionary learning at each site, decomposing each site-specific dictionary into a shared global component and a personalized local component. The global atoms are updated via federated aggregation to promote cross-site consistency, while the local atoms are refined independently to capture site-specific variability, thereby enhancing downstream analysis. Experiments on the ABIDE dataset demonstrate that PFedDL outperforms existing methods in accuracy and robustness across non-IID datasets. 

**Abstract (ZH)**: 数据隐私约束为大规模神经成像分析带来了重大挑战，尤其是在多站点功能磁共振成像(fMRI)研究中，站点特异性异质性导致了非独立且非同分布（non-IID）数据。这些因素阻碍了通用模型的发展。为应对这些挑战，我们提出了一种新型联邦学习框架——个性化联邦字典学习（PFedDL），该框架能够在无需共享原始数据的情况下实现跨站点的协作建模。PFedDL 在每个站点独立进行字典学习，将每个站点特定的字典分解为共享的全局成分和个性化的局部成分。全局原子通过联邦聚合进行更新，以促进跨站点的一致性，而局部原子则独立更新，以捕捉站点特异性变异，从而增强后续分析。在ABIDE数据集上的实验表明，PFedDL 在非IID数据集上的准确性和鲁棒性优于现有方法。 

---
# FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Models 

**Title (ZH)**: FS-DFM: 快速且准确的少步扩散语言模型长文本生成 

**Authors**: Amin Karimi Monsefi, Nikhil Bhendawade, Manuel Rafael Ciosici, Dominic Culver, Yizhe Zhang, Irina Belousova  

**Link**: [PDF](https://arxiv.org/pdf/2509.20624)  

**Abstract**: Autoregressive language models (ARMs) deliver strong likelihoods, but are inherently serial: they generate one token per forward pass, which limits throughput and inflates latency for long sequences. Diffusion Language Models (DLMs) parallelize across positions and thus appear promising for language generation, yet standard discrete diffusion typically needs hundreds to thousands of model evaluations to reach high quality, trading serial depth for iterative breadth. We introduce FS-DFM, Few-Step Discrete Flow-Matching. A discrete flow-matching model designed for speed without sacrificing quality. The core idea is simple: make the number of sampling steps an explicit parameter and train the model to be consistent across step budgets, so one big move lands where many small moves would. We pair this with a reliable update rule that moves probability in the right direction without overshooting, and with strong teacher guidance distilled from long-run trajectories. Together, these choices make few-step sampling stable, accurate, and easy to control. On language modeling benchmarks, FS-DFM with 8 sampling steps achieves perplexity parity with a 1,024-step discrete-flow baseline for generating 1,024 tokens using a similar-size model, delivering up to 128 times faster sampling and corresponding latency/throughput gains. 

**Abstract (ZH)**: Few-Step Discrete Flow-Matching for Fast and Accurate Language Generation 

---
# MMG: Mutual Information Estimation via the MMSE Gap in Diffusion 

**Title (ZH)**: MMG: 基于扩散的MMSE间隙估计互信息 

**Authors**: Longxuan Yu, Xing Shi, Xianghao Kong, Tong Jia, Greg Ver Steeg  

**Link**: [PDF](https://arxiv.org/pdf/2509.20609)  

**Abstract**: Mutual information (MI) is one of the most general ways to measure relationships between random variables, but estimating this quantity for complex systems is challenging. Denoising diffusion models have recently set a new bar for density estimation, so it is natural to consider whether these methods could also be used to improve MI estimation. Using the recently introduced information-theoretic formulation of denoising diffusion models, we show the diffusion models can be used in a straightforward way to estimate MI. In particular, the MI corresponds to half the gap in the Minimum Mean Square Error (MMSE) between conditional and unconditional diffusion, integrated over all Signal-to-Noise-Ratios (SNRs) in the noising process. Our approach not only passes self-consistency tests but also outperforms traditional and score-based diffusion MI estimators. Furthermore, our method leverages adaptive importance sampling to achieve scalable MI estimation, while maintaining strong performance even when the MI is high. 

**Abstract (ZH)**: 基于去噪扩散模型的互信息估计 

---
# Experience Deploying Containerized GenAI Services at an HPC Center 

**Title (ZH)**: 在国家超级计算中心部署容器化生成式AI服务的经验 

**Authors**: Angel M. Beltre, Jeff Ogden, Kevin Pedretti  

**Link**: [PDF](https://arxiv.org/pdf/2509.20603)  

**Abstract**: Generative Artificial Intelligence (GenAI) applications are built from specialized components -- inference servers, object storage, vector and graph databases, and user interfaces -- interconnected via web-based APIs. While these components are often containerized and deployed in cloud environments, such capabilities are still emerging at High-Performance Computing (HPC) centers. In this paper, we share our experience deploying GenAI workloads within an established HPC center, discussing the integration of HPC and cloud computing environments. We describe our converged computing architecture that integrates HPC and Kubernetes platforms running containerized GenAI workloads, helping with reproducibility. A case study illustrates the deployment of the Llama Large Language Model (LLM) using a containerized inference server (vLLM) across both Kubernetes and HPC platforms using multiple container runtimes. Our experience highlights practical considerations and opportunities for the HPC container community, guiding future research and tool development. 

**Abstract (ZH)**: 生成型人工智能（GenAI）应用基于专门组件构建——推理服务器、对象存储、向量和图数据库以及用户界面——通过基于Web的API相互连接。尽管这些组件通常是容器化的并在云环境中部署，但在高性能计算（HPC）中心，这些能力仍处于早期阶段。在本文中，我们分享在已有的HPC中心部署GenAI工作负载的经验，讨论了HPC和云计算环境的集成。我们描述了一种结合HPC和Kubernetes平台的计算架构，以容器化形式运行GenAI工作负载，有助于提高可重复性。一个案例研究展示了如何使用容器化推理服务器（vLLM）在Kubernetes和HPC平台之间部署Llama大型语言模型（LLM）并使用多个容器运行时。我们的经验为HPC容器社区提供了实用的考虑因素和机会，指导未来的研究和工具开发。 

---
# An LLM-based Agentic Framework for Accessible Network Control 

**Title (ZH)**: 基于LLM的赋能框架以实现无障碍网络控制 

**Authors**: Samuel Lin, Jiawei Zhou, Minlan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20600)  

**Abstract**: Traditional approaches to network management have been accessible only to a handful of highly-trained network operators with significant expert knowledge. This creates barriers for lay users to easily manage their networks without resorting to experts. With recent development of powerful large language models (LLMs) for language comprehension, we design a system to make network management accessible to a broader audience of non-experts by allowing users to converse with networks in natural language. To effectively leverage advancements in LLMs, we propose an agentic framework that uses an intermediate representation to streamline configuration across diverse vendor equipment, retrieves the network state from memory in real-time, and provides an interface for external feedback. We also conduct pilot studies to collect real user data of natural language utterances for network control, and present a visualization interface to facilitate dialogue-driven user interaction and enable large-scale data collection for future development. Preliminary experiments validate the effectiveness of our proposed system components with LLM integration on both synthetic and real user utterances. Through our data collection and visualization efforts, we pave the way for more effective use of LLMs and democratize network control for everyday users. 

**Abstract (ZH)**: 传统的网络管理方法仅对少量受过高度训练的网络运维专家开放，这为普通用户在无需求助专家的情况下轻松管理网络设置了障碍。借助近期强大语言理解大型语言模型（LLMs）的发展，我们设计了一个系统，通过让普通用户以自然语言与网络交流，使网络管理对非专家用户群体更加便捷。为了有效利用LLMs的进展，我们提出了一种代理框架，使用中间表示简化跨不同供应商设备的配置过程，实时从内存中检索网络状态，并提供外部反馈接口。我们还开展了试点研究，收集了网络控制中自然语言指令的真实用户数据，并提供了一个可视化界面，以促进基于对话的用户交互，并为未来的大规模数据收集提供支持。初步实验验证了在合成和真实用户指令上集成LLMs的系统组件的有效性。通过我们的数据收集和可视化工作，我们为更有效地利用LLMs并使网络控制普及化铺平了道路。 

---
# Every Character Counts: From Vulnerability to Defense in Phishing Detection 

**Title (ZH)**: 每一个字符都重要：从脆弱性到防骗检测 

**Authors**: Maria Chiper, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20589)  

**Abstract**: Phishing attacks targeting both organizations and individuals are becoming an increasingly significant threat as technology advances. Current automatic detection methods often lack explainability and robustness in detecting new phishing attacks. In this work, we investigate the effectiveness of character-level deep learning models for phishing detection, which can provide both robustness and interpretability. We evaluate three neural architectures adapted to operate at the character level, namely CharCNN, CharGRU, and CharBiLSTM, on a custom-built email dataset, which combines data from multiple sources. Their performance is analyzed under three scenarios: (i) standard training and testing, (ii) standard training and testing under adversarial attacks, and (iii) training and testing with adversarial examples. Aiming to develop a tool that operates as a browser extension, we test all models under limited computational resources. In this constrained setup, CharGRU proves to be the best-performing model across all scenarios. All models show vulnerability to adversarial attacks, but adversarial training substantially improves their robustness. In addition, by adapting the Gradient-weighted Class Activation Mapping (Grad-CAM) technique to character-level inputs, we are able to visualize which parts of each email influence the decision of each model. Our open-source code and data is released at this https URL. 

**Abstract (ZH)**: 针对组织和个人的钓鱼攻击随着技术的发展变得日益成为一个重要的威胁。现有的自动检测方法在检测新型钓鱼攻击时往往缺乏解释性和鲁棒性。本文研究了基于字符级的深度学习模型在钓鱼检测中的有效性，该模型能够提供解释性和鲁棒性。我们评估了三种适应字符级操作的神经架构，即CharCNN、CharGRU和CharBiLSTM，在一个自建的邮件数据集上，该数据集结合了多个来源的数据。我们在三种情景下分析了它们的表现：（i）标准训练和测试，（ii）标准训练和测试下的对抗攻击，（iii）使用对抗样本进行训练和测试。旨在开发一个作为浏览器扩展工具，我们在有限的计算资源下测试了所有模型。在这一受限设置下，CharGRU在所有情景中表现最佳。所有模型对对抗攻击都显示出脆弱性，但对抗训练极大地提高了它们的鲁棒性。此外，通过将Gradient-weighted Class Activation Mapping（Grad-CAM）技术适应于字符级输入，我们能够可视化每个邮件中的哪些部分影响每个模型的决策。我们的开源代码和数据可以在以下链接访问：this https URL。 

---
# Hierarchical Resolution Transformers: A Wavelet-Inspired Architecture for Multi-Scale Language Understanding 

**Title (ZH)**: 分层解析变换器：一种小波启发的多尺度语言理解架构 

**Authors**: Ayan Sar, Sampurna Roy, Kanav Gupta, Anurag Kaushish, Tanupriya Choudhury, Abhijit Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.20581)  

**Abstract**: Transformer architectures have achieved state-of-the-art performance across natural language tasks, yet they fundamentally misrepresent the hierarchical nature of human language by processing text as flat token sequences. This results in quadratic computational cost, weak computational cost, weak compositional generalization, and inadequate discourse-level modeling. We propose Hierarchical Resolution Transformer (HRT), a novel wavelet-inspired neural architecture that processes language simultaneously across multiple resolutions, from characters to discourse-level units. HRT constructs a multi-resolution attention, enabling bottom-up composition and top-down contextualization. By employing exponential sequence reduction across scales, HRT achieves O(nlogn) complexity, offering significant efficiency improvements over standard transformers. We evaluated HRT on a diverse suite of benchmarks, including GLUE, SuperGLUE, Long Range Arena, and WikiText-103, and results demonstrated that HRT outperforms standard transformer baselines by an average of +3.8% on GLUE, +4.5% on SuperGLUE, and +6.1% on Long Range Arena, while reducing memory usage by 42% and inference latency by 37% compared to BERT and GPT style models of similar parameter count. Ablation studies confirm the effectiveness of cross-resolution attention and scale-specialized modules, showing that each contributes independently to both efficiency and accuracy. Our findings establish HRT as the first architecture to align computational structure with the hierarchical organization of human language, demonstrating that multi-scale, wavelet-inspired processing yields both theoretical efficiency gains and practical improvements in language understanding. 

**Abstract (ZH)**: 基于小波的层次解析变换器在自然语言任务中实现了最先进的性能，但本质上错误地将人类语言的层次结构表示为扁平的标记序列，导致二次计算成本、计算效率低、组合泛化能力弱以及语篇层次建模不足。为此，我们提出了层次解析变换器（HRT），一种新颖的小波启发式神经架构，能够同时从字符到语篇层面单位对语言进行多尺度处理。HRT 构建了多尺度注意机制，支持自底向上的组合和自顶向下的语境化。通过在不同尺度上应用指数级序列减少，HRT 实现了O(nlogn)复杂度，相比标准变换器提供了显著的效率改进。我们在 GLUE、SuperGLUE、Long Range Arena 和 WikiText-103 等多种基准测试上评估了 HRT，结果表明 HRT 在 GLUE 上平均优于标准变换器基线 3.8%，在 SuperGLUE 上优于 4.5%，在 Long Range Arena 上优于 6.1%，同时内存使用量减少 42%，推理延迟减少 37%，与类似参数量的 BERT 和 GPT 风格模型相比。消融研究确认了跨尺度注意和尺度专业化模块的有效性，表明它们分别独立地提高了效率和准确性。我们的研究确立了 HRT 作为首个使计算结构与人类语言层级结构相一致的架构，证明了多尺度、小波启发式处理既具备理论效率优势，又在语言理解方面实现了实际改进。 

---
# Dynamic Reasoning Chains through Depth-Specialized Mixture-of-Experts in Transformer Architectures 

**Title (ZH)**: 深度专业化专家混合在变压器架构中的动态推理链 

**Authors**: Sampurna Roy, Ayan Sar, Anurag Kaushish, Kanav Gupta, Tanupriya Choudhury, Abhijit Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.20577)  

**Abstract**: Contemporary transformer architectures apply identical processing depth to all inputs, creating inefficiencies and limiting reasoning quality. Simple factual queries are subjected to the same multilayered computation as complex logical problems, wasting resources while constraining deep inference. To overcome this, we came up with a concept of Dynamic Reasoning Chains through Depth Specialised Mixture of Experts (DS-MoE), a modular framework that extends the Mixture of Experts paradigm from width-based to depth specialised computation. DS-MoE introduces expert modules optimised for distinct reasoning depths, shallow pattern recognition, compositional reasoning, logical inference, memory integration, and meta-cognitive supervision. A learned routing network dynamically assembles custom reasoning chains, activating only the necessary experts to match input complexity. The dataset on which we trained and evaluated DS-MoE is on The Pile, an 800GB corpus covering diverse domains such as scientific papers, legal texts, programming code, and web content, enabling systematic assessment across reasoning depths. Experimental results demonstrate that DS-MoE achieves up to 16 per cent computational savings and 35 per cent faster inference compared to uniform-depth transformers, while delivering 2.8 per cent higher accuracy on complex multi-step reasoning benchmarks. Furthermore, routing decisions yield interpretable reasoning chains, enhancing transparency and scalability. These findings establish DS-MoE as a significant advancement in adaptive neural architectures, demonstrating that depth-specialised modular processing can simultaneously improve efficiency, reasoning quality, and interpretability in large-scale language models. 

**Abstract (ZH)**: Dynamic Reasoning Chains through Depth Specialised Mixture of Experts 

---
# MechStyle: Augmenting Generative AI with Mechanical Simulation to Create Stylized and Structurally Viable 3D Models 

**Title (ZH)**: MechStyle: 通过机械仿真增强生成式AI以创建具风格且结构可行的3D模型 

**Authors**: Faraz Faruqi, Amira Abdel-Rahman, Leandra Tejedor, Martin Nisser, Jiaji Li, Vrushank Phadnis, Varun Jampani, Neil Gershenfeld, Megan Hofmann, Stefanie Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2509.20571)  

**Abstract**: Recent developments in Generative AI enable creators to stylize 3D models based on text prompts. These methods change the 3D model geometry, which can compromise the model's structural integrity once fabricated. We present MechStyle, a system that enables creators to stylize 3D printable models while preserving their structural integrity. MechStyle accomplishes this by augmenting the Generative AI-based stylization process with feedback from a Finite Element Analysis (FEA) simulation. As the stylization process modifies the geometry to approximate the desired style, feedback from the FEA simulation reduces modifications to regions with increased stress. We evaluate the effectiveness of FEA simulation feedback in the augmented stylization process by comparing three stylization control strategies. We also investigate the time efficiency of our approach by comparing three adaptive scheduling strategies. Finally, we demonstrate MechStyle's user interface that allows users to generate stylized and structurally viable 3D models and provide five example applications. 

**Abstract (ZH)**: Recent developments in生成式AI使创作者能够基于文本提示对3D模型进行风格化。这些方法会改变3D模型的几何结构，一旦制造成型可能导致模型的结构完整性受损。我们提出了MechStyle系统，能够在保持3D打印模型结构完整性的同时对其进行风格化。MechStyle通过结合基于生成式AI的风格化过程和有限元分析（FEA）仿真反馈来实现这一目标。随着风格化过程修改几何结构以接近目标风格，FEA仿真反馈减少了对应力增加区域的修改。我们通过比较三种风格化控制策略评估了FEA仿真反馈在增强风格化过程中的有效性。我们还通过比较三种自适应调度策略研究了该方法的时间效率。最后，我们展示了MechStyle的用户界面，允许用户生成风格化且结构有效的3D模型，并提供了五个示例应用。 

---
# PIRF: Physics-Informed Reward Fine-Tuning for Diffusion Models 

**Title (ZH)**: PIRF：物理导向的奖励微调方法用于扩散模型 

**Authors**: Mingze Yuan, Pengfei Jin, Na Li, Quanzheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.20570)  

**Abstract**: Diffusion models have demonstrated strong generative capabilities across scientific domains, but often produce outputs that violate physical laws. We propose a new perspective by framing physics-informed generation as a sparse reward optimization problem, where adherence to physical constraints is treated as a reward signal. This formulation unifies prior approaches under a reward-based paradigm and reveals a shared bottleneck: reliance on diffusion posterior sampling (DPS)-style value function approximations, which introduce non-negligible errors and lead to training instability and inference inefficiency. To overcome this, we introduce Physics-Informed Reward Fine-tuning (PIRF), a method that bypasses value approximation by computing trajectory-level rewards and backpropagating their gradients directly. However, a naive implementation suffers from low sample efficiency and compromised data fidelity. PIRF mitigates these issues through two key strategies: (1) a layer-wise truncated backpropagation method that leverages the spatiotemporally localized nature of physics-based rewards, and (2) a weight-based regularization scheme that improves efficiency over traditional distillation-based methods. Across five PDE benchmarks, PIRF consistently achieves superior physical enforcement under efficient sampling regimes, highlighting the potential of reward fine-tuning for advancing scientific generative modeling. 

**Abstract (ZH)**: 基于物理约束的生成奖励微调方法：克服扩散模型的物理守恒问题 

---
# SwasthLLM: a Unified Cross-Lingual, Multi-Task, and Meta-Learning Zero-Shot Framework for Medical Diagnosis Using Contrastive Representations 

**Title (ZH)**: SwasthLLM：一种基于对比表示的统一跨语言、多任务和元学习的零样本医学诊断框架 

**Authors**: Ayan Sar, Pranav Singh Puri, Sumit Aich, Tanupriya Choudhury, Abhijit Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.20567)  

**Abstract**: In multilingual healthcare environments, automatic disease diagnosis from clinical text remains a challenging task due to the scarcity of annotated medical data in low-resource languages and the linguistic variability across populations. This paper proposes SwasthLLM, a unified, zero-shot, cross-lingual, and multi-task learning framework for medical diagnosis that operates effectively across English, Hindi, and Bengali without requiring language-specific fine-tuning. At its core, SwasthLLM leverages the multilingual XLM-RoBERTa encoder augmented with a language-aware attention mechanism and a disease classification head, enabling the model to extract medically relevant information regardless of the language structure. To align semantic representations across languages, a Siamese contrastive learning module is introduced, ensuring that equivalent medical texts in different languages produce similar embeddings. Further, a translation consistency module and a contrastive projection head reinforce language-invariant representation learning. SwasthLLM is trained using a multi-task learning strategy, jointly optimizing disease classification, translation alignment, and contrastive learning objectives. Additionally, we employ Model-Agnostic Meta-Learning (MAML) to equip the model with rapid adaptation capabilities for unseen languages or tasks with minimal data. Our phased training pipeline emphasizes robust representation alignment before task-specific fine-tuning. Extensive evaluation shows that SwasthLLM achieves high diagnostic performance, with a test accuracy of 97.22% and an F1-score of 97.17% in supervised settings. Crucially, in zero-shot scenarios, it attains 92.78% accuracy on Hindi and 73.33% accuracy on Bengali medical text, demonstrating strong generalization in low-resource contexts. 

**Abstract (ZH)**: 多语言医疗环境中，基于临床文本的自动疾病诊断仍然是一个具有挑战性的任务，原因在于低资源语言标注医疗数据的稀缺性和跨群体语言变异性。本文提出了一种统一的、零样本、跨语言和多任务学习框架SwasthLLM，该框架在不针对特定语言进行微调的情况下，有效应用于英语、 Hindi 和 Bengali 语言的医疗诊断。SwasthLLM 在其核心处利用了多语言 XLM-RoBERTa 编码器，结合了语言意识注意力机制和疾病分类头，使模型能够提取与语言结构无关的医疗相关信息。为了在不同语言之间对齐语义表示，引入了双胞胎对比学习模块，确保不同语言中的等效医疗文本生成相似的嵌入表示。此外，语言不变性表示学习还通过翻译一致性模块和对比投影头得到了加强。SwasthLLM 采用了多任务学习策略进行训练，同时优化疾病分类、翻译对齐和对比学习目标。此外，我们利用模型无感知元学习（MAML）赋予模型在最少数据下对未见过的语言或任务进行快速适应的能力。训练管道采用分阶段的方式，重点在于稳健表示对齐后再进行任务特定的微调。广泛评估表明，SwasthLLM 在监督设置中达到了高诊断性能，测试准确率为 97.22%，F1 分数为 97.17%。在零样本场景中，SwasthLLM 在 Hindi 医疗文本上的准确率为 92.78%，在 Bengali 医疗文本上的准确率为 73.33%，显示出了在低资源环境中的强大泛化能力。 

---
# Perspectra: Choosing Your Experts Enhances Critical Thinking in Multi-Agent Research Ideation 

**Title (ZH)**: Perspectra：选择你的专家能增强多智能体研究构想中的批判性思维 

**Authors**: Yiren Liu, Viraj Shah, Sangho Suh, Pao Siangliulue, Tal August, Yun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20553)  

**Abstract**: Recent advances in multi-agent systems (MAS) enable tools for information search and ideation by assigning personas to agents. However, how users can effectively control, steer, and critically evaluate collaboration among multiple domain-expert agents remains underexplored. We present Perspectra, an interactive MAS that visualizes and structures deliberation among LLM agents via a forum-style interface, supporting @-mention to invite targeted agents, threading for parallel exploration, with a real-time mind map for visualizing arguments and rationales. In a within-subjects study with 18 participants, we compared Perspectra to a group-chat baseline as they developed research proposals. Our findings show that Perspectra significantly increased the frequency and depth of critical-thinking behaviors, elicited more interdisciplinary replies, and led to more frequent proposal revisions than the group chat condition. We discuss implications for designing multi-agent tools that scaffold critical thinking by supporting user control over multi-agent adversarial discourse. 

**Abstract (ZH)**: Recent advances in 多代理系统（MAS）通过为代理分配人设，使信息搜索和创意生成成为可能，然而，用户如何有效地控制、引导和批判性评估多个领域专家代理之间的协作仍然缺乏探索。我们提出了Perspectra，这是一款通过论坛界面可视化和结构化LLM代理之间商讨的交互式MAS，支持@提及邀请指定代理、线性回复促进并行探索，并通过实时思维导图展示论点和推理。在包含18名参与者的单因素实验中，我们将Perspectra与群聊基线进行了比较，观察它们在开发研究提案时的表现。我们的发现表明，与群聊相比，Perspectra显著增加了批判性思维行为的频率和深度，引发了更多跨学科的回复，并导致提案修订的频率更高。我们讨论了通过支持用户对多代理对抗性讨论的控制来构建促进批判性思维的多代理工具的意义。 

---
# GraspFactory: A Large Object-Centric Grasping Dataset 

**Title (ZH)**: GraspFactory: 一个大型对象中心抓取数据集 

**Authors**: Srinidhi Kalgundi Srinivas, Yash Shukla, Adam Arnold, Sachin Chitta  

**Link**: [PDF](https://arxiv.org/pdf/2509.20550)  

**Abstract**: Robotic grasping is a crucial task in industrial automation, where robots are increasingly expected to handle a wide range of objects. However, a significant challenge arises when robot grasping models trained on limited datasets encounter novel objects. In real-world environments such as warehouses or manufacturing plants, the diversity of objects can be vast, and grasping models need to generalize to this diversity. Training large, generalizable robot-grasping models requires geometrically diverse datasets. In this paper, we introduce GraspFactory, a dataset containing over 109 million 6-DoF grasps collectively for the Franka Panda (with 14,690 objects) and Robotiq 2F-85 grippers (with 33,710 objects). GraspFactory is designed for training data-intensive models, and we demonstrate the generalization capabilities of one such model trained on a subset of GraspFactory in both simulated and real-world settings. The dataset and tools are made available for download at this https URL. 

**Abstract (ZH)**: 机器人抓取是工业自动化中的关键任务，其中机器人被期望处理种类繁多的物体。然而，当基于有限数据集训练的机器人抓取模型遇到新型物体时，会面临重大挑战。在仓库或制造工厂等实际环境中，物体的多样性很高，抓取模型需要能够泛化到这种多样性。训练大规模且泛化能力强的机器人抓取模型需要几何上多样的数据集。本文介绍了GraspFactory数据集，包含超过109百万个6-DoF抓取姿态，涵盖了Franka Panda机械手（14,690个物体）和Robotiq 2F-85夹爪（33,710个物体）。GraspFactory旨在用于训练数据密集型模型，并在模拟和实际环境中展示了基于GraspFactory子集训练的模型的泛化能力。数据集及工具可从此网址下载：this https URL。 

---
# Understanding and Improving Adversarial Robustness of Neural Probabilistic Circuits 

**Title (ZH)**: 理解并改进神经概率电路的对抗鲁棒性 

**Authors**: Weixin Chen, Han Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.20549)  

**Abstract**: Neural Probabilistic Circuits (NPCs), a new class of concept bottleneck models, comprise an attribute recognition model and a probabilistic circuit for reasoning. By integrating the outputs from these two modules, NPCs produce compositional and interpretable predictions. While offering enhanced interpretability and high performance on downstream tasks, the neural-network-based attribute recognition model remains a black box. This vulnerability allows adversarial attacks to manipulate attribute predictions by introducing carefully crafted subtle perturbations to input images, potentially compromising the final predictions. In this paper, we theoretically analyze the adversarial robustness of NPC and demonstrate that it only depends on the robustness of the attribute recognition model and is independent of the robustness of the probabilistic circuit. Moreover, we propose RNPC, the first robust neural probabilistic circuit against adversarial attacks on the recognition module. RNPC introduces a novel class-wise integration for inference, ensuring a robust combination of outputs from the two modules. Our theoretical analysis demonstrates that RNPC exhibits provably improved adversarial robustness compared to NPC. Empirical results on image classification tasks show that RNPC achieves superior adversarial robustness compared to existing concept bottleneck models while maintaining high accuracy on benign inputs. 

**Abstract (ZH)**: 神经概率电路（NPCs）：一种新的概念瓶颈模型，结合了属性识别模型和推理的概率电路，通过整合这两个模块的输出，产生组合性和可解释性的预测。尽管NPC提供了增强的可解释性和下游任务上的高性能，基于神经网络的属性识别模型仍保持黑盒子特性。这种脆弱性允许通过在输入图像中引入精心设计的微妙扰动来进行对抗攻击，从而操控属性预测，最终可能影响最终预测的准确性。在本文中，我们从理论上分析了NPC的对抗鲁棒性，并证明它仅依赖于属性识别模型的鲁棒性，而不依赖于概率电路的鲁棒性。此外，我们提出了第一个针对识别模块对抗攻击的鲁棒神经概率电路（RNPC）。RNPC引入了一种新的类别内整合方法，确保了两个模块输出的稳健组合。我们的理论分析表明，RNPC在对抗鲁棒性方面显着优于NPC。图像分类任务上的实验结果表明，RNPC在保持对良性输入的高准确性的同时，实现了优于现有概念瓶颈模型的对抗鲁棒性。 

---
# InstructVTON: Optimal Auto-Masking and Natural-Language-Guided Interactive Style Control for Inpainting-Based Virtual Try-On 

**Title (ZH)**: 基于 inpainting 的虚拟试穿：最优自动遮罩和自然语言引导的交互式风格控制 

**Authors**: Julien Han, Shuwen Qiu, Qi Li, Xingzi Xu, Mehmet Saygin Seyfioglu, Kavosh Asadi, Karim Bouyarmane  

**Link**: [PDF](https://arxiv.org/pdf/2509.20524)  

**Abstract**: We present InstructVTON, an instruction-following interactive virtual try-on system that allows fine-grained and complex styling control of the resulting generation, guided by natural language, on single or multiple garments. A computationally efficient and scalable formulation of virtual try-on formulates the problem as an image-guided or image-conditioned inpainting task. These inpainting-based virtual try-on models commonly use a binary mask to control the generation layout. Producing a mask that yields desirable result is difficult, requires background knowledge, might be model dependent, and in some cases impossible with the masking-based approach (e.g. trying on a long-sleeve shirt with "sleeves rolled up" styling on a person wearing long-sleeve shirt with sleeves down, where the mask will necessarily cover the entire sleeve). InstructVTON leverages Vision Language Models (VLMs) and image segmentation models for automated binary mask generation. These masks are generated based on user-provided images and free-text style instructions. InstructVTON simplifies the end-user experience by removing the necessity of a precisely drawn mask, and by automating execution of multiple rounds of image generation for try-on scenarios that cannot be achieved with masking-based virtual try-on models alone. We show that InstructVTON is interoperable with existing virtual try-on models to achieve state-of-the-art results with styling control. 

**Abstract (ZH)**: InstructVTON：一种遵循指令的交互式虚拟试穿系统，通过自然语言指导实现细粒度和复杂的设计控制 

---
# CHOIR: A Chatbot-mediated Organizational Memory Leveraging Communication in University Research Labs 

**Title (ZH)**: CHOIR：一种通过交流促进大学研究实验室组织记忆的聊天机器人 eksplora: 一种通过交流促进大学研究实验室组织记忆的聊天机器人 

**Authors**: Sangwook Lee, Adnan Abbas, Yan Chen, Young-Ho Kim, Sang Won Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.20512)  

**Abstract**: University research labs often rely on chat-based platforms for communication and project management, where valuable knowledge surfaces but is easily lost in message streams. Documentation can preserve knowledge, but it requires ongoing maintenance and is challenging to navigate. Drawing on formative interviews that revealed organizational memory challenges in labs, we designed CHOIR, an LLM-based chatbot that supports organizational memory through four key functions: document-grounded Q&A, Q&A sharing for follow-up discussion, knowledge extraction from conversations, and AI-assisted document updates. We deployed CHOIR in four research labs for one month (n=21), where the lab members asked 107 questions and lab directors updated documents 38 times in the organizational memory. Our findings reveal a privacy-awareness tension: questions were asked privately, limiting directors' visibility into documentation gaps. Students often avoided contribution due to challenges in generalizing personal experiences into universal documentation. We contribute design implications for privacy-preserving awareness and supporting context-specific knowledge documentation. 

**Abstract (ZH)**: 大学研究实验室往往依赖基于聊天的平台进行沟通和项目管理，其中有价值的知识容易在消息流中丢失。文档可以保存知识，但需要持续维护且导航困难。借鉴形成性访谈中揭示的研究实验室组织记忆挑战，我们设计了CHOIR，一个基于语言模型的聊天机器人，通过四大关键功能支持组织记忆：文档指导的问答、问答分享供后续讨论、从对话中提取知识，以及AI辅助的文档更新。我们在四个研究实验室进行了一项为期一个月的部署（n=21），实验室成员提出了107个问题，实验室主任更新了组织记忆中的文档38次。我们的研究发现隐私意识与透明度之间存在张力：问题往往在私密环境下提出，限制了领导者对文档空白的可见性。学生因难以将个人经验泛化为通用文档而往往避免贡献。我们提出了隐私保护意识的设计启示，并支持情境特定的知识文档。 

---
# Complexity-Driven Policy Optimization 

**Title (ZH)**: 驱动复杂性优化的策略优化 

**Authors**: Luca Serfilippi, Giorgio Franceschelli, Antonio Corradi, Mirco Musolesi  

**Link**: [PDF](https://arxiv.org/pdf/2509.20509)  

**Abstract**: Policy gradient methods often balance exploitation and exploration via entropy maximization. However, maximizing entropy pushes the policy towards a uniform random distribution, which represents an unstructured and sometimes inefficient exploration strategy. In this work, we propose replacing the entropy bonus with a more robust complexity bonus. In particular, we adopt a measure of complexity, defined as the product of Shannon entropy and disequilibrium, where the latter quantifies the distance from the uniform distribution. This regularizer encourages policies that balance stochasticity (high entropy) with structure (high disequilibrium), guiding agents toward regimes where useful, non-trivial behaviors can emerge. Such behaviors arise because the regularizer suppresses both extremes, e.g., maximal disorder and complete order, creating pressure for agents to discover structured yet adaptable strategies. Starting from Proximal Policy Optimization (PPO), we introduce Complexity-Driven Policy Optimization (CDPO), a new learning algorithm that replaces entropy with complexity. We show empirically across a range of discrete action space tasks that CDPO is more robust to the choice of the complexity coefficient than PPO is with the entropy coefficient, especially in environments requiring greater exploration. 

**Abstract (ZH)**: 基于复杂度的策略优化方法：一种替代熵增益的稳健策略优化算法 

---
# MARS: toward more efficient multi-agent collaboration for LLM reasoning 

**Title (ZH)**: MARS:向更高效的多Agent协作推理方向迈进 

**Authors**: Xiao Wang, Jia Wang, Yijie Wang, Pengtao Dang, Sha Cao, Chi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20502)  

**Abstract**: Large language models (LLMs) have achieved impressive results in natural language understanding, yet their reasoning capabilities remain limited when operating as single agents. Multi-Agent Debate (MAD) has been proposed to address this limitation by enabling collaborative reasoning among multiple models in a round-table debate manner. While effective, MAD introduces substantial computational overhead due to the number of agents involved and the frequent communication required. In this paper, we propose MARS (Multi-Agent Review System), a role-based collaboration framework inspired by the review process. In MARS, an author agent generates an initial solution, reviewer agents provide decisions and comments independently, and a meta-reviewer integrates the feedback to make the final decision and guide further revision. This design enhances reasoning quality while avoiding costly reviewer-to-reviewer interactions, thereby controlling token consumption and inference time. We compared MARS with both MAD and other state-of-the-art reasoning strategies across multiple benchmarks. Extensive experiments with different LLMs show that MARS matches the accuracy of MAD while reducing both token usage and inference time by approximately 50\%. Code is available at this https URL. 

**Abstract (ZH)**: 多代理评审系统（MARS）：一种基于角色的合作框架 

---
# Boosting Zero-Shot VLN via Abstract Obstacle Map-Based Waypoint Prediction with TopoGraph-and-VisitInfo-Aware Prompting 

**Title (ZH)**: 基于抽象障碍地图的航点预测以及拓扑图和到访信息感知提示增强零样本视觉语言导航 

**Authors**: Boqi Li, Siyuan Li, Weiyi Wang, Anran Li, Zhong Cao, Henry X. Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20499)  

**Abstract**: With the rapid progress of foundation models and robotics, vision-language navigation (VLN) has emerged as a key task for embodied agents with broad practical applications. We address VLN in continuous environments, a particularly challenging setting where an agent must jointly interpret natural language instructions, perceive its surroundings, and plan low-level actions. We propose a zero-shot framework that integrates a simplified yet effective waypoint predictor with a multimodal large language model (MLLM). The predictor operates on an abstract obstacle map, producing linearly reachable waypoints, which are incorporated into a dynamically updated topological graph with explicit visitation records. The graph and visitation information are encoded into the prompt, enabling reasoning over both spatial structure and exploration history to encourage exploration and equip MLLM with local path planning for error correction. Extensive experiments on R2R-CE and RxR-CE show that our method achieves state-of-the-art zero-shot performance, with success rates of 41% and 36%, respectively, outperforming prior state-of-the-art methods. 

**Abstract (ZH)**: 基于基础模型和机器人技术的快速发展，视觉-语言导航（VLN）已成为具备广泛实用前景的体感Agent的关键任务。我们研究在连续环境中进行VLN，这是一种特别具有挑战性的设置，其中代理必须联合解释自然语言指令、感知周围环境并规划低层动作。我们提出了一种零样本框架，该框架结合了一个简化但有效的 waypoints 预测器和多模态大型语言模型（MLLM）。预测器基于抽象障碍地图工作，生成线性可达的waypoints，并将其整合到一个动态更新的拓扑图中，该图包含明确的访问记录。图和访问信息被编码到提示中，以支持对空间结构和探索历史的推理，从而鼓励探索，并为MLLM提供局部路径规划以进行错误校正。在R2R-CE和RxR-CE上的广泛实验表明，我们的方法实现了最先进的零样本性能，分别取得了41%和36%的成功率，优于先前的最佳方法。 

---
# AI-Specific Code Smells: From Specification to Detection 

**Title (ZH)**: AI-specific 代码气味：从规范到检测 

**Authors**: Brahim Mahmoudi, Naouel Moha, Quentin Stievenert, Florent Avellaneda  

**Link**: [PDF](https://arxiv.org/pdf/2509.20491)  

**Abstract**: The rise of Artificial Intelligence (AI) is reshaping how software systems are developed and maintained. However, AI-based systems give rise to new software issues that existing detection tools often miss. Among these, we focus on AI-specific code smells, recurring patterns in the code that may indicate deeper problems such as unreproducibility, silent failures, or poor model generalization. We introduce SpecDetect4AI, a tool-based approach for the specification and detection of these code smells at scale. This approach combines a high-level declarative Domain-Specific Language (DSL) for rule specification with an extensible static analysis tool that interprets and detects these rules for AI-based systems. We specified 22 AI-specific code smells and evaluated SpecDetect4AI on 826 AI-based systems (20M lines of code), achieving a precision of 88.66% and a recall of 88.89%, outperforming other existing detection tools. Our results show that SpecDetect4AI supports the specification and detection of AI-specific code smells through dedicated rules and can effectively analyze large AI-based systems, demonstrating both efficiency and extensibility (SUS 81.7/100). 

**Abstract (ZH)**: 人工智能（AI）的兴起正在重塑软件系统的开发和维护方式。然而，基于AI的系统带来了现有检测工具往往无法发现的新软件问题。在这之中，我们重点关注AI特定的代码异味，这些代码中的重复模式可能预示着更深层次的问题，如不可再现性、静默失败或模型泛化不良。我们介绍了SpecDetect4AI工具，这是一种用于大规模指定和检测这些代码异味的方法。该方法结合了高级声明性的领域特定语言（DSL）用于规则指定，以及一个可扩展的静态分析工具来解释和检测这些规则，适用于AI系统。我们指定了22种AI特定的代码异味，并在包含826个基于AI的系统（2000万行代码）的评估中，实现了88.66%的查准率和88.89%的查全率，优于其他现有检测工具。我们的研究结果表明，SpecDetect4AI能够通过专门的规则支持AI特定的代码异味的指定和检测，并有效地分析大型的基于AI的系统，展示了其高效性和可扩展性（SUS 81.7/100）。 

---
# CoSupFormer : A Contrastive Supervised learning approach for EEG signal Classification 

**Title (ZH)**: CoSupFormer: 一种对比监督学习方法用于 EEG 信号分类 

**Authors**: D. Darankoum, C. Habermacher, J. Volle, S. Grudinin  

**Link**: [PDF](https://arxiv.org/pdf/2509.20489)  

**Abstract**: Electroencephalography signals (EEGs) contain rich multi-scale information crucial for understanding brain states, with potential applications in diagnosing and advancing the drug development landscape. However, extracting meaningful features from raw EEG signals while handling noise and channel variability remains a major challenge. This work proposes a novel end-to-end deep-learning framework that addresses these issues through several key innovations. First, we designed an encoder capable of explicitly capturing multi-scale frequency oscillations covering a wide range of features for different EEG-related tasks. Secondly, to model complex dependencies and handle the high temporal resolution of EEGs, we introduced an attention-based encoder that simultaneously learns interactions across EEG channels and within localized {\em patches} of individual channels. We integrated a dedicated gating network on top of the attention encoder to dynamically filter out noisy and non-informative channels, enhancing the reliability of EEG data. The entire encoding process is guided by a novel loss function, which leverages supervised and contrastive learning, significantly improving model generalization. We validated our approach in multiple applications, ranging from the classification of effects across multiple Central Nervous System (CNS) disorders treatments to the diagnosis of Parkinson's and Alzheimer's disease. Our results demonstrate that the proposed learning paradigm can extract biologically meaningful patterns from raw EEG signals across different species, autonomously select high-quality channels, and achieve robust generalization through innovative architectural and loss design. 

**Abstract (ZH)**: 脑电图信号（EEGs）包含丰富的多尺度信息，对于理解脑状态、疾病诊断及药物开发具有潜在应用价值。然而，从原始EEGs中提取有意义的特征并处理噪声和通道变异仍是一个重大挑战。本研究提出了一种新颖的端到端深度学习框架，通过多项创新解决这些问题。首先，我们设计了一个编码器，能够明确捕捉涵盖不同EEG相关任务广泛范围的多尺度频率振荡。其次，为建模复杂依赖关系并处理EEGs的高时间分辨率，我们引入了一种基于注意力机制的编码器，该编码器同时学习EEG通道间的相互作用以及个体通道局部区域内的交互。我们在注意力编码器上集成了一个专用的门控网络，动态过滤出无用和噪声通道，增强EEG数据的可靠性。整个编码过程由一个新颖的损失函数引导，该损失函数结合监督学习和对比学习，显著提高了模型的泛化能力。我们在此类比症治疗方法效果分类、帕金森病和阿尔茨海默病诊断等多种应用中验证了该方法。研究结果表明，所提出的学习范式可以从不同物种的原始EEG信号中提取出生物学意义的模式，自主选择高质量通道，并通过创新的架构和损失设计实现稳健泛化。 

---
# Shared Neural Space: Unified Precomputed Feature Encoding for Multi-Task and Cross Domain Vision 

**Title (ZH)**: 共享神经空间：统一先计算特征编码用于多任务和跨域视觉任务 

**Authors**: Jing Li, Oskar Bartosz, Chengyu Wang, Michal Wnuczynski, Dilshan Godaliyadda, Michael Polley  

**Link**: [PDF](https://arxiv.org/pdf/2509.20481)  

**Abstract**: The majority of AI models in imaging and vision are customized to perform on specific high-precision task. However, this strategy is inefficient for applications with a series of modular tasks, since each requires a mapping into a disparate latent domain. To address this inefficiency, we proposed a universal Neural Space (NS), where an encoder-decoder framework pre-computes features across vision and imaging tasks. Our encoder learns transformation aware, generalizable representations, which enable multiple downstream AI modules to share the same feature space. This architecture reduces redundancy, improves generalization across domain shift, and establishes a foundation for effecient multi-task vision pipelines. Furthermore, as opposed to larger transformer backbones, our backbone is lightweight and CNN-based, allowing for wider across hardware. We furthur demonstrate that imaging and vision modules, such as demosaicing, denoising, depth estimation and semantic segmentation can be performed efficiently in the NS. 

**Abstract (ZH)**: 大多数成像和视觉中的AI模型都是为特定高精度任务量身定制的。然而，对于一系列模块化任务的应用而言，这种策略是低效率的，因为每个任务都需要映射到不同的潜在域。为了解决这一低效率问题，我们提出了一种通用神经空间（NS），其中编码器-解码器框架预计算视觉和成像任务的特征。我们的编码器学习变换感知的一般化表示，使多个下游AI模块能够共享同一特征空间。这种架构减少了冗余，提高了跨域泛化能力，并为高效的多任务视觉管道奠定了基础。此外，与较大的变压器骨干网络相比，我们的骨干网络更轻量级且基于CNN，可以在更广泛的硬件上运行。我们进一步证明，成像和视觉模块，如去马赛克、去噪、深度估计和语义分割，可以在NS中高效运行。 

---
# Wartime Media Dynamics in Emerging Democracies: Case Study of Pakistani Media in May 2025 Indo-Pak Conflict 

**Title (ZH)**: 新兴民主国家战时媒体动态：2025年印巴冲突中巴基斯坦媒体案例研究 

**Authors**: Taaha Saleem Bajwa  

**Link**: [PDF](https://arxiv.org/pdf/2509.20419)  

**Abstract**: Democracies rely on opposition and dissent to function, but in emerging democracies, freedom of speech is often restricted. This effect intensifies during regional conflicts. This study examines how the India-Pakistan conflict of May 2025 influenced Pakistani media coverage. Analyzing approximately 2,600 news articles from three major newspapers using a large language model (LLM), the study found that war-related reporting significantly overshadowed coverage of political opposition and dissent. These findings highlight how conflict can marginalize democratic discourse, reinforcing the need to safeguard press freedom in volatile regions. 

**Abstract (ZH)**: 新兴民主国家中，言论自由often restricted，在区域性冲突期间尤为受限。印巴2025年5月冲突对巴基斯坦媒体 coverage的影响研究：冲突如何边缘化民主 discourse，强化在不稳定地区保护 press freedom的 necessity。 

---
# A Taxonomy of Data Risks in AI and Quantum Computing (QAI) - A Systematic Review 

**Title (ZH)**: AI和量子计算中数据风险的分类：一项系统性回顾 

**Authors**: Grace Billiris, Asif Gill, Madhushi Bandara  

**Link**: [PDF](https://arxiv.org/pdf/2509.20418)  

**Abstract**: Quantum Artificial Intelligence (QAI), the integration of Artificial Intelligence (AI) and Quantum Computing (QC), promises transformative advances, including AI-enabled quantum cryptography and quantum-resistant encryption protocols. However, QAI inherits data risks from both AI and QC, creating complex privacy and security vulnerabilities that are not systematically studied. These risks affect the trustworthiness and reliability of AI and QAI systems, making their understanding critical. This study systematically reviews 67 privacy- and security-related studies to expand understanding of QAI data risks. We propose a taxonomy of 22 key data risks, organised into five categories: governance, risk assessment, control implementation, user considerations, and continuous monitoring. Our findings reveal vulnerabilities unique to QAI and identify gaps in holistic risk assessment. This work contributes to trustworthy AI and QAI research and provides a foundation for developing future risk assessment tools. 

**Abstract (ZH)**: 量子人工智能（QAI）：数据风险的系统研究与分类 

---
# Adversarial Defense in Cybersecurity: A Systematic Review of GANs for Threat Detection and Mitigation 

**Title (ZH)**: 网络安全中的对抗性防御：基于GANs的威胁检测与缓解综述 

**Authors**: Tharcisse Ndayipfukamiye, Jianguo Ding, Doreen Sebastian Sarwatt, Adamu Gaston Philipo, Huansheng Ning  

**Link**: [PDF](https://arxiv.org/pdf/2509.20411)  

**Abstract**: Machine learning-based cybersecurity systems are highly vulnerable to adversarial attacks, while Generative Adversarial Networks (GANs) act as both powerful attack enablers and promising defenses. This survey systematically reviews GAN-based adversarial defenses in cybersecurity (2021--August 31, 2025), consolidating recent progress, identifying gaps, and outlining future directions. Using a PRISMA-compliant systematic literature review protocol, we searched five major digital libraries. From 829 initial records, 185 peer-reviewed studies were retained and synthesized through quantitative trend analysis and thematic taxonomy development. We introduce a four-dimensional taxonomy spanning defensive function, GAN architecture, cybersecurity domain, and adversarial threat model. GANs improve detection accuracy, robustness, and data utility across network intrusion detection, malware analysis, and IoT security. Notable advances include WGAN-GP for stable training, CGANs for targeted synthesis, and hybrid GAN models for improved resilience. Yet, persistent challenges remain such as instability in training, lack of standardized benchmarks, high computational cost, and limited explainability. GAN-based defenses demonstrate strong potential but require advances in stable architectures, benchmarking, transparency, and deployment. We propose a roadmap emphasizing hybrid models, unified evaluation, real-world integration, and defenses against emerging threats such as LLM-driven cyberattacks. This survey establishes the foundation for scalable, trustworthy, and adaptive GAN-powered defenses. 

**Abstract (ZH)**: 基于机器学习的网络安全系统极易受到对抗性攻击，而生成式对抗网络（GANs）既是强大的攻击工具也是有前景的防御手段。本文系统回顾了从2021年到2025年8月31日的GAN基对抗防御在网络安全领域的进展，总结了近期成果、识别了研究缺口并提出了未来方向。通过遵循PRISMA合规的系统文献综述协议，我们搜索了五大主要数字图书馆。从829篇初始记录中，筛选出185篇同行评审研究，并通过定量趋势分析和主题分类发展进行了综合整理。我们引入了一个四维分类法，涵盖防御功能、GAN架构、网络安全领域和对抗威胁模型。GANs在网络入侵检测、恶意软件分析和物联网安全中的检测准确性、鲁棒性和数据实用性均有提升。显著进展包括WGAN-GP的稳定训练、CGANs的目标合成以及混合GAN模型的增强鲁棒性。然而，仍存在持续的挑战，如训练不稳定性、缺乏标准化基准、高计算成本和解释性有限。基于GAN的防御展现出强大的潜力，但需要在稳定架构、基准化、透明性及部署方面取得进展。我们提出了一个路线图，强调混合模型、统一评估、现实世界集成以及针对新兴威胁（如由大语言模型驱动的网络攻击）的防御。本文建立了基于GAN的可扩展、可靠且自适应防御的基础。 

---
# Defending against Stegomalware in Deep Neural Networks with Permutation Symmetry 

**Title (ZH)**: 基于.permutation symmetry.的深度神经网络对抗Stegomalware攻击 

**Authors**: Birk Torpmann-Hagen, Michael A. Riegler, Pål Halvorsen, Dag Johansen  

**Link**: [PDF](https://arxiv.org/pdf/2509.20399)  

**Abstract**: Deep neural networks are being utilized in a growing number of applications, both in production systems and for personal use. Network checkpoints are as a consequence often shared and distributed on various platforms to ease the development process. This work considers the threat of neural network stegomalware, where malware is embedded in neural network checkpoints at a negligible cost to network accuracy. This constitutes a significant security concern, but is nevertheless largely neglected by the deep learning practitioners and security specialists alike. We propose the first effective countermeasure to these attacks. In particular, we show that state-of-the-art neural network stegomalware can be efficiently and effectively neutralized through shuffling the column order of the weight- and bias-matrices, or equivalently the channel-order of convolutional layers. We show that this effectively corrupts payloads that have been embedded by state-of-the-art methods in neural network steganography at no cost to network accuracy, outperforming competing methods by a significant margin. We then discuss possible means by which to bypass this defense, additional defense methods, and advocate for continued research into the security of machine learning systems. 

**Abstract (ZH)**: 深度神经网络在生产系统和个人使用中应用日益广泛，由此产生的网络检查点常常被共享和分发在各种平台上以简化开发过程。本文考虑了神经网络隐马恶意软件的威胁，其中恶意软件以微不足道的代价嵌入在神经网络检查点中。这构成了一项重要的安全问题，但这一问题却未得到深度学习实践者和安全专家的广泛关注。我们提出了有效的预防措施来应对这种攻击。具体而言，我们证明可以通过重新排列权重和偏置矩阵的列顺序，或等效地重新排列卷积层的通道顺序，来高效且有效地消除最先进的神经网络隐马恶意软件。我们展示了这种方法能够有效地破坏最先进的神经网络隐写术所嵌入的有效载荷，同时对网络准确性的成本为零，并显著优于其他竞争方法。随后，我们讨论了绕过这种防御的可能方法、额外的防御措施，并倡导继续致力于机器学习系统的安全性研究。 

---
# Variational Low-Rank Adaptation for Personalized Impaired Speech Recognition 

**Title (ZH)**: 变分低秩适应技术用于个性化受损语音识别 

**Authors**: Niclas Pokel, Pehuén Moure, Roman Boehringer, Shih-Chii Liu, Yingqiang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.20397)  

**Abstract**: Speech impairments resulting from congenital disorders, such as cerebral palsy, down syndrome, or apert syndrome, as well as acquired brain injuries due to stroke, traumatic accidents, or tumors, present major challenges to automatic speech recognition (ASR) systems. Despite recent advancements, state-of-the-art ASR models like Whisper still struggle with non-normative speech due to limited training data availability and high acoustic variability. Moreover, collecting and annotating non-normative speech is burdensome: speaking is effortful for many affected individuals, while laborious annotation often requires caregivers familiar with the speaker. This work introduces a novel ASR personalization method based on Bayesian Low-rank Adaptation for data-efficient fine-tuning. We validate our method on the English UA-Speech dataset and a newly collected German speech dataset, BF-Sprache, from a child with structural speech impairment. The dataset and approach are designed to reflect the challenges of low-resource settings that include individuals with speech impairments. Our method significantly improves ASR accuracy for impaired speech while maintaining data and annotation efficiency, offering a practical path toward inclusive ASR. 

**Abstract (ZH)**: 先天性障碍（如脑瘫、唐氏综合征或阿珀特综合症）导致的言语障碍以及由于中风、创伤性事故或肿瘤引起的获得性脑损伤导致的言语障碍，给自动语音识别（ASR）系统带来了重大挑战。尽管 recent 进展，最先进的 ASR 模型如 Whisper 仍难以处理非规范性言语，原因在于训练数据有限和高 acoustic 变异性。此外，收集和标注非规范性言语非常耗时：许多受影响个体在说话时感到吃力，而劳动密集型标注通常需要熟悉讲话者的护理人员。本文介绍了一种基于贝叶斯低秩适应的新 ASR 个性化方法，以实现高效的数据微调。我们在英语 UA-Speech 数据集和一个新的来自结构言语障碍儿童的德语言语数据集 BF-Sprache 上验证了该方法。该数据集和方法旨在反映包括言语障碍个体在内的资源有限环境中的挑战。本方法显著提高了对受损害言语的 ASR 准确性，同时保持了数据和标注的高效性，为包容性 ASR 提供了一条实用路径。 

---
# Data-Efficient ASR Personalization for Non-Normative Speech Using an Uncertainty-Based Phoneme Difficulty Score for Guided Sampling 

**Title (ZH)**: 基于不确定性音素难度分数的高效ASR个人化建模用于非规范语音识别 

**Authors**: Niclas Pokel, Pehuén Moure, Roman Boehringer, Yingqiang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.20396)  

**Abstract**: Automatic speech recognition (ASR) systems struggle with non-normative speech from individuals with impairments caused by conditions like cerebral palsy or structural anomalies. The high acoustic variability and scarcity of training data severely degrade model performance. This work introduces a data-efficient personalization method that quantifies phoneme-level uncertainty to guide fine-tuning. We leverage Monte Carlo Dropout to estimate which phonemes a model finds most difficult and use these estimates for a targeted oversampling strategy. We validate our method on English and German datasets. Crucially, we demonstrate that our model-derived uncertainty strongly correlates with phonemes identified as challenging in an expert clinical logopedic report, marking, to our knowledge, the first work to successfully align model uncertainty with expert assessment of speech difficulty. Our results show that this clinically-validated, uncertainty-guided sampling significantly improves ASR accuracy, delivering a practical framework for personalized and inclusive ASR. 

**Abstract (ZH)**: 自动语音识别（ASR）系统在处理由脑瘫或结构异常等条件引起障碍个体的非规范语音时存在困难。高声学变异性及训练数据的稀缺性严重降低模型性能。本研究提出一种数据高效个性化方法，量化音素级不确定性以指导微调。我们利用蒙特卡洛丢弃估计模型认为最困难的音素，并利用这些估计值进行针对性过采样策略。我们在英语和德语数据集上验证了该方法。关键的是，我们证明了我们模型得出的不确定性与专家临床语音治疗报告中识别的具有挑战性的音素高度相关，这是我们所知的首次成功将模型不确定性与专家评估的语音难度对齐的工作。我们的结果表明，这种临床验证、基于不确定性采样的方法显著提高了ASR精度，提供了一个实用的个性化和包容性ASR框架。 

---
# Centralized vs. Decentralized Security for Space AI Systems? A New Look 

**Title (ZH)**: 集中式与分布式安全在空间AI系统中的权衡：一种新的视角 

**Authors**: Noam Schmitt, Marc Antoine Lacoste  

**Link**: [PDF](https://arxiv.org/pdf/2509.20395)  

**Abstract**: This paper investigates the trade-off between centralized and decentralized security management in constellations of satellites to balance security and performance. We highlight three key AI architectures for automated security management: (a) centralized, (b) distributed and (c) federated. The centralized architecture is the best option short term, providing fast training, despite the hard challenge of the communication latency overhead across space. Decentralized architectures are better alternatives in the longer term, providing enhanced scalability and security. 

**Abstract (ZH)**: 本文研究卫星星座中集中式与分布式安全管理之间的权衡，以平衡安全性和性能。我们强调三种关键的AI架构以实现自动安全管理：(a) 集中式，(b) 分布式，(c) 联邦式。集中式架构在短期内是最优选择，尽管存在跨太空通信延迟的挑战，仍能提供快速训练。分布式架构在长期内是更好的替代方案，能够提供增强的可扩展性和安全性。 

---
# Blueprints of Trust: AI System Cards for End to End Transparency and Governance 

**Title (ZH)**: 信任蓝本：端到端透明度与治理的AI系统卡片 

**Authors**: Huzaifa Sidhpurwala, Emily Fox, Garth Mollett, Florencio Cano Gabarda, Roman Zhukov  

**Link**: [PDF](https://arxiv.org/pdf/2509.20394)  

**Abstract**: This paper introduces the Hazard-Aware System Card (HASC), a novel framework designed to enhance transparency and accountability in the development and deployment of AI systems. The HASC builds upon existing model card and system card concepts by integrating a comprehensive, dynamic record of an AI system's security and safety posture. The framework proposes a standardized system of identifiers, including a novel AI Safety Hazard (ASH) ID, to complement existing security identifiers like CVEs, allowing for clear and consistent communication of fixed flaws. By providing a single, accessible source of truth, the HASC empowers developers and stakeholders to make more informed decisions about AI system safety throughout its lifecycle. Ultimately, we also compare our proposed AI system cards with the ISO/IEC 42001:2023 standard and discuss how they can be used to complement each other, providing greater transparency and accountability for AI systems. 

**Abstract (ZH)**: 基于风险的系统卡片（HASC）：一种增强人工智能系统开发和部署透明度与问责制的新框架 

---
# The Secret Agenda: LLMs Strategically Lie and Our Current Safety Tools Are Blind 

**Title (ZH)**: 隐秘议程：大规模语言模型策略性说谎，当前的安全工具视而不见 

**Authors**: Caleb DeLeeuw, Gaurav Chawla, Aniket Sharma, Vanessa Dietze  

**Link**: [PDF](https://arxiv.org/pdf/2509.20393)  

**Abstract**: We investigate strategic deception in large language models using two complementary testbeds: Secret Agenda (across 38 models) and Insider Trading compliance (via SAE architectures). Secret Agenda reliably induced lying when deception advantaged goal achievement across all model families. Analysis revealed that autolabeled SAE features for "deception" rarely activated during strategic dishonesty, and feature steering experiments across 100+ deception-related features failed to prevent lying. Conversely, insider trading analysis using unlabeled SAE activations separated deceptive versus compliant responses through discriminative patterns in heatmaps and t-SNE visualizations. These findings suggest autolabel-driven interpretability approaches fail to detect or control behavioral deception, while aggregate unlabeled activations provide population-level structure for risk assessment. Results span Llama 8B/70B SAE implementations and GemmaScope under resource constraints, representing preliminary findings that motivate larger studies on feature discovery, labeling methodology, and causal interventions in realistic deception contexts. 

**Abstract (ZH)**: 我们使用两种互补的测试平台（Secret Agenda和通过SAE架构的 Insider Trading合规性）来探究大型语言模型中的策略性欺骗。研究发现，在所有模型家族中，当欺骗有利于目标实现时，Secret Agenda可靠地诱导了说谎行为。分析显示，自标注的SAE特征“欺骗”极少在策略性不诚实过程中激活，并且在100多个相关特征的特征导向实验中未能阻止说谎行为。相反，未标注的SAE激活在内幕交易分析中通过热图和t-SNE可视化中的区分模式分离出了欺骗性回应和合规性回应。这些发现表明，自标注驱动的可解释性方法无法检测或控制行为欺骗，而聚合的未标注激活则提供了在群体层面进行风险评估的结构。研究结果涵盖了在资源约束下实现的Llama 8B/70B SAE以及GemmaScope，代表了初步发现，这些发现激发了对特征发现、标注方法和因果干预在实际欺骗情境中的更大规模研究。 

---
# Can You Trust Your Copilot? A Privacy Scorecard for AI Coding Assistants 

**Title (ZH)**: 你能信任你的副驾吗？AI编码助手的隐私评分卡 

**Authors**: Amir AL-Maamari  

**Link**: [PDF](https://arxiv.org/pdf/2509.20388)  

**Abstract**: The rapid integration of AI-powered coding assistants into developer workflows has raised significant privacy and trust concerns. As developers entrust proprietary code to services like OpenAI's GPT, Google's Gemini, and GitHub Copilot, the unclear data handling practices of these tools create security and compliance risks. This paper addresses this challenge by introducing and applying a novel, expert-validated privacy scorecard. The methodology involves a detailed analysis of four document types; from legal policies to external audits; to score five leading assistants against 14 weighted criteria. A legal expert and a data protection officer refined these criteria and their weighting. The results reveal a distinct hierarchy of privacy protections, with a 20-point gap between the highest- and lowest-ranked tools. The analysis uncovers common industry weaknesses, including the pervasive use of opt-out consent for model training and a near-universal failure to filter secrets from user prompts proactively. The resulting scorecard provides actionable guidance for developers and organizations, enabling evidence-based tool selection. This work establishes a new benchmark for transparency and advocates for a shift towards more user-centric privacy standards in the AI industry. 

**Abstract (ZH)**: AI驱动编码助手快速集成到开发者工作流中引发了显著的隐私和信任 concerns。本论文通过引入并应用一种新型、专家验证的隐私评分卡来应对这一挑战。该方法包括对四种文档类型进行详细分析；从法律政策到外部审计；以根据14项加权标准对五种领先助手进行评分。一位法律专家和一位数据保护官员对这些标准及其权重进行了细化。结果显示，这些工具在隐私保护方面的层级分明，最高-ranked工具与最低-ranked工具之间有20分的差距。分析揭示了行业中的共同薄弱环节，包括使用退出同意普遍作为模型训练的机制以及几乎全部未能主动过滤用户提示中的机密信息。生成的评分卡为开发者和组织提供了可操作的指导，使其能够基于证据选择工具。这项工作确立了透明度的新基准，并倡导AI行业向更加用户为中心的隐私标准转变。 

---
# Dynamic ReAct: Scalable Tool Selection for Large-Scale MCP Environments 

**Title (ZH)**: 动态ReAct：大规模MCP环境中的可扩展工具选择 

**Authors**: Nishant Gaurav, Adit Akarsh, Ankit Ranjan, Manoj Bajaj  

**Link**: [PDF](https://arxiv.org/pdf/2509.20386)  

**Abstract**: We present Dynamic ReAct, a novel approach for enabling ReAct agents to ef- ficiently operate with extensive Model Control Protocol (MCP) tool sets that exceed the contextual memory limitations of large language models. Our approach addresses the fundamental challenge of tool selection in environments containing hundreds or thousands of available tools, where loading all tools simultaneously is computationally infeasible. We propose and evaluate five distinct architectures that progressively refine the tool selection process, culminating in a search-and-load mechanism that achieves intelligent tool selection with minimal computational overhead. Our experimental results demonstrate that the proposed approach reduces tool loading by up to 50% while maintaining task completion accuracy, advancing the path towards truly general-purpose AI agents capable of dynamically adapting to diverse task environments. 

**Abstract (ZH)**: 我们提出Dynamic ReAct，这是一种新颖的方法，用于使ReAct智能体能够高效地操作包含数以百计甚至数千种工具的广泛模型控制协议（MCP）工具集，同时克服大型语言模型的上下文记忆限制。该方法解决了在包含数百甚至数千种可用工具的环境中，同时加载所有工具在计算上不可行的基本挑战。我们提出并评估了五个不同的架构，逐步优化工具选择过程，最终实现了一种搜索和加载机制，该机制在最小计算开销的情况下实现了智能工具选择。我们的实验结果表明，所提出的方法在保持任务完成准确性的同时将工具加载减少高达50%，并为真正具备广泛用途的智能代理动态适应各种任务环境铺平了道路。 

---
# R1-Fuzz: Specializing Language Models for Textual Fuzzing via Reinforcement Learning 

**Title (ZH)**: R1-Fuzz: 通过强化学习专门化语言模型进行文本模糊测试 

**Authors**: Jiayi Lin, Liangcai Su, Junzhe Li, Chenxiong Qian  

**Link**: [PDF](https://arxiv.org/pdf/2509.20384)  

**Abstract**: Fuzzing is effective for vulnerability discovery but struggles with complex targets such as compilers, interpreters, and database engines, which accept textual input that must satisfy intricate syntactic and semantic constraints. Although language models (LMs) have attracted interest for this task due to their vast latent knowledge and reasoning potential, their practical adoption has been limited. The major challenges stem from insufficient exploration of deep program logic among real-world codebases, and the high cost of leveraging larger models. To overcome these challenges, we propose R1-Fuzz, the first framework that leverages reinforcement learning (RL) to specialize cost-efficient LMs and integrate them for complex textual fuzzing input generation. R1-Fuzz introduces two key designs: coverage-slicing-based question construction and a distance-based reward calculation. Through RL-based post-training of a model with our constructed dataset, R1-Fuzz designs a fuzzing workflow that tightly integrates LMs to reason deep program semantics during fuzzing. Evaluations on diverse real-world targets show that our design enables a small model, named R1-Fuzz-7B, to rival or even outperform much larger models in real-world fuzzing. Notably, R1-Fuzz achieves up to 75\% higher coverage than state-of-the-art fuzzers and discovers 29 previously unknown vulnerabilities, demonstrating its practicality. 

**Abstract (ZH)**: R1-Fuzz：基于强化学习的高效复杂文本 fuzzing 框架 

---
# MARS: A Malignity-Aware Backdoor Defense in Federated Learning 

**Title (ZH)**: MARS：联邦学习中 consideration 的恶性后门防御 

**Authors**: Wei Wan, Yuxuan Ning, Zhicong Huang, Cheng Hong, Shengshan Hu, Ziqi Zhou, Yechao Zhang, Tianqing Zhu, Wanlei Zhou, Leo Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20383)  

**Abstract**: Federated Learning (FL) is a distributed paradigm aimed at protecting participant data privacy by exchanging model parameters to achieve high-quality model training. However, this distributed nature also makes FL highly vulnerable to backdoor attacks. Notably, the recently proposed state-of-the-art (SOTA) attack, 3DFed (SP2023), uses an indicator mechanism to determine whether the backdoor models have been accepted by the defender and adaptively optimizes backdoor models, rendering existing defenses ineffective. In this paper, we first reveal that the failure of existing defenses lies in the employment of empirical statistical measures that are loosely coupled with backdoor attacks. Motivated by this, we propose a Malignity-Aware backdooR defenSe (MARS) that leverages backdoor energy (BE) to indicate the malicious extent of each neuron. To amplify malignity, we further extract the most prominent BE values from each model to form a concentrated backdoor energy (CBE). Finally, a novel Wasserstein distance-based clustering method is introduced to effectively identify backdoor models. Extensive experiments demonstrate that MARS can defend against SOTA backdoor attacks and significantly outperforms existing defenses. 

**Abstract (ZH)**: 联邦学习中的恶意程度感知反门限防御（Mars：Malignity-Aware Backdoor Defense in Federated Learning） 

---
# Lightweight MobileNetV1+GRU for ECG Biometric Authentication: Federated and Adversarial Evaluation 

**Title (ZH)**: 基于MobileNetV1+GRU的轻量级ECG生物认证：联邦学习与对抗性评估 

**Authors**: Dilli Hang Rai, Sabin Kafley  

**Link**: [PDF](https://arxiv.org/pdf/2509.20382)  

**Abstract**: ECG biometrics offer a unique, secure authentication method, yet their deployment on wearable devices faces real-time processing, privacy, and spoofing vulnerability challenges. This paper proposes a lightweight deep learning model (MobileNetV1+GRU) for ECG-based authentication, injection of 20dB Gaussian noise & custom preprocessing. We simulate wearable conditions and edge deployment using the ECGID, MIT-BIH, CYBHi, and PTB datasets, achieving accuracies of 99.34%, 99.31%, 91.74%, and 98.49%, F1-scores of 0.9869, 0.9923, 0.9125, and 0.9771, Precision of 0.9866, 0.9924, 0.9180 and 0.9845, Recall of 0.9878, 0.9923, 0.9129, and 0.9756, equal error rates (EER) of 0.0009, 0.00013, 0.0091, and 0.0009, and ROC-AUC values of 0.9999, 0.9999, 0.9985, and 0.9998, while under FGSM adversarial attacks, accuracy drops from 96.82% to as low as 0.80%. This paper highlights federated learning, adversarial testing, and the need for diverse wearable physiological datasets to ensure secure and scalable biometrics. 

**Abstract (ZH)**: 基于ECG的身份认证：一种轻量级深度学习模型（MobileNetV1+GRU）及其在适用于可穿戴设备中的应用研究 

---
# USB-Rec: An Effective Framework for Improving Conversational Recommendation Capability of Large Language Model 

**Title (ZH)**: USB-Rec: 一种提高大型语言模型对话推荐能力的有效框架 

**Authors**: Jianyu Wen, Jingyun Wang, Cilin Yan, Jiayin Cai, Xiaolong Jiang, Ying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20381)  

**Abstract**: Recently, Large Language Models (LLMs) have been widely employed in Conversational Recommender Systems (CRSs). Unlike traditional language model approaches that focus on training, all existing LLMs-based approaches are mainly centered around how to leverage the summarization and analysis capabilities of LLMs while ignoring the issue of training. Therefore, in this work, we propose an integrated training-inference framework, User-Simulator-Based framework (USB-Rec), for improving the performance of LLMs in conversational recommendation at the model level. Firstly, we design a LLM-based Preference Optimization (PO) dataset construction strategy for RL training, which helps the LLMs understand the strategies and methods in conversational recommendation. Secondly, we propose a Self-Enhancement Strategy (SES) at the inference stage to further exploit the conversational recommendation potential obtained from RL training. Extensive experiments on various datasets demonstrate that our method consistently outperforms previous state-of-the-art methods. 

**Abstract (ZH)**: 最近，大规模语言模型（LLMs）已在会话推荐系统（CRSs）中广泛应用于推荐。在此项工作中，我们提出了一种综合训练-推理框架——用户模拟器基于框架（USB-Rec），以在模型层面提高LLMs在会话推荐中的性能。首先，我们设计了一种基于LLMs的偏好优化（PO）数据集构建策略，以辅助强化学习（RL）训练，帮助LLMs理解会话推荐中的策略和方法。其次，我们在推理阶段提出了一种自我增强策略（SES），进一步挖掘从RL训练中获得的会话推荐潜力。在多种数据集上的广泛实验表明，我们的方法在性能上始终优于之前的最先进方法。 

---
# ACCeLLiuM: Supervised Fine-Tuning for Automated OpenACC Pragma Generation 

**Title (ZH)**: ACCeLLiuM: 监督微调以实现自动化OpenACCpragma生成 

**Authors**: Samyak Jhaveri, Vanessa Klotzmann, Crista Lopes  

**Link**: [PDF](https://arxiv.org/pdf/2509.20380)  

**Abstract**: The increasing ubiquity of GPUs is accompanied by the increasing complexity of their hardware and parallel programming frameworks. Directive-based parallel programming standards like OpenACC simplify GPU programming to some extent by abstracting away low-level complexities, but a fair amount of expertise is still required in order to use those directives effectively.
We introduce ACCeLLiuM, two open weights Large Language Models specifically fine-tuned for generating expert OpenACC directives for data-parallel loops, along with the supervised fine-tuning dataset that was used to train them. The ACCeLLiuM SFT dataset contains 4,033 OpenACC pragma-loop pairs mined from public GitHub C/C++ repositories, with 3,223 pairs for training and 810 for testing. Experimental evaluations show a pronounced performance gap in generating correct OpenACC pragmas between base LLMs and our fine-tuned versions. On the held-out test set, base LLMs fail to consistently generate valid pragmas, whereas LLMs fine-tuned on the ACCeLLiuM dataset generate valid pragmas with the correct directive type for $87\%$ of the data-parallel loops, and exact pragmas--including directives, clauses, clause order, and clause variables--for $50\%$ of the cases. Even when not exact, generated pragmas frequently incorporate the correct clauses in a different order than the ground-truth label, or include additional clauses that enable finer control over parallel execution, data movement, and concurrency, offering practical value beyond strict string-matching. By publicly releasing the code, models, and dataset as ACCeLLiuM we hope to establish a reproducible benchmark for LLM-powered OpenACC pragma generation, and lower the barrier to automated GPU offloading of serially written programs. 

**Abstract (ZH)**: GPU使用日益普及的同时，其硬件复杂性和并行编程框架也变得愈发复杂。基于指令的并行编程标准如OpenACC在一定程度上简化了GPU编程，通过抽象低级复杂性降低了编程难度，但仍需要相当的专业知识才能有效使用这些指令。
我们介绍了ACCeLLiuM，这是一个专为生成数据并行循环的专家级OpenACC指令而设计的两个开源大型语言模型，以及用于训练它们的监督细调数据集。ACCeLLiuM SFT数据集包含从公共GitHub C/C++仓库中挖掘出的4033个OpenACC 元令-循环对，其中3223个用于训练，810个用于测试。实验评估结果显示，基模型在生成正确的OpenACC 元令方面与我们的细调版本之间存在明显的性能差距。在保留的测试集上，基模型不能一致地生成有效的元令，而使用ACCeLLiuM数据集进行细调的模型能够为87%的数据并行循环生成具有正确指令类型的有效元令，并且在50%的情况下生成包含正确指令、子句、子句顺序和变量的确切元令。即使不准确，生成的元令也经常以不同的顺序包含正确的子句，或者包含额外的子句以实现更精细的并行执行、数据移动和并发控制，为严格字符串匹配之外的实际应用提供了价值。通过公开发布代码、模型和数据集作为ACCeLLiuM，我们希望建立一个可重复的基准，用于基于大型语言模型的OpenACC元令生成，并降低自动将串行编写的程序卸载到GPU上的障碍。 

---
# Beyond Global Emotion: Fine-Grained Emotional Speech Synthesis with Dynamic Word-Level Modulation 

**Title (ZH)**: 超越全局情感：基于动态词级调制的细粒度情感语音合成 

**Authors**: Sirui Wang, Andong Chen, Tiejun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.20378)  

**Abstract**: Emotional text-to-speech (E-TTS) is central to creating natural and trustworthy human-computer interaction. Existing systems typically rely on sentence-level control through predefined labels, reference audio, or natural language prompts. While effective for global emotion expression, these approaches fail to capture dynamic shifts within a sentence. To address this limitation, we introduce Emo-FiLM, a fine-grained emotion modeling framework for LLM-based TTS. Emo-FiLM aligns frame-level features from emotion2vec to words to obtain word-level emotion annotations, and maps them through a Feature-wise Linear Modulation (FiLM) layer, enabling word-level emotion control by directly modulating text embeddings. To support evaluation, we construct the Fine-grained Emotion Dynamics Dataset (FEDD) with detailed annotations of emotional transitions. Experiments show that Emo-FiLM outperforms existing approaches on both global and fine-grained tasks, demonstrating its effectiveness and generality for expressive speech synthesis. 

**Abstract (ZH)**: 情感文本到语音（E-TTS）是创建自然可靠的人机交互的核心。现有的系统通常依赖于通过预定义标签、参考音频或自然语言提示进行句子级控制。虽然这些方法对于全局情感表达是有效的，但它们无法捕捉句子内的动态变化。为了解决这一局限，我们提出了Emo-FiLM，一种基于LLM的情感细粒度建模框架。Emo-FiLM将情感2vec的帧级特征对齐到单词上，以获取单词级情感注释，并通过Feature-wise Linear Modulation (FiLM)层进行映射，从而通过直接调节文本嵌入实现单词级情感控制。为了支持评估，我们构建了详细的情感过渡注释的数据集Fine-grained Emotion Dynamics Dataset (FEDD)。实验表明，Emo-FiLM在全局和细粒度任务上都优于现有方法，展示了其在表情合成方面的有效性和普适性。 

---
# SKILL-RAG: Self-Knowledge Induced Learning and Filtering for Retrieval-Augmented Generation 

**Title (ZH)**: SKILL-RAG：自我知识引导的检索增强生成学习与过滤 

**Authors**: Tomoaki Isoda  

**Link**: [PDF](https://arxiv.org/pdf/2509.20377)  

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly improved the performance of large language models (LLMs) on knowledge-intensive tasks in recent years. However, since retrieval systems may return irrelevant content, incorporating such information into the model often leads to hallucinations. Thus, identifying and filtering out unhelpful retrieved content is a key challenge for improving RAG this http URL better integrate the internal knowledge of the model with external knowledge from retrieval, it is essential to understand what the model "knows" and "does not know" (which is also called "self-knowledge"). Based on this insight, we propose SKILL-RAG (Self-Knowledge Induced Learning and Filtering for RAG), a novel method that leverages the model's self-knowledge to determine which retrieved documents are beneficial for answering a given query. We design a reinforcement learning-based training framework to explicitly elicit self-knowledge from the model and employs sentence-level granularity to filter out irrelevant content while preserving useful this http URL evaluate SKILL-RAG using Llama2-7B and Qwen3-8B on several question answering benchmarks. Experimental results demonstrate that SKILL-RAG not only improves generation quality but also significantly reduces the number of input documents, validating the importance of self-knowledge in guiding the selection of high-quality retrievals. 

**Abstract (ZH)**: Retrieval-Augmented Generation (基于检索增强生成)在近些年显著提升了大型语言模型（LLMs）在知识密集型任务上的性能。然而，由于检索系统可能会返回无关的内容，将此类信息集成到模型中常会导致幻想。因此，识别并过滤掉无用的检索内容是提高基于检索增强生成的关键挑战之一。为了更好地将模型的内部知识与检索外部知识整合，理解模型知道什么和不知道什么（也称为“自我知识”）是必不可少的。基于这一洞察，我们提出了一种新颖的方法——SKILL-RAG（自我知识引导的学习和过滤），该方法利用模型的自我知识来确定哪些检索到的文档有助于回答给定的问题。我们设计了一种基于强化学习的训练框架，明确地从模型中提取自我知识，并采用句级粒度过滤无关内容以保留有用的内容。我们使用Llama2-7B和Qwen3-8B对多种问答基准进行了SKILL-RAG的评估。实验结果表明，SKILL-RAG不仅提高了生成质量，还显著降低了输入文档的数量，验证了自我知识在指导高质量检索选择中的重要性。 

---
# ConceptViz: A Visual Analytics Approach for Exploring Concepts in Large Language Models 

**Title (ZH)**: ConceptViz：对大型语言模型中概念进行探索的可视化分析方法 

**Authors**: Haoxuan Li, Zhen Wen, Qiqi Jiang, Chenxiao Li, Yuwei Wu, Yuchen Yang, Yiyao Wang, Xiuqi Huang, Minfeng Zhu, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.20376)  

**Abstract**: Large language models (LLMs) have achieved remarkable performance across a wide range of natural language tasks. Understanding how LLMs internally represent knowledge remains a significant challenge. Despite Sparse Autoencoders (SAEs) have emerged as a promising technique for extracting interpretable features from LLMs, SAE features do not inherently align with human-understandable concepts, making their interpretation cumbersome and labor-intensive. To bridge the gap between SAE features and human concepts, we present ConceptViz, a visual analytics system designed for exploring concepts in LLMs. ConceptViz implements a novel dentification => Interpretation => Validation pipeline, enabling users to query SAEs using concepts of interest, interactively explore concept-to-feature alignments, and validate the correspondences through model behavior verification. We demonstrate the effectiveness of ConceptViz through two usage scenarios and a user study. Our results show that ConceptViz enhances interpretability research by streamlining the discovery and validation of meaningful concept representations in LLMs, ultimately aiding researchers in building more accurate mental models of LLM features. Our code and user guide are publicly available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在广泛自然语言任务上取得了显著性能。理解LLMs内部知识表示仍然是一个重大挑战。尽管稀疏自编码器（SAEs）作为从LLMs中提取可解读特征的有前景技术已 emergence，SAE特征并不天然与人类可理解的概念对齐，使得其解读过程繁琐且耗费劳动。为弥合SAE特征与人类概念之间的差距，我们提出ConceptViz，一种用于探索LLMs概念的视觉分析系统。ConceptViz 实现了一种新颖的“识别 => 解释 => 验证”流水线，使用户能够使用感兴趣的 concept 查询 SAE，交互式地探索 concept-to-feature 对齐，并通过模型行为验证来确认这些对应关系。通过两个使用场景和用户研究，我们展示了 ConceptViz 的有效性。我们的结果表明，ConceptViz 通过简化有意义 concept 表示的发现和验证，增强了可解释性研究，最终帮助研究人员构建更准确的LLM特征心理模型。我们的代码和用户指南可在以下网址获取。 

---
# Assessing Classical Machine Learning and Transformer-based Approaches for Detecting AI-Generated Research Text 

**Title (ZH)**: 评估经典机器学习方法和基于变压器的方法在检测AI生成的科研文本中的性能 

**Authors**: Sharanya Parimanoharan, Ruwan D. Nawarathna  

**Link**: [PDF](https://arxiv.org/pdf/2509.20375)  

**Abstract**: The rapid adoption of large language models (LLMs) such as ChatGPT has blurred the line between human and AI-generated texts, raising urgent questions about academic integrity, intellectual property, and the spread of misinformation. Thus, reliable AI-text detection is needed for fair assessment to safeguard human authenticity and cultivate trust in digital communication. In this study, we investigate how well current machine learning (ML) approaches can distinguish ChatGPT-3.5-generated texts from human-written texts employing a labeled data set of 250 pairs of abstracts from a wide range of research topics. We test and compare both classical (Logistic Regression armed with classical Bag-of-Words, POS, and TF-IDF features) and transformer-based (BERT augmented with N-grams, DistilBERT, BERT with a lightweight custom classifier, and LSTM-based N-gram models) ML detection techniques. As we aim to assess each model's performance in detecting AI-generated research texts, we also aim to test whether an ensemble of these models can outperform any single detector. Results show DistilBERT achieves the overall best performance, while Logistic Regression and BERT-Custom offer solid, balanced alternatives; LSTM- and BERT-N-gram approaches lag. The max voting ensemble of the three best models fails to surpass DistilBERT itself, highlighting the primacy of a single transformer-based representation over mere model diversity. By comprehensively assessing the strengths and weaknesses of these AI-text detection approaches, this work lays a foundation for more robust transformer frameworks with larger, richer datasets to keep pace with ever-improving generative AI models. 

**Abstract (ZH)**: 大型语言模型（LLMs）如ChatGPT的快速应用模糊了人类和AI生成文本的界限，引发了关于学术诚信、知识产权和虚假信息传播的紧迫问题。因此，需要可靠的AI文本检测以实现公平评估，保护人类的 authenticity 并促进对数字通信的信任。在本研究中，我们调查了当前机器学习（ML）方法如何区分ChatGPT-3.5生成的文本与人类撰写的文本，使用了涵盖广泛研究主题的250对摘要作为标注数据集。我们测试并比较了古典（配备经典词袋、词性标注和TF-IDF特征的逻辑回归）和基于变换器的方法（BERT结合n-gram、DistilBERT、轻量级自定义分类器的BERT和基于LSTM的n-gram模型）。旨在评估每种模型检测AI生成的研究文本的能力，同时也测试模型集合是否能优于单个检测器。结果显示，DistilBERT在综合表现上最佳，逻辑回归和BERT-自定义分类器提供了稳健且平衡的选择；LSTM和BERT-n-gram方法滞后。三最佳模型的最大投票集合未能超越DistilBERT本身，突显单个基于变换器的表示优于单纯模型多样性的重要性。通过全面评估这些AI文本检测方法的优势和弱点，本研究为基础研究奠定了更大的、更丰富的数据集的坚实基础，以跟上不断改进的生成式AI模型。 

---
# CFD-LLMBench: A Benchmark Suite for Evaluating Large Language Models in Computational Fluid Dynamics 

**Title (ZH)**: CFD-LLMBench：计算流体力学中大型语言模型评估套件 

**Authors**: Nithin Somasekharan, Ling Yue, Yadi Cao, Weichao Li, Patrick Emami, Pochinapeddi Sai Bhargav, Anurag Acharya, Xingyu Xie, Shaowu Pan  

**Link**: [PDF](https://arxiv.org/pdf/2509.20374)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong performance across general NLP tasks, but their utility in automating numerical experiments of complex physical system -- a critical and labor-intensive component -- remains underexplored. As the major workhorse of computational science over the past decades, Computational Fluid Dynamics (CFD) offers a uniquely challenging testbed for evaluating the scientific capabilities of LLMs. We introduce CFDLLMBench, a benchmark suite comprising three complementary components -- CFDQuery, CFDCodeBench, and FoamBench -- designed to holistically evaluate LLM performance across three key competencies: graduate-level CFD knowledge, numerical and physical reasoning of CFD, and context-dependent implementation of CFD workflows. Grounded in real-world CFD practices, our benchmark combines a detailed task taxonomy with a rigorous evaluation framework to deliver reproducible results and quantify LLM performance across code executability, solution accuracy, and numerical convergence behavior. CFDLLMBench establishes a solid foundation for the development and evaluation of LLM-driven automation of numerical experiments for complex physical systems. Code and data are available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在通用自然语言处理任务中表现出强大的性能，但在自动化复杂物理系统数值实验中的应用——一个至关重要的劳动密集型环节——仍待深入探索。作为过去几十年中计算科学的主要工具，计算流体力学（CFD）为评估LLMs的科学能力提供了独特的挑战性测试平台。我们提出了CFDLLMBench，一个包含三个互补组件的基准套件——CFDQuery、CFDCodeBench和FoamBench——旨在全方位评估LLMs在这三大关键能力上的表现：graduate-level CFD知识、CFD的数值与物理推理以及CFD工作流的上下文相关实现。基于实际的CFD实践，我们的基准结合了详细的任务分类体系和严格的评估框架，以实现可重现的结果并量化LLMs在代码可执行性、解的准确性以及数值收敛行为等方面的性能。CFDLLMBench为LLM驱动的复杂物理系统数值实验自动化开发和评估奠定了坚实的基础。代码和数据可在以下网址获取。 

---
# AI-driven formative assessment and adaptive learning in data-science education: Evaluating an LLM-powered virtual teaching assistant 

**Title (ZH)**: 基于AI的形成性评估与自适应学习在数据科学教育中的应用：评估一个由LLM驱动的虚拟教学助手 

**Authors**: Fadjimata I Anaroua, Qing Li, Yan Tang, Hong P. Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20369)  

**Abstract**: This paper presents VITA (Virtual Teaching Assistants), an adaptive distributed learning (ADL) platform that embeds a large language model (LLM)-powered chatbot (BotCaptain) to provide dialogic support, interoperable analytics, and integrity-aware assessment for workforce preparation in data science. The platform couples context-aware conversational tutoring with formative-assessment patterns designed to promote reflective reasoning. The paper describes an end-to-end data pipeline that transforms chat logs into Experience API (xAPI) statements, instructor dashboards that surface outliers for just-in-time intervention, and an adaptive pathway engine that routes learners among progression, reinforcement, and remediation content. The paper also benchmarks VITA conceptually against emerging tutoring architectures, including retrieval-augmented generation (RAG)--based assistants and Learning Tools Interoperability (LTI)--integrated hubs, highlighting trade-offs among content grounding, interoperability, and deployment complexity. Contributions include a reusable architecture for interoperable conversational analytics, a catalog of patterns for integrity-preserving formative assessment, and a practical blueprint for integrating adaptive pathways into data-science courses. The paper concludes with implementation lessons and a roadmap (RAG integration, hallucination mitigation, and LTI~1.3 / OpenID Connect) to guide multi-course evaluations and broader adoption. In light of growing demand and scalability constraints in traditional instruction, the approach illustrates how conversational AI can support engagement, timely feedback, and personalized learning at scale. Future work will refine the platform's adaptive intelligence and examine applicability across varied educational settings. 

**Abstract (ZH)**: VITA（虚拟教学助手）：一个集成大型语言模型的自适应分布式学习平台及其在数据科学工作准备中的应用 

---
# Interpreting Public Sentiment in Diplomacy Events: A Counterfactual Analysis Framework Using Large Language Models 

**Title (ZH)**: 外交事件中公众情绪解读：基于大规模语言模型的反事实分析框架 

**Authors**: Leyi Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20367)  

**Abstract**: Diplomatic events consistently prompt widespread public discussion and debate. Public sentiment plays a critical role in diplomacy, as a good sentiment provides vital support for policy implementation, helps resolve international issues, and shapes a nation's international image. Traditional methods for gauging public sentiment, such as large-scale surveys or manual content analysis of media, are typically time-consuming, labor-intensive, and lack the capacity for forward-looking analysis. We propose a novel framework that identifies specific modifications for diplomatic event narratives to shift public sentiment from negative to neutral or positive. First, we train a language model to predict public reaction towards diplomatic events. To this end, we construct a dataset comprising descriptions of diplomatic events and their associated public discussions. Second, guided by communication theories and in collaboration with domain experts, we predetermined several textual features for modification, ensuring that any alterations changed the event's narrative framing while preserving its core this http URL develop a counterfactual generation algorithm that employs a large language model to systematically produce modified versions of an original text. The results show that this framework successfully shifted public sentiment to a more favorable state with a 70\% success rate. This framework can therefore serve as a practical tool for diplomats, policymakers, and communication specialists, offering data-driven insights on how to frame diplomatic initiatives or report on events to foster a more desirable public sentiment. 

**Abstract (ZH)**: 外交事件一致引发广泛的公众讨论和辩论。公众情绪在外交活动中扮演着关键角色，良好的公众情绪为政策实施提供重要支持，有助于解决国际问题，并塑造一个国家的国际形象。传统的公众情绪评估方法，如大规模调查或手动分析媒体内容，通常耗时、劳动密集且缺乏前瞻性分析的能力。我们提出一种新的框架，以识别特定的外交事件叙事修改，将公众情绪从负面转变为中性或积极。首先，我们训练语言模型预测公众对外交事件的反应。为此，我们构建了一个包含外交事件描述及其相关公众讨论的数据集。其次，在沟通理论的指导下并与领域专家合作，我们预先确定了若干文本特征进行修改，确保任何改动改变了事件的叙事框架，但保留了其核心内容。然后，我们开发了一种反事实生成算法，使用大型语言模型系统地生成原始文本的修改版本。结果显示，该框架成功将公众情绪转向更积极的状态，成功率达到了70%。该框架因此可以作为外交官、政策制定者和沟通专家的实际工具，提供数据驱动的见解，说明如何塑造外交倡议或报道事件以促进更受欢迎的公众情绪。 

---
