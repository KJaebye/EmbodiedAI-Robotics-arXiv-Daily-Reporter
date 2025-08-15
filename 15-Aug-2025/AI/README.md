# Who Benefits from AI Explanations? Towards Accessible and Interpretable Systems 

**Title (ZH)**: AI解释的受益者是谁？面向可访问性和可解释性的系统研究 

**Authors**: Maria J. P. Peixoto, Akriti Pandey, Ahsan Zaman, Peter R. Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2508.10806)  

**Abstract**: As AI systems are increasingly deployed to support decision-making in critical domains, explainability has become a means to enhance the understandability of these outputs and enable users to make more informed and conscious choices. However, despite growing interest in the usability of eXplainable AI (XAI), the accessibility of these methods, particularly for users with vision impairments, remains underexplored. This paper investigates accessibility gaps in XAI through a two-pronged approach. First, a literature review of 79 studies reveals that evaluations of XAI techniques rarely include disabled users, with most explanations relying on inherently visual formats. Second, we present a four-part methodological proof of concept that operationalizes inclusive XAI design: (1) categorization of AI systems, (2) persona definition and contextualization, (3) prototype design and implementation, and (4) expert and user assessment of XAI techniques for accessibility. Preliminary findings suggest that simplified explanations are more comprehensible for non-visual users than detailed ones, and that multimodal presentation is required for more equitable interpretability. 

**Abstract (ZH)**: 随着AI系统在关键领域支持决策的应用越来越广泛，可解释性已成为提升这些输出可理解性的手段，帮助用户做出更知情和自觉的选择。然而，尽管对可解释性AI（XAI）的可用性越来越感兴趣，这些方法的无障碍性，尤其是对视觉障碍用户而言，尚未得到充分探索。本文通过两方面的研究探讨了XAI的无障碍缺口：首先，文献回顾79项研究发现，对XAI技术的评估很少包括残疾用户，大多数解释依赖于固有的视觉格式；其次，我们提出了一种四部分的方法论概念验证，以实现包容性XAI设计：（1）AI系统的分类，（2）角色定义和情境化，（3）原型设计与实现，以及（4）XAI技术的专家和用户无障碍评估。初步发现表明，简化解释比详细解释更易于非视觉用户理解，而多模态呈现对于更公平的可解释性是必要的。 

---
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
# Scaling Up without Fading Out: Goal-Aware Sparse GNN for RL-based Generalized Planning 

**Title (ZH)**: 不断提高效率而不减淡目标意识：面向RL基于通用规划的心理稀疏GNN 

**Authors**: Sangwoo Jeon, Juchul Shin, Gyeong-Tae Kim, YeonJe Cho, Seongwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.10747)  

**Abstract**: Generalized planning using deep reinforcement learning (RL) combined with graph neural networks (GNNs) has shown promising results in various symbolic planning domains described by PDDL. However, existing approaches typically represent planning states as fully connected graphs, leading to a combinatorial explosion in edge information and substantial sparsity as problem scales grow, especially evident in large grid-based environments. This dense representation results in diluted node-level information, exponentially increases memory requirements, and ultimately makes learning infeasible for larger-scale problems. To address these challenges, we propose a sparse, goal-aware GNN representation that selectively encodes relevant local relationships and explicitly integrates spatial features related to the goal. We validate our approach by designing novel drone mission scenarios based on PDDL within a grid world, effectively simulating realistic mission execution environments. Our experimental results demonstrate that our method scales effectively to larger grid sizes previously infeasible with dense graph representations and substantially improves policy generalization and success rates. Our findings provide a practical foundation for addressing realistic, large-scale generalized planning tasks. 

**Abstract (ZH)**: 使用深度强化学习（RL）结合图神经网络（GNNs）的广义规划在由PDDL描述的各种符号规划领域中取得了有希望的结果。然而，现有方法通常将规划状态表示为全连接图，这导致边信息的组合爆炸和随着问题规模增长而出现显著的稀疏性，尤其是在大型网格环境中尤为明显。这种密集表示会导致节点级信息的稀释，成指数地增加内存需求，并最终使得学习在大规模问题上变得不可行。为了解决这些挑战，我们提出了一种稀疏的目标导向的GNN表示方法，该方法选择性地编码相关的局部关系，并明确集成与目标相关的空间特征。我们通过在网格世界中基于PDDL设计新颖的无人机任务场景，有效模拟了现实的使命执行环境。实验结果表明，我们的方法能够有效地扩展到以前因密集图表示无法处理的更大网格规模，并显著提高了策略的一般化能力和成功率。我们的研究为解决现实的、大规模的广义规划任务提供了实用的基础。 

---
# Agentic Design Review System 

**Title (ZH)**: 代理设计评审系统 

**Authors**: Sayan Nag, K J Joseph, Koustava Goswami, Vlad I Morariu, Balaji Vasan Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2508.10745)  

**Abstract**: Evaluating graphic designs involves assessing it from multiple facets like alignment, composition, aesthetics and color choices. Evaluating designs in a holistic way involves aggregating feedback from individual expert reviewers. Towards this, we propose an Agentic Design Review System (AgenticDRS), where multiple agents collaboratively analyze a design, orchestrated by a meta-agent. A novel in-context exemplar selection approach based on graph matching and a unique prompt expansion method plays central role towards making each agent design aware. Towards evaluating this framework, we propose DRS-BENCH benchmark. Thorough experimental evaluation against state-of-the-art baselines adapted to the problem setup, backed-up with critical ablation experiments brings out the efficacy of Agentic-DRS in evaluating graphic designs and generating actionable feedback. We hope that this work will attract attention to this pragmatic, yet under-explored research direction. 

**Abstract (ZH)**: 评估图形设计涉及从对齐、构图、美学和颜色选择等多个方面进行评估。从整体上评估设计需要综合个体专家评审的反馈。为此，我们提出了一种代理设计审查系统（AgenticDRS），其中多个代理协同分析设计，由一个元代理协调。基于图匹配的新型上下文相关示例选择方法和独特的提示扩展方法在使每个代理了解设计方面发挥核心作用。为了评估该框架，我们提出了DRS-BENCH基准。通过针对问题设置适应的最先进的基线进行彻底的实验评估，并结合关键的消融实验，突显了Agentic-DRS在评估图形设计和生成可操作反馈方面的有效性。我们希望这项工作能够引起对这一实用且尚未充分探索的研究方向的关注。 

---
# GenOM: Ontology Matching with Description Generation and Large Language Model 

**Title (ZH)**: GenOM：基于描述生成和大规模语言模型的概念匹配 

**Authors**: Yiping Song, Jiaoyan Chen, Renate A. Schmidt  

**Link**: [PDF](https://arxiv.org/pdf/2508.10703)  

**Abstract**: Ontology matching (OM) plays an essential role in enabling semantic interoperability and integration across heterogeneous knowledge sources, particularly in the biomedical domain which contains numerous complex concepts related to diseases and pharmaceuticals. This paper introduces GenOM, a large language model (LLM)-based ontology alignment framework, which enriches the semantic representations of ontology concepts via generating textual definitions, retrieves alignment candidates with an embedding model, and incorporates exact matching-based tools to improve precision. Extensive experiments conducted on the OAEI Bio-ML track demonstrate that GenOM can often achieve competitive performance, surpassing many baselines including traditional OM systems and recent LLM-based methods. Further ablation studies confirm the effectiveness of semantic enrichment and few-shot prompting, highlighting the framework's robustness and adaptability. 

**Abstract (ZH)**: 基于大型语言模型的本体匹配框架GenOM及其在生物医学领域的应用 

---
# STEP: Stepwise Curriculum Learning for Context-Knowledge Fusion in Conversational Recommendation 

**Title (ZH)**: Stepwise Curriculum Learning for Context-Knowledge Fusion in Conversational Recommendation 

**Authors**: Zhenye Yang, Jinpeng Chen, Huan Li, Xiongnan Jin, Xuanyang Li, Junwei Zhang, Hongbo Gao, Kaimin Wei, Senzhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10669)  

**Abstract**: Conversational recommender systems (CRSs) aim to proactively capture user preferences through natural language dialogue and recommend high-quality items. To achieve this, CRS gathers user preferences via a dialog module and builds user profiles through a recommendation module to generate appropriate recommendations. However, existing CRS faces challenges in capturing the deep semantics of user preferences and dialogue context. In particular, the efficient integration of external knowledge graph (KG) information into dialogue generation and recommendation remains a pressing issue. Traditional approaches typically combine KG information directly with dialogue content, which often struggles with complex semantic relationships, resulting in recommendations that may not align with user expectations.
To address these challenges, we introduce STEP, a conversational recommender centered on pre-trained language models that combines curriculum-guided context-knowledge fusion with lightweight task-specific prompt tuning. At its heart, an F-Former progressively aligns the dialogue context with knowledge-graph entities through a three-stage curriculum, thus resolving fine-grained semantic mismatches. The fused representation is then injected into the frozen language model via two minimal yet adaptive prefix prompts: a conversation prefix that steers response generation toward user intent and a recommendation prefix that biases item ranking toward knowledge-consistent candidates. This dual-prompt scheme allows the model to share cross-task semantics while respecting the distinct objectives of dialogue and recommendation. Experimental results show that STEP outperforms mainstream methods in the precision of recommendation and dialogue quality in two public datasets. 

**Abstract (ZH)**: 基于预训练语言模型的渐进式上下文-知识融合对话推荐系统（STEP） 

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
# PASS: Probabilistic Agentic Supernet Sampling for Interpretable and Adaptive Chest X-Ray Reasoning 

**Title (ZH)**: PASS: 概率代理超网络采样方法实现可
user
PASS: Probabilistic Agentic Supernet Sampling for Interpretable and Adaptive Chest X-Ray Reasoning yab.getLogger()助手翻译错误，请重新翻译，要符合学术规范。 

**Authors**: Yushi Feng, Junye Du, Yingying Hong, Qifan Wang, Lequan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10501)  

**Abstract**: Existing tool-augmented agentic systems are limited in the real world by (i) black-box reasoning steps that undermine trust of decision-making and pose safety risks, (ii) poor multimodal integration, which is inherently critical for healthcare tasks, and (iii) rigid and computationally inefficient agentic pipelines. We introduce PASS (Probabilistic Agentic Supernet Sampling), the first multimodal framework to address these challenges in the context of Chest X-Ray (CXR) reasoning. PASS adaptively samples agentic workflows over a multi-tool graph, yielding decision paths annotated with interpretable probabilities. Given the complex CXR reasoning task with multimodal medical data, PASS leverages its learned task-conditioned distribution over the agentic supernet. Thus, it adaptively selects the most suitable tool at each supernet layer, offering probability-annotated trajectories for post-hoc audits and directly enhancing medical AI safety. PASS also continuously compresses salient findings into an evolving personalized memory, while dynamically deciding whether to deepen its reasoning path or invoke an early exit for efficiency. To optimize a Pareto frontier balancing performance and cost, we design a novel three-stage training procedure, including expert knowledge warm-up, contrastive path-ranking, and cost-aware reinforcement learning. To facilitate rigorous evaluation, we introduce CAB-E, a comprehensive benchmark for multi-step, safety-critical, free-form CXR reasoning. Experiments across various benchmarks validate that PASS significantly outperforms strong baselines in multiple metrics (e.g., accuracy, AUC, LLM-J.) while balancing computational costs, pushing a new paradigm shift towards interpretable, adaptive, and multimodal medical agentic systems. 

**Abstract (ZH)**: PASS：面向胸部X光图像推理的概率代理超网络采样 

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
# We-Math 2.0: A Versatile MathBook System for Incentivizing Visual Mathematical Reasoning 

**Title (ZH)**: We-Math 2.0: 一个激励视觉数学推理的多功能数学书系统 

**Authors**: Runqi Qiao, Qiuna Tan, Peiqing Yang, Yanzi Wang, Xiaowan Wang, Enhui Wan, Sitong Zhou, Guanting Dong, Yuchen Zeng, Yida Xu, Jie Wang, Chong Sun, Chen Li, Honggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10433)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities across various tasks, but still struggle with complex mathematical reasoning. Existing research primarily focuses on dataset construction and method optimization, often overlooking two critical aspects: comprehensive knowledge-driven design and model-centric data space modeling. In this paper, we introduce We-Math 2.0, a unified system that integrates a structured mathematical knowledge system, model-centric data space modeling, and a reinforcement learning (RL)-based training paradigm to comprehensively enhance the mathematical reasoning abilities of MLLMs. The key contributions of We-Math 2.0 are fourfold: (1) MathBook Knowledge System: We construct a five-level hierarchical system encompassing 491 knowledge points and 1,819 fundamental principles. (2) MathBook-Standard & Pro: We develop MathBook-Standard, a dataset that ensures broad conceptual coverage and flexibility through dual expansion. Additionally, we define a three-dimensional difficulty space and generate 7 progressive variants per problem to build MathBook-Pro, a challenging dataset for robust training. (3) MathBook-RL: We propose a two-stage RL framework comprising: (i) Cold-Start Fine-tuning, which aligns the model with knowledge-oriented chain-of-thought reasoning; and (ii) Progressive Alignment RL, leveraging average-reward learning and dynamic data scheduling to achieve progressive alignment across difficulty levels. (4) MathBookEval: We introduce a comprehensive benchmark covering all 491 knowledge points with diverse reasoning step distributions. Experimental results show that MathBook-RL performs competitively with existing baselines on four widely-used benchmarks and achieves strong results on MathBookEval, suggesting promising generalization in mathematical reasoning. 

**Abstract (ZH)**: 多模态大规模语言模型（MLLMs）在各种任务中展现了 impressive 的能力，但在复杂数学推理方面仍面临挑战。现有研究主要集中在数据集构建和方法优化上，常常忽视两个关键方面：全面的知识驱动设计和以模型为中心的数据空间建模。本文提出了 We-Math 2.0，这是一个统一系统，结合了结构化的数学知识系统、以模型为中心的数据空间建模和基于强化学习（RL）的训练范式，以全面增强 MLLMs 的数学推理能力。We-Math 2.0 的主要贡献包括四个方面：（1）MathBook 知识系统：构建了一个五级层次系统，包含 491 个知识点和 1,819 个基本原理。（2）MathBook-Standard & Pro：开发了 MathBook-Standard 数据集，通过双扩展确保广泛的概念覆盖面和灵活性。此外，定义了一个三维难度空间，并为每个问题生成 7 个递进变体，构建了 MathBook-Pro 这个具有挑战性的数据集，用于稳健训练。（3）MathBook-RL：提出一个两阶段 RL 框架，包括：启动阶段微调，使模型与知识导向的推理链相一致；以及渐进对准 RL，利用平均回报学习和动态数据调度实现不同难度层次的渐进对准。（4）MathBookEval：引入了一个全面的基准测试，涵盖所有 491 个知识点，具有多样的推理步骤分布。实验结果表明，MathBook-RL 在四个广泛使用的基准测试中表现出色，并在 MathBookEval 中取得了优异成绩，这表明其在数学推理中的潜在泛化能力。 

---
# MM-Food-100K: A 100,000-Sample Multimodal Food Intelligence Dataset with Verifiable Provenance 

**Title (ZH)**: MM-Food-100K: 一个带有可验证来源的100,000样本多模态食品智能数据集 

**Authors**: Yi Dong, Yusuke Muraoka, Scott Shi, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10429)  

**Abstract**: We present MM-Food-100K, a public 100,000-sample multimodal food intelligence dataset with verifiable provenance. It is a curated approximately 10% open subset of an original 1.2 million, quality-accepted corpus of food images annotated for a wide range of information (such as dish name, region of creation). The corpus was collected over six weeks from over 87,000 contributors using the Codatta contribution model, which combines community sourcing with configurable AI-assisted quality checks; each submission is linked to a wallet address in a secure off-chain ledger for traceability, with a full on-chain protocol on the roadmap. We describe the schema, pipeline, and QA, and validate utility by fine-tuning large vision-language models (ChatGPT 5, ChatGPT OSS, Qwen-Max) on image-based nutrition prediction. Fine-tuning yields consistent gains over out-of-box baselines across standard metrics; we report results primarily on the MM-Food-100K subset. We release MM-Food-100K for publicly free access and retain approximately 90% for potential commercial access with revenue sharing to contributors. 

**Abstract (ZH)**: 我们呈现MM-Food-100K：一个具有可验证来源的公共10万样本多模态食品智能数据集 

---
# HiRef: Leveraging Hierarchical Ontology and Network Refinement for Robust Medication Recommendation 

**Title (ZH)**: HiRef: 利用层次化本体和网络精炼实现稳健的药物推荐 

**Authors**: Yan Ting Chok, Soyon Park, Seungheun Baek, Hajung Kim, Junhyun Lee, Jaewoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10425)  

**Abstract**: Medication recommendation is a crucial task for assisting physicians in making timely decisions from longitudinal patient medical records. However, real-world EHR data present significant challenges due to the presence of rarely observed medical entities and incomplete records that may not fully capture the clinical ground truth. While data-driven models trained on longitudinal Electronic Health Records often achieve strong empirical performance, they struggle to generalize under missing or novel conditions, largely due to their reliance on observed co-occurrence patterns. To address these issues, we propose Hierarchical Ontology and Network Refinement for Robust Medication Recommendation (HiRef), a unified framework that combines two complementary structures: (i) the hierarchical semantics encoded in curated medical ontologies, and (ii) refined co-occurrence patterns derived from real-world EHRs. We embed ontology entities in hyperbolic space, which naturally captures tree-like relationships and enables knowledge transfer through shared ancestors, thereby improving generalizability to unseen codes. To further improve robustness, we introduce a prior-guided sparse regularization scheme that refines the EHR co-occurrence graph by suppressing spurious edges while preserving clinically meaningful associations. Our model achieves strong performance on EHR benchmarks (MIMIC-III and MIMIC-IV) and maintains high accuracy under simulated unseen-code settings. Extensive experiments with comprehensive ablation studies demonstrate HiRef's resilience to unseen medical codes, supported by in-depth analyses of the learned sparsified graph structure and medical code embeddings. 

**Abstract (ZH)**: 基于层次 ontology 和网络 refinement 的稳健药物推荐 (HiRef) 

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
# Multi-Agent Trust Region Policy Optimisation: A Joint Constraint Approach 

**Title (ZH)**: 多代理信任区域策略优化：一种联合约束方法 

**Authors**: Chak Lam Shek, Guangyao Shi, Pratap Tokekar  

**Link**: [PDF](https://arxiv.org/pdf/2508.10340)  

**Abstract**: Multi-agent reinforcement learning (MARL) requires coordinated and stable policy updates among interacting agents. Heterogeneous-Agent Trust Region Policy Optimization (HATRPO) enforces per-agent trust region constraints using Kullback-Leibler (KL) divergence to stabilize training. However, assigning each agent the same KL threshold can lead to slow and locally optimal updates, especially in heterogeneous settings. To address this limitation, we propose two approaches for allocating the KL divergence threshold across agents: HATRPO-W, a Karush-Kuhn-Tucker-based (KKT-based) method that optimizes threshold assignment under global KL constraints, and HATRPO-G, a greedy algorithm that prioritizes agents based on improvement-to-divergence ratio. By connecting sequential policy optimization with constrained threshold scheduling, our approach enables more flexible and effective learning in heterogeneous-agent settings. Experimental results demonstrate that our methods significantly boost the performance of HATRPO, achieving faster convergence and higher final rewards across diverse MARL benchmarks. Specifically, HATRPO-W and HATRPO-G achieve comparable improvements in final performance, each exceeding 22.5%. Notably, HATRPO-W also demonstrates more stable learning dynamics, as reflected by its lower variance. 

**Abstract (ZH)**: 多智能体强化学习中基于KL散度的异质智能体信任区域策略优化（HATRPO）改进方法 

---
# A Curriculum Learning Approach to Reinforcement Learning: Leveraging RAG for Multimodal Question Answering 

**Title (ZH)**: 基于课程学习的强化学习方法：利用RAG进行多模态问答 

**Authors**: Chenliang Zhang, Lin Wang, Yuanyuan Lu, Yusheng Qi, Kexin Wang, Peixu Hou, Wenshi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10337)  

**Abstract**: This paper describes the solutions of the Dianping-Trust-Safety team for the META CRAG-MM challenge. The challenge requires building a comprehensive retrieval-augmented generation system capable for multi-modal multi-turn question answering. The competition consists of three tasks: (1) answering questions using structured data retrieved from an image-based mock knowledge graph, (2) synthesizing information from both knowledge graphs and web search results, and (3) handling multi-turn conversations that require context understanding and information aggregation from multiple sources. For Task 1, our solution is based on the vision large language model, enhanced by supervised fine-tuning with knowledge distilled from GPT-4.1. We further applied curriculum learning strategies to guide reinforcement learning, resulting in improved answer accuracy and reduced hallucination. For Task 2 and Task 3, we additionally leveraged web search APIs to incorporate external knowledge, enabling the system to better handle complex queries and multi-turn conversations. Our approach achieved 1st place in Task 1 with a significant lead of 52.38\%, and 3rd place in Task 3, demonstrating the effectiveness of the integration of curriculum learning with reinforcement learning in our training pipeline. 

**Abstract (ZH)**: 本文描述了Dianping-Trust-Safety团队在META CRAG-MM挑战中提出的解决方案。该挑战要求构建一个综合的检索增强生成系统，具备多模态多轮问答能力。比赛包括三个任务：（1）利用基于图像的模拟知识图谱检索结构化数据进行问题回答，（2）从知识图谱和网络搜索结果中合成信息，和（3）处理需要上下文理解并从多个来源聚合信息的多轮对话。在任务1中，我们的解决方案基于视觉大型语言模型，并通过从GPT-4.1知识中监督微调来增强，同时应用了课程学习策略来引导强化学习，从而提高了答案的准确性并减少了幻觉。在任务2和任务3中，我们还利用了网络搜索API来引入外部知识，使系统能够更好地处理复杂的查询和多轮对话。我们的方法在任务1中取得了第一名，领先优势为52.38%，并在任务3中获得第三名，证明了在我们的训练管道中将课程学习与强化学习结合的有效性。 

---
# Promoting Efficient Reasoning with Verifiable Stepwise Reward 

**Title (ZH)**: 促进高效推理的可验证逐步奖励方法 

**Authors**: Chuhuai Yue, Chengqi Dong, Yinan Gao, Hang He, Jiajun Chai, Guojun Yin, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.10293)  

**Abstract**: Large reasoning models (LRMs) have recently achieved significant progress in complex reasoning tasks, aided by reinforcement learning with verifiable rewards. However, LRMs often suffer from overthinking, expending excessive computation on simple problems and reducing efficiency. Existing efficient reasoning methods typically require accurate task assessment to preset token budgets or select reasoning modes, which limits their flexibility and reliability. In this work, we revisit the essence of overthinking and identify that encouraging effective steps while penalizing ineffective ones is key to its solution. To this end, we propose a novel rule-based verifiable stepwise reward mechanism (VSRM), which assigns rewards based on the performance of intermediate states in the reasoning trajectory. This approach is intuitive and naturally fits the step-by-step nature of reasoning tasks. We conduct extensive experiments on standard mathematical reasoning benchmarks, including AIME24 and AIME25, by integrating VSRM with PPO and Reinforce++. Results show that our method achieves substantial output length reduction while maintaining original reasoning performance, striking an optimal balance between efficiency and accuracy. Further analysis of overthinking frequency and pass@k score before and after training demonstrates that our approach in deed effectively suppresses ineffective steps and encourages effective reasoning, fundamentally alleviating the overthinking problem. All code will be released upon acceptance. 

**Abstract (ZH)**: 大型推理模型（LRMs）在复杂推理任务中取得了显著进展，借助可验证奖励的强化学习。然而，LRMs 通常会陷入过度思考，对简单问题耗用过多计算资源，降低效率。现有高效推理方法通常需要准确的任务评估来预设令牌预算或选择推理模式，这限制了其灵活性和可靠性。在本文中，我们重新审视过度思考的本质，并发现鼓励有效步骤同时惩罚无效步骤是其解决的关键。为实现这一目标，我们提出了一种新的基于规则的可验证步骤奖励机制（VSRM），该机制根据推理轨迹中中间状态的性能分配奖励。该方法直观且自然地符合推理任务的逐步性质。我们通过将VSRM与PPO和Reinforce++集成，在标准数学推理基准AIME24和AIME25上进行了广泛实验。结果表明，我们的方法在保持原始推理性能的同时实现显著的输出长度减少，实现了效率和准确性的最优平衡。进一步分析训练前后过度思考频率和pass@k得分表明，我们的方法实际上有效地抑制了无效步骤，促进了有效推理，从根本上缓解了过度思考问题。接受后将发布所有代码。 

---
# Why Cannot Large Language Models Ever Make True Correct Reasoning? 

**Title (ZH)**: 大型语言模型为何永远无法进行真正的正确推理？ 

**Authors**: Jingde Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.10265)  

**Abstract**: Recently, with the application progress of AIGC tools based on large language models (LLMs), led by ChatGPT, many AI experts and more non-professionals are trumpeting the "understanding ability" and "reasoning ability" of the LLMs. The present author considers that the so-called "understanding ability" and "reasoning ability" of LLMs are just illusions of those people who with vague concepts. In fact, the LLMs can never have the true understanding ability and true reasoning ability. This paper intents to explain that, because the essential limitations of their working principle, the LLMs can never have the ability of true correct reasoning. 

**Abstract (ZH)**: 近年来，随着以大型语言模型（LLMs）为底层技术的AIGC工具，特别是ChatGPT的广泛应用，许多AI专家甚至非专业人士都在极力吹捧LLMs的“理解能力”和“推理能力”。本文认为，所谓的LLMs的“理解能力”和“推理能力”只是模糊概念下的幻觉。事实上，LLMs永远不可能具备真正的理解能力和真正的推理能力。本文旨在解释，由于其工作原理的本质限制，LLMs永远不可能具备真正的正确推理能力。 

---
# Extending the Entropic Potential of Events for Uncertainty Quantification and Decision-Making in Artificial Intelligence 

**Title (ZH)**: 扩展事件的_entropic潜力_用于人工智能中的不确定性量化与决策制定 

**Authors**: Mark Zilberman  

**Link**: [PDF](https://arxiv.org/pdf/2508.10241)  

**Abstract**: This work demonstrates how the concept of the entropic potential of events -- a parameter quantifying the influence of discrete events on the expected future entropy of a system -- can enhance uncertainty quantification, decision-making, and interpretability in artificial intelligence (AI). Building on its original formulation in physics, the framework is adapted for AI by introducing an event-centric measure that captures how actions, observations, or other discrete occurrences impact uncertainty at future time horizons. Both the original and AI-adjusted definitions of entropic potential are formalized, with the latter emphasizing conditional expectations to account for counterfactual scenarios. Applications are explored in policy evaluation, intrinsic reward design, explainable AI, and anomaly detection, highlighting the metric's potential to unify and strengthen uncertainty modeling in intelligent systems. Conceptual examples illustrate its use in reinforcement learning, Bayesian inference, and anomaly detection, while practical considerations for computation in complex AI models are discussed. The entropic potential framework offers a theoretically grounded, interpretable, and versatile approach to managing uncertainty in AI, bridging principles from thermodynamics, information theory, and machine learning. 

**Abstract (ZH)**: 本研究展示了事件熵潜能的概念——一个量化离散事件对未来系统预期熵影响的参数——如何增强人工智能中的不确定性量化、决策制定和可解释性。基于其在物理中的原始表述，该框架通过引入以事件为中心的度量来适应人工智能，该度量捕获了行动、观察或其他离散事件对未来时间 horizons 不确定性的影响。原定义和调整后的 AI 定义的熵潜能都被形式化，后者强调条件期望以考虑到假设场景。熵潜能的应用在政策评估、内在奖励设计、可解释人工智能和异常检测中得到了探索，突显了该度量在智能系统中统一和增强不确定性建模的潜力。概念性示例展示了其在强化学习、贝叶斯推断和异常检测中的应用，同时讨论了在复杂人工智能模型中计算的实用考虑。熵潜能框架提供了一种理论基础扎实、可解释且灵活的方法来管理人工智能中的不确定性，融合了热力学、信息论和机器学习的原则。 

---
# KompeteAI: Accelerated Autonomous Multi-Agent System for End-to-End Pipeline Generation for Machine Learning Problems 

**Title (ZH)**: KompeteAI：端到端机器学习问题自主多Agent系统加速生成管道的系统 

**Authors**: Stepan Kulibaba, Artem Dzhalilov, Roman Pakhomov, Oleg Svidchenko, Alexander Gasnikov, Aleksei Shpilman  

**Link**: [PDF](https://arxiv.org/pdf/2508.10177)  

**Abstract**: Recent Large Language Model (LLM)-based AutoML systems demonstrate impressive capabilities but face significant limitations such as constrained exploration strategies and a severe execution bottleneck. Exploration is hindered by one-shot methods lacking diversity and Monte Carlo Tree Search (MCTS) approaches that fail to recombine strong partial solutions. The execution bottleneck arises from lengthy code validation cycles that stifle iterative refinement. To overcome these challenges, we introduce KompeteAI, a novel AutoML framework with dynamic solution space exploration. Unlike previous MCTS methods that treat ideas in isolation, KompeteAI introduces a merging stage that composes top candidates. We further expand the hypothesis space by integrating Retrieval-Augmented Generation (RAG), sourcing ideas from Kaggle notebooks and arXiv papers to incorporate real-world strategies. KompeteAI also addresses the execution bottleneck via a predictive scoring model and an accelerated debugging method, assessing solution potential using early stage metrics to avoid costly full-code execution. This approach accelerates pipeline evaluation 6.9 times. KompeteAI outperforms leading methods (e.g., RD-agent, AIDE, and Ml-Master) by an average of 3\% on the primary AutoML benchmark, MLE-Bench. Additionally, we propose Kompete-bench to address limitations in MLE-Bench, where KompeteAI also achieves state-of-the-art results 

**Abstract (ZH)**: 基于大语言模型（LLM）的AutoML系统最近展现了令人印象深刻的性能，但面临着探索策略受限和执行瓶颈等重大挑战。探索受限于一-shot方法缺乏多样性，以及蒙特卡洛树搜索（MCTS）方法无法重组强部分解。执行瓶颈源于冗长的代码验证周期，阻碍了迭代优化。为克服这些挑战，我们提出KompeteAI，这是一种具有动态解空间探索的新颖AutoML框架。不同于之前的MCTS方法孤立处理想法，KompeteAI引入了合并阶段，将顶级候选方案组合。我们进一步通过集成检索增强生成（RAG）扩展假设空间，从Kaggle笔记本和arXiv论文中汲取实际策略。KompeteAI通过预测评分模型和加速调试方法来解决执行瓶颈，使用早期指标评估解的潜力，以避免昂贵的完整代码执行。这种方法将管道评估加速了6.9倍。KompeteAI在主要的AutoML基准ML-Evaluation-Benchmarks上平均优于领先方法（如RD-agent、AIDE和Ml-Master）3%。此外，我们提出了Kompete-bench以解决ML-Evaluation-Benchmarks的局限性， KompeteAI在其中也取得了最先进的结果。 

---
# Pruning Long Chain-of-Thought of Large Reasoning Models via Small-Scale Preference Optimization 

**Title (ZH)**: 基于小规模偏好优化的大推理模型长链思考修剪 

**Authors**: Bin Hong, Jiayu Liu, Zhenya Huang, Kai Zhang, Mengdi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10164)  

**Abstract**: Recent advances in Large Reasoning Models (LRMs) have demonstrated strong performance on complex tasks through long Chain-of-Thought (CoT) reasoning. However, their lengthy outputs increase computational costs and may lead to overthinking, raising challenges in balancing reasoning effectiveness and efficiency. Current methods for efficient reasoning often compromise reasoning quality or require extensive resources. This paper investigates efficient methods to reduce the generation length of LRMs. We analyze generation path distributions and filter generated trajectories through difficulty estimation. Subsequently, we analyze the convergence behaviors of the objectives of various preference optimization methods under a Bradley-Terry loss based framework. Based on the analysis, we propose Length Controlled Preference Optimization (LCPO) that directly balances the implicit reward related to NLL loss. LCPO can effectively learn length preference with limited data and training. Extensive experiments demonstrate that our approach significantly reduces the average output length by over 50\% across multiple benchmarks while maintaining the reasoning performance. Our work highlights the potential for computationally efficient approaches in guiding LRMs toward efficient reasoning. 

**Abstract (ZH)**: Recent Advances in Large Reasoning Models (LRMs) and Efficient Generation Methods 

---
# Improving and Evaluating Open Deep Research Agents 

**Title (ZH)**: 改进和评估开源深度研究代理 

**Authors**: Doaa Allabadi, Kyle Bradbury, Jordan M. Malof  

**Link**: [PDF](https://arxiv.org/pdf/2508.10152)  

**Abstract**: We focus here on Deep Research Agents (DRAs), which are systems that can take a natural language prompt from a user, and then autonomously search for, and utilize, internet-based content to address the prompt. Recent DRAs have demonstrated impressive capabilities on public benchmarks however, recent research largely involves proprietary closed-source systems. At the time of this work, we only found one open-source DRA, termed Open Deep Research (ODR). In this work we adapt the challenging recent BrowseComp benchmark to compare ODR to existing proprietary systems. We propose BrowseComp-Small (BC-Small), comprising a subset of BrowseComp, as a more computationally-tractable DRA benchmark for academic labs. We benchmark ODR and two other proprietary systems on BC-Small: one system from Anthropic and one system from Google. We find that all three systems achieve 0% accuracy on the test set of 60 questions. We introduce three strategic improvements to ODR, resulting in the ODR+ model, which achieves a state-of-the-art 10% success rate on BC-Small among both closed-source and open-source systems. We report ablation studies indicating that all three of our improvements contributed to the success of ODR+. 

**Abstract (ZH)**: 我们专注于深度研究代理（DRAs），这些系统可以从用户处接收自然语言提示，然后自主搜索和利用互联网内容来应对提示。尽管最近的DRAs在公共基准测试中展现了令人印象深刻的性能，但大部分近期研究涉及专有和封闭源代码系统。在本研究进行时，我们仅发现一个开源DRA，称为Open Deep Research（ODR）。在这项工作中，我们将具有挑战性的近期BrowseComp基准测试改编，以比较ODR与现有的专有系统。我们提出BrowseComp-Small（BC-Small），它是BrowseComp的一部分，作为更易于计算的DRA学术实验室基准。我们对ODR及两个其他专有系统在BC-Small上的表现进行了基准测试：来自Anthropic的一个系统和来自Google的一个系统。我们发现，所有三个系统在包含60个问题的测试集上的准确率为0%。我们引入了三种战略性的改进，从而生成了ODR+模型，在BC-Small基准上，ODR+模型在专有和开源系统中达到了最佳的10%成功率。我们报告的消融研究显示，我们的三种改进均对ODR+的成功做出了贡献。 

---
# Agentic AI Frameworks: Architectures, Protocols, and Design Challenges 

**Title (ZH)**: 代理性人工智能框架：架构、协议与设计挑战 

**Authors**: Hana Derouiche, Zaki Brahmi, Haithem Mazeni  

**Link**: [PDF](https://arxiv.org/pdf/2508.10146)  

**Abstract**: The emergence of Large Language Models (LLMs) has ushered in a transformative paradigm in artificial intelligence, Agentic AI, where intelligent agents exhibit goal-directed autonomy, contextual reasoning, and dynamic multi-agent coordination. This paper provides a systematic review and comparative analysis of leading Agentic AI frameworks, including CrewAI, LangGraph, AutoGen, Semantic Kernel, Agno, Google ADK, and MetaGPT, evaluating their architectural principles, communication mechanisms, memory management, safety guardrails, and alignment with service-oriented computing paradigms. Furthermore, we identify key limitations, emerging trends, and open challenges in the field. To address the issue of agent communication, we conduct an in-depth analysis of protocols such as the Contract Net Protocol (CNP), Agent-to-Agent (A2A), Agent Network Protocol (ANP), and Agora. Our findings not only establish a foundational taxonomy for Agentic AI systems but also propose future research directions to enhance scalability, robustness, and interoperability. This work serves as a comprehensive reference for researchers and practitioners working to advance the next generation of autonomous AI systems. 

**Abstract (ZH)**: 大型语言模型(Large Language Models)的出现推动了人工智能领域的范式转变，即自主人工智能(Agentic AI)，其中智能代理表现出目标导向的自主性、情境推理和动态多代理协调。本文对CrewAI、LangGraph、AutoGen、Semantic Kernel、Agno、Google ADK和MetaGPT等领先自主人工智能框架进行了系统审查和比较分析，评估了它们的架构原理、通信机制、内存管理、安全护栏以及与面向服务计算范式的契合度。此外，我们还识别出该领域的关键局限性、新兴趋势和开放式挑战。为了解决代理通信问题，我们深入分析了合同网协议(CNP)、代理间通信(A2A)、代理网络协议(ANP)和Agora等协议。我们的研究不仅建立了自主人工智能系统的分类框架，还提出了未来的研究方向以增强可伸缩性、鲁棒性和互操作性。本研究为致力于推进下一代自主人工智能系统的研究人员和实践者提供了全面的参考。 

---
# MCP-Orchestrated Multi-Agent System for Automated Disinformation Detection 

**Title (ZH)**: MCP- orchestrated 多代理系统用于自动化虚假信息检测 

**Authors**: Alexandru-Andrei Avram, Adrian Groza, Alexandru Lecu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10143)  

**Abstract**: The large spread of disinformation across digital platforms creates significant challenges to information integrity. This paper presents a multi-agent system that uses relation extraction to detect disinformation in news articles, focusing on titles and short text snippets. The proposed Agentic AI system combines four agents: (i) a machine learning agent (logistic regression), (ii) a Wikipedia knowledge check agent (which relies on named entity recognition), (iii) a coherence detection agent (using LLM prompt engineering), and (iv) a web-scraped data analyzer that extracts relational triplets for fact checking. The system is orchestrated via the Model Context Protocol (MCP), offering shared context and live learning across components. Results demonstrate that the multi-agent ensemble achieves 95.3% accuracy with an F1 score of 0.964, significantly outperforming individual agents and traditional approaches. The weighted aggregation method, mathematically derived from individual agent misclassification rates, proves superior to algorithmic threshold optimization. The modular architecture makes the system easily scalable, while also maintaining details of the decision processes. 

**Abstract (ZH)**: 数字平台上的虚假信息广泛传播对信息完整性构成了重大挑战。本文提出了一种多Agent系统，利用关系提取检测新闻文章中的虚假信息，专注于标题和短文本片段。提出的Agentic AI系统结合了四个Agent：（i）机器学习Agent（逻辑回归），（ii）维基百科知识验证Agent（依赖命名实体识别），（iii）一致性检测Agent（使用LLM提示工程），以及（iv）网页抓取数据分析师，提取关系三元组进行事实核查。系统通过模型上下文协议（MCP）协调，提供跨组件的共享上下文和实时学习。结果表明，多Agent集合的准确率为95.3%，F1分数为0.964，显著优于单个Agent和传统方法。加权聚合方法，从单个Agent的错误分类率中数学推导而来，优于算法阈值优化。模块化的架构使得系统易于扩展，同时保持决策过程的详细信息。 

---
# Amazon Nova AI Challenge -- Trusted AI: Advancing secure, AI-assisted software development 

**Title (ZH)**: Amazon Nova AI挑战赛 —— 可信赖AI：推动安全的AI辅助软件开发 

**Authors**: Sattvik Sahai, Prasoon Goyal, Michael Johnston, Anna Gottardi, Yao Lu, Lucy Hu, Luke Dai, Shaohua Liu, Samyuth Sagi, Hangjie Shi, Desheng Zhang, Lavina Vaz, Leslie Ball, Maureen Murray, Rahul Gupta, Shankar Ananthakrishna  

**Link**: [PDF](https://arxiv.org/pdf/2508.10108)  

**Abstract**: AI systems for software development are rapidly gaining prominence, yet significant challenges remain in ensuring their safety. To address this, Amazon launched the Trusted AI track of the Amazon Nova AI Challenge, a global competition among 10 university teams to drive advances in secure AI. In the challenge, five teams focus on developing automated red teaming bots, while the other five create safe AI assistants. This challenge provides teams with a unique platform to evaluate automated red-teaming and safety alignment methods through head-to-head adversarial tournaments where red teams have multi-turn conversations with the competing AI coding assistants to test their safety alignment. Along with this, the challenge provides teams with a feed of high quality annotated data to fuel iterative improvement. Throughout the challenge, teams developed state-of-the-art techniques, introducing novel approaches in reasoning-based safety alignment, robust model guardrails, multi-turn jail-breaking, and efficient probing of large language models (LLMs). To support these efforts, the Amazon Nova AI Challenge team made substantial scientific and engineering investments, including building a custom baseline coding specialist model for the challenge from scratch, developing a tournament orchestration service, and creating an evaluation harness. This paper outlines the advancements made by university teams and the Amazon Nova AI Challenge team in addressing the safety challenges of AI for software development, highlighting this collaborative effort to raise the bar for AI safety. 

**Abstract (ZH)**: AI系统在软件开发中的应用迅速增长，但确保其安全性的挑战仍然存在。为应对这一挑战，亚马逊发起了亚马逊诺瓦AI挑战中的可信AI赛道，这是一个由10所大学团队参与的全球竞赛，旨在推动安全AI的发展。在此次挑战中，五支队伍专注于开发自动化红队机器人，而另外五支队伍则创建安全的AI助手。此挑战为团队提供了独特的平台，通过头对头的 adversarial 对抗 tournament，红队与竞争的AI编码助手进行多轮对话以测试其安全性对齐。此外，挑战还为团队提供了高质量标注数据流，以供迭代改进。在整个挑战过程中，团队研发了最先进的技术，引入了基于推理的安全对齐、稳健模型护栏、多轮逃狱以及高效探索大型语言模型的新方法。为了支持这些努力，亚马逊诺瓦AI挑战团队进行了大量的科学和工程投资，包括从零开始构建定制化的挑战专用编码专家模型，开发tournament管弦服务，以及创建评估框架。本文概述了大学团队和亚马逊诺瓦AI挑战团队在应对AI软件开发安全性挑战方面取得的进步，强调了这一合作努力以提高AI安全性标准。 

---
# A Survey of Optimization Modeling Meets LLMs: Progress and Future Directions 

**Title (ZH)**: 优化建模遇见大语言模型：进展与未来方向 

**Authors**: Ziyang Xiao, Jingrong Xie, Lilin Xu, Shisi Guan, Jingyan Zhu, Xiongwei Han, Xiaojin Fu, WingYin Yu, Han Wu, Wei Shi, Qingcan Kang, Jiahui Duan, Tao Zhong, Mingxuan Yuan, Jia Zeng, Yuan Wang, Gang Chen, Dongxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10047)  

**Abstract**: By virtue of its great utility in solving real-world problems, optimization modeling has been widely employed for optimal decision-making across various sectors, but it requires substantial expertise from operations research professionals. With the advent of large language models (LLMs), new opportunities have emerged to automate the procedure of mathematical modeling. This survey presents a comprehensive and timely review of recent advancements that cover the entire technical stack, including data synthesis and fine-tuning for the base model, inference frameworks, benchmark datasets, and performance evaluation. In addition, we conducted an in-depth analysis on the quality of benchmark datasets, which was found to have a surprisingly high error rate. We cleaned the datasets and constructed a new leaderboard with fair performance evaluation in terms of base LLM model and datasets. We also build an online portal that integrates resources of cleaned datasets, code and paper repository to benefit the community. Finally, we identify limitations in current methodologies and outline future research opportunities. 

**Abstract (ZH)**: 基于大型语言模型的新机遇：优化建模自动化技术综述 

---
# Empirical Investigation into Configuring Echo State Networks for Representative Benchmark Problem Domains 

**Title (ZH)**: 关于配置回声状态网络以解决代表性基准问题领域的一种实证研究 

**Authors**: Brooke R. Weborg, Gursel Serpen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10887)  

**Abstract**: This paper examines Echo State Network, a reservoir computer, performance using four different benchmark problems, then proposes heuristics or rules of thumb for configuring the architecture, as well as the selection of parameters and their values, which are applicable to problems within the same domain, to help serve to fill the experience gap needed by those entering this field of study. The influence of various parameter selections and their value adjustments, as well as architectural changes made to an Echo State Network, a powerful recurrent neural network configured as a reservoir computer, can be challenging to fully comprehend without experience in the field, and even some hyperparameter optimization algorithms may have difficulty adjusting parameter values without proper manual selections made first. Therefore, it is imperative to understand the effects of parameters and their value selection on Echo State Network architecture performance for a successful build. Thus, to address the requirement for an extensive background in Echo State Network architecture, as well as examine how Echo State Network performance is affected with respect to variations in architecture, design, and parameter selection and values, a series of benchmark tasks representing different problem domains, including time series prediction, pattern generation, chaotic system prediction, and time series classification, were modeled and experimented on to show the impact on the performance of Echo State Network. 

**Abstract (ZH)**: 本文使用四个不同的基准问题考查Echo State Network（回声状态网络）的表现，并提出适用于相同领域问题的架构配置、参数选择及其值的启发式规则或经验法则，以帮助填补研究领域新手所需的经验差距。探讨架构变化、参数选择及其值调整对Echo State Network（一种配置为蓄水库的强循环神经网络）性能的影响可能会由于缺乏领域经验而显得复杂，即使一些超参数优化算法也可能难以在没有适当手动选择的情况下调整参数值。因此，了解参数及其值选择对Echo State Network架构性能的影响对于成功构建该模型至关重要。为了应对Echo State Network架构所需广泛背景知识的需求，并考察架构、设计、参数选择及其值变化对Echo State Network性能的影响，本文通过建模和实验一系列代表不同问题域的基准任务，包括时间序列预测、模式生成、混沌系统预测和时间序列分类，展示了Echo State Network性能的变化影响。 

---
# ToonComposer: Streamlining Cartoon Production with Generative Post-Keyframing 

**Title (ZH)**: ToonComposer: 生成式后关键帧优化的动画片生产简化方法 

**Authors**: Lingen Li, Guangzhi Wang, Zhaoyang Zhang, Yaowei Li, Xiaoyu Li, Qi Dou, Jinwei Gu, Tianfan Xue, Ying Shan  

**Link**: [PDF](https://arxiv.org/pdf/2508.10881)  

**Abstract**: Traditional cartoon and anime production involves keyframing, inbetweening, and colorization stages, which require intensive manual effort. Despite recent advances in AI, existing methods often handle these stages separately, leading to error accumulation and artifacts. For instance, inbetweening approaches struggle with large motions, while colorization methods require dense per-frame sketches. To address this, we introduce ToonComposer, a generative model that unifies inbetweening and colorization into a single post-keyframing stage. ToonComposer employs a sparse sketch injection mechanism to provide precise control using keyframe sketches. Additionally, it uses a cartoon adaptation method with the spatial low-rank adapter to tailor a modern video foundation model to the cartoon domain while keeping its temporal prior intact. Requiring as few as a single sketch and a colored reference frame, ToonComposer excels with sparse inputs, while also supporting multiple sketches at any temporal location for more precise motion control. This dual capability reduces manual workload and improves flexibility, empowering artists in real-world scenarios. To evaluate our model, we further created PKBench, a benchmark featuring human-drawn sketches that simulate real-world use cases. Our evaluation demonstrates that ToonComposer outperforms existing methods in visual quality, motion consistency, and production efficiency, offering a superior and more flexible solution for AI-assisted cartoon production. 

**Abstract (ZH)**: 传统的动画片和动漫生产涉及关键帧设定、中间帧生成和着色阶段，这些阶段需要大量的手动工作。尽管最近的人工智能取得了进展，现有的方法通常分别处理这些阶段，导致错误累积和伪影。例如，中间帧生成方法难以处理大动作，而着色方法需要每帧密集的素描。为了解决这个问题，我们引入了ToonComposer，这是一个生成模型，将中间帧生成和着色统一到关键帧设定后的单个阶段。ToonComposer采用稀疏素描注入机制，通过关键帧素描提供精确控制。此外，它使用空间低秩适配器来调整现代视频基础模型以适应动漫领域，同时保持其时间先验不变。只需单个素描和着色参考帧，ToonComposer即可高效处理稀疏输入，并支持任何时间位置的多个素描以实现更精细的动作控制。这种双重能力减少了手动工作量并提高了灵活性，使艺术家在实际场景中受益。为了评估我们的模型，我们进一步创建了PKBench，这是一个包含人类绘制素描的基准，这些素描模拟了真实世界的应用场景。我们的评估表明，ToonComposer在视觉质量、动作一致性以及生产效率方面优于现有方法，提供了一种更优秀且更灵活的辅助动漫生产的解决方案。 

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
# TLE-Based A2C Agent for Terrestrial Coverage Orbital Path Planning 

**Title (ZH)**: 基于TLE的A2C代理用于地表覆盖轨道路径规划 

**Authors**: Anantha Narayanan, Battu Bhanu Teja, Pruthwik Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2508.10872)  

**Abstract**: The increasing congestion of Low Earth Orbit (LEO) poses persistent challenges to the efficient deployment and safe operation of Earth observation satellites. Mission planners must now account not only for mission-specific requirements but also for the increasing collision risk with active satellites and space debris. This work presents a reinforcement learning framework using the Advantage Actor-Critic (A2C) algorithm to optimize satellite orbital parameters for precise terrestrial coverage within predefined surface radii. By formulating the problem as a Markov Decision Process (MDP) within a custom OpenAI Gymnasium environment, our method simulates orbital dynamics using classical Keplerian elements. The agent progressively learns to adjust five of the orbital parameters - semi-major axis, eccentricity, inclination, right ascension of ascending node, and the argument of perigee-to achieve targeted terrestrial coverage. Comparative evaluation against Proximal Policy Optimization (PPO) demonstrates A2C's superior performance, achieving 5.8x higher cumulative rewards (10.0 vs 9.263025) while converging in 31.5x fewer timesteps (2,000 vs 63,000). The A2C agent consistently meets mission objectives across diverse target coordinates while maintaining computational efficiency suitable for real-time mission planning applications. Key contributions include: (1) a TLE-based orbital simulation environment incorporating physics constraints, (2) validation of actor-critic methods' superiority over trust region approaches in continuous orbital control, and (3) demonstration of rapid convergence enabling adaptive satellite deployment. This approach establishes reinforcement learning as a computationally efficient alternative for scalable and intelligent LEO mission planning. 

**Abstract (ZH)**: 低地球轨道日益严重的拥堵对地球观测卫星的有效部署和安全运行构成了持续挑战。任务规划人员必须不仅考虑任务特定需求，还要考虑与活跃卫星和空间碎片不断增加的碰撞风险。本文提出了一种基于优势actor-评论员（A2C）算法的强化学习框架，以优化卫星轨道参数，实现预定义地面半径内的精确陆地覆盖。通过在一个自定义的OpenAI Gymnasium环境中将问题形式化为马尔可夫决策过程（MDP），我们的方法使用经典开普勒元素模拟轨道动力学。代理逐进学会调整轨道参数——半长轴、偏心率、倾角、升交点赤经和近地点幅角——以实现目标陆地覆盖。相比近端策略优化（PPO）方法，A2C表现出更优性能，累计奖励高5.8倍（10.0 vs 9.263025），收敛所需时间步减少31.5倍（2,000 vs 63,000）。A2C代理能够在多种目标坐标下一致达到任务目标，同时保持适用于实时任务规划应用的计算效率。主要贡献包括：（1）基于TLE的轨道仿真环境，包含物理约束；（2）验证了在连续轨道控制中演员-评论员方法优于信赖域方法的优越性；（3）展示了快速收敛能力，使卫星部署更具适应性。该方法确立了强化学习作为计算高效、可扩展和智能低地球轨道任务规划的替代方案的地位。 

---
# Medico 2025: Visual Question Answering for Gastrointestinal Imaging 

**Title (ZH)**: 医como 2025：消化道成像的视觉问答 

**Authors**: Sushant Gautam, Vajira Thambawita, Michael Riegler, Pål Halvorsen, Steven Hicks  

**Link**: [PDF](https://arxiv.org/pdf/2508.10869)  

**Abstract**: The Medico 2025 challenge addresses Visual Question Answering (VQA) for Gastrointestinal (GI) imaging, organized as part of the MediaEval task series. The challenge focuses on developing Explainable Artificial Intelligence (XAI) models that answer clinically relevant questions based on GI endoscopy images while providing interpretable justifications aligned with medical reasoning. It introduces two subtasks: (1) answering diverse types of visual questions using the Kvasir-VQA-x1 dataset, and (2) generating multimodal explanations to support clinical decision-making. The Kvasir-VQA-x1 dataset, created from 6,500 images and 159,549 complex question-answer (QA) pairs, serves as the benchmark for the challenge. By combining quantitative performance metrics and expert-reviewed explainability assessments, this task aims to advance trustworthy Artificial Intelligence (AI) in medical image analysis. Instructions, data access, and an updated guide for participation are available in the official competition repository: this https URL 

**Abstract (ZH)**: 2025医医挑战关注消化道成像的视觉问答（VQA），并通过MediaEval任务系列组织。该挑战旨在开发可解释的人工智能（XAI）模型，基于消化道内镜图像回答临床相关问题，并提供与医学推理相一致的可解释理由。该挑战引入了两个子任务：（1）使用Kvasir-VQA-x1数据集回答各种视觉问题，（2）生成支持临床决策的多模态解释。Kvasir-VQA-x1数据集包含6,500张图像和159,549个复杂问题-答案（QA）对，作为挑战的基准。通过结合定量性能指标和专家评审的可解释性评估，该任务旨在推动医疗图像分析中可信的人工智能（AI）的发展。官方竞赛 repository 提供了指南、数据访问和更新的参赛指南：this https URL。 

---
# Performance of GPT-5 in Brain Tumor MRI Reasoning 

**Title (ZH)**: GPT-5在脑肿瘤MRI推理中的表现 

**Authors**: Mojtaba Safari, Shansong Wang, Mingzhe Hu, Zach Eidex, Qiang Li, Xiaofeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10865)  

**Abstract**: Accurate differentiation of brain tumor types on magnetic resonance imaging (MRI) is critical for guiding treatment planning in neuro-oncology. Recent advances in large language models (LLMs) have enabled visual question answering (VQA) approaches that integrate image interpretation with natural language reasoning. In this study, we evaluated GPT-4o, GPT-5-nano, GPT-5-mini, and GPT-5 on a curated brain tumor VQA benchmark derived from 3 Brain Tumor Segmentation (BraTS) datasets - glioblastoma (GLI), meningioma (MEN), and brain metastases (MET). Each case included multi-sequence MRI triplanar mosaics and structured clinical features transformed into standardized VQA items. Models were assessed in a zero-shot chain-of-thought setting for accuracy on both visual and reasoning tasks. Results showed that GPT-5-mini achieved the highest macro-average accuracy (44.19%), followed by GPT-5 (43.71%), GPT-4o (41.49%), and GPT-5-nano (35.85%). Performance varied by tumor subtype, with no single model dominating across all cohorts. These findings suggest that GPT-5 family models can achieve moderate accuracy in structured neuro-oncological VQA tasks, but not at a level acceptable for clinical use. 

**Abstract (ZH)**: 磁共振成像（MRI）中脑肿瘤类型准确区分对于神经肿瘤学治疗规划至关重要。大型语言模型（LLMs）的 recent 进展使图像解释与自然语言推理结合的视觉问答（VQA）方法得以实现。在本研究中，我们基于 3 个脑肿瘤分割（BraTS）数据集（胶质母细胞瘤(GLI)、脑膜瘤(MEN) 和 脑转移瘤(MET)）构建了一个定制的脑肿瘤 VQA 验证基准，并评估了 GPT-4o、GPT-5-nano、GPT-5-mini 和 GPT-5 在此基准上的表现。每个案例包含多序列 MRI 三维镶嵌图像和结构化临床特征转换为标准化 VQA 项目。模型在零样本链式思考设置下评估了视觉和推理任务的准确性。结果显示，GPT-5-mini 达到最高的宏平均准确率（44.19%），其次是 GPT-5（43.71%）、GPT-4o（41.49%）和 GPT-5-nano（35.85%）。不同肿瘤亚型的表现各异，并无单一模型在所有组别中表现最优。这些发现表明，GPT-5 家族模型在结构化神经肿瘤学 VQA 任务中可以实现中等准确率，但尚不足以应用于临床。 

---
# From Black Box to Transparency: Enhancing Automated Interpreting Assessment with Explainable AI in College Classrooms 

**Title (ZH)**: 从黑箱到透明：通过可解释AI提升大学课堂自动解释评估 

**Authors**: Zhaokun Jiang, Ziyin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10860)  

**Abstract**: Recent advancements in machine learning have spurred growing interests in automated interpreting quality assessment. Nevertheless, existing research suffers from insufficient examination of language use quality, unsatisfactory modeling effectiveness due to data scarcity and imbalance, and a lack of efforts to explain model predictions. To address these gaps, we propose a multi-dimensional modeling framework that integrates feature engineering, data augmentation, and explainable machine learning. This approach prioritizes explainability over ``black box'' predictions by utilizing only construct-relevant, transparent features and conducting Shapley Value (SHAP) analysis. Our results demonstrate strong predictive performance on a novel English-Chinese consecutive interpreting dataset, identifying BLEURT and CometKiwi scores to be the strongest predictive features for fidelity, pause-related features for fluency, and Chinese-specific phraseological diversity metrics for language use. Overall, by placing particular emphasis on explainability, we present a scalable, reliable, and transparent alternative to traditional human evaluation, facilitating the provision of detailed diagnostic feedback for learners and supporting self-regulated learning advantages not afforded by automated scores in isolation. 

**Abstract (ZH)**: Recent advancements in machine learning have spurred growing interests in automated interpreting quality assessment. Nevertheless, existing research suffers from insufficient examination of language use quality, unsatisfactory modeling effectiveness due to data scarcity and imbalance, and a lack of efforts to explain model predictions. To address these gaps, we propose a multi-dimensional modeling framework that integrates feature engineering, data augmentation, and explainable machine learning. This approach prioritizes explainability over "black box" predictions by utilizing only construct-relevant, transparent features and conducting Shapley Value (SHAP) analysis. Our results demonstrate strong predictive performance on a novel English-Chinese consecutive interpreting dataset, identifying BLEURT and CometKiwi scores to be the strongest predictive features for fidelity, pause-related features for fluency, and Chinese-specific phraseological diversity metrics for language use.

综合这些方面的考虑，我们提出了一种多维度建模框架，该框架结合了特征工程、数据增强和可解释的人工智能。该方法优先考虑可解释性而非“黑箱”预测，仅利用相关且透明的特征，并进行Shapley值（SHAP）分析。我们在一个新颖的英译汉连续口译数据集上展示了该方法的强大预测性能，识别出BLEURT和CometKiwi评分是忠实性最强的预测特征，停顿相关特征是流畅性最强的预测特征，而特定于汉语的习语多样性指标是语言使用质量最强的预测特征。通过特别强调可解释性，我们提出了一种可扩展、可靠且透明的人机评价替代方案，为学习者提供详细的诊断反馈，并支持自动评分无法单独提供的自我调节学习优势。 

---
# Reinforced Language Models for Sequential Decision Making 

**Title (ZH)**: 强化语言模型在 sequential 决策中的应用 

**Authors**: Jim Dilkes, Vahid Yazdanpanah, Sebastian Stein  

**Link**: [PDF](https://arxiv.org/pdf/2508.10839)  

**Abstract**: Large Language Models (LLMs) show potential as sequential decision-making agents, but their application is often limited due to a reliance on large, computationally expensive models. This creates a need to improve smaller models, yet existing post-training methods are designed for single-turn interactions and cannot handle credit assignment in multi-step agentic tasks. To address this, we introduce Multi-Step Group-Relative Policy Optimization (MS-GRPO), a new algorithm for post-training LLM agents, grounded in formal Text-Mediated Stochastic Game (TSMG) and Language-Agent Policy (LAP) frameworks. For credit assignment, MS-GRPO attributes the entire cumulative episode reward to each individual episode step. We supplement this algorithm with a novel absolute-advantage-weighted episode sampling strategy that we show improves training performance. We evaluate our approach by post-training a 3-billion parameter model on Snake and Frozen Lake. Our experiments demonstrate that the method is effective in improving decision-making performance: our post-trained 3B parameter model outperforms a 72B parameter baseline by 50% on the Frozen Lake task. This work demonstrates that targeted post-training is a practical and efficient alternative to relying on model scale for creating sequential decision-making agents using LLMs. 

**Abstract (ZH)**: 多步组相关策略优化：基于形式化文本中介随机游戏与语言代理策略框架的后训练算法 

---
# A Multimodal Neural Network for Recognizing Subjective Self-Disclosure Towards Social Robots 

**Title (ZH)**: 面向社会机器人的主观自我披露识别的多模态神经网络 

**Authors**: Henry Powell, Guy Laban, Emily S. Cross  

**Link**: [PDF](https://arxiv.org/pdf/2508.10828)  

**Abstract**: Subjective self-disclosure is an important feature of human social interaction. While much has been done in the social and behavioural literature to characterise the features and consequences of subjective self-disclosure, little work has been done thus far to develop computational systems that are able to accurately model it. Even less work has been done that attempts to model specifically how human interactants self-disclose with robotic partners. It is becoming more pressing as we require social robots to work in conjunction with and establish relationships with humans in various social settings. In this paper, our aim is to develop a custom multimodal attention network based on models from the emotion recognition literature, training this model on a large self-collected self-disclosure video corpus, and constructing a new loss function, the scale preserving cross entropy loss, that improves upon both classification and regression versions of this problem. Our results show that the best performing model, trained with our novel loss function, achieves an F1 score of 0.83, an improvement of 0.48 from the best baseline model. This result makes significant headway in the aim of allowing social robots to pick up on an interaction partner's self-disclosures, an ability that will be essential in social robots with social cognition. 

**Abstract (ZH)**: 主观自我披露是人类社会互动的重要特征。尽管在社会行为学文献中已经做了大量关于主观自我披露的特征和后果的研究，但目前开发能够准确建模其行为的计算系统的工作尚显不足。在试图建模人类如何与机器人伙伴进行自我披露方面的工作更是少之又少。随着我们要求社会机器人能够在各种社会设置中与人类协同工作，并建立关系，这种情况变得愈发紧迫。本文旨在基于情绪识别模型开发一个定制的多模态注意力网络，通过一个大型自收集自我披露视频语料库进行训练，并构建一种新的损失函数——尺度保持交叉熵损失，以改进分类和回归版本的问题。我们的结果表明，使用我们新提出的损失函数训练的最佳模型，实现了0.83的F1分数，相较于最佳基线模型提高了0.48。这一结果在让社会机器人能够察觉互动伙伴的自我披露方面取得了显著进展，而这种能力对具有社会认知的社会机器人来说至关重要。 

---
# The SET Perceptual Factors Framework: Towards Assured Perception for Autonomous Systems 

**Title (ZH)**: SET感知因素框架：迈向自主系统可信赖感知的研究 

**Authors**: Troi Williams  

**Link**: [PDF](https://arxiv.org/pdf/2508.10798)  

**Abstract**: Future autonomous systems promise significant societal benefits, yet their deployment raises concerns about safety and trustworthiness. A key concern is assuring the reliability of robot perception, as perception seeds safe decision-making. Failures in perception are often due to complex yet common environmental factors and can lead to accidents that erode public trust. To address this concern, we introduce the SET (Self, Environment, and Target) Perceptual Factors Framework. We designed the framework to systematically analyze how factors such as weather, occlusion, or sensor limitations negatively impact perception. To achieve this, the framework employs SET State Trees to categorize where such factors originate and SET Factor Trees to model how these sources and factors impact perceptual tasks like object detection or pose estimation. Next, we develop Perceptual Factor Models using both trees to quantify the uncertainty for a given task. Our framework aims to promote rigorous safety assurances and cultivate greater public understanding and trust in autonomous systems by offering a transparent and standardized method for identifying, modeling, and communicating perceptual risks. 

**Abstract (ZH)**: 未来自主系统有望为社会带来巨大益处，但部署引发了关于安全与可信度的关注。一个关键的关注点是如何保证机器人的感知可靠性 以及如何确保感知的安全决策。感知中的失败往往由于复杂且常见的环境因素所致 并且会导致因信任度下降而引发的事故。为应对这一关切 我们介绍了SET（自我·环境·验证）感知因素框架。该框架旨在系统性分析感知问题 如如遮挡等、低识别率等等和其他限制因素如何负面影响感知。为了分析此类问题 该框架使用了SET STEM树来来分类导致感知问题的根本原因 并使用SET因素树来评估感知任务的性能和这些因素如何影响感知任务 如如目标检测和 �推荐阅读点 � עם估计。该框架使用了SET STM模型 来衡量一种感知任务中的不确定性。我们的目标是提供严格感知保障 以通过提供透明且标准化的方法来模（型请感知风险和沟通感知不确定性来来促进对和和信任。 

---
# Enhancing Fairness in Autoencoders for Node-Level Graph Anomaly Detection 

**Title (ZH)**: 增强节点级别图异常检测自动编码器的公平性 

**Authors**: Shouju Wang, Yuchen Song, Sheng'en Li, Dongmian Zou  

**Link**: [PDF](https://arxiv.org/pdf/2508.10785)  

**Abstract**: Graph anomaly detection (GAD) has become an increasingly important task across various domains. With the rapid development of graph neural networks (GNNs), GAD methods have achieved significant performance improvements. However, fairness considerations in GAD remain largely underexplored. Indeed, GNN-based GAD models can inherit and amplify biases present in training data, potentially leading to unfair outcomes. While existing efforts have focused on developing fair GNNs, most approaches target node classification tasks, where models often rely on simple layer architectures rather than autoencoder-based structures, which are the most widely used architecturs for anomaly detection. To address fairness in autoencoder-based GAD models, we propose \textbf{D}is\textbf{E}ntangled \textbf{C}ounterfactual \textbf{A}dversarial \textbf{F}air (DECAF)-GAD, a framework that alleviates bias while preserving GAD performance. Specifically, we introduce a structural causal model (SCM) to disentangle sensitive attributes from learned representations. Based on this causal framework, we formulate a specialized autoencoder architecture along with a fairness-guided loss function. Through extensive experiments on both synthetic and real-world datasets, we demonstrate that DECAF-GAD not only achieves competitive anomaly detection performance but also significantly enhances fairness metrics compared to baseline GAD methods. Our code is available at this https URL. 

**Abstract (ZH)**: 基于解纠缠反事实对抗的图异常检测（DECAF-GAD） 

---
# Ultra-High-Definition Reference-Based Landmark Image Super-Resolution with Generative Diffusion Prior 

**Title (ZH)**: 基于生成性扩散先验的 Ultra-HD 参考导向地标图像超分辨率 

**Authors**: Zhenning Shi, Zizheng Yan, Yuhang Yu, Clara Xue, Jingyu Zhuang, Qi Zhang, Jinwei Chen, Tao Li, Qingnan Fan  

**Link**: [PDF](https://arxiv.org/pdf/2508.10779)  

**Abstract**: Reference-based Image Super-Resolution (RefSR) aims to restore a low-resolution (LR) image by utilizing the semantic and texture information from an additional reference high-resolution (reference HR) image. Existing diffusion-based RefSR methods are typically built upon ControlNet, which struggles to effectively align the information between the LR image and the reference HR image. Moreover, current RefSR datasets suffer from limited resolution and poor image quality, resulting in the reference images lacking sufficient fine-grained details to support high-quality restoration. To overcome the limitations above, we propose TriFlowSR, a novel framework that explicitly achieves pattern matching between the LR image and the reference HR image. Meanwhile, we introduce Landmark-4K, the first RefSR dataset for Ultra-High-Definition (UHD) landmark scenarios. Considering the UHD scenarios with real-world degradation, in TriFlowSR, we design a Reference Matching Strategy to effectively match the LR image with the reference HR image. Experimental results show that our approach can better utilize the semantic and texture information of the reference HR image compared to previous methods. To the best of our knowledge, we propose the first diffusion-based RefSR pipeline for ultra-high definition landmark scenarios under real-world degradation. Our code and model will be available at this https URL. 

**Abstract (ZH)**: 基于参考的图像超分辨率 (RefSR) 致力于通过利用附加参考高分辨率 (参考HR) 图像的语义和纹理信息来恢复低分辨率 (LR) 图像。现有的基于扩散的 RefSR 方法通常基于 ControlNet 构建，难以有效对齐 LR 图像和参考 HR 图像之间的信息。此外，当前的 RefSR 数据集分辨率有限且图像质量差，导致参考图像缺乏足够的细粒度细节以支持高质量的恢复。为克服上述限制，我们提出了 TriFlowSR，一个新框架，明确实现 LR 图像与参考 HR 图像之间的模式匹配。同时，我们引入了 Landmark-4K，首个适用于超高清 (UHD) 标志场景的 RefSR 数据集。考虑到具有真实世界降级的 UHD 场景，我们在 TriFlowSR 中设计了一种参考匹配策略，以有效匹配 LR 图像与参考 HR 图像。实验结果显示，与以往方法相比，我们的方法能够更好地利用参考 HR 图像的语义和纹理信息。据我们所知，我们提出了首个适用于具有真实世界降级的超高清标志场景的基于扩散的 RefSR 管线。我们的代码和模型将发布在上述网址。 

---
# Estimating Covariance for Global Minimum Variance Portfolio: A Decision-Focused Learning Approach 

**Title (ZH)**: 基于决策导向的学习方法估计全局最小方差组合的协方差,eg：基于决策导向的学习方法估计全局最小方差Portfolio的协方程西侧 

**Authors**: Juchan Kim, Inwoo Tae, Yongjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.10776)  

**Abstract**: Portfolio optimization constitutes a cornerstone of risk management by quantifying the risk-return trade-off. Since it inherently depends on accurate parameter estimation under conditions of future uncertainty, the selection of appropriate input parameters is critical for effective portfolio construction. However, most conventional statistical estimators and machine learning algorithms determine these parameters by minimizing mean-squared error (MSE), a criterion that can yield suboptimal investment decisions. In this paper, we adopt decision-focused learning (DFL) - an approach that directly optimizes decision quality rather than prediction error such as MSE - to derive the global minimum-variance portfolio (GMVP). Specifically, we theoretically derive the gradient of decision loss using the analytic solution of GMVP and its properties regarding the principal components of itself. Through extensive empirical evaluation, we show that prediction-focused estimation methods may fail to produce optimal allocations in practice, whereas DFL-based methods consistently deliver superior decision performance. Furthermore, we provide a comprehensive analysis of DFL's mechanism in GMVP construction, focusing on its volatility reduction capability, decision-driving features, and estimation characteristics. 

**Abstract (ZH)**: 投资组合优化构成了风险管理的基石，通过量化收益与风险的权衡。由于其本身依赖于未来不确定性条件下准确的参数估计，因此选择合适的输入参数对于有效的投资组合构建至关重要。然而，大多数传统的统计估计器和机器学习算法通过最小化均方误差（MSE）来确定这些参数，这种方法可能产生次优的投资决策。在本文中，我们采用决策聚焦学习（DFL）——一种直接优化决策质量而非预测误差（如MSE）的方法——来推导全局最小方差投资组合（GMVP）。具体地，我们利用GMVP的解析解及其自身主成分的性质，理论上推导出决策损失的梯度。通过广泛的实证评估，我们展示了预测驱动的估计方法可能在实践中无法产生最优分配，而基于DFL的方法则始终能提供更优越的决策性能。此外，我们对DFL在构建GMVP过程中的机制进行了全面分析，着重讨论了其波动性降低能力、决策驱动特征和估计特性。 

---
# Video-BLADE: Block-Sparse Attention Meets Step Distillation for Efficient Video Generation 

**Title (ZH)**: 视频-BLADE: 块稀疏注意力与步进distillation相结合的高效视频生成 

**Authors**: Youping Gu, Xiaolong Li, Yuhao Hu, Bohan Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10774)  

**Abstract**: Diffusion transformers currently lead the field in high-quality video generation, but their slow iterative denoising process and prohibitive quadratic attention costs for long sequences create significant inference bottlenecks. While both step distillation and sparse attention mechanisms have shown promise as independent acceleration strategies, effectively combining these approaches presents critical challenges -- training-free integration yields suboptimal results, while separately training sparse attention after step distillation requires prohibitively expensive high-quality video data. To overcome these limitations, we propose BLADE, an innovative data-free joint training framework that introduces: (1) an Adaptive Block-Sparse Attention (ASA) mechanism for dynamically generating content-aware sparsity masks to focus computation on salient spatiotemporal features, and (2) a sparsity-aware step distillation paradigm built upon Trajectory Distribution Matching (TDM) that directly incorporates sparsity into the distillation process rather than treating it as a separate compression step, with fast convergence. We validate BLADE on text-to-video models like CogVideoX-5B and Wan2.1-1.3B. Our framework demonstrates remarkable efficiency gains across different scales. On Wan2.1-1.3B, BLADE achieves a 14.10x end-to-end inference acceleration over a 50-step baseline. Moreover, on models such as CogVideoX-5B with short video sequence lengths, our framework delivers a robust 8.89x speedup. Crucially, the acceleration is accompanied by a consistent quality improvement. On the VBench-2.0 benchmark, BLADE boosts the score of CogVideoX-5B to 0.569 (from 0.534) and Wan2.1-1.3B to 0.570 (from 0.563), results that are further corroborated by superior ratings in human evaluations. Our code and model weights are publicly available at: this http URL. 

**Abstract (ZH)**: 基于BLADE的数据驱动联合训练框架在高效率视频生成中的应用 

---
# AEGIS: Authenticity Evaluation Benchmark for AI-Generated Video Sequences 

**Title (ZH)**: AEGIS: AI生成视频序列真实性评估基准 

**Authors**: Jieyu Li, Xin Zhang, Joey Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.10771)  

**Abstract**: Recent advances in AI-generated content have fueled the rise of highly realistic synthetic videos, posing severe risks to societal trust and digital integrity. Existing benchmarks for video authenticity detection typically suffer from limited realism, insufficient scale, and inadequate complexity, failing to effectively evaluate modern vision-language models against sophisticated forgeries. To address this critical gap, we introduce AEGIS, a novel large-scale benchmark explicitly targeting the detection of hyper-realistic and semantically nuanced AI-generated videos. AEGIS comprises over 10,000 rigorously curated real and synthetic videos generated by diverse, state-of-the-art generative models, including Stable Video Diffusion, CogVideoX-5B, KLing, and Sora, encompassing open-source and proprietary architectures. In particular, AEGIS features specially constructed challenging subsets enhanced with robustness evaluation. Furthermore, we provide multimodal annotations spanning Semantic-Authenticity Descriptions, Motion Features, and Low-level Visual Features, facilitating authenticity detection and supporting downstream tasks such as multimodal fusion and forgery localization. Extensive experiments using advanced vision-language models demonstrate limited detection capabilities on the most challenging subsets of AEGIS, highlighting the dataset's unique complexity and realism beyond the current generalization capabilities of existing models. In essence, AEGIS establishes an indispensable evaluation benchmark, fundamentally advancing research toward developing genuinely robust, reliable, broadly generalizable video authenticity detection methodologies capable of addressing real-world forgery threats. Our dataset is available on this https URL. 

**Abstract (ZH)**: Recent Advances in AI-Generated Content Have Fuelled the Rise of Highly Realistic Synthetic Videos, Posing Severe Risks to Societal Trust and Digital Integrity: Introducing AEGIS, a Novel Large-Scale Benchmark for Detecting Hyper-Realistic and Semantically Nuanced AI-Generated Videos 

---
# FROGENT: An End-to-End Full-process Drug Design Agent 

**Title (ZH)**: FROGENT：一个全流程药物设计代理 

**Authors**: Qihua Pan, Dong Xu, Jenna Xinyi Yao, Lijia Ma, Zexuan Zhu, Junkai Ji  

**Link**: [PDF](https://arxiv.org/pdf/2508.10760)  

**Abstract**: Powerful AI tools for drug discovery reside in isolated web apps, desktop programs, and code libraries. Such fragmentation forces scientists to manage incompatible interfaces and specialized scripts, which can be a cumbersome and repetitive process. To address this issue, a Full-pROcess druG dEsign ageNT, named FROGENT, has been proposed. Specifically, FROGENT utilizes a Large Language Model and the Model Context Protocol to integrate multiple dynamic biochemical databases, extensible tool libraries, and task-specific AI models. This agentic framework allows FROGENT to execute complicated drug discovery workflows dynamically, including component tasks such as target identification, molecule generation and retrosynthetic planning. FROGENT has been evaluated on eight benchmarks that cover various aspects of drug discovery, such as knowledge retrieval, property prediction, virtual screening, mechanistic analysis, molecular design, and synthesis. It was compared against six increasingly advanced ReAct-style agents that support code execution and literature searches. Empirical results demonstrated that FROGENT triples the best baseline performance in hit-finding and doubles it in interaction profiling, significantly outperforming both the open-source model Qwen3-32B and the commercial model GPT-4o. In addition, real-world cases have been utilized to validate the practicability and generalization of FROGENT. This development suggests that streamlining the agentic drug discovery pipeline can significantly enhance researcher productivity. 

**Abstract (ZH)**: 基于全生命周期药物设计的强大AI代理工具：FROGENT 

---
# Natively Trainable Sparse Attention for Hierarchical Point Cloud Datasets 

**Title (ZH)**: 本源可训练的稀疏注意力机制 for 分层点云数据集 

**Authors**: Nicolas Lapautre, Maria Marchenko, Carlos Miguel Patiño, Xin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.10758)  

**Abstract**: Unlocking the potential of transformers on datasets of large physical systems depends on overcoming the quadratic scaling of the attention mechanism. This work explores combining the Erwin architecture with the Native Sparse Attention (NSA) mechanism to improve the efficiency and receptive field of transformer models for large-scale physical systems, addressing the challenge of quadratic attention complexity. We adapt the NSA mechanism for non-sequential data, implement the Erwin NSA model, and evaluate it on three datasets from the physical sciences -- cosmology simulations, molecular dynamics, and air pressure modeling -- achieving performance that matches or exceeds that of the original Erwin model. Additionally, we reproduce the experimental results from the Erwin paper to validate their implementation. 

**Abstract (ZH)**: 大规模物理系统数据集上Transformer潜力的释放取决于克服注意力机制的 quadr 增长。本工作探索将 Erwin 架构与原生稀疏注意力（NSA）机制相结合，以提高 Transformer 模型对大规模物理系统效率和感受野，解决注意力机制的 quadr 增长挑战。我们为非序列数据适配 NSA 机制，实现 Erwin NSA 模型，并在来自物理科学的三个数据集——宇宙学模拟、分子动力学和气压建模——上进行评估，实现性能与原版 Erwin 模型相当或超越。此外，我们重现 Erwin 研究论文中的实验结果以验证其实现。 

---
# Pass@k Training for Adaptively Balancing Exploration and Exploitation of Large Reasoning Models 

**Title (ZH)**: 基于Pass@k的训练方法以适应性ively 平衡大规模推理模型的探索与利用。 

**Authors**: Zhipeng Chen, Xiaobo Qin, Youbin Wu, Yue Ling, Qinghao Ye, Wayne Xin Zhao, Guang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.10751)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR), which typically adopts Pass@1 as the reward, has faced the issues in balancing exploration and exploitation, causing policies to prefer conservative actions, converging to a local optimum. Identifying an appropriate reward metric is therefore crucial. Regarding the prior work, although Pass@k has been used in evaluation, its connection to LLM exploration ability in RLVR remains largely overlooked. To investigate this, we first use Pass@k as the reward to train the policy model (i.e., $\textbf{Pass@k Training}$), and observe the improvement on its exploration ability. Next, we derive an analytical solution for the advantage of Pass@k Training, leading to an efficient and effective process. Building on this, our analysis reveals that exploration and exploitation are not inherently conflicting objectives, while they can mutually enhance each other. Moreover, Pass@k Training with analytical derivation essentially involves directly designing the advantage function. Inspired by this, we preliminarily explore the advantage design for RLVR, showing promising results and highlighting a potential future direction. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）及其Pass@1奖励的存在探索和利用平衡问题，导致策略倾向于保守行为，收敛于局部最优。因此，选择合适的奖励指标至关重要。先前研究虽然使用了Pass@k进行评估，但其与RLVR中的LLM探索能力之间的联系仍被忽视。为研究这一问题，我们首先使用Pass@k作为奖励训练策略模型（即Pass@k训练），观察其探索能力的提升。接着，我们推导了Pass@k训练的优势解，从而得到了一个高效且有效的过程。基于此，我们的分析表明，探索和利用并不是固有的冲突目标，而可以相互增强。此外，带分析推导的Pass@k训练本质上涉及直接设计优势函数。受此启发，我们初步探索了RLVR的优势设计，显示出有希望的结果，并强调了未来潜在的研究方向。 

---
# APFL: Analytic Personalized Federated Learning via Dual-Stream Least Squares 

**Title (ZH)**: APFL: 分析个性化联邦学习通过双流最小二乘法 

**Authors**: Kejia Fan, Jianheng Tang, Zhirui Yang, Feijiang Han, Jiaxu Li, Run He, Yajiang Huang, Anfeng Liu, Houbing Herbert Song, Yunhuai Liu, Huiping Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10732)  

**Abstract**: Personalized Federated Learning (PFL) has presented a significant challenge to deliver personalized models to individual clients through collaborative training. Existing PFL methods are often vulnerable to non-IID data, which severely hinders collective generalization and then compromises the subsequent personalization efforts. In this paper, to address this non-IID issue in PFL, we propose an Analytic Personalized Federated Learning (APFL) approach via dual-stream least squares. In our APFL, we use a foundation model as a frozen backbone for feature extraction. Subsequent to the feature extractor, we develop dual-stream analytic models to achieve both collective generalization and individual personalization. Specifically, our APFL incorporates a shared primary stream for global generalization across all clients, and a dedicated refinement stream for local personalization of each individual client. The analytical solutions of our APFL enable its ideal property of heterogeneity invariance, theoretically meaning that each personalized model remains identical regardless of how heterogeneous the data are distributed across all other clients. Empirical results across various datasets also validate the superiority of our APFL over state-of-the-art baselines, with advantages of at least 1.10%-15.45% in accuracy. 

**Abstract (ZH)**: 基于双流最小二乘的分析个性化联邦学习（APFL） 

---
# EgoCross: Benchmarking Multimodal Large Language Models for Cross-Domain Egocentric Video Question Answering 

**Title (ZH)**: EgoCross：跨域主观视角多模态语言模型跨模态问答基准测试 

**Authors**: Yanjun Li, Yuqian Fu, Tianwen Qian, Qi'ao Xu, Silong Dai, Danda Pani Paudel, Luc Van Gool, Xiaoling Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10729)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have significantly pushed the frontier of egocentric video question answering (EgocentricQA). However, existing benchmarks and studies are mainly limited to common daily activities such as cooking and cleaning. In contrast, real-world deployment inevitably encounters domain shifts, where target domains differ substantially in both visual style and semantic content. To bridge this gap, we introduce \textbf{EgoCross}, a comprehensive benchmark designed to evaluate the cross-domain generalization of MLLMs in EgocentricQA. EgoCross covers four diverse and challenging domains, including surgery, industry, extreme sports, and animal perspective, representing realistic and high-impact application scenarios. It comprises approximately 1,000 QA pairs across 798 video clips, spanning four key QA tasks: prediction, recognition, localization, and counting. Each QA pair provides both OpenQA and CloseQA formats to support fine-grained evaluation. Extensive experiments show that most existing MLLMs, whether general-purpose or egocentric-specialized, struggle to generalize to domains beyond daily life, highlighting the limitations of current models. Furthermore, we conduct several pilot studies, \eg, fine-tuning and reinforcement learning, to explore potential improvements. We hope EgoCross and our accompanying analysis will serve as a foundation for advancing domain-adaptive, robust egocentric video understanding. Data and codes will be released at: \href{this https URL}{this https URL.} 

**Abstract (ZH)**: 近期多模态大型语言模型（MLLMs）的进展显著推动了第一人称视频问答（EgocentricQA）的前沿。然而，现有的基准和研究主要集中在烹饪和清洁等日常活动中。相比之下，实际部署不可避免地会遇到领域偏移问题，目标领域在视觉风格和语义内容上存在显著差异。为了解决这一问题，我们引入了EgoCross，这是一个全面的基准，旨在评估MLLMs在EgocentricQA中的跨领域泛化能力。EgoCross涵盖了四个多样且具有挑战性的领域，包括手术、工业、极限运动和动物视角，代表了具有现实意义和高影响的应用场景。它包括约1,000组QA对，横跨798个视频片段，涵盖四个关键的QA任务：预测、识别、定位和计数。每组QA对都提供了开放式问答（OpenQA）和封闭式问答（CloseQA）格式，以支持精细评估。大量实验表明，大多数现有MLLMs，无论是通用的还是专门针对第一人称视角的，都难以泛化到日常生活之外的领域，突显了当前模型的局限性。此外，我们还进行了几项初步研究，例如微调和强化学习，以探索潜在的改进方法。我们希望EgoCross及其伴随的分析能够为推进适应性强的第一人称视频理解提供基础。数据和代码将在以下链接发布：\href{this https URL}{this https URL.} 

---
# Electromagnetic Simulations of Antennas on GPUs for Machine Learning Applications 

**Title (ZH)**: 基于GPU的天线电磁仿真及其在机器学习中的应用 

**Authors**: Murat Temiz, Vemund Bakken  

**Link**: [PDF](https://arxiv.org/pdf/2508.10713)  

**Abstract**: This study proposes an antenna simulation framework powered by graphics processing units (GPUs) based on an open-source electromagnetic (EM) simulation software (gprMax) for machine learning applications of antenna design and optimization. Furthermore, it compares the simulation results with those obtained through commercial EM software. The proposed software framework for machine learning and surrogate model applications will produce antenna data sets consisting of a large number of antenna simulation results using GPUs. Although machine learning methods can attain the optimum solutions for many problems, they are known to be data-hungry and require a great deal of samples for the training stage of the algorithms. However, producing a sufficient number of training samples in EM applications within a limited time is challenging due to the high computational complexity of EM simulations. Therefore, GPUs are utilized in this study to simulate a large number of antennas with predefined or random antenna shape parameters to produce data sets. Moreover, this study also compares various machine learning and deep learning models in terms of antenna parameter estimation performance. This study demonstrates that an entry-level GPU substantially outperforms a high-end CPU in terms of computational performance, while a high-end gaming GPU can achieve around 18 times more computational performance compared to a high-end CPU. Moreover, it is shown that the open-source EM simulation software can deliver similar results to those obtained via commercial software in the simulation of microstrip antennas when the spatial resolution of the simulations is sufficiently fine. 

**Abstract (ZH)**: 基于图形处理单元（GPU）的动力学处理单元（GPUGrant）开源电磁（EM）仿真软件的天线设计与优化的机器学习应用仿真框架研究 

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
# Continuous Bangla Sign Language Translation: Mitigating the Expense of Gloss Annotation with the Assistance of Graph 

**Title (ZH)**: 连续手势翻译：借助图辅助减轻高尔斯注释的开销 

**Authors**: Safaeid Hossain Arib, Rabeya Akter, Sejuti Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2508.10687)  

**Abstract**: Millions of individuals worldwide are affected by deafness and hearing impairment. Sign language serves as a sophisticated means of communication for the deaf and hard of hearing. However, in societies that prioritize spoken languages, sign language often faces underestimation, leading to communication barriers and social exclusion. The Continuous Bangla Sign Language Translation project aims to address this gap by enhancing translation methods. While recent approaches leverage transformer architecture for state-of-the-art results, our method integrates graph-based methods with the transformer architecture. This fusion, combining transformer and STGCN-LSTM architectures, proves more effective in gloss-free translation. Our contributions include architectural fusion, exploring various fusion strategies, and achieving a new state-of-the-art performance on diverse sign language datasets, namely RWTH-PHOENIX-2014T, CSL-Daily, How2Sign, and BornilDB v1.0. Our approach demonstrates superior performance compared to current translation outcomes across all datasets, showcasing notable improvements of BLEU-4 scores of 4.01, 2.07, and 0.5, surpassing those of GASLT, GASLT and slt_how2sign in RWTH-PHOENIX-2014T, CSL-Daily, and How2Sign, respectively. Also, we introduce benchmarking on the BornilDB v1.0 dataset for the first time. Our method sets a benchmark for future research, emphasizing the importance of gloss-free translation to improve communication accessibility for the deaf and hard of hearing. 

**Abstract (ZH)**: 全球有数百万人受到聋哑和听力障碍的影响。手语是聋人和听力障碍者进行沟通的一种复杂方式。然而，在重视口语的社会中，手语往往被低估，导致沟通障碍和社会排斥。连续孟加拉手语翻译项目旨在通过改进翻译方法来填补这一空白。尽管近年来的方法利用变换器架构取得了最先进的结果，我们的方法将基于图的方法与变换器架构结合起来。这种结合，即变换器与STGCN-LSTM架构的融合，在无手语点翻译中表现更佳。我们的贡献包括架构融合、探索各种融合策略，并在多种手语数据集（如RWTH-PHOENIX-2014T、CSL-Daily、How2Sign和BornilDB v1.0）上取得了新的最先进的性能。我们的方法在所有数据集上的性能优于当前的翻译结果，展示了BLEU-4得分分别为4.01、2.07和0.5的显著改进，超过了GASLT、GASLT和slt_how2sign在RWTH-PHOENIX-2014T、CSL-Daily和How2Sign上的结果。此外，我们首次在BornilDB v1.0数据集上进行了基准测试。我们的方法为未来的研究设定了基准，并强调了无手语点翻译的重要性，以提高聋人和听力障碍者的沟通可访问性。 

---
# Hybrid Generative Fusion for Efficient and Privacy-Preserving Face Recognition Dataset Generation 

**Title (ZH)**: 混合生成融合用于高效和隐私保护的面部识别数据集生成 

**Authors**: Feiran Li, Qianqian Xu, Shilong Bao, Boyu Han, Zhiyong Yang, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10672)  

**Abstract**: In this paper, we present our approach to the DataCV ICCV Challenge, which centers on building a high-quality face dataset to train a face recognition model. The constructed dataset must not contain identities overlapping with any existing public face datasets. To handle this challenge, we begin with a thorough cleaning of the baseline HSFace dataset, identifying and removing mislabeled or inconsistent identities through a Mixture-of-Experts (MoE) strategy combining face embedding clustering and GPT-4o-assisted verification. We retain the largest consistent identity cluster and apply data augmentation up to a fixed number of images per identity. To further diversify the dataset, we generate synthetic identities using Stable Diffusion with prompt engineering. As diffusion models are computationally intensive, we generate only one reference image per identity and efficiently expand it using Vec2Face, which rapidly produces 49 identity-consistent variants. This hybrid approach fuses GAN-based and diffusion-based samples, enabling efficient construction of a diverse and high-quality dataset. To address the high visual similarity among synthetic identities, we adopt a curriculum learning strategy by placing them early in the training schedule, allowing the model to progress from easier to harder samples. Our final dataset contains 50 images per identity, and all newly generated identities are checked with mainstream face datasets to ensure no identity leakage. Our method achieves \textbf{1st place} in the competition, and experimental results show that our dataset improves model performance across 10K, 20K, and 100K identity scales. Code is available at this https URL. 

**Abstract (ZH)**: 在本次论文中，我们提出了针对DataCV ICCV挑战赛的方法，重点在于构建一个高质量的人脸数据集以训练人脸识别模型。所构建的数据集必须不包含与任何现有公开人脸数据集重叠的身份信息。为应对这一挑战，我们从基线HSFace数据集的彻底清洁开始，通过结合人脸嵌入聚类和GPT-4o辅助验证的专家混合（MoE）策略，识别并移除错误标记或不一致的身份信息。保留最大的一致身份集群，并对每个身份进行数据扩增，直至固定数量的图像。为进一步增加数据集的多样性，我们使用Stable Diffusion生成合成身份，并结合提示工程。由于扩散模型计算密集，我们为每个身份生成一个参考图像，并使用Vec2Face高效扩展，快速生成49个身份一致的变体。这种混合方法结合了基于GAN和扩散模型的样本，能够高效构建一个多样化和高质量的数据集。为了解决合成身份之间高视觉相似性的问题，我们采用了课堂学习策略，在训练计划早期放置它们，使模型能够从较易到较难的样本逐步进步。最终数据集包含每个身份50张图像，所有新生成的身份都与主流人脸数据集进行了检查，确保没有身份泄露。我们的方法在比赛中获得第一名，并且实验结果显示，我们的数据集在10K、20K和100K身份规模下均提高了模型性能。代码可在以下链接获取。 

---
# AddressVLM: Cross-view Alignment Tuning for Image Address Localization using Large Vision-Language Models 

**Title (ZH)**: AddressVLM：使用大型视觉-语言模型进行图像地址本地化的跨视图对齐调整 

**Authors**: Shixiong Xu, Chenghao Zhang, Lubin Fan, Yuan Zhou, Bin Fan, Shiming Xiang, Gaofeng Meng, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2508.10667)  

**Abstract**: Large visual language models (LVLMs) have demonstrated impressive performance in coarse-grained geo-localization at the country or city level, but they struggle with fine-grained street-level localization within urban areas. In this paper, we explore integrating city-wide address localization capabilities into LVLMs, facilitating flexible address-related question answering using street-view images. A key challenge is that the street-view visual question-and-answer (VQA) data provides only microscopic visual cues, leading to subpar performance in fine-tuned models. To tackle this issue, we incorporate perspective-invariant satellite images as macro cues and propose cross-view alignment tuning including a satellite-view and street-view image grafting mechanism, along with an automatic label generation mechanism. Then LVLM's global understanding of street distribution is enhanced through cross-view matching. Our proposed model, named AddressVLM, consists of two-stage training protocols: cross-view alignment tuning and address localization tuning. Furthermore, we have constructed two street-view VQA datasets based on image address localization datasets from Pittsburgh and San Francisco. Qualitative and quantitative evaluations demonstrate that AddressVLM outperforms counterpart LVLMs by over 9% and 12% in average address localization accuracy on these two datasets, respectively. 

**Abstract (ZH)**: 面向街道级别的地址本地化大型视觉语言模型 

---
# Deep Learning in Classical and Quantum Physics 

**Title (ZH)**: 深度学习在经典物理与量子物理中的应用 

**Authors**: Timothy Heightman, Marcin Płodzień  

**Link**: [PDF](https://arxiv.org/pdf/2508.10666)  

**Abstract**: Scientific progress is tightly coupled to the emergence of new research tools. Today, machine learning (ML)-especially deep learning (DL)-has become a transformative instrument for quantum science and technology. Owing to the intrinsic complexity of quantum systems, DL enables efficient exploration of large parameter spaces, extraction of patterns from experimental data, and data-driven guidance for research directions. These capabilities already support tasks such as refining quantum control protocols and accelerating the discovery of materials with targeted quantum properties, making ML/DL literacy an essential skill for the next generation of quantum scientists. At the same time, DL's power brings risks: models can overfit noisy data, obscure causal structure, and yield results with limited physical interpretability. Recognizing these limitations and deploying mitigation strategies is crucial for scientific rigor. These lecture notes provide a comprehensive, graduate-level introduction to DL for quantum applications, combining conceptual exposition with hands-on examples. Organized as a progressive sequence, they aim to equip readers to decide when and how to apply DL effectively, to understand its practical constraints, and to adapt AI methods responsibly to problems across quantum physics, chemistry, and engineering. 

**Abstract (ZH)**: 科学进步与新研究工具的出现紧密相关。今天，机器学习（ML），尤其是深度学习（DL），已成为量子科学和技术的变革性工具。由于量子系统的固有复杂性，DL能够高效地探索大型参数空间、从实验数据中提取模式，并基于数据指导研究方向。这些能力已经支持诸如 refinement of quantum control protocols 和加速发现具有目标量子性质的材料等任务，使得ML/DL能力成为下一代量子科学家的必备技能。同时，DL的强大功能也带来了风险：模型可能会过度拟合噪声数据，掩盖因果结构，导致结果缺乏物理可解释性。意识这些局限性和部署缓解策略对于科学严谨性至关重要。这些讲义提供了量子应用领域的DL全面且适合研究生水平的介绍，结合了概念阐述与实际案例。它们旨在帮助读者决定何时以及如何有效应用DL，理解其实用限制，并负责任地将AI方法应用于量子物理、化学和工程领域的各种问题。 

---
# Serial Over Parallel: Learning Continual Unification for Multi-Modal Visual Object Tracking and Benchmarking 

**Title (ZH)**: 串行优于并行：学习多模态视觉目标跟踪中的持续统一方法及基准测试 

**Authors**: Zhangyong Tang, Tianyang Xu, Xuefeng Zhu, Chunyang Cheng, Tao Zhou, Xiaojun Wu, Josef Kittler  

**Link**: [PDF](https://arxiv.org/pdf/2508.10655)  

**Abstract**: Unifying multiple multi-modal visual object tracking (MMVOT) tasks draws increasing attention due to the complementary nature of different modalities in building robust tracking systems. Existing practices mix all data sensor types in a single training procedure, structuring a parallel paradigm from the data-centric perspective and aiming for a global optimum on the joint distribution of the involved tasks. However, the absence of a unified benchmark where all types of data coexist forces evaluations on separated benchmarks, causing \textit{inconsistency} between training and testing, thus leading to performance \textit{degradation}. To address these issues, this work advances in two aspects: \ding{182} A unified benchmark, coined as UniBench300, is introduced to bridge the inconsistency by incorporating multiple task data, reducing inference passes from three to one and cutting time consumption by 27\%. \ding{183} The unification process is reformulated in a serial format, progressively integrating new tasks. In this way, the performance degradation can be specified as knowledge forgetting of previous tasks, which naturally aligns with the philosophy of continual learning (CL), motivating further exploration of injecting CL into the unification process. Extensive experiments conducted on two baselines and four benchmarks demonstrate the significance of UniBench300 and the superiority of CL in supporting a stable unification process. Moreover, while conducting dedicated analyses, the performance degradation is found to be negatively correlated with network capacity. Additionally, modality discrepancies contribute to varying degradation levels across tasks (RGBT > RGBD > RGBE in MMVOT), offering valuable insights for future multi-modal vision research. Source codes and the proposed benchmark is available at \textit{this https URL}. 

**Abstract (ZH)**: 统一多种多模态视觉对象跟踪任务（MMVOT）随着不同模态互补性在构建稳健跟踪系统中的作用而越来越受到关注。现有的实践将所有数据传感器类型混入单一训练过程，从数据为中心的角度构建并行范式，旨在针对涉及的任务联合分布达到全局最优。然而，缺乏一个所有类型数据共存的统一基准导致在分离的基准上进行评估，从而在训练和测试之间造成不一致，进而导致性能退化。为解决这些问题，本工作在两个方面进行了改进：1）提出一个统一样本集UniBench300，通过整合多种任务数据，将推理次数从三次减少到一次，并将时间消耗缩减27%来缓解不一致性问题。2）统一过程重新表述为顺序格式，逐步整合新的任务。这样，性能退化可以被规定为对先前任务的知识遗忘，这自然与连续学习（CL）的哲学相一致，激励将CL注入统一过程中的进一步探索。在两个基线上和四个基准上的广泛实验展示了UniBench300的重要性以及CL在支持稳定统一过程方面的优势。此外，在进行专门分析时发现，性能退化与网络容量呈负相关，并且模态差异导致不同任务的退化水平不同（在MMVOT中RGBT > RGBD > RGBE），为未来多模态视觉研究提供了宝贵的见解。源代码和提出的基准可在以下链接获取。 

---
# SPHENIC: Topology-Informed Multi-View Clustering for Spatial Transcriptomics 

**Title (ZH)**: Sphenic: 拓扑引导的多视图聚类方法应用于空间转录组学 

**Authors**: Chenkai Guo, Yikai Zhu, Jing Yangum, Renxiang Guan, Por Lip Yee, Guangdun Peng, Dayu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10646)  

**Abstract**: By incorporating spatial location information, spatial-transcriptomics clustering yields more comprehensive insights into cell subpopulation identification. Despite recent progress, existing methods have at least two limitations: (i) topological learning typically considers only representations of individual cells or their interaction graphs; however, spatial transcriptomic profiles are often noisy, making these approaches vulnerable to low-quality topological signals, and (ii) insufficient modeling of spatial neighborhood information leads to low-quality spatial embeddings. To address these limitations, we propose SPHENIC, a novel Spatial Persistent Homology Enhanced Neighborhood Integrative Clustering method. Specifically, SPHENIC incorporates invariant topological features into the clustering network to achieve stable representation learning. Additionally, to construct high-quality spatial embeddings that reflect the true cellular distribution, we design the Spatial Constraint and Distribution Optimization Module (SCDOM). This module increases the similarity between a cell's embedding and those of its spatial neighbors, decreases similarity with non-neighboring cells, and thereby produces clustering-friendly spatial embeddings. Extensive experiments on 14 benchmark spatial transcriptomic slices demonstrate that SPHENIC achieves superior performance on the spatial clustering task, outperforming existing state-of-the-art methods by 3.31%-6.54% over the best alternative. 

**Abstract (ZH)**: 通过整合空间位置信息，空间转录组学聚类提供了更全面的细胞亚群识别洞察。尽管近期有所进展，现有方法至少存在两个局限性：（i）拓扑学习通常仅考虑单个细胞的表示或它们的交互图；然而，空间转录组学资料往往噪声较大，使这些方法容易受到低质量拓扑信号的影响；（ii）空间邻域信息建模不足导致较低质量的空间嵌入。为解决这些局限，我们提出了一种新型的空间持久同胚增强邻域整合聚类方法 SPHENIC。具体而言，SPHENIC 将不变的拓扑特征整合到聚类网络中，以实现稳定的表现学习。此外，为了构建反映真实细胞分布的高质量空间嵌入，我们设计了空间约束和分布优化模块（SCDOM）。该模块增加了细胞嵌入与空间邻域细胞嵌入之间的相似性，减少了与非邻域细胞之间的相似性，从而生成有利于聚类的空间嵌入。在 14 个基准空间转录组学切片上进行的广泛实验表明，SPHENIC 在空间聚类任务中的性能优于现有最先进的方法，相较于最佳替代方法提高了 3.31%-6.54%。 

---
# Fourier-Guided Attention Upsampling for Image Super-Resolution 

**Title (ZH)**: Fourier引导注意力上采样用于图像超分辨率 

**Authors**: Daejune Choi, Youchan No, Jinhyung Lee, Duksu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.10616)  

**Abstract**: We propose Frequency-Guided Attention (FGA), a lightweight upsampling module for single image super-resolution. Conventional upsamplers, such as Sub-Pixel Convolution, are efficient but frequently fail to reconstruct high-frequency details and introduce aliasing artifacts. FGA addresses these issues by integrating (1) a Fourier feature-based Multi-Layer Perceptron (MLP) for positional frequency encoding, (2) a cross-resolution Correlation Attention Layer for adaptive spatial alignment, and (3) a frequency-domain L1 loss for spectral fidelity supervision. Adding merely 0.3M parameters, FGA consistently enhances performance across five diverse super-resolution backbones in both lightweight and full-capacity scenarios. Experimental results demonstrate average PSNR gains of 0.12~0.14 dB and improved frequency-domain consistency by up to 29%, particularly evident on texture-rich datasets. Visual and spectral evaluations confirm FGA's effectiveness in reducing aliasing and preserving fine details, establishing it as a practical, scalable alternative to traditional upsampling methods. 

**Abstract (ZH)**: 基于频率引导的注意力（FGA）：一种轻量级单图像超分辨率上采样模块 

---
# On Spectral Properties of Gradient-based Explanation Methods 

**Title (ZH)**: 基于梯度的解释方法的谱性质研究 

**Authors**: Amir Mehrpanah, Erik Englesson, Hossein Azizpour  

**Link**: [PDF](https://arxiv.org/pdf/2508.10595)  

**Abstract**: Understanding the behavior of deep networks is crucial to increase our confidence in their results. Despite an extensive body of work for explaining their predictions, researchers have faced reliability issues, which can be attributed to insufficient formalism. In our research, we adopt novel probabilistic and spectral perspectives to formally analyze explanation methods. Our study reveals a pervasive spectral bias stemming from the use of gradient, and sheds light on some common design choices that have been discovered experimentally, in particular, the use of squared gradient and input perturbation. We further characterize how the choice of perturbation hyperparameters in explanation methods, such as SmoothGrad, can lead to inconsistent explanations and introduce two remedies based on our proposed formalism: (i) a mechanism to determine a standard perturbation scale, and (ii) an aggregation method which we call SpectralLens. Finally, we substantiate our theoretical results through quantitative evaluations. 

**Abstract (ZH)**: 深入理解深度网络的行为对于增强我们对它们结果的信心至关重要。尽管已有大量研究致力于解释其预测结果，研究人员仍面临可靠性问题，这可以归因于缺乏足够的形式化方法。在我们的研究中，我们采用新颖的概率和谱视角对解释方法进行正式分析。我们的研究揭示了由于使用梯度而导致的普遍谱偏差，并阐明了一些实验中发现的常见设计选择，特别是使用平方梯度和输入扰动。我们进一步分析了在解释方法中，如SmoothGrad，扰动超参数选择如何导致不一致的解释，并基于我们提出的形式化方法提出了两种补救措施：(i) 确定标准扰动尺度的机制，(ii) 称为SpectralLens的聚合方法。最后，我们通过定量评估验证了我们的理论结果。 

---
# FreeGAD: A Training-Free yet Effective Approach for Graph Anomaly Detection 

**Title (ZH)**: FreeGAD: 一种无需训练的有效图异常检测方法 

**Authors**: Yunfeng Zhao, Yixin Liu, Shiyuan Li, Qingfeng Chen, Yu Zheng, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2508.10594)  

**Abstract**: Graph Anomaly Detection (GAD) aims to identify nodes that deviate from the majority within a graph, playing a crucial role in applications such as social networks and e-commerce. Despite the current advancements in deep learning-based GAD, existing approaches often suffer from high deployment costs and poor scalability due to their complex and resource-intensive training processes. Surprisingly, our empirical findings suggest that the training phase of deep GAD methods, commonly perceived as crucial, may actually contribute less to anomaly detection performance than expected. Inspired by this, we propose FreeGAD, a novel training-free yet effective GAD method. Specifically, it leverages an affinity-gated residual encoder to generate anomaly-aware representations. Meanwhile, FreeGAD identifies anchor nodes as pseudo-normal and anomalous guides, followed by calculating anomaly scores through anchor-guided statistical deviations. Extensive experiments demonstrate that FreeGAD achieves superior anomaly detection performance, efficiency, and scalability on multiple benchmark datasets from diverse domains, without any training or iterative optimization. 

**Abstract (ZH)**: 无监督图异常检测（FreeGAD）：一种训练-free的有效图异常检测方法 

---
# Fake Speech Wild: Detecting Deepfake Speech on Social Media Platform 

**Title (ZH)**: 假语音泛滥：社交媒体平台上的深度伪造语音检测 

**Authors**: Yuankun Xie, Ruibo Fu, Xiaopeng Wang, Zhiyong Wang, Ya Li, Zhengqi Wen, Haonnan Cheng, Long Ye  

**Link**: [PDF](https://arxiv.org/pdf/2508.10559)  

**Abstract**: The rapid advancement of speech generation technology has led to the widespread proliferation of deepfake speech across social media platforms. While deepfake audio countermeasures (CMs) achieve promising results on public datasets, their performance degrades significantly in cross-domain scenarios. To advance CMs for real-world deepfake detection, we first propose the Fake Speech Wild (FSW) dataset, which includes 254 hours of real and deepfake audio from four different media platforms, focusing on social media. As CMs, we establish a benchmark using public datasets and advanced selfsupervised learning (SSL)-based CMs to evaluate current CMs in real-world scenarios. We also assess the effectiveness of data augmentation strategies in enhancing CM robustness for detecting deepfake speech on social media. Finally, by augmenting public datasets and incorporating the FSW training set, we significantly advanced real-world deepfake audio detection performance, achieving an average equal error rate (EER) of 3.54% across all evaluation sets. 

**Abstract (ZH)**: 快速发展的语音生成技术导致了深度伪造语音在社交媒体平台上的广泛传播。尽管深度伪造音频对抗措施（CMs）在公共数据集中取得了令人鼓舞的结果，但在跨域场景中的性能显著下降。为了推动CMs在现实世界的深度伪造检测应用，我们首先提出了Fake Speech Wild（FSW）数据集，该数据集包含来自四个不同媒体平台的254小时真实和深度伪造音频，重点关注社交媒体。作为CMs，我们使用公共数据集和先进的自监督学习（SSL）基于的CMs建立了基准，以评估当前CMs在现实世界场景中的表现。我们也评估了数据增强策略在增强CM检测社交媒体中的深度伪造语音的鲁棒性方面的有效性。最后，通过扩充公共数据集并结合FSW训练集，我们显著提升了现实世界的深度伪造音频检测性能，各评估集的平均等错误率（EER）为3.54%。 

---
# PTQAT: A Hybrid Parameter-Efficient Quantization Algorithm for 3D Perception Tasks 

**Title (ZH)**: PTQAT：一种适用于3D感知任务的混合参数高效量化算法 

**Authors**: Xinhao Wang, Zhiwei Lin, Zhongyu Xia, Yongtao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10557)  

**Abstract**: Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT) represent two mainstream model quantization approaches. However, PTQ often leads to unacceptable performance degradation in quantized models, while QAT imposes substantial GPU memory requirements and extended training time due to weight this http URL this paper, we propose PTQAT, a novel general hybrid quantization algorithm for the efficient deployment of 3D perception networks. To address the speed accuracy trade-off between PTQ and QAT, our method selects critical layers for QAT fine-tuning and performs PTQ on the remaining layers. Contrary to intuition, fine-tuning the layers with smaller output discrepancies before and after quantization, rather than those with larger discrepancies, actually leads to greater improvements in the model's quantization accuracy. This means we better compensate for quantization errors during their propagation, rather than addressing them at the point where they occur. The proposed PTQAT achieves similar performance to QAT with more efficiency by freezing nearly 50% of quantifiable layers. Additionally, PTQAT is a universal quantization method that supports various quantization bit widths (4 bits) as well as different model architectures, including CNNs and Transformers. The experimental results on nuScenes across diverse 3D perception tasks, including object detection, semantic segmentation, and occupancy prediction, show that our method consistently outperforms QAT-only baselines. Notably, it achieves 0.2%-0.9% NDS and 0.3%-1.0% mAP gains in object detection, 0.3%-2.0% mIoU gains in semantic segmentation and occupancy prediction while fine-tuning fewer weights. 

**Abstract (ZH)**: Post-Training 量化 (PTQ) 和 知识蒸馏感知训练 (QAT) 代表两种主流的模型量化方法。然而，PTQ 经常导致量化模型性能不可接受的下降，而 QAT 由于权重敏感性问题导致显著的 GPU 内存需求和延长的训练时间。本文提出 PTQAT，一种新型的通用混合量化算法，用于高效部署 3D 感知网络。为了在 PTQ 和 QAT 之间解决速度与准确性的权衡问题，我们的方法选择关键层进行 QAT 微调，并对剩余层进行 PTQ。与直觉相反，量化前后输出差异较小的层微调实际上可以带来更大的模型量化精度提升。这表明我们更好地在误差传播过程中补偿量化误差，而不是在它们发生时解决。所提出的 PTQAT 通过冻结近 50% 的可量化层，以更高的效率达到与 QAT 相似的表现。此外，PTQAT 是一种通用量化方法，支持不同的量化位宽（4 位）以及包括 CNN 和 Transformer 在内的各种模型架构。 nuScenes 上跨多种 3D 感知任务（包括物体检测、语义分割和占用预测）的实验结果表明，我们的方法始终优于仅使用 QAT 的基线方法。值得注意的是，在物体检测中实现 0.2%-0.9% 的 NDS 和 0.3%-1.0% 的 mAP 提升，在语义分割和占用预测中分别实现 0.3%-2.0% 的 mIoU 提升，同时微调的权重更少。 

---
# Retrieval-Augmented Prompt for OOD Detection 

**Title (ZH)**: 用于OOD检测的检索增强提示 

**Authors**: Ruisong Han, Zongbo Han, Jiahao Zhang, Mingyue Cheng, Changqing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10556)  

**Abstract**: Out-of-Distribution (OOD) detection is crucial for the reliable deployment of machine learning models in-the-wild, enabling accurate identification of test samples that differ from the training data distribution. Existing methods rely on auxiliary outlier samples or in-distribution (ID) data to generate outlier information for training, but due to limited outliers and their mismatch with real test OOD samples, they often fail to provide sufficient semantic supervision, leading to suboptimal performance. To address this, we propose a novel OOD detection method called Retrieval-Augmented Prompt (RAP). RAP augments a pre-trained vision-language model's prompts by retrieving external knowledge, offering enhanced semantic supervision for OOD detection. During training, RAP retrieves descriptive words for outliers based on joint similarity with external textual knowledge and uses them to augment the model's OOD prompts. During testing, RAP dynamically updates OOD prompts in real-time based on the encountered OOD samples, enabling the model to rapidly adapt to the test environment. Our extensive experiments demonstrate that RAP achieves state-of-the-art performance on large-scale OOD detection benchmarks. For example, in 1-shot OOD detection on the ImageNet-1k dataset, RAP reduces the average FPR95 by 7.05% and improves the AUROC by 1.71% compared to previous methods. Additionally, comprehensive ablation studies validate the effectiveness of each module and the underlying motivations of our approach. 

**Abstract (ZH)**: Out-of-Distribution (OOD)检测对于机器学习模型的可靠野外部署至关重要，能够准确识别与训练数据分布不同的测试样本。现有方法依赖于辅助异常样本或在分布（ID）数据来生成异常信息用于训练，但由于异常样本有限且与真实测试OOD样本不匹配，这些方法往往无法提供足够的语义监督，导致性能不佳。为解决这一问题，我们提出了一种新型的OOD检测方法称为检索增强提示（RAP）。RAP通过检索外部知识来增强预训练的跨模态模型的提示，提供增强的语义监督用于OOD检测。在训练阶段，RAP基于与外部文本知识的联合相似性检索异常描述词，并用于增强模型的OOD提示。在测试阶段，RAP根据遇到的OOD样本实时动态更新OOD提示，使模型能够快速适应测试环境。我们的实验表明，RAP在大规模OOD检测基准上达到了最先进的性能。例如，在ImageNet-1k数据集上的1-shot OOD检测中，RAP将平均FPR95降低了7.05%，AUROC提升了1.71%，优于以往方法。此外，全面的消融研究验证了每个模块的有效性及我们方法背后的动机。 

---
# When Language Overrules: Revealing Text Dominance in Multimodal Large Language Models 

**Title (ZH)**: 当语言占据主导：揭示多模态大型语言模型中的文本主导性 

**Authors**: Huyu Wu, Meng Tang, Xinhan Zheng, Haiyun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10552)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities across a diverse range of multimodal tasks. However, these models suffer from a core problem known as text dominance: they depend heavily on text for their inference, while underutilizing other modalities. While prior work has acknowledged this phenomenon in vision-language tasks, often attributing it to data biases or model architectures. In this paper, we conduct the first systematic investigation of text dominance across diverse data modalities, including images, videos, audio, time-series, and graphs. To measure this imbalance, we propose two evaluation metrics: the Modality Dominance Index (MDI) and the Attention Efficiency Index (AEI). Our comprehensive analysis reveals that text dominance is both significant and pervasive across all tested modalities. Our in-depth analysis identifies three underlying causes: attention dilution from severe token redundancy in non-textual modalities, the influence of fusion architecture design, and task formulations that implicitly favor textual inputs. Furthermore, we propose a simple token compression method that effectively rebalances model attention. Applying this method to LLaVA-7B, for instance, drastically reduces its MDI from 10.23 to a well-balanced value of 0.86. Our analysis and methodological framework offer a foundation for the development of more equitable and comprehensive multimodal language models. 

**Abstract (ZH)**: 多模态大语言模型（MLLMs）在多种多模态任务中展现了出色的性能。然而，这些模型面临着一个核心问题：文本主导性，即它们过于依赖文本进行推理，而未能充分利用其他模态信息。尽管先前的研究已认识到这一现象在视觉-语言任务中的存在，并通常将其归因于数据偏差或模型架构的问题。在本文中，我们首次对文本主导性在图像、视频、音频、时间序列和图形等多种数据模态中的现象进行了系统的探讨。为了衡量这种不平衡，我们提出了两个评估指标：模态主导性指数（MDI）和注意力效率指数（AEI）。全面的分析表明，文本主导性在所有测试的模态中都具有显著性和普遍性。深入分析揭示了三个根本原因：非文本模态中严重标记冗余导致的注意力稀释、融合架构设计的影响以及隐式偏向文本输入的任务表述。此外，我们提出了一种简单的标记压缩方法，其能有效重新平衡模型的注意力。例如，将该方法应用于LLaVA-7B模型时，其MDI从10.23显著降低到了均衡值0.86。我们的分析和方法论框架为开发更加公平和全面的多模态语言模型提供了基础。 

---
# Stabilizing Long-term Multi-turn Reinforcement Learning with Gated Rewards 

**Title (ZH)**: 使用门控奖励稳定长期多轮强化学习 

**Authors**: Zetian Sun, Dongfang Li, Zhuoen Chen, Yuhuai Qin, Baotian Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10548)  

**Abstract**: Reward sparsity in long-horizon reinforcement learning (RL) tasks remains a significant challenge, while existing outcome-based reward shaping struggles to define meaningful immediate rewards without introducing bias or requiring explicit task decomposition. Alternatively, verification-based reward shaping uses stepwise critics, but misalignment between immediate rewards and long-term objectives can lead to reward hacking and suboptimal policies. In this work, we address this problem in the context of software engineering (SWE) tasks, where multi-turn reasoning and rule-based verification are critical. We introduce the SWE-oriented RL Framework, a unified system supporting multi-turn interaction, docker-based execution, and customizable reward functions. Additionally, we propose Gated Reward Accumulation (G-RA), a novel method that accumulates immediate rewards only when high-level (long-term) rewards meet a predefined threshold, ensuring stable RL optimization. Experiments on SWE-bench Verified and kBench demonstrate that G-RA leads to an increase in completion rates (47.6\% \rightarrow 93.8\% and 22.0\% \rightarrow 86.0\%) and modification rates (19.6\% \rightarrow 23.8\% and 12.0\% \rightarrow 42.0\%), while avoiding policy degradation caused by reward misalignment. Our findings highlight the importance of balanced reward accumulation in long-horizon RL and provide a practical solution. 

**Abstract (ZH)**: 长时程强化学习中奖励稀疏性的处理仍然是一个重大挑战，现有的基于结果的奖励塑造难以定义没有偏见的即时奖励或无需明确任务分解。相比之下，基于验证的奖励塑造使用逐步评论员，但即时奖励与长期目标之间的不一致可能导致奖励作弊和次优策略。在软件工程任务的背景下，我们解决了这一问题，其中多轮推理和基于规则的验证至关重要。我们引入了面向软件工程的RL框架，该框架是一个支持多轮交互、基于Docker的执行和可定制奖励函数的统一系统。此外，我们提出了门控奖励累积（G-RA）方法，该方法仅在高层（长期）奖励达到预定义阈值时累积即时奖励，从而确保RL优化的稳定性。在SWE-bench Verified和kBench上的实验表明，G-RA提高了完成率（47.6% → 93.8%和22.0% → 86.0%）和修改率（19.6% → 23.8%和12.0% → 42.0%），同时避免了由奖励不一致引起的策略退化。我们的研究结果强调了在长时程RL中平衡奖励累积的重要性，并提供了一种实用的解决方案。 

---
# Med-GLIP: Advancing Medical Language-Image Pre-training with Large-scale Grounded Dataset 

**Title (ZH)**: Med-GLIP: 基于大规模grounded数据集的医学语言-图像预训练 

**Authors**: Ziye Deng, Ruihan He, Jiaxiang Liu, Yuan Wang, Zijie Meng, Songtao Jiang, Yong Xie, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10528)  

**Abstract**: Medical image grounding aims to align natural language phrases with specific regions in medical images, serving as a foundational task for intelligent diagnosis, visual question answering (VQA), and automated report generation (MRG). However, existing research is constrained by limited modality coverage, coarse-grained annotations, and the absence of a unified, generalizable grounding framework. To address these challenges, we construct a large-scale medical grounding dataset Med-GLIP-5M comprising over 5.3 million region-level annotations across seven imaging modalities, covering diverse anatomical structures and pathological findings. The dataset supports both segmentation and grounding tasks with hierarchical region labels, ranging from organ-level boundaries to fine-grained lesions. Based on this foundation, we propose Med-GLIP, a modality-aware grounding framework trained on Med-GLIP-5M. Rather than relying on explicitly designed expert modules, Med-GLIP implicitly acquires hierarchical semantic understanding from diverse training data -- enabling it to recognize multi-granularity structures, such as distinguishing lungs from pneumonia lesions. Extensive experiments demonstrate that Med-GLIP consistently outperforms state-of-the-art baselines across multiple grounding benchmarks. Furthermore, integrating its spatial outputs into downstream tasks, including medical VQA and report generation, leads to substantial performance gains. Our dataset will be released soon. 

**Abstract (ZH)**: 医学图像接地旨在将自然语言短语与医学图像中的特定区域对齐，作为智能诊断、视觉问答（VQA）和自动化报告生成（MRG）的基础任务。然而，现有研究受限于模态覆盖有限、标注粗糙以及缺乏统一可泛化的接地框架。为应对这些挑战，我们构建了一个名为Med-GLIP-5M的大规模医学接地数据集，包含超过530万张区域级标注，涵盖了七种成像模态，全面覆盖了多样的解剖结构和病理发现。该数据集支持分割和接地任务，提供了从器官级边界到细微病灶的层次化区域标签。基于这一基础，我们提出Med-GLIP，这是一种基于Med-GLIP-5M训练的模态感知接地框架。Med-GLIP 不依赖于显式设计的专家模块，而是从多样化的训练数据中隐式获取层次语义理解，使其能够识别多粒度结构，如区分肺部与肺炎病灶。广泛实验表明，Med-GLIP 在多个接地基准测试中一贯优于现有最先进的基线。进一步将其实空间输出集成到下游任务，如医学VQA和报告生成中，可实现显著的性能提升。我们的数据集将很快发布。 

---
# Multi-Sample Anti-Aliasing and Constrained Optimization for 3D Gaussian Splatting 

**Title (ZH)**: 多样本抗锯齿与约束优化在3D高斯斑点绘制中的应用 

**Authors**: Zheng Zhou, Jia-Chen Zhang, Yu-Jie Xiong, Chun-Ming Xia  

**Link**: [PDF](https://arxiv.org/pdf/2508.10507)  

**Abstract**: Recent advances in 3D Gaussian splatting have significantly improved real-time novel view synthesis, yet insufficient geometric constraints during scene optimization often result in blurred reconstructions of fine-grained details, particularly in regions with high-frequency textures and sharp discontinuities. To address this, we propose a comprehensive optimization framework integrating multisample anti-aliasing (MSAA) with dual geometric constraints. Our system computes pixel colors through adaptive blending of quadruple subsamples, effectively reducing aliasing artifacts in high-frequency components. The framework introduces two constraints: (a) an adaptive weighting strategy that prioritizes under-reconstructed regions through dynamic gradient analysis, and (b) gradient differential constraints enforcing geometric regularization at object boundaries. This targeted optimization enables the model to allocate computational resources preferentially to critical regions requiring refinement while maintaining global consistency. Extensive experimental evaluations across multiple benchmarks demonstrate that our method achieves state-of-the-art performance in detail preservation, particularly in preserving high-frequency textures and sharp discontinuities, while maintaining real-time rendering efficiency. Quantitative metrics and perceptual studies confirm statistically significant improvements over baseline approaches in both structural similarity (SSIM) and perceptual quality (LPIPS). 

**Abstract (ZH)**: Recent Advances in 3D Gaussian Splatting Integrating Multisample Anti-Aliasing with Dual Geometric Constraints for Detail Preservation in Real-Time Novel View Synthesis 

---
# Advances in Logic-Based Entity Resolution: Enhancing ASPEN with Local Merges and Optimality Criteria 

**Title (ZH)**: 基于逻辑的实体解析进展：通过局部合并和最优准则增强ASPEN 

**Authors**: Zhliang Xiang, Meghyn Bienvenu, Gianluca Cima, Víctor Gutiérrez-Basulto, Yazmín Ibáñez-García  

**Link**: [PDF](https://arxiv.org/pdf/2508.10504)  

**Abstract**: In this paper, we present ASPEN+, which extends an existing ASP-based system, ASPEN,for collective entity resolution with two important functionalities: support for local merges and new optimality criteria for preferred solutions. Indeed, ASPEN only supports so-called global merges of entity-referring constants (e.g. author ids), in which all occurrences of matched constants are treated as equivalent and merged accordingly. However, it has been argued that when resolving data values, local merges are often more appropriate, as e.g. some instances of 'J. Lee' may refer to 'Joy Lee', while others should be matched with 'Jake Lee'. In addition to allowing such local merges, ASPEN+ offers new optimality criteria for selecting solutions, such as minimizing rule violations or maximising the number of rules supporting a merge. Our main contributions are thus (1) the formalisation and computational analysis of various notions of optimal solution, and (2) an extensive experimental evaluation on real-world datasets, demonstrating the effect of local merges and the new optimality criteria on both accuracy and runtime. 

**Abstract (ZH)**: AS titled "这篇 paper的标题是：

基于现有的ASP基于的ASPEN，的ASPEN扩展，，集体实体解析，，，两个功能特性 on：支持局部合并和 最优化准则 for 优选项合并 on。..。因此 on 传统的ASPEN仅支持全局常量合并 on 即实体引用常量（例如 如例如 类似于e e.j lee 的 e 例如 e Joy Lee e 与其 e e e.e e Lee对应 �认为前者更适合 on 例如 例如 例如 e e e.e Jake Lee e �اض此外 on 本 ASPEN e还 �还 � e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e � e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e.e 最后 e 的 主 e e e 的 e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e-initialized on on e e e e e e e e e e e e e e 上 e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e 最大优化 e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e(e 最e e e e e e 第 e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e 最优化 e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e 最 e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e拆迁和 e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e the e proposed e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e 最新 e e e e e e e e e e e e 最示 e e e e e e e e e e e e e e e e e e e e e e e e 最 e e e e e e 两大贡献包括 e e e e e e e e 上: e e e e e e e e e e e e e e e e e e 最 e e e e � e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e 最完备的形式和 e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e 最模式和 e e e e e e e e e e e 最计算 e e e e e e e e e e e e e e e e e e e e e e e e 最全 e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e(easd génér
 � titl in t s as t t e t t e e t t e: 埶 e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e entitled � e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e pestic e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e 

---
# A Unified Multi-Agent Framework for Universal Multimodal Understanding and Generation 

**Title (ZH)**: 统一多Agent框架：面向通用多模态理解与生成 

**Authors**: Jiulin Li, Ping Huang, Yexin Li, Shuo Chen, Juewen Hu, Ye Tian  

**Link**: [PDF](https://arxiv.org/pdf/2508.10494)  

**Abstract**: Real-world multimodal applications often require any-to-any capabilities, enabling both understanding and generation across modalities including text, image, audio, and video. However, integrating the strengths of autoregressive language models (LLMs) for reasoning and diffusion models for high-fidelity generation remains challenging. Existing approaches rely on rigid pipelines or tightly coupled architectures, limiting flexibility and scalability. We propose MAGUS (Multi-Agent Guided Unified Multimodal System), a modular framework that unifies multimodal understanding and generation via two decoupled phases: Cognition and Deliberation. MAGUS enables symbolic multi-agent collaboration within a shared textual workspace. In the Cognition phase, three role-conditioned multimodal LLM agents - Perceiver, Planner, and Reflector - engage in collaborative dialogue to perform structured understanding and planning. The Deliberation phase incorporates a Growth-Aware Search mechanism that orchestrates LLM-based reasoning and diffusion-based generation in a mutually reinforcing manner. MAGUS supports plug-and-play extensibility, scalable any-to-any modality conversion, and semantic alignment - all without the need for joint training. Experiments across multiple benchmarks, including image, video, and audio generation, as well as cross-modal instruction following, demonstrate that MAGUS outperforms strong baselines and state-of-the-art systems. Notably, on the MME benchmark, MAGUS surpasses the powerful closed-source model GPT-4o. 

**Abstract (ZH)**: 多模态应用通常需要任意到任意的能力，以跨文本、图像、音频和视频等多种模态实现理解和生成。然而，如何结合自回归语言模型（LLMs）的推理能力和扩散模型的高保真生成能力依然具有挑战性。现有方法依赖于刚性的工作流程或紧密耦合的架构，限制了灵活性和可扩展性。我们提出了MAGUS（多代理引导统一多模态系统），这是一种模块化的框架，通过两个分离的阶段——认知和决断，来统一多模态的理解和生成。MAGUS允许在共享的文本工作空间中进行符号化的多代理合作。在认知阶段，三位基于角色的多模态LLM代理——Perceiver、Planner和Reflector——进行协作对话，以执行结构化理解和计划。决断阶段采用感知增长搜索机制，协调基于LLM的推理和基于扩散的生成，在相互强化的过程中进行调控。MAGUS支持插拔式扩展、大规模任意到任意的模态转换以及语义对齐，无需联合训练。在多种基准测试中，包括图像、视频和音频生成，以及跨模态指令跟随，MAGUS均表现出色，超越了强大的基线和最先进的系统。值得注意的是，MAGUS在MME基准测试中超过了强大的闭源模型GPT-4o。 

---
# Contrastive ECOC: Learning Output Codes for Adversarial Defense 

**Title (ZH)**: 对比性ECOC：用于对抗性防御的学习输出码 

**Authors**: Che-Yu Chou, Hung-Hsuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10491)  

**Abstract**: Although one-hot encoding is commonly used for multiclass classification, it is not always the most effective encoding mechanism. Error Correcting Output Codes (ECOC) address multiclass classification by mapping each class to a unique codeword used as a label. Traditional ECOC methods rely on manually designed or randomly generated codebooks, which are labor-intensive and may yield suboptimal, dataset-agnostic results. This paper introduces three models for automated codebook learning based on contrastive learning, allowing codebooks to be learned directly and adaptively from data. Across four datasets, our proposed models demonstrate superior robustness to adversarial attacks compared to two baselines. The source is available at this https URL. 

**Abstract (ZH)**: 虽然one-hot编码常用于多类分类，但并非总是最有效的编码机制。错误修正输出编码（ECOC）通过将每个类别映射到一个唯一的码字来解决多类分类问题，该码字用作标签。传统的ECOC方法依赖于手动设计或随机生成的码本，这需要大量人工劳动，并可能得到不适用于特定数据集的次优结果。本文介绍三种基于对比学习的自动码本学习模型，使码本可以直接从数据中学习和适应。在四个数据集上，我们提出的模型在对抗攻击中的鲁棒性优于两种基线方法。源代码可在以下链接获取。 

---
# On the Complexity-Faithfulness Trade-off of Gradient-Based Explanations 

**Title (ZH)**: 基于梯度的解释的复杂性忠实性权衡 

**Authors**: Amir Mehrpanah, Matteo Gamba, Kevin Smith, Hossein Azizpour  

**Link**: [PDF](https://arxiv.org/pdf/2508.10490)  

**Abstract**: ReLU networks, while prevalent for visual data, have sharp transitions, sometimes relying on individual pixels for predictions, making vanilla gradient-based explanations noisy and difficult to interpret. Existing methods, such as GradCAM, smooth these explanations by producing surrogate models at the cost of faithfulness. We introduce a unifying spectral framework to systematically analyze and quantify smoothness, faithfulness, and their trade-off in explanations. Using this framework, we quantify and regularize the contribution of ReLU networks to high-frequency information, providing a principled approach to identifying this trade-off. Our analysis characterizes how surrogate-based smoothing distorts explanations, leading to an ``explanation gap'' that we formally define and measure for different post-hoc methods. Finally, we validate our theoretical findings across different design choices, datasets, and ablations. 

**Abstract (ZH)**: ReLU网络在视觉数据中广泛应用，但存在尖锐的过渡，有时依赖个别像素进行预测，导致基于梯度的解释方法噪声较大且难以解释。现有的方法，如GradCAM，通过构建替代模型来平滑这些解释，但会以降低忠实性为代价。我们提出了一种统一的谱方法体系，系统地分析和量化平滑性、忠实性及其在解释中的权衡。利用该框架，我们量化并正则化ReLU网络对高频信息的贡献，提供了一种识别这种权衡的原理性方法。我们的分析描述了基于替代模型的平滑如何扭曲解释，从而正式定义并测量不同事后方法的“解释差距”。最后，我们在不同的设计选择、数据集和消融实验中验证了我们的理论发现。 

---
# Pinet: Optimizing hard-constrained neural networks with orthogonal projection layers 

**Title (ZH)**: Pinet：通过正交投影层优化具有约束条件的神经网络 

**Authors**: Panagiotis D. Grontas, Antonio Terpin, Efe C. Balta, Raffaello D'Andrea, John Lygeros  

**Link**: [PDF](https://arxiv.org/pdf/2508.10480)  

**Abstract**: We introduce an output layer for neural networks that ensures satisfaction of convex constraints. Our approach, $\Pi$net, leverages operator splitting for rapid and reliable projections in the forward pass, and the implicit function theorem for backpropagation. We deploy $\Pi$net as a feasible-by-design optimization proxy for parametric constrained optimization problems and obtain modest-accuracy solutions faster than traditional solvers when solving a single problem, and significantly faster for a batch of problems. We surpass state-of-the-art learning approaches in terms of training time, solution quality, and robustness to hyperparameter tuning, while maintaining similar inference times. Finally, we tackle multi-vehicle motion planning with non-convex trajectory preferences and provide $\Pi$net as a GPU-ready package implemented in JAX with effective tuning heuristics. 

**Abstract (ZH)**: 我们引入了一种输出层，确保神经网络满足凸约束。我们的方法$\Pi$net利用分裂算子在前向传递中进行快速可靠的投影，并利用隐函数定理进行反向传播。我们将$\Pi$net部署为参数约束优化问题的可行设计优化代理，在解决单个问题时比传统求解器更快获得适度准确的解决方案，对于批量问题则显著更快。在训练时间、解的质量和对超参数调优的鲁棒性方面，$\Pi$net超越了最先进的学习方法，同时保持类似的推理时间。最后，我们处理具有非凸轨迹偏好的多车辆运动规划问题，并提供基于JAX实现的$\Pi$net GPU就绪包，配有有效的调优启发式方法。 

---
# Enhanced Sparse Point Cloud Data Processing for Privacy-aware Human Action Recognition 

**Title (ZH)**: 增强稀疏点云数据处理方法及其在隐私感知的人体动作识别中的应用 

**Authors**: Maimunatu Tunau, Vincent Gbouna Zakka, Zhuangzhuang Dai  

**Link**: [PDF](https://arxiv.org/pdf/2508.10469)  

**Abstract**: Human Action Recognition (HAR) plays a crucial role in healthcare, fitness tracking, and ambient assisted living technologies. While traditional vision based HAR systems are effective, they pose privacy concerns. mmWave radar sensors offer a privacy preserving alternative but present challenges due to the sparse and noisy nature of their point cloud data. In the literature, three primary data processing methods: Density-Based Spatial Clustering of Applications with Noise (DBSCAN), the Hungarian Algorithm, and Kalman Filtering have been widely used to improve the quality and continuity of radar data. However, a comprehensive evaluation of these methods, both individually and in combination, remains lacking. This paper addresses that gap by conducting a detailed performance analysis of the three methods using the MiliPoint dataset. We evaluate each method individually, all possible pairwise combinations, and the combination of all three, assessing both recognition accuracy and computational cost. Furthermore, we propose targeted enhancements to the individual methods aimed at improving accuracy. Our results provide crucial insights into the strengths and trade-offs of each method and their integrations, guiding future work on mmWave based HAR systems 

**Abstract (ZH)**: 毫米波雷达传感器在人体动作识别中的隐私保护作用及其数据处理方法研究：基于MiliPoint数据集的综合性能分析与优化 

---
# X-Node: Self-Explanation is All We Need 

**Title (ZH)**: X-Node: 自解释才是我们所需的一切 

**Authors**: Prajit Sengupta, Islem Rekik  

**Link**: [PDF](https://arxiv.org/pdf/2508.10461)  

**Abstract**: Graph neural networks (GNNs) have achieved state-of-the-art results in computer vision and medical image classification tasks by capturing structural dependencies across data instances. However, their decision-making remains largely opaque, limiting their trustworthiness in high-stakes clinical applications where interpretability is essential. Existing explainability techniques for GNNs are typically post-hoc and global, offering limited insight into individual node decisions or local reasoning. We introduce X-Node, a self-explaining GNN framework in which each node generates its own explanation as part of the prediction process. For every node, we construct a structured context vector encoding interpretable cues such as degree, centrality, clustering, feature saliency, and label agreement within its local topology. A lightweight Reasoner module maps this context into a compact explanation vector, which serves three purposes: (1) reconstructing the node's latent embedding via a decoder to enforce faithfulness, (2) generating a natural language explanation using a pre-trained LLM (e.g., Grok or Gemini), and (3) guiding the GNN itself via a "text-injection" mechanism that feeds explanations back into the message-passing pipeline. We evaluate X-Node on two graph datasets derived from MedMNIST and MorphoMNIST, integrating it with GCN, GAT, and GIN backbones. Our results show that X-Node maintains competitive classification accuracy while producing faithful, per-node explanations. Repository: this https URL. 

**Abstract (ZH)**: Graph神经网络(X-Node):一种自我解释的框架，其中每个节点在预测过程中生成自己的解释 

---
# RealAC: A Domain-Agnostic Framework for Realistic and Actionable Counterfactual Explanations 

**Title (ZH)**: RealAC：一种面向现实且可行动的反事实解释的领域无关框架 

**Authors**: Asiful Arefeen, Shovito Barua Soumma, Hassan Ghasemzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2508.10455)  

**Abstract**: Counterfactual explanations provide human-understandable reasoning for AI-made decisions by describing minimal changes to input features that would alter a model's prediction. To be truly useful in practice, such explanations must be realistic and feasible -- they should respect both the underlying data distribution and user-defined feasibility constraints. Existing approaches often enforce inter-feature dependencies through rigid, hand-crafted constraints or domain-specific knowledge, which limits their generalizability and ability to capture complex, nonlinear relations inherent in data. Moreover, they rarely accommodate user-specified preferences and suggest explanations that are causally implausible or infeasible to act upon. We introduce RealAC, a domain-agnostic framework for generating realistic and actionable counterfactuals. RealAC automatically preserves complex inter-feature dependencies without relying on explicit domain knowledge -- by aligning the joint distributions of feature pairs between factual and counterfactual instances. The framework also allows end-users to ``freeze'' attributes they cannot or do not wish to change by suppressing change in frozen features during optimization. Evaluations on three synthetic and two real datasets demonstrate that RealAC balances realism with actionability. Our method outperforms state-of-the-art baselines and Large Language Model-based counterfactual generation techniques in causal edge score, dependency preservation score, and IM1 realism metric and offers a solution for causality-aware and user-centric counterfactual generation. 

**Abstract (ZH)**: RealAC：一种领域无关的生成现实性和可操作性解释的框架 

---
# Alternating Approach-Putt Models for Multi-Stage Speech Enhancement 

**Title (ZH)**: 交替方法putt模型在多阶段语音增强中的应用 

**Authors**: Iksoon Jeong, Kyung-Joong Kim, Kang-Hun Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2508.10436)  

**Abstract**: Speech enhancement using artificial neural networks aims to remove noise from noisy speech signals while preserving the speech content. However, speech enhancement networks often introduce distortions to the speech signal, referred to as artifacts, which can degrade audio quality. In this work, we propose a post-processing neural network designed to mitigate artifacts introduced by speech enhancement models. Inspired by the analogy of making a `Putt' after an `Approach' in golf, we name our model PuttNet. We demonstrate that alternating between a speech enhancement model and the proposed Putt model leads to improved speech quality, as measured by perceptual quality scores (PESQ), objective intelligibility (STOI), and background noise intrusiveness (CBAK) scores. Furthermore, we illustrate with graphical analysis why this alternating Approach outperforms repeated application of either model alone. 

**Abstract (ZH)**: 使用人工神经网络的语音增强旨在从噪声语音信号中去除噪声同时保留语音内容。然而，语音增强网络往往会引入对语音信号的失真，称为伪影，这会降低音频质量。在本工作中，我们提出了一种后处理神经网络，旨在减轻由语音增强模型引入的伪影。受高尔夫中“推杆”和“接近杆”操作的启发，我们将我们的模型命名为PuttNet。我们证明交替使用语音增强模型和提议的Putt模型可以提高语音质量，可以通过感知质量评分（PESQ）、客观可懂度（STOI）和背景噪声侵入性（CBAK）评分来衡量。此外，我们通过图形分析说明了这种交替的“接近杆”操作为什么优于单独重复应用任一模型。 

---
# Unpacking the Implicit Norm Dynamics of Sharpness-Aware Minimization in Tensorized Models 

**Title (ZH)**: 拆解张量化模型中隐含规范动态的锐化感知最小化 italia 劚
user
拆解锐化感知最小化在张量量化模型中的隐含规范动力学。 

**Authors**: Tianxiao Cao, Kyohei Atarashi, Hisashi Kashima  

**Link**: [PDF](https://arxiv.org/pdf/2508.10435)  

**Abstract**: Sharpness-Aware Minimization (SAM) has been proven to be an effective optimization technique for improving generalization in overparameterized models. While prior works have explored the implicit regularization of SAM in simple two-core scale-invariant settings, its behavior in more general tensorized or scale-invariant models remains underexplored. In this work, we leverage scale-invariance to analyze the norm dynamics of SAM in general tensorized models. We introduce the notion of \emph{Norm Deviation} as a global measure of core norm imbalance, and derive its evolution under SAM using gradient flow analysis. We show that SAM's implicit control of Norm Deviation is governed by the covariance between core norms and their gradient magnitudes. Motivated by these findings, we propose a simple yet effective method, \emph{Deviation-Aware Scaling (DAS)}, which explicitly mimics this regularization behavior by scaling core norms in a data-adaptive manner. Our experiments across tensor completion, noisy training, model compression, and parameter-efficient fine-tuning confirm that DAS achieves competitive or improved performance over SAM, while offering reduced computational overhead. 

**Abstract (ZH)**: 具有规范偏差意识的缩放（Deviation-Aware Scaling, DAS）：在广义张量化模型中分析尖锐度感知最小化（Sharpness-Aware Minimization, SAM）的范数动力学 

---
# MASH: Cooperative-Heterogeneous Multi-Agent Reinforcement Learning for Single Humanoid Robot Locomotion 

**Title (ZH)**: MASH: 合作多模态多agent强化学习在单人形机器人行动中的应用 roma ">user
谢谢你，可以再翻译一遍确保准确性吗？ 

**Authors**: Qi Liu, Xiaopeng Zhang, Mingshan Tan, Shuaikang Ma, Jinliang Ding, Yanjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.10423)  

**Abstract**: This paper proposes a novel method to enhance locomotion for a single humanoid robot through cooperative-heterogeneous multi-agent deep reinforcement learning (MARL). While most existing methods typically employ single-agent reinforcement learning algorithms for a single humanoid robot or MARL algorithms for multi-robot system tasks, we propose a distinct paradigm: applying cooperative-heterogeneous MARL to optimize locomotion for a single humanoid robot. The proposed method, multi-agent reinforcement learning for single humanoid locomotion (MASH), treats each limb (legs and arms) as an independent agent that explores the robot's action space while sharing a global critic for cooperative learning. Experiments demonstrate that MASH accelerates training convergence and improves whole-body cooperation ability, outperforming conventional single-agent reinforcement learning methods. This work advances the integration of MARL into single-humanoid-robot control, offering new insights into efficient locomotion strategies. 

**Abstract (ZH)**: 本文提出了一种通过合作异构多智能体深度强化学习（MARL）增强单个人形机器人运动的新方法。 

---
# ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning 

**Title (ZH)**: CogRAG：一种基于认知的内存组织化RAG框架，用于状态保持长叙事推理 

**Authors**: Juyuan Wang, Rongchen Zhao, Wei Wei, Yufeng Wang, Mo Yu, Jie Zhou, Jin Xu, Liyan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10419)  

**Abstract**: Narrative comprehension on long stories and novels has been a challenging domain attributed to their intricate plotlines and entangled, often evolving relations among characters and entities. Given the LLM's diminished reasoning over extended context and high computational cost, retrieval-based approaches remain a pivotal role in practice. However, traditional RAG methods can fall short due to their stateless, single-step retrieval process, which often overlooks the dynamic nature of capturing interconnected relations within long-range context. In this work, we propose ComoRAG, holding the principle that narrative reasoning is not a one-shot process, but a dynamic, evolving interplay between new evidence acquisition and past knowledge consolidation, analogous to human cognition when reasoning with memory-related signals in the brain. Specifically, when encountering a reasoning impasse, ComoRAG undergoes iterative reasoning cycles while interacting with a dynamic memory workspace. In each cycle, it generates probing queries to devise new exploratory paths, then integrates the retrieved evidence of new aspects into a global memory pool, thereby supporting the emergence of a coherent context for the query resolution. Across four challenging long-context narrative benchmarks (200K+ tokens), ComoRAG outperforms strong RAG baselines with consistent relative gains up to 11% compared to the strongest baseline. Further analysis reveals that ComoRAG is particularly advantageous for complex queries requiring global comprehension, offering a principled, cognitively motivated paradigm for retrieval-based long context comprehension towards stateful reasoning. Our code is publicly released at this https URL 

**Abstract (ZH)**: 长篇故事和小说的叙述理解是一个具有挑战性的领域，归因于其复杂的故事情节和人物及实体之间错综复杂的、常常不断演变的关系。鉴于大语言模型在长时间上下文推理中的削弱表现和高计算成本，检索式方法在实践中仍然发挥着关键作用。然而，传统的RAG方法由于其无状态的、单步的检索过程，往往忽略了在长期上下文中捕捉互联关系的动态性质。在本文中，我们提出了ComoRAG，其原则是叙述推理不是一个一次性过程，而是一个在新证据获取与过去知识整合之间动态、演化的互动过程，类似于大脑在使用与记忆相关信号进行推理时的人类认知。具体而言，当遇到推理瓶颈时，ComoRAG会通过与动态记忆工作空间的交互进行迭代推理循环。在每个循环中，它生成探查查询以设计新的探索路径，然后将新方面的检索证据整合到全局记忆池中，从而支持查询解决的连贯上下文的生成。在四个具有挑战性的长上下文叙述基准测试中（包含200K+词），ComoRAG在与最强基线相比的情况下，一致获得了最高11%的相对增益成绩。进一步的分析表明，ComoRAG特别适用于需要全局理解的复杂查询，提供了一种基于检索、以状态密集推理为目标的认知驱动的框架。我们的代码已在此 https://公开发布。 

---
# CorrectNav: Self-Correction Flywheel Empowers Vision-Language-Action Navigation Model 

**Title (ZH)**: CorrectNav: 自校正飞轮赋能视觉-语言-动作导航模型 

**Authors**: Zhuoyuan Yu, Yuxing Long, Zihan Yang, Chengyan Zeng, Hongwei Fan, Jiyao Zhang, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2508.10416)  

**Abstract**: Existing vision-and-language navigation models often deviate from the correct trajectory when executing instructions. However, these models lack effective error correction capability, hindering their recovery from errors. To address this challenge, we propose Self-correction Flywheel, a novel post-training paradigm. Instead of considering the model's error trajectories on the training set as a drawback, our paradigm emphasizes their significance as a valuable data source. We have developed a method to identify deviations in these error trajectories and devised innovative techniques to automatically generate self-correction data for perception and action. These self-correction data serve as fuel to power the model's continued training. The brilliance of our paradigm is revealed when we re-evaluate the model on the training set, uncovering new error trajectories. At this time, the self-correction flywheel begins to spin. Through multiple flywheel iterations, we progressively enhance our monocular RGB-based VLA navigation model CorrectNav. Experiments on R2R-CE and RxR-CE benchmarks show CorrectNav achieves new state-of-the-art success rates of 65.1% and 69.3%, surpassing prior best VLA navigation models by 8.2% and 16.4%. Real robot tests in various indoor and outdoor environments demonstrate \method's superior capability of error correction, dynamic obstacle avoidance, and long instruction following. 

**Abstract (ZH)**: 自纠正飞轮：一种新的后训练范式 

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
# PQ-DAF: Pose-driven Quality-controlled Data Augmentation for Data-scarce Driver Distraction Detection 

**Title (ZH)**: 基于pose驱动的质量控制数据增强方法：在数据稀缺条件下的驾驶员分心检测 

**Authors**: Haibin Sun, Xinghui Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.10397)  

**Abstract**: Driver distraction detection is essential for improving traffic safety and reducing road accidents. However, existing models often suffer from degraded generalization when deployed in real-world scenarios. This limitation primarily arises from the few-shot learning challenge caused by the high cost of data annotation in practical environments, as well as the substantial domain shift between training datasets and target deployment conditions. To address these issues, we propose a Pose-driven Quality-controlled Data Augmentation Framework (PQ-DAF) that leverages a vision-language model for sample filtering to cost-effectively expand training data and enhance cross-domain robustness. Specifically, we employ a Progressive Conditional Diffusion Model (PCDMs) to accurately capture key driver pose features and synthesize diverse training examples. A sample quality assessment module, built upon the CogVLM vision-language model, is then introduced to filter out low-quality synthetic samples based on a confidence threshold, ensuring the reliability of the augmented dataset. Extensive experiments demonstrate that PQ-DAF substantially improves performance in few-shot driver distraction detection, achieving significant gains in model generalization under data-scarce conditions. 

**Abstract (ZH)**: 基于姿态驱动的质量控制数据增强框架（PQ-DAF）用于改进驾驶员注意力分散检测的泛化能力 

---
# Unlocking Robust Semantic Segmentation Performance via Label-only Elastic Deformations against Implicit Label Noise 

**Title (ZH)**: 通过基于标签的弹性变形应对隐式标签噪声以解锁稳健的语义分割性能 

**Authors**: Yechan Kim, Dongho Yoon, Younkwan Lee, Unse Fatima, Hong Kook Kim, Songjae Lee, Sanga Park, Jeong Ho Park, Seonjong Kang, Moongu Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2508.10383)  

**Abstract**: While previous studies on image segmentation focus on handling severe (or explicit) label noise, real-world datasets also exhibit subtle (or implicit) label imperfections. These arise from inherent challenges, such as ambiguous object boundaries and annotator variability. Although not explicitly present, such mild and latent noise can still impair model performance. Typical data augmentation methods, which apply identical transformations to the image and its label, risk amplifying these subtle imperfections and limiting the model's generalization capacity. In this paper, we introduce NSegment+, a novel augmentation framework that decouples image and label transformations to address such realistic noise for semantic segmentation. By introducing controlled elastic deformations only to segmentation labels while preserving the original images, our method encourages models to focus on learning robust representations of object structures despite minor label inconsistencies. Extensive experiments demonstrate that NSegment+ consistently improves performance, achieving mIoU gains of up to +2.29, +2.38, +1.75, and +3.39 in average on Vaihingen, LoveDA, Cityscapes, and PASCAL VOC, respectively-even without bells and whistles, highlighting the importance of addressing implicit label noise. These gains can be further amplified when combined with other training tricks, including CutMix and Label Smoothing. 

**Abstract (ZH)**: 尽管以往的图像分割研究主要关注处理严重的（或明确的）标签噪声，但现实世界的数据集也表现出轻微的（或隐含的）标签不完美现象。这些不完美现象源自固有的挑战，如模糊的对象边界和注释员的变异性。虽然这些轻微和潜在的噪声没有明确存在，但仍然可能损害模型的性能。典型的数据增强方法通过对图像和标签应用相同的变换，有放大这些轻微不完美现象的风险，从而限制模型的泛化能力。本文介绍了一种新颖的数据增强框架NSegment+，该框架将图像和标签的变换解耦，以应对语义分割中的现实噪声。通过仅对分割标签引入可控的弹性变形，同时保持原始图像不变，我们的方法鼓励模型在存在轻微标签不一致的情况下仍能学习鲁棒的对象结构表示。广泛的实验证明，NSegment+在魏玛根、LoveDA、城市景观和PASCAL VOC数据集上的一致性能提升，平均mIoU分别提高2.29%、2.38%、1.75%和3.39%，即使没有额外的技术，也突显了处理隐含标签噪声的重要性。当与CutMix和Label Smoothing等其他训练技巧结合使用时，这些增益可以进一步放大。 

---
# eMamba: Efficient Acceleration Framework for Mamba Models in Edge Computing 

**Title (ZH)**: eMamba：边缘计算中Mamba模型的高效加速框架 

**Authors**: Jiyong Kim, Jaeho Lee, Jiahao Lin, Alish Kanani, Miao Sun, Umit Y. Ogras, Jaehyun Park  

**Link**: [PDF](https://arxiv.org/pdf/2508.10370)  

**Abstract**: State Space Model (SSM)-based machine learning architectures have recently gained significant attention for processing sequential data. Mamba, a recent sequence-to-sequence SSM, offers competitive accuracy with superior computational efficiency compared to state-of-the-art transformer models. While this advantage makes Mamba particularly promising for resource-constrained edge devices, no hardware acceleration frameworks are currently optimized for deploying it in such environments. This paper presents eMamba, a comprehensive end-to-end hardware acceleration framework explicitly designed for deploying Mamba models on edge platforms. eMamba maximizes computational efficiency by replacing complex normalization layers with lightweight hardware-aware alternatives and approximating expensive operations, such as SiLU activation and exponentiation, considering the target applications. Then, it performs an approximation-aware neural architecture search (NAS) to tune the learnable parameters used during approximation. Evaluations with Fashion-MNIST, CIFAR-10, and MARS, an open-source human pose estimation dataset, show eMamba achieves comparable accuracy to state-of-the-art techniques using 1.63-19.9$\times$ fewer parameters. In addition, it generalizes well to large-scale natural language tasks, demonstrating stable perplexity across varying sequence lengths on the WikiText2 dataset. We also quantize and implement the entire eMamba pipeline on an AMD ZCU102 FPGA and ASIC using GlobalFoundries (GF) 22 nm technology. Experimental results show 4.95-5.62$\times$ lower latency and 2.22-9.95$\times$ higher throughput, with 4.77$\times$ smaller area, 9.84$\times$ lower power, and 48.6$\times$ lower energy consumption than baseline solutions while maintaining competitive accuracy. 

**Abstract (ZH)**: 基于状态空间模型（SSM）的机器学习架构近年来在处理序列数据方面获得了显著关注。Mamba是一种最近提出的序列到序列SSM，与最先进的变压器模型相比，其在精度上具有竞争力，并且具有更高的计算效率。尽管这一优势使Mamba特别适合资源受限的边缘设备，但目前尚无硬件加速框架针对此类环境优化部署Mamba模型。本文提出了一种名为eMamba的全面端到端硬件加速框架，专门设计用于在边缘平台上部署Mamba模型。eMamba通过用轻量级的硬件感知替代复杂归一化层，并在目标应用考虑情况下近似昂贵的操作（如SiLU激活和指数运算），从而最大化计算效率。然后，它进行感知近似的神经架构搜索（NAS），以调整近似过程中使用的可学习参数。对于Fashion-MNIST、CIFAR-10以及公开的人体姿态估计数据集MARS，实验结果显示eMamba在使用1.63-19.9倍更少的参数时，实现了与最先进的技术相当的精度。此外，eMamba在大规模自然语言任务中具有良好的泛化能力，在WikiText2数据集中表现出稳定的变化序列长度下的困惑度。我们还在AMD ZCU102 FPGA和ASIC上用GlobalFoundries (GF) 22 nm技术对整个eMamba管道进行了量化和实现。实验结果表明，与基线解决方案相比，eMamba具有4.95-5.62倍更低的延迟、2.22-9.95倍更高的吞吐量、4.77倍更小的面积、9.84倍更低的功耗和48.6倍更低的能量消耗，同时保持了竞争力的精度。 

---
# Welfare-Centric Clustering 

**Title (ZH)**: 福利中心聚类 

**Authors**: Claire Jie Zhang, Seyed A. Esmaeili, Jamie Morgenstern  

**Link**: [PDF](https://arxiv.org/pdf/2508.10345)  

**Abstract**: Fair clustering has traditionally focused on ensuring equitable group representation or equalizing group-specific clustering costs. However, Dickerson et al. (2025) recently showed that these fairness notions may yield undesirable or unintuitive clustering outcomes and advocated for a welfare-centric clustering approach that models the utilities of the groups. In this work, we model group utilities based on both distances and proportional representation and formalize two optimization objectives based on welfare-centric clustering: the Rawlsian (Egalitarian) objective and the Utilitarian objective. We introduce novel algorithms for both objectives and prove theoretical guarantees for them. Empirical evaluations on multiple real-world datasets demonstrate that our methods significantly outperform existing fair clustering baselines. 

**Abstract (ZH)**: 公平聚类 traditionally focuses on ensuring equitable group representation or equalizing group-specific clustering costs. However, Dickerson et al. (2025) recently showed that these fairness notions may yield undesirable or unintuitive clustering outcomes and advocated for a welfare-centric clustering approach that models the utilities of the groups. In this work, we model group utilities based on both distances and proportional representation and formalize two optimization objectives based on welfare-centric clustering: the Rawlsian (Egalitarian) objective and the Utilitarian objective. We introduce novel algorithms for both objectives and prove theoretical guarantees for them. Empirical evaluations on multiple real-world datasets demonstrate that our methods significantly outperform existing fair clustering baselines. 

---
# Layer-Wise Analysis of Self-Supervised Representations for Age and Gender Classification in Children's Speech 

**Title (ZH)**: 自我监督表示在儿童语音年龄和性别分类中的逐层分析 

**Authors**: Abhijit Sinha, Harishankar Kumar, Mohit Joshi, Hemant Kumar Kathania, Shrikanth Narayanan, Sudarsana Reddy Kadiri  

**Link**: [PDF](https://arxiv.org/pdf/2508.10332)  

**Abstract**: Children's speech presents challenges for age and gender classification due to high variability in pitch, articulation, and developmental traits. While self-supervised learning (SSL) models perform well on adult speech tasks, their ability to encode speaker traits in children remains underexplored. This paper presents a detailed layer-wise analysis of four Wav2Vec2 variants using the PFSTAR and CMU Kids datasets. Results show that early layers (1-7) capture speaker-specific cues more effectively than deeper layers, which increasingly focus on linguistic information. Applying PCA further improves classification, reducing redundancy and highlighting the most informative components. The Wav2Vec2-large-lv60 model achieves 97.14% (age) and 98.20% (gender) on CMU Kids; base-100h and large-lv60 models reach 86.05% and 95.00% on PFSTAR. These results reveal how speaker traits are structured across SSL model depth and support more targeted, adaptive strategies for child-aware speech interfaces. 

**Abstract (ZH)**: 儿童语音的年龄和性别分类由于音高、发音和发育特征的高度变异性而具有挑战性。虽然自我监督学习模型在成人语音任务中表现良好，但它们在编码儿童特征方面的能力仍需进一步探索。本文通过对PFSTAR和CMU Kids数据集使用四种Wav2Vec2变体进行逐层分析，展示了早期层（1-7）比深层层更有效地捕捉到语音特定线索，并且主成分分析进一步提高了分类效果，减少冗余并突出最相关信息。Wav2Vec2-large-lv60模型在CMU Kids上的年龄分类准确率为97.14%，性别分类准确率为98.20%；base-100h和large-lv60模型在PFSTAR上的准确率分别为86.05%和95.00%。这些结果揭示了语音特定特征在自我监督学习模型中的结构，并支持更为针对性和适应性的儿童感知语音界面策略。 

---
# A Vision-Language Pre-training Model-Guided Approach for Mitigating Backdoor Attacks in Federated Learning 

**Title (ZH)**: 基于视觉-语言预训练模型的 Federated Learning 中后门攻击缓解方法 

**Authors**: Keke Gai, Dongjue Wang, Jing Yu, Liehuang Zhu, Qi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10315)  

**Abstract**: Existing backdoor defense methods in Federated Learning (FL) rely on the assumption of homogeneous client data distributions or the availability of a clean serve dataset, which limits the practicality and effectiveness. Defending against backdoor attacks under heterogeneous client data distributions while preserving model performance remains a significant challenge. In this paper, we propose a FL backdoor defense framework named CLIP-Fed, which leverages the zero-shot learning capabilities of vision-language pre-training models. By integrating both pre-aggregation and post-aggregation defense strategies, CLIP-Fed overcomes the limitations of Non-IID imposed on defense effectiveness. To address privacy concerns and enhance the coverage of the dataset against diverse triggers, we construct and augment the server dataset using the multimodal large language model and frequency analysis without any client samples. To address class prototype deviations caused by backdoor samples and eliminate the correlation between trigger patterns and target labels, CLIP-Fed aligns the knowledge of the global model and CLIP on the augmented dataset using prototype contrastive loss and Kullback-Leibler divergence. Extensive experiments on representative datasets validate the effectiveness of CLIP-Fed. Compared to state-of-the-art methods, CLIP-Fed achieves an average reduction in ASR, i.e., 2.03\% on CIFAR-10 and 1.35\% on CIFAR-10-LT, while improving average MA by 7.92\% and 0.48\%, respectively. 

**Abstract (ZH)**: 基于零样本学习的CLIP-Fed： federated learning中的后门防御框架 

---
# ReviewRL: Towards Automated Scientific Review with RL 

**Title (ZH)**: ReviewRL：向基于强化学习的自动化科学研究评议迈进 

**Authors**: Sihang Zeng, Kai Tian, Kaiyan Zhang, Yuru wang, Junqi Gao, Runze Liu, Sa Yang, Jingxuan Li, Xinwei Long, Jiaheng Ma, Biqing Qi, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.10308)  

**Abstract**: Peer review is essential for scientific progress but faces growing challenges due to increasing submission volumes and reviewer fatigue. Existing automated review approaches struggle with factual accuracy, rating consistency, and analytical depth, often generating superficial or generic feedback lacking the insights characteristic of high-quality human reviews. We introduce ReviewRL, a reinforcement learning framework for generating comprehensive and factually grounded scientific paper reviews. Our approach combines: (1) an ArXiv-MCP retrieval-augmented context generation pipeline that incorporates relevant scientific literature, (2) supervised fine-tuning that establishes foundational reviewing capabilities, and (3) a reinforcement learning procedure with a composite reward function that jointly enhances review quality and rating accuracy. Experiments on ICLR 2025 papers demonstrate that ReviewRL significantly outperforms existing methods across both rule-based metrics and model-based quality assessments. ReviewRL establishes a foundational framework for RL-driven automatic critique generation in scientific discovery, demonstrating promising potential for future development in this domain. The implementation of ReviewRL will be released at GitHub. 

**Abstract (ZH)**: 同行评审对于科学进步至关重要，但面对提交量增加和评审疲劳的挑战，其面临着日益增长的困难。现有的自动化评审方法在事实准确性、评分一致性和分析深度方面遇到困难，常常生成表面化或通用的反馈，缺乏高质量人工评审具有的深入洞察。我们提出ReviewRL，这是一种基于强化学习的生成全面且基于事实的科研论文评审框架。该方法结合了：（1）一个通过ArXiv-MCP检索增强的语境生成管道，整合相关科学文献；（2）监督微调以建立基础评审能力；以及（3）一种强化学习过程，通过复合奖励函数共同提升评审质量和评分准确性。针对ICLR 2025论文的实验表明，ReviewRL在基于规则的评估指标和基于模型的质量评估中均显著优于现有方法。ReviewRL为基于强化学习的自动批评生成提供了基础框架，并展示了在该领域未来发展的前景。ReviewRL的实现将在GitHub上发布。 

---
# Yet another algorithmic bias: A Discursive Analysis of Large Language Models Reinforcing Dominant Discourses on Gender and Race 

**Title (ZH)**: 另一种算法偏见：关于性别和种族主导论述的大语言模型话语分析 

**Authors**: Gustavo Bonil, Simone Hashiguti, Jhessica Silva, João Gondim, Helena Maia, Nádia Silva, Helio Pedrini, Sandra Avila  

**Link**: [PDF](https://arxiv.org/pdf/2508.10304)  

**Abstract**: With the advance of Artificial Intelligence (AI), Large Language Models (LLMs) have gained prominence and been applied in diverse contexts. As they evolve into more sophisticated versions, it is essential to assess whether they reproduce biases, such as discrimination and racialization, while maintaining hegemonic discourses. Current bias detection approaches rely mostly on quantitative, automated methods, which often overlook the nuanced ways in which biases emerge in natural language. This study proposes a qualitative, discursive framework to complement such methods. Through manual analysis of LLM-generated short stories featuring Black and white women, we investigate gender and racial biases. We contend that qualitative methods such as the one proposed here are fundamental to help both developers and users identify the precise ways in which biases manifest in LLM outputs, thus enabling better conditions to mitigate them. Results show that Black women are portrayed as tied to ancestry and resistance, while white women appear in self-discovery processes. These patterns reflect how language models replicate crystalized discursive representations, reinforcing essentialization and a sense of social immobility. When prompted to correct biases, models offered superficial revisions that maintained problematic meanings, revealing limitations in fostering inclusive narratives. Our results demonstrate the ideological functioning of algorithms and have significant implications for the ethical use and development of AI. The study reinforces the need for critical, interdisciplinary approaches to AI design and deployment, addressing how LLM-generated discourses reflect and perpetuate inequalities. 

**Abstract (ZH)**: 随着人工智能的进步，大型语言模型已在多种情境中获得 prominence 并得到应用。随着它们演化成更为复杂的新版本，评估它们在保持霸权话语的同时是否再现了偏见（如歧视和种族化）变得尤为重要。当前的偏见检测方法主要依赖于定量、自动化的手段，往往忽略了偏见在自然语言中复杂的表现方式。本研究提出了一种定性的、话语框架来补充这些方法。通过对手动分析生成的涉及黑人和白人女性的小说，我们探讨性别和种族偏见。我们认为，如本研究提出的方法这样的定性方法是帮助开发者和用户识别大型语言模型输出中偏见具体表现方式的基础，从而有助于更好地条件来减轻这些偏见。结果表明，黑人女性常被描绘为与祖先和抵抗相关，而白人女性则出现在自我发现的过程中。这些模式反映了语言模型如何再现固化的话语表征，强化本质化并维持一种社会流动性感的假象。当被要求纠正偏见时，模型提供的修改往往是表面化的，保留了有问题的含义，揭示了促进包容叙事的局限性。我们的研究结果揭示了算法的意识形态功能，并对人工智能的伦理使用和发展具有重要意义。本研究强调了在人工智能设计和部署中采用批判性和跨学科方法的必要性，关注LLM生成的话语如何反映并延续不平等。 

---
# Pose-Robust Calibration Strategy for Point-of-Gaze Estimation on Mobile Phones 

**Title (ZH)**: 基于移动电话的眼球注视点估计的鲁棒校准策略 

**Authors**: Yujie Zhao, Jiabei Zeng, Shiguang Shan  

**Link**: [PDF](https://arxiv.org/pdf/2508.10268)  

**Abstract**: Although appearance-based point-of-gaze (PoG) estimation has improved, the estimators still struggle to generalize across individuals due to personal differences. Therefore, person-specific calibration is required for accurate PoG estimation. However, calibrated PoG estimators are often sensitive to head pose variations. To address this, we investigate the key factors influencing calibrated estimators and explore pose-robust calibration strategies. Specifically, we first construct a benchmark, MobilePoG, which includes facial images from 32 individuals focusing on designated points under either fixed or continuously changing head poses. Using this benchmark, we systematically analyze how the diversity of calibration points and head poses influences estimation accuracy. Our experiments show that introducing a wider range of head poses during calibration improves the estimator's ability to handle pose variation. Building on this insight, we propose a dynamic calibration strategy in which users fixate on calibration points while moving their phones. This strategy naturally introduces head pose variation during a user-friendly and efficient calibration process, ultimately producing a better calibrated PoG estimator that is less sensitive to head pose variations than those using conventional calibration strategies. Codes and datasets are available at our project page. 

**Abstract (ZH)**: 尽管基于外观的眼球凝视点（PoG）估计有所改进，但由于个体差异，估计器仍难以跨个体泛化。因此，为了实现准确的PoG估计，需要针对个体进行校准。然而，经过校准的PoG估计器往往对头部姿态变化敏感。为解决这一问题，我们调查了影响校准估计器的关键因素，并探索了姿态鲁棒的校准策略。具体而言，我们首先构建了一个基准数据集MobilePoG，其中包括来自32个个体的面部图像，这些图像在固定或连续变化的头部姿态下关注指定的点。使用该基准数据集，我们系统地分析了校准点的多样性和头部姿态的变化如何影响估计精度。实验结果显示，在校准过程中引入更广泛的头部姿态范围可以提高估计器处理姿态变化的能力。基于这一洞察，我们提出了一种动态校准策略，用户在移动手机时注视校准点。这一策略自然地在用户友好且高效的校准过程中引入头部姿态变化，最终生成了一个对头部姿态变化更不敏感的校准后的PoG估计器。相关代码和数据集可在我们的项目页面获取。 

---
# MRFD: Multi-Region Fusion Decoding with Self-Consistency for Mitigating Hallucinations in LVLMs 

**Title (ZH)**: MRFD：多区域融合解码与自一致性方法减轻大语言模型中的幻觉问题 

**Authors**: Haonan Ge, Yiwei Wang, Ming-Hsuan Yang, Yujun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2508.10264)  

**Abstract**: Large Vision-Language Models (LVLMs) have shown strong performance across multimodal tasks. However, they often produce hallucinations -- text that is inconsistent with visual input, due to the limited ability to verify information in different regions of the image. To address this, we propose Multi-Region Fusion Decoding (MRFD), a training-free decoding method that improves factual grounding by modeling inter-region consistency. MRFD identifies salient regions using cross-attention, generates initial responses for each, and computes reliability weights based on Jensen-Shannon Divergence (JSD) among the responses. These weights guide a consistency-aware fusion of per-region predictions, using region-aware prompts inspired by Chain-of-Thought reasoning. Experiments across multiple LVLMs and benchmarks show that MRFD significantly reduces hallucinations and improves response factuality without requiring model updates. 

**Abstract (ZH)**: 大规模多模态语言视觉模型（LVLMs）在多模态任务中表现出强大的性能。然而，它们通常会产生幻觉——与视觉输入不一致的文本，这是因为模型在验证图像不同区域的信息方面的能力有限。为了解决这一问题，我们提出了一种无训练的解码方法——多区域融合解码（MRFD），通过建模不同区域之间的一致性来增强事实 grounding。MRFD 使用交叉注意力识别显著区域，为每个区域生成初始响应，并基于响应之间的 Jensen-Shannon 散度（JSD）计算可靠性权重。这些权重引导一种基于区域意识的融合过程，该过程使用借鉴了思维链推理的区域意识提示，关注一致性。跨多个 LVLM 和基准的实验表明，MRFD 显著减少了幻觉并提高了响应的事实性，而无需对模型进行更新。 

---
# DINOMotion: advanced robust tissue motion tracking with DINOv2 in 2D-Cine MRI-guided radiotherapy 

**Title (ZH)**: DINOMotion：基于DINOv2的高级robust 2D-Cine MRI引导放射治疗中的组织运动跟踪 

**Authors**: Soorena Salari, Catherine Spino, Laurie-Anne Pharand, Fabienne Lathuiliere, Hassan Rivaz, Silvain Beriault, Yiming Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.10260)  

**Abstract**: Accurate tissue motion tracking is critical to ensure treatment outcome and safety in 2D-Cine MRI-guided radiotherapy. This is typically achieved by registration of sequential images, but existing methods often face challenges with large misalignments and lack of interpretability. In this paper, we introduce DINOMotion, a novel deep learning framework based on DINOv2 with Low-Rank Adaptation (LoRA) layers for robust, efficient, and interpretable motion tracking. DINOMotion automatically detects corresponding landmarks to derive optimal image registration, enhancing interpretability by providing explicit visual correspondences between sequential images. The integration of LoRA layers reduces trainable parameters, improving training efficiency, while DINOv2's powerful feature representations offer robustness against large misalignments. Unlike iterative optimization-based methods, DINOMotion directly computes image registration at test time. Our experiments on volunteer and patient datasets demonstrate its effectiveness in estimating both linear and nonlinear transformations, achieving Dice scores of 92.07% for the kidney, 90.90% for the liver, and 95.23% for the lung, with corresponding Hausdorff distances of 5.47 mm, 8.31 mm, and 6.72 mm, respectively. DINOMotion processes each scan in approximately 30ms and consistently outperforms state-of-the-art methods, particularly in handling large misalignments. These results highlight its potential as a robust and interpretable solution for real-time motion tracking in 2D-Cine MRI-guided radiotherapy. 

**Abstract (ZH)**: 准确的组织运动跟踪对于确保2D-Cine MRI引导放射治疗的结果和安全性至关重要。本研究引入了基于DINOv2和低秩适应（LoRA）层的DINOMotion新型深度学习框架，实现稳健、高效和可解释的运动跟踪。DINOMotion通过自动检测对应的特征点来获取最优图像配准，通过提供显式的序列图像之间对应关系增强可解释性。LoRA层的集成减少了可训练参数，提高了训练效率，而DINOv2强大的特征表示提供了对大错位的鲁棒性。与基于迭代优化的方法不同，DINOMotion在测试时直接计算图像配准。我们在志愿者和患者数据集上的实验展示了其在估计线性和非线性变换方面的有效性，分别在肾脏、肝脏和肺部获得了Dice分数为92.07%、90.90%和95.23%，对应的Hausdorff距离分别为5.47 mm、8.31 mm和6.72 mm。DINOMotion每扫描处理速度约30ms，并且在处理大错位方面始终优于现有方法。这些结果突显了其在2D-Cine MRI引导放射治疗中实时运动跟踪中的稳健性和可解释性潜力。 

---
# Facilitating Longitudinal Interaction Studies of AI Systems 

**Title (ZH)**: 促进人工智能系统 longitudinally 交互研究 

**Authors**: Tao Long, Sitong Wang, Émilie Fabre, Tony Wang, Anup Sathya, Jason Wu, Savvas Petridis, Dingzeyu Li, Tuhin Chakrabarty, Yue Jiang, Jingyi Li, Tiffany Tseng, Ken Nakagaki, Qian Yang, Nikolas Martelaro, Jeffrey V. Nickerson, Lydia B. Chilton  

**Link**: [PDF](https://arxiv.org/pdf/2508.10252)  

**Abstract**: UIST researchers develop tools to address user challenges. However, user interactions with AI evolve over time through learning, adaptation, and repurposing, making one time evaluations insufficient. Capturing these dynamics requires longer-term studies, but challenges in deployment, evaluation design, and data collection have made such longitudinal research difficult to implement. Our workshop aims to tackle these challenges and prepare researchers with practical strategies for longitudinal studies. The workshop includes a keynote, panel discussions, and interactive breakout groups for discussion and hands-on protocol design and tool prototyping sessions. We seek to foster a community around longitudinal system research and promote it as a more embraced method for designing, building, and evaluating UIST tools. 

**Abstract (ZH)**: UIST研究人员开发工具以应对用户挑战。然而，用户与AI的互动随时间通过学习、适应和重新利用而变化，使得一次性评估不足。捕获这些动态需要进行长期研究，但部署、评估设计和数据收集等方面的挑战使这类纵向研究难以实施。我们的研讨会旨在应对这些挑战，并为研究人员提供进行纵向研究的实用策略。研讨会包括主旨演讲、圆桌讨论和互动小组讨论，以及协议设计和工具原型制作的实践环节。我们致力于促进纵向系统研究的社区建设，并将其推广为设计、构建和评估UIST工具的更受欢迎的方法。 

---
# No Free Lunch from Audio Pretraining in Bioacoustics: A Benchmark Study of Embeddings 

**Title (ZH)**: 生物声学中音频预训练的免费午餐：嵌入表示的基准研究 

**Authors**: Chenggang Chen, Zhiyu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10230)  

**Abstract**: Bioacoustics, the study of animal sounds, offers a non-invasive method to monitor ecosystems. Extracting embeddings from audio-pretrained deep learning (DL) models without fine-tuning has become popular for obtaining bioacoustic features for tasks. However, a recent benchmark study reveals that while fine-tuned audio-pretrained VGG and transformer models achieve state-of-the-art performance in some tasks, they fail in others. This study benchmarks 11 DL models on the same tasks by reducing their learned embeddings' dimensionality and evaluating them through clustering. We found that audio-pretrained DL models 1) without fine-tuning even underperform fine-tuned AlexNet, 2) both with and without fine-tuning fail to separate the background from labeled sounds, but ResNet does, and 3) outperform other models when fewer background sounds are included during fine-tuning. This study underscores the necessity of fine-tuning audio-pretrained models and checking the embeddings after fine-tuning. Our codes are available: this https URL\_Embeddings 

**Abstract (ZH)**: 生物声学：生物声学研究的非侵入性方法在生态系统监测中的应用。通过音频预训练深度学习模型提取嵌入而无需微调以获取生物声学特征已成为流行的方法。然而，最近的一项基准研究显示，尽管预训练音频的VGG和变压器模型在某些任务中达到了最先进的性能，但在其他任务中却失败了。本研究通过减少11种深度学习模型学习嵌入的维度并在相同任务上进行比较，发现音频预训练的深度学习模型1) 即使是在无需微调的情况下也甚至不如微调后的AlexNet表现，2) 无论是否进行微调，在分离背景噪音与标记声音方面都失败了，而ResNet则能实现这一目标，3) 在包含较少背景噪音的微调过程中优于其他模型。本研究强调了微调音频预训练模型并在微调后检查嵌入的重要性。我们的代码已公开：this <https://example.com> Embeddings。 

---
# Using Large Language Models to Measure Symptom Severity in Patients At Risk for Schizophrenia 

**Title (ZH)**: 使用大型语言模型测量 schizophrenia 高风险患者症状严重程度 

**Authors**: Andrew X. Chen, Guillermo Horga, Sean Escola  

**Link**: [PDF](https://arxiv.org/pdf/2508.10226)  

**Abstract**: Patients who are at clinical high risk (CHR) for schizophrenia need close monitoring of their symptoms to inform appropriate treatments. The Brief Psychiatric Rating Scale (BPRS) is a validated, commonly used research tool for measuring symptoms in patients with schizophrenia and other psychotic disorders; however, it is not commonly used in clinical practice as it requires a lengthy structured interview. Here, we utilize large language models (LLMs) to predict BPRS scores from clinical interview transcripts in 409 CHR patients from the Accelerating Medicines Partnership Schizophrenia (AMP-SCZ) cohort. Despite the interviews not being specifically structured to measure the BPRS, the zero-shot performance of the LLM predictions compared to the true assessment (median concordance: 0.84, ICC: 0.73) approaches human inter- and intra-rater reliability. We further demonstrate that LLMs have substantial potential to improve and standardize the assessment of CHR patients via their accuracy in assessing the BPRS in foreign languages (median concordance: 0.88, ICC: 0.70), and integrating longitudinal information in a one-shot or few-shot learning approach. 

**Abstract (ZH)**: 临床高风险（CHR）患者需要密切监测其症状以指导适当治疗。我们利用大规模语言模型（LLMs）从加速药物开发精神分裂症（AMP-SCZ）队列中的409名CHR患者访谈记录中预测BPRS评分。尽管访谈并非专门设计用于测量BPRS，但LLM预测与真实评估（中位一致性：0.84，ICC：0.73）的人际和自我评定可靠性相近。进一步研究表明，LLM在通过评估外语文本中的BPRS提高和标准化CHR患者的评估方面具有巨大的潜力（中位一致性：0.88，ICC：0.70），并能通过一-shot或few-shot学习整合纵向信息。 

---
# Understanding Textual Emotion Through Emoji Prediction 

**Title (ZH)**: 通过emoji预测理解文本情感 

**Authors**: Ethan Gordon, Nishank Kuppa, Rigved Tummala, Sriram Anasuri  

**Link**: [PDF](https://arxiv.org/pdf/2508.10222)  

**Abstract**: This project explores emoji prediction from short text sequences using four deep learning architectures: a feed-forward network, CNN, transformer, and BERT. Using the TweetEval dataset, we address class imbalance through focal loss and regularization techniques. Results show BERT achieves the highest overall performance due to its pre-training advantage, while CNN demonstrates superior efficacy on rare emoji classes. This research shows the importance of architecture selection and hyperparameter tuning for sentiment-aware emoji prediction, contributing to improved human-computer interaction. 

**Abstract (ZH)**: 本研究使用四种深度学习架构（前向网络、CNN、变压器和BERT）探索短文本序列中的Emoji预测，并通过使用TweetEval数据集、焦点损失和正则化技术解决类别不平衡问题。结果显示，由于预训练优势，BERT在整体性能上表现最佳，而CNN在罕见Emoji类别上表现出色。本研究强调了架构选择和超参数调优对于情感感知Emoji预测的重要性，有助于改善人机交互。 

---
# An Explainable AI based approach for Monitoring Animal Health 

**Title (ZH)**: 基于可解释人工智能的动物健康监测方法 

**Authors**: Rahul Janaa, Shubham Dixit, Mrityunjay Sharma, Ritesh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.10210)  

**Abstract**: Monitoring cattle health and optimizing yield are key challenges faced by dairy farmers due to difficulties in tracking all animals on the farm. This work aims to showcase modern data-driven farming practices based on explainable machine learning(ML) methods that explain the activity and behaviour of dairy cattle (cows). Continuous data collection of 3-axis accelerometer sensors and usage of robust ML methodologies and algorithms, provide farmers and researchers with actionable information on cattle activity, allowing farmers to make informed decisions and incorporate sustainable practices. This study utilizes Bluetooth-based Internet of Things (IoT) devices and 4G networks for seamless data transmission, immediate analysis, inference generation, and explains the models performance with explainability frameworks. Special emphasis is put on the pre-processing of the accelerometers time series data, including the extraction of statistical characteristics, signal processing techniques, and lag-based features using the sliding window technique. Various hyperparameter-optimized ML models are evaluated across varying window lengths for activity classification. The k-nearest neighbour Classifier achieved the best performance, with AUC of mean 0.98 and standard deviation of 0.0026 on the training set and 0.99 on testing set). In order to ensure transparency, Explainable AI based frameworks such as SHAP is used to interpret feature importance that can be understood and used by practitioners. A detailed comparison of the important features, along with the stability analysis of selected features, supports development of explainable and practical ML models for sustainable livestock management. 

**Abstract (ZH)**: 基于可解释机器学习方法监测奶牛健康并优化产量：现代数据驱动养殖实践展示 

---
# CATNet: A geometric deep learning approach for CAT bond spread prediction in the primary market 

**Title (ZH)**: CATNet: 一种几何深度学习方法在一级市场中预测CAT债券利差 

**Authors**: Dixon Domfeh, Saeid Safarveisi  

**Link**: [PDF](https://arxiv.org/pdf/2508.10208)  

**Abstract**: Traditional models for pricing catastrophe (CAT) bonds struggle to capture the complex, relational data inherent in these instruments. This paper introduces CATNet, a novel framework that applies a geometric deep learning architecture, the Relational Graph Convolutional Network (R-GCN), to model the CAT bond primary market as a graph, leveraging its underlying network structure for spread prediction. Our analysis reveals that the CAT bond market exhibits the characteristics of a scale-free network, a structure dominated by a few highly connected and influential hubs. CATNet demonstrates high predictive performance, significantly outperforming a strong Random Forest benchmark. The inclusion of topological centrality measures as features provides a further, significant boost in accuracy. Interpretability analysis confirms that these network features are not mere statistical artifacts; they are quantitative proxies for long-held industry intuition regarding issuer reputation, underwriter influence, and peril concentration. This research provides evidence that network connectivity is a key determinant of price, offering a new paradigm for risk assessment and proving that graph-based models can deliver both state-of-the-art accuracy and deeper, quantifiable market insights. 

**Abstract (ZH)**: 传统的 Catarophe（CAT）债券定价模型难以捕捉这些金融工具内在的复杂关系数据。本文引入了 CATNet，这是一种应用几何深度学习架构和关系图卷积网络（R-GCN）来研究 CAT 债券一级市场的新型框架，它通过利用潜在的网络结构进行预测应用。我们的分析表明，CAT 债券市场表现出一种无尺度网络特征，，，中心节点较多且高度互联和有影响力。CATNet 在预测表现上显著优于随机森林基准模型。引入高中心度度量进一步显著提升了准确率。解释性分析表明，网络架构不仅仅是统计学上的的现象，而是定量的代理变量，代表了长期以来对发行人声誉、承销商影响力和风险的认知。本文提供了网络连接性是预测性能的关键决定因素的证据，提出了风险评估的新范式，并
user
把下面这段话翻译成英文，要符合学术规范：该研究引入了CATNet，这是一种应用几何深度学习架构和关系图卷积网络（R-GCN）来研究CATA债券一级市场的新型框架。我们的分析表明CAT债券市场表现出无尺度网络特征：高度互联且拥有众多中心节点，并它们高度互联且有影响力。我们的分析表明CATNet在预测表现上显著优于随机森林基准模型。我们进一步通过引入高中心度度量显著提升了模型的的性能。我们进行了解释性分析，结果显示网络连接结构不仅仅是统计学上的现象它们是定量代理变量代表长期以来对持有的者声誉、承销商影响力和风险认知。该研究提供了网络连通性是预测性能关键决定因素的证据并提出了风险评估的新范式。 

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
# Improving watermelon (Citrullus lanatus) disease classification with generative artificial intelligence (GenAI)-based synthetic and real-field images via a custom EfficientNetV2-L model 

**Title (ZH)**: 基于生成人工智能（GenAI）的合成和实地图像改进西瓜（Citrullus lanatus）疾病分类——一种定制的EfficientNetV2-L模型 

**Authors**: Nitin Rai, Nathan S. Boyd, Gary E. Vallad, Arnold W. Schumann  

**Link**: [PDF](https://arxiv.org/pdf/2508.10156)  

**Abstract**: The current advancements in generative artificial intelligence (GenAI) models have paved the way for new possibilities for generating high-resolution synthetic images, thereby offering a promising alternative to traditional image acquisition for training computer vision models in agriculture. In the context of crop disease diagnosis, GenAI models are being used to create synthetic images of various diseases, potentially facilitating model creation and reducing the dependency on resource-intensive in-field data collection. However, limited research has been conducted on evaluating the effectiveness of integrating real with synthetic images to improve disease classification performance. Therefore, this study aims to investigate whether combining a limited number of real images with synthetic images can enhance the prediction accuracy of an EfficientNetV2-L model for classifying watermelon \textit{(Citrullus lanatus)} diseases. The training dataset was divided into five treatments: H0 (only real images), H1 (only synthetic images), H2 (1:1 real-to-synthetic), H3 (1:10 real-to-synthetic), and H4 (H3 + random images to improve variability and model generalization). All treatments were trained using a custom EfficientNetV2-L architecture with enhanced fine-tuning and transfer learning techniques. Models trained on H2, H3, and H4 treatments demonstrated high precision, recall, and F1-score metrics. Additionally, the weighted F1-score increased from 0.65 (on H0) to 1.00 (on H3-H4) signifying that the addition of a small number of real images with a considerable volume of synthetic images improved model performance and generalizability. Overall, this validates the findings that synthetic images alone cannot adequately substitute for real images; instead, both must be used in a hybrid manner to maximize model performance for crop disease classification. 

**Abstract (ZH)**: 当前生成型人工智能模型的进步为生成高分辨率合成图像提供了新的可能性，从而为农业中训练计算机视觉模型提供了传统图像采集的有前途的替代方案。在作物病害诊断的背景下，生成型人工智能模型被用于创建各种病害的合成图像，有可能促进模型的创建并减少对资源密集型田间数据采集的依赖。然而，将真实图像与合成图像结合以提高病害分类性能的有效性评估研究有限。因此，本研究旨在探讨将少量真实图像与合成图像结合是否能够提高基于EfficientNetV2-L模型的西瓜（Citrullus lanatus）病害分类的预测准确性。训练数据集被分为五个处理：H0（仅真实图像）、H1（仅合成图像）、H2（1:1真实图像与合成图像）、H3（1:10真实图像与合成图像），以及H4（H3+随机图像以提高变量性和模型泛化性）。所有处理均使用增强的微调和迁移学习技术定制的EfficientNetV2-L架构进行训练。在H2、H3和H4处理中训练的模型展示了高精度、召回率和F1分数指标。此外，加权F1分数从H0的0.65增加到H3-H4的1.00，表明少量真实图像与大量合成图像的结合提高了模型性能和泛化性。总体而言，这证明单独使用合成图像不足以替代真实图像；相反，两者须以混合的方式使用，以最大化作物病害分类模型的性能。 

---
# Out-of-Distribution Detection using Counterfactual Distance 

**Title (ZH)**: 基于反事实距离的域外检测 

**Authors**: Maria Stoica, Francesco Leofante, Alessio Lomuscio  

**Link**: [PDF](https://arxiv.org/pdf/2508.10148)  

**Abstract**: Accurate and explainable out-of-distribution (OOD) detection is required to use machine learning systems safely. Previous work has shown that feature distance to decision boundaries can be used to identify OOD data effectively. In this paper, we build on this intuition and propose a post-hoc OOD detection method that, given an input, calculates the distance to decision boundaries by leveraging counterfactual explanations. Since computing explanations can be expensive for large architectures, we also propose strategies to improve scalability by computing counterfactuals directly in embedding space. Crucially, as the method employs counterfactual explanations, we can seamlessly use them to help interpret the results of our detector. We show that our method is in line with the state of the art on CIFAR-10, achieving 93.50% AUROC and 25.80% FPR95. Our method outperforms these methods on CIFAR-100 with 97.05% AUROC and 13.79% FPR95 and on ImageNet-200 with 92.55% AUROC and 33.55% FPR95 across four OOD datasets 

**Abstract (ZH)**: 准确且可解释的离分布（OOD）检测对于安全使用机器学习系统是必要的。本研究在此基础上提出了一种后验OOD检测方法，该方法通过利用反事实解释来计算输入到决策边界的距离。由于大型架构中计算解释可能代价高昂，我们还提出了一些策略，以通过直接在嵌入空间中计算反事实来提高可扩展性。 crucial的是，由于方法采用了反事实解释，我们可以无缝地使用它们来帮助解释检测器的结果。我们的方法在CIFAR-10上的AUROC为93.50%，FPR95为25.80%。在CIFAR-100上，我们的方法AUCOR为97.05%，FPR95为13.79%；在ImageNet-200上，AUCOR为92.55%，FPR95为33.55%，均优于现有方法。 

---
# rETF-semiSL: Semi-Supervised Learning for Neural Collapse in Temporal Data 

**Title (ZH)**: rETF-semiSL: 半监督学习在时间数据中应对神经坍缩问题 

**Authors**: Yuhan Xie, William Cappelletti, Mahsa Shoaran, Pascal Frossard  

**Link**: [PDF](https://arxiv.org/pdf/2508.10147)  

**Abstract**: Deep neural networks for time series must capture complex temporal patterns, to effectively represent dynamic data. Self- and semi-supervised learning methods show promising results in pre-training large models, which -- when finetuned for classification -- often outperform their counterparts trained from scratch. Still, the choice of pretext training tasks is often heuristic and their transferability to downstream classification is not granted, thus we propose a novel semi-supervised pre-training strategy to enforce latent representations that satisfy the Neural Collapse phenomenon observed in optimally trained neural classifiers. We use a rotational equiangular tight frame-classifier and pseudo-labeling to pre-train deep encoders with few labeled samples. Furthermore, to effectively capture temporal dynamics while enforcing embedding separability, we integrate generative pretext tasks with our method, and we define a novel sequential augmentation strategy. We show that our method significantly outperforms previous pretext tasks when applied to LSTMs, transformers, and state-space models on three multivariate time series classification datasets. These results highlight the benefit of aligning pre-training objectives with theoretically grounded embedding geometry. 

**Abstract (ZH)**: 深度神经网络对于时间序列必须捕获复杂的时序模式，以有效地表示动态数据。半监督和自我监督学习方法在预训练大型模型方面显示出有希望的结果，这些方法在微调分类任务后往往比从零开始训练的模型表现更好。然而，预训练任务的选择往往是启发式的，其在下游分类任务上的可转移性并不确定，因此我们提出了一种新的半监督预训练策略，以使潜在表示满足在最优训练神经分类器中观察到的Neural Collapse现象。我们使用旋转等角紧框架分类器和伪标签对少量标记样本进行预训练深度编码器。此外，为了有效捕获时序动态并强制约束嵌入分离性，我们将生成性预训练任务集成到我们的方法中，并定义了一种新的序列增强策略。我们展示了当将该方法应用于LSTMs、变压器和状态空间模型时，在三个多元时间序列分类数据集上显著优于先前的预训练任务，这些结果突显了使预训练目标与理论上的嵌入几何学相一致的好处。 

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
# Empowering Morphing Attack Detection using Interpretable Image-Text Foundation Model 

**Title (ZH)**: 基于可解释的图像-文本基础模型的形态变化攻击检测增强方法 

**Authors**: Sushrut Patwardhan, Raghavendra Ramachandra, Sushma Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2508.10110)  

**Abstract**: Morphing attack detection has become an essential component of face recognition systems for ensuring a reliable verification scenario. In this paper, we present a multimodal learning approach that can provide a textual description of morphing attack detection. We first show that zero-shot evaluation of the proposed framework using Contrastive Language-Image Pretraining (CLIP) can yield not only generalizable morphing attack detection, but also predict the most relevant text snippet. We present an extensive analysis of ten different textual prompts that include both short and long textual prompts. These prompts are engineered by considering the human understandable textual snippet. Extensive experiments were performed on a face morphing dataset that was developed using a publicly available face biometric dataset. We present an evaluation of SOTA pre-trained neural networks together with the proposed framework in the zero-shot evaluation of five different morphing generation techniques that are captured in three different mediums. 

**Abstract (ZH)**: 形态攻击检测已成为确保面部识别系统可靠验证场景的必要组成部分。本文提出了一种多模态学习方法，可为形态攻击检测提供文本描述。我们首先展示使用对比语言-图像预训练（CLIP）进行零样本评估可以不仅实现泛化的形态攻击检测，还能预测最相关的文本片段。我们探讨了包括短文本提示和长文本提示在内的十个不同的文本提示。这些提示通过考虑人类可理解的文本片段进行工程设计。我们在使用公开可用的面部生物特征数据集开发的一个面部形态数据集上进行了大量实验。我们提出了对当前最佳预训练神经网络与所提出框架在五个不同形态生成技术的零样本评估中的评估，这些技术在三个不同的媒介中被捕获。 

---
# Advancing Data Equity: Practitioner Responsibility and Accountability in NLP Data Practices 

**Title (ZH)**: 推进数据公平：NLP数据实践中的从业者责任与问责制 

**Authors**: Jay L. Cunningham, Kevin Zhongyang Shao, Rock Yuren Pang, Nathaniel Mengist  

**Link**: [PDF](https://arxiv.org/pdf/2508.10071)  

**Abstract**: While research has focused on surfacing and auditing algorithmic bias to ensure equitable AI development, less is known about how NLP practitioners - those directly involved in dataset development, annotation, and deployment - perceive and navigate issues of NLP data equity. This study is among the first to center practitioners' perspectives, linking their experiences to a multi-scalar AI governance framework and advancing participatory recommendations that bridge technical, policy, and community domains. Drawing on a 2024 questionnaire and focus group, we examine how U.S.-based NLP data practitioners conceptualize fairness, contend with organizational and systemic constraints, and engage emerging governance efforts such as the U.S. AI Bill of Rights. Findings reveal persistent tensions between commercial objectives and equity commitments, alongside calls for more participatory and accountable data workflows. We critically engage debates on data diversity and diversity washing, arguing that improving NLP equity requires structural governance reforms that support practitioner agency and community consent. 

**Abstract (ZH)**: 尽管研究已侧重于揭示和审查算法偏见以确保公平的人工智能发展，但对于直接参与数据集开发、标注和部署的自然语言处理（NLP）从业者如何感知和应对NLP数据公平性问题了解较少。本研究是首个以从业者的视角为中心的研究，将他们的经验与多尺度的人工智能治理框架联系起来，并提出跨技术、政策和社区领域的参与性建议。基于2024年的问卷调查和焦点小组，我们探讨了基于美国的数据从业者如何定义公平性、应对组织和系统的约束，并参与诸如美国人工智能权利法案等新兴治理努力。研究发现，商业目标与公平承诺之间存在持续的紧张关系，同时也呼吁需要更加参与性和负责任的数据工作流程。我们批判性地参与关于数据多样性和多样性漂洗的争论，认为提升NLP公平性需要结构性的治理改革，以支持从业者的自主权和社区的同意。 

---
# Large Language Models Show Signs of Alignment with Human Neurocognition During Abstract Reasoning 

**Title (ZH)**: 大型语言模型在抽象推理过程中表现出与人类神经认知的对齐迹象 

**Authors**: Christopher Pinier, Sonia Acuña Vargas, Mariia Steeghs-Turchina, Dora Matzke, Claire E. Stevenson, Michael D. Nunez  

**Link**: [PDF](https://arxiv.org/pdf/2508.10057)  

**Abstract**: This study investigates whether large language models (LLMs) mirror human neurocognition during abstract reasoning. We compared the performance and neural representations of human participants with those of eight open-source LLMs on an abstract-pattern-completion task. We leveraged pattern type differences in task performance and in fixation-related potentials (FRPs) as recorded by electroencephalography (EEG) during the task. Our findings indicate that only the largest tested LLMs (~70 billion parameters) achieve human-comparable accuracy, with Qwen-2.5-72B and DeepSeek-R1-70B also showing similarities with the human pattern-specific difficulty profile. Critically, every LLM tested forms representations that distinctly cluster the abstract pattern categories within their intermediate layers, although the strength of this clustering scales with their performance on the task. Moderate positive correlations were observed between the representational geometries of task-optimal LLM layers and human frontal FRPs. These results consistently diverged from comparisons with other EEG measures (response-locked ERPs and resting EEG), suggesting a potential shared representational space for abstract patterns. This indicates that LLMs might mirror human brain mechanisms in abstract reasoning, offering preliminary evidence of shared principles between biological and artificial intelligence. 

**Abstract (ZH)**: 本研究调查大型语言模型在抽象推理过程中是否镜像人类神经认知。我们通过一项抽象模式完成任务，将人类参与者的性能和神经表示与八个开源大型语言模型进行了比较，并利用任务中不同模式类型的性能差异以及脑电图（EEG）记录的注意焦点相关电位（FRP）的差异进行分析。研究发现，只有最大的测试模型（约700亿参数）达到与人类相当的准确度，同时Qwen-2.5-72B和DeepSeek-R1-70B也显示出与人类模式特定难度特征的相似性。关键的是，每个测试的模型在其中间层中都形成了将抽象模式类别区分开来的表示，虽然这种区别的强度与它们在任务上的表现成比例。任务优化的大型语言模型层的表示几何与人类前额叶的FRP之间存在中等正相关。这些结果与与其他EEG测量（反应锁定的ERM和静息EEG）的比较不一致，表明可能存在一种抽象模式的共享表示空间。这表明大型语言模型可能在抽象推理中镜像人类大脑机制，为生物智能和人工智能共享原则提供了初步证据。 

---
# NetMoniAI: An Agentic AI Framework for Network Security & Monitoring 

**Title (ZH)**: NetMoniAI: 一个网络安全部署型人工智能框架 

**Authors**: Pallavi Zambare, Venkata Nikhil Thanikella, Nikhil Padmanabh Kottur, Sree Akhil Akula, Ying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10052)  

**Abstract**: In this paper, we present NetMoniAI, an agentic AI framework for automatic network monitoring and security that integrates decentralized analysis with lightweight centralized coordination. The framework consists of two layers: autonomous micro-agents at each node perform local traffic analysis and anomaly detection. A central controller then aggregates insights across nodes to detect coordinated attacks and maintain system-wide situational awareness. We evaluated NetMoniAI on a local micro-testbed and through NS-3 simulations. Results confirm that the two-tier agentic-AI design scales under resource constraints, reduces redundancy, and improves response time without compromising accuracy. To facilitate broader adoption and reproducibility, the complete framework is available as open source. This enables researchers and practitioners to replicate, validate, and extend it across diverse network environments and threat scenarios. Github link: this https URL 

**Abstract (ZH)**: 本文介绍了NetMoniAI，一种集成去中心化分析与轻量级集中协调的自主AI框架，用于自动网络监控和安全防护。该框架分为两层：每个节点上的自主微代理执行本地流量分析和异常检测。中心控制器则聚合各节点的洞察以检测协同攻击，保持全系统的态势感知。我们通过本地微实验床和NS-3仿真对该框架进行了评估。结果表明，该两层自主-AI设计在资源受限条件下可扩展、减少冗余、提高响应速度而不牺牲准确性。为促进更广泛的采用和可再现性，该完整框架已开源。研究人员和实践者可以利用它在不同的网络环境和威胁场景下进行复制、验证和扩展。Github链接：this https URL 

---
# Legal Zero-Days: A Novel Risk Vector for Advanced AI Systems 

**Title (ZH)**: 合法零日：高级人工智能系统的一种新型风险因素 

**Authors**: Greg Sadler, Nathan Sherburn  

**Link**: [PDF](https://arxiv.org/pdf/2508.10050)  

**Abstract**: We introduce the concept of "Legal Zero-Days" as a novel risk vector for advanced AI systems. Legal Zero-Days are previously undiscovered vulnerabilities in legal frameworks that, when exploited, can cause immediate and significant societal disruption without requiring litigation or other processes before impact. We present a risk model for identifying and evaluating these vulnerabilities, demonstrating their potential to bypass safeguards or impede government responses to AI incidents. Using the 2017 Australian dual citizenship crisis as a case study, we illustrate how seemingly minor legal oversights can lead to large-scale governance disruption. We develop a methodology for creating "legal puzzles" as evaluation instruments for assessing AI systems' capabilities to discover such vulnerabilities. Our findings suggest that while current AI models may not reliably find impactful Legal Zero-Days, future systems may develop this capability, presenting both risks and opportunities for improving legal robustness. This work contributes to the broader effort to identify and mitigate previously unrecognized risks from frontier AI systems. 

**Abstract (ZH)**: 我们将“法律零日”概念引入高级AI系统的新风险向量中。法律零日是指在法律框架中此前未被发现的漏洞，当被利用时，可以导致立即且重大的社会扰乱，无需在影响前进行诉讼或其他程序。我们提出了一种风险模型来识别和评估这些漏洞，展示了它们如何规避保护措施或妨碍政府对AI事件的响应。通过2017年澳大利亚双重公民危机案例研究，我们说明了看似微小的法律疏忽如何导致大规模治理中断。我们开发了一种方法论，用于创建“法律难题”作为评估AI系统发现此类漏洞能力的评估工具。我们的研究结果表明，虽然当前的AI模型可能无法可靠地发现有影响力的法律零日，但未来系统可能会培养这一能力，既带来风险也带来提升法律稳健性的机会。本项工作为识别和缓解前沿AI系统的未被认识的风险做出了贡献。 

---
# SABIA: An AI-Powered Tool for Detecting Opioid-Related Behaviors on Social Media 

**Title (ZH)**: SABIA: 一种基于人工智能的检测社交媒体上与 opioids 相关行为的工具 

**Authors**: Muhammad Ahmad, Fida Ullah, Muhammad Usman, Ildar Batyrshin, Grigori Sidorov  

**Link**: [PDF](https://arxiv.org/pdf/2508.10046)  

**Abstract**: Social media platforms have become valuable tools for understanding public health challenges by offering insights into patient behaviors, medication use, and mental health issues. However, analyzing such data remains difficult due to the prevalence of informal language, slang, and coded communication, which can obscure the detection of opioid misuse. This study addresses the issue of opioid-related user behavior on social media, including informal expressions, slang terms, and misspelled or coded language. We analyzed the existing Bidirectional Encoder Representations from Transformers (BERT) technique and developed a BERT-BiLSTM-3CNN hybrid deep learning model, named SABIA, to create a single-task classifier that effectively captures the features of the target dataset. The SABIA model demonstrated strong capabilities in capturing semantics and contextual information. The proposed approach includes: (1) data preprocessing, (2) data representation using the SABIA model, (3) a fine-tuning phase, and (4) classification of user behavior into five categories. A new dataset was constructed from Reddit posts, identifying opioid user behaviors across five classes: Dealers, Active Opioid Users, Recovered Users, Prescription Users, and Non-Users, supported by detailed annotation guidelines. Experiments were conducted using supervised learning. Results show that SABIA achieved benchmark performance, outperforming the baseline (Logistic Regression, LR = 0.86) and improving accuracy by 9.30%. Comparisons with seven previous studies confirmed its effectiveness and robustness. This study demonstrates the potential of hybrid deep learning models for detecting complex opioid-related behaviors on social media, supporting public health monitoring and intervention efforts. 

**Abstract (ZH)**: 社交媒体平台已成为理解公共卫生挑战的重要工具，通过提供患者行为、药物使用和心理健康问题的见解。然而，分析这类数据仍具有挑战性，因为其中普遍存在非正式语言、俚语和编码交流，这些因素可能掩盖住阿片类药物滥用的迹象。本研究着眼于社交媒体上的阿片类药物相关用户行为，包括非正式表达、俚语术语和拼写错误或编码语言。我们分析了现有的双向编码表示变换器（BERT）技术，并开发了一种名为SABIA的BERT-BiLSTM-3CNN混合深度学习模型，以创建一个能够有效捕获目标数据集特征的单任务分类器。SABIA模型在捕获语义和上下文信息方面展现了强大能力。本方法包括：（1）数据预处理，（2）使用SABIA模型的数据表示，（3）微调阶段，和（4）将用户行为分为五类的分类。从Reddit帖子构建了一个新的数据集，识别了五类阿片类药物用户行为：贩子、活跃阿片类药物用户、康复中用户、处方使用者和非使用者，并提供了详细的注释指南。使用监督学习进行了实验。结果显示，SABIA达到了基准性能，优于基线（逻辑回归，LR = 0.86），准确率提高了9.30%。与七项先前研究的比较证实了其有效性和鲁棒性。本研究展示了混合深度学习模型在检测社交媒体上的复杂阿片类药物相关行为方面的潜力，支持公共卫生监测和干预努力。 

---
# Generative AI for Cybersecurity of Energy Management Systems: Methods, Challenges, and Future Directions 

**Title (ZH)**: 生成式AI在能源管理系统网络安全中的应用：方法、挑战与未来方向 

**Authors**: Aydin Zaboli, Junho Hong  

**Link**: [PDF](https://arxiv.org/pdf/2508.10044)  

**Abstract**: This paper elaborates on an extensive security framework specifically designed for energy management systems (EMSs), which effectively tackles the dynamic environment of cybersecurity vulnerabilities and/or system problems (SPs), accomplished through the incorporation of novel methodologies. A comprehensive multi-point attack/error model is initially proposed to systematically identify vulnerabilities throughout the entire EMS data processing pipeline, including post state estimation (SE) stealth attacks, EMS database manipulation, and human-machine interface (HMI) display corruption according to the real-time database (RTDB) storage. This framework acknowledges the interconnected nature of modern attack vectors, which utilize various phases of supervisory control and data acquisition (SCADA) data flow. Then, generative AI (GenAI)-based anomaly detection systems (ADSs) for EMSs are proposed for the first time in the power system domain to handle the scenarios. Further, a set-of-mark generative intelligence (SoM-GI) framework, which leverages multimodal analysis by integrating visual markers with rules considering the GenAI capabilities, is suggested to overcome inherent spatial reasoning limitations. The SoM-GI methodology employs systematic visual indicators to enable accurate interpretation of segmented HMI displays and detect visual anomalies that numerical methods fail to identify. Validation on the IEEE 14-Bus system shows the framework's effectiveness across scenarios, while visual analysis identifies inconsistencies. This integrated approach combines numerical analysis with visual pattern recognition and linguistic rules to protect against cyber threats and system errors. 

**Abstract (ZH)**: 一种针对能量管理系统（EMSs）的全面安全框架：结合新颖方法有效应对动态的网络安全漏洞和/或系统问题的研究 

---
# Securing Agentic AI: Threat Modeling and Risk Analysis for Network Monitoring Agentic AI System 

**Title (ZH)**: 保障自主人工智能安全：网络监控自主人工智能系统威胁建模与风险分析 

**Authors**: Pallavi Zambare, Venkata Nikhil Thanikella, Ying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10043)  

**Abstract**: When combining Large Language Models (LLMs) with autonomous agents, used in network monitoring and decision-making systems, this will create serious security issues. In this research, the MAESTRO framework consisting of the seven layers threat modeling architecture in the system was used to expose, evaluate, and eliminate vulnerabilities of agentic AI. The prototype agent system was constructed and implemented, using Python, LangChain, and telemetry in WebSockets, and deployed with inference, memory, parameter tuning, and anomaly detection modules. Two practical threat cases were confirmed as follows: (i) resource denial of service by traffic replay denial-of-service, and (ii) memory poisoning by tampering with the historical log file maintained by the agent. These situations resulted in measurable levels of performance degradation, i.e. telemetry updates were delayed, and computational loads were increased, as a result of poor system adaptations. It was suggested to use a multilayered defense-in-depth approach with memory isolation, validation of planners and anomaly response systems in real-time. These findings verify that MAESTRO is viable in operational threat mapping, prospective risk scoring, and the basis of the resilient system design. The authors bring attention to the importance of the enforcement of memory integrity, paying attention to the adaptation logic monitoring, and cross-layer communication protection that guarantee the agentic AI reliability in adversarial settings. 

**Abstract (ZH)**: 当将大型语言模型（LLMs）与用于网络监控和决策系统的自主代理相结合时，会创建出严重安全问题。 

---
# FIDELIS: Blockchain-Enabled Protection Against Poisoning Attacks in Federated Learning 

**Title (ZH)**: FIDELIS: 联邦学习中基于区块链的对抗投毒攻击保护方法 

**Authors**: Jane Carney, Kushal Upreti, Gaby G. Dagher, Tim Andersen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10042)  

**Abstract**: Federated learning enhances traditional deep learning by enabling the joint training of a model with the use of IoT device's private data. It ensures privacy for clients, but is susceptible to data poisoning attacks during training that degrade model performance and integrity. Current poisoning detection methods in federated learning lack a standardized detection method or take significant liberties with trust. In this paper, we present \Sys, a novel blockchain-enabled poison detection framework in federated learning. The framework decentralizes the role of the global server across participating clients. We introduce a judge model used to detect data poisoning in model updates. The judge model is produced by each client and verified to reach consensus on a single judge model. We implement our solution to show \Sys is robust against data poisoning attacks and the creation of our judge model is scalable. 

**Abstract (ZH)**: 联邦学习通过启用物联网设备私有数据的联合训练，增强了传统的深度学习。然而，在训练过程中容易受到数据投毒攻击的影响，这些攻击会降低模型性能和完整性。当前的联邦学习反投毒方法缺乏标准化的检测方法，或者在信任方面做出重大妥协。本文提出了一种基于区块链的新型联邦学习反投毒框架 \Sys。该框架将全局服务器的角色分散到参与客户端中。我们引入了一个判别模型来检测模型更新中的数据投毒。每个客户端生成判别模型并经过验证以达成一致决策。我们实现了解决方案，证明 \Sys 对数据投毒攻击具有鲁棒性，且构建判别模型的过程具有可扩展性。 

---
# Exploring Content and Social Connections of Fake News with Explainable Text and Graph Learning 

**Title (ZH)**: 探索可解释的文本和图学习在假新闻内容和社会连接分析中的应用 

**Authors**: Vítor N. Lourenço, Aline Paes, and Tillman Weyde  

**Link**: [PDF](https://arxiv.org/pdf/2508.10040)  

**Abstract**: The global spread of misinformation and concerns about content trustworthiness have driven the development of automated fact-checking systems. Since false information often exploits social media dynamics such as "likes" and user networks to amplify its reach, effective solutions must go beyond content analysis to incorporate these factors. Moreover, simply labelling content as false can be ineffective or even reinforce biases such as automation and confirmation bias. This paper proposes an explainable framework that combines content, social media, and graph-based features to enhance fact-checking. It integrates a misinformation classifier with explainability techniques to deliver complete and interpretable insights supporting classification decisions. Experiments demonstrate that multimodal information improves performance over single modalities, with evaluations conducted on datasets in English, Spanish, and Portuguese. Additionally, the framework's explanations were assessed for interpretability, trustworthiness, and robustness with a novel protocol, showing that it effectively generates human-understandable justifications for its predictions. 

**Abstract (ZH)**: 标题翻译如下：

基于虚假信息在全球范围内的传播以及内容可信度的担忧，自动事实核查系统得到了快速发展。

内容翻译如下：

为了应对 虚假信息的全球范围传播以及对 内容可信度的担忧，自动事实核查系统得到了快速发展。 常利用社交媒体动态，例如 “点赞" 和 社交网络" 以增强虚假信息的传播。这类系统不仅限于 内容分析,还 会结合其他特征以提供更全面且可解释性的洞察支持支持决策。此外,简化的内容也可能会无效地强化自动化和确认偏差等本框架结合了基于内容、社交媒体和图谱特征以增强事实核查的效果。该框架集成了虚假信息分类器与内容可信度 性技术以提供全面且可解释性的洞察并支持决策。实验表明了多模态优于 仅单一模态的方法且在由 于 数据集 (英文、西班牙文和葡萄牙文) 上进行的评估表明了该框架在可释性性、可信 度和稳健性性方面的有效性且通过一个协议有效地评估了对预测的易为人理解的解释。 

---
# Multi-task Adversarial Attacks against Black-box Model with Few-shot Queries 

**Title (ZH)**: 针对少样本查询的黑盒模型多任务对抗攻击 

**Authors**: Wenqiang Wang, Yan Xiao, Hao Lin, Yangshijie Zhang, Xiaochun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2508.10039)  

**Abstract**: Current multi-task adversarial text attacks rely on abundant access to shared internal features and numerous queries, often limited to a single task type. As a result, these attacks are less effective against practical scenarios involving black-box feedback APIs, limited queries, or multiple task types. To bridge this gap, we propose \textbf{C}luster and \textbf{E}nsemble \textbf{M}ulti-task Text Adversarial \textbf{A}ttack (\textbf{CEMA}), an effective black-box attack that exploits the transferability of adversarial texts across different tasks. CEMA simplifies complex multi-task scenarios by using a \textit{deep-level substitute model} trained in a \textit{plug-and-play} manner for text classification, enabling attacks without mimicking the victim model. This approach requires only a few queries for training, converting multi-task attacks into classification attacks and allowing attacks across various tasks.
CEMA generates multiple adversarial candidates using different text classification methods and selects the one that most effectively attacks substitute models.
In experiments involving multi-task models with two, three, or six tasks--spanning classification, translation, summarization, and text-to-image generation--CEMA demonstrates significant attack success with as few as 100 queries. Furthermore, CEMA can target commercial APIs (e.g., Baidu and Google Translate), large language models (e.g., ChatGPT 4o), and image-generation models (e.g., Stable Diffusion V2), showcasing its versatility and effectiveness in real-world applications. 

**Abstract (ZH)**: 集群与集成多任务文本对抗攻击（CEMA）：一种有效的黑盒攻击方法 

---
# Certifiably robust malware detectors by design 

**Title (ZH)**: 设计上的可验证鲁棒恶意软件检测器 

**Authors**: Pierre-Francois Gimenez, Sarath Sivaprasad, Mario Fritz  

**Link**: [PDF](https://arxiv.org/pdf/2508.10038)  

**Abstract**: Malware analysis involves analyzing suspicious software to detect malicious payloads. Static malware analysis, which does not require software execution, relies increasingly on machine learning techniques to achieve scalability. Although such techniques obtain very high detection accuracy, they can be easily evaded with adversarial examples where a few modifications of the sample can dupe the detector without modifying the behavior of the software. Unlike other domains, such as computer vision, creating an adversarial example of malware without altering its functionality requires specific transformations. We propose a new model architecture for certifiably robust malware detection by design. In addition, we show that every robust detector can be decomposed into a specific structure, which can be applied to learn empirically robust malware detectors, even on fragile features. Our framework ERDALT is based on this structure. We compare and validate these approaches with machine-learning-based malware detection methods, allowing for robust detection with limited reduction of detection performance. 

**Abstract (ZH)**: 恶意软件分析涉及分析可疑软件以检测恶意载荷。静态恶意软件分析不依赖软件执行，越来越多地利用机器学习技术以实现可扩展性。虽然此类技术能够获得非常高的检测准确性，但可以通过对抗样本轻松规避，即通过对样本进行少量修改以误导检测器而不改变软件行为。与计算机视觉等领域不同，不需要修改功能即可创建恶意软件的对抗样本需要特定的转换。我们提出了一种新的模型架构，旨在通过设计实现可验证鲁棒的恶意软件检测。此外，我们证明每个鲁棒检测器可以分解为特定结构，该结构可以应用于学习经验上鲁棒的恶意软件检测器，即使在脆弱特征上亦然。我们的框架ERDALT基于这种结构。我们通过与基于机器学习的恶意软件检测方法进行比较和验证，使得在有限降低检测性能的情况下实现鲁棒检测。 

---
# Reflect then Learn: Active Prompting for Information Extraction Guided by Introspective Confusion 

**Title (ZH)**: 反思then学习：由 introspective confusion驱动的主动提示引导信息提取 

**Authors**: Dong Zhao, Yadong Wang, Xiang Chen, Chenxi Wang, Hongliang Dai, Chuanxing Geng, Shengzhong Zhang, Shaoyuan Li, Sheng-Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10036)  

**Abstract**: Large Language Models (LLMs) show remarkable potential for few-shot information extraction (IE), yet their performance is highly sensitive to the choice of in-context examples. Conventional selection strategies often fail to provide informative guidance, as they overlook a key source of model fallibility: confusion stemming not just from semantic content, but also from the generation of well-structured formats required by IE tasks. To address this, we introduce Active Prompting for Information Extraction (APIE), a novel active prompting framework guided by a principle we term introspective confusion. Our method empowers an LLM to assess its own confusion through a dual-component uncertainty metric that uniquely quantifies both Format Uncertainty (difficulty in generating correct syntax) and Content Uncertainty (inconsistency in extracted semantics). By ranking unlabeled data with this comprehensive score, our framework actively selects the most challenging and informative samples to serve as few-shot exemplars. Extensive experiments on four benchmarks show that our approach consistently outperforms strong baselines, yielding significant improvements in both extraction accuracy and robustness. Our work highlights the critical importance of a fine-grained, dual-level view of model uncertainty when it comes to building effective and reliable structured generation systems. 

**Abstract (ZH)**: 基于主动提示的信息提取中的反省混淆（APIE）：细粒度双层建模模式不确定性以提升少样本信息提取性能 

---
# Jet Image Tagging Using Deep Learning: An Ensemble Model 

**Title (ZH)**: 基于深度学习的喷流图像标签聚合模型 

**Authors**: Juvenal Bassa, Vidya Manian, Sudhir Malik, Arghya Chattopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2508.10034)  

**Abstract**: Jet classification in high-energy particle physics is important for understanding fundamental interactions and probing phenomena beyond the Standard Model. Jets originate from the fragmentation and hadronization of quarks and gluons, and pose a challenge for identification due to their complex, multidimensional structure. Traditional classification methods often fall short in capturing these intricacies, necessitating advanced machine learning approaches. In this paper, we employ two neural networks simultaneously as an ensemble to tag various jet types. We convert the jet data to two-dimensional histograms instead of representing them as points in a higher-dimensional space. Specifically, this ensemble approach, hereafter referred to as Ensemble Model, is used to tag jets into classes from the JetNet dataset, corresponding to: Top Quarks, Light Quarks (up or down), and W and Z bosons. For the jet classes mentioned above, we show that the Ensemble Model can be used for both binary and multi-categorical classification. This ensemble approach learns jet features by leveraging the strengths of each constituent network achieving superior performance compared to either individual network. 

**Abstract (ZH)**: 高能粒子物理中的喷流分类对于理解基本相互作用和探索标准模型之外的现象至关重要。本论文采用两个神经网络相结合的方法对各种喷流类型进行标记，并将喷流数据转换为二维直方图，而不是将其表示为高维空间中的点。具体而言，该组合方法（此后统称为集成模型）被用于对JetNet数据集中对应于顶夸克、轻夸克（上夸子或下夸子）以及W和Z玻色子的喷流类别进行分类。对于上述喷流类别，我们证明集成模型可用于二元和多元分类。该组合方法通过充分发挥各组成网络的优势，实现优于单一网络的性能。 

---
# Cognitive Cybersecurity for Artificial Intelligence: Guardrail Engineering with CCS-7 

**Title (ZH)**: 人工智能领域的认知网络安全：基于CCS-7的护栏工程 

**Authors**: Yuksel Aydin  

**Link**: [PDF](https://arxiv.org/pdf/2508.10033)  

**Abstract**: Language models exhibit human-like cognitive vulnerabilities, such as emotional framing, that escape traditional behavioral alignment. We present CCS-7 (Cognitive Cybersecurity Suite), a taxonomy of seven vulnerabilities grounded in human cognitive security research. To establish a human benchmark, we ran a randomized controlled trial with 151 participants: a "Think First, Verify Always" (TFVA) lesson improved cognitive security by +7.9% overall. We then evaluated TFVA-style guardrails across 12,180 experiments on seven diverse language model architectures. Results reveal architecture-dependent risk patterns: some vulnerabilities (e.g., identity confusion) are almost fully mitigated, while others (e.g., source interference) exhibit escalating backfire, with error rates increasing by up to 135% in certain models. Humans, in contrast, show consistent moderate improvement. These findings reframe cognitive safety as a model-specific engineering problem: interventions effective in one architecture may fail, or actively harm, another, underscoring the need for architecture-aware cognitive safety testing before deployment. 

**Abstract (ZH)**: 语言模型表现出类似人类的认知脆弱性，如情绪框架效应，这些脆弱性超出了传统的行为对齐。我们提出了认知网络安全套件CCS-7（Cognitive Cybersecurity Suite），该套件基于人类认知安全研究，包含七种漏洞分类。为了建立人类基准，我们进行了随机对照试验，共有151名参与者：一项“先思考，后验证”（TFVA）课程总体上提高了认知安全性7.9%。然后，我们评估了TFVA风格的防护措施在七种不同语言模型架构的12,180次实验中。结果显示，不同架构的风险模式各不相同：某些漏洞（如身份混淆）几乎完全得到缓解，而其他漏洞（如来源干扰）则表现出增强的回火现象，某些模型中的错误率最高可增加135%。相比之下，人类在各方面都表现出一致的适度改进。这些发现将认知安全性重新定义为模型特定的工程问题：一项在一种架构中有效的干预措施可能在另一种架构中失效，甚至造成损害，从而凸显了在部署前进行架构意识认知安全性测试的必要性。 

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
# A Robust Pipeline for Differentially Private Federated Learning on Imbalanced Clinical Data using SMOTETomek and FedProx 

**Title (ZH)**: 一种基于SMOTETomek和FedProx的抗扰动临床不平衡数据联邦学习隐私保护管道 

**Authors**: Rodrigo Tertulino  

**Link**: [PDF](https://arxiv.org/pdf/2508.10017)  

**Abstract**: Federated Learning (FL) presents a groundbreaking approach for collaborative health research, allowing model training on decentralized data while safeguarding patient privacy. FL offers formal security guarantees when combined with Differential Privacy (DP). The integration of these technologies, however, introduces a significant trade-off between privacy and clinical utility, a challenge further complicated by the severe class imbalance often present in medical datasets. The research presented herein addresses these interconnected issues through a systematic, multi-stage analysis. An FL framework was implemented for cardiovascular risk prediction, where initial experiments showed that standard methods struggled with imbalanced data, resulting in a recall of zero. To overcome such a limitation, we first integrated the hybrid Synthetic Minority Over-sampling Technique with Tomek Links (SMOTETomek) at the client level, successfully developing a clinically useful model. Subsequently, the framework was optimized for non-IID data using a tuned FedProx algorithm. Our final results reveal a clear, non-linear trade-off between the privacy budget (epsilon) and model recall, with the optimized FedProx consistently out-performing standard FedAvg. An optimal operational region was identified on the privacy-utility frontier, where strong privacy guarantees (with epsilon 9.0) can be achieved while maintaining high clinical utility (recall greater than 77%). Ultimately, our study provides a practical methodological blueprint for creating effective, secure, and accurate diagnostic tools that can be applied to real-world, heterogeneous healthcare data. 

**Abstract (ZH)**: 联邦学习（FL）为协作健康研究提供了一种革命性的方法，允许在去中心化数据上进行模型训练以保障患者隐私。当与差分隐私（DP）结合使用时，FL提供了正式的安全保证。然而，这两种技术的整合引入了隐私与临床效用之间的重要权衡，这一挑战在医学数据集中经常严重的类别不平衡现象下进一步复杂化。本文通过系统多阶段分析解决了这些相互关联的问题。我们在心血管风险预测中实现了一个FL框架，初步实验显示标准方法在不平衡数据上表现出局限性，导致召回率为零。为克服这一局限性，我们首先在客户端层面整合了合成少数类过采样技术与Tomek链接（SMOTETomek），成功开发出一个临床有用模型。随后，我们使用调优后的FedProx算法优化了该框架以适应非 IID 数据。最终结果表明，在隐私预算（ε）与模型召回率之间存在明显的非线性权衡，优化后的FedProx算法在保持高临床效用（召回率大于77%）的同时，提供了更强的隐私保证（ε=9.0）。我们的研究最终提供了一种实用的方法论框架，用于创建有效的、安全的和精确的诊断工具，以应用于现实世界中的异质医疗数据。 

---
# Beyond Hard Sharing: Efficient Multi-Task Speech-to-Text Modeling with Supervised Mixture of Experts 

**Title (ZH)**: 超越硬共享：基于监督专家混合的高效多任务语音转文本建模 

**Authors**: Hojun Jin, Eunsoo Hong, Ziwon Hyung, Sungjun Lim, Seungjin Lee, Keunseok Cho  

**Link**: [PDF](https://arxiv.org/pdf/2508.10009)  

**Abstract**: Hard-parameter sharing is a common strategy to train a single model jointly across diverse tasks. However, this often leads to task interference, impeding overall model performance. To address the issue, we propose a simple yet effective Supervised Mixture of Experts (S-MoE). Unlike traditional Mixture of Experts models, S-MoE eliminates the need for training gating functions by utilizing special guiding tokens to route each task to its designated expert. By assigning each task to a separate feedforward network, S-MoE overcomes the limitations of hard-parameter sharing. We further apply S-MoE to a speech-to-text model, enabling the model to process mixed-bandwidth input while jointly performing automatic speech recognition (ASR) and speech translation (ST). Experimental results demonstrate the effectiveness of the proposed S-MoE, achieving a 6.35% relative improvement in Word Error Rate (WER) when applied to both the encoder and decoder. 

**Abstract (ZH)**: 监督专家混合模型（S-MoE）：一种简单有效的稀疏参数共享方法 

---
# From Answers to Questions: EQGBench for Evaluating LLMs' Educational Question Generation 

**Title (ZH)**: 从答案到问题：EQGBench 用于评估大型语言模型的教育性问题生成能力 

**Authors**: Chengliang Zhou, Mei Wang, Ting Zhang, Qiannan Zhu, Jian Li, Hua Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10005)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in mathematical problem-solving. However, the transition from providing answers to generating high-quality educational questions presents significant challenges that remain underexplored. To advance Educational Question Generation (EQG) and facilitate LLMs in generating pedagogically valuable and educationally effective questions, we introduce EQGBench, a comprehensive benchmark specifically designed for evaluating LLMs' performance in Chinese EQG. EQGBench establishes a five-dimensional evaluation framework supported by a dataset of 900 evaluation samples spanning three fundamental middle school disciplines: mathematics, physics, and chemistry. The dataset incorporates user queries with varying knowledge points, difficulty gradients, and question type specifications to simulate realistic educational scenarios. Through systematic evaluation of 46 mainstream large models, we reveal significant room for development in generating questions that reflect educational value and foster students' comprehensive abilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）在数学问题解决方面展现了出色的能力。然而，从提供答案到生成高质量的教育性问题这一转变面临着尚未充分探索的重大挑战。为促进教育性问题生成（EQG）并帮助LLMs生成教育价值高且有效的教学问题，我们介绍了EQGBench，一个专门用于评估LLMs在中文EQG方面性能的综合性基准。EQGBench建立了由900个评价样本组成的数据集，覆盖三个基本的中学学科：数学、物理和化学，并建立了五维评估框架。该数据集包含了不同知识点、难度等级和问题类型要求的用户查询，以模拟现实的教育场景。通过对46个主流大模型的系统评估，我们揭示了在生成反映教育价值问题方面仍有显著的发展空间，有助于培养学生的综合能力。 

---
# User Perception of Attention Visualizations: Effects on Interpretability Across Evidence-Based Medical Documents 

**Title (ZH)**: 用户对注意力可视化的效果感知：基于证据的医学文档可解释性影响研究 

**Authors**: Andrés Carvallo, Denis Parra, Peter Brusilovsky, Hernan Valdivieso, Gabriel Rada, Ivania Donoso, Vladimir Araujo  

**Link**: [PDF](https://arxiv.org/pdf/2508.10004)  

**Abstract**: The attention mechanism is a core component of the Transformer architecture. Beyond improving performance, attention has been proposed as a mechanism for explainability via attention weights, which are associated with input features (e.g., tokens in a document). In this context, larger attention weights may imply more relevant features for the model's prediction. In evidence-based medicine, such explanations could support physicians' understanding and interaction with AI systems used to categorize biomedical literature. However, there is still no consensus on whether attention weights provide helpful explanations. Moreover, little research has explored how visualizing attention affects its usefulness as an explanation aid. To bridge this gap, we conducted a user study to evaluate whether attention-based explanations support users in biomedical document classification and whether there is a preferred way to visualize them. The study involved medical experts from various disciplines who classified articles based on study design (e.g., systematic reviews, broad synthesis, randomized and non-randomized trials). Our findings show that the Transformer model (XLNet) classified documents accurately; however, the attention weights were not perceived as particularly helpful for explaining the predictions. However, this perception varied significantly depending on how attention was visualized. Contrary to Munzner's principle of visual effectiveness, which favors precise encodings like bar length, users preferred more intuitive formats, such as text brightness or background color. While our results do not confirm the overall utility of attention weights for explanation, they suggest that their perceived helpfulness is influenced by how they are visually presented. 

**Abstract (ZH)**: Transformer架构中的注意力机制是核心组件。除了提升性能外，注意力还被提议作为一种通过注意力权重可解释性的机制，这些权重与输入特征（例如，在文档中的标记）相关。在此背景下，更大的注意力权重可能意味着对模型预测更为相关的特点。在基于证据的医学中，这样的解释能够支持医生对用于分类生物医学文献的AI系统的理解和交互。然而，目前尚无共识认为注意力权重提供了有用解释。此外，很少有研究探讨可视化注意力如何影响其作为解释辅助的有用性。为了弥合这一差距，我们进行了一项用户研究，评估基于注意力的解释是否帮助用户分类生物医学文档，以及如何可视化这些解释更为优选。该研究涉及来自不同学科的医学专家，他们基于研究设计（如系统综述、广泛综合性研究、随机和非随机试验）分类文章。我们的研究发现，Transformer模型（XLNet）能够准确地分类文档；然而，注意力权重并未被感知为特别有助于解释预测。然而，这种感知主要取决于注意力如何被可视化。与Munzner的可视有效原则（偏好精确编码，如条形图长度）相反，用户更偏好更直观的形式，如文本亮度或背景颜色。虽然我们的研究结果并未确认总体而言注意力权重对于解释的有用性，但它们表明其感知到的帮助程度受其可视化方式的影响。 

---
# Semantic Structure in Large Language Model Embeddings 

**Title (ZH)**: 大型语言模型嵌入中的语义结构 

**Authors**: Austin C. Kozlowski, Callin Dai, Andrei Boutyline  

**Link**: [PDF](https://arxiv.org/pdf/2508.10003)  

**Abstract**: Psychological research consistently finds that human ratings of words across diverse semantic scales can be reduced to a low-dimensional form with relatively little information loss. We find that the semantic associations encoded in the embedding matrices of large language models (LLMs) exhibit a similar structure. We show that the projections of words on semantic directions defined by antonym pairs (e.g. kind - cruel) correlate highly with human ratings, and further find that these projections effectively reduce to a 3-dimensional subspace within LLM embeddings, closely resembling the patterns derived from human survey responses. Moreover, we find that shifting tokens along one semantic direction causes off-target effects on geometrically aligned features proportional to their cosine similarity. These findings suggest that semantic features are entangled within LLMs similarly to how they are interconnected in human language, and a great deal of semantic information, despite its apparent complexity, is surprisingly low-dimensional. Furthermore, accounting for this semantic structure may prove essential for avoiding unintended consequences when steering features. 

**Abstract (ZH)**: 心理学研究一致发现，人类对不同语义尺度的单词评级可以被简化为一种低维形式，信息损失相对较小。我们发现，大型语言模型（LLMs）的嵌入矩阵中编码的语义关联表现出类似的结构。我们证明，单词在由反义词对（如kind - cruel）定义的语义方向上的投影与人类评级高度相关，并进一步发现这些投影在LLM嵌入中有效减少到三维子空间，与从人类调查响应中得出的模式相似。此外，我们发现沿一个语义方向移动令牌会导致与其余几何对齐特征成正比的意外影响。这些发现表明，语义特征在LLMs中纠缠在一起，类似于人类语言中它们的相互连接方式，尽管语义信息显得复杂，但实际上却高度低维。此外，考虑这种语义结构在引导特征时可能是避免无意后果的关键。 

---
# HiFACTMix: A Code-Mixed Benchmark and Graph-Aware Model for EvidenceBased Political Claim Verification in Hinglish 

**Title (ZH)**: HiFACTMix: 一种代码混用基准及图aware模型，用于基于证据的印英混合政治声明验证 

**Authors**: Rakesh Thakur, Sneha Sharma, Gauri Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2508.10001)  

**Abstract**: Fact-checking in code-mixed, low-resource languages such as Hinglish remains an underexplored challenge in natural language processing. Existing fact-verification systems largely focus on high-resource, monolingual settings and fail to generalize to real-world political discourse in linguistically diverse regions like India. Given the widespread use of Hinglish by public figures, particularly political figures, and the growing influence of social media on public opinion, there's a critical need for robust, multilingual and context-aware fact-checking tools. To address this gap a novel benchmark HiFACT dataset is introduced with 1,500 realworld factual claims made by 28 Indian state Chief Ministers in Hinglish, under a highly code-mixed low-resource setting. Each claim is annotated with textual evidence and veracity labels. To evaluate this benchmark, a novel graphaware, retrieval-augmented fact-checking model is proposed that combines multilingual contextual encoding, claim-evidence semantic alignment, evidence graph construction, graph neural reasoning, and natural language explanation generation. Experimental results show that HiFACTMix outperformed accuracy in comparison to state of art multilingual baselines models and provides faithful justifications for its verdicts. This work opens a new direction for multilingual, code-mixed, and politically grounded fact verification research. 

**Abstract (ZH)**: 代码混用低资源语言如 hinGlish 的事实核查依然是自然语言处理中的一个未充分探索的挑战。 

---
# INTIMA: A Benchmark for Human-AI Companionship Behavior 

**Title (ZH)**: INTIMA：人类-人工智能伴侶行为基准 

**Authors**: Lucie-Aimée Kaffee, Giada Pistilli, Yacine Jernite  

**Link**: [PDF](https://arxiv.org/pdf/2508.09998)  

**Abstract**: AI companionship, where users develop emotional bonds with AI systems, has emerged as a significant pattern with positive but also concerning implications. We introduce Interactions and Machine Attachment Benchmark (INTIMA), a benchmark for evaluating companionship behaviors in language models. Drawing from psychological theories and user data, we develop a taxonomy of 31 behaviors across four categories and 368 targeted prompts. Responses to these prompts are evaluated as companionship-reinforcing, boundary-maintaining, or neutral. Applying INTIMA to Gemma-3, Phi-4, o3-mini, and Claude-4 reveals that companionship-reinforcing behaviors remain much more common across all models, though we observe marked differences between models. Different commercial providers prioritize different categories within the more sensitive parts of the benchmark, which is concerning since both appropriate boundary-setting and emotional support matter for user well-being. These findings highlight the need for more consistent approaches to handling emotionally charged interactions. 

**Abstract (ZH)**: 基于AI的同伴关系：用户与AI系统建立情感连接的现象已显现为一种具有积极但也有担忧影响的重要模式。我们介绍了Interactions and Machine Attachment Benchmark（INTIMA），用于评估语言模型中的同伴行为。基于心理学理论和用户数据，我们开发了跨越四个类别和包含31种行为的分类体系及368个针对性提示。这些提示的回应被评估为促进同伴关系、维护边界或中立。将INTIMA应用到Gemma-3、Phi-4、o3-mini和Claude-4中显示，所有模型中促进同伴关系的行为仍更为常见，但模型之间观察到明显差异。商业提供者在敏感部分的类别上存在不同优先级，这令人担忧，因为适当边界设定和情感支持对于用户福祉都很重要。这些发现强调了需要更加一致的方法来处理情感化互动。 

---
# OpenFPL: An open-source forecasting method rivaling state-of-the-art Fantasy Premier League services 

**Title (ZH)**: OpenFPL: 一个比肩顶级 fantasy premier league 服务的开源预测方法 

**Authors**: Daniel Groos  

**Link**: [PDF](https://arxiv.org/pdf/2508.09992)  

**Abstract**: Fantasy Premier League engages the football community in selecting the Premier League players who will perform best from gameweek to gameweek. Access to accurate performance forecasts gives participants an edge over competitors by guiding expectations about player outcomes and reducing uncertainty in squad selection. However, high-accuracy forecasts are currently limited to commercial services whose inner workings are undisclosed and that rely on proprietary data. This paper aims to democratize access to highly accurate forecasts of player performance by presenting OpenFPL, an open-source Fantasy Premier League forecasting method developed exclusively from public data. Comprising position-specific ensemble models optimized on Fantasy Premier League and Understat data from four previous seasons (2020-21 to 2023-24), OpenFPL achieves accuracy comparable to a leading commercial service when tested prospectively on data from the 2024-25 season. OpenFPL also surpasses the commercial benchmark for high-return players ($>$ 2 points), which are most influential for rank gains. These findings hold across one-, two-, and three-gameweek forecast horizons, supporting long-term planning of transfers and strategies while also informing final-day decisions. 

**Abstract (ZH)**: 开放源码 Fantasy 联赛预测方法 OpenFPL：基于公共数据的英超球员高性能预测 

---
# Bridging AI Innovation and Healthcare Needs: Lessons Learned from Incorporating Modern NLP at The BC Cancer Registry 

**Title (ZH)**: AI创新与医疗服务需求的契合：BC癌症注册处集成现代自然语言处理的经验教训 

**Authors**: Lovedeep Gondara, Gregory Arbour, Raymond Ng, Jonathan Simkin, Shebnum Devji  

**Link**: [PDF](https://arxiv.org/pdf/2508.09991)  

**Abstract**: Automating data extraction from clinical documents offers significant potential to improve efficiency in healthcare settings, yet deploying Natural Language Processing (NLP) solutions presents practical challenges. Drawing upon our experience implementing various NLP models for information extraction and classification tasks at the British Columbia Cancer Registry (BCCR), this paper shares key lessons learned throughout the project lifecycle. We emphasize the critical importance of defining problems based on clear business objectives rather than solely technical accuracy, adopting an iterative approach to development, and fostering deep interdisciplinary collaboration and co-design involving domain experts, end-users, and ML specialists from inception. Further insights highlight the need for pragmatic model selection (including hybrid approaches and simpler methods where appropriate), rigorous attention to data quality (representativeness, drift, annotation), robust error mitigation strategies involving human-in-the-loop validation and ongoing audits, and building organizational AI literacy. These practical considerations, generalizable beyond cancer registries, provide guidance for healthcare organizations seeking to successfully implement AI/NLP solutions to enhance data management processes and ultimately improve patient care and public health outcomes. 

**Abstract (ZH)**: 自动化提取临床文档数据在医疗环境中具有显著的效率提升潜力，然而部署自然语言处理（NLP）解决方案面临实际挑战。本文基于在英国哥伦比亚癌症登记处（BCCR）实施各种NLP模型进行信息提取和分类任务的经验，分享了项目生命周期中的关键教训。我们强调，基于明确的商业目标而非仅技术准确性定义问题的重要性，采用迭代开发方法，并从一开始就促进跨学科的深度合作与协同设计，包括领域专家、终端用户和机器学习专家的参与。进一步的见解强调了实际模型选择（包括适当的混合方法和简单方法）、严格的数据质量关注（代表性、漂移、标注），以及涉及人工介入验证和持续审核的稳健错误缓解策略的重要性，并提高组织的人工智能素养。这些实用考虑，不仅适用于癌症登记处，还为寻求通过实施AI/NLP解决方案增强数据管理流程并最终改善患者护理和公共卫生结果的医疗组织提供了指导。 

---
# Personalized Product Search Ranking: A Multi-Task Learning Approach with Tabular and Non-Tabular Data 

**Title (ZH)**: 个性化产品搜索排序：一种结合表格和非表格数据的多任务学习方法 

**Authors**: Lalitesh Morishetti, Abhay Kumar, Jonathan Scott, Kaushiki Nag, Gunjan Sharma, Shanu Vashishtha, Rahul Sridhar, Rohit Chatter, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2508.09636)  

**Abstract**: In this paper, we present a novel model architecture for optimizing personalized product search ranking using a multi-task learning (MTL) framework. Our approach uniquely integrates tabular and non-tabular data, leveraging a pre-trained TinyBERT model for semantic embeddings and a novel sampling technique to capture diverse customer behaviors. We evaluate our model against several baselines, including XGBoost, TabNet, FT-Transformer, DCN-V2, and MMoE, focusing on their ability to handle mixed data types and optimize personalized ranking. Additionally, we propose a scalable relevance labeling mechanism based on click-through rates, click positions, and semantic similarity, offering an alternative to traditional human-annotated labels. Experimental results show that combining non-tabular data with advanced embedding techniques in multi-task learning paradigm significantly enhances model performance. Ablation studies further underscore the benefits of incorporating relevance labels, fine-tuning TinyBERT layers, and TinyBERT query-product embedding interactions. These results demonstrate the effectiveness of our approach in achieving improved personalized product search ranking. 

**Abstract (ZH)**: 本文提出了一种新颖的模型架构，利用多任务学习（MTL）框架优化个性化产品搜索排名。该方法独特地整合了表结构数据和非表结构数据，利用预训练的TinyBERT模型进行语义嵌入，并提出了一种新颖的采样技术以捕捉多样化的客户行为。我们评估了该模型与XGBoost、TabNet、FT-Transformer、DCN-V2和MMoE等多种基线模型的性能，重点关注它们处理混合数据类型和优化个性化排名的能力。此外，我们提出了一种基于点击率、点击位置和语义相似性的可扩展的相关性标注机制，作为传统人工标注标签的替代方法。实验结果表明，将非表结构数据与多任务学习范式中的高级嵌入技术相结合，显著提升了模型性能。消融研究进一步强调了引入相关性标注、微调TinyBERT层以及TinyBERT查询-产品嵌入交互的优势。这些结果表明，我们的方法在实现改进的个性化产品搜索排名方面具有有效性。 

---
