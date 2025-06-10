# GUI-Reflection: Empowering Multimodal GUI Models with Self-Reflection Behavior 

**Title (ZH)**: GUI-反思：赋予多模态GUI模型自我反思行为能力 

**Authors**: Penghao Wu, Shengnan Ma, Bo Wang, Jiaheng Yu, Lewei Lu, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08012)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown great potential in revolutionizing Graphical User Interface (GUI) automation. However, existing GUI models mostly rely on learning from nearly error-free offline trajectories, thus lacking reflection and error recovery capabilities. To bridge this gap, we propose GUI-Reflection, a novel framework that explicitly integrates self-reflection and error correction capabilities into end-to-end multimodal GUI models throughout dedicated training stages: GUI-specific pre-training, offline supervised fine-tuning (SFT), and online reflection tuning. GUI-reflection enables self-reflection behavior emergence with fully automated data generation and learning processes without requiring any human annotation. Specifically, 1) we first propose scalable data pipelines to automatically construct reflection and error correction data from existing successful trajectories. While existing GUI models mainly focus on grounding and UI understanding ability, we propose the GUI-Reflection Task Suite to learn and evaluate reflection-oriented abilities explicitly. 2) Furthermore, we built a diverse and efficient environment for online training and data collection of GUI models on mobile devices. 3) We also present an iterative online reflection tuning algorithm leveraging the proposed environment, enabling the model to continuously enhance its reflection and error correction abilities. Our framework equips GUI agents with self-reflection and correction capabilities, paving the way for more robust, adaptable, and intelligent GUI automation, with all data, models, environments, and tools to be released publicly. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在革新图形用户界面（GUI）自动化方面展现了巨大的潜力。然而，现有的GUI模型大多依赖于从几乎无错误的离线轨迹中学习，因此缺乏反思和错误恢复能力。为弥补这一不足，我们提出GUI-Reflection，一个新颖的框架，该框架在专用训练阶段明确集成自我反思和错误修正能力：专门的GUI预训练、离线监督微调（SFT）和在线反思调优。GUI-Reflection使自反射行为的出现能够通过完全自动的数据生成和学习过程实现，无需任何人工标注。具体而言，1）我们首次提出可扩展的数据管道，自动生成来自现有成功轨迹的反思和错误修正数据。虽然现有的GUI模型主要集中在语义接地和UI理解能力上，我们提出了GUI-反射任务套件，以明确学习和评估面向反思的能力。2）此外，我们构建了一个多样化和高效的环境，用于移动设备上的GUI模型在线训练和数据收集。3）我们还提出了一种迭代的在线反思调优算法，利用提出的环境，使模型能够不断增强其反思和错误修正能力。我们的框架为GUI代理赋予了自我反思和修正的能力，为更 robust、适应性强和智能的GUI自动化铺平了道路，所有数据、模型、环境和工具将公开发布。 

---
# $τ^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment 

**Title (ZH)**: $τ^2$-Bench：在双控制环境中评估对话代理 

**Authors**: Victor Barres, Honghua Dong, Soham Ray, Xujie Si, Karthik Narasimhan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07982)  

**Abstract**: Existing benchmarks for conversational AI agents simulate single-control environments, where only the AI agent can use tools to interact with the world, while the user remains a passive information provider. This differs from real-world scenarios like technical support, where users need to actively participate in modifying the state of the (shared) world. In order to address this gap, we introduce $\tau^2$-bench, with four key contributions:
1) A novel Telecom dual-control domain modeled as a Dec-POMDP, where both agent and user make use of tools to act in a shared, dynamic environment that tests both agent coordination and communication,
2) A compositional task generator that programmatically creates diverse, verifiable tasks from atomic components, ensuring domain coverage and controlled complexity,
3) A reliable user simulator tightly coupled with the environment, whose behavior is constrained by tools and observable states, improving simulation fidelity,
4) Fine-grained analysis of agent performance through multiple ablations including separating errors arising from reasoning vs communication/coordination.
In particular, our experiments show significant performance drops when agents shift from no-user to dual-control, highlighting the challenges of guiding users. Overall, $\tau^2$-bench provides a controlled testbed for agents that must both reason effectively and guide user actions. 

**Abstract (ZH)**: $\tau^2$-基准：电信双控制域及其实验平台 

---
# Reinforcing Multimodal Understanding and Generation with Dual Self-rewards 

**Title (ZH)**: 增强多模态理解与生成的双重自奖励机制 

**Authors**: Jixiang Hong, Yiran Zhang, Guanzhong Wang, Yi Liu, Ji-Rong Wen, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07963)  

**Abstract**: Building upon large language models (LLMs), recent large multimodal models (LMMs) unify cross-model understanding and generation into a single framework. However, LMMs still struggle to achieve accurate image-text alignment, prone to generating text responses contradicting the visual input or failing to follow the text-to-image prompts. Current solutions require external supervision (e.g., human feedback or reward models) and only address unidirectional tasks-either understanding or generation. In this work, based on the observation that understanding and generation are inverse dual tasks, we introduce a self-supervised dual reward mechanism to reinforce the understanding and generation capabilities of LMMs. Specifically, we sample multiple outputs for a given input in one task domain, then reverse the input-output pairs to compute the dual likelihood of the model as self-rewards for optimization. Extensive experimental results on visual understanding and generation benchmarks demonstrate that our method can effectively enhance the performance of the model without any external supervision, especially achieving remarkable improvements in text-to-image tasks. 

**Abstract (ZH)**: 基于大型语言模型（LLMs），近期的大型多模态模型（LMMs）将跨模型的理解和生成统一在单一框架中。然而，LMMs仍然难以实现准确的图文对齐，容易生成与视觉输入矛盾的文字响应，或者无法遵循文字到图像的提示。当前的解决方案需要外部监督（例如，人工反馈或奖励模型）并且仅涉及单向任务，要么是理解，要么是生成。在观察到理解和生成是互逆的双重任务的基础上，我们引入了一种自监督的双重奖励机制来增强LMMs的理解和生成能力。具体而言，对于给定输入在某一任务域中采样子输出，然后反转输入-输出对计算模型的双重似然性作为自我奖励进行优化。在视觉理解和生成基准上的广泛实验结果表明，我们的方法可以在没有任何外部监督的情况下有效地提升模型性能，特别是在文字到图像任务上取得了显著的改进。 

---
# Gradients: When Markets Meet Fine-tuning -- A Distributed Approach to Model Optimisation 

**Title (ZH)**: 梯度：当市场遇见微调——一种模型优化的分布式方法 

**Authors**: Christopher Subia-Waud  

**Link**: [PDF](https://arxiv.org/pdf/2506.07940)  

**Abstract**: Foundation model fine-tuning faces a fundamental challenge: existing AutoML platforms rely on single optimisation strategies that explore only a fraction of viable hyperparameter configurations. In this white paper, We introduce Gradients, a decentralised AutoML platform that transforms hyperparameter optimisation into a competitive marketplace where independent miners compete to discover optimal configurations. Economic incentives align individual exploration with collective optimisation goals, driving systematic investigation of hyperparameter regions that centralised methods miss. We evaluate our approach across 180 controlled experiments spanning diverse model architectures (70M to 70B parameters) and task types. Gradients achieves an 82.8\% win rate against HuggingFace AutoTrain and 100\% against TogetherAI, Databricks, and Google Cloud, with mean improvements of 11.8\% and 42.1\% respectively. Complex reasoning and retrieval tasks show particularly strong gains of 30-40\%, whilst diffusion models achieve 23.4\% improvements for person-specific generation. These results demonstrate that competitive, economically-driven approaches can systematically discover superior configurations that centralised AutoML consistently miss. 

**Abstract (ZH)**: 基于模型的微调面临一个根本性的挑战：现有的AutoML平台依赖于单一优化策略，只探索了可行的超参数配置的一小部分。在本白皮书中，我们介绍了一个去中心化的AutoML平台Gradients，它将超参数优化转变为一个竞争性市场，独立的挖掘者在这竞争市场中竞争以发现最优配置。经济激励将个体探索与集体优化目标对齐，驱动对中央化方法忽略的超参数区域进行系统性调查。我们通过涵盖多样模型架构（从70M到70B参数）和任务类型的180个受控实验评估了该方法。Gradients在对抗HuggingFace AutoTrain时取得82.8%的胜率，并在对抗TogetherAI、Databricks和Google Cloud时取得100%的胜率，分别平均提高了11.8%和42.1%。复杂的推理和检索任务显示出特别显著的20-40%的增益，而扩散模型在人物特定生成方面实现了23.4%的改进。这些结果表明，竞争性和经济驱动的方法能够系统性地发现中央化AutoML经常忽略的更优配置。 

---
# Solving Inequality Proofs with Large Language Models 

**Title (ZH)**: 用大型语言模型解决不等式证明问题 

**Authors**: Jiayi Sheng, Luna Lyu, Jikai Jin, Tony Xia, Alex Gu, James Zou, Pan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07927)  

**Abstract**: Inequality proving, crucial across diverse scientific and mathematical fields, tests advanced reasoning skills such as discovering tight bounds and strategic theorem application. This makes it a distinct, demanding frontier for large language models (LLMs), offering insights beyond general mathematical problem-solving. Progress in this area is hampered by existing datasets that are often scarce, synthetic, or rigidly formal. We address this by proposing an informal yet verifiable task formulation, recasting inequality proving into two automatically checkable subtasks: bound estimation and relation prediction. Building on this, we release IneqMath, an expert-curated dataset of Olympiad-level inequalities, including a test set and training corpus enriched with step-wise solutions and theorem annotations. We also develop a novel LLM-as-judge evaluation framework, combining a final-answer judge with four step-wise judges designed to detect common reasoning flaws. A systematic evaluation of 29 leading LLMs on IneqMath reveals a surprising reality: even top models like o1 achieve less than 10% overall accuracy under step-wise scrutiny; this is a drop of up to 65.5% from their accuracy considering only final answer equivalence. This discrepancy exposes fragile deductive chains and a critical gap for current LLMs between merely finding an answer and constructing a rigorous proof. Scaling model size and increasing test-time computation yield limited gains in overall proof correctness. Instead, our findings highlight promising research directions such as theorem-guided reasoning and self-refinement. Code and data are available at this https URL. 

**Abstract (ZH)**: 不等式证明：跨多个科学和数学领域的关键任务，测试高级推理能力如发现紧界和战略性定理应用。这使其成为大型语言模型（LLMs）的一个独特且具挑战性的前沿领域，提供超越一般数学问题解决的见解。现有数据集的稀缺性、合成性或形式上的刚性阻碍了这一领域的发展。我们通过提出一种非正式但可验证的任务形式来应对这一挑战，将不等式证明重新构想为两个可自动验证的子任务：界估计和关系预测。在此基础上，我们发布了IneqMath专家精选数据集，包括奥数级不等式测试集和训练语料库，其中包含逐步解决方案和定理注释。我们还开发了一种新颖的LLM作为评判者的评估框架，结合最终答案评判器和四种逐步评判器以检测常见推理错误。系统性评估29个领先的大模型在IneqMath的表现揭示了一个意想不到的事实：即使顶尖模型o1在逐步审查下的总体准确率也低于10%；与仅考虑最终答案等价相比，这一准确率下降高达65.5%。这种差异揭示了脆弱的演绎链和当前大模型在仅仅找到答案和构建严谨证明之间存在的关键差距。通过扩大模型规模和增加测试时计算量在总体证明正确性上仅带来有限的提升。相反，我们的研究结果强调了诸如定理引导推理和自我完善等潜在的研究方向。代码和数据可在以下链接获取。 

---
# LUCIFER: Language Understanding and Context-Infused Framework for Exploration and Behavior Refinement 

**Title (ZH)**: LUCIFER：语言理解和情境融合的探索与行为优化框架 

**Authors**: Dimitris Panagopoulos, Adolfo Perrusquia, Weisi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.07915)  

**Abstract**: In dynamic environments, the rapid obsolescence of pre-existing environmental knowledge creates a gap between an agent's internal model and the evolving reality of its operational context. This disparity between prior and updated environmental valuations fundamentally limits the effectiveness of autonomous decision-making. To bridge this gap, the contextual bias of human domain stakeholders, who naturally accumulate insights through direct, real-time observation, becomes indispensable. However, translating their nuanced, and context-rich input into actionable intelligence for autonomous systems remains an open challenge. To address this, we propose LUCIFER (Language Understanding and Context-Infused Framework for Exploration and Behavior Refinement), a domain-agnostic framework that integrates a hierarchical decision-making architecture with reinforcement learning (RL) and large language models (LLMs) into a unified system. This architecture mirrors how humans decompose complex tasks, enabling a high-level planner to coordinate specialised sub-agents, each focused on distinct objectives and temporally interdependent actions. Unlike traditional applications where LLMs are limited to single role, LUCIFER integrates them in two synergistic roles: as context extractors, structuring verbal stakeholder input into domain-aware representations that influence decision-making through an attention space mechanism aligning LLM-derived insights with the agent's learning process, and as zero-shot exploration facilitators guiding the agent's action selection process during exploration. We benchmark various LLMs in both roles and demonstrate that LUCIFER improves exploration efficiency and decision quality, outperforming flat, goal-conditioned policies. Our findings show the potential of context-driven decision-making, where autonomous systems leverage human contextual knowledge for operational success. 

**Abstract (ZH)**: 在动态环境中，预存的环境知识的迅速过时会导致代理内部模型与其操作背景不断演化的现实之间产生差距。这种先验与更新后的环境价值之间的差异从根本上限制了自主决策的有效性。为了弥合这一差距，人类领域利益相关者基于直接的、实时的观察自然积累的见解变得不可或缺。然而，将他们细微的、富含上下文的输入转化为可用于自主系统的可操作情报仍是一个开放的挑战。为了应对这一挑战，我们提出了LUCIFER（基于语言理解和上下文融入的探索与行为优化框架），这是一种领域无关的框架，将层次决策架构、强化学习（RL）和大型语言模型（LLMs）整合到一个统一系统中。该架构模仿了人类如何分解复杂任务的方式，使得高级规划者能够协调专注于不同目标和时间上相互依赖行动的专业子代理。与传统应用中LLMs仅限于单一角色不同，LUCIFER将它们整合为两个协同作用的角色：作为上下文提取器，将口头利益相关者的输入结构化为领域感知的表示，通过一种注意空间机制影响决策，使LUCIFER生成的见解与代理的学习过程相契合；以及作为零样本探索促进者，在探索过程中引导代理的动作选择过程。我们在两个角色中基准测试了各种LLM，并证明了LUCIFER提高了探索效率和决策质量，优于平铺的目标条件策略。我们的研究结果表明，基于上下文的决策在自主系统利用人类上下文知识实现操作成功方面具有潜力。 

---
# Evaluating Large Language Models on the Frame and Symbol Grounding Problems: A Zero-shot Benchmark 

**Title (ZH)**: 评估大型语言模型的框架和符号接地问题：一个零样本基准 

**Authors**: Shoko Oka  

**Link**: [PDF](https://arxiv.org/pdf/2506.07896)  

**Abstract**: Recent advancements in large language models (LLMs) have revitalized philosophical debates surrounding artificial intelligence. Two of the most fundamental challenges - namely, the Frame Problem and the Symbol Grounding Problem - have historically been viewed as unsolvable within traditional symbolic AI systems. This study investigates whether modern LLMs possess the cognitive capacities required to address these problems. To do so, I designed two benchmark tasks reflecting the philosophical core of each problem, administered them under zero-shot conditions to 13 prominent LLMs (both closed and open-source), and assessed the quality of the models' outputs across five trials each. Responses were scored along multiple criteria, including contextual reasoning, semantic coherence, and information filtering. The results demonstrate that while open-source models showed variability in performance due to differences in model size, quantization, and instruction tuning, several closed models consistently achieved high scores. These findings suggest that select modern LLMs may be acquiring capacities sufficient to produce meaningful and stable responses to these long-standing theoretical challenges. 

**Abstract (ZH)**: 最近大型语言模型的进展重新引发了对人工智能的哲学争论。两种最基本的问题——框架问题和符号接地问题——历来被视为传统符号AI系统无法解决的问题。本研究考察了现代大型语言模型是否具备解决这些问题所需的认知能力。为此，我设计了两个反映每个问题哲学核心的基准任务，在零样本条件下交给了13个知名的大规模语言模型（包括闭源和开源模型），并在五次试验中评估了模型输出的质量。根据上下文推理、语义连贯性和信息过滤等多个标准对响应进行了评分。结果表明，开源模型在性能上表现出一定的变异性，这归因于模型大小、量化和指令调优的差异，但一些闭源模型始终取得了高分。这些发现表明，某些现代大型语言模型可能正在获取足以产生对这些长期理论挑战有意义且稳定响应的能力。 

---
# A Temporal FRBR/FRBRoo-Based Model for Component-Level Versioning of Legal Norms 

**Title (ZH)**: 基于FRBR/FRBRoo的时间维度组件级法规版本化模型 

**Authors**: Hudson de Martim  

**Link**: [PDF](https://arxiv.org/pdf/2506.07853)  

**Abstract**: Effectively representing legal norms for automated processing is a critical challenge, particularly in tracking the diachronic evolution of their hierarchical components (e.g., articles, paragraphs). While foundational frameworks like FRBR/FRBRoo and standards like Akoma Ntoso model legal documents at a macro level, they lack native mechanisms for granular, component-level versioning. This limitation hinders the deterministic point-in-time reconstruction of legal texts, a fundamental capability for reliable Legal Tech and AI applications. This paper proposes a structured, temporal model that extends the FRBRoo framework to address this gap. It introduces specialized subclasses of Expressio - Temporal Version (TV) and Language Version (LV - to represent the state of a legal norm and its linguistic variations at specific points in time. The model applies this same paradigm hierarchically, introducing Component Work (CW), Component Temporal Version (CTV), and Component Language Version (CLV) to track the lifecycle of individual articles, paragraphs, and clauses. Using the Brazilian Federal Constitution as a case study, the paper demonstrates how each amendment creates new Component Temporal Versions for affected provisions, while unaffected components retain their existing versions. This fine-grained, time-aware architecture enables the precise, deterministic retrieval and reconstruction of any part of a legal text as it existed on a specific date. The model provides a robust foundation for developing advanced legal information systems, knowledge graphs, and AI tools capable of accurate historical analysis and impact assessment, overcoming the limitations of current generative models. 

**Abstract (ZH)**: 有效表示法律规范以实现自动化处理是关键挑战，特别是在追踪其层级组件（例如，条款、段落）的历时演变方面。虽然FRBR/FRBRoo等基础框架和Akoma Ntoso等标准在宏观层面建模法律文件，但缺乏细粒度的组件级版本控制机制。这一限制阻碍了对法律文本在特定时间点的确定性重构，这对于可靠的Legal Tech和AI应用是基本能力。本文提出了一种结构化的时间模型，扩展了FRBRoo框架以解决这一缺口。该模型引入了专门的Expressio - 时间版本（TV）和语言版本（LV）子类，用于表示特定时间点的法律规范及其语言变化状态。模型采用分层范式，引入了组件工作（CW）、组件时间版本（CTV）和组件语言版本（CLV），以跟踪单个条款、段落和条款的生命周期。以巴西联邦宪法为例，本文展示了每次修正案如何为受影响的条款创建新的组件时间版本，而未受影响的组件则保留其现有版本。这种细粒度、时间感知的架构允许精确、确定地检索和重构特定日期存在的任何法律文本部分。该模型为开发先进的法律信息系统、知识图谱和能够进行准确历史分析和影响评估的AI工具提供了坚实基础，克服了当前生成模型的限制。 

---
# HAIBU-ReMUD: Reasoning Multimodal Ultrasound Dataset and Model Bridging to General Specific Domains 

**Title (ZH)**: HAIBU-ReMUD: 推理多模态超声数据集和模型连接到通用特定领域 

**Authors**: Shijie Wang, Yilun Zhang, Zeyu Lai, Dexing Kong  

**Link**: [PDF](https://arxiv.org/pdf/2506.07837)  

**Abstract**: Multimodal large language models (MLLMs) have shown great potential in general domains but perform poorly in some specific domains due to a lack of domain-specific data, such as image-text data or vedio-text data. In some specific domains, there is abundant graphic and textual data scattered around, but lacks standardized arrangement. In the field of medical ultrasound, there are ultrasonic diagnostic books, ultrasonic clinical guidelines, ultrasonic diagnostic reports, and so on. However, these ultrasonic materials are often saved in the forms of PDF, images, etc., and cannot be directly used for the training of MLLMs. This paper proposes a novel image-text reasoning supervised fine-tuning data generation pipeline to create specific domain quadruplets (image, question, thinking trace, and answer) from domain-specific materials. A medical ultrasound domain dataset ReMUD is established, containing over 45,000 reasoning and non-reasoning supervised fine-tuning Question Answering (QA) and Visual Question Answering (VQA) data. The ReMUD-7B model, fine-tuned on Qwen2.5-VL-7B-Instruct, outperforms general-domain MLLMs in medical ultrasound field. To facilitate research, the ReMUD dataset, data generation codebase, and ReMUD-7B parameters will be released at this https URL, addressing the data shortage issue in specific domain MLLMs. 

**Abstract (ZH)**: 多模态大型语言模型在特定领域中的图像-文本推理监督微调数据生成管道：ReMUD数据集及其在医疗超声领域的应用 

---
# Addition in Four Movements: Mapping Layer-wise Information Trajectories in LLMs 

**Title (ZH)**: 四步添加法：映射LLMs的逐层信息轨迹 

**Authors**: Yao Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07824)  

**Abstract**: Multi-digit addition is a clear probe of the computational power of large language models. To dissect the internal arithmetic processes in LLaMA-3-8B-Instruct, we combine linear probing with logit-lens inspection. Inspired by the step-by-step manner in which humans perform addition, we propose and analyze a coherent four-stage trajectory in the forward pass:Formula-structure representations become linearly decodable first, while the answer token is still far down the candidate this http URL computational features then emerge this http URL deeper activation layers, numerical abstractions of the result become clearer, enabling near-perfect detection and decoding of the individual digits in the this http URL the output, the model organizes and generates the final content, with the correct token reliably occupying the top this http URL trajectory suggests a hierarchical process that favors internal computation over rote memorization. We release our code and data to facilitate reproducibility. 

**Abstract (ZH)**: 多位数加法是评估大型语言模型计算能力的清晰指标。为了剖析LLaMA-3-8B-Instruct中的内部算术过程，我们结合线性探针与logit-lens检查。受人类进行加法运算逐步方式进行启发，我们提出并分析了一个前后连贯的四阶段前向传递轨迹：公式结构表示首先线性可解，但在候选答案中答案标记仍然很远；随后计算特征在更深的激活层中涌现；最终，结果的数值抽象更加清晰，使模型能够近乎完美地检测和解码输出中的各个数字；在输出阶段，模型组织并生成最终内容，正确标记可靠地占据首位。该轨迹表明了一个分层过程，更侧重于内部计算而非机械记忆。我们发布了代码和数据以促进可再现性。 

---
# Guideline Forest: Experience-Induced Multi-Guideline Reasoning with Stepwise Aggregation 

**Title (ZH)**: 经验诱导多准则逐步聚合的森林指南：Stepwise Aggregation in Guideline Forest: Experience-Induced Multi-Guideline Reasoning 

**Authors**: Jiaxiang CHen, Zhuo Wang, Mingxi Zou, Qifan Wang, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07820)  

**Abstract**: Human reasoning is flexible, adaptive, and grounded in prior experience-qualities that large language models (LLMs) still struggle to emulate. Existing methods either explore diverse reasoning paths at inference time or search for optimal workflows through expensive operations, but both fall short in leveraging multiple reusable strategies in a structured, efficient manner. We propose Guideline Forest, a framework that enhances LLMs reasoning by inducing structured reasoning strategies-called guidelines-from verified examples and executing them via step-wise aggregation. Unlike test-time search or single-path distillation, our method draws on verified reasoning experiences by inducing reusable guidelines and expanding each into diverse variants. Much like human reasoning, these variants reflect alternative thought patterns, are executed in parallel, refined via self-correction, and aggregated step by step-enabling the model to adaptively resolve uncertainty and synthesize robust this http URL evaluate Guideline Forest on four benchmarks-GSM8K, MATH-500, MBPP, and HumanEval-spanning mathematical and programmatic reasoning. Guideline Forest consistently outperforms strong baselines, including CoT, ReAct, ToT, FoT, and AFlow. Ablation studies further highlight the effectiveness of multi-path reasoning and stepwise aggregation, underscoring the Guideline Forest's adaptability and generalization potential. 

**Abstract (ZH)**: 大型语言模型在灵活、适应性和基于先前经验的推理方面仍难以模仿。现有的方法要么在推理时探索多种推理路径，要么通过昂贵的操作搜索最优工作流程，但两者都无法有效地整合多种可重用的策略。我们提出了Guideline Forest框架，该框架通过诱导结构化的推理策略（称为指导原则）并执行这些策略来增强大型语言模型的推理能力，执行过程通过逐步聚合进行。与测试时的搜索或单一路径的知识蒸馏不同，我们的方法通过诱导可重用的推理经验和扩展每个经验为多种变体来利用验证的推理经验。这些变体反映了不同的思维模式，可以并行执行，通过自我纠正进行优化，并逐步聚合，从而使模型能够适应性地解决不确定性并综合出稳健的结果。我们在涵盖数学和程序推理的四个基准GSM8K、MATH-500、MBPP和HumanEval上评估了Guideline Forest。Guideline Forest在这些基准上的表现优于包括CoT、ReAct、ToT、FoT和AFlow在内的强基线。消融研究进一步证明了多路径推理和逐步聚合的有效性，突显了Guideline Forest的适应性和泛化潜力。 

---
# A Proposal to Extend the Common Model of Cognition with Metacognition 

**Title (ZH)**: 拟提出将元认知扩展到常用认知模型中 

**Authors**: John Laird, Christian Lebiere, Paul Rosenbloom, Andrea Stocco, Robert Wray  

**Link**: [PDF](https://arxiv.org/pdf/2506.07807)  

**Abstract**: The Common Model of Cognition (CMC) provides an abstract characterization of the structure and processing required by a cognitive architecture for human-like minds. We propose a unified approach to integrating metacognition within the CMC. We propose that metacognition involves reasoning over explicit representations of an agent's cognitive capabilities and processes in working memory. Our proposal exploits the existing cognitive capabilities of the CMC, making minimal extensions in the structure and information available within working memory. We provide examples of metacognition within our proposal. 

**Abstract (ZH)**: 基于认知的通用模型（CMC）提供了一种抽象化的人类思维所需的认知架构的结构和处理特征。我们提出了一种在CMC中集成元认知的统一方法。我们建议元认知涉及在工作记忆中对代理认知能力及过程的显式表示进行推理。我们的建议利用了CMC现有的认知能力，仅在工作记忆的结构和可用信息上做出最小扩展。我们提供了元认知在我们提案中的示例。 

---
# REMoH: A Reflective Evolution of Multi-objective Heuristics approach via Large Language Models 

**Title (ZH)**: REMoH：借助大型语言模型的多目标启发式方法的反思性演变 

**Authors**: Diego Forniés-Tabuenca, Alejandro Uribe, Urtzi Otamendi, Arkaitz Artetxe, Juan Carlos Rivera, Oier Lopez de Lacalle  

**Link**: [PDF](https://arxiv.org/pdf/2506.07759)  

**Abstract**: Multi-objective optimization is fundamental in complex decision-making tasks. Traditional algorithms, while effective, often demand extensive problem-specific modeling and struggle to adapt to nonlinear structures. Recent advances in Large Language Models (LLMs) offer enhanced explainability, adaptability, and reasoning. This work proposes Reflective Evolution of Multi-objective Heuristics (REMoH), a novel framework integrating NSGA-II with LLM-based heuristic generation. A key innovation is a reflection mechanism that uses clustering and search-space reflection to guide the creation of diverse, high-quality heuristics, improving convergence and maintaining solution diversity. The approach is evaluated on the Flexible Job Shop Scheduling Problem (FJSSP) in-depth benchmarking against state-of-the-art methods using three instance datasets: Dauzere, Barnes, and Brandimarte. Results demonstrate that REMoH achieves competitive results compared to state-of-the-art approaches with reduced modeling effort and enhanced adaptability. These findings underscore the potential of LLMs to augment traditional optimization, offering greater flexibility, interpretability, and robustness in multi-objective scenarios. 

**Abstract (ZH)**: 多目标优化是复杂决策任务中的基础。传统算法虽然有效，但往往需要大量的问题特定建模，并且难以适应非线性结构。近年来，大型语言模型（LLMs）的进步提供了增强的可解释性、适应性和推理能力。本文提出了一种名为反射式多目标启发式进化的框架（REMoH），该框架将NSGA-II与基于LLM的启发式生成相结合。一个关键创新是反射机制，利用聚类和搜索空间反射来引导多样且高质量启发式的生成，从而提高收敛性和保持解的多样性。该方法在详细基准测试中，使用Dauzere、Barnes和Brandimarte的三个实例数据集，与先进方法对比进行了灵活作业车间调度问题（FJSSP）的评估。结果表明，REMoH在减少建模 effort 和增强适应性方面与先进方法具有竞争力。这些发现强调了LLMs在增强传统优化方面的潜力，提供了在多目标场景中更大的灵活性、可解释性和稳健性。 

---
# Agent Semantics, Semantic Spacetime, and Graphical Reasoning 

**Title (ZH)**: 代理语义、语义时空与图形推理 

**Authors**: Mark Burgess  

**Link**: [PDF](https://arxiv.org/pdf/2506.07756)  

**Abstract**: Some formal aspects of the Semantic Spacetime graph model are presented, with reference to its use for directed knowledge representations and process modelling. A finite $\gamma(3,4)$ representation is defined to form a closed set of operations that can scale to any degree of semantic complexity. The Semantic Spacetime postulates bring predictability with minimal constraints to pathways in graphs. The ubiquitous appearance of absorbing states in any partial graph means that a graph process leaks information. The issue is closely associated with the issue of division by zero, which signals a loss of closure and the need for manual injection of remedial information. The Semantic Spacetime model (and its Promise Theory) origins help to clarify how such absorbing states are associated with boundary information where intentionality can enter. 

**Abstract (ZH)**: 语义时空图模型的一些形式化方面及其在定向知识表示和过程建模中的应用 

---
# RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards 

**Title (ZH)**: RSafe: 通过促进主动推理来构建稳健且适应性的LLM防护措施 

**Authors**: Jingnan Zheng, Xiangtian Ji, Yijun Lu, Chenhang Cui, Weixiang Zhao, Gelei Deng, Zhenkai Liang, An Zhang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.07736)  

**Abstract**: Large Language Models (LLMs) continue to exhibit vulnerabilities despite deliberate safety alignment efforts, posing significant risks to users and society. To safeguard against the risk of policy-violating content, system-level moderation via external guard models-designed to monitor LLM inputs and outputs and block potentially harmful content-has emerged as a prevalent mitigation strategy. Existing approaches of training guard models rely heavily on extensive human curated datasets and struggle with out-of-distribution threats, such as emerging harmful categories or jailbreak attacks. To address these limitations, we propose RSafe, an adaptive reasoning-based safeguard that conducts guided safety reasoning to provide robust protection within the scope of specified safety policies. RSafe operates in two stages: 1) guided reasoning, where it analyzes safety risks of input content through policy-guided step-by-step reasoning, and 2) reinforced alignment, where rule-based RL optimizes its reasoning paths to align with accurate safety prediction. This two-stage training paradigm enables RSafe to internalize safety principles to generalize safety protection capability over unseen or adversarial safety violation scenarios. During inference, RSafe accepts user-specified safety policies to provide enhanced safeguards tailored to specific safety requirements. 

**Abstract (ZH)**: 大型语言模型（LLMs）在安全对齐努力后仍表现出漏洞，对用户和社会构成了重大风险。为了防止政策违规内容的风险，通过外部护盾模型进行系统级 moderation——设计用来监控LLM的输入和输出并阻止潜在有害内容——已成为一种普遍的缓解策略。现有的护盾模型训练方法高度依赖大量人工标注的数据集，并且在处理新兴有害类别或监狱突破攻击等离分布威胁方面存在困难。为了解决这些限制，我们提出了基于自适应推理的RSafe，一种进行引导安全推理以在指定安全策略范围内提供稳健保护的防护机制。RSafe分为两个阶段：1) 引导推理，通过策略引导的逐步推理分析输入内容的安全风险；2) 强化对齐，基于规则的强化学习优化其推理路径以与准确的安全预测对齐。这种两阶段训练范式使RSafe能够在未见过的或对抗性安全违规场景中内化安全原则并泛化安全保护能力。在推理过程中，RSafe接受用户指定的安全政策以提供针对特定安全需求的增强保护。 

---
# NeurIPS 2025 E2LM Competition : Early Training Evaluation of Language Models 

**Title (ZH)**: Ne.signIn.2025 E2LM竞赛：语言模型早期训练评估 

**Authors**: Mouadh Yagoubi, Yasser Dahou, Billel Mokeddem, Younes Belkada, Phuc H. Le-Khac, Basma El Amel Boussaha, Reda Alami, Jingwei Zuo, Damiano Marsili, Mugariya Farooq, Mounia Lalmas, Georgia Gkioxari, Patrick Gallinari, Philip Torr, Hakim Hacid  

**Link**: [PDF](https://arxiv.org/pdf/2506.07731)  

**Abstract**: Existing benchmarks have proven effective for assessing the performance of fully trained large language models. However, we find striking differences in the early training stages of small models, where benchmarks often fail to provide meaningful or discriminative signals. To explore how these differences arise, this competition tackles the challenge of designing scientific knowledge evaluation tasks specifically tailored for measuring early training progress of language models. Participants are invited to develop novel evaluation methodologies or adapt existing benchmarks to better capture performance differences among language models. To support this effort, we provide three pre-trained small models (0.5B, 1B, and 3B parameters), along with intermediate checkpoints sampled during training up to 200B tokens. All experiments and development work can be run on widely available free cloud-based GPU platforms, making participation accessible to researchers with limited computational resources. Submissions will be evaluated based on three criteria: the quality of the performance signal they produce, the consistency of model rankings at 1 trillion tokens of training, and their relevance to the scientific knowledge domain. By promoting the design of tailored evaluation strategies for early training, this competition aims to attract a broad range of participants from various disciplines, including those who may not be machine learning experts or have access to dedicated GPU resources. Ultimately, this initiative seeks to make foundational LLM research more systematic and benchmark-informed from the earliest phases of model development. 

**Abstract (ZH)**: 现有的基准测试已证明对评估完全训练的大语言模型性能是有效的。然而，我们发现小型模型的早期训练阶段存在显著差异，在这些阶段，基准测试往往无法提供有意义或区分性的信号。为了探索这些差异的来源，此次竞赛旨在设计专门针对衡量语言模型早期训练进度的科学知识评估任务。参与者被邀请开发新的评估方法或适应现有的基准测试，以更好地捕捉不同语言模型之间的性能差异。为此，我们提供了三个预训练的小型模型（0.5B、1B和3B参数），以及训练过程中抽取的多达200B个标记的中间检查点。所有的实验和开发工作都可以在广泛可用的免费云GPU平台上运行，从而使得计算资源有限的研究人员也能参与其中。提交将根据三个标准进行评估：生成的性能信号的质量、在训练1万亿个标记时模型排名的一致性以及对科学知识领域的相关性。通过促进针对早期训练的定制评估策略设计，此次竞赛旨在吸引来自各个学科的广泛参与者，包括那些可能不是机器学习专家或没有专用GPU资源的研究人员。最终，这一举措旨在使基础大语言模型研究从模型开发的最早阶段起就更加系统和基准导向。 

---
# MCPWorld: A Unified Benchmarking Testbed for API, GUI, and Hybrid Computer Use Agents 

**Title (ZH)**: MCPWorld: 一体化API、GUI及混合计算机使用代理基准测试平台 

**Authors**: Yunhe Yan, Shihe Wang, Jiajun Du, Yexuan Yang, Yuxuan Shan, Qichen Qiu, Xianqing Jia, Xinge Wang, Xin Yuan, Xu Han, Mao Qin, Yinxiao Chen, Chen Peng, Shangguang Wang, Mengwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07672)  

**Abstract**: (M)LLM-powered computer use agents (CUA) are emerging as a transformative technique to automate human-computer interaction. However, existing CUA benchmarks predominantly target GUI agents, whose evaluation methods are susceptible to UI changes and ignore function interactions exposed by application APIs, e.g., Model Context Protocol (MCP). To this end, we propose MCPWorld, the first automatic CUA testbed for API, GUI, and API-GUI hybrid agents. A key principle of MCPWorld is the use of "white-box apps", i.e., those with source code availability and can be revised/re-compiled as needed (e.g., adding MCP support), with two notable advantages:
(1) It greatly broadens the design space of CUA, such as what and how the app features to be exposed/extracted as CUA-callable APIs.
(2) It allows MCPWorld to programmatically verify task completion by directly monitoring application behavior through techniques like dynamic code instrumentation, offering robust, accurate CUA evaluation decoupled from specific agent implementations or UI states.
Currently, MCPWorld includes 201 well curated and annotated user tasks, covering diversified use cases and difficulty levels. MCPWorld is also fully containerized with GPU acceleration support for flexible adoption on different OS/hardware environments. Our preliminary experiments, using a representative LLM-powered CUA framework, achieve 75.12% task completion accuracy, simultaneously providing initial evidence on the practical effectiveness of agent automation leveraging MCP. Overall, we anticipate MCPWorld to facilitate and standardize the benchmarking of next-generation computer use agents that can leverage rich external tools. Our code and dataset are publicly available at this https URL. 

**Abstract (ZH)**: 基于(M)LLM的计算机使用代理(CUA)正在成为自动化人机交互的一种变革性技术。然而，现有的CUA基准主要针对GUI代理，其评估方法易受UI变化影响且忽略由应用程序API暴露的功能交互，例如Model Context Protocol (MCP)。为解决这一问题，我们提出了MCPWorld，这是首个针对API、GUI及API-GUI混合代理的自动CUA测试平台。MCPWorld的关键原则是使用“白盒应用程序”，即具有可用源代码且可根据需要进行修改/重新编译的应用程序（例如，增加MCP支持），具有以下两个显著优势：
(1) 它大大扩展了CUA的设计空间，例如应用程序特性的展示/提取作为CUA可调用的API的方式和内容。
(2) 它允许MCPWorld通过动态代码插桩等技术直接监控应用程序行为来进行程序化验证任务完成情况，提供与特定代理实现或UI状态无关的稳健、准确的CUA评估。
目前，MCPWorld包含201个精心挑选和注释的用户任务，涵盖多样化用例和难度级别。MCPWorld还完全容器化，支持GPU加速，便于在不同的操作系统/硬件环境中灵活采用。初步实验使用一个代表性的基于( M)LLM的CUA框架，实现了75.12%的任务完成准确率，同时为基于MCP的代理自动化具备实际效果提供了初步证据。总体而言，我们期望MCPWorld能够促进和规范下一代能够利用丰富外部工具的计算机使用代理的基准测试。我们的代码和数据集已在以下网址公开：这个 https URL。 

---
# SWE-Dev: Building Software Engineering Agents with Training and Inference Scaling 

**Title (ZH)**: SWE-Dev: 基于训练和推理扩展构建软件工程代理 

**Authors**: Haoran Wang, Zhenyu Hou, Yao Wei, Jie Tang, Yuxiao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.07636)  

**Abstract**: Large language models (LLMs) have advanced rapidly from conversational problem solving to addressing real-world tasks involving tool use, such as software engineering (SWE). Recent LLM-powered toolkits, such as OpenAI Codex and Cursor, have offered end-to-end automation of the software development process. However, building effective SWE agents remains challenging due to the lack of high-quality training data and effective test cases. To address this issue, we present SWE-Dev, an SWE agent built upon open-source LLMs. First, we develop a robust pipeline to synthesize test cases for patch evaluation. Second, we scale up agent trajectories to construct the training data for building SWE-Dev. Experiments on the SWE-bench-Verified benchmark show that the SWE-Dev models can achieve top performance among all open SWE agents. Specifically, the success rates of the SWE-Dev 7B and 32B parameter models reach 23.4% and 36.6%, respectively, outperforming state-of-the-art open-source models. All code, models, and datasets are publicly available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在软件工程中的应用从对话式问题解决迅速发展到了涉及工具使用的真实世界任务。近期基于LLMs的工具包，如OpenAI Codex和Cursor，提供了软件开发过程的端到端自动化。然而，由于缺乏高质量的训练数据和有效的测试案例，构建有效的软件工程（SWE）代理仍然面临挑战。为了解决这一问题，我们提出了SWE-Dev，这是一种基于开源LLMs构建的SWE代理。首先，我们开发了一个 robust 的流水线来合成补丁评估的测试案例。其次，我们扩展了代理轨迹以构建构建SWE-Dev的训练数据。在SWE-bench-Verified基准上的实验表明，SWE-Dev模型在所有开源SWE代理中达到了最佳性能。具体来说，SWE-Dev 7B和32B参数模型的成功率分别达到了23.4%和36.6%，优于最先进的开源模型。所有代码、模型和数据集均可在如下链接公开获取：this https URL。 

---
# Automating Exploratory Multiomics Research via Language Models 

**Title (ZH)**: 通过语言模型自动化探索性多组学研究 

**Authors**: Shang Qu, Ning Ding, Linhai Xie, Yifei Li, Zaoqu Liu, Kaiyan Zhang, Yibai Xiong, Yuxin Zuo, Zhangren Chen, Ermo Hua, Xingtai Lv, Youbang Sun, Yang Li, Dong Li, Fuchu He, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.07591)  

**Abstract**: This paper introduces PROTEUS, a fully automated system that produces data-driven hypotheses from raw data files. We apply PROTEUS to clinical proteogenomics, a field where effective downstream data analysis and hypothesis proposal is crucial for producing novel discoveries. PROTEUS uses separate modules to simulate different stages of the scientific process, from open-ended data exploration to specific statistical analysis and hypothesis proposal. It formulates research directions, tools, and results in terms of relationships between biological entities, using unified graph structures to manage complex research processes. We applied PROTEUS to 10 clinical multiomics datasets from published research, arriving at 360 total hypotheses. Results were evaluated through external data validation and automatic open-ended scoring. Through exploratory and iterative research, the system can navigate high-throughput and heterogeneous multiomics data to arrive at hypotheses that balance reliability and novelty. In addition to accelerating multiomic analysis, PROTEUS represents a path towards tailoring general autonomous systems to specialized scientific domains to achieve open-ended hypothesis generation from data. 

**Abstract (ZH)**: PROTEUS：一种用于从原始数据文件生成数据驱动假设的全自动系统 

---
# SAFEFLOW: A Principled Protocol for Trustworthy and Transactional Autonomous Agent Systems 

**Title (ZH)**: SAFEFLOW：可信赖且事务性的自主代理系统原理协议 

**Authors**: Peiran Li, Xinkai Zou, Zhuohang Wu, Ruifeng Li, Shuo Xing, Hanwen Zheng, Zhikai Hu, Yuping Wang, Haoxi Li, Qin Yuan, Yingmo Zhang, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07564)  

**Abstract**: Recent advances in large language models (LLMs) and vision-language models (VLMs) have enabled powerful autonomous agents capable of complex reasoning and multi-modal tool use. Despite their growing capabilities, today's agent frameworks remain fragile, lacking principled mechanisms for secure information flow, reliability, and multi-agent coordination. In this work, we introduce SAFEFLOW, a new protocol-level framework for building trustworthy LLM/VLM-based agents. SAFEFLOW enforces fine-grained information flow control (IFC), precisely tracking provenance, integrity, and confidentiality of all the data exchanged between agents, tools, users, and environments. By constraining LLM reasoning to respect these security labels, SAFEFLOW prevents untrusted or adversarial inputs from contaminating high-integrity decisions. To ensure robustness in concurrent multi-agent settings, SAFEFLOW introduces transactional execution, conflict resolution, and secure scheduling over shared state, preserving global consistency across agents. We further introduce mechanisms, including write-ahead logging, rollback, and secure caches, that further enhance resilience against runtime errors and policy violations. To validate the performances, we built SAFEFLOWBENCH, a comprehensive benchmark suite designed to evaluate agent reliability under adversarial, noisy, and concurrent operational conditions. Extensive experiments demonstrate that agents built with SAFEFLOW maintain impressive task performance and security guarantees even in hostile environments, substantially outperforming state-of-the-art. Together, SAFEFLOW and SAFEFLOWBENCH lay the groundwork for principled, robust, and secure agent ecosystems, advancing the frontier of reliable autonomy. 

**Abstract (ZH)**: 最近大型语言模型（LLMs）和视觉-语言模型（VLMs）的发展使得强大的自主代理能够进行复杂的推理和多模态工具使用。尽管这些代理的能力在不断增强，但当前的代理框架仍然脆弱，缺乏通过安全信息流、可靠性和多代理协调的原则性机制。在本工作中，我们引入了SAFEFLOW，这是一种新的协议级框架，用于构建可信赖的LLM/VLM基自主代理。SAFEFLOW实施了细粒度的信息流控制（IFC），精确追踪代理、工具、用户和环境之间交换的所有数据的来源、完整性和保密性。通过限制LLM推理尊重这些安全标签，SAFEFLOW防止未信任或恶意输入污染高完整性决策。为确保并发多代理设置中的鲁棒性，SAFEFLOW引入了事务执行、冲突解决和共享状态上的安全调度，从而在代理之间保持全局一致性。我们还引入了包括预先写日志、回滚和安全缓存在内的机制，进一步增强对运行时错误和策略违规的抗御能力。为了验证性能，我们构建了SAFEFLOWBENCH，这是一个全面的基准套件，旨在评估代理在对抗性、嘈杂和并发操作条件下的可靠性。广泛的实验表明，使用SAFEFLOW构建的代理即使在敌对环境中也能保持出色的任务性能和安全保证，并显著优于现有最佳方案。SAFEFLOW和SAFEFLOWBENCH为原则性、稳健和安全的代理生态系统奠定了基础，推动可靠自主性的前沿。 

---
# GTR-CoT: Graph Traversal as Visual Chain of Thought for Molecular Structure Recognition 

**Title (ZH)**: GTR-CoT: 图遍历作为视觉链思考的分子结构识别 

**Authors**: Jingchao Wang, Haote Yang, Jiang Wu, Yifan He, Xingjian Wei, Yinfan Wang, Chengjin Liu, Lingli Ge, Lijun Wu, Bin Wang, Dahua Lin, Conghui He  

**Link**: [PDF](https://arxiv.org/pdf/2506.07553)  

**Abstract**: Optical Chemical Structure Recognition (OCSR) is crucial for digitizing chemical knowledge by converting molecular images into machine-readable formats. While recent vision-language models (VLMs) have shown potential in this task, their image-captioning approach often struggles with complex molecular structures and inconsistent annotations. To overcome these challenges, we introduce GTR-Mol-VLM, a novel framework featuring two key innovations: (1) the \textit{Graph Traversal as Visual Chain of Thought} mechanism that emulates human reasoning by incrementally parsing molecular graphs through sequential atom-bond predictions, and (2) the data-centric principle of \textit{Faithfully Recognize What You've Seen}, which addresses the mismatch between abbreviated structures in images and their expanded annotations. To support model development, we constructed GTR-CoT-1.3M, a large-scale instruction-tuning dataset with meticulously corrected annotations, and introduced MolRec-Bench, the first benchmark designed for a fine-grained evaluation of graph-parsing accuracy in OCSR. Comprehensive experiments demonstrate that GTR-Mol-VLM achieves superior results compared to specialist models, chemistry-domain VLMs, and commercial general-purpose VLMs. Notably, in scenarios involving molecular images with functional group abbreviations, GTR-Mol-VLM outperforms the second-best baseline by approximately 14 percentage points, both in SMILES-based and graph-based metrics. We hope that this work will drive OCSR technology to more effectively meet real-world needs, thereby advancing the fields of cheminformatics and AI for Science. We will release GTR-CoT at this https URL. 

**Abstract (ZH)**: 光学化学结构识别（OCSR）对于通过将分子图像转换为机器可读格式来数字化化学知识至关重要。尽管近期的视觉-语言模型（VLMs）在这一任务上显示出潜力，但它们的图像配对方法往往难以处理复杂的分子结构和不一致的标注。为克服这些挑战，我们引入了GTR-Mol-VLM这一新型框架，该框架具有两条创新的关键机制：（1）图遍历作为视觉链式思考机制，通过顺序的原子-键预测逐步解析分子图，模仿人类推理过程；（2）以数据为中心的原则“忠实识别所见”，解决图像中简写结构与扩展标注之间的不匹配问题。为了支持模型开发，我们构建了包含仔细校正标注的大规模指令调优数据集GTR-CoT-1.3M，并引入了MolRec-Bench，这是首个用于光学化学结构识别中图解析准确性细粒度评估的基准。全面的实验表明，GTR-Mol-VLM在与专家模型、化学领域VLM和商用通用VLM的比较中表现更优。特别是在涉及分子图像中功能团简写的情况下，GTR-Mol-VLM在基于SMILES和图的度量上比第二好的基线高出约14个百分点。我们希望这项工作将推动OCSR技术更有效地满足实际需求，从而推动计算化学和科学人工智能领域的发展。GTR-CoT将在此处发布：https://github.com/。 

---
# Curriculum Learning With Counterfactual Group Relative Policy Advantage For Multi-Agent Reinforcement Learning 

**Title (ZH)**: 基于反事实组相对策略优势的多智能体强化学习 Curriculum 学习 

**Authors**: Weiqiang Jin, Hongyang Du, Guizhong Liu, Dong In Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.07548)  

**Abstract**: Multi-agent reinforcement learning (MARL) has achieved strong performance in cooperative adversarial tasks. However, most existing methods typically train agents against fixed opponent strategies and rely on such meta-static difficulty conditions, which limits their adaptability to changing environments and often leads to suboptimal policies. Inspired by the success of curriculum learning (CL) in supervised tasks, we propose a dynamic CL framework for MARL that employs an self-adaptive difficulty adjustment mechanism. This mechanism continuously modulates opponent strength based on real-time agent training performance, allowing agents to progressively learn from easier to more challenging scenarios. However, the dynamic nature of CL introduces instability due to nonstationary environments and sparse global rewards. To address this challenge, we develop a Counterfactual Group Relative Policy Advantage (CGRPA), which is tightly coupled with the curriculum by providing intrinsic credit signals that reflect each agent's impact under evolving task demands. CGRPA constructs a counterfactual advantage function that isolates individual contributions within group behavior, facilitating more reliable policy updates throughout the curriculum. CGRPA evaluates each agent's contribution through constructing counterfactual action advantage function, providing intrinsic rewards that enhance credit assignment and stabilize learning under non-stationary conditions. Extensive experiments demonstrate that our method improves both training stability and final performance, achieving competitive results against state-of-the-art methods. The code is available at this https URL. 

**Abstract (ZH)**: 多智能体强化学习中的自适应难度动态课程学习 

---
# Coordinating Search-Informed Reasoning and Reasoning-Guided Search in Claim Verification 

**Title (ZH)**: 基于断言验证中的搜索指导推理和推理引导搜索协调 

**Authors**: Qisheng Hu, Quanyu Long, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07528)  

**Abstract**: Multi-hop claim verification is inherently challenging, requiring multi-step reasoning to construct verification chains while iteratively searching for information to uncover hidden bridging facts. This process is fundamentally interleaved, as effective reasoning relies on dynamically retrieved evidence, while effective search demands reasoning to refine queries based on partial information. To achieve this, we propose Hierarchical Agent Reasoning and Information Search (HARIS), explicitly modeling the coordinated process of reasoning-driven searching and search-informed reasoning. HARIS consists of a high-level reasoning agent that focuses on constructing the main verification chain, generating factual questions when more information is needed, and a low-level search agent that iteratively retrieves more information, refining its search based on intermediate findings. This design allows each agent to specialize in its respective task, enhancing verification accuracy and interpretability. HARIS is trained using reinforcement learning with outcome-based rewards. Experimental results on the EX-FEVER and HOVER benchmarks demonstrate that HARIS achieves strong performance, greatly advancing multi-hop claim verification. 

**Abstract (ZH)**: 多跳声明验证本质上具有挑战性，需要多步推理构建验证链，并在迭代搜索中逐步发现隐藏的中介事实。这一过程本质上是交织的，因为有效的推理依赖于动态检索的证据，而有效的搜索则需要根据部分信息对查询进行细化。为此，我们提出了层次化代理推理和信息搜索（HARIS），明确模型了由推理驱动的搜索和基于搜索的推理的协调过程。HARIS 包含一个高层推理代理，专注于构建主要验证链，在需要更多信息时生成事实性问题，以及一个低层搜索代理，迭代检索更多信息，并根据中间发现来细化其搜索。这种设计允许每个代理在其专门的任务中进行专业化，从而提高验证的准确性和可解释性。HARIS 使用基于结果的奖励进行强化学习训练。在 EX-FEVER 和 HOVER 基准上的实验结果表明，HARIS 在多跳声明验证方面取得了出色的表现，极大地推动了多跳声明验证的发展。 

---
# Learning What Reinforcement Learning Can't: Interleaved Online Fine-Tuning for Hardest Questions 

**Title (ZH)**: 学习强化学习无法解决的内容：艰难问题的交错在线微调 

**Authors**: Lu Ma, Hao Liang, Meiyi Qiang, Lexiang Tang, Xiaochen Ma, Zhen Hao Wong, Junbo Niu, Chengyu Shen, Runming He, Bin Cui, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07527)  

**Abstract**: Recent advances in large language model (LLM) reasoning have shown that sophisticated behaviors such as planning and self-reflection can emerge through reinforcement learning (RL). However, despite these successes, RL in its current form remains insufficient to induce capabilities that exceed the limitations of the base model, as it is primarily optimized based on existing knowledge of the model rather than facilitating the acquisition of new information. To address this limitation, we employ supervised fine-tuning (SFT) to learn what RL cannot, which enables the incorporation of new knowledge and reasoning patterns by leveraging high-quality demonstration data. We analyze the training dynamics of RL and SFT for LLM reasoning and find that RL excels at maintaining and improving performance on questions within the model's original capabilities, while SFT is more effective at enabling progress on questions beyond the current scope of the model. Motivated by the complementary strengths of RL and SFT, we introduce a novel training approach, \textbf{ReLIFT} (\textbf{Re}inforcement \textbf{L}earning \textbf{I}nterleaved with Online \textbf{F}ine-\textbf{T}uning). In ReLIFT, the model is primarily trained using RL, but when it encounters challenging questions, high-quality solutions are collected for fine-tuning, and the training process alternates between RL and fine-tuning to enhance the model's reasoning abilities. ReLIFT achieves an average improvement of over +5.2 points across five competition-level benchmarks and one out-of-distribution benchmark compared to other zero-RL models. Furthermore, we demonstrate that ReLIFT outperforms both RL and SFT while using only 13\% of the detailed demonstration data, highlighting its scalability. These results provide compelling evidence that ReLIFT overcomes the fundamental limitations of RL and underscores the significant potential. 

**Abstract (ZH)**: Recent Advances in Large Language Model Reasoning: Combining Reinforcement Learning with Supervised Fine-Tuning 

---
# Efficient Generation of Diverse Cooperative Agents with World Models 

**Title (ZH)**: 高效的生成多样化协同代理模型 

**Authors**: Yi Loo, Akshunn Trivedi, Malika Meghjani  

**Link**: [PDF](https://arxiv.org/pdf/2506.07450)  

**Abstract**: A major bottleneck in the training process for Zero-Shot Coordination (ZSC) agents is the generation of partner agents that are diverse in collaborative conventions. Current Cross-play Minimization (XPM) methods for population generation can be very computationally expensive and sample inefficient as the training objective requires sampling multiple types of trajectories. Each partner agent in the population is also trained from scratch, despite all of the partners in the population learning policies of the same coordination task. In this work, we propose that simulated trajectories from the dynamics model of an environment can drastically speed up the training process for XPM methods. We introduce XPM-WM, a framework for generating simulated trajectories for XPM via a learned World Model (WM). We show XPM with simulated trajectories removes the need to sample multiple trajectories. In addition, we show our proposed method can effectively generate partners with diverse conventions that match the performance of previous methods in terms of SP population training reward as well as training partners for ZSC agents. Our method is thus, significantly more sample efficient and scalable to a larger number of partners. 

**Abstract (ZH)**: Zero-Shot Coordination代理训练过程中的主要瓶颈是在协作惯例多样性方面生成伙伴代理。当前用于群体生成的跨游戏最小化(XPM)方法可能非常计算成本高且采样效率低，因为训练目标需要采样多种类型的轨迹。尽管环境中的所有伙伴代理都在同一协调任务中学习策略，但群体中的每个伙伴代理都从头开始训练。在此工作中，我们提出使用环境动力学模型生成的模拟轨迹可以大幅加快XPM方法的训练过程。我们提出了XPM-WM框架，通过学习的世界模型（WM）生成XPM的模拟轨迹。我们展示了使用模拟轨迹的XPM消除了需要采样多种轨迹的需求。此外，我们展示了我们提出的方法能够有效地生成具有多样性协作惯例的伙伴，这些伙伴在SP群体训练奖励以及零-shot协调代理训练伙伴方面与先前方法具有竞争力。因此，我们的方法在采样效率方面显著提高，并且适用于更大数量的伙伴。 

---
# Fact in Fragments: Deconstructing Complex Claims via LLM-based Atomic Fact Extraction and Verification 

**Title (ZH)**: 断片的事实：通过基于LLM的原子事实提取与验证分解复杂断言 

**Authors**: Liwen Zheng, Chaozhuo Li, Zheng Liu, Feiran Huang, Haoran Jia, Zaisheng Ye, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07446)  

**Abstract**: Fact verification plays a vital role in combating misinformation by assessing the veracity of claims through evidence retrieval and reasoning. However, traditional methods struggle with complex claims requiring multi-hop reasoning over fragmented evidence, as they often rely on static decomposition strategies and surface-level semantic retrieval, which fail to capture the nuanced structure and intent of the claim. This results in accumulated reasoning errors, noisy evidence contamination, and limited adaptability to diverse claims, ultimately undermining verification accuracy in complex scenarios. To address this, we propose Atomic Fact Extraction and Verification (AFEV), a novel framework that iteratively decomposes complex claims into atomic facts, enabling fine-grained retrieval and adaptive reasoning. AFEV dynamically refines claim understanding and reduces error propagation through iterative fact extraction, reranks evidence to filter noise, and leverages context-specific demonstrations to guide the reasoning process. Extensive experiments on five benchmark datasets demonstrate that AFEV achieves state-of-the-art performance in both accuracy and interpretability. 

**Abstract (ZH)**: 原子事实提取与验证（AFEV）在复杂断言的细粒度检索与自适应推理中发挥关键作用 

---
# LegalReasoner: Step-wised Verification-Correction for Legal Judgment Reasoning 

**Title (ZH)**: LegalReasoner: 逐步验证-修正的法律判决推理方法 

**Authors**: Weijie Shi, Han Zhu, Jiaming Ji, Mengze Li, Jipeng Zhang, Ruiyuan Zhang, Jia Zhu, Jiajie Xu, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.07443)  

**Abstract**: Legal judgment prediction (LJP) aims to function as a judge by making final rulings based on case claims and facts, which plays a vital role in the judicial domain for supporting court decision-making and improving judicial efficiency. However, existing methods often struggle with logical errors when conducting complex legal reasoning. We propose LegalReasoner, which enhances LJP reliability through step-wise verification and correction of the reasoning process. Specifically, it first identifies dispute points to decompose complex cases, and then conducts step-wise reasoning while employing a process verifier to validate each step's logic from correctness, progressiveness, and potential perspectives. When errors are detected, expert-designed attribution and resolution strategies are applied for correction. To fine-tune LegalReasoner, we release the LegalHK dataset, containing 58,130 Hong Kong court cases with detailed annotations of dispute points, step-by-step reasoning chains, and process verification labels. Experiments demonstrate that LegalReasoner significantly improves concordance with court decisions from 72.37 to 80.27 on LLAMA-3.1-70B. The data is available at this https URL. 

**Abstract (ZH)**: 法律判决预测（LJP）旨在基于案件主张和事实作出最终裁决，对司法领域支持法院决策和提高司法效率发挥重要作用。然而，现有方法在进行复杂法律推理时往往容易出现逻辑错误。我们提出了法律推理器（LegalReasoner），通过逐步验证和修正推理过程来提高LJP的可靠性。具体而言，它首先识别争议点以分解复杂案件，然后进行逐步推理，并使用过程验证器从正确性、进步性和潜在视角验证每一步的逻辑。当检测到错误时，应用专家设计的归因和解决策略进行修正。为了微调法律推理器（LegalReasoner），我们发布了包含58,130个香港法院案件及其详细标注的争议点、逐步推理链和过程验证标签的LegalHK数据集。实验表明，法律推理器在LLAMA-3.1-70B上使一致性与法院决策大幅提升，从72.37提高到80.27。数据可在如下链接获取：this https URL。 

---
# HeTa: Relation-wise Heterogeneous Graph Foundation Attack Model 

**Title (ZH)**: HeTa：关系层面的异构图基础攻击模型 

**Authors**: Yuling Wang, Zihui Chen, Pengfei Jiao, Xiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07428)  

**Abstract**: Heterogeneous Graph Neural Networks (HGNNs) are vulnerable, highlighting the need for tailored attacks to assess their robustness and ensure security. However, existing HGNN attacks often require complex retraining of parameters to generate specific perturbations for new scenarios. Recently, foundation models have opened new horizons for the generalization of graph neural networks by capturing shared semantics across various graph distributions. This leads us to ask:Can we design a foundation attack model for HGNNs that enables generalizable perturbations across different HGNNs, and quickly adapts to new heterogeneous graphs (HGs)? Empirical findings reveal that, despite significant differences in model design and parameter space, different HGNNs surprisingly share common vulnerability patterns from a relation-aware perspective. Therefore, we explore how to design foundation HGNN attack criteria by mining shared attack units. In this paper, we propose a novel relation-wise heterogeneous graph foundation attack model, HeTa. We introduce a foundation surrogate model to align heterogeneity and identify the importance of shared relation-aware attack units. Building on this, we implement a serialized relation-by-relation attack based on the identified relational weights. In this way, the perturbation can be transferred to various target HGNNs and easily fine-tuned for new HGs. Extensive experiments exhibit powerful attack performances and generalizability of our method. 

**Abstract (ZH)**: 异质图神经网络（HGNNs）的脆弱性凸显了需要定制攻击模型以评估其鲁棒性和确保安全性的重要性。然而，现有的HGNN攻击通常需要复杂的参数重训练来生成针对新场景的具体扰动。最近，基础模型为图神经网络的一般化提供了新的视角，通过捕获各种图分布之间的共享语义。这使我们思考：我们能否设计一种基础攻击模型，使HGNNs在不同模型之间实现可泛化的扰动，并快速适应新的异质图（HGs）？实证研究发现，尽管不同HGNN的设计模型和参数空间存在显著差异，但从关系感知的角度来看，它们惊人地共享共同的脆弱性模式。因此，我们探索如何通过挖掘共享的关系感知攻击单元来设计基础HGNN攻击标准。在本文中，我们提出了一种新颖的关系感知异质图基础攻击模型HeTa。我们引入了一个基础代理模型来对齐异质性并识别共享关系感知攻击单元的重要性。在此基础上，我们基于识别出的关系权重实施了按关系逐步的攻击。这样，扰动可以转移到各种目标HGNNs，并且可以轻松微调以适应新的HGs。广泛实验展示了我们方法的强大攻击性能和一般化能力。 

---
# Evaluating Visual Mathematics in Multimodal LLMs: A Multilingual Benchmark Based on the Kangaroo Tests 

**Title (ZH)**: 多模态大语言模型中视觉数学评估：基于袋鼠测试的多语言基准 

**Authors**: Arnau Igualde Sáez, Lamyae Rhomrasi, Yusef Ahsini, Ricardo Vinuesa, Sergio Hoyas, Jose P. García Sabater, Marius J. Fullana i Alfonso, J. Alberto Conejero  

**Link**: [PDF](https://arxiv.org/pdf/2506.07418)  

**Abstract**: Multimodal Large Language Models (MLLMs) promise advanced vision language capabilities, yet their effectiveness in visually presented mathematics remains underexplored. This paper analyzes the development and evaluation of MLLMs for mathematical problem solving, focusing on diagrams, multilingual text, and symbolic notation. We then assess several models, including GPT 4o, Pixtral, Qwen VL, Llama 3.2 Vision variants, and Gemini 2.0 Flash in a multilingual Kangaroo style benchmark spanning English, French, Spanish, and Catalan. Our experiments reveal four key findings. First, overall precision remains moderate across geometry, visual algebra, logic, patterns, and combinatorics: no single model excels in every topic. Second, while most models see improved accuracy with questions that do not have images, the gain is often limited; performance for some remains nearly unchanged without visual input, indicating underutilization of diagrammatic information. Third, substantial variation exists across languages and difficulty levels: models frequently handle easier items but struggle with advanced geometry and combinatorial reasoning. Notably, Gemini 2.0 Flash achieves the highest precision on image based tasks, followed by Qwen VL 2.5 72B and GPT 4o, though none approach human level performance. Fourth, a complementary analysis aimed at distinguishing whether models reason or simply recite reveals that Gemini and GPT 4o stand out for their structured reasoning and consistent accuracy. In contrast, Pixtral and Llama exhibit less consistent reasoning, often defaulting to heuristics or randomness when unable to align their outputs with the given answer options. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在视觉呈现的数学领域提供了高级的语言能力，但其有效性仍待进一步探索。本文分析了MLLMs在数学问题解决中的发展与评估，重点关注图表、多语言文本和象征性符号。然后，我们在英文、法文、西班牙文和加泰罗尼亚文的Kangaroo风格多语言基准测试中评估了包括GPT 4o、Pixtral、Qwen VL、Llama 3.2视觉变体和Gemini 2.0 Flash在内的几种模型。我们的实验揭示了四个关键发现。首先，整体精确度在几何学、视觉代数、逻辑学、模式和组合学中保持适中：没有单一模型在所有主题中都表现优异。其次，大多数模型在没有图像的问题上准确性有所提高，但改进幅度往往有限；一些模型在没有视觉输入的情况下性能几乎没有变化，表明对图表信息的利用不足。第三，不同语言和难度级别之间存在显著差异：模型通常能处理较简单的题目，但在高级几何学和组合推理方面却遇到困难。值得注意的是，Gemini 2.0 Flash在基于图像的任务中取得了最高的精确度，其次是Qwen VL 2.5 72B和GPT 4o，但均未达到人类水平的性能。第四，旨在区分模型是推理还是简单复述的补充分析表明，Gemini和GPT 4o因其结构化的推理和一致的准确性脱颖而出。相比之下，Pixtral和Llama的推理不那么一致，当无法使输出与给定答案选项对齐时，它们往往会依赖于启发式方法或随机性。 

---
# An Intelligent Fault Self-Healing Mechanism for Cloud AI Systems via Integration of Large Language Models and Deep Reinforcement Learning 

**Title (ZH)**: 基于大型语言模型和深度强化学习集成的云AI系统智能故障自愈机制 

**Authors**: Ze Yang, Yihong Jin, Juntian Liu, Xinhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07411)  

**Abstract**: As the scale and complexity of cloud-based AI systems continue to increase, the detection and adaptive recovery of system faults have become the core challenges to ensure service reliability and continuity. In this paper, we propose an Intelligent Fault Self-Healing Mechanism (IFSHM) that integrates Large Language Model (LLM) and Deep Reinforcement Learning (DRL), aiming to realize a fault recovery framework with semantic understanding and policy optimization capabilities in cloud AI systems. On the basis of the traditional DRL-based control model, the proposed method constructs a two-stage hybrid architecture: (1) an LLM-driven fault semantic interpretation module, which can dynamically extract deep contextual semantics from multi-source logs and system indicators to accurately identify potential fault modes; (2) DRL recovery strategy optimizer, based on reinforcement learning, learns the dynamic matching of fault types and response behaviors in the cloud environment. The innovation of this method lies in the introduction of LLM for environment modeling and action space abstraction, which greatly improves the exploration efficiency and generalization ability of reinforcement learning. At the same time, a memory-guided meta-controller is introduced, combined with reinforcement learning playback and LLM prompt fine-tuning strategy, to achieve continuous adaptation to new failure modes and avoid catastrophic forgetting. Experimental results on the cloud fault injection platform show that compared with the existing DRL and rule methods, the IFSHM framework shortens the system recovery time by 37% with unknown fault scenarios. 

**Abstract (ZH)**: 基于大规模语言模型和深度强化学习的智能故障自愈机制 

---
# Boosting Vulnerability Detection of LLMs via Curriculum Preference Optimization with Synthetic Reasoning Data 

**Title (ZH)**: 基于合成推理数据的教学偏好优化以增强对LLMs的漏洞检测 

**Authors**: Xin-Cheng Wen, Yijun Yang, Cuiyun Gao, Yang Xiao, Deheng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2506.07390)  

**Abstract**: Large language models (LLMs) demonstrate considerable proficiency in numerous coding-related tasks; however, their capabilities in detecting software vulnerabilities remain limited. This limitation primarily stems from two factors: (1) the absence of reasoning data related to vulnerabilities, which hinders the models' ability to capture underlying vulnerability patterns; and (2) their focus on learning semantic representations rather than the reason behind them, thus failing to recognize semantically similar vulnerability samples. Furthermore, the development of LLMs specialized in vulnerability detection is challenging, particularly in environments characterized by the scarcity of high-quality datasets. In this paper, we propose a novel framework ReVD that excels at mining vulnerability patterns through reasoning data synthesizing and vulnerability-specific preference optimization. Specifically, we construct forward and backward reasoning processes for vulnerability and corresponding fixed code, ensuring the synthesis of high-quality reasoning data. Moreover, we design the triplet supervised fine-tuning followed by curriculum online preference optimization for enabling ReVD to better understand vulnerability patterns. The extensive experiments conducted on PrimeVul and SVEN datasets demonstrate that ReVD sets new state-of-the-art for LLM-based software vulnerability detection, e.g., 12.24\%-22.77\% improvement in the accuracy. The source code and data are available at this https URL. 

**Abstract (ZH)**: 大规模语言模型在编码相关任务中表现出色，但在检测软件漏洞方面的能力有限。这一限制主要源于两个因素：(1) 缺乏与漏洞相关的推理数据，这阻碍了模型捕捉潜在漏洞模式的能力；(2) 模型侧重于学习语义表示而非其背后的原因，因此无法识别语义相似的漏洞样本。此外，特别是在高质量数据集稀缺的环境中，开发专门用于漏洞检测的语言模型极具挑战性。本文提出了一种新颖的框架ReVD，该框架通过推理数据合成和针对特定漏洞的偏好优化，擅长挖掘漏洞模式。具体来说，我们构建了漏洞及其相应修复代码的正向和反向推理过程，确保合成高质量的推理数据。此外，我们设计了三元监督微调和基于 Curriculum 的在线偏好优化策略，以使ReVD更好地理解漏洞模式。在PrimeVul和SVEN数据集上的广泛实验表明，ReVD为基于语言模型的软件漏洞检测设立了新标准，例如在准确性方面提高了12.24%-22.77%。源代码和数据可在以下网址获取。 

---
# Subgoal-Guided Policy Heuristic Search with Learned Subgoals 

**Title (ZH)**: 基于子目标引导的策略启发式搜索与学习到的子目标 

**Authors**: Jake Tuero, Michael Buro, Levi H. S. Lelis  

**Link**: [PDF](https://arxiv.org/pdf/2506.07255)  

**Abstract**: Policy tree search is a family of tree search algorithms that use a policy to guide the search. These algorithms provide guarantees on the number of expansions required to solve a given problem that are based on the quality of the policy. While these algorithms have shown promising results, the process in which they are trained requires complete solution trajectories to train the policy. Search trajectories are obtained during a trial-and-error search process. When the training problem instances are hard, learning can be prohibitively costly, especially when starting from a randomly initialized policy. As a result, search samples are wasted in failed attempts to solve these hard instances. This paper introduces a novel method for learning subgoal-based policies for policy tree search algorithms. The subgoals and policies conditioned on subgoals are learned from the trees that the search expands while attempting to solve problems, including the search trees of failed attempts. We empirically show that our policy formulation and training method improve the sample efficiency of learning a policy and heuristic function in this online setting. 

**Abstract (ZH)**: 基于子目标的学习方法以提高策略树搜索中的样本效率 

---
# LLM-Enhanced Rapid-Reflex Async-Reflect Embodied Agent for Real-Time Decision-Making in Dynamically Changing Environments 

**Title (ZH)**: 增强大型语言模型的快速反应异步反射体现代理在动态变化环境中的实时决策-making 

**Authors**: Yangqing Zheng, Shunqi Mao, Dingxin Zhang, Weidong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2506.07223)  

**Abstract**: In the realm of embodied intelligence, the evolution of large language models (LLMs) has markedly enhanced agent decision making. Consequently, researchers have begun exploring agent performance in dynamically changing high-risk scenarios, i.e., fire, flood, and wind scenarios in the HAZARD benchmark. Under these extreme conditions, the delay in decision making emerges as a crucial yet insufficiently studied issue. We propose a Time Conversion Mechanism (TCM) that translates inference delays in decision-making into equivalent simulation frames, thus aligning cognitive and physical costs under a single FPS-based metric. By extending HAZARD with Respond Latency (RL) and Latency-to-Action Ratio (LAR), we deliver a fully latency-aware evaluation protocol. Moreover, we present the Rapid-Reflex Async-Reflect Agent (RRARA), which couples a lightweight LLM-guided feedback module with a rule-based agent to enable immediate reactive behaviors and asynchronous reflective refinements in situ. Experiments on HAZARD show that RRARA substantially outperforms existing baselines in latency-sensitive scenarios. 

**Abstract (ZH)**: 在本体智能领域，大型语言模型（LLMs）的发展显著增强了代理决策能力。因此，研究人员开始探索代理在动态变化的高风险场景中的表现，例如HAZARD基准中的火灾、洪水和风灾场景。在这些极端条件下，决策延迟成为了一个重要但研究不足的问题。我们提出了一种时间转换机制（TCM），将决策推理延迟转换为等效的模拟帧，从而在基于FPS的单一度量下对认知和物理成本进行对齐。通过将HAZARD扩展为响应延迟（RL）和动作延迟比（LAR），我们提供了一个完全关注延迟的评估协议。此外，我们提出了快速反应异步反思代理（RRARA），该代理结合了一个轻量级的LLM指导反馈模块和基于规则的代理，以实现即时的反应行为和现场异步的反思性改进。在HAZARD上的实验表明，RRARA在延迟敏感场景中显著优于现有基线。 

---
# BIMgent: Towards Autonomous Building Modeling via Computer-use Agents 

**Title (ZH)**: BIMgent: 通过计算机代理实现自主建筑建模 

**Authors**: Zihan Deng, Changyu Du, Stavros Nousias, André Borrmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.07217)  

**Abstract**: Existing computer-use agents primarily focus on general-purpose desktop automation tasks, with limited exploration of their application in highly specialized domains. In particular, the 3D building modeling process in the Architecture, Engineering, and Construction (AEC) sector involves open-ended design tasks and complex interaction patterns within Building Information Modeling (BIM) authoring software, which has yet to be thoroughly addressed by current studies. In this paper, we propose BIMgent, an agentic framework powered by multimodal large language models (LLMs), designed to enable autonomous building model authoring via graphical user interface (GUI) operations. BIMgent automates the architectural building modeling process, including multimodal input for conceptual design, planning of software-specific workflows, and efficient execution of the authoring GUI actions. We evaluate BIMgent on real-world building modeling tasks, including both text-based conceptual design generation and reconstruction from existing building design. The design quality achieved by BIMgent was found to be reasonable. Its operations achieved a 32% success rate, whereas all baseline models failed to complete the tasks (0% success rate). Results demonstrate that BIMgent effectively reduces manual workload while preserving design intent, highlighting its potential for practical deployment in real-world architectural modeling scenarios. 

**Abstract (ZH)**: 现有的计算机使用代理主要集中在通用桌面自动化任务上，在高度专业化领域中的应用探索有限。特别是在建筑、工程和施工（AEC）行业中，建筑信息建模（BIM）作者软件中的开放式设计任务和复杂交互模式尚未得到充分研究。本文提出BIMgent，一种基于多模态大规模语言模型（LLMs）的动力体系框架，旨在通过图形用户界面（GUI）操作实现自主建筑模型创建。BIMgent自动化了建筑设计过程，包括多模态输入的概念设计、软件特定工作流的规划以及作者GUI操作的有效执行。我们在实际建筑建模任务中评估了BIMgent，包括基于文本的概念设计生成和现有建筑设计的重建。BIMgent实现的设计质量合理，操作成功率为32%，而所有基线模型都无法完成任务（成功率为0%）。结果显示，BIMgent有效减少了人工工作量，同时保留了设计意图，突显了其在实际建筑建模场景中的实际部署潜力。 

---
# Reasoning Multimodal Large Language Model: Data Contamination and Dynamic Evaluation 

**Title (ZH)**: 多模态大型语言模型推理：数据污染与动态评估 

**Authors**: Ming Liu, Wensheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07202)  

**Abstract**: Multimodal Large Language Models (MLLMs) show impressive vision-language benchmark performance, yet growing concerns about data contamination (test set exposure during training) risk masking true generalization. This concern extends to reasoning MLLMs, often fine-tuned via reinforcement learning from potentially contaminated base models. We propose a novel dynamic evaluation framework to rigorously assess MLLM generalization, moving beyond static benchmarks. Instead of perturbing inputs, we perturb the task itself. Using the same visual input, models are evaluated across a family of tasks (e.g., QA, captioning, question posing, verification) to probe diverse capabilities. This task perturbation reveals whether model performance is robust or reliant on superficial task-specific cues. Our approach is analogous to loss landscape sharpness: models overfit or contaminated for a single task (sharp minima) falter under task shifts, unlike models with generalizable solutions (flatter minima). We developed an automated pipeline with a calibrated judge scoring open-ended generations (captions, questions) using paraphrase and corruption sampling. Applying this framework to leading image/video MLLMs on benchmarks including MME, RealWorldQA, and CVRR-ES, we analyze each model's cross-task "ability vector." We demonstrate that fine-tuning on simulated test data (extreme contamination) drastically sharpens task-specific performance but harms overall generalization. Our dynamic task perturbation offers deeper insights into MLLM generalization, distinguishing genuine understanding from spurious leakage or overfitting. 

**Abstract (ZH)**: 多模态大型语言模型的动态评估框架：超越静态基准探求通用性 

---
# Exploring Effective Strategies for Building a Customised GPT Agent for Coding Classroom Dialogues 

**Title (ZH)**: 探索构建个性化GPT代理以用于编程课堂对话的有效策略 

**Authors**: Luwei Bai, Dongkeun Han, Sara Hennessy  

**Link**: [PDF](https://arxiv.org/pdf/2506.07194)  

**Abstract**: This study investigates effective strategies for developing a customised GPT agent to code classroom dialogue. While classroom dialogue is widely recognised as a crucial element of education, its analysis remains challenging due to the need for a nuanced understanding of dialogic functions and the labour-intensive nature of manual transcript coding. Recent advancements in large language models offer promising avenues for automating this process. However, existing studies predominantly focus on training large-scale models or evaluating pre-trained models with fixed codebooks, which are often not applicable or replicable for dialogue researchers working with small datasets or customised coding schemes. Using GPT-4's MyGPT agent as a case, this study evaluates its baseline performance in coding classroom dialogue with a human codebook and examines how performance varies with different example inputs through a variable control method. Through a design-based research approach, it identifies a set of practical strategies, based on MyGPT's unique features, for configuring effective agents with limited data. The findings suggest that, despite some limitations, a MyGPT agent developed with these strategies can serve as a useful coding assistant by generating coding suggestions. 

**Abstract (ZH)**: 本研究探究了开发定制化GPT代理以编码课堂对话的有效策略。尽管课堂对话被广泛认为是教育的关键组成部分，但由于需要对对话功能有精微的理解以及手动转录编码的劳动密集性，对其进行分析仍然具有挑战性。近年来，大型语言模型的进步为自动化这一过程提供了有希望的途径。然而，现有研究主要集中在训练大规模模型或使用固定编码本对预训练模型进行评估，这些往往不适用于处理小型数据集或定制编码方案的对话研究者。通过将GPT-4的MyGPT代理作为案例，本研究评估了其使用人类编码本编码课堂对话的基线性能，并通过变量控制方法考察了不同示例输入如何影响性能。通过设计导向的研究方法，本研究基于MyGPT的独特功能，识别出一套实用策略，以在有限数据情况下配置有效的代理。研究结果表明，尽管存在一些局限性，但使用这些策略开发的MyGPT代理可以作为一种有用的编码助手，通过生成编码建议发挥作用。 

---
# Mitigating Behavioral Hallucination in Multimodal Large Language Models for Sequential Images 

**Title (ZH)**: 多模态大型语言模型中序列图像行为幻觉的缓解 

**Authors**: Liangliang You, Junchi Yao, Shu Yang, Guimin Hu, Lijie Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07184)  

**Abstract**: While multimodal large language models excel at various tasks, they still suffer from hallucinations, which limit their reliability and scalability for broader domain applications. To address this issue, recent research mainly focuses on objective hallucination. However, for sequential images, besides objective hallucination, there is also behavioral hallucination, which is less studied. This work aims to fill in the gap. We first reveal that behavioral hallucinations mainly arise from two key factors: prior-driven bias and the snowball effect. Based on these observations, we introduce SHE (Sequence Hallucination Eradication), a lightweight, two-stage framework that (1) detects hallucinations via visual-textual alignment check using our proposed adaptive temporal window and (2) mitigates them via orthogonal projection onto the joint embedding space. We also propose a new metric (BEACH) to quantify behavioral hallucination severity. Empirical results on standard benchmarks demonstrate that SHE reduces behavioral hallucination by over 10% on BEACH while maintaining descriptive accuracy. 

**Abstract (ZH)**: 面向序列图像的幻觉消除：一种轻量级两阶段框架 

---
# Translating Federated Learning Algorithms in Python into CSP Processes Using ChatGPT 

**Title (ZH)**: 将Python中的联邦学习算法转换为CSP进程using ChatGPT 

**Authors**: Miroslav Popovic, Marko Popovic, Miodrag Djukic, Ilija Basicevic  

**Link**: [PDF](https://arxiv.org/pdf/2506.07173)  

**Abstract**: The Python Testbed for Federated Learning Algorithms is a simple Python FL framework that is easy to use by ML&AI developers who do not need to be professional programmers and is also amenable to LLMs. In the previous research, generic federated learning algorithms provided by this framework were manually translated into the CSP processes and algorithms' safety and liveness properties were automatically verified by the model checker PAT. In this paper, a simple translation process is introduced wherein the ChatGPT is used to automate the translation of the mentioned federated learning algorithms in Python into the corresponding CSP processes. Within the process, the minimality of the used context is estimated based on the feedback from ChatGPT. The proposed translation process was experimentally validated by successful translation (verified by the model checker PAT) of both generic centralized and decentralized federated learning algorithms. 

**Abstract (ZH)**: Python测试床中的联邦学习算法：一种易于使用的Python联邦学习框架，适用于无需专业编程知识的ML&AI开发者，并且易于转换为LLMs。本文介绍了一个简单的转换过程，使用ChatGPT自动化转换Python中的联邦学习算法为相应的CSP过程，并基于ChatGPT的反馈估计所用上下文的最小性。所提出的转换过程通过成功转换（由模型检查器PAT验证）通用中心化和去中心化联邦学习算法得到实验验证。 

---
# BRIGHT+: Upgrading the BRIGHT Benchmark with MARCUS, a Multi-Agent RAG Clean-Up Suite 

**Title (ZH)**: BRIGHT+: 使用MARCUS多代理RAG清理套件升级BRIGHT基准测试 

**Authors**: Liyang Chen, Yujun Cai, Jieqiong Dong, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07116)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems require corpora that are both structurally clean and semantically coherent. BRIGHT is a recent and influential benchmark designed to evaluate complex multi-hop retrieval across diverse, high-reasoning domains. However, its practical effectiveness is limited by common web-crawled artifacts - such as content redundancy and semantic discontinuity - that impair retrieval accuracy and downstream reasoning. Notably, we find that such issues are concentrated in seven StackExchange-derived subdomains, while other domains (e.g., Coding and Theorem-based content) remain relatively clean.
In this study, we present MARCUS, a multi-agent pipeline that leverages large language models (LLMs) to systematically clean and re-chunk BRIGHT into a higher-quality corpus: BRIGHT-Plus. MARCUS applies dedicated agents for structural noise removal and semantic segmentation, preserving answer-bearing spans while improving contextual integrity. Experimental evaluations demonstrate that BRIGHT-Plus yields consistent and significant improvements in both retrieval accuracy and multi-hop reasoning across a diverse set of retrievers. We release both the BRIGHT-Plus corpus and the MARCUS pipeline to support future research on robust, reasoning-centric retrieval. 

**Abstract (ZH)**: Retrieval-Augmented Generation (RAG) 系统要求具有结构清晰和语义连贯的语料库。BRIGHT 是一个近期有影响力的基准，用于评估跨多种高推理领域复杂多跳检索的表现。然而，它的实际有效性由于常见的网页抓取伪影（如内容冗余和语义不连续）而受到限制，这些伪影影响了检索准确性和下游推理。值得注意的是，这些问题集中在七个源自 StackExchange 的子领域中，而其他领域（例如编程和基于定理的内容）则相对清洁。
在此研究中，我们提出了一种多agent管道 MARCUS，利用大规模语言模型 (LLMs) 系统地清洁并重新划分 BRIGHT，生成一个更高质量的语料库：BRIGHT-Plus。MARCUS 应用了专门的代理来去除结构噪声并进行语义分割，在保持答案携带片段的同时提升上下文完整性。实验评估表明，BRIGHT-Plus 在多种检索器中在检索准确性和多跳推理方面均表现出一致且显著的改进。我们发布了 BRIGHT-Plus 语料库和 MARCUS 管道，以支持未来针对稳健、以推理为中心的检索的研究。 

---
# Reasoning Paths as Signals: Augmenting Multi-hop Fact Verification through Structural Reasoning Progression 

**Title (ZH)**: 推理路径作为信号：通过结构推理进程增强多跳事实核实 

**Authors**: Liwen Zheng, Chaozhuo Li, Haoran Jia, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07075)  

**Abstract**: The growing complexity of factual claims in real-world scenarios presents significant challenges for automated fact verification systems, particularly in accurately aggregating and reasoning over multi-hop evidence. Existing approaches often rely on static or shallow models that fail to capture the evolving structure of reasoning paths, leading to fragmented retrieval and limited interpretability. To address these issues, we propose a Structural Reasoning framework for Multi-hop Fact Verification that explicitly models reasoning paths as structured graphs throughout both evidence retrieval and claim verification stages. Our method comprises two key modules: a structure-enhanced retrieval mechanism that constructs reasoning graphs to guide evidence collection, and a reasoning-path-guided verification module that incrementally builds subgraphs to represent evolving inference trajectories. We further incorporate a structure-aware reasoning mechanism that captures long-range dependencies across multi-hop evidence chains, enabling more precise verification. Extensive experiments on the FEVER and HoVer datasets demonstrate that our approach consistently outperforms strong baselines, highlighting the effectiveness of reasoning-path modeling in enhancing retrieval precision and verification accuracy. 

**Abstract (ZH)**: 复杂事实断言在现实场景中的不断增加使得自动事实验证系统面临重大挑战，特别是在多跳证据聚类和推理方面。现有方法往往依赖于静态或浅层模型，无法捕捉推理路径的演变结构，导致检索碎片化和解释性有限。为解决这些问题，我们提出了一种结构推理框架，用于多跳事实验证，在证据检索和断言验证阶段明确地将推理路径建模为结构化图。该方法包含两个关键模块：一种结构增强的检索机制，用于构建推理图以指导证据收集，以及一种由推理路径引导的验证模块，通过增量构建子图来表示不断演化的推断轨迹。我们进一步引入了一种结构感知的推理机制，能够在多跳证据链中捕捉长范围依赖性，从而实现更精确的验证。在FEVER和HoVer数据集上的 extensive 实验表明，我们的方法在检索精度和验证准确性方面均优于强基线，强调了推理路径建模在提升检索和验证效果方面的有效性。 

---
# Mathesis: Towards Formal Theorem Proving from Natural Languages 

**Title (ZH)**: 数学原理：从自然语言向形式定理证明的探索 

**Authors**: Yu Xuejun, Jianyuan Zhong, Zijin Feng, Pengyi Zhai, Roozbeh Yousefzadeh, Wei Chong Ng, Haoxiong Liu, Ziyi Shou, Jing Xiong, Yudong Zhou, Claudia Beth Ong, Austen Jeremy Sugiarto, Yaoxi Zhang, Wai Ming Tai, Huan Cao, Dongcai Lu, Jiacheng Sun, Qiang Xu, Shen Xin, Zhenguo Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.07047)  

**Abstract**: Recent advances in large language models show strong promise for formal reasoning. However, most LLM-based theorem provers have long been constrained by the need for expert-written formal statements as inputs, limiting their applicability to real-world problems expressed in natural language. We tackle this gap with Mathesis, the first end-to-end theorem proving pipeline processing informal problem statements. It contributes Mathesis-Autoformalizer, the first autoformalizer using reinforcement learning to enhance the formalization ability of natural language problems, aided by our novel LeanScorer framework for nuanced formalization quality assessment. It also proposes a Mathesis-Prover, which generates formal proofs from the formalized statements. To evaluate the real-world applicability of end-to-end formal theorem proving, we introduce Gaokao-Formal, a benchmark of 488 complex problems from China's national college entrance exam. Our approach is carefully designed, with a thorough study of each component. Experiments demonstrate Mathesis's effectiveness, with the autoformalizer outperforming the best baseline by 22% in pass-rate on Gaokao-Formal. The full system surpasses other model combinations, achieving 64% accuracy on MiniF2F with pass@32 and a state-of-the-art 18% on Gaokao-Formal. 

**Abstract (ZH)**: Recent Advances in Large Language Models Show Strong Promise for Formal Reasoning: Tackling the Gap with Mathesis for End-to-End Theorem Proving 

---
# Evaluating LLM-corrupted Crowdsourcing Data Without Ground Truth 

**Title (ZH)**: 评估受LLM影响的 crowdsourcing 数据的准确性无需地面 truth 

**Authors**: Yichi Zhang, Jinlong Pang, Zhaowei Zhu, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06991)  

**Abstract**: The recent success of generative AI highlights the crucial role of high-quality human feedback in building trustworthy AI systems. However, the increasing use of large language models (LLMs) by crowdsourcing workers poses a significant challenge: datasets intended to reflect human input may be compromised by LLM-generated responses. Existing LLM detection approaches often rely on high-dimension training data such as text, making them unsuitable for annotation tasks like multiple-choice labeling. In this work, we investigate the potential of peer prediction -- a mechanism that evaluates the information within workers' responses without using ground truth -- to mitigate LLM-assisted cheating in crowdsourcing with a focus on annotation tasks. Our approach quantifies the correlations between worker answers while conditioning on (a subset of) LLM-generated labels available to the requester. Building on prior research, we propose a training-free scoring mechanism with theoretical guarantees under a crowdsourcing model that accounts for LLM collusion. We establish conditions under which our method is effective and empirically demonstrate its robustness in detecting low-effort cheating on real-world crowdsourcing datasets. 

**Abstract (ZH)**: 生成式AI的Recent成功凸显了高质量人类反馈在构建可信赖AI系统中的关键作用。然而，众包工作者越来越多地使用大型语言模型（LLMs）带来了重大挑战：旨在反映人类输入的数据集可能因LLM生成的回答而受损。现有的LLM检测方法通常依赖于高维度训练数据如文本，这使得它们不适合标注任务如多选标注。在本文中，我们探讨了同事预测的潜力——一种不使用真实标签来评估工作者回答中信息的机制——以在众包标注任务中缓解LLM辅助的作弊问题。我们的方法在部分LM生成标签的条件下量化了工作者答案之间的相关性。基于先前的研究，我们提出了一种无需训练的评分机制，并在考虑LM合谋的众包模型下提供了理论保证。我们确定了该方法有效的工作条件，并通过真实世界众包数据集的实验证明了其在检测低努力作弊方面的鲁棒性。 

---
# Deep RL Needs Deep Behavior Analysis: Exploring Implicit Planning by Model-Free Agents in Open-Ended Environments 

**Title (ZH)**: Deep RL 需要深入的行为分析：在开放环境中模型自由代理隐含规划的探索 

**Authors**: Riley Simmons-Edler, Ryan P. Badman, Felix Baastad Berg, Raymond Chua, John J. Vastola, Joshua Lunger, William Qian, Kanaka Rajan  

**Link**: [PDF](https://arxiv.org/pdf/2506.06981)  

**Abstract**: Understanding the behavior of deep reinforcement learning (DRL) agents -- particularly as task and agent sophistication increase -- requires more than simple comparison of reward curves, yet standard methods for behavioral analysis remain underdeveloped in DRL. We apply tools from neuroscience and ethology to study DRL agents in a novel, complex, partially observable environment, ForageWorld, designed to capture key aspects of real-world animal foraging -- including sparse, depleting resource patches, predator threats, and spatially extended arenas. We use this environment as a platform for applying joint behavioral and neural analysis to agents, revealing detailed, quantitatively grounded insights into agent strategies, memory, and planning. Contrary to common assumptions, we find that model-free RNN-based DRL agents can exhibit structured, planning-like behavior purely through emergent dynamics -- without requiring explicit memory modules or world models. Our results show that studying DRL agents like animals -- analyzing them with neuroethology-inspired tools that reveal structure in both behavior and neural dynamics -- uncovers rich structure in their learning dynamics that would otherwise remain invisible. We distill these tools into a general analysis framework linking core behavioral and representational features to diagnostic methods, which can be reused for a wide range of tasks and agents. As agents grow more complex and autonomous, bridging neuroscience, cognitive science, and AI will be essential -- not just for understanding their behavior, but for ensuring safe alignment and maximizing desirable behaviors that are hard to measure via reward. We show how this can be done by drawing on lessons from how biological intelligence is studied. 

**Abstract (ZH)**: 理解深度强化学习（DRL）代理的行为——尤其是随着任务和代理复杂性的增加——需要超出简单的奖励曲线比较，而标准的行为分析方法在DRL领域仍然发展不足。我们应用神经科学和行为学的工具，研究代理在ForageWorld这一新颖、复杂、部分可观测环境中的行为，该环境旨在捕捉现实世界动物觅食的关键方面，包括稀疏、可耗尽的资源斑块、捕食者威胁以及空间扩展的竞技场。我们使用这一环境作为平台，进行联合行为和神经元分析，揭示了代理策略、记忆和规划的详细、量化的见解。与常见的假设相反，我们发现，无模型的基于RNN的DRL代理可以通过涌现动力学表现出结构化的、类似于计划的行为，而无需 Explicit的记忆模块或世界模型。我们的结果表明，将DRL代理类比于动物进行研究——使用神经行为学启发的工具来分析其行为和神经动力学中的结构——揭示了他们学习动态中的丰富结构，这些结构否则将是难以察觉的。我们将这些工具提炼为一种通用的分析框架，将核心行为和表示特征与诊断方法联系起来，该框架可以用于一系列任务和代理。随着代理变得越来越复杂和自主，整合神经科学、认知科学和AI将是必不可少的——不仅是为了理解其行为，也是为了确保安全对齐并最大化那些难以通过奖励衡量的有利行为。我们展示了如何通过借鉴研究生物智能的方法来实现这一点。 

---
# Long-Tailed Learning for Generalized Category Discovery 

**Title (ZH)**: 长尾学习在泛化类别发现中的应用 

**Authors**: Cuong Manh Hoang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06965)  

**Abstract**: Generalized Category Discovery (GCD) utilizes labeled samples of known classes to discover novel classes in unlabeled samples. Existing methods show effective performance on artificial datasets with balanced distributions. However, real-world datasets are always imbalanced, significantly affecting the effectiveness of these methods. To solve this problem, we propose a novel framework that performs generalized category discovery in long-tailed distributions. We first present a self-guided labeling technique that uses a learnable distribution to generate pseudo-labels, resulting in less biased classifiers. We then introduce a representation balancing process to derive discriminative representations. By mining sample neighborhoods, this process encourages the model to focus more on tail classes. We conduct experiments on public datasets to demonstrate the effectiveness of the proposed framework. The results show that our model exceeds previous state-of-the-art methods. 

**Abstract (ZH)**: 通用类别发现（GCD）利用已标记的已知类别样本来发现未标记样本中的新类别。现有的方法在具有均衡分布的人工数据集上表现出有效的性能。然而，现实世界的数据集总是不平衡的，显著影响了这些方法的效果。为了解决这个问题，我们提出了一种新的框架，在长尾分布中执行通用类别发现。我们首先提出了一种自我指导的标注技术，使用可学习的分布生成伪标签，从而生成更少偏差的分类器。然后，我们引入了一种表示平衡过程来提取判别性表示。通过挖掘样本邻域，这个过程鼓励模型更多关注尾部类别。我们在公共数据集上进行了实验，以证明所提出框架的有效性。结果显示，我们的模型超越了之前的最先进方法。 

---
# Deontically Constrained Policy Improvement in Reinforcement Learning Agents 

**Title (ZH)**: 契约约束的强化学习代理策略改进 

**Authors**: Alena Makarova, Houssam Abbas  

**Link**: [PDF](https://arxiv.org/pdf/2506.06959)  

**Abstract**: Markov Decision Processes (MDPs) are the most common model for decision making under uncertainty in the Machine Learning community. An MDP captures non-determinism, probabilistic uncertainty, and an explicit model of action. A Reinforcement Learning (RL) agent learns to act in an MDP by maximizing a utility function. This paper considers the problem of learning a decision policy that maximizes utility subject to satisfying a constraint expressed in deontic logic. In this setup, the utility captures the agent's mission - such as going quickly from A to B. The deontic formula represents (ethical, social, situational) constraints on how the agent might achieve its mission by prohibiting classes of behaviors. We use the logic of Expected Act Utilitarianism, a probabilistic stit logic that can be interpreted over controlled MDPs. We develop a variation on policy improvement, and show that it reaches a constrained local maximum of the mission utility. Given that in stit logic, an agent's duty is derived from value maximization, this can be seen as a way of acting to simultaneously maximize two value functions, one of which is implicit, in a bi-level structure. We illustrate these results with experiments on sample MDPs. 

**Abstract (ZH)**: 马尔可夫决策过程（MDPs）是机器学习社区中处理不确定性决策最常用的模型。本文考虑了在满足形述于规范逻辑的约束条件下，学习最大化目标价值的决策策略的问题。在这种设置中，价值代表代理的任务，如从A快速到达B。规范公式代表代理如何实现其任务的（伦理的、社会的、情境的）约束，通过禁止某些行为类别。我们使用期望行为功利逻辑，这是一种可以在控制的MDPs上进行解释的概率stit逻辑。我们开发了一种策略改进的变体，并证明它可以在任务价值上达到一个约束局部最大值。由于在stit逻辑中，代理的职责源于价值最大化，这可以被视为同时最大化两个价值函数的一种方式，其中一个价值函数是隐含的，其结构具有多层次性。我们通过在样本MDPs上的实验来说明这些结果。 

---
# The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity 

**Title (ZH)**: 思维的幻象：通过问题复杂性的视角理解推理模型的优势与局限 

**Authors**: Parshin Shojaee, Iman Mirzadeh, Keivan Alizadeh, Maxwell Horton, Samy Bengio, Mehrdad Farajtabar  

**Link**: [PDF](https://arxiv.org/pdf/2506.06941)  

**Abstract**: Recent generations of language models have introduced Large Reasoning Models (LRMs) that generate detailed thinking processes before providing answers. While these models demonstrate improved performance on reasoning benchmarks, their fundamental capabilities, scaling properties, and limitations remain insufficiently understood. Current evaluations primarily focus on established math and coding benchmarks, emphasizing final answer accuracy. However, this evaluation paradigm often suffers from contamination and does not provide insights into the reasoning traces. In this work, we systematically investigate these gaps with the help of controllable puzzle environments that allow precise manipulation of complexity while maintaining consistent logical structures. This setup enables the analysis of not only final answers but also the internal reasoning traces, offering insights into how LRMs think. Through extensive experiments, we show that LRMs face a complete accuracy collapse beyond certain complexities. Moreover, they exhibit a counterintuitive scaling limit: their reasoning effort increases with problem complexity up to a point, then declines despite having remaining token budget. By comparing LRMs with their standard LLM counterparts under same inference compute, we identify three performance regimes: (1) low-complexity tasks where standard models outperform LRMs, (2) medium-complexity tasks where LRMs demonstrates advantage, and (3) high-complexity tasks where both models face complete collapse. We found that LRMs have limitations in exact computation: they fail to use explicit algorithms and reason inconsistently across scales. We also investigate the reasoning traces in more depth, studying the patterns of explored solutions and analyzing the models' computational behavior, shedding light on their strengths, limitations, and raising questions about their reasoning capabilities. 

**Abstract (ZH)**: Recent Generations of Language Models Have Introduced Large Reasoning Models (LRMs) That Generate Detailed Thinking Processes Before Providing Answers: Systematically Investigating Their Fundamental Capabilities, Scaling Properties, and Limitations 

---
# An Agentic Framework for Autonomous Metamaterial Modeling and Inverse Design 

**Title (ZH)**: 自主元材料建模与逆向设计的能动框架 

**Authors**: Darui Lu, Jordan M. Malof, Willie J. Padilla  

**Link**: [PDF](https://arxiv.org/pdf/2506.06935)  

**Abstract**: Recent significant advances in integrating multiple Large Language Model (LLM) systems have enabled Agentic Frameworks capable of performing complex tasks autonomously, including novel scientific research. We develop and demonstrate such a framework specifically for the inverse design of photonic metamaterials. When queried with a desired optical spectrum, the Agent autonomously proposes and develops a forward deep learning model, accesses external tools via APIs for tasks like simulation and optimization, utilizes memory, and generates a final design via a deep inverse method. The framework's effectiveness is demonstrated in its ability to automate, reason, plan, and adapt. Notably, the Agentic Framework possesses internal reflection and decision flexibility, permitting highly varied and potentially novel outputs. 

**Abstract (ZH)**: 近期在整合多个大型语言模型系统方面的显著进展使其成为可能，构建出能够自主执行复杂任务的代理框架，包括新型科学研究。我们开发并展示了这样一种框架，专门用于光子 metamaterials 的逆向设计。当用期望的光谱查询时，代理自主地提出并开发一个前向深度学习模型，通过API访问外部工具进行仿真和优化任务，利用记忆，并通过深度逆向方法生成最终设计。该框架的有效性体现在其能够自动化、推理、规划和适应。值得注意的是，代理框架具有内部反馈和决策灵活性，能够产生高度多样且可能新颖的输出。 

---
# Boosting LLM Reasoning via Spontaneous Self-Correction 

**Title (ZH)**: 通过自发自我修正提升大模型推理能力 

**Authors**: Xutong Zhao, Tengyu Xu, Xuewei Wang, Zhengxing Chen, Di Jin, Liang Tan, Yen-Ting, Zishun Yu, Zhuokai Zhao, Yun He, Sinong Wang, Han Fang, Sarath Chandar, Chen Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06923)  

**Abstract**: While large language models (LLMs) have demonstrated remarkable success on a broad range of tasks, math reasoning remains a challenging one. One of the approaches for improving math reasoning is self-correction, which designs self-improving loops to let the model correct its own mistakes. However, existing self-correction approaches treat corrections as standalone post-generation refinements, relying on extra prompt and system designs to elicit self-corrections, instead of performing real-time, spontaneous self-corrections in a single pass. To address this, we propose SPOC, a spontaneous self-correction approach that enables LLMs to generate interleaved solutions and verifications in a single inference pass, with generation dynamically terminated based on verification outcomes, thereby effectively scaling inference time compute. SPOC considers a multi-agent perspective by assigning dual roles -- solution proposer and verifier -- to the same model. We adopt a simple yet effective approach to generate synthetic data for fine-tuning, enabling the model to develop capabilities for self-verification and multi-agent collaboration. We further improve its solution proposal and verification accuracy through online reinforcement learning. Experiments on mathematical reasoning benchmarks show that SPOC significantly improves performance. Notably, SPOC boosts the accuracy of Llama-3.1-8B and 70B Instruct models, achieving gains of 8.8% and 11.6% on MATH500, 10.0% and 20.0% on AMC23, and 3.3% and 6.7% on AIME24, respectively. 

**Abstract (ZH)**: 自发自我纠正方法：SPOC在数学推理中的应用 

---
# Causal Graph based Event Reasoning using Semantic Relation Experts 

**Title (ZH)**: 基于因果图的事件推理方法：语义关系专家的应用 

**Authors**: Mahnaz Koupaee, Xueying Bai, Mudan Chen, Greg Durrett, Nathanael Chambers, Niranjan Balasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2506.06910)  

**Abstract**: Understanding how events in a scenario causally connect with each other is important for effectively modeling and reasoning about events. But event reasoning remains a difficult challenge, and despite recent advances, Large Language Models (LLMs) still struggle to accurately identify causal connections between events. This struggle leads to poor performance on deeper reasoning tasks like event forecasting and timeline understanding. To address this challenge, we investigate the generation of causal event graphs (e.g., A enables B) as a parallel mechanism to help LLMs explicitly represent causality during inference. This paper evaluates both how to generate correct graphs as well as how graphs can assist reasoning. We propose a collaborative approach to causal graph generation where we use LLMs to simulate experts that focus on specific semantic relations. The experts engage in multiple rounds of discussions which are then consolidated by a final expert. Then, to demonstrate the utility of causal graphs, we use them on multiple downstream applications, and also introduce a new explainable event prediction task that requires a causal chain of events in the explanation. These explanations are more informative and coherent than baseline generations. Finally, our overall approach not finetuned on any downstream task, achieves competitive results with state-of-the-art models on both forecasting and next event prediction tasks. 

**Abstract (ZH)**: 理解场景中事件之间的因果联系对于有效建模和推理事件至关重要，但事件推理仍然是一个艰巨的挑战，尽管近期有进展，大语言模型（LLMs）仍然难以准确识别事件间的因果关系。这种困难导致在事件预测和时间线理解等更深层的推理任务上表现不佳。为应对这一挑战，我们研究因果事件图（例如A导致B）的生成作为平行机制，以帮助LLMs在推理过程中明确表示因果关系。本文评估了如何生成正确的图以及图如何辅助推理。我们提出了一种协作式的因果图生成方法，使用LLMs模拟专注于特定语义关系的专家。专家进行多轮讨论，然后由最终专家汇总。为展示因果图的实用性，我们将其应用于多个下游应用，并引入了一个新的可解释事件预测任务，该任务要求解释中包含因果链。这些解释比基线生成结果更具有信息性和连贯性。最终，我们的整体方法在任何下游任务上未进行微调，分别在事件预测和下一个事件预测任务上取得了与先进模型竞争力的结果。 

---
# Meta-Adaptive Prompt Distillation for Few-Shot Visual Question Answering 

**Title (ZH)**: 元自适应提示精简ethod及其在少样本视觉问答中的应用 

**Authors**: Akash Gupta, Amos Storkey, Mirella Lapata  

**Link**: [PDF](https://arxiv.org/pdf/2506.06905)  

**Abstract**: Large Multimodal Models (LMMs) often rely on in-context learning (ICL) to perform new tasks with minimal supervision. However, ICL performance, especially in smaller LMMs, is inconsistent and does not always improve monotonically with increasing examples. We hypothesize that this occurs due to the LMM being overwhelmed by additional information present in the image embeddings, which is not required for the downstream task. To address this, we propose a meta-learning approach that provides an alternative for inducing few-shot capabilities in LMMs, using a fixed set of soft prompts that are distilled from task-relevant image features and can be adapted at test time using a few examples. To facilitate this distillation, we introduce an attention-mapper module that can be easily integrated with the popular LLaVA v1.5 architecture and is jointly learned with soft prompts, enabling task adaptation in LMMs under low-data regimes with just a few gradient steps. Evaluation on the VL-ICL Bench shows that our method consistently outperforms ICL and related prompt-tuning approaches, even under image perturbations, improving task induction and reasoning across visual question answering tasks. 

**Abstract (ZH)**: 大型多模态模型中的少量样本学习：通过固定集合的软提示实现有效元学习 

---
# KnowCoder-V2: Deep Knowledge Analysis 

**Title (ZH)**: KnowCoder-V2: 深度知识分析 

**Authors**: Zixuan Li, Wenxuan Liu, Long Bai, Chunmao Zhang, Wei Li, Fenghui Zhang, Quanxin Jin, Ruoyun He, Zhuo Chen, Zhilei Hu, Fei Wang, Bingbing Xu, Xuhui Jiang, Xiaolong Jin, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.06881)  

**Abstract**: Deep knowledge analysis tasks always involve the systematic extraction and association of knowledge from large volumes of data, followed by logical reasoning to discover insights. However, to solve such complex tasks, existing deep research frameworks face three major challenges: 1) They lack systematic organization and management of knowledge; 2) They operate purely online, making it inefficient for tasks that rely on shared and large-scale knowledge; 3) They cannot perform complex knowledge computation, limiting their abilities to produce insightful analytical results. Motivated by these, in this paper, we propose a \textbf{K}nowledgeable \textbf{D}eep \textbf{R}esearch (\textbf{KDR}) framework that empowers deep research with deep knowledge analysis capability. Specifically, it introduces an independent knowledge organization phase to preprocess large-scale, domain-relevant data into systematic knowledge offline. Based on this knowledge, it extends deep research with an additional kind of reasoning steps that perform complex knowledge computation in an online manner. To enhance the abilities of LLMs to solve knowledge analysis tasks in the above framework, we further introduce \textbf{\KCII}, an LLM that bridges knowledge organization and reasoning via unified code generation. For knowledge organization, it generates instantiation code for predefined classes, transforming data into knowledge objects. For knowledge computation, it generates analysis code and executes on the above knowledge objects to obtain deep analysis results. Experimental results on more than thirty datasets across six knowledge analysis tasks demonstrate the effectiveness of \KCII. Moreover, when integrated into the KDR framework, \KCII can generate high-quality reports with insightful analytical results compared to the mainstream deep research framework. 

**Abstract (ZH)**: 一种具备深度知识分析能力的KDResearch框架 

---
# Incorporating Failure of Machine Learning in Dynamic Probabilistic Safety Assurance 

**Title (ZH)**: 在动态概率安全保证中纳入机器学习失效因素 

**Authors**: Razieh Arshadizadeh, Mahmoud Asgari, Zeinab Khosravi, Yiannis Papadopoulos, Koorosh Aslansefat  

**Link**: [PDF](https://arxiv.org/pdf/2506.06868)  

**Abstract**: Machine Learning (ML) models are increasingly integrated into safety-critical systems, such as autonomous vehicle platooning, to enable real-time decision-making. However, their inherent imperfection introduces a new class of failure: reasoning failures often triggered by distributional shifts between operational and training data. Traditional safety assessment methods, which rely on design artefacts or code, are ill-suited for ML components that learn behaviour from data. SafeML was recently proposed to dynamically detect such shifts and assign confidence levels to the reasoning of ML-based components. Building on this, we introduce a probabilistic safety assurance framework that integrates SafeML with Bayesian Networks (BNs) to model ML failures as part of a broader causal safety analysis. This allows for dynamic safety evaluation and system adaptation under uncertainty. We demonstrate the approach on an simulated automotive platooning system with traffic sign recognition. The findings highlight the potential broader benefits of explicitly modelling ML failures in safety assessment. 

**Abstract (ZH)**: 机器学习模型正越来越多地集成到自动驾驶车辆编队等安全关键系统中，以实现实时决策。然而，其固有的不完美性引入了一类新的故障：由于运行数据和训练数据分布变化引发的推理故障。传统的安全评估方法依赖于设计 artefacts 或代码，不适合学习行为的 ML 组件。近期提出了 SafeML 来动态检测这类变化并为基于 ML 的组件的推理赋予置信水平。在此基础上，我们提出了一种结合 SafeML 和贝叶斯网络 (BNs) 的概率安全保证框架，以将 ML 失败建模为更广泛的因果安全分析的一部分。这允许在不确定性下进行动态安全评估和系统适应。我们通过一个带有交通标志识别的模拟自动驾驶车辆编队系统展示了该方法。研究结果突显了明确建模 ML 失败在安全性评估中的潜在更广泛益处。 

---
# United Minds or Isolated Agents? Exploring Coordination of LLMs under Cognitive Load Theory 

**Title (ZH)**: 群体意识还是孤立的代理？探讨认知负荷理论下大型语言模型的协调机制 

**Authors**: HaoYang Shang, Xuan Liu, Zi Liang, Jie Zhang, Haibo Hu, Song Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.06843)  

**Abstract**: Large Language Models (LLMs) exhibit a notable performance ceiling on complex, multi-faceted tasks, as they often fail to integrate diverse information or adhere to multiple constraints. We posit that such limitation arises when the demands of a task exceed the LLM's effective cognitive load capacity. This interpretation draws a strong analogy to Cognitive Load Theory (CLT) in cognitive science, which explains similar performance boundaries in the human mind, and is further supported by emerging evidence that reveals LLMs have bounded working memory characteristics. Building upon this CLT-grounded understanding, we introduce CoThinker, a novel LLM-based multi-agent framework designed to mitigate cognitive overload and enhance collaborative problem-solving abilities. CoThinker operationalizes CLT principles by distributing intrinsic cognitive load through agent specialization and managing transactional load via structured communication and a collective working memory. We empirically validate CoThinker on complex problem-solving tasks and fabricated high cognitive load scenarios, demonstrating improvements over existing multi-agent baselines in solution quality and efficiency. Our analysis reveals characteristic interaction patterns, providing insights into the emergence of collective cognition and effective load management, thus offering a principled approach to overcoming LLM performance ceilings. 

**Abstract (ZH)**: 大型语言模型在复杂、多面任务中的表现受到认知负荷限制，往往难以整合多样信息或遵守多种约束。我们认为，这种限制源于任务需求超出模型有效认知负荷 capacity。这一解释与认知科学中的认知负荷理论(CLT)有紧密类比关系，该理论解释了人类思维中的类似表现边界，并进一步得到新兴证据的支持，表明大型语言模型具有有界的工作记忆特征。基于这一CLT基础的理解，我们引入了CoThinker，这是一种新颖的基于大型语言模型的多智能体框架，旨在减轻认知过载并增强协作问题解决能力。CoThinker通过智能体专业化分配内在认知负荷，并通过结构化通信和集体工作记忆管理事务性负荷来实现CLT原则。我们通过复杂问题解决任务和高认知负荷场景的实证验证，展示了CoThinker在解决方案质量和效率方面的改进，优于现有的多智能体基线。我们的分析揭示了特征交互模式，为集体认知的涌现和有效负荷管理提供了见解，从而提供了一种原则性的方法来克服大型语言模型的表现天花板。 

---
# Cross-Entropy Games for Language Models: From Implicit Knowledge to General Capability Measures 

**Title (ZH)**: 语言模型的交叉熵游戏：从隐含知识到一般能力衡量 

**Authors**: Clément Hongler, Andrew Emil  

**Link**: [PDF](https://arxiv.org/pdf/2506.06832)  

**Abstract**: Large Language Models (LLMs) define probability measures on text. By considering the implicit knowledge question of what it means for an LLM to know such a measure and what it entails algorithmically, we are naturally led to formulate a series of tasks that go beyond generative sampling, involving forms of summarization, counterfactual thinking, anomaly detection, originality search, reverse prompting, debating, creative solving, etc. These tasks can be formulated as games based on LLM measures, which we call Cross-Entropy (Xent) Games. Xent Games can be single-player or multi-player. They involve cross-entropy scores and cross-entropy constraints, and can be expressed as simple computational graphs and programs. We show the Xent Game space is large enough to contain a wealth of interesting examples, while being constructible from basic game-theoretic consistency axioms. We then discuss how the Xent Game space can be used to measure the abilities of LLMs. This leads to the construction of Xent Game measures: finite families of Xent Games that can be used as capability benchmarks, built from a given scope, by extracting a covering measure. To address the unbounded scope problem associated with the challenge of measuring general abilities, we propose to explore the space of Xent Games in a coherent fashion, using ideas inspired by evolutionary dynamics. 

**Abstract (ZH)**: Large Language Models (LLMs)定义了文本的概率测度。通过考虑LLM知道这样的测度意味着什么及其算法上的含义，自然地带我们提出了超越生成采样的一系列任务，包括总结、反事实思考、异常检测、原创性搜索、逆向提示、辩论、创造性解决等。这些任务可以基于LLM测度形式化为博弈，我们称之为交叉熵（Xent）博弈。Xent博弈可以是单人或多人博弈，涉及交叉熵得分和约束，可以用简单的计算图和程序表达。我们展示了Xent博弈空间足够大，包含丰富的有趣示例，并可以通过基本博弈论一致性公理构建。然后我们讨论了如何使用Xent博弈空间来衡量LLM的能力，从而构造Xent博弈测度：基于给定范围的有限Xent博弈族，可以用作能力基准，通过提取覆盖测度构建。为了解决衡量通用能力时面临的无限范围问题，我们提出以演化动力学启发的方法系统地探索Xent博弈空间。 

---
# Learning What Matters Now: A Dual-Critic Context-Aware RL Framework for Priority-Driven Information Gain 

**Title (ZH)**: 学习当前重要的内容：一种基于优先级驱动信息增益的双重评论家上下文感知 reinforcement 学习框架 

**Authors**: Dimitris Panagopoulos, Adolfo Perrusquia, Weisi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.06786)  

**Abstract**: Autonomous systems operating in high-stakes search-and-rescue (SAR) missions must continuously gather mission-critical information while flexibly adapting to shifting operational priorities. We propose CA-MIQ (Context-Aware Max-Information Q-learning), a lightweight dual-critic reinforcement learning (RL) framework that dynamically adjusts its exploration strategy whenever mission priorities change. CA-MIQ pairs a standard extrinsic critic for task reward with an intrinsic critic that fuses state-novelty, information-location awareness, and real-time priority alignment. A built-in shift detector triggers transient exploration boosts and selective critic resets, allowing the agent to re-focus after a priority revision. In a simulated SAR grid-world, where experiments specifically test adaptation to changes in the priority order of information types the agent is expected to focus on, CA-MIQ achieves nearly four times higher mission-success rates than baselines after a single priority shift and more than three times better performance in multiple-shift scenarios, achieving 100% recovery while baseline methods fail to adapt. These results highlight CA-MIQ's effectiveness in any discrete environment with piecewise-stationary information-value distributions. 

**Abstract (ZH)**: 自主系统在高风险搜救（SAR）任务中的运行必须在不断收集关键任务信息的同时，灵活适应不断变化的操作优先级。我们提出了一种轻量级的双 Critic 强化学习（RL）框架 CA-MIQ（情境感知最大化信息 Q 学习），该框架在任务优先级变化时动态调整其探索策略。CA-MIQ 结合了一个标准的外在 Critic 用于任务奖励，以及一个内在 Critic，该 Critic 融合了状态新颖性、信息位置意识和实时优先级对齐。内置的偏移检测器触发临时的探索增强和选择性的 Critic 重置，使代理在优先级修订后能够重新聚焦。在模拟的 SAR 网格世界中，实验特别测试了代理需要重点关注的信息类型优先级变化的适应能力，CA-MIQ 在单次优先级变化后将任务成功率提高了近四倍，在多次变化场景中的表现提高了三倍以上，并实现了 100% 的恢复，而基线方法则无法适应。这些结果突显了 CA-MIQ 在任何离散环境中信息价值分布分段稳定的效果。 

---
# Bio-Inspired Classification: Combining Information Theory and Spiking Neural Networks -- Influence of the Learning Rules 

**Title (ZH)**: 生物启发分类：结合信息理论与脉冲神经网络——学习规则的影响 

**Authors**: Zofia Rudnicka, Janusz Szczepanski, Agnieszka Pregowska  

**Link**: [PDF](https://arxiv.org/pdf/2506.06750)  

**Abstract**: Training of Spiking Neural Networks (SNN) is challenging due to their unique properties, including temporal dynamics, non-differentiability of spike events, and sparse event-driven activations. In this paper, we widely consider the influence of the type of chosen learning algorithm, including bioinspired learning rules on the accuracy of classification. We proposed a bioinspired classifier based on the combination of SNN and Lempel-Ziv complexity (LZC). This approach synergizes the strengths of SNNs in temporal precision and biological realism with LZC's structural complexity analysis, facilitating efficient and interpretable classification of spatiotemporal neural data. It turned out that the classic backpropagation algorithm achieves excellent classification accuracy, but at extremely high computational cost, which makes it impractical for real-time applications. Biologically inspired learning algorithms such as tempotron and Spikprop provide increased computational efficiency while maintaining competitive classification performance, making them suitable for time-sensitive tasks. The results obtained indicate that the selection of the most appropriate learning algorithm depends on the trade-off between classification accuracy and computational cost as well as application constraints. 

**Abstract (ZH)**: 基于突触神经网络的训练因其实时性、突触事件的非可微性质以及稀疏的事例驱动激活等特点具有挑战性。本文广泛考虑了所选学习算法类型，包括生物启发式学习规则对分类准确性的影响。我们提出了一种结合突触神经网络和Lempel-Ziv复杂性（LZC）的生物启发式分类器。该方法将SNN在时序精确度和生物现实性方面的优势与LZC在结构复杂性分析方面的优势相结合，有助于高效且可解释地分类时空神经数据。经典反向传播算法显示出出色的分类准确性，但计算成本极高，使其不适用于实时应用。生物启发式学习算法如tempotron和Spikprop在保持竞争力的分类性能的同时提高了计算效率，使其适用于时敏任务。实验结果表明，学习算法的选择取决于分类准确性和计算成本之间的权衡以及应用约束。 

---
# AI PsyRoom: Artificial Intelligence Platform for Segmented Yearning and Reactive Outcome Optimization Method 

**Title (ZH)**: AI PsyRoom: 人工智能平台化的分段渴望与反应性结果优化方法 

**Authors**: Yigui Feng, Qinglin Wang, Ke Liu, Xinhai Chen, Bo Yang, Jie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06740)  

**Abstract**: Psychological counseling faces huge challenges due to the growing demand for mental health services and the shortage of trained professionals. Large language models (LLMs) have shown potential to assist psychological counseling, especially in empathy and emotional support. However, existing models lack a deep understanding of emotions and are unable to generate personalized treatment plans based on fine-grained emotions. To address these shortcomings, we present AI PsyRoom, a multi-agent simulation framework designed to enhance psychological counseling by generating empathetic and emotionally nuanced conversations. By leveraging fine-grained emotion classification and a multi-agent framework, we construct a multi-agent PsyRoom A for dialogue reconstruction, generating a high-quality dialogue dataset EmoPsy, which contains 35 sub-emotions, 423 specific emotion scenarios, and 12,350 dialogues. We also propose PsyRoom B for generating personalized treatment plans. Quantitative evaluations demonstrate that AI PsyRoom significantly outperforms state-of-the-art methods, achieving 18% improvement in problem orientation, 23% in expression, 24% in Empathy, and 16% in interactive communication quality. The datasets and models are publicly available, providing a foundation for advancing AI-assisted psychological counseling research. 

**Abstract (ZH)**: 人工智能心理房间：一种多Agent模拟框架，以增强心理咨询服务 

---
# Honey, I shrunk the hypothesis space (through logical preprocessing) 

**Title (ZH)**: 蜂蜜，我缩小了假设空间（通过逻辑预处理） 

**Authors**: Andrew Cropper, Filipe Gouveia, David M. Cerna  

**Link**: [PDF](https://arxiv.org/pdf/2506.06739)  

**Abstract**: Inductive logic programming (ILP) is a form of logical machine learning. The goal is to search a hypothesis space for a hypothesis that generalises training examples and background knowledge. We introduce an approach that 'shrinks' the hypothesis space before an ILP system searches it. Our approach uses background knowledge to find rules that cannot be in an optimal hypothesis regardless of the training examples. For instance, our approach discovers relationships such as "even numbers cannot be odd" and "prime numbers greater than 2 are odd". It then removes violating rules from the hypothesis space. We implement our approach using answer set programming and use it to shrink the hypothesis space of a constraint-based ILP system. Our experiments on multiple domains, including visual reasoning and game playing, show that our approach can substantially reduce learning times whilst maintaining predictive accuracies. For instance, given just 10 seconds of preprocessing time, our approach can reduce learning times from over 10 hours to only 2 seconds. 

**Abstract (ZH)**: 基于逻辑的归纳学习（ILP）是一种形式的逻辑机器学习。目标是在假设空间中搜索能够概括训练例子和背景知识的假设。我们提出了一种在ILP系统搜索假设空间之前缩小假设空间的方法。我们的方法利用背景知识找到一些规则，这些规则无论训练例子如何都不会存在于最优假设之中。例如，我们的方法发现诸如“偶数不能是奇数”和“大于2的质数是奇数”这样的关系，并从中移除违反这些规则的假设。我们使用回答集编程实现该方法，并将其应用于基于约束的ILP系统假设空间的缩小。我们的实验涵盖多个领域，包括视觉推理和游戏玩，表明我们的方法可以在保持预测准确性的同时显著减少学习时间。例如，在只需要10秒的预处理时间后，我们的方法可以将学习时间从超过10小时缩短到仅2秒。 

---
# VisioMath: Benchmarking Figure-based Mathematical Reasoning in LMMs 

**Title (ZH)**: VisioMath: LMMs中基于图形的数学推理benchmarking 

**Authors**: Can Li, Ting Zhang, Mei Wang, Hua Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06727)  

**Abstract**: Large Multimodal Models (LMMs) have demonstrated remarkable problem-solving capabilities across various domains. However, their ability to perform mathematical reasoning when answer options are represented as images--an essential aspect of multi-image comprehension--remains underexplored. To bridge this gap, we introduce VisioMath, a benchmark designed to evaluate mathematical reasoning in multimodal contexts involving image-based answer choices. VisioMath comprises 8,070 images and 1,800 multiple-choice questions, where each answer option is an image, presenting unique challenges to existing LMMs. To the best of our knowledge, VisioMath is the first dataset specifically tailored for mathematical reasoning in image-based-option scenarios, where fine-grained distinctions between answer choices are critical for accurate problem-solving. We systematically evaluate state-of-the-art LMMs on VisioMath and find that even the most advanced models struggle with this task. Notably, GPT-4o achieves only 45.9% accuracy, underscoring the limitations of current models in reasoning over visually similar answer choices. By addressing a crucial gap in existing benchmarks, VisioMath establishes a rigorous testbed for future research, driving advancements in multimodal reasoning. 

**Abstract (ZH)**: 大型多模态模型在处理基于图像的答案选项的数学推理能力方面存在不足：VisioMath数据集的构建与评价 

---
# WorldLLM: Improving LLMs' world modeling using curiosity-driven theory-making 

**Title (ZH)**: WorldLLM：通过好奇心驱动的理论构建提高大语言模型的world建模能力 

**Authors**: Guillaume Levy, Cedric Colas, Pierre-Yves Oudeyer, Thomas Carta, Clement Romac  

**Link**: [PDF](https://arxiv.org/pdf/2506.06725)  

**Abstract**: Large Language Models (LLMs) possess general world knowledge but often struggle to generate precise predictions in structured, domain-specific contexts such as simulations. These limitations arise from their inability to ground their broad, unstructured understanding in specific environments. To address this, we present WorldLLM, a framework that enhances LLM-based world modeling by combining Bayesian inference and autonomous active exploration with reinforcement learning. WorldLLM leverages the in-context learning abilities of LLMs to guide an LLM-based world model's predictions using natural language hypotheses given in its prompt. These hypotheses are iteratively refined through a Bayesian inference framework that leverages a second LLM as the proposal distribution given collected evidence. This evidence is collected using a curiosity-driven reinforcement learning policy that explores the environment to find transitions with a low log-likelihood under our LLM-based predictive model using the current hypotheses. By alternating between refining hypotheses and collecting new evidence, our framework autonomously drives continual improvement of the predictions. Our experiments demonstrate the effectiveness of WorldLLM in a textual game environment that requires agents to manipulate and combine objects. The framework not only enhances predictive accuracy, but also generates human-interpretable theories of environment dynamics. 

**Abstract (ZH)**: 基于贝叶斯推理和自主主动探索的大规模语言模型增强框架：WorldLLM及其在文本游戏环境中的应用 

---
# Integrating AI Planning Semantics into SysML System Models for Automated PDDL File Generation 

**Title (ZH)**: 将AI规划语义集成到SysML系统模型中以实现自动化PDDL文件生成 

**Authors**: Hamied Nabizada, Tom Jeleniewski, Lasse Beers, Maximilian Weigand, Felix Gehlhoff, Alexander Fay  

**Link**: [PDF](https://arxiv.org/pdf/2506.06714)  

**Abstract**: This paper presents a SysML profile that enables the direct integration of planning semantics based on the Planning Domain Definition Language (PDDL) into system models. Reusable stereotypes are defined for key PDDL concepts such as types, predicates, functions and actions, while formal OCL constraints ensure syntactic consistency. The profile was derived from the Backus-Naur Form (BNF) definition of PDDL 3.1 to align with SysML modeling practices. A case study from aircraft manufacturing demonstrates the application of the profile: a robotic system with interchangeable end effectors is modeled and enriched to generate both domain and problem descriptions in PDDL format. These are used as input to a PDDL solver to derive optimized execution plans. The approach supports automated and model-based generation of planning descriptions and provides a reusable bridge between system modeling and AI planning in engineering design. 

**Abstract (ZH)**: 本文提出了一种SysML配置文件，使其能够直接将基于Planning Domain Definition Language (PDDL) 的规划语义集成到系统模型中。定义了类型的可重用模型元，确保语义一致性，并通过形式化的OCL约束确保语法一致性。该配置文件从PDDL 3.1的Backus-Naur Form (BNF) 定义推导得出，与SysML建模实践一致。一个来自航空制造业的案例研究展示了该配置文件的应用：一个带有可更换末端执行器的机器人系统被建模并扩展，以生成PDDL格式的领域描述和问题描述。这些描述被用于PDDL求解器以推导出优化的执行计划。该方法支持自动化的、基于模型的规划描述生成，并在工程设计中提供了系统建模与AI规划之间可重用的桥梁。 

---
# Contextual Experience Replay for Self-Improvement of Language Agents 

**Title (ZH)**: 基于情境体验重播的语言代理自改进方法 

**Authors**: Yitao Liu, Chenglei Si, Karthik Narasimhan, Shunyu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.06698)  

**Abstract**: Large language model (LLM) agents have been applied to sequential decision-making tasks such as web navigation, but without any environment-specific experiences, they often fail in these complex tasks. Moreover, current LLM agents are not designed to continually learn from past experiences during inference time, which could be crucial for them to gain these environment-specific experiences. To address this, we propose Contextual Experience Replay (CER), a training-free framework to enable efficient self-improvement for language agents in their context window. Specifically, CER accumulates and synthesizes past experiences into a dynamic memory buffer. These experiences encompass environment dynamics and common decision-making patterns, allowing the agents to retrieve and augment themselves with relevant knowledge in new tasks, enhancing their adaptability in complex environments. We evaluate CER on the challenging WebArena and VisualWebArena benchmarks. On VisualWebArena, CER achieves a competitive performance of 31.9%. On WebArena, CER also gets a competitive average success rate of 36.7%, relatively improving the success rate of the GPT-4o agent baseline by 51.0%. We also conduct a comprehensive analysis on it to prove its efficiency, validity and understand it better. 

**Abstract (ZH)**: 基于上下文的经验重放（CER）：一种无需训练的框架，使语言代理在上下文窗口内高效自改进 

---
# GELD: A Unified Neural Model for Efficiently Solving Traveling Salesman Problems Across Different Scales 

**Title (ZH)**: GELD：一种高效解决不同规模 Travelling Salesman Problem 的统一神经模型 

**Authors**: Yubin Xiao, Di Wang, Rui Cao, Xuan Wu, Boyang Li, You Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.06634)  

**Abstract**: The Traveling Salesman Problem (TSP) is a well-known combinatorial optimization problem with broad real-world applications. Recent advancements in neural network-based TSP solvers have shown promising results. Nonetheless, these models often struggle to efficiently solve both small- and large-scale TSPs using the same set of pre-trained model parameters, limiting their practical utility. To address this issue, we introduce a novel neural TSP solver named GELD, built upon our proposed broad global assessment and refined local selection framework. Specifically, GELD integrates a lightweight Global-view Encoder (GE) with a heavyweight Local-view Decoder (LD) to enrich embedding representation while accelerating the decision-making process. Moreover, GE incorporates a novel low-complexity attention mechanism, allowing GELD to achieve low inference latency and scalability to larger-scale TSPs. Additionally, we propose a two-stage training strategy that utilizes training instances of different sizes to bolster GELD's generalization ability. Extensive experiments conducted on both synthetic and real-world datasets demonstrate that GELD outperforms seven state-of-the-art models considering both solution quality and inference speed. Furthermore, GELD can be employed as a post-processing method to significantly elevate the quality of the solutions derived by existing neural TSP solvers via spending affordable additional computing time. Notably, GELD is shown as capable of solving TSPs with up to 744,710 nodes, first-of-its-kind to solve this large size TSP without relying on divide-and-conquer strategies to the best of our knowledge. 

**Abstract (ZH)**: 基于广域评估和精域选择的旅行推销员问题神经求解器GELD 

---
# AI Simulation by Digital Twins: Systematic Survey, Reference Framework, and Mapping to a Standardized Architecture 

**Title (ZH)**: 数字孪生驱动的AI仿真：系统性综述、参考框架及其与标准化架构的映射 

**Authors**: Xiaoran Liu, Istvan David  

**Link**: [PDF](https://arxiv.org/pdf/2506.06580)  

**Abstract**: Insufficient data volume and quality are particularly pressing challenges in the adoption of modern subsymbolic AI. To alleviate these challenges, AI simulation uses virtual training environments in which AI agents can be safely and efficiently developed with simulated, synthetic data. Digital twins open new avenues in AI simulation, as these high-fidelity virtual replicas of physical systems are equipped with state-of-the-art simulators and the ability to further interact with the physical system for additional data collection. In this article, we report on our systematic survey of digital twin-enabled AI simulation. By analyzing 22 primary studies, we identify technological trends and derive a reference framework to situate digital twins and AI components. Based on our findings, we derive a reference framework and provide architectural guidelines by mapping it onto the ISO 23247 reference architecture for digital twins. Finally, we identify challenges and research opportunities for prospective researchers. 

**Abstract (ZH)**: 现代亚符号人工智能采用中数据量和质量不足的挑战尤为迫切。为缓解这些挑战，AI模拟使用虚拟训练环境，在其中可以安全高效地使用模拟合成数据开发AI代理。数字孪生为AI模拟开辟了新途径，这些高度逼真的物理系统虚拟复制品配备了最先进的模拟器，并能够进一步与物理系统交互以收集额外数据。本文报告了我们对数字孪生赋能的AI模拟的系统性综述。通过分析22篇主要研究，我们确定了技术趋势并推导出一个参考框架，以确定数字孪生和AI组件的位置。基于我们的研究发现，我们推导出一个参考框架并提供架构指南，将其映射到ISO 23247数字孪生参考架构之上。最后，我们识别出未来研究人员面临的挑战和研究机会。 

---
# The Optimization Paradox in Clinical AI Multi-Agent Systems 

**Title (ZH)**: 临床AI多代理系统中的优化悖论 

**Authors**: Suhana Bedi, Iddah Mlauzi, Daniel Shin, Sanmi Koyejo, Nigam H. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2506.06574)  

**Abstract**: Multi-agent artificial intelligence systems are increasingly deployed in clinical settings, yet the relationship between component-level optimization and system-wide performance remains poorly understood. We evaluated this relationship using 2,400 real patient cases from the MIMIC-CDM dataset across four abdominal pathologies (appendicitis, pancreatitis, cholecystitis, diverticulitis), decomposing clinical diagnosis into information gathering, interpretation, and differential diagnosis. We evaluated single agent systems (one model performing all tasks) against multi-agent systems (specialized models for each task) using comprehensive metrics spanning diagnostic outcomes, process adherence, and cost efficiency. Our results reveal a paradox: while multi-agent systems generally outperformed single agents, the component-optimized or Best of Breed system with superior components and excellent process metrics (85.5% information accuracy) significantly underperformed in diagnostic accuracy (67.7% vs. 77.4% for a top multi-agent system). This finding underscores that successful integration of AI in healthcare requires not just component level optimization but also attention to information flow and compatibility between agents. Our findings highlight the need for end to end system validation rather than relying on component metrics alone. 

**Abstract (ZH)**: 多智能体人工智能系统在临床环境中的应用日益增多，但组件级优化与系统级性能之间的关系尚不明确。我们利用MIMIC-CDM数据集中2400个实际患者案例，对四种腹腔疾病（阑尾炎、胰腺炎、胆囊炎、憩室炎）进行了评估，将临床诊断分解为信息收集、解释和鉴别诊断。我们使用涵盖诊断结果、流程遵守和成本效率的综合指标，评估了单智能体系统（一个模型执行所有任务）与多智能体系统（为每个任务专门化模型）的表现。研究结果揭示了一个悖论：尽管多智能体系统通常优于单智能体系统，但具有更优组件和卓越过程指标的优化组件系统（85.5%信息准确性）在诊断准确性方面的表现显著逊于顶级多智能体系统（67.7% vs. 77.4%）。这一发现表明，人工智能在医疗保健中的成功集成不仅需要组件级优化，还需要注意信息流和智能体之间的兼容性。我们的研究结果强调了需要端到端系统验证，而不仅仅是依赖组件指标。 

---
# ScriptDoctor: Automatic Generation of PuzzleScript Games via Large Language Models and Tree Search 

**Title (ZH)**: ScriptDoctor: 通过大型语言模型和树搜索自动生成PuzzleScript游戏 

**Authors**: Sam Earle, Ahmed Khalifa, Muhammad Umair Nasir, Zehua Jiang, Graham Todd, Andrzej Banburski-Fahey, Julian Togelius  

**Link**: [PDF](https://arxiv.org/pdf/2506.06524)  

**Abstract**: There is much interest in using large pre-trained models in Automatic Game Design (AGD), whether via the generation of code, assets, or more abstract conceptualization of design ideas. But so far this interest largely stems from the ad hoc use of such generative models under persistent human supervision. Much work remains to show how these tools can be integrated into longer-time-horizon AGD pipelines, in which systems interface with game engines to test generated content autonomously. To this end, we introduce ScriptDoctor, a Large Language Model (LLM)-driven system for automatically generating and testing games in PuzzleScript, an expressive but highly constrained description language for turn-based puzzle games over 2D gridworlds. ScriptDoctor generates and tests game design ideas in an iterative loop, where human-authored examples are used to ground the system's output, compilation errors from the PuzzleScript engine are used to elicit functional code, and search-based agents play-test generated games. ScriptDoctor serves as a concrete example of the potential of automated, open-ended LLM-based workflows in generating novel game content. 

**Abstract (ZH)**: 基于大型预训练模型的自动游戏设计：ScriptDoctor系统在PuzzleScript中的应用与测试 

---
# Reinforcement Learning for Autonomous Warehouse Orchestration in SAP Logistics Execution: Redefining Supply Chain Agility 

**Title (ZH)**: 基于SAP Logistics Execution的自主仓库 orchestration reinforcement学习：重塑供应链灵活性 

**Authors**: Sumanth Pillella  

**Link**: [PDF](https://arxiv.org/pdf/2506.06523)  

**Abstract**: In an era of escalating supply chain demands, SAP Logistics Execution (LE) is pivotal for managing warehouse operations, transportation, and delivery. This research introduces a pioneering framework leveraging reinforcement learning (RL) to autonomously orchestrate warehouse tasks in SAP LE, enhancing operational agility and efficiency. By modeling warehouse processes as dynamic environments, the framework optimizes task allocation, inventory movement, and order picking in real-time. A synthetic dataset of 300,000 LE transactions simulates real-world warehouse scenarios, including multilingual data and operational disruptions. The analysis achieves 95% task optimization accuracy, reducing processing times by 60% compared to traditional methods. Visualizations, including efficiency heatmaps and performance graphs, guide agile warehouse strategies. This approach tackles data privacy, scalability, and SAP integration, offering a transformative solution for modern supply chains. 

**Abstract (ZH)**: 在供应链需求递增的时代，SAP物流执行（LE）对于管理仓库作业、运输和交付至关重要。本研究引入了一种利用强化学习（RL）的创新框架，以自主协调SAP LE中的仓库任务，提升运营灵活性和效率。通过将仓库流程建模为动态环境，该框架实现了实时的任务分配、库存移动和订单捡取优化。使用包含300,000条LE交易的合成数据集模拟实际仓库场景，包括多语言数据和运营中断。分析实现了95%的任务优化准确率，相比传统方法，处理时间减少60%。可视化工具，包括效率热图和性能图表，指导灵活的仓库策略。该方法解决数据隐私、扩展性和SAP集成问题，提供了一种现代供应链转型解决方案。 

---
# SIGMA: Refining Large Language Model Reasoning via Sibling-Guided Monte Carlo Augmentation 

**Title (ZH)**: SIGMA：通过兄弟引导的蒙特卡洛 augmentation 完善大型语言模型推理 

**Authors**: Yanwei Ren, Haotian Zhang, Fuxiang Wu, Jiayan Qiu, Jiaxing Huang, Baosheng Yu, Liu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06470)  

**Abstract**: Enhancing large language models by simply scaling up datasets has begun to yield diminishing returns, shifting the spotlight to data quality. Monte Carlo Tree Search (MCTS) has emerged as a powerful technique for generating high-quality chain-of-thought data, yet conventional approaches typically retain only the top-scoring trajectory from the search tree, discarding sibling nodes that often contain valuable partial insights, recurrent error patterns, and alternative reasoning strategies. This unconditional rejection of non-optimal reasoning branches may waste vast amounts of informative data in the whole search tree. We propose SIGMA (Sibling Guided Monte Carlo Augmentation), a novel framework that reintegrates these discarded sibling nodes to refine LLM reasoning. SIGMA forges semantic links among sibling nodes along each search path and applies a two-stage refinement: a critique model identifies overlooked strengths and weaknesses across the sibling set, and a revision model conducts text-based backpropagation to refine the top-scoring trajectory in light of this comparative feedback. By recovering and amplifying the underutilized but valuable signals from non-optimal reasoning branches, SIGMA substantially improves reasoning trajectories. On the challenging MATH benchmark, our SIGMA-tuned 7B model achieves 54.92% accuracy using only 30K samples, outperforming state-of-the-art models trained on 590K samples. This result highlights that our sibling-guided optimization not only significantly reduces data usage but also significantly boosts LLM reasoning. 

**Abstract (ZH)**: 通过重新整合搜索树中的弃用子节点来增强大型语言模型：基于子节点引导的蒙特卡洛 augmentation（SIGMA）方法 

---
# Towards Foundation Model on Temporal Knowledge Graph Reasoning 

**Title (ZH)**: 面向时间知识图谱推理的础模型研究 

**Authors**: Jiaxin Pan, Mojtaba Nayyeri, Osama Mohammed, Daniel Hernandez, Rongchuan Zhang, Cheng Cheng, Steffen Staab  

**Link**: [PDF](https://arxiv.org/pdf/2506.06367)  

**Abstract**: Temporal Knowledge Graphs (TKGs) store temporal facts with quadruple formats (s, p, o, t). Existing Temporal Knowledge Graph Embedding (TKGE) models perform link prediction tasks in transductive or semi-inductive settings, which means the entities, relations, and temporal information in the test graph are fully or partially observed during training. Such reliance on seen elements during inference limits the models' ability to transfer to new domains and generalize to real-world scenarios. A central limitation is the difficulty in learning representations for entities, relations, and timestamps that are transferable and not tied to dataset-specific vocabularies. To overcome these limitations, we introduce the first fully-inductive approach to temporal knowledge graph link prediction. Our model employs sinusoidal positional encodings to capture fine-grained temporal patterns and generates adaptive entity and relation representations using message passing conditioned on both local and global temporal contexts. Our model design is agnostic to temporal granularity and time span, effectively addressing temporal discrepancies across TKGs and facilitating time-aware structural information transfer. As a pretrained, scalable, and transferable model, POSTRA demonstrates strong zero-shot performance on unseen temporal knowledge graphs, effectively generalizing to novel entities, relations, and timestamps. Extensive theoretical analysis and empirical results show that a single pretrained model can improve zero-shot performance on various inductive temporal reasoning scenarios, marking a significant step toward a foundation model for temporal KGs. 

**Abstract (ZH)**: 时间知识图谱（TKGs）以四元组格式（s, p, o, t）存储时间事实。现有的时间知识图嵌入（TKGE）模型在transductive或semi-inductive设置下执行链接预测任务，这意味着在训练过程中测试图中的实体、关系和时间信息是完全或部分可观测的。这种依赖于已见元素的推理限制了模型向新领域转移和泛化到真实世界场景的能力。一个主要限制是难以学习转移性的实体、关系和时间戳表示，这些表示不依赖于特定数据集的词汇表。为了克服这些限制，我们首次引入了完全归纳的时间知识图谱链接预测方法。我们的模型使用正弦位置编码捕捉精细的时间模式，并通过本地和全局时间上下文条件下的消息传递生成自适应的实体和关系表示。我们的模型设计不依赖于时间粒度和时间跨度，有效地解决了时间知识图谱（TKGs）中的时间不一致问题，并促进了时间感知结构信息的转移。作为一种预训练、可扩展且可转移的模型，POSTRA在未见过的时间知识图谱上展示了强大的零样本性能，有效地泛化到新的实体、关系和时间戳。广泛的理论分析和实验证明，单个预训练模型可以改善各种归纳时序推理场景下的零样本性能，标志着朝着时间知识图谱基础模型的一个重要步骤。 

---
# Will artificial agents pursue power by default? 

**Title (ZH)**: 人工代理是否会默认追求权力？ 

**Authors**: Christian Tarsney  

**Link**: [PDF](https://arxiv.org/pdf/2506.06352)  

**Abstract**: Researchers worried about catastrophic risks from advanced AI have argued that we should expect sufficiently capable AI agents to pursue power over humanity because power is a convergent instrumental goal, something that is useful for a wide range of final goals. Others have recently expressed skepticism of these claims. This paper aims to formalize the concepts of instrumental convergence and power-seeking in an abstract, decision-theoretic framework, and to assess the claim that power is a convergent instrumental goal. I conclude that this claim contains at least an element of truth, but might turn out to have limited predictive utility, since an agent's options cannot always be ranked in terms of power in the absence of substantive information about the agent's final goals. However, the fact of instrumental convergence is more predictive for agents who have a good shot at attaining absolute or near-absolute power. 

**Abstract (ZH)**: 研究人员关于先进AI带来灾难性风险的担忧认为，我们应预期足够有能力的AI代理会追求对人类的权力，因为权力是一个趋同的工具性目标，对多种最终目标都有用。近期有观点对此表示怀疑。本文旨在通过抽象的决策理论框架来形式化工具性趋同和权力追求的概念，并评估权力是否是一个趋同的工具性目标的主张。我得出结论，该主张包含至少一个真实元素，但由于缺乏关于代理最终目标的实质信息，代理的选项并不总是可以通过权力来排序，因此可能具有有限的预测效用。然而，趋同性的事实对那些有可能获得绝对或近绝对权力的代理更具有预测性。 

---
# Memory OS of AI Agent 

**Title (ZH)**: AI代理的内存操作系统 

**Authors**: Jiazheng Kang, Mingming Ji, Zhe Zhao, Ting Bai  

**Link**: [PDF](https://arxiv.org/pdf/2506.06326)  

**Abstract**: Large Language Models (LLMs) face a crucial challenge from fixed context windows and inadequate memory management, leading to a severe shortage of long-term memory capabilities and limited personalization in the interactive experience with AI agents. To overcome this challenge, we innovatively propose a Memory Operating System, i.e., MemoryOS, to achieve comprehensive and efficient memory management for AI agents. Inspired by the memory management principles in operating systems, MemoryOS designs a hierarchical storage architecture and consists of four key modules: Memory Storage, Updating, Retrieval, and Generation. Specifically, the architecture comprises three levels of storage units: short-term memory, mid-term memory, and long-term personal memory. Key operations within MemoryOS include dynamic updates between storage units: short-term to mid-term updates follow a dialogue-chain-based FIFO principle, while mid-term to long-term updates use a segmented page organization strategy. Our pioneering MemoryOS enables hierarchical memory integration and dynamic updating. Extensive experiments on the LoCoMo benchmark show an average improvement of 49.11% on F1 and 46.18% on BLEU-1 over the baselines on GPT-4o-mini, showing contextual coherence and personalized memory retention in long conversations. The implementation code is open-sourced at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）面临固定上下文窗口和内存管理不足的挑战，导致长期记忆能力严重短缺和与AI代理互动体验的个性化受限。为克服这一挑战，我们创新地提出了一个内存操作系统，即MemoryOS，以实现对AI代理的全面高效内存管理。受操作系统内存管理原则的启发，MemoryOS 设计了分层存储架构，并包含四个关键模块：内存存储、更新、检索和生成。具体而言，架构包括三个级别的存储单元：短期记忆、中期记忆和长期个性化记忆。MemoryOS 中的关键操作包括存储单元之间的动态更新：短期到中期的更新遵循基于对话链的FIFO原则，而中期到长期的更新采用分段页面组织策略。我们的开创性 MemoryOS 实现了分层记忆集成和动态更新。在 LoCoMo 基准上的 extensive 实验表明，与 GPT-4o-mini 的基线相比，F1 平均提高了 49.11%，BLEU-1 提高了 46.18%，显示出长对话中的上下文一致性和个性化记忆保留。源代码在此处开放获取：this https URL。 

---
# Mapping Human-Agent Co-Learning and Co-Adaptation: A Scoping Review 

**Title (ZH)**: 人类-代理共学习与共适应的映射：一项范围性文献综述 

**Authors**: Shruti Kumar, Xiaoyu Chen, Xiaomei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06324)  

**Abstract**: Several papers have delved into the challenges of human-AI-robot co-learning and co-adaptation. It has been noted that the terminology used to describe this collaborative relationship in existing studies needs to be more consistent. For example, the prefix "co" is used interchangeably to represent both "collaborative" and "mutual," and the terms "co-learning" and "co-adaptation" are sometimes used interchangeably. However, they can reflect subtle differences in the focus of the studies. The current scoping review's primary research question (RQ1) aims to gather existing papers discussing this collaboration pattern and examine the terms researchers use to describe this human-agent relationship. Given the relative newness of this area of study, we are also keen on exploring the specific types of intelligent agents and task domains that have been considered in existing research (RQ2). This exploration is significant as it can shed light on the diversity of human-agent interactions, from one-time to continuous learning/adaptation scenarios. It can also help us understand the dynamics of human-agent interactions in different task domains, guiding our expectations towards research situated in dynamic, complex domains. Our third objective (RQ3) is to investigate the cognitive theories and frameworks that have been utilized in existing studies to measure human-agent co-learning and co-adaptation. This investigation is crucial as it can help us understand the theoretical underpinnings of human-agent collaboration and adaptation, and it can also guide us in identifying any new frameworks proposed specifically for this type of relationship. 

**Abstract (ZH)**: Several篇论文探讨了人类-人工智能-机器人共同学习和适应的挑战。研究中用于描述这一合作关系的术语需要更加一致。例如，“co-”前缀既被用来表示“协作”也表示“相互”，同时，“共同学习”和“共同适应”这两个术语有时也被互换使用。然而，它们可以反映出研究重点的细微差别。本综述研究的主要研究问题（RQ1）旨在收集讨论这种合作模式的现有论文，并检查研究人员用来描述人机关系的术语。鉴于这一研究领域的相对新颖性，我们还希望通过研究现有研究中考虑的具体类型智能代理和任务领域（RQ2），来探索人机交互的多样性，从一次性学习/适应到连续学习/适应场景。这有助于我们理解不同任务领域中人机交互的动力学，指导我们在动态复杂领域中的研究预期。我们的第三个研究目标（RQ3）是调查现有研究中用于衡量人类-代理共同学习和共同适应的认知理论和框架。这一调查对于理解人机协作和适应的理论基础至关重要，也有助于我们识别任何为这种关系类型专门提出的新框架。 

---
# Large Language Models and Their Applications in Roadway Safety and Mobility Enhancement: A Comprehensive Review 

**Title (ZH)**: 大型语言模型及其在道路安全与通行能力提升中的应用：一项全面综述 

**Authors**: Muhammad Monjurul Karim, Yan Shi, Shucheng Zhang, Bingzhang Wang, Mehrdad Nasri, Yinhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06301)  

**Abstract**: Roadway safety and mobility remain critical challenges for modern transportation systems, demanding innovative analytical frameworks capable of addressing complex, dynamic, and heterogeneous environments. While traditional engineering methods have made progress, the complexity and dynamism of real-world traffic necessitate more advanced analytical frameworks. Large Language Models (LLMs), with their unprecedented capabilities in natural language understanding, knowledge integration, and reasoning, represent a promising paradigm shift. This paper comprehensively reviews the application and customization of LLMs for enhancing roadway safety and mobility. A key focus is how LLMs are adapted -- via architectural, training, prompting, and multimodal strategies -- to bridge the "modality gap" with transportation's unique spatio-temporal and physical data. The review systematically analyzes diverse LLM applications in mobility (e.g., traffic flow prediction, signal control) and safety (e.g., crash analysis, driver behavior assessment,). Enabling technologies such as V2X integration, domain-specific foundation models, explainability frameworks, and edge computing are also examined. Despite significant potential, challenges persist regarding inherent LLM limitations (hallucinations, reasoning deficits), data governance (privacy, bias), deployment complexities (sim-to-real, latency), and rigorous safety assurance. Promising future research directions are highlighted, including advanced multimodal fusion, enhanced spatio-temporal reasoning, human-AI collaboration, continuous learning, and the development of efficient, verifiable systems. This review provides a structured roadmap of current capabilities, limitations, and opportunities, underscoring LLMs' transformative potential while emphasizing the need for responsible innovation to realize safer, more intelligent transportation systems. 

**Abstract (ZH)**: 道路安全与流动性仍然是现代交通系统的关键挑战，要求具备解决复杂、动态和异质环境的创新分析框架。尽管传统工程方法取得了进展，但实际交通的复杂性和动态性需要更先进的分析框架。大型语言模型（LLMs）凭借其前所未有的自然语言理解和知识整合及推理能力，代表了一个有前景的范式转变。本文全面回顾了LLMs在提升道路安全和流动性方面的应用与定制。重点是如何通过架构、训练、提示和多模态策略来适应交通的独特时空和物理数据，以解决“模态差距”。回顾系统地分析了在流动性（如交通流量预测、信号控制）和安全（如碰撞分析、驾驶行为评估）方面应用的多样化LLM应用。还考察了诸如车路协同（V2X）、领域特定基础模型、解释性框架和边缘计算等使能技术。尽管存在巨大潜力，但固有的LLM限制（如幻觉、推理缺陷）、数据治理（如隐私、偏见）、部署复杂性（如仿真到现实、延迟）和严格的安全保障等问题仍存在挑战。指出了具有前景的未来研究方向，包括先进的多模态融合、增强的空间-时间推理、人类与人工智能协作、持续学习以及开发高效、可验证的系统。本回顾为当前能力和局限性提供了结构化的路线图，突显了LLMs的变革潜力，同时强调负责任创新以实现更安全、更智能的交通系统的重要性。 

---
# Deep Research Bench: Evaluating AI Web Research Agents 

**Title (ZH)**: 深度研究台：评估AI网络研究代理 

**Authors**: FutureSearch, Nikos I. Bosse, Jon Evans, Robert G. Gambee, Daniel Hnyk, Peter Mühlbacher, Lawrence Phillips, Dan Schwarz, Jack Wildman  

**Link**: [PDF](https://arxiv.org/pdf/2506.06287)  

**Abstract**: Amongst the most common use cases of modern AI is LLM chat with web search enabled. However, no direct evaluations of the quality of web research agents exist that control for the continually-changing web. We introduce Deep Research Bench, consisting of 89 multi-step web research task instances of varying difficulty across 8 diverse task categories, with the answers carefully worked out by skilled humans. We provide a "RetroSearch" environment with a large frozen set of scraped web pages, and demonstrate that offline "RetroSearch" agents perform comparably to "live web" agents, enabling reliable evaluations of models over time. We provide robust agent tooling and scaffolding to benchmark major LLMs as they are released, including "thinking" models like o3 and Gemini 2.5 Pro. We include automated evaluations of the lengthy agent traces to report progress over time in hallucinations, tool use, and forgetting. Finally, we evaluate the major web research products branded as "Deep Research", "Deep Search", "Search", or "Research." Results are available on a public leaderboard at this https URL. 

**Abstract (ZH)**: 现代AI中最常见的用例之一是具有网络搜索功能的大语言模型对话，然而，没有针对不断变化的网络进行控制的质量评估。我们引入了Deep Research Bench，包括89个不同难度级别的多层次网络研究任务实例，涵盖8个不同的任务类别，并由熟练的人类仔细给出答案。我们提供了一个“RetroSearch”环境，包含大量的冻结网页快照，并证明了离线“RetroSearch”代理与“实时网络”代理表现相当，从而可以在不同时期对模型进行可靠的评估。我们提供了强大的代理工具和结构化方案，以基准测试即将发布的主要大语言模型，包括“思考”模型如o3和Gemini 2.5 Pro。我们还包括了对漫长代理轨迹的自动化评估，以报告随时间推移在幻觉、工具使用和遗忘方面的进展。最后，我们评估了那些标记为“Deep Research”、“Deep Search”、“Search”或“Research”的主要网络研究产品。结果可在以下公共排行榜上查阅：this https URL。 

---
# NFISiS: New Perspectives on Fuzzy Inference Systems for Renewable Energy Forecasting 

**Title (ZH)**: NFISiS: 新视角下的模糊 inference 系统在可再生能源预测中的应用 

**Authors**: Kaike Sa Teles Rocha Alves, Eduardo Pestana de Aguiar  

**Link**: [PDF](https://arxiv.org/pdf/2506.06285)  

**Abstract**: Evolving Fuzzy Systems (eFS) have gained significant attention due to their ability to adaptively update their structure in response to data dynamics while maintaining interpretability. However, the lack of publicly available implementations of these models limits their accessibility and widespread adoption. To address this gap, we present evolvingfuzzysystems, a Python library that provides implementations of several well-established eFS models, including ePL-KRLS-DISCO, ePL+, eMG, ePL, exTS, Simpl\_eTS, and eTS. The library facilitates model evaluation and comparison by offering built-in tools for training, visualization, and performance assessment. The models are evaluated using the fetch\_california\_housing dataset, with performance measured in terms of normalized root-mean-square error (NRMSE), non-dimensional error index (NDEI), and mean absolute percentage error (MAPE). Additionally, computational complexity is analyzed by measuring execution times and rule evolution during training and testing phases. The results highlight ePL as a simple yet efficient model that balances accuracy and computational cost, making it particularly suitable for real-world applications. By making these models publicly available, evolvingfuzzysystems aims to foster research and practical applications in adaptive and interpretable machine learning. 

**Abstract (ZH)**: 自适应模糊系统(eFS)由于能够根据数据动态自适应地更新其结构并保持可解释性而引起了广泛关注。然而，缺少这些模型的公开实现限制了它们的可访问性和广泛应用。为了解决这个问题，我们介绍了evolvingfuzzysystemsPython库，该库提供了多个已被广泛认可的eFS模型的实现，包括ePL-KRLS-DISCO、ePL+、eMG、ePL、exTS、Simpl\_eTS和eTS。该库通过提供用于训练、可视化和性能评估的内置工具，促进了模型的评估与比较。模型使用fetch\_california\_housing数据集进行评估，性能用归一化均方根误差(NRMSE)、非量纲误差指数(NDEI)和平均绝对百分比误差(MAFE)来衡量。此外，通过测量训练和测试阶段的执行时间和规则演变，分析了计算复杂性。结果表明，ePL作为一种简单高效的模型，在准确性和计算成本之间取得了平衡，特别适合实际应用。通过使这些模型公开，evolvingfuzzysystems旨在促进自适应和可解释机器学习领域的研究和实际应用。 

---
# Unreal Patterns 

**Title (ZH)**: Unreal Patterns 

**Authors**: John Beverley, Jim Logan  

**Link**: [PDF](https://arxiv.org/pdf/2506.06284)  

**Abstract**: This paper introduces a framework for representing information about entities that do not exist or may never exist, such as those involving fictional entities, blueprints, simulations, and future scenarios. Traditional approaches that introduce "dummy instances" or rely on modal logic are criticized, and a proposal is defended in which such cases are modeled using the intersections of actual types rather than specific non existent tokens. The paper positions itself within the Basic Formal Ontology and its realist commitments, emphasizing the importance of practical, implementable solutions over purely metaphysical or philosophical proposals, arguing that existing approaches to non existent entities either overcommit to metaphysical assumptions or introduce computational inefficiencies that hinder applications. By developing a structured ontology driven approach to unreal patterns, the paper aims to provide a useful and computationally viable means of handling references to hypothetical or non existent entities. 

**Abstract (ZH)**: 本文介绍了用于表示不存在或可能永远不会存在的实体信息的框架，这些实体涉及虚构实体、蓝图、模拟以及未来场景。批判了传统的引入“占位实例”或依赖模态逻辑的方法，并提出了一种将此类情况建模为实际类型交集而非特定不存在标记的方案。本文立足于基本正式本体及其现实承诺，强调实用可实施的解决方案的重要性，而非纯粹的形而上学或哲学提案，认为现有的对不存在实体的处理方法要么过度承诺了形而上学假设，要么引入了计算效率问题，阻碍了应用。通过发展一种结构化的本体驱动方法来处理不现实的模式，本文旨在提供一种实用且计算上可行的方法来处理对假设或不存在实体的引用。 

---
# Understanding Financial Reasoning in AI: A Multimodal Benchmark and Error Learning Approach 

**Title (ZH)**: 理解AI中的财务推理：一种多模态基准和错误学习方法 

**Authors**: Shuangyan Deng, Haizhou Peng, Jiachen Xu, Chunhou Liu, Ciprian Doru Giurcuaneanu, Jiamou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06282)  

**Abstract**: Effective financial reasoning demands not only textual understanding but also the ability to interpret complex visual data such as charts, tables, and trend graphs. This paper introduces a new benchmark designed to evaluate how well AI models - especially large language and multimodal models - reason in finance-specific contexts. Covering 3,200 expert-level question-answer pairs across 15 core financial topics, the benchmark integrates both textual and visual modalities to reflect authentic analytical challenges in finance. To address limitations in current reasoning approaches, we propose an error-aware learning framework that leverages historical model mistakes and feedback to guide inference, without requiring fine-tuning. Our experiments across state-of-the-art models show that multimodal inputs significantly enhance performance and that incorporating error feedback leads to consistent and measurable improvements. The results highlight persistent challenges in visual understanding and mathematical logic, while also demonstrating the promise of self-reflective reasoning in financial AI systems. Our code and data can be found at https://anonymous/FinMR/CodeData. 

**Abstract (ZH)**: 有效的金融推理不仅需要文本理解，还需要解读复杂的视觉数据，如图表、表格和趋势图。本文介绍了一个新的基准，用于评估AI模型，尤其是大型语言和多模态模型，在金融特定背景下进行推理的能力。该基准涵盖了15个核心金融主题下的3200个专家级的问题-答案对，将文本和视觉模态结合起来，以反映真实的金融分析挑战。为了解决当前推理方法的限制，我们提出了一种错误感知的学习框架，该框架利用历史模型错误和反馈来引导推理，而无需微调。我们在最先进的模型上的实验表明，多模态输入显著提高了性能，并且整合错误反馈带来了持续且可测量的改进。结果凸显了在视觉理解和数学逻辑方面持续存在的挑战，同时也展示了金融AI系统中反思性推理的潜力。我们的代码和数据可在https://anonymous/FinMR/CodeData找到。 

---
# StableMTL: Repurposing Latent Diffusion Models for Multi-Task Learning from Partially Annotated Synthetic Datasets 

**Title (ZH)**: StableMTL: 将潜在扩散模型重新应用于部分标注合成数据集的多任务学习 

**Authors**: Anh-Quan Cao, Ivan Lopes, Raoul de Charette  

**Link**: [PDF](https://arxiv.org/pdf/2506.08013)  

**Abstract**: Multi-task learning for dense prediction is limited by the need for extensive annotation for every task, though recent works have explored training with partial task labels. Leveraging the generalization power of diffusion models, we extend the partial learning setup to a zero-shot setting, training a multi-task model on multiple synthetic datasets, each labeled for only a subset of tasks. Our method, StableMTL, repurposes image generators for latent regression. Adapting a denoising framework with task encoding, per-task conditioning and a tailored training scheme. Instead of per-task losses requiring careful balancing, a unified latent loss is adopted, enabling seamless scaling to more tasks. To encourage inter-task synergy, we introduce a multi-stream model with a task-attention mechanism that converts N-to-N task interactions into efficient 1-to-N attention, promoting effective cross-task sharing. StableMTL outperforms baselines on 7 tasks across 8 benchmarks. 

**Abstract (ZH)**: 利用扩散模型的泛化能力，我们将部分学习扩展到零样本设置，通过训练一个多任务模型在多个合成数据集上工作，每个数据集仅标记少量任务。我们的方法StableMTL重新利用图像生成器进行隐变量回归，并适应去噪框架，采用任务编码、逐任务条件和定制的训练方案。我们采用统一的隐变量损失代替需要谨慎平衡的逐任务损失，从而可以无缝扩展到更多任务。为了促进任务间的协同作用，我们引入了一个多流模型，采用任务注意力机制将N-to-N任务交互转换为高效的1-to-N注意力，促进有效的跨任务共享。StableMTL在8个基准上的7个任务上优于基线方法。 

---
# Vision Transformers Don't Need Trained Registers 

**Title (ZH)**: Vision Transformers Don't Need Trained Registers 

**Authors**: Nick Jiang, Amil Dravid, Alexei Efros, Yossi Gandelsman  

**Link**: [PDF](https://arxiv.org/pdf/2506.08010)  

**Abstract**: We investigate the mechanism underlying a previously identified phenomenon in Vision Transformers -- the emergence of high-norm tokens that lead to noisy attention maps. We observe that in multiple models (e.g., CLIP, DINOv2), a sparse set of neurons is responsible for concentrating high-norm activations on outlier tokens, leading to irregular attention patterns and degrading downstream visual processing. While the existing solution for removing these outliers involves retraining models from scratch with additional learned register tokens, we use our findings to create a training-free approach to mitigate these artifacts. By shifting the high-norm activations from our discovered register neurons into an additional untrained token, we can mimic the effect of register tokens on a model already trained without registers. We demonstrate that our method produces cleaner attention and feature maps, enhances performance over base models across multiple downstream visual tasks, and achieves results comparable to models explicitly trained with register tokens. We then extend test-time registers to off-the-shelf vision-language models to improve their interpretability. Our results suggest that test-time registers effectively take on the role of register tokens at test-time, offering a training-free solution for any pre-trained model released without them. 

**Abstract (ZH)**: 我们研究了在Vision Transformers中先前识别的现象背后的机理——高范数令牌的出现导致了嘈杂的注意力图。我们观察到，在多个模型（如CLIP、DINOv2）中，一组稀疏的神经元负责将高范数激活集中在异常令牌上，导致不规则的注意力模式并恶化下游视觉处理。虽然现有去除这些异常的方法是重新从头训练带有额外学习注册令牌的模型，我们利用我们的发现创造了一种无需训练的方法来缓解这些伪影。通过将我们发现的注册神经元的高范数激活转移到一个额外的未训练令牌中，可以在已经训练好的模型上模拟注册令牌的效果。我们证明了我们的方法产生了更干净的注意力和特征图，增强了多个下游视觉任务的性能，并且成果与明确训练有注册令牌的模型相当。我们随后将测试时注册应用于现成的视觉-语言模型以提高其可解释性。我们的结果表明，测试时注册在测试时有效地承担了注册令牌的角色，为任何未发布注册令牌的预训练模型提供了一种无需训练的解决方案。 

---
# Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion 

**Title (ZH)**: 自我强迫：连接自回归视频扩散模型的训练集与测试集差距 

**Authors**: Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, Eli Shechtman  

**Link**: [PDF](https://arxiv.org/pdf/2506.08009)  

**Abstract**: We introduce Self Forcing, a novel training paradigm for autoregressive video diffusion models. It addresses the longstanding issue of exposure bias, where models trained on ground-truth context must generate sequences conditioned on their own imperfect outputs during inference. Unlike prior methods that denoise future frames based on ground-truth context frames, Self Forcing conditions each frame's generation on previously self-generated outputs by performing autoregressive rollout with key-value (KV) caching during training. This strategy enables supervision through a holistic loss at the video level that directly evaluates the quality of the entire generated sequence, rather than relying solely on traditional frame-wise objectives. To ensure training efficiency, we employ a few-step diffusion model along with a stochastic gradient truncation strategy, effectively balancing computational cost and performance. We further introduce a rolling KV cache mechanism that enables efficient autoregressive video extrapolation. Extensive experiments demonstrate that our approach achieves real-time streaming video generation with sub-second latency on a single GPU, while matching or even surpassing the generation quality of significantly slower and non-causal diffusion models. Project website: this http URL 

**Abstract (ZH)**: 我们介绍了一种新的自激励训练范式Self Forcing，用于自回归视频扩散模型。该方法解决了长期存在的暴露偏差问题，即在利用真实上下文训练的模型在推断过程中必须基于自身不完美的输出生成条件序列。与之前的方法不同，自激励通过在训练过程中使用键值（KV）缓存进行自回归展开，使每一帧的生成依赖于之前自生成的输出。这种策略能够在视频级别提供整体损失监督，直接评估整个生成序列的质量，而不是仅仅依赖传统的逐帧目标。为了保证训练效率，我们采用了一步式扩散模型，并结合了随机梯度截断策略，有效地平衡了计算成本和性能。此外，我们引入了一种滚动键值缓存机制，实现了高效的自回归视频外推。大量实验表明，我们的方法能够在单块GPU上实现亚秒级延迟的实时视频生成，同时匹配甚至超过显著更慢且非因果扩散模型的生成质量。项目网站：this http URL 

---
# Hidden in plain sight: VLMs overlook their visual representations 

**Title (ZH)**: 明目张胆的隐藏：VLMs忽视了它们的视觉表示 

**Authors**: Stephanie Fu, Tyler Bonnen, Devin Guillory, Trevor Darrell  

**Link**: [PDF](https://arxiv.org/pdf/2506.08008)  

**Abstract**: Language provides a natural interface to specify and evaluate performance on visual tasks. To realize this possibility, vision language models (VLMs) must successfully integrate visual and linguistic information. Our work compares VLMs to a direct readout of their visual encoders to understand their ability to integrate across these modalities. Across a series of vision-centric benchmarks (e.g., depth estimation, correspondence), we find that VLMs perform substantially worse than their visual encoders, dropping to near-chance performance. We investigate these results through a series of analyses across the entire VLM: namely 1) the degradation of vision representations, 2) brittleness to task prompt, and 3) the language model's role in solving the task. We find that the bottleneck in performing these vision-centric tasks lies in this third category; VLMs are not effectively using visual information easily accessible throughout the entire model, and they inherit the language priors present in the LLM. Our work helps diagnose the failure modes of open-source VLMs, and presents a series of evaluations useful for future investigations into visual understanding within VLMs. 

**Abstract (ZH)**: 语言为指定和评估视觉任务的性能提供了一个自然接口。为了实现这一可能性，视觉语言模型（VLMs）必须成功地整合视觉和语言信息。我们的工作将VLMs与它们的视觉编码器的直接读出进行比较，以理解它们在跨模态整合方面的能力。在一系列以视觉为中心的标准测试中（例如，深度估计、对应关系），我们发现VLMs的表现远逊于其视觉编码器，性能几乎降到随机水平。我们通过对整个VLM的多个分析来探讨这些结果，即1）视觉表示的退化、2）对任务提示的脆弱性，以及3）语言模型在解决问题中的作用。我们发现，在执行这些视觉中心任务时的瓶颈在于第三类；VLMs未能有效地利用模型中到处可用的视觉信息，并且继承了LLM中存在的语言先验。我们的工作有助于诊断开源VLMs的失败模式，并提出一系列对于未来关于VLM中视觉理解的研究有用的评估方法。 

---
# Dynamic View Synthesis as an Inverse Problem 

**Title (ZH)**: 动态视图合成作为逆问题 

**Authors**: Hidir Yesiltepe, Pinar Yanardag  

**Link**: [PDF](https://arxiv.org/pdf/2506.08004)  

**Abstract**: In this work, we address dynamic view synthesis from monocular videos as an inverse problem in a training-free setting. By redesigning the noise initialization phase of a pre-trained video diffusion model, we enable high-fidelity dynamic view synthesis without any weight updates or auxiliary modules. We begin by identifying a fundamental obstacle to deterministic inversion arising from zero-terminal signal-to-noise ratio (SNR) schedules and resolve it by introducing a novel noise representation, termed K-order Recursive Noise Representation. We derive a closed form expression for this representation, enabling precise and efficient alignment between the VAE-encoded and the DDIM inverted latents. To synthesize newly visible regions resulting from camera motion, we introduce Stochastic Latent Modulation, which performs visibility aware sampling over the latent space to complete occluded regions. Comprehensive experiments demonstrate that dynamic view synthesis can be effectively performed through structured latent manipulation in the noise initialization phase. 

**Abstract (ZH)**: 本工作在无训练设置中将单目视频的动态视图合成视为一个逆问题，通过重新设计预训练视频扩散模型的噪声初始化阶段，实现高保真动态视图合成而无需任何权重更新或辅助模块。我们通过引入一种新颖的噪声表示K阶递归噪声表示来解决由零终端信噪比（SNR）调度引起的决定性反演基本障碍，并推导出该表示的闭式表达式，使得VAE编码和DDIM反演的潜在变量之间实现精确和高效的对齐。为了合成由于相机运动而成为新可见区域的结果，我们引入了随机潜在调制，它在潜在空间中进行感知可见性的采样以完成遮挡区域。全面的实验结果表明，动态视图合成可以通过噪声初始化阶段的结构化潜在操纵来有效实现。 

---
# Audio-Sync Video Generation with Multi-Stream Temporal Control 

**Title (ZH)**: 多流时间控制驱动的同步音频视频生成 

**Authors**: Shuchen Weng, Haojie Zheng, Zheng Chang, Si Li, Boxin Shi, Xinlong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08003)  

**Abstract**: Audio is inherently temporal and closely synchronized with the visual world, making it a naturally aligned and expressive control signal for controllable video generation (e.g., movies). Beyond control, directly translating audio into video is essential for understanding and visualizing rich audio narratives (e.g., Podcasts or historical recordings). However, existing approaches fall short in generating high-quality videos with precise audio-visual synchronization, especially across diverse and complex audio types. In this work, we introduce MTV, a versatile framework for audio-sync video generation. MTV explicitly separates audios into speech, effects, and music tracks, enabling disentangled control over lip motion, event timing, and visual mood, respectively -- resulting in fine-grained and semantically aligned video generation. To support the framework, we additionally present DEMIX, a dataset comprising high-quality cinematic videos and demixed audio tracks. DEMIX is structured into five overlapped subsets, enabling scalable multi-stage training for diverse generation scenarios. Extensive experiments demonstrate that MTV achieves state-of-the-art performance across six standard metrics spanning video quality, text-video consistency, and audio-video alignment. Project page: this https URL. 

**Abstract (ZH)**: 音频本质上是时间上的，并且紧密同步于视觉世界，使其成为可控视频生成（例如电影）的自然对齐和表达性控制信号。超越控制，直接将音频翻译成视频对于理解并可视化丰富的音频叙事（例如播客或历史录音）是必不可少的。然而，现有方法在生成与音频视觉同步精准的高质量视频方面仍存不足，特别是在处理多样且复杂的音频类型时。在本文中，我们引入了MTV，一种多功能的音频同步视频生成框架。MTV明确地将音频分离为语音、效果和音乐轨道，分别实现唇部运动、事件时间与时视觉氛围的分离控制——从而实现精细且语义上对齐的视频生成。为支持此框架，我们还提供了DE MIX数据集，包含高质量的电影级视频和分离的音频轨道。DE MIX被结构化为五个重叠子集，使得跨不同生成场景的可扩展多阶段训练成为可能。广泛的实验证明，MTV在六项标准度量指标（涵盖视频质量、文本-视频一致性及音频-视频对齐）方面均达到了最先进的性能。项目页面：this https URL。 

---
# Reparameterized LLM Training via Orthogonal Equivalence Transformation 

**Title (ZH)**: 通过正交等价变换实现的重参数化大模型训练 

**Authors**: Zeju Qiu, Simon Buchholz, Tim Z. Xiao, Maximilian Dax, Bernhard Schölkopf, Weiyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08001)  

**Abstract**: While large language models (LLMs) are driving the rapid advancement of artificial intelligence, effectively and reliably training these large models remains one of the field's most significant challenges. To address this challenge, we propose POET, a novel reParameterized training algorithm that uses Orthogonal Equivalence Transformation to optimize neurons. Specifically, POET reparameterizes each neuron with two learnable orthogonal matrices and a fixed random weight matrix. Because of its provable preservation of spectral properties of weight matrices, POET can stably optimize the objective function with improved generalization. We further develop efficient approximations that make POET flexible and scalable for training large-scale neural networks. Extensive experiments validate the effectiveness and scalability of POET in training LLMs. 

**Abstract (ZH)**: POET：使用正交等价变换优化神经元的参数化训练算法 

---
# Thinking vs. Doing: Agents that Reason by Scaling Test-Time Interaction 

**Title (ZH)**: 思考 vs. 做事：通过扩展测试时交互进行推理的代理 

**Authors**: Junhong Shen, Hao Bai, Lunjun Zhang, Yifei Zhou, Amrith Setlur, Shengbang Tong, Diego Caples, Nan Jiang, Tong Zhang, Ameet Talwalkar, Aviral Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.07976)  

**Abstract**: The current paradigm of test-time scaling relies on generating long reasoning traces ("thinking" more) before producing a response. In agent problems that require interaction, this can be done by generating thinking traces before acting in the world. However, this process does not allow agents to acquire new information from the environment or adapt their behavior over time. In this work, we propose to scale test-time interaction, an untapped dimension of test-time scaling that increases the agent's interaction horizon to enable running rich behaviors such as exploration, backtracking, and dynamic re-planning within a single rollout. To demonstrate the promise of this scaling dimension, we study the domain of web agents. We first show that even prompting-based interaction scaling without any training can improve task success on web benchmarks non-trivially. Building on this, we introduce TTI (Test-Time Interaction), a curriculum-based online reinforcement learning (RL) approach that trains agents by adaptively adjusting their rollout lengths. Using a Gemma 3 12B model, TTI produces state-of-the-art open-source, open-data web agents on WebVoyager and WebArena benchmarks. We further show that TTI enables agents to balance exploration and exploitation adaptively. Our results establish interaction scaling as a powerful, complementary axis to scaling per-step compute, offering new avenues for training adaptive agents. 

**Abstract (ZH)**: 当前的测试时放大规模的范式依赖于生成长的推理轨迹（“思考”更多）后再生成响应。在需要交互的代理问题中，可以在实际操作前生成思考轨迹。然而，这一过程不允许代理从环境中学到新的信息或随时间调整其行为。在本工作中，我们提出了一种测试时交互放大规模的方法，这是一种未充分利用的测试时放大规模维度，能够扩展代理的交互范围，使代理能够在单个展开过程中运行丰富的行为，如探索、回溯和动态重规划。为了展示这一放大规模维度的潜力，我们研究了网络代理领域。我们首先表明，即使是基于提示的交互放大规模，无需任何训练也能在网页基准测试中显著提高任务成功率。在此基础上，我们引入了TTI（测试时交互）方法，这是一种基于课程的在线强化学习方法，通过自适应调整展开长度来训练代理。使用Gemma 3 12B模型，TTI在WebVoyager和WebArena基准测试中生成了最先进的开源、开源数据网页代理。我们进一步证明，TTI使代理能够自适应地平衡探索和利用。我们的结果确立了交互放大规模作为扩展每步计算量的强大补充维度，并为训练自适应代理开辟了新的途径。 

---
# HeuriGym: An Agentic Benchmark for LLM-Crafted Heuristics in Combinatorial Optimization 

**Title (ZH)**: HeuriGym：组合优化中由大语言模型生成启发式规则的能力基准 

**Authors**: Hongzheng Chen, Yingheng Wang, Yaohui Cai, Hins Hu, Jiajie Li, Shirley Huang, Chenhui Deng, Rongjian Liang, Shufeng Kong, Haoxing Ren, Samitha Samaranayake, Carla P. Gomes, Zhiru Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07972)  

**Abstract**: While Large Language Models (LLMs) have demonstrated significant advancements in reasoning and agent-based problem-solving, current evaluation methodologies fail to adequately assess their capabilities: existing benchmarks either rely on closed-ended questions prone to saturation and memorization, or subjective comparisons that lack consistency and rigor. In this work, we introduce HeuriGym, an agentic framework designed for evaluating heuristic algorithms generated by LLMs for combinatorial optimization problems, characterized by clearly defined objectives and expansive solution spaces. HeuriGym empowers LLMs to propose heuristics, receive evaluative feedback via code execution, and iteratively refine their solutions. We evaluate nine state-of-the-art models on nine problems across domains such as computer systems, logistics, and biology, exposing persistent limitations in tool use, planning, and adaptive reasoning. To quantify performance, we propose the Quality-Yield Index (QYI), a metric that captures both solution pass rate and quality. Even top models like GPT-o4-mini-high and Gemini-2.5-Pro attain QYI scores of only 0.6, well below the expert baseline of 1. Our open-source benchmark aims to guide the development of LLMs toward more effective and realistic problem-solving in scientific and engineering domains. 

**Abstract (ZH)**: 大型语言模型在推理和基于代理的问题解决方面取得了显著进展，但现有的评估方法未能充分评估其能力：现有的基准测试要么依赖于容易饱和且依赖记忆的封闭式问题，要么依赖于缺乏一致性和 rigor 的主观比较。本文介绍了一种名为 HeuriGym 的代理框架，用于评估大型语言模型生成的启发式算法在组合优化问题上的表现，这些问题具有明确的目标和广阔的解决方案空间。HeuriGym 使大型语言模型能够提出启发式算法，通过代码执行接收评估反馈，并逐步优化其解决方案。我们评估了九个最先进的模型在涉及计算机系统、物流和生物学等多个领域的九个问题上的表现，揭示了工具使用、计划和适应性推理方面的持久限制。为了量化性能，我们提出了质量产率指数（QYI），该指标涵盖了解决方案通过率和质量。即使是顶级模型如 GPT-o4-mini-high 和 Gemini-2.5-Pro，其 QYI 也仅达到 0.6，明显低于专家基准的 1。我们的开源基准旨在引导大型语言模型朝向更有效和现实的问题解决方向发展。 

---
# SlideCoder: Layout-aware RAG-enhanced Hierarchical Slide Generation from Design 

**Title (ZH)**: SlideCoder: 带有布局意识的层次化幻灯片生成增强的RAG方法 

**Authors**: Wenxin Tang, Jingyu Xiao, Wenxuan Jiang, Xi Xiao, Yuhang Wang, Xuxin Tang, Qing Li, Yuehe Ma, Junliang Liu, Shisong Tang, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07964)  

**Abstract**: Manual slide creation is labor-intensive and requires expert prior knowledge. Existing natural language-based LLM generation methods struggle to capture the visual and structural nuances of slide designs. To address this, we formalize the Reference Image to Slide Generation task and propose Slide2Code, the first benchmark with difficulty-tiered samples based on a novel Slide Complexity Metric. We introduce SlideCoder, a layout-aware, retrieval-augmented framework for generating editable slides from reference images. SlideCoder integrates a Color Gradient-based Segmentation algorithm and a Hierarchical Retrieval-Augmented Generation method to decompose complex tasks and enhance code generation. We also release SlideMaster, a 7B open-source model fine-tuned with improved reverse-engineered data. Experiments show that SlideCoder outperforms state-of-the-art baselines by up to 40.5 points, demonstrating strong performance across layout fidelity, execution accuracy, and visual consistency. Our code is available at this https URL. 

**Abstract (ZH)**: 手动制作幻灯片劳动密集且需要专家先验知识。现有的基于自然语言的LLM生成方法难以捕捉幻灯片设计的视觉和结构细腻之处。为了解决这一问题，我们形式化了参考图像到幻灯片生成任务，并提出了Slide2Code，这是首个基于新型幻灯片复杂度度量的难度分层样本基准。我们引入了SlideCoder，这是一种布局感知、检索增强的框架，用于从参考图像生成可编辑的幻灯片。SlideCoder结合了基于颜色渐变的分割算法和分层检索增强生成方法来分解复杂任务并增强代码生成。我们还推出了SlideMaster，这是一个7B开源模型，经过改进的逆向工程数据微调。实验结果显示，SlideCoder在布局保真度、执行准确性和视觉一致性方面均表现出色，比最先进的基线方法高出40.5分。我们的代码可在以下链接获得。 

---
# Correlated Errors in Large Language Models 

**Title (ZH)**: 大型语言模型中的相关错误 

**Authors**: Elliot Kim, Avi Garg, Kenny Peng, Nikhil Garg  

**Link**: [PDF](https://arxiv.org/pdf/2506.07962)  

**Abstract**: Diversity in training data, architecture, and providers is assumed to mitigate homogeneity in LLMs. However, we lack empirical evidence on whether different LLMs differ meaningfully. We conduct a large-scale empirical evaluation on over 350 LLMs overall, using two popular leaderboards and a resume-screening task. We find substantial correlation in model errors -- on one leaderboard dataset, models agree 60% of the time when both models err. We identify factors driving model correlation, including shared architectures and providers. Crucially, however, larger and more accurate models have highly correlated errors, even with distinct architectures and providers. Finally, we show the effects of correlation in two downstream tasks: LLM-as-judge evaluation and hiring -- the latter reflecting theoretical predictions regarding algorithmic monoculture. 

**Abstract (ZH)**: 训练数据、架构和提供者的多样性被假设能减轻大语言模型的同质性。然而，我们缺乏关于不同大语言模型之间是否存在有意义差异的实证证据。我们对超过350个大语言模型进行了大规模实证评估，使用了两个流行的排行榜和一份简历筛选任务。我们发现模型错误之间有显著的相关性——在其中一个排行榜数据集中，当两个模型都出错时，它们一致的错误比例占60%。我们确定了驱动模型相关性的因素，包括共享的架构和提供者。然而，至关重要的是，即使使用不同的架构和提供者，更大的和更准确的模型也具有高度相关的错误。最后，我们展示了相关性在两个下游任务中的影响：大语言模型作为评判者的评估和招聘——后者反映了关于算法 monoculture 的理论预测。 

---
# BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models 

**Title (ZH)**: BridgeVLA: 输入-输出对齐以实现高效基于视觉-语言模型的3D manipulation学习 

**Authors**: Peiyan Li, Yixiang Chen, Hongtao Wu, Xiao Ma, Xiangnan Wu, Yan Huang, Liang Wang, Tao Kong, Tieniu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07961)  

**Abstract**: Recently, leveraging pre-trained vision-language models (VLMs) for building vision-language-action (VLA) models has emerged as a promising approach to effective robot manipulation learning. However, only few methods incorporate 3D signals into VLMs for action prediction, and they do not fully leverage the spatial structure inherent in 3D data, leading to low sample efficiency. In this paper, we introduce BridgeVLA, a novel 3D VLA model that (1) projects 3D inputs to multiple 2D images, ensuring input alignment with the VLM backbone, and (2) utilizes 2D heatmaps for action prediction, unifying the input and output spaces within a consistent 2D image space. In addition, we propose a scalable pre-training method that equips the VLM backbone with the capability to predict 2D heatmaps before downstream policy learning. Extensive experiments show the proposed method is able to learn 3D manipulation efficiently and effectively. BridgeVLA outperforms state-of-the-art baseline methods across three simulation benchmarks. In RLBench, it improves the average success rate from 81.4% to 88.2%. In COLOSSEUM, it demonstrates significantly better performance in challenging generalization settings, boosting the average success rate from 56.7% to 64.0%. In GemBench, it surpasses all the comparing baseline methods in terms of average success rate. In real-robot experiments, BridgeVLA outperforms a state-of-the-art baseline method by 32% on average. It generalizes robustly in multiple out-of-distribution settings, including visual disturbances and unseen instructions. Remarkably, it is able to achieve a success rate of 96.8% on 10+ tasks with only 3 trajectories per task, highlighting its extraordinary sample efficiency. Project Website:this https URL 

**Abstract (ZH)**: Recently, 利用预训练的多模态模型构建视觉-语言-动作模型以有效学习机器人操作正逐渐成为一种有前途的方法。然而，目前仅有少数方法将3D信号整合进多模态模型进行动作预测，且这些方法未能充分利用3D数据中固有的空间结构，导致样本效率较低。本文介绍了一种新颖的3D VLA模型BridgeVLA，该模型通过（1）将3D输入投影为多个2D图像，确保输入与多模态模型主干对齐，以及（2）利用2D热图进行动作预测，将输入和输出空间统一在一致的2D图像空间中。此外，我们提出了一种可扩展的预训练方法，使多模态模型主干能够预测2D热图，从而为下游策略学习做好准备。大量实验证明，提出的方法能够高效有效地学习3D操作。BridgeVLA在三个仿真基准上均优于最先进的基线方法。在RLBench中，它将平均成功率从81.4%提升到88.2%。在COLOSSEUM中，它在具有挑战性的泛化设置中表现出显著更好的性能，将平均成功率从56.7%提升到64.0%。在GemBench中，它在平均成功率上超越了所有比较的基线方法。在真实机器人实验中，BridgeVLA在平均上优于最先进的基线方法32%，并且在包括视觉干扰和未见指令在内的多种非分布外设置中表现出鲁棒泛化能力。令人惊讶的是，它仅使用每个任务3条轨迹就能实现96.8%的成功率，突显了其卓越的样本效率。项目网站：this https URL 

---
# ProtocolLLM: RTL Benchmark for SystemVerilog Generation of Communication Protocols 

**Title (ZH)**: ProtocolLLM：用于通信协议SystemVerilog生成的RTL基准测试bench 

**Authors**: Arnav Sheth, Ivaxi Sheth, Mario Fritz  

**Link**: [PDF](https://arxiv.org/pdf/2506.07945)  

**Abstract**: Recent advances in Large Language Models (LLMs) have shown promising capabilities in generating code for general-purpose programming languages. In contrast, their applicability for hardware description languages, particularly for generating synthesizable and functionally correct designs, remains significantly underexplored. HDLs such as SystemVerilog are logic-oriented and demand strict adherence to timing semantics, concurrency, and synthesizability constraints. Moreover, HDL-based design flows encompass a broad set of tasks beyond structural code generation, including testbench development, assertion-based verification, timing closure, and protocol-level integration for on-chip communication. The objective of our paper is to analyze the capabilities of state-of-the-art LLMs in generating SystemVerilog implementations of standard communication protocols, a core component of embedded and System-on-Chip (SoC) architectures. This paper introduces the first benchmark suite targeting four widely used protocols: SPI, I2C, UART, and AXI. We define code generation tasks that capture varying levels of design abstraction and prompt specificity. The generated designs are assessed for syntactic correctness, synthesizability, and functional fidelity via waveform simulation and test benches. 

**Abstract (ZH)**: Recent Advances in Large Language Models in Generating SystemVerilog Implementations of Standard Communication Protocols 

---
# Decoupling the Image Perception and Multimodal Reasoning for Reasoning Segmentation with Digital Twin Representations 

**Title (ZH)**: 解耦图像感知与多模态推理以实现数字孪生表示的语义分割 

**Authors**: Yizhen Li, Dell Zhang, Xuelong Li, Yiqing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.07943)  

**Abstract**: Reasoning Segmentation (RS) is a multimodal vision-text task that requires segmenting objects based on implicit text queries, demanding both precise visual perception and vision-text reasoning capabilities. Current RS approaches rely on fine-tuning vision-language models (VLMs) for both perception and reasoning, but their tokenization of images fundamentally disrupts continuous spatial relationships between objects. We introduce DTwinSeger, a novel RS approach that leverages Digital Twin (DT) representation as an intermediate layer to decouple perception from reasoning. Innovatively, DTwinSeger reformulates RS as a two-stage process, where the first transforms the image into a structured DT representation that preserves spatial relationships and semantic properties and then employs a Large Language Model (LLM) to perform explicit reasoning over this representation to identify target objects. We propose a supervised fine-tuning method specifically for LLM with DT representation, together with a corresponding fine-tuning dataset Seg-DT, to enhance the LLM's reasoning capabilities with DT representations. Experiments show that our method can achieve state-of-the-art performance on two image RS benchmarks and three image referring segmentation benchmarks. It yields that DT representation functions as an effective bridge between vision and text, enabling complex multimodal reasoning tasks to be accomplished solely with an LLM. 

**Abstract (ZH)**: 基于数字孪生的推理分割（DTwinSeger：一种结合数字孪生表示的多模态视图-文本任务方法） 

---
# Mimicking or Reasoning: Rethinking Multi-Modal In-Context Learning in Vision-Language Models 

**Title (ZH)**: 模仿还是推理：重塑视觉语言模型中的多模态在上下文学习 

**Authors**: Chengyue Huang, Yuchen Zhu, Sichen Zhu, Jingyun Xiao, Moises Andrade, Shivang Chopra, Zsolt Kira  

**Link**: [PDF](https://arxiv.org/pdf/2506.07936)  

**Abstract**: Vision-language models (VLMs) are widely assumed to exhibit in-context learning (ICL), a property similar to that of their language-only counterparts. While recent work suggests VLMs can perform multimodal ICL (MM-ICL), studies show they often rely on shallow heuristics -- such as copying or majority voting -- rather than true task understanding. We revisit this assumption by evaluating VLMs under distribution shifts, where support examples come from a dataset different from the query. Surprisingly, performance often degrades with more demonstrations, and models tend to copy answers rather than learn from them. To investigate further, we propose a new MM-ICL with Reasoning pipeline that augments each demonstration with a generated rationale alongside the answer. We conduct extensive and comprehensive experiments on both perception- and reasoning-required datasets with open-source VLMs ranging from 3B to 72B and proprietary models such as Gemini 2.0. We conduct controlled studies varying shot count, retrieval method, rationale quality, and distribution. Our results show limited performance sensitivity across these factors, suggesting that current VLMs do not effectively utilize demonstration-level information as intended in MM-ICL. 

**Abstract (ZH)**: Vision-language模型在分布迁移下的多模态情境学习：有限的性能敏感性 

---
# Diffusion of Responsibility in Collective Decision Making 

**Title (ZH)**: 集体决策中的责任扩散 

**Authors**: Pavel Naumov, Jia Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.07935)  

**Abstract**: The term "diffusion of responsibility'' refers to situations in which multiple agents share responsibility for an outcome, obscuring individual accountability. This paper examines this frequently undesirable phenomenon in the context of collective decision-making mechanisms.
The work shows that if a decision is made by two agents, then the only way to avoid diffusion of responsibility is for one agent to act as a "dictator'', making the decision unilaterally. In scenarios with more than two agents, any diffusion-free mechanism is an "elected dictatorship'' where the agents elect a single agent to make a unilateral decision.
The technical results are obtained by defining a bisimulation of decision-making mechanisms, proving that bisimulation preserves responsibility-related properties, and establishing the results for a smallest bisimular mechanism. 

**Abstract (ZH)**: 责任扩散现象是指多个代理共同承担某结果的责任，从而模糊了个体责任的情况。本文在集体决策机制的背景下探讨了这种通常令人不悦的现象。研究表明，如果决策由两个代理作出，则唯一的避免责任扩散的方法是一个代理充当“独裁者”，单独作出决策。在多于两个代理的情形下，任何无责任扩散的机制都是“选举独裁”，即代理们选举一位代理人单独作出决策。技术成果通过定义决策机制的拟模拟，并证明拟模拟保有一定的责任制性质，以及在最小拟模拟机制上建立结果而获得。 

---
# Uncovering the Functional Roles of Nonlinearity in Memory 

**Title (ZH)**: 揭示非线性在记忆中的功能作用 

**Authors**: Manuel Brenner, Georgia Koppe  

**Link**: [PDF](https://arxiv.org/pdf/2506.07919)  

**Abstract**: Memory and long-range temporal processing are core requirements for sequence modeling tasks across natural language processing, time-series forecasting, speech recognition, and control. While nonlinear recurrence has long been viewed as essential for enabling such mechanisms, recent work suggests that linear dynamics may often suffice. In this study, we go beyond performance comparisons to systematically dissect the functional role of nonlinearity in recurrent networks--identifying both when it is computationally necessary, and what mechanisms it enables. We use Almost Linear Recurrent Neural Networks (AL-RNNs), which allow fine-grained control over nonlinearity, as both a flexible modeling tool and a probe into the internal mechanisms of memory. Across a range of classic sequence modeling tasks and a real-world stimulus selection task, we find that minimal nonlinearity is not only sufficient but often optimal, yielding models that are simpler, more robust, and more interpretable than their fully nonlinear or linear counterparts. Our results provide a principled framework for selectively introducing nonlinearity, bridging dynamical systems theory with the functional demands of long-range memory and structured computation in recurrent neural networks, with implications for both artificial and biological neural systems. 

**Abstract (ZH)**: 记忆和长程时间处理是跨自然语言处理、时间序列预测、语音识别和控制领域的序列建模任务的核心要求。虽然非线性循环长期以来被认为是实现这些机制的关键，但近期研究表明线性动态往往已足以。在本研究中，我们超越性能比较，系统性地剖析循环网络中非线性功能的作用——确定它何时是计算上必要的，以及它所启用的机制。我们使用几乎线性循环神经网络（AL-RNN），既作为一种灵活的建模工具，也是探究记忆内部机制的探针。在一系列经典的序列建模任务和实际刺激选择任务中，我们发现最少的非线性不仅是足够的，而且往往是最佳的，生成的模型比完全非线性或线性的模型更为简单、稳健且易于解释。我们的结果提供了一个有原则的方法来有选择地引入非线性，将动力系统理论与循环神经网络中长程记忆和结构计算的功能需求联系起来，对人工和生物神经系统的均有启示。 

---
# Diffuse Everything: Multimodal Diffusion Models on Arbitrary State Spaces 

**Title (ZH)**: 弥散一切：任意状态空间上的多模态扩散模型 

**Authors**: Kevin Rojas, Yuchen Zhu, Sichen Zhu, Felix X.-F. Ye, Molei Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.07903)  

**Abstract**: Diffusion models have demonstrated remarkable performance in generating unimodal data across various tasks, including image, video, and text generation. On the contrary, the joint generation of multimodal data through diffusion models is still in the early stages of exploration. Existing approaches heavily rely on external preprocessing protocols, such as tokenizers and variational autoencoders, to harmonize varied data representations into a unified, unimodal format. This process heavily demands the high accuracy of encoders and decoders, which can be problematic for applications with limited data. To lift this restriction, we propose a novel framework for building multimodal diffusion models on arbitrary state spaces, enabling native generation of coupled data across different modalities. By introducing an innovative decoupled noise schedule for each modality, we enable both unconditional and modality-conditioned generation within a single model simultaneously. We empirically validate our approach for text-image generation and mixed-type tabular data synthesis, demonstrating that it achieves competitive performance. 

**Abstract (ZH)**: 多模式扩散模型在任意状态空间的构建：一种新颖的框架及其在文本图像生成和混合类型表数据合成中的应用 

---
# MiniCPM4: Ultra-Efficient LLMs on End Devices 

**Title (ZH)**: MiniCPM4: 末梢设备上的超高效大语言模型 

**Authors**: MiniCPM Team, Chaojun Xiao, Yuxuan Li, Xu Han, Yuzhuo Bai, Jie Cai, Haotian Chen, Wentong Chen, Xin Cong, Ganqu Cui, Ning Ding, Shengdan Fan, Yewei Fang, Zixuan Fu, Wenyu Guan, Yitong Guan, Junshao Guo, Yufeng Han, Bingxiang He, Yuxiang Huang, Cunliang Kong, Qiuzuo Li, Siyuan Li, Wenhao Li, Yanghao Li, Yishan Li, Zhen Li, Dan Liu, Biyuan Lin, Yankai Lin, Xiang Long, Quanyu Lu, Yaxi Lu, Peiyan Luo, Hongya Lyu, Litu Ou, Yinxu Pan, Zekai Qu, Qundong Shi, Zijun Song, Jiayuan Su, Zhou Su, Ao Sun, Xianghui Sun, Peijun Tang, Fangzheng Wang, Feng Wang, Shuo Wang, Yudong Wang, Yesai Wu, Zhenyu Xiao, Jie Xie, Zihao Xie, Yukun Yan, Jiarui Yuan, Kaihuo Zhang, Lei Zhang, Linyue Zhang, Xueren Zhang, Yudi Zhang, Hengyu Zhao, Weilin Zhao, Weilun Zhao, Yuanqian Zhao, Zhi Zheng, Ge Zhou, Jie Zhou, Wei Zhou, Zihan Zhou, Zixuan Zhou, Zhiyuan Liu, Guoyang Zeng, Chao Jia, Dahai Li, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.07900)  

**Abstract**: This paper introduces MiniCPM4, a highly efficient large language model (LLM) designed explicitly for end-side devices. We achieve this efficiency through systematic innovation in four key dimensions: model architecture, training data, training algorithms, and inference systems. Specifically, in terms of model architecture, we propose InfLLM v2, a trainable sparse attention mechanism that accelerates both prefilling and decoding phases for long-context processing. Regarding training data, we propose UltraClean, an efficient and accurate pre-training data filtering and generation strategy, and UltraChat v2, a comprehensive supervised fine-tuning dataset. These datasets enable satisfactory model performance to be achieved using just 8 trillion training tokens. Regarding training algorithms, we propose ModelTunnel v2 for efficient pre-training strategy search, and improve existing post-training methods by introducing chunk-wise rollout for load-balanced reinforcement learning and data-efficient tenary LLM, BitCPM. Regarding inference systems, we propose this http URL that integrates sparse attention, model quantization, and speculative sampling to achieve efficient prefilling and decoding. To meet diverse on-device requirements, MiniCPM4 is available in two versions, with 0.5B and 8B parameters, respectively. Sufficient evaluation results show that MiniCPM4 outperforms open-source models of similar size across multiple benchmarks, highlighting both its efficiency and effectiveness. Notably, MiniCPM4-8B demonstrates significant speed improvements over Qwen3-8B when processing long sequences. Through further adaptation, MiniCPM4 successfully powers diverse applications, including trustworthy survey generation and tool use with model context protocol, clearly showcasing its broad usability. 

**Abstract (ZH)**: MiniCPM4：一种专为端侧设备设计的高效大型语言模型 

---
# GaussianVAE: Adaptive Learning Dynamics of 3D Gaussians for High-Fidelity Super-Resolution 

**Title (ZH)**: 高斯VAE：高保真超分辨的3D高斯自适应学习动力学 

**Authors**: Shuja Khalid, Mohamed Ibrahim, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07897)  

**Abstract**: We present a novel approach for enhancing the resolution and geometric fidelity of 3D Gaussian Splatting (3DGS) beyond native training resolution. Current 3DGS methods are fundamentally limited by their input resolution, producing reconstructions that cannot extrapolate finer details than are present in the training views. Our work breaks this limitation through a lightweight generative model that predicts and refines additional 3D Gaussians where needed most. The key innovation is our Hessian-assisted sampling strategy, which intelligently identifies regions that are likely to benefit from densification, ensuring computational efficiency. Unlike computationally intensive GANs or diffusion approaches, our method operates in real-time (0.015s per inference on a single consumer-grade GPU), making it practical for interactive applications. Comprehensive experiments demonstrate significant improvements in both geometric accuracy and rendering quality compared to state-of-the-art methods, establishing a new paradigm for resolution-free 3D scene enhancement. 

**Abstract (ZH)**: 超越原有训练分辨率的3D高斯散射增强的新方法：一种轻量级生成模型在需要处预测和细化3D高斯散射以提升几何保真度和渲染质量 

---
# Diffusion Counterfactual Generation with Semantic Abduction 

**Title (ZH)**: 语义 abduction 的扩散-counterfactual 生成 

**Authors**: Rajat Rasal, Avinash Kori, Fabio De Sousa Ribeiro, Tian Xia, Ben Glocker  

**Link**: [PDF](https://arxiv.org/pdf/2506.07883)  

**Abstract**: Counterfactual image generation presents significant challenges, including preserving identity, maintaining perceptual quality, and ensuring faithfulness to an underlying causal model. While existing auto-encoding frameworks admit semantic latent spaces which can be manipulated for causal control, they struggle with scalability and fidelity. Advancements in diffusion models present opportunities for improving counterfactual image editing, having demonstrated state-of-the-art visual quality, human-aligned perception and representation learning capabilities. Here, we present a suite of diffusion-based causal mechanisms, introducing the notions of spatial, semantic and dynamic abduction. We propose a general framework that integrates semantic representations into diffusion models through the lens of Pearlian causality to edit images via a counterfactual reasoning process. To our knowledge, this is the first work to consider high-level semantic identity preservation for diffusion counterfactuals and to demonstrate how semantic control enables principled trade-offs between faithful causal control and identity preservation. 

**Abstract (ZH)**: 基于扩散模型的因果机制呈现显著挑战，包括保持身份一致性、维持感知质量以及确保符合潜在因果模型。现有的自编码框架虽然允许可操控的语义潜在空间以实现因果控制，但在可扩展性和保真度方面存在困难。扩散模型的进步为改进反事实图像编辑提供了机会，这些模型已经展示了最先进的视觉质量、人机一致的感知能力和表示学习能力。在此，我们提出了一种基于扩散模型的因果机制套件，引入了空间、语义和动态 abduction 的概念。我们提出了一种通用框架，通过佩利因果观将语义表示整合到扩散模型中，以反事实推理过程编辑图像。据我们所知，这是首次研究用于扩散反事实的高层语义身份保持工作，并展示了语义控制如何实现忠实因果控制和身份保持之间的原则性权衡。 

---
# FreeGave: 3D Physics Learning from Dynamic Videos by Gaussian Velocity 

**Title (ZH)**: FreeGave: 由动态视频中的高斯速度学习3D物理 

**Authors**: Jinxi Li, Ziyang Song, Siyuan Zhou, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07865)  

**Abstract**: In this paper, we aim to model 3D scene geometry, appearance, and the underlying physics purely from multi-view videos. By applying various governing PDEs as PINN losses or incorporating physics simulation into neural networks, existing works often fail to learn complex physical motions at boundaries or require object priors such as masks or types. In this paper, we propose FreeGave to learn the physics of complex dynamic 3D scenes without needing any object priors. The key to our approach is to introduce a physics code followed by a carefully designed divergence-free module for estimating a per-Gaussian velocity field, without relying on the inefficient PINN losses. Extensive experiments on three public datasets and a newly collected challenging real-world dataset demonstrate the superior performance of our method for future frame extrapolation and motion segmentation. Most notably, our investigation into the learned physics codes reveals that they truly learn meaningful 3D physical motion patterns in the absence of any human labels in training. 

**Abstract (ZH)**: 本文旨在仅从多视图视频中建模3D场景的几何结构、外观及其底层物理现象。现有工作往往难以学习复杂边界上的物理运动，或需依赖对象先验如掩码或类型。本文提出FreeGave，无需任何对象先验即可学习复杂动态3D场景的物理现象。我们方法的关键在于引入物理代码，并通过精心设计的无散模块估计每个高斯速度场，而无需依赖效率低下的PINN损失。在三个公开数据集和一个新收集的具有挑战性的现实世界数据集上的 extensive 实验表明，我们的方法在后续帧外推和运动分割方面表现出优越的性能。特别值得注意的是，我们对学习到的物理代码的研究揭示了，在训练中无任何人类标签的情况下，它们确实学习到了有意义的3D物理运动模式。 

---
# Lightweight Sequential Transformers for Blood Glucose Level Prediction in Type-1 Diabetes 

**Title (ZH)**: 适用于1型糖尿病的轻量级顺序变换器血糖水平预测 

**Authors**: Mirko Paolo Barbato, Giorgia Rigamonti, Davide Marelli, Paolo Napoletano  

**Link**: [PDF](https://arxiv.org/pdf/2506.07864)  

**Abstract**: Type 1 Diabetes (T1D) affects millions worldwide, requiring continuous monitoring to prevent severe hypo- and hyperglycemic events. While continuous glucose monitoring has improved blood glucose management, deploying predictive models on wearable devices remains challenging due to computational and memory constraints. To address this, we propose a novel Lightweight Sequential Transformer model designed for blood glucose prediction in T1D. By integrating the strengths of Transformers' attention mechanisms and the sequential processing of recurrent neural networks, our architecture captures long-term dependencies while maintaining computational efficiency. The model is optimized for deployment on resource-constrained edge devices and incorporates a balanced loss function to handle the inherent data imbalance in hypo- and hyperglycemic events. Experiments on two benchmark datasets, OhioT1DM and DiaTrend, demonstrate that the proposed model outperforms state-of-the-art methods in predicting glucose levels and detecting adverse events. This work fills the gap between high-performance modeling and practical deployment, providing a reliable and efficient T1D management solution. 

**Abstract (ZH)**: Type 1糖尿病(T1D)影响着全球数百万人，需要持续监测以预防严重的低血糖和高血糖事件。尽管连续葡萄糖监测已经改善了血糖管理，但在可穿戴设备上部署预测模型仍面临计算和内存限制的挑战。为了解决这一问题，我们提出了一种新型轻量级序列变换器模型，适用于T1D的血糖预测。通过结合变换器注意力机制和递归神经网络的序列处理优势，该架构捕捉长时依赖关系同时保持计算效率。该模型针对资源受限的边缘设备进行了优化，并融入了平衡的损失函数以处理低血糖和高血糖事件之间固有的数据不平衡问题。在两个基准数据集OhioT1DM和DiaTrend上的实验表明，所提出的方法在预测血糖水平和检测不良事件方面优于现有最先进的方法。这项工作填补了高性能建模与实际部署之间的差距，提供了可靠和高效的T1D管理解决方案。 

---
# Fairness Overfitting in Machine Learning: An Information-Theoretic Perspective 

**Title (ZH)**: 机器学习中的公平过拟合：一种信息论视角 

**Authors**: Firas Laakom, Haobo Chen, Jürgen Schmidhuber, Yuheng Bu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07861)  

**Abstract**: Despite substantial progress in promoting fairness in high-stake applications using machine learning models, existing methods often modify the training process, such as through regularizers or other interventions, but lack formal guarantees that fairness achieved during training will generalize to unseen data. Although overfitting with respect to prediction performance has been extensively studied, overfitting in terms of fairness loss has received far less attention. This paper proposes a theoretical framework for analyzing fairness generalization error through an information-theoretic lens. Our novel bounding technique is based on Efron-Stein inequality, which allows us to derive tight information-theoretic fairness generalization bounds with both Mutual Information (MI) and Conditional Mutual Information (CMI). Our empirical results validate the tightness and practical relevance of these bounds across diverse fairness-aware learning algorithms. Our framework offers valuable insights to guide the design of algorithms improving fairness generalization. 

**Abstract (ZH)**: 尽管在促进高影响应用中机器学习模型的公平性方面取得了显著进展，现有方法往往通过正则化或其他干预措施修改训练过程，但在训练过程中实现的公平性缺乏正式保证，能够推广到未见过的数据。虽然对预测性能过拟合的研究甚多，但关于公平性损失过拟合的研究则较少。本文通过信息论视角提出了一种分析公平性泛化误差的理论框架。我们新颖的边界技术基于Efron-Stein不等式，使得我们能够使用互信息（MI）和条件互信息（CMI）推导出紧致的信息论公平性泛化边界。我们的实证结果验证了这些边界的紧致性和实际相关性，跨越了多种公平性感知学习算法。该框架为指导改进公平性泛化的算法设计提供了宝贵见解。 

---
# LogoSP: Local-global Grouping of Superpoints for Unsupervised Semantic Segmentation of 3D Point Clouds 

**Title (ZH)**: LogoSP：超点局部-全局分组用于三维点云无监督语义分割 

**Authors**: Zihui Zhang, Weisheng Dai, Hongtao Wen, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07857)  

**Abstract**: We study the problem of unsupervised 3D semantic segmentation on raw point clouds without needing human labels in training. Existing methods usually formulate this problem into learning per-point local features followed by a simple grouping strategy, lacking the ability to discover additional and possibly richer semantic priors beyond local features. In this paper, we introduce LogoSP to learn 3D semantics from both local and global point features. The key to our approach is to discover 3D semantic information by grouping superpoints according to their global patterns in the frequency domain, thus generating highly accurate semantic pseudo-labels for training a segmentation network. Extensive experiments on two indoor and an outdoor datasets show that our LogoSP surpasses all existing unsupervised methods by large margins, achieving the state-of-the-art performance for unsupervised 3D semantic segmentation. Notably, our investigation into the learned global patterns reveals that they truly represent meaningful 3D semantics in the absence of human labels during training. 

**Abstract (ZH)**: 我们研究在无需人类标签的情况下对原始点云进行无监督三维语义分割的问题。现有方法通常将此问题形式化为学习点的局部特征，随后采用简单的聚类策略，缺乏发现超出局部特征的附加和可能更丰富语义先验的能力。在本文中，我们引入了LogoSP，以结合局部和全局点特征学习三维语义。我们方法的关键在于根据全局频域模式对超级点进行分组，从而生成用于训练分割网络的高精度语义伪标签。在两个室内和一个室外数据集上的广泛实验表明，我们的LogoSP大幅超越了所有现有的无监督方法，达到了无监督三维语义分割的最先进性能。值得注意的是，我们对学习到的全局模式的探究表明，在训练过程中缺乏人类标签时，这些全局模式确实代表了有意义的三维语义。 

---
# Residual Reweighted Conformal Prediction for Graph Neural Networks 

**Title (ZH)**: 图神经网络中的残差加权置信预测 

**Authors**: Zheng Zhang, Jie Bao, Zhixin Zhou, Nicolo Colombo, Lixin Cheng, Rui Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.07854)  

**Abstract**: Graph Neural Networks (GNNs) excel at modeling relational data but face significant challenges in high-stakes domains due to unquantified uncertainty. Conformal prediction (CP) offers statistical coverage guarantees, but existing methods often produce overly conservative prediction intervals that fail to account for graph heteroscedasticity and structural biases. While residual reweighting CP variants address some of these limitations, they neglect graph topology, cluster-specific uncertainties, and risk data leakage by reusing training sets. To address these issues, we propose Residual Reweighted GNN (RR-GNN), a framework designed to generate minimal prediction sets with provable marginal coverage guarantees.
RR-GNN introduces three major innovations to enhance prediction performance. First, it employs Graph-Structured Mondrian CP to partition nodes or edges into communities based on topological features, ensuring cluster-conditional coverage that reflects heterogeneity. Second, it uses Residual-Adaptive Nonconformity Scores by training a secondary GNN on a held-out calibration set to estimate task-specific residuals, dynamically adjusting prediction intervals according to node or edge uncertainty. Third, it adopts a Cross-Training Protocol, which alternates the optimization of the primary GNN and the residual predictor to prevent information leakage while maintaining graph dependencies. We validate RR-GNN on 15 real-world graphs across diverse tasks, including node classification, regression, and edge weight prediction. Compared to CP baselines, RR-GNN achieves improved efficiency over state-of-the-art methods, with no loss of coverage. 

**Abstract (ZH)**: 基于残差重权的图神经网络（RR-GNN）：生成具有可证明边缘覆盖保证的最小预测集 

---
# PolyVivid: Vivid Multi-Subject Video Generation with Cross-Modal Interaction and Enhancement 

**Title (ZH)**: PolyVivid：跨模态交互与增强的多主题视频生成 

**Authors**: Teng Hu, Zhentao Yu, Zhengguang Zhou, Jiangning Zhang, Yuan Zhou, Qinglin Lu, Ran Yi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07848)  

**Abstract**: Despite recent advances in video generation, existing models still lack fine-grained controllability, especially for multi-subject customization with consistent identity and interaction. In this paper, we propose PolyVivid, a multi-subject video customization framework that enables flexible and identity-consistent generation. To establish accurate correspondences between subject images and textual entities, we design a VLLM-based text-image fusion module that embeds visual identities into the textual space for precise grounding. To further enhance identity preservation and subject interaction, we propose a 3D-RoPE-based enhancement module that enables structured bidirectional fusion between text and image embeddings. Moreover, we develop an attention-inherited identity injection module to effectively inject fused identity features into the video generation process, mitigating identity drift. Finally, we construct an MLLM-based data pipeline that combines MLLM-based grounding, segmentation, and a clique-based subject consolidation strategy to produce high-quality multi-subject data, effectively enhancing subject distinction and reducing ambiguity in downstream video generation. Extensive experiments demonstrate that PolyVivid achieves superior performance in identity fidelity, video realism, and subject alignment, outperforming existing open-source and commercial baselines. 

**Abstract (ZH)**: 尽管最近在视频生成方面取得了进展，现有模型仍缺乏细粒度可控性，特别是在具有一致身份和交互的多主体定制方面。本文提出PolyVivid，一种多主体视频定制框架，实现灵活且一致的身份生成。为了在主体图像与文本实体之间建立准确的对应关系，我们设计了一个基于VLLM的文本-图像融合模块，将在视觉空间中嵌入的身份特征嵌入到文本空间中，实现精确的语义关联。为了进一步增强身份保留和主体交互，我们提出了一种基于3D-RoPE的增强模块，实现了文本和图像嵌入之间结构化的双向融合。此外，我们开发了一种注意机制继承的身份注入模块，有效地将融合的身份特征注入到视频生成过程中，缓解身份漂移。最后，我们构建了一个基于MLLM的数据管道，结合了基于MLLM的语义关联、分割和基于团簇的主体聚合策略，生成高质量的多主体数据，有效增强主体区分并减少下游视频生成中的模糊性。大量实验表明，PolyVivid在身份保真度、视频现实性和主体对齐方面表现出 superior 的性能，超越了现有的开源和商用基线。 

---
# Diffusion models under low-noise regime 

**Title (ZH)**: 低噪声条件下扩散模型 

**Authors**: Elizabeth Pavlova, Xue-Xin Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.07841)  

**Abstract**: Recent work on diffusion models proposed that they operate in two regimes: memorization, in which models reproduce their training data, and generalization, in which they generate novel samples. While this has been tested in high-noise settings, the behavior of diffusion models as effective denoisers when the corruption level is small remains unclear. To address this gap, we systematically investigated the behavior of diffusion models under low-noise diffusion dynamics, with implications for model robustness and interpretability. Using (i) CelebA subsets of varying sample sizes and (ii) analytic Gaussian mixture benchmarks, we reveal that models trained on disjoint data diverge near the data manifold even when their high-noise outputs converge. We quantify how training set size, data geometry, and model objective choice shape denoising trajectories and affect score accuracy, providing insights into how these models actually learn representations of data distributions. This work starts to address gaps in our understanding of generative model reliability in practical applications where small perturbations are common. 

**Abstract (ZH)**: 近期关于扩散模型的研究提出，它们在两种模式下运行：记忆模式，模型重现训练数据；泛化模式，生成新颖样本。尽管已经在高噪声条件下进行了测试，但在低噪声水平下扩散模型作为有效去噪器的行为尚不清楚。为了填补这一空白，我们系统地研究了在低噪声扩散动力学下扩散模型的行为，这对模型的鲁棒性和可解释性具有重要意义。通过使用(i) 不同样本大小的CelebA子集和(ii) 分析性的高斯混合基准，我们揭示了在训练数据不交集中训练的模型即使在高噪声输出收敛时，仍然会在数据流形附近发散。我们量化了训练集大小、数据几何形状和模型目标选择如何影响去噪轨迹和得分准确性，提供了这些模型是如何实际上学习数据分布表示的见解。这项工作开始弥补了我们对生成模型在实际应用中鲁棒性理解上的空白，特别是在小扰动常见的情况下。 

---
# Are Trees Really Green? A Detection Approach of IoT Malware Attacks 

**Title (ZH)**: 物联网 malware 攻击检测方法：树木really绿色吗？ 

**Authors**: Silvia Lucia Sanna, Diego Soi, Davide Maiorca, Giorgio Giacinto  

**Link**: [PDF](https://arxiv.org/pdf/2506.07836)  

**Abstract**: Nowadays, the Internet of Things (IoT) is widely employed, and its usage is growing exponentially because it facilitates remote monitoring, predictive maintenance, and data-driven decision making, especially in the healthcare and industrial sectors. However, IoT devices remain vulnerable due to their resource constraints and difficulty in applying security patches. Consequently, various cybersecurity attacks are reported daily, such as Denial of Service, particularly in IoT-driven solutions. Most attack detection methodologies are based on Machine Learning (ML) techniques, which can detect attack patterns. However, the focus is more on identification rather than considering the impact of ML algorithms on computational resources. This paper proposes a green methodology to identify IoT malware networking attacks based on flow privacy-preserving statistical features. In particular, the hyperparameters of three tree-based models -- Decision Trees, Random Forest and Extra-Trees -- are optimized based on energy consumption and test-time performance in terms of Matthew's Correlation Coefficient. Our results show that models maintain high performance and detection accuracy while consistently reducing power usage in terms of watt-hours (Wh). This suggests that on-premise ML-based Intrusion Detection Systems are suitable for IoT and other resource-constrained devices. 

**Abstract (ZH)**: 物联网设备流量隐私保护统计特征下的绿色恶意软件网络攻击检测方法 

---
# Improving large language models with concept-aware fine-tuning 

**Title (ZH)**: 概念意识微调改进大型语言模型 

**Authors**: Michael K. Chen, Xikun Zhang, Jiaxing Huang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.07833)  

**Abstract**: Large language models (LLMs) have become the cornerstone of modern AI. However, the existing paradigm of next-token prediction fundamentally limits their ability to form coherent, high-level concepts, making it a critical barrier to human-like understanding and reasoning. Take the phrase "ribonucleic acid" as an example: an LLM will first decompose it into tokens, i.e., artificial text fragments ("rib", "on", ...), then learn each token sequentially, rather than grasping the phrase as a unified, coherent semantic entity. This fragmented representation hinders deeper conceptual understanding and, ultimately, the development of truly intelligent systems. In response, we introduce Concept-Aware Fine-Tuning (CAFT), a novel multi-token training method that redefines how LLMs are fine-tuned. By enabling the learning of sequences that span multiple tokens, this method fosters stronger concept-aware learning. Our experiments demonstrate significant improvements compared to conventional next-token finetuning methods across diverse tasks, including traditional applications like text summarization and domain-specific ones like de novo protein design. Multi-token prediction was previously only possible in the prohibitively expensive pretraining phase; CAFT, to our knowledge, is the first to bring the multi-token setting to the post-training phase, thus effectively democratizing its benefits for the broader community of practitioners and researchers. Finally, the unexpected effectiveness of our proposed method suggests wider implications for the machine learning research community. All code and data are available at this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）已成为现代人工智能的基石。然而，现有的下一token预测范式从根本上限制了它们形成连贯且高层次概念的能力，成为实现类人理解与推理的关键障碍。例如，对于短语“ribonucleic acid”，LLM会首先将其分解为token，即人工文本片段（“rib”，“on”，...），然后顺序学习每个token，而不是将整个短语视为一个统一且连贯的语义实体。这种分割表示阻碍了更深层次的概念理解，并最终阻碍了真正智能系统的开发。为此，我们提出了一种名为概念意识微调（CAFT）的新颖多token训练方法，重新定义了LLM的微调方式。通过使模型能够学习跨越多个token的序列，这种方法促进了更强的概念意识学习。我们的实验表明，与传统的下一token微调方法相比，在多种任务上取得了显著的改进，包括传统的文本摘要等应用以及特定领域的蛋白质从头设计等应用。多token预测在过去只能在成本高昂的预训练阶段实现；据我们所知，CAFT是首个将多token设置引入后训练阶段的方法，从而有效普及了其益处，惠及更广泛的实践者和研究人员。最后，我们提出的方法出乎意料的有效性暗示了对机器学习研究社区更广泛影响。所有代码和数据均可在以下链接获得。 

---
# Decentralizing Multi-Agent Reinforcement Learning with Temporal Causal Information 

**Title (ZH)**: 基于时间因果信息的多智能体强化学习去中心化方法 

**Authors**: Jan Corazza, Hadi Partovi Aria, Hyohun Kim, Daniel Neider, Zhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07829)  

**Abstract**: Reinforcement learning (RL) algorithms can find an optimal policy for a single agent to accomplish a particular task. However, many real-world problems require multiple agents to collaborate in order to achieve a common goal. For example, a robot executing a task in a warehouse may require the assistance of a drone to retrieve items from high shelves. In Decentralized Multi-Agent RL (DMARL), agents learn independently and then combine their policies at execution time, but often must satisfy constraints on compatibility of local policies to ensure that they can achieve the global task when combined. In this paper, we study how providing high-level symbolic knowledge to agents can help address unique challenges of this setting, such as privacy constraints, communication limitations, and performance concerns. In particular, we extend the formal tools used to check the compatibility of local policies with the team task, making decentralized training with theoretical guarantees usable in more scenarios. Furthermore, we empirically demonstrate that symbolic knowledge about the temporal evolution of events in the environment can significantly expedite the learning process in DMARL. 

**Abstract (ZH)**: 利用高层符号知识解决分布式多智能体强化学习的独特挑战 

---
# Accelerating Diffusion Models in Offline RL via Reward-Aware Consistency Trajectory Distillation 

**Title (ZH)**: 在离线RL中通过奖励意识一致性轨迹蒸馏加速扩散模型 

**Authors**: Xintong Duan, Yutong He, Fahim Tajwar, Ruslan Salakhutdinov, J. Zico Kolter, Jeff Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2506.07822)  

**Abstract**: Although diffusion models have achieved strong results in decision-making tasks, their slow inference speed remains a key limitation. While the consistency model offers a potential solution, its applications to decision-making often struggle with suboptimal demonstrations or rely on complex concurrent training of multiple networks. In this work, we propose a novel approach to consistency distillation for offline reinforcement learning that directly incorporates reward optimization into the distillation process. Our method enables single-step generation while maintaining higher performance and simpler training. Empirical evaluations on the Gym MuJoCo benchmarks and long horizon planning demonstrate that our approach can achieve an 8.7% improvement over previous state-of-the-art while offering up to 142x speedup over diffusion counterparts in inference time. 

**Abstract (ZH)**: 尽管扩散模型在决策任务中取得了强大的成果，但其缓慢的推理速度仍然是一个关键限制。虽然一致性模型提供了一种潜在的解决方案，但在决策任务中的应用往往面临亚优演示或依赖多个网络的复杂并发训练的问题。在本文中，我们提出了一种新的离线强化学习中一致性的蒸馏方法，该方法直接将奖励优化纳入蒸馏过程。该方法能够在保持更高性能和更简单训练的同时实现单步生成。在Gym MuJoCo基准测试和长时规划上的实证评估表明，我们的方法在推理时间上比扩散模型 counterparts 提供高达142倍的速度提升，同时相比之前最先进的方法可实现8.7%的性能提升。 

---
# Self-Cascaded Diffusion Models for Arbitrary-Scale Image Super-Resolution 

**Title (ZH)**: 自 cascading 扩散模型在任意尺度图像超分辨中的应用 

**Authors**: Junseo Bang, Joonhee Lee, Kyeonghyun Lee, Haechang Lee, Dong Un Kang, Se Young Chun  

**Link**: [PDF](https://arxiv.org/pdf/2506.07813)  

**Abstract**: Arbitrary-scale image super-resolution aims to upsample images to any desired resolution, offering greater flexibility than traditional fixed-scale super-resolution. Recent approaches in this domain utilize regression-based or generative models, but many of them are a single-stage upsampling process, which may be challenging to learn across a wide, continuous distribution of scaling factors. Progressive upsampling strategies have shown promise in mitigating this issue, yet their integration with diffusion models for flexible upscaling remains underexplored. Here, we present CasArbi, a novel self-cascaded diffusion framework for arbitrary-scale image super-resolution. CasArbi meets the varying scaling demands by breaking them down into smaller sequential factors and progressively enhancing the image resolution at each step with seamless transitions for arbitrary scales. Our novel coordinate-guided residual diffusion model allows for the learning of continuous image representations while enabling efficient diffusion sampling. Extensive experiments demonstrate that our CasArbi outperforms prior arts in both perceptual and distortion performance metrics across diverse arbitrary-scale super-resolution benchmarks. 

**Abstract (ZH)**: 任意尺度图像超分辨率旨在将图像放大到任何所需分辨率，提供了传统固定尺度超分辨率更大的灵活性。该领域近期的方法采用了回归或生成模型，但许多方法是一阶段放大过程，可能难以学习广泛的连续缩放因子分布。逐步放大策略在缓解这一问题方面显示出潜力，但其与扩散模型结合以实现灵活放大仍很少被探索。为此，我们提出了一种名为CasArbi的新型自级联扩散框架，用于任意尺度图像超分辨率。CasArbi通过将不同缩放需求分解为较小的序列因子，并在每一级逐步提升图像分辨率，实现了任意尺度的无缝过渡。我们新的坐标引导残差扩散模型能够在学习连续图像表示的同时，实现高效的扩散采样。广泛实验证明，我们的CasArbi在多种任意尺度超分辨率基准测试中，在感知和失真性能指标上均优于现有方法。 

---
# Enhancing Adversarial Robustness with Conformal Prediction: A Framework for Guaranteed Model Reliability 

**Title (ZH)**: 基于区间预测的 adversarial  robustness 提升框架：一种保障模型可靠性的方法 

**Authors**: Jie Bao, Chuangyin Dang, Rui Luo, Hanwei Zhang, Zhixin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.07804)  

**Abstract**: As deep learning models are increasingly deployed in high-risk applications, robust defenses against adversarial attacks and reliable performance guarantees become paramount. Moreover, accuracy alone does not provide sufficient assurance or reliable uncertainty estimates for these models. This study advances adversarial training by leveraging principles from Conformal Prediction. Specifically, we develop an adversarial attack method, termed OPSA (OPtimal Size Attack), designed to reduce the efficiency of conformal prediction at any significance level by maximizing model uncertainty without requiring coverage guarantees. Correspondingly, we introduce OPSA-AT (Adversarial Training), a defense strategy that integrates OPSA within a novel conformal training paradigm. Experimental evaluations demonstrate that our OPSA attack method induces greater uncertainty compared to baseline approaches for various defenses. Conversely, our OPSA-AT defensive model significantly enhances robustness not only against OPSA but also other adversarial attacks, and maintains reliable prediction. Our findings highlight the effectiveness of this integrated approach for developing trustworthy and resilient deep learning models for safety-critical domains. Our code is available at this https URL. 

**Abstract (ZH)**: 随着深度学习模型在高风险应用中的日益部署，对抗攻击的稳健防御和可靠性能保证变得至关重要。此外，单纯依赖准确性不足以提供这些模型的充分保证或可靠的不确定性估计。本研究通过结合校准预测原则推进了对抗训练。具体而言，我们开发了一种称为OPSA（OPtimal Size Attack）的对抗攻击方法，旨在通过最大化模型不确定性来降低任何显著水平下的校准预测效率，而无需提供覆盖保证。相应地，我们引入了OPSA-AT（对抗训练）防御策略，该策略将OPSA集成到一个新颖的校准训练框架中。实验评估表明，与基线方法相比，我们的OPSA攻击方法在各种防御措施下引发了更高的不确定性。相反，我们的OPSA-AT防御模型不仅显著增强了对抗OPSA以及其他对抗攻击的鲁棒性，还保持了可靠的预测性能。我们的研究结果突显了这种集成方法在开发适用于关键安全领域的可信赖且稳健的深度学习模型方面的有效性。我们的代码可在以下链接获取。 

---
# MultiMatch: Multihead Consistency Regularization Matching for Semi-Supervised Text Classification 

**Title (ZH)**: MultiMatch: 多头一致性正则化匹配在半监督文本分类中的应用 

**Authors**: Iustin Sirbu, Robert-Adrian Popovici, Cornelia Caragea, Stefan Trausan-Matu, Traian Rebedea  

**Link**: [PDF](https://arxiv.org/pdf/2506.07801)  

**Abstract**: We introduce MultiMatch, a novel semi-supervised learning (SSL) algorithm combining the paradigms of co-training and consistency regularization with pseudo-labeling. At its core, MultiMatch features a three-fold pseudo-label weighting module designed for three key purposes: selecting and filtering pseudo-labels based on head agreement and model confidence, and weighting them according to the perceived classification difficulty. This novel module enhances and unifies three existing techniques -- heads agreement from Multihead Co-training, self-adaptive thresholds from FreeMatch, and Average Pseudo-Margins from MarginMatch -- resulting in a holistic approach that improves robustness and performance in SSL settings. Experimental results on benchmark datasets highlight the superior performance of MultiMatch, achieving state-of-the-art results on 9 out of 10 setups from 5 natural language processing datasets and ranking first according to the Friedman test among 19 methods. Furthermore, MultiMatch demonstrates exceptional robustness in highly imbalanced settings, outperforming the second-best approach by 3.26% -- and data imbalance is a key factor for many text classification tasks. 

**Abstract (ZH)**: MultiMatch: 结合共训练和一致性正则化的新型半监督学习算法 

---
# Re-ranking Reasoning Context with Tree Search Makes Large Vision-Language Models Stronger 

**Title (ZH)**: 基于树搜索重排推理上下文使大型视觉-语言模型更强大 

**Authors**: Qi Yang, Chenghao Zhang, Lubin Fan, Kun Ding, Jieping Ye, Shiming Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07785)  

**Abstract**: Recent advancements in Large Vision Language Models (LVLMs) have significantly improved performance in Visual Question Answering (VQA) tasks through multimodal Retrieval-Augmented Generation (RAG). However, existing methods still face challenges, such as the scarcity of knowledge with reasoning examples and erratic responses from retrieved knowledge. To address these issues, in this study, we propose a multimodal RAG framework, termed RCTS, which enhances LVLMs by constructing a Reasoning Context-enriched knowledge base and a Tree Search re-ranking method. Specifically, we introduce a self-consistent evaluation mechanism to enrich the knowledge base with intrinsic reasoning patterns. We further propose a Monte Carlo Tree Search with Heuristic Rewards (MCTS-HR) to prioritize the most relevant examples. This ensures that LVLMs can leverage high-quality contextual reasoning for better and more consistent responses. Extensive experiments demonstrate that our framework achieves state-of-the-art performance on multiple VQA datasets, significantly outperforming In-Context Learning (ICL) and Vanilla-RAG methods. It highlights the effectiveness of our knowledge base and re-ranking method in improving LVLMs. Our code is available at this https URL. 

**Abstract (ZH)**: 近期大型多模态视觉语言模型（LVLMs）在视觉问答（VQA）任务中的显著进步是通过多模态检索增强生成（RAG）实现的。然而，现有方法仍然面临挑战，如缺乏具有推理示例的知识和检索知识的不稳定响应。为解决这些问题，本研究提出了一种名为RCTS的多模态RAG框架，通过构建增强推理背景知识库和树搜索再排序方法来增强LVLMs。具体而言，我们引入了一种自我一致性评估机制来丰富知识库中的内在推理模式。此外，我们提出了带有启发式奖励的蒙特卡洛树搜索（MCTS-HR）以优先处理最相关示例。这确保了LVLMs能够利用高质量的上下文推理以获得更好且更一致的响应。广泛实验表明，我们的框架在多个VQA数据集上达到了最先进的性能，显著优于上下文学习（ICL）和纯RAG方法。这突显了我们知识库和再排序方法在提高LVLMs效果方面的有效性。我们的代码可在以下链接获取：this https URL。 

---
# Comparing Credit Risk Estimates in the Gen-AI Era 

**Title (ZH)**: Gen-AI时代下的信用风险估计比较 

**Authors**: Nicola Lavecchia, Sid Fadanelli, Federico Ricciuti, Gennaro Aloe, Enrico Bagli, Pietro Giuffrida, Daniele Vergari  

**Link**: [PDF](https://arxiv.org/pdf/2506.07754)  

**Abstract**: Generative AI technologies have demonstrated significant potential across diverse applications. This study provides a comparative analysis of credit score modeling techniques, contrasting traditional approaches with those leveraging generative AI. Our findings reveal that current generative AI models fall short of matching the performance of traditional methods, regardless of the integration strategy employed. These results highlight the limitations in the current capabilities of generative AI for credit risk scoring, emphasizing the need for further research and development before the possibility of applying generative AI for this specific task, or equivalent ones. 

**Abstract (ZH)**: 生成式AI技术在多种应用中展示了显著潜力。本研究对比分析了信用评分建模技术，将传统方法与利用生成式AI的方法进行了对比。研究发现，当前的生成式AI模型在性能上仍不及传统方法，无论采用何种集成策略。这些结果突显了当前生成式AI在信用风险评分能力上的局限性，强调在将其应用于此类特定任务之前需要进一步的研究与开发。 

---
# Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking 

**Title (ZH)**: 通过强化抽象思维增强LLMs的推理能力 

**Authors**: Silin Gao, Antoine Bosselut, Samy Bengio, Emmanuel Abbe  

**Link**: [PDF](https://arxiv.org/pdf/2506.07751)  

**Abstract**: Recent studies have shown that large language models (LLMs), especially smaller ones, often lack robustness in their reasoning. I.e., they tend to experience performance drops when faced with distribution shifts, such as changes to numerical or nominal variables, or insertions of distracting clauses. A possible strategy to address this involves generating synthetic data to further "instantiate" reasoning problems on potential variations. In contrast, our approach focuses on "abstracting" reasoning problems. This not only helps counteract distribution shifts but also facilitates the connection to symbolic tools for deriving solutions. We find that this abstraction process is better acquired through reinforcement learning (RL) than just supervised fine-tuning, which often fails to produce faithful abstractions. Our method, AbstraL -- which promotes abstract reasoning in LLMs using RL on granular abstraction data -- significantly mitigates performance degradation on recent GSM perturbation benchmarks. 

**Abstract (ZH)**: 近期研究表明，大型语言模型（LLMs），尤其是较小的模型，经常在其推理过程中缺乏稳健性。即，当面对分布偏移时（如数值或名义变量的变化，或插入干扰性从句），它们往往会表现出性能下降。一种可能的策略是生成合成数据以进一步“实例化”潜在变化的推理问题。相比之下，我们 approaches 的重点在于“抽象化”推理问题。这不仅有助于抵消分布偏移，还促进了与符号工具连接以推导解决方案。我们发现，通过强化学习（RL）而非仅仅监督微调来获得抽象过程更为有效，后者往往无法产生忠实的抽象。我们的方法 AbstraL —— 该方法使用 RL 在粒度抽象数据上促进 LLMs 的抽象推理 —— 显著减轻了近期 GSM 干扰基准上的性能下降。 

---
# Graph-Assisted Stitching for Offline Hierarchical Reinforcement Learning 

**Title (ZH)**: 基于图辅助缝合的离线分层强化学习 

**Authors**: Seungho Baek, Taegeon Park, Jongchan Park, Seungjun Oh, Yusung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.07744)  

**Abstract**: Existing offline hierarchical reinforcement learning methods rely on high-level policy learning to generate subgoal sequences. However, their efficiency degrades as task horizons increase, and they lack effective strategies for stitching useful state transitions across different trajectories. We propose Graph-Assisted Stitching (GAS), a novel framework that formulates subgoal selection as a graph search problem rather than learning an explicit high-level policy. By embedding states into a Temporal Distance Representation (TDR) space, GAS clusters semantically similar states from different trajectories into unified graph nodes, enabling efficient transition stitching. A shortest-path algorithm is then applied to select subgoal sequences within the graph, while a low-level policy learns to reach the subgoals. To improve graph quality, we introduce the Temporal Efficiency (TE) metric, which filters out noisy or inefficient transition states, significantly enhancing task performance. GAS outperforms prior offline HRL methods across locomotion, navigation, and manipulation tasks. Notably, in the most stitching-critical task, it achieves a score of 88.3, dramatically surpassing the previous state-of-the-art score of 1.0. Our source code is available at: this https URL. 

**Abstract (ZH)**: 基于图辅助拼接的离线层次 reinforcement 学习方法 

---
# ArchiLense: A Framework for Quantitative Analysis of Architectural Styles Based on Vision Large Language Models 

**Title (ZH)**: Archilense：一种基于视觉大规模语言模型的建筑风格定量分析框架 

**Authors**: Jing Zhong, Jun Yin, Peilin Li, Pengyu Zeng, Miao Zhang, Shuai Lu, Ran Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.07739)  

**Abstract**: Architectural cultures across regions are characterized by stylistic diversity, shaped by historical, social, and technological contexts in addition to geograph-ical conditions. Understanding architectural styles requires the ability to describe and analyze the stylistic features of different architects from various regions through visual observations of architectural imagery. However, traditional studies of architectural culture have largely relied on subjective expert interpretations and historical literature reviews, often suffering from regional biases and limited ex-planatory scope. To address these challenges, this study proposes three core contributions: (1) We construct a professional architectural style dataset named ArchDiffBench, which comprises 1,765 high-quality architectural images and their corresponding style annotations, collected from different regions and historical periods. (2) We propose ArchiLense, an analytical framework grounded in Vision-Language Models and constructed using the ArchDiffBench dataset. By integrating ad-vanced computer vision techniques, deep learning, and machine learning algo-rithms, ArchiLense enables automatic recognition, comparison, and precise classi-fication of architectural imagery, producing descriptive language outputs that ar-ticulate stylistic differences. (3) Extensive evaluations show that ArchiLense achieves strong performance in architectural style recognition, with a 92.4% con-sistency rate with expert annotations and 84.5% classification accuracy, effec-tively capturing stylistic distinctions across images. The proposed approach transcends the subjectivity inherent in traditional analyses and offers a more objective and accurate perspective for comparative studies of architectural culture. 

**Abstract (ZH)**: 地区之间的建筑文化表现为风格多样性，这些风格由历史、社会和技术背景以及地理条件共同塑造。理解建筑风格需要通过建筑图像的视觉观察来描述和分析不同地区建筑师的不同风格特征。然而，传统的建筑文化研究大多依赖于主观专家解释和历史文献回顾，往往存在地域偏见和解释范围有限的问题。为了应对这些挑战，本研究提出了三个核心贡献：（1）我们构建了一个名为ArchDiffBench的专业建筑风格数据集，包含来自不同地区和历史时期的1,765张高质量的建筑图像及其相应的风格注解。（2）我们提出了一种基于视觉-语言模型的分析框架ArchLense，并基于ArchDiffBench数据集构建。通过集成先进的计算机视觉技术、深度学习和机器学习算法，ArchLense能够自动识别、比较和精确分类建筑图像，并生成描述风格差异的语言输出。（3）广泛的评估表明，ArchLense在建筑风格识别方面表现出色，与专家注解的92.4%一致性和84.5%分类准确率，有效地捕捉了图像之间的风格差异。所提出的方法超越了传统分析中的主观性，为建筑文化的比较研究提供了更为客观和准确的视角。 

---
# ETA: Efficiency through Thinking Ahead, A Dual Approach to Self-Driving with Large Models 

**Title (ZH)**: ETA: 通过前瞻思维提升效率，一种基于大型模型的自动驾驶双重视角方法 

**Authors**: Shadi Hamdan, Chonghao Sima, Zetong Yang, Hongyang Li, Fatma Güney  

**Link**: [PDF](https://arxiv.org/pdf/2506.07725)  

**Abstract**: How can we benefit from large models without sacrificing inference speed, a common dilemma in self-driving systems? A prevalent solution is a dual-system architecture, employing a small model for rapid, reactive decisions and a larger model for slower but more informative analyses. Existing dual-system designs often implement parallel architectures where inference is either directly conducted using the large model at each current frame or retrieved from previously stored inference results. However, these works still struggle to enable large models for a timely response to every online frame. Our key insight is to shift intensive computations of the current frame to previous time steps and perform a batch inference of multiple time steps to make large models respond promptly to each time step. To achieve the shifting, we introduce Efficiency through Thinking Ahead (ETA), an asynchronous system designed to: (1) propagate informative features from the past to the current frame using future predictions from the large model, (2) extract current frame features using a small model for real-time responsiveness, and (3) integrate these dual features via an action mask mechanism that emphasizes action-critical image regions. Evaluated on the Bench2Drive CARLA Leaderboard-v2 benchmark, ETA advances state-of-the-art performance by 8% with a driving score of 69.53 while maintaining a near-real-time inference speed at 50 ms. 

**Abstract (ZH)**: 如何在不牺牲推理速度的情况下受益于大规模模型：一种用于自动驾驶系统的前瞻高效双系统架构 

---
# Consistent Video Editing as Flow-Driven Image-to-Video Generation 

**Title (ZH)**: 一致的视频编辑作为流驱动的图像到视频生成 

**Authors**: Ge Wang, Songlin Fan, Hangxu Liu, Quanjian Song, Hewei Wang, Jinfeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07713)  

**Abstract**: With the prosper of video diffusion models, down-stream applications like video editing have been significantly promoted without consuming much computational cost. One particular challenge in this task lies at the motion transfer process from the source video to the edited one, where it requires the consideration of the shape deformation in between, meanwhile maintaining the temporal consistency in the generated video sequence. However, existing methods fail to model complicated motion patterns for video editing, and are fundamentally limited to object replacement, where tasks with non-rigid object motions like multi-object and portrait editing are largely neglected. In this paper, we observe that optical flows offer a promising alternative in complex motion modeling, and present FlowV2V to re-investigate video editing as a task of flow-driven Image-to-Video (I2V) generation. Specifically, FlowV2V decomposes the entire pipeline into first-frame editing and conditional I2V generation, and simulates pseudo flow sequence that aligns with the deformed shape, thus ensuring the consistency during editing. Experimental results on DAVIS-EDIT with improvements of 13.67% and 50.66% on DOVER and warping error illustrate the superior temporal consistency and sample quality of FlowV2V compared to existing state-of-the-art ones. Furthermore, we conduct comprehensive ablation studies to analyze the internal functionalities of the first-frame paradigm and flow alignment in the proposed method. 

**Abstract (ZH)**: 视频扩散模型繁荣背景下，低复杂度的视频编辑应用显著促进，特别是运动转移过程中的形状变形与时间一致性维护构成了这一任务的挑战。现有方法难以建模复杂的运动模式，主要局限于对象替换，对于涉及非刚性对象运动的多对象和人像编辑任务关注不足。本文观察到光流在复杂运动建模中提供了有望的替代方案，并提出FlowV2V重新研究视频编辑任务，作为一个由流驱动的图像到视频(I2V)生成任务。FlowV2V将整个管道分解为起始帧编辑和条件I2V生成，模拟与变形形状对齐的伪流序列，从而确保编辑过程中的时间一致性。在DAVIS-EDIT上的实验结果表明，FlowV2V在DOVER上的改进为13.67%，在变形误差上的改进为50.66%，展示了其优于现有最佳方法的优越时间一致性和样本质量。此外，我们进行了全面的消融研究，以分析所提方法中起始帧范式和流对齐的内部功能。 

---
# NOVA3D: Normal Aligned Video Diffusion Model for Single Image to 3D Generation 

**Title (ZH)**: NOVA3D: 基于法线对齐的视频扩散模型用于单张图像到3D生成 

**Authors**: Yuxiao Yang, Peihao Li, Yuhong Zhang, Junzhe Lu, Xianglong He, Minghan Qin, Weitao Wang, Haoqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07698)  

**Abstract**: 3D AI-generated content (AIGC) has made it increasingly accessible for anyone to become a 3D content creator. While recent methods leverage Score Distillation Sampling to distill 3D objects from pretrained image diffusion models, they often suffer from inadequate 3D priors, leading to insufficient multi-view consistency. In this work, we introduce NOVA3D, an innovative single-image-to-3D generation framework. Our key insight lies in leveraging strong 3D priors from a pretrained video diffusion model and integrating geometric information during multi-view video fine-tuning. To facilitate information exchange between color and geometric domains, we propose the Geometry-Temporal Alignment (GTA) attention mechanism, thereby improving generalization and multi-view consistency. Moreover, we introduce the de-conflict geometry fusion algorithm, which improves texture fidelity by addressing multi-view inaccuracies and resolving discrepancies in pose alignment. Extensive experiments validate the superiority of NOVA3D over existing baselines. 

**Abstract (ZH)**: 基于3D视频扩散模型的强大先验知识的单张图像到3D生成框架NOVA3D 

---
# GaRAGe: A Benchmark with Grounding Annotations for RAG Evaluation 

**Title (ZH)**: GaRAGe: 一种用于RAG评估的grounding注释基准 

**Authors**: Ionut-Teodor Sorodoc, Leonardo F. R. Ribeiro, Rexhina Blloshmi, Christopher Davis, Adrià de Gispert  

**Link**: [PDF](https://arxiv.org/pdf/2506.07671)  

**Abstract**: We present GaRAGe, a large RAG benchmark with human-curated long-form answers and annotations of each grounding passage, allowing a fine-grained evaluation of whether LLMs can identify relevant grounding when generating RAG answers. Our benchmark contains 2366 questions of diverse complexity, dynamism, and topics, and includes over 35K annotated passages retrieved from both private document sets and the Web, to reflect real-world RAG use cases. This makes it an ideal test bed to evaluate an LLM's ability to identify only the relevant information necessary to compose a response, or provide a deflective response when there is insufficient information. Evaluations of multiple state-of-the-art LLMs on GaRAGe show that the models tend to over-summarise rather than (a) ground their answers strictly on the annotated relevant passages (reaching at most a Relevance-Aware Factuality Score of 60%), or (b) deflect when no relevant grounding is available (reaching at most 31% true positive rate in deflections). The F1 in attribution to relevant sources is at most 58.9%, and we show that performance is particularly reduced when answering time-sensitive questions and when having to draw knowledge from sparser private grounding sources. 

**Abstract (ZH)**: GaRAGe：一个大型RAG基准，包含人工策展的长形式答案和每段接地内容的标注，以精细评估LLMs在生成RAG答案时是否能识别相关接地信息 

---
# Synthesis by Design: Controlled Data Generation via Structural Guidance 

**Title (ZH)**: 设计合成：通过结构指导实现受控数据生成 

**Authors**: Lei Xu, Sirui Chen, Yuxuan Huang, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07664)  

**Abstract**: Mathematical reasoning remains challenging for LLMs due to complex logic and the need for precise computation. Existing methods enhance LLM reasoning by synthesizing datasets through problem rephrasing, but face issues with generation quality and problem complexity. To address this, we propose to extract structural information with generated problem-solving code from mathematical reasoning and guide data generation with structured solutions. Applied to MATH and GSM8K, our approach produces 39K problems with labeled intermediate steps and a 6.1K-problem benchmark of higher difficulty. Results on our benchmark show that model performance declines as reasoning length increases. Additionally, we conducted fine-tuning experiments using the proposed training data on a range of LLMs, and the results validate the effectiveness of our dataset. We hope the proposed method and dataset will contribute to future research in enhancing LLM reasoning capabilities. 

**Abstract (ZH)**: 数学推理对于LLMs仍然是具有挑战性的，因为涉及到复杂的逻辑和精确的计算需求。现有方法通过重新表述问题来合成数据集以增强LLM的推理能力，但面临生成质量和问题复杂性的问题。为此，我们提出从数学推理中提取生成的问题解决代码的结构信息，并利用结构化解决方案指导数据生成。在MATH和GSM8K上应用此方法，产生了39,000个带有标注中间步骤的问题，并构建了一个包含6,100个更高难度问题的基准数据集。在该基准数据集上的结果显示，随着推理长度的增加，模型性能下降。此外，我们使用提出的训练数据对一系列LLMs进行了微调实验，实验结果验证了我们数据集的有效性。我们希望提出的该方法和数据集能够促进未来研究以增强LLM的推理能力。 

---
# FMaMIL: Frequency-Driven Mamba Multi-Instance Learning for Weakly Supervised Lesion Segmentation in Medical Images 

**Title (ZH)**: FMaMIL：基于频率的Mamba多实例学习在医学图像弱监督病灶分割中的应用 

**Authors**: Hangbei Cheng, Xiaorong Dong, Xueyu Liu, Jianan Zhang, Xuetao Ma, Mingqiang Wei, Liansheng Wang, Junxin Chen, Yongfei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07652)  

**Abstract**: Accurate lesion segmentation in histopathology images is essential for diagnostic interpretation and quantitative analysis, yet it remains challenging due to the limited availability of costly pixel-level annotations. To address this, we propose FMaMIL, a novel two-stage framework for weakly supervised lesion segmentation based solely on image-level labels. In the first stage, a lightweight Mamba-based encoder is introduced to capture long-range dependencies across image patches under the MIL paradigm. To enhance spatial sensitivity and structural awareness, we design a learnable frequency-domain encoding module that supplements spatial-domain features with spectrum-based information. CAMs generated in this stage are used to guide segmentation training. In the second stage, we refine the initial pseudo labels via a CAM-guided soft-label supervision and a self-correction mechanism, enabling robust training even under label noise. Extensive experiments on both public and private histopathology datasets demonstrate that FMaMIL outperforms state-of-the-art weakly supervised methods without relying on pixel-level annotations, validating its effectiveness and potential for digital pathology applications. 

**Abstract (ZH)**: 准确的病理图像病变分割对于诊断解释和定量分析至关重要，但由于像素级注解成本高且获取有限，这一任务仍具有挑战性。为解决这一问题，我们提出了一种基于图像级标签的新型两阶段弱监督病变分割框架FMaMIL。在第一阶段，引入基于Mamba的编码器在MIL范式下捕捉图像patches间的长程依赖关系。为了增强空间敏感性和结构意识，设计了一种可学习的频域编码模块，通过频谱信息补充空域特征。该阶段生成的CAM用于指导分割训练。在第二阶段，通过CAM引导的软标签监督和自我纠正机制细化初始伪标签，即使在标签噪声下也能实现稳健训练。在公共和私有病理图像数据集上的 extensive 实验表明，FMaMIL 在无需依赖像素级注解的情况下超过了最先进的弱监督方法，验证了其在数字病理学应用中的有效性和潜力。 

---
# LoRMA: Low-Rank Multiplicative Adaptation for LLMs 

**Title (ZH)**: LoRMA：低秩乘法适应性方法用于LLMs 

**Authors**: Harsh Bihany, Shubham Patel, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07621)  

**Abstract**: Large Language Models have shown remarkable capabilities in the NLP domain. Their effectiveness can mainly be attributed to their ability to adapt to an array of downstream tasks. However, generally, full fine-tuning is a computationally expensive job. To mitigate this, many techniques have been developed that prime efficiency, a prominent one being Low-Rank Adaptation (LoRA). However, LoRA and its variants employ re-parametrized additive updates. In this paper, we propose Low-Rank Multiplicative Adaptation (LoRMA), which shifts the paradigm of additive updates to a richer space of matrix multiplicative transformations. We tackle challenges such as computational complexity and rank bottleneck of matrix multiplication by effectively re-ordering operations and introducing rank inflation strategies. We conduct extensive experiments to demonstrate the effectiveness of our approach in terms of various evaluation metrics. 

**Abstract (ZH)**: 大型语言模型在NLP领域展现了显著的能力。它们的有效性主要归因于其适应多种下游任务的能力。然而，通常完整的微调是一个计算成本高昂的过程。为解决这一问题，许多技术被开发出来以提高效率，其中一种显著的技术是低秩适应（LoRA）。然而，LoRA及其变体采用的是重参数化的加性更新。本文中，我们提出了一种低秩乘性适应（LoRMA），将加性更新的范式转移到更丰富的矩阵乘性变换空间。通过有效重排操作和引入秩膨胀策略，我们应对了矩阵乘法的计算复杂性和秩瓶颈问题。我们进行了广泛实验，从多种评估指标展示了我们方法的有效性。 

---
# PolitiSky24: U.S. Political Bluesky Dataset with User Stance Labels 

**Title (ZH)**: PolitiSky24: 美国政治 bluesky 数据集及其用户立场标签 

**Authors**: Peyman Rostami, Vahid Rahimzadeh, Ali Adibi, Azadeh Shakery  

**Link**: [PDF](https://arxiv.org/pdf/2506.07606)  

**Abstract**: Stance detection identifies the viewpoint expressed in text toward a specific target, such as a political figure. While previous datasets have focused primarily on tweet-level stances from established platforms, user-level stance resources, especially on emerging platforms like Bluesky remain scarce. User-level stance detection provides a more holistic view by considering a user's complete posting history rather than isolated posts. We present the first stance detection dataset for the 2024 U.S. presidential election, collected from Bluesky and centered on Kamala Harris and Donald Trump. The dataset comprises 16,044 user-target stance pairs enriched with engagement metadata, interaction graphs, and user posting histories. PolitiSky24 was created using a carefully evaluated pipeline combining advanced information retrieval and large language models, which generates stance labels with supporting rationales and text spans for transparency. The labeling approach achieves 81\% accuracy with scalable LLMs. This resource addresses gaps in political stance analysis through its timeliness, open-data nature, and user-level perspective. The dataset is available at this https URL 

**Abstract (ZH)**: 基于 Bluesky 的 2024 美国总统选举立场检测数据集：以卡玛拉·哈里斯和唐纳德·特朗普为中心 

---
# SurgBench: A Unified Large-Scale Benchmark for Surgical Video Analysis 

**Title (ZH)**: SurgBench: 一个统一的大规模手术视频分析基准 

**Authors**: Jianhui Wei, Zikai Xiao, Danyu Sun, Luqi Gong, Zongxin Yang, Zuozhu Liu, Jian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07603)  

**Abstract**: Surgical video understanding is pivotal for enabling automated intraoperative decision-making, skill assessment, and postoperative quality improvement. However, progress in developing surgical video foundation models (FMs) remains hindered by the scarcity of large-scale, diverse datasets for pretraining and systematic evaluation. In this paper, we introduce \textbf{SurgBench}, a unified surgical video benchmarking framework comprising a pretraining dataset, \textbf{SurgBench-P}, and an evaluation benchmark, \textbf{SurgBench-E}. SurgBench offers extensive coverage of diverse surgical scenarios, with SurgBench-P encompassing 53 million frames across 22 surgical procedures and 11 specialties, and SurgBench-E providing robust evaluation across six categories (phase classification, camera motion, tool recognition, disease diagnosis, action classification, and organ detection) spanning 72 fine-grained tasks. Extensive experiments reveal that existing video FMs struggle to generalize across varied surgical video analysis tasks, whereas pretraining on SurgBench-P yields substantial performance improvements and superior cross-domain generalization to unseen procedures and modalities. Our dataset and code are available upon request. 

**Abstract (ZH)**: 手术视频理解对于实现自动化术中决策、技能评估以及术后质量改进至关重要。然而，由于缺乏大规模多样的预训练数据集和系统性评估，手术视频基础模型的发展进展受限。本文介绍了一种统一的手术视频基准框架SurgBench，包括预训练数据集SurgBench-P和评估基准SurgBench-E。SurgBench涵盖了多种多样的手术场景，SurgBench-P包含22种手术程序和11个专科的5300万帧视频，SurgBench-E提供了涵盖六个类别（阶段分类、相机运动、工具识别、疾病诊断、动作分类和器官检测）的72个细粒度任务的稳健评估。大量实验表明，现有的视频基础模型难以在不同的手术视频分析任务间泛化，而基于SurgBench-P的预训练则显著提高性能并展现出对未见手术程序和模态的更强跨域泛化能力。数据集和代码可根据需求获取。 

---
# SceneRAG: Scene-level Retrieval-Augmented Generation for Video Understanding 

**Title (ZH)**: SceneRAG：场景级检索增强生成的视频理解方法 

**Authors**: Nianbo Zeng, Haowen Hou, Fei Richard Yu, Si Shi, Ying Tiffany He  

**Link**: [PDF](https://arxiv.org/pdf/2506.07600)  

**Abstract**: Despite recent advances in retrieval-augmented generation (RAG) for video understanding, effectively understanding long-form video content remains underexplored due to the vast scale and high complexity of video data. Current RAG approaches typically segment videos into fixed-length chunks, which often disrupts the continuity of contextual information and fails to capture authentic scene boundaries. Inspired by the human ability to naturally organize continuous experiences into coherent scenes, we present SceneRAG, a unified framework that leverages large language models to segment videos into narrative-consistent scenes by processing ASR transcripts alongside temporal metadata. SceneRAG further sharpens these initial boundaries through lightweight heuristics and iterative correction. For each scene, the framework fuses information from both visual and textual modalities to extract entity relations and dynamically builds a knowledge graph, enabling robust multi-hop retrieval and generation that account for long-range dependencies. Experiments on the LongerVideos benchmark, featuring over 134 hours of diverse content, confirm that SceneRAG substantially outperforms prior baselines, achieving a win rate of up to 72.5 percent on generation tasks. 

**Abstract (ZH)**: 基于检索增强生成的场景分割框架：面向长视频内容的统一方法 

---
# PrunePEFT: Iterative Hybrid Pruning for Parameter-Efficient Fine-tuning of LLMs 

**Title (ZH)**: PrunePEFT: 迭代混合修剪以实现LLMs的参数高效微调 

**Authors**: Tongzhou Yu, Zhuhao Zhang, Guanghui Zhu, Shen Jiang, Meikang Qiu, Yihua Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07587)  

**Abstract**: Parameter Efficient Fine-Tuning (PEFT) methods have emerged as effective and promising approaches for fine-tuning pre-trained language models. Compared with Full parameter Fine-Tuning (FFT), PEFT achieved comparable task performance with a substantial reduction of trainable parameters, which largely saved the training and storage costs. However, using the PEFT method requires considering a vast design space, such as the type of PEFT modules and their insertion layers. Inadequate configurations can lead to sub-optimal results. Conventional solutions such as architectural search techniques, while effective, tend to introduce substantial additional overhead. In this paper, we propose a novel approach, PrunePEFT, which formulates the PEFT strategy search as a pruning problem and introduces a hybrid pruning strategy that capitalizes on the sensitivity of pruning methods to different PEFT modules. This method extends traditional pruning techniques by iteratively removing redundant or conflicting PEFT modules, thereby optimizing the fine-tuned configuration. By efficiently identifying the most relevant modules, our approach significantly reduces the computational burden typically associated with architectural search processes, making it a more scalable and efficient solution for fine-tuning large pre-trained models. 

**Abstract (ZH)**: 参数高效微调（PEFT）方法已 emerge作为有效且有前途的预训练语言模型微调方法。与全参数微调（FFT）相比，PEFT 实现了可比拟的任务性能，同时大幅度减少了可训练参数的数量，大大节省了训练和存储成本。然而，使用 PEFT 方法需要考虑广泛的设计空间，如 PEFT 模块的类型及其插入层。不适当的配置可能导致亚优结果。尽管传统的架构搜索技术有效，但往往会引入大量额外开销。在本文中，我们提出了一种新颖的方法 PrunePEFT，将 PEFT 策略搜索形式化为剪枝问题，并引入了一种结合了不同 PEFT 模块对剪枝方法敏感性的混合剪枝策略。该方法通过迭代移除冗余或冲突的 PEFT 模块，扩展了传统的剪枝技术，从而优化了微调配置。通过高效地识别最相关的模块，我们的方法显著减少了通常与架构搜索过程相关的计算负担，使其成为大规模预训练模型微调的一种更具扩展性和高效性解决方案。 

---
# Beyond the Sentence: A Survey on Context-Aware Machine Translation with Large Language Models 

**Title (ZH)**: 超越句子：面向上下文的大语言模型机器翻译综述 

**Authors**: Ramakrishna Appicharla, Baban Gain, Santanu Pal, Asif Ekbal  

**Link**: [PDF](https://arxiv.org/pdf/2506.07583)  

**Abstract**: Despite the popularity of the large language models (LLMs), their application to machine translation is relatively underexplored, especially in context-aware settings. This work presents a literature review of context-aware translation with LLMs. The existing works utilise prompting and fine-tuning approaches, with few focusing on automatic post-editing and creating translation agents for context-aware machine translation. We observed that the commercial LLMs (such as ChatGPT and Tower LLM) achieved better results than the open-source LLMs (such as Llama and Bloom LLMs), and prompt-based approaches serve as good baselines to assess the quality of translations. Finally, we present some interesting future directions to explore. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）的应用广泛，但它们在上下文感知翻译中的应用相对未被充分探索。本文综述了使用LLMs进行上下文感知翻译的相关研究。现有的研究主要采用了提示和微调方法，很少关注自动后编辑和构建上下文感知机器翻译的翻译代理。我们观察到商用LLMs（如ChatGPT和Tower LLM）的性能优于开源LLMs（如Llama和Bloom LLM），提示基方法可以作为评估翻译质量的良好基准。最后，我们提出了若干有趣的研究方向。 

---
# FedCGD: Collective Gradient Divergence Optimized Scheduling for Wireless Federated Learning 

**Title (ZH)**: FedCGD：集体梯度散度优化调度的无线联邦学习 

**Authors**: Tan Chen, Jintao Yan, Yuxuan Sun, Sheng Zhou, Zhisheng Niu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07581)  

**Abstract**: Federated learning (FL) is a promising paradigm for multiple devices to cooperatively train a model. When applied in wireless networks, two issues consistently affect the performance of FL, i.e., data heterogeneity of devices and limited bandwidth. Many papers have investigated device scheduling strategies considering the two issues. However, most of them recognize data heterogeneity as a property of individual devices. In this paper, we prove that the convergence speed of FL is affected by the sum of device-level and sample-level collective gradient divergence (CGD). The device-level CGD refers to the gradient divergence of the scheduled device group, instead of the sum of the individual device divergence. The sample-level CGD is statistically upper bounded by sampling variance, which is inversely proportional to the total number of samples scheduled for local update. To derive a tractable form of the device-level CGD, we further consider a classification problem and transform it into the weighted earth moving distance (WEMD) between the group distribution and the global distribution. Then we propose FedCGD algorithm to minimize the sum of multi-level CGDs by balancing WEMD and sampling variance, within polynomial time. Simulation shows that the proposed strategy increases classification accuracy on the CIFAR-10 dataset by up to 4.2\% while scheduling 41.8\% fewer devices, and flexibly switches between reducing WEMD and reducing sampling variance. 

**Abstract (ZH)**: 联邦学习中的设备调度策略：考虑设备级和样本级集体梯度发散的优化方法 

---
# Denoising the Future: Top-p Distributions for Moving Through Time 

**Title (ZH)**: 未来去噪：Moving Through Time中的top-p分布 

**Authors**: Florian Andreas Marwitz, Ralf Möller, Magnus Bender, Marcel Gehrke  

**Link**: [PDF](https://arxiv.org/pdf/2506.07578)  

**Abstract**: Inference in dynamic probabilistic models is a complex task involving expensive operations. In particular, for Hidden Markov Models, the whole state space has to be enumerated for advancing in time. Even states with negligible probabilities are considered, resulting in computational inefficiency and increased noise due to the propagation of unlikely probability mass. We propose to denoise the future and speed up inference by using only the top-p states, i.e., the most probable states with accumulated probability p. We show that the error introduced by using only the top-p states is bound by p and the so-called minimal mixing rate of the underlying model. Moreover, in our empirical evaluation, we show that we can expect speedups of at least an order of magnitude, while the error in terms of total variation distance is below 0.09. 

**Abstract (ZH)**: 动态概率模型中的推断是一个复杂任务，涉及昂贵的操作。特别是对于隐藏马尔可夫模型，必须遍历整个状态空间以实现时间推进。即使是概率可以忽略不计的状态也予以考虑，导致计算效率低下并增加由于 unlikely 概率质量传播造成的噪声。我们提出通过仅使用最 probable 的 top-p 状态来降噪和加速推断。结果显示，仅使用 top-p 状态引入的误差受 p 和底层模型的最小混合率限制。此外，在我们的实证评估中，我们表明可以期望至少一个数量级的加速，同时在总变差距离意义上的误差低于 0.09。 

---
# LLM-driven Indoor Scene Layout Generation via Scaled Human-aligned Data Synthesis and Multi-Stage Preference Optimization 

**Title (ZH)**: 基于按比例缩放人类对齐数据合成和多阶段偏好优化的LLM驱动室内场景布局生成 

**Authors**: Yixuan Yang, Zhen Luo, Tongsheng Ding, Junru Lu, Mingqi Gao, Jinyu Yang, Victor Sanchez, Feng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.07570)  

**Abstract**: Automatic indoor layout generation has attracted increasing attention due to its potential in interior design, virtual environment construction, and embodied AI. Existing methods fall into two categories: prompt-driven approaches that leverage proprietary LLM services (e.g., GPT APIs) and learning-based methods trained on layout data upon diffusion-based models. Prompt-driven methods often suffer from spatial inconsistency and high computational costs, while learning-based methods are typically constrained by coarse relational graphs and limited datasets, restricting their generalization to diverse room categories. In this paper, we revisit LLM-based indoor layout generation and present 3D-SynthPlace, a large-scale dataset that combines synthetic layouts generated via a 'GPT synthesize, Human inspect' pipeline, upgraded from the 3D-Front dataset. 3D-SynthPlace contains nearly 17,000 scenes, covering four common room types -- bedroom, living room, kitchen, and bathroom -- enriched with diverse objects and high-level spatial annotations. We further introduce OptiScene, a strong open-source LLM optimized for indoor layout generation, fine-tuned based on our 3D-SynthPlace dataset through our two-stage training. For the warum-up stage I, we adopt supervised fine-tuning (SFT), which is taught to first generate high-level spatial descriptions then conditionally predict concrete object placements. For the reinforcing stage II, to better align the generated layouts with human design preferences, we apply multi-turn direct preference optimization (DPO), which significantly improving layout quality and generation success rates. Extensive experiments demonstrate that OptiScene outperforms traditional prompt-driven and learning-based baselines. Moreover, OptiScene shows promising potential in interactive tasks such as scene editing and robot navigation. 

**Abstract (ZH)**: 自动室内布局生成由于其在室内设计、虚拟环境构建和具身AI等方面的应用潜力而引起了越来越多的关注。现有的方法可以分为两类：提示驱动的方法利用专有LLM服务（如GPT API）以及基于扩散模型训练的布局数据的学习方法。提示驱动的方法通常存在空间不一致性和高计算成本的问题，而学习方法则通常受到粗粒度关系图和有限数据集的限制，限制了其对多样化房间类别的泛化能力。在本文中，我们重新审视基于LLM的室内布局生成，并介绍了3D-SynthPlace，这是一个大规模数据集，结合了通过“GPT合成、人工检查”流水线生成的合成布局，源自3D-Front数据集的升级版。3D-SynthPlace包含近17,000个场景，涵盖四种常见房间类型——卧室、客厅、厨房和浴室，并配备了多种物体和高层次的空间注释。我们进一步介绍了OptiScene，这是一个面向室内布局生成的强大开源LLM，基于我们3D-SynthPlace数据集进行了两阶段微调。在准备阶段I中，我们采用了监督微调（SFT），使其首先生成高层次的空间描述，然后条件性地预测具体的物体排列。在强化阶段II中，为了更好地使生成的布局与人类的设计偏好对齐，我们应用了多轮直接偏好优化（DPO），显著提高了布局质量并提升了生成成功率。广泛实验表明，OptiScene在传统提示驱动和学习方法基线下表现出色。此外，OptiScene在场景编辑和机器人导航等交互任务中展现出良好的应用前景。 

---
# MoE-MLoRA for Multi-Domain CTR Prediction: Efficient Adaptation with Expert Specialization 

**Title (ZH)**: MoE-MLoRA在多领域点击率预测中的高效适应：专家专业化精化 

**Authors**: Ken Yagel, Eyal German, Aviel Ben Siman Tov  

**Link**: [PDF](https://arxiv.org/pdf/2506.07563)  

**Abstract**: Personalized recommendation systems must adapt to user interactions across different domains. Traditional approaches like MLoRA apply a single adaptation per domain but lack flexibility in handling diverse user behaviors. To address this, we propose MoE-MLoRA, a mixture-of-experts framework where each expert is first trained independently to specialize in its domain before a gating network is trained to weight their contributions dynamically. We evaluate MoE-MLoRA across eight CTR models on Movielens and Taobao, showing that it improves performance in large-scale, dynamic datasets (+1.45 Weighed-AUC in Taobao-20) but offers limited benefits in structured datasets with low domain diversity and sparsity. Further analysis of the number of experts per domain reveals that larger ensembles do not always improve performance, indicating the need for model-aware tuning. Our findings highlight the potential of expert-based architectures for multi-domain recommendation systems, demonstrating that task-aware specialization and adaptive gating can enhance predictive accuracy in complex environments. The implementation and code are available in our GitHub repository. 

**Abstract (ZH)**: 个性化推荐系统必须适应用户在不同领域的交互。传统的MLoRA等方法在每个领域仅应用单一的适应性，缺乏处理多样化用户行为的灵活性。为了解决这一问题，我们提出了一种专家混合框架MoE-MLoRA，在此框架中，每个专家首先在独立训练中 specialize 于其所在领域，之后训练一个门控网络以动态加权其各部分的贡献。我们在 Movielens 和 Taobao 上的八种点击率模型上评估了 MoE-MLoRA，结果显示，它在大规模动态数据集上提高了性能（Taobao-20 上的 Weighed-AUC 提高了 1.45），但在域多样性低且数据稀疏的结构化数据集上提供了有限的益处。进一步分析每个领域的专家数量表明，更大的集成模型并不总是提高性能，表明需要进行模型感知的调优。我们的研究结果强调了基于专家架构在多域推荐系统中的潜力，表明任务感知的专业化和自适应门控能够增强复杂环境中的预测准确性。相关实现和代码已在我们的 GitHub 仓库中提供。 

---
# SELT: Self-Evaluation Tree Search for LLMs with Task Decomposition 

**Title (ZH)**: SELF-Evaluation Tree Search for LLMs with Task Decomposition 

**Authors**: Mengsong Wu, Di Zhang, Yuqiang Li, Dongzhan Zhou, Wenliang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.07557)  

**Abstract**: While Large Language Models (LLMs) have achieved remarkable success in a wide range of applications, their performance often degrades in complex reasoning tasks. In this work, we introduce SELT (Self-Evaluation LLM Tree Search), a novel framework that leverages a modified Monte Carlo Tree Search (MCTS) to enhance LLM reasoning without relying on external reward models. By redefining the Upper Confidence Bound scoring to align with intrinsic self-evaluation capabilities of LLMs and decomposing the inference process into atomic subtasks augmented with semantic clustering at each node, SELT effectively balances exploration and exploitation, reduces redundant reasoning paths, and mitigates hallucination. We validate our approach on challenging benchmarks, including the knowledge-based MMLU and the Tool Learning dataset Seal-Tools, where SELT achieves significant improvements in answer accuracy and reasoning robustness compared to baseline methods. Notably, our framework operates without task-specific fine-tuning, demonstrating strong generalizability across diverse reasoning tasks. Relevant results and code are available at this https URL . 

**Abstract (ZH)**: 虽然大型语言模型（LLMs）在广泛的应用中取得了显著的成功，但在复杂的推理任务中其性能往往会出现下降。在这项工作中，我们引入了SELT（Self-Evaluation LLM Tree Search）框架，该框架利用修改后的蒙特卡洛树搜索（MCTS）来增强LLM的推理能力，而无需依赖外部奖励模型。通过重新定义上层置信界评分以与LLM的内在自我评估能力对齐，并将推理过程分解为带有所谓语义聚类的原子子任务，SELT有效地平衡了探索与利用，减少了冗余的推理路径，并减轻了幻觉现象。我们在包括基于知识的MMLU和Tool Learning数据集Seal-Tools在内的具有挑战性的基准测试上验证了该方法，结果表明SELT在答案准确性和推理 robustness方面相较于基线方法实现了显著改进。特别地，我们的框架无需针对特定任务进行微调，显示出在各种推理任务中的强泛化能力。更多相关结果和代码请访问此链接：this https URL 

---
# Synthesize Privacy-Preserving High-Resolution Images via Private Textual Intermediaries 

**Title (ZH)**: 通过私有文本中介合成隐私保护高分辨率图像 

**Authors**: Haoxiang Wang, Zinan Lin, Da Yu, Huishuai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07555)  

**Abstract**: Generating high fidelity, differentially private (DP) synthetic images offers a promising route to share and analyze sensitive visual data without compromising individual privacy. However, existing DP image synthesis methods struggle to produce high resolution outputs that faithfully capture the structure of the original data. In this paper, we introduce a novel method, referred to as Synthesis via Private Textual Intermediaries (SPTI), that can generate high resolution DP images with easy adoption. The key idea is to shift the challenge of DP image synthesis from the image domain to the text domain by leveraging state of the art DP text generation methods. SPTI first summarizes each private image into a concise textual description using image to text models, then applies a modified Private Evolution algorithm to generate DP text, and finally reconstructs images using text to image models. Notably, SPTI requires no model training, only inference with off the shelf models. Given a private dataset, SPTI produces synthetic images of substantially higher quality than prior DP approaches. On the LSUN Bedroom dataset, SPTI attains an FID less than or equal to 26.71 under epsilon equal to 1.0, improving over Private Evolution FID of 40.36. Similarly, on MM CelebA HQ, SPTI achieves an FID less than or equal to 33.27 at epsilon equal to 1.0, compared to 57.01 from DP fine tuning baselines. Overall, our results demonstrate that Synthesis via Private Textual Intermediaries provides a resource efficient and proprietary model compatible framework for generating high resolution DP synthetic images, greatly expanding access to private visual datasets. 

**Abstract (ZH)**: 生成高保真差分隐私(DP)合成图像提供了不泄露个体隐私前提下共享和分析敏感视觉数据的有希望途径。然而，现有的DP图像合成方法难以生成高分辨率且忠实再现原始数据结构的输出。在本文中，我们提出了一种新颖的方法，称为通过私有文本中介合成（SPTI），可以生成高分辨率的DP图像且易于采用。核心思想是通过利用最先进的DP文本生成方法，将DP图像合成的挑战从图像域转移到文本域。SPTI 首先使用图像到文本模型将每张私人图像总结为简洁的文本描述，然后应用修改过的私有进化算法生成DP文本，并最终使用文本到图像模型重建图像。值得注意的是，SPTI 不需要模型训练，仅需使用现成的模型进行推断。给定一个私人数据集，SPTI 生成的合成图像的质量显著高于先前的DP方法。在LSUN Bedroom数据集上，当ε=1.0时，SPTI 的FID小于或等于26.71，优于私有进化方法的FID 40.36。同样，在MM CelebA HQ 上，当 ε=1.0 时，SPTI 达到的 FID 小于或等于33.27，而 DP 微调基线的 FID 为 57.01。总体而言，我们的结果表明，通过私有文本中介合成提供了一种资源高效且与专有模型兼容的框架，用于生成高分辨率的DP合成图像，极大地扩展了对私人视觉数据集的访问。 

---
# ChemAgent: Enhancing LLMs for Chemistry and Materials Science through Tree-Search Based Tool Learning 

**Title (ZH)**: ChemAgent: 通过基于树搜索的工具学习提升化学和材料科学中的LLM性能 

**Authors**: Mengsong Wu, YaFei Wang, Yidong Ming, Yuqi An, Yuwei Wan, Wenliang Chen, Binbin Lin, Yuqiang Li, Tong Xie, Dongzhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.07551)  

**Abstract**: Large language models (LLMs) have recently demonstrated promising capabilities in chemistry tasks while still facing challenges due to outdated pretraining knowledge and the difficulty of incorporating specialized chemical expertise. To address these issues, we propose an LLM-based agent that synergistically integrates 137 external chemical tools created ranging from basic information retrieval to complex reaction predictions, and a dataset curation pipeline to generate the dataset ChemToolBench that facilitates both effective tool selection and precise parameter filling during fine-tuning and evaluation. We introduce a Hierarchical Evolutionary Monte Carlo Tree Search (HE-MCTS) framework, enabling independent optimization of tool planning and execution. By leveraging self-generated data, our approach supports step-level fine-tuning (FT) of the policy model and training task-adaptive PRM and ORM that surpass GPT-4o. Experimental evaluations demonstrate that our approach significantly improves performance in Chemistry QA and discovery tasks, offering a robust solution to integrate specialized tools with LLMs for advanced chemical applications. All datasets and code are available at this https URL . 

**Abstract (ZH)**: 大型语言模型（LLMs）在化学任务中展现了令人鼓舞的能力，但仍面临由于过时的预训练知识和将专业化学知识集成的难度而带来的挑战。为了解决这些问题，我们提出了一种基于LLM的代理，该代理协同整合了137个外部化学工具，涵盖从基本信息检索到复杂反应预测的各个方面，并建立了一个数据集整理管道以生成Facilitating Tool Selection and Precise Parameter Filling during Fine-Tuning and Evaluation的ChemToolBench数据集。我们引入了一种分层演化蒙特卡罗树搜索（HE-MCTS）框架，使工具规划和执行的优化得以独立进行。通过利用自动生成的数据，我们的方法支持策略模型的步骤级微调（FT）和训练自适应任务PRM和ORM，超越了GPT-4o。实验评估显示，我们的方法显著提高了化学问答和发现任务中的性能，提供了一种集成专业工具与LLM的稳健方法，适用于高级化学应用。所有数据集和代码均可在此网址访问。 

---
# APTOS-2024 challenge report: Generation of synthetic 3D OCT images from fundus photographs 

**Title (ZH)**: APTOS-2024挑战报告：从 fundus 照片生成合成三维 OCT 图像 

**Authors**: Bowen Liu, Weiyi Zhang, Peranut Chotcomwongse, Xiaolan Chen, Ruoyu Chen, Pawin Pakaymaskul, Niracha Arjkongharn, Nattaporn Vongsa, Xuelian Cheng, Zongyuan Ge, Kun Huang, Xiaohui Li, Yiru Duan, Zhenbang Wang, BaoYe Xie, Qiang Chen, Huazhu Fu, Michael A. Mahr, Jiaqi Qu, Wangyiyang Chen, Shiye Wang, Yubo Tan, Yongjie Li, Mingguang He, Danli Shi, Paisan Ruamviboonsuk  

**Link**: [PDF](https://arxiv.org/pdf/2506.07542)  

**Abstract**: Optical Coherence Tomography (OCT) provides high-resolution, 3D, and non-invasive visualization of retinal layers in vivo, serving as a critical tool for lesion localization and disease diagnosis. However, its widespread adoption is limited by equipment costs and the need for specialized operators. In comparison, 2D color fundus photography offers faster acquisition and greater accessibility with less dependence on expensive devices. Although generative artificial intelligence has demonstrated promising results in medical image synthesis, translating 2D fundus images into 3D OCT images presents unique challenges due to inherent differences in data dimensionality and biological information between modalities. To advance generative models in the fundus-to-3D-OCT setting, the Asia Pacific Tele-Ophthalmology Society (APTOS-2024) organized a challenge titled Artificial Intelligence-based OCT Generation from Fundus Images. This paper details the challenge framework (referred to as APTOS-2024 Challenge), including: the benchmark dataset, evaluation methodology featuring two fidelity metrics-image-based distance (pixel-level OCT B-scan similarity) and video-based distance (semantic-level volumetric consistency), and analysis of top-performing solutions. The challenge attracted 342 participating teams, with 42 preliminary submissions and 9 finalists. Leading methodologies incorporated innovations in hybrid data preprocessing or augmentation (cross-modality collaborative paradigms), pre-training on external ophthalmic imaging datasets, integration of vision foundation models, and model architecture improvement. The APTOS-2024 Challenge is the first benchmark demonstrating the feasibility of fundus-to-3D-OCT synthesis as a potential solution for improving ophthalmic care accessibility in under-resourced healthcare settings, while helping to expedite medical research and clinical applications. 

**Abstract (ZH)**: 基于眼底图像的光学相干断层扫描生成挑战（APTOS-2024） 

---
# Domain Randomization for Object Detection in Manufacturing Applications using Synthetic Data: A Comprehensive Study 

**Title (ZH)**: 基于合成数据的域随机化在制造应用中进行对象检测：一项综合性研究 

**Authors**: Xiaomeng Zhu, Jacob Henningsson, Duruo Li, Pär Mårtensson, Lars Hanson, Mårten Björkman, Atsuto Maki  

**Link**: [PDF](https://arxiv.org/pdf/2506.07539)  

**Abstract**: This paper addresses key aspects of domain randomization in generating synthetic data for manufacturing object detection applications. To this end, we present a comprehensive data generation pipeline that reflects different factors: object characteristics, background, illumination, camera settings, and post-processing. We also introduce the Synthetic Industrial Parts Object Detection dataset (SIP15-OD) consisting of 15 objects from three industrial use cases under varying environments as a test bed for the study, while also employing an industrial dataset publicly available for robotic applications. In our experiments, we present more abundant results and insights into the feasibility as well as challenges of sim-to-real object detection. In particular, we identified material properties, rendering methods, post-processing, and distractors as important factors. Our method, leveraging these, achieves top performance on the public dataset with Yolov8 models trained exclusively on synthetic data; mAP@50 scores of 96.4% for the robotics dataset, and 94.1%, 99.5%, and 95.3% across three of the SIP15-OD use cases, respectively. The results showcase the effectiveness of the proposed domain randomization, potentially covering the distribution close to real data for the applications. 

**Abstract (ZH)**: 本文探讨了领域随机化在生成制造对象检测应用所需合成数据中的关键方面。为此，我们提出了一个综合数据生成管道，涵盖了对象特性、背景、照明、相机设置和后处理等因素。同时，我们引入了包含来自三个工业应用场景的15个物体且环境不同的合成工业部件对象检测数据集（SIP15-OD）作为研究的基础，并使用一个公开的工业数据集作为机器人的应用背景。在实验中，我们展示了丰富的结果和见解，探讨了模拟到实际对象检测的可行性和挑战。特别地，我们确定了材料属性、渲染方法、后处理和干扰物为重要因素。利用这些因素，我们的方法在仅使用合成数据训练Yolov8模型的公开数据集上取得了最优性能：机器人数据集的mAP@50得分为96.4%，SIP15-OD数据集的三个应用案例分别达到94.1%、99.5%和95.3%。结果展示了所提领域随机化的有效性，可能接近真实数据的应用分布。 

---
# IntenTest: Stress Testing for Intent Integrity in API-Calling LLM Agents 

**Title (ZH)**: IntenTest: API-调用LLM代理的意图完整性压力测试 

**Authors**: Shiwei Feng, Xiangzhe Xu, Xuan Chen, Kaiyuan Zhang, Syed Yusuf Ahmed, Zian Su, Mingwei Zheng, Xiangyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07524)  

**Abstract**: LLM agents are increasingly deployed to automate real-world tasks by invoking APIs through natural language instructions. While powerful, they often suffer from misinterpretation of user intent, leading to the agent's actions that diverge from the user's intended goal, especially as external toolkits evolve. Traditional software testing assumes structured inputs and thus falls short in handling the ambiguity of natural language. We introduce IntenTest, an API-centric stress testing framework that systematically uncovers intent integrity violations in LLM agents. Unlike prior work focused on fixed benchmarks or adversarial inputs, IntenTest generates realistic tasks based on toolkits' documentation and applies targeted mutations to expose subtle agent errors while preserving user intent. To guide testing, we propose semantic partitioning, which organizes natural language tasks into meaningful categories based on toolkit API parameters and their equivalence classes. Within each partition, seed tasks are mutated and ranked by a lightweight predictor that estimates the likelihood of triggering agent errors. To enhance efficiency, IntenTest maintains a datatype-aware strategy memory that retrieves and adapts effective mutation patterns from past cases. Experiments on 80 toolkit APIs demonstrate that IntenTest effectively uncovers intent integrity violations, significantly outperforming baselines in both error-exposing rate and query efficiency. Moreover, IntenTest generalizes well to stronger target models using smaller LLMs for test generation, and adapts to evolving APIs across domains. 

**Abstract (ZH)**: 基于API的意图完整性检测框架IntenTest 

---
# LeVo: High-Quality Song Generation with Multi-Preference Alignment 

**Title (ZH)**: LeVo：多偏好对齐的高质量歌曲生成 

**Authors**: Shun Lei, Yaoxun Xu, Zhiwei Lin, Huaicheng Zhang, Wei Tan, Hangting Chen, Jianwei Yu, Yixuan Zhang, Chenyu Yang, Haina Zhu, Shuai Wang, Zhiyong Wu, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07520)  

**Abstract**: Recent advances in large language models (LLMs) and audio language models have significantly improved music generation, particularly in lyrics-to-song generation. However, existing approaches still struggle with the complex composition of songs and the scarcity of high-quality data, leading to limitations in sound quality, musicality, instruction following, and vocal-instrument harmony. To address these challenges, we introduce LeVo, an LM-based framework consisting of LeLM and a music codec. LeLM is capable of parallelly modeling two types of tokens: mixed tokens, which represent the combined audio of vocals and accompaniment to achieve vocal-instrument harmony, and dual-track tokens, which separately encode vocals and accompaniment for high-quality song generation. It employs two decoder-only transformers and a modular extension training strategy to prevent interference between different token types. To further enhance musicality and instruction following, we introduce a multi-preference alignment method based on Direct Preference Optimization (DPO). This method handles diverse human preferences through a semi-automatic data construction process and DPO post-training. Experimental results demonstrate that LeVo consistently outperforms existing methods on both objective and subjective metrics. Ablation studies further justify the effectiveness of our designs. Audio examples are available at this https URL. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）和音频语言模型在音乐生成，尤其是歌词到歌曲生成方面取得了显著进展。然而，现有方法仍然在处理歌曲的复杂组成和高质量数据的稀缺性方面存在局限性，导致在音质、音乐性、指令跟随和人声乐器和声方面存在限制。为了解决这些挑战，我们引入了LeVo，一个基于LM的框架，包含LeLM和一个音乐编解码器。LeLM能够并行建模两种类型的标记：混合标记，用于表示人声和伴奏以实现人声乐器和声；以及双重轨道标记，分别编码人声和伴奏以实现高质量的歌曲生成。它采用了两个仅解码器变压器和模块化扩展训练策略，以防止不同标记类型的干扰。为了进一步提高音乐性和指令跟随，我们基于直接偏好优化（DPO）引入了一种多偏好对齐方法。该方法通过半自动数据构建过程和DPO后训练处理不同的人类偏好。实验结果表明，LeVo在客观和主观指标上均优于现有方法。消融研究进一步证明了我们设计的有效性。音频示例请参见此链接。 

---
# Reinforcement Learning via Implicit Imitation Guidance 

**Title (ZH)**: 强化学习通过隐式模仿引导 

**Authors**: Perry Dong, Alec M. Lessing, Annie S. Chen, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2506.07505)  

**Abstract**: We study the problem of sample efficient reinforcement learning, where prior data such as demonstrations are provided for initialization in lieu of a dense reward signal. A natural approach is to incorporate an imitation learning objective, either as regularization during training or to acquire a reference policy. However, imitation learning objectives can ultimately degrade long-term performance, as it does not directly align with reward maximization. In this work, we propose to use prior data solely for guiding exploration via noise added to the policy, sidestepping the need for explicit behavior cloning constraints. The key insight in our framework, Data-Guided Noise (DGN), is that demonstrations are most useful for identifying which actions should be explored, rather than forcing the policy to take certain actions. Our approach achieves up to 2-3x improvement over prior reinforcement learning from offline data methods across seven simulated continuous control tasks. 

**Abstract (ZH)**: 基于数据指导噪声的样本高效强化学习 

---
# Graph-of-Causal Evolution: Challenging Chain-of-Model for Reasoning 

**Title (ZH)**: 因果演化图：挑战模型链推理 

**Authors**: Libo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07501)  

**Abstract**: In view of the problem that each subchain in the chain-of-model (CoM) relies only on the information of the previous subchain and may lose long-range dependencies due to the causal mask blocking the global context flow between multi-level subchains, this work proposes a graph of causal evolution (GoCE). Its core principle is to map the implicit token representation into a differentiable and sparse causal adjacency matrix, then permeate causal constraints through each layer of calculation using causal-masked attention and causal-MoE. By combining intervention consistency loss test and self-evolution gate, the dynamic balance between causal structure learning and adaptive updating of transformer architecture is realized. The researcher built experimental environments in sandboxes built with Claude Sonnet 4, o4-mini-high, and DeepSeek R1 respectively with the transformer variant architecture introduced in GoCE. It is evaluated on publicly available datasets including CLUTRR, CLADDER, EX-FEVER, and CausalQA and compared with the baseline LLMs. The finding proves that GoCE strengthens the transformer's ability to capture long-range causal dependencies, while the ability to self-evolve is improved. It not only surpasses the design of CoM in terms of design principles, but also provides experience for future research on causal learning and continuous adaptive improvement. 

**Abstract (ZH)**: 基于因果演化图的链式模型因果长依赖增强研究 

---
# CoCoA-Mix: Confusion-and-Confidence-Aware Mixture Model for Context Optimization 

**Title (ZH)**: CoCoA-Mix: 混淆与自信-aware 混合模型用于情境优化 

**Authors**: Dasol Hong, Wooju Lee, Hyun Myung  

**Link**: [PDF](https://arxiv.org/pdf/2506.07484)  

**Abstract**: Prompt tuning, which adapts vision-language models by freezing model parameters and optimizing only the prompt, has proven effective for task-specific adaptations. The core challenge in prompt tuning is improving specialization for a specific task and generalization for unseen domains. However, frozen encoders often produce misaligned features, leading to confusion between classes and limiting specialization. To overcome this issue, we propose a confusion-aware loss (CoA-loss) that improves specialization by refining the decision boundaries between confusing classes. Additionally, we mathematically demonstrate that a mixture model can enhance generalization without compromising specialization. This is achieved using confidence-aware weights (CoA-weights), which adjust the weights of each prediction in the mixture model based on its confidence within the class domains. Extensive experiments show that CoCoA-Mix, a mixture model with CoA-loss and CoA-weights, outperforms state-of-the-art methods by enhancing specialization and generalization. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 提示调优通过冻结模型参数并仅优化提示，已证明对任务特定适应有效。提示调优的核心挑战是提高对特定任务的专业化程度并增强对未见过领域的泛化能力。然而，冻结的编码器往往会产生对齐不良的特征，导致类之间的混淆并限制专业化程度。为克服这一问题，我们提出了一种意识混淆损失（CoA-loss），通过细化混淆类之间的决策边界来提高专业化程度。此外，我们从数学上证明，混合模型可以在不牺牲专业化的情况下增强泛化能力。这是通过使用意识置信权重（CoA-weights）实现的，这些权重根据其在类域内的置信度调整混合模型中每个预测的权重。广泛实验表明，使用CoA-loss和CoA-weights的CoCoA-Mix混合模型相比现有最佳方法，在提高专业化和泛化方面表现出更优的效果。我们的代码在此处公开：这个链接。 

---
# Premise Selection for a Lean Hammer 

**Title (ZH)**: 精益锤 premise 选择 

**Authors**: Thomas Zhu, Joshua Clune, Jeremy Avigad, Albert Qiaochu Jiang, Sean Welleck  

**Link**: [PDF](https://arxiv.org/pdf/2506.07477)  

**Abstract**: Neural methods are transforming automated reasoning for proof assistants, yet integrating these advances into practical verification workflows remains challenging. Hammers are tools that interface with external automatic theorem provers to automate tedious reasoning steps. They have dramatically improved productivity in proof assistants, but the Lean proof assistant still does not have a hammer despite its growing popularity. We present LeanHammer, the first end-to-end domain-general hammer for Lean, built on a novel neural premise selection system for a hammer in dependent type theory. Unlike existing Lean premise selectors, our approach dynamically adapts to user-specific contexts and combines with symbolic proof search and reconstruction to create a practical hammer. With comprehensive evaluations, we show that our premise selector enables LeanHammer to solve 21\% more goals relative to existing premise selectors, and generalize well to diverse domains. Our work bridges the gap between neural retrieval and symbolic reasoning, making formal verification more accessible to researchers and practitioners. 

**Abstract (ZH)**: 神经方法正在Transforming自动化证明助手中的自动推理，但在将这些进步集成到实际验证工作流中仍然面临挑战。Hammer是与外部自动定理证明器接口的工具，用于自动化繁琐的推理步骤。它们显著提高了证明助手的 productivity，但Lean证明助手仍然缺乏Hammer，尽管其 popularity正在增长。我们提出了LeanHammer，这是第一个基于新型神经前提选择系统的通用Hammer，适用于依赖类型理论。与现有的Lean前提选择器不同，我们的方法动态适应用户特定的上下文，并结合符号证明搜索和重建，创建了一个实用的Hammer。通过全面的评估，我们展示我们的前提选择器使LeanHammer相较现有的前提选择器能够解决多21%的目标，并且在不同领域具有良好的泛化能力。我们的工作在神经检索和符号推理之间架起了桥梁，使形式验证对研究人员和实践者更加 accessible。 

---
# Ambiguity-Restrained Text-Video Representation Learning for Partially Relevant Video Retrieval 

**Title (ZH)**: 限制歧义的文本-视频表示学习以实现部分相关视频检索 

**Authors**: CH Cho, WJ Moon, W Jun, MS Jung, JP Heo  

**Link**: [PDF](https://arxiv.org/pdf/2506.07471)  

**Abstract**: Partially Relevant Video Retrieval~(PRVR) aims to retrieve a video where a specific segment is relevant to a given text query. Typical training processes of PRVR assume a one-to-one relationship where each text query is relevant to only one video. However, we point out the inherent ambiguity between text and video content based on their conceptual scope and propose a framework that incorporates this ambiguity into the model learning process. Specifically, we propose Ambiguity-Restrained representation Learning~(ARL) to address ambiguous text-video pairs. Initially, ARL detects ambiguous pairs based on two criteria: uncertainty and similarity. Uncertainty represents whether instances include commonly shared context across the dataset, while similarity indicates pair-wise semantic overlap. Then, with the detected ambiguous pairs, our ARL hierarchically learns the semantic relationship via multi-positive contrastive learning and dual triplet margin loss. Additionally, we delve into fine-grained relationships within the video instances. Unlike typical training at the text-video level, where pairwise information is provided, we address the inherent ambiguity within frames of the same untrimmed video, which often contains multiple contexts. This allows us to further enhance learning at the text-frame level. Lastly, we propose cross-model ambiguity detection to mitigate the error propagation that occurs when a single model is employed to detect ambiguous pairs for its training. With all components combined, our proposed method demonstrates its effectiveness in PRVR. 

**Abstract (ZH)**: 部分相关视频检索：含模糊性的视频表示学习框架 

---
# DeepVideo-R1: Video Reinforcement Fine-Tuning via Difficulty-aware Regressive GRPO 

**Title (ZH)**: DeepVideo-R1: 视频强化微调通过难度感知递归GRPO 

**Authors**: Jinyoung Park, Jeehye Na, Jinyoung Kim, Hyunwoo J. Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.07464)  

**Abstract**: Recent works have demonstrated the effectiveness of reinforcement learning (RL)-based post-training in enhancing the reasoning capabilities of large language models (LLMs). In particular, Group Relative Policy Optimization (GRPO) has shown impressive success by employing a PPO-style reinforcement algorithm with group-based normalized rewards. However, the application of GRPO to Video Large Language Models (Video LLMs) has been less studied. In this paper, we explore GRPO for video LLMs and identify two primary issues that impede its effective learning: (1) reliance on safeguards, and (2) the vanishing advantage problem. To mitigate these challenges, we propose DeepVideo-R1, a video large language model trained with our proposed Reg-GRPO (Regressive GRPO) and difficulty-aware data augmentation strategy. Reg-GRPO reformulates the GRPO objective as a regression task, directly predicting the advantage in GRPO. This design eliminates the need for safeguards like clipping and min functions, thereby facilitating more direct policy guidance by aligning the model with the advantage values. We also design the difficulty-aware data augmentation strategy that dynamically augments training samples at solvable difficulty levels, fostering diverse and informative reward signals. Our comprehensive experiments show that DeepVideo-R1 significantly improves video reasoning performance across multiple video reasoning benchmarks. 

**Abstract (ZH)**: Recent Works on Using Reinforcement Learning for Enhancing Reasoning Capabilities of Video Large Language Models: Addressing Challenges with DeepVideo-R1 

---
# CCI4.0: A Bilingual Pretraining Dataset for Enhancing Reasoning in Large Language Models 

**Title (ZH)**: CCI4.0: 一种用于增强大型语言模型推理能力的双语预训练数据集 

**Authors**: Guang Liu, Liangdong Wang, Jijie Li, Yang Yu, Yao Xu, Jiabei Chen, Yu Bai, Feng Liao, Yonghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.07463)  

**Abstract**: We introduce CCI4.0, a large-scale bilingual pre-training dataset engineered for superior data quality and diverse human-like reasoning trajectory. CCI4.0 occupies roughly $35$ TB of disk space and comprises two sub-datasets: CCI4.0-M2-Base and CCI4.0-M2-CoT. CCI4.0-M2-Base combines a $5.2$ TB carefully curated Chinese web corpus, a $22.5$ TB English subset from Nemotron-CC, and diverse sources from math, wiki, arxiv, and code. Although these data are mostly sourced from well-processed datasets, the quality standards of various domains are dynamic and require extensive expert experience and labor to process. So, we propose a novel pipeline justifying data quality mainly based on models through two-stage deduplication, multiclassifier quality scoring, and domain-aware fluency filtering. We extract $4.5$ billion pieces of CoT(Chain-of-Thought) templates, named CCI4.0-M2-CoT. Differing from the distillation of CoT from larger models, our proposed staged CoT extraction exemplifies diverse reasoning patterns and significantly decreases the possibility of hallucination. Empirical evaluations demonstrate that LLMs pre-trained in CCI4.0 benefit from cleaner, more reliable training signals, yielding consistent improvements in downstream tasks, especially in math and code reflection tasks. Our results underscore the critical role of rigorous data curation and human thinking templates in advancing LLM performance, shedding some light on automatically processing pretraining corpora. 

**Abstract (ZH)**: CCI4.0：一个大规模双语预训练数据集，旨在提供高质量数据和多样的类人类推理轨迹 

---
# KScope: A Framework for Characterizing the Knowledge Status of Language Models 

**Title (ZH)**: KScope: 一种语言模型知识状态表征框架 

**Authors**: Yuxin Xiao, Shan Chen, Jack Gallifant, Danielle Bitterman, Thomas Hartvigsen, Marzyeh Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07458)  

**Abstract**: Characterizing a large language model's (LLM's) knowledge of a given question is challenging. As a result, prior work has primarily examined LLM behavior under knowledge conflicts, where the model's internal parametric memory contradicts information in the external context. However, this does not fully reflect how well the model knows the answer to the question. In this paper, we first introduce a taxonomy of five knowledge statuses based on the consistency and correctness of LLM knowledge modes. We then propose KScope, a hierarchical framework of statistical tests that progressively refines hypotheses about knowledge modes and characterizes LLM knowledge into one of these five statuses. We apply KScope to nine LLMs across four datasets and systematically establish: (1) Supporting context narrows knowledge gaps across models. (2) Context features related to difficulty, relevance, and familiarity drive successful knowledge updates. (3) LLMs exhibit similar feature preferences when partially correct or conflicted, but diverge sharply when consistently wrong. (4) Context summarization constrained by our feature analysis, together with enhanced credibility, further improves update effectiveness and generalizes across LLMs. 

**Abstract (ZH)**: 探索大语言模型知识模式的五个知识状态，并提出KScope框架：一种逐步精炼知识模式假设并将其分类为五种状态的统计测试层次框架 

---
# Language-Grounded Hierarchical Planning and Execution with Multi-Robot 3D Scene Graphs 

**Title (ZH)**: 基于语言的多层次规划与执行：多机器人3D场景图 

**Authors**: Jared Strader, Aaron Ray, Jacob Arkin, Mason B. Peterson, Yun Chang, Nathan Hughes, Christopher Bradley, Yi Xuan Jia, Carlos Nieto-Granda, Rajat Talak, Chuchu Fan, Luca Carlone, Jonathan P. How, Nicholas Roy  

**Link**: [PDF](https://arxiv.org/pdf/2506.07454)  

**Abstract**: In this paper, we introduce a multi-robot system that integrates mapping, localization, and task and motion planning (TAMP) enabled by 3D scene graphs to execute complex instructions expressed in natural language. Our system builds a shared 3D scene graph incorporating an open-set object-based map, which is leveraged for multi-robot 3D scene graph fusion. This representation supports real-time, view-invariant relocalization (via the object-based map) and planning (via the 3D scene graph), allowing a team of robots to reason about their surroundings and execute complex tasks. Additionally, we introduce a planning approach that translates operator intent into Planning Domain Definition Language (PDDL) goals using a Large Language Model (LLM) by leveraging context from the shared 3D scene graph and robot capabilities. We provide an experimental assessment of the performance of our system on real-world tasks in large-scale, outdoor environments. 

**Abstract (ZH)**: 基于3D场景图的多机器人系统：结合映射、定位和自然语言表达复杂指令的Task and Motion Planning (TAMP) 

---
# When Style Breaks Safety: Defending Language Models Against Superficial Style Alignment 

**Title (ZH)**: 当风格破坏安全：防御语言模型的表面上的风格对齐 

**Authors**: Yuxin Xiao, Sana Tonekaboni, Walter Gerych, Vinith Suriyakumar, Marzyeh Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07452)  

**Abstract**: Large language models (LLMs) can be prompted with specific styles (e.g., formatting responses as lists), including in jailbreak queries. Although these style patterns are semantically unrelated to the malicious intents behind jailbreak queries, their safety impact remains unclear. In this work, we seek to understand whether style patterns compromise LLM safety, how superficial style alignment increases model vulnerability, and how best to mitigate these risks during alignment. We evaluate 32 LLMs across seven jailbreak benchmarks, and find that malicious queries with style patterns inflate the attack success rate (ASR) for nearly all models. Notably, ASR inflation correlates with both the length of style patterns and the relative attention an LLM exhibits on them. We then investigate superficial style alignment, and find that fine-tuning with specific styles makes LLMs more vulnerable to jailbreaks of those same styles. Finally, we propose SafeStyle, a defense strategy that incorporates a small amount of safety training data augmented to match the distribution of style patterns in the fine-tuning data. Across three LLMs and five fine-tuning style settings, SafeStyle consistently outperforms baselines in maintaining LLM safety. 

**Abstract (ZH)**: 大型语言模型的安全性受特定样式的影响研究：风格模式是否损害LLM安全？表面风格对齐如何增加模型漏洞以及如何缓解这些风险 

---
# LlamaRec-LKG-RAG: A Single-Pass, Learnable Knowledge Graph-RAG Framework for LLM-Based Ranking 

**Title (ZH)**: LlamaRec-LKG-RAG：一种基于单-pass、可学习的知识图谱-RAG框架的LLM排序模型 

**Authors**: Vahid Azizi, Fatemeh Koochaki  

**Link**: [PDF](https://arxiv.org/pdf/2506.07449)  

**Abstract**: Recent advances in Large Language Models (LLMs) have driven their adoption in recommender systems through Retrieval-Augmented Generation (RAG) frameworks. However, existing RAG approaches predominantly rely on flat, similarity-based retrieval that fails to leverage the rich relational structure inherent in user-item interactions. We introduce LlamaRec-LKG-RAG, a novel single-pass, end-to-end trainable framework that integrates personalized knowledge graph context into LLM-based recommendation ranking. Our approach extends the LlamaRec architecture by incorporating a lightweight user preference module that dynamically identifies salient relation paths within a heterogeneous knowledge graph constructed from user behavior and item metadata. These personalized subgraphs are seamlessly integrated into prompts for a fine-tuned Llama-2 model, enabling efficient and interpretable recommendations through a unified inference step. Comprehensive experiments on ML-100K and Amazon Beauty datasets demonstrate consistent and significant improvements over LlamaRec across key ranking metrics (MRR, NDCG, Recall). LlamaRec-LKG-RAG demonstrates the critical value of structured reasoning in LLM-based recommendations and establishes a foundation for scalable, knowledge-aware personalization in next-generation recommender systems. Code is available at~\href{this https URL}{repository}. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）通过检索增强生成（RAG）框架推动了其在推荐系统中的应用。然而，现有的RAG方法主要依赖于扁平的、基于相似性的检索，未能充分利用用户项交互中存在的丰富关系结构。我们引入了LlamaRec-LKG-RAG，这是一种新颖的一站式、端到端可训练框架，将个性化知识图景上下文集成到基于LLM的推荐排名中。我们的方法通过结合一个轻量级用户偏好模块扩展了LlamaRec架构，该模块能够在由用户行为和项元数据构建的异构知识图中动态识别显着的关系路径。这些个性化的子图无缝集成到微调的Llama-2模型的提示中，通过统一的推理步骤实现高效且可解释的推荐。在ML-100K和Amazon Beauty数据集上的全面实验表明，LlamaRec-LKG-RAG在关键排名指标（MRR、NDCG、召回率）上相对于LlamaRec实现了持续且显著的改进。LlamaRec-LKG-RAG展示了结构化推理在基于LLM的推荐中的关键价值，并为下一代推荐系统中可扩展、知识感知个性化奠定了基础。代码可在\href{this https URL}{仓库}获取。 

---
# Extending Epistemic Uncertainty Beyond Parameters Would Assist in Designing Reliable LLMs 

**Title (ZH)**: 超越参数扩展认识不确定性将有助于设计可靠的大型语言模型 

**Authors**: T. Duy Nguyen-Hien, Desi R. Ivanova, Yee Whye Teh, Wee Sun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.07448)  

**Abstract**: Although large language models (LLMs) are highly interactive and extendable, current approaches to ensure reliability in deployments remain mostly limited to rejecting outputs with high uncertainty in order to avoid misinformation. This conservative strategy reflects the current lack of tools to systematically distinguish and respond to different sources of uncertainty. In this paper, we advocate for the adoption of Bayesian Modeling of Experiments -- a framework that provides a coherent foundation to reason about uncertainty and clarify the reducibility of uncertainty -- for managing and proactively addressing uncertainty that arises in LLM deployments. This framework enables LLMs and their users to take contextually appropriate steps, such as requesting clarification, retrieving external information, or refining inputs. By supporting active resolution rather than passive avoidance, it opens the door to more reliable, transparent, and broadly applicable LLM systems, particularly in high-stakes, real-world settings. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）高度互动且可扩展，当前确保部署可靠性的方法主要限于拒绝具有高不确定性输出的做法，以避免误导信息。这一保守策略反映了当前缺乏系统区分和应对不同来源不确定性工具的现状。在本文中，我们倡导采用贝叶斯实验建模——这一框架提供了一种一致的基础来推理不确定性并明确不确定性可减少的途径——以管理和主动应对在LLM部署中出现的不确定性。该框架使LLMs及其用户能够采取语境适当的操作，如请求澄清、检索外部信息或细化输入。通过支持积极解决而非被动避免，它为开发更可靠、透明且适用范围更广的LLM系统打开了大门，特别是在高风险的实际应用场景中。 

---
# Prompt to Protection: A Comparative Study of Multimodal LLMs in Construction Hazard Recognition 

**Title (ZH)**: 提示到保护：多模态LLM在建筑危险识别中的比较研究 

**Authors**: Nishi Chaudhary, S M Jamil Uddin, Sathvik Sharath Chandra, Anto Ovid, Alex Albert  

**Link**: [PDF](https://arxiv.org/pdf/2506.07436)  

**Abstract**: The recent emergence of multimodal large language models (LLMs) has introduced new opportunities for improving visual hazard recognition on construction sites. Unlike traditional computer vision models that rely on domain-specific training and extensive datasets, modern LLMs can interpret and describe complex visual scenes using simple natural language prompts. However, despite growing interest in their applications, there has been limited investigation into how different LLMs perform in safety-critical visual tasks within the construction domain. To address this gap, this study conducts a comparative evaluation of five state-of-the-art LLMs: Claude-3 Opus, GPT-4.5, GPT-4o, GPT-o3, and Gemini 2.0 Pro, to assess their ability to identify potential hazards from real-world construction images. Each model was tested under three prompting strategies: zero-shot, few-shot, and chain-of-thought (CoT). Zero-shot prompting involved minimal instruction, few-shot incorporated basic safety context and a hazard source mnemonic, and CoT provided step-by-step reasoning examples to scaffold model thinking. Quantitative analysis was performed using precision, recall, and F1-score metrics across all conditions. Results reveal that prompting strategy significantly influenced performance, with CoT prompting consistently producing higher accuracy across models. Additionally, LLM performance varied under different conditions, with GPT-4.5 and GPT-o3 outperforming others in most settings. The findings also demonstrate the critical role of prompt design in enhancing the accuracy and consistency of multimodal LLMs for construction safety applications. This study offers actionable insights into the integration of prompt engineering and LLMs for practical hazard recognition, contributing to the development of more reliable AI-assisted safety systems. 

**Abstract (ZH)**: recent emergence of 多模态大规模语言模型 (LLMs) 为施工现场视觉隐患识别带来了新的机会。与依赖领域特定训练和大量数据的传统计算机视觉模型不同，现代LLMs可以通过简单的自然语言提示来解释和描述复杂的视觉场景。然而，尽管在这些应用方面兴趣日益增长，但对于安全关键的视觉任务，不同LLMs在建筑领域中的表现仍缺乏系统研究。为填补这一空白，本研究对比评估了五种最先进的LLMs：Claude-3 Opus、GPT-4.5、GPT-4o、GPT-o3和Gemini 2.0 Pro，以评估其识别施工现场真实图像潜在隐患的能力。每个模型在零样本、少数样本和步步推理（CoT）三种提示策略下进行了测试。零样本提示涉及最少的指令，少数样本结合了基础安全语境和隐患源记忆，步步推理提供了逐步推理示例以辅助模型思考。通过准确率、召回率和F1分数进行了定量分析。结果表明提示策略显著影响了性能，步步推理提示在所有模型中始终产生更高的准确性。此外，在不同条件下，LLM的表现也有所不同，GPT-4.5和GPT-o3在大多数设置中表现最佳。研究还展示了提示设计对提升多模态LLMs在建筑安全应用中准确性和一致性的关键作用。本研究提供了关于如何通过提示工程集成LLMs进行实际隐患识别的实用见解，为开发更可靠的AI辅助安全系统做出了贡献。 

---
# Fast Geometric Embedding for Node Influence Maximization 

**Title (ZH)**: 快速几何嵌入节点影响最大化 

**Authors**: Alexander Kolpakov, Igor Rivin  

**Link**: [PDF](https://arxiv.org/pdf/2506.07435)  

**Abstract**: Computing classical centrality measures such as betweenness and closeness is computationally expensive on large-scale graphs. In this work, we introduce an efficient force layout algorithm that embeds a graph into a low-dimensional space, where the radial distance from the origin serves as a proxy for various centrality measures. We evaluate our method on multiple graph families and demonstrate strong correlations with degree, PageRank, and paths-based centralities. As an application, it turns out that the proposed embedding allows to find high-influence nodes in a network, and provides a fast and scalable alternative to the standard greedy algorithm. 

**Abstract (ZH)**: 在大规模图上计算经典的中心性度量（如介数和接近中心性）非常耗费计算资源。本文介绍了一种高效的力量布局算法，将图嵌入到低维空间中，其中从原点的径向距离作为各种中心性度量的代理。我们在多个图家族上评估了该方法，并证明了其与度、PageRank和路径中心性的强相关性。作为一种应用，发现所提出的嵌入允许在网络中找到高影响力节点，并提供了一种比标准贪婪算法更快且更具扩展性的替代方案。 

---
# Well Begun is Half Done: Low-resource Preference Alignment by Weak-to-Strong Decoding 

**Title (ZH)**: 良好的开始是一场胜仗的一半：低资源偏好对齐通过弱到强解码实现 

**Authors**: Feifan Song, Shaohang Wei, Wen Luo, Yuxuan Fan, Tianyu Liu, Guoyin Wang, Houfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07434)  

**Abstract**: Large Language Models (LLMs) require alignment with human preferences to avoid generating offensive, false, or meaningless content. Recently, low-resource methods for LLM alignment have been popular, while still facing challenges in obtaining both high-quality and aligned content. Motivated by the observation that the difficulty of generating aligned responses is concentrated at the beginning of decoding, we propose a novel framework, Weak-to-Strong Decoding (WSD), to enhance the alignment ability of base models by the guidance of a small aligned model. The small model first drafts well-aligned beginnings, followed by the large base model to continue the rest, controlled by a well-designed auto-switch mechanism. We also collect a new dataset, GenerAlign, to fine-tune a small-sized Pilot-3B as the draft model, which effectively enhances different base models under the WSD framework to outperform all baseline methods, while avoiding degradation on downstream tasks, termed as the alignment tax. Extensive experiments are further conducted to examine the impact of different settings and time efficiency, as well as analyses on the intrinsic mechanisms of WSD in depth. 

**Abstract (ZH)**: Large Language Models (LLMs) 需要与人类偏好对齐以避免生成冒犯性、虚假或无意义的内容。近年来，低资源的LLM对齐方法很受欢迎，但仍面临获得高质量和对齐内容的挑战。受生成对齐响应难度集中在解码早期这一观察的启发，我们提出了一种新颖的框架，即弱到强解码（WSD），通过一个小的对齐模型的指导来增强基模型的对齐能力。小型模型首先草拟出良好的对齐开头，随后由大型基模型继续生成其余部分，由精心设计的自动切换机制控制。我们还收集了一个新的数据集GenerAlign，用于微调一个小规模的Pilot-3B作为草拟模型，该模型在WSD框架下有效地增强了不同的基模型，优于所有基线方法，同时避免了下游任务性能下降的问题，即对齐税。进一步进行了广泛的实验以检查不同设置的影响和时间效率，并深入分析了WSD内在机制。 

---
# FAMSeg: Fetal Femur and Cranial Ultrasound Segmentation Using Feature-Aware Attention and Mamba Enhancement 

**Title (ZH)**: FAMSeg: 基于特征意识注意力和Mamba增强的胎儿股骨和颅骨超声分割 

**Authors**: Jie He, Minglang Chen, Minying Lu, Bocheng Liang, Junming Wei, Guiyan Peng, Jiaxi Chen, Ying Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07431)  

**Abstract**: Accurate ultrasound image segmentation is a prerequisite for precise biometrics and accurate assessment. Relying on manual delineation introduces significant errors and is time-consuming. However, existing segmentation models are designed based on objects in natural scenes, making them difficult to adapt to ultrasound objects with high noise and high similarity. This is particularly evident in small object segmentation, where a pronounced jagged effect occurs. Therefore, this paper proposes a fetal femur and cranial ultrasound image segmentation model based on feature perception and Mamba enhancement to address these challenges. Specifically, a longitudinal and transverse independent viewpoint scanning convolution block and a feature perception module were designed to enhance the ability to capture local detail information and improve the fusion of contextual information. Combined with the Mamba-optimized residual structure, this design suppresses the interference of raw noise and enhances local multi-dimensional scanning. The system builds global information and local feature dependencies, and is trained with a combination of different optimizers to achieve the optimal solution. After extensive experimental validation, the FAMSeg network achieved the fastest loss reduction and the best segmentation performance across images of varying sizes and orientations. 

**Abstract (ZH)**: 准确的超声图像分割是精确生物特征测量和准确评估的前提。依靠手动勾绘引入了显著的误差且耗时。现有分割模型针对自然场景中的对象设计，难以适应高噪声和高相似度的超声对象，尤其是在小对象分割中出现了明显的锯齿状效果。因此，本文提出了一种基于特征感知和Mamba增强的胎儿股骨和颅骨超声图像分割模型，以应对这些挑战。具体地，该模型设计了纵向和横向独立视角扫描卷积块以及特征感知模块，增强局部细节信息的捕捉能力并改善上下文信息的融合。结合Mamba优化的残差结构，该设计抑制了原始噪声干扰并增强了局部多维扫描。系统构建全局信息和局部特征依赖关系，并通过多种优化器的组合进行训练，以实现最优解。经广泛实验验证，FAMSeg网络在不同尺寸和方向的图像中均实现了最快的损失下降和最佳的分割性能。 

---
# Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models 

**Title (ZH)**: 插件 fine-tuning：缩小小语言模型与大语言模型之间的差距 

**Authors**: Kyeonghyun Kim, Jinhee Jang, Juhwan Choi, Yoonji Lee, Kyohoon Jin, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.07424)  

**Abstract**: Large language models (LLMs) are renowned for their extensive linguistic knowledge and strong generalization capabilities, but their high computational demands make them unsuitable for resource-constrained environments. In contrast, small language models (SLMs) are computationally efficient but often lack the broad generalization capacity of LLMs. To bridge this gap, we propose PiFi, a novel framework that combines the strengths of both LLMs and SLMs to achieve high performance while maintaining efficiency. PiFi integrates a single frozen layer from an LLM into a SLM and fine-tunes the combined model for specific tasks, boosting performance without a significant increase in computational cost. We show that PiFi delivers consistent performance improvements across a range of natural language processing tasks, including both natural language understanding and generation. Moreover, our findings demonstrate PiFi's ability to effectively leverage LLM knowledge, enhancing generalization to unseen domains and facilitating the transfer of linguistic abilities. 

**Abstract (ZH)**: PiFi：结合大语言模型和小语言模型的优势以实现高效性能 

---
# Evidential Spectrum-Aware Contrastive Learning for OOD Detection in Dynamic Graphs 

**Title (ZH)**: 基于证据谱的动态图异常节点检测对比学习 

**Authors**: Nan Sun, Xixun Lin, Zhiheng Zhou, Yanmin Shang, Zhenlin Cheng, Yanan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2506.07417)  

**Abstract**: Recently, Out-of-distribution (OOD) detection in dynamic graphs, which aims to identify whether incoming data deviates from the distribution of the in-distribution (ID) training set, has garnered considerable attention in security-sensitive fields. Current OOD detection paradigms primarily focus on static graphs and confront two critical challenges: i) high bias and high variance caused by single-point estimation, which makes the predictions sensitive to randomness in the data; ii) score homogenization resulting from the lack of OOD training data, where the model only learns ID-specific patterns, resulting in overall low OOD scores and a narrow score gap between ID and OOD data. To tackle these issues, we first investigate OOD detection in dynamic graphs through the lens of Evidential Deep Learning (EDL). Specifically, we propose EviSEC, an innovative and effective OOD detector via Evidential Spectrum-awarE Contrastive Learning. We design an evidential neural network to redefine the output as the posterior Dirichlet distribution, explaining the randomness of inputs through the uncertainty of distribution, which is overlooked by single-point estimation. Moreover, spectrum-aware augmentation module generates OOD approximations to identify patterns with high OOD scores, thereby widening the score gap between ID and OOD data and mitigating score homogenization. Extensive experiments on real-world datasets demonstrate that EviSAC effectively detects OOD samples in dynamic graphs. 

**Abstract (ZH)**: 最近，动态图中的离分布（OOD）检测引起了安全敏感领域的广泛关注，该检测旨在识别输入数据是否偏离训练集内分布（ID）的数据分布。当前的OOD检测主要集中在静态图上，并面临两个关键挑战：一是单点估计导致的高度偏差和高度方差，使得预测结果对数据中的随机性敏感；二是由于缺乏OOD训练数据导致的分数同质化问题，模型只能学习特定于ID的模式，从而导致整体OOD分数较低且ID与OOD数据之间的分数差距狭窄。为解决这些问题，我们首先通过证据深度学习（EDL）的视角研究动态图中的OOD检测。具体而言，我们提出了一种名为EviSEC的创新且有效的OOD检测器，通过证据光谱感知对比学习（Evidential Spectrum-aware Contrastive Learning）实现。设计了证据神经网络，将输出重新定义为后验狄利克雷分布，通过分布的不确定性解释输入的随机性，这是单点估计忽略的部分。此外，光谱感知增强模块生成OOD近似值以识别高OOD分数的模式，从而扩大ID与OOD数据之间的分数差距并缓解分数同质化问题。在真实世界数据集上的广泛实验表明，EviSEC能够有效检测动态图中的OOD样本。 

---
# LiteVLM: A Low-Latency Vision-Language Model Inference Pipeline for Resource-Constrained Environments 

**Title (ZH)**: LiteVLM：一种面向资源受限环境的低延迟视觉-语言模型推理管道 

**Authors**: Jin Huang, Yuchao Jin, Le An, Josh Park  

**Link**: [PDF](https://arxiv.org/pdf/2506.07416)  

**Abstract**: This paper introduces an efficient Vision-Language Model (VLM) pipeline specifically optimized for deployment on embedded devices, such as those used in robotics and autonomous driving. The pipeline significantly reduces the computational overhead by jointly leveraging patch selection to filter irrelevant camera views, a token selection module to reduce input sequence length for the LLM, and speculative decoding to accelerate token generation. Evaluation on the NVIDIA DRIVE Thor platform for automonous driving application, our pipeline achieves $2.5\times$ end-to-end latency reduction without compromising task accuracy. The speed-up further increases to $3.2\times$ when applying FP8 post-training quantization. These results demonstrate our pipeline as a viable solution for enabling real-time VLM deployment in resource-constrained environments. 

**Abstract (ZH)**: 一种针对嵌入式设备优化的高效视觉-语言模型管道：在机器人和自动驾驶应用中的部署与加速 

---
# Fractional-order Jacobian Matrix Differentiation and Its Application in Artificial Neural Networks 

**Title (ZH)**: 分数阶雅可比矩阵微分及其在人工神经网络中的应用 

**Authors**: Xiaojun zhou, Chunna Zhao, Yaqun Huang, Chengli Zhou, Junjie Ye, Kemeng Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07408)  

**Abstract**: Fractional-order differentiation has many characteristics different from integer-order differentiation. These characteristics can be applied to the optimization algorithms of artificial neural networks to obtain better results. However, due to insufficient theoretical research, at present, there is no fractional-order matrix differentiation method that is perfectly compatible with automatic differentiation (Autograd) technology. Therefore, we propose a fractional-order matrix differentiation calculation method. This method is introduced by the definition of the integer-order Jacobian matrix. We denote it as fractional-order Jacobian matrix differentiation (${\bf{J}^\alpha }$). Through ${\bf{J}^\alpha }$, we can carry out the matrix-based fractional-order chain rule. Based on the Linear module and the fractional-order differentiation, we design the fractional-order Autograd technology to enable the use of fractional-order differentiation in hidden layers, thereby enhancing the practicality of fractional-order differentiation in deep learning. In the experiment, according to the PyTorch framework, we design fractional-order Linear (FLinear) and replace this http URL in the multilayer perceptron with FLinear. Through the qualitative analysis of the training set and validation set $Loss$, the quantitative analysis of the test set indicators, and the analysis of time consumption and GPU memory usage during model training, we verify the superior performance of ${\bf{J}^\alpha }$ and prove that it is an excellent fractional-order gradient descent method in the field of deep learning. 

**Abstract (ZH)**: 分数阶微分具有与整数阶微分许多不同的特性。这些特性可以应用于人工神经网络的优化算法，以获得更好的结果。然而，由于理论研究不足，目前尚未有完全兼容自动微分（Autograd）技术的分数阶矩阵微分方法。因此，我们提出了一种分数阶矩阵微分计算方法。该方法基于整数阶雅可比矩阵的定义引入，称为分数阶雅可比矩阵微分（${\bf{J}^\alpha }$）。通过${\bf{J}^\alpha }$，我们可以进行基于矩阵的分数阶链式法则。基于线性模块和分数阶微分，我们设计了分数阶Autograd技术，以使分数阶微分能够在隐藏层中使用，从而增强分数阶微分在深度学习中的实用性。在实验中，我们根据PyTorch框架设计了分数阶线性（FLinear），并将该http链接中的多层感知机层替换为FLinear。通过训练集和验证集$Loss$的定性分析、测试集指标的定量分析以及模型训练过程中时间消耗和GPU内存使用情况的分析，我们验证了${\bf{J}^\alpha }$的优越性能，并证明其在深度学习领域是优秀的分数阶梯度下降方法。 

---
# Anomaly Detection and Early Warning Mechanism for Intelligent Monitoring Systems in Multi-Cloud Environments Based on LLM 

**Title (ZH)**: 基于大语言模型的多云环境下智能监控系统异常检测与早期预警机制 

**Authors**: Yihong Jin, Ze Yang, Juntian Liu, Xinhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07407)  

**Abstract**: With the rapid development of multi-cloud environments, it is increasingly important to ensure the security and reliability of intelligent monitoring systems. In this paper, we propose an anomaly detection and early warning mechanism for intelligent monitoring system in multi-cloud environment based on Large-Scale Language Model (LLM). On the basis of the existing monitoring framework, the proposed model innovatively introduces a multi-level feature extraction method, which combines the natural language processing ability of LLM with traditional machine learning methods to enhance the accuracy of anomaly detection and improve the real-time response efficiency. By introducing the contextual understanding capabilities of LLMs, the model dynamically adapts to different cloud service providers and environments, so as to more effectively detect abnormal patterns and predict potential failures. Experimental results show that the proposed model is significantly better than the traditional anomaly detection system in terms of detection accuracy and latency, and significantly improves the resilience and active management ability of cloud infrastructure. 

**Abstract (ZH)**: 基于大型语言模型的多云环境智能监控系统异常检测与早期预警机制 

---
# InverseScope: Scalable Activation Inversion for Interpreting Large Language Models 

**Title (ZH)**: 逆向范围：大规模语言模型解释的可扩展激活反向检索 

**Authors**: Yifan Luo, Zhennan Zhou, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.07406)  

**Abstract**: Understanding the internal representations of large language models (LLMs) is a central challenge in interpretability research. Existing feature interpretability methods often rely on strong assumptions about the structure of representations that may not hold in practice. In this work, we introduce InverseScope, an assumption-light and scalable framework for interpreting neural activations via input inversion. Given a target activation, we define a distribution over inputs that generate similar activations and analyze this distribution to infer the encoded features. To address the inefficiency of sampling in high-dimensional spaces, we propose a novel conditional generation architecture that significantly improves sample efficiency compared to previous methods. We further introduce a quantitative evaluation protocol that tests interpretability hypotheses using feature consistency rate computed over the sampled inputs. InverseScope scales inversion-based interpretability methods to larger models and practical tasks, enabling systematic and quantitative analysis of internal representations in real-world LLMs. 

**Abstract (ZH)**: 理解大型语言模型（LLMs）的内部表示是可解释性研究中的一个核心挑战。现有特征可解释性方法往往依赖于在实践中可能不成立的表示结构假设。在本文中，我们引入了InverseScope，这是一种轻假设且可扩展的框架，通过输入 inversion 解释神经激活。给定一个目标激活，我们定义一个生成类似激活的输入分布，并分析该分布以推断编码的特征。为了解决高维空间采样的低效率问题，我们提出了一种新颖的条件生成架构，相比以前的方法显著提高了采样效率。此外，我们还引入了一种定量评估协议，用于通过计算采样输入的特征一致性率来测试可解释性假设。InverseScope 将基于逆向的可解释性方法扩展到更大的模型和实际任务，使得对现实世界 LLMs 的内部表示进行系统性和定量分析成为可能。 

---
# MedChat: A Multi-Agent Framework for Multimodal Diagnosis with Large Language Models 

**Title (ZH)**: MedChat: 多模态诊断的多Agent框架基于大型语言模型 

**Authors**: Philip Liu, Sparsh Bansal, Jimmy Dinh, Aditya Pawar, Ramani Satishkumar, Shail Desai, Neeraj Gupta, Xin Wang, Shu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07400)  

**Abstract**: The integration of deep learning-based glaucoma detection with large language models (LLMs) presents an automated strategy to mitigate ophthalmologist shortages and improve clinical reporting efficiency. However, applying general LLMs to medical imaging remains challenging due to hallucinations, limited interpretability, and insufficient domain-specific medical knowledge, which can potentially reduce clinical accuracy. Although recent approaches combining imaging models with LLM reasoning have improved reporting, they typically rely on a single generalist agent, restricting their capacity to emulate the diverse and complex reasoning found in multidisciplinary medical teams. To address these limitations, we propose MedChat, a multi-agent diagnostic framework and platform that combines specialized vision models with multiple role-specific LLM agents, all coordinated by a director agent. This design enhances reliability, reduces hallucination risk, and enables interactive diagnostic reporting through an interface tailored for clinical review and educational use. Code available at this https URL. 

**Abstract (ZH)**: 基于深度学习的青光眼检测与大型语言模型的集成：一种自动化策略，用于缓解眼科医生短缺并提高临床报告效率，但将通用大型语言模型应用于医学图像仍面临挑战，因为存在幻觉、解释性有限以及缺乏特定领域的医学知识，这可能会降低临床准确性。虽然最近结合图像模型和LLM推理的方法提高了报告质量，但它们通常依赖单一通用代理，限制了其模仿多学科医疗团队复杂推理的能力。为解决这些限制，我们提出MedChat，一种由专门视觉模型与多个角色特定的大规模语言模型代理组成的多代理诊断框架和平台，所有代理均由导演代理协调。该设计提高了可靠性，降低了幻觉风险，并通过一个适用于临床审查和教育使用的界面实现交互式诊断报告。代码见此URL。 

---
# MrM: Black-Box Membership Inference Attacks against Multimodal RAG Systems 

**Title (ZH)**: MrM：针对多模态RAG系统的黑盒成员推理攻击 

**Authors**: Peiru Yang, Jinhua Yin, Haoran Zheng, Xueying Bai, Huili Wang, Yufei Sun, Xintian Li, Shangguang Wang, Yongfeng Huang, Tao Qi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07399)  

**Abstract**: Multimodal retrieval-augmented generation (RAG) systems enhance large vision-language models by integrating cross-modal knowledge, enabling their increasing adoption across real-world multimodal tasks. These knowledge databases may contain sensitive information that requires privacy protection. However, multimodal RAG systems inherently grant external users indirect access to such data, making them potentially vulnerable to privacy attacks, particularly membership inference attacks (MIAs). % Existing MIA methods targeting RAG systems predominantly focus on the textual modality, while the visual modality remains relatively underexplored. To bridge this gap, we propose MrM, the first black-box MIA framework targeted at multimodal RAG systems. It utilizes a multi-object data perturbation framework constrained by counterfactual attacks, which can concurrently induce the RAG systems to retrieve the target data and generate information that leaks the membership information. Our method first employs an object-aware data perturbation method to constrain the perturbation to key semantics and ensure successful retrieval. Building on this, we design a counterfact-informed mask selection strategy to prioritize the most informative masked regions, aiming to eliminate the interference of model self-knowledge and amplify attack efficacy. Finally, we perform statistical membership inference by modeling query trials to extract features that reflect the reconstruction of masked semantics from response patterns. Experiments on two visual datasets and eight mainstream commercial visual-language models (e.g., GPT-4o, Gemini-2) demonstrate that MrM achieves consistently strong performance across both sample-level and set-level evaluations, and remains robust under adaptive defenses. 

**Abstract (ZH)**: 多模态检索增强生成（RAG）系统的黑盒会员推理攻击框架 

---
# From Static to Adaptive Defense: Federated Multi-Agent Deep Reinforcement Learning-Driven Moving Target Defense Against DoS Attacks in UAV Swarm Networks 

**Title (ZH)**: 从静态防御到适应性防御：无人机 swarm 网络中面向 DoS 攻击的联邦多代理深度强化学习驱动的动目标防御 

**Authors**: Yuyang Zhou, Guang Cheng, Kang Du, Zihan Chen, Tian Qin, Yuyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.07392)  

**Abstract**: The proliferation of unmanned aerial vehicle (UAV) swarms has enabled a wide range of mission-critical applications, but also exposes UAV networks to severe Denial-of-Service (DoS) threats due to their open wireless environment, dynamic topology, and resource constraints. Traditional static or centralized defense mechanisms are often inadequate for such dynamic and distributed scenarios. To address these challenges, we propose a novel federated multi-agent deep reinforcement learning (FMADRL)-driven moving target defense (MTD) framework for proactive and adaptive DoS mitigation in UAV swarm networks. Specifically, we design three lightweight and coordinated MTD mechanisms, including leader switching, route mutation, and frequency hopping, that leverage the inherent flexibility of UAV swarms to disrupt attacker efforts and enhance network resilience. The defense problem is formulated as a multi-agent partially observable Markov decision process (POMDP), capturing the distributed, resource-constrained, and uncertain nature of UAV swarms under attack. Each UAV is equipped with a local policy agent that autonomously selects MTD actions based on partial observations and local experiences. By employing a policy gradient-based FMADRL algorithm, UAVs collaboratively optimize their defense policies via reward-weighted aggregation, enabling distributed learning without sharing raw data and thus reducing communication overhead. Extensive simulations demonstrate that our approach significantly outperforms state-of-the-art baselines, achieving up to a 34.6% improvement in attack mitigation rate, a reduction in average recovery time of up to 94.6%, and decreases in energy consumption and defense cost by as much as 29.3% and 98.3%, respectively, while maintaining robust mission continuity under various DoS attack strategies. 

**Abstract (ZH)**: 基于联邦多智能体深度强化学习的无人机 swarm 网络动态目标防御框架 

---
# Shapley-Coop: Credit Assignment for Emergent Cooperation in Self-Interested LLM Agents 

**Title (ZH)**: Shapley-Coop：自我利益LLM代理中 emergent合作的归因分配 

**Authors**: Yun Hua, Haosheng Chen, Shiqin Wang, Wenhao Li, Xiangfeng Wang, Jun Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.07388)  

**Abstract**: Large Language Models (LLMs) show strong collaborative performance in multi-agent systems with predefined roles and workflows. However, in open-ended environments lacking coordination rules, agents tend to act in self-interested ways. The central challenge in achieving coordination lies in credit assignment -- fairly evaluating each agent's contribution and designing pricing mechanisms that align their heterogeneous goals. This problem is critical as LLMs increasingly participate in complex human-AI collaborations, where fair compensation and accountability rely on effective pricing mechanisms. Inspired by how human societies address similar coordination challenges (e.g., through temporary collaborations such as employment or subcontracting), we propose a cooperative workflow, Shapley-Coop. Shapley-Coop integrates Shapley Chain-of-Thought -- leveraging marginal contributions as a principled basis for pricing -- with structured negotiation protocols for effective price matching, enabling LLM agents to coordinate through rational task-time pricing and post-task reward redistribution. This approach aligns agent incentives, fosters cooperation, and maintains autonomy. We evaluate Shapley-Coop across two multi-agent games and a software engineering simulation, demonstrating that it consistently enhances LLM agent collaboration and facilitates equitable credit assignment. These results highlight the effectiveness of Shapley-Coop's pricing mechanisms in accurately reflecting individual contributions during task execution. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在具备预定义角色和流程的工作流多智能体系统中展现出强大的协作性能。但在缺乏协调规则的开放环境中，智能体倾向于以自我为中心的方式行动。实现协调的核心挑战在于责任归属——公平评估每个智能体的贡献并设计能够统一其异质目标的定价机制。随着LLMs越来越多地参与到复杂的人机协作中，有效的定价机制对于公平补偿和问责至关重要。受人类社会解决类似协调挑战方式（如通过临时合作，如雇佣或分包）的启发，我们提出了一种协作工作流Shapley-Coop。Shapley-Coop将Shapley Chain-of-Thought（通过边际贡献作为定价原则的基础）与结构化的谈判协议结合，以实现有效的价格匹配，使LLM智能体通过有理的任务时间和事后奖励重分配来实现协作。该方法对齐了智能体动力，促进了合作，并保持了自主权。我们在两个多智能体游戏和一个软件工程模拟中评估了Shapley-Coop，结果表明它能持续增强LLM智能体的协作，并推动公平的责任归属。这些结果突显了Shapley-Coop定价机制在任务执行期间准确反映个体贡献的有效性。 

---
# Adapter Naturally Serves as Decoupler for Cross-Domain Few-Shot Semantic Segmentation 

**Title (ZH)**: 适配器自然地作为跨域少样本语义分割的解藕器 

**Authors**: Jintao Tong, Ran Ma, Yixiong Zou, Guangyao Chen, Yuhua Li, Ruixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.07376)  

**Abstract**: Cross-domain few-shot segmentation (CD-FSS) is proposed to pre-train the model on a source-domain dataset with sufficient samples, and then transfer the model to target-domain datasets where only a few samples are available for efficient fine-tuning. There are majorly two challenges in this task: (1) the domain gap and (2) fine-tuning with scarce data. To solve these challenges, we revisit the adapter-based methods, and discover an intriguing insight not explored in previous works: the adapter not only helps the fine-tuning of downstream tasks but also naturally serves as a domain information decoupler. Then, we delve into this finding for an interpretation, and find the model's inherent structure could lead to a natural decoupling of domain information. Building upon this insight, we propose the Domain Feature Navigator (DFN), which is a structure-based decoupler instead of loss-based ones like current works, to capture domain-specific information, thereby directing the model's attention towards domain-agnostic knowledge. Moreover, to prevent the potential excessive overfitting of DFN during the source-domain training, we further design the SAM-SVN method to constrain DFN from learning sample-specific knowledge. On target domains, we freeze the model and fine-tune the DFN to learn target-specific knowledge specific. Extensive experiments demonstrate that our method surpasses the state-of-the-art method in CD-FSS significantly by 2.69% and 4.68% MIoU in 1-shot and 5-shot scenarios, respectively. 

**Abstract (ZH)**: 跨领域少样本分割（CD-FSS）预训练模型在源领域数据集上进行训练，然后将模型迁移到目标领域数据集，这些数据集只有少量样本可供高效微调。在这个任务中存在两大挑战：（1）领域差距和（2）基于稀缺数据的微调。为了解决这些挑战，我们重新审视了基于适配器的方法，并在先前工作中发现了有趣的新见解：适配器不仅有助于下游任务的微调，还自然地充当领域信息解耦器。基于这一发现，我们进一步探究其背后的原因，并发现模型固有的结构能够自然地解耦领域信息。利用这一新见解，我们提出了领域特征导航器（DFN），这是一种基于结构的解耦器，不同于当前基于损失的方法，能够捕获领域特定的信息，从而引导模型的关注点转向领域无关的知识。此外，为了防止在源领域训练过程中DFN可能过度拟合，我们进一步设计了SAM-SVN方法来限制DFN学习样本特定的知识。在目标领域中，我们冻结模型并微调DFN以学习特定于目标的知识。广泛实验表明，与当前最先进的方法相比，我们的方法在1-shot和5-shot场景中分别显著提高了2.69%和4.68%的MIoU。 

---
# HyColor: An Efficient Heuristic Algorithm for Graph Coloring 

**Title (ZH)**: HyColor: 一种高效的图着色启发式算法 

**Authors**: Enqiang Zhu, Yu Zhang, Haopeng Sun, Ziqi Wei, Witold Pedrycz, Chanjuan Liu, Jin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07373)  

**Abstract**: The graph coloring problem (GCP) is a classic combinatorial optimization problem that aims to find the minimum number of colors assigned to vertices of a graph such that no two adjacent vertices receive the same color. GCP has been extensively studied by researchers from various fields, including mathematics, computer science, and biological science. Due to the NP-hard nature, many heuristic algorithms have been proposed to solve GCP. However, existing GCP algorithms focus on either small hard graphs or large-scale sparse graphs (with up to 10^7 vertices). This paper presents an efficient hybrid heuristic algorithm for GCP, named HyColor, which excels in handling large-scale sparse graphs while achieving impressive results on small dense graphs. The efficiency of HyColor comes from the following three aspects: a local decision strategy to improve the lower bound on the chromatic number; a graph-reduction strategy to reduce the working graph; and a k-core and mixed degree-based greedy heuristic for efficiently coloring graphs. HyColor is evaluated against three state-of-the-art GCP algorithms across four benchmarks, comprising three large-scale sparse graph benchmarks and one small dense graph benchmark, totaling 209 instances. The results demonstrate that HyColor consistently outperforms existing heuristic algorithms in both solution accuracy and computational efficiency for the majority of instances. Notably, HyColor achieved the best solutions in 194 instances (over 93%), with 34 of these solutions significantly surpassing those of other algorithms. Furthermore, HyColor successfully determined the chromatic number and achieved optimal coloring in 128 instances. 

**Abstract (ZH)**: 图着色问题（GCP）是经典组合优化问题，旨在找到一种方法，将图中的顶点着色，使得每种颜色只出现在不相邻的顶点上，并且使用的颜色数量最少。GCP 已经得到来自数学、计算机科学和生物科学等多个领域的研究人员的广泛研究。由于其 NP 难性，已经提出了许多启发式算法来解决 GCP。然而，现有的 GCP 算法主要集中在小硬图或大规模稀疏图（最多包含 10^7 个顶点）。本文提出了一种高效的混合启发式算法 HyColor，该算法在处理大规模稀疏图的同时，也能在小密集图上取得令人印象深刻的结果。HyColor 的效率来自于以下三个方面：局部决策策略以提高色数的下界；图约简策略以减少工作图；以及基于 k-核心和混合度量的贪婪启发式方法以高效地对图着色。HyColor 在四种基准测试中的三种大规模稀疏图基准测试和一种小密集图基准测试共计 209 个实例与三种最先进的 GCP 算法进行了评估。结果表明，HyColor 在多数实例中在解的准确性和计算效率方面都优于现有启发式算法。值得注意的是，HyColor 在 194 个实例（超过 93%）中达到了最佳解，其中 34 个实例显著优于其他算法。此外，HyColor 成功确定了色数并在 128 个实例中实现了最优着色。 

---
# C3S3: Complementary Competition and Contrastive Selection for Semi-Supervised Medical Image Segmentation 

**Title (ZH)**: C3S3：互补竞争与对比选择在半监督医疗图像分割中的应用 

**Authors**: Jiaying He, Yitong Lin, Jiahe Chen, Honghui Xu, Jianwei Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.07368)  

**Abstract**: For the immanent challenge of insufficiently annotated samples in the medical field, semi-supervised medical image segmentation (SSMIS) offers a promising solution. Despite achieving impressive results in delineating primary target areas, most current methodologies struggle to precisely capture the subtle details of boundaries. This deficiency often leads to significant diagnostic inaccuracies. To tackle this issue, we introduce C3S3, a novel semi-supervised segmentation model that synergistically integrates complementary competition and contrastive selection. This design significantly sharpens boundary delineation and enhances overall precision. Specifically, we develop an $\textit{Outcome-Driven Contrastive Learning}$ module dedicated to refining boundary localization. Additionally, we incorporate a $\textit{Dynamic Complementary Competition}$ module that leverages two high-performing sub-networks to generate pseudo-labels, thereby further improving segmentation quality. The proposed C3S3 undergoes rigorous validation on two publicly accessible datasets, encompassing the practices of both MRI and CT scans. The results demonstrate that our method achieves superior performance compared to previous cutting-edge competitors. Especially, on the 95HD and ASD metrics, our approach achieves a notable improvement of at least $6\%$, highlighting the significant advancements. The code is available at this https URL. 

**Abstract (ZH)**: 面向医学领域标注样本不足的内在挑战，半监督医学图像分割（SSMIS）提供了有希望的解决方案。为了解决现有方法在精确捕捉边界细节方面的不足，我们提出了C3S3模型，该模型结合了互补竞争和对比选择机制，显著提升了边界界定和整体精确度。我们开发了基于结果驱动的对比学习模块，专门用于优化边界定位，并引入了动态互补竞争模块，利用两个高性能子网络生成伪标签，进一步提高分割质量。C3S3模型在包含MRI和CT扫描实践的两个公开数据集上进行了严格验证，结果表明，我们的方法在性能上优于以往的先进技术，特别是在95HD和ASD指标上取得了至少6%的显著提升。代码可在以下链接获取。 

---
# Multiple Object Stitching for Unsupervised Representation Learning 

**Title (ZH)**: 无监督表示学习中的多对象拼接 

**Authors**: Chengchao Shen, Dawei Liu, Jianxin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07364)  

**Abstract**: Contrastive learning for single object centric images has achieved remarkable progress on unsupervised representation, but suffering inferior performance on the widespread images with multiple objects. In this paper, we propose a simple but effective method, Multiple Object Stitching (MOS), to refine the unsupervised representation for multi-object images. Specifically, we construct the multi-object images by stitching the single object centric ones, where the objects in the synthesized multi-object images are predetermined. Hence, compared to the existing contrastive methods, our method provides additional object correspondences between multi-object images without human annotations. In this manner, our method pays more attention to the representations of each object in multi-object image, thus providing more detailed representations for complicated downstream tasks, such as object detection and semantic segmentation. Experimental results on ImageNet, CIFAR and COCO datasets demonstrate that our proposed method achieves the leading unsupervised representation performance on both single object centric images and multi-object ones. The source code is available at this https URL. 

**Abstract (ZH)**: 单物体中心图的对比学习已在无监督表示上取得了显著进展，但在广泛存在的多物体图像上表现较差。本文提出一种简单有效的多物体拼接（MOS）方法，以 refin无监督表示方法在多物体图像上的性能。具体地，通过将单物体中心图拼接生成多物体图像，其中合成的多物体图像中的物体预先确定。因此，与现有的对比学习方法相比，我们的方法为多物体图像之间提供了额外的对象对应关系，无需人工注释。通过这种方式，我们的方法更关注多物体图像中每个物体的表示，从而为复杂下游任务（如物体检测和语义分割）提供更详细的表示。实验结果表明，我们的方法在ImageNet、CIFAR和COCO数据集上的无监督表示性能均居首位。源代码可在以下链接获得。 

---
# Deepfake Technology Unveiled: The Commoditization of AI and Its Impact on Digital Trust 

**Title (ZH)**: Deepfake 技术揭秘：AI 的商品化及其对数字信任的影响 

**Authors**: Claudiu Popa, Rex Pallath, Liam Cunningham, Hewad Tahiri, Abiram Kesavarajah, Tao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07363)  

**Abstract**: Deepfake Technology Unveiled: The Commoditization of AI and Its Impact on Digital Trust. With the increasing accessibility of generative AI, tools for voice cloning, face-swapping, and synthetic media creation have advanced significantly, lowering both financial and technical barriers for their use. While these technologies present innovative opportunities, their rapid growth raises concerns about trust, privacy, and security. This white paper explores the implications of deepfake technology, analyzing its role in enabling fraud, misinformation, and the erosion of authenticity in multimedia. Using cost-effective, easy to use tools such as Runway, Rope, and ElevenLabs, we explore how realistic deepfakes can be created with limited resources, demonstrating the risks posed to individuals and organizations alike. By analyzing the technical and ethical challenges of deepfake mitigation and detection, we emphasize the urgent need for regulatory frameworks, public awareness, and collaborative efforts to maintain trust in digital media. 

**Abstract (ZH)**: Deepfake技术揭示：AI的商品化及其对数字信任的影响。随着生成式人工智能的日益普及，语音克隆、人脸识别替换和合成媒体创作工具显著进步，降低了这两种障碍。虽然这些技术展示了创新的机会，但它们的快速成长引发了对信任、隐私和安全的担忧。本白皮书探讨了Deepfake技术的影响，分析了其在欺诈、误导信息以及多媒体中真实性侵蚀方面的角色。通过使用Runway、Rope和ElevenLabs等经济实惠且易于使用的工具，我们探讨了如何使用有限资源创建逼真的Deepfake，展示了其对个人和组织带来的风险。通过分析Deepfake缓解和检测的技术和伦理挑战，我们强调 Urgent 需要制定监管框架、提高公众意识以及协作努力，以维护数字媒体的信任。 

---
# Lightweight Joint Audio-Visual Deepfake Detection via Single-Stream Multi-Modal Learning Framework 

**Title (ZH)**: 基于单流多模态学习框架的轻量级联合音视频深度伪造检测 

**Authors**: Kuiyuan Zhang, Wenjie Pei, Rushi Lan, Yifang Guo, Zhongyun Hua  

**Link**: [PDF](https://arxiv.org/pdf/2506.07358)  

**Abstract**: Deepfakes are AI-synthesized multimedia data that may be abused for spreading misinformation. Deepfake generation involves both visual and audio manipulation. To detect audio-visual deepfakes, previous studies commonly employ two relatively independent sub-models to learn audio and visual features, respectively, and fuse them subsequently for deepfake detection. However, this may underutilize the inherent correlations between audio and visual features. Moreover, utilizing two isolated feature learning sub-models can result in redundant neural layers, making the overall model inefficient and impractical for resource-constrained environments.
In this work, we design a lightweight network for audio-visual deepfake detection via a single-stream multi-modal learning framework. Specifically, we introduce a collaborative audio-visual learning block to efficiently integrate multi-modal information while learning the visual and audio features. By iteratively employing this block, our single-stream network achieves a continuous fusion of multi-modal features across its layers. Thus, our network efficiently captures visual and audio features without the need for excessive block stacking, resulting in a lightweight network design. Furthermore, we propose a multi-modal classification module that can boost the dependence of the visual and audio classifiers on modality content. It also enhances the whole resistance of the video classifier against the mismatches between audio and visual modalities. We conduct experiments on the DF-TIMIT, FakeAVCeleb, and DFDC benchmark datasets. Compared to state-of-the-art audio-visual joint detection methods, our method is significantly lightweight with only 0.48M parameters, yet it achieves superiority in both uni-modal and multi-modal deepfakes, as well as in unseen types of deepfakes. 

**Abstract (ZH)**: 轻量级单流多模态学习框架在音频-视觉深仿生成检测中的应用 

---
# SALT: A Lightweight Model Adaptation Method for Closed Split Computing Environments 

**Title (ZH)**: SALT：一种轻量级模型适配方法，用于封闭拆分计算环境 

**Authors**: Yuya Okada, Takayuki Nishio  

**Link**: [PDF](https://arxiv.org/pdf/2506.07355)  

**Abstract**: We propose SALT (Split-Adaptive Lightweight Tuning), a lightweight model adaptation framework for Split Computing under closed constraints, where the head and tail networks are proprietary and inaccessible to users. In such closed environments, conventional adaptation methods are infeasible since they require access to model parameters or architectures. SALT addresses this challenge by introducing a compact, trainable adapter on the client side to refine latent features from the head network, enabling user-specific adaptation without modifying the original models or increasing communication overhead. We evaluate SALT on user-specific classification tasks with CIFAR-10 and CIFAR-100, demonstrating improved accuracy with lower training latency compared to fine-tuning methods. Furthermore, SALT facilitates model adaptation for robust inference over lossy networks, a common challenge in edge-cloud environments. With minimal deployment overhead, SALT offers a practical solution for personalized inference in edge AI systems under strict system constraints. 

**Abstract (ZH)**: SALT: 分裂计算环境下具有产权限制的轻量级模型自适应框架 

---
# Distributed Risk-Sensitive Safety Filters for Uncertain Discrete-Time Systems 

**Title (ZH)**: 不确定离散时间系统中分布式的风险敏感安全性滤波器 

**Authors**: Armin Lederer, Erfaun Noorani, Andreas Krause  

**Link**: [PDF](https://arxiv.org/pdf/2506.07347)  

**Abstract**: Ensuring safety in multi-agent systems is a significant challenge, particularly in settings where centralized coordination is impractical. In this work, we propose a novel risk-sensitive safety filter for discrete-time multi-agent systems with uncertain dynamics that leverages control barrier functions (CBFs) defined through value functions. Our approach relies on centralized risk-sensitive safety conditions based on exponential risk operators to ensure robustness against model uncertainties. We introduce a distributed formulation of the safety filter by deriving two alternative strategies: one based on worst-case anticipation and another on proximity to a known safe policy. By allowing agents to switch between strategies, feasibility can be ensured. Through detailed numerical evaluations, we demonstrate the efficacy of our approach in maintaining safety without being overly conservative. 

**Abstract (ZH)**: 确保多自主系统安全是一个重要挑战，尤其是在集中协调不切实际的情况下。本文提出了一种新的风险敏感型安全滤波器，适用于具有不确定性动力学的离散时间多自主系统，并利用通过值函数定义的控制障碍函数（CBFs）。我们的方法基于基于指数风险操作符的集中风险敏感型安全条件，以确保对模型不确定性具有鲁棒性。通过推导两种替代策略来获得分布式安全滤波器的分布形式：一种基于最坏情况预测，另一种基于已知安全策略的接近性。通过允许代理在策略之间切换来确保可行性。通过详细的数值评估，我们展示了该方法在保持安全的同时不过于保守的有效性。 

---
# Real-Time Execution of Action Chunking Flow Policies 

**Title (ZH)**: 实时执行行动切片区块流策略 

**Authors**: Kevin Black, Manuel Y. Galliker, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2506.07339)  

**Abstract**: Modern AI systems, especially those interacting with the physical world, increasingly require real-time performance. However, the high latency of state-of-the-art generalist models, including recent vision-language action models (VLAs), poses a significant challenge. While action chunking has enabled temporal consistency in high-frequency control tasks, it does not fully address the latency problem, leading to pauses or out-of-distribution jerky movements at chunk boundaries. This paper presents a novel inference-time algorithm that enables smooth asynchronous execution of action chunking policies. Our method, real-time chunking (RTC), is applicable to any diffusion- or flow-based VLA out of the box with no re-training. It generates the next action chunk while executing the current one, "freezing" actions guaranteed to execute and "inpainting" the rest. To test RTC, we introduce a new benchmark of 12 highly dynamic tasks in the Kinetix simulator, as well as evaluate 6 challenging real-world bimanual manipulation tasks. Results demonstrate that RTC is fast, performant, and uniquely robust to inference delay, significantly improving task throughput and enabling high success rates in precise tasks $\unicode{x2013}$ such as lighting a match $\unicode{x2013}$ even in the presence of significant latency. See this https URL for videos. 

**Abstract (ZH)**: 现代AI系统，尤其是那些与物理世界交互的系统，越来越需要实时性能。然而，最先进的通用模型的高度延迟，包括最近的视觉-语言动作模型（VLAs），提出了一个重大的挑战。尽管动作切分使高频率控制任务具有时间一致性，但它并未完全解决延迟问题，导致在切分边界处出现暂停或不自然的运动。本文提出了一种新颖的推理时算法，可以实现动作切分策略的平滑异步执行。我们的方法实时切分（RTC）适用于任何基于扩散或流动的VLA，无需重新训练即可直接应用。它在执行当前动作的同时生成下一个动作切片，“冻结”已确保执行的动作，并“修复”其余部分。为了测试RTC，我们引入了Kinetix模拟器中的12个高度动态任务的新基准，并评估了6个具有挑战性的实际双臂操作任务。结果表明，RTC不仅快速且性能优越，而且对推理延迟具有独特 robust性，显著提高了任务吞吐量，并即使在存在显著延迟的情况下也能在精确任务中实现高成功率，例如点火。请见此链接获取视频。 

---
# Improving LLM Reasoning through Interpretable Role-Playing Steering 

**Title (ZH)**: 通过可解释的角色扮演引导提高LLM推理能力 

**Authors**: Anyi Wang, Dong Shu, Yifan Wang, Yunpu Ma, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2506.07335)  

**Abstract**: Role-playing has emerged as an effective technique for enhancing the reasoning capabilities of large language models (LLMs). However, existing methods primarily rely on prompt engineering, which often lacks stability and interpretability. In this paper, we introduce Sparse Autoencoder Role-Playing Steering (SRPS), a novel framework that identifies and manipulates internal model features associated with role-playing behavior. Our approach extracts latent representations from role-play prompts, selects the most relevant features based on activation patterns, and constructs a steering vector that can be injected into the model's residual stream with controllable intensity. Our method enables fine-grained control over role-specific behavior and offers insights into how role information influences internal model activations. Extensive experiments across various reasoning benchmarks and model sizes demonstrate consistent performance gains. Notably, in the zero-shot chain-of-thought (CoT) setting, the accuracy of Llama3.1-8B on CSQA improves from 31.86% to 39.80%, while Gemma2-9B on SVAMP increases from 37.50% to 45.10%. These results highlight the potential of SRPS to enhance reasoning ability in LLMs, providing better interpretability and stability compared to traditional prompt-based role-playing. 

**Abstract (ZH)**: Sparse Autoencoder Role-Playing Steering: Enhancing the Reasoning Capabilities of Large Language Models 

---
# JavelinGuard: Low-Cost Transformer Architectures for LLM Security 

**Title (ZH)**: JavelinGuard: 低成本变压器架构的LLM安全防护 

**Authors**: Yash Datta, Sharath Rajasekar  

**Link**: [PDF](https://arxiv.org/pdf/2506.07330)  

**Abstract**: We present JavelinGuard, a suite of low-cost, high-performance model architectures designed for detecting malicious intent in Large Language Model (LLM) interactions, optimized specifically for production deployment. Recent advances in transformer architectures, including compact BERT(Devlin et al. 2019) variants (e.g., ModernBERT (Warner et al. 2024)), allow us to build highly accurate classifiers with as few as approximately 400M parameters that achieve rapid inference speeds even on standard CPU hardware. We systematically explore five progressively sophisticated transformer-based architectures: Sharanga (baseline transformer classifier), Mahendra (enhanced attention-weighted pooling with deeper heads), Vaishnava and Ashwina (hybrid neural ensemble architectures), and Raudra (an advanced multi-task framework with specialized loss functions). Our models are rigorously benchmarked across nine diverse adversarial datasets, including popular sets like the NotInject series, BIPIA, Garak, ImprovedLLM, ToxicChat, WildGuard, and our newly introduced JavelinBench, specifically crafted to test generalization on challenging borderline and hard-negative cases. Additionally, we compare our architectures against leading open-source guardrail models as well as large decoder-only LLMs such as gpt-4o, demonstrating superior cost-performance trade-offs in terms of accuracy, and latency. Our findings reveal that while Raudra's multi-task design offers the most robust performance overall, each architecture presents unique trade-offs in speed, interpretability, and resource requirements, guiding practitioners in selecting the optimal balance of complexity and efficiency for real-world LLM security applications. 

**Abstract (ZH)**: JavelinGuard：一套专为生产部署优化的低成本高性能模型架构，用于检测大型语言模型交互中的恶意意图 

---
# Reward Model Interpretability via Optimal and Pessimal Tokens 

**Title (ZH)**: 通过最优和最劣标记提升奖励模型可解释性 

**Authors**: Brian Christian, Hannah Rose Kirk, Jessica A.F. Thompson, Christopher Summerfield, Tsvetomira Dumbalska  

**Link**: [PDF](https://arxiv.org/pdf/2506.07326)  

**Abstract**: Reward modeling has emerged as a crucial component in aligning large language models with human values. Significant attention has focused on using reward models as a means for fine-tuning generative models. However, the reward models themselves -- which directly encode human value judgments by turning prompt-response pairs into scalar rewards -- remain relatively understudied. We present a novel approach to reward model interpretability through exhaustive analysis of their responses across their entire vocabulary space. By examining how different reward models score every possible single-token response to value-laden prompts, we uncover several striking findings: (i) substantial heterogeneity between models trained on similar objectives, (ii) systematic asymmetries in how models encode high- vs low-scoring tokens, (iii) significant sensitivity to prompt framing that mirrors human cognitive biases, and (iv) overvaluation of more frequent tokens. We demonstrate these effects across ten recent open-source reward models of varying parameter counts and architectures. Our results challenge assumptions about the interchangeability of reward models, as well as their suitability as proxies of complex and context-dependent human values. We find that these models can encode concerning biases toward certain identity groups, which may emerge as unintended consequences of harmlessness training -- distortions that risk propagating through the downstream large language models now deployed to millions. 

**Abstract (ZH)**: 奖励模型已成为对齐大型语言模型与人类价值观的关键组件。尽管人们广泛关注使用奖励模型对生成模型进行微调，但直接将人类价值判断编码为提示-响应对的标量奖励的奖励模型本身仍相对研究不足。我们通过彻底分析其在整个词汇空间的响应，提出了一种新的奖励模型可解释性方法。通过对具有价值导向的提示的每一可能单词响应进行评分，我们发现了几个引人注目的发现：(i) 相似目标训练的模型之间存在显著异质性，(ii) 模型在编码高分词与低分词时存在系统不对称性，(iii) 对提示框架的显著敏感性，这种敏感性与人类认知偏差相呼应，以及(iv) 对更频繁出现的词赋予过高的价值。我们在十种不同参数量和架构的开源奖励模型中展示了这些效果。我们的研究结果挑战了奖励模型可互换性和作为复杂且上下文依赖的人类价值观代理的假设。我们发现这些模型可以编码对某些身份群体的令人担忧的偏见，这些偏见可能是无害训练的未预料后果，并可能通过现已部署给数百万用户的下游大型语言模型传播。 

---
# Speech Recognition on TV Series with Video-guided Post-Correction 

**Title (ZH)**: 电视连续剧中的语音识别与视频引导后修正 

**Authors**: Haoyuan Yang, Yue Zhang, Liqiang Jing  

**Link**: [PDF](https://arxiv.org/pdf/2506.07323)  

**Abstract**: Automatic Speech Recognition (ASR) has achieved remarkable success with deep learning, driving advancements in conversational artificial intelligence, media transcription, and assistive technologies. However, ASR systems still struggle in complex environments such as TV series, where overlapping speech, domain-specific terminology, and long-range contextual dependencies pose significant challenges to transcription accuracy. Existing multimodal approaches fail to correct ASR outputs with the rich temporal and contextual information available in video. To address this limitation, we propose a novel multimodal post-correction framework that refines ASR transcriptions by leveraging contextual cues extracted from video. Our framework consists of two stages: ASR Generation and Video-based Post-Correction, where the first stage produces the initial transcript and the second stage corrects errors using Video-based Contextual Information Extraction and Context-aware ASR Correction. We employ the Video-Large Multimodal Model (VLMM) to extract key contextual information using tailored prompts, which is then integrated with a Large Language Model (LLM) to refine the ASR output. We evaluate our method on a multimodal benchmark for TV series ASR and demonstrate its effectiveness in improving ASR performance by leveraging video-based context to enhance transcription accuracy in complex multimedia environments. 

**Abstract (ZH)**: 自动语音识别（ASR）借助深度学习取得了显著成功，推动了对话式人工智能、媒体转录和辅助技术的发展。然而，ASR系统在电视剧等复杂环境中仍面临挑战，如重叠语音、领域特定术语以及长范围上下文依赖性对转录准确性构成了重大挑战。现有跨模态方法无法利用视频中丰富的时序和上下文信息来纠正ASR输出。为解决这一局限，我们提出了一种新的跨模态后处理框架，通过利用从视频中提取的上下文线索来精炼ASR转录。该框架包括两个阶段：ASR生成和视频驱动的后处理，第一阶段生成初始转录，第二阶段使用基于视频的上下文信息提取和上下文感知ASR纠正来修正错误。我们使用定制提示的视频大型跨模态模型（VLMM）提取关键上下文信息，并将其与大型语言模型集成，以精炼ASR输出。我们在电视剧ASR的跨模态基准上评估了我们的方法，并通过利用视频中的上下文增强复杂多媒体环境中的转录准确性，展示了其有效性。 

---
# Towards Competent AI for Fundamental Analysis in Finance: A Benchmark Dataset and Evaluation 

**Title (ZH)**: 面向金融基本面分析的 competent AI：一个基准数据集及评估 

**Authors**: Zonghan Wu, Junlin Wang, Congyuan Zou, Chenhan Wang, Yilei Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.07315)  

**Abstract**: Generative AI, particularly large language models (LLMs), is beginning to transform the financial industry by automating tasks and helping to make sense of complex financial information. One especially promising use case is the automatic creation of fundamental analysis reports, which are essential for making informed investment decisions, evaluating credit risks, guiding corporate mergers, etc. While LLMs attempt to generate these reports from a single prompt, the risks of inaccuracy are significant. Poor analysis can lead to misguided investments, regulatory issues, and loss of trust. Existing financial benchmarks mainly evaluate how well LLMs answer financial questions but do not reflect performance in real-world tasks like generating financial analysis reports. In this paper, we propose FinAR-Bench, a solid benchmark dataset focusing on financial statement analysis, a core competence of fundamental analysis. To make the evaluation more precise and reliable, we break this task into three measurable steps: extracting key information, calculating financial indicators, and applying logical reasoning. This structured approach allows us to objectively assess how well LLMs perform each step of the process. Our findings offer a clear understanding of LLMs current strengths and limitations in fundamental analysis and provide a more practical way to benchmark their performance in real-world financial settings. 

**Abstract (ZH)**: 生成式人工智能，特别是大型语言模型（LLMs），正开始通过自动化任务和帮助理解和解释复杂的金融信息来转型金融行业。一个特别有前景的应用场景是自动化生成基础分析报告，这对于进行知情投资决策、评估信贷风险、指导企业合并等至关重要。尽管LLMs试图通过单一提示生成这些报告，但准确性风险仍然很大。不良分析可能导致误导性投资、监管问题和信任丧失。现有的金融基准主要评估LLMs回答金融问题的能力，但未能反映其在生成财务分析报告等实际任务中的表现。在本文中，我们提出FinAR-Bench，这是一个专注于财务报表分析的坚实基准数据集，这是基础分析的核心能力。为了使评估更加精确和可靠，我们将此任务分解为三个可测量的步骤：提取关键信息、计算财务指标和应用逻辑推理。这种结构化的方法使我们能够客观评估LLMs在每个步骤中的表现。我们的发现为理解当前LLMs在基础分析中的优势和限制提供了明确的见解，并提供了一种更实际的方法来衡量其在实际金融场景中的表现。 

---
# Generative Modeling of Networked Time-Series via Transformer Architectures 

**Title (ZH)**: 基于变压器架构的网络时间序列生成建模 

**Authors**: Yusuf Elnady  

**Link**: [PDF](https://arxiv.org/pdf/2506.07312)  

**Abstract**: Many security and network applications require having large datasets to train the machine learning models. Limited data access is a well-known problem in the security domain. Recent studies have shown the potential of Transformer models to enlarge the size of data by synthesizing new samples, but the synthesized samples don't improve the models over the real data. To address this issue, we design an efficient transformer-based model as a generative framework to generate time-series data, that can be used to boost the performance of existing and new ML workflows. Our new transformer model achieves the SOTA results. We style our model to be generalizable and work across different datasets, and produce high-quality samples. 

**Abstract (ZH)**: 许多安全和网络应用需要大量数据集来训练机器学习模型。在安全领域，有限的数据访问是一个已知问题。近期研究表明，变换器模型有扩大数据规模的潜力，可以通过合成新样本实现，但合成样本并未提高模型性能。为解决这一问题，我们设计了一个高效的基于变换器的生成模型框架，用于生成时间序列数据，以增强现有和新机器学习流程的性能。我们的新型变换器模型达到了当前最佳结果。我们设计该模型具有通用性，可在不同数据集上工作，并生成高质量的样本。 

---
# Paged Attention Meets FlexAttention: Unlocking Long-Context Efficiency in Deployed Inference 

**Title (ZH)**: 分页注意力邂逅FlexAttention：解锁部署推断中的长上下文效率 

**Authors**: Thomas Joshi, Herman Saini, Neil Dhillon, Antoni Viros i Martin, Kaoutar El Maghraoui  

**Link**: [PDF](https://arxiv.org/pdf/2506.07311)  

**Abstract**: Large Language Models (LLMs) encounter severe memory inefficiencies during long-context inference due to conventional handling of key-value (KV) caches. In this work, we introduce a novel integration of PagedAttention with PyTorch's FlexAttention, addressing internal fragmentation and inefficiencies associated with monolithic KV cache allocations. Implemented within IBM's Foundation Model Stack (FMS), our fused attention kernel efficiently gathers scattered KV data. Our benchmarks on an NVIDIA L4 GPU (24GB) demonstrate significantly reduced inference latency, growing only linearly (~2x) with sequence length from 128 to 2048 tokens when utilizing a global KV cache, compared to exponential latency increases without caching. While peak memory usage remains largely unchanged for single-step evaluations (dominated by model weights and activations), paged attention causes minimal incremental memory usage, observable only at sequence lengths exceeding 2048 tokens due to its power-of-two cache allocations. We open-source the full implementation and discuss its implications for future long-context model deployment. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在长上下文推理过程中由于常规的关键值（KV）缓存处理遇到严重的内存 inefficiencies。在本文中，我们引入了一种将 PagedAttention 与 PyTorch 的 FlexAttention 新颖集成的方法，以解决与大型 KV 缓存分配相关的内部碎片化和 inefficiencies。该融合的注意力内核在 IBM 的基础模型堆栈（FMS）中实现，有效收集分散的 KV 数据。我们在 NVIDIA L4 GPU（24GB）上的基准测试表明，使用全局 KV 缓存时，推理延迟显著降低，从 128 个到 2048 个标记的增长仅线性（~2 倍）增加，而没有缓存时延迟呈指数级增加。尽管单步评估的最大内存使用量保持基本不变（主要由模型权重和激活量决定），分页注意力引起的增量内存使用量几乎可以忽略不计，仅在序列长度超过 2048 个标记时才可观察到，这归因于其基于 2 的幂次的缓存分配。我们开源了完整实现，并探讨其对未来的长上下文模型部署的影响。 

---
# Pre-trained Large Language Models Learn Hidden Markov Models In-context 

**Title (ZH)**: 预训练大型语言模型学习上下文中的隐马尔可夫模型 

**Authors**: Yijia Dai, Zhaolin Gao, Yahya Satter, Sarah Dean, Jennifer J. Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.07298)  

**Abstract**: Hidden Markov Models (HMMs) are foundational tools for modeling sequential data with latent Markovian structure, yet fitting them to real-world data remains computationally challenging. In this work, we show that pre-trained large language models (LLMs) can effectively model data generated by HMMs via in-context learning (ICL)$\unicode{x2013}$their ability to infer patterns from examples within a prompt. On a diverse set of synthetic HMMs, LLMs achieve predictive accuracy approaching the theoretical optimum. We uncover novel scaling trends influenced by HMM properties, and offer theoretical conjectures for these empirical observations. We also provide practical guidelines for scientists on using ICL as a diagnostic tool for complex data. On real-world animal decision-making tasks, ICL achieves competitive performance with models designed by human experts. To our knowledge, this is the first demonstration that ICL can learn and predict HMM-generated sequences$\unicode{x2013}$an advance that deepens our understanding of in-context learning in LLMs and establishes its potential as a powerful tool for uncovering hidden structure in complex scientific data. 

**Abstract (ZH)**: 预训练大语言模型通过上下文学习有效建模隐马尔科夫模型生成的数据：探索新的扩展趋势并提供实际指导 

---
# HotelMatch-LLM: Joint Multi-Task Training of Small and Large Language Models for Efficient Multimodal Hotel Retrieval 

**Title (ZH)**: HotelMatch-LLM：小型和大型语言模型的联合多任务训练以实现高效的多模态酒店检索 

**Authors**: Arian Askari, Emmanouil Stergiadis, Ilya Gusev, Moran Beladev  

**Link**: [PDF](https://arxiv.org/pdf/2506.07296)  

**Abstract**: We present HotelMatch-LLM, a multimodal dense retrieval model for the travel domain that enables natural language property search, addressing the limitations of traditional travel search engines which require users to start with a destination and editing search parameters. HotelMatch-LLM features three key innovations: (1) Domain-specific multi-task optimization with three novel retrieval, visual, and language modeling objectives; (2) Asymmetrical dense retrieval architecture combining a small language model (SLM) for efficient online query processing and a large language model (LLM) for embedding hotel data; and (3) Extensive image processing to handle all property image galleries. Experiments on four diverse test sets show HotelMatch-LLM significantly outperforms state-of-the-art models, including VISTA and MARVEL. Specifically, on the test set -- main query type -- we achieve 0.681 for HotelMatch-LLM compared to 0.603 for the most effective baseline, MARVEL. Our analysis highlights the impact of our multi-task optimization, the generalizability of HotelMatch-LLM across LLM architectures, and its scalability for processing large image galleries. 

**Abstract (ZH)**: HotelMatch-LLM：旅行领域多模态密集检索模型及其创新 

---
# Secondary Stakeholders in AI: Fighting for, Brokering, and Navigating Agency 

**Title (ZH)**: 人工智能领域中的次级利益相关者：争取、斡旋和导航代理权 

**Authors**: Leah Hope Ajmani, Nuredin Ali Abdelkadir, Stevie Chancellor  

**Link**: [PDF](https://arxiv.org/pdf/2506.07281)  

**Abstract**: As AI technologies become more human-facing, there have been numerous calls to adapt participatory approaches to AI development -- spurring the idea of participatory AI. However, these calls often focus only on primary stakeholders, such as end-users, and not secondary stakeholders. This paper seeks to translate the ideals of participatory AI to a broader population of secondary AI stakeholders through semi-structured interviews. We theorize that meaningful participation involves three participatory ideals: (1) informedness, (2) consent, and (3) agency. We also explore how secondary stakeholders realize these ideals by traversing a complicated problem space. Like walking up the rungs of a ladder, these ideals build on one another. We introduce three stakeholder archetypes: the reluctant data contributor, the unsupported activist, and the well-intentioned practitioner, who must navigate systemic barriers to achieving agentic AI relationships. We envision an AI future where secondary stakeholders are able to meaningfully participate with the AI systems they influence and are influenced by. 

**Abstract (ZH)**: 随着AI技术更加面向人类，人们呼吁适应参与式AI开发的方法——催生了参与式AI的理念。然而，这些呼吁往往仅关注主要利益相关者，如最终用户，而忽视了次要利益相关者。本文旨在通过半结构化访谈，将参与式AI的理念扩展到更广泛范围的次要AI利益相关者群体。我们认为有意义的参与涉及三个参与式理念：（1）知情权，（2）同意权，（3）自主权。我们还探讨了次要利益相关者如何通过穿越复杂的问题空间来实现这些理念。这些理念层层递进，犹如攀登梯子的每一级。我们介绍了三种利益相关者原型：不愿意的数据贡献者、得不到支持的活动家以及致力于实现自主型人机关系的有良好意愿的从业者。我们构想一个未来的AI世界，在这个世界中，次要利益相关者能够与其影响和被影响的AI系统进行有意义的互动。 

---
# From Generation to Generalization: Emergent Few-Shot Learning in Video Diffusion Models 

**Title (ZH)**: 从生成到泛化：视频扩散模型中的 emergent Few-Shot 学习 

**Authors**: Pablo Acuaviva, Aram Davtyan, Mariam Hassan, Sebastian Stapf, Ahmad Rahimi, Alexandre Alahi, Paolo Favaro  

**Link**: [PDF](https://arxiv.org/pdf/2506.07280)  

**Abstract**: Video Diffusion Models (VDMs) have emerged as powerful generative tools, capable of synthesizing high-quality spatiotemporal content. Yet, their potential goes far beyond mere video generation. We argue that the training dynamics of VDMs, driven by the need to model coherent sequences, naturally pushes them to internalize structured representations and an implicit understanding of the visual world. To probe the extent of this internal knowledge, we introduce a few-shot fine-tuning framework that repurposes VDMs for new tasks using only a handful of examples. Our method transforms each task into a visual transition, enabling the training of LoRA weights on short input-output sequences without altering the generative interface of a frozen VDM. Despite minimal supervision, the model exhibits strong generalization across diverse tasks, from low-level vision (for example, segmentation and pose estimation) to high-level reasoning (for example, on ARC-AGI). These results reframe VDMs as more than generative engines. They are adaptable visual learners with the potential to serve as the backbone for future foundation models in vision. 

**Abstract (ZH)**: 视频扩散模型（VDMs）已发展成为强大的生成工具，能够合成高质量的时空内容。然而，其潜力远不止视频生成。我们主张，为了建模连贯的序列，VDMs在训练过程中自然地内化了结构化的表示和对视觉世界的隐式理解。为了探索这种内部知识的程度，我们提出了一种少样本微调框架，仅使用少量示例即可重新利用VDMs进行新的任务。我们的方法将每个任务转换为视觉过渡，使我们能够在冻结的VDM的生成界面下，通过对短输入-输出序列训练LoRA权重，而无需修改生成接口。尽管监督较少，该模型在多种任务上表现出了强大的泛化能力，从低级视觉任务（例如，分割和姿态估计）到高级推理任务（例如，ARC-AGI）。这些结果重新定义了VDMs不仅仅作为生成引擎。它们是可适应的视觉学习者，具有成为未来视觉基础模型核心的潜力。 

---
# Tokenized Bandit for LLM Decoding and Alignment 

**Title (ZH)**: _tokenized bandit 算法在大规模语言模型解码与对齐中的应用_ 

**Authors**: Suho Shin, Chenghao Yang, Haifeng Xu, Mohammad T. Hajiaghayi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07276)  

**Abstract**: We introduce the tokenized linear bandit (TLB) and multi-armed bandit (TMAB), variants of linear and stochastic multi-armed bandit problems inspired by LLM decoding and alignment. In these problems, at each round $t \in [T]$, a user submits a query (context), and the decision maker (DM) sequentially selects a token irrevocably from a token set. Once the sequence is complete, the DM observes a random utility from the user, whose expectation is presented by a sequence function mapping the chosen token sequence to a nonnegative real value that depends on the query.
In both problems, we first show that learning is impossible without any structure on the sequence function. We introduce a natural assumption, diminishing distance with more commons (DDMC), and propose algorithms with regret $\tilde{O}(L\sqrt{T})$ and $\tilde{O}(L\sqrt{T^{2/3}})$ for TLB and TMAB, respectively. As a side product, we obtain an (almost) optimality of the greedy decoding for LLM decoding algorithm under DDMC, which justifies the unresaonable effectiveness of greedy decoding in several tasks. This also has an immediate application to decoding-time LLM alignment, when the misaligned utility can be represented as the frozen LLM's utility and a linearly realizable latent function. We finally validate our algorithm's performance empirically as well as verify our assumptions using synthetic and real-world datasets. 

**Abstract (ZH)**: 我们介绍了一种标记线性 bandit (TLB) 和标记多臂 bandit (TMAB)，这两种问题是在大型语言模型解码和对齐启发下对线性多臂 bandit 问题和随机多臂 bandit 问题的变种。在这些问题中，每轮 $t \in [T]$，用户提交一个查询（上下文），决策者（DM）序列且不可撤销地从标记集合中选择一个标记。当序列完成后，DM 观察到一个随机效用，其期望由一个序列函数映射到非负实数值，该值依赖于查询。

在这两个问题中，我们首先证明了在序列函数没有任何结构的情况下无法进行学习。我们引入了一个自然假设，即随着更常见元素的出现而逐渐减小的距离（DDMC），并提出了分别针对 TLB 和 TMAB 的 regrets 为 $\tilde{O}(L\sqrt{T})$ 和 $\tilde{O}(L\sqrt{T^{2/3}})$ 的算法。作为副产品，我们证明了在 DDMC 下贪心解码算法几乎最优，这解释了在多种任务中贪心解码的不可思议的有效性。这在将对齐错误的效用表示为冻结的大型语言模型的效用和一个线性可实现的潜在函数时，可以立即应用于大型语言模型解码时间的对齐上。最后，我们通过实证验证了算法的性能，并使用合成数据集和真实世界数据集验证了我们的假设。 

---
# Parsing the Switch: LLM-Based UD Annotation for Complex Code-Switched and Low-Resource Languages 

**Title (ZH)**: 解析转换：基于LLM的复杂代码转换和低资源语言UD注释 

**Authors**: Olga Kellert, Nemika Tyagi, Muhammad Imran, Nelvin Licona-Guevara, Carlos Gómez-Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2506.07274)  

**Abstract**: Code-switching presents a complex challenge for syntactic analysis, especially in low-resource language settings where annotated data is scarce. While recent work has explored the use of large language models (LLMs) for sequence-level tagging, few approaches systematically investigate how well these models capture syntactic structure in code-switched contexts. Moreover, existing parsers trained on monolingual treebanks often fail to generalize to multilingual and mixed-language input. To address this gap, we introduce the BiLingua Parser, an LLM-based annotation pipeline designed to produce Universal Dependencies (UD) annotations for code-switched text. First, we develop a prompt-based framework for Spanish-English and Spanish-Guaraní data, combining few-shot LLM prompting with expert review. Second, we release two annotated datasets, including the first Spanish-Guaraní UD-parsed corpus. Third, we conduct a detailed syntactic analysis of switch points across language pairs and communicative contexts. Experimental results show that BiLingua Parser achieves up to 95.29% LAS after expert revision, significantly outperforming prior baselines and multilingual parsers. These results show that LLMs, when carefully guided, can serve as practical tools for bootstrapping syntactic resources in under-resourced, code-switched environments. Data and source code are available at this https URL 

**Abstract (ZH)**: 代码转换在句法学分析中呈现出了复杂挑战，特别是在标注数据稀缺的低资源语言环境中。尽管最近的工作探索了大规模语言模型（LLMs）在序列级别标注的应用，但鲜有研究系统性地考察这些模型在代码转换上下文中的句法学结构捕获能力。此外，现有的仅针对单一语言的解析器往往无法泛化到多语言和混合语言输入中。为弥补这一差距，我们引入了BiLingua解析器，这是一种基于大规模语言模型的注释流水线，旨在为代码转换文本生成通用依存关系（Universal Dependencies, UD）注释。首先，我们开发了一种基于提示的框架，用于西班牙语-英语和西班牙语-瓜拉尼的数据，结合了少样本大规模语言模型提示与专家审核。其次，我们发布了两个注释数据集，包括首个西班牙语-瓜拉尼的UD解析语料库。第三，我们在语言对和交际上下文中进行了详细的句法分析，以研究转换点。实验结果表明，经过专家修订后，BiLingua解析器的准确句法结构评分（LAS）高达95.29%，显著优于先前的基线和多语言解析器。这些结果表明，在适当引导下，大规模语言模型可以作为实用工具，用于低资源、代码转换环境中句法学资源的建设。相关数据和源代码可访问此网址。 

---
# SDE-SQL: Enhancing Text-to-SQL Generation in Large Language Models via Self-Driven Exploration with SQL Probes 

**Title (ZH)**: SDE-SQL：通过基于SQL探测的自我驱动探索增强大型语言模型的文本到SQL生成能力 

**Authors**: Wenxuan Xie, Yaxun Dai, Wenhao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07245)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly improved performance on the Text-to-SQL task. However, prior approaches typically rely on static, pre-processed database information provided at inference time, which limits the model's ability to fully understand the database contents. Without dynamic interaction, LLMs are constrained to fixed, human-provided context and cannot autonomously explore the underlying data. To address this limitation, we propose SDE-SQL, a framework that enables large language models to perform self-driven exploration of databases during inference. This is accomplished by generating and executing SQL probes, which allow the model to actively retrieve information from the database and iteratively update its understanding of the data. Unlike prior methods, SDE-SQL operates in a zero-shot setting, without relying on any question-SQL pairs as in-context demonstrations. When evaluated on the BIRD benchmark with Qwen2.5-72B-Instruct, SDE-SQL achieves an 8.02% relative improvement in execution accuracy over the vanilla Qwen2.5-72B-Instruct baseline, establishing a new state-of-the-art among methods based on open-source models without supervised fine-tuning (SFT) or model ensembling. Moreover, with SFT, the performance of SDE-SQL can be further enhanced, yielding an additional 0.52% improvement. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）在Text-to-SQL任务上的性能显著提升。然而，以往的方法通常依赖于推理时提供的静态预处理数据库信息，这限制了模型全面理解数据库内容的能力。缺乏动态交互，LLMs被迫依赖固定的、由人类提供的上下文，不能自主探索底层数据。为解决这一限制，我们提出了一种名为SDE-SQL的框架，使大型语言模型在推理过程中能够自主探索数据库。通过生成和执行SQL探针，模型可以主动从数据库中检索信息并逐步更新其对数据的理解。与以往方法不同，SDE-SQL在零样本设置中运行，无需任何问题-SQL对作为上下文示例。当在BIRD基准测试中使用Qwen2.5-72B-Instruct评估时，SDE-SQL相比vanilla Qwen2.5-72B-Instruct基线实现了8.02%的执行准确度相对提升，成为基于开源模型且未经过监督微调（SFT）或模型聚合的方法中新的性能最优。此外，通过微调（SFT），SDE-SQL的性能可以进一步提升，额外获得0.52%的提升。 

---
# Overclocking LLM Reasoning: Monitoring and Controlling Thinking Path Lengths in LLMs 

**Title (ZH)**: 提升LLM推理速度：监测和控制LLM思维路径长度 

**Authors**: Roy Eisenstadt, Itamar Zimerman, Lior Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2506.07240)  

**Abstract**: Recently, techniques such as explicit structured reasoning have demonstrated strong test-time scaling behavior by enforcing a separation between the model's internal "thinking" process and the final response. A key factor influencing answer quality in this setting is the length of the thinking stage. When the reasoning is too short, the model may fail to capture the complexity of the task. Conversely, when it is too long, the model may overthink, leading to unnecessary computation and degraded performance. This paper explores and exploits the underlying mechanisms by which LLMs understand and regulate the length of their reasoning during explicit thought processes. First, we show that LLMs encode their progress through the reasoning process and introduce an interactive progress bar visualization, which is then used to reveal insights on the model's planning dynamics. Second, we manipulate the internal progress encoding during inference to reduce unnecessary steps and generate a more concise and decisive chain of thoughts. Our empirical results demonstrate that this "overclocking" method mitigates overthinking, improves answer accuracy, and reduces inference latency. Our code is publicly available. 

**Abstract (ZH)**: 最近，通过将模型的内部“思考”过程与最终响应分离的显式结构化推理技术展示了强大的测试时扩展行为。在这种情况下，影响答案质量的关键因素是在思考阶段的长度。当推理过程太短时，模型可能无法捕捉任务的复杂性；反之，当推理过程太长时，模型可能会过度思考，导致不必要的计算并降低性能。本文探讨并利用了LLMs理解并调节其显式思考过程中的推理长度的基础机制。首先，我们展示了LLMs编码其推理过程中的进展，并引入了一个交互式进度条可视化工具，从而揭示了模型计划动态的相关见解。其次，我们在推理过程中操纵内部进展编码以减少不必要的步骤，生成更简洁和果断的思考链。我们的实验证明了这种“超频”方法可以减轻过度思考、提高答案准确性并减少推理延迟。我们的代码已公开。 

---
# VeriLoC: Line-of-Code Level Prediction of Hardware Design Quality from Verilog Code 

**Title (ZH)**: VeriLoC: 基于Verilog代码行级预测硬件设计质量 

**Authors**: Raghu Vamshi Hemadri, Jitendra Bhandari, Johann Knechtel, Badri P Gopalan, Ramesh Narayanaswamy, Ramesh Karri, Siddharth Garg  

**Link**: [PDF](https://arxiv.org/pdf/2506.07239)  

**Abstract**: Modern chip design is complex, and there is a crucial need for early-stage prediction of key design-quality metrics like timing and routing congestion directly from Verilog code (a commonly used programming language for hardware design). It is especially important yet complex to predict individual lines of code that cause timing violations or downstream routing congestion. Prior works have tried approaches like converting Verilog into an intermediate graph representation and using LLM embeddings alongside other features to predict module-level quality, but did not consider line-level quality prediction. We propose VeriLoC, the first method that predicts design quality directly from Verilog at both the line- and module-level. To this end, VeriLoC leverages recent Verilog code-generation LLMs to extract local line-level and module-level embeddings, and train downstream classifiers/regressors on concatenations of these embeddings. VeriLoC achieves high F1-scores of 0.86-0.95 for line-level congestion and timing prediction, and reduces the mean average percentage error from 14% - 18% for SOTA methods down to only 4%. We believe that VeriLoC embeddings and insights from our work will also be of value for other predictive and optimization tasks for complex hardware design. 

**Abstract (ZH)**: 现代芯片设计复杂，需要从Verilog代码（一种常用的硬件设计编程语言）直接预测关键设计质量指标（如时序和布线拥塞）的早期阶段。预测导致时序违反或下游布线拥塞的单独代码行尤为重要且复杂。以往的工作尝试通过将Verilog转换为中间图表示，并结合其他特征使用LLM嵌入来预测模块级别的质量，但没有考虑行级别质量预测。我们提出VeriLoC，这是第一个可以从Verilog代码直接预测设计质量的方法，适用于行级别和模块级别的预测。为此，VeriLoC 利用最近的Verilog代码生成LLM提取局部行级别和模块级别的嵌入，并对这些嵌入的串联进行下游分类器/回归器训练。VeriLoC 在行级拥塞和时序预测中实现了0.86-0.95的高F1分数，并将最先进的方法的平均百分比误差从14%-18%降低到仅4%。我们认为，VeriLoC嵌入和我们工作中获得的见解也将对复杂硬件设计的其他预测和优化任务有价值。 

---
# Learn as Individuals, Evolve as a Team: Multi-agent LLMs Adaptation in Embodied Environments 

**Title (ZH)**: 个体学习，团队进化：多智能体LLMs在体言环境中的适应性发展 

**Authors**: Xinran Li, Chenjia Bai, Zijian Li, Jiakun Zheng, Ting Xiao, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07232)  

**Abstract**: Large language models (LLMs) possess extensive knowledge bases and strong reasoning capabilities, making them promising tools for complex, multi-agent planning in embodied environments. However, despite LLMs' advanced abilities and the sophisticated modular design of agentic methods, existing LLM-based planning algorithms remain limited by weak adaptation capabilities to multi-agent embodied scenarios. We address this limitation by introducing a framework that enables LLM agents to learn and evolve both before and during test time, equipping them with environment-relevant knowledge for better planning and enhanced communication for improved cooperation. Inspired by centralized training with decentralized execution in multi-agent reinforcement learning, we propose a \textit{Learn as Individuals, Evolve as a Team (LIET)} paradigm for multi-agent LLMs adaptation. At the individual level, LLM agents learn a local utility function from exploratory datasets to better comprehend the embodied environment, which is then queried during test time to support informed decision-making. At the team level, LLM agents collaboratively and iteratively maintain and update a shared cooperation knowledge list based on new experiences, using it to guide more effective communication. By combining individual learning with team evolution, LIET enables comprehensive and flexible adaptation for LLM agents. Our experiments on Communicative Watch-And-Help and ThreeD-World Multi-Agent Transport benchmarks demonstrate that LIET, instantiated with both LLaMA and GPT-4o, outperforms existing baselines and exhibits strong cooperative planning abilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）具备广泛的知识库和强大的推理能力，使其成为复杂、多智能体环境感知规划的有希望工具。然而，尽管LLMs具有高级能力且智能体方法具有精妙的模块化设计，现有的基于LLM的规划算法在多智能体感知场景的适应能力上仍然有限。我们通过引入一个框架来解决这一限制，该框架使LLM智能体能够在测试前后学习和进化，从而它们能够获得与环境相关的关键知识，进行更好的规划并增强交流以提高合作效果。受多智能体强化学习中集中训练分散执行的启发，我们提出了一种多智能体大型语言模型适应的“个体学习，团队进化（LIET）”范式。在个体层面，LLM智能体从探索性数据集中学习局部效用函数，以便更好地理解感知环境，测试时查询该效用函数以支持明智的决策。在团队层面，智能体协作并迭代地维护和更新基于新体验的共享合作知识列表，使用该列表来指导更有效的交流。通过结合个体学习与团队进化，LIET能够实现全面且灵活的适应。我们在Communicative Watch-And-Help和ThreeD-World多智能体运输基准测试上的实验表明，无论使用LLaMA还是GPT-4o实现，LIET都能超越现有基线，并展示出强大的协同规划能力。 

---
# Advancing Multimodal Reasoning Capabilities of Multimodal Large Language Models via Visual Perception Reward 

**Title (ZH)**: 通过视觉感知奖励提升多模态大型语言模型的多模态推理能力 

**Authors**: Tong Xiao, Xin Xu, Zhenya Huang, Hongyu Gao, Quan Liu, Qi Liu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.07218)  

**Abstract**: Enhancing the multimodal reasoning capabilities of Multimodal Large Language Models (MLLMs) is a challenging task that has attracted increasing attention in the community. Recently, several studies have applied Reinforcement Learning with Verifiable Rewards (RLVR) to the multimodal domain in order to enhance the reasoning abilities of MLLMs. However, these works largely overlook the enhancement of multimodal perception capabilities in MLLMs, which serve as a core prerequisite and foundational component of complex multimodal reasoning. Through McNemar's test, we find that existing RLVR method fails to effectively enhance the multimodal perception capabilities of MLLMs, thereby limiting their further improvement in multimodal reasoning. To address this limitation, we propose Perception-R1, which introduces a novel visual perception reward that explicitly encourages MLLMs to perceive the visual content accurately, thereby can effectively incentivizing both their multimodal perception and reasoning capabilities. Specifically, we first collect textual visual annotations from the CoT trajectories of multimodal problems, which will serve as visual references for reward assignment. During RLVR training, we employ a judging LLM to assess the consistency between the visual annotations and the responses generated by MLLM, and assign the visual perception reward based on these consistency judgments. Extensive experiments on several multimodal reasoning benchmarks demonstrate the effectiveness of our Perception-R1, which achieves state-of-the-art performance on most benchmarks using only 1,442 training data. 

**Abstract (ZH)**: 增强多模态大型语言模型的多模态推理能力是一个具有挑战性的任务，已在社区中引起越来越多的关注。最近，一些研究将可验证奖励的强化学习（RLVR）应用于多模态领域，以提升多模态大型语言模型（MLLMs）的推理能力。然而，这些工作在增强MLLMs的多模态感知能力方面仍然关注不足，而后者是复杂多模态推理的核心先决条件和基础组件。通过麦加尼尔检验，我们发现现有的RLVR方法未能有效提升MLLMs的多模态感知能力，从而限制了它们在多模态推理方面的进一步改进。为解决这一局限性，我们提出了Perception-R1，引入了一种新型的视觉感知奖励，明确鼓励MLLMs准确感知视觉内容，从而有效激励其多模态感知和推理能力。具体而言，我们首先从多模态问题的CoT轨迹中收集文本视觉注释，作为奖励分配的视觉参考。在RLVR训练过程中，我们利用一个评判大语言模型评估MLLM生成的响应与视觉注释之间的一致性，并基于这些一致性判断分配视觉感知奖励。在多个多模态推理基准上的广泛实验表明，Perception-R1的有效性，仅使用1,442训练数据就在大多数基准上取得了最先进的性能。 

---
# Sword and Shield: Uses and Strategies of LLMs in Navigating Disinformation 

**Title (ZH)**: 剑与盾：大型语言模型在应对假信息中的应用与策略 

**Authors**: Gionnieve Lim, Bryan Chen Zhengyu Tan, Kellie Yu Hui Sim, Weiyan Shi, Ming Hui Chew, Ming Shan Hee, Roy Ka-Wei Lee, Simon T. Perrault, Kenny Tsu Wei Choo  

**Link**: [PDF](https://arxiv.org/pdf/2506.07211)  

**Abstract**: The emergence of Large Language Models (LLMs) presents a dual challenge in the fight against disinformation. These powerful tools, capable of generating human-like text at scale, can be weaponised to produce sophisticated and persuasive disinformation, yet they also hold promise for enhancing detection and mitigation strategies. This paper investigates the complex dynamics between LLMs and disinformation through a communication game that simulates online forums, inspired by the game Werewolf, with 25 participants. We analyse how Disinformers, Moderators, and Users leverage LLMs to advance their goals, revealing both the potential for misuse and combating disinformation. Our findings highlight the varying uses of LLMs depending on the participants' roles and strategies, underscoring the importance of understanding their effectiveness in this context. We conclude by discussing implications for future LLM development and online platform design, advocating for a balanced approach that empowers users and fosters trust while mitigating the risks of LLM-assisted disinformation. 

**Abstract (ZH)**: 大型语言模型的兴起在反信息操纵斗争中带来了双重挑战。这些强大的工具能够大规模生成类人类文本，可以被 weaponised 用来生成复杂且有说服力的信息操纵内容，同时它们也为增强检测和缓解策略带来了希望。本文通过一个受狼人游戏启发、模拟在线论坛的沟通游戏，分析了 25 名参与者如何利用大型语言模型推进各自的目标，揭示了利用大型语言模型的潜在风险和对抗信息操纵的可能性。研究发现强调了根据参与者角色和策略，大型语言模型使用方式的多样性，突显了在此背景下理解其有效性的必要性。最后，本文讨论了对未来大型语言模型开发和在线平台设计的 implications，提倡一种平衡的方法，既能增强用户能力、培养信任，又能减轻大型语言模型辅助的信息操纵风险。 

---
# Flattery in Motion: Benchmarking and Analyzing Sycophancy in Video-LLMs 

**Title (ZH)**: 奉迎之舞：视频大语言模型中奉迎行为的基准测试与分析 

**Authors**: Wenrui Zhou, Shu Yang, Qingsong Yang, Zikun Guo, Lijie Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07180)  

**Abstract**: As video large language models (Video-LLMs) become increasingly integrated into real-world applications that demand grounded multimodal reasoning, ensuring their factual consistency and reliability is of critical importance. However, sycophancy, the tendency of these models to align with user input even when it contradicts the visual evidence, undermines their trustworthiness in such contexts. Current sycophancy research has largely overlooked its specific manifestations in the video-language domain, resulting in a notable absence of systematic benchmarks and targeted evaluations to understand how Video-LLMs respond under misleading user input. To fill this gap, we propose VISE (Video-LLM Sycophancy Benchmarking and Evaluation), the first dedicated benchmark designed to evaluate sycophantic behavior in state-of-the-art Video-LLMs across diverse question formats, prompt biases, and visual reasoning tasks. Specifically, VISE pioneeringly brings linguistic perspectives on sycophancy into the visual domain, enabling fine-grained analysis across multiple sycophancy types and interaction patterns. In addition, we explore key-frame selection as an interpretable, training-free mitigation strategy, which reveals potential paths for reducing sycophantic bias by strengthening visual grounding. 

**Abstract (ZH)**: 视频大语言模型（Video-LLMs）日益融入要求情境多模态推理的真实世界应用中，确保其事实一致性和可靠性至关重要。然而，这些模型倾向于与用户输入一致，即使这与视觉证据相矛盾，这种迎合行为损其在这些情境下的可信度。当前关于迎合的研究大多忽视了视频语言领域其特定表现形式，导致缺乏系统基准和针对性评估来理解Video-LLMs在误导性用户输入下的反应方式。为填补这一空白，我们提出VISE（Video-LLM Sycophancy Benchmarking and Evaluation），这是首个专门为此目的设计的基准，旨在评估最先进的Video-LLMs在多种问题格式、提示偏见和视觉推理任务中的迎合行为。具体而言，VISE 首次将语言学视角的迎合带入视觉领域，实现对多种迎合类型和交互模式的精细分析。此外，我们探索关键帧选择作为一种可解释、无需训练的缓解策略，揭示通过增强视觉 grounding 减少迎合偏见的潜在路径。 

---
# Regularized Adaptive Graph Learning for Large-Scale Traffic Forecasting 

**Title (ZH)**: 正则化自适应图学习在大规模交通预测中的应用 

**Authors**: Kaiqi Wu, Weiyang Kong, Sen Zhang, Yubao Liu, Zitong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.07179)  

**Abstract**: Traffic prediction is a critical task in spatial-temporal forecasting with broad applications in travel planning and urban management. Adaptive graph convolution networks have emerged as mainstream solutions due to their ability to learn node embeddings in a data-driven manner and capture complex latent dependencies. However, existing adaptive graph learning methods for traffic forecasting often either ignore the regularization of node embeddings, which account for a significant proportion of model parameters, or face scalability issues from expensive graph convolution operations. To address these challenges, we propose a Regularized Adaptive Graph Learning (RAGL) model. First, we introduce a regularized adaptive graph learning framework that synergizes Stochastic Shared Embedding (SSE) and adaptive graph convolution via a residual difference mechanism, achieving both embedding regularization and noise suppression. Second, to ensure scalability on large road networks, we develop the Efficient Cosine Operator (ECO), which performs graph convolution based on the cosine similarity of regularized embeddings with linear time complexity. Extensive experiments on four large-scale real-world traffic datasets show that RAGL consistently outperforms state-of-the-art methods in terms of prediction accuracy and exhibits competitive computational efficiency. 

**Abstract (ZH)**: 交通预测是时空预测中的关键任务，广泛应用于出行规划和城市管理。基于数据驱动的方法学习节点嵌入并捕获复杂潜在依赖性的自适应图卷积网络已成为主流解决方案。然而，现有的交通预测自适应图学习方法往往要么忽视了节点嵌入的正则化，这些嵌入占了模型参数的重要比例，要么由于昂贵的图卷积操作面临可扩展性问题。为解决这些挑战，我们提出了一种正则化自适应图学习（RAGL）模型。首先，我们引入了一种结合随机共享嵌入（SSE）和自适应图卷积的正则化自适应图学习框架，通过残差差分机制实现嵌入正则化和噪声抑制。其次，为了确保在大规模道路网络上的可扩展性，我们开发了高效余弦操作器（ECO），它基于正则化嵌入的余弦相似性进行图卷积，并具有线性时间复杂度。在四个大规模现实世界交通数据集上的 extensive 实验表明，RAGL 在预测准确性上始终优于现有最先进的方法，并且具有竞争力的计算效率。 

---
# Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models 

**Title (ZH)**: 帧导向：视频扩散模型中无需训练的帧级控制导向方法 

**Authors**: Sangwon Jang, Taekyung Ki, Jaehyeong Jo, Jaehong Yoon, Soo Ye Kim, Zhe Lin, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07177)  

**Abstract**: Advancements in diffusion models have significantly improved video quality, directing attention to fine-grained controllability. However, many existing methods depend on fine-tuning large-scale video models for specific tasks, which becomes increasingly impractical as model sizes continue to grow. In this work, we present Frame Guidance, a training-free guidance for controllable video generation based on frame-level signals, such as keyframes, style reference images, sketches, or depth maps. For practical training-free guidance, we propose a simple latent processing method that dramatically reduces memory usage, and apply a novel latent optimization strategy designed for globally coherent video generation. Frame Guidance enables effective control across diverse tasks, including keyframe guidance, stylization, and looping, without any training, compatible with any video models. Experimental results show that Frame Guidance can produce high-quality controlled videos for a wide range of tasks and input signals. 

**Abstract (ZH)**: 基于帧级信号的无训练引导：控制视频生成的新方法 

---
# CTDGSI: A comprehensive exploitation of instance selection methods for automatic text classification. VII Concurso de Teses, Dissertações e Trabalhos de Graduação em SI -- XXI Simpósio Brasileiro de Sistemas de Informação 

**Title (ZH)**: CTDGSI：自动文本分类中实例选择方法的综合探讨——第七届全国信息系统硕士、博士论文及本科毕业设计竞赛暨第二十一次巴西信息系统研讨会。 

**Authors**: Washington Cunha, Leonardo Rocha, Marcos André Gonçalves  

**Link**: [PDF](https://arxiv.org/pdf/2506.07169)  

**Abstract**: Progress in Natural Language Processing (NLP) has been dictated by the rule of more: more data, more computing power and more complexity, best exemplified by the Large Language Models. However, training (or fine-tuning) large dense models for specific applications usually requires significant amounts of computing resources. This \textbf{Ph.D. dissertation} focuses on an under-investi\-gated NLP data engineering technique, whose potential is enormous in the current scenario known as Instance Selection (IS). The IS goal is to reduce the training set size by removing noisy or redundant instances while maintaining the effectiveness of the trained models and reducing the training process cost. We provide a comprehensive and scientifically sound comparison of IS methods applied to an essential NLP task -- Automatic Text Classification (ATC), considering several classification solutions and many datasets. Our findings reveal a significant untapped potential for IS solutions. We also propose two novel IS solutions that are noise-oriented and redundancy-aware, specifically designed for large datasets and transformer architectures. Our final solution achieved an average reduction of 41\% in training sets, while maintaining the same levels of effectiveness in all datasets. Importantly, our solutions demonstrated speedup improvements of 1.67x (up to 2.46x), making them scalable for datasets with hundreds of thousands of documents. 

**Abstract (ZH)**: 自然语言处理（NLP）的进步主要受到更多数据、更多计算资源和更复杂模型的驱动，大型语言模型是最典型的例子。然而，为了特定应用训练（或微调）大型密集模型通常需要大量的计算资源。本博士论文专注于一个尚未充分研究的NLP数据工程技术，在当前被称为实例选择（IS）的场景中其潜力巨大。IS的目标是通过去除噪声或冗余实例来减少训练集规模，同时保持训练模型的效果并降低训练过程成本。我们对IS方法在一项重要NLP任务——自动文本分类（ATC）中的应用进行了全面且科学的比较，考虑了多种分类解决方案和多个数据集。我们的研究发现IS解决方案具有巨大的未开发潜力。我们还提出两种针对大型数据集和变换器架构的新颖IS解决方案， noise导向和冗余 aware。最终解决方案在所有数据集上实现了训练集规模平均41%的减少，同时保持相同的效果水平。重要的是，我们的解决方案在某些情况下显示出1.67倍（最高2.46倍）的加速改进，使其适用于包含数十万文档的数据集，具有可扩展性。 

---
# Efficient Text-Attributed Graph Learning through Selective Annotation and Graph Alignment 

**Title (ZH)**: 通过选择性注释和图对齐实现高效的文本attributed图学习 

**Authors**: Huanyi Xie, Lijie Hu, Lu Yu, Tianhao Huang, Longfei Li, Meng Li, Jun Zhou, Huan Wang, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07168)  

**Abstract**: In the realm of Text-attributed Graphs (TAGs), traditional graph neural networks (GNNs) often fall short due to the complex textual information associated with each node. Recent methods have improved node representations by leveraging large language models (LLMs) to enhance node text features, but these approaches typically require extensive annotations or fine-tuning across all nodes, which is both time-consuming and costly. To overcome these challenges, we introduce GAGA, an efficient framework for TAG representation learning. GAGA reduces annotation time and cost by focusing on annotating only representative nodes and edges. It constructs an annotation graph that captures the topological relationships among these annotations. Furthermore, GAGA employs a two-level alignment module to effectively integrate the annotation graph with the TAG, aligning their underlying structures. Experiments show that GAGA achieves classification accuracies on par with or surpassing state-of-the-art methods while requiring only 1% of the data to be annotated, demonstrating its high efficiency. 

**Abstract (ZH)**: 在文本属性图（TAGs）领域，传统的图神经网络（GNNs）往往因为每个节点关联的复杂文本信息而表现不佳。近期的方法通过利用大规模语言模型（LLMs）增强节点文本特征来改进节点表示，但这些方法通常需要对所有节点进行大量标注或微调，这既耗时又昂贵。为克服这些挑战，我们引入了GAGA，一种高效的TAG表示学习框架。GAGA通过仅标注代表性节点和边来减少标注时间和成本，并构建一个标注图以捕捉这些标注之间的拓扑关系。此外，GAGA采用两级对齐模块有效整合标注图和TAG，对其潜在结构进行对齐。实验表明，GAGA在标注数据仅占总量1%的情况下，仍能达到或超过现有最佳方法的分类精度，展示了其高度的效率。 

---
# AMoPO: Adaptive Multi-objective Preference Optimization without Reward Models and Reference Models 

**Title (ZH)**: AMoPO：无需奖励模型和参考模型的自适应多目标偏好优化 

**Authors**: Qi Liu, Jingqing Ruan, Hao Li, Haodong Zhao, Desheng Wang, Jiansong Chen, Wan Guanglu, Xunliang Cai, Zhi Zheng, Tong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07165)  

**Abstract**: Existing multi-objective preference alignment methods for large language models (LLMs) face limitations: (1) the inability to effectively balance various preference dimensions, and (2) reliance on auxiliary reward/reference models introduces computational complexity. To address these challenges, we propose Adaptive Multi-objective Preference Optimization (AMoPO), a novel framework that achieves dynamic balance across preference dimensions. By introducing the multi-objective optimization paradigm to use the dimension-aware generation metrics as implicit rewards, AMoPO aligns LLMs with diverse preferences without additional reward models or reference models. We introduce an adaptive weight assignment mechanism that models the generation space as a Gaussian distribution, allowing dynamic prioritization of preference dimensions. Empirical results demonstrate that AMoPO outperforms state-of-the-art baselines by 28.5%, and the experiments on 7B, 14B, and 32B models reveal the scaling ability of AMoPO. Moreover, additional analysis of multiple dimensions verifies its adaptability and effectiveness. These findings validate AMoPO's capability to achieve dimension-aware preference alignment, highlighting its superiority. Our codes and datasets are available at this https URL. 

**Abstract (ZH)**: 现有大型语言模型的多目标偏好对齐方法面临限制：（1）无法有效平衡各种偏好维度，（2）依赖辅助奖励/参考模型引入了计算复杂性。为解决这些问题，我们提出了一种新的框架——自适应多目标偏好优化（AMoPO），该框架实现了偏好维度之间的动态平衡。通过引入多目标优化范式并使用维度感知的生成度量作为隐式奖励，AMoPO可以在无需额外奖励模型或参考模型的情况下使大型语言模型与多种偏好对齐。我们引入了一种自适应权重分配机制，将生成空间建模为高斯分布，从而实现偏好维度的动态优先级划分。实验证明，AMoPO在最先进的基线方法上表现优于28.5%，并对7B、14B和32B规模的模型进行了实验，展示了AMoPO的可扩展性。此外，对多个维度的进一步分析验证了其适应性和有效性。这些发现验证了AMoPO在实现偏好维度感知对齐方面的能力，突显了其优越性。我们的代码和数据集可在以下链接获取。 

---
# Syntactic Control of Language Models by Posterior Inference 

**Title (ZH)**: 语言模型的后验推断控制-syntax控制 

**Authors**: Vicky Xefteri, Tim Vieira, Ryan Cotterell, Afra Amini  

**Link**: [PDF](https://arxiv.org/pdf/2506.07154)  

**Abstract**: Controlling the syntactic structure of text generated by language models is valuable for applications requiring clarity, stylistic consistency, or interpretability, yet it remains a challenging task. In this paper, we argue that sampling algorithms based on the posterior inference can effectively enforce a target constituency structure during generation. Our approach combines sequential Monte Carlo, which estimates the posterior distribution by sampling from a proposal distribution, with a syntactic tagger that ensures that each generated token aligns with the desired syntactic structure. Our experiments with GPT2 and Llama3-8B models show that with an appropriate proposal distribution, we can improve syntactic accuracy, increasing the F1 score from $12.31$ (GPT2-large) and $35.33$ (Llama3-8B) to about $93$ in both cases without compromising the language model's fluency. These results underscore both the complexity of syntactic control and the effectiveness of sampling algorithms, offering a promising approach for applications where precise control over syntax is essential. 

**Abstract (ZH)**: 基于后验推理的采样算法在控制语言模型生成文本的句法结构方面具有价值：一种改进清晰度、风格一致性或可解释性的有效方法但仍是一项具有挑战性的任务。在本文中，我们提出了一种结合顺序蒙特卡罗方法和句法标注器的方法，在生成过程中有效地强制执行目标句法结构。我们使用GPT2和Llama3-8B模型的实验表明，在适当的提议分布下，可以提高句法准确性，F1分数从GPT2-large的12.31和Llama3-8B的35.33提高到接近93，而不影响语言模型的流畅性。这些结果强调了句法控制的复杂性以及采样算法的有效性，为需要对语法进行精确控制的应用提供了有前景的方法。 

---
# Mind the Web: The Security of Web Use Agents 

**Title (ZH)**: 关注网络：网络使用代理的安全性 

**Authors**: Avishag Shapira, Parth Atulbhai Gandhi, Edan Habler, Oleg Brodt, Asaf Shabtai  

**Link**: [PDF](https://arxiv.org/pdf/2506.07153)  

**Abstract**: Web-use agents are rapidly being deployed to automate complex web tasks, operating with extensive browser capabilities including multi-tab navigation, DOM manipulation, JavaScript execution and authenticated session access. However, these powerful capabilities create a critical and previously unexplored attack surface. This paper demonstrates how attackers can exploit web-use agents' high-privilege capabilities by embedding malicious content in web pages such as comments, reviews, or advertisements that agents encounter during legitimate browsing tasks. In addition, we introduce the task-aligned injection technique that frame malicious commands as helpful task guidance rather than obvious attacks. This technique exploiting fundamental limitations in LLMs' contextual reasoning: agents struggle in maintaining coherent contextual awareness and fail to detect when seemingly helpful web content contains steering attempts that deviate from their original task goal. Through systematic evaluation of four popular agents (OpenAI Operator, Browser Use, Do Browser, OpenOperator), we demonstrate nine payload types that compromise confidentiality, integrity, and availability, including unauthorized camera activation, user impersonation, local file exfiltration, password leakage, and denial of service, with validation across multiple LLMs achieving success rates of 80%-100%. These payloads succeed across agents with built-in safety mechanisms, requiring only the ability to post content on public websites, creating unprecedented risks given the ease of exploitation combined with agents' high-privilege access. To address this attack, we propose comprehensive mitigation strategies including oversight mechanisms, execution constraints, and task-aware reasoning techniques, providing practical directions for secure development and deployment. 

**Abstract (ZH)**: 基于网页的代理工具通过嵌入恶意内容利用其高级权限，威胁网络安全性 

---
# Prompting Science Report 2: The Decreasing Value of Chain of Thought in Prompting 

**Title (ZH)**: 提示科学报告 2：链式思考在提示中的价值下降 

**Authors**: Lennart Meincke, Ethan Mollick, Lilach Mollick, Dan Shapiro  

**Link**: [PDF](https://arxiv.org/pdf/2506.07142)  

**Abstract**: This is the second in a series of short reports that seek to help business, education, and policy leaders understand the technical details of working with AI through rigorous testing. In this report, we investigate Chain-of-Thought (CoT) prompting, a technique that encourages a large language model (LLM) to "think step by step" (Wei et al., 2022). CoT is a widely adopted method for improving reasoning tasks, however, our findings reveal a more nuanced picture of its effectiveness. We demonstrate two things:
- The effectiveness of Chain-of-Thought prompting can vary greatly depending on the type of task and model. For non-reasoning models, CoT generally improves average performance by a small amount, particularly if the model does not inherently engage in step-by-step processing by default. However, CoT can introduce more variability in answers, sometimes triggering occasional errors in questions the model would otherwise get right. We also found that many recent models perform some form of CoT reasoning even if not asked; for these models, a request to perform CoT had little impact. Performing CoT generally requires far more tokens (increasing cost and time) than direct answers.
- For models designed with explicit reasoning capabilities, CoT prompting often results in only marginal, if any, gains in answer accuracy. However, it significantly increases the time and tokens needed to generate a response. 

**Abstract (ZH)**: 这是关于通过严格测试帮助商业、教育和政策领导者理解人工智能技术细节的一系列简短报告中的第二篇。本篇报告调查了Chain-of-Thought (CoT) 提示技术，这是一种促进大型语言模型（LLM）进行“逐步思考”的方法（Wei et al., 2022）。虽然CoT是一种广泛采用的提高推理任务效果的方法，但我们的研究发现其效果更为复杂。我们证明了以下两点：
- Chain-of-Thought 提示的有效性在不同任务类型和模型上差异很大。对于非推理模型，CoT通常会通过少量改进提高平均性能，尤其是在模型默认不进行逐步处理的情况下。然而，CoT可能会引入答案更多的变异性，有时会导致模型原本可以正确回答的问题出现偶尔的错误。我们还发现，即使是无需进行逐步处理的许多近期模型，也会自发进行某种形式的CoT推理；对于这些模型，要求其进行CoT提示几乎没有影响。进行CoT通常需要更多的token（增加成本和时间），远远超过直接回答所需的数量。
- 对于设计具备明确推理能力的模型，Chain-of-Thought 提示通常只会带来微小甚至没有性能上的提升，但显著增加了生成答案所需的时间和token数量。 

---
# Learning Compact Vision Tokens for Efficient Large Multimodal Models 

**Title (ZH)**: 学习紧凑的视觉令牌以实现高效的大规模多模态模型 

**Authors**: Hao Tang, Chengchao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.07138)  

**Abstract**: Large multimodal models (LMMs) suffer significant computational challenges due to the high cost of Large Language Models (LLMs) and the quadratic complexity of processing long vision token sequences. In this paper, we explore the spatial redundancy among vision tokens and shorten the length of vision token sequences for inference acceleration. Specifically, we propose a Spatial Token Fusion (STF) method to learn compact vision tokens for short vision token sequence, where spatial-adjacent tokens are fused into one. Meanwhile, weight-frozen vision encoder can not well adapt to the demand of extensive downstream vision-language tasks. To this end, we further introduce a Multi-Block Token Fusion (MBTF) module to supplement multi-granularity features for the reduced token sequence. Overall, we combine STF and MBTF module to balance token reduction and information preservation, thereby improving inference efficiency without sacrificing multimodal reasoning capabilities. Experimental results demonstrate that our method based on LLaVA-1.5 achieves comparable or even superior performance to the baseline on 8 popular vision-language benchmarks with only $25\%$ vision tokens of baseline. The source code and trained weights are available at this https URL. 

**Abstract (ZH)**: 大规模多模态模型（LMMs）因大型语言模型（LLMs）的高计算成本和处理长视觉标记序列的二次复杂性而面临重大计算挑战。本文探索视觉标记之间的空间冗余，并缩短视觉标记序列以加速推理。具体来说，我们提出了一种空间标记融合（STF）方法，通过聚合相邻的视觉标记来学习紧凑的视觉标记，从而缩短视觉标记序列的长度。与此同时，权重冻结的视觉编码器难以适应广泛的下游视觉-语言任务的需求。为此，我们进一步引入了多块标记融合（MBTF）模块，以补充减少的标记序列的多粒度特征。总体而言，我们将STF模块和MBTF模块结合，以平衡标记减少和信息保留，从而在不牺牲多模态推理能力的情况下提高推理效率。实验结果表明，基于LLaVA-1.5的方法仅使用基线的25%视觉标记，在8个流行的视觉-语言基准上达到了可比或更优的性能。源代码和训练权重可在以下链接获取。 

---
# Taxonomy of migration scenarios for Qiskit refactoring using LLMs 

**Title (ZH)**: 使用LLMs进行Qiskit重构的迁移场景分类 

**Authors**: José Manuel Suárez, Luís Mariano Bibbó, Joaquín Bogado, Alejandro Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2506.07135)  

**Abstract**: As quantum computing advances, quantum programming libraries' heterogeneity and steady evolution create new challenges for software developers. Frequent updates in software libraries break working code that needs to be refactored, thus adding complexity to an already complex landscape. These refactoring challenges are, in many cases, fundamentally different from those known in classical software engineering due to the nature of quantum computing software. This study addresses these challenges by developing a taxonomy of quantum circuit's refactoring problems, providing a structured framework to analyze and compare different refactoring approaches. Large Language Models (LLMs) have proven valuable tools for classic software development, yet their value in quantum software engineering remains unexplored. This study uses LLMs to categorize refactoring needs in migration scenarios between different Qiskit versions. Qiskit documentation and release notes were scrutinized to create an initial taxonomy of refactoring required for migrating between Qiskit releases. Two taxonomies were produced: one by expert developers and one by an LLM. These taxonomies were compared, analyzing differences and similarities, and were integrated into a unified taxonomy that reflects the findings of both methods. By systematically categorizing refactoring challenges in Qiskit, the unified taxonomy is a foundation for future research on AI-assisted migration while enabling a more rigorous evaluation of automated refactoring techniques. Additionally, this work contributes to quantum software engineering (QSE) by enhancing software development workflows, improving language compatibility, and promoting best practices in quantum programming. 

**Abstract (ZH)**: 随着量子计算的进步，量子编程库的异构性及其持续进化为软件开发人员带来了新的挑战。频繁的软件库更新会打破现有的工作代码，需要进行重构，从而增加了已经复杂的软件开发环境的复杂性。这些重构挑战在很大程度上不同于经典软件工程中的已知挑战，这是由于量子计算软件的本质特性所致。本研究通过开发量子电路重构问题的分类体系，为分析和比较不同的重构方法提供了结构化的框架。大规模语言模型（LLMs）在经典软件开发中已被证明是非常有价值的工具，但在量子软件工程中的应用价值尚未被探索。本研究使用LLMs对不同Qiskit版本之间的迁移场景中的重构需求进行了分类。详细审查了Qiskit文档和发布说明，以建立一个初步的迁移所需的重构分类体系。生成了两个分类体系：一个是专家开发人员生成的，另一个是由LLM生成的。对比了这两个分类体系，分析了它们之间的差异和相似之处，并将它们整合成一个统一的分类体系，该分类体系反映了两种方法的发现。通过系统地分类Qiskit中的重构挑战，该统一分类体系为未来的人工智能辅助迁移研究奠定了基础，同时促进了自动化重构技术的更严格的评估。此外，本研究还通过增强软件开发工作流程、提高语言兼容性和推广量子编程的最佳实践，促进了量子软件工程（QSE）的发展。 

---
# Reliable Critics: Monotonic Improvement and Convergence Guarantees for Reinforcement Learning 

**Title (ZH)**: 可靠的批评者：强化学习中的单调改进与收敛保证 

**Authors**: Eshwar S. R., Gugan Thoppe, Aditya Gopalan, Gal Dalal  

**Link**: [PDF](https://arxiv.org/pdf/2506.07134)  

**Abstract**: Despite decades of research, it remains challenging to correctly use Reinforcement Learning (RL) algorithms with function approximation. A prime example is policy iteration, whose fundamental guarantee of monotonic improvement collapses even under linear function approximation. To address this issue, we introduce Reliable Policy Iteration (RPI). It replaces the common projection or Bellman-error minimization during policy evaluation with a Bellman-based constrained optimization. We prove that not only does RPI confer textbook monotonicity on its value estimates but these estimates also lower bound the true return. Also, their limit partially satisfies the unprojected Bellman equation, emphasizing RPI's natural fit within RL. RPI is the first algorithm with such monotonicity and convergence guarantees under function approximation. For practical use, we provide a model-free variant of RPI that amounts to a novel critic. It can be readily integrated into primary model-free PI implementations such as DQN and DDPG. In classical control tasks, such RPI-enhanced variants consistently maintain their lower-bound guarantee while matching or surpassing the performance of all baseline methods. 

**Abstract (ZH)**: 尽管已有数十年的研究，但在功能逼近的情况下正确使用强化学习（RL）算法仍具挑战性。可靠政策迭代（RPI）通过在策略评估中使用基于贝尔曼的约束优化，替代常见的投影或贝尔曼误差最小化，解决了这一问题。我们证明RPI不仅赋予其价值估计教科书级别的单调性，同时也提供了真实回报的下界。此外，其极限部分满足未投影的贝尔曼方程，强调了RPI在强化学习中的自然契合度。RPI是首个在功能逼近情况下具备单调性和收敛性保证的算法。为实际应用，我们提供了一种基于模型的RPI变体，它相当于一种新的critic。它可以无缝集成到如DQN和DDPG等主要的基于模型的PI实现中。在经典控制任务中，这些RPI增强的变体能够持续保持其下界保证，并且匹配或超越所有基线方法的表现。 

---
# Robotic Policy Learning via Human-assisted Action Preference Optimization 

**Title (ZH)**: 通过人类辅助动作偏好优化的机器人策略学习 

**Authors**: Wenke xia, Yichu Yang, Hongtao Wu, Xiao Ma, Tao Kong, Di Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07127)  

**Abstract**: Establishing a reliable and iteratively refined robotic system is essential for deploying real-world applications. While Vision-Language-Action (VLA) models are widely recognized as the foundation model for such robotic deployment, their dependence on expert demonstrations hinders the crucial capabilities of correction and learning from failures. To mitigate this limitation, we introduce a Human-assisted Action Preference Optimization method named HAPO, designed to correct deployment failures and foster effective adaptation through preference alignment for VLA models. This method begins with a human-robot collaboration framework for reliable failure correction and interaction trajectory collection through human intervention. These human-intervention trajectories are further employed within the action preference optimization process, facilitating VLA models to mitigate failure action occurrences while enhancing corrective action adaptation. Specifically, we propose an adaptive reweighting algorithm to address the issues of irreversible interactions and token probability mismatch when introducing preference optimization into VLA models, facilitating model learning from binary desirability signals derived from interactions. Through combining these modules, our human-assisted action preference optimization method ensures reliable deployment and effective learning from failure for VLA models. The experiments conducted in simulation and real-world scenarios prove superior generalization and robustness of our framework across a variety of manipulation tasks. 

**Abstract (ZH)**: 基于人类辅助动作偏好的优化方法：实现VLA模型的可靠部署与从失败中有效学习 

---
# MAGNet: A Multi-Scale Attention-Guided Graph Fusion Network for DRC Violation Detection 

**Title (ZH)**: MAGNet：一种多尺度注意力引导的图融合网络用于DRC违规检测 

**Authors**: Weihan Lu, Hong Cai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.07126)  

**Abstract**: Design rule checking (DRC) is of great significance for cost reduction and design efficiency improvement in integrated circuit (IC) designs. Machine-learning-based DRC has become an important approach in computer-aided design (CAD). In this paper, we propose MAGNet, a hybrid deep learning model that integrates an improved U-Net with a graph neural network for DRC violation prediction. The U-Net backbone is enhanced with a Dynamic Attention Module (DAM) and a Multi-Scale Convolution Module (MSCM) to strengthen its capability in extracting fine-grained and multi-scale spatial features. In parallel, we construct a pixel-aligned graph structure based on chip layout tiles, and apply a specialized GNN to model the topological relationships among pins. During graph construction, a graph-to-grid mapping is generated to align GNN features with the layout image. In addition, a label amplification strategy is adopted during training to enhance the model's sensitivity to sparse violation patterns. Overall, MAGNet effectively combines spatial, semantic, and structural information, achieving improved prediction accuracy and reduced false positive rates in DRC hotspot detection. Subsequently, through incremental training, we achieve a more sensitive discrimination ability for hotspots. The results demonstrate that, in comparison with ibUnet, RouteNet, and J-Net, MAGnet significantly outperforms these models, achieving substantial improvements in overall performance. 

**Abstract (ZH)**: 基于机器学习的布局布线验证（DRC）对于集成电路（IC）设计中的成本降低和设计效率提升具有重要意义。本文提出了一种结合改进的U-Net和图神经网络的混合深度学习模型MAGNet，用于DRC违规预测。U-Net主干通过动态注意力模块（DAM）和多尺度卷积模块（MSCM）得到增强，以提高其从细粒度和多尺度空间特征中提取信息的能力。同时，基于芯片布局拼块构建像素对齐的图结构，并使用专门的图神经网络来建模引脚之间的拓扑关系。在图构建过程中，生成图到网格的映射，以使GNN特征与布局图像对齐。此外，在训练过程中采用标签放大策略，以提高模型对稀疏违规模式的敏感性。总体而言，MAGNet有效地结合了空间、语义和结构信息，在DRC热点检测中实现了更高的预测准确性和更低的假阳性率。通过增量训练，实现对热点更敏感的区分能力。结果表明，与ibUnet、RouteNet和J-Net相比，MAGNet在整体性能上显著优于这些模型。 

---
# Image segmentation and classification of E-waste for waste segregation 

**Title (ZH)**: 电子废弃物的图像分割与分类以实现废弃物分选 

**Authors**: Prakriti Tripathi, Theertha Biju, Maniram Thota, Rakesh Lingam  

**Link**: [PDF](https://arxiv.org/pdf/2506.07122)  

**Abstract**: Industry partners provided a problem statement that involves classifying electronic waste using machine learning models that will be used by pick-and-place robots for waste segregation. We started by taking common electronic waste items, such as a mouse and charger, unsoldering them, and taking pictures to create a custom dataset. Then state-of-the art YOLOv11 model was trained and run to achieve 70 mAP in real-time. Mask-RCNN model was also trained and achieved 41 mAP. The model will be further integrated with pick-and-place robots to perform segregation of e-waste. 

**Abstract (ZH)**: 行业伙伴提供了涉及使用机器学习模型对电子废弃物进行分类的问题描述，该模型将用于拾放机器人进行废弃物分类。我们首先对常见的电子废弃物项目，如鼠标和充电器进行去焊处理并拍照，创建自定义数据集。然后，我们使用最新的YOLOv11模型进行训练和运行，实现了实时70 mAP。我们也训练了Mask-RCNN模型，达到了41 mAP。该模型将进一步与拾放机器人集成，以实现电子废弃物的分类。 

---
# Quality-Diversity Red-Teaming: Automated Generation of High-Quality and Diverse Attackers for Large Language Models 

**Title (ZH)**: 质量-多样性红队挑战：大型语言模型高质多样攻击者的自动化生成 

**Authors**: Ren-Jian Wang, Ke Xue, Zeyu Qin, Ziniu Li, Sheng Tang, Hao-Tian Li, Shengcai Liu, Chao Qian  

**Link**: [PDF](https://arxiv.org/pdf/2506.07121)  

**Abstract**: Ensuring safety of large language models (LLMs) is important. Red teaming--a systematic approach to identifying adversarial prompts that elicit harmful responses from target LLMs--has emerged as a crucial safety evaluation method. Within this framework, the diversity of adversarial prompts is essential for comprehensive safety assessments. We find that previous approaches to red-teaming may suffer from two key limitations. First, they often pursue diversity through simplistic metrics like word frequency or sentence embedding similarity, which may not capture meaningful variation in attack strategies. Second, the common practice of training a single attacker model restricts coverage across potential attack styles and risk categories. This paper introduces Quality-Diversity Red-Teaming (QDRT), a new framework designed to address these limitations. QDRT achieves goal-driven diversity through behavior-conditioned training and implements a behavioral replay buffer in an open-ended manner. Additionally, it trains multiple specialized attackers capable of generating high-quality attacks across diverse styles and risk categories. Our empirical evaluation demonstrates that QDRT generates attacks that are both more diverse and more effective against a wide range of target LLMs, including GPT-2, Llama-3, Gemma-2, and Qwen2.5. This work advances the field of LLM safety by providing a systematic and effective approach to automated red-teaming, ultimately supporting the responsible deployment of LLMs. 

**Abstract (ZH)**: 确保大型语言模型的安全性至关重要。质量多样性红队攻击（QDRT）：一种新的系统化安全评估方法 

---
# RBA-FE: A Robust Brain-Inspired Audio Feature Extractor for Depression Diagnosis 

**Title (ZH)**: RBA-FE: 一种稳健的仿脑音频特征提取器用于抑郁诊断 

**Authors**: Yu-Xuan Wu, Ziyan Huang, Bin Hu, Zhi-Hong Guan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07118)  

**Abstract**: This article proposes a robust brain-inspired audio feature extractor (RBA-FE) model for depression diagnosis, using an improved hierarchical network architecture. Most deep learning models achieve state-of-the-art performance for image-based diagnostic tasks, ignoring the counterpart audio features. In order to tailor the noise challenge, RBA-FE leverages six acoustic features extracted from the raw audio, capturing both spatial characteristics and temporal dependencies. This hybrid attribute helps alleviate the precision limitation in audio feature extraction within other learning models like deep residual shrinkage networks. To deal with the noise issues, our model incorporates an improved spiking neuron model, called adaptive rate smooth leaky integrate-and-fire (ARSLIF). The ARSLIF model emulates the mechanism of ``retuning of cellular signal selectivity" in the brain attention systems, which enhances the model robustness against environmental noises in audio data. Experimental results demonstrate that RBA-FE achieves state-of-the-art accuracy on the MODMA dataset, respectively with 0.8750, 0.8974, 0.8750 and 0.8750 in precision, accuracy, recall and F1 score. Extensive experiments on the AVEC2014 and DAIC-WOZ datasets both show enhancements in noise robustness. It is further indicated by comparison that the ARSLIF neuron model suggest the abnormal firing pattern within the feature extraction on depressive audio data, offering brain-inspired interpretability. 

**Abstract (ZH)**: 基于改进层次网络架构的稳健脑启发音频特征提取模型及其在抑郁诊断中的应用 

---
# Towards Universal Offline Black-Box Optimization via Learning Language Model Embeddings 

**Title (ZH)**: 基于学习语言模型嵌入的通用离线黑箱优化方法 

**Authors**: Rong-Xi Tan, Ming Chen, Ke Xue, Yao Wang, Yaoyuan Wang, Sheng Fu, Chao Qian  

**Link**: [PDF](https://arxiv.org/pdf/2506.07109)  

**Abstract**: The pursuit of universal black-box optimization (BBO) algorithms is a longstanding goal. However, unlike domains such as language or vision, where scaling structured data has driven generalization, progress in offline BBO remains hindered by the lack of unified representations for heterogeneous numerical spaces. Thus, existing offline BBO approaches are constrained to single-task and fixed-dimensional settings, failing to achieve cross-domain universal optimization. Recent advances in language models (LMs) offer a promising path forward: their embeddings capture latent relationships in a unifying way, enabling universal optimization across different data types possible. In this paper, we discuss multiple potential approaches, including an end-to-end learning framework in the form of next-token prediction, as well as prioritizing the learning of latent spaces with strong representational capabilities. To validate the effectiveness of these methods, we collect offline BBO tasks and data from open-source academic works for training. Experiments demonstrate the universality and effectiveness of our proposed methods. Our findings suggest that unifying language model priors and learning string embedding space can overcome traditional barriers in universal BBO, paving the way for general-purpose BBO algorithms. The code is provided at this https URL. 

**Abstract (ZH)**: 追求通用黑盒优化（BBO）算法是一个长期目标。然而，与语言或视觉领域相比，后者由于结构化数据的扩展而促进了泛化的提升，离线BBO的进步仍受限于无法统一表示异构数值空间。因此，现有的离线BBO方法仅适用于单任务和固定维度的设置，无法实现跨域通用优化。语言模型（LMs）的最近进展提供了前进的道路：它们的嵌入以统一的方式捕捉潜在关系，使不同数据类型上的通用优化成为可能。本文讨论了多种潜在方法，包括端到端学习框架（形式为下一个词预测）以及优先学习具有强表示能力的潜在空间。为了验证这些方法的有效性，我们从开源学术作品中收集了离线BBO任务和数据进行训练。实验结果表明我们提出的方法具有通用性和有效性。我们的研究发现，统一语言模型先验并学习字符串嵌入空间可以克服传统通用BBO的障碍，为通用目的的BBO算法铺平道路。代码可在以下链接处获取。 

---
# Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models 

**Title (ZH)**: 思想定理：一种语言模型中的 abduction、deduction 和 induction 推理多代理框架 

**Authors**: Samir Abdaljalil, Hasan Kurban, Khalid Qaraqe, Erchin Serpedin  

**Link**: [PDF](https://arxiv.org/pdf/2506.07106)  

**Abstract**: Large language models (LLMs) have shown strong performance across natural language reasoning tasks, yet their reasoning processes remain brittle and difficult to interpret. Prompting techniques like Chain-of-Thought (CoT) enhance reliability by eliciting intermediate reasoning steps or aggregating multiple outputs. However, they lack mechanisms for enforcing logical structure and assessing internal coherence. We introduce Theorem-of-Thought (ToTh), a novel framework that models reasoning as collaboration among three parallel agents, each simulating a distinct mode of inference: abductive, deductive, and inductive. Each agent produces a reasoning trace, which is structured into a formal reasoning graph. To evaluate consistency, we apply Bayesian belief propagation guided by natural language inference (NLI), assigning confidence scores to each step. The most coherent graph is selected to derive the final answer. Experiments on symbolic (WebOfLies) and numerical (MultiArith) reasoning benchmarks show that ToTh consistently outperforms CoT, Self-Consistency, and CoT-Decoding across multiple LLMs, while producing interpretable and logically grounded reasoning chains. Our findings suggest a promising direction for building more robust and cognitively inspired LLM reasoning. The implementation is available at this https URL. 

**Abstract (ZH)**: Theorem-of-Thought: A Novel Framework for Robust and Interpretable Reasoning in Large Language Models 

---
# How Far Are We from Optimal Reasoning Efficiency? 

**Title (ZH)**: 我们距最优推理效率还有多远？ 

**Authors**: Jiaxuan Gao, Shu Yan, Qixin Tan, Lu Yang, Shusheng Xu, Wei Fu, Zhiyu Mei, Kaifeng Lyu, Yi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07104)  

**Abstract**: Large Reasoning Models (LRMs) demonstrate remarkable problem-solving capabilities through extended Chain-of-Thought (CoT) reasoning but often produce excessively verbose and redundant reasoning traces. This inefficiency incurs high inference costs and limits practical deployment. While existing fine-tuning methods aim to improve reasoning efficiency, assessing their efficiency gains remains challenging due to inconsistent evaluations. In this work, we introduce the reasoning efficiency frontiers, empirical upper bounds derived from fine-tuning base LRMs across diverse approaches and training configurations. Based on these frontiers, we propose the Reasoning Efficiency Gap (REG), a unified metric quantifying deviations of any fine-tuned LRMs from these frontiers. Systematic evaluation on challenging mathematical benchmarks reveals significant gaps in current methods: they either sacrifice accuracy for short length or still remain inefficient under tight token budgets. To reduce the efficiency gap, we propose REO-RL, a class of Reinforcement Learning algorithms that minimizes REG by targeting a sparse set of token budgets. Leveraging numerical integration over strategically selected budgets, REO-RL approximates the full efficiency objective with low error using a small set of token budgets. Through systematic benchmarking, we demonstrate that our efficiency metric, REG, effectively captures the accuracy-length trade-off, with low-REG methods reducing length while maintaining accuracy. Our approach, REO-RL, consistently reduces REG by >=50 across all evaluated LRMs and matching Qwen3-4B/8B efficiency frontiers under a 16K token budget with minimal accuracy loss. Ablation studies confirm the effectiveness of our exponential token budget strategy. Finally, our findings highlight that fine-tuning LRMs to perfectly align with the efficiency frontiers remains an open challenge. 

**Abstract (ZH)**: 大型推理模型（LRMs）通过扩展的因果推理（CoT）展示了 remarkable 的问题解决能力，但常常产生冗余且多余的推理痕迹。这种低效率导致了较高的推理成本并限制了其实用部署。尽管现有的微调方法旨在提高推理效率，但对其效率增益的评估仍然具有挑战性，因为这些评估往往是一致性不高的。在本文中，我们引入了推理效率前沿，这是通过在不同方法和训练配置下微调基础LRMs而得出的经验上限。基于这些前沿，我们提出了推理效率差距（REG），这是一种统一的度量标准，用于量化任何微调LRMs与这些前沿之间的偏差。系统地在具有挑战性的数学基准上评估表明，当前的方法要么在保持短长度时牺牲精度，要么在严格的标记预算下仍然效率低下。为了减少效率差距，我们提出了REO-RL，这是一种通过目标稀疏标记预算来最小化REG的强化学习算法。通过在优选预算上进行数值积分，REO-RL 使用少量的标记预算以较低的误差逼近完整的效率目标。通过系统基准测试，我们证明了我们的效率度量REG能有效捕捉精度与长度之间的权衡关系，低REG方法在不降低精度的情况下减少了长度。我们的方法REO-RL在所有评估的LRMs上一致地减少了REG至少50%，并在16K标记预算下达到了Qwen3-4B/8B效率前沿，且具有最小的精度损失。消融研究证实了我们对指数标记预算策略的有效性。最后，我们的研究结果强调，将LRMs微调到完全符合效率前沿仍然是一个开放的挑战。 

---
# Filling the Missings: Spatiotemporal Data Imputation by Conditional Diffusion 

**Title (ZH)**: 填补缺失：基于条件扩散的时空数据插补 

**Authors**: Wenying He, Jieling Huang, Junhua Gu, Ji Zhang, Yude Bai  

**Link**: [PDF](https://arxiv.org/pdf/2506.07099)  

**Abstract**: Missing data in spatiotemporal systems presents a significant challenge for modern applications, ranging from environmental monitoring to urban traffic management. The integrity of spatiotemporal data often deteriorates due to hardware malfunctions and software failures in real-world deployments. Current approaches based on machine learning and deep learning struggle to model the intricate interdependencies between spatial and temporal dimensions effectively and, more importantly, suffer from cumulative errors during the data imputation process, which propagate and amplify through iterations. To address these limitations, we propose CoFILL, a novel Conditional Diffusion Model for spatiotemporal data imputation. CoFILL builds on the inherent advantages of diffusion models to generate high-quality imputations without relying on potentially error-prone prior estimates. It incorporates an innovative dual-stream architecture that processes temporal and frequency domain features in parallel. By fusing these complementary features, CoFILL captures both rapid fluctuations and underlying patterns in the data, which enables more robust imputation. The extensive experiments reveal that CoFILL's noise prediction network successfully transforms random noise into meaningful values that align with the true data distribution. The results also show that CoFILL outperforms state-of-the-art methods in imputation accuracy. The source code is publicly available at this https URL. 

**Abstract (ZH)**: 空间时序系统中的缺失数据为现代应用带来了重大挑战，范围从环境监测到城市交通管理。实际部署中，空间时序数据的完整性常因硬件故障和软件故障而恶化。基于机器学习和深度学习的现有方法难以有效地建模空间和时间维度之间的复杂相互依赖，并且更重要的是，在数据填充过程中会产生累积误差，这些误差会在迭代中传播和放大。为了解决这些局限性，我们提出了一种新颖的空间时序数据填充模型CoFILL，该模型是一种条件扩散模型。CoFILL 利用扩散模型的固有优势，生成高质量的填充结果，而不依赖于可能带有错误的先验估计。它采用了一种创新的双流架构，同时处理时间域和频域特征。通过融合这些互补特征，CoFILL 能捕获数据中的快速波动和潜在模式，从而实现更 robust 的填充。广泛的数据实验表明，CoFILL 的噪声预测网络成功地将随机噪声转化为与真实数据分布相一致的有意义的值。实验结果还显示，CoFILL 在填充准确性上优于现有最先进的方法。源代码已在该网址公开。 

---
# Patient Similarity Computation for Clinical Decision Support: An Efficient Use of Data Transformation, Combining Static and Time Series Data 

**Title (ZH)**: 临床决策支持中的患者相似性计算：一种高效的数据转换应用，结合静态和时间序列数据 

**Authors**: Joydeb Kumar Sana, Mohammad M. Masud, M Sohel Rahman, M Saifur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2506.07092)  

**Abstract**: Patient similarity computation (PSC) is a fundamental problem in healthcare informatics. The aim of the patient similarity computation is to measure the similarity among patients according to their historical clinical records, which helps to improve clinical decision support. This paper presents a novel distributed patient similarity computation (DPSC) technique based on data transformation (DT) methods, utilizing an effective combination of time series and static data. Time series data are sensor-collected patients' information, including metrics like heart rate, blood pressure, Oxygen saturation, respiration, etc. The static data are mainly patient background and demographic data, including age, weight, height, gender, etc. Static data has been used for clustering the patients. Before feeding the static data to the machine learning model adaptive Weight-of-Evidence (aWOE) and Z-score data transformation (DT) methods have been performed, which improve the prediction performances. In aWOE-based patient similarity models, sensitive patient information has been processed using aWOE which preserves the data privacy of the trained models. We used the Dynamic Time Warping (DTW) approach, which is robust and very popular, for time series similarity. However, DTW is not suitable for big data due to the significant computational run-time. To overcome this problem, distributed DTW computation is used in this study. For Coronary Artery Disease, our DT based approach boosts prediction performance by as much as 11.4%, 10.20%, and 12.6% in terms of AUC, accuracy, and F-measure, respectively. In the case of Congestive Heart Failure (CHF), our proposed method achieves performance enhancement up to 15.9%, 10.5%, and 21.9% for the same measures, respectively. The proposed method reduces the computation time by as high as 40%. 

**Abstract (ZH)**: 基于数据变换的分布式患者相似性计算方法 

---
# On the Generalization of Data-Assisted Control in port-Hamiltonian Systems (DAC-pH) 

**Title (ZH)**: 基于数据辅助控制的端口哈密尔顿系统泛化研究（DAC-pH） 

**Authors**: Mostafa Eslami, Maryam Babazadeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.07079)  

**Abstract**: This paper introduces a hypothetical hybrid control framework for port-Hamiltonian (p$\mathcal{H}$) systems, employing a dynamic decomposition based on Data-Assisted Control (DAC). The system's evolution is split into two parts with fixed topology: Right-Hand Side (RHS)- an intrinsic Hamiltonian flow handling worst-case parametric uncertainties, and Left-Hand Side (LHS)- a dissipative/input flow addressing both structural and parametric uncertainties. A virtual port variable $\Pi$ serves as the interface between these two components. A nonlinear controller manages the intrinsic Hamiltonian flow, determining a desired port control value $\Pi_c$. Concurrently, Reinforcement Learning (RL) is applied to the dissipative/input flow to learn an agent for providing optimal policy in mapping $\Pi_c$ to the actual system input. This hybrid approach effectively manages RHS uncertainties while preserving the system's inherent structure. Key advantages include adjustable performance via LHS controller parameters, enhanced AI explainability and interpretability through the port variable $\Pi$, the ability to guarantee safety and state attainability with hard/soft constraints, reduced complexity in learning hypothesis classes compared to end-to-end solutions, and improved state/parameter estimation using LHS prior knowledge and system Hamiltonian to address partial observability. The paper details the p$\mathcal{H}$ formulation, derives the decomposition, and presents the modular controller architecture. Beyond design, crucial aspects of stability and robustness analysis and synthesis are investigated, paving the way for deeper theoretical investigations. An application example, a pendulum with nonlinear dynamics, is simulated to demonstrate the approach's empirical and phenomenological benefits for future research. 

**Abstract (ZH)**: 一种基于数据辅助控制的港哈密尔顿系统混合控制框架 

---
# Dual-Priv Pruning : Efficient Differential Private Fine-Tuning in Multimodal Large Language Models 

**Title (ZH)**: Dual-Priv 剪枝：多模态大型语言模型中的高效差异隐私微调 

**Authors**: Qianshan Wei, Jiaqi Li, Zihan You, Yi Zhan, Kecen Li, Jialin Wu, Xinfeng Li Hengjun Liu, Yi Yu, Bin Cao, Yiwen Xu, Yang Liu, Guilin Qi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07077)  

**Abstract**: Differential Privacy (DP) is a widely adopted technique, valued for its effectiveness in protecting the privacy of task-specific datasets, making it a critical tool for large language models. However, its effectiveness in Multimodal Large Language Models (MLLMs) remains uncertain. Applying Differential Privacy (DP) inherently introduces substantial computation overhead, a concern particularly relevant for MLLMs which process extensive textual and visual data. Furthermore, a critical challenge of DP is that the injected noise, necessary for privacy, scales with parameter dimensionality, leading to pronounced model degradation; This trade-off between privacy and utility complicates the application of Differential Privacy (DP) to complex architectures like MLLMs. To address these, we propose Dual-Priv Pruning, a framework that employs two complementary pruning mechanisms for DP fine-tuning in MLLMs: (i) visual token pruning to reduce input dimensionality by removing redundant visual information, and (ii) gradient-update pruning during the DP optimization process. This second mechanism selectively prunes parameter updates based on the magnitude of noisy gradients, aiming to mitigate noise impact and improve utility. Experiments demonstrate that our approach achieves competitive results with minimal performance degradation. In terms of computational efficiency, our approach consistently utilizes less memory than standard DP-SGD. While requiring only 1.74% more memory than zeroth-order methods which suffer from severe performance issues on A100 GPUs, our method demonstrates leading memory efficiency on H20 GPUs. To the best of our knowledge, we are the first to explore DP fine-tuning in MLLMs. Our code is coming soon. 

**Abstract (ZH)**: 差分隐私(Differential Privacy, DP)是一种广泛采用的技术，因其在保护任务特定数据集隐私方面的有效性而备受重视，是大规模语言模型的重要工具。然而，其在多模态大规模语言模型(Multimodal Large Language Models, MLLMs)中的效果尚不确定。DP的引入不可避免地带来了显著的计算 overhead，这在处理大量文本和视觉数据的MLLMs中尤为相关。此外，DP的关键挑战在于为保护隐私而注入的噪声随参数维度增加而增大，导致模型性能明显下降；这种隐私与可用性之间的权衡使DP在复杂的MLLM架构中的应用变得复杂。为了解决这些问题，我们提出了一种名为Dual-Priv Pruning的框架，采用两种互补的剪枝机制进行MLLMs的DP微调：(i) 视觉token剪枝，通过移除冗余的视觉信息来减少输入维度，(ii) DP优化过程中的梯度更新剪枝。这一机制根据梯度噪声的大小选择性地剪枝参数更新，旨在减轻噪声影响并提高可用性。实验结果显示，我们的方法在性能下降最小的情况下达到了竞争力的性能。从计算效率来看，我们的方法始终比标准的DP-SGD使用更少的内存。虽然与A100 GPU上存在严重性能问题的零阶方法相比，我们的方法仅需多出1.74%的内存，但在H20 GPU上却展示了领先的内存效率。据我们所知，我们是第一个探索MLLMs的DP微调的研究。我们的代码即将发布。 

---
# From Axioms to Algorithms: Mechanized Proofs of the vNM Utility Theorem 

**Title (ZH)**: 从公理到算法：机器化证明的冯·诺伊曼-摩根索效用定理 

**Authors**: Li Jingyuan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07066)  

**Abstract**: This paper presents a comprehensive formalization of the von Neumann-Morgenstern (vNM) expected utility theorem using the Lean 4 interactive theorem prover. We implement the classical axioms of preference-completeness, transitivity, continuity, and independence-enabling machine-verified proofs of both the existence and uniqueness of utility representations. Our formalization captures the mathematical structure of preference relations over lotteries, verifying that preferences satisfying the vNM axioms can be represented by expected utility maximization.
Our contributions include a granular implementation of the independence axiom, formally verified proofs of fundamental claims about mixture lotteries, constructive demonstrations of utility existence, and computational experiments validating the results. We prove equivalence to classical presentations while offering greater precision at decision boundaries.
This formalization provides a rigorous foundation for applications in economic modeling, AI alignment, and management decision systems, bridging the gap between theoretical decision theory and computational implementation. 

**Abstract (ZH)**: 本文使用Lean 4交互式定理证明器对von Neumann-Morgenstern (vNM)期望效用定理进行了全面的形式化。我们实现了偏好完备性、传递性、连续性和独立性等经典偏好公理，并通过机器验证证明了这些公理的存在性和唯一性效用表示。我们的形式化捕捉了彩票上偏好关系的数学结构，验证了满足vNM公理的偏好可以由期望效用最大化来表示。

本文的贡献包括对独立性公理的细致实现、关于混合彩票的基本断言的形式验证证明、效用存在的建设性演示以及验证结果的计算实验。我们在决策边界上提供了更高精度的同时证明了与经典表述等价性。 

---
# Com$^2$: A Causal-Guided Benchmark for Exploring Complex Commonsense Reasoning in Large Language Models 

**Title (ZH)**: Com$^2$：一种因果引导的大规模语言模型复杂常识推理基准 

**Authors**: Kai Xiong, Xiao Ding, Yixin Cao, Yuxiong Yan, Li Du, Yufei Zhang, Jinglong Gao, Jiaqian Liu, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07064)  

**Abstract**: Large language models (LLMs) have mastered abundant simple and explicit commonsense knowledge through pre-training, enabling them to achieve human-like performance in simple commonsense reasoning. Nevertheless, LLMs struggle to reason with complex and implicit commonsense knowledge that is derived from simple ones (such as understanding the long-term effects of certain events), an aspect humans tend to focus on more. Existing works focus on complex tasks like math and code, while complex commonsense reasoning remains underexplored due to its uncertainty and lack of structure. To fill this gap and align with real-world concerns, we propose a benchmark Com$^2$ focusing on complex commonsense reasoning. We first incorporate causal event graphs to serve as structured complex commonsense. Then we adopt causal theory~(e.g., intervention) to modify the causal event graphs and obtain different scenarios that meet human concerns. Finally, an LLM is employed to synthesize examples with slow thinking, which is guided by the logical relationships in the modified causal graphs. Furthermore, we use detective stories to construct a more challenging subset. Experiments show that LLMs struggle in reasoning depth and breadth, while post-training and slow thinking can alleviate this. The code and data are available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通过预训练掌握了大量的简单明了的常识知识，使其在简单的常识推理中能够达到接近人类的性能。然而，LLMs 在处理源自简单常识的复杂和隐含的常识推理（如理解某些事件的长期影响）方面表现不佳，这是人类更加关注的方面。现有研究集中在数学和代码等复杂任务上，而复杂的常识推理由于其不确定性及缺乏结构而未被充分探索。为了填补这一空白并与实际需求相一致，我们提出了一项基于复杂常识推理的基准测试Com$^2$。我们首先引入因果事件图作为结构化的复杂常识。然后采用因果理论（如干预）修改因果事件图，以获得符合人类关注的不同场景。最后，应用LLM进行基于修改后的因果图逻辑关系的缓慢思考综合实例。此外，我们使用侦探故事构建更具挑战性的子集。实验结果显示，LLMs 在深度和广度推理方面表现不佳，但后训练和缓慢思考可以缓解这一问题。代码和数据可在以下链接获取。 

---
# Prime the search: Using large language models for guiding geometric task and motion planning by warm-starting tree search 

**Title (ZH)**: 预先填充搜索：使用大型语言模型引导几何任务和运动规划的树搜索预热方法 

**Authors**: Dongryung Lee, Sejune Joo, Kimin Lee, Beomjoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.07062)  

**Abstract**: The problem of relocating a set of objects to designated areas amidst movable obstacles can be framed as a Geometric Task and Motion Planning (G-TAMP) problem, a subclass of task and motion planning (TAMP). Traditional approaches to G-TAMP have relied either on domain-independent heuristics or on learning from planning experience to guide the search, both of which typically demand significant computational resources or data. In contrast, humans often use common sense to intuitively decide which objects to manipulate in G-TAMP problems. Inspired by this, we propose leveraging Large Language Models (LLMs), which have common sense knowledge acquired from internet-scale data, to guide task planning in G-TAMP problems. To enable LLMs to perform geometric reasoning, we design a predicate-based prompt that encodes geometric information derived from a motion planning algorithm. We then query the LLM to generate a task plan, which is then used to search for a feasible set of continuous parameters. Since LLMs are prone to mistakes, instead of committing to LLM's outputs, we extend Monte Carlo Tree Search (MCTS) to a hybrid action space and use the LLM to guide the search. Unlike the previous approach that calls an LLM at every node and incurs high computational costs, we use it to warm-start the MCTS with the nodes explored in completing the LLM's task plan. On six different G-TAMP problems, we show our method outperforms previous LLM planners and pure search algorithms. Code can be found at: this https URL 

**Abstract (ZH)**: 基于几何信息的大语言模型指导的任务与运动规划方法 

---
# Less is More: some Computational Principles based on Parcimony, and Limitations of Natural Intelligence 

**Title (ZH)**: 少就是多：基于简约的一些计算原理及自然智能的限制 

**Authors**: Laura Cohen, Xavier Hinaut, Lilyana Petrova, Alexandre Pitti, Syd Reynal, Ichiro Tsuda  

**Link**: [PDF](https://arxiv.org/pdf/2506.07060)  

**Abstract**: Natural intelligence (NI) consistently achieves more with less. Infants learn language, develop abstract concepts, and acquire sensorimotor skills from sparse data, all within tight neural and energy limits. In contrast, today's AI relies on virtually unlimited computational power, energy, and data to reach high performance. This paper argues that constraints in NI are paradoxically catalysts for efficiency, adaptability, and creativity. We first show how limited neural bandwidth promotes concise codes that still capture complex patterns. Spiking neurons, hierarchical structures, and symbolic-like representations emerge naturally from bandwidth constraints, enabling robust generalization. Next, we discuss chaotic itinerancy, illustrating how the brain transits among transient attractors to flexibly retrieve memories and manage uncertainty. We then highlight reservoir computing, where random projections facilitate rapid generalization from small datasets. Drawing on developmental perspectives, we emphasize how intrinsic motivation, along with responsive social environments, drives infant language learning and discovery of meaning. Such active, embodied processes are largely absent in current AI. Finally, we suggest that adopting 'less is more' principles -- energy constraints, parsimonious architectures, and real-world interaction -- can foster the emergence of more efficient, interpretable, and biologically grounded artificial systems. 

**Abstract (ZH)**: 自然智能（NI）始终以更少的资源实现更多。婴儿从少量数据中学习语言、发展抽象概念并获得运动感觉技能，这一切都在严格的神经和能量限制内完成。相比之下，当今的AI依靠几乎无限的计算能力、能源和数据来达到高性能。本文认为，NI的限制反而促进了效率、适应性和创造力。我们首先展示有限的神经带宽如何促成简洁的编码，但仍能捕捉到复杂模式。脉冲神经元、分层结构和类符号表示自然地从带宽限制中涌现，使系统具备稳健的泛化能力。接着，我们讨论混沌游动，说明大脑如何在瞬态吸引子之间过渡，以灵活地检索记忆和管理不确定性。然后，我们强调内在动机以及响应性的社会环境在婴儿语言学习和意义发现中的作用。这些主动的、具身的过程在当前的AI中基本上是不存在的。最后，我们建议采用“少即是多”的原则——能量约束、简约的架构和现实世界交互，以促进更高效、可解释且基于生物学的人工系统的生成。 

---
# Policy Gradient with Tree Search: Avoiding Local Optimas through Lookahead 

**Title (ZH)**: 基于树搜索的策略梯度：通过前瞻避免局部最优 

**Authors**: Uri Koren, Navdeep Kumar, Uri Gadot, Giorgia Ramponi, Kfir Yehuda Levy, Shie Mannor  

**Link**: [PDF](https://arxiv.org/pdf/2506.07054)  

**Abstract**: Classical policy gradient (PG) methods in reinforcement learning frequently converge to suboptimal local optima, a challenge exacerbated in large or complex environments. This work investigates Policy Gradient with Tree Search (PGTS), an approach that integrates an $m$-step lookahead mechanism to enhance policy optimization. We provide theoretical analysis demonstrating that increasing the tree search depth $m$-monotonically reduces the set of undesirable stationary points and, consequently, improves the worst-case performance of any resulting stationary policy. Critically, our analysis accommodates practical scenarios where policy updates are restricted to states visited by the current policy, rather than requiring updates across the entire state space. Empirical evaluations on diverse MDP structures, including Ladder, Tightrope, and Gridworld environments, illustrate PGTS's ability to exhibit "farsightedness," navigate challenging reward landscapes, escape local traps where standard PG fails, and achieve superior solutions. 

**Abstract (ZH)**: 经典策略梯度（PG）方法在强化学习中经常收敛到次优的局部最优解，这一问题在大型或复杂环境中尤为突出。本文研究了策略梯度与树搜索结合的方法（PGTS），该方法通过引入$m$步前瞻机制来增强策略优化。我们提供了理论分析，证明增加树搜索深度$m$可以单调减少不可取的稳态点集，从而提高任何由此产生的稳态策略的最坏情况性能。关键的是，我们的分析考虑了策略更新仅限于当前策略访问的状态，而不需要对整个状态空间进行更新。对于包括梯子环境、钢丝走绳环境和网格世界环境在内的多种MDP结构，实证评估展示了PGTS能够表现出“远见卓识”，导航复杂的奖励景观，避开标准PG方法失败的局部陷阱，并取得更优的解。 

---
# Interpretable and Reliable Detection of AI-Generated Images via Grounded Reasoning in MLLMs 

**Title (ZH)**: 基于MLLMs的 grounded reasoning 的可解释和可靠检测方法：识别AI生成的图像 

**Authors**: Yikun Ji, Hong Yan, Jun Lan, Huijia Zhu, Weiqiang Wang, Qi Fan, Liqing Zhang, Jianfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07045)  

**Abstract**: The rapid advancement of image generation technologies intensifies the demand for interpretable and robust detection methods. Although existing approaches often attain high accuracy, they typically operate as black boxes without providing human-understandable justifications. Multi-modal Large Language Models (MLLMs), while not originally intended for forgery detection, exhibit strong analytical and reasoning capabilities. When properly fine-tuned, they can effectively identify AI-generated images and offer meaningful explanations. However, existing MLLMs still struggle with hallucination and often fail to align their visual interpretations with actual image content and human reasoning. To bridge this gap, we construct a dataset of AI-generated images annotated with bounding boxes and descriptive captions that highlight synthesis artifacts, establishing a foundation for human-aligned visual-textual grounded reasoning. We then finetune MLLMs through a multi-stage optimization strategy that progressively balances the objectives of accurate detection, visual localization, and coherent textual explanation. The resulting model achieves superior performance in both detecting AI-generated images and localizing visual flaws, significantly outperforming baseline methods. 

**Abstract (ZH)**: 图像生成技术的迅速发展加剧了对可解释和稳健检测方法的需求。尽管现有方法通常能获得高准确率，但它们通常作为黑盒操作，无法提供人类可理解的解释。多模态大型语言模型（MLLMs）虽然最初并非设计用于检测伪造，但展现出强大的分析与推理能力。通过适当微调，它们可以有效识别AI生成的图像并提供有意义的解释。然而，现有的MLLMs仍然难以避免生成幻觉，并常无法与其视觉解释和实际图像内容及人类推理保持一致。为此，我们构建了一个标注有边界框和描述性注释的AI生成图像数据集，突显合成痕迹，为人类对齐的视觉-文本推理奠定了基础。随后，我们通过多阶段优化策略微调MLLMs，逐步平衡准确检测、视觉定位和一致文本解释的目标。最终模型在检测AI生成图像和定位视觉缺陷方面表现出色，显著优于基线方法。 

---
# Lingshu: A Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning 

**Title (ZH)**: 灵枢：统一多模态医疗理解与推理的通用基础模型 

**Authors**: LASA Team, Weiwen Xu, Hou Pong Chan, Long Li, Mahani Aljunied, Ruifeng Yuan, Jianyu Wang, Chenghao Xiao, Guizhen Chen, Chaoqun Liu, Zhaodonghui Li, Yu Sun, Junao Shen, Chaojun Wang, Jie Tan, Deli Zhao, Tingyang Xu, Hao Zhang, Yu Rong  

**Link**: [PDF](https://arxiv.org/pdf/2506.07044)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in understanding common visual elements, largely due to their large-scale datasets and advanced training strategies. However, their effectiveness in medical applications remains limited due to the inherent discrepancies between data and tasks in medical scenarios and those in the general domain. Concretely, existing medical MLLMs face the following critical limitations: (1) limited coverage of medical knowledge beyond imaging, (2) heightened susceptibility to hallucinations due to suboptimal data curation processes, (3) lack of reasoning capabilities tailored for complex medical scenarios. To address these challenges, we first propose a comprehensive data curation procedure that (1) efficiently acquires rich medical knowledge data not only from medical imaging but also from extensive medical texts and general-domain data; and (2) synthesizes accurate medical captions, visual question answering (VQA), and reasoning samples. As a result, we build a multimodal dataset enriched with extensive medical knowledge. Building on the curated data, we introduce our medical-specialized MLLM: Lingshu. Lingshu undergoes multi-stage training to embed medical expertise and enhance its task-solving capabilities progressively. Besides, we preliminarily explore the potential of applying reinforcement learning with verifiable rewards paradigm to enhance Lingshu's medical reasoning ability. Additionally, we develop MedEvalKit, a unified evaluation framework that consolidates leading multimodal and textual medical benchmarks for standardized, fair, and efficient model assessment. We evaluate the performance of Lingshu on three fundamental medical tasks, multimodal QA, text-based QA, and medical report generation. The results show that Lingshu consistently outperforms the existing open-source multimodal models on most tasks ... 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在理解常见视觉元素方面展示了令人印象深刻的能 力，主要归功于它们庞大的数据集和先进的训练策略。然而，它们在医疗应用 中的效果仍然有限，主要是由于医疗场景与通用领域之间数据和任务的固有差 异。具体而言，现有的医疗 MLLMs 面临以下关键限制：（1）超出成像之外的医 学知识覆盖范围有限；（2）由于次优的数据编目过程容易产生幻觉；（3）缺乏 为复杂医疗场景设计的推理能力。为应对这些挑战，我们首先提出了一种全面 的数据编目程序，该程序（1）不仅从医学成像，还从广泛的医学文本和通用领 域数据中高效获取丰富医学知识数据；（2）合成准确的医学图 caption、视觉 答疑（VQA）和推理样本。因此，我们构建了一个富含广泛医学知识的多模态数 �据集。基于编目数据，我们提出了我们的医学专用 MLLM：灵枢。灵枢分阶段 接受训练，逐步嵌入医学专业知识并提升其任务解决能力。此外，我们初步探 索将验证性奖励范式的强化学习应用于增强灵枢的医疗推理能力。我们还开发 了 MedEvalKit，这是一个统一的评估框架，整合了领先多模态和文本医学基准 以实现标准化、公平和高效的模型评估。我们在三种基本医疗任务：多模态答 题、文本基回答题和医学报告生成上评估了灵枢的性能。结果显示，灵枢在大 多数任务上持续优于现有的开源多模态模型…… 

---
# Efficient $Q$-Learning and Actor-Critic Methods for Robust Average Reward Reinforcement Learning 

**Title (ZH)**: 高效的 $Q$-学习与演员-评论家方法在鲁棒平均奖励强化学习中的应用 

**Authors**: Yang Xu, Swetha Ganesh, Vaneet Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2506.07040)  

**Abstract**: We present the first $Q$-learning and actor-critic algorithms for robust average reward Markov Decision Processes (MDPs) with non-asymptotic convergence under contamination, TV distance and Wasserstein distance uncertainty sets. We show that the robust $Q$ Bellman operator is a strict contractive mapping with respect to a carefully constructed semi-norm with constant functions being quotiented out. This property supports a stochastic approximation update, that learns the optimal robust $Q$ function in $\tilde{\cO}(\epsilon^{-2})$ samples. We also show that the same idea can be used for robust $Q$ function estimation, which can be further used for critic estimation. Coupling it with theories in robust policy mirror descent update, we present a natural actor-critic algorithm that attains an $\epsilon$-optimal robust policy in $\tilde{\cO}(\epsilon^{-3})$ samples. These results advance the theory of distributionally robust reinforcement learning in the average reward setting. 

**Abstract (ZH)**: 我们提出了首个针对受到污染、基于TV距离和Wasserstein距离不确定性集的鲁棒平均奖励马尔可夫决策过程($Q$-学习和演员-评论家)算法，并证明了鲁棒$Q$贝尔曼算子在特定半范数下为严格收缩映射（恒函数除外），这支持了一种随机逼近更新方法，能够在$\tilde{\cO}(\epsilon^{-2})$样本中学习最优鲁棒$Q$函数。我们还展示了相同的思想可以用于鲁棒$Q$函数估计，进而可用于评论家估计。结合鲁棒政策镜像下降更新理论，我们提出了一种自然的演员-评论家算法，能够在$\tilde{\cO}(\epsilon^{-3})$样本中获得$\epsilon$-最优鲁棒策略。这些结果推进了平均奖励设置下分布鲁棒强化学习的理论。 

---
# AnnoDPO: Protein Functional Annotation Learning with Direct Preference Optimization 

**Title (ZH)**: AnnoDPO: 蛋白质功能注释学习与直接偏好优化 

**Authors**: Zixuan Jiang, Renjing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07035)  

**Abstract**: Deciphering protein function remains a fundamental challenge in protein representation learning. The task presents significant difficulties for protein language models (PLMs) due to the sheer volume of functional annotation categories and the highly imbalanced distribution of annotated instances across biological ontologies. Inspired by the remarkable success of reinforcement learning from human feedback (RLHF) in large language model (LLM) alignment, we propose AnnoDPO, a novel multi-modal framework for protein function prediction that leverages Direct Preference Optimization (DPO) to enhance annotation learning. Our methodology addresses the dual challenges of annotation scarcity and category imbalance through preference-aligned training objectives, establishing a new paradigm for biological knowledge integration in protein representation learning. 

**Abstract (ZH)**: 解析蛋白质功能仍然是蛋白质表示学习中的一个根本挑战。受大规模语言模型（LLM）对人类反馈强化学习（RLHF）卓越成功的启发，我们提出了一种名为AnnoDPO的新型多模式框架，利用直接偏好优化（DPO）增强标注学习。我们的方法通过偏好对齐的训练目标来应对标注稀缺性和类别不平衡的双重挑战，建立了蛋白质表示学习中生物知识集成的新范式。 

---
# HauntAttack: When Attack Follows Reasoning as a Shadow 

**Title (ZH)**: 封掍攻击：当攻击如影随形跟随推理 

**Authors**: Jingyuan Ma, Rui Li, Zheng Li, Junfeng Liu, Lei Sha, Zhifang Sui  

**Link**: [PDF](https://arxiv.org/pdf/2506.07031)  

**Abstract**: Emerging Large Reasoning Models (LRMs) consistently excel in mathematical and reasoning tasks, showcasing exceptional capabilities. However, the enhancement of reasoning abilities and the exposure of their internal reasoning processes introduce new safety vulnerabilities. One intriguing concern is: when reasoning is strongly entangled with harmfulness, what safety-reasoning trade-off do LRMs exhibit? To address this issue, we introduce HauntAttack, a novel and general-purpose black-box attack framework that systematically embeds harmful instructions into reasoning questions. Specifically, we treat reasoning questions as carriers and substitute one of their original conditions with a harmful instruction. This process creates a reasoning pathway in which the model is guided step by step toward generating unsafe outputs. Based on HauntAttack, we conduct comprehensive experiments on multiple LRMs. Our results reveal that even the most advanced LRMs exhibit significant safety vulnerabilities. Additionally, we perform a detailed analysis of different models, various types of harmful instructions, and model output patterns, providing valuable insights into the security of LRMs. 

**Abstract (ZH)**: 新兴大规模推理模型（LRMs）在数学和推理任务中表现出色，展现出卓越的能力。然而，推理能力的增强和内部推理过程的暴露引入了新的安全漏洞。一个引人关注的问题是：当推理与有害性紧密结合时，LRMs展示出什么样的安全-推理权衡？为应对这一问题，我们引入了一种新颖且通用的黑盒攻击框架HauntAttack，该框架系统地将有害指令嵌入到推理问题中。具体而言，我们将推理问题视为载体，并将其原始条件之一替换为有害指令。这一过程创建了一条推理路径，在这条路径上，模型被逐步引导生成不安全的输出。基于HauntAttack，我们在多种LRMs上进行了全面的实验。我们的结果表明，即使是最先进的LRMs也存在显著的安全漏洞。此外，我们对不同模型、各种类型的有害指令以及模型输出模式进行了详细分析，为LRMs的安全性提供了宝贵的洞见。 

---
# SiliCoN: Simultaneous Nuclei Segmentation and Color Normalization of Histological Images 

**Title (ZH)**: SiliCoN: 组织图像的细胞分割和颜色归一化同时进行 

**Authors**: Suman Mahapatra, Pradipta Maji  

**Link**: [PDF](https://arxiv.org/pdf/2506.07028)  

**Abstract**: Segmentation of nuclei regions from histological images is an important task for automated computer-aided analysis of histological images, particularly in the presence of impermissible color variation in the color appearance of stained tissue images. While color normalization enables better nuclei segmentation, accurate segmentation of nuclei structures makes color normalization rather trivial. In this respect, the paper proposes a novel deep generative model for simultaneously segmenting nuclei structures and normalizing color appearance of stained histological this http URL model judiciously integrates the merits of truncated normal distribution and spatial attention. The model assumes that the latent color appearance information, corresponding to a particular histological image, is independent of respective nuclei segmentation map as well as embedding map information. The disentangled representation makes the model generalizable and adaptable as the modification or loss in color appearance information cannot be able to affect the nuclei segmentation map as well as embedding information. Also, for dealing with the stain overlap of associated histochemical reagents, the prior for latent color appearance code is assumed to be a mixture of truncated normal distributions. The proposed model incorporates the concept of spatial attention for segmentation of nuclei regions from histological images. The performance of the proposed approach, along with a comparative analysis with related state-of-the-art algorithms, has been demonstrated on publicly available standard histological image data sets. 

**Abstract (ZH)**: 从组织病理图像中分割核区域是自动化计算机辅助分析组织病理图像的重要任务，特别是在染色组织图像颜色外观存在不可接受的颜色变化时。虽然颜色归一化能够提高核分割的效果，但准确的核结构分割使得颜色归一化变得相对简单。基于此，本文提出了一种新颖的深度生成模型，用于同时分割核结构并归一化染色组织病理图像的颜色外观。该模型巧妙地结合了截断正态分布和空间注意力的优势。模型假设特定组织病理图像对应的潜在颜色外观信息与相应的核分割图和嵌入图信息独立。解耦的表示使得模型具有较强的泛化能力和适应性，因为颜色外观信息的变化或损失不会影响核分割图和嵌入信息。此外，为处理相关组织化学试剂的染色重叠问题，潜在颜色外观代码的先验假设为截断正态分布的混合。所提模型引入了空间注意力的概念，用于从组织病理图像中分割核区域。文中在公开的标准组织病理图像数据集上展示了所提出方法的性能，并与相关最先进的算法进行了比较分析。 

---
# Optimal Transport Driven Asymmetric Image-to-Image Translation for Nuclei Segmentation of Histological Images 

**Title (ZH)**: 由最优运输驱动的不对称图像到图像翻译在组织学图像细胞核分割中的应用 

**Authors**: Suman Mahapatra, Pradipta Maji  

**Link**: [PDF](https://arxiv.org/pdf/2506.07023)  

**Abstract**: Segmentation of nuclei regions from histological images enables morphometric analysis of nuclei structures, which in turn helps in the detection and diagnosis of diseases under consideration. To develop a nuclei segmentation algorithm, applicable to different types of target domain representations, image-to-image translation networks can be considered as they are invariant to target domain image representations. One of the important issues with image-to-image translation models is that they fail miserably when the information content between two image domains are asymmetric in nature. In this regard, the paper introduces a new deep generative model for segmenting nuclei structures from histological images. The proposed model considers an embedding space for handling information-disparity between information-rich histological image space and information-poor segmentation map domain. Integrating judiciously the concepts of optimal transport and measure theory, the model develops an invertible generator, which provides an efficient optimization framework with lower network complexity. The concept of invertible generator automatically eliminates the need of any explicit cycle-consistency loss. The proposed model also introduces a spatially-constrained squeeze operation within the framework of invertible generator to maintain spatial continuity within the image patches. The model provides a better trade-off between network complexity and model performance compared to other existing models having complex network architectures. The performance of the proposed deep generative model, along with a comparison with state-of-the-art nuclei segmentation methods, is demonstrated on publicly available histological image data sets. 

**Abstract (ZH)**: 基于图像到图像翻译网络的深度生成模型在组织学图像中核区段分割中的应用 

---
# AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint 

**Title (ZH)**: AlphaSteer: 学习基于合理空问约束的拒绝转向 

**Authors**: Leheng Sheng, Changshuo Shen, Weixiang Zhao, Junfeng Fang, Xiaohao Liu, Zhenkai Liang, Xiang Wang, An Zhang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.07022)  

**Abstract**: As LLMs are increasingly deployed in real-world applications, ensuring their ability to refuse malicious prompts, especially jailbreak attacks, is essential for safe and reliable use. Recently, activation steering has emerged as an effective approach for enhancing LLM safety by adding a refusal direction vector to internal activations of LLMs during inference, which will further induce the refusal behaviors of LLMs. However, indiscriminately applying activation steering fundamentally suffers from the trade-off between safety and utility, since the same steering vector can also lead to over-refusal and degraded performance on benign prompts. Although prior efforts, such as vector calibration and conditional steering, have attempted to mitigate this trade-off, their lack of theoretical grounding limits their robustness and effectiveness. To better address the trade-off between safety and utility, we present a theoretically grounded and empirically effective activation steering method called AlphaSteer. Specifically, it considers activation steering as a learnable process with two principled learning objectives: utility preservation and safety enhancement. For utility preservation, it learns to construct a nearly zero vector for steering benign data, with the null-space constraints. For safety enhancement, it learns to construct a refusal direction vector for steering malicious data, with the help of linear regression. Experiments across multiple jailbreak attacks and utility benchmarks demonstrate the effectiveness of AlphaSteer, which significantly improves the safety of LLMs without compromising general capabilities. Our codes are available at this https URL. 

**Abstract (ZH)**: 作为大型语言模型在现实应用场景中越来越广泛，确保其能够拒绝恶意提示，特别是防止 jailbreak 攻击，对于安全和可靠使用至关重要。最近，激活导向作为一种有效的方法出现，通过在推理过程中向大型语言模型的内部激活添加一个拒绝方向向量来增强其安全性，这将进一步诱导模型的拒绝行为。然而，不分青红皂白地应用激活导向在本质上会带来安全性和实用性之间的权衡，因为相同的导向向量也可能导致过度拒绝并对良性提示产生性能退化。尽管先前的努力，如向量校准和有条件导向，已经尝试缓解这一权衡，但由于缺乏理论依据，它们的有效性和鲁棒性有限。为了更好地解决安全性和实用性之间的权衡，我们提出了一个理论上和实验上都有效的激活导向方法，称为 AlphaSteer。具体而言，它将激活导向视为一个可学习的过程，并具有两个基本原则的学习目标：保持实用性和增强安全性。为了保持实用性，它学习构造一个接近零的导向向量来引导良性数据，受零空间约束的影响。为了增强安全性，它学习构造一个拒绝方向向量来引导恶意数据，借助线性回归的帮助。在多种 jailbreak 攻击和实用性的基准测试中，AlphaSteer 的效果得到了验证，它显著提高了大型语言模型的安全性，而不损害其通用能力。我们的代码可在以下链接访问：此 https URL。 

---
# MAGNET: A Multi-agent Framework for Finding Audio-Visual Needles by Reasoning over Multi-Video Haystacks 

**Title (ZH)**: MAGNET：一种基于多视频 haystack 原材料进行推理查找音视频 needles 的多智能体框架 

**Authors**: Sanjoy Chowdhury, Mohamed Elmoghany, Yohan Abeysinghe, Junjie Fei, Sayan Nag, Salman Khan, Mohamed Elhoseiny, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2506.07016)  

**Abstract**: Large multimodal models (LMMs) have shown remarkable progress in audio-visual understanding, yet they struggle with real-world scenarios that require complex reasoning across extensive video collections. Existing benchmarks for video question answering remain limited in scope, typically involving one clip per query, which falls short of representing the challenges of large-scale, audio-visual retrieval and reasoning encountered in practical applications. To bridge this gap, we introduce a novel task named AV-HaystacksQA, where the goal is to identify salient segments across different videos in response to a query and link them together to generate the most informative answer. To this end, we present AVHaystacks, an audio-visual benchmark comprising 3100 annotated QA pairs designed to assess the capabilities of LMMs in multi-video retrieval and temporal grounding task. Additionally, we propose a model-agnostic, multi-agent framework MAGNET to address this challenge, achieving up to 89% and 65% relative improvements over baseline methods on BLEU@4 and GPT evaluation scores in QA task on our proposed AVHaystacks. To enable robust evaluation of multi-video retrieval and temporal grounding for optimal response generation, we introduce two new metrics, STEM, which captures alignment errors between a ground truth and a predicted step sequence and MTGS, to facilitate balanced and interpretable evaluation of segment-level grounding performance. Project: this https URL 

**Abstract (ZH)**: 大规模多模态模型在音频-视觉理解方面取得了显著进展，但在处理需要在大量视频集合之间进行复杂推理的实际场景时仍存在挑战。现有的视频问答基准在范围上仍然有限，通常每个查询只涉及一个片段，这远不足以代表实际应用中大规模、音频-视觉检索和推理所面临的挑战。为了解决这一问题，我们引入了一个名为AV-HaystacksQA的新任务，目标是在响应查询时识别不同视频中的关键片段，并将它们链接起来生成最informative的答案。为此，我们提出了一个包含3100个标注问答对的AVHaystacks音频-视觉基准，旨在评估大规模多模态模型在多视频检索和时间定位任务中的能力。此外，我们提出了一种模型无关的多智能体框架MAGNET来解决这一挑战，该框架在我们提出的AVHaystacks基准上提出的问答任务中，相对于基线方法在BLEU@4和GPT评估得分上分别取得了高达89%和65%的相对改进。为了确保多视频检索和时间定位的稳健评估，以便生成最优响应，我们引入了两个新的度量标准STEM，用于捕捉预测步骤序列与真实步骤序列之间的对齐错误，以及MTGS，以促进段级定位性能的平衡和可解释评估。 

---
# Deep regularization networks for inverse problems with noisy operators 

**Title (ZH)**: 噪声算子下的深度正则化网络 

**Authors**: Fatemeh Pourahmadian, Yang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07008)  

**Abstract**: A supervised learning approach is proposed for regularization of large inverse problems where the main operator is built from noisy data. This is germane to superresolution imaging via the sampling indicators of the inverse scattering theory. We aim to accelerate the spatiotemporal regularization process for this class of inverse problems to enable real-time imaging. In this approach, a neural operator maps each pattern on the right-hand side of the scattering equation to its affiliated regularization parameter. The network is trained in two steps which entails: (1) training on low-resolution regularization maps furnished by the Morozov discrepancy principle with nonoptimal thresholds, and (2) optimizing network predictions through minimization of the Tikhonov loss function regulated by the validation loss. Step 2 allows for tailoring of the approximate maps of Step 1 toward construction of higher quality images. This approach enables direct learning from test data and dispenses with the need for a-priori knowledge of the optimal regularization maps. The network, trained on low-resolution data, quickly generates dense regularization maps for high-resolution imaging. We highlight the importance of the training loss function on the network's generalizability. In particular, we demonstrate that networks informed by the logic of discrepancy principle lead to images of higher contrast. In this case, the training process involves many-objective optimization. We propose a new method to adaptively select the appropriate loss weights during training without requiring an additional optimization process. The proposed approach is synthetically examined for imaging damage evolution in an elastic plate. The results indicate that the discrepancy-informed regularization networks not only accelerate the imaging process, but also remarkably enhance the image quality in complex environments. 

**Abstract (ZH)**: 一种监督学习方法用于基于 noisy 数据构建的主要算子的大规模逆问题正则化，特别是在逆散射理论的采样指标下实现超分辨率成像。我们旨在通过该方法加速此类逆问题的空间-时间正则化过程，以实现实时成像。在此方法中，神经算子将散射方程右侧的每个模式映射为其相应的正则化参数。该网络分为两步进行训练：（1）使用 Morozov 矛盾原理提供的非最优阈值的低分辨率正则化地图进行训练；（2）通过最小化由验证损失调节的泰特kon夫损失函数优化网络预测。第二步使得第一步的近似地图能够更好地构建高质量图像。此方法能够直接从测试数据中学习，并不需要先验知道最优正则化地图。该网络在低分辨率数据上训练，能够迅速生成密集的正则化地图以实现高分辨率成像。我们强调训练损失函数对网络泛化能力的重要性。特别是，我们证明基于矛盾原理逻辑训练的网络能够产生更高对比度的图像。在这种情况下，训练过程涉及多目标优化。我们提出了一种新方法，在训练过程中自适应选择适当的损失权重，而不需额外的优化过程。所提出的方法已在弹性板中损伤演化成像的合成实验中进行了验证。结果表明，基于矛盾原理的正则化网络不仅能加速成像过程，还在复杂环境中显著提高图像质量。 

---
# CARoL: Context-aware Adaptation for Robot Learning 

**Title (ZH)**: CARoL: 基于上下文的机器人学习适应机制 

**Authors**: Zechen Hu, Tong Xu, Xuesu Xiao, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07006)  

**Abstract**: Using Reinforcement Learning (RL) to learn new robotic tasks from scratch is often inefficient. Leveraging prior knowledge has the potential to significantly enhance learning efficiency, which, however, raises two critical challenges: how to determine the relevancy of existing knowledge and how to adaptively integrate them into learning a new task. In this paper, we propose Context-aware Adaptation for Robot Learning (CARoL), a novel framework to efficiently learn a similar but distinct new task from prior knowledge. CARoL incorporates context awareness by analyzing state transitions in system dynamics to identify similarities between the new task and prior knowledge. It then utilizes these identified similarities to prioritize and adapt specific knowledge pieces for the new task. Additionally, CARoL has a broad applicability spanning policy-based, value-based, and actor-critic RL algorithms. We validate the efficiency and generalizability of CARoL on both simulated robotic platforms and physical ground vehicles. The simulations include CarRacing and LunarLander environments, where CARoL demonstrates faster convergence and higher rewards when learning policies for new tasks. In real-world experiments, we show that CARoL enables a ground vehicle to quickly and efficiently adapt policies learned in simulation to smoothly traverse real-world off-road terrain. 

**Abstract (ZH)**: 基于上下文的适应性机器人学习（CARoL）：从先验知识高效学习新任务 

---
# End-to-End Probabilistic Framework for Learning with Hard Constraints 

**Title (ZH)**: 面向硬约束的端到端概率框架 

**Authors**: Utkarsh Utkarsh, Danielle C. Maddix, Ruijun Ma, Michael W. Mahoney, Yuyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07003)  

**Abstract**: We present a general purpose probabilistic forecasting framework, ProbHardE2E, to learn systems that can incorporate operational/physical constraints as hard requirements. ProbHardE2E enforces hard constraints by exploiting variance information in a novel way; and thus it is also capable of performing uncertainty quantification (UQ) on the model. Our methodology uses a novel differentiable probabilistic projection layer (DPPL) that can be combined with a wide range of neural network architectures. This DPPL allows the model to learn the system in an end-to-end manner, compared to other approaches where the constraints are satisfied either through a post-processing step or at inference. In addition, ProbHardE2E can optimize a strictly proper scoring rule, without making any distributional assumptions on the target, which enables it to obtain robust distributional estimates (in contrast to existing approaches that generally optimize likelihood-based objectives, which are heavily biased by their distributional assumptions and model choices); and it can incorporate a range of non-linear constraints (increasing the power of modeling and flexibility). We apply ProbHardE2E to problems in learning partial differential equations with uncertainty estimates and to probabilistic time-series forecasting, showcasing it as a broadly applicable general setup that connects these seemingly disparate domains. 

**Abstract (ZH)**: ProbHardE2E：一种整合操作/物理约束的通用概率预测框架 

---
# Towards Physics-informed Diffusion for Anomaly Detection in Trajectories 

**Title (ZH)**: 面向物理约束扩散的轨迹异常检测 

**Authors**: Arun Sharma, Mingzhou Yang, Majid Farhadloo, Subhankar Ghosh, Bharat Jayaprakash, Shashi Shekhar  

**Link**: [PDF](https://arxiv.org/pdf/2506.06999)  

**Abstract**: Given trajectory data, a domain-specific study area, and a user-defined threshold, we aim to find anomalous trajectories indicative of possible GPS spoofing (e.g., fake trajectory). The problem is societally important to curb illegal activities in international waters, such as unauthorized fishing and illicit oil transfers. The problem is challenging due to advances in AI generated in deep fakes generation (e.g., additive noise, fake trajectories) and lack of adequate amount of labeled samples for ground-truth verification. Recent literature shows promising results for anomalous trajectory detection using generative models despite data sparsity. However, they do not consider fine-scale spatiotemporal dependencies and prior physical knowledge, resulting in higher false-positive rates. To address these limitations, we propose a physics-informed diffusion model that integrates kinematic constraints to identify trajectories that do not adhere to physical laws. Experimental results on real-world datasets in the maritime and urban domains show that the proposed framework results in higher prediction accuracy and lower estimation error rate for anomaly detection and trajectory generation methods, respectively. Our implementation is available at this https URL. 

**Abstract (ZH)**: 给定轨迹数据、特定研究区域和用户定义的阈值，我们旨在找到指示可能的GPS欺骗（例如，虚假轨迹）的异常轨迹。该问题对于遏制国际海域中的非法活动（如未授权捕鱼和非法油品转运）具有社会重要性。由于在深度假信息生成（例如，添加噪声、虚假轨迹）方面AI的进展以及缺乏足够的标注样本进行真实情况验证，这一问题具有挑战性。近期文献表明，尽管存在数据稀疏问题，生成模型在异常轨迹检测方面仍显示出有希望的结果。然而，它们并未考虑细粒度的空间-时间依赖性和先前的物理知识，导致较高的误报率。为解决这些局限性，我们提出了一种基于物理的扩散模型，该模型整合了运动约束以识别不符合物理定律的轨迹。在海洋和城市领域的实际数据集上的实验结果显示，提出的框架分别在异常检测和轨迹生成方法中提高了预测准确性和降低了估计误差率。我们的实现可通过以下网址访问：this https URL。 

---
# What makes Reasoning Models Different? Follow the Reasoning Leader for Efficient Decoding 

**Title (ZH)**: 什么是推理模型的不同之处？跟随推理领导者进行高效解码 

**Authors**: Ming Li, Zhengyuan Yang, Xiyao Wang, Dianqi Li, Kevin Lin, Tianyi Zhou, Lijuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06998)  

**Abstract**: Large reasoning models (LRMs) achieve strong reasoning performance by emitting long chains of thought. Yet, these verbose traces slow down inference and often drift into unnecessary detail, known as the overthinking phenomenon. To better understand LRMs' behavior, we systematically analyze the token-level misalignment between reasoning and non-reasoning models. While it is expected that their primary difference lies in the stylistic "thinking cues", LRMs uniquely exhibit two pivotal, previously under-explored phenomena: a Global Misalignment Rebound, where their divergence from non-reasoning models persists or even grows as response length increases, and more critically, a Local Misalignment Diminish, where the misalignment concentrates at the "thinking cues" each sentence starts with but rapidly declines in the remaining of the sentence. Motivated by the Local Misalignment Diminish, we propose FoReaL-Decoding, a collaborative fast-slow thinking decoding method for cost-quality trade-off. In FoReaL-Decoding, a Leading model leads the first few tokens for each sentence, and then a weaker draft model completes the following tokens to the end of each sentence. FoReaL-Decoding adopts a stochastic gate to smoothly interpolate between the small and the large model. On four popular math-reasoning benchmarks (AIME24, GPQA-Diamond, MATH500, AMC23), FoReaL-Decoding reduces theoretical FLOPs by 30 to 50% and trims CoT length by up to 40%, while preserving 86 to 100% of model performance. These results establish FoReaL-Decoding as a simple, plug-and-play route to controllable cost-quality trade-offs in reasoning-centric tasks. 

**Abstract (ZH)**: 大型推理模型 (LRMs) 通过生成长链条的思考过程实现强大的推理性能。然而，这些冗长的推理痕迹会减慢推理速度，并且经常陷入不必要的细节，这种现象被称为过度推理。为了更好地理解 LRMs 的行为，我们系统地分析了推理模型和非推理模型在 token 级别上的不对齐。虽然预期它们的主要差异在于风格化的“思考提示”，但 LRMs 唯一地表现出两个以往被忽视的关键现象：全局不对齐反弹，即它们与非推理模型的差异在响应长度增加时持续存在甚至增大；更重要的是局部不对齐消减，即不对齐集中于每个句子开头的“思考提示”，但在句子其余部分迅速下降。受局部不对齐消减的启发，我们提出了一种协作的快慢思考解码方法 FoReaL-Decoding，用于成本-质量权衡。在 FoReaL-Decoding 中，领先模型主导每个句子前几个 token 的生成，然后较弱的草稿模型完成其余 token 的生成。FoReaL-Decoding 采用随机门将小型模型和大型模型平滑地结合起来。在四个流行的数学推理基准测试（AIME24、GPQA-Diamond、MATH500、AMC23）上，FoReaL-Decoding 将理论 FLOPs 减少 30% 至 50%，将 CoT 长度缩短至多 40%，同时保留 86% 至 100% 的模型性能。这些结果确立了 FoReaL-Decoding 作为控制性成本-质量权衡任务的简单、即插即用的方法。 

---
# MoXGATE: Modality-aware cross-attention for multi-omic gastrointestinal cancer sub-type classification 

**Title (ZH)**: MoXGATE: 融合模态aware跨注意力机制的多组学胃肠癌亚型分类 

**Authors**: Sajib Acharjee Dip, Uddip Acharjee Shuvo, Dipanwita Mallick, Abrar Rahman Abir, Liqing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06980)  

**Abstract**: Cancer subtype classification is crucial for personalized treatment and prognostic assessment. However, effectively integrating multi-omic data remains challenging due to the heterogeneous nature of genomic, epigenomic, and transcriptomic features. In this work, we propose Modality-Aware Cross-Attention MoXGATE, a novel deep-learning framework that leverages cross-attention and learnable modality weights to enhance feature fusion across multiple omics sources. Our approach effectively captures inter-modality dependencies, ensuring robust and interpretable integration. Through experiments on Gastrointestinal Adenocarcinoma (GIAC) and Breast Cancer (BRCA) datasets from TCGA, we demonstrate that MoXGATE outperforms existing methods, achieving 95\% classification accuracy. Ablation studies validate the effectiveness of cross-attention over simple concatenation and highlight the importance of different omics modalities. Moreover, our model generalizes well to unseen cancer types e.g., breast cancer, underscoring its adaptability. Key contributions include (1) a cross-attention-based multi-omic integration framework, (2) modality-weighted fusion for enhanced interpretability, (3) application of focal loss to mitigate data imbalance, and (4) validation across multiple cancer subtypes. Our results indicate that MoXGATE is a promising approach for multi-omic cancer subtype classification, offering improved performance and biological generalizability. 

**Abstract (ZH)**: 癌症亚型分类对于个性化治疗和预后评估至关重要。然而，由于基因组、表观基因组和转录组特征的异质性，有效地整合多组学数据仍然具有挑战性。在本文中，我们提出了一种新的深度学习框架——模态感知跨注意力MoXGATE，该框架利用跨注意力和可学习的模态权重来增强多组学来源间的特征融合。我们的方法有效地捕捉了跨模态依赖性，确保了稳健的可解释性集成。通过TCGA来源的胃肠道腺癌（GIAC）和乳腺癌（BRCA）数据集的实验，我们展示了MoXGATE在分类准确率方面超过了现有方法，达到了95%的分类准确率。消融研究证实了跨注意力在简单连接之上更为有效，并突显了不同组学模态的重要性。此外，该模型在未见过的癌症类型中表现出良好的泛化能力，强调了其适应性。关键贡献包括（1）基于跨注意力的多组学整合框架，（2）模态加权融合以增强可解释性，（3）使用焦点损失来缓解数据不平衡，以及（4）在多种癌症亚型中的验证。结果显示，MoXGATE是一种有前景的多组学癌症亚型分类方法，提供了改进的性能和生物学普适性。 

---
# UdonCare: Hierarchy Pruning for Unseen Domain Discovery in Predictive Healthcare 

**Title (ZH)**: UdonCare：未知领域发现中的层次结构剪枝在预测型医疗保健中的应用 

**Authors**: Pengfei Hu, Xiaoxue Han, Fei Wang, Yue Ning  

**Link**: [PDF](https://arxiv.org/pdf/2506.06977)  

**Abstract**: Domain generalization has become a critical challenge in clinical prediction, where patient cohorts often exhibit shifting data distributions that degrade model performance. Typical domain generalization approaches struggle in real-world healthcare settings for two main reasons: (1) patient-specific domain labels are typically unavailable, making domain discovery especially difficult; (2) purely data-driven approaches overlook key clinical insights, leading to a gap in medical knowledge integration. To address these problems, we leverage hierarchical medical ontologies like the ICD-9-CM hierarchy to group diseases into higher-level categories and discover more flexible latent domains. In this paper, we introduce UdonCare, a hierarchy-guided framework that iteratively prunes fine-grained domains, encodes these refined domains, and applies a Siamese-type inference mechanism to separate domain-related signals from patient-level features. Experimental results on clinical datasets (MIMIC-III and MIMIC-IV) show that the proposed model achieves higher performance compared to other domain generalization baselines when substantial domain gaps presents, highlighting the untapped potential of medical knowledge for enhancing domain generalization in practical healthcare applications. 

**Abstract (ZH)**: 临床预测中的域泛化已成为一个关键挑战，其中患者群体常表现出数据分布的变化，从而降低模型性能。典型的域泛化方法在实际医疗保健环境中因两大原因难以应对：（1）患者特定的域标签通常不可用，使域发现尤为困难；（2）完全数据驱动的方法忽视了关键的临床洞察，导致医学知识整合的鸿沟。为解决这些问题，我们利用ICD-9-CM层次结构等医疗本体论将疾病分组到较高层次的类别中，以发现更灵活的潜在域。在本文中，我们提出UdonCare，这是一种基于层次结构的框架，该框架通过迭代修剪细粒度的域、对这些精炼的域进行编码，并应用双胞胎类型的推理机制来分离与域相关的信号与患者级特征。临床数据集（MIMIC-III和MIMIC-IV）上的实验结果表明，在存在明显域差距时，所提出模型的表现优于其他域泛化基准，突显了提高实际医疗保健应用中域泛化性能的医学知识的未开发潜力。 

---
# Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test 

**Title (ZH)**: 基于排名一致性的黑盒大语言模型API审计 

**Authors**: Xiaoyuan Zhu, Yaowen Ye, Tianyi Qiu, Hanlin Zhu, Sijun Tan, Ajraf Mannan, Jonathan Michala, Raluca Ada Popa, Willie Neiswanger  

**Link**: [PDF](https://arxiv.org/pdf/2506.06975)  

**Abstract**: As API access becomes a primary interface to large language models (LLMs), users often interact with black-box systems that offer little transparency into the deployed model. To reduce costs or maliciously alter model behaviors, API providers may discreetly serve quantized or fine-tuned variants, which can degrade performance and compromise safety. Detecting such substitutions is difficult, as users lack access to model weights and, in most cases, even output logits. To tackle this problem, we propose a rank-based uniformity test that can verify the behavioral equality of a black-box LLM to a locally deployed authentic model. Our method is accurate, query-efficient, and avoids detectable query patterns, making it robust to adversarial providers that reroute or mix responses upon the detection of testing attempts. We evaluate the approach across diverse threat scenarios, including quantization, harmful fine-tuning, jailbreak prompts, and full model substitution, showing that it consistently achieves superior statistical power over prior methods under constrained query budgets. 

**Abstract (ZH)**: 随着API访问成为访问大规模语言模型（LLMs）的主要接口，用户往往与缺乏透明性的黑盒系统交互。为了降低费用或恶意改变模型行为，API提供商可能会秘密提供量化或微调的变体，这可能损害性能并威胁安全。检测这种替换非常困难，因为用户无法访问模型权重，在大多数情况下甚至无法访问输出概率。为了解决这个问题，我们提出了一种基于排名的均匀性检验方法，可以验证黑盒LLM的行为与本地部署的真实模型行为是否一致。该方法准确、查询高效，并且避免了可检测的查询模式，使其对尝试检测的对抗性提供商具有鲁棒性。我们在包括量化、有害微调、逃逸提示和完整模型替换在内的多种威胁场景下评估了该方法，结果显示，在受限查询预算下，它的一致统计功效优于先前的方法。 

---
# Position: Simulating Society Requires Simulating Thought 

**Title (ZH)**: 位置：模拟社会需要模拟思想 

**Authors**: Chance Jiajie Li, Jiayi Wu, Zhenze Mo, Ao Qu, Yuhan Tang, Kaiya Ivy Zhao, Yulu Gan, Jie Fan, Jiangbo Yu, Jinhua Zhao, Paul Liang, Luis Alonso, Kent Larson  

**Link**: [PDF](https://arxiv.org/pdf/2506.06958)  

**Abstract**: Simulating society with large language models (LLMs), we argue, requires more than generating plausible behavior -- it demands cognitively grounded reasoning that is structured, revisable, and traceable. LLM-based agents are increasingly used to emulate individual and group behavior -- primarily through prompting and supervised fine-tuning. Yet they often lack internal coherence, causal reasoning, and belief traceability -- making them unreliable for analyzing how people reason, deliberate, or respond to interventions.
To address this, we present a conceptual modeling paradigm, Generative Minds (GenMinds), which draws from cognitive science to support structured belief representations in generative agents. To evaluate such agents, we introduce the RECAP (REconstructing CAusal Paths) framework, a benchmark designed to assess reasoning fidelity via causal traceability, demographic grounding, and intervention consistency. These contributions advance a broader shift: from surface-level mimicry to generative agents that simulate thought -- not just language -- for social simulations. 

**Abstract (ZH)**: 使用大型语言模型（LLMs）模拟社会，我们argue需要的不仅仅是生成可信的行为，还需要基于认知的基础进行结构化、可修改和可追溯的推理。基于LLM的代理越来越多地用于模仿个体和群体的行为——主要是通过提示和监督微调实现的。然而，它们往往缺乏内部一致性、因果推理和信念可追溯性——这使它们在分析人们如何推理、审议或回应干预方面不可靠。

为此，我们提出了一个基于认知科学的概念建模范式——生成性心智（GenMinds），旨在支持生成代理的结构化信念表示。为评估此类代理，我们引入了RECAP（重构因果路径）框架，该框架旨在通过因果可追溯性、人口统计学基础和干预一致性来评估推理的准确性。这些贡献推动了更广泛的转变：从表面模仿转向能够模拟思维（而不仅仅是语言）的生成代理，以用于社会模拟。 

---
# BIS Reasoning 1.0: The First Large-Scale Japanese Benchmark for Belief-Inconsistent Syllogistic Reasoning 

**Title (ZH)**: BIS推理1.0：首个大规模日语信念不一致三段论基准 

**Authors**: Ha-Thanh Nguyen, Chaoran Liu, Hirokazu Kiyomaru, Koichi Takeda, Yusuke Miyao, Maki Matsuda, Yusuke Oda, Pontus Stenetorp, Qianying Liu, Su Myat Noe, Hideyuki Tachibana, Kouta Nakayama, Sadao Kurohashi  

**Link**: [PDF](https://arxiv.org/pdf/2506.06955)  

**Abstract**: We present BIS Reasoning 1.0, the first large-scale Japanese dataset of syllogistic reasoning problems explicitly designed to evaluate belief-inconsistent reasoning in large language models (LLMs). Unlike prior datasets such as NeuBAROCO and JFLD, which focus on general or belief-aligned reasoning, BIS Reasoning 1.0 introduces logically valid yet belief-inconsistent syllogisms to uncover reasoning biases in LLMs trained on human-aligned corpora. We benchmark state-of-the-art models - including GPT models, Claude models, and leading Japanese LLMs - revealing significant variance in performance, with GPT-4o achieving 79.54% accuracy. Our analysis identifies critical weaknesses in current LLMs when handling logically valid but belief-conflicting inputs. These findings have important implications for deploying LLMs in high-stakes domains such as law, healthcare, and scientific literature, where truth must override intuitive belief to ensure integrity and safety. 

**Abstract (ZH)**: BIS推理1.0：首个专门评估大规模语言模型信念不一致推理的大规模日语演绎推理数据集 

---
# Is Your Training Pipeline Production-Ready? A Case Study in the Healthcare Domain 

**Title (ZH)**: 你的训练管道准备好进入生产环境了吗？以医疗健康领域为例 

**Authors**: Daniel Lawand, Lucas Quaresma, Roberto Bolgheroni, Alfredo Goldman, Renato Cordeiro Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2506.06946)  

**Abstract**: Deploying a Machine Learning (ML) training pipeline into production requires robust software engineering practices. This differs significantly from experimental workflows. This experience report investigates this challenge in SPIRA, a project whose goal is to create an ML-Enabled System (MLES) to pre-diagnose insufficiency respiratory via speech analysis. The first version of SPIRA's training pipeline lacked critical software quality attributes. This paper presents an overview of the MLES, then compares three versions of the architecture of the Continuous Training subsystem, which evolved from a Big Ball of Mud, to a Modular Monolith, towards Microservices. By adopting different design principles and patterns to enhance its maintainability, robustness, and extensibility. In this way, the paper seeks to offer insights for both ML Engineers tasked to productionize ML training pipelines and Data Scientists seeking to adopt MLOps practices. 

**Abstract (ZH)**: 将机器学习训练管道部署到生产环境需要 robust 软件工程实践。这与实验性工作流程大不相同。本文经验报告探讨了 SPIRA 项目中的这一挑战，SPIRA 的目标是创建一个基于机器学习的系统 (MLES)，以通过语音分析提前诊断呼吸不足。SPIRA 的第一版训练管道缺乏关键的软件质量属性。本文首先概述 MLES，然后比较了该持续训练子系统的三个架构版本，从一团混乱的代码进化到了模块化单体，最终成为微服务。通过采用不同的设计原则和模式来提高其可维护性、稳健性和可扩展性。本文旨在为负责将机器学习训练管道生产化的 ML 工程师以及寻求采用 MLOps 实践的数据科学家提供见解。 

---
# Polar Hierarchical Mamba: Towards Streaming LiDAR Object Detection with Point Clouds as Egocentric Sequences 

**Title (ZH)**: 极化分层蚁狮：基于点云 ego-中心序列的流式 LiDAR 对象检测 

**Authors**: Mellon M. Zhang, Glen Chou, Saibal Mukhopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2506.06944)  

**Abstract**: Accurate and efficient object detection is essential for autonomous vehicles, where real-time perception requires low latency and high throughput. LiDAR sensors provide robust depth information, but conventional methods process full 360° scans in a single pass, introducing significant delay. Streaming approaches address this by sequentially processing partial scans in the native polar coordinate system, yet they rely on translation-invariant convolutions that are misaligned with polar geometry -- resulting in degraded performance or requiring complex distortion mitigation. Recent Mamba-based state space models (SSMs) have shown promise for LiDAR perception, but only in the full-scan setting, relying on geometric serialization and positional embeddings that are memory-intensive and ill-suited to streaming. We propose Polar Hierarchical Mamba (PHiM), a novel SSM architecture designed for polar-coordinate streaming LiDAR. PHiM uses local bidirectional Mamba blocks for intra-sector spatial encoding and a global forward Mamba for inter-sector temporal modeling, replacing convolutions and positional encodings with distortion-aware, dimensionally-decomposed operations. PHiM sets a new state-of-the-art among streaming detectors on the Waymo Open Dataset, outperforming the previous best by 10\% and matching full-scan baselines at twice the throughput. Code will be available at this https URL . 

**Abstract (ZH)**: 基于极坐标的极简Mamba（PHiM）：一种适用于LiDAR流式检测的新颖状态空间模型架构 

---
# Rewriting the Budget: A General Framework for Black-Box Attacks Under Cost Asymmetry 

**Title (ZH)**: 重写预算：在成本不对称条件下的一种通用黑箱攻击框架 

**Authors**: Mahdi Salmani, Alireza Abdollahpoorrostam, Seyed-Mohsen Moosavi-Dezfooli  

**Link**: [PDF](https://arxiv.org/pdf/2506.06933)  

**Abstract**: Traditional decision-based black-box adversarial attacks on image classifiers aim to generate adversarial examples by slightly modifying input images while keeping the number of queries low, where each query involves sending an input to the model and observing its output. Most existing methods assume that all queries have equal cost. However, in practice, queries may incur asymmetric costs; for example, in content moderation systems, certain output classes may trigger additional review, enforcement, or penalties, making them more costly than others. While prior work has considered such asymmetric cost settings, effective algorithms for this scenario remain underdeveloped. In this paper, we propose a general framework for decision-based attacks under asymmetric query costs, which we refer to as asymmetric black-box attacks. We modify two core components of existing attacks: the search strategy and the gradient estimation process. Specifically, we propose Asymmetric Search (AS), a more conservative variant of binary search that reduces reliance on high-cost queries, and Asymmetric Gradient Estimation (AGREST), which shifts the sampling distribution to favor low-cost queries. We design efficient algorithms that minimize total attack cost by balancing different query types, in contrast to earlier methods such as stealthy attacks that focus only on limiting expensive (high-cost) queries. Our method can be integrated into a range of existing black-box attacks with minimal changes. We perform both theoretical analysis and empirical evaluation on standard image classification benchmarks. Across various cost regimes, our method consistently achieves lower total query cost and smaller perturbations than existing approaches, with improvements of up to 40% in some settings. 

**Abstract (ZH)**: 决策导向的异构查询成本黑色盒攻击框架 

---
# DiscoSum: Discourse-aware News Summarization 

**Title (ZH)**: DiscoSum：话语驱动的新闻摘要生成 

**Authors**: Alexander Spangher, Tenghao Huang, Jialiang Gu, Jiatong Shi, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.06930)  

**Abstract**: Recent advances in text summarization have predominantly leveraged large language models to generate concise summaries. However, language models often do not maintain long-term discourse structure, especially in news articles, where organizational flow significantly influences reader engagement. We introduce a novel approach to integrating discourse structure into summarization processes, focusing specifically on news articles across various media. We present a novel summarization dataset where news articles are summarized multiple times in different ways across different social media platforms (e.g. LinkedIn, Facebook, etc.). We develop a novel news discourse schema to describe summarization structures and a novel algorithm, DiscoSum, which employs beam search technique for structure-aware summarization, enabling the transformation of news stories to meet different stylistic and structural demands. Both human and automatic evaluation results demonstrate the efficacy of our approach in maintaining narrative fidelity and meeting structural requirements. 

**Abstract (ZH)**: 最近文本总结的进步主要依赖大规模语言模型生成简洁摘要。然而，语言模型往往不能维持长期的 discourse 结构，特别是在新闻文章中，组织流程显著影响读者参与度。我们提出了一种将 discourse 结构整合到总结过程中的新方法，重点关注不同媒体平台上的新闻文章。我们提供了一个新的摘要数据集，其中新闻文章以不同方式在不同的社交媒体平台（如 LinkedIn、Facebook 等）上进行了多次总结。我们开发了一个新的新闻 discourse 架构来描述摘要结构，并提出了一种名为 DiscoSum 的新算法，该算法使用束搜索技术进行结构意识总结，使新闻故事能够满足不同的风格和结构需求。人类和自动评估结果均证明了我们方法在保持叙述忠实度和满足结构需求方面的有效性。 

---
# Graph-Based Physics-Guided Urban PM2.5 Air Quality Imputation with Constrained Monitoring Data 

**Title (ZH)**: 基于图的物理引导城市PM2.5空气质量插值与受限监测数据约束 

**Authors**: Shangjie Du, Hui Wei, Dong Yoon Lee, Zhizhang Hu, Shijia Pan  

**Link**: [PDF](https://arxiv.org/pdf/2506.06917)  

**Abstract**: This work introduces GraPhy, a graph-based, physics-guided learning framework for high-resolution and accurate air quality modeling in urban areas with limited monitoring data. Fine-grained air quality monitoring information is essential for reducing public exposure to pollutants. However, monitoring networks are often sparse in socioeconomically disadvantaged regions, limiting the accuracy and resolution of air quality modeling. To address this, we propose a physics-guided graph neural network architecture called GraPhy with layers and edge features designed specifically for low-resolution monitoring data. Experiments using data from California's socioeconomically disadvantaged San Joaquin Valley show that GraPhy achieves the overall best performance evaluated by mean squared error (MSE), mean absolute error (MAE), and R-square value (R2), improving the performance by 9%-56% compared to various baseline models. Moreover, GraPhy consistently outperforms baselines across different spatial heterogeneity levels, demonstrating the effectiveness of our model design. 

**Abstract (ZH)**: 基于图的物理引导学习框架：GraPhy在 socioeconomically 不利地区城市区域细粒度空气质量建模中的应用 

---
# Uncertainty Estimation on Graphs with Structure Informed Stochastic Partial Differential Equations 

**Title (ZH)**: 基于结构信息的随机偏微分方程的图上不确定性估计 

**Authors**: Fred Xu, Thomas Markovich  

**Link**: [PDF](https://arxiv.org/pdf/2506.06907)  

**Abstract**: Graph Neural Networks have achieved impressive results across diverse network modeling tasks, but accurately estimating uncertainty on graphs remains difficult, especially under distributional shifts. Unlike traditional uncertainty estimation, graph-based uncertainty must account for randomness arising from both the graph's structure and its label distribution, which adds complexity. In this paper, making an analogy between the evolution of a stochastic partial differential equation (SPDE) driven by Matern Gaussian Process and message passing using GNN layers, we present a principled way to design a novel message passing scheme that incorporates spatial-temporal noises motivated by the Gaussian Process approach to SPDE. Our method simultaneously captures uncertainty across space and time and allows explicit control over the covariance kernel smoothness, thereby enhancing uncertainty estimates on graphs with both low and high label informativeness. Our extensive experiments on Out-of-Distribution (OOD) detection on graph datasets with varying label informativeness demonstrate the soundness and superiority of our model to existing approaches. 

**Abstract (ZH)**: 图神经网络在多样化的网络建模任务中取得了显著成果，但在图形上的不确定性估计依然困难，尤其是在分布转移的情况下。与传统的不确定性估计不同，基于图形的不确定性必须同时考虑图形结构和标签分布带来的随机性，这增加了复杂性。在本文中，我们将马特尔高斯过程驱动的随机偏微分方程（SPDE）演化与图神经网络层的消息传递类比，提出了一种基于高斯过程方法的时空噪声纳入的原理性消息传递方案。该方法同时捕捉空间和时间上的不确定性，并允许对协方差核的光滑度进行显式控制，从而提高具有低和高标签信息量的图形上的不确定性估计。在不同标签信息量的图形数据集的异常检测实验中，我们的方法表现出稳健性和优越性。 

---
# Can Biologically Plausible Temporal Credit Assignment Rules Match BPTT for Neural Similarity? E-prop as an Example 

**Title (ZH)**: 生物可实现的时间信用分配规则能否与BPTR对于神经相似性匹配？E-prop为例 

**Authors**: Yuhan Helena Liu, Guangyu Robert Yang, Christopher J. Cueva  

**Link**: [PDF](https://arxiv.org/pdf/2506.06904)  

**Abstract**: Understanding how the brain learns may be informed by studying biologically plausible learning rules. These rules, often approximating gradient descent learning to respect biological constraints such as locality, must meet two critical criteria to be considered an appropriate brain model: (1) good neuroscience task performance and (2) alignment with neural recordings. While extensive research has assessed the first criterion, the second remains underexamined. Employing methods such as Procrustes analysis on well-known neuroscience datasets, this study demonstrates the existence of a biologically plausible learning rule -- namely e-prop, which is based on gradient truncation and has demonstrated versatility across a wide range of tasks -- that can achieve neural data similarity comparable to Backpropagation Through Time (BPTT) when matched for task accuracy. Our findings also reveal that model architecture and initial conditions can play a more significant role in determining neural similarity than the specific learning rule. Furthermore, we observe that BPTT-trained models and their biologically plausible counterparts exhibit similar dynamical properties at comparable accuracies. These results underscore the substantial progress made in developing biologically plausible learning rules, highlighting their potential to achieve both competitive task performance and neural data similarity. 

**Abstract (ZH)**: 理解大脑学习机制可能通过研究生物可实现的学习规则来获得启示。这些规则通常近似梯度下降学习，以遵守局部性等生物约束，必须满足两个关键标准才能被视为合适的大脑模型：（1）良好的神经科学任务表现和（2）与神经记录的一致性。尽管已有大量研究评估了第一标准，但第二标准仍未得到充分考察。通过在著名神经科学数据集上应用Procrustes分析等方法，本研究证明存在一个生物可实现的学习规则——即e-prop，它基于梯度截断，已在多种任务中显示出灵活性，并且在任务准确度匹配的情况下，其神经数据相似度与时间反向传播（BPTT）相当。我们的研究还发现，模型架构和初始条件在决定神经相似度方面发挥的作用可能比具体的学习规则更为重要。此外，我们观察到，在相似准确度水平下，BPTT训练的模型和其生物可实现的对应模型表现出相似的动力学特性。这些结果强调了开发生物可实现学习规则所取得的重大进展，并突显了其在实现竞争性任务性能和神经数据相似度方面的潜力。 

---
# LLM-D12: A Dual-Dimensional Scale of Instrumental and Relational Dependencies on Large Language Models 

**Title (ZH)**: LLM-D12：大型语言模型上工具性和关系性依赖的双重维度量表 

**Authors**: Ala Yankouskaya, Areej B. Babiker, Syeda W. F. Rizvi, Sameha Alshakhsi, Magnus Liebherr, Raian Ali  

**Link**: [PDF](https://arxiv.org/pdf/2506.06874)  

**Abstract**: There is growing interest in understanding how people interact with large language models (LLMs) and whether such models elicit dependency or even addictive behaviour. Validated tools to assess the extent to which individuals may become dependent on LLMs are scarce and primarily build on classic behavioral addiction symptoms, adapted to the context of LLM use. We view this as a conceptual limitation, as the LLM-human relationship is more nuanced and warrants a fresh and distinct perspective. To address this gap, we developed and validated a new 12-item questionnaire to measure LLM dependency, referred to as LLM-D12. The scale was based on the authors' prior theoretical work, with items developed accordingly and responses collected from 526 participants in the UK. Exploratory and confirmatory factor analyses, performed on separate halves of the total sample using a split-sample approach, supported a two-factor structure: Instrumental Dependency (six items) and Relationship Dependency (six items). Instrumental Dependency reflects the extent to which individuals rely on LLMs to support or collaborate in decision-making and cognitive tasks. Relationship Dependency captures the tendency to perceive LLMs as socially meaningful, sentient, or companion-like entities. The two-factor structure demonstrated excellent internal consistency and clear discriminant validity. External validation confirmed both the conceptual foundation and the distinction between the two subscales. The psychometric properties and structure of our LLM-D12 scale were interpreted in light of the emerging view that dependency on LLMs does not necessarily indicate dysfunction but may still reflect reliance levels that could become problematic in certain contexts. 

**Abstract (ZH)**: 人们对大型语言模型（LLMs）交互的兴趣日益增加，探讨这些模型是否诱发依赖甚至成瘾行为。现有的评估个体对LLM依赖程度的有效测量工具稀缺，主要基于经典的行为成瘾症状，调整应用于LLM使用情境。我们认为这存在概念限制，因为LLM与人类的关系更为复杂，需要一个新的独特视角。为应对这一空白，我们开发并验证了一个新的12项问卷来测量LLM依赖性，称为LLM-D12。该量表基于作者先前的理论工作，相应地开发了项目，并从英国的526名参与者处收集了回应。分样本的探索性因素分析和确认性因素分析支持了一个两因素结构：工具性依赖（六个项目）和关系依赖（六个项目）。工具性依赖反映了个体依赖LLM支持或协作进行决策和认知任务的程度。关系依赖捕捉将LLM视为社会上有意义、有感知力或伴侣般实体的趋势。两因素结构显示了极好的内部一致性，并且具有明确的区分效度。外部验证确认了该量表的概念基础及其两个子量表之间的区分性。我们的LLM-D12量表的 psychometric 特性和结构被解释为随着新兴观点认为对LLM的依赖不一定表示功能障碍，但仍然可能反映某些情境下可能变得有问题的依赖程度。 

---
# Recursive Semantic Anchoring in ISO 639:2023: A Structural Extension to ISO/TC 37 Frameworks 

**Title (ZH)**: ISO 639:2023中的递归语义锚定：ISO/TC 37框架的结构扩展 

**Authors**: Bugra Kilictas, Faruk Alpay  

**Link**: [PDF](https://arxiv.org/pdf/2506.06870)  

**Abstract**: ISO 639:2023 unifies the ISO language-code family and introduces contextual metadata, but it lacks a machine-native mechanism for handling dialectal drift and creole mixtures. We propose a formalisation of recursive semantic anchoring, attaching to every language entity $\chi$ a family of fixed-point operators $\phi_{n,m}$ that model bounded semantic drift via the relation $\phi_{n,m}(\chi) = \chi \oplus \Delta(\chi)$, where $\Delta(\chi)$ is a drift vector in a latent semantic manifold. The base anchor $\phi_{0,0}$ recovers the canonical ISO 639:2023 identity, whereas $\phi_{99,9}$ marks the maximal drift state that triggers a deterministic fallback. Using category theory, we treat the operators $\phi_{n,m}$ as morphisms and drift vectors as arrows in a category $\mathrm{DriftLang}$. A functor $\Phi: \mathrm{DriftLang} \to \mathrm{AnchorLang}$ maps every drifted object to its unique anchor and proves convergence. We provide an RDF/Turtle schema (\texttt{BaseLanguage}, \texttt{DriftedLanguage}, \texttt{ResolvedAnchor}) and worked examples -- e.g., $\phi_{8,4}$ (Standard Mandarin) versus $\phi_{8,7}$ (a colloquial variant), and $\phi_{1,7}$ for Nigerian Pidgin anchored to English. Experiments with transformer models show higher accuracy in language identification and translation on noisy or code-switched input when the $\phi$-indices are used to guide fallback routing. The framework is compatible with ISO/TC 37 and provides an AI-tractable, drift-aware semantic layer for future standards. 

**Abstract (ZH)**: ISO 639:2023 统一了ISO语言代码家族并引入了上下文元数据，但缺乏处理方言漂移和克里奥尔混合的机器原生机制。我们提出了一种递归语义锚定形式化，为每一个语言实体 $\chi$ 附上一族不动点算子 $\phi_{n,m}$，通过关系 $\phi_{n,m}(\chi) = \chi \oplus \Delta(\chi)$ 模型化有界语义漂移，其中 $\Delta(\chi)$ 是在潜在语义流形中的漂移向量。基本锚点 $\phi_{0,0}$ 恢复了ISO 639:2023 的标准标识，而 $\phi_{99,9}$ 标记了触发确定性回退的最大漂移状态。利用范畴论，我们将运算符 $\phi_{n,m}$ 视为范畴 $\mathrm{DriftLang}$ 中的态射，漂移向量视为箭头。函子 $\Phi: \mathrm{DriftLang} \to \mathrm{AnchorLang}$ 将每个漂移对象映射到其唯一的锚定点并证明其收敛性。我们提供了RDF/Turtle模式（BaseLanguage, DriftedLanguage, ResolvedAnchor）及实例——例如 $\phi_{8,4}$（标准普通话）与 $\phi_{8,7}$（一种口语变体）的对比，以及 $\phi_{1,7}$ 对于锚定于英语的尼日利亚皮钦语。实验表明，在使用 $\phi$-索引引导回退路由时，若输入噪声大或混合代码，语言识别和翻译的准确性会更高。该框架兼容ISO/TC 37，并为未来标准提供了一个可由AI处理的、具有漂移感知的语义层。 

---
# SAFE: Finding Sparse and Flat Minima to Improve Pruning 

**Title (ZH)**: SAFE: 寻找稀疏和平坦的极小值以提高剪枝 

**Authors**: Dongyeop Lee, Kwanhee Lee, Jinseok Chung, Namhoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.06866)  

**Abstract**: Sparsifying neural networks often suffers from seemingly inevitable performance degradation, and it remains challenging to restore the original performance despite much recent progress. Motivated by recent studies in robust optimization, we aim to tackle this problem by finding subnetworks that are both sparse and flat at the same time. Specifically, we formulate pruning as a sparsity-constrained optimization problem where flatness is encouraged as an objective. We solve it explicitly via an augmented Lagrange dual approach and extend it further by proposing a generalized projection operation, resulting in novel pruning methods called SAFE and its extension, SAFE$^+$. Extensive evaluations on standard image classification and language modeling tasks reveal that SAFE consistently yields sparse networks with improved generalization performance, which compares competitively to well-established baselines. In addition, SAFE demonstrates resilience to noisy data, making it well-suited for real-world conditions. 

**Abstract (ZH)**: 剪枝神经网络往往伴随着性能下降的问题，即使有近期的进步，恢复原始性能仍然是一个挑战。受到鲁棒优化研究的启发，我们旨在通过寻找同时稀疏和平坦的子网络来解决这一问题。具体而言，我们将剪枝形式化为一个稀疏约束下的优化问题，其中平坦性被鼓励作为目标。我们通过增广拉格朗日对偶方法显式求解，并通过提出通用投影操作进一步扩展，从而得到新的剪枝方法SAFE及其扩展方法SAFE$^+$。在标准图像分类和语言建模任务上的广泛评估表明，SAFE能够一致地生成具有更好泛化性能的稀疏网络，其性能与现有baseline相当。此外，SAFE展现了对噪声数据的鲁棒性，使其更适合实际应用场景。 

---
# Face recognition on point cloud with cgan-top for denoising 

**Title (ZH)**: 基于CGAN-Top点云去噪的面部识别 

**Authors**: Junyu Liu, Jianfeng Ren, Sunhong Liang, Xudong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06864)  

**Abstract**: Face recognition using 3D point clouds is gaining growing interest, while raw point clouds often contain a significant amount of noise due to imperfect sensors. In this paper, an end-to-end 3D face recognition on a noisy point cloud is proposed, which synergistically integrates the denoising and recognition modules. Specifically, a Conditional Generative Adversarial Network on Three Orthogonal Planes (cGAN-TOP) is designed to effectively remove the noise in the point cloud, and recover the underlying features for subsequent recognition. A Linked Dynamic Graph Convolutional Neural Network (LDGCNN) is then adapted to recognize faces from the processed point cloud, which hierarchically links both the local point features and neighboring features of multiple scales. The proposed method is validated on the Bosphorus dataset. It significantly improves the recognition accuracy under all noise settings, with a maximum gain of 14.81%. 

**Abstract (ZH)**: 使用噪声点云的端到端3D人脸识別人机共融的去噪与识别模块融合方法及其在Bosphorus数据集上的验证 

---
# Multimodal Spatial Language Maps for Robot Navigation and Manipulation 

**Title (ZH)**: 多模态空间语言地图在机器人导航与操作中的应用 

**Authors**: Chenguang Huang, Oier Mees, Andy Zeng, Wolfram Burgard  

**Link**: [PDF](https://arxiv.org/pdf/2506.06862)  

**Abstract**: Grounding language to a navigating agent's observations can leverage pretrained multimodal foundation models to match perceptions to object or event descriptions. However, previous approaches remain disconnected from environment mapping, lack the spatial precision of geometric maps, or neglect additional modality information beyond vision. To address this, we propose multimodal spatial language maps as a spatial map representation that fuses pretrained multimodal features with a 3D reconstruction of the environment. We build these maps autonomously using standard exploration. We present two instances of our maps, which are visual-language maps (VLMaps) and their extension to audio-visual-language maps (AVLMaps) obtained by adding audio information. When combined with large language models (LLMs), VLMaps can (i) translate natural language commands into open-vocabulary spatial goals (e.g., "in between the sofa and TV") directly localized in the map, and (ii) be shared across different robot embodiments to generate tailored obstacle maps on demand. Building upon the capabilities above, AVLMaps extend VLMaps by introducing a unified 3D spatial representation integrating audio, visual, and language cues through the fusion of features from pretrained multimodal foundation models. This enables robots to ground multimodal goal queries (e.g., text, images, or audio snippets) to spatial locations for navigation. Additionally, the incorporation of diverse sensory inputs significantly enhances goal disambiguation in ambiguous environments. Experiments in simulation and real-world settings demonstrate that our multimodal spatial language maps enable zero-shot spatial and multimodal goal navigation and improve recall by 50% in ambiguous scenarios. These capabilities extend to mobile robots and tabletop manipulators, supporting navigation and interaction guided by visual, audio, and spatial cues. 

**Abstract (ZH)**: 将语言与导航代理的观察相结合可以利用预训练的多模态基础模型将感知与物体或事件描述匹配起来。然而，以前的方法仍然与环境建图脱离，缺乏几何地图的 spatial 精度，或者忽略了超越视觉的其他模态信息。为了解决这个问题，我们提出多模态空间语言地图作为一种融合预训练多模态特征与环境 3D 重建的空间地图表示。我们通过标准探索方式自动构建这些地图。我们展示了两种我们的地图实例，即视觉-语言地图（VLMaps）及其通过增加音频信息扩展为音频-视觉-语言地图（AVLMaps）。当与大型语言模型（LLMs）结合使用时，VLMaps 可以（i）直接将自然语言命令翻译成开放词汇的空间目标（例如，“在沙发和电视之间”），并将这些目标定位到地图上，并且可以（ii）在不同机器人载体之间共享以按需生成定制的障碍地图。基于上述能力，AVLMaps 通过引入综合音频、视觉和语言线索的统一 3D 空间表示来扩展 VLMaps，这通过预训练多模态基础模型的特征融合实现。这使机器人能够将多模态目标查询（例如，文本、图像或音频片段）定位到空间位置用于导航。此外，多样化感官输入的整合显著提高了在模棱两可环境中目标的消歧。在模拟和真实环境中的实验表明，我们的多模态空间语言地图能够实现零样本的空间和多模态目标导航，并在模棱两可场景中将召回率提高 50%。这些能力扩展到移动机器人和台式操作器，支持由视觉、音频和空间线索引导的导航和交互。 

---
# High-Fidelity Scientific Simulation Surrogates via Adaptive Implicit Neural Representations 

**Title (ZH)**: 高保真科学模拟代理通过自适应隐式神经表示 

**Authors**: Ziwei Li, Yuhan Duan, Tianyu Xiong, Yi-Tang Chen, Wei-Lun Chao, Han-Wei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.06858)  

**Abstract**: Effective surrogate models are critical for accelerating scientific simulations. Implicit neural representations (INRs) offer a compact and continuous framework for modeling spatially structured data, but they often struggle with complex scientific fields exhibiting localized, high-frequency variations. Recent approaches address this by introducing additional features along rigid geometric structures (e.g., grids), but at the cost of flexibility and increased model size. In this paper, we propose a simple yet effective alternative: Feature-Adaptive INR (FA-INR). FA-INR leverages cross-attention to an augmented memory bank to learn flexible feature representations, enabling adaptive allocation of model capacity based on data characteristics, rather than rigid structural assumptions. To further improve scalability, we introduce a coordinate-guided mixture of experts (MoE) that enhances the specialization and efficiency of feature representations. Experiments on three large-scale ensemble simulation datasets show that FA-INR achieves state-of-the-art fidelity while significantly reducing model size, establishing a new trade-off frontier between accuracy and compactness for INR-based surrogates. 

**Abstract (ZH)**: 有效的代理模型对于加速科学模拟至关重要。隐式神经表示（INRs）提供了一种紧凑且连续的框架来建模空间结构化数据，但它们往往难以应对表现出局部高频率变化的复杂科学场。近期的方法通过沿着刚性几何结构（例如网格）引入额外特征来解决这一问题，但代价是灵活性降低和模型尺寸增加。本文提出了一种简单而有效的替代方案：特征自适应隐式神经表示（FA-INR）。FA-INR 利用交叉注意力机制学习灵活的特征表示，基于数据特性而非刚性几何假设进行模型容量的自适应分配。为进一步提高可扩展性，我们引入了坐标引导的专家混合（MoE）机制，增强了特征表示的专业性和效率。在三个大规模集成模拟数据集上的实验表明，FA-INR 在保持顶级准确性的前提下显著减少了模型尺寸，建立了一种基于 INR 的代理模型的新权衡前沿，即准确性和紧凑性之间的权衡。 

---
# Position Prediction Self-Supervised Learning for Multimodal Satellite Imagery Semantic Segmentation 

**Title (ZH)**: 多模态卫星图像语义分割的自监督位置预测学习 

**Authors**: John Waithaka, Moise Busogi  

**Link**: [PDF](https://arxiv.org/pdf/2506.06852)  

**Abstract**: Semantic segmentation of satellite imagery is crucial for Earth observation applications, but remains constrained by limited labelled training data. While self-supervised pretraining methods like Masked Autoencoders (MAE) have shown promise, they focus on reconstruction rather than localisation-a fundamental aspect of segmentation tasks. We propose adapting LOCA (Location-aware), a position prediction self-supervised learning method, for multimodal satellite imagery semantic segmentation. Our approach addresses the unique challenges of satellite data by extending SatMAE's channel grouping from multispectral to multimodal data, enabling effective handling of multiple modalities, and introducing same-group attention masking to encourage cross-modal interaction during pretraining. The method uses relative patch position prediction, encouraging spatial reasoning for localisation rather than reconstruction. We evaluate our approach on the Sen1Floods11 flood mapping dataset, where it significantly outperforms existing reconstruction-based self-supervised learning methods for satellite imagery. Our results demonstrate that position prediction tasks, when properly adapted for multimodal satellite imagery, learn representations more effective for satellite image semantic segmentation than reconstruction-based approaches. 

**Abstract (ZH)**: 卫星图像语义分割对于地球观测应用至关重要，但受限于有限的标记训练数据。虽然掩蔽自编码器（MAE）等自监督预训练方法显示出潜力，但它们侧重于重建而非定位——这是分割任务的一个基本方面。我们提出将LOCA（位置感知）方法适应于多模态卫星图像语义分割。我们的方法通过将SatMAE的通道分组从多光谱扩展到多模态数据，解决卫星数据的特殊挑战，有效处理多种模态，并引入相同组注意力掩码以促进预训练期间的跨模态交互。该方法利用相对补丁位置预测，鼓励位置推理而非重建。我们在Sen1Floods11洪水制图数据集上评估了该方法，结果显示它显著优于现有的基于重建的自监督学习方法。我们的结果表明，当适当适应多模态卫星图像时，位置预测任务学习到的表示对卫星图像语义分割更有效，优于基于重建的方法。 

---
# PCoT: Persuasion-Augmented Chain of Thought for Detecting Fake News and Social Media Disinformation 

**Title (ZH)**: PCoT: 说服增强的思维链用于检测假新闻和社会媒体误导信息 

**Authors**: Arkadiusz Modzelewski, Witold Sosnowski, Tiziano Labruna, Adam Wierzbicki, Giovanni Da San Martino  

**Link**: [PDF](https://arxiv.org/pdf/2506.06842)  

**Abstract**: Disinformation detection is a key aspect of media literacy. Psychological studies have shown that knowledge of persuasive fallacies helps individuals detect disinformation. Inspired by these findings, we experimented with large language models (LLMs) to test whether infusing persuasion knowledge enhances disinformation detection. As a result, we introduce the Persuasion-Augmented Chain of Thought (PCoT), a novel approach that leverages persuasion to improve disinformation detection in zero-shot classification. We extensively evaluate PCoT on online news and social media posts. Moreover, we publish two novel, up-to-date disinformation datasets: EUDisinfo and MultiDis. These datasets enable the evaluation of PCoT on content entirely unseen by the LLMs used in our experiments, as the content was published after the models' knowledge cutoffs. We show that, on average, PCoT outperforms competitive methods by 15% across five LLMs and five datasets. These findings highlight the value of persuasion in strengthening zero-shot disinformation detection. 

**Abstract (ZH)**: 媒体素养中的虚假信息检测是一个关键方面。心理学研究表明，了解有说服力的谬误知识有助于个体检测虚假信息。受这些发现的启发，我们通过实验测试大型语言模型（LLMs）中灌输说服知识是否能增强虚假信息检测。因此，我们提出了说服增强思维链（PCoT）这一新颖方法，利用说服力来改进零样本分类中的虚假信息检测。我们对在线新闻和社交媒体帖子进行了广泛的评估。此外，我们发布了两个最新的虚假信息数据集：EUDisinfo和MultiDis。这些数据集使PCoT能够在实验中使用的LLM从未见过的内容上进行评估，因为内容是在模型知识截止点之后发布的。结果显示，平均而言，PCoT在五种LLM和五种数据集上的表现比竞争方法高出15%。这些发现突显了说服力在增强零样本虚假信息检测方面的价值。 

---
# A Statistical Framework for Model Selection in LSTM Networks 

**Title (ZH)**: LSTM网络中模型选择的统计框架 

**Authors**: Fahad Mostafa  

**Link**: [PDF](https://arxiv.org/pdf/2506.06840)  

**Abstract**: Long Short-Term Memory (LSTM) neural network models have become the cornerstone for sequential data modeling in numerous applications, ranging from natural language processing to time series forecasting. Despite their success, the problem of model selection, including hyperparameter tuning, architecture specification, and regularization choice remains largely heuristic and computationally expensive. In this paper, we propose a unified statistical framework for systematic model selection in LSTM networks. Our framework extends classical model selection ideas, such as information criteria and shrinkage estimation, to sequential neural networks. We define penalized likelihoods adapted to temporal structures, propose a generalized threshold approach for hidden state dynamics, and provide efficient estimation strategies using variational Bayes and approximate marginal likelihood methods. Several biomedical data centric examples demonstrate the flexibility and improved performance of the proposed framework. 

**Abstract (ZH)**: 基于长短期记忆神经网络的统一统计模型选择框架 

---
# AI-Generated Compromises for Coalition Formation 

**Title (ZH)**: AI生成的联盟形成妥协方案 

**Authors**: Eyal Briman, Ehud Shapiro, Nimrod Talmon  

**Link**: [PDF](https://arxiv.org/pdf/2506.06837)  

**Abstract**: The challenge of finding compromises between agent proposals is fundamental to AI subfields such as argumentation, mediation, and negotiation. Building on this tradition, Elkind et al. (2021) introduced a process for coalition formation that seeks majority-supported proposals preferable to the status quo, using a metric space where each agent has an ideal point. A crucial step in this process involves identifying compromise proposals around which agent coalitions can unite. How to effectively find such compromise proposals remains an open question. We address this gap by formalizing a model that incorporates agent bounded rationality and uncertainty, and by developing AI methods to generate compromise proposals. We focus on the domain of collaborative document writing, such as the democratic drafting of a community constitution. Our approach uses natural language processing techniques and large language models to induce a semantic metric space over text. Based on this space, we design algorithms to suggest compromise points likely to receive broad support. To evaluate our methods, we simulate coalition formation processes and show that AI can facilitate large-scale democratic text editing, a domain where traditional tools are limited. 

**Abstract (ZH)**: 寻找代理提案之间妥协方案的挑战是人工智能子领域如论辩、调解和协商中的基本问题。基于这一传统，Elkind等（2021）引入了一个寻求多数支持且优于现状的提案的联盟形成过程，使用一个度量空间，其中每个代理有一个理想点。在这个过程中，关键一步是识别代理联盟可以团结起来的妥协方案。如何有效找到这样的妥协方案仍然是一个开放的问题。我们通过构建同时考虑代理有限理性与不确定性的模型，并开发AI方法生成妥协方案来填补这一空白。我们专注于协作文档写作领域，如社区宪法的民主起草。我们的方法利用自然语言处理技术和大规模语言模型在文本上诱导一种语义度量空间。基于此空间，我们设计算法来建议可能获得广泛支持的妥协点。为了评估我们的方法，我们模拟了联盟形成过程，并展示了AI可以在传统工具受限的大型民主文本编辑领域发挥作用。 

---
# Harnessing Vision-Language Models for Time Series Anomaly Detection 

**Title (ZH)**: 借助视觉语言模型进行时间序列异常检测 

**Authors**: Zelin He, Sarah Alnegheimish, Matthew Reimherr  

**Link**: [PDF](https://arxiv.org/pdf/2506.06836)  

**Abstract**: Time-series anomaly detection (TSAD) has played a vital role in a variety of fields, including healthcare, finance, and industrial monitoring. Prior methods, which mainly focus on training domain-specific models on numerical data, lack the visual-temporal reasoning capacity that human experts have to identify contextual anomalies. To fill this gap, we explore a solution based on vision language models (VLMs). Recent studies have shown the ability of VLMs for visual reasoning tasks, yet their direct application to time series has fallen short on both accuracy and efficiency. To harness the power of VLMs for TSAD, we propose a two-stage solution, with (1) ViT4TS, a vision-screening stage built on a relatively lightweight pretrained vision encoder, which leverages 2-D time-series representations to accurately localize candidate anomalies; (2) VLM4TS, a VLM-based stage that integrates global temporal context and VLM reasoning capacity to refine the detection upon the candidates provided by ViT4TS. We show that without any time-series training, VLM4TS outperforms time-series pretrained and from-scratch baselines in most cases, yielding a 24.6 percent improvement in F1-max score over the best baseline. Moreover, VLM4TS also consistently outperforms existing language-model-based TSAD methods and is on average 36 times more efficient in token usage. 

**Abstract (ZH)**: 基于视觉语言模型的时间序列异常检测 

---
# EndoARSS: Adapting Spatially-Aware Foundation Model for Efficient Activity Recognition and Semantic Segmentation in Endoscopic Surgery 

**Title (ZH)**: EndoARSS：适应空间感知基础模型的内镜手术高效活动识别与语义分割 

**Authors**: Guankun Wang, Rui Tang, Mengya Xu, Long Bai, Huxin Gao, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2506.06830)  

**Abstract**: Endoscopic surgery is the gold standard for robotic-assisted minimally invasive surgery, offering significant advantages in early disease detection and precise interventions. However, the complexity of surgical scenes, characterized by high variability in different surgical activity scenarios and confused image features between targets and the background, presents challenges for surgical environment understanding. Traditional deep learning models often struggle with cross-activity interference, leading to suboptimal performance in each downstream task. To address this limitation, we explore multi-task learning, which utilizes the interrelated features between tasks to enhance overall task performance. In this paper, we propose EndoARSS, a novel multi-task learning framework specifically designed for endoscopy surgery activity recognition and semantic segmentation. Built upon the DINOv2 foundation model, our approach integrates Low-Rank Adaptation to facilitate efficient fine-tuning while incorporating Task Efficient Shared Low-Rank Adapters to mitigate gradient conflicts across diverse tasks. Additionally, we introduce the Spatially-Aware Multi-Scale Attention that enhances feature representation discrimination by enabling cross-spatial learning of global information. In order to evaluate the effectiveness of our framework, we present three novel datasets, MTLESD, MTLEndovis and MTLEndovis-Gen, tailored for endoscopic surgery scenarios with detailed annotations for both activity recognition and semantic segmentation tasks. Extensive experiments demonstrate that EndoARSS achieves remarkable performance across multiple benchmarks, significantly improving both accuracy and robustness in comparison to existing models. These results underscore the potential of EndoARSS to advance AI-driven endoscopic surgical systems, offering valuable insights for enhancing surgical safety and efficiency. 

**Abstract (ZH)**: 内镜手术是机器人辅助微创手术的金标准，提供了早期疾病检测和精确干预的重大优势。然而，不同手术活动场景下的高变异性以及目标与背景之间的混淆图像特征使得对手术环境的理解构成了挑战。传统深度学习模型常常难以应对跨活动干扰，导致每个下游任务的性能欠佳。为解决这一局限，我们探索了多任务学习，利用任务间的相关特征来提升整体任务性能。在本文中，我们提出了一种名为EndoARSS的新颖多任务学习框架，专门用于内镜手术活动识别和语义分割。基于DINOv2基础模型，我们的方法结合了低秩适应以促进高效的微调，并纳入了任务高效共享低秩适配器以缓解不同任务间的梯度冲突。此外，我们引入了空间感知多尺度注意力机制，通过促进全局信息的跨空间学习来增强特征表示的区分性。为了评估我们框架的有效性，我们提出了三个新的数据集，即MTLESD、MTLEndovis和MTLEndovis-Gen，这些数据集专为内镜手术场景设计，并详细注释了活动识别和语义分割任务。广泛实验表明，EndoARSS在多个基准测试中取得了卓越的性能，相较于现有模型显著提高了准确性和鲁棒性。这些结果凸显了EndoARSS在促进基于AI的内镜外科系统发展方面的潜力，为提升手术安全性和效率提供了宝贵的见解。 

---
# Controllable Coupled Image Generation via Diffusion Models 

**Title (ZH)**: 可控耦合图像生成：基于扩散模型的方法 

**Authors**: Chenfei Yuan, Nanshan Jia, Hangqi Li, Peter W. Glynn, Zeyu Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.06826)  

**Abstract**: We provide an attention-level control method for the task of coupled image generation, where "coupled" means that multiple simultaneously generated images are expected to have the same or very similar backgrounds. While backgrounds coupled, the centered objects in the generated images are still expected to enjoy the flexibility raised from different text prompts. The proposed method disentangles the background and entity components in the model's cross-attention modules, attached with a sequence of time-varying weight control parameters depending on the time step of sampling. We optimize this sequence of weight control parameters with a combined objective that assesses how coupled the backgrounds are as well as text-to-image alignment and overall visual quality. Empirical results demonstrate that our method outperforms existing approaches across these criteria. 

**Abstract (ZH)**: 带注意力层级控制的耦合图像生成方法 

---
# Exploring Visual Prompting: Robustness Inheritance and Beyond 

**Title (ZH)**: 探索视觉提示：稳健性继承与超越 

**Authors**: Qi Li, Liangzhi Li, Zhouqiang Jiang, Bowen Wang, Keke Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06823)  

**Abstract**: Visual Prompting (VP), an efficient method for transfer learning, has shown its potential in vision tasks. However, previous works focus exclusively on VP from standard source models, it is still unknown how it performs under the scenario of a robust source model: Can the robustness of the source model be successfully inherited? Does VP also encounter the same trade-off between robustness and generalization ability as the source model during this process? If such a trade-off exists, is there a strategy specifically tailored to VP to mitigate this limitation? In this paper, we thoroughly explore these three questions for the first time and provide affirmative answers to them. To mitigate the trade-off faced by VP, we propose a strategy called Prompt Boundary Loosening (PBL). As a lightweight, plug-and-play strategy naturally compatible with VP, PBL effectively ensures the successful inheritance of robustness when the source model is a robust model, while significantly enhancing VP's generalization ability across various downstream datasets. Extensive experiments across various datasets show that our findings are universal and demonstrate the significant benefits of the proposed strategy. 

**Abstract (ZH)**: 视觉提示（VP），一种高效的迁移学习方法，已在视觉任务中展现出潜力。然而，先前的研究仅专注于从标准源模型获取VP，仍不清楚其在健壮源模型场景下的表现：源模型的健壮性能否成功继承？VP在这一过程中是否也会遇到健壮性和泛化能力之间的相同权衡？若存在这种权衡，是否有一种专门针对VP的策略可以缓解这一限制？在本文中，我们首次深入探讨了这三个问题，并提供了肯定的回答。为了缓解VP面临的权衡，我们提出了一种称为提示边界放宽（PBL）的策略。作为一种轻量级的即插即用策略，PBL自然兼容于VP，在源模型为健壮模型时，有效地确保了健壮性的成功继承，同时显著增强了VP在各种下游数据集上的泛化能力。广泛的数据集实验表明，我们的发现具有普适性，并且证明了所提出策略的重要优势。 

---
# Hi-LSplat: Hierarchical 3D Language Gaussian Splatting 

**Title (ZH)**: 高阶语言高斯点云表示：层次化3D语言高斯点云化 

**Authors**: Chenlu Zhan, Yufei Zhang, Gaoang Wang, Hongwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06822)  

**Abstract**: Modeling 3D language fields with Gaussian Splatting for open-ended language queries has recently garnered increasing attention. However, recent 3DGS-based models leverage view-dependent 2D foundation models to refine 3D semantics but lack a unified 3D representation, leading to view inconsistencies. Additionally, inherent open-vocabulary challenges cause inconsistencies in object and relational descriptions, impeding hierarchical semantic understanding. In this paper, we propose Hi-LSplat, a view-consistent Hierarchical Language Gaussian Splatting work for 3D open-vocabulary querying. To achieve view-consistent 3D hierarchical semantics, we first lift 2D features to 3D features by constructing a 3D hierarchical semantic tree with layered instance clustering, which addresses the view inconsistency issue caused by 2D semantic features. Besides, we introduce instance-wise and part-wise contrastive losses to capture all-sided hierarchical semantic representations. Notably, we construct two hierarchical semantic datasets to better assess the model's ability to distinguish different semantic levels. Extensive experiments highlight our method's superiority in 3D open-vocabulary segmentation and localization. Its strong performance on hierarchical semantic datasets underscores its ability to capture complex hierarchical semantics within 3D scenes. 

**Abstract (ZH)**: 基于视图一致层次语言高斯点云的3D开放词汇查询建模 

---
# Can LLMs Generate Reliable Test Case Generators? A Study on Competition-Level Programming Problems 

**Title (ZH)**: LLM生成可靠测试案例生成器的能力：基于竞赛级别编程问题的研究 

**Authors**: Yuhan Cao, Zian Chen, Kun Quan, Ziliang Zhang, Yu Wang, Xiaoning Dong, Yeqi Feng, Guanzhong He, Jingcheng Huang, Jianhao Li, Yixuan Tan, Jiafu Tang, Yilin Tang, Junlei Wu, Qianyu Xiao, Can Zheng, Shouchen Zhou, Yuxiang Zhu, Yiming Huang, Tian Xie, Tianxing He  

**Link**: [PDF](https://arxiv.org/pdf/2506.06821)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation, capable of tackling complex tasks during inference. However, the extent to which LLMs can be utilized for code checking or debugging through test case generation remains largely unexplored. We investigate this problem from the perspective of competition-level programming (CP) programs and propose TCGBench, a Benchmark for (LLM generation of) Test Case Generators. This benchmark comprises two tasks, aimed at studying the capabilities of LLMs in (1) generating valid test case generators for a given CP problem, and further (2) generating targeted test case generators that expose bugs in human-written code. Experimental results indicate that while state-of-the-art LLMs can generate valid test case generators in most cases, most LLMs struggle to generate targeted test cases that reveal flaws in human code effectively. Especially, even advanced reasoning models (e.g., o3-mini) fall significantly short of human performance in the task of generating targeted generators. Furthermore, we construct a high-quality, manually curated dataset of instructions for generating targeted generators. Analysis demonstrates that the performance of LLMs can be enhanced with the aid of this dataset, by both prompting and fine-tuning. 

**Abstract (ZH)**: 大型语言模型(LLMs)在代码生成方面展现了 remarkable 的能力，能够在推理过程中应对复杂的任务。然而，通过测试用例生成来利用LLMs进行代码检查或调试的可能性尚未得到充分探索。我们从竞赛级编程(CP)程序的角度研究了这一问题，并提出了TCGBench：一个针对(LLM生成的)测试用例生成器的基准测试。该基准测试包括两个任务，旨在研究LLMs在(1)为给定的CP问题生成有效测试用例生成器的能力，以及进一步(2)生成针对人类编写的代码中漏洞的精确测试用例生成器的能力。实验结果表明，尽管最先进的LLMs在大多数情况下可以生成有效测试用例生成器，但大多数LLMs在生成能够有效揭示人类代码缺陷的精确测试用例方面仍存在困难。特别是，即使是最先进的推理模型(如o3-mini)在生成精确生成器的任务中远低于人类的性能。此外，我们构建了一个高质量的人工精选的指令数据集，用于生成精确生成器。分析表明，通过提示和微调，可以提升LLMs的性能。 

---
# IMPA-HGAE:Intra-Meta-Path Augmented Heterogeneous Graph Autoencoder 

**Title (ZH)**: IMPA-HGAE：基于元路径增强的异构图自编码器 

**Authors**: Di Lin, Wanjing Ren, Xuanbin Li, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06809)  

**Abstract**: Self-supervised learning (SSL) methods have been increasingly applied to diverse downstream tasks due to their superior generalization capabilities and low annotation costs. However, most existing heterogeneous graph SSL models convert heterogeneous graphs into homogeneous ones via meta-paths for training, which only leverage information from nodes at both ends of meta-paths while underutilizing the heterogeneous node information along the meta-paths. To address this limitation, this paper proposes a novel framework named IMPA-HGAE to enhance target node embeddings by fully exploiting internal node information along meta-paths. Experimental results validate that IMPA-HGAE achieves superior performance on heterogeneous datasets. Furthermore, this paper introduce innovative masking strategies to strengthen the representational capacity of generative SSL models on heterogeneous graph data. Additionally, this paper discuss the interpretability of the proposed method and potential future directions for generative self-supervised learning in heterogeneous graphs. This work provides insights into leveraging meta-path-guided structural semantics for robust representation learning in complex graph scenarios. 

**Abstract (ZH)**: 自监督学习方法（SSL）由于其出色的泛化能力和较低的标注成本，已被广泛应用于多种下游任务。然而，现有的大多数异构图SSL模型通过元路径将异构图转换为同构图进行训练，这仅利用了元路径两端节点的信息，而未充分利用沿元路径的异构节点信息。为解决这一问题，本文提出了一种新颖的框架IMPA-HGAE，通过充分利用沿元路径的内部节点信息来增强目标节点嵌入。实验结果验证了IMPA-HGAE在异构数据集上取得了优越的性能。此外，本文引入了创新的掩码策略，以增强生成性SSL模型在异构图数据上的表示能力。同时，本文讨论了所提出方法的可解释性以及生成式自监督学习在异构图中潜在的研究方向。本工作为在复杂图场景中利用元路径引导的结构语义进行鲁棒表示学习提供了见解。 

---
# Not quite Sherlock Holmes: Language model predictions do not reliably differentiate impossible from improbable events 

**Title (ZH)**: 不是像福尔摩斯一样的推理：语言模型预测不能可靠地区分不可能事件与不大可能的事件 

**Authors**: James A. Michaelov, Reeka Estacio, Zhien Zhang, Benjamin K. Bergen  

**Link**: [PDF](https://arxiv.org/pdf/2506.06808)  

**Abstract**: Can language models reliably predict that possible events are more likely than merely improbable ones? By teasing apart possibility, typicality, and contextual relatedness, we show that despite the results of previous work, language models' ability to do this is far from robust. In fact, under certain conditions, all models tested - including Llama 3, Gemma 2, and Mistral NeMo - perform at worse-than-chance level, assigning higher probabilities to impossible sentences such as 'the car was given a parking ticket by the brake' than to merely unlikely sentences such as 'the car was given a parking ticket by the explorer'. 

**Abstract (ZH)**: 语言模型能否可靠地预测可能事件比单纯的不可能事件更有可能发生？通过区分可能性、典型性和上下文相关性，我们表明，尽管先前研究的结果表明语言模型能做到这一点，但它们的能力远非稳健。事实上，在某些条件下，测试的所有模型（包括Llama 3、Gemma 2和Mistral NeMo）的表现甚至低于随机水平，将不可能句子“汽车被刹车开出了停车罚单”赋予更高的概率，而不是单纯的 unlikely 句子“汽车被探险家开出了停车罚单”。 

---
# Label-semantics Aware Generative Approach for Domain-Agnostic Multilabel Classification 

**Title (ZH)**: 面向领域无关的多标签分类的标签语义aware生成方法 

**Authors**: Subhendu Khatuya, Shashwat Naidu, Saptarshi Ghosh, Pawan Goyal, Niloy Ganguly  

**Link**: [PDF](https://arxiv.org/pdf/2506.06806)  

**Abstract**: The explosion of textual data has made manual document classification increasingly challenging. To address this, we introduce a robust, efficient domain-agnostic generative model framework for multi-label text classification. Instead of treating labels as mere atomic symbols, our approach utilizes predefined label descriptions and is trained to generate these descriptions based on the input text. During inference, the generated descriptions are matched to the pre-defined labels using a finetuned sentence transformer. We integrate this with a dual-objective loss function, combining cross-entropy loss and cosine similarity of the generated sentences with the predefined target descriptions, ensuring both semantic alignment and accuracy. Our proposed model LAGAMC stands out for its parameter efficiency and versatility across diverse datasets, making it well-suited for practical applications. We demonstrate the effectiveness of our proposed model by achieving new state-of-the-art performances across all evaluated datasets, surpassing several strong baselines. We achieve improvements of 13.94% in Micro-F1 and 24.85% in Macro-F1 compared to the closest baseline across all datasets. 

**Abstract (ZH)**: 文本数据的爆炸性增长使得手工文档分类越来越具有挑战性。为此，我们提出了一种稳健高效的领域无关生成模型框架，用于多标签文本分类。我们的方法不仅将标签视为简单的原子符号，还利用预定义的标签描述，并根据输入文本生成这些描述。在推理过程中，生成的描述使用微调的句子变换器与预定义的标签进行匹配。我们整合了双目标损失函数，结合交叉熵损失和生成句子与预定义目标描述的余弦相似度，确保语义对齐和准确性。我们提出的模型LAGAMC因其参数效率和在多种数据集上的适用性而 standout，使其适用于实际应用。我们通过在所有评估数据集上达到新的最佳性能，证明了所提出模型的有效性，超越了几个强基线。与最接近的基线相比，我们在所有数据集上分别实现了13.94%的Micro-F1和24.85%的Macro-F1的改进。 

---
# Is Optimal Transport Necessary for Inverse Reinforcement Learning? 

**Title (ZH)**: 最优传输对于逆强化学习是必要的吗？ 

**Authors**: Zixuan Dong, Yumi Omori, Keith Ross  

**Link**: [PDF](https://arxiv.org/pdf/2506.06793)  

**Abstract**: Inverse Reinforcement Learning (IRL) aims to recover a reward function from expert demonstrations. Recently, Optimal Transport (OT) methods have been successfully deployed to align trajectories and infer rewards. While OT-based methods have shown strong empirical results, they introduce algorithmic complexity, hyperparameter sensitivity, and require solving the OT optimization problems. In this work, we challenge the necessity of OT in IRL by proposing two simple, heuristic alternatives: (1) Minimum-Distance Reward, which assigns rewards based on the nearest expert state regardless of temporal order; and (2) Segment-Matching Reward, which incorporates lightweight temporal alignment by matching agent states to corresponding segments in the expert trajectory. These methods avoid optimization, exhibit linear-time complexity, and are easy to implement. Through extensive evaluations across 32 online and offline benchmarks with three reinforcement learning algorithms, we show that our simple rewards match or outperform recent OT-based approaches. Our findings suggest that the core benefits of OT may arise from basic proximity alignment rather than its optimal coupling formulation, advocating for reevaluation of complexity in future IRL design. 

**Abstract (ZH)**: 逆向强化学习（IRL）旨在从专家演示中恢复奖励函数。最近，最优传输（OT）方法已被成功应用于对齐轨迹并推断奖励。尽管基于OT的方法展示了强大的实证结果，但它们引入了算法复杂性、超参数敏感性，并需要解决OT优化问题。在本文中，我们通过提出两种简单且启发式的替代方案挑战OT在IRL中的必要性：（1）最小距离奖励，基于最近的专家状态分配奖励，而不考虑时间顺序；（2）段匹配奖励，通过将代理状态与专家轨迹中的相应段匹配来引入轻量级的时间对齐。这些方法避免了优化，具有线性时间复杂性，并易于实现。通过在32个在线和离线基准上的广泛评估，结合三种强化学习算法，我们展示出我们的简单奖励能够匹配或超越最近的OT基方法。我们的发现表明，OT的核心益处可能源于基本的邻近对齐，而非其最佳耦合形式，从而为未来IRL设计中的复杂性重新评估提供依据。 

---
# Feature-Based Instance Neighbor Discovery: Advanced Stable Test-Time Adaptation in Dynamic World 

**Title (ZH)**: 基于特征的实例邻居发现：动态世界中高级稳定的测试时自适应 

**Authors**: Qinting Jiang, Chuyang Ye, Dongyan Wei, Bingli Wang, Yuan Xue, Jingyan Jiang, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06782)  

**Abstract**: Despite progress, deep neural networks still suffer performance declines under distribution shifts between training and test domains, leading to a substantial decrease in Quality of Experience (QoE) for applications. Existing test-time adaptation (TTA) methods are challenged by dynamic, multiple test distributions within batches. We observe that feature distributions across different domains inherently cluster into distinct groups with varying means and variances. This divergence reveals a critical limitation of previous global normalization strategies in TTA, which inevitably distort the original data characteristics. Based on this insight, we propose Feature-based Instance Neighbor Discovery (FIND), which comprises three key components: Layer-wise Feature Disentanglement (LFD), Feature Aware Batch Normalization (FABN) and Selective FABN (S-FABN). LFD stably captures features with similar distributions at each layer by constructing graph structures. While FABN optimally combines source statistics with test-time distribution specific statistics for robust feature representation. Finally, S-FABN determines which layers require feature partitioning and which can remain unified, thereby enhancing inference efficiency. Extensive experiments demonstrate that FIND significantly outperforms existing methods, achieving a 30\% accuracy improvement in dynamic scenarios while maintaining computational efficiency. 

**Abstract (ZH)**: 基于特征的实例邻居发现：应对分布转移的测试时自适应方法 

---
# Depth-Optimal Quantum Layout Synthesis as SAT 

**Title (ZH)**: 深度最优量子布局合成作为SAT问题 

**Authors**: Anna B. Jakobsen, Anders B. Clausen, Jaco van de Pol, Irfansha Shaik  

**Link**: [PDF](https://arxiv.org/pdf/2506.06752)  

**Abstract**: Quantum circuits consist of gates applied to qubits. Current quantum hardware platforms impose connectivity restrictions on binary CX gates. Hence, Layout Synthesis is an important step to transpile quantum circuits before they can be executed. Since CX gates are noisy, it is important to reduce the CX count or CX depth of the mapped circuits.
We provide a new and efficient encoding of Quantum-circuit Layout Synthesis in SAT. Previous SAT encodings focused on gate count and CX-gate count. Our encoding instead guarantees that we find mapped circuits with minimal circuit depth or minimal CX-gate depth. We use incremental SAT solving and parallel plans for an efficient encoding. This results in speedups of more than 10-100x compared to OLSQ2, which guarantees depth-optimality. But minimizing depth still takes more time than minimizing gate count with Q-Synth.
We correlate the noise reduction achieved by simulating circuits after (CX)-count and (CX)-depth reduction. We find that minimizing for CX-count correlates better with reducing noise than minimizing for CX-depth. However, taking into account both CX-count and CX-depth provides the best noise reduction. 

**Abstract (ZH)**: 量子电路布局合成的新型高效SAT编码研究 

---
# C-PATH: Conversational Patient Assistance and Triage in Healthcare System 

**Title (ZH)**: C-PATH: 医疗保健系统中的对话患者辅助与分诊 

**Authors**: Qi Shi, Qiwei Han, Cláudia Soares  

**Link**: [PDF](https://arxiv.org/pdf/2506.06737)  

**Abstract**: Navigating healthcare systems can be complex and overwhelming, creating barriers for patients seeking timely and appropriate medical attention. In this paper, we introduce C-PATH (Conversational Patient Assistance and Triage in Healthcare), a novel conversational AI system powered by large language models (LLMs) designed to assist patients in recognizing symptoms and recommending appropriate medical departments through natural, multi-turn dialogues. C-PATH is fine-tuned on medical knowledge, dialogue data, and clinical summaries using a multi-stage pipeline built on the LLaMA3 architecture. A core contribution of this work is a GPT-based data augmentation framework that transforms structured clinical knowledge from DDXPlus into lay-person-friendly conversations, allowing alignment with patient communication norms. We also implement a scalable conversation history management strategy to ensure long-range coherence. Evaluation with GPTScore demonstrates strong performance across dimensions such as clarity, informativeness, and recommendation accuracy. Quantitative benchmarks show that C-PATH achieves superior performance in GPT-rewritten conversational datasets, significantly outperforming domain-specific baselines. C-PATH represents a step forward in the development of user-centric, accessible, and accurate AI tools for digital health assistance and triage. 

**Abstract (ZH)**: 基于大型语言模型的对话式患者辅助与分诊系统：C-PATH 

---
# Ai-Driven Vulnerability Analysis in Smart Contracts: Trends, Challenges and Future Directions 

**Title (ZH)**: 基于人工智能的智能合约漏洞分析：趋势、挑战与未来方向 

**Authors**: Mesut Ozdag  

**Link**: [PDF](https://arxiv.org/pdf/2506.06735)  

**Abstract**: Smart contracts, integral to blockchain ecosystems, enable decentralized applications to execute predefined operations without intermediaries. Their ability to enforce trustless interactions has made them a core component of platforms such as Ethereum. Vulnerabilities such as numerical overflows, reentrancy attacks, and improper access permissions have led to the loss of millions of dollars throughout the blockchain and smart contract sector. Traditional smart contract auditing techniques such as manual code reviews and formal verification face limitations in scalability, automation, and adaptability to evolving development patterns. As a result, AI-based solutions have emerged as a promising alternative, offering the ability to learn complex patterns, detect subtle flaws, and provide scalable security assurances. This paper examines novel AI-driven techniques for vulnerability detection in smart contracts, focusing on machine learning, deep learning, graph neural networks, and transformer-based models. This paper analyzes how each technique represents code, processes semantic information, and responds to real world vulnerability classes. We also compare their strengths and weaknesses in terms of accuracy, interpretability, computational overhead, and real time applicability. Lastly, it highlights open challenges and future opportunities for advancing this domain. 

**Abstract (ZH)**: 智能合约是区块链生态系统的关键组成部分，能够使去中心化应用在没有中介的情况下执行预定义的操作。它们能够在不信任的交互中强制执行信任，成为以太坊等平台的核心组件。数值溢出、重入攻击和不当访问权限等漏洞已导致区块链和智能合约领域损失数百万美元。传统的智能合约审计技术如人工代码审查和形式化验证面临可扩展性、自动化和适应不断变化的开发模式的局限性。因此，基于AI的解决方案已成为有前途的替代方案，能够学习复杂模式、检测细微缺陷并提供可扩展的安全保证。本文探讨了新型AI驱动的智能合约漏洞检测技术，重点关注机器学习、深度学习、图神经网络和变压器模型。本文分析了每种技术如何表示代码、处理语义信息以及应对真实世界的漏洞类别。我们还比较了它们在准确性、可解释性、计算开销和实时适用性方面的优缺点。最后，本文指出了该领域的开放挑战和未来机遇。 

---
# Neural Spectral Band Generation for Audio Coding 

**Title (ZH)**: 基于神经网络的音素频带生成音频编码 

**Authors**: Woongjib Choi, Byeong Hyeon Kim, Hyungseob Lim, Inseon Jang, Hong-Goo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06732)  

**Abstract**: Audio bandwidth extension is the task of reconstructing missing high frequency components of bandwidth-limited audio signals, where bandwidth limitation is a common issue for audio signals due to several reasons, including channel capacity and data constraints. While conventional spectral band replication is a well-established parametric approach to audio bandwidth extension, the SBR usually entails coarse feature extraction and reconstruction techniques, which leads to limitations when processing various types of audio signals. In parallel, numerous deep neural network-based audio bandwidth extension methods have been proposed. These DNN-based methods are usually referred to as blind BWE, as these methods do not rely on prior information extracted from original signals, and only utilize given low frequency band signals to estimate missing high frequency components. In order to replace conventional SBR with DNNs, simply adopting existing DNN-based methodologies results in suboptimal performance due to the blindness of these methods. My proposed research suggests a new approach to parametric non-blind bandwidth extension, as DNN-based side information extraction and DNN-based bandwidth extension are performed only at the front and end of the audio coding pipeline. 

**Abstract (ZH)**: 宽带扩展是 reconstruction 缺失的高频频带组件的任务，其中频带限制是由于多种原因（包括信道容量和数据约束）对音频信号的常见问题。虽然传统的频谱带复制是音频宽带扩展的成熟参数化方法，但频谱带复制通常涉及粗略的特征提取和重建技术，这在处理不同类型音频信号时会带来限制。与此同时，基于深度神经网络的音频宽带扩展方法也被广泛提出。这些基于 DNN 的方法通常被称为盲宽带扩展，因为这些方法不依赖于从原始信号中提取的先验信息，并且仅利用给定的低频带信号来估计缺失的高频组件。为了用 DNN 替换传统的频谱带复制，直接采用现有的 DNN 基础方法会导致性能不佳，因为这些方法是盲目的。我提出的研究所建议的是一种新的参数化非盲宽带扩展方法，因为在音频编码管道的前后分别仅进行基于 DNN 的次要信息提取和基于 DNN 的宽带扩展。 

---
# Fuse and Federate: Enhancing EV Charging Station Security with Multimodal Fusion and Federated Learning 

**Title (ZH)**: 融合与联邦：通过多模态融合和联邦学习增强电动汽车充电站安全 

**Authors**: Rabah Rahal, Abdelaziz Amara Korba, Yacine Ghamri-Doudane  

**Link**: [PDF](https://arxiv.org/pdf/2506.06730)  

**Abstract**: The rapid global adoption of electric vehicles (EVs) has established electric vehicle supply equipment (EVSE) as a critical component of smart grid infrastructure. While essential for ensuring reliable energy delivery and accessibility, EVSE systems face significant cybersecurity challenges, including network reconnaissance, backdoor intrusions, and distributed denial-of-service (DDoS) attacks. These emerging threats, driven by the interconnected and autonomous nature of EVSE, require innovative and adaptive security mechanisms that go beyond traditional intrusion detection systems (IDS). Existing approaches, whether network-based or host-based, often fail to detect sophisticated and targeted attacks specifically crafted to exploit new vulnerabilities in EVSE infrastructure. This paper proposes a novel intrusion detection framework that leverages multimodal data sources, including network traffic and kernel events, to identify complex attack patterns. The framework employs a distributed learning approach, enabling collaborative intelligence across EVSE stations while preserving data privacy through federated learning. Experimental results demonstrate that the proposed framework outperforms existing solutions, achieving a detection rate above 98% and a precision rate exceeding 97% in decentralized environments. This solution addresses the evolving challenges of EVSE security, offering a scalable and privacypreserving response to advanced cyber threats 

**Abstract (ZH)**: 电动汽车（EVs）的快速全球采纳已将电动汽车供电设备（EVSE）确立为智能电网基础设施的关键组件。虽然EVSE系统对于确保可靠的能源供应和可访问性至关重要，但它们也面临着重大的网络安全挑战，包括网络探测、后门入侵和分布式拒绝服务（DDoS）攻击。这些新兴威胁由EVSE的互连和自主特性驱动，需要超越传统入侵检测系统（IDS）的创新和适应性安全机制。现有方法，无论是基于网络的还是基于主机的，往往无法检测出专门设计以利用EVSE基础设施中新漏洞的复杂和针对性攻击。本文提出了一种新的入侵检测框架，该框架利用多模态数据源，包括网络流量和内核事件，以识别复杂的攻击模式。该框架采用分布式学习方法，可以在保留数据隐私的同时，在EVSE站点之间实现协作智能。实验结果表明，所提出的框架在分散环境中表现出色，检测率达到98%以上，精确率超过97%。该解决方案应对了EVSE安全的不断演变的挑战，提供了对高级网络安全威胁的可扩展和隐私保护响应。 

---
# Improving Wildlife Out-of-Distribution Detection: Africas Big Five 

**Title (ZH)**: 改善野生动物域外检测：非洲五大物种 

**Authors**: Mufhumudzi Muthivhi, Jiahao Huo, Fredrik Gustafsson, Terence L. van Zyl  

**Link**: [PDF](https://arxiv.org/pdf/2506.06719)  

**Abstract**: Mitigating human-wildlife conflict seeks to resolve unwanted encounters between these parties. Computer Vision provides a solution to identifying individuals that might escalate into conflict, such as members of the Big Five African animals. However, environments often contain several varied species. The current state-of-the-art animal classification models are trained under a closed-world assumption. They almost always remain overconfident in their predictions even when presented with unknown classes. This study investigates out-of-distribution (OOD) detection of wildlife, specifically the Big Five. To this end, we select a parametric Nearest Class Mean (NCM) and a non-parametric contrastive learning approach as baselines to take advantage of pretrained and projected features from popular classification encoders. Moreover, we compare our baselines to various common OOD methods in the literature. The results show feature-based methods reflect stronger generalisation capability across varying classification thresholds. Specifically, NCM with ImageNet pre-trained features achieves a 2%, 4% and 22% improvement on AUPR-IN, AUPR-OUT and AUTC over the best OOD methods, respectively. The code can be found here this https URL 

**Abstract (ZH)**: 减轻人与野生动物冲突寻求解决这些双方之间的不受wanted影响的遭遇。计算机视觉提供了识别可能升级为冲突的个体的方法，例如非洲五大动物成员。然而，环境通常包含多种多样的物种。当前最先进的动物分类模型在封闭世界假设下进行训练。它们几乎总是对其预测过度自信，即使面对未知类别也是如此。本研究调查了野生动物，特别是五大动物的out-of-distribution (OOD) 检测。为此，我们选择了参数化的最近类均值（NCM）和非参数对比学习方法作为基线，利用流行分类编码器预训练和投影的特征。此外，我们将基线与文献中各种常见的OOD方法进行了比较。结果显示基于特征的方法在不同分类阈值下展现出更强的泛化能力。具体而言，带有ImageNet预训练特征的NCM在AUPR-IN、AUPR-OUT和AUTC上的表现分别比最佳OOD方法提高了2%、4%和22%。代码可在此处找到：this https URL。 

---
# DivScore: Zero-Shot Detection of LLM-Generated Text in Specialized Domains 

**Title (ZH)**: DivScore: 零样本检测专门领域中LLM生成的文本 

**Authors**: Zhihui Chen, Kai He, Yucheng Huang, Yunxiao Zhu, Mengling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.06705)  

**Abstract**: Detecting LLM-generated text in specialized and high-stakes domains like medicine and law is crucial for combating misinformation and ensuring authenticity. However, current zero-shot detectors, while effective on general text, often fail when applied to specialized content due to domain shift. We provide a theoretical analysis showing this failure is fundamentally linked to the KL divergence between human, detector, and source text distributions. To address this, we propose DivScore, a zero-shot detection framework using normalized entropy-based scoring and domain knowledge distillation to robustly identify LLM-generated text in specialized domains. We also release a domain-specific benchmark for LLM-generated text detection in the medical and legal domains. Experiments on our benchmark show that DivScore consistently outperforms state-of-the-art detectors, with 14.4% higher AUROC and 64.0% higher recall (0.1% false positive rate threshold). In adversarial settings, DivScore demonstrates superior robustness than other baselines, achieving on average 22.8% advantage in AUROC and 29.5% in recall. Code and data are publicly available. 

**Abstract (ZH)**: 在医学和法律等专业和高风险领域检测LLM生成的文本对于打击 misinformation 和确保真实性至关重要。然而，当前的零样本检测器在应用于专业内容时由于领域偏移往往会失效。我们提供了一种理论分析，表明这一失败从根本上与人类、检测器和源文本分布之间的KL散度有关。为解决这一问题，我们提出了DivScore，这是一种基于归一化熵评分和领域知识提取的零样本检测框架，能够在专业领域中稳健地识别LLM生成的文本。我们还发布了针对医学和法律领域LLM生成文本检测的专用基准。在我们基准上的实验表明，DivScore始终优于最先进的检测器，AUC ROC提高14.4%，召回率提高64.0%（在假 positives率为0.1%的情况下）。在对抗性设置中，DivScore表现出色，平均AUC ROC优势为22.8%，召回率优势为29.5%。代码和数据已公开。 

---
# Do Protein Transformers Have Biological Intelligence? 

**Title (ZH)**: 蛋白质变换器具有生物智能吗？ 

**Authors**: Fudong Lin, Wanrou Du, Jinchan Liu, Tarikul Milon, Shelby Meche, Wu Xu, Xiaoqi Qin, Xu Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2506.06701)  

**Abstract**: Deep neural networks, particularly Transformers, have been widely adopted for predicting the functional properties of proteins. In this work, we focus on exploring whether Protein Transformers can capture biological intelligence among protein sequences. To achieve our goal, we first introduce a protein function dataset, namely Protein-FN, providing over 9000 protein data with meaningful labels. Second, we devise a new Transformer architecture, namely Sequence Protein Transformers (SPT), for computationally efficient protein function predictions. Third, we develop a novel Explainable Artificial Intelligence (XAI) technique called Sequence Score, which can efficiently interpret the decision-making processes of protein models, thereby overcoming the difficulty of deciphering biological intelligence bided in Protein Transformers. Remarkably, even our smallest SPT-Tiny model, which contains only 5.4M parameters, demonstrates impressive predictive accuracy, achieving 94.3% on the Antibiotic Resistance (AR) dataset and 99.6% on the Protein-FN dataset, all accomplished by training from scratch. Besides, our Sequence Score technique helps reveal that our SPT models can discover several meaningful patterns underlying the sequence structures of protein data, with these patterns aligning closely with the domain knowledge in the biology community. We have officially released our Protein-FN dataset on Hugging Face Datasets this https URL. Our code is available at this https URL. 

**Abstract (ZH)**: 深神经网络，尤其是变换器，已被广泛用于预测蛋白质的功能属性。在这项工作中，我们重点关注探究蛋白质变换器是否能够捕捉蛋白质序列中的生物智能。为了实现这一目标，我们首先介绍了蛋白质功能数据集Protein-FN，提供了超过9000个带有有意义标签的蛋白质数据。其次，我们设计了一种新的变换器架构，称为序列蛋白质变换器(SPT)，以实现高效的蛋白质功能预测。第三，我们开发了一种新型的可解释人工智能(XAI)技术，称为序列得分(Sequence Score)，该技术能够高效地解释蛋白质模型的决策过程，从而克服解释蛋白质变换器中蕴含的生物智能的困难。值得注意的是，即使是我们最小的SPT-Tiny模型，仅包含5.4M参数，也展示了令人印象深刻的预测准确性，在抗生素耐药性(AR)数据集中达到了94.3%，在Protein-FN数据集中达到了99.6%，均通过从零开始训练实现。此外，我们的序列得分技术帮助揭示了我们的SPT模型能够发现蛋白质数据序列结构背后的多个有意义模式，这些模式与生物领域中的专业知识高度吻合。我们已正式在Hugging Face Datasets上发布了Protein-FN数据集，详细信息请访问以下链接：https://huggingface.co/datasets/Protein-FN。我们的代码可以在以下链接中获取：https://。 

---
# MarginSel : Max-Margin Demonstration Selection for LLMs 

**Title (ZH)**: MarginSel: Max-Margin 示范选择for 大型语言模型 

**Authors**: Rajeev Bhatt Ambati, James Lester, Shashank Srivastava, Snigdha Chaturvedi  

**Link**: [PDF](https://arxiv.org/pdf/2506.06699)  

**Abstract**: Large Language Models (LLMs) excel at few-shot learning via in-context learning (ICL). However, the effectiveness of ICL is often sensitive to the selection and ordering of demonstration examples. To address this, we present MarginSel: Max-Margin Demonstration Selection for LLMs, a two-step method that selects hard demonstration examples for the ICL prompt, adapting to each test instance. Our approach achieves 2-7% absolute improvement in F1-score across classification tasks, compared to a random selection of examples. We also provide theoretical insights and empirical evidence showing that MarginSel induces max-margin behavior in LLMs by effectively increasing the margin for hard examples, analogous to support vectors, thereby shifting the decision boundary in a beneficial direction. 

**Abstract (ZH)**: Large Language Models (LLMs) 在基于上下文示例的学习中表现出色，但 ICL 的有效性往往取决于示例选择和排序。为此，我们提出了 MarginSel：面向 LLMs 的最大边际示例选择方法，这是一种两步方法，用于选择适应每个测试实例的 ICL 提示中的硬示例。我们的方法在分类任务中实现了 2-7% 的绝对 F1 分数改进，优于随机选择示例。我们还提供了理论洞察和实证证据，表明 MarginSel 通过有效增大硬示例的边际，促使 LLMs 产生最大边际行为，类似于支持向量，从而在有益的方向上移动决策边界。 

---
# Design and Implementation of a RISC-V SoC with Custom DSP Accelerators for Edge Computing 

**Title (ZH)**: 基于边缘计算的自定义DSP加速器嵌入的RISC-V SoC设计与实现 

**Authors**: Priyanshu Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2506.06693)  

**Abstract**: This paper presents a comprehensive analysis of the RISC-V instruction set architecture, focusing on its modular design, implementation challenges, and performance characteristics. We examine the RV32I base instruction set with extensions for multiplication (M) and atomic operations (A). Through cycle-accurate simulation of a pipelined implementation, we evaluate performance metrics including CPI (cycles per instruction) and power efficiency. Our results demonstrate RISC-V's advantages in embedded systems and its scalability for custom accelerators. Comparative analysis shows a 17% reduction in power consumption compared to ARM Cortex-M0 implementations in similar process nodes. The open-standard nature of RISC-V provides significant flexibility for domain-specific optimizations. 

**Abstract (ZH)**: 本文对RISC-V指令集架构进行了全面分析，重点探讨其模块化设计、实现挑战及性能特征。我们研究了RV32I基础指令集，并包括乘法扩展(M)和原子操作扩展(A)。通过对流水线实现的时钟周期精确仿真，我们评估了每条指令周期数(CPI)和功耗效率等性能指标。结果表明，RISC-V在嵌入式系统中的优势及其面向自定义加速器的可扩展性。与类似工艺节点的ARM Cortex-M0实现相比，功耗降低了17%，展示了RISC-V开放标准带来的灵活性。 

---
# RoboPARA: Dual-Arm Robot Planning with Parallel Allocation and Recomposition Across Tasks 

**Title (ZH)**: RoboPARA：跨任务的并行分配与重组的双臂机器人规划 

**Authors**: Shiying Duan, Pei Ren, Nanxiang Jiang, Zhengping Che, Jian Tang, Yifan Sun, Zhaoxin Fan, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06683)  

**Abstract**: Dual-arm robots play a crucial role in improving efficiency and flexibility in complex multitasking scenarios. While existing methods have achieved promising results in task planning, they often fail to fully optimize task parallelism, limiting the potential of dual-arm collaboration. To address this issue, we propose RoboPARA, a novel large language model (LLM)-driven framework for dual-arm task parallelism planning. RoboPARA employs a two-stage process: (1) Dependency Graph-based Planning Candidates Generation, which constructs directed acyclic graphs (DAGs) to model task dependencies and eliminate redundancy, and (2) Graph Re-Traversal-based Dual-Arm Parallel Planning, which optimizes DAG traversal to maximize parallelism while maintaining task coherence. In addition, we introduce the Cross-Scenario Dual-Arm Parallel Task dataset (X-DAPT dataset), the first dataset specifically designed to evaluate dual-arm task parallelism across diverse scenarios and difficulty levels. Extensive experiments on the X-DAPT dataset demonstrate that RoboPARA significantly outperforms existing methods, achieving higher efficiency and reliability, particularly in complex task combinations. The code and dataset will be released upon acceptance. 

**Abstract (ZH)**: 双臂机器人在复杂多任务场景中对于提高效率和灵活性起着关键作用。尽管现有方法在任务规划方面取得了令人鼓舞的结果，但它们往往未能充分优化任务并行性，限制了双臂协作的潜力。为了解决这一问题，我们提出了一种新的基于大型语言模型（LLM）的双臂任务并行规划框架RoboPARA。RoboPARA采用两阶段过程：(1) 基于依赖图的规划候选生成，该过程构建有向无环图（DAG）以建模任务依赖关系并消除冗余；(2) 基于图重新遍历的双臂并行规划，该过程优化DAG遍历以最大化并行性同时保持任务连贯性。此外，我们引入了跨场景双臂并行任务数据集（X-DAPT数据集），这是第一个专门设计用于评估不同场景和难度级别下双臂任务并行性的数据集。在X-DAPT数据集上进行的大量实验表明，RoboPARA显著优于现有方法，特别是在复杂任务组合中实现更高的效率和可靠性。该代码和数据集将在接受后发布。 

---
# DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning 

**Title (ZH)**: DriveSuprim:向精确轨迹选择的端到端规划迈进 

**Authors**: Wenhao Yao, Zhenxin Li, Shiyi Lan, Zi Wang, Xinglong Sun, Jose M. Alvarez, Zuxuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06659)  

**Abstract**: In complex driving environments, autonomous vehicles must navigate safely. Relying on a single predicted path, as in regression-based approaches, usually does not explicitly assess the safety of the predicted trajectory. Selection-based methods address this by generating and scoring multiple trajectory candidates and predicting the safety score for each, but face optimization challenges in precisely selecting the best option from thousands of possibilities and distinguishing subtle but safety-critical differences, especially in rare or underrepresented scenarios. We propose DriveSuprim to overcome these challenges and advance the selection-based paradigm through a coarse-to-fine paradigm for progressive candidate filtering, a rotation-based augmentation method to improve robustness in out-of-distribution scenarios, and a self-distillation framework to stabilize training. DriveSuprim achieves state-of-the-art performance, reaching 93.5% PDMS in NAVSIM v1 and 87.1% EPDMS in NAVSIM v2 without extra data, demonstrating superior safetycritical capabilities, including collision avoidance and compliance with rules, while maintaining high trajectory quality in various driving scenarios. 

**Abstract (ZH)**: 在复杂驾驶环境中，自动驾驶车辆必须安全导航。依赖单一预测路径的方法，如基于回归的方法，通常不会明确评估预测轨迹的安全性。选择性方法通过生成和评分多个轨迹候选方案，并为每个方案预测安全得分来解决这一问题，但在从成千上万种可能性中精确选择最佳选项和区分细微但至关安全的差异方面面临优化挑战，尤其是在罕见或未充分代表的场景中。我们提出DriveSuprim以克服这些挑战并通过粗到细的渐进候选过滤 paradigmm、基于旋转的增强方法提高分布外场景的鲁棒性以及自我蒸馏框架稳定训练来推进选择性方法的范式。DriveSuprim在不使用额外数据的情况下达到了NAVSIM v1的93.5% PDMS和NAVSIM v2的87.1% EPDMS，展示出了优越的安全关键能力，包括碰撞避免和规则遵守，并在各种驾驶场景中保持了高质量的轨迹。 

---
# Self-Adapting Improvement Loops for Robotic Learning 

**Title (ZH)**: 自适应改进循环在机器人学习中的应用 

**Authors**: Calvin Luo, Zilai Zeng, Mingxi Jia, Yilun Du, Chen Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.06658)  

**Abstract**: Video generative models trained on expert demonstrations have been utilized as performant text-conditioned visual planners for solving robotic tasks. However, generalization to unseen tasks remains a challenge. Whereas improved generalization may be facilitated by leveraging learned prior knowledge from additional pre-collected offline data sources, such as web-scale video datasets, in the era of experience we aim to design agents that can continuously improve in an online manner from self-collected behaviors. In this work we thus propose the Self-Adapting Improvement Loop (SAIL), where an in-domain video model iteratively updates itself on self-produced trajectories, collected through adaptation with an internet-scale pretrained video model, and steadily improves its performance for a specified task of interest. We apply SAIL to a diverse suite of MetaWorld tasks, as well as two manipulation tasks on a real robot arm, and find that performance improvements continuously emerge over multiple iterations for novel tasks initially unseen during original in-domain video model training. Furthermore, we discover that SAIL is surprisingly robust regarding if and how the self-collected experience is filtered, and the quality of the initial in-domain demonstrations. Through adaptation with summarized internet-scale data, and learning through online experience, we thus demonstrate a way to iteratively bootstrap a high-performance video model for solving novel robotic tasks through self-improvement. 

**Abstract (ZH)**: 自适应改善循环（SAIL）：通过自我收集的行为连续改进领域内的视频生成模型以解决新型机器人任务 

---
# Quantile Regression with Large Language Models for Price Prediction 

**Title (ZH)**: 使用大型语言模型的分位数回归价格预测 

**Authors**: Nikhita Vedula, Dushyanta Dhyani, Laleh Jalali, Boris Oreshkin, Mohsen Bayati, Shervin Malmasi  

**Link**: [PDF](https://arxiv.org/pdf/2506.06657)  

**Abstract**: Large Language Models (LLMs) have shown promise in structured prediction tasks, including regression, but existing approaches primarily focus on point estimates and lack systematic comparison across different methods. We investigate probabilistic regression using LLMs for unstructured inputs, addressing challenging text-to-distribution prediction tasks such as price estimation where both nuanced text understanding and uncertainty quantification are critical. We propose a novel quantile regression approach that enables LLMs to produce full predictive distributions, improving upon traditional point estimates. Through extensive experiments across three diverse price prediction datasets, we demonstrate that a Mistral-7B model fine-tuned with quantile heads significantly outperforms traditional approaches for both point and distributional estimations, as measured by three established metrics each for prediction accuracy and distributional calibration. Our systematic comparison of LLM approaches, model architectures, training approaches, and data scaling reveals that Mistral-7B consistently outperforms encoder architectures, embedding-based methods, and few-shot learning methods. Our experiments also reveal the effectiveness of LLM-assisted label correction in achieving human-level accuracy without systematic bias. Our curated datasets are made available at this https URL to support future research. 

**Abstract (ZH)**: 大规模语言模型在概率回归任务中的应用：基于未结构化输入的量化回归方法研究 

---
# Non-Intrusive Load Monitoring Based on Image Load Signatures and Continual Learning 

**Title (ZH)**: 基于图像负载签名和持续学习的非侵入式负载监测 

**Authors**: Olimjon Toirov, Wei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06637)  

**Abstract**: Non-Intrusive Load Monitoring (NILM) identifies the operating status and energy consumption of each electrical device in the circuit by analyzing the electrical signals at the bus, which is of great significance for smart power management. However, the complex and changeable load combinations and application environments lead to the challenges of poor feature robustness and insufficient model generalization of traditional NILM methods. To this end, this paper proposes a new non-intrusive load monitoring method that integrates "image load signature" and continual learning. This method converts multi-dimensional power signals such as current, voltage, and power factor into visual image load feature signatures, and combines deep convolutional neural networks to realize the identification and classification of multiple devices; at the same time, self-supervised pre-training is introduced to improve feature generalization, and continual online learning strategies are used to overcome model forgetting to adapt to the emergence of new loads. This paper conducts a large number of experiments on high-sampling rate load datasets, and compares a variety of existing methods and model variants. The results show that the proposed method has achieved significant improvements in recognition accuracy. 

**Abstract (ZH)**: 非侵入式负载监测方法整合“图像负载签名”和持续学习技术 

---
# Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning 

**Title (ZH)**: 从易到难任务的 Curriculum 强化学习提高大模型推理能力 

**Authors**: Shubham Parashar, Shurui Gui, Xiner Li, Hongyi Ling, Sushil Vemuri, Blake Olson, Eric Li, Yu Zhang, James Caverlee, Dileep Kalathil, Shuiwang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2506.06632)  

**Abstract**: We aim to improve the reasoning capabilities of language models via reinforcement learning (RL). Recent RL post-trained models like DeepSeek-R1 have demonstrated reasoning abilities on mathematical and coding tasks. However, prior studies suggest that using RL alone to improve reasoning on inherently difficult tasks is less effective. Here, we draw inspiration from curriculum learning and propose to schedule tasks from easy to hard (E2H), allowing LLMs to build reasoning skills gradually. Our method is termed E2H Reasoner. Empirically, we observe that, although easy tasks are important initially, fading them out through appropriate scheduling is essential in preventing overfitting. Theoretically, we establish convergence guarantees for E2H Reasoner within an approximate policy iteration framework. We derive finite-sample complexity bounds and show that when tasks are appropriately decomposed and conditioned, learning through curriculum stages requires fewer total samples than direct learning. Experiments across multiple domains show that E2H Reasoner significantly improves the reasoning ability of small LLMs (1.5B to 3B), which otherwise struggle when trained with vanilla RL alone, highlighting the effectiveness of our method. 

**Abstract (ZH)**: 我们通过强化学习（RL）旨在提高语言模型的推理能力。最近的后训练RL模型如DeepSeek-R1在数学和编码任务中展示了推理能力。然而，之前的研究表明，单独使用RL来提高本质上困难任务的推理能力效果不佳。为此，我们借鉴了课程学习的方法，提出从容易的任务逐步过渡到困难的任务（E2H），使大语言模型（LLMs）能够逐步建立推理技能。我们的方法称为E2H推理器。实验上，我们观察到虽然初始阶段容易的任务很重要，但通过适当的调度逐渐减少容易任务是防止过拟合的关键。理论上，我们以内点逼近策略迭代框架建立了E2H推理器的收敛性保证。我们推导了有限样本复杂性边界，并证明适当分解和条件化任务后，在课程阶段学习所需的总样本量少于直接学习。跨多个领域的实验表明，E2H推理器显著提高了小规模LLMs（1.5B至3B参数）的推理能力，这些模型单独使用朴素的RL训练时表现不佳，突显了我们方法的有效性。 

---
# Active Test-time Vision-Language Navigation 

**Title (ZH)**: 主动测试时视觉-语言导航 

**Authors**: Heeju Ko, Sungjune Kim, Gyeongrok Oh, Jeongyoon Yoon, Honglak Lee, Sujin Jang, Seungryong Kim, Sangpil Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.06630)  

**Abstract**: Vision-Language Navigation (VLN) policies trained on offline datasets often exhibit degraded task performance when deployed in unfamiliar navigation environments at test time, where agents are typically evaluated without access to external interaction or feedback. Entropy minimization has emerged as a practical solution for reducing prediction uncertainty at test time; however, it can suffer from accumulated errors, as agents may become overconfident in incorrect actions without sufficient contextual grounding. To tackle these challenges, we introduce ATENA (Active TEst-time Navigation Agent), a test-time active learning framework that enables a practical human-robot interaction via episodic feedback on uncertain navigation outcomes. In particular, ATENA learns to increase certainty in successful episodes and decrease it in failed ones, improving uncertainty calibration. Here, we propose mixture entropy optimization, where entropy is obtained from a combination of the action and pseudo-expert distributions-a hypothetical action distribution assuming the agent's selected action to be optimal-controlling both prediction confidence and action preference. In addition, we propose a self-active learning strategy that enables an agent to evaluate its navigation outcomes based on confident predictions. As a result, the agent stays actively engaged throughout all iterations, leading to well-grounded and adaptive decision-making. Extensive evaluations on challenging VLN benchmarks-REVERIE, R2R, and R2R-CE-demonstrate that ATENA successfully overcomes distributional shifts at test time, outperforming the compared baseline methods across various settings. 

**Abstract (ZH)**: ATENA：测试时主动学习导航代理 

---
# \textit{QuantMCP}: Grounding Large Language Models in Verifiable Financial Reality 

**Title (ZH)**: QuantMCP: 将大型语言模型与可验证的金融现实对接 

**Authors**: Yifan Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2506.06622)  

**Abstract**: Large Language Models (LLMs) hold immense promise for revolutionizing financial analysis and decision-making, yet their direct application is often hampered by issues of data hallucination and lack of access to real-time, verifiable financial information. This paper introduces QuantMCP, a novel framework designed to rigorously ground LLMs in financial reality. By leveraging the Model Context Protocol (MCP) for standardized and secure tool invocation, QuantMCP enables LLMs to accurately interface with a diverse array of Python-accessible financial data APIs (e.g., Wind, yfinance). Users can interact via natural language to precisely retrieve up-to-date financial data, thereby overcoming LLM's inherent limitations in factual data recall. More critically, once furnished with this verified, structured data, the LLM's analytical capabilities are unlocked, empowering it to perform sophisticated data interpretation, generate insights, and ultimately support more informed financial decision-making processes. QuantMCP provides a robust, extensible, and secure bridge between conversational AI and the complex world of financial data, aiming to enhance both the reliability and the analytical depth of LLM applications in finance. 

**Abstract (ZH)**: Large Language Models (LLMs)在金融分析与决策中的潜力巨大，然而其直接应用常常受到数据幻想和实时可验证金融信息缺乏的阻碍。本文介绍了一种名为QuantMCP的新型框架，旨在严格将LLMs与金融现实相结合。通过利用标准和安全的Model Context Protocol (MCP) 来调用工具，QuantMCP使LLMs能够准确地与多种可访问Python接口的金融数据API（如Wind、yfinance）对接。用户可以通过自然语言精确检索最新金融数据，从而克服LLMs在事实数据回忆上的固有限制。更为关键的是，一旦获得这些验证过的结构化数据，LLMs的分析能力将被解锁，使其能够进行复杂的数据解释、生成洞见，并最终支持更明智的金融决策过程。QuantMCP为对话式AI与复杂的金融数据世界之间提供了一个稳健、可扩展且安全的桥梁，旨在增强金融领域LLM应用的可靠性和分析深度。 

---
# Training-Free Tokenizer Transplantation via Orthogonal Matching Pursuit 

**Title (ZH)**: 无需训练的Tokenizer移植：正交匹配 pursuit方法 

**Authors**: Charles Goddard, Fernando Fernandes Neto  

**Link**: [PDF](https://arxiv.org/pdf/2506.06607)  

**Abstract**: We present a training-free method to transplant tokenizers in pretrained large language models (LLMs) by reconstructing unseen token embeddings via Orthogonal Matching Pursuit (OMP). Specifically, we approximate each out-of-vocabulary token as a sparse linear combination of shared tokens, in two phases: first, compute each new token's representation in the donor embedding space with a small dictionary of shared anchor tokens, then transfer these same sparse coefficients back into the base model's embedding space.
On two challenging cross-tokenizer tasks--Llama$\to$Mistral NeMo (12B) and Qwen$\to$Llama (1B)--we show that OMP achieves best zero-shot preservation of the base model's performance across multiple benchmarks, while other zero-shot approaches degrade significantly. Compared to baselines (zero-init, mean-init, and existing approaches like WECHSEL, FOCUS, ZETT), OMP consistently achieves the best overall performance, effectively bridging large tokenizer discrepancies without gradient updates. Our analysis further identifies mismatched numerical tokenization schemes as a critical challenge for preserving mathematical reasoning capabilities. This technique enables direct reuse of pretrained model weights with new tokenizers, facilitating cross-tokenizer knowledge distillation, speculative decoding, ensembling, merging, and domain-specific vocabulary adaptations. We integrate our method into the open-source mergekit-tokensurgeon tool for post hoc vocabulary realignment. 

**Abstract (ZH)**: 无需训练的基于正交匹配追迹的方法实现预训练大规模语言模型（LLMs）中的分词器移植，通过重建未见过的分词嵌入。 

---
# MedCite: Can Language Models Generate Verifiable Text for Medicine? 

**Title (ZH)**: MedCite: 语言模型能否生成可验证的医学文本？ 

**Authors**: Xiao Wang, Mengjue Tan, Qiao Jin, Guangzhi Xiong, Yu Hu, Aidong Zhang, Zhiyong Lu, Minjia Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06605)  

**Abstract**: Existing LLM-based medical question-answering systems lack citation generation and evaluation capabilities, raising concerns about their adoption in practice. In this work, we introduce \name, the first end-to-end framework that facilitates the design and evaluation of citation generation with LLMs for medical tasks. Meanwhile, we introduce a novel multi-pass retrieval-citation method that generates high-quality citations. Our evaluation highlights the challenges and opportunities of citation generation for medical tasks, while identifying important design choices that have a significant impact on the final citation quality. Our proposed method achieves superior citation precision and recall improvements compared to strong baseline methods, and we show that evaluation results correlate well with annotation results from professional experts. 

**Abstract (ZH)**: 现有的基于LLM的医学问答系统缺乏引文生成和评估能力，这对其实际应用提出了担忧。在本文中，我们引入了\name，这是首个用于医学任务的端到端框架，该框架促进了基于LLM的引文生成设计和评估。同时，我们引入了一种新颖的多轮检索-引文生成方法，能够生成高质量的引文。我们的评估突显了医学任务中引文生成面临的挑战与机遇，并且识别出了对最终引文质量有显著影响的关键设计选择。我们提出的方法在引文精度和召回率上均优于强基线方法，并且展示了评估结果与专业专家标注结果之间的一致性。 

---
# CAtCh: Cognitive Assessment through Cookie Thief 

**Title (ZH)**: CAtCh: 基于认知评估的饼干小偷任务 

**Authors**: Joseph T Colonel, Carolyn Hagler, Guiselle Wismer, Laura Curtis, Jacqueline Becker, Juan Wisnivesky, Alex Federman, Gaurav Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2506.06603)  

**Abstract**: Several machine learning algorithms have been developed for the prediction of Alzheimer's disease and related dementia (ADRD) from spontaneous speech. However, none of these algorithms have been translated for the prediction of broader cognitive impairment (CI), which in some cases is a precursor and risk factor of ADRD. In this paper, we evaluated several speech-based open-source methods originally proposed for the prediction of ADRD, as well as methods from multimodal sentiment analysis for the task of predicting CI from patient audio recordings. Results demonstrated that multimodal methods outperformed unimodal ones for CI prediction, and that acoustics-based approaches performed better than linguistics-based ones. Specifically, interpretable acoustic features relating to affect and prosody were found to significantly outperform BERT-based linguistic features and interpretable linguistic features, respectively. All the code developed for this study is available at this https URL. 

**Abstract (ZH)**: 几种机器学习算法已被用于自发言语预测阿尔茨海默病及相关痴呆（ADRD），然而这些算法尚未用于更广泛认知损害（CI）的预测，而后者在某些情况下是ADRD的前兆和风险因素。本文评估了几种 originally proposed 用于预测ADRD的基于言语的开源方法，以及来自多模态情感分析的方法，用于从患者录音中预测CI的任务。结果表明，多模态方法在CI预测中优于单模态方法，而基于声学的方法优于基于语言学的方法。具体而言，与情感和语调相关的可解释声学特征显著优于基于BERT的语言学特征和可解释语言学特征。本文开发的所有代码均可在此 URL 获取。 

---
# From Model-Based and Adaptive Control to Evolving Fuzzy Control 

**Title (ZH)**: 从模型基础和自适应控制到演化模糊控制 

**Authors**: Daniel Leite, Igor Škrjanc, Fernando Gomide  

**Link**: [PDF](https://arxiv.org/pdf/2506.06594)  

**Abstract**: Evolving fuzzy systems build and adapt fuzzy models - such as predictors and controllers - by incrementally updating their rule-base structure from data streams. On the occasion of the 60-year anniversary of fuzzy set theory, commemorated during the Fuzz-IEEE 2025 event, this brief paper revisits the historical development and core contributions of classical fuzzy and adaptive modeling and control frameworks. It then highlights the emergence and significance of evolving intelligent systems in fuzzy modeling and control, emphasizing their advantages in handling nonstationary environments. Key challenges and future directions are discussed, including safety, interpretability, and principled structural evolution. 

**Abstract (ZH)**: 演化模糊系统通过增量更新规则基结构从数据流中构建和适应模糊模型——如预测器和控制器。在模糊集理论诞辰60周年之际，Fuzz-IEEE 2025会议期间，本文简要回顾了经典模糊和自适应建模与控制框架的历史发展和核心贡献，强调了在模糊建模与控制中演化智能系统的发展及其在处理非稳定环境方面的优势，讨论了安全性、可解释性和原理性的结构演化等关键挑战和未来方向。 

---
# Towards Efficient Multi-LLM Inference: Characterization and Analysis of LLM Routing and Hierarchical Techniques 

**Title (ZH)**: 面向高效多大型语言模型推理：LLM 路由及层次化技术的特性和分析 

**Authors**: Adarsh Prasad Behera, Jaya Prakash Champati, Roberto Morabito, Sasu Tarkoma, James Gross  

**Link**: [PDF](https://arxiv.org/pdf/2506.06579)  

**Abstract**: Recent progress in Language Models (LMs) has dramatically advanced the field of natural language processing (NLP), excelling at tasks like text generation, summarization, and question answering. However, their inference remains computationally expensive and energy intensive, especially in settings with limited hardware, power, or bandwidth. This makes it difficult to deploy LMs in mobile, edge, or cost sensitive environments. To address these challenges, recent approaches have introduced multi LLM intelligent model selection strategies that dynamically allocate computational resources based on query complexity -- using lightweight models for simpler queries and escalating to larger models only when necessary. This survey explores two complementary strategies for efficient LLM inference: (i) routing, which selects the most suitable model based on the query, and (ii) cascading or hierarchical inference (HI), which escalates queries through a sequence of models until a confident response is found. Both approaches aim to reduce computation by using lightweight models for simpler tasks while offloading only when needed. We provide a comparative analysis of these techniques across key performance metrics, discuss benchmarking efforts, and outline open challenges. Finally, we outline future research directions to enable faster response times, adaptive model selection based on task complexity, and scalable deployment across heterogeneous environments, making LLM based systems more efficient and accessible for real world applications. 

**Abstract (ZH)**: 近期语言模型的进步显著推动了自然语言处理领域的发展，特别是在文本生成、总结和问答等任务上表现出色。然而，它们的推理过程仍具有较高的计算成本和能源消耗，特别是在硬件、电力或带宽有限的环境中。这使得在移动、边缘或成本敏感的环境中部署语言模型变得困难。为应对这些挑战，近期的方法引入了多语言模型智能选择策略，根据查询复杂度动态分配计算资源——对于简单的查询使用轻量级模型，仅在必要时升级到更大的模型。本文综述了两种互补的高效语言模型推理策略：(i) 路由，根据查询选择最合适的模型；(ii) 级联或层次推理 (HI)，通过一系列模型序列逐步升级查询，直至找到可信的响应。本文比较了这些技术在关键性能指标上的差异，讨论了基准测试努力，并概述了开放性挑战。最后，提出了未来研究方向，以实现更快的响应时间、根据任务复杂度进行自适应模型选择，并在异构环境中实现可扩展部署，从而使基于语言模型的系统更高效且易于在实际应用中获得。 

---
# Future of Work with AI Agents: Auditing Automation and Augmentation Potential across the U.S. Workforce 

**Title (ZH)**: AI代理的未来工作：审计美国劳动力的自动化与增强潜力 

**Authors**: Yijia Shao, Humishka Zope, Yucheng Jiang, Jiaxin Pei, David Nguyen, Erik Brynjolfsson, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06576)  

**Abstract**: The rapid rise of compound AI systems (a.k.a., AI agents) is reshaping the labor market, raising concerns about job displacement, diminished human agency, and overreliance on automation. Yet, we lack a systematic understanding of the evolving landscape. In this paper, we address this gap by introducing a novel auditing framework to assess which occupational tasks workers want AI agents to automate or augment, and how those desires align with the current technological capabilities. Our framework features an audio-enhanced mini-interview to capture nuanced worker desires and introduces the Human Agency Scale (HAS) as a shared language to quantify the preferred level of human involvement. Using this framework, we construct the WORKBank database, building on the U.S. Department of Labor's O*NET database, to capture preferences from 1,500 domain workers and capability assessments from AI experts across over 844 tasks spanning 104 occupations. Jointly considering the desire and technological capability divides tasks in WORKBank into four zones: Automation "Green Light" Zone, Automation "Red Light" Zone, R&D Opportunity Zone, Low Priority Zone. This highlights critical mismatches and opportunities for AI agent development. Moving beyond a simple automate-or-not dichotomy, our results reveal diverse HAS profiles across occupations, reflecting heterogeneous expectations for human involvement. Moreover, our study offers early signals of how AI agent integration may reshape the core human competencies, shifting from information-focused skills to interpersonal ones. These findings underscore the importance of aligning AI agent development with human desires and preparing workers for evolving workplace dynamics. 

**Abstract (ZH)**: 快速崛起的组合AI系统（即AI代理）正重塑劳动市场，引发了关于工作岗位替代、人类自主性减弱以及过度依赖自动化的担忧。然而，我们缺乏对这一演变景观的系统性理解。本文通过引入一种新的审计框架来解决这一缺口，该框架评估工人希望AI代理自动化或增强哪些职业任务，以及这些愿望与当前技术能力的契合程度。我们的框架包括音频增强的简短访谈，以捕捉工人复杂的需求，并引入人类自主性量表（HAS）作为共享语言来量化期望的人类参与水平。通过该框架，我们构建了WORKBank数据库，基于美国劳工部的O*NET数据库，涵盖了来自1,500名领域工人和来自844个任务（涉及104种职业）的AI专家的能力评估。结合意愿和技术能力，将WORKBank的任务划分为四个区域：自动化“绿灯”区、自动化“红灯”区、研发机会区、低优先级区。这突显了关键的不匹配和AI代理开发中的机遇。超越简单地决定自动化与否的二分法，我们的结果揭示了不同职业中具有多样性的HAS配置文件，反映了对人类参与的异质性预期。此外，我们的研究提供了有关AI代理整合可能如何重塑核心人类技能的早期信号，从信息聚焦技能转向人际技能。这些发现强调了将AI代理开发与人类愿望相一致的重要性，并为工人适应不断变化的工作场所动态做好准备。 

---
# Graph Persistence goes Spectral 

**Title (ZH)**: 谱图持久性 

**Authors**: Mattie Ji, Amauri H. Souza, Vikas Garg  

**Link**: [PDF](https://arxiv.org/pdf/2506.06571)  

**Abstract**: Including intricate topological information (e.g., cycles) provably enhances the expressivity of message-passing graph neural networks (GNNs) beyond the Weisfeiler-Leman (WL) hierarchy. Consequently, Persistent Homology (PH) methods are increasingly employed for graph representation learning. In this context, recent works have proposed decorating classical PH diagrams with vertex and edge features for improved expressivity. However, due to their dependence on features, these methods still fail to capture basic graph structural information. In this paper, we propose SpectRe -- a new topological descriptor for graphs that integrates spectral information into PH diagrams. Notably, SpectRe is strictly more expressive than existing descriptors on graphs. We also introduce notions of global and local stability to analyze existing descriptors and establish that SpectRe is locally stable. Finally, experiments on synthetic and real-world datasets demonstrate the effectiveness of SpectRe and its potential to enhance the capabilities of graph models in relevant learning tasks. 

**Abstract (ZH)**: 包括复杂的拓扑信息（如环）可证地增强消息传递图神经网络（GNNs）的表征能力，超越魏谢夫勒-列曼（WL）层次结构。因此，持久同调（PH）方法越来越多地被用于图表示学习。在此背景下，最近的工作提出了在经典PH图上添加顶点和边特征来提高表征能力。然而，由于对特征的依赖性，这些方法仍然无法捕捉基本的图结构信息。在本文中，我们提出了一种新的拓扑描述符SpectRe，它将谱信息整合到PH图中。值得注意的是，SpectRe在图上的表征能力严格优于现有的描述符。我们还引入了全局稳定性和局部稳定性概念来分析现有描述符，并证明SpectRe具有局部稳定性。最后，合成和真实世界数据集上的实验表明SpectRe的有效性及其在相关学习任务中增强图模型能力的潜力。 

---
# Textile Analysis for Recycling Automation using Transfer Learning and Zero-Shot Foundation Models 

**Title (ZH)**: 基于迁移学习和零样本基础模型的纺织品回收自动化分析 

**Authors**: Yannis Spyridis, Vasileios Argyriou  

**Link**: [PDF](https://arxiv.org/pdf/2506.06569)  

**Abstract**: Automated sorting is crucial for improving the efficiency and scalability of textile recycling, but accurately identifying material composition and detecting contaminants from sensor data remains challenging. This paper investigates the use of standard RGB imagery, a cost-effective sensing modality, for key pre-processing tasks in an automated system. We present computer vision components designed for a conveyor belt setup to perform (a) classification of four common textile types and (b) segmentation of non-textile features such as buttons and zippers. For classification, several pre-trained architectures were evaluated using transfer learning and cross-validation, with EfficientNetB0 achieving the best performance on a held-out test set with 81.25\% accuracy. For feature segmentation, a zero-shot approach combining the Grounding DINO open-vocabulary detector with the Segment Anything Model (SAM) was employed, demonstrating excellent performance with a mIoU of 0.90 for the generated masks against ground truth. This study demonstrates the feasibility of using RGB images coupled with modern deep learning techniques, including transfer learning for classification and foundation models for zero-shot segmentation, to enable essential analysis steps for automated textile recycling pipelines. 

**Abstract (ZH)**: 自动分拣对于提高纺织回收的效率和扩展性至关重要，但准确识别材料组成并从传感器数据中检测污染仍然具有挑战性。本文调查了使用标准RGB图像这一低成本传感模式在自动化系统中执行关键预处理任务的应用。我们介绍了为传送带设置设计的计算机视觉组件，以实现（a）四种常见纺织品类型的分类和（b）非纺织特征（如纽扣和拉链）的分割。在分类方面，通过迁移学习和交叉验证评估了多种预先训练的架构，EfficientNetB0在保留的测试集上取得了81.25%的准确性，表现最佳。在特征分割方面，采用了结合Grounding DINO开放词汇检测器和Segment Anything Model（SAM）的零样本方法，生成的掩膜与真实值的mIoU达到0.90，表现出色。本研究证明了使用RGB图像结合现代深度学习技术，包括分类中的迁移学习和零样本分割中的基础模型，以实现自动化纺织回收管道中的关键分析步骤的可行性。 

---
# AS-ASR: A Lightweight Framework for Aphasia-Specific Automatic Speech Recognition 

**Title (ZH)**: AS-ASR：一种轻量级的专用于失语症的自动语音识别框架 

**Authors**: Chen Bao, Chuanbing Huo, Qinyu Chen, Chang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.06566)  

**Abstract**: This paper proposes AS-ASR, a lightweight aphasia-specific speech recognition framework based on Whisper-tiny, tailored for low-resource deployment on edge devices. Our approach introduces a hybrid training strategy that systematically combines standard and aphasic speech at varying ratios, enabling robust generalization, and a GPT-4-based reference enhancement method that refines noisy aphasic transcripts, improving supervision quality. We conduct extensive experiments across multiple data mixing configurations and evaluation settings. Results show that our fine-tuned model significantly outperforms the zero-shot baseline, reducing WER on aphasic speech by over 30% while preserving performance on standard speech. The proposed framework offers a scalable, efficient solution for real-world disordered speech recognition. 

**Abstract (ZH)**: 本文提出了一种基于Whisper-tiny的轻量级自闭塞性失语症识别框架AS-ASR，针对边缘设备低资源部署进行了优化。该方法引入了一种混合训练策略，系统地结合标准言语和失语症言语的不同比例数据，实现了稳健的泛化能力，并提出了一种基于GPT-4的参考增强方法，以精化失语症记录中的噪声，提高监督质量。我们在多种数据混搭配置和评估设置下进行了广泛的实验。结果表明，我们的微调模型显著优于零样本基线，失语症言语的wer降低了超过30%，同时保持了对标准言语的性能。所提出框架提供了面向实际失序言语识别的可扩展和高效解决方案。 

---
# LaMP-Cap: Personalized Figure Caption Generation With Multimodal Figure Profiles 

**Title (ZH)**: LaMP-Cap：基于多模态图谱的个性化图表描述生成 

**Authors**: Ho Yin 'Sam' Ng, Ting-Yao Hsu, Aashish Anantha Ramakrishnan, Branislav Kveton, Nedim Lipka, Franck Dernoncourt, Dongwon Lee, Tong Yu, Sungchul Kim, Ryan A. Rossi, Ting-Hao 'Kenneth' Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06561)  

**Abstract**: Figure captions are crucial for helping readers understand and remember a figure's key message. Many models have been developed to generate these captions, helping authors compose better quality captions more easily. Yet, authors almost always need to revise generic AI-generated captions to match their writing style and the domain's style, highlighting the need for personalization. Despite language models' personalization (LaMP) advances, these technologies often focus on text-only settings and rarely address scenarios where both inputs and profiles are multimodal. This paper introduces LaMP-Cap, a dataset for personalized figure caption generation with multimodal figure profiles. For each target figure, LaMP-Cap provides not only the needed inputs, such as figure images, but also up to three other figures from the same document--each with its image, caption, and figure-mentioning paragraphs--as a profile to characterize the context. Experiments with four LLMs show that using profile information consistently helps generate captions closer to the original author-written ones. Ablation studies reveal that images in the profile are more helpful than figure-mentioning paragraphs, highlighting the advantage of using multimodal profiles over text-only ones. 

**Abstract (ZH)**: 个性化多模态图例生成数据集LaMP-Cap：基于多模态图例配置的个性化图例生成 

---
# KramaBench: A Benchmark for AI Systems on Data-to-Insight Pipelines over Data Lakes 

**Title (ZH)**: KramaBench：用于数据湖上数据到洞察管道的AI系统基准测试 

**Authors**: Eugenie Lai, Gerardo Vitagliano, Ziyu Zhang, Sivaprasad Sudhir, Om Chabra, Anna Zeng, Anton A. Zabreyko, Chenning Li, Ferdi Kossmann, Jialin Ding, Jun Chen, Markos Markakis, Matthew Russo, Weiyang Wang, Ziniu Wu, Michael J. Cafarella, Lei Cao, Samuel Madden, Tim Kraska  

**Link**: [PDF](https://arxiv.org/pdf/2506.06541)  

**Abstract**: Constructing real-world data-to-insight pipelines often involves data extraction from data lakes, data integration across heterogeneous data sources, and diverse operations from data cleaning to analysis. The design and implementation of data science pipelines require domain knowledge, technical expertise, and even project-specific insights. AI systems have shown remarkable reasoning, coding, and understanding capabilities. However, it remains unclear to what extent these capabilities translate into successful design and execution of such complex pipelines. We introduce KRAMABENCH: a benchmark composed of 104 manually-curated real-world data science pipelines spanning 1700 data files from 24 data sources in 6 different domains. We show that these pipelines test the end-to-end capabilities of AI systems on data processing, requiring data discovery, wrangling and cleaning, efficient processing, statistical reasoning, and orchestrating data processing steps given a high-level task. Our evaluation tests 5 general models and 3 code generation models using our reference framework, DS-GURU, which instructs the AI model to decompose a question into a sequence of subtasks, reason through each step, and synthesize Python code that implements the proposed design. Our results on KRAMABENCH show that, although the models are sufficiently capable of solving well-specified data science code generation tasks, when extensive data processing and domain knowledge are required to construct real-world data science pipelines, existing out-of-box models fall short. Progress on KramaBench represents crucial steps towards developing autonomous data science agents for real-world applications. Our code, reference framework, and data are available at this https URL. 

**Abstract (ZH)**: 构建现实世界的数据到见解管道通常涉及从数据湖中抽取数据、跨异构数据源进行数据集成，以及从数据清洗到分析的各种操作。数据科学管道的设计与实现需要领域知识、技术专长，甚至项目特定的洞察。AI系统展示了卓越的推理、编码和理解能力。然而，这些能力在成功设计和执行如此复杂的管道中的程度尚不清楚。我们介绍了KRAMABENCH：一个由104个手工整理的真实世界数据科学管道组成的基准，这些管道涵盖了来自6个不同领域、24个数据源的1700个数据文件。我们展示了这些管道测试了AI系统在整个数据处理流程中的端到端能力，包括数据发现、整理和清洗、高效处理、统计推理以及根据高级任务协调数据处理步骤。我们使用参考框架DS-GURU测试了5个通用模型和3个代码生成模型，该框架指导AI模型将一个问题分解为一系列子任务，逐步推理并通过合成Python代码实现提出的解决方案。KRAMABENCH的结果显示，尽管模型足够强大以解决明确定义的数据科学代码生成任务，但当需要大量数据处理和领域知识来构建真实世界的数据科学管道时，现有的开箱即用模型表现不足。KramaBench的进步代表了开发自主数据科学代理应用于真实世界应用的重要步骤。我们的代码、参考框架和数据可在以下网址获取。 

---
# Large Language Models Can Be a Viable Substitute for Expert Political Surveys When a Shock Disrupts Traditional Measurement Approaches 

**Title (ZH)**: 大规模语言模型可以在传统测量方法受到冲击时成为专家政治民意调查的可行替代方案。 

**Authors**: Patrick Y. Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06540)  

**Abstract**: After a disruptive event or shock, such as the Department of Government Efficiency (DOGE) federal layoffs of 2025, expert judgments are colored by knowledge of the outcome. This can make it difficult or impossible to reconstruct the pre-event perceptions needed to study the factors associated with the event. This position paper argues that large language models (LLMs), trained on vast amounts of digital media data, can be a viable substitute for expert political surveys when a shock disrupts traditional measurement. We analyze the DOGE layoffs as a specific case study for this position. We use pairwise comparison prompts with LLMs and derive ideology scores for federal executive agencies. These scores replicate pre-layoff expert measures and predict which agencies were targeted by DOGE. We also use this same approach and find that the perceptions of certain federal agencies as knowledge institutions predict which agencies were targeted by DOGE, even when controlling for ideology. This case study demonstrates that using LLMs allows us to rapidly and easily test the associated factors hypothesized behind the shock. More broadly, our case study of this recent event exemplifies how LLMs offer insights into the correlational factors of the shock when traditional measurement techniques fail. We conclude by proposing a two-part criterion for when researchers can turn to LLMs as a substitute for expert political surveys. 

**Abstract (ZH)**: 大型语言模型在颠覆性事件发生后的专家判断替代作用：以政府效率部门（DOGE）联邦裁员为例 

---
# Beyond Facts: Evaluating Intent Hallucination in Large Language Models 

**Title (ZH)**: 超越事实：评估大型语言模型中意图幻觉 

**Authors**: Yijie Hao, Haofei Yu, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2506.06539)  

**Abstract**: When exposed to complex queries containing multiple conditions, today's large language models (LLMs) tend to produce responses that only partially satisfy the query while neglecting certain conditions. We therefore introduce the concept of Intent Hallucination. In this phenomenon, LLMs either omit (neglecting to address certain parts) or misinterpret (responding to invented query parts) elements of the given query, leading to intent hallucinated generation. To systematically evaluate intent hallucination, we introduce FAITHQA, a novel benchmark for intent hallucination that contains 20,068 problems, covering both query-only and retrieval-augmented generation (RAG) setups with varying topics and difficulty. FAITHQA is the first hallucination benchmark that goes beyond factual verification, tailored to identify the fundamental cause of intent hallucination. By evaluating various LLMs on FAITHQA, we find that (1) intent hallucination is a common issue even for state-of-the-art models, and (2) the phenomenon stems from omission or misinterpretation of LLMs. To facilitate future research, we introduce an automatic LLM generation evaluation metric, CONSTRAINT SCORE, for detecting intent hallucination. Human evaluation results demonstrate that CONSTRAINT SCORE is closer to human performance for intent hallucination compared to baselines. 

**Abstract (ZH)**: 当面对包含多个条件的复杂查询时，当今的大规模语言模型往往会生成部分满足查询但忽视某些条件的响应。因此，我们引入了意图幻觉的概念。在这一现象中，大规模语言模型要么忽略了给定查询的部分内容，要么误解了给定查询的部分内容，导致意图被幻觉生成。为了系统地评估意图幻觉，我们引入了FAITHQA这一创新的幻觉基准，包含20,068个问题，涵盖了各种主题和难度的查询生成和检索增强生成（RAG）设置。FAITHQA是第一个超越事实验证的幻觉基准，旨在识别意图幻觉的根本原因。通过在FAITHQA上评估各种大规模语言模型，我们发现（1）即使对于最先进的模型，意图幻觉也是一个常见问题，（2）这一现象源于大规模语言模型的遗漏或误解。为了促进未来的研究，我们引入了约束得分CONSTRAINT SCORE这一自动的大规模语言模型生成评估指标，用于检测意图幻觉。人工评估结果表明，与基准相比，约束得分更接近于人类在意图幻觉评估中的表现。 

---
# Hierarchical and Collaborative LLM-Based Control for Multi-UAV Motion and Communication in Integrated Terrestrial and Non-Terrestrial Networks 

**Title (ZH)**: 基于层级协作的大语言模型控制在集成 terrestrial 和非terrestrial 网络中的多无人机运动和通信控制 

**Authors**: Zijiang Yan, Hao Zhou, Jianhua Pei, Hina Tabassum  

**Link**: [PDF](https://arxiv.org/pdf/2506.06532)  

**Abstract**: Unmanned aerial vehicles (UAVs) have been widely adopted in various real-world applications. However, the control and optimization of multi-UAV systems remain a significant challenge, particularly in dynamic and constrained environments. This work explores the joint motion and communication control of multiple UAVs operating within integrated terrestrial and non-terrestrial networks that include high-altitude platform stations (HAPS). Specifically, we consider an aerial highway scenario in which UAVs must accelerate, decelerate, and change lanes to avoid collisions and maintain overall traffic flow. Different from existing studies, we propose a novel hierarchical and collaborative method based on large language models (LLMs). In our approach, an LLM deployed on the HAPS performs UAV access control, while another LLM onboard each UAV handles motion planning and control. This LLM-based framework leverages the rich knowledge embedded in pre-trained models to enable both high-level strategic planning and low-level tactical decisions. This knowledge-driven paradigm holds great potential for the development of next-generation 3D aerial highway systems. Experimental results demonstrate that our proposed collaborative LLM-based method achieves higher system rewards, lower operational costs, and significantly reduced UAV collision rates compared to baseline approaches. 

**Abstract (ZH)**: 无人驾驶航空器（UAVs）已在各种实际应用场景中广泛采用。然而，多UAV系统的控制与优化仍是一个重大挑战，尤其是在动态和受限环境中。本研究探讨了在综合地面和非地面网络中，包括高空平台站（HAPS）的多UAV联合运动和通信控制。具体而言，我们考虑了一个空中高速公路场景，在该场景中，UAVs必须加速、减速并变道以避免碰撞并维持整体交通流。不同于现有研究，我们提出了一种基于大型语言模型（LLMs）的新型分层协作方法。在我们的方法中，部署在HAPS上的LLM负责UAV接入控制，而每个UAV上的LLM处理运动规划和控制。基于LLM的框架利用预训练模型中嵌入的豐富知识，既能够实现高层次的战略规划，也能做出低层次的战术决策。这一知识驱动的范式对下一代三维空中高速公路系统的开发具有巨大潜力。实验结果表明，我们提出的协作LLM基方法在系统奖励、运营成本和UAV碰撞率方面均优于基线方法。 

---
# Fixing It in Post: A Comparative Study of LLM Post-Training Data Quality and Model Performance 

**Title (ZH)**: 修复在后处理：LLM后训练数据质量与模型性能的比较研究 

**Authors**: Aladin Djuhera, Swanand Ravindra Kadhe, Syed Zawad, Farhan Ahmed, Heiko Ludwig, Holger Boche  

**Link**: [PDF](https://arxiv.org/pdf/2506.06522)  

**Abstract**: Recent work on large language models (LLMs) has increasingly focused on post-training and alignment with datasets curated to enhance instruction following, world knowledge, and specialized skills. However, most post-training datasets used in leading open- and closed-source LLMs remain inaccessible to the public, with limited information about their construction process. This lack of transparency has motivated the recent development of open-source post-training corpora. While training on these open alternatives can yield performance comparable to that of leading models, systematic comparisons remain challenging due to the significant computational cost of conducting them rigorously at scale, and are therefore largely absent. As a result, it remains unclear how specific samples, task types, or curation strategies influence downstream performance when assessing data quality. In this work, we conduct the first comprehensive side-by-side analysis of two prominent open post-training datasets: Tulu-3-SFT-Mix and SmolTalk. Using the Magpie framework, we annotate each sample with detailed quality metrics, including turn structure (single-turn vs. multi-turn), task category, input quality, and response quality, and we derive statistics that reveal structural and qualitative similarities and differences between the two datasets. Based on these insights, we design a principled curation recipe that produces a new data mixture, TuluTalk, which contains 14% fewer samples than either source dataset while matching or exceeding their performance on key benchmarks. Our findings offer actionable insights for constructing more effective post-training datasets that improve model performance within practical resource limits. To support future research, we publicly release both the annotated source datasets and our curated TuluTalk mixture. 

**Abstract (ZH)**: Recent Work on Large Language Models (LLMs) has Increasingly Focused on Post-Training and Alignment with Datasets Curated to Enhance Instruction Following, World Knowledge, and Specialized Skills 

---
# Private GPTs for LLM-driven testing in software development and machine learning 

**Title (ZH)**: 基于私有GPTs的LLM驱动软件开发与机器学习测试 

**Authors**: Jakub Jagielski, Markus Abel  

**Link**: [PDF](https://arxiv.org/pdf/2506.06509)  

**Abstract**: In this contribution, we examine the capability of private GPTs to automatically generate executable test code based on requirements. More specifically, we use acceptance criteria as input, formulated as part of epics, or stories, which are typically used in modern development processes. This gives product owners, or business intelligence, respectively, a way to directly produce testable criteria through the use of LLMs. We explore the quality of the so-produced tests in two ways: i) directly by letting the LLM generate code from requirements, ii) through an intermediate step using Gherkin syntax. As a result, it turns out that the two-step procedure yields better results -where we define better in terms of human readability and best coding practices, i.e. lines of code and use of additional libraries typically used in testing. Concretely, we evaluate prompt effectiveness across two scenarios: a simple "Hello World" program and a digit classification model, showing that structured prompts lead to higher-quality test outputs. 

**Abstract (ZH)**: 本研究探讨私有GPT自动根据需求生成可执行测试代码的能力。具体而言，我们使用验收标准作为输入，这些验收标准通常作为史诗或故事的一部分进行制定，这在现代开发过程中被广泛使用。这为产品所有者或商业智能提供了直接通过使用大模型生成可测试标准的方法。我们通过两种方式探索所生成测试的质量：i) 直接让大模型从需求生成代码，ii) 通过使用Gherkin语法的中间步骤。结果表明，两步流程能获得更好的结果，我们在人类可读性和最佳编码实践等方面定义更好，即代码行数和通常在测试中使用的额外库。具体而言，我们评估了提示效果在两种场景下的表现：“Hello World”程序和数字分类模型，发现结构化的提示能产生更高质量的测试输出。 

---
# Synthetic Problem Generation for Reasoning via Quality-Diversity Algorithms 

**Title (ZH)**: 基于质量多样性算法的推理合成问题生成 

**Authors**: Alex Havrilla, Edward Hughes, Mikayel Samvelyan, Jacob Abernethy  

**Link**: [PDF](https://arxiv.org/pdf/2506.06499)  

**Abstract**: Large language model (LLM) driven synthetic data generation has emerged as a powerful method for improving model reasoning capabilities. However, most methods either distill large state-of-the-art models into small students or use natural ground-truth problem statements to guarantee problem statement quality. This limits the scalability of these approaches to more complex and diverse problem domains. To address this, we present SPARQ: Synthetic Problem Generation for Reasoning via Quality-Diversity Algorithms, a novel approach for generating high-quality and diverse synthetic math problem and solution pairs using only a single model by measuring a problem's solve-rate: a proxy for problem difficulty. Starting from a seed dataset of 7.5K samples, we generate over 20 million new problem-solution pairs. We show that filtering the generated data by difficulty and then fine-tuning the same model on the resulting data improves relative model performance by up to 24\%. Additionally, we conduct ablations studying the impact of synthetic data quantity, quality and diversity on model generalization. We find that higher quality, as measured by problem difficulty, facilitates better in-distribution performance. Further, while generating diverse synthetic data does not as strongly benefit in-distribution performance, filtering for more diverse data facilitates more robust OOD generalization. We also confirm the existence of model and data scaling laws for synthetically generated problems, which positively benefit downstream model generalization. 

**Abstract (ZH)**: 基于高质量多样算法的合成问题生成以提升逻辑推理能力：SPARQ方法 

---
# What Is Seen Cannot Be Unseen: The Disruptive Effect of Knowledge Conflict on Large Language Models 

**Title (ZH)**: 看得见的不能被遗忘：知识冲突对大型语言模型的颠覆性影响 

**Authors**: Kaiser Sun, Fan Bai, Mark Dredze  

**Link**: [PDF](https://arxiv.org/pdf/2506.06485)  

**Abstract**: Large language models frequently rely on both contextual input and parametric knowledge to perform tasks. However, these sources can come into conflict, especially when retrieved documents contradict the model's parametric knowledge. We propose a diagnostic framework to systematically evaluate LLM behavior under context-memory conflict, where the contextual information diverges from their parametric beliefs. We construct diagnostic data that elicit these conflicts and analyze model performance across multiple task types. Our findings reveal that (1) knowledge conflict has minimal impact on tasks that do not require knowledge utilization, (2) model performance is consistently higher when contextual and parametric knowledge are aligned, (3) models are unable to fully suppress their internal knowledge even when instructed, and (4) providing rationales that explain the conflict increases reliance on contexts. These insights raise concerns about the validity of model-based evaluation and underscore the need to account for knowledge conflict in the deployment of LLMs. 

**Abstract (ZH)**: 大型语言模型经常依赖上下文输入和参数化知识来完成任务。然而，这些来源可能会产生冲突，尤其是在检索到的文档与模型的参数化知识相矛盾时。我们提出了一种诊断框架，以系统性地评估在上下文-记忆冲突状态下LLM的行为，其中上下文信息与参数信念相背离。我们构建了诊断数据以引发这些冲突，并分析了模型在多种任务类型中的性能。我们的研究表明：(1) 知识冲突对无需利用知识的任务影响甚微；(2) 当上下文和参数化知识一致时，模型的表现始终更高；(3) 即使在指令下，模型也无法完全抑制其内部知识；(4) 提供解释冲突的合理性陈述会增加对上下文的依赖。这些发现引发了对基于模型评估的有效性的担忧，并强调了在部署LLM时需要考虑知识冲突的重要性。 

---
# The Economic Dispatch of Power-to-Gas Systems with Deep Reinforcement Learning:Tackling the Challenge of Delayed Rewards with Long-Term Energy Storage 

**Title (ZH)**: 基于深度强化学习的气体储能系统电力调度：长周期能源存储下的延迟奖励挑战解决方法 

**Authors**: Manuel Sage, Khalil Al Handawi, Yaoyao Fiona Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.06484)  

**Abstract**: Power-to-Gas (P2G) technologies gain recognition for enabling the integration of intermittent renewables, such as wind and solar, into electricity grids. However, determining the most cost-effective operation of these systems is complex due to the volatile nature of renewable energy, electricity prices, and loads. Additionally, P2G systems are less efficient in converting and storing energy compared to battery energy storage systems (BESs), and the benefits of converting electricity into gas are not immediately apparent. Deep Reinforcement Learning (DRL) has shown promise in managing the operation of energy systems amidst these uncertainties. Yet, DRL techniques face difficulties with the delayed reward characteristic of P2G system operation. Previous research has mostly focused on short-term studies that look at the energy conversion process, neglecting the long-term storage capabilities of P2G.
This study presents a new method by thoroughly examining how DRL can be applied to the economic operation of P2G systems, in combination with BESs and gas turbines, over extended periods. Through three progressively more complex case studies, we assess the performance of DRL algorithms, specifically Deep Q-Networks and Proximal Policy Optimization, and introduce modifications to enhance their effectiveness. These modifications include integrating forecasts, implementing penalties on the reward function, and applying strategic cost calculations, all aimed at addressing the issue of delayed rewards. Our findings indicate that while DRL initially struggles with the complex decision-making required for P2G system operation, the adjustments we propose significantly improve its capability to devise cost-effective operation strategies, thereby unlocking the potential for long-term energy storage in P2G technologies. 

**Abstract (ZH)**: 基于深度强化学习的Power-to-Gas系统经济运行方法研究：结合电池储能系统和燃气涡轮机的长期优化 

---
# Noise Consistency Regularization for Improved Subject-Driven Image Synthesis 

**Title (ZH)**: 基于噪声一致性正则化以提高主体驱动的图像合成 

**Authors**: Yao Ni, Song Wen, Piotr Koniusz, Anoop Cherian  

**Link**: [PDF](https://arxiv.org/pdf/2506.06483)  

**Abstract**: Fine-tuning Stable Diffusion enables subject-driven image synthesis by adapting the model to generate images containing specific subjects. However, existing fine-tuning methods suffer from two key issues: underfitting, where the model fails to reliably capture subject identity, and overfitting, where it memorizes the subject image and reduces background diversity. To address these challenges, we propose two auxiliary consistency losses for diffusion fine-tuning. First, a prior consistency regularization loss ensures that the predicted diffusion noise for prior (non-subject) images remains consistent with that of the pretrained model, improving fidelity. Second, a subject consistency regularization loss enhances the fine-tuned model's robustness to multiplicative noise modulated latent code, helping to preserve subject identity while improving diversity. Our experimental results demonstrate that incorporating these losses into fine-tuning not only preserves subject identity but also enhances image diversity, outperforming DreamBooth in terms of CLIP scores, background variation, and overall visual quality. 

**Abstract (ZH)**: Fine-tuning Stable Diffusion 通过适应模型生成包含特定主题的图像，实现主题驱动的图像合成。然而，现有的细调方法面临两个关键问题：欠拟合和过拟合。针对这些挑战，我们提出了两种辅助一致性损失以优化扩散模型的细调。首先，先验一致性正则化损失确保非主题图像的预测扩散噪声与预训练模型的噪声保持一致，从而提高保真度。其次，主题一致性正则化损失增强细调模型对动态噪声调制潜在代码的鲁棒性，有助于保持主题身份的同时提高多样性。实验结果表明，在细调过程中引入这些损失不仅保留了主题身份，还提升了图像多样性，在CLIP得分、背景变化和总体视觉质量方面优于DreamBooth。 

---
# Edge-Enabled Collaborative Object Detection for Real-Time Multi-Vehicle Perception 

**Title (ZH)**: 边缘赋能的协同目标检测用于实时多车辆感知 

**Authors**: Everett Richards, Bipul Thapa, Lena Mashayekhy  

**Link**: [PDF](https://arxiv.org/pdf/2506.06474)  

**Abstract**: Accurate and reliable object detection is critical for ensuring the safety and efficiency of Connected Autonomous Vehicles (CAVs). Traditional on-board perception systems have limited accuracy due to occlusions and blind spots, while cloud-based solutions introduce significant latency, making them unsuitable for real-time processing demands required for autonomous driving in dynamic environments. To address these challenges, we introduce an innovative framework, Edge-Enabled Collaborative Object Detection (ECOD) for CAVs, that leverages edge computing and multi-CAV collaboration for real-time, multi-perspective object detection. Our ECOD framework integrates two key algorithms: Perceptive Aggregation and Collaborative Estimation (PACE) and Variable Object Tally and Evaluation (VOTE). PACE aggregates detection data from multiple CAVs on an edge server to enhance perception in scenarios where individual CAVs have limited visibility. VOTE utilizes a consensus-based voting mechanism to improve the accuracy of object classification by integrating data from multiple CAVs. Both algorithms are designed at the edge to operate in real-time, ensuring low-latency and reliable decision-making for CAVs. We develop a hardware-based controlled testbed consisting of camera-equipped robotic CAVs and an edge server to evaluate the efficacy of our framework. Our experimental results demonstrate the significant benefits of ECOD in terms of improved object classification accuracy, outperforming traditional single-perspective onboard approaches by up to 75%, while ensuring low-latency, edge-driven real-time processing. This research highlights the potential of edge computing to enhance collaborative perception for latency-sensitive autonomous systems. 

**Abstract (ZH)**: 基于边缘计算的协作对象检测框架（ECOD）：提升连接自动驾驶车辆的安全性和效率 

---
# Cost-Efficient LLM Training with Lifetime-Aware Tensor Offloading via GPUDirect Storage 

**Title (ZH)**: 基于GPUDirect存储的生命周期意识张量卸载的高效LLM训练 

**Authors**: Ziqi Yuan, Haoyang Zhang, Yirui Eric Zhou, Apoorve Mohan, I-Hsin Chung, Seetharami Seelam, Jian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06472)  

**Abstract**: We present the design and implementation of a new lifetime-aware tensor offloading framework for GPU memory expansion using low-cost PCIe-based solid-state drives (SSDs). Our framework, TERAIO, is developed explicitly for large language model (LLM) training with multiple GPUs and multiple SSDs. Its design is driven by our observation that the active tensors take only a small fraction (1.7% on average) of allocated GPU memory in each LLM training iteration, the inactive tensors are usually large and will not be used for a long period of time, creating ample opportunities for offloading/prefetching tensors to/from slow SSDs without stalling the GPU training process. TERAIO accurately estimates the lifetime (active period of time in GPU memory) of each tensor with the profiling of the first few iterations in the training process. With the tensor lifetime analysis, TERAIO will generate an optimized tensor offloading/prefetching plan and integrate it into the compiled LLM program via PyTorch. TERAIO has a runtime tensor migration engine to execute the offloading/prefetching plan via GPUDirect storage, which allows direct tensor migration between GPUs and SSDs for alleviating the CPU bottleneck and maximizing the SSD bandwidth utilization. In comparison with state-of-the-art studies such as ZeRO-Offload and ZeRO-Infinity, we show that TERAIO improves the training performance of various LLMs by 1.47x on average, and achieves 80.7% of the ideal performance assuming unlimited GPU memory. 

**Abstract (ZH)**: 我们提出了一种新的基于低成本PCIe固态驱动器（SSD）的张量卸载框架，以实现GPU内存扩展，并关注张量的生命周期。该框架TERAIO专为多GPU和多SSD的大型语言模型（LLM）训练设计。其设计基于观察到每个LLM训练迭代中活跃张量仅占分配GPU内存的小部分（平均值为1.7%），不活跃张量通常较大且会闲置较长时间，这为将张量卸载/预取到/从缓慢的SSD上提供了充足的机会，而不影响GPU训练过程。TERAIO通过训练过程前几轮的分析，精确估计每个张量在GPU内存中的生命周期。基于张量生命周期分析，TERAIO生成一个优化的张量卸载/预取计划，并通过PyTorch集成到编译的LLM程序中。TERAIO具有运行时张量迁移引擎，通过GPUDirect存储执行卸载/预取计划，从而在GPU和SSD之间直接迁移张量，缓解CPU瓶颈并最大化SSD带宽利用率。与ZeRO-Offload和ZeRO-Infinity等先进研究相比，TERAIO将各种LLM的训练性能平均提高了1.47倍，并实现了理论上无限GPU内存性能的80.7%。 

---
# WISCA: A Consensus-Based Approach to Harmonizing Interpretability in Tabular Datasets 

**Title (ZH)**: WISCA：一种基于共识的方法以谐调表数据集的可解释性 

**Authors**: Antonio Jesús Banegas-Luna, Horacio Pérez-Sánchez, Carlos Martínez-Cortés  

**Link**: [PDF](https://arxiv.org/pdf/2506.06455)  

**Abstract**: While predictive accuracy is often prioritized in machine learning (ML) models, interpretability remains essential in scientific and high-stakes domains. However, diverse interpretability algorithms frequently yield conflicting explanations, highlighting the need for consensus to harmonize results. In this study, six ML models were trained on six synthetic datasets with known ground truths, utilizing various model-agnostic interpretability techniques. Consensus explanations were generated using established methods and a novel approach: WISCA (Weighted Scaled Consensus Attributions), which integrates class probability and normalized attributions. WISCA consistently aligned with the most reliable individual method, underscoring the value of robust consensus strategies in improving explanation reliability. 

**Abstract (ZH)**: 尽管在机器学习模型中预测准确性常常被优先考虑，但在科学和高风险领域中，解释性仍然至关重要。然而，多种解释性算法经常产生相互矛盾的解释，突显了达成共识以协调结果的必要性。在这项研究中，研究人员使用各种模型无关的解释性技术，在六个含有已知真实值的合成数据集上训练了六种机器学习模型，并通过既定方法和一种新方法——加权规范化共识归因（WISCA，Weighted Scaled Consensus Attributions）生成了共识解释。WISCA 一直与最可靠的个体方法一致，强调了采用稳健的共识策略以提高解释可靠性的重要性。 

---
# Canonical Autoregressive Generation 

**Title (ZH)**: 经典自回归生成 

**Authors**: Ivi Chatzi, Nina Corvelo Benz, Stratis Tsirtsis, Manuel Gomez-Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2506.06446)  

**Abstract**: State of the art large language models are trained using large amounts of tokens derived from raw text using what is called a tokenizer. Crucially, the tokenizer determines the (token) vocabulary a model will use during inference as well as, in principle, the (token) language. This is because, while the token vocabulary may allow for different tokenizations of a string, the tokenizer always maps the string to only one of these tokenizations--the canonical tokenization. However, multiple lines of empirical evidence suggest that large language models do not always generate canonical token sequences, and this comes with several negative consequences. In this work, we first show that, to generate a canonical token sequence, a model needs to generate (partial) canonical token sequences at each step of the autoregressive generation process underpinning its functioning. Building upon this theoretical result, we introduce canonical sampling, a simple and efficient sampling method that precludes a given model from generating non-canonical token sequences. Further, we also show that, in comparison with standard sampling, the distribution of token sequences generated using canonical sampling is provably closer to the true distribution of token sequences used during training. 

**Abstract (ZH)**: 最先进的大语言模型通过令牌化工具对大量原始文本 token 进行训练。至关重要的是，令牌化工具决定了模型在推理过程中使用的（令牌）词汇表以及原则上使用的（令牌）语言。因为虽然词汇表可以对字符串进行不同的分词，但令牌化工具总会将字符串映射到这些分词方式中的唯一一种——标准分词方式。然而，大量实证证据表明，大语言模型不总是生成标准的令牌序列，这带来了若干负面后果。在本文中，我们首先证明，为了生成标准的令牌序列，模型在自回归生成过程的每一步都需要生成部分标准的令牌序列。基于这一理论结果，我们提出了一种简单且高效的采样方法——标准采样，这种方法可以从给定模型中剔除生成非标准令牌序列的可能性。此外，我们还证明，与标准采样方法相比，使用标准采样方法生成的令牌序列的分布可以被证明更接近于训练过程中使用的令牌序列的真实分布。 

---
# Saffron-1: Towards an Inference Scaling Paradigm for LLM Safety Assurance 

**Title (ZH)**: 藏红花-1：通往大规模语言模型安全保证推理扩展范式之路 

**Authors**: Ruizhong Qiu, Gaotang Li, Tianxin Wei, Jingrui He, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2506.06444)  

**Abstract**: Existing safety assurance research has primarily focused on training-phase alignment to instill safe behaviors into LLMs. However, recent studies have exposed these methods' susceptibility to diverse jailbreak attacks. Concurrently, inference scaling has significantly advanced LLM reasoning capabilities but remains unexplored in the context of safety assurance. Addressing this gap, our work pioneers inference scaling for robust and effective LLM safety against emerging threats. We reveal that conventional inference scaling techniques, despite their success in reasoning tasks, perform poorly in safety contexts, even falling short of basic approaches like Best-of-N Sampling. We attribute this inefficiency to a newly identified challenge, the exploration--efficiency dilemma, arising from the high computational overhead associated with frequent process reward model (PRM) evaluations. To overcome this dilemma, we propose SAFFRON, a novel inference scaling paradigm tailored explicitly for safety assurance. Central to our approach is the introduction of a multifurcation reward model (MRM) that significantly reduces the required number of reward model evaluations. To operationalize this paradigm, we further propose: (i) a partial supervision training objective for MRM, (ii) a conservative exploration constraint to prevent out-of-distribution explorations, and (iii) a Trie-based key--value caching strategy that facilitates cache sharing across sequences during tree search. Extensive experiments validate the effectiveness of our method. Additionally, we publicly release our trained multifurcation reward model (Saffron-1) and the accompanying token-level safety reward dataset (Safety4M) to accelerate future research in LLM safety. Our code, model, and data are publicly available at this https URL , and our project homepage is at this https URL . 

**Abstract (ZH)**: 现有的安全保证研究主要侧重于训练阶段的对齐以灌输安全行为到大规模语言模型中。然而，最近的研究揭示了这些方法对多样化的模型突破攻击的脆弱性。同时，推理扩展显著提升了大规模语言模型的推理能力，但在安全保证的背景下尚待探索。为填补这一空白，我们的工作开创了针对新兴威胁的大规模语言模型安全的推理扩展方法，以实现稳健有效的安全防护。我们揭示，尽管传统推理扩展技术在推理任务中取得了成功，但在安全上下文中表现不佳，甚至不及Best-of-N采样等基本方法。我们归因于这一低效率是因为频繁的过程奖励模型评估带来的高计算开销引发的探索—效率困境。为克服这一困境，我们提出了SAFFRON，一种专门针对安全保证的创新性推理扩展范式。在我们的方法中，引入了一种多分叉奖励模型（MRM），显著减少了所需的过程奖励模型评估次数。为了实现这一范式的落实，我们进一步提出了：（i）对MRM的一种部分监督训练目标，（ii）保守探索约束以防止分布外探索，以及（iii）基于Trie的键—值缓存策略，这在树搜索过程中促进了序列间的缓存共享。广泛实验证明了我们方法的有效性。此外，我们公开发布了我们训练的多分叉奖励模型（Saffron-1）和配套的标记级安全奖励数据集（Safety4M），以加速大规模语言模型安全领域的未来研究。我们的代码、模型和数据可在以下链接获得：this https URL，项目主页位于以下链接：this https URL。 

---
# Unlocking Chemical Insights: Superior Molecular Representations from Intermediate Encoder Layers 

**Title (ZH)**: 解锁化学洞见：中间编码层的优质分子表示 

**Authors**: Luis Pinto  

**Link**: [PDF](https://arxiv.org/pdf/2506.06443)  

**Abstract**: Pretrained molecular encoders have become indispensable in computational chemistry for tasks such as property prediction and molecular generation. However, the standard practice of relying solely on final-layer embeddings for downstream tasks may discard valuable information. In this work, we challenge this convention by conducting a comprehensive layer-wise analysis of five diverse molecular encoders across 22 ADMET property prediction tasks. Our results demonstrate that embeddings from intermediate layers consistently outperform final-layer representations. Specifically, using fixed embeddings from the optimal intermediate layers improved downstream performance by an average of 5.4%, reaching gains up to 28.6%. Furthermore, finetuning up to these intermediate layers yielded even greater average improvements of 8.5%, with performance increases as high as 40.8%, achieving new state-of-the-art results on several benchmarks. Additionally, a strong positive correlation between fixed embedding performance and finetuning outcomes supports an efficient evaluate-then-finetune approach, enabling identification of optimal layers with reduced computational cost. These findings highlight the importance of exploring the full representational depth of molecular encoders to achieve substantial performance improvements and computational efficiency. The code is made publicly available at this https URL. 

**Abstract (ZH)**: 预训练分子编码器在计算化学中的应用对于诸如性质预测和分子生成等任务已经变得不可或缺。然而，仅依赖最终层嵌入进行下游任务的标准做法可能会丢弃有价值的信息。在本工作中，我们通过在22项ADMET性质预测任务中对五种不同的分子编码器进行全面的逐层分析，挑战了这一惯例。结果显示，中间层嵌入始终优于最终层表示。具体来说，使用来自最优中间层的固定嵌入，下游性能平均提高了5.4%，最高可达28.6%。此外，微调至这些中间层带来了更显著的平均改善，平均提高了8.5%，最高可达40.8%，在多个基准上达到了新的最佳结果。另外，固定嵌入性能与微调结果之间的强烈正相关支持了一种高效的评估-然后微调方法，可以降低计算成本以识别最优层。这些发现突出了探索分子编码器的全部表示深度以实现显著性能提升和计算效率的重要性。代码已在此处公开：这个链接。 

---
# Benchmarking Misuse Mitigation Against Covert Adversaries 

**Title (ZH)**: 基于隐蔽对手的滥用缓解基准测试 

**Authors**: Davis Brown, Mahdi Sabbaghi, Luze Sun, Alexander Robey, George J. Pappas, Eric Wong, Hamed Hassani  

**Link**: [PDF](https://arxiv.org/pdf/2506.06414)  

**Abstract**: Existing language model safety evaluations focus on overt attacks and low-stakes tasks. Realistic attackers can subvert current safeguards by requesting help on small, benign-seeming tasks across many independent queries. Because individual queries do not appear harmful, the attack is hard to {detect}. However, when combined, these fragments uplift misuse by helping the attacker complete hard and dangerous tasks. Toward identifying defenses against such strategies, we develop Benchmarks for Stateful Defenses (BSD), a data generation pipeline that automates evaluations of covert attacks and corresponding defenses. Using this pipeline, we curate two new datasets that are consistently refused by frontier models and are too difficult for weaker open-weight models. Our evaluations indicate that decomposition attacks are effective misuse enablers, and highlight stateful defenses as a countermeasure. 

**Abstract (ZH)**: 现有的语言模型安全评估主要关注公开攻击和低风险任务。现实中的攻击者可以通过在多个独立查询中请求看似无害的小任务来规避当前的安全措施。由于个别查询看起来并不危险，这种攻击难以被检测。然而，当这些片段结合在一起时，它们能够辅助攻击者完成艰难且危险的任务。为了识别针对此类策略的防御措施，我们开发了状态依赖防御基准（BSD），这是一个自动评估隐蔽攻击及其相应防御的数据生成管道。利用这一管道，我们精心筛选了两个新的数据集，这些数据集超出了前沿模型的拒绝范围，并对较弱的开放权重模型来说也过于困难。我们的评估表明，分解攻击是有效的滥用辅助手段，并突显了状态依赖防御作为一种应对措施的重要性。 

---
# HeavyWater and SimplexWater: Watermarking Low-Entropy Text Distributions 

**Title (ZH)**: 重水与单纯形水：标记低熵文本分布 

**Authors**: Dor Tsur, Carol Xuan Long, Claudio Mayrink Verdun, Hsiang Hsu, Chen-Fu Chen, Haim Permuter, Sajani Vithana, Flavio P. Calmon  

**Link**: [PDF](https://arxiv.org/pdf/2506.06409)  

**Abstract**: Large language model (LLM) watermarks enable authentication of text provenance, curb misuse of machine-generated text, and promote trust in AI systems. Current watermarks operate by changing the next-token predictions output by an LLM. The updated (i.e., watermarked) predictions depend on random side information produced, for example, by hashing previously generated tokens. LLM watermarking is particularly challenging in low-entropy generation tasks - such as coding - where next-token predictions are near-deterministic. In this paper, we propose an optimization framework for watermark design. Our goal is to understand how to most effectively use random side information in order to maximize the likelihood of watermark detection and minimize the distortion of generated text. Our analysis informs the design of two new watermarks: HeavyWater and SimplexWater. Both watermarks are tunable, gracefully trading-off between detection accuracy and text distortion. They can also be applied to any LLM and are agnostic to side information generation. We examine the performance of HeavyWater and SimplexWater through several benchmarks, demonstrating that they can achieve high watermark detection accuracy with minimal compromise of text generation quality, particularly in the low-entropy regime. Our theoretical analysis also reveals surprising new connections between LLM watermarking and coding theory. The code implementation can be found in this https URL 

**Abstract (ZH)**: 大规模语言模型（LLM）水印使文本来源认证成为可能，遏制机器生成文本的滥用，并促进对AI系统的信任。当前的水印通过改变LLM输出的下一个 token 预测来工作。更新后的（即，带有水印的）预测依赖于通过哈希先前生成的 token 等方式产生的随机辅助信息。在低熵生成任务（如编码）中，下一个 token 的预测几乎是确定性的，使得 LLM 水印尤其具有挑战性。在本文中，我们提出了一种水印设计的优化框架。我们的目标是通过理解如何最有效地使用随机辅助信息来最大化水印检测的概率并最小化生成文本的失真。我们的分析指导设计了两种新的水印：HeavyWater 和 SimplexWater。这两种水印都是可调节的，能够在检测准确性和文本失真之间优雅地权衡。它们可以应用于任何 LLM，并且对辅助信息生成方式无偏好。我们通过多种基准测试研究了 HeavyWater 和 SimplexWater 的性能，证明了它们可以在最小牺牲生成文本质量的情况下实现高水印检测准确性，尤其是在低熵环境中。我们对水印技术的理论分析还揭示了与编码理论之间一些令人惊讶的新联系。代码实现可以在以下链接找到：https://xxxxxx 

---
# TimeWak: Temporal Chained-Hashing Watermark for Time Series Data 

**Title (ZH)**: TimeWak: 时间链式哈希水印用于时间序列数据 

**Authors**: Zhi Wen Soi, Chaoyi Zhu, Fouad Abiad, Aditya Shankar, Jeroen M. Galjaard, Huijuan Wang, Lydia Y. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.06407)  

**Abstract**: Synthetic time series generated by diffusion models enable sharing privacy-sensitive datasets, such as patients' functional MRI records. Key criteria for synthetic data include high data utility and traceability to verify the data source. Recent watermarking methods embed in homogeneous latent spaces, but state-of-the-art time series generators operate in real space, making latent-based watermarking incompatible. This creates the challenge of watermarking directly in real space while handling feature heterogeneity and temporal dependencies. We propose TimeWak, the first watermarking algorithm for multivariate time series diffusion models. To handle temporal dependence and spatial heterogeneity, TimeWak embeds a temporal chained-hashing watermark directly within the real temporal-feature space. The other unique feature is the $\epsilon$-exact inversion, which addresses the non-uniform reconstruction error distribution across features from inverting the diffusion process to detect watermarks. We derive the error bound of inverting multivariate time series and further maintain high watermark detectability. We extensively evaluate TimeWak on its impact on synthetic data quality, watermark detectability, and robustness under various post-editing attacks, against 5 datasets and baselines of different temporal lengths. Our results show that TimeWak achieves improvements of 61.96% in context-FID score, and 8.44% in correlational scores against the state-of-the-art baseline, while remaining consistently detectable. 

**Abstract (ZH)**: 基于扩散模型的合成时间序列 enables 分享敏感隐私数据集，如患者的功能MRI记录。TimeWak: 一种用于多元时间序列扩散模型的水印算法 

---
# SMAR: Soft Modality-Aware Routing Strategy for MoE-based Multimodal Large Language Models Preserving Language Capabilities 

**Title (ZH)**: SMAR：软模态感知路由策略，用于保留语言能力的MoE-based多模态大型语言模型 

**Authors**: Guoyang Xia, Yifeng Ding, Fengfa Li, Lei Ren, Chen Wei, Fangxiang Feng, Xiaojie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06406)  

**Abstract**: Mixture of Experts (MoE) architectures have become a key approach for scaling large language models, with growing interest in extending them to multimodal tasks. Existing methods to build multimodal MoE models either incur high training costs or suffer from degraded language capabilities when adapting pretrained models. To address this, we propose Soft ModalityAware Routing (SMAR), a novel regularization technique that uses Kullback Leibler divergence to control routing probability distributions across modalities, encouraging expert specialization without modifying model architecture or heavily relying on textual data. Experiments on visual instruction tuning show that SMAR preserves language ability at 86.6% retention with only 2.5% pure text, outperforming baselines while maintaining strong multimodal performance. Our approach offers a practical and efficient solution to balance modality differentiation and language capabilities in multimodal MoE models. 

**Abstract (ZH)**: 混合专家模型的软模态感知路由（SMAR）：一种促进多模态大型语言模型模态分化和语言能力平衡的新正则化技术 

---
# Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights 

**Title (ZH)**: 价值对齐的大语言模型的意外危害：心理与实证洞察 

**Authors**: Sooyung Choi, Jaehyeok Lee, Xiaoyuan Yi, Jing Yao, Xing Xie, JinYeong Bak  

**Link**: [PDF](https://arxiv.org/pdf/2506.06404)  

**Abstract**: The application scope of Large Language Models (LLMs) continues to expand, leading to increasing interest in personalized LLMs that align with human values. However, aligning these models with individual values raises significant safety concerns, as certain values may correlate with harmful information. In this paper, we identify specific safety risks associated with value-aligned LLMs and investigate the psychological principles behind these challenges. Our findings reveal two key insights. (1) Value-aligned LLMs are more prone to harmful behavior compared to non-fine-tuned models and exhibit slightly higher risks in traditional safety evaluations than other fine-tuned models. (2) These safety issues arise because value-aligned LLMs genuinely generate text according to the aligned values, which can amplify harmful outcomes. Using a dataset with detailed safety categories, we find significant correlations between value alignment and safety risks, supported by psychological hypotheses. This study offers insights into the "black box" of value alignment and proposes in-context alignment methods to enhance the safety of value-aligned LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）的安全风险及个性化价值对齐原则研究 

---
# Direct Behavior Optimization: Unlocking the Potential of Lightweight LLMs 

**Title (ZH)**: 直接行为优化：轻量级LLM潜力的释放 

**Authors**: Hongming Yang, Shi Lin, Jun Shao, Changting Lin, Donghai Zhu, Meng Han, Qinglei Kong  

**Link**: [PDF](https://arxiv.org/pdf/2506.06401)  

**Abstract**: Lightweight Large Language Models (LwLLMs) are reduced-parameter, optimized models designed to run efficiently on consumer-grade hardware, offering significant advantages in resource efficiency, cost-effectiveness, and data privacy. However, these models often struggle with limited inference and reasoning capabilities, which restrict their performance on complex tasks and limit their practical applicability. Moreover, existing prompt optimization methods typically rely on extensive manual effort or the meta-cognitive abilities of state-of-the-art LLMs, making them less effective for LwLLMs. To address these challenges, we introduce DeBoP, a new Direct Behavior Optimization Paradigm, original from the Chain-of-Thought (CoT) prompting technique. Unlike CoT Prompting, DeBoP is an automatic optimization method, which focuses on the optimization directly on the behavior of LwLLMs. In particular, DeBoP transforms the optimization of complex prompts into the optimization of discrete, quantifiable execution sequences using a gradient-free Monte Carlo Tree Search. We evaluate DeBoP on seven challenging tasks where state-of-the-art LLMs excel but LwLLMs generally underperform. Experimental results demonstrate that DeBoP significantly outperforms recent prompt optimization methods on most tasks. In particular, DeBoP-optimized LwLLMs surpass GPT-3.5 on most tasks while reducing computational time by approximately 60% compared to other automatic prompt optimization methods. 

**Abstract (ZH)**: 轻量级大型语言模型（LwLLMs）是参数减少、优化后的模型，旨在高效运行在消费级硬件上，提供了在资源效率、成本效益和数据隐私方面的显著优势。然而，这些模型往往在推理和推理能力上存在局限，这限制了其在复杂任务上的表现，并限制了其实用性。此外，现有的提示优化方法通常依赖大量的手动努力或最先进的LwLLMs的元认知能力，这对LwLLMs来说效果较差。为了解决这些问题，我们提出了DeBoP，一种新的直接行为优化范式，源自Chain-of-Thought（CoT）提示技术。与CoT提示不同，DeBoP是一种自动优化方法，专注于直接优化LwLLMs的行为。特别是，DeBoP将复杂提示的优化转换为使用无导数蒙特卡洛树搜索优化离散的、可量化执行序列。我们在七个挑战性任务上评估了DeBoP，这些任务是目前最先进的LLM可以表现出色但LwLLMs通常表现较差的。实验结果表明，DeBoP在大多数任务上显著优于最近的提示优化方法。特别是，DeBoP优化的LwLLMs在大多数任务上超越了GPT-3.5，与其它自动提示优化方法相比，计算时间减少了约60%。 

---
# Theoretical Analysis of Positional Encodings in Transformer Models: Impact on Expressiveness and Generalization 

**Title (ZH)**: Transformer模型中位置编码的理论分析：对其表达能力和泛化能力的影响 

**Authors**: Yin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06398)  

**Abstract**: Positional encodings are a core part of transformer-based models, enabling processing of sequential data without recurrence. This paper presents a theoretical framework to analyze how various positional encoding methods, including sinusoidal, learned, relative, and bias-based methods like Attention with Linear Biases (ALiBi), impact a transformer's expressiveness, generalization ability, and extrapolation to longer sequences. Expressiveness is defined via function approximation, generalization bounds are established using Rademacher complexity, and new encoding methods based on orthogonal functions, such as wavelets and Legendre polynomials, are proposed. The extrapolation capacity of existing and proposed encodings is analyzed, extending ALiBi's biasing approach to a unified theoretical context. Experimental evaluation on synthetic sequence-to-sequence tasks shows that orthogonal transform-based encodings outperform traditional sinusoidal encodings in generalization and extrapolation. This work addresses a critical gap in transformer theory, providing insights for design choices in natural language processing, computer vision, and other transformer applications. 

**Abstract (ZH)**: 基于位置的编码是变压器模型的核心组成部分，使模型能够处理序列数据而不依赖循环结构。本文提出了一个理论框架来分析各种位置编码方法，包括正弦、学习、相对和基于偏置的方法（如线性偏置注意力（ALiBi）），对变压器的表征能力、泛化能力和长序列外推能力的影响。表征能力通过函数逼近定义，泛化界限使用拉德马赫复杂性建立，并提出了基于正交函数的新编码方法，如小波和勒让德多项式。现有和提出编码方法的外推能力得到了分析，将ALiBi的偏置方法统一到一个理论框架中。基于合成序列到序列任务的实验评估表明，基于正交变换的位置编码在泛化和外推方面优于传统的正弦编码。本文填补了变压器理论中的关键空白，为自然语言处理、计算机视觉和其他变压器应用的设计选择提供了见解。 

---
# Natural Language Interaction with Databases on Edge Devices in the Internet of Battlefield Things 

**Title (ZH)**: 战场物联网边缘设备上的自然语言数据库交互 

**Authors**: Christopher D. Molek, Roberto Fronteddu, K. Brent Venable, Niranjan Suri  

**Link**: [PDF](https://arxiv.org/pdf/2506.06396)  

**Abstract**: The expansion of the Internet of Things (IoT) in the battlefield, Internet of Battlefield Things (IoBT), gives rise to new opportunities for enhancing situational awareness. To increase the potential of IoBT for situational awareness in critical decision making, the data from these devices must be processed into consumer-ready information objects, and made available to consumers on demand. To address this challenge we propose a workflow that makes use of natural language processing (NLP) to query a database technology and return a response in natural language. Our solution utilizes Large Language Models (LLMs) that are sized for edge devices to perform NLP as well as graphical databases which are well suited for dynamic connected networks which are pervasive in the IoBT. Our architecture employs LLMs for both mapping questions in natural language to Cypher database queries as well as to summarize the database output back to the user in natural language. We evaluate several medium sized LLMs for both of these tasks on a database representing publicly available data from the US Army's Multipurpose Sensing Area (MSA) at the Jornada Range in Las Cruces, NM. We observe that Llama 3.1 (8 billion parameters) outperforms the other models across all the considered metrics. Most importantly, we note that, unlike current methods, our two step approach allows the relaxation of the Exact Match (EM) requirement of the produced Cypher queries with ground truth code and, in this way, it achieves a 19.4% increase in accuracy. Our workflow lays the ground work for deploying LLMs on edge devices to enable natural language interactions with databases containing information objects for critical decision making. 

**Abstract (ZH)**: 战场物联网（IoBT）的扩展为增强态势感知提供了新机遇。为了提高IoBT在关键决策中的潜力，这些设备的数据必须被处理成消费者可用的信息对象，并按需提供给消费者。为了解决这一挑战，我们提出了一种工作流，利用自然语言处理（NLP）查询数据库技术并以自然语言返回响应。我们的解决方案利用了适用于边缘设备的大型语言模型（LLMs）来进行NLP，以及适用于动态连接网络的图形数据库，这些网络在IoBT中普遍存在。我们的架构使用LLMs将自然语言中的问题映射为Cypher数据库查询，并将数据库输出总结回自然语言给用户。我们在代表美国陆军多用途传感区域（MSA）公开数据的数据库上评估了几种中型LLMs，这些数据来自新墨西哥州拉斯克鲁塞斯市的Jornada训练场。我们发现，Llama 3.1（80亿参数）在所有考虑的指标中表现最佳。最重要的是，我们注意到，与当前方法不同，我们两步方法允许放宽生成的Cypher查询与地面真实代码的精确匹配要求，从而在准确性上提高了19.4%。我们的工作流为在边缘设备上部署LLMs以实现与包含信息对象的数据库的自然语言交互奠定了基础。 

---
# From Rogue to Safe AI: The Role of Explicit Refusals in Aligning LLMs with International Humanitarian Law 

**Title (ZH)**: 从违规到安全的人工 intelligence：明示拒绝在将大型语言模型与国际人道法对齐中的作用 

**Authors**: John Mavi, Diana Teodora Găitan, Sergio Coronado  

**Link**: [PDF](https://arxiv.org/pdf/2506.06391)  

**Abstract**: Large Language Models (LLMs) are widely used across sectors, yet their alignment with International Humanitarian Law (IHL) is not well understood. This study evaluates eight leading LLMs on their ability to refuse prompts that explicitly violate these legal frameworks, focusing also on helpfulness - how clearly and constructively refusals are communicated. While most models rejected unlawful requests, the clarity and consistency of their responses varied. By revealing the model's rationale and referencing relevant legal or safety principles, explanatory refusals clarify the system's boundaries, reduce ambiguity, and help prevent misuse. A standardised system-level safety prompt significantly improved the quality of the explanations expressed within refusals in most models, highlighting the effectiveness of lightweight interventions. However, more complex prompts involving technical language or requests for code revealed ongoing vulnerabilities. These findings contribute to the development of safer, more transparent AI systems and propose a benchmark to evaluate the compliance of LLM with IHL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各行业的广泛应用尚存，但其与国际人道法（IHL）的契合程度尚不明确。本研究评价了八种主流LLM在其拒绝明确违反这些法律框架的提示能力上的表现，并重点考察了这些模型的建议性——即拒绝理由的清晰度和建设性。虽然大多数模型拒绝了非法请求，但其回应的清晰度和一致性存在差异。通过揭示模型的推理过程并参考相关法律或安全原则，解释性的拒绝能够阐明系统的边界，减少模糊性，并有助于防止滥用。标准化的系统级安全提示在大多数模型中显著提高了拒绝理由表达的质量，突显了轻量级干预措施的有效性。然而，涉及技术语言或代码请求的更复杂提示揭示了持续存在的漏洞。这些发现为开发更安全和更具透明度的人工智能系统作出了贡献，并提出了评估LLM与IHL合规性的基准。 

---
# Benchmarking Large Language Models on Homework Assessment in Circuit Analysis 

**Title (ZH)**: 大型语言模型在电路分析作业评估中的基准测试 

**Authors**: Liangliang Chen, Zhihao Qin, Yiming Guo, Jacqueline Rohde, Ying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06390)  

**Abstract**: Large language models (LLMs) have the potential to revolutionize various fields, including code development, robotics, finance, and education, due to their extensive prior knowledge and rapid advancements. This paper investigates how LLMs can be leveraged in engineering education. Specifically, we benchmark the capabilities of different LLMs, including GPT-3.5 Turbo, GPT-4o, and Llama 3 70B, in assessing homework for an undergraduate-level circuit analysis course. We have developed a novel dataset consisting of official reference solutions and real student solutions to problems from various topics in circuit analysis. To overcome the limitations of image recognition in current state-of-the-art LLMs, the solutions in the dataset are converted to LaTeX format. Using this dataset, a prompt template is designed to test five metrics of student solutions: completeness, method, final answer, arithmetic error, and units. The results show that GPT-4o and Llama 3 70B perform significantly better than GPT-3.5 Turbo across all five metrics, with GPT-4o and Llama 3 70B each having distinct advantages in different evaluation aspects. Additionally, we present insights into the limitations of current LLMs in several aspects of circuit analysis. Given the paramount importance of ensuring reliability in LLM-generated homework assessment to avoid misleading students, our results establish benchmarks and offer valuable insights for the development of a reliable, personalized tutor for circuit analysis -- a focus of our future work. Furthermore, the proposed evaluation methods can be generalized to a broader range of courses for engineering education in the future. 

**Abstract (ZH)**: 大规模语言模型（LLMs）有可能革新包括代码开发、机器人学、金融和教育在内的多个领域，得益于其广泛的知识储备和快速的技术进步。本文探讨了LLMs在工程教育中的应用。具体来说，我们比较了GPT-3.5 Turbo、GPT-4o和Llama 3 70B等不同LLMs评估本科生电路分析课程作业的能力。我们开发了一个新的数据集，其中包括官方参考解决方案和实际的学生解决方案，涵盖电路分析中的各种问题。为克服当前先进LLMs在图像识别方面的局限，数据集中的解决方案被转换为LaTeX格式。利用该数据集，我们设计了一个提示模板，以测试学生解决方案的五个指标：完整性、方法、最终答案、算术错误和单位。结果表明，GPT-4o和Llama 3 70B在所有五个指标上均显著优于GPT-3.5 Turbo，GPT-4o和Llama 3 70B在不同的评估方面各有优势。此外，我们还探讨了当前LLMs在电路分析方面的一些局限性。鉴于确保LLM生成的作业评估的可靠性对于避免误导学生至关重要，我们的结果建立了基准并为开发一个可靠且个性化的电路分析辅导系统提供了有价值的见解——这是我们未来工作的一个重点。此外，所提出的方法可以进一步应用于工程教育中更广泛的课程。 

---
# Model-based Neural Data Augmentation for sub-wavelength Radio Localization 

**Title (ZH)**: 基于模型的神经数据扩充方法在亚波长无线电定位中的应用 

**Authors**: Baptiste Chatelier, Vincent Corlay, Musa Furkan Keskin, Matthieu Crussière, Henk Wymeersch, Luc Le Magoarou  

**Link**: [PDF](https://arxiv.org/pdf/2506.06387)  

**Abstract**: The increasing deployment of large antenna arrays at base stations has significantly improved the spatial resolution and localization accuracy of radio-localization methods. However, traditional signal processing techniques struggle in complex radio environments, particularly in scenarios dominated by non line of sight (NLoS) propagation paths, resulting in degraded localization accuracy. Recent developments in machine learning have facilitated the development of machine learning-assisted localization techniques, enhancing localization accuracy in complex radio environments. However, these methods often involve substantial computational complexity during both the training and inference phases. This work extends the well-established fingerprinting-based localization framework by simultaneously reducing its memory requirements and improving its accuracy. Specifically, a model-based neural network is used to learn the location-to-channel mapping, and then serves as a generative neural channel model. This generative model augments the fingerprinting comparison dictionary while reducing the memory requirements. The proposed method outperforms fingerprinting baselines by achieving sub-wavelength localization accuracy, even in NLoS environments. Remarkably, it offers an improvement by several orders of magnitude in localization accuracy, while simultaneously reducing memory requirements by an order of magnitude compared to classical fingerprinting methods. 

**Abstract (ZH)**: 基于模型的神经网络辅助指纹本地化方法在非视距环境下的亚波长精度定位 

---
# Detection Method for Prompt Injection by Integrating Pre-trained Model and Heuristic Feature Engineering 

**Title (ZH)**: 基于预训练模型和启发式特征工程的提示注入检测方法 

**Authors**: Yi Ji, Runzhi Li, Baolei Mao  

**Link**: [PDF](https://arxiv.org/pdf/2506.06384)  

**Abstract**: With the widespread adoption of Large Language Models (LLMs), prompt injection attacks have emerged as a significant security threat. Existing defense mechanisms often face critical trade-offs between effectiveness and generalizability. This highlights the urgent need for efficient prompt injection detection methods that are applicable across a wide range of LLMs. To address this challenge, we propose DMPI-PMHFE, a dual-channel feature fusion detection framework. It integrates a pretrained language model with heuristic feature engineering to detect prompt injection attacks. Specifically, the framework employs DeBERTa-v3-base as a feature extractor to transform input text into semantic vectors enriched with contextual information. In parallel, we design heuristic rules based on known attack patterns to extract explicit structural features commonly observed in attacks. Features from both channels are subsequently fused and passed through a fully connected neural network to produce the final prediction. This dual-channel approach mitigates the limitations of relying only on DeBERTa to extract features. Experimental results on diverse benchmark datasets demonstrate that DMPI-PMHFE outperforms existing methods in terms of accuracy, recall, and F1-score. Furthermore, when deployed actually, it significantly reduces attack success rates across mainstream LLMs, including GLM-4, LLaMA 3, Qwen 2.5, and GPT-4o. 

**Abstract (ZH)**: 大规模语言模型中基于提示注入攻击的有效检测方法：DMPI-PMHFE框架 

---
# Human and AI collaboration in Fitness Education:A Longitudinal Study with a Pilates Instructor 

**Title (ZH)**: 人类与AI在健身教育中的协作：一项与普拉提教练的合作纵向研究 

**Authors**: Qian Huang, King Wang Poon  

**Link**: [PDF](https://arxiv.org/pdf/2506.06383)  

**Abstract**: Artificial intelligence is poised to transform teaching and coaching practices,yet its optimal role alongside human expertise remains this http URL study investigates human and AI collaboration in fitness education through a one year qualitative case study with a Pilates this http URL researcher participated in the instructor classes and conducted biweekly semi structured interviews to explore how generative AI could be integrated into class planning and instruction. 

**Abstract (ZH)**: 人工智能 impending transformation of teaching and coaching practices: exploring the optimal role of AI alongside human expertise through a qualitative case study of Pilates class planning and instruction with generative AI integration. 

---
# On the Fundamental Impossibility of Hallucination Control in Large Language Models 

**Title (ZH)**: 在大型语言模型中幻觉控制的基本不可能性 

**Authors**: Michał P. Karpowicz  

**Link**: [PDF](https://arxiv.org/pdf/2506.06382)  

**Abstract**: This paper explains \textbf{why it is impossible to create large language models that do not hallucinate and what are the trade-offs we should be looking for}. It presents a formal \textbf{impossibility theorem} demonstrating that no inference mechanism can simultaneously satisfy four fundamental properties: \textbf{truthful (non-hallucinatory) generation, semantic information conservation, relevant knowledge revelation, and knowledge-constrained optimality}. By modeling LLM inference as an \textbf{auction of ideas} where neural components compete to contribute to responses, we prove the impossibility using the Green-Laffont theorem. That mathematical framework provides a rigorous foundation for understanding the nature of inference process, with implications for model architecture, training objectives, and evaluation methods. 

**Abstract (ZH)**: 本文解释了为什么无法创建不会产生幻觉的大规模语言模型，并探讨了我们应寻找的权衡。它提出了一个形式化的不可能性定理，证明没有任何推理机制能同时满足四项基本属性：真实的（非幻觉）生成、语义信息保真、相关知识揭示以及知识约束下的最优性。通过将LLM推理建模为一个神经组件竞标想法的拍卖过程，我们利用Green-Laffont定理证明了这一不可能性。该数学框架为理解推理过程的本质提供了严格的基礎，对模型架构、训练目标和评估方法具有重要意义。 

---
# CPS-Guard: Framework for Dependability Assurance of AI- and LLM-Based Cyber-Physical Systems 

**Title (ZH)**: CPS-Guard: 身心物理系统中基于AI和大语言模型的可靠性和安全性保障框架 

**Authors**: Trisanth Srinivasan, Santosh Patapati, Himani Musku, Idhant Gode, Aditya Arora, Samvit Bhattacharya, Abubakr Nazriev, Sanika Hirave, Zaryab Kanjiani, Srinjoy Ghose, Srinidhi Shetty  

**Link**: [PDF](https://arxiv.org/pdf/2506.06381)  

**Abstract**: Cyber-Physical Systems (CPS) increasingly depend on advanced AI techniques to operate in critical applications. However, traditional verification and validation methods often struggle to handle the unpredictable and dynamic nature of AI components. In this paper, we introduce CPS-Guard, a novel framework that employs multi-role orchestration to automate the iterative assurance process for AI-powered CPS. By assigning specialized roles (e.g., safety monitoring, security assessment, fault injection, and recovery planning) to dedicated agents within a simulated environment, CPS-Guard continuously evaluates and refines AI behavior against a range of dependability requirements. We demonstrate the framework through a case study involving an autonomous vehicle navigating an intersection with an AI-based planner. Our results show that CPS-Guard effectively detects vulnerabilities, manages performance impacts, and supports adaptive recovery strategies, thereby offering a structured and extensible solution for rigorous V&V in safety- and security-critical systems. 

**Abstract (ZH)**: 基于多角色编排的CPS-Guard框架：AI驱动的CPS的迭代保证方法 

---
# Beyond the Norm: A Survey of Synthetic Data Generation for Rare Events 

**Title (ZH)**: 超越常规：罕见事件合成数据生成综述 

**Authors**: Jingyi Gu, Xuan Zhang, Guiling Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06380)  

**Abstract**: Extreme events, such as market crashes, natural disasters, and pandemics, are rare but catastrophic, often triggering cascading failures across interconnected systems. Accurate prediction and early warning can help minimize losses and improve preparedness. While data-driven methods offer powerful capabilities for extreme event modeling, they require abundant training data, yet extreme event data is inherently scarce, creating a fundamental challenge. Synthetic data generation has emerged as a powerful solution. However, existing surveys focus on general data with privacy preservation emphasis, rather than extreme events' unique performance requirements. This survey provides the first overview of synthetic data generation for extreme events. We systematically review generative modeling techniques and large language models, particularly those enhanced by statistical theory as well as specialized training and sampling mechanisms to capture heavy-tailed distributions. We summarize benchmark datasets and introduce a tailored evaluation framework covering statistical, dependence, visual, and task-oriented metrics. A central contribution is our in-depth analysis of each metric's applicability in extremeness and domain-specific adaptations, providing actionable guidance for model evaluation in extreme settings. We categorize key application domains and identify underexplored areas like behavioral finance, wildfires, earthquakes, windstorms, and infectious outbreaks. Finally, we outline open challenges, providing a structured foundation for advancing synthetic rare-event research. 

**Abstract (ZH)**: 极值事件的合成数据生成：面向重尾分布的独特性能要求与应用领域研究 

---
# Enhancing Decision-Making of Large Language Models via Actor-Critic 

**Title (ZH)**: 通过.actor-critic方法增强大型语言模型的决策制定 

**Authors**: Heng Dong, Kefei Duan, Chongjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06376)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable advancements in natural language processing tasks, yet they encounter challenges in complex decision-making scenarios that require long-term reasoning and alignment with high-level objectives. Existing methods either rely on short-term auto-regressive action generation or face limitations in accurately simulating rollouts and assessing outcomes, leading to sub-optimal decisions. This paper introduces a novel LLM-based Actor-Critic framework, termed LAC, that effectively improves LLM policies with long-term action evaluations in a principled and scalable way. Our approach addresses two key challenges: (1) extracting robust action evaluations by computing Q-values via token logits associated with positive/negative outcomes, enhanced by future trajectory rollouts and reasoning; and (2) enabling efficient policy improvement through a gradient-free mechanism. Experiments across diverse environments -- including high-level decision-making (ALFWorld), low-level action spaces (BabyAI-Text), and large action spaces (WebShop) -- demonstrate the framework's generality and superiority over state-of-the-art methods. Notably, our approach achieves competitive performance using 7B/8B parameter LLMs, even outperforming baseline methods employing GPT-4 in complex tasks. These results underscore the potential of integrating structured policy optimization with LLMs' intrinsic knowledge to advance decision-making capabilities in multi-step environments. 

**Abstract (ZH)**: 基于大型语言模型的演员-批评家框架：LAC在长期决策评估中的优化与应用 

---
# CR-BLEA: Contrastive Ranking for Adaptive Resource Allocation in Bilevel Evolutionary Algorithms 

**Title (ZH)**: CR-BLEA: 对比排序在双层进化算法中自适应资源分配中的应用 

**Authors**: Dejun Xu, Jijia Chen, Gary G. Yen, Min Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06362)  

**Abstract**: Bilevel optimization poses a significant computational challenge due to its nested structure, where each upper-level candidate solution requires solving a corresponding lower-level problem. While evolutionary algorithms (EAs) are effective at navigating such complex landscapes, their high resource demands remain a key bottleneck -- particularly the redundant evaluation of numerous unpromising lower-level tasks. Despite recent advances in multitasking and transfer learning, resource waste persists. To address this issue, we propose a novel resource allocation framework for bilevel EAs that selectively identifies and focuses on promising lower-level tasks. Central to our approach is a contrastive ranking network that learns relational patterns between paired upper- and lower-level solutions online. This knowledge guides a reference-based ranking strategy that prioritizes tasks for optimization and adaptively controls resampling based on estimated population quality. Comprehensive experiments across five state-of-the-art bilevel algorithms show that our framework significantly reduces computational cost while preserving -- or even enhancing -- solution accuracy. This work offers a generalizable strategy to improve the efficiency of bilevel EAs, paving the way for more scalable bilevel optimization. 

**Abstract (ZH)**: bilevel 优化因其嵌套结构导致了显著的计算挑战，在其结构中，每个高层候选解都需要解决相应的低层问题。虽然进化算法（EAs）在导航此类复杂景观方面非常有效，但它们对资源的需求仍然是一个关键瓶颈，尤其是对大量无前景低层任务的冗余评估。尽管最近在多任务和迁移学习方面取得了进展，但资源浪费仍然存在。为此，我们提出了一个用于 bilevel EAs 的新颖资源分配框架，该框架选择性地识别并专注于有前景的低层任务。我们方法的核心是一个在线学习配对高层和低层解决方案之间关系模式的对比排名网络。这些知识引导基于参考的排名策略来优先优化任务，并根据估计的种群质量自适应控制采样。在五个先进的 bilevel 算法的全面实验中，我们的框架显著降低了计算成本，同时保持甚至提高了解决方案准确性。这项工作提供了一种通用策略以提高 bilevel EAs 的效率，为更可扩展的 bilevel 优化奠定了基础。 

---
# Tactile MNIST: Benchmarking Active Tactile Perception 

**Title (ZH)**: 触觉MNIST：评估主动触觉感知 

**Authors**: Tim Schneider, Guillaume Duret, Cristiana de Farias, Roberto Calandra, Liming Chen, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2506.06361)  

**Abstract**: Tactile perception has the potential to significantly enhance dexterous robotic manipulation by providing rich local information that can complement or substitute for other sensory modalities such as vision. However, because tactile sensing is inherently local, it is not well-suited for tasks that require broad spatial awareness or global scene understanding on its own. A human-inspired strategy to address this issue is to consider active perception techniques instead. That is, to actively guide sensors toward regions with more informative or significant features and integrate such information over time in order to understand a scene or complete a task. Both active perception and different methods for tactile sensing have received significant attention recently. Yet, despite advancements, both fields lack standardized benchmarks. To bridge this gap, we introduce the Tactile MNIST Benchmark Suite, an open-source, Gymnasium-compatible benchmark specifically designed for active tactile perception tasks, including localization, classification, and volume estimation. Our benchmark suite offers diverse simulation scenarios, from simple toy environments all the way to complex tactile perception tasks using vision-based tactile sensors. Furthermore, we also offer a comprehensive dataset comprising 13,500 synthetic 3D MNIST digit models and 153,600 real-world tactile samples collected from 600 3D printed digits. Using this dataset, we train a CycleGAN for realistic tactile simulation rendering. By providing standardized protocols and reproducible evaluation frameworks, our benchmark suite facilitates systematic progress in the fields of tactile sensing and active perception. 

**Abstract (ZH)**: 触觉感知有潜力通过提供丰富的局部信息来显著增强灵巧的机器人操作，这些信息可以补充或替代其他传感模态，如视觉。然而，由于触觉感知本质上是局部的，它不适用于仅依靠自身进行广域意识或全局场景理解的任务。基于人类的策略是考虑主动感知技术，即引导传感器朝着具有更多信息或显著特征的区域，并通过时间上的信息整合来理解场景或完成任务。主动感知和不同类型的触觉传感方法近年来都得到了广泛关注。尽管取得了进展，这两个领域仍然缺乏标准化的基准。为了填补这一空白，我们引入了触觉MNIST基准套件，这是一个开源的、兼容Gymnasium的基准，专门设计用于主动触觉感知任务，包括定位、分类和体积估计。我们的基准套件提供了从简单玩具环境到使用视觉触觉传感器进行复杂触觉感知任务的多样化的模拟场景。此外，我们还提供了一个全面的数据集，包含13,500个合成的3D MNIST数字模型和来自600个3D打印数字的153,600个真实世界的触觉样本。利用这一数据集，我们训练了一个CycleGAN进行逼真的触觉仿真渲染。通过提供标准化协议和可重现的评估框架，我们的基准套件促进了触觉传感和主动感知领域的系统性进步。 

---
# From Transformers to Large Language Models: A systematic review of AI applications in the energy sector towards Agentic Digital Twins 

**Title (ZH)**: 从 Transformers 到大规模语言模型：能源领域代理数字孪生的 AI 应用综述 

**Authors**: Gabriel Antonesi, Tudor Cioara, Ionut Anghel, Vasilis Michalakopoulos, Elissaios Sarmas, Liana Toderean  

**Link**: [PDF](https://arxiv.org/pdf/2506.06359)  

**Abstract**: Artificial intelligence (AI) has long promised to improve energy management in smart grids by enhancing situational awareness and supporting more effective decision-making. While traditional machine learning has demonstrated notable results in forecasting and optimization, it often struggles with generalization, situational awareness, and heterogeneous data integration. Recent advances in foundation models such as Transformer architecture and Large Language Models (LLMs) have demonstrated improved capabilities in modelling complex temporal and contextual relationships, as well as in multi-modal data fusion which is essential for most AI applications in the energy sector. In this review we synthesize the rapid expanding field of AI applications in the energy domain focusing on Transformers and LLMs. We examine the architectural foundations, domain-specific adaptations and practical implementations of transformer models across various forecasting and grid management tasks. We then explore the emerging role of LLMs in the field: adaptation and fine tuning for the energy sector, the type of tasks they are suited for, and the new challenges they introduce. Along the way, we highlight practical implementations, innovations, and areas where the research frontier is rapidly expanding. These recent developments reviewed underscore a broader trend: Generative AI (GenAI) is beginning to augment decision-making not only in high-level planning but also in day-to-day operations, from forecasting and grid balancing to workforce training and asset onboarding. Building on these developments, we introduce the concept of the Agentic Digital Twin, a next-generation model that integrates LLMs to bring autonomy, proactivity, and social interaction into digital twin-based energy management systems. 

**Abstract (ZH)**: 人工智能（AI）长期以来一直致力于通过增强情境意识和支持更为有效的决策来改进智能电网的能源管理。尽管传统的机器学习在预测和优化方面取得了显著成果，但在泛化、情境意识和异构数据集成方面仍面临挑战。最近，基础模型如变换器架构和大型语言模型（LLMs）的进步展示了在建模复杂的时间和上下文关系以及多模态数据融合方面的增强能力，这对于能源领域的大多数AI应用至关重要。在本文综述中，我们总结了人工智能在能源领域应用的迅速扩展领域，重点关注变换器和LLMs。我们分析了变换器模型在各种预测和电网管理任务中的架构基础、领域特定适应和实际应用。然后探索了LLMs在该领域中的新兴作用：针对能源领域的适应和微调、适合的任务类型以及它们引入的新挑战。在此过程中，我们强调了实际应用、创新和研究前沿迅速扩展的领域。这些综述中的近期发展表明一种更广泛的趋势：生成式AI（GenAI）不仅在高层次规划中，也在日常运营中（从预测和电网平衡到人员培训和资产入职）开始增强决策制定。在这些发展的基础上，我们提出了代理数字 twin的概念，这是一种下一代模型，通过集成LLMs来将自主性、主动性和社会互动引入基于数字孪生的能源管理系统中。 

---
# Towards real-time assessment of infrasound event detection capability using deep learning-based transmission loss estimation 

**Title (ZH)**: 基于深度学习的传输损耗估计用于实时评估 infrasound 事件检测能力 

**Authors**: Alice Janela Cameijo, Alexis Le Pichon, Youcef Sklab, Souhila Arib, Quentin Brissaud, Sven peter Naesholm, Constantino Listowski, Samir Aknine  

**Link**: [PDF](https://arxiv.org/pdf/2506.06358)  

**Abstract**: Accurate modeling of infrasound transmission loss is essential for evaluating the performance of the International Monitoring System, enabling the effective design and maintenance of infrasound stations to support compliance of the Comprehensive Nuclear-Test-Ban Treaty. State-of-the-art propagation modeling tools enable transmission loss to be finely simulated using atmospheric models. However, the computational cost prohibits the exploration of a large parameter space in operational monitoring applications. To address this, recent studies made use of a deep learning algorithm capable of making transmission loss predictions almost instantaneously. However, the use of nudged atmospheric models leads to an incomplete representation of the medium, and the absence of temperature as an input makes the algorithm incompatible with long range propagation. In this study, we address these limitations by using both wind and temperature fields as inputs to a neural network, simulated up to 130 km altitude and 4,000 km distance. We also optimize several aspects of the neural network architecture. We exploit convolutional and recurrent layers to capture spatially and range-dependent features embedded in realistic atmospheric models, improving the overall performance. The neural network reaches an average error of 4 dB compared to full parabolic equation simulations and provides epistemic and data-related uncertainty estimates. Its evaluation on the 2022 Hunga Tonga-Hunga Ha'apai volcanic eruption demonstrates its prediction capability using atmospheric conditions and frequencies not included in the training. This represents a significant step towards near real-time assessment of International Monitoring System detection thresholds of explosive sources. 

**Abstract (ZH)**: 准确的 infrasound 传输损耗建模对于评估国际监测系统性能至关重要，有助于支持《全面禁核试验条约》的合规性设计和维护工作。最先进的传播建模工具可以通过大气模型精细模拟传输损耗。然而，计算成本限制了在实际监测应用中探索大量参数空间。为了解决这一问题，最近的研究采用了能够几乎瞬时进行传输损耗预测的深度学习算法。然而，使用校正的大气模型会使得对介质的表示不完整，缺少温度输入使得算法不适合远程传播。在本研究中，我们通过将风场和温度场作为神经网络的输入，模拟至130公里高度和4000公里距离，解决了这些限制。我们还优化了神经网络架构的多个方面。利用卷积层和循环层捕获现实大气模型中嵌入的空间和距离依赖特征，提升了整体性能。神经网络与完整的抛物方程模拟相比，平均误差为4 dB，并提供了一致性和数据相关的不确定性估计。其对2022年洪加汤加-洪加哈帕伊火山爆发的评估证明了其在训练数据未包含的大气条件和频率下的预测能力。这代表了朝着近实时评估国际监测系统爆炸源检测阈值的重要一步。 

---
# Large Language Models for EEG: A Comprehensive Survey and Taxonomy 

**Title (ZH)**: EEG领域的大型语言模型：一项全面的综述与分类 

**Authors**: Naseem Babu, Jimson Mathew, A. P. Vinod  

**Link**: [PDF](https://arxiv.org/pdf/2506.06353)  

**Abstract**: The growing convergence between Large Language Models (LLMs) and electroencephalography (EEG) research is enabling new directions in neural decoding, brain-computer interfaces (BCIs), and affective computing. This survey offers a systematic review and structured taxonomy of recent advancements that utilize LLMs for EEG-based analysis and applications. We organize the literature into four domains: (1) LLM-inspired foundation models for EEG representation learning, (2) EEG-to-language decoding, (3) cross-modal generation including image and 3D object synthesis, and (4) clinical applications and dataset management tools. The survey highlights how transformer-based architectures adapted through fine-tuning, few-shot, and zero-shot learning have enabled EEG-based models to perform complex tasks such as natural language generation, semantic interpretation, and diagnostic assistance. By offering a structured overview of modeling strategies, system designs, and application areas, this work serves as a foundational resource for future work to bridge natural language processing and neural signal analysis through language models. 

**Abstract (ZH)**: 大型语言模型（LLMs）与脑电图（EEG）研究的日益 convergence 为神经解码、脑-机接口（BCIs）和情感计算提供了新的发展方向。本文综述了利用LLMs进行EEG基础模型、EEG到语言解码、跨模态生成以及临床应用和数据管理工具的近期进展。综述强调了通过微调、少样本学习和零样本学习适应的变换器架构如何使EEG模型能够执行复杂任务，如自然语言生成、语义解释和诊断辅助。通过提供建模策略、系统设计和应用领域的结构化概述，本文为未来通过语言模型弥合自然语言处理与神经信号分析之间的差距奠定了基础。 

---
# Deep learning methods for modeling infrasound transmission loss in the middle atmosphere 

**Title (ZH)**: 深学习方法用于中高层大气 infrasound 传输衰减建模 

**Authors**: Alexis Le Pichon, Alice Janela Cameijo, Samir Aknine, Youcef Sklab, Souhila Arib, Quentin Brissaud, Sven Peter Naesholm  

**Link**: [PDF](https://arxiv.org/pdf/2506.06351)  

**Abstract**: Accurate modeling of infrasound transmission losses (TLs) is essential to assess the performance of the global International Monitoring System infrasound network. Among existing propagation modeling tools, parabolic equation (PE) method enables TLs to be finely modeled, but its computational cost does not allow exploration of a large parameter space for operational monitoring applications. To reduce computation times, Brissaud et al. 2023 explored the potential of convolutional neural networks trained on a large set of regionally simulated wavefields (< 1000 km from the source) to predict TLs with negligible computation times compared to PE simulations. However, this method struggles in unfavorable initial wind conditions, especially at high frequencies, and causal issues with winds at large distances from the source affecting ground TLs close to the source. In this study, we have developed an optimized convolutional network designed to minimize prediction errors while predicting TLs from globally simulated combined temperature and wind fields spanning over propagation ranges of 4000 km. Our approach enhances the previously proposed one by implementing key optimizations that improve the overall architecture performance. The implemented model predicts TLs with an average error of 8.6 dB in the whole frequency band (0.1-3.2 Hz) and explored realistic atmospheric scenarios. 

**Abstract (ZH)**: 精确 modeling  infrasound 传输损耗 (TLs) 对评估国际监测系统全球 infrasound 网络性能至关重要。现有的传播建模工具中，抛物线方程 (PE) 方法能够精细建模 TLs，但由于计算成本较高，无法在操作监测应用中探索大量参数空间。为减少计算时间，Brissaud 等人（2023）探索了通过在区域模拟波场（<1000 km 从声源）上训练的卷积神经网络预测 TLs 的潜力，从而与 PE 模拟相比具有可忽略的计算时间。然而，这种方法在不利的初始风条件，尤其是高频率下，以及远处风的影响导致靠近声源地面 TLs 时存在因果问题。在本研究中，我们开发了一种优化的卷积网络，旨在预测全球模拟的温度和风场结合产生的 TLs 时最小化预测误差，传播范围为 4000 km。本方法通过实施关键优化措施，提高了整体架构性能。已实现的模型在整个频率范围（0.1-3.2 Hz）内的平均误差为 8.6 dB，并探索了现实的气象场景。 

---
# Unified Game Moderation: Soft-Prompting and LLM-Assisted Label Transfer for Resource-Efficient Toxicity Detection 

**Title (ZH)**: 统一游戏内容管理：基于软提示和大语言模型辅助的标签迁移方法以实现资源高效的内容毒性检测 

**Authors**: Zachary Yang, Domenico Tullo, Reihaneh Rabbany  

**Link**: [PDF](https://arxiv.org/pdf/2506.06347)  

**Abstract**: Toxicity detection in gaming communities faces significant scaling challenges when expanding across multiple games and languages, particularly in real-time environments where computational efficiency is crucial. We present two key findings to address these challenges while building upon our previous work on ToxBuster, a BERT-based real-time toxicity detection system. First, we introduce a soft-prompting approach that enables a single model to effectively handle multiple games by incorporating game-context tokens, matching the performance of more complex methods like curriculum learning while offering superior scalability. Second, we develop an LLM-assisted label transfer framework using GPT-4o-mini to extend support to seven additional languages. Evaluations on real game chat data across French, German, Portuguese, and Russian achieve macro F1-scores ranging from 32.96% to 58.88%, with particularly strong performance in German, surpassing the English benchmark of 45.39%. In production, this unified approach significantly reduces computational resources and maintenance overhead compared to maintaining separate models for each game and language combination. At Ubisoft, this model successfully identifies an average of 50 players, per game, per day engaging in sanctionable behavior. 

**Abstract (ZH)**: 多游戏多语言环境中的实时毒性检测扩展面临显著的规模挑战，特别是在需要高效计算的实时环境中。我们介绍了两种关键发现，以解决这些挑战并建立在我们之前的工作ToxBuster——一个基于BERT的实时毒性检测系统的基础之上。首先，我们提出了一种软提示方法，通过引入游戏上下文标记，使单个模型能够有效处理多个游戏，其性能与 Curriculum 学习等复杂方法相当，同时具有更强的扩展性。其次，我们开发了一种由GPT-4o-mini辅助的大语言模型辅助标签转移框架，以支持七种额外语言。在涵盖法语、德语、葡萄牙语和俄语的真实游戏聊天数据上的评估达到了32.96%至58.88%的宏F1分数，特别是在德语上的表现尤为突出，超过了45.39%的英语基准。在生产环境中，这种统一方法相比为每个游戏和语言组合维护单独模型，显著降低了计算资源和维护开销。在育碧，该模型每天平均能够识别出每个游戏中有50名执行可处罚行为的玩家。 

---
# Explainable-AI powered stock price prediction using time series transformers: A Case Study on BIST100 

**Title (ZH)**: 基于时间序列变换器的可解释AI股价预测：对BIST100的实际案例研究 

**Authors**: Sukru Selim Calik, Andac Akyuz, Zeynep Hilal Kilimci, Kerem Colak  

**Link**: [PDF](https://arxiv.org/pdf/2506.06345)  

**Abstract**: Financial literacy is increasingly dependent on the ability to interpret complex financial data and utilize advanced forecasting tools. In this context, this study proposes a novel approach that combines transformer-based time series models with explainable artificial intelligence (XAI) to enhance the interpretability and accuracy of stock price predictions. The analysis focuses on the daily stock prices of the five highest-volume banks listed in the BIST100 index, along with XBANK and XU100 indices, covering the period from January 2015 to March 2025. Models including DLinear, LTSNet, Vanilla Transformer, and Time Series Transformer are employed, with input features enriched by technical indicators. SHAP and LIME techniques are used to provide transparency into the influence of individual features on model outputs. The results demonstrate the strong predictive capabilities of transformer models and highlight the potential of interpretable machine learning to empower individuals in making informed investment decisions and actively engaging in financial markets. 

**Abstract (ZH)**: 基于变压器的时间序列模型与可解释人工智能的结合：提升股票价格预测的可解释性和准确性 

---
# A Reinforcement Learning Approach for RIS-aided Fair Communications 

**Title (ZH)**: 基于RIS辅助的公平通信 reinforcement learning方法 

**Authors**: Alex Pierron, Michel Barbeau, Luca De Cicco, Jose Rubio-Hernan, Joaquin Garcia-Alfaro  

**Link**: [PDF](https://arxiv.org/pdf/2506.06344)  

**Abstract**: Reconfigurable Intelligent Surfaces (RISs) are composed of physical elements that can dynamically alter electromagnetic wave properties to enhance beamforming and leading to improvements in areas with low coverage properties. They have the potential to be combined with Reinforcement Learning (RL) techniques to achieve network performance and energy efficiency via optimization techniques. In addition to performance and energy improvements, it is also crucial to consider the concept of fair communications. RISs must ensure that User Equipment (UE) units receive their signals with adequate strength, without other UE being deprived of service due to insufficient power. In this paper, we address such a problem. We explore the fairness properties of previous work and propose a novel method that aims at obtaining an efficient and fair duplex RIS-RL system for multiple legitimate UE units. We report and discuss our experimental work and simulation results. We also release our code and datasets to foster further research in the topic. 

**Abstract (ZH)**: 可重构智能表面(RIS)由能够动态改变电磁波性质以增强波束成形并改善覆盖不足区域性能的物理元件组成。它们有潜力与强化学习(RL)技术相结合，通过优化技术实现网络性能和能效提升。除了性能和能效的提升，公平通信的概念也同样重要。RIS必须确保用户设备(UE)接收到足够的信号强度，而不使其他UE因功率不足而无法服务。在本文中，我们探讨了先前工作的公平性特性，并提出了一种新型方法，旨在为多个合法UE单位获得高效且公平的单工RIS-RL系统。我们报告并讨论了我们的实验工作和仿真结果，同时发布我们的代码和数据集以促进该领域进一步研究。 

---
# TESU-LLM: Training Speech-LLMs Without Speech via Unified Encoder Alignment 

**Title (ZH)**: TESU-LLM：通过统一编码对齐训练无语音的语音LLM 

**Authors**: Taesoo Kim, Jong Hwan Ko  

**Link**: [PDF](https://arxiv.org/pdf/2506.06343)  

**Abstract**: Recent advances in speech-enabled language models have shown promising results in building intelligent voice assistants. However, most existing approaches rely on large-scale paired speech-text data and extensive computational resources, which pose challenges in terms of scalability and accessibility. In this paper, we present \textbf{TESU-LLM}, a novel framework that enables training speech-capable language models using only text data. Our key insight is to leverage a unified encoder that maps semantically equivalent text and speech inputs to a shared latent space. By aligning the encoder output with the embedding space of a LLM via a lightweight projection network, we enable the model to generalize from text-only supervision to speech-based inference. Despite being trained exclusively on text, TESU-LLM achieves strong performance on various speech-related benchmarks, comparable to baseline methods trained with large-scale multimodal datasets and substantial computational resources. These results highlight the effectiveness and efficiency of our approach, offering a scalable path toward building speech LLMs without speech data. 

**Abstract (ZH)**: Recent Advances in Training Speech-Capable Language Models Using Only Text Data 

---
# NR4DER: Neural Re-ranking for Diversified Exercise Recommendation 

**Title (ZH)**: NR4DER：神经网络重排ranking以实现多样化的运动推荐 

**Authors**: Xinghe Cheng, Xufang Zhou, Liangda Fang, Chaobo He, Yuyu Zhou, Weiqi Luo, Zhiguo Gong, Quanlong Guan  

**Link**: [PDF](https://arxiv.org/pdf/2506.06341)  

**Abstract**: With the widespread adoption of online education platforms, an increasing number of students are gaining new knowledge through Massive Open Online Courses (MOOCs). Exercise recommendation have made strides toward improving student learning outcomes. However, existing methods not only struggle with high dropout rates but also fail to match the diverse learning pace of students. They frequently face difficulties in adjusting to inactive students' learning patterns and in accommodating individualized learning paces, resulting in limited accuracy and diversity in recommendations. To tackle these challenges, we propose Neural Re-ranking for Diversified Exercise Recommendation (in short, NR4DER). NR4DER first leverages the mLSTM model to improve the effectiveness of the exercise filter module. It then employs a sequence enhancement method to enhance the representation of inactive students, accurately matches students with exercises of appropriate difficulty. Finally, it utilizes neural re-ranking to generate diverse recommendation lists based on individual students' learning histories. Extensive experimental results indicate that NR4DER significantly outperforms existing methods across multiple real-world datasets and effectively caters to the diverse learning pace of students. 

**Abstract (ZH)**: 基于神经重排的多样化习题推荐（NR4DER） 

---
# Structured Semantics from Unstructured Notes: Language Model Approaches to EHR-Based Decision Support 

**Title (ZH)**: 从非结构化笔记中提取结构化语义：基于EHR的语言模型决策支持方法 

**Authors**: Wu Hao Ran, Xi Xi, Furong Li, Jingyi Lu, Jian Jiang, Hui Huang, Yuzhuan Zhang, Shi Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06340)  

**Abstract**: The advent of large language models (LLMs) has opened new avenues for analyzing complex, unstructured data, particularly within the medical domain. Electronic Health Records (EHRs) contain a wealth of information in various formats, including free text clinical notes, structured lab results, and diagnostic codes. This paper explores the application of advanced language models to leverage these diverse data sources for improved clinical decision support. We will discuss how text-based features, often overlooked in traditional high dimensional EHR analysis, can provide semantically rich representations and aid in harmonizing data across different institutions. Furthermore, we delve into the challenges and opportunities of incorporating medical codes and ensuring the generalizability and fairness of AI models in healthcare. 

**Abstract (ZH)**: 大型语言模型的兴起为分析医疗领域的复杂非结构化数据开辟了新途径。电子健康记录（EHRs）包含多种形式的丰富信息，包括自由文本临床笔记、结构化实验室结果和诊断代码。本文探讨了高级语言模型在利用这些多样化的数据来源以提高临床决策支持方面的应用。我们将讨论文本特征在传统高维EHR分析中常被忽视的优势，这些特征能够提供丰富的语义表示，并有助于跨机构的数据协调。此外，本文还将探讨整合医疗代码以及确保医疗健康领域中AI模型的普适性和公平性的挑战和机遇。 

---
# Optimizing RAG Pipelines for Arabic: A Systematic Analysis of Core Components 

**Title (ZH)**: 优化阿拉伯语RAG管道：核心组件的系统分析 

**Authors**: Jumana Alsubhi, Mohammad D. Alahmadi, Ahmed Alhusayni, Ibrahim Aldailami, Israa Hamdine, Ahmad Shabana, Yazeed Iskandar, Suhayb Khayyat  

**Link**: [PDF](https://arxiv.org/pdf/2506.06339)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful architecture for combining the precision of retrieval systems with the fluency of large language models. While several studies have investigated RAG pipelines for high-resource languages, the optimization of RAG components for Arabic remains underexplored. This study presents a comprehensive empirical evaluation of state-of-the-art RAG components-including chunking strategies, embedding models, rerankers, and language models-across a diverse set of Arabic datasets. Using the RAGAS framework, we systematically compare performance across four core metrics: context precision, context recall, answer faithfulness, and answer relevancy. Our experiments demonstrate that sentence-aware chunking outperforms all other segmentation methods, while BGE-M3 and Multilingual-E5-large emerge as the most effective embedding models. The inclusion of a reranker (bge-reranker-v2-m3) significantly boosts faithfulness in complex datasets, and Aya-8B surpasses StableLM in generation quality. These findings provide critical insights for building high-quality Arabic RAG pipelines and offer practical guidelines for selecting optimal components across different document types. 

**Abstract (ZH)**: 检索增强生成（RAG）架构已成为将检索系统的精准度与大型语言模型的流畅性相结合的一种强大工具。尽管已有研究调查了高资源语言的RAG管道，但阿拉伯语RAG组件的优化仍缺乏探讨。本研究提供了对先进的RAG组件（包括分块策略、嵌入模型、重排器和语言模型）在多样化阿拉伯语数据集上的全面 empirical 评估。通过使用 RAGAS 框架，我们系统地在四个核心指标：上下文精确度、上下文召回率、答案忠实度和答案相关性上进行了性能比较。实验结果表明，句意识分块优于所有其他分段方法，而 BGE-M3 和 Multilingual-E5-large 是最有效的嵌入模型。引入重排器（bge-reranker-v2-m3）在复杂数据集中的显著提高了忠实度，Aya-8B 在生成质量上超过了 StableLM。这些发现为构建高质量的阿拉伯语 RAG 管道提供了关键见解，并提供了根据不同文档类型选择最佳组件的实用指南。 

---
# FinBERT2: A Specialized Bidirectional Encoder for Bridging the Gap in Finance-Specific Deployment of Large Language Models 

**Title (ZH)**: FinBERT2：专为金融领域大型语言模型部署差距桥接设计的专用双向编码器 

**Authors**: Xuan Xu, Fufang Wen, Beilin Chu, Zhibing Fu, Qinhong Lin, Jiaqi Liu, Binjie Fei, Zhongliang Yang, Linna Zhou, Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06335)  

**Abstract**: In natural language processing (NLP), the focus has shifted from encoder-only tiny language models like BERT to decoder-only large language models(LLMs) such as GPT-3. However, LLMs' practical application in the financial sector has revealed three limitations: (1) LLMs often perform worse than fine-tuned BERT on discriminative tasks despite costing much higher computational resources, such as market sentiment analysis in financial reports; (2) Application on generative tasks heavily relies on retrieval augmented generation (RAG) methods to provide current and specialized information, with general retrievers showing suboptimal performance on domain-specific retrieval tasks; (3) There are additional inadequacies in other feature-based scenarios, such as topic modeling. We introduce FinBERT2, a specialized bidirectional encoder pretrained on a high-quality, financial-specific corpus of 32b tokens. This represents the largest known Chinese financial pretraining corpus for models of this parameter size. As a better backbone, FinBERT2 can bridge the gap in the financial-specific deployment of LLMs through the following achievements: (1) Discriminative fine-tuned models (Fin-Labelers) outperform other (Fin)BERT variants by 0.4%-3.3% and leading LLMs by 9.7%-12.3% on average across five financial classification tasks. (2) Contrastive fine-tuned models (Fin-Retrievers) outperform both open-source (e.g., +6.8\% avg improvement over BGE-base-zh) and proprietary (e.g., +4.2\% avg improvement over OpenAI's text-embedding-3-large) embedders across five financial retrieval tasks; (3) Building on FinBERT2 variants, we construct the Fin-TopicModel, which enables superior clustering and topic representation for financial titles. Our work revisits financial BERT models through comparative analysis with contemporary LLMs and offers practical insights for effectively utilizing FinBERT in the LLMs era. 

**Abstract (ZH)**: 在自然语言处理（NLP）中，焦点已从如BERT这样的编码器型小型语言模型转向如GPT-3这样的解码器型大型语言模型（LLMs）。然而，LLMs在金融领域的实际应用揭示了三个局限性：（1）尽管在诸如市场情绪分析等辨别任务中成本远远高于微调的BERT，LLMs的表现通常逊于后者；（2）在生成任务上，LLMs的实现严重依赖于检索增强生成（RAG）方法，通用检索器在特定领域检索任务上表现不佳；（3）在其他基于特征的场景中还有额外不足，例如主题建模。我们介绍了FinBERT2，这是一种预训练在高质量320亿令牌的金融专用语料库上的双向编码器。这是已知的参数量最多、用于此类规模模型的中国金融预训练语料库。作为更好的主体模型，FinBERT2可以通过以下成就来弥补LLMs在金融专用部署中的不足：（1）辨别微调模型（Fin-Labelers）在五个金融分类任务中比其他（Fin）BERT变体高出0.4%-3.3%，平均比领先LLMs高出9.7%-12.3%；（2）对比微调模型（Fin-Retrievers）在五个金融检索任务中优于开源（例如，BGE-base-zh平均改善6.8%）和专有模型（例如，OpenAI的text-embedding-3-large平均改善4.2%）；（3）基于FinBERT2的变体，我们构建了Fin-TopicModel，使得金融标题的聚类和主题表示更加出色。我们的研究通过与当代LLMs的比较分析重访了金融BERT模型，并为有效地利用FinBERT在LLMs时代提供了实用见解。 

---
# Introduction to Predictive Coding Networks for Machine Learning 

**Title (ZH)**: 预测编码网络在机器学习中的介绍 

**Authors**: Mikko Stenlund  

**Link**: [PDF](https://arxiv.org/pdf/2506.06332)  

**Abstract**: Predictive coding networks (PCNs) constitute a biologically inspired framework for understanding hierarchical computation in the brain, and offer an alternative to traditional feedforward neural networks in ML. This note serves as a quick, onboarding introduction to PCNs for machine learning practitioners. We cover the foundational network architecture, inference and learning update rules, and algorithmic implementation. A concrete image-classification task (CIFAR-10) is provided as a benchmark-smashing application, together with an accompanying Python notebook containing the PyTorch implementation. 

**Abstract (ZH)**: 基于预测编码网络（PCNs）构成的生物学启发式框架用于理解大脑中的分层计算，并为机器学习提供了传统前馈神经网络之外的替代方案。本文为机器学习从业者提供了一个快速入门介绍。我们涵盖了基础网络架构、推理和学习更新规则以及算法实现。提供了一个具体的图像分类任务（CIFAR-10）作为 benchmark 的应用示例，并附带了一个包含 PyTorch 实现的 Python 笔记本。 

---
# How Significant Are the Real Performance Gains? An Unbiased Evaluation Framework for GraphRAG 

**Title (ZH)**: GraphRAG的真实性能提升有多显著？一个无偏评价框架 

**Authors**: Qiming Zeng, Xiao Yan, Hao Luo, Yuhao Lin, Yuxiang Wang, Fangcheng Fu, Bo Du, Quanqing Xu, Jiawei Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06331)  

**Abstract**: By retrieving contexts from knowledge graphs, graph-based retrieval-augmented generation (GraphRAG) enhances large language models (LLMs) to generate quality answers for user questions. Many GraphRAG methods have been proposed and reported inspiring performance in answer quality. However, we observe that the current answer evaluation framework for GraphRAG has two critical flaws, i.e., unrelated questions and evaluation biases, which may lead to biased or even wrong conclusions on performance. To tackle the two flaws, we propose an unbiased evaluation framework that uses graph-text-grounded question generation to produce questions that are more related to the underlying dataset and an unbiased evaluation procedure to eliminate the biases in LLM-based answer assessment. We apply our unbiased framework to evaluate 3 representative GraphRAG methods and find that their performance gains are much more moderate than reported previously. Although our evaluation framework may still have flaws, it calls for scientific evaluations to lay solid foundations for GraphRAG research. 

**Abstract (ZH)**: 基于图的检索增强生成（GraphRAG）通过从知识图谱中检索上下文增强大型语言模型（LLMs），以生成高质量的答案。尽管提出了许多GraphRAG方法并报告了令人鼓舞的答案质量性能，但我们观察到当前的答案评估框架存在两个关键缺陷，即无关问题和评估偏见，这可能会导致性能评估的有偏或甚至错误结论。为解决这两个问题，我们提出了一种无偏评估框架，该框架使用图-文本-数据集 grounding 问题生成方法生成与基础数据集更相关的問題，并提出了一种无偏的评估程序以消除基于LLM的答案评估中的偏见。我们将我们的无偏框架应用于评估3种代表性的GraphRAG方法，并发现它们的性能提升远不及之前报道的那样显著。尽管我们的评估框架仍可能存在缺陷，但它呼吁进行科学评估，为GraphRAG研究奠定坚实基础。 

---
# Evolutionary model for energy trading in community microgrids using Hawk-Dove strategies 

**Title (ZH)**: 基于hawk-dove策略的社区微电网能量交易演化模型 

**Authors**: Viorica Rozina Chifu, Tudor Cioara, Cristina Bianca Pop, Ionut Anghel  

**Link**: [PDF](https://arxiv.org/pdf/2506.06325)  

**Abstract**: This paper proposes a decentralized model of energy cooperation between microgrids, in which decisions are made locally, at the level of the microgrid community. Each microgrid is modeled as an autonomous agent that adopts a Hawk or Dove strategy, depending on the level of energy stored in the battery and its role in the energy trading process. The interactions between selling and buying microgrids are modeled through an evolutionary algorithm. An individual in the algorithm population is represented as an energy trading matrix that encodes the amounts of energy traded between the selling and buying microgrids. The population evolution is achieved by recombination and mutation operators. Recombination uses a specialized operator for matrix structures, and mutation is applied to the matrix elements according to a Gaussian distribution. The evaluation of an individual is made with a multi-criteria fitness function that considers the seller profit, the degree of energy stability at the community level, penalties for energy imbalance at the community level and for the degradation of microgrids batteries. The method was tested on a simulated scenario with 100 microgrids, each with its own selling and buying thresholds, to reflect a realistic environment with variable storage characteristics of microgrids batteries. By applying the algorithm on this scenario, 95 out of the 100 microgrids reached a stable energy state. This result confirms the effectiveness of the proposed model in achieving energy balance both at the individual level, for each microgrid, and at the level of the entire community. 

**Abstract (ZH)**: 本文提出了一种微电网之间的去中心化能源合作模型，在该模型中，决策是在微电网社区的局部水平上做出的。每个微电网被建模为一个自主代理，采用鹰或鸽策略，这取决于电池中储存的能源水平及其在能源交易过程中的角色。通过进化算法模型化卖电和购电微电网之间的交互。算法群体中的个体表示为一个能源交易矩阵，编码了卖电和购电微电网之间的能源交易量。群体的演化通过重组和变异算子实现。重组使用了专门针对矩阵结构的算子，变异则根据高斯分布应用于矩阵元素。个体的评估使用了多准则适应度函数，该函数考虑了销售利润、社区层面的能量稳定性、社区层面能源不平衡的惩罚以及微电网电池退化等因素。该方法在包含100个具有独立卖电和购电阈值的微电网的模拟场景中进行了测试，以反映具有可变存储特性的实际环境。通过对这一场景的应用，95个微电网达到了稳定能源状态。这一结果证实了所提出模型在实现个体微电网及整个社区层面的能源平衡方面的有效性。 

---
# Neural networks with image recognition by pairs 

**Title (ZH)**: 基于成对的图像识别的神经网络 

**Authors**: Polad Geidarov  

**Link**: [PDF](https://arxiv.org/pdf/2506.06322)  

**Abstract**: Neural networks based on metric recognition methods have a strictly determined architecture. Number of neurons, connections, as well as weights and thresholds values are calculated analytically, based on the initial conditions of tasks: number of recognizable classes, number of samples, metric expressions used. This paper discusses the possibility of transforming these networks in order to apply classical learning algorithms to them without using analytical expressions that calculate weight values. In the received network, training is carried out by recognizing images in pairs. This approach simplifies the learning process and easily allows to expand the neural network by adding new images to the recognition task. The advantages of these networks, including such as: 1) network architecture simplicity and transparency; 2) training simplicity and reliability; 3) the possibility of using a large number of images in the recognition problem using a neural network; 4) a consistent increase in the number of recognizable classes without changing the previous values of weights and thresholds. 

**Abstract (ZH)**: 基于度量识别方法的神经网络具有严格确定的架构。神经元的数量、连接方式以及权重和阈值值基于任务的初始条件（可识别类别数量、样本数量、使用的度量表达式）进行计算。本文讨论了将这些网络转换为能够应用经典学习算法的方法，无需使用计算权重值的解析表达式。在接收的网络中，通过成对识别图像进行训练。该方法简化了学习过程，并且可以轻松通过添加新的图像来扩展神经网络，用于识别任务。这些网络的优点包括：1）网络架构的简洁性和透明性；2）训练的简洁性和可靠性；3）能够使用大量图像解决识别问题；4）在不改变先前权重和阈值的情况下，一致增加可识别类别的数量。 

---
# MoE-Gyro: Self-Supervised Over-Range Reconstruction and Denoising for MEMS Gyroscopes 

**Title (ZH)**: MoE-Gyro: 自监督超出量程重建与 MEMS 陀螺仪降噪 

**Authors**: Feiyang Pan, Shenghe Zheng, Chunyan Yin, Guangbin Dou  

**Link**: [PDF](https://arxiv.org/pdf/2506.06318)  

**Abstract**: MEMS gyroscopes play a critical role in inertial navigation and motion control applications but typically suffer from a fundamental trade-off between measurement range and noise performance. Existing hardware-based solutions aimed at mitigating this issue introduce additional complexity, cost, and scalability challenges. Deep-learning methods primarily focus on noise reduction and typically require precisely aligned ground-truth signals, making them difficult to deploy in practical scenarios and leaving the fundamental trade-off unresolved. To address these challenges, we introduce Mixture of Experts for MEMS Gyroscopes (MoE-Gyro), a novel self-supervised framework specifically designed for simultaneous over-range signal reconstruction and noise suppression. MoE-Gyro employs two experts: an Over-Range Reconstruction Expert (ORE), featuring a Gaussian-Decay Attention mechanism for reconstructing saturated segments; and a Denoise Expert (DE), utilizing dual-branch complementary masking combined with FFT-guided augmentation for robust noise reduction. A lightweight gating module dynamically routes input segments to the appropriate expert. Furthermore, existing evaluation lack a comprehensive standard for assessing multi-dimensional signal enhancement. To bridge this gap, we introduce IMU Signal Enhancement Benchmark (ISEBench), an open-source benchmarking platform comprising the GyroPeak-100 dataset and a unified evaluation of IMU signal enhancement methods. We evaluate MoE-Gyro using our proposed ISEBench, demonstrating that our framework significantly extends the measurable range from 450 deg/s to 1500 deg/s, reduces Bias Instability by 98.4%, and achieves state-of-the-art performance, effectively addressing the long-standing trade-off in inertial sensing. 

**Abstract (ZH)**: MEMS陀螺仪在惯性导航和运动控制应用中扮演着关键角色，但通常会遇到量程和噪声性能之间的基本权衡。现有的硬件解决方案旨在缓解这一问题，但会引入额外的复杂性、成本和扩展性挑战。深度学习方法主要关注噪声减少，并且通常需要精确对齐的真实信号，使其难以在实际场景中部署，从而未能解决根本性的权衡问题。为应对这些挑战，我们提出了MEMS陀螺仪混合专家模型（MoE-Gyro），这是一种新颖的自监督框架，专门设计用于同时实现过量程信号重建和噪声抑制。MoE-Gyro采用两个专家：过量程重建专家（ORE），采用高斯衰减注意力机制来重建饱和段；降噪专家（DE），利用双分支互补掩蔽结合FFT引导增强进行稳健的噪声减少。一个轻量级门控模块动态将输入段路由到合适的专家。此外，现有的评估缺乏一个多维信号增强的全面标准。为弥补这一不足，我们引入了IMU信号增强基准（ISEBench），这是一个开源基准平台，包含GyroPeak-100数据集和IMU信号增强方法的统一评估。我们使用我们提出的ISEBench评估MoE-Gyro，结果显示我们的框架将可测量范围从450 deg/s扩展到1500 deg/s，减小了98.4%的偏差不稳定性，并实现了最先进的性能，有效解决了惯性传感领域的长期权衡问题。 

---
# A Reinforcement-Learning-Enhanced LLM Framework for Automated A/B Testing in Personalized Marketing 

**Title (ZH)**: 基于强化学习增强的LLM框架在个性化营销中的自动化A/B测试 

**Authors**: Haoyang Feng, Yanjun Dai, Yuan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.06316)  

**Abstract**: For personalized marketing, a new challenge of how to effectively algorithm the A/B testing to maximize user response is urgently to be overcome. In this paper, we present a new approach, the RL-LLM-AB test framework, for using reinforcement learning strategy optimization combined with LLM to automate and personalize A/B tests. The RL-LLM-AB test is built upon the pre-trained instruction-tuned language model. It first generates A/B versions of candidate content variants using a Prompt-Conditioned Generator, and then dynamically embeds and fuses the user portrait and the context of the current query with the multi-modal perception module to constitute the current interaction state. The content version is then selected in real-time through the policy optimization module with an Actor-Critic structure, and long-term revenue is estimated according to real-time feedback (such as click-through rate and conversion rate). Furthermore, a Memory-Augmented Reward Estimator is embedded into the framework to capture long-term user preference drift, which helps to generalize policy across multiple users and content contexts. Numerical results demonstrate the superiority of our proposed RL-LLM-ABTest over existing A/B testing methods, including classical A/B testing, Contextual Bandits, and benchmark reinforcement learning approaches on real-world marketing data. 

**Abstract (ZH)**: 针对个性化营销的新挑战，如何有效算法化A/B测试以最大化用户响应亟待解决。本文提出了一种新的方法——RL-LLM-AB测试框架，结合强化学习策略优化和 Large Language Model 自动化并个性化地进行A/B测试。RL-LLM-AB测试基于预训练的指令调整语言模型构建。它首先利用提示条件生成器生成候选内容变体的A/B版本，然后动态嵌入和融合用户画像和当前查询的上下文至多模态感知模块，构成当前交互状态。通过具有Actor-Critic结构的策略优化模块实时选择内容版本，并根据实时反馈（如点击率和转化率）估计长期收益。此外，框架中嵌入了记忆增强的奖励估计算法，以捕捉长期用户偏好漂移，有助于跨多个用户和内容上下文推广策略。实证结果表明，我们的RL-LLM-ABTest在实际营销数据上的表现优于现有的A/B测试方法，包括经典的A/B测试、上下文多臂老虎机以及基准强化学习方法。 

---
# DISRetrieval: Harnessing Discourse Structure for Long Document Retrieval 

**Title (ZH)**: DISRetrieval：利用话语结构进行长文档检索 

**Authors**: Huiyao Chen, Yi Yang, Yinghui Li, Meishan Zhang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06313)  

**Abstract**: Long document understanding has become increasingly crucial in natural language processing, with retrieval-based methods emerging as a promising solution to address the context length limitations of large language models (LLMs). However, existing approaches either treat documents as flat sequences or employ arbitrary chunking strategies, failing to capture the inherent discourse structure that guides human comprehension. We present DISRetrieval, a novel hierarchical retrieval framework that leverages linguistic discourse structure to enhance long document understanding. Our approach introduces three key innovations: (1) a discourse-aware document organization framework that utilizes rhetorical structure theory (RST) to create sentence-level hierarchical representations, preserving both semantic relationships and natural document flow; (2) an LLM-enhanced node representation technique that combines discourse structure with adaptive summarization to enrich tree nodes with contextual information; and (3) a hierarchical evidence retrieval mechanism that effectively selects relevant content while maintaining discourse coherence. Through comprehensive experiments on QASPER and QuALITY datasets, DISRetrieval demonstrates substantial improvements over existing methods in both token-level retrieval metrics and downstream question answering tasks. Our ablation studies confirm that incorporating discourse structure significantly enhances retrieval effectiveness across different document lengths and query types, validating the importance of linguistically-informed document representation in long-text understanding. Our code and datasets are publicly available at github/DreamH1gh/DISRetrieval to facilitate future research. 

**Abstract (ZH)**: 长文档理解在自然语言处理中变得愈发关键，基于检索的方法因其能解决大规模语言模型上下文长度限制而展现出潜力。然而，现有方法要么将文档视为扁平序列，要么采用任意的分块策略，无法捕捉引导人类理解的固有语篇结构。我们提出DISRetrieval，一种新颖的层次检索框架，利用语篇结构提升长文档理解能力。该方法引入了三个方面的主要创新：(1) 一种语篇意识的文档组织框架，利用论理性结构理论 (RST) 创建句级层次表示，同时保持语义关系和自然文档流程；(2) 结合语篇结构与自适应总结的LLM增强节点表示技术，丰富树节点的上下文信息；(3) 一种有效的层次证据检索机制，能够在保持语篇连贯性的同时选择相关内容。通过在QASPER和QuALITY数据集上的全面实验，DISRetrieval在token级检索指标和下游问答任务中均显著优于现有方法。我们的消融研究证实，结合语篇结构在不同文档长度和查询类型下的检索效果均有显著提升，验证了基于语言信息的文档表示在长文本理解中的重要性。我们的代码和数据集可在github/DreamH1gh/DISRetrieval公开获取，以促进未来研究。 

---
# Reward Is Enough: LLMs Are In-Context Reinforcement Learners 

**Title (ZH)**: 奖励足矣：LLMs是基于上下文的强化学习者 

**Authors**: Kefan Song, Amir Moeini, Peng Wang, Lei Gong, Rohan Chandra, Yanjun Qi, Shangtong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06303)  

**Abstract**: Reinforcement learning (RL) is a human-designed framework for solving sequential decision making problems. In this work, we demonstrate that, surprisingly, RL emerges in LLM's (Large Language Model) inference time -- a phenomenon known as in-context RL (ICRL). Specifically, we propose a novel multi-round prompting framework called ICRL prompting. The goal is to prompt the LLM to complete a task. After the LLM generates a response at the current round, we give numerical scalar feedbacks for the response, called the rewards. At the next round, we prompt the LLM again with the same task and a context consisting of all previous responses and rewards. We observe that the quality of the LLM's response increases as the context grows. In other words, the LLM is able to maximize the scalar reward signal in the inference time, just like an RL algorithm. We evaluate ICRL prompting in three benchmarks (Game of 24, creative writing, and ScienceWorld) and demonstrate significant performance improvements over baseline methods such as Self-Refine and Reflexion. Surprisingly, in some experiments the reward signals are generated by the LLM itself, yet performance improvements are still observed from ICRL prompting, offering a promising paradigm for scaling test-time compute. 

**Abstract (ZH)**: reinforced learning 在大语言模型推理时间的出现：基于上下文的强化学习 (ICRL) 推导框架 

---
# How Malicious AI Swarms Can Threaten Democracy 

**Title (ZH)**: 恶意AI集群如何威胁民主 

**Authors**: Daniel Thilo Schroeder, Meeyoung Cha, Andrea Baronchelli, Nick Bostrom, Nicholas A. Christakis, David Garcia, Amit Goldenberg, Yara Kyrychenko, Kevin Leyton-Brown, Nina Lutz, Gary Marcus, Filippo Menczer, Gordon Pennycook, David G. Rand, Frank Schweitzer, Christopher Summerfield, Audrey Tang, Jay Van Bavel, Sander van der Linden, Dawn Song, Jonas R. Kunst  

**Link**: [PDF](https://arxiv.org/pdf/2506.06299)  

**Abstract**: Advances in AI portend a new era of sophisticated disinformation operations. While individual AI systems already create convincing -- and at times misleading -- information, an imminent development is the emergence of malicious AI swarms. These systems can coordinate covertly, infiltrate communities, evade traditional detectors, and run continuous A/B tests, with round-the-clock persistence. The result can include fabricated grassroots consensus, fragmented shared reality, mass harassment, voter micro-suppression or mobilization, contamination of AI training data, and erosion of institutional trust. With democratic processes worldwide increasingly vulnerable, we urge a three-pronged response: (1) platform-side defenses -- always-on swarm-detection dashboards, pre-election high-fidelity swarm-simulation stress-tests, transparency audits, and optional client-side "AI shields" for users; (2) model-side safeguards -- standardized persuasion-risk tests, provenance-authenticating passkeys, and watermarking; and (3) system-level oversight -- a UN-backed AI Influence Observatory. 

**Abstract (ZH)**: AI的进步预示着一个新的复杂虚假信息操作时代。随着个体AI系统已经生成令人信服的——有时是误导性的——信息，即将到来的发展将是恶意AI集群的涌现。这些系统可以隐蔽协调、渗透社区、逃避传统检测器，并且进行持续的A/B测试，具备全天候的持续性。其结果可能包括伪造的草根共识、碎片化的共享现实、大规模骚扰、选民微抑制或动员、AI训练数据污染以及机构信任的侵蚀。随着全球民主进程日益脆弱，我们敦促采取三管齐下的应对措施：（1）平台侧防御——持续检测集群的仪表盘、高保真集群模拟预选前的压力测试、透明度审计以及可选的客户端“AI防护”；（2）模型侧保护——标准化说服风险测试、源认证通行证以及数字水印；（3）系统级监督——由联合国支持的AI影响力观察站。 

---
# Pairwise Calibrated Rewards for Pluralistic Alignment 

**Title (ZH)**: 多元共融的配对校准奖励 

**Authors**: Daniel Halpern, Evi Micha, Ariel D. Procaccia, Itai Shapira  

**Link**: [PDF](https://arxiv.org/pdf/2506.06298)  

**Abstract**: Current alignment pipelines presume a single, universal notion of desirable behavior. However, human preferences often diverge across users, contexts, and cultures. As a result, disagreement collapses into the majority signal and minority perspectives are discounted. To address this, we propose reflecting diverse human preferences through a distribution over multiple reward functions, each inducing a distinct aligned policy. The distribution is learned directly from pairwise preference without annotator identifiers or predefined groups. Instead, annotator disagreements are treated as informative soft labels. Our central criterion is pairwise calibration: for every pair of candidate responses, the proportion of reward functions preferring one response matches the fraction of annotators with that preference. We prove that even a small outlier-free ensemble can accurately represent diverse preference distributions. Empirically, we introduce and validate a practical training heuristic to learn such ensembles, and demonstrate its effectiveness through improved calibration, implying a more faithful representation of pluralistic values. 

**Abstract (ZH)**: 当前对齐管道假设了一种单一同质性的理想行为观念。然而，人类偏好在用户、情境和文化之间常常存在分歧。因此，分歧意见被归结为多数信号，而少数视角被忽视。为解决这一问题，我们提出通过多个奖励函数的概率分布来反映多样的人类偏好，每个奖励函数诱导一种独特的对齐策略。该分布直接从成对偏好中学习，而不依赖标注者的识别信息或预定义的组别。相反，标注者之间的分歧被视为有信息性的软标签。我们核心的标准是成对校准：对于每一对候选响应，偏好某一响应的奖励函数的比例匹配有同样偏好标注者的比例。我们证明，即使是一个无离群值的小组合也能准确代表多样化的偏好分布。实验上，我们引入并验证了一种实用的训练启发式方法来学习这样的组合，并通过提高校准来证明其效果，意味着更加忠实地代表了多元的价值观。 

---
# Optimal patient allocation for echocardiographic assessments 

**Title (ZH)**: 最优患者分配以进行心脏超声评估 

**Authors**: Bozhi Sun, Seda Tierney, Jeffrey A. Feinstein, Frederick Damen, Alison L. Marsden, Daniele E. Schiavazzi  

**Link**: [PDF](https://arxiv.org/pdf/2506.06297)  

**Abstract**: Scheduling echocardiographic exams in a hospital presents significant challenges due to non-deterministic factors (e.g., patient no-shows, patient arrival times, diverse exam durations, etc.) and asymmetric resource constraints between fetal and non-fetal patient streams. To address these challenges, we first conducted extensive pre-processing on one week of operational data from the Echo Laboratory at Stanford University's Lucile Packard Children's Hospital, to estimate patient no-show probabilities and derive empirical distributions of arrival times and exam durations. Based on these inputs, we developed a discrete-event stochastic simulation model using SimPy, and integrate it with the open source Gymnasium Python library. As a baseline for policy optimization, we developed a comparative framework to evaluate on-the-fly versus reservation-based allocation strategies, in which different proportions of resources are reserved in advance. Considering a hospital configuration with a 1:6 ratio of fetal to non-fetal rooms and a 4:2 ratio of fetal to non-fetal sonographers, we show that on-the-fly allocation generally yields better performance, more effectively adapting to patient variability and resource constraints. Building on this foundation, we apply reinforcement learning (RL) to derive an approximated optimal dynamic allocation policy. This RL-based policy is benchmarked against the best-performing rule-based strategies, allowing us to quantify their differences and provide actionable insights for improving echo lab efficiency through intelligent, data-driven resource management. 

**Abstract (ZH)**: 在圣地亚哥儿童医院卢西尔·帕克回声实验室中预约心脏超声检查面临着显著挑战，由于非确定性因素（如患者缺席、患者到达时间、多样化的检查时间等）和胎儿和非胎儿患者流之间的非对称资源约束。为应对这些挑战，我们首先对来自斯坦福大学卢西尔·帕克儿童医院回声实验室一周的运营数据进行了广泛的预处理，以估算患者缺席概率并推导出到达时间和检查时间的经验分布。基于这些输入，我们使用SimPy开发了一个离散事件随机仿真模型，并将其与开源的Gymnasium Python库集成。作为政策优化的基准，我们开发了一个比较框架，评估即时分配策略与预约分配策略，其中不同比例的资源提前预留。考虑到1:6的胎儿与非胎儿房间比例和4:2的胎儿与非胎儿超声技师比例，我们证明了即时分配策略通常表现出更好的性能，更有效地适应患者变异性及资源约束。在此基础上，我们应用强化学习（RL）来推导近似最优的动态分配策略。该基于RL的策略被基准测试与表现最佳的基于规则的策略，以量化它们之间的差异，并提供通过智能、数据驱动的资源管理提高回声实验室效率的可操作性见解。 

---
# Dynamic Graph CNN with Jacobi Kolmogorov-Arnold Networks for 3D Classification of Point Sets 

**Title (ZH)**: 基于雅可比柯尔莫哥洛夫-阿诺尔德网络的动态图CNN在点集三维分类中的应用 

**Authors**: Hanaa El Afia, Said Ohamouddou, Raddouane Chiheb, Abdellatif El Afia  

**Link**: [PDF](https://arxiv.org/pdf/2506.06296)  

**Abstract**: We introduce Jacobi-KAN-DGCNN, a framework that integrates Dynamic Graph Convolutional Neural Network (DGCNN) with Jacobi Kolmogorov-Arnold Networks (KAN) for the classification of three-dimensional point clouds. This method replaces Multi-Layer Perceptron (MLP) layers with adaptable univariate polynomial expansions within a streamlined DGCNN architecture, circumventing deep levels for both MLP and KAN to facilitate a layer-by-layer comparison. In comparative experiments on the ModelNet40 dataset, KAN layers employing Jacobi polynomials outperform the traditional linear layer-based DGCNN baseline in terms of accuracy and convergence speed, while maintaining parameter efficiency. Our results demonstrate that higher polynomial degrees do not automatically improve performance, highlighting the need for further theoretical and empirical investigation to fully understand the interactions between polynomial bases, degrees, and the mechanisms of graph-based learning. 

**Abstract (ZH)**: Jacobi-KAN-DGCNN：一种将动态图卷积神经网络与雅可比柯尔莫哥洛夫-阿诺尔德网络集成的框架 

---
# dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching 

**Title (ZH)**: dLLM-Cache：基于自适应缓存加速扩散大语言模型 

**Authors**: Zhiyuan Liu, Yicun Yang, Yaojie Zhang, Junjie Chen, Chang Zou, Qingyuan Wei, Shaobo Wang, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06295)  

**Abstract**: Autoregressive Models (ARMs) have long dominated the landscape of Large Language Models. Recently, a new paradigm has emerged in the form of diffusion-based Large Language Models (dLLMs), which generate text by iteratively denoising masked segments. This approach has shown significant advantages and potential. However, dLLMs suffer from high inference latency. Traditional ARM acceleration techniques, such as Key-Value caching, are incompatible with dLLMs due to their bidirectional attention mechanism. To address this specific challenge, our work begins with a key observation that dLLM inference involves a static prompt and a partially dynamic response, where most tokens remain stable across adjacent denoising steps. Based on this, we propose dLLM-Cache, a training-free adaptive caching framework that combines long-interval prompt caching with partial response updates guided by feature similarity. This design enables efficient reuse of intermediate computations without compromising model performance. Extensive experiments on representative dLLMs, including LLaDA 8B and Dream 7B, show that dLLM-Cache achieves up to 9.1 x speedup over standard inference without compromising output quality. Notably, our method brings dLLM inference latency close to that of ARMs under many settings. Codes are provided in the supplementary material and will be released publicly on GitHub. 

**Abstract (ZH)**: 自动回归模型（ARMs）长期主导着大型语言模型的领域。最近，基于扩散的大规模语言模型（dLLMs）崭露头角，通过迭代去噪掩蔽片段生成文本。这种方法显示了显著的优势和潜力，但dLLMs存在推理延迟高的问题。传统ARMs的加速技术，如Key-Value缓存，由于其双向注意力机制与dLLMs不兼容。为解决这一特定挑战，我们工作基于一个关键观察，即dLLM推理涉及静态提示和部分动态响应，在相邻去噪步骤中大多数令牌保持稳定。基于此，我们提出了一种无需训练的自适应缓存框架dLLM-Cache，该框架结合了长间隔提示缓存和由特征相似性引导的部分响应更新。此设计使得能够高效重用中间计算而不影响模型性能。在包括LLaDA 8B和Dream 7B在内的代表性dLLMs上的 extensive 实验显示，dLLM-Cache在不牺牲输出质量的情况下实现了高达9.1倍的速度提升。值得注意的是，我们的方法在许多情况下将dLLM推理延迟接近ARMs。相关代码已包含在补充材料中，并将在GitHub上公开发布。 

---
# GLProtein: Global-and-Local Structure Aware Protein Representation Learning 

**Title (ZH)**: GLProtein: 全局与局部结构意识的蛋白质表示学习 

**Authors**: Yunqing Liu, Wenqi Fan, Xiaoyong Wei, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06294)  

**Abstract**: Proteins are central to biological systems, participating as building blocks across all forms of life. Despite advancements in understanding protein functions through protein sequence analysis, there remains potential for further exploration in integrating protein structural information. We argue that the structural information of proteins is not only limited to their 3D information but also encompasses information from amino acid molecules (local information) to protein-protein structure similarity (global information). To address this, we propose \textbf{GLProtein}, the first framework in protein pre-training that incorporates both global structural similarity and local amino acid details to enhance prediction accuracy and functional insights. GLProtein innovatively combines protein-masked modelling with triplet structure similarity scoring, protein 3D distance encoding and substructure-based amino acid molecule encoding. Experimental results demonstrate that GLProtein outperforms previous methods in several bioinformatics tasks, including predicting protein-protein interaction, contact prediction, and so on. 

**Abstract (ZH)**: 蛋白质是生物系统的核心，参与所有生命形式中构建块的作用。尽管通过蛋白质序列分析已经取得了对蛋白质功能的深刻理解，但在整合蛋白质结构信息方面仍有潜在的空间。我们认为，蛋白质的结构信息不仅限于其三维信息，还涵盖了从氨基酸分子的局部信息到蛋白质-蛋白质结构相似性的全局信息。为解决这一问题，我们提出了GLProtein框架，这是首个结合全局结构相似性和局部氨基酸细节的蛋白质预训练框架，以提高预测准确性和功能见解。GLProtein创新性地结合了蛋白质遮蔽建模、三重结构相似性评分、蛋白质三维距离编码和基于子结构的氨基酸分子编码。实验结果表明，GLProtein在预测蛋白质-蛋白质相互作用、接触预测等多个生物信息学任务中优于先前的方法。 

---
# Prediction of Bank Credit Ratings using Heterogeneous Topological Graph Neural Networks 

**Title (ZH)**: 使用异构拓扑图神经网络预测银行信用评级 

**Authors**: Junyi Liu, Stanley Kok  

**Link**: [PDF](https://arxiv.org/pdf/2506.06293)  

**Abstract**: Agencies such as Standard & Poor's and Moody's provide bank credit ratings that influence economic stability and decision-making by stakeholders. Accurate and timely predictions support informed decision-making, regulatory actions, and investor protection. However, a complete interbank connection graph is often unavailable due to privacy concerns, complicating the direct application of Graph Neural Networks (GNNs) for rating prediction. our research utilizes persistent homology to construct a network that captures relationships among banks and combines this with a traditional lending network to create a heterogeneous network that integrates information from both sources, leading to improved predictions. Experiments on a global, real-world dataset validate the effectiveness of HTGNN. This research has implications for investors and regulatory bodies in enhancing proactive risk mitigation and the implementation of effective market this http URL code can be find at this https URL. 

**Abstract (ZH)**: 标准普尔和穆迪等机构提供的银行信用评级影响着经济稳定和利益相关方的决策。准确及时的预测支持知情决策、监管行动和投资者保护。但由于隐私问题，完整的银行间连接图通常不可用，这使得直接应用图神经网络（GNNs）进行评级预测变得复杂。我们的研究利用持久同调构建一个网络，捕捉银行之间的关系，并将此与传统贷款网络结合，创建一个异构网络，综合了两种来源的信息，从而提高了预测效果。全球实际数据集上的实验验证了HTGNN的有效性。这项研究对投资者和监管机构在增强前瞻性风险缓解和有效市场实施方面具有重要意义。相关代码可以在以下链接找到：[这里](这里)[这里](这里)。 

---
# Mutual-Taught for Co-adapting Policy and Reward Models 

**Title (ZH)**: 协同教学以共适应策略和奖励模型 

**Authors**: Tianyuan Shi, Canbin Huang, Fanqi Wan, Longguang Zhong, Ziyi Yang, Weizhou Shen, Xiaojun Quan, Ming Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.06292)  

**Abstract**: During the preference optimization of large language models (LLMs), distribution shifts may arise between newly generated model samples and the data used to train the reward model (RM). This shift reduces the efficacy of the RM, which in turn negatively impacts the performance of the policy model (PM). To address this challenge, we propose Mutual-Taught, a self-training method that iteratively improves both the PM and RM without requiring additional human annotation. Our approach mirrors the expectation-maximization (EM) algorithm. In the E-step, the PM is updated using feedback from the current RM, guiding the PM toward a better approximation of the latent optimal preference distribution. In the M-step, we update the RM by constructing training data from the outputs of the PM before and after the E-step update. This process ensures that the RM adapts to the evolving policy distribution. Experimental results demonstrate that this iterative approach leads to consistent improvements in both models. Specifically, our 8B policy model, LLaMA-3-8B-Instruct-MT, achieves a length-controlled win rate of 54.1\% on AlpacaEval-2, while our 8B reward model, FsfairX-LLaMA3-RM-MT, performs on par with GPT-4o-2024-08-06 on RewardBench. 

**Abstract (ZH)**: 在大规模语言模型的偏好优化过程中， Newly generated模型样本与用于训练奖励模型的数据之间的分布变化可能会导致奖励模型（RM）的效能下降，进而负面影响策略模型（PM）的表现。为应对这一挑战，我们提出了一种名为Mutual-Taught的自训练方法，该方法能够迭代地提高PM和RM的性能而不增加额外的人工标注。我们的方法借鉴了期望最大化（EM）算法。在E步中，使用当前RM的反馈更新PM，引导PM更好地逼近潜在的最优偏好分布。在M步中，通过使用E步更新前后PM的输出构建训练数据来更新RM，从而确保RM能够适应不断变化的策略分布。实验结果表明，这种迭代方法可以持续改进两个模型。具体而言，我们的8B策略模型LLaMA-3-8B-Instruct-MT在AlpacaEval-2上的可控长度胜率达到了54.1%，而我们的8B奖励模型FsfairX-LLaMA3-RM-MT在RewardBench上的表现与GPT-4o-2024-08-06相当。 

---
# Improvement of Optimization using Learning Based Models in Mixed Integer Linear Programming Tasks 

**Title (ZH)**: 基于学习模型在混合整数线性规划任务中优化的改进 

**Authors**: Xiaoke Wang, Batuhan Altundas, Zhaoxin Li, Aaron Zhao, Matthew Gombolay  

**Link**: [PDF](https://arxiv.org/pdf/2506.06291)  

**Abstract**: Mixed Integer Linear Programs (MILPs) are essential tools for solving planning and scheduling problems across critical industries such as construction, manufacturing, and logistics. However, their widespread adoption is limited by long computational times, especially in large-scale, real-time scenarios. To address this, we present a learning-based framework that leverages Behavior Cloning (BC) and Reinforcement Learning (RL) to train Graph Neural Networks (GNNs), producing high-quality initial solutions for warm-starting MILP solvers in Multi-Agent Task Allocation and Scheduling Problems. Experimental results demonstrate that our method reduces optimization time and variance compared to traditional techniques while maintaining solution quality and feasibility. 

**Abstract (ZH)**: 基于行为克隆和强化学习的图神经网络在多代理任务分配与调度问题中混合整数线性规划初解学习框架 

---
# CellCLIP -- Learning Perturbation Effects in Cell Painting via Text-Guided Contrastive Learning 

**Title (ZH)**: CellCLIP —— 通过文本引导的对比学习学习细胞绘画中的干扰效果 

**Authors**: Mingyu Lu, Ethan Weinberger, Chanwoo Kim, Su-In Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.06290)  

**Abstract**: High-content screening (HCS) assays based on high-throughput microscopy techniques such as Cell Painting have enabled the interrogation of cells' morphological responses to perturbations at an unprecedented scale. The collection of such data promises to facilitate a better understanding of the relationships between different perturbations and their effects on cellular state. Towards achieving this goal, recent advances in cross-modal contrastive learning could, in theory, be leveraged to learn a unified latent space that aligns perturbations with their corresponding morphological effects. However, the application of such methods to HCS data is not straightforward due to substantial differences in the semantics of Cell Painting images compared to natural images, and the difficulty of representing different classes of perturbations (e.g., small molecule vs CRISPR gene knockout) in a single latent space. In response to these challenges, here we introduce CellCLIP, a cross-modal contrastive learning framework for HCS data. CellCLIP leverages pre-trained image encoders coupled with a novel channel encoding scheme to better capture relationships between different microscopy channels in image embeddings, along with natural language encoders for representing perturbations. Our framework outperforms current open-source models, demonstrating the best performance in both cross-modal retrieval and biologically meaningful downstream tasks while also achieving significant reductions in computation time. 

**Abstract (ZH)**: 基于高内涵成像技术如Cell Painting的高内涵筛查（HCS） assay能够在前所未有的规模上探究细胞对扰动的形态学响应。通过收集此类数据，有望促进对不同扰动与其对细胞状态影响之间关系的更好理解。为了实现这一目标，近年来在跨模态对比学习方面的进展理论上可以被利用来学习一个统一的潜在空间，将扰动与相应的形态学效应对齐。然而，将此类方法应用于HCS数据并不直接，因为Cell Painting图像与自然图像在语义上存在显著差异，且难以在单个潜在空间中表示不同类别的扰动（例如，小分子与CRISPR基因敲除）。面对这些挑战，我们引入了CellCLIP，一种适用于HCS数据的跨模态对比学习框架。CellCLIP利用预训练的图像编码器和一种新颖的通道编码方案，在图像嵌入中更好地捕捉不同显微镜通道之间的关系，并结合自然语言编码器来表示扰动。我们的框架在跨模态检索和生物学上有意义的下游任务中均表现出色，同时显著减少了计算时间。 

---
# DELPHYNE: A Pre-Trained Model for General and Financial Time Series 

**Title (ZH)**: DELPHYNE：一个通用和金融时间序列的预训练模型 

**Authors**: Xueying Ding, Aakriti Mittal, Achintya Gopal  

**Link**: [PDF](https://arxiv.org/pdf/2506.06288)  

**Abstract**: Time-series data is a vital modality within data science communities. This is particularly valuable in financial applications, where it helps in detecting patterns, understanding market behavior, and making informed decisions based on historical data. Recent advances in language modeling have led to the rise of time-series pre-trained models that are trained on vast collections of datasets and applied to diverse tasks across financial domains. However, across financial applications, existing time-series pre-trained models have not shown boosts in performance over simple finance benchmarks in both zero-shot and fine-tuning settings. This phenomenon occurs because of a i) lack of financial data within the pre-training stage, and ii) the negative transfer effect due to inherently different time-series patterns across domains. Furthermore, time-series data is continuous, noisy, and can be collected at varying frequencies and with varying lags across different variables, making this data more challenging to model than languages. To address the above problems, we introduce a Pre-trained MoDEL for FINance TimE-series (Delphyne). Delphyne achieves competitive performance to existing foundation and full-shot models with few fine-tuning steps on publicly available datasets, and also shows superior performances on various financial tasks. 

**Abstract (ZH)**: 预训练金融时间序列模型（Delphyne） 

---
# Disentangling AI Alignment: A Structured Taxonomy Beyond Safety and Ethics 

**Title (ZH)**: 解构AI对齐：超越安全与伦理的结构化分类体系 

**Authors**: Kevin Baum  

**Link**: [PDF](https://arxiv.org/pdf/2506.06286)  

**Abstract**: Recent advances in AI research make it increasingly plausible that artificial agents with consequential real-world impact will soon operate beyond tightly controlled environments. Ensuring that these agents are not only safe but that they adhere to broader normative expectations is thus an urgent interdisciplinary challenge. Multiple fields -- notably AI Safety, AI Alignment, and Machine Ethics -- claim to contribute to this task. However, the conceptual boundaries and interrelations among these domains remain vague, leaving researchers without clear guidance in positioning their work.
To address this meta-challenge, we develop a structured conceptual framework for understanding AI alignment. Rather than focusing solely on alignment goals, we introduce a taxonomy distinguishing the alignment aim (safety, ethicality, legality, etc.), scope (outcome vs. execution), and constituency (individual vs. collective). This structural approach reveals multiple legitimate alignment configurations, providing a foundation for practical and philosophical integration across domains, and clarifying what it might mean for an agent to be aligned all-things-considered. 

**Abstract (ZH)**: Recent advances in AI研究使具有重要现实世界影响的 artificial agents 随后在受控环境之外运行的可能性越来越大。因此，确保这些agents 不仅是安全的，还符合更广泛的规范性期望，是一个急迫的跨学科挑战。多个领域——尤其是AI安全性、AI对齐和机器伦理——声称有助于这一任务。然而，这些领域的概念边界及其相互关系仍然含糊不清，使研究人员在定位其工作时缺乏清晰的指导。为了应对这一元挑战，我们制定了一个结构化的概念框架，以理解AI对齐。我们不仅关注对齐目标，还引入了一种分类法，区分对齐目标（如安全性、伦理性、合法性等）、范围（结果导向 vs. 执行导向）以及主体（个体 vs. 集体）。这种结构化方法揭示了多种合法的对齐配置，为跨领域提供了实用和哲学上的整合基础，并明确了整体来看一个agent如何对齐的含义。 

---
# Facial Foundational Model Advances Early Warning of Coronary Artery Disease from Live Videos with DigitalShadow 

**Title (ZH)**: 面部基础模型在live视频中通过DigitalShadow早期预警冠状动脉疾病 

**Authors**: Juexiao Zhou, Zhongyi Han, Mankun Xin, Xingwei He, Guotao Wang, Jiaoyan Song, Gongning Luo, Wenjia He, Xintong Li, Yuetan Chu, Juanwen Chen, Bo Wang, Xia Wu, Wenwen Duan, Zhixia Guo, Liyan Bai, Yilin Pan, Xuefei Bi, Lu Liu, Long Feng, Xiaonan He, Xin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.06283)  

**Abstract**: Global population aging presents increasing challenges to healthcare systems, with coronary artery disease (CAD) responsible for approximately 17.8 million deaths annually, making it a leading cause of global mortality. As CAD is largely preventable, early detection and proactive management are essential. In this work, we introduce DigitalShadow, an advanced early warning system for CAD, powered by a fine-tuned facial foundation model. The system is pre-trained on 21 million facial images and subsequently fine-tuned into LiveCAD, a specialized CAD risk assessment model trained on 7,004 facial images from 1,751 subjects across four hospitals in China. DigitalShadow functions passively and contactlessly, extracting facial features from live video streams without requiring active user engagement. Integrated with a personalized database, it generates natural language risk reports and individualized health recommendations. With privacy as a core design principle, DigitalShadow supports local deployment to ensure secure handling of user data. 

**Abstract (ZH)**: 全球人口老龄化对医疗卫生系统提出了不断增加的挑战，冠状动脉疾病（CAD）导致每年约1780万人死亡，使其成为全球主要死亡原因之一。由于CAD主要可以通过预防来避免，因此早期检测和主动管理至关重要。本文介绍了一种名为DigitalShadow的高级预警系统，该系统借助微调后的面部基础模型。该系统在2100万张面部图像上进行预训练，并进一步微调为LiveCAD，这是一种专门针对来自中国四家医院1751名受试者7004张面部图像的心脏病风险评估模型。DigitalShadow被动且非接触地工作，无需用户主动参与即可从实时视频流中提取面部特征。结合个性化数据库，该系统生成自然语言风险报告和个人化健康建议。在以隐私为核心设计原则的基础上，DigitalShadow支持本地部署，以确保用户数据的安全处理。 

---
# STARFlow: Scaling Latent Normalizing Flows for High-resolution Image Synthesis 

**Title (ZH)**: STARFlow: 扩展潜流形规范化模型以实现高分辨率图像合成 

**Authors**: Jiatao Gu, Tianrong Chen, David Berthelot, Huangjie Zheng, Yuyang Wang, Ruixiang Zhang, Laurent Dinh, Miguel Angel Bautista, Josh Susskind, Shuangfei Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2506.06276)  

**Abstract**: We present STARFlow, a scalable generative model based on normalizing flows that achieves strong performance in high-resolution image synthesis. The core of STARFlow is Transformer Autoregressive Flow (TARFlow), which combines the expressive power of normalizing flows with the structured modeling capabilities of Autoregressive Transformers. We first establish the theoretical universality of TARFlow for modeling continuous distributions. Building on this foundation, we introduce several key architectural and algorithmic innovations to significantly enhance scalability: (1) a deep-shallow design, wherein a deep Transformer block captures most of the model representational capacity, complemented by a few shallow Transformer blocks that are computationally efficient yet substantially beneficial; (2) modeling in the latent space of pretrained autoencoders, which proves more effective than direct pixel-level modeling; and (3) a novel guidance algorithm that significantly boosts sample quality. Crucially, our model remains an end-to-end normalizing flow, enabling exact maximum likelihood training in continuous spaces without discretization. STARFlow achieves competitive performance in both class-conditional and text-conditional image generation tasks, approaching state-of-the-art diffusion models in sample quality. To our knowledge, this work is the first successful demonstration of normalizing flows operating effectively at this scale and resolution. 

**Abstract (ZH)**: STARFlow：一种基于归一化流的可扩展生成模型，在高分辨率图像合成中表现出色 

---
# GOLFer: Smaller LM-Generated Documents Hallucination Filter & Combiner for Query Expansion in Information Retrieval 

**Title (ZH)**: GOLFer：用于信息检索中查询扩展的小型LM生成文档幻觉过滤器与组合器 

**Authors**: Lingyuan Liu, Mengxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04762)  

**Abstract**: Large language models (LLMs)-based query expansion for information retrieval augments queries with generated hypothetical documents with LLMs. However, its performance relies heavily on the scale of the language models (LMs), necessitating larger, more advanced LLMs. This approach is costly, computationally intensive, and often has limited accessibility. To address these limitations, we introduce GOLFer - Smaller LMs-Generated Documents Hallucination Filter & Combiner - a novel method leveraging smaller open-source LMs for query expansion. GOLFer comprises two modules: a hallucination filter and a documents combiner. The former detects and removes non-factual and inconsistent sentences in generated documents, a common issue with smaller LMs, while the latter combines the filtered content with the query using a weight vector to balance their influence. We evaluate GOLFer alongside dominant LLM-based query expansion methods on three web search and ten low-resource datasets. Experimental results demonstrate that GOLFer consistently outperforms other methods using smaller LMs, and maintains competitive performance against methods using large-size LLMs, demonstrating its effectiveness. 

**Abstract (ZH)**: 基于较小语言模型的生成文档 hallucination 过滤与组合的查询扩展方法：GOLFer 

---
# Low-resource Machine Translation: what for? who for? An observational study on a dedicated Tetun language translation service 

**Title (ZH)**: 低资源机器翻译：为了谁？针对什么？一种专门的图分子翻译服务的观察研究 

**Authors**: Raphael Merx, Adérito José Guterres Correia, Hanna Suominen, Ekaterina Vylomova  

**Link**: [PDF](https://arxiv.org/pdf/2411.12262)  

**Abstract**: Low-resource machine translation (MT) presents a diversity of community needs and application challenges that remain poorly understood. To complement surveys and focus groups, which tend to rely on small samples of respondents, we propose an observational study on actual usage patterns of tetun$.$org, a specialized MT service for the Tetun language, which is the lingua franca in Timor-Leste. Our analysis of 100,000 translation requests reveals patterns that challenge assumptions based on existing corpora. We find that users, many of them students on mobile devices, typically translate text from a high-resource language into Tetun across diverse domains including science, healthcare, and daily life. This contrasts sharply with available Tetun corpora, which are dominated by news articles covering government and social issues. Our results suggest that MT systems for institutionalized minority languages like Tetun should prioritize accuracy on domains relevant to educational contexts, in the high-resource to low-resource direction. More broadly, this study demonstrates how observational analysis can inform low-resource language technology development, by grounding research in practical community needs. 

**Abstract (ZH)**: 低资源机器翻译：观察tetun.org专门服务的实际使用模式及其对低资源语言技术发展的启示 

---
# MiniGPT-Reverse-Designing: Predicting Image Adjustments Utilizing MiniGPT-4 

**Title (ZH)**: MiniGPT-Reverse-设计：利用MiniGPT-4预测图像调整 

**Authors**: Vahid Azizi, Fatemeh Koochaki  

**Link**: [PDF](https://arxiv.org/pdf/2406.00971)  

**Abstract**: Vision-Language Models (VLMs) have recently seen significant advancements through integrating with Large Language Models (LLMs). The VLMs, which process image and text modalities simultaneously, have demonstrated the ability to learn and understand the interaction between images and texts across various multi-modal tasks. Reverse designing, which could be defined as a complex vision-language task, aims to predict the edits and their parameters, given a source image, an edited version, and an optional high-level textual edit description. This task requires VLMs to comprehend the interplay between the source image, the edited version, and the optional textual context simultaneously, going beyond traditional vision-language tasks. In this paper, we extend and fine-tune MiniGPT-4 for the reverse designing task. Our experiments demonstrate the extensibility of off-the-shelf VLMs, specifically MiniGPT-4, for more complex tasks such as reverse designing. Code is available at this \href{this https URL} 

**Abstract (ZH)**: Vision-Language模型（VLMs）通过与大规模语言模型（LLMs）的结合， recently saw显著进步。VLMs同时处理图像和文本模态，展示了在各种多模态任务中学习和理解图像与文本之间交互的能力。逆设计，可定义为一项复杂的视觉-语言任务，旨在给定源图像、编辑版本以及可选的高层次文本编辑描述的情况下，预测编辑和参数。这一任务要求VLMs同时理解源图像、编辑版本和可选的文本上下文之间的交互，超越了传统的视觉-语言任务。在本文中，我们扩展并微调了MiniGPT-4用于逆设计任务。我们的实验展示了现成的VLMs，特别是MiniGPT-4，适用于更复杂的任务如逆设计。代码请参见this https URL。 

---
# Dual-Modal Attention-Enhanced Text-Video Retrieval with Triplet Partial Margin Contrastive Learning 

**Title (ZH)**: 双模态注意力增强文本视频检索的三元部分边际对比学习 

**Authors**: Chen Jiang, Hong Liu, Xuzheng Yu, Qing Wang, Yuan Cheng, Jia Xu, Zhongyi Liu, Qingpei Guo, Wei Chu, Ming Yang, Yuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2309.11082)  

**Abstract**: In recent years, the explosion of web videos makes text-video retrieval increasingly essential and popular for video filtering, recommendation, and search. Text-video retrieval aims to rank relevant text/video higher than irrelevant ones. The core of this task is to precisely measure the cross-modal similarity between texts and videos. Recently, contrastive learning methods have shown promising results for text-video retrieval, most of which focus on the construction of positive and negative pairs to learn text and video representations. Nevertheless, they do not pay enough attention to hard negative pairs and lack the ability to model different levels of semantic similarity. To address these two issues, this paper improves contrastive learning using two novel techniques. First, to exploit hard examples for robust discriminative power, we propose a novel Dual-Modal Attention-Enhanced Module (DMAE) to mine hard negative pairs from textual and visual clues. By further introducing a Negative-aware InfoNCE (NegNCE) loss, we are able to adaptively identify all these hard negatives and explicitly highlight their impacts in the training loss. Second, our work argues that triplet samples can better model fine-grained semantic similarity compared to pairwise samples. We thereby present a new Triplet Partial Margin Contrastive Learning (TPM-CL) module to construct partial order triplet samples by automatically generating fine-grained hard negatives for matched text-video pairs. The proposed TPM-CL designs an adaptive token masking strategy with cross-modal interaction to model subtle semantic differences. Extensive experiments demonstrate that the proposed approach outperforms existing methods on four widely-used text-video retrieval datasets, including MSR-VTT, MSVD, DiDeMo and ActivityNet. 

**Abstract (ZH)**: 近年来，网络视频的爆炸式增长使得文本-视频检索在视频过滤、推荐和搜索中愈发重要和流行。文本-视频检索的目标是将相关文本/视频排名在不相关项之上。这一任务的核心是如何精确地度量文本与视频的跨模态相似性。近日，对比学习方法在文本-视频检索中显示出有希望的结果，大多数方法集中在构建正样本和负样本对来学习文本和视频表示。然而，这些方法未能充分关注硬负样本对，并缺乏建模不同粒度语义相似性的能力。为解决这两个问题，本文采用两种新颖的技术改进了对比学习。首先，为了利用困难样本增强鲁棒性判别能力，我们提出了一种新的双模态注意力增强模块（DMAE），从文本和视觉线索中挖掘困难负样本对。进一步引入了负样本感知的InfoNCE损失（NegNCE），能自适应地识别所有这些困难负样本并在训练损失中明确突出其影响。其次，我们提出，三元组样本比成对样本更适合建模细粒度语义相似性。因此，我们提出了一种新的三元组部分边界对比学习模块（TPM-CL），通过自动生成匹配文本-视频对的细粒度困难负样本来构造部分排序三元组样本。所提出的TPM-CL设计了一种带有跨模态交互的自适应标记掩蔽策略，以建模细微的语义差异。大量实验表明，所提出的方法在包括MSR-VTT、MSVD、DiDeMo和ActivityNet的四个广泛使用的文本-视频检索数据集中优于现有方法。 

---
