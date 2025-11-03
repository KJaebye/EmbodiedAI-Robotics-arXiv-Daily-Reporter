# Modified-Emergency Index (MEI): A Criticality Metric for Autonomous Driving in Lateral Conflict 

**Title (ZH)**: 修改后的紧急指数（MEI）：自主驾驶横向冲突的 Criticality 指标 

**Authors**: Hao Cheng, Yanbo Jiang, Qingyuan Shi, Qingwen Meng, Keyu Chen, Wenhao Yu, Jianqiang Wang, Sifa Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.27333)  

**Abstract**: Effective, reliable, and efficient evaluation of autonomous driving safety is essential to demonstrate its trustworthiness. Criticality metrics provide an objective means of assessing safety. However, as existing metrics primarily target longitudinal conflicts, accurately quantifying the risks of lateral conflicts - prevalent in urban settings - remains challenging. This paper proposes the Modified-Emergency Index (MEI), a metric designed to quantify evasive effort in lateral conflicts. Compared to the original Emergency Index (EI), MEI refines the estimation of the time available for evasive maneuvers, enabling more precise risk quantification. We validate MEI on a public lateral conflict dataset based on Argoverse-2, from which we extract over 1,500 high-quality AV conflict cases, including more than 500 critical events. MEI is then compared with the well-established ACT and the widely used PET metrics. Results show that MEI consistently outperforms them in accurately quantifying criticality and capturing risk evolution. Overall, these findings highlight MEI as a promising metric for evaluating urban conflicts and enhancing the safety assessment framework for autonomous driving. The open-source implementation is available at this https URL. 

**Abstract (ZH)**: 有效的、可靠的且高效的自主驾驶安全性评估对于展示其可信度至关重要。临界指标提供了一种客观的安全评估手段。然而，由于现有的指标主要针对纵向冲突，准确量化城市环境中常见的横向冲突的风险仍然颇具挑战性。本文提出了一种改良的紧急指数（MEI），以量化横向冲突中的避碰努力。与原始紧急指数（EI）相比，MEI 对可用于避碰操作的时间进行了精细化估计，从而能够更精确地量化风险。我们基于Argoverse-2公共横向冲突数据集验证了MEI，从中提取了超过1,500个高质量的AV冲突案例，包括超过500个关键事件。然后将MEI与成熟的ACT指标和广泛使用的PET指标进行比较。结果表明，MEI在准确量化临界性和捕捉风险演变方面始终优于它们。总体而言，这些发现突出了MEI作为评估城市冲突和增强自主驾驶安全性评估框架的有前途的指标。开源实现可在以下链接获取：this https URL。 

---
# RepV: Safety-Separable Latent Spaces for Scalable Neurosymbolic Plan Verification 

**Title (ZH)**: RepV: 安全分离的潜在空间以实现可扩展的神经符号计划验证 

**Authors**: Yunhao Yang, Neel P. Bhatt, Pranay Samineni, Rohan Siva, Zhanyang Wang, Ufuk Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26935)  

**Abstract**: As AI systems migrate to safety-critical domains, verifying that their actions comply with well-defined rules remains a challenge. Formal methods provide provable guarantees but demand hand-crafted temporal-logic specifications, offering limited expressiveness and accessibility. Deep learning approaches enable evaluation of plans against natural-language constraints, yet their opaque decision process invites misclassifications with potentially severe consequences. We introduce RepV, a neurosymbolic verifier that unifies both views by learning a latent space where safe and unsafe plans are linearly separable. Starting from a modest seed set of plans labeled by an off-the-shelf model checker, RepV trains a lightweight projector that embeds each plan, together with a language model-generated rationale, into a low-dimensional space; a frozen linear boundary then verifies compliance for unseen natural-language rules in a single forward pass.
Beyond binary classification, RepV provides a probabilistic guarantee on the likelihood of correct verification based on its position in the latent space. This guarantee enables a guarantee-driven refinement of the planner, improving rule compliance without human annotations. Empirical evaluations show that RepV improves compliance prediction accuracy by up to 15% compared to baseline methods while adding fewer than 0.2M parameters. Furthermore, our refinement framework outperforms ordinary fine-tuning baselines across various planning domains. These results show that safety-separable latent spaces offer a scalable, plug-and-play primitive for reliable neurosymbolic plan verification. Code and data are available at: this https URL. 

**Abstract (ZH)**: 随着AI系统迁移到安全关键领域，验证其行为是否符合预定义规则仍是一项挑战。形式化方法可以提供可证明的保证，但需要手工构建时间逻辑规范，这在表达能力和可访问性方面都有限。深度学习方法允许用自然语言约束评估计划，但其不透明的决策过程可能导致潜在严重后果的误分类。我们提出了RepV，这是一种神经符号验证器，通过学习一个潜在空间统一了这两种视角，在该潜在空间中，安全和不安全的计划是线性可分的。从一个现成模型检查器标注的初始计划种子集开始，RepV 训练一个轻量级的投影器，将每个计划与其语言模型生成的解释嵌入到低维空间中；然后，一个冻结的线性边界在单次前向传播中验证未见过的自然语言规则的合规性。 

---
# Cooperative Integrated Estimation-Guidance for Simultaneous Interception of Moving Targets 

**Title (ZH)**: 协同集成估测-制导以同时拦截移动目标 

**Authors**: Lohitvel Gopikannan, Shashi Ranjan Kumar, Abhinav Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2510.26948)  

**Abstract**: This paper proposes a cooperative integrated estimation-guidance framework for simultaneous interception of a non-maneuvering target using a team of unmanned autonomous vehicles, assuming only a subset of vehicles are equipped with dedicated sensors to measure the target's states. Unlike earlier approaches that focus solely on either estimation or guidance design, the proposed framework unifies both within a cooperative architecture. To circumvent the limitation posed by heterogeneity in target observability, sensorless vehicles estimate the target's state by leveraging information exchanged with neighboring agents over a directed communication topology through a prescribed-time observer. The proposed approach employs true proportional navigation guidance (TPNG), which uses an exact time-to-go formulation and is applicable across a wide spectrum of target motions. Furthermore, prescribed-time observer and controller are employed to achieve convergence to true target's state and consensus in time-to-go within set predefined times, respectively. Simulations demonstrate the effectiveness of the proposed framework under various engagement scenarios. 

**Abstract (ZH)**: 基于自主无人车辆团队的非机动目标协同综合估计与制导框架 

---
# MolChord: Structure-Sequence Alignment for Protein-Guided Drug Design 

**Title (ZH)**: MolChord: 结构-序列对齐方法在蛋白质导向药物设计中的应用 

**Authors**: Wei Zhang, Zekun Guo, Yingce Xia, Peiran Jin, Shufang Xie, Tao Qin, Xiang-Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.27671)  

**Abstract**: Structure-based drug design (SBDD), which maps target proteins to candidate molecular ligands, is a fundamental task in drug discovery. Effectively aligning protein structural representations with molecular representations, and ensuring alignment between generated drugs and their pharmacological properties, remains a critical challenge. To address these challenges, we propose MolChord, which integrates two key techniques: (1) to align protein and molecule structures with their textual descriptions and sequential representations (e.g., FASTA for proteins and SMILES for molecules), we leverage NatureLM, an autoregressive model unifying text, small molecules, and proteins, as the molecule generator, alongside a diffusion-based structure encoder; and (2) to guide molecules toward desired properties, we curate a property-aware dataset by integrating preference data and refine the alignment process using Direct Preference Optimization (DPO). Experimental results on CrossDocked2020 demonstrate that our approach achieves state-of-the-art performance on key evaluation metrics, highlighting its potential as a practical tool for SBDD. 

**Abstract (ZH)**: 基于结构的药物设计（SBDD），即将目标蛋白质映射到候选分子配体，是药物发现中的一个基本任务。有效地对蛋白质结构表示与分子表示进行对齐，并确保生成的药物与其药理学性质之间的对齐，仍然是一个关键挑战。为了解决这些挑战，我们提出MolChord，该方法整合了两个关键技术：（1）使用NatureLM，一种统一了文本、小分子和蛋白质的自回归模型，作为分子生成器，并结合基于扩散的结构编码器，以对准蛋白质和分子结构与其文本描述和序列表示（例如，蛋白质的FASTA和分子的SMILES）；（2）为了引导分子向期望的性质发展，我们通过整合偏好数据整理了一个具有性质意识的数据集，并使用直接偏好优化（DPO）细化对齐过程。CrossDocked2020上的实验结果表明，我们的方法在关键评估指标上达到了最先进的性能，突显了其作为SBDD实用工具的潜力。 

---
# VeriMoA: A Mixture-of-Agents Framework for Spec-to-HDL Generation 

**Title (ZH)**: VeriMoA：一种基于代理混合的从规格到HDL生成框架 

**Authors**: Heng Ping, Arijit Bhattacharjee, Peiyu Zhang, Shixuan Li, Wei Yang, Anzhe Cheng, Xiaole Zhang, Jesse Thomason, Ali Jannesari, Nesreen Ahmed, Paul Bogdan  

**Link**: [PDF](https://arxiv.org/pdf/2510.27617)  

**Abstract**: Automation of Register Transfer Level (RTL) design can help developers meet increasing computational demands. Large Language Models (LLMs) show promise for Hardware Description Language (HDL) generation, but face challenges due to limited parametric knowledge and domain-specific constraints. While prompt engineering and fine-tuning have limitations in knowledge coverage and training costs, multi-agent architectures offer a training-free paradigm to enhance reasoning through collaborative generation. However, current multi-agent approaches suffer from two critical deficiencies: susceptibility to noise propagation and constrained reasoning space exploration. We propose VeriMoA, a training-free mixture-of-agents (MoA) framework with two synergistic innovations. First, a quality-guided caching mechanism to maintain all intermediate HDL outputs and enables quality-based ranking and selection across the entire generation process, encouraging knowledge accumulation over layers of reasoning. Second, a multi-path generation strategy that leverages C++ and Python as intermediate representations, decomposing specification-to-HDL translation into two-stage processes that exploit LLM fluency in high-resource languages while promoting solution diversity. Comprehensive experiments on VerilogEval 2.0 and RTLLM 2.0 benchmarks demonstrate that VeriMoA achieves 15--30% improvements in Pass@1 across diverse LLM backbones, especially enabling smaller models to match larger models and fine-tuned alternatives without requiring costly training. 

**Abstract (ZH)**: 自动化注册传输级（RTL）设计可以帮助开发者满足日益增长的计算需求。大规模语言模型（LLMs）在硬件描述语言（HDL）生成中展现出潜力，但由于参数知识有限和领域特定约束，面临挑战。尽管提示工程和微调在知识覆盖和训练成本方面存在限制，多智能体架构提供了一种无需训练的范式，通过协作生成增强推理。然而，当前的多智能体方法存在两大关键缺陷：容易传播噪声和受限的推理空间探索。我们提出了VeriMoA，一种无需训练的混合多智能体（MoA）框架，并结合了两项协同创新。首先，一种质量指导的缓存机制，以保持所有中间HDL输出，并在整个生成过程中基于质量进行排名和选择，促进多层推理的知识积累。其次，一种多路径生成策略，利用C++和Python作为中间表示，将规格化到HDL的翻译分解为两个阶段的过程，利用大规模语言模型在高资源语言中的流畅性，同时促进解决方案的多样性。在VerilogEval 2.0和RTLLM 2.0基准测试上的全面实验表明，VeriMoA在不同大规模语言模型架构下实现了15-30%的Pass@1改进，特别是使较小的模型能够达到与较大模型和微调替代方案相当的表现，而无需进行昂贵的训练。 

---
# SIGMA: Search-Augmented On-Demand Knowledge Integration for Agentic Mathematical Reasoning 

**Title (ZH)**: SIGMA：增强搜索的动态知识集成以促进能动数学推理 

**Authors**: Ali Asgarov, Umid Suleymanov, Aadyant Khatri  

**Link**: [PDF](https://arxiv.org/pdf/2510.27568)  

**Abstract**: Solving mathematical reasoning problems requires not only accurate access to relevant knowledge but also careful, multi-step thinking. However, current retrieval-augmented models often rely on a single perspective, follow inflexible search strategies, and struggle to effectively combine information from multiple sources. We introduce SIGMA (Search-Augmented On-Demand Knowledge Integration for AGentic Mathematical reAsoning), a unified framework that orchestrates specialized agents to independently reason, perform targeted searches, and synthesize findings through a moderator mechanism. Each agent generates hypothetical passages to optimize retrieval for its analytic perspective, ensuring knowledge integration is both context-sensitive and computation-efficient. When evaluated on challenging benchmarks such as MATH500, AIME, and PhD-level science QA GPQA, SIGMA consistently outperforms both open- and closed-source systems, achieving an absolute performance improvement of 7.4%. Our results demonstrate that multi-agent, on-demand knowledge integration significantly enhances both reasoning accuracy and efficiency, offering a scalable approach for complex, knowledge-intensive problem-solving. We will release the code upon publication. 

**Abstract (ZH)**: 解决数学推理问题不仅需要准确访问相关知识，还需要进行仔细的多步思考。当前的检索增强模型往往依赖单一视角，遵循僵化的搜索策略，并难以有效地综合多个来源的信息。我们提出了SIGMA（搜索增强的按需知识集成以实现代理数学推理），这是一种统一框架，通过调解机制协调专门的代理分别进行推理、执行针对性搜索并综合发现结果。每个代理生成假设性段落以优化其分析视角的检索，确保知识集成既具有上下文敏感性又具有计算效率。在MATH500、AIME和博士级科学问答GPQA等具有挑战性的基准测试中，SIGMA在开放源代码和闭源系统中均表现出色，绝对性能提升7.4%。我们的结果表明，多代理、按需知识集成显著提高了推理的准确性和效率，提供了一种解决复杂、知识密集型问题的可扩展方法。论文发表后我们将发布代码。 

---
# Dialogue as Discovery: Navigating Human Intent Through Principled Inquiry 

**Title (ZH)**: 对话即发现：通过原则性的询问导航人类意图 

**Authors**: Jianwen Sun, Yukang Feng, Yifan Chang, Chuanhao Li, Zizhen Li, Jiaxin Ai, Fanrui Zhang, Yu Dai, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27410)  

**Abstract**: A fundamental bottleneck in human-AI collaboration is the "intention expression gap," the difficulty for humans to effectively convey complex, high-dimensional thoughts to AI. This challenge often traps users in inefficient trial-and-error loops and is exacerbated by the diverse expertise levels of users. We reframe this problem from passive instruction following to a Socratic collaboration paradigm, proposing an agent that actively probes for information to resolve its uncertainty about user intent. we name the proposed agent Nous, trained to acquire proficiency in this inquiry policy. The core mechanism of Nous is a training framework grounded in the first principles of information theory. Within this framework, we define the information gain from dialogue as an intrinsic reward signal, which is fundamentally equivalent to the reduction of Shannon entropy over a structured task space. This reward design enables us to avoid reliance on costly human preference annotations or external reward models. To validate our framework, we develop an automated simulation pipeline to generate a large-scale, preference-based dataset for the challenging task of scientific diagram generation. Comprehensive experiments, including ablations, subjective and objective evaluations, and tests across user expertise levels, demonstrate the effectiveness of our proposed framework. Nous achieves leading efficiency and output quality, while remaining robust to varying user expertise. Moreover, its design is domain-agnostic, and we show evidence of generalization beyond diagram generation. Experimental results prove that our work offers a principled, scalable, and adaptive paradigm for resolving uncertainty about user intent in complex human-AI collaboration. 

**Abstract (ZH)**: 人类与AI协作中的一个根本瓶颈是“意图表达差距”，即人类难以有效地将复杂的、高维度的思想传达给AI。这一挑战通常将用户困在低效的试错循环中，并因用户专业知识水平的多样性而加剧。我们从被动的指令遵循重新定义这一问题，转向苏格拉底式协作范式，提出一个能够主动探询信息、解决其对用户意图不确定性的代理。我们称这个提出的代理为Nous，并训练其掌握这种查询策略。Nous的核心机制是基于信息论基本原理的训练框架，在这个框架中，我们将对话的信息增益定义为内在的奖励信号，本质上等同于在结构化任务空间中香农熵的减少。这一奖励设计使我们能够避免依赖昂贵的人类偏好注释或外部奖励模型。为了验证我们的框架，我们开发了一个自动化的仿真管道，用于生成大规模的基于偏好的数据集，用于科学图表生成这一具有挑战性的任务。包括消融实验、主观评价、客观评价和跨用户专业知识水平的测试在内的全面实验显示了我们提出的框架的有效性。Nous在效率和输出质量方面达到了领先水平，并且在不同的用户专业知识水平下保持了鲁棒性。此外，其设计具有领域通用性，并展示了其在图表生成之外的泛化能力。实验结果证明，我们的工作提供了一种解决复杂人类与AI协作中用户意图不确定性问题的原则性、可扩展和适应性范式。 

---
# Discriminative Rule Learning for Outcome-Guided Process Model Discovery 

**Title (ZH)**: 基于结果导向的过程模型发现的区分性规则学习 

**Authors**: Ali Norouzifar, Wil van der Aalst  

**Link**: [PDF](https://arxiv.org/pdf/2510.27343)  

**Abstract**: Event logs extracted from information systems offer a rich foundation for understanding and improving business processes. In many real-world applications, it is possible to distinguish between desirable and undesirable process executions, where desirable traces reflect efficient or compliant behavior, and undesirable ones may involve inefficiencies, rule violations, delays, or resource waste. This distinction presents an opportunity to guide process discovery in a more outcome-aware manner. Discovering a single process model without considering outcomes can yield representations poorly suited for conformance checking and performance analysis, as they fail to capture critical behavioral differences. Moreover, prioritizing one behavior over the other may obscure structural distinctions vital for understanding process outcomes. By learning interpretable discriminative rules over control-flow features, we group traces with similar desirability profiles and apply process discovery separately within each group. This results in focused and interpretable models that reveal the drivers of both desirable and undesirable executions. The approach is implemented as a publicly available tool and it is evaluated on multiple real-life event logs, demonstrating its effectiveness in isolating and visualizing critical process patterns. 

**Abstract (ZH)**: 来自信息系统提取的事件日志为理解和改进业务流程提供了丰富的基础。在许多实际应用中，可以区分期望和非期望的流程执行，其中期望的痕迹反映了高效或合规的行为，而不期望的痕迹可能涉及低效率、规则违反、延迟或资源浪费。这种区分为以更注重结果的方式指导流程发现提供了机会。不考虑结果发现单一的流程模型可能会导致不合适的表示，因为它们未能捕捉到关键的行为差异。此外，优先考虑一种行为而忽视另一种行为可能会模糊对理解流程结果至关重要的结构差异。通过学习控制流特征上的可解释区分规则，我们将具有相似期望性概况的痕迹分组，并在每个组内独立地进行流程发现。这导致了专注于结果并具有解释性的模型，揭示了期望和非期望执行的驱动因素。该方法作为开源工具实现，并在多个实际事件日志上进行评估，证明了其在隔离和可视化关键流程模式方面的有效性。 

---
# From product to system network challenges in system of systems lifecycle management 

**Title (ZH)**: 从产品到系统网络在系统体系生命周期管理中的挑战 

**Authors**: Vahid Salehi, Josef Vilsmeier, Shirui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27194)  

**Abstract**: Today, products are no longer isolated artifacts, but nodes in networked systems. This means that traditional, linearly conceived life cycle models are reaching their limits: Interoperability across disciplines, variant and configuration management, traceability, and governance across organizational boundaries are becoming key factors. This collective contribution classifies the state of the art and proposes a practical frame of reference for SoS lifecycle management, model-based systems engineering (MBSE) as the semantic backbone, product lifecycle management (PLM) as the governance and configuration level, CAD-CAE as model-derived domains, and digital thread and digital twin as continuous feedback. Based on current literature and industry experience, mobility, healthcare, and the public sector, we identify four principles: (1) referenced architecture and data models, (2) end-to-end configuration sovereignty instead of tool silos, (3) curated models with clear review gates, and (4) measurable value contributions along time, quality, cost, and sustainability. A three-step roadmap shows the transition from product- to network- centric development: piloting with reference architecture, scaling across variant and supply chain spaces, organizational anchoring (roles, training, compliance). The results are increased change robustness, shorter throughput times, improved reuse, and informed sustainability decisions. This article is aimed at decision-makers and practitioners who want to make complexity manageable and design SoS value streams to be scalable. 

**Abstract (ZH)**: 当今的产品不再是孤立的实体，而是网络系统中的节点。这表明传统的线性生命周期模型已达到其极限：跨学科的互操作性、变体和配置管理、可追溯性以及跨越组织边界的治理已成为关键因素。本文综述了该领域的现状，并基于基于模型的系统工程（MBSE）语义骨干、产品生命周期管理（PLM）的治理和配置层级、CAD-CAE的模型派生领域、以及数字主线和数字孪生的持续反馈，提出了一种实际的框架参考，适用于SoS生命周期管理。根据当前文献和行业经验，我们确定了四个原则：（1）引用的架构和数据模型，（2）端到端的配置主权而非工具孤岛，（3）经过精心筛选的模型，并设有明确的评审关卡，（4）按照时间、质量、成本和可持续性衡量的价值贡献。三个阶段的路线图展示了从产品导向到网络导向的发展转变：以参考架构为试点，跨不同变体和供应链空间进行扩展，组织嵌入（角色、培训、合规性）。结果包括增强的变更稳健性、缩短的流动时间、提高的重用性以及知情的可持续性决策。本文旨在为希望管理复杂性和设计可扩展SoS价值流的决策者和实践者提供指导。 

---
# Adaptive Data Flywheel: Applying MAPE Control Loops to AI Agent Improvement 

**Title (ZH)**: 自适应数据飞轮：将MAPE控制环应用于AI代理改进 

**Authors**: Aaditya Shukla, Sidney Knowles, Meenakshi Madugula, Dave Farris, Ryan Angilly, Santiago Pombo, Anbang Xu, Lu An, Abhinav Balasubramanian, Tan Yu, Jiaxiang Ren, Rama Akkiraju  

**Link**: [PDF](https://arxiv.org/pdf/2510.27051)  

**Abstract**: Enterprise AI agents must continuously adapt to maintain accuracy, reduce latency, and remain aligned with user needs. We present a practical implementation of a data flywheel in NVInfo AI, NVIDIA's Mixture-of-Experts (MoE) Knowledge Assistant serving over 30,000 employees. By operationalizing a MAPE-driven data flywheel, we built a closed-loop system that systematically addresses failures in retrieval-augmented generation (RAG) pipelines and enables continuous learning. Over a 3-month post-deployment period, we monitored feedback and collected 495 negative samples. Analysis revealed two major failure modes: routing errors (5.25\%) and query rephrasal errors (3.2\%). Using NVIDIA NeMo microservices, we implemented targeted improvements through fine-tuning. For routing, we replaced a Llama 3.1 70B model with a fine-tuned 8B variant, achieving 96\% accuracy, a 10x reduction in model size, and 70\% latency improvement. For query rephrasal, fine-tuning yielded a 3.7\% gain in accuracy and a 40\% latency reduction. Our approach demonstrates how human-in-the-loop (HITL) feedback, when structured within a data flywheel, transforms enterprise AI agents into self-improving systems. Key learnings include approaches to ensure agent robustness despite limited user feedback, navigating privacy constraints, and executing staged rollouts in production. This work offers a repeatable blueprint for building robust, adaptive enterprise AI agents capable of learning from real-world usage at scale. 

**Abstract (ZH)**: 企业AI代理必须持续适应以维持准确性、降低延迟并保持与用户需求一致。我们提出了NVInfo AI中的一种实用数据飞轮实现，这是一种运行在NVIDIA Mixture-of-Experts（MoE）知识助手上的系统，该助手服务于超过30,000名员工。通过在MAPE驱动的数据飞轮中实现这一目标，我们构建了一个闭环系统，系统性地解决了检索增强生成（RAG）管道中的故障，并实现了持续学习。在部署后的3个月内，我们监测反馈并收集了495个负样本。分析显示了两种主要的故障模式：路由错误（5.25%）和查询重表述错误（3.2%）。利用NVIDIA NeMo微服务，我们通过微调实现了有针对性的改进。对于路由问题，我们将一个Llama 3.1 70B模型替换为一个微调后的8B变体，实现了96%的准确率，模型大小减少了10倍，延迟降低了70%。对于查询重表述问题，微调带来了3.7%的准确率提升和40%的延迟减少。我们的方法展示了如何通过数据飞轮中的环路反馈机制，将人类在环（HITL）反馈转化为自我改进的企业级AI代理系统。关键经验教训包括确保代理稳健性的方法，尽管用户反馈有限；处理隐私限制的方法；以及在生产环境中执行分阶段滚动部署的方法。这项工作为企业级AI代理构建可重复的稳健、自适应系统提供了蓝图，这些系统能够大规模地从实际使用中学习。 

---
# e1: Learning Adaptive Control of Reasoning Effort 

**Title (ZH)**: 学习自适应推理努力控制 

**Authors**: Michael Kleinman, Matthew Trager, Alessandro Achille, Wei Xia, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2510.27042)  

**Abstract**: Increasing the thinking budget of AI models can significantly improve accuracy, but not all questions warrant the same amount of reasoning. Users may prefer to allocate different amounts of reasoning effort depending on how they value output quality versus latency and cost. To leverage this tradeoff effectively, users need fine-grained control over the amount of thinking used for a particular query, but few approaches enable such control. Existing methods require users to specify the absolute number of desired tokens, but this requires knowing the difficulty of the problem beforehand to appropriately set the token budget for a query. To address these issues, we propose Adaptive Effort Control, a self-adaptive reinforcement learning method that trains models to use a user-specified fraction of tokens relative to the current average chain-of-thought length for each query. This approach eliminates dataset- and phase-specific tuning while producing better cost-accuracy tradeoff curves compared to standard methods. Users can dynamically adjust the cost-accuracy trade-off through a continuous effort parameter specified at inference time. We observe that the model automatically learns to allocate resources proportionally to the task difficulty and, across model scales ranging from 1.5B to 32B parameters, our approach enables approximately 3x reduction in chain-of-thought length while maintaining or improving performance relative to the base model used for RL training. 

**Abstract (ZH)**: 自适应努力控制：一种基于自适应强化学习的细粒度计算控制方法 

---
# Causal Masking on Spatial Data: An Information-Theoretic Case for Learning Spatial Datasets with Unimodal Language Models 

**Title (ZH)**: 基于因果掩蔽的空间数据：使用单模语言模型学习空间数据的信息论案例 

**Authors**: Jared Junkin, Samuel Nathanson  

**Link**: [PDF](https://arxiv.org/pdf/2510.27009)  

**Abstract**: Language models are traditionally designed around causal masking. In domains with spatial or relational structure, causal masking is often viewed as inappropriate, and sequential linearizations are instead used. Yet the question of whether it is viable to accept the information loss introduced by causal masking on nonsequential data has received little direct study, in part because few domains offer both spatial and sequential representations of the same dataset. In this work, we investigate this issue in the domain of chess, which naturally supports both representations. We train language models with bidirectional and causal self-attention mechanisms on both spatial (board-based) and sequential (move-based) data. Our results show that models trained on spatial board states - \textit{even with causal masking} - consistently achieve stronger playing strength than models trained on sequential data. While our experiments are conducted on chess, our results are methodological and may have broader implications: applying causal masking to spatial data is a viable procedure for training unimodal LLMs on spatial data, and in some domains is even preferable to sequentialization. 

**Abstract (ZH)**: 语言模型传统上围绕因果掩码设计。在具有空间或关系结构的领域中，因果掩码通常被视为不恰当的，因此使用序列线性化代替。然而，关于因果掩码在非序列数据中引入信息损失的可接受性在很大程度上没有直接研究，部分原因是很少有领域同时提供空间和序列表示的数据集。在本研究中，我们在支持这两种表示方式的象棋领域中探讨了这一问题。我们使用双向和因果注意力机制分别在基于棋盘的空间数据和基于移动的序列数据上训练语言模型。我们发现，即使使用因果掩码，在空间棋盘状态上训练的模型——其表现 consistently 比在序列数据上训练的模型更强。虽然我们的实验是在象棋上进行的，但我们的结果具有方法论意义，并可能具有更广泛的影响：对空间数据训练单模LLM时，使用因果掩码是可行的程序，甚至在某些领域中，其效果优于序列化。 

---
# SUSTAINABLE Platform: Seamless Smart Farming Integration Towards Agronomy Automation 

**Title (ZH)**: 可持续平台：向农作自动化无缝集成的智能 farming 解决方案 

**Authors**: Agorakis Bompotas, Konstantinos Koutras, Nikitas Rigas Kalogeropoulos, Panagiotis Kechagias, Dimitra Gariza, Athanasios P. Kalogeras, Christos Alexakos  

**Link**: [PDF](https://arxiv.org/pdf/2510.26989)  

**Abstract**: The global agricultural sector is undergoing a transformative shift, driven by increasing food demands, climate variability and the need for sustainable practices. SUSTAINABLE is a smart farming platform designed to integrate IoT, AI, satellite imaging, and role-based task orchestration to enable efficient, traceable, and sustainable agriculture with a pilot usecase in viticulture. This paper explores current smart agriculture solutions, presents a comparative evaluation, and introduces SUSTAINABLE's key features, including satellite index integration, real-time environmental data, and role-aware task management tailored to Mediterranean vineyards. 

**Abstract (ZH)**: 全球农业领域正经历一场转型变革，受到不断增长的粮食需求、气候变化以及可持续实践的需要的驱动。SUSTAINABLE是一个智能 farming 平台，旨在集成物联网、人工智能、卫星成像和基于角色的任务编排，以实现高效、可追溯和可持续的农业，并在葡萄种植业中进行了试点应用。本文探讨了当前的智能农业解决方案，进行了比较评估，并介绍了SUSTAINABLE的关键功能，包括卫星指数整合、实时环境数据和针对地中海葡萄园的角色感知任务管理。 

---
# Inverse Knowledge Search over Verifiable Reasoning: Synthesizing a Scientific Encyclopedia from a Long Chains-of-Thought Knowledge Base 

**Title (ZH)**: 可验证推理中的逆向知识搜索：从长链条思考知识库合成科学百科全书 

**Authors**: Yu Li, Yuan Huang, Tao Wang, Caiyu Fan, Xiansheng Cai, Sihan Hu, Xinzijian Liu, Cheng Shi, Mingjun Xu, Zhen Wang, Yan Wang, Xiangqi Jin, Tianhan Zhang, Linfeng Zhang, Lei Wang, Youjin Deng, Pan Zhang, Weijie Sun, Xingyu Li, Weinan E, Linfeng Zhang, Zhiyuan Yao, Kun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.26854)  

**Abstract**: Most scientific materials compress reasoning, presenting conclusions while omitting the derivational chains that justify them. This compression hinders verification by lacking explicit, step-wise justifications and inhibits cross-domain links by collapsing the very pathways that establish the logical and causal connections between concepts. We introduce a scalable framework that decompresses scientific reasoning, constructing a verifiable Long Chain-of-Thought (LCoT) knowledge base and projecting it into an emergent encyclopedia, SciencePedia. Our pipeline operationalizes an endpoint-driven, reductionist strategy: a Socratic agent, guided by a curriculum of around 200 courses, generates approximately 3 million first-principles questions. To ensure high fidelity, multiple independent solver models generate LCoTs, which are then rigorously filtered by prompt sanitization and cross-model answer consensus, retaining only those with verifiable endpoints. This verified corpus powers the Brainstorm Search Engine, which performs inverse knowledge search -- retrieving diverse, first-principles derivations that culminate in a target concept. This engine, in turn, feeds the Plato synthesizer, which narrates these verified chains into coherent articles. The initial SciencePedia comprises approximately 200,000 fine-grained entries spanning mathematics, physics, chemistry, biology, engineering, and computation. In evaluations across six disciplines, Plato-synthesized articles (conditioned on retrieved LCoTs) exhibit substantially higher knowledge-point density and significantly lower factual error rates than an equally-prompted baseline without retrieval (as judged by an external LLM). Built on this verifiable LCoT knowledge base, this reasoning-centric approach enables trustworthy, cross-domain scientific synthesis at scale and establishes the foundation for an ever-expanding encyclopedia. 

**Abstract (ZH)**: 一种可扩展的框架：分解科学推理，构建可验证的长推理链知识库并投影至科学百宗百科 

---
# Challenges in Credit Assignment for Multi-Agent Reinforcement Learning in Open Agent Systems 

**Title (ZH)**: 开放代理系统中多智能体强化学习中的信用分配挑战 

**Authors**: Alireza Saleh Abadi, Leen-Kiat Soh  

**Link**: [PDF](https://arxiv.org/pdf/2510.27659)  

**Abstract**: In the rapidly evolving field of multi-agent reinforcement learning (MARL), understanding the dynamics of open systems is crucial. Openness in MARL refers to the dynam-ic nature of agent populations, tasks, and agent types with-in a system. Specifically, there are three types of openness as reported in (Eck et al. 2023) [2]: agent openness, where agents can enter or leave the system at any time; task openness, where new tasks emerge, and existing ones evolve or disappear; and type openness, where the capabil-ities and behaviors of agents change over time. This report provides a conceptual and empirical review, focusing on the interplay between openness and the credit assignment problem (CAP). CAP involves determining the contribution of individual agents to the overall system performance, a task that becomes increasingly complex in open environ-ments. Traditional credit assignment (CA) methods often assume static agent populations, fixed and pre-defined tasks, and stationary types, making them inadequate for open systems. We first conduct a conceptual analysis, in-troducing new sub-categories of openness to detail how events like agent turnover or task cancellation break the assumptions of environmental stationarity and fixed team composition that underpin existing CAP methods. We then present an empirical study using representative temporal and structural algorithms in an open environment. The results demonstrate that openness directly causes credit misattribution, evidenced by unstable loss functions and significant performance degradation. 

**Abstract (ZH)**: 多代理强化学习（MARL）领域中的开放系统动态研究：开放性与信用分配问题的关系 

---
# Community Detection on Model Explanation Graphs for Explainable AI 

**Title (ZH)**: 模型解释图上的社区检测 for 可解释人工智能 

**Authors**: Ehsan Moradi  

**Link**: [PDF](https://arxiv.org/pdf/2510.27655)  

**Abstract**: Feature-attribution methods (e.g., SHAP, LIME) explain individual predictions but often miss higher-order structure: sets of features that act in concert. We propose Modules of Influence (MoI), a framework that (i) constructs a model explanation graph from per-instance attributions, (ii) applies community detection to find feature modules that jointly affect predictions, and (iii) quantifies how these modules relate to bias, redundancy, and causality patterns. Across synthetic and real datasets, MoI uncovers correlated feature groups, improves model debugging via module-level ablations, and localizes bias exposure to specific modules. We release stability and synergy metrics, a reference implementation, and evaluation protocols to benchmark module discovery in XAI. 

**Abstract (ZH)**: 基于模块的影响特征归因方法（MoI）：构建模型解释图、发现特征模块并量化其与偏见、冗余和因果关系模式的相关性 

---
# Best Practices for Biorisk Evaluations on Open-Weight Bio-Foundation Models 

**Title (ZH)**: 开放源代码生物基础模型的生物风险评估最佳实践 

**Authors**: Boyi Wei, Zora Che, Nathaniel Li, Udari Madhushani Sehwag, Jasper Götting, Samira Nedungadi, Julian Michael, Summer Yue, Dan Hendrycks, Peter Henderson, Zifan Wang, Seth Donoughe, Mantas Mazeika  

**Link**: [PDF](https://arxiv.org/pdf/2510.27629)  

**Abstract**: Open-weight bio-foundation models present a dual-use dilemma. While holding great promise for accelerating scientific research and drug development, they could also enable bad actors to develop more deadly bioweapons. To mitigate the risk posed by these models, current approaches focus on filtering biohazardous data during pre-training. However, the effectiveness of such an approach remains unclear, particularly against determined actors who might fine-tune these models for malicious use. To address this gap, we propose \eval, a framework to evaluate the robustness of procedures that are intended to reduce the dual-use capabilities of bio-foundation models. \eval assesses models' virus understanding through three lenses, including sequence modeling, mutational effects prediction, and virulence prediction. Our results show that current filtering practices may not be particularly effective: Excluded knowledge can be rapidly recovered in some cases via fine-tuning, and exhibits broader generalizability in sequence modeling. Furthermore, dual-use signals may already reside in the pretrained representations, and can be elicited via simple linear probing. These findings highlight the challenges of data filtering as a standalone procedure, underscoring the need for further research into robust safety and security strategies for open-weight bio-foundation models. 

**Abstract (ZH)**: 开放权重生物基础模型存在双重用途难题。尽管这些模型有望加速科学研究和药物开发，但它们也可能被恶意行为者用于开发更具杀伤力的生物武器。为减轻这些模型带来的风险，当前的方法主要集中在预训练过程中过滤生物危害数据。然而，这种方法的有效性尚不明确，尤其对于那些可能精细调整这些模型用于恶意用途的坚定行为者。为填补这一空白，我们提出了一种名为\eval的框架，用于评估旨在降低生物基础模型双重用途能力的程序的鲁棒性。\eval通过序列建模、突变效应预测和致病性预测三个视角评估模型对病毒的理解。我们的结果显示，当前的过滤实践可能并不特别有效：在某些情况下，通过精细调整可以迅速恢复被排除的知识，并且在序列建模中表现出更广泛的泛化能力。此外，双重用途信号可能已经存在于预训练表示中，并可以通过简单的线性探测被引发。这些发现突显了单独依赖数据过滤的挑战，强调了对未来开放权重生物基础模型稳健的安全与安全策略研究的迫切需求。 

---
# Towards Universal Video Retrieval: Generalizing Video Embedding via Synthesized Multimodal Pyramid Curriculum 

**Title (ZH)**: 面向通用视频检索：通过合成多模态分层课程泛化视频嵌入 

**Authors**: Zhuoning Guo, Mingxin Li, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27571)  

**Abstract**: The prevailing video retrieval paradigm is structurally misaligned, as narrow benchmarks incentivize correspondingly limited data and single-task training. Therefore, universal capability is suppressed due to the absence of a diagnostic evaluation that defines and demands multi-dimensional generalization. To break this cycle, we introduce a framework built on the co-design of evaluation, data, and modeling. First, we establish the Universal Video Retrieval Benchmark (UVRB), a suite of 16 datasets designed not only to measure performance but also to diagnose critical capability gaps across tasks and domains. Second, guided by UVRB's diagnostics, we introduce a scalable synthesis workflow that generates 1.55 million high-quality pairs to populate the semantic space required for universality. Finally, we devise the Modality Pyramid, a curriculum that trains our General Video Embedder (GVE) by explicitly leveraging the latent interconnections within our diverse data. Extensive experiments show GVE achieves state-of-the-art zero-shot generalization on UVRB. In particular, our analysis reveals that popular benchmarks are poor predictors of general ability and that partially relevant retrieval is a dominant but overlooked scenario. Overall, our co-designed framework provides a practical path to escape the limited scope and advance toward truly universal video retrieval. 

**Abstract (ZH)**: 普遍视频检索范式结构上存在偏差，窄化的基准激励对应受限的数据和单任务训练。因此，由于缺乏诊断性评估来定义和要求多维度通用性，普遍能力受到抑制。为打破这一循环，我们提出了一种基于评估、数据和建模联合设计的框架。首先，我们建立了通用视频检索基准（UVRB），一套包含16个数据集，不仅用于衡量性能，还用于诊断任务和领域间的关键能力差距。其次，根据UVRB的诊断结果，我们引入了一个可扩展的合成工作流，生成155万个高质量的数据对，以填充实现通用性所需的语义空间。最后，我们设计了模态金字塔，通过明确利用我们多样化数据中的潜在关联来训练我们的通用视频嵌入器（GVE）。 extensive 实验表明，GVE 在 UVRB 上实现了最先进的零样本泛化能力。特别地，我们的分析显示流行基准对于预测通用能力效果不佳，部分相关检索是一种占主导但被忽视的场景。总体而言，我们联合设计的框架提供了一条实用的道路，以超越有限范围并朝着真正具有通用性的视频检索迈进。 

---
# CodeAlignBench: Assessing Code Generation Models on Developer-Preferred Code Adjustments 

**Title (ZH)**: CodeAlignBench: 评估代码生成模型在开发人员优选代码调整方面的表现 

**Authors**: Forough Mehralian, Ryan Shar, James R. Rae, Alireza Hashemi  

**Link**: [PDF](https://arxiv.org/pdf/2510.27565)  

**Abstract**: As large language models become increasingly capable of generating code, evaluating their performance remains a complex and evolving challenge. Existing benchmarks primarily focus on functional correctness, overlooking the diversity of real-world coding tasks and developer expectations. To this end, we introduce a multi-language benchmark that evaluates LLM instruction-following capabilities and is extensible to operate on any set of standalone coding problems. Our benchmark evaluates instruction following in two key settings: adherence to pre-defined constraints specified with the initial problem, and the ability to perform refinements based on follow-up instructions. For this paper's analysis, we empirically evaluated our benchmarking pipeline with programming tasks from LiveBench, that are also automatically translated from Python into Java and JavaScript. Our automated benchmark reveals that models exhibit differing levels of performance across multiple dimensions of instruction-following. Our benchmarking pipeline provides a more comprehensive evaluation of code generation models, highlighting their strengths and limitations across languages and generation goals. 

**Abstract (ZH)**: 随着大型语言模型在生成代码方面的能力不断增强，评估其性能仍然是一项复杂且不断发展的挑战。现有的基准测试主要侧重于功能正确性，忽视了现实世界编码任务的多样性和开发者期望。为此，我们引入了一个多语言基准测试，评估LLM的指令跟随能力，并且可以扩展到任何独立编码问题的集合。我们的基准测试在两个关键设置下评估指令跟随能力：遵守初始问题中预定义的约束，以及根据后续指令进行改进的能力。对于本文的分析，我们使用LiveBench中的编程任务来实证评估基准测试流程，这些任务还可以自动从Python翻译成Java和JavaScript。我们的自动化基准测试显示，模型在指令跟随的不同维度上表现出不同的性能水平。我们的基准测试流程提供了对代码生成模型更为全面的评估，突显了它们在多种语言和生成目标下的优缺点。 

---
# Sybil-Resistant Service Discovery for Agent Economies 

**Title (ZH)**: Sybil攻击抵御的服务发现机制在代理经济中 

**Authors**: David Shi, Kevin Joo  

**Link**: [PDF](https://arxiv.org/pdf/2510.27554)  

**Abstract**: x402 enables Hypertext Transfer Protocol (HTTP) services like application programming interfaces (APIs), data feeds, and inference providers to accept cryptocurrency payments for access. As agents increasingly consume these services, discovery becomes critical: which swap interface should an agent trust? Which data provider is the most reliable? We introduce TraceRank, a reputation-weighted ranking algorithm where payment transactions serve as endorsements. TraceRank seeds addresses with precomputed reputation metrics and propagates reputation through payment flows weighted by transaction value and temporal recency. Applied to x402's payment graph, this surfaces services preferred by high-reputation users rather than those with high transaction volume. Our system combines TraceRank with semantic search to respond to natural language queries with high quality results. We argue that reputation propagation resists Sybil attacks by making spam services with many low-reputation payers rank below legitimate services with few high-reputation payers. Ultimately, we aim to construct a search method for x402 enabled services that avoids infrastructure bias and has better performance than purely volume based or semantic methods. 

**Abstract (ZH)**: x402使Hyper文本传输协议（HTTP）服务如应用程序编程接口（APIs）、数据流和推理提供商能够接受加密货币支付以获取访问权限。随着代理越来越多地消费这些服务，发现变得至关重要：代理应该信任哪一个兑换接口？哪个数据提供商最为可靠？我们引入了TraceRank，这是一种基于声誉加权的排名算法，其中支付交易作为推荐。TraceRank通过支付流传播由交易价值和时间最近性加权的声誉，并预先计算地址的声誉度量指标进行初始化。应用于x402的支付图，这会呈现由高声誉用户偏好的服务，而不是高交易量的服务。我们的系统结合了TraceRank和语义搜索，以用高质量的结果应答自然语言查询。我们认为，声誉传播能够抵御Sybil攻击，因为众多低声誉支付者的仿冒服务排名会低于少数高声誉支付者的合法服务。最终，我们旨在构建一种避免基础设施偏见且性能优于基于纯粹交易量或语义方法的x402使能服务的搜索方法。 

---
# Leveraging Generic Time Series Foundation Models for EEG Classification 

**Title (ZH)**: 利用通用时间序列基础模型进行EEG分类 

**Authors**: Théo Gnassounou, Yessin Moakher, Shifeng Xie, Vasilii Feofanov, Ievgen Redko  

**Link**: [PDF](https://arxiv.org/pdf/2510.27522)  

**Abstract**: Foundation models for time series are emerging as powerful general-purpose backbones, yet their potential for domain-specific biomedical signals such as electroencephalography (EEG) remains rather unexplored. In this work, we investigate the applicability a recently proposed time series classification foundation model, to a different EEG tasks such as motor imagery classification and sleep stage prediction. We test two pretraining regimes: (a) pretraining on heterogeneous real-world time series from multiple domains, and (b) pretraining on purely synthetic data. We find that both variants yield strong performance, consistently outperforming EEGNet, a widely used convolutional baseline, and CBraMod, the most recent EEG-specific foundation model. These results suggest that generalist time series foundation models, even when pretrained on data of non-neural origin or on synthetic signals, can transfer effectively to EEG. Our findings highlight the promise of leveraging cross-domain pretrained models for brain signal analysis, suggesting that EEG may benefit from advances in the broader time series literature. 

**Abstract (ZH)**: 基于时间序列的基础模型正逐步成为强有力的通用骨干模型，但在如脑电图（EEG）等特定生物医学信号领域的应用潜力尚未被充分探索。在本文中，我们研究了一种 recently 提出的时间序列分类基础模型在不同EEG任务中的适用性，如运动想象分类和睡眠阶段预测。我们测试了两种预训练方式：（a）基于多个领域的真实世界异构时间序列数据进行预训练；（b）基于纯合成数据进行预训练。我们发现这两种方式均表现优异，一致性地超过了广泛使用的卷积基线EEGNet和最新的特定于EEG的基础模型CBraMod。这些结果表明，即使基于非神经源数据或合成信号进行预训练，通用时间序列基础模型也能有效转移应用于EEG。我们的研究结果强调了交叉领域预训练模型在脑电信号分析中的潜力，表明EEG可能从更广泛的时间序列文献进展中受益。 

---
# DP-FedPGN: Finding Global Flat Minima for Differentially Private Federated Learning via Penalizing Gradient Norm 

**Title (ZH)**: DP-FedPGN: 通过惩罚梯度范数寻找差分隐私联邦学习中的全局平坦最小值 

**Authors**: Junkang Liu, Yuxuan Tian, Fanhua Shang, Yuanyuan Liu, Hongying Liu, Junchao Zhou, Daorui Ding  

**Link**: [PDF](https://arxiv.org/pdf/2510.27504)  

**Abstract**: To prevent inference attacks in Federated Learning (FL) and reduce the leakage of sensitive information, Client-level Differentially Private Federated Learning (CL-DPFL) is widely used. However, current CL-DPFL methods usually result in sharper loss landscapes, which leads to a decrease in model generalization after differential privacy protection. By using Sharpness Aware Minimization (SAM), the current popular federated learning methods are to find a local flat minimum value to alleviate this problem. However, the local flatness may not reflect the global flatness in CL-DPFL. Therefore, to address this issue and seek global flat minima of models, we propose a new CL-DPFL algorithm, DP-FedPGN, in which we introduce a global gradient norm penalty to the local loss to find the global flat minimum. Moreover, by using our global gradient norm penalty, we not only find a flatter global minimum but also reduce the locally updated norm, which means that we further reduce the error of gradient clipping. From a theoretical perspective, we analyze how DP-FedPGN mitigates the performance degradation caused by DP. Meanwhile, the proposed DP-FedPGN algorithm eliminates the impact of data heterogeneity and achieves fast convergence. We also use Rényi DP to provide strict privacy guarantees and provide sensitivity analysis for local updates. Finally, we conduct effectiveness tests on both ResNet and Transformer models, and achieve significant improvements in six visual and natural language processing tasks compared to existing state-of-the-art algorithms. The code is available at this https URL 

**Abstract (ZH)**: 防止联邦学习中推理攻击并减少敏感信息泄露的客户端差分隐私联邦学习：通过全局梯度范数惩罚寻找全局平坦极小值 

---
# Who Does Your Algorithm Fail? Investigating Age and Ethnic Bias in the MAMA-MIA Dataset 

**Title (ZH)**: 你的算法对哪些人群失效？MAMA-MIA数据集中的年龄和种族偏见探究 

**Authors**: Aditya Parikh, Sneha Das, Aasa Feragen  

**Link**: [PDF](https://arxiv.org/pdf/2510.27421)  

**Abstract**: Deep learning models aim to improve diagnostic workflows, but fairness evaluation remains underexplored beyond classification, e.g., in image segmentation. Unaddressed segmentation bias can lead to disparities in the quality of care for certain populations, potentially compounded across clinical decision points and amplified through iterative model development. Here, we audit the fairness of the automated segmentation labels provided in the breast cancer tumor segmentation dataset MAMA-MIA. We evaluate automated segmentation quality across age, ethnicity, and data source. Our analysis reveals an intrinsic age-related bias against younger patients that continues to persist even after controlling for confounding factors, such as data source. We hypothesize that this bias may be linked to physiological factors, a known challenge for both radiologists and automated systems. Finally, we show how aggregating data from multiple data sources influences site-specific ethnic biases, underscoring the necessity of investigating data at a granular level. 

**Abstract (ZH)**: 深度学习模型旨在改进诊断流程，但公平性评估在分割等任务中仍被忽视。未解决的分割偏差可能导致某些人群的医疗服务质量差距，这种差距可能在临床决策点上加剧，并在迭代模型发展中放大。在这里，我们审查了MAMA-MIA乳腺癌肿瘤分割数据集中自动分割标签的公平性。我们评估了不同年龄、种族和地区来源的数据集中的自动分割质量。分析结果显示，即使在控制混杂因素（如数据来源）后，仍然存在与年龄相关的内在偏差，对年轻患者不利。我们推测这种偏差可能与生理因素有关，这是放射科医生和自动系统面临的已知挑战之一。最后，我们展示了从多个数据源聚合数据如何影响特定站点的种族偏差，突显了在细粒度层面上调查数据的必要性。 

---
# FedMuon: Accelerating Federated Learning with Matrix Orthogonalization 

**Title (ZH)**: FedMuon: 用矩阵正交化加速 federated learning 

**Authors**: Junkang Liu, Fanhua Shang, Junchao Zhou, Hongying Liu, Yuanyuan Liu, Jin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27403)  

**Abstract**: The core bottleneck of Federated Learning (FL) lies in the communication rounds. That is, how to achieve more effective local updates is crucial for reducing communication rounds. Existing FL methods still primarily use element-wise local optimizers (Adam/SGD), neglecting the geometric structure of the weight matrices. This often leads to the amplification of pathological directions in the weights during local updates, leading deterioration in the condition number and slow convergence. Therefore, we introduce the Muon optimizer in local, which has matrix orthogonalization to optimize matrix-structured parameters. Experimental results show that, in IID setting, Local Muon significantly accelerates the convergence of FL and reduces communication rounds compared to Local SGD and Local AdamW. However, in non-IID setting, independent matrix orthogonalization based on the local distributions of each client induces strong client drift. Applying Muon in non-IID FL poses significant challenges: (1) client preconditioner leading to client drift; (2) moment reinitialization. To address these challenges, we propose a novel Federated Muon optimizer (FedMuon), which incorporates two key techniques: (1) momentum aggregation, where clients use the aggregated momentum for local initialization; (2) local-global alignment, where the local gradients are aligned with the global update direction to significantly reduce client drift. Theoretically, we prove that \texttt{FedMuon} achieves a linear speedup convergence rate without the heterogeneity assumption, where $S$ is the number of participating clients per round, $K$ is the number of local iterations, and $R$ is the total number of communication rounds. Empirically, we validate the effectiveness of FedMuon on language and vision models. Compared to several baselines, FedMuon significantly reduces communication rounds and improves test accuracy. 

**Abstract (ZH)**: 联邦学习中联邦学习的核心瓶颈在于通信轮次。具体来说，如何实现更有效的局部更新对于减少通信轮次至关重要。现有的联邦学习方法仍然主要使用元素-wise的局部优化器（如Adam/SGD），忽略了权重矩阵的几何结构。这往往导致在局部更新中权重的病态方向被放大，从而恶化条件数并导致收敛速度变慢。因此，我们引入了在局部使用的Muon优化器，该优化器对矩阵结构的参数进行矩阵正交化优化。实验结果表明，在IID设置下，局部Muon显著加速了联邦学习的收敛速度并减少了通信轮次，相较于Local SGD和Local AdamW。然而，在非IID设置下，基于每个客户端局部分布的独立矩阵正交化会导致强烈的客户端漂移。将Muon应用于非IID联邦学习提出了重大挑战：（1）客户端预条件化导致客户端漂移；（2）动量再初始化。为了解决这些挑战，我们提出了一个新颖的联邦Muon优化器（FedMuon），结合了两种关键技术：（1）动量聚合，客户端使用聚合后的动量进行局部初始化；（2）局部-全局对齐，将局部梯度与全局更新方向对齐，从而显著减少客户端漂移。理论上，我们证明在没有异质性假设的情况下，FedMuon实现了线性加速的收敛速率，其中$S$是每轮参与的客户端数目，$K$是局部迭代次数，$R$是总的通信轮次。实验上，我们验证了FedMuon在语言和视觉模型上的有效性。相比于几种基线方法，FedMuon显著减少了通信轮次并提高了测试精度。 

---
# Spiking Neural Networks: The Future of Brain-Inspired Computing 

**Title (ZH)**: 脉冲神经网络：脑启发计算的未来 

**Authors**: Sales G. Aribe Jr  

**Link**: [PDF](https://arxiv.org/pdf/2510.27379)  

**Abstract**: Spiking Neural Networks (SNNs) represent the latest generation of neural computation, offering a brain-inspired alternative to conventional Artificial Neural Networks (ANNs). Unlike ANNs, which depend on continuous-valued signals, SNNs operate using distinct spike events, making them inherently more energy-efficient and temporally dynamic. This study presents a comprehensive analysis of SNN design models, training algorithms, and multi-dimensional performance metrics, including accuracy, energy consumption, latency, spike count, and convergence behavior. Key neuron models such as the Leaky Integrate-and-Fire (LIF) and training strategies, including surrogate gradient descent, ANN-to-SNN conversion, and Spike-Timing Dependent Plasticity (STDP), are examined in depth. Results show that surrogate gradient-trained SNNs closely approximate ANN accuracy (within 1-2%), with faster convergence by the 20th epoch and latency as low as 10 milliseconds. Converted SNNs also achieve competitive performance but require higher spike counts and longer simulation windows. STDP-based SNNs, though slower to converge, exhibit the lowest spike counts and energy consumption (as low as 5 millijoules per inference), making them optimal for unsupervised and low-power tasks. These findings reinforce the suitability of SNNs for energy-constrained, latency-sensitive, and adaptive applications such as robotics, neuromorphic vision, and edge AI systems. While promising, challenges persist in hardware standardization and scalable training. This study concludes that SNNs, with further refinement, are poised to propel the next phase of neuromorphic computing. 

**Abstract (ZH)**: 脉冲神经网络（SNNs）代表了新一代神经计算，提供了与传统人工神经网络（ANNs）相媲美的脑启发替代方案。与依赖连续信号的ANNs不同，SNNs通过独特的脉冲事件进行操作，使其更加固能并具有时间动态性。本研究全面分析了SNN设计模型、训练算法以及包括准确率、能耗、延迟、脉冲计数和收敛行为在内的多维性能指标。深入探讨了诸如泄漏积分并触发（LIF）等关键神经元模型以及替代梯度下降、ANN转SNN转换和突触定时依赖可塑性（STDP）等训练策略。结果显示，经过替代梯度训练的SNNs在准确率上与ANNs相差1-2%，在第20个epoch后收敛速度加快，延迟低至10毫秒。转换得到的SNNs也表现出竞争力，但需要更高的脉冲计数和更长的模拟窗口。基于STDP的SNNs虽然收敛速度较慢，但脉冲计数和能量消耗最低（低至5毫焦耳每推理），使其适用于无监督和低功耗任务。这些发现强调了SNNs在受能源限制、延迟敏感和自适应应用，如机器人技术、类神经视觉和边缘AI系统中的适用性。尽管充满前景，但ハード件标准化和可扩展训练仍面临挑战。本研究结论认为，通过进一步完善，SNNs有可能推动神经形态计算的下一次飞跃。 

---
# Measuring Chain-of-Thought Monitorability Through Faithfulness and Verbosity 

**Title (ZH)**: 通过忠实度和啰嗦度衡量思维链可监控性 

**Authors**: Austin Meek, Eitan Sprejer, Iván Arcuschin, Austin J. Brockmeier, Steven Basart  

**Link**: [PDF](https://arxiv.org/pdf/2510.27378)  

**Abstract**: Chain-of-thought (CoT) outputs let us read a model's step-by-step reasoning. Since any long, serial reasoning process must pass through this textual trace, the quality of the CoT is a direct window into what the model is thinking. This visibility could help us spot unsafe or misaligned behavior (monitorability), but only if the CoT is transparent about its internal reasoning (faithfulness). Fully measuring faithfulness is difficult, so researchers often focus on examining the CoT in cases where the model changes its answer after adding a cue to the input. This proxy finds some instances of unfaithfulness but loses information when the model maintains its answer, and does not investigate aspects of reasoning not tied to the cue. We extend these results to a more holistic sense of monitorability by introducing verbosity: whether the CoT lists every factor needed to solve the task. We combine faithfulness and verbosity into a single monitorability score that shows how well the CoT serves as the model's external `working memory', a property that many safety schemes based on CoT monitoring depend on. We evaluate instruction-tuned and reasoning models on BBH, GPQA, and MMLU. Our results show that models can appear faithful yet remain hard to monitor when they leave out key factors, and that monitorability differs sharply across model families. We release our evaluation code using the Inspect library to support reproducible future work. 

**Abstract (ZH)**: 链式思考（CoT）输出使我们能够阅读模型的逐步推理过程。由于任何长序列的推理过程都必须通过这一文本痕迹，CoT的质量直接反映了模型的思考过程。这种透明性有助于我们识别不安全或对齐错误的行为（可观测性），但前提是CoT必须忠实反映其内部推理过程。完全衡量忠实性是困难的，因此研究人员通常专注于检查模型在输入中加入提示后改变答案的情况。这种代理方法可以发现一些不忠实的情况，但在模型保持其答案时会丢失信息，并不调查与提示无关的推理方面。我们通过引入冗余性（即CoT是否列出了完成任务所需的所有因素）将这些结果扩展到更全面的可观测性概念。我们将忠实性和冗余性结合成一个单一的可观测性评分，以显示CoT作为模型的外部“工作记忆”表现如何，这是许多基于CoT监控的安全方案所依赖的特性。我们对指令调整和推理模型在BBH、GPQA和MMLU上的表现进行了评估。我们的结果显示，当模型忽略关键因素时，模型可以显得忠实但难以监控，且不同模型家族的可观测性存在明显差异。我们使用Inspect库发布我们的评估代码，以支持未来工作的可重复性。 

---
# Un-Attributability: Computing Novelty From Retrieval & Semantic Similarity 

**Title (ZH)**: 不可归因性：从检索与语义相似性计算新颖性 

**Authors**: Philipp Davydov, Ameya Prabhu, Matthias Bethge, Elisa Nguyen, Seong Joon Oh  

**Link**: [PDF](https://arxiv.org/pdf/2510.27313)  

**Abstract**: Understanding how language-model outputs relate to the pretraining corpus is central to studying model behavior. Most training data attribution (TDA) methods ask which training examples causally influence a given output, often using leave-one-out tests. We invert the question: which outputs cannot be attributed to any pretraining example? We introduce un-attributability as an operational measure of semantic novelty: an output is novel if the pretraining corpus contains no semantically similar context. We approximate this with a simple two-stage retrieval pipeline: index the corpus with lightweight GIST embeddings, retrieve the top-n candidates, then rerank with ColBERTv2. If the nearest corpus item is less attributable than a human-generated text reference, we consider the output of the model as novel. We evaluate on SmolLM and SmolLM2 and report three findings: (1) models draw on pretraining data across much longer spans than previously reported; (2) some domains systematically promote or suppress novelty; and (3) instruction tuning not only alters style but also increases novelty. Reframing novelty assessment around un-attributability enables efficient analysis at pretraining scale. We release ~20 TB of corpus chunks and index artifacts to support replication and large-scale extension of our analysis at this https URL 

**Abstract (ZH)**: 理解语言模型输出与预训练语料的关系是研究模型行为的关键。大多数训练数据归属（TDA）方法询问哪些训练示例因果影响给定的输出，常常使用删一法测试。我们反转了这个问题：哪些输出不能归属到任何预训练示例？我们引入不可归属性作为语义新颖性的操作性度量：如果预训练语料中不存在语义相似的上下文，则输出被认为是新颖的。我们通过一个简单的两阶段检索管道进行近似：使用轻量级GIST嵌入标注语料，检索前n个候选项，然后使用ColBERTv2重新排序。如果最近的语料库项比人类生成的文本参考更难以归属，我们则认为模型的输出是新颖的。我们在SmolLM和SmolLM2上进行了评估，并记录了三项发现：（1）模型跨更长的跨度利用预训练数据；（2）某些领域系统地促进或抑制新颖性；（3）指令调优不仅改变风格，还增加了新颖性。将新颖性评估重新构想为不可归属性为特征，可以高效地在预训练规模下进行分析。我们发布约20 TB的语料片段和索引制品，以支持复制和大规模扩展我们的分析，详见这里。 

---
# HiF-DTA: Hierarchical Feature Learning Network for Drug-Target Affinity Prediction 

**Title (ZH)**: HiF-DTA：药物-靶标亲和力预测的分层特征学习网络 

**Authors**: Minghui Li, Yuanhang Wang, Peijin Guo, Wei Wan, Shengshan Hu, Shengqing Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27281)  

**Abstract**: Accurate prediction of Drug-Target Affinity (DTA) is crucial for reducing experimental costs and accelerating early screening in computational drug discovery. While sequence-based deep learning methods avoid reliance on costly 3D structures, they still overlook simultaneous modeling of global sequence semantic features and local topological structural features within drugs and proteins, and represent drugs as flat sequences without atomic-level, substructural-level, and molecular-level multi-scale features. We propose HiF-DTA, a hierarchical network that adopts a dual-pathway strategy to extract both global sequence semantic and local topological features from drug and protein sequences, and models drugs multi-scale to learn atomic, substructural, and molecular representations fused via a multi-scale bilinear attention module. Experiments on Davis, KIBA, and Metz datasets show HiF-DTA outperforms state-of-the-art baselines, with ablations confirming the importance of global-local extraction and multi-scale fusion. 

**Abstract (ZH)**: 准确预测药物-靶标亲和力（DTA）对于降低实验成本并加速计算药物发现中的早期筛选至关重要。虽然基于序列的深度学习方法避免了依赖昂贵的三维结构，但仍忽视了同时建模药物和蛋白质中全局序列语义特征和局部拓扑结构特征，以及将药物表示为扁平序列而未考虑原子级、亚结构级和分子级的多尺度特征。我们提出了一种分级网络HiF-DTA，采用双路径策略从药物和蛋白质序列中提取全局序列语义和局部拓扑特征，并多尺度建模药物以通过多尺度双线性注意力模块学习原子、亚结构和分子表示。在Davis、KIBA和Metz数据集上的实验表明，HiF-DTA优于现有基线方法，消融实验进一步证实了全局-局部提取和多尺度融合的重要性。 

---
# Why Do Multilingual Reasoning Gaps Emerge in Reasoning Language Models? 

**Title (ZH)**: 为什么多语言推理差距会在推理语言模型中出现？ 

**Authors**: Deokhyung Kang, Seonjeong Hwang, Daehui Kim, Hyounghun Kim, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.27269)  

**Abstract**: Reasoning language models (RLMs) achieve strong performance on complex reasoning tasks, yet they still suffer from a multilingual reasoning gap, performing better in high-resource languages than in low-resource ones. While recent efforts have reduced this gap, its underlying causes remain largely unexplored. In this paper, we address this by showing that the multilingual reasoning gap largely stems from failures in language understanding-the model's inability to represent the multilingual input meaning into the dominant language (i.e., English) within its reasoning trace. This motivates us to examine whether understanding failures can be detected, as this ability could help mitigate the multilingual reasoning gap. To this end, we evaluate a range of detection methods and find that understanding failures can indeed be identified, with supervised approaches performing best. Building on this, we propose Selective Translation, a simple yet effective strategy that translates the multilingual input into English only when an understanding failure is detected. Experimental results show that Selective Translation bridges the multilingual reasoning gap, achieving near full-translation performance while using translation for only about 20% of inputs. Together, our work demonstrates that understanding failures are the primary cause of the multilingual reasoning gap and can be detected and selectively mitigated, providing key insight into its origin and a promising path toward more equitable multilingual reasoning. Our code and data are publicly available at this https URL. 

**Abstract (ZH)**: 多语言推理差距主要源于语言理解失败：检测与缓解策略 

---
# Higher-order Linear Attention 

**Title (ZH)**: 高阶线性注意力 

**Authors**: Yifan Zhang, Zhen Qin, Quanquan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27258)  

**Abstract**: The quadratic cost of scaled dot-product attention is a central obstacle to scaling autoregressive language models to long contexts. Linear-time attention and State Space Models (SSMs) provide scalable alternatives but are typically restricted to first-order or kernel-based approximations, which can limit expressivity. We introduce Higher-order Linear Attention (HLA), a causal, streaming mechanism that realizes higher interactions via compact prefix sufficient statistics. In the second-order case, HLA maintains a constant-size state and computes per-token outputs in linear time without materializing any $n \times n$ matrices. We give closed-form streaming identities, a strictly causal masked variant using two additional summaries, and a chunk-parallel training scheme based on associative scans that reproduces the activations of a serial recurrence exactly. We further outline extensions to third and higher orders. Collectively, these results position HLA as a principled, scalable building block that combines attention-like, data-dependent mixing with the efficiency of modern recurrent architectures. Project Page: this https URL. 

**Abstract (ZH)**: 缩放点积注意力的二次成本是将自回归语言模型扩展到长上下文的主要障碍。高阶线性注意力（HLA）是一种因果 Streaming 机制，通过紧凑的前缀充分统计量实现更高阶交互。在二阶情况下，HLA 维持恒定大小的状态并在线性时间内计算每词输出，无需构造任何 \(n \times n\) 矩阵。我们给出了闭合形式的 Streaming 标识，一个严格因果的屏蔽变体使用两个额外的摘要，以及基于关联扫描的分块并行训练方案，该方案能够准确再现串行递归的激活。我们还概述了高阶（三阶及以上）的扩展。这些结果将HLA定位为一个兼具注意力样式的数据依赖混合与现代递归架构效率的有原则且可扩展的基本构建块。项目页面：这个 https URL。 

---
# Not All Instances Are Equally Valuable: Towards Influence-Weighted Dataset Distillation 

**Title (ZH)**: 并非所有实例的价值都相同：面向影响加权数据集蒸馏 

**Authors**: Qiyan Deng, Changqian Zheng, Lianpeng Qiao, Yuping Wang, Chengliang Chai, Lei Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.27253)  

**Abstract**: Dataset distillation condenses large datasets into synthetic subsets, achieving performance comparable to training on the full dataset while substantially reducing storage and computation costs. Most existing dataset distillation methods assume that all real instances contribute equally to the process. In practice, real-world datasets contain both informative and redundant or even harmful instances, and directly distilling the full dataset without considering data quality can degrade model performance. In this work, we present Influence-Weighted Distillation IWD, a principled framework that leverages influence functions to explicitly account for data quality in the distillation process. IWD assigns adaptive weights to each instance based on its estimated impact on the distillation objective, prioritizing beneficial data while downweighting less useful or harmful ones. Owing to its modular design, IWD can be seamlessly integrated into diverse dataset distillation frameworks. Our empirical results suggest that integrating IWD tends to improve the quality of distilled datasets and enhance model performance, with accuracy gains of up to 7.8%. 

**Abstract (ZH)**: 数据集蒸馏将大规模数据集凝练为合成子集，在显著减少存储和计算成本的同时，实现与使用完整数据集训练相当的性能。现有大多数数据集蒸馏方法假设所有真实实例对过程的贡献均等。实际上，真实世界数据集包含既有信息性又有冗余或甚至有害的实例，如果不考虑数据质量直接蒸馏完整数据集可能会降低模型性能。在本文中，我们提出了影响加权蒸馏（IWD），这是一种原理性的框架，利用影响函数明确考虑数据质量。IWD 根据每个实例对蒸馏目标的估计影响赋予自适应权重，优先选择有益的数据，同时降低无用或有害数据的权重。由于其模块化设计，IWD 可以无缝集成到各种数据集蒸馏框架中。我们的实验证据表明，集成 IWD 通常可以提高蒸馏数据集的质量并增强模型性能，准确率提升高达 7.8%。 

---
# Reconstructing Unseen Sentences from Speech-related Biosignals for Open-vocabulary Neural Communication 

**Title (ZH)**: 从语音相关生物信号重构未见句子的开放词汇神经通信 

**Authors**: Deok-Seon Kim, Seo-Hyun Lee, Kang Yin, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.27247)  

**Abstract**: Brain-to-speech (BTS) systems represent a groundbreaking approach to human communication by enabling the direct transformation of neural activity into linguistic expressions. While recent non-invasive BTS studies have largely focused on decoding predefined words or sentences, achieving open-vocabulary neural communication comparable to natural human interaction requires decoding unconstrained speech. Additionally, effectively integrating diverse signals derived from speech is crucial for developing personalized and adaptive neural communication and rehabilitation solutions for patients. This study investigates the potential of speech synthesis for previously unseen sentences across various speech modes by leveraging phoneme-level information extracted from high-density electroencephalography (EEG) signals, both independently and in conjunction with electromyography (EMG) signals. Furthermore, we examine the properties affecting phoneme decoding accuracy during sentence reconstruction and offer neurophysiological insights to further enhance EEG decoding for more effective neural communication solutions. Our findings underscore the feasibility of biosignal-based sentence-level speech synthesis for reconstructing unseen sentences, highlighting a significant step toward developing open-vocabulary neural communication systems adapted to diverse patient needs and conditions. Additionally, this study provides meaningful insights into the development of communication and rehabilitation solutions utilizing EEG-based decoding technologies. 

**Abstract (ZH)**: 基于脑电的无约束语音合成及其在神经交流系统中的应用研究 

---
# Vintage Code, Modern Judges: Meta-Validation in Low Data Regimes 

**Title (ZH)**: 陈旧代码，现代法官：低数据情况下的元验证 

**Authors**: Ora Nova Fandina, Gal Amram, Eitan Farchi, Shmulik Froimovich, Raviv Gal, Wesam Ibraheem, Rami Katan, Alice Podolsky, Orna Raz  

**Link**: [PDF](https://arxiv.org/pdf/2510.27244)  

**Abstract**: Application modernization in legacy languages such as COBOL, PL/I, and REXX faces an acute shortage of resources, both in expert availability and in high-quality human evaluation data. While Large Language Models as a Judge (LaaJ) offer a scalable alternative to expert review, their reliability must be validated before being trusted in high-stakes workflows. Without principled validation, organizations risk a circular evaluation loop, where unverified LaaJs are used to assess model outputs, potentially reinforcing unreliable judgments and compromising downstream deployment decisions. Although various automated approaches to validating LaaJs have been proposed, alignment with human judgment remains a widely used and conceptually grounded validation strategy. In many real-world domains, the availability of human-labeled evaluation data is severely limited, making it difficult to assess how well a LaaJ aligns with human judgment. We introduce SparseAlign, a formal framework for assessing LaaJ alignment with sparse human-labeled data. SparseAlign combines a novel pairwise-confidence concept with a score-sensitive alignment metric that jointly capture ranking consistency and score proximity, enabling reliable evaluator selection even when traditional statistical methods are ineffective due to limited annotated examples. SparseAlign was applied internally to select LaaJs for COBOL code explanation. The top-aligned evaluators were integrated into assessment workflows, guiding model release decisions. We present a case study of four LaaJs to demonstrate SparseAlign's utility in real-world evaluation scenarios. 

**Abstract (ZH)**: 基于少量人工标注数据评估大型语言模型作为评判者的稀疏对齐框架 

---
# DRAMA: Unifying Data Retrieval and Analysis for Open-Domain Analytic Queries 

**Title (ZH)**: DRAMA: 统一开放域分析查询中的数据检索与分析 

**Authors**: Chuxuan Hu, Maxwell Yang, James Weiland, Yeji Lim, Suhas Palawala, Daniel Kang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27238)  

**Abstract**: Manually conducting real-world data analyses is labor-intensive and inefficient. Despite numerous attempts to automate data science workflows, none of the existing paradigms or systems fully demonstrate all three key capabilities required to support them effectively: (1) open-domain data collection, (2) structured data transformation, and (3) analytic reasoning.
To overcome these limitations, we propose DRAMA, an end-to-end paradigm that answers users' analytic queries in natural language on large-scale open-domain data. DRAMA unifies data collection, transformation, and analysis as a single pipeline. To quantitatively evaluate system performance on tasks representative of DRAMA, we construct a benchmark, DRAMA-Bench, consisting of two categories of tasks: claim verification and question answering, each comprising 100 instances. These tasks are derived from real-world applications that have gained significant public attention and require the retrieval and analysis of open-domain data. We develop DRAMA-Bot, a multi-agent system designed following DRAMA. It comprises a data retriever that collects and transforms data by coordinating the execution of sub-agents, and a data analyzer that performs structured reasoning over the retrieved data. We evaluate DRAMA-Bot on DRAMA-Bench together with five state-of-the-art baseline agents. DRAMA-Bot achieves 86.5% task accuracy at a cost of $0.05, outperforming all baselines with up to 6.9 times the accuracy and less than 1/6 of the cost. DRAMA is publicly available at this https URL. 

**Abstract (ZH)**: 手动进行现实世界数据的分析劳动密集且效率低下。尽管已经尝试通过自动化数据科学工作流来解决这一问题，但现有的范式或系统无法全面展示支持这些工作流所需的三项关键能力：(1) 开放领域数据收集，(2) 结构化数据转换，以及 (3) 分析推理。为了克服这些限制，我们提出DRAMA，一种端到端范式，能够使用自然语言在大规模开放领域数据上回答用户的分析查询。DRAMA将数据收集、转换和分析统一为一个单一的工作流。为了定量评估系统性能，我们构建了DRAMA-Bench基准测试，包含两类任务：声明验证和问答，每类各包含100个实例。这些任务来自具有显著公众关注的实际应用，需要检索和分析开放领域数据。我们开发了遵循DRAMA设计的多智能体系统DRAMA-Bot。它包括一个数据检索器，通过协调子智能体的执行来收集和转换数据，并且包括一个数据分析师，对检索到的数据进行结构化推理。我们使用DRAMA-Bench与五种最先进的基线智能体一起评估了DRAMA-Bot。DRAMA-Bot在任务准确率方面达到86.5%，成本为0.05，比所有基线的准确率高出多达6.9倍，且成本低于基线的1/6。DRAMA可以在以下链接公开访问：this https URL。DRAMA端到端范式在大规模开放领域数据上的自然语言分析查询 

---
# Soft Task-Aware Routing of Experts for Equivariant Representation Learning 

**Title (ZH)**: 专家软任务感知路由的不变表示学习 

**Authors**: Jaebyeong Jeon, Hyeonseo Jang, Jy-yong Sohn, Kibok Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.27222)  

**Abstract**: Equivariant representation learning aims to capture variations induced by input transformations in the representation space, whereas invariant representation learning encodes semantic information by disregarding such transformations. Recent studies have shown that jointly learning both types of representations is often beneficial for downstream tasks, typically by employing separate projection heads. However, this design overlooks information shared between invariant and equivariant learning, which leads to redundant feature learning and inefficient use of model capacity. To address this, we introduce Soft Task-Aware Routing (STAR), a routing strategy for projection heads that models them as experts. STAR induces the experts to specialize in capturing either shared or task-specific information, thereby reducing redundant feature learning. We validate this effect by observing lower canonical correlations between invariant and equivariant embeddings. Experimental results show consistent improvements across diverse transfer learning tasks. The code is available at this https URL. 

**Abstract (ZH)**: 变换成 invariant 和 equivariant 表征的学习方法旨在分别捕获输入变换在表示空间中引起的变化和不变的信息，而 recent 研究表明，同时学习这两种类型的表征通常对下游任务是有益的，通常通过使用分开的投影头来实现。但这种设计忽略了不变学习和变换成学习之间共享的信息，导致冗余特征学习和模型容量的低效利用。为了解决这一问题，我们引入了 Soft Task-Aware Routing (STAR) —— 一种将投影头建模为专家的路由策略，STAR 促使专家专门捕获共享或任务特定的信息，从而减少冗余特征学习。我们通过观察不变和变换成嵌入的统计相关性的降低验证了这一效果。实验结果表明，STAR 在多种迁移学习任务上一致性地提高了性能。代码可在该链接获取。 

---
# Privacy-Aware Continual Self-Supervised Learning on Multi-Window Chest Computed Tomography for Domain-Shift Robustness 

**Title (ZH)**: 面向隐私保护的多窗格胸部 computed tomography 持续自监督学习及其在领域迁移稳健性中的应用 

**Authors**: Ren Tasai, Guang Li, Ren Togo, Takahiro Ogawa, Kenji Hirata, Minghui Tang, Takaaki Yoshimura, Hiroyuki Sugimori, Noriko Nishioka, Yukie Shimizu, Kohsuke Kudo, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2510.27213)  

**Abstract**: We propose a novel continual self-supervised learning (CSSL) framework for simultaneously learning diverse features from multi-window-obtained chest computed tomography (CT) images and ensuring data privacy. Achieving a robust and highly generalizable model in medical image diagnosis is challenging, mainly because of issues, such as the scarcity of large-scale, accurately annotated datasets and domain shifts inherent to dynamic healthcare environments. Specifically, in chest CT, these domain shifts often arise from differences in window settings, which are optimized for distinct clinical purposes. Previous CSSL frameworks often mitigated domain shift by reusing past data, a typically impractical approach owing to privacy constraints. Our approach addresses these challenges by effectively capturing the relationship between previously learned knowledge and new information across different training stages through continual pretraining on unlabeled images. Specifically, by incorporating a latent replay-based mechanism into CSSL, our method mitigates catastrophic forgetting due to domain shifts during continual pretraining while ensuring data privacy. Additionally, we introduce a feature distillation technique that integrates Wasserstein distance-based knowledge distillation (WKD) and batch-knowledge ensemble (BKE), enhancing the ability of the model to learn meaningful, domain-shift-robust representations. Finally, we validate our approach using chest CT images obtained across two different window settings, demonstrating superior performance compared with other approaches. 

**Abstract (ZH)**: 我们提出了一种新颖的持续自监督学习（CSSL）框架，用于同时从多窗口获取的胸部计算机断层扫描（CT）图像中学习多种特征，并确保数据隐私。 

---
# Feature-Function Curvature Analysis: A Geometric Framework for Explaining Differentiable Models 

**Title (ZH)**: 特征-函数曲率分析：可微模型解释的几何框架 

**Authors**: Hamed Najafi, Dongsheng Luo, Jason Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27207)  

**Abstract**: Explainable AI (XAI) is critical for building trust in complex machine learning models, yet mainstream attribution methods often provide an incomplete, static picture of a model's final state. By collapsing a feature's role into a single score, they are confounded by non-linearity and interactions. To address this, we introduce Feature-Function Curvature Analysis (FFCA), a novel framework that analyzes the geometry of a model's learned function. FFCA produces a 4-dimensional signature for each feature, quantifying its: (1) Impact, (2) Volatility, (3) Non-linearity, and (4) Interaction. Crucially, we extend this framework into Dynamic Archetype Analysis, which tracks the evolution of these signatures throughout the training process. This temporal view moves beyond explaining what a model learned to revealing how it learns. We provide the first direct, empirical evidence of hierarchical learning, showing that models consistently learn simple linear effects before complex interactions. Furthermore, this dynamic analysis provides novel, practical diagnostics for identifying insufficient model capacity and predicting the onset of overfitting. Our comprehensive experiments demonstrate that FFCA, through its static and dynamic components, provides the essential geometric context that transforms model explanation from simple quantification to a nuanced, trustworthy analysis of the entire learning process. 

**Abstract (ZH)**: 可解释人工智能（XAI）对于建立对复杂机器学习模型的信任至关重要，然而主流的归因方法往往提供的是模型最终状态的一种不完整、静态的视角。通过将特征的作用归结为单一评分，它们会受到非线性和交互作用的影响。为了解决这一问题，我们引入了特征函数曲率分析（FFCA）这一新颖框架，该框架分析了模型学习函数的几何结构。FFCA为每个特征生成了一个4维签名，量化了其：（1）影响，（2）波动性，（3）非线性，以及（4）交互作用。关键的是，我们进一步将这一框架扩展到动态典型模式分析，该分析追踪这些签名在整个训练过程中的演变。这种时间维度的视角不仅解释了模型学到了什么，还揭示了它是如何学习的。提供了首个直接的实证证据，展示了分层次学习的现象，表明模型在学习复杂交互作用之前一直学习简单的线性效应。此外，这一动态分析还提供了新的实用诊断方法，用于识别模型容量不足并预测过度拟合的出现。我们的全面实验表明，FFCA通过其静态和动态部分，提供了模型解释所需的必要几何上下文，将模型解释从简单的量化转变为对整个学习过程的细微、可靠的分析。 

---
# Vectorized Online POMDP Planning 

**Title (ZH)**: 向量化在线POMDP规划 

**Authors**: Marcus Hoerger, Muhammad Sudrajat, Hanna Kurniawati  

**Link**: [PDF](https://arxiv.org/pdf/2510.27191)  

**Abstract**: Planning under partial observability is an essential capability of autonomous robots. The Partially Observable Markov Decision Process (POMDP) provides a powerful framework for planning under partial observability problems, capturing the stochastic effects of actions and the limited information available through noisy observations. POMDP solving could benefit tremendously from massive parallelization of today's hardware, but parallelizing POMDP solvers has been challenging. They rely on interleaving numerical optimization over actions with the estimation of their values, which creates dependencies and synchronization bottlenecks between parallel processes that can quickly offset the benefits of parallelization. In this paper, we propose Vectorized Online POMDP Planner (VOPP), a novel parallel online solver that leverages a recent POMDP formulation that analytically solves part of the optimization component, leaving only the estimation of expectations for numerical computation. VOPP represents all data structures related to planning as a collection of tensors and implements all planning steps as fully vectorized computations over this representation. The result is a massively parallel solver with no dependencies and synchronization bottlenecks between parallel computations. Experimental results indicate that VOPP is at least 20X more efficient in computing near-optimal solutions compared to an existing state-of-the-art parallel online solver. 

**Abstract (ZH)**: 部分可观测性的规划是自主机器人的一项基本能力。部分可观测马尔可夫决策过程（POMDP）为部分可观测性问题提供了一个强大的框架，捕捉行动的随机效应以及通过噪声观测获得的有限信息。POMDP的求解可以从当今硬件的并行化中受益匪浅，但并行化POMDP求解器颇具挑战性。它们依赖于行动的数值优化和其价值的估计交织进行，这在并行过程中产生了依赖性和同步瓶颈，这些瓶颈会迅速抵消并行化的益处。在本文中，我们提出了一种新颖的并行在线求解器——向量在线POMDP规划器（VOPP），该求解器利用了最近的POMDP形式化方法，该方法对优化组件的一部分进行了分析解算，仅留期望估计用于数值计算。VOPP将所有与规划相关的数据结构表示为张量集合，并实现所有规划步骤为对该表示的完全向量化计算。结果是一种无依赖性和同步瓶颈的并行求解器。实验结果表明，与现有最先进的并行在线求解器相比，VOPP在计算近最优解时至少快20倍。 

---
# Sparse Model Inversion: Efficient Inversion of Vision Transformers for Data-Free Applications 

**Title (ZH)**: 稀疏模型反转：视觉变换器的高效无数据应用反演 

**Authors**: Zixuan Hu, Yongxian Wei, Li Shen, Zhenyi Wang, Lei Li, Chun Yuan, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2510.27186)  

**Abstract**: Model inversion, which aims to reconstruct the original training data from pre-trained discriminative models, is especially useful when the original training data is unavailable due to privacy, usage rights, or size constraints. However, existing dense inversion methods attempt to reconstruct the entire image area, making them extremely inefficient when inverting high-resolution images from large-scale Vision Transformers (ViTs). We further identify two underlying causes of this inefficiency: the redundant inversion of noisy backgrounds and the unintended inversion of spurious correlations--a phenomenon we term "hallucination" in model inversion. To address these limitations, we propose a novel sparse model inversion strategy, as a plug-and-play extension to speed up existing dense inversion methods with no need for modifying their original loss functions. Specifically, we selectively invert semantic foregrounds while stopping the inversion of noisy backgrounds and potential spurious correlations. Through both theoretical and empirical studies, we validate the efficacy of our approach in achieving significant inversion acceleration (up to 3.79 faster) while maintaining comparable or even enhanced downstream performance in data-free model quantization and data-free knowledge transfer. Code is available at this https URL. 

**Abstract (ZH)**: 从预训练辨别模型重建原始训练数据的模型反演，特别是在原始训练数据因隐私、使用权限或大小限制等原因不可用时，尤其有用。然而，现有的密集反演方法试图重构整幅图像区域，使得它们在从大规模视觉变换器（ViTs）反演高分辨率图像时极其低效。我们进一步识别出这一低效性的两个根本原因：噪声背景的冗余反演和未预期的伪相关性的反演——我们将这种现象在模型反演中称为“幻觉”。为解决这些局限，我们提出了一种新的稀疏模型反演策略，作为即插即用扩展，无需修改原始损失函数即可加速现有的密集反演方法。具体来说，我们选择性地反演语义前景，而停止噪声背景和潜在伪相关性的反演。通过理论和实验研究，我们验证了该方法在实现显著反演加速（最高3.79倍）的同时，保持或甚至提升了数据免费模型量化和数据免费知识传递的下游性能。代码可在以下链接获取。 

---
# FMint-SDE: A Multimodal Foundation Model for Accelerating Numerical Simulation of SDEs via Error Correction 

**Title (ZH)**: FMint-SDE：一种通过误差校正加速SDE数值模拟的多模态基础模型 

**Authors**: Jiaxin Yuan, Haizhao Yang, Maria Cameron  

**Link**: [PDF](https://arxiv.org/pdf/2510.27173)  

**Abstract**: Fast and accurate simulation of dynamical systems is a fundamental challenge across scientific and engineering domains. Traditional numerical integrators often face a trade-off between accuracy and computational efficiency, while existing neural network-based approaches typically require training a separate model for each case. To overcome these limitations, we introduce a novel multi-modal foundation model for large-scale simulations of differential equations: FMint-SDE (Foundation Model based on Initialization for stochastic differential equations). Based on a decoder-only transformer with in-context learning, FMint-SDE leverages numerical and textual modalities to learn a universal error-correction scheme. It is trained using prompted sequences of coarse solutions generated by conventional solvers, enabling broad generalization across diverse systems. We evaluate our models on a suite of challenging SDE benchmarks spanning applications in molecular dynamics, mechanical systems, finance, and biology. Experimental results show that our approach achieves a superior accuracy-efficiency tradeoff compared to classical solvers, underscoring the potential of FMint-SDE as a general-purpose simulation tool for dynamical systems. 

**Abstract (ZH)**: 快速且准确地模拟动态系统是跨各个科学和工程领域的一项基本挑战。传统数值积分器往往在精度和计算效率之间存在权衡，而现有的基于神经网络的方法通常需要为每种情况训练一个单独的模型。为克服这些限制，我们提出了一种基于初始化的新型多模态基础模型，用于大规模微分方程模拟：FMint-SDE（基于初始化的面向随机微分方程的基础模型）。基于只有解码器的变压器并利用上下文学习，FMint-SDE 利用数值和文本模态学习一种通用的误差校正方案。它通过使用由传统求解器生成的粗糙解序列进行提示学习，从而在不同系统之间实现广泛的泛化能力。我们在涵盖分子动力学、机械系统、金融和生物学等多个应用领域的随机微分方程基准测试上评估了我们的模型。实验结果表明，与经典求解器相比，我们的方法在准确性和效率之间取得了更优的权衡，突显了FMint-SDE 作为动态系统通用模拟工具的潜力。 

---
# H2-Cache: A Novel Hierarchical Dual-Stage Cache for High-Performance Acceleration of Generative Diffusion Models 

**Title (ZH)**: H2-Cache: 一种用于生成性扩散模型高性能加速的新型分层两级缓存结构 

**Authors**: Mingyu Sung, Il-Min Kim, Sangseok Yun, Jae-Mo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27171)  

**Abstract**: Diffusion models have emerged as state-of-the-art in image generation, but their practical deployment is hindered by the significant computational cost of their iterative denoising process. While existing caching techniques can accelerate inference, they often create a challenging trade-off between speed and fidelity, suffering from quality degradation and high computational overhead. To address these limitations, we introduce H2-Cache, a novel hierarchical caching mechanism designed for modern generative diffusion model architectures. Our method is founded on the key insight that the denoising process can be functionally separated into a structure-defining stage and a detail-refining stage. H2-cache leverages this by employing a dual-threshold system, using independent thresholds to selectively cache each stage. To ensure the efficiency of our dual-check approach, we introduce pooled feature summarization (PFS), a lightweight technique for robust and fast similarity estimation. Extensive experiments on the Flux architecture demonstrate that H2-cache achieves significant acceleration (up to 5.08x) while maintaining image quality nearly identical to the baseline, quantitatively and qualitatively outperforming existing caching methods. Our work presents a robust and practical solution that effectively resolves the speed-quality dilemma, significantly lowering the barrier for the real-world application of high-fidelity diffusion models. Source code is available at this https URL. 

**Abstract (ZH)**: H2-Cache：一种用于现代生成扩散模型架构的新型分层缓存机制 

---
# MARIA: A Framework for Marginal Risk Assessment without Ground Truth in AI Systems 

**Title (ZH)**: MARIA：AI系统中无地面真实数据的边际风险评估框架 

**Authors**: Jieshan Chen, Suyu Ma, Qinghua Lu, Sung Une Lee, Liming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27163)  

**Abstract**: Before deploying an AI system to replace an existing process, it must be compared with the incumbent to ensure improvement without added risk. Traditional evaluation relies on ground truth for both systems, but this is often unavailable due to delayed or unknowable outcomes, high costs, or incomplete data, especially for long-standing systems deemed safe by convention. The more practical solution is not to compute absolute risk but the difference between systems. We therefore propose a marginal risk assessment framework, that avoids dependence on ground truth or absolute risk. It emphasizes three kinds of relative evaluation methodology, including predictability, capability and interaction dominance. By shifting focus from absolute to relative evaluation, our approach equips software teams with actionable guidance: identifying where AI enhances outcomes, where it introduces new risks, and how to adopt such systems responsibly. 

**Abstract (ZH)**: 在部署AI系统替代现有流程之前，必须对比两者以确保无额外风险的改进。传统的评估依赖于两者的ground truth，但在许多情况下由于结果延迟或不可知、高昂的成本或数据不完整，尤其是对于被认为安全的长期系统，ground truth往往不可用。更实用的解决方案是从计算绝对风险转向评估系统之间的差异。因此，我们提出了一种边际风险评估框架，该框架避免了对ground truth或绝对风险的依赖。它强调三种相对评估方法，包括可预测性、能力和相互主导性。通过从绝对评估转向相对评估，我们的方法为软件团队提供了可操作的指导：识别AI如何提升结果、引入新风险的领域，以及如何负责任地采用此类系统。 

---
# Exploring Landscapes for Better Minima along Valleys 

**Title (ZH)**: 探索景观以在山谷中获得更好的极小值 

**Authors**: Tong Zhao, Jiacheng Li, Yuanchang Zhou, Guangming Tan, Weile Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.27153)  

**Abstract**: Finding lower and better-generalizing minima is crucial for deep learning. However, most existing optimizers stop searching the parameter space once they reach a local minimum. Given the complex geometric properties of the loss landscape, it is difficult to guarantee that such a point is the lowest or provides the best generalization. To address this, we propose an adaptor "E" for gradient-based optimizers. The adapted optimizer tends to continue exploring along landscape valleys (areas with low and nearly identical losses) in order to search for potentially better local minima even after reaching a local minimum. This approach increases the likelihood of finding a lower and flatter local minimum, which is often associated with better generalization. We also provide a proof of convergence for the adapted optimizers in both convex and non-convex scenarios for completeness. Finally, we demonstrate their effectiveness in an important but notoriously difficult training scenario, large-batch training, where Lamb is the benchmark optimizer. Our testing results show that the adapted Lamb, ALTO, increases the test accuracy (generalization) of the current state-of-the-art optimizer by an average of 2.5% across a variety of large-batch training tasks. This work potentially opens a new research direction in the design of optimization algorithms. 

**Abstract (ZH)**: 寻找更低且泛化性能更好的局部极小值对于深度学习至关重要。然而，现有的优化器在达到局部极小值后通常会停止搜索参数空间。鉴于损失景观具有复杂的几何特性，难以保证这样的点是全局最低点或提供最佳泛化性能。为此，我们提出了一个用于梯度基优化器的“E”适配器。适配后的优化器倾向于继续在景观谷地（低且几乎相同的损失区域）中探索，即使在达到局部极小值后也在寻找潜在更好的局部极小值。这种方法增加了找到更低且更平坦的局部极小值的可能性，这通常与更好的泛化性能相关。我们还为适配优化器提供了凸和非凸场景下的收敛性证明以保持完整性。最后，我们在一个关键但历来难以处理的训练场景——大规模训练——中展示了它们的有效性，其中Lamb是基准优化器。我们的测试结果表明，适配后的Lamb（ALTO）在各种大规模训练任务中平均提高了当前最佳优化器的测试准确率（泛化性能）2.5%。这项工作可能为优化算法的设计开辟一个新的研究方向。 

---
# Expressive Range Characterization of Open Text-to-Audio Models 

**Title (ZH)**: 开放文本到语音模型的表达范围characterization 

**Authors**: Jonathan Morse, Azadeh Naderi, Swen Gaudl, Mark Cartwright, Amy K. Hoover, Mark J. Nelson  

**Link**: [PDF](https://arxiv.org/pdf/2510.27102)  

**Abstract**: Text-to-audio models are a type of generative model that produces audio output in response to a given textual prompt. Although level generators and the properties of the functional content that they create (e.g., playability) dominate most discourse in procedurally generated content (PCG), games that emotionally resonate with players tend to weave together a range of creative and multimodal content (e.g., music, sounds, visuals, narrative tone), and multimodal models have begun seeing at least experimental use for this purpose. However, it remains unclear what exactly such models generate, and with what degree of variability and fidelity: audio is an extremely broad class of output for a generative system to target.
Within the PCG community, expressive range analysis (ERA) has been used as a quantitative way to characterize generators' output space, especially for level generators. This paper adapts ERA to text-to-audio models, making the analysis tractable by looking at the expressive range of outputs for specific, fixed prompts. Experiments are conducted by prompting the models with several standardized prompts derived from the Environmental Sound Classification (ESC-50) dataset. The resulting audio is analyzed along key acoustic dimensions (e.g., pitch, loudness, and timbre). More broadly, this paper offers a framework for ERA-based exploratory evaluation of generative audio models. 

**Abstract (ZH)**: 基于文本的音频模型是一类生成模型，能够根据给定的文本提示生成音频输出。尽管层级生成器及其生成的功能内容属性（如可玩性）主导了程序化生成内容（PCG）的大部分讨论，但能够与玩家产生情感共鸣的游戏往往会综合各种创意和多模态内容（如音乐、声音、视觉、叙述语气），并且多模态模型已经开始为此目的进行实验性使用。然而，尚不清楚此类模型生成的具体内容及其多变性和保真度：音频是生成系统需要为目标的极其广泛的一类输出。在PCG领域，表达范围分析（ERA）已被用作一种定量方法来表征生成器的输出空间，尤其是对于层级生成器。本文将ERA应用于基于文本的音频模型，通过分析特定固定提示的输出表达范围使分析变得可行。实验通过使用源自Environmental Sound Classification (ESC-50)数据集的多个标准化提示来提示模型，分析生成的音频在关键声学维度上的表现（如音高、响度和音色）。更广泛地说，本文提供了一种基于ERA的探索性评估生成音频模型的框架。 

---
# Towards a Measure of Algorithm Similarity 

**Title (ZH)**: 向着算法相似性度量的研究 

**Authors**: Shairoz Sohail, Taher Ali  

**Link**: [PDF](https://arxiv.org/pdf/2510.27063)  

**Abstract**: Given two algorithms for the same problem, can we determine whether they are meaningfully different? In full generality, the question is uncomputable, and empirically it is muddied by competing notions of similarity. Yet, in many applications (such as clone detection or program synthesis) a pragmatic and consistent similarity metric is necessary. We review existing equivalence and similarity notions and introduce EMOC: An Evaluation-Memory-Operations-Complexity framework that embeds algorithm implementations into a feature space suitable for downstream tasks. We compile PACD, a curated dataset of verified Python implementations across three problems, and show that EMOC features support clustering and classification of algorithm types, detection of near-duplicates, and quantification of diversity in LLM-generated programs. Code, data, and utilities for computing EMOC embeddings are released to facilitate reproducibility and future work on algorithm similarity. 

**Abstract (ZH)**: 给定同一问题的两个算法，我们能否确定它们是否有意义的差异？在一般情况下，这个问题是不可计算的，而且由于相似性概念的竞争性，从经验上来看会变得模糊不清。然而，在许多应用中（如克隆检测或程序合成），需要一种实用且一致的相似性度量标准。我们回顾了现有的等价性和相似性概念，并引入了EMOC：一种评价-内存-操作复杂性框架，将算法实现嵌入到适合下游任务的特征空间中。我们编译了PACD，这是一个包含三个问题的验证Python实现的精编数据集，并展示了EMOC特征支持算法类型聚类和分类、接近重复检测以及LLM生成程序多样性的量化。提供了用于计算EMOC嵌入的代码、数据和工具，以促进算法相似性研究的可重复性和未来工作。 

---
# Dataset Creation and Baseline Models for Sexism Detection in Hausa 

**Title (ZH)**: Hausa性別主義检测的数据集创建与基线模型构建 

**Authors**: Fatima Adam Muhammad, Shamsuddeen Muhammad Hassan, Isa Inuwa-Dutse  

**Link**: [PDF](https://arxiv.org/pdf/2510.27038)  

**Abstract**: Sexism reinforces gender inequality and social exclusion by perpetuating stereotypes, bias, and discriminatory norms. Noting how online platforms enable various forms of sexism to thrive, there is a growing need for effective sexism detection and mitigation strategies. While computational approaches to sexism detection are widespread in high-resource languages, progress remains limited in low-resource languages where limited linguistic resources and cultural differences affect how sexism is expressed and perceived. This study introduces the first Hausa sexism detection dataset, developed through community engagement, qualitative coding, and data augmentation. For cultural nuances and linguistic representation, we conducted a two-stage user study (n=66) involving native speakers to explore how sexism is defined and articulated in everyday discourse. We further experiment with both traditional machine learning classifiers and pre-trained multilingual language models and evaluating the effectiveness few-shot learning in detecting sexism in Hausa. Our findings highlight challenges in capturing cultural nuance, particularly with clarification-seeking and idiomatic expressions, and reveal a tendency for many false positives in such cases. 

**Abstract (ZH)**: 性别歧视通过传播刻板印象、偏见和歧视性规范固化性别不平等和社会排斥。鉴于在线平台使得各种形式的性别歧视得以滋生，有效检测和减轻性别歧视的策略需求日益增加。尽管在高资源语言中存在广泛的性别歧视检测计算方法，但在低资源语言中，由于有限的语言资源和文化差异影响性别歧视的表达和感知，进展仍然有限。本研究通过社区参与、定性编码和数据增广，引入了首个豪萨语性别歧视检测数据集。为探究文化细微差别和语言表达，我们进行了两阶段用户研究（n=66），邀请母语者探索性别歧视在日常 discourse 中的定义和表达方式。我们进一步尝试了传统机器学习分类器和预训练多语言语言模型，并评估了少样本学习在检测豪萨语性别歧视中的有效性。我们的研究结果强调了在捕捉文化细微差别方面面临的挑战，特别是在含糊不清和成语表达的情况下，发现了许多假阳性的情况。 

---
# A Framework for Fair Evaluation of Variance-Aware Bandit Algorithms 

**Title (ZH)**: 面向方差感知bandit算法公平评估的框架 

**Authors**: Elise Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2510.27001)  

**Abstract**: Multi-armed bandit (MAB) problems serve as a fundamental building block for more complex reinforcement learning algorithms. However, evaluating and comparing MAB algorithms remains challenging due to the lack of standardized conditions and replicability. This is particularly problematic for variance-aware extensions of classical methods like UCB, whose performance can heavily depend on the underlying environment. In this study, we address how performance differences between bandit algorithms can be reliably observed, and under what conditions variance-aware algorithms outperform classical ones. We present a reproducible evaluation designed to systematically compare eight classical and variance-aware MAB algorithms. The evaluation framework, implemented in our Bandit Playground codebase, features clearly defined experimental setups, multiple performance metrics (reward, regret, reward distribution, value-at-risk, and action optimality), and an interactive evaluation interface that supports consistent and transparent analysis. We show that variance-aware algorithms can offer advantages in settings with high uncertainty where the difficulty arises from subtle differences between arm rewards. In contrast, classical algorithms often perform equally well or better in more separable scenarios or if fine-tuned extensively. Our contributions are twofold: (1) a framework for systematic evaluation of MAB algorithms, and (2) insights into the conditions under which variance-aware approaches outperform their classical counterparts. 

**Abstract (ZH)**: 多臂-bandit (MAB) 问题作为更复杂强化学习算法的基本构建块至关重要。然而，由于缺乏标准化条件和可重复性，评估和比较MAB算法仍然具有挑战性。这特别对经典方法如UCB的方差感知扩展提出了问题，其性能可能强烈依赖于底层环境。在本研究中，我们探讨了如何可靠地观察不同bandit算法之间的性能差异，并在什么条件下方差感知算法优于经典算法。我们呈现了一个可重复的评估框架，旨在系统比较八种经典和方差感知MAB算法。该评估框架在我们的Bandit Playground代码库中实现，包含明确定义的实验设置、多种性能指标（奖励、遗憾、奖励分布、值-at-风险和动作优化性），以及支持一致和透明分析的交互式评估界面。我们展示了方差感知算法在高不确定性环境下具有优势，这些环境因手臂奖励之间的微妙差异而具有挑战性。相比之下，在更分离的场景或经过广泛微调的情况下，经典算法通常表现得同样好或更好。我们的贡献有两个方面：(1) 一种MAB算法系统的评估框架，(2) 方差感知方法在什么条件下优于其经典对手的见解。 

---
# AIOT based Smart Education System: A Dual Layer Authentication and Context-Aware Tutoring Framework for Learning Environments 

**Title (ZH)**: 基于AIOT的智能教育系统：学习环境中基于双层认证和情景感知的辅导框架 

**Authors**: Adithya Neelakantan, Pratik Satpute, Prerna Shinde, Tejas Manjunatha Devang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26999)  

**Abstract**: The AIoT-Based Smart Education System integrates Artificial Intelligence and IoT to address persistent challenges in contemporary classrooms: attendance fraud, lack of personalization, student disengagement, and inefficient resource use. The unified platform combines four core modules: (1) a dual-factor authentication system leveraging RFID-based ID scans and WiFi verification for secure, fraud-resistant attendance; (2) an AI-powered assistant that provides real-time, context-aware support and dynamic quiz generation based on instructor-supplied materials; (3) automated test generators to streamline adaptive assessment and reduce administrative overhead; and (4) the EcoSmart Campus module, which autonomously regulates classroom lighting, air quality, and temperature using IoT sensors and actuators. Simulated evaluations demonstrate the system's effectiveness in delivering robust real-time monitoring, fostering inclusive engagement, preventing fraudulent practices, and supporting operational scalability. Collectively, the AIoT-Based Smart Education System offers a secure, adaptive, and efficient learning environment, providing a scalable blueprint for future educational innovation and improved student outcomes through the synergistic application of artificial intelligence and IoT technologies. 

**Abstract (ZH)**: 基于AIoT的智能教育系统整合人工智能和物联网以应对当代教室中长期存在的挑战：考勤作弊、个性化不足、学生活动参与度低和资源使用效率低下。该统一平台结合了四个核心模块：（1）基于RFID身份扫描和WiFi验证的双因素认证系统，用于安全、防作弊的考勤；（2）基于人工智能的助手，提供实时、情境感知的支持和依据教师提供的材料生成的动态测验；（3）自动化测试生成器，以简化适应性评估并减少行政负担；以及（4）EcoSmart校园模块，利用物联网传感器和执行器自主调节教室照明、空气质量及温度。模拟评估验证了该系统的有效性，能够在实时监控、促进包容性参与、防止欺诈行为以及支持操作扩展方面发挥重要作用。总体而言，基于AIoT的智能教育系统提供了一个安全、适应性强且高效的的学习环境，为未来教育创新和通过人工智能与物联网技术的协同应用提高学生成果提供了可扩展的蓝图。 

---
# Fine-Grained Iterative Adversarial Attacks with Limited Computation Budget 

**Title (ZH)**: 细粒度迭代对抗攻击在有限计算预算下 

**Authors**: Zhichao Hou, Weizhi Gao, Xiaorui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26981)  

**Abstract**: This work tackles a critical challenge in AI safety research under limited compute: given a fixed computation budget, how can one maximize the strength of iterative adversarial attacks? Coarsely reducing the number of attack iterations lowers cost but substantially weakens effectiveness. To fulfill the attainable attack efficacy within a constrained budget, we propose a fine-grained control mechanism that selectively recomputes layer activations across both iteration-wise and layer-wise levels. Extensive experiments show that our method consistently outperforms existing baselines at equal cost. Moreover, when integrated into adversarial training, it attains comparable performance with only 30% of the original budget. 

**Abstract (ZH)**: 在有限计算资源下最大化迭代对抗攻击效果的研究：一种细粒度控制机制 

---
# Overview of the MEDIQA-OE 2025 Shared Task on Medical Order Extraction from Doctor-Patient Consultations 

**Title (ZH)**: 2025年MEDIQA-OE共享任务：从医生-患者咨询中提取医疗 orders 概览 

**Authors**: Jean-Philippe Corbeil, Asma Ben Abacha, Jerome Tremblay, Phillip Swazinna, Akila Jeeson Daniel, Miguel Del-Agua, Francois Beaulieu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26974)  

**Abstract**: Clinical documentation increasingly uses automatic speech recognition and summarization, yet converting conversations into actionable medical orders for Electronic Health Records remains unexplored. A solution to this problem can significantly reduce the documentation burden of clinicians and directly impact downstream patient care. We introduce the MEDIQA-OE 2025 shared task, the first challenge on extracting medical orders from doctor-patient conversations. Six teams participated in the shared task and experimented with a broad range of approaches, and both closed- and open-weight large language models (LLMs). In this paper, we describe the MEDIQA-OE task, dataset, final leaderboard ranking, and participants' solutions. 

**Abstract (ZH)**: 临床文档越来越多地使用自动语音识别和总结技术，然而将对话转换为电子健康记录中的可操作医疗订单仍然未被探索。解决这一问题可以显著减轻临床人员的文档负担，并直接影响下游患者的护理。我们介绍了MEDIQA-OE 2025共享任务，这是首次针对从医生-患者对话中抽取医疗订单的挑战。六支队伍参与了共享任务，并尝试了广泛的方法，包括闭合权重和开放权重的大语言模型。在本文中，我们描述了MEDIQA-OE任务、数据集、最终排行榜以及参赛队伍的解决方案。 

---
# Frame Semantic Patterns for Identifying Underreporting of Notifiable Events in Healthcare: The Case of Gender-Based Violence 

**Title (ZH)**: 基于框架语义模式识别医疗保健中可报告事件上报不足的情况：以基于性别的暴力为例 

**Authors**: Lívia Dutra, Arthur Lorenzi, Laís Berno, Franciany Campos, Karoline Biscardi, Kenneth Brown, Marcelo Viridiano, Frederico Belcavello, Ely Matos, Olívia Guaranha, Erik Santos, Sofia Reinach, Tiago Timponi Torrent  

**Link**: [PDF](https://arxiv.org/pdf/2510.26969)  

**Abstract**: We introduce a methodology for the identification of notifiable events in the domain of healthcare. The methodology harnesses semantic frames to define fine-grained patterns and search them in unstructured data, namely, open-text fields in e-medical records. We apply the methodology to the problem of underreporting of gender-based violence (GBV) in e-medical records produced during patients' visits to primary care units. A total of eight patterns are defined and searched on a corpus of 21 million sentences in Brazilian Portuguese extracted from e-SUS APS. The results are manually evaluated by linguists and the precision of each pattern measured. Our findings reveal that the methodology effectively identifies reports of violence with a precision of 0.726, confirming its robustness. Designed as a transparent, efficient, low-carbon, and language-agnostic pipeline, the approach can be easily adapted to other health surveillance contexts, contributing to the broader, ethical, and explainable use of NLP in public health systems. 

**Abstract (ZH)**: 一种在医疗领域识别可报告事件的方法及其在电子医疗记录中基于性别的暴力报告识别中的应用 

---
# Can machines think efficiently? 

**Title (ZH)**: 机器能高效思考吗？ 

**Authors**: Adam Winchell  

**Link**: [PDF](https://arxiv.org/pdf/2510.26954)  

**Abstract**: The Turing Test is no longer adequate for distinguishing human and machine intelligence. With advanced artificial intelligence systems already passing the original Turing Test and contributing to serious ethical and environmental concerns, we urgently need to update the test. This work expands upon the original imitation game by accounting for an additional factor: the energy spent answering the questions. By adding the constraint of energy, the new test forces us to evaluate intelligence through the lens of efficiency, connecting the abstract problem of thinking to the concrete reality of finite resources. Further, this proposed new test ensures the evaluation of intelligence has a measurable, practical finish line that the original test lacks. This additional constraint compels society to weigh the time savings of using artificial intelligence against its total resource cost. 

**Abstract (ZH)**: 图灵测试已不足以区分人类和机器智能。鉴于先进人工智能系统已通过原始图灵测试并引发严重伦理和环境问题，我们迫切需要更新这一测试。本研究在原始模仿游戏中引入了额外的因素：回答问题所消耗的能源。通过增加能源约束，新模式试要求我们从效率的角度评估智能，将抽象的思考问题与有限资源的现实联系起来。此外，这一提出的新型测试确保了智能评估有一个可度量且实用的终点，这是原始测试所缺乏的。这一额外的约束促使社会权衡使用人工智能节省的时间与总体资源成本之间的关系。 

---
# Mind the Gaps: Auditing and Reducing Group Inequity in Large-Scale Mobility Prediction 

**Title (ZH)**: 注意差距：审计和减少大规模移动预测中的群体不公正性 

**Authors**: Ashwin Kumar, Hanyu Zhang, David A. Schweidel, William Yeoh  

**Link**: [PDF](https://arxiv.org/pdf/2510.26940)  

**Abstract**: Next location prediction underpins a growing number of mobility, retail, and public-health applications, yet its societal impacts remain largely unexplored. In this paper, we audit state-of-the-art mobility prediction models trained on a large-scale dataset, highlighting hidden disparities based on user demographics. Drawing from aggregate census data, we compute the difference in predictive performance on racial and ethnic user groups and show a systematic disparity resulting from the underlying dataset, resulting in large differences in accuracy based on location and user groups. To address this, we propose Fairness-Guided Incremental Sampling (FGIS), a group-aware sampling strategy designed for incremental data collection settings. Because individual-level demographic labels are unavailable, we introduce Size-Aware K-Means (SAKM), a clustering method that partitions users in latent mobility space while enforcing census-derived group proportions. This yields proxy racial labels for the four largest groups in the state: Asian, Black, Hispanic, and White. Built on these labels, our sampling algorithm prioritizes users based on expected performance gains and current group representation. This method incrementally constructs training datasets that reduce demographic performance gaps while preserving overall accuracy. Our method reduces total disparity between groups by up to 40\% with minimal accuracy trade-offs, as evaluated on a state-of-art MetaPath2Vec model and a transformer-encoder model. Improvements are most significant in early sampling stages, highlighting the potential for fairness-aware strategies to deliver meaningful gains even in low-resource settings. Our findings expose structural inequities in mobility prediction pipelines and demonstrate how lightweight, data-centric interventions can improve fairness with little added complexity, especially for low-data applications. 

**Abstract (ZH)**: 基于大规模数据训练的最新移动预测模型在不同用户群体间隐含差异未被充分探索。本文审查了这些模型，揭示了基于用户人口统计学的隐藏不平等。通过汇总的人口普查数据，我们计算了不同种族和 Ethnic 用户群体的预测性能差异，并展示了由于底层数据集导致的系统性差异，从而引起了基于位置和用户群体的准确性差异。为解决这一问题，我们提出了一种公平指导的增量采样（FGIS）方法，这是一种针对增量数据收集场景设计的群体意识采样策略。由于无法获取个体级别的人口统计标签，我们引入了一种大小感知的 K-均值聚类方法（SAKM），该方法在潜在移动空间中分区用户，同时强制执行人口普查推断的群体比例。这为州内四个最大的群体（亚裔、非裔、拉丁裔和白人）提供了代理种族标签。基于这些标签，我们的采样算法根据预期性能提升和当前群体代表性来优先选择用户。这种方法逐步构建训练数据集，以减少人口统计学性能差距，同时保持总体准确性。在评估 MetaPath2Vec 模型和 transformer-编码模型时，我们的方法将组间的总不平等性降低了最多40%，且对准确性影响微乎其微。早期采样阶段的改进最为显著，强调了公平意识策略即使在资源有限的情况下也能带来显著的改进潜力。我们的研究揭示了移动预测管道中的结构性不平等，并展示了轻量级的数据导向干预措施如何在增加复杂性极小的情况下改善公平性，特别是在数据稀缺的应用场景中。 

---
# Accurate Target Privacy Preserving Federated Learning Balancing Fairness and Utility 

**Title (ZH)**: 准确的目标隐私保护联邦学习：平衡公平性和效用 

**Authors**: Kangkang Sun, Jun Wu, Minyi Guo, Jianhua Li, Jianwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26841)  

**Abstract**: Federated Learning (FL) enables collaborative model training without data sharing, yet participants face a fundamental challenge, e.g., simultaneously ensuring fairness across demographic groups while protecting sensitive client data. We introduce a differentially private fair FL algorithm (\textit{FedPF}) that transforms this multi-objective optimization into a zero-sum game where fairness and privacy constraints compete against model utility. Our theoretical analysis reveals a surprising inverse relationship, i.e., stricter privacy protection fundamentally limits the system's ability to detect and correct demographic biases, creating an inherent tension between privacy and fairness. Counterintuitively, we prove that moderate fairness constraints initially improve model generalization before causing performance degradation, where a non-monotonic relationship that challenges conventional wisdom about fairness-utility tradeoffs. Experimental validation demonstrates up to 42.9 % discrimination reduction across three datasets while maintaining competitive accuracy, but more importantly, reveals that the privacy-fairness tension is unavoidable, i.e., achieving both objectives simultaneously requires carefully balanced compromises rather than optimization of either in isolation. The source code for our proposed algorithm is publicly accessible at this https URL. 

**Abstract (ZH)**: 联邦学习（FL）允许在不共享数据的情况下进行协作模型训练，但参与者面临着一个基本挑战，即在保护敏感客户端数据的同时，确保跨不同人口组的公平性。我们提出了一种基于差分隐私的公平联邦学习算法（FedPF），将多目标优化问题转化为公平性和隐私约束与模型性能之间的零和博弈。我们的理论分析揭示了一个令人意外的反相关关系，即更为严格的隐私保护从根本上限制了系统检测和纠正人口偏差的能力，从而在隐私与公平之间产生了固有的矛盾。出乎意料的是，我们证明了适度的公平性约束最初会改善模型泛化能力，但在后续过程中导致性能下降，形成了一个挑战传统公平性-性能权衡观念的非单调关系。实验验证结果显示，在保持竞争力的同时，该算法在三个数据集上实现了高达42.9%的歧视减少，更重要的是，揭示了隐私-公平性权衡不可避免，即同时实现双重目标需要精心平衡的妥协，而非单独优化任一方。我们所提出的算法的源代码可在以下网址公开访问：this https URL。 

---
# SpotIt: Evaluating Text-to-SQL Evaluation with Formal Verification 

**Title (ZH)**: SpotIt: 使用形式验证评估文本到SQL转换 

**Authors**: Rocky Klopfenstein, Yang He, Andrew Tremante, Yuepeng Wang, Nina Narodytska, Haoze Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26840)  

**Abstract**: Community-driven Text-to-SQL evaluation platforms play a pivotal role in tracking the state of the art of Text-to-SQL performance. The reliability of the evaluation process is critical for driving progress in the field. Current evaluation methods are largely test-based, which involves comparing the execution results of a generated SQL query and a human-labeled ground-truth on a static test database. Such an evaluation is optimistic, as two queries can coincidentally produce the same output on the test database while actually being different. In this work, we propose a new alternative evaluation pipeline, called SpotIt, where a formal bounded equivalence verification engine actively searches for a database that differentiates the generated and ground-truth SQL queries. We develop techniques to extend existing verifiers to support a richer SQL subset relevant to Text-to-SQL. A performance evaluation of ten Text-to-SQL methods on the high-profile BIRD dataset suggests that test-based methods can often overlook differences between the generated query and the ground-truth. Further analysis of the verification results reveals a more complex picture of the current Text-to-SQL evaluation. 

**Abstract (ZH)**: 社区驱动的文本到SQL评估平台在追踪文本到SQL性能的最新状态中发挥着关键作用。评估过程的可靠性对于推动该领域的发展至关重要。当前的评估方法主要是基于测试的方法，涉及将生成的SQL查询的执行结果与人工标注的_ground-truth_在静态测试数据库上的结果进行比较。此类评估可能过于乐观，因为两个查询可能在测试数据库上巧合地产生相同的结果，但实际上却是不同的。在此工作中，我们提出了一种新的替代评估管道，称为SpotIt，其中正式边界等价性验证引擎主动寻找一个能够区分生成的和地标的SQL查询的数据库。我们开发了技术，将现有的验证器扩展以支持与文本到SQL相关的更丰富的SQL子集。对包括高度关注的BIRD数据集在内的十个文本到SQL方法的性能评估表明，基于测试的方法经常忽略生成查询与地标的差异。进一步分析验证结果揭示了当前文本到SQL评估更为复杂的情况。 

---
# R3GAN-based Optimal Strategy for Augmenting Small Medical Dataset 

**Title (ZH)**: 基于R3GAN的优化策略以扩充小型医疗数据集 

**Authors**: Tsung-Wei Pan, Chang-Hong Wu, Jung-Hua Wang, Ming-Jer Chen, Yu-Chiao Yi, Tsung-Hsien Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.26828)  

**Abstract**: Medical image analysis often suffers from data scarcity and class imbalance, limiting the effectiveness of deep learning models in clinical applications. Using human embryo time-lapse imaging (TLI) as a case study, this work investigates how generative adversarial networks (GANs) can be optimized for small datasets to generate realistic and diagnostically meaningful images. Based on systematic experiments with R3GAN, we established effective training strategies and designed an optimized configuration for 256x256-resolution datasets, featuring a full burn-in phase and a low, gradually increasing gamma range (5 -> 40). The generated samples were used to balance an imbalanced embryo dataset, leading to substantial improvement in classification performance. The recall and F1-score of t3 increased from 0.06 to 0.69 and 0.11 to 0.60, respectively, without compromising other classes. These results demonstrate that tailored R3GAN training strategies can effectively alleviate data scarcity and improve model robustness in small-scale medical imaging tasks. 

**Abstract (ZH)**: 医学图像分析往往受到数据稀缺性和类别不平衡的限制，这限制了深度学习模型在临床应用中的有效性。以人类胚胎时间 lapse 成像（TLI）为例，本工作探讨了如何通过生成对抗网络（GANs）优化小数据集以生成具有诊断意义的现实图像。基于R3GAN的系统实验，我们制定了有效的训练策略，并为256x256分辨率的数据集设计了优化配置，包括完整的预热阶段和低、逐渐增加的伽马范围（5 -> 40）。生成的样本用于平衡不平衡的胚胎数据集，显著提高了分类性能。t3的召回率和F1分数分别从0.06提高到0.69和0.11提高到0.60，而其他类别的性能保持不变。这些结果表明，定制的R3GAN训练策略可以有效地缓解数据稀缺性并提高小型医学成像任务中模型的鲁棒性。 

---
# LeMat-Synth: a multi-modal toolbox to curate broad synthesis procedure databases from scientific literature 

**Title (ZH)**: LeMat-Synth：多模态工具箱，用于从科学文献中整理广泛的合成程序数据库 

**Authors**: Magdalena Lederbauer, Siddharth Betala, Xiyao Li, Ayush Jain, Amine Sehaba, Georgia Channing, Grégoire Germain, Anamaria Leonescu, Faris Flaifil, Alfonso Amayuelas, Alexandre Nozadze, Stefan P. Schmid, Mohd Zaki, Sudheesh Kumar Ethirajan, Elton Pan, Mathilde Franckel, Alexandre Duval, N. M. Anoop Krishnan, Samuel P. Gleason  

**Link**: [PDF](https://arxiv.org/pdf/2510.26824)  

**Abstract**: The development of synthesis procedures remains a fundamental challenge in materials discovery, with procedural knowledge scattered across decades of scientific literature in unstructured formats that are challenging for systematic analysis. In this paper, we propose a multi-modal toolbox that employs large language models (LLMs) and vision language models (VLMs) to automatically extract and organize synthesis procedures and performance data from materials science publications, covering text and figures. We curated 81k open-access papers, yielding LeMat-Synth (v 1.0): a dataset containing synthesis procedures spanning 35 synthesis methods and 16 material classes, structured according to an ontology specific to materials science. The extraction quality is rigorously evaluated on a subset of 2.5k synthesis procedures through a combination of expert annotations and a scalable LLM-as-a-judge framework. Beyond the dataset, we release a modular, open-source software library designed to support community-driven extension to new corpora and synthesis domains. Altogether, this work provides an extensible infrastructure to transform unstructured literature into machine-readable information. This lays the groundwork for predictive modeling of synthesis procedures as well as modeling synthesis--structure--property relationships. 

**Abstract (ZH)**: 材料科学合成程序的开发仍然是材料发现中的一个基本挑战，相关知识散落于数十年的科学文献中，以无结构格式存在，难以系统分析。本文提出一个多模态工具箱，利用大规模语言模型（LLMs）和视觉语言模型（VLMs）自动从材料科学出版物中提取和组织合成程序和性能数据，涵盖文本和图表。我们整理了81,000篇开放获取论文，生成了LeMat-Synth（v 1.0）：一个包含35种合成方法和16种材料类别合成程序的数据集，按照特定于材料科学的本体进行结构化。通过专家注解和可扩展的LLM作为法官框架，对2,500条合成程序的子集进行了提取质量的严格评估。除了数据集，我们还发布了模块化开源软件库，旨在支持社区驱动的新语料库和合成领域的扩展。整体而言，这项工作提供了一个可扩展的基础设施，将无结构文献转化为可机器读取的信息。这为合成程序的预测建模以及合成-结构-性能关系的建模奠定了基础。 

---
# Cross-Corpus Validation of Speech Emotion Recognition in Urdu using Domain-Knowledge Acoustic Features 

**Title (ZH)**: 使用领域知识声学特征的乌尔都语语音情感识别跨语料库验证 

**Authors**: Unzela Talpur, Zafi Sherhan Syed, Muhammad Shehram Shah Syed, Abbas Shah Syed  

**Link**: [PDF](https://arxiv.org/pdf/2510.26823)  

**Abstract**: Speech Emotion Recognition (SER) is a key affective computing technology that enables emotionally intelligent artificial intelligence. While SER is challenging in general, it is particularly difficult for low-resource languages such as Urdu. This study investigates Urdu SER in a cross-corpus setting, an area that has remained largely unexplored. We employ a cross-corpus evaluation framework across three different Urdu emotional speech datasets to test model generalization. Two standard domain-knowledge based acoustic feature sets, eGeMAPS and ComParE, are used to represent speech signals as feature vectors which are then passed to Logistic Regression and Multilayer Perceptron classifiers. Classification performance is assessed using unweighted average recall (UAR) whilst considering class-label imbalance. Results show that Self-corpus validation often overestimates performance, with UAR exceeding cross-corpus evaluation by up to 13%, underscoring that cross-corpus evaluation offers a more realistic measure of model robustness. Overall, this work emphasizes the importance of cross-corpus validation for Urdu SER and its implications contribute to advancing affective computing research for underrepresented language communities. 

**Abstract (ZH)**: 乌尔都语语音情感识别中的跨语料库研究 

---
# Systematic Absence of Low-Confidence Nighttime Fire Detections in VIIRS Active Fire Product: Evidence of Undocumented Algorithmic Filtering 

**Title (ZH)**: VIIRS活性火产品中低置信度夜间火灾检测系统性缺失的证据：未记录的算法过滤现象 

**Authors**: Rohit Rajendra Dhage  

**Link**: [PDF](https://arxiv.org/pdf/2510.26816)  

**Abstract**: The Visible Infrared Imaging Radiometer Suite (VIIRS) active fire product is widely used for global fire monitoring, yet its confidence classification scheme exhibits an undocumented systematic pattern. Through analysis of 21,540,921 fire detections spanning one year (January 2023 - January 2024), I demonstrate a complete absence of low-confidence classifications during nighttime observations. Of 6,007,831 nighttime fires, zero were classified as low confidence, compared to an expected 696,908 under statistical independence (chi-squared = 1,474,795, p < 10^-15, Z = -833). This pattern persists globally across all months, latitude bands, and both NOAA-20 and Suomi-NPP satellites. Machine learning reverse-engineering (88.9% accuracy), bootstrap simulation (1,000 iterations), and spatial-temporal analysis confirm this is an algorithmic constraint rather than a geophysical phenomenon. Brightness temperature analysis reveals nighttime fires below approximately 295K are likely excluded entirely rather than flagged as low-confidence, while daytime fires show normal confidence distributions. This undocumented behavior affects 27.9% of all VIIRS fire detections and has significant implications for fire risk assessment, day-night detection comparisons, confidence-weighted analyses, and any research treating confidence levels as uncertainty metrics. I recommend explicit documentation of this algorithmic constraint in VIIRS user guides and reprocessing strategies for affected analyses. 

**Abstract (ZH)**: VIIRS可见红外成像辐射计套件主动火产品存在未记录的系统性模式：夜间观测中低置信度分类完全缺失及其影响 

---
# Impact of clinical decision support systems (cdss) on clinical outcomes and healthcare delivery in low- and middle-income countries: protocol for a systematic review and meta-analysis 

**Title (ZH)**: 临床决策支持系统（CDSS）对低收入和中等收入国家临床结果和医疗保健交付的影响：系统评价和meta分析的方案 

**Authors**: Garima Jain, Anand Bodade, Sanghamitra Pati  

**Link**: [PDF](https://arxiv.org/pdf/2510.26812)  

**Abstract**: Clinical decision support systems (CDSS) are used to improve clinical and service outcomes, yet evidence from low- and middle-income countries (LMICs) is dispersed. This protocol outlines methods to quantify the impact of CDSS on patient and healthcare delivery outcomes in LMICs. We will include comparative quantitative designs (randomized trials, controlled before-after, interrupted time series, comparative cohorts) evaluating CDSS in World Bank-defined LMICs. Standalone qualitative studies are excluded; mixed-methods studies are eligible only if they report comparative quantitative outcomes, for which we will extract the quantitative component. Searches (from inception to 30 September 2024) will cover MEDLINE, Embase, CINAHL, CENTRAL, Web of Science, Global Health, Scopus, IEEE Xplore, LILACS, African Index Medicus, and IndMED, plus grey sources. Screening and extraction will be performed in duplicate. Risk of bias will be assessed with RoB 2 (randomized trials) and ROBINS-I (non-randomized). Random-effects meta-analysis will be performed where outcomes are conceptually or statistically comparable; otherwise, a structured narrative synthesis will be presented. Heterogeneity will be explored using relative and absolute metrics and a priori subgroups or meta-regression (condition area, care level, CDSS type, readiness proxies, study design). 

**Abstract (ZH)**: 临床决策支持系统（CDSS）在改善临床和服务结果方面得到应用，然而低收入和中等收入国家（LMICs）的相关证据分散。本研究方案概述了量化CDSS对LMICs患者和医疗服务结果影响的方法。我们将包括世界银行定义的LMICs中评估CDSS的比较定量设计（随机试验、控制前后设计、中断时间序列、比较队列研究）。独立定性研究将被排除；只有报告比较定量结果的混合方法研究才符合条件，我们将提取其中的定量部分。文献检索（从创刊号到2024年9月30日）将覆盖MEDLINE、Embase、CINAHL、CENTRAL、Web of Science、Global Health、Scopus、IEEE Xplore、LILACS、African Index Medicus和IndMED，以及灰色文献。筛查和数据提取将进行双重操作。偏倚风险将使用RoB 2（随机试验）和ROBINS-I（非随机试验）评估。当结果在概念或统计上可比时，将进行随机效应元分析；否则，将呈现结构化的综述合成。异质性将通过相对和绝对指标以及先验子组或meta回归（条件领域、护理级别、CDSS类型、准备度指标、研究设计）进行探索。 

---
# EARS-UDE: Evaluating Auditory Response in Sensory Overload with Universal Differential Equations 

**Title (ZH)**: EARS-UDE: 评估感官超载中听觉反应的通用微分方程方法 

**Authors**: Miheer Salunke, Prathamesh Dinesh Joshi, Raj Abhijit Dandekar, Rajat Dandekar, Sreedath Panat  

**Link**: [PDF](https://arxiv.org/pdf/2510.26804)  

**Abstract**: Auditory sensory overload affects 50-70% of individuals with Autism Spectrum Disorder (ASD), yet existing approaches, such as mechanistic models (Hodgkin Huxley type, Wilson Cowan, excitation inhibition balance), clinical tools (EEG/MEG, Sensory Profile scales), and ML methods (Neural ODEs, predictive coding), either assume fixed parameters or lack interpretability, missing autism heterogeneity. We present a Scientific Machine Learning approach using Universal Differential Equations (UDEs) to model sensory adaptation dynamics in autism. Our framework combines ordinary differential equations grounded in biophysics with neural networks to capture both mechanistic understanding and individual variability. We demonstrate that UDEs achieve a 90.8% improvement over pure Neural ODEs while using 73.5% fewer parameters. The model successfully recovers physiological parameters within the 2% error and provides a quantitative risk assessment for sensory overload, predicting 17.2% risk for pulse stimuli with specific temporal patterns. This framework establishes foundations for personalized, evidence-based interventions in autism, with direct applications to wearable technology and clinical practice. 

**Abstract (ZH)**: 听觉感觉过载影响自闭症谱系障碍（ASD）个体的50-70%，现有方法如机制模型（Hodgkin Huxley类型、Wilson Cowan模型、兴奋与抑制平衡）、临床工具（EEG/MEG、感觉量表）及机器学习方法（神经ODEs、预测编码）要么假设固定参数，要么缺乏解释性，未能捕捉自闭症异质性。我们提出了一种使用通用微分方程（UDEs）的科学机器学习方法，以建模自闭症的感觉适应动力学。该框架结合了源自生物物理的常微分方程和神经网络，以捕捉机制理解与个体变异性。研究表明，UDEs在参数减少73.5%的情况下，相比纯神经ODEs实现了90.8%的性能提升。该模型成功地在2%误差范围内恢复了生理参数，并提供了感觉过载的定量风险评估，预测特定时间模式的脉冲刺激有17.2%的风险。该框架为自闭症个性化、证据为基础的干预奠定了基础，并直接应用于可穿戴技术及临床实践。 

---
# VeriStruct: AI-assisted Automated Verification of Data-Structure Modules in Verus 

**Title (ZH)**: VeriStruct: AI辅助的数据结构模块自动验证在Verus中的应用 

**Authors**: Chuyue Sun, Yican Sun, Daneshvar Amrollahi, Ethan Zhang, Shuvendu Lahiri, Shan Lu, David Dill, Clark Barrett  

**Link**: [PDF](https://arxiv.org/pdf/2510.25015)  

**Abstract**: We introduce VeriStruct, a novel framework that extends AI-assisted automated verification from single functions to more complex data structure modules in Verus. VeriStruct employs a planner module to orchestrate the systematic generation of abstractions, type invariants, specifications, and proof code. To address the challenge that LLMs often misunderstand Verus' annotation syntax and verification-specific semantics, VeriStruct embeds syntax guidance within prompts and includes a repair stage to automatically correct annotation errors. In an evaluation on eleven Rust data structure modules, VeriStruct succeeds on ten of the eleven, successfully verifying 128 out of 129 functions (99.2%) in total. These results represent an important step toward the goal of automatic AI-assisted formal verification. 

**Abstract (ZH)**: 我们介绍了VeriStruct，这是一种新型框架，它将AI辅助自动化验证从单个函数扩展到Verus中更复杂的数据结构模块。VeriStruct采用规划器模块来协调抽象、类型不变式、规范和证明代码的系统生成。为了应对LLMs常误解Verus注解语法和验证特定语义的挑战，VeriStruct在提示中嵌入了语法指导，并包含一个修复阶段以自动纠正注解错误。在对 eleven 个 Rust 数据结构模块的评估中，VeriStruct 在 eleven 个模块中有十个成功，总共成功验证了 129 个函数中的 128 个（99.2%）。这些结果代表了朝着自动AI辅助形式化验证目标迈出的重要一步。 

---
# A Transformer-based Neural Architecture Search Method 

**Title (ZH)**: 基于Transformer的神经架构搜索方法 

**Authors**: Shang Wang, Huanrong Tang, Jianquan Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2505.01314)  

**Abstract**: This paper presents a neural architecture search method based on Transformer architecture, searching cross multihead attention computation ways for different number of encoder and decoder combinations. In order to search for neural network structures with better translation results, we considered perplexity as an auxiliary evaluation metric for the algorithm in addition to BLEU scores and iteratively improved each individual neural network within the population by a multi-objective genetic algorithm. Experimental results show that the neural network structures searched by the algorithm outperform all the baseline models, and that the introduction of the auxiliary evaluation metric can find better models than considering only the BLEU score as an evaluation metric. 

**Abstract (ZH)**: 基于Transformer架构的神经架构搜索方法：跨多头注意力机制的编码器-解码器组合搜索 

---
# A Neural Architecture Search Method using Auxiliary Evaluation Metric based on ResNet Architecture 

**Title (ZH)**: 基于ResNet架构的辅助评价指标神经架构搜索方法 

**Authors**: Shang Wang, Huanrong Tang, Jianquan Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2505.01313)  

**Abstract**: This paper proposes a neural architecture search space using ResNet as a framework, with search objectives including parameters for convolution, pooling, fully connected layers, and connectivity of the residual network. In addition to recognition accuracy, this paper uses the loss value on the validation set as a secondary objective for optimization. The experimental results demonstrate that the search space of this paper together with the optimisation approach can find competitive network architectures on the MNIST, Fashion-MNIST and CIFAR100 datasets. 

**Abstract (ZH)**: 本文提出了一种以ResNet为准架构的神经架构搜索空间，搜索目标包括卷积参数、池化参数、全连接层参数以及残差网络的连接性。除了识别精度外，本文还将验证集的损失值作为优化的次要目标。实验结果表明，本文提出的搜索空间与优化方法可以在MNIST、Fashion-MNIST和CIFAR100数据集上找到具有竞争力的网络架构。 

---
