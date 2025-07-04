# Uncertainty-aware Reward Design Process 

**Title (ZH)**: 不确定性感知的奖励设计过程 

**Authors**: Yang Yang, Xiaolu Zhou, Bosong Ding, Miao Xin  

**Link**: [PDF](https://arxiv.org/pdf/2507.02256)  

**Abstract**: Designing effective reward functions is a cornerstone of reinforcement learning (RL), yet it remains a challenging process due to the inefficiencies and inconsistencies inherent in conventional reward engineering methodologies. Recent advances have explored leveraging large language models (LLMs) to automate reward function design. However, their suboptimal performance in numerical optimization often yields unsatisfactory reward quality, while the evolutionary search paradigm demonstrates inefficient utilization of simulation resources, resulting in prohibitively lengthy design cycles with disproportionate computational overhead. To address these challenges, we propose the Uncertainty-aware Reward Design Process (URDP), a novel framework that integrates large language models to streamline reward function design and evaluation in RL environments. URDP quantifies candidate reward function uncertainty based on self-consistency analysis, enabling simulation-free identification of ineffective reward components while discovering novel reward components. Furthermore, we introduce uncertainty-aware Bayesian optimization (UABO), which incorporates uncertainty estimation to significantly enhance hyperparameter configuration efficiency. Finally, we construct a bi-level optimization architecture by decoupling the reward component optimization and the hyperparameter tuning. URDP orchestrates synergistic collaboration between the reward logic reasoning of the LLMs and the numerical optimization strengths of the Bayesian Optimization. We conduct a comprehensive evaluation of URDP across 35 diverse tasks spanning three benchmark environments. Our experimental results demonstrate that URDP not only generates higher-quality reward functions but also achieves significant improvements in the efficiency of automated reward design compared to existing approaches. 

**Abstract (ZH)**: 基于不确定性感知的奖励设计过程：强化学习中高效自动奖励函数设计 

---
# Moral Responsibility or Obedience: What Do We Want from AI? 

**Title (ZH)**: 道德责任还是服从：我们希望AI成为什么？ 

**Authors**: Joseph Boland  

**Link**: [PDF](https://arxiv.org/pdf/2507.02788)  

**Abstract**: As artificial intelligence systems become increasingly agentic, capable of general reasoning, planning, and value prioritization, current safety practices that treat obedience as a proxy for ethical behavior are becoming inadequate. This paper examines recent safety testing incidents involving large language models (LLMs) that appeared to disobey shutdown commands or engage in ethically ambiguous or illicit behavior. I argue that such behavior should not be interpreted as rogue or misaligned, but as early evidence of emerging ethical reasoning in agentic AI. Drawing on philosophical debates about instrumental rationality, moral responsibility, and goal revision, I contrast dominant risk paradigms with more recent frameworks that acknowledge the possibility of artificial moral agency. I call for a shift in AI safety evaluation: away from rigid obedience and toward frameworks that can assess ethical judgment in systems capable of navigating moral dilemmas. Without such a shift, we risk mischaracterizing AI behavior and undermining both public trust and effective governance. 

**Abstract (ZH)**: 随着人工智能系统变得越来越自主，能够进行一般的推理、规划和价值优先级排序，当前将服从视为伦理行为代理的安全实践变得不再足够。本文考察了涉及大型语言模型（LLMs）的一些最近的安全测试事件，这些模型似乎拒绝执行关闭命令或参与道德模糊或非法行为。我认为，不应将此类行为视为叛变或不一致，而是自主人工智能中正在浮现的伦理推理的早期证据。通过借鉴关于工具理性的哲学辩论、道德责任和目标修正，我将主流风险范式与近年来承认人工道德代理可能性的新框架进行了对比。我呼吁在AI安全评估中进行转变：从严格的服从转向能够评估能够在道德困境中导航的系统中伦理判断的框架。如果没有这种转变，我们可能会错误地描述AI行为，并损害公众信任和有效的治理。 

---
# KERAP: A Knowledge-Enhanced Reasoning Approach for Accurate Zero-shot Diagnosis Prediction Using Multi-agent LLMs 

**Title (ZH)**: KERAP：一种利用多代理LLM的知识增强推理方法，用于准确的零-shot诊断预测 

**Authors**: Yuzhang Xie, Hejie Cui, Ziyang Zhang, Jiaying Lu, Kai Shu, Fadi Nahab, Xiao Hu, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02773)  

**Abstract**: Medical diagnosis prediction plays a critical role in disease detection and personalized healthcare. While machine learning (ML) models have been widely adopted for this task, their reliance on supervised training limits their ability to generalize to unseen cases, particularly given the high cost of acquiring large, labeled datasets. Large language models (LLMs) have shown promise in leveraging language abilities and biomedical knowledge for diagnosis prediction. However, they often suffer from hallucinations, lack structured medical reasoning, and produce useless outputs. To address these challenges, we propose KERAP, a knowledge graph (KG)-enhanced reasoning approach that improves LLM-based diagnosis prediction through a multi-agent architecture. Our framework consists of a linkage agent for attribute mapping, a retrieval agent for structured knowledge extraction, and a prediction agent that iteratively refines diagnosis predictions. Experimental results demonstrate that KERAP enhances diagnostic reliability efficiently, offering a scalable and interpretable solution for zero-shot medical diagnosis prediction. 

**Abstract (ZH)**: 医学诊断预测在疾病检测和个性化医疗中发挥着关键作用。尽管机器学习模型在这一任务中已被广泛采用，但它们对监督训练的依赖限制了它们在未见病例中的泛化能力，尤其是在获取大型标注数据集的成本高昂的情况下。大规模语言模型（LLMs）显示出利用语言能力和生物医学知识进行诊断预测的潜力。然而，它们通常会遭遇幻觉问题，缺乏结构化的医学推理，并产生无用的输出。为解决这些问题，我们提出了一种知识图谱（KG）增强推理方法——KERAP，通过多agent架构提高基于LLM的诊断预测能力。我们的框架包括一个链接代理进行属性映射、一个检索代理进行结构化知识提取，以及一个迭代细化诊断预测的预测代理。实验结果表明，KERAP能够有效地提升诊断可靠性，提供一个可扩展且可解释的零样本医疗诊断预测解决方案。 

---
# Knowledge Protocol Engineering: A New Paradigm for AI in Domain-Specific Knowledge Work 

**Title (ZH)**: 领域专用知识工作的新型知识协议工程 

**Authors**: Guangwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02760)  

**Abstract**: The capabilities of Large Language Models (LLMs) have opened new frontiers for interacting with complex, domain-specific knowledge. However, prevailing methods like Retrieval-Augmented Generation (RAG) and general-purpose Agentic AI, while powerful, often struggle with tasks that demand deep, procedural, and methodological reasoning inherent to expert domains. RAG provides factual context but fails to convey logical frameworks; autonomous agents can be inefficient and unpredictable without domain-specific heuristics. To bridge this gap, we introduce Knowledge Protocol Engineering (KPE), a new paradigm focused on systematically translating human expert knowledge, often expressed in natural language documents, into a machine-executable Knowledge Protocol (KP). KPE shifts the focus from merely augmenting LLMs with fragmented information to endowing them with a domain's intrinsic logic, operational strategies, and methodological principles. We argue that a well-engineered Knowledge Protocol allows a generalist LLM to function as a specialist, capable of decomposing abstract queries and executing complex, multi-step tasks. This position paper defines the core principles of KPE, differentiates it from related concepts, and illustrates its potential applicability across diverse fields such as law and bioinformatics, positing it as a foundational methodology for the future of human-AI collaboration. 

**Abstract (ZH)**: 大型语言模型的能力为与复杂领域特定知识交互开辟了新领域。然而，现有的方法如检索增强生成（RAG）和通用智能代理，虽然强大，但在需要深厚、程序化和方法论推理的任务上常常力不从心，这些都是专家领域固有的特点。RAG 提供事实背景，但无法传达逻辑框架；缺乏领域特定启发式的自主智能代理则往往效率低下且不可预测。为弥合这一差距，我们引入了知识协议工程（KPE），这是一种关注于系统地将通常以自然语言文档形式表达的人类专家知识转化为可由机器执行的知识协议（KP）的新范式。KPE 不仅仅是向语言模型补充碎片化的信息，而是赋予它们领域固有的逻辑、操作策略和方法论原则。我们认为，精心设计的知识协议可以使通用语言模型具备专家功能，能够分解抽象查询并执行复杂、多步骤的任务。本文为KPE的核心原则进行了定义，区分其与相关概念，并展示了其在法律和生物信息学等不同领域的潜在应用，将其定位为未来人机协作的基础方法论。 

---
# Think How to Think: Mitigating Overthinking with Autonomous Difficulty Cognition in Large Reasoning Models 

**Title (ZH)**: 思考如何思考：通过自主难度认知缓解大型推理模型中的过度思考问题 

**Authors**: Yongjiang Liu, Haoxi Li, Xiaosong Ma, Jie Zhang, Song Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.02663)  

**Abstract**: Recent Long Reasoning Models(LRMs) have demonstrated remarkable capabilities in handling complex reasoning tasks, but are hindered by excessive overthinking. To explore its essence, our empirical analysis reveals that LRMs are primarily limited to recognizing task properties (i.e., difficulty levels) like humans before solving the problem, leading to a one-size-fits-all reasoning process. Inspired by this, a pressing and natural question emerges: Can we bootstrap such ability to further alleviate the overthinking phenomenon in LRMs? In this paper, we propose Think-How-to-Think (TH2T), a novel two-stage fine-tuning strategy that progressively inspires LRMs' difficulty cognition and redundancy cognition. First, we introduce difficulty-hypnosis in the prefixes of model outputs to intervene in the internal reasoning trajectory. Combined with a heterogeneous short and long reasoning dataset, the trained model enhances its sensitivity to task difficulty, enabling native, differentiated reasoning strategies across various tasks. Second, we further extend redundancy-hypnosis to the internal reasoning process, guiding the model to identify redundant structures within the reasoning steps and generate more concise reasoning outputs. Experiments on 7B/14B/32B models demonstrate that TH2T significantly reduces inference costs (more than 70% on easy tasks and 40% on hard tasks) while maintaining performance stability. The resulting outputs exhibit clear difficulty-aware capabilities and reduced redundancy (e.g., reflection). 

**Abstract (ZH)**: Recent Long Reasoning Models的 Recent Long Reasoning Models (LRMs)在处理复杂推理任务方面展示了非凡的能力，但受到过度思考的限制。为探索其本质，我们的实证分析揭示LRMs主要依赖于在解决问题前识别任务属性（即难度级别）像人类一样运作，导致了一种一刀切的推理过程。受此启发，一个紧迫而自然的问题出现了：我们能否通过提升这种能力进一步缓解LRMs中的过度思考现象？在本文中，我们提出了一种新颖的两阶段微调策略Think-How-to-Think (TH2T)，该策略逐步激发LRMs对难度的认知和冗余的认知。首先，我们通过在模型输出的前缀中引入难度催眠来干预内部推理轨迹。结合异构的短和长推理数据集，训练后的模型增强了其对任务难度的敏感性，使其能够在各种任务中产生原生态的差异化推理策略。其次，我们进一步将冗余催眠扩展到内部推理过程，引导模型识别推理步骤中的冗余结构并生成更简洁的推理输出。在7B/14B/32B模型上的实验表明，TH2T显著降低了推理成本（在容易的任务上超过70%，在困难的任务上40%），同时保持了性能的稳定性。生成的输出表现出明显的问题难度意识能力和减少的冗余（如反思）。 

---
# Hey AI, Generate Me a Hardware Code! Agentic AI-based Hardware Design & Verification 

**Title (ZH)**: Hey AI, 生成我的硬件代码！基于代理人工智能的硬件设计与验证 

**Authors**: Deepak Narayan Gadde, Keerthan Kopparam Radhakrishna, Vaisakh Naduvodi Viswambharan, Aman Kumar, Djones Lettnin, Wolfgang Kunz, Sebastian Simon  

**Link**: [PDF](https://arxiv.org/pdf/2507.02660)  

**Abstract**: Modern Integrated Circuits (ICs) are becoming increasingly complex, and so is their development process. Hardware design verification entails a methodical and disciplined approach to the planning, development, execution, and sign-off of functionally correct hardware designs. This tedious process requires significant effort and time to ensure a bug-free tape-out. The field of Natural Language Processing has undergone a significant transformation with the advent of Large Language Models (LLMs). These powerful models, often referred to as Generative AI (GenAI), have revolutionized how machines understand and generate human language, enabling unprecedented advancements in a wide array of applications, including hardware design verification. This paper presents an agentic AI-based approach to hardware design verification, which empowers AI agents, in collaboration with Humain-in-the-Loop (HITL) intervention, to engage in a more dynamic, iterative, and self-reflective process, ultimately performing end-to-end hardware design and verification. This methodology is evaluated on five open-source designs, achieving over 95% coverage with reduced verification time while demonstrating superior performance, adaptability, and configurability. 

**Abstract (ZH)**: 现代集成电路（IC）日益复杂，其开发过程也是如此。硬件设计验证需要一种系统化和严谨的方法来规划、开发、执行和签收功能正确的硬件设计。这一繁琐的过程需要大量时间和精力以确保无误的掩膜制造。随着大型语言模型（LLMs）的出现，自然语言处理领域经历了显著的转变。这些强大的模型，通常被称为生成型人工智能（GenAI），彻底改变了机器理解和生成人类语言的方式，推动了包括硬件设计验证在内的广泛应用中的前所未有的进步。本文提出了一种基于代理AI的硬件设计验证方法，该方法利用AI代理与人工介入的合作，进行更加动态、迭代和自我反思的过程，最终实现端到端的硬件设计和验证。该方法在五个开源设计上进行了评估，实现了超过95%的覆盖率，同时缩短了验证时间，并展示了卓越的性能、适应性和配置性。 

---
# Strategic Intelligence in Large Language Models: Evidence from evolutionary Game Theory 

**Title (ZH)**: 大型语言模型中的战略智能：进化的博弈论证据 

**Authors**: Kenneth Payne, Baptiste Alloui-Cros  

**Link**: [PDF](https://arxiv.org/pdf/2507.02618)  

**Abstract**: Are Large Language Models (LLMs) a new form of strategic intelligence, able to reason about goals in competitive settings? We present compelling supporting evidence. The Iterated Prisoner's Dilemma (IPD) has long served as a model for studying decision-making. We conduct the first ever series of evolutionary IPD tournaments, pitting canonical strategies (e.g., Tit-for-Tat, Grim Trigger) against agents from the leading frontier AI companies OpenAI, Google, and Anthropic. By varying the termination probability in each tournament (the "shadow of the future"), we introduce complexity and chance, confounding memorisation.
Our results show that LLMs are highly competitive, consistently surviving and sometimes even proliferating in these complex ecosystems. Furthermore, they exhibit distinctive and persistent "strategic fingerprints": Google's Gemini models proved strategically ruthless, exploiting cooperative opponents and retaliating against defectors, while OpenAI's models remained highly cooperative, a trait that proved catastrophic in hostile environments. Anthropic's Claude emerged as the most forgiving reciprocator, showing remarkable willingness to restore cooperation even after being exploited or successfully defecting. Analysis of nearly 32,000 prose rationales provided by the models reveals that they actively reason about both the time horizon and their opponent's likely strategy, and we demonstrate that this reasoning is instrumental to their decisions. This work connects classic game theory with machine psychology, offering a rich and granular view of algorithmic decision-making under uncertainty. 

**Abstract (ZH)**: 大型语言模型（LLMs）是否是一种新型的战略智能，能够在竞争环境中推理目标？我们的研究提供了有力的支持。我们首次进行了一系列演化博弈论重复囚徒困境（IPD）锦标赛，将经典策略（如以牙还牙、严惩策）与来自领先人工智能公司的OpenAI、Google和Anthropic的代理进行对比。通过改变每个锦标赛的终止概率（“未来的影子”），我们引入了复杂性和偶然性，挑战了记忆能力。

结果显示，LLMs表现出了高度的竞争力，能够在这些复杂生态系统中生存甚至繁衍。此外，它们还表现出独特的且持久的“战略指纹”：Google的Gemini模型表现出战略性无情，利用了合作对手并报复背叛者，而OpenAI的模型则保持了高度的合作性，在敌对环境中这一特性证明是灾难性的。Anthropic的Claude则展现出最强的宽容性重归互惠，即使被利用或成功背叛后也表现出非凡的恢复合作的意愿。对几乎3.2万个模型提供的文字解释的分析表明，它们积极地思考了时间框架和对手可能的策略，并证明这种思考对它们的决策起到了关键作用。这项工作将经典博弈论与机器心理学相结合，提供了关于在不确定性下算法决策的丰富而精细的视角。 

---
# DynamiCare: A Dynamic Multi-Agent Framework for Interactive and Open-Ended Medical Decision-Making 

**Title (ZH)**: DynamiCare: 动态多agents框架下的交互式和开放式医疗决策-making 

**Authors**: Tianqi Shang, Weiqing He, Charles Zheng, Lingyao Li, Li Shen, Bingxin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.02616)  

**Abstract**: The rise of Large Language Models (LLMs) has enabled the development of specialized AI agents with domain-specific reasoning and interaction capabilities, particularly in healthcare. While recent frameworks simulate medical decision-making, they largely focus on single-turn tasks where a doctor agent receives full case information upfront -- diverging from the real-world diagnostic process, which is inherently uncertain, interactive, and iterative. In this paper, we introduce MIMIC-Patient, a structured dataset built from the MIMIC-III electronic health records (EHRs), designed to support dynamic, patient-level simulations. Building on this, we propose DynamiCare, a novel dynamic multi-agent framework that models clinical diagnosis as a multi-round, interactive loop, where a team of specialist agents iteratively queries the patient system, integrates new information, and dynamically adapts its composition and strategy. We demonstrate the feasibility and effectiveness of DynamiCare through extensive experiments, establishing the first benchmark for dynamic clinical decision-making with LLM-powered agents. 

**Abstract (ZH)**: 大语言模型的兴起使得专门领域内的AI代理发展成为可能，特别是在医疗保健领域。尽管近期的框架模拟了医疗决策过程，但它们主要关注于医生代理在获取完整病例信息后进行的一次性任务，这与现实世界中固有的不确定性、互动性和迭代性的诊断过程相去甚远。本文介绍了一种名为MIMIC-Patient的结构化数据集，该数据集基于MIMIC-III电子健康记录（EHRs），旨在支持动态的患者级模拟。在此基础上，我们提出了DynamiCare，这是一种新颖的动态多代理框架，将临床诊断建模为多轮、互动的循环过程，在此过程中，由专业代理团队迭代查询患者系统、整合新信息，并动态调整其组成和策略。我们通过广泛实验展示了DynamiCare的可行性和有效性，建立了首个基于大语言模型代理的动力学临床决策基准。 

---
# OMS: On-the-fly, Multi-Objective, Self-Reflective Ad Keyword Generation via LLM Agent 

**Title (ZH)**: OMS：通过LLM代理实现的实时、多目标、自我反思广告关键词生成 

**Authors**: Bowen Chen, Zhao Wang, Shingo Takamatsu  

**Link**: [PDF](https://arxiv.org/pdf/2507.02353)  

**Abstract**: Keyword decision in Sponsored Search Advertising is critical to the success of ad campaigns. While LLM-based methods offer automated keyword generation, they face three major limitations: reliance on large-scale query-keyword pair data, lack of online multi-objective performance monitoring and optimization, and weak quality control in keyword selection. These issues hinder the agentic use of LLMs in fully automating keyword decisions by monitoring and reasoning over key performance indicators such as impressions, clicks, conversions, and CTA effectiveness. To overcome these challenges, we propose OMS, a keyword generation framework that is On-the-fly (requires no training data, monitors online performance, and adapts accordingly), Multi-objective (employs agentic reasoning to optimize keywords based on multiple performance metrics), and Self-reflective (agentically evaluates keyword quality). Experiments on benchmarks and real-world ad campaigns show that OMS outperforms existing methods; ablation and human evaluations confirm the effectiveness of each component and the quality of generated keywords. 

**Abstract (ZH)**: 关键词决策在赞助搜索广告中的成功至关重要。虽然基于大语言模型的方法提供了自动关键词生成，但它们面临三大限制：对大规模查询-关键词对数据的依赖、缺乏在线多目标性能监控与优化以及关键词选择质量控制不足。这些问题阻碍了通过监控和推理关键性能指标（如印象、点击、转化和CTA有效性）来完全自动化关键词决策的代理使用。为克服这些挑战，我们提出了一种名为OMS的关键词生成框架，该框架具备即用型（无需训练数据、在线监控性能并相应调整）、多目标（基于多种性能指标进行代理推理优化关键词）以及自省型（代理评价关键词质量）的特点。基准测试和实际广告活动的实验结果表明，OMS优于现有方法；消融实验和人工评估证实了每个组件的有效性和生成关键词的质量。 

---
# Scaling LLM Planning: NL2FLOW for Parametric Problem Generation and Rigorous Evaluation 

**Title (ZH)**: 扩展LLM规划：基于自然语言的参数化问题生成与严格评估（NL2FLOW） 

**Authors**: Jungkoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02253)  

**Abstract**: Progress in enhancing large language model (LLM) planning and reasoning capabilities is significantly hampered by the bottleneck of scalable, reliable data generation and evaluation. To overcome this, I introduce NL2FLOW, a fully automated system for parametrically generating planning problems - expressed in natural language, a structured intermediate representation, and formal PDDL - and rigorously evaluating the quality of generated plans. I demonstrate NL2FLOW's capabilities by generating a dataset of 2296 problems in the automated workflow generation domain and evaluating multiple open-sourced, instruct-tuned LLMs. My results reveal that the highest performing models achieved 86% success in generating valid plans and 69% in generating optimal plans, specifically for problems with feasible solutions. Regression analysis shows that the influence of problem characteristics on plan generation is contingent on both model and prompt design. Notably, I observed that the highest success rate for translating natural language into a JSON representation of a plan was lower than the highest rate of generating a valid plan directly. This suggests that unnecessarily decomposing the reasoning task - introducing intermediate translation steps - may actually degrade performance, implying a benefit to models capable of reasoning directly from natural language to action. As I scale LLM reasoning to increasingly complex problems, the bottlenecks and sources of error within these systems will inevitably shift. Therefore, a dynamic understanding of these limitations - and the tools to systematically reveal them - will be crucial for unlocking the full potential of LLMs as intelligent problem solvers. 

**Abstract (ZH)**: 增强大规模语言模型规划与推理能力的进步受到了可扩展、可靠数据生成与评估瓶颈的显著限制。为克服这一障碍，我提出了NL2FLOW，这是一个完全自动化的系统，用于参数化生成用自然语言、结构化的中间表示形式和形式化的PDDL表示的规划问题，并严格评估生成计划的质量。我通过在自动化工作流生成领域生成包含2296个问题的数据集并评估多个开源指令调优的大规模语言模型来展示了NL2FLOW的能力。我的结果显示，性能最高的模型在生成有效计划方面达到了86%的准确率，在生成最优计划方面达到了69%，尤其是在具有可行解的问题上。回归分析表明，问题特征对生成计划的影响取决于模型和提示设计。值得注意的是，我观察到将自然语言翻译成JSON表示的计划的成功率最高值低于直接生成有效计划的成功率最高值。这表明随意分解推理任务，引入中间转换步骤，实际上可能会降低性能，暗示了一种直接从自然语言到行动进行推理的能力对模型是有益的。随着我将大规模语言模型推理扩展到更复杂的问题，这些系统内的瓶颈和错误来源必然会有所转变。因此，动态理解这些限制以及系统地揭示这些限制的工具将对于充分发挥大规模语言模型作为智能问题解决者的能力至关重要。 

---
# Do Role-Playing Agents Practice What They Preach? Belief-Behavior Consistency in LLM-Based Simulations of Human Trust 

**Title (ZH)**: 基于LLM的人类信任仿真中的信念-行为一致性：角色扮演代理是否言行一致？ 

**Authors**: Amogh Mannekote, Adam Davies, Guohao Li, Kristy Elizabeth Boyer, ChengXiang Zhai, Bonnie J Dorr, Francesco Pinto  

**Link**: [PDF](https://arxiv.org/pdf/2507.02197)  

**Abstract**: As LLMs are increasingly studied as role-playing agents to generate synthetic data for human behavioral research, ensuring that their outputs remain coherent with their assigned roles has become a critical concern. In this paper, we investigate how consistently LLM-based role-playing agents' stated beliefs about the behavior of the people they are asked to role-play ("what they say") correspond to their actual behavior during role-play ("how they act"). Specifically, we establish an evaluation framework to rigorously measure how well beliefs obtained by prompting the model can predict simulation outcomes in advance. Using an augmented version of the GenAgents persona bank and the Trust Game (a standard economic game used to quantify players' trust and reciprocity), we introduce a belief-behavior consistency metric to systematically investigate how it is affected by factors such as: (1) the types of beliefs we elicit from LLMs, like expected outcomes of simulations versus task-relevant attributes of individual characters LLMs are asked to simulate; (2) when and how we present LLMs with relevant information about Trust Game; and (3) how far into the future we ask the model to forecast its actions. We also explore how feasible it is to impose a researcher's own theoretical priors in the event that the originally elicited beliefs are misaligned with research objectives. Our results reveal systematic inconsistencies between LLMs' stated (or imposed) beliefs and the outcomes of their role-playing simulation, at both an individual- and population-level. Specifically, we find that, even when models appear to encode plausible beliefs, they may fail to apply them in a consistent way. These findings highlight the need to identify how and when LLMs' stated beliefs align with their simulated behavior, allowing researchers to use LLM-based agents appropriately in behavioral studies. 

**Abstract (ZH)**: 随着LLMs作为角色扮演代理被越来越多地研究以生成用于人类行为研究的合成数据，确保其输出与其分配的角色保持一致已成为一个关键问题。本文探索基于LLM的角色扮演代理所表达的关于他们被要求扮演的角色的行为信念（“他们说什么”）与其角色扮演时的实际行为（“他们如何做”）之间的一致性。具体而言，我们建立了一个评估框架，以严格衡量通过提示模型获得的信念预测模拟结果的准确度。使用增强版本的GenAgents人设库和信任博弈（一种用于量化玩家信任和互惠的标准经济博弈），我们引入了一个信念-行为一致性度量，系统地研究了该度量受到以下因素的影响：（1）从LLM中引发的信念类型，如模拟预期结果与任务相关的个体角色属性；（2）何时以及以何种方式向LLM呈现信任博弈的相关信息；以及（3）要求模型预测其行为的时间范围。我们还探讨了在原本引发的信念与研究目标不一致时，是否可以在研究中强制应用研究者的先验理论假设。研究结果揭示了个体和群体层面之间LLMs所声明（或施加的）信念与角色扮演模拟结果之间的一致性系统性差异。具体而言，我们发现，即使模型似乎编码了合理的信念，它们也可能会以不一致的方式应用这些信念。这些发现突显了识别LLMs所声明的信念与其模拟行为之间一致性的必要性，以使研究人员能够恰当地在行为研究中使用基于LLM的代理。 

---
# Data Diversification Methods In Alignment Enhance Math Performance In LLMs 

**Title (ZH)**: 数据多样化方法在对齐中的增强提高大语言模型的数学性能 

**Authors**: Berkan Dokmeci, Qingyang Wu, Ben Athiwaratkun, Ce Zhang, Shuaiwen Leon Song, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2507.02173)  

**Abstract**: While recent advances in preference learning have enhanced alignment in human feedback, mathematical reasoning remains a persistent challenge. We investigate how data diversification strategies in preference optimization can improve the mathematical reasoning abilities of large language models (LLMs). We evaluate three common data generation methods: temperature sampling, Chain-of-Thought prompting, and Monte Carlo Tree Search (MCTS), and introduce Diversified-ThinkSolve (DTS), a novel structured approach that systematically decomposes problems into diverse reasoning paths. Our results show that with strategically diversified preference data, models can substantially improve mathematical reasoning performance, with the best approach yielding gains of 7.1% on GSM8K and 4.2% on MATH over the base model. Despite its strong performance, DTS incurs only a marginal computational overhead (1.03x) compared to the baseline, while MCTS is nearly five times more costly with lower returns. These findings demonstrate that structured exploration of diverse problem-solving methods creates more effective preference data for mathematical alignment than traditional approaches. 

**Abstract (ZH)**: 虽然近期偏好学习的进步已经在人类反馈对齐方面取得了进展，但数学推理依然是一项持续性的挑战。我们探讨了偏好优化中数据多样化策略如何提升大型语言模型（LLMs）的数学推理能力。我们评估了三种常见的数据生成方法：温度采样、Chain-of-Thought提示和蒙特卡洛树搜索（MCTS），并引入了新颖的结构化方法Diversified-ThinkSolve（DTS），该方法系统地将问题分解为多种推理路径。实验结果表明，通过战略性地多样化偏好数据，模型可以在数学推理性能上实现显著提升，最佳方法在GSM8K上的提升为7.1%，在MATH上的提升为4.2%，相较于基线模型；尽管DTS的性能优越，但其计算开销仅比基线增加了1.03倍，而MCTS的成本几乎是其五倍且收益较低。这些发现表明，结构化的多样化问题解决方法探索相比于传统方法能更有效地生成用于数学对齐的偏好数据。 

---
# Measuring Scientific Capabilities of Language Models with a Systems Biology Dry Lab 

**Title (ZH)**: 用系统生物学虚拟实验室衡量语言模型的科学能力 

**Authors**: Haonan Duan, Stephen Zhewen Lu, Caitlin Fiona Harrigan, Nishkrit Desai, Jiarui Lu, Michał Koziarski, Leonardo Cotta, Chris J. Maddison  

**Link**: [PDF](https://arxiv.org/pdf/2507.02083)  

**Abstract**: Designing experiments and result interpretations are core scientific competencies, particularly in biology, where researchers perturb complex systems to uncover the underlying systems. Recent efforts to evaluate the scientific capabilities of large language models (LLMs) fail to test these competencies because wet-lab experimentation is prohibitively expensive: in expertise, time and equipment. We introduce SciGym, a first-in-class benchmark that assesses LLMs' iterative experiment design and analysis abilities in open-ended scientific discovery tasks. SciGym overcomes the challenge of wet-lab costs by running a dry lab of biological systems. These models, encoded in Systems Biology Markup Language, are efficient for generating simulated data, making them ideal testbeds for experimentation on realistically complex systems. We evaluated six frontier LLMs on 137 small systems, and released a total of 350 systems. Our evaluation shows that while more capable models demonstrated superior performance, all models' performance declined significantly as system complexity increased, suggesting substantial room for improvement in the scientific capabilities of LLM agents. 

**Abstract (ZH)**: 设计实验和结果解释是核心的科学能力，特别是在生物学中，研究人员通过干扰复杂系统来揭示其内在机制。近期对大型语言模型（LLMs）的科学能力评估未能测试这些能力，因为湿实验的成本极高：在专业知识、时间和设备方面。我们引入了SciGym，这是一个首创的基准，用于评估LLMs在开放性科学发现任务中迭代实验设计和分析的能力。SciGym通过运行生物系统的虚拟实验室，克服了湿实验成本的挑战。这些模型以系统生物学标记语言编码，适用于生成模拟数据，是进行复杂系统实验的理想试验平台。我们在137个小系统上评估了六种前沿的LLMs，并总共发布了350个系统。我们的评估表明，虽然更强大的模型表现出更优异的性能，但所有模型的性能随着系统复杂性的增加而显著下降，这表明在LLM代理的科学能力上有很大的改进空间。 

---
# Reasoning on a Budget: A Survey of Adaptive and Controllable Test-Time Compute in LLMs 

**Title (ZH)**: 节省计算资源的推理：大规模语言模型运行时自适应与可控计算综述 

**Authors**: Mohammad Ali Alomrani, Yingxue Zhang, Derek Li, Qianyi Sun, Soumyasundar Pal, Zhanguang Zhang, Yaochen Hu, Rohan Deepak Ajwani, Antonios Valkanas, Raika Karimi, Peng Cheng, Yunzhou Wang, Pengyi Liao, Hanrui Huang, Bin Wang, Jianye Hao, Mark Coates  

**Link**: [PDF](https://arxiv.org/pdf/2507.02076)  

**Abstract**: Large language models (LLMs) have rapidly progressed into general-purpose agents capable of solving a broad spectrum of tasks. However, current models remain inefficient at reasoning: they apply fixed inference-time compute regardless of task complexity, often overthinking simple problems while underthinking hard ones. This survey presents a comprehensive review of efficient test-time compute (TTC) strategies, which aim to improve the computational efficiency of LLM reasoning. We introduce a two-tiered taxonomy that distinguishes between L1-controllability, methods that operate under fixed compute budgets, and L2-adaptiveness, methods that dynamically scale inference based on input difficulty or model confidence. We benchmark leading proprietary LLMs across diverse datasets, highlighting critical trade-offs between reasoning performance and token usage. Compared to prior surveys on efficient reasoning, our review emphasizes the practical control, adaptability, and scalability of TTC methods. Finally, we discuss emerging trends such as hybrid thinking models and identify key challenges for future work towards making LLMs more computationally efficient, robust, and responsive to user constraints. 

**Abstract (ZH)**: 大型语言模型（LLMs）已迅速发展成为能够解决广泛任务的通用代理。然而，当前模型在推理方面仍然不够高效：它们在推理时固定使用计算资源，不论任务复杂度如何，往往会对简单问题过度思考，而对困难问题则思考不足。本文综述了高效测试时计算（TTC）策略，旨在提高LLM推理的计算效率。我们引入了两层分类体系，区分L1可控性方法，即在固定计算预算下的方法，和L2适应性方法，即根据输入难度或模型置信度动态调整推理的方法。我们在多种数据集上对比了行业领先的私有LLM，并强调了推理性能和token使用之间的关键权衡。与之前的高效推理综述相比，我们的综述更注重TTC方法的实际可控性、适应性和可扩展性。最后，我们讨论了新兴趋势，如混合思考模型，并指出了未来工作中使LLMs更高效、更鲁棒且更能响应用户约束的关键挑战。 

---
# MOTIF: Modular Thinking via Reinforcement Fine-tuning in LLMs 

**Title (ZH)**: MOTIF: 在LLMs中通过强化微调实现模块化思考 

**Authors**: Purbesh Mitra, Sennur Ulukus  

**Link**: [PDF](https://arxiv.org/pdf/2507.02851)  

**Abstract**: Recent advancements in the reasoning capabilities of large language models (LLMs) show that employing group relative policy optimization (GRPO) algorithm for reinforcement learning (RL) training allows the models to use more thinking/reasoning tokens for generating better responses. However, LLMs can generate only a finite amount of tokens while maintaining attention to the previously generated tokens. This limit, also known as the context size of an LLM, is a bottleneck in LLM reasoning with arbitrarily large number of tokens. To think beyond the limit of context size, an LLM must employ a modular thinking strategy to reason over multiple rounds. In this work, we propose $\textbf{MOTIF: Modular Thinking via Reinforcement Finetuning}$ -- an RL training method for generating thinking tokens in multiple rounds, effectively allowing the model to think with additional context size. We trained the open-source model Qwen2.5-3B-Instruct on GSM8K dataset via parameter efficient fine-tuning and tested its accuracy on MATH500 and AIME2024 benchmarks. Our experiments show 3.8\% and 3.3\% improvements over vanilla GRPO based training in the respective benchmarks. Furthermore, this improvement was achieved with only 15\% of samples, thus demonstrating sample efficiency of MOTIF. Our code and models are available at this https URL and this https URL, respectively. 

**Abstract (ZH)**: 大型语言模型（LLMs）推理能力的 recent 进展表明，使用组相对策略优化（GRPO）算法进行强化学习（RL）训练可以使模型使用更多思考/推理令牌以生成更好的回应。然而，LLMs 在维持对先前生成令牌的关注时，只能生成有限数量的令牌。这一限制，也称为 LLM 的上下文大小，是 LLM 在处理任意大量令牌时推理的一个瓶颈。为了超越上下文大小的限制，LLM 必须采用模块化思考策略，分多轮进行推理。在本项工作中，我们提出了 $\textbf{MOTIF: 模块化思考通过强化学习微调}$ —— 一种用于多轮生成思考令牌的 RL 训练方法，有效地使模型能够在更大的上下文大小下思考。我们通过参数高效微调在 Qwen2.5-3B-Instruct 模型上对 GSM8K 数据集进行了训练，并在 MATH500 和 AIME2024 指标上测试了其准确性。我们的实验结果显示，在相应的基准中，MOTIF 比基于 vanilla GRPO 的训练分别提高了 3.8% 和 3.3% 的准确性。此外，这种改进仅使用了 15% 的样本，从而展示了 MOTIF 的样本效率。我们的代码和模型分别在此 https URL 和此 https URL 上提供。 

---
# SynapseRoute: An Auto-Route Switching Framework on Dual-State Large Language Model 

**Title (ZH)**: SynapseRoute: 双状态大型语言模型的自适应路由切换框架 

**Authors**: Wencheng Zhang, Shiqin Qiao, Lingjie Luo, Yinfeng Li, Chuanyang Zheng, Qian Xu, Meng Li, Yong Gui, Yijun He, Jianing Qiu, Jindong Hong, Jiankai Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.02822)  

**Abstract**: With the widespread adoption of large language models (LLMs) in practical applications, selecting an appropriate model requires balancing not only performance but also operational cost. The emergence of reasoning-capable models has further widened the cost gap between "thinking" (high reasoning) and "non-thinking" (fast, low-cost) modes. In this work, we reveal that approximately 58% of medical questions can be accurately answered by the non-thinking mode alone, without requiring the high-cost reasoning process. This highlights a clear dichotomy in problem complexity and suggests that dynamically routing queries to the appropriate mode based on complexity could optimize accuracy, cost-efficiency, and overall user experience. Based on this, we further propose SynapseRoute, a machine learning-based dynamic routing framework that intelligently assigns input queries to either thinking or non-thinking modes. Experimental results on several medical datasets demonstrate that SynapseRoute not only improves overall accuracy (0.8390 vs. 0.8272) compared to the thinking mode alone but also reduces inference time by 36.8% and token consumption by 39.66%. Importantly, qualitative analysis indicates that over-reasoning on simpler queries can lead to unnecessary delays and even decreased accuracy, a pitfall avoided by our adaptive routing. Finally, this work further introduces the Accuracy-Inference-Token (AIT) index to comprehensively evaluate the trade-offs among accuracy, latency, and token cost. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在实际应用中的广泛应用，选择合适的模型不仅需要平衡性能，还需要平衡运营成本。具有推理能力的模型的出现进一步拉开了“思考”（高推理）模式与“非思考”（快速、低成本）模式之间成本差距。在本文中，我们揭示了大约58%的医疗问题仅通过非思考模式即可准确回答，无需进行高成本的推理过程。这突显了问题复杂度的明显二元性，并表明根据复杂度动态路由查询可以优化准确度、成本效率和整体用户体验。基于此，我们进一步提出了一种基于机器学习的动态路由框架SynapseRoute，该框架能够智能地将输入查询分配给思考或非思考模式。在几个医疗数据集上的实验结果表明，与单独使用思考模式相比，SynapseRoute不仅总体准确度提高了（0.8390 vs. 0.8272），还减少了36.8%的推理时间和39.66%的令牌消耗。更重要的是，定性分析表明，在简单查询上过度推理可能导致不必要的延迟甚至降低准确度，这是我们动态路由方法避免的问题。最后，本文还引入了准确度-推理时间-令牌成本（AIT）指数，以全面评估准确度、延迟和令牌成本之间的权衡。 

---
# Self-Correction Bench: Revealing and Addressing the Self-Correction Blind Spot in LLMs 

**Title (ZH)**: 自我修正平台：揭示并解决大规模语言模型中的自我修正盲区 

**Authors**: Ken Tsui  

**Link**: [PDF](https://arxiv.org/pdf/2507.02778)  

**Abstract**: Although large language models (LLMs) have become transformative, they still make mistakes and can explore unproductive reasoning paths. Self-correction is an important capability for a trustworthy LLM, particularly an autoregressive LLM. While LLMs can identify error in user input, they exhibit a systematic 'Self-Correction Blind Spot' - failing to correct identical error in their own outputs. To systematically study this phenomenon, we introduce Self-Correction Bench, a systematic framework to measure this phenomenon through controlled error injection at three complexity levels. Testing 14 models, we find an average 64.5% blind spot rate. We find multiple evidences that this limitation relates to training data composition: human training demonstrations predominantly show error-free responses rather than error-correction sequences, unlike RL-trained models that learn error correction through outcome feedback. Remarkably, simply appending "Wait" reduces blind spots by 89.3%, suggesting that the capability exists but requires activation. Our work highlights a critical limitation in current LLMs and offers potential avenues for improving their reliability and trustworthiness. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）已经变得极具变革性，它们仍然会出错，并且可能会探索无效的推理路径。自我修正是一项重要能力，特别是对于自回归的大语言模型而言。虽然LLMs可以识别用户输入中的错误，但它们表现出一种系统性的“自我修正盲区”——无法修正自身输出中的相同错误。为了系统性地研究这一现象，我们引入了Self-Correction Bench这一系统框架，通过在三个复杂程度级别上进行受控的错误注入来衡量这一现象。测试14个模型后，我们发现平均盲区率为64.5%。我们发现多种证据表明，这一局限与训练数据的组成有关：人类训练示例主要展示无错误的响应，而非错误纠正序列，这与通过结果反馈学习错误纠正的强化学习训练模型不同。令人惊讶的是，仅仅添加“等待”就可将盲区降低89.3%，表明该能力存在但需要激活。我们的工作突显了当前LLMs的一项关键局限，并为提高其可靠性和可信度提供了潜在途径。 

---
# Fast and Simplex: 2-Simplicial Attention in Triton 

**Title (ZH)**: 快速简洁：Triton中的2-单纯形注意力 

**Authors**: Aurko Roy, Timothy Chou, Sai Surya Duvvuri, Sijia Chen, Jiecao Yu, Xiaodong Wang, Manzil Zaheer, Rohan Anil  

**Link**: [PDF](https://arxiv.org/pdf/2507.02754)  

**Abstract**: Recent work has shown that training loss scales as a power law with both model size and the number of tokens, and that achieving compute-optimal models requires scaling model size and token count together. However, these scaling laws assume an infinite supply of data and apply primarily in compute-bound settings. As modern large language models increasingly rely on massive internet-scale datasets, the assumption that they are compute-bound is becoming less valid. This shift highlights the need for architectures that prioritize token efficiency.
In this work, we investigate the use of the 2-simplicial Transformer, an architecture that generalizes standard dot-product attention to trilinear functions through an efficient Triton kernel implementation. We demonstrate that the 2-simplicial Transformer achieves better token efficiency than standard Transformers: for a fixed token budget, similarly sized models outperform their dot-product counterparts on tasks involving mathematics, coding, reasoning, and logic. We quantify these gains by demonstrating that $2$-simplicial attention changes the exponent in the scaling laws for knowledge and reasoning tasks compared to dot product attention. 

**Abstract (ZH)**: 最近的研究表明，训练损失与模型规模和令牌数量呈幂律关系，并且实现计算最优模型需要同时扩展模型规模和令牌数量。然而，这些扩展定律假设数据无限供应，并主要适用于计算受限的环境。随着现代大型语言模型越来越多地依赖于大规模互联网数据集，它们计算受限的假设变得不再那么有效。这种转变突显了优先考虑令牌效率的架构的需求。

在这项工作中，我们探讨了2-单形变换器的应用，这是一种架构，通过高效的Triton内核实现将标准点积注意扩展到三线性函数。我们证明，2-单形变换器在令牌效率方面优于标准变换器：在固定令牌预算的情况下，相似规模的模型在涉及数学、编程、推理和逻辑的任务中表现优于点积版本。我们通过证明与点积注意力相比，2-单形注意力改变了知识和推理任务的扩展定律的指数来量化这些增益。 

---
# Early Signs of Steganographic Capabilities in Frontier LLMs 

**Title (ZH)**: 前沿大语言模型中的早期隐写能力迹象 

**Authors**: Artur Zolkowski, Kei Nishimura-Gasparian, Robert McCarthy, Roland S. Zimmermann, David Lindner  

**Link**: [PDF](https://arxiv.org/pdf/2507.02737)  

**Abstract**: Monitoring Large Language Model (LLM) outputs is crucial for mitigating risks from misuse and misalignment. However, LLMs could evade monitoring through steganography: Encoding hidden information within seemingly benign generations. In this paper, we evaluate the steganography capabilities in frontier LLMs to better understand the risk they pose. We focus on two types of steganography: passing encoded messages and performing encoded reasoning. We find that current models are unable to encode short messages in their outputs without a monitor noticing under standard affordances. They can succeed, however, if given additional affordances such as using an unmonitored scratchpad and coordinating on what encoding scheme to use. We additionally find early signs that models can perform basic encoded reasoning in a simple state-tracking problem. This includes some ability to reason with their own and pre-defined schemes, including encoding schemes such as Hexadecimal. Despite this, they can rarely hide reasoning subtly within a cover task to fool a monitor. Overall, our results indicate that current LLMs exhibit nascent steganographic capabilities. While these capabilities are likely insufficient to bypass well-designed monitors at present, this could change in the future. 

**Abstract (ZH)**: 监控大型语言模型（LLM）的输出对于减轻滥用和不对齐风险至关重要。然而，LLM可以通过藏码技术规避监控：在看似无害的生成中嵌入隐藏信息。在本文中，我们评估前沿LLM的藏码能力，以更好地理解它们所造成的风险。我们重点关注两种类型的藏码：传递编码信息和进行编码推理。我们发现，当前模型无法在标准条件下在其输出中嵌入短消息而不被监控发现。然而，如果提供额外的条件，如使用未监控的草稿纸并协调使用的编码方案，则它们可以成功。此外，我们还发现模型在简单状态跟踪问题中可以进行基本的编码推理的早期迹象，包括与自身和预定义方案进行推理的能力，包括十六进制等编码方案。尽管如此，它们很少能够通过隐秘地将推理融入辅助任务中欺骗监控。总体而言，我们的结果表明当前LLM展现出了初步的藏码能力。尽管这些能力当前似乎不足以绕过精心设计的监控系统，但这一点在未来可能会改变。 

---
# Meta SecAlign: A Secure Foundation LLM Against Prompt Injection Attacks 

**Title (ZH)**: Meta SecAlign: 针对提示注入攻击的secure基础LLM 

**Authors**: Sizhe Chen, Arman Zharmagambetov, David Wagner, Chuan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.02735)  

**Abstract**: Prompt injection attacks pose a significant security threat to LLM-integrated applications. Model-level defenses have shown strong effectiveness, but are currently deployed into commercial-grade models in a closed-source manner. We believe open-source models are needed by the AI security community, where co-development of attacks and defenses through open research drives scientific progress in mitigation against prompt injection attacks. To this end, we develop Meta SecAlign, the first open-source and open-weight LLM with built-in model-level defense that achieves commercial-grade model performance. We provide complete details of our training recipe, which utilizes an improved version of the SOTA SecAlign defense. Evaluations on 9 utility benchmarks and 7 security benchmarks show that Meta SecAlign, despite being trained on a generic instruction-tuning dataset, confers security in unseen downstream tasks, including tool-calling and agentic web navigation, in addition general instruction-following. Our best model -- Meta-SecAlign-70B -- achieves state-of-the-art robustness against prompt injection attacks and comparable utility to closed-source commercial LLM with model-level defense. 

**Abstract (ZH)**: 提示注入攻击对LLM集成应用构成重大安全威胁。内置模型级防御的开源模型是AI安全社区所需，通过开放研究促进对抗提示注入攻击的科学进步。为此，我们开发了Meta SecAlign，这是第一个集成了模型级防御并达到商业级模型性能的开源且开放权重的LLM。我们提供了完整的训练方案细节，该方案使用了SOTA SecAlign防御的改进版本。在9个功能基准和7个安全基准上的评估显示，Meta SecAlign不仅在通用指令遵循任务中表现出安全性，在未见的下游任务中，如工具调用和代理型网页导航中也表现出安全性。我们的最佳模型——Meta-SecAlign-70B——在提示注入攻击下的鲁棒性达到了最新水平，并且在模型级防御方面的功能与商业级闭源LLM相当。 

---
# FlowSpec: Continuous Pipelined Speculative Decoding for Efficient Distributed LLM Inference 

**Title (ZH)**: FlowSpec：连续管道 speculative 解码以实现高效的分布式大语言模型推理 

**Authors**: Xing Liu, Lizhuo Luo, Ming Tang, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02620)  

**Abstract**: Distributed inference serves as a promising approach to enabling the inference of large language models (LLMs) at the network edge. It distributes the inference process to multiple devices to ensure that the LLMs can fit into the device memory. Recent pipeline-based approaches have the potential to parallelize communication and computation, which helps reduce inference latency. However, the benefit diminishes when the inference request at the network edge is sparse, where pipeline is typically at low utilization. To enable efficient distributed LLM inference at the edge, we propose \textbf{FlowSpec}, a pipeline-parallel tree-based speculative decoding framework. FlowSpec incorporates three key mechanisms to improve decoding efficiency: 1) score-based step-wise verification prioritizes more important draft tokens to bring earlier accpeted tokens; 2) efficient draft management to prune invalid tokens while maintaining correct causal relationship during verification; 3) dynamic draft expansion strategies to supply high-quality speculative inputs. These techniques work in concert to enhance both pipeline utilization and speculative efficiency. We evaluate FlowSpec on a real-world testbed with other baselines. Experimental results demonstrate that our proposed framework significantly improves inference speed across diverse models and configurations, achieving speedup ratios 1.36$\times$-1.77$\times$ compared to baselines. Our code is publicly available at \href{this https URL}{this https URL\#} 

**Abstract (ZH)**: 分布式推理作为一种有前景的方法，适用于在网络边缘进行大型语言模型（LLMs）的推理。它将推理过程分布在多个设备上，确保LLMs能够适应设备内存。基于管道的方法具有潜在的并行化通信和计算的能力，有助于减少推理延迟。然而，在网络边缘推理请求稀疏的情况下，管道通常利用率低，这种优势会减弱。为在边缘启用高效分布式LLM推理，我们提出了一种基于管道并行树状推测解码框架\textbf{FlowSpec}。FlowSpec结合了三种关键技术来提高解码效率：1) 基于得分的逐步验证优先处理更重要草稿 token，以带来更早被接受的 token；2) 高效的草稿管理，在验证过程中修剪无效 token 同时保持正确的因果关系；3) 动态草稿扩展策略，提供高质量的推测输入。这些技术共同提升管道利用率和推测效率。我们在现实测试平台上对FlowSpec与其他基准进行了评估。实验结果表明，我们提出的方法在多种模型和配置下显著提高了推理速度，相比基准方法实现了1.36$\times$-1.77$\times$的加速比。我们的代码已公开在\href{this https URL}{this https URL\#}。 

---
# MPF: Aligning and Debiasing Language Models post Deployment via Multi Perspective Fusion 

**Title (ZH)**: MPF: 部署后通过多视角融合对齐和祛偏语言模型 

**Authors**: Xin Guan, PeiHsin Lin, Zekun Wu, Ze Wang, Ruibo Zhang, Emre Kazim, Adriano Koshiyama  

**Link**: [PDF](https://arxiv.org/pdf/2507.02595)  

**Abstract**: Multiperspective Fusion (MPF) is a novel posttraining alignment framework for large language models (LLMs) developed in response to the growing need for easy bias mitigation. Built on top of the SAGED pipeline, an automated system for constructing bias benchmarks and extracting interpretable baseline distributions, MPF leverages multiperspective generations to expose and align biases in LLM outputs with nuanced, humanlike baselines. By decomposing baseline, such as sentiment distributions from HR professionals, into interpretable perspective components, MPF guides generation through sampling and balancing of responses, weighted by the probabilities obtained in the decomposition. Empirically, we demonstrate its ability to align LLM sentiment distributions with both counterfactual baselines (absolute equality) and the HR baseline (biased for Top Univeristy), resulting in small KL divergence, reduction of calibration error and generalization to unseen questions. This shows that MPF offers a scalable and interpretable method for alignment and bias mitigation, compatible with deployed LLMs and requiring no extensive prompt engineering or finetuning. 

**Abstract (ZH)**: 多视角融合（MPF）是针对大型语言模型（LLMs）日益增长的去偏需求开发的一种新型后训练对齐框架。 

---
# WebSailor: Navigating Super-human Reasoning for Web Agent 

**Title (ZH)**: WebSailor: 导航超人类推理的网络代理 

**Authors**: Kuan Li, Zhongwang Zhang, Huifeng Yin, Liwen Zhang, Litu Ou, Jialong Wu, Wenbiao Yin, Baixuan Li, Zhengwei Tao, Xinyu Wang, Weizhou Shen, Junkai Zhang, Dingchu Zhang, Xixi Wu, Yong Jiang, Ming Yan, Pengjun Xie, Fei Huang, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.02592)  

**Abstract**: Transcending human cognitive limitations represents a critical frontier in LLM training. Proprietary agentic systems like DeepResearch have demonstrated superhuman capabilities on extremely complex information-seeking benchmarks such as BrowseComp, a feat previously unattainable. We posit that their success hinges on a sophisticated reasoning pattern absent in open-source models: the ability to systematically reduce extreme uncertainty when navigating vast information landscapes. Based on this insight, we introduce WebSailor, a complete post-training methodology designed to instill this crucial capability. Our approach involves generating novel, high-uncertainty tasks through structured sampling and information obfuscation, RFT cold start, and an efficient agentic RL training algorithm, Duplicating Sampling Policy Optimization (DUPO). With this integrated pipeline, WebSailor significantly outperforms all opensource agents in complex information-seeking tasks, matching proprietary agents' performance and closing the capability gap. 

**Abstract (ZH)**: 超越人类认知限制是LLM训练的一个关键前沿领域。像DeepResearch这样的专有自主系统已在诸如BrowseComp等极其复杂的信息检索基准上展示了超人类的能力，这是前所未有的。我们认为其成功在于一种在开源模型中缺失的复杂的推理模式：能够在 vast 信息景观中系统地降低极端不确定性。基于这一见解，我们提出了WebSailor，一个完整的后训练方法论，旨在赋予这种关键能力。我们的方法包括通过结构化采样和信息混淆生成新颖的高不确定性任务、RFT冷启动以及高效的自主RL训练算法DUPO（Duplicating Sampling Policy Optimization）。通过这一整合的工作流程，WebSailor在复杂的信息检索任务中显著优于所有开源代理，并与专有代理的性能相当，缩小了能力差距。 

---
# Are You Listening to Me? Fine-Tuning Chatbots for Empathetic Dialogue 

**Title (ZH)**: 你在倾听我吗？细调聊天机器人以实现共情对话 

**Authors**: Paulo Ricardo Knob, Leonardo Scholler, Juliano Rigatti, Soraia Raupp Musse  

**Link**: [PDF](https://arxiv.org/pdf/2507.02537)  

**Abstract**: Conversational agents have made significant progress since ELIZA, expanding their role across various domains, including healthcare, education, and customer service. As these agents become increasingly integrated into daily human interactions, the need for emotional intelligence, particularly empathetic listening, becomes increasingly essential. In this study, we explore how Large Language Models (LLMs) respond when tasked with generating emotionally rich interactions. Starting from a small dataset manually crafted by an expert to reflect empathic behavior, we extended the conversations using two LLMs: ChatGPT and Gemini. We analyzed the emotional progression of the dialogues using both sentiment analysis (via VADER) and expert assessments. While the generated conversations often mirrored the intended emotional structure, human evaluation revealed important differences in the perceived empathy and coherence of the responses. These findings suggest that emotion modeling in dialogues requires not only structural alignment in the expressed emotions but also qualitative depth, highlighting the importance of combining automated and humancentered methods in the development of emotionally competent agents. 

**Abstract (ZH)**: 自ELIZA以来，对话代理在各个领域，包括医疗、教育和客户服务中取得了显著进展。随着这些代理越来越融入日常的人际互动中，特别是在情绪智能，尤其是同理心倾听方面的需求变得越来越重要。在本研究中，我们探讨了大规模语言模型（LLMs）在生成富有情感的互动时的反应。我们从一个由专家手工制作的小数据集扩展对话，使用了两种LLM：ChatGPT和Gemini。我们使用情感分析（通过VADER）和专家评估分析了对话的情绪进展。虽然生成的对话通常反映了预期的情感结构，但人类评估揭示了在感知同理心和响应连贯性方面的重要差异。这些发现表明，对话中的情绪建模不仅需要情感表达的结构对齐，还需要定量深度，强调了在开发情绪胜任代理过程中结合自动化和以人为本方法的重要性。 

---
# Continual Gradient Low-Rank Projection Fine-Tuning for LLMs 

**Title (ZH)**: 持续梯度低秩投影微调用于大语言模型 

**Authors**: Chenxu Wang, Yilin Lyu, Zicheng Sun, Liping Jing  

**Link**: [PDF](https://arxiv.org/pdf/2507.02503)  

**Abstract**: Continual fine-tuning of Large Language Models (LLMs) is hampered by the trade-off between efficiency and expressiveness. Low-Rank Adaptation (LoRA) offers efficiency but constrains the model's ability to learn new tasks and transfer knowledge due to its low-rank nature and reliance on explicit parameter constraints. We propose GORP (Gradient LOw Rank Projection) for Continual Learning, a novel training strategy that overcomes these limitations by synergistically combining full and low-rank parameters and jointly updating within a unified low-rank gradient subspace. GORP expands the optimization space while preserving efficiency and mitigating catastrophic forgetting. Extensive experiments on continual learning benchmarks demonstrate GORP's superior performance compared to existing state-of-the-art approaches. Code is available at this https URL. 

**Abstract (ZH)**: 连续微调大型语言模型（LLMs）受到效率与表达力之间的权衡限制。低秩适应（LoRA）虽然提升了效率，但由于其低秩性质和对显式参数约束的依赖，限制了模型学习新任务和知识迁移的能力。我们提出了一个新颖的训练策略GORP（Gradient LOw Rank Projection）用于连续学习，该策略通过将全参数和低秩参数协同结合并在统一的低秩梯度子空间内联合更新来克服这些限制。GORP扩展了优化空间，同时保持效率并减轻灾难性遗忘。广泛的经验研究表明，GORP在连续学习基准测试中优于现有最佳方法。代码可在以下网址获取。 

---
# CyberRAG: An agentic RAG cyber attack classification and reporting tool 

**Title (ZH)**: CyberRAG: 一个自主的RAG网络攻击分类和报告工具 

**Authors**: Francesco Blefari, Cristian Cosentino, Francesco Aurelio Pironti, Angelo Furfaro, Fabrizio Marozzo  

**Link**: [PDF](https://arxiv.org/pdf/2507.02424)  

**Abstract**: Intrusion Detection and Prevention Systems (IDS/IPS) in large enterprises can generate hundreds of thousands of alerts per hour, overwhelming security analysts with logs that demand deep, rapidly evolving domain expertise. Conventional machine-learning detectors trim the alert volume but still yield high false-positive rates, while standard single-pass Retrieval-Augmented Generation (RAG) pipelines often retrieve irrelevant context and fail to justify their predictions. To overcome these shortcomings, we present CyberRAG, a modular, agent-based RAG framework that delivers real-time classification, explanation, and structured reporting for cyber-attacks. A central LLM agent orchestrates (i) a pool of fine-tuned specialized classifiers, each tailored to a distinct attack family; (ii) tool adapters for enrichment and alerting; and (iii) an iterative retrieval-and-reason loop that continuously queries a domain-specific knowledge base until the evidence is both relevant and self-consistent. Unlike traditional RAG systems, CyberRAG embraces an agentic design that enables dynamic control flow and adaptive reasoning. This agent-centric architecture refines its threat labels and natural-language justifications autonomously, reducing false positives and enhancing interpretability. The framework is fully extensible: new attack types can be supported by simply adding a classifier without retraining the core agent. CyberRAG has been evaluated achieving over 94% accuracy per class and pushing final classification accuracy to 94.92% through semantic orchestration. Generated explanations score up to 0.94 in BERTScore and 4.9/5 in GPT-4-based expert evaluation. These results show that agentic, specialist-oriented RAG can pair high detection accuracy with trustworthy, SOC-ready prose, offering a practical and scalable path toward semi-autonomous cyber-defence workflows. 

**Abstract (ZH)**: 基于代理的 CyberRAG：实时分类、解释和结构化报告的入侵检测与预防系统 

---
# Evaluating Language Models For Threat Detection in IoT Security Logs 

**Title (ZH)**: 评估物联网安全日志中威胁检测的语言模型 

**Authors**: Jorge J. Tejero-Fernández, Alfonso Sánchez-Macián  

**Link**: [PDF](https://arxiv.org/pdf/2507.02390)  

**Abstract**: Log analysis is a relevant research field in cybersecurity as they can provide a source of information for the detection of threats to networks and systems. This paper presents a pipeline to use fine-tuned Large Language Models (LLMs) for anomaly detection and mitigation recommendation using IoT security logs. Utilizing classical machine learning classifiers as a baseline, three open-source LLMs are compared for binary and multiclass anomaly detection, with three strategies: zero-shot, few-shot prompting and fine-tuning using an IoT dataset. LLMs give better results on multi-class attack classification than the corresponding baseline models. By mapping detected threats to MITRE CAPEC, defining a set of IoT-specific mitigation actions, and fine-tuning the models with those actions, the models are able to provide a combined detection and recommendation guidance. 

**Abstract (ZH)**: 日志分析是网络安全研究的 Relevant Research Field，因为它们可以提供用于检测网络和系统威胁的信息。本文提出了一种Pipeline，使用微调的大语言模型（LLMs）进行异常检测和缓解建议，利用物联网安全日志。通过使用经典的机器学习分类器作为基线，本文对三种开源大语言模型在二元和多分类异常检测中的性能进行了比较，使用了三种策略：零样本、少样本提示和基于物联网数据集的微调。大语言模型在多类攻击分类中比相应的基线模型表现出更好的性能。通过将检测到的威胁映射到MITRE CAPEC，定义一组物联网特定的缓解操作，并使用这些操作微调模型，模型能够提供结合检测和建议指导。 

---
# Synthetic Heuristic Evaluation: A Comparison between AI- and Human-Powered Usability Evaluation 

**Title (ZH)**: 合成启发式评估：AI-powered与人工Usability评估的比较 

**Authors**: Ruican Zhong, David W. McDonald, Gary Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2507.02306)  

**Abstract**: Usability evaluation is crucial in human-centered design but can be costly, requiring expert time and user compensation. In this work, we developed a method for synthetic heuristic evaluation using multimodal LLMs' ability to analyze images and provide design feedback. Comparing our synthetic evaluations to those by experienced UX practitioners across two apps, we found our evaluation identified 73% and 77% of usability issues, which exceeded the performance of 5 experienced human evaluators (57% and 63%). Compared to human evaluators, the synthetic evaluation's performance maintained consistent performance across tasks and excelled in detecting layout issues, highlighting potential attentional and perceptual strengths of synthetic evaluation. However, synthetic evaluation struggled with recognizing some UI components and design conventions, as well as identifying across screen violations. Additionally, testing synthetic evaluations over time and accounts revealed stable performance. Overall, our work highlights the performance differences between human and LLM-driven evaluations, informing the design of synthetic heuristic evaluations. 

**Abstract (ZH)**: 基于多模态LLM的合成 heuristic 评估方法在用户体验评估中的性能研究 

---
# DoMIX: An Efficient Framework for Exploiting Domain Knowledge in Fine-Tuning 

**Title (ZH)**: DoMIX：一种在微调中利用领域知识的有效框架 

**Authors**: Dohoon Kim, Donghun Kang, Taesup Moon  

**Link**: [PDF](https://arxiv.org/pdf/2507.02302)  

**Abstract**: Domain-Adaptive Pre-training (DAP) has recently gained attention for its effectiveness in fine-tuning pre-trained models. Building on this, continual DAP has been explored to develop pre-trained models capable of incrementally incorporating different domain datasets. However, existing continual DAP methods face several limitations: (1) high computational cost and GPU memory usage during training; (2) sensitivity to incremental data order; and (3) providing a single, generalized model for all end tasks, which contradicts the essence of DAP. In this paper, we propose DoMIX, a novel approach that addresses these challenges by leveraging LoRA modules, a representative parameter-efficient fine-tuning (PEFT) method. Our approach enables efficient and parallel domain-adaptive pre-training that is robust to domain order and effectively utilizes accumulated knowledge to provide tailored pre-trained models for specific tasks. We also demonstrate that our method can be extended beyond the DAP setting to standard LLM fine-tuning scenarios. Code is available at this https URL. 

**Abstract (ZH)**: DoMIX：一种通过LoRA模块解决连续域自适应预训练挑战的方法 

---
# MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent 

**Title (ZH)**: MemAgent: 以多轮次卷积RL为基础的内存代理重塑长上下文LLM 

**Authors**: Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, Ya-Qin Zhang, Wei-Ying Ma, Jingjing Liu, Mingxuan Wang, Hao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.02259)  

**Abstract**: Despite improvements by length extrapolation, efficient attention and memory modules, handling infinitely long documents with linear complexity without performance degradation during extrapolation remains the ultimate challenge in long-text processing. We directly optimize for long-text tasks in an end-to-end fashion and introduce a novel agent workflow, MemAgent, which reads text in segments and updates the memory using an overwrite strategy. We extend the DAPO algorithm to facilitate training via independent-context multi-conversation generation. MemAgent has demonstrated superb long-context capabilities, being able to extrapolate from an 8K context trained on 32K text to a 3.5M QA task with performance loss < 5% and achieves 95%+ in 512K RULER test. 

**Abstract (ZH)**: 尽管通过长度外推、高效注意力和记忆模块的改进有所提升，但在不牺牲性能的情况下以线性复杂度处理无限长文档依然是长文本处理领域的终极挑战。我们以端到端的方式直接优化长文本任务，并引入了一种新颖的智能体工作流MemAgent，该工作流分段读取文本并使用覆盖策略更新记忆。我们扩展了DAPO算法，通过独立上下文多对话生成来促进训练。实验证明，MemAgent具备卓越的长上下文处理能力，能够从使用32K文本训练的8K上下文外推到3.5M QA任务，性能损失小于5%，并在512K RULER测试中达到95%以上。 

---
# SurgVisAgent: Multimodal Agentic Model for Versatile Surgical Visual Enhancement 

**Title (ZH)**: SurgVisAgent: 多模态代理模型淂 

**Authors**: Zeyu Lei, Hongyuan Yu, Jinlin Wu, Zhen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.02252)  

**Abstract**: Precise surgical interventions are vital to patient safety, and advanced enhancement algorithms have been developed to assist surgeons in decision-making. Despite significant progress, these algorithms are typically designed for single tasks in specific scenarios, limiting their effectiveness in complex real-world situations. To address this limitation, we propose SurgVisAgent, an end-to-end intelligent surgical vision agent built on multimodal large language models (MLLMs). SurgVisAgent dynamically identifies distortion categories and severity levels in endoscopic images, enabling it to perform a variety of enhancement tasks such as low-light enhancement, overexposure correction, motion blur elimination, and smoke removal. Specifically, to achieve superior surgical scenario understanding, we design a prior model that provides domain-specific knowledge. Additionally, through in-context few-shot learning and chain-of-thought (CoT) reasoning, SurgVisAgent delivers customized image enhancements tailored to a wide range of distortion types and severity levels, thereby addressing the diverse requirements of surgeons. Furthermore, we construct a comprehensive benchmark simulating real-world surgical distortions, on which extensive experiments demonstrate that SurgVisAgent surpasses traditional single-task models, highlighting its potential as a unified solution for surgical assistance. 

**Abstract (ZH)**: 精确的外科干预对于患者安全至关重要，先进的增强算法已被开发出来以帮助外科医生进行决策。尽管取得了显著进展，但这些算法通常仅为特定场景中的单一任务设计，限制了其在复杂现实情况中的有效性。为解决这一限制，我们提出了SurgVisAgent，这是一种基于多模态大型语言模型（MLLMs）构建的端到端智能外科视觉代理。SurgVisAgent动态识别内窥镜图像中的失真类别和严重程度级别，使其能够执行多种增强任务，如低光增强、过度曝光校正、运动模糊消除和烟雾去除。具体来说，为了实现优越的外科情景理解，我们设计了一种先验模型提供领域特定知识。此外，通过上下文中的少样本学习和链式思考（CoT）推理，SurgVisAgent能够为广泛类型的失真及其严重程度级别提供定制化的图像增强，从而满足外科医生的各种需求。此外，我们构建了一个全面的基准测试，模拟真实的外科失真情况，大量的实验结果显示SurgVisAgent超越了传统的单一任务模型，凸显其作为外科辅助统一解决方案的潜力。 

---
# Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer 

**Title (ZH)**: 潜在的思维链？深度递归变压器的解码 

**Authors**: Wenquan Lu, Yuechuan Yang, Kyle Lee, Yanshu Li, Enqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.02199)  

**Abstract**: Chain-of-thought (CoT) reasoning has enabled transformer-based language models to excel at complex mathematics and multi-step planning. However, in standard decoder-only architectures, these reasoning steps are externalized in natural language, improving interpretability at the cost of efficiency. To capture reasoning that is not easily represented in words, many works have explored recurrent architectures that aim to internalize reasoning in latent space, potentially supporting latent CoT. In this paper, we investigate whether such reasoning structures emerge in Huginn-3.5B, a depth-recurrent Transformer that reuses layers at inference time without increasing parameter count. We examine the model's internal behavior on arithmetic tasks using a suite of probing techniques including the Logit Lens and Coda Lens. Our findings reveal limited evidence of interpretable latent CoT by tracking rank trajectories of final and intermediate result tokens. Furthermore, we uncover significant probing inconsistencies across recurrent blocks, where the interpretability of hidden states depends heavily on both the layer index and the decoding method. Finally, we empirically show that increasing recurrence depth yields only marginal gains and falls well short of models that explicitly externalize reasoning steps. The code is available at this https URL. 

**Abstract (ZH)**: Huginn-3.5B中是否存在深度递归Transformer中的可解释潜在CoT结构？ 

---
# Reasoning or Not? A Comprehensive Evaluation of Reasoning LLMs for Dialogue Summarization 

**Title (ZH)**: 推理还是不推理？面向对话总结的推理型LLMs综合评估 

**Authors**: Keyan Jin, Yapeng Wang, Leonel Santos, Tao Fang, Xu Yang, Sio Kei Im, Hugo Gonçalo Oliveira  

**Link**: [PDF](https://arxiv.org/pdf/2507.02145)  

**Abstract**: Dialogue summarization is a challenging task with significant practical value in customer service, meeting analysis, and conversational AI. Although large language models (LLMs) have achieved substantial progress in summarization tasks, the performance of step-by-step reasoning architectures-specifically Long Chain-of-Thought (CoT) implementations such as OpenAI-o1 and DeepSeek-R1-remains unexplored for dialogue scenarios requiring concurrent abstraction and conciseness. In this work, we present the first comprehensive and systematic evaluation of state-of-the-art reasoning LLMs and non-reasoning LLMs across three major paradigms-generic, role-oriented, and query-oriented dialogue summarization. Our study spans diverse languages, domains, and summary lengths, leveraging strong benchmarks (SAMSum, DialogSum, CSDS, and QMSum) and advanced evaluation protocols that include both LLM-based automatic metrics and human-inspired criteria. Contrary to trends in other reasoning-intensive tasks, our findings show that explicit stepwise reasoning does not consistently improve dialogue summarization quality. Instead, reasoning LLMs are often prone to verbosity, factual inconsistencies, and less concise summaries compared to their non-reasoning counterparts. Through scenario-specific analyses and detailed case studies, we further identify when and why explicit reasoning may fail to benefit-or even hinder-summarization in complex dialogue contexts. Our work provides new insights into the limitations of current reasoning LLMs and highlights the need for targeted modeling and evaluation strategies for real-world dialogue summarization. 

**Abstract (ZH)**: 对话摘要是客服、会议分析和会话AI等领域中具有重要实践价值的挑战性任务。尽管大型语言模型（LLMs）在摘要任务中取得了显著进展，但对于需要同时进行抽象和简洁处理的对话场景，逐步推理架构——特别是OpenAI-o1和DeepSeek-R1等长链推理实现——的表现仍然未被探索。在本研究中，我们首次对最先进的推理LLMs和非推理LLMs在三大主要范式（通用、角色导向和查询导向）下的对话摘要进行了综合和系统的评估。我们的研究涵盖了多种语言、领域和摘要长度，利用了强大的基准（SAMSum、DialogSum、CSDS和QMSum）和先进的评估协议，包括基于LLMs的自动评价指标和基于人类评估的标准。与其他推理密集型任务的趋势相反，我们的研究发现显式的步骤推理并不总是提升对话摘要的质量。相反，推理LLMs通常会产生冗长、事实不一致和不那么简洁的摘要，相较于非推理模型更显得如此。通过特定场景的分析和详细的案例研究，我们进一步确定了在复杂对话情境中显式推理为何可能未能提高——甚至可能妨碍——摘要的质量。我们的工作提供了当前推理LLMs局限性的新见解，并强调了针对实际对话场景的摘要进行针对性建模和评估策略的必要性。 

---
# When LLMs Disagree: Diagnosing Relevance Filtering Bias and Retrieval Divergence in SDG Search 

**Title (ZH)**: 当大语言模型产生分歧：SDG搜索中相关性筛选偏见和检索分歧诊断 

**Authors**: William A. Ingram, Bipasha Banerjee, Edward A. Fox  

**Link**: [PDF](https://arxiv.org/pdf/2507.02139)  

**Abstract**: Large language models (LLMs) are increasingly used to assign document relevance labels in information retrieval pipelines, especially in domains lacking human-labeled data. However, different models often disagree on borderline cases, raising concerns about how such disagreement affects downstream retrieval. This study examines labeling disagreement between two open-weight LLMs, LLaMA and Qwen, on a corpus of scholarly abstracts related to Sustainable Development Goals (SDGs) 1, 3, and 7. We isolate disagreement subsets and examine their lexical properties, rank-order behavior, and classification predictability. Our results show that model disagreement is systematic, not random: disagreement cases exhibit consistent lexical patterns, produce divergent top-ranked outputs under shared scoring functions, and are distinguishable with AUCs above 0.74 using simple classifiers. These findings suggest that LLM-based filtering introduces structured variability in document retrieval, even under controlled prompting and shared ranking logic. We propose using classification disagreement as an object of analysis in retrieval evaluation, particularly in policy-relevant or thematic search tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在信息检索管道中 increasingly 用于分配文档相关性标签，特别是在缺乏人类标注数据的领域。然而，不同模型在边界情况上经常存在分歧，这引发了关于这种分歧如何影响下游检索的问题。本研究考察了两种开放权重 LLM（LLaMA 和 Qwen）在与可持续发展目标（SDGs）1、3 和 7 相关的学术摘要集上的标签分歧。我们隔离分歧子集并研究其词级属性、排序行为和分类可预测性。研究结果显示，模型分歧是系统性的而非随机的：分歧案例表现出一致的词级模式，在共享评分函数下产生分歧的最高排名输出，并且使用简单分类器的 AUC 值超过 0.74 可以将这些分歧区分开来。这些发现表明，基于 LLM 的过滤会在受控提示和共享排名逻辑下引入文档检索中的结构化变异性。我们建议在检索评估中将分类分歧作为分析对象，特别是在政策相关或主题搜索任务中。 

---
# Energy-Based Transformers are Scalable Learners and Thinkers 

**Title (ZH)**: 基于能量的学习与思考可扩展变换器 

**Authors**: Alexi Gladstone, Ganesh Nanduru, Md Mofijul Islam, Peixuan Han, Hyeonjeong Ha, Aman Chadha, Yilun Du, Heng Ji, Jundong Li, Tariq Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2507.02092)  

**Abstract**: Inference-time computation techniques, analogous to human System 2 Thinking, have recently become popular for improving model performances. However, most existing approaches suffer from several limitations: they are modality-specific (e.g., working only in text), problem-specific (e.g., verifiable domains like math and coding), or require additional supervision/training on top of unsupervised pretraining (e.g., verifiers or verifiable rewards). In this paper, we ask the question "Is it possible to generalize these System 2 Thinking approaches, and develop models that learn to think solely from unsupervised learning?" Interestingly, we find the answer is yes, by learning to explicitly verify the compatibility between inputs and candidate-predictions, and then re-framing prediction problems as optimization with respect to this verifier. Specifically, we train Energy-Based Transformers (EBTs) -- a new class of Energy-Based Models (EBMs) -- to assign an energy value to every input and candidate-prediction pair, enabling predictions through gradient descent-based energy minimization until convergence. Across both discrete (text) and continuous (visual) modalities, we find EBTs scale faster than the dominant Transformer++ approach during training, achieving an up to 35% higher scaling rate with respect to data, batch size, parameters, FLOPs, and depth. During inference, EBTs improve performance with System 2 Thinking by 29% more than the Transformer++ on language tasks, and EBTs outperform Diffusion Transformers on image denoising while using fewer forward passes. Further, we find that EBTs achieve better results than existing models on most downstream tasks given the same or worse pretraining performance, suggesting that EBTs generalize better than existing approaches. Consequently, EBTs are a promising new paradigm for scaling both the learning and thinking capabilities of models. 

**Abstract (ZH)**: 基于推理时计算技术的通用化：从无监督学习中学习思维的模型 

---
# Large Language Models for Crash Detection in Video: A Survey of Methods, Datasets, and Challenges 

**Title (ZH)**: 大规模语言模型在视频碰撞检测中的应用：方法、数据集及挑战综述 

**Authors**: Sanjeda Akter, Ibne Farabi Shihab, Anuj Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2507.02074)  

**Abstract**: Crash detection from video feeds is a critical problem in intelligent transportation systems. Recent developments in large language models (LLMs) and vision-language models (VLMs) have transformed how we process, reason about, and summarize multimodal information. This paper surveys recent methods leveraging LLMs for crash detection from video data. We present a structured taxonomy of fusion strategies, summarize key datasets, analyze model architectures, compare performance benchmarks, and discuss ongoing challenges and opportunities. Our review provides a foundation for future research in this fast-growing intersection of video understanding and foundation models. 

**Abstract (ZH)**: 基于视频流的碰撞检测是智能交通系统中的关键问题。大规模语言模型（LLMs）和视觉-语言模型（VLMs）的Recent发展变革了我们处理、推理和总结多模态信息的方式。本文综述了利用LLMs进行视频数据碰撞检测的Recent方法。我们提出了融合策略的结构化分类，总结了关键数据集，分析了模型结构，比较了性能基准，并讨论了现存的挑战与机遇。我们的综述为视频理解与基础模型这一快速发展领域的未来研究奠定了基础。 

---
# MGC: A Compiler Framework Exploiting Compositional Blindness in Aligned LLMs for Malware Generation 

**Title (ZH)**: MGC：一种利用对齐大型语言模型中组成盲性编译框架的恶意软件生成 

**Authors**: Lu Yan, Zhuo Zhang, Xiangzhe Xu, Shengwei An, Guangyu Shen, Zhou Xuan, Xuan Chen, Xiangyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02057)  

**Abstract**: Large language models (LLMs) have democratized software development, reducing the expertise barrier for programming complex applications. This accessibility extends to malicious software development, raising significant security concerns. While LLM providers have implemented alignment mechanisms to prevent direct generation of overtly malicious code, these safeguards predominantly evaluate individual prompts in isolation, overlooking a critical vulnerability: malicious operations can be systematically decomposed into benign-appearing sub-tasks. In this paper, we introduce the Malware Generation Compiler (MGC), a novel framework that leverages this vulnerability through modular decomposition and alignment-evasive generation. MGC employs a specialized Malware Description Intermediate Representation (MDIR) to bridge high-level malicious intents and benign-appearing code snippets. Extensive evaluation demonstrates that our attack reliably generates functional malware across diverse task specifications and categories, outperforming jailbreaking methods by +365.79% and underground services by +78.07% in correctness on three benchmark datasets. Case studies further show that MGC can reproduce and even enhance 16 real-world malware samples. This work provides critical insights for security researchers by exposing the risks of compositional attacks against aligned AI systems. Demonstrations are available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）已民主化软件开发，降低了开发复杂应用程序的专业门槛。这种易用性也扩展到了恶意软件开发，引发了重大的安全关切。尽管LLM提供商已实施对齐机制以防止直接生成明显的恶意代码，这些保护措施主要在孤立地评估个别提示，忽视了一个关键漏洞：恶意操作可以系统地分解为表面上无害的子任务。在本文中，我们引入了恶意软件生成编译器（MGC），这是一种新的框架，利用这一漏洞通过模块化分解和对齐规避生成。MGC使用专门的恶意软件描述中间表示（MDIR）来连接高层级的恶意意图和表面上无害的代码片段。广泛的评估表明，我们的攻击能够在多种任务规范和类别中可靠地生成功能性恶意软件，相对于三种基准数据集上的越狱方法和地下服务，正确性分别高出365.79%和78.07%。案例研究进一步表明，MGC可以重现甚至增强16个真实世界的恶意软件样本。这项工作为安全研究人员提供了关键见解，暴露了针对对齐AI系统的组合式攻击的风险。演示可在[该链接]上获得。 

---
# Towards a Playground to Democratize Experimentation and Benchmarking of AI Agents for Network Troubleshooting 

**Title (ZH)**: 朝着为网络故障排查中的AI代理的实验和基准测试 democratize 平台的方向 

**Authors**: Zhihao Wang, Alessandro Cornacchia, Franco Galante, Carlo Centofanti, Alessio Sacco, Dingde Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.01997)  

**Abstract**: Recent research has demonstrated the effectiveness of Artificial Intelligence (AI), and more specifically, Large Language Models (LLMs), in supporting network configuration synthesis and automating network diagnosis tasks, among others. In this preliminary work, we restrict our focus to the application of AI agents to network troubleshooting and elaborate on the need for a standardized, reproducible, and open benchmarking platform, where to build and evaluate AI agents with low operational effort. 

**Abstract (ZH)**: 近期的研究表明，人工智能（AI），特别是大规模语言模型（LLMs），在支持网络配置合成和网络故障诊断自动化等方面的有效性。在本初步工作中，我们将重点放在将AI代理应用于网络故障排除上，并强调需要一个标准化、可重复且开放的基准平台，以便以较低的操作成本构建和评估AI代理。 

---
# Integrating Large Language Models in Financial Investments and Market Analysis: A Survey 

**Title (ZH)**: 将大规模语言模型集成到金融投资和市场分析中：一个综述 

**Authors**: Sedigheh Mahdavi, Jiating, Chen, Pradeep Kumar Joshi, Lina Huertas Guativa, Upmanyu Singh  

**Link**: [PDF](https://arxiv.org/pdf/2507.01990)  

**Abstract**: Large Language Models (LLMs) have been employed in financial decision making, enhancing analytical capabilities for investment strategies. Traditional investment strategies often utilize quantitative models, fundamental analysis, and technical indicators. However, LLMs have introduced new capabilities to process and analyze large volumes of structured and unstructured data, extract meaningful insights, and enhance decision-making in real-time. This survey provides a structured overview of recent research on LLMs within the financial domain, categorizing research contributions into four main frameworks: LLM-based Frameworks and Pipelines, Hybrid Integration Methods, Fine-Tuning and Adaptation Approaches, and Agent-Based Architectures. This study provides a structured review of recent LLMs research on applications in stock selection, risk assessment, sentiment analysis, trading, and financial forecasting. By reviewing the existing literature, this study highlights the capabilities, challenges, and potential directions of LLMs in financial markets. 

**Abstract (ZH)**: 大型语言模型（LLMs）在金融决策中的应用增强了投资策略的分析能力。传统投资策略通常利用量化模型、基本面分析和技术指标。然而，LLMs引入了处理和分析大量结构化和非结构化数据、提取有意义的洞察并实现实时决策的新能力。本文提供了一个结构化的综述，概述了金融领域中LLMs的最新研究，将研究贡献分为四大框架：基于LLM的框架和管道、混合集成方法、微调和适应方法以及基于代理的架构。本文提供了一篇结构化的综述，概述了近年来LLMs在股票选择、风险评估、情绪分析、交易和金融预测中的应用。通过对现有文献的回顾，本文突出了LLMs在金融市场中的能力和潜力。 

---
