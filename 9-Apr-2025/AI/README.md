# FEABench: Evaluating Language Models on Multiphysics Reasoning Ability 

**Title (ZH)**: FEABench: 评估语言模型在多物理推理能力上的表现 

**Authors**: Nayantara Mudur, Hao Cui, Subhashini Venugopalan, Paul Raccuglia, Michael P. Brenner, Peter Norgaard  

**Link**: [PDF](https://arxiv.org/pdf/2504.06260)  

**Abstract**: Building precise simulations of the real world and invoking numerical solvers to answer quantitative problems is an essential requirement in engineering and science. We present FEABench, a benchmark to evaluate the ability of large language models (LLMs) and LLM agents to simulate and solve physics, mathematics and engineering problems using finite element analysis (FEA). We introduce a comprehensive evaluation scheme to investigate the ability of LLMs to solve these problems end-to-end by reasoning over natural language problem descriptions and operating COMSOL Multiphysics$^\circledR$, an FEA software, to compute the answers. We additionally design a language model agent equipped with the ability to interact with the software through its Application Programming Interface (API), examine its outputs and use tools to improve its solutions over multiple iterations. Our best performing strategy generates executable API calls 88% of the time. LLMs that can successfully interact with and operate FEA software to solve problems such as those in our benchmark would push the frontiers of automation in engineering. Acquiring this capability would augment LLMs' reasoning skills with the precision of numerical solvers and advance the development of autonomous systems that can tackle complex problems in the real world. The code is available at this https URL 

**Abstract (ZH)**: 构建真实世界的精确仿真并调用数值求解器以解答定量问题是工程和科学领域的基本要求。我们提出了FEABench，用于评估大型语言模型（LLMs）及其代理利用有限元分析（FEA）模拟和解决物理学、数学和工程问题的能力。我们引入了一种全面的评估方案，通过推理自然语言问题描述并操作COMSOL Multiphysics®（一种FEA软件），来调查LLMs端到端解决这些问题的能力。此外，我们设计了一个具备与软件交互能力的语言模型代理，并通过评估其输出和使用工具在多轮迭代中改进其解决方案。我们的最佳策略88%的时间生成可执行的API调用。能够成功与FEA软件交互并解决基准中问题的LLMs，将推动工程自动化领域的边界。获得这一能力将增强LLMs的推理技能，与数值求解器的精确性相结合，推进能够应对现实世界复杂问题的自主系统的开发。代码可在以下链接获取：this https URL 

---
# TxGemma: Efficient and Agentic LLMs for Therapeutics 

**Title (ZH)**: TxGemma: 高效自主的治疗用语言模型 

**Authors**: Eric Wang, Samuel Schmidgall, Paul F. Jaeger, Fan Zhang, Rory Pilgrim, Yossi Matias, Joelle Barral, David Fleet, Shekoofeh Azizi  

**Link**: [PDF](https://arxiv.org/pdf/2504.06196)  

**Abstract**: Therapeutic development is a costly and high-risk endeavor that is often plagued by high failure rates. To address this, we introduce TxGemma, a suite of efficient, generalist large language models (LLMs) capable of therapeutic property prediction as well as interactive reasoning and explainability. Unlike task-specific models, TxGemma synthesizes information from diverse sources, enabling broad application across the therapeutic development pipeline. The suite includes 2B, 9B, and 27B parameter models, fine-tuned from Gemma-2 on a comprehensive dataset of small molecules, proteins, nucleic acids, diseases, and cell lines. Across 66 therapeutic development tasks, TxGemma achieved superior or comparable performance to the state-of-the-art generalist model on 64 (superior on 45), and against state-of-the-art specialist models on 50 (superior on 26). Fine-tuning TxGemma models on therapeutic downstream tasks, such as clinical trial adverse event prediction, requires less training data than fine-tuning base LLMs, making TxGemma suitable for data-limited applications. Beyond these predictive capabilities, TxGemma features conversational models that bridge the gap between general LLMs and specialized property predictors. These allow scientists to interact in natural language, provide mechanistic reasoning for predictions based on molecular structure, and engage in scientific discussions. Building on this, we further introduce Agentic-Tx, a generalist therapeutic agentic system powered by Gemini 2.5 that reasons, acts, manages diverse workflows, and acquires external domain knowledge. Agentic-Tx surpasses prior leading models on the Humanity's Last Exam benchmark (Chemistry & Biology) with 52.3% relative improvement over o3-mini (high) and 26.7% over o3-mini (high) on GPQA (Chemistry) and excels with improvements of 6.3% (ChemBench-Preference) and 2.4% (ChemBench-Mini) over o3-mini (high). 

**Abstract (ZH)**: 疗法治开发是一项成本高、风险大的过程，往往伴随着高失败率。为解决这一问题，我们引入了TxGemma，这是一个高效的、通用的大语言模型套件，能够进行治疗性质预测、交互推理和可解释性分析。与任务特定模型不同，TxGemma能够综合多种来源的信息，使其在治疗开发管道的各个阶段具有广泛的应用。该套件包括2亿、9亿和27亿参数的模型，这些模型基于Gemmar-2，在涉及小分子、蛋白质、核酸、疾病和细胞系的全面数据集上进行微调。在66个治疗开发任务中，TxGemma在64个任务上优于或达到了最先进的通用模型的最佳性能（其中45个任务表现更优），在50个任务上优于最先进的专用模型（其中26个任务表现更优）。针对诸如临床试验不良事件预测等治疗下游任务微调TxGemma模型需要比微调基础大语言模型更少的训练数据，使得TxGemma适用于数据受限的应用。除了预测能力之外，TxGemma还配备了对话模型，填补了一般大语言模型和专门属性预测器之间的差距。这些模型允许科学家自然语言交流，根据分子结构提供机制推理，并参与科学讨论。在此基础上，我们进一步引入了由Gemini 2.5驱动的通用治疗决策系统Agentic-Tx，该系统能够进行推理、行动、管理多样化的工作流，并获取外部专业知识。Agentic-Tx在人类最终考试基准（ Chemistry & Biology）上超越了之前的领先模型，相对改进了52.3%（相对于o3-mini 高版本），在GPQA（ Chemistry）上相对改进了26.7%（相对于o3-mini 高版本），在ChemBench-Preference和ChemBench-Mini上分别取得了6.3%和2.4%的改进（相对于o3-mini 高版本）。 

---
# SkillFlow: Efficient Skill and Code Transfer Through Communication in Adapting AI Agents 

**Title (ZH)**: SkillFlow：通过适应性AI代理沟通实现高效技能和代码转移 

**Authors**: Pagkratios Tagkopoulos, Fangzhou Li, Ilias Tagkopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2504.06188)  

**Abstract**: AI agents are autonomous systems that can execute specific tasks based on predefined programming. Here, we present SkillFlow, a modular, technology-agnostic framework that allows agents to expand their functionality in an ad-hoc fashion by acquiring new skills from their environment or other agents. We present a theoretical model that examines under which conditions this framework would be beneficial, and we then explore SkillFlow's ability to accelerate task completion and lead to lower cumulative costs in a real-world application, namely scheduling agents for calendar events. We demonstrate that within a few iterations, SkillFlow leads to considerable (24.8%, p-value = $6.4\times10^{-3}$) gains in time and cost, especially when the communication cost is high. Finally, we draw analogies from well-studied biological systems and compare this framework to that of lateral gene transfer, a significant process of adaptation and evolution in novel environments. 

**Abstract (ZH)**: 基于预定义编程的AI代理是自主系统，能够根据特定编程执行特定任务。本文介绍了SkillFlow，这是一种模块化、技术无关的框架，允许代理通过从环境或其他代理获取新技能以即兴方式扩展其功能。我们提出了一种理论模型，探讨了在何种条件下该框架会有益，并研究了SkillFlow在实际应用中加速任务完成以及降低累积成本的能力，具体应用为日历事件调度代理。结果显示，在几轮迭代后，SkillFlow在时间和成本方面取得了显著（24.8%，p值=6.4×10^-3）的提升，尤其是在通信成本较高时更为明显。最后，我们从研究较为成熟的生物系统中汲取灵感，将此框架与水平基因转移进行类比，水平基因转移是适应和演化的重要过程，尤其在新型环境中。 

---
# Decentralizing AI Memory: SHIMI, a Semantic Hierarchical Memory Index for Scalable Agent Reasoning 

**Title (ZH)**: 分散化AI记忆：SHIMI，一种语义分层记忆索引，用于可扩展的智能体推理 

**Authors**: Tooraj Helmi  

**Link**: [PDF](https://arxiv.org/pdf/2504.06135)  

**Abstract**: Retrieval-Augmented Generation (RAG) and vector-based search have become foundational tools for memory in AI systems, yet they struggle with abstraction, scalability, and semantic precision - especially in decentralized environments. We present SHIMI (Semantic Hierarchical Memory Index), a unified architecture that models knowledge as a dynamically structured hierarchy of concepts, enabling agents to retrieve information based on meaning rather than surface similarity. SHIMI organizes memory into layered semantic nodes and supports top-down traversal from abstract intent to specific entities, offering more precise and explainable retrieval. Critically, SHIMI is natively designed for decentralized ecosystems, where agents maintain local memory trees and synchronize them asynchronously across networks. We introduce a lightweight sync protocol that leverages Merkle-DAG summaries, Bloom filters, and CRDT-style conflict resolution to enable partial synchronization with minimal overhead. Through benchmark experiments and use cases involving decentralized agent collaboration, we demonstrate SHIMI's advantages in retrieval accuracy, semantic fidelity, and scalability - positioning it as a core infrastructure layer for decentralized cognitive systems. 

**Abstract (ZH)**: SHIMI：语义层次记忆索引 

---
# Leanabell-Prover: Posttraining Scaling in Formal Reasoning 

**Title (ZH)**: Leanabell-Prover: 训练后权重缩放在形式推理中的应用 

**Authors**: Jingyuan Zhang, Qi Wang, Xingguang Ji, Yahui Liu, Yang Yue, Fuzheng Zhang, Di Zhang, Guorui Zhou, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2504.06122)  

**Abstract**: Recent advances in automated theorem proving (ATP) through LLMs have highlighted the potential of formal reasoning with Lean 4 codes. However, ATP has not yet be revolutionized by the recent posttraining scaling as demonstrated by Open AI O1/O3 and Deepseek R1. In this work, we investigate the entire posttraining of ATP, aiming to align it with breakthroughs in reasoning models in natural this http URL begin, we continual train current ATP models with a hybrid dataset, which consists of numerous statement-proof pairs, and additional data aimed at incorporating cognitive behaviors that emulate human reasoning and hypothesis refinement. Next, we explore reinforcement learning with the use of outcome reward returned by Lean 4 compiler. Through our designed continual training and reinforcement learning processes, we have successfully improved existing formal provers, including both DeepSeek-Prover-v1.5 and Goedel-Prover, achieving state-of-the-art performance in the field of whole-proof generation. For example, we achieve a 59.8% pass rate (pass@32) on MiniF2F. This is an on-going project and we will progressively update our findings, release our data and training details. 

**Abstract (ZH)**: Recent advances in automated theorem proving through LLMs have highlighted the potential of formal reasoning with Lean 4 codes. However, ATP has not yet been revolutionized by recent posttraining scaling as demonstrated by Open AI O1/O3 and Deepseek R1. In this work, we investigate the entire posttraining of ATP, aiming to align it with breakthroughs in reasoning models in natural language processing. To begin, we continually train current ATP models with a hybrid dataset, which consists of numerous statement-proof pairs, and additional data aimed at incorporating cognitive behaviors that emulate human reasoning and hypothesis refinement. Next, we explore reinforcement learning with the use of outcome reward returned by Lean 4 compiler. Through our designed continual training and reinforcement learning processes, we have successfully improved existing formal provers, including both DeepSeek-Prover-v1.5 and Goedel-Prover, achieving state-of-the-art performance in the field of whole-proof generation. For example, we achieve a 59.8% pass rate (pass@32) on MiniF2F. This is an on-going project and we will progressively update our findings, release our data and training details. 

---
# Information-Theoretic Reward Decomposition for Generalizable RLHF 

**Title (ZH)**: 信息论奖励分解促进通用的RLHF 

**Authors**: Liyuan Mao, Haoran Xu, Amy Zhang, Weinan Zhang, Chenjia Bai  

**Link**: [PDF](https://arxiv.org/pdf/2504.06020)  

**Abstract**: A generalizable reward model is crucial in Reinforcement Learning from Human Feedback (RLHF) as it enables correctly evaluating unseen prompt-response pairs. However, existing reward models lack this ability, as they are typically trained by increasing the reward gap between chosen and rejected responses, while overlooking the prompts that the responses are conditioned on. Consequently, when the trained reward model is evaluated on prompt-response pairs that lie outside the data distribution, neglecting the effect of prompts may result in poor generalization of the reward model. To address this issue, we decompose the reward value into two independent components: prompt-free reward and prompt-related reward. Prompt-free reward represents the evaluation that is determined only by responses, while the prompt-related reward reflects the reward that derives from both the prompt and the response. We extract these two components from an information-theoretic perspective, which requires no extra models. Subsequently, we propose a new reward learning algorithm by prioritizing data samples based on their prompt-free reward values. Through toy examples, we demonstrate that the extracted prompt-free and prompt-related rewards effectively characterize two parts of the reward model. Further, standard evaluations show that our method improves both the alignment performance and the generalization capability of the reward model. 

**Abstract (ZH)**: 可泛化奖励模型对于从人类反馈强化学习（RLHF）至关重要，因为它使得正确评估未见过的提示-响应对成为可能。然而，现有的奖励模型缺乏这一能力，因为它们通常通过增加所选和拒绝响应之间的奖励差距来进行训练，而忽视了响应所依赖的提示。因此，当训练好的奖励模型应用于数据分布之外的提示-响应对时，忽视提示的影响可能导致奖励模型泛化能力较差。为解决这一问题，我们将奖励价值分解为两个独立的组成部分：与提示无关的奖励和与提示相关的奖励。与提示无关的奖励仅由响应决定的评估，而与提示相关的奖励反映了来自提示和响应两者的奖励。我们从信息论的角度提取这两个组成部分，无需额外模型。随后，我们提出了一种新的奖励学习算法，根据数据样本的与提示无关的奖励值优先处理数据样本。通过玩具示例，我们展示了提取的与提示无关和与提示相关的奖励有效地表征了奖励模型的两部分。进一步的标准评估表明，我们的方法提高了奖励模型的对齐性能和泛化能力。 

---
# Representing Normative Regulations in OWL DL for Automated Compliance Checking Supported by Text Annotation 

**Title (ZH)**: 基于文本标注支持的自动合规检查的OWL DL中规范性规则表示 

**Authors**: Ildar Baimuratov, Denis Turygin  

**Link**: [PDF](https://arxiv.org/pdf/2504.05951)  

**Abstract**: Compliance checking is the process of determining whether a regulated entity adheres to these regulations. Currently, compliance checking is predominantly manual, requiring significant time and highly skilled experts, while still being prone to errors caused by the human factor. Various approaches have been explored to automate compliance checking, however, representing regulations in OWL DL language which enables compliance checking through OWL reasoning has not been adopted. In this work, we propose an annotation schema and an algorithm that transforms text annotations into machine-interpretable OWL DL code. The proposed approach is validated through a proof-of-concept implementation applied to examples from the building construction domain. 

**Abstract (ZH)**: 合规检查是确定受监管实体是否遵守相关法规的过程。目前，合规检查主要依赖人工进行，耗时且需要高度专业的专家，仍然容易受到人为因素导致的错误。尽管已经探索了多种自动化合规检查的方法，但通过OWL DL语言表示法规以利用OWL推理进行合规检查的方法尚未被采用。本文提出了一种标注方案和算法，将文本标注转换为机器可解释的OWL DL代码。所提出的方法通过建筑施工领域的实例概念证明实现进行了验证。 

---
# AEGIS: Human Attention-based Explainable Guidance for Intelligent Vehicle Systems 

**Title (ZH)**: AEGIS：基于人类注意力的可解释指导智能车辆系统 

**Authors**: Zhuoli Zhuang, Cheng-You Lu, Yu-Cheng Fred Chang, Yu-Kai Wang, Thomas Do, Chin-Teng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.05950)  

**Abstract**: Improving decision-making capabilities in Autonomous Intelligent Vehicles (AIVs) has been a heated topic in recent years. Despite advancements, training machines to capture regions of interest for comprehensive scene understanding, like human perception and reasoning, remains a significant challenge. This study introduces a novel framework, Human Attention-based Explainable Guidance for Intelligent Vehicle Systems (AEGIS). AEGIS utilizes human attention, converted from eye-tracking, to guide reinforcement learning (RL) models to identify critical regions of interest for decision-making. AEGIS uses a pre-trained human attention model to guide RL models to identify critical regions of interest for decision-making. By collecting 1.2 million frames from 20 participants across six scenarios, AEGIS pre-trains a model to predict human attention patterns. 

**Abstract (ZH)**: 提高自主智能车辆（AIVs）的决策能力一直是近年来的一个热点话题。尽管取得了进展，但训练机器捕捉对全面场景理解至关重要的区域，如人类感知和推理，仍然是一个重大挑战。本研究提出了一种新的框架，基于人类注意力的可解释引导自主智能车辆系统（AEGIS）。AEGIS 利用从眼动追踪转换而来的人类注意力来引导强化学习（RL）模型识别决策所需的关键区域。AEGIS 使用预训练的人类注意力模型来引导 RL 模型识别决策所需的关键区域。通过来自六个场景的 20 名参与者收集的 120 万帧数据，AEGIS 预训练了一个人类注意力模式预测模型。 

---
# Systematic Parameter Decision in Approximate Model Counting 

**Title (ZH)**: 近似模型计数中的系统参数决策 

**Authors**: Jinping Lei, Toru Takisaka, Junqiang Peng, Mingyu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2504.05874)  

**Abstract**: This paper proposes a novel approach to determining the internal parameters of the hashing-based approximate model counting algorithm $\mathsf{ApproxMC}$. In this problem, the chosen parameter values must ensure that $\mathsf{ApproxMC}$ is Probably Approximately Correct (PAC), while also making it as efficient as possible. The existing approach to this problem relies on heuristics; in this paper, we solve this problem by formulating it as an optimization problem that arises from generalizing $\mathsf{ApproxMC}$'s correctness proof to arbitrary parameter values.
Our approach separates the concerns of algorithm soundness and optimality, allowing us to address the former without the need for repetitive case-by-case argumentation, while establishing a clear framework for the latter. Furthermore, after reduction, the resulting optimization problem takes on an exceptionally simple form, enabling the use of a basic search algorithm and providing insight into how parameter values affect algorithm performance. Experimental results demonstrate that our optimized parameters improve the runtime performance of the latest $\mathsf{ApproxMC}$ by a factor of 1.6 to 2.4, depending on the error tolerance. 

**Abstract (ZH)**: 本文提出了一种确定基于哈希的近似模型计数算法 $\mathsf{ApproxMC}$ 内部参数的新型方法。在这种情况下，选择的参数值必须确保 $\mathsf{ApproxMC}$ 达到可能近似正确的（PAC）标准，同时也要使其尽可能高效。现有方法依赖于启发式方法；本文通过将 $\mathsf{ApproxMC}$ 的正确性证明推广到任意参数值，将其问题形式化为一个优化问题来解决这一问题。 

---
# Agent Guide: A Simple Agent Behavioral Watermarking Framework 

**Title (ZH)**: 智能体引导：一种简单的智能体行为水印框架 

**Authors**: Kaibo Huang, Zhongliang Yang, Linna Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.05871)  

**Abstract**: The increasing deployment of intelligent agents in digital ecosystems, such as social media platforms, has raised significant concerns about traceability and accountability, particularly in cybersecurity and digital content protection. Traditional large language model (LLM) watermarking techniques, which rely on token-level manipulations, are ill-suited for agents due to the challenges of behavior tokenization and information loss during behavior-to-action translation. To address these issues, we propose Agent Guide, a novel behavioral watermarking framework that embeds watermarks by guiding the agent's high-level decisions (behavior) through probability biases, while preserving the naturalness of specific executions (action). Our approach decouples agent behavior into two levels, behavior (e.g., choosing to bookmark) and action (e.g., bookmarking with specific tags), and applies watermark-guided biases to the behavior probability distribution. We employ a z-statistic-based statistical analysis to detect the watermark, ensuring reliable extraction over multiple rounds. Experiments in a social media scenario with diverse agent profiles demonstrate that Agent Guide achieves effective watermark detection with a low false positive rate. Our framework provides a practical and robust solution for agent watermarking, with applications in identifying malicious agents and protecting proprietary agent systems. 

**Abstract (ZH)**: 智能代理在数字生态系统中的不断增加部署引发了对跟踪性和问责性的重大关注，特别是在网络安全和数字内容保护方面。传统的基于令牌级操纵的大型语言模型（LLM）水印技术不适合智能代理，因为行为到操作转换过程中存在行为标记化和信息丢失的挑战。为了解决这些问题，我们提出了一种新的行为水印框架——Agent Guide，通过概率偏差指导智能代理的高层次决策（行为），同时保持特定执行（操作）的自然性。我们的方法将智能代理行为分为两个层次，行为（例如，选择书签）和操作（例如，使用特定标签进行书签操作），并应用于行为概率分布的水印指导偏差。我们采用基于z统计量的统计分析来检测水印，确保在多轮中可靠地提取。在具有多样智能代理配置的社交媒体场景中进行的实验表明，Agent Guide能够以较低的假阳性率有效地检测水印。该框架为智能代理水印提供了一种实用和 robust 的解决方案，应用于识别恶意代理和保护专有代理系统。 

---
# Are Generative AI Agents Effective Personalized Financial Advisors? 

**Title (ZH)**: 生成式AI代理是否有效的个性化金融顾问？ 

**Authors**: Takehiro Takayanagi, Kiyoshi Izumi, Javier Sanz-Cruzado, Richard McCreadie, Iadh Ounis  

**Link**: [PDF](https://arxiv.org/pdf/2504.05862)  

**Abstract**: Large language model-based agents are becoming increasingly popular as a low-cost mechanism to provide personalized, conversational advice, and have demonstrated impressive capabilities in relatively simple scenarios, such as movie recommendations. But how do these agents perform in complex high-stakes domains, where domain expertise is essential and mistakes carry substantial risk? This paper investigates the effectiveness of LLM-advisors in the finance domain, focusing on three distinct challenges: (1) eliciting user preferences when users themselves may be unsure of their needs, (2) providing personalized guidance for diverse investment preferences, and (3) leveraging advisor personality to build relationships and foster trust. Via a lab-based user study with 64 participants, we show that LLM-advisors often match human advisor performance when eliciting preferences, although they can struggle to resolve conflicting user needs. When providing personalized advice, the LLM was able to positively influence user behavior, but demonstrated clear failure modes. Our results show that accurate preference elicitation is key, otherwise, the LLM-advisor has little impact, or can even direct the investor toward unsuitable assets. More worryingly, users appear insensitive to the quality of advice being given, or worse these can have an inverse relationship. Indeed, users reported a preference for and increased satisfaction as well as emotional trust with LLMs adopting an extroverted persona, even though those agents provided worse advice. 

**Abstract (ZH)**: 基于大型语言模型的顾问在金融领域的有效性：应对三大挑战 

---
# Meta-Continual Learning of Neural Fields 

**Title (ZH)**: 元持续学习的神经场学习 

**Authors**: Seungyoon Woo, Junhyeog Yun, Gunhee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.05806)  

**Abstract**: Neural Fields (NF) have gained prominence as a versatile framework for complex data representation. This work unveils a new problem setting termed \emph{Meta-Continual Learning of Neural Fields} (MCL-NF) and introduces a novel strategy that employs a modular architecture combined with optimization-based meta-learning. Focused on overcoming the limitations of existing methods for continual learning of neural fields, such as catastrophic forgetting and slow convergence, our strategy achieves high-quality reconstruction with significantly improved learning speed. We further introduce Fisher Information Maximization loss for neural radiance fields (FIM-NeRF), which maximizes information gains at the sample level to enhance learning generalization, with proved convergence guarantee and generalization bound. We perform extensive evaluations across image, audio, video reconstruction, and view synthesis tasks on six diverse datasets, demonstrating our method's superiority in reconstruction quality and speed over existing MCL and CL-NF approaches. Notably, our approach attains rapid adaptation of neural fields for city-scale NeRF rendering with reduced parameter requirement. 

**Abstract (ZH)**: 基于神经场的元持续学习（MCL-NF）：模块化架构结合优化元学习的新策略 

---
# From Superficial to Deep: Integrating External Knowledge for Follow-up Question Generation Using Knowledge Graph and LLM 

**Title (ZH)**: 从表层到深度：利用知识图谱和大语言模型进行跟进问题生成的外部知识集成方法 

**Authors**: Jianyu Liu, Yi Huang, Sheng Bi, Junlan Feng, Guilin Qi  

**Link**: [PDF](https://arxiv.org/pdf/2504.05801)  

**Abstract**: In a conversational system, dynamically generating follow-up questions based on context can help users explore information and provide a better user experience. Humans are usually able to ask questions that involve some general life knowledge and demonstrate higher order cognitive skills. However, the questions generated by existing methods are often limited to shallow contextual questions that are uninspiring and have a large gap to the human level. In this paper, we propose a three-stage external knowledge-enhanced follow-up question generation method, which generates questions by identifying contextual topics, constructing a knowledge graph (KG) online, and finally combining these with a large language model to generate the final question. The model generates information-rich and exploratory follow-up questions by introducing external common sense knowledge and performing a knowledge fusion operation. Experiments show that compared to baseline models, our method generates questions that are more informative and closer to human questioning levels while maintaining contextual relevance. 

**Abstract (ZH)**: 基于上下文的外部知识增强型跟随问题生成方法：多层次生成启发用户探索信息的问题 

---
# AI-Driven Prognostics for State of Health Prediction in Li-ion Batteries: A Comprehensive Analysis with Validation 

**Title (ZH)**: 基于AI驱动的锂离子电池健康状态预测 prognostics：一种综合验证分析 

**Authors**: Tianqi Ding, Dawei Xiang, Tianyao Sun, YiJiashum Qi, Zunduo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.05728)  

**Abstract**: This paper presents a comprehensive review of AI-driven prognostics for State of Health (SoH) prediction in lithium-ion batteries. We compare the effectiveness of various AI algorithms, including FFNN, LSTM, and BiLSTM, across multiple datasets (CALCE, NASA, UDDS) and scenarios (e.g., varying temperatures and driving conditions). Additionally, we analyze the factors influencing SoH fluctuations, such as temperature and charge-discharge rates, and validate our findings through simulations. The results demonstrate that BiLSTM achieves the highest accuracy, with an average RMSE reduction of 15% compared to LSTM, highlighting its robustness in real-world applications. 

**Abstract (ZH)**: 本论文对AI驱动的锂离子电池健康状态（SoH）预测进行了全面综述，并比较了各种AI算法（包括FFNN、LSTM和BiLSTM）在多个数据集（CALCE、NASA、UDDS）和不同场景（如温度和驱动条件变化）下的有效性。此外，我们分析了影响SoH波动的因素，如温度和充放电速率，并通过仿真验证了研究结果。研究结果表明，BiLSTM在准确性方面表现最佳，与LSTM相比，平均RMSE降低了15%，突显了其在实际应用中的稳健性。 

---
# Automated Archival Descriptions with Federated Intelligence of LLMs 

**Title (ZH)**: 联邦大语言模型智能自动化档案描述 

**Authors**: Jinghua Groppe, Andreas Marquet, Annabel Walz, Sven Groppe  

**Link**: [PDF](https://arxiv.org/pdf/2504.05711)  

**Abstract**: Enforcing archival standards requires specialized expertise, and manually creating metadata descriptions for archival materials is a tedious and error-prone task. This work aims at exploring the potential of agentic AI and large language models (LLMs) in addressing the challenges of implementing a standardized archival description process. To this end, we introduce an agentic AI-driven system for automated generation of high-quality metadata descriptions of archival materials. We develop a federated optimization approach that unites the intelligence of multiple LLMs to construct optimal archival metadata. We also suggest methods to overcome the challenges associated with using LLMs for consistent metadata generation. To evaluate the feasibility and effectiveness of our techniques, we conducted extensive experiments using a real-world dataset of archival materials, which covers a variety of document types and data formats. The evaluation results demonstrate the feasibility of our techniques and highlight the superior performance of the federated optimization approach compared to single-model solutions in metadata quality and reliability. 

**Abstract (ZH)**: 实施存档标准需要专门的知识和技能，手动创建存档材料的元数据描述是一个繁琐且容易出错的任务。本研究旨在探讨自主人工智能（agentic AI）和大型语言模型（LLMs）在解决实施标准化存档描述流程挑战方面的潜力。为此，我们提出了一种基于自主人工智能的自动化生成高质量存档材料元数据描述的系统。我们开发了一种联邦优化方法，将多个LLM的智能结合起来以构建最优存档元数据。我们还提出了克服使用LLMs进行一致的元数据生成所面临挑战的方法。为了评估我们技术的可行性和有效性，我们使用包含各种文件类型和数据格式的真实世界存档材料数据集进行了广泛的实验。评估结果表明了我们技术的可行性，并强调了联邦优化方法在元数据质量和可靠性方面的优越性能，优于单一模型解决方案。 

---
# StayLTC: A Cost-Effective Multimodal Framework for Hospital Length of Stay Forecasting 

**Title (ZH)**: StayLTC：一种成本效益高的多模态医院住院时长预测框架 

**Authors**: Sudeshna Jana, Manjira Sinha, Tirthankar Dasgupta  

**Link**: [PDF](https://arxiv.org/pdf/2504.05691)  

**Abstract**: Accurate prediction of Length of Stay (LOS) in hospitals is crucial for improving healthcare services, resource management, and cost efficiency. This paper presents StayLTC, a multimodal deep learning framework developed to forecast real-time hospital LOS using Liquid Time-Constant Networks (LTCs). LTCs, with their continuous-time recurrent dynamics, are evaluated against traditional models using structured data from Electronic Health Records (EHRs) and clinical notes. Our evaluation, conducted on the MIMIC-III dataset, demonstrated that LTCs significantly outperform most of the other time series models, offering enhanced accuracy, robustness, and efficiency in resource utilization. Additionally, LTCs demonstrate a comparable performance in LOS prediction compared to time series large language models, while requiring significantly less computational power and memory, underscoring their potential to advance Natural Language Processing (NLP) tasks in healthcare. 

**Abstract (ZH)**: 准确预测医院住院时间对于提高医疗服务质量、资源管理及成本效率至关重要。本文介绍了一种基于液态时间常数网络（LTCs）的多模态深度学习框架StayLTC，用于实时预测医院住院时间。我们在MIMIC-III数据集上的评估显示，LTCs在时间序列预测任务中显著优于其他传统模型，提供了更高的准确性和资源利用效率。此外，LTCs在住院时间预测任务上的性能与时间序列大语言模型相当，但所需计算资源和内存远少于后者，突显了其在医疗自然语言处理任务中的应用潜力。 

---
# Continual Learning of Multiple Cognitive Functions with Brain-inspired Temporal Development Mechanism 

**Title (ZH)**: 基于脑启发的时间发展机制的多种认知功能连续学习 

**Authors**: Bing Han, Feifei Zhao, Yinqian Sun, Wenxuan Pan, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2504.05621)  

**Abstract**: Cognitive functions in current artificial intelligence networks are tied to the exponential increase in network scale, whereas the human brain can continuously learn hundreds of cognitive functions with remarkably low energy consumption. This advantage is in part due to the brain cross-regional temporal development mechanisms, where the progressive formation, reorganization, and pruning of connections from basic to advanced regions, facilitate knowledge transfer and prevent network redundancy. Inspired by these, we propose the Continual Learning of Multiple Cognitive Functions with Brain-inspired Temporal Development Mechanism(TD-MCL), enabling cognitive enhancement from simple to complex in Perception-Motor-Interaction(PMI) multiple cognitive task scenarios. The TD-MCL model proposes the sequential evolution of long-range connections between different cognitive modules to promote positive knowledge transfer, while using feedback-guided local connection inhibition and pruning to effectively eliminate redundancies in previous tasks, reducing energy consumption while preserving acquired knowledge. Experiments show that the proposed method can achieve continual learning capabilities while reducing network scale, without introducing regularization, replay, or freezing strategies, and achieving superior accuracy on new tasks compared to direct learning. The proposed method shows that the brain's developmental mechanisms offer a valuable reference for exploring biologically plausible, low-energy enhancements of general cognitive abilities. 

**Abstract (ZH)**: 基于大脑启发的时间发展机制实现多认知功能的持续学习（TD-MCL）：从简单到复杂感知-运动-交互（PMI）多认知任务场景中的认知增强 

---
# SciSciGPT: Advancing Human-AI Collaboration in the Science of Science 

**Title (ZH)**: SciSciGPT：推动科学学中的人类-人工智能协作 

**Authors**: Erzhuo Shao, Yifang Wang, Yifan Qian, Zhenyu Pan, Han Liu, Dashun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05559)  

**Abstract**: The increasing availability of large-scale datasets has fueled rapid progress across many scientific fields, creating unprecedented opportunities for research and discovery while posing significant analytical challenges. Recent advances in large language models (LLMs) and AI agents have opened new possibilities for human-AI collaboration, offering powerful tools to navigate this complex research landscape. In this paper, we introduce SciSciGPT, an open-source, prototype AI collaborator that uses the science of science as a testbed to explore the potential of LLM-powered research tools. SciSciGPT automates complex workflows, supports diverse analytical approaches, accelerates research prototyping and iteration, and facilitates reproducibility. Through case studies, we demonstrate its ability to streamline a wide range of empirical and analytical research tasks while highlighting its broader potential to advance research. We further propose an LLM Agent capability maturity model for human-AI collaboration, envisioning a roadmap to further improve and expand upon frameworks like SciSciGPT. As AI capabilities continue to evolve, frameworks like SciSciGPT may play increasingly pivotal roles in scientific research and discovery, unlocking further opportunities. At the same time, these new advances also raise critical challenges, from ensuring transparency and ethical use to balancing human and AI contributions. Addressing these issues may shape the future of scientific inquiry and inform how we train the next generation of scientists to thrive in an increasingly AI-integrated research ecosystem. 

**Abstract (ZH)**: 大规模数据集的不断增加促进了多个科学领域的迅速进步，创造了前所未有的研究与发现机会，同时也带来了重大的分析挑战。大型语言模型（LLMs）和AI代理的最新进展为人类与AI的合作开辟了新的可能性，提供了强大的工具以应对这一复杂的研究环境。本文介绍了SciSciGPT，这是一种开源原型AI合作者，利用科学的科学研究作为试验场，探索LLM驱动的研究工具的潜力。SciSciGPT自动化复杂的工作流，支持多样化的分析方法，加速研究原型设计和迭代，并促进可重复性。通过案例研究，我们展示了它在一系列实证和分析研究任务中简化流程的能力，并阐明了其更广泛的潜力以推动研究发展。我们进一步提出了AI代理能力成熟模型，展望了改进和扩展如SciSciGPT等框架的道路。随着AI能力的不断演进，框架如SciSciGPT可能在科学研究与发现中发挥越来越关键的作用，解锁新的机会。与此同时，这些新进展也引发了重要的挑战，从确保透明性和伦理使用到平衡人类和AI的贡献。解决这些问题可能会影响科学研究的未来，并指导我们如何培养下一代科学家在日益集成AI的研究生态系统中茁壮成长。 

---
# Prism: Dynamic and Flexible Benchmarking of LLMs Code Generation with Monte Carlo Tree Search 

**Title (ZH)**: Prism: 基于蒙特卡洛树搜索的LLM代码生成动态灵活基准测试 

**Authors**: Vahid Majdinasab, Amin Nikanjam, Foutse Khomh  

**Link**: [PDF](https://arxiv.org/pdf/2504.05500)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has outpaced traditional evaluation methods. Static benchmarks fail to capture the depth and breadth of LLM capabilities and eventually become obsolete, while most dynamic approaches either rely too heavily on LLM-based evaluation or remain constrained by predefined test sets. We introduce Prism, a flexible, dynamic benchmarking framework designed for comprehensive LLM assessment. Prism builds on three key components: (1) a tree-based state representation that models evaluation as a Markov Decision Process, (2) a Monte Carlo Tree Search algorithm adapted to uncover challenging evaluation scenarios, and (3) a multi-agent evaluation pipeline that enables simultaneous assessment of diverse capabilities. To ensure robust evaluation, Prism integrates structural measurements of tree exploration patterns with performance metrics across difficulty levels, providing detailed diagnostics of error patterns, test coverage, and solution approaches. Through extensive experiments on five state-of-the-art LLMs, we analyze how model architecture and scale influence code generation performance across varying task difficulties. Our results demonstrate Prism's effectiveness as a dynamic benchmark that evolves with model advancements while offering deeper insights into their limitations. 

**Abstract (ZH)**: 大型语言模型（LLMs）的迅速发展超出了传统评价方法的速度。静态基准无法捕捉LLM的能力深度和广度，最终变得过时，而大多数动态方法要么过于依赖LLM评价，要么受限于预定义的测试集。我们引入了Prism，一个灵活的动态基准框架，用于全面评估LLM。Prism基于三个关键组件构建：（1）一种基于树的状态表示，将评价建模为马尔可夫决策过程；（2）一种适应性蒙特卡洛树搜索算法，用于揭示具有挑战性的评价场景；（3）一种多代理评价流水线，可同时评估多种能力。为确保评估的稳健性，Prism将树探索模式的结构测量与不同难度级别的性能指标集成，提供详细的错误模式、测试覆盖和解题方法诊断。通过在五种先进LLM上的广泛实验，我们分析了模型架构和规模对不同任务难度下的代码生成性能影响。我们的结果表明，Prism是一个随着模型进步而不断进化、并提供深入见解的动态基准。 

---
# Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification 

**Title (ZH)**: 推理模型知其所以然：探查隐藏状态实现自我验证 

**Authors**: Anqi Zhang, Yulin Chen, Jane Pan, Chen Zhao, Aurojit Panda, Jinyang Li, He He  

**Link**: [PDF](https://arxiv.org/pdf/2504.05419)  

**Abstract**: Reasoning models have achieved remarkable performance on tasks like math and logical reasoning thanks to their ability to search during reasoning. However, they still suffer from overthinking, often performing unnecessary reasoning steps even after reaching the correct answer. This raises the question: can models evaluate the correctness of their intermediate answers during reasoning? In this work, we study whether reasoning models encode information about answer correctness through probing the model's hidden states. The resulting probe can verify intermediate answers with high accuracy and produces highly calibrated scores. Additionally, we find models' hidden states encode correctness of future answers, enabling early prediction of the correctness before the intermediate answer is fully formulated. We then use the probe as a verifier to decide whether to exit reasoning at intermediate answers during inference, reducing the number of inference tokens by 24\% without compromising performance. These findings confirm that reasoning models do encode a notion of correctness yet fail to exploit it, revealing substantial untapped potential to enhance their efficiency. 

**Abstract (ZH)**: 推理模型通过探测模型隐藏状态来编码答案正确性的信息，从而在推理过程中评估中间答案的正确性，提高推理效率 

---
# Interactive Explanations for Reinforcement-Learning Agents 

**Title (ZH)**: 强化学习代理的交互式解释 

**Authors**: Yotam Amitai, Ofra Amir, Guy Avni  

**Link**: [PDF](https://arxiv.org/pdf/2504.05393)  

**Abstract**: As reinforcement learning methods increasingly amass accomplishments, the need for comprehending their solutions becomes more crucial. Most explainable reinforcement learning (XRL) methods generate a static explanation depicting their developers' intuition of what should be explained and how. In contrast, literature from the social sciences proposes that meaningful explanations are structured as a dialog between the explainer and the explainee, suggesting a more active role for the user and her communication with the agent. In this paper, we present ASQ-IT -- an interactive explanation system that presents video clips of the agent acting in its environment based on queries given by the user that describe temporal properties of behaviors of interest. Our approach is based on formal methods: queries in ASQ-IT's user interface map to a fragment of Linear Temporal Logic over finite traces (LTLf), which we developed, and our algorithm for query processing is based on automata theory. User studies show that end-users can understand and formulate queries in ASQ-IT and that using ASQ-IT assists users in identifying faulty agent behaviors. 

**Abstract (ZH)**: 基于查询的交互解释系统ASQ-IT：一种基于形式方法的时间逻辑片段查询处理方法 

---
# EduPlanner: LLM-Based Multi-Agent Systems for Customized and Intelligent Instructional Design 

**Title (ZH)**: EduPlanner: 基于大语言模型的多代理系统，实现个性化和智能的教学设计 

**Authors**: Xueqiao Zhang, Chao Zhang, Jianwen Sun, Jun Xiao, Yi Yang, Yawei Luo  

**Link**: [PDF](https://arxiv.org/pdf/2504.05370)  

**Abstract**: Large Language Models (LLMs) have significantly advanced smart education in the Artificial General Intelligence (AGI) era. A promising application lies in the automatic generalization of instructional design for curriculum and learning activities, focusing on two key aspects: (1) Customized Generation: generating niche-targeted teaching content based on students' varying learning abilities and states, and (2) Intelligent Optimization: iteratively optimizing content based on feedback from learning effectiveness or test scores. Currently, a single large LLM cannot effectively manage the entire process, posing a challenge for designing intelligent teaching plans. To address these issues, we developed EduPlanner, an LLM-based multi-agent system comprising an evaluator agent, an optimizer agent, and a question analyst, working in adversarial collaboration to generate customized and intelligent instructional design for curriculum and learning activities. Taking mathematics lessons as our example, EduPlanner employs a novel Skill-Tree structure to accurately model the background mathematics knowledge of student groups, personalizing instructional design for curriculum and learning activities according to students' knowledge levels and learning abilities. Additionally, we introduce the CIDDP, an LLM-based five-dimensional evaluation module encompassing clarity, Integrity, Depth, Practicality, and Pertinence, to comprehensively assess mathematics lesson plan quality and bootstrap intelligent optimization. Experiments conducted on the GSM8K and Algebra datasets demonstrate that EduPlanner excels in evaluating and optimizing instructional design for curriculum and learning activities. Ablation studies further validate the significance and effectiveness of each component within the framework. Our code is publicly available at this https URL 

**Abstract (ZH)**: 大规模语言模型（LLMs）在人工智能通用时代（AGI）显著推动了智能教育的发展。一种有前途的应用在于基于自动化的教学设计的一般化，主要集中在两个关键方面：（1）个性化生成：根据学生不同的学习能力和状态生成针对特定需求的教学内容，（2）智能优化：根据学习效果或测试分数的反馈迭代优化内容。当前，单个大规模语言模型无法有效管理整个过程，为设计智能教学计划带来了挑战。为解决这些问题，我们开发了EduPlanner，一个基于大规模语言模型的多智能体系统，包括评估代理、优化代理和题库分析师，它们以对抗性合作的方式工作，为教学计划和学习活动生成个性化和智能化的教学设计。以数学课程为例，EduPlanner采用了一种新颖的技能树结构，准确建模学生群体的背景数学知识，并根据学生的知识水平和学习能力个性化教学设计。此外，我们引入了CIDDP，一种基于大规模语言模型的五维评估模块，涵盖清晰性、完整性、深度、实践性和相关性，全面评估数学课程计划的质量并启动智能优化。在GSM8K和代数数据集上的实验表明，EduPlanner在评估和优化教学设计方面表现出色。消融研究表明，框架内的每个组件的重要性及其有效性。我们的代码已公开在这个网址。 

---
# GOLLuM: Gaussian Process Optimized LLMs -- Reframing LLM Finetuning through Bayesian Optimization 

**Title (ZH)**: GOLLuM：高斯过程优化的大语言模型——通过贝叶斯优化重新构想大语言模型微调 

**Authors**: Bojana Ranković, Philippe Schwaller  

**Link**: [PDF](https://arxiv.org/pdf/2504.06265)  

**Abstract**: Large Language Models (LLMs) can encode complex relationships in their latent spaces, yet harnessing them for optimization under uncertainty remains challenging. We address this gap with a novel architecture that reframes LLM finetuning as Gaussian process (GP) marginal likelihood optimization via deep kernel methods. We introduce LLM-based deep kernels, jointly optimized with GPs to preserve the benefits of both - LLMs to provide a rich and flexible input space for Bayesian optimization and - GPs to model this space with predictive uncertainty for more efficient sampling. Applied to Buchwald-Hartwig reaction optimization, our method nearly doubles the discovery rate of high-performing reactions compared to static LLM embeddings (from 24% to 43% coverage of the top 5% reactions in just 50 optimization iterations). We also observe a 14% improvement over domain-specific representations without requiring specialized features. Extensive empirical evaluation across 19 benchmarks - ranging from general chemistry to reaction and molecular property optimization - demonstrates our method's robustness, generality, and consistent improvements across: (1) tasks, (2) LLM architectures (encoder, decoder, encoder-decoder), (3) pretraining domains (chemistry-related or general-purpose) and (4) hyperparameter settings (tuned once on a single dataset). Finally, we explain these improvements: joint LLM-GP optimization through marginal likelihood implicitly performs contrastive learning, aligning representations to produce (1) better-structured embedding spaces, (2) improved uncertainty calibration, and (3) more efficient sampling - without requiring any external loss. This work provides both practical advances in sample-efficient optimization and insights into what makes effective Bayesian optimization. 

**Abstract (ZH)**: 大规模语言模型（LLMs）能够在其潜在空间中编码复杂的关系，但将其用于不确定性的优化仍然具有挑战性。我们通过深度核方法将LLM微调重新框架为高斯过程（GP）边际似然优化，提出了一种新颖的架构。我们引入了基于LLM的深度核，并与GP联合优化，以保留两者的优点——LLM提供丰富的灵活输入空间以供贝叶斯优化使用，GP则使用预测不确定性来建模该空间以实现更高效的采样。应用于Buchwald-Hartwig反应优化，我们的方法在50次优化迭代中将高表现反应的发现率几乎翻了一番（从静态LLM嵌入中的24%提高到43%的前5%反应覆盖率）。我们还观察到在不需要专门特征的情况下，对领域特定表示有14%的改进。跨19个基准的广泛实证评估涵盖了从通用化学到反应和分子性质优化，证实了该方法的稳健性、通用性和在以下方面的持续改进：（1）任务，（2）LLM架构（编码器、解码器、编码器-解码器），（3）预训练领域（化学相关或通用用途）和（4）超参数设置（在单一数据集中调整一次）。最后，我们解释了这些改进：联合LLM-GP优化通过边际似然隐式执行对比学习，使表示对齐以产生（1）更好的结构化嵌入空间，（2）更好的不确定性校准，以及（3）更高效的采样——而无需任何外部损失。该工作不仅提供了样本高效优化的实际进展，还揭示了有效贝叶斯优化的内在机制。 

---
# Decentralized Federated Domain Generalization with Style Sharing: A Formal Modeling and Convergence Analysis 

**Title (ZH)**: 去中心化联邦领域泛化中的风格共享：正式建模与收敛性分析 

**Authors**: Shahryar Zehtabi, Dong-Jun Han, Seyyedali Hosseinalipour, Christopher G. Brinton  

**Link**: [PDF](https://arxiv.org/pdf/2504.06235)  

**Abstract**: Much of the federated learning (FL) literature focuses on settings where local dataset statistics remain the same between training and testing time. Recent advances in domain generalization (DG) aim to use data from source (training) domains to train a model that generalizes well to data from unseen target (testing) domains. In this paper, we are motivated by two major gaps in existing work on FL and DG: (1) the lack of formal mathematical analysis of DG objectives and training processes; and (2) DG research in FL being limited to the conventional star-topology architecture. Addressing the second gap, we develop $\textit{Decentralized Federated Domain Generalization with Style Sharing}$ ($\texttt{StyleDDG}$), a fully decentralized DG algorithm designed to allow devices in a peer-to-peer network to achieve DG based on sharing style information inferred from their datasets. Additionally, we fill the first gap by providing the first systematic approach to mathematically analyzing style-based DG training optimization. We cast existing centralized DG algorithms within our framework, and employ their formalisms to model $\texttt{StyleDDG}$. Based on this, we obtain analytical conditions under which a sub-linear convergence rate of $\texttt{StyleDDG}$ can be obtained. Through experiments on two popular DG datasets, we demonstrate that $\texttt{StyleDDG}$ can obtain significant improvements in accuracy across target domains with minimal added communication overhead compared to decentralized gradient methods that do not employ style sharing. 

**Abstract (ZH)**: 去中心化联邦风格共享领域泛化算法 

---
# From 128K to 4M: Efficient Training of Ultra-Long Context Large Language Models 

**Title (ZH)**: 从128K到4M：高效训练超长上下文大语言模型 

**Authors**: Chejian Xu, Wei Ping, Peng Xu, Zihan Liu, Boxin Wang, Mohammad Shoeybi, Bo Li, Bryan Catanzaro  

**Link**: [PDF](https://arxiv.org/pdf/2504.06214)  

**Abstract**: Long-context capabilities are essential for a wide range of applications, including document and video understanding, in-context learning, and inference-time scaling, all of which require models to process and reason over long sequences of text and multimodal data. In this work, we introduce a efficient training recipe for building ultra-long context LLMs from aligned instruct model, pushing the boundaries of context lengths from 128K to 1M, 2M, and 4M tokens. Our approach leverages efficient continued pretraining strategies to extend the context window and employs effective instruction tuning to maintain the instruction-following and reasoning abilities. Our UltraLong-8B, built on Llama3.1-Instruct with our recipe, achieves state-of-the-art performance across a diverse set of long-context benchmarks. Importantly, models trained with our approach maintain competitive performance on standard benchmarks, demonstrating balanced improvements for both long and short context tasks. We further provide an in-depth analysis of key design choices, highlighting the impacts of scaling strategies and data composition. Our findings establish a robust framework for efficiently scaling context lengths while preserving general model capabilities. We release all model weights at: this https URL. 

**Abstract (ZH)**: 长上下文能力对于文档和视频理解、上下文学习以及推理时的扩展等多种应用至关重要，这些应用要求模型能够处理和推理长文本和多模态数据。在本工作中，我们提出了一种高效的训练方法，用于构建超长上下文语言模型，将上下文长度从128K扩展到1M、2M和4M tokens。我们的方法利用高效的持续预训练策略扩展上下文窗口，并采用有效的指令微调保持指令遵循和推理能力。基于我们的方法构建的UltraLong-8B在多种长上下文基准测试中取得了最佳性能。重要的是，使用我们方法训练的模型在标准基准测试中保持了竞争力，证明了对长上下文和短上下文任务的均衡改进。我们还对关键设计选择进行了深入分析，强调了扩展策略和数据组成的影响。我们的发现建立了一个在高效扩展上下文长度的同时保留通用模型能力的稳健框架。所有模型权重已发布于：this https URL。 

---
# An experimental survey and Perspective View on Meta-Learning for Automated Algorithms Selection and Parametrization 

**Title (ZH)**: 元学习在自动化算法选择与参数化中的实验调查与视角分析 

**Authors**: Moncef Garouani  

**Link**: [PDF](https://arxiv.org/pdf/2504.06207)  

**Abstract**: Considerable progress has been made in the recent literature studies to tackle the Algorithms Selection and Parametrization (ASP) problem, which is diversified in multiple meta-learning setups. Yet there is a lack of surveys and comparative evaluations that critically analyze, summarize and assess the performance of existing methods. In this paper, we provide an overview of the state of the art in this continuously evolving field. The survey sheds light on the motivational reasons for pursuing classifiers selection through meta-learning. In this regard, Automated Machine Learning (AutoML) is usually treated as an ASP problem under the umbrella of the democratization of machine learning. Accordingly, AutoML makes machine learning techniques accessible to domain scientists who are interested in applying advanced analytics but lack the required expertise. It can ease the task of manually selecting ML algorithms and tuning related hyperparameters. We comprehensively discuss the different phases of classifiers selection based on a generic framework that is formed as an outcome of reviewing prior works. Subsequently, we propose a benchmark knowledge base of 4 millions previously learned models and present extensive comparative evaluations of the prominent methods for classifiers selection based on 08 classification algorithms and 400 benchmark datasets. The comparative study quantitatively assesses the performance of algorithms selection methods along while emphasizing the strengths and limitations of existing studies. 

**Abstract (ZH)**: 近期文献在处理算法选择与参数化（ASP）问题方面取得了显著进展，该问题在多种元学习设置中表现出多样性。然而，缺乏对现有方法进行批判性分析、总结和评估的综述和比较评估。本文提供了该不断发展的领域的一种综述，探讨了通过元学习进行分类器选择的动机原因。在此方面，自动化机器学习（AutoML）通常被视作ASP问题，作为使机器学习技术普惠的手段，使希望应用先进分析但缺乏所需专业知识的领域科学家能够利用机器学习技术。AutoML可以简化手动选择ML算法及其相关超参数调整的任务。本文基于审阅先前工作形成的一般框架，全面讨论了分类器选择的不同阶段。随后，我们提出一个包含400万之前学习模型的基准知识库，并基于8种分类算法和400个基准数据集，对分类器选择的主导方法进行了详尽的比较评估。比较研究不仅定量评估了算法选择方法的性能，还强调了现有研究的优缺点。 

---
# Heuristic Methods are Good Teachers to Distill MLPs for Graph Link Prediction 

**Title (ZH)**: 启发式方法是良好的教师，用于提炼用于图链接预测的MLP模型 

**Authors**: Zongyue Qin, Shichang Zhang, Mingxuan Ju, Tong Zhao, Neil Shah, Yizhou Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.06193)  

**Abstract**: Link prediction is a crucial graph-learning task with applications including citation prediction and product recommendation. Distilling Graph Neural Networks (GNNs) teachers into Multi-Layer Perceptrons (MLPs) students has emerged as an effective approach to achieve strong performance and reducing computational cost by removing graph dependency. However, existing distillation methods only use standard GNNs and overlook alternative teachers such as specialized model for link prediction (GNN4LP) and heuristic methods (e.g., common neighbors). This paper first explores the impact of different teachers in GNN-to-MLP distillation. Surprisingly, we find that stronger teachers do not always produce stronger students: MLPs distilled from GNN4LP can underperform those distilled from simpler GNNs, while weaker heuristic methods can teach MLPs to near-GNN performance with drastically reduced training costs. Building on these insights, we propose Ensemble Heuristic-Distilled MLPs (EHDM), which eliminates graph dependencies while effectively integrating complementary signals via a gating mechanism. Experiments on ten datasets show an average 7.93% improvement over previous GNN-to-MLP approaches with 1.95-3.32 times less training time, indicating EHDM is an efficient and effective link prediction method. 

**Abstract (ZH)**: 链接预测是包括引文预测和产品推荐在内的图学习关键任务。将图神经网络（GNNs）教师提炼成多层感知器（MLPs）学生已成为一种有效的方法，通过去除图依赖来实现强大的性能并减少计算成本。然而，现有的提炼方法仅使用标准的GNNs，忽略了专为链接预测设计的模型（如GNN4LP）和启发式方法（如共同邻居）等其他教师的影响。本文首先探讨了不同教师对GNN-to-MLP提炼的影响。令人惊讶的是，更强的教师并不总是产生更强的学生：从GNN4LP提炼的MLPs可能会逊色于从简单GNN提炼的学生，而较弱的启发式方法能在大幅减少训练成本的情况下将MLPs训练到接近GNN性能。基于这些见解，我们提出了集成启发式提炼的MLPs（EHDM），通过门控机制有效地整合互补信号，消除了图依赖。在十个数据集上的实验表明，与之前的GNN-to-MLP方法相比，EHDM平均提高了7.93%的性能，并且训练时间减少了1.95-3.32倍，表明EHDM是一种高效且有效的链接预测方法。 

---
# WoundAmbit: Bridging State-of-the-Art Semantic Segmentation and Real-World Wound Care 

**Title (ZH)**: WoundAmbit: 连接最先进的语义分割与实际伤口护理 

**Authors**: Vanessa Borst, Timo Dittus, Tassilo Dege, Astrid Schmieder, Samuel Kounev  

**Link**: [PDF](https://arxiv.org/pdf/2504.06185)  

**Abstract**: Chronic wounds affect a large population, particularly the elderly and diabetic patients, who often exhibit limited mobility and co-existing health conditions. Automated wound monitoring via mobile image capture can reduce in-person physician visits by enabling remote tracking of wound size. Semantic segmentation is key to this process, yet wound segmentation remains underrepresented in medical imaging research. To address this, we benchmark state-of-the-art deep learning models from general-purpose vision, medical imaging, and top methods from public wound challenges. For fair comparison, we standardize training, data augmentation, and evaluation, conducting cross-validationto minimize partitioning bias. We also assess real-world deployment aspects, including generalization to an out-of-distribution wound dataset, computational efficiency, and interpretability. Additionally, we propose a reference object-based approach to convert AI-generated masks into clinically relevant wound size estimates, and evaluate this, along with mask quality, for the best models based on physician assessments. Overall, the transformer-based TransNeXt showed the highest levels of generalizability. Despite variations in inference times, all models processed at least one image per second on the CPU, which is deemed adequate for the intended application. Interpretability analysis typically revealed prominent activations in wound regions, emphasizing focus on clinically relevant features. Expert evaluation showed high mask approval for all analyzed models, with VWFormer and ConvNeXtS backbone performing the best. Size retrieval accuracy was similar across models, and predictions closely matched expert annotations. Finally, we demonstrate how our AI-driven wound size estimation framework, WoundAmbit, can be integrated into a custom telehealth system. Our code will be made available on GitHub upon publication. 

**Abstract (ZH)**: 慢性伤口影响大量人群，特别是老年人和糖尿病患者，他们常常表现为活动受限和共存的健康问题。通过移动图像捕获进行的自动化伤口监测可以减少亲自就医的次数，通过实现伤口尺寸的远程跟踪。语义分割是这个过程中的关键步骤，但伤口分割在医学成像研究中仍然相对不足。为了解决这一问题，我们对比了通用视觉、医学影像以及公开伤口挑战的顶级方法中的先进深度学习模型。为确保公平比较，我们统一了训练、数据增强和评估的标准，并通过交叉验证来最小化分层偏差。我们还评估了实际部署方面的因素，包括模型对未见伤口数据集的泛化能力、计算效率和可解释性。此外，我们提出了一种基于参考对象的方法，将AI生成的掩码转换为临床相关的伤口大小估计值，并基于医生评估来评估这些模型的性能，包括掩码质量。总体而言，基于变压器的TransNeXt模型在泛化能力方面表现出最高水平。尽管推断时间存在差异，所有模型在CPU上每秒至少处理一幅图像，这被认为足够满足应用需求。可解释性分析通常显示显著激活集中在伤口区域，强调对临床相关特征的关注。专家评估表明，所有分析的模型都获得了高掩码批准率，VWFormer和ConvNeXtS骨干表现最佳。伤口大小检索的准确性在不同模型间相似，预测值与专家标注高度一致。最后，我们展示了如何将我们的AI驱动的伤口大小估计框架WoundAmbit整合到自定义的远程医疗系统中。代码将在发表后发布于GitHub上。 

---
# A Self-Supervised Framework for Space Object Behaviour Characterisation 

**Title (ZH)**: 自我监督的空间目标行为特征化框架 

**Authors**: Ian Groves, Andrew Campbell, James Fernandes, Diego Rodriguez, Paul Murray, Massimiliano Vasile, Victoria Nockles  

**Link**: [PDF](https://arxiv.org/pdf/2504.06176)  

**Abstract**: Foundation Models, pre-trained on large unlabelled datasets before task-specific fine-tuning, are increasingly being applied to specialised domains. Recent examples include ClimaX for climate and Clay for satellite Earth observation, but a Foundation Model for Space Object Behavioural Analysis has not yet been developed. As orbital populations grow, automated methods for characterising space object behaviour are crucial for space safety. We present a Space Safety and Sustainability Foundation Model focusing on space object behavioural analysis using light curves (LCs). We implemented a Perceiver-Variational Autoencoder (VAE) architecture, pre-trained with self-supervised reconstruction and masked reconstruction on 227,000 LCs from the MMT-9 observatory. The VAE enables anomaly detection, motion prediction, and LC generation. We fine-tuned the model for anomaly detection & motion prediction using two independent LC simulators (CASSANDRA and GRIAL respectively), using CAD models of boxwing, Sentinel-3, SMOS, and Starlink platforms. Our pre-trained model achieved a reconstruction error of 0.01%, identifying potentially anomalous light curves through reconstruction difficulty. After fine-tuning, the model scored 88% and 82% accuracy, with 0.90 and 0.95 ROC AUC scores respectively in both anomaly detection and motion mode prediction (sun-pointing, spin, etc.). Analysis of high-confidence anomaly predictions on real data revealed distinct patterns including characteristic object profiles and satellite glinting. Here, we demonstrate how self-supervised learning can simultaneously enable anomaly detection, motion prediction, and synthetic data generation from rich representations learned in pre-training. Our work therefore supports space safety and sustainability through automated monitoring and simulation capabilities. 

**Abstract (ZH)**: 基于大型未标注数据集预训练后再进行任务特定微调的基础模型正 increasingly被应用于专门领域。近期的例子包括用于气候分析的ClimaX和用于卫星地球观测的Clay，但尚未开发出用于太空物体行为分析的基础模型。随着轨道物体数量的增长，自动化的太空物体行为表征方法对于太空安全至关重要。我们提出了一种专注于太空物体行为分析的太空安全与可持续性基础模型，采用光变曲线（LC）。我们实现了Perceiver-变分自编码器（VAE）架构，使用自监督重建和掩码重建在MMT-9观测站的227,000个LC上进行预训练。VAE能够实现异常检测、运动预测和光变曲线生成。我们使用两个独立的光变曲线模拟器（CASSANDRA和GRIAL）分别对模型进行了微调，使用盒翼、Sentinel-3、SMOS和Starlink平台的CAD模型。预训练模型的重建误差为0.01%，并通过重建难度识别出潜在异常的光变曲线。微调后，模型在异常检测和运动模式预测（太阳定向、自旋等）中的准确率分别为88%和82%，ROC AUC分数分别为0.90和0.95。对真实数据中高置信度异常预测的分析揭示了不同的模式，包括特征物体配置和卫星反射现象。通过自监督学习，我们展示了如何同时实现异常检测、运动预测和从丰富表示学习中生成合成数据的能力。因此，我们的工作通过自动监测和模拟能力支持了太空安全与可持续性。 

---
# Multi-Modality Sensing in mmWave Beamforming for Connected Vehicles Using Deep Learning 

**Title (ZH)**: 基于深学习的连接车辆毫米波波束形成多模态感知 

**Authors**: Muhammad Baqer Mollah, Honggang Wang, Mohammad Ataul Karim, Hua Fang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06173)  

**Abstract**: Beamforming techniques are considered as essential parts to compensate severe path losses in millimeter-wave (mmWave) communications. In particular, these techniques adopt large antenna arrays and formulate narrow beams to obtain satisfactory received powers. However, performing accurate beam alignment over narrow beams for efficient link configuration by traditional standard defined beam selection approaches, which mainly rely on channel state information and beam sweeping through exhaustive searching, imposes computational and communications overheads. And, such resulting overheads limit their potential use in vehicle-to-infrastructure (V2I) and vehicle-to-vehicle (V2V) communications involving highly dynamic scenarios. In comparison, utilizing out-of-band contextual information, such as sensing data obtained from sensor devices, provides a better alternative to reduce overheads. This paper presents a deep learning-based solution for utilizing the multi-modality sensing data for predicting the optimal beams having sufficient mmWave received powers so that the best V2I and V2V line-of-sight links can be ensured proactively. The proposed solution has been tested on real-world measured mmWave sensing and communication data, and the results show that it can achieve up to 98.19% accuracies while predicting top-13 beams. Correspondingly, when compared to existing been sweeping approach, the beam sweeping searching space and time overheads are greatly shortened roughly by 79.67% and 91.89%, respectively which confirm a promising solution for beamforming in mmWave enabled communications. 

**Abstract (ZH)**: 基于深度学习的多模态感知数据应用于预测最优毫米波波束以确保车辆到基础设施和车辆到车辆的直线视距连接 

---
# Real-Time Pitch/F0 Detection Using Spectrogram Images and Convolutional Neural Networks 

**Title (ZH)**: 基于谱图图像和卷积神经网络的实时音调/F0检测 

**Authors**: Xufang Zhao, Omer Tsimhoni  

**Link**: [PDF](https://arxiv.org/pdf/2504.06165)  

**Abstract**: This paper presents a novel approach to detect F0 through Convolutional Neural Networks and image processing techniques to directly estimate pitch from spectrogram images. Our new approach demonstrates a very good detection accuracy; a total of 92% of predicted pitch contours have strong or moderate correlations to the true pitch contours. Furthermore, the experimental comparison between our new approach and other state-of-the-art CNN methods reveals that our approach can enhance the detection rate by approximately 5% across various Signal-to-Noise Ratio conditions. 

**Abstract (ZH)**: 本文提出了一种通过卷积神经网络和图像处理技术从spectrogram图像直接估计音高的新颖方法。我们的新方法展示了非常良好的检测准确性；预测的音调轮廓中有92%与真实音调轮廓具有强烈或中等的相关性。此外，与现有的最佳CNN方法的实验比较表明，我们的方法可以在各种信噪比条件下将检测率提高约5%。 

---
# Navigating the Rabbit Hole: Emergent Biases in LLM-Generated Attack Narratives Targeting Mental Health Groups 

**Title (ZH)**: 穿越兔子洞：大语言模型生成的针对心理健康群体的攻击性叙事中的涌现偏差 

**Authors**: Rijul Magu, Arka Dutta, Sean Kim, Ashiqur R. KhudaBukhsh, Munmun De Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2504.06160)  

**Abstract**: Large Language Models (LLMs) have been shown to demonstrate imbalanced biases against certain groups. However, the study of unprovoked targeted attacks by LLMs towards at-risk populations remains underexplored. Our paper presents three novel contributions: (1) the explicit evaluation of LLM-generated attacks on highly vulnerable mental health groups; (2) a network-based framework to study the propagation of relative biases; and (3) an assessment of the relative degree of stigmatization that emerges from these attacks. Our analysis of a recently released large-scale bias audit dataset reveals that mental health entities occupy central positions within attack narrative networks, as revealed by a significantly higher mean centrality of closeness (p-value = 4.06e-10) and dense clustering (Gini coefficient = 0.7). Drawing from sociological foundations of stigmatization theory, our stigmatization analysis indicates increased labeling components for mental health disorder-related targets relative to initial targets in generation chains. Taken together, these insights shed light on the structural predilections of large language models to heighten harmful discourse and highlight the need for suitable approaches for mitigation. 

**Abstract (ZH)**: 大型语言模型（LLMs）已被证明对某些群体表现出不平衡的偏见。然而，LLMs对脆弱群体进行无缘无故的针对性攻击的研究仍然欠探索。本文提出三大新颖贡献：（1）明确评估LLM生成的针对精神健康高度脆弱群体的攻击；（2）基于网络的框架以研究相对偏见的传播；（3）评估这些攻击中精神健康领域所引起的相对污名化程度。我们分析一个新发布的大型偏见审计数据集表明，精神健康实体在攻击叙事网络中占据中心位置，显示出较高的平均接近性中心度（p值=4.06e-10）和密集聚类（基尼系数=0.7）。基于污名化理论的社会学基础，我们的污名化分析显示，精神健康障碍相关目标在生成链中的标签成分增加幅度大于初始目标。综合这些见解，它们揭示了大型语言模型在加剧有害话语方面的结构性倾向，并突显了需要采取适当方法进行缓解的必要性。 

---
# ARLO: A Tailorable Approach for Transforming Natural Language Software Requirements into Architecture using LLMs 

**Title (ZH)**: ARLO：一种使用大语言模型将自然语言软件需求转换为架构的可配置方法 

**Authors**: Tooraj Helmi  

**Link**: [PDF](https://arxiv.org/pdf/2504.06143)  

**Abstract**: Software requirements expressed in natural language (NL) frequently suffer from verbosity, ambiguity, and inconsistency. This creates a range of challenges, including selecting an appropriate architecture for a system and assessing different architectural alternatives. Relying on human expertise to accomplish the task of mapping NL requirements to architecture is time-consuming and error-prone. This paper proposes ARLO, an approach that automates this task by leveraging (1) a set of NL requirements for a system, (2) an existing standard that specifies architecturally relevant software quality attributes, and (3) a readily available Large Language Model (LLM). Specifically, ARLO determines the subset of NL requirements for a given system that is architecturally relevant and maps that subset to a tailorable matrix of architectural choices. ARLO applies integer linear programming on the architectural-choice matrix to determine the optimal architecture for the current requirements. We demonstrate ARLO's efficacy using a set of real-world examples. We highlight ARLO's ability (1) to trace the selected architectural choices to the requirements and (2) to isolate NL requirements that exert a particular influence on a system's architecture. This allows the identification, comparative assessment, and exploration of alternative architectural choices based on the requirements and constraints expressed therein. 

**Abstract (ZH)**: 自然语言表达的软件需求经常存在冗长、模糊和不一致的问题，这给选择合适的系统架构及评估不同的架构选项带来了挑战。依赖人类专业知识将自然语言需求映射到架构的过程耗时且易出错。本文提出ARLO方法，通过利用（1）系统的自然语言需求集、（2）一个现有的规范，该规范规定了与架构相关的软件质量属性，以及（3）一个现成的大规模语言模型（LLM），自动化这一任务。ARLO确定给定系统中与架构相关的自然语言需求子集，并将其映射到可定制的架构选择矩阵。ARLO通过对架构选择矩阵应用整数线性规划来确定当前需求的最优架构。我们使用一组实际案例展示了ARLO的有效性，并突出了ARLO的能力，即（1）追踪所选架构选择与需求的关系，以及（2）隔离对系统架构具有特定影响的自然语言需求，从而根据其中表达的需求和约束条件进行架构选择的标识、比较评估和探索。 

---
# A Multimedia Analytics Model for the Foundation Model Era 

**Title (ZH)**: 基础模型时代多模态分析模型 

**Authors**: Marcel Worring, Jan Zahálka, Stef van den Elzen, Maximilian Fischer, Daniel Keim  

**Link**: [PDF](https://arxiv.org/pdf/2504.06138)  

**Abstract**: The rapid advances in Foundation Models and agentic Artificial Intelligence are transforming multimedia analytics by enabling richer, more sophisticated interactions between humans and analytical systems. Existing conceptual models for visual and multimedia analytics, however, do not adequately capture the complexity introduced by these powerful AI paradigms. To bridge this gap, we propose a comprehensive multimedia analytics model specifically designed for the foundation model era. Building upon established frameworks from visual analytics, multimedia analytics, knowledge generation, analytic task definition, mixed-initiative guidance, and human-in-the-loop reinforcement learning, our model emphasizes integrated human-AI teaming based on visual analytics agents from both technical and conceptual perspectives. Central to the model is a seamless, yet explicitly separable, interaction channel between expert users and semi-autonomous analytical processes, ensuring continuous alignment between user intent and AI behavior. The model addresses practical challenges in sensitive domains such as intelligence analysis, investigative journalism, and other fields handling complex, high-stakes data. We illustrate through detailed case studies how our model facilitates deeper understanding and targeted improvement of multimedia analytics solutions. By explicitly capturing how expert users can optimally interact with and guide AI-powered multimedia analytics systems, our conceptual framework sets a clear direction for system design, comparison, and future research. 

**Abstract (ZH)**: Foundation Models和自主人工智能的 rapid进展正在通过促进人类与分析系统之间更为丰富和复杂的交互，变革多媒体分析。现有的视觉和多媒体分析概念模型未能充分捕捉这些强大AI范式引入的复杂性。为了弥补这一差距，我们提出了一种专门为Foundation Model时代设计的综合多媒体分析模型。该模型在技术与概念层面均基于视觉分析代理，结合了视觉分析框架、多媒体分析框架、知识生成、分析任务定义、混合主动指导以及人类在环路强化学习等现有框架。模型的核心在于专家用户与半自主分析过程之间无缝但可分离的交互通道，确保用户意图与AI行为之间持续一致。该模型解决了情报分析、调查 journalism等敏感领域以及处理复杂、高 stakes数据的其他领域的实际挑战。通过详细的案例研究，我们展示了该模型如何促进对多媒体分析解决方案的深入理解和具体改进。通过明确捕获专家用户与AI驱动的多媒体分析系统的理想交互和指导方式，我们的概念框架为系统设计、比较和未来研究指明了方向。 

---
# QGen Studio: An Adaptive Question-Answer Generation, Training and Evaluation Platform 

**Title (ZH)**: QGen Studio: 一种自适应问题-答案生成、训练与评估平台 

**Authors**: Movina Moses, Mohab Elkaref, James Barry, Shinnosuke Tanaka, Vishnudev Kuruvanthodi, Nathan Herr, Campbell D Watson, Geeth De Mel  

**Link**: [PDF](https://arxiv.org/pdf/2504.06136)  

**Abstract**: We present QGen Studio: an adaptive question-answer generation, training, and evaluation platform. QGen Studio enables users to leverage large language models (LLMs) to create custom question-answer datasets and fine-tune models on this synthetic data. It features a dataset viewer and model explorer to streamline this process. The dataset viewer provides key metrics and visualizes the context from which the QA pairs are generated, offering insights into data quality. The model explorer supports model comparison, allowing users to contrast the performance of their trained LLMs against other models, supporting performance benchmarking and refinement. QGen Studio delivers an interactive, end-to-end solution for generating QA datasets and training scalable, domain-adaptable models. The studio will be open-sourced soon, allowing users to deploy it locally. 

**Abstract (ZH)**: QGen Studio: 一种自适应的问答生成、训练和评估平台 

---
# Uncertainty-Aware Hybrid Machine Learning in Virtual Sensors for Vehicle Sideslip Angle Estimation 

**Title (ZH)**: 面向车辆侧滑角估计的不确定性感知混合机器学习在虚拟传感器中的应用 

**Authors**: Abinav Kalyanasundaram, Karthikeyan Chandra Sekaran, Philipp Stauber, Michael Lange, Wolfgang Utschick, Michael Botsch  

**Link**: [PDF](https://arxiv.org/pdf/2504.06105)  

**Abstract**: Precise vehicle state estimation is crucial for safe and reliable autonomous driving. The number of measurable states and their precision offered by the onboard vehicle sensor system are often constrained by cost. For instance, measuring critical quantities such as the Vehicle Sideslip Angle (VSA) poses significant commercial challenges using current optical sensors. This paper addresses these limitations by focusing on the development of high-performance virtual sensors to enhance vehicle state estimation for active safety. The proposed Uncertainty-Aware Hybrid Learning (UAHL) architecture integrates a machine learning model with vehicle motion models to estimate VSA directly from onboard sensor data. A key aspect of the UAHL architecture is its focus on uncertainty quantification for individual model estimates and hybrid fusion. These mechanisms enable the dynamic weighting of uncertainty-aware predictions from machine learning and vehicle motion models to produce accurate and reliable hybrid VSA estimates. This work also presents a novel dataset named Real-world Vehicle State Estimation Dataset (ReV-StED), comprising synchronized measurements from advanced vehicle dynamic sensors. The experimental results demonstrate the superior performance of the proposed method for VSA estimation, highlighting UAHL as a promising architecture for advancing virtual sensors and enhancing active safety in autonomous vehicles. 

**Abstract (ZH)**: 精确的车辆状态估计对于安全可靠的自动驾驶至关重要。车载车辆传感器系统可测量的状态数量及其精度往往受限于成本。例如，使用当前的光学传感器测量关键参数（如汽车侧滑角）存在显著的商业挑战。本文通过专注于高性能虚拟传感器的开发来克服这些限制，以提高车辆状态估计算法中的主动安全性。所提出的一种结合机器学习模型和车辆运动模型的不确定性意识混合学习（UAHL）架构能够直接从车载传感器数据中估计汽车侧滑角。UAHL架构的关键方面在于其对单个模型估计值的不确定性量化和混合融合机制。这些机制使不确定性意识预测能够动态加权结合，从而产生准确可靠的混合侧滑角估计。此外，本文还提出了一种名为Real-world Vehicle State Estimation Dataset (ReV-StED)的新数据集，包含高级车辆动态传感器的同步测量数据。实验结果表明，所提出的方法在侧滑角估计方面表现出优越的性能，突出了UAHL架构在推进虚拟传感器发展和提高自动驾驶车辆主动安全性方面的前景。 

---
# Towards Varroa destructor mite detection using a narrow spectra illumination 

**Title (ZH)**: 使用窄谱illumination用于检测Varroa destructor寄生虫的研究 

**Authors**: Samuel Bielik, Simon Bilik  

**Link**: [PDF](https://arxiv.org/pdf/2504.06099)  

**Abstract**: This paper focuses on the development and modification of a beehive monitoring device and Varroa destructor detection on the bees with the help of hyperspectral imagery while utilizing a U-net, semantic segmentation architecture, and conventional computer vision methods. The main objectives were to collect a dataset of bees and mites, and propose the computer vision model which can achieve the detection between bees and mites. 

**Abstract (ZH)**: 本文专注于利用高光谱影像技术开发和修改养蜂监测设备，并通过U-net语义分割架构和传统计算机视觉方法检测蜜蜂上的Varroa destructor。主要目标是收集蜜蜂和螨虫的 datasets，并提出一种计算机视觉模型以实现蜜蜂与螨虫之间的检测。 

---
# Real-Time LaCAM 

**Title (ZH)**: 实时LaCAM 

**Authors**: Runzhe Liang, Rishi Veerapaneni, Daniel Harabor, Jiaoyang Li, Maxim Likhachev  

**Link**: [PDF](https://arxiv.org/pdf/2504.06091)  

**Abstract**: The vast majority of Multi-Agent Path Finding (MAPF) methods with completeness guarantees require planning full horizon paths. However, planning full horizon paths can take too long and be impractical in real-world applications. Instead, real-time planning and execution, which only allows the planner a finite amount of time before executing and replanning, is more practical for real world multi-agent systems. Several methods utilize real-time planning schemes but none are provably complete, which leads to livelock or deadlock. Our main contribution is to show the first Real-Time MAPF method with provable completeness guarantees. We do this by leveraging LaCAM (Okumura 2023) in an incremental fashion. Our results show how we can iteratively plan for congested environments with a cutoff time of milliseconds while still maintaining the same success rate as full horizon LaCAM. We also show how it can be used with a single-step learned MAPF policy. The proposed Real-Time LaCAM also provides us with a general mechanism for using iterative constraints for completeness in future real-time MAPF algorithms. 

**Abstract (ZH)**: 实时 Multi-Agent Path Finding 方法的完备性保证研究 

---
# MCAT: Visual Query-Based Localization of Standard Anatomical Clips in Fetal Ultrasound Videos Using Multi-Tier Class-Aware Token Transformer 

**Title (ZH)**: MCAT: 基于视觉查询的胎儿超声视频中标准解剖剪辑的多层类意识标记转换器定位 

**Authors**: Divyanshu Mishra, Pramit Saha, He Zhao, Netzahualcoyotl Hernandez-Cruz, Olga Patey, Aris Papageorghiou, J. Alison Noble  

**Link**: [PDF](https://arxiv.org/pdf/2504.06088)  

**Abstract**: Accurate standard plane acquisition in fetal ultrasound (US) videos is crucial for fetal growth assessment, anomaly detection, and adherence to clinical guidelines. However, manually selecting standard frames is time-consuming and prone to intra- and inter-sonographer variability. Existing methods primarily rely on image-based approaches that capture standard frames and then classify the input frames across different anatomies. This ignores the dynamic nature of video acquisition and its interpretation. To address these challenges, we introduce Multi-Tier Class-Aware Token Transformer (MCAT), a visual query-based video clip localization (VQ-VCL) method, to assist sonographers by enabling them to capture a quick US sweep. By then providing a visual query of the anatomy they wish to analyze, MCAT returns the video clip containing the standard frames for that anatomy, facilitating thorough screening for potential anomalies. We evaluate MCAT on two ultrasound video datasets and a natural image VQ-VCL dataset based on Ego4D. Our model outperforms state-of-the-art methods by 10% and 13% mIoU on the ultrasound datasets and by 5.35% mIoU on the Ego4D dataset, using 96% fewer tokens. MCAT's efficiency and accuracy have significant potential implications for public health, especially in low- and middle-income countries (LMICs), where it may enhance prenatal care by streamlining standard plane acquisition, simplifying US-based screening, diagnosis and allowing sonographers to examine more patients. 

**Abstract (ZH)**: 基于视觉查询的多层级类意识 Tokens 转换器在胎儿超声视频中的标准平面 Acquisition 

---
# Confidence Regularized Masked Language Modeling using Text Length 

**Title (ZH)**: 长度正则化掩蔽语言模型中的信心调节 

**Authors**: Seunghyun Ji, Soowon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.06037)  

**Abstract**: Masked language modeling, which is a task to predict a randomly masked word in the input text, is an efficient language representation learning method. Masked language modeling ignores various words which people can think of for filling in the masked position and calculates the loss with a single word. Especially when the input text is short, the entropy of the word distribution that can fill in the masked position can be high. This may cause the model to be overconfident in the single answer. To address this issue, we propose a novel confidence regularizer that controls regularizing strength dynamically by the input text length. Experiments with GLUE and SQuAD datasets showed that our method achieves better accuracy and lower expected calibration error. 

**Abstract (ZH)**: 掩码语言模型是一种用于预测输入文本中随机掩蔽词的高效语言表示学习方法。掩码语言模型忽略了人们可以用来填充掩蔽位置的各种词语，并使用单个词语计算损失。尤其是在输入文本较短时，可以填充掩蔽位置的词分布的熵可能会很高，这可能导致模型对单一答案过于自信。为解决这一问题，我们提出了一种新的置信度正则化方法，通过动态调整正则化强度来控制模型的置信度。实验结果表明，我们的方法在GLUE和SQuAD数据集上取得了更高的准确率和更低的预期校准误差。 

---
# The Hall of AI Fears and Hopes: Comparing the Views of AI Influencers and those of Members of the U.S. Public Through an Interactive Platform 

**Title (ZH)**: AI恐惧与希望之堂：通过交互平台对比AI影响者与美国公众的观点 

**Authors**: Gustavo Moreira, Edyta Paulina Bogucka, Marios Constantinides, Daniele Quercia  

**Link**: [PDF](https://arxiv.org/pdf/2504.06016)  

**Abstract**: AI development is shaped by academics and industry leaders - let us call them ``influencers'' - but it is unclear how their views align with those of the public. To address this gap, we developed an interactive platform that served as a data collection tool for exploring public views on AI, including their fears, hopes, and overall sense of hopefulness. We made the platform available to 330 participants representative of the U.S. population in terms of age, sex, ethnicity, and political leaning, and compared their views with those of 100 AI influencers identified by Time magazine. The public fears AI getting out of control, while influencers emphasize regulation, seemingly to deflect attention from their alleged focus on monetizing AI's potential. Interestingly, the views of AI influencers from underrepresented groups such as women and people of color often differ from the views of underrepresented groups in the public. 

**Abstract (ZH)**: AI的发展受到学术界和工业界领导者的塑造——让我们称之为“影响者”——但他们的观点与公众的观点尚不清楚。为了填补这一空白，我们开发了一个交互平台，作为收集公众对AI看法的数据工具，包括他们的恐惧、希望以及总体的乐观感。我们将这一平台提供给330名在年龄、性别、种族和政治倾向上具有代表性的美国人口参与者，并将他们的观点与《时代》杂志认定的100名AI影响者的观点进行了比较。公众担心AI失控，而影响者则强调监管，似乎是为了转移公众对他们可能利用AI潜在价值的关注。有趣的是，代表性不足群体（如女性和有色人种）的影响者与公众中代表性不足群体的观点往往存在差异。 

---
# Optuna vs Code Llama: Are LLMs a New Paradigm for Hyperparameter Tuning? 

**Title (ZH)**: Optuna vs Code Llama：大规模语言模型是否成为超参数调优的新范式？ 

**Authors**: Roman Kochnev, Arash Torabi Goodarzi, Zofia Antonina Bentyn, Dmitry Ignatov, Radu Timofte  

**Link**: [PDF](https://arxiv.org/pdf/2504.06006)  

**Abstract**: Optimal hyperparameter selection is critical for maximizing neural network performance, especially as models grow in complexity. This work investigates the viability of using large language models (LLMs) for hyperparameter optimization by employing a fine-tuned version of Code Llama. Through parameter-efficient fine-tuning using LoRA, we adapt the LLM to generate accurate and efficient hyperparameter recommendations tailored to diverse neural network architectures. Unlike traditional methods such as Optuna, which rely on exhaustive trials, the proposed approach achieves competitive or superior results in terms of Root Mean Square Error (RMSE) while significantly reducing computational overhead. Our approach highlights that LLM-based optimization not only matches state-of-the-art methods like Tree-structured Parzen Estimators but also accelerates the tuning process. This positions LLMs as a promising alternative to conventional optimization techniques, particularly for rapid experimentation. Furthermore, the ability to generate hyperparameters in a single inference step makes this method particularly well-suited for resource-constrained environments such as edge devices and mobile applications, where computational efficiency is paramount. The results confirm that LLMs, beyond their efficiency, offer substantial time savings and comparable stability, underscoring their value in advancing machine learning workflows. All generated hyperparameters are included in the LEMUR Neural Network (NN) Dataset, which is publicly available and serves as an open-source benchmark for hyperparameter optimization research. 

**Abstract (ZH)**: 使用大型语言模型进行超参数优化：Code Llama的参数高效微调技术 

---
# NativQA Framework: Enabling LLMs with Native, Local, and Everyday Knowledge 

**Title (ZH)**: NativQA框架：使大语言模型蕴含本地化、日常知识能力 

**Authors**: Firoj Alam, Md Arid Hasan, Sahinur Rahman Laskar, Mucahid Kutlu, Shammur Absar Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2504.05995)  

**Abstract**: The rapid advancement of large language models (LLMs) has raised concerns about cultural bias, fairness, and their applicability in diverse linguistic and underrepresented regional contexts. To enhance and benchmark the capabilities of LLMs, there is a need to develop large-scale resources focused on multilingual, local, and cultural contexts. In this study, we propose a framework, NativQA, that can seamlessly construct large-scale, culturally and regionally aligned QA datasets in native languages. The framework utilizes user-defined seed queries and leverages search engines to collect location-specific, everyday information. It has been evaluated across 39 locations in 24 countries and in 7 languages, ranging from extremely low-resource to high-resource languages, which resulted over 300K Question Answer (QA) pairs. The developed resources can be used for LLM benchmarking and further fine-tuning. The framework has been made publicly available for the community (this https URL). 

**Abstract (ZH)**: 大规模语言模型的迅速进步引发了对其文化偏见、公平性及其在多元语言和欠代表地区应用场景的顾虑。为提升并评估语言模型的能力，有必要开发专注于多语言、地方性和文化背景的大规模资源。本研究提出了一种名为NativQA的框架，该框架能够无缝构建大规模、文化上和地区上对齐的多语言问答数据集。该框架利用用户定义的种子查询，并利用搜索引擎收集地点特定的日常信息。该框架在24个国家的39个地点进行了评估，涵盖7种从极度低资源到高资源的语言，生成了超过30万条问答对。开发的资源可用于语言模型的评估和进一步微调。该框架已向社区公开（this https URL）。 

---
# Temporal Alignment-Free Video Matching for Few-shot Action Recognition 

**Title (ZH)**: 面向少量样本动作识别的时序对齐自由视频匹配 

**Authors**: SuBeen Lee, WonJun Moon, Hyun Seok Seong, Jae-Pil Heo  

**Link**: [PDF](https://arxiv.org/pdf/2504.05956)  

**Abstract**: Few-Shot Action Recognition (FSAR) aims to train a model with only a few labeled video instances. A key challenge in FSAR is handling divergent narrative trajectories for precise video matching. While the frame- and tuple-level alignment approaches have been promising, their methods heavily rely on pre-defined and length-dependent alignment units (e.g., frames or tuples), which limits flexibility for actions of varying lengths and speeds. In this work, we introduce a novel TEmporal Alignment-free Matching (TEAM) approach, which eliminates the need for temporal units in action representation and brute-force alignment during matching. Specifically, TEAM represents each video with a fixed set of pattern tokens that capture globally discriminative clues within the video instance regardless of action length or speed, ensuring its flexibility. Furthermore, TEAM is inherently efficient, using token-wise comparisons to measure similarity between videos, unlike existing methods that rely on pairwise comparisons for temporal alignment. Additionally, we propose an adaptation process that identifies and removes common information across classes, establishing clear boundaries even between novel categories. Extensive experiments demonstrate the effectiveness of TEAM. Codes are available at this http URL. 

**Abstract (ZH)**: 基于少样本动作识别的时空单元自由匹配（TEMPOREAL ALIGNMENT-FREE MATCHING FOR FEW-SHOT ACTION RECOGNITION） 

---
# CKGAN: Training Generative Adversarial Networks Using Characteristic Kernel Integral Probability Metrics 

**Title (ZH)**: CKGAN：使用特征核积分概率度量训练生成对抗网络 

**Authors**: Kuntian Zhang, Simin Yu, Yaoshu Wang, Makoto Onizuka, Chuan Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2504.05945)  

**Abstract**: In this paper, we propose CKGAN, a novel generative adversarial network (GAN) variant based on an integral probability metrics framework with characteristic kernel (CKIPM). CKIPM, as a distance between two probability distributions, is designed to optimize the lowerbound of the maximum mean discrepancy (MMD) in a reproducing kernel Hilbert space, and thus can be used to train GANs. CKGAN mitigates the notorious problem of mode collapse by mapping the generated images back to random noise. To save the effort of selecting the kernel function manually, we propose a soft selection method to automatically learn a characteristic kernel function. The experimental evaluation conducted on a set of synthetic and real image benchmarks (MNIST, CelebA, etc.) demonstrates that CKGAN generally outperforms other MMD-based GANs. The results also show that at the cost of moderately more training time, the automatically selected kernel function delivers very close performance to the best of manually fine-tuned one on real image benchmarks and is able to improve the performances of other MMD-based GANs. 

**Abstract (ZH)**: 基于特征内积概率度量的CKGAN生成式对抗网络 

---
# Uncovering Fairness through Data Complexity as an Early Indicator 

**Title (ZH)**: 通过数据复杂性作为早期指标发现公平性 

**Authors**: Juliett Suárez Ferreira, Marija Slavkovik, Jorge Casillas  

**Link**: [PDF](https://arxiv.org/pdf/2504.05923)  

**Abstract**: Fairness constitutes a concern within machine learning (ML) applications. Currently, there is no study on how disparities in classification complexity between privileged and unprivileged groups could influence the fairness of solutions, which serves as a preliminary indicator of potential unfairness. In this work, we investigate this gap, specifically, we focus on synthetic datasets designed to capture a variety of biases ranging from historical bias to measurement and representational bias to evaluate how various complexity metrics differences correlate with group fairness metrics. We then apply association rule mining to identify patterns that link disproportionate complexity differences between groups with fairness-related outcomes, offering data-centric indicators to guide bias mitigation. Our findings are also validated by their application in real-world problems, providing evidence that quantifying group-wise classification complexity can uncover early indicators of potential fairness challenges. This investigation helps practitioners to proactively address bias in classification tasks. 

**Abstract (ZH)**: 公平性是机器学习（ML）应用中的一个关注点。目前，尚未有研究探讨分类复杂度在特权和非特权群体之间的差异如何影响解决方案的公平性，这被视为潜在不公平性的初步指标。在这项工作中，我们探讨了这一差距，具体而言，我们关注设计用于捕捉历史偏差、测量偏差和表征偏差等多种偏差的合成数据集，以评估各种复杂度度量差异与群体公平性度量之间的关联。然后，我们应用关联规则挖掘来识别将群体之间不均衡的复杂度差异与公平性相关结果联系起来的模式，提供以数据为中心的指标来指导偏差缓解。我们的发现也通过应用于实际问题得到了验证，提供了证据表明量化群体间的分类复杂性可以揭示潜在公平性挑战的早期指标。这项调查有助于实践者在分类任务中前瞻性地应对偏差。 

---
# PRIMEDrive-CoT: A Precognitive Chain-of-Thought Framework for Uncertainty-Aware Object Interaction in Driving Scene Scenario 

**Title (ZH)**: PRIMEDrive-CoT: 一种用于驾驶场景中不确定性aware对象交互的先知性链思考框架 

**Authors**: Sriram Mandalika, Lalitha V, Athira Nambiar  

**Link**: [PDF](https://arxiv.org/pdf/2504.05908)  

**Abstract**: Driving scene understanding is a critical real-world problem that involves interpreting and associating various elements of a driving environment, such as vehicles, pedestrians, and traffic signals. Despite advancements in autonomous driving, traditional pipelines rely on deterministic models that fail to capture the probabilistic nature and inherent uncertainty of real-world driving. To address this, we propose PRIMEDrive-CoT, a novel uncertainty-aware model for object interaction and Chain-of-Thought (CoT) reasoning in driving scenarios. In particular, our approach combines LiDAR-based 3D object detection with multi-view RGB references to ensure interpretable and reliable scene understanding. Uncertainty and risk assessment, along with object interactions, are modelled using Bayesian Graph Neural Networks (BGNNs) for probabilistic reasoning under ambiguous conditions. Interpretable decisions are facilitated through CoT reasoning, leveraging object dynamics and contextual cues, while Grad-CAM visualizations highlight attention regions. Extensive evaluations on the DriveCoT dataset demonstrate that PRIMEDrive-CoT outperforms state-of-the-art CoT and risk-aware models. 

**Abstract (ZH)**: PRIMEDrive-CoT：一种用于驾驶场景的认知推理和不确定性感知模型 

---
# Turin3D: Evaluating Adaptation Strategies under Label Scarcity in Urban LiDAR Segmentation with Semi-Supervised Techniques 

**Title (ZH)**: Turin3D：在半监督技术下的稀疏标签条件下城市LiDAR分割适应策略评估 

**Authors**: Luca Barco, Giacomo Blanco, Gaetano Chiriaco, Alessia Intini, Luigi La Riccia, Vittorio Scolamiero, Piero Boccardo, Paolo Garza, Fabrizio Dominici  

**Link**: [PDF](https://arxiv.org/pdf/2504.05882)  

**Abstract**: 3D semantic segmentation plays a critical role in urban modelling, enabling detailed understanding and mapping of city environments. In this paper, we introduce Turin3D: a new aerial LiDAR dataset for point cloud semantic segmentation covering an area of around 1.43 km2 in the city centre of Turin with almost 70M points. We describe the data collection process and compare Turin3D with others previously proposed in the literature. We did not fully annotate the dataset due to the complexity and time-consuming nature of the process; however, a manual annotation process was performed on the validation and test sets, to enable a reliable evaluation of the proposed techniques. We first benchmark the performances of several point cloud semantic segmentation models, trained on the existing datasets, when tested on Turin3D, and then improve their performances by applying a semi-supervised learning technique leveraging the unlabelled training set. The dataset will be publicly available to support research in outdoor point cloud segmentation, with particular relevance for self-supervised and semi-supervised learning approaches given the absence of ground truth annotations for the training set. 

**Abstract (ZH)**: 3D语义分割在城市建模中发挥着关键作用， enables 对城市环境的详细理解与建模。在本文中，我们介绍 Turin3D：一个新的覆盖意大利都灵市中心约1.43 km²区域的机载LiDAR数据集，包含几乎7000万个点的点云语义分割数据集。我们描述了数据收集过程，并将Turin3D与文献中之前提出的数据集进行比较。由于注释过程的复杂性和耗时性，我们未对整个数据集进行注释；然而，我们在验证集和测试集上进行了手动注释，以确保对所提方法进行可靠评估。我们首先对标记在现有数据集上训练的几种点云语义分割模型在Turin3D上的性能进行基准测试，然后通过利用未标记训练集的半监督学习技术来提高其性能。该数据集将公开发布以支持室外点云分割研究，特别是对于自监督和半监督学习方法具有重要意义，因为训练集缺乏地面真实标注。 

---
# Towards an AI-Driven Video-Based American Sign Language Dictionary: Exploring Design and Usage Experience with Learners 

**Title (ZH)**: 面向AI驱动的基于视频的手语词典：探索使用者学习体验的设计与使用 

**Authors**: Saad Hassan, Matyas Bohacek, Chaelin Kim, Denise Crochet  

**Link**: [PDF](https://arxiv.org/pdf/2504.05857)  

**Abstract**: Searching for unfamiliar American Sign Language (ASL) signs is challenging for learners because, unlike spoken languages, they cannot type a text-based query to look up an unfamiliar sign. Advances in isolated sign recognition have enabled the creation of video-based dictionaries, allowing users to submit a video and receive a list of the closest matching signs. Previous HCI research using Wizard-of-Oz prototypes has explored interface designs for ASL dictionaries. Building on these studies, we incorporate their design recommendations and leverage state-of-the-art sign-recognition technology to develop an automated video-based dictionary. We also present findings from an observational study with twelve novice ASL learners who used this dictionary during video-comprehension and question-answering tasks. Our results address human-AI interaction challenges not covered in previous WoZ research, including recording and resubmitting signs, unpredictable outputs, system latency, and privacy concerns. These insights offer guidance for designing and deploying video-based ASL dictionary systems. 

**Abstract (ZH)**: 探索不熟悉的美国手语（ASL）手势对学习者来说具有挑战性，因为他们无法像使用书面查询那样通过键盘查询不熟悉的手势。手语孤立手势识别的进步使得基于视频的手语词典得以创建，用户可以提交视频并收到最接近的手势列表。基于此，我们融合以往人机交互（HCI）研究中的界面设计建议，并利用最先进的手语识别技术开发了一个自动化的基于视频的手语词典。我们还介绍了十二位初级ASL学习者在视频理解和问答任务中使用该词典的观察研究结果。我们的研究结果解决了之前Wizard-of-Oz（巫师学徒）原型研究未涵盖的人机交互挑战，包括手势的录制与重新提交、不可预测的输出、系统延迟和隐私问题。这些见解为设计和部署基于视频的手语词典系统提供了指导。 

---
# Enhancing Coreference Resolution with Pretrained Language Models: Bridging the Gap Between Syntax and Semantics 

**Title (ZH)**: 使用预训练语言模型增强同指替解析：缩小语法与语义之间的差距 

**Authors**: Xingzu Liu, Songhang deng, Mingbang Wang, Zhang Dong, Le Dai, Jiyuan Li, Ruilin Nong  

**Link**: [PDF](https://arxiv.org/pdf/2504.05855)  

**Abstract**: Large language models have made significant advancements in various natural language processing tasks, including coreference resolution. However, traditional methods often fall short in effectively distinguishing referential relationships due to a lack of integration between syntactic and semantic information. This study introduces an innovative framework aimed at enhancing coreference resolution by utilizing pretrained language models. Our approach combines syntax parsing with semantic role labeling to accurately capture finer distinctions in referential relationships. By employing state-of-the-art pretrained models to gather contextual embeddings and applying an attention mechanism for fine-tuning, we improve the performance of coreference tasks. Experimental results across diverse datasets show that our method surpasses conventional coreference resolution systems, achieving notable accuracy in disambiguating references. This development not only improves coreference resolution outcomes but also positively impacts other natural language processing tasks that depend on precise referential understanding. 

**Abstract (ZH)**: 大型语言模型在各种自然语言处理任务中取得了显著进展，包括核心ference解析。然而，传统方法往往因缺乏语法和语义信息的整合而在有效区分指称关系方面力有未逮。本研究提出了一种创新框架，旨在通过利用预训练语言模型来提升核心ference解析。我们的方法结合了句法解析和语义角色标注，以准确捕获指称关系中的细微区别。通过使用最先进的预训练模型来收集上下文嵌入，并应用注意力机制进行微调，我们提高了核心ference任务的性能。跨多个数据集的实验结果表明，我们的方法超越了传统的核心ference解析系统，实现了显著的参考消歧准确性。这一发展不仅提升了核心ference解析的效果，还对依赖精确指称理解的其他自然语言处理任务产生了积极影响。 

---
# Physics-aware generative models for turbulent fluid flows through energy-consistent stochastic interpolants 

**Title (ZH)**: 物理感知的涡流流体生成模型通过能量一致的随机插值 

**Authors**: Nikolaj T. Mücke, Benjamin Sanderse  

**Link**: [PDF](https://arxiv.org/pdf/2504.05852)  

**Abstract**: Generative models have demonstrated remarkable success in domains such as text, image, and video synthesis. In this work, we explore the application of generative models to fluid dynamics, specifically for turbulence simulation, where classical numerical solvers are computationally expensive. We propose a novel stochastic generative model based on stochastic interpolants, which enables probabilistic forecasting while incorporating physical constraints such as energy stability and divergence-freeness. Unlike conventional stochastic generative models, which are often agnostic to underlying physical laws, our approach embeds energy consistency by making the parameters of the stochastic interpolant learnable coefficients. We evaluate our method on a benchmark turbulence problem - Kolmogorov flow - demonstrating superior accuracy and stability over state-of-the-art alternatives such as autoregressive conditional diffusion models (ACDMs) and PDE-Refiner. Furthermore, we achieve stable results for significantly longer roll-outs than standard stochastic interpolants. Our results highlight the potential of physics-aware generative models in accelerating and enhancing turbulence simulations while preserving fundamental conservation properties. 

**Abstract (ZH)**: 生成模型在文本、图像和视频合成等领域的应用已经显示出显著的成功。本文探索将生成模型应用于流体力学，特别是涡流模拟，其中经典数值求解器计算成本高。我们提出了一种基于随机插值的新型随机生成模型，该模型能够进行概率预测并纳入如能量稳定性与散度为零等物理约束。与通常对底层物理定律视而不见的常规随机生成模型不同，我们的方法通过使随机插值的参数成为可学习的系数，嵌入能量一致性。我们通过Kolmogorov流这一基准涡流问题评估了该方法，结果显示其在与自回归条件扩散模型（ACDMs）及PDE-Refiner等最先进的替代方法相比时具有更高的准确性和稳定性。此外，我们在远超标准随机插值的较长运行时间下获得了稳定的结果。我们的研究结果突显了物理感知生成模型在保持基本守恒性质的同时加速和增强涡流模拟的潜在价值。 

---
# PathGPT: Leveraging Large Language Models for Personalized Route Generation 

**Title (ZH)**: PathGPT：利用大型语言模型进行个性化路径生成 

**Authors**: Steeve Cuthbert Marcelyn, Yucen Gao, Yuzhe Zhang, Xiaofeng Gao, Guihai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.05846)  

**Abstract**: The proliferation of GPS enabled devices has led to the accumulation of a substantial corpus of historical trajectory data. By leveraging these data for training machine learning models,researchers have devised novel data-driven methodologies that address the personalized route recommendation (PRR) problem. In contrast to conventional algorithms such as Dijkstra shortest path algorithm,these novel algorithms possess the capacity to discern and learn patterns within the data,thereby facilitating the generation of more personalized paths. However,once these models have been trained,their application is constrained to the generation of routes that align with their training patterns. This limitation renders them less adaptable to novel scenarios and the deployment of multiple machine learning models might be necessary to address new possible scenarios,which can be costly as each model must be trained separately. Inspired by recent advances in the field of Large Language Models (LLMs),we leveraged their natural language understanding capabilities to develop a unified model to solve the PRR problem while being seamlessly adaptable to new scenarios without additional training. To accomplish this,we combined the extensive knowledge LLMs acquired during training with further access to external hand-crafted context information,similar to RAG (Retrieved Augmented Generation) systems,to enhance their ability to generate paths according to user-defined requirements. Extensive experiments on different datasets show a considerable uplift in LLM performance on the PRR problem. 

**Abstract (ZH)**: GPS-enable装置的普及导致了大量历史轨迹数据的积累。通过利用这些数据训练机器学习模型，研究人员设计出了新的数据驱动方法来解决个性化路线推荐问题。与传统的算法（如迪杰斯特拉最短路径算法）相比，这些新型算法能够识别和学习数据中的模式，从而生成更个性化的路径。然而，一旦这些模型被训练好，它们的应用就仅限于生成与其训练模式相符的路线。这一限制使得它们难以适应新的场景，而部署多个机器学习模型以应对新场景可能会非常昂贵，因为每个模型都需要单独训练。受大型语言模型（LLMs）领域近期进展的启发，我们利用了其自然语言理解能力，开发了一个统一的模型来解决个性化路线推荐问题，并使其能够无缝适应新场景而无需额外训练。为此，我们结合了LLMs在训练过程中获得的丰富知识，并进一步提供了手动生成的外部上下文信息，类似于RAG（检索增强生成）系统，以增强其根据用户自定义需求生成路径的能力。在不同数据集上的 extensive 实验显示了 LLM 在个性化路线推荐问题上的显著性能提升。 

---
# Momentum Boosted Episodic Memory for Improving Learning in Long-Tailed RL Environments 

**Title (ZH)**: 动量增强的 episodic 记忆以改善长尾 RL 环境中的学习 

**Authors**: Dolton Fernandes, Pramod Kaushik, Harsh Shukla, Bapi Raju Surampudi  

**Link**: [PDF](https://arxiv.org/pdf/2504.05840)  

**Abstract**: Traditional Reinforcement Learning (RL) algorithms assume the distribution of the data to be uniform or mostly uniform. However, this is not the case with most real-world applications like autonomous driving or in nature where animals roam. Some experiences are encountered frequently, and most of the remaining experiences occur rarely; the resulting distribution is called Zipfian. Taking inspiration from the theory of complementary learning systems, an architecture for learning from Zipfian distributions is proposed where important long tail trajectories are discovered in an unsupervised manner. The proposal comprises an episodic memory buffer containing a prioritised memory module to ensure important rare trajectories are kept longer to address the Zipfian problem, which needs credit assignment to happen in a sample efficient manner. The experiences are then reinstated from episodic memory and given weighted importance forming the trajectory to be executed. Notably, the proposed architecture is modular, can be incorporated in any RL architecture and yields improved performance in multiple Zipfian tasks over traditional architectures. Our method outperforms IMPALA by a significant margin on all three tasks and all three evaluation metrics (Zipfian, Uniform, and Rare Accuracy) and also gives improvements on most Atari environments that are considered challenging 

**Abstract (ZH)**: 源自Zipfian分发的无监督学习架构：一种处理稀有重要轨迹的强化学习方法 

---
# Mind the Trojan Horse: Image Prompt Adapter Enabling Scalable and Deceptive Jailbreaking 

**Title (ZH)**: 警惕木马 Horse：图像提示适配器实现可扩展且欺骗性的 Jailbreaking 

**Authors**: Junxi Chen, Junhao Dong, Xiaohua Xie  

**Link**: [PDF](https://arxiv.org/pdf/2504.05838)  

**Abstract**: Recently, the Image Prompt Adapter (IP-Adapter) has been increasingly integrated into text-to-image diffusion models (T2I-DMs) to improve controllability. However, in this paper, we reveal that T2I-DMs equipped with the IP-Adapter (T2I-IP-DMs) enable a new jailbreak attack named the hijacking attack. We demonstrate that, by uploading imperceptible image-space adversarial examples (AEs), the adversary can hijack massive benign users to jailbreak an Image Generation Service (IGS) driven by T2I-IP-DMs and mislead the public to discredit the service provider. Worse still, the IP-Adapter's dependency on open-source image encoders reduces the knowledge required to craft AEs. Extensive experiments verify the technical feasibility of the hijacking attack. In light of the revealed threat, we investigate several existing defenses and explore combining the IP-Adapter with adversarially trained models to overcome existing defenses' limitations. Our code is available at this https URL. 

**Abstract (ZH)**: 最近，Image Prompt Adapter (IP-Adapter) 越来越多地被集成到文本到图像扩散模型（T2I-DMs）中以提高可控性。然而，在本文中，我们揭示出配备 IP-Adapter 的 T2I-DMs（T2I-IP-DMs）能够启用一种新的 Jailbreak 攻击——劫持攻击。我们证明通过上传不可感知的图像空间 adversarial examples (AEs)，攻击者可以劫持大量无辜用户，使他们 Jailbreak 由 T2I-IP-DMs 驱动的图像生成服务（IGS），误导公众对服务提供商产生不良看法。更糟的是，IP-Adapter 对开源图像编码器的依赖降低了构造 AEs 所需的知识。广泛实验验证了劫持攻击的技术可行性。鉴于揭示出的威胁，我们调查了几种现有防御措施，并探讨将 IP-Adapter 与对抗训练模型结合以克服现有防御措施的局限性。我们的代码可在以下网址获取。 

---
# Human Activity Recognition using RGB-Event based Sensors: A Multi-modal Heat Conduction Model and A Benchmark Dataset 

**Title (ZH)**: 基于RGB-事件传感器的人类活动识别：一个多模态热传导模型及基准数据集 

**Authors**: Shiao Wang, Xiao Wang, Bo Jiang, Lin Zhu, Guoqi Li, Yaowei Wang, Yonghong Tian, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05830)  

**Abstract**: Human Activity Recognition (HAR) primarily relied on traditional RGB cameras to achieve high-performance activity recognition. However, the challenging factors in real-world scenarios, such as insufficient lighting and rapid movements, inevitably degrade the performance of RGB cameras. To address these challenges, biologically inspired event cameras offer a promising solution to overcome the limitations of traditional RGB cameras. In this work, we rethink human activity recognition by combining the RGB and event cameras. The first contribution is the proposed large-scale multi-modal RGB-Event human activity recognition benchmark dataset, termed HARDVS 2.0, which bridges the dataset gaps. It contains 300 categories of everyday real-world actions with a total of 107,646 paired videos covering various challenging scenarios. Inspired by the physics-informed heat conduction model, we propose a novel multi-modal heat conduction operation framework for effective activity recognition, termed MMHCO-HAR. More in detail, given the RGB frames and event streams, we first extract the feature embeddings using a stem network. Then, multi-modal Heat Conduction blocks are designed to fuse the dual features, the key module of which is the multi-modal Heat Conduction Operation layer. We integrate RGB and event embeddings through a multi-modal DCT-IDCT layer while adaptively incorporating the thermal conductivity coefficient via FVEs into this module. After that, we propose an adaptive fusion module based on a policy routing strategy for high-performance classification. Comprehensive experiments demonstrate that our method consistently performs well, validating its effectiveness and robustness. The source code and benchmark dataset will be released on this https URL 

**Abstract (ZH)**: 基于RGB和事件相机的人类活动识别（HAR）：HARDVS 2.0 多模态基准数据集及MMHCO-HAR框架 

---
# Parasite: A Steganography-based Backdoor Attack Framework for Diffusion Models 

**Title (ZH)**: 寄生虫：一种基于隐写术的扩散模型后门攻击框架 

**Authors**: Jiahao Chen, Yu Pan, Yi Du, Chunkai Wu, Lin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05815)  

**Abstract**: Recently, the diffusion model has gained significant attention as one of the most successful image generation models, which can generate high-quality images by iteratively sampling noise. However, recent studies have shown that diffusion models are vulnerable to backdoor attacks, allowing attackers to enter input data containing triggers to activate the backdoor and generate their desired output. Existing backdoor attack methods primarily focused on target noise-to-image and text-to-image tasks, with limited work on backdoor attacks in image-to-image tasks. Furthermore, traditional backdoor attacks often rely on a single, conspicuous trigger to generate a fixed target image, lacking concealability and flexibility. To address these limitations, we propose a novel backdoor attack method called "Parasite" for image-to-image tasks in diffusion models, which not only is the first to leverage steganography for triggers hiding, but also allows attackers to embed the target content as a backdoor trigger to achieve a more flexible attack. "Parasite" as a novel attack method effectively bypasses existing detection frameworks to execute backdoor attacks. In our experiments, "Parasite" achieved a 0 percent backdoor detection rate against the mainstream defense frameworks. In addition, in the ablation study, we discuss the influence of different hiding coefficients on the attack results. You can find our code at this https URL. 

**Abstract (ZH)**: Recent 差生攻击：一种高效的图像生成模型中的隐秘后门攻击方法 

---
# How to Enable LLM with 3D Capacity? A Survey of Spatial Reasoning in LLM 

**Title (ZH)**: 如何启用具有三维能力的LLM？关于LLM中空间推理的综述 

**Authors**: Jirong Zha, Yuxuan Fan, Xiao Yang, Chen Gao, Xinlei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.05786)  

**Abstract**: 3D spatial understanding is essential in real-world applications such as robotics, autonomous vehicles, virtual reality, and medical imaging. Recently, Large Language Models (LLMs), having demonstrated remarkable success across various domains, have been leveraged to enhance 3D understanding tasks, showing potential to surpass traditional computer vision methods. In this survey, we present a comprehensive review of methods integrating LLMs with 3D spatial understanding. We propose a taxonomy that categorizes existing methods into three branches: image-based methods deriving 3D understanding from 2D visual data, point cloud-based methods working directly with 3D representations, and hybrid modality-based methods combining multiple data streams. We systematically review representative methods along these categories, covering data representations, architectural modifications, and training strategies that bridge textual and 3D modalities. Finally, we discuss current limitations, including dataset scarcity and computational challenges, while highlighting promising research directions in spatial perception, multi-modal fusion, and real-world applications. 

**Abstract (ZH)**: 三维空间理解在机器人技术、自动驾驶车辆、虚拟现实和医学成像等实际应用中至关重要。近年来，大型语言模型（LLMs）在各个领域展现出显著的成功，并被用于增强三维理解任务，显示出超越传统计算机视觉方法的潜力。在本文综述中，我们对将LLMs与三维空间理解相结合的方法进行了全面回顾。我们提出了一个分类体系，将现有方法归类为三大分支：基于图像的方法、基于点云的方法以及混合模态方法。我们系统地回顾了这些类别中的代表性方法，涵盖数据表示、架构修改以及将文本和三维模态结合起来的训练策略。最后，我们讨论了当前的局限性，包括数据集稀缺性和计算挑战，并强调了在空间感知、多模态融合和实际应用方面的有前景的研究方向。 

---
# Video Flow as Time Series: Discovering Temporal Consistency and Variability for VideoQA 

**Title (ZH)**: 视频流作为时间序列：发现视频问答中的 temporal consistency 和 variability 

**Authors**: Zijie Song, Zhenzhen Hu, Yixiao Ma, Jia Li, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2504.05783)  

**Abstract**: Video Question Answering (VideoQA) is a complex video-language task that demands a sophisticated understanding of both visual content and temporal dynamics. Traditional Transformer-style architectures, while effective in integrating multimodal data, often simplify temporal dynamics through positional encoding and fail to capture non-linear interactions within video sequences. In this paper, we introduce the Temporal Trio Transformer (T3T), a novel architecture that models time consistency and time variability. The T3T integrates three key components: Temporal Smoothing (TS), Temporal Difference (TD), and Temporal Fusion (TF). The TS module employs Brownian Bridge for capturing smooth, continuous temporal transitions, while the TD module identifies and encodes significant temporal variations and abrupt changes within the video content. Subsequently, the TF module synthesizes these temporal features with textual cues, facilitating a deeper contextual understanding and response accuracy. The efficacy of the T3T is demonstrated through extensive testing on multiple VideoQA benchmark datasets. Our results underscore the importance of a nuanced approach to temporal modeling in improving the accuracy and depth of video-based question answering. 

**Abstract (ZH)**: 视频问答（VideoQA）是一种复杂的时间-语言任务，要求对视觉内容和时间动态有精深的理解。传统的基于Transformer的架构虽然在整合多模态数据方面很有效，但往往通过位置编码简化了时间动态，无法捕获视频序列内的非线性交互。本文提出了一种新颖的时间三重Transformer（T3T）架构，用于建模时间和时间变异性。T3T集成了三个关键模块：时间平滑（TS）、时间差分（TD）和时间融合（TF）。TS模块采用布朗桥来捕捉平滑连续的时间过渡，TD模块识别并编码视频内容中的重要时间变化和突然变化，TF模块将这些时间特征与文本线索合成，促进更深入的语境理解和答案准确性。T3T的有效性通过在多个视频问答基准数据集上的广泛测试得到验证。我们的结果强调了在提高基于视频的问题回答准确性和深度方面精细化时间建模的重要性。 

---
# MDK12-Bench: A Multi-Discipline Benchmark for Evaluating Reasoning in Multimodal Large Language Models 

**Title (ZH)**: MDK12-Bench: 一种多学科基准，用于评估多模态大型语言模型的推理能力 

**Authors**: Pengfei Zhou, Fanrui Zhang, Xiaopeng Peng, Zhaopan Xu, Jiaxin Ai, Yansheng Qiu, Chuanhao Li, Zhen Li, Ming Li, Yukang Feng, Jianwen Sun, Haoquan Zhang, Zizhen Li, Xiaofeng Mao, Wangbo Zhao, Kai Wang, Xiaojun Chang, Wenqi Shao, Yang You, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05782)  

**Abstract**: Multimodal reasoning, which integrates language and visual cues into problem solving and decision making, is a fundamental aspect of human intelligence and a crucial step toward artificial general intelligence. However, the evaluation of multimodal reasoning capabilities in Multimodal Large Language Models (MLLMs) remains inadequate. Most existing reasoning benchmarks are constrained by limited data size, narrow domain coverage, and unstructured knowledge distribution. To close these gaps, we introduce MDK12-Bench, a multi-disciplinary benchmark assessing the reasoning capabilities of MLLMs via real-world K-12 examinations. Spanning six disciplines (math, physics, chemistry, biology, geography, and information science), our benchmark comprises 140K reasoning instances across diverse difficulty levels from primary school to 12th grade. It features 6,827 instance-level knowledge point annotations based on a well-organized knowledge structure, detailed answer explanations, difficulty labels and cross-year partitions, providing a robust platform for comprehensive evaluation. Additionally, we present a novel dynamic evaluation framework to mitigate data contamination issues by bootstrapping question forms, question types, and image styles during evaluation. Extensive experiment on MDK12-Bench reveals the significant limitation of current MLLMs in multimodal reasoning. The findings on our benchmark provide insights into the development of the next-generation models. Our data and codes are available at this https URL. 

**Abstract (ZH)**: 多模态推理能力评估：MDK12-Bench多学科基准测试 

---
# Transferable Mask Transformer: Cross-domain Semantic Segmentation with Region-adaptive Transferability Estimation 

**Title (ZH)**: 可转移掩码变换器：基于区域自适应转移性估计的跨域语义分割 

**Authors**: Enming Zhang, Zhengyu Li, Yanru Wu, Jingge Wang, Yang Tan, Ruizhe Zhao, Guan Wang, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.05774)  

**Abstract**: Recent advances in Vision Transformers (ViTs) have set new benchmarks in semantic segmentation. However, when adapting pretrained ViTs to new target domains, significant performance degradation often occurs due to distribution shifts, resulting in suboptimal global attention. Since self-attention mechanisms are inherently data-driven, they may fail to effectively attend to key objects when source and target domains exhibit differences in texture, scale, or object co-occurrence patterns. While global and patch-level domain adaptation methods provide partial solutions, region-level adaptation with dynamically shaped regions is crucial due to spatial heterogeneity in transferability across different image areas. We present Transferable Mask Transformer (TMT), a novel region-level adaptation framework for semantic segmentation that aligns cross-domain representations through spatial transferability analysis. TMT consists of two key components: (1) An Adaptive Cluster-based Transferability Estimator (ACTE) that dynamically segments images into structurally and semantically coherent regions for localized transferability assessment, and (2) A Transferable Masked Attention (TMA) module that integrates region-specific transferability maps into ViTs' attention mechanisms, prioritizing adaptation in regions with low transferability and high semantic uncertainty. Comprehensive evaluations across 20 cross-domain pairs demonstrate TMT's superiority, achieving an average 2% MIoU improvement over vanilla fine-tuning and a 1.28% increase compared to state-of-the-art baselines. The source code will be publicly available. 

**Abstract (ZH)**: Recent Advances in Vision Transformers (ViTs)已在语义分割任务中设立了新基准。然而，在将预训练的ViTs应用于新的目标领域时，由于分布偏移，通常会出现显著的性能下降，导致全局注意效果欠佳。由于自注意力机制本质上是数据驱动的，当源领域和目标领域在纹理、尺度或物体共现模式上存在差异时，它们可能无法有效关注关键对象。虽然全局和 patch 级别领域适应方法提供了一部分解决方案，但在不同图像区域间转移性具有空间异质性的背景下，基于区域的动态形状区域适应至关重要。我们提出了可转移掩码变压器（TMT），这是一种新颖的基于区域的适应框架，通过空间转移性分析对跨域表示进行对齐。TMT 包含两个关键组件：（1）自适应集群基转移性估计器（ACTE），它可以动态地将图像分割成结构上和语义上一致的区域，以进行局部转移性评估；（2）可转移遮罩注意力（TMA）模块，该模块将区域特定的转移性图集成到 ViTs 的注意力机制中，在转移性低且语义不确定性高的区域优先进行适应。在20个跨域配对的全面评估中，TMT 展示了其优越性，平均MIoU提高2%，相比标准微调提高了1.28%，相比最新基线提高了1.28%。源代码将公开。 

---
# A Lightweight Multi-Module Fusion Approach for Korean Character Recognition 

**Title (ZH)**: 一种轻量级多模块融合方法用于韩文字识别 

**Authors**: Inho Jake Park, Jaehoon Jay Jeong, Ho-Sang Jo  

**Link**: [PDF](https://arxiv.org/pdf/2504.05770)  

**Abstract**: Optical Character Recognition (OCR) is essential in applications such as document processing, license plate recognition, and intelligent surveillance. However, existing OCR models often underperform in real-world scenarios due to irregular text layouts, poor image quality, character variability, and high computational costs.
This paper introduces SDA-Net (Stroke-Sensitive Attention and Dynamic Context Encoding Network), a lightweight and efficient architecture designed for robust single-character recognition. SDA-Net incorporates: (1) a Dual Attention Mechanism to enhance stroke-level and spatial feature extraction; (2) a Dynamic Context Encoding module that adaptively refines semantic information using a learnable gating mechanism; (3) a U-Net-inspired Feature Fusion Strategy for combining low-level and high-level features; and (4) a highly optimized lightweight backbone that reduces memory and computational demands.
Experimental results show that SDA-Net achieves state-of-the-art accuracy on challenging OCR benchmarks, with significantly faster inference, making it well-suited for deployment in real-time and edge-based OCR systems. 

**Abstract (ZH)**: 基于笔画敏感注意力和动态上下文编码的轻量化OCR网络（SDA-Net）：面向实时和边缘端OCR系统的高效单字符识别 

---
# Temporal Dynamic Embedding for Irregularly Sampled Time Series 

**Title (ZH)**: 不规则采样时间序列的时序动态嵌入 

**Authors**: Mincheol Kim, Soo-Yong Shin  

**Link**: [PDF](https://arxiv.org/pdf/2504.05768)  

**Abstract**: In several practical applications, particularly healthcare, clinical data of each patient is individually recorded in a database at irregular intervals as required. This causes a sparse and irregularly sampled time series, which makes it difficult to handle as a structured representation of the prerequisites of neural network models. We therefore propose temporal dynamic embedding (TDE), which enables neural network models to receive data that change the number of variables over time. TDE regards each time series variable as an embedding vector evolving over time, instead of a conventional fixed structured representation, which causes a critical missing problem. For each time step, TDE allows for the selective adoption and aggregation of only observed variable subsets and represents the current status of patient based on current observations. The experiment was conducted on three clinical datasets: PhysioNet 2012, MIMIC-III, and PhysioNet 2019. The TDE model performed competitively or better than the imputation-based baseline and several recent state-of-the-art methods with reduced training runtime. 

**Abstract (ZH)**: 在几个实际应用中，尤其是在医疗健康领域，每位患者的临床数据会在不规则的时间间隔内以个体化的方式记录在数据库中。这会产生稀疏且不规则采样的时间序列，使得将其作为神经网络模型结构化表示处理变得困难。因此，我们提出了时间动态嵌入（TDE），该方法使神经网络模型能够接收随时间变化的变量数量的数据。TDE 将每个时间序列变量视为随时间演化的嵌入向量，而不是传统的固定结构表示，从而解决了关键的缺失问题。对于每个时间步，TDE 只允许选择性地采用和聚合仅观察到的变量子集，并基于当前观察来表示患者当前的状态。在三项临床数据集中对 TDE 模型进行了实验：PhysioNet 2012、MIMIC-III 和 PhysioNet 2019。实验结果表明，与基于插补的基线方法以及几种最新的最先进方法相比，TDE 模型在减少训练运行时间的情况下表现竞争力或更佳。 

---
# Unraveling Human-AI Teaming: A Review and Outlook 

**Title (ZH)**: 探究人机协作：综述与展望 

**Authors**: Bowen Lou, Tian Lu, Raghu Santanam, Yingjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05755)  

**Abstract**: Artificial Intelligence (AI) is advancing at an unprecedented pace, with clear potential to enhance decision-making and productivity. Yet, the collaborative decision-making process between humans and AI remains underdeveloped, often falling short of its transformative possibilities. This paper explores the evolution of AI agents from passive tools to active collaborators in human-AI teams, emphasizing their ability to learn, adapt, and operate autonomously in complex environments. This paradigm shifts challenges traditional team dynamics, requiring new interaction protocols, delegation strategies, and responsibility distribution frameworks. Drawing on Team Situation Awareness (SA) theory, we identify two critical gaps in current human-AI teaming research: the difficulty of aligning AI agents with human values and objectives, and the underutilization of AI's capabilities as genuine team members. Addressing these gaps, we propose a structured research outlook centered on four key aspects of human-AI teaming: formulation, coordination, maintenance, and training. Our framework highlights the importance of shared mental models, trust-building, conflict resolution, and skill adaptation for effective teaming. Furthermore, we discuss the unique challenges posed by varying team compositions, goals, and complexities. This paper provides a foundational agenda for future research and practical design of sustainable, high-performing human-AI teams. 

**Abstract (ZH)**: 人工智能（AI）的发展速度史无前例，具有增强决策和生产力的潜力。然而，人类与AI的合作决策过程仍处于初级阶段，往往未能充分发挥其变革潜力。本文探讨了AI代理从被动工具向人类-AI团队中主动合作者的演变，强调其在复杂环境中的学习、适应和自主操作能力。这一范式转变挑战了传统的团队动态，要求新的交互协议、委派策略和责任分配框架。基于团队态势意识（SA）理论，我们指出了当前人类-AI团队研究中的两个关键缺口：AI代理与人类价值观和目标的对齐困难，以及AI作为真正团队成员潜力的未充分利用。针对这些缺口，我们提出了一种以人类-AI团队四方核心方面为焦点的整体研究方向：建模、协调、维护和训练。我们的框架强调了共享认知模型、信任建立、冲突解决和技能适应对于有效团队合作的重要性。此外，我们讨论了不同团队组成、目标和复杂性带来的独特挑战。本文为未来研究和可持续、高性能的人类-AI团队的实际设计提供了基础议程。 

---
# DDT: Decoupled Diffusion Transformer 

**Title (ZH)**: DDT: 解耦扩散变换器 

**Authors**: Shuai Wang, Zhi Tian, Weilin Huang, Limin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05741)  

**Abstract**: Diffusion transformers have demonstrated remarkable generation quality, albeit requiring longer training iterations and numerous inference steps. In each denoising step, diffusion transformers encode the noisy inputs to extract the lower-frequency semantic component and then decode the higher frequency with identical modules. This scheme creates an inherent optimization dilemma: encoding low-frequency semantics necessitates reducing high-frequency components, creating tension between semantic encoding and high-frequency decoding. To resolve this challenge, we propose a new \textbf{\color{ddt}D}ecoupled \textbf{\color{ddt}D}iffusion \textbf{\color{ddt}T}ransformer~(\textbf{\color{ddt}DDT}), with a decoupled design of a dedicated condition encoder for semantic extraction alongside a specialized velocity decoder. Our experiments reveal that a more substantial encoder yields performance improvements as model size increases. For ImageNet $256\times256$, Our DDT-XL/2 achieves a new state-of-the-art performance of {1.31 FID}~(nearly $4\times$ faster training convergence compared to previous diffusion transformers). For ImageNet $512\times512$, Our DDT-XL/2 achieves a new state-of-the-art FID of 1.28. Additionally, as a beneficial by-product, our decoupled architecture enhances inference speed by enabling the sharing self-condition between adjacent denoising steps. To minimize performance degradation, we propose a novel statistical dynamic programming approach to identify optimal sharing strategies. 

**Abstract (ZH)**: 解耦扩散变压器（DDT）：解耦设计的专用语义编码器与专门的速度解码器 

---
# Rank-Then-Score: Enhancing Large Language Models for Automated Essay Scoring 

**Title (ZH)**: 基于排名然后评分的方法：增强大规模语言模型以实现自动化作文评分 

**Authors**: Yida Cai, Kun Liang, Sanwoo Lee, Qinghan Wang, Yunfang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.05736)  

**Abstract**: In recent years, large language models (LLMs) achieve remarkable success across a variety of tasks. However, their potential in the domain of Automated Essay Scoring (AES) remains largely underexplored. Moreover, compared to English data, the methods for Chinese AES is not well developed. In this paper, we propose Rank-Then-Score (RTS), a fine-tuning framework based on large language models to enhance their essay scoring capabilities. Specifically, we fine-tune the ranking model (Ranker) with feature-enriched data, and then feed the output of the ranking model, in the form of a candidate score set, with the essay content into the scoring model (Scorer) to produce the final score. Experimental results on two benchmark datasets, HSK and ASAP, demonstrate that RTS consistently outperforms the direct prompting (Vanilla) method in terms of average QWK across all LLMs and datasets, and achieves the best performance on Chinese essay scoring using the HSK dataset. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在各种任务中取得了显著成功。然而，它们在自动作文评分（AES）领域的潜力尚未得到充分探索。此外，与英语数据相比，中文AES的方法发展不够成熟。在本文中，我们提出了一种基于大规模语言模型的Rank-Then-Score（RTS）微调框架，以增强其作文评分能力。具体来说，我们使用特征丰富化的数据对排序模型（Ranker）进行微调，然后将排序模型的输出（候选评分集合）与作文内容一起输入评分模型（Scorer），以生成最终评分。在HSK和ASAP两个基准数据集上的实验结果表明，RTS在所有LLM和数据集中平均QWK指标上始终优于直接提示（Vanilla）方法，并在HSK数据集上实现了最佳的中文作文评分性能。 

---
# Architecture independent generalization bounds for overparametrized deep ReLU networks 

**Title (ZH)**: 独立于架构的一般化界面对过参数化深ReLU网络的分析 

**Authors**: Thomas Chen, Chun-Kai Kevin Chien, Patricia Muñoz Ewald, Andrew G. Moore  

**Link**: [PDF](https://arxiv.org/pdf/2504.05695)  

**Abstract**: We prove that overparametrized neural networks are able to generalize with a test error that is independent of the level of overparametrization, and independent of the Vapnik-Chervonenkis (VC) dimension. We prove explicit bounds that only depend on the metric geometry of the test and training sets, on the regularity properties of the activation function, and on the operator norms of the weights and norms of biases. For overparametrized deep ReLU networks with a training sample size bounded by the input space dimension, we explicitly construct zero loss minimizers without use of gradient descent, and prove that the generalization error is independent of the network architecture. 

**Abstract (ZH)**: 我们证明，过参数化的神经网络能够实现与过参数化程度和Vapnik-Chervonenkis维数无关的测试误差泛化。我们推导了仅依赖于测试集和训练集的度量几何特性、激活函数的正则性质、权重的操作范数以及偏置的范数的显式边界。对于输入空间维数受限的训练样本大小的过参数化深层ReLU网络，我们明确构造了无需使用梯度下降的零损失最小化器，并证明了泛化误差与网络架构无关。 

---
# Large Language Models Enhanced Hyperbolic Space Recommender Systems 

**Title (ZH)**: 大型语言模型增强的双曲空间推荐系统 

**Authors**: Wentao Cheng, Zhida Qin, Zexue Wu, Pengzhan Zhou, Tianyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05694)  

**Abstract**: Large Language Models (LLMs) have attracted significant attention in recommender systems for their excellent world knowledge capabilities. However, existing methods that rely on Euclidean space struggle to capture the rich hierarchical information inherent in textual and semantic data, which is essential for capturing user preferences. The geometric properties of hyperbolic space offer a promising solution to address this issue. Nevertheless, integrating LLMs-based methods with hyperbolic space to effectively extract and incorporate diverse hierarchical information is non-trivial. To this end, we propose a model-agnostic framework, named HyperLLM, which extracts and integrates hierarchical information from both structural and semantic perspectives. Structurally, HyperLLM uses LLMs to generate multi-level classification tags with hierarchical parent-child relationships for each item. Then, tag-item and user-item interactions are jointly learned and aligned through contrastive learning, thereby providing the model with clear hierarchical information. Semantically, HyperLLM introduces a novel meta-optimized strategy to extract hierarchical information from semantic embeddings and bridge the gap between the semantic and collaborative spaces for seamless integration. Extensive experiments show that HyperLLM significantly outperforms recommender systems based on hyperbolic space and LLMs, achieving performance improvements of over 40%. Furthermore, HyperLLM not only improves recommender performance but also enhances training stability, highlighting the critical role of hierarchical information in recommender systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）在推荐系统中的世界知识能力使其在推荐领域引起了广泛关注。然而，现有的依赖欧几里得空间的方法难以捕捉文本和语义数据中固有的丰富层次信息，而这些信息对于捕捉用户偏好至关重要。双曲空间的几何性质提供了解决这一问题的潜在方案。然而，将LLMs方法与双曲空间有效结合以提取和整合多种层次信息仍然是一个挑战。为此，我们提出了一种模型通用框架HyperLLM，从结构和语义两个视角提取并整合层次信息。从结构上看，HyperLLM 使用LLMs为每个项目生成具有层次父子女关系的多级分类标签。然后，通过对比学习联合学习和对齐标签-项目和用户-项目的交互，从而为模型提供了清晰的层次信息。从语义上看，HyperLLM 引入了一种新颖的元优化策略，从语义嵌入中提取层次信息，并在语义和协同空间之间建立桥梁，以实现无缝集成。广泛的实验表明，HyperLLM 显著优于基于双曲空间和LLMs的推荐系统，性能提升超过40%。此外，HyperLLM 不仅提高了推荐性能，还增强了训练稳定性，突显了层次信息在推荐系统中的关键作用。 

---
# STRIVE: A Think & Improve Approach with Iterative Refinement for Enhancing Question Quality Estimation 

**Title (ZH)**: STRIVE: 一种迭代优化的思考与改进方法以提升问题质量估测 

**Authors**: Aniket Deroy, Subhankar Maity  

**Link**: [PDF](https://arxiv.org/pdf/2504.05693)  

**Abstract**: Automatically assessing question quality is crucial for educators as it saves time, ensures consistency, and provides immediate feedback for refining teaching materials. We propose a novel methodology called STRIVE (Structured Thinking and Refinement with multiLLMs for Improving Verified Question Estimation) using a series of Large Language Models (LLMs) for automatic question evaluation. This approach aims to improve the accuracy and depth of question quality assessment, ultimately supporting diverse learners and enhancing educational practices. The method estimates question quality in an automated manner by generating multiple evaluations based on the strengths and weaknesses of the provided question and then choosing the best solution generated by the LLM. Then the process is improved by iterative review and response with another LLM until the evaluation metric values converge. This sophisticated method of evaluating question quality improves the estimation of question quality by automating the task of question quality evaluation. Correlation scores show that using this proposed method helps to improve correlation with human judgments compared to the baseline method. Error analysis shows that metrics like relevance and appropriateness improve significantly relative to human judgments by using STRIVE. 

**Abstract (ZH)**: 自动评估问题质量对于教育者至关重要，因为它可以节省时间、确保一致性和提供即时反馈以改进教学材料。我们提出了一种新型方法STRIVE（结构化思考与多LLMs辅助改进验证问题估计），利用一系列大型语言模型（LLMs）进行自动问题评估。该方法旨在提高问题质量评估的准确性和深度，最终支持多样化的学习者并增强教育实践。该方法通过对提供的问题的优点和缺点生成多个评估来自动估计问题质量，然后选择由LLM生成的最佳解决方案。该过程通过与另一个LLM进行迭代审查和响应，直到评估指标值收敛，从而改进了问题质量评估。相关性分析表明，使用该提出的 STRIVE 方法有助于提高与人类判断的相关性，而基准方法则不然。误差分析表明，通过使用 STRIVE，相关度和适宜性等指标相对于人类判断有显著改善。 

---
# kNN-SVC: Robust Zero-Shot Singing Voice Conversion with Additive Synthesis and Concatenation Smoothness Optimization 

**Title (ZH)**: kNN-SVC：基于加性合成和拼接平滑优化的鲁棒零样本歌唱语音转换 

**Authors**: Keren Shao, Ke Chen, Matthew Baas, Shlomo Dubnov  

**Link**: [PDF](https://arxiv.org/pdf/2504.05686)  

**Abstract**: Robustness is critical in zero-shot singing voice conversion (SVC). This paper introduces two novel methods to strengthen the robustness of the kNN-VC framework for SVC. First, kNN-VC's core representation, WavLM, lacks harmonic emphasis, resulting in dull sounds and ringing artifacts. To address this, we leverage the bijection between WavLM, pitch contours, and spectrograms to perform additive synthesis, integrating the resulting waveform into the model to mitigate these issues. Second, kNN-VC overlooks concatenative smoothness, a key perceptual factor in SVC. To enhance smoothness, we propose a new distance metric that filters out unsuitable kNN candidates and optimize the summing weights of the candidates during inference. Although our techniques are built on the kNN-VC framework for implementation convenience, they are broadly applicable to general concatenative neural synthesis models. Experimental results validate the effectiveness of these modifications in achieving robust SVC. Demo: this http URL Code: this https URL 

**Abstract (ZH)**: 零样本唱歌语音转换的鲁棒性至关重要。本文介绍了两种新的方法以增强kNN-VC框架在唱歌语音转换（SVC）中的鲁棒性。首先，kNN-VC的核心表示WavLM缺乏谐波强调，导致声音单调且出现振铃效应。为解决此问题，我们利用WavLM、音高轮廓和频谱图之间的双射关系进行加法合成，将生成的波形整合到模型中以减轻这些问题。其次，kNN-VC忽视了连接平滑性，这是SVC中的关键感知因素。为了提高平滑性，我们提出了一种新的距离度量，该度量过滤出不合适的kNN候选者，并在推理过程中优化候选者的加权求和。尽管我们的技术基于kNN-VC框架以便于实现，但它们广泛适用于一般的连接合成神经网络模型。实验结果验证了这些修改在实现鲁棒SVC方面的有效性。Demo: this http URL Code: this https URL 

---
# TARO: Timestep-Adaptive Representation Alignment with Onset-Aware Conditioning for Synchronized Video-to-Audio Synthesis 

**Title (ZH)**: TARO：基于起始意识调节的时间步自适应表示对齐术用于同步视频到音频合成 

**Authors**: Tri Ton, Ji Woo Hong, Chang D. Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2504.05684)  

**Abstract**: This paper introduces Timestep-Adaptive Representation Alignment with Onset-Aware Conditioning (TARO), a novel framework for high-fidelity and temporally coherent video-to-audio synthesis. Built upon flow-based transformers, which offer stable training and continuous transformations for enhanced synchronization and audio quality, TARO introduces two key innovations: (1) Timestep-Adaptive Representation Alignment (TRA), which dynamically aligns latent representations by adjusting alignment strength based on the noise schedule, ensuring smooth evolution and improved fidelity, and (2) Onset-Aware Conditioning (OAC), which integrates onset cues that serve as sharp event-driven markers of audio-relevant visual moments to enhance synchronization with dynamic visual events. Extensive experiments on the VGGSound and Landscape datasets demonstrate that TARO outperforms prior methods, achieving relatively 53\% lower Frechet Distance (FD), 29% lower Frechet Audio Distance (FAD), and a 97.19% Alignment Accuracy, highlighting its superior audio quality and synchronization precision. 

**Abstract (ZH)**: 基于起峰感知条件的时间步自适应表示对齐（TARO）：高保真和时序一致性视频到音频合成的新框架 

---
# Towards Smarter Hiring: Are Zero-Shot and Few-Shot Pre-trained LLMs Ready for HR Spoken Interview Transcript Analysis? 

**Title (ZH)**: 基于零样本和少样本预训练大语言模型，招聘面试转录分析是否 Ready 了？ 

**Authors**: Subhankar Maity, Aniket Deroy, Sudeshna Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2504.05683)  

**Abstract**: This research paper presents a comprehensive analysis of the performance of prominent pre-trained large language models (LLMs), including GPT-4 Turbo, GPT-3.5 Turbo, text-davinci-003, text-babbage-001, text-curie-001, text-ada-001, llama-2-7b-chat, llama-2-13b-chat, and llama-2-70b-chat, in comparison to expert human evaluators in providing scores, identifying errors, and offering feedback and improvement suggestions to candidates during mock HR (Human Resources) interviews. We introduce a dataset called HURIT (Human Resource Interview Transcripts), which comprises 3,890 HR interview transcripts sourced from real-world HR interview scenarios. Our findings reveal that pre-trained LLMs, particularly GPT-4 Turbo and GPT-3.5 Turbo, exhibit commendable performance and are capable of producing evaluations comparable to those of expert human evaluators. Although these LLMs demonstrate proficiency in providing scores comparable to human experts in terms of human evaluation metrics, they frequently fail to identify errors and offer specific actionable advice for candidate performance improvement in HR interviews. Our research suggests that the current state-of-the-art pre-trained LLMs are not fully conducive for automatic deployment in an HR interview assessment. Instead, our findings advocate for a human-in-the-loop approach, to incorporate manual checks for inconsistencies and provisions for improving feedback quality as a more suitable strategy. 

**Abstract (ZH)**: 本研究论文对包括GPT-4 Turbo、GPT-3.5 Turbo、text-davinci-003、text-babbage-001、text-curie-001、text-ada-001、llama-2-7b-chat、llama-2-13b-chat和llama-2-70b-chat等 prominent 预训练大规模语言模型（LLMs）在模拟人力资源（HR）面试中提供评分、识别错误和提供反馈及改进建议方面的性能进行了全面分析，并将其与专家人力资源评估员进行了比较。我们介绍了名为HURIT（人力资源面试转录）的数据集，包含3,890份实际人力资源面试场景的转录。我们的研究发现，预训练LLMs，特别是GPT-4 Turbo和GPT-3.5 Turbo表现出色，能够产生与专家人力资源评估员相当的评价。尽管这些LLMs在提供评分方面在人力资源评估指标中表现出与人类专家相当的熟练度，但在识别错误和为候选人面试表现提供具体可操作的改进建议方面却经常失败。研究结果表明，当前最先进的预训练LLMs并不完全适合在人力资源面试评估中的自动部署。相反，我们的研究建议采用人工在环中的方法，结合人工检查不一致性和提供提高反馈质量的方法，作为更合适的策略。 

---
# Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-spoofing 

**Title (ZH)**: Nes2Net：一种轻量级嵌套架构，用于基础模型驱动的语音防 spoofing 

**Authors**: Tianchi Liu, Duc-Tuan Truong, Rohan Kumar Das, Kong Aik Lee, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.05657)  

**Abstract**: Speech foundation models have significantly advanced various speech-related tasks by providing exceptional representation capabilities. However, their high-dimensional output features often create a mismatch with downstream task models, which typically require lower-dimensional inputs. A common solution is to apply a dimensionality reduction (DR) layer, but this approach increases parameter overhead, computational costs, and risks losing valuable information. To address these issues, we propose Nested Res2Net (Nes2Net), a lightweight back-end architecture designed to directly process high-dimensional features without DR layers. The nested structure enhances multi-scale feature extraction, improves feature interaction, and preserves high-dimensional information. We first validate Nes2Net on CtrSVDD, a singing voice deepfake detection dataset, and report a 22% performance improvement and an 87% back-end computational cost reduction over the state-of-the-art baseline. Additionally, extensive testing across four diverse datasets: ASVspoof 2021, ASVspoof 5, PartialSpoof, and In-the-Wild, covering fully spoofed speech, adversarial attacks, partial spoofing, and real-world scenarios, consistently highlights Nes2Net's superior robustness and generalization capabilities. The code package and pre-trained models are available at this https URL. 

**Abstract (ZH)**: 基于演讲的基础模型显著推进了各种演讲相关任务，但其高维输出特征经常会与下游任务模型所需的低维输入产生不匹配。一种常见的解决方案是应用降维层，但这会增加参数开销、计算成本，并有损失有价值信息的风险。为解决这些问题，我们提出了一种轻量级后端架构Nested Res2Net（Nes2Net），该架构可以直接处理高维特征而不使用降维层。嵌套结构增强了多尺度特征提取、提高了特征交互并保留了高维信息。我们在 singing voice deepfake 检测数据集 CtrSVDD 上验证了 Nes2Net，并报告了相比最先进的基线提高了22%的性能和87%的后端计算成本降低。此外，跨四个不同数据集（ASVspoof 2021、ASVspoof 5、PartialSpoof 和 In-the-Wild，涵盖完全欺骗性语音、对抗性攻击、部分欺骗性和真实场景）的广泛测试一致突出了 Nes2Net 优越的鲁棒性和泛化能力。 code package 和预训练模型可从此链接下载。 

---
# Lattice: Learning to Efficiently Compress the Memory 

**Title (ZH)**: Lattice: 学习高效压缩内存 

**Authors**: Mahdi Karami, Vahab Mirrokni  

**Link**: [PDF](https://arxiv.org/pdf/2504.05646)  

**Abstract**: Attention mechanisms have revolutionized sequence learning but suffer from quadratic computational complexity. This paper introduces Lattice, a novel recurrent neural network (RNN) mechanism that leverages the inherent low-rank structure of K-V matrices to efficiently compress the cache into a fixed number of memory slots, achieving sub-quadratic complexity. We formulate this compression as an online optimization problem and derive a dynamic memory update rule based on a single gradient descent step. The resulting recurrence features a state- and input-dependent gating mechanism, offering an interpretable memory update process. The core innovation is the orthogonal update: each memory slot is updated exclusively with information orthogonal to its current state hence incorporation of only novel, non-redundant data, which minimizes the interference with previously stored information. The experimental results show that Lattice achieves the best perplexity compared to all baselines across diverse context lengths, with performance improvement becoming more pronounced as the context length increases. 

**Abstract (ZH)**: 注意力机制革新了序列学习但遭受着二次时间复杂度的困扰。本文提出了一种新颖的递归神经网络机制Lattice，利用K-V矩阵固有的低秩结构高效压缩缓存至固定数量的记忆单元，实现次二次时间复杂度。我们将这种压缩形式化为在线优化问题，并基于单步梯度下降推导出动态内存更新规则。该递归机制包含状态和输入依赖的门控机制，提供可解释的记忆更新过程。核心创新在于正交更新：每个记忆单元仅根据与当前状态正交的信息进行更新，从而仅 Incorporate 新颖且无冗余的数据，最大限度地减少对已存储信息的干扰。实验结果表明，Lattice 在各种上下文长度下的困惑度均优于所有基线模型，且随着上下文长度增加，性能提升更为显著。 

---
# DBOT: Artificial Intelligence for Systematic Long-Term Investing 

**Title (ZH)**: DBOT：系统长期投资的人工智能方法 

**Authors**: Vasant Dhar, João Sedoc  

**Link**: [PDF](https://arxiv.org/pdf/2504.05639)  

**Abstract**: Long-term investing was previously seen as requiring human judgment. With the advent of generative artificial intelligence (AI) systems, automated systematic long-term investing is now feasible. In this paper, we present DBOT, a system whose goal is to reason about valuation like Aswath Damodaran, who is a unique expert in the investment arena in terms of having published thousands of valuations on companies in addition to his numerous writings on the topic, which provide ready training data for an AI system. DBOT can value any publicly traded company. DBOT can also be back-tested, making its behavior and performance amenable to scientific inquiry. We compare DBOT to its analytic parent, Damodaran, and highlight the research challenges involved in raising its current capability to that of Damodaran's. Finally, we examine the implications of DBOT-like AI agents for the financial industry, especially how they will impact the role of human analysts in valuation. 

**Abstract (ZH)**: 长线投资 previously viewed as requiring human判断，现在借助生成型人工智能系统变得可行。本文介绍DBOT系统，其目标是像拥有数千家企业估值及大量相关著作的Aswath Damodaran一样进行估值推理。DBOT可以估值任何公开交易的公司，并可进行回测，使其行为和表现易于科学探究。我们将DBOT与其分析型前辈Damodaran进行比较，并强调提高DBOT当前能力至Damodaran水平的研究挑战。最后，我们探讨类似DBOT的AI代理对金融行业的影响，尤其是它们如何影响估值中的分析师角色。 

---
# Reasoning Towards Fairness: Mitigating Bias in Language Models through Reasoning-Guided Fine-Tuning 

**Title (ZH)**: 基于推理的公平性推理：通过推理指导微调减轻语言模型中的偏见 

**Authors**: Sanchit Kabra, Akshita Jha, Chandan Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2504.05632)  

**Abstract**: Recent advances in large-scale generative language models have shown that reasoning capabilities can significantly improve model performance across a variety of tasks. However, the impact of reasoning on a model's ability to mitigate stereotypical responses remains largely underexplored. In this work, we investigate the crucial relationship between a model's reasoning ability and fairness, and ask whether improved reasoning capabilities can mitigate harmful stereotypical responses, especially those arising due to shallow or flawed reasoning. We conduct a comprehensive evaluation of multiple open-source LLMs, and find that larger models with stronger reasoning abilities exhibit substantially lower stereotypical bias on existing fairness benchmarks. Building on this insight, we introduce ReGiFT -- Reasoning Guided Fine-Tuning, a novel approach that extracts structured reasoning traces from advanced reasoning models and infuses them into models that lack such capabilities. We use only general-purpose reasoning and do not require any fairness-specific supervision for bias mitigation. Notably, we see that models fine-tuned using ReGiFT not only improve fairness relative to their non-reasoning counterparts but also outperform advanced reasoning models on fairness benchmarks. We also analyze how variations in the correctness of the reasoning traces and their length influence model fairness and their overall performance. Our findings highlight that enhancing reasoning capabilities is an effective, fairness-agnostic strategy for mitigating stereotypical bias caused by reasoning flaws. 

**Abstract (ZH)**: 近年来，大规模生成语言模型的研究进展表明，推理能力可以显著提高模型在各种任务中的性能。然而，推理对模型减轻刻板印象响应能力的影响仍然很大程度上未被探索。在本文中，我们研究了模型的推理能力与其公平性之间的关键关系，并询问改进的推理能力是否能够减轻有害的刻板印象响应，尤其是那些由于浅薄或有缺陷的推理引起的响应。我们对多个开源大语言模型进行了全面评估，发现具有更强推理能力的较大模型在现有的公平性基准测试中表现出显著较低的刻板印象偏见。基于这一洞察，我们提出了一种名为ReGiFT（Reasoning Guided Fine-Tuning）的新方法，该方法从先进的推理模型中提取结构化的推理轨迹，并将它们注入缺乏此类能力的模型中。我们仅使用通用推理，无需任何针对公平性的特定监督即可减轻偏见。值得注意的是，使用ReGiFT进行微调后的模型不仅相对于其非推理版本提高了公平性，还在公平性基准测试中优于先进的推理模型。我们还分析了推理轨迹正确性和长度的变化如何影响模型的公平性和整体性能。我们的发现强调，增强推理能力是一种公平性无关的有效策略，用于减轻由于推理缺陷引起的刻板印象偏见。 

---
# Technical Report: Full Version of Analyzing and Optimizing Perturbation of DP-SGD Geometrically 

**Title (ZH)**: 技术报告：几何角度分析与优化DP-SGD扰动的完整版本 

**Authors**: Jiawei Duan, Haibo Hu, Qingqing Ye, Xinyue Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.05618)  

**Abstract**: Differential privacy (DP) has become a prevalent privacy model in a wide range of machine learning tasks, especially after the debut of DP-SGD. However, DP-SGD, which directly perturbs gradients in the training iterations, fails to mitigate the negative impacts of noise on gradient direction. As a result, DP-SGD is often inefficient. Although various solutions (e.g., clipping to reduce the sensitivity of gradients and amplifying privacy bounds to save privacy budgets) are proposed to trade privacy for model efficiency, the root cause of its inefficiency is yet unveiled.
In this work, we first generalize DP-SGD and theoretically derive the impact of DP noise on the training process. Our analysis reveals that, in terms of a perturbed gradient, only the noise on direction has eminent impact on the model efficiency while that on magnitude can be mitigated by optimization techniques, i.e., fine-tuning gradient clipping and learning rate. Besides, we confirm that traditional DP introduces biased noise on the direction when adding unbiased noise to the gradient itself. Overall, the perturbation of DP-SGD is actually sub-optimal from a geometric perspective. Motivated by this, we design a geometric perturbation strategy GeoDP within the DP framework, which perturbs the direction and the magnitude of a gradient, respectively. By directly reducing the noise on the direction, GeoDP mitigates the negative impact of DP noise on model efficiency with the same DP guarantee. Extensive experiments on two public datasets (i.e., MNIST and CIFAR-10), one synthetic dataset and three prevalent models (i.e., Logistic Regression, CNN and ResNet) confirm the effectiveness and generality of our strategy. 

**Abstract (ZH)**: 差分隐私（DP）已成为广泛机器学习任务中的一种主流隐私模型，尤其是在DP-SGD出现之后。然而，DP-SGD直接在训练迭代中扰动梯度，未能减轻噪声对梯度方向的负面影响，导致其效率低下。尽管提出了多种解决方案（例如梯度裁剪以降低梯度敏感性、放大隐私边界以节省隐私预算）来权衡隐私与模型效率，但其效率低下的根本原因仍未揭开。

在本文中，我们首先泛化DP-SGD，并从理论上推导出DP噪声对训练过程的影响。我们的分析表明，在扰动梯度的情况下，仅噪声的方向部分对模型效率产生了显着影响，而幅度噪声则可以通过优化技术（如精细调节梯度裁剪和学习率）来缓解。此外，我们确认当对梯度本身添加无偏噪声时，传统DP会引入偏置方向噪声。整体而言，从几何角度看，DP-SGD的扰动实际上是次优的。受此启发，我们在DP框架内设计了一种几何扰动策略GeoDP，分别扰动梯度的方向和幅度。通过直接减少方向上的噪声，GeoDP在保留相同DP保证的情况下，减轻了DP噪声对模型效率的负面影响。在两个公开数据集（即MNIST和CIFAR-10）、一个合成数据集和三种常见模型（即逻辑回归、卷积神经网络和残差网络）上的广泛实验确认了我们策略的有效性和普适性。 

---
# FedEFC: Federated Learning Using Enhanced Forward Correction Against Noisy Labels 

**Title (ZH)**: FedEFC: 面向噪声标签的增强前向纠错联邦学习 

**Authors**: Seunghun Yu, Jin-Hyun Ahn, Joonhyuk Kang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05615)  

**Abstract**: Federated Learning (FL) is a powerful framework for privacy-preserving distributed learning. It enables multiple clients to collaboratively train a global model without sharing raw data. However, handling noisy labels in FL remains a major challenge due to heterogeneous data distributions and communication constraints, which can severely degrade model performance. To address this issue, we propose FedEFC, a novel method designed to tackle the impact of noisy labels in FL. FedEFC mitigates this issue through two key techniques: (1) prestopping, which prevents overfitting to mislabeled data by dynamically halting training at an optimal point, and (2) loss correction, which adjusts model updates to account for label noise. In particular, we develop an effective loss correction tailored to the unique challenges of FL, including data heterogeneity and decentralized training. Furthermore, we provide a theoretical analysis, leveraging the composite proper loss property, to demonstrate that the FL objective function under noisy label distributions can be aligned with the clean label distribution. Extensive experimental results validate the effectiveness of our approach, showing that it consistently outperforms existing FL techniques in mitigating the impact of noisy labels, particularly under heterogeneous data settings (e.g., achieving up to 41.64% relative performance improvement over the existing loss correction method). 

**Abstract (ZH)**: 联邦学习中的噪声标签应对方法：FedEFC 

---
# FactGuard: Leveraging Multi-Agent Systems to Generate Answerable and Unanswerable Questions for Enhanced Long-Context LLM Extraction 

**Title (ZH)**: FactGuard：利用多智能体系统生成可回答和不可回答的问题以增强长上下文语言模型提取 

**Authors**: Qian-Wen Zhang, Fang Li, Jie Wang, Lingfeng Qiao, Yifei Yu, Di Yin, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.05607)  

**Abstract**: Extractive reading comprehension systems are designed to locate the correct answer to a question within a given text. However, a persistent challenge lies in ensuring these models maintain high accuracy in answering questions while reliably recognizing unanswerable queries. Despite significant advances in large language models (LLMs) for reading comprehension, this issue remains critical, particularly as the length of supported contexts continues to expand. To address this challenge, we propose an innovative data augmentation methodology grounded in a multi-agent collaborative framework. Unlike traditional methods, such as the costly human annotation process required for datasets like SQuAD 2.0, our method autonomously generates evidence-based question-answer pairs and systematically constructs unanswerable questions. Using this methodology, we developed the FactGuard-Bench dataset, which comprises 25,220 examples of both answerable and unanswerable question scenarios, with context lengths ranging from 8K to 128K. Experimental evaluations conducted on seven popular LLMs reveal that even the most advanced models achieve only 61.79% overall accuracy. Furthermore, we emphasize the importance of a model's ability to reason about unanswerable questions to avoid generating plausible but incorrect answers. By implementing efficient data selection and generation within the multi-agent collaborative framework, our method significantly reduces the traditionally high costs associated with manual annotation and provides valuable insights for the training and optimization of LLMs. 

**Abstract (ZH)**: 基于多Agent协作框架的数据增强方法：提高不可答查询识别的抽取式阅读理解系统 

---
# Class Imbalance Correction for Improved Universal Lesion Detection and Tagging in CT 

**Title (ZH)**: CT中改进的通用病灶检测和标记的类别不平衡校正 

**Authors**: Peter D. Erickson, Tejas Sudharshan Mathai, Ronald M. Summers  

**Link**: [PDF](https://arxiv.org/pdf/2504.05591)  

**Abstract**: Radiologists routinely detect and size lesions in CT to stage cancer and assess tumor burden. To potentially aid their efforts, multiple lesion detection algorithms have been developed with a large public dataset called DeepLesion (32,735 lesions, 32,120 CT slices, 10,594 studies, 4,427 patients, 8 body part labels). However, this dataset contains missing measurements and lesion tags, and exhibits a severe imbalance in the number of lesions per label category. In this work, we utilize a limited subset of DeepLesion (6\%, 1331 lesions, 1309 slices) containing lesion annotations and body part label tags to train a VFNet model to detect lesions and tag them. We address the class imbalance by conducting three experiments: 1) Balancing data by the body part labels, 2) Balancing data by the number of lesions per patient, and 3) Balancing data by the lesion size. In contrast to a randomly sampled (unbalanced) data subset, our results indicated that balancing the body part labels always increased sensitivity for lesions >= 1cm for classes with low data quantities (Bone: 80\% vs. 46\%, Kidney: 77\% vs. 61\%, Soft Tissue: 70\% vs. 60\%, Pelvis: 83\% vs. 76\%). Similar trends were seen for three other models tested (FasterRCNN, RetinaNet, FoveaBox). Balancing data by lesion size also helped the VFNet model improve recalls for all classes in contrast to an unbalanced dataset. We also provide a structured reporting guideline for a ``Lesions'' subsection to be entered into the ``Findings'' section of a radiology report. To our knowledge, we are the first to report the class imbalance in DeepLesion, and have taken data-driven steps to address it in the context of joint lesion detection and tagging. 

**Abstract (ZH)**: 放射ologist在CT中常规检测和评估病灶以分期癌症和评估肿瘤负荷。为了潜在地辅助他们的工作，开发了多种病灶检测算法，并使用了一个名为DeepLesion的大公开数据集（32,735个病灶，32,120个CT切片，10,594项研究，4,427名患者，8个解剖部位标签）。然而，该数据集包含缺失的测量值和病灶标签，并在病灶数量方面表现出严重的类别不平衡。本文利用一个有限子集的DeepLesion数据集（6%，1,331个病灶，1,309个切片，包含病灶标注和解剖部位标签），训练VFNet模型进行病灶检测和标签标注。我们通过三种实验解决类别不平衡问题：1）按解剖部位标签平衡数据，2）按患者病灶数量平衡数据，3）按病灶大小平衡数据。与随机采样的（不平衡的）数据子集相比，我们的结果显示，按解剖部位标签平衡数据总是提高了对病灶≥1cm的灵敏度（骨骼：80% vs. 46%，肾脏：77% vs. 61%，软组织：70% vs. 60%，骨盆：83% vs. 76%）。其他三种模型（FasterRCNN、RetinaNet、FoveaBox）也观察到类似的趋势。此外，按病灶大小平衡数据也有助于提高交回模型在所有类别的召回率。我们还提供了一份在放射学报告中“病灶”部分结构化报告指南。据我们所知，我们是第一个报告DeepLesion类别不平衡的研究，并且采取了数据驱动的方法来解决这一问题，以联合病灶检测和标签标注的背景中。 

---
# Multi-fidelity Reinforcement Learning Control for Complex Dynamical Systems 

**Title (ZH)**: 多保真强化学习控制复杂动力学系统 

**Authors**: Luning Sun, Xin-Yang Liu, Siyan Zhao, Aditya Grover, Jian-Xun Wang, Jayaraman J. Thiagarajan  

**Link**: [PDF](https://arxiv.org/pdf/2504.05588)  

**Abstract**: Controlling instabilities in complex dynamical systems is challenging in scientific and engineering applications. Deep reinforcement learning (DRL) has seen promising results for applications in different scientific applications. The many-query nature of control tasks requires multiple interactions with real environments of the underlying physics. However, it is usually sparse to collect from the experiments or expensive to simulate for complex dynamics. Alternatively, controlling surrogate modeling could mitigate the computational cost issue. However, a fast and accurate learning-based model by offline training makes it very hard to get accurate pointwise dynamics when the dynamics are chaotic. To bridge this gap, the current work proposes a multi-fidelity reinforcement learning (MFRL) framework that leverages differentiable hybrid models for control tasks, where a physics-based hybrid model is corrected by limited high-fidelity data. We also proposed a spectrum-based reward function for RL learning. The effect of the proposed framework is demonstrated on two complex dynamics in physics. The statistics of the MFRL control result match that computed from many-query evaluations of the high-fidelity environments and outperform other SOTA baselines. 

**Abstract (ZH)**: 复杂动力学系统中控制稳定性的问题在科学和工程应用中具有挑战性。深度强化学习（DRL）在不同科学应用中的前景令人鼓舞。由于控制任务的多查询性质，需要与基础物理的实际情况进行多次互动。然而，从实验中收集数据通常是稀疏的，对于复杂动力学而言，模拟则通常非常昂贵。作为替代方案，代理模型控制可以缓解计算成本问题。然而，通过离线训练获得快速且准确的学习模型使得在动力学混沌时难以获得准确的点wise动力学。为解决这一问题，当前工作提出了一种多保真度强化学习（MFRL）框架，利用可微混合模型进行控制任务，其中基于物理的混合模型通过有限的高保真数据进行修正。我们还提出了基于频谱的奖励函数以供RL学习。所提出的框架在物理学中的两个复杂动力学问题上得到了验证，其统计结果与高保真环境的多查询评估结果相符，并优于其他最先进的基准方法。 

---
# Finding Fantastic Experts in MoEs: A Unified Study for Expert Dropping Strategies and Observations 

**Title (ZH)**: MoEs中寻找卓越专家：专家丢弃策略与观察的统一研究 

**Authors**: Ajay Jaiswal, Jianyu Wang, Yixiao Li, Pingzhi Li, Tianlong Chen, Zhangyang Wang, Chong Wang, Ruoming Pang, Xianzhi Du  

**Link**: [PDF](https://arxiv.org/pdf/2504.05586)  

**Abstract**: Sparsely activated Mixture-of-Experts (SMoE) has shown promise in scaling up the learning capacity of neural networks. However, vanilla SMoEs have issues such as expert redundancy and heavy memory requirements, making them inefficient and non-scalable, especially for resource-constrained scenarios. Expert-level sparsification of SMoEs involves pruning the least important experts to address these limitations. In this work, we aim to address three questions: (1) What is the best recipe to identify the least knowledgeable subset of experts that can be dropped with minimal impact on performance? (2) How should we perform expert dropping (one-shot or iterative), and what correction measures can we undertake to minimize its drastic impact on SMoE subnetwork capabilities? (3) What capabilities of full-SMoEs are severely impacted by the removal of the least dominant experts, and how can we recover them? Firstly, we propose MoE Experts Compression Suite (MC-Suite), which is a collection of some previously explored and multiple novel recipes to provide a comprehensive benchmark for estimating expert importance from diverse perspectives, as well as unveil numerous valuable insights for SMoE experts. Secondly, unlike prior works with a one-shot expert pruning approach, we explore the benefits of iterative pruning with the re-estimation of the MC-Suite criterion. Moreover, we introduce the benefits of task-agnostic fine-tuning as a correction mechanism during iterative expert dropping, which we term MoE Lottery Subnetworks. Lastly, we present an experimentally validated conjecture that, during expert dropping, SMoEs' instruction-following capabilities are predominantly hurt, which can be restored to a robust level subject to external augmentation of instruction-following capabilities using k-shot examples and supervised fine-tuning. 

**Abstract (ZH)**: 稀疏激活专家混合模型（SMoE）在扩展神经网络的learning容量方面展现了潜力。然而，vanilla SMoE存在专家冗余和高内存需求等问题，导致其效率低下且不具扩展性，特别是在资源受限的场景中更为明显。通过剪枝最不重要的专家，专家级SMoE的稀疏化旨在解决这些问题。本文旨在回答三个问题：（1）什么是识别对性能影响最小的最不重要的专家的最佳方法？（2）我们应该采用一次性还是迭代方式剪枝专家，以及在迭代剪枝过程中可以采取哪些措施来最小化对SMoE子网络能力的严重影响？（3）删除最不占主导地位的专家会对全SMoE的哪些能力产生严重影响，我们如何恢复这些能力？首先，我们提出MoE专家压缩套件（MC-Suite），这是一个包含一些先前探索和多种新方法的集合，用于从多角度提供专家重要性的全面基准，并揭示大量关于SMoE专家的有价值见解。其次，与以往采用一次性专家剪枝的方法不同，我们探索了重新估计MC-Suite标准的迭代剪枝优势。此外，我们引入了任务无关的微调作为一种迭代专家剪枝过程中的校正机制，称之为MoE彩票子网络。最后，我们提出一个经实验验证的猜测：在专家剪枝过程中，SMoE的指令遵循能力受到主要损害，但可以通过使用k-shot示例和监督微调的方式对外部增加指令遵循能力来恢复到稳健的水平。 

---
# TW-CRL: Time-Weighted Contrastive Reward Learning for Efficient Inverse Reinforcement Learning 

**Title (ZH)**: 时间加权对比奖励学习：高效逆强化学习 

**Authors**: Yuxuan Li, Ning Yang, Stephen Xia  

**Link**: [PDF](https://arxiv.org/pdf/2504.05585)  

**Abstract**: Episodic tasks in Reinforcement Learning (RL) often pose challenges due to sparse reward signals and high-dimensional state spaces, which hinder efficient learning. Additionally, these tasks often feature hidden "trap states" -- irreversible failures that prevent task completion but do not provide explicit negative rewards to guide agents away from repeated errors. To address these issues, we propose Time-Weighted Contrastive Reward Learning (TW-CRL), an Inverse Reinforcement Learning (IRL) framework that leverages both successful and failed demonstrations. By incorporating temporal information, TW-CRL learns a dense reward function that identifies critical states associated with success or failure. This approach not only enables agents to avoid trap states but also encourages meaningful exploration beyond simple imitation of expert trajectories. Empirical evaluations on navigation tasks and robotic manipulation benchmarks demonstrate that TW-CRL surpasses state-of-the-art methods, achieving improved efficiency and robustness. 

**Abstract (ZH)**: 时间加权对比奖励学习（TW-CRL）：一种集成成功与失败示例的逆强化学习框架 

---
# SoundVista: Novel-View Ambient Sound Synthesis via Visual-Acoustic Binding 

**Title (ZH)**: SoundVista: 基于视听结合的新型视角环境声合成 

**Authors**: Mingfei Chen, Israel D. Gebru, Ishwarya Ananthabhotla, Christian Richardt, Dejan Markovic, Jake Sandakly, Steven Krenn, Todd Keebler, Eli Shlizerman, Alexander Richard  

**Link**: [PDF](https://arxiv.org/pdf/2504.05576)  

**Abstract**: We introduce SoundVista, a method to generate the ambient sound of an arbitrary scene at novel viewpoints. Given a pre-acquired recording of the scene from sparsely distributed microphones, SoundVista can synthesize the sound of that scene from an unseen target viewpoint. The method learns the underlying acoustic transfer function that relates the signals acquired at the distributed microphones to the signal at the target viewpoint, using a limited number of known recordings. Unlike existing works, our method does not require constraints or prior knowledge of sound source details. Moreover, our method efficiently adapts to diverse room layouts, reference microphone configurations and unseen environments. To enable this, we introduce a visual-acoustic binding module that learns visual embeddings linked with local acoustic properties from panoramic RGB and depth data. We first leverage these embeddings to optimize the placement of reference microphones in any given scene. During synthesis, we leverage multiple embeddings extracted from reference locations to get adaptive weights for their contribution, conditioned on target viewpoint. We benchmark the task on both publicly available data and real-world settings. We demonstrate significant improvements over existing methods. 

**Abstract (ZH)**: SoundVista：生成任意场景在新型视角下的环境声音的方法 

---
# MicroNN: An On-device Disk-resident Updatable Vector Database 

**Title (ZH)**: MicroNN: 一种设备本地磁盘驻留可更新向量数据库 

**Authors**: Jeffrey Pound, Floris Chabert, Arjun Bhushan, Ankur Goswami, Anil Pacaci, Shihabur Rahman Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2504.05573)  

**Abstract**: Nearest neighbour search over dense vector collections has important applications in information retrieval, retrieval augmented generation (RAG), and content ranking. Performing efficient search over large vector collections is a well studied problem with many existing approaches and open source implementations. However, most state-of-the-art systems are generally targeted towards scenarios using large servers with an abundance of memory, static vector collections that are not updatable, and nearest neighbour search in isolation of other search criteria. We present Micro Nearest Neighbour (MicroNN), an embedded nearest-neighbour vector search engine designed for scalable similarity search in low-resource environments. MicroNN addresses the problem of on-device vector search for real-world workloads containing updates and hybrid search queries that combine nearest neighbour search with structured attribute filters. In this scenario, memory is highly constrained and disk-efficient index structures and algorithms are required, as well as support for continuous inserts and deletes. MicroNN is an embeddable library that can scale to large vector collections with minimal resources. MicroNN is used in production and powers a wide range of vector search use-cases on-device. MicroNN takes less than 7 ms to retrieve the top-100 nearest neighbours with 90% recall on publicly available million-scale vector benchmark while using ~10 MB of memory. 

**Abstract (ZH)**: 密集向量集合的最近邻搜索在信息检索、检索增强生成（RAG）和内容排名中具有重要应用。针对低资源环境下的可扩展相似性搜索，我们提出了嵌入式最近邻向量搜索引擎Micro Nearest Neighbour (MicroNN)。MicroNN 解决了包含更新和结合最近邻搜索与结构化属性过滤器的混合搜索查询的设备端向量搜索问题。在这一场景中，内存高度受限，需要高效的磁盘索引结构和算法，以及连续插入和删除的支持。MicroNN 是一个可嵌入的库，能够在最小资源消耗的情况下扩展到大规模向量集合。MicroNN 在生产环境中使用，并支持多种向量搜索用例。在使用约 10 MB 内存的情况下，MicroNN 能在公开的百万规模向量基准上以小于 7 ms 的时间检索 90% 召回率的 top-100 最近邻。 

---
# Knowledge-Instruct: Effective Continual Pre-training from Limited Data using Instructions 

**Title (ZH)**: Knowledge-Instruct：使用指令从有限数据中进行有效的持续预训练 

**Authors**: Oded Ovadia, Meni Brief, Rachel Lemberg, Eitam Sheetrit  

**Link**: [PDF](https://arxiv.org/pdf/2504.05571)  

**Abstract**: While Large Language Models (LLMs) acquire vast knowledge during pre-training, they often lack domain-specific, new, or niche information. Continual pre-training (CPT) attempts to address this gap but suffers from catastrophic forgetting and inefficiencies in low-data regimes. We introduce Knowledge-Instruct, a novel approach to efficiently inject knowledge from limited corpora through pure instruction-tuning. By generating information-dense synthetic instruction data, it effectively integrates new knowledge while preserving general reasoning and instruction-following abilities. Knowledge-Instruct demonstrates superior factual memorization, minimizes catastrophic forgetting, and remains scalable by leveraging synthetic data from relatively small language models. Additionally, it enhances contextual understanding, including complex multi-hop reasoning, facilitating integration with retrieval systems. We validate its effectiveness across diverse benchmarks, including Companies, a new dataset that we release to measure knowledge injection capabilities. 

**Abstract (ZH)**: Knowledge-Instruct：通过纯粹的指令调优高效注入有限语料库中的知识 

---
# Path Database Guidance for Motion Planning 

**Title (ZH)**: 路径数据库指导的运动规划 

**Authors**: Amnon Attali, Praval Telagi, Marco Morales, Nancy M. Amato  

**Link**: [PDF](https://arxiv.org/pdf/2504.05550)  

**Abstract**: One approach to using prior experience in robot motion planning is to store solutions to previously seen problems in a database of paths. Methods that use such databases are characterized by how they query for a path and how they use queries given a new problem. In this work we present a new method, Path Database Guidance (PDG), which innovates on existing work in two ways. First, we use the database to compute a heuristic for determining which nodes of a search tree to expand, in contrast to prior work which generally pastes the (possibly transformed) queried path or uses it to bias a sampling distribution. We demonstrate that this makes our method more easily composable with other search methods by dynamically interleaving exploration according to a baseline algorithm with exploitation of the database guidance. Second, in contrast to other methods that treat the database as a single fixed prior, our database (and thus our queried heuristic) updates as we search the implicitly defined robot configuration space. We experimentally demonstrate the effectiveness of PDG in a variety of explicitly defined environment distributions in simulation. 

**Abstract (ZH)**: 一种在机器人运动规划中利用先验经验的方法是将以往解决的问题路径存储在一个路径数据库中。本研究提出了一种名为路径数据库引导（PDG）的新方法，该方法在两方面创新了现有工作。首先，我们利用数据库来计算一个启发式函数，用于确定搜索树中需要扩展的节点，与以往工作不同，以往工作通常会直接应用（可能经过变换的）查询路径或将其用于偏倚采样分布。我们通过动态交替进行基线算法的探索和数据库引导的利用，证明这种方法更容易与其他搜索方法组合。其次，与其他将数据库视为单一固定先验的方法不同，我们的数据库（以及因此我们的查询启发式）在搜索隐式定义的机器人配置空间时会进行更新。我们在仿真中对PDG的有效性进行了实验性验证。 

---
# Towards Efficient Real-Time Video Motion Transfer via Generative Time Series Modeling 

**Title (ZH)**: 基于生成时间序列建模的高效实时视频运动转移 

**Authors**: Tasmiah Haque, Md. Asif Bin Syed, Byungheon Jeong, Xue Bai, Sumit Mohan, Somdyuti Paul, Imtiaz Ahmed, Srinjoy Das  

**Link**: [PDF](https://arxiv.org/pdf/2504.05537)  

**Abstract**: We propose a deep learning framework designed to significantly optimize bandwidth for motion-transfer-enabled video applications, including video conferencing, virtual reality interactions, health monitoring systems, and vision-based real-time anomaly detection. To capture complex motion effectively, we utilize the First Order Motion Model (FOMM), which encodes dynamic objects by detecting keypoints and their associated local affine transformations. These keypoints are identified using a self-supervised keypoint detector and arranged into a time series corresponding to the successive frames. Forecasting is performed on these keypoints by integrating two advanced generative time series models into the motion transfer pipeline, namely the Variational Recurrent Neural Network (VRNN) and the Gated Recurrent Unit with Normalizing Flow (GRU-NF). The predicted keypoints are subsequently synthesized into realistic video frames using an optical flow estimator paired with a generator network, thereby facilitating accurate video forecasting and enabling efficient, low-frame-rate video transmission. We validate our results across three datasets for video animation and reconstruction using the following metrics: Mean Absolute Error, Joint Embedding Predictive Architecture Embedding Distance, Structural Similarity Index, and Average Pair-wise Displacement. Our results confirm that by utilizing the superior reconstruction property of the Variational Autoencoder, the VRNN integrated FOMM excels in applications involving multi-step ahead forecasts such as video conferencing. On the other hand, by leveraging the Normalizing Flow architecture for exact likelihood estimation, and enabling efficient latent space sampling, the GRU-NF based FOMM exhibits superior capabilities for producing diverse future samples while maintaining high visual quality for tasks like real-time video-based anomaly detection. 

**Abstract (ZH)**: 我们提出了一种深度学习框架，旨在显著优化支持运动转移的视频应用中的带宽，包括视频会议、虚拟现实交互、健康监测系统和基于视觉的实时异常检测。该框架利用First Order Motion Model (FOMM) 捕捉复杂运动，通过检测关键点及其相关的局部仿射变换来编码动态对象。关键点通过自我监督的关键点检测器识别，并按顺序帧对应的时间序列进行排列。运动转移管道中整合了两种先进的生成时间序列模型——变分递归神经网络 (VRNN) 和带有归一化流的门控递归单元 (GRU-NF)——以预测关键点。预测的关键点随后通过光学流估计器与生成器网络相结合，合成出逼真的视频帧，从而实现准确的视频预测和高效的低帧率视频传输。我们使用以下指标在三个视频动画和重建数据集上验证了我们的结果：绝对平均误差、联合嵌入预测架构嵌入距离、结构相似性指数和平均成对位移。我们的结果证实，通过利用变分自编码器的优越重建特性，FOMM与VRNN整合在涉及多步预测的应用场景（如视频会议）中表现出色。另一方面，通过利用归一化流架构进行精确似然估计，并允许高效的潜在空间采样，基于GRU-NF的FOMM在生成多样化未来样本的同时保持高质量视觉效果，这在基于视频的实时异常检测任务中表现出色。 

---
# FORCE: Feature-Oriented Representation with Clustering and Explanation 

**Title (ZH)**: FORCE:面向特征的表示与聚类解释 

**Authors**: Rishav Mukherjee, Jeffrey Ahearn Thompson  

**Link**: [PDF](https://arxiv.org/pdf/2504.05530)  

**Abstract**: Learning about underlying patterns in data using latent unobserved structures to improve the accuracy of predictive models has become an active avenue of deep learning research. Most approaches cluster the original features to capture certain latent structures. However, the information gained in the process can often be implicitly derived by sufficiently complex models. Thus, such approaches often provide minimal benefits. We propose a SHAP (Shapley Additive exPlanations) based supervised deep learning framework FORCE which relies on two-stage usage of SHAP values in the neural network architecture, (i) an additional latent feature to guide model training, based on clustering SHAP values, and (ii) initiating an attention mechanism within the architecture using latent information. This approach gives a neural network an indication about the effect of unobserved values that modify feature importance for an observation. The proposed framework is evaluated on three real life datasets. Our results demonstrate that FORCE led to dramatic improvements in overall performance as compared to networks that did not incorporate the latent feature and attention framework (e.g., F1 score for presence of heart disease 0.80 vs 0.72). Using cluster assignments and attention based on SHAP values guides deep learning, enhancing latent pattern learning and overall discriminative capability. 

**Abstract (ZH)**: 使用SHAP值基于监督的深度学习框架FORCE挖掘潜在模式以提高预测模型的准确性 

---
# Bridging Industrial Expertise and XR with LLM-Powered Conversational Agents 

**Title (ZH)**: 工业 expertise 与 XR 的桥梁：基于 LLM 的对话代理 

**Authors**: Despina Tomkou, George Fatouros, Andreas Andreou, Georgios Makridis, Fotis Liarokapis, Dimitrios Dardanis, Athanasios Kiourtis, John Soldatos, Dimosthenis Kyriazis  

**Link**: [PDF](https://arxiv.org/pdf/2504.05527)  

**Abstract**: This paper introduces a novel integration of Retrieval-Augmented Generation (RAG) enhanced Large Language Models (LLMs) with Extended Reality (XR) technologies to address knowledge transfer challenges in industrial environments. The proposed system embeds domain-specific industrial knowledge into XR environments through a natural language interface, enabling hands-free, context-aware expert guidance for workers. We present the architecture of the proposed system consisting of an LLM Chat Engine with dynamic tool orchestration and an XR application featuring voice-driven interaction. Performance evaluation of various chunking strategies, embedding models, and vector databases reveals that semantic chunking, balanced embedding models, and efficient vector stores deliver optimal performance for industrial knowledge retrieval. The system's potential is demonstrated through early implementation in multiple industrial use cases, including robotic assembly, smart infrastructure maintenance, and aerospace component servicing. Results indicate potential for enhancing training efficiency, remote assistance capabilities, and operational guidance in alignment with Industry 5.0's human-centric and resilient approach to industrial development. 

**Abstract (ZH)**: 本文介绍了一种将增强型大型语言模型（LLMs）与检索增强生成（RAG）技术结合拓展现实（XR）技术的方法，以解决工业环境中的知识转移挑战。提出的系统通过自然语言接口将特定领域的工业知识嵌入XR环境中，为工人提供免手持、情境感知的专业指导。本文介绍了一种LLM聊天引擎架构，其中包含动态工具编排，并展示了一款以其为基础的具备语音驱动交互功能的XR应用程序。各种分块策略、嵌入模型和向量数据库的性能评估表明，语义分块、均衡嵌入模型和高效向量存储为工业知识检索提供了最优性能。通过在多种工业应用场景中的早期实施，包括机器人装配、智能基础设施维护和航空部件服务，展示了该系统的潜力。结果表明，该系统有助于提高培训效率、远程协助能力，并与工业5.0以人类为中心、具备弹性的工业发展方法保持一致。 

---
# Deep Reinforcement Learning Algorithms for Option Hedging 

**Title (ZH)**: 深度强化学习算法在期权对冲中的应用 

**Authors**: Andrei Neagu, Frédéric Godin, Leila Kosseim  

**Link**: [PDF](https://arxiv.org/pdf/2504.05521)  

**Abstract**: Dynamic hedging is a financial strategy that consists in periodically transacting one or multiple financial assets to offset the risk associated with a correlated liability. Deep Reinforcement Learning (DRL) algorithms have been used to find optimal solutions to dynamic hedging problems by framing them as sequential decision-making problems. However, most previous work assesses the performance of only one or two DRL algorithms, making an objective comparison across algorithms difficult. In this paper, we compare the performance of eight DRL algorithms in the context of dynamic hedging; Monte Carlo Policy Gradient (MCPG), Proximal Policy Optimization (PPO), along with four variants of Deep Q-Learning (DQL) and two variants of Deep Deterministic Policy Gradient (DDPG). Two of these variants represent a novel application to the task of dynamic hedging. In our experiments, we use the Black-Scholes delta hedge as a baseline and simulate the dataset using a GJR-GARCH(1,1) model. Results show that MCPG, followed by PPO, obtain the best performance in terms of the root semi-quadratic penalty. Moreover, MCPG is the only algorithm to outperform the Black-Scholes delta hedge baseline with the allotted computational budget, possibly due to the sparsity of rewards in our environment. 

**Abstract (ZH)**: 动态对冲是一种金融策略，涉及周期性交易一个或多个金融资产以抵消与相关负债相关的风险。深度强化学习（DRL）算法已被用于通过将其表述为顺序决策问题来寻找动态对冲问题的最优解。然而，大部分先前的工作仅评估了一两种DRL算法的表现，这使得算法之间的客观比较变得困难。在这篇论文中，我们比较了八种DRL算法在动态对冲中的表现；蒙特卡洛策略梯度（MCPG）、 proportional策略优化（PPO），以及四种深度Q学习（DQL）和两种深度确定性策略梯度（DDPG）的变体。在这两种变体中，其中一种是首次应用于动态对冲任务。在我们的实验中，我们使用布莱克-斯科尔斯Delta对冲作为基线，并使用GJR-GARCH(1,1)模型模拟数据集。结果表明，MCPG之后是PPO，在根半四次惩罚方面获得最佳表现。此外，MCPG是唯一一种在给定计算预算内表现优于布莱克-斯科尔斯Delta对冲基线的算法，这可能是由于我们环境中奖励的稀疏性。 

---
# Large-Scale Classification of Shortwave Communication Signals with Machine Learning 

**Title (ZH)**: 基于机器学习的短波通信信号大规模分类 

**Authors**: Stefan Scholl  

**Link**: [PDF](https://arxiv.org/pdf/2504.05455)  

**Abstract**: This paper presents a deep learning approach to the classification of 160 shortwave radio signals. It addresses the typical challenges of the shortwave spectrum, which are the large number of different signal types, the presence of various analog modulations and ionospheric propagation. As a classifier a deep convolutional neural network is used, that is trained to recognize 160 typical shortwave signal classes. The approach is blind and therefore does not require preknowledge or special preprocessing of the signal and no manual design of discriminative features for each signal class. The network is trained on a large number of synthetically generated signals and high quality recordings. Finally, the network is evaluated on real-world radio signals obtained from globally deployed receiver hardware and achieves up to 90% accuracy for an observation time of only 1 second. 

**Abstract (ZH)**: 一种用于160个短波无线电信号分类的深度学习方法 

---
# GraphPINE: Graph Importance Propagation for Interpretable Drug Response Prediction 

**Title (ZH)**: GraphPINE: 图结构的重要性传播药物响应可解释预测 

**Authors**: Yoshitaka Inoue, Tianfan Fu, Augustin Luna  

**Link**: [PDF](https://arxiv.org/pdf/2504.05454)  

**Abstract**: Explainability is necessary for many tasks in biomedical research. Recent explainability methods have focused on attention, gradient, and Shapley value. These do not handle data with strong associated prior knowledge and fail to constrain explainability results based on known relationships between predictive features.
We propose GraphPINE, a graph neural network (GNN) architecture leveraging domain-specific prior knowledge to initialize node importance optimized during training for drug response prediction. Typically, a manual post-prediction step examines literature (i.e., prior knowledge) to understand returned predictive features. While node importance can be obtained for gradient and attention after prediction, node importance from these methods lacks complementary prior knowledge; GraphPINE seeks to overcome this limitation. GraphPINE differs from other GNN gating methods by utilizing an LSTM-like sequential format. We introduce an importance propagation layer that unifies 1) updates for feature matrix and node importance and 2) uses GNN-based graph propagation of feature values. This initialization and updating mechanism allows for informed feature learning and improved graph representation.
We apply GraphPINE to cancer drug response prediction using drug screening and gene data collected for over 5,000 gene nodes included in a gene-gene graph with a drug-target interaction (DTI) graph for initial importance. The gene-gene graph and DTIs were obtained from curated sources and weighted by article count discussing relationships between drugs and genes. GraphPINE achieves a PR-AUC of 0.894 and ROC-AUC of 0.796 across 952 drugs. Code is available at this https URL. 

**Abstract (ZH)**: GraphPINE：利用领域特定先验知识的图神经网络架构用于药物响应预测 

---
# A Behavior-Based Knowledge Representation Improves Prediction of Players' Moves in Chess by 25% 

**Title (ZH)**: 行为为基础的知识表示在棋盘游戏中提高玩家棋局预测准确率25% 

**Authors**: Benny Skidanov, Daniel Erbesfeld, Gera Weiss, Achiya Elyasaf  

**Link**: [PDF](https://arxiv.org/pdf/2504.05425)  

**Abstract**: Predicting player behavior in strategic games, especially complex ones like chess, presents a significant challenge. The difficulty arises from several factors. First, the sheer number of potential outcomes stemming from even a single position, starting from the initial setup, makes forecasting a player's next move incredibly complex. Second, and perhaps even more challenging, is the inherent unpredictability of human behavior. Unlike the optimized play of engines, humans introduce a layer of variability due to differing playing styles and decision-making processes. Each player approaches the game with a unique blend of strategic thinking, tactical awareness, and psychological tendencies, leading to diverse and often unexpected actions. This stylistic variation, combined with the capacity for creativity and even irrational moves, makes predicting human play difficult. Chess, a longstanding benchmark of artificial intelligence research, has seen significant advancements in tools and automation. Engines like Deep Blue, AlphaZero, and Stockfish can defeat even the most skilled human players. However, despite their exceptional ability to outplay top-level grandmasters, predicting the moves of non-grandmaster players, who comprise most of the global chess community -- remains complicated for these engines. This paper proposes a novel approach combining expert knowledge with machine learning techniques to predict human players' next moves. By applying feature engineering grounded in domain expertise, we seek to uncover the patterns in the moves of intermediate-level chess players, particularly during the opening phase of the game. Our methodology offers a promising framework for anticipating human behavior, advancing both the fields of AI and human-computer interaction. 

**Abstract (ZH)**: 基于专家知识与机器学习技术预测棋手在战略游戏中，尤其是象棋中，下一步行为 

---
# Safe Automated Refactoring for Efficient Migration of Imperative Deep Learning Programs to Graph Execution 

**Title (ZH)**: 安全自动重构以提高 imperative 深度学习程序向图执行迁移的效率 

**Authors**: Raffi Khatchadourian, Tatiana Castro Vélez, Mehdi Bagherzadeh, Nan Jia, Anita Raja  

**Link**: [PDF](https://arxiv.org/pdf/2504.05424)  

**Abstract**: Efficiency is essential to support responsiveness w.r.t. ever-growing datasets, especially for Deep Learning (DL) systems. DL frameworks have traditionally embraced deferred execution-style DL code -- supporting symbolic, graph-based Deep Neural Network (DNN) computation. While scalable, such development is error-prone, non-intuitive, and difficult to debug. Consequently, more natural, imperative DL frameworks encouraging eager execution have emerged at the expense of run-time performance. Though hybrid approaches aim for the "best of both worlds," using them effectively requires subtle considerations to make code amenable to safe, accurate, and efficient graph execution. We present an automated refactoring approach that assists developers in specifying whether their otherwise eagerly-executed imperative DL code could be reliably and efficiently executed as graphs while preserving semantics. The approach, based on a novel imperative tensor analysis, automatically determines when it is safe and potentially advantageous to migrate imperative DL code to graph execution. The approach is implemented as a PyDev Eclipse IDE plug-in that integrates the WALA Ariadne analysis framework and evaluated on 19 Python projects consisting of 132.05 KLOC. We found that 326 of 766 candidate functions (42.56%) were refactorable, and an average speedup of 2.16 on performance tests was observed. The results indicate that the approach is useful in optimizing imperative DL code to its full potential. 

**Abstract (ZH)**: 效率对于支持不断增长的数据集的响应性至关重要，特别是在深度学习系统中。传统的深度学习框架采用了延迟执行风格的代码，支持基于符号和图形的深层神经网络（DNN）计算。尽管这种开发具有可扩展性，但它容易出错、不直观且难以调试。因此，新兴的鼓励即时执行的自然化指令式深度学习框架以牺牲运行时性能为代价出现。虽然混合方法旨在兼得两者之所长，但有效地使用它们需要细致的考虑，以使代码适合安全、准确且高效的图执行。我们提出了一种自动化重构方法，帮助开发者指定其否则即时执行的指令式深度学习代码是否可以可靠且高效地作为图形执行，同时保留语义。该方法基于一种新颖的指令式张量分析，自动确定何时将指令式深度学习代码迁移到图形执行既安全又有利。该方法以PyDev Eclipse IDE插件的形式实现，结合使用WALA Ariadne分析框架，并在包含132,050行代码的19个Python项目上进行了评估。结果显示，766个候选函数中有326个（42.56%）可以重构，并且性能测试观察到了平均2.16倍的加速。结果表明，该方法有助于充分利用指令式深度学习代码的潜力。 

---
# PreSumm: Predicting Summarization Performance Without Summarizing 

**Title (ZH)**: 预总结：无需总结预测摘要性能 

**Authors**: Steven Koniaev, Ori Ernst, Jackie Chi Kit Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2504.05420)  

**Abstract**: Despite recent advancements in automatic summarization, state-of-the-art models do not summarize all documents equally well, raising the question: why? While prior research has extensively analyzed summarization models, little attention has been given to the role of document characteristics in influencing summarization performance. In this work, we explore two key research questions. First, do documents exhibit consistent summarization quality across multiple systems? If so, can we predict a document's summarization performance without generating a summary? We answer both questions affirmatively and introduce PreSumm, a novel task in which a system predicts summarization performance based solely on the source document. Our analysis sheds light on common properties of documents with low PreSumm scores, revealing that they often suffer from coherence issues, complex content, or a lack of a clear main theme. In addition, we demonstrate PreSumm's practical utility in two key applications: improving hybrid summarization workflows by identifying documents that require manual summarization and enhancing dataset quality by filtering outliers and noisy documents. Overall, our findings highlight the critical role of document properties in summarization performance and offer insights into the limitations of current systems that could serve as the basis for future improvements. 

**Abstract (ZH)**: 尽管自动摘要领域取得了 recent advancements，最先进的模型在摘要不同文档的效果上并不一致，这引发了一个问题：原因何在？尽管先前的研究广泛分析了摘要模型，但很少有研究关注文档特性如何影响摘要性能。在本文中，我们探讨了两个关键的研究问题。首先，文档在多个系统中的摘要质量是否具有一致性？如果一致，我们能否在生成摘要之前预测文档的摘要性能？我们对这两个问题的答案均为肯定，并引入了一个新颖的任务 PreSumm：系统仅基于源文档预测摘要性能。我们的分析揭示了 PreSumm 得分较低的文档的共同特性，表明这些文档通常存在连贯性问题、复杂内容或缺乏清晰的主题。此外，我们展示了 PreSumm 在两个关键应用中的实际价值：通过识别需要人工摘要的文档以改进混合摘要流程，并通过过滤异常值和噪声文档来提高数据集质量。总体而言，我们的发现强调了文档特性在摘要性能中的关键作用，并提供了有关当前系统限制的见解，这些见解可以作为未来改进的基础。 

---
# Fast Controlled Generation from Language Models with Adaptive Weighted Rejection Sampling 

**Title (ZH)**: 基于自适应加权拒绝采样的语言模型快速受控生成 

**Authors**: Benjamin Lipkin, Benjamin LeBrun, Jacob Hoover Vigly, João Loula, David R. MacIver, Li Du, Jason Eisner, Ryan Cotterell, Vikash Mansinghka, Timothy J. O'Donnell, Alexander K. Lew, Tim Vieira  

**Link**: [PDF](https://arxiv.org/pdf/2504.05410)  

**Abstract**: The dominant approach to generating from language models subject to some constraint is locally constrained decoding (LCD), incrementally sampling tokens at each time step such that the constraint is never violated. Typically, this is achieved through token masking: looping over the vocabulary and excluding non-conforming tokens. There are two important problems with this approach. (i) Evaluating the constraint on every token can be prohibitively expensive -- LM vocabularies often exceed $100,000$ tokens. (ii) LCD can distort the global distribution over strings, sampling tokens based only on local information, even if they lead down dead-end paths. This work introduces a new algorithm that addresses both these problems. First, to avoid evaluating a constraint on the full vocabulary at each step of generation, we propose an adaptive rejection sampling algorithm that typically requires orders of magnitude fewer constraint evaluations. Second, we show how this algorithm can be extended to produce low-variance, unbiased estimates of importance weights at a very small additional cost -- estimates that can be soundly used within previously proposed sequential Monte Carlo algorithms to correct for the myopic behavior of local constraint enforcement. Through extensive empirical evaluation in text-to-SQL, molecular synthesis, goal inference, pattern matching, and JSON domains, we show that our approach is superior to state-of-the-art baselines, supporting a broader class of constraints and improving both runtime and performance. Additional theoretical and empirical analyses show that our method's runtime efficiency is driven by its dynamic use of computation, scaling with the divergence between the unconstrained and constrained LM, and as a consequence, runtime improvements are greater for better models. 

**Abstract (ZH)**: 生成受约束的语言模型输出的新算法：解决局部约束解码的关键问题 

---
# SoK: Frontier AI's Impact on the Cybersecurity Landscape 

**Title (ZH)**: SoK: 前沿人工智能对网络安全 landscape 的影响 

**Authors**: Wenbo Guo, Yujin Potter, Tianneng Shi, Zhun Wang, Andy Zhang, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.05408)  

**Abstract**: As frontier AI advances rapidly, understanding its impact on cybersecurity and inherent risks is essential to ensuring safe AI evolution (e.g., guiding risk mitigation and informing policymakers). While some studies review AI applications in cybersecurity, none of them comprehensively discuss AI's future impacts or provide concrete recommendations for navigating its safe and secure usage. This paper presents an in-depth analysis of frontier AI's impact on cybersecurity and establishes a systematic framework for risk assessment and mitigation. To this end, we first define and categorize the marginal risks of frontier AI in cybersecurity and then systemically analyze the current and future impacts of frontier AI in cybersecurity, qualitatively and quantitatively. We also discuss why frontier AI likely benefits attackers more than defenders in the short term from equivalence classes, asymmetry, and economic impact. Next, we explore frontier AI's impact on future software system development, including enabling complex hybrid systems while introducing new risks. Based on our findings, we provide security recommendations, including constructing fine-grained benchmarks for risk assessment, designing AI agents for defenses, building security mechanisms and provable defenses for hybrid systems, enhancing pre-deployment security testing and transparency, and strengthening defenses for users. Finally, we present long-term research questions essential for understanding AI's future impacts and unleashing its defensive capabilities. 

**Abstract (ZH)**: 随着前沿人工智能的迅速发展，理解其对网络安全的影响和固有风险对于确保安全的人工智能演变（例如，指导风险缓解并为决策者提供信息）至关重要。虽然有一些研究回顾了人工智能在 cybersecurity 中的应用，但没有一项研究全面讨论人工智能未来的影响或提供具体建议以确保其安全和安全使用。本文深入分析了前沿人工智能对网络安全的影响，并建立了一个系统性的风险评估与缓解框架。首先，我们定义并分类了前沿人工智能在网络安全中的边际风险，然后系统地分析了前沿人工智能当前和未来对网络安全的影响，从定性和定量的角度进行分析。我们还讨论了在短期内等价类、不对称性和经济影响使得前沿人工智能更可能对攻击者而不是防御者有利。接下来，我们探索前沿人工智能对未来软件系统开发的影响，包括启用复杂混合系统的同时引入新风险。基于我们的发现，我们提供了安全建议，包括构建细粒度基准以进行风险评估、设计用于防御的人工智能代理、构建适用于混合系统的安全机制和可验证防御、增强部署前的安全测试和透明度、以及加强用户防御。最后，我们提出了对于理解人工智能未来影响以及释放其防御能力至关重要的长期研究问题。 

---
# TRATSS: Transformer-Based Task Scheduling System for Autonomous Vehicles 

**Title (ZH)**: TRATSS: 基于 Transformer 的自主车辆任务调度系统 

**Authors**: Yazan Youssef, Paulo Ricardo Marques de Araujo, Aboelmagd Noureldin, Sidney Givigi  

**Link**: [PDF](https://arxiv.org/pdf/2504.05407)  

**Abstract**: Efficient scheduling remains a critical challenge in various domains, requiring solutions to complex NP-hard optimization problems to achieve optimal resource allocation and maximize productivity. In this paper, we introduce a framework called Transformer-Based Task Scheduling System (TRATSS), designed to address the intricacies of single agent scheduling in graph-based environments. By integrating the latest advancements in reinforcement learning and transformer architecture, TRATSS provides a novel system that outputs optimized task scheduling decisions while dynamically adapting to evolving task requirements and resource availability. Leveraging the self-attention mechanism in transformers, TRATSS effectively captures complex task dependencies, thereby providing solutions with enhanced resource utilization and task completion efficiency. Experimental evaluations on benchmark datasets demonstrate TRATSS's effectiveness in providing high-quality solutions to scheduling problems that involve multiple action profiles. 

**Abstract (ZH)**: 基于Transformer的任务调度系统（TRATSS）：图环境单代理调度的高效解决方案 

---
# The Role of Environment Access in Agnostic Reinforcement Learning 

**Title (ZH)**: 环境访问在agnostic强化学习中的作用 

**Authors**: Akshay Krishnamurthy, Gene Li, Ayush Sekhari  

**Link**: [PDF](https://arxiv.org/pdf/2504.05405)  

**Abstract**: We study Reinforcement Learning (RL) in environments with large state spaces, where function approximation is required for sample-efficient learning. Departing from a long history of prior work, we consider the weakest possible form of function approximation, called agnostic policy learning, where the learner seeks to find the best policy in a given class $\Pi$, with no guarantee that $\Pi$ contains an optimal policy for the underlying task. Although it is known that sample-efficient agnostic policy learning is not possible in the standard online RL setting without further assumptions, we investigate the extent to which this can be overcome with stronger forms of access to the environment. Specifically, we show that: 1. Agnostic policy learning remains statistically intractable when given access to a local simulator, from which one can reset to any previously seen state. This result holds even when the policy class is realizable, and stands in contrast to a positive result of [MFR24] showing that value-based learning under realizability is tractable with local simulator access. 2. Agnostic policy learning remains statistically intractable when given online access to a reset distribution with good coverage properties over the state space (the so-called $\mu$-reset setting). We also study stronger forms of function approximation for policy learning, showing that PSDP [BKSN03] and CPI [KL02] provably fail in the absence of policy completeness. 3. On a positive note, agnostic policy learning is statistically tractable for Block MDPs with access to both of the above reset models. We establish this via a new algorithm that carefully constructs a policy emulator: a tabular MDP with a small state space that approximates the value functions of all policies $\pi \in \Pi$. These values are approximated without any explicit value function class. 

**Abstract (ZH)**: 我们研究具有大规模状态空间环境中的强化学习（RL），在这种情况下需要使用功能近似以实现样本高效学习。不同于以往工作的长期历史，我们考虑功能近似中最弱的形式，即无放大型策略学习，其中学习者寻求在给定策略类$\Pi$中找到最优策略，但没有保证$\Pi$中包含底层任务的最优策略。虽然在标准在线RL设置中，无需进一步假设无法在无放大型策略学习中实现样本高效学习，但我们研究了更强环境访问形式在这种限制下的克服程度。具体而言，我们展示了：1. 即使策略类是可以实现的，当具有访问局部模拟器的能力时（可以重置到已见过的任何状态），无放大型策略学习仍然统计上不可行。这一结果与[MFR24]中关于在局部模拟器访问下基于值的学习可以通过对实现性假设进行处理而变得可行的积极结果形成了对比。2. 即使具有对具有良好状态空间覆盖性质的重置分布的在线访问（所谓的$\mu$-重置设置），无放大型策略学习仍然统计上不可行。我们还研究了策略学习中的更强形式的功能近似，展示了PSDP [BKSN03]和CPI [KL02]在缺少策略完备性时会失效。3. 在一个积极的方面，当具有上述两种重置模型的访问时，无放大型策略学习对于Block MDP来说是统计上可处理的。我们通过一个新的算法建立这一结论，该算法仔细构建了一个策略模拟器：一个具有小状态空间的表格MDP，它可以近似所有策略$\pi \in \Pi$的价值函数。这些价值是在没有任何显式价值函数类的情况下近似的。 

---
# GARF: Learning Generalizable 3D Reassembly for Real-World Fractures 

**Title (ZH)**: GARF: 学习可泛化的三维复原以应对真实世界的断裂 

**Authors**: Sihang Li, Zeyu Jiang, Grace Chen, Chenyang Xu, Siqi Tan, Xue Wang, Irving Fang, Kristof Zyskowski, Shannon P. McPherron, Radu Iovita, Chen Feng, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05400)  

**Abstract**: 3D reassembly is a challenging spatial intelligence task with broad applications across scientific domains. While large-scale synthetic datasets have fueled promising learning-based approaches, their generalizability to different domains is limited. Critically, it remains uncertain whether models trained on synthetic datasets can generalize to real-world fractures where breakage patterns are more complex. To bridge this gap, we propose GARF, a generalizable 3D reassembly framework for real-world fractures. GARF leverages fracture-aware pretraining to learn fracture features from individual fragments, with flow matching enabling precise 6-DoF alignments. At inference time, we introduce one-step preassembly, improving robustness to unseen objects and varying numbers of fractures. In collaboration with archaeologists, paleoanthropologists, and ornithologists, we curate Fractura, a diverse dataset for vision and learning communities, featuring real-world fracture types across ceramics, bones, eggshells, and lithics. Comprehensive experiments have shown our approach consistently outperforms state-of-the-art methods on both synthetic and real-world datasets, achieving 82.87\% lower rotation error and 25.15\% higher part accuracy. This sheds light on training on synthetic data to advance real-world 3D puzzle solving, demonstrating its strong generalization across unseen object shapes and diverse fracture types. 

**Abstract (ZH)**: 可泛化的3D重构框架：面向真实世界裂痕的 GARF 

---
# A Nature-Inspired Colony of Artificial Intelligence System with Fast, Detailed, and Organized Learner Agents for Enhancing Diversity and Quality 

**Title (ZH)**: 受自然界启发的人工智能 colony 体系，配备快速、详细且组织化的学习代理，以增强多样性和质量 

**Authors**: Shan Suthaharan  

**Link**: [PDF](https://arxiv.org/pdf/2504.05365)  

**Abstract**: The concepts of convolutional neural networks (CNNs) and multi-agent systems are two important areas of research in artificial intelligence (AI). In this paper, we present an approach that builds a CNN-based colony of AI agents to serve as a single system and perform multiple tasks (e.g., predictions or classifications) in an environment. The proposed system impersonates the natural environment of a biological system, like an ant colony or a human colony. The proposed colony of AI that is defined as a role-based system uniquely contributes to accomplish tasks in an environment by incorporating AI agents that are fast learners, detailed learners, and organized learners. These learners can enhance their localized learning and their collective decisions as a single system of colony of AI agents. This approach also enhances the diversity and quality of the colony of AI with the help of Genetic Algorithms and their crossover and mutation mechanisms. The evolution of fast, detailed, and organized learners in the colony of AI is achieved by introducing a unique one-to-one mapping between these learners and the pretrained VGG16, VGG19, and ResNet50 models, respectively. This role-based approach creates two parent-AI agents using the AI models through the processes, called the intra- and inter-marriage of AI, so that they can share their learned knowledge (weights and biases) based on a probabilistic rule and produce diversified child-AI agents to perform new tasks. This process will form a colony of AI that consists of families of multi-model and mixture-model AI agents to improve diversity and quality. Simulations show that the colony of AI, built using the VGG16, VGG19, and ResNet50 models, can provide a single system that generates child-AI agents of excellent predictive performance, ranging between 82% and 95% of F1-scores, to make diversified collective and quality decisions on a task. 

**Abstract (ZH)**: 基于卷积神经网络的多agent系统在人工智能中的应用：一种角色化的AI蚁群模型 

---
# Of All StrIPEs: Investigating Structure-informed Positional Encoding for Efficient Music Generation 

**Title (ZH)**: 全方位探究结构导向的位置编码以实现高效的音乐生成 

**Authors**: Manvi Agarwal, Changhong Wang, Gael Richard  

**Link**: [PDF](https://arxiv.org/pdf/2504.05364)  

**Abstract**: While music remains a challenging domain for generative models like Transformers, a two-pronged approach has recently proved successful: inserting musically-relevant structural information into the positional encoding (PE) module and using kernel approximation techniques based on Random Fourier Features (RFF) to lower the computational cost from quadratic to linear. Yet, it is not clear how such RFF-based efficient PEs compare with those based on rotation matrices, such as Rotary Positional Encoding (RoPE). In this paper, we present a unified framework based on kernel methods to analyze both families of efficient PEs. We use this framework to develop a novel PE method called RoPEPool, capable of extracting causal relationships from temporal sequences. Using RFF-based PEs and rotation-based PEs, we demonstrate how seemingly disparate PEs can be jointly studied by considering the content-context interactions they induce. For empirical validation, we use a symbolic music generation task, namely, melody harmonization. We show that RoPEPool, combined with highly-informative structural priors, outperforms all methods. 

**Abstract (ZH)**: 虽然对生成模型如变换器来说音乐仍是一个具有挑战性的领域，但最近一种两步方法已被证明非常成功：将音乐相关的结构信息插入到位置编码（PE）模块中，并利用随机傅里叶特征（RFF）为基础的核近似技术将计算成本从二次降低到线性。然而，基于RFF的高效位置编码与基于旋转矩阵的方法，如旋转位置编码（RoPE），之间的性能对比尚不明确。在本文中，我们提出了一种基于核方法的统一框架来分析这两种高效的PE类型。我们利用该框架开发了一种新的PE方法RoPEPool，它可以从中提取时间序列中的因果关系。通过基于RFF的PE和基于旋转的PE，我们展示了如何通过考虑它们引起的内容-上下文交互来联合研究看似不同的PE。为了进行实证验证，我们使用了符号音乐生成任务，即旋律和声。我们表明，结合高度信息的结构先验，RoPEPool优于所有方法。 

---
# Debate-Feedback: A Multi-Agent Framework for Efficient Legal Judgment Prediction 

**Title (ZH)**: 辩论-反馈：一种高效的法律判决预测多agent框架 

**Authors**: Xi Chen, Mao Mao, Shuo Li, Haotian Shangguan  

**Link**: [PDF](https://arxiv.org/pdf/2504.05358)  

**Abstract**: The use of AI in legal analysis and prediction (LegalAI) has gained widespread attention, with past research focusing on retrieval-based methods and fine-tuning large models. However, these approaches often require large datasets and underutilize the capabilities of modern large language models (LLMs). In this paper, inspired by the debate phase of real courtroom trials, we propose a novel legal judgment prediction model based on the Debate-Feedback architecture, which integrates LLM multi-agent debate and reliability evaluation models. Unlike traditional methods, our model achieves significant improvements in efficiency by minimizing the need for large historical datasets, thus offering a lightweight yet robust solution. Comparative experiments show that it outperforms several general-purpose and domain-specific legal models, offering a dynamic reasoning process and a promising direction for future LegalAI research. 

**Abstract (ZH)**: AI在法律分析与预测中的应用（LegalAI）：基于 Debate-Feedback 架构的新型法律判决预测模型及其优势 

---
# Find A Winning Sign: Sign Is All We Need to Win the Lottery 

**Title (ZH)**: 寻找获胜的征兆：只需符号即可赢得彩票 

**Authors**: Junghun Oh, Sungyong Baik, Kyoung Mu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.05357)  

**Abstract**: The Lottery Ticket Hypothesis (LTH) posits the existence of a sparse subnetwork (a.k.a. winning ticket) that can generalize comparably to its over-parameterized counterpart when trained from scratch. The common approach to finding a winning ticket is to preserve the original strong generalization through Iterative Pruning (IP) and transfer information useful for achieving the learned generalization by applying the resulting sparse mask to an untrained network. However, existing IP methods still struggle to generalize their observations beyond ad-hoc initialization and small-scale architectures or datasets, or they bypass these challenges by applying their mask to trained weights instead of initialized ones. In this paper, we demonstrate that the parameter sign configuration plays a crucial role in conveying useful information for generalization to any randomly initialized network. Through linear mode connectivity analysis, we observe that a sparse network trained by an existing IP method can retain its basin of attraction if its parameter signs and normalization layer parameters are preserved. To take a step closer to finding a winning ticket, we alleviate the reliance on normalization layer parameters by preventing high error barriers along the linear path between the sparse network trained by our method and its counterpart with initialized normalization layer parameters. Interestingly, across various architectures and datasets, we observe that any randomly initialized network can be optimized to exhibit low error barriers along the linear path to the sparse network trained by our method by inheriting its sparsity and parameter sign information, potentially achieving performance comparable to the original. The code is available at this https URL\this http URL 

**Abstract (ZH)**: lottery票假设（LTH）认为，在特定条件下，存在一个稀疏子网络（即 winning ticket），该子网络可以从头开始训练，其泛化能力与参数丰富网络相当。通常找到 winning ticket 的方法是通过迭代剪枝（IP）保留原始网络的强泛化能力，并通过将得到的稀疏掩码应用于未训练网络来传递实现学习泛化的有用信息。然而，现有的 IP 方法仍然难以将观察结果扩展到非随意初始化和小型架构或数据集之外，或者通过将掩码应用于训练后权重而非初始化权重来绕过这些挑战。本文我们证明了参数符号配置在向任意随机初始化网络传递泛化有用信息中起着关键作用。通过线性模式连通性分析，我们发现，使用现有 IP 方法训练的稀疏网络可以保留其吸引子盆地，前提是其参数符号和归一化层参数保持不变。为了更接近找到 winning ticket，我们通过防止从使用我们方法训练的稀疏网络到具有初始化归一化层参数的对应网络的线性路径上的高误差障碍来减轻依赖归一化层参数的限制。有趣的是，在各种架构和数据集上，我们观察到任何随机初始化的网络都可以被优化，以便在其到使用我们方法训练的稀疏网络的线性路径上表现出低误差障碍，潜在地达到与原始网络相当的性能。代码可在以下网址获得：this https URL this http URL 

---
# DyTTP: Trajectory Prediction with Normalization-Free Transformers 

**Title (ZH)**: DyTTP：无需归一化变换器的轨迹预测 

**Authors**: Yunxiang Liu, Hongkuo Niu  

**Link**: [PDF](https://arxiv.org/pdf/2504.05356)  

**Abstract**: Accurate trajectory prediction is a cornerstone for the safe operation of autonomous driving systems, where understanding the dynamic behavior of surrounding agents is crucial. Transformer-based architectures have demonstrated significant promise in capturing complex spatio-temporality dependencies. However, their reliance on normalization layers can lead to computation overhead and training instabilities. In this work, we present a two-fold approach to address these challenges. First, we integrate DynamicTanh (DyT), which is the latest method to promote transformers, into the backbone, replacing traditional layer normalization. This modification simplifies the network architecture and improves the stability of the inference. We are the first work to deploy the DyT to the trajectory prediction task. Complementing this, we employ a snapshot ensemble strategy to further boost trajectory prediction performance. Using cyclical learning rate scheduling, multiple model snapshots are captured during a single training run. These snapshots are then aggregated via simple averaging at inference time, allowing the model to benefit from diverse hypotheses without incurring substantial additional computational cost. Extensive experiments on Argoverse datasets demonstrate that our combined approach significantly improves prediction accuracy, inference speed and robustness in diverse driving scenarios. This work underscores the potential of normalization-free transformer designs augmented with lightweight ensemble techniques in advancing trajectory forecasting for autonomous vehicles. 

**Abstract (ZH)**: 基于DynamicTanh和轻量级集成技术的无归一化层变换器在自动驾驶轨迹预测中的应用 

---
# Achieving binary weight and activation for LLMs using Post-Training Quantization 

**Title (ZH)**: 使用后训练量化实现LLMs的二值权重和激活 

**Authors**: Siqing Song, Chuang Wang, Ruiqi Wang, Yi Yang, Xuyao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05352)  

**Abstract**: Quantizing large language models (LLMs) to 1-bit precision significantly reduces computational costs, but existing quantization techniques suffer from noticeable performance degradation when using weight and activation precisions below 4 bits (W4A4). In this paper, we propose a post-training quantization framework with W(1+1)A(1*4) configuration, where weights are quantized to 1 bit with an additional 1 bit for fine-grain grouping and activations are quantized to 1 bit with a 4-fold increase in the number of channels. For weight quantization, we propose utilizing Hessian-aware fine-grained grouping along with an EM-based quantization scheme. For activation quantization, we decompose INT4-quantized activations into a 4 * INT1 format equivalently and simultaneously smooth the scaling factors based on quantization errors, which further reduces the quantization errors in activations. Our method surpasses state-of-the-art (SOTA) LLM quantization baselines on W2A4 across multiple tasks, pushing the boundaries of existing LLM quantization methods toward fully binarized models. 

**Abstract (ZH)**: 将大型语言模型（LLMs）量化到1比特精度显著降低了计算成本，但现有的量化技术在使用低于4比特的权重和激活精度（W4A4）时会遭受明显的性能下降。本文提出了一种后训练量化框架，配置为W(1+1)A(1*4)，其中权重被量化为1比特，并附加1比特用于细粒度分组，激活值被量化为1比特，通道数增加4倍。在权重量化方面，我们提出了利用海森矩阵感知的细粒度分组结合EM基于的量化方案。在激活量化方面，我们将INT4量化激活等效分解为4 * INT1格式，并同时根据量化误差平滑缩放因子，进一步减少激活的量化误差。我们的方法在多个任务上超越了W2A4的最新基准量化方法，推动了现有LLM量化方法向全二值化模型的边界。 

---
# Non-linear Phillips Curve for India: Evidence from Explainable Machine Learning 

**Title (ZH)**: 印度的非线性菲利普斯曲线：可解释机器学习的证据 

**Authors**: Shovon Sengupta, Bhanu Pratap, Amit Pawar  

**Link**: [PDF](https://arxiv.org/pdf/2504.05350)  

**Abstract**: The conventional linear Phillips curve model, while widely used in policymaking, often struggles to deliver accurate forecasts in the presence of structural breaks and inherent nonlinearities. This paper addresses these limitations by leveraging machine learning methods within a New Keynesian Phillips Curve framework to forecast and explain headline inflation in India, a major emerging economy. Our analysis demonstrates that machine learning-based approaches significantly outperform standard linear models in forecasting accuracy. Moreover, by employing explainable machine learning techniques, we reveal that the Phillips curve relationship in India is highly nonlinear, characterized by thresholds and interaction effects among key variables. Headline inflation is primarily driven by inflation expectations, followed by past inflation and the output gap, while supply shocks, except rainfall, exert only a marginal influence. These findings highlight the ability of machine learning models to improve forecast accuracy and uncover complex, nonlinear dynamics in inflation data, offering valuable insights for policymakers. 

**Abstract (ZH)**: 基于机器学习的新凯恩斯菲利普斯曲线框架下的印度核心通胀预测与解释：超越传统的线性模型 

---
# Hyperflows: Pruning Reveals the Importance of Weights 

**Title (ZH)**: 超流：剪枝揭示了权重的重要性 

**Authors**: Eugen Barbulescu, Antonio Alexoaie  

**Link**: [PDF](https://arxiv.org/pdf/2504.05349)  

**Abstract**: Network pruning is used to reduce inference latency and power consumption in large neural networks. However, most existing methods struggle to accurately assess the importance of individual weights due to their inherent interrelatedness, leading to poor performance, especially at extreme sparsity levels. We introduce Hyperflows, a dynamic pruning approach that estimates each weight's importance by observing the network's gradient response to the weight's removal. A global pressure term continuously drives all weights toward pruning, with those critical for accuracy being automatically regrown based on their flow, the aggregated gradient signal when they are absent. We explore the relationship between final sparsity and pressure, deriving power-law equations similar to those found in neural scaling laws. Empirically, we demonstrate state-of-the-art results with ResNet-50 and VGG-19 on CIFAR-10 and CIFAR-100. 

**Abstract (ZH)**: Hyperflows：一种基于梯度响应的动态剪枝方法及其在稀疏性与压力关系上的探索 

---
# Thanos: A Block-wise Pruning Algorithm for Efficient Large Language Model Compression 

**Title (ZH)**: Thanos：一种块级裁剪算法，用于高效的大语言模型压缩 

**Authors**: Ivan Ilin, Peter Richtarik  

**Link**: [PDF](https://arxiv.org/pdf/2504.05346)  

**Abstract**: This paper presents Thanos, a novel weight-pruning algorithm designed to reduce the memory footprint and enhance the computational efficiency of large language models (LLMs) by removing redundant weights while maintaining accuracy. Thanos introduces a block-wise pruning strategy with adaptive masks that dynamically adjust to weight importance, enabling flexible sparsity patterns and structured formats, such as $n:m$ sparsity, optimized for hardware acceleration. Experimental evaluations demonstrate that Thanos achieves state-of-the-art performance in structured pruning and outperforms existing methods in unstructured pruning. By providing an efficient and adaptable approach to model compression, Thanos offers a practical solution for deploying large models in resource-constrained environments. 

**Abstract (ZH)**: Thanos：一种新型的权重剪枝算法，通过去除冗余权重以减少大语言模型的内存占用并提高计算效率，同时保持准确性 

---
# Divergent Paths: Separating Homophilic and Heterophilic Learning for Enhanced Graph-level Representations 

**Title (ZH)**: 分歧的道路：分离同ophilic和异ophilic学习以增强图级表示 

**Authors**: Han Lei, Jiaxing Xu, Xia Dong, Yiping Ke  

**Link**: [PDF](https://arxiv.org/pdf/2504.05344)  

**Abstract**: Graph Convolutional Networks (GCNs) are predominantly tailored for graphs displaying homophily, where similar nodes connect, but often fail on heterophilic graphs. The strategy of adopting distinct approaches to learn from homophilic and heterophilic components in node-level tasks has been widely discussed and proven effective both theoretically and experimentally. However, in graph-level tasks, research on this topic remains notably scarce. Addressing this gap, our research conducts an analysis on graphs with nodes' category ID available, distinguishing intra-category and inter-category components as embodiment of homophily and heterophily, respectively. We find while GCNs excel at extracting information within categories, they frequently capture noise from inter-category components. Consequently, it is crucial to employ distinct learning strategies for intra- and inter-category elements. To alleviate this problem, we separately learn the intra- and inter-category parts by a combination of an intra-category convolution (IntraNet) and an inter-category high-pass graph convolution (InterNet). Our IntraNet is supported by sophisticated graph preprocessing steps and a novel category-based graph readout function. For the InterNet, we utilize a high-pass filter to amplify the node disparities, enhancing the recognition of details in the high-frequency components. The proposed approach, DivGNN, combines the IntraNet and InterNet with a gated mechanism and substantially improves classification performance on graph-level tasks, surpassing traditional GNN baselines in effectiveness. 

**Abstract (ZH)**: 基于区分内类间和跨类别的图卷积网络（DivGNN）研究 

---
# AROMA: Autonomous Rank-one Matrix Adaptation 

**Title (ZH)**: AROMA：自主秩一矩阵适应 

**Authors**: Hao Nan Sheng, Zhi-yong Wang, Mingrui Yang, Hing Cheung So  

**Link**: [PDF](https://arxiv.org/pdf/2504.05343)  

**Abstract**: As large language models continue to grow in size, parameter-efficient fine-tuning has become increasingly crucial. While low-rank adaptation (LoRA) offers a solution through low-rank updates, its static rank allocation may yield suboptimal results. Adaptive low-rank adaptation (AdaLoRA) improves this with dynamic allocation but remains sensitive to initial and target rank configurations. We introduce AROMA, a framework that automatically constructs layer-specific updates by iteratively building up rank-one components with very few trainable parameters that gradually diminish to zero. Unlike existing methods that employ rank reduction mechanisms, AROMA introduces a dual-loop architecture for rank growth. The inner loop extracts information from each rank-one subspace, while the outer loop determines the number of rank-one subspaces, i.e., the optimal rank. We reset optimizer states to maintain subspace independence. AROMA significantly reduces parameters compared to LoRA and AdaLoRA while achieving superior performance on natural language understanding and commonsense reasoning tasks, offering new insights into adaptive parameter-efficient fine-tuning. The code is available at \href{this https URL}{AROMA}. 

**Abstract (ZH)**: 随着大型语言模型的不断扩大，参数高效的微调变得越来越重要。虽然低秩适应（LoRA）通过低秩更新提供了解决方案，但其静态秩分配可能导致次优结果。自适应低秩适应（AdaLoRA）通过动态分配改进了这一问题，但仍对初始和目标秩配置敏感。我们提出了AROMA框架，该框架通过迭代构建具有极少可训练参数的秩一子空间，并逐渐减少到零，自动构建层特定的更新。AROMA不同于现有方法使用秩减少机制，引入了双环架构以促进秩增长。内环从每个秩一子空间中提取信息，外环确定秩一子空间的数量，即最优秩。我们重置优化器状态以保持子空间独立性。AROMA在参数量显著少于LoRA和AdaLoRA的情况下，实现了自然语言理解和常识推理任务上的优越性能，为自适应参数高效微调提供了新的见解。代码可在AROMA：[点击这里](this https URL)获取。 

---
# MASS: MoErging through Adaptive Subspace Selection 

**Title (ZH)**: MASS: 通过自适应子空间选择的MoE化 

**Authors**: Donato Crisostomi, Alessandro Zirilli, Antonio Andrea Gargiulo, Maria Sofia Bucarelli, Simone Scardapane, Fabrizio Silvestri, Iacopo Masi, Emanuele Rodolà  

**Link**: [PDF](https://arxiv.org/pdf/2504.05342)  

**Abstract**: Model merging has recently emerged as a lightweight alternative to ensembling, combining multiple fine-tuned models into a single set of parameters with no additional training overhead. Yet, existing merging methods fall short of matching the full accuracy of separately fine-tuned endpoints. We present MASS (MoErging through Adaptive Subspace Selection), a new approach that closes this gap by unifying multiple fine-tuned models while retaining near state-of-the-art performance across tasks. Building on the low-rank decomposition of per-task updates, MASS stores only the most salient singular components for each task and merges them into a shared model. At inference time, a non-parametric, data-free router identifies which subspace (or combination thereof) best explains an input's intermediate features and activates the corresponding task-specific block. This procedure is fully training-free and introduces only a two-pass inference overhead plus a ~2 storage factor compared to a single pretrained model, irrespective of the number of tasks. We evaluate MASS on CLIP-based image classification using ViT-B-16, ViT-B-32 and ViT-L-14 for benchmarks of 8, 14 and 20 tasks respectively, establishing a new state-of-the-art. Most notably, MASS recovers up to ~98% of the average accuracy of individual fine-tuned models, making it a practical alternative to ensembling at a fraction of the storage cost. 

**Abstract (ZH)**: MASS：通过自适应子空间选择的MoErging 

---
# Three-Factor Learning in Spiking Neural Networks: An Overview of Methods and Trends from a Machine Learning Perspective 

**Title (ZH)**: 从机器学习视角出发的.spi神经网络三因素学习综述：方法与趋势 

**Authors**: Szymon Mazurek, Jakub Caputa, Jan K. Argasiński, Maciej Wielgosz  

**Link**: [PDF](https://arxiv.org/pdf/2504.05341)  

**Abstract**: Three-factor learning rules in Spiking Neural Networks (SNNs) have emerged as a crucial extension to traditional Hebbian learning and Spike-Timing-Dependent Plasticity (STDP), incorporating neuromodulatory signals to improve adaptation and learning efficiency. These mechanisms enhance biological plausibility and facilitate improved credit assignment in artificial neural systems. This paper takes a view on this topic from a machine learning perspective, providing an overview of recent advances in three-factor learning, discusses theoretical foundations, algorithmic implementations, and their relevance to reinforcement learning and neuromorphic computing. In addition, we explore interdisciplinary approaches, scalability challenges, and potential applications in robotics, cognitive modeling, and AI systems. Finally, we highlight key research gaps and propose future directions for bridging the gap between neuroscience and artificial intelligence. 

**Abstract (ZH)**: 三因子学习规则在神经脉冲网络（SNNs）中的新兴作用：作为传统 Hebbsian 学习和时序依赖可塑性（STDP）的关键扩展，通过引入神经调制信号以提高适应性和学习效率。这些机制增强了生物可行性，并促进了人工神经系统中的改进责任指派。从机器学习的角度出发，本文概述了三因子学习的最近进展，讨论了其理论基础、算法实现及其与强化学习和神经形态计算的相关性。此外，我们探讨了跨学科方法、可扩展性挑战及其在机器人、认知建模和AI系统中的潜在应用。最后，我们强调了关键研究空白，并提议了将神经科学与人工智能联系起来的未来研究方向。 

---
# Improving Early Prediction of Type 2 Diabetes Mellitus with ECG-DiaNet: A Multimodal Neural Network Leveraging Electrocardiogram and Clinical Risk Factors 

**Title (ZH)**: 基于心电图和临床危险因素的多模态神经网络ECG-DiaNet早期预测2型糖尿病 Mellitus 的改进研究 

**Authors**: Farida Mohsen, Zubair Shah  

**Link**: [PDF](https://arxiv.org/pdf/2504.05338)  

**Abstract**: Type 2 Diabetes Mellitus (T2DM) remains a global health challenge, underscoring the need for early and accurate risk prediction. This study presents ECG-DiaNet, a multimodal deep learning model that integrates electrocardiogram (ECG) features with clinical risk factors (CRFs) to enhance T2DM onset prediction. Using data from Qatar Biobank (QBB), we trained and validated models on a development cohort (n=2043) and evaluated performance on a longitudinal test set (n=395) with five-year follow-up. ECG-DiaNet outperformed unimodal ECG-only and CRF-only models, achieving a higher AUROC (0.845 vs 0.8217) than the CRF-only model, with statistical significance (DeLong p<0.001). Reclassification metrics further confirmed improvements: Net Reclassification Improvement (NRI=0.0153) and Integrated Discrimination Improvement (IDI=0.0482). Risk stratification into low-, medium-, and high-risk groups showed ECG-DiaNet achieved superior positive predictive value (PPV) in high-risk individuals. The model's reliance on non-invasive and widely available ECG signals supports its feasibility in clinical and community health settings. By combining cardiac electrophysiology and systemic risk profiles, ECG-DiaNet addresses the multifactorial nature of T2DM and supports precision prevention. These findings highlight the value of multimodal AI in advancing early detection and prevention strategies for T2DM, particularly in underrepresented Middle Eastern populations. 

**Abstract (ZH)**: Type 2糖尿病 Mellitus (T2DM) 继续是全球健康挑战，凸显了早期和准确风险预测的必要性。本研究提出了一种多模态深度学习模型ECG-DiaNet，该模型将心电图 (ECG) 特征与临床风险因素 (CRFs) 结合起来，以提高T2DM发病预测。使用卡塔尔生物银行 (QBB) 数据，我们在开发队列 (n=2043) 上训练和验证了模型，并在包含五年人随访的数据队列 (n=395) 上进行了评估。ECG-DiaNet 在 AUCROC (0.845 vs 0.8217) 方面优于仅使用ECG 和仅使用CRF 的模型，具有统计学意义 (DeLong p<0.001)。重新分类指标进一步证实了这些改善：Net Reclassification Improvement (NRI=0.0153) 和 Integrated Discrimination Improvement (IDI=0.0482)。风险分层结果显示ECG-DiaNet 在高风险个体中的阳性预测值 (PPV) 更高。该模型依赖于非侵入性且广泛可用的心电图信号，支持其在临床和社区健康设置中的可行性。通过结合心脏电生理学和系统风险特征，ECG-DiaNet 解决了T2DM 的多因素特征，并支持精准预防。这些发现强调了多模态AI 在促进T2DM 早期检测和预防策略方面的价值，特别是在未被充分代表的中东人群中。 

---
# Level Generation with Constrained Expressive Range 

**Title (ZH)**: 带有受限表达范围的关卡生成 

**Authors**: Mahsa Bazzaz, Seth Cooper  

**Link**: [PDF](https://arxiv.org/pdf/2504.05334)  

**Abstract**: Expressive range analysis is a visualization-based technique used to evaluate the performance of generative models, particularly in game level generation. It typically employs two quantifiable metrics to position generated artifacts on a 2D plot, offering insight into how content is distributed within a defined metric space. In this work, we use the expressive range of a generator as the conceptual space of possible creations. Inspired by the quality diversity paradigm, we explore this space to generate levels. To do so, we use a constraint-based generator that systematically traverses and generates levels in this space. To train the constraint-based generator we use different tile patterns to learn from the initial example levels. We analyze how different patterns influence the exploration of the expressive range. Specifically, we compare the exploration process based on time, the number of successful and failed sample generations, and the overall interestingness of the generated levels. Unlike typical quality diversity approaches that rely on random generation and hope to get good coverage of the expressive range, this approach systematically traverses the grid ensuring more coverage. This helps create unique and interesting game levels while also improving our understanding of the generator's strengths and limitations. 

**Abstract (ZH)**: 表达范围分析是一种基于可视化的技术，用于评估生成模型的表现，特别是在游戏关卡生成中的应用。通常，它使用两个可量化指标在2D图上定位生成的元素，提供内容在定义的度量空间内的分布洞察。在本文中，我们将生成器的表达范围作为可能创作的概念空间。受质量多样性范式的启发，我们在该空间中探索以生成关卡。为此，我们使用基于约束的生成器，系统地遍历并生成该空间中的关卡。为了训练基于约束的生成器，我们使用不同的瓷砖图案从初始示例关卡中学习。我们分析不同模式如何影响表达范围的探索过程。具体而言，我们根据时间、成功和失败样本生成的数量以及生成关卡的总体趣味性来比较探索过程。与依赖随机生成以期望获得表达范围良好覆盖的典型质量多样性方法不同，这种方法系统地遍历网格，确保更好的覆盖。这有助于创建独特的有趣游戏关卡，同时也有助于我们更好地理解生成器的优势和局限性。 

---
# When is using AI the rational choice? The importance of counterfactuals in AI deployment decisions 

**Title (ZH)**: 何时使用AI是理性选择？反事实推理在AI部署决策中的重要性 

**Authors**: Paul Lehner, Elinor Yeo  

**Link**: [PDF](https://arxiv.org/pdf/2504.05333)  

**Abstract**: Decisions to deploy AI capabilities are often driven by counterfactuals - a comparison of decisions made using AI to decisions that would have been made if the AI were not used. Counterfactual misses, which are poor decisions that are attributable to using AI, may have disproportionate disutility to AI deployment decision makers. Counterfactual hits, which are good decisions attributable to AI usage, may provide little benefit beyond the benefit of better decisions. This paper explores how to include counterfactual outcomes into usage decision expected utility assessments. Several properties emerge when counterfactuals are explicitly included. First, there are many contexts where the expected utility of AI usage is positive for intended beneficiaries and strongly negative for stakeholders and deployment decision makers. Second, high levels of complementarity, where differing AI and user assessments are merged beneficially, often leads to substantial disutility for stakeholders. Third, apparently small changes in how users interact with an AI capability can substantially impact stakeholder utility. Fourth, cognitive biases such as expert overconfidence and hindsight bias exacerbate the perceived frequency of costly counterfactual misses. The expected utility assessment approach presented here is intended to help AI developers and deployment decision makers to navigate the subtle but substantial impact of counterfactuals so as to better ensure that beneficial AI capabilities are used. 

**Abstract (ZH)**: 基于反事实结果的AI能力部署决策预期效用评估 

---
# Not someone, but something: Rethinking trust in the age of medical AI 

**Title (ZH)**: 不仅仅是某个人，而是一种东西：再思考医疗AI时代的信任问题 

**Authors**: Jan Beger  

**Link**: [PDF](https://arxiv.org/pdf/2504.05331)  

**Abstract**: As artificial intelligence (AI) becomes embedded in healthcare, trust in medical decision-making is changing fast. This opinion paper argues that trust in AI isn't a simple transfer from humans to machines -- it's a dynamic, evolving relationship that must be built and maintained. Rather than debating whether AI belongs in medicine, this paper asks: what kind of trust must AI earn, and how? Drawing from philosophy, bioethics, and system design, it explores the key differences between human trust and machine reliability -- emphasizing transparency, accountability, and alignment with the values of care. It argues that trust in AI shouldn't rely on mimicking empathy or intuition, but on thoughtful design, responsible deployment, and clear moral responsibility. The goal is a balanced view -- one that avoids blind optimism and reflexive fear. Trust in AI must be treated not as a given, but as something to be earned over time. 

**Abstract (ZH)**: 随着人工智能（AI）在医疗领域的嵌入，对医疗决策的信任正在迅速变化。本文认为，对AI的信任不是从人类简单转移到机器上的单一过程——而是一种动态且不断演变的关系，需要通过建设和维护来形成。本文不讨论AI是否属于医疗领域，而是探讨AI应当获得何种信任以及如何获得。结合哲学、生物伦理学和系统设计的视角，本文探讨了人类信任与机器可靠性之间的关键差异，强调透明度、可问责性和与关爱价值观的一致性。本文 argue，对AI的信任不应依赖于模仿同情或直觉，而应依赖于深思熟虑的设计、负责任的应用和明确的道德责任。目标是形成一个平衡的观点——既避免盲目的乐观主义，也避免无根据的恐惧。对AI的信任不应被视为理所当然，而应视为需要通过时间来赢得的。 

---
# Unequal Opportunities: Examining the Bias in Geographical Recommendations by Large Language Models 

**Title (ZH)**: 机会不均等：考察大型语言模型在地理位置推荐中的偏见 

**Authors**: Shiran Dudy, Thulasi Tholeti, Resmi Ramachandranpillai, Muhammad Ali, Toby Jia-Jun Li, Ricardo Baeza-Yates  

**Link**: [PDF](https://arxiv.org/pdf/2504.05325)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have made them a popular information-seeking tool among end users. However, the statistical training methods for LLMs have raised concerns about their representation of under-represented topics, potentially leading to biases that could influence real-world decisions and opportunities. These biases could have significant economic, social, and cultural impacts as LLMs become more prevalent, whether through direct interactions--such as when users engage with chatbots or automated assistants--or through their integration into third-party applications (as agents), where the models influence decision-making processes and functionalities behind the scenes. Our study examines the biases present in LLMs recommendations of U.S. cities and towns across three domains: relocation, tourism, and starting a business. We explore two key research questions: (i) How similar LLMs responses are, and (ii) How this similarity might favor areas with certain characteristics over others, introducing biases. We focus on the consistency of LLMs responses and their tendency to over-represent or under-represent specific locations. Our findings point to consistent demographic biases in these recommendations, which could perpetuate a ``rich-get-richer'' effect that widens existing economic disparities. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的进步使其成为终端用户常用的检索信息工具。然而，LLMs的统计训练方法引发了对其对代表性不足主题的表示的担忧，可能导致偏见，进而影响现实世界的决策和机会。随着LLMs的应用越来越广泛，无论通过直接交互（如用户与聊天机器人或自动化助手互动）还是通过其嵌入第三方应用程序（作为代理），这些偏见都可能产生重大的经济、社会和文化影响。我们的研究考察了LLMs在美国城市和城镇推荐中的偏见，涵盖了三个领域：迁移、旅游和创业。我们探讨了两个核心研究问题：（i）LLMs响应的相似性，以及（ii）这种相似性如何可能有利于某些特征的地区，引入偏见。我们关注LLMs响应的一致性和其过度代表或低估特定地点的倾向。研究发现这些推荐中存在一致的人口统计学偏见，这可能会加剧现有的经济不平等。 

---
# Hybrid Retrieval for Hallucination Mitigation in Large Language Models: A Comparative Analysis 

**Title (ZH)**: 大型语言模型中幻觉缓解的混合检索方法：一种比较分析 

**Authors**: Chandana Sree Mala, Gizem Gezici, Fosca Giannotti  

**Link**: [PDF](https://arxiv.org/pdf/2504.05324)  

**Abstract**: Large Language Models (LLMs) excel in language comprehension and generation but are prone to hallucinations, producing factually incorrect or unsupported outputs. Retrieval Augmented Generation (RAG) systems address this issue by grounding LLM responses with external knowledge. This study evaluates the relationship between retriever effectiveness and hallucination reduction in LLMs using three retrieval approaches: sparse retrieval based on BM25 keyword search, dense retrieval using semantic search with Sentence Transformers, and a proposed hybrid retrieval module. The hybrid module incorporates query expansion and combines the results of sparse and dense retrievers through a dynamically weighted Reciprocal Rank Fusion score. Using the HaluBench dataset, a benchmark for hallucinations in question answering tasks, we assess retrieval performance with metrics such as mean average precision and normalised discounted cumulative gain, focusing on the relevance of the top three retrieved documents. Results show that the hybrid retriever achieves better relevance scores, outperforming both sparse and dense retrievers. Further evaluation of LLM-generated answers against ground truth using metrics such as accuracy, hallucination rate, and rejection rate reveals that the hybrid retriever achieves the highest accuracy on fails, the lowest hallucination rate, and the lowest rejection rate. These findings highlight the hybrid retriever's ability to enhance retrieval relevance, reduce hallucination rates, and improve LLM reliability, emphasising the importance of advanced retrieval techniques in mitigating hallucinations and improving response accuracy. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在语言理解和生成方面表现出色，但容易产生幻觉，生成事实错误或缺乏支持的输出。检索增强生成（RAG）系统通过将LLM响应与外部知识相结合来解决这一问题。本研究使用三种检索方法评估检索器有效性与幻觉减少之间的关系：基于BM25关键词搜索的稀疏检索、使用Sentence Transformers进行语义搜索的密集检索，以及一个提出的混合检索模块。混合模块结合了查询扩展，并通过动态加权的互逆排名融合得分将稀疏检索和密集检索的结果结合起来。使用HaluBench数据集，这是一个用于问答任务中幻觉的基准测试集，我们使用平均精度均值和归一化累积增益等指标评估检索性能，重点关注检索的前三份文档的相关性。结果表明，混合检索器在相关性评分方面表现更好，优于稀疏和密集检索器。进一步使用准确率、幻觉率和拒绝率等指标评估LLM生成的答案与真实答案的差异，结果显示混合检索器在错误上的准确率最高、幻觉率最低、拒绝率最低。这些发现突显了混合检索器增强检索相关性、降低幻觉率和提高LLM可靠性的能力，强调了先进检索技术在减轻幻觉和提高响应准确性方面的重要性。 

---
# Multi-Perspective Attention Mechanism for Bias-Aware Sequential Recommendation 

**Title (ZH)**: 面向偏差感知的多视角注意力机制序列推荐 

**Authors**: Mingjian Fu, Hengsheng Chen, Dongchun Jiang, Yanchao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.05323)  

**Abstract**: In the era of advancing information technology, recommender systems have emerged as crucial tools for dealing with information overload. However, traditional recommender systems still have limitations in capturing the dynamic evolution of user behavior. To better understand and predict user behavior, especially taking into account the complexity of temporal evolution, sequential recommender systems have gradually become the focus of research. Currently, many sequential recommendation algorithms ignore the amplification effects of prevalent biases, which leads to recommendation results being susceptible to the Matthew Effect. Additionally, it will impose limitations on the recommender system's ability to deeply perceive and capture the dynamic shifts in user preferences, thereby diminishing the extent of its recommendation reach. To address this issue effectively, we propose a recommendation system based on sequential information and attention mechanism called Multi-Perspective Attention Bias Sequential Recommendation (MABSRec). Firstly, we reconstruct user sequences into three short types and utilize graph neural networks for item weighting. Subsequently, an adaptive multi-bias perspective attention module is proposed to enhance the accuracy of recommendations. Experimental results show that the MABSRec model exhibits significant advantages in all evaluation metrics, demonstrating its excellent performance in the sequence recommendation task. 

**Abstract (ZH)**: 电子商务环境下基于多视角注意力偏好修正的序列推荐系统（MABSRec） 

---
# VALUE: Value-Aware Large Language Model for Query Rewriting via Weighted Trie in Sponsored Search 

**Title (ZH)**: 价值感知型大型语言模型：基于加权trie树的赞助搜索查询重写 

**Authors**: Boyang Zuo, Xiao Zhang, Feng Li, Pengjie Wang, Jian Xu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.05321)  

**Abstract**: In the realm of sponsored search advertising, matching advertisements with the search intent of a user's query is crucial. Query-to-bidwords(i.e. bidding keywords) rewriting is a vital technique that has garnered significant attention. Recently, with the prevalence of LLMs, generative retrieval methods have proven effective in producing high-relevance rewrites. However, we have identified a significant limitation in existing approaches: While fine-tuning LLMs for specific domains enhances semantic relevance, these models have no perception of the intrinsic value of their generated outputs, such as commercial value. Therefore, after SFT, a RLHF phase is often employed to address this issue. Nevertheless, traditional preference alignment methods often face challenges in aligning fine-grained values and are susceptible to overfitting, which diminishes the effectiveness and quality of the generated results. To address these challenges, we propose VALUE(Value-Aware Large language model for qUery rewriting via wEighted trie), the first framework that ensures the generation of high-value and highly relevant bidwords. Our approach utilizes weighted trie, an innovative modification of the traditional trie data structure. By modulating the LLM's output probability distribution with value information from the trie during decoding process, we constrain the generation space and guide the trajectory of text production. Offline experiments demonstrate the effectiveness of our method in semantic matching and preference alignment, showing a remarkable improvement in the value attribute by more than fivefold. Online A/B tests further revealed that our Revenue Per Mille (RPM) metric increased by 1.64%. VALUE has been deployed on our advertising system since October 2024 and served the Double Eleven promotions, the biggest shopping carnival in China. 

**Abstract (ZH)**: 基于赞助搜索广告的查询到出价词重写：考虑价值的价值感知大规模语言模型trie框架 

---
# Predictive Modeling: BIM Command Recommendation Based on Large-scale Usage Logs 

**Title (ZH)**: 基于大规模使用日志的BIM命令预测模型 

**Authors**: Changyu Du, Zihan Deng, Stavros Nousias, André Borrmann  

**Link**: [PDF](https://arxiv.org/pdf/2504.05319)  

**Abstract**: The adoption of Building Information Modeling (BIM) and model-based design within the Architecture, Engineering, and Construction (AEC) industry has been hindered by the perception that using BIM authoring tools demands more effort than conventional 2D drafting. To enhance design efficiency, this paper proposes a BIM command recommendation framework that predicts the optimal next actions in real-time based on users' historical interactions. We propose a comprehensive filtering and enhancement method for large-scale raw BIM log data and introduce a novel command recommendation model. Our model builds upon the state-of-the-art Transformer backbones originally developed for large language models (LLMs), incorporating a custom feature fusion module, dedicated loss function, and targeted learning strategy. In a case study, the proposed method is applied to over 32 billion rows of real-world log data collected globally from the BIM authoring software Vectorworks. Experimental results demonstrate that our method can learn universal and generalizable modeling patterns from anonymous user interaction sequences across different countries, disciplines, and projects. When generating recommendations for the next command, our approach achieves a Recall@10 of approximately 84%. 

**Abstract (ZH)**: 建筑信息建模（BIM）和基于模型的设计在建筑、工程和施工（AEC）行业的采用受到使用BIM创作工具比传统2D制图需要更多努力的认知阻碍。为了提高设计效率，本文提出了一种BIM命令推荐框架，该框架基于用户的历史交互实时预测最优化的下一步操作。我们提出了一种针对大规模原始BIM日志数据的全面过滤和增强方法，并引入了一种新型命令推荐模型。该模型基于专门为大型语言模型（LLMs）开发的最先进的Transformer骨干架构，结合了自定义特征融合模块、专用损失函数和目标化学习策略。在一项案例研究中，该方法应用于全球从Vectorworks BIM创作软件收集的超过320亿条真实日志数据。实验结果表明，我们的方法可以从不同国家、学科和项目中的匿名用户交互序列中学习到通用且可泛化的建模模式。在生成下一个命令的推荐时，我们的方法实现了约84%的Recall@10。 

---
# Efficient Multi-Task Learning via Generalist Recommender 

**Title (ZH)**: 高效多任务学习通过通用推荐器 

**Authors**: Luyang Wang, Cangcheng Tang, Chongyang Zhang, Jun Ruan, Kai Huang, Jason Dai  

**Link**: [PDF](https://arxiv.org/pdf/2504.05318)  

**Abstract**: Multi-task learning (MTL) is a common machine learning technique that allows the model to share information across different tasks and improve the accuracy of recommendations for all of them. Many existing MTL implementations suffer from scalability issues as the training and inference performance can degrade with the increasing number of tasks, which can limit production use case scenarios for MTL-based recommender systems. Inspired by the recent advances of large language models, we developed an end-to-end efficient and scalable Generalist Recommender (GRec). GRec takes comprehensive data signals by utilizing NLP heads, parallel Transformers, as well as a wide and deep structure to process multi-modal inputs. These inputs are then combined and fed through a newly proposed task-sentence level routing mechanism to scale the model capabilities on multiple tasks without compromising performance. Offline evaluations and online experiments show that GRec significantly outperforms our previous recommender solutions. GRec has been successfully deployed on one of the largest telecom websites and apps, effectively managing high volumes of online traffic every day. 

**Abstract (ZH)**: 多任务学习（MTL）是一种常见的机器学习技术，允许模型在不同任务之间共享信息并提高所有任务的推荐准确性。许多现有的MTL实现由于训练和推理性能会随着任务数量的增加而恶化，从而限制了基于MTL的推荐系统的生产应用场景。受大型语言模型近期进展的启发，我们开发了一个端到端高效且可扩展的通用推荐系统（GRec）。GRec通过利用NLP头部、并行Transformer以及宽深结构来处理多模态输入，并采取了一种新的任务-句子级别路由机制，使其能够在多个任务上扩展模型能力而不牺牲性能。离线评估和在线实验表明，GRec显著优于我们之前的推荐解决方案。GRec已成功部署在最大的电信网站和应用程序之一上，每天有效管理大量的在线流量。 

---
# On Synthesizing Data for Context Attribution in Question Answering 

**Title (ZH)**: 基于上下文归因的问答数据合成 

**Authors**: Gorjan Radevski, Kiril Gashteovski, Shahbaz Syed, Christopher Malon, Sebastien Nicolas, Chia-Chien Hung, Timo Sztyler, Verena Heußer, Wiem Ben Rim, Masafumi Enomoto, Kunihiro Takeoka, Masafumi Oyamada, Goran Glavaš, Carolin Lawrence  

**Link**: [PDF](https://arxiv.org/pdf/2504.05317)  

**Abstract**: Question Answering (QA) accounts for a significant portion of LLM usage "in the wild". However, LLMs sometimes produce false or misleading responses, also known as "hallucinations". Therefore, grounding the generated answers in contextually provided information -- i.e., providing evidence for the generated text -- is paramount for LLMs' trustworthiness. Providing this information is the task of context attribution. In this paper, we systematically study LLM-based approaches for this task, namely we investigate (i) zero-shot inference, (ii) LLM ensembling, and (iii) fine-tuning of small LMs on synthetic data generated by larger LLMs. Our key contribution is SynQA: a novel generative strategy for synthesizing context attribution data. Given selected context sentences, an LLM generates QA pairs that are supported by these sentences. This leverages LLMs' natural strengths in text generation while ensuring clear attribution paths in the synthetic training data. We show that the attribution data synthesized via SynQA is highly effective for fine-tuning small LMs for context attribution in different QA tasks and domains. Finally, with a user study, we validate the usefulness of small LMs (fine-tuned on synthetic data from SynQA) in context attribution for QA. 

**Abstract (ZH)**: 基于LLM的方法在问答任务中合成上下文归因数据的研究 

---
# Scale Up Composed Image Retrieval Learning via Modification Text Generation 

**Title (ZH)**: 通过修改文本生成扩展组合图像检索学习 

**Authors**: Yinan Zhou, Yaxiong Wang, Haokun Lin, Chen Ma, Li Zhu, Zhedong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.05316)  

**Abstract**: Composed Image Retrieval (CIR) aims to search an image of interest using a combination of a reference image and modification text as the query. Despite recent advancements, this task remains challenging due to limited training data and laborious triplet annotation processes. To address this issue, this paper proposes to synthesize the training triplets to augment the training resource for the CIR problem. Specifically, we commence by training a modification text generator exploiting large-scale multimodal models and scale up the CIR learning throughout both the pretraining and fine-tuning stages. During pretraining, we leverage the trained generator to directly create Modification Text-oriented Synthetic Triplets(MTST) conditioned on pairs of images. For fine-tuning, we first synthesize reverse modification text to connect the target image back to the reference image. Subsequently, we devise a two-hop alignment strategy to incrementally close the semantic gap between the multimodal pair and the target image. We initially learn an implicit prototype utilizing both the original triplet and its reversed version in a cycle manner, followed by combining the implicit prototype feature with the modification text to facilitate accurate alignment with the target image. Extensive experiments validate the efficacy of the generated triplets and confirm that our proposed methodology attains competitive recall on both the CIRR and FashionIQ benchmarks. 

**Abstract (ZH)**: 组成图像检索（CIR）旨在使用参考图像和修改文本的组合作为查询来搜索感兴趣的图像。尽管近期取得了进展，但由于训练数据有限和三元组注释过程繁琐，这一任务仍然具有挑战性。为了解决这一问题，本文提出合成训练三元组以增强CIR问题的训练资源。具体而言，我们首先利用大型多模态模型训练一个修改文本生成器，并在整个预训练和微调阶段扩展CIR学习。在预训练阶段，我们利用训练好的生成器直接根据图像对生成面向修改文本的合成三元组（MTST）。在微调阶段，我们首先合成逆向修改文本将目标图像连接回参考图像。随后，我们设计了一种两阶段对齐策略逐步缩小多模态对和目标图像之间的语义差距。我们首先以循环方式利用原三元组及其逆向版本学习一个隐式原型，随后将隐式原型特征与修改文本结合以促进与目标图像的准确对齐。广泛的实验验证了生成三元组的有效性，并证实我们提出的方法在CIRR和FashionIQ基准上取得了竞争力的召回率。 

---
# Multimodal Quantitative Language for Generative Recommendation 

**Title (ZH)**: 多模态定量语言生成推荐 

**Authors**: Jianyang Zhai, Zi-Feng Mai, Chang-Dong Wang, Feidiao Yang, Xiawu Zheng, Hui Li, Yonghong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2504.05314)  

**Abstract**: Generative recommendation has emerged as a promising paradigm aiming at directly generating the identifiers of the target candidates. Most existing methods attempt to leverage prior knowledge embedded in Pre-trained Language Models (PLMs) to improve the recommendation performance. However, they often fail to accommodate the differences between the general linguistic knowledge of PLMs and the specific needs of recommendation systems. Moreover, they rarely consider the complementary knowledge between the multimodal information of items, which represents the multi-faceted preferences of users. To facilitate efficient recommendation knowledge transfer, we propose a novel approach called Multimodal Quantitative Language for Generative Recommendation (MQL4GRec). Our key idea is to transform items from different domains and modalities into a unified language, which can serve as a bridge for transferring recommendation knowledge. Specifically, we first introduce quantitative translators to convert the text and image content of items from various domains into a new and concise language, known as quantitative language, with all items sharing the same vocabulary. Then, we design a series of quantitative language generation tasks to enrich quantitative language with semantic information and prior knowledge. Finally, we achieve the transfer of recommendation knowledge from different domains and modalities to the recommendation task through pre-training and fine-tuning. We evaluate the effectiveness of MQL4GRec through extensive experiments and comparisons with existing methods, achieving improvements over the baseline by 11.18\%, 14.82\%, and 7.95\% on the NDCG metric across three different datasets, respectively. 

**Abstract (ZH)**: 多模态定量语言促进生成性推荐（MQL4GRec） 

---
# Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation 

**Title (ZH)**: 面向自适应记忆优化的增强检索辅助生成 

**Authors**: Qitao Qin, Yucong Luo, Yihang Lu, Zhibo Chu, Xianwei Meng  

**Link**: [PDF](https://arxiv.org/pdf/2504.05312)  

**Abstract**: Retrieval-Augmented Generation (RAG), by integrating non-parametric knowledge from external knowledge bases into models, has emerged as a promising approach to enhancing response accuracy while mitigating factual errors and hallucinations. This method has been widely applied in tasks such as Question Answering (QA). However, existing RAG methods struggle with open-domain QA tasks because they perform independent retrieval operations and directly incorporate the retrieved information into generation without maintaining a summarizing memory or using adaptive retrieval strategies, leading to noise from redundant information and insufficient information integration. To address these challenges, we propose Adaptive memory-based optimization for enhanced RAG (Amber) for open-domain QA tasks, which comprises an Agent-based Memory Updater, an Adaptive Information Collector, and a Multi-granular Content Filter, working together within an iterative memory updating paradigm. Specifically, Amber integrates and optimizes the language model's memory through a multi-agent collaborative approach, ensuring comprehensive knowledge integration from previous retrieval steps. It dynamically adjusts retrieval queries and decides when to stop retrieval based on the accumulated knowledge, enhancing retrieval efficiency and effectiveness. Additionally, it reduces noise by filtering irrelevant content at multiple levels, retaining essential information to improve overall model performance. We conduct extensive experiments on several open-domain QA datasets, and the results demonstrate the superiority and effectiveness of our method and its components. The source code is available \footnote{this https URL}. 

**Abstract (ZH)**: 检索增强生成（RAG）通过将外部知识库中的非参数知识集成到模型中，已发展成为一种增强响应准确性、减轻事实错误和幻觉的有效方法。该方法在问答（QA）等任务中得到了广泛应用。然而，现有的RAG方法在处理开放域QA任务时存在问题，因为它们独立执行检索操作，并直接将检索到的信息融入生成过程中，而不保留总结性记忆或使用适应性检索策略，导致冗余信息的噪音和信息整合不足。为了解决这些挑战，我们提出了适用于开放域QA任务的自适应记忆优化以增强RAG（Amber），该方法包括基于代理的记忆更新器、自适应信息收集器和多粒度内容过滤器，共同在迭代的记忆更新范式中工作。具体而言，Amber通过多代理合作方法整合和优化语言模型的记忆，确保从之前的检索步骤中进行全面的知识整合。它会根据累积的知识动态调整检索查询，并决定何时停止检索，从而提高检索效率和效果。此外，它通过在多个级别上过滤无关内容来减少噪音，保留关键信息以提高整体模型性能。我们在多个开放域QA数据集上进行了广泛的实验，结果表明我们方法及其组件的优越性和有效性。源代码可通过以下链接获取。 

---
# IterQR: An Iterative Framework for LLM-based Query Rewrite in e-Commercial Search System 

**Title (ZH)**: IterQR：基于LLM的电子商务搜索系统中查询重写的一种迭代框架 

**Authors**: Shangyu Chen, Xinyu Jia, Yingfei Zhang, Shuai Zhang, Xiang Li, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.05309)  

**Abstract**: The essence of modern e-Commercial search system lies in matching user's intent and available candidates depending on user's query, providing personalized and precise service. However, user's query may be incorrect due to ambiguous input and typo, leading to inaccurate search. These cases may be released by query rewrite: modify query to other representation or expansion. However, traditional query rewrite replies on static rewrite vocabulary, which is manually established meanwhile lacks interaction with both domain knowledge in e-Commercial system and common knowledge in the real world. In this paper, with the ability to generate text content of Large Language Models (LLMs), we provide an iterative framework to generate query rewrite. The framework incorporates a 3-stage procedure in each iteration: Rewrite Generation with domain knowledge by Retrieval-Augmented Generation (RAG) and query understanding by Chain-of-Thoughts (CoT); Online Signal Collection with automatic positive rewrite update; Post-training of LLM with multi task objective to generate new rewrites. Our work (named as IterQR) provides a comprehensive framework to generate \textbf{Q}uery \textbf{R}ewrite with both domain / real-world knowledge. It automatically update and self-correct the rewrites during \textbf{iter}ations. \method{} has been deployed in Meituan Delivery's search system (China's leading food delivery platform), providing service for users with significant improvement. 

**Abstract (ZH)**: 现代电子商务搜索系统的核心在于根据用户的查询匹配用户的意图和可用的候选对象，提供个性化和精准的服务。然而，由于输入模糊和拼写错误，用户的查询可能会不准确，导致搜索结果不准确。这些情况可以通过查询重写来缓解：即将查询修改为其他表示形式或扩展。然而，传统的查询重写依赖于静态的重写词汇表，该词汇表是手动建立的，并且缺乏与电子商务系统领域知识和现实世界中的普通知识的交互。本文利用大规模语言模型（LLMs）生成文本内容的能力，提供了一个迭代框架来生成查询重写。该框架在每次迭代中包含三个阶段的过程：通过检索增强生成（RAG）和基于链式思考（CoT）的查询理解来进行重写生成；自动收集在线信号以更新正向重写；以及使用多任务目标对LLMs进行后训练以生成新的重写。我们的工作（命名为IterQR）提供了一个全面的框架，结合了领域知识和现实世界知识来生成查询重写。该框架在迭代过程中自动更新和自我校正重写。我们在美团配送的搜索系统（中国领先的食品配送平台）中部署了该方法，为用户提供显著改进的服务。 

---
# Toward Total Recall: Enhancing FAIRness through AI-Driven Metadata Standardization 

**Title (ZH)**: 向着全面回忆：通过AI驱动的元数据标准化提升FAIR性 

**Authors**: Sowmya S Sundaram, Mark A Musen  

**Link**: [PDF](https://arxiv.org/pdf/2504.05307)  

**Abstract**: Current metadata often suffer from incompleteness, inconsistency, and incorrect formatting, hindering effective data reuse and discovery. Using GPT-4 and a metadata knowledge base (CEDAR), we devised a method that standardizes metadata in scientific data sets, ensuring the adherence to community standards. The standardization process involves correcting and refining metadata entries to conform to established guidelines, significantly improving search performance and recall metrics. The investigation uses BioSample and GEO repositories to demonstrate the impact of these enhancements, showcasing how standardized metadata lead to better retrieval outcomes. The average recall improves significantly, rising from 17.65\% with the baseline raw datasets of BioSample and GEO to 62.87\% with our proposed metadata standardization pipeline. This finding highlights the transformative impact of integrating advanced AI models with structured metadata curation tools in achieving more effective and reliable data retrieval. 

**Abstract (ZH)**: 当前元数据往往存在不完整、不一致和格式错误的问题，阻碍了数据的有效重用和发现。利用GPT-4和元数据知识库（CEDAR），我们提出了一种方法，对科学数据集中的元数据进行标准化，确保其符合社区标准。标准化过程涉及修正和完善元数据条目的格式，以符合既定指南，显著提高搜索性能和召回率指标。该研究使用BioSample和GEO存储库来展示这些改进的影响，展示了标准化元数据如何提高检索效果。平均召回率显著提升，从基础原始BioSample和GEO数据集的17.65%提升到我们提出的元数据标准化流程的62.87%。这一发现强调了将先进AI模型与结构化元数据管理工具结合使用在实现更有效和可靠的数据检索方面的变革性影响。 

---
# When Reasoning Meets Compression: Benchmarking Compressed Large Reasoning Models on Complex Reasoning Tasks 

**Title (ZH)**: 当推理遇到压缩：压缩大型推理模型在复杂推理任务上的基准测试 

**Authors**: Nan Zhang, Yusen Zhang, Prasenjit Mitra, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02010)  

**Abstract**: Recent open-source large reasoning models (LRMs) exhibit strong performance on complex reasoning tasks, but their large parameter count makes them prohibitively expensive for individuals. The compression of large language models (LLMs) offers an effective solution to reduce cost of computational resources. However, systematic studies on the performance of compressed LLMs in complex reasoning tasks, especially for LRMs, are lacking. Most works on quantization and pruning focus on preserving language modeling performance, while existing distillation works do not comprehensively benchmark student models based on reasoning difficulty or compression impact on knowledge and reasoning. In this paper, we benchmark compressed DeepSeek-R1 models on four different reasoning datasets (AIME 2024, FOLIO, Temporal Sequences of BIG-Bench Hard, and MuSiQue), ranging from mathematical to multihop reasoning, using quantization, distillation, and pruning methods. We benchmark 2.51-, 1.73-, and 1.58-bit R1 models that adopt dynamic quantization. We also benchmark distilled R1 models that are based on LLaMA or Qwen and run SparseGPT on them to obtain various sparsity levels. Studying the performance and behavior of compressed LRMs, we report their performance scores and test-time compute (number of tokens spent on each question). Notably, using MuSiQue, we find that parameter count has a much greater impact on LRMs' knowledge memorization than on their reasoning capability, which can inform the choice of compression techniques. Through our empirical analysis of test-time compute, we find that shorter model outputs generally achieve better performance than longer ones across several benchmarks for both R1 and its compressed variants, highlighting the need for more concise reasoning chains. 

**Abstract (ZH)**: Recent开源大型推理模型在复杂推理任务中表现出色，但其庞大的参数量使个人使用成本高昂。大型语言模型的压缩为降低计算资源成本提供了有效解决方案。然而，针对压缩大型语言模型在复杂推理任务中的性能研究，特别是针对推理型模型的研究仍然不足。大部分关于量化和剪枝的工作侧重于保留语言建模性能，而现有的蒸馏工作并没有全面基于推理难度或压缩对知识和推理的影响来评估学生模型。在本文中，我们使用量化、蒸馏和剪枝方法，在四个不同推理数据集（AIME 2024、FOLIO、BIG-Bench Hard中的多跳推理和MuSiQue）上对DeepSeek-R1压缩模型进行基准测试，涵盖从数学到多跳推理的任务。我们还测试了基于LLaMA或Qwen的蒸馏R1模型，并使用SparseGPT获得不同稀疏程度。通过研究压缩推理模型的性能和行为，我们报告了它们的性能分数和测试时间计算（每个问题消耗的token数）。值得注意的是，使用MuSiQue，我们发现参数量对推理模型的知识记忆影响更大，而对推理能力的影响相对较小，这可以指导压缩技术的选择。通过实证分析测试时间计算量，我们发现对于R1及其压缩变体，在多个基准上较短的模型输出通常比较长的模型输出具有更好的性能，强调了更简洁推理链的需求。 

---
