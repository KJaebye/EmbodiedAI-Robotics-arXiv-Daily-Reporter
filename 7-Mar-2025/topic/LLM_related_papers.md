# Pretrained LLMs as Real-Time Controllers for Robot Operated Serial Production Line 

**Title (ZH)**: 预训练大语言模型作为实时控制器用于机器人操作的连续生产线 

**Authors**: Muhammad Waseem, Kshitij Bhatta, Chen Li, Qing Chang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03889)  

**Abstract**: The manufacturing industry is undergoing a transformative shift, driven by cutting-edge technologies like 5G, AI, and cloud computing. Despite these advancements, effective system control, which is crucial for optimizing production efficiency, remains a complex challenge due to the intricate, knowledge-dependent nature of manufacturing processes and the reliance on domain-specific expertise. Conventional control methods often demand heavy customization, considerable computational resources, and lack transparency in decision-making. In this work, we investigate the feasibility of using Large Language Models (LLMs), particularly GPT-4, as a straightforward, adaptable solution for controlling manufacturing systems, specifically, mobile robot scheduling. We introduce an LLM-based control framework to assign mobile robots to different machines in robot assisted serial production lines, evaluating its performance in terms of system throughput. Our proposed framework outperforms traditional scheduling approaches such as First-Come-First-Served (FCFS), Shortest Processing Time (SPT), and Longest Processing Time (LPT). While it achieves performance that is on par with state-of-the-art methods like Multi-Agent Reinforcement Learning (MARL), it offers a distinct advantage by delivering comparable throughput without the need for extensive retraining. These results suggest that the proposed LLM-based solution is well-suited for scenarios where technical expertise, computational resources, and financial investment are limited, while decision transparency and system scalability are critical concerns. 

**Abstract (ZH)**: 制造业正经历由5G、AI和云计算等前沿技术驱动的转型。尽管取得了这些进展，有效系统控制——这对于优化生产效率至关重要——依然是一个复杂挑战，因其制造业过程的复杂性和对领域专长的依赖。传统控制方法往往需要大量定制、较多的计算资源，并且在决策透明度方面存在不足。在本研究中，我们探讨使用大型语言模型（LLMs），特别是GPT-4，作为控制制造系统的简便且具有弹性的解决方案，特别应用于协作机器人调度。我们提出了一种基于LLM的控制框架，用于在协作机器人辅助的流水线中分配移动机器人，并评估其在系统吞吐量方面的性能。我们提出的框架在系统吞吐量方面优于传统的调度方法，如先到先服务（FCFS）、最短处理时间（SPT）和最长处理时间（LPT）。它在性能上与最先进的方法（如多代理强化学习MARL）相当，但通过不需大量重新训练即可提供相当的吞吐量，展现出明显的优势。这些结果表明，所提出的基于LLM的解决方案适用于技术专长、计算资源和财务投资有限而决策透明度和系统扩展性是关键考虑的场景。 

---
# Benchmarking Reasoning Robustness in Large Language Models 

**Title (ZH)**: 大规模语言模型中推理稳健性的基准测试 

**Authors**: Tong Yu, Yongcheng Jing, Xikun Zhang, Wentao Jiang, Wenjie Wu, Yingjie Wang, Wenbin Hu, Bo Du, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04550)  

**Abstract**: Despite the recent success of large language models (LLMs) in reasoning such as DeepSeek, we for the first time identify a key dilemma in reasoning robustness and generalization: significant performance degradation on novel or incomplete data, suggesting a reliance on memorized patterns rather than systematic reasoning. Our closer examination reveals four key unique limitations underlying this issue:(1) Positional bias--models favor earlier queries in multi-query inputs but answering the wrong one in the latter (e.g., GPT-4o's accuracy drops from 75.8 percent to 72.8 percent); (2) Instruction sensitivity--performance declines by 5.0 to 7.5 percent in the Qwen2.5 Series and by 5.0 percent in DeepSeek-V3 with auxiliary guidance; (3) Numerical fragility--value substitution sharply reduces accuracy (e.g., GPT-4o drops from 97.5 percent to 82.5 percent, GPT-o1-mini drops from 97.5 percent to 92.5 percent); and (4) Memory dependence--models resort to guesswork when missing critical data. These findings further highlight the reliance on heuristic recall over rigorous logical inference, demonstrating challenges in reasoning robustness. To comprehensively investigate these robustness challenges, this paper introduces a novel benchmark, termed as Math-RoB, that exploits hallucinations triggered by missing information to expose reasoning gaps. This is achieved by an instruction-based approach to generate diverse datasets that closely resemble training distributions, facilitating a holistic robustness assessment and advancing the development of more robust reasoning frameworks. Bad character(s) in field Abstract. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）如DeepSeek在推理方面取得了近期的成功，但我们首次识别出推理稳健性和泛化之间的一个关键困境：在处理新颖或不完整数据时，表现出显著的性能退化，这表明模型依赖于记忆中的模式而非系统的推理。我们进一步的检查揭示了这一问题背后的四个关键独特限制：（1）位置偏差——模型在多查询输入中偏好早期查询，但在后期给出错误的答案（例如，GPT-4o的准确性从75.8%降至72.8%）；（2）指令敏感性——Qwen2.5系列和DeepSeek-V3在辅助指导下性能分别下降5.0%至7.5%和5.0%；（3）数值脆弱性——数值替换显著降低了准确性（例如，GPT-4o从97.5%降至82.5%，GPT-o1-mini从97.5%降至92.5%）；（4）记忆依赖——当缺乏关键数据时，模型会陷入猜测。这些发现进一步突显了对启发式回忆的依赖而非严格的逻辑推理，展示了推理稳健性方面的挑战。为了全面探究这些稳健性挑战，本文引入了一个新的基准，称为Math-RoB，利用缺少信息引发的幻觉来揭示推理缺口。这种评估通过基于指令的方法生成与训练分布紧密相似的多样数据集得以实现，有助于进行全面的稳健性评估，推动更稳健的推理框架的发展。 

---
# SOLAR: Scalable Optimization of Large-scale Architecture for Reasoning 

**Title (ZH)**: SOLAR: 大规模推理架构的可扩展优化 

**Authors**: Chen Li, Yinyi Luo, Anudeep Bolimera, Marios Savvides  

**Link**: [PDF](https://arxiv.org/pdf/2503.04530)  

**Abstract**: Large Language Models (LLMs) excel in reasoning but remain constrained by their Chain-of-Thought (CoT) approach, which struggles with complex tasks requiring more nuanced topological reasoning. We introduce SOLAR, Scalable Optimization of Large-scale Architecture for Reasoning, a framework that dynamically optimizes various reasoning topologies to enhance accuracy and efficiency.
Our Topological Annotation Generation (TAG) system automates topological dataset creation and segmentation, improving post-training and evaluation. Additionally, we propose Topological-Scaling, a reward-driven framework that aligns training and inference scaling, equipping LLMs with adaptive, task-aware reasoning.
SOLAR achieves substantial gains on MATH and GSM8K: +5% accuracy with Topological Tuning, +9% with Topological Reward, and +10.02% with Hybrid Scaling. It also reduces response length by over 5% for complex problems, lowering inference latency.
To foster the reward system, we train a multi-task Topological Reward Model (M-TRM), which autonomously selects the best reasoning topology and answer in a single pass, eliminating the need for training and inference on multiple single-task TRMs (S-TRMs), thus reducing both training cost and inference latency. In addition, in terms of performance, M-TRM surpasses all S-TRMs, improving accuracy by +10% and rank correlation by +9%.
To the best of our knowledge, SOLAR sets a new benchmark for scalable, high-precision LLM reasoning while introducing an automated annotation process and a dynamic reasoning topology competition mechanism. 

**Abstract (ZH)**: 大型语言模型（LLMs）在推理方面表现出色，但受限于其链式思维（CoT）方法，这在处理需要更细腻拓扑推理的复杂任务时存在局限。我们提出了SOLAR框架，即大规模推理架构优化框架，该框架能够动态优化各种推理拓扑结构，以提升准确性和效率。

我们的拓扑注释生成（TAG）系统自动创建和分割拓扑数据集，从而提高训练后评估的效果。此外，我们提出了基于奖励的拓扑缩放框架，该框架能够使训练和推理缩放相匹配，为LLMs提供适应性和任务感知的推理能力。

在MATH和GSM8K上，SOLAR取得了显著的提升：拓扑调优提高了5%的准确率，拓扑奖励提高了9%的准确率，混合缩放提高了10.02%的准确率。同时，对于复杂问题，SOLAR还能减少超过5%的响应长度，降低推理延迟。

为了促进奖励系统，我们训练了一个多任务拓扑奖励模型（M-TRM），该模型能够在一次通过中自主选择最佳推理拓扑结构和答案，消除了对多个单任务奖励模型（S-TRM）进行训练和推理的需要，从而降低了训练成本和推理延迟。此外，从性能角度看，M-TRM超越了所有S-TRM，准确率提高了10%，秩相关度提高了9%。

据我们所知，SOLAR在可扩展和高精度的LLM推理方面设立了一个新的基准，同时引入了自动化注释过程和动态推理拓扑结构竞争机制。 

---
# ToolFuzz -- Automated Agent Tool Testing 

**Title (ZH)**: ToolFuzz -- 自动化代理工具测试 

**Authors**: Ivan Milev, Mislav Balunović, Maximilian Baader, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2503.04479)  

**Abstract**: Large Language Model (LLM) Agents leverage the advanced reasoning capabilities of LLMs in real-world applications. To interface with an environment, these agents often rely on tools, such as web search or database APIs. As the agent provides the LLM with tool documentation along the user query, the completeness and correctness of this documentation is critical. However, tool documentation is often over-, under-, or ill-specified, impeding the agent's accuracy. Standard software testing approaches struggle to identify these errors as they are expressed in natural language. Thus, despite its importance, there currently exists no automated method to test the tool documentation for agents. To address this issue, we present ToolFuzz, the first method for automated testing of tool documentations. ToolFuzz is designed to discover two types of errors: (1) user queries leading to tool runtime errors and (2) user queries that lead to incorrect agent responses. ToolFuzz can generate a large and diverse set of natural inputs, effectively finding tool description errors at a low false positive rate. Further, we present two straightforward prompt-engineering approaches. We evaluate all three tool testing approaches on 32 common LangChain tools and 35 newly created custom tools and 2 novel benchmarks to further strengthen the assessment. We find that many publicly available tools suffer from underspecification. Specifically, we show that ToolFuzz identifies 20x more erroneous inputs compared to the prompt-engineering approaches, making it a key component for building reliable AI agents. 

**Abstract (ZH)**: 大型语言模型代理通过利用大型语言模型的高级推理能力在实际应用中发挥作用。为了与环境交互，这些代理通常依赖于工具，如网络搜索或数据库API。当代理将工具文档连同用户查询一起提供给大型语言模型时，文档的完整性和正确性至关重要。然而，工具文档经常存在过度、不足或表述不清的问题，影响代理的准确性。标准软件测试方法难以识别这些错误，因为这些错误用自然语言表达。因此，尽管这一点至关重要，目前尚不存在自动测试代理工具文档的方法。为解决这一问题，我们提出ToolFuzz，这是第一个用于自动化测试工具文档的方法。ToolFuzz旨在发现两类错误：（1）导致工具运行时错误的用户查询和（2）导致不正确代理响应的用户查询。ToolFuzz能够生成大量多样化的自然输入，有效地在较低的误报率下发现工具描述错误。此外，我们介绍了两个简单的提示工程方法。我们在32个常见的LangChain工具、35个新创建的定制工具以及2个新型基准上评估了所有三种工具测试方法，进一步加强了评估。我们发现许多公开可用的工具存在表述不足的问题。具体而言，我们展示了ToolFuzz比提示工程方法识别出20倍更多的错误输入，使其成为构建可靠AI代理的关键组成部分。 

---
# Activation Space Interventions Can Be Transferred Between Large Language Models 

**Title (ZH)**: 大型语言模型之间可以转移激活空间干预。 

**Authors**: Narmeen Oozeer, Dhruv Nathawani, Nirmalendu Prakash, Michael Lan, Abir Harrasse, Amirali Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2503.04429)  

**Abstract**: The study of representation universality in AI models reveals growing convergence across domains, modalities, and architectures. However, the practical applications of representation universality remain largely unexplored. We bridge this gap by demonstrating that safety interventions can be transferred between models through learned mappings of their shared activation spaces. We demonstrate this approach on two well-established AI safety tasks: backdoor removal and refusal of harmful prompts, showing successful transfer of steering vectors that alter the models' outputs in a predictable way. Additionally, we propose a new task, \textit{corrupted capabilities}, where models are fine-tuned to embed knowledge tied to a backdoor. This tests their ability to separate useful skills from backdoors, reflecting real-world challenges. Extensive experiments across Llama, Qwen and Gemma model families show that our method enables using smaller models to efficiently align larger ones. Furthermore, we demonstrate that autoencoder mappings between base and fine-tuned models can serve as reliable ``lightweight safety switches", allowing dynamic toggling between model behaviors. 

**Abstract (ZH)**: AI模型中表示通用性的研究揭示了跨领域、模态和架构的日益趋同现象。然而，表示通用性的实际应用仍然很大程度上未被探索。我们通过对已建立的安全干预措施进行迁移，展示了通过学习其共享激活空间的映射可以实现这一点。我们在两个已建立的AI安全任务——后门移除和拒绝有害提示上证明了这一方法的有效性，展示了一系列能够以可预测方式改变模型输出方向向量的成功传输。此外，我们提出了一种新任务“受损能力”，其中模型经过微调以嵌入与后门相关的知识。这测试了模型区分有用技能和后门的能力，反映了现实世界中的挑战。广泛的实验表明，我们的方法可以使用较小的模型有效地调整较大的模型。同时，我们展示了从基础模型到微调模型的自动编码器映射可以作为可靠的“轻量级安全开关”，允许动态切换模型行为。 

---
# Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search 

**Title (ZH)**: wider 或者更深？基于自适应分支树搜索的大型语言模型推理时计算扩展 

**Authors**: Kou Misaki, Yuichi Inoue, Yuki Imajuku, So Kuroki, Taishi Nakamura, Takuya Akiba  

**Link**: [PDF](https://arxiv.org/pdf/2503.04412)  

**Abstract**: Recent advances demonstrate that increasing inference-time computation can significantly boost the reasoning capabilities of large language models (LLMs). Although repeated sampling (i.e., generating multiple candidate outputs) is a highly effective strategy, it does not leverage external feedback signals for refinement, which are often available in tasks like coding. In this work, we propose $\textit{Adaptive Branching Monte Carlo Tree Search (AB-MCTS)}$, a novel inference-time framework that generalizes repeated sampling with principled multi-turn exploration and exploitation. At each node in the search tree, AB-MCTS dynamically decides whether to "go wider" by expanding new candidate responses or "go deeper" by revisiting existing ones based on external feedback signals. We evaluate our method on complex coding and engineering tasks using frontier models. Empirical results show that AB-MCTS consistently outperforms both repeated sampling and standard MCTS, underscoring the importance of combining the response diversity of LLMs with multi-turn solution refinement for effective inference-time scaling. 

**Abstract (ZH)**: 最近的研究表明，增加推理时间计算可以显著提升大语言模型（LLMs）的推理能力。尽管重复采样（即生成多个候选输出）是一种非常有效的策略，但它不利用可用外部反馈信号进行细化，而在编程等任务中这些信号通常可用。在本文中，我们提出了一种新颖的推理时框架——自适应分支蒙特卡洛树搜索（AB-MCTS），该框架在有原则的多轮探索和利用基础上泛化了重复采样。在搜索树的每一个节点，AB-MCTS基于外部反馈信号动态决定是“扩大分支”生成新候选响应，还是“深入挖掘”回顾现有响应。我们使用前沿模型在复杂的编程和工程任务上评估了该方法。实验证明，AB-MCTS在多种测试任务中均优于重复采样和标准MCTS方法，突显了结合LLMs的响应多样性和多轮解题细化对于有效推理时间扩展的重要性。 

---
# AgentSafe: Safeguarding Large Language Model-based Multi-agent Systems via Hierarchical Data Management 

**Title (ZH)**: AgentSafe: 通过层次化数据管理保障基于大型语言模型的多代理系统安全 

**Authors**: Junyuan Mao, Fanci Meng, Yifan Duan, Miao Yu, Xiaojun Jia, Junfeng Fang, Yuxuan Liang, Kun Wang, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.04392)  

**Abstract**: Large Language Model based multi-agent systems are revolutionizing autonomous communication and collaboration, yet they remain vulnerable to security threats like unauthorized access and data breaches. To address this, we introduce AgentSafe, a novel framework that enhances MAS security through hierarchical information management and memory protection. AgentSafe classifies information by security levels, restricting sensitive data access to authorized agents. AgentSafe incorporates two components: ThreatSieve, which secures communication by verifying information authority and preventing impersonation, and HierarCache, an adaptive memory management system that defends against unauthorized access and malicious poisoning, representing the first systematic defense for agent memory. Experiments across various LLMs show that AgentSafe significantly boosts system resilience, achieving defense success rates above 80% under adversarial conditions. Additionally, AgentSafe demonstrates scalability, maintaining robust performance as agent numbers and information complexity grow. Results underscore effectiveness of AgentSafe in securing MAS and its potential for real-world application. 

**Abstract (ZH)**: 基于大型语言模型的多代理系统通过层次信息管理和内存保护增强安全性，但仍易受到未授权访问和数据泄露等安全威胁的影响。为此，我们提出了AgentSafe，这是一种新颖的框架，通过层级信息管理和内存保护增强多代理系统的安全性。AgentSafe按安全级别分类信息，限制敏感数据的访问仅限于授权代理。AgentSafe包含两个组件：ThreatSieve，通过验证信息权威性和防止冒充来保障通信安全；HierarCache，一种自适应内存管理系统，防御未授权访问和恶意污染，是首个针对代理内存的系统性防御方案。实验结果表明，在对抗条件下，AgentSafe显著提高了系统的韧性，防御成功率超过80%。此外，AgentSafe展示了可扩展性，在代理数量和信息复杂性增加时仍能维持稳健的性能。实验结果强调了AgentSafe在保障多代理系统安全方面的有效性及其在实际应用中的潜力。 

---
# Mapping AI Benchmark Data to Quantitative Risk Estimates Through Expert Elicitation 

**Title (ZH)**: 通过专家判断将AI基准数据映射到定量风险估计 

**Authors**: Malcolm Murray, Henry Papadatos, Otter Quarks, Pierre-François Gimenez, Simeon Campos  

**Link**: [PDF](https://arxiv.org/pdf/2503.04299)  

**Abstract**: The literature and multiple experts point to many potential risks from large language models (LLMs), but there are still very few direct measurements of the actual harms posed. AI risk assessment has so far focused on measuring the models' capabilities, but the capabilities of models are only indicators of risk, not measures of risk. Better modeling and quantification of AI risk scenarios can help bridge this disconnect and link the capabilities of LLMs to tangible real-world harm. This paper makes an early contribution to this field by demonstrating how existing AI benchmarks can be used to facilitate the creation of risk estimates. We describe the results of a pilot study in which experts use information from Cybench, an AI benchmark, to generate probability estimates. We show that the methodology seems promising for this purpose, while noting improvements that can be made to further strengthen its application in quantitative AI risk assessment. 

**Abstract (ZH)**: 大型语言模型的文献和多位专家指出其存在诸多潜在风险，但直接测量实际危害的研究仍然非常有限。目前对人工智能风险的评估主要集中在测量模型的能力上，但模型的能力只是风险的指示器，而非风险的量化指标。更好地建模和量化人工智能风险情景，有助于弥合这一差距，将大型语言模型的能力与具体的现实危害联系起来。本论文通过展示如何利用现有的人工智能基准来促进风险估计的创建，做出了早期贡献。我们描述了一个试点研究的结果，专家们使用Cybench这一人工智能基准的信息，生成概率估计。我们展示这种方法似乎适用于这一目的，同时指出可以进一步改进的方法，以加强其在定量人工智能风险评估中的应用。 

---
# TIMER: Temporal Instruction Modeling and Evaluation for Longitudinal Clinical Records 

**Title (ZH)**: TIMER： longitudinally临床记录的Temporal指令建模与评估 

**Authors**: Hejie Cui, Alyssa Unell, Bowen Chen, Jason Alan Fries, Emily Alsentzer, Sanmi Koyejo, Nigam Shah  

**Link**: [PDF](https://arxiv.org/pdf/2503.04176)  

**Abstract**: Large language models (LLMs) have emerged as promising tools for assisting in medical tasks, yet processing Electronic Health Records (EHRs) presents unique challenges due to their longitudinal nature. While LLMs' capabilities to perform medical tasks continue to improve, their ability to reason over temporal dependencies across multiple patient visits and time frames remains unexplored. We introduce TIMER (Temporal Instruction Modeling and Evaluation for Longitudinal Clinical Records), a framework that incorporate instruction-response pairs grounding to different parts of a patient's record as a critical dimension in both instruction evaluation and tuning for longitudinal clinical records. We develop TIMER-Bench, the first time-aware benchmark that evaluates temporal reasoning capabilities over longitudinal EHRs, as well as TIMER-Instruct, an instruction-tuning methodology for LLMs to learn reasoning over time. We demonstrate that models fine-tuned with TIMER-Instruct improve performance by 7.3% on human-generated benchmarks and 9.2% on TIMER-Bench, indicating that temporal instruction-tuning improves model performance for reasoning over EHR. 

**Abstract (ZH)**: Large语言模型（LLMs）已成为协助医疗任务的有前途工具，但由于其纵向性质，处理电子健康记录（EHRs）提出了unique挑战。尽管LLMs执行医疗任务的能力不断提升，但其在多个患者访问和时间框架之间推理时间依赖性的能力尚未被探索。我们介绍了TIMER（Time-aware Instruction Modeling and Evaluation for Longitudinal Clinical Records）框架，该框架将指令-响应对与患者记录的不同部分进行关联，作为纵向临床记录指令评估和调优的关键维度。我们开发了TIMER-Bench，这是第一个时间意识基准，用于评估LLMs在纵向EHRs中的时间推理能力，以及TIMER-Instruct，这是一种指令调优方法，使LLMs学习时间推理。我们证明，使用TIMER-Instruct微调的模型在人类生成的基准测试上的性能提高了7.3%，在TIMER-Bench上的性能提高了9.2%，表明时间指令调优可以提高模型在EHR时间推理方面的性能。 

---
# KidneyTalk-open: No-code Deployment of a Private Large Language Model with Medical Documentation-Enhanced Knowledge Database for Kidney Disease 

**Title (ZH)**: KidneyTalk-open:无需代码部署的带医学文档增强知识库的私有大型语言模型用于肾病 

**Authors**: Yongchao Long, Chao Yang, Gongzheng Tang, Jinwei Wang, Zhun Sui, Yuxi Zhou, Shenda Hong, Luxia Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04153)  

**Abstract**: Privacy-preserving medical decision support for kidney disease requires localized deployment of large language models (LLMs) while maintaining clinical reasoning capabilities. Current solutions face three challenges: 1) Cloud-based LLMs pose data security risks; 2) Local model deployment demands technical expertise; 3) General LLMs lack mechanisms to integrate medical knowledge. Retrieval-augmented systems also struggle with medical document processing and clinical usability. We developed KidneyTalk-open, a desktop system integrating three technical components: 1) No-code deployment of state-of-the-art (SOTA) open-source LLMs (such as DeepSeek-r1, Qwen2.5) via local inference engine; 2) Medical document processing pipeline combining context-aware chunking and intelligent filtering; 3) Adaptive Retrieval and Augmentation Pipeline (AddRep) employing agents collaboration for improving the recall rate of medical documents. A graphical interface was designed to enable clinicians to manage medical documents and conduct AI-powered consultations without technical expertise. Experimental validation on 1,455 challenging nephrology exam questions demonstrates AddRep's effectiveness: achieving 29.1% accuracy (+8.1% over baseline) with intelligent knowledge integration, while maintaining robustness through 4.9% rejection rate to suppress hallucinations. Comparative case studies with the mainstream products (AnythingLLM, Chatbox, GPT4ALL) demonstrate KidneyTalk-open's superior performance in real clinical query. KidneyTalk-open represents the first no-code medical LLM system enabling secure documentation-enhanced medical Q&A on desktop. Its designs establishes a new framework for privacy-sensitive clinical AI applications. The system significantly lowers technical barriers while improving evidence traceability, enabling more medical staff or patients to use SOTA open-source LLMs conveniently. 

**Abstract (ZH)**: 隐私保护的肾脏疾病医疗决策支持需要本地部署大型语言模型（LLMs）以保持临床推理能力。当前解决方案面临三大挑战：1）基于云的LLMs存在数据安全风险；2）本地模型部署需要专业技术；3）通用LLMs缺乏整合医学知识的机制。检索增强系统在医学文档处理和临床应用方面也存在困难。我们开发了KidneyTalk-open，这是一种桌面系统，结合了三个技术组件：1）通过本地推理引擎无代码部署最先进的开源LLMs（如DeepSeek-r1、Qwen2.5）；2）结合上下文感知切块和智能过滤的医学文档处理流水线；3）采用代理协作的自适应检索和增强流水线（AddRep），旨在提高医学文档召回率。设计了图形界面，使临床工作者能够在无需技术支持的情况下管理和进行基于AI的咨询。针对1,455个具有挑战性的肾病学考试问题的实验验证表明AddRep的有效性：通过智能知识整合实现29.1%的准确率（基线基础上提高8.1%），同时通过4.9%的拒绝率来保持鲁棒性以抑制幻觉。与主流产品（AnythingLLM、Chatbox、GPT4ALL）进行的对比案例研究显示，KidneyTalk-open在实际临床查询方面表现更优。KidneyTalk-open代表了首个无代码医学LLMs系统，使其能够在桌面上实现安全文档增强的医疗问答。其设计为隐私敏感的临床AI应用建立了新的框架。该系统显著降低技术障碍，提高证据追溯性，使更多医疗人员或患者能够方便地使用最先进的开源LLMs。 

---
# L$^2$M: Mutual Information Scaling Law for Long-Context Language Modeling 

**Title (ZH)**: L$^2$M：长上下文语言建模的互信息缩放律 

**Authors**: Zhuo Chen, Oriol Mayné i Comas, Zhuotao Jin, Di Luo, Marin Soljačić  

**Link**: [PDF](https://arxiv.org/pdf/2503.04725)  

**Abstract**: We rigorously establish a bipartite mutual information scaling law in natural language that governs long-range dependencies. This scaling law, which we show is distinct from and scales independently of the conventional two-point mutual information, is the key to understanding long-context language modeling. Using this scaling law, we formulate the Long-context Language Modeling (L$^2$M) condition, which relates a model's capacity for effective long context length modeling to the scaling of its latent state size for storing past information. Our results are validated through experiments on both transformers and state space models. This work establishes a theoretical foundation that guides the development of large language models toward longer context lengths. 

**Abstract (ZH)**: 我们严谨地建立了自然语言中调控长距离依赖的二分互信息缩放定律。这一缩放定律与传统的两点互信息不同且独立缩放，是理解长上下文语言建模的关键。利用这一缩放定律，我们提出了长上下文语言建模（L$^2$M）条件，该条件将模型有效建模长上下文长度的能力与其潜状态大小以存储过去信息的缩放关系联系起来。我们的结果通过Transformer和状态空间模型上的实验得到了验证。这项工作建立了一个理论基础，指导大型语言模型向更长的上下文长度发展。 

---
# Shifting Long-Context LLMs Research from Input to Output 

**Title (ZH)**: 从输入转向输出：长上下文LLM研究的转变 

**Authors**: Yuhao Wu, Yushi Bai, Zhiqing Hu, Shangqing Tu, Ming Shan Hee, Juanzi Li, Roy Ka-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.04723)  

**Abstract**: Recent advancements in long-context Large Language Models (LLMs) have primarily concentrated on processing extended input contexts, resulting in significant strides in long-context comprehension. However, the equally critical aspect of generating long-form outputs has received comparatively less attention. This paper advocates for a paradigm shift in NLP research toward addressing the challenges of long-output generation. Tasks such as novel writing, long-term planning, and complex reasoning require models to understand extensive contexts and produce coherent, contextually rich, and logically consistent extended text. These demands highlight a critical gap in current LLM capabilities. We underscore the importance of this under-explored domain and call for focused efforts to develop foundational LLMs tailored for generating high-quality, long-form outputs, which hold immense potential for real-world applications. 

**Abstract (ZH)**: Recent advancements in long-context Large Language Models (LLMs) have mainly focused on processing extended input contexts, leading to significant progress in long-context comprehension. However, the equally crucial aspect of generating long-form outputs has received relatively less attention. This paper advocates for a paradigm shift in NLP research towards addressing the challenges of long-output generation. Tasks such as novel writing, long-term planning, and complex reasoning require models to understand extensive contexts and produce coherent, contextually rich, and logically consistent extended text. These demands highlight a critical gap in current LLM capabilities. We emphasize the importance of this under-explored domain and call for focused efforts to develop foundational LLMs tailored for generating high-quality, long-form outputs, which hold immense potential for real-world applications. 

---
# Enough Coin Flips Can Make LLMs Act Bayesian 

**Title (ZH)**: 足够的硬币翻转可以使大语言模型表现得像贝叶斯模型。 

**Authors**: Ritwik Gupta, Rodolfo Corona, Jiaxin Ge, Eric Wang, Dan Klein, Trevor Darrell, David M. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2503.04722)  

**Abstract**: Large language models (LLMs) exhibit the ability to generalize given few-shot examples in their input prompt, an emergent capability known as in-context learning (ICL). We investigate whether LLMs utilize ICL to perform structured reasoning in ways that are consistent with a Bayesian framework or rely on pattern matching. Using a controlled setting of biased coin flips, we find that: (1) LLMs often possess biased priors, causing initial divergence in zero-shot settings, (2) in-context evidence outweighs explicit bias instructions, (3) LLMs broadly follow Bayesian posterior updates, with deviations primarily due to miscalibrated priors rather than flawed updates, and (4) attention magnitude has negligible effect on Bayesian inference. With sufficient demonstrations of biased coin flips via ICL, LLMs update their priors in a Bayesian manner. 

**Abstract (ZH)**: 大型语言模型通过Few-shot示例在其输入提示中展现出迁移学习的能力，这种新兴能力称为上下文内学习（ICL）。我们探讨了LLMs是通过符合贝叶斯框架的方式进行结构化推理，还是依赖于模式匹配来进行ICL。通过有偏硬币翻转的受控实验设置，我们发现：（1）LLMs经常具有偏见的先验，导致零-shot设置下的初始偏差；（2）上下文内的证据超过了明确的偏见指令；（3）LLMs普遍遵循贝叶斯后验更新，偏差主要源于先验的偏差而非更新过程有缺陷；（4）注意强度对贝叶斯推理影响不大。通过ICL充分展示有偏硬币翻转，LLMs以贝叶斯方式更新其先验。 

---
# Predictable Scale: Part I -- Optimal Hyperparameter Scaling Law in Large Language Model Pretraining 

**Title (ZH)**: 可预测的规模：第一部分——大规模语言模型预训练的最优超参数缩放定律 

**Authors**: Houyi Li, Wenzheng Zheng, Jingcheng Hu, Qiufeng Wang, Hanshan Zhang, Zili Wang, Yangshijie Xu, Shuigeng Zhou, Xiangyu Zhang, Daxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04715)  

**Abstract**: The impressive capabilities of Large Language Models (LLMs) across diverse tasks are now well-established, yet their effective deployment necessitates careful hyperparameter optimization. Through extensive empirical studies involving grid searches across diverse configurations, we discover universal scaling laws governing these hyperparameters: optimal learning rate follows a power-law relationship with both model parameters and data sizes, while optimal batch size scales primarily with data sizes. Our analysis reveals a convex optimization landscape for hyperparameters under fixed models and data size conditions. This convexity implies an optimal hyperparameter plateau. We contribute a universal, plug-and-play optimal hyperparameter tool for the community. Its estimated values on the test set are merely 0.07\% away from the globally optimal LLM performance found via an exhaustive search. These laws demonstrate remarkable robustness across variations in model sparsity, training data distribution, and model shape. To our best known, this is the first work that unifies different model shapes and structures, such as Mixture-of-Experts models and dense transformers, as well as establishes optimal hyperparameter scaling laws across diverse data distributions. This exhaustive optimization process demands substantial computational resources, utilizing nearly one million NVIDIA H800 GPU hours to train 3,700 LLMs of varying sizes and hyperparameters from scratch and consuming approximately 100 trillion tokens in total. To facilitate reproducibility and further research, we will progressively release all loss measurements and model checkpoints through our designated repository this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）在多元任务上的 impressive 能力现在已经得到了广泛证实，但其有效的部署需要精细的超参数优化。通过广泛的实证研究，我们发现这些超参数的通用标度法则：最优学习率与模型参数和数据量遵循幂律关系，而最优批次大小主要与数据量相关。我们的分析揭示，在固定模型和数据量条件下，超参数的优化景观是凹的，并暗示了一个最优超参数平台。我们为社区贡献了一个通用的插拔即用型最优超参数工具。其在测试集上的估计值与通过 exhaustive 搜索找到的全局最优 LLM 性能相差仅为 0.07%。这些法则在模型稀疏性、训练数据分布和模型结构的变化下表现出显著的稳健性。据我们所知，这是首次将不同模型结构（如 Mixture-of-Experts 模型和密集型变压器）统一起来，并建立了跨不同数据分布的最优超参数标度法则的研究工作。这个优化过程耗用了大量计算资源，利用近一百万 NVIDIA H800 GPU 小时训练了 3,700 个不同规模和超参数的 LLM，并总共消耗了约 100 万亿个令牌。为了便于再现和进一步研究，我们将在指定的存储库中逐步发布所有损失度量和模型检查点（详见该链接）。 

---
# Universality of Layer-Level Entropy-Weighted Quantization Beyond Model Architecture and Size 

**Title (ZH)**: 超越模型架构和大小的层级熵加权量化通用性 

**Authors**: Alireza Behtash, Marijan Fofonjka, Ethan Baird, Tyler Mauer, Hossein Moghimifam, David Stout, Joel Dennison  

**Link**: [PDF](https://arxiv.org/pdf/2503.04704)  

**Abstract**: We present a novel approach to selective model quantization that transcends the limitations of architecture-specific and size-dependent compression methods for Large Language Models (LLMs) using Entropy-Weighted Quantization (EWQ). By analyzing the entropy distribution across transformer blocks, EWQ determines which blocks can be safely quantized without causing significant performance degradation, independent of model architecture or size. Our method outperforms uniform quantization approaches, maintaining Massive Multitask Language Understanding (MMLU) accuracy scores within 0.5% of unquantized models while reducing memory usage by up to 18%. We demonstrate the effectiveness of EWQ across multiple architectures-from 1.6B to 70B parameters-showcasing consistent improvements in the quality-compression trade-off regardless of model scale or architectural design. A surprising finding of EWQ is its ability to reduce perplexity compared to unquantized models, suggesting the presence of beneficial regularization through selective precision reduction. This improvement holds across different model families, indicating a fundamental relationship between layer-level entropy and optimal precision requirements. Additionally, we introduce FastEWQ, a rapid method for entropy distribution analysis that eliminates the need for loading model weights. This technique leverages universal characteristics of entropy distribution that persist across various architectures and scales, enabling near-instantaneous quantization decisions while maintaining 80% classification accuracy with full entropy analysis. Our results demonstrate that effective quantization strategies can be developed independently of specific architectural choices or model sizes, opening new possibilities for efficient LLM deployment. 

**Abstract (ZH)**: 基于熵加权量化的一种新型大型语言模型选择性模型量化方法 

---
# L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning 

**Title (ZH)**: L1: 用强化学习控制推理模型思考的时间长短 

**Authors**: Pranjal Aggarwal, Sean Welleck  

**Link**: [PDF](https://arxiv.org/pdf/2503.04697)  

**Abstract**: Reasoning language models have shown an uncanny ability to improve performance at test-time by ``thinking longer''-that is, by generating longer chain-of-thought sequences and hence using more compute. However, the length of their chain-of-thought reasoning is not controllable, making it impossible to allocate test-time compute to achieve a desired level of performance. We introduce Length Controlled Policy Optimization (LCPO), a simple reinforcement learning method that optimizes for accuracy and adherence to user-specified length constraints. We use LCPO to train L1, a reasoning language model that produces outputs satisfying a length constraint given in its prompt. L1's length control allows for smoothly trading off computational cost and accuracy on a wide range of tasks, and outperforms the state-of-the-art S1 method for length control. Furthermore, we uncover an unexpected short chain-of-thought capability in models trained with LCPO. For instance, our 1.5B L1 model surpasses GPT-4o at equal reasoning lengths. Overall, LCPO enables precise control over reasoning length, allowing for fine-grained allocation of test-time compute and accuracy. We release code and models at this https URL 

**Abstract (ZH)**: Length Controlled Policy Optimization: Enabling Precise Control over Reasoning Length in Language Models 

---
# Implicit Cross-Lingual Rewarding for Efficient Multilingual Preference Alignment 

**Title (ZH)**: 隐式跨语言奖励以实现高效的多语言偏好对齐 

**Authors**: Wen Yang, Junhong Wu, Chen Wang, Chengqing Zong, Jiajun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04647)  

**Abstract**: Direct Preference Optimization (DPO) has become a prominent method for aligning Large Language Models (LLMs) with human preferences. While DPO has enabled significant progress in aligning English LLMs, multilingual preference alignment is hampered by data scarcity. To address this, we propose a novel approach that $\textit{captures}$ learned preferences from well-aligned English models by implicit rewards and $\textit{transfers}$ them to other languages through iterative training. Specifically, we derive an implicit reward model from the logits of an English DPO-aligned model and its corresponding reference model. This reward model is then leveraged to annotate preference relations in cross-lingual instruction-following pairs, using English instructions to evaluate multilingual responses. The annotated data is subsequently used for multilingual DPO fine-tuning, facilitating preference knowledge transfer from English to other languages. Fine-tuning Llama3 for two iterations resulted in a 12.72% average improvement in Win Rate and a 5.97% increase in Length Control Win Rate across all training languages on the X-AlpacaEval leaderboard. Our findings demonstrate that leveraging existing English-aligned models can enable efficient and effective multilingual preference alignment, significantly reducing the need for extensive multilingual preference data. The code is available at this https URL 

**Abstract (ZH)**: 直接偏好优化(DPO)已成为将大型语言模型(LLMs)与人类偏好对齐的 prominent 方法。尽管DPO已在对齐英语LLMs方面取得了显著进展，但多语言偏好对齐受限于数据稀缺性。为解决这一问题，我们提出了一种新方法，该方法通过隐式奖励 $\textit{捕获}$ 已对齐的英语模型中的偏好，并通过迭代训练 $\textit{转移}$ 至其他语言。具体来说，我们从英语DPO对齐模型及其对应参考模型的logits中推导出一个隐式奖励模型。然后利用该奖励模型对跨语言指令跟随配对中的偏好关系进行标注，使用英语指令评估多语言响应。标注的数据随后用于多语言DPO微调，促进偏好知识从英语向其他语言的转移。对Llama3进行两轮微调后，我们在X-AlpacaEval排行榜上所有训练语言中平均提高了12.72%的胜率，并提高了5.97%的长度控制胜率。我们的研究结果表明，利用现有的英语对齐模型可以实现高效的多语言偏好对齐，显著减少了对大量多语言偏好数据的需求。代码可在以下链接获取：this https URL。 

---
# Mark Your LLM: Detecting the Misuse of Open-Source Large Language Models via Watermarking 

**Title (ZH)**: 标记你的LLM：通过水印技术检测开源大规模语言模型的滥用 

**Authors**: Yijie Xu, Aiwei Liu, Xuming Hu, Lijie Wen, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.04636)  

**Abstract**: As open-source large language models (LLMs) like Llama3 become more capable, it is crucial to develop watermarking techniques to detect their potential misuse. Existing watermarking methods either add watermarks during LLM inference, which is unsuitable for open-source LLMs, or primarily target classification LLMs rather than recent generative LLMs. Adapting these watermarks to open-source LLMs for misuse detection remains an open challenge. This work defines two misuse scenarios for open-source LLMs: intellectual property (IP) violation and LLM Usage Violation. Then, we explore the application of inference-time watermark distillation and backdoor watermarking in these contexts. We propose comprehensive evaluation methods to assess the impact of various real-world further fine-tuning scenarios on watermarks and the effect of these watermarks on LLM performance. Our experiments reveal that backdoor watermarking could effectively detect IP Violation, while inference-time watermark distillation is applicable in both scenarios but less robust to further fine-tuning and has a more significant impact on LLM performance compared to backdoor watermarking. Exploring more advanced watermarking methods for open-source LLMs to detect their misuse should be an important future direction. 

**Abstract (ZH)**: 开源大规模语言模型的滥用检测 watermarking 技术：针对 Llama3 等开源大语言模型的知识产权违规和模型使用违规的滥用检测方法探究 

---
# The Next Frontier of LLM Applications: Open Ecosystems and Hardware Synergy 

**Title (ZH)**: LLM应用的下一个前沿：开放生态系统与硬件协同 

**Authors**: Xinyi Hou, Yanjie Zhao, Haoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04596)  

**Abstract**: Large Language Model (LLM) applications, including LLM app stores and autonomous agents, are shaping the future of AI ecosystems. However, platform silos, fragmented hardware integration, and the absence of standardized interfaces limit scalability, interoperability, and resource efficiency. While LLM app stores democratize AI, their closed ecosystems restrict modular AI reuse and cross-platform portability. Meanwhile, agent-based frameworks offer flexibility but often lack seamless integration across diverse environments. This paper envisions the future of LLM applications and proposes a three-layer decoupled architecture grounded in software engineering principles such as layered system design, service-oriented architectures, and hardware-software co-design. This architecture separates application logic, communication protocols, and hardware execution, enhancing modularity, efficiency, and cross-platform compatibility. Beyond architecture, we highlight key security and privacy challenges for safe, scalable AI deployment and outline research directions in software and security engineering. This vision aims to foster open, secure, and interoperable LLM ecosystems, guiding future advancements in AI applications. 

**Abstract (ZH)**: 大型语言模型（LLM）应用，包括LLM应用商店和自主代理，正在塑造未来的AI生态系统。然而，平台孤岛、碎片化的硬件集成以及缺乏标准化接口限制了可扩展性、兼容性和资源效率。虽然LLM应用商店使AI民主化，但它们封闭的生态系统限制了模块化AI的重用和跨平台移植。与此同时，基于代理的框架提供了灵活性，但通常跨不同环境的无缝集成能力不足。本文设想了LLM应用的未来，并提出了一种三层解耦架构，该架构基于分层系统设计、面向服务的架构和硬件-软件协同设计等软件工程原则。该架构将应用程序逻辑、通信协议和硬件执行分层，增强模块化、效率和跨平台兼容性。展望架构之外，我们强调了安全和隐私的关键挑战，以确保大规模AI部署的安全性和可扩展性，并概述了软件和安全工程方向的研究方向。该愿景旨在促进开放、安全和兼容的LLM生态系统，指导未来AI应用的发展。 

---
# Compositional Causal Reasoning Evaluation in Language Models 

**Title (ZH)**: 语言模型中的组合因果推理评估 

**Authors**: Jacqueline R. M. A. Maasch, Alihan Hüyük, Xinnuo Xu, Aditya V. Nori, Javier Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2503.04556)  

**Abstract**: Causal reasoning and compositional reasoning are two core aspirations in generative AI. Measuring the extent of these behaviors requires principled evaluation methods. We explore a unified perspective that considers both behaviors simultaneously, termed compositional causal reasoning (CCR): the ability to infer how causal measures compose and, equivalently, how causal quantities propagate through graphs. We instantiate a framework for the systematic evaluation of CCR for the average treatment effect and the probability of necessity and sufficiency. As proof of concept, we demonstrate the design of CCR tasks for language models in the LLama, Phi, and GPT families. On a math word problem, our framework revealed a range of taxonomically distinct error patterns. Additionally, CCR errors increased with the complexity of causal paths for all models except o1. 

**Abstract (ZH)**: 生成式AI中的因果推理和组合推理是两个核心目标。衡量这些行为的程度需要原则性的评估方法。我们探索了一个同时考虑这两种行为的统一视角，称为组合因果推理（CCR）：推断因果度量如何组合的能力，等价地，因果量如何在图中传播。我们建立了一个系统评估CCR框架，用于平均处理效应和必要性和充分性的概率。作为概念证明，我们为LLama、Phi和GPT家族的语言模型设计了CCR任务。在数学文字题上，我们的框架揭示了分类不同的错误模式。此外，除了o1模型外，CCR错误随着因果路径复杂性的增加而增加。 

---
# Keeping Yourself is Important in Downstream Tuning Multimodal Large Language Model 

**Title (ZH)**: 保持自身重要性在多模态大型语言模型下游调优中 

**Authors**: Wenke Huang, Jian Liang, Xianda Guo, Yiyang Fang, Guancheng Wan, Xuankun Rong, Chi Wen, Zekun Shi, Qingyun Li, Didi Zhu, Yanbiao Ma, Ke Liang, Bin Yang, He Li, Jiawei Shao, Mang Ye, Bo Du  

**Link**: [PDF](https://arxiv.org/pdf/2503.04543)  

**Abstract**: Multi-modal Large Language Models (MLLMs) integrate visual and linguistic reasoning to address complex tasks such as image captioning and visual question answering. While MLLMs demonstrate remarkable versatility, MLLMs appears limited performance on special applications. But tuning MLLMs for downstream tasks encounters two key challenges: Task-Expert Specialization, where distribution shifts between pre-training and target datasets constrain target performance, and Open-World Stabilization, where catastrophic forgetting erases the model general knowledge. In this work, we systematically review recent advancements in MLLM tuning methodologies, classifying them into three paradigms: (I) Selective Tuning, (II) Additive Tuning, and (III) Reparameterization Tuning. Furthermore, we benchmark these tuning strategies across popular MLLM architectures and diverse downstream tasks to establish standardized evaluation analysis and systematic tuning principles. Finally, we highlight several open challenges in this domain and propose future research directions. To facilitate ongoing progress in this rapidly evolving field, we provide a public repository that continuously tracks developments: this https URL. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）通过整合视觉和语言推理来应对图像caption和视觉问答等复杂任务。尽管MLLMs表现出色，但在特定应用上显得性能有限。对MLLMs进行下游任务微调会遇到两个关键挑战：任务专家专业化，即从预训练到目标数据集的分布变化限制了目标性能，以及开放世界稳定化，即灾难性遗忘消除了模型的通用知识。在本文中，我们系统地回顾了MLLMs微调方法的最新进展，将其分类为三个范式：（I）选择性微调，（II）添加剂微调，以及（III）重参数化微调。此外，我们在流行MLLM架构和多样化的下游任务上对这些微调策略进行了基准测试，以建立标准化的评估分析和系统化的微调原则。最后，我们指出了该领域的一些开放挑战并提出了未来研究方向。为了促进这一快速发展的领域的持续进步，我们提供了一个公共仓库，持续追踪进展：https://this.url。 

---
# Generalized Interpolating Discrete Diffusion 

**Title (ZH)**: 广义插值离散扩散 

**Authors**: Dimitri von Rütte, Janis Fluri, Yuhui Ding, Antonio Orvieto, Bernhard Schölkopf, Thomas Hofmann  

**Link**: [PDF](https://arxiv.org/pdf/2503.04482)  

**Abstract**: While state-of-the-art language models achieve impressive results through next-token prediction, they have inherent limitations such as the inability to revise already generated tokens. This has prompted exploration of alternative approaches such as discrete diffusion. However, masked diffusion, which has emerged as a popular choice due to its simplicity and effectiveness, reintroduces this inability to revise words. To overcome this, we generalize masked diffusion and derive the theoretical backbone of a family of general interpolating discrete diffusion (GIDD) processes offering greater flexibility in the design of the noising processes. Leveraging a novel diffusion ELBO, we achieve compute-matched state-of-the-art performance in diffusion language modeling. Exploiting GIDD's flexibility, we explore a hybrid approach combining masking and uniform noise, leading to improved sample quality and unlocking the ability for the model to correct its own mistakes, an area where autoregressive models notoriously have struggled. Our code and models are open-source: this https URL 

**Abstract (ZH)**: 虽然最先进的语言模型通过下一个词预测取得了令人印象深刻的成果，但它们固有的局限性，如无法修订已生成的词，促使人们探索替代方法，如离散扩散。然而，由于其简单性和有效性，掩码扩散成为了流行的选项，但它重新引入了无法修订词的问题。为了解决这一问题，我们对掩码扩散进行了泛化，并推导出了一种通用插值离散扩散（GIDD）过程的理论基础，提供了在噪声过程设计方面的更大灵活性。通过利用一种新颖的扩散ELBO，我们在扩散语言模型中实现了计算量匹配的最优性能。利用GIDD的灵活性，我们探索了结合掩码和均匀噪声的混合方法，提高了样本质量，并使模型能够纠正自己的错误，这是自回归模型历来难以做到的。我们的代码和模型为开源：[此链接]。 

---
# DAST: Difficulty-Adaptive Slow-Thinking for Large Reasoning Models 

**Title (ZH)**: DAST：针对大规模推理模型的难度自适应慢思考方法 

**Authors**: Yi Shen, Jian Zhang, Jieyun Huang, Shuming Shi, Wenjing Zhang, Jiangze Yan, Ning Wang, Kai Wang, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2503.04472)  

**Abstract**: Recent advancements in slow-thinking reasoning models have shown exceptional performance in complex reasoning tasks. However, these models often exhibit overthinking-generating redundant reasoning steps for simple problems, leading to excessive computational resource usage. While current mitigation strategies uniformly reduce reasoning tokens, they risk degrading performance on challenging tasks that require extended reasoning. This paper introduces Difficulty-Adaptive Slow-Thinking (DAST), a novel framework that enables models to autonomously adjust the length of Chain-of-Thought(CoT) based on problem difficulty. We first propose a Token Length Budget (TLB) metric to quantify difficulty, then leveraging length-aware reward shaping and length preference optimization to implement DAST. DAST penalizes overlong responses for simple tasks while incentivizing sufficient reasoning for complex problems. Experiments on diverse datasets and model scales demonstrate that DAST effectively mitigates overthinking (reducing token usage by over 30\% on average) while preserving reasoning accuracy on complex problems. 

**Abstract (ZH)**: 近期在缓慢思考推理模型方面的进展在复杂推理任务中表现出卓越的性能。然而，这些模型往往为简单的問題生成冗余的推理步骤，导致大量计算资源的浪费。虽然当前的缓解策略会均匀减少推理令牌，但这也可能在需要长时间推理的挑战性任务中降低性能。本文引入了难度自适应缓慢思考（DAST），这是一种新型框架，使模型能够根据问题的难度自主调整推理链（CoT）的长度。我们首先提出了一种令牌长度预算（TLB）指标来量化难度，然后利用长度感知的奖励塑造和长度偏好优化来实现DAST。DAST在简单任务中惩罚过长的响应，而在复杂问题中激励充分的推理。实验结果表明，DAST在减少令牌使用量（平均减少超过30%）的同时，能够有效地缓解过度思考并保持复杂问题的推理准确性。 

---
# Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling 

**Title (ZH)**: 投机MoE：基于投机令牌和专家预调度的通信高效并行MoE推理 

**Authors**: Yan Li, Pengfei Zheng, Shuang Chen, Zewei Xu, Yunfei Du, Zhengang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04398)  

**Abstract**: MoE (Mixture of Experts) prevails as a neural architecture that can scale modern transformer-based LLMs (Large Language Models) to unprecedented scales. Nevertheless, large MoEs' great demands of computing power, memory capacity and memory bandwidth make scalable serving a fundamental challenge and efficient parallel inference has become a requisite to attain adequate throughput under latency constraints. DeepSpeed-MoE, one state-of-the-art MoE inference framework, adopts a 3D-parallel paradigm including EP (Expert Parallelism), TP (Tensor Parallel) and DP (Data Parallelism). However, our analysis shows DeepSpeed-MoE's inference efficiency is largely bottlenecked by EP, which is implemented with costly all-to-all collectives to route token activation. Our work aims to boost DeepSpeed-MoE by strategically reducing EP's communication overhead with a technique named Speculative MoE. Speculative MoE has two speculative parallelization schemes, speculative token shuffling and speculative expert grouping, which predict outstanding tokens' expert routing paths and pre-schedule tokens and experts across devices to losslessly trim EP's communication volume. Besides DeepSpeed-MoE, we also build Speculative MoE into a prevailing MoE inference engine SGLang. Experiments show Speculative MoE can significantly boost state-of-the-art MoE inference frameworks on fast homogeneous and slow heterogeneous interconnects. 

**Abstract (ZH)**: MoE（专家混合）作为能够将现代基于Transformer的大语言模型扩展到前所未有的玩家神经架构，继续占据主导地位。然而，大型MoE对计算能力、内存容量和内存带宽的庞大需求使得可扩展的服务成为一个基本挑战，高效的并行推断已成为在延迟约束下实现适当吞吐量的必要条件。DeepSpeed-MoE，一种最先进的MoE推断框架，采用包括EP（专家并行）、TP（张量并行）和DP（数据并行）的3D并行 paradigm。然而，我们的分析显示DeepSpeed-MoE的推断效率受到EP的严重影响，EP通过昂贵的全对全集合通信来路由token激活。我们通过一种名为Speculative MoE的技术旨在提升DeepSpeed-MoE，通过策略性减少EP的通信开销。Speculative MoE有两种推测性并行方案：推测性token混排和推测性专家分组，这些方案预测出色token的专家路由路径，并预调度token和专家跨设备以无损地削减EP的通信量。除了DeepSpeed-MoE之外，我们还将Speculative MoE整合进一种流行的MoE推断引擎SGLang。实验表明，Speculative MoE能够在快速同构和慢速异构互联环境下显著提升最先进的MoE推断框架的性能。 

---
# Solving Word-Sense Disambiguation and Word-Sense Induction with Dictionary Examples 

**Title (ZH)**: 利用词典例证解决词义消歧和词义归纳问题 

**Authors**: Tadej Škvorc, Marko Robnik-Šikonja  

**Link**: [PDF](https://arxiv.org/pdf/2503.04328)  

**Abstract**: Many less-resourced languages struggle with a lack of large, task-specific datasets that are required for solving relevant tasks with modern transformer-based large language models (LLMs). On the other hand, many linguistic resources, such as dictionaries, are rarely used in this context despite their large information contents. We show how LLMs can be used to extend existing language resources in less-resourced languages for two important tasks: word-sense disambiguation (WSD) and word-sense induction (WSI). We approach the two tasks through the related but much more accessible word-in-context (WiC) task where, given a pair of sentences and a target word, a classification model is tasked with predicting whether the sense of a given word differs between sentences. We demonstrate that a well-trained model for this task can distinguish between different word senses and can be adapted to solve the WSD and WSI tasks. The advantage of using the WiC task, instead of directly predicting senses, is that the WiC task does not need pre-constructed sense inventories with a sufficient number of examples for each sense, which are rarely available in less-resourced languages. We show that sentence pairs for the WiC task can be successfully generated from dictionary examples using LLMs. The resulting prediction models outperform existing models on WiC, WSD, and WSI tasks. We demonstrate our methodology on the Slovene language, where a monolingual dictionary is available, but word-sense resources are tiny. 

**Abstract (ZH)**: 少资源语言中大型语言模型在词汇义消歧和词汇义归纳任务中的资源扩展研究 

---
# Malware Detection at the Edge with Lightweight LLMs: A Performance Evaluation 

**Title (ZH)**: 边缘处基于轻量级LLM的恶意软件检测：性能评估 

**Authors**: Christian Rondanini, Barbara Carminati, Elena Ferrari, Antonio Gaudiano, Ashish Kundu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04302)  

**Abstract**: The rapid evolution of malware attacks calls for the development of innovative detection methods, especially in resource-constrained edge computing. Traditional detection techniques struggle to keep up with modern malware's sophistication and adaptability, prompting a shift towards advanced methodologies like those leveraging Large Language Models (LLMs) for enhanced malware detection. However, deploying LLMs for malware detection directly at edge devices raises several challenges, including ensuring accuracy in constrained environments and addressing edge devices' energy and computational limits. To tackle these challenges, this paper proposes an architecture leveraging lightweight LLMs' strengths while addressing limitations like reduced accuracy and insufficient computational power. To evaluate the effectiveness of the proposed lightweight LLM-based approach for edge computing, we perform an extensive experimental evaluation using several state-of-the-art lightweight LLMs. We test them with several publicly available datasets specifically designed for edge and IoT scenarios and different edge nodes with varying computational power and characteristics. 

**Abstract (ZH)**: 快速演变的恶意软件攻击促使开发创新检测方法，特别是在资源受限的边缘计算环境中。传统的检测技术难以跟上现代恶意软件的高度复杂性和适应性，推动了向高级方法的转变，例如利用大型语言模型（LLMs）来增强恶意软件检测。然而，直接在边缘设备上部署LLMs用于恶意软件检测也面临着诸多挑战，包括在受限环境中确保准确性以及解决边缘设备的能源和计算能力限制。为了应对这些挑战，本文提出了一种架构，利用轻量级LLMs的优势并解决了如准确性降低和计算能力不足等问题。为了评估基于轻量级LLMs的边缘计算检测方法的有效性，我们使用多种最先进的轻量级LLMs进行了广泛的实验评估。我们使用专门为边缘和物联网场景设计的多个公开数据集以及不同计算能力和特性的边缘节点进行测试。 

---
# How to Mitigate Overfitting in Weak-to-strong Generalization? 

**Title (ZH)**: 如何减轻从弱到强泛化的过拟合？ 

**Authors**: Junhao Shi, Qinyuan Cheng, Zhaoye Fei, Yining Zheng, Qipeng Guo, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04249)  

**Abstract**: Aligning powerful AI models on tasks that surpass human evaluation capabilities is the central problem of \textbf{superalignment}. To address this problem, weak-to-strong generalization aims to elicit the capabilities of strong models through weak supervisors and ensure that the behavior of strong models aligns with the intentions of weak supervisors without unsafe behaviors such as deception. Although weak-to-strong generalization exhibiting certain generalization capabilities, strong models exhibit significant overfitting in weak-to-strong generalization: Due to the strong fit ability of strong models, erroneous labels from weak supervisors may lead to overfitting in strong models. In addition, simply filtering out incorrect labels may lead to a degeneration in question quality, resulting in a weak generalization ability of strong models on hard questions. To mitigate overfitting in weak-to-strong generalization, we propose a two-stage framework that simultaneously improves the quality of supervision signals and the quality of input questions. Experimental results in three series of large language models and two mathematical benchmarks demonstrate that our framework significantly improves PGR compared to naive weak-to-strong generalization, even achieving up to 100\% PGR on some models. 

**Abstract (ZH)**: 超对齐中的弱到强泛化：一种双阶段框架以同时提高监督信号和输入问题的质量 

---
# Knowledge-Decoupled Synergetic Learning: An MLLM based Collaborative Approach to Few-shot Multimodal Dialogue Intention Recognition 

**Title (ZH)**: 知识解耦协同学习：一种基于MLLM的少样本多模态对话意图识别协作方法 

**Authors**: Bin Chen, Yu Zhang, Hongfei Ye, Ziyi Huang, Hongyang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.04201)  

**Abstract**: Few-shot multimodal dialogue intention recognition is a critical challenge in the e-commerce domainn. Previous methods have primarily enhanced model classification capabilities through post-training techniques. However, our analysis reveals that training for few-shot multimodal dialogue intention recognition involves two interconnected tasks, leading to a seesaw effect in multi-task learning. This phenomenon is attributed to knowledge interference stemming from the superposition of weight matrix updates during the training process. To address these challenges, we propose Knowledge-Decoupled Synergetic Learning (KDSL), which mitigates these issues by utilizing smaller models to transform knowledge into interpretable rules, while applying the post-training of larger models. By facilitating collaboration between the large and small multimodal large language models for prediction, our approach demonstrates significant improvements. Notably, we achieve outstanding results on two real Taobao datasets, with enhancements of 6.37\% and 6.28\% in online weighted F1 scores compared to the state-of-the-art method, thereby validating the efficacy of our framework. 

**Abstract (ZH)**: 少量样本多模态对话意图识别是电子商务领域的一个关键挑战。先前的方法主要通过后训练技术增强模型分类能力。然而，我们的分析显示，少量样本多模态对话意图识别的训练涉及两个相互关联的任务，导致多任务学习中的跷跷板效应。这种现象归因于训练过程中权重矩阵更新的叠加引起的知识干扰。为了解决这些挑战，我们提出了知识解耦协同学习（KDSL），该方法通过使用较小的模型将知识转化为可解释的规则，同时对较大的模型进行后训练，以促进大规模和小型多模态大语言模型之间的协作预测，从而展示了显著的性能提升。特别地，我们在两个真实的淘宝数据集上取得了出色的结果，与最先进的方法相比，线上加权F1分数分别提升了6.37%和6.28%，从而验证了我们框架的有效性。 

---
# MASTER: Multimodal Segmentation with Text Prompts 

**Title (ZH)**: MASTER：带有文本提示的多模态分割 

**Authors**: Fuyang Liu, Shun Lu, Jilin Mei, Yu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04199)  

**Abstract**: RGB-Thermal fusion is a potential solution for various weather and light conditions in challenging scenarios. However, plenty of studies focus on designing complex modules to fuse different modalities. With the widespread application of large language models (LLMs), valuable information can be more effectively extracted from natural language. Therefore, we aim to leverage the advantages of large language models to design a structurally simple and highly adaptable multimodal fusion model architecture. We proposed MultimodAl Segmentation with TExt PRompts (MASTER) architecture, which integrates LLM into the fusion of RGB-Thermal multimodal data and allows complex query text to participate in the fusion process. Our model utilizes a dual-path structure to extract information from different modalities of images. Additionally, we employ LLM as the core module for multimodal fusion, enabling the model to generate learnable codebook tokens from RGB, thermal images, and textual information. A lightweight image decoder is used to obtain semantic segmentation results. The proposed MASTER performs exceptionally well in benchmark tests across various automated driving scenarios, yielding promising results. 

**Abstract (ZH)**: RGB-热融合在多种天气和光照条件下的挑战场景中是一种潜在的解决方案。然而，许多研究集中在设计复杂的模块来融合不同的模态。随着大规模语言模型（LLMs）的广泛应用，自然语言中的有价值信息可以更有效地被提取。因此，我们旨在利用大规模语言模型的优势，设计一种结构简单且高度适配的多模态融合模型架构。我们提出了MultimodAl Segmentation with TExt PRompts (MASTER) 架构，将LLM整合到RGB-热多模态数据的融合过程中，并使复杂的查询文本参与到融合过程中。我们的模型采用双路径结构从不同模态的图像中提取信息。此外，我们使用LLM作为核心模块进行多模态融合，使模型能够从RGB、热图像和文本信息中生成可学习的代码书令牌。一种轻量级的图像解码器用于获得语义分割结果。提出的MASTER在各种自动驾驶场景的基准测试中表现出色，取得了令人鼓舞的结果。 

---
# Large-Scale AI in Telecom: Charting the Roadmap for Innovation, Scalability, and Enhanced Digital Experiences 

**Title (ZH)**: 电信领域的大规模AI：创新、可扩展性和增强数字体验的道路图 

**Authors**: Adnan Shahid, Adrian Kliks, Ahmed Al-Tahmeesschi, Ahmed Elbakary, Alexandros Nikou, Ali Maatouk, Ali Mokh, Amirreza Kazemi, Antonio De Domenico, Athanasios Karapantelakis, Bo Cheng, Bo Yang, Bohao Wang, Carlo Fischione, Chao Zhang, Chaouki Ben Issaid, Chau Yuen, Chenghui Peng, Chongwen Huang, Christina Chaccour, Christo Kurisummoottil Thomas, Dheeraj Sharma, Dimitris Kalogiros, Dusit Niyato, Eli De Poorter, Elissa Mhanna, Emilio Calvanese Strinati, Faouzi Bader, Fathi Abdeldayem, Fei Wang, Fenghao Zhu, Gianluca Fontanesi, Giovanni Geraci, Haibo Zhou, Hakimeh Purmehdi, Hamed Ahmadi, Hang Zou, Hongyang Du, Hoon Lee, Howard H. Yang, Iacopo Poli, Igor Carron, Ilias Chatzistefanidis, Inkyu Lee, Ioannis Pitsiorlas, Jaron Fontaine, Jiajun Wu, Jie Zeng, Jinan Li, Jinane Karam, Johny Gemayel, Juan Deng, Julien Frison, Kaibin Huang, Kehai Qiu, Keith Ball, Kezhi Wang, Kun Guo, Leandros Tassiulas, Lecorve Gwenole, Liexiang Yue, Lina Bariah, Louis Powell, Marcin Dryjanski, Maria Amparo Canaveras Galdon, Marios Kountouris, Maryam Hafeez, Maxime Elkael, Mehdi Bennis, Mehdi Boudjelli, Meiling Dai, Merouane Debbah, Michele Polese, Mohamad Assaad, Mohamed Benzaghta, Mohammad Al Refai, Moussab Djerrab, Mubeen Syed, Muhammad Amir, Na Yan, Najla Alkaabi, Nan Li, Nassim Sehad, Navid Nikaein, Omar Hashash, Pawel Sroka, Qianqian Yang, Qiyang Zhao, Rasoul Nikbakht Silab, Rex Ying, Roberto Morabito, Rongpeng Li, Ryad Madi, Salah Eddine El Ayoubi, Salvatore D'Oro, Samson Lasaulce, Serveh Shalmashi, Sige Liu, Sihem Cherrared, Swarna Bindu Chetty  

**Link**: [PDF](https://arxiv.org/pdf/2503.04184)  

**Abstract**: This white paper discusses the role of large-scale AI in the telecommunications industry, with a specific focus on the potential of generative AI to revolutionize network functions and user experiences, especially in the context of 6G systems. It highlights the development and deployment of Large Telecom Models (LTMs), which are tailored AI models designed to address the complex challenges faced by modern telecom networks. The paper covers a wide range of topics, from the architecture and deployment strategies of LTMs to their applications in network management, resource allocation, and optimization. It also explores the regulatory, ethical, and standardization considerations for LTMs, offering insights into their future integration into telecom infrastructure. The goal is to provide a comprehensive roadmap for the adoption of LTMs to enhance scalability, performance, and user-centric innovation in telecom networks. 

**Abstract (ZH)**: 这篇白皮书探讨了大规模AI在电信行业的角色，重点关注生成型AI在重构网络功能和用户经验方面的潜在能力，特别是在6G系统中的应用背景。它强调了大型电信模型（LTMs）的发展和部署，这些模型是针对现代电信网络所面临的复杂挑战定制的人工智能模型。本文涵盖了从LTMs的架构和部署策略到其在网络管理、资源分配和优化方面的应用的广泛话题。还探讨了LTMs的监管、伦理和标准化考虑，提供了一些关于其如何集成到电信基础设施中的见解。目标是提供一份全面的路线图，以促进LTMs的应用，增强电信网络的可扩展性、性能和以用户为中心的创新。 

---
# Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation 

**Title (ZH)**: 基于语义检索增强对比学习的序列推荐 

**Authors**: Ziqiang Cui, Yunpeng Weng, Xing Tang, Xiaokun Zhang, Dugang Liu, Shiwei Li, Peiyang Liu, Bowei He, Weihong Luo, Xiuqiang He, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.04162)  

**Abstract**: Sequential recommendation aims to model user preferences based on historical behavior sequences, which is crucial for various online platforms. Data sparsity remains a significant challenge in this area as most users have limited interactions and many items receive little attention. To mitigate this issue, contrastive learning has been widely adopted. By constructing positive sample pairs from the data itself and maximizing their agreement in the embedding space,it can leverage available data more effectively. Constructing reasonable positive sample pairs is crucial for the success of contrastive learning. However, current approaches struggle to generate reliable positive pairs as they either rely on representations learned from inherently sparse collaborative signals or use random perturbations which introduce significant uncertainty. To address these limitations, we propose a novel approach named Semantic Retrieval Augmented Contrastive Learning (SRA-CL), which leverages semantic information to improve the reliability of contrastive samples. SRA-CL comprises two main components: (1) Cross-Sequence Contrastive Learning via User Semantic Retrieval, which utilizes large language models (LLMs) to understand diverse user preferences and retrieve semantically similar users to form reliable positive samples through a learnable sample synthesis method; and (2) Intra-Sequence Contrastive Learning via Item Semantic Retrieval, which employs LLMs to comprehend items and retrieve similar items to perform semantic-based item substitution, thereby creating semantically consistent augmented views for contrastive learning. SRA-CL is plug-and-play and can be integrated into standard sequential recommendation models. Extensive experiments on four public datasets demonstrate the effectiveness and generalizability of the proposed approach. 

**Abstract (ZH)**: 基于序列的推荐旨在根据用户历史行为序列建模用户偏好，对于各种在线平台至关重要。数据稀疏性仍然是这一领域的一个重大挑战，因为大多数用户与系统的互动有限，许多项目受到的关注也很少。为缓解这一问题，对比学习已被广泛采用。通过从数据本身构建正样本对，并在嵌入空间中最大化它们的一致性，它可以更有效地利用可用数据。构建合理的正样本对对于对比学习的成功至关重要。然而，当前的方法难以生成可靠的数据对，因为它们要么依赖于从固有稀疏的合作信号中学习到的表示，要么使用随机扰动引入了大量不确定性。为解决这些局限性，我们提出了一种名为语义检索增强对比学习（SRA-CL）的新方法，该方法利用语义信息以提高对比样本的可靠性。SRA-CL 包含两个主要组成部分：1）基于用户语义检索的跨序列对比学习，该方法利用大型语言模型（LLMs）理解多样化的用户偏好，并通过可学习的样本合成方法检索语义相似的用户，以形成可靠的正样本对；2）基于物品语义检索的序列内对照学习，该方法利用大型语言模型理解物品并检索相似的物品，以执行基于语义的物品替代，从而创建语义上一致的增强视图用于对比学习。SRA-CL 是即插即用的，可以集成到标准的基于序列的推荐模型中。在四个公开数据集上的广泛实验表明，所提出的方法的有效性和普适性。 

---
# Ticktack : Long Span Temporal Alignment of Large Language Models Leveraging Sexagenary Cycle Time Expression 

**Title (ZH)**: Ticktack：利用六十甲子时间表达进行大型语言模型长时间跨度时间对齐 

**Authors**: Xue Han, Qian Hu, Yitong Wang, Wenchun Gao, Lianlian Zhang, Qing Wang, Lijun Mei, Chao Deng, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2503.04150)  

**Abstract**: Large language models (LLMs) suffer from temporal misalignment issues especially across long span of time. The issue arises from knowing that LLMs are trained on large amounts of data where temporal information is rather sparse over long times, such as thousands of years, resulting in insufficient learning or catastrophic forgetting by the LLMs. This paper proposes a methodology named "Ticktack" for addressing the LLM's long-time span misalignment in a yearly setting. Specifically, we first propose to utilize the sexagenary year expression instead of the Gregorian year expression employed by LLMs, achieving a more uniform distribution in yearly granularity. Then, we employ polar coordinates to model the sexagenary cycle of 60 terms and the year order within each term, with additional temporal encoding to ensure LLMs understand them. Finally, we present a temporal representational alignment approach for post-training LLMs that effectively distinguishes time points with relevant knowledge, hence improving performance on time-related tasks, particularly over a long period. We also create a long time span benchmark for evaluation. Experimental results prove the effectiveness of our proposal. 

**Abstract (ZH)**: 大型语言模型（LLMs）在长时间跨度上遭受时间对齐问题，特别是跨长时间段时更为明显。这些问题源于LLMs在训练时面对的是大量数据，其中长期时间上的时间信息相对稀疏，如上千年的数据，导致LLMs在长期学习中出现不足或灾难性遗忘。本文提出了一种名为“Ticktack”的方法，以解决LLMs在年际跨度上的时间对齐问题。具体而言，我们首先提议使用六十甲子年表示法而非LLMs使用的格里高利历表示法，从而在年际粒度上实现更均匀的分布。然后，我们利用极坐标来建模每60个周期的六十甲子周期和每个周期内的年序，并加入额外的时间编码，以确保LLMs能够理解这些时间信息。最后，我们提出了一种后训练时间表示对齐方法，该方法能够有效区分具有相关知识的时间点，从而在长时间相关任务中提高性能。我们还创建了一个长时间跨度基准进行评估。实验结果证明了我们方法的有效性。 

---
# Dynamic Benchmarking of Reasoning Capabilities in Code Large Language Models Under Data Contamination 

**Title (ZH)**: 代码大型语言模型在数据污染情况下推理能力的动态基准测试 

**Authors**: Simin Chen, Pranav Pusarla, Baishakhi Ray  

**Link**: [PDF](https://arxiv.org/pdf/2503.04149)  

**Abstract**: The rapid evolution of code largelanguage models underscores the need for effective and transparent benchmarking of their reasoning capabilities. However, the current benchmarking approach heavily depends on publicly available, human-created datasets. The widespread use of these fixed benchmark datasets makes the benchmarking process to be static and thus particularly susceptible to data contamination, an unavoidable consequence of the extensive data collection processes used to train Code LLMs. Existing approaches that address data contamination often suffer from human effort limitations and imbalanced problem complexity. To tackle these challenges, we propose \tool, a novel benchmarking suite for evaluating Code LLMs under potential data contamination. Given a seed programming problem, \tool employs multiple agents to extract and modify the context without altering the core logic, generating semantically equivalent variations. We introduce a dynamic data generation methods and conduct empirical studies on two seed datasets across 21 Code LLMs. Results show that \tool effectively benchmarks reasoning capabilities under contamination risks while generating diverse problem sets to ensure consistent and reliable evaluations. 

**Abstract (ZH)**: 代码大型语言模型的快速演进强调了对其推理能力进行有效和透明的基准测试的必要性。然而，当前的基准测试方法严重依赖于公开的人类创建数据集。广泛使用这些固定基准数据集使得基准测试过程变得静态，从而特别容易受到数据污染的影响，这是训练代码LLM过程中大量数据收集不可避免的后果。现有的处理数据污染的方法往往受到人力资源限制和问题复杂性不平衡的困扰。为应对这些挑战，我们提出了一种名为\tool的新型基准测试套件，以评估代码LLM在潜在数据污染情况下的推理能力。给定一个种子编程问题，\tool 使用多个代理提取和修改上下文而不改变核心逻辑，生成语义等价的变化。我们介绍了一种动态数据生成方法，并在两个种子数据集上对21个代码LLM进行了实证研究。结果表明，\tool 在生成多样化的问题集以确保一致和可靠评估的同时，有效地对推理能力在污染风险下的表现进行了基准测试。 

---
# Disparities in LLM Reasoning Accuracy and Explanations: A Case Study on African American English 

**Title (ZH)**: LLM推理准确性和解释的差异：关于非洲裔美国英语的案例研究 

**Authors**: Runtao Zhou, Guangya Wan, Saadia Gabriel, Sheng Li, Alexander J Gates, Maarten Sap, Thomas Hartvigsen  

**Link**: [PDF](https://arxiv.org/pdf/2503.04099)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in reasoning tasks, leading to their widespread deployment. However, recent studies have highlighted concerning biases in these models, particularly in their handling of dialectal variations like African American English (AAE). In this work, we systematically investigate dialectal disparities in LLM reasoning tasks. We develop an experimental framework comparing LLM performance given Standard American English (SAE) and AAE prompts, combining LLM-based dialect conversion with established linguistic analyses. We find that LLMs consistently produce less accurate responses and simpler reasoning chains and explanations for AAE inputs compared to equivalent SAE questions, with disparities most pronounced in social science and humanities domains. These findings highlight systematic differences in how LLMs process and reason about different language varieties, raising important questions about the development and deployment of these systems in our multilingual and multidialectal world. Our code repository is publicly available at this https URL. 

**Abstract (ZH)**: 大型语言模型在推理任务中展示了 remarkable 能力，导致它们被广泛应用。然而，近期研究指出这些模型中存在令人担忧的偏见，特别体现在方言变体处理方面，如美国黑人英语（AAE）。在本工作中，我们系统地研究了不同方言在大型语言模型推理任务中的差异。我们开发了一个实验框架，比较大型语言模型在标准美国英语（SAE）和AAE提示下的表现，结合基于大型语言模型的方言转换与传统的语言学分析。我们发现，对于AAE输入，大型语言模型生成的响应更不准确，推理链条和解释也更简单，这些差异在社会科学和人文学科领域尤为显著。这些发现强调了大型语言模型在处理和推理不同语言变体时的系统性差异，引发了关于这些系统在多语言、多方言世界中的开发和部署的重要问题。我们的代码库已在此 https URL 公开。 

---
# Uncovering inequalities in new knowledge learning by large language models across different languages 

**Title (ZH)**: 探究大型语言模型在不同语言中新知识学习中的不平等现象 

**Authors**: Chenglong Wang, Haoyu Tang, Xiyuan Yang, Yueqi Xie, Jina Suh, Sunayana Sitaram, Junming Huang, Yu Xie, Zhaoya Gong, Xing Xie, Fangzhao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04064)  

**Abstract**: As large language models (LLMs) gradually become integral tools for problem solving in daily life worldwide, understanding linguistic inequality is becoming increasingly important. Existing research has primarily focused on static analyses that assess the disparities in the existing knowledge and capabilities of LLMs across languages. However, LLMs are continuously evolving, acquiring new knowledge to generate up-to-date, domain-specific responses. Investigating linguistic inequalities within this dynamic process is, therefore, also essential. In this paper, we explore inequalities in new knowledge learning by LLMs across different languages and four key dimensions: effectiveness, transferability, prioritization, and robustness. Through extensive experiments under two settings (in-context learning and fine-tuning) using both proprietary and open-source models, we demonstrate that low-resource languages consistently face disadvantages across all four dimensions. By shedding light on these disparities, we aim to raise awareness of linguistic inequalities in LLMs' new knowledge learning, fostering the development of more inclusive and equitable future LLMs. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在全球日常生活中逐渐成为解决问题的重要工具，理解语言不平等现象变得越来越重要。现有研究主要关注静态分析，评估不同语言中LLMs的知识和能力差异。然而，LLMs在不断进化，获得新的知识以生成最新的、针对性的回答。因此，研究这一动态过程中的语言不平等同样至关重要。在本文中，我们探讨了LLMs在不同语言和四个关键维度（有效性、可迁移性、优先级和稳健性）中学习新知识过程中的不平等现象。通过在两种设置（上下文学习和微调）下进行广泛实验，使用自有的和开源的模型，我们展示了低资源语言在所有四个维度上均面临不利状况。通过揭示这些差异，我们旨在提高对LLMs新知识学习中的语言不平等的认识，促进更加包容和公平的未来LLMs的发展。 

---
# Benchmarking Large Language Models on Multiple Tasks in Bioinformatics NLP with Prompting 

**Title (ZH)**: 大型语言模型在生物信息学NLP任务中的多任务基准测试与提示方法 

**Authors**: Jiyue Jiang, Pengan Chen, Jiuming Wang, Dongchen He, Ziqin Wei, Liang Hong, Licheng Zong, Sheng Wang, Qinze Yu, Zixian Ma, Yanyu Chen, Yimin Fan, Xiangyu Shi, Jiawei Sun, Chuan Wu, Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.04013)  

**Abstract**: Large language models (LLMs) have become important tools in solving biological problems, offering improvements in accuracy and adaptability over conventional methods. Several benchmarks have been proposed to evaluate the performance of these LLMs. However, current benchmarks can hardly evaluate the performance of these models across diverse tasks effectively. In this paper, we introduce a comprehensive prompting-based benchmarking framework, termed Bio-benchmark, which includes 30 key bioinformatics tasks covering areas such as proteins, RNA, drugs, electronic health records, and traditional Chinese medicine. Using this benchmark, we evaluate six mainstream LLMs, including GPT-4o and Llama-3.1-70b, etc., using 0-shot and few-shot Chain-of-Thought (CoT) settings without fine-tuning to reveal their intrinsic capabilities. To improve the efficiency of our evaluations, we demonstrate BioFinder, a new tool for extracting answers from LLM responses, which increases extraction accuracy by round 30% compared to existing methods. Our benchmark results show the biological tasks suitable for current LLMs and identify specific areas requiring enhancement. Furthermore, we propose targeted prompt engineering strategies for optimizing LLM performance in these contexts. Based on these findings, we provide recommendations for the development of more robust LLMs tailored for various biological applications. This work offers a comprehensive evaluation framework and robust tools to support the application of LLMs in bioinformatics. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已成为解决生物学问题的重要工具，展现出比传统方法更高的准确性和适应性。已提出了多种基准来评估这些LLMs的性能。然而，当前的基准难以有效地全面评估这些模型在多样任务中的表现。本文引入了一个全面的提示驱动基准框架，名为Bio-benchmark，包括30项关键生物信息学任务，涵盖蛋白质、RNA、药物、电子健康记录以及传统中药等领域。利用这一基准，我们评估了包括GPT-4o和Llama-3.1-70b在内的六种主流LLMs，在未进行微调的情况下采用零样本和少量样本链式思维（CoT）设置，揭示其内在能力。为了提高评估效率，我们展示了BioFinder这一新的工具，用于从LLM响应中提取答案，其提取准确率相比现有方法提升了约30%。基准结果展示了当前LLMs适合的生物任务，并指出了需要改进的具体领域。此外，我们提出了针对这些领域的目标提示工程策略以优化LLM性能。基于这些发现，我们提出了针对各种生物应用开发更稳健LLMs的建议。本研究提供了一个全面的评估框架和可靠的工具，以支持LLMs在生物信息学中的应用。 

---
# RetinalGPT: A Retinal Clinical Preference Conversational Assistant Powered by Large Vision-Language Models 

**Title (ZH)**: RetinalGPT：由大规模视觉-语言模型驱动的视网膜临床偏好对话助理 

**Authors**: Wenhui Zhu, Xin Li, Xiwen Chen, Peijie Qiu, Vamsi Krishna Vasa, Xuanzhao Dong, Yanxi Chen, Natasha Lepore, Oana Dumitrascu, Yi Su, Yalin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03987)  

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have gained significant attention for their remarkable ability to process and analyze non-textual data, such as images, videos, and audio. Notably, several adaptations of general-domain MLLMs to the medical field have been explored, including LLaVA-Med. However, these medical adaptations remain insufficiently advanced in understanding and interpreting retinal images. In contrast, medical experts emphasize the importance of quantitative analyses for disease detection and interpretation. This underscores a gap between general-domain and medical-domain MLLMs: while general-domain MLLMs excel in broad applications, they lack the specialized knowledge necessary for precise diagnostic and interpretative tasks in the medical field. To address these challenges, we introduce \textit{RetinalGPT}, a multimodal conversational assistant for clinically preferred quantitative analysis of retinal images. Specifically, we achieve this by compiling a large retinal image dataset, developing a novel data pipeline, and employing customized visual instruction tuning to enhance both retinal analysis and enrich medical knowledge. In particular, RetinalGPT outperforms MLLM in the generic domain by a large margin in the diagnosis of retinal diseases in 8 benchmark retinal datasets. Beyond disease diagnosis, RetinalGPT features quantitative analyses and lesion localization, representing a pioneering step in leveraging LLMs for an interpretable and end-to-end clinical research framework. The code is available at this https URL 

**Abstract (ZH)**: 最近，多模态大型语言模型（MLLMs）因其处理和分析非文本数据（如图像、视频和音频）的出色能力而获得了广泛关注。值得注意的是，已有研究将通用领域MLLMs应用于医疗领域，包括LLaVA-Med。然而，这些医疗适应性在理解与解释眼底图像方面仍然不够深入。相比之下，医疗专家强调定量分析对于疾病检测和解释的重要性。这突显了通用领域与医疗领域MLLMs之间的差距：虽然通用领域MLLMs在广泛的应用中表现出色，但在医疗领域的精确诊断和解释任务中缺乏必要的专门知识。为解决这些挑战，我们提出了\textit{RetinalGPT}——一种临床优选的多模态对话助手，专门用于眼底图像的定量分析。具体而言，我们通过构建大规模眼底图像数据集、开发新型数据管道以及采用定制化的视觉指令调优，增强了眼底分析能力并丰富了医疗知识。特别地，RetinalGPT在8个基准眼底数据集中的眼底疾病诊断上远超通用领域MLLM。除了疾病诊断，RetinalGPT还具备定量分析和病灶定位的功能，标志着利用LLMs构建可解释和端到端的临床研究框架的先驱性步骤。代码已发布。 

---
# Not-Just-Scaling Laws: Towards a Better Understanding of the Downstream Impact of Language Model Design Decisions 

**Title (ZH)**: 不仅仅是规模法则：关于语言模型设计决策下游影响的更好理解 

**Authors**: Emmy Liu, Amanda Bertsch, Lintang Sutawika, Lindia Tjuatja, Patrick Fernandes, Lara Marinov, Michael Chen, Shreya Singhal, Carolin Lawrence, Aditi Raghunathan, Kiril Gashteovski, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2503.03862)  

**Abstract**: Improvements in language model capabilities are often attributed to increasing model size or training data, but in some cases smaller models trained on curated data or with different architectural decisions can outperform larger ones trained on more tokens. What accounts for this? To quantify the impact of these design choices, we meta-analyze 92 open-source pretrained models across a wide array of scales, including state-of-the-art open-weights models as well as less performant models and those with less conventional design decisions. We find that by incorporating features besides model size and number of training tokens, we can achieve a relative 3-28% increase in ability to predict downstream performance compared with using scale alone. Analysis of model design decisions reveal insights into data composition, such as the trade-off between language and code tasks at 15-25\% code, as well as the better performance of some architectural decisions such as choosing rotary over learned embeddings. Broadly, our framework lays a foundation for more systematic investigation of how model development choices shape final capabilities. 

**Abstract (ZH)**: 语言模型能力的改进通常归因于模型规模的增加或训练数据量的增多，但在某些情况下，经过精心选择数据或采用不同架构决策训练的较小模型可能会优于更大规模但训练数据更多的模型。是什么造成了这种差异？为了量化这些设计选择的影响，我们对92个开源的预训练模型进行了元分析，这些模型涵盖了从最先进的开放权重模型到性能较低且设计较为非传统的模型，包括各种规模的模型。我们发现在除了模型规模和训练tokens数量之外，加入其他特征可以使下游性能预测能力相对提高3%-28%。通过对模型设计决策的分析，我们揭示了数据组成方面的见解，例如当代码任务占比在15%-25%时语言和代码任务之间的权衡，并且某些架构决策，如选择旋转嵌入而非学习嵌入，具有更好的性能。总体而言，我们的框架为更系统地研究模型开发选择如何塑造最终能力奠定了基础。 

---
# RiskAgent: Autonomous Medical AI Copilot for Generalist Risk Prediction 

**Title (ZH)**: RiskAgent: 自主医疗AI副驾，用于通用风险预测 

**Authors**: Fenglin Liu, Jinge Wu, Hongjian Zhou, Xiao Gu, Soheila Molaei, Anshul Thakur, Lei Clifton, Honghan Wu, David A. Clifton  

**Link**: [PDF](https://arxiv.org/pdf/2503.03802)  

**Abstract**: The application of Large Language Models (LLMs) to various clinical applications has attracted growing research attention. However, real-world clinical decision-making differs significantly from the standardized, exam-style scenarios commonly used in current efforts. In this paper, we present the RiskAgent system to perform a broad range of medical risk predictions, covering over 387 risk scenarios across diverse complex diseases, e.g., cardiovascular disease and cancer. RiskAgent is designed to collaborate with hundreds of clinical decision tools, i.e., risk calculators and scoring systems that are supported by evidence-based medicine. To evaluate our method, we have built the first benchmark MedRisk specialized for risk prediction, including 12,352 questions spanning 154 diseases, 86 symptoms, 50 specialties, and 24 organ systems. The results show that our RiskAgent, with 8 billion model parameters, achieves 76.33% accuracy, outperforming the most recent commercial LLMs, o1, o3-mini, and GPT-4.5, and doubling the 38.39% accuracy of GPT-4o. On rare diseases, e.g., Idiopathic Pulmonary Fibrosis (IPF), RiskAgent outperforms o1 and GPT-4.5 by 27.27% and 45.46% accuracy, respectively. Finally, we further conduct a generalization evaluation on an external evidence-based diagnosis benchmark and show that our RiskAgent achieves the best results. These encouraging results demonstrate the great potential of our solution for diverse diagnosis domains. To improve the adaptability of our model in different scenarios, we have built and open-sourced a family of models ranging from 1 billion to 70 billion parameters. Our code, data, and models are all available at this https URL. 

**Abstract (ZH)**: 大型语言模型在各种临床应用中的应用吸引了日益增长的研究关注。然而，真实的临床决策与当前研究中常用的标准化考试场景存在显著差异。本文介绍RiskAgent系统，用于进行广泛的医疗风险预测，覆盖387多种疾病的风险场景，包括心血管疾病和癌症。RiskAgent旨在与数百种临床决策工具协作，即受循证医学支持的风险计算器和评分系统。为评估该方法，我们构建了第一个专门用于风险预测的基准MedRisk，包含12,352个问题，覆盖154种疾病、86种症状、50个专科和24个器官系统。结果显示，我们的RiskAgent（拥有80亿个模型参数）的准确率为76.33%，优于最近的商业LLM o1、o3-mini和GPT-4.5，并将GPT-4o的38.39%准确率翻了一番。在罕见疾病领域，如特发性肺纤维化（IPF），RiskAgent的准确率分别超出o1和GPT-4.5 27.27%和45.46%。最后，我们在外部循证诊断基准上进行了泛化评估，显示RiskAgent取得了最佳结果。这些令人鼓舞的结果证明了我们的解决方案在多种诊断领域的巨大潜力。为了提高模型在不同场景下的适应性，我们构建并开源了一系列从10亿到70亿参数的模型。我们的代码、数据和模型均在此处公开。 

---
# Multi-Agent Systems Powered by Large Language Models: Applications in Swarm Intelligence 

**Title (ZH)**: 由大型语言模型驱动的多智能体系统：在群智智能中的应用 

**Authors**: Cristian Jimenez-Romero, Alper Yegenoglu, Christian Blum  

**Link**: [PDF](https://arxiv.org/pdf/2503.03800)  

**Abstract**: This work examines the integration of large language models (LLMs) into multi-agent simulations by replacing the hard-coded programs of agents with LLM-driven prompts. The proposed approach is showcased in the context of two examples of complex systems from the field of swarm intelligence: ant colony foraging and bird flocking. Central to this study is a toolchain that integrates LLMs with the NetLogo simulation platform, leveraging its Python extension to enable communication with GPT-4o via the OpenAI API. This toolchain facilitates prompt-driven behavior generation, allowing agents to respond adaptively to environmental data. For both example applications mentioned above, we employ both structured, rule-based prompts and autonomous, knowledge-driven prompts. Our work demonstrates how this toolchain enables LLMs to study self-organizing processes and induce emergent behaviors within multi-agent environments, paving the way for new approaches to exploring intelligent systems and modeling swarm intelligence inspired by natural phenomena. We provide the code, including simulation files and data at this https URL. 

**Abstract (ZH)**: 本研究探讨了通过用大语言模型（LLMs）驱动的提示替换代理的硬编码程序，将大语言模型集成到多代理模拟中的方法。该研究以王国智能领域两种复杂系统的例子——蚁群觅食和鸟群集群行为——为背景进行了展示。本研究的核心在于一个将LLMs与NetLogo模拟平台集成的工具链，利用其Python扩展通过OpenAI API与GPT-4o进行通信。该工具链实现基于提示的行为生成，使代理能够适应性地响应环境数据。对于上述两个示例应用，我们均采用了结构化、基于规则的提示和自主、基于知识的提示。我们的研究表明，该工具链使大语言模型能够研究自我组织过程，在多代理环境中诱导涌现行为，为探索智能系统和基于自然现象启发的群体智能建模开辟了新途径。我们在此提供了相关代码、模拟文件和数据。 

---
# FlexInfer: Breaking Memory Constraint via Flexible and Efficient Offloading for On-Device LLM Inference 

**Title (ZH)**: FlexInfer: 超越内存约束的柔性高效卸载以实现设备上大语言模型推理 

**Authors**: Hongchao Du, Shangyu Wu, Arina Kharlamova, Nan Guan, Chun Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2503.03777)  

**Abstract**: Large Language Models (LLMs) face challenges for on-device inference due to high memory demands. Traditional methods to reduce memory usage often compromise performance and lack adaptability. We propose FlexInfer, an optimized offloading framework for on-device inference, addressing these issues with techniques like asynchronous prefetching, balanced memory locking, and flexible tensor preservation. These strategies enhance memory efficiency and mitigate I/O bottlenecks, ensuring high performance within user-specified resource constraints. Experiments demonstrate that FlexInfer significantly improves throughput under limited resources, achieving up to 12.5 times better performance than existing methods and facilitating the deployment of large models on resource-constrained devices. 

**Abstract (ZH)**: FlexInfer：一种优化的设备端推理卸载框架 

---
# M2-omni: Advancing Omni-MLLM for Comprehensive Modality Support with Competitive Performance 

**Title (ZH)**: M2-omni: 推动全面模态支持的竞争力性能 omnibig语言模型 

**Authors**: Qingpei Guo, Kaiyou Song, Zipeng Feng, Ziping Ma, Qinglong Zhang, Sirui Gao, Xuzheng Yu, Yunxiao Sun, Tai-WeiChang, Jingdong Chen, Ming Yang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.18778)  

**Abstract**: We present M2-omni, a cutting-edge, open-source omni-MLLM that achieves competitive performance to GPT-4o. M2-omni employs a unified multimodal sequence modeling framework, which empowers Large Language Models(LLMs) to acquire comprehensive cross-modal understanding and generation capabilities. Specifically, M2-omni can process arbitrary combinations of audio, video, image, and text modalities as input, generating multimodal sequences interleaving with audio, image, or text outputs, thereby enabling an advanced and interactive real-time experience. The training of such an omni-MLLM is challenged by significant disparities in data quantity and convergence rates across modalities. To address these challenges, we propose a step balance strategy during pre-training to handle the quantity disparities in modality-specific data. Additionally, a dynamically adaptive balance strategy is introduced during the instruction tuning stage to synchronize the modality-wise training progress, ensuring optimal convergence. Notably, we prioritize preserving strong performance on pure text tasks to maintain the robustness of M2-omni's language understanding capability throughout the training process. To our best knowledge, M2-omni is currently a very competitive open-source model to GPT-4o, characterized by its comprehensive modality and task support, as well as its exceptional performance. We expect M2-omni will advance the development of omni-MLLMs, thus facilitating future research in this domain. 

**Abstract (ZH)**: M2-omni：一种与GPT-4o性能相当的开源全模态大语言模型 

---
