# ValuePilot: A Two-Phase Framework for Value-Driven Decision-Making 

**Title (ZH)**: 价值航标：一种 duas 阶段框架，用于价值驱动的决策制定 

**Authors**: Yitong Luo, Hou Hei Lam, Ziang Chen, Zhenliang Zhang, Xue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2503.04569)  

**Abstract**: Despite recent advances in artificial intelligence (AI), it poses challenges to ensure personalized decision-making in tasks that are not considered in training datasets. To address this issue, we propose ValuePilot, a two-phase value-driven decision-making framework comprising a dataset generation toolkit DGT and a decision-making module DMM trained on the generated data. DGT is capable of generating scenarios based on value dimensions and closely mirroring real-world tasks, with automated filtering techniques and human curation to ensure the validity of the dataset. In the generated dataset, DMM learns to recognize the inherent values of scenarios, computes action feasibility and navigates the trade-offs between multiple value dimensions to make personalized decisions. Extensive experiments demonstrate that, given human value preferences, our DMM most closely aligns with human decisions, outperforming Claude-3.5-Sonnet, Gemini-2-flash, Llama-3.1-405b and GPT-4o. This research is a preliminary exploration of value-driven decision-making. We hope it will stimulate interest in value-driven decision-making and personalized decision-making within the community. 

**Abstract (ZH)**: 尽管近年来人工智能取得了进展，但确保在未包含在训练数据集中的任务中实现个性化决策仍然面临挑战。为应对这一问题，我们提出了一种两阶段的价值导向决策框架ValuePilot，该框架包括一个数据生成工具包DGT和一个基于生成数据训练的决策模块DMM。DGT能够基于价值维度生成情景，并紧密结合实际任务，采用自动化过滤技术和人工校勘以确保数据集的有效性。在生成的数据集中，DMM学习识别情景内在的价值，计算行动可行性，并在多个价值维度之间权衡以做出个性化决策。广泛实验表明，给定人类的价值偏好，我们的DMM最接近人类决策，优于Claude-3.5-Sonnet、Gemini-2-flash、Llama-3.1-405b和GPT-4o。本研究是对价值导向决策的初步探索，我们希望它能激发社区对价值导向决策和个性化决策的兴趣。 

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
# Dynamic Pricing for On-Demand DNN Inference in the Edge-AI Market 

**Title (ZH)**: 基于边缘AI市场的按需DNN推理动态定价 

**Authors**: Songyuan Li, Jia Hu, Geyong Min, Haojun Huang, Jiwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04521)  

**Abstract**: The convergence of edge computing and AI gives rise to Edge-AI, which enables the deployment of real-time AI applications and services at the network edge. One of the fundamental research issues in Edge-AI is edge inference acceleration, which aims to realize low-latency high-accuracy DNN inference services by leveraging the fine-grained offloading of partitioned inference tasks from end devices to edge servers. However, existing research has yet to adopt a practical Edge-AI market perspective, which would systematically explore the personalized inference needs of AI users (e.g., inference accuracy, latency, and task complexity), the revenue incentives for AI service providers that offer edge inference services, and multi-stakeholder governance within a market-oriented context. To bridge this gap, we propose an Auction-based Edge Inference Pricing Mechanism (AERIA) for revenue maximization to tackle the multi-dimensional optimization problem of DNN model partition, edge inference pricing, and resource allocation. We investigate the multi-exit device-edge synergistic inference scheme for on-demand DNN inference acceleration, and analyse the auction dynamics amongst the AI service providers, AI users and edge infrastructure provider. Owing to the strategic mechanism design via randomized consensus estimate and cost sharing techniques, the Edge-AI market attains several desirable properties, including competitiveness in revenue maximization, incentive compatibility, and envy-freeness, which are crucial to maintain the effectiveness, truthfulness, and fairness of our auction outcomes. The extensive simulation experiments based on four representative DNN inference workloads demonstrate that our AERIA mechanism significantly outperforms several state-of-the-art approaches in revenue maximization, demonstrating the efficacy of AERIA for on-demand DNN inference in the Edge-AI market. 

**Abstract (ZH)**: 边缘计算与人工智能的融合催生了边缘人工智能（Edge-AI），使其能够在网络边缘部署实时AI应用和服务。边缘人工智能研究中的一个基本问题是边缘推断加速，旨在通过将分区推断任务从终端设备卸载到边缘服务器来实现低延迟高准确度的DNN推断服务。然而，现有研究尚未从实践的边缘人工智能市场视角出发，系统地探讨AI用户个性化推断需求（如推断准确性、延迟和任务复杂度）、提供边缘推断服务的AI服务提供商的收入激励机制，以及市场导向背景下多利益相关者的治理问题。为弥补这一差距，我们提出了一种基于拍卖的边缘推断定价机制（AERIA）以实现收入最大化，并解决DNN模型分区、边缘推断定价和资源分配的多维优化问题。我们研究了多出口设备-边缘协同推断方案以满足按需的DNN推断加速，并分析了AI服务提供商、AI用户和边缘基础设施提供商之间的拍卖动态。通过随机一致性估计和成本分摊等机制设计技术，边缘人工智能市场获得了包括收入最大化竞争力、激励相容性和无嫉妒性在内的多个理想属性，这对保持我们的拍卖结果的有效性、真实性和公平性至关重要。基于四个代表性的DNN推断负载的广泛仿真实验表明，我们的AERIA机制在收入最大化方面显著优于几种最先进的方法，证明了AERIA在边缘人工智能市场按需DNN推断中的有效性。 

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
# From Idea to CAD: A Language Model-Driven Multi-Agent System for Collaborative Design 

**Title (ZH)**: 从理念到CAD：基于语言模型的多agent系统协作设计 

**Authors**: Felix Ocker, Stefan Menzel, Ahmed Sadik, Thiago Rios  

**Link**: [PDF](https://arxiv.org/pdf/2503.04417)  

**Abstract**: Creating digital models using Computer Aided Design (CAD) is a process that requires in-depth expertise. In industrial product development, this process typically involves entire teams of engineers, spanning requirements engineering, CAD itself, and quality assurance. We present an approach that mirrors this team structure with a Vision Language Model (VLM)-based Multi Agent System, with access to parametric CAD tooling and tool documentation. Combining agents for requirements engineering, CAD engineering, and vision-based quality assurance, a model is generated automatically from sketches and/ or textual descriptions. The resulting model can be refined collaboratively in an iterative validation loop with the user. Our approach has the potential to increase the effectiveness of design processes, both for industry experts and for hobbyists who create models for 3D printing. We demonstrate the potential of the architecture at the example of various design tasks and provide several ablations that show the benefits of the architecture's individual components. 

**Abstract (ZH)**: 使用计算机辅助设计（CAD）创建数字模型是一个需要深厚专业知识的过程。在工业产品开发中，这一过程通常涉及整个工程师团队，涵盖需求工程、CAD本身和质量保证。我们提出了一种镜像这种团队结构的方法，采用基于视觉语言模型（VLM）的多智能体系统，并具有参数化CAD工具和工具文档的访问权限。结合需求工程、CAD工程和基于视觉的质量保证智能体，可以从草图和/或文本描述自动生成模型。生成的模型可以与用户在迭代验证循环中进行协作性细化。我们的方法有望提高设计过程的有效性，无论是对于工业专家还是对于创建3D打印模型的爱好者。我们通过各种设计任务的例子展示了该架构的潜力，并提供了几个消融实验，展示了架构各个组件的好处。 

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
# MathMistake Checker: A Comprehensive Demonstration for Step-by-Step Math Problem Mistake Finding by Prompt-Guided LLMs 

**Title (ZH)**: 数学错误检查器：基于提示引导的大语言模型逐步数学问题错误查找综合演示 

**Authors**: Tianyang Zhang, Zhuoxuan Jiang, Haotian Zhang, Lin Lin, Shaohua Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04291)  

**Abstract**: We propose a novel system, MathMistake Checker, designed to automate step-by-step mistake finding in mathematical problems with lengthy answers through a two-stage process. The system aims to simplify grading, increase efficiency, and enhance learning experiences from a pedagogical perspective. It integrates advanced technologies, including computer vision and the chain-of-thought capabilities of the latest large language models (LLMs). Our system supports open-ended grading without reference answers and promotes personalized learning by providing targeted feedback. We demonstrate its effectiveness across various types of math problems, such as calculation and word problems. 

**Abstract (ZH)**: 数学错误检查系统：一种通过两阶段过程自动识别数学问题中详细答案步骤错误的新型系统 

---
# Guidelines for Applying RL and MARL in Cybersecurity Applications 

**Title (ZH)**: RL和MARL在网络安全应用中的应用指南 

**Authors**: Vasilios Mavroudis, Gregory Palmer, Sara Farmer, Kez Smithson Whitehead, David Foster, Adam Price, Ian Miles, Alberto Caron, Stephen Pasteris  

**Link**: [PDF](https://arxiv.org/pdf/2503.04262)  

**Abstract**: Reinforcement Learning (RL) and Multi-Agent Reinforcement Learning (MARL) have emerged as promising methodologies for addressing challenges in automated cyber defence (ACD). These techniques offer adaptive decision-making capabilities in high-dimensional, adversarial environments. This report provides a structured set of guidelines for cybersecurity professionals and researchers to assess the suitability of RL and MARL for specific use cases, considering factors such as explainability, exploration needs, and the complexity of multi-agent coordination. It also discusses key algorithmic approaches, implementation challenges, and real-world constraints, such as data scarcity and adversarial interference. The report further outlines open research questions, including policy optimality, agent cooperation levels, and the integration of MARL systems into operational cybersecurity frameworks. By bridging theoretical advancements and practical deployment, these guidelines aim to enhance the effectiveness of AI-driven cyber defence strategies. 

**Abstract (ZH)**: 强化学习（RL）和多代理强化学习（MARL）已成为应对自动网络防御挑战（ACD）的有前景的方法论。这些技术提供了在高维、对抗性环境中的适应性决策能力。本报告为网络安全专业人员和研究人员提供了一套结构化的指南，以评估RL和MARL在特定应用场景中的适用性，考虑的因素包括可解释性、探索需求以及多代理协调的复杂性。报告还讨论了关键算法方法、实施挑战以及实际约束条件，如数据稀缺性和对抗性干扰。此外，报告概述了开放的研究问题，包括策略最优性、代理合作水平以及MARL系统在运营网络安全框架中的集成。通过融合理论进步和实践部署，这些指南旨在增强基于AI的网络防御策略的有效性。 

---
# VirtualXAI: A User-Centric Framework for Explainability Assessment Leveraging GPT-Generated Personas 

**Title (ZH)**: 基于GPT生成的角色的用户中心解释性评估框架VirtualXAI 

**Authors**: Georgios Makridis, Vasileios Koukos, Georgios Fatouros, Dimosthenis Kyriazis  

**Link**: [PDF](https://arxiv.org/pdf/2503.04261)  

**Abstract**: In today's data-driven era, computational systems generate vast amounts of data that drive the digital transformation of industries, where Artificial Intelligence (AI) plays a key role. Currently, the demand for eXplainable AI (XAI) has increased to enhance the interpretability, transparency, and trustworthiness of AI models. However, evaluating XAI methods remains challenging: existing evaluation frameworks typically focus on quantitative properties such as fidelity, consistency, and stability without taking into account qualitative characteristics such as satisfaction and interpretability. In addition, practitioners face a lack of guidance in selecting appropriate datasets, AI models, and XAI methods -a major hurdle in human-AI collaboration. To address these gaps, we propose a framework that integrates quantitative benchmarking with qualitative user assessments through virtual personas based on the "Anthology" of backstories of the Large Language Model (LLM). Our framework also incorporates a content-based recommender system that leverages dataset-specific characteristics to match new input data with a repository of benchmarked datasets. This yields an estimated XAI score and provides tailored recommendations for both the optimal AI model and the XAI method for a given scenario. 

**Abstract (ZH)**: 在数据驱动时代，计算系统生成大量数据推动产业的数字化转型，其中人工智能（AI）发挥着关键作用。目前，对可解释人工智能（XAI）的需求增加，以提高AI模型的可解释性、透明度和可信度。然而，评估XAI方法依然具有挑战性：现有的评估框架通常侧重于精度、一致性、稳定性等量化属性，而忽视了满意度和可解释性等定性特征。此外，实践者在选择合适的数据集、AI模型和XAI方法方面缺乏指导——这是人机协作中的一个重大障碍。为了解决这些缺口，我们提出了一种框架，通过基于大规模语言模型（LLM）背景故事的虚拟人物整合定量基准测试与定性用户评估。该框架还包含了基于内容的推荐系统，利用数据集特定特性将新输入数据与基准数据集仓库匹配，从而估计XAI评分，并为给定场景提供个性化的AI模型和XAI方法推荐。 

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
# Artificial Intelligence in Pronunciation Teaching: Use and Beliefs of Foreign Language Teachers 

**Title (ZH)**: 人工智能在语音教学中的应用与外语教师的信念 

**Authors**: Georgios P. Georgiou  

**Link**: [PDF](https://arxiv.org/pdf/2503.04128)  

**Abstract**: Pronunciation instruction in foreign language classrooms has often been an overlooked area of focus. With the widespread adoption of Artificial Intelligence (AI) and its potential benefits, investigating how AI is utilized in pronunciation teaching and understanding the beliefs of teachers about this tool is essential for improving learning outcomes. This study aims to examine how AI use for pronunciation instruction varies across different demographic and professional factors among teachers, and how these factors, including AI use, influence the beliefs of teachers about AI. The study involved 117 English as a Foreign Language (EFL) in-service teachers working in Cyprus, who completed an online survey designed to assess their beliefs about the effectiveness of AI, its drawbacks, and their willingness to integrate AI into their teaching practices. The results revealed that teachers were significantly more likely to agree on the perceived effectiveness of AI and their willingness to adopt it, compared to their concerns about its use. Furthermore, teachers working in higher education and adult education, as well as those who had received more extensive training, reported using AI more frequently in their teaching. Teachers who utilized AI more often expressed stronger agreement with its effectiveness, while those who had received more training were less likely to express concerns about its integration. Given the limited training that many teachers currently receive, these findings demonstrate the need for tailored training sessions that address the specific needs and concerns of educators, ultimately fostering the adoption of AI in pronunciation instruction. 

**Abstract (ZH)**: 外国语言课堂中的发音教学指令往往是一个被忽视的研究领域。随着人工智能（AI）的广泛应用及其潜在益处，探讨AI在发音教学中的使用方式以及教师对这一工具的看法对于改善学习成果至关重要。本研究旨在探究不同人口统计学和专业因素下教师在发音教学中使用AI的差异，并分析这些因素，包括AI使用情况，如何影响教师对AI的看法。研究对象为来自塞浦路斯的117名英语作为外语（EFL）在职教师，他们完成了旨在评估其对AI有效性的看法、其局限性及其在教学中整合AI意愿的在线问卷。结果表明，教师们更倾向于认为AI的有效性，并愿意采用它，而不是对其使用的担忧。此外，从事高等教育和成人教育的教师以及接受过更广泛培训的教师报告称在教学中更频繁地使用AI。经常使用AI的教师更认同其有效性，而接受过更多培训的教师则较少对AI的整合表达担忧。鉴于教师当前接受的有限培训，这些发现表明需要针对教育工作者的具体需求和担忧定制培训课程，最终促进AI在发音教学中的应用。 

---
# SED2AM: Solving Multi-Trip Time-Dependent Vehicle Routing Problem using Deep Reinforcement Learning 

**Title (ZH)**: SED2AM：使用深度强化学习解决多趟时间依赖车辆路线问题 

**Authors**: Arash Mozhdehi, Yunli Wang, Sun Sun, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04085)  

**Abstract**: Deep reinforcement learning (DRL)-based frameworks, featuring Transformer-style policy networks, have demonstrated their efficacy across various vehicle routing problem (VRP) variants. However, the application of these methods to the multi-trip time-dependent vehicle routing problem (MTTDVRP) with maximum working hours constraints -- a pivotal element of urban logistics -- remains largely unexplored. This paper introduces a DRL-based method called the Simultaneous Encoder and Dual Decoder Attention Model (SED2AM), tailored for the MTTDVRP with maximum working hours constraints. The proposed method introduces a temporal locality inductive bias to the encoding module of the policy networks, enabling it to effectively account for the time-dependency in travel distance or time. The decoding module of SED2AM includes a vehicle selection decoder that selects a vehicle from the fleet, effectively associating trips with vehicles for functional multi-trip routing. Additionally, this decoding module is equipped with a trip construction decoder leveraged for constructing trips for the vehicles. This policy model is equipped with two classes of state representations, fleet state and routing state, providing the information needed for effective route construction in the presence of maximum working hours constraints. Experimental results using real-world datasets from two major Canadian cities not only show that SED2AM outperforms the current state-of-the-art DRL-based and metaheuristic-based baselines but also demonstrate its generalizability to solve larger-scale problems. 

**Abstract (ZH)**: 基于深强化学习（DRL）且采用Transformer风格策略网络的框架在各种车辆路线问题（VRP）变体中展现了其有效性。然而，这些方法在具有最大工作时间约束的多趟时间依赖车辆路线问题（MTTDVRP）中的应用——这是城市物流的关键要素——仍然鲜有探索。本文提出了一种名为Simultaneous Encoder and Dual Decoder Attention Model（SED2AM）的基于DRL的方法，专门针对具有最大工作时间约束的MTTDVRP。所提出的方法在策略网络的编码模块中引入了时间局部性归纳偏置，使其能够有效地考虑旅行距离或时间的时间依赖性。SED2AM的解码模块包括一个车辆选择解码器，可以有效地将趟次与车辆关联起来实现功能性的多趟路由，同时还配备了用于为车辆构建趟次的行程构建解码器。该策略模型配备了两类状态表示——车队状态和路由状态——提供了在存在最大工作时间约束情况下有效构建路线所需的信息。使用来自两个主要加拿大城市的现实世界数据集进行的实验结果不仅表明SED2AM在当前最先进的基于DRL和元启发式的基线方法中具有优越性，还展示了其解决更大规模问题的普适性。 

---
# Learning to Negotiate via Voluntary Commitment 

**Title (ZH)**: 通过自愿承诺学习谈判 

**Authors**: Shuhui Zhu, Baoxiang Wang, Sriram Ganapathi Subramanian, Pascal Poupart  

**Link**: [PDF](https://arxiv.org/pdf/2503.03866)  

**Abstract**: The partial alignment and conflict of autonomous agents lead to mixed-motive scenarios in many real-world applications. However, agents may fail to cooperate in practice even when cooperation yields a better outcome. One well known reason for this failure comes from non-credible commitments. To facilitate commitments among agents for better cooperation, we define Markov Commitment Games (MCGs), a variant of commitment games, where agents can voluntarily commit to their proposed future plans. Based on MCGs, we propose a learnable commitment protocol via policy gradients. We further propose incentive-compatible learning to accelerate convergence to equilibria with better social welfare. Experimental results in challenging mixed-motive tasks demonstrate faster empirical convergence and higher returns for our method compared with its counterparts. Our code is available at this https URL. 

**Abstract (ZH)**: 自主代理的部分对齐与冲突导致许多实际应用中的混合动机场景。即使合作能够产生更好的结果，代理在实践中也可能无法实现合作。这种失败的一个已知原因是不可信的承诺。为促进代理之间的承诺以实现更好的合作，我们定义了马尔可夫承诺博弈（MCGs），这是一种承诺博弈的变体，其中代理可以自愿承诺其提出的未来计划。基于MCGs，我们提出了一种通过策略梯度学习的可学习承诺协议。我们进一步提出了一种激励相容学习方法，以加速收敛并获得更好的社会效益。实验结果表明，与同类方法相比，我们的方法在具有挑战性的混合动机任务中实现了更快的实际收敛和更高的回报。我们的代码可在以下链接获取：this https URL。 

---
# Predicting Team Performance from Communications in Simulated Search-and-Rescue 

**Title (ZH)**: 从模拟搜索救援中通信预测团队绩效 

**Authors**: Ali Jalal-Kamali, Nikolos Gurney, David Pynadath  

**Link**: [PDF](https://arxiv.org/pdf/2503.03791)  

**Abstract**: Understanding how individual traits influence team performance is valuable, but these traits are not always directly observable. Prior research has inferred traits like trust from behavioral data. We analyze conversational data to identify team traits and their correlation with teaming outcomes. Using transcripts from a Minecraft-based search-and-rescue experiment, we apply topic modeling and clustering to uncover key interaction patterns. Our findings show that variations in teaming outcomes can be explained through these inferences, with different levels of predictive power derived from individual traits and team dynamics. 

**Abstract (ZH)**: 理解个体特质如何影响团队表现是有价值的，但这些特质并不总是直接可观测的。先前的研究通过行为数据推断出诸如信任之类的特质。我们分析对话数据以识别团队特质及其与团队成果的相关性。使用基于 Minecraft 的搜索和救援实验的转录数据，我们应用主题建模和聚类来发现关键的交互模式。我们的研究结果表明，团队成果的变化可以通过这些推断来解释，个体特质和团队动态的不同水平可以预测这些变化。 

---
# Fair Play in the Fast Lane: Integrating Sportsmanship into Autonomous Racing Systems 

**Title (ZH)**: 公平竞赛在快车道上的实现：将运动精神融入自主赛车系统 

**Authors**: Zhenmin Huang, Ce Hao, Wei Zhan, Jun Ma, Masayoshi Tomizuka  

**Link**: [PDF](https://arxiv.org/pdf/2503.03774)  

**Abstract**: Autonomous racing has gained significant attention as a platform for high-speed decision-making and motion control. While existing methods primarily focus on trajectory planning and overtaking strategies, the role of sportsmanship in ensuring fair competition remains largely unexplored. In human racing, rules such as the one-motion rule and the enough-space rule prevent dangerous and unsportsmanlike behavior. However, autonomous racing systems often lack mechanisms to enforce these principles, potentially leading to unsafe maneuvers. This paper introduces a bi-level game-theoretic framework to integrate sportsmanship (SPS) into versus racing. At the high level, we model racing intentions using a Stackelberg game, where Monte Carlo Tree Search (MCTS) is employed to derive optimal strategies. At the low level, vehicle interactions are formulated as a Generalized Nash Equilibrium Problem (GNEP), ensuring that all agents follow sportsmanship constraints while optimizing their trajectories. Simulation results demonstrate the effectiveness of the proposed approach in enforcing sportsmanship rules while maintaining competitive performance. We analyze different scenarios where attackers and defenders adhere to or disregard sportsmanship rules and show how knowledge of these constraints influences strategic decision-making. This work highlights the importance of balancing competition and fairness in autonomous racing and provides a foundation for developing ethical and safe AI-driven racing systems. 

**Abstract (ZH)**: 自主赛车比赛作为一种高速决策和运动控制的平台引起了广泛关注。虽然现有方法主要集中在轨迹规划和超越策略上，但在确保公平竞争中体育精神的作用仍被很大程度上忽视。在人类赛车中，一动规则和足够空间规则等规则可以防止危险和不道德行为。然而，自主赛车系统往往缺乏执行这些原则的机制，可能导致不安全的操作。本文介绍了一种多层次的游戏理论框架，将体育精神（SPS）整合到对抗赛车中。在高层次上，我们使用Stackelberg博弈来建模赛车意图，并利用蒙特卡洛树搜索（MCTS）来推导最优策略。在低层次上，车辆交互被形式化为广义纳什均衡问题（GNEP），以确保所有代理都遵循体育精神约束并优化它们的轨迹。仿真结果显示，所提出的方法在维护竞争力的同时有效执行体育精神规则。我们分析了攻击者和防守者遵守或忽视体育精神规则的不同情境，并展示了这些约束知识如何影响战略决策。本文强调了在自主赛车中平衡竞争与公平的重要性，并为开发具有伦理和安全性的人工智能驱动赛车系统奠定了基础。 

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
# Scaling Rich Style-Prompted Text-to-Speech Datasets 

**Title (ZH)**: 基于丰富风格提示的文本转语音数据集扩展 

**Authors**: Anuj Diwan, Zhisheng Zheng, David Harwath, Eunsol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2503.04713)  

**Abstract**: We introduce Paralinguistic Speech Captions (ParaSpeechCaps), a large-scale dataset that annotates speech utterances with rich style captions. While rich abstract tags (e.g. guttural, nasal, pained) have been explored in small-scale human-annotated datasets, existing large-scale datasets only cover basic tags (e.g. low-pitched, slow, loud). We combine off-the-shelf text and speech embedders, classifiers and an audio language model to automatically scale rich tag annotations for the first time. ParaSpeechCaps covers a total of 59 style tags, including both speaker-level intrinsic tags and utterance-level situational tags. It consists of 342 hours of human-labelled data (PSC-Base) and 2427 hours of automatically annotated data (PSC-Scaled). We finetune Parler-TTS, an open-source style-prompted TTS model, on ParaSpeechCaps, and achieve improved style consistency (+7.9% Consistency MOS) and speech quality (+15.5% Naturalness MOS) over the best performing baseline that combines existing rich style tag datasets. We ablate several of our dataset design choices to lay the foundation for future work in this space. Our dataset, models and code are released at this https URL . 

**Abstract (ZH)**: 我们介绍Paralinguistic Speech Captions（Paralinguistic SpeechCaps），一个大规模数据集，用于用丰富的风格注释标注语音片段。尽管在小规模的人工标注数据集中已经探索了丰富的抽象标签（例如喉音、鼻音、痛苦），但现有的大规模数据集只涵盖了基本标签（例如低音、缓慢、大声）。我们将即用型文本和语音嵌入器、分类器以及音频语言模型结合，首次自动扩展丰富的标签注释。Paralinguistic SpeechCaps 包括总计59种风格标签，涵盖演讲者级别的固有标签和片段级别的情境标签。该数据集包含342小时的人工标注数据（PSC-Base）和2427小时的自动标注数据（PSC-Scaled）。我们针对Paralinguistic SpeechCaps微调一个开源的风格提示音合成模型Parler-TTS，并在综合现有丰富风格标签数据集的基线上实现了更好的风格一致性（+7.9%一致性MOS）和语音质量（+15.5%自然度MOS）。我们通过消除部分数据集设计选择为基础未来在此领域的工作奠定基础。我们的数据集、模型和代码在此链接发布：this https URL。 

---
# Self-Supervised Models for Phoneme Recognition: Applications in Children's Speech for Reading Learning 

**Title (ZH)**: 自监督模型在儿童语音识别中的应用：阅读学习中的发音识别 

**Authors**: Lucas Block Medin, Thomas Pellegrini, Lucile Gelin  

**Link**: [PDF](https://arxiv.org/pdf/2503.04710)  

**Abstract**: Child speech recognition is still an underdeveloped area of research due to the lack of data (especially on non-English languages) and the specific difficulties of this task. Having explored various architectures for child speech recognition in previous work, in this article we tackle recent self-supervised models. We first compare wav2vec 2.0, HuBERT and WavLM models adapted to phoneme recognition in French child speech, and continue our experiments with the best of them, WavLM base+. We then further adapt it by unfreezing its transformer blocks during fine-tuning on child speech, which greatly improves its performance and makes it significantly outperform our base model, a Transformer+CTC. Finally, we study in detail the behaviour of these two models under the real conditions of our application, and show that WavLM base+ is more robust to various reading tasks and noise levels. Index Terms: speech recognition, child speech, self-supervised learning 

**Abstract (ZH)**: 儿童语音识别仍是一个由于数据不足（尤其是非英语语言数据）和此任务的具体困难而未充分开发的研究领域。在先前工作中探索了各种儿童语音识别架构后，本文着眼于近期的自监督模型。我们首先比较了适用于法语儿童语音音素识别的wav2vec 2.0、HuBERT和WavLM模型，并在这些模型中选择最优者WavLM base+ 进行进一步实验。接着，在儿童语音识别上解冻其Transformer块进行微调，这大大提高了模型性能，使其显著优于我们的基模型Transformer+CTC。最后，我们在实际应用条件下详细研究了这两种模型的行为，并表明WavLM base+ 对各种阅读任务和噪声水平具有更高的鲁棒性。关键词：语音识别，儿童语音，自监督学习 

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
# Matrix Factorization for Inferring Associations and Missing Links 

**Title (ZH)**: 矩阵因子分解用于推断关联和缺失链接 

**Authors**: Ryan Barron, Maksim E. Eren, Duc P. Truong, Cynthia Matuszek, James Wendelberger, Mary F. Dorn, Boian Alexandrov  

**Link**: [PDF](https://arxiv.org/pdf/2503.04680)  

**Abstract**: Missing link prediction is a method for network analysis, with applications in recommender systems, biology, social sciences, cybersecurity, information retrieval, and Artificial Intelligence (AI) reasoning in Knowledge Graphs. Missing link prediction identifies unseen but potentially existing connections in a network by analyzing the observed patterns and relationships. In proliferation detection, this supports efforts to identify and characterize attempts by state and non-state actors to acquire nuclear weapons or associated technology - a notoriously challenging but vital mission for global security. Dimensionality reduction techniques like Non-Negative Matrix Factorization (NMF) and Logistic Matrix Factorization (LMF) are effective but require selection of the matrix rank parameter, that is, of the number of hidden features, k, to avoid over/under-fitting. We introduce novel Weighted (WNMFk), Boolean (BNMFk), and Recommender (RNMFk) matrix factorization methods, along with ensemble variants incorporating logistic factorization, for link prediction. Our methods integrate automatic model determination for rank estimation by evaluating stability and accuracy using a modified bootstrap methodology and uncertainty quantification (UQ), assessing prediction reliability under random perturbations. We incorporate Otsu threshold selection and k-means clustering for Boolean matrix factorization, comparing them to coordinate descent-based Boolean thresholding. Our experiments highlight the impact of rank k selection, evaluate model performance under varying test-set sizes, and demonstrate the benefits of UQ for reliable predictions using abstention. We validate our methods on three synthetic datasets (Boolean and uniformly distributed) and benchmark them against LMF and symmetric LMF (symLMF) on five real-world protein-protein interaction networks, showcasing an improved prediction performance. 

**Abstract (ZH)**: 缺失边预测是一种网络分析方法，应用于推荐系统、生物、社会科学、网络安全、信息检索以及知识图谱中的AI推理。缺失边预测通过分析观测模式和关系来识别潜在存在的但未被观察到的网络连接，在 proliferation 检测中支持识别和表征国家及非国家行为体获取核武器或相关技术的企图，这是一项具有挑战性但至关重要的全球安全任务。非负矩阵分解（NMF）和逻辑矩阵分解（LMF）等降维技术有效但需要选择矩阵秩参数，即隐藏特征的数量 k，以避免过拟合或欠拟合。我们引入了带权矩阵分解（WNMFk）、布尔矩阵分解（BNMFk）和推荐矩阵分解（RNMFk）方法及其结合逻辑因子分解的集成变体，用于边预测。我们的方法通过评价稳定性与准确性来实现自动模型选择，利用修改后的Bootstrap方法和不确定性量化（UQ）评估预测可靠性。我们结合了Otsu阈值选择和k-means聚类进行布尔矩阵分解，并将其与基于坐标下降的布尔阈值方法进行了比较。实验结果突出了秩 k 选择的影响，评估了模型性能在不同测试集大小下的表现，并展示了UQ在可靠预测中的优势，通过避免预测来展示其益处。我们在三个合成数据集（布尔型和均匀分布）上验证了这些方法，并使用LMF和对称LMF（symLMF）在五个真实世界蛋白质-蛋白质相互作用网络上进行了基准测试，展示了更好的预测性能。 

---
# Multi-Agent Inverse Q-Learning from Demonstrations 

**Title (ZH)**: 基于演示的多agent逆Q学习 

**Authors**: Nathaniel Haynam, Adam Khoja, Dhruv Kumar, Vivek Myers, Erdem Bıyık  

**Link**: [PDF](https://arxiv.org/pdf/2503.04679)  

**Abstract**: When reward functions are hand-designed, deep reinforcement learning algorithms often suffer from reward misspecification, causing them to learn suboptimal policies in terms of the intended task objectives. In the single-agent case, inverse reinforcement learning (IRL) techniques attempt to address this issue by inferring the reward function from expert demonstrations. However, in multi-agent problems, misalignment between the learned and true objectives is exacerbated due to increased environment non-stationarity and variance that scales with multiple agents. As such, in multi-agent general-sum games, multi-agent IRL algorithms have difficulty balancing cooperative and competitive objectives. To address these issues, we propose Multi-Agent Marginal Q-Learning from Demonstrations (MAMQL), a novel sample-efficient framework for multi-agent IRL. For each agent, MAMQL learns a critic marginalized over the other agents' policies, allowing for a well-motivated use of Boltzmann policies in the multi-agent context. We identify a connection between optimal marginalized critics and single-agent soft-Q IRL, allowing us to apply a direct, simple optimization criterion from the single-agent domain. Across our experiments on three different simulated domains, MAMQL significantly outperforms previous multi-agent methods in average reward, sample efficiency, and reward recovery by often more than 2-5x. We make our code available at this https URL . 

**Abstract (ZH)**: 多代理边际Q学习从示范中学习（MAMQL）：一种新的多代理逆强化学习高效框架 

---
# Implicit Cross-Lingual Rewarding for Efficient Multilingual Preference Alignment 

**Title (ZH)**: 隐式跨语言奖励以实现高效的多语言偏好对齐 

**Authors**: Wen Yang, Junhong Wu, Chen Wang, Chengqing Zong, Jiajun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04647)  

**Abstract**: Direct Preference Optimization (DPO) has become a prominent method for aligning Large Language Models (LLMs) with human preferences. While DPO has enabled significant progress in aligning English LLMs, multilingual preference alignment is hampered by data scarcity. To address this, we propose a novel approach that $\textit{captures}$ learned preferences from well-aligned English models by implicit rewards and $\textit{transfers}$ them to other languages through iterative training. Specifically, we derive an implicit reward model from the logits of an English DPO-aligned model and its corresponding reference model. This reward model is then leveraged to annotate preference relations in cross-lingual instruction-following pairs, using English instructions to evaluate multilingual responses. The annotated data is subsequently used for multilingual DPO fine-tuning, facilitating preference knowledge transfer from English to other languages. Fine-tuning Llama3 for two iterations resulted in a 12.72% average improvement in Win Rate and a 5.97% increase in Length Control Win Rate across all training languages on the X-AlpacaEval leaderboard. Our findings demonstrate that leveraging existing English-aligned models can enable efficient and effective multilingual preference alignment, significantly reducing the need for extensive multilingual preference data. The code is available at this https URL 

**Abstract (ZH)**: 直接偏好优化(DPO)已成为将大型语言模型(LLMs)与人类偏好对齐的 prominent 方法。尽管DPO已在对齐英语LLMs方面取得了显著进展，但多语言偏好对齐受限于数据稀缺性。为解决这一问题，我们提出了一种新方法，该方法通过隐式奖励 $\textit{捕获}$ 已对齐的英语模型中的偏好，并通过迭代训练 $\textit{转移}$ 至其他语言。具体来说，我们从英语DPO对齐模型及其对应参考模型的logits中推导出一个隐式奖励模型。然后利用该奖励模型对跨语言指令跟随配对中的偏好关系进行标注，使用英语指令评估多语言响应。标注的数据随后用于多语言DPO微调，促进偏好知识从英语向其他语言的转移。对Llama3进行两轮微调后，我们在X-AlpacaEval排行榜上所有训练语言中平均提高了12.72%的胜率，并提高了5.97%的长度控制胜率。我们的研究结果表明，利用现有的英语对齐模型可以实现高效的多语言偏好对齐，显著减少了对大量多语言偏好数据的需求。代码可在以下链接获取：this https URL。 

---
# Simulating the Real World: A Unified Survey of Multimodal Generative Models 

**Title (ZH)**: 模拟现实世界：多模态生成模型综述 

**Authors**: Yuqi Hu, Longguang Wang, Xian Liu, Ling-Hao Chen, Yuwei Guo, Yukai Shi, Ce Liu, Anyi Rao, Zeyu Wang, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.04641)  

**Abstract**: Understanding and replicating the real world is a critical challenge in Artificial General Intelligence (AGI) research. To achieve this, many existing approaches, such as world models, aim to capture the fundamental principles governing the physical world, enabling more accurate simulations and meaningful interactions. However, current methods often treat different modalities, including 2D (images), videos, 3D, and 4D representations, as independent domains, overlooking their interdependencies. Additionally, these methods typically focus on isolated dimensions of reality without systematically integrating their connections. In this survey, we present a unified survey for multimodal generative models that investigate the progression of data dimensionality in real-world simulation. Specifically, this survey starts from 2D generation (appearance), then moves to video (appearance+dynamics) and 3D generation (appearance+geometry), and finally culminates in 4D generation that integrate all dimensions. To the best of our knowledge, this is the first attempt to systematically unify the study of 2D, video, 3D and 4D generation within a single framework. To guide future research, we provide a comprehensive review of datasets, evaluation metrics and future directions, and fostering insights for newcomers. This survey serves as a bridge to advance the study of multimodal generative models and real-world simulation within a unified framework. 

**Abstract (ZH)**: 理解并重现现实世界是人工通用智能（AGI）研究中的一个关键挑战。为此，许多现有方法，比如世界模型，旨在捕捉支配物理世界的基本原则，从而实现更准确的仿真和更有意义的交互。然而，当前的方法通常将不同的模态，包括2D（图像）、视频、3D和4D表示，视为独立的领域，忽视了它们之间的相互依赖性。此外，这些方法通常专注于现实的孤立维度，而不系统地整合它们之间的连接。在本文综述中，我们提出了一种统一的多模态生成模型综述，探讨现实世界仿真中的数据维度进展。具体而言，本文综述从2D生成（外观）开始，然后过渡到视频（外观+动态）和3D生成（外观+几何），最终达到整合所有维度的4D生成。据我们所知，这是我们首次尝试系统地在单一框架中统一研究2D、视频、3D和4D生成。为了指导未来的研究，我们提供了全面的数据集、评估指标和未来方向的综述，并为新进入者提供洞见。本文综述旨在作为桥梁，促进在统一框架下对多模态生成模型和现实世界仿真的研究。 

---
# Mark Your LLM: Detecting the Misuse of Open-Source Large Language Models via Watermarking 

**Title (ZH)**: 标记你的LLM：通过水印技术检测开源大规模语言模型的滥用 

**Authors**: Yijie Xu, Aiwei Liu, Xuming Hu, Lijie Wen, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.04636)  

**Abstract**: As open-source large language models (LLMs) like Llama3 become more capable, it is crucial to develop watermarking techniques to detect their potential misuse. Existing watermarking methods either add watermarks during LLM inference, which is unsuitable for open-source LLMs, or primarily target classification LLMs rather than recent generative LLMs. Adapting these watermarks to open-source LLMs for misuse detection remains an open challenge. This work defines two misuse scenarios for open-source LLMs: intellectual property (IP) violation and LLM Usage Violation. Then, we explore the application of inference-time watermark distillation and backdoor watermarking in these contexts. We propose comprehensive evaluation methods to assess the impact of various real-world further fine-tuning scenarios on watermarks and the effect of these watermarks on LLM performance. Our experiments reveal that backdoor watermarking could effectively detect IP Violation, while inference-time watermark distillation is applicable in both scenarios but less robust to further fine-tuning and has a more significant impact on LLM performance compared to backdoor watermarking. Exploring more advanced watermarking methods for open-source LLMs to detect their misuse should be an important future direction. 

**Abstract (ZH)**: 开源大规模语言模型的滥用检测 watermarking 技术：针对 Llama3 等开源大语言模型的知识产权违规和模型使用违规的滥用检测方法探究 

---
# IDInit: A Universal and Stable Initialization Method for Neural Network Training 

**Title (ZH)**: IDInit: 一种通用且稳定的神经网络训练初始化方法 

**Authors**: Yu Pan, Chaozheng Wang, Zekai Wu, Qifan Wang, Min Zhang, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04626)  

**Abstract**: Deep neural networks have achieved remarkable accomplishments in practice. The success of these networks hinges on effective initialization methods, which are vital for ensuring stable and rapid convergence during training. Recently, initialization methods that maintain identity transition within layers have shown good efficiency in network training. These techniques (e.g., Fixup) set specific weights to zero to achieve identity control. However, settings of remaining weight (e.g., Fixup uses random values to initialize non-zero weights) will affect the inductive bias that is achieved only by a zero weight, which may be harmful to training. Addressing this concern, we introduce fully identical initialization (IDInit), a novel method that preserves identity in both the main and sub-stem layers of residual networks. IDInit employs a padded identity-like matrix to overcome rank constraints in non-square weight matrices. Furthermore, we show the convergence problem of an identity matrix can be solved by stochastic gradient descent. Additionally, we enhance the universality of IDInit by processing higher-order weights and addressing dead neuron problems. IDInit is a straightforward yet effective initialization method, with improved convergence, stability, and performance across various settings, including large-scale datasets and deep models. 

**Abstract (ZH)**: 深层神经网络在实践中取得了显著成就。这些网络的成功依赖于有效的初始化方法，这些方法对于确保训练期间的稳定和快速收敛至关重要。近年来，在层内保持身份过渡的初始化方法在网络训练中显示出良好的效率。这些技术（例如，Fixup）通过将特定权重设为零来实现身份控制。然而，剩余权重的设置（例如，Fixup 使用随机值初始化非零权重）会影响仅由零权重实现的归纳偏置，这可能对训练有害。为解决这一问题，我们引入了一种新颖的方法——全身份初始化（IDInit），该方法在残差网络的主要层和子层中均保持身份。IDInit 使用填充的身份矩阵来克服非方矩阵的秩约束。此外，我们通过处理高阶权重和解决死亡神经元问题，展示了身份矩阵收敛问题可以通过随机梯度下降解决。同时，我们通过增强IDInit的通用性来处理更广泛的情况。IDInit 是一个简单而有效的初始化方法，具有改进的收敛性、稳定性和性能，在包括大规模数据集和深度模型的各种设置中均适用。 

---
# The Best of Both Worlds: Integrating Language Models and Diffusion Models for Video Generation 

**Title (ZH)**: 兼收并蓄：将语言模型与扩散模型结合用于视频生成 

**Authors**: Aoxiong Yin, Kai Shen, Yichong Leng, Xu Tan, Xinyu Zhou, Juncheng Li, Siliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04606)  

**Abstract**: Recent advancements in text-to-video (T2V) generation have been driven by two competing paradigms: autoregressive language models and diffusion models. However, each paradigm has intrinsic limitations: language models struggle with visual quality and error accumulation, while diffusion models lack semantic understanding and causal modeling. In this work, we propose LanDiff, a hybrid framework that synergizes the strengths of both paradigms through coarse-to-fine generation. Our architecture introduces three key innovations: (1) a semantic tokenizer that compresses 3D visual features into compact 1D discrete representations through efficient semantic compression, achieving a $\sim$14,000$\times$ compression ratio; (2) a language model that generates semantic tokens with high-level semantic relationships; (3) a streaming diffusion model that refines coarse semantics into high-fidelity videos. Experiments show that LanDiff, a 5B model, achieves a score of 85.43 on the VBench T2V benchmark, surpassing the state-of-the-art open-source models Hunyuan Video (13B) and other commercial models such as Sora, Keling, and Hailuo. Furthermore, our model also achieves state-of-the-art performance in long video generation, surpassing other open-source models in this field. Our demo can be viewed at this https URL. 

**Abstract (ZH)**: Recent advancements in text-to-video (T2V) generation have been driven by two competing paradigms: autoregressive language models and diffusion models. However, each paradigm has intrinsic limitations: language models struggle with visual quality and error accumulation, while diffusion models lack semantic understanding and causal modeling. In this work, we propose LanDiff, a hybrid framework that synergizes the strengths of both paradigms through coarse-to-fine generation. 

---
# HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization 

**Title (ZH)**: HybridNorm: 向量稳定且高效的变换器训练通过混合规范化 

**Authors**: Zhijian Zhuo, Yutao Zeng, Ya Wang, Sijun Zhang, Jian Yang, Xiaoqing Li, Xun Zhou, Jinwen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.04598)  

**Abstract**: Transformers have become the de facto architecture for a wide range of machine learning tasks, particularly in large language models (LLMs). Despite their remarkable performance, challenges remain in training deep transformer networks, especially regarding the location of layer normalization. While Pre-Norm structures facilitate easier training due to their more prominent identity path, they often yield suboptimal performance compared to Post-Norm. In this paper, we propose $\textbf{HybridNorm}$, a straightforward yet effective hybrid normalization strategy that integrates the advantages of both Pre-Norm and Post-Norm approaches. Specifically, HybridNorm employs QKV normalization within the attention mechanism and Post-Norm in the feed-forward network (FFN) of each transformer block. This design not only stabilizes training but also enhances performance, particularly in the context of LLMs. Comprehensive experiments in both dense and sparse architectures show that HybridNorm consistently outperforms both Pre-Norm and Post-Norm approaches, achieving state-of-the-art results across various benchmarks. These findings highlight the potential of HybridNorm as a more stable and effective technique for improving the training and performance of deep transformer models. %Code will be made publicly available. Code is available at this https URL. 

**Abstract (ZH)**: HybridNorm：一种结合Pre-Norm和Post-Norm优势的混合归一化策略 

---
# The Next Frontier of LLM Applications: Open Ecosystems and Hardware Synergy 

**Title (ZH)**: LLM应用的下一个前沿：开放生态系统与硬件协同 

**Authors**: Xinyi Hou, Yanjie Zhao, Haoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04596)  

**Abstract**: Large Language Model (LLM) applications, including LLM app stores and autonomous agents, are shaping the future of AI ecosystems. However, platform silos, fragmented hardware integration, and the absence of standardized interfaces limit scalability, interoperability, and resource efficiency. While LLM app stores democratize AI, their closed ecosystems restrict modular AI reuse and cross-platform portability. Meanwhile, agent-based frameworks offer flexibility but often lack seamless integration across diverse environments. This paper envisions the future of LLM applications and proposes a three-layer decoupled architecture grounded in software engineering principles such as layered system design, service-oriented architectures, and hardware-software co-design. This architecture separates application logic, communication protocols, and hardware execution, enhancing modularity, efficiency, and cross-platform compatibility. Beyond architecture, we highlight key security and privacy challenges for safe, scalable AI deployment and outline research directions in software and security engineering. This vision aims to foster open, secure, and interoperable LLM ecosystems, guiding future advancements in AI applications. 

**Abstract (ZH)**: 大型语言模型（LLM）应用，包括LLM应用商店和自主代理，正在塑造未来的AI生态系统。然而，平台孤岛、碎片化的硬件集成以及缺乏标准化接口限制了可扩展性、兼容性和资源效率。虽然LLM应用商店使AI民主化，但它们封闭的生态系统限制了模块化AI的重用和跨平台移植。与此同时，基于代理的框架提供了灵活性，但通常跨不同环境的无缝集成能力不足。本文设想了LLM应用的未来，并提出了一种三层解耦架构，该架构基于分层系统设计、面向服务的架构和硬件-软件协同设计等软件工程原则。该架构将应用程序逻辑、通信协议和硬件执行分层，增强模块化、效率和跨平台兼容性。展望架构之外，我们强调了安全和隐私的关键挑战，以确保大规模AI部署的安全性和可扩展性，并概述了软件和安全工程方向的研究方向。该愿景旨在促进开放、安全和兼容的LLM生态系统，指导未来AI应用的发展。 

---
# Fundamental Limits of Hierarchical Secure Aggregation with Cyclic User Association 

**Title (ZH)**: 层次安全聚合中循环用户关联的基本界限 

**Authors**: Xiang Zhang, Zhou Li, Kai Wan, Hua Sun, Mingyue Ji, Giuseppe Caire  

**Link**: [PDF](https://arxiv.org/pdf/2503.04564)  

**Abstract**: Secure aggregation is motivated by federated learning (FL) where a cloud server aims to compute an averaged model (i.e., weights of deep neural networks) of the locally-trained models of numerous clients, while adhering to data security requirements. Hierarchical secure aggregation (HSA) extends this concept to a three-layer network, where clustered users communicate with the server through an intermediate layer of relays. In HSA, beyond conventional server security, relay security is also enforced to ensure that the relays remain oblivious to the users' inputs (an abstraction of the local models in FL). Existing study on HSA assumes that each user is associated with only one relay, limiting opportunities for coding across inter-cluster users to achieve efficient communication and key generation. In this paper, we consider HSA with a cyclic association pattern where each user is connected to $B$ consecutive relays in a wrap-around manner. We propose an efficient aggregation scheme which includes a message design for the inputs inspired by gradient coding-a well-known technique for efficient communication in distributed computing-along with a highly nontrivial security key design. We also derive novel converse bounds on the minimum achievable communication and key rates using information-theoretic arguments. 

**Abstract (ZH)**: 基于联邦学习的分层安全聚合：每用户关联连续若干relay的周期性关联模式研究 

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
# STX-Search: Explanation Search for Continuous Dynamic Spatio-Temporal Models 

**Title (ZH)**: STX-Search：连续动态时空模型的解释性搜索 

**Authors**: Saif Anwar, Nathan Griffiths, Thomas Popham, Abhir Bhalerao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04509)  

**Abstract**: Recent improvements in the expressive power of spatio-temporal models have led to performance gains in many real-world applications, such as traffic forecasting and social network modelling. However, understanding the predictions from a model is crucial to ensure reliability and trustworthiness, particularly for high-risk applications, such as healthcare and transport. Few existing methods are able to generate explanations for models trained on continuous-time dynamic graph data and, of these, the computational complexity and lack of suitable explanation objectives pose challenges. In this paper, we propose $\textbf{S}$patio-$\textbf{T}$emporal E$\textbf{X}$planation $\textbf{Search}$ (STX-Search), a novel method for generating instance-level explanations that is applicable to static and dynamic temporal graph structures. We introduce a novel search strategy and objective function, to find explanations that are highly faithful and interpretable. When compared with existing methods, STX-Search produces explanations of higher fidelity whilst optimising explanation size to maintain interpretability. 

**Abstract (ZH)**: 时空解释搜索：适用于静态和动态时空图结构的实例级解释生成方法 

---
# Multi-modal Summarization in Model-Based Engineering: Automotive Software Development Case Study 

**Title (ZH)**: 基于模型的工程中的多模态总结：汽车软件开发案例研究 

**Authors**: Nenad Petrovic, Yurui Zhang, Moaad Maaroufi, Kuo-Yi Chao, Lukasz Mazur, Fengjunjie Pan, Vahid Zolfaghari, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2503.04506)  

**Abstract**: Multimodal summarization integrating information from diverse data modalities presents a promising solution to aid the understanding of information within various processes. However, the application and advantages of multimodal summarization have not received much attention in model-based engineering (MBE), where it has become a cornerstone in the design and development of complex systems, leveraging formal models to improve understanding, validation and automation throughout the engineering lifecycle. UML and EMF diagrams in model-based engineering contain a large amount of multimodal information and intricate relational data. Hence, our study explores the application of multimodal large language models within the domain of model-based engineering to evaluate their capacity for understanding and identifying relationships, features, and functionalities embedded in UML and EMF diagrams. We aim to demonstrate the transformative potential benefits and limitations of multimodal summarization in improving productivity and accuracy in MBE practices. The proposed approach is evaluated within the context of automotive software development, while many promising state-of-art models were taken into account. 

**Abstract (ZH)**: 多模态总结在模型基础工程中的应用及其对理解和识别UML和EMF图中嵌入的关系、特性和功能的潜力研究 

---
# Interpretable Transformation and Analysis of Timelines through Learning via Surprisability 

**Title (ZH)**: 通过学习 Surpriseability 进行可解释的时间线转换与分析 

**Authors**: Osnat Mokryn, Teddy Lazebnik, Hagit Ben Shoshan  

**Link**: [PDF](https://arxiv.org/pdf/2503.04502)  

**Abstract**: The analysis of high-dimensional timeline data and the identification of outliers and anomalies is critical across diverse domains, including sensor readings, biological and medical data, historical records, and global statistics. However, conventional analysis techniques often struggle with challenges such as high dimensionality, complex distributions, and sparsity. These limitations hinder the ability to extract meaningful insights from complex temporal datasets, making it difficult to identify trending features, outliers, and anomalies effectively. Inspired by surprisability -- a cognitive science concept describing how humans instinctively focus on unexpected deviations - we propose Learning via Surprisability (LvS), a novel approach for transforming high-dimensional timeline data. LvS quantifies and prioritizes anomalies in time-series data by formalizing deviations from expected behavior. LvS bridges cognitive theories of attention with computational methods, enabling the detection of anomalies and shifts in a way that preserves critical context, offering a new lens for interpreting complex datasets. We demonstrate the usefulness of LvS on three high-dimensional timeline use cases: a time series of sensor data, a global dataset of mortality causes over multiple years, and a textual corpus containing over two centuries of State of the Union Addresses by U.S. presidents. Our results show that the LvS transformation enables efficient and interpretable identification of outliers, anomalies, and the most variable features along the timeline. 

**Abstract (ZH)**: 高维时间线数据中的异常和异常值分析及其识别在传感器读数、生物医学数据、历史记录和全球统计等领域至关重要。然而，传统的分析技术常常难以应对高维度、复杂分布和稀疏性等挑战。这些限制妨碍了从复杂的时间序列数据中提取有意义的见解，使得准确识别趋势特征、异常值和异常变得困难。受意外程度——这一认知科学概念描述人类如何本能地关注意外偏差的影响，我们提出了基于意外程度的学习（LvS），一种新的高维时间线数据转换方法。LvS通过形式化偏离预期行为的方式量化并优先处理异常事件。LvS将注意的认知理论与计算方法相结合，以保留关键背景的方式检测异常和变化，提供了一种解释复杂数据集的新视角。我们在三个高维时间线应用场景中展示了LvS的有效性：传感器数据的时间序列、多年来的全球死亡原因统计数据以及包含两百多年美国总统国情咨文的文本语料库。结果显示，LvS转换能够高效且可解释地识别异常值、异常以及时间线上最波动的特征。 

---
# ReynoldsFlow: Exquisite Flow Estimation via Reynolds Transport Theorem 

**Title (ZH)**: ReynoldsFlow: 通过雷诺输运定理实现精细流场估计 

**Authors**: Yu-Hsi Chen, Chin-Tien Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04500)  

**Abstract**: Optical flow is a fundamental technique for motion estimation, widely applied in video stabilization, interpolation, and object tracking. Recent advancements in artificial intelligence (AI) have enabled deep learning models to leverage optical flow as an important feature for motion analysis. However, traditional optical flow methods rely on restrictive assumptions, such as brightness constancy and slow motion constraints, limiting their effectiveness in complex scenes. Deep learning-based approaches require extensive training on large domain-specific datasets, making them computationally demanding. Furthermore, optical flow is typically visualized in the HSV color space, which introduces nonlinear distortions when converted to RGB and is highly sensitive to noise, degrading motion representation accuracy. These limitations inherently constrain the performance of downstream models, potentially hindering object tracking and motion analysis tasks. To address these challenges, we propose Reynolds flow, a novel training-free flow estimation inspired by the Reynolds transport theorem, offering a principled approach to modeling complex motion dynamics. Beyond the conventional HSV-based visualization, denoted ReynoldsFlow, we introduce an alternative representation, ReynoldsFlow+, designed to improve flow visualization. We evaluate ReynoldsFlow and ReynoldsFlow+ across three video-based benchmarks: tiny object detection on UAVDB, infrared object detection on Anti-UAV, and pose estimation on GolfDB. Experimental results demonstrate that networks trained with ReynoldsFlow+ achieve state-of-the-art (SOTA) performance, exhibiting improved robustness and efficiency across all tasks. 

**Abstract (ZH)**: 基于瑞利传输定理的无训练流场估计：Reynolds流 

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
# TPC: Cross-Temporal Prediction Connection for Vision-Language Model Hallucination Reduction 

**Title (ZH)**: TPC: 不同时间预测连接以减少视觉-语言模型的幻觉 

**Authors**: Chao Wang, Weiwei Fu, Yang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.04457)  

**Abstract**: Vision-language models (VLMs) have achieved remarkable advancements, capitalizing on the impressive capabilities of large language models (LLMs) across diverse tasks. Despite this, a critical challenge known as hallucination occurs when models overconfidently describe objects or attributes absent from the image, a problem exacerbated by the tendency of VLMs to rely on linguistic priors. This limitation reduces model reliability in high-stakes applications. In this work, we have observed the characteristic of logits' continuity consistency enhancement and introduced a straightforward and efficient method, Cross-Temporal Prediction Connection (TPC), designed to enhance the semantic consistency of logits by connecting them temporally across timesteps. TPC amplifies information flow and improves coherence, effectively reducing hallucination. Extensive experiments show that TPC surpasses existing representatives, delivering superior performance in both accuracy and efficiency while maintaining robustness in open-ended text generation tasks. 

**Abstract (ZH)**: Vision-语言模型(VLMs)在利用大型语言模型(LLMs)的多样化任务能力方面取得了显著进展，但在高风险应用中因模型对图像不存在的对象或属性过度自信地描述而引发的语言先验依赖导致幻觉问题依然存在。为解决这一问题，我们观察到了logits连续一致性增强的特点，并提出了一种简单有效的方法——跨时间预测连接(TPC)，旨在通过在时间步之间连接logits来增强语义一致性，从而增强信息流并提高连贯性，有效减少幻觉现象。广泛实验表明，TPC在准确性和效率上优于现有方法，并且在开放文本生成任务中保持了稳健性。 

---
# Privacy Preserving and Robust Aggregation for Cross-Silo Federated Learning in Non-IID Settings 

**Title (ZH)**: 非均衡分布环境下的跨机构联邦学习的隐私保护和鲁棒聚合 

**Authors**: Marco Arazzi, Mert Cihangiroglu, Antonino Nocera  

**Link**: [PDF](https://arxiv.org/pdf/2503.04451)  

**Abstract**: Federated Averaging remains the most widely used aggregation strategy in federated learning due to its simplicity and scalability. However, its performance degrades significantly in non-IID data settings, where client distributions are highly imbalanced or skewed. Additionally, it relies on clients transmitting metadata, specifically the number of training samples, which introduces privacy risks and may conflict with regulatory frameworks like the European GDPR. In this paper, we propose a novel aggregation strategy that addresses these challenges by introducing class-aware gradient masking. Unlike traditional approaches, our method relies solely on gradient updates, eliminating the need for any additional client metadata, thereby enhancing privacy protection. Furthermore, our approach validates and dynamically weights client contributions based on class-specific importance, ensuring robustness against non-IID distributions, convergence prevention, and backdoor attacks. Extensive experiments on benchmark datasets demonstrate that our method not only outperforms FedAvg and other widely accepted aggregation strategies in non-IID settings but also preserves model integrity in adversarial scenarios. Our results establish the effectiveness of gradient masking as a practical and secure solution for federated learning. 

**Abstract (ZH)**: 联邦平均仍然是联邦学习中最广泛使用的聚合策略，由于其简单性和可扩展性。然而，在非-IID数据设置中，其性能显著下降，尤其是在客户端分布高度不平衡或偏斜的情况下。此外，它依赖于客户端传输元数据，特别是训练样本的数量，这引入了隐私风险，并可能与GDPR等监管框架冲突。在本文中，我们提出了一种新的聚合策略，通过引入类别感知梯度遮蔽来应对这些挑战。与传统方法不同，我们的方法仅依赖于梯度更新，消除了对任何额外客户端元数据的需求，从而增强隐私保护。此外，我们的方法根据类别特定的重要性验证并动态加权客户端贡献，确保在非-IID分布、收敛抑制和后门攻击方面的鲁棒性。广泛的基准数据集实验表明，与联邦平均及其他广泛接受的聚合策略相比，我们的方法在非-IID设置中不仅性能更优，而且在对抗性场景下也保持了模型的一致性。我们的结果证实了梯度遮蔽作为联邦学习中实用且安全的解决方案的有效性。 

---
# PDX: A Data Layout for Vector Similarity Search 

**Title (ZH)**: PDX：一种向量相似性搜索的数据布局 

**Authors**: Leonardo Kuffo, Elena Krippner, Peter Boncz  

**Link**: [PDF](https://arxiv.org/pdf/2503.04422)  

**Abstract**: We propose Partition Dimensions Across (PDX), a data layout for vectors (e.g., embeddings) that, similar to PAX [6], stores multiple vectors in one block, using a vertical layout for the dimensions (Figure 1). PDX accelerates exact and approximate similarity search thanks to its dimension-by-dimension search strategy that operates on multiple-vectors-at-a-time in tight loops. It beats SIMD-optimized distance kernels on standard horizontal vector storage (avg 40% faster), only relying on scalar code that gets auto-vectorized. We combined the PDX layout with recent dimension-pruning algorithms ADSampling [19] and BSA [52] that accelerate approximate vector search. We found that these algorithms on the horizontal vector layout can lose to SIMD-optimized linear scans, even if they are SIMD-optimized. However, when used on PDX, their benefit is restored to 2-7x. We find that search on PDX is especially fast if a limited number of dimensions has to be scanned fully, which is what the dimension-pruning approaches do. We finally introduce PDX-BOND, an even more flexible dimension-pruning strategy, with good performance on exact search and reasonable performance on approximate search. Unlike previous pruning algorithms, it can work on vector data "as-is" without preprocessing; making it attractive for vector databases with frequent updates. 

**Abstract (ZH)**: PDX：一种用于向量的分区维度布局及其实现方法 

---
# Learning Transformer-based World Models with Contrastive Predictive Coding 

**Title (ZH)**: 基于对比预测编码的变换器式世界模型学习 

**Authors**: Maxime Burchi, Radu Timofte  

**Link**: [PDF](https://arxiv.org/pdf/2503.04416)  

**Abstract**: The DreamerV3 algorithm recently obtained remarkable performance across diverse environment domains by learning an accurate world model based on Recurrent Neural Networks (RNNs). Following the success of model-based reinforcement learning algorithms and the rapid adoption of the Transformer architecture for its superior training efficiency and favorable scaling properties, recent works such as STORM have proposed replacing RNN-based world models with Transformer-based world models using masked self-attention. However, despite the improved training efficiency of these methods, their impact on performance remains limited compared to the Dreamer algorithm, struggling to learn competitive Transformer-based world models. In this work, we show that the next state prediction objective adopted in previous approaches is insufficient to fully exploit the representation capabilities of Transformers. We propose to extend world model predictions to longer time horizons by introducing TWISTER (Transformer-based World model wIth contraSTivE Representations), a world model using action-conditioned Contrastive Predictive Coding to learn high-level temporal feature representations and improve the agent performance. TWISTER achieves a human-normalized mean score of 162% on the Atari 100k benchmark, setting a new record among state-of-the-art methods that do not employ look-ahead search. 

**Abstract (ZH)**: 基于Transformer的目标引导表示的梦回算法V3扩展版：通过延长时间_horizon实现更高性能 

---
# Training-Free Graph Filtering via Multimodal Feature Refinement for Extremely Fast Multimodal Recommendation 

**Title (ZH)**: 无需训练的图过滤方法通过多模态特征精炼实现极端快速的多模态推荐 

**Authors**: Yu-Seung Roh, Joo-Young Kim, Jin-Duk Park, Won-Yong Shin  

**Link**: [PDF](https://arxiv.org/pdf/2503.04406)  

**Abstract**: Multimodal recommender systems improve the performance of canonical recommender systems with no item features by utilizing diverse content types such as text, images, and videos, while alleviating inherent sparsity of user-item interactions and accelerating user engagement. However, current neural network-based models often incur significant computational overhead due to the complex training process required to learn and integrate information from multiple modalities. To overcome this limitation, we propose MultiModal-Graph Filtering (MM-GF), a training-free method based on the notion of graph filtering (GF) for efficient and accurate multimodal recommendations. Specifically, MM-GF first constructs multiple similarity graphs through nontrivial multimodal feature refinement such as robust scaling and vector shifting by addressing the heterogeneous characteristics across modalities. Then, MM-GF optimally fuses multimodal information using linear low-pass filters across different modalities. Extensive experiments on real-world benchmark datasets demonstrate that MM-GF not only improves recommendation accuracy by up to 13.35% compared to the best competitor but also dramatically reduces computational costs by achieving the runtime of less than 10 seconds. 

**Abstract (ZH)**: 多模态推荐系统通过利用文本、图像和视频等多样化内容类型，改进了仅有物品特征的 canonical 推荐系统，同时缓解了用户-物品交互的固有稀疏性并加速了用户参与。然而，当前基于神经网络的模型往往由于需要学习和整合多种模态的信息而产生显著的计算开销。为克服这一限制，我们提出了一种基于图过滤（GF）概念的无训练方法——MultiModal-Graph Filtering（MM-GF），以实现高效准确的多模态推荐。具体来说，MM-GF 首先通过处理模态间异构特性来构建多个相似性图，例如鲁棒缩放和矢量平移等非平凡多模态特征 refinement。然后，MM-GF 使用跨不同模态的线性低通滤波器来最优融合多模态信息。在实际基准数据集上的广泛实验表明，MM-GF 不仅将推荐精度提高了高达 13.35%，而且通过实现小于 10 秒的运行时间显著降低了计算成本。 

---
# Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling 

**Title (ZH)**: 投机MoE：基于投机令牌和专家预调度的通信高效并行MoE推理 

**Authors**: Yan Li, Pengfei Zheng, Shuang Chen, Zewei Xu, Yunfei Du, Zhengang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04398)  

**Abstract**: MoE (Mixture of Experts) prevails as a neural architecture that can scale modern transformer-based LLMs (Large Language Models) to unprecedented scales. Nevertheless, large MoEs' great demands of computing power, memory capacity and memory bandwidth make scalable serving a fundamental challenge and efficient parallel inference has become a requisite to attain adequate throughput under latency constraints. DeepSpeed-MoE, one state-of-the-art MoE inference framework, adopts a 3D-parallel paradigm including EP (Expert Parallelism), TP (Tensor Parallel) and DP (Data Parallelism). However, our analysis shows DeepSpeed-MoE's inference efficiency is largely bottlenecked by EP, which is implemented with costly all-to-all collectives to route token activation. Our work aims to boost DeepSpeed-MoE by strategically reducing EP's communication overhead with a technique named Speculative MoE. Speculative MoE has two speculative parallelization schemes, speculative token shuffling and speculative expert grouping, which predict outstanding tokens' expert routing paths and pre-schedule tokens and experts across devices to losslessly trim EP's communication volume. Besides DeepSpeed-MoE, we also build Speculative MoE into a prevailing MoE inference engine SGLang. Experiments show Speculative MoE can significantly boost state-of-the-art MoE inference frameworks on fast homogeneous and slow heterogeneous interconnects. 

**Abstract (ZH)**: MoE（专家混合）作为能够将现代基于Transformer的大语言模型扩展到前所未有的玩家神经架构，继续占据主导地位。然而，大型MoE对计算能力、内存容量和内存带宽的庞大需求使得可扩展的服务成为一个基本挑战，高效的并行推断已成为在延迟约束下实现适当吞吐量的必要条件。DeepSpeed-MoE，一种最先进的MoE推断框架，采用包括EP（专家并行）、TP（张量并行）和DP（数据并行）的3D并行 paradigm。然而，我们的分析显示DeepSpeed-MoE的推断效率受到EP的严重影响，EP通过昂贵的全对全集合通信来路由token激活。我们通过一种名为Speculative MoE的技术旨在提升DeepSpeed-MoE，通过策略性减少EP的通信开销。Speculative MoE有两种推测性并行方案：推测性token混排和推测性专家分组，这些方案预测出色token的专家路由路径，并预调度token和专家跨设备以无损地削减EP的通信量。除了DeepSpeed-MoE之外，我们还将Speculative MoE整合进一种流行的MoE推断引擎SGLang。实验表明，Speculative MoE能够在快速同构和慢速异构互联环境下显著提升最先进的MoE推断框架的性能。 

---
# Dedicated Feedback and Edit Models Empower Inference-Time Scaling for Open-Ended General-Domain Tasks 

**Title (ZH)**: 专用于反馈和编辑的模型赋能开放域任务推理时的扩展能力 

**Authors**: Zhilin Wang, Jiaqi Zeng, Olivier Delalleau, Daniel Egert, Ellie Evans, Hoo-Chang Shin, Felipe Soares, Yi Dong, Oleksii Kuchaiev  

**Link**: [PDF](https://arxiv.org/pdf/2503.04378)  

**Abstract**: Inference-Time Scaling has been critical to the success of recent models such as OpenAI o1 and DeepSeek R1. However, many techniques used to train models for inference-time scaling require tasks to have answers that can be verified, limiting their application to domains such as math, coding and logical reasoning. We take inspiration from how humans make first attempts, ask for detailed feedback from others and make improvements based on such feedback across a wide spectrum of open-ended endeavors. To this end, we collect data for and train dedicated Feedback and Edit Models that are capable of performing inference-time scaling for open-ended general-domain tasks. In our setup, one model generates an initial response, which are given feedback by a second model, that are then used by a third model to edit the response. We show that performance on Arena Hard, a benchmark strongly predictive of Chatbot Arena Elo can be boosted by scaling the number of initial response drafts, effective feedback and edited responses. When scaled optimally, our setup based on 70B models from the Llama 3 family can reach SoTA performance on Arena Hard at 92.7 as of 5 Mar 2025, surpassing OpenAI o1-preview-2024-09-12 with 90.4 and DeepSeek R1 with 92.3. 

**Abstract (ZH)**: 推理时缩放对于最近的模型如OpenAI o1和DeepSeek R1的成功至关重要。然而，许多用于训练适用于推理时缩放模型的技术需要任务具有可验证的答案，限制了它们在数学、编程和逻辑推理等领域的应用。我们从人类如何进行首次尝试、寻求他人详细反馈并基于此类反馈改进工作的方式中汲取灵感，应用于各种开放性任务中。为此，我们收集数据并训练专门的反馈和编辑模型，这些模型能够处理开放性通用领域任务的推理时缩放。在我们的设置中，一个模型生成初始响应，该响应由第二个模型提供反馈，然后第三个模型基于此类反馈编辑响应。我们展示了通过增加初始响应草稿数量、有效的反馈和编辑响应，Arena Hard基准上的表现可以显著提升，该基准强烈预测着聊天机器人竞技场的Elo排名。基于Llama 3家族70B参数模型的设置，在2025年3月5日达到了92.7的SOTA性能，超过了2024年9月12日的OpenAI o1-preview-2024-09-12的90.4和DeepSeek R1的92.3。 

---
# Causally Reliable Concept Bottleneck Models 

**Title (ZH)**: 因果可靠概念瓶颈模型 

**Authors**: Giovanni De Felice, Arianna Casanova Flores, Francesco De Santis, Silvia Santini, Johannes Schneider, Pietro Barbiero, Alberto Termine  

**Link**: [PDF](https://arxiv.org/pdf/2503.04363)  

**Abstract**: Concept-based models are an emerging paradigm in deep learning that constrains the inference process to operate through human-interpretable concepts, facilitating explainability and human interaction. However, these architectures, on par with popular opaque neural models, fail to account for the true causal mechanisms underlying the target phenomena represented in the data. This hampers their ability to support causal reasoning tasks, limits out-of-distribution generalization, and hinders the implementation of fairness constraints. To overcome these issues, we propose \emph{Causally reliable Concept Bottleneck Models} (C$^2$BMs), a class of concept-based architectures that enforce reasoning through a bottleneck of concepts structured according to a model of the real-world causal mechanisms. We also introduce a pipeline to automatically learn this structure from observational data and \emph{unstructured} background knowledge (e.g., scientific literature). Experimental evidence suggest that C$^2$BM are more interpretable, causally reliable, and improve responsiveness to interventions w.r.t. standard opaque and concept-based models, while maintaining their accuracy. 

**Abstract (ZH)**: 基于概念的因果可靠瓶颈模型（C$^2$BMs）：一种根据现实世界因果机制结构化概念的架构 

---
# A Generalist Cross-Domain Molecular Learning Framework for Structure-Based Drug Discovery 

**Title (ZH)**: 基于结构的药物发现中通用型跨域分子学习框架 

**Authors**: Yiheng Zhu, Mingyang Li, Junlong Liu, Kun Fu, Jiansheng Wu, Qiuyi Li, Mingze Yin, Jieping Ye, Jian Wu, Zheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04362)  

**Abstract**: Structure-based drug discovery (SBDD) is a systematic scientific process that develops new drugs by leveraging the detailed physical structure of the target protein. Recent advancements in pre-trained models for biomolecules have demonstrated remarkable success across various biochemical applications, including drug discovery and protein engineering. However, in most approaches, the pre-trained models primarily focus on the characteristics of either small molecules or proteins, without delving into their binding interactions which are essential cross-domain relationships pivotal to SBDD. To fill this gap, we propose a general-purpose foundation model named BIT (an abbreviation for Biomolecular Interaction Transformer), which is capable of encoding a range of biochemical entities, including small molecules, proteins, and protein-ligand complexes, as well as various data formats, encompassing both 2D and 3D structures. Specifically, we introduce Mixture-of-Domain-Experts (MoDE) to handle the biomolecules from diverse biochemical domains and Mixture-of-Structure-Experts (MoSE) to capture positional dependencies in the molecular structures. The proposed mixture-of-experts approach enables BIT to achieve both deep fusion and domain-specific encoding, effectively capturing fine-grained molecular interactions within protein-ligand complexes. Then, we perform cross-domain pre-training on the shared Transformer backbone via several unified self-supervised denoising tasks. Experimental results on various benchmarks demonstrate that BIT achieves exceptional performance in downstream tasks, including binding affinity prediction, structure-based virtual screening, and molecular property prediction. 

**Abstract (ZH)**: 基于结构的药物发现（SBDD）是一种通过利用目标蛋白的详细物理结构来开发新药物的系统科学过程。预训练模型在生物分子领域的最新进展已经在包括药物发现和蛋白质工程在内的各种生物化学应用中取得了显著的成功。然而，在大多数方法中，预训练模型主要关注小分子或蛋白质的特点，而忽视了对它们之间相互作用的研究，这些相互作用是SBDD中至关重要的跨域关系。为弥补这一不足，我们提出了一种通用的基础模型Bit（Biomolecular Interaction Transformer的缩写），能够编码包括小分子、蛋白质以及蛋白质-配体复合物在内的多种生物化学实体，并能够处理包括2D和3D结构在内的多种数据格式。具体而言，我们引入了领域专家混合（MoDE）来处理来自不同生物化学领域的生物分子，并引入了结构专家混合（MoSE）来捕获分子结构中的位置依赖性。所提出专家混合的方法使Bit能够实现深层次的融合和领域特定的编码，有效地捕捉蛋白质-配体复合物中的细微分子相互作用。然后，我们通过多个统一的自监督去噪任务在共享的Transformer主干上进行跨域预训练。在各种基准测试上的实验结果表明，Bit在下游任务，包括结合亲和力预测、结构导向的虚拟筛选和分子性质预测中表现出色。 

---
# scDD: Latent Codes Based scRNA-seq Dataset Distillation with Foundation Model Knowledge 

**Title (ZH)**: scDD：基于潜伏代码的单细胞RNA-seq数据集提炼，融合基础模型知识 

**Authors**: Zhen Yu, Jianan Han, Yang Liu, Qingchao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.04357)  

**Abstract**: Single-cell RNA sequencing (scRNA-seq) technology has profiled hundreds of millions of human cells across organs, diseases, development and perturbations to date. However, the high-dimensional sparsity, batch effect noise, category imbalance, and ever-increasing data scale of the original sequencing data pose significant challenges for multi-center knowledge transfer, data fusion, and cross-validation between scRNA-seq datasets. To address these barriers, (1) we first propose a latent codes-based scRNA-seq dataset distillation framework named scDD, which transfers and distills foundation model knowledge and original dataset information into a compact latent space and generates synthetic scRNA-seq dataset by a generator to replace the original dataset. Then, (2) we propose a single-step conditional diffusion generator named SCDG, which perform single-step gradient back-propagation to help scDD optimize distillation quality and avoid gradient decay caused by multi-step back-propagation. Meanwhile, SCDG ensures the scRNA-seq data characteristics and inter-class discriminability of the synthetic dataset through flexible conditional control and generation quality assurance. Finally, we propose a comprehensive benchmark to evaluate the performance of scRNA-seq dataset distillation in different data analysis tasks. It is validated that our proposed method can achieve 7.61% absolute and 15.70% relative improvement over previous state-of-the-art methods on average task. 

**Abstract (ZH)**: 单细胞RNA测序数据集蒸馏框架scDD及单步条件扩散生成器SCDG的研究 

---
# Talking Back -- human input and explanations to interactive AI systems 

**Title (ZH)**: 与AI对话：人类对交互式AI系统的输入与解释 

**Authors**: Alan Dix, Tommaso Turchi, Ben Wilson, Anna Monreale, Matt Roach  

**Link**: [PDF](https://arxiv.org/pdf/2503.04343)  

**Abstract**: While XAI focuses on providing AI explanations to humans, can the reverse - humans explaining their judgments to AI - foster richer, synergistic human-AI systems? This paper explores various forms of human inputs to AI and examines how human explanations can guide machine learning models toward automated judgments and explanations that align more closely with human concepts. 

**Abstract (ZH)**: 人类向AI解释判断能否促进更加丰富和协同的人机系统：探索人类输入的各种形式及其引导机器学习模型生成与人类概念更加一致的自动化判断和解释的能力 

---
# Solving Word-Sense Disambiguation and Word-Sense Induction with Dictionary Examples 

**Title (ZH)**: 利用词典例证解决词义消歧和词义归纳问题 

**Authors**: Tadej Škvorc, Marko Robnik-Šikonja  

**Link**: [PDF](https://arxiv.org/pdf/2503.04328)  

**Abstract**: Many less-resourced languages struggle with a lack of large, task-specific datasets that are required for solving relevant tasks with modern transformer-based large language models (LLMs). On the other hand, many linguistic resources, such as dictionaries, are rarely used in this context despite their large information contents. We show how LLMs can be used to extend existing language resources in less-resourced languages for two important tasks: word-sense disambiguation (WSD) and word-sense induction (WSI). We approach the two tasks through the related but much more accessible word-in-context (WiC) task where, given a pair of sentences and a target word, a classification model is tasked with predicting whether the sense of a given word differs between sentences. We demonstrate that a well-trained model for this task can distinguish between different word senses and can be adapted to solve the WSD and WSI tasks. The advantage of using the WiC task, instead of directly predicting senses, is that the WiC task does not need pre-constructed sense inventories with a sufficient number of examples for each sense, which are rarely available in less-resourced languages. We show that sentence pairs for the WiC task can be successfully generated from dictionary examples using LLMs. The resulting prediction models outperform existing models on WiC, WSD, and WSI tasks. We demonstrate our methodology on the Slovene language, where a monolingual dictionary is available, but word-sense resources are tiny. 

**Abstract (ZH)**: 少资源语言中大型语言模型在词汇义消歧和词汇义归纳任务中的资源扩展研究 

---
# Provable Robust Overfitting Mitigation in Wasserstein Distributionally Robust Optimization 

**Title (ZH)**: 可验证鲁棒过拟合缓解在 Wasserstein 分布ally鲁棒优化中的应用 

**Authors**: Shuang Liu, Yihan Wang, Yifan Zhu, Yibo Miao, Xiao-Shan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04315)  

**Abstract**: Wasserstein distributionally robust optimization (WDRO) optimizes against worst-case distributional shifts within a specified uncertainty set, leading to enhanced generalization on unseen adversarial examples, compared to standard adversarial training which focuses on pointwise adversarial perturbations. However, WDRO still suffers fundamentally from the robust overfitting problem, as it does not consider statistical error. We address this gap by proposing a novel robust optimization framework under a new uncertainty set for adversarial noise via Wasserstein distance and statistical error via Kullback-Leibler divergence, called the Statistically Robust WDRO. We establish a robust generalization bound for the new optimization framework, implying that out-of-distribution adversarial performance is at least as good as the statistically robust training loss with high probability. Furthermore, we derive conditions under which Stackelberg and Nash equilibria exist between the learner and the adversary, giving an optimal robust model in certain sense. Finally, through extensive experiments, we demonstrate that our method significantly mitigates robust overfitting and enhances robustness within the framework of WDRO. 

**Abstract (ZH)**: 基于Wasserstein距离和统计误差的统计鲁棒Wasserstein分布鲁棒优化 

---
# Malware Detection at the Edge with Lightweight LLMs: A Performance Evaluation 

**Title (ZH)**: 边缘处基于轻量级LLM的恶意软件检测：性能评估 

**Authors**: Christian Rondanini, Barbara Carminati, Elena Ferrari, Antonio Gaudiano, Ashish Kundu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04302)  

**Abstract**: The rapid evolution of malware attacks calls for the development of innovative detection methods, especially in resource-constrained edge computing. Traditional detection techniques struggle to keep up with modern malware's sophistication and adaptability, prompting a shift towards advanced methodologies like those leveraging Large Language Models (LLMs) for enhanced malware detection. However, deploying LLMs for malware detection directly at edge devices raises several challenges, including ensuring accuracy in constrained environments and addressing edge devices' energy and computational limits. To tackle these challenges, this paper proposes an architecture leveraging lightweight LLMs' strengths while addressing limitations like reduced accuracy and insufficient computational power. To evaluate the effectiveness of the proposed lightweight LLM-based approach for edge computing, we perform an extensive experimental evaluation using several state-of-the-art lightweight LLMs. We test them with several publicly available datasets specifically designed for edge and IoT scenarios and different edge nodes with varying computational power and characteristics. 

**Abstract (ZH)**: 快速演变的恶意软件攻击促使开发创新检测方法，特别是在资源受限的边缘计算环境中。传统的检测技术难以跟上现代恶意软件的高度复杂性和适应性，推动了向高级方法的转变，例如利用大型语言模型（LLMs）来增强恶意软件检测。然而，直接在边缘设备上部署LLMs用于恶意软件检测也面临着诸多挑战，包括在受限环境中确保准确性以及解决边缘设备的能源和计算能力限制。为了应对这些挑战，本文提出了一种架构，利用轻量级LLMs的优势并解决了如准确性降低和计算能力不足等问题。为了评估基于轻量级LLMs的边缘计算检测方法的有效性，我们使用多种最先进的轻量级LLMs进行了广泛的实验评估。我们使用专门为边缘和物联网场景设计的多个公开数据集以及不同计算能力和特性的边缘节点进行测试。 

---
# How Do Hackathons Foster Creativity? Towards AI Collaborative Evaluation of Creativity at Scale 

**Title (ZH)**: 黑客松如何培养创造力？面向大规模AI协作评估创造力 

**Authors**: Jeanette Falk, Yiyi Chen, Janet Rafner, Mike Zhang, Johannes Bjerva, Alexander Nolte  

**Link**: [PDF](https://arxiv.org/pdf/2503.04290)  

**Abstract**: Hackathons have become popular collaborative events for accelerating the development of creative ideas and prototypes. There are several case studies showcasing creative outcomes across domains such as industry, education, and research. However, there are no large-scale studies on creativity in hackathons which can advance theory on how hackathon formats lead to creative outcomes. We conducted a computational analysis of 193,353 hackathon projects. By operationalizing creativity through usefulness and novelty, we refined our dataset to 10,363 projects, allowing us to analyze how participant characteristics, collaboration patterns, and hackathon setups influence the development of creative projects. The contribution of our paper is twofold: We identified means for organizers to foster creativity in hackathons. We also explore the use of large language models (LLMs) to augment the evaluation of creative outcomes and discuss challenges and opportunities of doing this, which has implications for creativity research at large. 

**Abstract (ZH)**: hackathons在促进创意成果方面的协作事件已成为流行的发展平台：对193,353个hackathon项目的计算分析揭示了参与者特征、协作模式和hackathon设置如何影响创意项目的发展，以及大型语言模型在评估创意成果中的应用及其挑战与机遇。 

---
# Explainable AI in Time-Sensitive Scenarios: Prefetched Offline Explanation Model 

**Title (ZH)**: 时间敏感场景中的可解释AI：预取的离线解释模型 

**Authors**: Fabio Michele Russo, Carlo Metta, Anna Monreale, Salvatore Rinzivillo, Fabio Pinelli  

**Link**: [PDF](https://arxiv.org/pdf/2503.04283)  

**Abstract**: As predictive machine learning models become increasingly adopted and advanced, their role has evolved from merely predicting outcomes to actively shaping them. This evolution has underscored the importance of Trustworthy AI, highlighting the necessity to extend our focus beyond mere accuracy and toward a comprehensive understanding of these models' behaviors within the specific contexts of their applications. To further progress in explainability, we introduce Poem, Prefetched Offline Explanation Model, a model-agnostic, local explainability algorithm for image data. The algorithm generates exemplars, counterexemplars and saliency maps to provide quick and effective explanations suitable for time-sensitive scenarios. Leveraging an existing local algorithm, \poem{} infers factual and counterfactual rules from data to create illustrative examples and opposite scenarios with an enhanced stability by design. A novel mechanism then matches incoming test points with an explanation base and produces diverse exemplars, informative saliency maps and believable counterexemplars. Experimental results indicate that Poem outperforms its predecessor Abele in speed and ability to generate more nuanced and varied exemplars alongside more insightful saliency maps and valuable counterexemplars. 

**Abstract (ZH)**: 随着预测型机器学习模型的日益采用和不断进步，其角色已从 merely 预测结果转变为积极塑造结果。这一演变强调了可信赖人工智能的重要性，凸显了我们需要将关注点从单纯的准确性扩展到对这些模型在其应用场景中的行为进行全面理解的必要性。为进一步推进可解释性，我们介绍了 Poem（预取本地解释模型），这是一种适用于图像数据的模型无关的局部解释算法。该算法生成示例、反例和显著性图，提供快速有效的解释，适用于时间敏感的场景。Poem 利用现有的局部算法从数据中推断事实和反事实规则，通过设计增强稳定性，创建具有说明性的示例和相反场景。一种新型机制将传入的测试点与解释基进行匹配，生成多样化的示例、信息丰富的显著性图和可信的反例。实验结果显示，Poem 在速度上优于其 predecessors Abele，并能够生成更细致多样且更具洞察力的显著性图和更有价值的反例。 

---
# Towards Autonomous Reinforcement Learning for Real-World Robotic Manipulation with Large Language Models 

**Title (ZH)**: 面向大型语言模型的现实世界机器人 manipulation 的自主强化学习方法 

**Authors**: Niccolò Turcato, Matteo Iovino, Aris Synodinos, Alberto Dalla Libera, Ruggero Carli, Pietro Falco  

**Link**: [PDF](https://arxiv.org/pdf/2503.04280)  

**Abstract**: Recent advancements in Large Language Models (LLMs) and Visual Language Models (VLMs) have significantly impacted robotics, enabling high-level semantic motion planning applications. Reinforcement Learning (RL), a complementary paradigm, enables agents to autonomously optimize complex behaviors through interaction and reward signals. However, designing effective reward functions for RL remains challenging, especially in real-world tasks where sparse rewards are insufficient and dense rewards require elaborate design. In this work, we propose Autonomous Reinforcement learning for Complex HumanInformed Environments (ARCHIE), an unsupervised pipeline leveraging GPT-4, a pre-trained LLM, to generate reward functions directly from natural language task descriptions. The rewards are used to train RL agents in simulated environments, where we formalize the reward generation process to enhance feasibility. Additionally, GPT-4 automates the coding of task success criteria, creating a fully automated, one-shot procedure for translating human-readable text into deployable robot skills. Our approach is validated through extensive simulated experiments on single-arm and bi-manual manipulation tasks using an ABB YuMi collaborative robot, highlighting its practicality and effectiveness. Tasks are demonstrated on the real robot setup. 

**Abstract (ZH)**: Recent advancements in大规模语言模型（LLMs）和视觉语言模型（VLMs）极大地影响了机器人技术，使得高级语义运动规划应用成为可能。强化学习（RL）作为一种补充范式，允许智能体通过互动和奖励信号自主优化复杂行为。然而，在实际任务中，设计有效的奖励函数仍然具有挑战性，尤其是当稀疏奖励不足且密集奖励需要复杂设计时。在这项工作中，我们提出了自主强化学习以复杂人类导向环境为背景（ARCHIE），这是一种无监督的流水线，利用预训练的大规模语言模型GPT-4直接从自然语言任务描述生成奖励函数。生成的奖励用于在模拟环境中训练RL智能体，并我们形式化了奖励生成过程以提高可行性。此外，GPT-4自动化了任务成功标准的编码，创造了将人类可读文本自动转换为可部署机器人技能的完全自动化、一次性流程。我们的方法通过在使用ABB YuMi协作机器人的单臂和双臂操作任务上的大量模拟实验得到了验证，突显了其实用性和有效性。任务在实际机器人设置中得到了演示。 

---
# Prompt Programming: A Platform for Dialogue-based Computational Problem Solving with Generative AI Models 

**Title (ZH)**: Prompt编程：一种基于对话的生成型AI模型计算问题求解平台 

**Authors**: Victor-Alexandru Pădurean, Paul Denny, Alkis Gotovos, Adish Singla  

**Link**: [PDF](https://arxiv.org/pdf/2503.04267)  

**Abstract**: Computing students increasingly rely on generative AI tools for programming assistance, often without formal instruction or guidance. This highlights a need to teach students how to effectively interact with AI models, particularly through natural language prompts, to generate and critically evaluate code for solving computational tasks. To address this, we developed a novel platform for prompt programming that enables authentic dialogue-based interactions, supports problems involving multiple interdependent functions, and offers on-request execution of generated code. Data analysis from over 900 students in an introductory programming course revealed high engagement, with the majority of prompts occurring within multi-turn dialogues. Problems with multiple interdependent functions encouraged iterative refinement, with progression graphs highlighting several common strategies. Students were highly selective about the code they chose to test, suggesting that on-request execution of generated code promoted critical thinking. Given the growing importance of learning dialogue-based programming with AI, we provide this tool as a publicly accessible resource, accompanied by a corpus of programming problems for educational use. 

**Abstract (ZH)**: 计算专业学生日益依赖生成式AI工具进行编程辅助，往往缺乏正式的指导或培训。这突显出教授学生如何有效与AI模型互动，尤其是在通过自然语言提示生成和批判性评估代码以解决计算任务方面，的需求。为应对这一挑战，我们开发了一个新颖的提示编程平台，该平台支持真实的基于对话的交互，能够处理涉及多个相互依赖函数的问题，并允许用户请求执行生成的代码。来自超过900名学生的初步编程课程的数据分析显示了高度的参与度，大多数提示发生在多轮对话中。涉及多个相互依赖函数的问题促进了迭代精炼，进度图表突显了多种常见策略。学生们对测试的代码选择性很强，表明请求执行生成的代码促进了批判性思维。鉴于学习基于对话的编程与AI日益重要的地位，我们提供此工具作为公开可访问的资源，并附带一个编程问题语料库用于教育目的。 

---
# TAIL: Text-Audio Incremental Learning 

**Title (ZH)**: TAIL: 文本-音频增量学习 

**Authors**: Yingfei Sun, Xu Gu, Wei Ji, Hanbin Zhao, Hao Fei, Yifang Yin, Roger Zimmermann  

**Link**: [PDF](https://arxiv.org/pdf/2503.04258)  

**Abstract**: Many studies combine text and audio to capture multi-modal information but they overlook the model's generalization ability on new datasets. Introducing new datasets may affect the feature space of the original dataset, leading to catastrophic forgetting. Meanwhile, large model parameters can significantly impact training performance. To address these limitations, we introduce a novel task called Text-Audio Incremental Learning (TAIL) task for text-audio retrieval, and propose a new method, PTAT, Prompt Tuning for Audio-Text incremental learning. This method utilizes prompt tuning to optimize the model parameters while incorporating an audio-text similarity and feature distillation module to effectively mitigate catastrophic forgetting. We benchmark our method and previous incremental learning methods on AudioCaps, Clotho, BBC Sound Effects and Audioset datasets, and our method outperforms previous methods significantly, particularly demonstrating stronger resistance to forgetting on older datasets. Compared to the full-parameters Finetune (Sequential) method, our model only requires 2.42\% of its parameters, achieving 4.46\% higher performance. 

**Abstract (ZH)**: 一种用于文本-音频检索的文本-音频增量学习任务及其方法：Prompt Tuning for Audio-Text Incremental Learning (PTAT) 

---
# How to Move Your Dragon: Text-to-Motion Synthesis for Large-Vocabulary Objects 

**Title (ZH)**: 如何移动你的龙：大词汇量对象的文本到运动合成 

**Authors**: Wonkwang Lee, Jongwon Jeong, Taehong Moon, Hyeon-Jong Kim, Jaehyeon Kim, Gunhee Kim, Byeong-Uk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.04257)  

**Abstract**: Motion synthesis for diverse object categories holds great potential for 3D content creation but remains underexplored due to two key challenges: (1) the lack of comprehensive motion datasets that include a wide range of high-quality motions and annotations, and (2) the absence of methods capable of handling heterogeneous skeletal templates from diverse objects. To address these challenges, we contribute the following: First, we augment the Truebones Zoo dataset, a high-quality animal motion dataset covering over 70 species, by annotating it with detailed text descriptions, making it suitable for text-based motion synthesis. Second, we introduce rig augmentation techniques that generate diverse motion data while preserving consistent dynamics, enabling models to adapt to various skeletal configurations. Finally, we redesign existing motion diffusion models to dynamically adapt to arbitrary skeletal templates, enabling motion synthesis for a diverse range of objects with varying structures. Experiments show that our method learns to generate high-fidelity motions from textual descriptions for diverse and even unseen objects, setting a strong foundation for motion synthesis across diverse object categories and skeletal templates. Qualitative results are available on this link: this http URL 

**Abstract (ZH)**: 多样物体类别的运动合成在3D内容创作中具有巨大潜力，但由于两个关键挑战而鲜有探索：（1）缺乏包含广泛高质量运动和注释的综合运动数据集，（2）缺乏能够处理来自多样化物体的异构骨骼模板的方法。为应对这些挑战，我们做出了以下贡献：首先，我们通过添加详细文本描述来扩展Truebones Zoo数据集，这是一个涵盖超过70种物种的高质量动物运动数据集，使其适合基于文本的运动合成。其次，我们引入了 rig 增强技术，生成多样化运动数据同时保持一致的动力学，使模型能够适应各种骨骼配置。最后，我们重新设计现有的运动扩散模型，使其能够动态适应任意骨骼模板，从而实现具有不同结构的多样化物体的运动合成。实验表明，我们的方法能够从文本描述中生成高保真度的运动，适用于多样甚至未见过的物体，为多样物体类别和骨骼模板的运动合成奠定了坚实的基础。定性结果可在以下链接查看：this http URL 

---
# Knowledge Retention for Continual Model-Based Reinforcement Learning 

**Title (ZH)**: 基于模型的强化学习连续知识保留 

**Authors**: Yixiang Sun, Haotian Fu, Michael Littman, George Konidaris  

**Link**: [PDF](https://arxiv.org/pdf/2503.04256)  

**Abstract**: We propose DRAGO, a novel approach for continual model-based reinforcement learning aimed at improving the incremental development of world models across a sequence of tasks that differ in their reward functions but not the state space or dynamics. DRAGO comprises two key components: Synthetic Experience Rehearsal, which leverages generative models to create synthetic experiences from past tasks, allowing the agent to reinforce previously learned dynamics without storing data, and Regaining Memories Through Exploration, which introduces an intrinsic reward mechanism to guide the agent toward revisiting relevant states from prior tasks. Together, these components enable the agent to maintain a comprehensive and continually developing world model, facilitating more effective learning and adaptation across diverse environments. Empirical evaluations demonstrate that DRAGO is able to preserve knowledge across tasks, achieving superior performance in various continual learning scenarios. 

**Abstract (ZH)**: 我们提出DRAGO，一种针对连续模型为基础的强化学习的新方法，旨在通过一系列奖励函数不同但状态空间和动力学相同的任务，改进世界模型的增量开发。DRAGO包括两个关键组件：合成经验排练，利用生成模型从过去的任务中创建合成经验，使智能体可以在不存储数据的情况下强化之前学习的动力学；以及通过探索恢复记忆，通过引入内在奖励机制引导智能体重返以前任务中的相关状态。这两个组件共同使智能体能够维护一个全面且持续发展的世界模型，促进在各种环境中的更有效的学习和适应。实证评估表明，DRAGO能够在任务之间保持知识，并在各种连续学习场景中实现更好的性能。 

---
# How to Mitigate Overfitting in Weak-to-strong Generalization? 

**Title (ZH)**: 如何减轻从弱到强泛化的过拟合？ 

**Authors**: Junhao Shi, Qinyuan Cheng, Zhaoye Fei, Yining Zheng, Qipeng Guo, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04249)  

**Abstract**: Aligning powerful AI models on tasks that surpass human evaluation capabilities is the central problem of \textbf{superalignment}. To address this problem, weak-to-strong generalization aims to elicit the capabilities of strong models through weak supervisors and ensure that the behavior of strong models aligns with the intentions of weak supervisors without unsafe behaviors such as deception. Although weak-to-strong generalization exhibiting certain generalization capabilities, strong models exhibit significant overfitting in weak-to-strong generalization: Due to the strong fit ability of strong models, erroneous labels from weak supervisors may lead to overfitting in strong models. In addition, simply filtering out incorrect labels may lead to a degeneration in question quality, resulting in a weak generalization ability of strong models on hard questions. To mitigate overfitting in weak-to-strong generalization, we propose a two-stage framework that simultaneously improves the quality of supervision signals and the quality of input questions. Experimental results in three series of large language models and two mathematical benchmarks demonstrate that our framework significantly improves PGR compared to naive weak-to-strong generalization, even achieving up to 100\% PGR on some models. 

**Abstract (ZH)**: 超对齐中的弱到强泛化：一种双阶段框架以同时提高监督信号和输入问题的质量 

---
# One-Shot Clustering for Federated Learning 

**Title (ZH)**: 联邦学习中的一次聚类方法 

**Authors**: Maciej Krzysztof Zuziak, Roberto Pellungrini, Salvatore Rinzivillo  

**Link**: [PDF](https://arxiv.org/pdf/2503.04231)  

**Abstract**: Federated Learning (FL) is a widespread and well adopted paradigm of decentralized learning that allows training one model from multiple sources without the need to directly transfer data between participating clients. Since its inception in 2015, it has been divided into numerous sub-fields that deal with application-specific issues, be it data heterogeneity or resource allocation. One such sub-field, Clustered Federated Learning (CFL), is dealing with the problem of clustering the population of clients into separate cohorts to deliver personalized models. Although few remarkable works have been published in this domain, the problem is still largely unexplored, as its basic assumption and settings are slightly different from standard FL. In this work, we present One-Shot Clustered Federated Learning (OCFL), a clustering-agnostic algorithm that can automatically detect the earliest suitable moment for clustering. Our algorithm is based on the computation of cosine similarity between gradients of the clients and a temperature measure that detects when the federated model starts to converge. We empirically evaluate our methodology by testing various one-shot clustering algorithms for over thirty different tasks on three benchmark datasets. Our experiments showcase the good performance of our approach when used to perform CFL in an automated manner without the need to adjust hyperparameters. 

**Abstract (ZH)**: 联邦学习中的一键聚类联邦学习（One-Shot Clustered Federated Learning） 

---
# Quantum-Inspired Reinforcement Learning in the Presence of Epistemic Ambivalence 

**Title (ZH)**: 量子启发的在认识模态不确定性下的强化学习 

**Authors**: Alireza Habibi, Saeed Ghoorchian, Setareh Maghsudi  

**Link**: [PDF](https://arxiv.org/pdf/2503.04219)  

**Abstract**: The complexity of online decision-making under uncertainty stems from the requirement of finding a balance between exploiting known strategies and exploring new possibilities. Naturally, the uncertainty type plays a crucial role in developing decision-making strategies that manage complexity effectively. In this paper, we focus on a specific form of uncertainty known as epistemic ambivalence (EA), which emerges from conflicting pieces of evidence or contradictory experiences. It creates a delicate interplay between uncertainty and confidence, distinguishing it from epistemic uncertainty that typically diminishes with new information. Indeed, ambivalence can persist even after additional knowledge is acquired. To address this phenomenon, we propose a novel framework, called the epistemically ambivalent Markov decision process (EA-MDP), aiming to understand and control EA in decision-making processes. This framework incorporates the concept of a quantum state from the quantum mechanics formalism, and its core is to assess the probability and reward of every possible outcome. We calculate the reward function using quantum measurement techniques and prove the existence of an optimal policy and an optimal value function in the EA-MDP framework. We also propose the EA-epsilon-greedy Q-learning algorithm. To evaluate the impact of EA on decision-making and the expedience of our framework, we study two distinct experimental setups, namely the two-state problem and the lattice problem. Our results show that using our methods, the agent converges to the optimal policy in the presence of EA. 

**Abstract (ZH)**: 在线不确定性条件下的决策复杂性源于在利用已知策略和探索新可能性之间寻找平衡。自然地，不确定性类型在有效管理这种复杂性方面发挥着关键作用。本文重点关注一种特定形式的不确定性，即源于冲突性证据或矛盾性体验的认识模棱两可（EA）。这种模棱两可在不确定性和信心之间创造出一种微妙的互动，区别于通常随新信息增多而减弱的认识不确定性。事实上，即使获取了更多知识，模棱两可也可能持续存在。为应对这一现象，我们提出了一种新的框架，称为认识模棱两可马尔科夫决策过程（EA-MDP），旨在理解和控制决策过程中认识模棱两可。该框架借鉴了量子力学形式主义中的量子态概念，其核心是评估每种可能结果的概率和回报。我们使用量子测量技术计算回报函数，并证明在EA-MDP框架中存在最优策略和最优价值函数。我们还提出了EA-ε-贪婪Q学习算法。为了评估认识模棱两可对决策的影响以及我们框架的有效性，我们在两种不同的实验设置中进行了研究，即两态问题和格点问题。结果表明，使用我们的方法，在认识模棱两可的情况下，代理能够收敛到最优策略。 

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
# CrowdHMTware: A Cross-level Co-adaptation Middleware for Context-aware Mobile DL Deployment 

**Title (ZH)**: CrowdHMTware: 一种面向上下文感知移动DL部署的跨层次协同适应中间件 

**Authors**: Sicong Liu, Bin Guo, Shiyan Luo, Yuzhan Wang, Hao Luo, Cheng Fang, Yuan Xu, Ke Ma, Yao Li, Zhiwen Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04183)  

**Abstract**: There are many deep learning (DL) powered mobile and wearable applications today continuously and unobtrusively sensing the ambient surroundings to enhance all aspects of human this http URL enable robust and private mobile sensing, DL models are often deployed locally on resource-constrained mobile devices using techniques such as model compression or this http URL, existing methods, either front-end algorithm level (i.e. DL model compression/partitioning) or back-end scheduling level (i.e. operator/resource scheduling), cannot be locally online because they require offline retraining to ensure accuracy or rely on manually pre-defined strategies, struggle with dynamic this http URL primary challenge lies in feeding back runtime performance from the back-end level to the front-end level optimization decision. Moreover, the adaptive mobile DL model porting middleware with cross-level co-adaptation is less explored, particularly in mobile environments with diversity and dynamics. In response, we introduce CrowdHMTware, a dynamic context-adaptive DL model deployment middleware for heterogeneous mobile devices. It establishes an automated adaptation loop between cross-level functional components, i.e. elastic inference, scalable offloading, and model-adaptive engine, enhancing scalability and adaptability. Experiments with four typical tasks across 15 platforms and a real-world case study demonstrate that CrowdHMTware can effectively scale DL model, offloading, and engine actions across diverse platforms and tasks. It hides run-time system issues from developers, reducing the required developer expertise. 

**Abstract (ZH)**: CrowdHMTware：一种适应动态上下文的异构移动设备DL模型部署中间件 

---
# Towards Intelligent Transportation with Pedestrians and Vehicles In-the-Loop: A Surveillance Video-Assisted Federated Digital Twin Framework 

**Title (ZH)**: 基于监视视频辅助的车行人联动智能交通联合数字孪生框架 

**Authors**: Xiaolong Li, Jianhao Wei, Haidong Wang, Li Dong, Ruoyang Chen, Changyan Yi, Jun Cai, Dusit Niyato, Xuemin, Shen  

**Link**: [PDF](https://arxiv.org/pdf/2503.04170)  

**Abstract**: In intelligent transportation systems (ITSs), incorporating pedestrians and vehicles in-the-loop is crucial for developing realistic and safe traffic management solutions. However, there is falls short of simulating complex real-world ITS scenarios, primarily due to the lack of a digital twin implementation framework for characterizing interactions between pedestrians and vehicles at different locations in different traffic environments. In this article, we propose a surveillance video assisted federated digital twin (SV-FDT) framework to empower ITSs with pedestrians and vehicles in-the-loop. Specifically, SVFDT builds comprehensive pedestrian-vehicle interaction models by leveraging multi-source traffic surveillance videos. Its architecture consists of three layers: (i) the end layer, which collects traffic surveillance videos from multiple sources; (ii) the edge layer, responsible for semantic segmentation-based visual understanding, twin agent-based interaction modeling, and local digital twin system (LDTS) creation in local regions; and (iii) the cloud layer, which integrates LDTSs across different regions to construct a global DT model in realtime. We analyze key design requirements and challenges and present core guidelines for SVFDT's system implementation. A testbed evaluation demonstrates its effectiveness in optimizing traffic management. Comparisons with traditional terminal-server frameworks highlight SV-FDT's advantages in mirroring delays, recognition accuracy, and subjective evaluation. Finally, we identify some open challenges and discuss future research directions. 

**Abstract (ZH)**: 基于监控视频辅助联邦数字孪生的智能运输系统行人与车辆在环仿真框架 

---
# The Role of Visual Modality in Multimodal Mathematical Reasoning: Challenges and Insights 

**Title (ZH)**: 视觉模态在多模态数学推理中的作用：挑战与见解 

**Authors**: Yufang Liu, Yao Du, Tao Ji, Jianing Wang, Yang Liu, Yuanbin Wu, Aimin Zhou, Mengdi Zhang, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2503.04167)  

**Abstract**: Recent research has increasingly focused on multimodal mathematical reasoning, particularly emphasizing the creation of relevant datasets and benchmarks. Despite this, the role of visual information in reasoning has been underexplored. Our findings show that existing multimodal mathematical models minimally leverage visual information, and model performance remains largely unaffected by changes to or removal of images in the dataset. We attribute this to the dominance of textual information and answer options that inadvertently guide the model to correct answers. To improve evaluation methods, we introduce the HC-M3D dataset, specifically designed to require image reliance for problem-solving and to challenge models with similar, yet distinct, images that change the correct answer. In testing leading models, their failure to detect these subtle visual differences suggests limitations in current visual perception capabilities. Additionally, we observe that the common approach of improving general VQA capabilities by combining various types of image encoders does not contribute to math reasoning performance. This finding also presents a challenge to enhancing visual reliance during math reasoning. Our benchmark and code would be available at \href{this https URL}{this https URL\_modality\_role}. 

**Abstract (ZH)**: 最近的研究越来越多地关注多模态数学推理，特别强调相关数据集和基准的创建。然而，视觉信息在推理中的作用仍然被 недо探索。我们的研究结果表明，现有的多模态数学模型对视觉信息的利用 minimal，且 dataset 中图像的变化或移除对模型性能几乎没有影响。我们归因于文本信息和答案选项的主导地位，这些文本信息和选项无意中指导模型得出正确答案。为了改进评估方法，我们介绍了 HC-M3D 数据集，该数据集特别设计要求图像依赖性以解决问题，并用相似但独特的图像挑战模型，这些图像会改变正确答案。在测试顶级模型时，它们未能检测到这些微妙的视觉差异表明当前的视觉感知能力存在局限性。此外，我们还观察到，通过结合各种图像编码器来提高通用视觉问答 (VQA) 能力的做法并不能提升数学推理性能。这一发现也对在数学推理中增强视觉依赖性构成了挑战。我们的基准测试和代码可在 \href{this https URL}{this https URL\_modality\_role} 获取。 

---
# Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation 

**Title (ZH)**: 基于语义检索增强对比学习的序列推荐 

**Authors**: Ziqiang Cui, Yunpeng Weng, Xing Tang, Xiaokun Zhang, Dugang Liu, Shiwei Li, Peiyang Liu, Bowei He, Weihong Luo, Xiuqiang He, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.04162)  

**Abstract**: Sequential recommendation aims to model user preferences based on historical behavior sequences, which is crucial for various online platforms. Data sparsity remains a significant challenge in this area as most users have limited interactions and many items receive little attention. To mitigate this issue, contrastive learning has been widely adopted. By constructing positive sample pairs from the data itself and maximizing their agreement in the embedding space,it can leverage available data more effectively. Constructing reasonable positive sample pairs is crucial for the success of contrastive learning. However, current approaches struggle to generate reliable positive pairs as they either rely on representations learned from inherently sparse collaborative signals or use random perturbations which introduce significant uncertainty. To address these limitations, we propose a novel approach named Semantic Retrieval Augmented Contrastive Learning (SRA-CL), which leverages semantic information to improve the reliability of contrastive samples. SRA-CL comprises two main components: (1) Cross-Sequence Contrastive Learning via User Semantic Retrieval, which utilizes large language models (LLMs) to understand diverse user preferences and retrieve semantically similar users to form reliable positive samples through a learnable sample synthesis method; and (2) Intra-Sequence Contrastive Learning via Item Semantic Retrieval, which employs LLMs to comprehend items and retrieve similar items to perform semantic-based item substitution, thereby creating semantically consistent augmented views for contrastive learning. SRA-CL is plug-and-play and can be integrated into standard sequential recommendation models. Extensive experiments on four public datasets demonstrate the effectiveness and generalizability of the proposed approach. 

**Abstract (ZH)**: 基于序列的推荐旨在根据用户历史行为序列建模用户偏好，对于各种在线平台至关重要。数据稀疏性仍然是这一领域的一个重大挑战，因为大多数用户与系统的互动有限，许多项目受到的关注也很少。为缓解这一问题，对比学习已被广泛采用。通过从数据本身构建正样本对，并在嵌入空间中最大化它们的一致性，它可以更有效地利用可用数据。构建合理的正样本对对于对比学习的成功至关重要。然而，当前的方法难以生成可靠的数据对，因为它们要么依赖于从固有稀疏的合作信号中学习到的表示，要么使用随机扰动引入了大量不确定性。为解决这些局限性，我们提出了一种名为语义检索增强对比学习（SRA-CL）的新方法，该方法利用语义信息以提高对比样本的可靠性。SRA-CL 包含两个主要组成部分：1）基于用户语义检索的跨序列对比学习，该方法利用大型语言模型（LLMs）理解多样化的用户偏好，并通过可学习的样本合成方法检索语义相似的用户，以形成可靠的正样本对；2）基于物品语义检索的序列内对照学习，该方法利用大型语言模型理解物品并检索相似的物品，以执行基于语义的物品替代，从而创建语义上一致的增强视图用于对比学习。SRA-CL 是即插即用的，可以集成到标准的基于序列的推荐模型中。在四个公开数据集上的广泛实验表明，所提出的方法的有效性和普适性。 

---
# Unseen Fake News Detection Through Casual Debiasing 

**Title (ZH)**: 未见假新闻检测通过偶然去bias化 

**Authors**: Shuzhi Gong, Richard Sinnott, Jianzhong Qi, Cecile Paris  

**Link**: [PDF](https://arxiv.org/pdf/2503.04160)  

**Abstract**: The widespread dissemination of fake news on social media poses significant risks, necessitating timely and accurate detection. However, existing methods struggle with unseen news due to their reliance on training data from past events and domains, leaving the challenge of detecting novel fake news largely unresolved. To address this, we identify biases in training data tied to specific domains and propose a debiasing solution FNDCD. Originating from causal analysis, FNDCD employs a reweighting strategy based on classification confidence and propagation structure regularization to reduce the influence of domain-specific biases, enhancing the detection of unseen fake news. Experiments on real-world datasets with non-overlapping news domains demonstrate FNDCD's effectiveness in improving generalization across domains. 

**Abstract (ZH)**: 社交媒体上广泛传播的假新闻带来重大风险，需要及时准确地检测。然而，现有的方法由于依赖过去的事件和领域的训练数据，在检测前所未见的假新闻时遇到了挑战，使得检测新颖假新闻的问题仍未得到充分解决。为应对这一挑战，我们识别了与特定领域相关联的训练数据偏见，并提出了一种去偏方案FNDCD。基于因果分析，FNDCD采用基于分类置信度和传播结构正则化的重新加权策略，以降低领域特定偏见的影响，从而提高对前所未见假新闻的检测能力。实验表明，FNDCD在具有非重叠新闻领域的现实数据集上有效提高了跨领域的泛化能力。 

---
# CA-W3D: Leveraging Context-Aware Knowledge for Weakly Supervised Monocular 3D Detection 

**Title (ZH)**: CA-W3D: 利用上下文感知知识进行弱监督单目3D检测 

**Authors**: Chupeng Liu, Runkai Zhao, Weidong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2503.04154)  

**Abstract**: Weakly supervised monocular 3D detection, while less annotation-intensive, often struggles to capture the global context required for reliable 3D reasoning. Conventional label-efficient methods focus on object-centric features, neglecting contextual semantic relationships that are critical in complex scenes. In this work, we propose a Context-Aware Weak Supervision for Monocular 3D object detection, namely CA-W3D, to address this limitation in a two-stage training paradigm. Specifically, we first introduce a pre-training stage employing Region-wise Object Contrastive Matching (ROCM), which aligns regional object embeddings derived from a trainable monocular 3D encoder and a frozen open-vocabulary 2D visual grounding model. This alignment encourages the monocular encoder to discriminate scene-specific attributes and acquire richer contextual knowledge. In the second stage, we incorporate a pseudo-label training process with a Dual-to-One Distillation (D2OD) mechanism, which effectively transfers contextual priors into the monocular encoder while preserving spatial fidelity and maintaining computational efficiency during inference. Extensive experiments conducted on the public KITTI benchmark demonstrate the effectiveness of our approach, surpassing the SoTA method over all metrics, highlighting the importance of contextual-aware knowledge in weakly-supervised monocular 3D detection. 

**Abstract (ZH)**: 基于上下文感知的单目3D检测弱监督方法：CA-W3D 

---
# Robust Multi-View Learning via Representation Fusion of Sample-Level Attention and Alignment of Simulated Perturbation 

**Title (ZH)**: 基于样本级注意力表示融合与模拟扰动对齐的鲁棒多视图学习 

**Authors**: Jie Xu, Na Zhao, Gang Niu, Masashi Sugiyama, Xiaofeng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04151)  

**Abstract**: Recently, multi-view learning (MVL) has garnered significant attention due to its ability to fuse discriminative information from multiple views. However, real-world multi-view datasets are often heterogeneous and imperfect, which usually makes MVL methods designed for specific combinations of views lack application potential and limits their effectiveness. To address this issue, we propose a novel robust MVL method (namely RML) with simultaneous representation fusion and alignment. Specifically, we introduce a simple yet effective multi-view transformer fusion network where we transform heterogeneous multi-view data into homogeneous word embeddings, and then integrate multiple views by the sample-level attention mechanism to obtain a fused representation. Furthermore, we propose a simulated perturbation based multi-view contrastive learning framework that dynamically generates the noise and unusable perturbations for simulating imperfect data conditions. The simulated noisy and unusable data obtain two distinct fused representations, and we utilize contrastive learning to align them for learning discriminative and robust representations. Our RML is self-supervised and can also be applied for downstream tasks as a regularization. In experiments, we employ it in unsupervised multi-view clustering, noise-label classification, and as a plug-and-play module for cross-modal hashing retrieval. Extensive comparison experiments and ablation studies validate the effectiveness of RML. 

**Abstract (ZH)**: 最近，由于其能够融合多视角下的判别信息，多视图学习（MVL）受到了广泛关注。然而，现实中的多视图数据通常具有异质性和不完美性，这通常使得针对特定视图组合设计的MVL方法缺乏应用潜力并限制了其效果。为解决这一问题，我们提出了一种新颖的鲁棒多视图学习方法（即RML），并同时实现了表示融合和对齐。具体而言，我们引入了一种简单有效的多视图变压器融合网络，将异质的多视图数据转换为同构的词嵌入，并通过样本级注意力机制整合多个视图以获得融合表示。此外，我们提出了一个基于模拟扰动的多视图对比学习框架，可以动态生成噪声和不可用的扰动以模拟不完美的数据条件。模拟得到的噪声和不可用数据获得两种不同的融合表示，我们利用对比学习对齐它们以学习判别性和鲁棒性的表示。我们的RML是自监督的，并且也可以作为正则化应用于下游任务。在实验中，我们将它应用于无监督多视图聚类、带噪声标签的分类任务，以及跨模态哈希检索的插件模块。广泛的比较实验和消融研究验证了RML的有效性。 

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
# DM-Adapter: Domain-Aware Mixture-of-Adapters for Text-Based Person Retrieval 

**Title (ZH)**: DM-Adapter: 域 aware 混合适配器用于基于文本的人像检索 

**Authors**: Yating Liu, Zimo Liu, Xiangyuan Lan, Wenming Yang, Yaowei Li, Qingmin Liao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04144)  

**Abstract**: Text-based person retrieval (TPR) has gained significant attention as a fine-grained and challenging task that closely aligns with practical applications. Tailoring CLIP to person domain is now a emerging research topic due to the abundant knowledge of vision-language pretraining, but challenges still remain during fine-tuning: (i) Previous full-model fine-tuning in TPR is computationally expensive and prone to overfitting.(ii) Existing parameter-efficient transfer learning (PETL) for TPR lacks of fine-grained feature extraction. To address these issues, we propose Domain-Aware Mixture-of-Adapters (DM-Adapter), which unifies Mixture-of-Experts (MOE) and PETL to enhance fine-grained feature representations while maintaining efficiency. Specifically, Sparse Mixture-of-Adapters is designed in parallel to MLP layers in both vision and language branches, where different experts specialize in distinct aspects of person knowledge to handle features more finely. To promote the router to exploit domain information effectively and alleviate the routing imbalance, Domain-Aware Router is then developed by building a novel gating function and injecting learnable domain-aware prompts. Extensive experiments show that our DM-Adapter achieves state-of-the-art performance, outperforming previous methods by a significant margin. 

**Abstract (ZH)**: 基于文本的人检索（TPR）已成为一项精细且具有挑战性的任务，紧密契合实际应用需求， Tailored CLIP技术在人员领域正成为一项新兴的研究话题，但由于在细调过程中仍存在挑战：（i）TPR中的全模型细调计算成本高，容易过拟合。（ii）现有针对TPR的参数高效转移学习（PETL）缺乏细粒度特征提取。为解决这些问题，我们提出了域感知混合专家适应器（DM-Adapter），将其与PETL统一起来，在增强细粒度特征表示的同时保持效率。具体而言，稀疏混合专家适应器并行设计在视觉和语言分支的MLP层中，不同的专家专注于人员知识的不同方面，以更精细地处理特征。为了促进路由器有效利用领域信息并缓解路由失衡，我们开发了域感知路由器，通过构建新的门控函数并注入可学习的领域感知提示来实现。大量实验表明，我们的DM-Adapter取得了最先进的性能，显著优于先前的方法。 

---
# MTS: A Deep Reinforcement Learning Portfolio Management Framework with Time-Awareness and Short-Selling 

**Title (ZH)**: MTS：一种具有时间意识和卖空的深度 reinforcement 学习投资组合管理框架 

**Authors**: Fengchen Gu, Zhengyong Jiang, Ángel F. García-Fernández, Angelos Stefanidis, Jionglong Su, Huakang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.04143)  

**Abstract**: Portfolio management remains a crucial challenge in finance, with traditional methods often falling short in complex and volatile market environments. While deep reinforcement approaches have shown promise, they still face limitations in dynamic risk management, exploitation of temporal markets, and incorporation of complex trading strategies such as short-selling. These limitations can lead to suboptimal portfolio performance, increased vulnerability to market volatility, and missed opportunities in capturing potential returns from diverse market conditions. This paper introduces a Deep Reinforcement Learning Portfolio Management Framework with Time-Awareness and Short-Selling (MTS), offering a robust and adaptive strategy for sustainable investment performance. This framework utilizes a novel encoder-attention mechanism to address the limitations by incorporating temporal market characteristics, a parallel strategy for automated short-selling based on market trends, and risk management through innovative Incremental Conditional Value at Risk, enhancing adaptability and performance. Experimental validation on five diverse datasets from 2019 to 2023 demonstrates MTS's superiority over traditional algorithms and advanced machine learning techniques. MTS consistently achieves higher cumulative returns, Sharpe, Omega, and Sortino ratios, underscoring its effectiveness in balancing risk and return while adapting to market dynamics. MTS demonstrates an average relative increase of 30.67% in cumulative returns and 29.33% in Sharpe ratio compared to the next best-performing strategies across various datasets. 

**Abstract (ZH)**: 具有时间意识和卖空的深度强化学习投资组合管理框架（MTS） 

---
# Simple Self Organizing Map with Visual Transformer 

**Title (ZH)**: 简单自组织映射结合视觉变压器 

**Authors**: Alan Luo, Kaiwen Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.04121)  

**Abstract**: Vision Transformers (ViTs) have demonstrated exceptional performance in various vision tasks. However, they tend to underperform on smaller datasets due to their inherent lack of inductive biases. Current approaches address this limitation implicitly-often by pairing ViTs with pretext tasks or by distilling knowledge from convolutional neural networks (CNNs) to strengthen the prior. In contrast, Self-Organizing Maps (SOMs), a widely adopted self-supervised framework, are inherently structured to preserve topology and spatial organization, making them a promising candidate to directly address the limitations of ViTs in limited or small training datasets. Despite this potential, equipping SOMs with modern deep learning architectures remains largely unexplored. In this study, we conduct a novel exploration on how Vision Transformers (ViTs) and Self-Organizing Maps (SOMs) can empower each other, aiming to bridge this critical research gap. Our findings demonstrate that these architectures can synergistically enhance each other, leading to significantly improved performance in both unsupervised and supervised tasks. Code will be publicly available. 

**Abstract (ZH)**: 视觉变换器（ViTs）在各种视觉任务中已经展示了出色的表现。然而，它们在较小的数据集上往往会表现出色不足，这主要是由于其固有的归纳偏置缺乏。当前的方法通常通过将ViTs与先验知识增强的预设任务或从卷积神经网络（CNNs）中提取知识来隐式地解决这一限制。相比之下，自组织映射（SOMs）作为一种广泛采用的自监督框架，天然地保持拓扑和空间组织结构，这使其成为直接解决ViTs在有限或小型训练数据集上的限制的有前途的候选者。尽管如此，如何将现代深度学习架构与SOMs结合仍是一个未被充分探索的领域。在本研究中，我们进行了一种新颖的探索，研究视觉变换器（ViTs）和自组织映射（SOMs）如何相互赋能，旨在填补这一关键的研究空白。我们的研究表明，这些架构可以协同增强彼此，从而在无监督和监督任务中均显著提高性能。代码将公开发布。 

---
# Generalizability of Neural Networks Minimizing Empirical Risk Based on Expressive Ability 

**Title (ZH)**: 基于表征能力最小化经验风险的神经网络泛化性研究 

**Authors**: Lijia Yu, Yibo Miao, Yifan Zhu, Xiao-Shan Gao, Lijun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04111)  

**Abstract**: The primary objective of learning methods is generalization. Classic uniform generalization bounds, which rely on VC-dimension or Rademacher complexity, fail to explain the significant attribute that over-parameterized models in deep learning exhibit nice generalizability. On the other hand, algorithm-dependent generalization bounds, like stability bounds, often rely on strict assumptions. To establish generalizability under less stringent assumptions, this paper investigates the generalizability of neural networks that minimize or approximately minimize empirical risk. We establish a lower bound for population accuracy based on the expressiveness of these networks, which indicates that with an adequate large number of training samples and network sizes, these networks, including over-parameterized ones, can generalize effectively. Additionally, we provide a necessary condition for generalization, demonstrating that, for certain data distributions, the quantity of training data required to ensure generalization exceeds the network size needed to represent the corresponding data distribution. Finally, we provide theoretical insights into several phenomena in deep learning, including robust generalization, importance of over-parameterization, and effect of loss function on generalization. 

**Abstract (ZH)**: 学习方法的主要目标是泛化。经典的统一泛化界，如VC维或Rademacher复杂性，未能解释深度学习中过参数化模型表现出的良好泛化能力。另一方面，依赖于算法的泛化界，如稳定性界，通常依赖于严格的假设。为了在较不严格的假设下建立泛化性，本文研究了能够最小化或近似最小化经验风险的神经网络的泛化能力。基于这些网络的表达能力，建立了总体准确率的下界，表明在适当大量训练样本和网络规模的情况下，包括过参数化在内的这些网络能够有效地泛化。此外，我们提供了泛化的必要条件，证明了对于某些数据分布，确保泛化所需的训练数据量超过表示相应数据分布所需的网络规模。最后，本文提供了关于深度学习中robust泛化、过参数化的重要性以及损失函数对泛化的影响的一些理论见解。 

---
# InterChat: Enhancing Generative Visual Analytics using Multimodal Interactions 

**Title (ZH)**: InterChat: 利用多模态交互增强生成性视觉分析 

**Authors**: Juntong Chen, Jiang Wu, Jiajing Guo, Vikram Mohanty, Xueming Li, Jorge Piazentin Ono, Wenbin He, Liu Ren, Dongyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04110)  

**Abstract**: The rise of Large Language Models (LLMs) and generative visual analytics systems has transformed data-driven insights, yet significant challenges persist in accurately interpreting users' analytical and interaction intents. While language inputs offer flexibility, they often lack precision, making the expression of complex intents inefficient, error-prone, and time-intensive. To address these limitations, we investigate the design space of multimodal interactions for generative visual analytics through a literature review and pilot brainstorming sessions. Building on these insights, we introduce a highly extensible workflow that integrates multiple LLM agents for intent inference and visualization generation. We develop InterChat, a generative visual analytics system that combines direct manipulation of visual elements with natural language inputs. This integration enables precise intent communication and supports progressive, visually driven exploratory data analyses. By employing effective prompt engineering, and contextual interaction linking, alongside intuitive visualization and interaction designs, InterChat bridges the gap between user interactions and LLM-driven visualizations, enhancing both interpretability and usability. Extensive evaluations, including two usage scenarios, a user study, and expert feedback, demonstrate the effectiveness of InterChat. Results show significant improvements in the accuracy and efficiency of handling complex visual analytics tasks, highlighting the potential of multimodal interactions to redefine user engagement and analytical depth in generative visual analytics. 

**Abstract (ZH)**: 大型语言模型（LLMs）和生成式可视化分析系统的兴起已经转变了数据驱动的洞察，但准确解读用户分析和交互意图仍面临重大挑战。虽然语言输入提供了灵活性，但往往缺乏精确性，使得复杂意图的表达inefficient、error-prone且时间消耗。为克服这些限制，我们通过文献综述和初步头脑风暴探讨了生成式可视化分析中的多模态交互设计空间。基于这些见解，我们引入了一种高度可扩展的工作流程，该流程整合了多个LLM代理以进行意图推断和可视化生成。我们开发了InterChat，一个结合直接操作可视化元素与自然语言输入的生成式可视化分析系统。这种整合允许精确的意图通信，并支持逐步、以视觉为导向的数据探索分析。通过采用有效的提示工程、上下文交互链接，以及直观的可视化和交互设计，InterChat弥合了用户交互与LLM驱动的可视化之间的差距，提升了解释性和易用性。广泛的评估，包括两种使用场景、用户研究和专家反馈，证明了InterChat的有效性。结果显示，在处理复杂可视化分析任务方面的准确性和效率显著提高，突显了多模态交互在生成式可视化分析中重新定义用户参与度和分析深度的潜力。 

---
# Disparities in LLM Reasoning Accuracy and Explanations: A Case Study on African American English 

**Title (ZH)**: LLM推理准确性和解释的差异：关于非洲裔美国英语的案例研究 

**Authors**: Runtao Zhou, Guangya Wan, Saadia Gabriel, Sheng Li, Alexander J Gates, Maarten Sap, Thomas Hartvigsen  

**Link**: [PDF](https://arxiv.org/pdf/2503.04099)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in reasoning tasks, leading to their widespread deployment. However, recent studies have highlighted concerning biases in these models, particularly in their handling of dialectal variations like African American English (AAE). In this work, we systematically investigate dialectal disparities in LLM reasoning tasks. We develop an experimental framework comparing LLM performance given Standard American English (SAE) and AAE prompts, combining LLM-based dialect conversion with established linguistic analyses. We find that LLMs consistently produce less accurate responses and simpler reasoning chains and explanations for AAE inputs compared to equivalent SAE questions, with disparities most pronounced in social science and humanities domains. These findings highlight systematic differences in how LLMs process and reason about different language varieties, raising important questions about the development and deployment of these systems in our multilingual and multidialectal world. Our code repository is publicly available at this https URL. 

**Abstract (ZH)**: 大型语言模型在推理任务中展示了 remarkable 能力，导致它们被广泛应用。然而，近期研究指出这些模型中存在令人担忧的偏见，特别体现在方言变体处理方面，如美国黑人英语（AAE）。在本工作中，我们系统地研究了不同方言在大型语言模型推理任务中的差异。我们开发了一个实验框架，比较大型语言模型在标准美国英语（SAE）和AAE提示下的表现，结合基于大型语言模型的方言转换与传统的语言学分析。我们发现，对于AAE输入，大型语言模型生成的响应更不准确，推理链条和解释也更简单，这些差异在社会科学和人文学科领域尤为显著。这些发现强调了大型语言模型在处理和推理不同语言变体时的系统性差异，引发了关于这些系统在多语言、多方言世界中的开发和部署的重要问题。我们的代码库已在此 https URL 公开。 

---
# Chart-HQA: A Benchmark for Hypothetical Question Answering in Charts 

**Title (ZH)**: Chart-HQA：图表中的假设性问题回答基准 

**Authors**: Xiangnan Chen, Yuancheng Fang, Qian Xiao, Juncheng Li, Jun Lin, Siliang Tang, Yi Yang, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04095)  

**Abstract**: Multimodal Large Language Models (MLLMs) have garnered significant attention for their strong visual-semantic understanding. Most existing chart benchmarks evaluate MLLMs' ability to parse information from charts to answer this http URL, they overlook the inherent output biases of MLLMs, where models rely on their parametric memory to answer questions rather than genuinely understanding the chart content. To address this limitation, we introduce a novel Chart Hypothetical Question Answering (HQA) task, which imposes assumptions on the same question to compel models to engage in counterfactual reasoning based on the chart content. Furthermore, we introduce HAI, a human-AI interactive data synthesis approach that leverages the efficient text-editing capabilities of LLMs alongside human expert knowledge to generate diverse and high-quality HQA data at a low cost. Using HAI, we construct Chart-HQA, a challenging benchmark synthesized from publicly available data sources. Evaluation results on 18 MLLMs of varying model sizes reveal that current models face significant generalization challenges and exhibit imbalanced reasoning performance on the HQA task. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）因其强大的视觉语义理解能力而备受关注。现有大多数图表基准主要评估MLLMs从图表中解析信息以回答问题的能力，但忽视了MLLMs固有的输出偏见，模型依赖参数记忆而不是真正理解图表内容来回答问题。为解决这一局限，我们引入了一种新的图表假设性问题回答（HQA）任务，该任务通过对同一问题进行假设，促使模型基于图表内容进行反事实推理。此外，我们引入了HAI，这是一种结合了LLM高效文本编辑能力和人类专家知识的人机交互式数据合成方法，以低成本生成多样性和高质量的HQA数据。使用HAI，我们构建了Chart-HQA，这是一个从公开数据源合成的具有挑战性的基准。对18个不同模型规模的MLLMs的评估结果表明，当前模型在HQA任务上面临着显著的泛化挑战，并且在推理性能上表现出不平衡。 

---
# Can We Optimize Deep RL Policy Weights as Trajectory Modeling? 

**Title (ZH)**: 我们可以将深度RL策略权重优化视为轨迹建模吗？ 

**Authors**: Hongyao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04074)  

**Abstract**: Learning the optimal policy from a random network initialization is the theme of deep Reinforcement Learning (RL). As the scale of DRL training increases, treating DRL policy network weights as a new data modality and exploring the potential becomes appealing and possible. In this work, we focus on the policy learning path in deep RL, represented by the trajectory of network weights of historical policies, which reflects the evolvement of the policy learning process. Taking the idea of trajectory modeling with Transformer, we propose Transformer as Implicit Policy Learner (TIPL), which processes policy network weights in an autoregressive manner. We collect the policy learning path data by running independent RL training trials, with which we then train our TIPL model. In the experiments, we demonstrate that TIPL is able to fit the implicit dynamics of policy learning and perform the optimization of policy network by inference. 

**Abstract (ZH)**: 从随机网络初始化学习最优策略是深度强化学习的主题。随着DRL训练规模的扩大，将DRL策略网络权重视为一种新的数据模态并探索其潜力变得诱人且可能。在本文中，我们关注深度RL中的策略学习路径，即历史策略网络权重的轨迹，反映策略学习过程的演变。借鉴轨迹建模的思想，我们提出了一种隐式策略学习器（TIPL），以自回归方式处理策略网络权重。通过运行独立的RL训练试验收集策略学习路径数据，然后用于训练我们的TIPL模型。在实验中，我们展示了TIPL能够拟合策略学习的隐式动态并通过推理优化策略网络。 

---
# PP-DocBee: Improving Multimodal Document Understanding Through a Bag of Tricks 

**Title (ZH)**: PP-DocBee: 通过多种技巧提升多模态文档理解 

**Authors**: Feng Ni, Kui Huang, Yao Lu, Wenyu Lv, Guanzhong Wang, Zeyu Chen, Yi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04065)  

**Abstract**: With the rapid advancement of digitalization, various document images are being applied more extensively in production and daily life, and there is an increasingly urgent need for fast and accurate parsing of the content in document images. Therefore, this report presents PP-DocBee, a novel multimodal large language model designed for end-to-end document image understanding. First, we develop a data synthesis strategy tailored to document scenarios in which we build a diverse dataset to improve the model generalization. Then, we apply a few training techniques, including dynamic proportional sampling, data preprocessing, and OCR postprocessing strategies. Extensive evaluations demonstrate the superior performance of PP-DocBee, achieving state-of-the-art results on English document understanding benchmarks and even outperforming existing open source and commercial models in Chinese document understanding. The source code and pre-trained models are publicly available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 随着数字化的快速发展，各种文档图像在生产和日常生活中的应用越来越广泛，对文档图像内容的快速准确解析需求日益迫切。因此，本报告提出了一种专为端到端文档图像理解设计的新型多模态大语言模型PP-DocBee。首先，我们开发了一种针对文档场景的数据合成策略，构建了多样化的数据集以提高模型的泛化能力。然后，我们应用了包括动态比例采样、数据预处理和OCR后处理策略在内的多种训练技术。广泛的评估结果表明，PP-DocBee在英语文档理解基准测试中取得了最先进的性能，并在中文文档理解中甚至优于现有的开源和商用模型。源代码和预训练模型已公开可供访问。 

---
# Uncovering inequalities in new knowledge learning by large language models across different languages 

**Title (ZH)**: 探究大型语言模型在不同语言中新知识学习中的不平等现象 

**Authors**: Chenglong Wang, Haoyu Tang, Xiyuan Yang, Yueqi Xie, Jina Suh, Sunayana Sitaram, Junming Huang, Yu Xie, Zhaoya Gong, Xing Xie, Fangzhao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04064)  

**Abstract**: As large language models (LLMs) gradually become integral tools for problem solving in daily life worldwide, understanding linguistic inequality is becoming increasingly important. Existing research has primarily focused on static analyses that assess the disparities in the existing knowledge and capabilities of LLMs across languages. However, LLMs are continuously evolving, acquiring new knowledge to generate up-to-date, domain-specific responses. Investigating linguistic inequalities within this dynamic process is, therefore, also essential. In this paper, we explore inequalities in new knowledge learning by LLMs across different languages and four key dimensions: effectiveness, transferability, prioritization, and robustness. Through extensive experiments under two settings (in-context learning and fine-tuning) using both proprietary and open-source models, we demonstrate that low-resource languages consistently face disadvantages across all four dimensions. By shedding light on these disparities, we aim to raise awareness of linguistic inequalities in LLMs' new knowledge learning, fostering the development of more inclusive and equitable future LLMs. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在全球日常生活中逐渐成为解决问题的重要工具，理解语言不平等现象变得越来越重要。现有研究主要关注静态分析，评估不同语言中LLMs的知识和能力差异。然而，LLMs在不断进化，获得新的知识以生成最新的、针对性的回答。因此，研究这一动态过程中的语言不平等同样至关重要。在本文中，我们探讨了LLMs在不同语言和四个关键维度（有效性、可迁移性、优先级和稳健性）中学习新知识过程中的不平等现象。通过在两种设置（上下文学习和微调）下进行广泛实验，使用自有的和开源的模型，我们展示了低资源语言在所有四个维度上均面临不利状况。通过揭示这些差异，我们旨在提高对LLMs新知识学习中的语言不平等的认识，促进更加包容和公平的未来LLMs的发展。 

---
# Continual Optimization with Symmetry Teleportation for Multi-Task Learning 

**Title (ZH)**: 基于对称 teleportation 的持续优化多任务学习 

**Authors**: Zhipeng Zhou, Ziqiao Meng, Pengcheng Wu, Peilin Zhao, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04046)  

**Abstract**: Multi-task learning (MTL) is a widely explored paradigm that enables the simultaneous learning of multiple tasks using a single model. Despite numerous solutions, the key issues of optimization conflict and task imbalance remain under-addressed, limiting performance. Unlike existing optimization-based approaches that typically reweight task losses or gradients to mitigate conflicts or promote progress, we propose a novel approach based on Continual Optimization with Symmetry Teleportation (COST). During MTL optimization, when an optimization conflict arises, we seek an alternative loss-equivalent point on the loss landscape to reduce conflict. Specifically, we utilize a low-rank adapter (LoRA) to facilitate this practical teleportation by designing convergent, loss-invariant objectives. Additionally, we introduce a historical trajectory reuse strategy to continually leverage the benefits of advanced optimizers. Extensive experiments on multiple mainstream datasets demonstrate the effectiveness of our approach. COST is a plug-and-play solution that enhances a wide range of existing MTL methods. When integrated with state-of-the-art methods, COST achieves superior performance. 

**Abstract (ZH)**: 基于连续优化与对称 teleport 的多任务学习（COST） 

---
# TextDoctor: Unified Document Image Inpainting via Patch Pyramid Diffusion Models 

**Title (ZH)**: 文本医生：基于patch金字塔扩散模型的统一文档图像插补 

**Authors**: Wanglong Lu, Lingming Su, Jingjing Zheng, Vinícius Veloso de Melo, Farzaneh Shoeleh, John Hawkin, Terrence Tricco, Hanli Zhao, Xianta Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04021)  

**Abstract**: Digital versions of real-world text documents often suffer from issues like environmental corrosion of the original document, low-quality scanning, or human interference. Existing document restoration and inpainting methods typically struggle with generalizing to unseen document styles and handling high-resolution images. To address these challenges, we introduce TextDoctor, a novel unified document image inpainting method. Inspired by human reading behavior, TextDoctor restores fundamental text elements from patches and then applies diffusion models to entire document images instead of training models on specific document types. To handle varying text sizes and avoid out-of-memory issues, common in high-resolution documents, we propose using structure pyramid prediction and patch pyramid diffusion models. These techniques leverage multiscale inputs and pyramid patches to enhance the quality of inpainting both globally and locally. Extensive qualitative and quantitative experiments on seven public datasets validated that TextDoctor outperforms state-of-the-art methods in restoring various types of high-resolution document images. 

**Abstract (ZH)**: TextDoctor：一种基于人体阅读行为的统一文档图像 inpainting 方法 

---
# Benchmarking Large Language Models on Multiple Tasks in Bioinformatics NLP with Prompting 

**Title (ZH)**: 大型语言模型在生物信息学NLP任务中的多任务基准测试与提示方法 

**Authors**: Jiyue Jiang, Pengan Chen, Jiuming Wang, Dongchen He, Ziqin Wei, Liang Hong, Licheng Zong, Sheng Wang, Qinze Yu, Zixian Ma, Yanyu Chen, Yimin Fan, Xiangyu Shi, Jiawei Sun, Chuan Wu, Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.04013)  

**Abstract**: Large language models (LLMs) have become important tools in solving biological problems, offering improvements in accuracy and adaptability over conventional methods. Several benchmarks have been proposed to evaluate the performance of these LLMs. However, current benchmarks can hardly evaluate the performance of these models across diverse tasks effectively. In this paper, we introduce a comprehensive prompting-based benchmarking framework, termed Bio-benchmark, which includes 30 key bioinformatics tasks covering areas such as proteins, RNA, drugs, electronic health records, and traditional Chinese medicine. Using this benchmark, we evaluate six mainstream LLMs, including GPT-4o and Llama-3.1-70b, etc., using 0-shot and few-shot Chain-of-Thought (CoT) settings without fine-tuning to reveal their intrinsic capabilities. To improve the efficiency of our evaluations, we demonstrate BioFinder, a new tool for extracting answers from LLM responses, which increases extraction accuracy by round 30% compared to existing methods. Our benchmark results show the biological tasks suitable for current LLMs and identify specific areas requiring enhancement. Furthermore, we propose targeted prompt engineering strategies for optimizing LLM performance in these contexts. Based on these findings, we provide recommendations for the development of more robust LLMs tailored for various biological applications. This work offers a comprehensive evaluation framework and robust tools to support the application of LLMs in bioinformatics. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已成为解决生物学问题的重要工具，展现出比传统方法更高的准确性和适应性。已提出了多种基准来评估这些LLMs的性能。然而，当前的基准难以有效地全面评估这些模型在多样任务中的表现。本文引入了一个全面的提示驱动基准框架，名为Bio-benchmark，包括30项关键生物信息学任务，涵盖蛋白质、RNA、药物、电子健康记录以及传统中药等领域。利用这一基准，我们评估了包括GPT-4o和Llama-3.1-70b在内的六种主流LLMs，在未进行微调的情况下采用零样本和少量样本链式思维（CoT）设置，揭示其内在能力。为了提高评估效率，我们展示了BioFinder这一新的工具，用于从LLM响应中提取答案，其提取准确率相比现有方法提升了约30%。基准结果展示了当前LLMs适合的生物任务，并指出了需要改进的具体领域。此外，我们提出了针对这些领域的目标提示工程策略以优化LLM性能。基于这些发现，我们提出了针对各种生物应用开发更稳健LLMs的建议。本研究提供了一个全面的评估框架和可靠的工具，以支持LLMs在生物信息学中的应用。 

---
# Subgraph Federated Learning for Local Generalization 

**Title (ZH)**: 局部泛化的子图联邦学习 

**Authors**: Sungwon Kim, Yoonho Lee, Yunhak Oh, Namkyeong Lee, Sukwon Yun, Junseok Lee, Sein Kim, Carl Yang, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2503.03995)  

**Abstract**: Federated Learning (FL) on graphs enables collaborative model training to enhance performance without compromising the privacy of each client. However, existing methods often overlook the mutable nature of graph data, which frequently introduces new nodes and leads to shifts in label distribution. Since they focus solely on performing well on each client's local data, they are prone to overfitting to their local distributions (i.e., local overfitting), which hinders their ability to generalize to unseen data with diverse label distributions. In contrast, our proposed method, FedLoG, effectively tackles this issue by mitigating local overfitting. Our model generates global synthetic data by condensing the reliable information from each class representation and its structural information across clients. Using these synthetic data as a training set, we alleviate the local overfitting problem by adaptively generalizing the absent knowledge within each local dataset. This enhances the generalization capabilities of local models, enabling them to handle unseen data effectively. Our model outperforms baselines in our proposed experimental settings, which are designed to measure generalization power to unseen data in practical scenarios. Our code is available at this https URL 

**Abstract (ZH)**: 图上的联邦学习（FL）能够通过协作训练模型来提升性能，同时不牺牲每个客户端的隐私。然而，现有方法往往忽视了图数据的可变性，这导致新节点的引入和标签分布的变化。由于它们仅专注于在每个客户端本地数据上的表现，这些方法容易过度拟合本地分布（即局部过拟合），这阻碍了它们向具有不同标签分布的未见数据泛化的能... 

---
# RetinalGPT: A Retinal Clinical Preference Conversational Assistant Powered by Large Vision-Language Models 

**Title (ZH)**: RetinalGPT：由大规模视觉-语言模型驱动的视网膜临床偏好对话助理 

**Authors**: Wenhui Zhu, Xin Li, Xiwen Chen, Peijie Qiu, Vamsi Krishna Vasa, Xuanzhao Dong, Yanxi Chen, Natasha Lepore, Oana Dumitrascu, Yi Su, Yalin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03987)  

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have gained significant attention for their remarkable ability to process and analyze non-textual data, such as images, videos, and audio. Notably, several adaptations of general-domain MLLMs to the medical field have been explored, including LLaVA-Med. However, these medical adaptations remain insufficiently advanced in understanding and interpreting retinal images. In contrast, medical experts emphasize the importance of quantitative analyses for disease detection and interpretation. This underscores a gap between general-domain and medical-domain MLLMs: while general-domain MLLMs excel in broad applications, they lack the specialized knowledge necessary for precise diagnostic and interpretative tasks in the medical field. To address these challenges, we introduce \textit{RetinalGPT}, a multimodal conversational assistant for clinically preferred quantitative analysis of retinal images. Specifically, we achieve this by compiling a large retinal image dataset, developing a novel data pipeline, and employing customized visual instruction tuning to enhance both retinal analysis and enrich medical knowledge. In particular, RetinalGPT outperforms MLLM in the generic domain by a large margin in the diagnosis of retinal diseases in 8 benchmark retinal datasets. Beyond disease diagnosis, RetinalGPT features quantitative analyses and lesion localization, representing a pioneering step in leveraging LLMs for an interpretable and end-to-end clinical research framework. The code is available at this https URL 

**Abstract (ZH)**: 最近，多模态大型语言模型（MLLMs）因其处理和分析非文本数据（如图像、视频和音频）的出色能力而获得了广泛关注。值得注意的是，已有研究将通用领域MLLMs应用于医疗领域，包括LLaVA-Med。然而，这些医疗适应性在理解与解释眼底图像方面仍然不够深入。相比之下，医疗专家强调定量分析对于疾病检测和解释的重要性。这突显了通用领域与医疗领域MLLMs之间的差距：虽然通用领域MLLMs在广泛的应用中表现出色，但在医疗领域的精确诊断和解释任务中缺乏必要的专门知识。为解决这些挑战，我们提出了\textit{RetinalGPT}——一种临床优选的多模态对话助手，专门用于眼底图像的定量分析。具体而言，我们通过构建大规模眼底图像数据集、开发新型数据管道以及采用定制化的视觉指令调优，增强了眼底分析能力并丰富了医疗知识。特别地，RetinalGPT在8个基准眼底数据集中的眼底疾病诊断上远超通用领域MLLM。除了疾病诊断，RetinalGPT还具备定量分析和病灶定位的功能，标志着利用LLMs构建可解释和端到端的临床研究框架的先驱性步骤。代码已发布。 

---
# Training neural networks faster with minimal tuning using pre-computed lists of hyperparameters for NAdamW 

**Title (ZH)**: 使用预计算的超参数列表加速NAdamW神经网络训练且无需精细调参 

**Authors**: Sourabh Medapati, Priya Kasimbeg, Shankar Krishnan, Naman Agarwal, George Dahl  

**Link**: [PDF](https://arxiv.org/pdf/2503.03986)  

**Abstract**: If we want to train a neural network using any of the most popular optimization algorithms, we are immediately faced with a dilemma: how to set the various optimization and regularization hyperparameters? When computational resources are abundant, there are a variety of methods for finding good hyperparameter settings, but when resources are limited the only realistic choices are using standard default values of uncertain quality and provenance, or tuning only a couple of the most important hyperparameters via extremely limited handdesigned sweeps. Extending the idea of default settings to a modest tuning budget, Metz et al. (2020) proposed using ordered lists of well-performing hyperparameter settings, derived from a broad hyperparameter search on a large library of training workloads. However, to date, no practical and performant hyperparameter lists that generalize to representative deep learning workloads have been demonstrated. In this paper, we present hyperparameter lists for NAdamW derived from extensive experiments on the realistic workloads in the AlgoPerf: Training Algorithms benchmark. Our hyperparameter lists also include values for basic regularization techniques (i.e. weight decay, label smoothing, and dropout). In particular, our best NAdamW hyperparameter list performs well on AlgoPerf held-out workloads not used to construct it, and represents a compelling turn-key approach to tuning when restricted to five or fewer trials. It also outperforms basic learning rate/weight decay sweeps and an off-the-shelf Bayesian optimization tool when restricted to the same budget. 

**Abstract (ZH)**: 基于AlgoPerf: Training Algorithms基准的实用且性能良好的NAdamW超参数列表 

---
# All-atom Diffusion Transformers: Unified generative modelling of molecules and materials 

**Title (ZH)**: 全原子扩散变换器：分子和材料的一体化生成建模 

**Authors**: Chaitanya K. Joshi, Xiang Fu, Yi-Lun Liao, Vahe Gharakhanyan, Benjamin Kurt Miller, Anuroop Sriram, Zachary W. Ulissi  

**Link**: [PDF](https://arxiv.org/pdf/2503.03965)  

**Abstract**: Diffusion models are the standard toolkit for generative modelling of 3D atomic systems. However, for different types of atomic systems - such as molecules and materials - the generative processes are usually highly specific to the target system despite the underlying physics being the same. We introduce the All-atom Diffusion Transformer (ADiT), a unified latent diffusion framework for jointly generating both periodic materials and non-periodic molecular systems using the same model: (1) An autoencoder maps a unified, all-atom representations of molecules and materials to a shared latent embedding space; and (2) A diffusion model is trained to generate new latent embeddings that the autoencoder can decode to sample new molecules or materials. Experiments on QM9 and MP20 datasets demonstrate that jointly trained ADiT generates realistic and valid molecules as well as materials, exceeding state-of-the-art results from molecule and crystal-specific models. ADiT uses standard Transformers for both the autoencoder and diffusion model, resulting in significant speedups during training and inference compared to equivariant diffusion models. Scaling ADiT up to half a billion parameters predictably improves performance, representing a step towards broadly generalizable foundation models for generative chemistry. Open source code: this https URL 

**Abstract (ZH)**: 全域原子扩散变换器（ADiT）：统一的周期材料与非周期分子系统的联合生成框架 

---
# WIP: Assessing the Effectiveness of ChatGPT in Preparatory Testing Activities 

**Title (ZH)**: WIP: 评估ChatGPT在预备性测试活动中的有效性 

**Authors**: Susmita Haldar, Mary Pierce, Luiz Fernando Capretz  

**Link**: [PDF](https://arxiv.org/pdf/2503.03951)  

**Abstract**: This innovative practice WIP paper describes a research study that explores the integration of ChatGPT into the software testing curriculum and evaluates its effectiveness compared to human-generated testing artifacts. In a Capstone Project course, students were tasked with generating preparatory testing artifacts using ChatGPT prompts, which they had previously created manually. Their understanding and the effectiveness of the Artificial Intelligence generated artifacts were assessed through targeted questions. The results, drawn from this in-class assignment at a North American community college indicate that while ChatGPT can automate many testing preparation tasks, it cannot fully replace human expertise. However, students, already familiar with Information Technology at the postgraduate level, found the integration of ChatGPT into their workflow to be straightforward. The study suggests that AI can be gradually introduced into software testing education to keep pace with technological advancements. 

**Abstract (ZH)**: 这篇创新性的实践WIP论文描述了一项研究，探讨了将ChatGPT整合到软件测试课程中的方法，并评估了其效果，对比了由ChatGPT生成的测试制品与人工生成的测试制品。在一项综合项目课程中，学生被要求使用ChatGPT提示生成预备测试制品，这些提示他们之前是手动创建的。通过对目标问题的回答评估了学生对生成的人工智能制品的理解和效果。这些结果来自北美社区学院的一项课堂作业，表明虽然ChatGPT可以自动化许多测试准备任务，但无法完全替代人类的专业知识。然而，已经具备研究生水平信息技术背景的学生发现将ChatGPT整合到他们的工作流程中是简单的。研究建议可以在软件测试教育中逐步引入人工智能，以适应技术的发展。 

---
# COARSE: Collaborative Pseudo-Labeling with Coarse Real Labels for Off-Road Semantic Segmentation 

**Title (ZH)**: COARSE: 基于粗略真实标签的协作伪标签生成方法在离路语义分割中的应用 

**Authors**: Aurelio Noca, Xianmei Lei, Jonathan Becktor, Jeffrey Edlund, Anna Sabel, Patrick Spieler, Curtis Padgett, Alexandre Alahi, Deegan Atha  

**Link**: [PDF](https://arxiv.org/pdf/2503.03947)  

**Abstract**: Autonomous off-road navigation faces challenges due to diverse, unstructured environments, requiring robust perception with both geometric and semantic understanding. However, scarce densely labeled semantic data limits generalization across domains. Simulated data helps, but introduces domain adaptation issues. We propose COARSE, a semi-supervised domain adaptation framework for off-road semantic segmentation, leveraging sparse, coarse in-domain labels and densely labeled out-of-domain data. Using pretrained vision transformers, we bridge domain gaps with complementary pixel-level and patch-level decoders, enhanced by a collaborative pseudo-labeling strategy on unlabeled data. Evaluations on RUGD and Rellis-3D datasets show significant improvements of 9.7\% and 8.4\% respectively, versus only using coarse data. Tests on real-world off-road vehicle data in a multi-biome setting further demonstrate COARSE's applicability. 

**Abstract (ZH)**: 自主离路面导航由于面对多样且结构不规则的环境而面临挑战，需要兼具几何理解和语义理解的鲁棒感知。然而，稀缺的密集标注语义数据限制了其跨领域的泛化能力。模拟数据有所帮助，但引入了领域适应问题。我们提出COARSE，一种结合领域内稀疏粗略标注和领域外密集标注数据的半监督领域适应框架，利用预训练的视觉变换器，通过互补的像素级和块级解码器，并结合协作的伪标签策略，跨越领域鸿沟。在RUGD和Rellis-3D数据集上的评估显示，相比于仅使用粗略标注数据，准确率分别提高了9.7%和8.4%。在多种生物群落的真实世界离路面车辆数据上进一步测试证明了COARSE的适用性。 

---
# GlucoLens: Explainable Postprandial Blood Glucose Prediction from Diet and Physical Activity 

**Title (ZH)**: GlucoLens：基于饮食和体育活动的可解释餐后血糖预测 

**Authors**: Abdullah Mamun, Asiful Arefeen, Susan B. Racette, Dorothy D. Sears, Corrie M. Whisner, Matthew P. Buman, Hassan Ghasemzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2503.03935)  

**Abstract**: Postprandial hyperglycemia, marked by the blood glucose level exceeding the normal range after meals, is a critical indicator of progression toward type 2 diabetes in prediabetic and healthy individuals. A key metric for understanding blood glucose dynamics after eating is the postprandial area under the curve (PAUC). Predicting PAUC in advance based on a person's diet and activity level and explaining what affects postprandial blood glucose could allow an individual to adjust their lifestyle accordingly to maintain normal glucose levels. In this paper, we propose GlucoLens, an explainable machine learning approach to predict PAUC and hyperglycemia from diet, activity, and recent glucose patterns. We conducted a five-week user study with 10 full-time working individuals to develop and evaluate the computational model. Our machine learning model takes multimodal data including fasting glucose, recent glucose, recent activity, and macronutrient amounts, and provides an interpretable prediction of the postprandial glucose pattern. Our extensive analyses of the collected data revealed that the trained model achieves a normalized root mean squared error (NRMSE) of 0.123. On average, GlucoLense with a Random Forest backbone provides a 16% better result than the baseline models. Additionally, GlucoLens predicts hyperglycemia with an accuracy of 74% and recommends different options to help avoid hyperglycemia through diverse counterfactual explanations. Code available: this https URL. 

**Abstract (ZH)**: 餐后高血糖，表现为餐后血糖水平超出正常范围，是预测糖尿病前期和健康个体向2型糖尿病进展的关键指标。理解进食后血糖动态的一个重要指标是餐后曲线下面积（PAUC）。基于个人饮食和活动水平提前预测PAUC，并解释影响餐后血糖的因素，可以使个体根据需要调整生活方式以维持正常的血糖水平。本文提出了一种名为GlucoLens的可解释机器学习方法，用于从饮食、活动和近期血糖模式预测PAUC和高血糖。我们进行了为期五周的用户研究，涉及10名全职工作人员，以开发和评估计算模型。我们的机器学习模型利用空腹血糖、近期血糖、近期活动和宏量营养素含量等多种模态数据，提供可解释的餐后血糖模式预测。通过对收集数据的广泛分析，我们发现训练模型的归一化均方根误差（NRMSE）为0.123。平均而言，基于随机森林的GlucoLens比基线模型提供了16%更好的结果。此外，GlucoLens在高血糖预测方面准确率为74%，并通过多种反事实解释推荐不同的选项以帮助避免高血糖。代码可用：this https URL。 

---
# "Impressively Scary:" Exploring User Perceptions and Reactions to Unraveling Machine Learning Models in Social Media Applications 

**Title (ZH)**: “令人震撼地可怕”：探索用户对揭开社交媒体应用中机器学习模型的感知与反应 

**Authors**: Jack West, Bengisu Cagiltay, Shirley Zhang, Jingjie Li, Kassem Fawaz, Suman Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2503.03927)  

**Abstract**: Machine learning models deployed locally on social media applications are used for features, such as face filters which read faces in-real time, and they expose sensitive attributes to the apps. However, the deployment of machine learning models, e.g., when, where, and how they are used, in social media applications is opaque to users. We aim to address this inconsistency and investigate how social media user perceptions and behaviors change once exposed to these models. We conducted user studies (N=21) and found that participants were unaware to both what the models output and when the models were used in Instagram and TikTok, two major social media platforms. In response to being exposed to the models' functionality, we observed long term behavior changes in 8 participants. Our analysis uncovers the challenges and opportunities in providing transparency for machine learning models that interact with local user data. 

**Abstract (ZH)**: 本地部署在社交媒体应用上的机器学习模型用于面部过滤等功能，暴露出敏感属性给应用程序。然而，这些模型的部署，如何时、何地以及如何使用，对用户是透明度不足的。我们旨在解决这一不一致性，并调查用户在接触这些模型后对这些模型的看法和行为如何变化。我们进行了用户研究（N=21），发现参与者对Instagram和TikTok这两个主要社交媒体平台上的模型输出和使用时机并不了解。当参与者接触到这些模型的功能时，我们观察到8名参与者出现了长期的行为变化。我们的分析揭示了提供与本地用户数据交互的机器学习模型透明度所面临的挑战与机遇。 

---
# De-skilling, Cognitive Offloading, and Misplaced Responsibilities: Potential Ironies of AI-Assisted Design 

**Title (ZH)**: 技能分流、认知卸载与责任错位：AI辅助设计的潜在讽刺之处 

**Authors**: Prakash Shukla, Phuong Bui, Sean S Levy, Max Kowalski, Ali Baigelenov, Paul Parsons  

**Link**: [PDF](https://arxiv.org/pdf/2503.03924)  

**Abstract**: The rapid adoption of generative AI (GenAI) in design has sparked discussions about its benefits and unintended consequences. While AI is often framed as a tool for enhancing productivity by automating routine tasks, historical research on automation warns of paradoxical effects, such as de-skilling and misplaced responsibilities. To assess UX practitioners' perceptions of AI, we analyzed over 120 articles and discussions from UX-focused subreddits. Our findings indicate that while practitioners express optimism about AI reducing repetitive work and augmenting creativity, they also highlight concerns about over-reliance, cognitive offloading, and the erosion of critical design skills. Drawing from human-automation interaction literature, we discuss how these perspectives align with well-documented automation ironies and function allocation challenges. We argue that UX professionals should critically evaluate AI's role beyond immediate productivity gains and consider its long-term implications for creative autonomy and expertise. This study contributes empirical insights into practitioners' perspectives and links them to broader debates on automation in design. 

**Abstract (ZH)**: Generative AI在设计中的快速采用引发了对其益处和意外后果的讨论。尽管AI常被视为通过自动化常规任务来增强生产力的工具，历史上的自动化研究警告可能产生悖论效应，如技能退化和责任错位。为了评估用户体验从业者对AI的看法，我们分析了来自专注于用户体验的reddit子版块的超过120篇文章和讨论。我们的研究发现，虽然从业者对AI减轻重复工作并增强创造力表达了乐观态度，但他们也指出了过度依赖、认知卸载以及设计关键技能侵蚀的担忧。借鉴人机交互领域的研究文献，我们讨论了这些观点如何与广泛的自动化悖论和功能分配挑战相吻合。我们认为，用户体验专业人士应当批判性地评估AI的role，而不仅仅是其短期生产效率的提升，并考虑其对创造性自主权和专业知识的长期影响。本研究为从业者的观点提供了实证洞察，并将其与设计领域更广泛的自动化辩论联系起来。 

---
# CREStE: Scalable Mapless Navigation with Internet Scale Priors and Counterfactual Guidance 

**Title (ZH)**: CREStE: 基于互联网规模先验和反事实指导的可扩展无地图导航 

**Authors**: Arthur Zhang, Harshit Sikchi, Amy Zhang, Joydeep Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2503.03921)  

**Abstract**: We address the long-horizon mapless navigation problem: enabling robots to traverse novel environments without relying on high-definition maps or precise waypoints that specify exactly where to navigate. Achieving this requires overcoming two major challenges -- learning robust, generalizable perceptual representations of the environment without pre-enumerating all possible navigation factors and forms of perceptual aliasing and utilizing these learned representations to plan human-aligned navigation paths. Existing solutions struggle to generalize due to their reliance on hand-curated object lists that overlook unforeseen factors, end-to-end learning of navigation features from scarce large-scale robot datasets, and handcrafted reward functions that scale poorly to diverse scenarios. To overcome these limitations, we propose CREStE, the first method that learns representations and rewards for addressing the full mapless navigation problem without relying on large-scale robot datasets or manually curated features. CREStE leverages visual foundation models trained on internet-scale data to learn continuous bird's-eye-view representations capturing elevation, semantics, and instance-level features. To utilize learned representations for planning, we propose a counterfactual-based loss and active learning procedure that focuses on the most salient perceptual cues by querying humans for counterfactual trajectory annotations in challenging scenes. We evaluate CREStE in kilometer-scale navigation tasks across six distinct urban environments. CREStE significantly outperforms all state-of-the-art approaches with 70% fewer human interventions per mission, including a 2-kilometer mission in an unseen environment with just 1 intervention; showcasing its robustness and effectiveness for long-horizon mapless navigation. For videos and additional materials, see this https URL . 

**Abstract (ZH)**: 我们解决长期视角的无地图导航问题：使机器人能够在不依赖高精度地图或精确航点的情况下穿越新型环境。实现这一目标需要克服两大挑战——学习适应环境的健壯且通用的感知表示而不预先列举所有可能的导航因素和感知歧义形式，并利用这些学习到的表示来规划人本导向的导航路径。现有解决方案由于依赖于手选对象列表、从大规模机器人数据集中端到端学习导航特征、以及难以扩展的手工艺奖励函数而难以泛化。为克服这些限制，我们提出了CREStE，这是首个不依赖大规模机器人数据集或手工选择特征来学习表示和奖励的方法，以解决完整无地图导航问题。CREStE利用训练于互联网规模数据的视觉基础模型学习连续视角表示，捕捉地形、语义和实例级特征。为了利用学习到的表示来进行规划，我们提出了一种基于反事实的损失和主动学习过程，在具有挑战性的场景中通过查询人类对抗事实轨迹注解来关注最具显著性的感知线索。我们评估了CREStE在六种不同城市环境中的千米级导航任务。CREStE在每次任务中显著优于所有最先进的方法，包括在未见过的环境中完成2千米任务时仅需1次干预，展示了其在长期视角无地图导航中的稳健性和有效性。更多视频和材料请见此链接：[链接]。 

---
# Not-Just-Scaling Laws: Towards a Better Understanding of the Downstream Impact of Language Model Design Decisions 

**Title (ZH)**: 不仅仅是规模法则：关于语言模型设计决策下游影响的更好理解 

**Authors**: Emmy Liu, Amanda Bertsch, Lintang Sutawika, Lindia Tjuatja, Patrick Fernandes, Lara Marinov, Michael Chen, Shreya Singhal, Carolin Lawrence, Aditi Raghunathan, Kiril Gashteovski, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2503.03862)  

**Abstract**: Improvements in language model capabilities are often attributed to increasing model size or training data, but in some cases smaller models trained on curated data or with different architectural decisions can outperform larger ones trained on more tokens. What accounts for this? To quantify the impact of these design choices, we meta-analyze 92 open-source pretrained models across a wide array of scales, including state-of-the-art open-weights models as well as less performant models and those with less conventional design decisions. We find that by incorporating features besides model size and number of training tokens, we can achieve a relative 3-28% increase in ability to predict downstream performance compared with using scale alone. Analysis of model design decisions reveal insights into data composition, such as the trade-off between language and code tasks at 15-25\% code, as well as the better performance of some architectural decisions such as choosing rotary over learned embeddings. Broadly, our framework lays a foundation for more systematic investigation of how model development choices shape final capabilities. 

**Abstract (ZH)**: 语言模型能力的改进通常归因于模型规模的增加或训练数据量的增多，但在某些情况下，经过精心选择数据或采用不同架构决策训练的较小模型可能会优于更大规模但训练数据更多的模型。是什么造成了这种差异？为了量化这些设计选择的影响，我们对92个开源的预训练模型进行了元分析，这些模型涵盖了从最先进的开放权重模型到性能较低且设计较为非传统的模型，包括各种规模的模型。我们发现在除了模型规模和训练tokens数量之外，加入其他特征可以使下游性能预测能力相对提高3%-28%。通过对模型设计决策的分析，我们揭示了数据组成方面的见解，例如当代码任务占比在15%-25%时语言和代码任务之间的权衡，并且某些架构决策，如选择旋转嵌入而非学习嵌入，具有更好的性能。总体而言，我们的框架为更系统地研究模型开发选择如何塑造最终能力奠定了基础。 

---
# Task-Agnostic Attacks Against Vision Foundation Models 

**Title (ZH)**: 面向任务的视觉基础模型攻击 

**Authors**: Brian Pulfer, Yury Belousov, Vitaliy Kinakh, Teddy Furon, Slava Voloshynovskiy  

**Link**: [PDF](https://arxiv.org/pdf/2503.03842)  

**Abstract**: The study of security in machine learning mainly focuses on downstream task-specific attacks, where the adversarial example is obtained by optimizing a loss function specific to the downstream task. At the same time, it has become standard practice for machine learning practitioners to adopt publicly available pre-trained vision foundation models, effectively sharing a common backbone architecture across a multitude of applications such as classification, segmentation, depth estimation, retrieval, question-answering and more. The study of attacks on such foundation models and their impact to multiple downstream tasks remains vastly unexplored. This work proposes a general framework that forges task-agnostic adversarial examples by maximally disrupting the feature representation obtained with foundation models. We extensively evaluate the security of the feature representations obtained by popular vision foundation models by measuring the impact of this attack on multiple downstream tasks and its transferability between models. 

**Abstract (ZH)**: 机器学习中的安全性研究主要侧重于下游任务特定的攻击，其中恶意样本通过优化特定于下游任务的损失函数获得。同时，机器学习 practitioners 广泛采用公开可用的预训练视觉基础模型，有效地在分类、分割、深度估计、检索、问答等多种应用中共享一个共同的基本架构。对这类基础模型及其对多个下游任务的影响的研究仍严重不足。本工作提出了一种通用框架，通过最大程度地破坏基础模型获得的特征表示来生成任务无关的恶意样本。我们通过测量该攻击对多个下游任务的影响及其在模型间迁移性来全面评估流行视觉基础模型的特征表示安全性。 

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
# VoiceGRPO: Modern MoE Transformers with Group Relative Policy Optimization GRPO for AI Voice Health Care Applications on Voice Pathology Detection 

**Title (ZH)**: VoiceGRPO: 基于组相对策略优化GRPO的现代MoE变压器在语音病理检测中的AI语音健康管理应用 

**Authors**: Enkhtogtokh Togootogtokh, Christian Klasen  

**Link**: [PDF](https://arxiv.org/pdf/2503.03797)  

**Abstract**: This research introduces a novel AI techniques as Mixture-of-Experts Transformers with Group Relative Policy Optimization (GRPO) for voice health care applications on voice pathology detection. With the architectural innovations, we adopt advanced training paradigms inspired by reinforcement learning, namely Proximal Policy Optimization (PPO) and Group-wise Regularized Policy Optimization (GRPO), to enhance model stability and performance. Experiments conducted on a synthetically generated voice pathology dataset demonstrate that our proposed models significantly improve diagnostic accuracy, F1 score, and ROC-AUC compared to conventional approaches. These findings underscore the potential of integrating transformer architectures with novel training strategies to advance automated voice pathology detection and ultimately contribute to more effective healthcare delivery. The code we used to train and evaluate our models is available at this https URL 

**Abstract (ZH)**: 这种研究引入了一种新型AI技术，即Mixture-of-Experts Transformer与Group Relative Policy Optimization (GRPO)方法，用于语音病理检测的语音健康管理应用。通过架构创新，我们采用由强化学习启发的先进训练范式，即Proximal Policy Optimization (PPO)和Group-wise Regularized Policy Optimization (GRPO)，以增强模型稳定性和性能。实验结果表明，与传统方法相比，我们提出的模型显著提高了诊断准确率、F1分数和ROC-AUC。这些发现强调了将变压器架构与新型训练策略集成以推动自动化语音病理检测的潜力，并最终有助于更有效的健康医疗服务。用于训练和评估我们模型的代码可在以下网址获取：this https URL。 

---
# Human Implicit Preference-Based Policy Fine-tuning for Multi-Agent Reinforcement Learning in USV Swarm 

**Title (ZH)**: 基于人类隐含偏好的多agent强化学习USV群簇政策微调 

**Authors**: Hyeonjun Kim, Kanghoon Lee, Junho Park, Jiachen Li, Jinkyoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2503.03796)  

**Abstract**: Multi-Agent Reinforcement Learning (MARL) has shown promise in solving complex problems involving cooperation and competition among agents, such as an Unmanned Surface Vehicle (USV) swarm used in search and rescue, surveillance, and vessel protection. However, aligning system behavior with user preferences is challenging due to the difficulty of encoding expert intuition into reward functions. To address the issue, we propose a Reinforcement Learning with Human Feedback (RLHF) approach for MARL that resolves credit-assignment challenges through an Agent-Level Feedback system categorizing feedback into intra-agent, inter-agent, and intra-team types. To overcome the challenges of direct human feedback, we employ a Large Language Model (LLM) evaluator to validate our approach using feedback scenarios such as region constraints, collision avoidance, and task allocation. Our method effectively refines USV swarm policies, addressing key challenges in multi-agent systems while maintaining fairness and performance consistency. 

**Abstract (ZH)**: 多代理强化学习（MARL）在解决涉及代理间合作与竞争的复杂问题中显示出前景，例如用于搜索救援、监视和船只保护的无人水面 vehicle（USV）群。然而，由于难以将专家直觉编码到奖励函数中，使得系统行为与用户偏好对齐具有挑战性。为解决这一问题，我们提出了一种基于人类反馈的强化学习（RLHF）方法，通过代理级反馈系统将反馈分类为代理内、代理间和团队内类型，以解决信用分配问题。为了克服直接人类反馈的挑战，我们采用大型语言模型（LLM）评估器，使用区域约束、避碰和任务分配等反馈场景验证我们的方法。该方法有效地细化了USV群的策略，同时解决了多代理系统中的关键挑战，并保持了公平性和性能一致性。 

---
# Synthetic Data Augmentation for Enhancing Harmful Algal Bloom Detection with Machine Learning 

**Title (ZH)**: 合成数据增强以提高机器学习在水华检测中的效果 

**Authors**: Tianyi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03794)  

**Abstract**: Harmful Algal Blooms (HABs) pose severe threats to aquatic ecosystems and public health, resulting in substantial economic losses globally. Early detection is crucial but often hindered by the scarcity of high-quality datasets necessary for training reliable machine learning (ML) models. This study investigates the use of synthetic data augmentation using Gaussian Copulas to enhance ML-based HAB detection systems. Synthetic datasets of varying sizes (100-1,000 samples) were generated using relevant environmental features$\unicode{x2015}$water temperature, salinity, and UVB radiation$\unicode{x2015}$with corrected Chlorophyll-a concentration as the target variable. Experimental results demonstrate that moderate synthetic augmentation significantly improves model performance (RMSE reduced from 0.4706 to 0.1850; $p < 0.001$). However, excessive synthetic data introduces noise and reduces predictive accuracy, emphasizing the need for a balanced approach to data augmentation. These findings highlight the potential of synthetic data to enhance HAB monitoring systems, offering a scalable and cost-effective method for early detection and mitigation of ecological and public health risks. 

**Abstract (ZH)**: 合成数据增强在使用高斯copula改善基于机器学习的水华检测系统中的应用：对 aquatic ecosystems 和公众健康的严重威胁要求早期检测，但由于高质量数据集的稀缺性，这常常受到阻碍。本研究探讨了使用高斯copula生成合成数据增强以提高基于机器学习的水华检测系统性能的方法。通过相关环境特征（水温、盐度和UVB辐射）生成大小不同的合成数据集（100-1,000样本），以修正的叶绿素-a浓度为目标变量。实验结果表明，适度的合成数据增强显着提高了模型性能（均方根误差从0.4706降低到0.1850，p<0.001）。然而，过度生成合成数据会引入噪声并降低预测准确性，强调需要在数据增强方面寻求平衡的方法。这些发现突显了合成数据在增强水华监测系统中的潜力，提供了一种可扩展且成本有效的早期检测和减轻生态和公共卫生风险的方法。 

---
# Rebalanced Multimodal Learning with Data-aware Unimodal Sampling 

**Title (ZH)**: 数据意识单模抽样驱动的重新平衡多模态学习 

**Authors**: Qingyuan Jiang, Zhouyang Chi, Xiao Ma, Qirong Mao, Yang Yang, Jinhui Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03792)  

**Abstract**: To address the modality learning degeneration caused by modality imbalance, existing multimodal learning~(MML) approaches primarily attempt to balance the optimization process of each modality from the perspective of model learning. However, almost all existing methods ignore the modality imbalance caused by unimodal data sampling, i.e., equal unimodal data sampling often results in discrepancies in informational content, leading to modality imbalance. Therefore, in this paper, we propose a novel MML approach called \underline{D}ata-aware \underline{U}nimodal \underline{S}ampling~(\method), which aims to dynamically alleviate the modality imbalance caused by sampling. Specifically, we first propose a novel cumulative modality discrepancy to monitor the multimodal learning process. Based on the learning status, we propose a heuristic and a reinforcement learning~(RL)-based data-aware unimodal sampling approaches to adaptively determine the quantity of sampled data at each iteration, thus alleviating the modality imbalance from the perspective of sampling. Meanwhile, our method can be seamlessly incorporated into almost all existing multimodal learning approaches as a plugin. Experiments demonstrate that \method~can achieve the best performance by comparing with diverse state-of-the-art~(SOTA) baselines. 

**Abstract (ZH)**: 数据感知单模态采样以缓解模态不平衡的多模态学习方法 

---
# Positive-Unlabeled Diffusion Models for Preventing Sensitive Data Generation 

**Title (ZH)**: 正 unlabeled 扩散模型防止敏感数据生成 

**Authors**: Hiroshi Takahashi, Tomoharu Iwata, Atsutoshi Kumagai, Yuuki Yamanaka, Tomoya Yamashita  

**Link**: [PDF](https://arxiv.org/pdf/2503.03789)  

**Abstract**: Diffusion models are powerful generative models but often generate sensitive data that are unwanted by users, mainly because the unlabeled training data frequently contain such sensitive data. Since labeling all sensitive data in the large-scale unlabeled training data is impractical, we address this problem by using a small amount of labeled sensitive data. In this paper, we propose positive-unlabeled diffusion models, which prevent the generation of sensitive data using unlabeled and sensitive data. Our approach can approximate the evidence lower bound (ELBO) for normal (negative) data using only unlabeled and sensitive (positive) data. Therefore, even without labeled normal data, we can maximize the ELBO for normal data and minimize it for labeled sensitive data, ensuring the generation of only normal data. Through experiments across various datasets and settings, we demonstrated that our approach can prevent the generation of sensitive images without compromising image quality. 

**Abstract (ZH)**: 正 unlabeled 扩散模型：使用未标记的敏感数据防止生成敏感数据 

---
# Sarcasm Detection as a Catalyst: Improving Stance Detection with Cross-Target Capabilities 

**Title (ZH)**: 讽刺检测作为催化剂：跨目标能力提升立场检测 

**Authors**: Gibson Nkhata Shi Yin Hong, Susan Gauch  

**Link**: [PDF](https://arxiv.org/pdf/2503.03787)  

**Abstract**: Stance Detection (SD) has become a critical area of interest due to its applications in various contexts leading to increased research within NLP. Yet the subtlety and complexity of texts sourced from online platforms often containing sarcastic language pose significant challenges for SD algorithms in accurately determining the authors stance. This paper addresses this by employing sarcasm for SD. It also tackles the issue of insufficient annotated data for training SD models on new targets by conducting Cross-Target SD (CTSD). The proposed approach involves fine-tuning BERT and RoBERTa models followed by concatenating additional deep learning layers. The approach is assessed against various State-Of-The-Art baselines for SD demonstrating superior performance using publicly available datasets. Notably our model outperforms the best SOTA models on both in-domain SD and CTSD tasks even before the incorporation of sarcasm-detection pre-training. The integration of sarcasm knowledge into the model significantly reduces misclassifications of sarcastic text elements in SD allowing our model to accurately predict 85% of texts that were previously misclassified without sarcasm-detection pre-training on in-domain SD. This enhancement contributes to an increase in the models average macro F1-score. The CTSD task achieves performance comparable to that of the in-domain task despite using a zero-shot finetuning. We also reveal that the success of the transfer-learning framework relies on the correlation between the lexical attributes of sarcasm detection and SD. This study represents the first exploration of sarcasm detection as an intermediate transfer-learning task within the context of SD while also leveraging the concatenation of BERT or RoBERTa with other deep-learning techniques. The proposed approach establishes a foundational baseline for future research in this domain. 

**Abstract (ZH)**: 立场检测（SD）已成为一个重要研究领域，由于其在各种上下文中的应用，导致NLP领域内的研究不断增加。然而，来源于在线平台的文本往往包含讽刺语言，这给SD算法准确判断作者立场带来了挑战。本文通过引入讽刺进行SD，同时通过跨目标立场检测（CTSD）解决了训练SD模型时标注数据不足的问题。提出的方法包括微调BERT和RoBERTa模型，并附加额外的深度学习层。该方法在各种最新基准方法上进行了评估，展示了在公开数据集上的优越性能。值得注意的是，即使在未引入讽刺检测预训练的情况下，我们的模型在领域内SD和CTSD任务上均优于最新模型的最佳表现。将讽刺知识整合到模型中显著减少了SD中讽刺文本元素的误分类，使得我们的模型在未进行讽刺检测预训练的情况下准确预测了85%的被误分类的文本。这一增强提高了模型的平均宏F1分数。尽管使用零样本微调，CTSD任务仍可达到与领域内任务相似的性能。我们还揭示了转移学习框架成功依赖于讽刺检测和SD之间词法属性的相关性。本文是首个在SD背景下将讽刺检测作为中间转移学习任务进行探索的研究，还利用了BERT或RoBERTa与其他深度学习技术的串联方法。提出的方案为该领域的未来研究奠定了基础。 

---
# Passive Heart Rate Monitoring During Smartphone Use in Everyday Life 

**Title (ZH)**: 日常生活中使用智能手机时的被动心率监测 

**Authors**: Shun Liao, Paolo Di Achille, Jiang Wu, Silviu Borac, Jonathan Wang, Xin Liu, Eric Teasley, Lawrence Cai, Yun Liu, Daniel McDuff, Hao-Wei Su, Brent Winslow, Anupam Pathak, Shwetak Patel, Jameson K. Rogers, Ming-Zher Poh  

**Link**: [PDF](https://arxiv.org/pdf/2503.03783)  

**Abstract**: Resting heart rate (RHR) is an important biomarker of cardiovascular health and mortality, but tracking it longitudinally generally requires a wearable device, limiting its availability. We present PHRM, a deep learning system for passive heart rate (HR) and RHR measurements during everyday smartphone use, using facial video-based photoplethysmography. Our system was developed using 225,773 videos from 495 participants and validated on 185,970 videos from 205 participants in laboratory and free-living conditions, representing the largest validation study of its kind. Compared to reference electrocardiogram, PHRM achieved a mean absolute percentage error (MAPE) < 10% for HR measurements across three skin tone groups of light, medium and dark pigmentation; MAPE for each skin tone group was non-inferior versus the others. Daily RHR measured by PHRM had a mean absolute error < 5 bpm compared to a wearable HR tracker, and was associated with known risk factors. These results highlight the potential of smartphones to enable passive and equitable heart health monitoring. 

**Abstract (ZH)**: 被动心率测量的深学习系统：基于面部视频的光体积描记术在日常智能手机使用中的心率和静息心率监测 

---
# Accelerating Focal Search in Multi-Agent Path Finding with Tighter Lower Bounds 

**Title (ZH)**: 基于更紧的下界加速多智能体路径规划中的焦点搜索 

**Authors**: Yimin Tang, Zhenghong Yu, Jiaoyang Li, Sven Koenig  

**Link**: [PDF](https://arxiv.org/pdf/2503.03779)  

**Abstract**: Multi-Agent Path Finding (MAPF) involves finding collision-free paths for multiple agents while minimizing a cost function--an NP-hard problem. Bounded suboptimal methods like Enhanced Conflict-Based Search (ECBS) and Explicit Estimation CBS (EECBS) balance solution quality with computational efficiency using focal search mechanisms. While effective, traditional focal search faces a limitation: the lower bound (LB) value determining which nodes enter the FOCAL list often increases slowly in early search stages, resulting in a constrained search space that delays finding valid solutions. In this paper, we propose a novel bounded suboptimal algorithm, double-ECBS (DECBS), to address this issue by first determining the maximum LB value and then employing a best-first search guided by this LB to find a collision-free path. Experimental results demonstrate that DECBS outperforms ECBS in most test cases and is compatible with existing optimization techniques. DECBS can reduce nearly 30% high-level CT nodes and 50% low-level focal search nodes. When agent density is moderate to high, DECBS achieves a 23.5% average runtime improvement over ECBS with identical suboptimality bounds and optimizations. 

**Abstract (ZH)**: 多智能体路径规划中的双ECBS算法：一种新的有界次优方法 

---
# FlexInfer: Breaking Memory Constraint via Flexible and Efficient Offloading for On-Device LLM Inference 

**Title (ZH)**: FlexInfer: 超越内存约束的柔性高效卸载以实现设备上大语言模型推理 

**Authors**: Hongchao Du, Shangyu Wu, Arina Kharlamova, Nan Guan, Chun Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2503.03777)  

**Abstract**: Large Language Models (LLMs) face challenges for on-device inference due to high memory demands. Traditional methods to reduce memory usage often compromise performance and lack adaptability. We propose FlexInfer, an optimized offloading framework for on-device inference, addressing these issues with techniques like asynchronous prefetching, balanced memory locking, and flexible tensor preservation. These strategies enhance memory efficiency and mitigate I/O bottlenecks, ensuring high performance within user-specified resource constraints. Experiments demonstrate that FlexInfer significantly improves throughput under limited resources, achieving up to 12.5 times better performance than existing methods and facilitating the deployment of large models on resource-constrained devices. 

**Abstract (ZH)**: FlexInfer：一种优化的设备端推理卸载框架 

---
# BotUmc: An Uncertainty-Aware Twitter Bot Detection with Multi-view Causal Inference 

**Title (ZH)**: BotUmc：一种基于多视图因果推理的不确定性aware推特机器人检测方法 

**Authors**: Tao Yang, Yang Hu, Feihong Lu, Ziwei Zhang, Qingyun Sun, Jianxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.03775)  

**Abstract**: Social bots have become widely known by users of social platforms. To prevent social bots from spreading harmful speech, many novel bot detections are proposed. However, with the evolution of social bots, detection methods struggle to give high-confidence answers for samples. This motivates us to quantify the uncertainty of the outputs, informing the confidence of the results. Therefore, we propose an uncertainty-aware bot detection method to inform the confidence and use the uncertainty score to pick a high-confidence decision from multiple views of a social network under different environments. Specifically, our proposed BotUmc uses LLM to extract information from tweets. Then, we construct a graph based on the extracted information, the original user information, and the user relationship and generate multiple views of the graph by causal interference. Lastly, an uncertainty loss is used to force the model to quantify the uncertainty of results and select the result with low uncertainty in one view as the final decision. Extensive experiments show the superiority of our method. 

**Abstract (ZH)**: 社交机器人检测方法中考虑不确定性以提高决策信心 

---
# Efficient Finetuning for Dimensional Speech Emotion Recognition in the Age of Transformers 

**Title (ZH)**: 面向Transformer时代的高效细调维度语音情感识别 

**Authors**: Aneesha Sampath, James Tavernor, Emily Mower Provost  

**Link**: [PDF](https://arxiv.org/pdf/2503.03756)  

**Abstract**: Accurate speech emotion recognition is essential for developing human-facing systems. Recent advancements have included finetuning large, pretrained transformer models like Wav2Vec 2.0. However, the finetuning process requires substantial computational resources, including high-memory GPUs and significant processing time. As the demand for accurate emotion recognition continues to grow, efficient finetuning approaches are needed to reduce the computational burden. Our study focuses on dimensional emotion recognition, predicting attributes such as activation (calm to excited) and valence (negative to positive). We present various finetuning techniques, including full finetuning, partial finetuning of transformer layers, finetuning with mixed precision, partial finetuning with caching, and low-rank adaptation (LoRA) on the Wav2Vec 2.0 base model. We find that partial finetuning with mixed precision achieves performance comparable to full finetuning while increasing training speed by 67%. Caching intermediate representations further boosts efficiency, yielding an 88% speedup and a 71% reduction in learnable parameters. We recommend finetuning the final three transformer layers in mixed precision to balance performance and training efficiency, and adding intermediate representation caching for optimal speed with minimal performance trade-offs. These findings lower the barriers to finetuning speech emotion recognition systems, making accurate emotion recognition more accessible to a broader range of researchers and practitioners. 

**Abstract (ZH)**: 准确的语音情感识别对于开发面向人类的系统至关重要。近期进展包括对预训练变换器模型如Wav2Vec 2.0进行微调。然而，微调过程需要大量的计算资源，包括高性能GPU和大量的处理时间。随着对准确情感识别需求的不断增长，需要高效的微调方法以减轻计算负担。本研究侧重于维度情感识别，预测如激活（平静到兴奋）和价值（负向到正向）等属性。我们提出了多种微调技术，包括全程微调、变压器层的部分微调、混合精度微调、带有缓存的部分微调以及低秩适应（LoRA）在Wav2Vec 2.0基模型上的应用。我们发现，部分微调并使用混合精度可以达到与全程微调相当的性能，同时将训练速度提高67%。进一步使用中间表示的缓存进一步提高了效率，使训练速度提升88%，并且使可学习参数减少了71%。我们建议在混合精度下微调最后三层变压器层以平衡性能和训练效率，并通过添加中间表示的缓存来实现最佳速度，同时将性能降低控制在最小范围内。这些发现降低了语音情感识别系统微调的门槛，使准确的情感识别对更广泛的科研人员和实践者更加可及。 

---
# Generative Diffusion Model-based Compression of MIMO CSI 

**Title (ZH)**: 基于生成扩散模型的MIMO CSI压缩 

**Authors**: Heasung Kim, Taekyun Lee, Hyeji Kim, Gustavo De Veciana, Mohamed Amine Arfaoui, Asil Koc, Phil Pietraski, Guodong Zhang, John Kaewell  

**Link**: [PDF](https://arxiv.org/pdf/2503.03753)  

**Abstract**: While neural lossy compression techniques have markedly advanced the efficiency of Channel State Information (CSI) compression and reconstruction for feedback in MIMO communications, efficient algorithms for more challenging and practical tasks-such as CSI compression for future channel prediction and reconstruction with relevant side information-remain underexplored, often resulting in suboptimal performance when existing methods are extended to these scenarios. To that end, we propose a novel framework for compression with side information, featuring an encoding process with fixed-rate compression using a trainable codebook for codeword quantization, and a decoding procedure modeled as a backward diffusion process conditioned on both the codeword and the side information. Experimental results show that our method significantly outperforms existing CSI compression algorithms, often yielding over twofold performance improvement by achieving comparable distortion at less than half the data rate of competing methods in certain scenarios. These findings underscore the potential of diffusion-based compression for practical deployment in communication systems. 

**Abstract (ZH)**: 一种基于侧信息的扩散压缩框架：CSI压缩与重建在未来信道预测中的应用 

---
# Multimodal AI predicts clinical outcomes of drug combinations from preclinical data 

**Title (ZH)**: 多模态AI从预临床数据预测药物组合的临床结果 

**Authors**: Yepeng Huang, Xiaorui Su, Varun Ullanat, Ivy Liang, Lindsay Clegg, Damilola Olabode, Nicholas Ho, Bino John, Megan Gibbs, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2503.02781)  

**Abstract**: Predicting clinical outcomes from preclinical data is essential for identifying safe and effective drug combinations. Current models rely on structural or target-based features to identify high-efficacy, low-toxicity drug combinations. However, these approaches fail to incorporate the multimodal data necessary for accurate, clinically-relevant predictions. Here, we introduce MADRIGAL, a multimodal AI model that learns from structural, pathway, cell viability, and transcriptomic data to predict drug combination effects across 953 clinical outcomes and 21842 compounds, including combinations of approved drugs and novel compounds in development. MADRIGAL uses a transformer bottleneck module to unify preclinical drug data modalities while handling missing data during training and inference--a major challenge in multimodal learning. It outperforms single-modality methods and state-of-the-art models in predicting adverse drug interactions. MADRIGAL performs virtual screening of anticancer drug combinations and supports polypharmacy management for type II diabetes and metabolic dysfunction-associated steatohepatitis (MASH). It identifies transporter-mediated drug interactions. MADRIGAL predicts resmetirom, the first and only FDA-approved drug for MASH, among therapies with the most favorable safety profile. It supports personalized cancer therapy by integrating genomic profiles from cancer patients. Using primary acute myeloid leukemia samples and patient-derived xenograft models, it predicts the efficacy of personalized drug combinations. Integrating MADRIGAL with a large language model allows users to describe clinical outcomes in natural language, improving safety assessment by identifying potential adverse interactions and toxicity risks. MADRIGAL provides a multimodal approach for designing combination therapies with improved predictive accuracy and clinical relevance. 

**Abstract (ZH)**: 从预临床数据预测临床结果对于识别安全有效的药物组合至关重要。现有的模型依赖于结构或靶标特征来识别高效低毒的药物组合。然而，这些方法未能 Incorporate 准确且临床相关的多模态数据预测所需的信息。在这里，我们介绍了 MADRIGAL，这是一种多模态AI模型，它从结构、途径、细胞活力和转录组数据中学习，以预测跨越 953 临床结果和 21842 种化合物（包括已批准药物和正在开发的新型化合物）的药物组合效果。MADRIGAL 使用变压器瓶颈模块在训练和推理过程中统一预临床药物数据模态，解决了多模态学习中的一个重要挑战。它在预测不良药物相互作用方面超过了单模态方法和最先进的模型。MADRIGAL 用于抗癌药物组合的虚拟筛选，并支持 II 型糖尿病和代谢功能障碍相关性脂肪肝炎（MASH）的多药治疗管理。它识别转运体介导的药物相互作用。MADRIGAL 预测了 FDA 批准的第一个也是唯一的用于 MASH 的药物 resmetirom，其安全特性最佳。它通过整合癌症患者的基因组特征支持个性化癌症治疗。使用急性髓系白血病的原代样本和患者衍生的异种移植物模型，它预测个性化药物组合的有效性。将 MADRIGAL 与大型语言模型结合使用可以让用户以自然语言描述临床结果，通过识别潜在的不良相互作用和毒性风险提高安全性评估。MADRIGAL 提供了一种多模态方法，以提高预测准确性和临床相关性来设计组合疗法。 

---
# M2-omni: Advancing Omni-MLLM for Comprehensive Modality Support with Competitive Performance 

**Title (ZH)**: M2-omni: 推动全面模态支持的竞争力性能 omnibig语言模型 

**Authors**: Qingpei Guo, Kaiyou Song, Zipeng Feng, Ziping Ma, Qinglong Zhang, Sirui Gao, Xuzheng Yu, Yunxiao Sun, Tai-WeiChang, Jingdong Chen, Ming Yang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.18778)  

**Abstract**: We present M2-omni, a cutting-edge, open-source omni-MLLM that achieves competitive performance to GPT-4o. M2-omni employs a unified multimodal sequence modeling framework, which empowers Large Language Models(LLMs) to acquire comprehensive cross-modal understanding and generation capabilities. Specifically, M2-omni can process arbitrary combinations of audio, video, image, and text modalities as input, generating multimodal sequences interleaving with audio, image, or text outputs, thereby enabling an advanced and interactive real-time experience. The training of such an omni-MLLM is challenged by significant disparities in data quantity and convergence rates across modalities. To address these challenges, we propose a step balance strategy during pre-training to handle the quantity disparities in modality-specific data. Additionally, a dynamically adaptive balance strategy is introduced during the instruction tuning stage to synchronize the modality-wise training progress, ensuring optimal convergence. Notably, we prioritize preserving strong performance on pure text tasks to maintain the robustness of M2-omni's language understanding capability throughout the training process. To our best knowledge, M2-omni is currently a very competitive open-source model to GPT-4o, characterized by its comprehensive modality and task support, as well as its exceptional performance. We expect M2-omni will advance the development of omni-MLLMs, thus facilitating future research in this domain. 

**Abstract (ZH)**: M2-omni：一种与GPT-4o性能相当的开源全模态大语言模型 

---
# BIOSCAN-5M: A Multimodal Dataset for Insect Biodiversity 

**Title (ZH)**: BIOSCAN-5M：一种多模态昆虫生物多样性数据集 

**Authors**: Zahra Gharaee, Scott C. Lowe, ZeMing Gong, Pablo Millan Arias, Nicholas Pellegrino, Austin T. Wang, Joakim Bruslund Haurum, Iuliia Zarubiieva, Lila Kari, Dirk Steinke, Graham W. Taylor, Paul Fieguth, Angel X. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2406.12723)  

**Abstract**: As part of an ongoing worldwide effort to comprehend and monitor insect biodiversity, this paper presents the BIOSCAN-5M Insect dataset to the machine learning community and establish several benchmark tasks. BIOSCAN-5M is a comprehensive dataset containing multi-modal information for over 5 million insect specimens, and it significantly expands existing image-based biological datasets by including taxonomic labels, raw nucleotide barcode sequences, assigned barcode index numbers, geographical, and size information. We propose three benchmark experiments to demonstrate the impact of the multi-modal data types on the classification and clustering accuracy. First, we pretrain a masked language model on the DNA barcode sequences of the BIOSCAN-5M dataset, and demonstrate the impact of using this large reference library on species- and genus-level classification performance. Second, we propose a zero-shot transfer learning task applied to images and DNA barcodes to cluster feature embeddings obtained from self-supervised learning, to investigate whether meaningful clusters can be derived from these representation embeddings. Third, we benchmark multi-modality by performing contrastive learning on DNA barcodes, image data, and taxonomic information. This yields a general shared embedding space enabling taxonomic classification using multiple types of information and modalities. The code repository of the BIOSCAN-5M Insect dataset is available at this https URL. 

**Abstract (ZH)**: BIOSCAN-5M昆虫数据集及其在机器学习领域的基准任务 

---
