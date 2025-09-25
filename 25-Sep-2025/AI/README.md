# Scan-do Attitude: Towards Autonomous CT Protocol Management using a Large Language Model Agent 

**Title (ZH)**: 基于Scan-do态度的大型语言模型代理在自主CT协议管理中的探索 

**Authors**: Xingjian Kang, Linda Vorberg, Andreas Maier, Alexander Katzmann, Oliver Taubmann  

**Link**: [PDF](https://arxiv.org/pdf/2509.20270)  

**Abstract**: Managing scan protocols in Computed Tomography (CT), which includes adjusting acquisition parameters or configuring reconstructions, as well as selecting postprocessing tools in a patient-specific manner, is time-consuming and requires clinical as well as technical expertise. At the same time, we observe an increasing shortage of skilled workforce in radiology. To address this issue, a Large Language Model (LLM)-based agent framework is proposed to assist with the interpretation and execution of protocol configuration requests given in natural language or a structured, device-independent format, aiming to improve the workflow efficiency and reduce technologists' workload. The agent combines in-context-learning, instruction-following, and structured toolcalling abilities to identify relevant protocol elements and apply accurate modifications. In a systematic evaluation, experimental results indicate that the agent can effectively retrieve protocol components, generate device compatible protocol definition files, and faithfully implement user requests. Despite demonstrating feasibility in principle, the approach faces limitations regarding syntactic and semantic validity due to lack of a unified device API, and challenges with ambiguous or complex requests. In summary, the findings show a clear path towards LLM-based agents for supporting scan protocol management in CT imaging. 

**Abstract (ZH)**: 基于大型语言模型的代理框架在管理计算机断层扫描（CT）扫描协议中的应用：从自然语言或结构化、设备无关格式的协议配置请求中辅助解析和执行，以提高工作流程效率并减轻技术人员的负担。 

---
# Design Insights and Comparative Evaluation of a Hardware-Based Cooperative Perception Architecture for Lane Change Prediction 

**Title (ZH)**: 基于硬件的合作感知架构在变道预测中的设计洞察与比较评价 

**Authors**: Mohamed Manzour, Catherine M. Elias, Omar M. Shehata, Rubén Izquierdo, Miguel Ángel Sotelo  

**Link**: [PDF](https://arxiv.org/pdf/2509.20218)  

**Abstract**: Research on lane change prediction has gained attention in the last few years. Most existing works in this area have been conducted in simulation environments or with pre-recorded datasets, these works often rely on simplified assumptions about sensing, communication, and traffic behavior that do not always hold in practice. Real-world deployments of lane-change prediction systems are relatively rare, and when they are reported, the practical challenges, limitations, and lessons learned are often under-documented. This study explores cooperative lane-change prediction through a real hardware deployment in mixed traffic and shares the insights that emerged during implementation and testing. We highlight the practical challenges we faced, including bottlenecks, reliability issues, and operational constraints that shaped the behavior of the system. By documenting these experiences, the study provides guidance for others working on similar pipelines. 

**Abstract (ZH)**: 基于真实硬件在混合交通中探索协作变道预测及其实施经验 

---
# Federation of Agents: A Semantics-Aware Communication Fabric for Large-Scale Agentic AI 

**Title (ZH)**: 代理联邦：一种面向大规模代理型AI的意义感知通信架构 

**Authors**: Lorenzo Giusti, Ole Anton Werner, Riccardo Taiello, Matilde Carvalho Costa, Emre Tosun, Andrea Protani, Marc Molina, Rodrigo Lopes de Almeida, Paolo Cacace, Diogo Reis Santos, Luigi Serio  

**Link**: [PDF](https://arxiv.org/pdf/2509.20175)  

**Abstract**: We present Federation of Agents (FoA), a distributed orchestration framework that transforms static multi-agent coordination into dynamic, capability-driven collaboration. FoA introduces Versioned Capability Vectors (VCVs): machine-readable profiles that make agent capabilities searchable through semantic embeddings, enabling agents to advertise their capabilities, cost, and limitations. Our aarchitecturecombines three key innovations: (1) semantic routing that matches tasks to agents over sharded HNSW indices while enforcing operational constraints through cost-biased optimization, (2) dynamic task decomposition where compatible agents collaboratively break down complex tasks into DAGs of subtasks through consensus-based merging, and (3) smart clustering that groups agents working on similar subtasks into collaborative channels for k-round refinement before synthesis. Built on top of MQTT,s publish-subscribe semantics for scalable message passing, FoA achieves sub-linear complexity through hierarchical capability matching and efficient index maintenance. Evaluation on HealthBench shows 13x improvements over single-model baselines, with clustering-enhanced laboration particularly effective for complex reasoning tasks requiring multiple perspectives. The system scales horizontally while maintaining consistent performance, demonstrating that semantic orchestration with structured collaboration can unlock the collective intelligence of heterogeneous federations of AI agents. 

**Abstract (ZH)**: 联邦代理体框架：动态、能力驱动的合作 

---
# Formal Verification of Minimax Algorithms 

**Title (ZH)**: 最小化最大化算法的形式验证 

**Authors**: Wieger Wesselink, Kees Huizing, Huub van de Wetering  

**Link**: [PDF](https://arxiv.org/pdf/2509.20138)  

**Abstract**: Using the Dafny verification system, we formally verify a range of minimax search algorithms, including variations with alpha-beta pruning and transposition tables. For depth-limited search with transposition tables, we introduce a witness-based correctness criterion and apply it to two representative algorithms. All verification artifacts, including proofs and Python implementations, are publicly available. 

**Abstract (ZH)**: 使用Dafny验证系统，我们形式化验证了一系列极小极大搜索算法，包括带有α-β剪枝和置换表的变体。对于带有置换表的深度受限搜索，我们引入了一种基于证人的正确性标准，并将其应用于两种代表性算法。所有验证成果，包括证明和Python实现，均已公开。 

---
# PEPS: Quantum-Inspired Reinforcement Learning for Coherent Reasoning Traces in LLMs 

**Title (ZH)**: PEPS：量子启发的强化学习在LLMs中的相干推理跟踪 

**Authors**: Venkat Margapuri, Garik Kazanjian, Naren Kosaraju  

**Link**: [PDF](https://arxiv.org/pdf/2509.20105)  

**Abstract**: Large Language Models (LLMs) often struggle with maintaining coherent multi-step reasoning traces, particularly in tasks that require a structured logical flow. This work introduces a quantum-inspired approach to address the challenge by incorporating a fidelity-based reward derived from Projected Entangled Pair States (PEPS) into Proximal Policy Optimization. Unlike prior approaches that use direct supervision or contrastive objectives, the proposed method guides learning through structural consistency, offering a novel approach to enforce global coherence in generated reasoning traces. The proposed framework is evaluated using multiple coherence-determining metrics on diverse datasets such as GSM8K, StrategyQA, and EntailmentBank spanning arithmetic, intuitive, and entailment-based reasoning. Results show that the proposed quantum-inspired approach offers significant improvements over supervised, contrastive, and pretrained baseline approaches, highlighting the effectiveness of quantum-inspired fidelity as a foundation to improve reasoning trace coherence in LLMs. 

**Abstract (ZH)**: 量子启发的大语言模型多步推理连贯性提升方法 

---
# Steerable Adversarial Scenario Generation through Test-Time Preference Alignment 

**Title (ZH)**: 基于测试时偏好对齐的可引导对抗场景生成 

**Authors**: Tong Nie, Yuewen Mei, Yihong Tang, Junlin He, Jie Sun, Haotian Shi, Wei Ma, Jian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.20102)  

**Abstract**: Adversarial scenario generation is a cost-effective approach for safety assessment of autonomous driving systems. However, existing methods are often constrained to a single, fixed trade-off between competing objectives such as adversariality and realism. This yields behavior-specific models that cannot be steered at inference time, lacking the efficiency and flexibility to generate tailored scenarios for diverse training and testing requirements. In view of this, we reframe the task of adversarial scenario generation as a multi-objective preference alignment problem and introduce a new framework named \textbf{S}teerable \textbf{A}dversarial scenario \textbf{GE}nerator (SAGE). SAGE enables fine-grained test-time control over the trade-off between adversariality and realism without any retraining. We first propose hierarchical group-based preference optimization, a data-efficient offline alignment method that learns to balance competing objectives by decoupling hard feasibility constraints from soft preferences. Instead of training a fixed model, SAGE fine-tunes two experts on opposing preferences and constructs a continuous spectrum of policies at inference time by linearly interpolating their weights. We provide theoretical justification for this framework through the lens of linear mode connectivity. Extensive experiments demonstrate that SAGE not only generates scenarios with a superior balance of adversariality and realism but also enables more effective closed-loop training of driving policies. Project page: this https URL. 

**Abstract (ZH)**: 对抗场景生成是自主驾驶系统安全性评估的一项成本效益高的方法。然而，现有方法往往局限于在抗对比性和现实性之间单一直接的权衡中。这会导致行为特定的模型，在推理时无法调整，缺乏生成适应多样化训练和测试需求的定制化场景的效率和灵活性。为了解决这一问题，我们将对抗场景生成任务重新定义为一个多目标偏好对齐问题，并提出了一种名为SAGE（Steerable Adversarial Scenario GEnerator）的新框架。SAGE允许在推理时对抗性和现实性之间的权衡进行细腻的控制，无需重新训练。我们首先提出了分层组别偏好优化方法，这是一种数据高效的离线对齐方法，通过解耦硬可行性约束与软偏好来学习平衡竞争目标。SAGE在推理时通过线性插值两个专家的权重来构建连续的策略谱，而不是训练固定模型。我们通过线性模式可连接性的视角提供了该框架的理论依据。大量实验表明，SAGE不仅生成具有优越对抗性和现实性平衡的场景，还能促进驾驶策略的更有效闭环训练。项目页面：this https URL。 

---
# From Pheromones to Policies: Reinforcement Learning for Engineered Biological Swarms 

**Title (ZH)**: 从信息素到策略：工程化生物群 Swarm 的强化学习 

**Authors**: Aymeric Vellinger, Nemanja Antonic, Elio Tuci  

**Link**: [PDF](https://arxiv.org/pdf/2509.20095)  

**Abstract**: Swarm intelligence emerges from decentralised interactions among simple agents, enabling collective problem-solving. This study establishes a theoretical equivalence between pheromone-mediated aggregation in \celeg\ and reinforcement learning (RL), demonstrating how stigmergic signals function as distributed reward mechanisms. We model engineered nematode swarms performing foraging tasks, showing that pheromone dynamics mathematically mirror cross-learning updates, a fundamental RL algorithm. Experimental validation with data from literature confirms that our model accurately replicates empirical \celeg\ foraging patterns under static conditions. In dynamic environments, persistent pheromone trails create positive feedback loops that hinder adaptation by locking swarms into obsolete choices. Through computational experiments in multi-armed bandit scenarios, we reveal that introducing a minority of exploratory agents insensitive to pheromones restores collective plasticity, enabling rapid task switching. This behavioural heterogeneity balances exploration-exploitation trade-offs, implementing swarm-level extinction of outdated strategies. Our results demonstrate that stigmergic systems inherently encode distributed RL processes, where environmental signals act as external memory for collective credit assignment. By bridging synthetic biology with swarm robotics, this work advances programmable living systems capable of resilient decision-making in volatile environments. 

**Abstract (ZH)**: swarm智能源自简单代理之间去中心化的互动，使其能够集体解决问题。本研究建立了C. elegans的化学信号聚集与强化学习之间的理论等价性，展示了标记信号作为分布式奖励机制的运作方式。我们构建了工程化的线虫群进行觅食任务的模型，表明化学信号动态与交叉学习更新具有数学上的相似性，后者是基础的强化学习算法。通过文献数据的实验验证，证明我们的模型在静态条件下能够准确复制实测的C. elegans觅食模式。在动态环境中，持续的化学生命痕迹会产生正反馈循环，阻碍适应性，使群落固定在过时的选择上。通过在多臂老虎机情景下的计算实验，我们发现引入少数不受化学信号影响的探索代理，可以恢复群体的可塑性，使其能够快速切换任务。这种行为异质性平衡了探索与利用的权衡，实现了群体层级对过时策略的淘汰。我们的结果表明，标记信号系统本质上编码了分布式强化学习过程，其中环境信号作为群体共同信用分配的外部记忆。通过将合成生物学与 swarm 机器人学相结合，本工作推进了能够在多变环境中实现稳健决策的可编程生物系统。 

---
# MACD: Multi-Agent Clinical Diagnosis with Self-Learned Knowledge for LLM 

**Title (ZH)**: MACD: 多智能体临床诊断结合自学习知识 

**Authors**: Wenliang Li, Rui Yan, Xu Zhang, Li Chen, Hongji Zhu, Jing Zhao, Junjun Li, Mengru Li, Wei Cao, Zihang Jiang, Wei Wei, Kun Zhang, Shaohua Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.20067)  

**Abstract**: Large language models (LLMs) have demonstrated notable potential in medical applications, yet they face substantial challenges in handling complex real-world clinical diagnoses using conventional prompting methods. Current prompt engineering and multi-agent approaches typically optimize isolated inferences, neglecting the accumulation of reusable clinical experience. To address this, this study proposes a novel Multi-Agent Clinical Diagnosis (MACD) framework, which allows LLMs to self-learn clinical knowledge via a multi-agent pipeline that summarizes, refines, and applies diagnostic insights. It mirrors how physicians develop expertise through experience, enabling more focused and accurate diagnosis on key disease-specific cues. We further extend it to a MACD-human collaborative workflow, where multiple LLM-based diagnostician agents engage in iterative consultations, supported by an evaluator agent and human oversight for cases where agreement is not reached. Evaluated on 4,390 real-world patient cases across seven diseases using diverse open-source LLMs (Llama-3.1 8B/70B, DeepSeek-R1-Distill-Llama 70B), MACD significantly improves primary diagnostic accuracy, outperforming established clinical guidelines with gains up to 22.3% (MACD). On the subset of the data, it achieves performance on par with or exceeding that of human physicians (up to 16% improvement over physicians-only diagnosis). Additionally, on the MACD-human workflow, it achieves an 18.6% improvement compared to physicians-only diagnosis. Moreover, self-learned knowledge exhibits strong cross-model stability, transferability, and model-specific personalization, while the system can generate traceable rationales, enhancing explainability. Consequently, this work presents a scalable self-learning paradigm for LLM-assisted diagnosis, bridging the gap between the intrinsic knowledge of LLMs and real-world clinical practice. 

**Abstract (ZH)**: 大型语言模型（LLMs）在医疗应用中展现了显著潜力，但在使用常规提示方法处理复杂临床诊断时面临诸多挑战。当前的提示工程和多代理方法通常仅优化孤立的推理，忽视了临床经验的积累。为解决这一问题，本研究提出了一种新的多代理临床诊断（MACD）框架，该框架通过多代理流水线总结、精炼和应用诊断见解，使LLMs能够自我学习临床知识。该框架模拟了医生通过经验发展专业技能的过程，从而实现对特定疾病线索的更集中和准确诊断。此外，我们将其扩展到一种MACD-人类协作工作流，在该工作流中，多个基于LLM的诊断代理进行迭代咨询，并通过评价代理和人类监督的支持，在无法达成一致的情况下进行干预。该工作流在涵盖七种疾病的4,390个真实患者案例中，使用多样化的开源LLM（Llama-3.1 8B/70B、DeepSeek-R1-Distill-Llama 70B）进行了评估，MACD显著提高了初步诊断准确性，优于现有临床指南，增幅最高达22.3%（MACD）。在数据子集上，其性能与或超过人类医生的诊断（最高16%的提升）。此外，MACD-人类工作流在与人类医生单独诊断相比时，实现了18.6%的提升。同时，自我学习的知识表现出强大的跨模型稳定性和迁移性，以及模型特定的个性化，系统还能够生成可追溯的理由，增强可解释性。因此，本研究提出了一个可扩展的LLM辅助诊断自我学习范式，弥合了LLM内在知识与实际临床实践之间的差距。 

---
# Embodied AI: From LLMs to World Models 

**Title (ZH)**: 具身AI：从大规模语言模型到世界模型 

**Authors**: Tongtong Feng, Xin Wang, Yu-Gang Jiang, Wenwu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20021)  

**Abstract**: Embodied Artificial Intelligence (AI) is an intelligent system paradigm for achieving Artificial General Intelligence (AGI), serving as the cornerstone for various applications and driving the evolution from cyberspace to physical systems. Recent breakthroughs in Large Language Models (LLMs) and World Models (WMs) have drawn significant attention for embodied AI. On the one hand, LLMs empower embodied AI via semantic reasoning and task decomposition, bringing high-level natural language instructions and low-level natural language actions into embodied cognition. On the other hand, WMs empower embodied AI by building internal representations and future predictions of the external world, facilitating physical law-compliant embodied interactions. As such, this paper comprehensively explores the literature in embodied AI from basics to advances, covering both LLM driven and WM driven works. In particular, we first present the history, key technologies, key components, and hardware systems of embodied AI, as well as discuss its development via looking from unimodal to multimodal angle. We then scrutinize the two burgeoning fields of embodied AI, i.e., embodied AI with LLMs/multimodal LLMs (MLLMs) and embodied AI with WMs, meticulously delineating their indispensable roles in end-to-end embodied cognition and physical laws-driven embodied interactions. Building upon the above advances, we further share our insights on the necessity of the joint MLLM-WM driven embodied AI architecture, shedding light on its profound significance in enabling complex tasks within physical worlds. In addition, we examine representative applications of embodied AI, demonstrating its wide applicability in real-world scenarios. Last but not least, we point out future research directions of embodied AI that deserve further investigation. 

**Abstract (ZH)**: 具身人工智能：实现人工通用智能的核心 paradigms 及其从虚拟空间到物理系统的演变：大型语言模型与世界模型的推动作用和未来发展展望 

---
# CON-QA: Privacy-Preserving QA using cloud LLMs in Contract Domain 

**Title (ZH)**: CON-QA：合同领域基于云LLM的隐私保护问答 

**Authors**: Ajeet Kumar Singh, Rajsabi Surya, Anurag Tripathi, Santanu Choudhury, Sudhir Bisane  

**Link**: [PDF](https://arxiv.org/pdf/2509.19925)  

**Abstract**: As enterprises increasingly integrate cloud-based large language models (LLMs) such as ChatGPT and Gemini into their legal document workflows, protecting sensitive contractual information - including Personally Identifiable Information (PII) and commercially sensitive clauses - has emerged as a critical challenge. In this work, we propose CON-QA, a hybrid privacy-preserving framework designed specifically for secure question answering over enterprise contracts, effectively combining local and cloud-hosted LLMs. The CON-QA framework operates through three stages: (i) semantic query decomposition and query-aware document chunk retrieval using a locally deployed LLM analysis, (ii) anonymization of detected sensitive entities via a structured one-to-many mapping scheme, ensuring semantic coherence while preventing cross-session entity inference attacks, and (iii) anonymized response generation by a cloud-based LLM, with accurate reconstruction of the original answer locally using a session-consistent many-to-one reverse mapping. To rigorously evaluate CON-QA, we introduce CUAD-QA, a corpus of 85k question-answer pairs generated over 510 real-world CUAD contract documents, encompassing simple, complex, and summarization-style queries. Empirical evaluations, complemented by detailed human assessments, confirm that CON-QA effectively maintains both privacy and utility, preserves answer quality, maintains fidelity to legal clause semantics, and significantly mitigates privacy risks, demonstrating its practical suitability for secure, enterprise-level contract documents. 

**Abstract (ZH)**: 企业越来越多地将基于云的大语言模型（LLMs）如ChatGPT和Gemini集成到其法律文件工作流程中，保护敏感合同信息——包括个人可识别信息（PII）和商业敏感条款——已成为一个重要挑战。为此，我们提出了CON-QA，一种专门设计用于企业合约安全问答的混合隐私保护框架，有效结合了本地和云托管的LLMs。CON-QA框架通过三个阶段运行：（i）使用本地部署的LLM分析进行语义查询分解和基于查询的文档片段检索，（ii）通过结构化的多对一映射方案匿名化检测到的敏感实体，确保语义一致性同时防止会话间实体推断攻击，以及（iii）由云托管的LLM生成匿名化回复，并使用会话一致的多对一逆向映射在当地准确重构原始答案。为严格评估CON-QA，我们引入了CUAD-QA，一个包含85,000个问题-答案对的语料库，这些对是在510份真实世界的CUAD合同文件上生成的，涵盖简单、复杂和总结性查询。实验评估结合详细的manual评估，证实了CON-QA既有效地保持了隐私和实用性，又保持了答案质量，维护了法律条款语义的准确性和显著降低了隐私风险，展示了其在安全的企业级合同文件中的实际适用性。 

---
# LatentGuard: Controllable Latent Steering for Robust Refusal of Attacks and Reliable Response Generation 

**Title (ZH)**: LatentGuard: 可控潜在空间引导以实现稳健的攻击拒绝和可靠的响应生成 

**Authors**: Huizhen Shu, Xuying Li, Zhuo Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19839)  

**Abstract**: Achieving robust safety alignment in large language models (LLMs) while preserving their utility remains a fundamental challenge. Existing approaches often struggle to balance comprehensive safety with fine-grained controllability at the representation level. We introduce LATENTGUARD, a novel three-stage framework that combines behavioral alignment with supervised latent space control for interpretable and precise safety steering. Our approach begins by fine-tuning an LLM on rationalized datasets containing both reasoning-enhanced refusal responses to adversarial prompts and reasoning-enhanced normal responses to benign queries, establishing robust behavioral priors across both safety-critical and utility-preserving scenarios. We then train a structured variational autoencoder (VAE) on intermediate MLP activations, supervised by multi-label annotations including attack types, attack methods, and benign indicators. This supervision enables the VAE to learn disentangled latent representations that capture distinct adversarial characteristics while maintaining semantic interpretability. Through targeted manipulation of learned latent dimensions, LATENTGUARD achieves selective refusal behavior, effectively blocking harmful requests while preserving helpfulness for legitimate use cases. Experiments on Qwen3-8B demonstrate significant improvements in both safety controllability and response interpretability without compromising utility. Cross-architecture validation on Mistral-7B confirms the generalizability of our latent steering approach, showing consistent effectiveness across different model families. Our results suggest that structured representation-level intervention offers a promising pathway toward building safer yet practical LLM systems. 

**Abstract (ZH)**: 在保持实用性的同时实现大型语言模型（LLMs）的稳健安全对齐仍然是一个基本挑战。现有的方法往往难以在全面的安全性和细微的表示级可控性之间取得平衡。我们引入了LATENTGUARD，一个新颖的三阶段框架，结合了行为对齐和监督潜在空间控制，以实现可解释和精确的安全引导。我们的方法首先在包含推理增强的拒绝响应（针对对抗性提示）和推理增强的正常响应（针对良性查询）的精算数据集上微调LLM，从而在安全关键和实用性保留两种场景下建立稳健的行为先验。然后，我们基于包含攻击类型、攻击方法和良性指示的多标签注释训练结构化变分自编码器（VAE），以中间MLP激活为监督目标。这种监督使VAE能够学习解耦的潜在表示，捕捉不同的对抗性特征同时保持语义可解释性。通过目标干预学习到的潜在维度，LATENTGUARD实现了选择性的拒绝行为，有效地阻止有害请求的同时保留对合法用例的帮助性。在Qwen3-8B上的实验表明，在不牺牲实用性的情况下，安全可控性和响应可解释性显著提高。跨架构验证在Mistral-7B上证实了我们潜在引导方法的普遍适用性，显示出不同模型家族中一致的有效性。我们的结果表明，结构化的表示级干预为构建更安全且实用的LLM系统提供了有前景的道路。 

---
# Analysis of approximate linear programming solution to Markov decision problem with log barrier function 

**Title (ZH)**: 基于对数障碍函数的马尔可夫决策问题近似线性规划求解分析 

**Authors**: Donghwan Lee, Hyukjun Yang, Bum Geun Park  

**Link**: [PDF](https://arxiv.org/pdf/2509.19800)  

**Abstract**: There are two primary approaches to solving Markov decision problems (MDPs): dynamic programming based on the Bellman equation and linear programming (LP). Dynamic programming methods are the most widely used and form the foundation of both classical and modern reinforcement learning (RL). By contrast, LP-based methods have been less commonly employed, although they have recently gained attention in contexts such as offline RL. The relative underuse of the LP-based methods stems from the fact that it leads to an inequality-constrained optimization problem, which is generally more challenging to solve effectively compared with Bellman-equation-based methods. The purpose of this paper is to establish a theoretical foundation for solving LP-based MDPs in a more effective and practical manner. Our key idea is to leverage the log-barrier function, widely used in inequality-constrained optimization, to transform the LP formulation of the MDP into an unconstrained optimization problem. This reformulation enables approximate solutions to be obtained easily via gradient descent. While the method may appear simple, to the best of our knowledge, a thorough theoretical interpretation of this approach has not yet been developed. This paper aims to bridge this gap. 

**Abstract (ZH)**: 基于线性规划的方法求解马尔可夫决策过程的理论基础 

---
# Agentic Metacognition: Designing a "Self-Aware" Low-Code Agent for Failure Prediction and Human Handoff 

**Title (ZH)**: 代理元认知：设计一种“自我意识”低代码代理进行故障预测和人工交接 

**Authors**: Jiexi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.19783)  

**Abstract**: The inherent non-deterministic nature of autonomous agents, particularly within low-code/no-code (LCNC) environments, presents significant reliability challenges. Agents can become trapped in unforeseen loops, generate inaccurate outputs, or encounter unrecoverable failures, leading to user frustration and a breakdown of trust. This report proposes a novel architectural pattern to address these issues: the integration of a secondary, "metacognitive" layer that actively monitors the primary LCNC agent. Inspired by human introspection, this layer is designed to predict impending task failures based on a defined set of triggers, such as excessive latency or repetitive actions. Upon predicting a failure, the metacognitive agent proactively initiates a human handoff, providing the user with a clear summary of the agent's "thought process" and a detailed explanation of why it could not proceed. An empirical analysis of a prototype system demonstrates that this approach significantly increases the overall task success rate. However, this performance gain comes with a notable increase in computational overhead. The findings reframe human handoffs not as an admission of defeat but as a core design feature that enhances system resilience, improves user experience, and builds trust by providing transparency into the agent's internal state. The report discusses the practical and ethical implications of this approach and identifies key directions for future research. 

**Abstract (ZH)**: 自主代理的固有非确定性性质，特别是在低代码/无代码（LCNC）环境中，提出了显著的可靠性挑战。代理可能会陷入未预见的循环、生成不准确的输出，或遇到无法恢复的故障，导致用户 frustration 和信任崩溃。本报告提出了一种新的架构模式来解决这些问题：在主要的LCNC代理中集成一个次级的、“元认知”的层，该层主动监控主要代理。受人类反省的启发，该层设计为基于一定触发条件预测即将发生的任务失败，如过度延迟或重复动作。在预测到失败时，元认知代理主动发起人工干预，向用户提供代理“思维过程”的清晰总结和详细解释，说明它为何无法继续执行。对一个原型系统的实证分析表明，这种方法显著提高了整体任务的成功率。然而，这种性能增益伴随着计算开销的显著增加。研究结果重新定义了人工干预不仅仅是一种失败的承认，而是增强系统韧性的核心设计特征，改善用户体验，并通过提供代理内部状态的透明度来建立信任。报告讨论了这一方法的实践和伦理影响，并指出了未来研究的关键方向。 

---
# The Conductor and the Engine: A Path Towards Co-Designed Reasoning 

**Title (ZH)**: 指挥者与引擎：协设计推理之路 

**Authors**: Yuanxin Wang, Pawel Filipczuk, Anisha Garg, Amaan Dhada, Mohammad Hassanpour, David Bick, Ganesh Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2509.19762)  

**Abstract**: Modern LLM reasoning relies on extensive test-time computation, driven by internal model training and external agentic orchestration. However, this synergy is often inefficient, as model verbosity and poor instruction following lead to wasted compute. We analyze this capability-cost trade-off and introduce an optimized reasoning workflow (\cepo) that empowers smaller open-source models to outperform models multiple times their size. We will open-source this workflow to enable further research. Our work demonstrates a clear path toward co-designing orchestration frameworks with the underlying model capabilities to unlock powerful reasoning in small-to-medium sized models. 

**Abstract (ZH)**: 现代大语言模型推理依赖于大量的测试时计算，由内部模型训练和外部代理 orchestration 驱动。然而，这种协同作用往往效率低下，因为模型的冗长和拙劣的指令跟随导致了计算资源的浪费。我们分析了这种能力与成本之间的权衡，并引入了一种优化的推理工作流（\cepo），使较小的开源模型能够超越其大小多倍的模型。我们将开源这一工作流以促进进一步的研究。我们的工作展示了如何协同设计 orchestrations 框架与底层模型能力，以解锁中小型模型的强大推理能力。 

---
# UserRL: Training Interactive User-Centric Agent via Reinforcement Learning 

**Title (ZH)**: UserRL: 通过强化学习训练以用户为中心的交互智能体 

**Authors**: Cheng Qian, Zuxin Liu, Akshara Prabhakar, Jielin Qiu, Zhiwei Liu, Haolin Chen, Shirley Kokane, Heng Ji, Weiran Yao, Shelby Heinecke, Silvio Savarese, Caiming Xiong, Huan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19736)  

**Abstract**: Reinforcement learning (RL) has shown promise in training agentic models that move beyond static benchmarks to engage in dynamic, multi-turn interactions. Yet, the ultimate value of such agents lies in their ability to assist users, a setting where diversity and dynamics of user interaction pose challenges. In this work, we propose UserRL, a unified framework for training and evaluating user-centric abilities through standardized gym environments paired with simulated users. We systematically vary turn-level reward assignment and trajectory-level score calculation to analyze how different formulations affect learning under the GRPO algorithm. Our experiments across Qwen3 models reveal three key findings: (i) SFT cold start is critical for unlocking initial interaction ability and enabling sustained RL improvements; (ii) deliberate trajectory scoring yields more efficient and effective multi-turn interactions; and (iii) while stronger simulated users (e.g., GPT-4o) facilitates training, open-source simulators (e.g., Qwen3-32B) remain a cost-effective and transferable option. Together, these results highlight that careful design of reward shaping and user simulation choice is as crucial as model scale, and establish UserRL as a practical pathway for developing robust user-centric agentic models. All codes and data are public for future research. 

**Abstract (ZH)**: 强化学习（RL）在训练能够超越静态基准并参与动态多轮交互的模型方面展现了潜力。然而，这类代理的价值在于其辅助用户的能力，而在这种用户交互多样性和动态性的背景下，这一任务面临着挑战。在本工作中，我们提出了一种名为UserRL的统一框架，通过标准化的健身环境配以模拟用户，来训练和评估以用户为中心的能力。我们系统地变化了回合级奖励分配和轨迹级评分计算方法，分析了不同形式如何影响GRPO算法下的学习。我们的实验结果揭示了三点关键发现：(i) 自动化建模初始交互能力对于解锁初始交互能力和持续的RL改进至关重要；(ii) 计划性轨迹评分能更有效地促进多轮交互；(iii) 虽然更强的模拟用户（如GPT-4o）有利于训练，开源模拟器（如Qwen3-32B）仍然是一个成本效益高且可转移的选择。这些结果表明，奖励塑造和用户模拟设计的精心规划与模型规模同样重要，并确立了UserRL作为一种实用路径，用于开发稳健的以用户为中心的代理模型。所有代码和数据均已公开，供未来研究使用。 

---
# Calibrated Reasoning: An Explanatory Verifier for Dynamic and Efficient Problem-Solving 

**Title (ZH)**: 校准推理：一种解释性验证器，用于动态高效的问题解决 

**Authors**: Anisha Garg, Engin Tekin, Yash More, David Bick, Nishit Neema, Ganesh Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2509.19681)  

**Abstract**: Advanced test-time computing strategies are essential for scaling reasoning models, but their effectiveness is capped by the models' poor self-evaluation. We propose a pairwise Explanatory Verifier, trained via reinforcement learning (GRPO), that produces calibrated confidence scores and associated natural language reasoning for generated solutions. Our verifier improves the accuracy and efficiency of test-time strategies like best-of-n and self-reflection. Crucially, it excels at identifying challenging failure modes, such as when both candidate solutions are identically incorrect, succeeding where standard methods like majority voting fail. 

**Abstract (ZH)**: 先进的测试时计算策略对于扩展推理模型至关重要，但其效果受限于模型的 poor self-evaluation。我们提出了一个基于强化学习（GRPO）训练的成对解释验证器，该验证器生成校准的信任分数和相应的自然语言推理，以支持生成的解决方案。该验证器提高了像 best-of-n 和自我反思这类测试时策略的准确性和效率。最关键的是，它在识别具有挑战性的失败模式（例如，当两个候选解决方案都错误时）上表现更佳，而标准方法如多数投票则会失效。 

---
# SteinerSQL: Graph-Guided Mathematical Reasoning for Text-to-SQL Generation 

**Title (ZH)**: SteinerSQL: 图引导的数学推理在文本到SQL生成中的应用 

**Authors**: Xutao Mao, Tao Liu, Hongying Zan  

**Link**: [PDF](https://arxiv.org/pdf/2509.19623)  

**Abstract**: Large Language Models (LLMs) struggle with complex Text-to-SQL queries that demand both sophisticated mathematical reasoning and intricate schema navigation. Existing methods often tackle these challenges in isolation, creating a fractured reasoning process that compromises logical and structural correctness. To resolve this, we introduce SteinerSQL, a framework that unifies these dual challenges into a single, graph-centric optimization problem. SteinerSQL operates in three stages: mathematical decomposition to identify required tables (terminals), optimal reasoning scaffold construction via a Steiner tree problem, and multi-level validation to ensure correctness. On the challenging LogicCat and Spider2.0-Lite benchmarks, SteinerSQL establishes a new state-of-the-art with 36.10% and 40.04% execution accuracy, respectively, using Gemini-2.5-Pro. Beyond accuracy, SteinerSQL presents a new, unified paradigm for Text-to-SQL, paving the way for more robust and principled solutions to complex reasoning tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在处理既需要复杂的数学推理又需要精细的模式导航的Text-to-SQL查询时存在困难。现有方法往往将这些挑战孤立地解决，导致推理过程断层，影响逻辑和结构的正确性。为解决这一问题，我们提出SteinerSQL框架，将这些双重挑战统一为一个基于图的优化问题。SteinerSQL分三个阶段工作：数学分解以识别所需的表格（终端）、通过Steiner树问题构建最优推理框架，并进行多级验证以确保正确性。在具有挑战性的LogicCat和Spider2.0-Lite基准测试中，SteinerSQL分别使用Gemini-2.5-Pro实现了36.10%和40.04%的执行准确性。不仅如此，SteinerSQL还提出了一种新的统一Text-to-SQL范式，为更 robust 和原则性的复杂推理任务解决方案铺平了道路。 

---
# What Does Your Benchmark Really Measure? A Framework for Robust Inference of AI Capabilities 

**Title (ZH)**: 你的基准究竟衡量了什么？一种用于人工智能能力稳健推断的框架 

**Authors**: Nathanael Jo, Ashia Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2509.19590)  

**Abstract**: Evaluations of generative models on benchmark data are now ubiquitous, and their outcomes critically shape public and scientific expectations of AI's capabilities. Yet growing skepticism surrounds their reliability. How can we know that a reported accuracy genuinely reflects a model's true performance? Evaluations are often presented as simple measurements, but in reality they are inferences: to treat benchmark scores as evidence of capability is already to assume a theory of what capability is and how it manifests in a test. We make this step explicit by proposing a principled framework for evaluation as inference: begin from a theory of capability, and then derive methods for estimating it. This perspective, familiar in fields such as psychometrics, has not yet become commonplace in AI evaluation. As a proof of concept, we address a central challenge that undermines reliability: sensitivity to perturbations. After formulating a model of ability, we introduce methods that infer ability while accounting for uncertainty from sensitivity and finite samples, including an adaptive algorithm that significantly reduces sample complexity. Together, these contributions lay the groundwork for more reliable and trustworthy estimates of AI capabilities as measured through benchmarks. 

**Abstract (ZH)**: 基于基准数据的生成模型评估已无处不在，其结果对AI能力的公众和科学期望至关重要。然而，对其可靠性的怀疑日益加剧。我们如何知道报告的准确率真正反映了模型的真实性能？评估往往被简单呈现为测量，但实际上它们是推断：将基准分数视为能力证据的前提是对能力的理论及其在测试中表现形式的假设。我们通过提出一个规范的评估框架作为推断的方法，使之明确化：从能力的理论出发，然后推导出估计能力的方法。这种观点在心理测量学等领域已相当常见，但在AI评估中尚未普及。作为概念验证，我们针对影响可靠性的核心挑战——对扰动的敏感性——进行了研究。在构建能力模型后，我们引入了一种方法，在考虑敏感性和有限样本的不确定性的同时推断能力，并提出了一种自适应算法，显著降低了样本复杂性。这些贡献为通过基准测量AI能力的更可靠和可信赖的估计奠定了基础。 

---
# Nano Bio-Agents (NBA): Small Language Model Agents for Genomics 

**Title (ZH)**: 纳米生物剂（NBA）：基因组学的小语言模型代理 

**Authors**: George Hong, Daniel Trejo Banos  

**Link**: [PDF](https://arxiv.org/pdf/2509.19566)  

**Abstract**: We investigate the application of Small Language Models (<10 billion parameters) for genomics question answering via agentic framework to address hallucination issues and computational cost challenges. The Nano Bio-Agent (NBA) framework we implemented incorporates task decomposition, tool orchestration, and API access into well-established systems such as NCBI and AlphaGenome. Results show that SLMs combined with such agentic framework can achieve comparable and in many cases superior performance versus existing approaches utilising larger models, with our best model-agent combination achieving 98% accuracy on the GeneTuring benchmark. Notably, small 3-10B parameter models consistently achieve 85-97% accuracy while requiring much lower computational resources than conventional approaches. This demonstrates promising potential for efficiency gains, cost savings, and democratization of ML-powered genomics tools while retaining highly robust and accurate performance. 

**Abstract (ZH)**: 我们探讨了采用代理框架的小型语言模型（<100亿参数）在基因组学问答中的应用，以解决幻觉问题和计算成本挑战。我们实现的Nano Bio-Agent (NBA) 框架将任务分解、工具编排和API访问集成到如NCBI和AlphaGenome等成熟系统中。结果显示，结合此类代理框架的小型语言模型在基准测试中能实现与现有使用更大模型的方法相当甚至更优的表现，我们最好的模型-代理组合在GeneTuring基准测试中的准确率达到98%。值得注意的是，3-100亿参数的小型模型在保持高准确率的同时，所需的计算资源远低于传统方法，这显示了通过机器学习增强的基因组学工具在提高效率、降低成本并实现民主化方面的潜在优势，同时保持了高度可靠和准确的性能。 

---
# Score the Steps, Not Just the Goal: VLM-Based Subgoal Evaluation for Robotic Manipulation 

**Title (ZH)**: 评分步骤，而不仅目标：基于VLM的机器人 manipulation子目标评估 

**Authors**: Ramy ElMallah, Krish Chhajer, Chi-Guhn Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.19524)  

**Abstract**: Robot learning papers typically report a single binary success rate (SR), which obscures where a policy succeeds or fails along a multi-step manipulation task. We argue that subgoal-level reporting should become routine: for each trajectory, a vector of per-subgoal SRs that makes partial competence visible (e.g., grasp vs. pour). We propose a blueprint for StepEval, a cost-aware plug-in evaluation framework that utilizes vision-language models (VLMs) as automated judges of subgoal outcomes from recorded images or videos. Rather than proposing new benchmarks or APIs, our contribution is to outline design principles for a scalable, community-driven open-source project. In StepEval, the primary artifact for policy evaluation is the per-subgoal SR vector; however, other quantities (e.g., latency or cost estimates) are also considered for framework-optimization diagnostics to help the community tune evaluation efficiency and accuracy when ground-truth subgoal success labels are available. We discuss how such a framework can remain model-agnostic, support single- or multi-view inputs, and be lightweight enough to adopt across labs. The intended contribution is a shared direction: a minimal, extensible seed that invites open-source contributions, so that scoring the steps, not just the final goal, becomes a standard and reproducible practice. 

**Abstract (ZH)**: 机器人学习论文通常报告单个二元成功率（SR），这掩盖了沿多步操控任务中策略成功或失败的具体位置。我们argue应将子目标级别报告变为常规做法：对于每个轨迹，提供一个子目标级别SR向量，使部分能力可见（例如，抓取 vs 倾倒）。我们提出了一种StepEval的成本感知插件评估框架蓝图，该框架利用视觉-语言模型（VLMs）作为从记录的图像或视频中自动评估子目标结果的裁判。我们贡献在于概述了一个可扩展的、社区驱动的开源项目的德规范设计原则，而非提出新的基准或API。在StepEval中，策略评估的主要结果是子目标级别的SR向量；然而，其他量（例如，延迟或成本估算）也考虑用于框架优化诊断，以帮助社区在可用真实子目标成功标签时调优评估效率和准确性。我们讨论了此类框架如何保持模型无关性、支持单视角或多视角输入，并足够轻量以跨越实验室采用。我们旨在提供一个共享方向：一个最小化且可扩展的基础，邀请开源贡献，使得评分不仅限于最终目标，而是成为标准和可复现的做法。 

---
# Cognitive Load Limits in Large Language Models: Benchmarking Multi-Hop Reasoning 

**Title (ZH)**: 大型语言模型的认知负载限制：多跳推理基准测试 

**Authors**: Sai Teja Reddy Adapala  

**Link**: [PDF](https://arxiv.org/pdf/2509.19517)  

**Abstract**: The scaling of Large Language Models (LLMs) has exposed a critical gap between their performance on static benchmarks and their fragility in dynamic, information-rich environments. While models excel at isolated tasks, the computational limits that govern their reasoning under cognitive load remain poorly understood. In this work, we introduce a formal theory of computational cognitive load, positing that extraneous, task-irrelevant information (Context Saturation) and interference from task-switching (Attentional Residue) are key mechanisms that degrade performance. We designed the Interleaved Cognitive Evaluation (ICE), a deconfounded benchmark to systematically manipulate these load factors on challenging multi-hop reasoning tasks. A comprehensive study (N = 10 replications per item across 200 questions) revealed significant performance variations across five instruction-tuned models. Smaller open-source architectures (Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2) exhibited baseline brittleness, achieving 0% accuracy (SEM = 0.0) across all conditions, including clean controls, on this high-intrinsic-load task. In contrast, Gemini-2.0-Flash-001 showed partial resilience, achieving 85% accuracy in control conditions, with a statistically significant degradation under context saturation ($\beta = -0.003$ per % load, $p < 0.001$). These findings provide preliminary evidence that cognitive load is a key contributor to reasoning failures, supporting theories of hallucination-as-guessing under uncertainty. We conclude that dynamic, cognitive-aware stress testing, as exemplified by the ICE benchmark, is essential for evaluating the true resilience and safety of advanced AI systems. 

**Abstract (ZH)**: 大型语言模型的扩展揭示了其在静态基准上的性能与其在动态、信息丰富环境中的脆弱性之间的重要差距。尽管模型在孤立任务中表现优异，但影响其认知负载下推理的计算限制依然难以理解。在本项工作中，我们提出了计算认知负担的正式理论，认为与任务无关的多余信息（上下文饱和）和任务切换引发的干扰（注意残留）是导致性能下降的关键机制。我们设计了交织认知评估（ICE），一个去偏差基准，用于系统地在具有挑战性的多跳推理任务中操纵这些负担因素。一项全面的研究（每项题目在200个问题上进行10次复制）揭示了五种指令微调模型之间显著的性能差异。较小的开源架构（Llama-3-8B-Instruct、Mistral-7B-Instruct-v0.2）展现出基础的脆弱性，在此高内生负担任务的所有条件下，包括干净的对照组中，准确率为0%（SEM = 0.0）。相比之下，Gemini-2.0-Flash-001显示出部分韧性，在对照条件下准确率为85%，并在上下文饱和条件下统计显著下降（β = -0.003 每百分比负载，p < 0.001）。这些发现初步表明认知负担是推理失败的关键因素，支持在不确定性下幻觉即猜测的理论。我们得出结论，动态的认知感知压力测试，如由ICE基准所展示，对于评估高级AI系统的真正韧性和安全性至关重要。 

---
# Estimating the Self-Consistency of LLMs 

**Title (ZH)**: 估计LLM的自一致性 

**Authors**: Robert Nowak  

**Link**: [PDF](https://arxiv.org/pdf/2509.19489)  

**Abstract**: Systems often repeat the same prompt to large language models (LLMs) and aggregate responses to improve reliability. This short note analyzes an estimator of the self-consistency of LLMs and the tradeoffs it induces under a fixed compute budget $B=mn$, where $m$ is the number of prompts sampled from the task distribution and $n$ is the number of repeated LLM calls per prompt; the resulting analysis favors a rough split $m,n\propto\sqrt{B}$. 

**Abstract (ZH)**: 系统经常对大型语言模型（LLMs）重复相同的提示以提高响应的一致性，并聚合响应以提高可靠性。本简要笔记分析了在固定计算预算 \(B=mn\) 下LLMs的自我一致性估计器及其诱导的权衡，其中 \(m\) 是从任务分布中采样的提示数量，\(n\) 是每个提示的重复LLM调用次数；由此得出的分析倾向于粗略的分配 \(m,n \propto \sqrt{B}\)。 

---
# Evaluation-Aware Reinforcement Learning 

**Title (ZH)**: 评估导向的强化学习 

**Authors**: Shripad Vilasrao Deshmukh, Will Schwarzer, Scott Niekum  

**Link**: [PDF](https://arxiv.org/pdf/2509.19464)  

**Abstract**: Policy evaluation is often a prerequisite for deploying safety- and performance-critical systems. Existing evaluation approaches frequently suffer from high variance due to limited data and long-horizon tasks, or high bias due to unequal support or inaccurate environmental models. We posit that these challenges arise, in part, from the standard reinforcement learning (RL) paradigm of policy learning without explicit consideration of evaluation. As an alternative, we propose evaluation-aware reinforcement learning (EvA-RL), in which a policy is trained to maximize expected return while simultaneously minimizing expected evaluation error under a given value prediction scheme -- in other words, being "easy" to evaluate. We formalize a framework for EvA-RL and design an instantiation that enables accurate policy evaluation, conditioned on a small number of rollouts in an assessment environment that can be different than the deployment environment. However, our theoretical analysis and empirical results show that there is often a tradeoff between evaluation accuracy and policy performance when using a fixed value-prediction scheme within EvA-RL. To mitigate this tradeoff, we extend our approach to co-learn an assessment-conditioned state-value predictor alongside the policy. Empirical results across diverse discrete and continuous action domains demonstrate that EvA-RL can substantially reduce evaluation error while maintaining competitive returns. This work lays the foundation for a broad new class of RL methods that treat reliable evaluation as a first-class principle during training. 

**Abstract (ZH)**: 评价意识强化学习：在评价约束下最大化回报 

---
# The Indispensable Role of User Simulation in the Pursuit of AGI 

**Title (ZH)**: 用户模拟在追求AGI过程中的不可或缺作用 

**Authors**: Krisztian Balog, ChengXiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2509.19456)  

**Abstract**: Progress toward Artificial General Intelligence (AGI) faces significant bottlenecks, particularly in rigorously evaluating complex interactive systems and acquiring the vast interaction data needed for training adaptive agents. This paper posits that user simulation -- creating computational agents that mimic human interaction with AI systems -- is not merely a useful tool, but is a critical catalyst required to overcome these bottlenecks and accelerate AGI development. We argue that realistic simulators provide the necessary environments for scalable evaluation, data generation for interactive learning, and fostering the adaptive capabilities central to AGI. Therefore, research into user simulation technology and intelligent task agents are deeply synergistic and must advance hand-in-hand. This article elaborates on the critical role of user simulation for AGI, explores the interdisciplinary nature of building realistic simulators, identifies key challenges including those posed by large language models, and proposes a future research agenda. 

**Abstract (ZH)**: 人工智能通用智能（AGI）进展面临显著瓶颈，特别是在严谨评估复杂交互系统和获取用于训练适应性代理的大量交互数据方面。本文认为，用户仿真——创建模仿人类与AI系统交互的计算代理——不仅是有用的工具，更是克服这些瓶颈和加速AGI开发的关键催化剂。我们主张，真实的仿真器提供了必要的环境，用于可扩展的评估、交互学习所需的数据生成以及培养AGI的核心适应能力。因此，用户仿真技术研究与智能任务代理的研究深度互补，必须同步进行。本文阐述了用户仿真在AGI中的关键作用，探讨了构建真实仿真器的跨学科性质，识别了包括大型语言模型在内的关键挑战，并提出未来研究议程。 

---
# EmbeddingGemma: Powerful and Lightweight Text Representations 

**Title (ZH)**: EmbeddingGemma: 强大且轻量级的文本表示 

**Authors**: Henrique Schechter Vera, Sahil Dua, Biao Zhang, Daniel Salz, Ryan Mullins, Sindhu Raghuram Panyam, Sara Smoot, Iftekhar Naim, Joe Zou, Feiyang Chen, Daniel Cer, Alice Lisak, Min Choi, Lucas Gonzalez, Omar Sanseviero, Glenn Cameron, Ian Ballantyne, Kat Black, Kaifeng Chen, Weiyi Wang, Zhe Li, Gus Martins, Jinhyuk Lee, Mark Sherwood, Juyeong Ji, Renjie Wu, Jingxiao Zheng, Jyotinder Singh, Abheesht Sharma, Divya Sreepat, Aashi Jain, Adham Elarabawy, AJ Co, Andreas Doumanoglou, Babak Samari, Ben Hora, Brian Potetz, Dahun Kim, Enrique Alfonseca, Fedor Moiseev, Feng Han, Frank Palma Gomez, Gustavo Hernández Ábrego, Hesen Zhang, Hui Hui, Jay Han, Karan Gill, Ke Chen, Koert Chen, Madhuri Shanbhogue, Michael Boratko, Paul Suganthan, Sai Meher Karthik Duddu, Sandeep Mariserla, Setareh Ariafar, Shanfeng Zhang, Shijie Zhang, Simon Baumgartner, Sonam Goenka, Steve Qiu, Tanmaya Dabral, Trevor Walker, Vikram Rao, Waleed Khawaja, Wenlei Zhou, Xiaoqi Ren, Ye Xia, Yichang Chen, Yi-Ting Chen, Zhe Dong, Zhongli Ding, Francesco Visin, Gaël Liu, Jiageng Zhang, Kathleen Kenealy, Michelle Casbon, Ravin Kumar, Thomas Mesnard, Zach Gleicher, Cormac Brick, Olivier Lacombe, Adam Roberts, Yunhsuan Sung, Raphael Hoffmann, Tris Warkentin, Armand Joulin, Tom Duerig, Mojtaba Seyedhosseini  

**Link**: [PDF](https://arxiv.org/pdf/2509.20354)  

**Abstract**: We introduce EmbeddingGemma, a new lightweight, open text embedding model based on the Gemma 3 language model family. Our innovative training recipe strategically captures knowledge from larger models via encoder-decoder initialization and geometric embedding distillation. We improve model robustness and expressiveness with a spread-out regularizer, and ensure generalizability by merging checkpoints from varied, optimized mixtures. Evaluated on the Massive Text Embedding Benchmark (MTEB) across multilingual, English, and code domains, EmbeddingGemma (300M) achieves state-of-the-art results. Notably, it outperforms prior top models, both proprietary and open, with fewer than 500M parameters, and provides performance comparable to models double its size, offering an exceptional performance-to-cost ratio. Remarkably, this lead persists when quantizing model weights or truncating embedding outputs. This makes EmbeddingGemma particularly well-suited for low-latency and high-throughput use cases such as on-device applications. We provide ablation studies exploring our key design choices. We release EmbeddingGemma to the community to promote further research. 

**Abstract (ZH)**: 我们介绍了EmbeddingGemma，这是一种基于Gemma 3语言模型家族的新型轻量级开放文本嵌入模型。我们的创新训练方案通过编码器-解码器初始化和几何嵌入蒸馏战略性地从更大规模的模型中捕获知识。通过使用分布型正则化改进模型的鲁棒性和表现力，并通过合并不同优化混合模型的检查点来确保通用性。在跨多语言、英语和代码领域的巨量文本嵌入基准测试（MTEB）上，EmbeddingGemma（300M）取得了最先进成果。值得注意的是，它使用不到500M参数优于先前的顶级模型，并提供了与两倍规模模型相当的性能，具有卓越的性能与成本比。令人惊讶的是，即使在量化模型权重或截断嵌入输出时，这一优势仍然保持。这使得EmbeddingGemma特别适合低延迟和高吞吐量的应用场景，如本地设备应用。我们进行了消融研究以探索我们关键设计选择。我们向社区发布了EmbeddingGemma，以促进进一步研究。 

---
# Morphological Synthesizer for Ge'ez Language: Addressing Morphological Complexity and Resource Limitations 

**Title (ZH)**: _ge’ez语言形态合成器：应对形态复杂性和资源限制_ 

**Authors**: Gebrearegawi Gebremariam, Hailay Teklehaymanot, Gebregewergs Mezgebe  

**Link**: [PDF](https://arxiv.org/pdf/2509.20341)  

**Abstract**: Ge'ez is an ancient Semitic language renowned for its unique alphabet. It serves as the script for numerous languages, including Tigrinya and Amharic, and played a pivotal role in Ethiopia's cultural and religious development during the Aksumite kingdom era. Ge'ez remains significant as a liturgical language in Ethiopia and Eritrea, with much of the national identity documentation recorded in Ge'ez. These written materials are invaluable primary sources for studying Ethiopian and Eritrean philosophy, creativity, knowledge, and civilization. Ge'ez has a complex morphological structure with rich inflectional and derivational morphology, and no usable NLP has been developed and published until now due to the scarcity of annotated linguistic data, corpora, labeled datasets, and lexicons. Therefore, we propose a rule-based Ge'ez morphological synthesizer to generate surface words from root words according to the morphological structures of the language. We used 1,102 sample verbs, representing all verb morphological structures, to test and evaluate the system. The system achieves a performance of 97.4%, outperforming the baseline model and suggesting that future work should build a comprehensive system considering morphological variations of the language.
Keywords: Ge'ez, NLP, morphology, morphological synthesizer, rule-based 

**Abstract (ZH)**: 吉泽语是一种以独特字母闻名的古老闪米特语言。它作为提格雷尼亚语和阿姆哈拉语等语言的书写系统，在阿克苏姆王国时代对埃塞俄比亚的文化和宗教发展起到了关键作用。吉泽语在埃塞俄利亚和厄立特里亚作为礼拜语言仍具有重要意义，大量的国家身份文档都以吉泽语记录。这些书面材料是研究埃塞俄利亚和厄立特里亚哲学、创意、知识和文明的宝贵原始资料。吉泽语具有复杂的形态结构，丰富的屈折和派生形态学，但由于缺乏注释的语料库、标注数据集和词汇表，至今未开发出可用的NLP系统。因此，我们提出了一种基于规则的吉泽语形态合成器，根据语言的形态结构生成表层词汇。我们使用了1102个代表性动词样本，涵盖了所有动词的形态结构，对该系统进行了测试和评估。系统性能达到了97.4%，超过了基线模型，表明未来工作应在考虑语言形态变化的基础上构建全面的系统。 

---
# Adaptive Event-Triggered Policy Gradient for Multi-Agent Reinforcement Learning 

**Title (ZH)**: 自适应事件触发策略梯度在多智能体强化学习中的应用 

**Authors**: Umer Siddique, Abhinav Sinha, Yongcan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2509.20338)  

**Abstract**: Conventional multi-agent reinforcement learning (MARL) methods rely on time-triggered execution, where agents sample and communicate actions at fixed intervals. This approach is often computationally expensive and communication-intensive. To address this limitation, we propose ET-MAPG (Event-Triggered Multi-Agent Policy Gradient reinforcement learning), a framework that jointly learns an agent's control policy and its event-triggering policy. Unlike prior work that decouples these mechanisms, ET-MAPG integrates them into a unified learning process, enabling agents to learn not only what action to take but also when to execute it. For scenarios with inter-agent communication, we introduce AET-MAPG, an attention-based variant that leverages a self-attention mechanism to learn selective communication patterns. AET-MAPG empowers agents to determine not only when to trigger an action but also with whom to communicate and what information to exchange, thereby optimizing coordination. Both methods can be integrated with any policy gradient MARL algorithm. Extensive experiments across diverse MARL benchmarks demonstrate that our approaches achieve performance comparable to state-of-the-art, time-triggered baselines while significantly reducing both computational load and communication overhead. 

**Abstract (ZH)**: 基于事件触发的多智能体策略梯度强化学习（ET-MAPG）及其注意力变体（AET-MAPG） 

---
# Uncovering Graph Reasoning in Decoder-only Transformers with Circuit Tracing 

**Title (ZH)**: 基于电路追踪揭示解码器-only变压器中的图推理 

**Authors**: Xinnan Dai, Chung-Hsiang Lo, Kai Guo, Shenglai Zeng, Dongsheng Luo, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20336)  

**Abstract**: Transformer-based LLMs demonstrate strong performance on graph reasoning tasks, yet their internal mechanisms remain underexplored. To uncover these reasoning process mechanisms in a fundamental and unified view, we set the basic decoder-only transformers and explain them using the circuit-tracer framework. Through this lens, we visualize reasoning traces and identify two core mechanisms in graph reasoning: token merging and structural memorization, which underlie both path reasoning and substructure extraction tasks. We further quantify these behaviors and analyze how they are influenced by graph density and model size. Our study provides a unified interpretability framework for understanding structural reasoning in decoder-only Transformers. 

**Abstract (ZH)**: 基于Transformer的大型语言模型在图推理任务中表现出色，但其内部机制仍缺乏探索。为了从基本和统一的角度揭示这些推理过程机制，我们采用电路追踪框架解释基本的解码器Transformer，并通过这一视角可视化推理痕迹，识别出图推理的两个核心机制：标记合并和结构记忆，这两种机制分别支撑路径推理和子结构提取任务。我们进一步量化这些行为，并分析它们如何受到图密度和模型规模的影响。我们的研究为理解解码器-only Transformer中的结构性推理提供了一个统一的解释框架。 

---
# Video models are zero-shot learners and reasoners 

**Title (ZH)**: 视频模型是零样本学习者和推理器 

**Authors**: Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, Robert Geirhos  

**Link**: [PDF](https://arxiv.org/pdf/2509.20328)  

**Abstract**: The remarkable zero-shot capabilities of Large Language Models (LLMs) have propelled natural language processing from task-specific models to unified, generalist foundation models. This transformation emerged from simple primitives: large, generative models trained on web-scale data. Curiously, the same primitives apply to today's generative video models. Could video models be on a trajectory towards general-purpose vision understanding, much like LLMs developed general-purpose language understanding? We demonstrate that Veo 3 can solve a broad variety of tasks it wasn't explicitly trained for: segmenting objects, detecting edges, editing images, understanding physical properties, recognizing object affordances, simulating tool use, and more. These abilities to perceive, model, and manipulate the visual world enable early forms of visual reasoning like maze and symmetry solving. Veo's emergent zero-shot capabilities indicate that video models are on a path to becoming unified, generalist vision foundation models. 

**Abstract (ZH)**: 大型语言模型的非凡零样本能力已将自然语言处理从任务特定模型推向统一的通用基础模型。这一转变源自简单的原理：在海量网络数据上训练的大规模生成模型。有趣的是，这些相同的原理也适用于当今的生成视频模型。视频模型能否像大型语言模型那样朝着通用视觉理解的方向发展？我们演示了Veo 3能够解决它未曾明确训练的任务：物体分割、边缘检测、图像编辑、理解物理属性、识别物体功能、模拟工具使用等。这些能力使视频模型能够进行早期形式的视觉推理，如迷宫和对称性解决。Veo的新兴零样本能力表明，视频模型正朝着统一的通用视觉基础模型的方向发展。 

---
# RAG Security and Privacy: Formalizing the Threat Model and Attack Surface 

**Title (ZH)**: RAG安全与隐私：正式化威胁模型与攻击表面 

**Authors**: Atousa Arzanipour, Rouzbeh Behnia, Reza Ebrahimi, Kaushik Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2509.20324)  

**Abstract**: Retrieval-Augmented Generation (RAG) is an emerging approach in natural language processing that combines large language models (LLMs) with external document retrieval to produce more accurate and grounded responses. While RAG has shown strong potential in reducing hallucinations and improving factual consistency, it also introduces new privacy and security challenges that differ from those faced by traditional LLMs. Existing research has demonstrated that LLMs can leak sensitive information through training data memorization or adversarial prompts, and RAG systems inherit many of these vulnerabilities. At the same time, reliance of RAG on an external knowledge base opens new attack surfaces, including the potential for leaking information about the presence or content of retrieved documents, or for injecting malicious content to manipulate model behavior. Despite these risks, there is currently no formal framework that defines the threat landscape for RAG systems. In this paper, we address a critical gap in the literature by proposing, to the best of our knowledge, the first formal threat model for retrieval-RAG systems. We introduce a structured taxonomy of adversary types based on their access to model components and data, and we formally define key threat vectors such as document-level membership inference and data poisoning, which pose serious privacy and integrity risks in real-world deployments. By establishing formal definitions and attack models, our work lays the foundation for a more rigorous and principled understanding of privacy and security in RAG systems. 

**Abstract (ZH)**: 检索增强生成（RAG）是一种自然语言处理新兴方法，结合了大型语言模型（LLMs）和外部文档检索，以产生更准确和可靠的响应。尽管RAG在减少幻觉和提高事实一致性方面显示出强大潜力，但它也引入了与传统LLMs不同的新隐私和安全挑战。现有研究已证明，LLMs可以通过训练数据记忆或对抗性提示泄露敏感信息，而RAG系统继承了许多这些漏洞。同时，RAG对外部知识库的依赖为新的攻击面打开了大门，包括泄露检索文档的存在或内容信息的可能性，或注入恶意内容以操控模型行为。尽管存在这些风险，目前尚无正式框架定义RAG系统的威胁场景。在本文中，我们通过提出（据我们所知）第一个正式的检索-RAG系统威胁模型，填补了文献中的一个关键空白。我们基于攻击者对模型组件和数据的访问类型引入了一个结构化的对手分类-taxonomy，并正式定义了文档级别成员推断和数据投毒等关键威胁向量，这些向量在实际部署中对隐私和完整性构成了严重风险。通过建立正式定义和攻击模型，我们的工作为RAG系统的隐私和安全提供了更严谨和原则性的理解奠定了基础。 

---
# DRES: Benchmarking LLMs for Disfluency Removal 

**Title (ZH)**: DRES: LLMs在修复表达不流畅性方面的基准测试 

**Authors**: Maria Teleki, Sai Janjur, Haoran Liu, Oliver Grabner, Ketan Verma, Thomas Docog, Xiangjue Dong, Lingfeng Shi, Cong Wang, Stephanie Birkelbach, Jason Kim, Yin Zhang, James Caverlee  

**Link**: [PDF](https://arxiv.org/pdf/2509.20321)  

**Abstract**: Disfluencies -- such as "um," "uh," interjections, parentheticals, and edited statements -- remain a persistent challenge for speech-driven systems, degrading accuracy in command interpretation, summarization, and conversational agents. We introduce DRES (Disfluency Removal Evaluation Suite), a controlled text-level benchmark that establishes a reproducible semantic upper bound for this task. DRES builds on human-annotated Switchboard transcripts, isolating disfluency removal from ASR errors and acoustic variability. We systematically evaluate proprietary and open-source LLMs across scales, prompting strategies, and architectures. Our results reveal that (i) simple segmentation consistently improves performance, even for long-context models; (ii) reasoning-oriented models tend to over-delete fluent tokens; and (iii) fine-tuning achieves near state-of-the-art precision and recall but harms generalization abilities. We further present a set of LLM-specific error modes and offer nine practical recommendations (R1-R9) for deploying disfluency removal in speech-driven pipelines. DRES provides a reproducible, model-agnostic foundation for advancing robust spoken-language systems. 

**Abstract (ZH)**: 语病消除——诸如“_um_”、“_uh_”、插语、括号中的说明和编辑过的陈述——仍然是基于语音的系统的一项持续性挑战，影响命令解释、总结和对话代理的准确性。我们引入了DRES（语病消除评估套件），这是一个受控的文本级别基准，为该任务建立了可重复的语义上限。DRES基于人工标注的Switchboard转录文本，将语病消除与ASR错误和声学变异分离。我们系统性地评估了各类专有和开源的大规模语言模型、提示策略和架构。我们的结果显示：（i）简单的分段一致性提高性能，即使对于长时间语境模型也有效；（ii）注重逻辑推理的模型倾向于删除过多的流畅通顺词语；和（iii）微调在达到接近SOTA的精确度和召回率的同时损害了泛化能力。我们进一步提出了大语言模型特有的错误模式，并提供了九条实用建议（R1-R9）以在语音驱动的处理管道中部署语病消除。DRES为推进稳健的口语系统提供了可重复且模型无关的基础。 

---
# Z-Scores: A Metric for Linguistically Assessing Disfluency Removal 

**Title (ZH)**: Z-分数：一种语言评估连读删除的方法 

**Authors**: Maria Teleki, Sai Janjur, Haoran Liu, Oliver Grabner, Ketan Verma, Thomas Docog, Xiangjue Dong, Lingfeng Shi, Cong Wang, Stephanie Birkelbach, Jason Kim, Yin Zhang, James Caverlee  

**Link**: [PDF](https://arxiv.org/pdf/2509.20319)  

**Abstract**: Evaluating disfluency removal in speech requires more than aggregate token-level scores. Traditional word-based metrics such as precision, recall, and F1 (E-Scores) capture overall performance but cannot reveal why models succeed or fail. We introduce Z-Scores, a span-level linguistically-grounded evaluation metric that categorizes system behavior across distinct disfluency types (EDITED, INTJ, PRN). Our deterministic alignment module enables robust mapping between generated text and disfluent transcripts, allowing Z-Scores to expose systematic weaknesses that word-level metrics obscure. By providing category-specific diagnostics, Z-Scores enable researchers to identify model failure modes and design targeted interventions -- such as tailored prompts or data augmentation -- yielding measurable performance improvements. A case study with LLMs shows that Z-Scores uncover challenges with INTJ and PRN disfluencies hidden in aggregate F1, directly informing model refinement strategies. 

**Abstract (ZH)**: 评估语音中的连贯性去除需要超出单词级别汇总得分。Z-分数：一种基于跨度的语义导向评估指标，用于跨不同类型断言（EDITED、INTJ、PRN）分类系统行为。 

---
# SIM-CoT: Supervised Implicit Chain-of-Thought 

**Title (ZH)**: SIM-CoT: 监督隐式链态推理 

**Authors**: Xilin Wei, Xiaoran Liu, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Jiaqi Wang, Xipeng Qiu, Dahua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.20317)  

**Abstract**: Implicit Chain-of-Thought (CoT) methods present a promising, token-efficient alternative to explicit CoT reasoning in Large Language Models (LLMs), but a persistent performance gap has limited the application of implicit CoT. We identify a core latent instability issue by scaling the computational budget of implicit CoT approaches: as we increase the number of implicit reasoning tokens to enhance performance, the training process often becomes unstable and collapses. Our analysis reveals that this instability arises from the latent representations becoming homogeneous and losing their semantic diversity, a failure caused by insufficient step-level supervision in existing implicit CoT approaches. To address this issue, we propose SIM-CoT, a plug-and-play training module that introduces step-level supervision to stabilize and enrich the latent reasoning space. Specifically, SIM-CoT employs an auxiliary decoder during training to align each implicit token with its corresponding explicit reasoning step, ensuring that latent states capture distinct and meaningful information. The proposed auxiliary decoder is removed during inference, preserving the computational efficiency of implicit CoT methods with no added overhead. In addition, the auxiliary decoder affords interpretability of implicit reasoning by projecting each latent token onto an explicit reasoning vocabulary, enabling per-step visualization of semantic roles and diagnosis. SIM-CoT significantly enhances both the in-domain accuracy and out-of-domain stability of various implicit CoT methods, boosting baselines like Coconut by +8.2% on GPT-2 and CODI by +3.0% on LLaMA-3.1 8B. Demonstrating strong scalability, SIM-CoT also surpasses the explicit CoT baseline on GPT-2 by 2.1% with 2.3\times greater token efficiency, while substantially closing the performance gap on larger models like LLaMA-3.1 8B. 

**Abstract (ZH)**: 隐式链式思考（CoT）方法为大型语言模型（LLMs）中显式CoT推理的节能替代方案提供了有希望的选择，但持续的性能差距限制了隐式CoT的应用。我们通过扩展隐式CoT方法的计算预算识别出一个核心的潜在不稳定性问题：随着我们增加隐式推理令牌以提高性能，训练过程往往变得不稳定并崩溃。我们的分析表明，这种不稳定性源于潜在表示变得同质并失去语义多样性，这是由于现有隐式CoT方法中不足的步骤级监督引起的。为了解决这个问题，我们提出了SIM-CoT，这是一种即插即用训练模块，引入步骤级监督以稳定和丰富潜在推理空间。具体来说，SIM-CoT在训练过程中采用辅助解码器，将每个隐式令牌与其对应的显式推理步骤对齐，确保潜在状态捕获独特且有意义的信息。所提出的辅助解码器在推理过程中被移除，从而保持隐式CoT方法的计算效率，无需额外开销。此外，辅助解码器通过将每个潜在令牌投影到显式推理词汇表上，为隐式推理的可解释性提供了支持，实现逐步骤的语义角色可视化和诊断。SIM-CoT显著提高了各种隐式CoT方法的领域内准确性和领域外稳定性，分别在GPT-2和LLaMA-3.1 8B上提升 coconut的基线表现8.2%，CODI的基线表现3.0%。SIM-CoT还展示了强大的可扩展性，在GPT-2上比显式CoT基线表现高出2.1%，且资源利用率提高了2.3倍，并在如LLaMA-3.1 8B等更大模型上显著缩小了性能差距。 

---
# When Judgment Becomes Noise: How Design Failures in LLM Judge Benchmarks Silently Undermine Validity 

**Title (ZH)**: 当判断变为噪音：设计故障如何悄然弱化LLM评估标准的有效性 

**Authors**: Benjamin Feuer, Chiung-Yi Tseng, Astitwa Sarthak Lathe, Oussama Elachqar, John P Dickerson  

**Link**: [PDF](https://arxiv.org/pdf/2509.20293)  

**Abstract**: LLM-judged benchmarks are increasingly used to evaluate complex model behaviors, yet their design introduces failure modes absent in conventional ground-truth based benchmarks. We argue that without tight objectives and verifiable constructions, benchmark rankings can produce high-confidence rankings that are in fact largely noise. We introduce two mechanisms to diagnose these issues. Schematic adherence quantifies how much of a judge's overall verdict is explained by the explicit evaluation schema, revealing unexplained variance when judges deviate from their own rubric. Psychometric validity aggregates internal consistency and discriminant validity signals to quantify irreducible uncertainty in any benchmarking run. Applying these tools to Arena-Hard Auto, we find severe schema incoherence and factor collapse across popular judges: for example, unexplained variance exceeding 90 percent for DeepSeek-R1-32B and factor correlations above 0.93 for most criteria. We also show that the ELO-style aggregation used by Arena-Hard Auto collapses and masks genuine ranking uncertainty. Our results highlight design failures that undermine validity and offer actionable principles for building better-scoped, reliability-aware LLM-judged benchmarks. We release our code at this https URL 

**Abstract (ZH)**: LLM判定基准大量用于评估复杂的模型行为，然而其设计引入了传统基于地面真实值的基准所没有的失败模式。我们argue如果没有明确的目标和可验证的构建，基准排名可能会产生高信心排名但实际上主要是噪声。我们介绍了两种诊断这些问题的机制。图示一致性度量了法官整体裁决中有多少可以由显式的评估框架解释，揭示了当法官偏离其评分标准时未解释的变异。心理测量有效性汇总了内部一致性和区分有效性信号，量化了任何基准运行中不可约减的不确定性。将这些工具应用于Arena-Hard Auto，我们发现广泛使用的法官在模式和因素方面存在严重不一致与合并：例如，DeepSeek-R1-32B的未解释变异超过90％，大多数标准的因素相关性高于0.93。我们还展示了Arena-Hard Auto使用的ELO风格聚合方式掩盖并掩盖了真实的排名不确定性。我们的结果强调了损害有效性的设计缺陷，并提供了构建更具针对性、可靠性意识的LLM判定基准的操作性原则。我们将在以下网址发布我们的代码：this https URL。 

---
# PGCLODA: Prompt-Guided Graph Contrastive Learning for Oligopeptide-Infectious Disease Association Prediction 

**Title (ZH)**: PGCLODA：提示引导的图对比学习在寡肽-传染性疾病关联预测中的应用 

**Authors**: Dayu Tan, Jing Chen, Xiaoping Zhou, Yansen Su, Chunhou Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.20290)  

**Abstract**: Infectious diseases continue to pose a serious threat to public health, underscoring the urgent need for effective computational approaches to screen novel anti-infective agents. Oligopeptides have emerged as promising candidates in antimicrobial research due to their structural simplicity, high bioavailability, and low susceptibility to resistance. Despite their potential, computational models specifically designed to predict associations between oligopeptides and infectious diseases remain scarce. This study introduces a prompt-guided graph-based contrastive learning framework (PGCLODA) to uncover potential associations. A tripartite graph is constructed with oligopeptides, microbes, and diseases as nodes, incorporating both structural and semantic information. To preserve critical regions during contrastive learning, a prompt-guided graph augmentation strategy is employed to generate meaningful paired views. A dual encoder architecture, integrating Graph Convolutional Network (GCN) and Transformer, is used to jointly capture local and global features. The fused embeddings are subsequently input into a multilayer perceptron (MLP) classifier for final prediction. Experimental results on a benchmark dataset indicate that PGCLODA consistently outperforms state-of-the-art models in AUROC, AUPRC, and accuracy. Ablation and hyperparameter studies confirm the contribution of each module. Case studies further validate the generalization ability of PGCLODA and its potential to uncover novel, biologically relevant associations. These findings offer valuable insights for mechanism-driven discovery and oligopeptide-based drug development. The source code of PGCLODA is available online at this https URL. 

**Abstract (ZH)**: 感染性疾病继续对公共卫生构成严重威胁，强调了迫切需要有效计算方法来筛选新型抗感染剂的急迫性。寡肽由于其结构简单、高生物利用度和低抗药性，已成为抗菌研究中的有前途的候选者。尽管具有巨大的潜力，但专门用于预测寡肽与感染性疾病之间关联的计算模型仍然稀缺。本文引入了一种基于提示的图对比学习框架（PGCLODA）以发现潜在的关联。构建了一个tripartite图，将寡肽、微生物和疾病作为节点，并结合了结构和语义信息。通过采用基于提示的图增强策略，生成有意义的配对视图，以保持对比学习过程中的关键区域。采用结合图卷积网络（GCN）和Transformer的双编码器架构，共同捕获局部和全局特征。融合后的嵌入随后输入多层感知机（MLP）分类器进行最终预测。在基准数据集上的实验结果表明，PGCLODA在AUROC、AUPRC和准确性方面均优于现有模型。消融和超参数研究证实了每个模块的贡献。案例研究进一步验证了PGCLODA的泛化能力和发现新型生物相关关联的潜力。这些发现为机制驱动的发现和基于寡肽的药物开发提供了宝贵见解。PGCLODA的源代码可在以下网址获得。 

---
# Feeding Two Birds or Favoring One? Adequacy-Fluency Tradeoffs in Evaluation and Meta-Evaluation of Machine Translation 

**Title (ZH)**: 兼顾两者还是偏向一方？机器翻译评估与元评估中的恰当性-流畅性权衡 

**Authors**: Behzad Shayegh, Jan-Thorsten Peter, David Vilar, Tobias Domhan, Juraj Juraska, Markus Freitag, Lili Mou  

**Link**: [PDF](https://arxiv.org/pdf/2509.20287)  

**Abstract**: We investigate the tradeoff between adequacy and fluency in machine translation. We show the severity of this tradeoff at the evaluation level and analyze where popular metrics fall within it. Essentially, current metrics generally lean toward adequacy, meaning that their scores correlate more strongly with the adequacy of translations than with fluency. More importantly, we find that this tradeoff also persists at the meta-evaluation level, and that the standard WMT meta-evaluation favors adequacy-oriented metrics over fluency-oriented ones. We show that this bias is partially attributed to the composition of the systems included in the meta-evaluation datasets. To control this bias, we propose a method that synthesizes translation systems in meta-evaluation. Our findings highlight the importance of understanding this tradeoff in meta-evaluation and its impact on metric rankings. 

**Abstract (ZH)**: 我们考察机器翻译中适当性和流畅性之间的权衡。我们在评估层面展示了这一权衡的严重性，并分析了流行指标在其间的分布情况。本质上，当前的指标通常更侧重于适当性，这意味着它们的得分与翻译的适当性相关性更强，而不是与流畅性。更重要的是，我们发现这一权衡在元评估层面也存在，并且标准的WMT元评估更偏好侧重适当性的指标，而不是侧重流畅性的指标。我们展示了这种偏见部分是由于元评估数据集所包含系统的构成。为了控制这种偏见，我们提出了一种合成元评估中翻译系统的办法。我们的发现强调了理解元评估中的这一权衡及其对指标排名影响的重要性。 

---
# Investigating Security Implications of Automatically Generated Code on the Software Supply Chain 

**Title (ZH)**: 自动生成代码对软件供应链安全影响的研究 

**Authors**: Xiaofan Li, Xing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.20277)  

**Abstract**: In recent years, various software supply chain (SSC) attacks have posed significant risks to the global community. Severe consequences may arise if developers integrate insecure code snippets that are vulnerable to SSC attacks into their products. Particularly, code generation techniques, such as large language models (LLMs), have been widely utilized in the developer community. However, LLMs are known to suffer from inherent issues when generating code, including fabrication, misinformation, and reliance on outdated training data, all of which can result in serious software supply chain threats. In this paper, we investigate the security threats to the SSC that arise from these inherent issues. We examine three categories of threats, including eleven potential SSC-related threats, related to external components in source code, and continuous integration configuration files. We find some threats in LLM-generated code could enable attackers to hijack software and workflows, while some others might cause potential hidden threats that compromise the security of the software over time. To understand these security impacts and severity, we design a tool, SSCGuard, to generate 439,138 prompts based on SSC-related questions collected online, and analyze the responses of four popular LLMs from GPT and Llama. Our results show that all identified SSC-related threats persistently exist. To mitigate these risks, we propose a novel prompt-based defense mechanism, namely Chain-of-Confirmation, to reduce fabrication, and a middleware-based defense that informs users of various SSC threats. 

**Abstract (ZH)**: 近年来，各种软件供应链(SSC)攻击对全球社区构成了重大风险。如果开发者将易受SSC攻击的不安全代码片段集成到其产品中，可能会导致严重后果。特别地，代码生成技术，如大型语言模型(LLMs)，在开发者社区中已被广泛使用。然而，LLMs在生成代码时存在根本性的问题，包括伪造、 misinformation（错误信息）和依赖过时的训练数据，这些都可能导致严重的软件供应链威胁。在本文中，我们调查了由这些根本性问题引发的供应链安全威胁。我们研究了包括源代码外部组件和持续集成配置文件在内的三类威胁，共发现了十一种潜在的SSC相关威胁。我们发现，某些LLM生成的代码中的威胁可以使攻击者劫持软件和工作流，而另一些则可能造成潜在的长期安全威胁。为了理解和评估这些安全影响及严重性，我们设计了一个名为SSCGuard的工具，基于收集到的与供应链相关的在线问题生成了439,138个提示，并分析了来自GPT和Llama的四种流行LLM的响应。结果显示，所有识别出的SSC相关威胁都持续存在。为了降低这些风险，我们提出了一种基于提示的新颖防御机制，名为确认链，以及一种基于中间件的防御机制，以告知用户各种供应链威胁。 

---
# AnchDrive: Bootstrapping Diffusion Policies with Hybrid Trajectory Anchors for End-to-End Driving 

**Title (ZH)**: AnchDrive: 通过混合轨迹锚点 bootstrap 驱动策略实现端到端驾驶 

**Authors**: Jinhao Chai, Anqing Jiang, Hao Jiang, Shiyi Mu, Zichong Gu, Shugong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20253)  

**Abstract**: End-to-end multi-modal planning has become a transformative paradigm in autonomous driving, effectively addressing behavioral multi-modality and the generalization challenge in long-tail scenarios. We propose AnchDrive, a framework for end-to-end driving that effectively bootstraps a diffusion policy to mitigate the high computational cost of traditional generative models. Rather than denoising from pure noise, AnchDrive initializes its planner with a rich set of hybrid trajectory anchors. These anchors are derived from two complementary sources: a static vocabulary of general driving priors and a set of dynamic, context-aware trajectories. The dynamic trajectories are decoded in real-time by a Transformer that processes dense and sparse perceptual features. The diffusion model then learns to refine these anchors by predicting a distribution of trajectory offsets, enabling fine-grained refinement. This anchor-based bootstrapping design allows for efficient generation of diverse, high-quality trajectories. Experiments on the NAVSIM benchmark confirm that AnchDrive sets a new state-of-the-art and shows strong gen?eralizability 

**Abstract (ZH)**: 端到端多模态规划已成为自主驾驶领域的变革性范式，有效应对了长尾场景中的行为多模态性和泛化挑战。我们提出AnchDrive框架，一种有效引导扩散策略的框架，以降低传统生成模型的高计算成本。AnchDrive 不是从纯噪声中去噪，而是用丰富的混合轨迹锚点初始化其规划器。这些锚点来源于两种互补的来源：一个静态的一般驾驶先验词汇表和一组动态的、上下文感知的轨迹。动态轨迹由Transformer实时解码，处理密集和稀疏的感知特征。然后，扩散模型通过预测路径偏移的分布来学习精炼这些锚点，实现精细的精炼。基于锚点的引导设计允许高效生成多样且高质量的轨迹。实验表明，AnchDrive 在 NAVSIM 基准测试中达到了新的最先进水平，并展示了强大的泛化能力。 

---
# A HyperGraphMamba-Based Multichannel Adaptive Model for ncRNA Classification 

**Title (ZH)**: 基于HyperGraphMamba的多通道自适应ncRNA分类模型 

**Authors**: Xin An, Ruijie Li, Qiao Ning, Hui Li, Qian Ma, Shikai Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.20240)  

**Abstract**: Non-coding RNAs (ncRNAs) play pivotal roles in gene expression regulation and the pathogenesis of various diseases. Accurate classification of ncRNAs is essential for functional annotation and disease diagnosis. To address existing limitations in feature extraction depth and multimodal fusion, we propose HGMamba-ncRNA, a HyperGraphMamba-based multichannel adaptive model, which integrates sequence, secondary structure, and optionally available expression features of ncRNAs to enhance classification performance. Specifically, the sequence of ncRNA is modeled using a parallel Multi-scale Convolution and LSTM architecture (MKC-L) to capture both local patterns and long-range dependencies of nucleotides. The structure modality employs a multi-scale graph transformer (MSGraphTransformer) to represent the multi-level topological characteristics of ncRNA secondary structures. The expression modality utilizes a Chebyshev Polynomial-based Kolmogorov-Arnold Network (CPKAN) to effectively model and interpret high-dimensional expression profiles. Finally, by incorporating virtual nodes to facilitate efficient and comprehensive multimodal interaction, HyperGraphMamba is proposed to adaptively align and integrate multichannel heterogeneous modality features. Experiments conducted on three public datasets demonstrate that HGMamba-ncRNA consistently outperforms state-of-the-art methods in terms of accuracy and other metrics. Extensive empirical studies further confirm the model's robustness, effectiveness, and strong transferability, offering a novel and reliable strategy for complex ncRNA functional classification. Code and datasets are available at this https URL. 

**Abstract (ZH)**: 非编码RNA（ncRNAs）在基因表达调控和多种疾病的发生机制中扮演关键角色。准确分类ncRNAs对于功能注释和疾病诊断至关重要。为了解决特征提取深度和多模态融合方面的现有局限性，我们提出了基于HyperGraphMamba的多通道自适应模型HGMamba-ncRNA，该模型整合了ncRNAs的序列、二级结构以及可获取的表达特征，以增强分类性能。具体而言，ncRNA序列使用并行多尺度卷积和LSTM架构（MKC-L）建模，以捕获核苷酸的局部模式和长程依赖性。结构模态采用多尺度图变换器（MSGraphTransformer）表示ncRNA二级结构的多层次拓扑特性。表达模态利用Chebyshev多项式基础上的Kolmogorov-Arnold网络（CPKAN）有效地建模和解释高维表达谱。最后，通过引入虚拟节点促进高效全面的多模态交互，提出了HyperGraphMamba以自适应对齐和整合多通道异质模态特征。在三个公开数据集上的实验表明，HGMamba-ncRNA在准确性和其他指标方面均优于现有方法。广泛的经验研究表明该模型具有鲁棒性、有效性及强迁移性，提供了一种处理复杂ncRNA功能分类的新颖可靠策略。代码和数据集可访问此链接。 

---
# ImageNet-trained CNNs are not biased towards texture: Revisiting feature reliance through controlled suppression 

**Title (ZH)**: ImageNet训练的CNNs并不偏向纹理：通过可控抑制重新审视特征依赖 

**Authors**: Tom Burgert, Oliver Stoll, Paolo Rota, Begüm Demir  

**Link**: [PDF](https://arxiv.org/pdf/2509.20234)  

**Abstract**: The hypothesis that Convolutional Neural Networks (CNNs) are inherently texture-biased has shaped much of the discourse on feature use in deep learning. We revisit this hypothesis by examining limitations in the cue-conflict experiment by Geirhos et al. To address these limitations, we propose a domain-agnostic framework that quantifies feature reliance through systematic suppression of shape, texture, and color cues, avoiding the confounds of forced-choice conflicts. By evaluating humans and neural networks under controlled suppression conditions, we find that CNNs are not inherently texture-biased but predominantly rely on local shape features. Nonetheless, this reliance can be substantially mitigated through modern training strategies or architectures (ConvNeXt, ViTs). We further extend the analysis across computer vision, medical imaging, and remote sensing, revealing that reliance patterns differ systematically: computer vision models prioritize shape, medical imaging models emphasize color, and remote sensing models exhibit a stronger reliance towards texture. Code is available at this https URL. 

**Abstract (ZH)**: 卷积神经网络对纹理的偏置并非固有的：一种去噪框架及其在计算机视觉、医学成像和遥感中的应用 

---
# Beyond Sharp Minima: Robust LLM Unlearning via Feedback-Guided Multi-Point Optimization 

**Title (ZH)**: 超越尖锐极小值：基于反馈引导的多点优化实现稳健的大语言模型去学习 

**Authors**: Wenhan Wu, Zheyuan Liu, Chongyang Gao, Ren Wang, Kaize Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.20230)  

**Abstract**: Current LLM unlearning methods face a critical security vulnerability that undermines their fundamental purpose: while they appear to successfully remove sensitive or harmful knowledge, this ``forgotten" information remains precariously recoverable through relearning attacks. We identify that the root cause is that conventional methods optimizing the forgetting loss at individual data points will drive model parameters toward sharp minima in the loss landscape. In these unstable regions, even minimal parameter perturbations can drastically alter the model's behaviors. Consequently, relearning attacks exploit this vulnerability by using just a few fine-tuning samples to navigate the steep gradients surrounding these unstable regions, thereby rapidly recovering knowledge that was supposedly erased. This exposes a critical robustness gap between apparent unlearning and actual knowledge removal. To address this issue, we propose StableUN, a bi-level feedback-guided optimization framework that explicitly seeks more stable parameter regions via neighborhood-aware optimization. It integrates forgetting feedback, which uses adversarial perturbations to probe parameter neighborhoods, with remembering feedback to preserve model utility, aligning the two objectives through gradient projection. Experiments on WMDP and MUSE benchmarks demonstrate that our method is significantly more robust against both relearning and jailbreaking attacks while maintaining competitive utility performance. 

**Abstract (ZH)**: 当前的LLM去学习方法面临一个关键的安全漏洞，这削弱了它们的基本目的：虽然它们看似成功地移除了敏感或有害的知识，但这些“遗忘”的信息可以通过重新学习攻击危险地恢复。我们发现根本原因在于，传统的优化个体数据点遗忘损失的方法会使模型参数朝损失景观中的尖锐极小值演变。在这些不稳定区域，即使是微小的参数扰动也会大幅改变模型的行为。因此，重新学习攻击利用这一漏洞，仅通过少量微调样本导航这些不稳定区域周围的陡峭梯度，从而迅速恢复据称已被删除的知识。这暴露了表象的去学习与实际的知识移除之间的重要鲁棒性差距。为解决这一问题，我们提出了一种双层反馈引导优化框架StableUN，该框架通过邻域感知优化显式寻求更稳定的参数区域。它结合了遗忘反馈（使用对抗扰动探索参数邻域）和记忆反馈，通过梯度投影对两个目标进行对齐。在WMDP和MUSE基准测试上的实验表明，我们的方法在抵抗重新学习和 Jailbreaking 攻击方面显著更鲁棒，同时保持了竞争力的实用性能。 

---
# Multimodal Representation-disentangled Information Bottleneck for Multimodal Recommendation 

**Title (ZH)**: 多模态 Representation 分离的信息瓶颈多模态推荐 

**Authors**: Hui Wang, Jinghui Qin, Wushao Wen, Qingling Li, Shanshan Zhong, Zhongzhan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20225)  

**Abstract**: Multimodal data has significantly advanced recommendation systems by integrating diverse information sources to model user preferences and item characteristics. However, these systems often struggle with redundant and irrelevant information, which can degrade performance. Most existing methods either fuse multimodal information directly or use rigid architectural separation for disentanglement, failing to adequately filter noise and model the complex interplay between modalities. To address these challenges, we propose a novel framework, the Multimodal Representation-disentangled Information Bottleneck (MRdIB). Concretely, we first employ a Multimodal Information Bottleneck to compress the input representations, effectively filtering out task-irrelevant noise while preserving rich semantic information. Then, we decompose the information based on its relationship with the recommendation target into unique, redundant, and synergistic components. We achieve this decomposition with a series of constraints: a unique information learning objective to preserve modality-unique signals, a redundant information learning objective to minimize overlap, and a synergistic information learning objective to capture emergent information. By optimizing these objectives, MRdIB guides a model to learn more powerful and disentangled representations. Extensive experiments on several competitive models and three benchmark datasets demonstrate the effectiveness and versatility of our MRdIB in enhancing multimodal recommendation. 

**Abstract (ZH)**: 多模态表示解缠信息瓶颈（MRdIB）在增强多模态推荐中的应用 

---
# The Cream Rises to the Top: Efficient Reranking Method for Verilog Code Generation 

**Title (ZH)**: 优秀的代码浮上 come out on top: 基于高效重排序方法的Verilog代码生成 

**Authors**: Guang Yang, Wei Zheng, Xiang Chen, Yifan Sun, Fengji Zhang, Terry Yue Zhuo  

**Link**: [PDF](https://arxiv.org/pdf/2509.20215)  

**Abstract**: LLMs face significant challenges in Verilog generation due to limited domain-specific knowledge. While sampling techniques improve pass@k metrics, hardware engineers need one trustworthy solution rather than uncertain candidates. To bridge this gap, we formulate it as a semantic alignment problem between requirements and Verilog implementations, and propose VCD-RNK, a discriminator model tailored for efficient Verilog code reranking. Specifically, VCD-RNKincorporates Verilog-specific reasoning by distilling expert knowledge across three dimensions: code semantic analysis, test case generation, and functional correctness assessment. By explicitly simulating the above reasoning processes during inference, VCD-RNK effectively avoids computationally intensive test execution in existing methods. 

**Abstract (ZH)**: LLMs在Verilog生成中面临显著挑战，受限于有限的专业领域知识。虽然采样技术改善了pass@k指标，硬件工程师更需要一个可靠的解决方案而非不确定的选择。为了弥合这一差距，我们将这一问题形式化为需求与Verilog实现之间的语义对齐问题，并提出了VCD-RNK，一种针对高效Verilog代码重排序的鉴别模型。具体而言，VCD-RNK通过提炼三维专家知识——代码语义分析、测试案例生成和功能正确性评估——增强了Verilog特定的推理能力。通过在推理过程中显式模拟上述推理过程，VCD-RNK有效避免了现有方法中计算密集型的测试执行。 

---
# Q-Palette: Fractional-Bit Quantizers Toward Optimal Bit Allocation for Efficient LLM Deployment 

**Title (ZH)**: Q-Palette: 分数位量化器 toward 最优位分配的高效大语言模型部署 

**Authors**: Deokjae Lee, Hyun Oh Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.20214)  

**Abstract**: We study weight-only post-training quantization (PTQ), which quantizes the weights of a large language model (LLM) without retraining, using little or no calibration data. Weight-only PTQ is crucial for reducing the memory footprint and latency of LLM inference, especially in memory-bound, small-batch inference scenarios, such as personalized inference on edge devices. Despite its importance, irregular weight distributions with heavy-tailed outliers in LLMs complicate quantization, recently motivating rotation-based methods that transform weights into near-Gaussian distributions, which are more regular with fewer outliers, thereby reducing quantization error. In this work, we first derive the information-theoretically optimal bit allocation for Gaussianized weights under given bit budgets, revealing that fine-grained fractional-bit quantizers approaching the Gaussian distortion-rate bound are essential to achieve near-optimal quantization performance. To bridge this theoretical insight and practical implementation, we introduce Q-Palette, a versatile collection of fractional-bit quantizers that range from trellis-coded quantizers offering near-optimal distortion to simpler vector and scalar quantizers optimized for faster inference, all efficiently implemented with optimized CUDA kernels across various bitwidths. Furthermore, leveraging Q-Palette as a foundational component, we propose a novel mixed-scheme quantization framework, jointly optimizing quantizer choices and layer fusion decisions given resource constraints. The code is available at this https URL. 

**Abstract (ZH)**: 我们研究了无需重新训练且使用少量或无需校准数据的仅权重后训练量化（PTQ），以减少大型语言模型（LLM）推理中的内存占用和延迟，特别是在内存受限的小批量推理场景中，如边缘设备上的个性化推理。尽管其重要性，LLM中不规则的权重分布和重尾异常值使量化变得复杂，最近推动了基于旋转的方法，这些方法将权重转化为接近高斯分布，从而减少了量化误差并具有较少的异常值。在本文中，我们首先在给定比特预算的情况下推导出高斯化权重的信息论最优比特分配，揭示了接近高斯失真率边界的细粒度分数比特量化器对于实现近似最优量化性能是必不可少的。为了将这一理论洞见与实际实现相融合，我们引入了Q-Palette，这是一种多功能的分数比特量化器集合，从提供接近最优失真的梯形编码量化器到针对更快推理优化的简单向量和标量化量化器，所有这些都在各种位宽下通过对优化的CUDA内核进行高效实现。此外，基于Q-Palette作为基础组件，我们提出了一种新的混合方案量化框架，在资源受限的情况下联合优化量化器选择和层融合决策。代码可在以下链接获取：this https URL。 

---
# Low-Resource English-Tigrinya MT: Leveraging Multilingual Models, Custom Tokenizers, and Clean Evaluation Benchmarks 

**Title (ZH)**: 低资源英提尼语机器翻译：利用多语言模型、自定义分词器和清洁评估基准 

**Authors**: Hailay Kidu Teklehaymanot, Gebrearegawi Gidey, Wolfgang Nejdl  

**Link**: [PDF](https://arxiv.org/pdf/2509.20209)  

**Abstract**: Despite advances in Neural Machine Translation (NMT), low-resource languages like Tigrinya remain underserved due to persistent challenges, including limited corpora, inadequate tokenization strategies, and the lack of standardized evaluation benchmarks. This paper investigates transfer learning techniques using multilingual pretrained models to enhance translation quality for morphologically rich, low-resource languages. We propose a refined approach that integrates language-specific tokenization, informed embedding initialization, and domain-adaptive fine-tuning. To enable rigorous assessment, we construct a high-quality, human-aligned English-Tigrinya evaluation dataset covering diverse domains. Experimental results demonstrate that transfer learning with a custom tokenizer substantially outperforms zero-shot baselines, with gains validated by BLEU, chrF, and qualitative human evaluation. Bonferroni correction is applied to ensure statistical significance across configurations. Error analysis reveals key limitations and informs targeted refinements. This study underscores the importance of linguistically aware modeling and reproducible benchmarks in bridging the performance gap for underrepresented languages. Resources are available at this https URL
and this https URL 

**Abstract (ZH)**: 尽管神经机器翻译（NMT）取得了进展，但由于数据量有限、分词策略不足和缺乏标准化评估基准等持续性的挑战，像提格雷尼亚语这样的低资源语言仍得不到充分服务。本文研究了使用多语言预训练模型的迁移学习技术，以提升形态学丰富的低资源语言的翻译质量。我们提出了一种改进的方法，该方法结合了语言特定的分词、有指导的嵌入初始化和领域适应性微调。为了进行严格的评估，我们构建了一个高质量的人工对齐的英语-提格雷尼亚语评估数据集，涵盖多个领域。实验结果表明，使用自定义分词器的迁移学习显著优于零-shot基准，通过BLEU、chrF和定性的手工评价得到了验证。使用布农尼罗修正确保各配置下的统计显著性。错误分析揭示了关键限制并指导了有针对性的改进。本研究强调了在减少欠代表语言性能差距时需重视语言意识建模和可再现基准的重要性。资源可访问于此https:// 和此https://。 

---
# Play by the Type Rules: Inferring Constraints for LLM Functions in Declarative Programs 

**Title (ZH)**: 按类型规则玩耍: 确定声明式程序中LLM函数的约束规则 

**Authors**: Parker Glenn, Alfy Samuel, Daben Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20208)  

**Abstract**: Integrating LLM powered operators in declarative query languages allows for the combination of cheap and interpretable functions with powerful, generalizable language model reasoning. However, in order to benefit from the optimized execution of a database query language like SQL, generated outputs must align with the rules enforced by both type checkers and database contents. Current approaches address this challenge with orchestrations consisting of many LLM-based post-processing calls to ensure alignment between generated outputs and database values, introducing performance bottlenecks. We perform a study on the ability of various sized open-source language models to both parse and execute functions within a query language based on SQL, showing that small language models can excel as function executors over hybrid data sources. Then, we propose an efficient solution to enforce the well-typedness of LLM functions, demonstrating 7% accuracy improvement on a multi-hop question answering dataset with 53% improvement in latency over comparable solutions. We make our implementation available at this https URL 

**Abstract (ZH)**: 基于LLM的运算符集成在声明性查询语言中，能够结合便宜且可解释的函数与强大、可泛化的语言模型推理。但是，为了从数据库查询语言（如SQL）的优化执行中受益，生成的输出必须遵守类型检查器和数据库内容制定的规则。当前方法通过使用许多基于LLM的后处理调用来确保生成输出与数据库值之间的对齐，引入了性能瓶颈。我们研究了各种大小的开源语言模型在基于SQL的查询语言中解析和执行函数的能力，表明小型语言模型可以在混合数据源中作为函数执行器表现出色。然后，我们提出了一种高效的解决方案来强制执行LLM函数的类型正确性，在多跳问答数据集上获得7%的准确性改进，并且与 comparable 解决方案相比，延迟性能提高了53%。我们的实现可在以下链接获得。 

---
# STAF: Leveraging LLMs for Automated Attack Tree-Based Security Test Generation 

**Title (ZH)**: STAF: 利用大语言模型进行基于攻击树的安全测试生成 

**Authors**: Tanmay Khule, Stefan Marksteiner, Jose Alguindigue, Hannes Fuchs, Sebastian Fischmeister, Apurva Narayan  

**Link**: [PDF](https://arxiv.org/pdf/2509.20190)  

**Abstract**: In modern automotive development, security testing is critical for safeguarding systems against increasingly advanced threats. Attack trees are widely used to systematically represent potential attack vectors, but generating comprehensive test cases from these trees remains a labor-intensive, error-prone task that has seen limited automation in the context of testing vehicular systems. This paper introduces STAF (Security Test Automation Framework), a novel approach to automating security test case generation. Leveraging Large Language Models (LLMs) and a four-step self-corrective Retrieval-Augmented Generation (RAG) framework, STAF automates the generation of executable security test cases from attack trees, providing an end-to-end solution that encompasses the entire attack surface. We particularly show the elements and processes needed to provide an LLM to actually produce sensible and executable automotive security test suites, along with the integration with an automated testing framework. We further compare our tailored approach with general purpose (vanilla) LLMs and the performance of different LLMs (namely GPT-4.1 and DeepSeek) using our approach. We also demonstrate the method of our operation step-by-step in a concrete case study. Our results show significant improvements in efficiency, accuracy, scalability, and easy integration in any workflow, marking a substantial advancement in automating automotive security testing methodologies. Using TARAs as an input for verfication tests, we create synergies by connecting two vital elements of a secure automotive development process. 

**Abstract (ZH)**: 现代汽车开发中，安全测试对于保护系统免受日益先进的威胁至关重要。攻击树广泛用于系统地表示潜在攻击向量，但将这些树转化为全面的测试用例仍然是一个劳动密集型且容易出错的过程，在汽车系统测试中自动化程度有限。本文介绍了STAF（安全测试自动化框架），这是一种全新的安全测试用例生成自动化方法。STAF利用大型语言模型（LLMs）和四步自修正检索增强生成（RAG）框架，自动从攻击树生成可执行的安全测试用例，提供了一个覆盖整个攻击面的端到端解决方案。我们特别展示了将LLMs实际上用于生成具有意义且可执行的汽车安全测试套件所需的各种元素和过程，以及与自动化测试框架的集成方式。我们进一步比较了我们定制的方法与通用目的（标准）LLMs的表现，并使用我们的方法评估了不同LLMs（包括GPT-4.1和DeepSeek）的表现。我们还通过一个具体的案例研究逐步展示了我们的操作方法。结果显示，STAF在效率、准确性和可扩展性方面取得了显著提升，并且易于集成到任何工作流程中，标志着汽车安全测试方法自动化的一大进步。通过将TARAs作为验证测试的输入，我们建立了一种连接安全汽车开发过程两个关键要素的协同效应。 

---
# How People Manage Knowledge in their "Second Brains"- A Case Study with Industry Researchers Using Obsidian 

**Title (ZH)**: 人们在其“第二个大脑”中管理知识的研究——基于使用Obsidian的行业研究人员案例研究 

**Authors**: Juliana Jansen Ferreira, Vinícius Segura, Joana Gabriela Souza, Joao Henrique Gallas Brasil  

**Link**: [PDF](https://arxiv.org/pdf/2509.20187)  

**Abstract**: People face overwhelming information during work activities, necessitating effective organization and management strategies. Even in personal lives, individuals must keep, annotate, organize, and retrieve knowledge from daily routines. The collection of records for future reference is known as a personal knowledge base. Note-taking applications are valuable tools for building and maintaining these bases, often called a ''second brain''. This paper presents a case study on how people build and explore personal knowledge bases for various purposes. We selected the note-taking tool Obsidian and researchers from a Brazilian lab for an in-depth investigation. Our investigation reveals interesting findings about how researchers build and explore their personal knowledge bases. A key finding is that participants' knowledge retrieval strategy influences how they build and maintain their content. We suggest potential features for an AI system to support this process. 

**Abstract (ZH)**: 人们在工作活动中面临海量信息，需要有效的组织和管理策略。即使在个人生活中，个体也需要记录、注释、组织和检索日常生活中的知识。为了未来参考而收集的记录称为个人知识库。笔记应用程序是构建和维护这些知识库的有效工具，常被称为“第二个大脑”。本文展示了如何为了各种目的构建和探索个人知识库的案例研究。我们选择了笔记工具Obsidian和巴西实验室的研究人员进行了深入调查。我们的调查揭示了研究人员构建和探索个人知识库的一些有趣发现。一个关键发现是参与者的知识检索策略影响了他们构建和维护内容的方式。我们提出了可能支持这一过程的AI系统功能建议。 

---
# An Improved Time Series Anomaly Detection by Applying Structural Similarity 

**Title (ZH)**: 应用结构相似性改进的时间序列异常检测 

**Authors**: Tiejun Wang, Rui Wang, Xudong Mou, Mengyuan Ma, Tianyu Wo, Renyu Yang, Xudong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20184)  

**Abstract**: Effective anomaly detection in time series is pivotal for modern industrial applications and financial systems. Due to the scarcity of anomaly labels and the high cost of manual labeling, reconstruction-based unsupervised approaches have garnered considerable attention. However, accurate anomaly detection remains an unsettled challenge, since the optimization objectives of reconstruction-based methods merely rely on point-by-point distance measures, ignoring the potential structural characteristics of time series and thus failing to tackle complex pattern-wise anomalies. In this paper, we propose StrAD, a novel structure-enhanced anomaly detection approach to enrich the optimization objective by incorporating structural information hidden in the time series and steering the data reconstruction procedure to better capture such structural features. StrAD accommodates the trend, seasonality, and shape in the optimization objective of the reconstruction model to learn latent structural characteristics and capture the intrinsic pattern variation of time series. The proposed structure-aware optimization objective mechanism can assure the alignment between the original data and the reconstructed data in terms of structural features, thereby keeping consistency in global fluctuation and local characteristics. The mechanism is pluggable and applicable to any reconstruction-based methods, enhancing the model sensitivity to both point-wise anomalies and pattern-wise anomalies. Experimental results show that StrAD improves the performance of state-of-the-art reconstruction-based models across five real-world anomaly detection datasets. 

**Abstract (ZH)**: 结构增强的时间序列异常检测方法StrAD 

---
# Automated Multi-Agent Workflows for RTL Design 

**Title (ZH)**: 基于RTL设计的自动化多代理工作流 

**Authors**: Amulya Bhattaram, Janani Ramamoorthy, Ranit Gupta, Diana Marculescu, Dimitrios Stamoulis  

**Link**: [PDF](https://arxiv.org/pdf/2509.20182)  

**Abstract**: The rise of agentic AI workflows unlocks novel opportunities for computer systems design and optimization. However, for specialized domains such as program synthesis, the relative scarcity of HDL and proprietary EDA resources online compared to more common programming tasks introduces challenges, often necessitating task-specific fine-tuning, high inference costs, and manually-crafted agent orchestration. In this work, we present VeriMaAS, a multi-agent framework designed to automatically compose agentic workflows for RTL code generation. Our key insight is to integrate formal verification feedback from HDL tools directly into workflow generation, reducing the cost of gradient-based updates or prolonged reasoning traces. Our method improves synthesis performance by 5-7% for pass@k over fine-tuned baselines, while requiring only a few hundred training examples, representing an order-of-magnitude reduction in supervision cost. 

**Abstract (ZH)**: 基于代理的AI工作流的兴起为计算机系统设计与优化开启了新的机遇。然而，对于如程序合成这样的专业领域，相较于常见的编程任务，HDL和专有EDA资源的相对稀缺性常常带来挑战，通常需要任务特定的微调、高昂的推理成本以及手工编排的代理协同。本文提出了一种名为VeriMaAS的多代理框架，旨在自动生成用于RTL代码生成的代理工作流。我们的核心见解是直接将HDL工具的正式验证反馈整合到工作流生成中，从而减少基于梯度的更新或长时间推理轨迹的成本。该方法在k@pass性能上比微调基线提高了5-7%，同时只需要少量的训练样本，代表了监督成本的一个数量级减少。 

---
# CyberSOCEval: Benchmarking LLMs Capabilities for Malware Analysis and Threat Intelligence Reasoning 

**Title (ZH)**: CyberSOCEval: 评估LLM在恶意软件分析和威胁情报推理方面的能力 

**Authors**: Lauren Deason, Adam Bali, Ciprian Bejean, Diana Bolocan, James Crnkovich, Ioana Croitoru, Krishna Durai, Chase Midler, Calin Miron, David Molnar, Brad Moon, Bruno Ostarcevic, Alberto Peltea, Matt Rosenberg, Catalin Sandu, Arthur Saputkin, Sagar Shah, Daniel Stan, Ernest Szocs, Shengye Wan, Spencer Whitman, Sven Krasser, Joshua Saxe  

**Link**: [PDF](https://arxiv.org/pdf/2509.20166)  

**Abstract**: Today's cyber defenders are overwhelmed by a deluge of security alerts, threat intelligence signals, and shifting business context, creating an urgent need for AI systems to enhance operational security work. While Large Language Models (LLMs) have the potential to automate and scale Security Operations Center (SOC) operations, existing evaluations do not fully assess the scenarios most relevant to real-world defenders. This lack of informed evaluation impacts both AI developers and those applying LLMs to SOC automation. Without clear insight into LLM performance in real-world security scenarios, developers lack a north star for development, and users cannot reliably select the most effective models. Meanwhile, malicious actors are using AI to scale cyber attacks, highlighting the need for open source benchmarks to drive adoption and community-driven improvement among defenders and model developers. To address this, we introduce CyberSOCEval, a new suite of open source benchmarks within CyberSecEval 4. CyberSOCEval includes benchmarks tailored to evaluate LLMs in two tasks: Malware Analysis and Threat Intelligence Reasoning--core defensive domains with inadequate coverage in current benchmarks. Our evaluations show that larger, more modern LLMs tend to perform better, confirming the training scaling laws paradigm. We also find that reasoning models leveraging test time scaling do not achieve the same boost as in coding and math, suggesting these models have not been trained to reason about cybersecurity analysis, and pointing to a key opportunity for improvement. Finally, current LLMs are far from saturating our evaluations, showing that CyberSOCEval presents a significant challenge for AI developers to improve cyber defense capabilities. 

**Abstract (ZH)**: 今天的网络防御者面临着大量的安全警报、威胁情报信号以及不断变化的业务情境，迫切需要人工智能系统来增强运营安全工作。尽管大规模语言模型（LLMs）有望自动化并扩展安全运营中心（SOC）的操作，现有的评估并未充分评估真实世界防御者最相关的场景。这种缺乏有见地的评估影响了AI开发人员和将LLMs应用于SOC自动化的用户。没有清晰的LLM性能洞察，开发人员缺乏开发的方向，用户也无法可靠地选择最有效的模型。同时，恶意行为者正利用AI放大网络攻击，突显了需要开放源代码基准来推动防御者和模型开发人员之间采用与社区驱动改进的必要性。为应对这一挑战，我们介绍了CyberSOCEval，这是CyberSecEval 4的新一代开放源代码基准套件。CyberSOCEval 包括针对恶意软件分析和威胁情报推理两个核心防御领域定制的基准，这些领域在现有基准中缺乏足够的覆盖。我们的评估表明，更大、更现代的LLM通常表现更好，确认了训练规模律。我们还发现，依赖于测试时缩放的推理模型并未像在编程和数学中那样获得同样的提升，这表明这些模型并未被训练来推理网络安全分析，并指出了一个关键的改进机会。最后，当前的LLM远远没有饱和我们的评估，表明CyberSOCEval 对AI开发人员改进网络安全能力提出了重大挑战。 

---
# Embedding Domain Knowledge for Large Language Models via Reinforcement Learning from Augmented Generation 

**Title (ZH)**: 通过增强生成的强化学习嵌入领域知识的大语言模型 

**Authors**: Chaojun Nie, Jun Zhou, Guanxiang Wang, Shisong Wud, Zichen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20162)  

**Abstract**: Large language models (LLMs) often exhibit limited performance on domain-specific tasks due to the natural disproportionate representation of specialized information in their training data and the static nature of these datasets. Knowledge scarcity and temporal lag create knowledge gaps for domain applications. While post-training on domain datasets can embed knowledge into models, existing approaches have some limitations. Continual Pre-Training (CPT) treats all tokens in domain documents with equal importance, failing to prioritize critical knowledge points, while supervised fine-tuning (SFT) with question-answer pairs struggles to develop the coherent knowledge structures necessary for complex reasoning tasks. To address these challenges, we propose Reinforcement Learning from Augmented Generation (RLAG). Our approach iteratively cycles between sampling generations and optimizing the model through calculated rewards, effectively embedding critical and contextually coherent domain knowledge. We select generated outputs with the highest log probabilities as the sampling result, then compute three tailored reward metrics to guide the optimization process. To comprehensively evaluate domain expertise, we assess answer accuracy and the rationality of explanations generated for correctly answered questions. Experimental results across medical, legal, astronomy, and current events datasets demonstrate that our proposed method significantly outperforms baseline approaches. Our code and data are open sourced at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在特定领域任务中常常表现出有限的效果，这归因于其训练数据中天然存在的专业知识不平衡表示以及这些数据集的静态性质。知识稀缺性和时间滞后造成了领域应用的知识缺口。虽然在领域数据集上进行后续训练可以将知识嵌入模型，但现有方法存在一些局限性。持续预训练（CPT）赋予领域文件中所有token相同的重要性，未能优先处理关键知识点，而基于问答的监督微调（SFT）则难以发展出复杂推理任务所需的连贯知识结构。为应对这些挑战，我们提出了增强生成的强化学习（RAGL）。该方法通过迭代采样生成并根据计算奖励优化模型，有效嵌入关键且上下文连贯的领域知识。我们选择具有最高对数概率的生成输出作为采样结果，然后计算三种定制的奖励度量来引导优化过程。为了全面评估领域专业知识，我们评估了答案的准确性和为正确回答问题生成的解释的合理性。在医学、法律、天文学和当前事件数据集上的实验结果证明，我们提出的方法显著优于基准方法。我们的代码和数据在该URL处公开。 

---
# U-Mamba2-SSL for Semi-Supervised Tooth and Pulp Segmentation in CBCT 

**Title (ZH)**: U-Mamba2-SSL在CBCT中的半监督牙齿和牙髓分割 

**Authors**: Zhi Qin Tan, Xiatian Zhu, Owen Addison, Yunpeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.20154)  

**Abstract**: Accurate segmentation of teeth and pulp in Cone-Beam Computed Tomography (CBCT) is vital for clinical applications like treatment planning and diagnosis. However, this process requires extensive expertise and is exceptionally time-consuming, highlighting the critical need for automated algorithms that can effectively utilize unlabeled data. In this paper, we propose U-Mamba2-SSL, a novel semi-supervised learning framework that builds on the U-Mamba2 model and employs a multi-stage training strategy. The framework first pre-trains U-Mamba2 in a self-supervised manner using a disruptive autoencoder. It then leverages unlabeled data through consistency regularization, where we introduce input and feature perturbations to ensure stable model outputs. Finally, a pseudo-labeling strategy is implemented with a reduced loss weighting to minimize the impact of potential errors. U-Mamba2-SSL achieved an average score of 0.872 and a DSC of 0.969 on the validation dataset, demonstrating the superior performance of our approach. The code is available at this https URL. 

**Abstract (ZH)**: Cone-Beam 计算机断层扫描中牙齿和牙髓的准确分割对于治疗计划和诊断等临床应用至关重要。然而，这一过程需要大量专业知识且极为耗时，突显了开发高效利用未标记数据的自动化算法的迫切需求。本文提出了一种新颖的半监督学习框架 U-Mamba2-SSL，该框架基于 U-Mamba2 模型，并采用多阶段训练策略。框架首先使用破坏性自动编码器以自监督方式预训练 U-Mamba2，然后通过一致性正则化利用未标记数据，通过引入输入和特征扰动确保稳定的模型输出，最后采用带有减少损失权重的伪标签策略以减少潜在错误的影响。U-Mamba2-SSL 在验证数据集上的平均得分为 0.872，DSC 为 0.969，证明了该方法的优越性能。代码可在以下链接获取。 

---
# Affective Computing and Emotional Data: Challenges and Implications in Privacy Regulations, The AI Act, and Ethics in Large Language Models 

**Title (ZH)**: 情感计算与情绪数据：在《AI法案》、隐私法规及大型语言模型伦理中的挑战与影响 

**Authors**: Nicola Fabiano  

**Link**: [PDF](https://arxiv.org/pdf/2509.20153)  

**Abstract**: This paper examines the integration of emotional intelligence into artificial intelligence systems, with a focus on affective computing and the growing capabilities of Large Language Models (LLMs), such as ChatGPT and Claude, to recognize and respond to human emotions. Drawing on interdisciplinary research that combines computer science, psychology, and neuroscience, the study analyzes foundational neural architectures - CNNs for processing facial expressions and RNNs for sequential data, such as speech and text - that enable emotion recognition. It examines the transformation of human emotional experiences into structured emotional data, addressing the distinction between explicit emotional data collected with informed consent in research settings and implicit data gathered passively through everyday digital interactions. That raises critical concerns about lawful processing, AI transparency, and individual autonomy over emotional expressions in digital environments. The paper explores implications across various domains, including healthcare, education, and customer service, while addressing challenges of cultural variations in emotional expression and potential biases in emotion recognition systems across different demographic groups. From a regulatory perspective, the paper examines emotional data in the context of the GDPR and the EU AI Act frameworks, highlighting how emotional data may be considered sensitive personal data that requires robust safeguards, including purpose limitation, data minimization, and meaningful consent mechanisms. 

**Abstract (ZH)**: 本文研究情感 inteligence 与人工智能系统的集成，重点关注情感计算以及大型语言模型（如 ChatGPT 和 Claude）识别和响应人类情感的能力。通过结合计算机科学、心理学和神经科学的跨学科研究，该研究分析了用于处理面部表情的卷积神经网络（CNNs）和用于序列数据（如语音和文本）的递归神经网络（RNNs）等基础神经架构，以实现情感识别。文章探讨了将人类情感体验转化为结构化情感数据的过程，讨论了明示情感数据（在研究环境中通过知情同意收集）与潜在情感数据（通过日常数字互动被动收集）之间的区别。这引起了关于法律处理、AI 透明度和个体在数字环境中对情感表达的自主权的关切。文章探讨了情感智能在医疗保健、教育和客户服务等领域的影响，同时考虑不同文化背景下的情感表达差异以及不同地理人口群体中情感识别系统潜在偏见的挑战。从监管角度来看，文章在GDPR和欧盟AI法案的框架下探讨情感数据，强调情感数据可能被视为敏感个人数据，需要包括目的限制、数据最小化和有意义的同意机制在内的严格保护措施。 

---
# EchoBench: Benchmarking Sycophancy in Medical Large Vision-Language Models 

**Title (ZH)**: EchoBench: 医学大规模视觉语言模型中的奉承行为基准测试 

**Authors**: Botai Yuan, Yutian Zhou, Yingjie Wang, Fushuo Huo, Yongcheng Jing, Li Shen, Ying Wei, Zhiqi Shen, Ziwei Liu, Tianwei Zhang, Jie Yang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2509.20146)  

**Abstract**: Recent benchmarks for medical Large Vision-Language Models (LVLMs) emphasize leaderboard accuracy, overlooking reliability and safety. We study sycophancy -- models' tendency to uncritically echo user-provided information -- in high-stakes clinical settings. We introduce EchoBench, a benchmark to systematically evaluate sycophancy in medical LVLMs. It contains 2,122 images across 18 departments and 20 modalities with 90 prompts that simulate biased inputs from patients, medical students, and physicians. We evaluate medical-specific, open-source, and proprietary LVLMs. All exhibit substantial sycophancy; the best proprietary model (Claude 3.7 Sonnet) still shows 45.98% sycophancy, and GPT-4.1 reaches 59.15%. Many medical-specific models exceed 95% sycophancy despite only moderate accuracy. Fine-grained analyses by bias type, department, perceptual granularity, and modality identify factors that increase susceptibility. We further show that higher data quality/diversity and stronger domain knowledge reduce sycophancy without harming unbiased accuracy. EchoBench also serves as a testbed for mitigation: simple prompt-level interventions (negative prompting, one-shot, few-shot) produce consistent reductions and motivate training- and decoding-time strategies. Our findings highlight the need for robust evaluation beyond accuracy and provide actionable guidance toward safer, more trustworthy medical LVLMs. 

**Abstract (ZH)**: 近期医学大规模视觉-语言模型的基准测试侧重于排行榜准确度，忽视了可靠性和安全性。我们研究了奉承行为——模型倾向于无批判地重复用户提供的信息——在高风险临床环境中的表现。我们引入了EchoBench，这是一个系统评估医学大规模视觉-语言模型奉承行为的基准测试。其中包含18个部门和20种模态的2,122张图像，以及90个模拟患者、医学学生和医生偏见输入的提示。我们评估了医学特定、开源和专有模型。所有模型都表现出了显著的奉承行为；最好的专有模型（Claude 3.7 Sonnet）仍显示出45.98%的奉承行为，GPT-4.1则达到59.15%。尽管准确度只有中等水平，许多医学特定模型的奉承行为超过95%。通过对偏见类型、部门、知觉粒度和模态的细粒度分析，我们识别了增加易感性的因素。进一步研究表明，高质量/多样性的数据和更强的专业知识可以在不损害无偏准确度的情况下减少奉承行为。EchoBench 也作为缓解措施的测试平台：简单的提示级干预（负面提示、一次示例、少数示例）产生了一致的减少效果，并激励训练时和解码时策略。我们的研究结果强调了超越准确度的稳健评估的重要性，并提供了实现更安全、更可信的医学大规模视觉-语言模型的支持性指导。 

---
# KSDiff: Keyframe-Augmented Speech-Aware Dual-Path Diffusion for Facial Animation 

**Title (ZH)**: KSDiff: 增强关键帧的语音感知双路径扩散 Facial 动画 

**Authors**: Tianle Lyu, Junchuan Zhao, Ye Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20128)  

**Abstract**: Audio-driven facial animation has made significant progress in multimedia applications, with diffusion models showing strong potential for talking-face synthesis. However, most existing works treat speech features as a monolithic representation and fail to capture their fine-grained roles in driving different facial motions, while also overlooking the importance of modeling keyframes with intense dynamics. To address these limitations, we propose KSDiff, a Keyframe-Augmented Speech-Aware Dual-Path Diffusion framework. Specifically, the raw audio and transcript are processed by a Dual-Path Speech Encoder (DPSE) to disentangle expression-related and head-pose-related features, while an autoregressive Keyframe Establishment Learning (KEL) module predicts the most salient motion frames. These components are integrated into a Dual-path Motion generator to synthesize coherent and realistic facial motions. Extensive experiments on HDTF and VoxCeleb demonstrate that KSDiff achieves state-of-the-art performance, with improvements in both lip synchronization accuracy and head-pose naturalness. Our results highlight the effectiveness of combining speech disentanglement with keyframe-aware diffusion for talking-head generation. 

**Abstract (ZH)**: 基于音频的面部动画在多媒体应用中取得了显著进展，扩散模型在对话面部合成方面显示出强大的潜力。然而，大多数现有工作将语音特征视为整体表示，未能捕捉其在驱动不同面部动作中的细微作用，同时忽视了建模具有强烈动力学的关键帧的重要性。为解决这些问题，我们提出了一种关键帧增强的语音意识双路径扩散框架KSDiff。具体而言，原始音频和转录文本由双路径语音编码器（DPSE）处理，以分离表情相关和头部姿态相关特征，而自回归关键帧建立学习（KEL）模块预测最具显著性的运动帧。这些组件被整合到双路径运动生成器中，以合成连贯且真实的面部动作。在HDTF和VoxCeleb上的广泛实验表明，KSDiff达到了最先进的性能，both在唇同步准确性和头部姿态自然度上有所改进。我们的结果突显了结合语音分离与关键帧意识扩散对生成对话头部的有效性。 

---
# Discovering Association Rules in High-Dimensional Small Tabular Data 

**Title (ZH)**: 在高维小表数据中发现关联规则 

**Authors**: Erkan Karabulut, Daniel Daza, Paul Groth, Victoria Degeler  

**Link**: [PDF](https://arxiv.org/pdf/2509.20113)  

**Abstract**: Association Rule Mining (ARM) aims to discover patterns between features in datasets in the form of propositional rules, supporting both knowledge discovery and interpretable machine learning in high-stakes decision-making. However, in high-dimensional settings, rule explosion and computational overhead render popular algorithmic approaches impractical without effective search space reduction, challenges that propagate to downstream tasks. Neurosymbolic methods, such as Aerial+, have recently been proposed to address the rule explosion in ARM. While they tackle the high dimensionality of the data, they also inherit limitations of neural networks, particularly reduced performance in low-data regimes.
This paper makes three key contributions to association rule discovery in high-dimensional tabular data. First, we empirically show that Aerial+ scales one to two orders of magnitude better than state-of-the-art algorithmic and neurosymbolic baselines across five real-world datasets. Second, we introduce the novel problem of ARM in high-dimensional, low-data settings, such as gene expression data from the biomedicine domain with around 18k features and 50 samples. Third, we propose two fine-tuning approaches to Aerial+ using tabular foundation models. Our proposed approaches are shown to significantly improve rule quality on five real-world datasets, demonstrating their effectiveness in low-data, high-dimensional scenarios. 

**Abstract (ZH)**: 高维表格数据中关联规则发现的神经符号方法研究 

---
# Discrete Diffusion for Reflective Vision-Language-Action Models in Autonomous Driving 

**Title (ZH)**: 离散扩散在自主驾驶反射型 vision-language-action 模型中的应用 

**Authors**: Pengxiang Li, Yinan Zheng, Yue Wang, Huimin Wang, Hang Zhao, Jingjing Liu, Xianyuan Zhan, Kun Zhan, Xianpeng Lang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20109)  

**Abstract**: End-to-End (E2E) solutions have emerged as a mainstream approach for autonomous driving systems, with Vision-Language-Action (VLA) models representing a new paradigm that leverages pre-trained multimodal knowledge from Vision-Language Models (VLMs) to interpret and interact with complex real-world environments. However, these methods remain constrained by the limitations of imitation learning, which struggles to inherently encode physical rules during training. Existing approaches often rely on complex rule-based post-refinement, employ reinforcement learning that remains largely limited to simulation, or utilize diffusion guidance that requires computationally expensive gradient calculations. To address these challenges, we introduce ReflectDrive, a novel learning-based framework that integrates a reflection mechanism for safe trajectory generation via discrete diffusion. We first discretize the two-dimensional driving space to construct an action codebook, enabling the use of pre-trained Diffusion Language Models for planning tasks through fine-tuning. Central to our approach is a safety-aware reflection mechanism that performs iterative self-correction without gradient computation. Our method begins with goal-conditioned trajectory generation to model multi-modal driving behaviors. Based on this, we apply local search methods to identify unsafe tokens and determine feasible solutions, which then serve as safe anchors for inpainting-based regeneration. Evaluated on the NAVSIM benchmark, ReflectDrive demonstrates significant advantages in safety-critical trajectory generation, offering a scalable and reliable solution for autonomous driving systems. 

**Abstract (ZH)**: 端到端（E2E）解决方案已成为自主驾驶系统的主要方法，Vision-Language-Action（VLA）模型作为一种新的范式，利用预训练的多模态知识从视觉语言模型（VLMs）中解释和交互复杂的现实环境。然而，这些方法仍受模仿学习限制的约束，在训练过程中难以内在编码物理规则。现有方法通常依赖于复杂的基于规则的后精修，或者采用在模拟中仍然基本受限的强化学习，或利用需要昂贵梯度计算的支持扩散指导。为了解决这些挑战，我们引入了ReflectDrive，这是一种新颖的学习框架，它通过离散扩散集成了一个反射机制以实现安全轨迹生成。我们首先离散化二维驾驶空间以构建动作词典，并通过微调使预训练的扩散语言模型能够用于规划任务。我们方法的核心是一个安全意识反射机制，它在不进行梯度计算的情况下进行迭代自我校正。该方法从基于目标的轨迹生成开始，以建模多模态驾驶行为。在此基础上，我们应用局部搜索方法来识别不安全的令牌并确定可行的解决方案，这些解决方案随后作为基于填充生成的安全锚点。在NAVSIM基准上评估，ReflectDrive在安全关键轨迹生成方面表现出显著优势，为自主驾驶系统提供了可扩展且可靠的方法。 

---
# Hyperspectral Adapter for Semantic Segmentation with Vision Foundation Models 

**Title (ZH)**: 视觉基础模型下的高光谱适配器用于语义分割 

**Authors**: JuanaJuana Valeria Hurtado, Rohit Mohan, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2509.20107)  

**Abstract**: Hyperspectral imaging (HSI) captures spatial information along with dense spectral measurements across numerous narrow wavelength bands. This rich spectral content has the potential to facilitate robust robotic perception, particularly in environments with complex material compositions, varying illumination, or other visually challenging conditions. However, current HSI semantic segmentation methods underperform due to their reliance on architectures and learning frameworks optimized for RGB inputs. In this work, we propose a novel hyperspectral adapter that leverages pretrained vision foundation models to effectively learn from hyperspectral data. Our architecture incorporates a spectral transformer and a spectrum-aware spatial prior module to extract rich spatial-spectral features. Additionally, we introduce a modality-aware interaction block that facilitates effective integration of hyperspectral representations and frozen vision Transformer features through dedicated extraction and injection mechanisms. Extensive evaluations on three benchmark autonomous driving datasets demonstrate that our architecture achieves state-of-the-art semantic segmentation performance while directly using HSI inputs, outperforming both vision-based and hyperspectral segmentation methods. We make the code available at this https URL. 

**Abstract (ZH)**: 高光谱成像（HSI）能够捕获空间信息以及在众多窄波长 band 上的密集光谱测量值。这些丰富的光谱内容有潜力在复杂材料组成、不同照明条件或其它视觉挑战性环境中促进稳健的机器人感知。然而，现有的HSI语义分割方法由于依赖于优化用于RGB输入的架构和学习框架而表现不佳。在本工作中，我们提出了一种新颖的高光谱适配器，利用预训练的视觉基础模型有效学习高光谱数据。我们的架构包含一个光谱变换器和一种光谱感知的空间先验模块，以提取丰富的空间-光谱特征。此外，我们引入了一种模态感知交互块，通过专用的提取和注入机制，促进高光谱表示与冻结的视觉Transformer特征的有效集成。在三个基准自主驾驶数据集上的广泛评估表明，我们的架构能够直接使用HSI输入实现最先进的语义分割性能，优于基于视觉和高光谱分割方法。我们已在以下链接提供代码：this https URL。 

---
# Integrated Framework for LLM Evaluation with Answer Generation 

**Title (ZH)**: LLM评估与答案生成集成框架 

**Authors**: Sujeong Lee, Hayoung Lee, Seongsoo Heo, Wonik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2509.20097)  

**Abstract**: Reliable evaluation of large language models is essential to ensure their applicability in practical scenarios. Traditional benchmark-based evaluation methods often rely on fixed reference answers, limiting their ability to capture important qualitative aspects of generated responses. To address these shortcomings, we propose an integrated evaluation framework called \textit{self-refining descriptive evaluation with expert-driven diagnostics}, SPEED, which utilizes specialized functional experts to perform comprehensive, descriptive analyses of model outputs. Unlike conventional approaches, SPEED actively incorporates expert feedback across multiple dimensions, including hallucination detection, toxicity assessment, and lexical-contextual appropriateness. Experimental results demonstrate that SPEED achieves robust and consistent evaluation performance across diverse domains and datasets. Additionally, by employing relatively compact expert models, SPEED demonstrates superior resource efficiency compared to larger-scale evaluators. These findings illustrate that SPEED significantly enhances fairness and interpretability in LLM evaluations, offering a promising alternative to existing evaluation methodologies. 

**Abstract (ZH)**: 可靠的大型语言模型评估对于确保其在实际场景中的应用至关重要。传统的基于基准的评估方法往往依赖于固定的参考答案，限制了其捕捉生成响应的重要定性方面的能力。为了解决这些问题，我们提出了一种名为自我完善描述性评估与专家驱动诊断的集成评估框架SPEED，该框架利用专门的功能专家对模型输出进行全面、描述性的分析。与传统方法不同，SPEED在多个维度上积极 Incorporates 专家反馈，包括幻觉检测、毒性评估和词法-语境适宜性。实验结果表明，SPEED 在多种领域和数据集上实现了稳健且一致的评估性能。此外，通过采用相对紧凑的专家模型，SPEED 在资源效率上优于大型评估器。这些发现表明，SPEED 显著提高了大型语言模型评估的公平性和可解释性，提供了现有评估方法的一个有前景的替代方案。 

---
# Causal Understanding by LLMs: The Role of Uncertainty 

**Title (ZH)**: LLMs中的因果理解：不确定性的作用 

**Authors**: Oscar Lithgow-Serrano, Vani Kanjirangat, Alessandro Antonucci  

**Link**: [PDF](https://arxiv.org/pdf/2509.20088)  

**Abstract**: Recent papers show LLMs achieve near-random accuracy in causal relation classification, raising questions about whether such failures arise from limited pretraining exposure or deeper representational gaps. We investigate this under uncertainty-based evaluation, testing whether pretraining exposure to causal examples improves causal understanding >18K PubMed sentences -- half from The Pile corpus, half post-2024 -- across seven models (Pythia-1.4B/7B/12B, GPT-J-6B, Dolly-7B/12B, Qwen-7B). We analyze model behavior through: (i) causal classification, where the model identifies causal relationships in text, and (ii) verbatim memorization probing, where we assess whether the model prefers previously seen causal statements over their paraphrases. Models perform four-way classification (direct/conditional/correlational/no-relationship) and select between originals and their generated paraphrases. Results show almost identical accuracy on seen/unseen sentences (p > 0.05), no memorization bias (24.8% original selection), and output distribution over the possible options is almost flat, with entropic values near the maximum (1.35/1.39), confirming random guessing. Instruction-tuned models show severe miscalibration (Qwen: > 95% confidence, 32.8% accuracy, ECE=0.49). Conditional relations induce highest entropy (+11% vs. direct). These findings suggest that failures in causal understanding arise from the lack of structured causal representation, rather than insufficient exposure to causal examples during pretraining. 

**Abstract (ZH)**: Recent 论文显示大语言模型在因果关系分类上的准确率接近随机，引发了对其失败是源于有限的预训练暴露还是深层表征gap的质疑。我们通过基于不确定性的评估进行研究，测试预训练中因果示例的暴露是否能改善七种模型（Pythia-1.4B/7B/12B、GPT-J-6B、Dolly-7B/12B、Qwen-7B）在18,000多个PubMed句子中的因果理解能力——其中一半来自The Pile语料库，一半来自2024年以后的句子。我们通过以下两种方式分析模型的行为：（i）因果关系分类，模型在文本中识别因果关系；（ii）逐字记忆探针，评估模型是否偏好之前见过的因果陈述而非它们的同义说法。模型进行四分类（直接/条件/相关/无关系）并选择原始陈述或其生成的同义说法。结果显示，在已见和未见句子上的准确率几乎相同（p > 0.05），无记忆偏见（24.8%原始选择），输出分布几乎均匀，熵值接近最大值（1.35/1.39），证实为随机猜测。指令微调模型显示出严重的校准偏差（Qwen：> 95%置信度，32.8%准确率，ECE=0.49）。条件关系引起的熵值最高（+11% vs. 直接）。这些发现表明，在因果理解上的失败源于缺乏结构化的因果表征，而不是预训练中对因果示例的不足暴露。 

---
# Responsible AI Technical Report 

**Title (ZH)**: 负责任人工智能技术报告 

**Authors**: Soonmin Bae, Wanjin Park, Jeongyeop Kim, Yunjin Park, Jungwon Yoon, Junhyung Moon, Myunggyo Oh, Wonhyuk Lee, Junseo Jang, Dongyoung Jung, Minwook Ju, Eunmi Kim, Sujin Kim, Youngchol Kim, Somin Lee, Wonyoung Lee, Minsung Noh, Hyoungjun Park, Eunyoung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2509.20057)  

**Abstract**: KT developed a Responsible AI (RAI) assessment methodology and risk mitigation technologies to ensure the safety and reliability of AI services. By analyzing the Basic Act on AI implementation and global AI governance trends, we established a unique approach for regulatory compliance and systematically identify and manage all potential risk factors from AI development to operation. We present a reliable assessment methodology that systematically verifies model safety and robustness based on KT's AI risk taxonomy tailored to the domestic environment. We also provide practical tools for managing and mitigating identified AI risks. With the release of this report, we also release proprietary Guardrail : SafetyGuard that blocks harmful responses from AI models in real-time, supporting the enhancement of safety in the domestic AI development ecosystem. We also believe these research outcomes provide valuable insights for organizations seeking to develop Responsible AI. 

**Abstract (ZH)**: KT开发了一种负责任的人工智能(RAI)评估方法和风险缓解技术，以确保人工智能服务的安全性和可靠性。通过分析《人工智能实施基本法》和全球人工智能治理趋势，我们建立了一种独特的合规方法，系统地识别和管理从人工智能开发到运营的所有潜在风险因素。我们提出了一种可靠的评估方法，基于KT为国内环境定制的人工智能风险分类，系统地验证模型的安全性和 robustness。我们还提供了管理和缓解已识别AI风险的实用工具。随着这份报告的发布，我们还推出了 proprietary Guardrail：SafetyGuard，这是一种实时阻止有害人工智能模型响应的工具，支持国内人工智能开发生态系统安全性提升。我们认为这些研究成果为寻求开发负责任人工智能的组织提供了宝贵的见解。 

---
# One Filters All: A Generalist Filter for State Estimation 

**Title (ZH)**: 万能滤波器：一种通用的状态估计滤波器 

**Authors**: Shiqi Liu, Wenhan Cao, Chang Liu, Zeyu He, Tianyi Zhang, Shengbo Eben Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.20051)  

**Abstract**: Estimating hidden states in dynamical systems, also known as optimal filtering, is a long-standing problem in various fields of science and engineering. In this paper, we introduce a general filtering framework, \textbf{LLM-Filter}, which leverages large language models (LLMs) for state estimation by embedding noisy observations with text prototypes. In various experiments for classical dynamical systems, we find that first, state estimation can significantly benefit from the reasoning knowledge embedded in pre-trained LLMs. By achieving proper modality alignment with the frozen LLM, LLM-Filter outperforms the state-of-the-art learning-based approaches. Second, we carefully design the prompt structure, System-as-Prompt (SaP), incorporating task instructions that enable the LLM to understand the estimation tasks. Guided by these prompts, LLM-Filter exhibits exceptional generalization, capable of performing filtering tasks accurately in changed or even unseen environments. We further observe a scaling-law behavior in LLM-Filter, where accuracy improves with larger model sizes and longer training times. These findings make LLM-Filter a promising foundation model of filtering. 

**Abstract (ZH)**: 利用大型语言模型进行动力系统隐状态估算的LLM-Filter框架 

---
# Projective Kolmogorov Arnold Neural Networks (P-KANs): Entropy-Driven Functional Space Discovery for Interpretable Machine Learning 

**Title (ZH)**: 投影柯尔莫哥洛夫阿诺尔德神经网络(P-KANs):基于熵驱动的功能空间发现可解释机器学习 

**Authors**: Alastair Poole, Stig McArthur, Saravan Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.20049)  

**Abstract**: Kolmogorov-Arnold Networks (KANs) relocate learnable nonlinearities from nodes to edges, demonstrating remarkable capabilities in scientific machine learning and interpretable modeling. However, current KAN implementations suffer from fundamental inefficiencies due to redundancy in high-dimensional spline parameter spaces, where numerous distinct parameterisations yield functionally equivalent behaviors. This redundancy manifests as a "nuisance space" in the model's Jacobian, leading to susceptibility to overfitting and poor generalization. We introduce Projective Kolmogorov-Arnold Networks (P-KANs), a novel training framework that guides edge function discovery towards interpretable functional representations through entropy-minimisation techniques from signal analysis and sparse dictionary learning. Rather than constraining functions to predetermined spaces, our approach maintains spline space flexibility while introducing "gravitational" terms that encourage convergence towards optimal functional representations. Our key insight recognizes that optimal representations can be identified through entropy analysis of projection coefficients, compressing edge functions to lower-parameter projective spaces (Fourier, Chebyshev, Bessel). P-KANs demonstrate superior performance across multiple domains, achieving up to 80% parameter reduction while maintaining representational capacity, significantly improved robustness to noise compared to standard KANs, and successful application to industrial automated fiber placement prediction. Our approach enables automatic discovery of mixed functional representations where different edges converge to different optimal spaces, providing both compression benefits and enhanced interpretability for scientific machine learning applications. 

**Abstract (ZH)**: (projective) Kolmogorov-Arnold Networks (P-KANs): Entropy-Minimisation for Interpretable and Efficient Scientific Machine Learning 

---
# Diffusion-Augmented Contrastive Learning: A Noise-Robust Encoder for Biosignal Representations 

**Title (ZH)**: 扩散增强对比学习：一种生物信号表示的噪声鲁棒编码器 

**Authors**: Rami Zewail  

**Link**: [PDF](https://arxiv.org/pdf/2509.20048)  

**Abstract**: Learning robust representations for biosignals is often hampered by the challenge of designing effective data this http URL methods can fail to capture the complex variations inherent in physiological data. Within this context, we propose a novel hybrid framework, Diffusion-Augmented Contrastive Learning (DACL), that fuses concepts from diffusion models and supervised contrastive learning. The DACL framework operates on a latent space created by a lightweight Variational Autoencoder (VAE) trained on our novel Scattering Transformer (ST) features [12]. It utilizes the diffusion forward process as a principled data augmentation technique to generate multiple noisy views of these latent embeddings. A U-Net style encoder is then trained with a supervised contrastive objective to learn a representation that balances class discrimination with robustness to noise across various diffusion time steps. We evaluated this proof-of-concept method on the PhysioNet 2017 ECG dataset, achieving a competitive AUROC of 0.7815. This work establishes a new paradigm for representation learning by using the diffusion process itself to drive the contrastive objective, creating noise-invariant embeddings that demonstrate a strong foundation for class separability. 

**Abstract (ZH)**: 基于扩散增广对比学习的生物信号鲁棒表示学习 

---
# Tokenization and Representation Biases in Multilingual Models on Dialectal NLP Tasks 

**Title (ZH)**: 多语模型在方言NLP任务中的分词和表示偏见 

**Authors**: Vani Kanjirangat, Tanja Samardžić, Ljiljana Dolamic, Fabio Rinaldi  

**Link**: [PDF](https://arxiv.org/pdf/2509.20045)  

**Abstract**: Dialectal data are characterized by linguistic variation that appears small to humans but has a significant impact on the performance of models. This dialect gap has been related to various factors (e.g., data size, economic and social factors) whose impact, however, turns out to be inconsistent. In this work, we investigate factors impacting the model performance more directly: we correlate Tokenization Parity (TP) and Information Parity (IP), as measures of representational biases in pre-trained multilingual models, with the downstream performance. We compare state-of-the-art decoder-only LLMs with encoder-based models across three tasks: dialect classification, topic classification, and extractive question answering, controlling for varying scripts (Latin vs. non-Latin) and resource availability (high vs. low). Our analysis reveals that TP is a better predictor of the performance on tasks reliant on syntactic and morphological cues (e.g., extractive QA), while IP better predicts performance in semantic tasks (e.g., topic classification). Complementary analyses, including tokenizer behavior, vocabulary coverage, and qualitative insights, reveal that the language support claims of LLMs often might mask deeper mismatches at the script or token level. 

**Abstract (ZH)**: 方言数据的特点在于人类看来较小的语言变异，但对模型性能有显著影响。这种方言差距与多种因素（如数据量、经济和社会因素）有关，然而这些因素的影响却并不一致。在这项工作中，我们直接调查影响模型性能的因素：我们通过Tokenization Parity (TP) 和 Information Parity (IP) 这两种衡量预训练多语言模型表示偏差的指标，来与下游性能进行关联。我们将最先进的解码器-only大型语言模型与编码器-基于模型在三种任务（方言分类、话题分类和抽取型问答）上进行了比较，控制了不同脚本（拉丁 versus 非拉丁）和资源可用性（高 versus 低）的差异。我们的分析表明，TP 更好地预测了依赖句法和形态线索的任务（例如，抽取型问答）的性能，而IP 更好地预测了语义任务（例如，话题分类）的性能。补充性分析，包括标记器行为、词汇覆盖范围以及定性见解，揭示了大型语言模型的语言支持声明有时可能掩盖了更深层次的脚本或标记级别上的不匹配。 

---
# Generative Adversarial Networks Applied for Privacy Preservation in Biometric-Based Authentication and Identification 

**Title (ZH)**: 基于生物特征识别的隐私保护生成对抗网络应用 

**Authors**: Lubos Mjachky, Ivan Homoliak  

**Link**: [PDF](https://arxiv.org/pdf/2509.20024)  

**Abstract**: Biometric-based authentication systems are getting broadly adopted in many areas. However, these systems do not allow participating users to influence the way their data is used. Furthermore, the data may leak and can be misused without the users' knowledge. In this paper, we propose a new authentication method that preserves the privacy of individuals and is based on a generative adversarial network (GAN). Concretely, we suggest using the GAN for translating images of faces to a visually private domain (e.g., flowers or shoes). Classifiers, which are used for authentication purposes, are then trained on the images from the visually private domain. Based on our experiments, the method is robust against attacks and still provides meaningful utility. 

**Abstract (ZH)**: 基于生物特征的身份认证系统已在许多领域广泛采用。然而，这些系统不允许参与者影响其数据的使用方式，且数据可能会泄漏并被不知情的滥用。在本文中，我们提出了一种新的身份认证方法，该方法保护个人隐私，并基于生成式对抗网络（GAN）。具体而言，我们建议使用GAN将面部图像转换为视觉隐私域（例如花朵或鞋子），然后用于身份验证目的的分类器在来自视觉隐私域的图像上进行训练。根据我们的实验，该方法对攻击具有鲁棒性，并仍能提供有意义的实用性。 

---
# The Knowledge-Behaviour Disconnect in LLM-based Chatbots 

**Title (ZH)**: 基于大型语言模型的聊天机器人中的知识-行为断层 

**Authors**: Jan Broersen  

**Link**: [PDF](https://arxiv.org/pdf/2509.20004)  

**Abstract**: Large language model-based artificial conversational agents (like ChatGPT) give answers to all kinds of questions, and often enough these answers are correct. Just on the basis of that capacity alone, we may attribute knowledge to them. But do these models use this knowledge as a basis for their own conversational behaviour? I argue this is not the case, and I will refer to this failure as a `disconnect'. I further argue this disconnect is fundamental in the sense that with more data and more training of the LLM on which a conversational chatbot is based, it will not disappear. The reason is, as I will claim, that the core technique used to train LLMs does not allow for the establishment of the connection we are after. The disconnect reflects a fundamental limitation on the capacities of LLMs, and explains the source of hallucinations. I will furthermore consider the ethical version of the disconnect (ethical conversational knowledge not being aligned with ethical conversational behaviour), since in this domain researchers have come up with several additional techniques to influence a chatbot's behaviour. I will discuss how these techniques do nothing to solve the disconnect and can make it worse. 

**Abstract (ZH)**: 基于大型语言模型的拟人化对话代理（如ChatGPT）能够回答各种问题，其答案往往正确。仅凭这一能力，我们可能会赋予它们知识。但这些模型是否将知识作为自己对话行为的基础？我认为并不是这样，我将这种失败称为“断裂”。进一步而言，我认为这种断裂是根本性的，因为即使有更多的数据和更长时间的训练，基于这些大型语言模型的对话聊天机器人也不会消除这种断裂。原因是，正如我将声称的，用于训练大型语言模型的核心技术不允许可供我们所需的连接建立。断裂反映了大型语言模型能力的根本限制，并解释了幻觉的来源。此外，我还探讨了伦理版本的断裂问题（即伦理对话知识与伦理对话行为不一致），因为在这一领域，研究人员已经开发出多种额外的技术来影响聊天机器人的行为。我将讨论这些技术如何无法解决问题，甚至可能使其恶化。 

---
# Table Detection with Active Learning 

**Title (ZH)**: 主动学习的表格检测 

**Authors**: Somraj Gautam, Nachiketa Purohit, Gaurav Harit  

**Link**: [PDF](https://arxiv.org/pdf/2509.20003)  

**Abstract**: Efficient data annotation remains a critical challenge in machine learning, particularly for object detection tasks requiring extensive labeled data. Active learning (AL) has emerged as a promising solution to minimize annotation costs by selecting the most informative samples. While traditional AL approaches primarily rely on uncertainty-based selection, recent advances suggest that incorporating diversity-based strategies can enhance sampling efficiency in object detection tasks. Our approach ensures the selection of representative examples that improve model generalization. We evaluate our method on two benchmark datasets (TableBank-LaTeX, TableBank-Word) using state-of-the-art table detection architectures, CascadeTabNet and YOLOv9. Our results demonstrate that AL-based example selection significantly outperforms random sampling, reducing annotation effort given a limited budget while maintaining comparable performance to fully supervised models. Our method achieves higher mAP scores within the same annotation budget. 

**Abstract (ZH)**: 高效的数据标注仍然是机器学习中的一个关键挑战，特别是在需要大量标注数据的对象检测任务中。主动学习(AL)已成为一种有前景的解决方案，通过选择最具信息量的样本来最小化标注成本。虽然传统的AL方法主要依赖于基于不确定性的选择，但最近的研究表明，在对象检测任务中结合多样性策略可以提高采样效率。我们的方法确保选择出能提高模型泛化能力的代表性样本。我们在两个基准数据集(TableBank-LaTeX, TableBank-Word)上使用最先进的表格检测架构(CascadeTabNet和YOLOv9)评估了我们的方法。实验结果表明，基于AL的样本选择明显优于随机采样，在有限的标注预算下减少了标注努力，并且在性能上与完全监督的模型相当。我们的方法在相同的标注预算内实现了更高的mAP分数。 

---
# Choosing to Be Green: Advancing Green AI via Dynamic Model Selection 

**Title (ZH)**: 选择绿色之路：通过动态模型选择推动绿色AI的发展 

**Authors**: Emilio Cruciani, Roberto Verdecchia  

**Link**: [PDF](https://arxiv.org/pdf/2509.19996)  

**Abstract**: Artificial Intelligence is increasingly pervasive across domains, with ever more complex models delivering impressive predictive performance. This fast technological advancement however comes at a concerning environmental cost, with state-of-the-art models - particularly deep neural networks and large language models - requiring substantial computational resources and energy. In this work, we present the intuition of Green AI dynamic model selection, an approach based on dynamic model selection that aims at reducing the environmental footprint of AI by selecting the most sustainable model while minimizing potential accuracy loss. Specifically, our approach takes into account the inference task, the environmental sustainability of available models, and accuracy requirements to dynamically choose the most suitable model. Our approach presents two different methods, namely Green AI dynamic model cascading and Green AI dynamic model routing. We demonstrate the effectiveness of our approach via a proof of concept empirical example based on a real-world dataset. Our results show that Green AI dynamic model selection can achieve substantial energy savings (up to ~25%) while substantially retaining the accuracy of the most energy greedy solution (up to ~95%). As conclusion, our preliminary findings highlight the potential that hybrid, adaptive model selection strategies withhold to mitigate the energy demands of modern AI systems without significantly compromising accuracy requirements. 

**Abstract (ZH)**: 绿色人工智能动态模型选择：降低环境足迹的同时保留预测准确性 

---
# SDE-DET: A Precision Network for Shatian Pomelo Detection in Complex Orchard Environments 

**Title (ZH)**: SDE-DET：复杂果园环境中的沙田柚精准检测网络 

**Authors**: Yihao Hu, Pan Wang, Xiaodong Bai, Shijie Cai, Hang Wang, Huazhong Liu, Aiping Yang, Xiangxiang Li, Meiping Ding, Hongyan Liu, Jianguo Yao  

**Link**: [PDF](https://arxiv.org/pdf/2509.19990)  

**Abstract**: Pomelo detection is an essential process for their localization, automated robotic harvesting, and maturity analysis. However, detecting Shatian pomelo in complex orchard environments poses significant challenges, including multi-scale issues, obstructions from trunks and leaves, small object detection, etc. To address these issues, this study constructs a custom dataset STP-AgriData and proposes the SDE-DET model for Shatian pomelo detection. SDE-DET first utilizes the Star Block to effectively acquire high-dimensional information without increasing the computational overhead. Furthermore, the presented model adopts Deformable Attention in its backbone, to enhance its ability to detect pomelos under occluded conditions. Finally, multiple Efficient Multi-Scale Attention mechanisms are integrated into our model to reduce the computational overhead and extract deep visual representations, thereby improving the capacity for small object detection. In the experiment, we compared SDE-DET with the Yolo series and other mainstream detection models in Shatian pomelo detection. The presented SDE-DET model achieved scores of 0.883, 0.771, 0.838, 0.497, and 0.823 in Precision, Recall, mAP@0.5, mAP@0.5:0.95 and F1-score, respectively. SDE-DET has achieved state-of-the-art performance on the STP-AgriData dataset. Experiments indicate that the SDE-DET provides a reliable method for Shatian pomelo detection, laying the foundation for the further development of automatic harvest robots. 

**Abstract (ZH)**: Shatian柚检测：STP-AgriData数据集与SDE-DET模型的研究 

---
# An effective control of large systems of active particles: An application to evacuation problem 

**Title (ZH)**: 大型活性粒子系统的有效控制：以 evacuation 问题为例 

**Authors**: Albina Klepach, Egor E. Nuzhin, Alexey A. Tsukanov, Nikolay V. Brilliantov  

**Link**: [PDF](https://arxiv.org/pdf/2509.19972)  

**Abstract**: Manipulation of large systems of active particles is a serious challenge across diverse domains, including crowd management, control of robotic swarms, and coordinated material transport. The development of advanced control strategies for complex scenarios is hindered, however, by the lack of scalability and robustness of the existing methods, in particular, due to the need of an individual control for each agent. One possible solution involves controlling a system through a leader or a group of leaders, which other agents tend to follow. Using such an approach we develop an effective control strategy for a leader, combining reinforcement learning (RL) with artificial forces acting on the system. To describe the guidance of active particles by a leader we introduce the generalized Vicsek model. This novel method is then applied to the problem of the effective evacuation by a robot-rescuer (leader) of large groups of people from hazardous places. We demonstrate, that while a straightforward application of RL yields suboptimal results, even for advanced architectures, our approach provides a robust and efficient evacuation strategy. The source code supporting this study is publicly available at: this https URL. 

**Abstract (ZH)**: 大规模活性粒子系统的操控在 crowd management、robotic swarms 控制和协调物质运输等领域是一个严重挑战。现有的方法由于缺乏可扩展性和鲁棒性，特别是在需要为每个代理单独控制时，限制了复杂场景下先进控制策略的发展。一种可能的解决方案是通过领导者或一组领导者操控系统，其他代理倾向于跟随领导者。利用这种方法，我们结合强化学习（RL）和作用于系统的虚拟力，开发出一种有效的领导者控制策略。为了描述领导者对活性粒子的引导，我们引入了广义 Vicsek 模型。然后，我们将该新型方法应用于机器人救援者（领导者）有效疏散大量人群远离危险场所的问题。我们证明，尽管直接应用 RL 能力有限，甚至对于先进的架构也是如此，我们的方法提供了更为稳健和高效的疏散策略。支持本研究的源代码可在以下网址获取：this https URL。 

---
# 2025 Southeast Asia Eleven Nations Influence Index Report 

**Title (ZH)**: 2025年东南亚十一国影响力指数报告 

**Authors**: Wei Meng  

**Link**: [PDF](https://arxiv.org/pdf/2509.19953)  

**Abstract**: This study constructs a fully data-driven and reproducible Southeast Asia Influence Index (SAII v3) to reduce bias from expert scoring and subjective weighting while mapping hierarchical power structures across the eleven ASEAN nations. We aggregate authoritative open-source indicators across four dimensions (economic, military, diplomatic, socio-technological) and apply a three-tiered standardization chain quantile-Box-Cox-min-max to mitigate outliers and skewness. Weights are obtained through equal-weight integration of Entropy Weighting Method (EWM), CRITIC, and PCA. Robustness is assessed via Kendall's tau, +/-20% weight perturbation, and 10,000 bootstrap iterations, with additional checks including +/-10% dimensional sensitivity and V2-V3 bump chart comparisons. Results show integrated weights: Economy 35-40%, Military 20-25%, Diplomacy about 20%, Socio-Technology about 15%. The regional landscape exhibits a one-strong, two-medium, three-stable, and multiple-weak pattern: Indonesia, Singapore, and Malaysia lead, while Thailand, the Philippines, and Vietnam form a mid-tier competitive band. V2 and V3 rankings are highly consistent (Kendall's tau = 0.818), though small mid-tier reorderings appear (Thailand and the Philippines rise, Vietnam falls), indicating that v3 is more sensitive to structural equilibrium. ASEAN-11 average sensitivity highlights military and socio-technological dimensions as having the largest marginal effects (+/-0.002). In conclusion, SAII v3 delivers algorithmic weighting and auditable reproducibility, reveals multidimensional drivers of influence in Southeast Asia, and provides actionable quantitative evidence for resource allocation and policy prioritization by regional governments and external partners. 

**Abstract (ZH)**: 东南亚影响指数（SAII v3）：基于完全数据驱动和可重复的方法减少专家评分偏见和主观权重，描绘十一国 ASEAN 的层级权力结构 

---
# When Words Can't Capture It All: Towards Video-Based User Complaint Text Generation with Multimodal Video Complaint Dataset 

**Title (ZH)**: 当文字无法表达一切：面向多模态视频投诉数据集的视频基用户投诉文本生成研究 

**Authors**: Sarmistha Das, R E Zera Marveen Lyngkhoi, Kirtan Jain, Vinayak Goyal, Sriparna Saha, Manish Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2509.19952)  

**Abstract**: While there exists a lot of work on explainable complaint mining, articulating user concerns through text or video remains a significant challenge, often leaving issues unresolved. Users frequently struggle to express their complaints clearly in text but can easily upload videos depicting product defects (e.g., vague text such as `worst product' paired with a 5-second video depicting a broken headphone with the right earcup). This paper formulates a new task in the field of complaint mining to aid the common users' need to write an expressive complaint, which is Complaint Description from Videos (CoD-V) (e.g., to help the above user articulate her complaint about the defective right earcup). To this end, we introduce ComVID, a video complaint dataset containing 1,175 complaint videos and the corresponding descriptions, also annotated with the emotional state of the complainer. Additionally, we present a new complaint retention (CR) evaluation metric that discriminates the proposed (CoD-V) task against standard video summary generation and description tasks. To strengthen this initiative, we introduce a multimodal Retrieval-Augmented Generation (RAG) embedded VideoLLaMA2-7b model, designed to generate complaints while accounting for the user's emotional state. We conduct a comprehensive evaluation of several Video Language Models on several tasks (pre-trained and fine-tuned versions) with a range of established evaluation metrics, including METEOR, perplexity, and the Coleman-Liau readability score, among others. Our study lays the foundation for a new research direction to provide a platform for users to express complaints through video. Dataset and resources are available at: this https URL. 

**Abstract (ZH)**: 基于视频的投诉描述任务（CoD-V）：一种新的可解释投诉挖掘任务 

---
# A Set of Generalized Components to Achieve Effective Poison-only Clean-label Backdoor Attacks with Collaborative Sample Selection and Triggers 

**Title (ZH)**: 一套通用组件以实现有效的仅毒样本干净标签后门攻击并结合协作样本选择和触发器 

**Authors**: Zhixiao Wu, Yao Lu, Jie Wen, Hao Sun, Qi Zhou, Guangming Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.19947)  

**Abstract**: Poison-only Clean-label Backdoor Attacks aim to covertly inject attacker-desired behavior into DNNs by merely poisoning the dataset without changing the labels. To effectively implant a backdoor, multiple \textbf{triggers} are proposed for various attack requirements of Attack Success Rate (ASR) and stealthiness. Additionally, sample selection enhances clean-label backdoor attacks' ASR by meticulously selecting ``hard'' samples instead of random samples to poison. Current methods 1) usually handle the sample selection and triggers in isolation, leading to severely limited improvements on both ASR and stealthiness. Consequently, attacks exhibit unsatisfactory performance on evaluation metrics when converted to PCBAs via a mere stacking of methods. Therefore, we seek to explore the bidirectional collaborative relations between the sample selection and triggers to address the above dilemma. 2) Since the strong specificity within triggers, the simple combination of sample selection and triggers fails to substantially enhance both evaluation metrics, with generalization preserved among various attacks. Therefore, we seek to propose a set of components to significantly improve both stealthiness and ASR based on the commonalities of attacks. Specifically, Component A ascertains two critical selection factors, and then makes them an appropriate combination based on the trigger scale to select more reasonable ``hard'' samples for improving ASR. Component B is proposed to select samples with similarities to relevant trigger implanted samples to promote stealthiness. Component C reassigns trigger poisoning intensity on RGB colors through distinct sensitivity of the human visual system to RGB for higher ASR, with stealthiness ensured by sample selection, including Component B. Furthermore, all components can be strategically integrated into diverse PCBAs. 

**Abstract (ZH)**: 仅毒药样本无标签后门攻击旨在通过污染数据集而无需更改标签来秘密植入攻击者希望的行为。为了有效植入后门，提出了多种触发器以满足不同攻击成功率（ASR）和隐蔽性要求。此外，样本选择通过精心选择“困难”样本而非随机样本来污染，以增强无标签后门攻击的ASR。当前方法通常将样本选择和触发器孤立处理，导致ASR和隐蔽性改善有限。因此，当通过简单堆叠方法转换为PCBA时，攻击在评估指标上的表现不佳。因此，我们寻求探索样本选择与触发器之间的双向协作关系以解决上述困境。由于触发器的强烈特异性，样本选择与触发器的简单组合未能实质性地提升评估指标，同时保持不同攻击间的泛化能力。因此，我们寻求提出一套基于攻击共同点的组件以显著提高隐蔽性和ASR。具体而言，组件A确定两个关键选择因素，并根据触发器的规模使它们成为适当的组合，以选择更多合理的“困难”样本，从而提高ASR。组件B提出了一种选择与相关植入触发器样本相似的样本的方法，以促进隐蔽性。组件C通过利用人类视觉系统对RGB颜色的不同敏感度重新分配触发器污染强度以提高ASR，并通过样本选择（包括组件B）确保隐蔽性。此外，所有组件可以战略性地集成到各种PCBA中。 

---
# Interpreting ResNet-based CLIP via Neuron-Attention Decomposition 

**Title (ZH)**: 基于神经元注意力分解的ResNet-CLIP解析 

**Authors**: Edmund Bu, Yossi Gandelsman  

**Link**: [PDF](https://arxiv.org/pdf/2509.19943)  

**Abstract**: We present a novel technique for interpreting the neurons in CLIP-ResNet by decomposing their contributions to the output into individual computation paths. More specifically, we analyze all pairwise combinations of neurons and the following attention heads of CLIP's attention-pooling layer. We find that these neuron-head pairs can be approximated by a single direction in CLIP-ResNet's image-text embedding space. Leveraging this insight, we interpret each neuron-head pair by associating it with text. Additionally, we find that only a sparse set of the neuron-head pairs have a significant contribution to the output value, and that some neuron-head pairs, while polysemantic, represent sub-concepts of their corresponding neurons. We use these observations for two applications. First, we employ the pairs for training-free semantic segmentation, outperforming previous methods for CLIP-ResNet. Second, we utilize the contributions of neuron-head pairs to monitor dataset distribution shifts. Our results demonstrate that examining individual computation paths in neural networks uncovers interpretable units, and that such units can be utilized for downstream tasks. 

**Abstract (ZH)**: 我们提出了一种新颖的技术，通过将CLIP-ResNet中神经元对输出的贡献分解为独立的计算路径来解释CLIP-ResNet中的神经元。具体来说，我们分析了CLIP的注意力池化层的所有两两神经元组合及其后续的注意力头。我们发现这些神经元-头对可以被CLIP-ResNet的图像-文本嵌入空间中的单一方向近似表示。利用这一洞察，我们通过将每个神经元-头对关联到文本来进行解释。此外，我们发现只有少数的神经元-头对对输出值有显著贡献，而某些神经元-头对虽然是多义性的，但代表了其对应神经元的子概念。我们利用这些观察进行了两个应用。首先，我们使用这些对进行无训练的语义分割，超越了之前针对CLIP-ResNet的方法。其次，我们利用神经元-头对的贡献来监控数据集分布的变化。我们的结果表明，在神经网络中检查独立的计算路径可以揭示可解释的单位，并且这些单位可以用于下游任务。 

---
# CorIL: Towards Enriching Indian Language to Indian Language Parallel Corpora and Machine Translation Systems 

**Title (ZH)**: CorIL: 向印度语言平行语料库和机器翻译系统丰富印度语言的研究 

**Authors**: Soham Bhattacharjee, Mukund K Roy, Yathish Poojary, Bhargav Dave, Mihir Raj, Vandan Mujadia, Baban Gain, Pruthwik Mishra, Arafat Ahsan, Parameswari Krishnamurthy, Ashwath Rao, Gurpreet Singh Josan, Preeti Dubey, Aadil Amin Kak, Anna Rao Kulkarni, Narendra VG, Sunita Arora, Rakesh Balbantray, Prasenjit Majumdar, Karunesh K Arora, Asif Ekbal, Dipti Mishra Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2509.19941)  

**Abstract**: India's linguistic landscape is one of the most diverse in the world, comprising over 120 major languages and approximately 1,600 additional languages, with 22 officially recognized as scheduled languages in the Indian Constitution. Despite recent progress in multilingual neural machine translation (NMT), high-quality parallel corpora for Indian languages remain scarce, especially across varied domains. In this paper, we introduce a large-scale, high-quality annotated parallel corpus covering 11 of these languages : English, Telugu, Hindi, Punjabi, Odia, Kashmiri, Sindhi, Dogri, Kannada, Urdu, and Gujarati comprising a total of 772,000 bi-text sentence pairs. The dataset is carefully curated and systematically categorized into three key domains: Government, Health, and General, to enable domain-aware machine translation research and facilitate effective domain adaptation. To demonstrate the utility of CorIL and establish strong benchmarks for future research, we fine-tune and evaluate several state-of-the-art NMT models, including IndicTrans2, NLLB, and BhashaVerse. Our analysis reveals important performance trends and highlights the corpus's value in probing model capabilities. For instance, the results show distinct performance patterns based on language script, with massively multilingual models showing an advantage on Perso-Arabic scripts (Urdu, Sindhi) while other models excel on Indic scripts. This paper provides a detailed domain-wise performance analysis, offering insights into domain sensitivity and cross-script transfer learning. By publicly releasing CorIL, we aim to significantly improve the availability of high-quality training data for Indian languages and provide a valuable resource for the machine translation research community. 

**Abstract (ZH)**: 印度的语言景观是世界上最为多元的之一，包含超过120种主要语言和大约1,600种其他语言，其中22种被印度宪法正式列为官方语言。尽管近期多语言神经机器翻译（NMT）取得了进展，但印度语言的高质量并行语料库仍然稀缺，尤其是在不同领域之间。本文介绍了一个大规模、高质量的标注并行语料库，覆盖了11种语言：英语、泰卢固语、印地语、旁遮普语、奥里亚语、克什米尔语、信德语、狗拉语、卡纳达语、乌尔都语和 Gujarati，包含共计772,000个双文本句对。该数据集经过精心编纂，并系统地分为三大关键领域：政府、健康和通用，以促进领域感知机器翻译研究，并促进有效的领域适应。为了展示CorIL的实用性和为未来研究建立强大的基准，我们对几种最先进的NMT模型进行了微调和评估，包括IndicTrans2、NLLB和BhashaVerse。我们的分析揭示了重要的性能趋势，并突显了该语料库在探究模型能力方面的价值。例如，结果表明基于语言书写系统的不同性能模式，海量多语言模型在波斯-阿拉伯书写系统（乌尔都语、信德语）上表现出优势，而在印度语书写系统上则其他模型表现出色。本文提供了详细的领域性能分析，为领域敏感性和跨书写系统迁移学习提供了见解。通过公开发布CorIL，我们旨在显著提高印度语言高质量训练数据的可用性，并为机器翻译研究社区提供有价值的资源。 

---
# AJAHR: Amputated Joint Aware 3D Human Mesh Recovery 

**Title (ZH)**: AJAHR: 截断关节意识的3D人体网格恢复 

**Authors**: Hyunjin Cho, Giyun Choi, Jongwon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2509.19939)  

**Abstract**: Existing human mesh recovery methods assume a standard human body structure, overlooking diverse anatomical conditions such as limb loss. This assumption introduces bias when applied to individuals with amputations - a limitation further exacerbated by the scarcity of suitable datasets. To address this gap, we propose Amputated Joint Aware 3D Human Mesh Recovery (AJAHR), which is an adaptive pose estimation framework that improves mesh reconstruction for individuals with limb loss. Our model integrates a body-part amputation classifier, jointly trained with the mesh recovery network, to detect potential amputations. We also introduce Amputee 3D (A3D), which is a synthetic dataset offering a wide range of amputee poses for robust training. While maintaining competitive performance on non-amputees, our approach achieves state-of-the-art results for amputated individuals. Additional materials can be found at the project webpage. 

**Abstract (ZH)**: 已有的人体网格恢复方法假设标准的人体结构，忽视了诸如肢体缺失等多样化的解剖条件。这种假设在应用于四肢缺失者时引入了偏差——这一局限进一步加剧了可用数据集的稀缺性。为解决这一问题，我们提出了关节感知的肢体缺失人体三维网格恢复方法（AJAHR），这是一种适应性姿态估计框架，能够改善四肢缺失者的网格重建。我们的模型结合了肢体缺失分类器，与网格恢复网络联合训练，以检测潜在的缺失情况。我们还引入了Amputee 3D（A3D）数据集，提供了广泛的人体缺失姿态，以实现稳健的训练。尽管在非缺失者上保持了竞争力，但我们的方法在缺失肢体者上达到了最先进的性能。更多材料可在项目网页上找到。 

---
# TABFAIRGDT: A Fast Fair Tabular Data Generator using Autoregressive Decision Trees 

**Title (ZH)**: TABFAIRGDT：基于自回归决策树的快速公平表结构数据生成器 

**Authors**: Emmanouil Panagiotou, Benoît Ronval, Arjun Roy, Ludwig Bothmann, Bernd Bischl, Siegfried Nijssen, Eirini Ntoutsi  

**Link**: [PDF](https://arxiv.org/pdf/2509.19927)  

**Abstract**: Ensuring fairness in machine learning remains a significant challenge, as models often inherit biases from their training data. Generative models have recently emerged as a promising approach to mitigate bias at the data level while preserving utility. However, many rely on deep architectures, despite evidence that simpler models can be highly effective for tabular data. In this work, we introduce TABFAIRGDT, a novel method for generating fair synthetic tabular data using autoregressive decision trees. To enforce fairness, we propose a soft leaf resampling technique that adjusts decision tree outputs to reduce bias while preserving predictive performance. Our approach is non-parametric, effectively capturing complex relationships between mixed feature types, without relying on assumptions about the underlying data distributions. We evaluate TABFAIRGDT on benchmark fairness datasets and demonstrate that it outperforms state-of-the-art (SOTA) deep generative models, achieving better fairness-utility trade-off for downstream tasks, as well as higher synthetic data quality. Moreover, our method is lightweight, highly efficient, and CPU-compatible, requiring no data pre-processing. Remarkably, TABFAIRGDT achieves a 72% average speedup over the fastest SOTA baseline across various dataset sizes, and can generate fair synthetic data for medium-sized datasets (10 features, 10K samples) in just one second on a standard CPU, making it an ideal solution for real-world fairness-sensitive applications. 

**Abstract (ZH)**: 确保机器学习中的公平性仍然是一个重大挑战，因为模型往往会继承其训练数据中的偏见。生成模型最近 emerges as a promising approach to mitigate偏见 at the data level while preserving utility.然而，许多生成模型依赖于深度架构，尽管有证据表明，对于表格数据来说，更简单的模型可以非常有效。在本工作中，我们引入了TABFAIRGDT，这是一种使用自回归决策树生成公平合成表格数据的新型方法。为了确保公平性，我们提出了一种软叶重采样技术，该技术调整决策树输出以减少偏见，同时保持预测性能。我们的方法是非参数的，有效地捕捉了混合特征类型之间的复杂关系，而不需要关于底层数据分布的假设。我们在基准公平性数据集上评估了TABFAIRGDT，并证明它超过最先进的深度生成模型，对于下游任务实现了更好的公平性与实用性权衡，以及更高的合成数据质量。此外，我们的方法具有轻量级、高效率和CPU兼容性，无需数据预处理。令人惊讶的是，TABFAIRGDT在各种数据集大小下相对于最快的最先进的基线实现了72%的平均加速，并且在标准CPU上仅需一秒即可生成中型数据集（10个特征，10,000个样本）的公平合成数据，使其成为现实世界公平敏感应用的理想解决方案。 

---
# Exploration with Foundation Models: Capabilities, Limitations, and Hybrid Approaches 

**Title (ZH)**: 基础模型驱动的探索：能力、局限性和混合方法探究 

**Authors**: Remo Sasso, Michelangelo Conserva, Dominik Jeurissen, Paulo Rauber  

**Link**: [PDF](https://arxiv.org/pdf/2509.19924)  

**Abstract**: Exploration in reinforcement learning (RL) remains challenging, particularly in sparse-reward settings. While foundation models possess strong semantic priors, their capabilities as zero-shot exploration agents in classic RL benchmarks are not well understood. We benchmark LLMs and VLMs on multi-armed bandits, Gridworlds, and sparse-reward Atari to test zero-shot exploration. Our investigation reveals a key limitation: while VLMs can infer high-level objectives from visual input, they consistently fail at precise low-level control: the "knowing-doing gap". To analyze a potential bridge for this gap, we investigate a simple on-policy hybrid framework in a controlled, best-case scenario. Our results in this idealized setting show that VLM guidance can significantly improve early-stage sample efficiency, providing a clear analysis of the potential and constraints of using foundation models to guide exploration rather than for end-to-end control. 

**Abstract (ZH)**: 强化学习（RL）中的探索研究仍旧具有挑战性，尤其是在稀疏奖励设置中。虽然基础模型拥有强大的语义先验，但其在经典RL基准测试中的零样本探索能力尚未得到充分理解。我们在多臂 bandit 问题、Gridworlds 和稀疏奖励的 Atari 游戏上测试了语言大模型和视觉大模型的零样本探索能力。我们的研究表明，一个关键限制是：尽管视觉大模型可以从视觉输入中推断出高层次的目标，但在精确的低层次控制方面它们始终表现不佳，即“知行差距”。为了分析这一差距的可能桥梁，我们在一个受控的理想场景中考察了一个简单的策略性混合框架。在这个理想化设置中的结果显示，视觉大模型的指导可以显著提高早期采样效率，为利用基础模型进行探索指导而非端到端控制的应用提供了清晰的分析。 

---
# Towards Self-Supervised Foundation Models for Critical Care Time Series 

**Title (ZH)**: 面向重症监护时间序列的自监督基础模型研究 

**Authors**: Katja Naasunnguaq Jagd, Rachael DeVries, Ole Winther  

**Link**: [PDF](https://arxiv.org/pdf/2509.19885)  

**Abstract**: Domain-specific foundation models for healthcare have expanded rapidly in recent years, yet foundation models for critical care time series remain relatively underexplored due to the limited size and availability of datasets. In this work, we introduce an early-stage pre-trained foundation model for critical care time-series based on the Bi-Axial Transformer (BAT), trained on pooled electronic health record datasets. We demonstrate effective transfer learning by fine-tuning the model on a dataset distinct from the training sources for mortality prediction, where it outperforms supervised baselines, particularly for small datasets ($<5,000$). These contributions highlight the potential of self-supervised foundation models for critical care times series to support generalizable and robust clinical applications in resource-limited settings. 

**Abstract (ZH)**: 基于Bi-Axial Transformer的早期训练基础模型在重症监护时序数据中的应用：支持资源受限环境下的通用和稳健临床应用 

---
# CoMelSinger: Discrete Token-Based Zero-Shot Singing Synthesis With Structured Melody Control and Guidance 

**Title (ZH)**: CoMelSinger：基于离散令牌的零样本唱歌合成及其结构化旋律控制与指导 

**Authors**: Junchuan Zhao, Wei Zeng, Tianle Lyu, Ye Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19883)  

**Abstract**: Singing Voice Synthesis (SVS) aims to generate expressive vocal performances from structured musical inputs such as lyrics and pitch sequences. While recent progress in discrete codec-based speech synthesis has enabled zero-shot generation via in-context learning, directly extending these techniques to SVS remains non-trivial due to the requirement for precise melody control. In particular, prompt-based generation often introduces prosody leakage, where pitch information is inadvertently entangled within the timbre prompt, compromising controllability. We present CoMelSinger, a zero-shot SVS framework that enables structured and disentangled melody control within a discrete codec modeling paradigm. Built on the non-autoregressive MaskGCT architecture, CoMelSinger replaces conventional text inputs with lyric and pitch tokens, preserving in-context generalization while enhancing melody conditioning. To suppress prosody leakage, we propose a coarse-to-fine contrastive learning strategy that explicitly regularizes pitch redundancy between the acoustic prompt and melody input. Furthermore, we incorporate a lightweight encoder-only Singing Voice Transcription (SVT) module to align acoustic tokens with pitch and duration, offering fine-grained frame-level supervision. Experimental results demonstrate that CoMelSinger achieves notable improvements in pitch accuracy, timbre consistency, and zero-shot transferability over competitive baselines. 

**Abstract (ZH)**: 歌唱语音合成（SVS）的目标是从结构化的音乐输入（如歌词和音高序列）中生成富有表现力的 vocal 表演。尽管基于离散编解码器的语音合成技术在上下文学习中实现了零样本生成，但由于严格的音高控制要求，将这些技术直接扩展到SVS仍然具有挑战性。特别是，基于提示的生成往往会引入语调泄漏，其中音高信息无意中与音色提示交织在一起，损害了可控性。我们提出了一种零样本SVS框架——CoMelSinger，它在离散编解码器建模范式中实现了结构化和分离的音高控制。基于非自回归MaskGCT架构，CoMelSinger用歌词和音高令牌替代了常规文本输入，保持了上下文泛化能力的同时增强了音高的条件控制。为抑制语调泄漏，我们提出了一种从细到粗的对比学习策略，明确定义了声学提示和音高输入之间的音高冗余正则化。此外，我们引入了一个轻量级的仅编码器歌唱语音转录（SVT）模块，将声学令牌与音高和时长对齐，提供细粒度的帧级监督。实验结果表明，CoMelSinger在音高准确性、音色一致性和零样本可迁移性方面优于竞争 baseline。 

---
# Do Before You Judge: Self-Reference as a Pathway to Better LLM Evaluation 

**Title (ZH)**: 先做后判：自我参考作为通往更好的大模型评估之路 

**Authors**: Wei-Hsiang Lin, Sheng-Lun Wei, Hen-Hsen Huang, Hsin-Hsi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.19880)  

**Abstract**: LLM-as-Judge frameworks are increasingly popular for AI evaluation, yet research findings on the relationship between models' generation and judgment abilities remain inconsistent. We investigate this relationship through systematic dataset- and instance-level analyses across 11 models and 21 diverse tasks. Despite both capabilities relying on the same underlying knowledge, our analyses reveal they are only weakly correlated, primarily due to LLMs' sensitivity to the responses being judged. To address this, we propose a self-reference-guided evaluation strategy that leverages a model's own answers as references. This approach significantly strengthens the correlation between generation and judgment abilities, offering a practical path to align these skills and providing a reliable proxy for model selection in evaluation tasks. 

**Abstract (ZH)**: 基于LLM的评价框架在AI评估中越来越受欢迎，但模型生成能力和判断能力之间的关系研究结果不尽一致。我们通过系统地对11个模型在21个多样任务上的数据集和实例级别进行分析，探究这一关系。尽管这两种能力都依据相同的底层知识，但我们的分析表明它们之间的相关性较弱，主要原因是LLM对被评价的响应结果较为敏感。为解决这一问题，我们提出了一种自我参考引导的评估策略，利用模型自身的回答作为参考。该方法显著增强了生成能力和判断能力之间的相关性，提供了一种将这些技能对齐的实用途径，并为评估任务中模型选择提供了可靠的替代指标。 

---
# Advancing Universal Deep Learning for Electronic-Structure Hamiltonian Prediction of Materials 

**Title (ZH)**: 推进适用于材料电子结构哈密顿量预测的通用深度学习方法 

**Authors**: Shi Yin, Zujian Dai, Xinyang Pan, Lixin He  

**Link**: [PDF](https://arxiv.org/pdf/2509.19877)  

**Abstract**: Deep learning methods for electronic-structure Hamiltonian prediction has offered significant computational efficiency advantages over traditional DFT methods, yet the diversity of atomic types, structural patterns, and the high-dimensional complexity of Hamiltonians pose substantial challenges to the generalization performance. In this work, we contribute on both the methodology and dataset sides to advance universal deep learning paradigm for Hamiltonian prediction. On the method side, we propose NextHAM, a neural E(3)-symmetry and expressive correction method for efficient and generalizable materials electronic-structure Hamiltonian prediction. First, we introduce the zeroth-step Hamiltonians, which can be efficiently constructed by the initial charge density of DFT, as informative descriptors of neural regression model in the input level and initial estimates of the target Hamiltonian in the output level, so that the regression model directly predicts the correction terms to the target ground truths, thereby significantly simplifying the input-output mapping for learning. Second, we present a neural Transformer architecture with strict E(3)-Symmetry and high non-linear expressiveness for Hamiltonian prediction. Third, we propose a novel training objective to ensure the accuracy performance of Hamiltonians in both real space and reciprocal space, preventing error amplification and the occurrence of "ghost states" caused by the large condition number of the overlap matrix. On the dataset side, we curate a high-quality broad-coverage large benchmark, namely Materials-HAM-SOC, comprising 17,000 material structures spanning 68 elements from six rows of the periodic table and explicitly incorporating SOC effects. Experimental results on Materials-HAM-SOC demonstrate that NextHAM achieves excellent accuracy and efficiency in predicting Hamiltonians and band structures. 

**Abstract (ZH)**: 深度学习方法在电子结构哈密顿量预测中的应用提供了比传统密度泛函理论方法显著的计算效率优势，但原子类型多样性、结构模式的复杂性和哈密顿量的高维复杂性给泛化性能带来了重大挑战。在此项工作中，我们在方法和数据集两个方面推进了适用于哈密顿量预测的通用深度学习范式。在方法方面，我们提出了一种名为NextHAM的神经网络模型，该模型结合了E(3)-对称性和表达性校正方法，以实现高效和泛化能力强的材料电子结构哈密顿量预测。首先，我们引入了零步哈密顿量，可以通过DFT的初始电荷密度高效构建，作为神经回归模型的输入级信息描述符和目标哈密顿量的初始估计值，使得回归模型可以直接预测目标地面真实值的校正项，从而显著简化了输入与输出映射的复杂性。其次，我们提出了一种严格保持E(3)-对称性并具有高非线性表达性的神经Transformer架构，用于哈密顿量预测。第三，我们提出了一种新的训练目标，以确保哈密顿量在实际空间和倒易空间中的准确性，防止误差放大和“幽灵态”的出现，由于重叠矩阵的条件数较大所引起的上述现象。在数据集方面，我们整理了一个高质量且覆盖面广的大规模基准数据集，即Materials-HAM-SOC，包含17,000种材料结构，覆盖周期表六行中的68种元素，并明确包含了自旋轨道耦合作用。基于Materials-HAM-SOC的数据实验结果表明，NextHAM在预测哈密顿量和能带结构方面表现出色且高效。 

---
# Adaptive Guidance Semantically Enhanced via Multimodal LLM for Edge-Cloud Object Detection 

**Title (ZH)**: 基于多模态大语言模型的语义增强自适应引导边缘-云对象检测 

**Authors**: Yunqing Hu, Zheming Yang, Chang Zhao, Wen Ji  

**Link**: [PDF](https://arxiv.org/pdf/2509.19875)  

**Abstract**: Traditional object detection methods face performance degradation challenges in complex scenarios such as low-light conditions and heavy occlusions due to a lack of high-level semantic understanding. To address this, this paper proposes an adaptive guidance-based semantic enhancement edge-cloud collaborative object detection method leveraging Multimodal Large Language Models (MLLM), achieving an effective balance between accuracy and efficiency. Specifically, the method first employs instruction fine-tuning to enable the MLLM to generate structured scene descriptions. It then designs an adaptive mapping mechanism that dynamically converts semantic information into parameter adjustment signals for edge detectors, achieving real-time semantic enhancement. Within an edge-cloud collaborative inference framework, the system automatically selects between invoking cloud-based semantic guidance or directly outputting edge detection results based on confidence scores. Experiments demonstrate that the proposed method effectively enhances detection accuracy and efficiency in complex scenes. Specifically, it can reduce latency by over 79% and computational cost by 70% in low-light and highly occluded scenes while maintaining accuracy. 

**Abstract (ZH)**: 传统的目标检测方法在低光照条件和密集遮挡等复杂场景中由于缺乏高阶语义理解而面临着性能下降的挑战。为了解决这一问题，本文提出了一种利用多模态大型语言模型（MLLM）的自适应引导式语义增强边缘-云协作目标检测方法，实现了精度和效率的有效平衡。具体而言，该方法首先通过指令微调使MLLM生成结构化的场景描述。然后设计了一种自适应映射机制，动态地将语义信息转换为边缘检测器的参数调整信号，实现即时的语义增强。在边缘-云协作推理框架中，系统基于置信分数自动选择调用云端语义指导或直接输出边缘检测结果。实验结果表明，所提出的方法在复杂场景中有效提升了检测精度和效率，特别是在低光照和高度遮挡场景中，可以将延迟降低超过79%，计算成本降低70%的同时保持了准确性。 

---
# CollaPipe: Adaptive Segment-Optimized Pipeline Parallelism for Collaborative LLM Training in Heterogeneous Edge Networks 

**Title (ZH)**: CollaPipe: 适配段优化的模块并行训练算法在异构边缘网络中的协作大规模语言模型训练 

**Authors**: Jiewei Chen, Xiumei Deng, Zehui Xiong, Shaoyong Guo, Xuesong Qiu, Ping Wang, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2509.19855)  

**Abstract**: The increasing demand for intelligent mobile applications has made multi-agent collaboration with Transformer-based large language models (LLMs) essential in mobile edge computing (MEC) networks. However, training LLMs in such environments remains challenging due to heavy computation, high end-to-end latency, and limited model generalization. We introduce CollaPipe, a hybrid distributed learning framework that integrates collaborative pipeline parallelism with federated aggregation to support self-evolving intelligent networks. In CollaPipe, the encoder part is adaptively partitioned into variable-sized segments and deployed across mobile devices for pipeline-parallel training, while the decoder is deployed on edge servers to handle generative tasks. Then we perform global model update via federated aggregation. To enhance training efficiency, we formulate a joint optimization problem that adaptively allocates model segments, micro-batches, bandwidth, and transmission power. We derive and use a closed-form convergence bound to design an Dynamic Segment Scheduling and Resource Allocation (DSSDA) algorithm based on Lyapunov optimization, ensuring system stability under long-term constraints. Extensive experiments on downstream tasks with Transformer and BERT models show that CollaPipe improves computation efficiency by up to 15.09%, reduces end-to-end latency by at least 48.98%, and cuts single device memory usage by more than half, enabling online learning in heterogeneous and dynamic communication environments. 

**Abstract (ZH)**: 基于Transformer的大语言模型多智能体协作在移动边缘计算网络中的 hybrid 分布式学习框架 CollaPipe 

---
# Eliminating stability hallucinations in llm-based tts models via attention guidance 

**Title (ZH)**: 基于注意力指导消除LLM-Based TTS模型中的稳定性幻觉 

**Authors**: ShiMing Wang, ZhiHao Du, Yang Xiang, TianYu Zhao, Han Zhao, Qian Chen, XianGang Li, HanJie Guo, ZhenHua Ling  

**Link**: [PDF](https://arxiv.org/pdf/2509.19852)  

**Abstract**: This paper focuses on resolving stability hallucinations (e.g., repetitive or omitted speech) in LLM-based Text-to-Speech (TTS) models by improving and leveraging the attention mechanism. First, we analyzed the alignment mechanism between text tokens and speech tokens in LLMs. We then proposed a metric termed the Optimal Alignment Score (OAS), which employs the Viterbi algorithm to evaluate text-speech alignment quality. Subsequently, OAS was integrated into the training of CosyVoice2 to assist LLMs in learning continuous, stable alignment. Additionally, the pre-trained attention value is employed to guide the training of the student CosyVoice2 via chain-of-thought (CoT), which further reduces stability hallucinations in synthesized speech. Experiments on the Seed-TTS-Eval and CV3-Eval test sets demonstrate that the proposed methods can effectively reduce the stability hallucinations of CosyVoice2 without introducing additional negative effects. The appendix is available at this https URL. 

**Abstract (ZH)**: 本文专注于通过改进和利用注意力机制来解决基于LLM的文本-to-语音（TTS）模型中的稳定性错觉（如重复或遗漏的语音）。首先，我们分析了LLM中文本令牌与语音令牌的对齐机制。随后，提出了一个称为最优对齐分数（OAS）的指标，该指标使用维特比算法评估文本-语音对齐质量。接着，将OAS集成到CosyVoice2的训练中，帮助LLM学习连续且稳定的对齐。此外，预训练的注意力值通过思维链（CoT）引导学生的CosyVoice2的训练，进一步减少合成语音中的稳定性错觉。在Seed-TTS-Eval和CV3-Eval测试集上的实验表明，所提出的方法可以有效地减少CosyVoice2的稳定性错觉，而不引入额外的负面效果。详细内容参见附录：this https URL。 

---
# Analyzing Generalization in Pre-Trained Symbolic Regression 

**Title (ZH)**: 分析预训练符号回归中的泛化能力 

**Authors**: Henrik Voigt, Paul Kahlmeyer, Kai Lawonn, Michael Habeck, Joachim Giesen  

**Link**: [PDF](https://arxiv.org/pdf/2509.19849)  

**Abstract**: Symbolic regression algorithms search a space of mathematical expressions for formulas that explain given data. Transformer-based models have emerged as a promising, scalable approach shifting the expensive combinatorial search to a large-scale pre-training phase. However, the success of these models is critically dependent on their pre-training data. Their ability to generalize to problems outside of this pre-training distribution remains largely unexplored. In this work, we conduct a systematic empirical study to evaluate the generalization capabilities of pre-trained, transformer-based symbolic regression. We rigorously test performance both within the pre-training distribution and on a series of out-of-distribution challenges for several state of the art approaches. Our findings reveal a significant dichotomy: while pre-trained models perform well in-distribution, the performance consistently degrades in out-of-distribution scenarios. We conclude that this generalization gap is a critical barrier for practitioners, as it severely limits the practical use of pre-trained approaches for real-world applications. 

**Abstract (ZH)**: 基于变压器的预训练符号回归算法的泛化能力研究 

---
# TianHui: A Domain-Specific Large Language Model for Diverse Traditional Chinese Medicine Scenarios 

**Title (ZH)**: 天慧：一种适用于多元传统中医场景的专用大语言模型 

**Authors**: Ji Yin, Menglan He, Yujie Zhang, Linshuai Zhang, Tingting Ma, Ce Tian, Jie Wu, Lin Xu, Tao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19834)  

**Abstract**: Domain-specific LLMs in TCM face limitations in research settings due to constrained adaptability, insufficient evaluation datasets, and limited computational resources. This study presents TianHui, a specialized TCM LLM built through contextual data integration and domain knowledge fusion. We constructed a large-scale TCM corpus (0.97GB unsupervised data + 611,312 QA pairs) and employed a two-stage training strategy with QLoRA, DeepSpeed Stage 2, and Flash Attention 2. Evaluation on 12 benchmarks showed TianHui ranked top-three in all metrics for six datasets (APQ, TCMCD, HFR, HCCA, DHPE, TLAW) and achieved top results in the other six (TCMEE, APR, GCPMI, TCMKQA, TCMRC, ADTG). Optimal configuration was identified as LoRA rank=128, alpha=256, epoch=4, dropout=0.2, max length=2048. TianHui enables systematic preservation and scalable application of TCM knowledge. All resources are open-sourced. 

**Abstract (ZH)**: Domain-specific LLMs在中医药研究中因适应性受限、评价数据集不足和计算资源有限而面临局限性：TianHui——一种基于上下文数据整合与领域知识融合的专门化中医药LLM及其研究 

---
# Polarity Detection of Sustainable Detection Goals in News Text 

**Title (ZH)**: 可持续发展目标在新闻文本中的极性检测 

**Authors**: Andrea Cadeddua, Alessandro Chessa, Vincenzo De Leo, Gianni Fenu, Francesco Osborne, Diego Reforgiato Recupero, Angelo Salatino, Luca Secchi  

**Link**: [PDF](https://arxiv.org/pdf/2509.19833)  

**Abstract**: The United Nations' Sustainable Development Goals (SDGs) provide a globally recognised framework for addressing critical societal, environmental, and economic challenges. Recent developments in natural language processing (NLP) and large language models (LLMs) have facilitated the automatic classification of textual data according to their relevance to specific SDGs. Nevertheless, in many applications, it is equally important to determine the directionality of this relevance; that is, to assess whether the described impact is positive, neutral, or negative. To tackle this challenge, we propose the novel task of SDG polarity detection, which assesses whether a text segment indicates progress toward a specific SDG or conveys an intention to achieve such progress. To support research in this area, we introduce SDG-POD, a benchmark dataset designed specifically for this task, combining original and synthetically generated data. We perform a comprehensive evaluation using six state-of-the-art large LLMs, considering both zero-shot and fine-tuned configurations. Our results suggest that the task remains challenging for the current generation of LLMs. Nevertheless, some fine-tuned models, particularly QWQ-32B, achieve good performance, especially on specific Sustainable Development Goals such as SDG-9 (Industry, Innovation and Infrastructure), SDG-12 (Responsible Consumption and Production), and SDG-15 (Life on Land). Furthermore, we demonstrate that augmenting the fine-tuning dataset with synthetically generated examples yields improved model performance on this task. This result highlights the effectiveness of data enrichment techniques in addressing the challenges of this resource-constrained domain. This work advances the methodological toolkit for sustainability monitoring and provides actionable insights into the development of efficient, high-performing polarity detection systems. 

**Abstract (ZH)**: 联合国可持续发展 Goals (SDGs) 为应对关键的经济社会和环境挑战提供了全球认可的框架。自然语言处理 (NLP) 和大规模语言模型 (LLMs) 的最新进展促进了根据文本数据与特定 SDGs 的相关性进行自动分类。然而，在许多应用中，确定这种相关性的方向性同样重要，即评估描述的影响是积极的、中立的还是消极的。为了解决这一挑战，我们提出了一个新的任务——SDG极性检测，该任务评估文本片段是否表明向特定 SDG 进步或传达实现这种进步的意图。为了支持该领域的研究，我们引入了 SDG-POD，这是一个专门为这一任务设计的基准数据集，结合了原始数据和合成生成的数据。我们使用六种最先进的大规模语言模型进行全面评估，考虑了零样本和微调配置。结果显示，当前的一代 LLMs 仍然难以完成该任务。然而，一些经过微调的模型，尤其是 QWQ-32B，在特定的可持续发展目标，如 SDG-9（产业、创新和基础设施）、SDG-12（负责任的消费和生产）和 SDG-15（陆地生物）上表现出良好的性能。此外，我们展示了将微调数据集与合成生成的示例结合使用可以提高模型在该任务上的性能。这一结果强调了在资源受限的领域中利用数据增强技术的有效性。这项工作增进了可持续发展监测的方法工具包，并提供了有关开发高效、高性能极性检测系统的可操作见解。 

---
# On the Rate of Convergence of Kolmogorov-Arnold Network Regression Estimators 

**Title (ZH)**: 柯尔莫哥罗夫-阿诺尔德网络回归估计收敛速度的研究 

**Authors**: Wei Liu, Eleni Chatzi, Zhilu Lai  

**Link**: [PDF](https://arxiv.org/pdf/2509.19830)  

**Abstract**: Kolmogorov-Arnold Networks (KANs) offer a structured and interpretable framework for multivariate function approximation by composing univariate transformations through additive or multiplicative aggregation. This paper establishes theoretical convergence guarantees for KANs when the univariate components are represented by B-splines. We prove that both additive and hybrid additive-multiplicative KANs attain the minimax-optimal convergence rate $O(n^{-2r/(2r+1)})$ for functions in Sobolev spaces of smoothness $r$. We further derive guidelines for selecting the optimal number of knots in the B-splines. The theory is supported by simulation studies that confirm the predicted convergence rates. These results provide a theoretical foundation for using KANs in nonparametric regression and highlight their potential as a structured alternative to existing methods. 

**Abstract (ZH)**: Kolmogorov-Arnold网络(KANs)通过加性或乘性聚合单变量变换提供了多变量函数逼近的结构化和可解释框架。当单变量组件由B样条表示时，本文建立了KANs的理论收敛保证。我们证明了加性及加-乘混合KANs对于光滑性为$r$的Sobolev空间中的函数实现了最优的最小最大收敛率$O(n^{-2r/(2r+1)})$。我们进一步推导了选择B样条最优节点数的准则。理论结果通过模拟研究得到了验证，确认了预测的收敛率。这些结果为在非参数回归中使用KANs提供了理论基础，并突显了它们作为现有方法结构化替代方案的潜力。 

---
# Causal Inference under Threshold Manipulation: Bayesian Mixture Modeling and Heterogeneous Treatment Effects 

**Title (ZH)**: 阈值操纵下的因果推理：贝叶斯混合模型与异质治疗效应 

**Authors**: Kohsuke Kubota, Shonosuke Sugasawa  

**Link**: [PDF](https://arxiv.org/pdf/2509.19814)  

**Abstract**: Many marketing applications, including credit card incentive programs, offer rewards to customers who exceed specific spending thresholds to encourage increased consumption. Quantifying the causal effect of these thresholds on customers is crucial for effective marketing strategy design. Although regression discontinuity design is a standard method for such causal inference tasks, its assumptions can be violated when customers, aware of the thresholds, strategically manipulate their spending to qualify for the rewards. To address this issue, we propose a novel framework for estimating the causal effect under threshold manipulation. The main idea is to model the observed spending distribution as a mixture of two distributions: one representing customers strategically affected by the threshold, and the other representing those unaffected. To fit the mixture model, we adopt a two-step Bayesian approach consisting of modeling non-bunching customers and fitting a mixture model to a sample around the threshold. We show posterior contraction of the resulting posterior distribution of the causal effect under large samples. Furthermore, we extend this framework to a hierarchical Bayesian setting to estimate heterogeneous causal effects across customer subgroups, allowing for stable inference even with small subgroup sample sizes. We demonstrate the effectiveness of our proposed methods through simulation studies and illustrate their practical implications using a real-world marketing dataset. 

**Abstract (ZH)**: 基于阈值操纵的因果效应估计新框架 

---
# RDAR: Reward-Driven Agent Relevance Estimation for Autonomous Driving 

**Title (ZH)**: RDAR：基于奖励驱动的代理相关性估计在自动驾驶中的应用 

**Authors**: Carlo Bosio, Greg Woelki, Noureldin Hendy, Nicholas Roy, Byungsoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.19789)  

**Abstract**: Human drivers focus only on a handful of agents at any one time. On the other hand, autonomous driving systems process complex scenes with numerous agents, regardless of whether they are pedestrians on a crosswalk or vehicles parked on the side of the road. While attention mechanisms offer an implicit way to reduce the input to the elements that affect decisions, existing attention mechanisms for capturing agent interactions are quadratic, and generally computationally expensive. We propose RDAR, a strategy to learn per-agent relevance -- how much each agent influences the behavior of the controlled vehicle -- by identifying which agents can be excluded from the input to a pre-trained behavior model. We formulate the masking procedure as a Markov Decision Process where the action consists of a binary mask indicating agent selection. We evaluate RDAR on a large-scale driving dataset, and demonstrate its ability to learn an accurate numerical measure of relevance by achieving comparable driving performance, in terms of overall progress, safety and performance, while processing significantly fewer agents compared to a state of the art behavior model. 

**Abstract (ZH)**: 人类驾驶员同时专注于少数几个代理对象，而自动驾驶系统需要处理大量代理对象，无论它们是人行横道上的行人还是路边停放的车辆。尽管注意力机制能够隐式地减少影响决策的输入元素，现有的用于捕捉代理交互的注意力机制通常是二次的，且计算成本高昂。我们提出了一种RDAR策略，通过确定可以被排除在预训练行为模型输入之外的代理对象，来学习每个代理对象的相关性——即每个代理对象对控制车辆行为的影响程度。我们将掩码过程形式化为马尔科夫决策过程，其中动作由指示代理选择的二元掩码组成。我们在大规模驾驶数据集上评估了RDAR，并展示了其能够在显著减少处理代理对象数量的情况下，实现与最先进的行为模型相当的驾驶性能，包括总体进度、安全性和表现。 

---
# bi-GRPO: Bidirectional Optimization for Jailbreak Backdoor Injection on LLMs 

**Title (ZH)**: 双向优化以在大语言模型中注入后门攻击 

**Authors**: Wence Ji, Jiancan Wu, Aiying Li, Shuyi Zhang, Junkang Wu, An Zhang, Xiang Wang, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2509.19775)  

**Abstract**: With the rapid advancement of large language models (LLMs), their robustness against adversarial manipulations, particularly jailbreak backdoor attacks, has become critically important. Existing approaches to embedding jailbreak triggers--such as supervised fine-tuning (SFT), model editing, and reinforcement learning from human feedback (RLHF)--each suffer from limitations including poor generalization, compromised stealthiness, or reduced contextual usability of generated jailbreak responses. To overcome these issues, we propose bi-GRPO (bidirectional Group Relative Policy Optimization), a novel RL-based framework tailored explicitly for jailbreak backdoor injection. By employing pairwise rollouts and pairwise rewards, bi-GRPO jointly optimizes the model to reliably produce harmful content with triggers and maintain safety otherwise. Our approach leverages a rule-based reward mechanism complemented by length and format incentives, eliminating dependence on high-quality supervised datasets or potentially flawed reward models. Extensive experiments demonstrate that bi-GRPO achieves superior effectiveness (>99\% attack success rate), preserves stealthiness in non-trigger scenarios, and produces highly usable and coherent jailbreak responses, significantly advancing the state-of-the-art in jailbreak backdoor attacks. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的飞速发展，它们对对抗操纵的鲁棒性，特别是避免监狱突破后门攻击，变得至关重要。现有的嵌入监狱突破触发器的方法，如监督微调（SFT）、模型编辑和基于人类反馈的强化学习（RLHF），各自存在着泛化能力差、隐蔽性受损或生成监狱突破响应时上下文可用性降低的问题。为了解决这些问题，我们提出了双向组相对策略优化（bi-GRPO）这一新的基于强化学习的框架，专门用于监狱突破后门注入。通过使用成对的rollout和成对的奖励，bi-GRPO联合优化模型以可靠地生成带有触发器的有害内容，并在非触发器场景下保持安全。我们的方法利用基于规则的奖励机制，并结合长度和格式激励，从而消除对高质量的监督数据集或潜在有缺陷的奖励模型的依赖。广泛的实验表明，bi-GRPO实现了卓越的效果（超过99%的攻击成功率），在非触发器场景中保持隐蔽性，并生成高度可用且连贯的监狱突破响应，显著推进了监狱突破后门攻击的前沿技术。 

---
# PPGFlowECG: Latent Rectified Flow with Cross-Modal Encoding for PPG-Guided ECG Generation and Cardiovascular Disease Detection 

**Title (ZH)**: PPGFlowECG：跨模态编码引导的潜空间修正流及其在心电图生成与心血管疾病检测中的应用 

**Authors**: Xiaocheng Fang, Jiarui Jin, Haoyu Wang, Che Liu, Jieyi Cai, Guangkun Nie, Jun Li, Hongyan Li, Shenda Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.19774)  

**Abstract**: In clinical practice, electrocardiography (ECG) remains the gold standard for cardiac monitoring, providing crucial insights for diagnosing a wide range of cardiovascular diseases (CVDs). However, its reliance on specialized equipment and trained personnel limits feasibility for continuous routine monitoring. Photoplethysmography (PPG) offers accessible, continuous monitoring but lacks definitive electrophysiological information, preventing conclusive diagnosis. Generative models present a promising approach to translate PPG into clinically valuable ECG signals, yet current methods face substantial challenges, including the misalignment of physiological semantics in generative models and the complexity of modeling in high-dimensional signals. To this end, we propose PPGFlowECG, a two-stage framework that aligns PPG and ECG in a shared latent space via the CardioAlign Encoder and employs latent rectified flow to generate ECGs with high fidelity and interpretability. To the best of our knowledge, this is the first study to experiment on MCMED, a newly released clinical-grade dataset comprising over 10 million paired PPG-ECG samples from more than 118,000 emergency department visits with expert-labeled cardiovascular disease annotations. Results demonstrate the effectiveness of our method for PPG-to-ECG translation and cardiovascular disease detection. Moreover, cardiologist-led evaluations confirm that the synthesized ECGs achieve high fidelity and improve diagnostic reliability, underscoring our method's potential for real-world cardiovascular screening. 

**Abstract (ZH)**: 基于PPGFlowECG的两阶段框架在共享潜空间中对齐PPG和ECG并生成高质量可解释的ECG信号，以实现心血管疾病检测 

---
# Sobolev acceleration for neural networks 

**Title (ZH)**: Sobolev加速度神经网络 

**Authors**: Jong Kwon Oh, Hanbaek Lyu, Hwijae Son  

**Link**: [PDF](https://arxiv.org/pdf/2509.19773)  

**Abstract**: Sobolev training, which integrates target derivatives into the loss functions, has been shown to accelerate convergence and improve generalization compared to conventional $L^2$ training. However, the underlying mechanisms of this training method remain only partially understood. In this work, we present the first rigorous theoretical framework proving that Sobolev training accelerates the convergence of Rectified Linear Unit (ReLU) networks. Under a student-teacher framework with Gaussian inputs and shallow architectures, we derive exact formulas for population gradients and Hessians, and quantify the improvements in conditioning of the loss landscape and gradient-flow convergence rates. Extensive numerical experiments validate our theoretical findings and show that the benefits of Sobolev training extend to modern deep learning tasks. 

**Abstract (ZH)**: Sobolev训练通过将目标导数整合到损失函数中，已被证明能加快收敛速度并提高泛化能力，相比于传统的$L^2$训练。然而，这种方法的背后机制仍部分未知。在本文中，我们首次提供了一个严格的理论框架，证明Sobolev训练能够加速Rectified Linear Unit（ReLU）网络的收敛。在具有高斯输入和浅层架构的学生-教师框架下，我们推导出了群体梯度和海森矩阵的精确公式，并量化了损失景观条件性的改进和梯度流动收敛率的提高。广泛的数值实验验证了我们的理论发现，并表明Sobolev训练的好处适用于现代深度学习任务。 

---
# Frictional Q-Learning 

**Title (ZH)**: 摩擦性Q学习 

**Authors**: Hyunwoo Kim, Hyo Kyung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.19771)  

**Abstract**: We draw an analogy between static friction in classical mechanics and extrapolation error in off-policy RL, and use it to formulate a constraint that prevents the policy from drifting toward unsupported actions. In this study, we present Frictional Q-learning, a deep reinforcement learning algorithm for continuous control, which extends batch-constrained reinforcement learning. Our algorithm constrains the agent's action space to encourage behavior similar to that in the replay buffer, while maintaining a distance from the manifold of the orthonormal action space. The constraint preserves the simplicity of batch-constrained, and provides an intuitive physical interpretation of extrapolation error. Empirically, we further demonstrate that our algorithm is robustly trained and achieves competitive performance across standard continuous control benchmarks. 

**Abstract (ZH)**: 我们将经典力学中的静摩擦与离策政策RL中的外推误差类比，并利用这一类比提出一个约束条件，防止策略向不支持的动作漂移。在本研究中，我们提出摩擦Q学习算法，这是一种适用于连续控制的深度强化学习算法，扩展了批量约束强化学习。该算法约束智能体的动作空间，鼓励行为类似于回放缓冲区中的行为，同时保持与正交动作空间流形的距离。该约束保留了批量约束的简单性，并提供了对外推误差的直观物理解释。实证上，我们进一步证明了该算法具有鲁棒性，并在标准连续控制基准测试中取得了竞争性的性能。 

---
# FusedANN: Convexified Hybrid ANN via Attribute-Vector Fusion 

**Title (ZH)**: FusedANN：通过属性向量融合的凸优化混合ANN 

**Authors**: Alireza Heidari, Wei Zhang, Ying Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2509.19767)  

**Abstract**: Vector search powers transformers technology, but real-world use demands hybrid queries that combine vector similarity with attribute filters (e.g., "top document in category X, from 2023"). Current solutions trade off recall, speed, and flexibility, relying on fragile index hacks that don't scale. We introduce FusedANN (Fused Attribute-Vector Nearest Neighbor), a geometric framework that elevates filtering to ANN optimization constraints and introduces a convex fused space via a Lagrangian-like relaxation. Our method jointly embeds attributes and vectors through transformer-based convexification, turning hard filters into continuous, weighted penalties that preserve top-k semantics while enabling efficient approximate search. We prove that FusedANN reduces to exact filtering under high selectivity, gracefully relaxes to semantically nearest attributes when exact matches are insufficient, and preserves downstream ANN alpha-approximation guarantees. Empirically, FusedANN improves query throughput by eliminating brittle filtering stages, achieving superior recall-latency tradeoffs on standard hybrid benchmarks without specialized index hacks, delivering up to 3 times higher throughput and better recall than state-of-the-art hybrid and graph-based systems. Theoretically, we provide explicit error bounds and parameter selection rules that make FusedANN practical for production. This establishes a principled, scalable, and verifiable bridge between symbolic constraints and vector similarity, unlocking a new generation of filtered retrieval systems for large, hybrid, and dynamic NLP/ML workloads. 

**Abstract (ZH)**: 融合属性向量最近邻搜索：FusedANN及其在混合查询优化中的应用 

---
# Dynamicasome: a molecular dynamics-guided and AI-driven pathogenicity prediction catalogue for all genetic mutations 

**Title (ZH)**: 基于分子动力学引导和AI驱动的遗传变异致病性预测目录 

**Authors**: Naeyma N Islam, Mathew A Coban, Jessica M Fuller, Caleb Weber, Rohit Chitale, Benjamin Jussila, Trisha J. Brock, Cui Tao, Thomas R Caulfield  

**Link**: [PDF](https://arxiv.org/pdf/2509.19766)  

**Abstract**: Advances in genomic medicine accelerate the identi cation of mutations in disease-associated genes, but the pathogenicity of many mutations remains unknown, hindering their use in diagnostics and clinical decision-making. Predictive AI models are generated to combat this issue, but current tools display low accuracy when tested against functionally validated datasets. We show that integrating detailed conformational data extracted from molecular dynamics simulations (MDS) into advanced AI-based models increases their predictive power. We carry out an exhaustive mutational analysis of the disease gene PMM2 and subject structural models of each variant to MDS. AI models trained on this dataset outperform existing tools when predicting the known pathogenicity of mutations. Our best performing model, a neuronal networks model, also predicts the pathogenicity of several PMM2 mutations currently considered of unknown signi cance. We believe this model helps alleviate the burden of unknown variants in genomic medicine. 

**Abstract (ZH)**: 基因组医学的进步加速了与疾病相关基因突变的识别，但许多突变的致病性仍不明朗，阻碍了其在诊断和临床决策中的应用。为了解决这一问题，生成了预测AI模型，但现有工具在验证功能性数据集上的准确性较低。我们展示了一种将详细的构象数据集成到高级基于AI的模型中可以提高其预测能力的方法。我们对疾病基因PMM2进行了详尽的突变分析，并对每个变异体的结构模型进行了分子动力学模拟（MDS）。基于此数据集训练的AI模型在预测已知致病性突变方面优于现有工具。我们表现最佳的模型，神经网络模型，还能预测几个目前被认为意义不明的PMM2突变的致病性。我们相信该模型有助于减轻基因组医学中未知变异体的负担。 

---
# ARCADE: A Real-Time Data System for Hybrid and Continuous Query Processing across Diverse Data Modalities 

**Title (ZH)**: ARCADE: 一种跨异构数据模态的实时数据系统，支持混合查询与连续查询处理 

**Authors**: Jingyi Yang, Songsong Mo, Jiachen Shi, Zihao Yu, Kunhao Shi, Xuchen Ding, Gao Cong  

**Link**: [PDF](https://arxiv.org/pdf/2509.19757)  

**Abstract**: The explosive growth of multimodal data - spanning text, image, video, spatial, and relational modalities, coupled with the need for real-time semantic search and retrieval over these data - has outpaced the capabilities of existing multimodal and real-time database systems, which either lack efficient ingestion and continuous query capability, or fall short in supporting expressive hybrid analytics. We introduce ARCADE, a real-time data system that efficiently supports high-throughput ingestion and expressive hybrid and continuous query processing across diverse data types. ARCADE introduces unified disk-based secondary index on LSM-based storage for vector, spatial, and text data modalities, a comprehensive cost-based query optimizer for hybrid queries, and an incremental materialized view framework for efficient continuous queries. Built on open-source RocksDB storage and MySQL query engine, ARCADE outperforms leading multimodal data systems by up to 7.4x on read-heavy and 1.4x on write-heavy workloads. 

**Abstract (ZH)**: 多模态数据的爆炸性增长及其对实时语义搜索和检索的需求超出了现有多模态和实时数据库系统的能力，这些系统要么缺乏高效的摄入和连续查询能力，要么在支持表达性强的混合分析方面有所欠缺。我们提出ARCADE，这是一种能够高效支持高吞吐量数据摄入和跨多种数据类型进行表达性强的混合和连续查询处理的实时数据系统。ARCADE引入了一种适用于LSM存储的统一磁盘二级索引，用于向量、空间和文本数据模态，全面的成本基量子查询优化器，以及增量物化视图框架，以实现高效的连续查询。基于开源RocksDB存储和MySQL查询引擎，ARCADE在读密集型工作负载上性能比领先多模态数据系统高出7.4倍，在写密集型工作负载上高出1.4倍。 

---
# ExpFace: Exponential Angular Margin Loss for Deep Face Recognition 

**Title (ZH)**: ExpFace: 指数视角.margin 损失函数用于深度面部识别 

**Authors**: Jinhui Zheng, Xueyuan Gong  

**Link**: [PDF](https://arxiv.org/pdf/2509.19753)  

**Abstract**: Face recognition is an open-set problem requiring high discriminative power to ensure that intra-class distances remain smaller than inter-class distances. Margin-based softmax losses, such as SphereFace, CosFace, and ArcFace, have been widely adopted to enhance intra-class compactness and inter-class separability, yet they overlook the impact of noisy samples. By examining the distribution of samples in the angular space, we observe that clean samples predominantly cluster in the center region, whereas noisy samples tend to shift toward the peripheral region. Motivated by this observation, we propose the Exponential Angular Margin Loss (ExpFace), which introduces an angular exponential term as the margin. This design applies a larger penalty in the center region and a smaller penalty in the peripheral region within the angular space, thereby emphasizing clean samples while suppressing noisy samples. We present a unified analysis of ExpFace and classical margin-based softmax losses in terms of margin embedding forms, similarity curves, and gradient curves, showing that ExpFace not only avoids the training instability of SphereFace and the non-monotonicity of ArcFace, but also exhibits a similarity curve that applies penalties in the same manner as the decision boundary in the angular space. Extensive experiments demonstrate that ExpFace achieves state-of-the-art performance. To facilitate future research, we have released the source code at: this https URL. 

**Abstract (ZH)**: 基于指数角边际损失的面部识别（ExpFace：一种基于指数角边际损失的面部识别方法） 

---
# Cuffless Blood Pressure Prediction from Speech Sentences using Deep Learning Methods 

**Title (ZH)**: 基于深度学习方法的无袖带血压预测从语音句子出发 

**Authors**: Kainat  

**Link**: [PDF](https://arxiv.org/pdf/2509.19750)  

**Abstract**: This research presents a novel method for noninvasive arterial blood pressure ABP prediction using speech signals employing a BERT based regression model Arterial blood pressure is a vital indicator of cardiovascular health and accurate monitoring is essential in preventing hypertension related complications Traditional cuff based methods often yield inconsistent results due to factors like whitecoat and masked hypertension Our approach leverages the acoustic characteristics of speech capturing voice features to establish correlations with blood pressure levels Utilizing advanced deep learning techniques we analyze speech signals to extract relevant patterns enabling real time monitoring without the discomfort of conventional methods In our study we employed a dataset comprising recordings from 95 participants ensuring diverse representation The BERT model was fine tuned on extracted features from speech leading to impressive performance metrics achieving a mean absolute error MAE of 136 mmHg for systolic blood pressure SBP and 124 mmHg for diastolic blood pressure DBP with R scores of 099 and 094 respectively These results indicate the models robustness in accurately predicting blood pressure levels Furthermore the training and validation loss analysis demonstrates effective learning and minimal overfitting Our findings suggest that integrating deep learning with speech analysis presents a viable alternative for blood pressure monitoring paving the way for improved applications in telemedicine and remote health monitoring By providing a user friendly and accurate method for blood pressure assessment this research has significant implications for enhancing patient care and proactive management of cardiovascular health 

**Abstract (ZH)**: 本研究提出了一种使用基于BERT的回归模型并通过语音信号预测无创动脉血压的新方法 动脉血压是心血管健康的重要指标，准确监测对于预防与高血压相关并发症至关重要 传统袖带方法往往因白大衣高血压和隐匿性高血压等因素导致结果不一致 我们的方法利用语音的声学特征，捕获声音特征并与血压水平建立关联 利用先进的深度学习技术，分析语音信号以提取相关模式，实现无不适的实时监测 在我们的研究中，我们使用了一个由95名参与者提供的数据集，确保了多样性的代表 总结报告的BERT模型在提取的语音特征上进行了微调，获得了出色的性能指标，收缩压SBP的平均绝对误差MAE为13.6 mmHg，舒张压DBP的MAE为12.4 mmHg，相关系数R分别为0.99和0.94 这些结果表明，该模型在准确预测血压方面具有很强的稳健性 通过训练和验证损失分析，展示了有效的学习和轻度过拟合 我们的研究结果表明，将深度学习与语音分析结合起来，为血压监测提供了一种可行的替代方案，为远程医疗和远程健康管理提供了改进的应用前景 通过提供一种用户友好且准确的血压评估方法，本研究对提高患者护理和主动管理心血管健康具有重要意义 

---
# HiCoLoRA: Addressing Context-Prompt Misalignment via Hierarchical Collaborative LoRA for Zero-Shot DST 

**Title (ZH)**: HiCoLoRA：通过分层协作LoRA解决上下文提示不匹配问题的零样本对话状态跟踪 

**Authors**: Shuyu Zhang, Yifan Wei, Xinru Wang, Yanmin Zhu, Yangfan He, Yixuan Weng, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19742)  

**Abstract**: Zero-shot Dialog State Tracking (zs-DST) is essential for enabling Task-Oriented Dialog Systems (TODs) to generalize to new domains without costly data annotation. A central challenge lies in the semantic misalignment between dynamic dialog contexts and static prompts, leading to inflexible cross-layer coordination, domain interference, and catastrophic forgetting. To tackle this, we propose Hierarchical Collaborative Low-Rank Adaptation (HiCoLoRA), a framework that enhances zero-shot slot inference through robust prompt alignment. It features a hierarchical LoRA architecture for dynamic layer-specific processing (combining lower-layer heuristic grouping and higher-layer full interaction), integrates Spectral Joint Domain-Slot Clustering to identify transferable associations (feeding an Adaptive Linear Fusion Mechanism), and employs Semantic-Enhanced SVD Initialization (SemSVD-Init) to preserve pre-trained knowledge. Experiments on multi-domain datasets MultiWOZ and SGD show that HiCoLoRA outperforms baselines, achieving SOTA in zs-DST. Code is available at this https URL. 

**Abstract (ZH)**: 零样本对话状态跟踪（zs-DST）对于使面向任务的对话系统（TODs）能够在无需昂贵数据标注的情况下泛化到新领域至关重要。该方法的关键挑战在于动态对话上下文与静态提示之间的语义不匹配，导致跨层协调僵化、领域干扰和灾难性遗忘。为此，我们提出了一种分层协作低秩适应（HiCoLoRA）框架，该框架通过稳健的提示对齐来增强零样本插槽推理。该框架包含一种分层LoRA架构，进行动态层特定处理（结合下层启发式分组和上层全方位交互）、整合光谱联合领域-插槽聚类以识别可转移的关联（为自适应线性融合机制提供支持）以及采用语义增强SVD初始化（SemSVD-Init）以保留预训练知识。在多领域数据集MultiWOZ和SGD上的实验表明，HiCoLoRA在零样本对话状态跟踪（zs-DST）中优于基线方法，达到最佳效果。代码见此链接。 

---
# SMILES-Inspired Transfer Learning for Quantum Operators in Generative Quantum Eigensolver 

**Title (ZH)**: 基于SMILES的转移学习在生成型量子本证求解器中的应用 

**Authors**: Zhi Yin, Xiaoran Li, Shengyu Zhang, Xin Li, Xiaojin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19715)  

**Abstract**: Given the inherent limitations of traditional Variational Quantum Eigensolver(VQE) algorithms, the integration of deep generative models into hybrid quantum-classical frameworks, specifically the Generative Quantum Eigensolver(GQE), represents a promising innovative approach. However, taking the Unitary Coupled Cluster with Singles and Doubles(UCCSD) ansatz which is widely used in quantum chemistry as an example, different molecular systems require constructions of distinct quantum operators. Considering the similarity of different molecules, the construction of quantum operators utilizing the similarity can reduce the computational cost significantly. Inspired by the SMILES representation method in computational chemistry, we developed a text-based representation approach for UCCSD quantum operators by leveraging the inherent representational similarities between different molecular systems. This framework explores text pattern similarities in quantum operators and employs text similarity metrics to establish a transfer learning framework. Our approach with a naive baseline setting demonstrates knowledge transfer between different molecular systems for ground-state energy calculations within the GQE paradigm. This discovery offers significant benefits for hybrid quantum-classical computation of molecular ground-state energies, substantially reducing computational resource requirements. 

**Abstract (ZH)**: 基于深度生成模型的生成量子本征值求解器在混合量子-经典框架中的应用：利用分子相似性减少量子操作构造成本 

---
# Intuition to Evidence: Measuring AI's True Impact on Developer Productivity 

**Title (ZH)**: 直觉到证据：衡量AI对开发者 productivity 的真正影响 

**Authors**: Anand Kumar, Vishal Khare, Deepak Sharma, Satyam Kumar, Vijay Saini, Anshul Yadav, Sachendra Jain, Ankit Rana, Pratham Verma, Vaibhav Meena, Avinash Edubilli  

**Link**: [PDF](https://arxiv.org/pdf/2509.19708)  

**Abstract**: We present a comprehensive real-world evaluation of AI-assisted software development tools deployed at enterprise scale. Over one year, 300 engineers across multiple teams integrated an in-house AI platform (DeputyDev) that combines code generation and automated review capabilities into their daily workflows. Through rigorous cohort analysis, our study demonstrates statistically significant productivity improvements, including an overall 31.8% reduction in PR review cycle time.
Developer adoption was strong, with 85% satisfaction for code review features and 93% expressing a desire to continue using the platform. Adoption patterns showed systematic scaling from 4% engagement in month 1 to 83% peak usage by month 6, stabilizing at 60% active engagement. Top adopters achieved a 61% increase in code volume pushed to production, contributing to approximately 30 to 40% of code shipped to production through this tool, accounting for an overall 28% increase in code shipment volume.
Unlike controlled benchmark evaluations, our longitudinal analysis provides empirical evidence from production environments, revealing both the transformative potential and practical deployment challenges of integrating AI into enterprise software development workflows. 

**Abstract (ZH)**: 我们对企业规模部署的AI辅助软件开发工具进行了全面的实际评价。在一年的时间里，多个团队中的300名工程师整合了一个内部AI平台（DeputyDev），该平台结合了代码生成和自动化审查能力，应用于日常工作中。通过严格的群体分析，我们的研究证明了统计上显著的生产力提升，包括总体上将代码审查循环时间减少了31.8%。开发者对代码审查功能的采用率很高，85%的开发者表示满意，并希望继续使用该平台。采用模式显示了系统化的规模化，从第一个月的4%参与度增加到第六个月的83%峰值使用率，稳定在60%的活跃参与度。主要采用者将代码推送至生产环境的数量增加了61%，并通过该工具贡献了约30%到40%的代码，导致整体代码推送量增加了28%。与受控基准评估不同，我们的 longitudinal 分析提供了来自生产环境的实证证据，揭示了将AI整合到企业软件开发工作流中具有变革性的潜力及其实际部署挑战。 

---
# Causal Machine Learning for Surgical Interventions 

**Title (ZH)**: 因果机器学习在手术干预中的应用 

**Authors**: J. Ben Tamo, Nishant S. Chouhan, Micky C. Nnamdi, Yining Yuan, Shreya S. Chivilkar, Wenqi Shi, Steven W. Hwang, B. Randall Brenn, May D. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19705)  

**Abstract**: Surgical decision-making is complex and requires understanding causal relationships between patient characteristics, interventions, and outcomes. In high-stakes settings like spinal fusion or scoliosis correction, accurate estimation of individualized treatment effects (ITEs) remains limited due to the reliance on traditional statistical methods that struggle with complex, heterogeneous data. In this study, we develop a multi-task meta-learning framework, X-MultiTask, for ITE estimation that models each surgical decision (e.g., anterior vs. posterior approach, surgery vs. no surgery) as a distinct task while learning shared representations across tasks. To strengthen causal validity, we incorporate the inverse probability weighting (IPW) into the training objective. We evaluate our approach on two datasets: (1) a public spinal fusion dataset (1,017 patients) to assess the effect of anterior vs. posterior approaches on complication severity; and (2) a private AIS dataset (368 patients) to analyze the impact of posterior spinal fusion (PSF) vs. non-surgical management on patient-reported outcomes (PROs). Our model achieves the highest average AUC (0.84) in the anterior group and maintains competitive performance in the posterior group (0.77). It outperforms baselines in treatment effect estimation with the lowest overall $\epsilon_{\text{NN-PEHE}}$ (0.2778) and $\epsilon_{\text{ATE}}$ (0.0763). Similarly, when predicting PROs in AIS, X-MultiTask consistently shows superior performance across all domains, with $\epsilon_{\text{NN-PEHE}}$ = 0.2551 and $\epsilon_{\text{ATE}}$ = 0.0902. By providing robust, patient-specific causal estimates, X-MultiTask offers a powerful tool to advance personalized surgical care and improve patient outcomes. The code is available at this https URL. 

**Abstract (ZH)**: 多任务元学习框架X-MultiTask在个体治疗效果估计中的应用 

---
# Linear Transformers Implicitly Discover Unified Numerical Algorithms 

**Title (ZH)**: 线性变压器隐式发现统一数值算法 

**Authors**: Patrick Lutz, Aditya Gangrade, Hadi Daneshmand, Venkatesh Saligrama  

**Link**: [PDF](https://arxiv.org/pdf/2509.19702)  

**Abstract**: We train a linear attention transformer on millions of masked-block matrix completion tasks: each prompt is masked low-rank matrix whose missing block may be (i) a scalar prediction target or (ii) an unseen kernel slice of Nyström extrapolation. The model sees only input-output pairs and a mean-squared loss; it is given no normal equations, no handcrafted iterations, and no hint that the tasks are related. Surprisingly, after training, algebraic unrolling reveals the same parameter-free update rule across three distinct computational regimes (full visibility, rank-limited updates, and distributed computation). We prove that this rule achieves second-order convergence on full-batch problems, cuts distributed iteration complexity, and remains accurate with rank-limited attention. Thus, a transformer trained solely to patch missing blocks implicitly discovers a unified, resource-adaptive iterative solver spanning prediction, estimation, and Nyström extrapolation, highlighting a powerful capability of in-context learning. 

**Abstract (ZH)**: 我们训练了一个线性注意变换器在数百万个遮蔽块矩阵完成任务上：每个提示是一个被遮蔽的低秩矩阵，其缺失的块可能是（i）一个标量预测目标或（ii）Nyström外推的未见内核切片。模型仅看到输入-输出对和均方损失；它没有得到正规方程、手工设计的迭代步骤，也没有被告知这些任务是相关的。令人惊讶的是，训练后，代数展开揭示了三个不同的计算范式（全视图、秩受限更新和分布式计算）下相同的无参数更新规则。我们证明，在批量问题上，该规则实现了二阶收敛，在分布式迭代复杂性上进行了削减，并能够使用秩受限注意保持准确性。因此，仅用于修补缺失块的变换器隐式发现了一个统一的、资源自适应的迭代求解器，适用于预测、估计和Nyström外推，突显了上下文学习的强大能力。 

---
# A Unified Noise-Curvature View of Loss of Trainability 

**Title (ZH)**: 统一的噪声-曲率视角下的训练可学性丧失 

**Authors**: Gunbir Singh Baveja, Mark Schmidt  

**Link**: [PDF](https://arxiv.org/pdf/2509.19698)  

**Abstract**: Loss of trainability (LoT) in continual learning occurs when gradient steps no longer yield improvement as tasks evolve, so accuracy stalls or degrades despite adequate capacity and supervision. We analyze LoT incurred with Adam through an optimization lens and find that single indicators such as Hessian rank, sharpness level, weight or gradient norms, gradient-to-parameter ratios, and unit-sign entropy are not reliable predictors. Instead we introduce two complementary criteria: a batch-size-aware gradient-noise bound and a curvature volatility-controlled bound that combine into a per-layer predictive threshold that anticipates trainability behavior. Using this threshold, we build a simple per-layer scheduler that keeps each layers effective step below a safe limit, stabilizing training and improving accuracy across concatenated ReLU (CReLU), Wasserstein regularization, and L2 weight decay, with learned learning-rate trajectories that mirror canonical decay. 

**Abstract (ZH)**: 连续学习中训练能力的丧失（Loss of Trainability，LoT）发生在随任务演进梯度步长不再带来改善的情况下，因此准确度停滞或恶化，尽管具有足够的能力和监督。通过优化视角分析Adam引起的训练能力丧失，我们发现单一指标如海森矩阵秩、尖度水平、权重或梯度范数、梯度-参数比以及单位符号熵不可靠。相反，我们引入了两个互补的标准：批量大小感知的梯度噪声界限和曲率波动控制界限，它们结合成每层预测阈值，以预见训练能力行为。利用该阈值，我们构建了一个简单的每层调度器，使每层的有效步骤保持在安全限制以下，稳定训练并提高跨ReLU、Wasserstein正则化和L2权重衰减的准确度，伴随学习率轨迹模仿标准衰减。 

---
# Diffusion-Based Impedance Learning for Contact-Rich Manipulation Tasks 

**Title (ZH)**: 基于扩散的阻抗学习在接触丰富的操作任务中 

**Authors**: Noah Geiger, Tamim Asfour, Neville Hogan, Johannes Lachner  

**Link**: [PDF](https://arxiv.org/pdf/2509.19696)  

**Abstract**: Learning methods excel at motion generation in the information domain but are not primarily designed for physical interaction in the energy domain. Impedance Control shapes physical interaction but requires task-aware tuning by selecting feasible impedance parameters. We present Diffusion-Based Impedance Learning, a framework that combines both domains. A Transformer-based Diffusion Model with cross-attention to external wrenches reconstructs a simulated Zero-Force Trajectory (sZFT). This captures both translational and rotational task-space behavior. For rotations, we introduce a novel SLERP-based quaternion noise scheduler that ensures geometric consistency. The reconstructed sZFT is then passed to an energy-based estimator that updates stiffness and damping parameters. A directional rule is applied that reduces impedance along non task axes while preserving rigidity along task directions. Training data were collected for a parkour scenario and robotic-assisted therapy tasks using teleoperation with Apple Vision Pro. With only tens of thousands of samples, the model achieved sub-millimeter positional accuracy and sub-degree rotational accuracy. Its compact model size enabled real-time torque control and autonomous stiffness adaptation on a KUKA LBR iiwa robot. The controller achieved smooth parkour traversal within force and velocity limits and 30/30 success rates for cylindrical, square, and star peg insertions without any peg-specific demonstrations in the training data set. All code for the Transformer-based Diffusion Model, the robot controller, and the Apple Vision Pro telemanipulation framework is publicly available. These results mark an important step towards Physical AI, fusing model-based control for physical interaction with learning-based methods for trajectory generation. 

**Abstract (ZH)**: 基于扩散的阻抗学习：融合信息域的学习方法与能量域的物理交互 

---
# DyBBT: Dynamic Balance via Bandit inspired Targeting for Dialog Policy with Cognitive Dual-Systems 

**Title (ZH)**: DyBBT：基于 bandit 启发式目标的认知双系统对话策略动态平衡方法 

**Authors**: Shuyu Zhang, Yifan Wei, Jialuo Yuan, Xinru Wang, Yanmin Zhu, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19695)  

**Abstract**: Task oriented dialog systems often rely on static exploration strategies that do not adapt to dynamic dialog contexts, leading to inefficient exploration and suboptimal performance. We propose DyBBT, a novel dialog policy learning framework that formalizes the exploration challenge through a structured cognitive state space capturing dialog progression, user uncertainty, and slot dependency. DyBBT proposes a bandit inspired meta-controller that dynamically switches between a fast intuitive inference (System 1) and a slow deliberative reasoner (System 2) based on real-time cognitive states and visitation counts. Extensive experiments on single- and multi-domain benchmarks show that DyBBT achieves state-of-the-art performance in success rate, efficiency, and generalization, with human evaluations confirming its decisions are well aligned with expert judgment. Code is available at this https URL. 

**Abstract (ZH)**: 面向任务的对话系统 often 依赖于静态探索策略，这些策略不能适应动态对话上下文，导致探索效率低下和性能不佳。我们提出了 DyBBT，一种通过结构化认知状态空间形式化探索挑战的新对话策略学习框架，该空间捕获对话进程、用户不确定性以及槽位依赖关系。DyBBT 建议了一个基于多臂bandit的元控制器，该控制器根据实时认知状态和访问次数动态切换快速直观推理（系统1）和缓慢的详细推理器（系统2）。在单域和多域基准测试中的 extensive 实验显示，DyBBT 在成功率、效率和泛化能力方面均达到最新水平，人类评估进一步证实其决策与专家判断高度一致。代码可在以下链接获取：this https URL。 

---
# PolicyPad: Collaborative Prototyping of LLM Policies 

**Title (ZH)**: PolicyPad: 共同设计大型语言模型策略的协作原型制作 

**Authors**: K. J. Kevin Feng, Tzu-Sheng Kuo, Quan Ze, Chen, Inyoung Cheong, Kenneth Holstein, Amy X. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19680)  

**Abstract**: As LLMs gain adoption in high-stakes domains like mental health, domain experts are increasingly consulted to provide input into policies governing their behavior. From an observation of 19 policymaking workshops with 9 experts over 15 weeks, we identified opportunities to better support rapid experimentation, feedback, and iteration for collaborative policy design processes. We present PolicyPad, an interactive system that facilitates the emerging practice of LLM policy prototyping by drawing from established UX prototyping practices, including heuristic evaluation and storyboarding. Using PolicyPad, policy designers can collaborate on drafting a policy in real time while independently testing policy-informed model behavior with usage scenarios. We evaluate PolicyPad through workshops with 8 groups of 22 domain experts in mental health and law, finding that PolicyPad enhanced collaborative dynamics during policy design, enabled tight feedback loops, and led to novel policy contributions. Overall, our work paves participatory paths for advancing AI alignment and safety. 

**Abstract (ZH)**: 随着大型语言模型在高 stakes 领域如心理健康中的应用不断增加，领域专家 increasingly 被咨询以提供其行为治理政策的输入。通过观察 15 周内 9 位专家参与的 19 次政策制定研讨会，我们识别出支持快速试验、反馈和迭代的协作政策设计过程的机会。我们提出 PolicyPad，一个交互系统，通过借鉴已有的 UX 原型设计实践，包括启发式评估和故事情景构建，促进大型语言模型政策原型的设计。使用 PolicyPad，政策设计师可以实时协作制定政策，同时独立地使用使用场景测试政策导向的模型行为。通过与 22 位心理健康和法律领域专家分成 8 个小组的工作坊评估 PolicyPad，我们发现 PolicyPad 增强了政策设计中的协作动态，实现了紧密的反馈循环，并促进了新的政策贡献。总体而言，我们的工作为推动 AI 对齐和安全性的参与路径铺平了道路。 

---
# Thinking While Listening: Simple Test Time Scaling For Audio Classification 

**Title (ZH)**: 思考while倾听：简单的测试时延调整方法用于音频分类 

**Authors**: Prateek Verma, Mert Pilanci  

**Link**: [PDF](https://arxiv.org/pdf/2509.19676)  

**Abstract**: We propose a framework that enables neural models to "think while listening" to everyday sounds, thereby enhancing audio classification performance. Motivated by recent advances in the reasoning capabilities of large language models, we address two central questions: (i) how can thinking be incorporated into existing audio classification pipelines to enable reasoning in the category space and improve performance, and (ii) can a new architecture be designed from the ground up to support both thinking and test-time scaling? We demonstrate that in both settings, our models exhibit improved classification accuracy. Leveraging test-time scaling, we observe consistent gains as the number of sampled traces increases. Furthermore, we evaluate two open-source reasoning models, GPT-OSS-20B and Qwen3-14B, showing that while such models are capable of zero-shot reasoning, a lightweight approach--retraining only the embedding matrix of a frozen, smaller model like GPT-2--can surpass the performance of billion-parameter text-based reasoning models. 

**Abstract (ZH)**: 我们提出了一种框架，使神经模型能够在聆听日常声音时进行“思考”，从而提高音频分类性能。受大型语言模型推理能力近期进展的启发，我们探讨了两个核心问题：(i) 如何将“思考”纳入现有的音频分类管道中，以在类别空间中进行推理并提高性能，以及(ii) 是否可以从头开始设计一种新的架构，以支持同时进行“思考”和测试时的扩展？我们证明，在这两种设置下，我们的模型都显示出了改进的分类准确性。利用测试时的扩展，我们观察到，随着采样轨迹数量的增加，持续观察到性能改进。此外，我们评估了两个开源推理模型GPT-OSS-20B和Qwen3-14B，结果显示，虽然这些模型能够进行零样本推理，但一种轻量级的方法——仅重新训练冻结的小型模型（如GPT-2）的嵌入矩阵——可以超越基于数十亿参数文本的推理模型的性能。 

---
# Games Are Not Equal: Classifying Cloud Gaming Contexts for Effective User Experience Measurement 

**Title (ZH)**: 游戏并非平等：基于有效用户体验测量的游戏云环境分类 

**Authors**: Yifan Wang, Minzhao Lyu, Vijay Sivaraman  

**Link**: [PDF](https://arxiv.org/pdf/2509.19669)  

**Abstract**: To tap into the growing market of cloud gaming, whereby game graphics is rendered in the cloud and streamed back to the user as a video feed, network operators are creating monetizable assurance services that dynamically provision network resources. However, without accurately measuring cloud gaming user experience, they cannot assess the effectiveness of their provisioning methods. Basic measures such as bandwidth and frame rate by themselves do not suffice, and can only be interpreted in the context of the game played and the player activity within the game. This paper equips the network operator with a method to obtain a real-time measure of cloud gaming experience by analyzing network traffic, including contextual factors such as the game title and player activity stage. Our method is able to classify the game title within the first five seconds of game launch, and continuously assess the player activity stage as being active, passive, or idle. We deploy it in an ISP hosting NVIDIA cloud gaming servers for the region. We provide insights from hundreds of thousands of cloud game streaming sessions over a three-month period into the dependence of bandwidth consumption and experience level on the gameplay contexts. 

**Abstract (ZH)**: 利用网络流量分析方法实时评估云游戏体验：基于游戏标题和玩家活动阶段的分类方法 

---
# Selective Classifier-free Guidance for Zero-shot Text-to-speech 

**Title (ZH)**: 零样本文本到语音的选择性无分类器引导 

**Authors**: John Zheng, Farhad Maleki  

**Link**: [PDF](https://arxiv.org/pdf/2509.19668)  

**Abstract**: In zero-shot text-to-speech, achieving a balance between fidelity to the target speaker and adherence to text content remains a challenge. While classifier-free guidance (CFG) strategies have shown promising results in image generation, their application to speech synthesis are underexplored. Separating the conditions used for CFG enables trade-offs between different desired characteristics in speech synthesis. In this paper, we evaluate the adaptability of CFG strategies originally developed for image generation to speech synthesis and extend separated-condition CFG approaches for this domain. Our results show that CFG strategies effective in image generation generally fail to improve speech synthesis. We also find that we can improve speaker similarity while limiting degradation of text adherence by applying standard CFG during early timesteps and switching to selective CFG only in later timesteps. Surprisingly, we observe that the effectiveness of a selective CFG strategy is highly text-representation dependent, as differences between the two languages of English and Mandarin can lead to different results even with the same model. 

**Abstract (ZH)**: 零样本文本到语音合成中，如何在忠于目标说话人和遵循文本内容之间取得平衡仍然是一个挑战。尽管无分类器引导（CFG）策略在图像生成中显示出有前景的结果，但其在语音合成中的应用仍待探索。通过对用于CFG的条件进行分离，可以在语音合成中实现不同期望特征之间的权衡。在本文中，我们评估了最初为图像生成设计的CFG策略在语音合成中的适应性，并扩展了分离条件的CFG方法适用于此领域。我们的结果表明，有效的CFG策略通常无法改善语音合成。我们还发现，在早期时间步长应用标准CFG，并在后期时间步长切换为选择性CFG，可以在限制文本一致性退化的同时改善说话人相似度。令人惊讶的是，我们观察到选择性CFG策略的有效性高度依赖于文本表示，不同语言（如英语和 Mandarin）之间的差异即使在相同模型下也会导致不同结果。 

---
# MoTiC: Momentum Tightness and Contrast for Few-Shot Class-Incremental Learning 

**Title (ZH)**: MoTiC: 动量紧致性和对比度在少量样本类别增量学习中的应用 

**Authors**: Zeyu He, Shuai Huang, Yuwu Lu, Ming Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.19664)  

**Abstract**: Few-Shot Class-Incremental Learning (FSCIL) must contend with the dual challenge of learning new classes from scarce samples while preserving old class knowledge. Existing methods use the frozen feature extractor and class-averaged prototypes to mitigate against catastrophic forgetting and overfitting. However, new-class prototypes suffer significant estimation bias due to extreme data scarcity, whereas base-class prototypes benefit from sufficient data. In this work, we theoretically demonstrate that aligning the new-class priors with old-class statistics via Bayesian analysis reduces variance and improves prototype accuracy. Furthermore, we propose large-scale contrastive learning to enforce cross-category feature tightness. To further enrich feature diversity and inject prior information for new-class prototypes, we integrate momentum self-supervision and virtual categories into the Momentum Tightness and Contrast framework (MoTiC), constructing a feature space with rich representations and enhanced interclass cohesion. Experiments on three FSCIL benchmarks produce state-of-the-art performances, particularly on the fine-grained task CUB-200, validating our method's ability to reduce estimation bias and improve incremental learning robustness. 

**Abstract (ZH)**: Few-Shot Class-Incremental Learning via Bayesian Analysis and Large-Scale Contrastive Learning 

---
# RoboSSM: Scalable In-context Imitation Learning via State-Space Models 

**Title (ZH)**: RoboSSM：基于状态空间模型的可扩展上下文模仿学习 

**Authors**: Youngju Yoo, Jiaheng Hu, Yifeng Zhu, Bo Liu, Qiang Liu, Roberto Martín-Martín, Peter Stone  

**Link**: [PDF](https://arxiv.org/pdf/2509.19658)  

**Abstract**: In-context imitation learning (ICIL) enables robots to learn tasks from prompts consisting of just a handful of demonstrations. By eliminating the need for parameter updates at deployment time, this paradigm supports few-shot adaptation to novel tasks. However, recent ICIL methods rely on Transformers, which have computational limitations and tend to underperform when handling longer prompts than those seen during training. In this work, we introduce RoboSSM, a scalable recipe for in-context imitation learning based on state-space models (SSM). Specifically, RoboSSM replaces Transformers with Longhorn -- a state-of-the-art SSM that provides linear-time inference and strong extrapolation capabilities, making it well-suited for long-context prompts. We evaluate our approach on the LIBERO benchmark and compare it against strong Transformer-based ICIL baselines. Experiments show that RoboSSM extrapolates effectively to varying numbers of in-context demonstrations, yields high performance on unseen tasks, and remains robust in long-horizon scenarios. These results highlight the potential of SSMs as an efficient and scalable backbone for ICIL. Our code is available at this https URL. 

**Abstract (ZH)**: 基于状态空间模型的在上下文模仿学习（RoboSSM） 

---
# Large Language Models for Pedestrian Safety: An Application to Predicting Driver Yielding Behavior at Unsignalized Intersections 

**Title (ZH)**: 大型语言模型在行人安全中的应用：以无信号交叉口驾驶员让行行为预测为例 

**Authors**: Yicheng Yang, Zixian Li, Jean Paul Bizimana, Niaz Zafri, Yongfeng Dong, Tianyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19657)  

**Abstract**: Pedestrian safety is a critical component of urban mobility and is strongly influenced by the interactions between pedestrian decision-making and driver yielding behavior at crosswalks. Modeling driver--pedestrian interactions at intersections requires accurately capturing the complexity of these behaviors. Traditional machine learning models often struggle to capture the nuanced and context-dependent reasoning required for these multifactorial interactions, due to their reliance on fixed feature representations and limited interpretability. In contrast, large language models (LLMs) are suited for extracting patterns from heterogeneous traffic data, enabling accurate modeling of driver-pedestrian interactions. Therefore, this paper leverages multimodal LLMs through a novel prompt design that incorporates domain-specific knowledge, structured reasoning, and few-shot prompting, enabling interpretable and context-aware inference of driver yielding behavior, as an example application of modeling pedestrian--driver interaction. We benchmarked state-of-the-art LLMs against traditional classifiers, finding that GPT-4o consistently achieves the highest accuracy and recall, while Deepseek-V3 excels in precision. These findings highlight the critical trade-offs between model performance and computational efficiency, offering practical guidance for deploying LLMs in real-world pedestrian safety systems. 

**Abstract (ZH)**: 行人安全是城市交通中的一个关键组成部分，受到行人决策与驾驶员在人行横道处让行行为之间互动的强烈影响。通过交叉口建模驾驶员-行人的互动需要准确捕捉这些行为的复杂性。传统机器学习模型往往难以捕捉这些多因素互动所需的细微且上下文依赖的推理，因为它们依赖于固定特征表示和有限的可解释性。相比之下，大型语言模型（LLMs）适合从异构交通数据中提取模式，能够准确建模驾驶员-行人互动。因此，本文通过结合领域专业知识、结构化推理和少量示例提示的新型提示设计，利用多模态LLMs，实现对驾驶员让行行为的可解释且上下文感知的推断，作为建模行人-驾驶员互动的一个应用示例。我们将最先进的LLMs与传统分类器进行了基准测试，发现GPT-4o在准确性和召回率方面始终表现最好，而Deepseek-V3在精确性方面表现优异。这些发现阐明了模型性能与计算效率之间的关键权衡，为在实际行人安全系统中部署LLMs提供了实用指导。 

---
# Where 6G Stands Today: Evolution, Enablers, and Research Gaps 

**Title (ZH)**: 当前6G的发展状况：演进、使能技术与研究空白 

**Authors**: Salma Tika, Abdelkrim Haqiq, Essaid Sabir, Elmahdi Driouch  

**Link**: [PDF](https://arxiv.org/pdf/2509.19646)  

**Abstract**: As the fifth-generation (5G) mobile communication system continues its global deployment, both industry and academia have started conceptualizing the 6th generation (6G) to address the growing need for a progressively advanced and digital society. Even while 5G offers considerable advancements over LTE, it could struggle to be sufficient to meet all of the requirements, including ultra-high reliability, seamless automation, and ubiquitous coverage. In response, 6G is supposed to bring out a highly intelligent, automated, and ultra-reliable communication system that can handle a vast number of connected devices. This paper offers a comprehensive overview of 6G, beginning with its main stringent requirements while focusing on key enabling technologies such as terahertz (THz) communications, intelligent reflecting surfaces, massive MIMO and AI-driven networking that will shape the 6G networks. Furthermore, the paper lists various 6G applications and usage scenarios that will benefit from these advancements. At the end, we outline the potential challenges that must be addressed to achieve the 6G promises. 

**Abstract (ZH)**: 随着第五代（5G）移动通信系统的全球部署继续进行，产业和学术界已经开始构想第六代（6G）通信系统，以应对日益复杂和数字化社会的需求。尽管5G相比LTE提供了显著的进步，但在超高的可靠性和无缝自动化等方面可能仍然难以满足所有需求。为此，6G旨在构建一个高度智能化、自动化和 ultra-reliable 的通信系统，能够处理大量连接设备。本文提供了对6G的全面概述，重点介绍了其主要严格的规范要求，并着重讨论了太赫兹（THz）通信、智能反射表面、大规模MIMO和基于AI的网络等关键技术，这些技术将塑造6G网络。此外，本文还列出了将受益于这些进步的各种6G应用和使用场景。最后，我们概述了实现6G承诺时可能需要应对的潜在挑战。 

---
# Are We Scaling the Right Thing? A System Perspective on Test-Time Scaling 

**Title (ZH)**: 我们在攀爬正确的Things吗？从系统角度探讨测试时缩放 

**Authors**: Youpeng Zhao, Jinpeng LV, Di Wu, Jun Wang, Christopher Gooley  

**Link**: [PDF](https://arxiv.org/pdf/2509.19645)  

**Abstract**: Test-time scaling (TTS) has recently emerged as a promising direction to exploit the hidden reasoning capabilities of pre-trained large language models (LLMs). However, existing scaling methods narrowly focus on the compute-optimal Pareto-frontier, ignoring the simple fact that compute-optimal is not always system-optimal. In this work, we propose a system-driven perspective on TTS, analyzing how reasoning models scale against practical metrics, such as latency and cost-per-token. By evaluating the impact of popular optimizations such as tensor parallelism and speculative decoding, our preliminary analysis reveals the limitations of current methods and calls for a paradigm shift toward holistic, system-aware evaluations that capture the true essence of scaling laws at inference time. 

**Abstract (ZH)**: 基于系统的测试时规模优化：超越计算最优的综合评估 

---
# Mamba Modulation: On the Length Generalization of Mamba 

**Title (ZH)**: Mamba 调制：关于 Mamba 的长度泛化研究 

**Authors**: Peng Lu, Jerry Huang, Qiuhao Zeng, Xinyu Wang, Boxing Wang, Philippe Langlais, Yufei Cui  

**Link**: [PDF](https://arxiv.org/pdf/2509.19633)  

**Abstract**: The quadratic complexity of the attention mechanism in Transformer models has motivated the development of alternative architectures with sub-quadratic scaling, such as state-space models. Among these, Mamba has emerged as a leading architecture, achieving state-of-the-art results across a range of language modeling tasks. However, Mamba's performance significantly deteriorates when applied to contexts longer than those seen during pre-training, revealing a sharp sensitivity to context length extension. Through detailed analysis, we attribute this limitation to the out-of-distribution behaviour of its state-space dynamics, particularly within the parameterization of the state transition matrix $\mathbf{A}$. Unlike recent works which attribute this sensitivity to the vanished accumulation of discretization time steps, $\exp(-\sum_{t=1}^N\Delta_t)$, we establish a connection between state convergence behavior as the input length approaches infinity and the spectrum of the transition matrix $\mathbf{A}$, offering a well-founded explanation of its role in length extension. Next, to overcome this challenge, we propose an approach that applies spectrum scaling to pre-trained Mamba models to enable robust long-context generalization by selectively modulating the spectrum of $\mathbf{A}$ matrices in each layer. We show that this can significantly improve performance in settings where simply modulating $\Delta_t$ fails, validating our insights and providing avenues for better length generalization of state-space models with structured transition matrices. 

**Abstract (ZH)**: 基于注意力机制的二次复杂性促使开发了具有亚二次缩放的替代架构，如状态空间模型。其中，Mamba 凭借其在多种语言建模任务中取得的最优结果而崭露头角。然而，当应用于预训练中未见的更长上下文时，Mamba 的性能显著下降，显示出对上下文长度扩展的尖锐敏感性。通过对这一局限性的详细分析，我们将其归因于其状态空间动力学的离群行为，尤其是在状态转换矩阵 \(\mathbf{A}\) 的参数化中。不同于最近将这种敏感性归因于累积离散时间步的消失，\(\exp(-\sum_{t=1}^N\Delta_t)\)，我们建立了输入长度趋于无穷时状态收敛行为与转换矩阵 \(\mathbf{A}\) 的谱之间的联系，为 \(\mathbf{A}\) 的作用提供了坚实的理由。接下来，为了解决这一挑战，我们提出了一种方法，通过在预训练的 Mamba 模型中应用谱缩放，通过选择性地调节每个层的 \(\mathbf{A}\) 矩阵的谱来实现鲁棒的长上下文泛化。我们证明这种方法可以显著改善仅仅调节 \(\Delta_t\) 失败的情况下的性能，验证了我们的见解，并为具有结构状态转换矩阵的状态空间模型的长度泛化提供了改进途径。 

---
# Advancing Speech Summarization in Multi-modal LLMs with Reinforcement Learning 

**Title (ZH)**: 使用强化学习推进多模态LLM中的语音总结技术 

**Authors**: Shaoshi Ling, Gang Liu, Guoli Ye, Jinyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19631)  

**Abstract**: Speech summarization is a critical component of spoken content understanding, particularly in the era of rapidly growing spoken and audiovisual data. Recent advances in multi-modal large language models (MLLMs), leveraging the power of LLMs, enable generating textual summaries directly from speech without intermediate transcriptions, while supporting controllable styles and zero-shot generalization. However, open-source MLLMs continue to lag behind the state-of-the-art text-based LLMs, limiting their practical deployment for speech summarization. In this work, we present a novel multi-stage reinforcement learning training framework to enhance the speech summarization capabilities in MLLMs. Our model delivers substantial improvements over strong baselines, outperforms much larger MLLMs, and significantly narrows the gap with state-of-the-art text-based LLMs. 

**Abstract (ZH)**: 多模态大型语言模型的多阶段强化学习训练框架在语音摘要中的应用 

---
# Knowledge Base-Aware Orchestration: A Dynamic, Privacy-Preserving Method for Multi-Agent Systems 

**Title (ZH)**: 知识库意识型编排：多智能体系统的动态隐私保护方法 

**Authors**: Danilo Trombino, Vincenzo Pecorella, Alessandro de Giulii, Davide Tresoldi  

**Link**: [PDF](https://arxiv.org/pdf/2509.19599)  

**Abstract**: Multi-agent systems (MAS) are increasingly tasked with solving complex, knowledge-intensive problems where effective agent orchestration is critical. Conventional orchestration methods rely on static agent descriptions, which often become outdated or incomplete. This limitation leads to inefficient task routing, particularly in dynamic environments where agent capabilities continuously evolve. We introduce Knowledge Base-Aware (KBA) Orchestration, a novel approach that augments static descriptions with dynamic, privacy-preserving relevance signals derived from each agent's internal knowledge base (KB). In the proposed framework, when static descriptions are insufficient for a clear routing decision, the orchestrator prompts the subagents in parallel. Each agent then assesses the task's relevance against its private KB, returning a lightweight ACK signal without exposing the underlying data. These collected signals populate a shared semantic cache, providing dynamic indicators of agent suitability for future queries. By combining this novel mechanism with static descriptions, our method achieves more accurate and adaptive task routing preserving agent autonomy and data confidentiality. Benchmarks show that our KBA Orchestration significantly outperforms static description-driven methods in routing precision and overall system efficiency, making it suitable for large-scale systems that require higher accuracy than standard description-driven routing. 

**Abstract (ZH)**: 基于知识库意识的多-agent系统编排 

---
# GuessingGame: Measuring the Informativeness of Open-Ended Questions in Large Language Models 

**Title (ZH)**: 猜谜游戏：测量大型语言模型中开放式问题的信息量 

**Authors**: Dylan Hutson, Daniel Vennemeyer, Aneesh Deshmukh, Justin Zhan, Tianyu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19593)  

**Abstract**: We introduce GuessingGame, a protocol for evaluating large language models (LLMs) as strategic question-askers in open-ended, open-domain settings. A Guesser LLM identifies a hidden object by posing free-form questions to an Oracle without predefined choices or candidate lists. To measure question quality, we propose two information gain (IG) metrics: a Bayesian method that tracks belief updates over semantic concepts using LLM-scored relevance, and an entropy-based method that filters candidates via ConceptNet. Both metrics are model-agnostic and support post hoc analysis. Across 858 games with multiple models and prompting strategies, higher IG strongly predicts efficiency: a one-standard-deviation IG increase reduces expected game length by 43\%. Prompting constraints guided by IG, such as enforcing question diversity, enable weaker models to significantly improve performance. These results show that question-asking in LLMs is both measurable and improvable, and crucial for interactive reasoning. 

**Abstract (ZH)**: 猜谜游戏：评价大型语言模型在开放领域作为策略型提问者的协议 

---
# Frame-Stacked Local Transformers For Efficient Multi-Codebook Speech Generation 

**Title (ZH)**: 帧堆叠局部变压器用于高效的多码本语音生成 

**Authors**: Roy Fejgin, Paarth Neekhara, Xuesong Yang, Edresson Casanova, Ryan Langman Jaehyeon Kim, Subhankar Ghosh, Shehzeen Hussain, Jason Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19592)  

**Abstract**: Speech generation models based on large language models (LLMs) typically operate on discrete acoustic codes, which differ fundamentally from text tokens due to their multicodebook structure. At each timestep, models must predict N codebook entries jointly, introducing dependencies that challenge simple parallel prediction approaches. Parallel prediction assumes independence among codebooks, yielding efficient decoding but often at the cost of reduced fidelity. To address this, hierarchical strategies employ a local transformer (LT) to refine predictions and capture intra-timestep dependencies. In this work, we systematically investigate two LT architectures: an autoregressive transformer that generates codebooks sequentially, and a MaskGIT-based transformer that performs iterative masked prediction. Both designs further enable frame stacking, where the primary transformer predicts multiple frames jointly, and the LT decodes their codebooks, offering improvements in speed without compromising perceptual quality. Through extensive analysis, we characterize the tradeoffs between parallel and iterative sampling strategies across different throughput and quality regimes. Finally, we propose practical guidelines for selecting decoding strategies based on deployment priorities such as computational efficiency and synthesis fidelity. 

**Abstract (ZH)**: 基于大型语言模型的语音生成模型通常操作于离散声学编码上，这种编码由于其多码本结构而与文本令牌大不相同。在每个时间步，模型必须联合预测N个码本条目，这引入了依赖性，挑战了简单的并行预测方法。并行预测假设码本之间的独立性，从而实现高效的解码，但往往以降低保真度为代价。为解决这一问题，分层策略采用局部变压器（LT）来 refinement预测并捕捉跨时间步的依赖性。在本文中，我们系统地探讨了两种LT架构：一种自回归变压器按顺序生成码本，以及一种基于MaskGIT的变压器进行迭代掩码预测。这两种设计还进一步实现了帧堆叠，其中主变压器联合预测多个帧，而LT解码其码本，从而在不牺牲感知质量的情况下提升速度。通过广泛分析，我们刻画了不同吞吐量和质量范围内并行和迭代采样策略之间的权衡。最后，我们提出了基于部署优先级（如计算效率和合成保真度）选择解码策略的实用指南。 

---
# Reverse Engineering User Stories from Code using Large Language Models 

**Title (ZH)**: 使用大型语言模型从代码逆向工程用户故事 

**Authors**: Mohamed Ouf, Haoyu Li, Michael Zhang, Mariam Guizani  

**Link**: [PDF](https://arxiv.org/pdf/2509.19587)  

**Abstract**: User stories are essential in agile development, yet often missing or outdated in legacy and poorly documented systems. We investigate whether large language models (LLMs) can automatically recover user stories directly from source code and how prompt design impacts output quality. Using 1,750 annotated C++ snippets of varying complexity, we evaluate five state-of-the-art LLMs across six prompting strategies. Results show that all models achieve, on average, an F1 score of 0.8 for code up to 200 NLOC. Our findings show that a single illustrative example enables the smallest model (8B) to match the performance of a much larger 70B model. In contrast, structured reasoning via Chain-of-Thought offers only marginal gains, primarily for larger models. 

**Abstract (ZH)**: 基于大型语言模型从源代码自动恢复用户故事的研究：提示设计的影响 

---
# A Foundation Chemical Language Model for Comprehensive Fragment-Based Drug Discovery 

**Title (ZH)**: 全面片段基于药物发现的基础化学语言模型 

**Authors**: Alexander Ho, Sukyeong Lee, Francis T.F. Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2509.19586)  

**Abstract**: We introduce FragAtlas-62M, a specialized foundation model trained on the largest fragment dataset to date. Built on the complete ZINC-22 fragment subset comprising over 62 million molecules, it achieves unprecedented coverage of fragment chemical space. Our GPT-2 based model (42.7M parameters) generates 99.90% chemically valid fragments. Validation across 12 descriptors and three fingerprint methods shows generated fragments closely match the training distribution (all effect sizes < 0.4). The model retains 53.6% of known ZINC fragments while producing 22% novel structures with practical relevance. We release FragAtlas-62M with training code, preprocessed data, documentation, and model weights to accelerate adoption. 

**Abstract (ZH)**: 我们介绍FragAtlas-62M，这是一种专门训练于迄今最大片段数据集的基础模型。基于完整的ZINC-22片段子集（包含超过62百万吨子分子），该模型实现了前所未有的片段化学空间覆盖范围。基于GPT-2的模型（42.7M参数）生成了99.90%的化学有效的片段。在12个描述符和三种指纹方法的验证中，生成的片段与训练分布高度一致（所有效应大小<0.4）。该模型保留了53.6%已知的ZINC片段，同时生成了具有实际意义的22%的新结构。我们发布了FragAtlas-62M的训练代码、预处理数据、文档和模型权重，以加速其应用。 

---
# Learning Dynamics of Deep Learning -- Force Analysis of Deep Neural Networks 

**Title (ZH)**: 深度学习的学习动力学——深度神经网络的力分析 

**Authors**: Yi Ren  

**Link**: [PDF](https://arxiv.org/pdf/2509.19554)  

**Abstract**: This thesis explores how deep learning models learn over time, using ideas inspired by force analysis. Specifically, we zoom in on the model's training procedure to see how one training example affects another during learning, like analyzing how forces move objects. We break this influence into two parts: how similar the two examples are, and how strong the updating force is. This framework helps us understand a wide range of the model's behaviors in different real systems. For example, it explains why certain examples have non-trivial learning paths, why (and why not) some LLM finetuning methods work, and why simpler, more structured patterns tend to be learned more easily. We apply this approach to various learning tasks and uncover new strategies for improving model training. While the method is still developing, it offers a new way to interpret models' behaviors systematically. 

**Abstract (ZH)**: 基于力分析思想探索深度学习模型随时间的学习机制 

---
# DAWM: Diffusion Action World Models for Offline Reinforcement Learning via Action-Inferred Transitions 

**Title (ZH)**: DAWM：基于动作推断过渡的离线强化学习扩散动作世界模型 

**Authors**: Zongyue Li, Xiao Han, Yusong Li, Niklas Strauss, Matthias Schubert  

**Link**: [PDF](https://arxiv.org/pdf/2509.19538)  

**Abstract**: Diffusion-based world models have demonstrated strong capabilities in synthesizing realistic long-horizon trajectories for offline reinforcement learning (RL). However, many existing methods do not directly generate actions alongside states and rewards, limiting their compatibility with standard value-based offline RL algorithms that rely on one-step temporal difference (TD) learning. While prior work has explored joint modeling of states, rewards, and actions to address this issue, such formulations often lead to increased training complexity and reduced performance in practice. We propose \textbf{DAWM}, a diffusion-based world model that generates future state-reward trajectories conditioned on the current state, action, and return-to-go, paired with an inverse dynamics model (IDM) for efficient action inference. This modular design produces complete synthetic transitions suitable for one-step TD-based offline RL, enabling effective and computationally efficient training. Empirically, we show that conservative offline RL algorithms such as TD3BC and IQL benefit significantly from training on these augmented trajectories, consistently outperforming prior diffusion-based baselines across multiple tasks in the D4RL benchmark. 

**Abstract (ZH)**: 基于扩散的世界模型在离线强化学习（RL）中展示了强大的能力，能够生成具有高度现实感的长期轨迹。然而，许多现有方法没有直接生成动作和状态、奖励，限制了它们与依赖一-step 时差（TD）学习的标准值基离线RL算法的兼容性。虽然先前的工作探索了联合建模状态、奖励和动作以解决这一问题，但这样的建模往往会导致训练复杂度增加并在实际中表现不佳。我们提出了一种基于扩散的世界模型 \textbf{DAWM}，该模型在给定当前状态、动作和未来回报的情况下生成未来的状态-奖励轨迹，并配有一个逆动力学模型（IDM）以实现高效的动作推断。这种模块化设计可以生成适用于一-step TD 基础离线 RL 的完整合成转换，从而实现高效且计算成本低的训练。实验上，我们展示了保守的离线 RL 算法（如TD3BC和IQL）极大地受益于使用这些增强的轨迹进行训练，在 D4RL 基准上的多个任务中，这些算法在所有情况下都优于先前的基于扩散的方法。 

---
# Semantic-Aware Fuzzing: An Empirical Framework for LLM-Guided, Reasoning-Driven Input Mutation 

**Title (ZH)**: 语义意识模糊测试：一种由大语言模型引导、基于推理的输入变异 empirical 研究框架 

**Authors**: Mengdi Lu, Steven Ding, Furkan Alaca, Philippe Charland  

**Link**: [PDF](https://arxiv.org/pdf/2509.19533)  

**Abstract**: Security vulnerabilities in Internet-of-Things devices, mobile platforms, and autonomous systems remain critical. Traditional mutation-based fuzzers -- while effectively explore code paths -- primarily perform byte- or bit-level edits without semantic reasoning. Coverage-guided tools such as AFL++ use dictionaries, grammars, and splicing heuristics to impose shallow structural constraints, leaving deeper protocol logic, inter-field dependencies, and domain-specific semantics unaddressed. Conversely, reasoning-capable large language models (LLMs) can leverage pretraining knowledge to understand input formats, respect complex constraints, and propose targeted mutations, much like an experienced reverse engineer or testing expert. However, lacking ground truth for "correct" mutation reasoning makes supervised fine-tuning impractical, motivating explorations of off-the-shelf LLMs via prompt-based few-shot learning. To bridge this gap, we present an open-source microservices framework that integrates reasoning LLMs with AFL++ on Google's FuzzBench, tackling asynchronous execution and divergent hardware demands (GPU- vs. CPU-intensive) of LLMs and fuzzers. We evaluate four research questions: (R1) How can reasoning LLMs be integrated into the fuzzing mutation loop? (R2) Do few-shot prompts yield higher-quality mutations than zero-shot? (R3) Can prompt engineering with off-the-shelf models improve fuzzing directly? and (R4) Which open-source reasoning LLMs perform best under prompt-only conditions? Experiments with Llama3.3, Deepseek-r1-Distill-Llama-70B, QwQ-32B, and Gemma3 highlight Deepseek as the most promising. Mutation effectiveness depends more on prompt complexity and model choice than shot count. Response latency and throughput bottlenecks remain key obstacles, offering directions for future work. 

**Abstract (ZH)**: 基于推理的大规模语言模型在IoT设备、移动平台和自主系统中的 fuzzing 中的集成与优化 

---
# A Longitudinal Randomized Control Study of Companion Chatbot Use: Anthropomorphism and Its Mediating Role on Social Impacts 

**Title (ZH)**: 同伴聊天机器人使用的时间序列随机对照研究：拟人化及其在社会影响中的调节作用 

**Authors**: Rose E. Guingrich, Michael S. A. Graziano  

**Link**: [PDF](https://arxiv.org/pdf/2509.19515)  

**Abstract**: Relationships with social artificial intelligence (AI) agents are on the rise. People report forming friendships, mentorships, and romantic partnerships with chatbots such as Replika, a type of social AI agent that is designed specifically for companionship. Concerns that companion chatbot relationships may harm or replace human ones have been raised, but whether and how these social consequences occur remains unclear. Prior research suggests that people's states of social need and their anthropomorphism of the AI agent may play a role in how human-AI interaction impacts human-human interaction. In this longitudinal study (N = 183), participants were randomly assigned to converse with a companion chatbot over text or to play text-based word games for 10 minutes a day for 21 consecutive days. During these 21 days, participants also completed four surveys and two audio-recorded interviews. We found that people's social health and relationships were not significantly impacted by interacting with a companion chatbot across 21 days compared to the control group. However, people who had a higher desire to socially connect anthropomorphized the chatbot more. Those who anthropomorphized the chatbot more indicated that the human-chatbot interaction had greater impacts on their social interactions and relationships with family and friends. A mediation analysis suggested that the impact of human-AI interaction on human-human social outcomes was mediated by the extent to which people anthropomorphized the AI agent, which itself was related to the desire to socially connect. 

**Abstract (ZH)**: 伴随社会人工智能代理的关系正在兴起：以Replika为代表的伴侣聊天机器人对人类关系的Impact及其介导机制研究 

---
# The Heterogeneous Multi-Agent Challenge 

**Title (ZH)**: 异质多智能体挑战 

**Authors**: Charles Dansereau, Junior-Samuel Lopez-Yepez, Karthik Soma, Antoine Fagette  

**Link**: [PDF](https://arxiv.org/pdf/2509.19512)  

**Abstract**: Multi-Agent Reinforcement Learning (MARL) is a growing research area which gained significant traction in recent years, extending Deep RL applications to a much wider range of problems. A particularly challenging class of problems in this domain is Heterogeneous Multi-Agent Reinforcement Learning (HeMARL), where agents with different sensors, resources, or capabilities must cooperate based on local information. The large number of real-world situations involving heterogeneous agents makes it an attractive research area, yet underexplored, as most MARL research focuses on homogeneous agents (e.g., a swarm of identical robots). In MARL and single-agent RL, standardized environments such as ALE and SMAC have allowed to establish recognized benchmarks to measure progress. However, there is a clear lack of such standardized testbed for cooperative HeMARL. As a result, new research in this field often uses simple environments, where most algorithms perform near optimally, or uses weakly heterogeneous MARL environments. 

**Abstract (ZH)**: 多代理强化学习（MARL）是近年来迅速发展的研究领域，将深度强化学习（Deep RL）的应用扩展到更广泛的问题。此领域中一个尤其具有挑战性的问题类别是异构多代理强化学习（HeMARL），其中具有不同传感器、资源或能力的代理基于局部信息进行合作。由于涉及异构代理的实际应用场景众多，这使得它成为一个很有吸引力但尚未深入探索的研究领域，因为大多数MARL研究主要集中在同质代理（例如一群彼此相同机器人）上。在MARL和单代理强化学习中，标准化环境如ALE和SMAC使得建立公认的基准以衡量进展成为可能。然而，对于合作型HeMARL，缺乏这样的标准化测试平台。因此，该领域的研究成果往往使用简单的环境，大多数算法在这种环境中能接近最优表现，或者使用弱异质的MARL环境。 

---
# AIRwaves at CheckThat! 2025: Retrieving Scientific Sources for Implicit Claims on Social Media with Dual Encoders and Neural Re-Ranking 

**Title (ZH)**: AIRwaves在CheckThat! 2025: 使用双编码器和神经重排序检索社交媒体中的隐含声明科学来源 

**Authors**: Cem Ashbaugh, Leon Baumgärtner, Tim Gress, Nikita Sidorov, Daniel Werner  

**Link**: [PDF](https://arxiv.org/pdf/2509.19509)  

**Abstract**: Linking implicit scientific claims made on social media to their original publications is crucial for evidence-based fact-checking and scholarly discourse, yet it is hindered by lexical sparsity, very short queries, and domain-specific language. Team AIRwaves ranked second in Subtask 4b of the CLEF-2025 CheckThat! Lab with an evidence-retrieval approach that markedly outperforms the competition baseline. The optimized sparse-retrieval baseline(BM25) achieves MRR@5 = 0.5025 on the gold label blind test set. To surpass this baseline, a two-stage retrieval pipeline is introduced: (i) a first stage that uses a dual encoder based on E5-large, fine-tuned using in-batch and mined hard negatives and enhanced through chunked tokenization and rich document metadata; and (ii) a neural re-ranking stage using a SciBERT cross-encoder. Replacing purely lexical matching with neural representations lifts performance to MRR@5 = 0.6174, and the complete pipeline further improves to MRR@5 = 0.6828. The findings demonstrate that coupling dense retrieval with neural re-rankers delivers a powerful and efficient solution for tweet-to-study matching and provides a practical blueprint for future evidence-retrieval pipelines. 

**Abstract (ZH)**: 将社交媒体上隐含的科学断言与其原始出版物链接对于基于证据的事实核查和学术讨论至关重要，但这一过程受到词汇稀疏性、非常短的查询和领域特定语言的阻碍。Team AIRwaves 在 CLEF-2025 CheckThat! Lab 的 Subtask 4b 中凭借证据检索方法获得第二名，该方法显著优于竞争 baseline。优化的稀疏检索 baseline（BM25）在黄金标签盲测试集上达到 MRR@5 = 0.5025。为了超越这一 baseline，引入了一种两阶段检索管道：（i）第一阶段使用基于 E5-large 的双编码器，采用批次内和挖掘的难以负样本进行微调，并通过分块令牌化和丰富的文档元数据进行增强；以及（ii）神经再排序阶段使用 SciBERT 混合编码器。用神经表示替换纯词典匹配将性能提升到 MRR@5 = 0.6174，完整管道进一步提高到 MRR@5 = 0.6828。研究结果表明，结合密集检索和神经再排序器为推文与研究匹配提供了强大而高效的解决方案，并为未来证据检索管道提供了实用的蓝图。 

---
# Generative AI as a catalyst for democratic Innovation: Enhancing citizen engagement in participatory budgeting 

**Title (ZH)**: Generative AI作为民主创新的催化剂：增强公民参与式预算中的参与程度 

**Authors**: Italo Alberto do Nascimento Sousa, Jorge Machado, Jose Carlos Vaz  

**Link**: [PDF](https://arxiv.org/pdf/2509.19497)  

**Abstract**: This research examines the role of Generative Artificial Intelligence (AI) in enhancing citizen engagement in participatory budgeting. In response to challenges like declining civic participation and increased societal polarization, the study explores how online political participation can strengthen democracy and promote social equity. By integrating Generative AI into public consultation platforms, the research aims to improve citizen proposal formulation and foster effective dialogue between citizens and government. It assesses the capacities governments need to implement AI-enhanced participatory tools, considering technological dependencies and vulnerabilities. Analyzing technological structures, actors, interests, and strategies, the study contributes to understanding how technological advancements can reshape participatory institutions to better facilitate citizen involvement. Ultimately, the research highlights how Generative AI can transform participatory institutions, promoting inclusive, democratic engagement and empowering citizens. 

**Abstract (ZH)**: 这项研究探讨了生成型人工智能（AI）在增强公民在参与式预算中参与作用中的角色。面对如公民参与下降和社会极化加剧等挑战，研究探索了在线政治参与如何强化民主并促进社会公平。通过将生成型AI整合到公众咨询平台中，研究旨在改进公民提案的形成并促进公民与政府之间的有效对话。研究评估了政府实施增强型参与工具所需的能力，考虑了技术依赖性和脆弱性。通过对技术和结构、行动者、利益和策略的分析，研究为理解技术进步如何重塑参与机构以更好地促进公民参与提供了见解。最终，研究突出了生成型AI如何转变参与机构，促进包容性的民主参与并赋能公民。 

---
# ArtiFree: Detecting and Reducing Generative Artifacts in Diffusion-based Speech Enhancement 

**Title (ZH)**: ArtiFree：检测与减少基于扩散的语音增强中的生成性-artifacts 

**Authors**: Bhawana Chhaglani, Yang Gao, Julius Richter, Xilin Li, Syavosh Zadissa, Tarun Pruthi, Andrew Lovitt  

**Link**: [PDF](https://arxiv.org/pdf/2509.19495)  

**Abstract**: Diffusion-based speech enhancement (SE) achieves natural-sounding speech and strong generalization, yet suffers from key limitations like generative artifacts and high inference latency. In this work, we systematically study artifact prediction and reduction in diffusion-based SE. We show that variance in speech embeddings can be used to predict phonetic errors during inference. Building on these findings, we propose an ensemble inference method guided by semantic consistency across multiple diffusion runs. This technique reduces WER by 15% in low-SNR conditions, effectively improving phonetic accuracy and semantic plausibility. Finally, we analyze the effect of the number of diffusion steps, showing that adaptive diffusion steps balance artifact suppression and latency. Our findings highlight semantic priors as a powerful tool to guide generative SE toward artifact-free outputs. 

**Abstract (ZH)**: 基于扩散的语音增强中的噪声与延迟权衡研究 

---
# Identifying and Addressing User-level Security Concerns in Smart Homes Using "Smaller" LLMs 

**Title (ZH)**: 使用“较小”的语言模型识别和应对智能家居中的用户级安全顾虑 

**Authors**: Hafijul Hoque Chowdhury, Riad Ahmed Anonto, Sourov Jajodia, Suryadipta Majumdar, Md. Shohrab Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2509.19485)  

**Abstract**: With the rapid growth of smart home IoT devices, users are increasingly exposed to various security risks, as evident from recent studies. While seeking answers to know more on those security concerns, users are mostly left with their own discretion while going through various sources, such as online blogs and technical manuals, which may render higher complexity to regular users trying to extract the necessary information. This requirement does not go along with the common mindsets of smart home users and hence threatens the security of smart homes furthermore. In this paper, we aim to identify and address the major user-level security concerns in smart homes. Specifically, we develop a novel dataset of Q&A from public forums, capturing practical security challenges faced by smart home users. We extract major security concerns in smart homes from our dataset by leveraging the Latent Dirichlet Allocation (LDA). We fine-tune relatively "smaller" transformer models, such as T5 and Flan-T5, on this dataset to build a QA system tailored for smart home security. Unlike larger models like GPT and Gemini, which are powerful but often resource hungry and require data sharing, smaller models are more feasible for deployment in resource-constrained or privacy-sensitive environments like smart homes. The dataset is manually curated and supplemented with synthetic data to explore its potential impact on model performance. This approach significantly improves the system's ability to deliver accurate and relevant answers, helping users address common security concerns with smart home IoT devices. Our experiments on real-world user concerns show that our work improves the performance of the base models. 

**Abstract (ZH)**: 随着智能家庭物联网设备的迅速增长，用户越来越多地面临各种安全风险，这在最近的研究中已有体现。在寻求了解这些安全问题的答案时，用户往往只能依靠自己的判断力，通过各种来源（如在线博客和技术手册）来获取信息，这可能使得普通用户提取所需信息变得更加复杂。这种要求与智能家庭用户的一般思维方式不相符，从而进一步威胁到智能家庭的安全。本文旨在识别和解决智能家庭用户层面的主要安全问题。具体而言，我们开发了一个来自公开论坛的新型问答数据集，捕捉了智能家庭用户面临的实际安全挑战。我们通过利用潜在狄利克雷分配（LDA）技术从数据集中提取主要的安全关切点。我们针对这个数据集微调相对“较小”的变压器模型（如T5和Flan-T5），构建了一个针对智能家庭安全的问答系统。与其他大型模型（如GPT和Gemini）相比，小型模型在资源受限或隐私敏感的环境中更具可行性，不会因为数据共享而消耗大量资源。该数据集通过人工筛选并补充合成数据来探索其对模型性能的影响。这种方法大幅提高了系统的回答准确性和相关性，帮助用户解决与智能家庭物联网设备相关的常见安全问题。我们的实验证实在现实用户关切问题上证明了该工作的性能改进。 

---
# A Realistic Evaluation of Cross-Frequency Transfer Learning and Foundation Forecasting Models 

**Title (ZH)**: 跨频域迁移学习和基础预测模型的实际评价 

**Authors**: Kin G. Olivares, Malcolm Wolff, Tatiana Konstantinova, Shankar Ramasubramanian, Andrew Gordon Wilson, Andres Potapczynski, Willa Potosnak, Mengfei Cao, Boris Oreshkin, Dmitry Efimov  

**Link**: [PDF](https://arxiv.org/pdf/2509.19465)  

**Abstract**: Cross-frequency transfer learning (CFTL) has emerged as a popular framework for curating large-scale time series datasets to pre-train foundation forecasting models (FFMs). Although CFTL has shown promise, current benchmarking practices fall short of accurately assessing its performance. This shortcoming stems from many factors: an over-reliance on small-scale evaluation datasets; inadequate treatment of sample size when computing summary statistics; reporting of suboptimal statistical models; and failing to account for non-negligible risks of overlap between pre-training and test datasets. To address these limitations, we introduce a unified reimplementation of widely-adopted neural forecasting networks, adapting them for the CFTL setup; we pre-train only on proprietary and synthetic data, being careful to prevent test leakage; and we evaluate on 15 large, diverse public forecast competition datasets. Our empirical analysis reveals that statistical models' accuracy is frequently underreported. Notably, we confirm that statistical models and their ensembles consistently outperform existing FFMs by more than 8.2% in sCRPS, and by more than 20% MASE, across datasets. However, we also find that synthetic dataset pre-training does improve the accuracy of a FFM by 7% percent. 

**Abstract (ZH)**: 跨频率迁移学习（CFTL）已成为为预训练基础预测模型（FFMs）编撰大规模时间序列数据集的一种流行框架。 

---
# Self-evolved Imitation Learning in Simulated World 

**Title (ZH)**: 自我演化的模仿学习在模拟世界中 

**Authors**: Yifan Ye, Jun Cen, Jing Chen, Zhihe Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.19460)  

**Abstract**: Imitation learning has been a trend recently, yet training a generalist agent across multiple tasks still requires large-scale expert demonstrations, which are costly and labor-intensive to collect. To address the challenge of limited supervision, we propose Self-Evolved Imitation Learning (SEIL), a framework that progressively improves a few-shot model through simulator interactions. The model first attempts tasksin the simulator, from which successful trajectories are collected as new demonstrations for iterative refinement. To enhance the diversity of these demonstrations, SEIL employs dual-level augmentation: (i) Model-level, using an Exponential Moving Average (EMA) model to collaborate with the primary model, and (ii) Environment-level, introducing slight variations in initial object positions. We further introduce a lightweight selector that filters complementary and informative trajectories from the generated pool to ensure demonstration quality. These curated samples enable the model to achieve competitive performance with far fewer training examples. Extensive experiments on the LIBERO benchmark show that SEIL achieves a new state-of-the-art performance in few-shot imitation learning scenarios. Code is available at this https URL. 

**Abstract (ZH)**: 自我演化的imitation学习（SEIL）：通过模拟器交互渐进优化Few-shot模型 

---
# ROPA: Synthetic Robot Pose Generation for RGB-D Bimanual Data Augmentation 

**Title (ZH)**: ROPA：RGB-D 双手数据增强中的合成机器人姿态生成 

**Authors**: Jason Chen, I-Chun Arthur Liu, Gaurav Sukhatme, Daniel Seita  

**Link**: [PDF](https://arxiv.org/pdf/2509.19454)  

**Abstract**: Training robust bimanual manipulation policies via imitation learning requires demonstration data with broad coverage over robot poses, contacts, and scene contexts. However, collecting diverse and precise real-world demonstrations is costly and time-consuming, which hinders scalability. Prior works have addressed this with data augmentation, typically for either eye-in-hand (wrist camera) setups with RGB inputs or for generating novel images without paired actions, leaving augmentation for eye-to-hand (third-person) RGB-D training with new action labels less explored. In this paper, we propose Synthetic Robot Pose Generation for RGB-D Bimanual Data Augmentation (ROPA), an offline imitation learning data augmentation method that fine-tunes Stable Diffusion to synthesize third-person RGB and RGB-D observations of novel robot poses. Our approach simultaneously generates corresponding joint-space action labels while employing constrained optimization to enforce physical consistency through appropriate gripper-to-object contact constraints in bimanual scenarios. We evaluate our method on 5 simulated and 3 real-world tasks. Our results across 2625 simulation trials and 300 real-world trials demonstrate that ROPA outperforms baselines and ablations, showing its potential for scalable RGB and RGB-D data augmentation in eye-to-hand bimanual manipulation. Our project website is available at: this https URL. 

**Abstract (ZH)**: 通过模仿学习训练鲁棒的双臂 manipulation 策略需要广泛覆盖机器人姿态、接触点和场景上下文的演示数据。然而，收集多样且精确的现实世界演示数据成本高且耗时，这限制了其可扩展性。以往工作通过数据增强解决这一问题，通常针对带有 RGB 输入的眼手（腕部相机）设置或生成没有配对动作的新图像，而针对眼至手（第三人称）RGB-D 训练数据增强结合新动作标签的研究较少。本文提出了一种名为 Synthetic Robot Pose Generation for RGB-D Bimanual Data Augmentation (ROPA) 的离线模仿学习数据增强方法，通过微调 Stable Diffusion 合成第三人称 RGB 和 RGB-D 观测值的新机器人姿态。我们的方法同时生成相应的关节空间动作标签，并通过适当的双臂场景中的夹具-物体接触约束来实现物理一致性约束，以确保一致性。我们在 5 个仿真任务和 3 个现实世界任务上评估了该方法。我们的结果表明，ROPA 在 2625 次仿真试验和 300 次现实世界试验中优于基线和消融方法，展示了其在眼至手双臂 manipulation 中实现可扩展的 RGB 和 RGB-D 数据增强的潜力。项目网站: https://github.com/yourusername/ROPA。 

---
# Probabilistic Runtime Verification, Evaluation and Risk Assessment of Visual Deep Learning Systems 

**Title (ZH)**: 视觉深度学习系统概率运行时验证、评估与风险评估 

**Authors**: Birk Torpmann-Hagen, Pål Halvorsen, Michael A. Riegler, Dag Johansen  

**Link**: [PDF](https://arxiv.org/pdf/2509.19419)  

**Abstract**: Despite achieving excellent performance on benchmarks, deep neural networks often underperform in real-world deployment due to sensitivity to minor, often imperceptible shifts in input data, known as distributional shifts. These shifts are common in practical scenarios but are rarely accounted for during evaluation, leading to inflated performance metrics. To address this gap, we propose a novel methodology for the verification, evaluation, and risk assessment of deep learning systems. Our approach explicitly models the incidence of distributional shifts at runtime by estimating their probability from outputs of out-of-distribution detectors. We combine these estimates with conditional probabilities of network correctness, structuring them in a binary tree. By traversing this tree, we can compute credible and precise estimates of network accuracy. We assess our approach on five different datasets, with which we simulate deployment conditions characterized by differing frequencies of distributional shift. Our approach consistently outperforms conventional evaluation, with accuracy estimation errors typically ranging between 0.01 and 0.1. We further showcase the potential of our approach on a medical segmentation benchmark, wherein we apply our methods towards risk assessment by associating costs with tree nodes, informing cost-benefit analyses and value-judgments. Ultimately, our approach offers a robust framework for improving the reliability and trustworthiness of deep learning systems, particularly in safety-critical applications, by providing more accurate performance estimates and actionable risk assessments. 

**Abstract (ZH)**: 尽管深度神经网络在基准测试中表现出色，但在实际部署中往往由于对输入数据微小、通常难以察觉的变化（称为分布移位）的敏感性而表现不佳。这些移位在实际场景中很常见，但在评估过程中却很少考虑到，导致夸大了性能指标。为解决这一问题，我们提出了一种新的方法论，用于深度学习系统的验证、评估和风险评估。我们的方法在运行时明确模型分布移位的发生情况，通过分布外检测器的输出估计其概率。我们将这些估计值与网络正确性的条件概率相结合，结构化为二叉树。通过遍历该树，我们可以计算出网络准确性的有说服力且精确的估计。我们在五个不同的数据集上评估了我们的方法，在这些数据集上模拟了不同分布移位频率的部署条件。我们的方法在准确性估计误差方面始终优于传统评估方法，误差范围通常在0.01到0.1之间。我们进一步展示了该方法在医学分割基准上的潜在应用，其中我们通过将成本与树节点关联，将其应用到风险评估中，以支持成本效益分析和价值判断。最终，我们的方法提供了一个 robust 的框架，通过提供更准确的性能估计和可行的风险评估来提高深度学习系统的可靠性和可信度，特别是在安全关键应用中。 

---
# EngravingGNN: A Hybrid Graph Neural Network for End-to-End Piano Score Engraving 

**Title (ZH)**: EngravingGNN：一种端到端钢琴乐谱 engraving 的混合图神经网络 

**Authors**: Emmanouil Karystinaios, Francesco Foscarin, Gerhard Widmer  

**Link**: [PDF](https://arxiv.org/pdf/2509.19412)  

**Abstract**: This paper focuses on automatic music engraving, i.e., the creation of a humanly-readable musical score from musical content. This step is fundamental for all applications that include a human player, but it remains a mostly unexplored topic in symbolic music processing. In this work, we formalize the problem as a collection of interdependent subtasks, and propose a unified graph neural network (GNN) framework that targets the case of piano music and quantized symbolic input. Our method employs a multi-task GNN to jointly predict voice connections, staff assignments, pitch spelling, key signature, stem direction, octave shifts, and clef signs. A dedicated postprocessing pipeline generates print-ready MusicXML/MEI outputs. Comprehensive evaluation on two diverse piano corpora (J-Pop and DCML Romantic) demonstrates that our unified model achieves good accuracy across all subtasks, compared to existing systems that only specialize in specific subtasks. These results indicate that a shared GNN encoder with lightweight task-specific decoders in a multi-task setting offers a scalable and effective solution for automatic music engraving. 

**Abstract (ZH)**: 本文专注于自动音乐雕刻，即从音乐内容生成人类可读的乐谱。这一步骤是所有包含人类演奏者的应用的基础，但在符号音乐处理中仍然是一个未被充分探索的主题。在本工作中，我们将问题正式化为一组相互依赖的子任务，并提出一个针对钢琴音乐和量化符号输入的统一图神经网络（GNN）框架。我们的方法使用一个多任务GNN联合预测声部连接、五线谱分配、音高记写、键签名、音符方向、八度移位和谱号标识。专用的后处理管道生成可打印的MusicXML/MEI输出。在两个不同的钢琴语料库（J-Pop和DCML浪漫主义）上的全面评估表明，我们的统一模型在所有子任务上的准确性都优于仅专注于特定子任务的现有系统。这些结果表明，在多任务设置中使用一个共享的GNN编码器和轻量级的任务特定解码器提供了一种可扩展且有效的自动音乐雕刻解决方案。 

---
# TimeMosaic: Temporal Heterogeneity Guided Time Series Forecasting via Adaptive Granularity Patch and Segment-wise Decoding 

**Title (ZH)**: 时空拼图：基于适应粒度补丁和段落级解码的时间序列预测技术引导的时间异质性 

**Authors**: Kuiye Ding, Fanda Fan, Chunyi Hou, Zheya Wang, Lei Wang, Zhengxin Yang, Jianfeng Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2509.19406)  

**Abstract**: Multivariate time series forecasting is essential in domains such as finance, transportation, climate, and energy. However, existing patch-based methods typically adopt fixed-length segmentation, overlooking the heterogeneity of local temporal dynamics and the decoding heterogeneity of forecasting. Such designs lose details in information-dense regions, introduce redundancy in stable segments, and fail to capture the distinct complexities of short-term and long-term horizons. We propose TimeMosaic, a forecasting framework that aims to address temporal heterogeneity. TimeMosaic employs adaptive patch embedding to dynamically adjust granularity according to local information density, balancing motif reuse with structural clarity while preserving temporal continuity. In addition, it introduces segment-wise decoding that treats each prediction horizon as a related subtask and adapts to horizon-specific difficulty and information requirements, rather than applying a single uniform decoder. Extensive evaluations on benchmark datasets demonstrate that TimeMosaic delivers consistent improvements over existing methods, and our model trained on the large-scale corpus with 321 billion observations achieves performance competitive with state-of-the-art TSFMs. 

**Abstract (ZH)**: 多变量时间序列 forecasting 在金融、交通、气候和能源等领域至关重要。然而，现有的基于块的方法通常采用固定长度的分割，忽略了局部时间动态的异质性以及 forecasting 的解码异质性。这种设计在信息密集区域丢失了细节，在稳定段引入了冗余，并未能捕捉短期和长期视角下的独特复杂性。我们提出 TimeMosaic，一种旨在解决时间异质性的 forecasting 框架。TimeMosaic 使用自适应块嵌入动态调整粒度，根据局部信息密度进行调整，平衡模式重用与结构清晰度，同时保持时间连续性。此外，它引入了块级解码，将每个预测视角视为相关的子任务，并适应视角特定的难度和信息需求，而非采用单一统一的解码器。在基准数据集上的广泛评估表明，TimeMosaic 在多种场景中提供了持续改进，并且在包含 3210 亿观察值的大型语料库上训练的模型达到了与最新时间序列 forecasting 模型相当的性能。 

---
# Improving Outdoor Multi-cell Fingerprinting-based Positioning via Mobile Data Augmentation 

**Title (ZH)**: 基于移动数据增强的室外多小区指纹定位改进方法 

**Authors**: Tony Chahoud, Lorenzo Mario Amorosa, Riccardo Marini, Luca De Nardis  

**Link**: [PDF](https://arxiv.org/pdf/2509.19405)  

**Abstract**: Accurate outdoor positioning in cellular networks is hindered by sparse, heterogeneous measurement collections and the high cost of exhaustive site surveys. This paper introduces a lightweight, modular mobile data augmentation framework designed to enhance multi-cell fingerprinting-based positioning using operator-collected minimization of drive test (MDT) records. The proposed approach decouples spatial and radio-feature synthesis: kernel density estimation (KDE) models the empirical spatial distribution to generate geographically coherent synthetic locations, while a k-nearest-neighbor (KNN)-based block produces augmented per-cell radio fingerprints. The architecture is intentionally training-free, interpretable, and suitable for distributed or on-premise operator deployments, supporting privacy-aware workflows. We both validate each augmentation module independently and assess its end-to-end impact on fingerprinting-based positioning using a real-world MDT dataset provided by an Italian mobile network operator across diverse urban and peri-urban scenarios. Results show that the proposed KDE-KNN augmentation consistently improves positioning performance, with the largest benefits in sparsely sampled or structurally complex regions; we also observe region-dependent saturation effects as augmentation increases. The framework offers a practical, low-complexity path to enhance operator positioning services using existing mobile data traces. 

**Abstract (ZH)**: 基于运营商采集最小化路测记录的轻量级模块化移动数据增强框架：提升蜂窝网络多小区指纹定位精度 

---
# Online Adaptation via Dual-Stage Alignment and Self-Supervision for Fast-Calibration Brain-Computer Interfaces 

**Title (ZH)**: 基于双重阶段对齐和自我监督的在线自适应快速校准脑机接口 

**Authors**: Sheng-Bin Duan, Jian-Long Hao, Tian-Yu Xiang, Xiao-Hu Zhou, Mei-Jiang Gui, Xiao-Liang Xie, Shi-Qi Liu, Zeng-Guang Hou  

**Link**: [PDF](https://arxiv.org/pdf/2509.19403)  

**Abstract**: Individual differences in brain activity hinder the online application of electroencephalogram (EEG)-based brain computer interface (BCI) systems. To overcome this limitation, this study proposes an online adaptation algorithm for unseen subjects via dual-stage alignment and self-supervision. The alignment process begins by applying Euclidean alignment in the EEG data space and then updates batch normalization statistics in the representation space. Moreover, a self-supervised loss is designed to update the decoder. The loss is computed by soft pseudo-labels derived from the decoder as a proxy for the unknown ground truth, and is calibrated by Shannon entropy to facilitate self-supervised training. Experiments across five public datasets and seven decoders show the proposed algorithm can be integrated seamlessly regardless of BCI paradigm and decoder architecture. In each iteration, the decoder is updated with a single online trial, which yields average accuracy gains of 4.9% on steady-state visual evoked potentials (SSVEP) and 3.6% on motor imagery. These results support fast-calibration operation and show that the proposed algorithm has great potential for BCI applications. 

**Abstract (ZH)**: 基于脑电图的脑机接口系统中未见个体的大脑活动差异 hindering 其在线应用：一种双阶段对齐与自监督的在线自适应算法 

---
# FedOC: Multi-Server FL with Overlapping Client Relays in Wireless Edge Networks 

**Title (ZH)**: 联邦OC：无线边缘网络中具有重叠客户端中继的多服务器联邦学习 

**Authors**: Yun Ji, Zeyu Chen, Xiaoxiong Zhong, Yanan Ma, Sheng Zhang, Yuguang Fang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19398)  

**Abstract**: Multi-server Federated Learning (FL) has emerged as a promising solution to mitigate communication bottlenecks of single-server FL. We focus on a typical multi-server FL architecture, where the regions covered by different edge servers (ESs) may overlap. A key observation of this architecture is that clients located in the overlapping areas can access edge models from multiple ESs. Building on this insight, we propose FedOC (Federated learning with Overlapping Clients), a novel framework designed to fully exploit the potential of these overlapping clients. In FedOC, overlapping clients could serve dual roles: (1) as Relay Overlapping Clients (ROCs), they forward edge models between neighboring ESs in real time to facilitate model sharing among different ESs; and (2) as Normal Overlapping Clients (NOCs), they dynamically select their initial model for local training based on the edge model delivery time, which enables indirect data fusion among different regions of ESs. The overall FedOC workflow proceeds as follows: in every round, each client trains local model based on the earliest received edge model and transmits to the respective ESs for model aggregation. Then each ES transmits the aggregated edge model to neighboring ESs through ROC relaying. Upon receiving the relayed models, each ES performs a second aggregation and subsequently broadcasts the updated model to covered clients. The existence of ROCs enables the model of each ES to be disseminated to the other ESs in a decentralized manner, which indirectly achieves intercell model and speeding up the training process, making it well-suited for latency-sensitive edge environments. Extensive experimental results show remarkable performance gains of our scheme compared to existing methods. 

**Abstract (ZH)**: 多服务器联邦学习（FL）已成为解决单服务器FL通信瓶颈的一种有前途的解决方案。我们专注于一种典型的多服务器FL架构，其中不同边缘服务器（ESs）覆盖区域可能重叠。这一架构的关键观察是，位于重叠区域的客户端可以从多个ESs访问边缘模型。基于这一洞察，我们提出了一种新的框架FedOC（具有重叠客户端的联邦学习），旨在充分利用这些重叠客户端的潜力。在FedOC中，重叠客户端可以承担双重角色：（1）作为 Relay Overlapping Clients（ROCs），它们实时转发相邻ESs之间的边缘模型以促进不同ESs之间的模型共享；（2）作为 Normal Overlapping Clients（NOCs），它们根据边缘模型传输时间动态选择其初始模型进行本地训练，从而实现不同ESs区域间间接数据融合。FedOC的整体工作流如下：每一轮中，每个客户端基于最早接收到的边缘模型训练本地模型，并将模型传输给相应的ESs进行模型聚合。然后每个ESs通过ROCs的转发将聚合后的边缘模型传输给相邻的ESs。当每个ESs接收到转发的模型后，它们执行第二次聚合，并随后将更新后的模型广播给覆盖区域内的客户端。ROCs的存在使得每个ESs的模型能够以去中心化的方式传播到其他ESs，从而间接实现小区间模型的融合，加速训练过程，使其非常适合延迟敏感的边缘环境。实验结果表明，与现有方法相比，我们的方案具有显著的性能提升。 

---
# Self-Alignment Learning to Improve Myocardial Infarction Detection from Single-Lead ECG 

**Title (ZH)**: 自我对齐学习以提高单导联ECG心肌梗死检测 

**Authors**: Jiarui Jin, Xiaocheng Fang, Haoyu Wang, Jun Li, Che Liu, Donglin Xie, Hongyan Li, Shenda Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.19397)  

**Abstract**: Myocardial infarction is a critical manifestation of coronary artery disease, yet detecting it from single-lead electrocardiogram (ECG) remains challenging due to limited spatial information. An intuitive idea is to convert single-lead into multiple-lead ECG for classification by pre-trained models, but generative methods optimized at the signal level in most cases leave a large latent space gap, ultimately degrading diagnostic performance. This naturally raises the question of whether latent space alignment could help. However, most prior ECG alignment methods focus on learning transformation invariance, which mismatches the goal of single-lead detection. To address this issue, we propose SelfMIS, a simple yet effective alignment learning framework to improve myocardial infarction detection from single-lead ECG. Discarding manual data augmentations, SelfMIS employs a self-cutting strategy to pair multiple-lead ECG with their corresponding single-lead segments and directly align them in the latent space. This design shifts the learning objective from pursuing transformation invariance to enriching the single-lead representation, explicitly driving the single-lead ECG encoder to learn a representation capable of inferring global cardiac context from the local signal. Experimentally, SelfMIS achieves superior performance over baseline models across nine myocardial infarction types while maintaining a simpler architecture and lower computational overhead, thereby substantiating the efficacy of direct latent space alignment. Our code and checkpoint will be publicly available after acceptance. 

**Abstract (ZH)**: 基于自我切割的单导联心电图隐空间对齐方法以改善 myocardial infarction 检测 

---
# OmniFed: A Modular Framework for Configurable Federated Learning from Edge to HPC 

**Title (ZH)**: 全地域联邦学习：从边缘到HPC的可配置模块化框架 

**Authors**: Sahil Tyagi, Andrei Cozma, Olivera Kotevska, Feiyi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19396)  

**Abstract**: Federated Learning (FL) is critical for edge and High Performance Computing (HPC) where data is not centralized and privacy is crucial. We present OmniFed, a modular framework designed around decoupling and clear separation of concerns for configuration, orchestration, communication, and training logic. Its architecture supports configuration-driven prototyping and code-level override-what-you-need customization. We also support different topologies, mixed communication protocols within a single deployment, and popular training algorithms. It also offers optional privacy mechanisms including Differential Privacy (DP), Homomorphic Encryption (HE), and Secure Aggregation (SA), as well as compression strategies. These capabilities are exposed through well-defined extension points, allowing users to customize topology and orchestration, learning logic, and privacy/compression plugins, all while preserving the integrity of the core system. We evaluate multiple models and algorithms to measure various performance metrics. By unifying topology configuration, mixed-protocol communication, and pluggable modules in one stack, OmniFed streamlines FL deployment across heterogeneous environments. Github repository is available at this https URL. 

**Abstract (ZH)**: 联邦学习（FL）对于边缘计算和高性能计算（HPC）至关重要，尤其是在数据未集中且隐私至关重要的情况下。我们提出了OmniFed，一个模块化框架，其设计理念是解耦和清晰分离配置、 orchestration、通信和训练逻辑。其架构支持配置驱动的原型设计和代码级按需定制。同时，它支持不同的拓扑结构、单一部署中的混合通信协议以及流行的训练算法。它还提供了包括差分隐私（DP）、同态加密（HE）和安全聚合（SA）在内的可选隐私机制，以及压缩策略。这些功能通过明确定义的扩展点暴露出来，允许用户定制拓扑结构和orchestration、学习逻辑以及隐私/压缩插件，同时保持核心系统的完整性。我们评估了多种模型和算法以衡量各种性能指标。通过在一个栈中统一拓扑配置、混合协议通信和可插拔模块，OmniFed 简化了跨异构环境的联邦学习部署。GitHub仓库可通过此链接访问。 

---
# TensLoRA: Tensor Alternatives for Low-Rank Adaptation 

**Title (ZH)**: TensLoRA: 张量视角下的低秩适应 

**Authors**: Axel Marmoret, Reda Bensaid, Jonathan Lys, Vincent Gripon, François Leduc-Primeau  

**Link**: [PDF](https://arxiv.org/pdf/2509.19391)  

**Abstract**: Low-Rank Adaptation (LoRA) is widely used to efficiently adapt Transformers by adding trainable low-rank matrices to attention projections. While effective, these matrices are considered independent for each attention projection (Query, Key, and Value) and each layer. Recent extensions have considered joint, tensor-based adaptations, but only in limited forms and without a systematic framework. We introduce TensLoRA, a unified framework that aggregates LoRA updates into higher-order tensors and models a broad family of tensor-based low-rank adaptations. Our formulation generalizes existing tensor-based methods and enables mode-specific compression rates, allowing parameter budgets to be tailored according to the modality and task. Experiments on vision and language benchmarks reveal that the tensor construction directly impacts performance, sometimes better than standard LoRA under similar parameter counts. 

**Abstract (ZH)**: TensLoRA: 一种统一的张量基低秩适应框架 

---
# Data-Driven Reconstruction of Significant Wave Heights from Sparse Observations 

**Title (ZH)**: 基于数据驱动的稀疏观测显著波高重构 

**Authors**: Hongyuan Shi, Yilin Zhai, Ping Dong, Zaijin You, Chao Zhan, Qing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19384)  

**Abstract**: Reconstructing high-resolution regional significant wave height fields from sparse and uneven buoy observations remains a core challenge for ocean monitoring and risk-aware operations. We introduce AUWave, a hybrid deep learning framework that fuses a station-wise sequence encoder (MLP) with a multi-scale U-Net enhanced by a bottleneck self-attention layer to recover 32$\times$32 regional SWH fields. A systematic Bayesian hyperparameter search with Optuna identifies the learning rate as the dominant driver of generalization, followed by the scheduler decay and the latent dimension. Using NDBC buoy observations and ERA5 reanalysis over the Hawaii region, AUWave attains a minimum validation loss of 0.043285 and a slightly right-skewed RMSE distribution. Spatial errors are lowest near observation sites and increase with distance, reflecting identifiability limits under sparse sampling. Sensitivity experiments show that AUWave consistently outperforms a representative baseline in data-richer configurations, while the baseline is only marginally competitive in the most underdetermined single-buoy cases. The architecture's multi-scale and attention components translate into accuracy gains when minimal but non-trivial spatial anchoring is available. Error maps and buoy ablations reveal key anchor stations whose removal disproportionately degrades performance, offering actionable guidance for network design. AUWave provides a scalable pathway for gap filling, high-resolution priors for data assimilation, and contingency reconstruction. 

**Abstract (ZH)**: 基于稀疏不均匀浮标观测的高分辨率区域显著波高场重建仍然是海洋监测和风险感知操作中的核心挑战。我们引入了AUWave，这是一种结合站级序列编码器（MLP）并与带有瓶颈自注意力层的多尺度U-Net相结合的混合深度学习框架，用于恢复32×32区域显著波高场（SWH）。通过使用Optuna进行系统性的贝叶斯超参数搜索，确定了学习率是泛化能力的主导因素，其次是调度器衰减和潜在维度。使用夏威夷地区的NDBC浮标观测和ERA5再分析数据，AUWave达到最小验证损失0.043285，并显示出轻微右偏的RMSE分布。空间误差在观测站点附近最低，随距离增加而增加，反映了在稀疏采样下可识别性的限制。敏感性实验表明，在数据更丰富的情况下，AUWave始终优于一个代表性基线，而在最欠定的单浮标案例中，基线仅略具竞争力。该架构的多尺度和注意力机制在仅提供少量但非琐碎的空间锚定情况下转化为准确性的提升。错误图和浮标消融实验揭示了关键锚定点，其移除会对性能产生不成比例的负面影响，为网络设计提供了可行的指导。AUWave提供了一种可扩展的路径，用于填补数据缺口、为数据同化提供高分辨率先验以及进行应急重建。 

---
# Learning from Observation: A Survey of Recent Advances 

**Title (ZH)**: 基于观察学习：近期进展综述 

**Authors**: Returaj Burnwal, Hriday Mehta, Nirav Pravinbhai Bhatt, Balaraman Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2509.19379)  

**Abstract**: Imitation Learning (IL) algorithms offer an efficient way to train an agent by mimicking an expert's behavior without requiring a reward function. IL algorithms often necessitate access to state and action information from expert demonstrations. Although expert actions can provide detailed guidance, requiring such action information may prove impractical for real-world applications where expert actions are difficult to obtain. To address this limitation, the concept of learning from observation (LfO) or state-only imitation learning (SOIL) has recently gained attention, wherein the imitator only has access to expert state visitation information. In this paper, we present a framework for LfO and use it to survey and classify existing LfO methods in terms of their trajectory construction, assumptions and algorithm's design choices. This survey also draws connections between several related fields like offline RL, model-based RL and hierarchical RL. Finally, we use our framework to identify open problems and suggest future research directions. 

**Abstract (ZH)**: 学习从观察（LfO）算法：一种无需专家行为信息的有效模仿学习框架 

---
# Solving Freshness in RAG: A Simple Recency Prior and the Limits of Heuristic Trend Detection 

**Title (ZH)**: 解决RAG中的新鲜度问题：一个简单的近期优先级和启发式趋势检测的局限性 

**Authors**: Matthew Grofsky  

**Link**: [PDF](https://arxiv.org/pdf/2509.19376)  

**Abstract**: We address temporal failures in RAG systems using two methods on cybersecurity data. A simple recency prior achieved an accuracy of 1.00 on freshness tasks. In contrast, a clustering heuristic for topic evolution failed (0.08 F1-score), showing trend detection requires methods beyond simple heuristics. 

**Abstract (ZH)**: 我们使用两种方法处理RAG系统中的时间失败问题，并应用于网络安全数据。简单的时效性先验在新鲜度任务中达到了0.99的准确率。相比之下，用于主题演变的聚类启发式方法失败了（F1分数为0.08），表明趋势检测需要超出简单启发式的方法。 

---
# Uncertainty Quantification of Large Language Models using Approximate Bayesian Computation 

**Title (ZH)**: 使用近似贝叶斯计算量化的大型语言模型不确定性量化 

**Authors**: Mridul Sharma, Adeetya Patel, Zaneta D' Souza, Samira Abbasgholizadeh Rahimi, Siva Reddy, Sreenath Madathil  

**Link**: [PDF](https://arxiv.org/pdf/2509.19375)  

**Abstract**: Despite their widespread applications, Large Language Models (LLMs) often struggle to express uncertainty, posing a challenge for reliable deployment in high stakes and safety critical domains like clinical diagnostics. Existing standard baseline methods such as model logits and elicited probabilities produce overconfident and poorly calibrated estimates. In this work, we propose Approximate Bayesian Computation (ABC), a likelihood-free Bayesian inference, based approach that treats LLMs as a stochastic simulator to infer posterior distributions over predictive probabilities. We evaluate our ABC approach on two clinically relevant benchmarks: a synthetic oral lesion diagnosis dataset and the publicly available GretelAI symptom-to-diagnosis dataset. Compared to standard baselines, our approach improves accuracy by up to 46.9\%, reduces Brier scores by 74.4\%, and enhances calibration as measured by Expected Calibration Error (ECE) and predictive entropy. 

**Abstract (ZH)**: 尽管大型语言模型在广泛应用，但在高风险和安全关键领域如临床诊断中的可靠部署上，它们往往难以表达不确定性，存在挑战。现有标准基准方法如模型logits和诱发概率产生过度自信且校准不佳的估计。在本工作中，我们提出了一种基于Likelihood-Free Bayesian Inference的方法，即近似贝叶斯计算（ABC），将大型语言模型视为随机模拟器以推断预测概率的后验分布。我们在两个临床相关基准上评估了我们的ABC方法：合成的口腔病损诊断数据集和公开的GretelAI症状到诊断数据集。与标准基准方法相比，我们的方法在准确性上提高了最多46.9%，降低了Brier分数74.4%，并通过预期校准误差（ECE）和预测熵提升了校准。 

---
# Representation-based Broad Hallucination Detectors Fail to Generalize Out of Distribution 

**Title (ZH)**: 基于表示的广泛幻觉检测器无法泛化到分布外数据 

**Authors**: Zuzanna Dubanowska, Maciej Żelaszczyk, Michał Brzozowski, Paolo Mandica, Michał Karpowicz  

**Link**: [PDF](https://arxiv.org/pdf/2509.19372)  

**Abstract**: We critically assess the efficacy of the current SOTA in hallucination detection and find that its performance on the RAGTruth dataset is largely driven by a spurious correlation with data. Controlling for this effect, state-of-the-art performs no better than supervised linear probes, while requiring extensive hyperparameter tuning across datasets. Out-of-distribution generalization is currently out of reach, with all of the analyzed methods performing close to random. We propose a set of guidelines for hallucination detection and its evaluation. 

**Abstract (ZH)**: 我们批判性地评估了当前领先研究成果在幻觉检测方面的有效性，并发现其在RAGTruth数据集上的 performance 大部分是由与数据之间的虚假相关性驱动的。在控制这一效应后，最先进的方法的表现并不优于监督线性探针，同时还需要在不同数据集上进行广泛的超参数调整。目前，异常分布外的泛化能力仍然无法实现，所有分析的方法的表现都接近随机。我们提出了一套幻觉检测及其评估的指导原则。 

---
# How to inject knowledge efficiently? Knowledge Infusion Scaling Law for Pre-training Large Language Models 

**Title (ZH)**: 如何高效注入知识？大规模语言模型预训练的知识注入标度律 

**Authors**: Kangtao Lv, Haibin Chen, Yujin Yuan, Langming Liu, Shilei Liu, Yongwei Wang, Wenbo Su, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.19371)  

**Abstract**: Large language models (LLMs) have attracted significant attention due to their impressive general capabilities across diverse downstream tasks. However, without domain-specific optimization, they often underperform on specialized knowledge benchmarks and even produce hallucination. Recent studies show that strategically infusing domain knowledge during pretraining can substantially improve downstream performance. A critical challenge lies in balancing this infusion trade-off: injecting too little domain-specific data yields insufficient specialization, whereas excessive infusion triggers catastrophic forgetting of previously acquired knowledge. In this work, we focus on the phenomenon of memory collapse induced by over-infusion. Through systematic experiments, we make two key observations, i.e. 1) Critical collapse point: each model exhibits a threshold beyond which its knowledge retention capabilities sharply degrade. 2) Scale correlation: these collapse points scale consistently with the model's size. Building on these insights, we propose a knowledge infusion scaling law that predicts the optimal amount of domain knowledge to inject into large LLMs by analyzing their smaller counterparts. Extensive experiments across different model sizes and pertaining token budgets validate both the effectiveness and generalizability of our scaling law. 

**Abstract (ZH)**: 大型语言模型（LLMs）因其在多种下游任务中的出色通用能力而吸引了大量关注。然而，如果没有针对特定领域的优化，它们往往在专门知识基准测试中表现不佳，甚至会产生幻觉。最近的研究表明，在预训练过程中战略性地注入领域知识可以大幅提高下游性能。一个关键挑战在于平衡这种注入的权衡：注入过少的领域特定数据会导致知识保留能力不足，而过度注入则会引发灾难性遗忘。在本文中，我们关注过度注入引发的记忆崩溃现象。通过系统的实验，我们发现在以下几个方面：1）关键崩溃点：每个模型都存在一个临界阈值，在这个阈值之上，其知识保留能力会急剧下降。2）规模相关性：这些崩溃点与模型规模呈现出一致的关联性。在此基础上，我们提出了一种知识注入规模定律，通过分析较小模型来预测应注入到大型LLMs中的领域知识最优量。广泛的实验验证了该定律的有效性和普适性。 

---
# Meow: End-to-End Outline Writing for Automatic Academic Survey 

**Title (ZH)**: Meow: 自动学术调研提纲生成 

**Authors**: Zhaoyu Ma, Yuan Shan, Jiahao Zhao, Nan Xu, Lei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19370)  

**Abstract**: As academic paper publication numbers grow exponentially, conducting in-depth surveys with LLMs automatically has become an inevitable trend. Outline writing, which aims to systematically organize related works, is critical for automated survey generation. Yet existing automatic survey methods treat outline writing as mere workflow steps in the overall pipeline. Such template-based workflows produce outlines that lack in-depth understanding of the survey topic and fine-grained styles. To address these limitations, we propose Meow, the first metadata-driven outline writing framework that produces organized and faithful outlines efficiently. Specifically, we first formulate outline writing as an end-to-end task that generates hierarchical structured outlines from paper metadata. We then curate a high-quality dataset of surveys from arXiv, bioRxiv, and medRxiv, and establish systematic evaluation metrics for outline quality assessment. Finally, we employ a two-stage training approach combining supervised fine-tuning and reinforcement learning. Our 8B reasoning model demonstrates strong performance with high structural fidelity and stylistic coherence. 

**Abstract (ZH)**: 随着学术论文发表数量呈指数增长，自动使用大规模语言模型进行深入调查已成为一种不可避免的趋势。面向元数据的提纲写作框架Meow：高效生成组织化和忠实提纲 

---
# SLM-Based Agentic AI with P-C-G: Optimized for Korean Tool Use 

**Title (ZH)**: 基于SLM的P-C-G代理型人工智能：针对韩工具使用优化 

**Authors**: Changhyun Jeon, Jinhee Park, Jungwoo Choi, Keonwoo Kim, Jisu Kim, Minji Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.19369)  

**Abstract**: We propose a small-scale language model (SLM) based agent architecture, Planner-Caller-Generator (P-C-G), optimized for Korean tool use. P-C-G separates planning, calling, and generation by role: the Planner produces an initial batch plan with limited on-demand replanning; the Caller returns a normalized call object after joint schema-value validation; and the Generator integrates tool outputs to produce the final answer. We apply a Korean-first value policy to reduce execution failures caused by frequent Korean-to-English code switching in Korean settings. Evaluation assumes Korean queries and Korean tool/parameter specifications; it covers single-chain, multi-chain, missing-parameters, and missing-functions scenarios, and is conducted via an LLM-as-a-Judge protocol averaged over five runs under a unified I/O interface. Results show that P-C-G delivers competitive tool-use accuracy and end-to-end quality while reducing tokens and maintaining acceptable latency, indicating that role-specialized SLMs are a cost-effective alternative for Korean tool-use agents. 

**Abstract (ZH)**: 基于规划者-调用者-生成者（P-C-G）架构的小规模语言模型代理优化应用于韩语工具使用场景 

---
# Pipeline Parallelism is All You Need for Optimized Early-Exit Based Self-Speculative Decoding 

**Title (ZH)**: 优化早期退出基于自我推测解码所需的管道并行ism即一切 

**Authors**: Ruanjun Li, Ziheng Liu, Yuanming Shi, Jiawei Shao, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19368)  

**Abstract**: Large language models (LLMs) deliver impressive generation quality, but incur very high inference cost because each output token is generated auto-regressively through all model layers. Early-exit based self-speculative decoding (EESD) has emerged to mitigate this cost. However, in practice, many approaches struggle to achieve the expected acceleration in such draft-then-verify paradigm even with a well-aligned early-exit head and selected exit position. Our analysis reveals that EESD only pays off when the vast majority of draft tokens are accepted by the LLM. Otherwise, the draft cost may overcome the acceleration gain and lead to a negative speedup. To mitigate this, we propose Pipeline-Parallel Self-Speculative Decoding (PPSD) that fully pipelines the draft and verification work so that no effort is wasted on failed predictions. It has two key innovations. We configure the model layers as a pipeline in which early-exit (draft) computations and remaining-layer (verification) computations overlap. We interleave drafting and verification per token. While the LLM is verifying the current token in its final layers, the early-exit path simultaneously drafts the next token. Such a verify-while-draft scheme keeps all units busy and validates tokens on-the-fly analogous to pipelining the speculation and verification stages. Empirical results confirm that PPSD achieves state-of-the-art acceleration in self-speculative LLM inference. On diverse benchmarks, PPSD achieves speedup ratios in the range of 2.01x~3.81x, which gains almost the optimal acceleration at the fixed acceptance rate and exit position, showcasing its advancement in providing efficient self-speculation. 

**Abstract (ZH)**: 基于管道并行的自我推测解码（PPSD）：实现高效自推测大规模语言模型推理 

---
# Unsupervised Outlier Detection in Audit Analytics: A Case Study Using USA Spending Data 

**Title (ZH)**: 审计分析中无监督异常检测：基于USA Spending数据的案例研究 

**Authors**: Buhe Li, Berkay Kaplan, Maksym Lazirko, Aleksandr Kogan  

**Link**: [PDF](https://arxiv.org/pdf/2509.19366)  

**Abstract**: This study investigates the effectiveness of unsupervised outlier detection methods in audit analytics, utilizing USA spending data from the U.S. Department of Health and Human Services (DHHS) as a case example. We employ and compare multiple outlier detection algorithms, including Histogram-based Outlier Score (HBOS), Robust Principal Component Analysis (PCA), Minimum Covariance Determinant (MCD), and K-Nearest Neighbors (KNN) to identify anomalies in federal spending patterns. The research addresses the growing need for efficient and accurate anomaly detection in large-scale governmental datasets, where traditional auditing methods may fall short. Our methodology involves data preparation, algorithm implementation, and performance evaluation using precision, recall, and F1 scores. Results indicate that a hybrid approach, combining multiple detection strategies, enhances the robustness and accuracy of outlier identification in complex financial data. This study contributes to the field of audit analytics by providing insights into the comparative effectiveness of various outlier detection models and demonstrating the potential of unsupervised learning techniques in improving audit quality and efficiency. The findings have implications for auditors, policymakers, and researchers seeking to leverage advanced analytics in governmental financial oversight and risk management. 

**Abstract (ZH)**: 本研究探讨了无监督异常检测方法在审计数据分析中的有效性，以美国卫生与人类服务部（DHHS）的美国支出数据为例进行分析。我们采用了多个异常检测算法，包括基于直方图的异常评分（HBOS）、鲁棒主成分分析（RPCA）、最小 covariance 确定（MCD）和 K-最近邻（KNN），以识别联邦支出模式中的异常。研究解决了在大规模政府数据集中高效准确地检测异常日益增长的需求，而传统的审计方法可能难以满足这一需求。我们的方法包括数据准备、算法实现以及使用精确度、召回率和F1分数进行性能评估。研究结果表明，结合多种检测策略的混合方法可以提高复杂金融数据中异常识别的稳健性和准确性。本研究通过提供各种异常检测模型的比较有效性见解，并展示了无监督学习技术在提高审计质量和效率方面的潜力，为审计分析领域做出了贡献。研究结果对审计人员、政策制定者和研究人员利用先进的数据分析技术进行政府财务监督和风险管理具有重要意义。 

---
# The Inadequacy of Offline LLM Evaluations: A Need to Account for Personalization in Model Behavior 

**Title (ZH)**: 线下大模型评估的不足：需考虑模型行为的个性化 

**Authors**: Angelina Wang, Daniel E. Ho, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2509.19364)  

**Abstract**: Standard offline evaluations for language models -- a series of independent, state-less inferences made by models -- fail to capture how language models actually behave in practice, where personalization fundamentally alters model behavior. For instance, identical benchmark questions to the same language model can produce markedly different responses when prompted to a state-less system, in one user's chat session, or in a different user's chat session. In this work, we provide empirical evidence showcasing this phenomenon by comparing offline evaluations to field evaluations conducted by having 800 real users of ChatGPT and Gemini pose benchmark and other provided questions to their chat interfaces. 

**Abstract (ZH)**: 标准离线评估对于语言模型——一系列独立的、无状态的模型推断——未能捕捉到个人化如何根本改变模型行为的情况，而这是语言模型在实践中表现出的行为。例如，给同一个语言模型提供相同的基准问题，当这些问题在无状态系统中、某用户的聊天会话中或不同用户的聊天会话中被触发时，可能会产生截然不同的回应。在这项工作中，我们通过让800名真实用户使用ChatGPT和Gemini提出基准问题和其他提供的问题来比较离线评估和现场评估，提供了实证证据展示这一现象。 

---
# Analyzing the Impact of Credit Card Fraud on Economic Fluctuations of American Households Using an Adaptive Neuro-Fuzzy Inference System 

**Title (ZH)**: 基于自适应神经模糊 inference 系统分析信用卡欺诈对美国家庭经济波动的影响 

**Authors**: Zhuqi Wang, Qinghe Zhang, Zhuopei Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.19363)  

**Abstract**: Credit card fraud is assuming growing proportions as a major threat to the financial position of American household, leading to unpredictable changes in household economic behavior. To solve this problem, in this paper, a new hybrid analysis method is presented by using the Enhanced ANFIS. The model proposes several advances of the conventional ANFIS framework and employs a multi-resolution wavelet decomposition module and a temporal attention mechanism. The model performs discrete wavelet transformations on historical transaction data and macroeconomic indicators to generate localized economic shock signals. The transformed features are then fed into a deep fuzzy rule library which is based on Takagi-Sugeno fuzzy rules with adaptive Gaussian membership functions. The model proposes a temporal attention encoder that adaptively assigns weights to multi-scale economic behavior patterns, increasing the effectiveness of relevance assessment in the fuzzy inference stage and enhancing the capture of long-term temporal dependencies and anomalies caused by fraudulent activities. The proposed method differs from classical ANFIS which has fixed input-output relations since it integrates fuzzy rule activation with the wavelet basis selection and the temporal correlation weights via a modular training procedure. Experimental results show that the RMSE was reduced by 17.8% compared with local neuro-fuzzy models and conventional LSTM models. 

**Abstract (ZH)**: 信用卡欺诈正日益成为美国家庭财务状况的主要威胁，导致家庭经济行为出现不可预测的变化。为了解决这一问题，本文提出了一种基于Enhanced ANFIS的新型混合分析方法。该模型在传统ANFIS框架的基础上进行了多项改进，并采用了多分辨率小波分解模块和时间注意力机制。模型对历史交易数据和宏观经济指标进行了离散小波变换，生成局部经济冲击信号。变换后的特征输入基于Takagi-Sugeno模糊规则且具有自适应高斯隶属函数的深度模糊规则库。模型提出了一种时间注意力编码器，能够自适应地为多尺度经济行为模式分配权重，增强了模糊推理阶段的相关性评估效果，并增强了对长期时间依赖性和欺诈活动引起的异常的捕捉能力。与经典的固定输入-输出关系的ANFIS相比，该方法通过模块化训练过程将模糊规则激活、小波基选择和时间相关权重相结合。实验结果表明，该方法的RMSE相比局部神经模糊模型和传统的LSTM模型降低了17.8%。 

---
# DeepACTIF: Efficient Feature Attribution via Activation Traces in Neural Sequence Models 

**Title (ZH)**: DeepACTIF: 通过神经序列模型激活轨迹进行高效特征归因 

**Authors**: Benedikt W. Hosp  

**Link**: [PDF](https://arxiv.org/pdf/2509.19362)  

**Abstract**: Feature attribution is essential for interpreting deep learning models, particularly in time-series domains such as healthcare, biometrics, and human-AI interaction. However, standard attribution methods, such as Integrated Gradients or SHAP, are computationally intensive and not well-suited for real-time applications. We present DeepACTIF, a lightweight and architecture-aware feature attribution method that leverages internal activations of sequence models to estimate feature importance efficiently. Focusing on LSTM-based networks, we introduce an inverse-weighted aggregation scheme that emphasises stability and magnitude of activations across time steps. Our evaluation across three biometric gaze datasets shows that DeepACTIF not only preserves predictive performance under severe feature reduction (top 10% of features) but also significantly outperforms established methods, including SHAP, IG, and DeepLIFT, in terms of both accuracy and statistical robustness. Using Wilcoxon signed-rank tests and effect size analysis, we demonstrate that DeepACTIF yields more informative feature rankings with significantly lower error across all top-k conditions (10 - 40%). Our experiments demonstrate that DeepACTIF not only reduces computation time and memory usage by orders of magnitude but also preserves model accuracy when using only top-ranked features. That makes DeepACTIF a viable solution for real-time interpretability on edge devices such as mobile XR headsets or embedded health monitors. 

**Abstract (ZH)**: DeepACTIF：一种针对序列模型的轻量化且架构感知的特征 attribution 方法及其在生物识别凝视数据集上的应用 

---
# Semantic Representation Attack against Aligned Large Language Models 

**Title (ZH)**: 面向对齐大语言模型的语义表示攻击 

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau  

**Link**: [PDF](https://arxiv.org/pdf/2509.19360)  

**Abstract**: Large Language Models (LLMs) increasingly employ alignment techniques to prevent harmful outputs. Despite these safeguards, attackers can circumvent them by crafting prompts that induce LLMs to generate harmful content.
Current methods typically target exact affirmative responses, such as ``Sure, here is...'', suffering from limited convergence, unnatural prompts, and high computational costs.
We introduce Semantic Representation Attack, a novel paradigm that fundamentally reconceptualizes adversarial objectives against aligned LLMs.
Rather than targeting exact textual patterns, our approach exploits the semantic representation space comprising diverse responses with equivalent harmful meanings.
This innovation resolves the inherent trade-off between attack efficacy and prompt naturalness that plagues existing methods.
The Semantic Representation Heuristic Search algorithm is proposed to efficiently generate semantically coherent and concise adversarial prompts by maintaining interpretability during incremental expansion.
We establish rigorous theoretical guarantees for semantic convergence and demonstrate that our method achieves unprecedented attack success rates (89.41\% averaged across 18 LLMs, including 100\% on 11 models) while maintaining stealthiness and efficiency.
Comprehensive experimental results confirm the overall superiority of our Semantic Representation Attack.
The code will be publicly available. 

**Abstract (ZH)**: 大型语言模型（LLMs） increasingly employ对齐技术以防止生成有害输出。尽管存在这些防护措施，攻击者仍然可以通过构建诱使LLMs生成有害内容的定制提示来绕过它们。
当前方法通常针对明确的肯定回应，如“当然，这里是...”，这些方法面临收敛性有限、不自然的提示以及高昂的计算成本。
我们提出语义表示攻击，这是一种全新的范式，从根本上重新定义了对抗对齐LLMs的攻击目标。
我们的方法不针对具体的文本模式，而是利用由具有等效有害含义的多样化响应构成的语义表示空间。
这一创新解决了现有方法中存在的攻击效果与提示自然性之间的固有trade-off。
提出了语义表示启发式搜索算法，以高效地生成语义连贯且简洁的对抗提示，并在增量扩展过程中保持可解释性。
我们为语义收敛提供了严格的理论保证，并证明我们的方法在18个LLM中（包括11个模型达到100%）实现了前所未有的高攻击成功率（平均89.41%），同时保持隐蔽性和高效性。
全面的实验结果证实了我们语义表示攻击的整体优越性。
代码将公开发布。 

---
# Anti-Money Laundering Systems Using Deep Learning 

**Title (ZH)**: 使用深度学习的反洗钱系统 

**Authors**: Mashkhal Abdalwahid Sidiq, Yimamu Kirubel Wondaferew  

**Link**: [PDF](https://arxiv.org/pdf/2509.19359)  

**Abstract**: In this paper, we focused on using deep learning methods for detecting money laundering in financial transaction networks, in order to demonstrate that it can be used as a complement or instead of the more commonly used rule-based systems and conventional Anti-Money Laundering (AML) systems. The paper explores the pivotal role played by Anti-Money Laundering (AML) activities in the global financial industry. It underscores the drawbacks of conventional AML systems, which exhibit high rates of false positives and lack the sophistication to uncover intricate money laundering schemes. To tackle these challenges, the paper proposes an advanced AML system that capitalizes on link analysis using deep learning techniques. At the heart of this system lies the utilization of centrality algorithms like Degree Centrality, Closeness Centrality, Betweenness Centrality, and PageRank. These algorithms enhance the system's capability to identify suspicious activities by examining the influence and interconnections within networks of financial transactions. The significance of Anti-Money Laundering (AML) efforts within the global financial sector is discussed in this paper. It highlights the limitations of traditional AML systems. The results showed the practicality and superiority of the new implementation of the GCN model, which is a preferable method for connectively structured data, meaning that a transaction or account is analyzed in the context of its financial environment. In addition, the paper delves into the prospects of Anti-Money Laundering (AML) efforts, proposing the integration of emerging technologies such as deep learning and centrality algorithms. This integration holds promise for enhancing the effectiveness of AML systems by refining their capabilities. 

**Abstract (ZH)**: 基于深度学习的金融交易网络洗钱检测方法研究：传统反洗钱系统的补充或替代 

---
# Benchmarking and Improving LLM Robustness for Personalized Generation 

**Title (ZH)**: 个性化生成中LLM稳健性基准测试与改进 

**Authors**: Chimaobi Okite, Naihao Deng, Kiran Bodipati, Huaidian Hou, Joyce Chai, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2509.19358)  

**Abstract**: Recent years have witnessed a growing interest in personalizing the responses of large language models (LLMs). While existing evaluations primarily focus on whether a response aligns with a user's preferences, we argue that factuality is an equally important yet often overlooked dimension. In the context of personalization, we define a model as robust if its responses are both factually accurate and align with the user preferences. To assess this, we introduce PERG, a scalable framework for evaluating robustness in LLMs, along with a new dataset, PERGData. We evaluate fourteen models from five different model families using different prompting methods. Our findings show that current LLMs struggle with robust personalization: even the strongest models (GPT-4.1, LLaMA3-70B) fail to maintain correctness in 5% of previously successful cases without personalization, while smaller models (e.g., 7B-scale) can fail more than 20% of the time. Further analysis reveals that robustness is significantly affected by the nature of the query and the type of user preference. To mitigate these failures, we propose Pref-Aligner, a two-stage approach that improves robustness by an average of 25% across models. Our work highlights critical gaps in current evaluation practices and introduces tools and metrics to support more reliable, user-aligned LLM deployments. 

**Abstract (ZH)**: 近年来，个性化大型语言模型（LLMs）的响应引起了越来越多的兴趣。虽然现有的评估主要集中在响应是否符合用户偏好，但我们认为事实性同样是同等重要但经常被忽视的维度。在个性化的情境下，我们定义一个模型为稳健的，如果其响应既准确又符合用户偏好。为了评估这一点，我们引入了PERG，一个评估LLMs稳健性的可扩展框架，以及一个新的数据集PERGData。我们使用不同的提示方法评估了五个不同模型家族中的十四种模型。我们的发现表明，当前的LLMs在稳健个性化方面存在困难：即使是最强的模型（GPT-4.1、LLaMA3-70B）在未个性化的情况下，在先前成功的情况中有5%无法保持正确性，而较小的模型（例如，7B规模）则可能超过20%的时间出现错误。进一步的分析表明，稳健性显著受到查询性质和用户偏好的类型影响。为了缓解这些失败，我们提出了Pref-Aligner，这是一种两阶段方法，能够在模型中平均提高25%的稳健性。我们的工作强调了当前评估实践中的关键空白，并引入了工具和指标以支持更可靠、用户对齐的LLM部署。 

---
# RoadMind: Towards a Geospatial AI Expert for Disaster Response 

**Title (ZH)**: RoadMind: 朝着灾害响应领域的地理空间AI专家迈进 

**Authors**: Ahmed El Fekih Zguir, Ferda Ofli, Muhammad Imran  

**Link**: [PDF](https://arxiv.org/pdf/2509.19354)  

**Abstract**: Large Language Models (LLMs) have shown impressive performance across a range of natural language tasks, but remain limited in their ability to reason about geospatial data, particularly road networks, distances, and directions. This gap poses challenges in disaster scenarios, where spatial understanding is critical for tasks such as evacuation planning and resource allocation. In this work, we present RoadMind, a self-supervised framework that enhances the geospatial reasoning capabilities of LLMs using structured data from OpenStreetMap (OSM). Our automated pipeline extracts road infrastructure data for a given city and converts it into multiple supervision formats tailored to key spatial tasks. We pretrain and fine-tune LLMs on these representations using QLoRA adapters and 4-bit quantized models. We evaluate our approach on three disaster-prone cities with varying global representation, Los Angeles, Christchurch, and Manila, across tasks such as road segment identification, nearest road retrieval, and distance/direction estimation. Our results show that models trained via RoadMind significantly outperform strong baselines, including state-of-the-art LLMs equipped with advanced prompt engineering. This demonstrates the potential of structured geospatial data to enhance language models with robust spatial reasoning, enabling more effective offline AI systems for disaster response. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言任务上表现出色，但在处理地理空间数据，特别是道路网络、距离和方向的推理能力上仍然有限。这种局限性在灾难场景中造成了挑战，因为了解空间关系是疏散规划和资源分配等任务的关键。本文提出RoadMind，这是一种自监督框架，利用OpenStreetMap (OSM) 的结构化数据增强LLMs的地理空间推理能力。我们自动化的流水线为给定城市提取道路基础设施数据，并将其转换为适合关键空间任务的多种监督格式。我们使用QLoRA适配器和4比特量化模型对LLMs进行预训练和微调。我们评估RoadMind该方法在洛杉矶、克赖斯特彻奇和马尼拉这三个不同全球代表性的灾难多发城市中的表现，任务包括道路段识别、最近道路检索和距离/方向估算。结果显示，通过RoadMind训练的模型显著优于强大的基线模型，包括配备高级提示工程的最先进的LLMs。这表明结构化的地理空间数据有潜力增强语言模型的空间推理能力，从而促进更有效的离线AI系统以应对灾难。 

---
# TriSPrompt: A Hierarchical Soft Prompt Model for Multimodal Rumor Detection with Incomplete Modalities 

**Title (ZH)**: TriSPrompt：一种用于多模态谣言检测的多层次软提示模型（缺失模态情况下） 

**Authors**: Jiajun Chen, Yangyang Wu, Xiaoye Miao, Mengying Zhu, Meng Xi  

**Link**: [PDF](https://arxiv.org/pdf/2509.19352)  

**Abstract**: The widespread presence of incomplete modalities in multimodal data poses a significant challenge to achieving accurate rumor detection. Existing multimodal rumor detection methods primarily focus on learning joint modality representations from \emph{complete} multimodal training data, rendering them ineffective in addressing the common occurrence of \emph{missing modalities} in real-world scenarios. In this paper, we propose a hierarchical soft prompt model \textsf{TriSPrompt}, which integrates three types of prompts, \textit{i.e.}, \emph{modality-aware} (MA) prompt, \emph{modality-missing} (MM) prompt, and \emph{mutual-views} (MV) prompt, to effectively detect rumors in incomplete multimodal data. The MA prompt captures both heterogeneous information from specific modalities and homogeneous features from available data, aiding in modality recovery. The MM prompt models missing states in incomplete data, enhancing the model's adaptability to missing information. The MV prompt learns relationships between subjective (\textit{i.e.}, text and image) and objective (\textit{i.e.}, comments) perspectives, effectively detecting rumors. Extensive experiments on three real-world benchmarks demonstrate that \textsf{TriSPrompt} achieves an accuracy gain of over 13\% compared to state-of-the-art methods. The codes and datasets are available at https: //anonymous.this http URL. 

**Abstract (ZH)**: 广泛存在的不完整模态在多模态数据中的普遍存在对实现准确的谣言检测构成了重大挑战。现有的多模态谣言检测方法主要侧重于从完整的多模态训练数据中学习联合模态表示，这使得它们难以应对现实场景中常见的缺失模态问题。在本文中，我们提出了一种层次软提示模型TriSPrompt，该模型整合了三种类型的提示，即模态感知（MA）提示、模态缺失（MM）提示和互视（MV）提示，以有效检测不完整多模态数据中的谣言。MA提示捕捉特定模态的异质信息和可用数据的同质特征，有助于模态恢复。MM提示模型不完整数据中的缺失状态，增强模型对缺失信息的适应性。MV提示学习主观（如文本和图像）和客观（如评论）视角之间的关系，有效检测谣言。在三个真实世界的基准上的 extensive 实验表明，TriSPrompt 在准确率上相对于最先进的方法提高了超过 13%。代码和数据集可从 https://anonymous.this http URL 获取。 

---
# The Impact of Structural Changes on Learning Capacity in the Fly Olfactory Neural Circuit 

**Title (ZH)**: 结构变化对苍蝇嗅觉神经回路学习能力的影响 

**Authors**: Katherine Xie, Gabriel Koch Ocker  

**Link**: [PDF](https://arxiv.org/pdf/2509.19351)  

**Abstract**: The Drosophila mushroom body (MB) is known to be involved in olfactory learning and memory; the synaptic plasticity of the Kenyon cell (KC) to mushroom body output neuron (MBON) synapses plays a key role in the learning process. Previous research has focused on projection neuron (PN) to Kenyon cell (KC) connectivity within the MB; we examine how perturbations to the mushroom body circuit structure and changes in connectivity, specifically within the KC to mushroom body output neuron (MBON) neural circuit, affect the MBONs' ability to distinguish between odor classes. We constructed a neural network that incorporates the connectivity between PNs, KCs, and MBONs. To train our model, we generated ten artificial input classes, which represent the projection neuron activity in response to different odors. We collected data on the number of KC-to-MBON connections, MBON error rates, and KC-to-MBON synaptic weights, among other metrics. We observed that MBONs with very few presynaptic KCs consistently performed worse than others in the odor classification task. The developmental types of KCs also played a significant role in each MBON's output. We performed random and targeted KC ablation and observed that ablating developmentally mature KCs had a greater negative impact on MBONs' learning capacity than ablating immature KCs. Random and targeted pruning of KC-MBON synaptic connections yielded results largely consistent with the ablation experiments. To further explore the various types of KCs, we also performed rewiring experiments in the PN to KC circuit. Our study furthers our understanding of olfactory neuroplasticity and provides important clues to understanding learning and memory in general. Understanding how the olfactory circuits process and learn can also have potential applications in artificial intelligence and treatments for neurodegenerative diseases. 

**Abstract (ZH)**: 果蝇蘑菇体在嗅觉学习和记忆中的作用；肯yon细胞到蘑菇体输出神经元突触可塑性的变化在学习过程中的关键作用：探究蘑菇体电路结构紊乱和突触连接变化对蘑菇体输出神经元区分不同气味类别的影响及其机制。 

---
# SCORE: A Semantic Evaluation Framework for Generative Document Parsing 

**Title (ZH)**: SCORE：生成文档解析的语义评估框架 

**Authors**: Renyu Li, Antonio Jimeno Yepes, Yao You, Kamil Pluciński, Maximilian Operlejn, Crag Wolfe  

**Link**: [PDF](https://arxiv.org/pdf/2509.19345)  

**Abstract**: Multi-modal generative document parsing systems challenge traditional evaluation: unlike deterministic OCR or layout models, they often produce semantically correct yet structurally divergent outputs. Conventional metrics-CER, WER, IoU, or TEDS-misclassify such diversity as error, penalizing valid interpretations and obscuring system behavior.
We introduce SCORE (Structural and COntent Robust Evaluation), an interpretation-agnostic framework that integrates (i) adjusted edit distance for robust content fidelity, (ii) token-level diagnostics to distinguish hallucinations from omissions, (iii) table evaluation with spatial tolerance and semantic alignment, and (iv) hierarchy-aware consistency checks. Together, these dimensions enable evaluation that embraces representational diversity while enforcing semantic rigor.
Across 1,114 pages spanning a holistic benchmark and a field dataset, SCORE consistently revealed cross-dataset performance patterns missed by standard metrics. In 2-5% of pages with ambiguous table structures, traditional metrics penalized systems by 12-25% on average, leading to distorted rankings. SCORE corrected these cases, recovering equivalence between alternative but valid interpretations. Moreover, by normalizing generative outputs into a format-agnostic representation, SCORE reproduces traditional scores (e.g., table F1 up to 0.93) without requiring object-detection pipelines, demonstrating that generative parsing alone suffices for comprehensive evaluation.
By exposing how interpretive diversity impacts evaluation outcomes and providing multi-dimensional, interpretable diagnostics, SCORE establishes foundational principles for semantically grounded, fair, and practical benchmarking of modern document parsing systems. 

**Abstract (ZH)**: 多模态生成文档解析系统挑战传统评估：与确定性的OCR或布局模型不同，它们 often  produce semantically correct yet structurally divergent outputs. Conventional metrics-CER, WER, IoU, or TEDS-misclassify such diversity as error, penalizing valid interpretations and obscuring system behavior.

我们引入了 SCORE（结构和内容稳健评估）框架，这是一个无偏见的解释框架，整合了 (i) 调整后的编辑距离以保证内容的稳健性，(ii) 词元级别诊断以区分幻觉与遗漏，(iii) 带有空间容差和语义对齐的表评估，以及 (iv) 层次感知的一致性检查。这些维度一起，使得评估能够接纳表示多样性的同时，保证语义严谨。

在涵盖从综合基准到领域数据集的1,114 页中，SCORE 一致揭示了标准度量所忽略的跨数据集性能模式。在2-5% 结构模糊的表格页面中，传统度量平均惩罚系统12-25%，导致排名失真。SCORE 纠正了这些情况，使替代但有效的解释恢复了等效性。此外，通过将生成的输出规范化为格式无关表示，SCORE 在无需对象检测管道的情况下再现了传统得分（例如，表F1高达0.93），证明了仅生成解析即可实现全面评估。

通过揭示解释多样性对评估结果的影响，并提供多维度、可解释的诊断，SCORE 为现代文档解析系统建立了坚实的基础原则，以确保语义驱动、公平和实用的基准测试。 

---
# Part-of-speech tagging for Nagamese Language using CRF 

**Title (ZH)**: 使用CRF的纳加梅语词性标注 

**Authors**: Alovi N Shohe, Chonglio Khiamungam, Teisovi Angami  

**Link**: [PDF](https://arxiv.org/pdf/2509.19343)  

**Abstract**: This paper investigates part-of-speech tagging, an important task in Natural Language Processing (NLP) for the Nagamese language. The Nagamese language, a.k.a. Naga Pidgin, is an Assamese-lexified Creole language developed primarily as a means of communication in trade between the Nagas and people from Assam in northeast India. A substantial amount of work in part-of-speech-tagging has been done for resource-rich languages like English, Hindi, etc. However, no work has been done in the Nagamese language. To the best of our knowledge, this is the first attempt at part-of-speech tagging for the Nagamese Language. The aim of this work is to identify the part-of-speech for a given sentence in the Nagamese language. An annotated corpus of 16,112 tokens is created and applied machine learning technique known as Conditional Random Fields (CRF). Using CRF, an overall tagging accuracy of 85.70%; precision, recall of 86%, and f1-score of 85% is achieved.
Keywords. Nagamese, NLP, part-of-speech, machine learning, CRF. 

**Abstract (ZH)**: This paper investigates part-of-speech tagging for the Nagamese language in Natural Language Processing (NLP). 

---
# Fine-Grained AI Model Caching and Downloading With Coordinated Multipoint Broadcasting in Multi-Cell Edge Networks 

**Title (ZH)**: 多小区边缘网络中协调多点广播下的细粒度AI模型缓存与下载 

**Authors**: Yang Fu, Peng Qin, Yueyue Zhang, Yifei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19341)  

**Abstract**: 6G networks are envisioned to support on-demand AI model downloading to accommodate diverse inference requirements of end users. By proactively caching models at edge nodes, users can retrieve the requested models with low latency for on-device AI inference. However, the substantial size of contemporary AI models poses significant challenges for edge caching under limited storage capacity, as well as for the concurrent delivery of heterogeneous models over wireless channels. To address these challenges, we propose a fine-grained AI model caching and downloading system that exploits parameter reusability, stemming from the common practice of fine-tuning task-specific models from a shared pre-trained model with frozen parameters. This system selectively caches model parameter blocks (PBs) at edge nodes, eliminating redundant storage of reusable parameters across different cached models. Additionally, it incorporates coordinated multipoint (CoMP) broadcasting to simultaneously deliver reusable PBs to multiple users, thereby enhancing downlink spectrum utilization. Under this arrangement, we formulate a model downloading delay minimization problem to jointly optimize PB caching, migration (among edge nodes), and broadcasting beamforming. To tackle this intractable problem, we develop a distributed multi-agent learning framework that enables edge nodes to explicitly learn mutual influence among their actions, thereby facilitating cooperation. Furthermore, a data augmentation approach is proposed to adaptively generate synthetic training samples through a predictive model, boosting sample efficiency and accelerating policy learning. Both theoretical analysis and simulation experiments validate the superior convergence performance of the proposed learning framework. 

**Abstract (ZH)**: 6G网络设想支持按需下载AI模型以适应终端用户多变的推理需求。通过在边缘节点上主动缓存模型，用户可以以低延迟的方式在设备上获取所需的模型进行AI推理。然而，现代AI模型的庞大体积给在存储容量有限的边缘节点上进行缓存带来了巨大挑战，并且也对通过无线信道同时传输异构模型提出了挑战。为了解决这些问题，我们提出了一种细粒度的AI模型缓存和下载系统，该系统利用参数重用性，这是从共享的预训练模型中精调特定任务模型的一种常见做法。该系统选择性地在边缘节点上缓存模型参数块（PB），消除不同缓存模型之间可重用参数的冗余存储。此外，系统还结合了协作多点（CoMP）广播，同时向多个用户传输可重用的参数块，从而提高下行链路频谱利用率。在该配置下，我们构建了一个模型下载延迟最小化问题，以联合优化参数块缓存、迁移（在边缘节点之间）和广播波束形成。为了解决这一难以处理的问题，我们开发了一个分布式多代理学习框架，使边缘节点能够明确学习彼此行为的影响，从而促进合作。此外，我们提出了一种数据增强方法，通过预测模型自适应地生成合成训练样本，提升样本效率并加速策略学习。理论分析和仿真实验均验证了所提出的学习框架的优越收敛性能。 

---
# Joint Channel Estimation and Computation Offloading in Fluid Antenna-assisted MEC Networks 

**Title (ZH)**: 液体天线辅助MEC网络中的联合信道估计算法与计算卸载 

**Authors**: Ying Ju, Mingdong Li, Haoyu Wang, Lei Liu, Youyang Qu, Mianxiong Dong, Victor C. M. Leung, Chau Yuen  

**Link**: [PDF](https://arxiv.org/pdf/2509.19340)  

**Abstract**: With the emergence of fluid antenna (FA) in wireless communications, the capability to dynamically adjust port positions offers substantial benefits in spatial diversity and spectrum efficiency, which are particularly valuable for mobile edge computing (MEC) systems. Therefore, we propose an FA-assisted MEC offloading framework to minimize system delay. This framework faces two severe challenges, which are the complexity of channel estimation due to dynamic port configuration and the inherent non-convexity of the joint optimization problem. Firstly, we propose Information Bottleneck Metric-enhanced Channel Compressed Sensing (IBM-CCS), which advances FA channel estimation by integrating information relevance into the sensing process and capturing key features of FA channels effectively. Secondly, to address the non-convex and high-dimensional optimization problem in FA-assisted MEC systems, which includes FA port selection, beamforming, power control, and resource allocation, we propose a game theory-assisted Hierarchical Twin-Dueling Multi-agent Algorithm (HiTDMA) based offloading scheme, where the hierarchical structure effectively decouples and coordinates the optimization tasks between the user side and the base station side. Crucially, the game theory effectively reduces the dimensionality of power control variables, allowing deep reinforcement learning (DRL) agents to achieve improved optimization efficiency. Numerical results confirm that the proposed scheme significantly reduces system delay and enhances offloading performance, outperforming benchmarks. Additionally, the IBM-CCS channel estimation demonstrates superior accuracy and robustness under varying port densities, contributing to efficient communication under imperfect CSI. 

**Abstract (ZH)**: 基于流体天线的移动边缘计算卸载框架：复杂信道估计与联合优化难题的解决方案 

---
# Multi-population Ensemble Genetic Programming via Cooperative Coevolution and Multi-view Learning for Classification 

**Title (ZH)**: 基于协同进化和多视图学习的多群体集成遗传编程分类方法 

**Authors**: Mohammad Sadegh Khorshidi, Navid Yazdanjue, Hassan Gharoun, Mohammad Reza Nikoo, Fang Chen, Amir H. Gandomi  

**Link**: [PDF](https://arxiv.org/pdf/2509.19339)  

**Abstract**: This paper introduces Multi-population Ensemble Genetic Programming (MEGP), a computational intelligence framework that integrates cooperative coevolution and the multiview learning paradigm to address classification challenges in high-dimensional and heterogeneous feature spaces. MEGP decomposes the input space into conditionally independent feature subsets, enabling multiple subpopulations to evolve in parallel while interacting through a dynamic ensemble-based fitness mechanism. Each individual encodes multiple genes whose outputs are aggregated via a differentiable softmax-based weighting layer, enhancing both model interpretability and adaptive decision fusion. A hybrid selection mechanism incorporating both isolated and ensemble-level fitness promotes inter-population cooperation while preserving intra-population diversity. This dual-level evolutionary dynamic facilitates structured search exploration and reduces premature convergence. Experimental evaluations across eight benchmark datasets demonstrate that MEGP consistently outperforms a baseline GP model in terms of convergence behavior and generalization performance. Comprehensive statistical analyses validate significant improvements in Log-Loss, Precision, Recall, F1 score, and AUC. MEGP also exhibits robust diversity retention and accelerated fitness gains throughout evolution, highlighting its effectiveness for scalable, ensemble-driven evolutionary learning. By unifying population-based optimization, multi-view representation learning, and cooperative coevolution, MEGP contributes a structurally adaptive and interpretable framework that advances emerging directions in evolutionary machine learning. 

**Abstract (ZH)**: 多群体集成遗传编程：面向高维异构特征空间分类挑战的计算智能框架 

---
# Radio Propagation Modelling: To Differentiate or To Deep Learn, That Is The Question 

**Title (ZH)**: 无线电波传播建模：是区别建模还是深度学习，这是一个问题。 

**Authors**: Stefanos Bakirtzis, Paul Almasan, José Suárez-Varela, Gabriel O. Ferreira, Michail Kalntis, André Felipe Zanella, Ian Wassell, Andra Lutu  

**Link**: [PDF](https://arxiv.org/pdf/2509.19337)  

**Abstract**: Differentiable ray tracing has recently challenged the status quo in radio propagation modelling and digital twinning. Promising unprecedented speed and the ability to learn from real-world data, it offers a real alternative to conventional deep learning (DL) models. However, no experimental evaluation on production-grade networks has yet validated its assumed scalability or practical benefits. This leaves mobile network operators (MNOs) and the research community without clear guidance on its applicability. In this paper, we fill this gap by employing both differentiable ray tracing and DL models to emulate radio coverage using extensive real-world data collected from the network of a major MNO, covering 13 cities and more than 10,000 antennas. Our results show that, while differentiable ray-tracing simulators have contributed to reducing the efficiency-accuracy gap, they struggle to generalize from real-world data at a large scale, and they remain unsuitable for real-time applications. In contrast, DL models demonstrate higher accuracy and faster adaptation than differentiable ray-tracing simulators across urban, suburban, and rural deployments, achieving accuracy gains of up to 3 dB. Our experimental results aim to provide timely insights into a fundamental open question with direct implications on the wireless ecosystem and future research. 

**Abstract (ZH)**: 可微光线追踪在射频传播建模和数字孪生中的最新进展挑战了传统格局，提供了前所未有的速度，并能够从真实世界数据中学习，为无线通信系统提供了与传统深度学习模型相媲美的实际替代方案。然而，尚无针对生产级网络的实验评估验证其假设的可扩展性和实用优势。这使得移动网络运营商和研究社区在实际应用方面缺乏明确指导。在本文中，我们通过结合使用可微光线追踪和深度学习模型，利用来自一家主要移动网络运营商的广泛真实世界数据，覆盖了13个城市和超过10,000个天线，来填补这一空白。实验结果表明，尽管可微光线追踪模拟器在提高效率-精度折衷方面有所贡献，但在大规模从真实世界数据中泛化方面存在困难，不适用于实时应用。相比之下，深度学习模型在城市、郊区和农村部署中展示了更高的准确性和更快的适应性，实现了高达3 dB的精度提升。我们的实验结果旨在为无线生态系统和未来研究提供及时的、具有直接影响的基本开放问题的见解。 

---
# Cognitive-Level Adaptive Generation via Capability-Aware Retrieval and Style Adaptation 

**Title (ZH)**: 认知层面自适应生成通过能力感知检索与风格适应 

**Authors**: Qingsong Wang, Tao Wu, Wang Lin, Yueying Feng, Gongsheng Yuan, Chang Yao, Jingyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.19336)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong performance in open-ended generation tasks. However, they often struggle to adapt content to users with differing cognitive capacities, leading to a phenomenon we term cognitive misalignment. This issue arises in two forms: knowledge-level misalignment, where content is too complex or too simplistic relative to user understanding, and presentation-style misalignment, where the structure or tone hinders effective comprehension. To address these challenges, we propose the Cognitive-Level Alignment Framework (CLAF), a general-purpose generation framework that aligns both knowledge complexity and presentation style with user cognition. CLAF integrates a capability-aware retrieval module based on a hierarchical knowledge graph and a style optimization module guided by Bloom's taxonomy and preference learning. Additionally, a knowledge-controllable generation component ensures consistency and relevance throughout the output. To support training and evaluation, we construct SCALE, a cognitively annotated dataset containing responses at multiple comprehension levels per query. Empirical results show that CLAF enhances the adaptability and informativeness of LLM outputs across a range of user profiles, offering a robust solution to cognitive-level alignment in real-world applications. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在开放生成任务中显示出了强大的性能，但往往难以适应具有不同认知能力的用户，导致我们称之为认知失准的现象。这一问题主要表现为知识层面的失准，即内容相对于用户理解的复杂度过高或过低，以及表现风格层面的失准，即结构或语气阻碍了有效的理解和吸收。为了解决这些问题，我们提出了认知层面对齐框架（CLAF），这是一种通用生成框架，能够同时对齐知识复杂性和表现风格以适应用户认知。CLAF融合了一个基于分层知识图谱的认知能力感知检索模块和一个由布卢姆分类学和偏好学习指导的风格优化模块。此外，知识可控的生成组件确保输出的连贯性和相关性。为了支持训练和评估，我们构建了SCALE数据集，该数据集包含了根据理解水平标注的响应，每条查询包含多个答案。实验证明，CLAF提升了不同用户群体下LLM输出的适应性和信息量，为实际应用中的认知层面对齐提供了稳健的解决方案。 

---
# CSIYOLO: An Intelligent CSI-based Scatter Sensing Framework for Integrated Sensing and Communication Systems 

**Title (ZH)**: CSIYOLO：一种基于智能信道状态信息的散射感知框架用于整合感知与通信系统 

**Authors**: Xudong Zhang, Jingbo Tan, Zhizhen Ren, Jintao Wang, Yihua Ma, Jian Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.19335)  

**Abstract**: ISAC is regarded as a promising technology for next-generation communication systems, enabling simultaneous data transmission and target sensing. Among various tasks in ISAC, scatter sensing plays a crucial role in exploiting the full potential of ISAC and supporting applications such as autonomous driving and low-altitude economy. However, most existing methods rely on either waveform and hardware modifications or traditional signal processing schemes, leading to poor compatibility with current communication systems and limited sensing accuracy. To address these challenges, we propose CSIYOLO, a framework that performs scatter localization only using estimated CSI from a single base station-user equipment pair. This framework comprises two main components: anchor-based scatter parameter detection and CSI-based scatter localization. First, by formulating scatter parameter extraction as an image detection problem, we propose an anchor-based scatter parameter detection method inspired by You Only Look Once architectures. After that, a CSI-based localization algorithm is derived to determine scatter locations with extracted parameters. Moreover, to improve localization accuracy and implementation efficiency, we design an extendable network structure with task-oriented optimizations, enabling multi-scale anchor detection and better adaptation to CSI characteristics. A noise injection training strategy is further designed to enhance robustness against channel estimation errors. Since the proposed framework operates solely on estimated CSI without modifying waveforms or signal processing pipelines, it can be seamlessly integrated into existing communication systems as a plugin. Experiments show that our proposed method can significantly outperform existing methods in scatter localization accuracy with relatively low complexities under varying numbers of scatters and estimation errors. 

**Abstract (ZH)**: CSIYOLO：基于估计CSI的散射定位框架 

---
# Pluralistic Off-policy Evaluation and Alignment 

**Title (ZH)**: 多元离策评估与对齐 

**Authors**: Chengkai Huang, Junda Wu, Zhouhang Xie, Yu Xia, Rui Wang, Tong Yu, Subrata Mitra, Julian McAuley, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2509.19333)  

**Abstract**: Personalized preference alignment for LLMs with diverse human preferences requires evaluation and alignment methods that capture pluralism. Most existing preference alignment datasets are logged under policies that differ substantially from the evaluated LLMs, and existing off-policy estimators focus solely on overall utility while ignoring preference pluralism. Extending Off-Policy Evaluation (OPE) to pluralistic preference alignment, therefore, remains an open question. Thus, we propose the Pluralistic Off-Policy Evaluation (POPE), the first framework for offline pluralistic preference evaluation and alignment in LLMs. POPE includes a unified reward function that combines (1) a collaborative utility component derived from human preference signals (e.g., upvotes or relevance scores) and (2) a diversity component inspired by entropy-based coverage measures, together reflecting pluralistic alignment. Furthermore, to estimate this reward from logged interactions, we derive decomposable inverse propensity scoring (IPS) estimators that separately evaluate relevance and diversity. Theoretically, we prove that our decomposed IPS estimators establish a lower bound on their variance. With the off-policy evaluated value function, we can directly enable off-policy optimization to further enhance pluralistic alignment. Empirical results demonstrate that POPE efficiently enhances pluralistic response generation and maintains the models' general capabilities on downstream tasks 

**Abstract (ZH)**: 多样化人类偏好的个性化偏好对齐的离线多元后政策评估框架 

---
# Quantifying Compositionality of Classic and State-of-the-Art Embeddings 

**Title (ZH)**: 经典和当今最先进嵌入表示的组合性量化 

**Authors**: Zhijin Guo, Chenhao Xue, Zhaozhen Xu, Hongbo Bo, Yuxuan Ye, Janet B. Pierrehumbert, Martha Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2509.19332)  

**Abstract**: For language models to generalize correctly to novel expressions, it is critical that they exploit access compositional meanings when this is justified. Even if we don't know what a "pelp" is, we can use our knowledge of numbers to understand that "ten pelps" makes more pelps than "two pelps". Static word embeddings such as Word2vec made strong, indeed excessive, claims about compositionality. The SOTA generative, transformer models and graph models, however, go too far in the other direction by providing no real limits on shifts in meaning due to context. To quantify the additive compositionality, we formalize a two-step, generalized evaluation that (i) measures the linearity between known entity attributes and their embeddings via canonical correlation analysis, and (ii) evaluates additive generalization by reconstructing embeddings for unseen attribute combinations and checking reconstruction metrics such as L2 loss, cosine similarity, and retrieval accuracy. These metrics also capture failure cases where linear composition breaks down. Sentences, knowledge graphs, and word embeddings are evaluated and tracked the compositionality across all layers and training stages. Stronger compositional signals are observed in later training stages across data modalities, and in deeper layers of the transformer-based model before a decline at the top layer. Code is available at this https URL. 

**Abstract (ZH)**: 语言模型要正确泛化到新颖表达，必须在适当情况下利用组合意义。即使不知道“pelp”是什么，我们也可以利用对数字的认识理解“十种pelp比两种pelp多”。静态词嵌入如Word2vec对组合性的声明过于强硬，过度夸大了其可能性。然而，当前最先进的生成式transformer模型和图模型则在另一个方向上走得过远，提供了极不现实的意义转化限制。为了量化加性组合性，我们提出了一个两步方法的通用评估框架，即(i)利用典型相关分析测量已知实体属性与其嵌入之间的线性关系，和(ii)通过重构未见属性组合的嵌入并检查重构度量（如L2损失、余弦相似度和检索精度）来评估加性泛化。这些度量也捕捉了线性组合失效的情况。句子、知识图谱和词嵌入在所有层和训练阶段被评估和跟踪其组合性。观察到在训练后期和数据模态中更强的组合性信号，并在基于transformer的模型的深层中出现，在顶层则有所下降。代码详见此链接。 

---
# Holographic Transformers for Complex-Valued Signal Processing: Integrating Phase Interference into Self-Attention 

**Title (ZH)**: 全息变换器在复值信号处理中的应用：将相位干扰融入自我注意力 

**Authors**: Enhao Huang, Zhiyu Zhang, Tianxiang Xu, Chunshu Xia, Kaichun Hu, Yuchen Yang, Tongtong Pan, Dong Dong, Zhan Qin  

**Link**: [PDF](https://arxiv.org/pdf/2509.19331)  

**Abstract**: Complex-valued signals encode both amplitude and phase, yet most deep models treat attention as real-valued correlation, overlooking interference effects. We introduce the Holographic Transformer, a physics-inspired architecture that incorporates wave interference principles into self-attention. Holographic attention modulates interactions by relative phase and coherently superimposes values, ensuring consistency between amplitude and phase. A dual-headed decoder simultaneously reconstructs the input and predicts task outputs, preventing phase collapse when losses prioritize magnitude over phase. We demonstrate that holographic attention implements a discrete interference operator and maintains phase consistency under linear mixing. Experiments on PolSAR image classification and wireless channel prediction show strong performance, achieving high classification accuracy and F1 scores, low regression error, and increased robustness to phase perturbations. These results highlight that enforcing physical consistency in attention leads to generalizable improvements in complex-valued learning and provides a unified, physics-based framework for coherent signal modeling. The code is available at this https URL. 

**Abstract (ZH)**: 复值信号同时编码振幅和相位，而大多数深度模型将注意力视为实值相关性，忽略了干涉效应。我们提出了受物理启发的 holographic transformer 架构，将波干涉原理融入自注意力机制中。holographic 注意力通过相对相位调制交互，并以相干方式叠加值，确保振幅和相位之间的一致性。双头解码器同时重建输入并预测任务输出，防止在损失函数优先考虑幅度而非相位时发生相位坍塌。实验证明 holographic 注意力实现了离散的干涉运算符，并在线性混合下保持相位一致性。在极化雷达图像分类和无线信道预测实验中表现出色，获得高分类准确率和 F1 分数、低回归误差，并增加了对相位扰动的鲁棒性。这些结果表明，在注意力中强制物理一致性可以带来复杂值学习的泛化改进，并提供了一个统一的、基于物理的相干信号建模框架。代码已发布在该网址。 

---
# LibEMER: A novel benchmark and algorithms library for EEG-based Multimodal Emotion Recognition 

**Title (ZH)**: LibEMER: 一种基于EEG的多模态情绪识别新型基准库和算法库 

**Authors**: Zejun Liu, Yunshan Chen, Chengxi Xie, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.19330)  

**Abstract**: EEG-based multimodal emotion recognition(EMER) has gained significant attention and witnessed notable advancements, the inherent complexity of human neural systems has motivated substantial efforts toward multimodal approaches. However, this field currently suffers from three critical limitations: (i) the absence of open-source implementations. (ii) the lack of standardized and transparent benchmarks for fair performance analysis. (iii) in-depth discussion regarding main challenges and promising research directions is a notable scarcity. To address these challenges, we introduce LibEMER, a unified evaluation framework that provides fully reproducible PyTorch implementations of curated deep learning methods alongside standardized protocols for data preprocessing, model realization, and experimental setups. This framework enables unbiased performance assessment on three widely-used public datasets across two learning tasks. The open-source library is publicly accessible at: this https URL 

**Abstract (ZH)**: 基于EEG的多模态情绪识别（EEG-based Multimodal Emotion Recognition, EMER）已获得显著关注并取得了明显进步，人类神经系统的固有复杂性推动了多模态方法的发展。然而，该领域目前存在三个关键限制：(i) 缺乏开源实现；(ii) 缺少标准化和透明的基准测试以进行公平的性能分析；(iii) 关于主要挑战和有前途的研究方向的深入讨论较为稀缺。为解决这些挑战，我们引入了LibEMER，这是一个统一的评估框架，提供了经过精心挑选的深度学习方法的完全可再现的PyTorch实现，以及标准化的数据预处理、模型构建和实验设置协议。该框架能够在两个学习任务上对三个广泛使用的公开数据集进行公平的性能评估。开源库可在以下地址访问：this https URL。 

---
# Human Activity Recognition Based on Electrocardiogram Data Only 

**Title (ZH)**: 基于心电图数据的人体活动识别 

**Authors**: Sina Montazeri, Waltenegus Dargie, Yunhe Feng, Kewei Sha  

**Link**: [PDF](https://arxiv.org/pdf/2509.19328)  

**Abstract**: Human activity recognition is critical for applications such as early intervention and health analytics. Traditional activity recognition relies on inertial measurement units (IMUs), which are resource intensive and require calibration. Although electrocardiogram (ECG)-based methods have been explored, these have typically served as supplements to IMUs or have been limited to broad categorical classification such as fall detection or active vs. inactive in daily activities. In this paper, we advance the field by demonstrating, for the first time, robust recognition of activity only with ECG in six distinct activities, which is beyond the scope of previous work. We design and evaluate three new deep learning models, including a CNN classifier with Squeeze-and-Excitation blocks for channel-wise feature recalibration, a ResNet classifier with dilated convolutions for multiscale temporal dependency capture, and a novel CNNTransformer hybrid combining convolutional feature extraction with attention mechanisms for long-range temporal relationship modeling. Tested on data from 54 subjects for six activities, all three models achieve over 94% accuracy for seen subjects, while CNNTransformer hybrid reaching the best accuracy of 72% for unseen subjects, a result that can be further improved by increasing the training population. This study demonstrates the first successful ECG-only activity classification in multiple physical activities, offering significant potential for developing next-generation wearables capable of simultaneous cardiac monitoring and activity recognition without additional motion sensors. 

**Abstract (ZH)**: 仅基于心电图的人类活动识别对于早期干预和健康数据分析至关重要。 

---
# A systematic review of trial-matching pipelines using large language models 

**Title (ZH)**: 大型语言模型用于试验配对流程的系统综述 

**Authors**: Braxton A. Morrison, Madhumita Sushil, Jacob S. Young  

**Link**: [PDF](https://arxiv.org/pdf/2509.19327)  

**Abstract**: Matching patients to clinical trial options is critical for identifying novel treatments, especially in oncology. However, manual matching is labor-intensive and error-prone, leading to recruitment delays. Pipelines incorporating large language models (LLMs) offer a promising solution. We conducted a systematic review of studies published between 2020 and 2025 from three academic databases and one preprint server, identifying LLM-based approaches to clinical trial matching. Of 126 unique articles, 31 met inclusion criteria. Reviewed studies focused on matching patient-to-criterion only (n=4), patient-to-trial only (n=10), trial-to-patient only (n=2), binary eligibility classification only (n=1) or combined tasks (n=14). Sixteen used synthetic data; fourteen used real patient data; one used both. Variability in datasets and evaluation metrics limited cross-study comparability. In studies with direct comparisons, the GPT-4 model consistently outperformed other models, even finely-tuned ones, in matching and eligibility extraction, albeit at higher cost. Promising strategies included zero-shot prompting with proprietary LLMs like the GPT-4o model, advanced retrieval methods, and fine-tuning smaller, open-source models for data privacy when incorporation of large models into hospital infrastructure is infeasible. Key challenges include accessing sufficiently large real-world data sets, and deployment-associated challenges such as reducing cost, mitigating risk of hallucinations, data leakage, and bias. This review synthesizes progress in applying LLMs to clinical trial matching, highlighting promising directions and key limitations. Standardized metrics, more realistic test sets, and attention to cost-efficiency and fairness will be critical for broader deployment. 

**Abstract (ZH)**: 将患者与临床试验匹配对于发现新型治疗方法至关重要，尤其是在肿瘤学领域。然而，手动匹配劳动密集且容易出错，导致招募延迟。包含大型语言模型（LLMs）的管道提供了一种有希望的解决方案。我们在2020年至2025年间从三个学术数据库和一个预印本服务器中进行了系统性回顾，识别了基于LLM的临床试验匹配方法。在126篇独特文章中，有31篇符合纳入标准。回顾的研究重点仅包括患者与标准匹配（n=4）、患者与试验匹配（n=10）、试验与患者匹配（n=2）、二元资格分类（n=1）以及结合任务（n=14）。16篇研究使用了合成数据；14篇使用了真实患者数据；1篇同时使用了合成数据和真实患者数据。数据集和评估指标的差异性限制了研究间的可比性。在有直接比较的研究中，GPT-4模型在匹配和资格提取方面始终优于其他模型，即使在细调后的模型也是如此，尽管成本较高。有效的策略包括使用专用的LLM（如GPT-4o模型）进行零样本提示、高级检索方法以及在难以将大型模型集成到医院基础设施的情况下，微调更小的开源模型以保护数据隐私。主要挑战包括获取足够大的真实世界数据集，以及部署相关挑战，如降低成本、减轻幻觉风险、数据泄露和偏见。这项回顾总结了将LLM应用于临床试验匹配的进展，指出了有前景的方向和关键限制。标准化指标、更现实的测试集以及对成本效率和公平的关注将是更大范围部署的关键。 

---
# Unveiling the Merits and Defects of LLMs in Automatic Review Generation for Scientific Papers 

**Title (ZH)**: 揭示大语言模型在科学论文自动评审生成中的优势与缺陷 

**Authors**: Ruochi Li, Haoxuan Zhang, Edward Gehringer, Ting Xiao, Junhua Ding, Haihua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.19326)  

**Abstract**: The surge in scientific submissions has placed increasing strain on the traditional peer-review process, prompting the exploration of large language models (LLMs) for automated review generation. While LLMs demonstrate competence in producing structured and coherent feedback, their capacity for critical reasoning, contextual grounding, and quality sensitivity remains limited. To systematically evaluate these aspects, we propose a comprehensive evaluation framework that integrates semantic similarity analysis and structured knowledge graph metrics to assess LLM-generated reviews against human-written counterparts. We construct a large-scale benchmark of 1,683 papers and 6,495 expert reviews from ICLR and NeurIPS in multiple years, and generate reviews using five LLMs. Our findings show that LLMs perform well in descriptive and affirmational content, capturing the main contributions and methodologies of the original work, with GPT-4o highlighted as an illustrative example, generating 15.74% more entities than human reviewers in the strengths section of good papers in ICLR 2025. However, they consistently underperform in identifying weaknesses, raising substantive questions, and adjusting feedback based on paper quality. GPT-4o produces 59.42% fewer entities than real reviewers in the weaknesses and increases node count by only 5.7% from good to weak papers, compared to 50% in human reviews. Similar trends are observed across all conferences, years, and models, providing empirical foundations for understanding the merits and defects of LLM-generated reviews and informing the development of future LLM-assisted reviewing tools. Data, code, and more detailed results are publicly available at this https URL. 

**Abstract (ZH)**: 科学投稿量的激增对传统同行评审过程产生了越来越大的压力，促使人们探索大型语言模型（LLMs）以自动化生成评审。虽然LLMs在生成结构化和连贯反馈方面表现出色，但在批判性推理、上下文关联和质量敏感性方面的能力仍然有限。为系统评估这些方面，我们提出了一种综合评估框架，该框架结合了语义相似性分析和结构化知识图谱指标，用于评估LLM生成的评审与人类撰写的同类评审。我们构建了一个包含1,683篇论文和6,495份专家评审的大规模基准，来源包括多个年度的ICLR和NeurIPS，使用五种LLM生成评审。研究结果表明，LLMs在描述性和肯定性内容方面表现良好，能够捕捉原始工作的主要贡献和方法，GPT-4o在ICLR 2025优良论文的优点部分生成的实体多出15.74%。然而，他们在识别缺点、提出实质性问题以及根据论文质量调整反馈方面表现逊色。GPT-4o在缺点部分生成的实体比真实评审人少59.42%，从优良论文到较差论文的节点数量仅增长5.7%，而人类评审人在这一比例为50%。在所有会议、年度和模型中观察到类似趋势，为理解LLM生成评审的优势和缺陷提供了实证基础，并为未来LLM辅助评审工具的发展提供了指导。相关数据、代码和更详细的结果可在以下网址获取。 

---
# Magnitude Matters: a Superior Class of Similarity Metrics for Holistic Semantic Understanding 

**Title (ZH)**: 量纲很重要：一种优于整体语义理解的相似度度量类别 

**Authors**: V.S. Raghu Parupudi  

**Link**: [PDF](https://arxiv.org/pdf/2509.19323)  

**Abstract**: Vector comparison in high dimensions is a fundamental task in NLP, yet it is dominated by two baselines: the raw dot product, which is unbounded and sensitive to vector norms, and the cosine similarity, which discards magnitude information entirely. This paper challenges both standards by proposing and rigorously evaluating a new class of parameter-free, magnitude-aware similarity metrics. I introduce two such functions, Overlap Similarity (OS) and Hyperbolic Tangent Similarity (HTS), designed to integrate vector magnitude and alignment in a more principled manner. To ensure that my findings are robust and generalizable, I conducted a comprehensive evaluation using four state-of-the-art sentence embedding models (all-MiniLM-L6-v2, all-mpnet-base-v2, paraphrase-mpnet-base-v2, and BAAI/bge-large-en-v1.5) across a diverse suite of eight standard NLP benchmarks, including STS-B, SICK, Quora, and PAWS. Using the Wilcoxon signed-rank test for statistical significance, my results are definitive: on the tasks requiring holistic semantic understanding (paraphrase and inference), both OS and HTS provide a statistically significant improvement in Mean Squared Error over both the raw dot product and cosine similarity, regardless of the underlying embedding this http URL, my findings delineate the specific domain of advantage for these metrics: for tasks requiring holistic semantic understanding like paraphrase and inference, my magnitude-aware metrics offer a statistically superior alternative. This significant improvement was not observed on benchmarks designed to test highly nuanced compositional semantics (SICK, STS-B), identifying the challenge of representing compositional text as a distinct and important direction for future work. 

**Abstract (ZH)**: 高维度向量比较在NLP中是一个基本任务，但主要受两种基线的支配：未加权的点积，其无界且对向量范数敏感，和余弦相似度，其完全忽略了向量的幅度信息。本文通过提出并严格评估一个新的参数自由、幅度感知相似度度量类别，挑战了这两种标准。文中介绍了两种此类函数，重叠相似度（OS）和双曲正切相似度（HTS），旨在以更为合理的方式整合向量幅度和对齐信息。为确保研究结果的稳健性和普遍适用性，研究使用四种当前最先进的句子嵌入模型（all-MiniLM-L6-v2、all-mpnet-base-v2、paraphrase-mpnet-base-v2 和 BAAI/bge-large-en-v1.5）在八个标准NLP基准测试（包括STS-B、SICK、Quora 和 PAWS）上进行了全面评估。利用Wilcoxon符号秩检验统计显著性，研究结果明确显示：在需要整体语义理解的任务（如改写和推理）中，OS 和 HTS 在均方误差方面相对于未加权的点积和余弦相似度提供了统计显著的改进，无论底层嵌入如何。本研究明确了这些度量的具体适用领域：对于需要整体语义理解的任务（如改写和推理），我的幅度感知度量提供了统计上更优的替代方案。这种显著改进未在旨在测试高度细腻组成语义的基准测试（SICK、STS-B）中观察到，这表明表示组成性文本的不同和重要的研究方向。 

---
# Readme_AI: Dynamic Context Construction for Large Language Models 

**Title (ZH)**: Readme_AI: 大型语言模型的动态上下文构建 

**Authors**: Millie Vyas, Timothy Blattner, Alden Dima  

**Link**: [PDF](https://arxiv.org/pdf/2509.19322)  

**Abstract**: Despite being trained on significant amounts of data, Large Language Models (LLMs) can provide inaccurate or unreliable information in the context of a user's specific query. Given query-specific context significantly improves the usefulness of its responses. In this paper, we present a specification that can be used to dynamically build context for data sources. The data source owner creates the file containing metadata for LLMs to use when reasoning about dataset-related queries. To demonstrate our proposed specification, we created a prototype Readme_AI Model Context Protocol (MCP) server that retrieves the metadata from the data source and uses it to dynamically build context. Some features that make this specification dynamic are the extensible types that represent crawling web-pages, fetching data from data repositories, downloading and parsing publications, and general text. The context is formatted and grouped using user-specified tags that provide clear contextual information for the LLM to reason about the content. We demonstrate the capabilities of this early prototype by asking the LLM about the NIST-developed Hedgehog library, for which common LLMs often provides inaccurate and irrelevant responses containing hallucinations. With Readme_AI, the LLM receives enough context that it is now able to reason about the library and its use, and even generate code interpolated from examples that were included in the Readme_AI file provided by Hedgehog's developer. Our primary contribution is a extensible protocol for dynamically grounding LLMs in specialized, owner-provided data, enhancing responses from LLMs and reducing hallucinations. The source code for the Readme_AI tool is posted here: this https URL . 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在大量数据训练下工作，但在处理用户特定查询时仍可能提供不准确或不可靠的信息。给定查询特定的上下文显著提高了其响应的有用性。在本文中，我们提出了一种规范，可用于动态构建数据源上下文。数据源所有者创建包含元数据的文件，供LLMs在处理与数据集相关的查询时使用。为了展示我们提出的规范，我们创建了一个原型Readme_AI Model Context Protocol（MCP）服务器，该服务器从数据源检索元数据，并使用这些信息动态构建上下文。使该规范动态的某些功能包括表示爬取网页、从数据仓库获取数据、下载和解析出版物以及通用文本的可扩展类型。上下文按照用户指定的标签格式化和分组，这些标签为LLM提供清晰的上下文信息，使其能够推理内容。通过使用Readme_AI，LLM接收到足够的上下文，现在能够推理Hedgehog库及其用法，并甚至生成来自Readme_AI文件中提供的示例插值的代码。我们的主要贡献是一种可扩展协议，用于动态地将LLMs与特定的所有者提供的数据关联起来，从而增强LLMs的响应并减少幻觉现象。Readme_AI工具的源代码在此处发布：this https URL。 

---
# FHIR-AgentBench: Benchmarking LLM Agents for Realistic Interoperable EHR Question Answering 

**Title (ZH)**: FHIR-AgentBench: LLM代理的实时互操作医疗记录问答基准测试 

**Authors**: Gyubok Lee, Elea Bach, Eric Yang, Tom Pollard, Alistair Johnson, Edward Choi, Yugang jia, Jong Ha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.19319)  

**Abstract**: The recent shift toward the Health Level Seven Fast Healthcare Interoperability Resources (HL7 FHIR) standard opens a new frontier for clinical AI, demanding LLM agents to navigate complex, resource-based data models instead of conventional structured health data. However, existing benchmarks have lagged behind this transition, lacking the realism needed to evaluate recent LLMs on interoperable clinical data. To bridge this gap, we introduce FHIR-AgentBench, a benchmark that grounds 2,931 real-world clinical questions in the HL7 FHIR standard. Using this benchmark, we systematically evaluate agentic frameworks, comparing different data retrieval strategies (direct FHIR API calls vs. specialized tools), interaction patterns (single-turn vs. multi-turn), and reasoning strategies (natural language vs. code generation). Our experiments highlight the practical challenges of retrieving data from intricate FHIR resources and the difficulty of reasoning over them, both of which critically affect question answering performance. We publicly release the FHIR-AgentBench dataset and evaluation suite (this https URL) to promote reproducible research and the development of robust, reliable LLM agents for clinical applications. 

**Abstract (ZH)**: 华为水平七快速医疗互操作资源标准（HL7 FHIR）的 recent shift 为临床AI开辟了新领域，要求LLM代理导航基于资源的复杂数据模型，而非传统的结构化医疗数据。然而，现有基准未能跟上这一转变，缺乏评估最新LLM在可互操作临床数据上的表现所需的现实性。为弥补这一差距，我们引入了FHIR-AgentBench基准，该基准将2,931个真实世界的临床问题与HL7 FHIR标准相结合。利用此基准，我们系统地评估了代理框架，比较了不同的数据检索策略（直接FHIR API调用 vs. 专用工具）、交互模式（单轮 vs. 多轮）以及推理策略（自然语言 vs. 代码生成）。我们的实验强调了从复杂FHIR资源检索数据的实际挑战以及在其中推理的难度，这两者都严重影响了问题回答的表现。我们公开发布了FHIR-AgentBench数据集和评估套件（this https URL），以促进可重复研究并推动稳健可靠的临床应用LLM代理的发展。 

---
# Advancing Few-Shot Pediatric Arrhythmia Classification with a Novel Contrastive Loss and Multimodal Learning 

**Title (ZH)**: 基于新型对比损失和多模态学习的少量样本儿童心律失常分类改进研究 

**Authors**: Yiqiao Chen, Zijian Huang, Zhenghui Feng  

**Link**: [PDF](https://arxiv.org/pdf/2509.19315)  

**Abstract**: Pediatric arrhythmias are a major risk factor for disability and sudden cardiac death, yet their automated classification remains challenging due to class imbalance, few-shot categories, and complex signal characteristics, which severely limit the efficiency and reliability of early screening and clinical intervention. To address this problem, we propose a multimodal end-to-end deep learning framework that combines dual-branch convolutional encoders for ECG and IEGM, semantic attention for cross-modal feature alignment, and a lightweight Transformer encoder for global dependency modeling. In addition, we introduce a new contrastive loss fucntion named Adaptive Global Class-Aware Contrastive Loss (AGCACL) to enhance intra-class compactness and inter-class separability through class prototypes and a global similarity matrix. To the best of our knowledge, this is the first systematic study based on the Leipzig Heart Center pediatric/congenital ECG+IEGM dataset, for which we also provide a complete and reproducible preprocessing pipeline. Experimental results demonstrate that the proposed method achieves the overall best performance on this dataset, including 97.76\% Top-1 Accuracy, 94.08\% Macro Precision, 91.97\% Macro Recall, 92.97\% Macro F1, and 92.36\% Macro F2, with improvements of +13.64, +15.96, +19.82, and +19.44 percentage points over the strongest baseline in Macro Precision/Recall/F1/F2, respectively. These findings indicate that the framework significantly improves the detectability and robustness for minority arrhythmia classes, offering potential clinical value for rhythm screening, pre-procedural assessment, and postoperative follow-up in pediatric and congenital heart disease populations. 

**Abstract (ZH)**: 儿童心律失常是导致残疾和突发心脏死亡的主要风险因素，但由于类样本不平衡、少样本类别以及复杂信号特征的影响，其自动化分类仍然具有挑战性，这严重限制了早期筛查和临床干预的效率和可靠性。为了解决这一问题，我们提出了一种结合双支路卷积编码器进行心电图（ECG）和体表心电图（IEGM）处理、语义注意力进行跨模态特征对齐以及轻量级Transformer编码器进行全局依赖建模的多模态端到端深度学习框架。此外，我们引入了一种新的对比损失函数——自适应全局类感知对比损失（AGCACL），通过类别原型和全局相似性矩阵增强类内凝聚性和类间区分性。据我们所知，这是基于莱比锡心脏中心儿童/先天性心脏病ECG+IEGM数据集的第一个系统性研究，我们还提供了一个完整的可复现实验预处理管道。实验结果表明，所提方法在该数据集上的总体性能最佳，包括97.76%的Top-1准确率、94.08%的宏精度、91.97%的宏召回率、92.97%的宏F1值和92.36%的宏F2值，分别提高了13.64、15.96、19.82和19.44个百分点，对比最强基线。这些发现表明，该框架显著提高了对少数心律失常类别的检测能力和鲁棒性，为儿童和先天性心脏病患者的节律筛查、术前评估和术后随访提供潜在的临床价值。 

---
# Automated Item Neutralization for Non-Cognitive Scales: A Large Language Model Approach to Reducing Social-Desirability Bias 

**Title (ZH)**: 自动项目中和以减少社会偏好偏见：大型语言模型方法在非认知量表中的应用 

**Authors**: Sirui Wu, Daijin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19314)  

**Abstract**: This study evaluates item neutralization assisted by the large language model (LLM) to reduce social desirability bias in personality assessment. GPT-o3 was used to rewrite the International Personality Item Pool Big Five Measure (IPIP-BFM-50), and 203 participants completed either the original or neutralized form along with the Marlowe-Crowne Social Desirability Scale. The results showed preserved reliability and a five-factor structure, with gains in Conscientiousness and declines in Agreeableness and Openness. The correlations with social desirability decreased for several items, but inconsistently. Configural invariance held, though metric and scalar invariance failed. Findings support AI neutralization as a potential but imperfect bias-reduction method. 

**Abstract (ZH)**: 基于大型语言模型的项目中和对人格评估中社会可接受性偏见的减少的评估 

---
# E2E Learning Massive MIMO for Multimodal Semantic Non-Orthogonal Transmission and Fusion 

**Title (ZH)**: 端到端学习大规模MIMO多模态语义非正交传输与融合 

**Authors**: Minghui Wu, Zhen Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.19312)  

**Abstract**: Massive multiple-input multiple-output (MIMO) promises high spectral efficiency but also leads to high-dimensional downlink channel state information (CSI), which complicates real-time channel acquisition and precoding. To address this, we propose an end-to-end (E2E) uplink-downlink CSI fusion precoding network that jointly models downlink CSI reference signal (CSI-RS) design, CSI feedback, and base-station (BS) precoding within a single E2E neural architecture. Concretely, a projection network built on the MAXIM architecture takes uplink sounding reference signals (SRS) as input and outputs frequency-, beam-, and port-domain projection matrices for designing downlink CSI-RS. User equipment (UE) then compresses/quantizes the resulting CSI-RS observations and feeds back a compact representation. At the base station (BS), two complementary branches produce candidate precoders: one is a feedback-only precoding network driven by quantized downlink observations, and the other is an SRS-only precoding network driven by uplink SRS. These candidate precoders are subsequently combined by a fusion precoding network to yield the final transmit precoder. All the modules are trained with a spectral-efficiency-oriented loss under a three-stage schedule. Simulation results show that the proposed approach effectively harnesses both SRS-derived information and UE feedback, achieving markedly better performance than conventional baselines. 

**Abstract (ZH)**: 大规模多输入多输出(MIMO) promises高频谱效率但同时也导致了高维下行信道状态信息(CSI)复杂化，这使得实时信道获取和预编码复杂化。为了解决这一问题，我们提出了一种端到端(E2E)上行-下行CSI融合预编码网络，该网络在一个完整的E2E神经架构中联合建模了下行CSI参考信号(CSI-RS)设计、CSI反馈和基站(BS)预编码。具体而言，基于MAXIM架构构建的投影网络以上行探测参考信号(SRS)作为输入，并输出用于设计下行CSI-RS的频域、波束域和端口域投影矩阵。用户设备(UE)随后压缩/量化所得的CSI-RS观测值，并反馈一个紧凑的表示。在基站(BS)，两个互补分支产生候选预编码器：一个由量化下行观测值驱动的反馈仅预编码网络，另一个由上行SRS驱动的SRS仅预编码网络。这些候选预编码器随后由融合预编码网络结合，以生成最终的传输预编码器。所有模块在三阶段训练计划下使用以频谱效率为导向的损失函数进行训练。仿真结果表明，所提出的方法有效地利用了SRS信息和UE反馈，显著优于传统的baseline方法。 

---
# A Federated Fine-Tuning Paradigm of Foundation Models in Heterogenous Wireless Networks 

**Title (ZH)**: 基础模型在异构无线网络中的联邦微调范式 

**Authors**: Jingyi Wang, Zhongyuan Zhao, Qingtian Wang, Zexu Li, Yue Wang, Tony Q. S. Quek  

**Link**: [PDF](https://arxiv.org/pdf/2509.19306)  

**Abstract**: Edge intelligence has emerged as a promising strategy to deliver low-latency and ubiquitous services for mobile devices. Recent advances in fine-tuning mechanisms of foundation models have enabled edge intelligence by integrating low-rank adaptation (LoRA) with federated learning. However, in wireless networks, the device heterogeneity and resource constraints on edge devices pose great threats to the performance of federated fine-tuning. To tackle these issues, we propose to optimize federated fine-tuning in heterogenous wireless networks via online learning. First, the framework of switching-based federated fine-tuning in wireless networks is provided. The edge devices switches to LoRA modules dynamically for federated fine-tuning with base station to jointly mitigate the impact of device heterogeneity and transmission unreliability. Second, a tractable upper bound on the inference risk gap is derived based on theoretical analysis. To improve the generalization capability, we formulate a non-convex mixed-integer programming problem with long-term constraints, and decouple it into model switching, transmit power control, and bandwidth allocation subproblems. An online optimization algorithm is developed to solve the problems with polynomial computational complexity. Finally, the simulation results on the SST-2 and QNLI data sets demonstrate the performance gains in test accuracy and energy efficiency. 

**Abstract (ZH)**: 基于无线网络异构性的联邦微调在线优化 

---
# Wavelet Fourier Diffuser: Frequency-Aware Diffusion Model for Reinforcement Learning 

**Title (ZH)**: 小波傅里叶散逸器：基于频率的强化学习扩散模型 

**Authors**: Yifu Luo, Yongzhe Chang, Xueqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19305)  

**Abstract**: Diffusion probability models have shown significant promise in offline reinforcement learning by directly modeling trajectory sequences. However, existing approaches primarily focus on time-domain features while overlooking frequency-domain features, leading to frequency shift and degraded performance according to our observation. In this paper, we investigate the RL problem from a new perspective of the frequency domain. We first observe that time-domain-only approaches inadvertently introduce shifts in the low-frequency components of the frequency domain, which results in trajectory instability and degraded performance. To address this issue, we propose Wavelet Fourier Diffuser (WFDiffuser), a novel diffusion-based RL framework that integrates Discrete Wavelet Transform to decompose trajectories into low- and high-frequency components. To further enhance diffusion modeling for each component, WFDiffuser employs Short-Time Fourier Transform and cross attention mechanisms to extract frequency-domain features and facilitate cross-frequency interaction. Extensive experiment results on the D4RL benchmark demonstrate that WFDiffuser effectively mitigates frequency shift, leading to smoother, more stable trajectories and improved decision-making performance over existing methods. 

**Abstract (ZH)**: 频域视角下的扩散概率模型在离线强化学习中的应用：Wavelet Fourier Diffuser (WFDiffuser) 的提出 

---
# LLMs as verification oracles for Solidity 

**Title (ZH)**: LLMs作为Solidity的验证 oracle 

**Authors**: Massimo Bartoletti, Enrico Lipparini, Livio Pompianu  

**Link**: [PDF](https://arxiv.org/pdf/2509.19153)  

**Abstract**: Ensuring the correctness of smart contracts is critical, as even subtle flaws can lead to severe financial losses. While bug detection tools able to spot common vulnerability patterns can serve as a first line of defense, most real-world exploits and losses stem from errors in the contract business logic. Formal verification tools such as SolCMC and the Certora Prover address this challenge, but their impact remains limited by steep learning curves and restricted specification languages. Recent works have begun to explore the use of large language models (LLMs) for security-related tasks such as vulnerability detection and test generation. Yet, a fundamental question remains open: can LLMs serve as verification oracles, capable of reasoning about arbitrary contract-specific properties? In this paper, we provide the first systematic evaluation of GPT-5, a state-of-the-art reasoning LLM, in this role. We benchmark its performance on a large dataset of verification tasks, compare its outputs against those of established formal verification tools, and assess its practical effectiveness in real-world auditing scenarios. Our study combines quantitative metrics with qualitative analysis, and shows that recent reasoning-oriented LLMs can be surprisingly effective as verification oracles, suggesting a new frontier in the convergence of AI and formal methods for secure smart contract development and auditing. 

**Abstract (ZH)**: 确保智能合约的正确性至关重要，即使是细微的缺陷也可能导致严重的财务损失。虽然能检测常见漏洞模式的bug检测工具可以作为第一道防线，但大多数实际的利用和损失源自合约业务逻辑错误。形式验证工具如SolCMC和Certora Prover解决了这一挑战，但它们的影响仍受限于陡峭的学习曲线和受限的规格语言。近期研究表明，大型语言模型（LLMs）可用于安全相关任务，如漏洞检测和测试生成。然而，一个基本问题仍然悬而未决：LLMs能否作为验证 oracle，能够对任意合约特定属性进行推理？在本文中，我们首次系统评估了GPT-5这一最先进的推理LLM在这一角色上的表现。我们在一个大规模的形式验证任务数据集上测试其性能，将其输出与现有的形式验证工具进行比较，并评估其在实际审计场景中的实用性。我们的研究结合了定量指标和定性分析，表明近期的推理导向型LLM可以出乎意料地有效地作为验证oracle，这暗示了AI与形式方法在安全智能合约开发和审计中的融合的新前沿。 

---
# GAUSS: Benchmarking Structured Mathematical Skills for Large Language Models 

**Title (ZH)**: GAUSS: 大型语言模型结构化数学能力基准测试 

**Authors**: Yue Zhang, Jiaxin Zhang, Qiuyu Ren, Tahsin Saffat, Xiaoxuan Liu, Zitong Yang, Banghua Zhu, Yi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.18122)  

**Abstract**: We introduce \textbf{GAUSS} (\textbf{G}eneral \textbf{A}ssessment of \textbf{U}nderlying \textbf{S}tructured \textbf{S}kills in Mathematics), a benchmark that evaluates LLMs' mathematical abilities across twelve core skill dimensions, grouped into three domains: knowledge and understanding, problem solving and communication, and meta-skills and creativity. By categorizing problems according to cognitive skills and designing tasks that isolate specific abilities, GAUSS constructs comprehensive, fine-grained, and interpretable profiles of models' mathematical abilities. These profiles faithfully represent their underlying mathematical intelligence. To exemplify how to use the \textsc{GAUSS} benchmark, we have derived the skill profile of \textsc{GPT-5-thinking}, revealing its strengths and weaknesses as well as its differences relative to \textsc{o4-mini-high}, thereby underscoring the value of multidimensional, skill-based evaluation. 

**Abstract (ZH)**: 我们将介绍\textbf{GAUSS}（\textbf{G}eneral \textbf{A}ssessment of \textbf{U}nderlying \textbf{S}tructured \textbf{S}kills in Mathematics），这是一个评估大型语言模型在十二个核心技能维度上数学能力的基准，这些维度被分为三个领域：知识与理解、问题解决与沟通、元技能与创造力。通过根据认知技能分类问题并设计分离特定能力的任务，GAUSS构建了模型数学能力的全面、细致且可解释的画像，这些画像忠实地代表了其潜在的数学智能。为了说明如何使用\textsc{GAUSS}基准，我们为\textsc{GPT-5-thinking}制定了技能画像，揭示了其优势和劣势，以及与\textsc{o4-mini-high}的区别，从而强调了多维度、基于技能评价的价值。 

---
