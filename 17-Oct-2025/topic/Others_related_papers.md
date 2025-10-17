# RDD: Retrieval-Based Demonstration Decomposer for Planner Alignment in Long-Horizon Tasks 

**Title (ZH)**: 基于检索的演示分解器：长期任务计划者对齐的检索式示例分解方法 

**Authors**: Mingxuan Yan, Yuping Wang, Zechun Liu, Jiachen Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14968)  

**Abstract**: To tackle long-horizon tasks, recent hierarchical vision-language-action (VLAs) frameworks employ vision-language model (VLM)-based planners to decompose complex manipulation tasks into simpler sub-tasks that low-level visuomotor policies can easily handle. Typically, the VLM planner is finetuned to learn to decompose a target task. This finetuning requires target task demonstrations segmented into sub-tasks by either human annotation or heuristic rules. However, the heuristic subtasks can deviate significantly from the training data of the visuomotor policy, which degrades task performance. To address these issues, we propose a Retrieval-based Demonstration Decomposer (RDD) that automatically decomposes demonstrations into sub-tasks by aligning the visual features of the decomposed sub-task intervals with those from the training data of the low-level visuomotor policies. Our method outperforms the state-of-the-art sub-task decomposer on both simulation and real-world tasks, demonstrating robustness across diverse settings. Code and more results are available at this http URL. 

**Abstract (ZH)**: 基于检索的演示分解器 (RDD)：自动将演示分解为子任务以应对长期任务 

---
# SADCHER: Scheduling using Attention-based Dynamic Coalitions of Heterogeneous Robots in Real-Time 

**Title (ZH)**: SADCHER：基于注意力动态异构机器人联盟的实时调度 

**Authors**: Jakob Bichler, Andreu Matoses Gimenez, Javier Alonso-Mora  

**Link**: [PDF](https://arxiv.org/pdf/2510.14851)  

**Abstract**: We present Sadcher, a real-time task assignment framework for heterogeneous multi-robot teams that incorporates dynamic coalition formation and task precedence constraints. Sadcher is trained through Imitation Learning and combines graph attention and transformers to predict assignment rewards between robots and tasks. Based on the predicted rewards, a relaxed bipartite matching step generates high-quality schedules with feasibility guarantees. We explicitly model robot and task positions, task durations, and robots' remaining processing times, enabling advanced temporal and spatial reasoning and generalization to environments with different spatiotemporal distributions compared to training. Trained on optimally solved small-scale instances, our method can scale to larger task sets and team sizes. Sadcher outperforms other learning-based and heuristic baselines on randomized, unseen problems for small and medium-sized teams with computation times suitable for real-time operation. We also explore sampling-based variants and evaluate scalability across robot and task counts. In addition, we release our dataset of 250,000 optimal schedules: this https URL 

**Abstract (ZH)**: 我们提出Sadcher——一种结合动态联盟形成和任务优先级约束的异构多机器人团队实时任务分配框架。Sadcher通过模仿学习训练，并结合图注意力和变压器来预测机器人与任务之间的分配奖励。基于预测的奖励，一个松弛的二部图匹配步骤生成具有可行性保证的高质量调度。我们明确建模了机器人和任务的位置、任务持续时间和剩余处理时间，从而实现高级的时间和空间推理，并能够推广到具有不同时空分布的环境。在最优解的小规模实例上训练，我们的方法可以扩展到更大的任务集和团队规模。在计算时间适合实时操作的情况下，Sadcher在随机未见过的小规模和中规模机器人的问题上优于其他基于学习和启发式的基线方法。我们还探索了采样变体，并评估了其在机器人和任务数量上的扩展性。此外，我们发布了包含250,000个最优调度的数据集：https://.... 

---
# Restoring Noisy Demonstration for Imitation Learning With Diffusion Models 

**Title (ZH)**: 基于扩散模型修复噪声示范以实现模仿学习 

**Authors**: Shang-Fu Chen, Co Yong, Shao-Hua Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.14467)  

**Abstract**: Imitation learning (IL) aims to learn a policy from expert demonstrations and has been applied to various applications. By learning from the expert policy, IL methods do not require environmental interactions or reward signals. However, most existing imitation learning algorithms assume perfect expert demonstrations, but expert demonstrations often contain imperfections caused by errors from human experts or sensor/control system inaccuracies. To address the above problems, this work proposes a filter-and-restore framework to best leverage expert demonstrations with inherent noise. Our proposed method first filters clean samples from the demonstrations and then learns conditional diffusion models to recover the noisy ones. We evaluate our proposed framework and existing methods in various domains, including robot arm manipulation, dexterous manipulation, and locomotion. The experiment results show that our proposed framework consistently outperforms existing methods across all the tasks. Ablation studies further validate the effectiveness of each component and demonstrate the framework's robustness to different noise types and levels. These results confirm the practical applicability of our framework to noisy offline demonstration data. 

**Abstract (ZH)**: 基于滤波与恢复的模仿学习框架：应对内在噪声的专家演示利用 

---
# Requirement Identification for Traffic Simulations in Driving Simulators 

**Title (ZH)**: 驾驶模拟器中交通模拟的需求识别 

**Authors**: Sven Tarlowski, Lutz Eckstein  

**Link**: [PDF](https://arxiv.org/pdf/2510.14653)  

**Abstract**: This paper addresses the challenge of ensuring realistic traffic conditions by proposing a methodology that systematically identifies traffic simulation requirements. Using a structured approach based on sub-goals in each study phase, specific technical needs are derived for microscopic levels, agent models, and visual representation. The methodology aims to maintain a high degree of fidelity, enhancing both the validity of experimental outcomes and participant engagement. By providing a clear link between study objectives and traffic simulation design, this approach supports robust automotive development and testing. 

**Abstract (ZH)**: 本文提出了一种方法学，以系统地识别交通仿真需求的方式应对确保现实主义交通条件的挑战。该方法学基于每个研究阶段的子目标采用结构化方法，为微观层次、代理模型和可视化表示提取具体的技术需求。该方法学旨在保持高度的保真度，提高实验结果的有效性和参与者 engagement。通过清晰地连接研究目标与交通仿真的设计，该方法支持 robust 的 automotive 开发与测试。 

---
# Purifying Task Vectors in Knowledge-Aware Subspace for Model Merging 

**Title (ZH)**: 知识感知子空间中任务向量的净化与模型融合 

**Authors**: Bang An, Yibo Yang, Philip Torr, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2510.14697)  

**Abstract**: Model merging aims to integrate task-specific abilities from individually fine-tuned models into a single model without extra training. In recent model merging methods, task vector has become a fundamental building block, as it can encapsulate the residual information from finetuning. However, the merged model often suffers from notable performance degradation due to the conflicts caused by task-irrelevant redundancy in task vectors. Existing efforts in overcoming redundancy by randomly dropping elements in the parameter space involves randomness and lacks knowledge awareness. To address these challenges, in this study, we propose Purifying TAsk Vectors (PAVE) in knowledge-aware subspace. Concretely, we sample some training examples from each task, and feed them into their corresponding fine-tuned models to acquire the covariance matrices before linear layers. We then perform a context-oriented singular value decomposition, which accentuates the weight components most relevant to the target knowledge. As a result, we can split fine-tuned model weights into task-relevant and redundant components in the knowledge-aware subspace, and purify the task vector by pruning the redundant components. To induce fair pruning efforts across models, we further introduce a spectral rank allocation strategy by optimizing a normalized activated pruning error. The task vector purification by our method as a plug-and-play scheme is applicable across various task vector-based merging methods to improve their performance. In experiments, we demonstrate the effectiveness of PAVE across a diverse set of merging methods, tasks, and model architectures. 

**Abstract (ZH)**: 知识导向子空间中净化任务向量（PAVE）模型融合 

---
# Practical, Utilitarian Algorithm Configuration 

**Title (ZH)**: 实用型实用算法配置 

**Authors**: Devon Graham, Kevin Leyton-Brown  

**Link**: [PDF](https://arxiv.org/pdf/2510.14683)  

**Abstract**: Utilitarian algorithm configuration identifies a parameter setting for a given algorithm that maximizes a user's utility. Utility functions offer a theoretically well-grounded approach to optimizing decision-making under uncertainty and are flexible enough to capture a user's preferences over algorithm runtimes (e.g., they can describe a sharp cutoff after which a solution is no longer required, a per-hour cost for compute, or diminishing returns from algorithms that take longer to run). COUP is a recently-introduced utilitarian algorithm configuration procedure which was designed mainly to offer strong theoretical guarantees about the quality of the configuration it returns, with less attention paid to its practical performance. This paper closes that gap, bringing theoretically-grounded, utilitarian algorithm configuration to the point where it is competitive with widely used, heuristic configuration procedures that offer no performance guarantees. We present a series of improvements to COUP that improve its empirical performance without degrading its theoretical guarantees and demonstrate their benefit experimentally. Using a case study, we also illustrate ways of exploring the robustness of a given solution to the algorithm selection problem to variations in the utility function. 

**Abstract (ZH)**: 效用算法配置确定给定算法的一个参数设置，以最大化用户的效用。效用函数提供了一种理论上扎实的方法来优化不确定性下的决策，并且足够灵活以捕捉用户对算法运行时间的偏好（例如，它们可以描述一个清晰的截止点，在此之后不再需要解决方案，每小时的计算成本，或者随运行时间增加而递减的回报）。COUP 是最近引入的一种效用算法配置程序，主要旨在提供关于返回的配置质量的强大理论保证，而在其实用性能方面则关注较少。本文填补了这一空白，使理论扎实的效用算法配置接近于广泛使用的、没有性能保证的启发式配置程序。我们提出了一系列改进 COUP 的方法，这些改进提高了其 empirical 性能而不牺牲其理论保证，并通过实验证明了这些改进的好处。使用案例研究，我们还展示了如何探索给定的算法选择问题解决方案对效用函数变化的鲁棒性。 

---
# NAEL: Non-Anthropocentric Ethical Logic 

**Title (ZH)**: NAEL: 非人性中心的伦理逻辑 

**Authors**: Bianca Maria Lerma, Rafael Peñaloza  

**Link**: [PDF](https://arxiv.org/pdf/2510.14676)  

**Abstract**: We introduce NAEL (Non-Anthropocentric Ethical Logic), a novel ethical framework for artificial agents grounded in active inference and symbolic reasoning. Departing from conventional, human-centred approaches to AI ethics, NAEL formalizes ethical behaviour as an emergent property of intelligent systems minimizing global expected free energy in dynamic, multi-agent environments. We propose a neuro-symbolic architecture to allow agents to evaluate the ethical consequences of their actions in uncertain settings. The proposed system addresses the limitations of existing ethical models by allowing agents to develop context-sensitive, adaptive, and relational ethical behaviour without presupposing anthropomorphic moral intuitions. A case study involving ethical resource distribution illustrates NAEL's dynamic balancing of self-preservation, epistemic learning, and collective welfare. 

**Abstract (ZH)**: NAEL（非anthropocentric伦理逻辑）：一种基于主动推断和符号推理的新型伦理框架 

---
# TITAN: Graph-Executable Reasoning for Cyber Threat Intelligence 

**Title (ZH)**: TITAN: 基于图执行推理的网络威胁情报分析 

**Authors**: Marco Simoni, Aleksandar Fontana, Andrea Saracino, Paolo Mori  

**Link**: [PDF](https://arxiv.org/pdf/2510.14670)  

**Abstract**: TITAN (Threat Intelligence Through Automated Navigation) is a framework that connects natural-language cyber threat queries with executable reasoning over a structured knowledge graph. It integrates a path planner model, which predicts logical relation chains from text, and a graph executor that traverses the TITAN Ontology to retrieve factual answers and supporting evidence. Unlike traditional retrieval systems, TITAN operates on a typed, bidirectional graph derived from MITRE, allowing reasoning to move clearly and reversibly between threats, behaviors, and defenses. To support training and evaluation, we introduce the TITAN Dataset, a corpus of 88209 examples (Train: 74258; Test: 13951) pairing natural language questions with executable reasoning paths and step by step Chain of Thought explanations. Empirical evaluations show that TITAN enables models to generate syntactically valid and semantically coherent reasoning paths that can be deterministically executed on the underlying graph. 

**Abstract (ZH)**: TITAN（威胁情报通过自动导航）是将自然语言网络威胁查询与结构化知识图谱上的可执行推理连接起来的框架。 

---
# Machine Learning and Public Health: Identifying and Mitigating Algorithmic Bias through a Systematic Review 

**Title (ZH)**: 机器学习与公共卫生：通过系统评价识别和缓解算法偏见 

**Authors**: Sara Altamirano, Arjan Vreeken, Sennay Ghebreab  

**Link**: [PDF](https://arxiv.org/pdf/2510.14669)  

**Abstract**: Machine learning (ML) promises to revolutionize public health through improved surveillance, risk stratification, and resource allocation. However, without systematic attention to algorithmic bias, ML may inadvertently reinforce existing health disparities. We present a systematic literature review of algorithmic bias identification, discussion, and reporting in Dutch public health ML research from 2021 to 2025. To this end, we developed the Risk of Algorithmic Bias Assessment Tool (RABAT) by integrating elements from established frameworks (Cochrane Risk of Bias, PROBAST, Microsoft Responsible AI checklist) and applied it to 35 peer-reviewed studies. Our analysis reveals pervasive gaps: although data sampling and missing data practices are well documented, most studies omit explicit fairness framing, subgroup analyses, and transparent discussion of potential harms. In response, we introduce a four-stage fairness-oriented framework called ACAR (Awareness, Conceptualization, Application, Reporting), with guiding questions derived from our systematic literature review to help researchers address fairness across the ML lifecycle. We conclude with actionable recommendations for public health ML practitioners to consistently consider algorithmic bias and foster transparency, ensuring that algorithmic innovations advance health equity rather than undermine it. 

**Abstract (ZH)**: 机器学习（ML）有望通过改进监控、风险分层和资源分配来革新公共卫生。然而，如果没有系统地关注算法偏见，ML可能无意中加剧现有的健康不平等。我们对2021年至2025年间荷兰公共卫生机器学习研究中的算法偏见识别、讨论和报告进行了系统文献回顾。为此，我们开发了算法偏见风险评估工具（RABAT），并将现有框架（Cochrane偏倚风险、PROBAST、微软负责任AI检查表）中的元素进行整合，并应用于35项同行评审研究。我们的分析揭示普遍存在空白：尽管数据采样和缺失数据处理方法已详细记录，但大多数研究未明确包含公平性框架、亚组分析以及潜在危害的透明讨论。为此，我们引入了一个四阶段公平导向框架（ACAR，意识、概念化、应用、报告），并从系统文献回顾中衍生出指导问题，以帮助研究人员在机器学习生命周期中全面考虑公平性。我们最终提出可操作的建议，指导公共卫生机器学习从业人员始终考虑算法偏见，促进透明度，确保算法创新促进卫生公平而非削弱它。 

---
# ColorBench: Benchmarking Mobile Agents with Graph-Structured Framework for Complex Long-Horizon Tasks 

**Title (ZH)**: ColorBench：基于图结构框架评估移动代理复杂长时间任务能力 

**Authors**: Yuanyi Song, Heyuan Huang, Qiqiang Lin, Yin Zhao, Xiangmou Qu, Jun Wang, Xingyu Lou, Weiwen Liu, Zhuosheng Zhang, Jun Wang, Yong Yu, Weinan Zhang, Zhaoxiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14621)  

**Abstract**: The rapid advancement of multimodal large language models has enabled agents to operate mobile devices by directly interacting with graphical user interfaces, opening new possibilities for mobile automation. However, real-world mobile tasks are often complex and allow for multiple valid solutions. This contradicts current mobile agent evaluation standards: offline static benchmarks can only validate a single predefined "golden path", while online dynamic testing is constrained by the complexity and non-reproducibility of real devices, making both approaches inadequate for comprehensively assessing agent capabilities. To bridge the gap between offline and online evaluation and enhance testing stability, this paper introduces a novel graph-structured benchmarking framework. By modeling the finite states observed during real-device interactions, it achieves static simulation of dynamic behaviors. Building on this, we develop ColorBench, a benchmark focused on complex long-horizon tasks. It supports evaluation of multiple valid solutions, subtask completion rate statistics, and atomic-level capability analysis. ColorBench contains 175 tasks (74 single-app, 101 cross-app) with an average length of over 13 steps. Each task includes at least two correct paths and several typical error paths, enabling quasi-dynamic interaction. By evaluating ColorBench across various baselines, we discover limitations of existing models and propose improvement directions and feasible technical pathways to enhance agents' performance on complex, long-horizon problems based on experimental results. Code and data are available at: this https URL. 

**Abstract (ZH)**: 多模态大型语言模型的快速进展使得代理能够通过直接与图形用户界面交互来操作移动设备，为移动自动化开启了新的可能性。然而，现实中的移动任务往往是复杂的，并允许多个有效的解决方案。这与当前的移动代理评估标准相矛盾：离线静态基准只能验证一个预定义的“黄金路径”，而在线动态测试受限于真实设备的复杂性和不可再现性，使这两种方法都无法全面评估代理的能力。为了弥合离线和在线评估之间的差距并增强测试稳定性，本文引入了一个新颖的图结构基准框架。通过建模实际设备交互中观察到的有限状态，实现了动态行为的静态仿真。在此基础上，我们开发了ColorBench，一个侧重于复杂长时任务的基准。它支持评估多个有效解决方案、子任务完成率统计和原子级能力分析。ColorBench 包含 175 个任务（74 个单应用，101 个跨应用），平均长度超过 13 步。每个任务至少包含两条正确的路径和几个典型的错误路径，实现准动态交互。通过在各种基线上评估 ColorBench，我们发现了现有模型的局限性，并基于实验结果提出改进方向和技术路径，以提高代理在处理复杂长时问题时的表现。代码和数据可在以下网址获取：this https URL。 

---
# Symbol Grounding in Neuro-Symbolic AI: A Gentle Introduction to Reasoning Shortcuts 

**Title (ZH)**: 神经符号AI中的符号 grounding：推理快捷方式的温和介绍 

**Authors**: Emanuele Marconato, Samuele Bortolotti, Emile van Krieken, Paolo Morettin, Elena Umili, Antonio Vergari, Efthymia Tsamoura, Andrea Passerini, Stefano Teso  

**Link**: [PDF](https://arxiv.org/pdf/2510.14538)  

**Abstract**: Neuro-symbolic (NeSy) AI aims to develop deep neural networks whose predictions comply with prior knowledge encoding, e.g. safety or structural constraints. As such, it represents one of the most promising avenues for reliable and trustworthy AI. The core idea behind NeSy AI is to combine neural and symbolic steps: neural networks are typically responsible for mapping low-level inputs into high-level symbolic concepts, while symbolic reasoning infers predictions compatible with the extracted concepts and the prior knowledge. Despite their promise, it was recently shown that - whenever the concepts are not supervised directly - NeSy models can be affected by Reasoning Shortcuts (RSs). That is, they can achieve high label accuracy by grounding the concepts incorrectly. RSs can compromise the interpretability of the model's explanations, performance in out-of-distribution scenarios, and therefore reliability. At the same time, RSs are difficult to detect and prevent unless concept supervision is available, which is typically not the case. However, the literature on RSs is scattered, making it difficult for researchers and practitioners to understand and tackle this challenging problem. This overview addresses this issue by providing a gentle introduction to RSs, discussing their causes and consequences in intuitive terms. It also reviews and elucidates existing theoretical characterizations of this phenomenon. Finally, it details methods for dealing with RSs, including mitigation and awareness strategies, and maps their benefits and limitations. By reformulating advanced material in a digestible form, this overview aims to provide a unifying perspective on RSs to lower the bar to entry for tackling them. Ultimately, we hope this overview contributes to the development of reliable NeSy and trustworthy AI models. 

**Abstract (ZH)**: 神经符号（NeSy）人工智能旨在开发遵循先验知识（例如安全性或结构约束）的深度神经网络。因此，它代表了可靠和可信赖人工智能的一个最有前途的方向。NeSy人工智能的核心思想是结合神经和符号步骤：神经网络通常负责将低级输入映射为高级符号概念，而符号推理则根据提取的概念和先验知识推断兼容的预测。尽管它们很有前景，但最近的研究显示，当概念未直接监督时，NeSy模型可能会受到推理捷径（RSs）的影响。也就是说，它们可以通过错误地链接概念来实现高标签准确性。RSs可以削弱模型解释的可解释性，影响异常分布场景中的性能，从而影响可靠性。同时，除非有概念监督，否则检测和防止RSs是困难的，而这种情况通常不会发生。然而，关于RSs的文献比较分散，这使得研究人员和实践者难以理解并解决这一具有挑战性的问题。本文综述通过提供一种直观的介绍RSs、讨论其成因和后果，并回顾和阐明现有的理论刻画，最后详细阐述解决RSs的方法，包括缓解和意识策略及其优缺点，旨在以一种易于理解的形式重新表述高级材料，为解决RSs提供统一的视角，降低解决它们的门槛。最终，我们希望本文综述能够促进可靠NeSy和可信赖人工智能模型的发展。 

---
# JSPLIT: A Taxonomy-based Solution for Prompt Bloating in Model Context Protocol 

**Title (ZH)**: JSPLIT：基于分类学的模型上下文协议中提示膨胀解决方案 

**Authors**: Emanuele Antonioni, Stefan Markovic, Anirudha Shankar, Jaime Bernardo, Lovro Markovic, Silvia Pareti, Benedetto Proietti  

**Link**: [PDF](https://arxiv.org/pdf/2510.14537)  

**Abstract**: AI systems are continually evolving and advancing, and user expectations are concurrently increasing, with a growing demand for interactions that go beyond simple text-based interaction with Large Language Models (LLMs). Today's applications often require LLMs to interact with external tools, marking a shift toward more complex agentic systems. To support this, standards such as the Model Context Protocol (MCP) have emerged, enabling agents to access tools by including a specification of the capabilities of each tool within the prompt. Although this approach expands what agents can do, it also introduces a growing problem: prompt bloating. As the number of tools increases, the prompts become longer, leading to high prompt token costs, increased latency, and reduced task success resulting from the selection of tools irrelevant to the prompt. To address this issue, we introduce JSPLIT, a taxonomy-driven framework designed to help agents manage prompt size more effectively when using large sets of MCP tools. JSPLIT organizes the tools into a hierarchical taxonomy and uses the user's prompt to identify and include only the most relevant tools, based on both the query and the taxonomy structure. In this paper, we describe the design of the taxonomy, the tool selection algorithm, and the dataset used to evaluate JSPLIT. Our results show that JSPLIT significantly reduces prompt size without significantly compromising the agent's ability to respond effectively. As the number of available tools for the agent grows substantially, JSPLIT even improves the tool selection accuracy of the agent, effectively reducing costs while simultaneously improving task success in high-complexity agent environments. 

**Abstract (ZH)**: 基于-taxonomy的JSPLIT框架：有效管理大规模MCP工具的提示大小 

---
# Helmsman: Autonomous Synthesis of Federated Learning Systems via Multi-Agent Collaboration 

**Title (ZH)**: Helmsman: 通过多智能体协作自主合成联邦学习系统 

**Authors**: Haoyuan Li, Mathias Funk, Aaqib Saeed  

**Link**: [PDF](https://arxiv.org/pdf/2510.14512)  

**Abstract**: Federated Learning (FL) offers a powerful paradigm for training models on decentralized data, but its promise is often undermined by the immense complexity of designing and deploying robust systems. The need to select, combine, and tune strategies for multifaceted challenges like data heterogeneity and system constraints has become a critical bottleneck, resulting in brittle, bespoke solutions. To address this, we introduce Helmsman, a novel multi-agent system that automates the end-to-end synthesis of federated learning systems from high-level user specifications. It emulates a principled research and development workflow through three collaborative phases: (1) interactive human-in-the-loop planning to formulate a sound research plan, (2) modular code generation by supervised agent teams, and (3) a closed-loop of autonomous evaluation and refinement in a sandboxed simulation environment. To facilitate rigorous evaluation, we also introduce AgentFL-Bench, a new benchmark comprising 16 diverse tasks designed to assess the system-level generation capabilities of agentic systems in FL. Extensive experiments demonstrate that our approach generates solutions competitive with, and often superior to, established hand-crafted baselines. Our work represents a significant step towards the automated engineering of complex decentralized AI systems. 

**Abstract (ZH)**: 联邦学习(Federated Learning)提供了一种强大的范式来在分布式数据上训练模型，但其潜力常因设计和部署稳健系统所面临的巨大复杂性而受挫。对数据异质性和系统约束等多方面挑战的策略选择、组合与调整已经成为关键瓶颈，导致了脆弱且定制的解决方案。为解决这一问题，我们引入了Helmsman，这是一种新颖的多代理系统，能够从高层次用户规范自动合成端到端的联邦学习系统。它通过三个协作阶段模仿了规范的研究和开发工作流程：(1) 交互式的人机环规划以制定严谨的研究计划，(2) 监督代理团队的模块化代码生成，以及(3) 沙箱仿真环境中的自主评价和优化闭环。为了促进严格的评估，我们还引入了AgentFL-Bench，这是一种包含16项不同任务的新基准，旨在评估代理系统在联邦学习中的系统级生成能力。全面的实验展示了我们方法生成的竞争性和优越性解决方案，通常优于现有的手工构建基准。我们的工作代表了自动工程复杂分布式AI系统的重要一步。 

---
# Eliminating Negative Occurrences of Derived Predicates from PDDL Axioms 

**Title (ZH)**: 从PDDL公理中消除派生谓词的负面出现 

**Authors**: Claudia Grundke, Gabriele Röger  

**Link**: [PDF](https://arxiv.org/pdf/2510.14412)  

**Abstract**: Axioms are a feature of the Planning Domain Definition Language PDDL that can be considered as a generalization of database query languages such as Datalog. The PDDL standard restricts negative occurrences of predicates in axiom bodies to predicates that are directly set by actions and not derived by axioms. In the literature, authors often deviate from this limitation and only require that the set of axioms is stratifiable. Both variants can express exactly the same queries as least fixed-point logic, indicating that negative occurrences of derived predicates can be eliminated. We present the corresponding transformation. 

**Abstract (ZH)**: axioms是Planning Domain Definition Language PDDL的一个特征，可以视为数据库查询语言如Datalog的一般化。PDDL标准限制了在公理体内出现的负谓词仅限于由动作直接设置的谓词，而非由公理推导出的谓词。文献中，作者通常偏离这一限制，只需确保公理集是可分层的。这两种变体都可以精确地表达最少固定点逻辑所能表达的所有查询，表明可以消除导出谓词的负出现象。我们提出了相应的转换。 

---
# Metacognitive Self-Correction for Multi-Agent System via Prototype-Guided Next-Execution Reconstruction 

**Title (ZH)**: 基于原型引导的下一次执行重建的元认知自我修正多agent系统 

**Authors**: Xu Shen, Qi Zhang, Song Wang, Zhen Tan, Xinyu Zhao, Laura Yao, Vaishnav Tadiparthi, Hossein Nourkhiz Mahjoub, Ehsan Moradi Pari, Kwonjoon Lee, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.14319)  

**Abstract**: Large Language Model based multi-agent systems (MAS) excel at collaborative problem solving but remain brittle to cascading errors: a single faulty step can propagate across agents and disrupt the trajectory. In this paper, we present MASC, a metacognitive framework that endows MAS with real-time, unsupervised, step-level error detection and self-correction. MASC rethinks detection as history-conditioned anomaly scoring via two complementary designs: (1) Next-Execution Reconstruction, which predicts the embedding of the next step from the query and interaction history to capture causal consistency, and (2) Prototype-Guided Enhancement, which learns a prototype prior over normal-step embeddings and uses it to stabilize reconstruction and anomaly scoring under sparse context (e.g., early steps). When an anomaly step is flagged, MASC triggers a correction agent to revise the acting agent's output before information flows downstream. On the Who&When benchmark, MASC consistently outperforms all baselines, improving step-level error detection by up to 8.47% AUC-ROC ; When plugged into diverse MAS frameworks, it delivers consistent end-to-end gains across architectures, confirming that our metacognitive monitoring and targeted correction can mitigate error propagation with minimal overhead. 

**Abstract (ZH)**: 基于大型语言模型的多智能体系统中的元认知框架：实时无监督的步骤级错误检测与自修正 

---
# MorphoBench: A Benchmark with Difficulty Adaptive to Model Reasoning 

**Title (ZH)**: MorphoBench: 一种适应模型推理难度的基准评测 

**Authors**: Xukai Wang, Xuanbo Liu, Mingrui Chen, Haitian Zhong, Xuanlin Yang, Bohan Zeng, Jinbo Hu, Hao Liang, Junbo Niu, Xuchen Li, Ruitao Wu, Ruichuan An, Yang Shi, Liu Liu, Xu-Yao Zhang, Qiang Liu, Zhouchen Lin, Wentao Zhang, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2510.14265)  

**Abstract**: With the advancement of powerful large-scale reasoning models, effectively evaluating the reasoning capabilities of these models has become increasingly important. However, existing benchmarks designed to assess the reasoning abilities of large models tend to be limited in scope and lack the flexibility to adapt their difficulty according to the evolving reasoning capacities of the models. To address this, we propose MorphoBench, a benchmark that incorporates multidisciplinary questions to evaluate the reasoning capabilities of large models and can adjust and update question difficulty based on the reasoning abilities of advanced models. Specifically, we curate the benchmark by selecting and collecting complex reasoning questions from existing benchmarks and sources such as Olympiad-level competitions. Additionally, MorphoBench adaptively modifies the analytical challenge of questions by leveraging key statements generated during the model's reasoning process. Furthermore, it includes questions generated using simulation software, enabling dynamic adjustment of benchmark difficulty with minimal resource consumption. We have gathered over 1,300 test questions and iteratively adjusted the difficulty of MorphoBench based on the reasoning capabilities of models such as o3 and GPT-5. MorphoBench enhances the comprehensiveness and validity of model reasoning evaluation, providing reliable guidance for improving both the reasoning abilities and scientific robustness of large models. The code has been released in this https URL. 

**Abstract (ZH)**: 随着强大大规模推理模型的发展，有效评估这些模型的推理能力变得越来越重要。然而，现有的用于评估大模型推理能力的基准测试往往范围有限，缺乏根据模型推理能力的发展调整难度的灵活性。为了解决这一问题，我们提出了MorphoBench这一基准测试，该测试结合了跨学科的问题来评估大模型的推理能力，并可根据高级模型的推理能力调整和更新问题难度。具体来说，我们通过从现有基准测试和如奥林匹克级别的竞赛中精选和收集复杂推理问题来编制基准测试。此外，MorphoBench通过利用模型推理过程中生成的关键语句，动态调整问题的分析挑战。此外，它还包含使用模拟软件生成的问题，能够在极低的资源消耗下动态调整基准测试难度。我们已经收集了超过1,300个测试问题，并根据如o3和GPT-5等模型的推理能力迭代调整了MorphoBench的难度。MorphoBench增强了模型推理评估的全面性和有效性，为提高大模型的推理能力和科学稳健性提供了可靠指导。代码已在此处发布：https://。 

---
# LiveResearchBench: A Live Benchmark for User-Centric Deep Research in the Wild 

**Title (ZH)**: LiveResearchBench: 一个面向用户的深度研究实时基准 

**Authors**: Jiayu Wang, Yifei Ming, Riya Dulepet, Qinglin Chen, Austin Xu, Zixuan Ke, Frederic Sala, Aws Albarghouthi, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2510.14240)  

**Abstract**: Deep research -- producing comprehensive, citation-grounded reports by searching and synthesizing information from hundreds of live web sources -- marks an important frontier for agentic systems. To rigorously evaluate this ability, four principles are essential: tasks should be (1) user-centric, reflecting realistic information needs, (2) dynamic, requiring up-to-date information beyond parametric knowledge, (3) unambiguous, ensuring consistent interpretation across users, and (4) multi-faceted and search-intensive, requiring search over numerous web sources and in-depth analysis. Existing benchmarks fall short of these principles, often focusing on narrow domains or posing ambiguous questions that hinder fair comparison. Guided by these principles, we introduce LiveResearchBench, a benchmark of 100 expert-curated tasks spanning daily life, enterprise, and academia, each requiring extensive, dynamic, real-time web search and synthesis. Built with over 1,500 hours of human labor, LiveResearchBench provides a rigorous basis for systematic evaluation. To evaluate citation-grounded long-form reports, we introduce DeepEval, a comprehensive suite covering both content- and report-level quality, including coverage, presentation, citation accuracy and association, consistency and depth of analysis. DeepEval integrates four complementary evaluation protocols, each designed to ensure stable assessment and high agreement with human judgments. Using LiveResearchBench and DeepEval, we conduct a comprehensive evaluation of 17 frontier deep research systems, including single-agent web search, single-agent deep research, and multi-agent systems. Our analysis reveals current strengths, recurring failure modes, and key system components needed to advance reliable, insightful deep research. 

**Abstract (ZH)**: 深度研究——通过搜索和综合来自数百个实时网络源的信息以生成全面、引文为基础的报告标志着代理系统的重大前沿。为了严谨地评估这一能力，四项原则是必不可少的：任务应（1）以用户为中心，反映现实的信息需求，（2）动态的，要求超出参数知识的最新信息，（3）明确的，确保用户之间的一致解释，以及（4）多维度和搜索密集型的，需要跨众多网络源进行搜索和深入分析。现有的基准测试未能满足这些原则，往往集中在狭窄的领域或提出了模糊的问题，这妨碍了公平的比较。根据这些原则，我们提出了LiveResearchBench，这是一个包含100项专家选择的任务基准，覆盖日常生活、企业与学术界，每项任务都需要广泛的动态实时网络搜索与综合。LiveResearchBench在超过1,500小时的人工努力下构建，为系统评估奠定了坚实的基础。为评估引文为基础的长篇报告，我们引入了DeepEval，这是一个全面的评估套件，涵盖了内容级和报告级质量，包括覆盖面、呈现方式、引文的准确性与关联性、一致性和分析的深度。DeepEval结合了四个互补的评估协议，每个协议都旨在确保稳定的评估和与人类判断的高一致性。利用LiveResearchBench和DeepEval，我们对17个前沿深度研究系统进行了全面评估，包括单代理网络搜索、单代理深度研究以及多代理系统。我们的分析揭示了当前的优势、反复出现的失败模式以及推进可靠而深刻的深度研究所需的关键系统组件。 

---
# Implementation of AI in Precision Medicine 

**Title (ZH)**: AI在精准医学中的实施 

**Authors**: Göktuğ Bender, Samer Faraj, Anand Bhardwaj  

**Link**: [PDF](https://arxiv.org/pdf/2510.14194)  

**Abstract**: Artificial intelligence (AI) has become increasingly central to precision medicine by enabling the integration and interpretation of multimodal data, yet implementation in clinical settings remains limited. This paper provides a scoping review of literature from 2019-2024 on the implementation of AI in precision medicine, identifying key barriers and enablers across data quality, clinical reliability, workflow integration, and governance. Through an ecosystem-based framework, we highlight the interdependent relationships shaping real-world translation and propose future directions to support trustworthy and sustainable implementation. 

**Abstract (ZH)**: 人工智能在精准医疗中的实施：2019-2024年间文献综述及其关键障碍与促成因素分析 

---
# ARM-FM: Automated Reward Machines via Foundation Models for Compositional Reinforcement Learning 

**Title (ZH)**: ARM-FM：通过基础模型实现自动化的奖励机器，用于组合强化学习 

**Authors**: Roger Creus Castanyer, Faisal Mohamed, Pablo Samuel Castro, Cyrus Neary, Glen Berseth  

**Link**: [PDF](https://arxiv.org/pdf/2510.14176)  

**Abstract**: Reinforcement learning (RL) algorithms are highly sensitive to reward function specification, which remains a central challenge limiting their broad applicability. We present ARM-FM: Automated Reward Machines via Foundation Models, a framework for automated, compositional reward design in RL that leverages the high-level reasoning capabilities of foundation models (FMs). Reward machines (RMs) -- an automata-based formalism for reward specification -- are used as the mechanism for RL objective specification, and are automatically constructed via the use of FMs. The structured formalism of RMs yields effective task decompositions, while the use of FMs enables objective specifications in natural language. Concretely, we (i) use FMs to automatically generate RMs from natural language specifications; (ii) associate language embeddings with each RM automata-state to enable generalization across tasks; and (iii) provide empirical evidence of ARM-FM's effectiveness in a diverse suite of challenging environments, including evidence of zero-shot generalization. 

**Abstract (ZH)**: Automated Reward Machines via Foundation Models：基于基础模型的自动化奖赏机器框架 

---
# STEMS: Spatial-Temporal Enhanced Safe Multi-Agent Coordination for Building Energy Management 

**Title (ZH)**: STEMS：基于时空增强的安全多Agent协同管理建筑能效 

**Authors**: Huiliang Zhang, Di Wu, Arnaud Zinflou, Benoit Boulet  

**Link**: [PDF](https://arxiv.org/pdf/2510.14112)  

**Abstract**: Building energy management is essential for achieving carbon reduction goals, improving occupant comfort, and reducing energy costs. Coordinated building energy management faces critical challenges in exploiting spatial-temporal dependencies while ensuring operational safety across multi-building systems. Current multi-building energy systems face three key challenges: insufficient spatial-temporal information exploitation, lack of rigorous safety guarantees, and system complexity. This paper proposes Spatial-Temporal Enhanced Safe Multi-Agent Coordination (STEMS), a novel safety-constrained multi-agent reinforcement learning framework for coordinated building energy management. STEMS integrates two core components: (1) a spatial-temporal graph representation learning framework using a GCN-Transformer fusion architecture to capture inter-building relationships and temporal patterns, and (2) a safety-constrained multi-agent RL algorithm incorporating Control Barrier Functions to provide mathematical safety guarantees. Extensive experiments on real-world building datasets demonstrate STEMS's superior performance over existing methods, showing that STEMS achieves 21% cost reduction, 18% emission reduction, and dramatically reduces safety violations from 35.1% to 5.6% while maintaining optimal comfort with only 0.13 discomfort proportion. The framework also demonstrates strong robustness during extreme weather conditions and maintains effectiveness across different building types. 

**Abstract (ZH)**: 基于空间-时间增强的安全多代理协调（STEMS）在多建筑能效管理中的应用 

---
# Generating Fair Consensus Statements with Social Choice on Token-Level MDPs 

**Title (ZH)**: 基于代币级别MDP的社会选择生成公平共识声明 

**Authors**: Carter Blair, Kate Larson  

**Link**: [PDF](https://arxiv.org/pdf/2510.14106)  

**Abstract**: Current frameworks for consensus statement generation with large language models lack the inherent structure needed to provide provable fairness guarantees when aggregating diverse free-form opinions. We model the task as a multi-objective, token-level Markov Decision Process (MDP), where each objective corresponds to an agent's preference. Token-level rewards for each agent are derived from their policy (e.g., a personalized language model). This approach utilizes the finding that such policies implicitly define optimal Q-functions, providing a principled way to quantify rewards at each generation step without a value function (Rafailov et al., 2024). This MDP formulation creates a formal structure amenable to analysis using principles from social choice theory. We propose two approaches grounded in social choice theory. First, we propose a stochastic generation policy guaranteed to be in the ex-ante core, extending core stability concepts from voting theory to text generation. This policy is derived from an underlying distribution over complete statements that maximizes proportional fairness (Nash Welfare). Second, for generating a single statement, we target the maximization of egalitarian welfare using search algorithms within the MDP framework. Empirically, experiments using language models to instantiate agent policies show that search guided by the egalitarian objective generates consensus statements with improved worst-case agent alignment compared to baseline methods, including the Habermas Machine (Tessler et al., 2024). 

**Abstract (ZH)**: 基于大规模语言模型的共识声明生成框架缺乏聚合多样化自由形式意见时提供可证明公平性保证的固有结构。我们将该任务建模为一个多目标、token级马尔可夫决策过程（MDP），其中每个目标对应于代理的偏好。每个代理的token级奖励源自其策略（例如，个性化语言模型）。该方法利用了这样的发现：这些策略隐含地定义了最优Q函数，提供了一种在每个生成步骤中量化奖励的规范方式，无需价值函数（Rafailov等，2024）。这种MDP形式通过社会选择理论中的原理创建了可进行正式分析的结构。我们提出了两种基于社会选择理论的方法。首先，我们提出了一种随机生成策略，保证其在事前核心中，将投票理论中的核心稳定性概念扩展到文本生成中。该策略源自最大化比例公平（纳什福利）的完整声明的底层分布。其次，在生成单个声明时，我们利用MDP框架中的搜索算法最大化激进福利。实验结果显示，基于激进目标的搜索生成的共识声明在最坏情况下的代理对齐优于基线方法，包括Habermas Machine（Tessler等，2024）。 

---
# GammaZero: Learning To Guide POMDP Belief Space Search With Graph Representations 

**Title (ZH)**: GammaZero：学习引导POMDP信念空间搜索的图表示方法 

**Authors**: Rajesh Mangannavar, Prasad Tadepalli  

**Link**: [PDF](https://arxiv.org/pdf/2510.14035)  

**Abstract**: We introduce an action-centric graph representation framework for learning to guide planning in Partially Observable Markov Decision Processes (POMDPs). Unlike existing approaches that require domain-specific neural architectures and struggle with scalability, GammaZero leverages a unified graph-based belief representation that enables generalization across problem sizes within a domain. Our key insight is that belief states can be systematically transformed into action-centric graphs where structural patterns learned on small problems transfer to larger instances. We employ a graph neural network with a decoder architecture to learn value functions and policies from expert demonstrations on computationally tractable problems, then apply these learned heuristics to guide Monte Carlo tree search on larger problems. Experimental results on standard POMDP benchmarks demonstrate that GammaZero achieves comparable performance to BetaZero when trained and tested on the same-sized problems, while uniquely enabling zero-shot generalization to problems 2-4 times larger than those seen during training, maintaining solution quality with reduced search requirements. 

**Abstract (ZH)**: 一种用于部分可观测马尔可夫决策过程规划引导的学习动作中心图表示框架 

---
# Decision Oriented Technique (DOTechnique): Finding Model Validity Through Decision-Maker Context 

**Title (ZH)**: 决策导向技术（DOT技术）：通过决策者背景寻找模型有效性 

**Authors**: Raheleh Biglari, Joachim Denil  

**Link**: [PDF](https://arxiv.org/pdf/2510.13858)  

**Abstract**: Model validity is as critical as the model itself, especially when guiding decision-making processes. Traditional approaches often rely on predefined validity frames, which may not always be available or sufficient. This paper introduces the Decision Oriented Technique (DOTechnique), a novel method for determining model validity based on decision consistency rather than output similarity. By evaluating whether surrogate models lead to equivalent decisions compared to high-fidelity models, DOTechnique enables efficient identification of validity regions, even in the absence of explicit validity boundaries. The approach integrates domain constraints and symbolic reasoning to narrow the search space, enhancing computational efficiency. A highway lane change system serves as a motivating example, demonstrating how DOTechnique can uncover the validity region of a simulation model. The results highlight the potential of the technique to support finding model validity through decision-maker context. 

**Abstract (ZH)**: 模型的有效性与模型本身一样至关重要，尤其是在指导决策过程时。传统方法往往依赖于预定义的有效性框架，但这些框架可能并不总是可用或足够的。本文引入了旨在基于决策一致性而非输出相似性来确定模型有效性的决策导向技术（DOT技术）。通过评估代理模型的决策是否与高保真模型相当，DOT技术能够在没有明确有效边界的情况下有效地识别有效性区域。该方法结合领域约束和符号推理来缩小搜索空间，增强计算效率。高速公路变道系统作为一个示例，展示了如何使用DOT技术发现仿真模型的有效性区域。结果突显了该技术在通过决策者背景支持发现模型有效性方面的潜力。 

---
# Circuit Insights: Towards Interpretability Beyond Activations 

**Title (ZH)**: 电路洞察：超越激活函数的可解释性探索 

**Authors**: Elena Golimblevskaia, Aakriti Jain, Bruno Puri, Ammar Ibrahim, Wojciech Samek, Sebastian Lapuschkin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14936)  

**Abstract**: The fields of explainable AI and mechanistic interpretability aim to uncover the internal structure of neural networks, with circuit discovery as a central tool for understanding model computations. Existing approaches, however, rely on manual inspection and remain limited to toy tasks. Automated interpretability offers scalability by analyzing isolated features and their activations, but it often misses interactions between features and depends strongly on external LLMs and dataset quality. Transcoders have recently made it possible to separate feature attributions into input-dependent and input-invariant components, providing a foundation for more systematic circuit analysis. Building on this, we propose WeightLens and CircuitLens, two complementary methods that go beyond activation-based analysis. WeightLens interprets features directly from their learned weights, removing the need for explainer models or datasets while matching or exceeding the performance of existing methods on context-independent features. CircuitLens captures how feature activations arise from interactions between components, revealing circuit-level dynamics that activation-only approaches cannot identify. Together, these methods increase interpretability robustness and enhance scalable mechanistic analysis of circuits while maintaining efficiency and quality. 

**Abstract (ZH)**: 可解释AI和机制可解释性领域的研究旨在揭示神经网络的内部结构，电路发现是理解模型计算的重要工具。现有方法依赖手动检查且仅限于玩具任务。自动可解释性通过分析孤立特征及其激活来实现可扩展性，但往往忽略了特征之间的交互，并且强烈依赖于外部LLM和数据集质量。编码器 recently 使能够将特征归属分解为输入依赖和输入不变的组件，为更系统的电路分析奠定了基础。基于此，我们提出WeightLens和CircuitLens两种互补的方法，超越了基于激活的分析。WeightLens直接从学习的权重中解释特征，无需使用解释器模型或数据集，同时在独立于上下文的特征上与现有方法性能相当或更好。CircuitLens捕捉特征激活由组件间交互引起的方式，揭示了仅基于激活的方法无法识别的电路级动态。结合使用，这些方法提高了可解释性的鲁棒性，增强了电路的可扩展机制分析的效率和质量。 

---
# Detecting Early and Implicit Suicidal Ideation via Longitudinal and Information Environment Signals on Social Media 

**Title (ZH)**: 通过社交媒体的纵向和信息环境信号检测早期和隐匿的自杀意念 

**Authors**: Soorya Ram Shimgekar, Ruining Zhao, Agam Goyal, Violeta J. Rodriguez, Paul A. Bloom, Hari Sundaram, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2510.14889)  

**Abstract**: On social media, many individuals experiencing suicidal ideation (SI) do not disclose their distress explicitly. Instead, signs may surface indirectly through everyday posts or peer interactions. Detecting such implicit signals early is critical but remains challenging. We frame early and implicit SI as a forward-looking prediction task and develop a computational framework that models a user's information environment, consisting of both their longitudinal posting histories as well as the discourse of their socially proximal peers. We adopted a composite network centrality measure to identify top neighbors of a user, and temporally aligned the user's and neighbors' interactions -- integrating the multi-layered signals in a fine-tuned DeBERTa-v3 model. In a Reddit study of 1,000 (500 Case and 500 Control) users, our approach improves early and implicit SI detection by 15% over individual-only baselines. These findings highlight that peer interactions offer valuable predictive signals and carry broader implications for designing early detection systems that capture indirect as well as masked expressions of risk in online environments. 

**Abstract (ZH)**: 在社交媒体上，很多经历自杀意念的个体不会明确披露其痛苦。相反，这些信号可能通过日常发布的帖子或同伴互动间接表现出来。及早检测这些隐性的信号至关重要但极具挑战性。我们将早期和隐性的自杀意念视为一种前瞻预测任务，并开发了一种计算框架，该框架模型了用户的信息环境，包括用户的纵向发帖历史及其社交邻近同伴的言论。我们采用了综合网络中心性度量来识别用户的重要邻居，并暂态对齐用户及其邻居的互动——在微调的DeBERTa-v3模型中整合了多层次的信号。在针对1,000名（500名病例和500名对照）Reddit用户的研究中，我们的方法相较于仅基于个体的基线提高了15%的早期和隐性自杀意念检测率。这些发现强调了同伴互动提供的宝贵预测信号，并具有更广泛的含义，即设计能在网络环境中捕捉隐性及掩饰的风险表达的早期检测系统。 

---
# Learning When Not to Learn: Risk-Sensitive Abstention in Bandits with Unbounded Rewards 

**Title (ZH)**: 学习在何时不学习：带无界奖励的Bandits中的风险敏感型回避 

**Authors**: Sarah Liaw, Benjamin Plaut  

**Link**: [PDF](https://arxiv.org/pdf/2510.14884)  

**Abstract**: In high-stakes AI applications, even a single action can cause irreparable damage. However, nearly all of sequential decision-making theory assumes that all errors are recoverable (e.g., by bounding rewards). Standard bandit algorithms that explore aggressively may cause irreparable damage when this assumption fails. Some prior work avoids irreparable errors by asking for help from a mentor, but a mentor may not always be available. In this work, we formalize a model of learning with unbounded rewards without a mentor as a two-action contextual bandit with an abstain option: at each round the agent observes an input and chooses either to abstain (always 0 reward) or to commit (execute a preexisting task policy). Committing yields rewards that are upper-bounded but can be arbitrarily negative, and the commit reward is assumed Lipschitz in the input. We propose a caution-based algorithm that learns when not to learn: it chooses a trusted region and commits only where the available evidence does not already certify harm. Under these conditions and i.i.d. inputs, we establish sublinear regret guarantees, theoretically demonstrating the effectiveness of cautious exploration for deploying learning agents safely in high-stakes environments. 

**Abstract (ZH)**: 在高风险AI应用中，即使单个行动也可能造成不可逆的损害。然而，几乎所有序贯决策理论都假设所有错误都是可恢复的（例如，通过限制奖励）。标准的探索性很强的多臂老虎机算法在这一假设不成立时可能会造成不可逆的损害。一些先前的工作通过寻求导师的帮助来避免不可逆的错误，但导师可能并不总是可用的。在这项工作中，我们以两行动上下文多臂老虎机模型的形式形式化了无导师的无界奖励学习模型，该模型附带弃权选项：在每一轮中，代理观察输入并选择弃权（总是0奖励）或投入（执行预存的任务策略）。投入产生上界受限但可以任意负的奖励，且投入奖励假设为输入的Lipschitz函数。我们提出了一种基于谨慎性的算法，学习何时不应学习：它选择一个可信赖的区域，并仅在现有证据未认证有危害时才投入。在这些条件下和独立同分布的输入下，我们建立了子线性后悔保证，从理论上证明了谨慎探索在高风险环境中安全部署学习代理的有效性。 

---
# Predicting kernel regression learning curves from only raw data statistics 

**Title (ZH)**: 仅从原始数据统计预测核回归学习曲线 

**Authors**: Dhruva Karkada, Joseph Turnbull, Yuxi Liu, James B. Simon  

**Link**: [PDF](https://arxiv.org/pdf/2510.14878)  

**Abstract**: We study kernel regression with common rotation-invariant kernels on real datasets including CIFAR-5m, SVHN, and ImageNet. We give a theoretical framework that predicts learning curves (test risk vs. sample size) from only two measurements: the empirical data covariance matrix and an empirical polynomial decomposition of the target function $f_*$. The key new idea is an analytical approximation of a kernel's eigenvalues and eigenfunctions with respect to an anisotropic data distribution. The eigenfunctions resemble Hermite polynomials of the data, so we call this approximation the Hermite eigenstructure ansatz (HEA). We prove the HEA for Gaussian data, but we find that real image data is often "Gaussian enough" for the HEA to hold well in practice, enabling us to predict learning curves by applying prior results relating kernel eigenstructure to test risk. Extending beyond kernel regression, we empirically find that MLPs in the feature-learning regime learn Hermite polynomials in the order predicted by the HEA. Our HEA framework is a proof of concept that an end-to-end theory of learning which maps dataset structure all the way to model performance is possible for nontrivial learning algorithms on real datasets. 

**Abstract (ZH)**: 我们研究了在CIFAR-5m、SVHN和ImageNet等真实数据集上使用共同的旋转不变核的核回归。我们提供了一个理论框架，仅从经验数据协方差矩阵和目标函数$f_*$的经验多项式分解中预测学习曲线（测试风险与样本大小的关系）。关键新想法是核在各向异性数据分布下的特征值和特征函数的分析近似。特征函数类似于数据的赫mite多项式，因此我们将这种近似称为赫mite特征结构假设（HEA）。我们证明了HEA在高斯数据上成立，但在实际图像数据上，HEA在实践中常常很好地成立，使我们能够通过将核特征结构与测试风险的关系应用到以前的结果来预测学习曲线。超越核回归，我们实验证明多层感知机在特征学习阶段学习的赫mite多项式顺序由HEA预测。我们的HEA框架证明了对于真实数据集上非平凡的学习算法，从数据集结构直接映射到模型性能的端到端理论是可能的。 

---
# Scaling Artificial Intelligence for Multi-Tumor Early Detection with More Reports, Fewer Masks 

**Title (ZH)**: 多肿瘤早期检测中通过增加报告数量减少掩膜数量以 Scaling Artificial Intelligence 的方式 

**Authors**: Pedro R. A. S. Bassi, Xinze Zhou, Wenxuan Li, Szymon Płotka, Jieneng Chen, Qi Chen, Zheren Zhu, Jakub Prządo, Ibrahim E. Hamacı, Sezgin Er, Yuhan Wang, Ashwin Kumar, Bjoern Menze, Jarosław B. Ćwikła, Yuyin Zhou, Akshay S. Chaudhari, Curtis P. Langlotz, Sergio Decherchi, Andrea Cavalli, Kang Wang, Yang Yang, Alan L. Yuille, Zongwei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.14803)  

**Abstract**: Early tumor detection save lives. Each year, more than 300 million computed tomography (CT) scans are performed worldwide, offering a vast opportunity for effective cancer screening. However, detecting small or early-stage tumors on these CT scans remains challenging, even for experts. Artificial intelligence (AI) models can assist by highlighting suspicious regions, but training such models typically requires extensive tumor masks--detailed, voxel-wise outlines of tumors manually drawn by radiologists. Drawing these masks is costly, requiring years of effort and millions of dollars. In contrast, nearly every CT scan in clinical practice is already accompanied by medical reports describing the tumor's size, number, appearance, and sometimes, pathology results--information that is rich, abundant, and often underutilized for AI training. We introduce R-Super, which trains AI to segment tumors that match their descriptions in medical reports. This approach scales AI training with large collections of readily available medical reports, substantially reducing the need for manually drawn tumor masks. When trained on 101,654 reports, AI models achieved performance comparable to those trained on 723 masks. Combining reports and masks further improved sensitivity by +13% and specificity by +8%, surpassing radiologists in detecting five of the seven tumor types. Notably, R-Super enabled segmentation of tumors in the spleen, gallbladder, prostate, bladder, uterus, and esophagus, for which no public masks or AI models previously existed. This study challenges the long-held belief that large-scale, labor-intensive tumor mask creation is indispensable, establishing a scalable and accessible path toward early detection across diverse tumor types.
We plan to release our trained models, code, and dataset at this https URL 

**Abstract (ZH)**: 早期肿瘤检测拯救生命。每年，全球范围内进行的CT扫描超过3亿次，提供了有效癌症筛查的巨大机会。然而，即使是专家也难以在这些CT扫描中检测到小型或早期阶段的肿瘤。人工智能（AI）模型可以通过突出显示可疑区域来提供帮助，但训练这些模型通常需要大量的人工绘制的肿瘤掩膜——由放射ologist手工绘制的详细体素级肿瘤轮廓。绘制这些掩膜成本高昂，需要数年时间和数百万美元。相比之下，临床实践中几乎每一张CT扫描都已经附带了描述肿瘤大小、数量、外观以及有时病理结果的医学报告——这些信息丰富且充足，但通常未被用于AI训练。我们介绍了R-Super，该方法训练AI将肿瘤与其医学报告中的描述进行匹配分隔。这种方法利用大量现成的医学报告来扩展AI训练规模，大幅减少了手动绘制肿瘤掩膜的需求。当使用101,654份报告进行训练时，AI模型的性能与使用723张掩膜进行训练的模型相当。结合使用报告和掩膜进一步提高了敏感性+13%和特异性+8%，超越了放射学家在检测七种肿瘤类型中的五种时的表现。值得一提的是，R-Super使得在脾脏、胆囊、前列腺、膀胱、子宫和食道等部位的肿瘤分割成为可能，而对于这些部位，之前并未存在公共掩膜或AI模型。这项研究挑战了大规模、劳动密集型肿瘤掩膜创建不可或缺的传统观念，为各类肿瘤类型的早期检测提供了一条可扩展且易获取的道路。 

---
# Morphology-Aware Prognostic model for Five-Year Survival Prediction in Colorectal Cancer from H&E Whole Slide Images 

**Title (ZH)**: 染色组织学全切片图像中基于形态学的结直肠癌五年生存率预测模型 

**Authors**: Usama Sajjad, Abdul Rehman Akbar, Ziyu Su, Deborah Knight, Wendy L. Frankel, Metin N. Gurcan, Wei Chen, Muhammad Khalid Khan Niazi  

**Link**: [PDF](https://arxiv.org/pdf/2510.14800)  

**Abstract**: Colorectal cancer (CRC) remains the third most prevalent malignancy globally, with approximately 154,000 new cases and 54,000 projected deaths anticipated for 2025. The recent advancement of foundation models in computational pathology has been largely propelled by task agnostic methodologies that can overlook organ-specific crucial morphological patterns that represent distinct biological processes that can fundamentally influence tumor behavior, therapeutic response, and patient outcomes. The aim of this study is to develop a novel, interpretable AI model, PRISM (Prognostic Representation of Integrated Spatial Morphology), that incorporates a continuous variability spectrum within each distinct morphology to characterize phenotypic diversity and reflecting the principle that malignant transformation occurs through incremental evolutionary processes rather than abrupt phenotypic shifts. PRISM is trained on 8.74 million histological images extracted from surgical resection specimens of 424 patients with stage III CRC. PRISM achieved superior prognostic performance for five-year OS (AUC = 0.70 +- 0.04; accuracy = 68.37% +- 4.75%; HR = 3.34, 95% CI = 2.28-4.90; p < 0.0001), outperforming existing CRC-specific methods by 15% and AI foundation models by ~23% accuracy. It showed sex-agnostic robustness (AUC delta = 0.02; accuracy delta = 0.15%) and stable performance across clinicopathological subgroups, with minimal accuracy fluctuation (delta = 1.44%) between 5FU/LV and CPT-11/5FU/LV regimens, replicating the Alliance cohort finding of no survival difference between treatments. 

**Abstract (ZH)**: 结直肠癌（CRC）仍然是全球第三大常见恶性肿瘤，预计2025年将有约154,000例新发病例和54,000例死亡。最近，在计算病理学中基础模型的发展很大程度上得益于任务无关的方法，这些方法可能会忽略特定器官的关键形态学模式，而这些模式代表了不同的生物学过程，这些过程可以从根本上影响肿瘤行为、治疗反应和患者预后。本研究的目标是开发一种新的可解释人工智能模型PRISM（预后综合空间形态学代表），该模型在每种独特形态内结合了连续的变异谱，以表征表型多样性，并反映恶性转化是通过渐进的进化过程，而不是突然的表型转变。PRISM在424例III期CRC患者的手术切除标本中提取的874万张组织学图像上进行训练。PRISM在五年总生存期的预后性能表现优异（AUC = 0.70 ± 0.04；准确率 = 68.37% ± 4.75%；HR = 3.34，95% CI = 2.28-4.90；p < 0.0001），其性能优于现有的CRC特异性方法15%，优于AI基础模型约23%的准确率。它表现出性别无偏好性稳健性（AUC变化 = 0.02；准确率变化 = 0.15%），并在临床病理学亚组中表现出稳定性能，治疗方案（FOLFOX和CPT-11/FOLFOX）之间的小幅准确率波动（变化 = 1.44%），复制了Alliance队列研究中治疗之间无生存差异的发现。 

---
# Cross-Scenario Unified Modeling of User Interests at Billion Scale 

**Title (ZH)**: 十亿规模跨场景统一建模用户兴趣 

**Authors**: Manjie Xu, Cheng Chen, Xin Jia, Jingyi Zhou, Yongji Wu, Zejian Wang, Chi Zhang, Kai Zuo, Yibo Chen, Xu Tang, Yao Hu, Yixin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14788)  

**Abstract**: User interests on content platforms are inherently diverse, manifesting through complex behavioral patterns across heterogeneous scenarios such as search, feed browsing, and content discovery. Traditional recommendation systems typically prioritize business metric optimization within isolated specific scenarios, neglecting cross-scenario behavioral signals and struggling to integrate advanced techniques like LLMs at billion-scale deployments, which finally limits their ability to capture holistic user interests across platform touchpoints. We propose RED-Rec, an LLM-enhanced hierarchical Recommender Engine for Diversified scenarios, tailored for industry-level content recommendation systems. RED-Rec unifies user interest representations across multiple behavioral contexts by aggregating and synthesizing actions from varied scenarios, resulting in comprehensive item and user modeling. At its core, a two-tower LLM-powered framework enables nuanced, multifaceted representations with deployment efficiency, and a scenario-aware dense mixing and querying policy effectively fuses diverse behavioral signals to capture cross-scenario user intent patterns and express fine-grained, context-specific intents during serving. We validate RED-Rec through online A/B testing on hundreds of millions of users in RedNote through online A/B testing, showing substantial performance gains in both content recommendation and advertisement targeting tasks. We further introduce a million-scale sequential recommendation dataset, RED-MMU, for comprehensive offline training and evaluation. Our work advances unified user modeling, unlocking deeper personalization and fostering more meaningful user engagement in large-scale UGC platforms. 

**Abstract (ZH)**: 用户在内容平台上的兴趣本质上是多元的，通过跨异构场景（如搜索、信息流浏览和内容发现）的复杂行为模式表现出来。传统的推荐系统通常优先优化孤立特定场景下的业务指标，忽视了跨场景的行为信号，并且难以在十亿规模的部署中集成先进的技术（如LLMs），从而限制了它们在全平台触点上捕捉用户整体兴趣的能力。我们提出了RED-Rec，一种增强型层次推荐引擎，专为工业级内容推荐系统设计。RED-Rec 通过聚合和综合来自不同场景的多种行为动作来统一用户的兴趣表示，实现全面的项目和用户建模。其核心是一个基于LLM的双塔框架，能够提供细腻、多维度的表示，并通过场景感知的密集混合和查询策略有效融合多种行为信号，以捕捉跨场景的用户意图模式，并在服务中表达细粒度的、特定于上下文的意图。通过在线A/B测试验证，我们在RedNote上对数亿用户进行了内容推荐和广告定向任务的验证，显示出了显著的性能提升。我们还引入了一个百万规模的序列推荐数据集RED-MMU，用于全面的离线训练和评估。我们的工作推进了统一的用户建模，实现了更深层次的个性化，并促进了大规模UGC平台上的更高质量的用户参与。 

---
# Seesaw: Accelerating Training by Balancing Learning Rate and Batch Size Scheduling 

**Title (ZH)**: Seesaw: 通过平衡学习率和批量大小调度加速训练 

**Authors**: Alexandru Meterez, Depen Morwani, Jingfeng Wu, Costin-Andrei Oncescu, Cengiz Pehlevan, Sham Kakade  

**Link**: [PDF](https://arxiv.org/pdf/2510.14717)  

**Abstract**: Increasing the batch size during training -- a ''batch ramp'' -- is a promising strategy to accelerate large language model pretraining. While for SGD, doubling the batch size can be equivalent to halving the learning rate, the optimal strategy for adaptive optimizers like Adam is less clear. As a result, any batch-ramp scheduling, if used at all, is typically tuned heuristically. This work develops a principled framework for batch-size scheduling and introduces Seesaw: whenever a standard scheduler would halve the learning rate, Seesaw instead multiplies it by $1/\sqrt{2}$ and doubles the batch size, preserving loss dynamics while reducing serial steps. Theoretically, we provide, to our knowledge, the first finite-sample proof of equivalence between learning-rate decay and batch-size ramp-up for SGD on noisy linear regression, and we extend this equivalence to normalized SGD, a tractable proxy for Adam, under a variance-dominated regime observed in practice. Empirically, on 150M/300M/600M-parameter models trained at Chinchilla scale using a constant (critical) batch size, Seesaw matches cosine decay at equal FLOPs while reducing wall-clock time by $\approx 36\%$, approaching the theoretical limit implied by our analysis. 

**Abstract (ZH)**: 在训练过程中逐步增加批次大小——“批次增长”——是一种加快大规模语言模型预训练的有前途的策略。对于SGD，加倍批次大小相当于减半学习率，但对于像Adam这样的自适应优化器，最优策略尚不明确。因此，如果使用任何批次增长策略，通常会进行启发式调整。本研究开发了一个原理性的框架来调度批次大小，并引入了Seesaw：每当标准调度器减半学习率时，Seesaw 会将其乘以 $1/\sqrt{2}$ 并将批次大小加倍，保持损失动态的同时减少串行步骤。理论上，我们提供了一个基于有限样本证明SGD在噪声线性回归中学习率衰减与批次大小增长等效，这是已知的第一个证明，并将这种等效性扩展到实践观察到的方差占主导地位的情况下可求解的归一化SGD，其作为Adam的可处理代理。实验上，在使用恒定（关键）批次大小的Chinchilla规模下训练150M/300M/600M参数模型时，Seesaw 在同等FLOPs的情况下达到余弦衰减效果，将墙钟时间减少了约36%，接近我们分析所暗示的理论极限。 

---
# FedPPA: Progressive Parameter Alignment for Personalized Federated Learning 

**Title (ZH)**: FedPPA: 进步参数对齐实现个性化联邦学习 

**Authors**: Maulidi Adi Prasetia, Muhamad Risqi U. Saputra, Guntur Dharma Putra  

**Link**: [PDF](https://arxiv.org/pdf/2510.14698)  

**Abstract**: Federated Learning (FL) is designed as a decentralized, privacy-preserving machine learning paradigm that enables multiple clients to collaboratively train a model without sharing their data. In real-world scenarios, however, clients often have heterogeneous computational resources and hold non-independent and identically distributed data (non-IID), which poses significant challenges during training. Personalized Federated Learning (PFL) has emerged to address these issues by customizing models for each client based on their unique data distribution. Despite its potential, existing PFL approaches typically overlook the coexistence of model and data heterogeneity arising from clients with diverse computational capabilities. To overcome this limitation, we propose a novel method, called Progressive Parameter Alignment (FedPPA), which progressively aligns the weights of common layers across clients with the global model's weights. Our approach not only mitigates inconsistencies between global and local models during client updates, but also preserves client's local knowledge, thereby enhancing personalization robustness in non-IID settings. To further enhance the global model performance while retaining strong personalization, we also integrate entropy-based weighted averaging into the FedPPA framework. Experiments on three image classification datasets, including MNIST, FMNIST, and CIFAR-10, demonstrate that FedPPA consistently outperforms existing FL algorithms, achieving superior performance in personalized adaptation. 

**Abstract (ZH)**: 联邦学习（FL）是一种分布式、保护隐私的机器学习范式，使多个客户端能够协同训练模型而无需共享其数据。在实际场景中，客户端往往具有异质计算资源并且持有非独立且同分布的数据（非-IID），这给训练带来了重大挑战。个性化联邦学习（PFL）作为一种解决这些问题的方法应运而生，通过根据各个客户端独特的数据分布定制模型。尽管具有潜力，现有的PFL方法通常忽视了来自不同计算能力客户端的模型和数据异质性共存的问题。为克服这一限制，我们提出了一种名为渐进参数对齐（FedPPA）的新方法，该方法逐步使客户端常见层的权重与全局模型的权重对齐。我们的方法不仅在客户端更新期间缓解了全局和局部模型之间的一致性问题，而且还保留了客户端的本地知识，从而在非-IID环境中增强个性化鲁棒性。为了进一步提高全局模型性能并保持强大的个性化，我们在FedPPA框架中整合了基于熵加权平均的方法。在包括MNIST、FMNIST和CIFAR-10的三个图像分类数据集上的实验表明，FedPPA在个性化适应方面始终优于现有FL算法，表现出更好的性能。 

---
# The Bidding Games: Reinforcement Learning for MEV Extraction on Polygon Blockchain 

**Title (ZH)**: Polygon区块链中的投标游戏：面向MEV提取的强化学习 

**Authors**: Andrei Seoev, Leonid Gremyachikh, Anastasiia Smirnova, Yash Madhwal, Alisa Kalacheva, Dmitry Belousov, Ilia Zubov, Aleksei Smirnov, Denis Fedyanin, Vladimir Gorgadze, Yury Yanovich  

**Link**: [PDF](https://arxiv.org/pdf/2510.14642)  

**Abstract**: In blockchain networks, the strategic ordering of transactions within blocks has emerged as a significant source of profit extraction, known as Maximal Extractable Value (MEV). The transition from spam-based Priority Gas Auctions to structured auction mechanisms like Polygon Atlas has transformed MEV extraction from public bidding wars into sealed-bid competitions under extreme time constraints. While this shift reduces network congestion, it introduces complex strategic challenges where searchers must make optimal bidding decisions within a sub-second window without knowledge of competitor behavior or presence. Traditional game-theoretic approaches struggle in this high-frequency, partially observable environment due to their reliance on complete information and static equilibrium assumptions. We present a reinforcement learning framework for MEV extraction on Polygon Atlas and make three contributions: (1) A novel simulation environment that accurately models the stochastic arrival of arbitrage opportunities and probabilistic competition in Atlas auctions; (2) A PPO-based bidding agent optimized for real-time constraints, capable of adaptive strategy formulation in continuous action spaces while maintaining production-ready inference speeds; (3) Empirical validation demonstrating our history-conditioned agent captures 49\% of available profits when deployed alongside existing searchers and 81\% when replacing the market leader, significantly outperforming static bidding strategies. Our work establishes that reinforcement learning provides a critical advantage in high-frequency MEV environments where traditional optimization methods fail, offering immediate value for industrial participants and protocol designers alike. 

**Abstract (ZH)**: 在区块链网络中，区块内交易的战略排序已成为一种重要的利润提取来源，称为最大可提取价值（MEV）。从基于垃圾交易的优先气体拍卖过渡到结构化拍卖机制（如Polygon Atlas），将MEV提取从公开竞价转变为在极端时间限制下的密封出价竞争。虽然这种转变减少了网络拥堵，但也引入了复杂的战略挑战，要求搜索者在不到一秒钟的时间内做出最优竞价决策，且缺乏对竞争对手行为或存在性的了解。传统博弈论方法在这种高频次、部分可观测的环境中难以发挥作用，因为它们依赖于完全信息和静态均衡假设。本文提出了一种针对Polygon Atlas的增强学习框架，并做出如下贡献：（1）一种新颖的仿真环境，准确模拟 Arbitrage 机会的随机到达和Atlas 拍卖中的概率性竞争；（2）一种基于PPO的竞价代理，优化了实时约束条件，能够在连续动作空间中形成自适应策略，并保持生产就绪的推理速度；（3）实证验证表明，当部署在现有搜索者旁边时，我们的历史条件代理捕获了49%的可用利润，而取代市场领导者时则捕获了81%的利润，显著优于静态竞价策略。我们的工作证明了增强学习在传统优化方法失效的高频MEV环境中提供了关键优势，为工业参与者和协议设计师提供了即时价值。 

---
# Causality Enhancement for Cross-Domain Recommendation 

**Title (ZH)**: 跨域推荐中的因果性增强 

**Authors**: Zhibo Wu, Yunfan Wu, Lin Jiang, Ping Yang, Yao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14641)  

**Abstract**: Cross-domain recommendation forms a crucial component in recommendation systems. It leverages auxiliary information through source domain tasks or features to enhance target domain recommendations. However, incorporating inconsistent source domain tasks may result in insufficient cross-domain modeling or negative transfer. While incorporating source domain features without considering the underlying causal relationships may limit their contribution to final predictions. Thus, a natural idea is to directly train a cross-domain representation on a causality-labeled dataset from the source to target domain. Yet this direction has been rarely explored, as identifying unbiased real causal labels is highly challenging in real-world scenarios. In this work, we attempt to take a first step in this direction by proposing a causality-enhanced framework, named CE-CDR. Specifically, we first reformulate the cross-domain recommendation as a causal graph for principled guidance. We then construct a causality-aware dataset heuristically. Subsequently, we derive a theoretically unbiased Partial Label Causal Loss to generalize beyond the biased causality-aware dataset to unseen cross-domain patterns, yielding an enriched cross-domain representation, which is then fed into the target model to enhance target-domain recommendations. Theoretical and empirical analyses, as well as extensive experiments, demonstrate the rationality and effectiveness of CE-CDR and its general applicability as a model-agnostic plugin. Moreover, it has been deployed in production since April 2025, showing its practical value in real-world applications. 

**Abstract (ZH)**: 因果增强跨域推荐框架：CE-CDR 

---
# GemiRec: Interest Quantization and Generation for Multi-Interest Recommendation 

**Title (ZH)**: GemiRec：兴趣量化与生成的多兴趣推荐 

**Authors**: Zhibo Wu, Yunfan Wu, Quan Liu, Lin Jiang, Ping Yang, Yao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14626)  

**Abstract**: Multi-interest recommendation has gained attention, especially in industrial retrieval stage. Unlike classical dual-tower methods, it generates multiple user representations instead of a single one to model comprehensive user interests. However, prior studies have identified two underlying limitations: the first is interest collapse, where multiple representations homogenize. The second is insufficient modeling of interest evolution, as they struggle to capture latent interests absent from a user's historical behavior. We begin with a thorough review of existing works in tackling these limitations. Then, we attempt to tackle these limitations from a new perspective. Specifically, we propose a framework-level refinement for multi-interest recommendation, named GemiRec. The proposed framework leverages interest quantization to enforce a structural interest separation and interest generation to learn the evolving dynamics of user interests explicitly. It comprises three modules: (a) Interest Dictionary Maintenance Module (IDMM) maintains a shared quantized interest dictionary. (b) Multi-Interest Posterior Distribution Module (MIPDM) employs a generative model to capture the distribution of user future interests. (c) Multi-Interest Retrieval Module (MIRM) retrieves items using multiple user-interest representations. Both theoretical and empirical analyses, as well as extensive experiments, demonstrate its advantages and effectiveness. Moreover, it has been deployed in production since March 2025, showing its practical value in industrial applications. 

**Abstract (ZH)**: 多兴趣推荐在工业检索阶段获得了关注。不同于经典的双塔方法，它生成多个用户表示而不是单个表示以建模用户的全方位兴趣。然而，先前的研究发现了两个潜在的局限性：一是兴趣坍缩，多个表示变得同质化。二是兴趣演化建模不足，难以捕捉用户历史行为中不存在的潜在兴趣。我们首先对现有工作的这些局限性进行了详尽的回顾。然后，我们尝试从一个新的视角来解决这些局限性。具体而言，我们提出了一种针对多兴趣推荐的框架级改进框架，称为GemiRec。该框架利用兴趣量化来强制构建结构化兴趣分离，并利用生成模型来明确学习用户兴趣的演化动态。它包括三个模块：(a) 兴趣字典维护模块 (IDMM) 维护一个共享的兴趣量化词典。(b) 多兴趣后验分布模块 (MIPDM) 求取用户的未来兴趣分布。(c) 多兴趣检索模块 (MIRM) 使用多个用户兴趣表示进行项目检索。理论和实证分析以及广泛实验表明其优势和有效性。此外，自2025年3月起已在生产中部署，展示了其在工业应用中的实用价值。 

---
# LeapFactual: Reliable Visual Counterfactual Explanation Using Conditional Flow Matching 

**Title (ZH)**: LeapFactual: 可靠的基于条件流匹配的视觉反事实解释 

**Authors**: Zhuo Cao, Xuan Zhao, Lena Krieger, Hanno Scharr, Ira Assent  

**Link**: [PDF](https://arxiv.org/pdf/2510.14623)  

**Abstract**: The growing integration of machine learning (ML) and artificial intelligence (AI) models into high-stakes domains such as healthcare and scientific research calls for models that are not only accurate but also interpretable. Among the existing explainable methods, counterfactual explanations offer interpretability by identifying minimal changes to inputs that would alter a model's prediction, thus providing deeper insights. However, current counterfactual generation methods suffer from critical limitations, including gradient vanishing, discontinuous latent spaces, and an overreliance on the alignment between learned and true decision boundaries. To overcome these limitations, we propose LeapFactual, a novel counterfactual explanation algorithm based on conditional flow matching. LeapFactual generates reliable and informative counterfactuals, even when true and learned decision boundaries diverge. Following a model-agnostic approach, LeapFactual is not limited to models with differentiable loss functions. It can even handle human-in-the-loop systems, expanding the scope of counterfactual explanations to domains that require the participation of human annotators, such as citizen science. We provide extensive experiments on benchmark and real-world datasets showing that LeapFactual generates accurate and in-distribution counterfactual explanations that offer actionable insights. We observe, for instance, that our reliable counterfactual samples with labels aligning to ground truth can be beneficially used as new training data to enhance the model. The proposed method is broadly applicable and enhances both scientific knowledge discovery and non-expert interpretability. 

**Abstract (ZH)**: 机器学习和人工智能在高 stakes 领域如医疗和科学研究中的日益集成呼唤着不仅准确而且可解释的模型。现有的可解释方法中，因果解释通过识别最小输入变化来改变模型预测，提供了深入的洞察，但当前的因果解释生成方法存在梯度消失、非连续的潜在空间以及对学习和真实决策边界的对齐过度依赖的关键限制。为了克服这些限制，我们提出了基于条件流匹配的 LeapFactual，这是一种新颖的因果解释算法。LeapFactual 即使在真实和学习的决策边界存在分歧时也能生成可靠和信息丰富的因果解释。遵循模型无关的方法，LeapFactual 不局限于具有可微损失函数的模型。它甚至可以处理人机交互系统，将因果解释的范围扩展到需要人类注释员参与的领域，例如公民科学。我们在基准数据集和真实世界数据集上的广泛实验表明，LeapFactual 生成了准确且在分布内的因果解释，提供了可操作的洞察。例如，我们可靠且标签对齐的因果样本可以有益地用作新的训练数据以增强模型。所提出的方法具有广泛的应用性，提升了科学知识发现和非专家可解释性。 

---
# Beyond Correctness: Evaluating Subjective Writing Preferences Across Cultures 

**Title (ZH)**: 超越正确性：跨文化评价主观写作偏好 

**Authors**: Shuangshuang Ying, Yunwen Li, Xingwei Qu, Xin Li, Sheng Jin, Minghao Liu, Zhoufutu Wen, Xeron Du, Tianyu Zheng, Yichi Zhang, Letian Ni, Yuyang Cheng, Qiguang Chen, Jingzhe Ding, Shengda Long, Wangchunshu Zhou, Jiazhan Feng, Wanjun Zhong, Libo Qin, Ge Zhang, Wenhao Huang, Wanxiang Che, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14616)  

**Abstract**: Current preference learning methods achieve high accuracy on standard benchmarks but exhibit significant performance degradation when objective quality signals are removed. We introduce WritingPreferenceBench, a dataset of 1,800 human-annotated preference pairs (1,200 English, 600 Chinese) across 8 creative writing genres, where responses are matched for objective correctness, factual accuracy, and length. On this benchmark, sequence-based reward models--the standard architecture for RLHF--achieve only 52.7% mean accuracy, while zero-shot language model judges perform at 53.9%. In contrast, generative reward models that produce explicit reasoning chains achieve 81.8% accuracy. We observe high within-model variance across genres: individual models range from 18.2% to 81.8% accuracy across different writing categories, with standard deviations averaging 10.1%. This variance persists regardless of model scale, with 27B parameter models showing no consistent improvement over 8B variants. Our results suggest that current RLHF methods primarily learn to detect objective errors rather than capture subjective quality preferences (e.g., creativity, stylistic flair, and emotional resonance), and that successful preference modeling may require intermediate reasoning representations rather than direct classification. 

**Abstract (ZH)**: 当前的偏好学习方法在标准基准上实现了高准确度，但在移除客观质量信号时表现出显著性能下降。我们介绍了WritingPreferenceBench数据集，包含1800个人工标注的偏好配对（1200个英语，600个中文），覆盖8种创造性写作体裁，其中响应在客观正确性、事实准确性和长度方面相配。在这个基准上，基于序列的奖励模型——RLHF的标准架构——仅实现了52.7%的平均准确度，而零样本语言模型评估器则达到了53.9%。相比之下，生成式奖励模型产生显式推理链的准确度达到了81.8%。我们观察到高模型内部变异性：个体模型在不同写作类别中的准确率范围从18.2%到81.8%，平均标准差为10.1%。这种变异性在不同模型规模下依然存在，27B参数模型并未显示一致性的改进效果。我们的结果表明，当前的RLHF方法主要学会了检测客观错误而非捕捉主观质量偏好（如创意、风格和情感共鸣），而成功的偏好建模可能需要中间推理表示而非直接分类。 

---
# Local Causal Discovery for Statistically Efficient Causal Inference 

**Title (ZH)**: 局部因果发现以实现统计效率的因果推断 

**Authors**: Mátyás Schubert, Tom Claassen, Sara Magliacane  

**Link**: [PDF](https://arxiv.org/pdf/2510.14582)  

**Abstract**: Causal discovery methods can identify valid adjustment sets for causal effect estimation for a pair of target variables, even when the underlying causal graph is unknown. Global causal discovery methods focus on learning the whole causal graph and therefore enable the recovery of optimal adjustment sets, i.e., sets with the lowest asymptotic variance, but they quickly become computationally prohibitive as the number of variables grows. Local causal discovery methods offer a more scalable alternative by focusing on the local neighborhood of the target variables, but are restricted to statistically suboptimal adjustment sets. In this work, we propose Local Optimal Adjustments Discovery (LOAD), a sound and complete causal discovery approach that combines the computational efficiency of local methods with the statistical optimality of global methods. First, LOAD identifies the causal relation between the targets and tests if the causal effect is identifiable by using only local information. If it is identifiable, it then finds the optimal adjustment set by leveraging local causal discovery to infer the mediators and their parents. Otherwise, it returns the locally valid parent adjustment sets based on the learned local structure. In our experiments on synthetic and realistic data LOAD outperforms global methods in scalability, while providing more accurate effect estimation than local methods. 

**Abstract (ZH)**: 局部最优调整发现（LOAD）：结合局部方法的高效性和全局方法的统计最优性进行因果发现 

---
# Selective Labeling with False Discovery Rate Control 

**Title (ZH)**: 选择性标注与假发现率控制 

**Authors**: Huipeng Huang, Wenbo Liao, Huajun Xi, Hao Zeng, Mengchen Zhao, Hongxin Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.14581)  

**Abstract**: Obtaining high-quality labels for large datasets is expensive, requiring massive annotations from human experts. While AI models offer a cost-effective alternative by predicting labels, their label quality is compromised by the unavoidable labeling errors. Existing methods mitigate this issue through selective labeling, where AI labels a subset and human labels the remainder. However, these methods lack theoretical guarantees on the quality of AI-assigned labels, often resulting in unacceptably high labeling error within the AI-labeled subset. To address this, we introduce \textbf{Conformal Labeling}, a novel method to identify instances where AI predictions can be provably trusted. This is achieved by controlling the false discovery rate (FDR), the proportion of incorrect labels within the selected subset. In particular, we construct a conformal $p$-value for each test instance by comparing AI models' predicted confidence to those of calibration instances mislabeled by AI models. Then, we select test instances whose $p$-values are below a data-dependent threshold, certifying AI models' predictions as trustworthy. We provide theoretical guarantees that Conformal Labeling controls the FDR below the nominal level, ensuring that a predefined fraction of AI-assigned labels is correct on average. Extensive experiments demonstrate that our method achieves tight FDR control with high power across various tasks, including image and text labeling, and LLM QA. 

**Abstract (ZH)**: Conformal Labeling：通过控制错误发现率来证明AI预测的可靠性 

---
# Real-Time Surgical Instrument Defect Detection via Non-Destructive Testing 

**Title (ZH)**: 基于非destructive testing的实时手术器械缺陷检测 

**Authors**: Qurrat Ul Ain, Atif Aftab Ahmed Jilani, Zunaira Shafqat, Nigar Azhar Butt  

**Link**: [PDF](https://arxiv.org/pdf/2510.14525)  

**Abstract**: Defective surgical instruments pose serious risks to sterility, mechanical integrity, and patient safety, increasing the likelihood of surgical complications. However, quality control in surgical instrument manufacturing often relies on manual inspection, which is prone to human error and inconsistency. This study introduces SurgScan, an AI-powered defect detection framework for surgical instruments. Using YOLOv8, SurgScan classifies defects in real-time, ensuring high accuracy and industrial scalability. The model is trained on a high-resolution dataset of 102,876 images, covering 11 instrument types and five major defect categories. Extensive evaluation against state-of-the-art CNN architectures confirms that SurgScan achieves the highest accuracy (99.3%) with real-time inference speeds of 4.2-5.8 ms per image, making it suitable for industrial deployment. Statistical analysis demonstrates that contrast-enhanced preprocessing significantly improves defect detection, addressing key limitations in visual inspection. SurgScan provides a scalable, cost-effective AI solution for automated quality control, reducing reliance on manual inspection while ensuring compliance with ISO 13485 and FDA standards, paving the way for enhanced defect detection in medical manufacturing. 

**Abstract (ZH)**: 缺陷手术器械对无菌性、机械完整性和患者安全构成严重风险，增加手术并发症的可能性。然而，手术器械制造中的质量控制往往依赖于人工检查，容易出现人为错误和不一致性。本文介绍了一种基于AI的手术器械缺陷检测框架SurgScan。利用YOLOv8，SurgScan实现了实时缺陷分类，保证了高精度和工业可扩展性。该模型在102,876张高分辨率图像的数据集上进行训练，涵盖了11种器械类型和五大主要缺陷类别。与最先进的CNN架构进行广泛评估证实，SurgScan在保持99.3%精度的同时，实现了每张图像4.2-5.8毫秒的实时推理速度，使其适合工业部署。统计分析表明，对比增强预处理显著提高了缺陷检测效果，解决了视觉检查的关键局限性。SurgScan提供了一种可扩展、低成本的AI解决方案，用于自动化质量控制，减少对人工检查的依赖，同时确保符合ISO 13485和FDA标准，为医疗制造业中的缺陷检测提升铺平了道路。 

---
# From Guess2Graph: When and How Can Unreliable Experts Safely Boost Causal Discovery in Finite Samples? 

**Title (ZH)**: 从Guess2Graph：不可靠的专家在有限样本中如何安全提升因果发现？ 

**Authors**: Sujai Hiremath, Dominik Janzing, Philipp Faller, Patrick Blöbaum, Elke Kirschbaum, Shiva Prasad Kasiviswanathan, Kyra Gan  

**Link**: [PDF](https://arxiv.org/pdf/2510.14488)  

**Abstract**: Causal discovery algorithms often perform poorly with limited samples. While integrating expert knowledge (including from LLMs) as constraints promises to improve performance, guarantees for existing methods require perfect predictions or uncertainty estimates, making them unreliable for practical use. We propose the Guess2Graph (G2G) framework, which uses expert guesses to guide the sequence of statistical tests rather than replacing them. This maintains statistical consistency while enabling performance improvements. We develop two instantiations of G2G: PC-Guess, which augments the PC algorithm, and gPC-Guess, a learning-augmented variant designed to better leverage high-quality expert input. Theoretically, both preserve correctness regardless of expert error, with gPC-Guess provably outperforming its non-augmented counterpart in finite samples when experts are "better than random." Empirically, both show monotonic improvement with expert accuracy, with gPC-Guess achieving significantly stronger gains. 

**Abstract (ZH)**: 因果发现算法在样本有限时常常表现不佳。通过将专家知识（包括来自大语言模型的知识）作为约束整合以提高性能的前景令人期待，但现有方法的可靠性依赖于完美的预测或不确定性估计，这使其在实际应用中不可靠。我们提出了Guess2Graph (G2G)框架，该框架利用专家猜测来引导统计测试的顺序，而不仅仅是取代它们。这保持了统计一致性的同时，允许性能提升。我们开发了G2G的两个实例：扩展PC算法的PC-Guess，以及更擅长利用高质量专家输入的可学习增强变体gPC-Guess。理论上，无论专家错误如何，两者均保持正确性，且在专家优于随机的情况下，gPC-Guess在有限样本中的表现优于其非增强版本。实验证明，两者均随专家准确性的提高而逐步提升，其中gPC-Guess获得了显著更大的提升。 

---
# Semantic representations emerge in biologically inspired ensembles of cross-supervising neural networks 

**Title (ZH)**: 生物启发的跨监督神经网络集成中的语义表示 

**Authors**: Roy Urbach, Elad Schneidman  

**Link**: [PDF](https://arxiv.org/pdf/2510.14486)  

**Abstract**: Brains learn to represent information from a large set of stimuli, typically by weak supervision. Unsupervised learning is therefore a natural approach for exploring the design of biological neural networks and their computations. Accordingly, redundancy reduction has been suggested as a prominent design principle of neural encoding, but its ``mechanistic'' biological implementation is unclear. Analogously, unsupervised training of artificial neural networks yields internal representations that allow for accurate stimulus classification or decoding, but typically rely on biologically-implausible implementations. We suggest that interactions between parallel subnetworks in the brain may underlie such learning: we present a model of representation learning by ensembles of neural networks, where each network learns to encode stimuli into an abstract representation space by cross-supervising interactions with other networks, for inputs they receive simultaneously or in close temporal proximity. Aiming for biological plausibility, each network has a small ``receptive field'', thus receiving a fixed part of the external input, and the networks do not share weights. We find that for different types of network architectures, and for both visual or neuronal stimuli, these cross-supervising networks learn semantic representations that are easily decodable and that decoding accuracy is comparable to supervised networks -- both at the level of single networks and the ensemble. We further show that performance is optimal for small receptive fields, and that sparse connectivity between networks is nearly as accurate as all-to-all interactions, with far fewer computations. We thus suggest a sparsely interacting collective of cross-supervising networks as an algorithmic framework for representational learning and collective computation in the brain. 

**Abstract (ZH)**: 大脑通过从大范围刺激中学习表示信息，通常依赖于弱监督。因此，无监督学习是探索生物神经网络设计及其计算的自然方法。相应地，冗余减少被建议为神经编码的一个重要设计原则，但其“机械”生物学实现尚不清楚。类似地，人工神经网络的无监督训练会产生内部表示，允许准确的刺激分类或解码，但通常依赖于不符合生物学实现的方案。我们建议大脑平行子网络之间的相互作用可能底层实现这种学习：我们提出了一种由神经网络集合实现的表示学习模型，其中每个网络通过与其他网络的交叉监督学习将刺激编码到抽象的表示空间中，用于它们同时接收到的输入或接近时间窗口内的输入。为了追求生物可行性，每个网络具有一个小的“感受野”，因此接收外部输入的一部分，并且网络之间不共享权重。我们发现，对于不同类型的网络架构和无论是视觉还是神经元刺激，这些交叉监督的网络都能够学习易于解码的语义表示，解码准确性与监督网络相当——无论是单个网络还是整个网络集合。我们进一步表明，性能在小感受野时最佳，网络传播的稀疏连接几乎与全互连接效果一样准确，并且计算量更少。因此，我们建议一种稀疏交互的交叉监督网络集群作为大脑表示学习和集体计算的算法框架。 

---
# Feature Selection and Regularization in Multi-Class Classification: An Empirical Study of One-vs-Rest Logistic Regression with Gradient Descent Optimization and L1 Sparsity Constraints 

**Title (ZH)**: 多类分类中特征选择与正则化：基于梯度下降优化和L1稀疏约束的一对多逻辑回归 empirical 研究 

**Authors**: Jahidul Arafat, Fariha Tasmin, Md Kaosar Uddin, Sanjaya Poudel, Eftakhar Ahmed Arnob  

**Link**: [PDF](https://arxiv.org/pdf/2510.14449)  

**Abstract**: Multi-class wine classification presents fundamental trade-offs between model accuracy, feature dimensionality, and interpretability - critical factors for production deployment in analytical chemistry. This paper presents a comprehensive empirical study of One-vs-Rest logistic regression on the UCI Wine dataset (178 samples, 3 cultivars, 13 chemical features), comparing from-scratch gradient descent implementation against scikit-learn's optimized solvers and quantifying L1 regularization effects on feature sparsity. Manual gradient descent achieves 92.59 percent mean test accuracy with smooth convergence, validating theoretical foundations, though scikit-learn provides 24x training speedup and 98.15 percent accuracy. Class-specific analysis reveals distinct chemical signatures with heterogeneous patterns where color intensity varies dramatically (0.31 to 16.50) across cultivars. L1 regularization produces 54-69 percent feature reduction with only 4.63 percent accuracy decrease, demonstrating favorable interpretability-performance trade-offs. We propose an optimal 5-feature subset achieving 62 percent complexity reduction with estimated 92-94 percent accuracy, enabling cost-effective deployment with 80 dollars savings per sample and 56 percent time reduction. Statistical validation confirms robust generalization with sub-2ms prediction latency suitable for real-time quality control. Our findings provide actionable guidelines for practitioners balancing comprehensive chemical analysis against targeted feature measurement in resource-constrained environments. 

**Abstract (ZH)**: 多类别葡萄酒分类展示了模型准确性、特征维度和可解释性之间的基本权衡——这对分析化学生产部署至关重要。本文对UCI葡萄酒数据集（178个样本，3个葡萄品种，13个化学特征）进行了One-vs-Rest逻辑回归的全面经验研究，比较了自实现梯度下降方法与scikit-learn优化求解器，并量化了L1正则化对特征稀疏性的影响。 

---
# Big Data Approaches to Bovine Bioacoustics: A FAIR-Compliant Dataset and Scalable ML Framework for Precision Livestock Welfare 

**Title (ZH)**: 基于大数据方法的牛生物声学研究：符合FAIR原则的数据集及可扩展的机器学习框架以实现精准畜牧福祉 

**Authors**: Mayuri Kate, Suresh Neethirajan  

**Link**: [PDF](https://arxiv.org/pdf/2510.14443)  

**Abstract**: The convergence of IoT sensing, edge computing, and machine learning is transforming precision livestock farming. Yet bioacoustic data streams remain underused because of computational complexity and ecological validity challenges. We present one of the most comprehensive bovine vocalization datasets to date, with 569 curated clips covering 48 behavioral classes, recorded across three commercial dairy farms using multiple microphone arrays and expanded to 2900 samples through domain informed augmentation. This FAIR compliant resource addresses major Big Data challenges - volume (90 hours of recordings, 65.6 GB), variety (multi farm and multi zone acoustics), velocity (real time processing), and veracity (noise robust feature extraction). Our distributed processing framework integrates advanced denoising using iZotope RX, multimodal synchronization through audio and video alignment, and standardized feature engineering with 24 acoustic descriptors generated from Praat, librosa, and openSMILE. Preliminary benchmarks reveal distinct class level acoustic patterns for estrus detection, distress classification, and maternal communication. The datasets ecological realism, reflecting authentic barn acoustics rather than controlled settings, ensures readiness for field deployment. This work establishes a foundation for animal centered AI, where bioacoustic data enable continuous and non invasive welfare assessment at industrial scale. By releasing standardized pipelines and detailed metadata, we promote reproducible research that connects Big Data analytics, sustainable agriculture, and precision livestock management. The framework supports UN SDG 9, showing how data science can turn traditional farming into intelligent, welfare optimized systems that meet global food needs while upholding ethical animal care. 

**Abstract (ZH)**: 物联网 sensing、边缘计算和机器学习的融合正在改变精准畜牧业。然而，由于计算复杂性和生态有效性方面的挑战，生物声学数据流仍被广泛应用不足。我们呈现了迄今为止最为全面的牛隻鸣叫数据集，包含569个经过筛选的片段，涵盖48种行为类别，并通过领域知识增强扩展至2900个样本，记录于三家商业奶牛场使用多个麦克风阵列。该资源符合FAIR原则，解决大规模数据挑战——数据量（90小时录音，65.6GB）、多样性（多农场和多区域声学）、速度（实时处理）和真实性（噪声鲁棒特征提取）。我们的分布式处理框架整合了先进的去噪技术（iZotope RX）、多模态同步（通过音频和视频对齐）以及标准化特征工程（由Praat、librosa和openSMILE生成24个声学描述符）。初步基准测试显示，对于发情检测、应激分类和母性交流具有明显的类别水平声学模式。该数据集的生态现实性，反映实际牛舍声学环境而非受控环境，确保其适用于现场部署。本研究为以动物为中心的人工智能奠定了基础，生物声学数据使大规模、连续和不侵入式的福利评估成为可能。通过发布标准化管道和详细的元数据，我们促进了可重复的研究，连接大规模数据分析、可持续农业和精准畜牧业管理。该框架支持联合国可持续发展目标9，展示了如何通过数据科学将传统农业转变为智能、福利优化的系统，满足全球食物需求的同时保持伦理的动物护理。 

---
# Instructions are all you need: Self-supervised Reinforcement Learning for Instruction Following 

**Title (ZH)**: 所需指令即一切：自监督强化学习在指令跟随中的应用 

**Authors**: Qingyu Ren, Qianyu He, Bowei Zhang, Jie Zeng, Jiaqing Liang, Yanghua Xiao, Weikang Zhou, Zeye Sun, Fei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14420)  

**Abstract**: Language models often struggle to follow multi-constraint instructions that are crucial for real-world applications. Existing reinforcement learning (RL) approaches suffer from dependency on external supervision and sparse reward signals from multi-constraint tasks. We propose a label-free self-supervised RL framework that eliminates dependency on external supervision by deriving reward signals directly from instructions and generating pseudo-labels for reward model training. Our approach introduces constraint decomposition strategies and efficient constraint-wise binary classification to address sparse reward challenges while maintaining computational efficiency. Experiments show that our approach generalizes well, achieving strong improvements across 3 in-domain and 5 out-of-domain datasets, including challenging agentic and multi-turn instruction following. The data and code are publicly available at this https URL 

**Abstract (ZH)**: 语言模型在遵循关键现实应用中的多约束指令时往往表现不佳。现有的强化学习（RL）方法依赖于外部监督和来自多约束任务的稀疏奖励信号。我们提出了一种标签-free自监督RL框架，通过直接从指令中推导奖励信号并为奖励模型训练生成伪标签，消除对外部监督的依赖。我们的方法引入了约束分解策略和高效的按约束二元分类，以应对稀疏奖励挑战并保持计算效率。实验表明，我们的方法具有良好的泛化能力，在3个领域内和5个领域外数据集上取得了显著改进，包括具有挑战性的代理性和多轮指令遵循任务。数据和代码已公开，可在此链接获取。 

---
# BinCtx: Multi-Modal Representation Learning for Robust Android App Behavior Detection 

**Title (ZH)**: BinCtx: 多模态表示学习在鲁棒Android应用程序行为检测中的应用 

**Authors**: Zichen Liu, Shao Yang, Xusheng Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.14344)  

**Abstract**: Mobile app markets host millions of apps, yet undesired behaviors (e.g., disruptive ads, illegal redirection, payment deception) remain hard to catch because they often do not rely on permission-protected APIs and can be easily camouflaged via UI or metadata edits. We present BINCTX, a learning approach that builds multi-modal representations of an app from (i) a global bytecode-as-image view that captures code-level semantics and family-style patterns, (ii) a contextual view (manifested actions, components, declared permissions, URL/IP constants) indicating how behaviors are triggered, and (iii) a third-party-library usage view summarizing invocation frequencies along inter-component call paths. The three views are embedded and fused to train a contextual-aware classifier. On real-world malware and benign apps, BINCTX attains a macro F1 of 94.73%, outperforming strong baselines by at least 14.92%. It remains robust under commercial obfuscation (F1 84% post-obfuscation) and is more resistant to adversarial samples than state-of-the-art bytecode-only systems. 

**Abstract (ZH)**: BINCTX：一种多模态学习方法用于移动应用的上下文感知分类 

---
# A Robust Classification Method using Hybrid Word Embedding for Early Diagnosis of Alzheimer's Disease 

**Title (ZH)**: 基于混合词嵌入的鲁棒分类方法在阿尔茨海默病早期诊断中的应用 

**Authors**: Yangyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14332)  

**Abstract**: Early detection of Alzheimer's Disease (AD) is greatly beneficial to AD patients, leading to early treatments that lessen symptoms and alleviating financial burden of health care. As one of the leading signs of AD, language capability changes can be used for early diagnosis of AD. In this paper, I develop a robust classification method using hybrid word embedding and fine-tuned hyperparameters to achieve state-of-the-art accuracy in the early detection of AD. Specifically, we create a hybrid word embedding based on word vectors from Doc2Vec and ELMo to obtain perplexity scores of the sentences. The scores identify whether a sentence is fluent or not and capture semantic context of the sentences. I enrich the word embedding by adding linguistic features to analyze syntax and semantics. Further, we input an embedded feature vector into logistic regression and fine tune hyperparameters throughout the pipeline. By tuning hyperparameters of the machine learning pipeline (e.g., model regularization parameter, learning rate and vector size of Doc2Vec, and vector size of ELMo), I achieve 91% classification accuracy and an Area Under the Curve (AUC) of 97% in distinguishing early AD from healthy subjects. Based on my knowledge, my model with 91% accuracy and 97% AUC outperforms the best existing NLP model for AD diagnosis with an accuracy of 88% [32]. I study the model stability through repeated experiments and find that the model is stable even though the training data is split randomly (standard deviation of accuracy = 0.0403; standard deviation of AUC = 0.0174). This affirms our proposed method is accurate and stable. This model can be used as a large-scale screening method for AD, as well as a complementary examination for doctors to detect AD. 

**Abstract (ZH)**: 早发性老年痴呆症（AD）的检测对AD患者极为有益，可以实现早期治疗，减轻症状并缓解医疗开支。作为AD的一个主要标志，语言能力的变化可以用于AD的早期诊断。本文提出了一种基于混合词嵌入和微调超参数的稳健分类方法，以实现AD早期检测的最新准确度。具体而言，我们基于Doc2Vec和ELMo的词向量创建了混合词嵌入，以获得句子的困惑度分数。这些分数可以识别句子是否流畅，并捕捉句子的语义上下文。通过添加语言特征来丰富词嵌入，以分析句法和语义。随后，我们将嵌入特征向量输入逻辑回归，并在整个流水线中微调超参数。通过微调机器学习流水线中的超参数（如模型正则化参数、学习率和Doc2Vec的向量大小、ELMo的向量大小），我们实现了91%的分类准确度和97%的曲线下面积（AUC），在区分早期AD和健康个体方面表现优异。据我所知，我的模型在准确度为91%和AUC为97%的情况下，比现有最佳的AD诊断NLP模型（准确度为88%）表现更佳。通过多次实验研究模型的稳定性，发现即使随机分割训练数据，模型也保持稳定（准确度标准差=0.0403；AUC标准差=0.0174）。这证实了我们提出的方法准确且稳定。该模型可以用作AD的大规模筛查方法，也可以作为医生诊断AD的补充检查工具。 

---
# Column Generation Using Domain-Independent Dynamic Programming 

**Title (ZH)**: 基于域无关动态规划的列生成方法 

**Authors**: Ryo Kuroiwa, Edward Lam  

**Link**: [PDF](https://arxiv.org/pdf/2510.14317)  

**Abstract**: Column generation and branch-and-price are leading methods for large-scale exact optimization. Column generation iterates between solving a master problem and a pricing problem. The master problem is a linear program, which can be solved using a generic solver. The pricing problem is highly dependent on the application but is usually discrete. Due to the difficulty of discrete optimization, high-performance column generation often relies on a custom pricing algorithm built specifically to exploit the problem's structure. This bespoke nature of the pricing solver prevents the reuse of components for other applications. We show that domain-independent dynamic programming, a software package for modeling and solving arbitrary dynamic programs, can be used as a generic pricing solver. We develop basic implementations of branch-and-price with pricing by domain-independent dynamic programming and show that they outperform a world-leading solver on static mixed integer programming formulations for seven problem classes. 

**Abstract (ZH)**: 列生成和分支定价是大型精确优化的主导方法。列生成在解决主问题和定价问题之间迭代。主问题是一个线性程序，可以使用通用求解器求解。定价问题高度依赖于应用，通常为离散型。由于离散优化的难度，高性能列生成往往依赖于针对问题结构特别定制的定价算法。这种定制化的定价求解器防止了组件在其他应用中的重用。我们展示了一种通用的定价求解器——基于通用动态规划的软件包，可以用于分支定价。我们开发了基于通用动态规划的分支定价的基本实现，并展示了它们在七个问题类别的静态混合整数规划形式化表示上优于世界领先的求解器。 

---
# TED++: Submanifold-Aware Backdoor Detection via Layerwise Tubular-Neighbourhood Screening 

**Title (ZH)**: TED++: 基于层wise筒形邻域筛选的子流形 Aware 后门检测 

**Authors**: Nam Le, Leo Yu Zhang, Kewen Liao, Shirui Pan, Wei Luo  

**Link**: [PDF](https://arxiv.org/pdf/2510.14299)  

**Abstract**: As deep neural networks power increasingly critical applications, stealthy backdoor attacks, where poisoned training inputs trigger malicious model behaviour while appearing benign, pose a severe security risk. Many existing defences are vulnerable when attackers exploit subtle distance-based anomalies or when clean examples are scarce. To meet this challenge, we introduce TED++, a submanifold-aware framework that effectively detects subtle backdoors that evade existing defences. TED++ begins by constructing a tubular neighbourhood around each class's hidden-feature manifold, estimating its local ``thickness'' from a handful of clean activations. It then applies Locally Adaptive Ranking (LAR) to detect any activation that drifts outside the admissible tube. By aggregating these LAR-adjusted ranks across all layers, TED++ captures how faithfully an input remains on the evolving class submanifolds. Based on such characteristic ``tube-constrained'' behaviour, TED++ flags inputs whose LAR-based ranking sequences deviate significantly. Extensive experiments are conducted on benchmark datasets and tasks, demonstrating that TED++ achieves state-of-the-art detection performance under both adaptive-attack and limited-data scenarios. Remarkably, even with only five held-out examples per class, TED++ still delivers near-perfect detection, achieving gains of up to 14\% in AUROC over the next-best method. The code is publicly available at this https URL. 

**Abstract (ZH)**: 随着深度神经网络在日益关键的应用中发挥作用，隐形后门攻击在训练过程中注入恶意样本，使模型在看似正常的输入下产生恶意行为，这一风险构成了严重的安全威胁。许多现有的防御方法在攻击者利用细微的距离异常或缺乏干净样本时容易失效。为应对这一挑战，我们引入了TED++，这是一个亚流形感知框架，能够有效检测现有防御方法无法察觉的隐蔽后门。TED++首先围绕每个类别的隐藏特征流形构建一个管状邻域，从少量干净激活中估计其局部“厚度”。然后，它应用局部自适应排名（LAR）来检测任何偏离允许管的激活。通过在所有层上聚合这些LAR调整后的排名，TED++捕捉输入如何忠实地保持在不断演化的类亚流形上。基于这种特征的“管约束”行为，TED++标记LAR基于的排名序列显著偏差的输入。我们在基准数据集和任务上进行了广泛实验，证明TED++在适应性攻击和少量数据场景下都达到了最先进的检测性能。即使每类仅有的五个保留样本，TED++仍能实现近乎完美的检测表现，在AUROC方面相对于下一最佳方法取得了高达14%的提升。代码已公开，可通过以下链接获取。 

---
# Beyond a Single Perspective: Towards a Realistic Evaluation of Website Fingerprinting Attacks 

**Title (ZH)**: 超越单一视角：面向网站指纹识别攻击的现实评估 

**Authors**: Xinhao Deng, Jingyou Chen, Linxiao Yu, Yixiang Zhang, Zhongyi Gu, Changhao Qiu, Xiyuan Zhao, Ke Xu, Qi Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14283)  

**Abstract**: Website Fingerprinting (WF) attacks exploit patterns in encrypted traffic to infer the websites visited by users, posing a serious threat to anonymous communication systems. Although recent WF techniques achieve over 90% accuracy in controlled experimental settings, most studies remain confined to single scenarios, overlooking the complexity of real-world environments. This paper presents the first systematic and comprehensive evaluation of existing WF attacks under diverse realistic conditions, including defense mechanisms, traffic drift, multi-tab browsing, early-stage detection, open-world settings, and few-shot scenarios. Experimental results show that many WF techniques with strong performance in isolated settings degrade significantly when facing other conditions. Since real-world environments often combine multiple challenges, current WF attacks are difficult to apply directly in practice. This study highlights the limitations of WF attacks and introduces a multidimensional evaluation framework, offering critical insights for developing more robust and practical WF attacks. 

**Abstract (ZH)**: 网站指纹识别（WF）攻击通过利用加密流量中的模式来推断用户访问的网站，对匿名通信系统构成严重威胁。尽管近期的WF技术在受控实验环境中实现超过90%的准确性，大部分研究仍然局限于单一场景，忽略了实际环境的复杂性。本文首次在多种现实条件下系统性地评估现有的WF攻击，包括防御机制、流量漂移、多标签浏览、早期检测、开放世界设置和少样本场景。实验结果表明，许多在孤立环境中表现优异的WF技术在面对其他条件时性能显著下降。由于实际环境经常结合多种挑战，当前的WF攻击在实践中难以直接应用。本文突出了WF攻击的局限性，并引入一个多维度的评估框架，为开发更 robust 和实际的WF攻击提供了关键见解。 

---
# Policy Regularized Distributionally Robust Markov Decision Processes with Linear Function Approximation 

**Title (ZH)**: 带线性函数逼近的策略正则化分布鲁棒马尔可夫决策过程 

**Authors**: Jingwen Gu, Yiting He, Zhishuai Liu, Pan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14246)  

**Abstract**: Decision-making under distribution shift is a central challenge in reinforcement learning (RL), where training and deployment environments differ. We study this problem through the lens of robust Markov decision processes (RMDPs), which optimize performance against adversarial transition dynamics. Our focus is the online setting, where the agent has only limited interaction with the environment, making sample efficiency and exploration especially critical. Policy optimization, despite its success in standard RL, remains theoretically and empirically underexplored in robust RL. To bridge this gap, we propose \textbf{D}istributionally \textbf{R}obust \textbf{R}egularized \textbf{P}olicy \textbf{O}ptimization algorithm (DR-RPO), a model-free online policy optimization method that learns robust policies with sublinear regret. To enable tractable optimization within the softmax policy class, DR-RPO incorporates reference-policy regularization, yielding RMDP variants that are doubly constrained in both transitions and policies. To scale to large state-action spaces, we adopt the $d$-rectangular linear MDP formulation and combine linear function approximation with an upper confidence bonus for optimistic exploration. We provide theoretical guarantees showing that policy optimization can achieve polynomial suboptimality bounds and sample efficiency in robust RL, matching the performance of value-based approaches. Finally, empirical results across diverse domains corroborate our theory and demonstrate the robustness of DR-RPO. 

**Abstract (ZH)**: 分布转移下的决策制定是强化学习（RL）中的一个核心挑战，其中训练环境和部署环境存在差异。我们通过鲁棒马尔可夫决策过程（RMDPs）的视角研究这一问题，RMDPs旨在优化对抗性转换动力学下的性能。我们的重点是在线设置，其中智能体与环境的交互非常有限，这使得样本效率和探索尤为重要。尽管策略优化在标准RL中取得了成功，但在鲁棒RL中，策略优化的理论和实证研究仍然不足。为弥合这一差距，我们提出了一种基于模型的在线策略优化方法——分布鲁棒正则化策略优化算法（DR-RPO），该方法能够学习具有亚线性遗憾的鲁棒策略。为了在softmax策略类中实现可处理的优化，DR-RPO引入了参考策略正则化，从而产生在转换和策略方面双约束的RMDP变体。为了扩展到大规模状态-动作空间，我们采用了$d$-矩形线性MDP形式化描述，并结合了线性函数逼近和乐观探索的上置信边界。我们提供了理论保证，表明策略优化在鲁棒RL中可以实现多项式次优性边界和样本效率，达到基于值的方法的性能。最后，我们在多个领域的实验结果验证了我们的理论，并展示了DR-RPO的鲁棒性。 

---
# Spatial Computing Communications for Multi-User Virtual Reality in Distributed Mobile Edge Computing Network 

**Title (ZH)**: 分布式移动边缘计算网络中多用户虚拟现实的时空计算通信 

**Authors**: Caolu Xu, Zhiyong Chen, Meixia Tao, Li Song, Wenjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14243)  

**Abstract**: Immersive virtual reality (VR) applications impose stringent requirements on latency, energy efficiency, and computational resources, particularly in multi-user interactive scenarios. To address these challenges, we introduce the concept of spatial computing communications (SCC), a framework designed to meet the latency and energy demands of multi-user VR over distributed mobile edge computing (MEC) networks. SCC jointly represents the physical space, defined by users and base stations, and the virtual space, representing shared immersive environments, using a probabilistic model of user dynamics and resource requirements. The resource deployment task is then formulated as a multi-objective combinatorial optimization (MOCO) problem that simultaneously minimizes system latency and energy consumption across distributed MEC resources. To solve this problem, we propose MO-CMPO, a multi-objective consistency model with policy optimization that integrates supervised learning and reinforcement learning (RL) fine-tuning guided by preference weights. Leveraging a sparse graph neural network (GNN), MO-CMPO efficiently generates Pareto-optimal solutions. Simulations with real-world New Radio base station datasets demonstrate that MO-CMPO achieves superior hypervolume performance and significantly lower inference latency than baseline methods. Furthermore, the analysis reveals practical deployment patterns: latency-oriented solutions favor local MEC execution to reduce transmission delay, while energy-oriented solutions minimize redundant placements to save energy. 

**Abstract (ZH)**: 沉浸式虚拟现实（VR）应用对延迟、能量效率和计算资源提出了严格要求，特别是在多用户交互场景中。为应对这些挑战，我们引入了空间计算通信（SCC）的概念，这是一种设计用于分布式移动边缘计算（MEC）网络中满足多用户VR延迟和能量需求的框架。SCC使用用户动态和资源需求的概率模型，联合表示由用户和基站定义的物理空间以及表示共享沉浸环境的虚拟空间。随后，资源部署任务被形式化为同时最小化分布式MEC资源上系统延迟和能量消耗的多目标组合优化（MOCO）问题。为了解决这个问题，我们提出了MO-CMPO，这是一种结合监督学习和由偏好权重引导的强化学习（RL）微调的多目标一致性模型和策略优化方法。利用稀疏图神经网络（GNN），MO-CMPO有效地生成了帕累托最优解。使用实际的New Radio基站数据集进行的仿真实验表明，MO-CMPO在超体积性能和推理延迟方面优于基线方法。此外，分析揭示了实际部署模式：延迟导向的解决方案倾向于局部执行MEC以减少传输延迟，而能量导向的解决方案倾向于最小化冗余放置以节省能量。 

---
# MAFA: A Multi-Agent Framework for Enterprise-Scale Annotation with Configurable Task Adaptation 

**Title (ZH)**: MAFA：一种具备可配置任务适应性的企业规模标注多代理框架 

**Authors**: Mahmood Hegazy, Aaron Rodrigues, Azzam Naeem  

**Link**: [PDF](https://arxiv.org/pdf/2510.14184)  

**Abstract**: We present MAFA (Multi-Agent Framework for Annotation), a production-deployed system that transforms enterprise-scale annotation workflows through configurable multi-agent collaboration. Addressing the critical challenge of annotation backlogs in financial services, where millions of customer utterances require accurate categorization, MAFA combines specialized agents with structured reasoning and a judge-based consensus mechanism. Our framework uniquely supports dynamic task adaptation, allowing organizations to define custom annotation types (FAQs, intents, entities, or domain-specific categories) through configuration rather than code changes. Deployed at JP Morgan Chase, MAFA has eliminated a 1 million utterance backlog while achieving, on average, 86% agreement with human annotators, annually saving over 5,000 hours of manual annotation work. The system processes utterances with annotation confidence classifications, which are typically 85% high, 10% medium, and 5% low across all datasets we tested. This enables human annotators to focus exclusively on ambiguous and low-coverage cases. We demonstrate MAFA's effectiveness across multiple datasets and languages, showing consistent improvements over traditional and single-agent annotation baselines: 13.8% higher Top-1 accuracy, 15.1% improvement in Top-5 accuracy, and 16.9% better F1 in our internal intent classification dataset and similar gains on public benchmarks. This work bridges the gap between theoretical multi-agent systems and practical enterprise deployment, providing a blueprint for organizations facing similar annotation challenges. 

**Abstract (ZH)**: 面向标注的多Agent框架：MAFA在金融机构大规模标注流程中的应用 

---
# Towards Reversible Model Merging For Low-rank Weights 

**Title (ZH)**: 面向低秩权重的可逆模型合并方法 

**Authors**: Mohammadsajad Alipour, Mohammad Mohammadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2510.14163)  

**Abstract**: Model merging aims to combine multiple fine-tuned models into a single set of weights that performs well across all source tasks. While prior work has shown that merging can approximate the performance of individual fine-tuned models for each task, it largely overlooks scenarios where models are compressed into low-rank representations, either through low-rank adaptation (LoRA) or post-training singular value decomposition (SVD). We first demonstrate that applying conventional merging methods to low-rank weights leads to severe performance degradation in the merged model. Motivated by this phenomenon, we propose a fundamentally different approach: instead of collapsing all adapters into one set of weights, we construct a compact basis (e.g., an equivalent of holding two or more models) from which original task-specific models can be recovered via linear combination. This reframes merging as generating a reconstruction-capable model space rather than producing a single merged model. Crucially, this allows us to ``revert'' to each individual model when needed, recognizing that no merged model can consistently outperform one specialized for its task. Building on this insight, we introduce our method, Reversible Model Merging (RMM), an efficient, data-free, and flexible method that provides a closed-form solution for selecting the optimal basis of model weights and task-specific coefficients for linear combination. Extensive experiments across diverse datasets and model scales demonstrate that RMM consistently outperforms existing merging approaches, preserving the performance of low-rank compressed models by a significant margin. 

**Abstract (ZH)**: 基于可逆模型合并的低秩模型压缩与重构 

---
# Inferred global dense residue transition graphs from primary structure sequences enable protein interaction prediction via directed graph convolutional neural networks 

**Title (ZH)**: 从初级结构序列推导出的全局密集残基转换图通过有向图卷积神经网络进行蛋白质相互作用预测 

**Authors**: Islam Akef Ebeid, Haoteng Tang, Pengfei Gu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14139)  

**Abstract**: Introduction Accurate prediction of protein-protein interactions (PPIs) is crucial for understanding cellular functions and advancing drug development. Existing in-silico methods use direct sequence embeddings from Protein Language Models (PLMs). Others use Graph Neural Networks (GNNs) for 3D protein structures. This study explores less computationally intensive alternatives. We introduce a novel framework for downstream PPI prediction through link prediction. Methods We introduce a two-stage graph representation learning framework, ProtGram-DirectGCN. First, we developed ProtGram. This approach models a protein's primary structure as a hierarchy of globally inferred n-gram graphs. In these graphs, residue transition probabilities define edge weights. Each edge connects a pair of residues in a directed graph. The probabilities are aggregated from a large corpus of sequences. Second, we propose DirectGCN, a custom directed graph convolutional neural network. This model features a unique convolutional layer. It processes information through separate path-specific transformations: incoming, outgoing, and undirected. A shared transformation is also applied. These paths are combined via a learnable gating mechanism. We apply DirectGCN to ProtGram graphs to learn residue-level embeddings. These embeddings are pooled via attention to generate protein-level embeddings for prediction. Results We first established the efficacy of DirectGCN on standard node classification benchmarks. Its performance matches established methods on general datasets. The model excels at complex, directed graphs with dense, heterophilic structures. When applied to PPI prediction, the full ProtGram-DirectGCN framework delivers robust predictive power. This strong performance holds even with limited training data. 

**Abstract (ZH)**: 介绍 准确预测蛋白质-蛋白质相互作用（PPIs）对于理解细胞功能和推进药物开发至关重要。现有计算方法使用蛋白质语言模型（PLMs）的直接序列嵌入。其他方法使用图神经网络（GNNs）处理三维蛋白质结构。本研究探索了更少计算成本的替代方案。我们引入了一种新的框架，通过链接预测进行下游PPI预测。方法 我们提出了一种两阶段图表示学习框架，名为ProtGram-DirectGCN。首先，我们开发了ProtGram。该方法将蛋白质的一级结构建模为全局推断的n-克隆图层次结构。在这类图中，残基转换概率定义边权。每条边连接有向图中的残基对。这些概率是从一个大规模序列语料库中聚合而来的。其次，我们提出了DirectGCN，这是一种定制化的有向图卷积神经网络。该模型具有独特的卷积层，通过单独的路径特定变换处理信息：入边、出边和无向边。此外还应用了共享变换。这些路径通过可学习的门控机制结合。我们将DirectGCN应用于ProtGram图，学习残基级嵌入，并通过注意力机制池化生成蛋白质级嵌入进行预测。结果 我们首先在标准节点分类基准上验证了DirectGCN的有效性。其性能在通用数据集上与现有方法相当。该模型在复杂、有向图且结构稠密和异质的结构上表现出色。在应用于PPI预测时，完整的ProtGram-DirectGCN框架提供了稳健的预测能力，即使在训练数据有限的情况下也能保持这种性能。 

---
# Extracting latent representations from X-ray spectra. Classification, regression, and accretion signatures of Chandra sources 

**Title (ZH)**: 从X射线光谱中提取潜藏表示。Chandra源的分类、回归及积累特征。 

**Authors**: Nicolò Oreste Pinciroli Vago, Juan Rafael Martínez-Galarza, Roberta Amato  

**Link**: [PDF](https://arxiv.org/pdf/2510.14102)  

**Abstract**: The study of X-ray spectra is crucial to understanding the physical nature of astrophysical sources. Machine learning methods can extract compact and informative representations of data from large datasets. The Chandra Source Catalog (CSC) provides a rich archive of X-ray spectral data, which remains largely underexplored in this context. This work aims to develop a compact and physically meaningful representation of Chandra X-ray spectra using deep learning. To verify that the learned representation captures relevant information, we evaluate it through classification, regression, and interpretability analyses. We use a transformer-based autoencoder to compress X-ray spectra. The input spectra, drawn from the CSC, include only high-significance detections. Astrophysical source types and physical summary statistics are compiled from external catalogs. We evaluate the learned representation in terms of spectral reconstruction accuracy, clustering performance on 8 known astrophysical source classes, and correlation with physical quantities such as hardness ratios and hydrogen column density ($N_H$). The autoencoder accurately reconstructs spectra with 8 latent variables. Clustering in the latent space yields a balanced classification accuracy of $\sim$40% across the 8 source classes, increasing to $\sim$69% when restricted to AGNs and stellar-mass compact objects exclusively. Moreover, latent features correlate with non-linear combinations of spectral fluxes, suggesting that the compressed representation encodes physically relevant information. The proposed autoencoder-based pipeline is a powerful tool for the representation and interpretation of X-ray spectra, providing a compact latent space that supports both classification and the estimation of physical properties. This work demonstrates the potential of deep learning for spectral studies and uncovering new patterns in X-ray data. 

**Abstract (ZH)**: X射线光谱研究对于理解天体物理源的物理性质至关重要。机器学习方法可以从大型数据集中提取紧凑且富有信息量的数据表示。钱德拉源目录（CSC）提供了丰富的X射线光谱数据集，这些数据在该领域尚未得到充分利用。本工作旨在使用深度学习开发Chandra X射线光谱的紧凑且物理意义丰富的表示。为验证所学习的表示是否捕获了相关信息，我们通过分类、回归和可解释性分析对其进行评估。我们使用基于变换器的自动编码器压缩X射线光谱。输入光谱仅包括CSC中的高信噪比检测结果。天体物理源类型和物理总括统计量是从外部目录中编制的。我们从光谱重建准确性、基于8个已知天体物理源类的聚类性能和与物理量（如硬度比和氢柱密度$N_H$）的相关性等方面评估所学习的表示。自动编码器使用8个潜在变量准确重建光谱。潜在空间聚类在8个源类上的平衡分类准确率为约40%，当仅限于活跃星系核（AGNs）和恒星质量致密天体时，分类准确率提高到约69%。此外，潜在特征与光谱通量的非线性组合相关，表明压缩表示包含了物理相关的信息。提出的基于自动编码器的方法是一个强大的工具，用于X射线光谱的表示和解释，提供了一个支持分类和物理性质估计的紧凑潜在空间。本工作展示了深度学习在光谱研究中以及在挖掘X射线数据中的新模式方面具有潜在价值。 

---
# Unlocking Out-of-Distribution Generalization in Transformers via Recursive Latent Space Reasoning 

**Title (ZH)**: 通过递归潜在空间推理解锁变压器模型的分布外泛化能力 

**Authors**: Awni Altabaa, Siyu Chen, John Lafferty, Zhuoran Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14095)  

**Abstract**: Systematic, compositional generalization beyond the training distribution remains a core challenge in machine learning -- and a critical bottleneck for the emergent reasoning abilities of modern language models. This work investigates out-of-distribution (OOD) generalization in Transformer networks using a GSM8K-style modular arithmetic on computational graphs task as a testbed. We introduce and explore a set of four architectural mechanisms aimed at enhancing OOD generalization: (i) input-adaptive recurrence; (ii) algorithmic supervision; (iii) anchored latent representations via a discrete bottleneck; and (iv) an explicit error-correction mechanism. Collectively, these mechanisms yield an architectural approach for native and scalable latent space reasoning in Transformer networks with robust algorithmic generalization capabilities. We complement these empirical results with a detailed mechanistic interpretability analysis that reveals how these mechanisms give rise to robust OOD generalization abilities. 

**Abstract (ZH)**: Transformer网络中基于计算图模块化算术的离分布泛化：一种增强离分布泛化能力的架构方法 

---
# DiffOPF: Diffusion Solver for Optimal Power Flow 

**Title (ZH)**: DiffOPF: 基于扩散的最优功率流求解器 

**Authors**: Milad Hoseinpour, Vladimir Dvorkin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14075)  

**Abstract**: The optimal power flow (OPF) is a multi-valued, non-convex mapping from loads to dispatch setpoints. The variability of system parameters (e.g., admittances, topology) further contributes to the multiplicity of dispatch setpoints for a given load. Existing deep learning OPF solvers are single-valued and thus fail to capture the variability of system parameters unless fully represented in the feature space, which is prohibitive. To solve this problem, we introduce a diffusion-based OPF solver, termed \textit{DiffOPF}, that treats OPF as a conditional sampling problem. The solver learns the joint distribution of loads and dispatch setpoints from operational history, and returns the marginal dispatch distributions conditioned on loads. Unlike single-valued solvers, DiffOPF enables sampling statistically credible warm starts with favorable cost and constraint satisfaction trade-offs. We explore the sample complexity of DiffOPF to ensure the OPF solution within a prescribed distance from the optimization-based solution, and verify this experimentally on power system benchmarks. 

**Abstract (ZH)**: 基于扩散的最优功率流求解器（DiffOPF） 

---
# Exploratory Causal Inference in SAEnce 

**Title (ZH)**: 探索性因果推理在科学中的应用 

**Authors**: Tommaso Mencattini, Riccardo Cadei, Francesco Locatello  

**Link**: [PDF](https://arxiv.org/pdf/2510.14073)  

**Abstract**: Randomized Controlled Trials are one of the pillars of science; nevertheless, they rely on hand-crafted hypotheses and expensive analysis. Such constraints prevent causal effect estimation at scale, potentially anchoring on popular yet incomplete hypotheses. We propose to discover the unknown effects of a treatment directly from data. For this, we turn unstructured data from a trial into meaningful representations via pretrained foundation models and interpret them via a sparse autoencoder. However, discovering significant causal effects at the neural level is not trivial due to multiple-testing issues and effects entanglement. To address these challenges, we introduce Neural Effect Search, a novel recursive procedure solving both issues by progressive stratification. After assessing the robustness of our algorithm on semi-synthetic experiments, we showcase, in the context of experimental ecology, the first successful unsupervised causal effect identification on a real-world scientific trial. 

**Abstract (ZH)**: 从数据中发现治疗的未知效果：一种解决神经级显著因果效应发现挑战的递归方法 

---
# On the expressivity of sparse maxout networks 

**Title (ZH)**: 稀疏MaxOut网络的表征能力 

**Authors**: Moritz Grillo, Tobias Hofmann  

**Link**: [PDF](https://arxiv.org/pdf/2510.14068)  

**Abstract**: We study the expressivity of sparse maxout networks, where each neuron takes a fixed number of inputs from the previous layer and employs a, possibly multi-argument, maxout activation. This setting captures key characteristics of convolutional or graph neural networks. We establish a duality between functions computable by such networks and a class of virtual polytopes, linking their geometry to questions of network expressivity. In particular, we derive a tight bound on the dimension of the associated polytopes, which serves as the central tool for our analysis. Building on this, we construct a sequence of depth hierarchies. While sufficiently deep sparse maxout networks are universal, we prove that if the required depth is not reached, width alone cannot compensate for the sparsity of a fixed indegree constraint. 

**Abstract (ZH)**: 我们研究稀疏maxout网络的表達能力，其中每个神经元从上一层固定数量的输入中选取，并使用可能多参数的maxout激活函数。这一设置捕获了卷积或图神经网络的关键特性。我们建立了此类网络可计算函数与一类虚拟多面体之间的对偶关系，将它们的几何结构与网络的表達能力联系起来。特别是，我们推导出与这些多面体相关的维度的紧致界，这成为我们分析的核心工具。基于此，我们构造了一系列深度层次结构。虽然足够深的稀疏maxout网络具有通用性，但我们证明了如果没有达到所需的深度，宽度 alone 无法弥补固定入度约束下的稀疏性。 

---
# Optical Computation-in-Communication enables low-latency, high-fidelity perception in telesurgery 

**Title (ZH)**: 光学计算-通信使远程手术实现低延迟、高保真感知 

**Authors**: Rui Yang, Jiaming Hu, Jian-Qing Zheng, Yue-Zhen Lu, Jian-Wei Cui, Qun Ren, Yi-Jie Yu, John Edward Wu, Zhao-Yu Wang, Xiao-Li Lin, Dandan Zhang, Mingchu Tang, Christos Masouros, Huiyun Liu, Chin-Pang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14058)  

**Abstract**: Artificial intelligence (AI) holds significant promise for enhancing intraoperative perception and decision-making in telesurgery, where physical separation impairs sensory feedback and control. Despite advances in medical AI and surgical robotics, conventional electronic AI architectures remain fundamentally constrained by the compounded latency from serial processing of inference and communication. This limitation is especially critical in latency-sensitive procedures such as endovascular interventions, where delays over 200 ms can compromise real-time AI reliability and patient safety. Here, we introduce an Optical Computation-in-Communication (OCiC) framework that reduces end-to-end latency significantly by performing AI inference concurrently with optical communication. OCiC integrates Optical Remote Computing Units (ORCUs) directly into the optical communication pathway, with each ORCU experimentally achieving up to 69 tera-operations per second per channel through spectrally efficient two-dimensional photonic convolution. The system maintains ultrahigh inference fidelity within 0.1% of CPU/GPU baselines on classification and coronary angiography segmentation, while intrinsically mitigating cumulative error propagation, a longstanding barrier to deep optical network scalability. We validated the robustness of OCiC through outdoor dark fibre deployments, confirming consistent and stable performance across varying environmental conditions. When scaled globally, OCiC transforms long-haul fibre infrastructure into a distributed photonic AI fabric with exascale potential, enabling reliable, low-latency telesurgery across distances up to 10,000 km and opening a new optical frontier for distributed medical intelligence. 

**Abstract (ZH)**: 光学计算在通信中的框架（OCiC）：显著降低端到端延迟以增强远程手术中的感知和决策 

---
# Cyber-Resilient System Identification for Power Grid through Bayesian Integration 

**Title (ZH)**: 通过贝叶斯集成实现电力系统的 cyber-韧性识别 

**Authors**: Shimiao Li, Guannan Qu, Bryan Hooi, Vyas Sekar, Soummya Kar, Larry Pileggi  

**Link**: [PDF](https://arxiv.org/pdf/2510.14043)  

**Abstract**: Power grids increasingly need real-time situational awareness under the ever-evolving cyberthreat landscape. Advances in snapshot-based system identification approaches have enabled accurately estimating states and topology from a snapshot of measurement data, under random bad data and topology errors. However, modern interactive, targeted false data can stay undetectable to these methods, and significantly compromise estimation accuracy. This work advances system identification that combines snapshot-based method with time-series model via Bayesian Integration, to advance cyber resiliency against both random and targeted false data. Using a distance-based time-series model, this work can leverage historical data of different distributions induced by changes in grid topology and other settings. The normal system behavior captured from historical data is integrated into system identification through a Bayesian treatment, to make solutions robust to targeted false data. We experiment on mixed random anomalies (bad data, topology error) and targeted false data injection attack (FDIA) to demonstrate our method's 1) cyber resilience: achieving over 70% reduction in estimation error under FDIA; 2) anomalous data identification: being able to alarm and locate anomalous data; 3) almost linear scalability: achieving comparable speed with the snapshot-based baseline, both taking <1min per time tick on the large 2,383-bus system using a laptop CPU. 

**Abstract (ZH)**: 基于贝叶斯整合的混合时间序列模型在电力系统识别中的应用：增强应对随机与 targeted 缺失数据的网络安全韧性 

---
# Context-Selective State Space Models: Feedback is All You Need 

**Title (ZH)**: 面向上下文的选择性状态空间模型：反馈即所必需 

**Authors**: Riccardo Zattra, Giacomo Baggio, Umberto Casti, Augusto Ferrante, Francesco Ticozzi  

**Link**: [PDF](https://arxiv.org/pdf/2510.14027)  

**Abstract**: Transformers, powered by the attention mechanism, are the backbone of most foundation models, yet they suffer from quadratic complexity and difficulties in dealing with long-range dependencies in the input sequence. Recent work has shown that state space models (SSMs) provide an efficient alternative, with the S6 module at the core of the Mamba architecture achieving state-of-the-art results on long-sequence benchmarks. In this paper, we introduce the COFFEE (COntext From FEEdback) model, a novel time-varying SSM that incorporates state feedback to enable context-dependent selectivity, while still allowing for parallel implementation. Whereas the selectivity mechanism of S6 only depends on the current input, COFFEE computes it from the internal state, which serves as a compact representation of the sequence history. This shift allows the model to regulate its dynamics based on accumulated context, improving its ability to capture long-range dependencies. In addition to state feedback, we employ an efficient model parametrization that removes redundancies present in S6 and leads to a more compact and trainable formulation. On the induction head task, COFFEE achieves near-perfect accuracy with two orders of magnitude fewer parameters and training sequences compared to S6. On MNIST, COFFEE largely outperforms S6 within the same architecture, reaching 97% accuracy with only 3585 parameters. These results showcase the role of state feedback as a key mechanism for building scalable and efficient sequence models. 

**Abstract (ZH)**: 基于反馈的COFFEE模型：一种新颖的时间varying状态空间模型 

---
# Conditional Clifford-Steerable CNNs with Complete Kernel Basis for PDE Modeling 

**Title (ZH)**: 基于完备核基的条件克利福德-可 steering CNNs 用于偏微分方程建模 

**Authors**: Bálint László Szarvas, Maksim Zhdanov  

**Link**: [PDF](https://arxiv.org/pdf/2510.14007)  

**Abstract**: Clifford-Steerable CNNs (CSCNNs) provide a unified framework that allows incorporating equivariance to arbitrary pseudo-Euclidean groups, including isometries of Euclidean space and Minkowski spacetime. In this work, we demonstrate that the kernel basis of CSCNNs is not complete, thus limiting the model expressivity. To address this issue, we propose Conditional Clifford-Steerable Kernels, which augment the kernels with equivariant representations computed from the input feature field. We derive the equivariance constraint for these input-dependent kernels and show how it can be solved efficiently via implicit parameterization. We empirically demonstrate an improved expressivity of the resulting framework on multiple PDE forecasting tasks, including fluid dynamics and relativistic electrodynamics, where our method consistently outperforms baseline methods. 

**Abstract (ZH)**: Clifford-可引导CNNs (CSCNNs) 提供了一个统一框架，使得可以包含任意伪欧几里得群的协变性，包括欧几里得空间的等距变换和闵可夫斯基时空的等距变换。在本文中，我们证明了CSCNNs的核基不完整，从而限制了模型的表征能力。为解决这一问题，我们提出了条件Clifford-可引导核，通过从输入特征场计算协变表示来增强核。我们推导了这些输入依赖核的协变约束，并展示了如何通过隐式参数化高效求解。我们通过多个偏微分方程预测任务的实验表明，该方法在流体动力学和相对论电磁动力学领域的一致上优于基线方法，改善了模型的表征能力。 

---
# Finding Holes: Pathologist Level Performance Using AI for Cribriform Morphology Detection in Prostate Cancer 

**Title (ZH)**: 寻找漏洞：使用AI检测前列腺癌 cribriform形态的人类病理学家水平性能 

**Authors**: Kelvin Szolnoky, Anders Blilie, Nita Mulliqi, Toyonori Tsuzuki, Hemamali Samaratunga, Matteo Titus, Xiaoyi Ji, Sol Erika Boman, Einar Gudlaugsson, Svein Reidar Kjosavik, José Asenjo, Marcello Gambacorta, Paolo Libretti, Marcin Braun, Radisław Kordek, Roman Łowicki, Brett Delahunt, Kenneth A. Iczkowski, Theo van der Kwast, Geert J. L. H. van Leenders, Katia R. M. Leite, Chin-Chen Pan, Emiel Adrianus Maria Janssen, Martin Eklund, Lars Egevad, Kimmo Kartasalo  

**Link**: [PDF](https://arxiv.org/pdf/2510.13995)  

**Abstract**: Background: Cribriform morphology in prostate cancer is a histological feature that indicates poor prognosis and contraindicates active surveillance. However, it remains underreported and subject to significant interobserver variability amongst pathologists. We aimed to develop and validate an AI-based system to improve cribriform pattern detection.
Methods: We created a deep learning model using an EfficientNetV2-S encoder with multiple instance learning for end-to-end whole-slide classification. The model was trained on 640 digitised prostate core needle biopsies from 430 patients, collected across three cohorts. It was validated internally (261 slides from 171 patients) and externally (266 slides, 104 patients from three independent cohorts). Internal validation cohorts included laboratories or scanners from the development set, while external cohorts used completely independent instruments and laboratories. Annotations were provided by three expert uropathologists with known high concordance. Additionally, we conducted an inter-rater analysis and compared the model's performance against nine expert uropathologists on 88 slides from the internal validation cohort.
Results: The model showed strong internal validation performance (AUC: 0.97, 95% CI: 0.95-0.99; Cohen's kappa: 0.81, 95% CI: 0.72-0.89) and robust external validation (AUC: 0.90, 95% CI: 0.86-0.93; Cohen's kappa: 0.55, 95% CI: 0.45-0.64). In our inter-rater analysis, the model achieved the highest average agreement (Cohen's kappa: 0.66, 95% CI: 0.57-0.74), outperforming all nine pathologists whose Cohen's kappas ranged from 0.35 to 0.62.
Conclusion: Our AI model demonstrates pathologist-level performance for cribriform morphology detection in prostate cancer. This approach could enhance diagnostic reliability, standardise reporting, and improve treatment decisions for prostate cancer patients. 

**Abstract (ZH)**: 背景：cribriform形态在前列腺癌中是一种预后不良的组织学特征，且提示不宜进行主动监测。然而，该特征在病理学家之间报道不足且存在显著的主观差异。我们旨在开发并验证一种基于AI的系统以提高cribriform模式检测的准确性。 

---
# Synthesizing Agentic Data for Web Agents with Progressive Difficulty Enhancement Mechanisms 

**Title (ZH)**: 使用渐进难度增强机制合成代理数据以供网络代理使用 

**Authors**: Shrey Pandit, Xuan-Phi Nguyen, Yifei Ming, Austin Xu, Jiayu Wang, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2510.13913)  

**Abstract**: Web-based 'deep research' agents aim to solve complex question - answering tasks through long-horizon interactions with online tools. These tasks remain challenging, as the underlying language models are often not optimized for long-horizon reasoning and exploration. Prior work has proposed workflows for constructing instruction-tuning datasets, often leveraging knowledge graphs. However, such methods typically lack fine-grained control over difficulty and quality, yielding synthetic data that falls short of capturing the complexity required for long-horizon reasoning. Furthermore, many studies conflate data and training effects by comparing models trained under different optimization recipes, making it difficult to isolate and evaluate the effectiveness of the data itself. We introduce a two-pronged data synthesis pipeline that generates question - answer pairs by progressively increasing task complexity until a frontier baseline web agent fails. The baseline agent plays multiple roles in this process: attempting the questions, validating factuality, checking for alternative answers, and enforcing filtering. To evaluate the effectiveness of our synthesis methods, we adopt a controlled training setup based on distillation from strong web agents. Experiments across multiple web-based benchmarks show that our dataset - despite being smaller - enables the training of more effective web agents than existing datasets. In particular, our data exhibits twice the diversity in tool-use actions, allowing models trained on it to achieve stronger performance while avoiding repetitive tool-calling behaviors. 

**Abstract (ZH)**: 基于Web的“深度研究”代理旨在通过与在线工具进行长期交互来解决复杂的问答任务。尽管这些任务具有挑战性，因为底层语言模型往往未优化用于长期推理和探索。先前的工作提出了一些构建指令调优数据集的方法，通常利用知识图谱。然而，这些方法通常缺乏对难度和质量的精细控制，生成的数据未能捕捉到长期推理所需的复杂性。此外，许多研究通过比较在不同优化配方下训练的模型来混淆数据和训练效果，使得难以隔离和评估数据本身的效用。我们提出了一种双管齐下的数据合成管道，通过逐步增加任务复杂性，直至基准Web代理失败来生成问题-答案对。基准代理在这个过程中扮演多重角色：尝试回答问题、验证事实性、检查替代答案并执行过滤。为了评估我们合成方法的有效性，我们采用了基于强Web代理蒸馏的受控训练设置。在多个基于Web的基准测试中进行的实验显示，尽管我们的数据集规模较小，但仍能使训练出的Web代理比现有数据集更有效。特别地，我们的数据展示了使用工具行为两倍的多样性，使在此类数据上训练的模型能够实现更优性能，同时避免重复调用工具的行为。 

---
# Benefits and Limitations of Communication in Multi-Agent Reasoning 

**Title (ZH)**: 多智能体推理中通信的利与弊 

**Authors**: Michael Rizvi-Martel, Satwik Bhattamishra, Neil Rathi, Guillaume Rabusseau, Michael Hahn  

**Link**: [PDF](https://arxiv.org/pdf/2510.13903)  

**Abstract**: Chain-of-thought prompting has popularized step-by-step reasoning in large language models, yet model performance still degrades as problem complexity and context length grow. By decomposing difficult tasks with long contexts into shorter, manageable ones, recent multi-agent paradigms offer a promising near-term solution to this problem. However, the fundamental capacities of such systems are poorly understood. In this work, we propose a theoretical framework to analyze the expressivity of multi-agent systems. We apply our framework to three algorithmic families: state tracking, recall, and $k$-hop reasoning. We derive bounds on (i) the number of agents required to solve the task exactly, (ii) the quantity and structure of inter-agent communication, and (iii) the achievable speedups as problem size and context scale. Our results identify regimes where communication is provably beneficial, delineate tradeoffs between agent count and bandwidth, and expose intrinsic limitations when either resource is constrained. We complement our theoretical analysis with a set of experiments on pretrained LLMs using controlled synthetic benchmarks. Empirical outcomes confirm the tradeoffs between key quantities predicted by our theory. Collectively, our analysis offers principled guidance for designing scalable multi-agent reasoning systems. 

**Abstract (ZH)**: 链式思维提示在大型语言模型中普及了逐步推理，但随着问题复杂性和上下文长度的增长，模型性能仍然会下降。通过将具有长期上下文的困难任务分解为更短、更易管理的任务，最近的多智能体范式为解决这一问题提供了有希望的短期解决方案。然而，这类系统的根本能力还知之甚少。在本文中，我们提出了一种理论框架来分析多智能体系统的表达能力。我们将该框架应用于三种算法家族：状态跟踪、回忆和$k$-跳推理。我们推导出了关于(i)完成任务所需的智能体数量、(ii) 交互智能体间通信的数量和结构，以及(iii) 随问题规模和上下文扩展可实现的加速比的上界。我们的结果明确了通信在可证明有益的区间、智能体数量与带宽之间的权衡，并在任一资源受限时揭示内在限制。我们通过使用受控合成基准对预训练的大语言模型进行实验，补充了我们的理论分析。实证结果证实了我们理论预测的关键量之间的权衡。综上所述，我们的分析为设计可扩展的多智能体推理系统提供了原则性指导。 

---
# Bayes or Heisenberg: Who(se) Rules? 

**Title (ZH)**: 贝叶斯还是海森堡：谁（的规则）？ 

**Authors**: Volker Tresp Hang Li, Federico Harjes, Yunpu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.13894)  

**Abstract**: Although quantum systems are generally described by quantum state vectors, we show that in certain cases their measurement processes can be reformulated as probabilistic equations expressed in terms of probabilistic state vectors. These probabilistic representations can, in turn, be approximated by the neural network dynamics of the Tensor Brain (TB) model.
The Tensor Brain is a recently proposed framework for modeling perception and memory in the brain, providing a biologically inspired mechanism for efficiently integrating generated symbolic representations into reasoning processes. 

**Abstract (ZH)**: 尽管量子系统一般由量子状态矢量描述，但我们展示了在某些情况下，其测量过程可以重新表述为用概率状态矢量表示的概率方程。这些概率表示可以近似为Tensor Brain (TB)模型中的神经网络动力学。Tensor Brain是一种 recently 提出的框架，用于模型大脑的感知和记忆，提供了一种受生物学启发的机制，以高效地将生成的符号表示集成到推理过程中。 

---
# K-frames: Scene-Driven Any-k Keyframe Selection for long video understanding 

**Title (ZH)**: K-框架：场景驱动的任意k关键帧选择用于长视频理解 

**Authors**: Yifeng Yao, Yike Yun, Jing Wang, Huishuai Zhang, Dongyan Zhao, Ke Tian, Zhihao Wang, Minghui Qiu, Tao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13891)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated significant capabilities in image understanding, but long-video are constrained by context windows and computational cost. Uniform frame sampling often leads to substantial information loss. Meanwhile existing keyframe selection methods such as text-frame retrieval or RL-based frame optimization typically yield sparse and temporally disjointed frames, overlooking scene continuity and lacking flexibility for multi-scale frame selection. To address these limitations, we introduce K-frames, a novel paradigm for scene-driven keyframe selection that preserves temporal continuity. Instead of selecting individual frames, K-frames predicts semantically coherent, query-relevant clips, which enables any-k keyframes selection to meet diverse user budgets. To achieve this approach, we first introduce PeakClips, a dataset of 200K video highlights conditioned by query. Building on this dataset, K-frames learns clip2frame selection using a three-stage progressive curriculum. It involves two Supervised Fine-Tuning stages for temporal grounding and key-clip perception, followed by a Reinforcement Learning stage that directly optimizes the scene-driven prediction policy for downstream task without further annotations. Extensive experiments on major long-video understanding benchmarks demonstrate that K-frames provides an effective, interpretable, and plug-and-play solution for keyframe selection at various scales. Our dataset and model will be available. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在图像理解方面展示了显著的能力，但长视频受限于上下文窗口和计算成本。均匀帧抽样往往导致信息丢失严重。同时，现有的关键帧选择方法如基于文本的帧检索或基于RL的帧优化通常会选取稀疏且时空断开的帧，忽视了场景连续性并缺乏多尺度帧选择的灵活性。为了解决这些局限，我们提出了K-帧，这是一种基于场景的关键帧选择新范式，能够保持时间连续性。K-帧通过预测语义一致、查询相关的片段来代替选择单个帧，使得任何数量的关键帧都能满足不同用户的需求。为了实现这一方法，我们首先引入了PeakClips数据集，该数据集包含20万条查询条件下的视频高光片段。在此数据集的基础上，K-帧通过三阶段逐步课程学习片段到帧的选择。第一阶段是对齐时间和关键片段感知的两阶段监督微调，第二阶段是直接优化基于场景的预测策略的强化学习阶段，无需进一步标注。在主要的长视频理解基准上的广泛实验表明，K-帧提供了一种有效、可解释且即插即用的关键帧选择解决方案。我们的数据集和模型将可供使用。 

---
# Incomplete Multi-view Clustering via Hierarchical Semantic Alignment and Cooperative Completion 

**Title (ZH)**: 基于分层语义对齐与协同完成的不完整多视图聚类 

**Authors**: Xiaojian Ding, Lin Zhao, Xian Li, Xiaoying Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13887)  

**Abstract**: Incomplete multi-view data, where certain views are entirely missing for some samples, poses significant challenges for traditional multi-view clustering methods. Existing deep incomplete multi-view clustering approaches often rely on static fusion strategies or two-stage pipelines, leading to suboptimal fusion results and error propagation issues. To address these limitations, this paper proposes a novel incomplete multi-view clustering framework based on Hierarchical Semantic Alignment and Cooperative Completion (HSACC). HSACC achieves robust cross-view fusion through a dual-level semantic space design. In the low-level semantic space, consistency alignment is ensured by maximizing mutual information across views. In the high-level semantic space, adaptive view weights are dynamically assigned based on the distributional affinity between individual views and an initial fused representation, followed by weighted fusion to generate a unified global representation. Additionally, HSACC implicitly recovers missing views by projecting aligned latent representations into high-dimensional semantic spaces and jointly optimizes reconstruction and clustering objectives, enabling cooperative learning of completion and clustering. Experimental results demonstrate that HSACC significantly outperforms state-of-the-art methods on five benchmark datasets. Ablation studies validate the effectiveness of the hierarchical alignment and dynamic weighting mechanisms, while parameter analysis confirms the model's robustness to hyperparameter variations. 

**Abstract (ZH)**: 基于层次语义对齐和协同补全的不完备多视图聚类框架 

---
# Physics-Informed autoencoder for DSC-MRI Perfusion post-processing: application to glioma grading 

**Title (ZH)**: 基于物理的自动编码器用于DSC-MRI灌注后处理：胶质瘤分级应用 

**Authors**: Pierre Fayolle, Alexandre Bône, Noëlie Debs, Mathieu Naudin, Pascal Bourdon, Remy Guillevin, David Helbert  

**Link**: [PDF](https://arxiv.org/pdf/2510.13886)  

**Abstract**: DSC-MRI perfusion is a medical imaging technique for diagnosing and prognosing brain tumors and strokes. Its analysis relies on mathematical deconvolution, but noise or motion artifacts in a clinical environment can disrupt this process, leading to incorrect estimate of perfusion parameters. Although deep learning approaches have shown promising results, their calibration typically rely on third-party deconvolution algorithms to generate reference outputs and are bound to reproduce their limitations.
To adress this problem, we propose a physics-informed autoencoder that leverages an analytical model to decode the perfusion parameters and guide the learning of the encoding network. This autoencoder is trained in a self-supervised fashion without any third-party software and its performance is evaluated on a database with glioma patients. Our method shows reliable results for glioma grading in accordance with other well-known deconvolution algorithms despite a lower computation time. It also achieved competitive performance even in the presence of high noise which is critical in a medical environment. 

**Abstract (ZH)**: 基于物理的自编码器在MRI灌注成像中的应用：一种无需第三方软件的自监督学习方法及其在胶质瘤分级中的性能评估 

---
# PAGE: Prompt Augmentation for text Generation Enhancement 

**Title (ZH)**: PAGE: 文本生成增强的提示增强 

**Authors**: Mauro Jose Pacchiotti, Luciana Ballejos, Mariel Ale  

**Link**: [PDF](https://arxiv.org/pdf/2510.13880)  

**Abstract**: In recent years, natural language generative models have shown outstanding performance in text generation tasks. However, when facing specific tasks or particular requirements, they may exhibit poor performance or require adjustments that demand large amounts of additional data. This work introduces PAGE (Prompt Augmentation for text Generation Enhancement), a framework designed to assist these models through the use of simple auxiliary modules. These modules, lightweight models such as classifiers or extractors, provide inferences from the input text. The output of these auxiliaries is then used to construct an enriched input that improves the quality and controllability of the generation. Unlike other generation-assistance approaches, PAGE does not require auxiliary generative models; instead, it proposes a simpler, modular architecture that is easy to adapt to different tasks. This paper presents the proposal, its components and architecture, and reports a proof of concept in the domain of requirements engineering, where an auxiliary module with a classifier is used to improve the quality of software requirements generation. 

**Abstract (ZH)**: 近年来，自然语言生成模型在文本生成任务中展现了出色的表现。然而，在面对特定任务或特殊要求时，它们可能表现出色不佳，或者需要通过大量额外数据进行调整。本文介绍了PAGE（Prompt Augmentation for Text Generation Enhancement）框架，该框架通过使用简单的辅助模块来辅助这些模型。这些模块，如分类器或提取器等轻量级模型，从输入文本中提供推断。这些辅助模块的输出用于构建增强输入，从而提高生成的质量和可控性。与其它生成辅助方法不同，PAGE 不需要辅助生成模型；它提出了一种更简单、模块化的架构，易于适应不同任务。本文在需求工程领域提出了该方法的提案、组件和架构，并报告了一个概念验证，其中使用了一个分类器辅助模块来提高软件需求生成的质量。 

---
# FRACCO: A gold-standard annotated corpus of oncological entities with ICD-O-3.1 normalisation 

**Title (ZH)**: FRACCO：一种基于ICD-O-3.1规范化的人类恶性肿瘤实体金标准标注语料库 

**Authors**: Johann Pignat, Milena Vucetic, Christophe Gaudet-Blavignac, Jamil Zaghir, Amandine Stettler, Fanny Amrein, Jonatan Bonjour, Jean-Philippe Goldman, Olivier Michielin, Christian Lovis, Mina Bjelogrlic  

**Link**: [PDF](https://arxiv.org/pdf/2510.13873)  

**Abstract**: Developing natural language processing tools for clinical text requires annotated datasets, yet French oncology resources remain scarce. We present FRACCO (FRench Annotated Corpus for Clinical Oncology) an expert-annotated corpus of 1301 synthetic French clinical cases, initially translated from the Spanish CANTEMIST corpus as part of the FRASIMED initiative. Each document is annotated with terms related to morphology, topography, and histologic differentiation, using the International Classification of Diseases for Oncology (ICD-O) as reference. An additional annotation layer captures composite expression-level normalisations that combine multiple ICD-O elements into unified clinical concepts. Annotation quality was ensured through expert review: 1301 texts were manually annotated for entity spans by two domain experts. A total of 71127 ICD-O normalisations were produced through a combination of automated matching and manual validation by a team of five annotators. The final dataset representing 399 unique morphology codes (from 2549 different expressions), 272 topography codes (from 3143 different expressions), and 2043 unique composite expressions (from 11144 different expressions). This dataset provides a reference standard for named entity recognition and concept normalisation in French oncology texts. 

**Abstract (ZH)**: FRASIMED initiative中的FRench Annotated Corpus for Clinical Oncology (FRACCO) 

---
# Joint Discriminative-Generative Modeling via Dual Adversarial Training 

**Title (ZH)**: 双对抗训练下的判别-生成联合建模 

**Authors**: Xuwang Yin, Claire Zhang, Julie Steele, Nir Shavit, Tony T. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13872)  

**Abstract**: Simultaneously achieving robust classification and high-fidelity generative modeling within a single framework presents a significant challenge. Hybrid approaches, such as Joint Energy-Based Models (JEM), interpret classifiers as EBMs but are often limited by the instability and poor sample quality inherent in SGLD-based training. We address these limitations by proposing a novel training framework that integrates adversarial training (AT) principles for both discriminative robustness and stable generative learning. The proposed method introduces three key innovations: (1) the replacement of SGLD-based JEM learning with a stable, AT-based approach that optimizes the energy function by discriminating between real data and PGD-generated contrastive samples using the BCE loss; (2) synergistic adversarial training for the discriminative component that enhances classification robustness while eliminating the need for explicit gradient penalties; and (3) a two-stage training procedure to resolve the incompatibility between batch normalization and EBM training. Experiments on CIFAR-10, CIFAR-100, and ImageNet demonstrate that our method substantially improves adversarial robustness over existing hybrid models while maintaining competitive generative performance. On ImageNet, when optimized for generative modeling, our model's generative fidelity surpasses that of BigGAN and approaches diffusion models, representing the first MCMC-based EBM approach to achieve high-quality generation on complex, high-resolution datasets. Our approach addresses key stability issues that have limited JEM scaling and demonstrates that adversarial training can serve as an effective foundation for unified frameworks capable of generating and robustly classifying visual data. 

**Abstract (ZH)**: 在单一框架中同时实现稳健分类和高保真生成建模是一项重大挑战。通过结合对抗训练原则进行判别稳健性和稳定生成学习的新型训练框架 

---
# CoLoR-GAN: Continual Few-Shot Learning with Low-Rank Adaptation in Generative Adversarial Networks 

**Title (ZH)**: CoLoR-GAN：生成 adversarial 网络中的低秩适应持续少样本学习 

**Authors**: Munsif Ali, Leonardo Rossi, Massimo Bertozzi  

**Link**: [PDF](https://arxiv.org/pdf/2510.13869)  

**Abstract**: Continual learning (CL) in the context of Generative Adversarial Networks (GANs) remains a challenging problem, particularly when it comes to learn from a few-shot (FS) samples without catastrophic forgetting. Current most effective state-of-the-art (SOTA) methods, like LFS-GAN, introduce a non-negligible quantity of new weights at each training iteration, which would become significant when considering the long term. For this reason, this paper introduces \textcolor{red}{\textbf{\underline{c}}}ontinual few-sh\textcolor{red}{\textbf{\underline{o}}}t learning with \textcolor{red}{\textbf{\underline{lo}}}w-\textcolor{red}{\textbf{\underline{r}}}ank adaptation in GANs named CoLoR-GAN, a framework designed to handle both FS and CL together, leveraging low-rank tensors to efficiently adapt the model to target tasks while reducing even more the number of parameters required. Applying a vanilla LoRA implementation already permitted us to obtain pretty good results. In order to optimize even further the size of the adapters, we challenged LoRA limits introducing a LoRA in LoRA (LLoRA) technique for convolutional layers. Finally, aware of the criticality linked to the choice of the hyperparameters of LoRA, we provide an empirical study to easily find the best ones. We demonstrate the effectiveness of CoLoR-GAN through experiments on several benchmark CL and FS tasks and show that our model is efficient, reaching SOTA performance but with a number of resources enormously reduced. Source code is available on \href{this https URL}{Github. 

**Abstract (ZH)**: 连续学习（CL）在生成对抗网络（GANs）的上下文中，特别是在从少量样本（FS）中学习且不导致灾难性遗忘的情况下，仍是一个具有挑战性的问题。当前最有效的状态最前沿方法，如LFS-GAN，在每次训练迭代中引入了不容忽视的新权重，长期来看这将变得相当重要。出于这个原因，本文介绍了一种名为CoLoR-GAN的框架，该框架通过低秩张量有效地适应目标任务，同时进一步减少所需的参数数量，以处理同时的少量样本和连续学习。通过简单的LoRA实现已经可以获得相当不错的结果。为了进一步优化适配器的大小，我们提出了在卷积层中引入改进的LoRA（LLoRA）技术。最后，鉴于LoRA超参数选择的重要性，我们提供了一个经验研究来轻松找到最优的超参数。通过在多个基准的连续学习（CL）和少量样本（FS）任务上的实验，我们展示了CoLoR-GAN的有效性，该模型在资源使用上大大减少但仍能达到状态最前沿的性能。源代码可在Github上获取。 

---
# FFT-Accelerated Auxiliary Variable MCMC for Fermionic Lattice Models: A Determinant-Free Approach with $O(N\log N)$ Complexity 

**Title (ZH)**: FFT加速辅助变量MCMC方法：基于O(NlogN)复杂性的行列式自由方法 

**Authors**: Deqian Kong, Shi Feng, Jianwen Xie, Ying Nian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13866)  

**Abstract**: We introduce a Markov Chain Monte Carlo (MCMC) algorithm that dramatically accelerates the simulation of quantum many-body systems, a grand challenge in computational science. State-of-the-art methods for these problems are severely limited by $O(N^3)$ computational complexity. Our method avoids this bottleneck, achieving near-linear $O(N \log N)$ scaling per sweep.
Our approach samples a joint probability measure over two coupled variable sets: (1) particle trajectories of the fundamental fermions, and (2) auxiliary variables that decouple fermion interactions. The key innovation is a novel transition kernel for particle trajectories formulated in the Fourier domain, revealing the transition probability as a convolution that enables massive acceleration via the Fast Fourier Transform (FFT). The auxiliary variables admit closed-form, factorized conditional distributions, enabling efficient exact Gibbs sampling update.
We validate our algorithm on benchmark quantum physics problems, accurately reproducing known theoretical results and matching traditional $O(N^3)$ algorithms on $32\times 32$ lattice simulations at a fraction of the wall-clock time, empirically demonstrating $N \log N$ scaling. By reformulating a long-standing physics simulation problem in machine learning language, our work provides a powerful tool for large-scale probabilistic inference and opens avenues for physics-inspired generative models. 

**Abstract (ZH)**: 一种Markov链蒙特卡洛算法，它极大地加速了量子多体系统的模拟，解决了计算科学中的重大挑战。该方法实现了每扫掠近线性O(N log N)的扩展，避免了立方阶的计算复杂性瓶颈。 

---
# Deep Edge Filter: Return of the Human-Crafted Layer in Deep Learning 

**Title (ZH)**: 深度边缘滤波器：回归到深度学习中的手工制作层 

**Authors**: Dongkwan Lee, Junhoo Lee, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2510.13865)  

**Abstract**: We introduce the Deep Edge Filter, a novel approach that applies high-pass filtering to deep neural network features to improve model generalizability. Our method is motivated by our hypothesis that neural networks encode task-relevant semantic information in high-frequency components while storing domain-specific biases in low-frequency components of deep features. By subtracting low-pass filtered outputs from original features, our approach isolates generalizable representations while preserving architectural integrity. Experimental results across diverse domains such as Vision, Text, 3D, and Audio demonstrate consistent performance improvements regardless of model architecture and data modality. Analysis reveals that our method induces feature sparsification and effectively isolates high-frequency components, providing empirical validation of our core hypothesis. The code is available at this https URL. 

**Abstract (ZH)**: 我们介绍了深度边缘滤波器，这是一种通过高通滤波深神经网络特征以提高模型泛化能力的新方法。该方法基于我们假设神经网络在深度特征的高频分量中编码任务相关语义信息，在低频分量中存储领域特定的偏见的假设。通过从原始特征中减去低通滤波输出，该方法分离出可泛化表示的同时保持网络架构的完整性。在视觉、文本、三维和音频等多个领域进行的实验结果表明，无论模型架构和数据模态如何，该方法都能一致地提高性能。分析表明，该方法引发特征稀疏化，并有效地分离出高频分量，提供了对核心假设的经验验证。代码可在以下网址获取。 

---
# Self-Training with Dynamic Weighting for Robust Gradual Domain Adaptation 

**Title (ZH)**: 动态加权的自我训练在鲁棒渐进域适应中的应用 

**Authors**: Zixi Wang, Yushe Cao, Yubo Huang, Jinzhu Wei, Jingzehua Xu, Shuai Zhang, Xin Lai  

**Link**: [PDF](https://arxiv.org/pdf/2510.13864)  

**Abstract**: In this paper, we propose a new method called Self-Training with Dynamic Weighting (STDW), which aims to enhance robustness in Gradual Domain Adaptation (GDA) by addressing the challenge of smooth knowledge migration from the source to the target domain. Traditional GDA methods mitigate domain shift through intermediate domains and self-training but often suffer from inefficient knowledge migration or incomplete intermediate data. Our approach introduces a dynamic weighting mechanism that adaptively balances the loss contributions of the source and target domains during training. Specifically, we design an optimization framework governed by a time-varying hyperparameter $\varrho$ (progressing from 0 to 1), which controls the strength of domain-specific learning and ensures stable adaptation. The method leverages self-training to generate pseudo-labels and optimizes a weighted objective function for iterative model updates, maintaining robustness across intermediate domains. Experiments on rotated MNIST, color-shifted MNIST, portrait datasets, and the Cover Type dataset demonstrate that STDW outperforms existing baselines. Ablation studies further validate the critical role of $\varrho$'s dynamic scheduling in achieving progressive adaptation, confirming its effectiveness in reducing domain bias and improving generalization. This work provides both theoretical insights and a practical framework for robust gradual domain adaptation, with potential applications in dynamic real-world scenarios. The code is available at this https URL. 

**Abstract (ZH)**: 自适应加权自我训练方法在渐进域适应中的应用：一种自我训练动态加权方法（Self-Training with Dynamic Weighting for Robust Gradual Domain Adaptation） 

---
# Information flow in multilayer perceptrons: an in-depth analysis 

**Title (ZH)**: 多层感知器中的信息流：深入分析 

**Authors**: Giuliano Armano  

**Link**: [PDF](https://arxiv.org/pdf/2510.13846)  

**Abstract**: Analysing how information flows along the layers of a multilayer perceptron is a topic of paramount importance in the field of artificial neural networks. After framing the problem from the point of view of information theory, in this position article a specific investigation is conducted on the way information is processed, with particular reference to the requirements imposed by supervised learning. To this end, the concept of information matrix is devised and then used as formal framework for understanding the aetiology of optimisation strategies and for studying the information flow. The underlying research for this article has also produced several key outcomes: i) the definition of a parametric optimisation strategy, ii) the finding that the optimisation strategy proposed in the information bottleneck framework shares strong similarities with the one derived from the information matrix, and iii) the insight that a multilayer perceptron serves as a kind of "adaptor", meant to process the input according to the given objective. 

**Abstract (ZH)**: 分析多层感知机各层间信息流变对人工神经网络领域至关重要。从信息论的角度界定问题后，本文针对信息处理方式进行具体调查，特别关注监督学习提出的要求。为此，我们提出了信息矩阵的概念，并将其用作理解优化策略起源和研究信息流的形式框架。本文的研究还产生了多个关键成果：i) 定义了参数化优化策略，ii) 发现信息瓶颈框架中提出的优化策略与从信息矩阵中推导出的优化策略具有很强的相似性，iii) 洞察到多层感知机充当了一种“适配器”，旨在根据给定的目标处理输入。 

---
# Serialized EHR make for good text representations 

**Title (ZH)**: 序列化的电子健康记录-make for good text representations 

**Authors**: Zhirong Chou, Quan Qin, Shi Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.13843)  

**Abstract**: The emergence of foundation models in healthcare has opened new avenues for learning generalizable representations from large scale clinical data. Yet, existing approaches often struggle to reconcile the tabular and event based nature of Electronic Health Records (EHRs) with the sequential priors of natural language models. This structural mismatch limits their ability to capture longitudinal dependencies across patient encounters. We introduce SerialBEHRT, a domain aligned foundation model that extends SciBERT through additional pretraining on structured EHR sequences. SerialBEHRT is designed to encode temporal and contextual relationships among clinical events, thereby producing richer patient representations. We evaluate its effectiveness on the task of antibiotic susceptibility prediction, a clinically meaningful problem in antibiotic stewardship. Through extensive benchmarking against state of the art EHR representation strategies, we demonstrate that SerialBEHRT achieves superior and more consistent performance, highlighting the importance of temporal serialization in foundation model pretraining for healthcare. 

**Abstract (ZH)**: 基础模型在医疗领域的出现为从大规模临床数据中学习可泛化的表示开辟了新的途径。然而，现有方法往往难以调和电子健康记录(EHRs)的表格式和事件驱动性质与自然语言模型的序列先验之间的结构性不匹配。这种结构性不匹配限制了它们捕捉患者就诊间纵向依赖性的能力。我们提出了一种领域对齐的基础模型SerialBEHRT，通过额外对结构化EHR序列进行预训练，扩展了SciBERT。SerialBEHRT旨在编码临床事件之间的时空关系，从而生成 richer 的患者表示。我们将其效果评估应用于抗生素敏感性预测任务，这是抗生素管理中的一个临床相关问题。通过与最先进的EHR表示策略的广泛基准测试，我们证明SerialBEHRT实现了更优且更一致的性能，强调了在医疗领域对基础模型进行时间序列化预训练的重要性。 

---
# Seeing Hate Differently: Hate Subspace Modeling for Culture-Aware Hate Speech Detection 

**Title (ZH)**: 从不同视角看待仇恨：基于文化意识的仇恨言辞检测的仇恨子空间建模 

**Authors**: Weibin Cai, Reza Zafarani  

**Link**: [PDF](https://arxiv.org/pdf/2510.13837)  

**Abstract**: Hate speech detection has been extensively studied, yet existing methods often overlook a real-world complexity: training labels are biased, and interpretations of what is considered hate vary across individuals with different cultural backgrounds. We first analyze these challenges, including data sparsity, cultural entanglement, and ambiguous labeling. To address them, we propose a culture-aware framework that constructs individuals' hate subspaces. To alleviate data sparsity, we model combinations of cultural attributes. For cultural entanglement and ambiguous labels, we use label propagation to capture distinctive features of each combination. Finally, individual hate subspaces, which in turn can further enhance classification performance. Experiments show our method outperforms state-of-the-art by 1.05\% on average across all metrics. 

**Abstract (ZH)**: 含有 Hate 言论检测的研究已十分广泛，但现有方法往往忽略了现实世界的复杂性：训练标签存有偏见，不同文化背景的人对什么是 Hate 的理解也各不相同。我们首先分析这些挑战，包括数据稀疏性、文化纠缠和模糊标签。为此，我们提出了一种文化意识框架，构建个体的 Hate 子空间。为缓解数据稀疏性，我们建模文化属性的组合。针对文化纠缠和模糊标签，我们使用标签传播来捕捉每种组合的独特特征。最终，个体的 Hate 子空间可以进一步提升分类性能。实验结果显示，我们的方法在所有指标上的平均性能比现有最佳方法高出 1.05%。 

---
# Entropy Meets Importance: A Unified Head Importance-Entropy Score for Stable and Efficient Transformer Pruning 

**Title (ZH)**: 熵遇重要性：统一的头重要性-熵得分方法以实现Transformer剪枝的稳定性和高效性 

**Authors**: Minsik Choi, Hyegang Son, Changhoon Kim, Young Geun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.13832)  

**Abstract**: Transformer-based models have achieved remarkable performance in NLP tasks. However, their structural characteristics-multiple layers and attention heads-introduce efficiency challenges in inference and deployment. To address these challenges, various pruning methods have recently been proposed. Notably, gradient-based methods using Head Importance Scores (HIS) have gained traction for interpretability, efficiency, and ability to identify redundant heads. However, HIS alone has limitations as it captures only the gradient-driven contribution, overlooking the diversity of attention patterns. To overcome these limitations, we introduce a novel pruning criterion, HIES (Head Importance-Entropy Score), which integrates head importance scores with attention entropy, providing complementary evidence on per-head contribution. Empirically, HIES-based pruning yields up to 15.2% improvement in model quality and 2.04x improvement in stability over HIS-only methods, enabling substantial model compression without sacrificing either accuracy or stability. Code will be released upon publication. 

**Abstract (ZH)**: 基于Transformer的模型在自然语言处理任务中取得了显著性能。然而，其结构特性——多层和注意力头——在推理和部署中引入了效率挑战。为应对这些挑战，最近提出了多种剪枝方法。值得注意的是，基于梯度的方法利用头重要性得分（HIS）因其实用性、效率以及识别冗余头的能力而受到关注。然而，HIS单独使用有局限性，因为它只捕捉了梯度驱动的贡献，忽略了注意力模式的多样性。为克服这些局限，我们提出了一种新的剪枝标准HIES（头重要性-熵得分），它将头重要性得分与注意力熵相结合，提供了头贡献的补充证据。实验证明，基于HIES的剪枝方法相比仅使用HIS的方法，在模型质量上提高了15.2%，在稳定性上提高了2.04倍，能够在不牺牲准确性和稳定性的前提下实现大规模模型压缩。代码将在发表后开源。 

---
# Leveraging Wireless Sensor Networks for Real-Time Monitoring and Control of Industrial Environments 

**Title (ZH)**: 利用无线传感器网络实现工业环境的实时监测与控制 

**Authors**: Muhammad Junaid Asif, Shazia Saqib, Rana Fayyaz Ahmad, Hamza Khan  

**Link**: [PDF](https://arxiv.org/pdf/2510.13820)  

**Abstract**: This research proposes an extensive technique for monitoring and controlling the industrial parameters using Internet of Things (IoT) technology based on wireless communication. We proposed a system based on NRF transceivers to establish a strong Wireless Sensor Network (WSN), enabling transfer of real-time data from multiple sensors to a central setup that is driven by ARDUINO microcontrollers. Different key parameters, crucial for industrial setup such as temperature, humidity, soil moisture and fire detection, are monitored and displayed on an LCD screen, enabling factory administration to oversee the industrial operations remotely over the internet. Our proposed system bypasses the need for physical presence for monitoring by addressing the shortcomings of conventional wired communication systems. Other than monitoring, there is an additional feature to remotely control these parameters by controlling the speed of DC motors through online commands. Given the rising incidence of industrial fires over the worldwide between 2020 and 2024 due to an array of hazards, this system with dual functionality boosts the overall operational efficiency and safety. This overall integration of IoT and Wireless Sensor Network (WSN) reduces the potential risks linked with physical monitoring, providing rapid responses in emergency scenarios, including the activation of firefighting equipment. The results show that innovations in wireless communication perform an integral part in industrial process automation and safety, paving the way to more intelligent and responsive operating environments. Overall, this study highlights the potential for change of IoT-enabled systems to revolutionize monitoring and control in a variety of industrial applications, resulting in increased productivity and safety. 

**Abstract (ZH)**: 基于物联网技术的无线通信工业参数监控与控制方法研究 

---
# GQVis: A Dataset of Genomics Data Questions and Visualizations for Generative AI 

**Title (ZH)**: GQVis: 用于生成式AI的基因组数据问题和可视化数据集 

**Authors**: Skylar Sargent Walters, Arthea Valderrama, Thomas C. Smits, David Kouřil, Huyen N. Nguyen, Sehi L'Yi, Devin Lange, Nils Gehlenborg  

**Link**: [PDF](https://arxiv.org/pdf/2510.13816)  

**Abstract**: Data visualization is a fundamental tool in genomics research, enabling the exploration, interpretation, and communication of complex genomic features. While machine learning models show promise for transforming data into insightful visualizations, current models lack the training foundation for domain-specific tasks. In an effort to provide a foundational resource for genomics-focused model training, we present a framework for generating a dataset that pairs abstract, low-level questions about genomics data with corresponding visualizations. Building on prior work with statistical plots, our approach adapts to the complexity of genomics data and the specialized representations used to depict them. We further incorporate multiple linked queries and visualizations, along with justifications for design choices, figure captions, and image alt-texts for each item in the dataset. We use genomics data retrieved from three distinct genomics data repositories (4DN, ENCODE, Chromoscope) to produce GQVis: a dataset consisting of 1.14 million single-query data points, 628k query pairs, and 589k query chains. The GQVis dataset and generation code are available at this https URL and this https URL. 

**Abstract (ZH)**: 基因组学研究中的数据可视化是探索、解释和传达复杂基因组特征的基本工具。虽然机器学习模型有望将数据转化为洞察性的可视化，但当前的模型缺乏针对特定领域任务的训练基础。为提供一个面向基因组学模型训练的基础资源，我们提出了一种生成数据集的框架，该数据集将抽象的、低级别的基因组数据问题与相应的可视化图像配对。在此前统计图工作的基础上，我们的方法适应了基因组数据的复杂性和专门的表示方法。我们进一步将多个链接的问题和可视化图像纳入其中，并为每项数据提供了设计选择的说明、图表标题以及图像替代文本。我们使用从三个不同的基因组数据存储库（4DN、ENCODE、Chromoscope）检索的基因组数据生成了GQVis数据集，该数据集包含114万单查询数据点、62.8万查询对以及58.9万查询链。GQVis数据集及其生成代码可在以下链接获得：this https URL 和 this https URL。 

---
# Reversing the Lens: Using Explainable AI to Understand Human Expertise 

**Title (ZH)**: 反转视角：运用可解释的人工智能理解人类专长 

**Authors**: Roussel Rahman, Aashwin Ananda Mishra, Wan-Lin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13814)  

**Abstract**: Both humans and machine learning models learn from experience, particularly in safety- and reliability-critical domains. While psychology seeks to understand human cognition, the field of Explainable AI (XAI) develops methods to interpret machine learning models. This study bridges these domains by applying computational tools from XAI to analyze human learning. We modeled human behavior during a complex real-world task -- tuning a particle accelerator -- by constructing graphs of operator subtasks. Applying techniques such as community detection and hierarchical clustering to archival operator data, we reveal how operators decompose the problem into simpler components and how these problem-solving structures evolve with expertise. Our findings illuminate how humans develop efficient strategies in the absence of globally optimal solutions, and demonstrate the utility of XAI-based methods for quantitatively studying human cognition. 

**Abstract (ZH)**: 人类和机器学习模型均通过经验学习，尤其在安全和可靠性关键领域。心理学旨在理解人类认知，而可解释人工智能（XAI）领域则发展方法以解释机器学习模型。本研究通过将XAI的计算工具应用到人类学习分析中，将这两个领域结合起来。我们通过构建操作子任务图来模拟复杂实际任务（调谐粒子加速器）中的人类行为。通过对存档的操作员数据应用社区检测和层次聚类等技术，揭示了操作员如何将问题分解为更简单的组件，以及这些解决问题的结构如何随着专业知识的提高而演变。我们的研究结果阐明了在不存在全局最优解的情况下，人类如何发展高效的策略，并展示了基于XAI的方法在定量研究人类认知方面的实用性。 

---
