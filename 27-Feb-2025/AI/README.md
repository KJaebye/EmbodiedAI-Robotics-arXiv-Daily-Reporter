# TheoremExplainAgent: Towards Multimodal Explanations for LLM Theorem Understanding 

**Title (ZH)**: TheoremExplainAgent: 向量多模态定理解释代理 

**Authors**: Max Ku, Thomas Chong, Jonathan Leung, Krish Shah, Alvin Yu, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.19400)  

**Abstract**: Understanding domain-specific theorems often requires more than just text-based reasoning; effective communication through structured visual explanations is crucial for deeper comprehension. While large language models (LLMs) demonstrate strong performance in text-based theorem reasoning, their ability to generate coherent and pedagogically meaningful visual explanations remains an open challenge. In this work, we introduce TheoremExplainAgent, an agentic approach for generating long-form theorem explanation videos (over 5 minutes) using Manim animations. To systematically evaluate multimodal theorem explanations, we propose TheoremExplainBench, a benchmark covering 240 theorems across multiple STEM disciplines, along with 5 automated evaluation metrics. Our results reveal that agentic planning is essential for generating detailed long-form videos, and the o3-mini agent achieves a success rate of 93.8% and an overall score of 0.77. However, our quantitative and qualitative studies show that most of the videos produced exhibit minor issues with visual element layout. Furthermore, multimodal explanations expose deeper reasoning flaws that text-based explanations fail to reveal, highlighting the importance of multimodal explanations. 

**Abstract (ZH)**: 理解领域特定的定理往往需要超越基于文本的推理；有效的结构性视觉解释对于深入理解至关重要。尽管大型语言模型在基于文本的定理推理方面表现出色，但它们生成连贯且有教育意义的视觉解释的能力仍然是一个开放的挑战。本工作中，我们引入了TheoremExplainAgent，这是一种使用Manim动画生成长格式定理解释视频（超过5分钟）的代理方法。为了系统地评估多模态定理解释，我们提出了TheoremExplainBench基准，该基准包含来自多个STEM学科的240个定理，以及5个自动评估指标。我们的结果显示，代理规划对于生成详细的长视频是必不可少的，o3-mini代理的成功率为93.8%，总体得分为0.77。然而，我们的定量和定性研究表明，大多数生成的视频在视觉元素布局方面存在较小的问题。此外，多模态解释揭示了基于文本解释无法揭示的更深层次的推理缺陷，突显了多模态解释的重要性。 

---
# Joint Optimal Transport and Embedding for Network Alignment 

**Title (ZH)**: joint最优运输与嵌入在网络对齐中的联合优化 

**Authors**: Qi Yu, Zhichen Zeng, Yuchen Yan, Lei Ying, R. Srikant, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2502.19334)  

**Abstract**: Network alignment, which aims to find node correspondence across different networks, is the cornerstone of various downstream multi-network and Web mining tasks. Most of the embedding-based methods indirectly model cross-network node relationships by contrasting positive and negative node pairs sampled from hand-crafted strategies, which are vulnerable to graph noises and lead to potential misalignment of nodes. Another line of work based on the optimal transport (OT) theory directly models cross-network node relationships and generates noise-reduced alignments. However, OT methods heavily rely on fixed, pre-defined cost functions that prohibit end-to-end training and are hard to generalize. In this paper, we aim to unify the embedding and OT-based methods in a mutually beneficial manner and propose a joint optimal transport and embedding framework for network alignment named JOENA. For one thing (OT for embedding), through a simple yet effective transformation, the noise-reduced OT mapping serves as an adaptive sampling strategy directly modeling all cross-network node pairs for robust embedding this http URL another (embedding for OT), on top of the learned embeddings, the OT cost can be gradually trained in an end-to-end fashion, which further enhances the alignment quality. With a unified objective, the mutual benefits of both methods can be achieved by an alternating optimization schema with guaranteed convergence. Extensive experiments on real-world networks validate the effectiveness and scalability of JOENA, achieving up to 16% improvement in MRR and 20x speedup compared with the state-of-the-art alignment methods. 

**Abstract (ZH)**: 网络对齐，旨在跨不同网络发现节点对应关系，是各种下游多网络和Web挖掘任务的基础。大多数基于嵌入的方法通过对比手工构建策略采样的正负节点对间接建模跨网络节点关系，这使其容易受到图噪声的影响并可能导致节点潜在的对齐错位。另一类基于最优传输理论的方法直接建模跨网络节点关系并生成噪声减少的对齐结果。然而，最优传输方法严重依赖固定的先验成本函数，这阻碍了端到端训练并难以通用。本文旨在以互惠互利的方式统一嵌入和最优传输方法，并提出了一种名为JOENA的联合最优传输和嵌入网络对齐框架。对于一方面（嵌入促进最优传输），通过简单的有效变换，噪声减少的最优传输映射作为自适应采样策略，直接建模所有跨网络节点对以实现稳健嵌入；另一方面（最优传输促进嵌入），在学习到的嵌入之上，最优传输成本可以逐次进行端到端训练，从而进一步提高对齐质量。通过统一目标，交替优化方案可以保证收敛并实现两种方法的互惠互利。大规模实验在真实世界网络上的验证结果表明，JOENA在广泛性和效性方面均优于最先进的对齐方法，MRR提高最多可达16%，并且速度提升20倍。 

---
# WOFOSTGym: A Crop Simulator for Learning Annual and Perennial Crop Management Strategies 

**Title (ZH)**: WOFOSTGym: 作物模拟器用于学习年度与多年生作物管理策略 

**Authors**: William Solow, Sandhya Saisubramanian, Alan Fern  

**Link**: [PDF](https://arxiv.org/pdf/2502.19308)  

**Abstract**: We introduce WOFOSTGym, a novel crop simulation environment designed to train reinforcement learning (RL) agents to optimize agromanagement decisions for annual and perennial crops in single and multi-farm settings. Effective crop management requires optimizing yield and economic returns while minimizing environmental impact, a complex sequential decision-making problem well suited for RL. However, the lack of simulators for perennial crops in multi-farm contexts has hindered RL applications in this domain. Existing crop simulators also do not support multiple annual crops. WOFOSTGym addresses these gaps by supporting 23 annual crops and two perennial crops, enabling RL agents to learn diverse agromanagement strategies in multi-year, multi-crop, and multi-farm settings. Our simulator offers a suite of challenging tasks for learning under partial observability, non-Markovian dynamics, and delayed feedback. WOFOSTGym's standard RL interface allows researchers without agricultural expertise to explore a wide range of agromanagement problems. Our experiments demonstrate the learned behaviors across various crop varieties and soil types, highlighting WOFOSTGym's potential for advancing RL-driven decision support in agriculture. 

**Abstract (ZH)**: WOFOSTGym：一种用于训练强化学习代理优化农管理决策的新型作物模拟环境 

---
# Complex LLM Planning via Automated Heuristics Discovery 

**Title (ZH)**: 通过自动启发式发现进行的复杂LLM规划 

**Authors**: Hongyi Ling, Shubham Parashar, Sambhav Khurana, Blake Olson, Anwesha Basu, Gaurangi Sinha, Zhengzhong Tu, James Caverlee, Shuiwang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2502.19295)  

**Abstract**: We consider enhancing large language models (LLMs) for complex planning tasks. While existing methods allow LLMs to explore intermediate steps to make plans, they either depend on unreliable self-verification or external verifiers to evaluate these steps, which demand significant data and computations. Here, we propose automated heuristics discovery (AutoHD), a novel approach that enables LLMs to explicitly generate heuristic functions to guide inference-time search, allowing accurate evaluation of intermediate states. These heuristic functions are further refined through a heuristic evolution process, improving their robustness and effectiveness. Our proposed method requires no additional model training or fine-tuning, and the explicit definition of heuristic functions generated by the LLMs provides interpretability and insights into the reasoning process. Extensive experiments across diverse benchmarks demonstrate significant gains over multiple baselines, including nearly twice the accuracy on some datasets, establishing our approach as a reliable and interpretable solution for complex planning tasks. 

**Abstract (ZH)**: 我们考虑增强大型语言模型（LLMs）以应对复杂的规划任务。现有方法允许LLMs探索中间步骤以制定计划，但这些方法要么依赖于不可靠的自我验证，要么依赖外部验证器来评估这些步骤，这需要大量的数据和计算资源。为此，我们提出了自动启发式发现（AutoHD），这是一种新颖的方法，使LLMs能够明确生成启发式函数以指导推理时的搜索，从而准确评估中间状态。这些启发式函数通过启发式进化过程进一步优化，提高其稳健性和有效性。我们的方法不需要额外的模型训练或微调，由LLMs生成的明确定义的启发式函数提供了可解释性和对推理过程的洞察。在多种基准上的广泛实验表明，与多个基线方法相比取得了显著改进，包括某些数据集上的准确率几乎是基线的两倍，确立了该方法作为解决复杂规划任务的可靠且可解释的解决方案的地位。 

---
# Multi-Agent Security Tax: Trading Off Security and Collaboration Capabilities in Multi-Agent Systems 

**Title (ZH)**: 多 Agent 安全税：多 Agent 系统中安全与协作能力的权衡 

**Authors**: Pierre Peigne-Lefebvre, Mikolaj Kniejski, Filip Sondej, Matthieu David, Jason Hoelscher-Obermaier, Christian Schroeder de Witt, Esben Kran  

**Link**: [PDF](https://arxiv.org/pdf/2502.19145)  

**Abstract**: As AI agents are increasingly adopted to collaborate on complex objectives, ensuring the security of autonomous multi-agent systems becomes crucial. We develop simulations of agents collaborating on shared objectives to study these security risks and security trade-offs. We focus on scenarios where an attacker compromises one agent, using it to steer the entire system toward misaligned outcomes by corrupting other agents. In this context, we observe infectious malicious prompts - the multi-hop spreading of malicious instructions. To mitigate this risk, we evaluated several strategies: two "vaccination" approaches that insert false memories of safely handling malicious input into the agents' memory stream, and two versions of a generic safety instruction strategy. While these defenses reduce the spread and fulfillment of malicious instructions in our experiments, they tend to decrease collaboration capability in the agent network. Our findings illustrate potential trade-off between security and collaborative efficiency in multi-agent systems, providing insights for designing more secure yet effective AI collaborations. 

**Abstract (ZH)**: 随着人工智能代理在复杂目标协作中的应用日益增多，确保自主多智能体系统的安全变得至关重要。我们通过模拟智能体在共享目标上的协作来研究这些安全风险及安全权衡。我们关注攻击者 compromize 一个智能体并利用其引导整个系统向不对齐结果演变的情景。在此背景下，我们观察到感染性的恶意提示——恶意指令的多跳传播。为了减轻这种风险，我们评估了几种策略：两种“疫苗”方法，即向智能体的记忆流中插入关于安全处理恶意输入的虚假记忆，以及两种通用安全指令策略的版本。虽然这些防御措施在实验中减少了恶意指令的传播和执行，但它们往往会降低智能体网络的协作能力。我们的研究结果表明，在多智能体系统中可能存在安全性和协作效率之间的权衡，并为设计更安全且有效的AI合作提供了见解。 

---
# A Temporal Planning Framework for Multi-Agent Systems via LLM-Aided Knowledge Base Management 

**Title (ZH)**: 基于LLM辅助知识库管理的多智能体系统时序规划框架 

**Authors**: Enrico Saccon, Ahmet Tikna, Davide De Martini, Edoardo Lamon, Luigi Palopoli, Marco Roveri  

**Link**: [PDF](https://arxiv.org/pdf/2502.19135)  

**Abstract**: This paper presents a novel framework, called PLANTOR (PLanning with Natural language for Task-Oriented Robots), that integrates Large Language Models (LLMs) with Prolog-based knowledge management and planning for multi-robot tasks. The system employs a two-phase generation of a robot-oriented knowledge base, ensuring reusability and compositional reasoning, as well as a three-step planning procedure that handles temporal dependencies, resource constraints, and parallel task execution via mixed-integer linear programming. The final plan is converted into a Behaviour Tree for direct use in ROS2. We tested the framework in multi-robot assembly tasks within a block world and an arch-building scenario. Results demonstrate that LLMs can produce accurate knowledge bases with modest human feedback, while Prolog guarantees formal correctness and explainability. This approach underscores the potential of LLM integration for advanced robotics tasks requiring flexible, scalable, and human-understandable planning. 

**Abstract (ZH)**: PLANTOR：基于自然语言的面向任务机器人规划框架 

---
# Nexus: A Lightweight and Scalable Multi-Agent Framework for Complex Tasks Automation 

**Title (ZH)**: Nexus：一种轻量级可扩展的多Agent复杂任务自动化框架 

**Authors**: Humza Sami, Mubashir ul Islam, Samy Charas, Asav Gandhi, Pierre-Emmanuel Gaillardon, Valerio Tenace  

**Link**: [PDF](https://arxiv.org/pdf/2502.19091)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have substantially evolved Multi-Agent Systems (MASs) capabilities, enabling systems that not only automate tasks but also leverage near-human reasoning capabilities. To achieve this, LLM-based MASs need to be built around two critical principles: (i) a robust architecture that fully exploits LLM potential for specific tasks -- or related task sets -- and ($ii$) an effective methodology for equipping LLMs with the necessary capabilities to perform tasks and manage information efficiently. It goes without saying that a priori architectural designs can limit the scalability and domain adaptability of a given MAS.
To address these challenges, in this paper we introduce Nexus: a lightweight Python framework designed to easily build and manage LLM-based MASs. Nexus introduces the following innovations: (i) a flexible multi-supervisor hierarchy, (ii) a simplified workflow design, and (iii) easy installation and open-source flexibility: Nexus can be installed via pip and is distributed under a permissive open-source license, allowing users to freely modify and extend its capabilities.
Experimental results demonstrate that architectures built with Nexus exhibit state-of-the-art performance across diverse domains. In coding tasks, Nexus-driven MASs achieve a 99% pass rate on HumanEval and a flawless 100% on VerilogEval-Human, outperforming cutting-edge reasoning language models such as o3-mini and DeepSeek-R1. Moreover, these architectures display robust proficiency in complex reasoning and mathematical problem solving, achieving correct solutions for all randomly selected problems from the MATH dataset. In the realm of multi-objective optimization, Nexus-based architectures successfully address challenging timing closure tasks on designs from the VTR benchmark suite, while guaranteeing, on average, a power saving of nearly 30%. 

**Abstract (ZH)**: 最近大型语言模型（LLMs）的进步显著提升了多智能体系统（MASs）的能力，使其不仅能够自动化任务，还能够利用接近人类的推理能力。为了实现这一点，基于LLM的MASs需要遵循两个关键原则：(i)一个稳健的架构，能够充分利用LLM在特定任务或相关任务集上的潜力；和(ii)有效的机制来赋予LLMs执行任务和高效管理信息所需的必要能力。不言而喻，先验的架构设计可能会限制给定MAS的可扩展性和领域适应性。

为了应对这些挑战，本文介绍了Nexus：一个轻量级的Python框架，旨在方便地构建和管理基于LLM的MASs。Nexus引入了以下创新：(i)灵活的多监督器层次结构，(ii)简化的工作流设计，和(iii)易于安装和开源灵活性：Nexus可以通过pip安装，并且分发在一种宽松的开源许可下，允许用户自由修改和扩展其功能。

实验结果表明，使用Nexus构建的架构在多个领域均表现出最先进的性能。在编程任务中，Nexus驱动的MASs在HumanEval上的通过率为99%，在VerilogEval-Human上达到完美的100%，超过了最先进的推理语言模型如o3-mini和DeepSeek-R1。此外，这些架构在复杂的推理和数学问题解决方面表现出强大的能力，能够正确解决数学数据集随机选择的所有问题。在多目标优化领域，基于Nexus的架构成功解决了来自VTR基准套件的设计上的困难的时序闭合问题，同时平均保证了近30%的功耗节省。 

---
# Dealing with Inconsistency for Reasoning over Knowledge Graphs: A Survey 

**Title (ZH)**: 知识图谱推理中处理不一致性综述 

**Authors**: Anastasios Nentidis, Charilaos Akasiadis, Angelos Charalambidis, Alexander Artikis  

**Link**: [PDF](https://arxiv.org/pdf/2502.19023)  

**Abstract**: In Knowledge Graphs (KGs), where the schema of the data is usually defined by particular ontologies, reasoning is a necessity to perform a range of tasks, such as retrieval of information, question answering, and the derivation of new knowledge. However, information to populate KGs is often extracted (semi-) automatically from natural language resources, or by integrating datasets that follow different semantic schemas, resulting in KG inconsistency. This, however, hinders the process of reasoning. In this survey, we focus on how to perform reasoning on inconsistent KGs, by analyzing the state of the art towards three complementary directions: a) the detection of the parts of the KG that cause the inconsistency, b) the fixing of an inconsistent KG to render it consistent, and c) the inconsistency-tolerant reasoning. We discuss existing work from a range of relevant fields focusing on how, and in which cases they are related to the above directions. We also highlight persisting challenges and future directions. 

**Abstract (ZH)**: 知识图谱中不一致性的推理研究综述 

---
# Talking like Piping and Instrumentation Diagrams (P&IDs) 

**Title (ZH)**: 模拟管道和仪表图（P&IDs）的表达方式 

**Authors**: Achmad Anggawirya Alimin, Dominik P. Goldstein, Lukas Schulze Balhorn, Artur M. Schweidtmann  

**Link**: [PDF](https://arxiv.org/pdf/2502.18928)  

**Abstract**: We propose a methodology that allows communication with Piping and Instrumentation Diagrams (P&IDs) using natural language. In particular, we represent P&IDs through the DEXPI data model as labeled property graphs and integrate them with Large Language Models (LLMs). The approach consists of three main parts: 1) P&IDs are cast into a graph representation from the DEXPI format using our pyDEXPI Python package. 2) A tool for generating P&ID knowledge graphs from pyDEXPI. 3) Integration of the P&ID knowledge graph to LLMs using graph-based retrieval augmented generation (graph-RAG). This approach allows users to communicate with P&IDs using natural language. It extends LLM's ability to retrieve contextual data from P&IDs and mitigate hallucinations. Leveraging the LLM's large corpus, the model is also able to interpret process information in PIDs, which could help engineers in their daily tasks. In the future, this work will also open up opportunities in the context of other generative Artificial Intelligence (genAI) solutions on P&IDs, and AI-assisted HAZOP studies. 

**Abstract (ZH)**: 我们提出了一种方法，使得可以通过自然语言与工艺和仪表图（P&IDs）进行通信。特别地，我们通过DEXPI数据模型将P&IDs表示为标记属性图，并与大型语言模型（LLMs）集成。该方法包括三个主要部分：1）使用我们编写的pyDEXPI Python包将P&IDs转换为从DEXPI格式表示的图表示。2）从pyDEXPI生成P&ID知识图谱的工具。3）使用基于图的检索增强生成（graph-RAG）将P&ID知识图谱集成到LLMs中。该方法使得用户能够使用自然语言与P&IDs进行通信。它扩展了LLMs从P&IDs中检索上下文数据的能力，并减轻了幻觉现象。利用LLMs庞大的语料库，该模型还能够解释PIDs中的工艺信息，这有助于工程师完成日常工作任务。未来，这项工作还将为P&IDs上的其他生成型人工智能（genAI）解决方案以及AI辅助的HAZOP研究开辟机会。 

---
# Multi-LLM Collaborative Search for Complex Problem Solving 

**Title (ZH)**: 多大型语言模型协作搜索以解决复杂问题 

**Authors**: Sen Yang, Yafu Li, Wai Lam, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.18873)  

**Abstract**: Large language models (LLMs) often struggle with complex reasoning tasks due to their limitations in addressing the vast reasoning space and inherent ambiguities of natural language. We propose the Mixture-of-Search-Agents (MoSA) paradigm, a novel approach leveraging the collective expertise of multiple LLMs to enhance search-based reasoning. MoSA integrates diverse reasoning pathways by combining independent exploration with iterative refinement among LLMs, mitigating the limitations of single-model approaches. Using Monte Carlo Tree Search (MCTS) as a backbone, MoSA enables multiple agents to propose and aggregate reasoning steps, resulting in improved accuracy. Our comprehensive evaluation across four reasoning benchmarks demonstrates MoSA's consistent performance improvements over single-agent and other multi-agent baselines, particularly in complex mathematical and commonsense reasoning tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）常常在复杂推理任务中遇到困难，这是因为它们在处理广阔推理空间和自然语言固有的歧义性方面存在局限。我们提出了一种新的混合搜索代理（MoSA）范式，该范式通过多LLM协同工作来增强基于搜索的推理能力。MoSA 通过结合独立探索与迭代精炼，整合了多条不同的推理路径，从而缓解了单模型方法的局限性。MoSA 以蒙特卡洛树搜索（MCTS）为基础，允许多个代理提出并聚合推理步骤，从而提高了准确性。我们在四个推理基准上的全面评估表明，MoSA 在复杂数学和常识推理任务中相较于单代理和其他多代理基线方法具有一致性性能提升。 

---
# Towards an AI co-scientist 

**Title (ZH)**: 向AI合作者方向发展 

**Authors**: Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Anil Palepu, Petar Sirkovic, Artiom Myaskovsky, Felix Weissenberger, Keran Rong, Ryutaro Tanno, Khaled Saab, Dan Popovici, Jacob Blum, Fan Zhang, Katherine Chou, Avinatan Hassidim, Burak Gokturk, Amin Vahdat, Pushmeet Kohli, Yossi Matias, Andrew Carroll, Kavita Kulkarni, Nenad Tomasev, Yuan Guan, Vikram Dhillon, Eeshit Dhaval Vaishnav, Byron Lee, Tiago R D Costa, José R Penadés, Gary Peltz, Yunhan Xu, Annalisa Pawlosky, Alan Karthikesalingam, Vivek Natarajan  

**Link**: [PDF](https://arxiv.org/pdf/2502.18864)  

**Abstract**: Scientific discovery relies on scientists generating novel hypotheses that undergo rigorous experimental validation. To augment this process, we introduce an AI co-scientist, a multi-agent system built on Gemini 2.0. The AI co-scientist is intended to help uncover new, original knowledge and to formulate demonstrably novel research hypotheses and proposals, building upon prior evidence and aligned to scientist-provided research objectives and guidance. The system's design incorporates a generate, debate, and evolve approach to hypothesis generation, inspired by the scientific method and accelerated by scaling test-time compute. Key contributions include: (1) a multi-agent architecture with an asynchronous task execution framework for flexible compute scaling; (2) a tournament evolution process for self-improving hypotheses generation. Automated evaluations show continued benefits of test-time compute, improving hypothesis quality. While general purpose, we focus development and validation in three biomedical areas: drug repurposing, novel target discovery, and explaining mechanisms of bacterial evolution and anti-microbial resistance. For drug repurposing, the system proposes candidates with promising validation findings, including candidates for acute myeloid leukemia that show tumor inhibition in vitro at clinically applicable concentrations. For novel target discovery, the AI co-scientist proposed new epigenetic targets for liver fibrosis, validated by anti-fibrotic activity and liver cell regeneration in human hepatic organoids. Finally, the AI co-scientist recapitulated unpublished experimental results via a parallel in silico discovery of a novel gene transfer mechanism in bacterial evolution. These results, detailed in separate, co-timed reports, demonstrate the potential to augment biomedical and scientific discovery and usher an era of AI empowered scientists. 

**Abstract (ZH)**: 科学发现依赖于科学家生成新颖的假说并进行严格的实验验证。为此，我们介绍了一种基于Gemini 2.0构建的AI合作者，这是一种多智能体系统。AI合作者旨在帮助揭示新的原创知识，并根据前期证据形成可验证的新颖研究假说和提案，这些假说和提案与科学家提供的研究目标和指导相一致。系统设计采用了生成、辩论和演化的假说生成方法，借鉴了科学方法，并通过扩展测试时的计算能力加速了这一过程。关键贡献包括：(1) 具有异步任务执行框架的多智能体架构，支持灵活的计算扩展；(2) 一种锦标赛进化过程，用于自我改进的假说生成。自动化评估结果显示，测试时的计算能力持续带来益处，提高了假说的质量。尽管具有通用性，但我们将开发和验证的重点放在三个生物医学领域：药物再利用、新型靶点发现以及解释细菌进化的机制和抗微生物耐药性。在药物再利用方面，该系统提出了一些有前景的候选药物，包括在临床适用浓度下显示体外肿瘤抑制作用的急性髓系白血病候选药物。在新型靶点发现方面，AI合作者提出了新的表观遗传学靶点，经过抗纤维化活性和人类肝类器官中肝细胞再生的验证。最后，AI合作者通过平行的计算发现揭示了一种新的基因转移机制，成功重现了未发表的实验结果。这些结果，分别在同期报告中详细阐述，展示了其在生物医学和科学发现中的增效潜力，并开启了以人工智能赋能的科学家时代。 

---
# Intelligence Test 

**Title (ZH)**: 智力测试 

**Authors**: Jingtao Zhan, Jiahao Zhao, Jiayu Li, Yiqun Liu, Bo Zhang, Qingyao Ai, Jiaxin Mao, Hongning Wang, Min Zhang, Shaoping Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.18858)  

**Abstract**: How does intelligence emerge? We propose that intelligence is not a sudden gift or random occurrence, but rather a necessary trait for species to survive through Natural Selection. If a species passes the test of Natural Selection, it demonstrates the intelligence to survive in nature. Extending this perspective, we introduce Intelligence Test, a method to quantify the intelligence of any subject on any task. Like how species evolve by trial and error, Intelligence Test quantifies intelligence by the number of failed attempts before success. Fewer failures correspond to higher intelligence. When the expectation and variance of failure counts are both finite, it signals the achievement of an autonomous level of intelligence. Using Intelligence Test, we comprehensively evaluate existing AI systems. Our results show that while AI systems achieve a level of autonomy in simple tasks, they are still far from autonomous in more complex tasks, such as vision, search, recommendation, and language. While scaling model size might help, this would come at an astronomical cost. Projections suggest that achieving general autonomy would require unimaginable $10^{26}$ parameters. Even if Moore's Law continuously holds, such a parameter scale would take $70$ years. This staggering cost highlights the complexity of human tasks and the inadequacies of current AI. To further understand this phenomenon, we conduct a theoretical analysis. Our simulations suggest that human tasks possess a criticality property. As a result, autonomy requires a deep understanding of the task's underlying mechanisms. Current AI, however, does not fully grasp these mechanisms and instead relies on superficial mimicry, making it difficult to reach an autonomous level. We believe Intelligence Test can not only guide the future development of AI but also offer profound insights into the intelligence of humans ourselves. 

**Abstract (ZH)**: 智能是如何产生的？我们提出智能不是突然的馈赠或随机的发生，而是物种通过自然选择生存的必要特质。如果物种通过了自然选择的考验，就证明了其在自然中生存的智能。从这一视角出发，我们引入了智能测试方法，用于衡量任何主体在任何任务上的智能水平。就像物种通过试错进化一样，智能测试通过计算在成功之前失败的次数来量化智能。失败次数越少，智能水平越高。当失败次数的期望值和方差都是有限的，这表明达到了自主智能的水平。通过智能测试，我们全面评估了现有AI系统。结果显示，虽然AI系统在简单任务中达到了一定程度的自主性，但在更复杂的任务如视觉、搜索、推荐和语言方面仍然相距甚远。虽然增加模型规模可能有所帮助，但这会带来天文数字般的成本。预测表明，实现普遍自主性可能需要难以想象的$10^{26}$个参数。即使摩尔定律持续有效，达到这一参数规模也需要70年。这一惊人的成本突显了人类任务的复杂性和当前AI的不足。为进一步理解这一现象，我们进行了理论分析。我们的模拟表明，人类任务具有临界性特征，因此自主性需要深入理解任务的基本机制。然而，当前的AI并未全面掌握这些机制，而是依赖于表面的模仿，难以达到自主水平。我们相信，智能测试不仅能够指导未来AI的发展，还能提供对人类自身智能的深刻洞见。 

---
# REALM-Bench: A Real-World Planning Benchmark for LLMs and Multi-Agent Systems 

**Title (ZH)**: REALM-Bench: 一个面向LLMs和多智能体系统的实际规划基准 

**Authors**: Longling Geng, Edward Y. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18836)  

**Abstract**: This benchmark suite provides a comprehensive evaluation framework for assessing both individual LLMs and multi-agent systems in real-world planning scenarios. The suite encompasses eleven designed problems that progress from basic to highly complex, incorporating key aspects such as multi-agent coordination, inter-agent dependencies, and dynamic environmental disruptions. Each problem can be scaled along three dimensions: the number of parallel planning threads, the complexity of inter-dependencies, and the frequency of unexpected disruptions requiring real-time adaptation. The benchmark includes detailed specifications, evaluation metrics, and baseline implementations using contemporary frameworks like LangGraph, enabling rigorous testing of both single-agent and multi-agent planning capabilities. Through standardized evaluation criteria and scalable complexity, this benchmark aims to drive progress in developing more robust and adaptable AI planning systems for real-world applications. 

**Abstract (ZH)**: 这个基准套件提供了一套全面的评估框架，用于评估单个大规模语言模型和多智能体系统在真实世界规划场景中的表现。该套件包含 eleven 个设计问题，从基础逐步进展到高度复杂，并融入了多智能体协调、智能体间依赖关系以及动态环境干扰等关键方面。每个问题可以在三个维度上进行扩展：并行规划线程的数量、相互依赖关系的复杂性，以及需要实时适应的意外干扰的频率。该基准套件包括详细的规范、评估指标以及基于当代框架（如 LangGraph）的基线实现，从而能严格测试单智能体和多智能体规划能力。通过标准化的评估标准和可扩展的复杂性，该基准套件旨在推动开发更具鲁棒性和适应性的 AI 规划系统，应用于实际应用中。 

---
# Data-Efficient Multi-Agent Spatial Planning with LLMs 

**Title (ZH)**: LLMs驱动的数据高效多智能体空间规划 

**Authors**: Huangyuan Su, Aaron Walsman, Daniel Garces, Sham Kakade, Stephanie Gil  

**Link**: [PDF](https://arxiv.org/pdf/2502.18822)  

**Abstract**: In this project, our goal is to determine how to leverage the world-knowledge of pretrained large language models for efficient and robust learning in multiagent decision making. We examine this in a taxi routing and assignment problem where agents must decide how to best pick up passengers in order to minimize overall waiting time. While this problem is situated on a graphical road network, we show that with the proper prompting zero-shot performance is quite strong on this task. Furthermore, with limited fine-tuning along with the one-at-a-time rollout algorithm for look ahead, LLMs can out-compete existing approaches with 50 times fewer environmental interactions. We also explore the benefits of various linguistic prompting approaches and show that including certain easy-to-compute information in the prompt significantly improves performance. Finally, we highlight the LLM's built-in semantic understanding, showing its ability to adapt to environmental factors through simple prompts. 

**Abstract (ZH)**: 本项目旨在探讨如何利用预训练大语言模型的世界知识，实现多agent决策制定中的高效和稳健学习。我们通过出租车路线规划和分配问题来检验这一方法，该问题要求agents决定如何最优地接乘客以最小化总体等待时间。尽管该问题基于图形道路网络，我们证明通过适当的提示，零样本性能在这一任务上表现较强。此外，通过有限的微调与单次展开前瞻算法相结合，LLMs可以在与现有方法相当的性能下，通过较少的环境交互次数（减少50倍）实现超越。我们还探讨了不同语言提示方法的益处，并表明在提示中包含某些易于计算的信息显著提高了性能。最后，我们强调了LLMs内置的语义理解能力，展示了其通过简单的提示适应环境因素的能力。 

---
# Holistic Audit Dataset Generation for LLM Unlearning via Knowledge Graph Traversal and Redundancy Removal 

**Title (ZH)**: 通过知识图谱遍历和冗余移除实现的LLM遗忘数据集生成方法 

**Authors**: Weipeng Jiang, Juan Zhai, Shiqing Ma, Ziyan Lei, Xiaofei Xie, Yige Wang, Chao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18810)  

**Abstract**: In recent years, Large Language Models (LLMs) have faced increasing demands to selectively remove sensitive information, protect privacy, and comply with copyright regulations through unlearning, by Machine Unlearning. While evaluating unlearning effectiveness is crucial, existing benchmarks are limited in scale and comprehensiveness, typically containing only a few hundred test cases. We identify two critical challenges in generating holistic audit datasets: ensuring audit adequacy and handling knowledge redundancy between forget and retain dataset. To address these challenges, we propose HANKER, an automated framework for holistic audit dataset generation leveraging knowledge graphs to achieve fine-grained coverage and eliminate redundant knowledge. Applying HANKER to the popular MUSE benchmark, we successfully generated over 69,000 and 111,000 audit cases for the News and Books datasets respectively, identifying thousands of knowledge memorization instances that the previous benchmark failed to detect. Our empirical analysis uncovers how knowledge redundancy significantly skews unlearning effectiveness metrics, with redundant instances artificially inflating the observed memorization measurements ROUGE from 19.7% to 26.1% and Entailment Scores from 32.4% to 35.2%, highlighting the necessity of systematic deduplication for accurate assessment. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在通过机器忘却技术选择性地去除敏感信息、保护隐私并遵守版权法规方面面临着不断增加的需求。评估忘却是至关重要的，但现有的基准在规模和全面性方面存在局限，通常仅包含几百个测试案例。我们识别出生成综合审计数据集的两个关键挑战：确保审计充分性和处理忘却集和保留集之间的知识冗余。为应对这些挑战，我们提出了一种名为HANKER的自动化框架，利用知识图谱实现细粒度覆盖并消除冗余知识。将HANKER应用于流行的MUSE基准，我们成功为新闻和书籍数据集分别生成了超过69,000和111,000个审计案例，发现先前基准未能检测到成千上万个知识记忆实例。我们的实证分析揭示了知识冗余如何显著扭曲忘却效果指标，冗余实例使观察到的记忆测量值ROUGE从19.7%提升至26.1%，Entailment Scores从32.4%提升至35.2%，强调了系统去重对于准确评估的必要性。 

---
# Like Father, Like Son: Kinship-Aware Preference Mapping (KARMA) for Automatic Alignment in Large Language Models 

**Title (ZH)**: 父似子：亲缘意识偏好映射（KARMA）在大型语言模型中的自动对齐 

**Authors**: Jeesu Jung, Chanjun Park, Sangkeun Jung  

**Link**: [PDF](https://arxiv.org/pdf/2502.18744)  

**Abstract**: Recent advancements in Large Language Model (LLM) alignment have sought to mitigate the cost of human annotations by leveraging pretrained models to generate preference data. However, existing methods often compare responses from models with substantially different capabilities, yielding superficial distinctions that fail to provide meaningful guidance on what constitutes a superior response. To address this limitation, we propose Kinship-Aware pReference MApping (KARMA), a novel framework that systematically pairs responses from models with comparable competencies. By constraining preference comparisons to outputs of similar complexity and quality, KARMA enhances the informativeness of preference data and improves the granularity of alignment signals. Empirical evaluations demonstrate that our kinship-aware approach leads to more consistent and interpretable alignment outcomes, ultimately facilitating a more principled and reliable pathway for aligning LLM behavior with human preferences. 

**Abstract (ZH)**: Recent advancements in Large Language Model (LLM) alignment have sought to mitigate the cost of human annotations by leveraging pretrained models to generate preference data. However, existing methods often compare responses from models with substantially different capabilities, yielding superficial distinctions that fail to provide meaningful guidance on what constitutes a superior response. To address this limitation, we propose Kinship-Aware Preference Mapping (KARMA), a novel framework that systematically pairs responses from models with comparable competencies. By constraining preference comparisons to outputs of similar complexity and quality, KARMA enhances the informativeness of preference data and improves the granularity of alignment signals. Empirical evaluations demonstrate that our kinship-aware approach leads to more consistent and interpretable alignment outcomes, ultimately facilitating a more principled and reliable pathway for aligning LLM behavior with human preferences.

翻译后的标题：
基于亲和性感知偏好映射的大型语言模型对齐进展 

---
# Talking to the brain: Using Large Language Models as Proxies to Model Brain Semantic Representation 

**Title (ZH)**: 与大脑对话：使用大型语言模型作为代理建模大脑语义表示 

**Authors**: Xin Liu, Ziyue Zhang, Jingxin Nie  

**Link**: [PDF](https://arxiv.org/pdf/2502.18725)  

**Abstract**: Traditional psychological experiments utilizing naturalistic stimuli face challenges in manual annotation and ecological validity. To address this, we introduce a novel paradigm leveraging multimodal large language models (LLMs) as proxies to extract rich semantic information from naturalistic images through a Visual Question Answering (VQA) strategy for analyzing human visual semantic representation. LLM-derived representations successfully predict established neural activity patterns measured by fMRI (e.g., faces, buildings), validating its feasibility and revealing hierarchical semantic organization across cortical regions. A brain semantic network constructed from LLM-derived representations identifies meaningful clusters reflecting functional and contextual associations. This innovative methodology offers a powerful solution for investigating brain semantic organization with naturalistic stimuli, overcoming limitations of traditional annotation methods and paving the way for more ecologically valid explorations of human cognition. 

**Abstract (ZH)**: 利用多模态大型语言模型作为代理通过视觉问答策略从自然图像中提取丰富语义信息，以解决传统自然刺激心理实验的手动标注挑战和生态效度问题：一种基于语言模型衍生表征构建脑语义网络的方法 

---
# TrajLLM: A Modular LLM-Enhanced Agent-Based Framework for Realistic Human Trajectory Simulation 

**Title (ZH)**: TrajLLM：一种基于代理的模块化大语言模型增强现实人类轨迹模拟框架 

**Authors**: Chenlu Ju, Jiaxin Liu, Shobhit Sinha, Hao Xue, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2502.18712)  

**Abstract**: This work leverages Large Language Models (LLMs) to simulate human mobility, addressing challenges like high costs and privacy concerns in traditional models. Our hierarchical framework integrates persona generation, activity selection, and destination prediction, using real-world demographic and psychological data to create realistic movement patterns. Both physical models and language models are employed to explore and demonstrate different methodologies for human mobility simulation. By structuring data with summarization and weighted density metrics, the system ensures scalable memory management while retaining actionable insights. Preliminary results indicate that LLM-driven simulations align with observed real-world patterns, offering scalable, interpretable insights for social problems such as urban planning, traffic management, and public health. The framework's ability to dynamically generate personas and activities enables it to provide adaptable and realistic daily routines. This study demonstrates the transformative potential of LLMs in advancing mobility modeling for societal and urban applications. The source code and interactive demo for our framework are available at this https URL. 

**Abstract (ZH)**: 本研究利用大型语言模型（LLMs）模拟人类移动，以应对传统模型中成本高和隐私问题的挑战。该分层框架整合了个性生成、活动选择和目的地预测，并使用现实世界的人口统计和心理数据来创建现实的移动模式。物理模型和语言模型都被应用于探索和展示不同的人类移动模拟方法。通过使用总结和加权密度度量结构化数据，系统确保了可扩展的内存管理，同时保留了可操作的洞察。初步结果显示，由LLM驱动的模拟与观察到的实际模式相符，提供了适用于社会问题如城市规划、交通管理及公共卫生的可扩展且可解释的洞察。该框架能够动态生成个性和活动，使其能够提供可适应且现实的日常安排。本研究展示了LLMs在促进社会和城市应用中的移动建模方面的变革潜力。有关该框架的源代码和交互式演示可在以下链接访问：this https URL。 

---
# Hybrid Voting-Based Task Assignment in Role-Playing Games 

**Title (ZH)**: 角色扮演游戏中基于混合投票的任务分配 

**Authors**: Daniel Weiner, Raj Korpan  

**Link**: [PDF](https://arxiv.org/pdf/2502.18690)  

**Abstract**: In role-playing games (RPGs), the level of immersion is critical-especially when an in-game agent conveys tasks, hints, or ideas to the player. For an agent to accurately interpret the player's emotional state and contextual nuances, a foundational level of understanding is required, which can be achieved using a Large Language Model (LLM). Maintaining the LLM's focus across multiple context changes, however, necessitates a more robust approach, such as integrating the LLM with a dedicated task allocation model to guide its performance throughout gameplay. In response to this need, we introduce Voting-Based Task Assignment (VBTA), a framework inspired by human reasoning in task allocation and completion. VBTA assigns capability profiles to agents and task descriptions to tasks, then generates a suitability matrix that quantifies the alignment between an agent's abilities and a task's requirements. Leveraging six distinct voting methods, a pre-trained LLM, and integrating conflict-based search (CBS) for path planning, VBTA efficiently identifies and assigns the most suitable agent to each task. While existing approaches focus on generating individual aspects of gameplay, such as single quests, or combat encounters, our method shows promise when generating both unique combat encounters and narratives because of its generalizable nature. 

**Abstract (ZH)**: 在角色扮演游戏（RPG）中，沉浸感的水平至关重要，特别是在游戏代理向玩家传达任务、提示或想法时。为了使代理能够准确理解玩家的情绪状态和语境细微差别，需要具备一定的理解基础，这可以通过使用大规模语言模型（LLM）来实现。然而，维持LLM在多次语境变化中的专注力需要更稳健的方法，例如将LLM与专门的任务分配模型集成，以指导其在整个游戏过程中的表现。为应对这一需求，我们引入了基于投票的任务分配框架（VBTA），该框架受人类任务分配与完成过程中推理的启发。VBTA为代理分配能力配置文件，并为任务分配任务描述，进而生成量化代理能力和任务要求之间契合度的适合性矩阵。利用六种不同的投票方法、预训练的LLM，并结合冲突基于搜索（CBS）进行路径规划，VBTA有效地识别并分配最适合的代理给每个任务。尽管现有方法侧重于生成游戏的单一方面，如单个任务或战斗遭遇，我们的方法因其通用性在生成独特战斗遭遇和叙述时显示出前景。 

---
# Speaking the Right Language: The Impact of Expertise Alignment in User-AI Interactions 

**Title (ZH)**: 讲对语言：用户与AI交互中专业知识匹配的影响 

**Authors**: Shramay Palta, Nirupama Chandrasekaran, Rachel Rudinger, Scott Counts  

**Link**: [PDF](https://arxiv.org/pdf/2502.18685)  

**Abstract**: Using a sample of 25,000 Bing Copilot conversations, we study how the agent responds to users of varying levels of domain expertise and the resulting impact on user experience along multiple dimensions. Our findings show that across a variety of topical domains, the agent largely responds at proficient or expert levels of expertise (77% of conversations) which correlates with positive user experience regardless of the user's level of expertise. Misalignment, such that the agent responds at a level of expertise below that of the user, has a negative impact on overall user experience, with the impact more profound for more complex tasks. We also show that users engage more, as measured by the number of words in the conversation, when the agent responds at a level of expertise commensurate with that of the user. Our findings underscore the importance of alignment between user and AI when designing human-centered AI systems, to ensure satisfactory and productive interactions. 

**Abstract (ZH)**: 基于25,000个必应AI助理对话样本，我们研究了代理如何根据不同水平的专业用户的回应，并分析多维度用户经验的影响。研究结果表明，无论用户的专业水平如何，代理在多种主题领域中大多以熟练或专家级别的专业知识作出回应（占77%的对话），这与积极的用户经验相关。若代理的回应水平低于用户的专业水平，这种不一致会负面影响整体的用户体验，尤其是对于复杂任务的影响更为显著。我们还表明，当代理的回应水平与用户相当时，用户在对话中会更加互动，表现为对话中词汇数量的增加。我们的研究强调，在设计以用户为中心的AI系统时，确保用户和AI之间的匹配对于实现满意的和富有成效的交互至关重要。 

---
# Independent Mobility GPT (IDM-GPT): A Self-Supervised Multi-Agent Large Language Model Framework for Customized Traffic Mobility Analysis Using Machine Learning Models 

**Title (ZH)**: 独立移动GPT（IDM-GPT）：一种基于自我监督的多agent大型语言模型框架，用于使用机器学习模型进行个性化交通移动分析 

**Authors**: Fengze Yang, Xiaoyue Cathy Liu, Lingjiu Lu, Bingzhang Wang, Chenxi  

**Link**: [PDF](https://arxiv.org/pdf/2502.18652)  

**Abstract**: With the urbanization process, an increasing number of sensors are being deployed in transportation systems, leading to an explosion of big data. To harness the power of this vast transportation data, various machine learning (ML) and artificial intelligence (AI) methods have been introduced to address numerous transportation challenges. However, these methods often require significant investment in data collection, processing, storage, and the employment of professionals with expertise in transportation and ML. Additionally, privacy issues are a major concern when processing data for real-world traffic control and management. To address these challenges, the research team proposes an innovative Multi-agent framework named Independent Mobility GPT (IDM-GPT) based on large language models (LLMs) for customized traffic analysis, management suggestions, and privacy preservation. IDM-GPT efficiently connects users, transportation databases, and ML models economically. IDM-GPT trains, customizes, and applies various LLM-based AI agents for multiple functions, including user query comprehension, prompts optimization, data analysis, model selection, and performance evaluation and enhancement. With IDM-GPT, users without any background in transportation or ML can efficiently and intuitively obtain data analysis and customized suggestions in near real-time based on their questions. Experimental results demonstrate that IDM-GPT delivers satisfactory performance across multiple traffic-related tasks, providing comprehensive and actionable insights that support effective traffic management and urban mobility improvement. 

**Abstract (ZH)**: 基于大规模语言模型的独立移动GPT多agent框架：定制化交通分析与隐私保护 

---
# Automated Knowledge Component Generation and Knowledge Tracing for Coding Problems 

**Title (ZH)**: 自动化知识组件生成与编程问题知识追踪 

**Authors**: Zhangqi Duan, Nigel Fernandez, Sri Kanakadandi, Bita Akram, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2502.18632)  

**Abstract**: Knowledge components (KCs) mapped to problems help model student learning, tracking their mastery levels on fine-grained skills thereby facilitating personalized learning and feedback in online learning platforms. However, crafting and tagging KCs to problems, traditionally performed by human domain experts, is highly labor-intensive. We present a fully automated, LLM-based pipeline for KC generation and tagging for open-ended programming problems. We also develop an LLM-based knowledge tracing (KT) framework to leverage these LLM-generated KCs, which we refer to as KCGen-KT. We conduct extensive quantitative and qualitative evaluations validating the effectiveness of KCGen-KT. On a real-world dataset of student code submissions to open-ended programming problems, KCGen-KT outperforms existing KT methods. We investigate the learning curves of generated KCs and show that LLM-generated KCs have a comparable level-of-fit to human-written KCs under the performance factor analysis (PFA) model. We also conduct a human evaluation to show that the KC tagging accuracy of our pipeline is reasonably accurate when compared to that by human domain experts. 

**Abstract (ZH)**: 知识组件（KCs）映射到问题有助于建模学生学习，追踪他们在细粒度技能上的掌握水平，从而在在线学习平台中促进个性化学习和反馈。然而，传统上由人类领域专家完成的KCs的构建和标记工作非常耗时。我们提出了一种完全自动化的基于LLM的管道，用于生成和标记开放编程问题的KCs。我们还开发了一种基于LLM的知识追踪（KT）框架，利用这些由LLM生成的KCs，称之为KCGen-KT。我们进行了广泛的定量和定性评估，验证了KCGen-KT的有效性。在真实世界的学生代码提交数据集上，KCGen-KT优于现有KT方法。我们研究了生成KCs的学习曲线，并在性能因素分析（PFA）模型下展示了由LLM生成的KCs与人工编写的KCs具有相当拟合度。此外，我们进行了一项人工评估，表明与领域专家相比，我们管道中的KC标记准确性是合理的。 

---
# CuDIP: Enhancing Theorem Proving in LLMs via Curriculum Learning-based Direct Preference Optimization 

**Title (ZH)**: CuDIP: 基于课程学习直接偏好优化增强的大语言模型定理证明 

**Authors**: Shuming Shi, Ruobing Zuo, Gaolei He, Jianlin Wang, Chenyang Xu, Zhengfeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18532)  

**Abstract**: Automated theorem proving (ATP) is one of the most challenging mathematical reasoning tasks for Large Language Models (LLMs). Most existing LLM-based ATP methods rely on supervised fine-tuning, which results in a limited alignment between the theorem proving process and human preferences. Direct Preference Optimization (DPO), which aligns LLMs with human preferences, has shown positive effects for certain tasks. However, the lack of high-quality preference data for theorem proving presents a significant challenge. In this paper, we innovatively apply DPO to formal automated theorem proving and introduces a Curriculum Learning-based DPO Iterative Theorem Proving (CuDIP) method. Specifically, we propose a method for constructing preference data which utilizes LLMs and existing theorem proving data to enhance the diversity of the preference data while reducing the reliance on human preference annotations. We then integrate this preference data construction method with curriculum learning to iteratively fine-tune the theorem proving model through DPO. Experimental results on the MiniF2F and ProofNet datasets demonstrate the effectiveness of the proposed method. 

**Abstract (ZH)**: Automated定理证明中的直接偏好优化 Curriculum Learning导向的迭代定理证明方法（CuDIP） 

---
# Enhancing Hepatopathy Clinical Trial Efficiency: A Secure, Large Language Model-Powered Pre-Screening Pipeline 

**Title (ZH)**: 提升肝病临床试验效率：一种安全的大语言模型驱动的预筛查流水线 

**Authors**: Xiongbin Gui, Hanlin Lv, Xiao Wang, Longting Lv, Yi Xiao, Lei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18531)  

**Abstract**: Background: Recruitment for cohorts involving complex liver diseases, such as hepatocellular carcinoma and liver cirrhosis, often requires interpreting semantically complex criteria. Traditional manual screening methods are time-consuming and prone to errors. While AI-powered pre-screening offers potential solutions, challenges remain regarding accuracy, efficiency, and data privacy. Methods: We developed a novel patient pre-screening pipeline that leverages clinical expertise to guide the precise, safe, and efficient application of large language models. The pipeline breaks down complex criteria into a series of composite questions and then employs two strategies to perform semantic question-answering through electronic health records - (1) Pathway A, Anthropomorphized Experts' Chain of Thought strategy, and (2) Pathway B, Preset Stances within an Agent Collaboration strategy, particularly in managing complex clinical reasoning scenarios. The pipeline is evaluated on three key metrics-precision, time consumption, and counterfactual inference - at both the question and criterion levels. Results: Our pipeline achieved high precision (0.921, in criteria level) and efficiency (0.44s per task). Pathway B excelled in complex reasoning, while Pathway A was effective in precise data extraction with faster processing times. Both pathways achieved comparable precision. The pipeline showed promising results in hepatocellular carcinoma (0.878) and cirrhosis trials (0.843). Conclusions: This data-secure and time-efficient pipeline shows high precision in hepatopathy trials, providing promising solutions for streamlining clinical trial workflows. Its efficiency and adaptability make it suitable for improving patient recruitment. And its capability to function in resource-constrained environments further enhances its utility in clinical settings. 

**Abstract (ZH)**: 背景：涉及复杂肝脏疾病（如肝细胞癌和肝硬化）的队列研究通常需要解释语义复杂的标准。传统的手动筛选方法耗时且易出错。尽管AI驱动的预筛选提供了潜在解决方案，但在准确性和效率以及数据隐私方面仍存在挑战。方法：我们开发了一种新型患者预筛选流程，利用临床专业知识指导大型语言模型的精确、安全和高效应用。该流程将复杂的标准分解为一系列复合问题，然后采用两种策略通过电子健康记录进行语义问答——（1）路径A，类人专家思维链策略；（2）路径B，代理合作框架内预设立场策略，特别是在处理复杂的临床推理场景时。该流程在问题和标准层面通过三个关键指标——精确度、时间消耗和反事实推理——进行了评估。结果：我们的流程在标准层面实现了高精确度（0.921）和高效率（每任务0.44秒）。路径B在复杂推理方面表现出色，而路径A在精确的数据提取方面更为有效且处理速度更快。两种路径在精确度方面达到相当水平。该流程在肝细胞癌（0.878）和肝硬化试验（0.843）中显示出有希望的结果。结论：这种数据安全、时间高效的流程在肝脏疾病试验中表现出高精确度，为简化临床试验流程提供了有希望的解决方案。其高效性和适应性使其适用于提高患者招募，同时其在资源受限环境中运行的能力进一步增强了其在临床设置中的实用性。 

---
# Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models 

**Title (ZH)**: Hi 机器人：基于层次视觉-语言-动作模型的开放性指令跟随 

**Authors**: Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, Liyiming Ke, Karl Pertsch, Quan Vuong, James Tanner, Anna Walling, Haohuan Wang, Niccolo Fusai, Adrian Li-Bell, Danny Driess, Lachy Groom, Sergey Levine, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2502.19417)  

**Abstract**: Generalist robots that can perform a range of different tasks in open-world settings must be able to not only reason about the steps needed to accomplish their goals, but also process complex instructions, prompts, and even feedback during task execution. Intricate instructions (e.g., "Could you make me a vegetarian sandwich?" or "I don't like that one") require not just the ability to physically perform the individual steps, but the ability to situate complex commands and feedback in the physical world. In this work, we describe a system that uses vision-language models in a hierarchical structure, first reasoning over complex prompts and user feedback to deduce the most appropriate next step to fulfill the task, and then performing that step with low-level actions. In contrast to direct instruction following methods that can fulfill simple commands ("pick up the cup"), our system can reason through complex prompts and incorporate situated feedback during task execution ("that's not trash"). We evaluate our system across three robotic platforms, including single-arm, dual-arm, and dual-arm mobile robots, demonstrating its ability to handle tasks such as cleaning messy tables, making sandwiches, and grocery shopping. 

**Abstract (ZH)**: 能够在开放环境中执行多种不同任务的通用机器人必须能够不仅推理出完成目标所需的步骤，还能处理复杂指令、提示，甚至在任务执行过程中接收反馈。复杂的指令（例如，“你能为我做一个素食三明治吗？”或“我不喜欢那个”）要求不仅具备执行个体步骤的能力，还具备将复杂命令和反馈置于物理世界中的能力。在本工作中，我们描述了一个层级结构中使用视觉语言模型的系统，首先通过推理复杂的提示和用户反馈来推断出最合适的下一步以完成任务，然后通过低级动作执行该步骤。与可以执行简单命令（如“拿起杯子”）的直接指令跟随方法不同，我们的系统能够通过复杂的提示进行推理，并在任务执行过程中整合位置反馈（如“那不是垃圾”）。我们跨三个机器人平台评估了该系统，包括单臂、双臂以及双臂移动机器人，展示了其处理诸如清理脏桌子、制作三明治和购物等任务的能力。 

---
# Norm Growth and Stability Challenges in Localized Sequential Knowledge Editing 

**Title (ZH)**: 局部序贯知识编辑中的范式增长与稳定性挑战 

**Authors**: Akshat Gupta, Christine Fang, Atahan Ozdemir, Maochuan Lu, Ahmed Alaa, Thomas Hartvigsen, Gopala Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2502.19416)  

**Abstract**: This study investigates the impact of localized updates to large language models (LLMs), specifically in the context of knowledge editing - a task aimed at incorporating or modifying specific facts without altering broader model capabilities. We first show that across different post-training interventions like continuous pre-training, full fine-tuning and LORA-based fine-tuning, the Frobenius norm of the updated matrices always increases. This increasing norm is especially detrimental for localized knowledge editing, where only a subset of matrices are updated in a model . We reveal a consistent phenomenon across various editing techniques, including fine-tuning, hypernetwork-based approaches, and locate-and-edit methods: the norm of the updated matrix invariably increases with successive updates. Such growth disrupts model balance, particularly when isolated matrices are updated while the rest of the model remains static, leading to potential instability and degradation of downstream performance. Upon deeper investigations of the intermediate activation vectors, we find that the norm of internal activations decreases and is accompanied by shifts in the subspaces occupied by these activations, which shows that these activation vectors now occupy completely different regions in the representation space compared to the unedited model. With our paper, we highlight the technical challenges with continuous and localized sequential knowledge editing and their implications for maintaining model stability and utility. 

**Abstract (ZH)**: 本研究调查了大型语言模型（LLMs）局部更新对其知识编辑任务的影响，知识编辑旨在 incorporaring 或修改特定事实而不改变模型的 broader 能力。我们首先表明，在不同的后训练干预措施如连续预训练、全量微调和 LORA 基础微调中，更新矩阵的 Frobenius 范数始终增加。这种范数的增加对局部知识编辑尤为不利，在这种编辑中只有模型的一部分矩阵被更新。我们揭示了各种编辑技术（包括微调、超网络方法和查找并编辑方法）中的一致现象：随着更新次数的增加，更新矩阵的范数不可避免地增加。这种增长破坏了模型的平衡，特别是在仅更新孤立矩阵而其余模型保持不变的情况下，可能导致潜在的不稳定性和下游性能退化。通过对中间激活向量的进一步研究，我们发现内部激活的范数减小，并且伴随这些激活所在的子空间发生变化，显示这些激活向量现在在表示空间中占据了完全不同的区域，与未编辑的模型完全不同。通过本文，我们突出显示了连续和局部顺序知识编辑的技术挑战及其对保持模型稳定性和实用性的影响。 

---
# Project Alexandria: Towards Freeing Scientific Knowledge from Copyright Burdens via LLMs 

**Title (ZH)**: 项目亚历山大：通过大语言模型解放科学知识的版权束缚 

**Authors**: Christoph Schuhmann, Gollam Rabby, Ameya Prabhu, Tawsif Ahmed, Andreas Hochlehnert, Huu Nguyen, Nick Akinci Heidrich, Ludwig Schmidt, Robert Kaczmarczyk, Sören Auer, Jenia Jitsev, Matthias Bethge  

**Link**: [PDF](https://arxiv.org/pdf/2502.19413)  

**Abstract**: Paywalls, licenses and copyright rules often restrict the broad dissemination and reuse of scientific knowledge. We take the position that it is both legally and technically feasible to extract the scientific knowledge in scholarly texts. Current methods, like text embeddings, fail to reliably preserve factual content, and simple paraphrasing may not be legally sound. We urge the community to adopt a new idea: convert scholarly documents into Knowledge Units using LLMs. These units use structured data capturing entities, attributes and relationships without stylistic content. We provide evidence that Knowledge Units: (1) form a legally defensible framework for sharing knowledge from copyrighted research texts, based on legal analyses of German copyright law and U.S. Fair Use doctrine, and (2) preserve most (~95%) factual knowledge from original text, measured by MCQ performance on facts from the original copyrighted text across four research domains. Freeing scientific knowledge from copyright promises transformative benefits for scientific research and education by allowing language models to reuse important facts from copyrighted text. To support this, we share open-source tools for converting research documents into Knowledge Units. Overall, our work posits the feasibility of democratizing access to scientific knowledge while respecting copyright. 

**Abstract (ZH)**: 付墙、许可协议和版权规则往往限制了科学知识的广泛传播和再利用。我们认为，从学术文本中提取科学知识在法律和技术上都是可行的。当前的方法，如文本嵌入，无法可靠地保留事实内容，而简单的改写可能不符合法律要求。我们呼吁学术界采纳一个新思路：使用大规模语言模型将学术文档转换为知识单元。这些单元使用结构化数据来捕捉实体、属性和关系，而不包含风格内容。我们提供了证据表明，知识单元：（1）基于德国版权法和美国公平使用原则的法律分析，构成了分享受版权的研究文本知识的合法框架；（2）在四个研究领域内，通过多项选择题测试事实的性能衡量，保留了约95%的原始事实知识。释放科学知识的版权有望对科学研究和教育产生变革性的影响，这将允许语言模型重新使用受版权保护文本的重要事实。为此，我们分享了将研究文档转换为知识单元的开源工具。总体而言，我们的工作提出了在尊重版权的同时使科学知识民主化访问的可能性。 

---
# Code to Think, Think to Code: A Survey on Code-Enhanced Reasoning and Reasoning-Driven Code Intelligence in LLMs 

**Title (ZH)**: 代码促进思考，思考驱动代码：基于代码增强的推理与推理驱动的代码智能综述 

**Authors**: Dayu Yang, Tianyang Liu, Daoan Zhang, Antoine Simoulin, Xiaoyi Liu, Yuwei Cao, Zhaopu Teng, Xin Qian, Grey Yang, Jiebo Luo, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2502.19411)  

**Abstract**: In large language models (LLMs), code and reasoning reinforce each other: code offers an abstract, modular, and logic-driven structure that supports reasoning, while reasoning translates high-level goals into smaller, executable steps that drive more advanced code intelligence. In this study, we examine how code serves as a structured medium for enhancing reasoning: it provides verifiable execution paths, enforces logical decomposition, and enables runtime validation. We also explore how improvements in reasoning have transformed code intelligence from basic completion to advanced capabilities, enabling models to address complex software engineering tasks through planning and debugging. Finally, we identify key challenges and propose future research directions to strengthen this synergy, ultimately improving LLM's performance in both areas. 

**Abstract (ZH)**: 在大型语言模型中，代码和推理相互增强：代码提供了一个抽象的、模块化的、逻辑驱动的结构，支持推理，而推理将高层次的目标转化为更小的、可执行的步骤，驱动更高级的代码智能。在本研究中，我们考察代码作为增强推理的结构化媒介的作用：它提供了可验证的执行路径，强制进行逻辑分解，并使运行时验证成为可能。我们还探讨了推理改进如何将代码智能从基本的完成转变为高级能力，从而通过计划和调试使模型能够应对复杂的软件工程任务。最后，我们确定了关键挑战，并提出了未来的研究方向，以加强这种协同作用，最终在两个领域提升LLM的性能。 

---
# Less or More: Towards Glanceable Explanations for LLM Recommendations Using Ultra-Small Devices 

**Title (ZH)**: 少一点还是多一点：面向超小型设备的LLM推荐精要解释研究 

**Authors**: Xinru Wang, Mengjie Yu, Hannah Nguyen, Michael Iuzzolino, Tianyi Wang, Peiqi Tang, Natasha Lynova, Co Tran, Ting Zhang, Naveen Sendhilnathan, Hrvoje Benko, Haijun Xia, Tanya Jonker  

**Link**: [PDF](https://arxiv.org/pdf/2502.19410)  

**Abstract**: Large Language Models (LLMs) have shown remarkable potential in recommending everyday actions as personal AI assistants, while Explainable AI (XAI) techniques are being increasingly utilized to help users understand why a recommendation is given. Personal AI assistants today are often located on ultra-small devices such as smartwatches, which have limited screen space. The verbosity of LLM-generated explanations, however, makes it challenging to deliver glanceable LLM explanations on such ultra-small devices. To address this, we explored 1) spatially structuring an LLM's explanation text using defined contextual components during prompting and 2) presenting temporally adaptive explanations to users based on confidence levels. We conducted a user study to understand how these approaches impacted user experiences when interacting with LLM recommendations and explanations on ultra-small devices. The results showed that structured explanations reduced users' time to action and cognitive load when reading an explanation. Always-on structured explanations increased users' acceptance of AI recommendations. However, users were less satisfied with structured explanations compared to unstructured ones due to their lack of sufficient, readable details. Additionally, adaptively presenting structured explanations was less effective at improving user perceptions of the AI compared to the always-on structured explanations. Together with users' interview feedback, the results led to design implications to be mindful of when personalizing the content and timing of LLM explanations that are displayed on ultra-small devices. 

**Abstract (ZH)**: 超小型设备上可解释的大语言模型推荐及解释的研究 

---
# Multi-modal Contrastive Learning for Tumor-specific Missing Modality Synthesis 

**Title (ZH)**: 多模态对比学习在肿瘤特异性缺失模态合成中的应用 

**Authors**: Minjoo Lim, Bogyeong Kang, Tae-Eui Kam  

**Link**: [PDF](https://arxiv.org/pdf/2502.19390)  

**Abstract**: Multi-modal magnetic resonance imaging (MRI) is essential for providing complementary information about brain anatomy and pathology, leading to more accurate diagnoses. However, obtaining high-quality multi-modal MRI in a clinical setting is difficult due to factors such as time constraints, high costs, and patient movement artifacts. To overcome this difficulty, there is increasing interest in developing generative models that can synthesize missing target modality images from the available source ones. Therefore, we design a generative model for missing MRI that integrates multi-modal contrastive learning with a focus on critical tumor regions. Specifically, we integrate multi-modal contrastive learning, tailored for multiple source modalities, and enhance its effectiveness by selecting features based on entropy during the contrastive learning process. Additionally, our network not only generates the missing target modality images but also predicts segmentation outputs, simultaneously. This approach improves the generator's capability to precisely generate tumor regions, ultimately improving performance in downstream segmentation tasks. By leveraging a combination of contrastive, segmentation, and additional self-representation losses, our model effectively reflects target-specific information and generate high-quality target images. Consequently, our results in the Brain MR Image Synthesis challenge demonstrate that the proposed model excelled in generating the missing modality. 

**Abstract (ZH)**: 多模态磁共振成像（MRI）对于提供关于脑解剖和病理的补充信息至关重要，有助于更准确的诊断。然而，在临床环境中获得高质量的多模态MRI受到时间限制、高成本和患者运动伪影等因素的影响。为克服这一困难，近年来越来越多的研究兴趣集中在开发生成模型，可以利用现有源模态合成缺失的目标模态图像。因此，我们设计了一种集成多模态对比学习的生成模型，重点关注关键肿瘤区域。具体而言，我们整合了针对多种源模态定制的多模态对比学习，并在其对比学习过程中通过选择基于熵的特征来增强其效果。此外，我们的网络不仅生成缺失的目标模态图像，还同时预测分割输出。这种方法提高了生成器对肿瘤区域精确生成的能力，最终提高了下游分割任务的性能。通过利用对比、分割以及额外的自我表征损失的组合，我们的模型有效地反映了目标特定的信息并生成高质量的目标图像。因此，在脑MR图像合成挑战中，我们的结果表明所提出的模型在生成缺失模态方面表现出色。 

---
# Efficient 4D fMRI ASD Classification using Spatial-Temporal-Omics-based Learning Framework 

**Title (ZH)**: 基于空间-时间-组学学习框架的高效4D fMRI ASD分类 

**Authors**: Ziqiao Weng, Weidong Cai, Bo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.19386)  

**Abstract**: Autism Spectrum Disorder (ASD) is a neurodevelopmental disorder impacting social and behavioral development. Resting-state fMRI, a non-invasive tool for capturing brain connectivity patterns, aids in early ASD diagnosis and differentiation from typical controls (TC). However, previous methods, which rely on either mean time series or full 4D data, are limited by a lack of spatial information or by high computational costs. This underscores the need for an efficient solution that preserves both spatial and temporal information. In this paper, we propose a novel, simple, and efficient spatial-temporal-omics learning framework designed to efficiently extract spatio-temporal features from fMRI for ASD classification. Our approach addresses these limitations by utilizing 3D time-domain derivatives as the spatial-temporal inter-voxel omics, which preserve full spatial resolution while capturing diverse statistical characteristics of the time series at each voxel. Meanwhile, functional connectivity features serve as the spatial-temporal inter-regional omics, capturing correlations across brain regions. Extensive experiments and ablation studies on the ABIDE dataset demonstrate that our framework significantly outperforms previous methods while maintaining computational efficiency. We believe our research offers valuable insights that will inform and advance future ASD studies, particularly in the realm of spatial-temporal-omics-based learning. 

**Abstract (ZH)**: 自闭症谱系障碍（ASD）是一种影响社会和行为发展的神经发育障碍。静息态fMRI是一种无创的工具，用于捕获脑连接模式，有助于早期ASD诊断并区分典型对照组（TC）。然而，之前的 方法依赖于平均时间序列或完整的4D数据，分别受限于缺乏空间信息和高计算成本。这强调了需要一种有效的方法来同时保留空间和时间信息。在本文中，我们提出了一种新颖的简单高效的空间-时间-组学学习框架，用于从fMRI中高效提取空间-时间特征进行ASD分类。我们的方法通过利用3D时间域导数作为空间-时间体素组学，同时保留了全空间分辨率并捕获每个体素的时间序列的多样化统计特征，解决了这些限制。同时，功能连接特征作为空间-时间区域组学，捕获了脑区间的相关性。在ABIDE数据集上的 extensive 实验和消融研究证明，我们的框架在保持计算效率的同时显著优于之前的方法。我们认为我们的研究提供了有价值的认识，将指导并推进未来的ASD研究，特别是在空间-时间-组学基础上的学习。 

---
# Preference-Based Gradient Estimation for ML-Based Approximate Combinatorial Optimization 

**Title (ZH)**: 基于偏好的梯度估计方法在基于机器学习的组合优化近似算法中的应用 

**Authors**: Arman Mielke, Uwe Bauknecht, Thilo Strauss, Mathias Niepert  

**Link**: [PDF](https://arxiv.org/pdf/2502.19377)  

**Abstract**: Combinatorial optimization (CO) problems arise in a wide range of fields from medicine to logistics and manufacturing. While exact solutions are often not necessary, many applications require finding high-quality solutions quickly. For this purpose, we propose a data-driven approach to improve existing non-learned approximation algorithms for CO. We parameterize the approximation algorithm and train a graph neural network (GNN) to predict parameter values that lead to the best possible solutions. Our pipeline is trained end-to-end in a self-supervised fashion using gradient estimation, treating the approximation algorithm as a black box. We propose a novel gradient estimation scheme for this purpose, which we call preference-based gradient estimation. Our approach combines the benefits of the neural network and the non-learned approximation algorithm: The GNN leverages the information from the dataset to allow the approximation algorithm to find better solutions, while the approximation algorithm guarantees that the solution is feasible. We validate our approach on two well-known combinatorial optimization problems, the travelling salesman problem and the minimum k-cut problem, and show that our method is competitive with state of the art learned CO solvers. 

**Abstract (ZH)**: 数据驱动的组合优化非学习近似算法改进方法 

---
# DataMan: Data Manager for Pre-training Large Language Models 

**Title (ZH)**: DataMan: 大型语言模型预训练的数据管理器 

**Authors**: Ru Peng, Kexin Yang, Yawen Zeng, Junyang Lin, Dayiheng Liu, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.19363)  

**Abstract**: The performance emergence of large language models (LLMs) driven by data scaling laws makes the selection of pre-training data increasingly important. However, existing methods rely on limited heuristics and human intuition, lacking comprehensive and clear guidelines. To address this, we are inspired by ``reverse thinking'' -- prompting LLMs to self-identify which criteria benefit its performance. As its pre-training capabilities are related to perplexity (PPL), we derive 14 quality criteria from the causes of text perplexity anomalies and introduce 15 common application domains to support domain mixing. In this paper, we train a Data Manager (DataMan) to learn quality ratings and domain recognition from pointwise rating, and use it to annotate a 447B token pre-training corpus with 14 quality ratings and domain type. Our experiments validate our approach, using DataMan to select 30B tokens to train a 1.3B-parameter language model, demonstrating significant improvements in in-context learning (ICL), perplexity, and instruction-following ability over the state-of-the-art baseline. The best-performing model, based on the Overall Score l=5 surpasses a model trained with 50% more data using uniform sampling. We continue pre-training with high-rated, domain-specific data annotated by DataMan to enhance domain-specific ICL performance and thus verify DataMan's domain mixing ability. Our findings emphasize the importance of quality ranking, the complementary nature of quality criteria, and their low correlation with perplexity, analyzing misalignment between PPL and ICL performance. We also thoroughly analyzed our pre-training dataset, examining its composition, the distribution of quality ratings, and the original document sources. 

**Abstract (ZH)**: 大型语言模型（LLM）性能的提升得益于数据规模法则，这使得预训练数据的选择越来越重要。然而，现有方法依赖于有限的启发式方法和人类直觉，缺乏全面和明确的指导方针。为了解决这一问题，我们借鉴了“反向思维”——提示LLM自我识别哪些标准对其性能有益。因为其预训练能力与困惑度（PPL）相关，我们从文本困惑度异常的原因中推导出14个质量标准，并引入了15个常见应用场景以支持领域混合。在本文中，我们训练了一个数据管理器（DataMan），使其从点wise评分中学习质量评级和领域识别，并使用它为一个447B令牌的预训练语料库标注14个质量评级和领域类型。我们的实验验证了该方法的有效性，使用DataMan选择30B令牌训练一个1.3B参数的语言模型，展示了在上下文学习（ICL）、困惑度和指令遵循能力方面相对于最先进的基线的显著改进。基于综合评分l=5的最佳模型超过了使用50%更多数据通过均匀抽样训练的模型。我们继续使用DataMan标注的高评分、领域特定数据进行预训练，以增强特定领域的ICL性能，从而验证了DataMan的领域混合能力。我们的研究强调了质量排名的重要性、质量标准之间的互补性以及它们与困惑度的低相关性，并分析了PPL与ICL性能之间的不一致性。我们还详细分析了预训练数据集的组成、质量评分的分布以及原始文档来源。 

---
# Physics-Based Hybrid Machine Learning for Critical Heat Flux Prediction with Uncertainty Quantification 

**Title (ZH)**: 基于物理的混合机器学习方法在不确定性量化条件下的critical heat flux预测 

**Authors**: Aidan Furlong, Xingang Zhao, Robert Salko, Xu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.19357)  

**Abstract**: Critical heat flux is a key quantity in boiling system modeling due to its impact on heat transfer and component temperature and performance. This study investigates the development and validation of an uncertainty-aware hybrid modeling approach that combines machine learning with physics-based models in the prediction of critical heat flux in nuclear reactors for cases of dryout. Two empirical correlations, Biasi and Bowring, were employed with three machine learning uncertainty quantification techniques: deep neural network ensembles, Bayesian neural networks, and deep Gaussian processes. A pure machine learning model without a base model served as a baseline for comparison. This study examines the performance and uncertainty of the models under both plentiful and limited training data scenarios using parity plots, uncertainty distributions, and calibration curves. The results indicate that the Biasi hybrid deep neural network ensemble achieved the most favorable performance (with a mean absolute relative error of 1.846% and stable uncertainty estimates), particularly in the plentiful data scenario. The Bayesian neural network models showed slightly higher error and uncertainty but superior calibration. By contrast, deep Gaussian process models underperformed by most metrics. All hybrid models outperformed pure machine learning configurations, demonstrating resistance against data scarcity. 

**Abstract (ZH)**: 临界热流密度是沸腾系统建模中的关键量，因其对传热及组件温度和性能的影响。本文研究了一种结合物理模型和机器学习的不确定认知混合建模方法在核反应堆干涸情况下预测临界热流密度的发展与验证。采用了Biasi和Bowring两个经验关联式，并结合了三种机器学习不确定性量化技术：深度神经网络集成、贝叶斯神经网络和深度高斯过程。无基础模型的纯机器学习模型作为基准进行对比。本文通过等效图、不确定性分布和校准曲线，评估模型在丰裕和有限训练数据条件下的性能和不确定性。结果表明，在丰裕数据情况下，Biasi混合深度神经网络集成取得了最优性能（平均绝对相对误差为1.846%，且不确定性估计稳定）。贝叶斯神经网络模型显示出稍高的误差和不确定性，但具有更好的校准性能。相比之下，深度高斯过程模型在大多数指标下表现较差。所有混合模型均优于纯机器学习配置，显示出对数据稀疏性的抗御能力。 

---
# Deep Learning-Based Transfer Learning for Classification of Cassava Disease 

**Title (ZH)**: 基于深度学习的迁移学习在甘薯疾病分类中的应用 

**Authors**: Ademir G. Costa Junior, Fábio S. da Silva, Ricardo Rios  

**Link**: [PDF](https://arxiv.org/pdf/2502.19351)  

**Abstract**: This paper presents a performance comparison among four Convolutional Neural Network architectures (EfficientNet-B3, InceptionV3, ResNet50, and VGG16) for classifying cassava disease images. The images were sourced from an imbalanced dataset from a competition. Appropriate metrics were employed to address class imbalance. The results indicate that EfficientNet-B3 achieved on this task accuracy of 87.7%, precision of 87.8%, revocation of 87.8% and F1-Score of 87.7%. These findings suggest that EfficientNet-B3 could be a valuable tool to support Digital Agriculture. 

**Abstract (ZH)**: 本文比较了四种卷积神经网络架构（EfficientNet-B3、InceptionV3、ResNet50和VGG16）在分类甘蔗病害图像中的性能。所用图像来自于一个竞赛的不平衡数据集。使用适当的指标解决类别不平衡问题。结果表明，EfficientNet-B3在该项任务中的准确率为87.7%，精确率为87.8%，召回率为87.8%，F1分数为87.7%。这些发现表明，EfficientNet-B3可能是支持数字农业的一个有价值工具。 

---
# Controlled Diversity: Length-optimized Natural Language Generation 

**Title (ZH)**: 控制多样性：长度优化的自然语言生成 

**Authors**: Diana Marie Schenke, Timo Baumann  

**Link**: [PDF](https://arxiv.org/pdf/2502.19347)  

**Abstract**: LLMs are not generally able to adjust the length of their outputs based on strict length requirements, a capability that would improve their usefulness in applications that require adherence to diverse user and system requirements. We present an approach to train LLMs to acquire this capability by augmenting existing data and applying existing fine-tuning techniques, which we compare based on the trained models' adherence to the length requirement and overall response quality relative to the baseline model. Our results demonstrate that these techniques can be successfully applied to train LLMs to adhere to length requirements, with the trained models generating texts which better align to the length requirements. Our results indicate that our method may change the response quality when using training data that was not generated by the baseline model. This allows simultaneous alignment to another training objective in certain scenarios, but is undesirable otherwise. Training on a dataset containing the model's own responses eliminates this issue. 

**Abstract (ZH)**: LLMs通常无法根据严格的长度要求调整其输出长度，这种能力将提高它们在需要遵守多样用户和系统要求的应用中的实用性。我们提出了一种通过扩展现有数据并应用现有微调技术来培训LLMs获得这种能力的方法，并基于训练模型遵守长度要求的程度和整体响应质量与基线模型的比较进行评估。我们的结果表明，这些技术可以成功应用于培训LLMs以遵守长度要求，从而使生成的文本更好地符合长度要求。我们的结果表明，在使用非基线模型生成的数据集上进行训练可能会改变响应质量。这在某些情况下可以同时满足另一个训练目标，但在其他情况下是不希望的。在包含模型自身响应的数据集上进行训练可以解决这一问题。 

---
# Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals for Reliable Reward Systems 

**Title (ZH)**: 代理人奖励建模：结合可验证正确性信号的人类偏好以构建可靠的奖励系统 

**Authors**: Hao Peng, Yunjia Qi, Xiaozhi Wang, Zijun Yao, Bin Xu, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.19328)  

**Abstract**: Reward models (RMs) are crucial for the training and inference-time scaling up of large language models (LLMs). However, existing reward models primarily focus on human preferences, neglecting verifiable correctness signals which have shown strong potential in training LLMs. In this paper, we propose agentic reward modeling, a reward system that combines reward models with verifiable correctness signals from different aspects to provide reliable rewards. We empirically implement a reward agent, named RewardAgent, that combines human preference rewards with two verifiable signals: factuality and instruction following, to provide more reliable rewards. We conduct comprehensive experiments on existing reward model benchmarks and inference time best-of-n searches on real-world downstream tasks. RewardAgent significantly outperforms vanilla reward models, demonstrating its effectiveness. We further construct training preference pairs using RewardAgent and train an LLM with the DPO objective, achieving superior performance on various NLP benchmarks compared to conventional reward models. Our codes are publicly released to facilitate further research (this https URL). 

**Abstract (ZH)**: 基于可验证正确性的代理奖励模型 

---
# Partition Tree Weighting for Non-Stationary Stochastic Bandits 

**Title (ZH)**: 非平稳随机臂问题的分区树加权方法 

**Authors**: Joel Veness, Marcus Hutter, Andras Gyorgy, Jordi Grau-Moya  

**Link**: [PDF](https://arxiv.org/pdf/2502.19325)  

**Abstract**: This paper considers a generalisation of universal source coding for interaction data, namely data streams that have actions interleaved with observations. Our goal will be to construct a coding distribution that is both universal \emph{and} can be used as a control policy. Allowing for action generation needs careful treatment, as naive approaches which do not distinguish between actions and observations run into the self-delusion problem in universal settings. We showcase our perspective in the context of the challenging non-stationary stochastic Bernoulli bandit problem. Our main contribution is an efficient and high performing algorithm for this problem that generalises the Partition Tree Weighting universal source coding technique for passive prediction to the control setting. 

**Abstract (ZH)**: 这篇论文考虑了一类通用源编码的拓展，即包含动作与观测交织的数据流。我们的目标是构建一个既通用又能作为控制策略的编码分布。允许动作生成需要谨慎处理，因为在通用环境中，不区分动作与观测的天真方法会遇到自我欺骗问题。我们通过非平稳随机伯努利臂部环境这一具有挑战性的例子展示了这一视角。我们的主要贡献是一种针对该问题的高效且高性能的算法，它将被动预测中的分区树加权通用源编码技术推广到了控制环境。 

---
# Shh, don't say that! Domain Certification in LLMs 

**Title (ZH)**: Shh, don't say that! LLMs的领域认证 

**Authors**: Cornelius Emde, Alasdair Paren, Preetham Arvind, Maxime Kayser, Tom Rainforth, Thomas Lukasiewicz, Bernard Ghanem, Philip H.S. Torr, Adel Bibi  

**Link**: [PDF](https://arxiv.org/pdf/2502.19320)  

**Abstract**: Large language models (LLMs) are often deployed to perform constrained tasks, with narrow domains. For example, customer support bots can be built on top of LLMs, relying on their broad language understanding and capabilities to enhance performance. However, these LLMs are adversarially susceptible, potentially generating outputs outside the intended domain. To formalize, assess, and mitigate this risk, we introduce domain certification; a guarantee that accurately characterizes the out-of-domain behavior of language models. We then propose a simple yet effective approach, which we call VALID that provides adversarial bounds as a certificate. Finally, we evaluate our method across a diverse set of datasets, demonstrating that it yields meaningful certificates, which bound the probability of out-of-domain samples tightly with minimum penalty to refusal behavior. 

**Abstract (ZH)**: 大型语言模型（LLMs）常被部署执行受限任务，领域狭窄。例如，可以在LLMs之上构建客服机器人，依赖其广泛的语言理解和能力以提升性能。然而，这些LLMs对抗性易受攻击，可能会生成超出预定领域的输出。为此，我们通过引入领域认证来正式化、评估和缓解这一风险；领域认证是一种保证，准确描述语言模型的越域行为。我们提出了一种简单而有效的方法称为VALID，提供对抗性界值作为证书。我们最终在多样化的数据集上评估了该方法，展示了其能够紧密界定制外样本的概率，同时对拒绝行为的代价最小。 

---
# FSPO: Few-Shot Preference Optimization of Synthetic Preference Data in LLMs Elicits Effective Personalization to Real Users 

**Title (ZH)**: FSPO: 少Shot偏好优化的合成偏好数据在大规模语言模型中的有效个性化 

**Authors**: Anikait Singh, Sheryl Hsu, Kyle Hsu, Eric Mitchell, Stefano Ermon, Tatsunori Hashimoto, Archit Sharma, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2502.19312)  

**Abstract**: Effective personalization of LLMs is critical for a broad range of user-interfacing applications such as virtual assistants and content curation. Inspired by the strong in-context learning capabilities of LLMs, we propose Few-Shot Preference Optimization (FSPO), which reframes reward modeling as a meta-learning problem. Under this framework, an LLM learns to quickly adapt to a user via a few labeled preferences from that user, constructing a personalized reward function for them. Additionally, since real-world preference data is scarce and challenging to collect at scale, we propose careful design choices to construct synthetic preference datasets for personalization, generating over 1M synthetic personalized preferences using publicly available LLMs. In particular, to successfully transfer from synthetic data to real users, we find it crucial for the data to exhibit both high diversity and coherent, self-consistent structure. We evaluate FSPO on personalized open-ended generation for up to 1,500 synthetic users across across three domains: movie reviews, pedagogical adaptation based on educational background, and general question answering, along with a controlled human study. Overall, FSPO achieves an 87% Alpaca Eval winrate on average in generating responses that are personalized to synthetic users and a 72% winrate with real human users in open-ended question answering. 

**Abstract (ZH)**: Few-Shot Preference Optimization for Effective Personalization of LLMs 

---
# Faithful Logic Embeddings in HOL -- A recipe to have it all: deep and shallow, automated and interactive, heavy and light, proofs and counterexamples, meta and object level 

**Title (ZH)**: HOL中忠实的逻辑嵌入——兼备深度与浅度、自动化与交互性、复杂与简洁、证明与反例、元层次与对象层次的配方 

**Authors**: Christoph Benzmüller  

**Link**: [PDF](https://arxiv.org/pdf/2502.19311)  

**Abstract**: Deep and shallow embeddings of non-classical logics in classical higher-order logic have been explored, implemented, and used in various automated reasoning tools in recent years. This paper presents a recipe for the simultaneous deployment of different forms of deep and shallow embeddings in classical higher-order logic, enabling not only flexible interactive and automated theorem proving and counterexample finding at meta and object level, but also automated faithfulness proofs between the logic embeddings. The approach, which is fruitful for logic education, research and application, is deliberately illustrated here using simple propositional modal logic. However, the work presented is conceptual in nature and not limited to such a simple logic context. 

**Abstract (ZH)**: 在经典高阶逻辑中同时部署不同形式的深度和浅层嵌入的技术已经在近年来的各种自动推理工具中被探索、实现并使用。本文提出了一种同时部署不同形式的深度和浅层嵌入的技术，不仅在元级和对象级实现了灵活的交互式和自动定理证明及反例查找，还实现了逻辑嵌入之间的自动忠实性证明。这种方法对于逻辑教育、研究和应用具有重要意义，并通过简单的命题模态逻辑进行了详细说明。然而，本研究的概念性工作不限于如此简单的逻辑环境。 

---
# Anomaly Detection in Complex Dynamical Systems: A Systematic Framework Using Embedding Theory and Physics-Inspired Consistency 

**Title (ZH)**: 复杂动力系统中的异常检测：一种基于嵌入理论和物理启发一致性系统框架 

**Authors**: Michael Somma, Thomas Gallien, Branka Stojanovic  

**Link**: [PDF](https://arxiv.org/pdf/2502.19307)  

**Abstract**: Anomaly detection in complex dynamical systems is essential for ensuring reliability, safety, and efficiency in industrial and cyber-physical infrastructures. Predictive maintenance helps prevent costly failures, while cybersecurity monitoring has become critical as digitized systems face growing threats. Many of these systems exhibit oscillatory behaviors and bounded motion, requiring anomaly detection methods that capture structured temporal dependencies while adhering to physical consistency principles. In this work, we propose a system-theoretic approach to anomaly detection, grounded in classical embedding theory and physics-inspired consistency principles. We build upon the Fractal Whitney Embedding Prevalence Theorem, extending traditional embedding techniques to complex system dynamics. Additionally, we introduce state-derivative pairs as an embedding strategy to capture system evolution. To enforce temporal coherence, we develop a Temporal Differential Consistency Autoencoder (TDC-AE), incorporating a TDC-Loss that aligns the approximated derivatives of latent variables with their dynamic representations. We evaluate our method on the C-MAPSS dataset, a benchmark for turbofan aeroengine degradation. TDC-AE outperforms LSTMs and Transformers while achieving a 200x reduction in MAC operations, making it particularly suited for lightweight edge computing. Our findings support the hypothesis that anomalies disrupt stable system dynamics, providing a robust, interpretable signal for anomaly detection. 

**Abstract (ZH)**: 复杂动力系统中的异常检测对于确保工业和物理 cyber-基础设施的可靠性和安全性以及提高效率是必不可少的。预测性维护有助于预防昂贵的故障，而随着数字化系统的威胁不断增加，网络安全监控已经成为关键。许多这些系统表现出振荡行为和有界运动，因此需要能够捕捉结构化时间依赖性并遵循物理一致性原则的异常检测方法。在本工作中，我们提出了一种基于经典嵌入理论和物理启发的一致性原则的系统理论方法来进行异常检测。我们基于分形Whitney嵌入普遍定理，将传统嵌入技术扩展到复杂系统动力学中，并引入状态-导数对作为嵌入策略来捕捉系统演化。为了确保时间连贯性，我们开发了一种时间差分一致性自编码器（TDC-AE），并引入了TDC损失，该损失使潜变量的近似导数与其动态表示保持一致。我们在C-MAPSS数据集上评估了该方法，该数据集是涡扇发动机退化的基准数据集。TDC-AE 在准确性和计算效率上均优于LSTMs和Transformers，尤其是其MAC操作减少了200倍，使其特别适合轻量级边缘计算。我们的研究结果支持异常破坏稳定系统动力学的假设，从而为异常检测提供了稳健且可解释的信号。 

---
# Corporate Fraud Detection in Rich-yet-Noisy Financial Graph 

**Title (ZH)**: 富且嘈杂的财务图谱中的公司欺诈检测 

**Authors**: Shiqi Wang, Zhibo Zhang, Libing Fang, Cam-Tu Nguyen, Wenzhon Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.19305)  

**Abstract**: Corporate fraud detection aims to automatically recognize companies that conduct wrongful activities such as fraudulent financial statements or illegal insider trading. Previous learning-based methods fail to effectively integrate rich interactions in the company network. To close this gap, we collect 18-year financial records in China to form three graph datasets with fraud labels. We analyze the characteristics of the financial graphs, highlighting two pronounced issues: (1) information overload: the dominance of (noisy) non-company nodes over company nodes hinders the message-passing process in Graph Convolution Networks (GCN); and (2) hidden fraud: there exists a large percentage of possible undetected violations in the collected data. The hidden fraud problem will introduce noisy labels in the training dataset and compromise fraud detection results. To handle such challenges, we propose a novel graph-based method, namely, Knowledge-enhanced GCN with Robust Two-stage Learning (${\rm KeGCN}_{R}$), which leverages Knowledge Graph Embeddings to mitigate the information overload and effectively learns rich representations. The proposed model adopts a two-stage learning method to enhance robustness against hidden frauds. Extensive experimental results not only confirm the importance of interactions but also show the superiority of ${\rm KeGCN}_{R}$ over a number of strong baselines in terms of fraud detection effectiveness and robustness. 

**Abstract (ZH)**: 企业欺诈检测旨在自动识别出 conducts wrongful activities 如欺诈财务报表或非法内幕交易的公司。以往基于学习的方法未能有效地整合公司在网络中的丰富交互。为了弥合这一差距，我们收集了18年的中国财务记录，形成了具有欺诈标签的三个图数据集。我们分析了财务图的特点，突出了两个显著问题：（1）信息过载：无用节点占主导地位阻碍了图卷积网络（GCN）中的信息传递过程；（2）隐匿欺诈：收集的数据中存在大量未被检测到的违规行为。隐匿欺诈问题将在训练数据集中引入噪声标签，影响欺诈检测结果。为了应对这些挑战，我们提出了一种新颖的图方法，即增强知识的两阶段学习图卷积网络（${\rm KeGCN}_{R}$），该方法利用知识图嵌入以减轻信息过载，并有效学习丰富表示。所提出的模型采用两阶段学习方法以增强对隐匿欺诈的鲁棒性。广泛的实验结果不仅证实了交互的重要性，而且还展示了${\rm KeGCN}_{R}$在欺诈检测有效性及鲁棒性方面优于多个强大基线模型。 

---
# Combining Planning and Reinforcement Learning for Solving Relational Multiagent Domains 

**Title (ZH)**: 结合规划与强化学习解决关系型多agent领域 

**Authors**: Nikhilesh Prabhakar, Ranveer Singh, Harsha Kokel, Sriraam Natarajan, Prasad Tadepalli  

**Link**: [PDF](https://arxiv.org/pdf/2502.19297)  

**Abstract**: Multiagent Reinforcement Learning (MARL) poses significant challenges due to the exponential growth of state and action spaces and the non-stationary nature of multiagent environments. This results in notable sample inefficiency and hinders generalization across diverse tasks. The complexity is further pronounced in relational settings, where domain knowledge is crucial but often underutilized by existing MARL algorithms. To overcome these hurdles, we propose integrating relational planners as centralized controllers with efficient state abstractions and reinforcement learning. This approach proves to be sample-efficient and facilitates effective task transfer and generalization. 

**Abstract (ZH)**: 多智能体强化学习（MARL）由于状态空间和动作空间的指数级增长以及多智能体环境的非稳态性质，面临着重大挑战。这导致显著的样本效率低下，并阻碍了在不同任务上的泛化。在关系型设置中，这一复杂性进一步加剧，领域知识至关重要但常常被现有MARL算法所忽视。为克服这些难题，我们提出将关系规划者作为拥有高效状态抽象的集中式控制器与强化学习相结合的方法。该方法证明具有样本高效性，并促进有效的任务迁移和泛化。 

---
# Integrating Biological and Machine Intelligence: Attention Mechanisms in Brain-Computer Interfaces 

**Title (ZH)**: 融合生物智能与机器智能：脑机接口中的注意力机制 

**Authors**: Jiyuan Wang, Weishan Ye, Jialin He, Li Zhang, Gan Huang, Zhuliang Yu, Zhen Liang  

**Link**: [PDF](https://arxiv.org/pdf/2502.19281)  

**Abstract**: With the rapid advancement of deep learning, attention mechanisms have become indispensable in electroencephalography (EEG) signal analysis, significantly enhancing Brain-Computer Interface (BCI) applications. This paper presents a comprehensive review of traditional and Transformer-based attention mechanisms, their embedding strategies, and their applications in EEG-based BCI, with a particular emphasis on multimodal data fusion. By capturing EEG variations across time, frequency, and spatial channels, attention mechanisms improve feature extraction, representation learning, and model robustness. These methods can be broadly categorized into traditional attention mechanisms, which typically integrate with convolutional and recurrent networks, and Transformer-based multi-head self-attention, which excels in capturing long-range dependencies. Beyond single-modality analysis, attention mechanisms also enhance multimodal EEG applications, facilitating effective fusion between EEG and other physiological or sensory data. Finally, we discuss existing challenges and emerging trends in attention-based EEG modeling, highlighting future directions for advancing BCI technology. This review aims to provide valuable insights for researchers seeking to leverage attention mechanisms for improved EEG interpretation and application. 

**Abstract (ZH)**: 深度学习的快速进步使注意机制在脑电图（EEG）信号分析中变得不可或缺，显著增强了脑-计算机接口（BCI）的应用。本文对传统的和Transformer-based注意机制、其嵌入策略及其在基于EEG的BCI中的应用进行了全面综述，特别强调了多模态数据融合。通过捕捉时间、频率和空间通道中的EEG变化，注意机制提升了特征提取、表示学习和模型稳健性。这些方法可以分为传统的注意机制，通常与卷积和递归网络集成，以及擅长捕捉长期依赖关系的Transformer-based多头自注意力。除了单模态分析，注意机制还增强了多模态EEG应用，促进了EEG与其他生理或感觉数据的有效融合。最后，我们讨论了基于注意机制的EEG建模中存在的挑战和新兴趋势，强调了推进BCI技术的发展方向。本文旨在为研究人员提供有价值的见解，帮助他们利用注意机制改进EEG解释和应用。 

---
# Multiview graph dual-attention deep learning and contrastive learning for multi-criteria recommender systems 

**Title (ZH)**: 多视图图双注意力深度学习与对比学习在多准则推荐系统中的应用 

**Authors**: Saman Forouzandeh, Pavel N. Krivitsky, Rohitash Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2502.19271)  

**Abstract**: Recommender systems leveraging deep learning models have been crucial for assisting users in selecting items aligned with their preferences and interests. However, a significant challenge persists in single-criteria recommender systems, which often overlook the diverse attributes of items that have been addressed by Multi-Criteria Recommender Systems (MCRS). Shared embedding vector for multi-criteria item ratings but have struggled to capture the nuanced relationships between users and items based on specific criteria. In this study, we present a novel representation for Multi-Criteria Recommender Systems (MCRS) based on a multi-edge bipartite graph, where each edge represents one criterion rating of items by users, and Multiview Dual Graph Attention Networks (MDGAT). Employing MDGAT is beneficial and important for adequately considering all relations between users and items, given the presence of both local (criterion-based) and global (multi-criteria) relations. Additionally, we define anchor points in each view based on similarity and employ local and global contrastive learning to distinguish between positive and negative samples across each view and the entire graph. We evaluate our method on two real-world datasets and assess its performance based on item rating predictions. The results demonstrate that our method achieves higher accuracy compared to the baseline method for predicting item ratings on the same datasets. MDGAT effectively capture the local and global impact of neighbours and the similarity between nodes. 

**Abstract (ZH)**: 基于多边双部图和多视图双图注意力网络的多准则推荐系统新表示方法 

---
# Drop-Upcycling: Training Sparse Mixture of Experts with Partial Re-initialization 

**Title (ZH)**: Drop-Upcycling: 以部分重新初始化训练稀疏专家混合模型 

**Authors**: Taishi Nakamura, Takuya Akiba, Kazuki Fujii, Yusuke Oda, Rio Yokota, Jun Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2502.19261)  

**Abstract**: The Mixture of Experts (MoE) architecture reduces the training and inference cost significantly compared to a dense model of equivalent capacity. Upcycling is an approach that initializes and trains an MoE model using a pre-trained dense model. While upcycling leads to initial performance gains, the training progresses slower than when trained from scratch, leading to suboptimal performance in the long term. We propose Drop-Upcycling - a method that effectively addresses this problem. Drop-Upcycling combines two seemingly contradictory approaches: utilizing the knowledge of pre-trained dense models while statistically re-initializing some parts of the weights. This approach strategically promotes expert specialization, significantly enhancing the MoE model's efficiency in knowledge acquisition. Extensive large-scale experiments demonstrate that Drop-Upcycling significantly outperforms previous MoE construction methods in the long term, specifically when training on hundreds of billions of tokens or more. As a result, our MoE model with 5.9B active parameters achieves comparable performance to a 13B dense model in the same model family, while requiring approximately 1/4 of the training FLOPs. All experimental resources, including source code, training data, model checkpoints and logs, are publicly available to promote reproducibility and future research on MoE. 

**Abstract (ZH)**: Drop-Upcycling: 一种有效解决MoE模型初始化问题的方法 

---
# EMT: A Visual Multi-Task Benchmark Dataset for Autonomous Driving in the Arab Gulf Region 

**Title (ZH)**: 阿拉伯湾地区自动驾驶的视觉多任务基准数据集：EMT 

**Authors**: Nadya Abdel Madjid, Murad Mebrahtu, Abdelmoamen Nasser, Bilal Hassan, Naoufel Werghi, Jorge Dias, Majid Khonji  

**Link**: [PDF](https://arxiv.org/pdf/2502.19260)  

**Abstract**: This paper introduces the Emirates Multi-Task (EMT) dataset - the first publicly available dataset for autonomous driving collected in the Arab Gulf region. The EMT dataset captures the unique road topology, high traffic congestion, and distinctive characteristics of the Gulf region, including variations in pedestrian clothing and weather conditions. It contains over 30,000 frames from a dash-camera perspective, along with 570,000 annotated bounding boxes, covering approximately 150 kilometers of driving routes. The EMT dataset supports three primary tasks: tracking, trajectory forecasting and intention prediction. Each benchmark dataset is complemented with corresponding evaluations: (1) multi-agent tracking experiments, focusing on multi-class scenarios and occlusion handling; (2) trajectory forecasting evaluation using deep sequential and interaction-aware models; and (3) intention benchmark experiments conducted for predicting agents intentions from observed trajectories. The dataset is publicly available at this https URL, and pre-processing scripts along with evaluation models can be accessed at this https URL. 

**Abstract (ZH)**: 这篇论文介绍了阿联酋多任务（EMT）数据集——阿拉伯海湾地区首个公开的自主驾驶数据集。EMT数据集捕捉到了独特的道路拓扑结构、高交通拥堵情况以及海湾地区的特色，包括行人服饰和天气条件的差异。数据集包含超过30,000帧前视摄像头视角的数据，以及570,000个标注的边界框，覆盖约150公里的驾驶路线。EMT数据集支持三项主要任务：跟踪、轨迹预测和意图预测。每个基准数据集都配备了相应的评估方法：（1）多代理跟踪实验，专注于多类别场景和遮挡处理；（2）使用深度序列和交互感知模型的轨迹预测评估；（3）意图基准实验，用于从观察轨迹预测代理的意图。数据集可在以下网址公开获取，预处理脚本及评估模型可在以下网址访问。 

---
# Poster: Long PHP webshell files detection based on sliding window attention 

**Title (ZH)**: Poster: 基于滑动窗口注意力的长PHP.WebShell文件检测 

**Authors**: Zhiqiang Wang, Haoyu Wang, Lu Hao  

**Link**: [PDF](https://arxiv.org/pdf/2502.19257)  

**Abstract**: Webshell is a type of backdoor, and web applications are widely exposed to webshell injection attacks. Therefore, it is important to study webshell detection techniques. In this study, we propose a webshell detection method. We first convert PHP source code to opcodes and then extract Opcode Double-Tuples (ODTs). Next, we combine CodeBert and FastText models for feature representation and classification. To address the challenge that deep learning methods have difficulty detecting long webshell files, we introduce a sliding window attention mechanism. This approach effectively captures malicious behavior within long files. Experimental results show that our method reaches high accuracy in webshell detection, solving the problem of traditional methods that struggle to address new webshell variants and anti-detection techniques. 

**Abstract (ZH)**: 基于opcode双元组和滑动窗口注意力机制的Webshell检测方法 

---
# Can RLHF be More Efficient with Imperfect Reward Models? A Policy Coverage Perspective 

**Title (ZH)**: 不完善的奖励模型下，RLHF能否更加高效？一种策略覆盖视角 

**Authors**: Jiawei Huang, Bingcong Li, Christoph Dann, Niao He  

**Link**: [PDF](https://arxiv.org/pdf/2502.19255)  

**Abstract**: Sample efficiency is critical for online Reinforcement Learning from Human Feedback (RLHF). While existing works investigate sample-efficient online exploration strategies, the potential of utilizing misspecified yet relevant reward models to accelerate learning remains underexplored. This paper studies how to transfer knowledge from those imperfect reward models in online RLHF. We start by identifying a novel property of the KL-regularized RLHF objective: \emph{a policy's ability to cover the optimal policy is captured by its sub-optimality}. Building on this insight, we propose a theoretical transfer learning algorithm with provable benefits compared to standard online learning. Our approach achieves low regret in the early stage by quickly adapting to the best available source reward models without prior knowledge of their quality, and over time, it attains an $\tilde{O}(\sqrt{T})$ regret bound \emph{independent} of structural complexity measures. Inspired by our theoretical findings, we develop an empirical algorithm with improved computational efficiency, and demonstrate its effectiveness empirically in summarization tasks. 

**Abstract (ZH)**: 基于人类反馈的在线强化学习（RLHF）中，样本效率至关重要。尽管现有工作研究了样本高效的在线探索策略，但利用不精确但相关的奖励模型来加速学习的潜力尚待探索。本文研究如何在在线RLHF中从这些不完美的奖励模型中转移知识。我们首先揭示了KL正则化RLHF目标的一个新特性：**政策覆盖最优政策的能力与其次优性相关**。基于此洞察，我们提出了一种具有可证明优势的理论迁移学习算法。我们的方法在早期通过迅速适应最可用的来源奖励模型而获得低遗憾值，而无需事先了解这些模型的质量，并随时间推移，其遗憾值上界独立于结构复杂性度量，为$\tilde{O}(\sqrt{T})$。受到理论发现的启发，我们开发了一种更具计算效率的实证算法，并通过摘要任务中的实证结果证明了其有效性。 

---
# GraphBridge: Towards Arbitrary Transfer Learning in GNNs 

**Title (ZH)**: GraphBridge: 向图神经网络中任意迁移学习的目标 

**Authors**: Li Ju, Xingyi Yang, Qi Li, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.19252)  

**Abstract**: Graph neural networks (GNNs) are conventionally trained on a per-domain, per-task basis. It creates a significant barrier in transferring the acquired knowledge to different, heterogeneous data setups. This paper introduces GraphBridge, a novel framework to enable knowledge transfer across disparate tasks and domains in GNNs, circumventing the need for modifications to task configurations or graph structures. Specifically, GraphBridge allows for the augmentation of any pre-trained GNN with prediction heads and a bridging network that connects the input to the output layer. This architecture not only preserves the intrinsic knowledge of the original model but also supports outputs of arbitrary dimensions. To mitigate the negative transfer problem, GraphBridg merges the source model with a concurrently trained model, thereby reducing the source bias when applied to the target domain. Our method is thoroughly evaluated across diverse transfer learning scenarios, including Graph2Graph, Node2Node, Graph2Node, and graph2point-cloud. Empirical validation, conducted over 16 datasets representative of these scenarios, confirms the framework's capacity for task- and domain-agnostic transfer learning within graph-like data, marking a significant advancement in the field of GNNs. 

**Abstract (ZH)**: GraphBridge：一种在图神经网络中实现跨任务和跨域知识迁移的新型框架 

---
# Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases 

**Title (ZH)**: 介于电路与乔姆斯基之间：在形式语言上进行预训练赋予了语言偏见 

**Authors**: Michael Y. Hu, Jackson Petty, Chuan Shi, William Merrill, Tal Linzen  

**Link**: [PDF](https://arxiv.org/pdf/2502.19249)  

**Abstract**: Pretraining language models on formal languages can improve their acquisition of natural language, but it is unclear which features of the formal language impart an inductive bias that leads to effective transfer. Drawing on insights from linguistics and complexity theory, we hypothesize that effective transfer occurs when the formal language both captures dependency structures in natural language and remains within the computational limitations of the model architecture. Focusing on transformers, we find that formal languages with both these properties enable language models to achieve lower loss on natural language and better linguistic generalization compared to other languages. In fact, pre-pretraining, or training on formal-then-natural language, reduces loss more efficiently than the same amount of natural language. For a 1B-parameter language model trained on roughly 1.6B tokens of natural language, pre-pretraining achieves the same loss and better linguistic generalization with a 33% smaller token budget. We also give mechanistic evidence of cross-task transfer from formal to natural language: attention heads acquired during formal language pretraining remain crucial for the model's performance on syntactic evaluations. 

**Abstract (ZH)**: 在正式语言上预训练语言模型可以提高其对自然语言的掌握，但不清楚是正式语言的哪些特性赋予了有效的迁移偏见。从语言学和复杂性理论中汲取灵感，我们假设当正式语言同时捕捉自然语言中的依赖结构并且保持在模型架构的计算限制之内时，有效迁移才会发生。聚焦于变压器模型，我们发现同时具备这两种特性的正式语言可以使语言模型在自然语言上获得更低的损失并展现出更好的语言通用性，与其他语言相比更是如此。事实上，先在正式语言上预训练再在自然语言上训练可以比同等量的自然语言训练更高效地降低损失。对于一个参数量为1B的語言模型，在大约1.6B令牌的自然语言上训练，先进行正式语言预训练再进行自然语言训练可以在减小33%令牌预算的情况下达到相同的损失并展现出更好的语言通用性。我们还提供了从正式语言到自然语言的跨任务迁移的机制性证据：在正式语言预训练过程中获得的注意力头对于模型在句法评估中的性能仍然至关重要。 

---
# AI-Powered Bayesian Inference 

**Title (ZH)**: AI驱动的贝叶斯推断 

**Authors**: Veronika Ročková, Sean O'Hagan  

**Link**: [PDF](https://arxiv.org/pdf/2502.19231)  

**Abstract**: The advent of Generative Artificial Intelligence (GAI) has heralded an inflection point that changed how society thinks about knowledge acquisition. While GAI cannot be fully trusted for decision-making, it may still provide valuable information that can be integrated into a decision pipeline. Rather than seeing the lack of certitude and inherent randomness of GAI as a problem, we view it as an opportunity. Indeed, variable answers to given prompts can be leveraged to construct a prior distribution which reflects assuredness of AI predictions. This prior distribution may be combined with tailored datasets for a fully Bayesian analysis with an AI-driven prior. In this paper, we explore such a possibility within a non-parametric Bayesian framework. The basic idea consists of assigning a Dirichlet process prior distribution on the data-generating distribution with AI generative model as its baseline. Hyper-parameters of the prior can be tuned out-of-sample to assess the informativeness of the AI prior. Posterior simulation is achieved by computing a suitably randomized functional on an augmented data that consists of observed (labeled) data as well as fake data whose labels have been imputed using AI. This strategy can be parallelized and rapidly produces iid samples from the posterior by optimization as opposed to sampling from conditionals. Our method enables (predictive) inference and uncertainty quantification leveraging AI predictions in a coherent probabilistic manner. 

**Abstract (ZH)**: 生成型人工智能（GAI）的兴起标志着一个转折点，改变了社会对知识获取方式的认知。虽然GAI在决策制定上无法完全信赖，但它仍可能提供有价值的信息，这些信息可以集成到决策流程中。我们不应将GAI缺乏确定性和固有的不确定性视为问题，而应视为一种机遇。事实上，对于给定提示的不同答案可以被利用来构建反映AI预测置信度的先验分布。这种先验分布可以与定制化数据集结合，进行由AI驱动的先验的贝叶斯分析。在本文中，我们探讨了在这种非参数贝叶斯框架下进行此类可能性的方法。基本思想是使用AI生成模型作为基础，在数据生成分布上赋予狄利克雷过程先验分布。先验的超参数可以通过离样本外调整来评估AI先验的信息量。通过在扩大数据集（包括观测到的标记数据以及使用AI填补标签的假数据）上计算适当随机化的函数来实现后验模拟。该策略可以并行化，快速通过优化从后验分布产生独立同分布样例，而非从条件分布采样。本方法使得以一致概率方式利用AI预测进行预测性推断和不确定性量化成为可能。 

---
# Enhancing the Scalability and Applicability of Kohn-Sham Hamiltonians for Molecular Systems 

**Title (ZH)**: 增强Kohn-Sham哈密顿量在分子系统中的可扩展性和适用性 

**Authors**: Yunyang Li, Zaishuo Xia, Lin Huang, Xinran Wei, Han Yang, Sam Harshe, Zun Wang, Chang Liu, Jia Zhang, Bin Shao, Mark B. Gerstein  

**Link**: [PDF](https://arxiv.org/pdf/2502.19227)  

**Abstract**: Density Functional Theory (DFT) is a pivotal method within quantum chemistry and materials science, with its core involving the construction and solution of the Kohn-Sham Hamiltonian. Despite its importance, the application of DFT is frequently limited by the substantial computational resources required to construct the Kohn-Sham Hamiltonian. In response to these limitations, current research has employed deep-learning models to efficiently predict molecular and solid Hamiltonians, with roto-translational symmetries encoded in their neural networks. However, the scalability of prior models may be problematic when applied to large molecules, resulting in non-physical predictions of ground-state properties. In this study, we generate a substantially larger training set (PubChemQH) than used previously and use it to create a scalable model for DFT calculations with physical accuracy. For our model, we introduce a loss function derived from physical principles, which we call Wavefunction Alignment Loss (WALoss). WALoss involves performing a basis change on the predicted Hamiltonian to align it with the observed one; thus, the resulting differences can serve as a surrogate for orbital energy differences, allowing models to make better predictions for molecular orbitals and total energies than previously possible. WALoss also substantially accelerates self-consistent-field (SCF) DFT calculations. Here, we show it achieves a reduction in total energy prediction error by a factor of 1347 and an SCF calculation speed-up by a factor of 18%. These substantial improvements set new benchmarks for achieving accurate and applicable predictions in larger molecular systems. 

**Abstract (ZH)**: 基于波函数对齐损失的密度泛函理论可扩展模型研究 

---
# A Lightweight and Extensible Cell Segmentation and Classification Model for Whole Slide Images 

**Title (ZH)**: 一种轻量级且可扩展的Whole Slide Images细胞分割与分类模型 

**Authors**: Nikita Shvetsov, Thomas K. Kilvaer, Masoud Tafavvoghi, Anders Sildnes, Kajsa Møllersen, Lill-Tove Rasmussen Busund, Lars Ailo Bongo  

**Link**: [PDF](https://arxiv.org/pdf/2502.19217)  

**Abstract**: Developing clinically useful cell-level analysis tools in digital pathology remains challenging due to limitations in dataset granularity, inconsistent annotations, high computational demands, and difficulties integrating new technologies into workflows. To address these issues, we propose a solution that enhances data quality, model performance, and usability by creating a lightweight, extensible cell segmentation and classification model. First, we update data labels through cross-relabeling to refine annotations of PanNuke and MoNuSAC, producing a unified dataset with seven distinct cell types. Second, we leverage the H-Optimus foundation model as a fixed encoder to improve feature representation for simultaneous segmentation and classification tasks. Third, to address foundation models' computational demands, we distill knowledge to reduce model size and complexity while maintaining comparable performance. Finally, we integrate the distilled model into QuPath, a widely used open-source digital pathology platform. Results demonstrate improved segmentation and classification performance using the H-Optimus-based model compared to a CNN-based model. Specifically, average $R^2$ improved from 0.575 to 0.871, and average $PQ$ score improved from 0.450 to 0.492, indicating better alignment with actual cell counts and enhanced segmentation quality. The distilled model maintains comparable performance while reducing parameter count by a factor of 48. By reducing computational complexity and integrating into workflows, this approach may significantly impact diagnostics, reduce pathologist workload, and improve outcomes. Although the method shows promise, extensive validation is necessary prior to clinical deployment. 

**Abstract (ZH)**: 开发用于数字病理学的临床有用细胞水平分析工具仍具有挑战性，由于数据集粒度不足、标注不一致、高计算需求以及将新技术集成到工作流程中困难等原因。为解决这些问题，我们提出了一种解决方案，通过创建一个轻量级且可扩展的细胞分割和分类模型来提升数据质量、模型性能和易用性。首先，通过交叉重新标注来更新数据标签，从而细化PANUKE和MoNuSAC的标注，生成一个包含七种不同细胞类型的统一数据集。其次，利用H-Optimus基础模型作为固定编码器，以提高同时进行分割和分类任务的特征表示。第三，为解决基础模型的高计算需求，通过知识蒸馏来减小模型大小和复杂性，同时保持相似的性能。最后，将蒸馏后的模型集成到QuPath这一广泛使用的开源数字病理学平台中。结果表明，基于H-Optimus模型的分割和分类性能优于基于CNN的模型，特别是平均$R^2$从0.575提升到0.871，平均$PQ$分数从0.450提升到0.492，显示出更好的细胞计数对齐性和增强的分割质量。蒸馏后的模型保持相似性能的同时，参数量减少了48倍。通过降低计算复杂度并集成到工作流程中，这种方法可能对诊断产生重大影响，减少病理学家的工作负担，并改善结果。尽管该方法显示出潜力，但在临床部署之前仍需进行广泛的验证。 

---
# FaithUn: Toward Faithful Forgetting in Language Models by Investigating the Interconnectedness of Knowledge 

**Title (ZH)**: FaithUn: 探究知识互联性以实现语言模型的忠实遗忘 

**Authors**: Nakyeong Yang, Minsung Kim, Seunghyun Yoon, Joongbo Shin, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2502.19207)  

**Abstract**: Various studies have attempted to remove sensitive or private knowledge from a language model to prevent its unauthorized exposure. However, prior studies have overlooked the complex and interconnected nature of knowledge, where related knowledge must be carefully examined. Specifically, they have failed to evaluate whether an unlearning method faithfully erases interconnected knowledge that should be removed, retaining knowledge that appears relevant but exists in a completely different context. To resolve this problem, we first define a new concept called superficial unlearning, which refers to the phenomenon where an unlearning method either fails to erase the interconnected knowledge it should remove or unintentionally erases irrelevant knowledge. Based on the definition, we introduce a new benchmark, FaithUn, to analyze and evaluate the faithfulness of unlearning in real-world knowledge QA settings. Furthermore, we propose a novel unlearning method, KLUE, which updates only knowledge-related neurons to achieve faithful unlearning. KLUE identifies knowledge neurons using an explainability method and updates only those neurons using selected unforgotten samples. Experimental results demonstrate that widely-used unlearning methods fail to ensure faithful unlearning, while our method shows significant effectiveness in real-world QA unlearning. 

**Abstract (ZH)**: 各种研究尝试通过从语言模型中移除敏感或私人信息来防止其未经授权的暴露。然而，先前的研究忽视了知识的复杂性和相互关联性，其中相关的知识必须仔细审查。具体而言，它们未能评估去学习方法是否忠实删除了应被移除的相关知识，而保留了看似相关但实际上存在于完全不同上下文中的知识。为了解决这一问题，我们首先引入了一个新的概念——表面去学习，指的是去学习方法要么未能删除应被移除的相关知识，要么无意中删除了无关知识。基于这一定义，我们引入了一个新的基准测试——FaithUn，用于分析和评估实际知识问答环境中去学习的忠实性。此外，我们提出了一种新型去学习方法——KLUE，该方法仅更新与知识相关的人工神经元以实现忠实去学习。KLUE 使用可解释性方法识别知识神经元，并仅使用选定的未遗忘样本更新这些神经元。实验结果表明，广泛使用的方法无法确保忠实去学习，而我们的方法在实际问答去学习中显示出显著的效果。 

---
# EGR-Net: A Novel Embedding Gramian Representation CNN for Intelligent Fault Diagnosis 

**Title (ZH)**: EGR-Net: 一种新的嵌入Gramian表示卷积神经网络用于智能故障诊断 

**Authors**: Linshan Jia  

**Link**: [PDF](https://arxiv.org/pdf/2502.19199)  

**Abstract**: Feature extraction is crucial in intelligent fault diagnosis of rotating machinery. It is easier for convolutional neural networks(CNNs) to visually recognize and learn fault features by converting the complicated one-dimensional (1D) vibrational signals into two-dimensional (2D) images with simple textures. However, the existing representation methods for encoding 1D signals as images have two main problems, including complicated computation and low separability. Meanwhile, the existing 2D-CNN fault diagnosis methods taking 2D images as the only inputs still suffer from the inevitable information loss because of the conversion process. Considering the above issues, this paper proposes a new 1D-to-2D conversion method called Embedding Gramian Representation (EGR), which is easy to calculate and shows good separability. In EGR, 1D signals are projected in the embedding space and the intrinsic periodicity of vibrational signals is captured enabling the faulty characteristics contained in raw signals to be uncovered. Second, aiming at the information loss problem of existing CNN models with the single input of converted images, a double-branch EGR-based CNN, called EGR-Net, is proposed to learn faulty features from both raw signal feature maps and their corresponding EGRs. The bridge connection is designed to improve the feature learning interaction between the two branches. Widely used open domain gearbox dataset and bearing dataset are used to verify the effectiveness and efficiency of the proposed methods. EGR-Net is compared with traditional and state-of-the-art approaches, and the results show that the proposed method can deliver enhanced performance. 

**Abstract (ZH)**: 基于嵌入格言表示的1D至2D转换方法在旋转机械智能故障诊断中的应用 

---
# Simulation of Language Evolution under Regulated Social Media Platforms: A Synergistic Approach of Large Language Models and Genetic Algorithms 

**Title (ZH)**: 受规制社交媒体平台上的语言进化模拟：大型语言模型与遗传算法的协同方法 

**Authors**: Jinyu Cai, Yusei Ishimizu, Mingyue Zhang, Munan Li, Jialong Li, Kenji Tei  

**Link**: [PDF](https://arxiv.org/pdf/2502.19193)  

**Abstract**: Social media platforms frequently impose restrictive policies to moderate user content, prompting the emergence of creative evasion language strategies. This paper presents a multi-agent framework based on Large Language Models (LLMs) to simulate the iterative evolution of language strategies under regulatory constraints. In this framework, participant agents, as social media users, continuously evolve their language expression, while supervisory agents emulate platform-level regulation by assessing policy violations. To achieve a more faithful simulation, we employ a dual design of language strategies (constraint and expression) to differentiate conflicting goals and utilize an LLM-driven GA (Genetic Algorithm) for the selection, mutation, and crossover of language strategies. The framework is evaluated using two distinct scenarios: an abstract password game and a realistic simulated illegal pet trade scenario. Experimental results demonstrate that as the number of dialogue rounds increases, both the number of uninterrupted dialogue turns and the accuracy of information transmission improve significantly. Furthermore, a user study with 40 participants validates the real-world relevance of the generated dialogues and strategies. Moreover, ablation studies validate the importance of the GA, emphasizing its contribution to long-term adaptability and improved overall results. 

**Abstract (ZH)**: 基于大型语言模型的多代理框架：监管约束下语言策略的迭代演化模拟 

---
# Provocations from the Humanities for Generative AI Research 

**Title (ZH)**: 人文视角对生成型人工智能研究的启发 

**Authors**: Lauren Klein, Meredith Martin, André Brock, Maria Antoniak, Melanie Walsh, Jessica Marie Johnson, Lauren Tilton, David Mimno  

**Link**: [PDF](https://arxiv.org/pdf/2502.19190)  

**Abstract**: This paper presents a set of provocations for considering the uses, impact, and harms of generative AI from the perspective of humanities researchers. We provide a working definition of humanities research, summarize some of its most salient theories and methods, and apply these theories and methods to the current landscape of AI. Drawing from foundational work in critical data studies, along with relevant humanities scholarship, we elaborate eight claims with broad applicability to current conversations about generative AI: 1) Models make words, but people make meaning; 2) Generative AI requires an expanded definition of culture; 3) Generative AI can never be representative; 4) Bigger models are not always better models; 5) Not all training data is equivalent; 6) Openness is not an easy fix; 7) Limited access to compute enables corporate capture; and 8) AI universalism creates narrow human subjects. We conclude with a discussion of the importance of resisting the extraction of humanities research by computer science and related fields. 

**Abstract (ZH)**: 本文从人文学科研究人员的角度探讨生成式AI的用途、影响及其潜在危害，并提出了一系列挑衅性观点。我们提供了人文学科研究的定义，总结了一些最突出的理论和方法，并将这些理论和方法应用于当前的人工智能 landscape。借鉴批判性数据研究领域的基础工作及相关的人文学科研究成果，我们提出八项具有广泛适用性的论点，涉及当前关于生成式AI的讨论：1）模型生成文字，但人们赋予意义；2）生成式AI需要扩展文化定义；3）生成式AI永远不会具有代表性；4）更大的模型并不总是更好的模型；5）并非所有训练数据都是等价的；6）开放性不是简单的解决方案；7）有限的计算访问权有助于企业垄断；8）AI普世性限定了狭隘的人类主体。最后讨论了人文学科研究被计算机科学及相关领域提取的重要性。 

---
# AutoML for Multi-Class Anomaly Compensation of Sensor Drift 

**Title (ZH)**: 自动机器学习在传感器漂移的多类异常补偿中的应用 

**Authors**: Melanie Schaller, Mathis Kruse, Antonio Ortega, Marius Lindauer, Bodo Rosenhahn  

**Link**: [PDF](https://arxiv.org/pdf/2502.19180)  

**Abstract**: Addressing sensor drift is essential in industrial measurement systems, where precise data output is necessary for maintaining accuracy and reliability in monitoring processes, as it progressively degrades the performance of machine learning models over time. Our findings indicate that the standard cross-validation method used in existing model training overestimates performance by inadequately accounting for drift. This is primarily because typical cross-validation techniques allow data instances to appear in both training and testing sets, thereby distorting the accuracy of the predictive evaluation. As a result, these models are unable to precisely predict future drift effects, compromising their ability to generalize and adapt to evolving data conditions. This paper presents two solutions: (1) a novel sensor drift compensation learning paradigm for validating models, and (2) automated machine learning (AutoML) techniques to enhance classification performance and compensate sensor drift. By employing strategies such as data balancing, meta-learning, automated ensemble learning, hyperparameter optimization, feature selection, and boosting, our AutoML-DC (Drift Compensation) model significantly improves classification performance against sensor drift. AutoML-DC further adapts effectively to varying drift severities. 

**Abstract (ZH)**: 解决传感器漂移对于工业测量系统至关重要，因为精确的数据输出对于维护监测过程中的准确性和可靠性是必要的，而传感器漂移会逐渐降低机器学习模型的性能。我们的研究发现，现有模型训练中常用的交叉验证方法会因为不充分考虑漂移而高估了性能。这主要是因为传统的交叉验证技术允许数据实例同时出现在训练集和测试集中，从而扭曲了预测评估的准确性。因此，这些模型无法精确预测未来的漂移影响，削弱了它们适应不断变化的数据条件的能力。本文提出了两种解决方案：（1）一种新颖的传感器漂移补偿学习范式用于验证模型，（2）自动化机器学习（AutoML）技术以增强分类性能并补偿传感器漂移。通过采用数据平衡、元学习、自动化集成学习、超参数优化、特征选择和提升等策略，我们的AutoML-DC（漂移补偿）模型显著提高了针对传感器漂移的分类性能，并且进一步适应了不同的漂移严重程度。 

---
# MEDDxAgent: A Unified Modular Agent Framework for Explainable Automatic Differential Diagnosis 

**Title (ZH)**: MEDDxAgent: 一种统一的模块化解释性自动鉴别诊断代理框架 

**Authors**: Daniel Rose, Chia-Chien Hung, Marco Lepri, Israa Alqassem, Kiril Gashteovski, Carolin Lawrence  

**Link**: [PDF](https://arxiv.org/pdf/2502.19175)  

**Abstract**: Differential Diagnosis (DDx) is a fundamental yet complex aspect of clinical decision-making, in which physicians iteratively refine a ranked list of possible diseases based on symptoms, antecedents, and medical knowledge. While recent advances in large language models have shown promise in supporting DDx, existing approaches face key limitations, including single-dataset evaluations, isolated optimization of components, unrealistic assumptions about complete patient profiles, and single-attempt diagnosis. We introduce a Modular Explainable DDx Agent (MEDDxAgent) framework designed for interactive DDx, where diagnostic reasoning evolves through iterative learning, rather than assuming a complete patient profile is accessible. MEDDxAgent integrates three modular components: (1) an orchestrator (DDxDriver), (2) a history taking simulator, and (3) two specialized agents for knowledge retrieval and diagnosis strategy. To ensure robust evaluation, we introduce a comprehensive DDx benchmark covering respiratory, skin, and rare diseases. We analyze single-turn diagnostic approaches and demonstrate the importance of iterative refinement when patient profiles are not available at the outset. Our broad evaluation demonstrates that MEDDxAgent achieves over 10% accuracy improvements in interactive DDx across both large and small LLMs, while offering critical explainability into its diagnostic reasoning process. 

**Abstract (ZH)**: 基于模块化可解释的诊断推理代理（MEDDxAgent）框架：交互式诊断推理中的迭代学习 

---
# TestNUC: Enhancing Test-Time Computing Approaches through Neighboring Unlabeled Data Consistency 

**Title (ZH)**: TestNUC: 通过未标记邻近数据一致性提升测试时计算方法 

**Authors**: Henry Peng Zou, Zhengyao Gu, Yue Zhou, Yankai Chen, Weizhi Zhang, Liancheng Fang, Yibo Wang, Yangning Li, Kay Liu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.19163)  

**Abstract**: Test-time computing approaches, which leverage additional computational resources during inference, have been proven effective in enhancing large language model performance. This work introduces a novel, linearly scaling approach, TestNUC, that improves test-time predictions by leveraging the local consistency of neighboring unlabeled data-it classifies an input instance by considering not only the model's prediction on that instance but also on neighboring unlabeled instances. We evaluate TestNUC across eight diverse datasets, spanning intent classification, topic mining, domain discovery, and emotion detection, demonstrating its consistent superiority over baseline methods such as standard prompting and self-consistency. Furthermore, TestNUC can be seamlessly integrated with existing test-time computing approaches, substantially boosting their performance. Our analysis reveals that TestNUC scales effectively with increasing amounts of unlabeled data and performs robustly across different embedding models, making it practical for real-world applications. Our code is available at this https URL. 

**Abstract (ZH)**: Test-Time Computing Approaches that Leverage Additional Computational Resources During Inference Have Been Proven Effective in Enhancing Large Language Model Performance. This Work Introduces a Novel, Linearly Scaling Approach, TestNUC, that Improves Test-Time Predictions by Leveraging the Local Consistency of Neighboring Unlabeled Data. We Evaluate TestNUC Across Eight Diverse Datasets, Spanning Intent Classification, Topic Mining, Domain Discovery, and Emotion Detection, Demonstrating Its Consistent Superiority Over Baseline Methods Such as Standard Prompting and Self-Consistency. Furthermore, TestNUC Can Be Seamlessly Integrated With Existing Test-Time Computing Approaches, Substantially Boosting Their Performance. Our Analysis Reveals That TestNUC Scales Effectively With Increasing Amounts of Unlabeled Data and Performs Robustly Across Different Embedding Models, Making It Practical for Real-World Applications. Our Code Is Available at This https URL. 

---
# Detecting Linguistic Indicators for Stereotype Assessment with Large Language Models 

**Title (ZH)**: 使用大型语言模型检测刻板印象评估的语言指标 

**Authors**: Rebekka Görge, Michael Mock, Héctor Allende-Cid  

**Link**: [PDF](https://arxiv.org/pdf/2502.19160)  

**Abstract**: Social categories and stereotypes are embedded in language and can introduce data bias into Large Language Models (LLMs). Despite safeguards, these biases often persist in model behavior, potentially leading to representational harm in outputs. While sociolinguistic research provides valuable insights into the formation of stereotypes, NLP approaches for stereotype detection rarely draw on this foundation and often lack objectivity, precision, and interpretability. To fill this gap, in this work we propose a new approach that detects and quantifies the linguistic indicators of stereotypes in a sentence. We derive linguistic indicators from the Social Category and Stereotype Communication (SCSC) framework which indicate strong social category formulation and stereotyping in language, and use them to build a categorization scheme. To automate this approach, we instruct different LLMs using in-context learning to apply the approach to a sentence, where the LLM examines the linguistic properties and provides a basis for a fine-grained assessment. Based on an empirical evaluation of the importance of different linguistic indicators, we learn a scoring function that measures the linguistic indicators of a stereotype. Our annotations of stereotyped sentences show that these indicators are present in these sentences and explain the strength of a stereotype. In terms of model performance, our results show that the models generally perform well in detecting and classifying linguistic indicators of category labels used to denote a category, but sometimes struggle to correctly evaluate the associated behaviors and characteristics. Using more few-shot examples within the prompts, significantly improves performance. Model performance increases with size, as Llama-3.3-70B-Instruct and GPT-4 achieve comparable results that surpass those of Mixtral-8x7B-Instruct, GPT-4-mini and Llama-3.1-8B-Instruct. 

**Abstract (ZH)**: 社会类别和刻板印象嵌入语言中，并可能引入大规模语言模型的数据偏差。尽管有保护措施，这些偏差往往仍然存在于模型行为中，可能导致输出表示危害。尽管社会语用学研究为刻板印象的形成提供了有价值的见解，但用于刻板印象检测的NLP方法很少基于这一基础，且往往缺乏客观性、精确性和解释性。为填补这一空白，本文提出了一种新方法，用于检测和量化句子中的刻板印象语言指标。我们从社会类别和刻板印象沟通框架中推导出语言指标，这些指标表明语言中存在强烈的社会类别表述和刻板印象，并据此构建分类方案。为自动化这一方法，我们使用基于上下文学习指令不同的LLM将该方法应用于句子，LLM检查语言属性并提供细粒度评估的基础。根据对不同语言指标重要性的实证评估，我们学习了一种评分函数，用于衡量刻板印象的语言指标。我们的标注结果显示，这些指标存在于这些语句中，解释了刻板印象的强度。在模型性能方面，我们的结果显示，模型通常能够很好地检测和分类标记类别标签所表示类别的语言指标，但在正确评估相关行为和特征方面有时会遇到困难。在提示中使用更多的少样本示例显著提高了性能。随着模型规模的增大，Llama-3.3-70B-Instruct和GPT-4实现了可比较的、超越Mixtral-8x7B-Instruct、GPT-4-mini和Llama-3.1-8B-Instruct的结果。 

---
# When Personalization Meets Reality: A Multi-Faceted Analysis of Personalized Preference Learning 

**Title (ZH)**: 当个性化遭遇现实：个性化偏好学习的多维度分析 

**Authors**: Yijiang River Dong, Tiancheng Hu, Yinhong Liu, Ahmet Üstün, Nigel Collier  

**Link**: [PDF](https://arxiv.org/pdf/2502.19158)  

**Abstract**: While Reinforcement Learning from Human Feedback (RLHF) is widely used to align Large Language Models (LLMs) with human preferences, it typically assumes homogeneous preferences across users, overlooking diverse human values and minority viewpoints. Although personalized preference learning addresses this by tailoring separate preferences for individual users, the field lacks standardized methods to assess its effectiveness. We present a multi-faceted evaluation framework that measures not only performance but also fairness, unintended effects, and adaptability across varying levels of preference divergence. Through extensive experiments comparing eight personalization methods across three preference datasets, we demonstrate that performance differences between methods could reach 36% when users strongly disagree, and personalization can introduce up to 20% safety misalignment. These findings highlight the critical need for holistic evaluation approaches to advance the development of more effective and inclusive preference learning systems. 

**Abstract (ZH)**: 基于人类反馈的强化学习（RLHF）在对齐大型语言模型（LLMs）与人类偏好方面广泛应用，但通常假设用户偏好 homogeneous，忽视了多样化的人类价值观和少数观点。尽管个性化偏好学习通过为每位用户量身定制不同的偏好来解决这一问题，但该领域缺乏评估其效果的标准方法。我们提出了一种多维度评估框架，不仅衡量性能，还评估公平性、意外效应和在不同偏好分歧水平下的适应性。通过在三个偏好数据集中比较八种个性化方法的广泛实验，我们发现当用户意见分歧强烈时，方法之间的性能差异可达到 36%，个性化可能导致高达 20% 的安全对齐偏差。这些发现强调了进行全面评估方法的迫切需求，以促进更有效和包容性的偏好学习系统的开发。 

---
# Voting or Consensus? Decision-Making in Multi-Agent Debate 

**Title (ZH)**: 投票还是共识？多智能体争论中的决策制定 

**Authors**: Lars Benedikt Kaesberg, Jonas Becker, Jan Philip Wahle, Terry Ruas, Bela Gipp  

**Link**: [PDF](https://arxiv.org/pdf/2502.19130)  

**Abstract**: Much of the success of multi-agent debates depends on carefully choosing the right parameters. Among them, the decision-making protocol stands out. Systematic comparison of decision protocols is difficult because studies alter multiple discussion parameters beyond the protocol. So far, it has been largely unknown how decision-making addresses the challenges of different tasks. This work systematically evaluates the impact of seven decision protocols (e.g., majority voting, unanimity consensus). We change only one variable at a time (i.e., decision protocol) to analyze how different methods affect the collaboration between agents and test different protocols on knowledge (MMLU, MMLU-Pro, GPQA) and reasoning datasets (StrategyQA, MuSR, SQuAD 2.0). Our results show that voting protocols improve performance by 13.2% in reasoning tasks and consensus protocols by 2.8% in knowledge tasks over the other decision protocol. Increasing the number of agents improves performance, while more discussion rounds before voting reduces it. To improve decision-making by increasing answer diversity, we propose two new methods, All-Agents Drafting (AAD) and Collective Improvement (CI). Our methods improve task performance by up to 3.3% with AAD and up to 7.4% with CI. This work demonstrates the importance of decision-making in multi-agent debates beyond scaling. 

**Abstract (ZH)**: 多代理辩论中成功的关键在于精心选择合适的参数，其中决策协议尤为突出。由于研究中会改变超出协议之外的多个讨论参数，系统性比较决策协议变得困难。目前尚不清楚决策如何应对不同类型任务的挑战。本研究系统评估了七种决策协议（如多数投票、一致共识）的影响。我们一次只改变一个变量（即决策协议）来分析不同方法如何影响代理之间的协作，并在知识（MMLU、MMLU-Pro、GPQA）和推理数据集（StrategyQA、MuSR、SQuAD 2.0）上测试不同的协议。结果显示，在推理任务中，投票协议提高了13.2%的性能，在知识任务中，一致协议提高了2.8%的性能。增加代理数量可提高性能，而在投票前增加讨论轮次会降低性能。为了通过增加答案多样性来改进决策，我们提出了两种新方法，全体代理起草（AAD）和集体改进（CI）。我们的方法在AAD上的任务性能最多可提高3.3%，在CI上的任务性能最多可提高7.4%。本研究证明了在多代理辩论中决策的重要性远超规模扩展。 

---
# From Traditional to Deep Learning Approaches in Whole Slide Image Registration: A Methodological Review 

**Title (ZH)**: 从传统方法到深度学习在全视野组织图像配准中的应用：一种方法学综述 

**Authors**: Behnaz Elhaminia, Abdullah Alsalemi, Esha Nasir, Mostafa Jahanifar, Ruqayya Awan, Lawrence S. Young, Nasir M. Rajpoot, Fayyaz Minhas, Shan E Ahmed Raza  

**Link**: [PDF](https://arxiv.org/pdf/2502.19123)  

**Abstract**: Whole slide image (WSI) registration is an essential task for analysing the tumour microenvironment (TME) in histopathology. It involves the alignment of spatial information between WSIs of the same section or serial sections of a tissue sample. The tissue sections are usually stained with single or multiple biomarkers before imaging, and the goal is to identify neighbouring nuclei along the Z-axis for creating a 3D image or identifying subclasses of cells in the TME. This task is considerably more challenging compared to radiology image registration, such as magnetic resonance imaging or computed tomography, due to various factors. These include gigapixel size of images, variations in appearance between differently stained tissues, changes in structure and morphology between non-consecutive sections, and the presence of artefacts, tears, and deformations. Currently, there is a noticeable gap in the literature regarding a review of the current approaches and their limitations, as well as the challenges and opportunities they present. We aim to provide a comprehensive understanding of the available approaches and their application for various purposes. Furthermore, we investigate current deep learning methods used for WSI registration, emphasising their diverse methodologies. We examine the available datasets and explore tools and software employed in the field. Finally, we identify open challenges and potential future trends in this area of research. 

**Abstract (ZH)**: 全视野图像（WSI）对齐在组织病理学分析肿瘤微环境（TME）中的重要性及其挑战 

---
# Chemical knowledge-informed framework for privacy-aware retrosynthesis learning 

**Title (ZH)**: 化学知识驱动的隐私意识 retrosynthesis 学习框架 

**Authors**: Guikun Chen, Xu Zhang, Yi Yang, Wenguan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.19119)  

**Abstract**: Chemical reaction data is a pivotal asset, driving advances in competitive fields such as pharmaceuticals, materials science, and industrial chemistry. Its proprietary nature renders it sensitive, as it often includes confidential insights and competitive advantages organizations strive to protect. However, in contrast to this need for confidentiality, the current standard training paradigm for machine learning-based retrosynthesis gathers reaction data from multiple sources into one single edge to train prediction models. This paradigm poses considerable privacy risks as it necessitates broad data availability across organizational boundaries and frequent data transmission between entities, potentially exposing proprietary information to unauthorized access or interception during storage and transfer. In the present study, we introduce the chemical knowledge-informed framework (CKIF), a privacy-preserving approach for learning retrosynthesis models. CKIF enables distributed training across multiple chemical organizations without compromising the confidentiality of proprietary reaction data. Instead of gathering raw reaction data, CKIF learns retrosynthesis models through iterative, chemical knowledge-informed aggregation of model parameters. In particular, the chemical properties of predicted reactants are leveraged to quantitatively assess the observable behaviors of individual models, which in turn determines the adaptive weights used for model aggregation. On a variety of reaction datasets, CKIF outperforms several strong baselines by a clear margin (e.g., ~20% performance improvement over FedAvg on USPTO-50K), showing its feasibility and superiority to stimulate further research on privacy-preserving retrosynthesis. 

**Abstract (ZH)**: 基于化学知识的隐私保护拆合反应模型学习框架（CKIF） 

---
# Improving customer service with automatic topic detection in user emails 

**Title (ZH)**: 基于用户电子邮件的主题自动检测改进客户服务 

**Authors**: Bojana Bašaragin, Darija Medvecki, Gorana Gojić, Milena Oparnica, Dragiša Mišković  

**Link**: [PDF](https://arxiv.org/pdf/2502.19115)  

**Abstract**: This study introduces a novel Natural Language Processing pipeline that enhances customer service efficiency at Telekom Srbija, a leading Serbian telecommunications company, through automated email topic detection and labelling. Central to the pipeline is BERTopic, a modular architecture that allows unsupervised topic modelling. After a series of preprocessing and post-processing steps, we assign one of 12 topics and several additional labels to incoming emails, allowing customer service to filter and access them through a custom-made application. The model's performance was evaluated by assessing the speed and correctness of the automatically assigned topics across a test dataset of 100 customer emails. The pipeline shows broad applicability across languages, particularly for those that are low-resourced and morphologically rich. The system now operates in the company's production environment, streamlining customer service operations through automated email classification. 

**Abstract (ZH)**: 本研究介绍了通过自动电子邮件主题检测和标注提高塞尔维亚领先电信公司Telekom Srbija客户服务效率的新颖自然语言处理管道。该管道的核心是允许无监督主题建模的BERTopic模块化架构。经过一系列预处理和后处理步骤后，我们将12个主题中的一个和多个附加标签分配给 incoming 电子邮件，使客户服务能够通过自定义应用程序对其进行筛选和访问。通过评估测试数据集中100封客户电子邮件的自动分配主题的速度和准确性，评估了模型的性能。该管道在多种语言中具有广泛的适用性，特别适用于低资源且形态丰富的语言。该系统现在在公司的生产环境中运行，通过自动电子邮件分类简化了客户服务操作。 

---
# The Shady Light of Art Automation 

**Title (ZH)**: 艺术自动化之光驳论 

**Authors**: Dejan Grba  

**Link**: [PDF](https://arxiv.org/pdf/2502.19107)  

**Abstract**: Generative artificial intelligence (generative AI) has entered the mainstream culture and become a subject of extensive academic investigation. However, the character and background of its impact on art require subtler scrutiny and more nuanced contextualization. This paper summarizes a broader study of the roles that AI's conceptual and ideological substrata play in influencing art notions. The focus is on divergent but coalescing and often questionable ideas, values, and political views that generative AI and other art-related AI technologies propagate from the computer science and AI/tech industry to the contemporary art and culture. The paper maps the main areas of this complex relationship and concisely critiques their key aspects. 

**Abstract (ZH)**: 生成性人工智能（生成性AI）已进入主流文化，并成为广泛学术研究的主题。然而，其对艺术的影响本质和背景需要更加细腻的审视和更为精细的语境化。本文总结了关于AI的概念和意识形态基础如何影响艺术观念的更大范围研究。重点在于生成性AI及其他相关艺术技术从计算机科学和AI/科技行业传播到当代艺术和文化领域的分歧但又相互融合且常具争议的概念、价值观和政治观点。本文勾画了这一复杂关系的主要领域，并简洁地批判其关键方面。 

---
# XSS Adversarial Attacks Based on Deep Reinforcement Learning: A Replication and Extension Study 

**Title (ZH)**: 基于深度强化学习的XSS对抗攻击：复制与扩展研究 

**Authors**: Samuele Pasini, Gianluca Maragliano, Jinhan Kim, Paolo Tonella  

**Link**: [PDF](https://arxiv.org/pdf/2502.19095)  

**Abstract**: Cross-site scripting (XSS) poses a significant threat to web application security. While Deep Learning (DL) has shown remarkable success in detecting XSS attacks, it remains vulnerable to adversarial attacks due to the discontinuous nature of its input-output mapping. These adversarial attacks employ mutation-based strategies for different components of XSS attack vectors, allowing adversarial agents to iteratively select mutations to evade detection. Our work replicates a state-of-the-art XSS adversarial attack, highlighting threats to validity in the reference work and extending it toward a more effective evaluation strategy. Moreover, we introduce an XSS Oracle to mitigate these threats. The experimental results show that our approach achieves an escape rate above 96% when the threats to validity of the replicated technique are addressed. 

**Abstract (ZH)**: 跨站点脚本(XSS)对网络应用程序安全构成显著威胁。尽管深度学习(DL)在检测XSS攻击方面表现出色，但由于其输入输出映射的不连续性，它仍然容易受到对抗性攻击的影响。这些对抗性攻击采用基于变异的策略，针对XSS攻击向量的不同组件实施，使攻击者能够逐步选择变异以逃避检测。我们的工作复制了一种最先进的XSS对抗性攻击，揭示了参考工作中有效性的潜在威胁，并进一步提出了一种更有效的评估策略。此外，我们引入了XSS Oracle以减轻这些威胁。实验结果表明，在解决了复制技术有效性的威胁后，我们的方法达到了超过96%的逃避率。 

---
# InternVQA: Advancing Compressed Video QualityAssessment with Distilling Large Foundation Model 

**Title (ZH)**: InternVQA： advancing compressed video quality assessment with distilling large foundation models 

**Authors**: Fengbin Guan, Zihao Yu, Yiting Lu, Xin Li, Zhibo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.19026)  

**Abstract**: Video quality assessment tasks rely heavily on the rich features required for video understanding, such as semantic information, texture, and temporal motion. The existing video foundational model, InternVideo2, has demonstrated strong potential in video understanding tasks due to its large parameter size and large-scale multimodal data pertaining. Building on this, we explored the transferability of InternVideo2 to video quality assessment under compression scenarios. To design a lightweight model suitable for this task, we proposed a distillation method to equip the smaller model with rich compression quality priors. Additionally, we examined the performance of different backbones during the distillation process. The results showed that, compared to other methods, our lightweight model distilled from InternVideo2 achieved excellent performance in compression video quality assessment. 

**Abstract (ZH)**: 基于压缩场景的视频质量评估中InternVideo2轻量化模型的研究与性能分析 

---
# Ground-level Viewpoint Vision-and-Language Navigation in Continuous Environments 

**Title (ZH)**: 连续环境中基于视点的视觉-语言导航 

**Authors**: Zerui Li, Gengze Zhou, Haodong Hong, Yanyan Shao, Wenqi Lyu, Yanyuan Qiao, Qi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.19024)  

**Abstract**: Vision-and-Language Navigation (VLN) empowers agents to associate time-sequenced visual observations with corresponding instructions to make sequential decisions. However, generalization remains a persistent challenge, particularly when dealing with visually diverse scenes or transitioning from simulated environments to real-world deployment. In this paper, we address the mismatch between human-centric instructions and quadruped robots with a low-height field of view, proposing a Ground-level Viewpoint Navigation (GVNav) approach to mitigate this issue. This work represents the first attempt to highlight the generalization gap in VLN across varying heights of visual observation in realistic robot deployments. Our approach leverages weighted historical observations as enriched spatiotemporal contexts for instruction following, effectively managing feature collisions within cells by assigning appropriate weights to identical features across different viewpoints. This enables low-height robots to overcome challenges such as visual obstructions and perceptual mismatches. Additionally, we transfer the connectivity graph from the HM3D and Gibson datasets as an extra resource to enhance spatial priors and a more comprehensive representation of real-world scenarios, leading to improved performance and generalizability of the waypoint predictor in real-world environments. Extensive experiments demonstrate that our Ground-level Viewpoint Navigation (GVnav) approach significantly improves performance in both simulated environments and real-world deployments with quadruped robots. 

**Abstract (ZH)**: 基于地面视角导航（GVNav）的视觉-语言导航（VLN） 

---
# Robust Over-the-Air Computation with Type-Based Multiple Access 

**Title (ZH)**: 基于类型分配的稳健无线计算 

**Authors**: Marc Martinez-Gost, Ana Pérez-Neira, Miguel Ángel Lagunas  

**Link**: [PDF](https://arxiv.org/pdf/2502.19014)  

**Abstract**: This paper utilizes the properties of type-based multiple access (TBMA) to investigate its effectiveness as a robust approach for over-the-air computation (AirComp) in the presence of Byzantine attacks, this is, adversarial strategies where malicious nodes intentionally distort their transmissions to corrupt the aggregated result. Unlike classical direct aggregation (DA) AirComp, which aggregates data in the amplitude of the signals and are highly vulnerable to attacks, TBMA distributes data over multiple radio resources, enabling the receiver to construct a histogram representation of the transmitted data. This structure allows the integration of classical robust estimators and supports the computation of diverse functions beyond the arithmetic mean, which is not feasible with DA. Through extensive simulations, we demonstrate that robust TBMA significantly outperforms DA, maintaining high accuracy even under adversarial conditions, and showcases its applicability in federated learning (FEEL) scenarios. Additionally, TBMA reduces channel state information (CSI) requirements, lowers energy consumption, and enhances resiliency by leveraging the diversity of the transmitted data. These results establish TBMA as a scalable and robust solution for AirComp, paving the way for secure and efficient aggregation in next-generation networks. 

**Abstract (ZH)**: 基于类型标注的多访问机制（TBMA）在拜占庭攻击下的空中计算鲁棒性研究 

---
# Distilling Reinforcement Learning Algorithms for In-Context Model-Based Planning 

**Title (ZH)**: 基于上下文的模型导向规划的强化学习算法提炼 

**Authors**: Jaehyeon Son, Soochan Lee, Gunhee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.19009)  

**Abstract**: Recent studies have shown that Transformers can perform in-context reinforcement learning (RL) by imitating existing RL algorithms, enabling sample-efficient adaptation to unseen tasks without parameter updates. However, these models also inherit the suboptimal behaviors of the RL algorithms they imitate. This issue primarily arises due to the gradual update rule employed by those algorithms. Model-based planning offers a promising solution to this limitation by allowing the models to simulate potential outcomes before taking action, providing an additional mechanism to deviate from the suboptimal behavior. Rather than learning a separate dynamics model, we propose Distillation for In-Context Planning (DICP), an in-context model-based RL framework where Transformers simultaneously learn environment dynamics and improve policy in-context. We evaluate DICP across a range of discrete and continuous environments, including Darkroom variants and Meta-World. Our results show that DICP achieves state-of-the-art performance while requiring significantly fewer environment interactions than baselines, which include both model-free counterparts and existing meta-RL methods. 

**Abstract (ZH)**: Recent Studies Show that Transformers Can Perform In-Context Reinforcement Learning by Imitating Existing RL Algorithms, but They Inherit Suboptimal Behaviors Due to the Gradual Update Rule of Those Algorithms. Model-Based Planning Offers a Promising Solution by Allowing Simulated Outcomes Before Taking Action, and We Propose Distillation for In-Context Planning (DICP) as an In-Context Model-Based RL Framework Where Transformers Learn Environment Dynamics and Improve Policy Simultaneously. Evaluations Across Discrete and Continuous Environments Demonstrate That DICP Achieves State-of-the-Art Performance with Significantly Fewer Environment Interactions Compared to Baselines. 

---
# Binary Neural Networks for Large Language Model: A Survey 

**Title (ZH)**: 大规模语言模型中的二值神经网络：一种综述 

**Authors**: Liangdong Liu, Zhitong Zheng, Cong Wang, Tianhuang Su, Zhenyu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.19008)  

**Abstract**: Large language models (LLMs) have wide applications in the field of natural language processing(NLP), such as GPT-4 and Llama. However, with the exponential growth of model parameter sizes, LLMs bring significant resource overheads. Low-bit quantization, as a key technique, reduces memory usage and computational demands by decreasing the bit-width of model parameters, activations, and gradients. Previous quantization methods for LLMs have largely employed Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT). PTQ does not require any retraining of the original model, while QAT involves optimizing precision during training to achieve the best quantization parameters. The BitNet team proposed a radically different approach, where quantization is performed from the start of model training, utilizing low-precision binary weights during the training process. This approach has led to the emergence of many binary quantization techniques for large language models. This paper provides a comprehensive review of these binary quantization techniques. Specifically, we will introduce binary quantization techniques in deep neural networks and further explore their application to LLMs, reviewing their various contributions, implementations, and applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言处理（NLP）领域有着广泛的应用，如GPT-4和Llama。然而，随着模型参数规模的指数级增长，LLMs带来了显著的资源开销。低比特量化作为一种关键技术，通过降低模型参数、激活和梯度的位宽来减少内存使用和计算需求。先前针对LLMs的量化方法主要采用了后训练量化（PTQ）和量化感知训练（QAT）。PTQ不需要重新训练原始模型，而QAT则在训练过程中优化精度以达到最佳的量化参数。BitNet团队提出了一种截然不同的方法，从模型训练之初就开始进行量化，并在训练过程中使用低精度二进制权重。这一方法导致了许多针对大型语言模型的二进制量化技术的出现。本文提供了对这些二进制量化技术的全面综述。特别是在介绍深度神经网络中的二进制量化技术后，我们将进一步探索其在大型语言模型中的应用，综述它们的各种贡献、实现和应用。 

---
# A Multi-Agent DRL-Based Framework for Optimal Resource Allocation and Twin Migration in the Multi-Tier Vehicular Metaverse 

**Title (ZH)**: 基于多Agent强化学习的多层级 vehicular元宇宙资源优化分配与孪生迁移框架 

**Authors**: Nahom Abishu Hayla, A. Mohammed Seid, Aiman Erbad, Tilahun M. Getu, Ala Al-Fuqaha, Mohsen Guizani  

**Link**: [PDF](https://arxiv.org/pdf/2502.19004)  

**Abstract**: Although multi-tier vehicular Metaverse promises to transform vehicles into essential nodes -- within an interconnected digital ecosystem -- using efficient resource allocation and seamless vehicular twin (VT) migration, this can hardly be achieved by the existing techniques operating in a highly dynamic vehicular environment, since they can hardly balance multi-objective optimization problems such as latency reduction, resource utilization, and user experience (UX). To address these challenges, we introduce a novel multi-tier resource allocation and VT migration framework that integrates Graph Convolutional Networks (GCNs), a hierarchical Stackelberg game-based incentive mechanism, and Multi-Agent Deep Reinforcement Learning (MADRL). The GCN-based model captures both spatial and temporal dependencies within the vehicular network; the Stackelberg game-based incentive mechanism fosters cooperation between vehicles and infrastructure; and the MADRL algorithm jointly optimizes resource allocation and VT migration in real time. By modeling this dynamic and multi-tier vehicular Metaverse as a Markov Decision Process (MDP), we develop a MADRL-based algorithm dubbed the Multi-Objective Multi-Agent Deep Deterministic Policy Gradient (MO-MADDPG), which can effectively balances the various conflicting objectives. Extensive simulations validate the effectiveness of this algorithm that is demonstrated to enhance scalability, reliability, and efficiency while considerably improving latency, resource utilization, migration cost, and overall UX by 12.8%, 9.7%, 14.2%, and 16.1%, respectively. 

**Abstract (ZH)**: 虽然多层 vehicular 超宇宙有望通过高效的资源分配和无缝 vehicular 双胞胎（VT）迁移将车辆转变为互联数字生态系统中的关键节点，但现有技术在高度动态的 vehicular 环境中难以实现这一点，因为它们难以平衡延迟减少、资源利用和用户体验等多目标优化问题。为应对这些挑战，我们提出了一种将图卷积网络（GCNs）、层次化的Stackelberg博弈激励机制和多智能体深度强化学习（MADRL）集成的新型多层资源分配和VT迁移框架。基于GCN的模型捕捉vehicle网络中的时空依赖关系；基于Stackelberg博弈的激励机制促进车辆与基础设施之间的合作；MADRL算法则实时联合优化资源分配和VT迁移。通过将这个动态和多层 vehicular 超宇宙建模为马尔可夫决策过程（MDP），我们开发了一种基于MADRL的算法——多目标多智能体深度确定性策略梯度（MO-MADDPG），该算法能够有效地平衡各种冲突目标。广泛仿真实验验证了该算法的有效性，该算法在增强可扩展性、可靠性和效率的同时，显著提升了延迟、资源利用、迁移成本和总体用户体验，分别提高了12.8%、9.7%、14.2%和16.1%。 

---
# The Sharpness Disparity Principle in Transformers for Accelerating Language Model Pre-Training 

**Title (ZH)**: Transformer中sharpness disparity原理在语言模型预训练中的加速应用 

**Authors**: Jinbo Wang, Mingze Wang, Zhanpeng Zhou, Junchi Yan, Weinan E, Lei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.19002)  

**Abstract**: Transformers consist of diverse building blocks, such as embedding layers, normalization layers, self-attention mechanisms, and point-wise feedforward networks. Thus, understanding the differences and interactions among these blocks is important. In this paper, we uncover a clear Sharpness Disparity across these blocks, which emerges early in training and intriguingly persists throughout the training process. Motivated by this finding, we propose Blockwise Learning Rate (LR), a strategy that tailors the LR to each block's sharpness, accelerating large language model (LLM) pre-training. By integrating Blockwise LR into AdamW, we consistently achieve lower terminal loss and nearly $2\times$ speedup compared to vanilla AdamW. We demonstrate this acceleration across GPT-2 and LLaMA, with model sizes ranging from 0.12B to 1.1B and datasets of OpenWebText and MiniPile. Finally, we incorporate Blockwise LR into Adam-mini (Zhang et al., 2024), a recently proposed memory-efficient variant of Adam, achieving a combined $2\times$ speedup and $2\times$ memory saving. These results underscore the potential of exploiting the sharpness disparity to improve LLM training. 

**Abstract (ZH)**: Transformers由嵌入层、规范化层、自注意力机制和点wise前馈网络等多种构建块组成。因此，理解这些构建块之间的差异及其相互作用至关重要。在本文中，我们揭示了这些构建块之间存在清晰的清晰度差异，这种差异在训练早期出现，并在整个训练过程中持续存在。受这一发现的启发，我们提出了一种块级学习率（Blockwise Learning Rate，BLR）策略，该策略根据每个构建块的清晰度调整学习率，加速大规模语言模型（LLM）的预训练。通过将块级学习率集成到AdamW中，我们一致实现了更低的终端损失，并且速度提高了近两倍。我们在GPT-2和LLaMA上展示了这种加速，模型大小从0.12B到1.1B不等，数据集包括OpenWebText和MiniPile。最后，我们将块级学习率集成到Adam-mini（Zhang et al., 2024）这一近期提出的大规模内存高效变体中，实现了两倍的速度和两倍的内存节省。这些结果突显了利用清晰度差异改进LLM训练的潜力。 

---
# PEToolLLM: Towards Personalized Tool Learning in Large Language Models 

**Title (ZH)**: PEToolLLM: 面向大型语言模型中个性化工具学习的研究 

**Authors**: Qiancheng Xu, Yongqi Li, Heming Xia, Fan Liu, Min Yang, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.18980)  

**Abstract**: Tool learning has emerged as a promising direction by extending Large Language Models' (LLMs) capabilities with external tools. Existing tool learning studies primarily focus on the general-purpose tool-use capability, which addresses explicit user requirements in instructions. However, they overlook the importance of personalized tool-use capability, leading to an inability to handle implicit user preferences. To address the limitation, we first formulate the task of personalized tool learning, which integrates user's interaction history towards personalized tool usage. To fill the gap of missing benchmarks, we construct PEToolBench, featuring diverse user preferences reflected in interaction history under three distinct personalized settings, and encompassing a wide range of tool-use scenarios. Moreover, we propose a framework PEToolLLaMA to adapt LLMs to the personalized tool learning task, which is trained through supervised fine-tuning and direct preference optimization. Extensive experiments on PEToolBench demonstrate the superiority of PEToolLLaMA over existing LLMs. 

**Abstract (ZH)**: 个性化的工具学习通过将大型语言模型的能力与外部工具扩展相结合而崭露头角。现有的工具学习研究主要集中在通用工具使用能力上，这针对了指令中的明确用户需求。然而，它们忽略了个性化工具使用能力的重要性，导致无法处理隐含的用户偏好。为解决这一限制，我们首先定义了个性化工具学习任务，该任务将用户的历史交互整合到个性化工具使用中。为填补缺失的基准，我们构建了PEToolBench，它包含在三种不同个性化设置下反映不同用户偏好的交互历史，涵盖了广泛的工具使用场景。此外，我们提出了一种PEToolLLaMA框架，将大规模语言模型适应个性化工具学习任务，该框架通过监督微调和直接偏好优化进行训练。在PEToolBench上的广泛实验表明，PEToolLLaMA优于现有的大规模语言模型。 

---
# Low-Confidence Gold: Refining Low-Confidence Samples for Efficient Instruction Tuning 

**Title (ZH)**: 低置信度金标准：细化低置信度样本以实现高效的指令调优 

**Authors**: Hongyi Cal, ie Li, Wenzhen Dong  

**Link**: [PDF](https://arxiv.org/pdf/2502.18978)  

**Abstract**: The effectiveness of instruction fine-tuning for Large Language Models is fundamentally constrained by the quality and efficiency of training datasets. This work introduces Low-Confidence Gold (LCG), a novel filtering framework that employs centroid-based clustering and confidence-guided selection for identifying valuable instruction pairs. Through a semi-supervised approach using a lightweight classifier trained on representative samples, LCG curates high-quality subsets while preserving data diversity. Experimental evaluation demonstrates that models fine-tuned on LCG-filtered subsets of 6K samples achieve superior performance compared to existing methods, with substantial improvements on MT-bench and consistent gains across comprehensive evaluation metrics. The framework's efficacy while maintaining model performance establishes a promising direction for efficient instruction tuning. 

**Abstract (ZH)**: 大型语言模型指令微调的效果从根本上受到训练数据集质量和效率的限制。本文引入了一种新颖的过滤框架Low-Confidence Gold (LCG)，该框架采用基于质心的聚类和置信度引导的选择方法来识别有价值的指令对。通过轻量级分类器在代表性样本上进行半监督训练，LCG能够在保持数据多样性的同时精炼高质量的数据子集。实验评估表明，使用LCG过滤的6千样本子集微调的模型在MT-bench等现有方法上表现出更优性能，并在全面的评估指标中实现了显著改进。该框架在保持模型性能的同时有效性的验证为高效的指令微调指明了有前景的方向。 

---
# (Mis)Fitting: A Survey of Scaling Laws 

**Title (ZH)**: 偏差 scaling：一项scaling定律综述 

**Authors**: Margaret Li, Sneha Kudugunta, Luke Zettlemoyer  

**Link**: [PDF](https://arxiv.org/pdf/2502.18969)  

**Abstract**: Modern foundation models rely heavily on using scaling laws to guide crucial training decisions. Researchers often extrapolate the optimal architecture and hyper parameters settings from smaller training runs by describing the relationship between, loss, or task performance, and scale. All components of this process vary, from the specific equation being fit, to the training setup, to the optimization method. Each of these factors may affect the fitted law, and therefore, the conclusions of a given study. We discuss discrepancies in the conclusions that several prior works reach, on questions such as the optimal token to parameter ratio. We augment this discussion with our own analysis of the critical impact that changes in specific details may effect in a scaling study, and the resulting altered conclusions. Additionally, we survey over 50 papers that study scaling trends: while 45 of these papers quantify these trends using a power law, most under-report crucial details needed to reproduce their findings. To mitigate this, we we propose a checklist for authors to consider while contributing to scaling law research. 

**Abstract (ZH)**: 现代基础模型在训练决策中高度依赖于放大定律的指导。研究人员经常通过描述损失、任务性能与规模之间的关系来外推最优架构和超参数设置。这一过程中涉及的因素众多，包括具体拟合的方程、训练设置以及优化方法等。每个因素都可能影响拟合出的定律，进而影响研究结论。我们讨论了若干先前工作中得出的不同结论，特别是在最优令牌与参数比方面的问题。我们增加了自己的分析，探讨特定细节变化对放大研究及其结论的影响。此外，我们回顾了超过50篇研究放大趋势的论文：其中45篇使用幂律量化这些趋势，但大多数论文未能充分报告必要的重复其发现的关键细节。为此，我们建议作者在贡献放大定律研究时考虑一个清单。 

---
# DualSpec: Text-to-spatial-audio Generation via Dual-Spectrogram Guided Diffusion Model 

**Title (ZH)**: DualSpec：基于双谱图引导扩散模型的文本到空间化音频生成 

**Authors**: Lei Zhao, Sizhou Chen, Linfeng Feng, Xiao-Lei Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.18952)  

**Abstract**: Text-to-audio (TTA), which generates audio signals from textual descriptions, has received huge attention in recent years. However, recent works focused on text to monaural audio only. As we know, spatial audio provides more immersive auditory experience than monaural audio, e.g. in virtual reality. To address this issue, we propose a text-to-spatial-audio (TTSA) generation framework named this http URL, it first trains variational autoencoders (VAEs) for extracting the latent acoustic representations from sound event audio. Then, given text that describes sound events and event directions, the proposed method uses the encoder of a pretrained large language model to transform the text into text features. Finally, it trains a diffusion model from the latent acoustic representations and text features for the spatial audio generation. In the inference stage, only the text description is needed to generate spatial audio. Particularly, to improve the synthesis quality and azimuth accuracy of the spatial sound events simultaneously, we propose to use two kinds of acoustic features. One is the Mel spectrograms which is good for improving the synthesis quality, and the other is the short-time Fourier transform spectrograms which is good at improving the azimuth accuracy. We provide a pipeline of constructing spatial audio dataset with text prompts, for the training of the VAEs and diffusion model. We also introduce new spatial-aware evaluation metrics to quantify the azimuth errors of the generated spatial audio recordings. Experimental results demonstrate that the proposed method can generate spatial audio with high directional and event consistency. 

**Abstract (ZH)**: 文本到立体声音频生成：http:// développent un框架 

---
# MathTutorBench: A Benchmark for Measuring Open-ended Pedagogical Capabilities of LLM Tutors 

**Title (ZH)**: MathTutorBench: 一个衡量LLM导师开放式教学能力的标准测试 

**Authors**: Jakub Macina, Nico Daheim, Ido Hakimi, Manu Kapur, Iryna Gurevych, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2502.18940)  

**Abstract**: Evaluating the pedagogical capabilities of AI-based tutoring models is critical for making guided progress in the field. Yet, we lack a reliable, easy-to-use, and simple-to-run evaluation that reflects the pedagogical abilities of models. To fill this gap, we present MathTutorBench, an open-source benchmark for holistic tutoring model evaluation. MathTutorBench contains a collection of datasets and metrics that broadly cover tutor abilities as defined by learning sciences research in dialog-based teaching. To score the pedagogical quality of open-ended teacher responses, we train a reward model and show it can discriminate expert from novice teacher responses with high accuracy. We evaluate a wide set of closed- and open-weight models on MathTutorBench and find that subject expertise, indicated by solving ability, does not immediately translate to good teaching. Rather, pedagogy and subject expertise appear to form a trade-off that is navigated by the degree of tutoring specialization of the model. Furthermore, tutoring appears to become more challenging in longer dialogs, where simpler questioning strategies begin to fail. We release the benchmark, code, and leaderboard openly to enable rapid benchmarking of future models. 

**Abstract (ZH)**: 基于AI的教学辅导模型的教育能力评估对于推动该领域的发展至关重要。然而，我们缺乏一种可靠、易于使用且简便的操作的评估方法来反映模型的教育能力。为填补这一空白，我们提出了MathTutorBench，这是一个开源的全方位教学辅导模型评估基准。MathTutorBench包含了一组涵盖学习科学研究中定义的基于对话教学的辅导能力的数据集和评估指标。为了评估开放型教师回答的质量，我们训练了一个奖励模型，并展示了其能够高效地区分专家与新手教师的回答。我们对多种闭合型和开放型权重模型在MathTutorBench上的表现进行了评估，发现解题能力并不能直接转化为良好的教学效果。教育策略与专业知识似乎形成一种权衡，这种权衡由模型的辅导专业化程度来导航。此外，较长的对话似乎使教学更具挑战性，其中简单的提问策略开始失效。我们公开发布基准、代码和排行榜，以便快速评估未来模型的表现。 

---
# JailBench: A Comprehensive Chinese Security Assessment Benchmark for Large Language Models 

**Title (ZH)**: JailBench: 一种全面的中文安全评估基准模型 

**Authors**: Shuyi Liu, Simiao Cui, Haoran Bu, Yuming Shang, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18935)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various applications, highlighting the urgent need for comprehensive safety evaluations. In particular, the enhanced Chinese language proficiency of LLMs, combined with the unique characteristics and complexity of Chinese expressions, has driven the emergence of Chinese-specific benchmarks for safety assessment. However, these benchmarks generally fall short in effectively exposing LLM safety vulnerabilities. To address the gap, we introduce JailBench, the first comprehensive Chinese benchmark for evaluating deep-seated vulnerabilities in LLMs, featuring a refined hierarchical safety taxonomy tailored to the Chinese context. To improve generation efficiency, we employ a novel Automatic Jailbreak Prompt Engineer (AJPE) framework for JailBench construction, which incorporates jailbreak techniques to enhance assessing effectiveness and leverages LLMs to automatically scale up the dataset through context-learning. The proposed JailBench is extensively evaluated over 13 mainstream LLMs and achieves the highest attack success rate against ChatGPT compared to existing Chinese benchmarks, underscoring its efficacy in identifying latent vulnerabilities in LLMs, as well as illustrating the substantial room for improvement in the security and trustworthiness of LLMs within the Chinese context. Our benchmark is publicly available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各种应用中展现了出色的能力，凸显了进行全面安全性评估的紧迫需求。特别地，LLMs增强的中文语言能力，结合中文表达的独特特征和复杂性，推动了专门针对中文的安全评估基准的出现。然而，这些基准在有效揭示LLMs的安全漏洞方面仍存在不足。为填补这一空白，我们引入了JailBench，这是首个全面的中文基准，用于评估LLMs深层次的安全漏洞，具备针对中文语境定制的精细分层安全分类体系。为提高生成效率，我们采用了一种新颖的自动脱牢笼提示工程师（AJPE）框架来构建JailBench，该框架集成了脱牢笼技术以增强评估效果，并利用LLMs通过上下文学习自动扩展数据集。提出的JailBench在13款主流LLMs上进行了广泛评估，并在对抗ChatGPT时取得了最高的攻击成功率，验证了其在识别LLMs潜藏漏洞方面的有效性，并展示了在中文语境下提升LLMs的安全性和可信度的巨大改善空间。我们的基准可从此链接获取： this https URL。 

---
# SLAM in the Dark: Self-Supervised Learning of Pose, Depth and Loop-Closure from Thermal Images 

**Title (ZH)**: 黑暗中SLAM：从热图像中自学姿态、深度和环路闭合 

**Authors**: Yangfan Xu, Qu Hao, Lilian Zhang, Jun Mao, Xiaofeng He, Wenqi Wu, Changhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18932)  

**Abstract**: Visual SLAM is essential for mobile robots, drone navigation, and VR/AR, but traditional RGB camera systems struggle in low-light conditions, driving interest in thermal SLAM, which excels in such environments. However, thermal imaging faces challenges like low contrast, high noise, and limited large-scale annotated datasets, restricting the use of deep learning in outdoor scenarios. We present DarkSLAM, a noval deep learning-based monocular thermal SLAM system designed for large-scale localization and reconstruction in complex lighting this http URL approach incorporates the Efficient Channel Attention (ECA) mechanism in visual odometry and the Selective Kernel Attention (SKA) mechanism in depth estimation to enhance pose accuracy and mitigate thermal depth degradation. Additionally, the system includes thermal depth-based loop closure detection and pose optimization, ensuring robust performance in low-texture thermal scenes. Extensive outdoor experiments demonstrate that DarkSLAM significantly outperforms existing methods like SC-Sfm-Learner and Shin et al., delivering precise localization and 3D dense mapping even in challenging nighttime environments. 

**Abstract (ZH)**: 基于深度学习的暗光SLAM：一种适用于复杂光照条件的大规模定位与重建系统 

---
# BeamVQ: Beam Search with Vector Quantization to Mitigate Data Scarcity in Physical Spatiotemporal Forecasting 

**Title (ZH)**: BeamVQ: 基于向量量化的方法搜索以缓解物理时空预测中的数据稀缺性 

**Authors**: Weiyan Wang, Xingjian Shi, Ruiqi Shu, Yuan Gao, Rui Ray Chen, Kun Wang, Fan Xu, Jinbao Xue, Shuaipeng Li, Yangyu Tao, Di Wang, Hao Wu, Xiaomeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18925)  

**Abstract**: In practice, physical spatiotemporal forecasting can suffer from data scarcity, because collecting large-scale data is non-trivial, especially for extreme events. Hence, we propose \method{}, a novel probabilistic framework to realize iterative self-training with new self-ensemble strategies, achieving better physical consistency and generalization on extreme events. Following any base forecasting model, we can encode its deterministic outputs into a latent space and retrieve multiple codebook entries to generate probabilistic outputs. Then BeamVQ extends the beam search from discrete spaces to the continuous state spaces in this field. We can further employ domain-specific metrics (e.g., Critical Success Index for extreme events) to filter out the top-k candidates and develop the new self-ensemble strategy by combining the high-quality candidates. The self-ensemble can not only improve the inference quality and robustness but also iteratively augment the training datasets during continuous self-training. Consequently, BeamVQ realizes the exploration of rare but critical phenomena beyond the original dataset. Comprehensive experiments on different benchmarks and backbones show that BeamVQ consistently reduces forecasting MSE (up to 39%), enhancing extreme events detection and proving its effectiveness in handling data scarcity. 

**Abstract (ZH)**: 物理时空预测中的数据稀缺性是一个挑战，尤其是在极端事件的情况下。为此，我们提出了一种名为\method{}的新型概率框架，以实现迭代自我训练，并通过新的自我集成策略在极端事件中获得更好的物理一致性和泛化能力。在任何基础预测模型之后，我们可以将其确定性输出编码到潜在空间中，并检索多个代码表条目以生成概率输出。BeamVQ进一步将该领域的 beam 搜索从离散空间扩展到连续状态空间。我们可以通过领域特定的度量（例如极端事件中的关键成功指数）筛选出前k个候选人，并通过结合高质量的候选人来开发新的自我集成策略。自我集成不仅可以提高推理质量和鲁棒性，还可以在持续自我训练过程中扩充训练数据集。因此，BeamVQ实现了超越原始数据集的稀有但关键现象的探索。在不同基准和骨干上的全面实验表明，BeamVQ一致地降低了预测 MSE（最高达39%），提高了极端事件检测能力，并证明了其在处理数据稀缺性方面的有效性。 

---
# END: Early Noise Dropping for Efficient and Effective Context Denoising 

**Title (ZH)**: END: 早期噪声去除以实现高效有效的上下文去噪 

**Authors**: Hongye Jin, Pei Chen, Jingfeng Yang, Zhengyang Wang, Meng Jiang, Yifan Gao, Binxuan Huang, Xinyang Zhang, Zheng Li, Tianyi Liu, Huasheng Li, Bing Yin  

**Link**: [PDF](https://arxiv.org/pdf/2502.18915)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of natural language processing tasks. However, they are often distracted by irrelevant or noisy context in input sequences that degrades output quality. This problem affects both long- and short-context scenarios, such as retrieval-augmented generation, table question-answering, and in-context learning. We reveal that LLMs can implicitly identify whether input sequences contain useful information at early layers, prior to token generation. Leveraging this insight, we introduce Early Noise Dropping (\textsc{END}), a novel approach to mitigate this issue without requiring fine-tuning the LLMs. \textsc{END} segments input sequences into chunks and employs a linear prober on the early layers of LLMs to differentiate between informative and noisy chunks. By discarding noisy chunks early in the process, \textsc{END} preserves critical information, reduces distraction, and lowers computational overhead. Extensive experiments demonstrate that \textsc{END} significantly improves both performance and efficiency across different LLMs on multiple evaluation datasets. Furthermore, by investigating LLMs' implicit understanding to the input with the prober, this work also deepens understanding of how LLMs do reasoning with contexts internally. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种自然语言处理任务中展现了出色的表现。然而，它们经常会受到输入序列中无关或噪声信息的干扰，从而降低输出质量。这一问题影响了长上下文和短上下文场景，例如检索增强生成、表格问答和上下文学习。我们揭示，大型语言模型可以在生成标记之前，早期层就已经隐式地识别输入序列中是否包含有用信息。基于这一洞察，我们提出了早期噪声丢弃（\textsc{END}）这一新颖的方法，无需微调大型语言模型即可缓解这一问题。\textsc{END} 将输入序列分割成块，并在大型语言模型的早期层使用线性探测器来区分信息块和噪声块。通过早期丢弃噪声块，\textsc{END} 保留了关键信息、减少了干扰并降低了计算开销。广泛的实验表明，\textsc{END} 显著提高了不同大型语言模型在多个评估数据集上的性能和效率。此外，通过探究探测器对输入的隐式理解，这项工作进一步加深了对大型语言模型如何在内部进行上下文推理的理解。 

---
# Dynamic Classification: Leveraging Self-Supervised Classification to Enhance Prediction Performance 

**Title (ZH)**: 动态分类：利用自监督分类提高预测性能 

**Authors**: Ziyuan Zhong, Junyang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.18891)  

**Abstract**: In this paper, we propose an innovative dynamic classification algorithm designed to achieve the objective of zero missed detections and minimal false positives. The algorithm partitions the data into N equivalent training subsets and N prediction subsets using a supervised model, followed by independent predictions from N separate predictive models. This enables each predictive model to operate within a smaller data range, thereby improving overall accuracy. Additionally, the algorithm leverages data generated through supervised learning to further refine prediction results, filtering out predictions that do not meet accuracy requirements without the need to introduce additional models. Experimental results demonstrate that, when data partitioning errors are minimal, the dynamic classification algorithm achieves exceptional performance with zero missed detections and minimal false positives, significantly outperforming existing model ensembles. Even in cases where classification errors are larger, the algorithm remains comparable to state of the art models. The key innovations of this study include self-supervised classification learning, the use of small-range subset predictions, and the direct rejection of substandard predictions. While the current algorithm still has room for improvement in terms of automatic parameter tuning and classification model efficiency, it has demonstrated outstanding performance across multiple datasets. Future research will focus on optimizing the classification component to further enhance the algorithm's robustness and adaptability. 

**Abstract (ZH)**: 本文提出了一种创新的动态分类算法，旨在实现零误检和最少的虚假报警目标。该算法使用监督模型将数据划分为N个等价的训练子集和N个预测子集，随后由N个独立的预测模型进行独立预测。这使得每个预测模型可以在较小的数据范围内运行，从而提高整体准确性。此外，该算法利用通过监督学习生成的数据进一步细化预测结果，无需引入额外模型即可筛选出不符合精度要求的预测。实验结果表明，在数据划分误差最小的情况下，动态分类算法达到了无误检和最少虚假报警的优异性能，显著优于现有模型集成。即使在分类误差较大的情况下，该算法仍可与最先进的模型媲美。本文研究的关键创新点包括自我监督分类学习、小范围子集预测以及直接拒绝不合格预测。虽然当前算法在自动参数调整和分类模型效率方面仍有改进空间，但已在多个数据集上展示了出色的性能。未来的研究将集中在优化分类部分，以进一步增强算法的鲁棒性和适应性。 

---
# Clip-TTS: Contrastive Text-content and Mel-spectrogram, A High-Huality Text-to-Speech Method based on Contextual Semantic Understanding 

**Title (ZH)**: Clip-TTS：对比内容文本和梅尔频谱，一种基于上下文语义理解的高质量文本到语音方法 

**Authors**: Tianyun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18889)  

**Abstract**: Traditional text-to-speech (TTS) methods primarily focus on establishing a mapping between phonemes and mel-spectrograms. However, during the phoneme encoding stage, there is often a lack of real mel-spectrogram auxiliary information, which results in the encoding process lacking true semantic understanding. At the same time, traditional TTS systems often struggle to balance the inference speed of the model with the quality of the synthesized speech. Methods that generate high-quality synthesized speech tend to have slower inference speeds, while faster inference methods often sacrifice speech quality. In this paper, I propose Clip-TTS, a TTS method based on the Clip architecture. This method uses the Clip framework to establish a connection between text content and real mel-spectrograms during the text encoding stage, enabling the text encoder to directly learn the true semantics of the global context, thereby ensuring the quality of the synthesized speech. In terms of model architecture, I adopt the basic structure of Transformer, which allows Clip-TTS to achieve fast inference speeds. Experimental results show that on the LJSpeech and Baker datasets, the speech generated by Clip-TTS achieves state-of-the-art MOS scores, and it also performs excellently on multi-emotion this http URL samples are available at: this https URL. 

**Abstract (ZH)**: 传统的文本到语音（TTS）方法主要关注建立音素和梅尔频谱图之间的映射。然而，在音素编码阶段，通常缺乏真实的梅尔频谱图辅助信息，导致编码过程缺乏真正的语义理解。同时，传统的TTS系统往往难以在模型推断速度与合成语音质量之间达到平衡。生成高质量合成语音的方法通常推断速度较慢，而推断速度快的方法往往牺牲语音质量。在本文中，提出了一种基于Clip架构的TTS方法——Clip-TTS。该方法在文本编码阶段利用Clip框架将文本内容与真实的梅尔频谱图建立联系，使文本编码器可以直接学习全局语义的真实含义，从而确保合成语音的质量。在模型结构方面，采用了Transformer的基本结构，使得Clip-TTS能够实现快速的推断速度。实验结果显示，Clip-TTS在LJSpeech和Baker数据集中生成的语音实现了最先进的MOS分数，并且在多情感语音合成方面表现出色。相关样本可在以下链接获取：[链接] [链接]。 

---
# SE(3)-Equivariant Ternary Complex Prediction Towards Target Protein Degradation 

**Title (ZH)**: SE(3)-共变三元复杂预测 towards 目标蛋白降解 

**Authors**: Fanglei Xue, Meihan Zhang, Shuqi Li, Xinyu Gao, James A. Wohlschlegel, Wenbing Huang, Yi Yang, Weixian Deng  

**Link**: [PDF](https://arxiv.org/pdf/2502.18875)  

**Abstract**: Targeted protein degradation (TPD) induced by small molecules has emerged as a rapidly evolving modality in drug discovery, targeting proteins traditionally considered "undruggable". Proteolysis-targeting chimeras (PROTACs) and molecular glue degraders (MGDs) are the primary small molecules that induce TPD. Both types of molecules form a ternary complex linking an E3 ligase with a target protein, a crucial step for drug discovery. While significant advances have been made in binary structure prediction for proteins and small molecules, ternary structure prediction remains challenging due to obscure interaction mechanisms and insufficient training data. Traditional methods relying on manually assigned rules perform poorly and are computationally demanding due to extensive random sampling. In this work, we introduce DeepTernary, a novel deep learning-based approach that directly predicts ternary structures in an end-to-end manner using an encoder-decoder architecture. DeepTernary leverages an SE(3)-equivariant graph neural network (GNN) with both intra-graph and ternary inter-graph attention mechanisms to capture intricate ternary interactions from our collected high-quality training dataset, TernaryDB. The proposed query-based Pocket Points Decoder extracts the 3D structure of the final binding ternary complex from learned ternary embeddings, demonstrating state-of-the-art accuracy and speed in existing PROTAC benchmarks without prior knowledge from known PROTACs. It also achieves notable accuracy on the more challenging MGD benchmark under the blind docking protocol. Remarkably, our experiments reveal that the buried surface area calculated from predicted structures correlates with experimentally obtained degradation potency-related metrics. Consequently, DeepTernary shows potential in effectively assisting and accelerating the development of TPDs for previously undruggable targets. 

**Abstract (ZH)**: 基于深度学习的 ternary 结构预测方法 DeepTernary 用于靶向蛋白质降解药物发现 

---
# Learning to Align Multi-Faceted Evaluation: A Unified and Robust Framework 

**Title (ZH)**: 学习多面向评价对齐：一个统一且 robust 的框架 

**Authors**: Kaishuai Xu, Tiezheng Yu, Wenjun Hou, Yi Cheng, Liangyou Li, Xin Jiang, Lifeng Shang, Qun Liu, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.18874)  

**Abstract**: Large Language Models (LLMs) are being used more and more extensively for automated evaluation in various scenarios. Previous studies have attempted to fine-tune open-source LLMs to replicate the evaluation explanations and judgments of powerful proprietary models, such as GPT-4. However, these methods are largely limited to text-based analyses under predefined general criteria, resulting in reduced adaptability for unseen instructions and demonstrating instability in evaluating adherence to quantitative and structural constraints. To address these limitations, we propose a novel evaluation framework, ARJudge, that adaptively formulates evaluation criteria and synthesizes both text-based and code-driven analyses to evaluate LLM responses. ARJudge consists of two components: a fine-tuned Analyzer that generates multi-faceted evaluation analyses and a tuning-free Refiner that combines and refines all analyses to make the final judgment. We construct a Composite Analysis Corpus that integrates tasks for evaluation criteria generation alongside text-based and code-driven analysis generation to train the Analyzer. Our results demonstrate that ARJudge outperforms existing fine-tuned evaluators in effectiveness and robustness. Furthermore, it demonstrates the importance of multi-faceted evaluation and code-driven analyses in enhancing evaluation capabilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）正被越来越多地用于各种场景下的自动化评估。以往的研究试图对开源LLMs进行微调，以复制强大专有模型（如GPT-4）的评估解释和判断，但这些方法大多仅限于在预定义的一般标准下的文本分析，导致对未见指令的适应性较低，并在评估定量和结构性约束的遵守情况时表现出不稳定。为解决这些局限性，我们提出了一种新颖的评估框架ARJudge，该框架能够自适应地制定评估标准，并结合文本驱动和代码驱动的分析来评估LLM的响应。ARJudge包括两个组件：一个已微调的Analyzer生成多维度的评估分析，以及一个无需微调的Refiner综合并优化所有分析以作出最终判断。我们构建了一个集成评估标准生成任务以及文本驱动和代码驱动分析生成的综合分析语料库来训练Analyzer。研究结果表明，ARJudge在有效性与稳健性方面优于现有微调的评估器。此外，它还强调了多维度评估和代码驱动分析对增强评估能力的重要性。 

---
# Inscanner: Dual-Phase Detection and Classification of Auxiliary Insulation Using YOLOv8 Models 

**Title (ZH)**: Inscanner: 辅助绝缘的两阶段检测与分类方法基于YOLOv8模型 

**Authors**: Youngtae Kim, Soonju Jeong, Sardar Arslan, Dhananjay Agnihotri, Yahya Ahmed, Ali Nawaz, Jinhee Song, Hyewon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.18871)  

**Abstract**: This study proposes a two-phase methodology for detecting and classifying auxiliary insulation in structural components. In the detection phase, a YOLOv8x model is trained on a dataset of complete structural blueprints, each annotated with bounding boxes indicating areas that should contain insulation. In the classification phase, these detected insulation patches are cropped and categorized into two classes: present or missing. These are then used to train a YOLOv8x-CLS model that determines the presence or absence of auxiliary insulation. Preprocessing steps for both datasets included annotation, augmentation, and appropriate cropping of the insulation regions. The detection model achieved a mean average precision (mAP) score of 82%, while the classification model attained an accuracy of 98%. These findings demonstrate the effectiveness of the proposed approach in automating insulation detection and classification, providing a foundation for further advancements in this domain. 

**Abstract (ZH)**: 本研究提出了一种两阶段方法，用于检测和分类结构组件中的辅助绝缘。在检测阶段，使用每个标注有表示应包含绝缘区域边界的边界框的完整结构蓝图数据集对YOLOv8x模型进行训练。在分类阶段，检测到的绝缘补丁被裁剪并归类为存在或缺失两类，然后使用这些补丁来训练YOLOv8x-CLS模型以确定辅助绝缘的存在或缺失。两个数据集的预处理步骤包括标注、增强和适当裁剪绝缘区域。检测模型达到了82%的平均精度（mAP），分类模型的准确率为98%。这些发现表明，所提出的方法在自动化绝缘检测和分类方面是有效的，为该领域的进一步发展奠定了基础。 

---
# A Theoretical Perspective: How to Prevent Model Collapse in Self-consuming Training Loops 

**Title (ZH)**: 一个理论视角：如何防止自消耗训练循环中的模型崩溃 

**Authors**: Shi Fu, Yingjie Wang, Yuzhu Chen, Xinmei Tian, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2502.18865)  

**Abstract**: High-quality data is essential for training large generative models, yet the vast reservoir of real data available online has become nearly depleted. Consequently, models increasingly generate their own data for further training, forming Self-consuming Training Loops (STLs). However, the empirical results have been strikingly inconsistent: some models degrade or even collapse, while others successfully avoid these failures, leaving a significant gap in theoretical understanding to explain this discrepancy. This paper introduces the intriguing notion of recursive stability and presents the first theoretical generalization analysis, revealing how both model architecture and the proportion between real and synthetic data influence the success of STLs. We further extend this analysis to transformers in in-context learning, showing that even a constant-sized proportion of real data ensures convergence, while also providing insights into optimal synthetic data sizing. 

**Abstract (ZH)**: 高质量的数据对于训练大规模生成模型至关重要，然而可供在线获取的真实数据已几乎耗尽。因此，模型越来越多地生成自己的数据以进行进一步训练，形成了自消耗训练循环（STLs）。然而，实证结果极为不一致：一些模型退化甚至崩溃，而另一些模型则成功避免了这些失败，留下了一个重要的理论空白来解释这种差异。本文引入了递归稳定性的有趣概念，并提出了第一个理论泛化分析，揭示了模型架构和真实数据与合成数据比例如何影响STLs的成功。我们进一步将此分析扩展到上下文学习中的转换器，表明即使真实数据的比例保持恒定，也能确保收敛，并提供有关最佳合成数据大小的见解。 

---
# Sherlock: Towards Multi-scene Video Abnormal Event Extraction and Localization via a Global-local Spatial-sensitive LLM 

**Title (ZH)**: Sherlock: 向量多场景视频异常事件提取与定位，借助全局-局部空间敏感的大规模语言模型 

**Authors**: Junxiao Ma, Jingjing Wang, Jiamin Luo, Peiying Yu, Guodong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.18863)  

**Abstract**: Prior studies on Video Anomaly Detection (VAD) mainly focus on detecting whether each video frame is abnormal or not in the video, which largely ignore the structured video semantic information (i.e., what, when, and where does the abnormal event happen). With this in mind, we propose a new chat-paradigm \textbf{M}ulti-scene Video Abnormal Event Extraction and Localization (M-VAE) task, aiming to extract the abnormal event quadruples (i.e., subject, event type, object, scene) and localize such event. Further, this paper believes that this new task faces two key challenges, i.e., global-local spatial modeling and global-local spatial balancing. To this end, this paper proposes a Global-local Spatial-sensitive Large Language Model (LLM) named Sherlock, i.e., acting like Sherlock Holmes to track down the criminal events, for this M-VAE task. Specifically, this model designs a Global-local Spatial-enhanced MoE (GSM) module and a Spatial Imbalance Regulator (SIR) to address the two challenges respectively. Extensive experiments on our M-VAE instruction dataset show the significant advantages of Sherlock over several advanced Video-LLMs. This justifies the importance of global-local spatial information for the M-VAE task and the effectiveness of Sherlock in capturing such information. 

**Abstract (ZH)**: 多场景视频异常事件提取与定位（M-VAE）任务及其全局-局部空间建模与平衡 

---
# Investigating Generalization of One-shot LLM Steering Vectors 

**Title (ZH)**: 探究单次学习大语言模型导向矢量的泛化能力 

**Authors**: Jacob Dunefsky, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2502.18862)  

**Abstract**: Steering vectors have emerged as a promising approach for interpreting and controlling LLMs, but current methods typically require large contrastive datasets that are often impractical to construct and may capture spurious correlations. We propose directly optimizing steering vectors through gradient descent on a single training example, and systematically investigate how these vectors generalize. We consider several steering optimization techniques, including multiple novel ones, and find that the resulting vectors effectively mediate safety-relevant behaviors in multiple models. Indeed, in experiments on an alignment-faking model, we are able to optimize one-shot steering vectors that induce harmful behavior on benign examples and whose negations suppress harmful behavior on malign examples. And in experiments on refusal suppression, we demonstrate that one-shot optimized steering vectors can transfer across inputs, yielding a Harmbench attack success rate of 96.9%. Furthermore, to quantitatively assess steering effectiveness in instruction-tuned models, we develop a novel evaluation framework using sequence probabilities from the corresponding base model. With this framework, we analyze how steering vectors modulate an instruction-tuned LLM's ability to recover from outputting false information, and find that this ability derives from the base model. Overall, our findings suggest that optimizing steering vectors on a single example can mediate misaligned behavior in LLMs, and provide a path toward better understanding the relationship between LLM behavior and activation space structure. 

**Abstract (ZH)**: 基于单个训练样本优化导向向量以调控和解析大规模语言模型 

---
# Reimagining Personal Data: Unlocking the Potential of AI-Generated Images in Personal Data Meaning-Making 

**Title (ZH)**: 重塑个人数据：解锁AI生成图像在个人数据意味建构中的潜力 

**Authors**: Soobin Park, Hankyung Kim, Youn-kyung Lim  

**Link**: [PDF](https://arxiv.org/pdf/2502.18853)  

**Abstract**: Image-generative AI provides new opportunities to transform personal data into alternative visual forms. In this paper, we illustrate the potential of AI-generated images in facilitating meaningful engagement with personal data. In a formative autobiographical design study, we explored the design and use of AI-generated images derived from personal data. Informed by this study, we designed a web-based application as a probe that represents personal data through generative images utilizing Open AI's GPT-4 model and DALL-E 3. We then conducted a 21-day diary study and interviews using the probe with 16 participants to investigate users' in-depth experiences with images generated by AI in everyday lives. Our findings reveal new qualities of experiences in users' engagement with data, highlighting how participants constructed personal meaning from their data through imagination and speculation on AI-generated images. We conclude by discussing the potential and concerns of leveraging image-generative AI for personal data meaning-making. 

**Abstract (ZH)**: 图像生成AI为将个人数据转换为替代视觉形式提供了新机会。本文阐述了AI生成图像在促进个人数据有意义互动方面的潜力。通过一种形成性自传式设计研究，我们探索了基于个人数据生成的AI图像的设计和应用。基于此研究，我们设计了一个基于Web的应用程序，作为探针，利用Open AI的GPT-4模型和DALL-E 3通过生成图像来表示个人数据。我们随后对16名参与者进行了为期21天的日志研究和访谈，调查他们在日常生活中与AI生成图像的互动体验。我们的发现揭示了用户在与数据互动中体验的新特质，强调了参与者如何通过想象和推测AI生成的图像来构建个人意义。最后，我们讨论了利用图像生成AI进行个人数据意义构建的潜力和关切。 

---
# Marking Code Without Breaking It: Code Watermarking for Detecting LLM-Generated Code 

**Title (ZH)**: 不破坏代码的标记：面向检测大模型生成代码的代码水印技术 

**Authors**: Jungin Kim, Shinwoo Park, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.18851)  

**Abstract**: Code watermarking identifies AI-generated code by embedding patterns into the code during generation. Effective watermarking requires meeting two key conditions: the watermark should be reliably detectable, and the code should retain its original functionality. However, existing methods often modify tokens that are critical for program logic, such as keywords in conditional expressions or operators in arithmetic computations. These modifications can cause syntax errors or functional failures, limiting the practical use of watermarking. We present STONE, a method that preserves functional integrity by selectively inserting watermarks only into non-syntax tokens. By excluding tokens essential for code execution, STONE minimizes the risk of functional degradation.
In addition, we introduce CWEM, a comprehensive evaluation metric that evaluates watermarking techniques based on correctness, detectability, and naturalness. While correctness and detectability have been widely used, naturalness remains underexplored despite its importance. Unnatural patterns can reveal the presence of a watermark, making it easier for adversaries to remove. We evaluate STONE using CWEM and compare its performance with the state-of-the-art approach. The results show that STONE achieves an average improvement of 7.69% in CWEM across Python, C++, and Java. Our code is available in this https URL. 

**Abstract (ZH)**: 代码水印通过在生成过程中嵌入模式来识别AI生成的代码。有效的水印需要满足两个关键条件：水印应该可靠地可被检测到，并且代码应保留其原始功能。然而，现有方法 often 修改关键程序逻辑的标记，如条件表达式中的关键字或算术计算中的操作符。这些修改可能导致语法错误或功能故障，限制了水印技术的实际应用。我们提出了一种名为STONE的方法，通过仅选择性地将水印插入非语法标记来保持功能性完整性。通过排除对于代码执行至关重要的标记，STONE最小化了功能退化的风险。
此外，我们引入了CWEM，这是一种全面的评估指标，基于正确性、可检测性和自然性来评估水印技术。虽然正确性和可检测性已广为使用，但自然性的重要性虽重要却尚未得到充分探索。不自然的模式可能会揭示水印的存在，使对手更易去除。我们使用CWEM评估了STONE，并将其性能与最先进的方法进行了比较。结果表明，STONE在Python、C++和Java中的CWEM平均提高了7.69%。我们的代码可在以下链接获取：this https URL。 

---
# A Causal Lens for Evaluating Faithfulness Metrics 

**Title (ZH)**: 一种因果视角下的信实性度量评价框架 

**Authors**: Kerem Zaman, Shashank Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2502.18848)  

**Abstract**: Large Language Models (LLMs) offer natural language explanations as an alternative to feature attribution methods for model interpretability. However, despite their plausibility, they may not reflect the model's internal reasoning faithfully, which is crucial for understanding the model's true decision-making processes. Although several faithfulness metrics have been proposed, a unified evaluation framework remains absent. To address this gap, we present Causal Diagnosticity, a framework to evaluate faithfulness metrics for natural language explanations. Our framework employs the concept of causal diagnosticity, and uses model-editing methods to generate faithful-unfaithful explanation pairs. Our benchmark includes four tasks: fact-checking, analogy, object counting, and multi-hop reasoning. We evaluate a variety of faithfulness metrics, including post-hoc explanation and chain-of-thought-based methods. We find that all tested faithfulness metrics often fail to surpass a random baseline. Our work underscores the need for improved metrics and more reliable interpretability methods in LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过自然语言解释为模型可解释性提供了一种替代特征归因方法的途径。然而，尽管这些方法具有合理性，它们可能并不忠实地反映模型的内部推理过程，这对于理解模型的真实决策过程至关重要。虽然已经提出了几种忠实地度量方法，但缺乏一个统一的评估框架。为此，我们提出了因果诊断性框架，用于评估自然语言解释的忠实地度量方法。我们的框架采用了因果诊断性的概念，并使用模型编辑方法生成忠实地度量对。我们的基准包括四个任务：事实核查、类比、物体计数和多跳推理。我们评估了多种忠实地度量方法，包括事后解释和基于推理链的方法。发现所有测试的忠实地度量方法通常不能超过随机基线。我们的工作强调了需要改进的度量方法和更可靠的可解释性方法在LLMs中的需求。 

---
# Sliding Window Attention Training for Efficient Large Language Models 

**Title (ZH)**: 滑动窗口注意力训练高效大型语言模型 

**Authors**: Zichuan Fu, Wentao Song, Yejing Wang, Xian Wu, Yefeng Zheng, Yingying Zhang, Derong Xu, Xuetao Wei, Tong Xu, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.18845)  

**Abstract**: Recent advances in transformer-based Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks. However, their quadratic computational complexity concerning sequence length remains a significant bottleneck for processing long documents. As a result, many efforts like sparse attention and state space models have been proposed to improve the efficiency of LLMs over long sequences. Though effective, these approaches compromise the performance or introduce structural complexity. This calls for a simple yet efficient model that preserves the fundamental Transformer architecture. To this end, we introduce SWAT, which enables efficient long-context handling via Sliding Window Attention Training. This paper first attributes the inefficiency of Transformers to the attention sink phenomenon resulting from the high variance of softmax operation. Then, we replace softmax with the sigmoid function and utilize a balanced ALiBi and Rotary Position Embedding for efficient information compression and retention. Experiments demonstrate that SWAT achieves SOTA performance compared with state-of-the-art linear recurrent architectures on eight benchmarks. Code is available at this https URL. 

**Abstract (ZH)**: 最近基于Transformer的大语言模型（LLMs）在各种任务中展现了显著的能力。然而，它们与序列长度相关的二次计算复杂性仍然是处理长文档的一个显著瓶颈。为此，人们提出了稀疏注意机制和状态空间模型等方法来提高LLMs在长序列上的效率。尽管这些方法有效，但它们会牺牲性能或增加结构复杂性。因此，需要一种简单且高效的模型来保留基本的Transformer架构。为此，我们提出了SWAT，通过滑动窗口注意力训练实现高效的长上下文处理。本文首先将Transformer的低效归因于softmax操作的高方差引起的注意陷井现象。然后，我们用sigmoid函数替代softmax，并使用平衡的ALiBi和旋转位置嵌入来进行高效的信息压缩和保留。实验结果表明，SWAT在八个基准测试中实现了与最先进的线性递归架构相当甚至更好的性能。代码可在以下链接获取：this https URL。 

---
# BarkXAI: A Lightweight Post-Hoc Explainable Method for Tree Species Classification with Quantifiable Concepts 

**Title (ZH)**: BarkXAI：一种基于可量化概念的轻量级后验可解释方法用于树种分类 

**Authors**: Yunmei Huang, Songlin Hou, Zachary Nelson Horve, Songlin Fei  

**Link**: [PDF](https://arxiv.org/pdf/2502.18844)  

**Abstract**: The precise identification of tree species is fundamental to forestry, conservation, and environmental monitoring. Though many studies have demonstrated that high accuracy can be achieved using bark-based species classification, these models often function as "black boxes", limiting interpretability, trust, and adoption in critical forestry applications. Attribution-based Explainable AI (XAI) methods have been used to address this issue in related works. However, XAI applications are often dependent on local features (such as a head shape or paw in animal applications) and cannot describe global visual features (such as ruggedness or smoothness) that are present in texture-dominant images such as tree bark. Concept-based XAI methods, on the other hand, offer explanations based on global visual features with concepts, but they tend to require large overhead in building external concept image datasets and the concepts can be vague and subjective without good means of precise quantification. To address these challenges, we propose a lightweight post-hoc method to interpret visual models for tree species classification using operators and quantifiable concepts. Our approach eliminates computational overhead, enables the quantification of complex concepts, and evaluates both concept importance and the model's reasoning process. To the best of our knowledge, our work is the first study to explain bark vision models in terms of global visual features with concepts. Using a human-annotated dataset as ground truth, our experiments demonstrate that our method significantly outperforms TCAV and Llama3.2 in concept importance ranking based on Kendall's Tau, highlighting its superior alignment with human perceptions. 

**Abstract (ZH)**: 基于概念的轻量级后处理方法解释树皮视觉模型的全局视觉特征 

---
# Attention-Guided Integration of CLIP and SAM for Precise Object Masking in Robotic Manipulation 

**Title (ZH)**: 基于注意力引导的CLIP和SAM集成方法在机器人操作中实现精确对象掩码 

**Authors**: Muhammad A. Muttaqien, Tomohiro Motoda, Ryo Hanai, Domae Yukiyasu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18842)  

**Abstract**: This paper introduces a novel pipeline to enhance the precision of object masking for robotic manipulation within the specific domain of masking products in convenience stores. The approach integrates two advanced AI models, CLIP and SAM, focusing on their synergistic combination and the effective use of multimodal data (image and text). Emphasis is placed on utilizing gradient-based attention mechanisms and customized datasets to fine-tune performance. While CLIP, SAM, and Grad- CAM are established components, their integration within this structured pipeline represents a significant contribution to the field. The resulting segmented masks, generated through this combined approach, can be effectively utilized as inputs for robotic systems, enabling more precise and adaptive object manipulation in the context of convenience store products. 

**Abstract (ZH)**: 本文提出了一种新的流水线，以提高便利商店产品机器人操作中对象遮罩的精度。该方法结合了CLIP和SAM两种先进的AI模型，强调它们的协同作用和多模态数据（图像和文本）的有效利用。重点在于利用基于梯度的注意力机制和定制的数据集来优化性能。虽然CLIP、SAM和Grad-CAM是现有的组件，但它们在该结构化流水线中的集成对领域做出了重要贡献。通过此结合方法生成的分割遮罩可以有效地作为机器人系统的输入，从而在便利商店产品上下文中实现更精确和适应性的对象操作。 

---
# BatteryLife: A Comprehensive Dataset and Benchmark for Battery Life Prediction 

**Title (ZH)**: BatteryLife：一个全面的电池寿命预测数据集和基准 

**Authors**: Ruifeng Tan, Weixiang Hong, Jiayue Tang, Xibin Lu, Ruijun Ma, Xiang Zheng, Jia Li, Jiaqiang Huang, Tong-Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18807)  

**Abstract**: Battery Life Prediction (BLP), which relies on time series data produced by battery degradation tests, is crucial for battery utilization, optimization, and production. Despite impressive advancements, this research area faces three key challenges. Firstly, the limited size of existing datasets impedes insights into modern battery life data. Secondly, most datasets are restricted to small-capacity lithium-ion batteries tested under a narrow range of diversity in labs, raising concerns about the generalizability of findings. Thirdly, inconsistent and limited benchmarks across studies obscure the effectiveness of baselines and leave it unclear if models popular in other time series fields are effective for BLP. To address these challenges, we propose BatteryLife, a comprehensive dataset and benchmark for BLP. BatteryLife integrates 16 datasets, offering a 2.4 times sample size compared to the previous largest dataset, and provides the most diverse battery life resource with batteries from 8 formats, 80 chemical systems, 12 operating temperatures, and 646 charge/discharge protocols, including both laboratory and industrial tests. Notably, BatteryLife is the first to release battery life datasets of zinc-ion batteries, sodium-ion batteries, and industry-tested large-capacity lithium-ion batteries. With the comprehensive dataset, we revisit the effectiveness of baselines popular in this and other time series fields. Furthermore, we propose CyclePatch, a plug-in technique that can be employed in a series of neural networks. Extensive benchmarking of 18 methods reveals that models popular in other time series fields can be unsuitable for BLP, and CyclePatch consistently improves model performance establishing state-of-the-art benchmarks. Moreover, BatteryLife evaluates model performance across aging conditions and domains. BatteryLife is available at this https URL. 

**Abstract (ZH)**: 电池寿命预测（BLP）：基于电池退化测试产生的时间序列数据，对于电池的利用、优化和生产至关重要。尽管取得了显著进展，但该研究领域仍面临三大关键挑战。首先，现有数据集规模有限，限制了对现代电池寿命数据的深入洞察。其次，大多数数据集仅限于实验室条件下测试的小容量锂离子电池，范围狭窄，这使得研究结果的普适性存疑。再次，研究之间不一致且有限的基准测试阻碍了基线的有效性评估，使得其他时间序列领域流行的模型是否适用于BLP不甚清楚。为应对这些挑战，我们提出了BatteryLife，这是一个全面的BLP数据集和基准数据集。BatteryLife整合了16个数据集，相比之前的最大数据集样本量增加了2.4倍，并提供了最多样化的电池寿命资源，包括8种电池格式、80种化学体系、12种工作温度和646种充放电协议，涵盖了实验室和工业测试。值得注意的是，BatteryLife是首次发布锌离子电池、钠离子电池和工业测试的大容量锂离子电池寿命数据集。借助全面的数据集，我们重新评估了这一领域和其他时间序列领域流行的基线的有效性。此外，我们提出了CyclePatch，这是一种可以在一系列神经网络中使用的插件技术。对18种方法的大规模基准测试表明，其他时间序列领域的流行模型可能不适用于BLP，而CyclePatch在模型性能上持续提升，建立了最先进的基准。此外，BatteryLife评估了模型在不同老化条件和领域的性能。BatteryLife可在以下链接访问：this https URL。 

---
# ANPMI: Assessing the True Comprehension Capabilities of LLMs for Multiple Choice Questions 

**Title (ZH)**: ANPMI: 评估大型语言模型在多项选择题中真正理解的能力 

**Authors**: Gyeongje Cho, Yeonkyoung So, Jaejin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.18798)  

**Abstract**: Multiple-choice benchmarks, consisting of various prompts and choices, are among the most widely used methods to assess a language model's natural language understanding capability. Given a specific prompt, we typically compute $P(Choice|Prompt)$ to evaluate how likely a language model is to generate the correct choice compared to incorrect ones. However, we observe that performance measured using this approach reflects not only the model's comprehension of the prompt but also its inherent biases for certain choices regardless of the prompt. This issue makes it challenging to accurately measure a model's natural language understanding, as models may select the answer without fully understanding the prompt. To address this limitation, we propose a novel metric called ANPMI, which normalizes Pointwise Mutual Information (PMI) by $-\log P(Choice)$. ANPMI provides a more accurate assessment of the model's natural language understanding by ensuring that it is challenging to answer a question without properly understanding the prompt. 

**Abstract (ZH)**: 多种选择基准，由各种提示和选项组成，是最广泛用于评估语言模型自然语言理解能力的方法之一。给定特定的提示，我们通常计算$P(Choice|Prompt)$来评估语言模型生成正确选项而非错误选项的可能性。然而，我们观察到，使用这种方法衡量的性能不仅反映出模型对提示的理解能力，还反映了模型对某些选项的固有偏好。这一问题使得准确衡量模型的自然语言理解能力变得困难，因为模型可能在未完全理解提示的情况下选择答案。为解决这一局限，我们提出了一个新的评估指标ANPMI，该指标通过$-\log P(Choice)$对点互信息（PMI）进行归一化。ANPMI通过确保没有正确理解提示就难以回答问题，提供了模型自然语言理解能力更为准确的评估。 

---
# Seeing the Forest for the Trees: A Large Scale, Continuously Updating Meta-Analysis of Frontier LLMs 

**Title (ZH)**: 从树木中见森林：前沿大语言模型的大型持续更新元分析 

**Authors**: Jungsoo Park, Junmo Kang, Gabriel Stanovsky, Alan Ritter  

**Link**: [PDF](https://arxiv.org/pdf/2502.18791)  

**Abstract**: The surge of LLM studies makes synthesizing their findings challenging. Meta-analysis can uncover important trends across studies, but its use is limited by the time-consuming nature of manual data extraction. Our study presents a semi-automated approach for meta-analysis that accelerates data extraction using LLMs. It automatically identifies relevant arXiv papers, extracts experimental results and related attributes, and organizes them into a structured dataset. We conduct a comprehensive meta-analysis of frontier LLMs using an automatically extracted dataset, reducing the effort of paper surveying and data extraction by more than 93\% compared to manual approaches. We validate our dataset by showing that it reproduces key findings from a recent manual meta-analysis about Chain-of-Thought (CoT), and also uncovers new insights that go beyond it, showing for example that in-context examples benefit multimodal tasks but offer limited gains in mathematical tasks compared to CoT. Our automatically updatable dataset enables continuous tracking of target models by extracting evaluation studies as new data becomes available. Through our scientific artifacts and empirical analysis, we provide novel insights into LLMs while facilitating ongoing meta-analyses of their behavior. 

**Abstract (ZH)**: LLM研究 surge 促使综合其成果变得更具挑战性：semi-自动化元分析方法加速数据提取的研究 

---
# NeuroTree: Hierarchical Functional Brain Pathway Decoding for Mental Health Disorders 

**Title (ZH)**: NeuroTree: 分层功能性脑路径解码在心理健康障碍中的应用 

**Authors**: Jun-En Ding, Dongsheng Luo, Anna Zilverstand, Feng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18786)  

**Abstract**: Analyzing functional brain networks using functional magnetic resonance imaging (fMRI) is crucial for understanding psychiatric disorders and addictive behaviors. While existing fMRI-based graph convolutional networks (GCNs) show considerable promise for feature extraction, they often fall short in characterizing complex relationships between brain regions and demographic factors and accounting for interpretable variables linked to psychiatric conditions. We propose NeuroTree to overcome these limitations, integrating a k-hop AGE-GCN with neural ordinary differential equations (ODEs). This framework leverages an attention mechanism to optimize functional connectivity (FC), thereby enhancing dynamic FC feature learning for brain disease classification. Furthermore, NeuroTree effectively decodes fMRI network features into tree structures, which improves the capture of high-order brain regional pathway features and enables the identification of hierarchical neural behavioral patterns essential for understanding disease-related brain subnetworks. Our empirical evaluations demonstrate that NeuroTree achieves state-of-the-art performance across two distinct mental disorder datasets and provides valuable insights into age-related deterioration patterns. These findings underscore the model's efficacy in predicting psychiatric disorders and elucidating their underlying neural mechanisms. 

**Abstract (ZH)**: 使用功能性磁共振成像（fMRI）分析功能脑网络对于理解精神疾病和成瘾行为至关重要。尽管现有的基于fMRI的图卷积网络（GCNs）在特征提取方面显示出巨大潜力，但它们往往在表征脑区间的复杂关系和人口统计学因素以及解释与精神疾病相关联的可解释变量方面存在不足。我们提出NeuroTree以克服这些局限，结合k-hop AGE-GCN与神经ordinary差分方程（ODEs）。该框架利用注意机制优化功能性连接性（FC），从而增强动态FC特征学习，提高脑疾病分类效果。此外，NeuroTree有效将fMRI网络特征解码为树结构，这有助于捕捉高阶脑区域路径特征，并使识别与理解与疾病相关的脑子网络相关的层次神经行为模式成为可能。我们的实证评估表明，NeuroTree在两个不同的精神障碍数据集中实现了最先进的性能，并提供了有关年龄相关退化模式的有价值见解。这些发现突显了该模型在预测精神疾病和阐明其潜在神经机制方面的有效性。 

---
# Research on Edge Computing and Cloud Collaborative Resource Scheduling Optimization Based on Deep Reinforcement Learning 

**Title (ZH)**: 基于深度强化学习的边缘计算与云协同资源调度优化研究 

**Authors**: Yuqing Wang, Xiao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18773)  

**Abstract**: This study addresses the challenge of resource scheduling optimization in edge-cloud collaborative computing using deep reinforcement learning (DRL). The proposed DRL-based approach improves task processing efficiency, reduces overall processing time, enhances resource utilization, and effectively controls task migrations. Experimental results demonstrate the superiority of DRL over traditional scheduling algorithms, particularly in managing complex task allocation, dynamic workloads, and multiple resource constraints. Despite its advantages, further improvements are needed to enhance learning efficiency, reduce training time, and address convergence issues. Future research should focus on increasing the algorithm's fault tolerance to handle more complex and uncertain scheduling scenarios, thereby advancing the intelligence and efficiency of edge-cloud computing systems. 

**Abstract (ZH)**: 本研究借助深度强化学习（DRL）解决了边缘-云协同计算中的资源调度优化挑战。提出的基于DRL的方法提高了任务处理效率，减少了整体处理时间，提高了资源利用率，并有效控制了任务迁移。实验结果表明，DRL在管理复杂任务分配、动态工作负载和多种资源约束方面优于传统调度算法。尽管具有优势，仍需进一步提高学习效率，减少训练时间，并解决收敛问题。未来研究应集中在增加算法的容错性，以处理更复杂和不确定的调度场景，从而推动边缘-云计算系统的智能化和效率提升。 

---
# Reward Shaping to Mitigate Reward Hacking in RLHF 

**Title (ZH)**: 通过奖励塑造减轻RLHF中的奖励破解问题 

**Authors**: Jiayi Fu, Xuandong Zhao, Chengyuan Yao, Heng Wang, Qi Han, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2502.18770)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is essential for aligning large language models (LLMs) with human values. However, RLHF is susceptible to reward hacking, where the agent exploits flaws in the reward function rather than learning the intended behavior, thus degrading alignment. While reward shaping helps stabilize RLHF and partially mitigate reward hacking, a systematic investigation into shaping techniques and their underlying principles remains lacking. To bridge this gap, we present a comprehensive study of the prevalent reward shaping methods. Our analysis suggests three key design principles: (1) RL reward is ideally bounded, (2) RL benefits from rapid initial growth followed by gradual convergence, and (3) RL reward is best formulated as a function of centered reward. Guided by these insights, we propose Preference As Reward (PAR), a novel approach that leverages the latent preferences embedded within the reward model itself as the signal for reinforcement learning. We evaluated PAR on two base models, Gemma2-2B and Llama3-8B, using two datasets, Ultrafeedback-Binarized and HH-RLHF. Experimental results demonstrate PAR's superior performance over other reward shaping methods. On the AlpacaEval 2.0 benchmark, PAR achieves a win rate at least 5 percentage points higher than competing approaches. Furthermore, PAR exhibits remarkable data efficiency, requiring only a single reference reward for optimal performance, and maintains robustness against reward hacking even after two full epochs of training. Code is available at this https URL. 

**Abstract (ZH)**: 人类反馈强化学习（RLHF）对于使大型语言模型（LLMs）与人类价值观保持一致是必不可少的。然而，RLHF 易受奖励作弊的影响，即代理利用奖励函数中的缺陷而非学习预期行为，从而降低对齐效果。尽管奖励塑形有助于稳定 RLHF 并部分缓解奖励作弊，但对塑形技术及其基本原理的系统性研究仍存在不足。为了弥补这一差距，我们对常见的奖励塑形方法进行了全面研究。我们的分析提出了三条关键设计原则：（1）RL 奖励应是有限的，（2）RL 从快速初期增长后逐渐收敛受益，（3）RL 奖励最好以中心化奖励的函数形式进行表达。受此见解的启发，我们提出了Preference As Reward (PAR) 新方法，该方法利用奖励模型本身嵌入的潜在偏好作为强化学习的信号。我们在两个基础模型Gemma2-2B和Llama3-8B上，使用Ultrafeedback-Binarized和HH-RLHF两个数据集对PAR进行了评估。实验结果表明PAR在其他奖励塑形方法中表现更优。在AlpacaEval 2.0基准测试中，PAR 的胜率至少比竞争方法高5个百分点。此外，PAR 表现出显著的数据效率，只需单个参考奖励即可实现最优性能，并且即使经过两次完整的训练周期后仍能保持对奖励作弊的鲁棒性。代码可在以下链接获取。 

---
# Online Prototypes and Class-Wise Hypergradients for Online Continual Learning with Pre-Trained Models 

**Title (ZH)**: 基于预训练模型的在线连续学习中在线原型和类别的超梯度方法 

**Authors**: Nicolas Michel, Maorong Wang, Jiangpeng He, Toshihiko Yamasaki  

**Link**: [PDF](https://arxiv.org/pdf/2502.18762)  

**Abstract**: Continual Learning (CL) addresses the problem of learning from a data sequence where the distribution changes over time. Recently, efficient solutions leveraging Pre-Trained Models (PTM) have been widely explored in the offline CL (offCL) scenario, where the data corresponding to each incremental task is known beforehand and can be seen multiple times. However, such solutions often rely on 1) prior knowledge regarding task changes and 2) hyper-parameter search, particularly regarding the learning rate. Both assumptions remain unavailable in online CL (onCL) scenarios, where incoming data distribution is unknown and the model can observe each datum only once. Therefore, existing offCL strategies fall largely behind performance-wise in onCL, with some proving difficult or impossible to adapt to the online scenario. In this paper, we tackle both problems by leveraging Online Prototypes (OP) and Class-Wise Hypergradients (CWH). OP leverages stable output representations of PTM by updating its value on the fly to act as replay samples without requiring task boundaries or storing past data. CWH learns class-dependent gradient coefficients during training to improve over sub-optimal learning rates. We show through experiments that both introduced strategies allow for a consistent gain in accuracy when integrated with existing approaches. We will make the code fully available upon acceptance. 

**Abstract (ZH)**: 持续学习中的在线原型和类内梯度系数在 Offline 到 Online 持续学习过渡中的应用 

---
# Learning Autonomy: Off-Road Navigation Enhanced by Human Input 

**Title (ZH)**: 自主学习：由人类输入增强的离线导航 

**Authors**: Akhil Nagariya, Dimitar Filev, Srikanth Saripalli, Gaurav Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2502.18760)  

**Abstract**: In the area of autonomous driving, navigating off-road terrains presents a unique set of challenges, from unpredictable surfaces like grass and dirt to unexpected obstacles such as bushes and puddles. In this work, we present a novel learning-based local planner that addresses these challenges by directly capturing human driving nuances from real-world demonstrations using only a monocular camera. The key features of our planner are its ability to navigate in challenging off-road environments with various terrain types and its fast learning capabilities. By utilizing minimal human demonstration data (5-10 mins), it quickly learns to navigate in a wide array of off-road conditions. The local planner significantly reduces the real world data required to learn human driving preferences. This allows the planner to apply learned behaviors to real-world scenarios without the need for manual fine-tuning, demonstrating quick adjustment and adaptability in off-road autonomous driving technology. 

**Abstract (ZH)**: 自主驾驶领域中的非铺装地形导航面临独特的挑战，包括不可预测的地面如草地和泥土，以及意外的障碍物如灌木和水坑。本文提出了一种新颖的学习型局部规划器，通过仅使用单目摄像头实时捕获真实驾驶示范中的人类驾驶细节，来应对这些挑战。该规划器的关键特性在于其能够在多种地形类型的挑战性非铺装环境中导航，并且具有快速学习能力。借助少量的人工示范数据（5-10分钟），它能够迅速学会在各种非铺装条件下导航。该局部规划器大大减少了学习人类驾驶偏好的所需真实世界数据量，使得规划器能够将学到的行为应用到真实世界场景中，而无需手动微调，从而展示了在非铺装自主驾驶技术中的快速调整和适应能力。 

---
# AgentSociety Challenge: Designing LLM Agents for User Modeling and Recommendation on Web Platforms 

**Title (ZH)**: AgentSociety挑战：设计面向Web平台的用户建模与推荐的LLM代理 

**Authors**: Yuwei Yan, Yu Shang, Qingbin Zeng, Yu Li, Keyu Zhao, Zhiheng Zheng, Xuefei Ning, Tianji Wu, Shengen Yan, Yu Wang, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.18754)  

**Abstract**: The AgentSociety Challenge is the first competition in the Web Conference that aims to explore the potential of Large Language Model (LLM) agents in modeling user behavior and enhancing recommender systems on web platforms. The Challenge consists of two tracks: the User Modeling Track and the Recommendation Track. Participants are tasked to utilize a combined dataset from Yelp, Amazon, and Goodreads, along with an interactive environment simulator, to develop innovative LLM agents. The Challenge has attracted 295 teams across the globe and received over 1,400 submissions in total over the course of 37 official competition days. The participants have achieved 21.9% and 20.3% performance improvement for Track 1 and Track 2 in the Development Phase, and 9.1% and 15.9% in the Final Phase, representing a significant accomplishment. This paper discusses the detailed designs of the Challenge, analyzes the outcomes, and highlights the most successful LLM agent designs. To support further research and development, we have open-sourced the benchmark environment at this https URL. 

**Abstract (ZH)**: Web Conference上的AgentSociety挑战是首次探索大型语言模型（LLM）代理在建模用户行为和增强Web平台推荐系统潜力的比赛。该挑战包含两个赛道：用户建模赛道和推荐赛道。参赛者利用来自Yelp、Amazon和Goodreads的联合数据集以及交互环境模拟器，开发创新的LLM代理。该挑战吸引了来自全球的295个团队，并在37个官方比赛日中接收到超过1,400份提交。参赛者在开发阶段和最终阶段分别实现了21.9%和20.3%的任务1性能提升，以及9.1%和15.9%的任务2性能提升，这是一个重要的成就。本文详细讨论了挑战的设计、分析了结果，并强调了最成功的LLM代理设计。为了支持进一步的研究和开发，我们已在https://this.is/开放了基准环境。 

---
# Intent Tagging: Exploring Micro-Prompting Interactions for Supporting Granular Human-GenAI Co-Creation Workflows 

**Title (ZH)**: 意图标注：探索微观提示交互以支持细粒度的人机协同创作工作流 

**Authors**: Frederic Gmeiner, Nicolai Marquardt, Michael Bentley, Hugo Romat, Michel Pahud, David Brown, Asta Roseway, Nikolas Martelaro, Kenneth Holstein, Ken Hinckley, Nathalie Riche  

**Link**: [PDF](https://arxiv.org/pdf/2502.18737)  

**Abstract**: Despite Generative AI (GenAI) systems' potential for enhancing content creation, users often struggle to effectively integrate GenAI into their creative workflows. Core challenges include misalignment of AI-generated content with user intentions (intent elicitation and alignment), user uncertainty around how to best communicate their intents to the AI system (prompt formulation), and insufficient flexibility of AI systems to support diverse creative workflows (workflow flexibility). Motivated by these challenges, we created IntentTagger: a system for slide creation based on the notion of Intent Tags - small, atomic conceptual units that encapsulate user intent - for exploring granular and non-linear micro-prompting interactions for Human-GenAI co-creation workflows. Our user study with 12 participants provides insights into the value of flexibly expressing intent across varying levels of ambiguity, meta-intent elicitation, and the benefits and challenges of intent tag-driven workflows. We conclude by discussing the broader implications of our findings and design considerations for GenAI-supported content creation workflows. 

**Abstract (ZH)**: 尽管生成式AI（GenAI）系统在增强内容创建方面具有潜力，用户往往难以有效将GenAI集成到其创作流程中。核心挑战包括生成内容与用户意图不匹配（意图提取和对齐）、用户在如何最好地向AI系统传达其意图方面存在不确定性（提示制定），以及AI系统缺乏支持多样化创作流程的灵活性（工作流程灵活性）。为应对这些挑战，我们开发了基于意图标签（Intent Tags）的概念——小而原子的概念单元，用以概括用户意图——的幻灯片创建系统，以探索细粒度和非线性的微提示互动，以支持人类与GenAI共创作的工作流程。我们的用户研究（包含12名参与者）提供了关于灵活表达意图、在不同模糊度级别上提取元意图的价值，以及基于意图标签驱动的工作流程的优势和挑战的见解。最后，我们讨论了研究发现的更广泛意义，并提出了支持GenAI辅助内容创作工作流程的设计考虑。 

---
# AI-Instruments: Embodying Prompts as Instruments to Abstract & Reflect Graphical Interface Commands as General-Purpose Tools 

**Title (ZH)**: AI-Instruments: 将提示拟人化为工具以抽象和反思图形界面命令作为通用工具 

**Authors**: Nathalie Riche, Anna Offenwanger, Frederic Gmeiner, David Brown, Hugo Romat, Michel Pahud, Nicolai Marquardt, Kori Inkpen, Ken Hinckley  

**Link**: [PDF](https://arxiv.org/pdf/2502.18736)  

**Abstract**: Chat-based prompts respond with verbose linear-sequential texts, making it difficult to explore and refine ambiguous intents, back up and reinterpret, or shift directions in creative AI-assisted design work. AI-Instruments instead embody "prompts" as interface objects via three key principles: (1) Reification of user-intent as reusable direct-manipulation instruments; (2) Reflection of multiple interpretations of ambiguous user-intents (Reflection-in-intent) as well as the range of AI-model responses (Reflection-in-response) to inform design "moves" towards a desired result; and (3) Grounding to instantiate an instrument from an example, result, or extrapolation directly from another instrument. Further, AI-Instruments leverage LLM's to suggest, vary, and refine new instruments, enabling a system that goes beyond hard-coded functionality by generating its own instrumental controls from content. We demonstrate four technology probes, applied to image generation, and qualitative insights from twelve participants, showing how AI-Instruments address challenges of intent formulation, steering via direct manipulation, and non-linear iterative workflows to reflect and resolve ambiguous intents. 

**Abstract (ZH)**: 基于聊天的提示生成冗长的线性文本，这使得在创造性AI辅助设计工作中探索、精炼模糊意图、回溯、重新解释或转向变得困难。相反，AI-Instruments将“提示”作为界面对象，通过三个关键原则实现：（1）用户意图的物化为可重复使用的直接操作工具；（2）通过意图内的反思（Reflection-in-intent）和响应内的反思（Reflection-in-response）反映模糊用户意图的多种解释以及AI模型响应的范围，从而指导设计“操作”以达成预期结果；（3）基于实例、结果或另一工具的推演进行工具的实例化。此外，AI-Instruments利用大语言模型（LLMs）来建议、变化和精炼新的工具，使系统能够自动生成其自身的控制工具，超越了硬编码的功能。我们通过应用图像生成技术探针，并从12名参与者中获得质性见解，展示了AI-Instruments如何解决意图表述、直接操作导向操纵以及非线性迭代工作流程中反映和解决模糊意图的挑战。 

---
# Cross-Modality Investigation on WESAD Stress Classification 

**Title (ZH)**: 跨模态研究在WESAD压力分类中的应用 

**Authors**: Eric Oliver, Sagnik Dakshit  

**Link**: [PDF](https://arxiv.org/pdf/2502.18733)  

**Abstract**: Deep learning's growing prevalence has driven its widespread use in healthcare, where AI and sensor advancements enhance diagnosis, treatment, and monitoring. In mobile health, AI-powered tools enable early diagnosis and continuous monitoring of conditions like stress. Wearable technologies and multimodal physiological data have made stress detection increasingly viable, but model efficacy depends on data quality, quantity, and modality. This study develops transformer models for stress detection using the WESAD dataset, training on electrocardiograms (ECG), electrodermal activity (EDA), electromyography (EMG), respiration rate (RESP), temperature (TEMP), and 3-axis accelerometer (ACC) signals. The results demonstrate the effectiveness of single-modality transformers in analyzing physiological signals, achieving state-of-the-art performance with accuracy, precision and recall values in the range of $99.73\%$ to $99.95\%$ for stress detection. Furthermore, this study explores cross-modal performance and also explains the same using 2D visualization of the learned embedding space and quantitative analysis based on data variance. Despite the large body of work on stress detection and monitoring, the robustness and generalization of these models across different modalities has not been explored. This research represents one of the initial efforts to interpret embedding spaces for stress detection, providing valuable information on cross-modal performance. 

**Abstract (ZH)**: 深度学习的广泛应用促进了其在医疗健康领域的广泛应用，其中人工智能和传感器的进步提升了诊断、治疗和监测的效果。在移动医疗中，基于人工智能的工具能够实现压力等疾病的早期诊断和持续监测。可穿戴技术与多模态生理数据使得压力检测愈发可行，但模型的有效性依赖于数据的质量、数量和模态。本研究使用WESAD数据集开发了针对压力检测的变换器模型，并通过对心电图（ECG）、电导率活动（EDA）、肌电图（EMG）、呼吸率（RESP）、温度（TEMP）以及3轴加速度计（ACC）信号的训练，展示了单模态变换器在分析生理信号方面的有效性，在压力检测中取得了高达99.73%至99.95%的准确率、精确率和召回率，同时探讨了跨模态性能，并通过学习嵌入空间的2D可视化及基于数据方差的定量分析进行了解释。尽管已有大量关于压力检测和监测的研究，但这些模型在不同模态下的鲁棒性和泛化性尚未被充分探索。本研究是首次尝试解释嵌入空间在压力检测中的跨模态性能，提供了有价值的信息。 

---
# Deep-Bench: Deep Learning Benchmark Dataset for Code Generation 

**Title (ZH)**: Deep-Bench: 用于代码生成的深度学习基准数据集 

**Authors**: Alireza Daghighfarsoodeh, Chung-Yu Wang, Hamed Taherkhani, Melika Sepidband, Mohammad Abdollahi, Hadi Hemmati, Hung Viet Pham  

**Link**: [PDF](https://arxiv.org/pdf/2502.18726)  

**Abstract**: Deep learning (DL) has revolutionized areas such as computer vision, natural language processing, and more. However, developing DL systems is challenging due to the complexity of DL workflows. Large Language Models (LLMs), such as GPT, Claude, Llama, Mistral, etc., have emerged as promising tools to assist in DL code generation, offering potential solutions to these challenges. Despite this, existing benchmarks such as DS-1000 are limited, as they primarily focus on small DL code snippets related to pre/post-processing tasks and lack a comprehensive coverage of the full DL pipeline, including different DL phases and input data types.
To address this, we introduce DeepBench, a novel benchmark dataset designed for function-level DL code generation. DeepBench categorizes DL problems based on three key aspects: phases such as pre-processing, model construction, and training; tasks, including classification, regression, and recommendation; and input data types such as tabular, image, and text.
GPT-4o -- the state-of-the-art LLM -- achieved 31% accuracy on DeepBench, significantly lower than its 60% on DS-1000. We observed similar difficulty for other LLMs (e.g., 28% vs. 54% for Claude, 21% vs. 41% for LLaMA, and 15% vs. 20% for Mistral). This result underscores DeepBench's greater complexity. We also construct a taxonomy of issues and bugs found in LLM-generated DL code, which highlights the distinct challenges that LLMs face when generating DL code compared to general code.
Furthermore, our analysis also reveals substantial performance variations across categories, with differences of up to 7% among phases and 37% among tasks. These disparities suggest that DeepBench offers valuable insights into the LLMs' performance and areas for potential improvement in the DL domain. 

**Abstract (ZH)**: 深度学习（DL）已 revolutionized 计算机视觉、自然语言处理等领域。然而，开发 DL 系统因 DL 工作流的复杂性而具有挑战性。大型语言模型（LLMs），如 GPT、Claude、Llama、Mistral 等，作为辅助 DL 代码生成的有希望工具出现，提供了应对这些挑战的潜在解决方案。尽管如此，现有的基准测试，如 DS-1000，仍有限制，因为它们主要关注与预处理/后处理任务相关的少量 DL 代码片段，并且缺乏对整个 DL 管道的综合覆盖，包括不同 DL 阶段和输入数据类型。

为应对这一挑战，我们引入了 DeepBench，一个用于功能级别 DL 代码生成的新型基准数据集。DeepBench 根据三个关键方面对 DL 问题进行分类：阶段，如预处理、模型构建和训练；任务，包括分类、回归和推荐；以及输入数据类型，如表格、图像和文本。

GPT-4o — 目前最先进的 LLM — 在 DeepBench 上的准确率为 31%，远低于其在 DS-1000 上 60% 的准确率。我们观察到其他 LLMs（例如，Claude 的 28% 对比 54%、LLaMA 的 21% 对比 41%、Mistral 的 15% 对比 20%）也存在类似难度。这一结果突显了 DeepBench 的更高复杂度。我们还构建了一个 LLM 生成的 DL 代码中存在的问题和错误的分类体系，这突显了 LLMs 在生成 DL 代码时与其他通用代码相比面临的独特挑战。

此外，我们的分析还揭示了不同类别之间存在显著的性能差异，不同阶段的性能差异可达 7%，而不同任务的性能差异可达 37%。这些差异表明 DeepBench 提供了有关 LLMs 在 DL 领域的表现和潜在改进领域的有价值的见解。 

---
# Bridging Critical Gaps in Convergent Learning: How Representational Alignment Evolves Across Layers, Training, and Distribution Shifts 

**Title (ZH)**: 汇聚学习中关键缺口的桥梁：表示对齐如何随层、训练和分布偏移演进 

**Authors**: Chaitanya Kapoor, Sudhanshu Srivastava, Meenakshi Khosla  

**Link**: [PDF](https://arxiv.org/pdf/2502.18710)  

**Abstract**: Understanding convergent learning -- the extent to which artificial and biological neural networks develop similar representations -- is crucial for neuroscience and AI, as it reveals shared learning principles and guides brain-like model design. While several studies have noted convergence in early and late layers of vision networks, key gaps remain. First, much existing work relies on a limited set of metrics, overlooking transformation invariances required for proper alignment. We compare three metrics that ignore specific irrelevant transformations: linear regression (ignoring affine transformations), Procrustes (ignoring rotations and reflections), and permutation/soft-matching (ignoring unit order). Notably, orthogonal transformations align representations nearly as effectively as more flexible linear ones, and although permutation scores are lower, they significantly exceed chance, indicating a robust representational basis. A second critical gap lies in understanding when alignment emerges during training. Contrary to expectations that convergence builds gradually with task-specific learning, our findings reveal that nearly all convergence occurs within the first epoch -- long before networks achieve optimal performance. This suggests that shared input statistics, architectural biases, or early training dynamics drive convergence rather than the final task solution. Finally, prior studies have not systematically examined how changes in input statistics affect alignment. Our work shows that out-of-distribution (OOD) inputs consistently amplify differences in later layers, while early layers remain aligned for both in-distribution and OOD inputs, suggesting that this alignment is driven by generalizable features stable across distribution shifts. These findings fill critical gaps in our understanding of representational convergence, with implications for neuroscience and AI. 

**Abstract (ZH)**: 理解收敛学习——人工和生物神经网络在多大程度上发展出相似的表示——对于神经科学和人工智能至关重要，因为它揭示了共享的学习原理，并指导类脑模型的设计。尽管已有研究表明视觉网络早期和晚期层存在收敛现象，但仍存在关键缺口。首先，现有工作很大程度上依赖于有限的度量标准，忽视了正确的对齐所需的变换不变性。我们比较了三种忽略特定无关变换的度量标准：线性回归（忽略仿射变换）、Procrustes（忽略旋转和镜像变换）和置换/软匹配（忽略单元顺序）。值得注意的是，正交变换几乎与更灵活的线性变换一样有效地对齐表示，并且尽管置换得分较低，但仍显著超过随机水平，表明有一个稳健的表征基础。第二个关键缺口在于理解对齐何时在训练期间出现。与期望的学习任务相关联的收敛现象会逐渐形成的观点相反，我们的发现揭示了几乎所有收敛现象都在第一个训练周期内出现——远在网络达到最佳性能之前。这表明，共享的输入统计、架构偏差或早期训练动力学驱动了收敛，而不是最终的任务解决方案。最后，先前的研究并没有系统地检查输入统计的变化如何影响对齐。我们的工作表明，域外（OOD）输入始终在后期层中放大差异，而早期层在既有分布和OOD输入条件下都保持对齐，这表明这种对齐是由贯穿分布变化仍保持稳定的可泛化特征驱动的。这些发现填补了我们对表示收敛理解的关键空白，对神经科学和人工智能具有重要影响。 

---
# H-FLTN: A Privacy-Preserving Hierarchical Framework for Electric Vehicle Spatio-Temporal Charge Prediction 

**Title (ZH)**: H-FLTN: 一种隐私保护的分层框架，用于电动车辆时空充电预测 

**Authors**: Robert Marlin, Raja Jurdak, Alsharif Abuadbba  

**Link**: [PDF](https://arxiv.org/pdf/2502.18697)  

**Abstract**: The widespread adoption of Electric Vehicles (EVs) poses critical challenges for energy providers, particularly in predicting charging time (temporal prediction), ensuring user privacy, and managing resources efficiently in mobility-driven networks. This paper introduces the Hierarchical Federated Learning Transformer Network (H-FLTN) framework to address these challenges. H-FLTN employs a three-tier hierarchical architecture comprising EVs, community Distributed Energy Resource Management Systems (DERMS), and the Energy Provider Data Centre (EPDC) to enable accurate spatio-temporal predictions of EV charging needs while preserving privacy. Temporal prediction is enhanced using Transformer-based learning, capturing complex dependencies in charging behavior. Privacy is ensured through Secure Aggregation, Additive Secret Sharing, and Peer-to-Peer (P2P) Sharing with Augmentation, which allow only secret shares of model weights to be exchanged while securing all transmissions. To improve training efficiency and resource management, H-FLTN integrates Dynamic Client Capping Mechanism (DCCM) and Client Rotation Management (CRM), ensuring that training remains both computationally and temporally efficient as the number of participating EVs increases. DCCM optimises client participation by limiting excessive computational loads, while CRM balances training contributions across epochs, preventing imbalanced participation. Our simulation results based on large-scale empirical vehicle mobility data reveal that DCCM and CRM reduce the training time complexity with increasing EVs from linear to constant. Its integration into real-world smart city infrastructure enhances energy demand forecasting, resource allocation, and grid stability, ensuring reliability and sustainability in future mobility ecosystems. 

**Abstract (ZH)**: 电动汽车（EVs）广泛采用对能源提供商提出了关键挑战，特别是充电时间预测（时空预测）、用户隐私保障以及在以移动性驱动的网络中高效管理资源。本文提出了层次联邦学习变换器网络（H-FLTN）框架以应对这些挑战。H-FLTN采用三层级架构，包括电动汽车、社区分布式能源资源管理系统（DERMS）和能源提供商数据中心（EPDC），以实现准确的电动汽车充电需求的时空预测同时保护隐私。通过基于变换器的学习增强时间预测，捕捉充电行为中的复杂依赖性。隐私通过安全聚合、加性秘密共享以及增强的点对点共享保障，仅交换秘密模型权重份额，确保所有传输的安全。为了提高训练效率和资源管理，H-FLTN整合了动态客户端容量控制机制（DCCM）和客户端旋转管理（CRM），确保随着参与电动汽车数量的增加，训练保持计算和时间上的高效。DCCM通过限制过重的计算负载优化客户端参与，而CRM在各周期内平衡训练贡献，防止参与不均衡。基于大规模实测车辆移动数据的仿真结果表明，DCCM和CRM随着电动汽车数量的增加，减少了训练时间复杂性，从线性变为常数。将其集成到真实的智慧城市基础设施中，可以增强能源需求预测、资源分配和电网稳定性，确保未来移动生态系统中的可靠性和可持续性。 

---
# Policy-as-Prompt: Rethinking Content Moderation in the Age of Large Language Models 

**Title (ZH)**: 政策作为提示：在大语言模型时代重新思考内容审核 

**Authors**: Konstantina Palla, José Luis Redondo García, Claudia Hauff, Francesco Fabbri, Henrik Lindström, Daniel R. Taber, Andreas Damianou, Mounia Lalmas  

**Link**: [PDF](https://arxiv.org/pdf/2502.18695)  

**Abstract**: Content moderation plays a critical role in shaping safe and inclusive online environments, balancing platform standards, user expectations, and regulatory frameworks. Traditionally, this process involves operationalising policies into guidelines, which are then used by downstream human moderators for enforcement, or to further annotate datasets for training machine learning moderation models. However, recent advancements in large language models (LLMs) are transforming this landscape. These models can now interpret policies directly as textual inputs, eliminating the need for extensive data curation. This approach offers unprecedented flexibility, as moderation can be dynamically adjusted through natural language interactions. This paradigm shift raises important questions about how policies are operationalised and the implications for content moderation practices. In this paper, we formalise the emerging policy-as-prompt framework and identify five key challenges across four domains: Technical Implementation (1. translating policy to prompts, 2. sensitivity to prompt structure and formatting), Sociotechnical (3. the risk of technological determinism in policy formation), Organisational (4. evolving roles between policy and machine learning teams), and Governance (5. model governance and accountability). Through analysing these challenges across technical, sociotechnical, organisational, and governance dimensions, we discuss potential mitigation approaches. This research provides actionable insights for practitioners and lays the groundwork for future exploration of scalable and adaptive content moderation systems in digital ecosystems. 

**Abstract (ZH)**: 政策-as-提示框架在内容审核中的作用与挑战 

---
# AI Mismatches: Identifying Potential Algorithmic Harms Before AI Development 

**Title (ZH)**: AI失配：在AI开发之前识别潜在算法危害 

**Authors**: Devansh Saxena, Ji-Youn Jung, Jodi Forlizzi, Kenneth Holstein, John Zimmerman  

**Link**: [PDF](https://arxiv.org/pdf/2502.18682)  

**Abstract**: AI systems are often introduced with high expectations, yet many fail to deliver, resulting in unintended harm and missed opportunities for benefit. We frequently observe significant "AI Mismatches", where the system's actual performance falls short of what is needed to ensure safety and co-create value. These mismatches are particularly difficult to address once development is underway, highlighting the need for early-stage intervention. Navigating complex, multi-dimensional risk factors that contribute to AI Mismatches is a persistent challenge. To address it, we propose an AI Mismatch approach to anticipate and mitigate risks early on, focusing on the gap between realistic model performance and required task performance. Through an analysis of 774 AI cases, we extracted a set of critical factors, which informed the development of seven matrices that map the relationships between these factors and highlight high-risk areas. Through case studies, we demonstrate how our approach can help reduce risks in AI development. 

**Abstract (ZH)**: AI系统常常伴随着高度的期望被引入，然而许多系统未能达到预期，从而导致了意外的危害和潜在利益的丧失。我们频繁地观察到显著的“AI不匹配”，系统的实际性能未能满足确保安全和共同创造价值的需求。这些不匹配在开发过程中尤其难以解决，突显了早期干预的必要性。导航导致AI不匹配的复杂多维风险因素是一个持续的挑战。为了解决这一问题，我们提出了一种AI不匹配方法，旨在早期预见和缓解风险，重点关注实际模型性能与所需任务性能之间的差距。通过分析774个AI案例，我们提取了一组关键因素，并据此开发了七个矩阵来映射这些因素之间的关系并突出高风险区域。通过案例研究，我们展示了这种方法如何帮助减少AI开发中的风险。 

---
# Comparing Native and Non-native English Speakers' Behaviors in Collaborative Writing through Visual Analytics 

**Title (ZH)**: 通过可视分析比较母语者与非母语者在协作写作中的行为差异 

**Authors**: Yuexi Chen, Yimin Xiao, Kazi Tasnim Zinat, Naomi Yamashita, Ge Gao, Zhicheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18681)  

**Abstract**: Understanding collaborative writing dynamics between native speakers (NS) and non-native speakers (NNS) is critical for enhancing collaboration quality and team inclusivity. In this paper, we partnered with communication researchers to develop visual analytics solutions for comparing NS and NNS behaviors in 162 writing sessions across 27 teams. The primary challenges in analyzing writing behaviors are data complexity and the uncertainties introduced by automated methods. In response, we present \textsc{COALA}, a novel visual analytics tool that improves model interpretability by displaying uncertainties in author clusters, generating behavior summaries using large language models, and visualizing writing-related actions at multiple granularities. We validated the effectiveness of \textsc{COALA} through user studies with domain experts (N=2+2) and researchers with relevant experience (N=8). We present the insights discovered by participants using \textsc{COALA}, suggest features for future AI-assisted collaborative writing tools, and discuss the broader implications for analyzing collaborative processes beyond writing. 

**Abstract (ZH)**: 理解母语者（NS）与非母语者（NNS）之间的协作写作动态对于提升协作质量和团队包容性至关重要。本文与沟通研究人员合作，开发了可视化分析解决方案，以比较27个团队中162次写作会话中母语者和非母语者的行为。分析写作行为的主要挑战包括数据复杂性和自动化方法引入的不确定性。为此，我们提出了COALA，这是一种新颖的可视化分析工具，通过显示作者集群中的不确定性、使用大规模语言模型生成行为总结以及以多种粒度层级可视化写作相关操作，来提高模型可解释性。通过与领域专家（N=2+2）和相关经验的研究者（N=8）进行用户研究，我们验证了COALA的有效性。我们展示了参与者使用COALA发现的洞见，建议了未来辅助协作写作的AI工具的功能，并讨论了其对分析协作过程（超越写作）的更广泛影响。 

---
# Assistance or Disruption? Exploring and Evaluating the Design and Trade-offs of Proactive AI Programming Support 

**Title (ZH)**: 辅助还是干扰？探究和评估主动AI编程支持的设计与权衡 

**Authors**: Kevin Pu, Daniel Lazaro, Ian Arawjo, Haijun Xia, Ziang Xiao, Tovi Grossman, Yan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18658)  

**Abstract**: AI programming tools enable powerful code generation, and recent prototypes attempt to reduce user effort with proactive AI agents, but their impact on programming workflows remains unexplored. We introduce and evaluate Codellaborator, a design probe LLM agent that initiates programming assistance based on editor activities and task context. We explored three interface variants to assess trade-offs between increasingly salient AI support: prompt-only, proactive agent, and proactive agent with presence and context (Codellaborator). In a within-subject study (N=18), we find that proactive agents increase efficiency compared to prompt-only paradigm, but also incur workflow disruptions. However, presence indicators and \revise{interaction context support} alleviated disruptions and improved users' awareness of AI processes. We underscore trade-offs of Codellaborator on user control, ownership, and code understanding, emphasizing the need to adapt proactivity to programming processes. Our research contributes to the design exploration and evaluation of proactive AI systems, presenting design implications on AI-integrated programming workflow. 

**Abstract (ZH)**: AI编程工具使代码生成更加强大，近期的原型试图通过前瞻性的AI代理减少用户 effort，但它们对编程工作流的影响尚未被探索。我们介绍并评估了Codellaborator这一设计探针语言模型代理，它基于编辑器活动和任务上下文来启动编程辅助。我们探索了三种界面变体，以评估从增加到更显著的AI支持之间的权衡：仅提示、前瞻代理以及具有存在感和上下文的前瞻代理（Codellaborator）。在一项针对同一参与者的实验研究中（N=18），我们发现前瞻代理相比于仅提示范式提高了效率，但也引发了工作流中断。然而，存在感指示符和交互上下文支持减轻了这些中断，并提高了用户对AI过程的意识。我们强调Codellaborator在用户控制、所有权和代码理解方面的权衡，并强调需要根据编程过程来适应前瞻性。我们的研究为前瞻AI系统的设计探索与评估做出了贡献，并提出了AI集成编程工作流的设计启示。 

---
# Enhancing Text Classification with a Novel Multi-Agent Collaboration Framework Leveraging BERT 

**Title (ZH)**: 利用BERT增强文本分类的新型多代理协作框架 

**Authors**: Hediyeh Baban, Sai A Pidapar, Aashutosh Nema, Sichen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18653)  

**Abstract**: We introduce a novel multi-agent collaboration framework designed to enhance the accuracy and robustness of text classification models. Leveraging BERT as the primary classifier, our framework dynamically escalates low-confidence predictions to a specialized multi-agent system comprising Lexical, Contextual, Logic, Consensus, and Explainability agents. This collaborative approach allows for comprehensive analysis and consensus-driven decision-making, significantly improving classification performance across diverse text classification tasks. Empirical evaluations on benchmark datasets demonstrate that our framework achieves a 5.5% increase in accuracy compared to standard BERT-based classifiers, underscoring its effectiveness and academic novelty in advancing multi-agent systems within natural language processing. 

**Abstract (ZH)**: 我们提出了一种新颖的多agent协作框架，旨在提升文本分类模型的准确性和鲁棒性。该框架利用BERT作为主要分类器，并动态提升低置信度预测至包括词汇、语境、逻辑、共识和解释性agent的专门多agent系统中。这种协作方法允许进行全面分析和基于共识的决策，显著提高了跨多种文本分类任务的分类性能。在基准数据集上的实证评价表明，与标准BERT基线分类器相比，该框架的准确率提高了5.5%，证明了其在自然语言处理中推进多agent系统方面的有效性和学术新颖性。 

---
# WhatELSE: Shaping Narrative Spaces at Configurable Level of Abstraction for AI-bridged Interactive Storytelling 

**Title (ZH)**: WhatELSE：在可配置抽象层次上塑造叙事空间的AI桥梁式互动叙事 

**Authors**: Zhuoran Lu, Qian Zhou, Yi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18641)  

**Abstract**: Generative AI significantly enhances player agency in interactive narratives (IN) by enabling just-in-time content generation that adapts to player actions. While delegating generation to AI makes IN more interactive, it becomes challenging for authors to control the space of possible narratives - within which the final story experienced by the player emerges from their interaction with AI. In this paper, we present WhatELSE, an AI-bridged IN authoring system that creates narrative possibility spaces from example stories. WhatELSE provides three views (narrative pivot, outline, and variants) to help authors understand the narrative space and corresponding tools leveraging linguistic abstraction to control the boundaries of the narrative space. Taking innovative LLM-based narrative planning approaches, WhatELSE further unfolds the narrative space into executable game events. Through a user study (N=12) and technical evaluations, we found that WhatELSE enables authors to perceive and edit the narrative space and generates engaging interactive narratives at play-time. 

**Abstract (ZH)**: 生成式AI显著增强了交互叙事（IN）中的玩家主动权，通过实现即时内容生成并根据玩家行为进行调整。虽然将生成任务委托给AI使IN更加互动，但作者控制可能叙事空间变得更具挑战性——在这一空间中，最终由玩家与AI互动体验到的故事得以浮现。本文介绍了WhatELSE，这是一种AI桥接的交互叙事作者系统，能够从示例故事中创建叙事可能性空间。WhatELSE 提供三种视图（叙事关键点、大纲和变体）来帮助作者理解叙事空间，并利用语言抽象提供相应工具以控制叙事空间的边界。利用创新的基于大语言模型的叙事规划方法，WhatELSE 进一步将叙事空间展开为可执行的游戏事件。通过用户研究（参与者数量为12）和技术评估，我们发现WhatELSE使作者能够感知和编辑叙事空间，并在游戏过程中生成引人入胜的交互叙事。 

---
# Quantum Machine Learning in Precision Medicine and Drug Discovery -- A Game Changer for Tailored Treatments? 

**Title (ZH)**: 量子机器学习在精准医学和药物发现中的应用——个性化治疗的颠覆者？ 

**Authors**: Markus Bertl, Alan Mott, Salvatore Sinno, Bhavika Bhalgamiya  

**Link**: [PDF](https://arxiv.org/pdf/2502.18639)  

**Abstract**: The digitization of healthcare presents numerous challenges, including the complexity of biological systems, vast data generation, and the need for personalized treatment plans. Traditional computational methods often fall short, leading to delayed and sometimes ineffective diagnoses and treatments. Quantum Computing (QC) and Quantum Machine Learning (QML) offer transformative advancements with the potential to revolutionize medicine. This paper summarizes areas where QC promises unprecedented computational power, enabling faster, more accurate diagnostics, personalized treatments, and enhanced drug discovery processes. However, integrating quantum technologies into precision medicine also presents challenges, including errors in algorithms and high costs. We show that mathematically-based techniques for specifying, developing, and verifying software (formal methods) can enhance the reliability and correctness of QC. By providing a rigorous mathematical framework, formal methods help to specify, develop, and verify systems with high precision. In genomic data analysis, formal specification languages can precisely (1) define the behavior and properties of quantum algorithms designed to identify genetic markers associated with diseases. Model checking tools can systematically explore all possible states of the algorithm to (2) ensure it behaves correctly under all conditions, while theorem proving techniques provide mathematical (3) proof that the algorithm meets its specified properties, ensuring accuracy and reliability. Additionally, formal optimization techniques can (4) enhance the efficiency and performance of quantum algorithms by reducing resource usage, such as the number of qubits and gate operations. Therefore, we posit that formal methods can significantly contribute to enabling QC to realize its full potential as a game changer in precision medicine. 

**Abstract (ZH)**: 医疗领域的数字化转型面临着众多挑战，包括生物系统的复杂性、大量数据的生成以及个性化治疗方案的需求。传统计算方法常常无法满足需求，导致诊断和治疗延误且有时无效。量子计算（QC）和量子机器学习（QML）提供了转变性的进展，有望彻底变革医学。本文总结了QC在提供前所未有的计算能力方面的领域，使诊断更快、更准确，治疗更个性化，并提升药物发现过程。然而，将量子技术整合到精准医疗中也带来了挑战，包括算法错误和高昂的成本。我们表明，基于数学的方法（形式化方法）可以增强QC的可靠性和正确性。通过提供严谨的数学框架，形式化方法有助于精确地指定、开发和验证系统。在基因组数据分析中，形式化规范语言可以精确地定义用于识别与疾病相关的遗传标记的量子算法的行为和属性。模型检查工具可以系统地探索算法的所有可能状态，以确保其在所有条件下正确执行，而定理证明技术则提供数学证明，确保算法满足其指定的属性，保证准确性和可靠性。此外，形式化优化技术可以通过减少资源使用，如量子位和门操作的数量，来增强量子算法的效率和性能。因此，我们提出，形式化方法可以显著贡献于使QC能够在精准医疗领域充分发挥其潜力。 

---
# Faster, Cheaper, Better: Multi-Objective Hyperparameter Optimization for LLM and RAG Systems 

**Title (ZH)**: 更快、更低成本、更优效果：多目标超参数优化在大语言模型和检索增强生成系统中的应用 

**Authors**: Matthew Barker, Andrew Bell, Evan Thomas, James Carr, Thomas Andrews, Umang Bhatt  

**Link**: [PDF](https://arxiv.org/pdf/2502.18635)  

**Abstract**: While Retrieval Augmented Generation (RAG) has emerged as a popular technique for improving Large Language Model (LLM) systems, it introduces a large number of choices, parameters and hyperparameters that must be made or tuned. This includes the LLM, embedding, and ranker models themselves, as well as hyperparameters governing individual RAG components. Yet, collectively optimizing the entire configuration in a RAG or LLM system remains under-explored - especially in multi-objective settings - due to intractably large solution spaces, noisy objective evaluations, and the high cost of evaluations. In this work, we introduce the first approach for multi-objective parameter optimization of cost, latency, safety and alignment over entire LLM and RAG systems. We find that Bayesian optimization methods significantly outperform baseline approaches, obtaining a superior Pareto front on two new RAG benchmark tasks. We conclude our work with important considerations for practitioners who are designing multi-objective RAG systems, highlighting nuances such as how optimal configurations may not generalize across tasks and objectives. 

**Abstract (ZH)**: 面向成本、延迟、安全性和对齐的大型语言模型和检索增强生成系统多目标参数优化方法 

---
# Diffusion Models for conditional MRI generation 

**Title (ZH)**: 条件MRI生成的扩散模型 

**Authors**: Miguel Herencia García del Castillo, Ricardo Moya Garcia, Manuel Jesús Cerezo Mazón, Ekaitz Arriola Garcia, Pablo Menéndez Fernández-Miranda  

**Link**: [PDF](https://arxiv.org/pdf/2502.18620)  

**Abstract**: In this article, we present a Latent Diffusion Model (LDM) for the generation of brain Magnetic Resonance Imaging (MRI), conditioning its generation based on pathology (Healthy, Glioblastoma, Sclerosis, Dementia) and acquisition modality (T1w, T1ce, T2w, Flair, PD).
To evaluate the quality of the generated images, the Fréchet Inception Distance (FID) and Multi-Scale Structural Similarity Index (MS-SSIM) metrics were employed. The results indicate that the model generates images with a distribution similar to real ones, maintaining a balance between visual fidelity and diversity. Additionally, the model demonstrates extrapolation capability, enabling the generation of configurations that were not present in the training data.
The results validate the potential of the model to increase in the number of samples in clinical datasets, balancing underrepresented classes, and evaluating AI models in medicine, contributing to the development of diagnostic tools in radiology without compromising patient privacy. 

**Abstract (ZH)**: 本文提出一种隐性扩散模型（LDM）用于生成脑磁共振成像（MRI），根据病理（健康、胶质母细胞瘤、硬化、痴呆）和成像模态（T1w、T1ce、T2w、FLAIR、PD）进行条件生成。为了评估生成图像的质量，采用Fréchet Inchesion Distance（FID）和Multi-Scale Structural Similarity Index（MS-SSIM）度量。结果表明，该模型生成的图像分布与真实图像相似，同时保持了视觉保真度和多样性之间的平衡。此外，该模型展示了外推能力，能够生成训练数据中未出现的配置。 

---
# Mind the Gap: Bridging the Divide Between AI Aspirations and the Reality of Autonomous Characterization 

**Title (ZH)**: 注意差距：弥合人工智能期望与自主特征化现实之间的鸿沟 

**Authors**: Grace Guinan, Addison Salvador, Michelle A. Smeaton, Andrew Glaws, Hilary Egan, Brian C. Wyatt, Babak Anasori, Kevin R. Fiedler, Matthew J. Olszta, Steven R. Spurgeon  

**Link**: [PDF](https://arxiv.org/pdf/2502.18604)  

**Abstract**: What does materials science look like in the "Age of Artificial Intelligence?" Each materials domain-synthesis, characterization, and modeling-has a different answer to this question, motivated by unique challenges and constraints. This work focuses on the tremendous potential of autonomous characterization within electron microscopy. We present our recent advancements in developing domain-aware, multimodal models for microscopy analysis capable of describing complex atomic systems. We then address the critical gap between the theoretical promise of autonomous microscopy and its current practical limitations, showcasing recent successes while highlighting the necessary developments to achieve robust, real-world autonomy. 

**Abstract (ZH)**: 人工智能时代材料科学的面貌：各材料领域（合成、表征和建模）对此问题有不同的答案，受独特挑战和约束的驱动。本工作聚焦自主表征在电子显微镜中的巨大潜力。我们介绍了在显微镜分析中开发领域 Awareness、多模态模型的最新进展，能够描述复杂的原子系统。随后，我们探讨了自主显微镜理论潜力与其当前实践限制之间的关键差距，展示了近期的成功案例，并突出了实现稳健的、实际应用中的自主性所必需的进一步发展。 

---
# Autonomous Vision-Guided Resection of Central Airway Obstruction 

**Title (ZH)**: 自主视觉导向中央气道阻塞切除术 

**Authors**: M. E. Smith, N. Yilmaz, T. Watts, P. M. Scheikl, J. Ge, A. Deguet, A. Kuntz, A. Krieger  

**Link**: [PDF](https://arxiv.org/pdf/2502.18586)  

**Abstract**: Existing tracheal tumor resection methods often lack the precision required for effective airway clearance, and robotic advancements offer new potential for autonomous resection. We present a vision-guided, autonomous approach for palliative resection of tracheal tumors. This system models the tracheal surface with a fifth-degree polynomial to plan tool trajectories, while a custom Faster R-CNN segmentation pipeline identifies the trachea and tumor boundaries. The electrocautery tool angle is optimized using handheld surgical demonstrations, and trajectories are planned to maintain a 1 mm safety clearance from the tracheal surface. We validated the workflow successfully in five consecutive experiments on ex-vivo animal tissue models, successfully clearing the airway obstruction without trachea perforation in all cases (with more than 90% volumetric tumor removal). These results support the feasibility of an autonomous resection platform, paving the way for future developments in minimally-invasive autonomous resection. 

**Abstract (ZH)**: 基于视觉引导的自主气管肿瘤切除方法 

---
# Scalable Best-of-N Selection for Large Language Models via Self-Certainty 

**Title (ZH)**: 大规模语言模型的可扩展最佳选项自信心选择方法 

**Authors**: Zhewei Kang, Xuandong Zhao, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2502.18581)  

**Abstract**: Best-of-N selection is a key technique for improving the reasoning performance of Large Language Models (LLMs) through increased test-time computation. Current state-of-the-art methods often employ computationally intensive reward models for response evaluation and selection. Reward-free alternatives, like self-consistency and universal self-consistency, are limited in their ability to handle open-ended generation tasks or scale effectively. To address these limitations, we propose self-certainty, a novel and efficient metric that leverages the inherent probability distribution of LLM outputs to estimate response quality without requiring external reward models. We hypothesize that higher distributional self-certainty, aggregated across multiple samples, correlates with improved response accuracy, as it reflects greater confidence in the generated output. Through extensive experiments on various reasoning tasks, we demonstrate that self-certainty (1) scales effectively with increasing sample size $N$, akin to reward models but without the computational overhead; (2) complements chain-of-thought, improving reasoning performance beyond greedy decoding; and (3) generalizes to open-ended tasks where traditional self-consistency methods fall short. Our findings establish self-certainty as a practical and efficient way for improving LLM reasoning capabilities. The code is available at this https URL 

**Abstract (ZH)**: 最佳-of-N 选择是通过增加测试时计算量来提高大型语言模型推理性能的关键技术。当前最先进的方法往往采用计算密集型奖励模型来进行响应评估和选择。无奖励替代方法，如自我一致性与普遍自我一致性，处理开放生成任务的能力有限，也很难有效扩展。为了解决这些限制，我们提出了一种新的高效度量标准自我确信，该度量标准利用大型语言模型输出的固有概率分布来估计响应质量，无需外部奖励模型。我们假设，在多个样本中聚合的更高概率分布自我确信与提高的响应准确性相关，因为它反映了更大的生成输出信心。通过在各种推理任务上的广泛实验，我们证明了自我确信:(1) 随样本数量 $N$ 的增加而有效扩展，类似于奖励模型但没有额外的计算开销；(2) 补充逐步推理，超越贪婪解码，进一步提高推理性能；(3) 在传统自我一致性方法表现不佳的开放生成任务中具有良好的泛化能力。我们的发现确立了自我确信作为一种实用且高效的大型语言模型推理能力提升方法的地位。代码可在以下链接获取。 

---
# Differentially Private Iterative Screening Rules for Linear Regression 

**Title (ZH)**: 差分隐私迭代筛选规则for线性回归 

**Authors**: Amol Khanna, Fred Lu, Edward Raff  

**Link**: [PDF](https://arxiv.org/pdf/2502.18578)  

**Abstract**: Linear $L_1$-regularized models have remained one of the simplest and most effective tools in data science. Over the past decade, screening rules have risen in popularity as a way to eliminate features when producing the sparse regression weights of $L_1$ models. However, despite the increasing need of privacy-preserving models for data analysis, to the best of our knowledge, no differentially private screening rule exists. In this paper, we develop the first private screening rule for linear regression. We initially find that this screening rule is too strong: it screens too many coefficients as a result of the private screening step. However, a weakened implementation of private screening reduces overscreening and improves performance. 

**Abstract (ZH)**: 线性$L_1$-正则化模型一直是数据科学中最为简单和有效的工具之一。在过去十年中，筛选规则因其能在生成$L_1$模型的稀疏回归权重时消除特征而日益流行。然而，尽管隐私保护模型在数据分析中的需求越来越大，据我们所知，尚未存在差分隐私筛选规则。在本文中，我们开发了首个适用于线性回归的差分隐私筛选规则。我们发现最初的隐私筛选规则过于严格：由于隐私筛选步骤导致筛选过多的系数。然而，弱化实施的隐私筛选减少了过度筛选并提升了性能。 

---
# FactReasoner: A Probabilistic Approach to Long-Form Factuality Assessment for Large Language Models 

**Title (ZH)**: FactReasoner: 大型语言模型长文本事实性评估的概率方法 

**Authors**: Radu Marinescu, Debarun Bhattacharjya, Junkyu Lee, Tigran Tchrakian, Javier Carnerero Cano, Yufang Hou, Elizabeth Daly, Alessandra Pascale  

**Link**: [PDF](https://arxiv.org/pdf/2502.18573)  

**Abstract**: Large language models (LLMs) have demonstrated vast capabilities on generative tasks in recent years, yet they struggle with guaranteeing the factual correctness of the generated content. This makes these models unreliable in realistic situations where factually accurate responses are expected. In this paper, we propose FactReasoner, a new factuality assessor that relies on probabilistic reasoning to assess the factuality of a long-form generated response. Specifically, FactReasoner decomposes the response into atomic units, retrieves relevant contexts for them from an external knowledge source, and constructs a joint probability distribution over the atoms and contexts using probabilistic encodings of the logical relationships (entailment, contradiction) between the textual utterances corresponding to the atoms and contexts. FactReasoner then computes the posterior probability of whether atomic units in the response are supported by the retrieved contexts. Our experiments on labeled and unlabeled benchmark datasets demonstrate clearly that FactReasoner improves considerably over state-of-the-art prompt-based approaches in terms of both factual precision and recall. 

**Abstract (ZH)**: 大型语言模型（LLMs）在近年来的生成任务中展示了巨大的能力，但它们在保证生成内容的准确性方面存在困难。这使得这些模型在需要事实准确回应的实际情况下不可靠。本文提出了一种新的事实性评估器FactReasoner，该评估器依赖概率推理来评估长文本生成回应的事实性。具体而言，FactReasoner将回应分解为原子单元，从外部知识源检索相关上下文，并使用逻辑关系（蕴含、矛盾）的概率编码构建原子单元和上下文之间的联合概率分布。FactReasoner然后计算所检索上下文支持回应中原子单元的后验概率。我们在标记和未标记的基准数据集上的实验清楚地表明，FactReasoner在事实精确度和召回率方面显著优于最先进的基于提示的方法。 

---
# Application of Attention Mechanism with Bidirectional Long Short-Term Memory (BiLSTM) and CNN for Human Conflict Detection using Computer Vision 

**Title (ZH)**: 基于注意力机制、双方向长短期记忆网络（BiLSTM）和CNN的人体冲突检测在计算机视觉中的应用 

**Authors**: Erick da Silva Farias, Eduardo Palhares Junior  

**Link**: [PDF](https://arxiv.org/pdf/2502.18555)  

**Abstract**: The automatic detection of human conflicts through videos is a crucial area in computer vision, with significant applications in monitoring and public safety policies. However, the scarcity of public datasets and the complexity of human interactions make this task challenging. This study investigates the integration of advanced deep learning techniques, including Attention Mechanism, Convolutional Neural Networks (CNNs), and Bidirectional Long ShortTerm Memory (BiLSTM), to improve the detection of violent behaviors in videos. The research explores how the use of the attention mechanism can help focus on the most relevant parts of the video, enhancing the accuracy and robustness of the model. The experiments indicate that the combination of CNNs with BiLSTM and the attention mechanism provides a promising solution for conflict monitoring, offering insights into the effectiveness of different strategies. This work opens new possibilities for the development of automated surveillance systems that can operate more efficiently in real-time detection of violent events. 

**Abstract (ZH)**: 通过视频自动检测人类冲突是计算机视觉中的一个重要领域，具有在监控和公共安全政策中的广泛应用。然而，公共数据集的稀缺性和人类互动的复杂性使得这一任务具有挑战性。本研究探讨了结合注意力机制、卷积神经网络（CNNs）和双向长短期记忆（BiLSTM）等高级深度学习技术的方法，以提高视频中暴力行为检测的性能。研究探讨了注意力机制在帮助聚焦视频中最相关部分方面的应用，从而提高模型的准确性和健壮性。实验表明，将CNNs与BiLSTM结合使用并加入注意力机制提供了一种有前景的冲突监测解决方案，揭示了不同策略的有效性。本研究为开发更有效地进行实时暴力事件检测的自动化监控系统开辟了新的可能性。 

---
# Applications of Statistical Field Theory in Deep Learning 

**Title (ZH)**: 统计场论在深度学习中的应用 

**Authors**: Zohar Ringel, Noa Rubin, Edo Mor, Moritz Helias, Inbar Seroussi  

**Link**: [PDF](https://arxiv.org/pdf/2502.18553)  

**Abstract**: Deep learning algorithms have made incredible strides in the past decade yet due to the complexity of these algorithms, the science of deep learning remains in its early stages. Being an experimentally driven field, it is natural to seek a theory of deep learning within the physics paradigm. As deep learning is largely about learning functions and distributions over functions, statistical field theory, a rich and versatile toolbox for tackling complex distributions over functions (fields) is an obvious choice of formalism. Research efforts carried out in the past few years have demonstrated the ability of field theory to provide useful insights on generalization, implicit bias, and feature learning effects. Here we provide a pedagogical review of this emerging line of research. 

**Abstract (ZH)**: 深度学习算法在过去十年中取得了惊人的进展，但由于这些算法的复杂性，深度学习科学仍处于初级阶段。作为以实验为主导的领域，自然会寻求一种基于物理范式的深度学习理论。由于深度学习主要涉及学习函数和函数上的分布，统计场理论作为一种处理复杂函数分布的强大而多功能工具箱，显然是合适的形式化方法。近年来进行的研究工作已证明场理论在提供关于泛化、隐式偏置和特征学习效应的有用见解方面的能力。在这里，我们提供了一种教学性的回顾，介绍这条新兴的研究路线。 

---
# What is the Alignment Objective of GRPO? 

**Title (ZH)**: GRPO的对齐目标是什么？ 

**Authors**: Milan Vojnovic, Se-Young Yun  

**Link**: [PDF](https://arxiv.org/pdf/2502.18548)  

**Abstract**: In this note, we examine the aggregation of preferences achieved by the Group Policy Optimisation (GRPO) algorithm, a reinforcement learning method used to train advanced artificial intelligence models such as DeepSeek-R1-Zero and DeepSeekMath. The GRPO algorithm trains a policy using a reward preference model, which is computed by sampling a set of outputs for a given context, observing the corresponding rewards, and applying shift-and-scale normalisation to these reward values. Additionally, it incorporates a penalty function to discourage deviations from a reference policy.
We present a framework that enables us to characterise the stationary policies of the GRPO algorithm. This analysis reveals that the aggregation of preferences differs fundamentally from standard logarithmic pooling, which is implemented by other approaches such as RLHF. The precise form of preference aggregation arises from the way the reward preference model is defined and from the penalty function, which we show to essentially correspond to the reverse Kullback-Leibler (KL) divergence between the aggregation policy and the reference policy.
Interestingly, we demonstrate that for groups of size two, the reward preference model corresponds to pairwise comparison preferences, similar to those in other alignment methods based on pairwise comparison feedback. We provide explicit characterisations of the aggregate preference for binary questions, for groups of size two, and in the limit of large group size. This provides insights into the dependence of the aggregate preference on parameters such as the regularisation constant and the confidence margin of question answers.
Finally, we discuss the aggregation of preferences obtained by modifying the GRPO algorithm to use direct KL divergence as the penalty or to use rewards without scale normalisation. 

**Abstract (ZH)**: 关于Group Policy Optimisation算法偏好聚合的分析：增强学习方法在高级人工智能模型训练中的应用 

---
# Steganography Beyond Space-Time With Chain of Multimodal AI Agents 

**Title (ZH)**: 时空之外的隐写术：多模态AI代理链 

**Authors**: Ching-Chun Chang, Isao Echizen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18547)  

**Abstract**: Steganography is the art and science of covert writing, with a broad range of applications interwoven within the realm of cybersecurity. As artificial intelligence continues to evolve, its ability to synthesise realistic content emerges as a threat in the hands of cybercriminals who seek to manipulate and misrepresent the truth. Such synthetic content introduces a non-trivial risk of overwriting the subtle changes made for the purpose of steganography. When the signals in both the spatial and temporal domains are vulnerable to unforeseen overwriting, it calls for reflection on what can remain invariant after all. This study proposes a paradigm in steganography for audiovisual media, where messages are concealed beyond both spatial and temporal domains. A chain of multimodal agents is developed to deconstruct audiovisual content into a cover text, embed a message within the linguistic domain, and then reconstruct the audiovisual content through synchronising both aural and visual modalities with the resultant stego text. The message is encoded by biasing the word sampling process of a language generation model and decoded by analysing the probability distribution of word choices. The accuracy of message transmission is evaluated under both zero-bit and multi-bit capacity settings. Fidelity is assessed through both biometric and semantic similarities, capturing the identities of the recorded face and voice, as well as the core ideas conveyed through the media. Secrecy is examined through statistical comparisons between cover and stego texts. Robustness is tested across various scenarios, including audiovisual compression, face-swapping, voice-cloning and their combinations. 

**Abstract (ZH)**: 视听媒体中的隐写图学：超越时空域的消息隐藏与传输 

---
# PII-Bench: Evaluating Query-Aware Privacy Protection Systems 

**Title (ZH)**: PII-Bench: 评估查询感知隐私保护系统 

**Authors**: Hao Shen, Zhouhong Gu, Haokai Hong, Weili Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.18545)  

**Abstract**: The widespread adoption of Large Language Models (LLMs) has raised significant privacy concerns regarding the exposure of personally identifiable information (PII) in user prompts. To address this challenge, we propose a query-unrelated PII masking strategy and introduce PII-Bench, the first comprehensive evaluation framework for assessing privacy protection systems. PII-Bench comprises 2,842 test samples across 55 fine-grained PII categories, featuring diverse scenarios from single-subject descriptions to complex multi-party interactions. Each sample is carefully crafted with a user query, context description, and standard answer indicating query-relevant PII. Our empirical evaluation reveals that while current models perform adequately in basic PII detection, they show significant limitations in determining PII query relevance. Even state-of-the-art LLMs struggle with this task, particularly in handling complex multi-subject scenarios, indicating substantial room for improvement in achieving intelligent PII masking. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的广泛应用引发了关于用户提示中个人可识别信息（PII）暴露的重大隐私 concern 的担忧。为应对这一挑战，我们提出了一种与查询无关的 PII 遮掩策略，并引入了第一个全面评估框架 PII-Bench，用于评估隐私保护系统。PII-Bench 包含 2,842 个测试样本，涵盖 55 个细粒度的 PII 类别，从单主体描述到复杂的多主体交互，各不相同。每个样本都包含用户查询、上下文描述和标准答案，其中标准答案指出了与查询相关的 PII。我们的实证评估表明，当前模型在基本 PII 检测方面表现良好，但在确定查询相关 PII 方面存在显著局限性。即使最新的 LLM 也在这项任务中遇到困难，尤其是在处理复杂多主体场景方面，这表明在实现智能 PII 遮(masking)方面存在着巨大的改进空间。 

---
# MA-GTS: A Multi-Agent Framework for Solving Complex Graph Problems in Real-World Applications 

**Title (ZH)**: MA-GTS: 一种解决实际应用中复杂图问题的多智能体框架 

**Authors**: Zike Yuan, Ming Liu, Hui Wang, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2502.18540)  

**Abstract**: Graph-theoretic problems arise in real-world applications like logistics, communication networks, and traffic optimization. These problems are often complex, noisy, and irregular, posing challenges for traditional algorithms. Large language models (LLMs) offer potential solutions but face challenges, including limited accuracy and input length constraints. To address these challenges, we propose MA-GTS (Multi-Agent Graph Theory Solver), a multi-agent framework that decomposes these complex problems through agent collaboration. MA-GTS maps the implicitly expressed text-based graph data into clear, structured graph representations and dynamically selects the most suitable algorithm based on problem constraints and graph structure scale. This approach ensures that the solution process remains efficient and the resulting reasoning path is interpretable. We validate MA-GTS using the G-REAL dataset, a real-world-inspired graph theory dataset we created. Experimental results show that MA-GTS outperforms state-of-the-art approaches in terms of efficiency, accuracy, and scalability, with strong results across multiple benchmarks (G-REAL 94.2%, GraCoRe 96.9%, NLGraph 98.4%).MA-GTS is open-sourced at this https URL. 

**Abstract (ZH)**: 图论问题在物流、通信网络和交通优化等领域中存在。这些问题往往复杂、噪声大且不规则，给传统算法带来了挑战。大型语言模型（LLMs）提供了潜在的解决方案，但面临准确性有限和输入长度限制等挑战。为了解决这些挑战，我们提出了MA-GTS（多代理图理论求解器）多代理框架，通过代理协作分解这些复杂问题。MA-GTS将隐式表示的文字型图数据映射为清晰、结构化的图表示，并根据问题约束和图结构规模动态选择最合适的算法。这种方法确保了解决过程保持高效，推理路径可解释。我们使用我们创建的G-REAL数据集对MA-GTS进行了验证，这是一个基于现实世界的图理论数据集。实验结果表明，MA-GTS在效率、准确性和可扩展性方面均优于现有方法，在多个基准测试中表现出色（G-REAL 94.2%，GraCoRe 96.9%，NLGraph 98.4%）。MA-GTS已开源，地址为：这个链接。 

---
# Revisiting Convolution Architecture in the Realm of DNA Foundation Models 

**Title (ZH)**: 在DNA基础模型领域的卷积架构再探讨 

**Authors**: Yu Bo, Weian Mao, Yanjun Shao, Weiqiang Bai, Peng Ye, Xinzhu Ma, Junbo Zhao, Hao Chen, Chunhua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18538)  

**Abstract**: In recent years, a variety of methods based on Transformer and state space model (SSM) architectures have been proposed, advancing foundational DNA language models. However, there is a lack of comparison between these recent approaches and the classical architecture convolutional networks (CNNs) on foundation model benchmarks. This raises the question: are CNNs truly being surpassed by these recent approaches based on transformer and SSM architectures? In this paper, we develop a simple but well-designed CNN-based method termed ConvNova. ConvNova identifies and proposes three effective designs: 1) dilated convolutions, 2) gated convolutions, and 3) a dual-branch framework for gating mechanisms. Through extensive empirical experiments, we demonstrate that ConvNova significantly outperforms recent methods on more than half of the tasks across several foundation model benchmarks. For example, in histone-related tasks, ConvNova exceeds the second-best method by an average of 5.8%, while generally utilizing fewer parameters and enabling faster computation. In addition, the experiments observed findings that may be related to biological characteristics. This indicates that CNNs are still a strong competitor compared to Transformers and SSMs. We anticipate that this work will spark renewed interest in CNN-based methods for DNA foundation models. 

**Abstract (ZH)**: 近年来，基于Transformer和状态空间模型（SSM）架构的各种方法被提出，推动了基础DNA语言模型的发展。然而，在基础模型基准测试中，这些最近的方法与经典的卷积网络（CNNs）架构之间缺乏比较。这引发了一个问题：基于Transformer和SSM架构的最近方法是否真正超越了CNNs？在本文中，我们提出了一种简单而设计良好的基于CNN的方法，称为ConvNova。ConvNova提出了三种有效的设计：1) 扩张卷积，2) 门控卷积，以及3) 门控机制的双分支框架。通过广泛的实证实验，我们证明ConvNova在多个基础模型基准测试中的多项任务上显著优于最近的方法。例如，在组蛋白相关任务中，ConvNova平均比第二优方法高出5.8%，同时参数更少，计算更快。此外，实验还观察到了可能与生物学特性相关的发现，这表明CNNs仍然是Transformer和SSM的强有力竞争对手。我们期望这项工作能重新激发对基于CNN的方法在DNA基础模型中的兴趣。 

---
# A Survey of Zero-Knowledge Proof Based Verifiable Machine Learning 

**Title (ZH)**: 基于零知识证明的可验证机器学习综述 

**Authors**: Zhizhi Peng, Taotao Wang, Chonghe Zhao, Guofu Liao, Zibin Lin, Yifeng Liu, Bin Cao, Long Shi, Qing Yang, Shengli Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18535)  

**Abstract**: As machine learning technologies advance rapidly across various domains, concerns over data privacy and model security have grown significantly. These challenges are particularly pronounced when models are trained and deployed on cloud platforms or third-party servers due to the computational resource limitations of users' end devices. In response, zero-knowledge proof (ZKP) technology has emerged as a promising solution, enabling effective validation of model performance and authenticity in both training and inference processes without disclosing sensitive data. Thus, ZKP ensures the verifiability and security of machine learning models, making it a valuable tool for privacy-preserving AI. Although some research has explored the verifiable machine learning solutions that exploit ZKP, a comprehensive survey and summary of these efforts remain absent. This survey paper aims to bridge this gap by reviewing and analyzing all the existing Zero-Knowledge Machine Learning (ZKML) research from June 2017 to December 2024. We begin by introducing the concept of ZKML and outlining its ZKP algorithmic setups under three key categories: verifiable training, verifiable inference, and verifiable testing. Next, we provide a comprehensive categorization of existing ZKML research within these categories and analyze the works in detail. Furthermore, we explore the implementation challenges faced in this field and discuss the improvement works to address these obstacles. Additionally, we highlight several commercial applications of ZKML technology. Finally, we propose promising directions for future advancements in this domain. 

**Abstract (ZH)**: 随着机器学习技术在各个领域迅速发展，数据隐私和模型安全的担忧日益增加。特别是在云平台或第三方服务器上训练和部署模型时，由于用户终端设备的计算资源限制，这些挑战尤为突出。为应对这些挑战，零知识证明（ZKP）技术 emerged 作为一种有前途的解决方案，能够在不泄露敏感数据的情况下，有效地验证模型性能和真实性，从而确保机器学习模型的可验证性和安全，使其成为隐私保护人工智能的重要工具。尽管已有研究探索了利用 ZKP 的可验证机器学习解决方案，但这些努力的全面综述和总结仍然缺乏。本文旨在填补这一空白，通过回顾和分析从2017年6月到2024年12月的所有现有零知识机器学习（ZKML）研究，进行综述和分析。首先，我们介绍了ZKML的概念，并按三大类——可验证训练、可验证推理和可验证测试——概述其ZKP算法设置。接着，我们对这些类别内的现有ZKML研究进行全面分类，并详细分析这些工作。此外，我们探讨了该领域实施挑战，并讨论了应对这些障碍的方法。我们还强调了几种ZKML技术的商业应用，并提出了该领域未来发展的一些有前途的方向。机器学习中的零知识证明技术：2017年6月至2024年12月的综述 

---
# MAFE: Multi-Agent Fair Environments for Decision-Making Systems 

**Title (ZH)**: 多Agent公平环境：决策系统中的应用 

**Authors**: Zachary McBride Lazri, Anirudh Nakra, Ivan Brugere, Danial Dervovic, Antigoni Polychroniadou, Furong Huang, Dana Dachman-Soled, Min Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18534)  

**Abstract**: Fairness constraints applied to machine learning (ML) models in static contexts have been shown to potentially produce adverse outcomes among demographic groups over time. To address this issue, emerging research focuses on creating fair solutions that persist over time. While many approaches treat this as a single-agent decision-making problem, real-world systems often consist of multiple interacting entities that influence outcomes. Explicitly modeling these entities as agents enables more flexible analysis of their interventions and the effects they have on a system's underlying dynamics. A significant challenge in conducting research on multi-agent systems is the lack of realistic environments that leverage the limited real-world data available for analysis. To address this gap, we introduce the concept of a Multi-Agent Fair Environment (MAFE) and present and analyze three MAFEs that model distinct social systems. Experimental results demonstrate the utility of our MAFEs as testbeds for developing multi-agent fair algorithms. 

**Abstract (ZH)**: 应用于静态上下文的公平性约束可能在时间上对不同的人口群体产生不利影响。为了应对这一问题，新兴研究侧重于开发持久公平的解决方案。尽管许多方法将此视为单一代理决策问题，但现实世界系统通常由多个相互作用的实体组成，这些实体影响结果。将这些实体显式建模为代理有助于更灵活地分析其干预措施及其对系统潜在动态的影响。在多代理系统研究中的一大挑战是缺乏利用可用于分析的有限现实世界数据的现实环境。为解决这一问题，我们引入了多代理公平环境（MAFE）的概念，并提出了并分析了三种建模不同社会系统的MAFE。实验结果表明，我们的MAFE适合作为开发多代理公平算法的测试平台。 

---
# Heterogeneous Decision Making in Mixed Traffic: Uncertainty-aware Planning and Bounded Rationality 

**Title (ZH)**: 混合交通中的异质性决策：不确定性aware规划与有限理性 

**Authors**: Hang Wang, Qiaoyi Fang, Junshan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18529)  

**Abstract**: The past few years have witnessed a rapid growth of the deployment of automated vehicles (AVs). Clearly, AVs and human-driven vehicles (HVs) will co-exist for many years, and AVs will have to operate around HVs, pedestrians, cyclists, and more, calling for fundamental breakthroughs in AI designed for mixed traffic to achieve mixed autonomy. Thus motivated, we study heterogeneous decision making by AVs and HVs in a mixed traffic environment, aiming to capture the interactions between human and machine decision-making and develop an AI foundation that enables vehicles to operate safely and efficiently. There are a number of challenges to achieve mixed autonomy, including 1) humans drivers make driving decisions with bounded rationality, and it remains open to develop accurate models for HVs' decision making; and 2) uncertainty-aware planning plays a critical role for AVs to take safety maneuvers in response to the human behavior. In this paper, we introduce a formulation of AV-HV interaction, where the HV makes decisions with bounded rationality and the AV employs uncertainty-aware planning based on the prediction on HV's future actions. We conduct a comprehensive analysis on AV and HV's learning regret to answer the questions: 1) {How does the learning performance depend on HV's bounded rationality and AV's planning}; 2) {How do different decision making strategies impact the overall learning performance}? Our findings reveal some intriguing phenomena, such as Goodhart's Law in AV's learning performance and compounding effects in HV's decision making process. By examining the dynamics of the regrets, we gain insights into the interplay between human and machine decision making. 

**Abstract (ZH)**: 过去的几年见证了自动驾驶车辆（AVs）部署的迅速增长。显然，AVs和人类驾驶车辆（HVs）将长期共存，AVs将不得不在复杂交通环境中与HVs、行人和骑行者等互动，这需要在混合交通环境中实现混合自动驾驶方面取得根本性的突破。受此驱动，我们研究了在混合交通环境中AVs和HVs异质决策机制，旨在捕捉人机决策之间的相互作用，并建立使车辆能够安全、高效运行的AI基础。实现混合自动驾驶面临着许多挑战，包括1) 人类驾驶员的决策具有有限理性，目前尚无法开发出准确的HVs决策模型；2) 以不确定性为导向的规划对于AVs应对人类行为并采取安全措施至关重要。在本文中，我们提出了一个AV-HV交互的建模框架，其中HVs的决策具有有限理性，而AVs则基于对HVs未来行为的预测采用不确定性导向的规划策略。我们全面分析了AVs和HVs的学习遗憾，以回答以下问题：1) 学习性能如何取决于HVs的有限理性以及AVs的规划；2) 不同决策策略对整体学习性能有何影响？我们的发现揭示了一些有趣的现象，如自动驾驶车辆学习性能的Goodhart定律以及人类驱动决策过程中的累积效应。通过研究遗憾的动态变化，我们深入了解了人机决策之间的相互作用。 

---
# ARACNE: An LLM-Based Autonomous Shell Pentesting Agent 

**Title (ZH)**: ARACNE：基于LLM的自主Shell渗透测试代理 

**Authors**: Tomas Nieponice, Veronica Valeros, Sebastian Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2502.18528)  

**Abstract**: We introduce ARACNE, a fully autonomous LLM-based pentesting agent tailored for SSH services that can execute commands on real Linux shell systems. Introduces a new agent architecture with multi-LLM model support. Experiments show that ARACNE can reach a 60\% success rate against the autonomous defender ShelLM and a 57.58\% success rate against the Over The Wire Bandit CTF challenges, improving over the state-of-the-art. When winning, the average number of actions taken by the agent to accomplish the goals was less than 5. The results show that the use of multi-LLM is a promising approach to increase accuracy in the actions. 

**Abstract (ZH)**: 我们介绍ARACNE：一种针对SSH服务的自主LLM基渗透测试代理，能够执行真实LinuxShell系统的命令。该代理架构支持多LLM模型，并实验证明，ARACNE在对抗自主防守方ShelLM时成功率达到60%，在对抗Over The Wire Bandit CTF挑战时成功率达到57.58%，超越了当前最优方法。当获胜时，代理完成目标所需的平均行动次数少于5次。结果表明，使用多LLM是提高行动准确性的有前途的方法。 

---
# GOD model: Privacy Preserved AI School for Personal Assistant 

**Title (ZH)**: GOD模型：保护隐私的人工智能个人助手学校 

**Authors**: PIN AI Team, Bill Qingyun Sun, Laura Florescu, Boliang Zhang, Regan Peng, Smile Hu, Shouqiao Wang, Ben Wu, Xi Wang, Davide Crapis, Gavin Zhen Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.18527)  

**Abstract**: Personal AI assistants (e.g., Apple Intelligence, Meta AI) offer proactive recommendations that simplify everyday tasks, but their reliance on sensitive user data raises concerns about privacy and trust. To address these challenges, we introduce the Guardian of Data (GOD), a secure, privacy-preserving framework for training and evaluating AI assistants directly on-device. Unlike traditional benchmarks, the GOD model measures how well assistants can anticipate user needs-such as suggesting gifts-while protecting user data and autonomy. Functioning like an AI school, it addresses the cold start problem by simulating user queries and employing a curriculum-based approach to refine the performance of each assistant. Running within a Trusted Execution Environment (TEE), it safeguards user data while applying reinforcement and imitation learning to refine AI recommendations. A token-based incentive system encourages users to share data securely, creating a data flywheel that drives continuous improvement. By integrating privacy, personalization, and trust, the GOD model provides a scalable, responsible path for advancing personal AI assistants. For community collaboration, part of the framework is open-sourced at this https URL. 

**Abstract (ZH)**: 个性化AI助手（例如Apple Intelligence、Meta AI）通过提供主动推荐来简化日常任务，但它们对敏感用户数据的依赖引发了隐私和信任方面的担忧。为应对这些挑战，我们提出了数据守护者（GOD）框架，这是一种安全的隐私保护框架，用于在设备本地训练和评估AI助手。与传统基准不同，GOD模型衡量助手预测用户需求（如建议礼物）的能力，同时保护用户数据和自主权。该模型类似于AI学校，通过模拟用户查询和采用基于课程的学习方法来解决冷启动问题，从而优化每个助手的性能。GOD模型在可信执行环境中运行，以保护用户数据并应用强化学习和模仿学习来改进AI推荐。基于代币的激励系统鼓励用户安全地共享数据，从而形成一个数据飞轮，推动持续改进。通过整合隐私、个性化和信任，GOD模型为推进个性化AI助手提供了可扩展且负责任的途径。部分框架已在此处开放合作：https://github.com/alibaba/Qwen-GOD。 

---
# Reinforcement Learning-based Approach for Vehicle-to-Building Charging with Heterogeneous Agents and Long Term Rewards 

**Title (ZH)**: 基于强化学习的方法：异构代理和长期奖励条件下的车辆到建筑充电 

**Authors**: Fangqi Liu, Rishav Sen, Jose Paolo Talusan, Ava Pettet, Aaron Kandel, Yoshinori Suzue, Ayan Mukhopadhyay, Abhishek Dubey  

**Link**: [PDF](https://arxiv.org/pdf/2502.18526)  

**Abstract**: Strategic aggregation of electric vehicle batteries as energy reservoirs can optimize power grid demand, benefiting smart and connected communities, especially large office buildings that offer workplace charging. This involves optimizing charging and discharging to reduce peak energy costs and net peak demand, monitored over extended periods (e.g., a month), which involves making sequential decisions under uncertainty and delayed and sparse rewards, a continuous action space, and the complexity of ensuring generalization across diverse conditions. Existing algorithmic approaches, e.g., heuristic-based strategies, fall short in addressing real-time decision-making under dynamic conditions, and traditional reinforcement learning (RL) models struggle with large state-action spaces, multi-agent settings, and the need for long-term reward optimization. To address these challenges, we introduce a novel RL framework that combines the Deep Deterministic Policy Gradient approach (DDPG) with action masking and efficient MILP-driven policy guidance. Our approach balances the exploration of continuous action spaces to meet user charging demands. Using real-world data from a major electric vehicle manufacturer, we show that our approach comprehensively outperforms many well-established baselines and several scalable heuristic approaches, achieving significant cost savings while meeting all charging requirements. Our results show that the proposed approach is one of the first scalable and general approaches to solving the V2B energy management challenge. 

**Abstract (ZH)**: 作为能量储存资源的电动车辆电池的战略聚合可优化电力需求，惠及智能互联社区，尤其是提供工作场所充电的大型办公楼。这涉及在长时间段（例如一个月）内优化充电和放电以降低峰值能源成本和净峰值需求，在不确定性条件下做出顺序决策，并处理延迟和稀疏奖励、连续的动作空间以及在各种条件下确保推广性的复杂性。现有的算法方法，例如基于启发式的策略，在应对动态条件下的实时决策时存在不足，传统的强化学习（RL）模型难以处理大型状态-动作空间、多智能体环境以及长期奖励优化的需求。为应对这些挑战，我们提出了一种新颖的RL框架，结合了深度确定性策略梯度方法（DDPG）、动作掩蔽以及高效的基于MILP的策略指导。我们的方法平衡了对连续动作空间的探索以满足用户充电需求。使用一家主要电动车辆制造商的实时数据，我们展示了我们的方法全面优于许多成熟的基线方法和几种可扩展的启发式方法，在满足所有充电需求的同时实现了显著的成本节约。我们的结果表明，所提出的方法是解决V2B能源管理挑战的第一个可扩展且通用的方法之一。 

---
# End-to-End Deep Learning for Structural Brain Imaging: A Unified Framework 

**Title (ZH)**: 端到端深度学习在结构脑成像中的统一框架 

**Authors**: Yao Su, Keqi Han, Mingjie Zeng, Lichao Sun, Liang Zhan, Carl Yang, Lifang He, Xiangnan Kong  

**Link**: [PDF](https://arxiv.org/pdf/2502.18523)  

**Abstract**: Brain imaging analysis is fundamental in neuroscience, providing valuable insights into brain structure and function. Traditional workflows follow a sequential pipeline-brain extraction, registration, segmentation, parcellation, network generation, and classification-treating each step as an independent task. These methods rely heavily on task-specific training data and expert intervention to correct intermediate errors, making them particularly burdensome for high-dimensional neuroimaging data, where annotations and quality control are costly and time-consuming. We introduce UniBrain, a unified end-to-end framework that integrates all processing steps into a single optimization process, allowing tasks to interact and refine each other. Unlike traditional approaches that require extensive task-specific annotations, UniBrain operates with minimal supervision, leveraging only low-cost labels (i.e., classification and extraction) and a single labeled atlas. By jointly optimizing extraction, registration, segmentation, parcellation, network generation, and classification, UniBrain enhances both accuracy and computational efficiency while significantly reducing annotation effort. Experimental results demonstrate its superiority over existing methods across multiple tasks, offering a more scalable and reliable solution for neuroimaging analysis. Our code and data can be found at this https URL 

**Abstract (ZH)**: 脑成像分析是神经科学的基础，提供了对脑结构和功能的宝贵见解。传统的 workflows 采用顺序管线流程——脑提取、注册、分割、分区、网络生成和分类——将每个步骤视为独立任务。这些方法严重依赖于特定任务的训练数据和专家干预来纠正中间错误，使得在高维度神经成像数据中尤为负担沉重，因为注释和质量控制成本高昂且耗时。我们引入了 UniBrain，这是一种统一的端到端框架，将所有处理步骤集成到单个优化过程中，使任务之间可以相互作用和相互精炼。与传统需要大量特定任务注释的方法不同，UniBrain 只需少量监督，利用低成本标签（即分类和提取）和一个标注图谱。通过联合优化提取、注册、分割、分区、网络生成和分类，UniBrain 提高了准确性和计算效率，同时显著减少了注释工作量。实验结果表明，UniBrain 在多个任务上优于现有方法，提供了一种更具扩展性和可靠性的神经成像分析解决方案。我们的代码和数据可在此处找到：this https URL。 

---
# Class-Conditional Neural Polarizer: A Lightweight and Effective Backdoor Defense by Purifying Poisoned Features 

**Title (ZH)**: 面向类别的神经极化器：一种通过净化中毒特征实现的轻量级有效后门防御方法 

**Authors**: Mingli Zhu, Shaokui Wei, Hongyuan Zha, Baoyuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18520)  

**Abstract**: Recent studies have highlighted the vulnerability of deep neural networks to backdoor attacks, where models are manipulated to rely on embedded triggers within poisoned samples, despite the presence of both benign and trigger information. While several defense methods have been proposed, they often struggle to balance backdoor mitigation with maintaining benign this http URL this work, inspired by the concept of optical polarizer-which allows light waves of specific polarizations to pass while filtering others-we propose a lightweight backdoor defense approach, NPD. This method integrates a neural polarizer (NP) as an intermediate layer within the compromised model, implemented as a lightweight linear transformation optimized via bi-level optimization. The learnable NP filters trigger information from poisoned samples while preserving benign content. Despite its effectiveness, we identify through empirical studies that NPD's performance degrades when the target labels (required for purification) are inaccurately estimated. To address this limitation while harnessing the potential of targeted adversarial mitigation, we propose class-conditional neural polarizer-based defense (CNPD). The key innovation is a fusion module that integrates the backdoored model's predicted label with the features to be purified. This architecture inherently mimics targeted adversarial defense mechanisms without requiring label estimation used in NPD. We propose three implementations of CNPD: the first is r-CNPD, which trains a replicated NP layer for each class and, during inference, selects the appropriate NP layer for defense based on the predicted class from the backdoored model. To efficiently handle a large number of classes, two variants are designed: e-CNPD, which embeds class information as additional features, and a-CNPD, which directs network attention using class information. 

**Abstract (ZH)**: 最近的研究强调了深度神经网络对后门攻击的脆弱性，这些攻击通过在受污染样本中嵌入触发器来操纵模型，使其依赖于这些触发器，即使存在无害和触发器信息。尽管已经提出了一些防御方法，但它们往往难以在减少后门攻击影响与保持无害性能之间取得平衡。受光学偏振器概念的启发，本工作提出了一种轻量级后门防御方法NPD。该方法在受损模型中引入了一个神经偏振器(NP)作为中间层，作为通过优化二阶优化实现的轻量级线性变换。可学习的NP筛选受污染样本中的触发器信息，同时保留无害内容。尽管NPD非常有效，但我们通过实验证实，当目标标签（用于净化）估计不准确时，NPD的性能会下降。为了解决这一限制，同时利用定向对抗防御的潜力，我们提出了基于类别条件神经偏振器的防御方法（CNPD）。关键创新是一种融合模块，将受污染模型预测的标签与待净化的特征结合在一起，从而在不需要NPD中使用的标签估计的情况下，内在地模拟了定向对抗防御机制。我们提出了CNPD的三种实现：首先是r-CNPD，它为每个类别训练一个复制的NP层，在推断过程中根据受污染模型预测的类别选择合适的NP层进行防御。为了高效处理大量类别，我们设计了两种变体：e-CNPD，它嵌入类别信息作为附加特征；a-CNPD，它使用类别信息引导网络注意力。 

---
# FreeTumor: Large-Scale Generative Tumor Synthesis in Computed Tomography Images for Improving Tumor Recognition 

**Title (ZH)**: FreeTumor: 在计算机断层扫描图像中大规模生成肿瘤合成以提高肿瘤识别 

**Authors**: Linshan Wu, Jiaxin Zhuang, Yanning Zhou, Sunan He, Jiabo Ma, Luyang Luo, Xi Wang, Xuefeng Ni, Xiaoling Zhong, Mingxiang Wu, Yinghua Zhao, Xiaohui Duan, Varut Vardhanabhuti, Pranav Rajpurkar, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18519)  

**Abstract**: Tumor is a leading cause of death worldwide, with an estimated 10 million deaths attributed to tumor-related diseases every year. AI-driven tumor recognition unlocks new possibilities for more precise and intelligent tumor screening and diagnosis. However, the progress is heavily hampered by the scarcity of annotated datasets, which demands extensive annotation efforts by radiologists. To tackle this challenge, we introduce FreeTumor, an innovative Generative AI (GAI) framework to enable large-scale tumor synthesis for mitigating data scarcity. Specifically, FreeTumor effectively leverages a combination of limited labeled data and large-scale unlabeled data for tumor synthesis training. Unleashing the power of large-scale data, FreeTumor is capable of synthesizing a large number of realistic tumors on images for augmenting training datasets. To this end, we create the largest training dataset for tumor synthesis and recognition by curating 161,310 publicly available Computed Tomography (CT) volumes from 33 sources, with only 2.3% containing annotated tumors. To validate the fidelity of synthetic tumors, we engaged 13 board-certified radiologists in a Visual Turing Test to discern between synthetic and real tumors. Rigorous clinician evaluation validates the high quality of our synthetic tumors, as they achieved only 51.1% sensitivity and 60.8% accuracy in distinguishing our synthetic tumors from real ones. Through high-quality tumor synthesis, FreeTumor scales up the recognition training datasets by over 40 times, showcasing a notable superiority over state-of-the-art AI methods including various synthesis methods and foundation models. These findings indicate promising prospects of FreeTumor in clinical applications, potentially advancing tumor treatments and improving the survival rates of patients. 

**Abstract (ZH)**: 肿瘤是全球主要的死亡原因，每年约有1000万人死于肿瘤相关疾病。基于AI的肿瘤识别技术开启了更为精准和智能的肿瘤筛查与诊断的新可能。然而，进展受到标注数据稀缺性的严重制约，这要求放射学家付出大量的标注努力。为应对这一挑战，我们引入了FreeTumor，一种创新的生成AI（GAI）框架，以缓解数据稀缺性问题。具体而言，FreeTumor有效地利用有限的标注数据和大量的未标注数据进行肿瘤合成训练。通过大规模数据的强大功能，FreeTumor能够在影像中合成大量逼真的肿瘤，以扩充训练数据集。为此，我们创建了最大的肿瘤合成与识别训练数据集，从33个来源收集了161,310个公开的计算机断层扫描（CT）体积，其中仅有2.3%包含标注的肿瘤。为了验证合成肿瘤的真实性，我们邀请了13名经过认证的放射学家进行视觉图灵测试，以区分合成肿瘤和真实肿瘤。严格的临床评估验证了我们合成肿瘤的高质量，它们仅在区分合成肿瘤和真实肿瘤方面达到了51.1%的敏感性和60.8%的准确性。凭借高质量的肿瘤合成，FreeTumor将识别训练数据集扩展了40多倍，展示了在其与各种合成方法和基础模型相比的显著优势。这些发现表明，FreeTumor在临床应用中拥有广阔的前景，可能促进肿瘤治疗并提高患者的生存率。 

---
# Swallowing the Poison Pills: Insights from Vulnerability Disparity Among LLMs 

**Title (ZH)**: 吞咽毒丸：来自语言模型漏洞差异的洞察 

**Authors**: Peng Yifeng, Wu Zhizheng, Chen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18518)  

**Abstract**: Modern large language models (LLMs) exhibit critical vulnerabilities to poison pill attacks: localized data poisoning that alters specific factual knowledge while preserving overall model utility. We systematically demonstrate these attacks exploit inherent architectural properties of LLMs, achieving 54.6% increased retrieval inaccuracy on long-tail knowledge versus dominant topics and up to 25.5% increase retrieval inaccuracy on compressed models versus original architectures. Through controlled mutations (e.g., temporal/spatial/entity alterations) and, our method induces localized memorization deterioration with negligible impact on models' performance on regular standard benchmarks (e.g., <2% performance drop on MMLU/GPQA), leading to potential detection evasion. Our findings suggest: (1) Disproportionate vulnerability in long-tail knowledge may result from reduced parameter redundancy; (2) Model compression may increase attack surfaces, with pruned/distilled models requiring 30% fewer poison samples for equivalent damage; (3) Associative memory enables both spread of collateral damage to related concepts and amplification of damage from simultaneous attack, particularly for dominant topics. These findings raise concerns over current scaling paradigms since attack costs are lowering while defense complexity is rising. Our work establishes poison pills as both a security threat and diagnostic tool, revealing critical security-efficiency trade-offs in language model compression that challenges prevailing safety assumptions. 

**Abstract (ZH)**: 现代大型语言模型（LLMs）对毒药攻击表现出关键性漏洞：局部数据污染改变了特定事实知识的同时保留了模型的整体实用性。 

---
# RewardDS: Privacy-Preserving Fine-Tuning for Large Language Models via Reward Driven Data Synthesis 

**Title (ZH)**: RewardDS：通过奖励驱动的数据合成实现大型语言模型的隐私保护微调 

**Authors**: Jianwei Wang, Junyao Yang, Haoran Li, Huiping Zhuang, Cen Chen, Ziqian Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2502.18517)  

**Abstract**: The success of large language models (LLMs) has attracted many individuals to fine-tune them for domain-specific tasks by uploading their data. However, in sensitive areas like healthcare and finance, privacy concerns often arise. One promising solution is to sample synthetic data with Differential Privacy (DP) guarantees to replace private data. However, these synthetic data contain significant flawed data, which are considered as noise. Existing solutions typically rely on naive filtering by comparing ROUGE-L scores or embedding similarities, which are ineffective in addressing the noise. To address this issue, we propose RewardDS, a novel privacy-preserving framework that fine-tunes a reward proxy model and uses reward signals to guide the synthetic data generation. Our RewardDS introduces two key modules, Reward Guided Filtering and Self-Optimizing Refinement, to both filter and refine the synthetic data, effectively mitigating the noise. Extensive experiments across medical, financial, and code generation domains demonstrate the effectiveness of our method. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的成功吸引了许多人通过上传其数据来针对特定领域任务进行微调。然而，在如医疗和金融等敏感领域，隐私问题经常出现。一种有前景的解决方案是使用具有差分隐私（DP）保证的合成数据来替代私人数据。然而，这些合成数据包含大量被视为噪声的错误数据。现有解决方案通常依赖于简单的基于ROUGE-L分数或嵌入相似性的过滤方法，这些方法在处理噪声方面无效。为此，我们提出了一种名为RewardDS的新颖隐私保护框架，该框架通过训练一个奖励代理模型，并利用奖励信号指导合成数据的生成。RewardDS引入了两个关键模块——奖励引导过滤和自我优化精炼，以过滤和精炼合成数据，有效减少了噪声。我们在医疗、金融和代码生成领域进行的广泛实验证明了该方法的有效性。 

---
# A Multi-Agent Framework for Automated Vulnerability Detection and Repair in Solidity and Move Smart Contracts 

**Title (ZH)**: 基于Solidity和Move智能合约的自动化漏洞检测与修复多Agent框架 

**Authors**: Rabimba Karanjai, Sam Blackshear, Lei Xu, Weidong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2502.18515)  

**Abstract**: The rapid growth of the blockchain ecosystem and the increasing value locked in smart contracts necessitate robust security measures. While languages like Solidity and Move aim to improve smart contract security, vulnerabilities persist. This paper presents Smartify, a novel multi-agent framework leveraging Large Language Models (LLMs) to automatically detect and repair vulnerabilities in Solidity and Move smart contracts. Unlike traditional methods that rely solely on vast pre-training datasets, Smartify employs a team of specialized agents working on different specially fine-tuned LLMs to analyze code based on underlying programming concepts and language-specific security principles. We evaluated Smartify on a dataset for Solidity and a curated dataset for Move, demonstrating its effectiveness in fixing a wide range of vulnerabilities. Our results show that Smartify (Gemma2+codegemma) achieves state-of-the-art performance, surpassing existing LLMs and enhancing general-purpose models' capabilities, such as Llama 3.1. Notably, Smartify can incorporate language-specific knowledge, such as the nuances of Move, without requiring massive language-specific pre-training datasets. This work offers a detailed analysis of various LLMs' performance on smart contract repair, highlighting the strengths of our multi-agent approach and providing a blueprint for developing more secure and reliable decentralized applications in the growing blockchain landscape. We also provide a detailed recipe for extending this to other similar use cases. 

**Abstract (ZH)**: 区块链生态系统迅速增长和智能合约中锁定价值的增加 necessitate robust security measures.尽管Solidity和Move等语言旨在提高智能合约安全性，但漏洞仍然存在。本文提出了一种新颖的多智能体框架Smartify，利用大型语言模型（LLMs）自动检测和修复Solidity和Move智能合约中的漏洞。Smartify不同于依赖庞大预训练数据集的传统方法，而是采用专门针对不同语言特定安全原则进行微调的智能体团队，基于底层编程概念分析代码。Smartify在Solidity和Move的专门数据集上进行了评估，证实了其在修复各种漏洞方面的效果。我们的结果显示，Smartify（Gemma2+codegemma）达到最先进的性能，超越了现有LLMs，并增强了通用模型（如Llama 3.1）的能力。尤其是，Smartify可以在不需大规模语言特定预训练数据集的情况下，融入特定语言的知识，如Move的独特之处。本文对各种LLMs在智能合约修复中的性能进行了详细分析，强调了我们多智能体方法的优势，并为开发更加安全可靠的去中心化应用提供了蓝图。我们还提供了将此扩展到其他类似用例的详细方法。 

---
# FCoT-VL:Advancing Text-oriented Large Vision-Language Models with Efficient Visual Token Compression 

**Title (ZH)**: FCoT-VL：高效的视觉令牌压缩促进面向文本的大规模跨模态模型发展 

**Authors**: Jianjian Li, Junquan Fan, Feng Tang, Gang Huang, Shitao Zhu, Songlin Liu, Nian Xie, Wulong Liu, Yong Liao  

**Link**: [PDF](https://arxiv.org/pdf/2502.18512)  

**Abstract**: The rapid success of Vision Large Language Models (VLLMs) often depends on the high-resolution images with abundant visual tokens, which hinders training and deployment efficiency. Current training-free visual token compression methods exhibit serious performance degradation in tasks involving high-resolution, text-oriented image understanding and reasoning. In this paper, we propose an efficient visual token compression framework for text-oriented VLLMs in high-resolution scenarios. In particular, we employ a light-weight self-distillation pre-training stage to compress the visual tokens, requiring a limited numbers of image-text pairs and minimal learnable parameters. Afterwards, to mitigate potential performance degradation of token-compressed models, we construct a high-quality post-train stage. To validate the effectiveness of our method, we apply it to an advanced VLLMs, InternVL2. Experimental results show that our approach significantly reduces computational overhead while outperforming the baselines across a range of text-oriented benchmarks. We will release the models and code soon. 

**Abstract (ZH)**: Vision 大语言模型中的高效视觉令牌压缩框架：针对高分辨率场景下的文本导向模型 

---
# ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models 

**Title (ZH)**: ELBA-Bench: 一种高效的大规模语言模型后门攻击基准 

**Authors**: Xuxu Liu, Siyuan Liang, Mengya Han, Yong Luo, Aishan Liu, Xiantao Cai, Zheng He, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2502.18511)  

**Abstract**: Generative large language models are crucial in natural language processing, but they are vulnerable to backdoor attacks, where subtle triggers compromise their behavior. Although backdoor attacks against LLMs are constantly emerging, existing benchmarks remain limited in terms of sufficient coverage of attack, metric system integrity, backdoor attack alignment. And existing pre-trained backdoor attacks are idealized in practice due to resource access constraints. Therefore we establish $\textit{ELBA-Bench}$, a comprehensive and unified framework that allows attackers to inject backdoor through parameter efficient fine-tuning ($\textit{e.g.,}$ LoRA) or without fine-tuning techniques ($\textit{e.g.,}$ In-context-learning). $\textit{ELBA-Bench}$ provides over 1300 experiments encompassing the implementations of 12 attack methods, 18 datasets, and 12 LLMs. Extensive experiments provide new invaluable findings into the strengths and limitations of various attack strategies. For instance, PEFT attack consistently outperform without fine-tuning approaches in classification tasks while showing strong cross-dataset generalization with optimized triggers boosting robustness; Task-relevant backdoor optimization techniques or attack prompts along with clean and adversarial demonstrations can enhance backdoor attack success while preserving model performance on clean samples. Additionally, we introduce a universal toolbox designed for standardized backdoor attack research, with the goal of propelling further progress in this vital area. 

**Abstract (ZH)**: 生成式大型语言模型在自然语言处理中至关重要，但它们容易遭受后门攻击，其中微妙的触发器会损害其行为。尽管针对LLMs的后门攻击不断出现，但现有基准在攻击覆盖范围、度量系统完整性和后门攻击对齐方面仍然有限。而且，由于资源访问限制，现有的预训练后门攻击在实践中往往是理想化的。因此，我们建立了ELBA-Bench，这是一个全面统一的框架，允许攻击者通过高效参数微调（如LoRA）或不使用微调技术（如上下文学习）注入后门。ELBA-Bench 包括超过1300个实验，涵盖12种攻击方法、18个数据集和12个LLM的实现。广泛的实验提供了有关各种攻击策略优势和局限性的新发现。例如，在分类任务中，PEFT攻击在没有微调的情况下始终表现出色，并且优化的触发器可以增强鲁棒性从而实现跨数据集的良好泛化；具有任务相关后门优化技术或攻击提示以及干净和对抗性示范的方法可以提高后门攻击的成功率，同时保持模型在干净样本上的性能。此外，我们介绍了一个通用工具箱，旨在促进标准化后门攻击研究，从而推动这一关键领域的进一步进展。 

---
# Protecting Users From Themselves: Safeguarding Contextual Privacy in Interactions with Conversational Agents 

**Title (ZH)**: 保护用户免受自身风险：在与对话代理互动时保障上下文隐私 

**Authors**: Ivoline Ngong, Swanand Kadhe, Hao Wang, Keerthiram Murugesan, Justin D. Weisz, Amit Dhurandhar, Karthikeyan Natesan Ramamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2502.18509)  

**Abstract**: Conversational agents are increasingly woven into individuals' personal lives, yet users often underestimate the privacy risks involved. The moment users share information with these agents (e.g., LLMs), their private information becomes vulnerable to exposure. In this paper, we characterize the notion of contextual privacy for user interactions with LLMs. It aims to minimize privacy risks by ensuring that users (sender) disclose only information that is both relevant and necessary for achieving their intended goals when interacting with LLMs (untrusted receivers). Through a formative design user study, we observe how even "privacy-conscious" users inadvertently reveal sensitive information through indirect disclosures. Based on insights from this study, we propose a locally-deployable framework that operates between users and LLMs, and identifies and reformulates out-of-context information in user prompts. Our evaluation using examples from ShareGPT shows that lightweight models can effectively implement this framework, achieving strong gains in contextual privacy while preserving the user's intended interaction goals through different approaches to classify information relevant to the intended goals. 

**Abstract (ZH)**: 对话代理越来越多地融入个人生活中，但用户往往低估了其中的隐私风险。当用户向这些代理（例如，大语言模型）共享信息时，其私人信息就会面临泄露的风险。在本文中，我们刻画了用户与大语言模型互动时的上下文隐私概念，旨在通过确保用户仅披露实现其预期目标所必需的相关信息来最小化隐私风险，而这些信息在与不可信接收者（大语言模型）互动时仅需披露。通过形式化设计用户研究，我们观察到即使是“隐私意识强”的用户也会无意中通过间接披露泄露敏感信息。基于该研究的洞察，我们提出了一种本地可部署的框架，在用户和大语言模型之间运作，并识别和重新构思用户提示中的脱节信息。通过对ShareGPT示例的评估显示，轻量级模型可以有效实施该框架，在不同分类信息相关性的方法下，实现强大的上下文隐私增强，同时保持用户预期的互动目标。 

---
# REFINE: Inversion-Free Backdoor Defense via Model Reprogramming 

**Title (ZH)**: REFINE：基于模型重编程的无 inversion 后门防御 

**Authors**: Yukun Chen, Shuo Shao, Enhao Huang, Yiming Li, Pin-Yu Chen, Zhan Qin, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2502.18508)  

**Abstract**: Backdoor attacks on deep neural networks (DNNs) have emerged as a significant security threat, allowing adversaries to implant hidden malicious behaviors during the model training phase. Pre-processing-based defense, which is one of the most important defense paradigms, typically focuses on input transformations or backdoor trigger inversion (BTI) to deactivate or eliminate embedded backdoor triggers during the inference process. However, these methods suffer from inherent limitations: transformation-based defenses often fail to balance model utility and defense performance, while BTI-based defenses struggle to accurately reconstruct trigger patterns without prior knowledge. In this paper, we propose REFINE, an inversion-free backdoor defense method based on model reprogramming. REFINE consists of two key components: \textbf{(1)} an input transformation module that disrupts both benign and backdoor patterns, generating new benign features; and \textbf{(2)} an output remapping module that redefines the model's output domain to guide the input transformations effectively. By further integrating supervised contrastive loss, REFINE enhances the defense capabilities while maintaining model utility. Extensive experiments on various benchmark datasets demonstrate the effectiveness of our REFINE and its resistance to potential adaptive attacks. 

**Abstract (ZH)**: 深度神经网络（DNNs）中的后门攻击已演变为一个重要的安全威胁，允许攻击者在模型训练阶段植入隐藏的恶意行为。基于预处理的防御措施是最重要的防御范式之一，通常侧重于输入变换或后门触发器反转（BTI）来在推理过程中去激活或消除嵌入的后门触发器。然而，这些方法存在固有的局限性：基于变换的防御措施往往难以在模型实用性和防御性能之间取得平衡，而基于BTI的防御措施在缺少先验知识时难以准确重建触发器模式。本文提出REFINE，一种基于模型再编程的无反变换后门防御方法。REFINE包含两个关键组件：（1）一个输入变换模块，扰乱正常的和后门的模式，生成新的正常特征；以及（2）一个输出重映射模块，重新定义模型的输出域以有效引导输入变换。通过进一步集成监督对比损失，REFINE增强了防御能力同时保持模型的实用性。在各种基准数据集上的广泛实验证明了REFINE的有效性和对潜在适应性攻击的鲁棒性。 

---
# Exploring Patient Data Requirements in Training Effective AI Models for MRI-based Breast Cancer Classification 

**Title (ZH)**: 基于MRI的乳腺癌分类中有效AI模型训练所需患者数据探索 

**Authors**: Solha Kang, Wesley De Neve, Francois Rameau, Utku Ozbulak  

**Link**: [PDF](https://arxiv.org/pdf/2502.18506)  

**Abstract**: The past decade has witnessed a substantial increase in the number of startups and companies offering AI-based solutions for clinical decision support in medical institutions. However, the critical nature of medical decision-making raises several concerns about relying on external software. Key issues include potential variations in image modalities and the medical devices used to obtain these images, potential legal issues, and adversarial attacks. Fortunately, the open-source nature of machine learning research has made foundation models publicly available and straightforward to use for medical applications. This accessibility allows medical institutions to train their own AI-based models, thereby mitigating the aforementioned concerns. Given this context, an important question arises: how much data do medical institutions need to train effective AI models? In this study, we explore this question in relation to breast cancer detection, a particularly contested area due to the prevalence of this disease, which affects approximately 1 in every 8 women. Through large-scale experiments on various patient sizes in the training set, we show that medical institutions do not need a decade's worth of MRI images to train an AI model that performs competitively with the state-of-the-art, provided the model leverages foundation models. Furthermore, we observe that for patient counts greater than 50, the number of patients in the training set has a negligible impact on the performance of models and that simple ensembles further improve the results without additional complexity. 

**Abstract (ZH)**: 过去十年见证了为医疗机构提供基于AI的临床决策支持解决方案的初创公司和企业的显著增加。然而，医疗决策的关键性性质引发了对依赖外部软件的若干担忧。关键问题包括潜在的影像模态变化、使用的医疗设备差异、潜在的法律问题以及对抗性攻击。幸运的是，机器学习研究的开源性质使得基础模型可以公开获取并简单地应用于医疗应用。这种可及性使医疗机构能够训练自己的基于AI的模型，从而减轻上述担忧。在此背景下，一个重要的问题出现了：医疗机构需要多少数据来训练有效的AI模型？在本研究中，我们探讨了这一问题，特别是在乳腺癌检测方面的应用，这是一个特别有争议的领域，因为这种疾病在约每8名女性中就有1人受到影响。通过在训练集包含不同患者数量的大规模实验，我们展示了在模型利用基础模型的情况下，医疗机构并不需要长时间的MRI图像数据来训练与最佳性能相当的AI模型。此外，我们观察到，在患者数量超过50时，训练集中的患者数量对模型性能几乎没有任何影响，简单的集成进一步提高了结果而无需增加复杂性。 

---
# Comprehensive Analysis of Transparency and Accessibility of ChatGPT, DeepSeek, And other SoTA Large Language Models 

**Title (ZH)**: 全面分析ChatGPT、DeepSeek及其他领先大型语言模型的透明度与可访问性 

**Authors**: Ranjan Sapkota, Shaina Raza, Manoj Karkee  

**Link**: [PDF](https://arxiv.org/pdf/2502.18505)  

**Abstract**: Despite increasing discussions on open-source Artificial Intelligence (AI), existing research lacks a discussion on the transparency and accessibility of state-of-the-art (SoTA) Large Language Models (LLMs). The Open Source Initiative (OSI) has recently released its first formal definition of open-source software. This definition, when combined with standard dictionary definitions and the sparse published literature, provide an initial framework to support broader accessibility to AI models such as LLMs, but more work is essential to capture the unique dynamics of openness in AI. In addition, concerns about open-washing, where models claim openness but lack full transparency, has been raised, which limits the reproducibility, bias mitigation, and domain adaptation of these models. In this context, our study critically analyzes SoTA LLMs from the last five years, including ChatGPT, DeepSeek, LLaMA, and others, to assess their adherence to transparency standards and the implications of partial openness. Specifically, we examine transparency and accessibility from two perspectives: open-source vs. open-weight models. Our findings reveal that while some models are labeled as open-source, this does not necessarily mean they are fully open-sourced. Even in the best cases, open-source models often do not report model training data, and code as well as key metrics, such as weight accessibility, and carbon emissions. To the best of our knowledge, this is the first study that systematically examines the transparency and accessibility of over 100 different SoTA LLMs through the dual lens of open-source and open-weight models. The findings open avenues for further research and call for responsible and sustainable AI practices to ensure greater transparency, accountability, and ethical deployment of these models.(DeepSeek transparency, ChatGPT accessibility, open source, DeepSeek open source) 

**Abstract (ZH)**: 尽管对开源人工智能（AI）的讨论日益增多，现有研究缺乏对最先进的（SoTA）大型语言模型（LLMs）的透明度和 accessibility 的讨论。开放源代码倡议（OSI）最近发布了其首个正式的开放源代码软件定义。结合标准词典定义和零星的已发表文献，这些定义为支持对如LLMs这样的AI模型更广泛的访问提供了一个初步框架，但还需要更多工作来捕获AI领域开放性的独特动态。此外，对所谓的“开放漂洗”现象的担忧被提出，即模型声称开放但缺乏充分的透明度，这限制了这些模型的可再现性、偏见缓解和领域适应性。在此背景下，我们批判性地分析了过去五年中的SoTA LLMs，包括ChatGPT、DeepSeek、LLaMA等，以评估它们对透明标准的遵守情况及其半开放性的影响。具体而言，我们从开源模型 vs. 开放权重模型的两个角度来考察透明度和可访问性。我们的研究发现，虽然一些模型被标记为开源，但这并不意味着它们是完全开源的。在最佳情况下，开源模型通常也不报告模型训练数据以及代码和关键指标，如权重的可访问性和碳排放。据我们所知，这是首次通过开源和开放权重模型的双重视角系统性地检查超过100个SoTA LLMs的透明度和可访问性。这些发现为进一步研究打开了新的途径，并呼吁负责任和可持续的AI实践，以确保这些模型具有更高的透明度、可问责性和道德利用。 

---
# TurboFuzzLLM: Turbocharging Mutation-based Fuzzing for Effectively Jailbreaking Large Language Models in Practice 

**Title (ZH)**: TurboFuzzLLM: 基于突变的 fuzzing 加速用于实践中的大型语言模型逃狱攻击 

**Authors**: Aman Goel, Xian Carrie Wu, Zhe Wang, Dmitriy Bespalov, Yanjun Qi  

**Link**: [PDF](https://arxiv.org/pdf/2502.18504)  

**Abstract**: Jailbreaking large-language models (LLMs) involves testing their robustness against adversarial prompts and evaluating their ability to withstand prompt attacks that could elicit unauthorized or malicious responses. In this paper, we present TurboFuzzLLM, a mutation-based fuzzing technique for efficiently finding a collection of effective jailbreaking templates that, when combined with harmful questions, can lead a target LLM to produce harmful responses through black-box access via user prompts. We describe the limitations of directly applying existing template-based attacking techniques in practice, and present functional and efficiency-focused upgrades we added to mutation-based fuzzing to generate effective jailbreaking templates automatically. TurboFuzzLLM achieves $\geq$ 95\% attack success rates (ASR) on public datasets for leading LLMs (including GPT-4o \& GPT-4 Turbo), shows impressive generalizability to unseen harmful questions, and helps in improving model defenses to prompt attacks. 

**Abstract (ZH)**: 大规模语言模型（LLM）的越狱涉及测试其对敌对提示的鲁棒性，并评估其抵御可能引发未经授权或恶意响应的提示攻击的能力。本文提出了一种基于变异的模糊测试技术TurboFuzzLLM，该技术能够高效地发现一批有效的越狱模板，当这些模板与有害问题结合使用时，通过用户的提示可以黑盒方式使目标LLM产生有害响应。我们描述了直接应用现有模板攻击技术的局限性，并展示了我们为变异模糊测试添加的功能性和效率优化，以自动生成有效的越狱模板。TurboFuzzLLM在领先LLM（包括GPT-4o和GPT-4 Turbo）的公开数据集上实现了≥95%的攻击成功率（ASR），展现出对未见过的有害问题的强大泛化能力，并有助于改进模型对提示攻击的防御。 

---
# Deep Learning-based Dual Watermarking for Image Copyright Protection and Authentication 

**Title (ZH)**: 基于深度学习的双重水印技术及其在图像版权保护与认证中的应用 

**Authors**: Sudev Kumar Padhi, Archana Tiwari, Sk. Subidh Ali  

**Link**: [PDF](https://arxiv.org/pdf/2502.18501)  

**Abstract**: Advancements in digital technologies make it easy to modify the content of digital images. Hence, ensuring digital images integrity and authenticity is necessary to protect them against various attacks that manipulate them. We present a Deep Learning (DL) based dual invisible watermarking technique for performing source authentication, content authentication, and protecting digital content copyright of images sent over the internet. Beyond securing images, the proposed technique demonstrates robustness to content-preserving image manipulations. It is also impossible to imitate or overwrite watermarks because the cryptographic hash of the image and the dominant features of the image in the form of perceptual hash are used as watermarks. We highlighted the need for source authentication to safeguard image integrity and authenticity, along with identifying similar content for copyright protection. After exhaustive testing, we obtained a high peak signal-to-noise ratio (PSNR) and structural similarity index measure (SSIM), which implies there is a minute change in the original image after embedding our watermarks. Our trained model achieves high watermark extraction accuracy and to the best of our knowledge, this is the first deep learning-based dual watermarking technique proposed in the literature. 

**Abstract (ZH)**: 基于深度学习的双重不可见水印技术及其在网络上传输的图像的源认证、内容认证和版权保护应用 

---
# Mechanistic Understanding of Language Models in Syntactic Code Completion 

**Title (ZH)**: 语言模型在句法规则代码补全中的机制理解 

**Authors**: Samuel Miller, Daking Rai, Ziyu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2502.18499)  

**Abstract**: Recently, language models (LMs) have shown impressive proficiency in code generation tasks, especially when fine-tuned on code-specific datasets, commonly known as Code LMs. However, our understanding of the internal decision-making processes of Code LMs, such as how they use their (syntactic or semantic) knowledge, remains limited, which could lead to unintended harm as they are increasingly used in real life. This motivates us to conduct one of the first Mechanistic Interpretability works to understand how Code LMs perform a syntactic completion task, specifically the closing parenthesis task, on the CodeLlama-7b model (Roziere et al. 2023). Our findings reveal that the model requires middle-later layers until it can confidently predict the correct label for the closing parenthesis task. Additionally, we identify that while both multi-head attention (MHA) and feed-forward (FF) sub-layers play essential roles, MHA is particularly crucial. Furthermore, we also discover attention heads that keep track of the number of already closed parentheses precisely but may or may not promote a correct number of closing parentheses that are still missing, leading to a positive or negative impact on the model's performance. 

**Abstract (ZH)**: 最近，语言模型在代码生成任务中展现了令人印象深刻的成熟度，尤其是当它们针对代码特定的数据集进行微调时，这类模型通常被称为代码语言模型（Code LMs）。然而，我们对Code LMs内部决策过程的理解仍然有限，这可能导致它们在实际应用中产生不可预见的损害。这促使我们开展了针对Code Llama-7b模型（Roziere et al. 2023）的语法补全任务，进行一项机制可解释性研究，以了解Code LMs如何完成语法补全任务。我们的研究发现表明，模型需要中间层直到后期才能自信地预测闭括号任务的正确标签。此外，我们发现多头注意力（MHA）和前馈（FF）子层都扮演着关键角色，但MHA尤其重要。同时，我们还发现一些跟踪已闭合括号数量的注意力头，但这些头可能促进也可能不促进足够的闭括号，从而对模型性能产生正面或负面的影响。 

---
# A Comprehensive Survey on Composed Image Retrieval 

**Title (ZH)**: 综述性研究：合成图像检索 

**Authors**: Xuemeng Song, Haoqiang Lin, Haokun Wen, Bohan Hou, Mingzhu Xu, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2502.18495)  

**Abstract**: Composed Image Retrieval (CIR) is an emerging yet challenging task that allows users to search for target images using a multimodal query, comprising a reference image and a modification text specifying the user's desired changes to the reference image. Given its significant academic and practical value, CIR has become a rapidly growing area of interest in the computer vision and machine learning communities, particularly with the advances in deep learning. To the best of our knowledge, there is currently no comprehensive review of CIR to provide a timely overview of this field. Therefore, we synthesize insights from over 120 publications in top conferences and journals, including ACM TOIS, SIGIR, and CVPR In particular, we systematically categorize existing supervised CIR and zero-shot CIR models using a fine-grained taxonomy. For a comprehensive review, we also briefly discuss approaches for tasks closely related to CIR, such as attribute-based CIR and dialog-based CIR. Additionally, we summarize benchmark datasets for evaluation and analyze existing supervised and zero-shot CIR methods by comparing experimental results across multiple datasets. Furthermore, we present promising future directions in this field, offering practical insights for researchers interested in further exploration. 

**Abstract (ZH)**: 多重模态的图像检索（Composed Image Retrieval, CIR）是一项新兴且具有挑战性的任务，允许用户使用包含参考图像和修改文本的多模态查询来搜索目标图像。随着深度学习的进步，CIR 成为了计算机视觉和机器学习领域的一个快速发展的研究热点。据我们所知，目前尚缺乏对该领域的全面综述。因此，我们综合了超过120篇发表在顶级会议和期刊上的论文，如ACM TOIS、SIGIR和CVPR，对其进行系统分类，并通过跨多个数据集的实验结果对现有监督和零样本CIR方法进行分析。此外，我们还简要讨论了与CIR紧密相关的任务，如基于属性的CIR和对话驱动的CIR，并介绍了用于评估的基准数据集，提出了该领域有前景的研究方向，为感兴趣的科研人员提供实用建议。 

---
# Rule-based autocorrection of Piping and Instrumentation Diagrams (P&IDs) on graphs 

**Title (ZH)**: 基于规则的管道和仪表图（P&IDs）在图形上的自动修正 

**Authors**: Lukas Schulze Balhorn, Niels Seijsener, Kevin Dao, Minji Kim, Dominik P. Goldstein, Ge H. M. Driessen, Artur M. Schweidtmann  

**Link**: [PDF](https://arxiv.org/pdf/2502.18493)  

**Abstract**: A piping and instrumentation diagram (P&ID) is a central reference document in chemical process engineering. Currently, chemical engineers manually review P&IDs through visual inspection to find and rectify errors. However, engineering projects can involve hundreds to thousands of P&ID pages, creating a significant revision workload. This study proposes a rule-based method to support engineers with error detection and correction in P&IDs. The method is based on a graph representation of P&IDs, enabling automated error detection and correction, i.e., autocorrection, through rule graphs. We use our pyDEXPI Python package to generate P&ID graphs from DEXPI-standard P&IDs. In this study, we developed 33 rules based on chemical engineering knowledge and heuristics, with five selected rules demonstrated as examples. A case study on an illustrative P&ID validates the reliability and effectiveness of the rule-based autocorrection method in revising P&IDs. 

**Abstract (ZH)**: 一种工艺和仪表图的基于规则的错误检测与纠正方法 

---
# LLM4EFFI: Leveraging Large Language Models to Enhance Code Efficiency and Correctness 

**Title (ZH)**: LLM4EFFI：利用大型语言模型提升代码效率与正确性 

**Authors**: Tong Ye, Weigang Huang, Xuhong Zhang, Tengfei Ma, Peiyu Liu, Jianwei Yin, Wenhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18489)  

**Abstract**: Large Language Models (LLMs), particularly Code LLMs, have demonstrated impressive performance in code generation. Current research primarily focuses on the correctness of generated code, while efficiency remains less explored. Recent works have focused on modifying the initial version of the code to improve its efficiency. However, such refinements are limited by the algorithmic design and overall logic of the initial code, resulting in only incremental improvements. In contrast, when human developers write high-quality code, they typically begin by designing several potential solutions at the logical level, evaluating various algorithms and their complexities, and then proceeding to implement and optimize the solution. In this study, we introduce \tool: \uline{L}arge \uline{L}anguage \uline{M}odel for Code \uline{Effi}ciency, a novel framework that enables LLMs to generate code that balances both efficiency and correctness. Specifically, \tool divides the efficiency optimization process into two domains: algorithmic exploration in the logic domain and implementation optimization in the code domain. The correctness of the code is then guaranteed through a synthetic test case refinement process. This approach, which prioritizes efficiency before ensuring correctness, offers a new paradigm for efficient code generation. Experiments demonstrate that \tool consistently improves both efficiency and correctness, achieving new state-of-the-art performance in code efficiency benchmarks across various LLM backbones. 

**Abstract (ZH)**: Large Language Models for Code Efficiency: A Framework Balancing Both Correctness and Efficiency 

---
# AuPair: Golden Example Pairs for Code Repair 

**Title (ZH)**: AuPair：代码修复的黄金例对 

**Authors**: Aditi Mavalankar, Hassan Mansoor, Zita Marinho, Masha Samsikova, Tom Schaul  

**Link**: [PDF](https://arxiv.org/pdf/2502.18487)  

**Abstract**: Scaling up inference-time compute has proven to be a valuable strategy in improving the performance of Large Language Models (LLMs) without fine-tuning. An important task that can benefit from additional inference-time compute is self-repair; given an initial flawed response, or guess, the LLM corrects its own mistake and produces an improved response, or fix. We leverage the in-context learning ability of LLMs to perform self-repair in the coding domain. The key contribution of our paper is an approach that synthesises and selects an ordered set of golden example pairs, or AuPairs, of these initial guesses and subsequent fixes for the corresponding problems. Each such AuPair is provided as a single in-context example at inference time to generate a repaired solution. For an inference-time compute budget of $N$ LLM calls per problem, $N$ AuPairs are used to generate $N$ repaired solutions, out of which the highest-scoring solution is selected as the final answer. The underlying intuition is that if the LLM is given a different example of fixing an incorrect guess each time, it can subsequently generate a diverse set of repaired solutions. Our algorithm selects these AuPairs in a manner that maximises complementarity and usefulness. We demonstrate the results of our algorithm on 5 LLMs across 7 competitive programming datasets for the code repair task. Our algorithm yields a significant boost in performance compared to best-of-$N$ and self-repair, and also exhibits strong generalisation across datasets and models. Moreover, our approach shows significantly stronger scaling with inference-time compute budget compared to baselines. 

**Abstract (ZH)**: 提升推理时计算能力已被证明是提高大型语言模型性能的重要策略，无需微调。我们的研究表明，在编码领域利用大型语言模型的上下文学习能力进行自我修复是一个有益的任务；给定一个初始的错误响应或猜测，大型语言模型纠正自身的错误并生成改进的响应或修复。我们利用大型语言模型的上下文学习能力，在编码领域实现自我修复。本文的主要贡献是一种合成并选择有序的一组金标准示例对（AuPairs），这些金标准示例对包括初始猜测和后续修复。在推理时，每次为每个问题提供一个此类AuPair作为单个上下文示例以生成修复解决方案。对于每个问题的推理时计算预算为$N$次大型语言模型调用，使用$N$个AuPairs生成$N$个修复解决方案，并从中选择最高评分的解决方案作为最终答案。我们的算法选择这些AuPairs以最大化互补性和有用性。我们使用5个大型语言模型在7个竞争编程数据集上对我们的算法进行了代码修复任务的结果展示。我们的算法相比最优选项和自我修复显著提升了性能，并且在数据集和模型上的泛化能力较强。此外，与基准方法相比，我们的方法在推理时计算预算上的扩展能力更强。 

---
# AI Enhanced Ontology Driven NLP for Intelligent Cloud Resource Query Processing Using Knowledge Graphs 

**Title (ZH)**: AI增强本体驱动的自然语言处理在知识图谱支持下的智能云资源查询处理 

**Authors**: Krishna Chaitanya Sunkara, Krishnaiah Narukulla  

**Link**: [PDF](https://arxiv.org/pdf/2502.18484)  

**Abstract**: The conventional resource search in cloud infrastructure relies on keyword-based searches or GUIDs, which demand exact matches and significant user effort to locate resources. These conventional search approaches often fail to interpret the intent behind natural language queries, making resource discovery inefficient and inaccessible to users. Though there exists some form of NLP based search engines, they are limited and focused more on analyzing the NLP query itself and extracting identifiers to find the resources. But they fail to search resources based on their behavior or operations or their capabilities or relationships or features or business relevance or the dynamic changing state or the knowledge these resources have. The search criteria has been changing with the inundation of AI based services which involved discovering not just the requested resources and identifiers but seeking insights. The real intent of a search has never been to just to list the resources but with some actual context such as to understand causes of some behavior in the system, compliance checks, capacity estimations, network constraints, or troubleshooting or business insights. This paper proposes an advanced Natural Language Processing (NLP) enhanced by ontology-based semantics to enable intuitive, human-readable queries which allows users to actually discover the intent-of-search itself. By constructing an ontology of cloud resources, their interactions, and behaviors, the proposed framework enables dynamic intent extraction and relevance ranking using Latent Semantic Indexing (LSI) and AI models. It introduces an automated pipeline which integrates ontology extraction by AI powered data crawlers, building a semantic knowledge base for context aware resource discovery. 

**Abstract (ZH)**: 基于本体的语义增强自然语言处理在云基础设施资源搜索中的应用 

---
# Modeling Churn in Recommender Systems with Aggregated Preferences 

**Title (ZH)**: 基于聚合偏好建模推荐系统中的客户流失 

**Authors**: Gur Keinan, Omer Ben-Porat  

**Link**: [PDF](https://arxiv.org/pdf/2502.18483)  

**Abstract**: While recommender systems (RSs) traditionally rely on extensive individual user data, regulatory and technological shifts necessitate reliance on aggregated user information. This shift significantly impacts the recommendation process, requiring RSs to engage in intensive exploration to identify user preferences. However, this approach risks user churn due to potentially unsatisfactory recommendations. In this paper, we propose a model that addresses the dual challenges of leveraging aggregated user information and mitigating churn risk. Our model assumes that the RS operates with a probabilistic prior over user types and aggregated satisfaction levels for various content types. We demonstrate that optimal policies naturally transition from exploration to exploitation in finite time, develop a branch-and-bound algorithm for computing these policies, and empirically validate its effectiveness. 

**Abstract (ZH)**: 推荐系统在利用聚合用户信息和减轻用户流失风险方面的模型 

---
# MixLLM: Dynamic Routing in Mixed Large Language Models 

**Title (ZH)**: MixLLM: 混合大型语言模型的动态路由 

**Authors**: Xinyuan Wang, Yanchi Liu, Wei Cheng, Xujiang Zhao, Zhengzhang Chen, Wenchao Yu, Yanjie Fu, Haifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18482)  

**Abstract**: Large Language Models (LLMs) exhibit potential artificial generic intelligence recently, however, their usage is costly with high response latency. Given mixed LLMs with their own strengths and weaknesses, LLM routing aims to identify the most suitable model for each query in the stream to maximize response quality and minimize cost and latency. However, the challenges involve: (1) dynamic trade-offs among quality, cost, and latency; (2) enabling continual learning in deployed systems; and (3) navigating a varying (e.g., new LLM addition or old LLM removal) set of LLM candidates over time. To bridge these gaps, we develop MixLLM, a dynamic contextual-bandit-based routing system for query-LLM assignment. Specifically, we first leverage query tags to enhance query embeddings for the routing task. Next, we design lightweight prediction models to estimate the response qualities and costs of queries over LLMs. We then devise a meta-decision maker to choose the query-LLM assignments to best tradeoff response quality, cost, and latency. Finally, the system benefits from continual training, allowing it to adapt to evolving queries and user feedback over time. Our extensive experiments show that MixLLM achieves the best trade-offs in response quality, cost, and latency (97.25% of GPT-4's quality at 24.18% of the cost under the time constraint). 

**Abstract (ZH)**: 大规模语言模型（LLMs） recently exhibited potential artificial general intelligence, but their usage is costly with high response latency. Given mixed LLMs with their own strengths and weaknesses, LLM routing aims to identify the most suitable model for each query in the stream to maximize response quality and minimize cost and latency. However, challenges include: (1) dynamic trade-offs among quality, cost, and latency; (2) enabling continual learning in deployed systems; and (3) navigating a varying (e.g., new LLM addition or old LLM removal) set of LLM candidates over time. To bridge these gaps, we develop MixLLM, a dynamic contextual-bandit-based routing system for query-LLM assignment. Specifically, we first leverage query tags to enhance query embeddings for the routing task. Next, we design lightweight prediction models to estimate the response qualities and costs of queries over LLMs. We then devise a meta-decision maker to choose the query-LLM assignments to best tradeoff response quality, cost, and latency. Finally, the system benefits from continual training, allowing it to adapt to evolving queries and user feedback over time. Our extensive experiments show that MixLLM achieves the best trade-offs in response quality, cost, and latency (97.25% of GPT-4's quality at 24.18% of the cost under the time constraint). 

---
# MDE: Modality Discrimination Enhancement for Multi-modal Recommendation 

**Title (ZH)**: MDE: 多模态推荐中的模态鉴别增强 

**Authors**: Hang Zhou, Yucheng Wang, Huijing Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2502.18481)  

**Abstract**: Multi-modal recommendation systems aim to enhance performance by integrating an item's content features across various modalities with user behavior data. Effective utilization of features from different modalities requires addressing two challenges: preserving semantic commonality across modalities (modality-shared) and capturing unique characteristics for each modality (modality-specific). Most existing approaches focus on aligning feature spaces across modalities, which helps represent modality-shared features. However, modality-specific distinctions are often neglected, especially when there are significant semantic variations between modalities. To address this, we propose a Modality Distinctiveness Enhancement (MDE) framework that prioritizes extracting modality-specific information to improve recommendation accuracy while maintaining shared features. MDE enhances differences across modalities through a novel multi-modal fusion module and introduces a node-level trade-off mechanism to balance cross-modal alignment and differentiation. Extensive experiments on three public datasets show that our approach significantly outperforms other state-of-the-art methods, demonstrating the effectiveness of jointly considering modality-shared and modality-specific features. 

**Abstract (ZH)**: 多模态推荐系统旨在通过整合项目内容特征与用户行为数据来提升性能。有效利用不同模态的特征需要解决两个挑战：保持模态间的语义一致性（模态共享）和捕捉每个模态的独特特征（模态特定）。大多数现有方法集中在对齐跨模态的特征空间，这有助于表示模态共享特征。然而，模态特定的差异往往被忽视，尤其是在模态间存在显著语义差异时更为明显。为解决这一问题，我们提出了一种模态区分性增强（MDE）框架，该框架优先提取模态特定信息以提高推荐准确性同时保持共享特征。MDE 通过一个新颖的多模态融合模块增强模态间的差异，并引入节点级权衡机制以平衡跨模态对齐和区分。在三个公开数据集上的广泛实验表明，我们的方法显著优于其他最先进的方法，证明了同时考虑模态共享和模态特定特征的有效性。 

---
# QExplorer: Large Language Model Based Query Extraction for Toxic Content Exploration 

**Title (ZH)**: QExplorer：基于大型语言模型的有毒内容查询提取 

**Authors**: Shaola Ren, Li Ke, Longtao Huang, Dehong Gao, Hui Xue  

**Link**: [PDF](https://arxiv.org/pdf/2502.18480)  

**Abstract**: Automatically extracting effective queries is challenging in information retrieval, especially in toxic content exploration, as such content is likely to be disguised. With the recent achievements in generative Large Language Model (LLM), we are able to leverage the capabilities of LLMs to extract effective queries for similar content exploration directly. This study proposes QExplorer, an approach of large language model based Query Extraction for toxic content Exploration. The QExplorer approach involves a 2-stage training process: instruction Supervised FineTuning (SFT) and preference alignment using Direct Preference Optimization (DPO), as well as the datasets construction with feedback of search system. To verify the effectiveness of QExplorer, a series of offline and online experiments are conducted on our real-world system. The offline empirical results demonstrate that the performance of our automatic query extraction outperforms that of several LLMs and humans. The online deployment shows a significant increase in the detection of toxic items. 

**Abstract (ZH)**: 基于大型语言模型的有毒内容探索查询提取方法QExplorer 

---
# Beyond Self-Consistency: Loss-Balanced Perturbation-Based Regularization Improves Industrial-Scale Ads Ranking 

**Title (ZH)**: 超越自我一致性：损失平衡扰动正则化提升工业规模广告排名 

**Authors**: Ilqar Ramazanli, Hamid Eghbalzadeh, Xiaoyi Liu, Yang Wang, Jiaxiang Fu, Kaushik Rangadurai, Sem Park, Bo Long, Xue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2502.18478)  

**Abstract**: Perturbation-based regularization techniques address many challenges in industrial-scale large models, particularly with sparse labels, and emphasize consistency and invariance for perturbation in model predictions. One of the popular regularization techniques has been various forms of self-consistency, which involve making small modifications to input data while preserving contextual information and enforcing similar predictions through auxiliary loss functions. In this work, we explore the first successful application of perturbation-based regularization algorithms in large-scale ads ranking models, and further propose a novel regularization algorithm, namely, Loss-Balanced Small Perturbation Regularization (LSPR) that can be used in potentially any deep learning model. We have successfully demonstrate that both Self-Consistency Regularization approaches (SCR) and LSPR are scalable and can improve ads delivery systems. By conducting industrial-scale experiments, and numerical analysis, we additionally show that our proposed LSPR, performs consistently better compared to SCR, across various groups and signal availability setups. Finally, we report a successful application of the proposed LSPR in a billion-scale industrial ranking system, which to the best of our knowledge, is the first of its kind, and it is specially designed to address the various scalability challenges (e.g, various surfaces, geological locations, clients and so on) as we will mention in this paper. 

**Abstract (ZH)**: 基于扰动的正则化技术在工业规模大型模型中解决了许多挑战，特别是稀疏标签问题，并强调扰动下模型预测的一致性和不变性。一种流行的正则化技术是各种形式的自一致性，这涉及在保持上下文信息的同时对输入数据进行小的修改，并通过辅助损失函数强制类似的预测。在本文中，我们探索了基于扰动的正则化算法在大规模广告排名模型中的首次成功应用，并进一步提出了一种新的正则化算法，即损失平衡的小扰动正则化（LSPR），该算法可以应用于任何深度学习模型。我们成功地证明了自一致性正则化方法（SCR）和LSPR都具有可扩展性并可以改进广告交付系统。通过进行工业规模的实验和数值分析，我们还表明，与SCR相比，我们提出的LSPR在各种组别和信号可用性设置下表现更一致。最后，我们报告了提出LSPR在十亿规模的工业排名系统中的成功应用，据我们所知，这是该领域的首次应用，并且特别设计用于解决各种可扩展性挑战（例如，各种表面、地质位置、客户等），如本文中所述。 

---
# Recommendations Beyond Catalogs: Diffusion Models for Personalized Generation 

**Title (ZH)**: 超越目录的推荐：面向个性化生成的扩散模型 

**Authors**: Gabriel Patron, Zhiwei Xu, Ishan Kapnadak, Felipe Maia Polo  

**Link**: [PDF](https://arxiv.org/pdf/2502.18477)  

**Abstract**: Modern recommender systems follow the guiding principle of serving the right user, the right item at the right time. One of their main limitations is that they are typically limited to items already in the catalog. We propose REcommendations BEyond CAtalogs, REBECA, a new class of probabilistic diffusion-based recommender systems that synthesize new items tailored to individual tastes rather than retrieve items from the catalog. REBECA combines efficient training in embedding space with a novel diffusion prior that only requires users' past ratings of items. We evaluate REBECA on real-world data and propose novel personalization metrics for generative recommender systems. Extensive experiments demonstrate that REBECA produces high-quality, personalized recommendations, generating images that align with users' unique preferences. 

**Abstract (ZH)**: 超越目录的推荐：REBECA 

---
# A Contemporary Survey of Large Language Model Assisted Program Analysis 

**Title (ZH)**: 大型语言模型辅助程序分析的当代调研 

**Authors**: Jiayimei Wang, Tao Ni, Wei-Bin Lee, Qingchuan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.18474)  

**Abstract**: The increasing complexity of software systems has driven significant advancements in program analysis, as traditional methods unable to meet the demands of modern software development. To address these limitations, deep learning techniques, particularly Large Language Models (LLMs), have gained attention due to their context-aware capabilities in code comprehension. Recognizing the potential of LLMs, researchers have extensively explored their application in program analysis since their introduction. Despite existing surveys on LLM applications in cybersecurity, comprehensive reviews specifically addressing their role in program analysis remain scarce. In this survey, we systematically review the application of LLMs in program analysis, categorizing the existing work into static analysis, dynamic analysis, and hybrid approaches. Moreover, by examining and synthesizing recent studies, we identify future directions and challenges in the field. This survey aims to demonstrate the potential of LLMs in advancing program analysis practices and offer actionable insights for security researchers seeking to enhance detection frameworks or develop domain-specific models. 

**Abstract (ZH)**: 随着软件系统的日益复杂，程序分析取得了显著进展，传统的分析方法已无法满足现代软件开发的需求。为了解决这些局限性，深度学习技术，尤其是大型语言模型（LLMs），因其在代码理解方面的上下文感知能力而引起了广泛关注。鉴于LLMs的潜力，研究人员在其推出后广泛探索了它们在程序分析中的应用。尽管已有针对LLMs在网络安全中应用的综述，但专门讨论其在程序分析中作用的综合研究仍然较少。在这篇综述中，我们系统地回顾了LLMs在程序分析中的应用，将现有研究归纳为静态分析、动态分析和混合方法。此外，通过对近期研究的分析和综合，我们指出了该领域的未来方向和挑战。本文旨在展示LLMs在推动程序分析实践方面的发展潜力，并为寻求改进检测框架或开发领域特定模型的网络安全研究人员提供可行的见解。 

---
# FinBloom: Knowledge Grounding Large Language Model with Real-time Financial Data 

**Title (ZH)**: FinBloom：基于实时金融数据的大型语言模型知识接地 

**Authors**: Ankur Sinha, Chaitanya Agarwal, Pekka Malo  

**Link**: [PDF](https://arxiv.org/pdf/2502.18471)  

**Abstract**: Large language models (LLMs) excel at generating human-like responses but often struggle with interactive tasks that require access to real-time information. This limitation poses challenges in finance, where models must access up-to-date information, such as recent news or price movements, to support decision-making. To address this, we introduce Financial Agent, a knowledge-grounding approach for LLMs to handle financial queries using real-time text and tabular data. Our contributions are threefold: First, we develop a Financial Context Dataset of over 50,000 financial queries paired with the required context. Second, we train FinBloom 7B, a custom 7 billion parameter LLM, on 14 million financial news articles from Reuters and Deutsche Presse-Agentur, alongside 12 million Securities and Exchange Commission (SEC) filings. Third, we fine-tune FinBloom 7B using the Financial Context Dataset to serve as a Financial Agent. This agent generates relevant financial context, enabling efficient real-time data retrieval to answer user queries. By reducing latency and eliminating the need for users to manually provide accurate data, our approach significantly enhances the capability of LLMs to handle dynamic financial tasks. Our proposed approach makes real-time financial decisions, algorithmic trading and other related tasks streamlined, and is valuable in contexts with high-velocity data flows. 

**Abstract (ZH)**: 基于知识 grounding 的大语言模型在处理实时金融查询方面的金融代理方法 

---
# SOK: Exploring Hallucinations and Security Risks in AI-Assisted Software Development with Insights for LLM Deployment 

**Title (ZH)**: SOK：探索AI辅助软件开发中的幻觉和安全风险及对大规模语言模型部署的洞察 

**Authors**: Ariful Haque, Sunzida Siddique, Md. Mahfuzur Rahman, Ahmed Rafi Hasan, Laxmi Rani Das, Marufa Kamal, Tasnim Masura, Kishor Datta Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2502.18468)  

**Abstract**: The integration of Large Language Models (LLMs) such as GitHub Copilot, ChatGPT, Cursor AI, and Codeium AI into software development has revolutionized the coding landscape, offering significant productivity gains, automation, and enhanced debugging capabilities. These tools have proven invaluable for generating code snippets, refactoring existing code, and providing real-time support to developers. However, their widespread adoption also presents notable challenges, particularly in terms of security vulnerabilities, code quality, and ethical concerns. This paper provides a comprehensive analysis of the benefits and risks associated with AI-powered coding tools, drawing on user feedback, security analyses, and practical use cases. We explore the potential for these tools to replicate insecure coding practices, introduce biases, and generate incorrect or non-sensical code (hallucinations). In addition, we discuss the risks of data leaks, intellectual property violations and the need for robust security measures to mitigate these threats. By comparing the features and performance of these tools, we aim to guide developers in making informed decisions about their use, ensuring that the benefits of AI-assisted coding are maximized while minimizing associated risks. 

**Abstract (ZH)**: 大型语言模型（LLMs）如GitHub Copilot、ChatGPT、Cursor AI和Codeium AI在软件开发中的集成已经革新了编程landscape，提供了显著的生产力提升、自动化和增强的调试能力。这些工具在生成代码片段、重构现有代码和为开发者提供实时支持方面证明了其价值。然而，它们的广泛应用也带来了显著的挑战，特别是在安全漏洞、代码质量以及伦理问题方面。本文通过对用户反馈、安全分析和实际案例进行综合分析，探讨了AI助力编程工具的优势与风险。我们研究了这些工具复制不安全编程实践、引入偏见以及生成错误或无意义代码（幻觉）的潜在风险。此外，我们还讨论了数据泄露、知识产权侵犯的风险以及需要采取 robust 安全措施来应对这些威胁。通过比较这些工具的功能和性能，我们旨在指导开发者做出明智的使用决策，确保AI辅助编程的益处最大化，同时最小化相关风险。 

---
# ChatGPT vs. DeepSeek: A Comparative Study on AI-Based Code Generation 

**Title (ZH)**: ChatGPT 与 DeepSeek 的基于 AI 的代码生成 comparative study 

**Authors**: Md Motaleb Hossen Manik  

**Link**: [PDF](https://arxiv.org/pdf/2502.18467)  

**Abstract**: Background: AI-powered code generation, fueled by Large Language Models (LLMs), is revolutionizing software development. Models like OpenAI's Codex and GPT-4, alongside DeepSeek, leverage vast code and natural language datasets. However, ensuring code quality, correctness, and managing complex tasks remains challenging, necessitating thorough evaluation. Methodology: This research compares ChatGPT (version o1) and DeepSeek (version R1) for Python code generation using online judge coding challenges. It evaluates correctness (online judge verdicts, up to three attempts), code quality (Pylint/Flake8), and efficiency (execution time/memory usage). Results: DeepSeek demonstrated higher correctness, particularly on algorithmic tasks, often achieving 'Accepted' on the first attempt. ChatGPT sometimes requires multiple attempts or failures. ChatGPT encountered fewer issues, used comparable or slightly less memory, consumed less execution times and wrote fewer lines of code. Conclusion: DeepSeek exhibited superior correctness in Python code generation, often requiring fewer attempts, suggesting an advantage in algorithmic problem-solving. Both models showed almost similar efficiency in execution time and memory use. Finally, this research provides insights for developers choosing AI coding assistants and informs future AI-driven software development research. 

**Abstract (ZH)**: 背景：由大规模语言模型（LLMs）驱动的人工智能代码生成正在革新软件开发。像OpenAI的Codex和GPT-4以及DeepSeek这样的模型利用了大量的代码和自然语言数据集。然而，确保代码质量、正确性并管理复杂任务仍然具有挑战性，需要进行全面评估。方法：本研究比较了ChatGPT（版本o1）和DeepSeek（版本R1）在使用在线判题编程挑战中的Python代码生成。本研究从正确性（在线判题结果，最多三次尝试）、代码质量（Pylint/Flake8）和效率（执行时间/内存使用）三个方面进行了评估。结果：DeepSeek在正确性方面表现更好，尤其是在算法任务方面，经常在第一次尝试就能得到“通过”。ChatGPT有时需要多次尝试或失败。ChatGPT遇到的问题较少，使用的内存与DeepSeek相当或略少，执行时间更短，并编写了更少的代码。结论：DeepSeek在Python代码生成的正确性方面展现出优势，通常需要较少的尝试，表明在算法问题解决方面具有优势。两种模型在执行时间和内存使用方面几乎相似。最终，本研究为选择AI编程助手的开发人员提供了见解，并对未来基于AI的软件开发研究产生了影响。 

---
# MLScent A tool for Anti-pattern detection in ML projects 

**Title (ZH)**: MLScent：一种用于检测ML项目中的反模式工具 

**Authors**: Karthik Shivashankar, Antonio Martini  

**Link**: [PDF](https://arxiv.org/pdf/2502.18466)  

**Abstract**: Machine learning (ML) codebases face unprecedented challenges in maintaining code quality and sustainability as their complexity grows exponentially. While traditional code smell detection tools exist, they fail to address ML-specific issues that can significantly impact model performance, reproducibility, and maintainability.
This paper introduces MLScent, a novel static analysis tool that leverages sophisticated Abstract Syntax Tree (AST) analysis to detect anti-patterns and code smells specific to ML projects.
MLScent implements 76 distinct detectors across major ML frameworks including TensorFlow (13 detectors), PyTorch (12 detectors), Scikit-learn (9 detectors), and Hugging Face (10 detectors), along with data science libraries like Pandas and NumPy (8 detectors each). The tool's architecture also integrates general ML smell detection (16 detectors), and specialized analysis for data preprocessing and model training workflows.
Our evaluation demonstrates MLScent's effectiveness through both quantitative classification metrics and qualitative assessment via user studies feedback with ML practitioners. Results show high accuracy in identifying framework-specific anti-patterns, data handling issues, and general ML code smells across real-world projects. 

**Abstract (ZH)**: 机器学习代码库随着复杂性的指数级增长，面临着前所未有的保持代码质量和可持续性的挑战。虽然传统代码异味检测工具存在，但它们无法解决严重影响模型性能、再现性和可维护性的机器学习特定问题。

本文介绍了一种名为MLScent的新型静态分析工具，该工具利用高级抽象语法树（AST）分析来检测特定于机器学习项目的反模式和代码异味。

MLScent在包括TensorFlow（13个检测器）、PyTorch（12个检测器）、Scikit-learn（9个检测器）、Hugging Face（10个检测器）等主要机器学习框架，以及Pandas和NumPy等数据分析库（每个8个检测器）中实施了76个不同的检测器。该工具的架构还集成了通用机器学习代码异味检测（16个检测器），以及数据预处理和模型训练工作流程的专门分析。

我们的评估通过定量分类指标和用户研究反馈的定性评估，展示了MLScent的有效性。结果表明，MLScent在识别框架特定的反模式、数据处理问题以及一般机器学习代码异味方面具有高准确性，适用于实际项目。 

---
