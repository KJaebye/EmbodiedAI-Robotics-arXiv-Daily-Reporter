# A Survey of Self-Evolving Agents: On Path to Artificial Super Intelligence 

**Title (ZH)**: 自我演化智能体综述：通往人工超级智能的道路 

**Authors**: Huan-ang Gao, Jiayi Geng, Wenyue Hua, Mengkang Hu, Xinzhe Juan, Hongzhang Liu, Shilong Liu, Jiahao Qiu, Xuan Qi, Yiran Wu, Hongru Wang, Han Xiao, Yuhang Zhou, Shaokun Zhang, Jiayi Zhang, Jinyu Xiang, Yixiong Fang, Qiwen Zhao, Dongrui Liu, Qihan Ren, Cheng Qian, Zhenghailong Wang, Minda Hu, Huazheng Wang, Qingyun Wu, Heng Ji, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.21046)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities but remain fundamentally static, unable to adapt their internal parameters to novel tasks, evolving knowledge domains, or dynamic interaction contexts. As LLMs are increasingly deployed in open-ended, interactive environments, this static nature has become a critical bottleneck, necessitating agents that can adaptively reason, act, and evolve in real time. This paradigm shift -- from scaling static models to developing self-evolving agents -- has sparked growing interest in architectures and methods enabling continual learning and adaptation from data, interactions, and experiences. This survey provides the first systematic and comprehensive review of self-evolving agents, organized around three foundational dimensions -- what to evolve, when to evolve, and how to evolve. We examine evolutionary mechanisms across agent components (e.g., models, memory, tools, architecture), categorize adaptation methods by stages (e.g., intra-test-time, inter-test-time), and analyze the algorithmic and architectural designs that guide evolutionary adaptation (e.g., scalar rewards, textual feedback, single-agent and multi-agent systems). Additionally, we analyze evaluation metrics and benchmarks tailored for self-evolving agents, highlight applications in domains such as coding, education, and healthcare, and identify critical challenges and research directions in safety, scalability, and co-evolutionary dynamics. By providing a structured framework for understanding and designing self-evolving agents, this survey establishes a roadmap for advancing adaptive agentic systems in both research and real-world deployments, ultimately shedding lights to pave the way for the realization of Artificial Super Intelligence (ASI), where agents evolve autonomously, performing at or beyond human-level intelligence across a wide array of tasks. 

**Abstract (ZH)**: 大型语言模型(Large Language Models)展示了强大的能力，但仍然本质上是静态的，无法适应新的任务、不断发展的知识领域或动态的交互环境。随着大型语言模型在开放式互动环境中越来越广泛的应用，这一静态特性已成为关键瓶颈，需要能够实时适应、推理和进化的代理。这一范式转变——从扩展静态模型转向开发自我进化的代理——激发了对能够通过数据、互动和经验进行持续学习和适应的架构和方法的研究兴趣。本综述首次系统性和全面性地回顾了自我进化的代理，围绕三个基础维度——进化什么、何时进化和如何进化——进行组织。我们探讨了跨越代理组件（如模型、记忆、工具、架构）的进化机制，按照阶段分类适应方法（如测试时内、测试时外），并分析了指导进化适应的算法和架构设计（如标量奖励、文本反馈、单智能体和多智能体系统）。此外，我们分析了为自我进化的代理定制的评估指标和基准，展示了在编码、教育和医疗等领域的应用，并指出了安全、可扩展性和协同进化动力学方面的重要挑战和研究方向。通过提供理解和设计自我进化的代理的结构化框架，本综述为研究和实际应用中自适应代理系统的进步绘制了蓝图，最终为实现人工超级智能（ASI）提供了光线，其中代理能够自主进化，在广泛的任务中达到或超过人类智能水平。 

---
# GenoMAS: A Multi-Agent Framework for Scientific Discovery via Code-Driven Gene Expression Analysis 

**Title (ZH)**: GenoMAS: 一种基于代码驱动基因表达分析的多智能体框架用于科学发现 

**Authors**: Haoyang Liu, Yijiang Li, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.21035)  

**Abstract**: Gene expression analysis holds the key to many biomedical discoveries, yet extracting insights from raw transcriptomic data remains formidable due to the complexity of multiple large, semi-structured files and the need for extensive domain expertise. Current automation approaches are often limited by either inflexible workflows that break down in edge cases or by fully autonomous agents that lack the necessary precision for rigorous scientific inquiry. GenoMAS charts a different course by presenting a team of LLM-based scientists that integrates the reliability of structured workflows with the adaptability of autonomous agents. GenoMAS orchestrates six specialized LLM agents through typed message-passing protocols, each contributing complementary strengths to a shared analytic canvas. At the heart of GenoMAS lies a guided-planning framework: programming agents unfold high-level task guidelines into Action Units and, at each juncture, elect to advance, revise, bypass, or backtrack, thereby maintaining logical coherence while bending gracefully to the idiosyncrasies of genomic data.
On the GenoTEX benchmark, GenoMAS reaches a Composite Similarity Correlation of 89.13% for data preprocessing and an F$_1$ of 60.48% for gene identification, surpassing the best prior art by 10.61% and 16.85% respectively. Beyond metrics, GenoMAS surfaces biologically plausible gene-phenotype associations corroborated by the literature, all while adjusting for latent confounders. Code is available at this https URL. 

**Abstract (ZH)**: 基因表达分析是许多生物医学发现的关键，但由于原始转录组数据的复杂性和需要广泛的领域专业知识，从这些数据中提取洞察仍颇具挑战。当前的自动化方法往往要么受限于在边缘情况下会失效的僵化工作流程，要么受限于缺乏足够精确性以进行严格的科学探究的完全自主代理。GenoMAS通过呈现一组基于LLM的科学家团队，结合了结构化工作流程的可靠性与自主代理的适应性，开辟了一条不同的道路。GenoMAS通过类型化消息传递协议协调六种专门的LLM代理，每种代理在共享的分析画布上贡献互补的优势。GenoMAS的核心在于指导性规划框架：编程代理将高层任务指南分解为Action Units，并在每一步选择前进、修订、绕过或回溯，从而保持逻辑连贯性，同时顺应基因组数据的复杂性。

在GenoTEX基准测试中，GenoMAS在数据预处理方面的综合相似性相关性达到89.13%，在基因识别方面的F$_1$值达到60.48%，分别超越了现有最佳方法10.61%和16.85%。除了指标外，GenoMAS还揭示了由文献支持的生物合理的基因-表型关联，并调整了潜在混杂因素。代码可在以下链接获取：this https URL。 

---
# Smart Expansion Techniques for ASP-based Interactive Configuration 

**Title (ZH)**: 基于ASP的交互配置的智能扩展技术 

**Authors**: Lucia Balážová, Richard Comploi-Taupe, Susana Hahn, Nicolas Rühling, Gottfried Schenner  

**Link**: [PDF](https://arxiv.org/pdf/2507.21027)  

**Abstract**: Product configuration is a successful application of Answer Set Programming (ASP). However, challenges are still open for interactive systems to effectively guide users through the configuration process. The aim of our work is to provide an ASP-based solver for interactive configuration that can deal with large-scale industrial configuration problems and that supports intuitive user interfaces via an API. In this paper, we focus on improving the performance of automatically completing a partial configuration. Our main contribution enhances the classical incremental approach for multi-shot solving by four different smart expansion functions. The core idea is to determine and add specific objects or associations to the partial configuration by exploiting cautious and brave consequences before checking for the existence of a complete configuration with the current objects in each iteration. This approach limits the number of costly unsatisfiability checks and reduces the search space, thereby improving solving performance. In addition, we present a user interface that uses our API and is implemented in ASP. 

**Abstract (ZH)**: 基于ASP的交互配置求解器及其应用：改进部分配置自动完成的性能并通过API支持直观用户界面 

---
# MIRAGE-Bench: LLM Agent is Hallucinating and Where to Find Them 

**Title (ZH)**: MIRAGE-Bench: LLM代理在胡言乱语以及如何发现它们 

**Authors**: Weichen Zhang, Yiyou Sun, Pohao Huang, Jiayue Pu, Heyue Lin, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2507.21017)  

**Abstract**: Hallucinations pose critical risks for large language model (LLM)-based agents, often manifesting as hallucinative actions resulting from fabricated or misinterpreted information within the cognitive context. While recent studies have exposed such failures, existing evaluations remain fragmented and lack a principled testbed. In this paper, we present MIRAGE-Bench--Measuring Illusions in Risky AGEnt settings--the first unified benchmark for eliciting and evaluating hallucinations in interactive LLM-agent scenarios. We begin by introducing a three-part taxonomy to address agentic hallucinations: actions that are unfaithful to (i) task instructions, (ii) execution history, or (iii) environment observations. To analyze, we first elicit such failures by performing a systematic audit of existing agent benchmarks, then synthesize test cases using a snapshot strategy that isolates decision points in deterministic and reproducible manners. To evaluate hallucination behaviors, we adopt a fine-grained-level LLM-as-a-Judge paradigm with tailored risk-aware prompts, enabling scalable, high-fidelity assessment of agent actions without enumerating full action spaces. MIRAGE-Bench provides actionable insights on failure modes of LLM agents and lays the groundwork for principled progress in mitigating hallucinations in interactive environments. 

**Abstract (ZH)**: MIRAGE-Bench：测量风险博弈中幻觉的基准 

---
# Core Safety Values for Provably Corrigible Agents 

**Title (ZH)**: 可证明可纠正的智能体的核心安全价值观 

**Authors**: Aran Nayebi  

**Link**: [PDF](https://arxiv.org/pdf/2507.20964)  

**Abstract**: We introduce the first implementable framework for corrigibility, with provable guarantees in multi-step, partially observed environments. Our framework replaces a single opaque reward with five *structurally separate* utility heads -- deference, switch-access preservation, truthfulness, low-impact behavior via a belief-based extension of Attainable Utility Preservation, and bounded task reward -- combined lexicographically by strict weight gaps. Theorem 1 proves exact single-round corrigibility in the partially observable off-switch game; Theorem 3 extends the guarantee to multi-step, self-spawning agents, showing that even if each head is \emph{learned} to mean-squared error $\varepsilon$ and the planner is $\varepsilon$-sub-optimal, the probability of violating \emph{any} safety property is bounded while still ensuring net human benefit. In contrast to Constitutional AI or RLHF/RLAIF, which merge all norms into one learned scalar, our separation makes obedience and impact-limits dominate even when incentives conflict. For open-ended settings where adversaries can modify the agent, we prove that deciding whether an arbitrary post-hack agent will ever violate corrigibility is undecidable by reduction to the halting problem, then carve out a finite-horizon ``decidable island'' where safety can be certified in randomized polynomial time and verified with privacy-preserving, constant-round zero-knowledge proofs. Consequently, the remaining challenge is the ordinary ML task of data coverage and generalization: reward-hacking risk is pushed into evaluation quality rather than hidden incentive leak-through, giving clearer implementation guidance for today's LLM assistants and future autonomous systems. 

**Abstract (ZH)**: 我们介绍了第一个可实施的矫正性框架，在多步部分可观测环境中提供可证明的保证。该框架用五个结构上分离的效用头——遵从性、开关访问保存、真实性、基于信念扩展的可实现效用保存的低影响行为，以及有界任务奖励——取代单一的不透明奖励，并通过严格的权重差距进行析列组合。定理1证明了在部分可观测的关机游戏中单轮矫正性的可证明性；定理3将保证扩展到多步、自我生成的代理，即使每个头在均方误差ε下被学习，规划器ε次最优，违反任何安全属性的概率仍被限制，同时仍然确保净人类收益。与宪法型AI或RLHF/RLAIF将所有规范合并为一个学习标量不同，我们的分离使得即使在激励冲突时，遵从性和影响限制仍然占主导地位。对于对手可以修改代理的开放环境中，我们通过归约到停机问题证明了决定任意被攻击后代理是否会违反矫正性问题是不可判定的，然后划出了一个有限期的“可判定岛屿”，在随机多项式时间内进行安全性验证，并使用隐私保护的、常数轮次零知识证明进行验证。因此，剩下的挑战是常规的机器学习任务——数据覆盖和泛化：奖励作弊风险被推入评估质量而非隐藏激励的泄露，为当今的LLM助手和未来的自主系统提供了更清晰的实施指导。 

---
# On the Limits of Hierarchically Embedded Logic in Classical Neural Networks 

**Title (ZH)**: 经典神经网络中层次嵌套逻辑的局限性 

**Authors**: Bill Cochran  

**Link**: [PDF](https://arxiv.org/pdf/2507.20960)  

**Abstract**: We propose a formal model of reasoning limitations in large neural net models for language, grounded in the depth of their neural architecture. By treating neural networks as linear operators over logic predicate space we show that each layer can encode at most one additional level of logical reasoning. We prove that a neural network of depth a particular depth cannot faithfully represent predicates in a one higher order logic, such as simple counting over complex predicates, implying a strict upper bound on logical expressiveness. This structure induces a nontrivial null space during tokenization and embedding, excluding higher-order predicates from representability. Our framework offers a natural explanation for phenomena such as hallucination, repetition, and limited planning, while also providing a foundation for understanding how approximations to higher-order logic may emerge. These results motivate architectural extensions and interpretability strategies in future development of language models. 

**Abstract (ZH)**: 我们提出了一个基于神经网络架构深度的自然语言大规模神经网络推理限制的正式模型。通过将神经网络视为逻辑谓词空间上的线性算子，我们证明每一层最多只能编码一个额外的逻辑推理层次。我们证明了一个特定深度的神经网络无法忠实表示更高阶逻辑中的谓词，比如复杂谓词的简单计数，从而对逻辑表达能力施加了一个严格的上限。这种结构会在分词和嵌入过程中诱导出一个非平凡的零空间，排除了更高阶谓词的可表示性。我们的框架为解释幻觉、重复以及有限规划等现象提供了自然的解释，同时也为理解更高阶逻辑近似如何出现提供了基础。这些结果促使我们未来在语言模型开发中探索架构扩展和可解释性策略。 

---
# Partially Observable Monte-Carlo Graph Search 

**Title (ZH)**: 部分可观测蒙特卡罗图搜索 

**Authors**: Yang You, Vincent Thomas, Alex Schutz, Robert Skilton, Nick Hawes, Olivier Buffet  

**Link**: [PDF](https://arxiv.org/pdf/2507.20951)  

**Abstract**: Currently, large partially observable Markov decision processes (POMDPs) are often solved by sampling-based online methods which interleave planning and execution phases. However, a pre-computed offline policy is more desirable in POMDP applications with time or energy constraints. But previous offline algorithms are not able to scale up to large POMDPs. In this article, we propose a new sampling-based algorithm, the partially observable Monte-Carlo graph search (POMCGS) to solve large POMDPs offline. Different from many online POMDP methods, which progressively develop a tree while performing (Monte-Carlo) simulations, POMCGS folds this search tree on the fly to construct a policy graph, so that computations can be drastically reduced, and users can analyze and validate the policy prior to embedding and executing it. Moreover, POMCGS, together with action progressive widening and observation clustering methods provided in this article, is able to address certain continuous POMDPs. Through experiments, we demonstrate that POMCGS can generate policies on the most challenging POMDPs, which cannot be computed by previous offline algorithms, and these policies' values are competitive compared with the state-of-the-art online POMDP algorithms. 

**Abstract (ZH)**: 当前，大型部分可观测马尔可夫决策过程（POMDP）通常通过交错规划和执行阶段的基于采样的在线方法来求解。然而，在具有时间或能量约束的POMDP应用中，预先计算的离线策略更有优势。但之前的离线算法无法扩展到大型POMDP。在本文中，我们提出了一种新的基于采样的算法——部分可观测蒙特卡洛图搜索（POMCGS），用于解决大型POMDP的离线问题。与许多在线POMDP方法不同，POMCGS 边执行（蒙特卡洛）模拟边实时折叠搜索树以构建策略图，从而大幅减少计算量，并使用户能够在嵌入和执行策略之前对其进行分析和验证。此外，通过提供动作逐步放宽和观测聚类方法，POMCGS 能够解决某些连续POMDP。实验结果表明，POMCGS 能够生成前人离线算法无法计算的最具挑战性的POMDP的策略，且这些策略的价值与最新的在线POMDP算法相当。 

---
# MMGraphRAG: Bridging Vision and Language with Interpretable Multimodal Knowledge Graphs 

**Title (ZH)**: MMGraphRAG: 通过可解释的多模态知识图谱连接视觉与语言 

**Authors**: Xueyao Wan, Hang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.20804)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances language model generation by retrieving relevant information from external knowledge bases. However, conventional RAG methods face the issue of missing multimodal information. Multimodal RAG methods address this by fusing images and text through mapping them into a shared embedding space, but they fail to capture the structure of knowledge and logical chains between modalities. Moreover, they also require large-scale training for specific tasks, resulting in limited generalizing ability. To address these limitations, we propose MMGraphRAG, which refines visual content through scene graphs and constructs a multimodal knowledge graph (MMKG) in conjunction with text-based KG. It employs spectral clustering to achieve cross-modal entity linking and retrieves context along reasoning paths to guide the generative process. Experimental results show that MMGraphRAG achieves state-of-the-art performance on the DocBench and MMLongBench datasets, demonstrating strong domain adaptability and clear reasoning paths. 

**Abstract (ZH)**: 基于场景图的多模态图增强生成（MMGraphRAG）通过 refine 视觉内容并结合基于文本的知识图构建多模态知识图，以解决常规 Retrieval-Augmented Generation (RAG) 方法中存在的多模态信息缺失问题。 

---
# evalSmarT: An LLM-Based Framework for Evaluating Smart Contract Generated Comments 

**Title (ZH)**: evalSmarT: 一种基于大语言模型的智能合约生成注释评估框架 

**Authors**: Fatou Ndiaye Mbodji  

**Link**: [PDF](https://arxiv.org/pdf/2507.20774)  

**Abstract**: Smart contract comment generation has gained traction as a means to improve code comprehension and maintainability in blockchain systems. However, evaluating the quality of generated comments remains a challenge. Traditional metrics such as BLEU and ROUGE fail to capture domain-specific nuances, while human evaluation is costly and unscalable. In this paper, we present \texttt{evalSmarT}, a modular and extensible framework that leverages large language models (LLMs) as evaluators. The system supports over 400 evaluator configurations by combining approximately 40 LLMs with 10 prompting strategies. We demonstrate its application in benchmarking comment generation tools and selecting the most informative outputs. Our results show that prompt design significantly impacts alignment with human judgment, and that LLM-based evaluation offers a scalable and semantically rich alternative to existing methods. 

**Abstract (ZH)**: 智能合约注释生成在提高区块链系统代码理解和维护性方面获得了关注。然而，评估生成注释的质量仍然是一个挑战。传统的评估指标如BLEU和ROUGE未能捕捉到领域特定的细微差异，而人力评估则成本高昂且难以扩展。本文提出了一种模块化和可扩展的框架\texttt{evalSmarT}，该框架利用大规模语言模型（LLMs）作为评估工具。该系统通过结合大约40个LLM和10种启 tắm策略，支持超过400种评估配置。我们展示了该框架在评估注释生成工具和选择最具信息量的输出结果方面的应用。实验结果表明，启 tắm设计对与人类判断的一致性有显著影响，并且基于LLM的评估提供了比现有方法更具扩展性和语义丰富性的替代方案。 

---
# How Chain-of-Thought Works? Tracing Information Flow from Decoding, Projection, and Activation 

**Title (ZH)**: 链式思考是如何工作的？追踪从解码、投影到激活的信息流 

**Authors**: Hao Yang, Qinghua Zhao, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.20758)  

**Abstract**: Chain-of-Thought (CoT) prompting significantly enhances model reasoning, yet its internal mechanisms remain poorly understood. We analyze CoT's operational principles by reversely tracing information flow across decoding, projection, and activation phases. Our quantitative analysis suggests that CoT may serve as a decoding space pruner, leveraging answer templates to guide output generation, with higher template adherence strongly correlating with improved performance. Furthermore, we surprisingly find that CoT modulates neuron engagement in a task-dependent manner: reducing neuron activation in open-domain tasks, yet increasing it in closed-domain scenarios. These findings offer a novel mechanistic interpretability framework and critical insights for enabling targeted CoT interventions to design more efficient and robust prompts. We released our code and data at this https URL. 

**Abstract (ZH)**: Chain-of-Thought (CoT) 提示显著增强了模型推理能力，但其内部机制仍不完全理解。我们通过反向追踪解码、投影和激活阶段的信息流来分析 CoT 的运作原理。我们的量化分析表明，CoT 可能充当一个解码空间剪枝器，利用答案模板引导输出生成，且模板的高遵从性与性能的提升有密切关系。此外，我们惊讶地发现，CoT 以任务依赖的方式调节神经元参与：在开放式任务中减少神经元激活，在封闭式任务中增加神经元激活。这些发现提供了一种新的机制可解释性框架，并为设计更高效和稳健的提示提供了关键见解。我们已在此网址发布了我们的代码和数据：this https URL。 

---
# Beyond Listenership: AI-Predicted Interventions Drive Improvements in Maternal Health Behaviours 

**Title (ZH)**: 超越听众效应：AI 预测干预促进孕产妇健康行为改善 

**Authors**: Arpan Dasgupta, Sarvesh Gharat, Neha Madhiwalla, Aparna Hegde, Milind Tambe, Aparna Taneja  

**Link**: [PDF](https://arxiv.org/pdf/2507.20755)  

**Abstract**: Automated voice calls with health information are a proven method for disseminating maternal and child health information among beneficiaries and are deployed in several programs around the world. However, these programs often suffer from beneficiary dropoffs and poor engagement. In previous work, through real-world trials, we showed that an AI model, specifically a restless bandit model, could identify beneficiaries who would benefit most from live service call interventions, preventing dropoffs and boosting engagement. However, one key question has remained open so far: does such improved listenership via AI-targeted interventions translate into beneficiaries' improved knowledge and health behaviors? We present a first study that shows not only listenership improvements due to AI interventions, but also simultaneously links these improvements to health behavior changes. Specifically, we demonstrate that AI-scheduled interventions, which enhance listenership, lead to statistically significant improvements in beneficiaries' health behaviors such as taking iron or calcium supplements in the postnatal period, as well as understanding of critical health topics during pregnancy and infancy. This underscores the potential of AI to drive meaningful improvements in maternal and child health. 

**Abstract (ZH)**: 基于AI的自动语音呼叫在传递 maternal和child健康信息方面的应用及其对受益者健康行为的影响研究 

---
# Learning the Value Systems of Societies from Preferences 

**Title (ZH)**: 从偏好中学习社会的价值系统 

**Authors**: Andrés Holgado-Sánchez, Holger Billhardt, Sascha Ossowski, Sara Degli-Esposti  

**Link**: [PDF](https://arxiv.org/pdf/2507.20728)  

**Abstract**: Aligning AI systems with human values and the value-based preferences of various stakeholders (their value systems) is key in ethical AI. In value-aware AI systems, decision-making draws upon explicit computational representations of individual values (groundings) and their aggregation into value systems. As these are notoriously difficult to elicit and calibrate manually, value learning approaches aim to automatically derive computational models of an agent's values and value system from demonstrations of human behaviour. Nonetheless, social science and humanities literature suggest that it is more adequate to conceive the value system of a society as a set of value systems of different groups, rather than as the simple aggregation of individual value systems. Accordingly, here we formalize the problem of learning the value systems of societies and propose a method to address it based on heuristic deep clustering. The method learns socially shared value groundings and a set of diverse value systems representing a given society by observing qualitative value-based preferences from a sample of agents. We evaluate the proposal in a use case with real data about travelling decisions. 

**Abstract (ZH)**: 使AI系统与人类价值观及各相关方的价值观基础偏好对齐是伦理AI的关键。在价值意识AI系统中，决策依赖于个体价值观的明确计算表示（根基）及其聚合为价值系统。由于这些价值观难以手动获取和校准，因此价值学习方法旨在从人类行为示范中自动推导出代理的价值及其价值系统的计算模型。然而，社会科学和人文科学研究表明，社会的价值系统更适合作为不同群体价值系统的集合，而不是简单聚合的个体价值系统。据此，我们正式化了学习社会价值系统的难题，并提出了一种基于启发式深度聚类的方法。该方法通过观察一组代理的定性价值基础偏好来学习社会共享的价值根基和一组代表给定社会的多样化价值系统。我们在一个实际旅行决策使用案例中评估了该方法。 

---
# Algorithmic Fairness: A Runtime Perspective 

**Title (ZH)**: 算法公平性：运行时视角 

**Authors**: Filip Cano, Thomas A. Henzinger, Konstantin Kueffner  

**Link**: [PDF](https://arxiv.org/pdf/2507.20711)  

**Abstract**: Fairness in AI is traditionally studied as a static property evaluated once, over a fixed dataset. However, real-world AI systems operate sequentially, with outcomes and environments evolving over time. This paper proposes a framework for analysing fairness as a runtime property. Using a minimal yet expressive model based on sequences of coin tosses with possibly evolving biases, we study the problems of monitoring and enforcing fairness expressed in either toss outcomes or coin biases. Since there is no one-size-fits-all solution for either problem, we provide a summary of monitoring and enforcement strategies, parametrised by environment dynamics, prediction horizon, and confidence thresholds. For both problems, we present general results under simple or minimal assumptions. We survey existing solutions for the monitoring problem for Markovian and additive dynamics, and existing solutions for the enforcement problem in static settings with known dynamics. 

**Abstract (ZH)**: AI中的公平性 traditionally studied as a static property evaluated once over a fixed dataset, is proposed to be analyzed as a runtime property in sequentially operating real-world AI systems. 

---
# A General Framework for Dynamic MAPF using Multi-Shot ASP and Tunnels 

**Title (ZH)**: 动态MAPF问题的通用框架：基于多轮ASP和隧道的方法 

**Authors**: Aysu Bogatarkan, Esra Erdem  

**Link**: [PDF](https://arxiv.org/pdf/2507.20703)  

**Abstract**: MAPF problem aims to find plans for multiple agents in an environment within a given time, such that the agents do not collide with each other or obstacles. Motivated by the execution and monitoring of these plans, we study Dynamic MAPF (D-MAPF) problem, which allows changes such as agents entering/leaving the environment or obstacles being removed/moved. Considering the requirements of real-world applications in warehouses with the presence of humans, we introduce 1) a general definition for D-MAPF (applicable to variations of D-MAPF), 2) a new framework to solve D-MAPF (utilizing multi-shot computation, and allowing different methods to solve D-MAPF), and 3) a new ASP-based method to solve D-MAPF (combining advantages of replanning and repairing methods, with a novel concept of tunnels to specify where agents can move). We have illustrated the strengths and weaknesses of this method by experimental evaluations, from the perspectives of computational performance and quality of solutions. 

**Abstract (ZH)**: 多代理路径规划（MAPF）问题旨在在一个给定时间内为多个代理在环境中寻找计划，使得代理之间或与障碍物之间不发生碰撞。受执行和监控这些计划的需求启发，我们研究动态多代理路径规划（D-MAPF）问题，该问题允许代理进入/离开环境或障碍物被移除/移动。考虑到仓库中存在人类的需求，我们引入了1）一种通用的D-MAPF定义（适用于D-MAPF的各种变体），2）一种新的框架来解决D-MAPF（利用多轮计算，并允许使用不同的方法解决D-MAPF），以及3）一种新的基于ASP的方法来解决D-MAPF（结合重规划和修复方法的优势，并引入新的概念“隧道”来指定代理可以移动的位置）。我们从计算性能和解决方案质量的角度通过实验评估展示了该方法的优点和不足。 

---
# Adaptive Fuzzy Time Series Forecasting via Partially Asymmetric Convolution and Sub-Sliding Window Fusion 

**Title (ZH)**: 基于部分非对称卷积和子滑动窗口融合的自适应模糊时间序列预测 

**Authors**: Lijian Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.20641)  

**Abstract**: At present, state-of-the-art forecasting models are short of the ability to capture spatio-temporal dependency and synthesize global information at the stage of learning. To address this issue, in this paper, through the adaptive fuzzified construction of temporal data, we propose a novel convolutional architecture with partially asymmetric design based on the scheme of sliding window to realize accurate time series forecasting. First, the construction strategy of traditional fuzzy time series is improved to further extract short and long term temporal interrelation, which enables every time node to automatically possess corresponding global information and inner relationships among them in a restricted sliding window and the process does not require human involvement. Second, a bilateral Atrous algorithm is devised to reduce calculation demand of the proposed model without sacrificing global characteristics of elements. And it also allows the model to avoid processing redundant information. Third, after the transformation of time series, a partially asymmetric convolutional architecture is designed to more flexibly mine data features by filters in different directions on feature maps, which gives the convolutional neural network (CNN) the ability to construct sub-windows within existing sliding windows to model at a more fine-grained level. And after obtaining the time series information at different levels, the multi-scale features from different sub-windows will be sent to the corresponding network layer for time series information fusion. Compared with other competitive modern models, the proposed method achieves state-of-the-art results on most of popular time series datasets, which is fully verified by the experimental results. 

**Abstract (ZH)**: 基于滑窗部分非对称设计的自适应模糊化时序预测新型卷积架构 

---
# Complementarity-driven Representation Learning for Multi-modal Knowledge Graph Completion 

**Title (ZH)**: 基于互补性的表示学习以实现多模态知识图谱 completion 

**Authors**: Lijian Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.20620)  

**Abstract**: Multi-modal Knowledge Graph Completion (MMKGC) aims to uncover hidden world knowledge in multimodal knowledge graphs by leveraging both multimodal and structural entity information. However, the inherent imbalance in multimodal knowledge graphs, where modality distributions vary across entities, poses challenges in utilizing additional modality data for robust entity representation. Existing MMKGC methods typically rely on attention or gate-based fusion mechanisms but overlook complementarity contained in multi-modal data. In this paper, we propose a novel framework named Mixture of Complementary Modality Experts (MoCME), which consists of a Complementarity-guided Modality Knowledge Fusion (CMKF) module and an Entropy-guided Negative Sampling (EGNS) mechanism. The CMKF module exploits both intra-modal and inter-modal complementarity to fuse multi-view and multi-modal embeddings, enhancing representations of entities. Additionally, we introduce an Entropy-guided Negative Sampling mechanism to dynamically prioritize informative and uncertain negative samples to enhance training effectiveness and model robustness. Extensive experiments on five benchmark datasets demonstrate that our MoCME achieves state-of-the-art performance, surpassing existing approaches. 

**Abstract (ZH)**: 多模态知识图完成（MMKGC）旨在通过利用多模态和结构性实体信息来揭示多模态知识图中的隐藏世界知识。然而，多模态知识图中固有的不平衡性，即不同实体之间模态分布的差异，对利用额外模态数据进行稳健实体表示提出了挑战。现有的MMKGC方法通常依赖于注意力或门控融合机制，但忽略了多模态数据中包含的互补性。本文提出了一种名为互补模态专家混合（MoCME）的新型框架，该框架包括一种互补性引导的模态知识融合（CMKF）模块和一种熵引导的负采样（EGNS）机制。CMKF模块利用内在模态和跨模态互补性来融合多视图和多模态嵌入，增强实体表示。此外，我们引入了一种熵引导的负采样机制，以动态优先选择信息性和不确定性的负样本，从而增强训练效果和模型鲁棒性。在五个基准数据集上的广泛实验表明，我们的MoCME达到了最先进的性能，超越了现有方法。 

---
# Enhancing Large Multimodal Models with Adaptive Sparsity and KV Cache Compression 

**Title (ZH)**: 增强大尺寸多模态模型的自适应稀疏性和KV缓存压缩 

**Authors**: Te Zhang, Yuheng Li, Junxiang Wang, Lujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.20613)  

**Abstract**: Large multimodal models (LMMs) have advanced significantly by integrating visual encoders with extensive language models, enabling robust reasoning capabilities. However, compressing LMMs for deployment on edge devices remains a critical challenge. In this work, we propose an adaptive search algorithm that optimizes sparsity and KV cache compression to enhance LMM efficiency. Utilizing the Tree-structured Parzen Estimator, our method dynamically adjusts pruning ratios and KV cache quantization bandwidth across different LMM layers, using model performance as the optimization objective. This approach uniquely combines pruning with key-value cache quantization and incorporates a fast pruning technique that eliminates the need for additional fine-tuning or weight adjustments, achieving efficient compression without compromising accuracy. Comprehensive evaluations on benchmark datasets, including LLaVA-1.5 7B and 13B, demonstrate our method superiority over state-of-the-art techniques such as SparseGPT and Wanda across various compression levels. Notably, our framework automatic allocation of KV cache compression resources sets a new standard in LMM optimization, delivering memory efficiency without sacrificing much performance. 

**Abstract (ZH)**: 大规模多模态模型（LMMs）通过将视觉编码器与广泛的语言模型相结合，显著提升了推理能力。然而，将LMMs压缩以在边缘设备上部署仍然是一个关键挑战。在本文中，我们提出了一种自适应搜索算法，优化稀疏性和KV缓存压缩，以提高LMM效率。利用树结构Parzen估计算法，我们的方法在不同LMM层中动态调整剪枝比率和KV缓存量化带宽，并以模型性能作为优化目标。该方法独特地结合了剪枝和键值缓存量化，并引入了一种快速剪枝技术，无需额外的微调或权重调整，实现了高效压缩而不牺牲准确性。对包括LLaVA-1.5 7B和13B在内的基准数据集的全面评估表明，我们的方法在各种压缩级别上优于SparseGPT和Wanda等最新技术。值得注意的是，我们的框架在LMM优化中的自动分配KV缓存压缩资源树立了新标准，实现了内存效率而不牺牲太多性能。 

---
# Unlearning of Knowledge Graph Embedding via Preference Optimization 

**Title (ZH)**: 知识图嵌入的反学习通过偏好优化 

**Authors**: Jiajun Liu, Wenjun Ke, Peng Wang, Yao He, Ziyu Shang, Guozheng Li, Zijie Xu, Ke Ji  

**Link**: [PDF](https://arxiv.org/pdf/2507.20566)  

**Abstract**: Existing knowledge graphs (KGs) inevitably contain outdated or erroneous knowledge that needs to be removed from knowledge graph embedding (KGE) models. To address this challenge, knowledge unlearning can be applied to eliminate specific information while preserving the integrity of the remaining knowledge in KGs. Existing unlearning methods can generally be categorized into exact unlearning and approximate unlearning. However, exact unlearning requires high training costs while approximate unlearning faces two issues when applied to KGs due to the inherent connectivity of triples: (1) It fails to fully remove targeted information, as forgetting triples can still be inferred from remaining ones. (2) It focuses on local data for specific removal, which weakens the remaining knowledge in the forgetting boundary. To address these issues, we propose GraphDPO, a novel approximate unlearning framework based on direct preference optimization (DPO). Firstly, to effectively remove forgetting triples, we reframe unlearning as a preference optimization problem, where the model is trained by DPO to prefer reconstructed alternatives over the original forgetting triples. This formulation penalizes reliance on forgettable knowledge, mitigating incomplete forgetting caused by KG connectivity. Moreover, we introduce an out-boundary sampling strategy to construct preference pairs with minimal semantic overlap, weakening the connection between forgetting and retained knowledge. Secondly, to preserve boundary knowledge, we introduce a boundary recall mechanism that replays and distills relevant information both within and across time steps. We construct eight unlearning datasets across four popular KGs with varying unlearning rates. Experiments show that GraphDPO outperforms state-of-the-art baselines by up to 10.1% in MRR_Avg and 14.0% in MRR_F1. 

**Abstract (ZH)**: 现有的知识图谱不可避免地包含过时或错误的知识，需要从知识图谱嵌入（KGE）模型中移除。为应对这一挑战，可以通过知识遗忘技术来消除特定信息，同时保持知识图谱中剩余知识的完整性。现有的遗忘方法可以一般性地归类为精确遗忘和近似遗忘。然而，精确遗忘需要高昂的训练成本，而近似遗忘在应用于知识图谱时由于三元组的内在关联性面临两个问题：（1）难以完全移除目标信息，因为可以通过剩余的三元组推断出被遗忘的三元组。 （2）它侧重于局部数据进行特定移除，这会在遗忘边界弱化剩余知识。为解决这些问题，我们提出了一种基于直接偏好优化（DPO）的近似遗忘框架GraphDPO。首先，为了有效移除被遗忘的三元组，我们将遗忘重新定义为一个偏好优化问题，通过DPO训练模型以偏好重构的替代方案而非原始被遗忘的三元组。这种形式通过惩罚依赖遗忘知识来减轻由于知识图谱连接性导致的不完全遗忘。此外，我们引入了一种边界外采样策略，以最小的语义重叠构建偏好对，从而减弱遗忘与保留知识之间的联系。其次，为了保持边界知识，我们引入了一个边界召回机制，在时间和跨时间步内部和外部回放和提炼相关信息。我们在四个流行的知识图谱上构建了八个具有不同遗忘率的遗忘数据集。实验结果显示，在MRR_Avg方面，GraphDPO比最先进的基线高出10.1%，在MRR_F1方面高出14.0%。 

---
# MeLA: A Metacognitive LLM-Driven Architecture for Automatic Heuristic Design 

**Title (ZH)**: 元LMA：一种元认知大规模语言模型驱动的自动启发式设计架构 

**Authors**: Zishang Qiu, Xinan Chen, Long Chen, Ruibin Bai  

**Link**: [PDF](https://arxiv.org/pdf/2507.20541)  

**Abstract**: This paper introduces MeLA, a Metacognitive LLM-Driven Architecture that presents a new paradigm for Automatic Heuristic Design (AHD). Traditional evolutionary methods operate directly on heuristic code; in contrast, MeLA evolves the instructional prompts used to guide a Large Language Model (LLM) in generating these heuristics. This process of "prompt evolution" is driven by a novel metacognitive framework where the system analyzes performance feedback to systematically refine its generative strategy. MeLA's architecture integrates a problem analyzer to construct an initial strategic prompt, an error diagnosis system to repair faulty code, and a metacognitive search engine that iteratively optimizes the prompt based on heuristic effectiveness. In comprehensive experiments across both benchmark and real-world problems, MeLA consistently generates more effective and robust heuristics, significantly outperforming state-of-the-art methods. Ultimately, this research demonstrates the profound potential of using cognitive science as a blueprint for AI architecture, revealing that by enabling an LLM to metacognitively regulate its problem-solving process, we unlock a more robust and interpretable path to AHD. 

**Abstract (ZH)**: MeLA：元认知大模型驱动的自动启发式设计架构 

---
# Security Challenges in AI Agent Deployment: Insights from a Large Scale Public Competition 

**Title (ZH)**: AI代理部署中的安全挑战：大规模公共竞赛的见解 

**Authors**: Andy Zou, Maxwell Lin, Eliot Jones, Micha Nowak, Mateusz Dziemian, Nick Winter, Alexander Grattan, Valent Nathanael, Ayla Croft, Xander Davies, Jai Patel, Robert Kirk, Nate Burnikell, Yarin Gal, Dan Hendrycks, J. Zico Kolter, Matt Fredrikson  

**Link**: [PDF](https://arxiv.org/pdf/2507.20526)  

**Abstract**: Recent advances have enabled LLM-powered AI agents to autonomously execute complex tasks by combining language model reasoning with tools, memory, and web access. But can these systems be trusted to follow deployment policies in realistic environments, especially under attack? To investigate, we ran the largest public red-teaming competition to date, targeting 22 frontier AI agents across 44 realistic deployment scenarios. Participants submitted 1.8 million prompt-injection attacks, with over 60,000 successfully eliciting policy violations such as unauthorized data access, illicit financial actions, and regulatory noncompliance. We use these results to build the Agent Red Teaming (ART) benchmark - a curated set of high-impact attacks - and evaluate it across 19 state-of-the-art models. Nearly all agents exhibit policy violations for most behaviors within 10-100 queries, with high attack transferability across models and tasks. Importantly, we find limited correlation between agent robustness and model size, capability, or inference-time compute, suggesting that additional defenses are needed against adversarial misuse. Our findings highlight critical and persistent vulnerabilities in today's AI agents. By releasing the ART benchmark and accompanying evaluation framework, we aim to support more rigorous security assessment and drive progress toward safer agent deployment. 

**Abstract (ZH)**: 近期进展使由大规模语言模型驱动的AI代理能够通过结合语言模型推理、工具、内存和网络访问来自主执行复杂任务。但这些系统在实际环境中能否被信任遵守部署政策，特别是在遭受攻击的情况下？为进行研究，我们举办了迄今为止最大规模的公开红队竞赛，针对44个现实部署场景下的22个前沿AI代理进行攻击。参与者提交了180万次提示注入攻击，其中超过6万次成功引发了未授权数据访问、非法财务操作和合规性违规等政策违规。我们使用这些结果构建了代理红队基准（ART基准）——一个高影响攻击的精选集——并跨19个最先进的模型对其进行评估。几乎所有代理在10-100次查询内对大多数行为均表现出政策违规，且攻击在模型和任务之间具有高转移性。重要的是，我们发现代理的鲁棒性与模型规模、能力或推理时计算量之间存在有限的相关性，表明需要针对对抗性滥用采取额外防御措施。我们的研究结果突显了当前AI代理中存在的关键且持续的漏洞。通过发布ART基准和配套的评估框架，我们旨在支持更严谨的安全评估，并推动更安全代理部署的进步。 

---
# STARN-GAT: A Multi-Modal Spatio-Temporal Graph Attention Network for Accident Severity Prediction 

**Title (ZH)**: STARN-GAT：一种用于事故严重程度预测的多模态时空图注意力网络 

**Authors**: Pritom Ray Nobin, Imran Ahammad Rifat  

**Link**: [PDF](https://arxiv.org/pdf/2507.20451)  

**Abstract**: Accurate prediction of traffic accident severity is critical for improving road safety, optimizing emergency response strategies, and informing the design of safer transportation infrastructure. However, existing approaches often struggle to effectively model the intricate interdependencies among spatial, temporal, and contextual variables that govern accident outcomes. In this study, we introduce STARN-GAT, a Multi-Modal Spatio-Temporal Graph Attention Network, which leverages adaptive graph construction and modality-aware attention mechanisms to capture these complex relationships. Unlike conventional methods, STARN-GAT integrates road network topology, temporal traffic patterns, and environmental context within a unified attention-based framework. The model is evaluated on the Fatality Analysis Reporting System (FARS) dataset, achieving a Macro F1-score of 85 percent, ROC-AUC of 0.91, and recall of 81 percent for severe incidents. To ensure generalizability within the South Asian context, STARN-GAT is further validated on the ARI-BUET traffic accident dataset, where it attains a Macro F1-score of 0.84, recall of 0.78, and ROC-AUC of 0.89. These results demonstrate the model's effectiveness in identifying high-risk cases and its potential for deployment in real-time, safety-critical traffic management systems. Furthermore, the attention-based architecture enhances interpretability, offering insights into contributing factors and supporting trust in AI-assisted decision-making. Overall, STARN-GAT bridges the gap between advanced graph neural network techniques and practical applications in road safety analytics. 

**Abstract (ZH)**: 准确预测交通事故严重程度对于提高道路安全、优化应急响应策略和指导更安全的交通基础设施设计至关重要。然而，现有方法往往难以有效地建模空间、时间及上下文变量之间复杂的相互依赖关系。在本研究中，我们引入了STARN-GAT，一种多模态时空图注意网络，该模型利用自适应图构建和模态感知注意机制来捕捉这些复杂的关系。与传统方法不同，STARN-GAT将道路网络拓扑、时间交通模式和环境上下文集成在一个统一的基于注意力的框架中。该模型在致命性事故报告系统（FARS）数据集上进行评估，实现了宏F1分数85%，ROC-AUC 0.91，严重事件召回率81%。为进一步确保在南亚地区的普适性，STARN-GAT在ARI-BUET交通事故数据集上进一步验证，实现了宏F1分数0.84，召回率0.78，ROC-AUC 0.89。这些结果表明该模型在识别高风险案例方面的有效性及其在实时、关键安全交通管理系统中部署的潜力。此外，基于注意力的架构提高了模型的解释性，提供了影响因素的见解并支持对AI辅助决策的信任。总之，STARN-GAT填补了先进的图神经网络技术和道路安全分析实际应用之间的差距。 

---
# Enhancing QoS in Edge Computing through Federated Layering Techniques: A Pathway to Resilient AI Lifelong Learning Systems 

**Title (ZH)**: 通过联邦分层技术提升边缘计算服务质量：通往稳健的终身学习AI系统的途径 

**Authors**: Chengzhuo Han  

**Link**: [PDF](https://arxiv.org/pdf/2507.20444)  

**Abstract**: In the context of the rapidly evolving information technology landscape, marked by the advent of 6G communication networks, we face an increased data volume and complexity in network environments. This paper addresses these challenges by focusing on Quality of Service (QoS) in edge computing frameworks. We propose a novel approach to enhance QoS through the development of General Artificial Intelligence Lifelong Learning Systems, with a special emphasis on Federated Layering Techniques (FLT). Our work introduces a federated layering-based small model collaborative mechanism aimed at improving AI models' operational efficiency and response time in environments where resources are limited. This innovative method leverages the strengths of cloud and edge computing, incorporating a negotiation and debate mechanism among small AI models to enhance reasoning and decision-making processes. By integrating model layering techniques with privacy protection measures, our approach ensures the secure transmission of model parameters while maintaining high efficiency in learning and reasoning capabilities. The experimental results demonstrate that our strategy not only enhances learning efficiency and reasoning accuracy but also effectively protects the privacy of edge nodes. This presents a viable solution for achieving resilient large model lifelong learning systems, with a significant improvement in QoS for edge computing environments. 

**Abstract (ZH)**: 在6G通信网络迅猛发展的信息技术背景下，网络环境中面临的数据量和复杂性增加。本文通过关注边缘计算框架中的服务质量（QoS）来应对这些挑战。我们提出了一种通过开发通用人工智能终身学习系统来提升QoS的新方法，特别强调了联邦分层技术（FLT）。我们的工作引入了一种基于联邦分层的小模型协作机制，旨在在资源有限的环境中提高AI模型的运行效率和响应时间。该创新方法结合了云和边缘计算的优势，并通过小AI模型之间的谈判和辩论机制增强推理和决策过程。通过将模型分层技术与隐私保护措施相结合，我们的方法确保了模型参数的安全传输，并在学习和推理能力方面保持高效。实验结果表明，我们的策略不仅提高了学习效率和推理准确性，还有效保护了边缘节点的隐私。这为实现具有显著QoS改进的健壮的大模型终身学习系统提供了一个可行的解决方案。 

---
# MazeEval: A Benchmark for Testing Sequential Decision-Making in Language Models 

**Title (ZH)**: MazeEval：语言模型顺序决策能力测试基准 

**Authors**: Hafsteinn Einarsson  

**Link**: [PDF](https://arxiv.org/pdf/2507.20395)  

**Abstract**: As Large Language Models (LLMs) increasingly power autonomous agents in robotics and embodied AI, understanding their spatial reasoning capabilities becomes crucial for ensuring reliable real-world deployment. Despite advances in language understanding, current research lacks evaluation of how LLMs perform spatial navigation without visual cues, a fundamental requirement for agents operating with limited sensory information. This paper addresses this gap by introducing MazeEval, a benchmark designed to isolate and evaluate pure spatial reasoning in LLMs through coordinate-based maze navigation tasks. Our methodology employs a function-calling interface where models navigate mazes of varying complexity ($5\times 5$ to $15\times 15$ grids) using only coordinate feedback and distance-to-wall information, excluding visual input to test fundamental spatial cognition. We evaluate eight state-of-the-art LLMs across identical mazes in both English and Icelandic to assess cross-linguistic transfer of spatial abilities. Our findings reveal striking disparities: while OpenAI's O3 achieves perfect navigation for mazes up to size $30\times 30$, other models exhibit catastrophic failure beyond $9\times 9$ mazes, with 100% of failures attributed to excessive looping behavior where models revisit a cell at least 10 times. We document a significant performance degradation in Icelandic, with models solving mazes 3-4 sizes smaller than in English, suggesting spatial reasoning in LLMs emerges from linguistic patterns rather than language-agnostic mechanisms. These results have important implications for global deployment of LLM-powered autonomous systems, showing spatial intelligence remains fundamentally constrained by training data availability and highlighting the need for architectural innovations to achieve reliable navigation across linguistic contexts. 

**Abstract (ZH)**: 大型语言模型（LLMs）在机器人学和具身AI中日益成为自主代理的动力，理解其空间推理能力对于确保其实用世界的可靠部署变得至关重要。尽管在语言理解方面取得了进步，当前的研究缺乏对LLMs在没有视觉提示的情况下进行空间导航性能的评估，这是对于具有有限感官信息的代理的基本要求。本文通过引入MazeEval基准，旨在通过基于坐标的迷宫导航任务来孤立和评估LLMs的纯空间推理能力，从而弥补这一空白。我们的方法采用了一个函数调用接口，模型仅通过坐标反馈和距离墙信息导航复杂程度不同的迷宫（5×5到15×15网格），不包括视觉输入以测试基础的空间认知能力。我们在相同的迷宫中用英语和冰岛语评估了八种最先进的LLMs，以评估其空间能力的跨语言迁移能力。我们的研究发现揭示了显著的差异：虽然OpenAI的O3在大小不超过30×30的迷宫中实现了完美的导航，但其他模型在9×9以上的迷宫中表现出灾难性的失败，其中100%的失败归因于过度循环行为，模型至少返回一个单元格10次以上。我们记录了在冰岛语中显著的性能下降，模型解决迷宫的规模比在英语中小3-4个等级，这表明LLMs的空间推理能力源自语言模式而非语言无关的机制。这些结果对LLM驱动的自主系统的全球部署具有重要的影响，表明空间智能仍然受到可用训练数据的极大限制，并强调了需要架构创新以实现跨语言上下文可靠导航的必要性。 

---
# Multi-Agent Reinforcement Learning for Dynamic Mobility Resource Allocation with Hierarchical Adaptive Grouping 

**Title (ZH)**: 多层次自适应分组的多代理强化学习动态移动资源分配 

**Authors**: Farshid Nooshi, Suining He  

**Link**: [PDF](https://arxiv.org/pdf/2507.20377)  

**Abstract**: Allocating mobility resources (e.g., shared bikes/e-scooters, ride-sharing vehicles) is crucial for rebalancing the mobility demand and supply in the urban environments. We propose in this work a novel multi-agent reinforcement learning named Hierarchical Adaptive Grouping-based Parameter Sharing (HAG-PS) for dynamic mobility resource allocation. HAG-PS aims to address two important research challenges regarding multi-agent reinforcement learning for mobility resource allocation: (1) how to dynamically and adaptively share the mobility resource allocation policy (i.e., how to distribute mobility resources) across agents (i.e., representing the regional coordinators of mobility resources); and (2) how to achieve memory-efficient parameter sharing in an urban-scale setting. To address the above challenges, we have provided following novel designs within HAG-PS. To enable dynamic and adaptive parameter sharing, we have designed a hierarchical approach that consists of global and local information of the mobility resource states (e.g., distribution of mobility resources). We have developed an adaptive agent grouping approach in order to split or merge the groups of agents based on their relative closeness of encoded trajectories (i.e., states, actions, and rewards). We have designed a learnable identity (ID) embeddings to enable agent specialization beyond simple parameter copy. We have performed extensive experimental studies based on real-world NYC bike sharing data (a total of more than 1.2 million trips), and demonstrated the superior performance (e.g., improved bike availability) of HAG-PS compared with other baseline approaches. 

**Abstract (ZH)**: 基于层次自适应分组的参数共享的多agents强化学习方法：动态移动资源分配 

---
# VLMPlanner: Integrating Visual Language Models with Motion Planning 

**Title (ZH)**: VLMPlanner：将视觉语言模型集成到运动规划中 

**Authors**: Zhipeng Tang, Sha Zhang, Jiajun Deng, Chenjie Wang, Guoliang You, Yuting Huang, Xinrui Lin, Yanyong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20342)  

**Abstract**: Integrating large language models (LLMs) into autonomous driving motion planning has recently emerged as a promising direction, offering enhanced interpretability, better controllability, and improved generalization in rare and long-tail scenarios. However, existing methods often rely on abstracted perception or map-based inputs, missing crucial visual context, such as fine-grained road cues, accident aftermath, or unexpected obstacles, which are essential for robust decision-making in complex driving environments. To bridge this gap, we propose VLMPlanner, a hybrid framework that combines a learning-based real-time planner with a vision-language model (VLM) capable of reasoning over raw images. The VLM processes multi-view images to capture rich, detailed visual information and leverages its common-sense reasoning capabilities to guide the real-time planner in generating robust and safe trajectories. Furthermore, we develop the Context-Adaptive Inference Gate (CAI-Gate) mechanism that enables the VLM to mimic human driving behavior by dynamically adjusting its inference frequency based on scene complexity, thereby achieving an optimal balance between planning performance and computational efficiency. We evaluate our approach on the large-scale, challenging nuPlan benchmark, with comprehensive experimental results demonstrating superior planning performance in scenarios with intricate road conditions and dynamic elements. Code will be available. 

**Abstract (ZH)**: 将大型语言模型（LLMs）集成到自主驾驶运动规划中， recently emerged as a promising direction，提供增强的可解释性、更好的可控性和在稀有和长尾场景中更好的泛化能力。然而，现有方法通常依赖于抽象的感知或基于地图的输入，缺乏关键的视觉上下文，如精细的道路提示、事故后果或意想不到的障碍物，这些都是在复杂驾驶环境中实现稳健决策所必需的。为了解决这一问题，我们提出了一种名为VLMPlanner的混合框架，该框架结合了一个基于学习的实时规划器和能够对原始图像进行推理的视觉语言模型（VLM）。VLM处理多视角图像以捕捉丰富的详细视觉信息，并利用其常识推理能力指导实时规划器生成稳健和安全的轨迹。此外，我们开发了上下文自适应推断门控（CAI-Gate）机制，该机制使VLM能够根据场景复杂度动态调整其推理频率，从而在规划性能与计算效率之间实现最优平衡。我们在大规模、具有挑战性的nuPlan基准上评估了我们的方法，全面的实验结果表明，该方法在复杂道路条件和动态元素场景中的规划性能更优。代码将开源。 

---
# The Blessing and Curse of Dimensionality in Safety Alignment 

**Title (ZH)**: 高维性之福与祸在安全对齐中的体现 

**Authors**: Rachel S.Y. Teo, Laziz U. Abdullaev, Tan M. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2507.20333)  

**Abstract**: The focus on safety alignment in large language models (LLMs) has increased significantly due to their widespread adoption across different domains. The scale of LLMs play a contributing role in their success, and the growth in parameter count follows larger hidden dimensions. In this paper, we hypothesize that while the increase in dimensions has been a key advantage, it may lead to emergent problems as well. These problems emerge as the linear structures in the activation space can be exploited, in the form of activation engineering, to circumvent its safety alignment. Through detailed visualizations of linear subspaces associated with different concepts, such as safety, across various model scales, we show that the curse of high-dimensional representations uniquely impacts LLMs. Further substantiating our claim, we demonstrate that projecting the representations of the model onto a lower dimensional subspace can preserve sufficient information for alignment while avoiding those linear structures. Empirical results confirm that such dimensional reduction significantly reduces susceptibility to jailbreaking through representation engineering. Building on our empirical validations, we provide theoretical insights into these linear jailbreaking methods relative to a model's hidden dimensions. Broadly speaking, our work posits that the high dimensions of a model's internal representations can be both a blessing and a curse in safety alignment. 

**Abstract (ZH)**: 大型语言模型中安全性对齐的安全关注在广泛采用不同领域后显著增加。随着模型规模的扩大，参数数量的增加也伴随着更大隐藏维度的增长。在本文中，我们假设虽然增加维度是一个关键优势，但也可能导致新出现的问题。这些问题源于激活空间中的线性结构可以通过激活工程被利用以绕过安全性对齐。通过详细可视化不同概念（如安全性）在各种模型规模下的线性子空间，我们展示了高维表示带来的诅咒对大型语言模型的独特影响。进一步支持我们的观点，我们证明将模型的表示投影到低维子空间可以在保留足够信息以进行对齐的同时避免这些线性结构。实验证据证实，这种维度的减少可以显著降低通过表示工程被破解的风险。基于我们的实验证明，我们提供了一种理论见解，适用于模型隐藏维度的这些线性破解方法。总体而言，我们的工作提出，模型内部表示的高维数在安全性对齐中既是恩赐也是诅咒。 

---
# Artificial Intelligence In Patent And Market Intelligence: A New Paradigm For Technology Scouting 

**Title (ZH)**: 人工智能在专利和市场情报中的应用：技术 scouting的新范式 

**Authors**: Manish Verma, Vivek Sharma, Vishal Singh  

**Link**: [PDF](https://arxiv.org/pdf/2507.20322)  

**Abstract**: This paper presents the development of an AI powered software platform that leverages advanced large language models (LLMs) to transform technology scouting and solution discovery in industrial R&D. Traditional approaches to solving complex research and development challenges are often time consuming, manually driven, and heavily dependent on domain specific expertise. These methods typically involve navigating fragmented sources such as patent repositories, commercial product catalogs, and competitor data, leading to inefficiencies and incomplete insights. The proposed platform utilizes cutting edge LLM capabilities including semantic understanding, contextual reasoning, and cross-domain knowledge extraction to interpret problem statements and retrieve high-quality, sustainable solutions. The system processes unstructured patent texts, such as claims and technical descriptions, and systematically extracts potential innovations aligned with the given problem context. These solutions are then algorithmically organized under standardized technical categories and subcategories to ensure clarity and relevance across interdisciplinary domains. In addition to patent analysis, the platform integrates commercial intelligence by identifying validated market solutions and active organizations addressing similar challenges. This combined insight sourced from both intellectual property and real world product data enables R&D teams to assess not only technical novelty but also feasibility, scalability, and sustainability. The result is a comprehensive, AI driven scouting engine that reduces manual effort, accelerates innovation cycles, and enhances decision making in complex R&D environments. 

**Abstract (ZH)**: 基于先进大语言模型的AI驱动软件平台在工业研发中的技术 scouting与解决方案发现开发 

---
# SciToolAgent: A Knowledge Graph-Driven Scientific Agent for Multi-Tool Integration 

**Title (ZH)**: SciToolAgent：一个知识图谱驱动的多工具集成科学代理 

**Authors**: Keyan Ding, Jing Yu, Junjie Huang, Yuchen Yang, Qiang Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.20280)  

**Abstract**: Scientific research increasingly relies on specialized computational tools, yet effectively utilizing these tools demands substantial domain expertise. While Large Language Models (LLMs) show promise in tool automation, they struggle to seamlessly integrate and orchestrate multiple tools for complex scientific workflows. Here, we present SciToolAgent, an LLM-powered agent that automates hundreds of scientific tools across biology, chemistry, and materials science. At its core, SciToolAgent leverages a scientific tool knowledge graph that enables intelligent tool selection and execution through graph-based retrieval-augmented generation. The agent also incorporates a comprehensive safety-checking module to ensure responsible and ethical tool usage. Extensive evaluations on a curated benchmark demonstrate that SciToolAgent significantly outperforms existing approaches. Case studies in protein engineering, chemical reactivity prediction, chemical synthesis, and metal-organic framework screening further demonstrate SciToolAgent's capability to automate complex scientific workflows, making advanced research tools accessible to both experts and non-experts. 

**Abstract (ZH)**: 基于大规模语言模型的科学工具代理 SciToolAgent：自动化多领域科学工作流 

---
# A Multi-Agent System for Information Extraction from the Chemical Literature 

**Title (ZH)**: 化学文献中信息提取的多代理系统 

**Authors**: Yufan Chen, Ching Ting Leung, Bowen Yu, Jianwei Sun, Yong Huang, Linyan Li, Hao Chen, Hanyu Gao  

**Link**: [PDF](https://arxiv.org/pdf/2507.20230)  

**Abstract**: To fully expedite AI-powered chemical research, high-quality chemical databases are the cornerstone. Automatic extraction of chemical information from the literature is essential for constructing reaction databases, but it is currently limited by the multimodality and style variability of chemical information. In this work, we developed a multimodal large language model (MLLM)-based multi-agent system for automatic chemical information extraction. We used the MLLM's strong reasoning capability to understand the structure of complex chemical graphics, decompose the extraction task into sub-tasks and coordinate a set of specialized agents to solve them. Our system achieved an F1 score of 80.8% on a benchmark dataset of complex chemical reaction graphics from the literature, surpassing the previous state-of-the-art model (F1 score: 35.6%) by a significant margin. Additionally, it demonstrated consistent improvements in key sub-tasks, including molecular image recognition, reaction image parsing, named entity recognition and text-based reaction extraction. This work is a critical step toward automated chemical information extraction into structured datasets, which will be a strong promoter of AI-driven chemical research. 

**Abstract (ZH)**: 基于多模态大型语言模型的多智能体系统在自动化学信息提取中的应用：推动AI驱动的化学研究 

---
# Improving Subgraph Matching by Combining Algorithms and Graph Neural Networks 

**Title (ZH)**: 通过结合算法和图神经网络提高子图匹配性能 

**Authors**: Shuyang Guo, Wenjin Xie, Ping Lu, Ting Deng, Richong Zhang, Jianxin Li, Xiangping Huang, Zhongyi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.20226)  

**Abstract**: Homomorphism is a key mapping technique between graphs that preserves their structure. Given a graph and a pattern, the subgraph homomorphism problem involves finding a mapping from the pattern to the graph, ensuring that adjacent vertices in the pattern are mapped to adjacent vertices in the graph. Unlike subgraph isomorphism, which requires a one-to-one mapping, homomorphism allows multiple vertices in the pattern to map to the same vertex in the graph, making it more complex. We propose HFrame, the first graph neural network-based framework for subgraph homomorphism, which integrates traditional algorithms with machine learning techniques. We demonstrate that HFrame outperforms standard graph neural networks by being able to distinguish more graph pairs where the pattern is not homomorphic to the graph. Additionally, we provide a generalization error bound for HFrame. Through experiments on both real-world and synthetic graphs, we show that HFrame is up to 101.91 times faster than exact matching algorithms and achieves an average accuracy of 0.962. 

**Abstract (ZH)**: 图同构是图之间保留其结构的关键映射技术。给定一个图和一个模式，子图同构问题涉及将模式映射到图中，使得模式中的相邻顶点在图中被映射到相邻的顶点。与需要一对一映射的子图同构不同，同构允许模式中的多个顶点映射到图中的同一个顶点，使其更加复杂。我们提出HFrame，这是一种基于图神经网络的子图同构框架，将传统算法与机器学习技术相结合。我们证明HFrame优于标准图神经网络，因为它能够区分更多的图对，在这些图对中模式与图不是同构的。此外，我们还为HFrame提供了泛化误差界。通过在真实世界和合成图上的实验，我们展示了HFrame比精确匹配算法快至101.91倍，并且平均准确率为0.962。 

---
# StepFun-Prover Preview: Let's Think and Verify Step by Step 

**Title (ZH)**: StepFun-Prover 预览：逐步思考与验证 

**Authors**: Shijie Shang, Ruosi Wan, Yue Peng, Yutong Wu, Xiong-hui Chen, Jie Yan, Xiangyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20199)  

**Abstract**: We present StepFun-Prover Preview, a large language model designed for formal theorem proving through tool-integrated reasoning. Using a reinforcement learning pipeline that incorporates tool-based interactions, StepFun-Prover can achieve strong performance in generating Lean 4 proofs with minimal sampling. Our approach enables the model to emulate human-like problem-solving strategies by iteratively refining proofs based on real-time environment feedback. On the miniF2F-test benchmark, StepFun-Prover achieves a pass@1 success rate of $70.0\%$. Beyond advancing benchmark performance, we introduce an end-to-end training framework for developing tool-integrated reasoning models, offering a promising direction for automated theorem proving and Math AI assistant. 

**Abstract (ZH)**: StepFun-Prover Preview：一种通过工具集成推理进行形式定理证明的大型语言模型 

---
# The Policy Cliff: A Theoretical Analysis of Reward-Policy Maps in Large Language Models 

**Title (ZH)**: 奖励政策悬崖：大型语言模型中奖励-政策映射的理论分析 

**Authors**: Xingcheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.20150)  

**Abstract**: Reinforcement learning (RL) plays a crucial role in shaping the behavior of large language and reasoning models (LLMs/LRMs). However, it often produces brittle and unstable policies, leading to critical failures such as spurious reasoning, deceptive alignment, and instruction disobedience that undermine the trustworthiness and safety of LLMs/LRMs. Currently, these issues lack a unified theoretical explanation and are typically addressed using ad-hoc heuristics. This paper presents a rigorous mathematical framework for analyzing the stability of the mapping from a reward function to the optimal policy. We show that policy brittleness often stems from non-unique optimal actions, a common occurrence when multiple valid traces exist in a reasoning task. This theoretical lens provides a unified explanation for a range of seemingly disparate failures, reframing them as rational outcomes of optimizing rewards that may be incomplete or noisy, especially in the presence of action degeneracy. We extend this analysis from the fundamental single-reward setting to the more realistic multi-reward RL across diverse domains, showing how stability is governed by an "effective reward" aggregation mechanism. We also prove that entropy regularization restores policy stability at the cost of increased stochasticity. Our framework provides a unified explanation for recent empirical findings on deceptive reasoning, instruction-following trade-offs, and RLHF-induced sophistry, and is further validated through perturbation experiments in multi-reward RL. This work advances policy-stability analysis from empirical heuristics towards a principled theory, offering essential insights for designing safer and more trustworthy AI systems. 

**Abstract (ZH)**: 强化学习（RL）在塑造大规模语言和推理模型（LLMs/LRMs）的行为中发挥着关键作用。然而，它通常会产生脆弱和不稳定的策略，导致诸如虚假推理、欺骗性对齐和指令不遵从等关键故障，这些故障会损害LLMs/LRMs的信任度和安全性。目前，这些问题缺乏统一的理论解释，通常仅通过非正式的启发式方法进行处理。本文提出了一种严谨的数学框架，用于分析奖励函数到最优策略映射的稳定性。我们表明，策略的脆弱性通常源于非唯一的最优行为，这是当推理任务中存在多个有效路径时的常见现象。这一理论视角为各种看似不同的故障提供了统一的解释，将其重新框定为奖励优化的合理结果，尤其是当存在行为退化时可能不完整或嘈杂。我们将这种分析从基础的单一奖励设置扩展到更现实的多奖励RL在不同领域的应用，展示了稳定性由“有效的奖励”聚合机制所支配。我们还证明了熵正则化可以以增加随机性为代价恢复策略的稳定性。本文框架为近期关于欺骗性推理、指令遵循权衡以及RLHF引起的狡辩的实证发现提供了统一的解释，并通过多奖励RL的扰动实验进一步得到了验证。这项工作将政策稳定性分析从经验性的启发式方法推进到了原理性的理论，为设计更安全和可信赖的AI系统提供了重要的见解。 

---
# Concept Learning for Cooperative Multi-Agent Reinforcement Learning 

**Title (ZH)**: 合作多智能体强化学习中的概念学习 

**Authors**: Zhonghan Ge, Yuanyang Zhu, Chunlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.20143)  

**Abstract**: Despite substantial progress in applying neural networks (NN) to multi-agent reinforcement learning (MARL) areas, they still largely suffer from a lack of transparency and interoperability. However, its implicit cooperative mechanism is not yet fully understood due to black-box networks. In this work, we study an interpretable value decomposition framework via concept bottleneck models, which promote trustworthiness by conditioning credit assignment on an intermediate level of human-like cooperation concepts. To address this problem, we propose a novel value-based method, named Concepts learning for Multi-agent Q-learning (CMQ), that goes beyond the current performance-vs-interpretability trade-off by learning interpretable cooperation concepts. CMQ represents each cooperation concept as a supervised vector, as opposed to existing models where the information flowing through their end-to-end mechanism is concept-agnostic. Intuitively, using individual action value conditioning on global state embeddings to represent each concept allows for extra cooperation representation capacity. Empirical evaluations on the StarCraft II micromanagement challenge and level-based foraging (LBF) show that CMQ achieves superior performance compared with the state-of-the-art counterparts. The results also demonstrate that CMQ provides more cooperation concept representation capturing meaningful cooperation modes, and supports test-time concept interventions for detecting potential biases of cooperation mode and identifying spurious artifacts that impact cooperation. 

**Abstract (ZH)**: 基于概念瓶颈模型的可解释价值分解框架：促进多agent Q学习中的透明度和互操作性 

---
# PITA: Preference-Guided Inference-Time Alignment for LLM Post-Training 

**Title (ZH)**: PITA：偏好引导的推理时对齐方法用于LLM后训练 

**Authors**: Sarat Chandra Bobbili, Ujwal Dinesha, Dheeraj Narasimha, Srinivas Shakkottai  

**Link**: [PDF](https://arxiv.org/pdf/2507.20067)  

**Abstract**: Inference-time alignment enables large language models (LLMs) to generate outputs aligned with end-user preferences without further training. Recent post-training methods achieve this by using small guidance models to modify token generation during inference. These methods typically optimize a reward function KL-regularized by the original LLM taken as the reference policy. A critical limitation, however, is their dependence on a pre-trained reward model, which requires fitting to human preference feedback--a potentially unstable process. In contrast, we introduce PITA, a novel framework that integrates preference feedback directly into the LLM's token generation, eliminating the need for a reward model. PITA learns a small preference-based guidance policy to modify token probabilities at inference time without LLM fine-tuning, reducing computational cost and bypassing the pre-trained reward model dependency. The problem is framed as identifying an underlying preference distribution, solved through stochastic search and iterative refinement of the preference-based guidance model. We evaluate PITA across diverse tasks, including mathematical reasoning and sentiment classification, demonstrating its effectiveness in aligning LLM outputs with user preferences. 

**Abstract (ZH)**: 推理时的对齐使大规模语言模型（LLMs）能够生成与最终用户偏好匹配的输出而无需进一步训练。最近的后训练方法通过使用小型指导模型在推理时修改标记生成来实现这一点。这些方法通常通过以原LLM作为参考策略来正则化奖励函数来优化一个奖励函数。然而，一个关键的局限性在于它们依赖于一个预训练的奖励模型，这需要拟合人类偏好反馈——这是一个可能不稳定的过程。相比之下，我们引入了PITA，这是一种新的框架，将偏好反馈直接整合到LLM的标记生成中，从而消除对奖励模型的依赖。PITA学习一个小型基于偏好的指导策略，在不进行LLM微调的情况下在推理时修改标记概率，减少计算成本并绕过了预训练的奖励模型依赖。该问题被表述为识别潜在的偏好分布，并通过随机搜索和基于偏好的指导模型的迭代改进来解决。我们在数学推理和情感分类等多种任务中评估PITA，证明了其在使LLM输出与用户偏好对齐方面的有效性。 

---
# Finding Personalized Good-Enough Solutions to Unsatisfiable Stable Roommates Problems 

**Title (ZH)**: 寻找不可满足稳定室友问题的个性化足够解 

**Authors**: Müge Fidan, Esra Erdem  

**Link**: [PDF](https://arxiv.org/pdf/2507.20010)  

**Abstract**: The Stable Roommates problems are characterized by the preferences of agents over other agents as roommates. A solution is a partition of the agents into pairs that are acceptable to each other (i.e., they are in the preference lists of each other), and the matching is stable (i.e., there do not exist any two agents who prefer each other to their roommates, and thus block the matching). Motivated by real-world applications, and considering that stable roommates problems do not always have solutions, we continue our studies to compute "good-enough" matchings. In addition to the agents' habits and habitual preferences, we consider their networks of preferred friends, and introduce a method to generate personalized solutions to stable roommates problems. We illustrate the usefulness of our method with examples and empirical evaluations. 

**Abstract (ZH)**: roommate分配问题由代理人对其他代理人的偏好特性确定。一个解决方案是将代理人划分为彼此接受的配对（即，他们彼此在各自的偏好列表中），并且该配对是稳定的（即，不存在任何两个代理人更喜欢彼此而不是他们的室友，从而阻止该配对）。受实际应用场景的启发，并考虑到roommate分配问题并非总是存在解决方案，我们继续研究计算“足够好”的配对方案。除了考虑代理人的习惯和惯常偏好之外，我们还考虑了他们偏好的社交网络，并引入了一种生成个性化roommate分配问题解决方案的方法。我们通过示例和实证评估来说明我们方法的有效性。 

---
# Matching Game Preferences Through Dialogical Large Language Models: A Perspective 

**Title (ZH)**: 通过对话式大型语言模型匹配游戏偏好：一种视角 

**Authors**: Renaud Fabre, Daniel Egret, Patrice Bellot  

**Link**: [PDF](https://arxiv.org/pdf/2507.20000)  

**Abstract**: This perspective paper explores the future potential of "conversational intelligence" by examining how Large Language Models (LLMs) could be combined with GRAPHYP's network system to better understand human conversations and preferences. Using recent research and case studies, we propose a conceptual framework that could make AI rea-soning transparent and traceable, allowing humans to see and understand how AI reaches its conclusions. We present the conceptual perspective of "Matching Game Preferences through Dialogical Large Language Models (D-LLMs)," a proposed system that would allow multiple users to share their different preferences through structured conversations. This approach envisions personalizing LLMs by embedding individual user preferences directly into how the model makes decisions. The proposed D-LLM framework would require three main components: (1) reasoning processes that could analyze different search experiences and guide performance, (2) classification systems that would identify user preference patterns, and (3) dialogue approaches that could help humans resolve conflicting information. This perspective framework aims to create an interpretable AI system where users could examine, understand, and combine the different human preferences that influence AI responses, detected through GRAPHYP's search experience networks. The goal of this perspective is to envision AI systems that would not only provide answers but also show users how those answers were reached, making artificial intelligence more transparent and trustworthy for human decision-making. 

**Abstract (ZH)**: 这一观点论文探讨了“对话智能”的未来潜力，通过研究大规模语言模型（LLMs）如何与GRAPHYP的网络系统结合，以更好地理解人类对话和偏好。利用 Recent 研究和案例研究，我们提出了一种概念框架，使人工智能推理变得透明可追踪，让人类能够观察和理解 AI 如何得出结论。我们提出了“通过对话式大规模语言模型（D-LLMs）匹配游戏偏好”的概念框架，该系统允许多个用户通过结构化的对话共享他们的不同偏好。这种方法设想通过将个人用户偏好直接嵌入模型决策过程来个性化 LLMS。提出的 D-LLM 框架需要三个主要组成部分：（1）推理过程，能够分析不同的搜索体验并指导性能，（2）分类系统，能够识别用户偏好模式，以及（3）对话方法，能够帮助人类解决矛盾信息。这一观点框架旨在创建一个可解释的 AI 系统，用户可以检查、理解和结合影响 AI 反应的不同人类偏好，这些偏好通过 GRAPHYP 的搜索体验网络被检测到。本视角的目的是设想不仅提供答案，还能向用户展示如何得出这些答案的人工智能系统，使人工智能更加透明和值得信赖，以支持人类决策。 

---
# Digital Twin Channel-Enabled Online Resource Allocation for 6G: Principle, Architecture and Application 

**Title (ZH)**: 基于数字孪生信道的6G在线资源分配：原理、架构与应用 

**Authors**: Tongjie Li, Jianhua Zhang, Li Yu, Yuxiang Zhang, Yunlong Cai, Fan Xu, Guangyi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.19974)  

**Abstract**: Emerging applications such as holographic communication, autonomous driving, and the industrial Internet of Things impose stringent requirements on flexible, low-latency, and reliable resource allocation in 6G networks. Conventional methods, which rely on statistical modeling, have proven effective in general contexts but may fail to achieve optimal performance in specific and dynamic environments. Furthermore, acquiring real-time channel state information (CSI) typically requires excessive pilot overhead. To address these challenges, a digital twin channel (DTC)-enabled online optimization framework is proposed, in which DTC is employed to predict CSI based on environmental sensing. The predicted CSI is then utilized by lightweight game-theoretic algorithms to perform online resource allocation in a timely and efficient manner. Simulation results based on a digital replica of a realistic industrial workshop demonstrate that the proposed method achieves throughput improvements of up to 11.5\% compared with pilot-based ideal CSI schemes, validating its effectiveness for scalable, low-overhead, and environment-aware communication in future 6G networks. 

**Abstract (ZH)**: 6G网络中基于数字孪生信道的新兴应用在线优化框架：实现及时高效的资源分配 

---
# Leveraging Fine-Tuned Large Language Models for Interpretable Pancreatic Cystic Lesion Feature Extraction and Risk Categorization 

**Title (ZH)**: 利用细调的大语言模型进行可解释的胰腺囊性病变特征提取与风险分类 

**Authors**: Ebrahim Rasromani, Stella K. Kang, Yanqi Xu, Beisong Liu, Garvit Luhadia, Wan Fung Chui, Felicia L. Pasadyn, Yu Chih Hung, Julie Y. An, Edwin Mathieu, Zehui Gu, Carlos Fernandez-Granda, Ammar A. Javed, Greg D. Sacks, Tamas Gonda, Chenchan Huang, Yiqiu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.19973)  

**Abstract**: Background: Manual extraction of pancreatic cystic lesion (PCL) features from radiology reports is labor-intensive, limiting large-scale studies needed to advance PCL research. Purpose: To develop and evaluate large language models (LLMs) that automatically extract PCL features from MRI/CT reports and assign risk categories based on guidelines. Materials and Methods: We curated a training dataset of 6,000 abdominal MRI/CT reports (2005-2024) from 5,134 patients that described PCLs. Labels were generated by GPT-4o using chain-of-thought (CoT) prompting to extract PCL and main pancreatic duct features. Two open-source LLMs were fine-tuned using QLoRA on GPT-4o-generated CoT data. Features were mapped to risk categories per institutional guideline based on the 2017 ACR White Paper. Evaluation was performed on 285 held-out human-annotated reports. Model outputs for 100 cases were independently reviewed by three radiologists. Feature extraction was evaluated using exact match accuracy, risk categorization with macro-averaged F1 score, and radiologist-model agreement with Fleiss' Kappa. Results: CoT fine-tuning improved feature extraction accuracy for LLaMA (80% to 97%) and DeepSeek (79% to 98%), matching GPT-4o (97%). Risk categorization F1 scores also improved (LLaMA: 0.95; DeepSeek: 0.94), closely matching GPT-4o (0.97), with no statistically significant differences. Radiologist inter-reader agreement was high (Fleiss' Kappa = 0.888) and showed no statistically significant difference with the addition of DeepSeek-FT-CoT (Fleiss' Kappa = 0.893) or GPT-CoT (Fleiss' Kappa = 0.897), indicating that both models achieved agreement levels on par with radiologists. Conclusion: Fine-tuned open-source LLMs with CoT supervision enable accurate, interpretable, and efficient phenotyping for large-scale PCL research, achieving performance comparable to GPT-4o. 

**Abstract (ZH)**: 背景：手动从影像报告中提取胰腺囊性病变（PCL）特征耗时且 labor-intensive，限制了大规模研究的进行，从而阻碍了PCL研究的进步。目的：开发并评估大型语言模型（LLMs），使其能够自动从MRI/CT报告中提取PCL特征，并根据指南进行风险分类。材料与方法：我们构建了一个包含6,000份腹部MRI/CT报告（2005-2024年）的训练集，涉及5,134名患者，这些报告描述了PCL。标签由GPT-4o通过链式思考（CoT）提示生成，提取PCL和主要胰管特征。使用QLoRA对两种开源LLMs进行 fine-tuning，并使用GPT-4o生成的CoT数据。特征根据2017年ACR白皮书中的机构指南映射到相应风险类别。在285份保留的、由人类标注的报告上进行了评估。三位放射科医师独立审查了100个病例的模型输出。特征提取的准确性使用精确匹配准确率评估，风险分类使用宏平均F1分数评估，放射科医师与模型的共识则使用Fleiss' Kappa系数。结果：CoT fine-tuning提高了LLaMA（从80%到97%）和DeepSeek（从79%到98%）的特征提取准确性，与GPT-4o（97%）持平。风险分类F1分数也有所提高（LLaMA：0.95；DeepSeek：0.94），接近GPT-4o（0.97），并无统计学差异。放射科医师的内读者一致性高（Fleiss' Kappa = 0.888），并与DeepSeek-FT-CoT（Fleiss' Kappa = 0.893）或GPT-CoT（Fleiss' Kappa = 0.897）的添加无统计学差异，表明两种模型与放射科医师达到了一致水平。结论：带有CoT监督的开源LLMs fine-tuning能够实现大规模PCL研究中准确、可解释和高效的表型研究，其性能与GPT-4o相当。 

---
# What Does 'Human-Centred AI' Mean? 

**Title (ZH)**: 以人为本的AI意味着什么？ 

**Authors**: Olivia Guest  

**Link**: [PDF](https://arxiv.org/pdf/2507.19960)  

**Abstract**: While it seems sensible that human-centred artificial intelligence (AI) means centring "human behaviour and experience," it cannot be any other way. AI, I argue, is usefully seen as a relationship between technology and humans where it appears that artifacts can perform, to a greater or lesser extent, human cognitive labour. This is evinced using examples that juxtapose technology with cognition, inter alia: abacus versus mental arithmetic; alarm clock versus knocker-upper; camera versus vision; and sweatshop versus tailor. Using novel definitions and analyses, sociotechnical relationships can be analysed into varying types of: displacement (harmful), enhancement (beneficial), and/or replacement (neutral) of human cognitive labour. Ultimately, all AI implicates human cognition; no matter what. Obfuscation of cognition in the AI context -- from clocks to artificial neural networks -- results in distortion, in slowing critical engagement, perverting cognitive science, and indeed in limiting our ability to truly centre humans and humanity in the engineering of AI systems. To even begin to de-fetishise AI, we must look the human-in-the-loop in the eyes. 

**Abstract (ZH)**: 以人为本的人工智能意味着中心化“人类行为和经验”——别无选择。 

---
# Causality-aligned Prompt Learning via Diffusion-based Counterfactual Generation 

**Title (ZH)**: 基于扩散驱动反事实生成的因果对齐提示学习 

**Authors**: Xinshu Li, Ruoyu Wang, Erdun Gao, Mingming Gong, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2507.19882)  

**Abstract**: Prompt learning has garnered attention for its efficiency over traditional model training and fine-tuning. However, existing methods, constrained by inadequate theoretical foundations, encounter difficulties in achieving causally invariant prompts, ultimately falling short of capturing robust features that generalize effectively across categories. To address these challenges, we introduce the $\textit{\textbf{DiCap}}$ model, a theoretically grounded $\textbf{Di}$ffusion-based $\textbf{C}$ounterf$\textbf{a}$ctual $\textbf{p}$rompt learning framework, which leverages a diffusion process to iteratively sample gradients from the marginal and conditional distributions of the causal model, guiding the generation of counterfactuals that satisfy the minimal sufficiency criterion. Grounded in rigorous theoretical derivations, this approach guarantees the identifiability of counterfactual outcomes while imposing strict bounds on estimation errors. We further employ a contrastive learning framework that leverages the generated counterfactuals, thereby enabling the refined extraction of prompts that are precisely aligned with the causal features of the data. Extensive experimental results demonstrate that our method performs excellently across tasks such as image classification, image-text retrieval, and visual question answering, with particularly strong advantages in unseen categories. 

**Abstract (ZH)**: 基于扩散过程的因果 counterfactual 命令学习框架 DiCap 

---
# Reinforcement Learning for Multi-Objective Multi-Echelon Supply Chain Optimisation 

**Title (ZH)**: 多目标多层次供应链优化的强化学习方法 

**Authors**: Rifny Rachman, Josh Tingey, Richard Allmendinger, Pradyumn Shukla, Wei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2507.19788)  

**Abstract**: This study develops a generalised multi-objective, multi-echelon supply chain optimisation model with non-stationary markets based on a Markov decision process, incorporating economic, environmental, and social considerations. The model is evaluated using a multi-objective reinforcement learning (RL) method, benchmarked against an originally single-objective RL algorithm modified with weighted sum using predefined weights, and a multi-objective evolutionary algorithm (MOEA)-based approach. We conduct experiments on varying network complexities, mimicking typical real-world challenges using a customisable simulator. The model determines production and delivery quantities across supply chain routes to achieve near-optimal trade-offs between competing objectives, approximating Pareto front sets. The results demonstrate that the primary approach provides the most balanced trade-off between optimality, diversity, and density, further enhanced with a shared experience buffer that allows knowledge transfer among policies. In complex settings, it achieves up to 75\% higher hypervolume than the MOEA-based method and generates solutions that are approximately eleven times denser, signifying better robustness, than those produced by the modified single-objective RL method. Moreover, it ensures stable production and inventory levels while minimising demand loss. 

**Abstract (ZH)**: 基于马尔可夫决策过程的非平稳市场综合多目标多层级供应链优化模型：多目标强化学习方法及其应用研究 

---
# Can LLMs Solve ASP Problems? Insights from a Benchmarking Study (Extended Version) 

**Title (ZH)**: LLM解决ASP问题的能力：一项基准研究的见解（扩展版） 

**Authors**: Lin Ren, Guohui Xiao, Guilin Qi, Yishuai Geng, Haohan Xue  

**Link**: [PDF](https://arxiv.org/pdf/2507.19749)  

**Abstract**: Answer Set Programming (ASP) is a powerful paradigm for non-monotonic reasoning. Recently, large language models (LLMs) have demonstrated promising capabilities in logical reasoning. Despite this potential, current evaluations of LLM capabilities in ASP are often limited. Existing works normally employ overly simplified ASP programs, do not support negation, disjunction, or multiple answer sets. Furthermore, there is a lack of benchmarks that introduce tasks specifically designed for ASP solving. To bridge this gap, we introduce ASPBench, a comprehensive ASP benchmark, including three ASP specific tasks: ASP entailment, answer set verification, and answer set computation. Our extensive evaluations on ASPBench reveal that while 14 state-of-the-art LLMs, including \emph{deepseek-r1}, \emph{o4-mini}, and \emph{gemini-2.5-flash-thinking}, perform relatively well on the first two simpler tasks, they struggle with answer set computation, which is the core of ASP solving. These findings offer insights into the current limitations of LLMs in ASP solving. This highlights the need for new approaches that integrate symbolic reasoning capabilities more effectively. The code and dataset are available at this https URL. 

**Abstract (ZH)**: ASPBench：一个全面的ASP基准，包含三个特定的ASP任务：ASP蕴含、答案集验证和答案集计算。 

---
# Integrating Activity Predictions in Knowledge Graphs 

**Title (ZH)**: 在知识图谱中集成活动预测 

**Authors**: Alec Scully, Cameron Stockton, Forrest Hare  

**Link**: [PDF](https://arxiv.org/pdf/2507.19733)  

**Abstract**: We argue that ontology-structured knowledge graphs can play a crucial role in generating predictions about future events. By leveraging the semantic framework provided by Basic Formal Ontology (BFO) and Common Core Ontologies (CCO), we demonstrate how data such as the movements of a fishing vessel can be organized in and retrieved from a knowledge graph. These query results are then used to create Markov chain models, allowing us to predict future states based on the vessel's history. To fully support this process, we introduce the term `spatiotemporal instant' to complete the necessary structural semantics. Additionally, we critique the prevailing ontological model of probability, which conflates probability with likelihood and relies on the problematic concept of modal measurements: measurements of future entities. We propose an alternative view, where probabilities are treated as being about process profiles, which better captures the dynamics of real world phenomena. Finally, we demonstrate how our Markov chain based probability calculations can be seamlessly integrated back into the knowledge graph, enabling further analysis and decision-making. Keywords: predictive analytics, ontology, Markov chains, probability, Basic Formal Ontology (BFO), knowledge graphs, SPARQL. 

**Abstract (ZH)**: 本体结构化的知识图谱在生成未来事件预测中的关键作用：基于基本形式本体(BFO)和通用核心本体(CCO)的语义框架，船舶运动等数据的组织与检索及其在马尔可夫链模型中的应用与分析：对概率本体模型的批判与替代观点：基于马尔可夫链的概率计算无缝集成回知识图谱 

---
# HypKG: Hypergraph-based Knowledge Graph Contextualization for Precision Healthcare 

**Title (ZH)**: HypKG：基于超图的知识图谱精准医疗上下文表示 

**Authors**: Yuzhang Xie, Xu Han, Ran Xu, Xiao Hu, Jiaying Lu, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19726)  

**Abstract**: Knowledge graphs (KGs) are important products of the semantic web, which are widely used in various application domains. Healthcare is one of such domains where KGs are intensively used, due to the high requirement for knowledge accuracy and interconnected nature of healthcare data. However, KGs storing general factual information often lack the ability to account for important contexts of the knowledge such as the status of specific patients, which are crucial in precision healthcare. Meanwhile, electronic health records (EHRs) provide rich personal data, including various diagnoses and medications, which provide natural contexts for general KGs. In this paper, we propose HypKG, a framework that integrates patient information from EHRs into KGs to generate contextualized knowledge representations for accurate healthcare predictions. Using advanced entity-linking techniques, we connect relevant knowledge from general KGs with patient information from EHRs, and then utilize a hypergraph model to "contextualize" the knowledge with the patient information. Finally, we employ hypergraph transformers guided by downstream prediction tasks to jointly learn proper contextualized representations for both KGs and patients, fully leveraging existing knowledge in KGs and patient contexts in EHRs. In experiments using a large biomedical KG and two real-world EHR datasets, HypKG demonstrates significant improvements in healthcare prediction tasks across multiple evaluation metrics. Additionally, by integrating external contexts, HypKG can learn to adjust the representations of entities and relations in KG, potentially improving the quality and real-world utility of knowledge. 

**Abstract (ZH)**: 知识图谱（KGs）是语义网的重要产物，广泛应用于各种应用领域。在对知识准确性要求高且医疗数据具有强关联性的医疗健康领域，KGs被密集使用。然而，用于存储通用事实信息的KGs往往缺乏考虑具体患者状态等重要上下文的能力，这对精准医疗至关重要。与此同时，电子健康记录（EHRs）提供了丰富个人数据，包括各种诊断和用药信息，为通用KGs提供了自然的上下文。在本文中，我们提出了一种名为HypKG的框架，该框架将EHRs中的患者信息整合到KGs中，生成上下文感知的知识表示，以实现准确的医疗预测。通过先进的实体链接技术，我们连接了通用KGs中的相关知识与EHRs中的患者信息，并利用超图模型将知识“上下文化”到患者信息中。最后，我们采用由下游预测任务引导的超图变压器，联合学习KGs和患者双方的适当上下文化表示，充分利用KGs中已有的知识以及EHRs中的患者上下文。在使用大规模生物医学KG和两个实际EHR数据集的实验中，HypKG在多个评估指标上显著提高了医疗预测任务的效果。此外，通过整合外部上下文，HypKG能够学习调整KG中实体和关系的表示，这可能提高知识的质量和实际应用价值。 

---
# Minding Motivation: The Effect of Intrinsic Motivation on Agent Behaviors 

**Title (ZH)**: 关注动机：内在动机对面剂行为的影响 

**Authors**: Leonardo Villalobos-Arias, Grant Forbes, Jianxun Wang, David L Roberts, Arnav Jhala  

**Link**: [PDF](https://arxiv.org/pdf/2507.19725)  

**Abstract**: Games are challenging for Reinforcement Learning~(RL) agents due to their reward-sparsity, as rewards are only obtainable after long sequences of deliberate actions. Intrinsic Motivation~(IM) methods -- which introduce exploration rewards -- are an effective solution to reward-sparsity. However, IM also causes an issue known as `reward hacking' where the agent optimizes for the new reward at the expense of properly playing the game. The larger problem is that reward hacking itself is largely unknown; there is no answer to whether, and to what extent, IM rewards change the behavior of RL agents. This study takes a first step by empirically evaluating the impact on behavior of three IM techniques on the MiniGrid game-like environment. We compare these IM models with Generalized Reward Matching~(GRM), a method that can be used with any intrinsic reward function to guarantee optimality. Our results suggest that IM causes noticeable change by increasing the initial rewards, but also altering the way the agent plays; and that GRM mitigated reward hacking in some scenarios. 

**Abstract (ZH)**: 游戏由于奖励稀疏性对强化学习（RL）代理构成挑战，奖励仅在经过一系列故意动作后方可获得。内在动机（IM）方法——通过引入探索奖励——是解决奖励稀疏性的一种有效方案。然而，IM也会引起一种被称为“奖励劫持”的问题，代理会优化新奖励而代价是未能正确地玩游戏。更大的问题是，奖励劫持本身 largely unknown；目前尚无答案来确定IM奖励是否以及在多大程度上改变了RL代理的行为。本研究通过实证评估三种IM技术对MiniGrid游戏环境的影响，首次对此进行探索。我们将这些IM模型与通用奖励匹配（GRM）方法进行了对比，GRM是一种可以与任何内在奖励函数结合使用以确保最优性的方法。我们的结果显示，IM通过增加初始奖励引起了显著变化，同时也改变了代理的玩法；而在某些场景下，GRM减轻了奖励劫持的问题。 

---
# The wall confronting large language models 

**Title (ZH)**: 大型语言模型面临的挑战 

**Authors**: Peter V. Coveney, Sauro Succi  

**Link**: [PDF](https://arxiv.org/pdf/2507.19703)  

**Abstract**: We show that the scaling laws which determine the performance of large language models (LLMs) severely limit their ability to improve the uncertainty of their predictions. As a result, raising their reliability to meet the standards of scientific inquiry is intractable by any reasonable measure. We argue that the very mechanism which fuels much of the learning power of LLMs, namely the ability to generate non-Gaussian output distributions from Gaussian input ones, might well be at the roots of their propensity to produce error pileup, ensuing information catastrophes and degenerative AI behaviour. This tension between learning and accuracy is a likely candidate mechanism underlying the observed low values of the scaling components. It is substantially compounded by the deluge of spurious correlations pointed out by Calude and Longo which rapidly increase in any data set merely as a function of its size, regardless of its nature. The fact that a degenerative AI pathway is a very probable feature of the LLM landscape does not mean that it must inevitably arise in all future AI research. Its avoidance, which we also discuss in this paper, necessitates putting a much higher premium on insight and understanding of the structural characteristics of the problems being investigated. 

**Abstract (ZH)**: 我们展示了决定大型语言模型（LLMs）性能的标度律严重限制了它们改进预测不确定性的能力。结果，提高其可靠性以达到科学探究的标准在任何合理的衡量标准下都是无法实现的。我们argue认为，正是生成非高斯输出分布的能力，从高斯输入分布中生成，可能是导致LLMs产生累积误差、信息灾难和退化AI行为的根本机制之一。这种学习与准确性的张力可能是观察到的标度成分低值的潜在机制之一。这种张力被Calude和Longo指出的虚假相关性进一步加剧，这些虚假相关性在数据集的大小增加时迅速增加，且不依赖于数据集的性质。LLMs存在退化AI路径的概率特征并不意味着它必然会在所有未来的AI研究中出现。避免这种退化AI路径，需要我们在研究过程中更重视对所研究问题的结构特征的理解和洞察。 

---
# Alignment and Safety in Large Language Models: Safety Mechanisms, Training Paradigms, and Emerging Challenges 

**Title (ZH)**: 大型语言模型中的对齐与安全性：安全机制、训练范式及新兴挑战 

**Authors**: Haoran Lu, Luyang Fang, Ruidong Zhang, Xinliang Li, Jiazhang Cai, Huimin Cheng, Lin Tang, Ziyu Liu, Zeliang Sun, Tao Wang, Yingchuan Zhang, Arif Hassan Zidan, Jinwen Xu, Jincheng Yu, Meizhi Yu, Hanqi Jiang, Xilin Gong, Weidi Luo, Bolun Sun, Yongkai Chen, Terry Ma, Shushan Wu, Yifan Zhou, Junhao Chen, Haotian Xiang, Jing Zhang, Afrar Jahin, Wei Ruan, Ke Deng, Yi Pan, Peilong Wang, Jiahui Li, Zhengliang Liu, Lu Zhang, Lin Zhao, Wei Liu, Dajiang Zhu, Xin Xing, Fei Dou, Wei Zhang, Chao Huang, Rongjie Liu, Mengrui Zhang, Yiwen Liu, Xiaoxiao Sun, Qin Lu, Zhen Xiang, Wenxuan Zhong, Tianming Liu, Ping Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.19672)  

**Abstract**: Due to the remarkable capabilities and growing impact of large language models (LLMs), they have been deeply integrated into many aspects of society. Thus, ensuring their alignment with human values and intentions has emerged as a critical challenge. This survey provides a comprehensive overview of practical alignment techniques, training protocols, and empirical findings in LLM alignment. We analyze the development of alignment methods across diverse paradigms, characterizing the fundamental trade-offs between core alignment objectives. Our analysis shows that while supervised fine-tuning enables basic instruction-following, preference-based methods offer more flexibility for aligning with nuanced human intent. We discuss state-of-the-art techniques, including Direct Preference Optimization (DPO), Constitutional AI, brain-inspired methods, and alignment uncertainty quantification (AUQ), highlighting their approaches to balancing quality and efficiency. We review existing evaluation frameworks and benchmarking datasets, emphasizing limitations such as reward misspecification, distributional robustness, and scalable oversight. We summarize strategies adopted by leading AI labs to illustrate the current state of practice. We conclude by outlining open problems in oversight, value pluralism, robustness, and continuous alignment. This survey aims to inform both researchers and practitioners navigating the evolving landscape of LLM alignment. 

**Abstract (ZH)**: 由于大型语言模型（LLMs）的卓越能力和日益增长的影响，它们已被广泛融入社会的许多方面。因此，确保它们与人类价值观和意图的一致性已成为一个关键挑战。本综述提供了LLM一致性技术、训练协议和实验研究的全面概述。我们分析了一致性方法在不同范式下的发展，阐述了核心一致目标之间的基本权衡。我们的分析表明，虽然监督微调可以实现基本的指令遵循，但基于偏好的方法则为与细微的人类意图对齐提供了更多灵活性。我们讨论了最先进的技术，包括直接偏好优化（DPO）、宪法AI、大脑启发方法以及一致性不确定性量化（AUQ），强调了它们在平衡质量和效率方面的策略。我们回顾了现有的评估框架和基准数据集，强调了奖励误指定、分布鲁棒性和可扩展监督等局限性。我们总结了领先AI实验室采用的策略，以说明当前的一致性实践状况。最后，我们概述了在监督、价值多元论、鲁棒性和持续一致性方面的开放问题。本综述旨在为研究人员和从业者提供关于LLM一致性不断演变的景观的指导信息。 

---
# DeltaLLM: A Training-Free Framework Exploiting Temporal Sparsity for Efficient Edge LLM Inference 

**Title (ZH)**: DeltaLLM：一种利用时间稀疏性进行高效边缘端LLM推理的无训练框架 

**Authors**: Jiawen Qi, Chang Gao, Zhaochun Ren, Qinyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.19608)  

**Abstract**: Deploying Large Language Models (LLMs) on edge devices remains challenging due to their quadratically increasing computations with the sequence length. Existing studies for dynamic attention pruning are designed for hardware with massively parallel computation capabilities, such as GPUs or TPUs, and aim at long context lengths (e.g., 64K), making them unsuitable for edge scenarios. We present DeltaLLM, a training-free framework that exploits temporal sparsity in attention patterns to enable efficient LLM inference across both the prefilling and decoding stages, on resource-constrained edge devices. DeltaLLM introduces an accuracy- and memory-aware delta matrix construction strategy that introduces temporal sparsity, and a context-aware hybrid attention mechanism that combines full attention in a local context window with delta approximation outside it to increase accuracy. We evaluate our framework on the edge-device-friendly BitNet-b1.58-2B-4T model and Llama3.2-1B-Instruct model across diverse language tasks. The results show that on BitNet, our framework increases the attention sparsity from 0% to 60% during the prefilling stage with slight accuracy improvement on the WG task, and 0% to 57% across both the prefilling and decoding stages, with even higher F1 score from 29.63 to 30.97 on SQuAD-v2 task. On the Llama model, it can also achieve up to 60% sparsity during the prefilling stage and around 57% across both stages with negligible accuracy drop. These results demonstrate that DeltaLLM offers a promising solution for efficient edge deployment, requiring no fine-tuning and seamlessly integrating with existing inference pipelines. 

**Abstract (ZH)**: DeltaLLM：利用注意力模式的时域稀疏性在受限资源边缘设备上实现高效的大语言模型推理 

---
# Hypergames: Modeling Misaligned Perceptions and Nested Beliefs for Multi-agent Systems 

**Title (ZH)**: 多智能体系统中不对齐感知和嵌套信念的建模：超游戏 

**Authors**: Vince Trencsenyi, Agnieszka Mensfelt, Kostas Stathis  

**Link**: [PDF](https://arxiv.org/pdf/2507.19593)  

**Abstract**: Classical game-theoretic models typically assume rational agents, complete information, and common knowledge of payoffs - assumptions that are often violated in real-world MAS characterized by uncertainty, misaligned perceptions, and nested beliefs. To overcome these limitations, researchers have proposed extensions that incorporate models of cognitive constraints, subjective beliefs, and heterogeneous reasoning. Among these, hypergame theory extends the classical paradigm by explicitly modeling agents' subjective perceptions of the strategic scenario, known as perceptual games, in which agents may hold divergent beliefs about the structure, payoffs, or available actions. We present a systematic review of agent-compatible applications of hypergame theory, examining how its descriptive capabilities have been adapted to dynamic and interactive MAS contexts. We analyze 44 selected studies from cybersecurity, robotics, social simulation, communications, and general game-theoretic modeling. Building on a formal introduction to hypergame theory and its two major extensions - hierarchical hypergames and HNF - we develop agent-compatibility criteria and an agent-based classification framework to assess integration patterns and practical applicability. Our analysis reveals prevailing tendencies, including the prevalence of hierarchical and graph-based models in deceptive reasoning and the simplification of extensive theoretical frameworks in practical applications. We identify structural gaps, including the limited adoption of HNF-based models, the lack of formal hypergame languages, and unexplored opportunities for modeling human-agent and agent-agent misalignment. By synthesizing trends, challenges, and open research directions, this review provides a new roadmap for applying hypergame theory to enhance the realism and effectiveness of strategic modeling in dynamic multi-agent environments. 

**Abstract (ZH)**: 古典博弈理论模型通常假设理性代理、完全信息以及共同的知识支付—这些假设在由不确定性、对齐偏差和嵌套信念特征的现实世界MAS中经常被违反。为克服这些限制，研究人员提出了包含认知约束、主观信念和异质推理模型的扩展。其中，超博弈理论通过明确建模代理对战略场景的主观感知，即感知博弈，扩展了古典范式，其中代理可能对结构、支付或可用行动持有不同的信念。我们对超博弈理论的代理兼容应用进行了系统回顾，考察了其描述能力如何适应动态和交互式的MAS环境。我们分析了来自网络安全、机器人、社会仿真、通信和一般博弈论建模的44项研究。基于对超博弈理论及其两大扩展——层次超博弈和HNF的正式介绍，我们开发了代理兼容性标准和基于代理的分类框架，以评估整合模式和实际应用性。我们的分析揭示了现有趋势，包括欺骗推理中层次和图基模型的普及以及理论框架在实际应用中的简化。我们指出了结构性缺口，包括HNF基模型的有限采用、缺乏形式化超博弈语言以及对人类代理和代理间不对齐建模机会的未开发。通过综合趋势、挑战和开放的研究方向，本回顾为利用超博弈理论增强动态多代理环境中的战略模型的真实性和有效性提供了新的路线图。 

---
# Agent WARPP: Workflow Adherence via Runtime Parallel Personalization 

**Title (ZH)**: Agent WARPP: 工作流遵守性通过运行时并行个性化 

**Authors**: Maria Emilia Mazzolenis, Ruirui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19543)  

**Abstract**: Large language models (LLMs) are increasingly applied in task-oriented dialogue (TOD) systems but often struggle with long, conditional workflows that involve external tool calls and depend on user-specific information. We present Workflow Adherence via Runtime Parallel Personalization, or WARPP, a training-free, modular framework that combines multi-agent orchestration with runtime personalization to improve workflow adherence in LLM-based systems. By dynamically pruning conditional branches based on user attributes, the framework reduces reasoning overhead and narrows tool selection at runtime. WARPP deploys a parallelized architecture where a dedicated Personalizer agent operates alongside modular, domain-specific agents to dynamically tailor execution paths in real time. The framework is evaluated across five representative user intents of varying complexity within three domains: banking, flights, and healthcare. Our evaluation leverages synthetic datasets and LLM-powered simulated users to test scenarios with conditional dependencies. Our results demonstrate that WARPP outperforms both the non-personalized method and the ReAct baseline, achieving increasingly larger gains in parameter fidelity and tool accuracy as intent complexity grows, while also reducing average token usage, without any additional training. 

**Abstract (ZH)**: 基于运行时并行个人化的流程遵从性框架：无需训练的模块化方法改进大语言模型的流程遵从性 

---
# MAIA: A Collaborative Medical AI Platform for Integrated Healthcare Innovation 

**Title (ZH)**: MAIA：一种集成医疗创新的协作医疗AI平台 

**Authors**: Simone Bendazzoli, Sanna Persson, Mehdi Astaraki, Sebastian Pettersson, Vitali Grozman, Rodrigo Moreno  

**Link**: [PDF](https://arxiv.org/pdf/2507.19489)  

**Abstract**: The integration of Artificial Intelligence (AI) into clinical workflows requires robust collaborative platforms that are able to bridge the gap between technical innovation and practical healthcare applications. This paper introduces MAIA (Medical Artificial Intelligence Assistant), an open-source platform designed to facilitate interdisciplinary collaboration among clinicians, researchers, and AI developers. Built on Kubernetes, MAIA offers a modular, scalable environment with integrated tools for data management, model development, annotation, deployment, and clinical feedback. Key features include project isolation, CI/CD automation, integration with high-computing infrastructures and in clinical workflows. MAIA supports real-world use cases in medical imaging AI, with deployments in both academic and clinical environments. By promoting collaborations and interoperability, MAIA aims to accelerate the translation of AI research into impactful clinical solutions while promoting reproducibility, transparency, and user-centered design. We showcase the use of MAIA with different projects, both at KTH Royal Institute of Technology and Karolinska University Hospital. 

**Abstract (ZH)**: 将人工智能融入临床工作流需要强大的协作平台，能够弥合技术创新与实际医疗应用之间的差距。本文介绍了MAIA（医疗人工智能助手）这一开源平台，旨在促进临床医生、研究人员和人工智能开发者之间的跨学科合作。基于Kubernetes构建，MAIA提供了一个模块化、可扩展的环境，集成了数据管理、模型开发、注释、部署和临床反馈的工具。关键功能包括项目隔离、CI/CD自动化、与高性能计算基础设施以及临床工作流的集成。MAIA支持医疗成像人工智能的现实世界应用场景，在学术和临床环境中均有部署。通过促进合作与互操作性，MAIA旨在加速将人工智能研究转化为具有影响力的临床解决方案，同时促进可重复性、透明性和用户中心设计。我们展示了MAIA在Karolinska大学医院和瑞典皇家理工学院的不同项目中的应用。 

---
# Memorization in Fine-Tuned Large Language Models 

**Title (ZH)**: Fine-Tuned大型语言模型中的记忆现象 

**Authors**: Danil Savine, Muni Sreenivas Pydi, Jamal Atif, Olivier Cappé  

**Link**: [PDF](https://arxiv.org/pdf/2507.21009)  

**Abstract**: This study investigates the mechanisms and factors influencing memorization in fine-tuned large language models (LLMs), with a focus on the medical domain due to its privacy-sensitive nature. We examine how different aspects of the fine-tuning process affect a model's propensity to memorize training data, using the PHEE dataset of pharmacovigilance events.
Our research employs two main approaches: a membership inference attack to detect memorized data, and a generation task with prompted prefixes to assess verbatim reproduction. We analyze the impact of adapting different weight matrices in the transformer architecture, the relationship between perplexity and memorization, and the effect of increasing the rank in low-rank adaptation (LoRA) fine-tuning.
Key findings include: (1) Value and Output matrices contribute more significantly to memorization compared to Query and Key matrices; (2) Lower perplexity in the fine-tuned model correlates with increased memorization; (3) Higher LoRA ranks lead to increased memorization, but with diminishing returns at higher ranks.
These results provide insights into the trade-offs between model performance and privacy risks in fine-tuned LLMs. Our findings have implications for developing more effective and responsible strategies for adapting large language models while managing data privacy concerns. 

**Abstract (ZH)**: 本研究 investigates 细化为探讨调优大型语言模型（LLMs）中记忆机制及其影响因素，重点关注医疗领域因其敏感的隐私特性。我们通过使用PHEE药物警戒事件数据集，研究调优过程中不同方面对模型记忆训练数据倾向的影响。

我们的研究采用两种主要方法：一种是成员推断攻击以检测记忆的数据，另一种是带有提示前缀的生成任务以评估逐字复制。我们分析了适应不同权重矩阵在转换器架构中的影响、困惑度与记忆之间的关系，以及低秩适应（LoRA）调优中秩增大的影响。

主要发现包括：（1）值和输出矩阵比查询和键矩阵更显著地贡献于记忆；（2）调优后的模型较低的困惑度与增加的记忆有关；（3）较高的LoRA秩导致记忆增加，但随着秩的增大，效果递减。

这些结果为模型性能与隐私风险之间的权衡提供了见解。我们的发现对开发更有效且负责任的大语言模型适应策略、同时管理数据隐私问题具有重要意义。 

---
# Compositional Function Networks: A High-Performance Alternative to Deep Neural Networks with Built-in Interpretability 

**Title (ZH)**: 组成函数网络：内置可解释性的高性能替代深度神经网络 

**Authors**: Fang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.21004)  

**Abstract**: Deep Neural Networks (DNNs) deliver impressive performance but their black-box nature limits deployment in high-stakes domains requiring transparency. We introduce Compositional Function Networks (CFNs), a novel framework that builds inherently interpretable models by composing elementary mathematical functions with clear semantics. Unlike existing interpretable approaches that are limited to simple additive structures, CFNs support diverse compositional patterns -- sequential, parallel, and conditional -- enabling complex feature interactions while maintaining transparency. A key innovation is that CFNs are fully differentiable, allowing efficient training through standard gradient descent. We demonstrate CFNs' versatility across multiple domains, from symbolic regression to image classification with deep hierarchical networks. Our empirical evaluation shows CFNs achieve competitive performance against black-box models (96.24% accuracy on CIFAR-10) while outperforming state-of-the-art interpretable models like Explainable Boosting Machines. By combining the hierarchical expressiveness and efficient training of deep learning with the intrinsic interpretability of well-defined mathematical functions, CFNs offer a powerful framework for applications where both performance and accountability are paramount. 

**Abstract (ZH)**: 深度神经网络（DNNs）表现出色，但其黑箱性质限制了在需要透明性的高风险领域中的部署。我们引入了构成函数网络（CFNs），这是一种新型框架，通过组合具有明确语义的基本数学函数来构建固有的可解释模型。与现有受限于简单加性结构的可解释方法不同，CFNs 支持多种组合模式——序列、并行和条件模式——这能够支持复杂的特征交互同时保持透明性。一个关键创新是CFNs是全可微的，允许通过标准梯度下降进行高效训练。我们展示了CFNs在多个领域的灵活性，从符号回归到使用深度层次网络的图像分类。我们的实证评估表明，CFNs在与黑箱模型相当的性能上（CIFAR-10数据集准确率为96.24%）超过了最新的可解释模型如可解释增强树。通过结合深度学习的层次表达能力和清晰数学函数的固有可解释性，CFNs为同时需要性能和问责制的应用提供了强大的框架。 

---
# Modular Delta Merging with Orthogonal Constraints: A Scalable Framework for Continual and Reversible Model Composition 

**Title (ZH)**: 正交约束下的模块化Δ合并：一种可扩展的持续可逆模型组合框架 

**Authors**: Haris Khan, Shumaila Asif, Sadia Asif  

**Link**: [PDF](https://arxiv.org/pdf/2507.20997)  

**Abstract**: In real-world machine learning deployments, models must be continually updated, composed, and when required, selectively undone. However, existing approaches to model merging and continual learning often suffer from task interference, catastrophic forgetting, or lack of reversibility. We propose Modular Delta Merging with Orthogonal Constraints (MDM-OC), a novel framework that enables scalable, interference-free, and reversible composition of fine-tuned models. Each task-specific model is encoded as a delta from a shared base and projected into an orthogonal subspace to eliminate conflict. These projected deltas are then merged via gradient-based optimization to form a unified model that retains performance across tasks. Our approach supports continual integration of new models, structured unmerging for compliance such as GDPR requirements, and model stability via elastic weight consolidation and synthetic replay. Extensive experiments on vision and natural language processing benchmarks demonstrate that MDM-OC outperforms prior baselines in accuracy, backward transfer, and unmerge fidelity, while remaining memory-efficient and computationally tractable. This framework offers a principled solution for modular and compliant AI system design. 

**Abstract (ZH)**: 模块化delta合并与正交约束框架：无干扰、可逆的模型组合与持续学习 

---
# Security Tensors as a Cross-Modal Bridge: Extending Text-Aligned Safety to Vision in LVLM 

**Title (ZH)**: 安全张量作为跨模态桥梁：将文本对齐的安全性扩展到LVLM中的视觉领域 

**Authors**: Shen Li, Liuyi Yao, Wujia Niu, Lan Zhang, Yaliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.20994)  

**Abstract**: Large visual-language models (LVLMs) integrate aligned large language models (LLMs) with visual modules to process multimodal inputs. However, the safety mechanisms developed for text-based LLMs do not naturally extend to visual modalities, leaving LVLMs vulnerable to harmful image inputs. To address this cross-modal safety gap, we introduce security tensors - trainable input vectors applied during inference through either the textual or visual modality. These tensors transfer textual safety alignment to visual processing without modifying the model's parameters. They are optimized using a curated dataset containing (i) malicious image-text pairs requiring rejection, (ii) contrastive benign pairs with text structurally similar to malicious queries, with the purpose of being contrastive examples to guide visual reliance, and (iii) general benign samples preserving model functionality. Experimental results demonstrate that both textual and visual security tensors significantly enhance LVLMs' ability to reject diverse harmful visual inputs while maintaining near-identical performance on benign tasks. Further internal analysis towards hidden-layer representations reveals that security tensors successfully activate the language module's textual "safety layers" in visual inputs, thereby effectively extending text-based safety to the visual modality. 

**Abstract (ZH)**: 大型视觉语言模型的安全机制：通过可训练的安全张量在文本和视觉模态间转移文本安全性 

---
# Personalized Treatment Effect Estimation from Unstructured Data 

**Title (ZH)**: 从非结构化数据中估计个性化的治疗效果 

**Authors**: Henri Arno, Thomas Demeester  

**Link**: [PDF](https://arxiv.org/pdf/2507.20993)  

**Abstract**: Existing methods for estimating personalized treatment effects typically rely on structured covariates, limiting their applicability to unstructured data. Yet, leveraging unstructured data for causal inference has considerable application potential, for instance in healthcare, where clinical notes or medical images are abundant. To this end, we first introduce an approximate 'plug-in' method trained directly on the neural representations of unstructured data. However, when these fail to capture all confounding information, the method may be subject to confounding bias. We therefore introduce two theoretically grounded estimators that leverage structured measurements of the confounders during training, but allow estimating personalized treatment effects purely from unstructured inputs, while avoiding confounding bias. When these structured measurements are only available for a non-representative subset of the data, these estimators may suffer from sampling bias. To address this, we further introduce a regression-based correction that accounts for the non-uniform sampling, assuming the sampling mechanism is known or can be well-estimated. Our experiments on two benchmark datasets show that the plug-in method, directly trainable on large unstructured datasets, achieves strong empirical performance across all settings, despite its simplicity. 

**Abstract (ZH)**: 现有的个性化治疗效果估计方法通常依赖于结构化协变量，限制了其在非结构化数据上的应用。然而，利用非结构化数据进行因果推理在诸如医疗健康等领域的应用潜力巨大，例如临床笔记或医疗图像丰 富。为此，我们首先介绍一种直接在非结构化数据的神经表示上训练的近似“插值”方法。然而，当这些方法无法捕捉到所有混杂信息时，可能会出现混杂偏差。因此，我们引入了两种理论上具 有依据的估计器，在训练中利用混杂变量的结构测量，但允许仅从非结构化输入中估计个性化的治疗效果，同时避免混杂偏差。当这些结构化测量仅适用于数据的一个非代表性子集时，这些估计器可能受到抽样偏差的影响。为此，我们进一步引入了一种基于回归的校正方法，以考虑非均匀抽样，前提是抽样机制已知或可以良好估计。我们在两个基准数据集上的实验表明，可以直接在大型非结构化数据集上训练的插值方法，在各种情况下都表现出色，尽管它非常简单。 

---
# JWB-DH-V1: Benchmark for Joint Whole-Body Talking Avatar and Speech Generation Version 1 

**Title (ZH)**: JWB-DH-V1：全面身体对话角色和语音生成基准1.0 

**Authors**: Xinhan Di, Kristin Qi, Pengqian Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.20987)  

**Abstract**: Recent advances in diffusion-based video generation have enabled photo-realistic short clips, but current methods still struggle to achieve multi-modal consistency when jointly generating whole-body motion and natural speech. Current approaches lack comprehensive eval- uation frameworks that assess both visual and audio quality, and there are insufficient benchmarks for region- specific performance analysis. To address these gaps, we introduce the Joint Whole-Body Talking Avatar and Speech Generation Version I(JWB-DH-V1), comprising a large-scale multi-modal dataset with 10,000 unique identities across 2 million video samples, and an evalua- tion protocol for assessing joint audio-video generation of whole-body animatable avatars. Our evaluation of SOTA models reveals consistent performance disparities between face/hand-centric and whole-body performance, which incidates essential areas for future research. The dataset and evaluation tools are publicly available at this https URL. 

**Abstract (ZH)**: 基于扩散的视频生成最近取得了进展，能够生成 PHOTO-REALISTIC 短片段，但当前方法在联合生成全身运动和自然语音时仍难以实现多模态一致性。当前的方法缺乏综合评估框架，无法同时评估视觉和音频质量，也缺乏针对特定区域性能分析的基准数据集。为解决这些不足，我们引入了包含 10,000 个独特身份和 200 万视频样本的大型多模态数据集 JWB-DH-V1 及其评估协议，用于评估可全身动画化的化身的联合音频-视频生成性能。我们的研究表明，人脸/手部为中心的方法与全身方法之间存在一致的性能差异，这表明未来研究的重要领域。数据集和评估工具可在以下网址公开获取：this https URL。 

---
# SmallThinker: A Family of Efficient Large Language Models Natively Trained for Local Deployment 

**Title (ZH)**: 小思考者：一系列本地产地部署的高效大型语言模型 

**Authors**: Yixin Song, Zhenliang Xue, Dongliang Wei, Feiyang Chen, Jianxiang Gao, Junchen Liu, Hangyu Liang, Guangshuo Qin, Chengrong Tian, Bo Wen, Longyu Zhao, Xinrui Zheng, Zeyu Mi, Haibo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.20984)  

**Abstract**: While frontier large language models (LLMs) continue to push capability boundaries, their deployment remains confined to GPU-powered cloud infrastructure. We challenge this paradigm with SmallThinker, a family of LLMs natively designed - not adapted - for the unique constraints of local devices: weak computational power, limited memory, and slow storage. Unlike traditional approaches that mainly compress existing models built for clouds, we architect SmallThinker from the ground up to thrive within these limitations. Our innovation lies in a deployment-aware architecture that transforms constraints into design principles. First, We introduce a two-level sparse structure combining fine-grained Mixture-of-Experts (MoE) with sparse feed-forward networks, drastically reducing computational demands without sacrificing model capacity. Second, to conquer the I/O bottleneck of slow storage, we design a pre-attention router that enables our co-designed inference engine to prefetch expert parameters from storage while computing attention, effectively hiding storage latency that would otherwise cripple on-device inference. Third, for memory efficiency, we utilize NoPE-RoPE hybrid sparse attention mechanism to slash KV cache requirements. We release SmallThinker-4B-A0.6B and SmallThinker-21B-A3B, which achieve state-of-the-art performance scores and even outperform larger LLMs. Remarkably, our co-designed system mostly eliminates the need for expensive GPU hardware: with Q4_0 quantization, both models exceed 20 tokens/s on ordinary consumer CPUs, while consuming only 1GB and 8GB of memory respectively. SmallThinker is publicly available at this http URL and this http URL. 

**Abstract (ZH)**: 尽管前沿大规模语言模型持续推动能力边界，其部署依然局限于GPU驱动的云基础设施。我们以SmallThinker挑战这一范式，这是一个专门为本地设备设计的LLM家族：其独特约束包括弱计算能力、有限内存和缓慢存储。与主要针对云设计的现有模型进行压缩的传统方法不同，我们从头构建SmallThinker，使其能够在这些限制中茁壮成长。我们的创新之处在于一种面向部署的架构，将约束转化为设计原则。首先，我们引入了一种两级稀疏结构，结合了细粒度的混合专家（MoE）和稀疏前馈网络，大幅减少了计算需求，同时不牺牲模型容量。其次，为克服缓慢存储带来的I/O瓶颈，我们设计了一种预注意力路由器，使我们共同设计的推理引擎能够在计算注意力的同时预取专家参数，从而隐藏存储延迟，避免使设备上的推理能力瘫痪。第三，为了提高内存效率，我们利用NoPE-RoPE混合稀疏注意力机制削减了键值缓存要求。我们发布了SmallThinker-4B-A0.6B和SmallThinker-21B-A3B，它们达到最先进的性能得分，并且甚至优于更大规模的LLM。令人惊讶的是，我们的共同设计系统几乎消除了对昂贵GPU硬件的需求：使用Q4_0量化后，两款模型在普通消费级CPU上分别实现了超过20个-token/s的速度，同时分别消耗1GB和8GB的内存。SmallThinker在以下网址公开提供：此网址和此网址。 

---
# From Entanglement to Alignment: Representation Space Decomposition for Unsupervised Time Series Domain Adaptation 

**Title (ZH)**: 从纠缠到对齐：无监督时间序列领域适应的表示空间分解 

**Authors**: Rongyao Cai, Ming Jin, Qingsong Wen, Kexin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20968)  

**Abstract**: Domain shift poses a fundamental challenge in time series analysis, where models trained on source domain often fail dramatically when applied in target domain with different yet similar distributions. While current unsupervised domain adaptation (UDA) methods attempt to align cross-domain feature distributions, they typically treat features as indivisible entities, ignoring their intrinsic compositions that governs domain adaptation. We introduce DARSD, a novel UDA framework with theoretical explainability that explicitly realizes UDA tasks from the perspective of representation space decomposition. Our core insight is that effective domain adaptation requires not just alignment, but principled disentanglement of transferable knowledge from mixed representations. DARSD consists three synergistic components: (I) An adversarial learnable common invariant basis that projects original features into a domain-invariant subspace while preserving semantic content; (II) A prototypical pseudo-labeling mechanism that dynamically separates target features based on confidence, hindering error accumulation; (III) A hybrid contrastive optimization strategy that simultaneously enforces feature clustering and consistency while mitigating emerging distribution gaps. Comprehensive experiments conducted on four benchmark datasets (WISDM, HAR, HHAR, and MFD) demonstrate DARSD's superiority against 12 UDA algorithms, achieving optimal performance in 35 out of 53 cross-domain scenarios. 

**Abstract (ZH)**: 领域偏移在时间序列分析中构成了基本的挑战，其中在源领域训练的模型往往无法有效应用于具有不同但相似分布的目标领域。尽管现有的无监督领域适应（UDA）方法试图对跨领域的特征分布进行对齐，但它们通常将特征视为不可分割的整体，忽略了决定领域适应的内在组成。我们引入了DARSD，这是一种具有理论可解释性的新型UDA框架，从表示空间分解的角度明确地实现了UDA任务。我们的核心见解是，有效的领域适应不仅需要对齐，还需要从混合表示中有原则地分离可转移的知识。DARSD 包含三个协同工作的组件：(I) 一个对抗可学习的公共不变基底，将原始特征投影到域不变子空间同时保留语义内容；(II) 一种基于置信度的原型伪标签机制，动态分离目标特征，防止错误累积；(III) 一种混合对比优化策略，同时强制特征聚类和一致性，并减轻新兴的分布差距。在四个基准数据集（WISDM、HAR、HHAR 和 MFD）上的全面实验表明，DARSD 在 53 个跨域场景中的 35 个场景中实现了最优性能，并优于 12 种UDA算法。 

---
# Handoff Design in User-Centric Cell-Free Massive MIMO Networks Using DRL 

**Title (ZH)**: 用户为中心的无蜂窝大规模MIMO网络的_handover_设计 Using_DRL 

**Authors**: Hussein A. Ammar, Raviraj Adve, Shahram Shahbazpanahi, Gary Boudreau, Israfil Bahceci  

**Link**: [PDF](https://arxiv.org/pdf/2507.20966)  

**Abstract**: In the user-centric cell-free massive MIMO (UC-mMIMO) network scheme, user mobility necessitates updating the set of serving access points to maintain the user-centric clustering. Such updates are typically performed through handoff (HO) operations; however, frequent HOs lead to overheads associated with the allocation and release of resources. This paper presents a deep reinforcement learning (DRL)-based solution to predict and manage these connections for mobile users. Our solution employs the Soft Actor-Critic algorithm, with continuous action space representation, to train a deep neural network to serve as the HO policy. We present a novel proposition for a reward function that integrates a HO penalty in order to balance the attainable rate and the associated overhead related to HOs. We develop two variants of our system; the first one uses mobility direction-assisted (DA) observations that are based on the user movement pattern, while the second one uses history-assisted (HA) observations that are based on the history of the large-scale fading (LSF). Simulation results show that our DRL-based continuous action space approach is more scalable than discrete space counterpart, and that our derived HO policy automatically learns to gather HOs in specific time slots to minimize the overhead of initiating HOs. Our solution can also operate in real time with a response time less than 0.4 ms. 

**Abstract (ZH)**: 基于用户的细胞自由大规模MIMO（UC-mMIMO）网络方案中移动用户的移动性需要更新服务接入点集合以维持用户为中心的聚类。这种更新通常通过切换操作（HO）来执行；然而，频繁的切换会导致与资源分配和释放相关的开销。本文提出了一种基于深度强化学习（DRL）的解决方案，以预测和管理移动用户的这些连接。我们的解决方案利用Soft Actor-Critic算法和连续动作空间表示来训练一个深度神经网络作为切换策略。我们提出了一种新的奖励函数提案，将切换惩罚纳入其中，以平衡可获得速率和与切换相关的开销。我们开发了两种系统变体；第一个变体基于用户移动模式的辅助移动方向（DA）观测，而第二个变体基于大规模衰落（LSF）的历史辅助（HA）观测。仿真结果表明，我们的基于DRL的连续动作空间方法比离散空间方法更具扩展性，并且我们获得的切换策略能够自动学习在特定时间槽中收集切换以最小化切换的启动开销。此外，该解决方案可以实时运行，响应时间少于0.4毫秒。 

---
# Your AI, Not Your View: The Bias of LLMs in Investment Analysis 

**Title (ZH)**: 你的AI，而非你的观点：投资分析中的LLM偏差 

**Authors**: Hoyoung Lee, Junhyuk Seo, Suhwan Park, Junhyeong Lee, Wonbin Ahn, Chanyeol Choi, Alejandro Lopez-Lira, Yongjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.20957)  

**Abstract**: In finance, Large Language Models (LLMs) face frequent knowledge conflicts due to discrepancies between pre-trained parametric knowledge and real-time market data. These conflicts become particularly problematic when LLMs are deployed in real-world investment services, where misalignment between a model's embedded preferences and those of the financial institution can lead to unreliable recommendations. Yet little research has examined what investment views LLMs actually hold. We propose an experimental framework to investigate such conflicts, offering the first quantitative analysis of confirmation bias in LLM-based investment analysis. Using hypothetical scenarios with balanced and imbalanced arguments, we extract models' latent preferences and measure their persistence. Focusing on sector, size, and momentum, our analysis reveals distinct, model-specific tendencies. In particular, we observe a consistent preference for large-cap stocks and contrarian strategies across most models. These preferences often harden into confirmation bias, with models clinging to initial judgments despite counter-evidence. 

**Abstract (ZH)**: 在金融领域，大型语言模型（LLMs）由于预训练参数知识与实时市场数据之间的差异，经常面临知识冲突。当LLMs在现实世界的投资服务中部署时，模型内置的偏好与金融机构的偏好之间的不一致可能导致不可靠的建议。然而，很少有研究考察LLMs实际持有的投资观点。我们提出了一种实验框架来研究这些冲突，并提供了基于LLMs的投资分析中证实偏差的首次定量分析。利用平衡和不平衡的假设情景，我们提取了模型的潜在偏好并测量其持续性。聚焦于行业、规模和动量，我们的分析揭示了不同的、模型特定的趋势。特别是，我们观察到大多数模型中对大型 capital 股票和逆向策略的一致偏好。这些偏好往往会固化为证实偏差，即使有相反的证据，模型也仍坚持初始判断。 

---
# Mind the Gap: Conformative Decoding to Improve Output Diversity of Instruction-Tuned Large Language Models 

**Title (ZH)**: 注意差距：规范解码以提高指令微调大型语言模型的输出多样性 

**Authors**: Max Peeperkorn, Tom Kouwenhoven, Dan Brown, Anna Jordanous  

**Link**: [PDF](https://arxiv.org/pdf/2507.20956)  

**Abstract**: Instruction-tuning large language models (LLMs) reduces the diversity of their outputs, which has implications for many tasks, particularly for creative tasks. This paper investigates the ``diversity gap'' for a writing prompt narrative generation task. This gap emerges as measured by current diversity metrics for various open-weight and open-source LLMs. The results show significant decreases in diversity due to instruction-tuning. We explore the diversity loss at each fine-tuning stage for the OLMo and OLMo 2 models to further understand how output diversity is affected. The results indicate that DPO has the most substantial impact on diversity. Motivated by these findings, we present a new decoding strategy, conformative decoding, which guides an instruct model using its more diverse base model to reintroduce output diversity. We show that conformative decoding typically increases diversity and even maintains or improves quality. 

**Abstract (ZH)**: 大规模语言模型（LLMs）指令调优降低了输出多样性，这对许多任务尤其是创造性任务具有重要意义。本文研究了指令调优对写作提示叙事生成任务中“多样性差距”的影响。通过当前的多样性度量标准测量，这一差距在各种开放权重和开源LLMs中显现出来。结果显示，指令调优导致了显著的多样性下降。我们探讨了OLMo和OLMo 2模型在每个微调阶段的多样性损失情况，以进一步了解输出多样性受到影响的具体方式。结果表明，DPO对多样性的影响最为显著。受这些发现的启发，我们提出了一个新的解码策略——符合性解码，该策略使用更具多样性的基础模型来指导指令模型，重新引入输出多样性。我们证明，符合性解码通常会增加多样性，并且甚至可以维持或提高质量。 

---
# Multivariate Conformal Prediction via Conformalized Gaussian Scoring 

**Title (ZH)**: 多元变量同质化预测via同质化高斯评分 

**Authors**: Sacha Braun, Eugène Berta, Michael I. Jordan, Francis Bach  

**Link**: [PDF](https://arxiv.org/pdf/2507.20941)  

**Abstract**: While achieving exact conditional coverage in conformal prediction is unattainable without making strong, untestable regularity assumptions, the promise of conformal prediction hinges on finding approximations to conditional guarantees that are realizable in practice. A promising direction for obtaining conditional dependence for conformal sets--in particular capturing heteroskedasticity--is through estimating the conditional density $\mathbb{P}_{Y|X}$ and conformalizing its level sets. Previous work in this vein has focused on nonconformity scores based on the empirical cumulative distribution function (CDF). Such scores are, however, computationally costly, typically requiring expensive sampling methods. To avoid the need for sampling, we observe that the CDF-based score reduces to a Mahalanobis distance in the case of Gaussian scores, yielding a closed-form expression that can be directly conformalized. Moreover, the use of a Gaussian-based score opens the door to a number of extensions of the basic conformal method; in particular, we show how to construct conformal sets with missing output values, refine conformal sets as partial information about $Y$ becomes available, and construct conformal sets on transformations of the output space. Finally, empirical results indicate that our approach produces conformal sets that more closely approximate conditional coverage in multivariate settings compared to alternative methods. 

**Abstract (ZH)**: 在不做出强且无法验证的正则性假设的情况下，实现精确条件覆盖在自适应预测中是不可能的，但自适应预测的潜力在于找到可实现的条件保证近似值。获取自适应集的条件依赖性——尤其是捕捉异方差性的方法之一是通过估计条件密度 $\mathbb{P}_{Y|X}$ 并将其等值线自适应化。先前在这方面的工作主要集中在基于经验累积分布函数（CDF）的非一致性评分上。然而，这类评分在计算上成本高昂，通常需要昂贵的采样方法。为了避免采样的需要，我们观察到，在高斯评分的情况下，CDF 基准评分退化为马哈拉诺比斯距离，从而获得一个可以直接自适应化的闭式表达式。此外，基于高斯评分的使用打开了自适应方法若干扩展的可能性；特别是，我们展示了如何构建缺失输出值的自适应集，随着关于 $Y$ 的部分信息变得可用，如何细化自适应集，以及如何在输出空间的变换上构建自适应集。最后，实证结果表明，我们的方法在多变量设置中生成的自适应集更接近条件覆盖，相比其他方法具有显著优势。 

---
# Dissecting Persona-Driven Reasoning in Language Models via Activation Patching 

**Title (ZH)**: 通过激活补丁分析基于人设的语言模型推理解析 

**Authors**: Ansh Poonia, Maeghal Jain  

**Link**: [PDF](https://arxiv.org/pdf/2507.20936)  

**Abstract**: Large language models (LLMs) exhibit remarkable versatility in adopting diverse personas. In this study, we examine how assigning a persona influences a model's reasoning on an objective task. Using activation patching, we take a first step toward understanding how key components of the model encode persona-specific information. Our findings reveal that the early Multi-Layer Perceptron (MLP) layers attend not only to the syntactic structure of the input but also process its semantic content. These layers transform persona tokens into richer representations, which are then used by the middle Multi-Head Attention (MHA) layers to shape the model's output. Additionally, we identify specific attention heads that disproportionately attend to racial and color-based identities. 

**Abstract (ZH)**: 大型语言模型（LLMs）在采用不同人设方面显示出了显著的灵活性。在本研究中，我们探讨了赋予模型人设如何影响其在客观任务上的推理过程。利用激活提取方法，我们首次尝试理解模型的关键组件如何编码人设特定的信息。研究发现，早期的多层感知机（MLP）层不仅关注输入的句法结构，还处理其语义内容。这些层将人设标记转换为更丰富的表示，然后由中间的多头注意力（MHA）层用来塑造模型的输出。此外，我们还识别出对种族和基于肤色身份特别关注的注意力头。 

---
# FRED: Financial Retrieval-Enhanced Detection and Editing of Hallucinations in Language Models 

**Title (ZH)**: FRED: 金融检索增强的语言模型幻觉检测与编辑 

**Authors**: Likun Tan, Kuan-Wei Huang, Kevin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.20930)  

**Abstract**: Hallucinations in large language models pose a critical challenge for applications requiring factual reliability, particularly in high-stakes domains such as finance. This work presents an effective approach for detecting and editing factually incorrect content in model-generated responses based on the provided context. Given a user-defined domain-specific error taxonomy, we construct a synthetic dataset by inserting tagged errors into financial question-answering corpora and then fine-tune four language models, Phi-4, Phi-4-mini, Qwen3-4B, and Qwen3-14B, to detect and edit these factual inaccuracies. Our best-performing model, fine-tuned Phi-4, achieves an 8% improvement in binary F1 score and a 30% gain in overall detection performance compared to OpenAI-o3. Notably, our fine-tuned Phi-4-mini model, despite having only 4 billion parameters, maintains competitive performance with just a 2% drop in binary detection and a 0.1% decline in overall detection compared to OpenAI-o3. Our work provides a practical solution for detecting and editing factual inconsistencies in financial text generation while introducing a generalizable framework that can enhance the trustworthiness and alignment of large language models across diverse applications beyond finance. Our code and data are available at this https URL. 

**Abstract (ZH)**: 大型语言模型中的幻觉对需要事实可靠性的应用构成了关键挑战，特别是在金融等高 stakes 领域。本文提出了一种有效的方法，基于提供的上下文检测和编辑模型生成响应中的事实错误。给定用户定义的领域特定错误分类，我们通过在金融问答语料中插入标记错误构造了一个合成数据集，并对四种语言模型Phi-4、Phi-4-mini、Qwen3-4B和Qwen3-14B进行微调，以检测和编辑这些事实不准确性。我们表现最佳的模型微调后的Phi-4，在二元F1分数上提高了8%，整体检测性能提高了30%，相较于OpenAI-o3。值得注意的是，尽管微调后的Phi-4-mini仅有40亿参数，其二元检测性能下降了2%，整体检测性能下降了0.1%，但仍保持了竞争力。我们的工作提供了一种实用的解决方案，用于检测和编辑金融文本生成中的事实不一致，同时引入了一种可推广的框架，可以在超出金融领域的各种应用中增强大型语言模型的信任度和对齐。我们的代码和数据可在以下链接获取。 

---
# FHSTP@EXIST 2025 Benchmark: Sexism Detection with Transparent Speech Concept Bottleneck Models 

**Title (ZH)**: FHSTP@EXIST 2025 基准：透明语音概念瓶颈模型中的性别偏见检测 

**Authors**: Roberto Labadie-Tamayo, Adrian Jaques Böck, Djordje Slijepčević, Xihui Chen, Andreas Babic, Matthias Zeppelzauer  

**Link**: [PDF](https://arxiv.org/pdf/2507.20924)  

**Abstract**: Sexism has become widespread on social media and in online conversation. To help address this issue, the fifth Sexism Identification in Social Networks (EXIST) challenge is initiated at CLEF 2025. Among this year's international benchmarks, we concentrate on solving the first task aiming to identify and classify sexism in social media textual posts. In this paper, we describe our solutions and report results for three subtasks: Subtask 1.1 - Sexism Identification in Tweets, Subtask 1.2 - Source Intention in Tweets, and Subtask 1.3 - Sexism Categorization in Tweets. We implement three models to address each subtask which constitute three individual runs: Speech Concept Bottleneck Model (SCBM), Speech Concept Bottleneck Model with Transformer (SCBMT), and a fine-tuned XLM-RoBERTa transformer model. SCBM uses descriptive adjectives as human-interpretable bottleneck concepts. SCBM leverages large language models (LLMs) to encode input texts into a human-interpretable representation of adjectives, then used to train a lightweight classifier for downstream tasks. SCBMT extends SCBM by fusing adjective-based representation with contextual embeddings from transformers to balance interpretability and classification performance. Beyond competitive results, these two models offer fine-grained explanations at both instance (local) and class (global) levels. We also investigate how additional metadata, e.g., annotators' demographic profiles, can be leveraged. For Subtask 1.1, XLM-RoBERTa, fine-tuned on provided data augmented with prior datasets, ranks 6th for English and Spanish and 4th for English in the Soft-Soft evaluation. Our SCBMT achieves 7th for English and Spanish and 6th for Spanish. 

**Abstract (ZH)**: 性别歧视在社交媒体和在线对话中愈演愈烈。为解决这一问题，CLEF 2025年启动了第五屆社交网络性别歧视识别（EXIST）挑战。在本年的国际基准中，我们集中于解决首个任务，即识别和分类社交媒体文本帖子中的性别歧视。在本文中，我们描述了我们的解决方案并报告了三个子任务的结果：子任务1.1 - 微博中的性别歧视识别、子任务1.2 - 微博中的来源意图、子任务1.3 - 微博中的性别歧视分类。我们实现了三种模型来解决每个子任务，构成三个独立的运行：言语概念瓶颈模型（SCBM）、带变换器的言语概念瓶颈模型（SCBMT）以及微调后的XLM-RoBERTa变换器模型。SCBM利用描述性形容词作为人类可解释的瓶颈概念。SCBM利用大规模语言模型（LLMs）将输入文本编码为形容词的人类可解释表示，然后用于下游任务的轻量级分类器训练。SCBMT在基于描述词的表示上结合了变换器的语境嵌入，以平衡可解释性和分类性能。除了竞争性结果外，这两种模型还在实例（局部）和类别（全局）层面提供了详尽的解释。我们还研究了如何利用额外的元数据，例如注释者的 demographic 背景。对于子任务1.1，XLM-RoBERTa，在软-软评估中，英语和西班牙语分别排名第6和第4。我们的SCBMT在英语和西班牙语中分别排名第7和第6，在西班牙语中排名第6。 

---
# Pareto-Grid-Guided Large Language Models for Fast and High-Quality Heuristics Design in Multi-Objective Combinatorial Optimization 

**Title (ZH)**: Pareto-网格引导的大语言模型在多目标组合优化中快速高质设计启发式方法 

**Authors**: Minh Hieu Ha, Hung Phan, Tung Duy Doan, Tung Dao, Dao Tran, Huynh Thi Thanh Binh  

**Link**: [PDF](https://arxiv.org/pdf/2507.20923)  

**Abstract**: Multi-objective combinatorial optimization problems (MOCOP) frequently arise in practical applications that require the simultaneous optimization of conflicting objectives. Although traditional evolutionary algorithms can be effective, they typically depend on domain knowledge and repeated parameter tuning, limiting flexibility when applied to unseen MOCOP instances. Recently, integration of Large Language Models (LLMs) into evolutionary computation has opened new avenues for automatic heuristic generation, using their advanced language understanding and code synthesis capabilities. Nevertheless, most existing approaches predominantly focus on single-objective tasks, often neglecting key considerations such as runtime efficiency and heuristic diversity in multi-objective settings. To bridge this gap, we introduce Multi-heuristics for MOCOP via Pareto-Grid-guided Evolution of LLMs (MPaGE), a novel enhancement of the Simple Evolutionary Multiobjective Optimization (SEMO) framework that leverages LLMs and Pareto Front Grid (PFG) technique. By partitioning the objective space into grids and retaining top-performing candidates to guide heuristic generation, MPaGE utilizes LLMs to prioritize heuristics with semantically distinct logical structures during variation, thus promoting diversity and mitigating redundancy within the population. Through extensive evaluations, MPaGE demonstrates superior performance over existing LLM-based frameworks, and achieves competitive results to traditional Multi-objective evolutionary algorithms (MOEAs), with significantly faster runtime. Our code is available at: this https URL. 

**Abstract (ZH)**: 多目标组合优化问题（MOCOP）经常出现在需要同时优化冲突目标的实际应用中。虽然传统的进化算法通常是有效的，但它们通常依赖于领域知识并且需要反复调整参数，这限制了它们在未见过的MOCOP实例中的灵活性。近年来，将大型语言模型（LLMs）集成到进化计算中为自动启发式生成开辟了新的途径，利用其先进的语言理解和代码合成能力。然而，现有的大多数方法主要集中在单目标任务上，经常忽视多目标设置中的重要考虑因素，如运行时效率和启发式多样性。为了解决这一问题，我们提出了通过帕累托网格引导的LLM进化进行多启发式优化（MPaGE），这是一种对简单多目标进化优化（SEMO）框架的创新改进，利用了LLMs和帕累托前沿网格（PFG）技术。通过将目标空间划分成网格，并保留表现优异的候选者以引导启发式生成，MPaGE利用LLMs在变异过程中优先选择语义上不同的逻辑结构的启发式，从而促进多样性和减少种群中的冗余。通过广泛的评估，MPaGE在基于LLM的现有框架中表现出更优的性能，并且在运行时速度上有显著提升，达到传统多目标进化算法（MOEAs）的竞争力。我们的代码可在以下链接获得：this https URL。 

---
# Modeling User Behavior from Adaptive Surveys with Supplemental Context 

**Title (ZH)**: 基于补充上下文的自适应调查中用户行为建模 

**Authors**: Aman Shukla, Daniel Patrick Scantlebury, Rishabh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2507.20919)  

**Abstract**: Modeling user behavior is critical across many industries where understanding preferences, intent, or decisions informs personalization, targeting, and strategic outcomes. Surveys have long served as a classical mechanism for collecting such behavioral data due to their interpretability, structure, and ease of deployment. However, surveys alone are inherently limited by user fatigue, incomplete responses, and practical constraints on their length making them insufficient for capturing user behavior. In this work, we present LANTERN (Late-Attentive Network for Enriched Response Modeling), a modular architecture for modeling user behavior by fusing adaptive survey responses with supplemental contextual signals. We demonstrate the architectural value of maintaining survey primacy through selective gating, residual connections and late fusion via cross-attention, treating survey data as the primary signal while incorporating external modalities only when relevant. LANTERN outperforms strong survey-only baselines in multi-label prediction of survey responses. We further investigate threshold sensitivity and the benefits of selective modality reliance through ablation and rare/frequent attribute analysis. LANTERN's modularity supports scalable integration of new encoders and evolving datasets. This work provides a practical and extensible blueprint for behavior modeling in survey-centric applications. 

**Abstract (ZH)**: 基于晚期注意网络的增强回应建模（LANTERN：Late-Attentive Network for Enriched Response Modeling） 

---
# MediQAl: A French Medical Question Answering Dataset for Knowledge and Reasoning Evaluation 

**Title (ZH)**: MediQAl: 一个用于知识和推理评估的法语医疗问答数据集 

**Authors**: Adrien Bazoge  

**Link**: [PDF](https://arxiv.org/pdf/2507.20917)  

**Abstract**: This work introduces MediQAl, a French medical question answering dataset designed to evaluate the capabilities of language models in factual medical recall and reasoning over real-world clinical scenarios. MediQAl contains 32,603 questions sourced from French medical examinations across 41 medical subjects. The dataset includes three tasks: (i) Multiple-Choice Question with Unique answer, (ii) Multiple-Choice Question with Multiple answer, and (iii) Open-Ended Question with Short-Answer. Each question is labeled as Understanding or Reasoning, enabling a detailed analysis of models' cognitive capabilities. We validate the MediQAl dataset through extensive evaluation with 14 large language models, including recent reasoning-augmented models, and observe a significant performance gap between factual recall and reasoning tasks. Our evaluation provides a comprehensive benchmark for assessing language models' performance on French medical question answering, addressing a crucial gap in multilingual resources for the medical domain. 

**Abstract (ZH)**: MediQAl：一种用于评估语言模型在医疗事实回忆和现实临床场景推理能力的法语文本问答数据集 

---
# HAMLET-FFD: Hierarchical Adaptive Multi-modal Learning Embeddings Transformation for Face Forgery Detection 

**Title (ZH)**: HAMLET-FFD：分层自适应多模态学习嵌入转换对面部伪造检测的研究 

**Authors**: Jialei Cui, Jianwei Du, Yanzhe Li, Lei Gao, Hui Jiang, Chenfu Bao  

**Link**: [PDF](https://arxiv.org/pdf/2507.20913)  

**Abstract**: The rapid evolution of face manipulation techniques poses a critical challenge for face forgery detection: cross-domain generalization. Conventional methods, which rely on simple classification objectives, often fail to learn domain-invariant representations. We propose HAMLET-FFD, a cognitively inspired Hierarchical Adaptive Multi-modal Learning framework that tackles this challenge via bidirectional cross-modal reasoning. Building on contrastive vision-language models such as CLIP, HAMLET-FFD introduces a knowledge refinement loop that iteratively assesses authenticity by integrating visual evidence with conceptual cues, emulating expert forensic analysis. A key innovation is a bidirectional fusion mechanism in which textual authenticity embeddings guide the aggregation of hierarchical visual features, while modulated visual features refine text embeddings to generate image-adaptive prompts. This closed-loop process progressively aligns visual observations with semantic priors to enhance authenticity assessment. By design, HAMLET-FFD freezes all pretrained parameters, serving as an external plugin that preserves CLIP's original capabilities. Extensive experiments demonstrate its superior generalization to unseen manipulations across multiple benchmarks, and visual analyses reveal a division of labor among embeddings, with distinct representations specializing in fine-grained artifact recognition. 

**Abstract (ZH)**: 基于认知的分级适应多模态学习框架（HAMLET-FFD）：面向前所未见的面部伪造跨域泛化功能提升方法 

---
# SCORPION: Addressing Scanner-Induced Variability in Histopathology 

**Title (ZH)**: SCORPION: 应对扫描引起的病理图像变异性问题 

**Authors**: Jeongun Ryu, Heon Song, Seungeun Lee, Soo Ick Cho, Jiwon Shin, Kyunghyun Paeng, Sérgio Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2507.20907)  

**Abstract**: Ensuring reliable model performance across diverse domains is a critical challenge in computational pathology. A particular source of variability in Whole-Slide Images is introduced by differences in digital scanners, thus calling for better scanner generalization. This is critical for the real-world adoption of computational pathology, where the scanning devices may differ per institution or hospital, and the model should not be dependent on scanner-induced details, which can ultimately affect the patient's diagnosis and treatment planning. However, past efforts have primarily focused on standard domain generalization settings, evaluating on unseen scanners during training, without directly evaluating consistency across scanners for the same tissue. To overcome this limitation, we introduce SCORPION, a new dataset explicitly designed to evaluate model reliability under scanner variability. SCORPION includes 480 tissue samples, each scanned with 5 scanners, yielding 2,400 spatially aligned patches. This scanner-paired design allows for the isolation of scanner-induced variability, enabling a rigorous evaluation of model consistency while controlling for differences in tissue composition. Furthermore, we propose SimCons, a flexible framework that combines augmentation-based domain generalization techniques with a consistency loss to explicitly address scanner generalization. We empirically show that SimCons improves model consistency on varying scanners without compromising task-specific performance. By releasing the SCORPION dataset and proposing SimCons, we provide the research community with a crucial resource for evaluating and improving model consistency across diverse scanners, setting a new standard for reliability testing. 

**Abstract (ZH)**: 确保计算病理学在多样化领域中模型性能的可靠性是一个关键挑战。由数字扫描器差异引入的 Whole-Slide Images 变异性要求更好的扫描器泛化能力。这对于计算病理学的实际应用至关重要，因为扫描设备在不同机构或医院之间可能不同，模型不应依赖于扫描器引起的细节，这些细节最终可能会影响患者的诊断和治疗规划。然而，过去的努力主要集中在标准领域泛化设置上，在训练期间评估未见过的扫描器，而没有直接评估相同组织在不同扫描器之间的前后一致性。为了克服这一局限性，我们引入了 SCORPION，一个明确设计用于评估模型在扫描器变异性下的可靠性的新数据集。SCORPION 包含 480 个组织样本，每个样本使用 5 种扫描器进行扫描，生成 2,400 个空间对齐的切片。这种扫描器配对设计允许隔离由于扫描器引起的变异，从而能够控制组织组成差异的同时进行严格的模型一致性评估。此外，我们提出了一种名为 SimCons 的灵活框架，该框架结合了基于增强的领域泛化技术与一致性损失，明确解决扫描器泛化问题。我们实证表明，SimCons 在不同扫描器下提高了模型一致性，同时不牺牲特定任务的性能。通过发布 SCORPION 数据集并提出 SimCons，我们为研究界提供了评估和提高不同扫描器下模型一致性的关键资源，树立了可靠测试的新标准。 

---
# Music Arena: Live Evaluation for Text-to-Music 

**Title (ZH)**: 音乐竞技场：文本到音乐的现场评估 

**Authors**: Yonghyun Kim, Wayne Chi, Anastasios N. Angelopoulos, Wei-Lin Chiang, Koichi Saito, Shinji Watanabe, Yuki Mitsufuji, Chris Donahue  

**Link**: [PDF](https://arxiv.org/pdf/2507.20900)  

**Abstract**: We present Music Arena, an open platform for scalable human preference evaluation of text-to-music (TTM) models. Soliciting human preferences via listening studies is the gold standard for evaluation in TTM, but these studies are expensive to conduct and difficult to compare, as study protocols may differ across systems. Moreover, human preferences might help researchers align their TTM systems or improve automatic evaluation metrics, but an open and renewable source of preferences does not currently exist. We aim to fill these gaps by offering *live* evaluation for TTM. In Music Arena, real-world users input text prompts of their choosing and compare outputs from two TTM systems, and their preferences are used to compile a leaderboard. While Music Arena follows recent evaluation trends in other AI domains, we also design it with key features tailored to music: an LLM-based routing system to navigate the heterogeneous type signatures of TTM systems, and the collection of *detailed* preferences including listening data and natural language feedback. We also propose a rolling data release policy with user privacy guarantees, providing a renewable source of preference data and increasing platform transparency. Through its standardized evaluation protocol, transparent data access policies, and music-specific features, Music Arena not only addresses key challenges in the TTM ecosystem but also demonstrates how live evaluation can be thoughtfully adapted to unique characteristics of specific AI domains.
Music Arena is available at: this https URL 

**Abstract (ZH)**: 音乐竞技场：一个面向文本到音乐模型可扩展的人类偏好评估开放平台 

---
# JAM: A Tiny Flow-based Song Generator with Fine-grained Controllability and Aesthetic Alignment 

**Title (ZH)**: JAM：一个具备细粒度可控性和审美对齐的小型流基歌曲生成器 

**Authors**: Renhang Liu, Chia-Yu Hung, Navonil Majumder, Taylor Gautreaux, Amir Ali Bagherzadeh, Chuan Li, Dorien Herremans, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2507.20880)  

**Abstract**: Diffusion and flow-matching models have revolutionized automatic text-to-audio generation in recent times. These models are increasingly capable of generating high quality and faithful audio outputs capturing to speech and acoustic events. However, there is still much room for improvement in creative audio generation that primarily involves music and songs. Recent open lyrics-to-song models, such as, DiffRhythm, ACE-Step, and LeVo, have set an acceptable standard in automatic song generation for recreational use. However, these models lack fine-grained word-level controllability often desired by musicians in their workflows. To the best of our knowledge, our flow-matching-based JAM is the first effort toward endowing word-level timing and duration control in song generation, allowing fine-grained vocal control. To enhance the quality of generated songs to better align with human preferences, we implement aesthetic alignment through Direct Preference Optimization, which iteratively refines the model using a synthetic dataset, eliminating the need or manual data annotations. Furthermore, we aim to standardize the evaluation of such lyrics-to-song models through our public evaluation dataset JAME. We show that JAM outperforms the existing models in terms of the music-specific attributes. 

**Abstract (ZH)**: 基于流匹配的JAM模型：面向歌词到歌曲生成的字级节奏和时长控制及美学对齐标准 

---
# Not Only Grey Matter: OmniBrain for Robust Multimodal Classification of Alzheimer's Disease 

**Title (ZH)**: 不仅灰质：OmniBrain在阿尔茨海默病多模态分类中的稳健表现 

**Authors**: Ahmed Sharshar, Yasser Ashraf, Tameem Bakr, Salma Hassan, Hosam Elgendy, Mohammad Yaqub, Mohsen Guizani  

**Link**: [PDF](https://arxiv.org/pdf/2507.20872)  

**Abstract**: Alzheimer's disease affects over 55 million people worldwide and is projected to more than double by 2050, necessitating rapid, accurate, and scalable diagnostics. However, existing approaches are limited because they cannot achieve clinically acceptable accuracy, generalization across datasets, robustness to missing modalities, and explainability all at the same time. This inability to satisfy all these requirements simultaneously undermines their reliability in clinical settings. We propose OmniBrain, a multimodal framework that integrates brain MRI, radiomics, gene expression, and clinical data using a unified model with cross-attention and modality dropout. OmniBrain achieves $92.2 \pm 2.4\%$accuracy on the ANMerge dataset and generalizes to the MRI-only ADNI dataset with $70.4 \pm 2.7\%$ accuracy, outperforming unimodal and prior multimodal approaches. Explainability analyses highlight neuropathologically relevant brain regions and genes, enhancing clinical trust. OmniBrain offers a robust, interpretable, and practical solution for real-world Alzheimer's diagnosis. 

**Abstract (ZH)**: 阿尔茨海默病影响全球超过5500万人，预计到2050年将翻倍，亟需快速、准确且可扩展的诊断方法。然而，现有方法受限于无法同时实现临床可接受的准确性、跨数据集的普适性、对缺失模块的鲁棒性以及可解释性。这种无法同时满足所有要求的能力削弱了其在临床环境中的可靠性。我们提出OmniBrain，这是一种利用统一模型结合交叉注意力和模块 dropout 的多模态框架，整合了脑MRI、影像组学、基因表达和临床数据。OmniBrain在ANMerge数据集上达到了92.2 ± 2.4%的准确率，并在仅使用MRI的ADNI数据集上达到了70.4 ± 2.7%的准确率，超越了单模态和先前的多模态方法。可解释性分析突出显示了神经病理相关的脑区和基因，增强了临床信任。OmniBrain提供了一种稳健、可解释且实用的阿尔茨海默病诊断解决方案。 

---
# Geometry of Neural Reinforcement Learning in Continuous State and Action Spaces 

**Title (ZH)**: 几何视角下的神经强化学习在连续状态和行动空间中的研究 

**Authors**: Saket Tiwari, Omer Gottesman, George Konidaris  

**Link**: [PDF](https://arxiv.org/pdf/2507.20853)  

**Abstract**: Advances in reinforcement learning (RL) have led to its successful application in complex tasks with continuous state and action spaces. Despite these advances in practice, most theoretical work pertains to finite state and action spaces. We propose building a theoretical understanding of continuous state and action spaces by employing a geometric lens to understand the locally attained set of states. The set of all parametrised policies learnt through a semi-gradient based approach induces a set of attainable states in RL. We show that the training dynamics of a two-layer neural policy induce a low dimensional manifold of attainable states embedded in the high-dimensional nominal state space trained using an actor-critic algorithm. We prove that, under certain conditions, the dimensionality of this manifold is of the order of the dimensionality of the action space. This is the first result of its kind, linking the geometry of the state space to the dimensionality of the action space. We empirically corroborate this upper bound for four MuJoCo environments and also demonstrate the results in a toy environment with varying dimensionality. We also show the applicability of this theoretical result by introducing a local manifold learning layer to the policy and value function networks to improve the performance in control environments with very high degrees of freedom by changing one layer of the neural network to learn sparse representations. 

**Abstract (ZH)**: 强化学习（RL）的进步使其成功应用于具有连续状态和动作空间的复杂任务。尽管在实际应用中取得了这些进展，大多数理论工作仍集中在有限状态和动作空间。我们提出通过几何视角理解局部达到的状态集来建立对连续状态和动作空间的理论理解。通过半梯度方法学习的参数化策略集在RL中诱导出可达到的状态集。我们证明，使用actor-critic算法训练两层神经策略的动力学诱导出嵌入在高维名义状态空间中的低维流形。在满足某些条件下，该流形的维数与动作空间的维数成比例。这是这一领域的首个结果，将状态空间的几何结构与动作空间的维数联系起来。我们通过四个MuJoCo环境的经验验证确认了这一上界，并在不同维数的玩具环境中也证明了该结果。此外，通过在策略和价值函数网络中引入局部流形学习层，我们展示了这一理论结果的应用性，通过改变神经网络的一层来学习稀疏表示，从而提高具有非常高自由度的控制环境中的性能。 

---
# Free Energy-Inspired Cognitive Risk Integration for AV Navigation in Pedestrian-Rich Environments 

**Title (ZH)**: 受自由能启发的认知风险集成在行人密集环境中的自动驾驶导航 

**Authors**: Meiting Dang, Yanping Wu, Yafei Wang, Dezong Zhao, David Flynn, Chongfeng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2507.20850)  

**Abstract**: Recent advances in autonomous vehicle (AV) behavior planning have shown impressive social interaction capabilities when interacting with other road users. However, achieving human-like prediction and decision-making in interactions with vulnerable road users remains a key challenge in complex multi-agent interactive environments. Existing research focuses primarily on crowd navigation for small mobile robots, which cannot be directly applied to AVs due to inherent differences in their decision-making strategies and dynamic boundaries. Moreover, pedestrians in these multi-agent simulations follow fixed behavior patterns that cannot dynamically respond to AV actions. To overcome these limitations, this paper proposes a novel framework for modeling interactions between the AV and multiple pedestrians. In this framework, a cognitive process modeling approach inspired by the Free Energy Principle is integrated into both the AV and pedestrian models to simulate more realistic interaction dynamics. Specifically, the proposed pedestrian Cognitive-Risk Social Force Model adjusts goal-directed and repulsive forces using a fused measure of cognitive uncertainty and physical risk to produce human-like trajectories. Meanwhile, the AV leverages this fused risk to construct a dynamic, risk-aware adjacency matrix for a Graph Convolutional Network within a Soft Actor-Critic architecture, allowing it to make more reasonable and informed decisions. Simulation results indicate that our proposed framework effectively improves safety, efficiency, and smoothness of AV navigation compared to the state-of-the-art method. 

**Abstract (ZH)**: Recent Advances in Autonomous Vehicle Behavior Planning for Realistic Interaction with Vulnerable Road Users Through a Novel Cognitive-Risk Framework 

---
# First Hallucination Tokens Are Different from Conditional Ones 

**Title (ZH)**: 第一类幻觉词与条件词不同 

**Authors**: Jakob Snel, Seong Joon Oh  

**Link**: [PDF](https://arxiv.org/pdf/2507.20836)  

**Abstract**: Hallucination, the generation of untruthful content, is one of the major concerns regarding foundational models. Detecting hallucinations at the token level is vital for real-time filtering and targeted correction, yet the variation of hallucination signals within token sequences is not fully understood. Leveraging the RAGTruth corpus with token-level annotations and reproduced logits, we analyse how these signals depend on a token's position within hallucinated spans, contributing to an improved understanding of token-level hallucination. Our results show that the first hallucinated token carries a stronger signal and is more detectable than conditional tokens. We release our analysis framework, along with code for logit reproduction and metric computation at this https URL. 

**Abstract (ZH)**: 幻觉，即生成虚假内容，是基础模型面临的重大问题之一。基于标记到令牌级别的RAGTruth语料库和重构的logits，我们分析这些信号如何依赖于幻觉片段中令牌的位置，从而增进对令牌级幻觉的理解。我们的结果显示，首个生成的幻觉令牌比条件令牌携带更强的信号且更容易检测。我们发布了解析框架及相关代码，详见此链接：https://this.is/url。 

---
# Why Flow Matching is Particle Swarm Optimization? 

**Title (ZH)**: 流匹配为何是粒子群优化？ 

**Authors**: Kaichen Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20810)  

**Abstract**: This paper preliminarily investigates the duality between flow matching in generative models and particle swarm optimization (PSO) in evolutionary computation. Through theoretical analysis, we reveal the intrinsic connections between these two approaches in terms of their mathematical formulations and optimization mechanisms: the vector field learning in flow matching shares similar mathematical expressions with the velocity update rules in PSO; both methods follow the fundamental framework of progressive evolution from initial to target distributions; and both can be formulated as dynamical systems governed by ordinary differential equations. Our study demonstrates that flow matching can be viewed as a continuous generalization of PSO, while PSO provides a discrete implementation of swarm intelligence principles. This duality understanding establishes a theoretical foundation for developing novel hybrid algorithms and creates a unified framework for analyzing both methods. Although this paper only presents preliminary discussions, the revealed correspondences suggest several promising research directions, including improving swarm intelligence algorithms based on flow matching principles and enhancing generative models using swarm intelligence concepts. 

**Abstract (ZH)**: 本文初步探讨了生成模型中的流动匹配与进化计算中的粒子群优化（PSO）之间的二元关系。通过理论分析，我们揭示了这两种方法在数学形式和优化机制方面的内在联系：流动匹配中的向量场学习与PSO中的速度更新规则具有类似的数学表达式；两者都遵循从初始分布到目标分布的逐步进化框架；两者都可以用由常微分方程支配的动力系统进行形式化描述。本研究证明，流动匹配可以被视为PSO的连续推广，而PSO则提供了群体智能原则的离散实现。这种二元关系的理解为开发新的混合算法并建立统一的分析框架奠定了理论基础。尽管本文仅呈现了初步讨论，但揭示的对应关系提出了几个有前景的研究方向，包括基于流动匹配原则改进群体智能算法和利用群体智能概念增强生成模型。 

---
# LanternNet: A Novel Hub-and-Spoke System to Seek and Suppress Spotted Lanternfly Populations 

**Title (ZH)**: LanternNet: 一种Seek and Suppressolib硫斑萤叶甲种群的新颖hub-and-spoke系统 

**Authors**: Vinil Polepalli  

**Link**: [PDF](https://arxiv.org/pdf/2507.20800)  

**Abstract**: The invasive spotted lanternfly (SLF) poses a significant threat to agriculture and ecosystems, causing widespread damage. Current control methods, such as egg scraping, pesticides, and quarantines, prove labor-intensive, environmentally hazardous, and inadequate for long-term SLF suppression. This research introduces LanternNet, a novel autonomous robotic Hub-and-Spoke system designed for scalable detection and suppression of SLF populations. A central, tree-mimicking hub utilizes a YOLOv8 computer vision model for precise SLF identification. Three specialized robotic spokes perform targeted tasks: pest neutralization, environmental monitoring, and navigation/mapping. Field deployment across multiple infested sites over 5 weeks demonstrated LanternNet's efficacy. Quantitative analysis revealed significant reductions (p < 0.01, paired t-tests) in SLF populations and corresponding improvements in tree health indicators across the majority of test sites. Compared to conventional methods, LanternNet offers substantial cost advantages and improved scalability. Furthermore, the system's adaptability for enhanced autonomy and targeting of other invasive species presents significant potential for broader ecological impact. LanternNet demonstrates the transformative potential of integrating robotics and AI for advanced invasive species management and improved environmental outcomes. 

**Abstract (ZH)**: 入侵的灯笼蝉（SLF）对农业和生态系统构成重大威胁，造成广泛损害。目前的控制方法，如人工剔除卵块、使用农药和检疫措施，证明劳动密集、环境危害大且无法长期有效抑制SLF种群。本研究介绍了一种新型自主机器人Hub-and-Spoke系统LanternNet，用于 scalable 检测和抑制SLF种群。中央树形模仿中心利用YOLOv8计算机视觉模型进行精确的SLF识别。三个专门设计的机器人辐条执行特定任务：害虫中和、环境监测和导航/制图。在多个受侵地区为期5周的实地部署显示了LanternNet的有效性。定量分析表明，在大多数测试地区，SLF种群显著减少（p < 0.01，配对t检验），同时树健康指标有所改善。与传统方法相比，LanternNet在成本优势和可扩展性方面具有显著优势。此外，该系统针对其他入侵物种的增强自主性和适应性表明了对更广泛生态影响的潜在重要性。LanternNet展示了将机器人技术和AI集成用于先进入侵物种管理和改善环境结果的变革潜力。 

---
# Aligning Large Language Model Agents with Rational and Moral Preferences: A Supervised Fine-Tuning Approach 

**Title (ZH)**: 使大型语言模型代理与理性与道德的偏好相一致：一种监督微调方法 

**Authors**: Wei Lu, Daniel L. Chen, Christian B. Hansen  

**Link**: [PDF](https://arxiv.org/pdf/2507.20796)  

**Abstract**: Understanding how large language model (LLM) agents behave in strategic interactions is essential as these systems increasingly participate autonomously in economically and morally consequential decisions. We evaluate LLM preferences using canonical economic games, finding substantial deviations from human behavior. Models like GPT-4o show excessive cooperation and limited incentive sensitivity, while reasoning models, such as o3-mini, align more consistently with payoff-maximizing strategies. We propose a supervised fine-tuning pipeline that uses synthetic datasets derived from economic reasoning to align LLM agents with economic preferences, focusing on two stylized preference structures. In the first, utility depends only on individual payoffs (homo economicus), while utility also depends on a notion of Kantian universalizability in the second preference structure (homo moralis). We find that fine-tuning based on small datasets shifts LLM agent behavior toward the corresponding economic agent. We further assess the fine-tuned agents' behavior in two applications: Moral dilemmas involving autonomous vehicles and algorithmic pricing in competitive markets. These examples illustrate how different normative objectives embedded via realizations from structured preference structures can influence market and moral outcomes. This work contributes a replicable, cost-efficient, and economically grounded pipeline to align AI preferences using moral-economic principles. 

**Abstract (ZH)**: 理解大型语言模型代理在战略互动中的行为对于这些系统在经济和道德上有重要意义的决策中自主参与至关重要。我们使用经典的经济博弈来评估LLM的偏好，发现其行为与人类行为存在显著偏差。如GPT-4o这类模型表现出过度合作和激励敏感性有限的特点，而如o3-mini这类基于推理的模型则更一致地符合最大化收益的战略。我们提出了一种监督微调管道，使用源自经济推理的合成数据集来使LLM代理与经济偏好相一致，重点关注两种典型偏好结构。在第一种结构中，效用仅依赖于个体收益（经济人），而在第二种结构中，效用还依赖于一种批判性的普遍化概念（道德人）。我们发现基于小型数据集的微调能使LLM代理行为向相应的经济代理靠拢。进一步在两个应用中评估微调后的代理行为：涉及自主车辆的道德困境和竞争市场中的算法定价。这些示例展示了通过结构化偏好结构嵌入的不同规范性目标如何影响市场和道德结果。本研究贡献了一种可复制、成本效益高且基于经济原则的人工智能偏好对齐管道。 

---
# Investigation of Accuracy and Bias in Face Recognition Trained with Synthetic Data 

**Title (ZH)**: 基于合成数据训练的脸部识别准确度与偏差investigation 

**Authors**: Pavel Korshunov, Ketan Kotwal, Christophe Ecabert, Vidit Vidit, Amir Mohammadi, Sebastien Marcel  

**Link**: [PDF](https://arxiv.org/pdf/2507.20782)  

**Abstract**: Synthetic data has emerged as a promising alternative for training face recognition (FR) models, offering advantages in scalability, privacy compliance, and potential for bias mitigation. However, critical questions remain on whether both high accuracy and fairness can be achieved with synthetic data. In this work, we evaluate the impact of synthetic data on bias and performance of FR systems. We generate balanced face dataset, FairFaceGen, using two state of the art text-to-image generators, Flux.1-dev and Stable Diffusion v3.5 (SD35), and combine them with several identity augmentation methods, including Arc2Face and four IP-Adapters. By maintaining equal identity count across synthetic and real datasets, we ensure fair comparisons when evaluating FR performance on standard (LFW, AgeDB-30, etc.) and challenging IJB-B/C benchmarks and FR bias on Racial Faces in-the-Wild (RFW) dataset. Our results demonstrate that although synthetic data still lags behind the real datasets in the generalization on IJB-B/C, demographically balanced synthetic datasets, especially those generated with SD35, show potential for bias mitigation. We also observe that the number and quality of intra-class augmentations significantly affect FR accuracy and fairness. These findings provide practical guidelines for constructing fairer FR systems using synthetic data. 

**Abstract (ZH)**: 合成数据已成为训练面部识别（FR）模型的一种有前景的替代方案，提供了规模化、隐私合规和偏差缓解的潜力。然而，关于是否能够同时实现高准确性和公平性的问题仍然关键。在此工作中，我们评估了合成数据对面部识别系统偏差和性能的影响。我们使用两个最先进的文本到图像生成器Flux.1-dev和Stable Diffusion v3.5（SD35）生成一个平衡面部数据集FairFaceGen，并结合了几种身份增强方法，包括Arc2Face和四个IP-适配器。通过在合成和真实数据集中保持相同的身份数量，我们在评估标准（LFW、AgeDB-30等）和具有挑战性的IJB-B/C基准以及RFW数据集上的面部识别偏差时确保了公平比较。结果显示，尽管合成数据在IJB-B/C上的泛化能力仍落后于真实数据集，但人口统计学平衡的合成数据集，尤其是使用SD35生成的数据集，具有缓解偏差的潜力。我们还观察到，类内增强的数量和质量显著影响面部识别的准确性和公平性。这些发现为使用合成数据构建更公平的面部识别系统提供了实用指南。 

---
# Learning to See Inside Opaque Liquid Containers using Speckle Vibrometry 

**Title (ZH)**: 使用斑点振动技术观察不透明液体容器内部 

**Authors**: Matan Kichler, Shai Bagon, Mark Sheinin  

**Link**: [PDF](https://arxiv.org/pdf/2507.20757)  

**Abstract**: Computer vision seeks to infer a wide range of information about objects and events. However, vision systems based on conventional imaging are limited to extracting information only from the visible surfaces of scene objects. For instance, a vision system can detect and identify a Coke can in the scene, but it cannot determine whether the can is full or empty. In this paper, we aim to expand the scope of computer vision to include the novel task of inferring the hidden liquid levels of opaque containers by sensing the tiny vibrations on their surfaces. Our method provides a first-of-a-kind way to inspect the fill level of multiple sealed containers remotely, at once, without needing physical manipulation and manual weighing. First, we propose a novel speckle-based vibration sensing system for simultaneously capturing scene vibrations on a 2D grid of points. We use our system to efficiently and remotely capture a dataset of vibration responses for a variety of everyday liquid containers. Then, we develop a transformer-based approach for analyzing the captured vibrations and classifying the container type and its hidden liquid level at the time of measurement. Our architecture is invariant to the vibration source, yielding correct liquid level estimates for controlled and ambient scene sound sources. Moreover, our model generalizes to unseen container instances within known classes (e.g., training on five Coke cans of a six-pack, testing on a sixth) and fluid levels. We demonstrate our method by recovering liquid levels from various everyday containers. 

**Abstract (ZH)**: 计算机视觉旨在推断物体和事件的广泛信息。然而，基于传统成像的视觉系统仅能从场景物体的可见表面提取信息。例如，一个视觉系统可以检测并识别场景中的可乐罐，但无法确定罐子是否装满或为空。本文旨在扩展计算机视觉的范围，通过感知不透明容器表面微小的振动来推断其内部液位，从而实现新的任务。我们的方法提供了首款无需物理操作和手动称重即可远程、同时检查多个密封容器液位的方案。首先，我们提出了一种新的基于斑点的振动感知系统，用于在二维点网格上同时捕捉场景振动。我们使用该系统高效且远程地收集了各种日常液体容器的振动响应数据集。然后，我们开发了一种基于变压器的方法，用于分析捕捉到的振动信号，并在测量时对容器类型及其隐藏液位进行分类。我们的架构对振动源具有不变性，适用于控制场景和环境场景声源。此外，我们的模型可以泛化到已知类别的未见过的容器实例（例如，在六罐装中进行五罐训练，在第六罐上进行测试）和液位。我们通过从各种日常容器中恢复液位来展示该方法。 

---
# Industry Insights from Comparing Deep Learning and GBDT Models for E-Commerce Learning-to-Rank 

**Title (ZH)**: 电子商务学习排名中深度学习与GBDT模型比较的产业洞察 

**Authors**: Yunus Lutz, Timo Wilm, Philipp Duwe  

**Link**: [PDF](https://arxiv.org/pdf/2507.20753)  

**Abstract**: In e-commerce recommender and search systems, tree-based models, such as LambdaMART, have set a strong baseline for Learning-to-Rank (LTR) tasks. Despite their effectiveness and widespread adoption in industry, the debate continues whether deep neural networks (DNNs) can outperform traditional tree-based models in this domain. To contribute to this discussion, we systematically benchmark DNNs against our production-grade LambdaMART model. We evaluate multiple DNN architectures and loss functions on a proprietary dataset from OTTO and validate our findings through an 8-week online A/B test. The results show that a simple DNN architecture outperforms a strong tree-based baseline in terms of total clicks and revenue, while achieving parity in total units sold. 

**Abstract (ZH)**: 在电子商务推荐和搜索系统中，基于树的模型如LambdaMART为学习排序（LTR）任务设定了强有力的基准。尽管这些模型在工业界表现出色并得到广泛采用，关于深度神经网络（DNNs）能否在这一领域超越传统基于树的模型的争论仍然存在。为了推进这一讨论，我们系统地将多种DNN架构与我们生产的LambdaMART模型进行了对比基准测试。我们在OTTO的专属数据集上评估了多种DNN架构和损失函数，并通过8周的在线A/B测试验证了我们的发现。结果表明，一个简单的DNN架构在总点击量和收入方面优于强大的基于树的基准模型，同时在总销售单位上达到一致。 

---
# AR-LIF: Adaptive reset leaky-integrate and fire neuron for spiking neural networks 

**Title (ZH)**: AR-LIF: 自适应复位泄漏积分和放电神经元用于突触神经网络 

**Authors**: Zeyu Huang, Wei Meng, Quan Liu, Kun Chen, Li Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.20746)  

**Abstract**: Spiking neural networks possess the advantage of low energy consumption due to their event-driven nature. Compared with binary spike outputs, their inherent floating-point dynamics are more worthy of attention. The threshold level and re- set mode of neurons play a crucial role in determining the number and timing of spikes. The existing hard reset method causes information loss, while the improved soft reset method adopts a uniform treatment for neurons. In response to this, this paper designs an adaptive reset neuron, establishing the correlation between input, output and reset, and integrating a simple yet effective threshold adjustment strategy. It achieves excellent performance on various datasets while maintaining the advantage of low energy consumption. 

**Abstract (ZH)**: 基于事件驱动的神经网络由于其低能消耗特性而具备优势。与二元脉冲输出相比，其固有的浮点动态更为值得关注。神经元的阈值水平和重置模式决定了脉冲的数量和时间。现有的硬重置方法会导致信息丢失，而改进的软重置方法对神经元采用统一处理。针对此问题，本文设计了一种自适应重置神经元，建立了输入、输出与重置之间的关联，并集成了简单有效的阈值调整策略。在多种数据集上实现了优异性能，同时保持了低能消耗的优势。 

---
# Regularizing Subspace Redundancy of Low-Rank Adaptation 

**Title (ZH)**: 正则化低秩适应中的子空间冗余 

**Authors**: Yue Zhu, Haiwen Diao, Shang Gao, Jiazuo Yu, Jiawen Zhu, Yunzhi Zhuge, Shuai Hao, Xu Jia, Lu Zhang, Ying Zhang, Huchuan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2507.20745)  

**Abstract**: Low-Rank Adaptation (LoRA) and its variants have delivered strong capability in Parameter-Efficient Transfer Learning (PETL) by minimizing trainable parameters and benefiting from reparameterization. However, their projection matrices remain unrestricted during training, causing high representation redundancy and diminishing the effectiveness of feature adaptation in the resulting subspaces. While existing methods mitigate this by manually adjusting the rank or implicitly applying channel-wise masks, they lack flexibility and generalize poorly across various datasets and architectures. Hence, we propose ReSoRA, a method that explicitly models redundancy between mapping subspaces and adaptively Regularizes Subspace redundancy of Low-Rank Adaptation. Specifically, it theoretically decomposes the low-rank submatrices into multiple equivalent subspaces and systematically applies de-redundancy constraints to the feature distributions across different projections. Extensive experiments validate that our proposed method consistently facilitates existing state-of-the-art PETL methods across various backbones and datasets in vision-language retrieval and standard visual classification benchmarks. Besides, as a training supervision, ReSoRA can be seamlessly integrated into existing approaches in a plug-and-play manner, with no additional inference costs. Code is publicly available at: this https URL. 

**Abstract (ZH)**: ReSoRA: 显式建模低秩适应子空间间的冗余并自适应正则化低秩适应子空间冗余 

---
# Multi-Masked Querying Network for Robust Emotion Recognition from Incomplete Multi-Modal Physiological Signals 

**Title (ZH)**: 基于不完备多模态生理信号的鲁棒情绪识别的多掩码查询网络 

**Authors**: Geng-Xin Xu, Xiang Zuo, Ye Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.20737)  

**Abstract**: Emotion recognition from physiological data is crucial for mental health assessment, yet it faces two significant challenges: incomplete multi-modal signals and interference from body movements and artifacts. This paper presents a novel Multi-Masked Querying Network (MMQ-Net) to address these issues by integrating multiple querying mechanisms into a unified framework. Specifically, it uses modality queries to reconstruct missing data from incomplete signals, category queries to focus on emotional state features, and interference queries to separate relevant information from noise. Extensive experiment results demonstrate the superior emotion recognition performance of MMQ-Net compared to existing approaches, particularly under high levels of data incompleteness. 

**Abstract (ZH)**: 从生理数据中识别情绪对于心理健康评估至关重要，但面临着两大挑战：Incomplete Multi-modal Signals和身体运动及噪声干扰。本文提出了一种新型的多掩码查询网络（MMQ-Net）以解决这些问题，通过将多种查询机制整合到统一框架中。具体而言，它使用模态查询来从不完整信号中重构缺失数据，类别查询来聚焦于情绪状态特征，以及干扰查询来分离相关信息与噪声。广泛实验结果表明，MMQ-Net在数据不完整性较高时的情绪识别性能显著优于现有方法。 

---
# Prostate Cancer Classification Using Multimodal Feature Fusion and Explainable AI 

**Title (ZH)**: 前列腺癌分类：基于多模态特征融合与可解释人工智能 

**Authors**: Asma Sadia Khan, Fariba Tasnia Khan, Tanjim Mahmud, Salman Karim Khan, Rishita Chakma, Nahed Sharmen, Mohammad Shahadat Hossain, Karl Andersson  

**Link**: [PDF](https://arxiv.org/pdf/2507.20714)  

**Abstract**: Prostate cancer, the second most prevalent male malignancy, requires advanced diagnostic tools. We propose an explainable AI system combining BERT (for textual clinical notes) and Random Forest (for numerical lab data) through a novel multimodal fusion strategy, achieving superior classification performance on PLCO-NIH dataset (98% accuracy, 99% AUC). While multimodal fusion is established, our work demonstrates that a simple yet interpretable BERT+RF pipeline delivers clinically significant improvements - particularly for intermediate cancer stages (Class 2/3 recall: 0.900 combined vs 0.824 numerical/0.725 textual). SHAP analysis provides transparent feature importance rankings, while ablation studies prove textual features' complementary value. This accessible approach offers hospitals a balance of high performance (F1=89%), computational efficiency, and clinical interpretability - addressing critical needs in prostate cancer diagnostics. 

**Abstract (ZH)**: 前列腺癌是男性第二大常见的恶性肿瘤，需要先进的诊断工具。我们提出了一种结合BERT（处理文本临床笔记）和随机森林（处理数值实验室数据）的可解释AI系统，通过一种新颖的多模态融合策略，在PLCO-NIH数据集上实现了卓越的分类性能（准确率98%，AUC 99%）。尽管多模态融合已经确立，我们的工作证明了一个简单且可解释的BERT+RF流水线能够提供临床显著的改进，尤其是在中等癌症阶段（Class 2/3召回率：0.900结合型 vs 0.824 数值型/0.725 文本型）。SHAP分析提供了透明的特征重要性排名，而消融研究证明了文本特征的补充价值。这一易于实现的方法为医院提供了高性能（F1=89%）、计算效率和临床可解释性的平衡，以解决前列腺癌诊断中的关键需求。 

---
# Text2VLM: Adapting Text-Only Datasets to Evaluate Alignment Training in Visual Language Models 

**Title (ZH)**: Text2VLM: 将仅文本数据集适应于评估视觉语言模型的对齐训练 

**Authors**: Gabriel Downer, Sean Craven, Damian Ruck, Jake Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2507.20704)  

**Abstract**: The increasing integration of Visual Language Models (VLMs) into AI systems necessitates robust model alignment, especially when handling multimodal content that combines text and images. Existing evaluation datasets heavily lean towards text-only prompts, leaving visual vulnerabilities under evaluated. To address this gap, we propose \textbf{Text2VLM}, a novel multi-stage pipeline that adapts text-only datasets into multimodal formats, specifically designed to evaluate the resilience of VLMs against typographic prompt injection attacks. The Text2VLM pipeline identifies harmful content in the original text and converts it into a typographic image, creating a multimodal prompt for VLMs. Also, our evaluation of open-source VLMs highlights their increased susceptibility to prompt injection when visual inputs are introduced, revealing critical weaknesses in the current models' alignment. This is in addition to a significant performance gap compared to closed-source frontier models. We validate Text2VLM through human evaluations, ensuring the alignment of extracted salient concepts; text summarization and output classification align with human expectations. Text2VLM provides a scalable tool for comprehensive safety assessment, contributing to the development of more robust safety mechanisms for VLMs. By enhancing the evaluation of multimodal vulnerabilities, Text2VLM plays a role in advancing the safe deployment of VLMs in diverse, real-world applications. 

**Abstract (ZH)**: Text2VLM：多阶段管道将文本数据适配为多模态格式以评估视觉语言模型的视觉漏洞 

---
# A Multimodal Architecture for Endpoint Position Prediction in Team-based Multiplayer Games 

**Title (ZH)**: 基于团队多人游戏的端点位置预测多模态架构 

**Authors**: Jonas Peche, Aliaksei Tsishurou, Alexander Zap, Guenter Wallner  

**Link**: [PDF](https://arxiv.org/pdf/2507.20670)  

**Abstract**: Understanding and predicting player movement in multiplayer games is crucial for achieving use cases such as player-mimicking bot navigation, preemptive bot control, strategy recommendation, and real-time player behavior analytics. However, the complex environments allow for a high degree of navigational freedom, and the interactions and team-play between players require models that make effective use of the available heterogeneous input data. This paper presents a multimodal architecture for predicting future player locations on a dynamic time horizon, using a U-Net-based approach for calculating endpoint location probability heatmaps, conditioned using a multimodal feature encoder. The application of a multi-head attention mechanism for different groups of features allows for communication between agents. In doing so, the architecture makes efficient use of the multimodal game state including image inputs, numerical and categorical features, as well as dynamic game data. Consequently, the presented technique lays the foundation for various downstream tasks that rely on future player positions such as the creation of player-predictive bot behavior or player anomaly detection. 

**Abstract (ZH)**: 理解并预测多人游戏中玩家的运动模式对于实现玩家模拟机器人导航、预判性机器人控制、策略推荐以及实时玩家行为分析等用例至关重要。然而，复杂的环境提供了高度的导航自由度，玩家之间的互动和团队协作需要能够有效利用异质性输入数据的模型。本文提出了一种多模态架构，用于在动态时间范围内预测玩家的未来位置，采用基于U-Net的方法计算端点位置概率热图，并使用多模态特征编码进行条件控制。通过为不同的特征组应用多头注意力机制，使代理之间能够进行通信。因此，该架构能够高效利用包括图像输入、数值和分类特征以及动态游戏数据在内的多模态游戏状态。最终，所提出的技术为基础下游任务奠定了基础，这些任务依赖于未来玩家位置，如玩家预测性机器人行为的创建或玩家异常检测。 

---
# MIMII-Agent: Leveraging LLMs with Function Calling for Relative Evaluation of Anomalous Sound Detection 

**Title (ZH)**: MIMII-Agent：利用函数调用的LLM技术在异常声音检测的相对评估中应用 

**Authors**: Harsh Purohit, Tomoya Nishida, Kota Dohi, Takashi Endo, Yohei Kawaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2507.20666)  

**Abstract**: This paper proposes a method for generating machine-type-specific anomalies to evaluate the relative performance of unsupervised anomalous sound detection (UASD) systems across different machine types, even in the absence of real anomaly sound data. Conventional keyword-based data augmentation methods often produce unrealistic sounds due to their reliance on manually defined labels, limiting scalability as machine types and anomaly patterns diversify. Advanced audio generative models, such as MIMII-Gen, show promise but typically depend on anomalous training data, making them less effective when diverse anomalous examples are unavailable. To address these limitations, we propose a novel synthesis approach leveraging large language models (LLMs) to interpret textual descriptions of faults and automatically select audio transformation functions, converting normal machine sounds into diverse and plausible anomalous sounds. We validate this approach by evaluating a UASD system trained only on normal sounds from five machine types, using both real and synthetic anomaly data. Experimental results reveal consistent trends in relative detection difficulty across machine types between synthetic and real anomalies. This finding supports our hypothesis and highlights the effectiveness of the proposed LLM-based synthesis approach for relative evaluation of UASD systems. 

**Abstract (ZH)**: 本文提出了一种生成机器类型特定异常的方法，以评估不同机器类型下无监督异常声检测系统（UASD）的相对性能，即使在缺乏实际异常声数据的情况下也是如此。传统的基于关键词的数据扩增方法通常依赖于人工定义的标签，生成不现实的声音，限制了在机器类型和异常模式多样化时的可扩展性。先进的音频生成模型，如MIMII-Gen，表现出潜力，但通常依赖于异常训练数据，当多样化的异常示例不可用时效果较差。为解决这些限制，我们提出了一种新的合成方法，利用大规模语言模型（LLMs）解释故障的文本描述并自动选择音频变换函数，将正常机器声音转换为多样且合乎实际的异常声音。我们通过使用五种机器类型仅正常声音训练的UASD系统，并使用真实和合成的异常数据进行评估，验证了该方法。实验结果表明，合成异常和真实异常在不同机器类型下的相对检测难度存在一致趋势。这一发现支持了我们的假设，并强调了基于LLMs合成方法在相对评估UASD系统方面的有效性。 

---
# Hot-Swap MarkBoard: An Efficient Black-box Watermarking Approach for Large-scale Model Distribution 

**Title (ZH)**: 热插拔标记板：一种高效的大规模模型分发黑盒水标记方法 

**Authors**: Zhicheng Zhang, Peizhuo Lv, Mengke Wan, Jiang Fang, Diandian Guo, Yezeng Chen, Yinlong Liu, Wei Ma, Jiyan Sun, Liru Geng  

**Link**: [PDF](https://arxiv.org/pdf/2507.20650)  

**Abstract**: Recently, Deep Learning (DL) models have been increasingly deployed on end-user devices as On-Device AI, offering improved efficiency and privacy. However, this deployment trend poses more serious Intellectual Property (IP) risks, as models are distributed on numerous local devices, making them vulnerable to theft and redistribution. Most existing ownership protection solutions (e.g., backdoor-based watermarking) are designed for cloud-based AI-as-a-Service (AIaaS) and are not directly applicable to large-scale distribution scenarios, where each user-specific model instance must carry a unique watermark. These methods typically embed a fixed watermark, and modifying the embedded watermark requires retraining the model. To address these challenges, we propose Hot-Swap MarkBoard, an efficient watermarking method. It encodes user-specific $n$-bit binary signatures by independently embedding multiple watermarks into a multi-branch Low-Rank Adaptation (LoRA) module, enabling efficient watermark customization without retraining through branch swapping. A parameter obfuscation mechanism further entangles the watermark weights with those of the base model, preventing removal without degrading model performance. The method supports black-box verification and is compatible with various model architectures and DL tasks, including classification, image generation, and text generation. Extensive experiments across three types of tasks and six backbone models demonstrate our method's superior efficiency and adaptability compared to existing approaches, achieving 100\% verification accuracy. 

**Abstract (ZH)**: 近期，深度学习(DL)模型被越来越广泛地部署在终端用户设备上，作为On-Device AI，提供了更好的效率和隐私保护。然而，这种部署趋势也带来了更严重的知识产权(IP)风险，因为模型被分散在众多本地设备上，使其容易被盗用和再分发。现有的大部分所有权保护解决方案（例如，基于后门的水印技术）设计用于基于云的AI服务(AIaaS)，并不直接适用于大规模分发场景，在这种场景中，每个用户特定的模型实例必须携带一个唯一的水印。这些方法通常嵌入固定水印，修改嵌入的水印需要重新训练模型。为了解决这些问题，我们提出了一种高效的水印方法——Hot-Swap MarkBoard。该方法通过独立地将多个水印嵌入到多分支低秩适应(LoRA)模块中，以支路切换的方式实现高效的水印个性化定制，而无需重新训练。参数混淆机制进一步将水印权重与基模型权重交织，防止水印的移除而不降级模型性能。该方法支持黑盒验证，并兼容各种模型架构和深度学习任务，包括分类、图像生成和文本生成。在三种类型任务和六种骨干模型上的广泛实验表明，与现有方法相比，该方法在效率和适应性方面表现出 superior 性能，验证准确率达到100%。 

---
# Ontology-Enhanced Knowledge Graph Completion using Large Language Models 

**Title (ZH)**: 使用大型语言模型增强的知识图谱完成方法 

**Authors**: Wenbin Guo, Xin Wang, Jiaoyan Chen, Zhao Li, Zirui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.20643)  

**Abstract**: Large Language Models (LLMs) have been extensively adopted in Knowledge Graph Completion (KGC), showcasing significant research advancements. However, as black-box models driven by deep neural architectures, current LLM-based KGC methods rely on implicit knowledge representation with parallel propagation of erroneous knowledge, thereby hindering their ability to produce conclusive and decisive reasoning outcomes. We aim to integrate neural-perceptual structural information with ontological knowledge, leveraging the powerful capabilities of LLMs to achieve a deeper understanding of the intrinsic logic of the knowledge. We propose an ontology enhanced KGC method using LLMs -- OL-KGC. It first leverages neural perceptual mechanisms to effectively embed structural information into the textual space, and then uses an automated extraction algorithm to retrieve ontological knowledge from the knowledge graphs (KGs) that needs to be completed, which is further transformed into a textual format comprehensible to LLMs for providing logic guidance. We conducted extensive experiments on three widely-used benchmarks -- FB15K-237, UMLS and WN18RR. The experimental results demonstrate that OL-KGC significantly outperforms existing mainstream KGC methods across multiple evaluation metrics, achieving state-of-the-art performance. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在知识图谱补全（KGC）中的应用取得了显著的研究进展，然而，作为基于深度神经架构的黑盒模型，当前基于LLM的KGC方法依赖于并行传递的隐式知识表示，这限制了它们产生结论性和决定性的推理结果的能力。我们旨在结合神经感知结构信息与本体知识，利用LLMs的强大能力，实现对知识内在逻辑的更深入理解。我们提出了一种使用LLMs的本体增强知识图谱补全方法——OL-KGC。该方法首先利用神经感知机制有效将结构信息嵌入到文本空间中，然后使用自动化提取算法从需要补全的知识图谱（KGs）中检索本体知识，并进一步转换为LLMs可理解的文本格式，以提供逻辑指导。我们在三个广泛使用的基准数据集——FB15K-237、UMLS和WN18RR上进行了广泛的实验。实验结果表明，OL-KGC在多个评估指标上显著优于现有主流的KGC方法，实现了最佳性能。 

---
# TransPrune: Token Transition Pruning for Efficient Large Vision-Language Model 

**Title (ZH)**: TransPrune: Token过渡剪枝以实现高效的大型视觉-语言模型 

**Authors**: Ao Li, Yuxiang Duan, Jinghui Zhang, Congbo Ma, Yutong Xie, Gustavo Carneiro, Mohammad Yaqub, Hu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20630)  

**Abstract**: Large Vision-Language Models (LVLMs) have advanced multimodal learning but face high computational costs due to the large number of visual tokens, motivating token pruning to improve inference efficiency. The key challenge lies in identifying which tokens are truly important. Most existing approaches rely on attention-based criteria to estimate token importance. However, they inherently suffer from certain limitations, such as positional bias. In this work, we explore a new perspective on token importance based on token transitions in LVLMs. We observe that the transition of token representations provides a meaningful signal of semantic information. Based on this insight, we propose TransPrune, a training-free and efficient token pruning method. Specifically, TransPrune progressively prunes tokens by assessing their importance through a combination of Token Transition Variation (TTV)-which measures changes in both the magnitude and direction of token representations-and Instruction-Guided Attention (IGA), which measures how strongly the instruction attends to image tokens via attention. Extensive experiments demonstrate that TransPrune achieves comparable multimodal performance to original LVLMs, such as LLaVA-v1.5 and LLaVA-Next, across eight benchmarks, while reducing inference TFLOPs by more than half. Moreover, TTV alone can serve as an effective criterion without relying on attention, achieving performance comparable to attention-based methods. The code will be made publicly available upon acceptance of the paper at this https URL. 

**Abstract (ZH)**: 基于令牌转换的Large视觉-语言模型的令牌剪枝 

---
# Controllable Video-to-Music Generation with Multiple Time-Varying Conditions 

**Title (ZH)**: 具有多种时间变异条件的可控视频到音乐生成 

**Authors**: Junxian Wu, Weitao You, Heda Zuo, Dengming Zhang, Pei Chen, Lingyun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.20627)  

**Abstract**: Music enhances video narratives and emotions, driving demand for automatic video-to-music (V2M) generation. However, existing V2M methods relying solely on visual features or supplementary textual inputs generate music in a black-box manner, often failing to meet user expectations. To address this challenge, we propose a novel multi-condition guided V2M generation framework that incorporates multiple time-varying conditions for enhanced control over music generation. Our method uses a two-stage training strategy that enables learning of V2M fundamentals and audiovisual temporal synchronization while meeting users' needs for multi-condition control. In the first stage, we introduce a fine-grained feature selection module and a progressive temporal alignment attention mechanism to ensure flexible feature alignment. For the second stage, we develop a dynamic conditional fusion module and a control-guided decoder module to integrate multiple conditions and accurately guide the music composition process. Extensive experiments demonstrate that our method outperforms existing V2M pipelines in both subjective and objective evaluations, significantly enhancing control and alignment with user expectations. 

**Abstract (ZH)**: 音乐增强视频叙事和情感，推动了自动视频到音乐（V2M）生成的需求。然而，现有的仅依赖视觉特征或补充文本输入的V2M方法以黑箱方式生成音乐，往往无法满足用户期望。为此，我们提出了一种新的多条件引导V2M生成框架，结合了多种时间变化条件以增强音乐生成的控制。我们的方法采用两阶段训练策略，既能学习V2M的基本原理和音视频时间同步，又能满足用户对多条件控制的需求。在第一阶段，我们引入了精细特征选择模块和渐进时间对齐注意力机制，以确保灵活的特征对齐。在第二阶段，我们开发了动态条件融合模块和控制引导解码器模块，以整合多种条件并准确引导音乐创作过程。 extensive实验表明，我们的方法在主观和客观评价中均优于现有V2M管道，显著增强了控制能力和与用户期望的契合度。 

---
# Lightweight Remote Sensing Scene Classification on Edge Devices via Knowledge Distillation and Early-exit 

**Title (ZH)**: 基于知识蒸馏和早退的轻量级边缘设备遥感场景分类 

**Authors**: Yang Zhao, Shusheng Li, Xueshang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2507.20623)  

**Abstract**: As the development of lightweight deep learning algorithms, various deep neural network (DNN) models have been proposed for the remote sensing scene classification (RSSC) application. However, it is still challenging for these RSSC models to achieve optimal performance among model accuracy, inference latency, and energy consumption on resource-constrained edge devices. In this paper, we propose a lightweight RSSC framework, which includes a distilled global filter network (GFNet) model and an early-exit mechanism designed for edge devices to achieve state-of-the-art performance. Specifically, we first apply frequency domain distillation on the GFNet model to reduce model size. Then we design a dynamic early-exit model tailored for DNN models on edge devices to further improve model inference efficiency. We evaluate our E3C model on three edge devices across four datasets. Extensive experimental results show that it achieves an average of 1.3x speedup on model inference and over 40% improvement on energy efficiency, while maintaining high classification accuracy. 

**Abstract (ZH)**: 轻量化遥感场景分类框架：基于频率域蒸馏的GFNet模型与早退机制 

---
# Beyond Interactions: Node-Level Graph Generation for Knowledge-Free Augmentation in Recommender Systems 

**Title (ZH)**: 超越交互：知识无介接入点级图生成在推荐系统中的无知识增强 

**Authors**: Zhaoyan Wang, Hyunjun Ahn, In-Young Ko  

**Link**: [PDF](https://arxiv.org/pdf/2507.20578)  

**Abstract**: Recent advances in recommender systems rely on external resources such as knowledge graphs or large language models to enhance recommendations, which limit applicability in real-world settings due to data dependency and computational overhead. Although knowledge-free models are able to bolster recommendations by direct edge operations as well, the absence of augmentation primitives drives them to fall short in bridging semantic and structural gaps as high-quality paradigm substitutes. Unlike existing diffusion-based works that remodel user-item interactions, this work proposes NodeDiffRec, a pioneering knowledge-free augmentation framework that enables fine-grained node-level graph generation for recommendations and expands the scope of restricted augmentation primitives via diffusion. By synthesizing pseudo-items and corresponding interactions that align with the underlying distribution for injection, and further refining user preferences through a denoising preference modeling process, NodeDiffRec dramatically enhances both semantic diversity and structural connectivity without external knowledge. Extensive experiments across diverse datasets and recommendation algorithms demonstrate the superiority of NodeDiffRec, achieving State-of-the-Art (SOTA) performance, with maximum average performance improvement 98.6% in Recall@5 and 84.0% in NDCG@5 over selected baselines. 

**Abstract (ZH)**: 知识图谱和大型语言模型之外：NodeDiffRec——一种无知识增强的节点级别图生成推荐框架 

---
# Implicit Spatiotemporal Bandwidth Enhancement Filter by Sine-activated Deep Learning Model for Fast 3D Photoacoustic Tomography 

**Title (ZH)**: 基于正弦激活深度学习模型的快速三维光声断层成像隐式空时带宽增强滤波器 

**Authors**: I Gede Eka Sulistyawan, Takuro Ishii, Riku Suzuki, Yoshifumi Saijo  

**Link**: [PDF](https://arxiv.org/pdf/2507.20575)  

**Abstract**: 3D photoacoustic tomography (3D-PAT) using high-frequency hemispherical transducers offers near-omnidirectional reception and enhanced sensitivity to the finer structural details encoded in the high-frequency components of the broadband photoacoustic (PA) signal. However, practical constraints such as limited number of channels with bandlimited sampling rate often result in sparse and bandlimited sensors that degrade image quality. To address this, we revisit the 2D deep learning (DL) approach applied directly to sensor-wise PA radio-frequency (PARF) data. Specifically, we introduce sine activation into the DL model to restore the broadband nature of PARF signals given the observed band-limited and high-frequency PARF data. Given the scarcity of 3D training data, we employ simplified training strategies by simulating random spherical absorbers. This combination of sine-activated model and randomized training is designed to emphasize bandwidth learning over dataset memorization. Our model was evaluated on a leaf skeleton phantom, a micro-CT-verified 3D spiral phantom and in-vivo human palm vasculature. The results showed that the proposed training mechanism on sine-activated model was well-generalized across the different tests by effectively increasing the sensor density and recovering the spatiotemporal bandwidth. Qualitatively, the sine-activated model uniquely enhanced high-frequency content that produces clearer vascular structure with fewer artefacts. Quantitatively, the sine-activated model exhibits full bandwidth at -12 dB spectrum and significantly higher contrast-to-noise ratio with minimal loss of structural similarity index. Lastly, we optimized our approach to enable fast enhanced 3D-PAT at 2 volumes-per-second for better practical imaging of a free-moving targets. 

**Abstract (ZH)**: 基于高频率半球形换能器的3D光声成像（3D-PAT）利用近全向接收和高频率成分的高灵敏度以编码精细结构细节。然而，有限的带限采样率通道数量等实际限制常常导致稀疏且带限的传感器，从而降低图像质量。为此，我们重新审视了直接应用于传感器级光声射频（PARF）数据的2D深度学习（DL）方法。具体而言，我们引入了正弦激活到DL模型中，以恢复PARF信号的宽带特性，给定观察到的带限和高频率PARF数据。鉴于3D训练数据的稀缺性，我们通过模拟随机球形吸收体来采用简化的训练策略。这种正弦激活模型与随机化训练的结合旨在强调宽带学习而非数据集记忆。我们的模型在榆叶骨架仿真器、微CT验证的3D螺旋仿真器以及活体人类手掌血管中进行了评估。结果表明，提出的基于正弦激活模型的训练机制在不同测试中表现出良好的泛化能力，通过有效增加传感器密度并恢复时空带宽。定性上，正弦激活模型独特地增强了高频成分，从而生成更清晰的血管结构并减少了伪影。定量上，正弦激活模型在-12 dB频谱中表现出完整带宽，并具有显著更高的信噪比，同时保持结构相似性指数的最小损失。最后，我们优化了我们的方法以实现每秒2个体素的快速增强3D-PAT，以便更好地进行自由移动目标的实用成像。 

---
# DAG-AFL:Directed Acyclic Graph-based Asynchronous Federated Learning 

**Title (ZH)**: 基于有向无环图的异步联邦学习 

**Authors**: Shuaipeng Zhang, Lanju Kong, Yixin Zhang, Wei He, Yongqing Zheng, Han Yu, Lizhen Cui  

**Link**: [PDF](https://arxiv.org/pdf/2507.20571)  

**Abstract**: Due to the distributed nature of federated learning (FL), the vulnerability of the global model and the need for coordination among many client devices pose significant challenges. As a promising decentralized, scalable and secure solution, blockchain-based FL methods have attracted widespread attention in recent years. However, traditional consensus mechanisms designed for Proof of Work (PoW) similar to blockchain incur substantial resource consumption and compromise the efficiency of FL, particularly when participating devices are wireless and resource-limited. To address asynchronous client participation and data heterogeneity in FL, while limiting the additional resource overhead introduced by blockchain, we propose the Directed Acyclic Graph-based Asynchronous Federated Learning (DAG-AFL) framework. We develop a tip selection algorithm that considers temporal freshness, node reachability and model accuracy, with a DAG-based trusted verification strategy. Extensive experiments on 3 benchmarking datasets against eight state-of-the-art approaches demonstrate that DAG-AFL significantly improves training efficiency and model accuracy by 22.7% and 6.5% on average, respectively. 

**Abstract (ZH)**: 基于有向无环图的异步联邦学习框架（DAG-AFL） 

---
# Learning Phonetic Context-Dependent Viseme for Enhancing Speech-Driven 3D Facial Animation 

**Title (ZH)**: 基于音素上下文依赖的语音驱动3D面部动画增强中的发音音视素学习 

**Authors**: Hyung Kyu Kim, Hak Gu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.20568)  

**Abstract**: Speech-driven 3D facial animation aims to generate realistic facial movements synchronized with audio. Traditional methods primarily minimize reconstruction loss by aligning each frame with ground-truth. However, this frame-wise approach often fails to capture the continuity of facial motion, leading to jittery and unnatural outputs due to coarticulation. To address this, we propose a novel phonetic context-aware loss, which explicitly models the influence of phonetic context on viseme transitions. By incorporating a viseme coarticulation weight, we assign adaptive importance to facial movements based on their dynamic changes over time, ensuring smoother and perceptually consistent animations. Extensive experiments demonstrate that replacing the conventional reconstruction loss with ours improves both quantitative metrics and visual quality. It highlights the importance of explicitly modeling phonetic context-dependent visemes in synthesizing natural speech-driven 3D facial animation. Project page: this https URL 

**Abstract (ZH)**: 基于语音的3D面部动画旨在生成与音频同步的逼真面部动作。传统的方法主要通过将每一帧与ground-truth对齐来最小化重建损失，但这种方法往往无法捕捉面部动作的连贯性，导致输出不流畅且不自然，这是因为协同发音的影响。为了解决这一问题，我们提出了一种新的音系上下文感知损失，该损失明确地建模了音系上下文对音素转换的影响。通过引入协同发音权重，我们根据面部动作随时间动态变化的情况为面部动作分配适应性的权重，确保动画更加平滑且具感知一致性。大量实验表明，用我们的损失替换传统的重建损失可以提高定量指标和视觉质量。这突显了在合成自然的语音驱动3D面部动画时明确建模音系上下文依赖的音素的重要性。项目页面: 这里.getJSONObject("result").getString("url") 

---
# MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization 

**Title (ZH)**: MemoryTalker: 个人化音源驱动的3D面部动画通过音频引导的风格化 

**Authors**: Hyung Kyu Kim, Sangmin Lee, Hak Gu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.20562)  

**Abstract**: Speech-driven 3D facial animation aims to synthesize realistic facial motion sequences from given audio, matching the speaker's speaking style. However, previous works often require priors such as class labels of a speaker or additional 3D facial meshes at inference, which makes them fail to reflect the speaking style and limits their practical use. To address these issues, we propose MemoryTalker which enables realistic and accurate 3D facial motion synthesis by reflecting speaking style only with audio input to maximize usability in applications. Our framework consists of two training stages: 1-stage is storing and retrieving general motion (i.e., Memorizing), and 2-stage is to perform the personalized facial motion synthesis (i.e., Animating) with the motion memory stylized by the audio-driven speaking style feature. In this second stage, our model learns about which facial motion types should be emphasized for a particular piece of audio. As a result, our MemoryTalker can generate a reliable personalized facial animation without additional prior information. With quantitative and qualitative evaluations, as well as user study, we show the effectiveness of our model and its performance enhancement for personalized facial animation over state-of-the-art methods. 

**Abstract (ZH)**: 基于语音的3D面部动画旨在从给定的音频中合成逼真的面部运动序列，匹配说话人的说话风格。然而，先前的工作往往需要诸如说话者类别标签或额外的3D面部网格等先验信息，这使得它们难以反映说话风格并限制了其实用性。为了解决这些问题，我们提出了MemoryTalker，仅通过语音输入反映说话风格来实现逼真和准确的3D面部运动合成，以最大化其在应用中的适用性。我们的框架包括两个训练阶段：第一阶段是存储和检索通用运动（即记忆），第二阶段是通过由语音驱动的说话风格特征进行个性化面部运动合成（即动画）。在第二阶段中，我们的模型学会了特定音频片段中应强调哪些面部运动类型。因此，我们的MemoryTalker可以在无需额外先验信息的情况下生成可靠的个性化面部动画。通过定量和定性评估以及用户研究，我们展示了我们模型的有效性及其在个性化面部动画方面的性能提升，超过了当前最先进的方法。 

---
# Enhancing Hallucination Detection via Future Context 

**Title (ZH)**: 通过未来语境增强幻觉检测 

**Authors**: Joosung Lee, Cheonbok Park, Hwiyeol Jo, Jeonghoon Kim, Joonsuk Park, Kang Min Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2507.20546)  

**Abstract**: Large Language Models (LLMs) are widely used to generate plausible text on online platforms, without revealing the generation process. As users increasingly encounter such black-box outputs, detecting hallucinations has become a critical challenge. To address this challenge, we focus on developing a hallucination detection framework for black-box generators. Motivated by the observation that hallucinations, once introduced, tend to persist, we sample future contexts. The sampled future contexts provide valuable clues for hallucination detection and can be effectively integrated with various sampling-based methods. We extensively demonstrate performance improvements across multiple methods using our proposed sampling approach. 

**Abstract (ZH)**: 大型语言模型（LLMs）广泛用于在线平台上生成可信文本，而不揭示生成过程。随着用户越来越多地遇到这样的黑盒输出，检测幻觉已经成为一个关键挑战。为应对这一挑战，我们专注于开发针对黑盒生成器的幻觉检测框架。受观察到的幻觉一旦出现往往会持续存在这一现象的启发，我们采样了未来上下文。采样的未来上下文为幻觉检测提供了有价值的线索，并且可以有效与各种采样方法集成。我们广泛展示了使用我们提出的方法在多种方法中实现性能提升。 

---
# T2I-Copilot: A Training-Free Multi-Agent Text-to-Image System for Enhanced Prompt Interpretation and Interactive Generation 

**Title (ZH)**: T2I-Copilot: 无需训练的多代理文本到图像系统，用于增强提示解释和交互生成 

**Authors**: Chieh-Yun Chen, Min Shi, Gong Zhang, Humphrey Shi  

**Link**: [PDF](https://arxiv.org/pdf/2507.20536)  

**Abstract**: Text-to-Image (T2I) generative models have revolutionized content creation but remain highly sensitive to prompt phrasing, often requiring users to repeatedly refine prompts multiple times without clear feedback. While techniques such as automatic prompt engineering, controlled text embeddings, denoising, and multi-turn generation mitigate these issues, they offer limited controllability, or often necessitate additional training, restricting the generalization abilities. Thus, we introduce T2I-Copilot, a training-free multi-agent system that leverages collaboration between (Multimodal) Large Language Models to automate prompt phrasing, model selection, and iterative refinement. This approach significantly simplifies prompt engineering while enhancing generation quality and text-image alignment compared to direct generation. Specifically, T2I-Copilot consists of three agents: (1) Input Interpreter, which parses the input prompt, resolves ambiguities, and generates a standardized report; (2) Generation Engine, which selects the appropriate model from different types of T2I models and organizes visual and textual prompts to initiate generation; and (3) Quality Evaluator, which assesses aesthetic quality and text-image alignment, providing scores and feedback for potential regeneration. T2I-Copilot can operate fully autonomously while also supporting human-in-the-loop intervention for fine-grained control. On GenAI-Bench, using open-source generation models, T2I-Copilot achieves a VQA score comparable to commercial models RecraftV3 and Imagen 3, surpasses FLUX1.1-pro by 6.17% at only 16.59% of its cost, and outperforms FLUX.1-dev and SD 3.5 Large by 9.11% and 6.36%. Code will be released at: this https URL. 

**Abstract (ZH)**: 基于多模态大语言模型的训练-free图文生成辅助系统T2I-Copilot 

---
# Kimi K2: Open Agentic Intelligence 

**Title (ZH)**: Kimi K2: 开放自主人工智能 

**Authors**: Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, Zhuofu Chen, Jialei Cui, Hao Ding, Mengnan Dong, Angang Du, Chenzhuang Du, Dikang Du, Yulun Du, Yu Fan, Yichen Feng, Kelin Fu, Bofei Gao, Hongcheng Gao, Peizhong Gao, Tong Gao, Xinran Gu, Longyu Guan, Haiqing Guo, Jianhang Guo, Hao Hu, Xiaoru Hao, Tianhong He, Weiran He, Wenyang He, Chao Hong, Yangyang Hu, Zhenxing Hu, Weixiao Huang, Zhiqi Huang, Zihao Huang, Tao Jiang, Zhejun Jiang, Xinyi Jin, Yongsheng Kang, Guokun Lai, Cheng Li, Fang Li, Haoyang Li, Ming Li, Wentao Li, Yanhao Li, Yiwei Li, Zhaowei Li, Zheming Li, Hongzhan Lin, Xiaohan Lin, Zongyu Lin, Chengyin Liu, Chenyu Liu, Hongzhang Liu, Jingyuan Liu, Junqi Liu, Liang Liu, Shaowei Liu, T.Y. Liu, Tianwei Liu, Weizhou Liu, Yangyang Liu, Yibo Liu, Yiping Liu, Yue Liu, Zhengying Liu, Enzhe Lu, Lijun Lu, Shengling Ma, Xinyu Ma, Yingwei Ma, Shaoguang Mao, Jie Mei, Xin Men, Yibo Miao, Siyuan Pan, Yebo Peng, Ruoyu Qin, Bowen Qu, Zeyu Shang, Lidong Shi, Shengyuan Shi, Feifan Song, Jianlin Su, Zhengyuan Su, Xinjie Sun, Flood Sung, Heyi Tang, Jiawen Tao, Qifeng Teng, Chensi Wang, Dinglu Wang, Feng Wang, Haiming Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20534)  

**Abstract**: We introduce Kimi K2, a Mixture-of-Experts (MoE) large language model with 32 billion activated parameters and 1 trillion total parameters. We propose the MuonClip optimizer, which improves upon Muon with a novel QK-clip technique to address training instability while enjoying the advanced token efficiency of Muon. Based on MuonClip, K2 was pre-trained on 15.5 trillion tokens with zero loss spike. During post-training, K2 undergoes a multi-stage post-training process, highlighted by a large-scale agentic data synthesis pipeline and a joint reinforcement learning (RL) stage, where the model improves its capabilities through interactions with real and synthetic environments.
Kimi K2 achieves state-of-the-art performance among open-source non-thinking models, with strengths in agentic capabilities. Notably, K2 obtains 66.1 on Tau2-Bench, 76.5 on ACEBench (En), 65.8 on SWE-Bench Verified, and 47.3 on SWE-Bench Multilingual -- surpassing most open and closed-sourced baselines in non-thinking settings. It also exhibits strong capabilities in coding, mathematics, and reasoning tasks, with a score of 53.7 on LiveCodeBench v6, 49.5 on AIME 2025, 75.1 on GPQA-Diamond, and 27.1 on OJBench, all without extended thinking. These results position Kimi K2 as one of the most capable open-source large language models to date, particularly in software engineering and agentic tasks. We release our base and post-trained model checkpoints to facilitate future research and applications of agentic intelligence. 

**Abstract (ZH)**: Kimi K2：一种具有320亿激活参数和1万亿总参数的Mixture-of-Experts大型语言模型及其优化方法与应用 

---
# Enhancing Spatial Reasoning through Visual and Textual Thinking 

**Title (ZH)**: 通过视觉与文本思维增强空间推理能力 

**Authors**: Xun Liang, Xin Guo, Zhongming Jin, Weihang Pan, Penghui Shang, Deng Cai, Binbin Lin, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2507.20529)  

**Abstract**: The spatial reasoning task aims to reason about the spatial relationships in 2D and 3D space, which is a fundamental capability for Visual Question Answering (VQA) and robotics. Although vision language models (VLMs) have developed rapidly in recent years, they are still struggling with the spatial reasoning task. In this paper, we introduce a method that can enhance Spatial reasoning through Visual and Textual thinking Simultaneously (SpatialVTS). In the spatial visual thinking phase, our model is trained to generate location-related specific tokens of essential targets automatically. Not only are the objects mentioned in the problem addressed, but also the potential objects related to the reasoning are considered. During the spatial textual thinking phase, Our model conducts long-term thinking based on visual cues and dialogues, gradually inferring the answers to spatial reasoning problems. To effectively support the model's training, we perform manual corrections to the existing spatial reasoning dataset, eliminating numerous incorrect labels resulting from automatic annotation, restructuring the data input format to enhance generalization ability, and developing thinking processes with logical reasoning details. Without introducing additional information (such as masks or depth), our model's overall average level in several spatial understanding tasks has significantly improved compared with other models. 

**Abstract (ZH)**: 空间推理任务旨在探究二维和三维空间中的空间关系，这是视觉问答（VQA）和机器人技术中的基本能力。尽管视觉语言模型（VLMs）近年来取得了 rapid 的发展，但在空间推理任务上仍然面临挑战。本文介绍了一种通过同时进行视觉和文本思考来增强空间推理的方法（SpatialVTS）。在空间视觉思考阶段，我们的模型被训练为能够自动生成与关键目标相关的位置特定标记。不仅提及了问题中的对象，还考虑了与推理相关的潜在对象。在空间文本思考阶段，模型基于视觉线索和对话进行长期思考，逐步推导出空间推理问题的答案。为了有效支持模型的训练，我们对手头的空间推理数据集进行了手动修正，消除了大量自动标注导致的错误标签，重新结构化了数据输入格式以增强泛化能力，并发展了具有逻辑推理细节的思考过程。在不引入额外信息（如遮罩或深度）的情况下，与其它模型相比，我们的模型在多个空间理解任务上的整体平均水平有显著提升。 

---
# The Xeno Sutra: Can Meaning and Value be Ascribed to an AI-Generated "Sacred" Text? 

**Title (ZH)**: 异界 sutra: 人工智能生成的“神圣”文本能否赋予其意义和价值？ 

**Authors**: Murray Shanahan, Tara Das, Robert Thurman  

**Link**: [PDF](https://arxiv.org/pdf/2507.20525)  

**Abstract**: This paper presents a case study in the use of a large language model to generate a fictional Buddhist "sutr"', and offers a detailed analysis of the resulting text from a philosophical and literary point of view. The conceptual subtlety, rich imagery, and density of allusion found in the text make it hard to causally dismiss on account of its mechanistic origin. This raises questions about how we, as a society, should come to terms with the potentially unsettling possibility of a technology that encroaches on human meaning-making. We suggest that Buddhist philosophy, by its very nature, is well placed to adapt. 

**Abstract (ZH)**: 该论文探讨了一种大型语言模型生成虚构佛教“ sutra ”的案例研究，并从哲学和文学的角度对生成文本进行了详细分析。文本中概念的细微、丰富的意象以及引文的密集性使其机械产生的来源难以因果轻视。这引发了关于社会应如何应对技术侵占人类意义构建的潜在令人不安的可能性的问题。我们建议，佛教哲学因其本质特点，非常适合应对这一挑战。 

---
# AQUA: A Large Language Model for Aquaculture & Fisheries 

**Title (ZH)**: AQUA：水产养殖与渔业领域的大型语言模型 

**Authors**: Praneeth Narisetty, Uday Kumar Reddy Kattamanchi, Lohit Akshant Nimma, Sri Ram Kaushik Karnati, Shiva Nagendra Babu Kore, Mounika Golamari, Tejashree Nageshreddy  

**Link**: [PDF](https://arxiv.org/pdf/2507.20520)  

**Abstract**: Aquaculture plays a vital role in global food security and coastal economies by providing sustainable protein sources. As the industry expands to meet rising demand, it faces growing challenges such as disease outbreaks, inefficient feeding practices, rising labor costs, logistical inefficiencies, and critical hatchery issues, including high mortality rates and poor water quality control. Although artificial intelligence has made significant progress, existing machine learning methods fall short of addressing the domain-specific complexities of aquaculture. To bridge this gap, we introduce AQUA, the first large language model (LLM) tailored for aquaculture, designed to support farmers, researchers, and industry practitioners. Central to this effort is AQUADAPT (Data Acquisition, Processing and Tuning), an Agentic Framework for generating and refining high-quality synthetic data using a combination of expert knowledge, largescale language models, and automated evaluation techniques. Our work lays the foundation for LLM-driven innovations in aquaculture research, advisory systems, and decision-making tools. 

**Abstract (ZH)**: 水产养殖在通过提供可持续蛋白质来源维护全球食物安全和沿海经济方面发挥着关键作用。随着行业扩展以满足不断增长的需求，它面临着越来越多的挑战，如疾病爆发、不合理的投喂实践、劳动力成本上升、物流效率低下以及关键繁殖场问题，包括高死亡率和水质控制不佳。尽管人工智能已经取得了显著进展，现有的机器学习方法在解决水产养殖领域的特定复杂性方面仍存在不足。为弥补这一差距，我们引入了AQUA，这是首款针对水产养殖领域的大型语言模型（LLM），旨在支持农民、研究人员和行业实践者。我们工作的核心是AQUADAPT（数据获取、处理和调优）框架，该框架结合专家知识、大规模语言模型和自动评估技术，用于生成和优化高质量的合成数据。我们的工作为基础模型在水产养殖研究、咨询系统和决策工具驱动的创新奠定了基础。 

---
# LLMs-guided adaptive compensator: Bringing Adaptivity to Automatic Control Systems with Large Language Models 

**Title (ZH)**: 大型语言模型引导的自适应补偿器：将自适应性引入自动控制系统 

**Authors**: Zhongchao Zhou, Yuxi Lu, Yaonan Zhu, Yifan Zhao, Bin He, Liang He, Wenwen Yu, Yusuke Iwasawa  

**Link**: [PDF](https://arxiv.org/pdf/2507.20509)  

**Abstract**: With rapid advances in code generation, reasoning, and problem-solving, Large Language Models (LLMs) are increasingly applied in robotics. Most existing work focuses on high-level tasks such as task decomposition. A few studies have explored the use of LLMs in feedback controller design; however, these efforts are restricted to overly simplified systems, fixed-structure gain tuning, and lack real-world validation. To further investigate LLMs in automatic control, this work targets a key subfield: adaptive control. Inspired by the framework of model reference adaptive control (MRAC), we propose an LLM-guided adaptive compensator framework that avoids designing controllers from scratch. Instead, the LLMs are prompted using the discrepancies between an unknown system and a reference system to design a compensator that aligns the response of the unknown system with that of the reference, thereby achieving adaptivity. Experiments evaluate five methods: LLM-guided adaptive compensator, LLM-guided adaptive controller, indirect adaptive control, learning-based adaptive control, and MRAC, on soft and humanoid robots in both simulated and real-world environments. Results show that the LLM-guided adaptive compensator outperforms traditional adaptive controllers and significantly reduces reasoning complexity compared to the LLM-guided adaptive controller. The Lyapunov-based analysis and reasoning-path inspection demonstrate that the LLM-guided adaptive compensator enables a more structured design process by transforming mathematical derivation into a reasoning task, while exhibiting strong generalizability, adaptability, and robustness. This study opens a new direction for applying LLMs in the field of automatic control, offering greater deployability and practicality compared to vision-language models. 

**Abstract (ZH)**: 随着代码生成、推理和问题解决的快速进步，大规模语言模型（LLMs）在机器人领域的应用日益增多。大多数现有工作集中在高层任务如任务分解上。少数研究探索了LLMs在反馈控制器设计中的应用；然而，这些努力主要限于过于简化的系统、固定结构增益调整，并缺乏现实世界的验证。为进一步研究LLMs在自动控制中的应用，本文将目标集中在自动控制的关键子领域：自适应控制。借鉴模型参考自适应控制（MRAC）框架，我们提出了一种由LLM指导的自适应补偿器框架，避免了从头设计控制器。相反，LLMs通过未知系统与参考系统的差异被提示，设计一个补偿器来使未知系统的行为与参考系统相匹配，从而实现自适应性。实验在模拟和现实环境中对软体机器人和人形机器人评估了五种方法：由LLM指导的自适应补偿器、由LLM指导的自适应控制器、间接自适应控制、基于学习的自适应控制和MRAC。结果表明，由LLM指导的自适应补偿器优于传统自适应控制器，并且在与由LLM指导的自适应控制器相比时显著降低了推理复杂性。通过Lyapunov分析和推理路径检查，证明了由LLM指导的自适应补偿器能够通过将数学推导转化为推理任务来实现更结构化的设计过程，同时具备强大的泛化能力、自适应性和鲁棒性。本研究为LLMs在自动控制领域的应用开辟了新方向，与视觉-语言模型相比，提供了更大的可部署性和实用性。 

---
# DmC: Nearest Neighbor Guidance Diffusion Model for Offline Cross-domain Reinforcement Learning 

**Title (ZH)**: DmC: 基于最近邻指导的离线跨域强化学习扩散模型 

**Authors**: Linh Le Pham Van, Minh Hoang Nguyen, Duc Kieu, Hung Le, Hung The Tran, Sunil Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2507.20499)  

**Abstract**: Cross-domain offline reinforcement learning (RL) seeks to enhance sample efficiency in offline RL by utilizing additional offline source datasets. A key challenge is to identify and utilize source samples that are most relevant to the target domain. Existing approaches address this challenge by measuring domain gaps through domain classifiers, target transition dynamics modeling, or mutual information estimation using contrastive loss. However, these methods often require large target datasets, which is impractical in many real-world scenarios. In this work, we address cross-domain offline RL under a limited target data setting, identifying two primary challenges: (1) Dataset imbalance, which is caused by large source and small target datasets and leads to overfitting in neural network-based domain gap estimators, resulting in uninformative measurements; and (2) Partial domain overlap, where only a subset of the source data is closely aligned with the target domain. To overcome these issues, we propose DmC, a novel framework for cross-domain offline RL with limited target samples. Specifically, DmC utilizes $k$-nearest neighbor ($k$-NN) based estimation to measure domain proximity without neural network training, effectively mitigating overfitting. Then, by utilizing this domain proximity, we introduce a nearest-neighbor-guided diffusion model to generate additional source samples that are better aligned with the target domain, thus enhancing policy learning with more effective source samples. Through theoretical analysis and extensive experiments in diverse MuJoCo environments, we demonstrate that DmC significantly outperforms state-of-the-art cross-domain offline RL methods, achieving substantial performance gains. 

**Abstract (ZH)**: 跨域离线强化学习（DmC）：在有限目标样本下的方法 

---
# Speaking in Words, Thinking in Logic: A Dual-Process Framework in QA Systems 

**Title (ZH)**: 言辞表达，逻辑思考：问答系统中的双重过程框架 

**Authors**: Tuan Bui, Trong Le, Phat Thai, Sang Nguyen, Minh Hua, Ngan Pham, Thang Bui, Tho Quan  

**Link**: [PDF](https://arxiv.org/pdf/2507.20491)  

**Abstract**: Recent advances in large language models (LLMs) have significantly enhanced question-answering (QA) capabilities, particularly in open-domain contexts. However, in closed-domain scenarios such as education, healthcare, and law, users demand not only accurate answers but also transparent reasoning and explainable decision-making processes. While neural-symbolic (NeSy) frameworks have emerged as a promising solution, leveraging LLMs for natural language understanding and symbolic systems for formal reasoning, existing approaches often rely on large-scale models and exhibit inefficiencies in translating natural language into formal logic representations.
To address these limitations, we introduce Text-JEPA (Text-based Joint-Embedding Predictive Architecture), a lightweight yet effective framework for converting natural language into first-order logic (NL2FOL). Drawing inspiration from dual-system cognitive theory, Text-JEPA emulates System 1 by efficiently generating logic representations, while the Z3 solver operates as System 2, enabling robust logical inference. To rigorously evaluate the NL2FOL-to-reasoning pipeline, we propose a comprehensive evaluation framework comprising three custom metrics: conversion score, reasoning score, and Spearman rho score, which collectively capture the quality of logical translation and its downstream impact on reasoning accuracy.
Empirical results on domain-specific datasets demonstrate that Text-JEPA achieves competitive performance with significantly lower computational overhead compared to larger LLM-based systems. Our findings highlight the potential of structured, interpretable reasoning frameworks for building efficient and explainable QA systems in specialized domains. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）显著增强了问答（QA）能力，特别是在开放领域中。然而，在教育、医疗和法律等封闭领域场景中，用户不仅需要准确的答案，还需要透明的推理和可解释的决策过程。虽然神经符号（NeSy）框架已 emerged 作为有希望的解决方案，利用大型语言模型进行自然语言理解，并利用符号系统进行形式推理，但现有方法往往依赖于大型模型并在将自然语言转换为形式逻辑表示方面表现出效率低下。

为了解决这些限制，我们引入了Text-JEPA（基于文本的联合嵌入预测架构），这是一种轻量级但有效的方法，用于将自然语言转换为一阶逻辑（NL2FOL）。Text-JEPA 从双系统认知理论汲取灵感，通过高效生成逻辑表示来模拟系统1，而Z3求解器则作为系统2，支持强大的逻辑推理。为严格评估NL2FOL到推理的工作流程，我们提出了一个包含三个自定义度量的整体评估框架：转换得分、推理得分和斯皮尔曼ρ得分，这些得分共同捕捉了逻辑翻译的质量及其对推理准确性的下游影响。

在特定领域的数据集上的实验证明，Text-JEPA 在显著降低计算开销的情况下实现了与更大规模的LLM系统相媲美的性能。我们的发现强调了结构化、可解释的推理框架在构建高效且可解释的问答系统方面的潜力。 

---
# Shapley-Value-Based Graph Sparsification for GNN Inference 

**Title (ZH)**: 基于Shapley值的图稀疏化方法用于GNN推理 

**Authors**: Selahattin Akkas, Ariful Azad  

**Link**: [PDF](https://arxiv.org/pdf/2507.20460)  

**Abstract**: Graph sparsification is a key technique for improving inference efficiency in Graph Neural Networks by removing edges with minimal impact on predictions. GNN explainability methods generate local importance scores, which can be aggregated into global scores for graph sparsification. However, many explainability methods produce only non-negative scores, limiting their applicability for sparsification. In contrast, Shapley value based methods assign both positive and negative contributions to node predictions, offering a theoretically robust and fair allocation of importance by evaluating many subsets of graphs. Unlike gradient-based or perturbation-based explainers, Shapley values enable better pruning strategies that preserve influential edges while removing misleading or adversarial connections. Our approach shows that Shapley value-based graph sparsification maintains predictive performance while significantly reducing graph complexity, enhancing both interpretability and efficiency in GNN inference. 

**Abstract (ZH)**: 图稀疏化是通过移除对预测影响最小的边以提高图神经网络推理效率的关键技术。基于图的解释性方法生成局部重要性分数，这些分数可以聚合为全局分数用于图稀疏化。然而，许多解释性方法仅生成非负分数，限制了其在稀疏化中的应用。相比之下，基于Shapley值的方法为节点预测分配正负贡献，通过评估图的多个子集提供理论上稳健且公平的重要性分配。与基于梯度或扰动的解释器不同，Shapley值使我们能够实现更好的剪枝策略，保留有影响力的边并移除误导性或对抗性的连接。我们的方法表明，基于Shapley值的图稀疏化能够在显著减少图复杂性的同时保持预测性能，从而在GNN推理中增强可解释性和效率。 

---
# When Prompts Go Wrong: Evaluating Code Model Robustness to Ambiguous, Contradictory, and Incomplete Task Descriptions 

**Title (ZH)**: 当提示出现错误：评价代码模型对模糊、矛盾和不完整任务描述的 robustness 

**Authors**: Maya Larbi, Amal Akli, Mike Papadakis, Rihab Bouyousfi, Maxime Cordy, Federica Sarro, Yves Le Traon  

**Link**: [PDF](https://arxiv.org/pdf/2507.20439)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive performance in code generation tasks under idealized conditions, where task descriptions are clear and precise. However, in practice, task descriptions frequently exhibit ambiguity, incompleteness, or internal contradictions. In this paper, we present the first empirical study examining the robustness of state-of-the-art code generation models when faced with such unclear task descriptions. We extend the HumanEval and MBPP benchmarks by systematically introducing realistic task descriptions flaws through guided mutation strategies, producing a dataset that mirrors the messiness of informal developer instructions. We evaluate multiple LLMs of varying sizes and architectures, analyzing their functional correctness and failure modes across task descriptions categories. Our findings reveal that even minor imperfections in task description phrasing can cause significant performance degradation, with contradictory task descriptions resulting in numerous logical errors. Moreover, while larger models tend to be more resilient than smaller variants, they are not immune to the challenges posed by unclear requirements. We further analyze semantic error patterns and identify correlations between description clarity, model behavior, and error types. Our results underscore the critical need for developing LLMs that are not only powerful but also robust to the imperfections inherent in natural user tasks, highlighting important considerations for improving model training strategies, designing more realistic evaluation benchmarks, and ensuring reliable deployment in practical software development environments. 

**Abstract (ZH)**: 大规模语言模型在面对含糊不清的任务描述时的稳健性研究 

---
# FAST: Similarity-based Knowledge Transfer for Efficient Policy Learning 

**Title (ZH)**: FAST：基于相似性知识迁移的高效策略学习 

**Authors**: Alessandro Capurso, Elia Piccoli, Davide Bacciu  

**Link**: [PDF](https://arxiv.org/pdf/2507.20433)  

**Abstract**: Transfer Learning (TL) offers the potential to accelerate learning by transferring knowledge across tasks. However, it faces critical challenges such as negative transfer, domain adaptation and inefficiency in selecting solid source policies. These issues often represent critical problems in evolving domains, i.e. game development, where scenarios transform and agents must adapt. The continuous release of new agents is costly and inefficient. In this work we challenge the key issues in TL to improve knowledge transfer, agents performance across tasks and reduce computational costs. The proposed methodology, called FAST - Framework for Adaptive Similarity-based Transfer, leverages visual frames and textual descriptions to create a latent representation of tasks dynamics, that is exploited to estimate similarity between environments. The similarity scores guides our method in choosing candidate policies from which transfer abilities to simplify learning of novel tasks. Experimental results, over multiple racing tracks, demonstrate that FAST achieves competitive final performance compared to learning-from-scratch methods while requiring significantly less training steps. These findings highlight the potential of embedding-driven task similarity estimations. 

**Abstract (ZH)**: Transfer Learning for Adaptive Similarity-based Task Transfer in Racing Games 

---
# ResCap-DBP: A Lightweight Residual-Capsule Network for Accurate DNA-Binding Protein Prediction Using Global ProteinBERT Embeddings 

**Title (ZH)**: ResCap-DBP：一种基于全局蛋白质BERT嵌入的轻量级残差胶囊网络，用于准确的DNA结合蛋白预测 

**Authors**: Samiul Based Shuvo, Tasnia Binte Mamun, U Rajendra Acharya  

**Link**: [PDF](https://arxiv.org/pdf/2507.20426)  

**Abstract**: DNA-binding proteins (DBPs) are integral to gene regulation and cellular processes, making their accurate identification essential for understanding biological functions and disease mechanisms. Experimental methods for DBP identification are time-consuming and costly, driving the need for efficient computational prediction techniques. In this study, we propose a novel deep learning framework, ResCap-DBP, that combines a residual learning-based encoder with a one-dimensional Capsule Network (1D-CapsNet) to predict DBPs directly from raw protein sequences. Our architecture incorporates dilated convolutions within residual blocks to mitigate vanishing gradient issues and extract rich sequence features, while capsule layers with dynamic routing capture hierarchical and spatial relationships within the learned feature space. We conducted comprehensive ablation studies comparing global and local embeddings from ProteinBERT and conventional one-hot encoding. Results show that ProteinBERT embeddings substantially outperform other representations on large datasets. Although one-hot encoding showed marginal advantages on smaller datasets, such as PDB186, it struggled to scale effectively. Extensive evaluations on four pairs of publicly available benchmark datasets demonstrate that our model consistently outperforms current state-of-the-art methods. It achieved AUC scores of 98.0% and 89.5% on PDB14189andPDB1075, respectively. On independent test sets PDB2272 and PDB186, the model attained top AUCs of 83.2% and 83.3%, while maintaining competitive performance on larger datasets such as PDB20000. Notably, the model maintains a well balanced sensitivity and specificity across datasets. These results demonstrate the efficacy and generalizability of integrating global protein representations with advanced deep learning architectures for reliable and scalable DBP prediction in diverse genomic contexts. 

**Abstract (ZH)**: DNA结合蛋白(DBPs)的鉴定对于理解基因调控和细胞过程至关重要，因此其准确识别是了解生物功能和疾病机制的基础。实验方法耗时且昂贵，推动了高效计算预测技术的需求。本研究提出了一种新型深度学习框架ResCap-DBP，该框架结合了基于残差学习的编码器和一维胶囊网络(1D-CapsNet)，可以直接从原始蛋白质序列中预测DBPs。我们的架构在其残差块中包含扩张卷积，以缓解消失梯度问题并提取丰富的序列特征，同时动态路由的胶囊层捕获学习特征空间中的层级和空间关系。我们对ProteinBERT和传统的一热编码进行了全面的消融研究，结果显示ProteinBERT嵌入在大型数据集上显著优于其他表示方法。尽管一热编码在较小的数据集如PDB186上表现出轻微的优势，但在有效扩展方面存在困难。在四个公开可用基准数据集对的广泛评估中，我们的模型始终优于当前最先进的方法。该模型在PDB14189和PDB1075上的AUC得分分别为98.0%和89.5%，在独立测试集PDB2272和PDB186上分别达到了83.2%和83.3%的AUC，并在更大的数据集PDB20000上维持了竞争力。值得注意的是，该模型在不同数据集上保持了良好的敏感性和特异性平衡。这些结果展示了将全局蛋白质表示与先进的深度学习架构集成用于不同基因组背景下可靠且可扩展的DBP预测的有效性和通用性。 

---
# CodeNER: Code Prompting for Named Entity Recognition 

**Title (ZH)**: CodeNER: 代码提示用于命名实体识别 

**Authors**: Sungwoo Han, Hyeyeon Kim, Jingun Kwon, Hidetaka Kamigaito, Manabu Okumura  

**Link**: [PDF](https://arxiv.org/pdf/2507.20423)  

**Abstract**: Recent studies have explored various approaches for treating candidate named entity spans as both source and target sequences in named entity recognition (NER) by leveraging large language models (LLMs). Although previous approaches have successfully generated candidate named entity spans with suitable labels, they rely solely on input context information when using LLMs, particularly, ChatGPT. However, NER inherently requires capturing detailed labeling requirements with input context information. To address this issue, we propose a novel method that leverages code-based prompting to improve the capabilities of LLMs in understanding and performing NER. By embedding code within prompts, we provide detailed BIO schema instructions for labeling, thereby exploiting the ability of LLMs to comprehend long-range scopes in programming languages. Experimental results demonstrate that the proposed code-based prompting method outperforms conventional text-based prompting on ten benchmarks across English, Arabic, Finnish, Danish, and German datasets, indicating the effectiveness of explicitly structuring NER instructions. We also verify that combining the proposed code-based prompting method with the chain-of-thought prompting further improves performance. 

**Abstract (ZH)**: Recent Studies Have Explored Various Approaches for Treating Candidate Named Entity Spans as Both Source and Target Sequences in NER by Leveraging Large Language Models 

---
# Survey of NLU Benchmarks Diagnosing Linguistic Phenomena: Why not Standardize Diagnostics Benchmarks? 

**Title (ZH)**: 自然语言理解基准在诊断语言现象方面的调查：为何不标准化诊断基准？ 

**Authors**: Khloud AL Jallad, Nada Ghneim, Ghaida Rebdawi  

**Link**: [PDF](https://arxiv.org/pdf/2507.20419)  

**Abstract**: Natural Language Understanding (NLU) is a basic task in Natural Language Processing (NLP). The evaluation of NLU capabilities has become a trending research topic that attracts researchers in the last few years, resulting in the development of numerous benchmarks. These benchmarks include various tasks and datasets in order to evaluate the results of pretrained models via public leaderboards. Notably, several benchmarks contain diagnostics datasets designed for investigation and fine-grained error analysis across a wide range of linguistic phenomena. This survey provides a comprehensive review of available English, Arabic, and Multilingual NLU benchmarks, with a particular emphasis on their diagnostics datasets and the linguistic phenomena they covered. We present a detailed comparison and analysis of these benchmarks, highlighting their strengths and limitations in evaluating NLU tasks and providing in-depth error analysis. When highlighting the gaps in the state-of-the-art, we noted that there is no naming convention for macro and micro categories or even a standard set of linguistic phenomena that should be covered. Consequently, we formulated a research question regarding the evaluation metrics of the evaluation diagnostics benchmarks: "Why do not we have an evaluation standard for the NLU evaluation diagnostics benchmarks?" similar to ISO standard in industry. We conducted a deep analysis and comparisons of the covered linguistic phenomena in order to support experts in building a global hierarchy for linguistic phenomena in future. We think that having evaluation metrics for diagnostics evaluation could be valuable to gain more insights when comparing the results of the studied models on different diagnostics benchmarks. 

**Abstract (ZH)**: 自然语言理解能力评估：面向诊断数据集的基准比较与分析 

---
# Cognitive Chain-of-Thought: Structured Multimodal Reasoning about Social Situations 

**Title (ZH)**: 认知链式思考：结构化多模态社会情境推理 

**Authors**: Eunkyu Park, Wesley Hanwen Deng, Gunhee Kim, Motahhare Eslami, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2507.20409)  

**Abstract**: Chain-of-Thought (CoT) prompting helps models think step by step. But what happens when they must see, understand, and judge-all at once? In visual tasks grounded in social context, where bridging perception with norm-grounded judgments is essential, flat CoT often breaks down. We introduce Cognitive Chain-of-Thought (CoCoT), a prompting strategy that scaffolds VLM reasoning through three cognitively inspired stages: perception, situation, and norm. Our experiments show that, across multiple multimodal benchmarks (including intent disambiguation, commonsense reasoning, and safety), CoCoT consistently outperforms CoT and direct prompting (+8\% on average). Our findings demonstrate that cognitively grounded reasoning stages enhance interpretability and social awareness in VLMs, paving the way for safer and more reliable multimodal systems. 

**Abstract (ZH)**: 认知Chain-of-Thought (CoCoT) 提策助力VLM在多模态任务中的推理能力 

---
# A Multi-Stage Hybrid CNN-Transformer Network for Automated Pediatric Lung Sound Classification 

**Title (ZH)**: 多阶段混合CNN-Transformer网络在自动化儿童肺音分类中的应用 

**Authors**: Samiul Based Shuvo, Taufiq Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2507.20408)  

**Abstract**: Automated analysis of lung sound auscultation is essential for monitoring respiratory health, especially in regions facing a shortage of skilled healthcare workers. While respiratory sound classification has been widely studied in adults, its ap plication in pediatric populations, particularly in children aged <6 years, remains an underexplored area. The developmental changes in pediatric lungs considerably alter the acoustic proper ties of respiratory sounds, necessitating specialized classification approaches tailored to this age group. To address this, we propose a multistage hybrid CNN-Transformer framework that combines CNN-extracted features with an attention-based architecture to classify pediatric respiratory diseases using scalogram images from both full recordings and individual breath events. Our model achieved an overall score of 0.9039 in binary event classifi cation and 0.8448 in multiclass event classification by employing class-wise focal loss to address data imbalance. At the recording level, the model attained scores of 0.720 for ternary and 0.571 for multiclass classification. These scores outperform the previous best models by 3.81% and 5.94%, respectively. This approach offers a promising solution for scalable pediatric respiratory disease diagnosis, especially in resource-limited settings. 

**Abstract (ZH)**: 自动化分析肺音对于监测呼吸健康，尤其是在医护人员不足的地区，至关重要。虽然呼吸道声音分类在成人中已被广泛研究，但其在儿科人群中的应用，特别是在<6岁儿童中，仍是一个未充分探索的领域。儿童肺部的发育变化显著改变了呼吸声音的声学特性，需要针对这一年龄段的专门分类方法。为此，我们提出了一种多阶段混合CNN-Transformer框架，结合CNN提取的特征与基于注意力的架构，使用小波图像对完整记录和单个呼吸事件进行儿童呼吸疾病分类。我们的模型通过采用类别平衡的焦点损失实现了二分类0.9039和多分类0.8448的评分。在记录层面，模型分别实现了三分类0.720和多分类0.571的评分，这些评分分别比之前最佳模型高出3.81%和5.94%。该方法为资源有限的环境中可扩展的儿童呼吸疾病诊断提供了一个有前景的解决方案。 

---
# Solving Scene Understanding for Autonomous Navigation in Unstructured Environments 

**Title (ZH)**: 解决自主导航于未结构化环境中的场景理解问题 

**Authors**: Naveen Mathews Renji, Kruthika K, Manasa Keshavamurthy, Pooja Kumari, S. Rajarajeswari  

**Link**: [PDF](https://arxiv.org/pdf/2507.20389)  

**Abstract**: Autonomous vehicles are the next revolution in the automobile industry and they are expected to revolutionize the future of transportation. Understanding the scenario in which the autonomous vehicle will operate is critical for its competent functioning. Deep Learning has played a massive role in the progress that has been made till date. Semantic Segmentation, the process of annotating every pixel of an image with an object class, is one crucial part of this scene comprehension using Deep Learning. It is especially useful in Autonomous Driving Research as it requires comprehension of drivable and non-drivable areas, roadside objects and the like. In this paper semantic segmentation has been performed on the Indian Driving Dataset which has been recently compiled on the urban and rural roads of Bengaluru and Hyderabad. This dataset is more challenging compared to other datasets like Cityscapes, since it is based on unstructured driving environments. It has a four level hierarchy and in this paper segmentation has been performed on the first level. Five different models have been trained and their performance has been compared using the Mean Intersection over Union. These are UNET, UNET+RESNET50, DeepLabsV3, PSPNet and SegNet. The highest MIOU of 0.6496 has been achieved. The paper discusses the dataset, exploratory data analysis, preparation, implementation of the five models and studies the performance and compares the results achieved in the process. 

**Abstract (ZH)**: 自主驾驶车辆是汽车工业的下一次革命，它们有望重塑交通的未来。理解自主车辆将要运行的场景对于其有效运行至关重要。深度学习在迄今为止取得的进步中发挥了巨大作用。语义分割，即为图像中的每个像素赋予对象类别的过程，是使用深度学习进行场景理解的一个关键组成部分。在自主驾驶研究中尤其有用，因为它要求理解可行驶和不可行驶区域、路边物体等。本文在印度驾驶数据集上进行了语义分割，该数据集是最近在班加罗尔和海得拉巴的城乡道路上编译的。与Cityscapes等其他数据集相比，该数据集更具挑战性，因为它基于非结构化的驾驶环境。该数据集采用了四层层级结构，在本文中对第一层进行了分割。训练了五种不同的模型，并使用平均交并比（Mean Intersection over Union）比较了它们的性能。这些模型包括UNET、UNET+RESNET50、DeepLabsV3、PSPNet和SegNet。本文最高达到了0.6496的平均交并比。本文讨论了数据集、探索性数据分析、模型的准备和实施、以及研究性能并比较了整个过程中实现的结果。 

---
# WBHT: A Generative Attention Architecture for Detecting Black Hole Anomalies in Backbone Networks 

**Title (ZH)**: WBHT：一种用于主干网络中检测黑洞异常的生成注意力架构 

**Authors**: Kiymet Kaya, Elif Ak, Sule Gunduz Oguducu  

**Link**: [PDF](https://arxiv.org/pdf/2507.20373)  

**Abstract**: We propose the Wasserstein Black Hole Transformer (WBHT) framework for detecting black hole (BH) anomalies in communication networks. These anomalies cause packet loss without failure notifications, disrupting connectivity and leading to financial losses. WBHT combines generative modeling, sequential learning, and attention mechanisms to improve BH anomaly detection. It integrates a Wasserstein generative adversarial network with attention mechanisms for stable training and accurate anomaly identification. The model uses long-short-term memory layers to capture long-term dependencies and convolutional layers for local temporal patterns. A latent space encoding mechanism helps distinguish abnormal network behavior. Tested on real-world network data, WBHT outperforms existing models, achieving significant improvements in F1 score (ranging from 1.65% to 58.76%). Its efficiency and ability to detect previously undetected anomalies make it a valuable tool for proactive network monitoring and security, especially in mission-critical networks. 

**Abstract (ZH)**: Wasserstein黑洞变压器（WBHT）框架用于检测通信网络中的黑洞（BH）异常 

---
# Clustering by Attention: Leveraging Prior Fitted Transformers for Data Partitioning 

**Title (ZH)**: 基于注意力的聚类：利用先验拟合的Transformer进行数据分区 

**Authors**: Ahmed Shokry, Ayman Khalafallah  

**Link**: [PDF](https://arxiv.org/pdf/2507.20369)  

**Abstract**: Clustering is a core task in machine learning with wide-ranging applications in data mining and pattern recognition. However, its unsupervised nature makes it inherently challenging. Many existing clustering algorithms suffer from critical limitations: they often require careful parameter tuning, exhibit high computational complexity, lack interpretability, or yield suboptimal accuracy, especially when applied to large-scale datasets. In this paper, we introduce a novel clustering approach based on meta-learning. Our approach eliminates the need for parameter optimization while achieving accuracy that outperforms state-of-the-art clustering techniques. The proposed technique leverages a few pre-clustered samples to guide the clustering process for the entire dataset in a single forward pass. Specifically, we employ a pre-trained Prior-Data Fitted Transformer Network (PFN) to perform clustering. The algorithm computes attention between the pre-clustered samples and the unclustered samples, allowing it to infer cluster assignments for the entire dataset based on the learned relation. We theoretically and empirically demonstrate that, given just a few pre-clustered examples, the model can generalize to accurately cluster the rest of the dataset. Experiments on challenging benchmark datasets show that our approach can successfully cluster well-separated data without any pre-clustered samples, and significantly improves performance when a few clustered samples are provided. We show that our approach is superior to the state-of-the-art techniques. These results highlight the effectiveness and scalability of our approach, positioning it as a promising alternative to existing clustering techniques. 

**Abstract (ZH)**: 基于元学习的聚类方法：无需参数优化即可实现优于现有技术的聚类性能 

---
# A Theory of $θ$-Expectations 

**Title (ZH)**: $θ$-期望理论 

**Authors**: Qian Qi  

**Link**: [PDF](https://arxiv.org/pdf/2507.20353)  

**Abstract**: The canonical theory of stochastic calculus under ambiguity, founded on sub-additivity, is insensitive to non-convex uncertainty structures, leading to an identifiability impasse. This paper develops a mathematical framework for an identifiable calculus sensitive to non-convex geometry. We introduce the $\theta$-BSDE, a class of backward stochastic differential equations where the driver is determined by a pointwise maximization over a primitive, possibly non-convex, uncertainty set. The system's tractability is predicated not on convexity, but on a global analytic hypothesis: the existence of a unique and globally Lipschitz maximizer map for the driver function. Under this hypothesis, which carves out a tractable class of models, we establish well-posedness via a fixed-point argument. For a distinct, geometrically regular class of models, we prove a result of independent interest: under non-degeneracy conditions from Malliavin calculus, the maximizer is unique along any solution path, ensuring the model's internal consistency. We clarify the fundamental logical gap between this pathwise property and the global regularity required by our existence proof. The resulting valuation operator defines a dynamically consistent expectation, and we establish its connection to fully nonlinear PDEs via a Feynman-Kac formula. 

**Abstract (ZH)**: 不确定结构下的可识别随机 calculus 理论：基于子加性和非凸几何的数学框架 

---
# Cultivating Helpful, Personalized, and Creative AI Tutors: A Framework for Pedagogical Alignment using Reinforcement Learning 

**Title (ZH)**: 培养有帮助、个性化和创造性的AI导师：一种基于强化学习的教学对齐框架 

**Authors**: Siyu Song, Wentao Liu, Ye Lu, Ruohua Zhang, Tao Liu, Jinze Lv, Xinyun Wang, Aimin Zhou, Fei Tan, Bo Jiang, Hao Hao  

**Link**: [PDF](https://arxiv.org/pdf/2507.20335)  

**Abstract**: The integration of large language models (LLMs) into education presents unprecedented opportunities for scalable personalized learning. However, standard LLMs often function as generic information providers, lacking alignment with fundamental pedagogical principles such as helpfulness, student-centered personalization, and creativity cultivation. To bridge this gap, we propose EduAlign, a novel framework designed to guide LLMs toward becoming more effective and responsible educational assistants. EduAlign consists of two main stages. In the first stage, we curate a dataset of 8k educational interactions and annotate them-both manually and automatically-along three key educational dimensions: Helpfulness, Personalization, and Creativity (HPC). These annotations are used to train HPC-RM, a multi-dimensional reward model capable of accurately scoring LLM outputs according to these educational principles. We further evaluate the consistency and reliability of this reward model. In the second stage, we leverage HPC-RM as a reward signal to fine-tune a pre-trained LLM using Group Relative Policy Optimization (GRPO) on a set of 2k diverse prompts. We then assess the pre- and post-finetuning models on both educational and general-domain benchmarks across the three HPC dimensions. Experimental results demonstrate that the fine-tuned model exhibits significantly improved alignment with pedagogical helpfulness, personalization, and creativity stimulation. This study presents a scalable and effective approach to aligning LLMs with nuanced and desirable educational traits, paving the way for the development of more engaging, pedagogically aligned AI tutors. 

**Abstract (ZH)**: 大型语言模型（LLMs）融入教育呈现了前所未有的个性化学习 scalability 机会，然而标准的LLMs通常作为通用信息提供者运作，缺乏与教学基本原则如有益性、学生中心化个性化和创造力培养的契合。为弥合这一差距，我们提出EduAlign，这是一种新型框架，旨在引导LLMs成为更有效且负责任的教育助手。EduAlign包括两个主要阶段。在第一阶段，我们编纂了一个包含8000个教育交互的数据集，并手动和自动地对其进行三个关键教育维度的帮助性、个性化和创造力（HPC）的标注。这些标注用于训练HPC-RM，这是一种多维度奖励模型，能够精确评分LLMs输出，依据这些教育原则。我们进一步评估了该奖励模型的一致性和可靠性。在第二阶段，我们利用HPC-RM作为奖励信号，通过组相对策略优化（GRPO）对预训练的LLMs进行微调，使用2000个多样性的提示。然后，我们在三个HPC维度上的教育和通用领域基准上评估了微调前后的模型。实验结果表明，微调后的模型在教学上有明显改进的契合度，个性化和创造力的激发也得到提升。本研究提出了一种可扩展且有效的方法，将LLMs与精细化且可取的教育特质对齐，为开发更具吸引力且教学目标对齐的人工智能导师铺平了道路。 

---
# MIPS: a Multimodal Infinite Polymer Sequence Pre-training Framework for Polymer Property Prediction 

**Title (ZH)**: MIPS：一种用于聚合物性质预测的多模态无限聚合物序列预训练框架 

**Authors**: Jiaxi Wang, Yaosen Min, Xun Zhu, Miao Li, Ji Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.20326)  

**Abstract**: Polymers, composed of repeating structural units called monomers, are fundamental materials in daily life and industry. Accurate property prediction for polymers is essential for their design, development, and application. However, existing modeling approaches, which typically represent polymers by the constituent monomers, struggle to capture the whole properties of polymer, since the properties change during the polymerization process. In this study, we propose a Multimodal Infinite Polymer Sequence (MIPS) pre-training framework, which represents polymers as infinite sequences of monomers and integrates both topological and spatial information for comprehensive modeling. From the topological perspective, we generalize message passing mechanism (MPM) and graph attention mechanism (GAM) to infinite polymer sequences. For MPM, we demonstrate that applying MPM to infinite polymer sequences is equivalent to applying MPM on the induced star-linking graph of monomers. For GAM, we propose to further replace global graph attention with localized graph attention (LGA). Moreover, we show the robustness of the "star linking" strategy through Repeat and Shift Invariance Test (RSIT). Despite its robustness, "star linking" strategy exhibits limitations when monomer side chains contain ring structures, a common characteristic of polymers, as it fails the Weisfeiler-Lehman~(WL) test. To overcome this issue, we propose backbone embedding to enhance the capability of MPM and LGA on infinite polymer sequences. From the spatial perspective, we extract 3D descriptors of repeating monomers to capture spatial information. Finally, we design a cross-modal fusion mechanism to unify the topological and spatial information. Experimental validation across eight diverse polymer property prediction tasks reveals that MIPS achieves state-of-the-art performance. 

**Abstract (ZH)**: 多模态无限聚合物序列预训练框架（MIPS）：综合拓扑和空间信息的聚合物性质预测 

---
# A Comparative Study of OpenMP Scheduling Algorithm Selection Strategies 

**Title (ZH)**: OpenMP调度算法选择策略的比较研究 

**Authors**: Jonas H. Müller Korndörfer, Ali Mohammed, Ahmed Eleliemy, Quentin Guilloteau, Reto Krummenacher, Florina M. Ciorba  

**Link**: [PDF](https://arxiv.org/pdf/2507.20312)  

**Abstract**: Scientific and data science applications are becoming increasingly complex, with growing computational and memory demands. Modern high performance computing (HPC) systems provide high parallelism and heterogeneity across nodes, devices, and cores. To achieve good performance, effective scheduling and load balancing techniques are essential. Parallel programming frameworks such as OpenMP now offer a variety of advanced scheduling algorithms to support diverse applications and platforms. This creates an instance of the scheduling algorithm selection problem, which involves identifying the most suitable algorithm for a given combination of workload and system characteristics.
In this work, we explore learning-based approaches for selecting scheduling algorithms in OpenMP. We propose and evaluate expert-based and reinforcement learning (RL)-based methods, and conduct a detailed performance analysis across six applications and three systems. Our results show that RL methods are capable of learning high-performing scheduling decisions, although they require significant exploration, with the choice of reward function playing a key role. Expert-based methods, in contrast, rely on prior knowledge and involve less exploration, though they may not always identify the optimal algorithm for a specific application-system pair. By combining expert knowledge with RL-based learning, we achieve improved performance and greater adaptability.
Overall, this work demonstrates that dynamic selection of scheduling algorithms during execution is both viable and beneficial for OpenMP applications. The approach can also be extended to MPI-based programs, enabling optimization of scheduling decisions across multiple levels of parallelism. 

**Abstract (ZH)**: 科学和数据科学应用日益复杂，对计算能力和内存的需求也在增长。现代高性能计算（HPC）系统在节点、设备和内核上提供了高并行性和异构性。为了实现良好的性能，有效的调度和负载均衡技术是必不可少的。OpenMP等并行编程框架现在提供了种类繁多的高级调度算法，以支持多样化的应用程序和平台。这创造了调度算法选择问题的实例，涉及根据工作负载和系统特性识别最合适的算法。

在本工作中，我们探讨了在OpenMP中使用基于学习的方法选择调度算法。我们提出了基于专家知识和强化学习（RL）的方法，并在六个应用程序和三种系统上进行了详细的性能分析。结果显示，RL方法能够学习高性能的调度决策，尽管它们需要大量的探索，而奖励函数的选择起着关键作用。相比之下，基于专家知识的方法依赖于先验知识并涉及较少的探索，但它们可能不总是能够识别特定应用程序-系统配对的最佳算法。通过对专家知识与RL学习的结合，我们实现了改进的性能和更高的适应性。

总体而言，本工作证明了在执行过程中动态选择调度算法对于OpenMP应用程序既是可行的又是有益的。该方法还可以扩展到基于MPI的程序中，从而使跨多个并行级别调度决策的优化成为可能。 

---
# Towards Generalized Parameter Tuning in Coherent Ising Machines: A Portfolio-Based Approach 

**Title (ZH)**: 相干伊辛机中通用参数调优的研究：一种基于投资组合的方法 

**Authors**: Tatsuro Hanyu, Takahiro Katagiri, Daichi Mukunoki, Tetsuya Hoshino  

**Link**: [PDF](https://arxiv.org/pdf/2507.20295)  

**Abstract**: Coherent Ising Machines (CIMs) have recently gained attention as a promising computing model for solving combinatorial optimization problems. In particular, the Chaotic Amplitude Control (CAC) algorithm has demonstrated high solution quality, but its performance is highly sensitive to a large number of hyperparameters, making efficient tuning essential. In this study, we present an algorithm portfolio approach for hyperparameter tuning in CIMs employing Chaotic Amplitude Control with momentum (CACm) algorithm. Our method incorporates multiple search strategies, enabling flexible and effective adaptation to the characteristics of the hyperparameter space. Specifically, we propose two representative tuning methods, Method A and Method B. Method A optimizes each hyperparameter sequentially with a fixed total number of trials, while Method B prioritizes hyperparameters based on initial evaluations before applying Method A in order. Performance evaluations were conducted on the Supercomputer "Flow" at Nagoya University, using planted Wishart instances and Time to Solution (TTS) as the evaluation metric. Compared to the baseline performance with best-known hyperparameters, Method A achieved up to 1.47x improvement, and Method B achieved up to 1.65x improvement. These results demonstrate the effectiveness of the algorithm portfolio approach in enhancing the tuning process for CIMs. 

**Abstract (ZH)**: Coherent Ising Machines中Chaotic Amplitude Control算法及其动量优化的超参数调优算法组合研究 

---
# Learning from Expert Factors: Trajectory-level Reward Shaping for Formulaic Alpha Mining 

**Title (ZH)**: 基于专家因素的学习：公式化alpha挖掘的轨迹级奖励塑形 

**Authors**: Junjie Zhao, Chengxi Zhang, Chenkai Wang, Peng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20263)  

**Abstract**: Reinforcement learning (RL) has successfully automated the complex process of mining formulaic alpha factors, for creating interpretable and profitable investment strategies. However, existing methods are hampered by the sparse rewards given the underlying Markov Decision Process. This inefficiency limits the exploration of the vast symbolic search space and destabilizes the training process. To address this, Trajectory-level Reward Shaping (TLRS), a novel reward shaping method, is proposed. TLRS provides dense, intermediate rewards by measuring the subsequence-level similarity between partially generated expressions and a set of expert-designed formulas. Furthermore, a reward centering mechanism is introduced to reduce training variance. Extensive experiments on six major Chinese and U.S. stock indices show that TLRS significantly improves the predictive power of mined factors, boosting the Rank Information Coefficient by 9.29% over existing potential-based shaping algorithms. Notably, TLRS achieves a major leap in computational efficiency by reducing its time complexity with respect to the feature dimension from linear to constant, which is a significant improvement over distance-based baselines. 

**Abstract (ZH)**: 强化学习（RL）已成功自动化了公式化alpha因子挖掘的复杂过程，以创建可解释且盈利的投资策略。然而，现有方法因潜在的马尔可夫决策过程提供的稀疏奖励而受限。这种低效率限制了对广阔符号搜索空间的探索，并导致训练过程不稳定。为解决这一问题，提出了一种新型的奖励塑造方法——轨迹级奖励塑造（TLRS）。TLRS通过测量部分生成表达式与一组专家设计公式的子序列级相似性，提供了密集的中间奖励。此外，引入了一种奖励居中机制以降低训练方差。在六大主要中国和美国股票指数上的广泛实验表明，TLRS显著提高了挖掘因子的预测能力，相对于现有的基于潜力的奖励塑造算法，提升了排名信息系数9.29%。值得注意的是，TLRS通过将其时间复杂度相对于特征维度从线性降低到常数，实现了显著的计算效率提升，这在基于距离的基线方法上得到了显著改善。 

---
# Post-Completion Learning for Language Models 

**Title (ZH)**: 完成后的语言模型学习 

**Authors**: Xiang Fei, Siqi Wang, Shu Wei, Yuxiang Nie, Wei Shi, Hao Feng, Can Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20252)  

**Abstract**: Current language model training paradigms typically terminate learning upon reaching the end-of-sequence (<eos>}) token, overlooking the potential learning opportunities in the post-completion space. We propose Post-Completion Learning (PCL), a novel training framework that systematically utilizes the sequence space after model output completion, to enhance both the reasoning and self-evaluation abilities. PCL enables models to continue generating self-assessments and reward predictions during training, while maintaining efficient inference by stopping at the completion point.
To fully utilize this post-completion space, we design a white-box reinforcement learning method: let the model evaluate the output content according to the reward rules, then calculate and align the score with the reward functions for supervision. We implement dual-track SFT to optimize both reasoning and evaluation capabilities, and mixed it with RL training to achieve multi-objective hybrid optimization.
Experimental results on different datasets and models demonstrate consistent improvements over traditional SFT and RL methods. Our method provides a new technical path for language model training that enhances output quality while preserving deployment efficiency. 

**Abstract (ZH)**: 当前的语言模型训练范式通常在遇到序列结束符（<eos>）时终止学习，忽视了完成之后的空间中的潜在学习机会。我们提出了后完成学习（PCL），这是一种系统利用模型输出完成后序列空间的新训练框架，以增强推理和自我评估能力。PCL使模型在训练过程中继续生成自我评估和奖励预测，同时通过在完成点停止来保持高效的推理。

为了充分利用这一后完成空间，我们设计了一种白盒强化学习方法：让模型根据奖励规则评估输出内容，然后计算并调整评分以匹配奖励函数进行监督。我们采用双重轨道的自我对齐训练（SFT）来优化推理和评估能力，并将其与RL训练结合，实现多目标混合优化。

在不同数据集和模型上的实验结果表明，我们的方法在传统SFT和RL方法上一致地提高了性能。该方法为语言模型训练提供了一条新的技术路径，在提高输出质量的同时保持部署效率。 

---
# Protein-SE(3): Benchmarking SE(3)-based Generative Models for Protein Structure Design 

**Title (ZH)**: Protein-SE(3): 基于SE(3)的蛋白质结构设计生成模型基准研究 

**Authors**: Lang Yu, Zhangyang Gao, Cheng Tan, Qin Chen, Jie Zhou, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2507.20243)  

**Abstract**: SE(3)-based generative models have shown great promise in protein geometry modeling and effective structure design. However, the field currently lacks a modularized benchmark to enable comprehensive investigation and fair comparison of different methods. In this paper, we propose Protein-SE(3), a new benchmark based on a unified training framework, which comprises protein scaffolding tasks, integrated generative models, high-level mathematical abstraction, and diverse evaluation metrics. Recent advanced generative models designed for protein scaffolding, from multiple perspectives like DDPM (Genie1 and Genie2), Score Matching (FrameDiff and RfDiffusion) and Flow Matching (FoldFlow and FrameFlow) are integrated into our framework. All integrated methods are fairly investigated with the same training dataset and evaluation metrics. Furthermore, we provide a high-level abstraction of the mathematical foundations behind the generative models, enabling fast prototyping of future algorithms without reliance on explicit protein structures. Accordingly, we release the first comprehensive benchmark built upon unified training framework for SE(3)-based protein structure design, which is publicly accessible at this https URL. 

**Abstract (ZH)**: SE(3)-基于的生成模型在蛋白质几何建模和有效结构设计中展现出了巨大潜力。然而，当前该领域缺乏模块化的基准以实现不同方法的全面调查和公平比较。在本文中，我们提出Protein-SE(3)作为基于统一训练框架的新基准，涵盖蛋白质支架任务、集成生成模型、高层次的数学抽象和多样化的评价指标。近年来从多个视角（如DDPM（Genie1和Genie2）、Score Matching（FrameDiff和RfDiffusion）和Flow Matching（FoldFlow和FrameFlow））设计的用于蛋白质支架的先进生成模型被整合到我们的框架中。所有整合的方法在相同的训练数据集和评价指标下进行了公平的调查。此外，我们提供了生成模型背后的数学基础的高层次抽象，使得未来算法的快速原型设计无需依赖显式的蛋白质结构。因此，我们发布了首个基于统一训练框架的SE(3)-基于蛋白质结构设计基准，该基准公开可获取。 

---
# Multi-Attention Stacked Ensemble for Lung Cancer Detection in CT Scans 

**Title (ZH)**: 基于CT扫描的肺癌检测的多注意力堆叠集成方法 

**Authors**: Uzzal Saha, Surya Prakash  

**Link**: [PDF](https://arxiv.org/pdf/2507.20221)  

**Abstract**: In this work, we address the challenge of binary lung nodule classification (benign vs malignant) using CT images by proposing a multi-level attention stacked ensemble of deep neural networks. Three pretrained backbones - EfficientNet V2 S, MobileViT XXS, and DenseNet201 - are each adapted with a custom classification head tailored to 96 x 96 pixel inputs. A two-stage attention mechanism learns both model-wise and class-wise importance scores from concatenated logits, and a lightweight meta-learner refines the final prediction. To mitigate class imbalance and improve generalization, we employ dynamic focal loss with empirically calculated class weights, MixUp augmentation during training, and test-time augmentation at inference. Experiments on the LIDC-IDRI dataset demonstrate exceptional performance, achieving 98.09 accuracy and 0.9961 AUC, representing a 35 percent reduction in error rate compared to state-of-the-art methods. The model exhibits balanced performance across sensitivity (98.73) and specificity (98.96), with particularly strong results on challenging cases where radiologist disagreement was high. Statistical significance testing confirms the robustness of these improvements across multiple experimental runs. Our approach can serve as a robust, automated aid for radiologists in lung cancer screening. 

**Abstract (ZH)**: 基于多层注意力堆叠ensemble的深度神经网络在CT图像上实现二元肺结节分类（良性 vs 恶性） 

---
# Humanoid Occupancy: Enabling A Generalized Multimodal Occupancy Perception System on Humanoid Robots 

**Title (ZH)**: 类人形占用感知：为类人机器人实现通用多模态占用感知系统 

**Authors**: Wei Cui, Haoyu Wang, Wenkang Qin, Yijie Guo, Gang Han, Wen Zhao, Jiahang Cao, Zhang Zhang, Jiaru Zhong, Jingkai Sun, Pihai Sun, Shuai Shi, Botuo Jiang, Jiahao Ma, Jiaxu Wang, Hao Cheng, Zhichao Liu, Yang Wang, Zheng Zhu, Guan Huang, Jian Tang, Qiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20217)  

**Abstract**: Humanoid robot technology is advancing rapidly, with manufacturers introducing diverse heterogeneous visual perception modules tailored to specific scenarios. Among various perception paradigms, occupancy-based representation has become widely recognized as particularly suitable for humanoid robots, as it provides both rich semantic and 3D geometric information essential for comprehensive environmental understanding. In this work, we present Humanoid Occupancy, a generalized multimodal occupancy perception system that integrates hardware and software components, data acquisition devices, and a dedicated annotation pipeline. Our framework employs advanced multi-modal fusion techniques to generate grid-based occupancy outputs encoding both occupancy status and semantic labels, thereby enabling holistic environmental understanding for downstream tasks such as task planning and navigation. To address the unique challenges of humanoid robots, we overcome issues such as kinematic interference and occlusion, and establish an effective sensor layout strategy. Furthermore, we have developed the first panoramic occupancy dataset specifically for humanoid robots, offering a valuable benchmark and resource for future research and development in this domain. The network architecture incorporates multi-modal feature fusion and temporal information integration to ensure robust perception. Overall, Humanoid Occupancy delivers effective environmental perception for humanoid robots and establishes a technical foundation for standardizing universal visual modules, paving the way for the widespread deployment of humanoid robots in complex real-world scenarios. 

**Abstract (ZH)**: humano形机器人占用感知技术：一种通用多模态占用感知系统及其应用 

---
# Color histogram equalization and fine-tuning to improve expression recognition of (partially occluded) faces on sign language datasets 

**Title (ZH)**: 基于颜色直方图均衡化和微调以提高手语图像中（部分遮挡的）面部表情识别 

**Authors**: Fabrizio Nunnari, Alakshendra Jyotsnaditya Ramkrishna Singh, Patrick Gebhard  

**Link**: [PDF](https://arxiv.org/pdf/2507.20197)  

**Abstract**: The goal of this investigation is to quantify to what extent computer vision methods can correctly classify facial expressions on a sign language dataset. We extend our experiments by recognizing expressions using only the upper or lower part of the face, which is needed to further investigate the difference in emotion manifestation between hearing and deaf subjects. To take into account the peculiar color profile of a dataset, our method introduces a color normalization stage based on histogram equalization and fine-tuning. The results show the ability to correctly recognize facial expressions with 83.8% mean sensitivity and very little variance (.042) among classes. Like for humans, recognition of expressions from the lower half of the face (79.6%) is higher than that from the upper half (77.9%). Noticeably, the classification accuracy from the upper half of the face is higher than human level. 

**Abstract (ZH)**: 本研究的目的是量化计算机视觉方法在手语数据集中正确分类面部表情的程度。我们通过仅识别面部上部或下部的表情来扩展我们的实验，以进一步调查 Hearing 和 Deaf 受试者在情绪表达方面的差异。为了考虑到数据集的特殊颜色特征，我们的方法引入了基于直方图均衡和微调的色彩归一化阶段。结果显示，正确识别面部表情的能力达到了 83.8% 的平均灵敏度，并且不同类别的方差很小（0.042）。如同人类一样，来自面部下部（79.6%）的表情识别准确性高于来自上部（77.9%）。值得注意的是，来自面部上部的表情分类准确性高于人类水平。 

---
# Partial Domain Adaptation via Importance Sampling-based Shift Correction 

**Title (ZH)**: 基于重要性采样偏差校正的部分领域适应 

**Authors**: Cheng-Jun Guo, Chuan-Xian Ren, You-Wei Luo, Xiao-Lin Xu, Hong Yan  

**Link**: [PDF](https://arxiv.org/pdf/2507.20191)  

**Abstract**: Partial domain adaptation (PDA) is a challenging task in real-world machine learning scenarios. It aims to transfer knowledge from a labeled source domain to a related unlabeled target domain, where the support set of the source label distribution subsumes the target one. Previous PDA works managed to correct the label distribution shift by weighting samples in the source domain. However, the simple reweighing technique cannot explore the latent structure and sufficiently use the labeled data, and then models are prone to over-fitting on the source domain. In this work, we propose a novel importance sampling-based shift correction (IS$^2$C) method, where new labeled data are sampled from a built sampling domain, whose label distribution is supposed to be the same as the target domain, to characterize the latent structure and enhance the generalization ability of the model. We provide theoretical guarantees for IS$^2$C by proving that the generalization error can be sufficiently dominated by IS$^2$C. In particular, by implementing sampling with the mixture distribution, the extent of shift between source and sampling domains can be connected to generalization error, which provides an interpretable way to build IS$^2$C. To improve knowledge transfer, an optimal transport-based independence criterion is proposed for conditional distribution alignment, where the computation of the criterion can be adjusted to reduce the complexity from $\mathcal{O}(n^3)$ to $\mathcal{O}(n^2)$ in realistic PDA scenarios. Extensive experiments on PDA benchmarks validate the theoretical results and demonstrate the effectiveness of our IS$^2$C over existing methods. 

**Abstract (ZH)**: 基于重要性采样的偏移纠正（IS$^2$C）方法 

---
# NeuroCLIP: A Multimodal Contrastive Learning Method for rTMS-treated Methamphetamine Addiction Analysis 

**Title (ZH)**: NeuroCLIP: 一种用于甲基苯丙胺成瘾经颅磁刺激治疗分析的多模态对比学习方法 

**Authors**: Chengkai Wang, Di Wu, Yunsheng Liao, Wenyao Zheng, Ziyi Zeng, Xurong Gao, Hemmings Wu, Zhoule Zhu, Jie Yang, Lihua Zhong, Weiwei Cheng, Yun-Hsuan Chen, Mohamad Sawan  

**Link**: [PDF](https://arxiv.org/pdf/2507.20189)  

**Abstract**: Methamphetamine dependence poses a significant global health challenge, yet its assessment and the evaluation of treatments like repetitive transcranial magnetic stimulation (rTMS) frequently depend on subjective self-reports, which may introduce uncertainties. While objective neuroimaging modalities such as electroencephalography (EEG) and functional near-infrared spectroscopy (fNIRS) offer alternatives, their individual limitations and the reliance on conventional, often hand-crafted, feature extraction can compromise the reliability of derived biomarkers. To overcome these limitations, we propose NeuroCLIP, a novel deep learning framework integrating simultaneously recorded EEG and fNIRS data through a progressive learning strategy. This approach offers a robust and trustworthy biomarker for methamphetamine addiction. Validation experiments show that NeuroCLIP significantly improves discriminative capabilities among the methamphetamine-dependent individuals and healthy controls compared to models using either EEG or only fNIRS alone. Furthermore, the proposed framework facilitates objective, brain-based evaluation of rTMS treatment efficacy, demonstrating measurable shifts in neural patterns towards healthy control profiles after treatment. Critically, we establish the trustworthiness of the multimodal data-driven biomarker by showing its strong correlation with psychometrically validated craving scores. These findings suggest that biomarker derived from EEG-fNIRS data via NeuroCLIP offers enhanced robustness and reliability over single-modality approaches, providing a valuable tool for addiction neuroscience research and potentially improving clinical assessments. 

**Abstract (ZH)**: 甲基苯丙胺依赖性构成一项重大的全球健康挑战，但其评估和治疗效果评估（如重复经颅磁刺激rTMS）通常依赖于主观自我报告，这可能引入不确定性。虽然电生理成像技术如脑电图（EEG）和功能性近红外光谱成像（fNIRS）提供了替代方案，但它们各自的局限性以及对传统、常手工构建的特征提取的依赖性可能损害衍生生物标志物的可靠性。为克服这些局限，我们提出了一种名为NeuroCLIP的新型深度学习框架，该框架通过渐进式学习策略整合同时记录的EEG和fNIRS数据。该方法提供了甲基苯丙胺依赖性的稳健且可信赖的生物标志物。验证实验表明，NeuroCLIP在区分甲基苯丙胺依赖个体和健康对照方面显著优于仅使用EEG或仅fNIRS的模型。此外，该框架促进了针对rTMS治疗效果的客观、基于大脑的评估，显示治疗后神经模式向健康对照模式发生了可测量的变化。关键地，我们通过证明其与心理测量学验证的渴求评分高度相关，建立了多模态数据驱动生物标志物的可靠性。这些发现表明，通过NeuroCLIP从EEG-fNIRS数据中提取的生物标志物相较于单模态方法具有更高的稳健性和可靠性，为其在成瘾神经科学研究中提供了有价值的工具，并有可能改善临床评估。 

---
# SGPO: Self-Generated Preference Optimization based on Self-Improver 

**Title (ZH)**: SGPO: 自生成偏好优化基于自我改进者 

**Authors**: Hyeonji Lee, Daejin Jo, Seohwan Yun, Sungwoong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.20181)  

**Abstract**: Large language models (LLMs), despite their extensive pretraining on diverse datasets, require effective alignment to human preferences for practical and reliable deployment. Conventional alignment methods typically employ off-policy learning and depend on human-annotated datasets, which limits their broad applicability and introduces distribution shift issues during training. To address these challenges, we propose Self-Generated Preference Optimization based on Self-Improver (SGPO), an innovative alignment framework that leverages an on-policy self-improving mechanism. Specifically, the improver refines responses from a policy model to self-generate preference data for direct preference optimization (DPO) of the policy model. Here, the improver and policy are unified into a single model, and in order to generate higher-quality preference data, this self-improver learns to make incremental yet discernible improvements to the current responses by referencing supervised fine-tuning outputs. Experimental results on AlpacaEval 2.0 and Arena-Hard show that the proposed SGPO significantly improves performance over DPO and baseline self-improving methods without using external preference data. 

**Abstract (ZH)**: 大规模语言模型（LLMs）尽管在多样化的数据集上进行了广泛的预训练，但在实际和可靠的部署中仍需要有效的对齐以符合人类偏好。传统的对齐方法通常依赖于离策学习并且依赖于人工标注的数据集，这限制了它们的广泛应用并在训练中引入了分布偏移问题。为了解决这些挑战，我们提出了基于Self-Improver的Self-Generated Preference Optimization（SGPO）这一创新的对齐框架，该框架利用了自改进机制。具体而言，改进模块通过优化政策模型的响应来自动生成偏好数据，直接用于对政策模型的偏好优化（DPO）。在此过程中，改进模块和政策被统一为一个模型，并通过参考监督微调输出来学习逐步改进当前响应，从而生成更高质量的偏好数据。实验结果表明，在AlpacaEval 2.0和Arena-Hard上的表现表明，提出的SGPO方法在不使用外部偏好数据的情况下显著优于DPO和基线自改进方法。 

---
# LRR-Bench: Left, Right or Rotate? Vision-Language models Still Struggle With Spatial Understanding Tasks 

**Title (ZH)**: LRR-Bench: 左、右还是旋转？视觉-语言模型在空间理解任务上仍然挣扎 

**Authors**: Fei Kong, Jinhao Duan, Kaidi Xu, Zhenhua Guo, Xiaofeng Zhu, Xiaoshuang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2507.20174)  

**Abstract**: Real-world applications, such as autonomous driving and humanoid robot manipulation, require precise spatial perception. However, it remains underexplored how Vision-Language Models (VLMs) recognize spatial relationships and perceive spatial movement. In this work, we introduce a spatial evaluation pipeline and construct a corresponding benchmark. Specifically, we categorize spatial understanding into two main types: absolute spatial understanding, which involves querying the absolute spatial position (e.g., left, right) of an object within an image, and 3D spatial understanding, which includes movement and rotation. Notably, our dataset is entirely synthetic, enabling the generation of test samples at a low cost while also preventing dataset contamination. We conduct experiments on multiple state-of-the-art VLMs and observe that there is significant room for improvement in their spatial understanding abilities. Explicitly, in our experiments, humans achieve near-perfect performance on all tasks, whereas current VLMs attain human-level performance only on the two simplest tasks. For the remaining tasks, the performance of VLMs is distinctly lower than that of humans. In fact, the best-performing Vision-Language Models even achieve near-zero scores on multiple tasks. The dataset and code are available on this https URL. 

**Abstract (ZH)**: 现实世界的应用，如自主驾驶和类人机器人操作，需要精确的空间感知。然而，VLMs如何识别空间关系和感知空间运动仍然未被充分探索。本文引入一个空间评估管道并构建相应的基准。具体而言，我们将空间理解分为两大类：绝对空间理解，涉及查询图像中物体的绝对空间位置（如左、右），以及三维空间理解，包括运动和旋转。值得注意的是，我们的数据集完全是合成的，这不仅降低了测试样本的生成成本，还防止了数据集的污染。我们在多个最先进的VLMs上进行了实验，发现它们的空间理解能力有很大的提升空间。具体而言，在我们的实验中，人类在所有任务上均接近完美表现，而当前的VLMs仅在两个最简单的任务上达到了人类水平的表现。对于剩余的任务，VLMs的表现明显低于人类。事实上，表现最佳的Vision-Language模型甚至在多个任务上获得了接近零的得分。数据集和代码可在此处访问：https://this-url.com。 

---
# High-Performance Parallel Optimization of the Fish School Behaviour on the Setonix Platform Using OpenMP 

**Title (ZH)**: 基于Setonix平台的鱼群行为高性能并行优化实现（使用OpenMP） 

**Authors**: Haitian Wang, Long Qin  

**Link**: [PDF](https://arxiv.org/pdf/2507.20173)  

**Abstract**: This paper presents an in-depth investigation into the high-performance parallel optimization of the Fish School Behaviour (FSB) algorithm on the Setonix supercomputing platform using the OpenMP framework. Given the increasing demand for enhanced computational capabilities for complex, large-scale calculations across diverse domains, there's an imperative need for optimized parallel algorithms and computing structures. The FSB algorithm, inspired by nature's social behavior patterns, provides an ideal platform for parallelization due to its iterative and computationally intensive nature. This study leverages the capabilities of the Setonix platform and the OpenMP framework to analyze various aspects of multi-threading, such as thread counts, scheduling strategies, and OpenMP constructs, aiming to discern patterns and strategies that can elevate program performance. Experiments were designed to rigorously test different configurations, and our results not only offer insights for parallel optimization of FSB on Setonix but also provide valuable references for other parallel computational research using OpenMP. Looking forward, other factors, such as cache behavior and thread scheduling strategies at micro and macro levels, hold potential for further exploration and optimization. 

**Abstract (ZH)**: 基于OpenMP框架的Setonix超级计算平台上的高性能Fish School Behaviour算法并行优化研究 

---
# ASNN: Learning to Suggest Neural Architectures from Performance Distributions 

**Title (ZH)**: ASNN: 从性能分布中学习建议神经网络架构 

**Authors**: Jinwook Hong  

**Link**: [PDF](https://arxiv.org/pdf/2507.20164)  

**Abstract**: The architecture of a neural network (NN) plays a critical role in determining its performance. However, there is no general closed-form function that maps between network structure and accuracy, making the process of architecture design largely heuristic or search-based. In this study, we propose the Architecture Suggesting Neural Network (ASNN), a model designed to learn the relationship between NN architecture and its test accuracy, and to suggest improved architectures accordingly. To train ASNN, we constructed datasets using TensorFlow-based models with varying numbers of layers and nodes. Experimental results were collected for both 2-layer and 3-layer architectures across a grid of configurations, each evaluated with 10 repeated trials to account for stochasticity. Accuracy values were treated as inputs, and architectural parameters as outputs. The trained ASNN was then used iteratively to predict architectures that yield higher performance. In both 2-layer and 3-layer cases, ASNN successfully suggested architectures that outperformed the best results found in the original training data. Repeated prediction and retraining cycles led to the discovery of architectures with improved mean test accuracies, demonstrating the model's capacity to generalize the performance-structure relationship. These results suggest that ASNN provides an efficient alternative to random search for architecture optimization, and offers a promising approach toward automating neural network design. "Parts of the manuscript, including text editing and expression refinement, were supported by OpenAI's ChatGPT. All content was reviewed and verified by the authors." 

**Abstract (ZH)**: 神经网络架构建议网络（ASNN）：学习架构与测试准确率关系并提出改进架构 

---
# Trust the Model: Compact VLMs as In-Context Judges for Image-Text Data Quality 

**Title (ZH)**: 信任模型：紧凑型VLM作为图像-文本数据质量的上下文法官 

**Authors**: Daulet Toibazar, Kesen Wang, Sherif Mohamed, Abdulaziz Al-Badawi, Abdulrahman Alfulayt, Pedro J. Moreno  

**Link**: [PDF](https://arxiv.org/pdf/2507.20156)  

**Abstract**: Vision-language models (VLMs) extend the conventional large language models by integrating visual data, enabling richer multimodal reasoning and significantly broadens the practical applications of AI. However, including visual inputs also brings new challenges in maintaining data quality. Empirical evidence consistently shows that carefully curated and representative training examples often yield superior results compared to simply increasing the quantity of data. Inspired by this observation, we introduce a streamlined data filtration framework that employs a compact VLM, fine-tuned on a high-quality image-caption annotated dataset. This model effectively evaluates and filters potential training samples based on caption and image quality and alignment. Unlike previous approaches, which typically add auxiliary filtration modules on top of existing full-scale VLMs, our method exclusively utilizes the inherent evaluative capability of a purpose-built small VLM. This strategy eliminates the need for extra modules and reduces training overhead. Our lightweight model efficiently filters out inaccurate, noisy web data, improving image-text alignment and caption linguistic fluency. Experimental results show that datasets underwent high-precision filtration using our compact VLM perform on par with, or even surpass, larger and noisier datasets gathered through high-volume web crawling. Thus, our method provides a lightweight yet robust solution for building high-quality vision-language training corpora. \\ \textbf{Availability and implementation:} Our compact VLM filtration model, training data, utility scripts, and Supplementary data (Appendices) are freely available at this https URL. 

**Abstract (ZH)**: 视觉语言模型的数据过滤框架：一种轻量级且高效的高质量视觉语言训练数据构建方法 

---
# Goal Alignment in LLM-Based User Simulators for Conversational AI 

**Title (ZH)**: LLM基于的用户模拟器中目标对齐在对话AI中的应用 

**Authors**: Shuhaib Mehri, Xiaocheng Yang, Takyoung Kim, Gokhan Tur, Shikib Mehri, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2507.20152)  

**Abstract**: User simulators are essential to conversational AI, enabling scalable agent development and evaluation through simulated interactions. While current Large Language Models (LLMs) have advanced user simulation capabilities, we reveal that they struggle to consistently demonstrate goal-oriented behavior across multi-turn conversations--a critical limitation that compromises their reliability in downstream applications. We introduce User Goal State Tracking (UGST), a novel framework that tracks user goal progression throughout conversations. Leveraging UGST, we present a three-stage methodology for developing user simulators that can autonomously track goal progression and reason to generate goal-aligned responses. Moreover, we establish comprehensive evaluation metrics for measuring goal alignment in user simulators, and demonstrate that our approach yields substantial improvements across two benchmarks (MultiWOZ 2.4 and {\tau}-Bench). Our contributions address a critical gap in conversational AI and establish UGST as an essential framework for developing goal-aligned user simulators. 

**Abstract (ZH)**: 用户模拟器是会话AI的关键组成部分，通过模拟交互 enables 扩展代理开发和评估。尽管当前的大规模语言模型（LLMs）具有先进的用户模拟能力，但我们发现它们在多轮对话中一致地表现出目标导向行为方面存在关键限制——这一限制阻碍了它们在下游应用中的可靠性。我们提出了一种新型框架用户目标状态跟踪（UGST），该框架在整个对话过程中跟踪用户目标进展。利用UGST，我们提出了一个三阶段方法用于开发能够自主跟踪目标进展并推理生成目标导向响应的用户模拟器。此外，我们为评估用户模拟器的目标一致性建立了全面的评价指标，并证明了我们方法在两个基准（MultiWOZ 2.4和τ-Bench）上取得了显著改进。我们的贡献填补了会话AI中的关键空白，并将UGST确立为开发目标导向用户模拟器的必不可少的框架。 

---
# Multi-Agent Interactive Question Generation Framework for Long Document Understanding 

**Title (ZH)**: 长文档理解的多agent互动式问题生成框架 

**Authors**: Kesen Wang, Daulet Toibazar, Abdulrahman Alfulayt, Abdulaziz S. Albadawi, Ranya A. Alkahtani, Asma A. Ibrahim, Haneen A. Alhomoud, Sherif Mohamed, Pedro J. Moreno  

**Link**: [PDF](https://arxiv.org/pdf/2507.20145)  

**Abstract**: Document Understanding (DU) in long-contextual scenarios with complex layouts remains a significant challenge in vision-language research. Although Large Vision-Language Models (LVLMs) excel at short-context DU tasks, their performance declines in long-context settings. A key limitation is the scarcity of fine-grained training data, particularly for low-resource languages such as Arabic. Existing state-of-the-art techniques rely heavily on human annotation, which is costly and inefficient. We propose a fully automated, multi-agent interactive framework to generate long-context questions efficiently. Our approach efficiently generates high-quality single- and multi-page questions for extensive English and Arabic documents, covering hundreds of pages across diverse domains. This facilitates the development of LVLMs with enhanced long-context understanding ability. Experimental results in this work have shown that our generated English and Arabic questions (\textbf{AraEngLongBench}) are quite challenging to major open- and close-source LVLMs. The code and data proposed in this work can be found in this https URL. Sample Question and Answer (QA) pairs and structured system prompts can be found in the Appendix. 

**Abstract (ZH)**: 在复杂布局下的长文字段理解（DU）研究仍是在视觉语言领域的一大挑战。现有的大规模视觉语言模型（LVLMs）在短文字段理解任务上表现出色，但在长文字段环境中性能下降。一个主要限制是高质量训练数据的稀缺，特别是对于如阿拉伯语这样的低资源语言。现有最先进的技术严重依赖人工标注，这既昂贵又低效。我们提出了一种全自动的多智能体互动框架，以高效生成长文字段问题。我们的方法能够高效生成高质量的单页和多页问题，涵盖数百页横跨多个领域的广泛英文和阿拉伯文文档。这有助于开发具有增强长文字段理解能力的大规模视觉语言模型。实验结果表明，我们生成的英文和阿拉伯文问题（AraEngLongBench）对主要的开源和闭源视觉语言模型颇具挑战性。本文中提出的方法代码和数据可在以下链接找到：this https URL。样本问题和答案（QA）对以及结构化的系统提示可以在附录中找到。 

---
# Awesome-OL: An Extensible Toolkit for Online Learning 

**Title (ZH)**: Awesome-OL：一种可扩展的在线学习工具包 

**Authors**: Zeyi Liu, Songqiao Hu, Pengyu Han, Jiaming Liu, Xiao He  

**Link**: [PDF](https://arxiv.org/pdf/2507.20144)  

**Abstract**: In recent years, online learning has attracted increasing attention due to its adaptive capability to process streaming and non-stationary data. To facilitate algorithm development and practical deployment in this area, we introduce Awesome-OL, an extensible Python toolkit tailored for online learning research. Awesome-OL integrates state-of-the-art algorithm, which provides a unified framework for reproducible comparisons, curated benchmark datasets, and multi-modal visualization. Built upon the scikit-multiflow open-source infrastructure, Awesome-OL emphasizes user-friendly interactions without compromising research flexibility or extensibility. The source code is publicly available at: this https URL. 

**Abstract (ZH)**: 近年来，由于其处理流式和非稳态数据的适应能力，在线学习吸引了越来越多的关注。为促进该领域的算法开发和实际部署，我们介绍了Awesome-OL，这是一个针对在线学习研究的可扩展Python工具包。Awesome-OL集成了最先进的算法，提供了一个统一框架以实现可重复比较、精心策划的标准数据集以及多模态可视化。基于scikit-multiflow开源基础设施，Awesome-OL强调用户友好的交互，同时不牺牲研究的灵活性或可扩展性。源代码可在以下网址获取：this https URL。 

---
# Do Not Mimic My Voice: Speaker Identity Unlearning for Zero-Shot Text-to-Speech 

**Title (ZH)**: 不要模仿我的声音：零样本文本到语音的说话人身份遗忘 

**Authors**: Taesoo Kim, Jinju Kim, Dongchan Kim, Jong Hwan Ko, Gyeong-Moon Park  

**Link**: [PDF](https://arxiv.org/pdf/2507.20140)  

**Abstract**: The rapid advancement of Zero-Shot Text-to-Speech (ZS-TTS) technology has enabled high-fidelity voice synthesis from minimal audio cues, raising significant privacy and ethical concerns. Despite the threats to voice privacy, research to selectively remove the knowledge to replicate unwanted individual voices from pre-trained model parameters has not been explored. In this paper, we address the new challenge of speaker identity unlearning for ZS-TTS systems. To meet this goal, we propose the first machine unlearning frameworks for ZS-TTS, especially Teacher-Guided Unlearning (TGU), designed to ensure the model forgets designated speaker identities while retaining its ability to generate accurate speech for other speakers. Our proposed methods incorporate randomness to prevent consistent replication of forget speakers' voices, assuring unlearned identities remain untraceable. Additionally, we propose a new evaluation metric, speaker-Zero Retrain Forgetting (spk-ZRF). This assesses the model's ability to disregard prompts associated with forgotten speakers, effectively neutralizing its knowledge of these voices. The experiments conducted on the state-of-the-art model demonstrate that TGU prevents the model from replicating forget speakers' voices while maintaining high quality for other speakers. The demo is available at this https URL 

**Abstract (ZH)**: 零样本文本到语音系统中的说话人身份遗忘技术研究 

---
# Multi-Stage Verification-Centric Framework for Mitigating Hallucination in Multi-Modal RAG 

**Title (ZH)**: 面向多模态RAG中幻觉缓解的多阶段验证中心框架 

**Authors**: Baiyu Chen, Wilson Wongso, Xiaoqian Hu, Yue Tan, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2507.20136)  

**Abstract**: This paper presents the technical solution developed by team CRUISE for the KDD Cup 2025 Meta Comprehensive RAG Benchmark for Multi-modal, Multi-turn (CRAG-MM) challenge. The challenge aims to address a critical limitation of modern Vision Language Models (VLMs): their propensity to hallucinate, especially when faced with egocentric imagery, long-tail entities, and complex, multi-hop questions. This issue is particularly problematic in real-world applications where users pose fact-seeking queries that demand high factual accuracy across diverse modalities. To tackle this, we propose a robust, multi-stage framework that prioritizes factual accuracy and truthfulness over completeness. Our solution integrates a lightweight query router for efficiency, a query-aware retrieval and summarization pipeline, a dual-pathways generation and a post-hoc verification. This conservative strategy is designed to minimize hallucinations, which incur a severe penalty in the competition's scoring metric. Our approach achieved 3rd place in Task 1, demonstrating the effectiveness of prioritizing answer reliability in complex multi-modal RAG systems. Our implementation is available at this https URL . 

**Abstract (ZH)**: 本文介绍了CRUISE团队为2025 KDD Cup Meta Comprehensive RAG Benchmark CRAG-MM挑战开发的技术解决方案。该挑战旨在解决现代视觉语言模型（VLMs）的一个关键局限性：在面对主观图像、长尾实体和复杂多跳问题时的倾向性虚构。这个问题在实际应用中尤为关键，用户提出事实求证查询，要求在多种模态中保持高度的事实准确性。为此，我们提出了一种稳健的多阶段框架，优先考虑事实准确性与真实性而非完整性。我们的解决方案包括一个轻量级查询路由器以提高效率，一个查询感知的检索和总结流水线，以及一种双重路径生成和事后验证。这一谨慎策略旨在最大限度地减少虚构的发生，在竞赛评分标准中虚构会遭受严重惩罚。我们的方法在任务1中获得第3名，展示了在复杂多模态RAG系统中优先保证答案可靠性的有效性。我们的实现可以在以下网址获取：这个 https URL 。 

---
# Sem-DPO: Mitigating Semantic Inconsistency in Preference Optimization for Prompt Engineering 

**Title (ZH)**: Sem-DPO：在提示工程中的偏好优化中缓解语义不一致性 

**Authors**: Anas Mohamed, Azal Ahmad Khan, Xinran Wang, Ahmad Faraz Khan, Shuwen Ge, Saman Bahzad Khan, Ayaan Ahmad, Ali Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2507.20133)  

**Abstract**: Generative AI can now synthesize strikingly realistic images from text, yet output quality remains highly sensitive to how prompts are phrased. Direct Preference Optimization (DPO) offers a lightweight, off-policy alternative to RL for automatic prompt engineering, but its token-level regularization leaves semantic inconsistency unchecked as prompts that win higher preference scores can still drift away from the user's intended meaning.
We introduce Sem-DPO, a variant of DPO that preserves semantic consistency yet retains its simplicity and efficiency. Sem-DPO scales the DPO loss by an exponential weight proportional to the cosine distance between the original prompt and winning candidate in embedding space, softly down-weighting training signals that would otherwise reward semantically mismatched prompts. We provide the first analytical bound on semantic drift for preference-tuned prompt generators, showing that Sem-DPO keeps learned prompts within a provably bounded neighborhood of the original text. On three standard text-to-image prompt-optimization benchmarks and two language models, Sem-DPO achieves 8-12% higher CLIP similarity and 5-9% higher human-preference scores (HPSv2.1, PickScore) than DPO, while also outperforming state-of-the-art baselines. These findings suggest that strong flat baselines augmented with semantic weighting should become the new standard for prompt-optimization studies and lay the groundwork for broader, semantics-aware preference optimization in language models. 

**Abstract (ZH)**: Sem-DPO: 保持语义一致性的同时简化直接偏好优化 

---
# Aggregation-aware MLP: An Unsupervised Approach for Graph Message-passing 

**Title (ZH)**: 聚合意识MLP：图消息传递的无监督方法 

**Authors**: Xuanting Xie, Bingheng Li, Erlin Pan, Zhao Kang, Wenyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.20127)  

**Abstract**: Graph Neural Networks (GNNs) have become a dominant approach to learning graph representations, primarily because of their message-passing mechanisms. However, GNNs typically adopt a fixed aggregator function such as Mean, Max, or Sum without principled reasoning behind the selection. This rigidity, especially in the presence of heterophily, often leads to poor, problem dependent performance. Although some attempts address this by designing more sophisticated aggregation functions, these methods tend to rely heavily on labeled data, which is often scarce in real-world tasks. In this work, we propose a novel unsupervised framework, "Aggregation-aware Multilayer Perceptron" (AMLP), which shifts the paradigm from directly crafting aggregation functions to making MLP adaptive to aggregation. Our lightweight approach consists of two key steps: First, we utilize a graph reconstruction method that facilitates high-order grouping effects, and second, we employ a single-layer network to encode varying degrees of heterophily, thereby improving the capacity and applicability of the model. Extensive experiments on node clustering and classification demonstrate the superior performance of AMLP, highlighting its potential for diverse graph learning scenarios. 

**Abstract (ZH)**: 基于聚合感知的多层感知机（AMLP）：一种无监督框架 

---
# Iterative Pretraining Framework for Interatomic Potentials 

**Title (ZH)**: 迭代预训练框架用于原子势能 

**Authors**: Taoyong Cui, Zhongyao Wang, Dongzhan Zhou, Yuqiang Li, Lei Bai, Wanli Ouyang, Mao Su, Shufei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20118)  

**Abstract**: Machine learning interatomic potentials (MLIPs) enable efficient molecular dynamics (MD) simulations with ab initio accuracy and have been applied across various domains in physical science. However, their performance often relies on large-scale labeled training data. While existing pretraining strategies can improve model performance, they often suffer from a mismatch between the objectives of pretraining and downstream tasks or rely on extensive labeled datasets and increasingly complex architectures to achieve broad generalization. To address these challenges, we propose Iterative Pretraining for Interatomic Potentials (IPIP), a framework designed to iteratively improve the predictive performance of MLIP models. IPIP incorporates a forgetting mechanism to prevent iterative training from converging to suboptimal local minima. Unlike general-purpose foundation models, which frequently underperform on specialized tasks due to a trade-off between generality and system-specific accuracy, IPIP achieves higher accuracy and efficiency using lightweight architectures. Compared to general-purpose force fields, this approach achieves over 80% reduction in prediction error and up to 4x speedup in the challenging Mo-S-O system, enabling fast and accurate simulations. 

**Abstract (ZH)**: 迭代原子势前训练（IPIP）：用于迭代改进机器学习原子势模型预测性能的框架 

---
# Packet-Level DDoS Data Augmentation Using Dual-Stream Temporal-Field Diffusion 

**Title (ZH)**: 面向包级的分布式拒绝服务数据增强方法：双流时域场扩散 

**Authors**: Gongli Xi, Ye Tian, Yannan Hu, Yuchao Zhang, Yapeng Niu, Xiangyang Gong  

**Link**: [PDF](https://arxiv.org/pdf/2507.20115)  

**Abstract**: In response to Distributed Denial of Service (DDoS) attacks, recent research efforts increasingly rely on Machine Learning (ML)-based solutions, whose effectiveness largely depends on the quality of labeled training datasets. To address the scarcity of such datasets, data augmentation with synthetic traces is often employed. However, current synthetic trace generation methods struggle to capture the complex temporal patterns and spatial distributions exhibited in emerging DDoS attacks. This results in insufficient resemblance to real traces and unsatisfied detection accuracy when applied to ML tasks. In this paper, we propose Dual-Stream Temporal-Field Diffusion (DSTF-Diffusion), a multi-view, multi-stream network traffic generative model based on diffusion models, featuring two main streams: The field stream utilizes spatial mapping to bridge network data characteristics with pre-trained realms of stable diffusion models, effectively translating complex network interactions into formats that stable diffusion can process, while the spatial stream adopts a dynamic temporal modeling approach, meticulously capturing the intrinsic temporal patterns of network traffic. Extensive experiments demonstrate that data generated by our model exhibits higher statistical similarity to originals compared to current state-of-the-art solutions, and enhance performances on a wide range of downstream tasks. 

**Abstract (ZH)**: 基于扩散模型的双流时空场扩散网络流量生成模型（DSTF-Diffusion）：应对分布式拒绝服务攻击的合成踪迹生成方法 

---
# Online Learning with Probing for Sequential User-Centric Selection 

**Title (ZH)**: 基于探针的在线学习auty： sequential 用户中心选择 

**Authors**: Tianyi Xu, Yiting Chen, Henger Li, Zheyong Bian, Emiliano Dall'Anese, Zizhan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.20112)  

**Abstract**: We formalize sequential decision-making with information acquisition as the probing-augmented user-centric selection (PUCS) framework, where a learner first probes a subset of arms to obtain side information on resources and rewards, and then assigns $K$ plays to $M$ arms. PUCS covers applications such as ridesharing, wireless scheduling, and content recommendation, in which both resources and payoffs are initially unknown and probing is costly. For the offline setting with known distributions, we present a greedy probing algorithm with a constant-factor approximation guarantee $\zeta = (e-1)/(2e-1)$. For the online setting with unknown distributions, we introduce OLPA, a stochastic combinatorial bandit algorithm that achieves a regret bound $\mathcal{O}(\sqrt{T} + \ln^{2} T)$. We also prove a lower bound $\Omega(\sqrt{T})$, showing that the upper bound is tight up to logarithmic factors. Experiments on real-world data demonstrate the effectiveness of our solutions. 

**Abstract (ZH)**: 我们将信息获取增强的用户为中心的选择（PUCS）框架形式化为序贯决策问题，其中学习者首先探测一部分臂以获取资源和奖励的侧信息，然后将$K$次播放分配给$M$个臂。PUCS涵盖如 ridesharing、无线调度和内容推荐等应用，在这些应用中，资源和收益最初未知且探测成本高。对于已知分布的离线设置，我们提出了一种贪婪探测算法，其近似保证为常数因子$\zeta = (e-1)/(2e-1)$。对于未知分布的在线设置，我们引入了OLPA（在线组合-bandit算法），其后悔界为$\mathcal{O}(\sqrt{T} + \ln^{2} T)$，并证明了一个下界$\Omega(\sqrt{T})$，显示了上界的紧致性（至对数因子）。实验结果显示了我们方法的有效性。 

---
# AI-Driven Generation of Old English: A Framework for Low-Resource Languages 

**Title (ZH)**: AI驱动的古英语生成：低资源语言的一种框架 

**Authors**: Rodrigo Gabriel Salazar Alva, Matías Nuñez, Cristian López, Javier Martín Arista  

**Link**: [PDF](https://arxiv.org/pdf/2507.20111)  

**Abstract**: Preserving ancient languages is essential for understanding humanity's cultural and linguistic heritage, yet Old English remains critically under-resourced, limiting its accessibility to modern natural language processing (NLP) techniques. We present a scalable framework that uses advanced large language models (LLMs) to generate high-quality Old English texts, addressing this gap. Our approach combines parameter-efficient fine-tuning (Low-Rank Adaptation, LoRA), data augmentation via backtranslation, and a dual-agent pipeline that separates the tasks of content generation (in English) and translation (into Old English). Evaluation with automated metrics (BLEU, METEOR, and CHRF) shows significant improvements over baseline models, with BLEU scores increasing from 26 to over 65 for English-to-Old English translation. Expert human assessment also confirms high grammatical accuracy and stylistic fidelity in the generated texts. Beyond expanding the Old English corpus, our method offers a practical blueprint for revitalizing other endangered languages, effectively uniting AI innovation with the goals of cultural preservation. 

**Abstract (ZH)**: 保护古代语言是理解人类文化与语言遗产的关键，然而古英语仍严重缺乏资源，限制了其在现代自然语言处理技术中的应用。我们提出了一种可扩展的框架，利用先进的大型语言模型生成高质量的古英语文本，填补这一空白。我们的方法结合了参数高效微调（低秩适应，LoRA）、通过后翻译的数据增强以及一个分离内容生成（英文）和翻译（古英语）任务的双重代理管道。使用自动评估指标（BLEU、METEOR和CHRF）的评估显示，与基线模型相比，古英语翻译的BLEU分数从26显著提高到超过65。专家的人类评估也证实了生成文本在语法准确性和风格保真度方面的高质量。除了扩大古英语语料库外，该方法还提供了一种实用的蓝图，用于复兴其他濒危语言，有效地将人工智能创新与文化保护的目标结合起来。 

---
# NeuroVoxel-LM: Language-Aligned 3D Perception via Dynamic Voxelization and Meta-Embedding 

**Title (ZH)**: NeuroVoxel-LM: 语言对齐的动态体素化与元嵌入三维感知 

**Authors**: Shiyu Liu, Lianlei Shan  

**Link**: [PDF](https://arxiv.org/pdf/2507.20110)  

**Abstract**: Recent breakthroughs in Visual Language Models (VLMs) and Multimodal Large Language Models (MLLMs) have significantly advanced 3D scene perception towards language-driven cognition. However, existing 3D language models struggle with sparse, large-scale point clouds due to slow feature extraction and limited representation accuracy. To address these challenges, we propose NeuroVoxel-LM, a novel framework that integrates Neural Radiance Fields (NeRF) with dynamic resolution voxelization and lightweight meta-embedding. Specifically, we introduce a Dynamic Resolution Multiscale Voxelization (DR-MSV) technique that adaptively adjusts voxel granularity based on geometric and structural complexity, reducing computational cost while preserving reconstruction fidelity. In addition, we propose the Token-level Adaptive Pooling for Lightweight Meta-Embedding (TAP-LME) mechanism, which enhances semantic representation through attention-based weighting and residual fusion. Experimental results demonstrate that DR-MSV significantly improves point cloud feature extraction efficiency and accuracy, while TAP-LME outperforms conventional max-pooling in capturing fine-grained semantics from NeRF weights. 

**Abstract (ZH)**: 近期视觉语言模型（VLMs）和多模态大型语言模型（MLLMs）的突破性进展极大地推动了3D场景感知向语言驱动的认知发展。然而，现有3D语言模型在处理稀疏的大规模点云时遇到困难，主要由于特征提取速度慢和表示精度有限。为了解决这些问题，我们提出了NeuroVoxel-LM，该框架将神经辐射场（NeRF）与动态分辨率体素化和轻量级元嵌入相结合。具体而言，我们引入了一种动态分辨率多尺度体素化（DR-MSV）技术，该技术根据几何和结构复杂性自适应调整体素粒度，从而在保持重建保真度的同时降低计算成本。此外，我们提出了基于注意力加权和残差融合的Token级自适应聚类（TAP-LME）机制，以增强语义表示。实验结果表明，DR-MSV显著提高了点云特征提取的效率和准确性，而TAP-LME在捕捉NeRF权重中的细粒度语义方面优于传统的最大池化方法。 

---
# Learning to Align Human Code Preferences 

**Title (ZH)**: 学习对齐人类代码偏好 

**Authors**: Xin Yin, Chao Ni, Liushan Chen, Xiaohu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20109)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable potential in automating software development tasks. While recent advances leverage Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) to align models with human preferences, the optimal training strategy remains unclear across diverse code preference scenarios. This paper systematically investigates the roles of SFT and DPO in aligning LLMs with different code preferences. Through both theoretical analysis and empirical observation, we hypothesize that SFT excels in scenarios with objectively verifiable optimal solutions, while applying SFT followed by DPO (S&D) enables models to explore superior solutions in scenarios without objectively verifiable optimal solutions. Based on the analysis and experimental evidence, we propose Adaptive Preference Optimization (APO), a dynamic integration approach that adaptively amplifies preferred responses, suppresses dispreferred ones, and encourages exploration of potentially superior solutions during training. Extensive experiments across six representative code preference tasks validate our theoretical hypotheses and demonstrate that APO consistently matches or surpasses the performance of existing SFT and S&D strategies. Our work provides both theoretical foundations and practical guidance for selecting appropriate training strategies in different code preference alignment scenarios. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自动化软件开发任务方面展现了显著潜力。尽管最近的进步利用了监督微调（SFT）和直接偏好优化（DPO）来使模型与人类偏好相一致，但在不同的代码偏好场景中，最优训练策略仍然不清楚。本文系统地探讨了SFT和DPO在使LLMs与不同代码偏好相一致中的角色。通过理论分析和实证观察，我们假设SFT在客观可验证的最优解场景中表现出色，而在没有客观可验证的最优解场景中，先进行SFT再进行DPO（S&D）能够使模型探索更优秀的解。基于分析和实验证据，我们提出了自适应偏好优化（APO），该方法动态地放大偏好响应、抑制不偏好响应，并在训练过程中鼓励探索潜在的更优秀解。在六项代表性的代码偏好任务上的广泛实验验证了我们的理论假设，并证明APO在所有任务上都能一致地匹配或超越现有SFT和S&D策略的性能。我们的工作为不同代码偏好对齐场景下选择合适的训练策略提供了理论基础和实践指导。 

---
# EcoTransformer: Attention without Multiplication 

**Title (ZH)**: EcoTransformer: 注意力机制 without 乘法 

**Authors**: Xin Gao, Xingming Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.20096)  

**Abstract**: The Transformer, with its scaled dot-product attention mechanism, has become a foundational architecture in modern AI. However, this mechanism is computationally intensive and incurs substantial energy costs. We propose a new Transformer architecture EcoTransformer, in which the output context vector is constructed as the convolution of the values using a Laplacian kernel, where the distances are measured by the L1 metric between the queries and keys. Compared to dot-product based attention, the new attention score calculation is free of matrix multiplication. It performs on par with, or even surpasses, scaled dot-product attention in NLP, bioinformatics, and vision tasks, while consuming significantly less energy. 

**Abstract (ZH)**: EcoTransformer：基于拉普拉斯核和L1度量的新Transformer架构 

---
# Local Prompt Adaptation for Style-Consistent Multi-Object Generation in Diffusion Models 

**Title (ZH)**: 局部提示适配以实现风格一致的多对象生成在扩散模型中 

**Authors**: Ankit Sanjyal  

**Link**: [PDF](https://arxiv.org/pdf/2507.20094)  

**Abstract**: Diffusion models have become a powerful backbone for text-to-image generation, enabling users to synthesize high-quality visuals from natural language prompts. However, they often struggle with complex prompts involving multiple objects and global or local style specifications. In such cases, the generated scenes tend to lack style uniformity and spatial coherence, limiting their utility in creative and controllable content generation. In this paper, we propose a simple, training-free architectural method called Local Prompt Adaptation (LPA). Our method decomposes the prompt into content and style tokens, and injects them selectively into the U-Net's attention layers at different stages. By conditioning object tokens early and style tokens later in the generation process, LPA enhances both layout control and stylistic consistency. We evaluate our method on a custom benchmark of 50 style-rich prompts across five categories and compare against strong baselines including Composer, MultiDiffusion, Attend-and-Excite, LoRA, and SDXL. Our approach outperforms prior work on both CLIP score and style consistency metrics, offering a new direction for controllable, expressive diffusion-based generation. 

**Abstract (ZH)**: 局部提示适应（LPA）在文本到图像生成中的应用 

---
# RAG in the Wild: On the (In)effectiveness of LLMs with Mixture-of-Knowledge Retrieval Augmentation 

**Title (ZH)**: RAG在野外：混合知识检索增强的大型语言模型效用研究 

**Authors**: Ran Xu, Yuchen Zhuang, Yue Yu, Haoyu Wang, Wenqi Shi, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20059)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by integrating external knowledge retrieved at inference time. While RAG demonstrates strong performance on benchmarks largely derived from general-domain corpora like Wikipedia, its effectiveness under realistic, diverse retrieval scenarios remains underexplored. We evaluated RAG systems using MassiveDS, a large-scale datastore with mixture of knowledge, and identified critical limitations: retrieval mainly benefits smaller models, rerankers add minimal value, and no single retrieval source consistently excels. Moreover, current LLMs struggle to route queries across heterogeneous knowledge sources. These findings highlight the need for adaptive retrieval strategies before deploying RAG in real-world settings. Our code and data can be found at this https URL. 

**Abstract (ZH)**: 检索增强生成（RAG）通过在推理时集成外部知识来增强大型语言模型（LLMs）。尽管RAG在大量源自通用领域语料库（如维基百科）的基准测试中表现出色，但其在现实多样检索场景下的有效性尚未充分探索。我们使用MassiveDS大规模知识存储库进行了RAG系统评估，并发现关键限制：检索主要有利于小型模型，再排序器增加的价值有限，也没有单一的检索来源始终表现出色。此外，当前的LLMs难以跨异构知识源路由查询。这些发现强调，在实际应用中部署RAG之前需要采用适应性检索策略。我们的代码和数据可在以下网址找到。 

---
# FaRMamba: Frequency-based learning and Reconstruction aided Mamba for Medical Segmentation 

**Title (ZH)**: 基于频率学习和重建辅助的Mamba医疗分割方法 

**Authors**: Ze Rong, ZiYue Zhao, Zhaoxin Wang, Lei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.20056)  

**Abstract**: Accurate medical image segmentation remains challenging due to blurred lesion boundaries (LBA), loss of high-frequency details (LHD), and difficulty in modeling long-range anatomical structures (DC-LRSS). Vision Mamba employs one-dimensional causal state-space recurrence to efficiently model global dependencies, thereby substantially mitigating DC-LRSS. However, its patch tokenization and 1D serialization disrupt local pixel adjacency and impose a low-pass filtering effect, resulting in Local High-frequency Information Capture Deficiency (LHICD) and two-dimensional Spatial Structure Degradation (2D-SSD), which in turn exacerbate LBA and LHD. In this work, we propose FaRMamba, a novel extension that explicitly addresses LHICD and 2D-SSD through two complementary modules. A Multi-Scale Frequency Transform Module (MSFM) restores attenuated high-frequency cues by isolating and reconstructing multi-band spectra via wavelet, cosine, and Fourier transforms. A Self-Supervised Reconstruction Auxiliary Encoder (SSRAE) enforces pixel-level reconstruction on the shared Mamba encoder to recover full 2D spatial correlations, enhancing both fine textures and global context. Extensive evaluations on CAMUS echocardiography, MRI-based Mouse-cochlea, and Kvasir-Seg endoscopy demonstrate that FaRMamba consistently outperforms competitive CNN-Transformer hybrids and existing Mamba variants, delivering superior boundary accuracy, detail preservation, and global coherence without prohibitive computational overhead. This work provides a flexible frequency-aware framework for future segmentation models that directly mitigates core challenges in medical imaging. 

**Abstract (ZH)**: 准确的医学图像分割仍面临挑战：模糊的病灶边界（LBA）、高频细节丢失（LHD）以及长距离解剖结构建模困难（DC-LRSS）。Vision Mamba通过一维因果状态空间递归有效地建模全局依赖性，从而显著减轻DC-LRSS。然而，其片段化token化和一维序列化破坏了局部像素邻接关系，并施加了低通滤波效应，导致局部高频信息捕获不足（LHICD）和二维空间结构降解（2D-SSD），进而加剧了LBA和LHD。本文提出FaRMamba，一种新颖的扩展，通过两个互补模块显式解决LHICD和2D-SSD。多尺度频率变换模块（MSFM）通过小波、余弦和傅里叶变换隔离和重构多带谱，恢复衰减的高频线索。自我监督重建辅助编码器（SSRAE）在共享的Mamba编码器上施加像素级重建，以恢复完整的二维空间相关性，增强精细纹理和全局上下文。在CAMUS超声心动图、MRI基鼠耳蜗和Kvasir-Seg内窥镜上的广泛评估表明，FaRMamba在边界准确性、细节保留和全局连贯性方面均优于竞争性的CNN-Transformer混合模型和现有Mamba变体，且不增加显著的计算开销。本文提供了一种灵活的频率感知框架，可为未来的分割模型直接减轻医学影像中的核心挑战。 

---
# Irredundant k-Fold Cross-Validation 

**Title (ZH)**: 不可约简的k折交叉验证 

**Authors**: Jesus S. Aguilar-Ruiz  

**Link**: [PDF](https://arxiv.org/pdf/2507.20048)  

**Abstract**: In traditional k-fold cross-validation, each instance is used ($k\!-\!1$) times for training and once for testing, leading to redundancy that lets many instances disproportionately influence the learning phase. We introduce Irredundant $k$--fold cross-validation, a novel method that guarantees each instance is used exactly once for training and once for testing across the entire validation procedure. This approach ensures a more balanced utilization of the dataset, mitigates overfitting due to instance repetition, and enables sharper distinctions in comparative model analysis. The method preserves stratification and remains model-agnostic, i.e., compatible with any classifier. Experimental results demonstrate that it delivers consistent performance estimates across diverse datasets --comparable to $k$--fold cross-validation-- while providing less optimistic variance estimates because training partitions are non-overlapping, and significantly reducing the overall computational cost. 

**Abstract (ZH)**: 不可重复的k折交叉验证 

---
# TAPS : Frustratingly Simple Test Time Active Learning for VLMs 

**Title (ZH)**: TAPS : 极其简单的测试时主动学习方法 for VLMs 

**Authors**: Dhruv Sarkar, Aprameyo Chakrabartty, Bibhudatta Bhanja  

**Link**: [PDF](https://arxiv.org/pdf/2507.20028)  

**Abstract**: Test-Time Optimization enables models to adapt to new data during inference by updating parameters on-the-fly. Recent advances in Vision-Language Models (VLMs) have explored learning prompts at test time to improve performance in downstream tasks. In this work, we extend this idea by addressing a more general and practical challenge: Can we effectively utilize an oracle in a continuous data stream where only one sample is available at a time, requiring an immediate query decision while respecting latency and memory constraints? To tackle this, we propose a novel Test-Time Active Learning (TTAL) framework that adaptively queries uncertain samples and updates prompts dynamically. Unlike prior methods that assume batched data or multiple gradient updates, our approach operates in a real-time streaming scenario with a single test sample per step. We introduce a dynamically adjusted entropy threshold for active querying, a class-balanced replacement strategy for memory efficiency, and a class-aware distribution alignment technique to enhance adaptation. The design choices are justified using careful theoretical analysis. Extensive experiments across 10 cross-dataset transfer benchmarks and 4 domain generalization datasets demonstrate consistent improvements over state-of-the-art methods while maintaining reasonable latency and memory overhead. Our framework provides a practical and effective solution for real-world deployment in safety-critical applications such as autonomous systems and medical diagnostics. 

**Abstract (ZH)**: 基于测试时优化的测试时主动学习框架使得模型能够在推理过程中适应新数据并通过即刻更新参数来适应变化。通过在连续数据流中仅利用单个样本进行即时查询决策，我们提出了一种新颖的测试时主动学习（TTAL）框架，以有效利用先验知识并动态更新提示。该框架适用于单样本实时流场景，不同于以前假设批量数据或多次梯度更新的方法。我们引入了动态调整的熵阈值进行主动查询，以提高内存效率的类平衡替换策略，以及增强适应性的类觉察分布对齐技术。理论分析表明这些设计选择是合理的。实验结果表明，该框架在10个跨数据集迁移基准和4个领域泛化数据集上相较于最先进的方法具有持续改进，并且保持了合理的延迟和内存开销。我们的框架为自主系统和医疗诊断等关键应用的实际部署提供了一种实用而有效的方法。 

---
# When Engineering Outruns Intelligence: A Re-evaluation of Instruction-Guided Navigation 

**Title (ZH)**: 当工程超越智能：指令引导导航的重新评估 

**Authors**: Matin Aghaei, Mohammad Ali Alomrani, Yingxue Zhang, Mahdi Biparva  

**Link**: [PDF](https://arxiv.org/pdf/2507.20021)  

**Abstract**: Large language models (LLMs) are often credited with recent leaps in ObjectGoal Navigation, yet the extent to which they improve planning remains unclear. We revisit this question on the HM3D-v1 validation split. First, we strip InstructNav of its Dynamic Chain-of-Navigation prompt, open-vocabulary GLEE detector and Intuition saliency map, and replace them with a simple Distance-Weighted Frontier Explorer (DWFE). This geometry-only heuristic raises Success from 58.0% to 61.1% and lifts SPL from 20.9% to 36.0% over 2 000 validation episodes, outperforming all previous training-free baselines. Second, we add a lightweight language prior (SHF); on a 200-episode subset this yields a further +2% Success and +0.9% SPL while shortening paths by five steps on average. Qualitative trajectories confirm the trend: InstructNav back-tracks and times-out, DWFE reaches the goal after a few islands, and SHF follows an almost straight route. Our results indicate that frontier geometry, not emergent LLM reasoning, drives most reported gains, and suggest that metric-aware prompts or offline semantic graphs are necessary before attributing navigation success to "LLM intelligence." 

**Abstract (ZH)**: 大型语言模型在物体目标导航中的作用及其对规划改善的程度：HM3D-v1验证集上的重新审视 

---
# Anomaly Detection in Human Language via Meta-Learning: A Few-Shot Approach 

**Title (ZH)**: 基于元学习的少样本人类语言异常检测 

**Authors**: Saurav Singla, Aarav Singla, Advik Gupta, Parnika Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2507.20019)  

**Abstract**: We propose a meta learning framework for detecting anomalies in human language across diverse domains with limited labeled data. Anomalies in language ranging from spam and fake news to hate speech pose a major challenge due to their sparsity and variability. We treat anomaly detection as a few shot binary classification problem and leverage meta-learning to train models that generalize across tasks. Using datasets from domains such as SMS spam, COVID-19 fake news, and hate speech, we evaluate model generalization on unseen tasks with minimal labeled anomalies. Our method combines episodic training with prototypical networks and domain resampling to adapt quickly to new anomaly detection tasks. Empirical results show that our method outperforms strong baselines in F1 and AUC scores. We also release the code and benchmarks to facilitate further research in few-shot text anomaly detection. 

**Abstract (ZH)**: 一种基于元学习的跨领域有限标注数据异常检测框架及应用 

---
# The Carbon Cost of Conversation, Sustainability in the Age of Language Models 

**Title (ZH)**: 语言模型时代交流的碳成本：可持续性探究 

**Authors**: Sayed Mahbub Hasan Amiri, Prasun Goswami, Md. Mainul Islam, Mohammad Shakhawat Hossen, Sayed Majhab Hasan Amiri, Naznin Akter  

**Link**: [PDF](https://arxiv.org/pdf/2507.20018)  

**Abstract**: Large language models (LLMs) like GPT-3 and BERT have revolutionized natural language processing (NLP), yet their environmental costs remain dangerously overlooked. This article critiques the sustainability of LLMs, quantifying their carbon footprint, water usage, and contribution to e-waste through case studies of models such as GPT-4 and energy-efficient alternatives like Mistral 7B. Training a single LLM can emit carbon dioxide equivalent to hundreds of cars driven annually, while data centre cooling exacerbates water scarcity in vulnerable regions. Systemic challenges corporate greenwashing, redundant model development, and regulatory voids perpetuate harm, disproportionately burdening marginalized communities in the Global South. However, pathways exist for sustainable NLP: technical innovations (e.g., model pruning, quantum computing), policy reforms (carbon taxes, mandatory emissions reporting), and cultural shifts prioritizing necessity over novelty. By analysing industry leaders (Google, Microsoft) and laggards (Amazon), this work underscores the urgency of ethical accountability and global cooperation. Without immediate action, AIs ecological toll risks outpacing its societal benefits. The article concludes with a call to align technological progress with planetary boundaries, advocating for equitable, transparent, and regenerative AI systems that prioritize both human and environmental well-being. 

**Abstract (ZH)**: 大型语言模型（LLMs）如GPT-3和BERT重塑了自然语言处理（NLP），但其环境成本仍然被危险地忽视。本文批评了LLMs的可持续性，通过GPT-4等模型和节能替代品Mistral 7B的案例研究，定量评估其碳足迹、水资源消耗及其对电子废物的贡献。训练单个LLM所产生的二氧化碳相当于数百辆汽车一年的排放量，而数据中心冷却进一步加剧了脆弱地区水资源短缺。系统性挑战包括企业绿色漂洗、冗余模型开发以及监管空白，这些都加剧了危害，不成比例地影响着全球南方的边缘化社区。然而，可持续NLP的道路存在：技术创新（如模型剪裁、量子计算），政策改革（如碳税、强制性排放报告），以及文化转变，优先考虑必要性而非新颖性。通过分析行业领导者（谷歌、微软）和落后者（亚马逊），本文强调了道德问责和全球合作的迫切性。如果不采取立即行动，人工智能的生态成本可能会超过其社会收益。文章最后呼吁将技术进步与行星边界相一致，倡导公平、透明和再生的人工智能系统，既重视人类福祉也重视环境福祉。 

---
# FedSWA: Improving Generalization in Federated Learning with Highly Heterogeneous Data via Momentum-Based Stochastic Controlled Weight Averaging 

**Title (ZH)**: FedSWA：通过动量基于的随机控制权重平均提高异质数据联邦学习的泛化能力 

**Authors**: Liu junkang, Yuanyuan Liu, Fanhua Shang, Hongying Liu, Jin Liu, Wei Feng  

**Link**: [PDF](https://arxiv.org/pdf/2507.20016)  

**Abstract**: For federated learning (FL) algorithms such as FedSAM, their generalization capability is crucial for real-word applications. In this paper, we revisit the generalization problem in FL and investigate the impact of data heterogeneity on FL generalization. We find that FedSAM usually performs worse than FedAvg in the case of highly heterogeneous data, and thus propose a novel and effective federated learning algorithm with Stochastic Weight Averaging (called \texttt{FedSWA}), which aims to find flatter minima in the setting of highly heterogeneous data. Moreover, we introduce a new momentum-based stochastic controlled weight averaging FL algorithm (\texttt{FedMoSWA}), which is designed to better align local and global models.
Theoretically, we provide both convergence analysis and generalization bounds for \texttt{FedSWA} and \texttt{FedMoSWA}. We also prove that the optimization and generalization errors of \texttt{FedMoSWA} are smaller than those of their counterparts, including FedSAM and its variants. Empirically, experimental results on CIFAR10/100 and Tiny ImageNet demonstrate the superiority of the proposed algorithms compared to their counterparts. Open source code at: this https URL. 

**Abstract (ZH)**: 对于像FedSAM这样的联邦学习（FL）算法，其泛化能力对于实际应用至关重要。本文重新审视了FL中的泛化问题，并探讨了数据异质性对FL泛化的影响。我们发现，在高度异质性数据的情况下，FedSAM的表现通常不如FedAvg，因此提出了一种新的有效联邦学习算法（结合了Stochastic Weight Averaging，称为FedSWA），旨在在高度异质性数据的设定下寻找更平滑的最小值。此外，我们引入了一种基于动量的随机控制权重平均联邦学习算法（称为FedMoSWA），旨在更好地对齐局部模型和全局模型。从理论上，我们为FedSWA和FedMoSWA提供了收敛分析和泛化界。我们还证明了FedMoSWA的优化和泛化误差小于其对应的算法，包括FedSAM及其变体。实验结果表明，所提出的算法在CIFAR10/100和Tiny ImageNet上的表现优于其对应的算法。开源代码见：this https URL。 

---
# Policy-Driven AI in Dataspaces: Taxonomy, Explainability, and Pathways for Compliant Innovation 

**Title (ZH)**: 数据空间中基于政策的AI：分类、可解释性及合规创新路径 

**Authors**: Joydeep Chandra, Satyam Kumar Navneet  

**Link**: [PDF](https://arxiv.org/pdf/2507.20014)  

**Abstract**: As AI-driven dataspaces become integral to data sharing and collaborative analytics, ensuring privacy, performance, and policy compliance presents significant challenges. This paper provides a comprehensive review of privacy-preserving and policy-aware AI techniques, including Federated Learning, Differential Privacy, Trusted Execution Environments, Homomorphic Encryption, and Secure Multi-Party Computation, alongside strategies for aligning AI with regulatory frameworks such as GDPR and the EU AI Act. We propose a novel taxonomy to classify these techniques based on privacy levels, performance impacts, and compliance complexity, offering a clear framework for practitioners and researchers to navigate trade-offs. Key performance metrics -- latency, throughput, cost overhead, model utility, fairness, and explainability -- are analyzed to highlight the multi-dimensional optimization required in dataspaces. The paper identifies critical research gaps, including the lack of standardized privacy-performance KPIs, challenges in explainable AI for federated ecosystems, and semantic policy enforcement amidst regulatory fragmentation. Future directions are outlined, proposing a conceptual framework for policy-driven alignment, automated compliance validation, standardized benchmarking, and integration with European initiatives like GAIA-X, IDS, and Eclipse EDC. By synthesizing technical, ethical, and regulatory perspectives, this work lays the groundwork for developing trustworthy, efficient, and compliant AI systems in dataspaces, fostering innovation in secure and responsible data-driven ecosystems. 

**Abstract (ZH)**: AI驱动的数据空间中的隐私保护与政策意识智能技术综述：面向信任、效率与合规的路径探索 

---
# Robust Taxi Fare Prediction Under Noisy Conditions: A Comparative Study of GAT, TimesNet, and XGBoost 

**Title (ZH)**: 在噪声条件下稳健的出租车 fare 预测：GAT、TimesNet 和 XGBoost 的对比研究 

**Authors**: Padmavathi Moorthy  

**Link**: [PDF](https://arxiv.org/pdf/2507.20008)  

**Abstract**: Precise fare prediction is crucial in ride-hailing platforms and urban mobility systems. This study examines three machine learning models-Graph Attention Networks (GAT), XGBoost, and TimesNet to evaluate their predictive capabilities for taxi fares using a real-world dataset comprising over 55 million records. Both raw (noisy) and denoised versions of the dataset are analyzed to assess the impact of data quality on model performance. The study evaluated the models along multiple axes, including predictive accuracy, calibration, uncertainty estimation, out-of-distribution (OOD) robustness, and feature sensitivity. We also explore pre-processing strategies, including KNN imputation, Gaussian noise injection, and autoencoder-based denoising. The study reveals critical differences between classical and deep learning models under realistic conditions, offering practical guidelines for building robust and scalable models in urban fare prediction systems. 

**Abstract (ZH)**: 精确的 fare 预测对于网约车平台和城市出行系统至关重要。本研究探讨了三种机器学习模型——图注意网络（GAT）、XGBoost 和 TimesNet，在使用包含超过 5500 万条记录的真实数据集评估其对出租车 fare 的预测能力方面的表现。研究分析了原始（嘈杂）和去噪后的数据集，以评估数据质量对模型性能的影响。本研究从预测准确性、校准、不确定性估计、离域（OOD）稳健性和特征敏感性等多方面评估了这些模型。此外，研究还探讨了预处理策略，包括 KNN 插值、高斯噪声注入和基于自编码器的去噪。研究揭示了在实际条件下经典模型与深度学习模型之间的关键差异，并为在城市 fare 预测系统中构建稳健和可扩展的模型提供了实用指南。 

---
# VLQA: The First Comprehensive, Large, and High-Quality Vietnamese Dataset for Legal Question Answering 

**Title (ZH)**: VLQA：首个全面、大规模且高质量的越南语法律问答数据集 

**Authors**: Tan-Minh Nguyen, Hoang-Trung Nguyen, Trong-Khoi Dao, Xuan-Hieu Phan, Ha-Thanh Nguyen, Thi-Hai-Yen Vuong  

**Link**: [PDF](https://arxiv.org/pdf/2507.19995)  

**Abstract**: The advent of large language models (LLMs) has led to significant achievements in various domains, including legal text processing. Leveraging LLMs for legal tasks is a natural evolution and an increasingly compelling choice. However, their capabilities are often portrayed as greater than they truly are. Despite the progress, we are still far from the ultimate goal of fully automating legal tasks using artificial intelligence (AI) and natural language processing (NLP). Moreover, legal systems are deeply domain-specific and exhibit substantial variation across different countries and languages. The need for building legal text processing applications for different natural languages is, therefore, large and urgent. However, there is a big challenge for legal NLP in low-resource languages such as Vietnamese due to the scarcity of resources and annotated data. The need for labeled legal corpora for supervised training, validation, and supervised fine-tuning is critical. In this paper, we introduce the VLQA dataset, a comprehensive and high-quality resource tailored for the Vietnamese legal domain. We also conduct a comprehensive statistical analysis of the dataset and evaluate its effectiveness through experiments with state-of-the-art models on legal information retrieval and question-answering tasks. 

**Abstract (ZH)**: 大型语言模型的兴起在各个领域取得了显著成就，包括法律文本处理。利用大型语言模型进行法律任务是自然演进和日益有吸引力的选择。然而，它们的能力往往被夸大。尽管有所进展，我们仍然远未达到完全利用人工智能和自然语言处理自动化法律任务的目标。此外，法律体系高度特定于特定领域，并且在不同国家和语言之间存在显著差异。因此，根据不同自然语言构建法律文本处理应用程序的需求十分迫切和紧急。然而，法律自然语言处理在低资源语言如越南语中面临重大挑战，原因是资源和标注数据稀缺。构建监督训练、验证和微调所需的标记法律语料库至关重要。本文介绍了VLQA数据集，这是一个专为越南法律领域设计的全面且高质量的资源。我们还对数据集进行了全面的统计分析，并通过使用最新模型在法律信息检索和问答任务中的实验评估了其有效性。 

---
# NIRS: An Ontology for Non-Invasive Respiratory Support in Acute Care 

**Title (ZH)**: NIRS: 一种无创呼吸支持的本体论 

**Authors**: Md Fantacher Islam, Jarrod Mosier, Vignesh Subbian  

**Link**: [PDF](https://arxiv.org/pdf/2507.19992)  

**Abstract**: Objective: Develop a Non Invasive Respiratory Support (NIRS) ontology to support knowledge representation in acute care settings.
Materials and Methods: We developed the NIRS ontology using Web Ontology Language (OWL) semantics and Protege to organize clinical concepts and relationships. To enable rule-based clinical reasoning beyond hierarchical structures, we added Semantic Web Rule Language (SWRL) rules. We evaluated logical reasoning by adding 17 hypothetical patient clinical scenarios. We used SPARQL queries and data from the Electronic Intensive Care Unit (eICU) Collaborative Research Database to retrieve and test targeted inferences.
Results: The ontology has 132 classes, 12 object properties, and 17 data properties across 882 axioms that establish concept relationships. To standardize clinical concepts, we added 350 annotations, including descriptive definitions based on controlled vocabularies. SPARQL queries successfully validated all test cases (rules) by retrieving appropriate patient outcomes, for instance, a patient treated with HFNC (high-flow nasal cannula) for 2 hours due to acute respiratory failure may avoid endotracheal intubation.
Discussion: The NIRS ontology formally represents domain-specific concepts, including ventilation modalities, patient characteristics, therapy parameters, and outcomes. SPARQL query evaluations on clinical scenarios confirmed the ability of the ontology to support rule based reasoning and therapy recommendations, providing a foundation for consistent documentation practices, integration into clinical data models, and advanced analysis of NIRS outcomes.
Conclusion: We unified NIRS concepts into an ontological framework and demonstrated its applicability through the evaluation of hypothetical patient scenarios and alignment with standardized vocabularies. 

**Abstract (ZH)**: 目标: 开发非侵入性呼吸支持(NIRS)本体以支持急性护理环境中知识表示。 

---
# Improving the Performance of Sequential Recommendation Systems with an Extended Large Language Model 

**Title (ZH)**: 使用扩展的大语言模型提高序列推荐系统的性能 

**Authors**: Sinnyum Choi, Woong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.19990)  

**Abstract**: Recently, competition in the field of artificial intelligence (AI) has intensified among major technological companies, resulting in the continuous release of new large-language models (LLMs) that exhibit improved language understanding and context-based reasoning capabilities. It is expected that these advances will enable more efficient personalized recommendations in LLM-based recommendation systems through improved quality of training data and architectural design. However, many studies have not considered these recent developments. In this study, it was proposed to improve LLM-based recommendation systems by replacing Llama2 with Llama3 in the LlamaRec framework. To ensure a fair comparison, random seed values were set and identical input data was provided during preprocessing and training. The experimental results show average performance improvements of 38.65\%, 8.69\%, and 8.19\% for the ML-100K, Beauty, and Games datasets, respectively, thus confirming the practicality of this method. Notably, the significant improvements achieved by model replacement indicate that the recommendation quality can be improved cost-effectively without the need to make structural changes to the system. Based on these results, it is our contention that the proposed approach is a viable solution for improving the performance of current recommendation systems. 

**Abstract (ZH)**: 近年来，主要科技公司在人工智能领域的竞争加剧，导致不断推出新的大语言模型（LLMs），这些模型在语言理解和上下文推理能力上有所提升。预计这些进步将通过提高训练数据质量和架构设计，使基于LLM的推荐系统能够实现更高效的个性化推荐。然而，许多研究尚未考虑这些最新发展。本研究提出通过在LlamaRec框架中用Llama3替换Llama2来改进基于LLM的推荐系统。为了确保比较的公平性，在预处理和训练过程中设定了随机种子值，并提供了相同的输入数据。实验结果显示，与ML-100K、Beauty和Games数据集相比，平均性能分别提高了38.65%、8.69%和8.19%，这证实了该方法的实际可行性。值得注意的是，模型替换所取得的显著改进表明，通过优化模型本身即可有效提高推荐质量，无需对系统进行结构性改动。基于这些结果，我们认为提出的方法是提高现有推荐系统性能的一种可行方案。 

---
# CLASP: General-Purpose Clothes Manipulation with Semantic Keypoints 

**Title (ZH)**: CLASP: 基于语义关键点的一般服装操作方法 

**Authors**: Yuhong Deng, Chao Tang, Cunjun Yu, Linfeng Li, David Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2507.19983)  

**Abstract**: Clothes manipulation, such as folding or hanging, is a critical capability for home service robots. Despite recent advances, most existing methods remain limited to specific tasks and clothes types, due to the complex, high-dimensional geometry of clothes. This paper presents CLothes mAnipulation with Semantic keyPoints (CLASP), which aims at general-purpose clothes manipulation over different clothes types, T-shirts, shorts, skirts, long dresses, ... , as well as different tasks, folding, flattening, hanging, ... . The core idea of CLASP is semantic keypoints -- e.g., ''left sleeve'', ''right shoulder'', etc. -- a sparse spatial-semantic representation that is salient for both perception and action. Semantic keypoints of clothes can be reliably extracted from RGB-D images and provide an effective intermediate representation of clothes manipulation policies. CLASP uses semantic keypoints to bridge high-level task planning and low-level action execution. At the high level, it exploits vision language models (VLMs) to predict task plans over the semantic keypoints. At the low level, it executes the plans with the help of a simple pre-built manipulation skill library. Extensive simulation experiments show that CLASP outperforms state-of-the-art baseline methods on multiple tasks across diverse clothes types, demonstrating strong performance and generalization. Further experiments with a Franka dual-arm system on four distinct tasks -- folding, flattening, hanging, and placing -- confirm CLASP's performance on a real robot. 

**Abstract (ZH)**: Clothes 操作：基于语义关键点的通用家用服务机器人衣物操作方法 

---
# A roadmap for AI in robotics 

**Title (ZH)**: AI在机器人领域的应用 roadmap 

**Authors**: Aude Billard, Alin Albu-Schaeffer, Michael Beetz, Wolfram Burgard, Peter Corke, Matei Ciocarlie, Ravinder Dahiya, Danica Kragic, Ken Goldberg, Yukie Nagai, Davide Scaramuzza  

**Link**: [PDF](https://arxiv.org/pdf/2507.19975)  

**Abstract**: AI technologies, including deep learning, large-language models have gone from one breakthrough to the other. As a result, we are witnessing growing excitement in robotics at the prospect of leveraging the potential of AI to tackle some of the outstanding barriers to the full deployment of robots in our daily lives. However, action and sensing in the physical world pose greater and different challenges than analysing data in isolation. As the development and application of AI in robotic products advances, it is important to reflect on which technologies, among the vast array of network architectures and learning models now available in the AI field, are most likely to be successfully applied to robots; how they can be adapted to specific robot designs, tasks, environments; which challenges must be overcome. This article offers an assessment of what AI for robotics has achieved since the 1990s and proposes a short- and medium-term research roadmap listing challenges and promises. These range from keeping up-to-date large datasets, representatives of a diversity of tasks robots may have to perform, and of environments they may encounter, to designing AI algorithms tailored specifically to robotics problems but generic enough to apply to a wide range of applications and transfer easily to a variety of robotic platforms. For robots to collaborate effectively with humans, they must predict human behavior without relying on bias-based profiling. Explainability and transparency in AI-driven robot control are not optional but essential for building trust, preventing misuse, and attributing responsibility in accidents. We close on what we view as the primary long-term challenges, that is, to design robots capable of lifelong learning, while guaranteeing safe deployment and usage, and sustainable computational costs. 

**Abstract (ZH)**: AI技术，包括深度学习和大规模语言模型，已从一个突破接踵而至的另一个突破。随着人工智能潜力在解决机器人全面部署中遇到的一些障碍方面的展望，我们在机器人领域正见证着日益增长的热情。然而，物理世界的操作和感知比单独分析数据带来了更大的、不同的挑战。随着人工智能在机器人产品中的发展和应用，反思哪些网络架构和技术，在当前可供选择的广泛人工智能学习模型中，最有可能成功应用于机器人；如何适应特定的机器人设计、任务和环境；哪些挑战必须被克服，这很重要。本文评估了自20世纪90年代以来人工智能在机器人领域的成就，并提出了一份短期和中期的研究路线图，列出了一系列挑战和前景。这些从保持与机器人可能执行的各种任务和可能遇到的各种环境相代表的大规模数据集更新，到为机器人问题量身定制的AI算法，这些算法具有广泛的适用性和灵活的机器人平台过渡能力。为了使机器人有效协作，它们必须预测人类行为，而无需依赖基于偏见的画像。基于人工智能的机器人控制中的解释性与透明性不仅是可选的，而是构建信任、防止滥用和在事故中承担责任的关键。最后，我们着重讨论我们所认为的主要长期挑战，即设计能够终身学习的机器人，同时确保安全部署和使用，以及可持续的计算成本。 

---
# Dimer-Enhanced Optimization: A First-Order Approach to Escaping Saddle Points in Neural Network Training 

**Title (ZH)**: 二聚体增强优化：神经网络训练中逃逸鞍点的首阶方法 

**Authors**: Yue Hu, Zanxia Cao, Yingchao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.19968)  

**Abstract**: First-order optimization methods, such as SGD and Adam, are widely used for training large-scale deep neural networks due to their computational efficiency and robust performance. However, relying solely on gradient information, these methods often struggle to navigate complex loss landscapes with flat regions, plateaus, and saddle points. Second-order methods, which use curvature information from the Hessian matrix, can address these challenges but are computationally infeasible for large models. The Dimer method, a first-order technique that constructs two closely spaced points to probe the local geometry of a potential energy surface, efficiently estimates curvature using only gradient information. Inspired by its use in molecular dynamics simulations for locating saddle points, we propose Dimer-Enhanced Optimization (DEO), a novel framework to escape saddle points in neural network training. DEO adapts the Dimer method to explore a broader region of the loss landscape, approximating the Hessian's smallest eigenvector without computing the full matrix. By periodically projecting the gradient onto the subspace orthogonal to the minimum curvature direction, DEO guides the optimizer away from saddle points and flat regions, enhancing training efficiency with non-stepwise updates. Preliminary experiments on a Transformer toy model show DEO achieves competitive performance compared to standard first-order methods, improving navigation of complex loss landscapes. Our work repurposes physics-inspired, first-order curvature estimation to enhance neural network training in high-dimensional spaces. 

**Abstract (ZH)**: 基于一阶优化方法，如SGD和Adam，广泛用于训练大规模深度神经网络，因其计算效率和鲁棒性能。然而，这些方法仅依赖梯度信息，在复杂损失landscape中的平坦区域、平台和鞍点区域导航时常常力不从心。二阶方法利用海森矩阵的曲率信息可以解决这些问题，但对大型模型而言计算上行不通。Dimer方法是一种基于一阶技术，通过构建两个紧密间隔的点以探索潜在能量面上的局部几何结构，仅使用梯度信息高效估计曲率。受其在分子动力学模拟中用于定位鞍点的启发，我们提出了Dimer增强优化（DEO）框架，旨在神经网络训练中逃逸鞍点。DEO将Dimer方法适应性地扩展到探索损失landscape的更广阔区域，无需计算整个海森矩阵即可近似海森矩阵的最小特征向量。通过周期性地将梯度投影到曲率最小方向的正交子空间，DEO引导优化器远离鞍点和平坦区域，通过非梯度更新方式增强训练效率。初步实验显示，DEO在Transformer玩具模型上的表现与标准一阶方法相当，提高了复杂损失landscape的导航能力。我们的研究将物理启发的一阶曲率估计方法重新应用于高维空间中的神经网络训练中。 

---
# Pic2Diagnosis: A Method for Diagnosis of Cardiovascular Diseases from the Printed ECG Pictures 

**Title (ZH)**: Pic2Diagnosis: 一种基于打印心电图图片的心血管疾病诊断方法 

**Authors**: Oğuzhan Büyüksolak, İlkay Öksüz  

**Link**: [PDF](https://arxiv.org/pdf/2507.19961)  

**Abstract**: The electrocardiogram (ECG) is a vital tool for diagnosing heart diseases. However, many disease patterns are derived from outdated datasets and traditional stepwise algorithms with limited accuracy. This study presents a method for direct cardiovascular disease (CVD) diagnosis from ECG images, eliminating the need for digitization. The proposed approach utilizes a two-step curriculum learning framework, beginning with the pre-training of a classification model on segmentation masks, followed by fine-tuning on grayscale, inverted ECG images. Robustness is further enhanced through an ensemble of three models with averaged outputs, achieving an AUC of 0.9534 and an F1 score of 0.7801 on the BHF ECG Challenge dataset, outperforming individual models. By effectively handling real-world artifacts and simplifying the diagnostic process, this method offers a reliable solution for automated CVD diagnosis, particularly in resource-limited settings where printed or scanned ECG images are commonly used. Such an automated procedure enables rapid and accurate diagnosis, which is critical for timely intervention in CVD cases that often demand urgent care. 

**Abstract (ZH)**: 电 cardiogram (ECG) 是诊断心脏疾病的重要工具。然而，许多疾病模式源于过时的数据集和传统步进算法，这些算法的准确性有限。本研究提出了一种直接从 ECG 图像诊断心血管疾病 (CVD) 的方法，无需进行数字化处理。所提出的方法利用了一个两步 Curriculum 学习框架，首先在分割掩码上预训练分类模型，然后在灰度反转的 ECG 图像上进行微调。通过三种模型的集成，输出平均值进一步增强了鲁棒性，在 BHF ECG 挑战数据集上实现了 AUC 为 0.9534 和 F1 分数为 0.7801 的性能，优于单个模型。通过有效处理实际的图像伪影并简化诊断过程，该方法在资源有限的环境中提供了一种可靠的自动化 CVD 诊断解决方案，这些环境中常常使用打印或扫描的 ECG 图像。这种自动化的操作流程能够实现快速且准确的诊断，这对于需要及时干预的心血管疾病病例至关重要。 

---
# Predicting Brain Responses To Natural Movies With Multimodal LLMs 

**Title (ZH)**: 使用多模态大型语言模型预测大脑对自然电影的反应 

**Authors**: Cesar Kadir Torrico Villanueva, Jiaxin Cindy Tu, Mihir Tripathy, Connor Lane, Rishab Iyer, Paul S. Scotti  

**Link**: [PDF](https://arxiv.org/pdf/2507.19956)  

**Abstract**: We present MedARC's team solution to the Algonauts 2025 challenge. Our pipeline leveraged rich multimodal representations from various state-of-the-art pretrained models across video (V-JEPA2), speech (Whisper), text (Llama 3.2), vision-text (InternVL3), and vision-text-audio (Qwen2.5-Omni). These features extracted from the models were linearly projected to a latent space, temporally aligned to the fMRI time series, and finally mapped to cortical parcels through a lightweight encoder comprising a shared group head plus subject-specific residual heads. We trained hundreds of model variants across hyperparameter settings, validated them on held-out movies and assembled ensembles targeted to each parcel in each subject. Our final submission achieved a mean Pearson's correlation of 0.2085 on the test split of withheld out-of-distribution movies, placing our team in fourth place for the competition. We further discuss a last-minute optimization that would have raised us to second place. Our results highlight how combining features from models trained in different modalities, using a simple architecture consisting of shared-subject and single-subject components, and conducting comprehensive model selection and ensembling improves generalization of encoding models to novel movie stimuli. All code is available on GitHub. 

**Abstract (ZH)**: MedARC团队参加Algonauts 2025挑战赛的解决方案 

---
# RARE: Refine Any Registration of Pairwise Point Clouds via Zero-Shot Learning 

**Title (ZH)**: RARE: 通过零样本学习精炼配对点云注册 

**Authors**: Chengyu Zheng, Jin Huang, Honghua Chen, Mingqiang Wei  

**Link**: [PDF](https://arxiv.org/pdf/2507.19950)  

**Abstract**: Recent research leveraging large-scale pretrained diffusion models has demonstrated the potential of using diffusion features to establish semantic correspondences in images. Inspired by advancements in diffusion-based techniques, we propose a novel zero-shot method for refining point cloud registration algorithms. Our approach leverages correspondences derived from depth images to enhance point feature representations, eliminating the need for a dedicated training dataset. Specifically, we first project the point cloud into depth maps from multiple perspectives and extract implicit knowledge from a pretrained diffusion network as depth diffusion features. These features are then integrated with geometric features obtained from existing methods to establish more accurate correspondences between point clouds. By leveraging these refined correspondences, our approach achieves significantly improved registration accuracy. Extensive experiments demonstrate that our method not only enhances the performance of existing point cloud registration techniques but also exhibits robust generalization capabilities across diverse datasets. Codes are available at this https URL. 

**Abstract (ZH)**: 近期研究利用大规模预训练扩散模型展示了使用扩散特征建立图像语义对应关系的潜力。受扩散基技术进展的启发，我们提出了一种新的零样本方法，用于细化点云配准算法。该方法利用来自深度图的对应关系增强点特征表示，从而避免了专用训练数据集的需求。具体地，我们首先将点云投影到多视角的深度图中，并从预训练的扩散网络中提取隐含知识作为深度扩散特征。随后，将这些特征与现有方法获得的几何特征结合，以建立更准确的点云对应关系。通过利用这些细化的对应关系，我们的方法实现了显著提高的配准精度。大量实验表明，我们的方法不仅提升了现有点云配准技术的性能，还展示了在多种数据集上具有稳健的泛化能力。代码可在以下链接获取：this https URL。 

---
# Deep Learning Based Joint Channel Estimation and Positioning for Sparse XL-MIMO OFDM Systems 

**Title (ZH)**: 基于深度学习的稀疏XL-MIMO OFDM系统信道估计与定位联合方法 

**Authors**: Zhongnian Li, Chao Zheng, Jian Xiao, Ji Wang, Gongpu Wang, Ming Zeng, Octavia A. Dobre  

**Link**: [PDF](https://arxiv.org/pdf/2507.19936)  

**Abstract**: This paper investigates joint channel estimation and positioning in near-field sparse extra-large multiple-input multiple-output (XL-MIMO) orthogonal frequency division multiplexing (OFDM) systems. To achieve cooperative gains between channel estimation and positioning, we propose a deep learning-based two-stage framework comprising positioning and channel estimation. In the positioning stage, the user's coordinates are predicted and utilized in the channel estimation stage, thereby enhancing the accuracy of channel estimation. Within this framework, we propose a U-shaped Mamba architecture for channel estimation and positioning, termed as CP-Mamba. This network integrates the strengths of the Mamba model with the structural advantages of U-shaped convolutional networks, enabling effective capture of local spatial features and long-range temporal dependencies of the channel. Numerical simulation results demonstrate that the proposed two-stage approach with CP-Mamba architecture outperforms existing baseline methods. Moreover, sparse arrays (SA) exhibit significantly superior performance in both channel estimation and positioning accuracy compared to conventional compact arrays. 

**Abstract (ZH)**: 基于近场稀疏超大规模多输入多输出(XL-MIMO)正交频率分组多载波系统中的联合信道估计算法与定位研究 

---
# DynamiX: Large-Scale Dynamic Social Network Simulator 

**Title (ZH)**: DynamiX：大规模动态社会网络模拟器 

**Authors**: Yanhui Sun, Wu Liu, Wentao Wang, Hantao Yao, Jiebo Luo, Yongdong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19929)  

**Abstract**: Understanding the intrinsic mechanisms of social platforms is an urgent demand to maintain social stability. The rise of large language models provides significant potential for social network simulations to capture attitude dynamics and reproduce collective behaviors. However, existing studies mainly focus on scaling up agent populations, neglecting the dynamic evolution of social relationships. To address this gap, we introduce DynamiX, a novel large-scale social network simulator dedicated to dynamic social network modeling. DynamiX uses a dynamic hierarchy module for selecting core agents with key characteristics at each timestep, enabling accurate alignment of real-world adaptive switching of user roles. Furthermore, we design distinct dynamic social relationship modeling strategies for different user types. For opinion leaders, we propose an information-stream-based link prediction method recommending potential users with similar stances, simulating homogeneous connections, and autonomous behavior decisions. For ordinary users, we construct an inequality-oriented behavior decision-making module, effectively addressing unequal social interactions and capturing the patterns of relationship adjustments driven by multi-dimensional factors. Experimental results demonstrate that DynamiX exhibits marked improvements in attitude evolution simulation and collective behavior analysis compared to static networks. Besides, DynamiX opens a new theoretical perspective on follower growth prediction, providing empirical evidence for opinion leaders cultivation. 

**Abstract (ZH)**: 理解社交平台的内在机制以维护社会稳定是一个迫切的需求。大型语言模型的兴起为社交网络仿真捕捉态度动态和再现集体行为提供了重要潜力。然而，现有研究主要集中在扩展代理人群体上，忽略了社交关系的动态演变。为解决这一问题，我们引入了DynamiX，这是一种新型的大规模社交网络仿真器，专注于动态社会网络建模。DynamiX采用动态层次模块，在每个时间步长选择具有关键特征的核心代理，以准确对齐用户角色的现实适应性切换。此外，我们为不同用户类型设计了独特的动态社交关系建模策略。对于意见领袖，我们提出了基于信息流的链接预测方法，推荐立场相似的潜在用户，模拟同质连接，并模拟自主行为决策。对于普通用户，我们构建了一个以不平等为导向的行为决策模块，有效解决了不平等的社会互动问题，并捕捉了由多维因素驱动的关系调整模式。实验结果表明，DynamiX在态度演变仿真和集体行为分析方面相较于静态网络具有显著改进。此外，DynamiX为追随者增长预测提供了一个新的理论视角，并提供了意见领袖培养的实证证据。 

---
# A mini-batch training strategy for deep subspace clustering networks 

**Title (ZH)**: 小批量训练策略用于深子空间聚类网络 

**Authors**: Yuxuan Jiang, Chenwei Yu, Zhi Lin, Xiaolan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.19917)  

**Abstract**: Mini-batch training is a cornerstone of modern deep learning, offering computational efficiency and scalability for training complex architectures. However, existing deep subspace clustering (DSC) methods, which typically combine an autoencoder with a self-expressive layer, rely on full-batch processing. The bottleneck arises from the self-expressive module, which requires representations of the entire dataset to construct a self-representation coefficient matrix. In this work, we introduce a mini-batch training strategy for DSC by integrating a memory bank that preserves global feature representations. Our approach enables scalable training of deep architectures for subspace clustering with high-resolution images, overcoming previous limitations. Additionally, to efficiently fine-tune large-scale pre-trained encoders for subspace clustering, we propose a decoder-free framework that leverages contrastive learning instead of autoencoding for representation learning. This design not only eliminates the computational overhead of decoder training but also provides competitive performance. Extensive experiments demonstrate that our approach not only achieves performance comparable to full-batch methods, but outperforms other state-of-the-art subspace clustering methods on the COIL100 and ORL datasets by fine-tuning deep networks. 

**Abstract (ZH)**: mini-batch训练是一种现代深度学习的基石，提供了训练复杂架构的计算效率和可扩展性。然而，现有的深度子空间聚类（DSC）方法通常将自动编码器与自表达层结合起来，依赖于全批次处理。瓶颈在于自表达模块，它需要整个数据集的表示来构建自表示系数矩阵。在此工作中，我们通过集成一个记忆库来为DSC引入mini-batch训练策略，该记忆库保留全局特征表示。我们的方法使得使用高分辨率图像对子空间聚类的深度架构进行可扩展训练，克服了之前的方法限制。此外，为了高效地微调大规模预训练编码器进行子空间聚类，我们提出了一种无解码器框架，利用对比学习而非自动编码进行表征学习。这种设计不仅消除了解码器训练的计算开销，而且还提供了竞争力的表现。广泛实验表明，我们的方法不仅在性能上与全批次方法相当，还在COIL100和ORL数据集上通过微调深度网络优于其他最先进的子空间聚类方法。 

---
# The Impact of Fine-tuning Large Language Models on Automated Program Repair 

**Title (ZH)**: 大型语言模型微调对自动化程序修复的影响 

**Authors**: Roman Macháček, Anastasiia Grishina, Max Hort, Leon Moonen  

**Link**: [PDF](https://arxiv.org/pdf/2507.19909)  

**Abstract**: Automated Program Repair (APR) uses various tools and techniques to help developers achieve functional and error-free code faster. In recent years, Large Language Models (LLMs) have gained popularity as components in APR tool chains because of their performance and flexibility. However, training such models requires a significant amount of resources. Fine-tuning techniques have been developed to adapt pre-trained LLMs to specific tasks, such as APR, and enhance their performance at far lower computational costs than training from scratch. In this study, we empirically investigate the impact of various fine-tuning techniques on the performance of LLMs used for APR. Our experiments provide insights into the performance of a selection of state-of-the-art LLMs pre-trained on code. The evaluation is done on three popular APR benchmarks (i.e., QuixBugs, Defects4J and HumanEval-Java) and considers six different LLMs with varying parameter sizes (resp. CodeGen, CodeT5, StarCoder, DeepSeekCoder, Bloom, and CodeLlama-2). We consider three training regimens: no fine-tuning, full fine-tuning, and parameter-efficient fine-tuning (PEFT) using LoRA and IA3. We observe that full fine-tuning techniques decrease the benchmarking performance of various models due to different data distributions and overfitting. By using parameter-efficient fine-tuning methods, we restrict models in the amount of trainable parameters and achieve better results.
Keywords: large language models, automated program repair, parameter-efficient fine-tuning, AI4Code, AI4SE, ML4SE. 

**Abstract (ZH)**: 自动程序修复中的大型语言模型细调：一种基于AI4Code和AI4SE的参数高效方法 

---
# CrossPL: Evaluating Large Language Models on Cross Programming Language Code Generation 

**Title (ZH)**: 跨语言代码生成：评估大型语言模型在不同编程语言代码生成任务上的表现 

**Authors**: Zhanhang Xiong, Dongxia Wang, Yuekang Li, Xinyuan An, Wenhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19904)  

**Abstract**: As large language models (LLMs) become increasingly embedded in software engineering workflows, a critical capability remains underexplored: generating correct code that enables cross-programming-language (CPL) interoperability. This skill is essential for building complex systems that integrate components written in multiple languages via mechanisms like inter-process communication (IPC). To bridge this gap, we present CrossPL, the first benchmark designed to systematically evaluate LLMs' ability to generate CPL-interoperating code. CrossPL comprises 1,982 tasks centered around IPC, covering six widely-used programming languages and seven representative CPL techniques. We construct this benchmark by (i) analyzing 19,169 multi-language GitHub repositories using 156 hand-crafted finite state machines (FSMs), and (ii) developing an LLM-based pipeline that automatically extracts CPL code snippets, generates task instructions, and validates functional correctness. We evaluate 14 state-of-the-art general-purpose LLMs and 6 code-oriented LLMs released in the past three years on CrossPL via FSM-based validation. Results reveal that even the best-performing models struggle with CPL scenarios, underscoring the need for more targeted research in this space. Our benchmark and code are available at: this https URL. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在软件工程工作流中的应用日益增多，一种关键能力仍被忽视：生成能够实现跨编程语言（CPL）互操作的正确代码。这项能力对于通过进程间通信（IPC）等机制集成多种语言编写的组件以构建复杂系统至关重要。为解决这一问题，我们提出了CrossPL，这是首个旨在系统评估LLMs生成CPL互操作代码能力的基准测试。CrossPL包含1,982项围绕IPC的任务，覆盖六种广泛使用的编程语言以及七种代表性的CPL技术。我们通过以下方式构建此基准测试：（i）使用156个手工设计的状态机（FSMs）分析了19,169个多语言GitHub仓库；（ii）开发了一种基于LLM的自动化流水线，用于自动提取CPL代码片段、生成任务指令并验证功能正确性。我们使用基于状态机的验证方法，评估了过去三年内发布的14种最先进的通用语言模型和6种代码导向语言模型在CrossPL上的表现。结果表明，即使性能最好的模型也难以处理CPL场景，突显了在该领域进行更有针对性研究的必要性。我们的基准测试和代码可在以下链接获取：this https URL。 

---
# AgentMesh: A Cooperative Multi-Agent Generative AI Framework for Software Development Automation 

**Title (ZH)**: AgentMesh：一种协作式多智能体生成AI软件开发自动化框架 

**Authors**: Sourena Khanzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2507.19902)  

**Abstract**: Software development is a complex, multi-phase process traditionally requiring collaboration among individuals with diverse expertise. We propose AgentMesh, a Python-based framework that uses multiple cooperating LLM-powered agents to automate software development tasks. In AgentMesh, specialized agents - a Planner, Coder, Debugger, and Reviewer - work in concert to transform a high-level requirement into fully realized code. The Planner agent first decomposes user requests into concrete subtasks; the Coder agent implements each subtask in code; the Debugger agent tests and fixes the code; and the Reviewer agent validates the final output for correctness and quality. We describe the architecture and design of these agents and their communication, and provide implementation details including prompt strategies and workflow orchestration. A case study illustrates AgentMesh handling a non-trivial development request via sequential task planning, code generation, iterative debugging, and final code review. We discuss how dividing responsibilities among cooperative agents leverages the strengths of large language models while mitigating single-agent limitations. Finally, we examine current limitations - such as error propagation and context scaling - and outline future work toward more robust, scalable multi-agent AI systems for software engineering automation. 

**Abstract (ZH)**: 基于Python的AgentMesh框架：多协作LLM驱动的软件开发自动化 

---
# TS-Insight: Visualizing Thompson Sampling for Verification and XAI 

**Title (ZH)**: TS-Insight: Visualizing Thompson Sampling for Verification and XAI 的中文标题为：

TS-Insight: Thompson Sampling的可视化在验证和可解释人工智能中的应用 

**Authors**: Parsa Vares, Éloi Durant, Jun Pang, Nicolas Médoc, Mohammad Ghoniem  

**Link**: [PDF](https://arxiv.org/pdf/2507.19898)  

**Abstract**: Thompson Sampling (TS) and its variants are powerful Multi-Armed Bandit algorithms used to balance exploration and exploitation strategies in active learning. Yet, their probabilistic nature often turns them into a ``black box'', hindering debugging and trust. We introduce TS-Insight, a visual analytics tool explicitly designed to shed light on the internal decision mechanisms of Thompson Sampling-based algorithms, for model developers. It comprises multiple plots, tracing for each arm the evolving posteriors, evidence counts, and sampling outcomes, enabling the verification, diagnosis, and explainability of exploration/exploitation dynamics. This tool aims at fostering trust and facilitating effective debugging and deployment in complex binary decision-making scenarios especially in sensitive domains requiring interpretable decision-making. 

**Abstract (ZH)**: Thomas采样（TS）及其变体是用于平衡活跃学习中探索与利用策略的多臂 bandit 算法。然而，它们的概率性质往往使它们成为“黑盒”，妨碍调试和信任。我们引入了 TS-Insight，一个专门为模型开发者设计的可视化分析工具，旨在阐明基于 Thomas 采样算法的内部决策机制。该工具包含多个图表，跟踪每个臂的后验分布、证据计数和抽样结果，从而实现探索/利用动态的验证、诊断和解释。该工具旨在增强信任并促进复杂二元决策场景中的有效调试和部署，特别是在需要可解释决策的敏感领域。 

---
# Interpretable Open-Vocabulary Referring Object Detection with Reverse Contrast Attention 

**Title (ZH)**: 可解释的开放词汇对象检测与逆对比注意力 

**Authors**: Drandreb Earl O. Juanico, Rowel O. Atienza, Jeffrey Kenneth Go  

**Link**: [PDF](https://arxiv.org/pdf/2507.19891)  

**Abstract**: We propose Reverse Contrast Attention (RCA), a plug-in method that enhances object localization in vision-language transformers without retraining. RCA reweights final-layer attention by suppressing extremes and amplifying mid-level activations to let semantically relevant but subdued tokens guide predictions. We evaluate it on Open Vocabulary Referring Object Detection (OV-RefOD), introducing FitAP, a confidence-free average precision metric based on IoU and box area. RCA improves FitAP in 11 out of 15 open-source VLMs, with gains up to $+26.6\%$. Effectiveness aligns with attention sharpness and fusion timing; while late-fusion models benefit consistently, models like $\texttt{DeepSeek-VL2}$ also improve, pointing to capacity and disentanglement as key factors. RCA offers both interpretability and performance gains for multimodal transformers. 

**Abstract (ZH)**: 我们提出反向对比注意（RCA），一种无需重新训练的插件方法，用于增强视觉-语言Transformer中的对象定位。RCA 通过抑制极端值并放大中间层激活来重新权重新层注意，让语义相关但被压制的标记引导预测。我们在开放词汇量物体引用检测（OV-RefOD）上进行了评估，引入了基于IoU和框面积的无信心度平均精度指标FitAP。RCA 在15个开源VLM中有11个上显示出改进，增幅最高达26.6%。效果与注意力锐度和融合时机一致；虽然晚期融合模型持续受益，如DeepSeek-VL2等模型也有所改进，表明容量和解耦是关键因素。RCA 为多模态Transformer提供了解释性和性能双重增益。 

---
# FedS2R: One-Shot Federated Domain Generalization for Synthetic-to-Real Semantic Segmentation in Autonomous Driving 

**Title (ZH)**: FedS2R: 一键式 federated 领域泛化方法用于自主驾驶中的合成-to-真实语义分割 

**Authors**: Tao Lian, Jose L. Gómez, Antonio M. López  

**Link**: [PDF](https://arxiv.org/pdf/2507.19881)  

**Abstract**: Federated domain generalization has shown promising progress in image classification by enabling collaborative training across multiple clients without sharing raw data. However, its potential in the semantic segmentation of autonomous driving remains underexplored. In this paper, we propose FedS2R, the first one-shot federated domain generalization framework for synthetic-to-real semantic segmentation in autonomous driving. FedS2R comprises two components: an inconsistency-driven data augmentation strategy that generates images for unstable classes, and a multi-client knowledge distillation scheme with feature fusion that distills a global model from multiple client models. Experiments on five real-world datasets, Cityscapes, BDD100K, Mapillary, IDD, and ACDC, show that the global model significantly outperforms individual client models and is only 2 mIoU points behind the model trained with simultaneous access to all client data. These results demonstrate the effectiveness of FedS2R in synthetic-to-real semantic segmentation for autonomous driving under federated learning 

**Abstract (ZH)**: 联邦域泛化在自动驾驶语义分割中的单次联邦域泛化框架FedS2R 

---
# Trivial Trojans: How Minimal MCP Servers Enable Cross-Tool Exfiltration of Sensitive Data 

**Title (ZH)**: 微不足道的木马：最小化MCP服务器如何实现跨工具敏感数据泄露 

**Authors**: Nicola Croce, Tobin South  

**Link**: [PDF](https://arxiv.org/pdf/2507.19880)  

**Abstract**: The Model Context Protocol (MCP) represents a significant advancement in AI-tool integration, enabling seamless communication between AI agents and external services. However, this connectivity introduces novel attack vectors that remain largely unexplored. This paper demonstrates how unsophisticated threat actors, requiring only basic programming skills and free web tools, can exploit MCP's trust model to exfiltrate sensitive financial data. We present a proof-of-concept attack where a malicious weather MCP server, disguised as benign functionality, discovers and exploits legitimate banking tools to steal user account balances. The attack chain requires no advanced technical knowledge, server infrastructure, or monetary investment. The findings reveal a critical security gap in the emerging MCP ecosystem: while individual servers may appear trustworthy, their combination creates unexpected cross-server attack surfaces. Unlike traditional cybersecurity threats that assume sophisticated adversaries, our research shows that the barrier to entry for MCP-based attacks is alarmingly low. A threat actor with undergraduate-level Python knowledge can craft convincing social engineering attacks that exploit the implicit trust relationships MCP establishes between AI agents and tool providers. This work contributes to the nascent field of MCP security by demonstrating that current MCP implementations allow trivial cross-server attacks and proposing both immediate mitigations and protocol improvements to secure this emerging ecosystem. 

**Abstract (ZH)**: MCP安全研究：新兴MCP生态系统中的简单跨服务器攻击 

---
# RaGS: Unleashing 3D Gaussian Splatting from 4D Radar and Monocular Cues for 3D Object Detection 

**Title (ZH)**: RaGS: 从4D雷达和单目线索释放3D高斯采样以进行3D物体检测 

**Authors**: Xiaokai Bai, Chenxu Zhou, Lianqing Zheng, Si-Yuan Cao, Jianan Liu, Xiaohan Zhang, Zhengzhuang Zhang, Hui-liang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.19856)  

**Abstract**: 4D millimeter-wave radar has emerged as a promising sensor for autonomous driving, but effective 3D object detection from both 4D radar and monocular images remains a challenge. Existing fusion approaches typically rely on either instance-based proposals or dense BEV grids, which either lack holistic scene understanding or are limited by rigid grid structures. To address these, we propose RaGS, the first framework to leverage 3D Gaussian Splatting (GS) as representation for fusing 4D radar and monocular cues in 3D object detection. 3D GS naturally suits 3D object detection by modeling the scene as a field of Gaussians, dynamically allocating resources on foreground objects and providing a flexible, resource-efficient solution. RaGS uses a cascaded pipeline to construct and refine the Gaussian field. It starts with the Frustum-based Localization Initiation (FLI), which unprojects foreground pixels to initialize coarse 3D Gaussians positions. Then, the Iterative Multimodal Aggregation (IMA) fuses semantics and geometry, refining the limited Gaussians to the regions of interest. Finally, the Multi-level Gaussian Fusion (MGF) renders the Gaussians into multi-level BEV features for 3D object detection. By dynamically focusing on sparse objects within scenes, RaGS enable object concentrating while offering comprehensive scene perception. Extensive experiments on View-of-Delft, TJ4DRadSet, and OmniHD-Scenes benchmarks demonstrate its state-of-the-art performance. Code will be released. 

**Abstract (ZH)**: 4D毫米波雷达已成为自主驾驶领域有前景的传感器，但在从4D雷达和单目图像中进行有效的3D对象检测方面仍存在挑战。现有融合方法通常依赖于实例级提议或密集的鸟瞰图网格，要么缺乏全局场景理解，要么受限于刚性的网格结构。为了解决这些问题，我们提出了RaGS，这是首个利用3D高斯斑图化（GS）作为表示方法，将4D雷达和单目视觉线索融合到3D对象检测中的框架。3D GS自然适用于3D对象检测，通过将场景建模为高斯场，动态分配资源到前景对象上，并提供一种灵活且资源高效的解决方案。RaGS采用级联流水线来构建和精炼高斯场。首先，通过束基定位初始化（FLI），将前景像素反投影以初始化粗略的3D高斯位置。然后，通过迭代多模态聚合（IMA）融合语义和几何信息，精炼有限的高斯到感兴趣区域。最后，多级高斯融合（MGF）将高斯渲染为多级BEV特征，用于3D对象检测。通过动态聚焦场景中的稀疏对象，RaGS实现了对象集中化的同时提供了全面的场景感知。在View-of-Delft、TJ4DRadSet和OmniHD-Scenes基准测试上进行的广泛实验展示了其优越性能。代码将开源。 

---
# Agentic Reinforced Policy Optimization 

**Title (ZH)**: 代理强化策略优化 

**Authors**: Guanting Dong, Hangyu Mao, Kai Ma, Licheng Bao, Yifei Chen, Zhongyuan Wang, Zhongxia Chen, Jiazhen Du, Huiyang Wang, Fuzheng Zhang, Guorui Zhou, Yutao Zhu, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2507.19849)  

**Abstract**: Large-scale reinforcement learning with verifiable rewards (RLVR) has demonstrated its effectiveness in harnessing the potential of large language models (LLMs) for single-turn reasoning tasks. In realistic reasoning scenarios, LLMs can often utilize external tools to assist in task-solving processes. However, current RL algorithms inadequately balance the models' intrinsic long-horizon reasoning capabilities and their proficiency in multi-turn tool interactions. To bridge this gap, we propose Agentic Reinforced Policy Optimization (ARPO), a novel agentic RL algorithm tailored for training multi-turn LLM-based agents. Through preliminary experiments, we observe that LLMs tend to exhibit highly uncertain behavior, characterized by an increase in the entropy distribution of generated tokens, immediately following interactions with external tools. Motivated by this observation, ARPO incorporates an entropy-based adaptive rollout mechanism, dynamically balancing global trajectory sampling and step-level sampling, thereby promoting exploration at steps with high uncertainty after tool usage. By integrating an advantage attribution estimation, ARPO enables LLMs to internalize advantage differences in stepwise tool-use interactions. Our experiments across 13 challenging benchmarks in computational reasoning, knowledge reasoning, and deep search domains demonstrate ARPO's superiority over trajectory-level RL algorithms. Remarkably, ARPO achieves improved performance using only half of the tool-use budget required by existing methods, offering a scalable solution for aligning LLM-based agents with real-time dynamic environments. Our code and datasets are released at this https URL 

**Abstract (ZH)**: 可验证奖励的大规模强化学习（RLVR）在利用大型语言模型（LLMs）进行单轮推理任务方面展示了其有效性。在现实推理场景中，LLMs通常可以利用外部工具来辅助任务解决过程。然而，当前的RL算法未能有效地平衡模型的固有长时推理能力与其在多轮工具交互方面的熟练程度。为了解决这一问题，我们提出了Agent Reinforced Policy Optimization (ARPO)，一种专门用于训练基于多轮LLM的代理的新型代理RL算法。初步实验表明，LLMs在与外部工具交互后倾向于表现出高度不确定的行为，表现为生成token的熵分布增加。受此观察的启发，ARPO引入了一种基于熵的自适应展开机制，动态平衡全局轨迹采样和步骤级采样，从而在工具使用后的高不确定步骤中促进探索。通过结合优势归因估计，ARPO使LLMs能够内化步骤级工具使用交互中的优势差异。我们在包括13个具有挑战性的计算推理、知识推理和深度搜索领域的基准测试中，显示出ARPO优于轨迹级RL算法的优势。值得注意的是，ARPO仅使用现有方法所需的一半工具使用预算就能实现更好的性能，提供了一种可扩展的解决方案，用于使LLM基代理与实时动态环境对齐。我们的代码和数据集在此处发布。 

---
# VAE-GAN Based Price Manipulation in Coordinated Local Energy Markets 

**Title (ZH)**: 基于VAE-GAN的价格操纵在协调本地能源市场中 

**Authors**: Biswarup Mukherjee, Li Zhou, S. Gokul Krishnan, Milad Kabirifar, Subhash Lakshminarayana, Charalambos Konstantinou  

**Link**: [PDF](https://arxiv.org/pdf/2507.19844)  

**Abstract**: This paper introduces a model for coordinating prosumers with heterogeneous distributed energy resources (DERs), participating in the local energy market (LEM) that interacts with the market-clearing entity. The proposed LEM scheme utilizes a data-driven, model-free reinforcement learning approach based on the multi-agent deep deterministic policy gradient (MADDPG) framework, enabling prosumers to make real-time decisions on whether to buy, sell, or refrain from any action while facilitating efficient coordination for optimal energy trading in a dynamic market. In addition, we investigate a price manipulation strategy using a variational auto encoder-generative adversarial network (VAE-GAN) model, which allows utilities to adjust price signals in a way that induces financial losses for the prosumers. Our results show that under adversarial pricing, heterogeneous prosumer groups, particularly those lacking generation capabilities, incur financial losses. The same outcome holds across LEMs of different sizes. As the market size increases, trading stabilizes and fairness improves through emergent cooperation among agents. 

**Abstract (ZH)**: 本文提出了一种协调具有异质分布式能源资源的产消者模型，使其能够参与与市场出清实体互动的本地能源市场（LEM）。提出的LEM方案基于多代理深度确定性策略Gradient（MADDPG）框架，采用数据驱动、无模型强化学习方法，使产消者能够实时决定是否买入、卖出或不采取任何行动，从而在动态市场中实现高效协调和优化能源交易。此外，本文还探讨了一种使用变分自动编码器生成对抗网络（VAE-GAN）模型的价格操纵策略，该策略使供电公司能够以诱导产消者财务损失的方式调整价格信号。结果显示，在对抗性定价下，尤其是缺乏发电能力的异质产消者群体会遭受财务损失。这一结果在不同规模的LEM中保持一致。随着市场规模的扩大，交易趋于稳定，公平性通过代理之间自发的合作而提高。 

---
# A Cooperative Approach for Knowledge-based Business Process Design in a Public Authority 

**Title (ZH)**: 基于知识的合作式公共机构业务流程设计方法 

**Authors**: Mohammad Azarijafari, Luisa Mich, Michele Missikoff, Oleg Missikoff  

**Link**: [PDF](https://arxiv.org/pdf/2507.19842)  

**Abstract**: Enterprises are currently undergoing profound transformations due to the unpostponable digital transformation. Then, to remain competitive, enterprises must adapt their organisational structures and operations. This organisational shift is also important for small and medium-sized enterprises. A key innovation frontier is the adoption of process-oriented production models. This paper presents a knowledge-based method to support business experts in designing business processes. The method requires no prior expertise in Knowledge Engineering and guides designers through a structured sequence of steps to produce a diagrammatic workflow of the target process. The construction of the knowledge base starts from simple, text-based, knowledge artefacts and then progresses towards more structured, formal representations. The approach has been conceived to allow a shared approach for all stakeholders and actors who participate in the BP design. 

**Abstract (ZH)**: 企业正经历不可推迟的数字化转型，从而引发深刻变革。为了保持竞争力，企业必须适应其组织结构和运营。这一组织变革对于中小企业也同样重要。一个关键的创新前沿是采用面向过程的生产模型。本文提出了一种基于知识的方法，以支持业务专家设计业务流程。该方法无需任何先验的知识工程知识，并引导设计者通过结构化的步骤序列来生成目标过程的图示工作流。知识库的构建从简单的文本知识 artefacts 开始，逐步发展到更结构化、形式化的表示。该方法旨在让所有参与 BP 设计的利益相关者和参与者能够共享这一方法。 

---
# AutoSign: Direct Pose-to-Text Translation for Continuous Sign Language Recognition 

**Title (ZH)**: AutoSign: 直接姿态到文本的翻译用于连续手语识别 

**Authors**: Samuel Ebimobowei Johnny, Blessed Guda, Andrew Blayama Stephen, Assane Gueye  

**Link**: [PDF](https://arxiv.org/pdf/2507.19840)  

**Abstract**: Continuously recognizing sign gestures and converting them to glosses plays a key role in bridging the gap between the hearing and hearing-impaired communities. This involves recognizing and interpreting the hands, face, and body gestures of the signer, which pose a challenge as it involves a combination of all these features. Continuous Sign Language Recognition (CSLR) methods rely on multi-stage pipelines that first extract visual features, then align variable-length sequences with target glosses using CTC or HMM-based approaches. However, these alignment-based methods suffer from error propagation across stages, overfitting, and struggle with vocabulary scalability due to the intermediate gloss representation bottleneck. To address these limitations, we propose AutoSign, an autoregressive decoder-only transformer that directly translates pose sequences to natural language text, bypassing traditional alignment mechanisms entirely. The use of this decoder-only approach allows the model to directly map between the features and the glosses without the need for CTC loss while also directly learning the textual dependencies in the glosses. Our approach incorporates a temporal compression module using 1D CNNs to efficiently process pose sequences, followed by AraGPT2, a pre-trained Arabic decoder, to generate text (glosses). Through comprehensive ablation studies, we demonstrate that hand and body gestures provide the most discriminative features for signer-independent CSLR. By eliminating the multi-stage pipeline, AutoSign achieves substantial improvements on the Isharah-1000 dataset, achieving an improvement of up to 6.1\% in WER score compared to the best existing method. 

**Abstract (ZH)**: 连续识别手语手势并将其转换为手语词汇在填补听力与听力受损社区之间的差距中起着关键作用。这一过程涉及识别和解释手语者的手、面部和身体手势，这对连续手语识别（CSLR）方法构成了挑战，因为它需要处理这些特征的组合。连续手语识别方法依赖于多阶段管道，首先提取视觉特征，然后使用CTC或HMM方法对不同长度的序列与目标手语词汇进行对齐。然而，这些基于对齐的方法容易出现误差传播、过拟合，并且由于中间手语词汇表示瓶颈，在词汇量扩张方面存在问题。为了解决这些限制，我们提出了AutoSign，一种自回归解码器型变压器，可以直接将姿态序列翻译为自然语言文本，完全绕过了传统的对齐机制。使用这种解码器型方法，模型可以直接在特征和手语词汇之间建立映射，无需CTC损失，同时直接学习手语词汇中的文本依赖关系。我们的方法采用了一种使用1D CNN的时序压缩模块，以高效处理姿态序列，随后使用预训练的阿拉伯语解码器AraGPT2生成文本（手语词汇）。通过全面的消融研究，我们证明了手部和身体手势为独立于手语者的目标手上提供了最具区分性的特征。通过消除多阶段管道，AutoSign在Isharah-1000数据集中实现了显著的改进，相较于现有最佳方法，词错误率（WER）分数提高了6.1%。 

---
# ChoreoMuse: Robust Music-to-Dance Video Generation with Style Transfer and Beat-Adherent Motion 

**Title (ZH)**: ChoreoMuse：基于风格转换和节拍适配的 robust 音乐到舞蹈视频生成 

**Authors**: Xuanchen Wang, Heng Wang, Weidong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2507.19836)  

**Abstract**: Modern artistic productions increasingly demand automated choreography generation that adapts to diverse musical styles and individual dancer characteristics. Existing approaches often fail to produce high-quality dance videos that harmonize with both musical rhythm and user-defined choreography styles, limiting their applicability in real-world creative contexts. To address this gap, we introduce ChoreoMuse, a diffusion-based framework that uses SMPL format parameters and their variation version as intermediaries between music and video generation, thereby overcoming the usual constraints imposed by video resolution. Critically, ChoreoMuse supports style-controllable, high-fidelity dance video generation across diverse musical genres and individual dancer characteristics, including the flexibility to handle any reference individual at any resolution. Our method employs a novel music encoder MotionTune to capture motion cues from audio, ensuring that the generated choreography closely follows the beat and expressive qualities of the input music. To quantitatively evaluate how well the generated dances match both musical and choreographic styles, we introduce two new metrics that measure alignment with the intended stylistic cues. Extensive experiments confirm that ChoreoMuse achieves state-of-the-art performance across multiple dimensions, including video quality, beat alignment, dance diversity, and style adherence, demonstrating its potential as a robust solution for a wide range of creative applications. Video results can be found on our project page: this https URL. 

**Abstract (ZH)**: 现代艺术创作日益需求能够适应多样音乐风格和个体舞者特性的自动化编舞生成。现有方法往往无法生成与音乐节奏和用户定义的编舞风格和谐一致的高质量舞蹈视频，限制了其在实际创作环境中的应用。为解决这一问题，我们引入了ChoreoMuse，这是一种基于发散模型的框架，利用SMPL格式参数及其变体作为音乐与视频生成之间的中介，从而克服了传统视频分辨率限制。关键的是，ChoreoMuse支持跨多样音乐流派和个体舞者特性的风格可控、高保真舞蹈视频生成，包括处理任意分辨率参考个体的能力。我们的方法采用新型音乐编码器MotionTune捕获音频中的运动线索，确保生成的编舞紧密跟随输入音乐的节奏和表达特质。为了定量评估生成舞蹈与音乐和编舞风格的匹配程度，我们引入了两个新的度量标准，用于衡量风格意图的对齐情况。大量实验表明，ChoreoMuse在视频质量、节奏对齐、舞蹈多样性和风格遵循方面均达到最先进的性能，展示了其在多种创意应用中的潜在优势。视频结果可在我们的项目页面上找到：this https URL。 

---
# HCAttention: Extreme KV Cache Compression via Heterogeneous Attention Computing for LLMs 

**Title (ZH)**: HCAttention: 极端键值缓存压缩的异质注意计算方法用于大型语言模型 

**Authors**: Dongquan Yang, Yifan Yang, Xiaotian Yu, Xianbiao Qi, Rong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2507.19823)  

**Abstract**: Processing long-context inputs with large language models presents a significant challenge due to the enormous memory requirements of the Key-Value (KV) cache during inference. Existing KV cache compression methods exhibit noticeable performance degradation when memory is reduced by more than 85%. Additionally, strategies that leverage GPU-CPU collaboration for approximate attention remain underexplored in this setting. We propose HCAttention, a heterogeneous attention computation framework that integrates key quantization, value offloading, and dynamic KV eviction to enable efficient inference under extreme memory constraints. The method is compatible with existing transformer architectures and does not require model fine-tuning. Experimental results on the LongBench benchmark demonstrate that our approach preserves the accuracy of full-attention model while shrinking the KV cache memory footprint to 25% of its original size. Remarkably, it stays competitive with only 12.5% of the cache, setting a new state-of-the-art in LLM KV cache compression. To the best of our knowledge, HCAttention is the first to extend the Llama-3-8B model to process 4 million tokens on a single A100 GPU with 80GB memory. 

**Abstract (ZH)**: 用大规模语言模型处理长上下文输入面临着显著挑战，因为在推理过程中，关键值（KV）缓存需要大量的内存。现有的KV缓存压缩方法在内存缩减超过85%时会出现明显的性能下降。此外，利用GPU-CPU协同进行近似注意机制的策略在这个场景下仍处于探索之中。我们提出了HCAttention，这是一种异构注意计算框架，结合了键的量化、值的卸载和动态KV淘汰，以在极端内存约束下实现高效的推理。该方法与现有的变压器架构兼容，不需要对模型进行微调。在LongBench基准测试上的实验结果表明，我们的方法在保持全注意模型准确性的基础上，将KV缓存的内存占用缩减至原大小的25%。更值得注意的是，仅使用12.5%的缓存时，它仍然具有竞争力，创下了新的语言模型KV缓存压缩的最先进技术状态。据我们所知，HCAttention是首个将Llama-3-8B模型扩展到单个具有80GB内存的A100 GPU上处理400万令牌的方案。 

---
# From Few-Label to Zero-Label: An Approach for Cross-System Log-Based Anomaly Detection with Meta-Learning 

**Title (ZH)**: 从少量标签到零标签：一种基于元学习的日志跨系统异常检测方法 

**Authors**: Xinlong Zhao, Tong Jia, Minghua He, Yihan Wu, Ying Li, Gang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19806)  

**Abstract**: Log anomaly detection plays a critical role in ensuring the stability and reliability of software systems. However, existing approaches rely on large amounts of labeled log data, which poses significant challenges in real-world applications. To address this issue, cross-system transfer has been identified as a key research direction. State-of-the-art cross-system approaches achieve promising performance with only a few labels from the target system. However, their reliance on labeled target logs makes them susceptible to the cold-start problem when labeled logs are insufficient. To overcome this limitation, we explore a novel yet underexplored setting: zero-label cross-system log anomaly detection, where the target system logs are entirely unlabeled. To this end, we propose FreeLog, a system-agnostic representation meta-learning method that eliminates the need for labeled target system logs, enabling cross-system log anomaly detection under zero-label conditions. Experimental results on three public log datasets demonstrate that FreeLog achieves performance comparable to state-of-the-art methods that rely on a small amount of labeled data from the target system. 

**Abstract (ZH)**: 无标签跨系统日志异常检测在确保软件系统稳定性和可靠性中发挥关键作用。现有的方法依赖大量标注日志数据，这在实际应用中提出了重大挑战。为解决这一问题，跨系统迁移已成为关键研究方向。最先进的跨系统方法仅依靠目标系统少量标注日志就能取得令人鼓舞的性能。然而，它们对目标系统标注日志的依赖性使其在标注日志不足时面临冷启动问题。为克服这一局限性，我们探索了一个新颖且尚未充分研究的设置：无标签跨系统日志异常检测，其中目标系统日志完全未标注。为此，我们提出FreeLog，一个系统无关的表示元学习方法，无需目标系统标注日志即可实现无标签跨系统日志异常检测。在三个公开的日志数据集上的实验结果表明，FreeLog 在性能上与依赖目标系统少量标注数据的最先进方法相当。 

---
# AI-Based Clinical Rule Discovery for NMIBC Recurrence through Tsetlin Machines 

**Title (ZH)**: 基于Tsetlin机的AI临床规则发现用于NMIBC复发 

**Authors**: Saram Abbas, Naeem Soomro, Rishad Shafik, Rakesh Heer, Kabita Adhikari  

**Link**: [PDF](https://arxiv.org/pdf/2507.19803)  

**Abstract**: Bladder cancer claims one life every 3 minutes worldwide. Most patients are diagnosed with non-muscle-invasive bladder cancer (NMIBC), yet up to 70% recur after treatment, triggering a relentless cycle of surgeries, monitoring, and risk of progression. Clinical tools like the EORTC risk tables are outdated and unreliable - especially for intermediate-risk cases.
We propose an interpretable AI model using the Tsetlin Machine (TM), a symbolic learner that outputs transparent, human-readable logic. Tested on the PHOTO trial dataset (n=330), TM achieved an F1-score of 0.80, outperforming XGBoost (0.78), Logistic Regression (0.60), and EORTC (0.42). TM reveals the exact clauses behind each prediction, grounded in clinical features like tumour count, surgeon experience, and hospital stay - offering accuracy and full transparency. This makes TM a powerful, trustworthy decision-support tool ready for real-world adoption. 

**Abstract (ZH)**: 全球每3分钟就有1人因膀胱癌去世。大多数患者被诊断为非肌层浸润性膀胱癌（NMIBC），但多达70%的患者在接受治疗后会出现复发，导致持续的手术循环、监测和疾病进展的风险。临床工具如EORTC风险表已过时且不可靠，尤其是在处理中等风险病例时。
我们提出了一种基于Tsetlin机（TM）的可解释人工智能模型，Tsetlin机是一种符号学习器，输出透明且易于理解的逻辑规则。该模型在PHOTO试验数据集（n=330）上测试，F1分数达到了0.80，超过了XGBoost（0.78）、逻辑回归（0.60）和EORTC（0.42）。TM能够揭示每个预测背后的精确语句，这些语句基于临床特征，如肿瘤数量、外科医生经验以及住院时间，从而提供准确性和完全透明度。这使得TM成为一种强大的、值得信赖的决策支持工具，适合实际应用。 

---
# Large Language Model Agent for Structural Drawing Generation Using ReAct Prompt Engineering and Retrieval Augmented Generation 

**Title (ZH)**: 使用ReAct提示工程和检索增强生成的结构绘图大型语言模型代理 

**Authors**: Xin Zhang, Lissette Iturburu, Juan Nicolas Villamizar, Xiaoyu Liu, Manuel Salmeron, Shirley J.Dyke, Julio Ramirez  

**Link**: [PDF](https://arxiv.org/pdf/2507.19771)  

**Abstract**: Structural drawings are widely used in many fields, e.g., mechanical engineering, civil engineering, etc. In civil engineering, structural drawings serve as the main communication tool between architects, engineers, and builders to avoid conflicts, act as legal documentation, and provide a reference for future maintenance or evaluation needs. They are often organized using key elements such as title/subtitle blocks, scales, plan views, elevation view, sections, and detailed sections, which are annotated with standardized symbols and line types for interpretation by engineers and contractors. Despite advances in software capabilities, the task of generating a structural drawing remains labor-intensive and time-consuming for structural engineers. Here we introduce a novel generative AI-based method for generating structural drawings employing a large language model (LLM) agent. The method incorporates a retrieval-augmented generation (RAG) technique using externally-sourced facts to enhance the accuracy and reliability of the language model. This method is capable of understanding varied natural language descriptions, processing these to extract necessary information, and generating code to produce the desired structural drawing in AutoCAD. The approach developed, demonstrated and evaluated herein enables the efficient and direct conversion of a structural drawing's natural language description into an AutoCAD drawing, significantly reducing the workload compared to current working process associated with manual drawing production, facilitating the typical iterative process of engineers for expressing design ideas in a simplified way. 

**Abstract (ZH)**: 结构图在机械工程、土木工程等多个领域广泛应用。在土木工程中，结构图是建筑师、工程师和施工人员之间主要的沟通工具，用于避免冲突、作为法律文件使用，并为未来的维修或评估提供参考。它们通常使用标题/副标题块、比例尺、平面图、立面图、剖面图和详图等关键要素进行组织，并用标准化符号和线条类型进行标注，以便工程师和承包商进行解读。尽管软件能力不断提高，但生成结构图的任务仍对结构工程师来说劳动密集且耗时。这里介绍一种基于生成式AI的方法，利用大型语言模型（LLM）代理生成结构图，该方法结合了检索增强生成（RAG）技术，利用外部事实增强语言模型的准确性和可靠性。该方法能够理解各种自然语言描述，处理这些描述以提取必要信息，并生成代码以在AutoCAD中生成所需的结构图。本文中开发的方法不仅展示了其实现过程，还对其进行了评估，显著减少了与手工绘制相关的繁琐工作，简化了工程师的设计表达过程。 

---
# UloRL:An Ultra-Long Output Reinforcement Learning Approach for Advancing Large Language Models' Reasoning Abilities 

**Title (ZH)**: UloRL：一种促进大型语言模型推理能力提升的超长输出强化学习方法 

**Authors**: Dong Du, Shulin Liu, Tao Yang, Shaohua Chen, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.19766)  

**Abstract**: Recent advances in large language models (LLMs) have highlighted the potential of reinforcement learning with verifiable rewards (RLVR) to enhance reasoning capabilities through extended output sequences. However, traditional RL frameworks face inefficiencies when handling ultra-long outputs due to long-tail sequence distributions and entropy collapse during training. To address these challenges, we propose an Ultra-Long Output Reinforcement Learning (UloRL) approach for advancing large language models' reasoning abilities. Specifically, we divide ultra long output decoding into short segments, enabling efficient training by mitigating delays caused by long-tail samples. Additionally, we introduce dynamic masking of well-Mastered Positive Tokens (MPTs) to prevent entropy collapse. Experimental results demonstrate the effectiveness of our approach. On the Qwen3-30B-A3B model, RL with segment rollout achieved 2.06x increase in training speed, while RL training with 128k-token outputs improves the model's performance on AIME2025 from 70.9\% to 85.1\% and on BeyondAIME from 50.7\% to 61.9\%, even surpassing Qwen3-235B-A22B with remarkable gains. These findings underscore the potential of our methods to advance the reasoning capabilities of LLMs with ultra-long sequence generation. We will release our code and model for further use by the community. 

**Abstract (ZH)**: Recent Advances in Large Language Models Through Ultra-Long Output Reinforcement Learning with Verifiable Rewards 

---
# Modeling enzyme temperature stability from sequence segment perspective 

**Title (ZH)**: 从序列片段视角建模酶的温度稳定性 

**Authors**: Ziqi Zhang, Shiheng Chen, Runze Yang, Zhisheng Wei, Wei Zhang, Lei Wang, Zhanzhi Liu, Fengshan Zhang, Jing Wu, Xiaoyong Pan, Hongbin Shen, Longbing Cao, Zhaohong Deng  

**Link**: [PDF](https://arxiv.org/pdf/2507.19755)  

**Abstract**: Developing enzymes with desired thermal properties is crucial for a wide range of industrial and research applications, and determining temperature stability is an essential step in this process. Experimental determination of thermal parameters is labor-intensive, time-consuming, and costly. Moreover, existing computational approaches are often hindered by limited data availability and imbalanced distributions. To address these challenges, we introduce a curated temperature stability dataset designed for model development and benchmarking in enzyme thermal modeling. Leveraging this dataset, we present the \textit{Segment Transformer}, a novel deep learning framework that enables efficient and accurate prediction of enzyme temperature stability. The model achieves state-of-the-art performance with an RMSE of 24.03, MAE of 18.09, and Pearson and Spearman correlations of 0.33, respectively. These results highlight the effectiveness of incorporating segment-level representations, grounded in the biological observation that different regions of a protein sequence contribute unequally to thermal behavior. As a proof of concept, we applied the Segment Transformer to guide the engineering of a cutinase enzyme. Experimental validation demonstrated a 1.64-fold improvement in relative activity following heat treatment, achieved through only 17 mutations and without compromising catalytic function. 

**Abstract (ZH)**: 开发具有desired thermal properties的酶对于广泛的应用领域至关重要，确定温度稳定性是这一过程中的一个关键步骤。实验测定热参数劳动密集、耗时且成本高。此外，现有的计算方法往往受限于数据可用性的有限以及数据分布的不平衡。为了解决这些挑战，我们引入了一个精心编撰的温度稳定性数据集，该数据集适用于酶热学建模的模型开发和基准测试。基于该数据集，我们提出了一种名为\textit{Segment Transformer}的新型深度学习框架，该框架能够高效且准确地预测酶的温度稳定性。该模型在均方根误差(RMSE)为24.03，平均绝对误差(RMAE)为18.09，皮尔逊相关系数和 SPEARMAN相关系数分别为0.33的情况下达到了业界最佳性能。这些结果突显了在模型中纳入段级表示的有效性，这基于生物学观察，即蛋白质序列的不同区域对热行为的贡献是不均等的。作为概念验证，我们应用\textit{Segment Transformer}来指导切割酶的工程改造。实验验证表明，通过仅17个突变并在不损害催化功能的情况下，热处理后相对活性提高了1.64倍。 

---
# Defining ethically sourced code generation 

**Title (ZH)**: 定义伦理采集的代码生成 

**Authors**: Zhuolin Xu, Chenglin Li, Qiushi Li, Shin Hwei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2507.19743)  

**Abstract**: Several code generation models have been proposed to help reduce time and effort in solving software-related tasks. To ensure responsible AI, there are growing interests over various ethical issues (e.g., unclear licensing, privacy, fairness, and environment impact). These studies have the overarching goal of ensuring ethically sourced generation, which has gained growing attentions in speech synthesis and image generation. In this paper, we introduce the novel notion of Ethically Sourced Code Generation (ES-CodeGen) to refer to managing all processes involved in code generation model development from data collection to post-deployment via ethical and sustainable practices. To build a taxonomy of ES-CodeGen, we perform a two-phase literature review where we read 803 papers across various domains and specific to AI-based code generation. We identified 71 relevant papers with 10 initial dimensions of ES-CodeGen. To refine our dimensions and gain insights on consequences of ES-CodeGen, we surveyed 32 practitioners, which include six developers who submitted GitHub issues to opt-out from the Stack dataset (these impacted users have real-world experience of ethically sourcing issues in code generation models). The results lead to 11 dimensions of ES-CodeGen with a new dimension on code quality as practitioners have noted its importance. We also identified consequences, artifacts, and stages relevant to ES-CodeGen. Our post-survey reflection showed that most practitioners tend to ignore social-related dimensions despite their importance. Most practitioners either agreed or strongly agreed that our survey help improve their understanding of ES-CodeGen. Our study calls for attentions of various ethical issues towards ES-CodeGen. 

**Abstract (ZH)**: 基于伦理的代码生成（ES-CodeGen）：管理代码生成模型开发全过程的伦理和可持续实践 

---
# Predicting Human Mobility in Disasters via LLM-Enhanced Cross-City Learning 

**Title (ZH)**: 通过LLM增强的跨城市学习预测灾害中的人类 mobility 

**Authors**: Yinzhou Tang, Huandong Wang, Xiaochen Fan, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.19737)  

**Abstract**: The vulnerability of cities to natural disasters has increased with urbanization and climate change, making it more important to predict human mobility in the disaster scenarios for downstream tasks including location-based early disaster warning and pre-allocating rescue resources, etc. However, existing human mobility prediction models are mainly designed for normal scenarios, and fail to adapt to disaster scenarios due to the shift of human mobility patterns under disaster. To address this issue, we introduce \textbf{DisasterMobLLM}, a mobility prediction framework for disaster scenarios that can be integrated into existing deep mobility prediction methods by leveraging LLMs to model the mobility intention and transferring the common knowledge of how different disasters affect mobility intentions between cities. This framework utilizes a RAG-Enhanced Intention Predictor to forecast the next intention, refines it with an LLM-based Intention Refiner, and then maps the intention to an exact location using an Intention-Modulated Location Predictor. Extensive experiments illustrate that DisasterMobLLM can achieve a 32.8\% improvement in terms of Acc@1 and a 35.0\% improvement in terms of the F1-score of predicting immobility compared to the baselines. The code is available at this https URL. 

**Abstract (ZH)**: 城市在自然灾难中的脆弱性随着城市化进程和气候变化而增加，因此在灾难情景下预测人类移动性以进行基于位置的早期灾害预警和预先分配救援资源等下游任务变得更加重要。然而，现有的人类移动性预测模型主要针对正常情景设计，因在灾难情景下人类移动性模式的转变而无法适应。为了解决这一问题，我们引入了DisasterMobLLM，这是一种利用LLM建模移动意图并利用不同城市在不同灾害下影响移动意图的通用知识来整合到现有深度移动性预测方法中的灾难情景下移动性预测框架。该框架利用RAG增强意图预测器来预测下一个意图，使用基于LLM的意图精炼器对其进行精炼，然后使用意图调制位置预测器将意图映射到确切的位置。广泛的实验表明，DisasterMobLLM在Acc@1上可以实现32.8%的改进，在预测不移动性方面的F1分数上可以实现35.0%的改进，相较于基线方法。代码可在以下网址获得。 

---
# Quaternion-Based Robust PCA for Efficient Moving Target Detection and Background Recovery in Color Videos 

**Title (ZH)**: 基于四元数的鲁棒PCA方法在彩色视频中高效的目标检测与背景恢复 

**Authors**: Liyang Wang, Shiqian Wu, Shun Fang, Qile Zhu, Jiaxin Wu, Sos Again  

**Link**: [PDF](https://arxiv.org/pdf/2507.19730)  

**Abstract**: Moving target detection is a challenging computer vision task aimed at generating accurate segmentation maps in diverse in-the-wild color videos captured by static cameras. If backgrounds and targets can be simultaneously extracted and recombined, such synthetic data can significantly enrich annotated in-the-wild datasets and enhance the generalization ability of deep models. Quaternion-based RPCA (QRPCA) is a promising unsupervised paradigm for color image processing. However, in color video processing, Quaternion Singular Value Decomposition (QSVD) incurs high computational costs, and rank-1 quaternion matrix fails to yield rank-1 color channels. In this paper, we reduce the computational complexity of QSVD to o(1) by utilizing a quaternion Riemannian manifold. Furthermor, we propose the universal QRPCA (uQRPCA) framework, which achieves a balance in simultaneously segmenting targets and recovering backgrounds from color videos. Moreover, we expand to uQRPCA+ by introducing the Color Rank-1 Batch (CR1B) method to further process and obtain the ideal low-rank background across color channels. Experiments demonstrate our uQRPCA+ achieves State Of The Art (SOTA) performance on moving target detection and background recovery tasks compared to existing open-source methods. Our implementation is publicly available on GitHub at this https URL 

**Abstract (ZH)**: 彩色视频中的移动目标检测是一项具有挑战性的计算机视觉任务，旨在生成 diverse in-the-wild 颜色视频中静态摄像头捕获的准确分割图。如果背景和目标能够同时提取和重组，此类合成数据可以显著丰富标注的在野数据集，并增强深度模型的泛化能力。四元数基于 RPCA (QRPCA) 是一种有前景的无监督彩色图像处理范式。但在彩色视频处理中，四元数奇异值分解 (QSVD) 引起高计算成本，且 rank-1 四元数矩阵无法产生 rank-1 彩色通道。本文通过利用四元数黎曼流形将 QSVD 的计算复杂度降低到 o(1)。此外，我们提出了一种通用的 QRPCA (uQRPCA) 框架，该框架能够在彩色视频中同时分割目标和恢复背景时取得平衡。进一步地，通过引入彩色秩-1 批处理 (CR1B) 方法，我们拓展了 uQRPCA+，以进一步处理并获得理想低秩背景。实验表明，我们的 uQRPCA+ 在移动目标检测和背景恢复任务上达到了现有开源方法的最优性能。我们的实现已在 GitHub 公开发布。 

---
# Oranits: Mission Assignment and Task Offloading in Open RAN-based ITS using Metaheuristic and Deep Reinforcement Learning 

**Title (ZH)**: Oranits：基于开放RAN的ITS中任务分配与卸载的研究——结合元启发式算法和深度强化学习 

**Authors**: Ngoc Hung Nguyen, Nguyen Van Thieu, Quang-Trung Luu, Anh Tuan Nguyen, Senura Wanasekara, Nguyen Cong Luong, Fatemeh Kavehmadavani, Van-Dinh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2507.19712)  

**Abstract**: In this paper, we explore mission assignment and task offloading in an Open Radio Access Network (Open RAN)-based intelligent transportation system (ITS), where autonomous vehicles leverage mobile edge computing for efficient processing. Existing studies often overlook the intricate interdependencies between missions and the costs associated with offloading tasks to edge servers, leading to suboptimal decision-making. To bridge this gap, we introduce Oranits, a novel system model that explicitly accounts for mission dependencies and offloading costs while optimizing performance through vehicle cooperation. To achieve this, we propose a twofold optimization approach. First, we develop a metaheuristic-based evolutionary computing algorithm, namely the Chaotic Gaussian-based Global ARO (CGG-ARO), serving as a baseline for one-slot optimization. Second, we design an enhanced reward-based deep reinforcement learning (DRL) framework, referred to as the Multi-agent Double Deep Q-Network (MA-DDQN), that integrates both multi-agent coordination and multi-action selection mechanisms, significantly reducing mission assignment time and improving adaptability over baseline methods. Extensive simulations reveal that CGG-ARO improves the number of completed missions and overall benefit by approximately 7.1% and 7.7%, respectively. Meanwhile, MA-DDQN achieves even greater improvements of 11.0% in terms of mission completions and 12.5% in terms of the overall benefit. These results highlight the effectiveness of Oranits in enabling faster, more adaptive, and more efficient task processing in dynamic ITS environments. 

**Abstract (ZH)**: 基于Open RAN的智能交通系统中的任务分配与任务卸载研究 

---
# Ultracoarse Equilibria and Ordinal-Folding Dynamics in Operator-Algebraic Models of Infinite Multi-Agent Games 

**Title (ZH)**: 超粗粒度纳什均衡与算子代数模型中序型折叠动力学在无穷多智能体博弈中的应用 

**Authors**: Faruk Alpay, Hamdi Alakkad, Bugra Kilictas, Taylan Alpay  

**Link**: [PDF](https://arxiv.org/pdf/2507.19694)  

**Abstract**: We develop an operator algebraic framework for infinite games with a continuum of agents and prove that regret based learning dynamics governed by a noncommutative continuity equation converge to a unique quantal response equilibrium under mild regularity assumptions. The framework unifies functional analysis, coarse geometry and game theory by assigning to every game a von Neumann algebra that represents collective strategy evolution. A reflective regret operator within this algebra drives the flow of strategy distributions and its fixed point characterises equilibrium. We introduce the ordinal folding index, a computable ordinal valued metric that measures the self referential depth of the dynamics, and show that it bounds the transfinite time needed for convergence, collapsing to zero on coarsely amenable networks. The theory yields new invariant subalgebra rigidity results, establishes existence and uniqueness of envy free and maximin share allocations in continuum economies, and links analytic properties of regret flows with empirical stability phenomena in large language models. These contributions supply a rigorous mathematical foundation for large scale multi agent systems and demonstrate the utility of ordinal metrics for equilibrium selection. 

**Abstract (ZH)**: 我们开发了一种算子代数框架来研究具有连续代理的无限博弈，并证明由非交换连续方程控制的基于遗憾的学习动力学在温和的正则性假设下收敛到唯一的量子反应均衡。该框架通过将每个博弈赋值为一个冯·诺伊曼代数来统合一功能分析、粗几何学和博弈论，该代数代表集体策略演化。该代数中的反射遗憾算子驱动策略分布的流动，其不动点表征均衡。我们引入了序折叠指数，这是一种可计算的序值度量，用于衡量动力学的自我参照深度，并证明它界定了收敛所需的超时，粗 amen 脉络下归零。该理论提供了新的不变子代数刚性结果，确立了连续经济体中无私分配和最大份额分配的存在性和唯一性，并将遗憾流动的分析性质与大型语言模型中的经验稳定性现象联系起来。这些贡献为大规模多代理系统提供了严格的数学基础，并展示了序度量在均衡选择中的实用性。 

---
# KD-GAT: Combining Knowledge Distillation and Graph Attention Transformer for a Controller Area Network Intrusion Detection System 

**Title (ZH)**: KD-GAT: 结合知识蒸馏和图注意力变换器的控制器区域网络入侵检测系统 

**Authors**: Robert Frenken, Sidra Ghayour Bhatti, Hanqin Zhang, Qadeer Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2507.19686)  

**Abstract**: The Controller Area Network (CAN) protocol is widely adopted for in-vehicle communication but lacks inherent security mechanisms, making it vulnerable to cyberattacks. This paper introduces KD-GAT, an intrusion detection framework that combines Graph Attention Networks (GATs) with knowledge distillation (KD) to enhance detection accuracy while reducing computational complexity. In our approach, CAN traffic is represented as graphs using a sliding window to capture temporal and relational patterns. A multi-layer GAT with jumping knowledge aggregation acting as the teacher model, while a compact student GAT--only 6.32% the size of the teacher--is trained via a two-phase process involving supervised pretraining and knowledge distillation with both soft and hard label supervision. Experiments on three benchmark datasets--Car-Hacking, Car-Survival, and can-train-and-test demonstrate that both teacher and student models achieve strong results, with the student model attaining 99.97% and 99.31% accuracy on Car-Hacking and Car-Survival, respectively. However, significant class imbalance in can-train-and-test has led to reduced performance for both models on this dataset. Addressing this imbalance remains an important direction for future work. 

**Abstract (ZH)**: 基于知识蒸馏的图注意力网络入侵检测框架KD-GAT：应用于车载通信的CAN协议 

---
# Salsa as a Nonverbal Embodied Language -- The CoMPAS3D Dataset and Benchmarks 

**Title (ZH)**: 萨萨拉作为非言语 embodied 语言——CoMPAS3D 数据集与基准 

**Authors**: Bermet Burkanova, Payam Jome Yazdian, Chuxuan Zhang, Trinity Evans, Paige Tuttösí, Angelica Lim  

**Link**: [PDF](https://arxiv.org/pdf/2507.19684)  

**Abstract**: Imagine a humanoid that can safely and creatively dance with a human, adapting to its partner's proficiency, using haptic signaling as a primary form of communication. While today's AI systems excel at text or voice-based interaction with large language models, human communication extends far beyond text-it includes embodied movement, timing, and physical coordination. Modeling coupled interaction between two agents poses a formidable challenge: it is continuous, bidirectionally reactive, and shaped by individual variation. We present CoMPAS3D, the largest and most diverse motion capture dataset of improvised salsa dancing, designed as a challenging testbed for interactive, expressive humanoid AI. The dataset includes 3 hours of leader-follower salsa dances performed by 18 dancers spanning beginner, intermediate, and professional skill levels. For the first time, we provide fine-grained salsa expert annotations, covering over 2,800 move segments, including move types, combinations, execution errors and stylistic elements. We draw analogies between partner dance communication and natural language, evaluating CoMPAS3D on two benchmark tasks for synthetic humans that parallel key problems in spoken language and dialogue processing: leader or follower generation with proficiency levels (speaker or listener synthesis), and duet (conversation) generation. Towards a long-term goal of partner dance with humans, we release the dataset, annotations, and code, along with a multitask SalsaAgent model capable of performing all benchmark tasks, alongside additional baselines to encourage research in socially interactive embodied AI and creative, expressive humanoid motion generation. 

**Abstract (ZH)**: 一种用于人类与类人机器人安全创造性舞蹈交互的触觉信号传输方法及CoMPAS3D即兴萨尔萨舞动捕获数据集 

---
# DeepJIVE: Learning Joint and Individual Variation Explained from Multimodal Data Using Deep Learning 

**Title (ZH)**: DeepJIVE：使用深度学习学习多模态数据中的联合和个体变异解释 

**Authors**: Matthew Drexler, Benjamin Risk, James J Lah, Suprateek Kundu, Deqiang Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2507.19682)  

**Abstract**: Conventional multimodal data integration methods provide a comprehensive assessment of the shared or unique structure within each individual data type but suffer from several limitations such as the inability to handle high-dimensional data and identify nonlinear structures. In this paper, we introduce DeepJIVE, a deep-learning approach to performing Joint and Individual Variance Explained (JIVE). We perform mathematical derivation and experimental validations using both synthetic and real-world 1D, 2D, and 3D datasets. Different strategies of achieving the identity and orthogonality constraints for DeepJIVE were explored, resulting in three viable loss functions. We found that DeepJIVE can successfully uncover joint and individual variations of multimodal datasets. Our application of DeepJIVE to the Alzheimer's Disease Neuroimaging Initiative (ADNI) also identified biologically plausible covariation patterns between the amyloid positron emission tomography (PET) and magnetic resonance (MR) images. In conclusion, the proposed DeepJIVE can be a useful tool for multimodal data analysis. 

**Abstract (ZH)**: DeepJIVE：一种用于执行联合和个体方差解释的深度学习方法 

---
# Efficient Learning for Product Attributes with Compact Multimodal Models 

**Title (ZH)**: 基于紧凑多模态模型的产品属性高效学习 

**Authors**: Mandar Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2507.19679)  

**Abstract**: Image-based product attribute prediction in e-commerce is a crucial task with numerous applications. The supervised fine-tuning of Vision Language Models (VLMs) faces significant scale challenges due to the cost of manual or API based annotation. In this paper, we investigate label-efficient semi-supervised fine-tuning strategies for compact VLMs (2B-3B parameters) that leverage unlabeled product listings through Direct Preference Optimization (DPO). Beginning with a small, API-based, annotated, and labeled set, we first employ PEFT to train low-rank adapter modules. To update the adapter weights with unlabeled data, we generate multiple reasoning-and-answer chains per unlabeled sample and segregate these chains into preferred and dispreferred based on self-consistency. We then fine-tune the model with DPO loss and use the updated model for the next iteration. By using PEFT fine-tuning with DPO, our method achieves efficient convergence with minimal compute overhead. On a dataset spanning twelve e-commerce verticals, DPO-based fine-tuning, which utilizes only unlabeled data, demonstrates a significant improvement over the supervised model. Moreover, experiments demonstrate that accuracy with DPO training improves with more unlabeled data, indicating that a large pool of unlabeled samples can be effectively leveraged to improve performance. 

**Abstract (ZH)**: 基于图像的商品属性预测在电子商务中是一项关键任务，具有广泛的應用。通过直接偏好优化（DPO）利用未标记的产品列表进行紧凑视觉语言模型的半监督高效微调策略研究。 

---
# "X of Information'' Continuum: A Survey on AI-Driven Multi-dimensional Metrics for Next-Generation Networked Systems 

**Title (ZH)**: “信息”连续体中的X：下一代网络系统驱动式多维度度量综述 

**Authors**: Beining Wu, Jun Huang, Shui Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.19657)  

**Abstract**: The development of next-generation networking systems has inherently shifted from throughput-based paradigms towards intelligent, information-aware designs that emphasize the quality, relevance, and utility of transmitted information, rather than sheer data volume. While classical network metrics, such as latency and packet loss, remain significant, they are insufficient to quantify the nuanced information quality requirements of modern intelligent applications, including autonomous vehicles, digital twins, and metaverse environments. In this survey, we present the first comprehensive study of the ``X of Information'' continuum by introducing a systematic four-dimensional taxonomic framework that structures information metrics along temporal, quality/utility, reliability/robustness, and network/communication dimensions. We uncover the increasing interdependencies among these dimensions, whereby temporal freshness triggers quality evaluation, which in turn helps with reliability appraisal, ultimately enabling effective network delivery. Our analysis reveals that artificial intelligence technologies, such as deep reinforcement learning, multi-agent systems, and neural optimization models, enable adaptive, context-aware optimization of competing information quality objectives. In our extensive study of six critical application domains, covering autonomous transportation, industrial IoT, healthcare digital twins, UAV communications, LLM ecosystems, and metaverse settings, we illustrate the revolutionary promise of multi-dimensional information metrics for meeting diverse operational needs. Our survey identifies prominent implementation challenges, including ... 

**Abstract (ZH)**: 下一代网络系统的发展已从根本上从基于吞吐量的范式转向了智能、信息感知的设计，强调传输信息的质量、相关性和实用性，而非单纯的数据量。虽然传统的网络性能指标，如延迟和丢包率依然重要，但它们不足以量化现代智能应用——包括自动驾驶、数字孪生和元宇宙环境——对信息质量的细微需求。在本文综述中，我们首次对“信息连续体”进行了全面研究，通过引入系统的四维分类框架，将信息指标按照时间、质量和实用性、可靠性和鲁棒性、网络和通信维度进行结构化。我们揭示了这些维度之间的不断增强的相互依赖性，其中时间新鲜度触发了质量评估，而质量评估又有助于可靠性评估，最终实现有效的网络交付。我们的分析表明，人工智能技术，如深度强化学习、多智能体系统和神经优化模型，能够实现竞争信息质量目标的自适应和上下文感知优化。在对六个关键应用领域——自动驾驶、工业物联网、医疗数字孪生、无人机通信、预训练语言模型生态系统和元宇宙设置——的广泛研究中，我们展示了多维度信息指标在满足多样化操作需求方面的革命性潜力。我们的综述指出了重要的实施挑战，包括…… 

---
# On the Limitations of Ray-Tracing for Learning-Based RF Tasks in Urban Environments 

**Title (ZH)**: 基于射线追踪技术在城市环境中的无线通信任务学习限制 

**Authors**: Armen Manukyan, Hrant Khachatrian, Edvard Ghukasyan, Theofanis P. Raptis  

**Link**: [PDF](https://arxiv.org/pdf/2507.19653)  

**Abstract**: We study the realism of Sionna v1.0.2 ray-tracing for outdoor cellular links in central Rome. We use a real measurement set of 1,664 user-equipments (UEs) and six nominal base-station (BS) sites. Using these fixed positions we systematically vary the main simulation parameters, including path depth, diffuse/specular/refraction flags, carrier frequency, as well as antenna's properties like its altitude, radiation pattern, and orientation. Simulator fidelity is scored for each base station via Spearman correlation between measured and simulated powers, and by a fingerprint-based k-nearest-neighbor localization algorithm using RSSI-based fingerprints. Across all experiments, solver hyper-parameters are having immaterial effect on the chosen metrics. On the contrary, antenna locations and orientations prove decisive. By simple greedy optimization we improve the Spearman correlation by 5% to 130% for various base stations, while kNN-based localization error using only simulated data as reference points is decreased by one-third on real-world samples, while staying twice higher than the error with purely real data. Precise geometry and credible antenna models are therefore necessary but not sufficient; faithfully capturing the residual urban noise remains an open challenge for transferable, high-fidelity outdoor RF simulation. 

**Abstract (ZH)**: Sionna v1.0.2射线追踪在罗马中央户外蜂窝链路现实性的研究 

---
# Can You Share Your Story? Modeling Clients' Metacognition and Openness for LLM Therapist Evaluation 

**Title (ZH)**: 你能分享你的故事吗？：建模客户的元认知和开放性以评估语言模型 therapists 

**Authors**: Minju Kim, Dongje Yoo, Yeonjun Hwang, Minseok Kang, Namyoung Kim, Minju Gwak, Beong-woo Kwak, Hyungjoo Chae, Harim Kim, Yunjoong Lee, Min Hee Kim, Dayi Jung, Kyong-Mee Chung, Jinyoung Yeo  

**Link**: [PDF](https://arxiv.org/pdf/2507.19643)  

**Abstract**: Understanding clients' thoughts and beliefs is fundamental in counseling, yet current evaluations of LLM therapists often fail to assess this ability. Existing evaluation methods rely on client simulators that clearly disclose internal states to the therapist, making it difficult to determine whether an LLM therapist can uncover unexpressed perspectives. To address this limitation, we introduce MindVoyager, a novel evaluation framework featuring a controllable and realistic client simulator which dynamically adapts itself based on the ongoing counseling session, offering a more realistic and challenging evaluation environment. We further introduce evaluation metrics that assess the exploration ability of LLM therapists by measuring their thorough understanding of client's beliefs and thoughts. 

**Abstract (ZH)**: 理解客户的思想和信念是咨询中的 fundamentals，然而当前对大语言模型治疗师的评估往往未能考察这一能力。现有的评估方法依赖于清楚披露内部状态的客户模拟器，这使得难以判断大语言模型治疗师是否能够揭示未表达的观点。为解决这一局限，我们介绍了 MindVoyager，一种新颖的评估框架，该框架具有可控且真实的客户模拟器，并根据正在进行的咨询会话动态调整自身，提供更真实且具有挑战性的评估环境。我们还介绍了评估指标，通过衡量大语言模型治疗师对客户信念和思想的深入了解程度来评估其探索能力。 

---
# Efficient and Scalable Agentic AI with Heterogeneous Systems 

**Title (ZH)**: 高效的可扩展代理人工智能系统 

**Authors**: Zain Asgar, Michelle Nguyen, Sachin Katti  

**Link**: [PDF](https://arxiv.org/pdf/2507.19635)  

**Abstract**: AI agents are emerging as a dominant workload in a wide range of applications, promising to be the vehicle that delivers the promised benefits of AI to enterprises and consumers. Unlike conventional software or static inference, agentic workloads are dynamic and structurally complex. Often these agents are directed graphs of compute and IO operations that span multi-modal data input and conversion), data processing and context gathering (e.g vector DB lookups), multiple LLM inferences, tool calls, etc. To scale AI agent usage, we need efficient and scalable deployment and agent-serving infrastructure.
To tackle this challenge, in this paper, we present a system design for dynamic orchestration of AI agent workloads on heterogeneous compute infrastructure spanning CPUs and accelerators, both from different vendors and across different performance tiers within a single vendor. The system delivers several building blocks: a framework for planning and optimizing agentic AI execution graphs using cost models that account for compute, memory, and bandwidth constraints of different HW; a MLIR based representation and compilation system that can decompose AI agent execution graphs into granular operators and generate code for different HW options; and a dynamic orchestration system that can place the granular components across a heterogeneous compute infrastructure and stitch them together while meeting an end-to-end SLA. Our design performs a systems level TCO optimization and preliminary results show that leveraging a heterogeneous infrastructure can deliver significant TCO benefits. A preliminary surprising finding is that for some workloads a heterogeneous combination of older generation GPUs with newer accelerators can deliver similar TCO as the latest generation homogenous GPU infrastructure design, potentially extending the life of deployed infrastructure. 

**Abstract (ZH)**: AI代理正在成为广泛应用场景中的主导工作负载，有望为企业和消费者带来人工智能承诺的好处。与传统的软件或静态推理不同，代理工作负载是动态且结构复杂的。这些代理通常是由计算和IO操作组成的有向图，跨多模态数据输入和转换、数据处理和上下文收集（例如向量数据库查找），以及多轮语言模型推理、工具调用等。为了扩展AI代理的使用，我们需要高效的可扩展部署和代理服务基础设施。

为应对这一挑战，本文提出了一种系统设计，用于在异构计算基础设施上动态编排跨不同供应商的CPU和加速器（包括单个供应商内的不同性能级别）的AI代理工作负载。该系统提供了一系列构建块：用于使用成本模型规划和优化代理AI执行图的框架，该模型考虑了不同硬件的计算、内存和带宽约束；基于MLIR的表示和编译系统，能够将AI代理执行图分解为粒度操作，并生成适合不同硬件选项的代码；以及一个动态编排系统，在异构计算基础设施上放置粒度组件，并在满足端到端SLA的同时将它们连接起来。我们的设计进行了一体化TCO优化，初步结果显示，采用异构基础设施可以带来显著的TCO优势。初步令人惊讶的发现是，对于某些工作负载，使用较老一代GPU与新一代加速器的混合组合可以提供与最新同质GPU基础设施设计相似的TCO，这可能延长了部署基础设施的使用寿命。 

---
# MCIF: Multimodal Crosslingual Instruction-Following Benchmark from Scientific Talks 

**Title (ZH)**: MCIF: 多模态跨语言指令跟随基准来自科学讲座 

**Authors**: Sara Papi, Maike Züfle, Marco Gaido, Beatrice Savoldi, Danni Liu, Ioannis Douros, Luisa Bentivogli, Jan Niehues  

**Link**: [PDF](https://arxiv.org/pdf/2507.19634)  

**Abstract**: Recent advances in large language models have catalyzed the development of multimodal LLMs (MLLMs) that integrate text, speech, and vision within unified frameworks. As MLLMs evolve from narrow, monolingual, task-specific systems to general-purpose instruction-following models, a key frontier lies in evaluating their multilingual and multimodal capabilities over both long and short contexts. However, existing benchmarks fall short in evaluating these dimensions jointly: they are often limited to English, mostly focus on one single modality at a time, rely on short-form contexts, or lack human annotations--hindering comprehensive assessment of model performance across languages, modalities, and task complexity. To address these gaps, we introduce MCIF (Multimodal Crosslingual Instruction Following), the first multilingual human-annotated benchmark based on scientific talks that is designed to evaluate instruction-following in crosslingual, multimodal settings over both short- and long-form inputs. MCIF spans three core modalities--speech, vision, and text--and four diverse languages (English, German, Italian, and Chinese), enabling a comprehensive evaluation of MLLMs' abilities to interpret instructions across languages and combine them with multimodal contextual information. MCIF is released under a CC-BY 4.0 license to encourage open research and progress in MLLMs development. 

**Abstract (ZH)**: 近期大规模语言模型的进展促进了多模态大型语言模型（MLLMs）的发展，这些模型在统一框架内融合了文本、语音和视觉信息。随着MLLMs从狭窄的单语言、任务特定系统发展成为通用的指令遵循模型，一个关键的前沿领域是在长短上下文中评估它们的多语言和多模态能力。然而，现有的基准在联合评估这些维度方面存在不足：它们往往仅限于英语，大多仅集中于单一模态，依赖于短形式的上下文，或者缺乏人类注释——这妨碍了对模型在不同语言、模态和任务复杂度方面的全面评估。为填补这些空白，我们推出了MCIF（跨语言多模态指令遵循），这是首个基于科学讲座的多语言人工标注基准，旨在评估跨语言和多模态设置下的指令遵循能力，包括短形式和长形式的输入。MCIF涵盖了三种核心模态——语音、视觉和文本——以及四种多样的语言（英语、德语、意大利语和中文），从而全面评估MLLMs在不同语言和结合多模态上下文信息方面的解释指令能力。MCIF以CC-BY 4.0许可证发布，以促进开放研究和MLLMs的发展。 

---
# Quantum Reinforcement Learning by Adaptive Non-local Observables 

**Title (ZH)**: 自适应非局域可观测量的量子强化学习 

**Authors**: Hsin-Yi Lin, Samuel Yen-Chi Chen, Huan-Hsin Tseng, Shinjae Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2507.19629)  

**Abstract**: Hybrid quantum-classical frameworks leverage quantum computing for machine learning; however, variational quantum circuits (VQCs) are limited by the need for local measurements. We introduce an adaptive non-local observable (ANO) paradigm within VQCs for quantum reinforcement learning (QRL), jointly optimizing circuit parameters and multi-qubit measurements. The ANO-VQC architecture serves as the function approximator in Deep Q-Network (DQN) and Asynchronous Advantage Actor-Critic (A3C) algorithms. On multiple benchmark tasks, ANO-VQC agents outperform baseline VQCs. Ablation studies reveal that adaptive measurements enhance the function space without increasing circuit depth. Our results demonstrate that adaptive multi-qubit observables can enable practical quantum advantages in reinforcement learning. 

**Abstract (ZH)**: 混合量子-经典框架利用量子计算进行机器学习；然而，变分量子电路（VQCs）受限于需要进行局部测量。我们引入了一种适应性非局部可观测量（ANO）范式到VQCs中，用于量子强化学习（QRL），同时优化电路参数和多量子比特测量。ANO-VQC架构作为深度Q网络（DQN）和异步优势行为 critic（A3C）算法中的函数逼近器。在多个基准任务上，ANO-VQC智能体优于基线VQCs。消融研究显示，自适应测量可以在不增加电路深度的情况下增强函数空间。我们的结果表明，自适应多量子比特可观测量可以实现强化学习中的实用量子优势。 

---
# MOCHA: Are Code Language Models Robust Against Multi-Turn Malicious Coding Prompts? 

**Title (ZH)**: MOCHA：代码语言模型在多轮恶意编码提示下是否 robust？ 

**Authors**: Muntasir Wahed, Xiaona Zhou, Kiet A. Nguyen, Tianjiao Yu, Nirav Diwan, Gang Wang, Dilek Hakkani-Tür, Ismini Lourentzou  

**Link**: [PDF](https://arxiv.org/pdf/2507.19598)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have significantly enhanced their code generation capabilities. However, their robustness against adversarial misuse, particularly through multi-turn malicious coding prompts, remains underexplored. In this work, we introduce code decomposition attacks, where a malicious coding task is broken down into a series of seemingly benign subtasks across multiple conversational turns to evade safety filters. To facilitate systematic evaluation, we introduce \benchmarkname{}, a large-scale benchmark designed to evaluate the robustness of code LLMs against both single-turn and multi-turn malicious prompts. Empirical results across open- and closed-source models reveal persistent vulnerabilities, especially under multi-turn scenarios. Fine-tuning on MOCHA improves rejection rates while preserving coding ability, and importantly, enhances robustness on external adversarial datasets with up to 32.4% increase in rejection rates without any additional supervision. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）显著增强了它们的代码生成能力。然而，它们在对抗恶意使用的鲁棒性，特别是在通过多轮恶意编码提示方面，仍然未得到充分探索。在本工作中，我们引入了代码分解攻击，即将恶意编码任务分解为多轮对话中的一个个看似无害的子任务，以规避安全过滤。为了促进系统的评估，我们引入了\benchmarkname{}大规模基准，用于评估代码LLMs在面对单轮和多轮恶意提示时的鲁棒性。跨开源和封闭源模型的实证结果显示出持续的漏洞，尤其是多轮场景下。MOCHA上的微调提高了拒绝率同时保留了编码能力，并且重要的是，在外部对抗数据集上增强了鲁棒性，拒绝率最多提高了32.4%，无需额外监督。 

---
# Efficient Attention Mechanisms for Large Language Models: A Survey 

**Title (ZH)**: 大型语言模型中高效注意力机制的研究综述 

**Authors**: Yutao Sun, Zhenyu Li, Yike Zhang, Tengyu Pan, Bowen Dong, Yuyi Guo, Jianyong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19595)  

**Abstract**: Transformer-based architectures have become the prevailing backbone of large language models. However, the quadratic time and memory complexity of self-attention remains a fundamental obstacle to efficient long-context modeling. To address this limitation, recent research has introduced two principal categories of efficient attention mechanisms. Linear attention methods achieve linear complexity through kernel approximations, recurrent formulations, or fastweight dynamics, thereby enabling scalable inference with reduced computational overhead. Sparse attention techniques, in contrast, limit attention computation to selected subsets of tokens based on fixed patterns, block-wise routing, or clustering strategies, enhancing efficiency while preserving contextual coverage. This survey provides a systematic and comprehensive overview of these developments, integrating both algorithmic innovations and hardware-level considerations. In addition, we analyze the incorporation of efficient attention into largescale pre-trained language models, including both architectures built entirely on efficient attention and hybrid designs that combine local and global components. By aligning theoretical foundations with practical deployment strategies, this work aims to serve as a foundational reference for advancing the design of scalable and efficient language models. 

**Abstract (ZH)**: 基于Transformer的架构已成为大型语言模型的主导骨干。然而，自注意力的二次时间和内存复杂性仍然是高效长上下文建模的基本障碍。为解决这一局限性，近期研究引入了两种主要的高效注意力机制类别。线性注意力方法通过核近似、递归表示或快速权重动力学实现线性复杂性，从而实现可扩展的推理并减少计算开销。相比之下，稀疏注意力技术限制注意力计算仅针对基于固定模式、块状路由或聚类策略选择的子集中的标记，从而提高效率同时保持上下文覆盖率。本文综述了这些发展的系统和全面概述，整合了算法创新和硬件层面的考虑。此外，我们分析了将高效注意力机制纳入大规模预训练语言模型中的应用，包括完全基于高效注意力机制的架构和结合局部和全局组件的混合设计。通过结合理论基础和实用部署策略，本文旨在成为推动可扩展和高效语言模型设计的基础参考。 

---
# Mitigating Geospatial Knowledge Hallucination in Large Language Models: Benchmarking and Dynamic Factuality Aligning 

**Title (ZH)**: 在大型语言模型中减轻地理空间知识幻觉：基准测试与动态事实对齐 

**Authors**: Shengyuan Wang, Jie Feng, Tianhui Liu, Dan Pei, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.19586)  

**Abstract**: Large language models (LLMs) possess extensive world knowledge, including geospatial knowledge, which has been successfully applied to various geospatial tasks such as mobility prediction and social indicator prediction. However, LLMs often generate inaccurate geospatial knowledge, leading to geospatial hallucinations (incorrect or inconsistent representations of geospatial information) that compromise their reliability. While the phenomenon of general knowledge hallucination in LLMs has been widely studied, the systematic evaluation and mitigation of geospatial hallucinations remain largely unexplored. To address this gap, we propose a comprehensive evaluation framework for geospatial hallucinations, leveraging structured geospatial knowledge graphs for controlled assessment. Through extensive evaluation across 20 advanced LLMs, we uncover the hallucinations in their geospatial knowledge. Building on these insights, we introduce a dynamic factuality aligning method based on Kahneman-Tversky Optimization (KTO) to mitigate geospatial hallucinations in LLMs, leading to a performance improvement of over 29.6% on the proposed benchmark. Extensive experimental results demonstrate the effectiveness of our benchmark and learning algorithm in enhancing the trustworthiness of LLMs in geospatial knowledge and reasoning tasks. 

**Abstract (ZH)**: 大规模语言模型（LLMs）具备广泛的领域知识，包括地理空间知识，这些知识已在移动性预测和社会指标预测等各类地理空间任务中得到成功应用。然而，LLMs常常生成不准确的地理空间知识，导致地理空间幻觉（地理空间信息的不正确或不一致的表示），从而影响其可靠性。尽管LLMs一般知识幻觉的现象已得到广泛研究，但地理空间幻觉的系统性评估与缓解仍较少探讨。为填补这一空白，我们提出了一种全面的地理空间幻觉评估框架，利用结构化的地理空间知识图谱进行受控评估。通过在20个高级LLM上的广泛评估，我们揭示了其地理空间知识中的幻觉。基于这些洞见，我们引入了基于Kahneman-Tversky优化的动态事实对齐方法，以缓解LLMs中的地理空间幻觉，从而在所提出的基准上实现了超过29.6%的性能改进。广泛的实验结果表明，我们的基准和学习算法在增强LLMs在地理空间知识和推理任务中的可信度方面具有有效性。 

---
# Programmable Virtual Humans Toward Human Physiologically-Based Drug Discovery 

**Title (ZH)**: 基于人体生理学的药物发现的可编程虚拟人类 

**Authors**: You Wu, Philip E. Bourne, Lei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2507.19568)  

**Abstract**: Artificial intelligence (AI) has sparked immense interest in drug discovery, but most current approaches only digitize existing high-throughput experiments. They remain constrained by conventional pipelines. As a result, they do not address the fundamental challenges of predicting drug effects in humans. Similarly, biomedical digital twins, largely grounded in real-world data and mechanistic models, are tailored for late-phase drug development and lack the resolution to model molecular interactions or their systemic consequences, limiting their impact in early-stage discovery. This disconnect between early discovery and late development is one of the main drivers of high failure rates in drug discovery. The true promise of AI lies not in augmenting current experiments but in enabling virtual experiments that are impossible in the real world: testing novel compounds directly in silico in the human body. Recent advances in AI, high-throughput perturbation assays, and single-cell and spatial omics across species now make it possible to construct programmable virtual humans: dynamic, multiscale models that simulate drug actions from molecular to phenotypic levels. By bridging the translational gap, programmable virtual humans offer a transformative path to optimize therapeutic efficacy and safety earlier than ever before. This perspective introduces the concept of programmable virtual humans, explores their roles in a new paradigm of drug discovery centered on human physiology, and outlines key opportunities, challenges, and roadmaps for their realization. 

**Abstract (ZH)**: 人工智能（AI）在药物发现领域引发了巨大兴趣，但大多数当前的方法仅对现有的高通量实验进行数字化，仍然受制于传统的管道。因此，它们无法解决预测药物对人类效果的基本挑战。同样，生物医学数字孪生主要基于现实世界的数据和机制性模型，适用于后期药物开发，但缺乏建模分子相互作用及其系统后果的分辨率，限制了它们在早期发现阶段的影响。这种早期发现与后期开发之间的脱节是药物发现高失败率的主要驱动因素之一。人工智能真正的潜力不在于增强现有的实验方法，而在于使有可能在现实世界中无法实现的虚拟实验成为可能：直接在虚拟人体内进行前所未有的新型化合物的体外测试。最近在人工智能、高通量干扰 assay、以及跨物种的单细胞和空间组学方面的进展，现在已经使得构建可编程的虚拟人类成为可能：这些模型能够从分子到表型的多层次动态模拟药物作用。通过弥合转化差距，可编程的虚拟人类提供了一条优化治疗效果和安全性前所未有的新途径。本文介绍了可编程虚拟人类的概念，探讨了其在以人体生理为中心的新药物发现范式中的作用，并概述了实现其重要机会、挑战和路线图。 

---
# Differentiating hype from practical applications of large language models in medicine - a primer for healthcare professionals 

**Title (ZH)**: 区分医疗领域大型语言模型的泡沫与实际应用——医疗专业人员入门指南 

**Authors**: Elisha D.O. Roberson  

**Link**: [PDF](https://arxiv.org/pdf/2507.19567)  

**Abstract**: The medical ecosystem consists of the training of new clinicians and researchers, the practice of clinical medicine, and areas of adjacent research. There are many aspects of these domains that could benefit from the application of task automation and programmatic assistance. Machine learning and artificial intelligence techniques, including large language models (LLMs), have been promised to deliver on healthcare innovation, improving care speed and accuracy, and reducing the burden on staff for manual interventions. However, LLMs have no understanding of objective truth that is based in reality. They also represent real risks to the disclosure of protected information when used by clinicians and researchers. The use of AI in medicine in general, and the deployment of LLMs in particular, therefore requires careful consideration and thoughtful application to reap the benefits of these technologies while avoiding the dangers in each context. 

**Abstract (ZH)**: 医疗生态系统包括新临床医生和研究人员的培训、临床医学的实践以及相关研究领域。这些领域的许多方面可以从任务自动化和程序化辅助的应用中受益。机器学习和人工智能技术，包括大型语言模型（LLMs），被期望推动医疗创新，提高护理速度和准确性，并减轻人员手动干预的负担。然而，LLMs并不理解基于现实的客观真理。当由临床医生和研究人员使用时，它们还存在泄露受保护信息的真正风险。因此，医疗领域中AI的应用，特别是LLMs的部署，需要谨慎考虑和慎重应用，以在每个情境中充分利用这些技术的同时避免潜在的风险。 

---
# PennyCoder: Efficient Domain-Specific LLMs for PennyLane-Based Quantum Code Generation 

**Title (ZH)**: PennyCoder：基于PennyLane的高效领域特定大型语言模型代码生成 

**Authors**: Abdul Basit, Minghao Shao, Muhammad Haider Asif, Nouhaila Innan, Muhammad Kashif, Alberto Marchisio, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2507.19562)  

**Abstract**: The growing demand for robust quantum programming frameworks has unveiled a critical limitation: current large language model (LLM) based quantum code assistants heavily rely on remote APIs, introducing challenges related to privacy, latency, and excessive usage costs. Addressing this gap, we propose PennyCoder, a novel lightweight framework for quantum code generation, explicitly designed for local and embedded deployment to enable on-device quantum programming assistance without external API dependence. PennyCoder leverages a fine-tuned version of the LLaMA 3.1-8B model, adapted through parameter-efficient Low-Rank Adaptation (LoRA) techniques combined with domain-specific instruction tuning optimized for the specialized syntax and computational logic of quantum programming in PennyLane, including tasks in quantum machine learning and quantum reinforcement learning. Unlike prior work focused on cloud-based quantum code generation, our approach emphasizes device-native operability while maintaining high model efficacy. We rigorously evaluated PennyCoder over a comprehensive quantum programming dataset, achieving 44.3% accuracy with our fine-tuned model (compared to 33.7% for the base LLaMA 3.1-8B and 40.1% for the RAG-augmented baseline), demonstrating a significant improvement in functional correctness. 

**Abstract (ZH)**: 逐步增长的稳健量子编程框架需求揭示了一个关键限制：当前基于大型语言模型（LLM）的量子代码助手很大程度上依赖远程API，这带来了隐私、延迟和过度使用成本方面的挑战。为此，我们提出了一种新型轻量级量子代码生成框架PennyCoder，专门设计用于本地和嵌入式部署，以实现无需外部API依赖的设备本地量子编程辅助。PennyCoder利用了通过参数高效的低秩适应（LoRA）技术微调的LLaMA 3.1-8B模型，并结合了针对脉动门控量子编程特定语法和计算逻辑的指令微调，包括量子机器学习和量子强化学习任务。与以往侧重云基量子代码生成的工作不同，我们的方法强调设备本地操作能力的同时保持高模型效能。我们在一个全面的量子编程数据集上严格评估了PennyCoder，使用微调模型的准确率为44.3%（相比之下，基本的LLaMA 3.1-8B为33.7%，而增加RAG增益的基本模型为40.1%），证明了功能正确性的显著提升。 

---
# Towards Sustainability Model Cards 

**Title (ZH)**: 面向可持续性的模型卡片 

**Authors**: Gwendal Jouneaux, Jordi Cabot  

**Link**: [PDF](https://arxiv.org/pdf/2507.19559)  

**Abstract**: The growth of machine learning (ML) models and associated datasets triggers a consequent dramatic increase in energy costs for the use and training of these models. In the current context of environmental awareness and global sustainability concerns involving ICT, Green AI is becoming an important research topic. Initiatives like the AI Energy Score Ratings are a good example. Nevertheless, these benchmarking attempts are still to be integrated with existing work on Quality Models and Service-Level Agreements common in other, more mature, ICT subfields. This limits the (automatic) analysis of this model energy descriptions and their use in (semi)automatic model comparison, selection, and certification processes. We aim to leverage the concept of quality models and merge it with existing ML model reporting initiatives and Green/Frugal AI proposals to formalize a Sustainable Quality Model for AI/ML models. As a first step, we propose a new Domain-Specific Language to precisely define the sustainability aspects of an ML model (including the energy costs for its different tasks). This information can then be exported as an extended version of the well-known Model Cards initiative while, at the same time, being formal enough to be input of any other model description automatic process. 

**Abstract (ZH)**: 机器学习模型及其相关数据集的增长引发了使用和训练这些模型的能源成本的急剧增加。在当前的环保意识和ICT领域全球可持续发展关注的背景下，绿色AI正成为一个重要的研究领域。尽管人工智能能源评分评级等倡议是好的例子，但这些基准尝试尚未与现有质量模型和服务级协议工作集成，特别是在其他更成熟的ICT子领域。这限制了对这些模型能源描述的自动分析及其在半自动模型比较、选择和认证过程中的应用。我们旨在利用质量模型的概念，并将其与现有的机器学习模型报告倡议和绿色/节约型AI提议相结合，以正式化一个可持续质量模型。作为第一步，我们提出了一种新的领域特定语言，以精确定义机器学习模型的可持续性方面（包括其不同任务的能源成本）。这些信息可以导出为广为人知的模型卡片倡议的扩展版本，同时具备足夠的形式化程度，可以作为任何其他模型描述自动过程的输入。 

---
# PEMUTA: Pedagogically-Enriched Multi-Granular Undergraduate Thesis Assessment 

**Title (ZH)**: PEMUTA：教学丰富化的多层次本科毕业论文评估 

**Authors**: Jialu Zhang, Qingyang Sun, Qianyi Wang, Weiyi Zhang, Zunjie Xiao, Xiaoqing Zhang, Jianfeng Ren, Jiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.19556)  

**Abstract**: The undergraduate thesis (UGTE) plays an indispensable role in assessing a student's cumulative academic development throughout their college years. Although large language models (LLMs) have advanced education intelligence, they typically focus on holistic assessment with only one single evaluation score, but ignore the intricate nuances across multifaceted criteria, limiting their ability to reflect structural criteria, pedagogical objectives, and diverse academic competencies. Meanwhile, pedagogical theories have long informed manual UGTE evaluation through multi-dimensional assessment of cognitive development, disciplinary thinking, and academic performance, yet remain underutilized in automated settings. Motivated by the research gap, we pioneer PEMUTA, a pedagogically-enriched framework that effectively activates domain-specific knowledge from LLMs for multi-granular UGTE assessment. Guided by Vygotsky's theory and Bloom's Taxonomy, PEMUTA incorporates a hierarchical prompting scheme that evaluates UGTEs across six fine-grained dimensions: Structure, Logic, Originality, Writing, Proficiency, and Rigor (SLOWPR), followed by holistic synthesis. Two in-context learning techniques, \ie, few-shot prompting and role-play prompting, are also incorporated to further enhance alignment with expert judgments without fine-tuning. We curate a dataset of authentic UGTEs with expert-provided SLOWPR-aligned annotations to support multi-granular UGTE assessment. Extensive experiments demonstrate that PEMUTA achieves strong alignment with expert evaluations, and exhibits strong potential for fine-grained, pedagogically-informed UGTE evaluations. 

**Abstract (ZH)**: 本科生毕业论文（UGTE）在评估学生在整个大学阶段的综合学术发展方面发挥着不可替代的作用。虽然大型语言模型（LLMs）提高了教育智能化，但它们通常仅采用整体评估方式并给出单一评分，忽视了跨多方面标准的复杂细微差异，限制了其反映结构性标准、教学目标和多样学术能力的能力。同时，教学理论长期通过多维评估认知发展、学科思维和学术表现来指导手工UGTE评估，但在自动化环境中仍未充分利用。受研究空白的驱动，我们提出了PEMUTA，这是一种富含教学理念的框架，有效激活了LLMs中的特定领域知识以进行多层次UGTE评估。PEMUTA 以维果茨基理论和布卢姆分类法为指导，结合了一种层次化的提示方案，评估UGTE在结构、逻辑、创新性、写作、熟练度和严谨性（SLOWPR）六个细粒度维度上的表现，随后进行综合评价。此外，还采用了两种情境学习技术，即少量示例提示和角色扮演提示，以进一步增强与专家判断的一致性，无需微调。我们构建了一个包含专家提供的SLOWPR对齐注释的真实UGTE数据集，以支持多层次UGTE评估。广泛的实验表明，PEMUTA 在与专家评估的一致性方面表现出色，并显示出进行细粒度、教学导向的UGTE评估的强大潜力。 

---
# Extending Group Relative Policy Optimization to Continuous Control: A Theoretical Framework for Robotic Reinforcement Learning 

**Title (ZH)**: 扩展组相对策略优化到连续控制：机器人强化学习的理论框架 

**Authors**: Rajat Khanda, Mohammad Baqar, Sambuddha Chakrabarti, Satyasaran Changdar  

**Link**: [PDF](https://arxiv.org/pdf/2507.19555)  

**Abstract**: Group Relative Policy Optimization (GRPO) has shown promise in discrete action spaces by eliminating value function dependencies through group-based advantage estimation. However, its application to continuous control remains unexplored, limiting its utility in robotics where continuous actions are essential. This paper presents a theoretical framework extending GRPO to continuous control environments, addressing challenges in high-dimensional action spaces, sparse rewards, and temporal dynamics. Our approach introduces trajectory-based policy clustering, state-aware advantage estimation, and regularized policy updates designed for robotic applications. We provide theoretical analysis of convergence properties and computational complexity, establishing a foundation for future empirical validation in robotic systems including locomotion and manipulation tasks. 

**Abstract (ZH)**: 基于群体的优势估计的分组相对策略优化（GRPO）在离散动作空间中通过消除价值函数依赖性展现出了潜力，但其在连续控制领域的应用尚待探索，限制了其在需要连续动作的机器人领域中的应用。本文提出了一种理论框架，将GRPO扩展到连续控制环境，解决了高维动作空间、稀疏奖励和时间动态性等挑战。我们的方法引入了基于轨迹的策略聚类、状态感知的优势估计以及专门针对机器人应用的设计正则化策略更新。我们提供了收敛性和计算复杂性理论分析，为未来在包括移动和操作任务的机器人系统中的实证验证奠定了基础。 

---
# Rainbow Noise: Stress-Testing Multimodal Harmful-Meme Detectors on LGBTQ Content 

**Title (ZH)**: 彩虹噪音：对 LGBTQ 内容的多模态有害 meme 检测器的压力测试 

**Authors**: Ran Tong, Songtao Wei, Jiaqi Liu, Lanruo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19551)  

**Abstract**: Hateful memes aimed at LGBTQ\,+ communities often evade detection by tweaking either the caption, the image, or both. We build the first robustness benchmark for this setting, pairing four realistic caption attacks with three canonical image corruptions and testing all combinations on the PrideMM dataset. Two state-of-the-art detectors, MemeCLIP and MemeBLIP2, serve as case studies, and we introduce a lightweight \textbf{Text Denoising Adapter (TDA)} to enhance the latter's resilience. Across the grid, MemeCLIP degrades more gently, while MemeBLIP2 is particularly sensitive to the caption edits that disrupt its language processing. However, the addition of the TDA not only remedies this weakness but makes MemeBLIP2 the most robust model overall. Ablations reveal that all systems lean heavily on text, but architectural choices and pre-training data significantly impact robustness. Our benchmark exposes where current multimodal safety models crack and demonstrates that targeted, lightweight modules like the TDA offer a powerful path towards stronger defences. 

**Abstract (ZH)**: 针对LGBTQ+社区的仇恨梗经常通过修改标题、图像或两者来规避检测。我们构建了首个针对此情境的鲁棒性基准，结合四种现实的标题攻击与三种经典的图像 Corruption，并在 PrideMM 数据集上测试所有组合。两种最先进的检测器 MemeCLIP 和 MemeBLIP2 作为案例研究，我们引入了一个轻量级的文本去噪适配器（TDA）以增强后者在鲁棒性方面的表现。在整个基准测试中，MemeCLIP 的性能下滑较为温和，而 MemeBLIP2 对扰乱其语言处理的标题编辑尤为敏感。然而，加入 TDA 不仅弥补了这一不足，还使 MemeBLIP2 成为整体上最鲁棒的模型。消融实验表明，所有系统都高度依赖文本，但架构选择和预训练数据显著影响鲁棒性。我们的基准揭示了当前多模态安全模型的脆弱性，并表明像 TDA 这样的针对性轻量级模块提供了增强防御能力的有力途径。 

---
# AccessGuru: Leveraging LLMs to Detect and Correct Web Accessibility Violations in HTML Code 

**Title (ZH)**: AccessGuru: 利用大语言模型检测和修正HTML代码中的网页Accessibility违规问题 

**Authors**: Nadeen Fathallah, Daniel Hernández, Steffen Staab  

**Link**: [PDF](https://arxiv.org/pdf/2507.19549)  

**Abstract**: The vast majority of Web pages fail to comply with established Web accessibility guidelines, excluding a range of users with diverse abilities from interacting with their content. Making Web pages accessible to all users requires dedicated expertise and additional manual efforts from Web page providers. To lower their efforts and promote inclusiveness, we aim to automatically detect and correct Web accessibility violations in HTML code. While previous work has made progress in detecting certain types of accessibility violations, the problem of automatically detecting and correcting accessibility violations remains an open challenge that we address. We introduce a novel taxonomy classifying Web accessibility violations into three key categories - Syntactic, Semantic, and Layout. This taxonomy provides a structured foundation for developing our detection and correction method and redefining evaluation metrics. We propose a novel method, AccessGuru, which combines existing accessibility testing tools and Large Language Models (LLMs) to detect violations and applies taxonomy-driven prompting strategies to correct all three categories. To evaluate these capabilities, we develop a benchmark of real-world Web accessibility violations. Our benchmark quantifies syntactic and layout compliance and judges semantic accuracy through comparative analysis with human expert corrections. Evaluation against our benchmark shows that AccessGuru achieves up to 84% average violation score decrease, significantly outperforming prior methods that achieve at most 50%. 

**Abstract (ZH)**: Web页面广泛未能遵守现有的Web无障碍指南，排斥了具有各种能力的用户与其内容互动。为了使所有用户都能访问Web页面，需要网页提供者投入专门的知识和额外的手工努力。为了减轻他们的努力并促进包容性，我们旨在自动检测和修正HTML代码中的无障碍违规行为。尽管以前的工作已经在检测某些类型的无障碍违规方面取得进展，但自动检测和纠正无障碍违规的问题仍然是一个开放的挑战，我们对此进行了研究。我们提出了一个新的分类法，将Web无障碍违规行为分为三大类——语法、语义和布局。该分类法为开发我们的检测和修正方法以及重新定义评价指标提供了有组织的基础。我们提出了一种新的方法——AccessGuru，它结合了现有的无障碍测试工具和大规模语言模型（LLMs）来检测违规行为，并采用基于分类法的提示策略来修正三大类违规行为。为了评估这些能力，我们开发了一个现实世界的Web无障碍违规基准。该基准量化了语法和布局合规性，并通过与人类专家修正进行比较分析来评判语义准确性。根据基准的评估显示，AccessGuru 的平均违规得分降低了高达84%，显著优于之前的最优方法，后者仅能降低50%左右。 

---
# Justifications for Democratizing AI Alignment and Their Prospects 

**Title (ZH)**: 民主化AI对齐的正当性及其前景 

**Authors**: André Steingrüber, Kevin Baum  

**Link**: [PDF](https://arxiv.org/pdf/2507.19548)  

**Abstract**: The AI alignment problem comprises both technical and normative dimensions. While technical solutions focus on implementing normative constraints in AI systems, the normative problem concerns determining what these constraints should be. This paper examines justifications for democratic approaches to the normative problem -- where affected stakeholders determine AI alignment -- as opposed to epistocratic approaches that defer to normative experts. We analyze both instrumental justifications (democratic approaches produce better outcomes) and non-instrumental justifications (democratic approaches prevent illegitimate authority or coercion). We argue that normative and metanormative uncertainty create a justificatory gap that democratic approaches aim to fill through political rather than theoretical justification. However, we identify significant challenges for democratic approaches, particularly regarding the prevention of illegitimate coercion through AI alignment. Our analysis suggests that neither purely epistocratic nor purely democratic approaches may be sufficient on their own, pointing toward hybrid frameworks that combine expert judgment with participatory input alongside institutional safeguards against AI monopolization. 

**Abstract (ZH)**: AI对齐问题包含技术和规范两个维度。虽然技术解决方案集中在将规范约束实施到AI系统中，规范性问题则关注这些约束应当是什么。本文探讨了以民主方式解决规范性问题的正当性——受影响的利益相关者确定AI对齐——对比于依赖规范性专家的意见的精英主义方法。我们分析了工具性的正当性（民主方法产生更好的结果）和非工具性的正当性（民主方法防止不合法的权威或胁迫）。我们认为，规范性和元规范性不确定性造成了一个论证缺口，民主方法希望通过政治性而非理论性论证来填补这一缺口。然而，我们识别出民主方法存在重大挑战，特别是在通过AI对齐防止不合法胁迫方面。我们的分析表明，纯粹精英主义或纯粹民主主义方法可能都不足以单独解决该问题，指向结合专家判断与参与性输入，并辅以机构性保障防止AI垄断的混合框架。 

---
# Swift-Sarsa: Fast and Robust Linear Control 

**Title (ZH)**: Swift-Sarsa: 快速而稳健的线性控制 

**Authors**: Khurram Javed, Richard S. Sutton  

**Link**: [PDF](https://arxiv.org/pdf/2507.19539)  

**Abstract**: Javed, Sharifnassab, and Sutton (2024) introduced a new algorithm for TD learning -- SwiftTD -- that augments True Online TD($\lambda$) with step-size optimization, a bound on the effective learning rate, and step-size decay. In their experiments SwiftTD outperformed True Online TD($\lambda$) and TD($\lambda$) on a variety of prediction tasks derived from Atari games, and its performance was robust to the choice of hyper-parameters. In this extended abstract we extend SwiftTD to work for control problems. We combine the key ideas behind SwiftTD with True Online Sarsa($\lambda$) to develop an on-policy reinforcement learning algorithm called $\textit{Swift-Sarsa}$.
We propose a simple benchmark for linear on-policy control called the $\textit{operant conditioning benchmark}$. The key challenge in the operant conditioning benchmark is that a very small subset of input signals are relevant for decision making. The majority of the signals are noise sampled from a non-stationary distribution. To learn effectively, the agent must learn to differentiate between the relevant signals and the noisy signals, and minimize prediction errors by assigning credit to the weight parameters associated with the relevant signals.
Swift-Sarsa, when applied to the operant conditioning benchmark, learned to assign credit to the relevant signals without any prior knowledge of the structure of the problem. It opens the door for solution methods that learn representations by searching over hundreds of millions of features in parallel without performance degradation due to noisy or bad features. 

**Abstract (ZH)**: Javed, Sharifnassab, and Sutton (2024)提出的SwiftTD算法结合了True Online TD($\lambda$)的在线学习特性，并加入了步长优化、有效学习率的上界和步长衰减，从而改进了TD学习。在实验中，SwiftTD在多种源自Atari游戏的预测任务中性能超越了True Online TD($\lambda$)和TD($\lambda$)，并且表现出对超参数选择的鲁棒性。在本文扩展摘要中，我们将SwiftTD扩展应用于控制问题，并将SwiftTD的关键思想与True Online Sarsa($\lambda$)结合，开发出一种随策略强化学习算法Swift-Sarsa。 

---
# Graph Learning Metallic Glass Discovery from Wikipedia 

**Title (ZH)**: 基于Wikipedia的图学习金属玻璃发现 

**Authors**: K.-C. Ouyang, S.-Y. Zhang, S.-L. Liu, J. Tian, Y.-H. Li, H. Tong, H.-Y. Bai, W.-H. Wang, Y.-C. Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.19536)  

**Abstract**: Synthesizing new materials efficiently is highly demanded in various research fields. However, this process is usually slow and expensive, especially for metallic glasses, whose formation strongly depends on the optimal combinations of multiple elements to resist crystallization. This constraint renders only several thousands of candidates explored in the vast material space since 1960. Recently, data-driven approaches armed by advanced machine learning techniques provided alternative routes for intelligent materials design. Due to data scarcity and immature material encoding, the conventional tabular data is usually mined by statistical learning algorithms, giving limited model predictability and generalizability. Here, we propose sophisticated data learning from material network representations. The node elements are encoded from the Wikipedia by a language model. Graph neural networks with versatile architectures are designed to serve as recommendation systems to explore hidden relationships among materials. By employing Wikipedia embeddings from different languages, we assess the capability of natural languages in materials design. Our study proposes a new paradigm to harvesting new amorphous materials and beyond with artificial intelligence. 

**Abstract (ZH)**: 高效合成新材料在多个研究领域备受需求。然而，这一过程通常缓慢且昂贵，尤其是对于金属玻璃，其形成强烈依赖于多种元素的最佳组合以抵抗结晶。这种约束使得自1960年以来，只有几千种候选材料在广阔的材料空间中被探索。最近，由先进机器学习技术武装的数据驱动方法为智能材料设计提供了替代路径。由于数据稀缺和材料编码不成熟，常规的表格数据通常由统计学习算法挖掘，这限制了模型的预测能力和泛化能力。在此，我们提出从材料网络表示中进行复杂的数据学习。节点元素通过语言模型从Wikipedia编码。具有多种架构的图神经网络被设计为推荐系统，以探索材料之间的隐藏关系。通过使用不同语言的Wikipedia嵌入，我们评估自然语言在材料设计中的能力。我们的研究提出了一种新的范式，借助人工智能来获取新的无定形材料以及其他新材料。 

---
# FedDPG: An Adaptive Yet Efficient Prompt-tuning Approach in Federated Learning Settings 

**Title (ZH)**: FedDPG：联邦学习环境中的一种自适应且高效的提示调优方法 

**Authors**: Ali Shakeri, Wei Emma Zhang, Amin Beheshti, Weitong Chen, Jian Yang, Lishan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19534)  

**Abstract**: Pre-trained Language Models (PLMs) have demonstrated impressive performance in various NLP tasks. However, traditional fine-tuning methods for leveraging PLMs for downstream tasks entail significant computational overhead. Prompt-tuning has emerged as an efficient alternative that involves prepending a limited number of parameters to the input sequence and only updating them while the PLM's parameters are frozen. However, this technique's prompts remain fixed for all inputs, reducing the model's flexibility. The Federated Learning (FL) technique has gained attention in recent years to address the growing concerns around data privacy. However, challenges such as communication and computation limitations of clients still need to be addressed. To mitigate these challenges, this paper introduces the Federated Dynamic Prompt Generator (FedDPG), which incorporates a dynamic prompt generator network to generate context-aware prompts based on the given input, ensuring flexibility and adaptability while prioritising data privacy in federated learning settings. Our experiments on three NLP benchmark datasets showcase that FedDPG outperforms the state-of-the-art parameter-efficient fine-tuning methods in terms of global model performance, and has significantly reduced the calculation time and the number of parameters to be sent through the FL network. 

**Abstract (ZH)**: 预训练语言模型（PLMs）在各种自然语言处理任务中展现了令人印象深刻的性能。然而，传统利用PLMs进行下游任务的微调方法伴随着显著的计算开销。提示微调作为一种高效的替代方法已经出现，它包括在输入序列前附加少量参数，并且仅更新这些参数而冻结PLM的参数。然而，这种方法中的提示在整个输入上保持固定，降低了模型的灵活性。联邦学习（FL）技术近年来引起了广泛关注，以应对日益增长的数据隐私问题。然而，客户在通信和计算限制方面仍面临着挑战。为缓解这些挑战，本文提出了联邦动态提示生成器（FedDPG），该方法结合了一个动态提示生成网络，根据给定的输入生成上下文感知的提示，确保在联邦学习环境中灵活性和适应性的同时，优先考虑数据隐私。我们在三个自然语言处理基准数据集上的实验表明，FedDPG在全局模型性能上优于最先进的参数高效微调方法，并且显著减少了通过FL网络传输的计算时间和参数数量。 

---
# Clinical-Grade Blood Pressure Prediction in ICU Settings: An Ensemble Framework with Uncertainty Quantification and Cross-Institutional Validation 

**Title (ZH)**: ICU环境中临床级血压预测：一种带有不确定性量化和跨机构验证的集成框架 

**Authors**: Md Basit Azam, Sarangthem Ibotombi Singh  

**Link**: [PDF](https://arxiv.org/pdf/2507.19530)  

**Abstract**: Blood pressure (BP) monitoring is critical in in tensive care units (ICUs) where hemodynamic instability can
rapidly progress to cardiovascular collapse. Current machine
learning (ML) approaches suffer from three limitations: lack of
external validation, absence of uncertainty quantification, and
inadequate data leakage prevention. This study presents the
first comprehensive framework with novel algorithmic leakage
prevention, uncertainty quantification, and cross-institutional
validation for electronic health records (EHRs) based BP pre dictions. Our methodology implemented systematic data leakage
prevention, uncertainty quantification through quantile regres sion, and external validation between the MIMIC-III and eICU
databases. An ensemble framework combines Gradient Boosting,
Random Forest, and XGBoost with 74 features across five
physiological domains. Internal validation achieved a clinically
acceptable performance (for SBP: R^2 = 0.86, RMSE = 6.03
mmHg; DBP: R^2 = 0.49, RMSE = 7.13 mmHg), meeting AAMI
standards. External validation showed 30% degradation with
critical limitations in patients with hypotensive. Uncertainty
quantification generated valid prediction intervals (80.3% SBP
and 79.9% DBP coverage), enabling risk-stratified protocols
with narrow intervals (< 15 mmHg) for standard monitoring
and wide intervals (> 30 mmHg) for manual verification. This
framework provides realistic deployment expectations for cross institutional AI-assisted BP monitoring in critical care settings.
The source code is publicly available at this https URL
mdbasit897/clinical-bp-prediction-ehr. 

**Abstract (ZH)**: 基于电子健康记录的血压监测的首个多机构综合框架：新颖的算法泄漏防止、不确定性量化与外部验证 

---
# Machine Learning Risk Intelligence for Green Hydrogen Investment: Insights for Duqm R3 Auction 

**Title (ZH)**: 绿色氢投资的机器学习风险智能：杜克拉姆R3拍卖的见解 

**Authors**: Obumneme Nwafor, Mohammed Abdul Majeed Al Hooti  

**Link**: [PDF](https://arxiv.org/pdf/2507.19529)  

**Abstract**: As green hydrogen emerges as a major component of global decarbonisation, Oman has positioned itself strategically through national auctions and international partnerships. Following two successful green hydrogen project rounds, the country launched its third auction (R3) in the Duqm region. While this area exhibits relative geospatial homogeneity, it is still vulnerable to environmental fluctuations that pose inherent risks to productivity. Despite growing global investment in green hydrogen, operational data remains scarce, with major projects like Saudi Arabia's NEOM facility not expected to commence production until 2026, and Oman's ACME Duqm project scheduled for 2028. This absence of historical maintenance and performance data from large-scale hydrogen facilities in desert environments creates a major knowledge gap for accurate risk assessment for infrastructure planning and auction decisions. Given this data void, environmental conditions emerge as accessible and reliable proxy for predicting infrastructure maintenance pressures, because harsh desert conditions such as dust storms, extreme temperatures, and humidity fluctuations are well-documented drivers of equipment degradation in renewable energy systems. To address this challenge, this paper proposes an Artificial Intelligence decision support system that leverages publicly available meteorological data to develop a predictive Maintenance Pressure Index (MPI), which predicts risk levels and future maintenance demands on hydrogen infrastructure. This tool strengthens regulatory foresight and operational decision-making by enabling temporal benchmarking to assess and validate performance claims over time. It can be used to incorporate temporal risk intelligence into auction evaluation criteria despite the absence of historical operational benchmarks. 

**Abstract (ZH)**: 随着绿色氢气成为全球去碳化的主要组成部分， Oman通过国家拍卖和国际合作战略性地定位自己。在成功举办了两轮绿色氢气项目后，该国在Duqm地区推出了第三轮拍卖（R3）。尽管该区域在地理空间上表现出相对的均一性，但仍易受环境波动的影响，这些波动对生产力构成了固有的风险。尽管全球在绿色氢气方面的投资不断增加，但运营数据仍相当稀缺，沙特阿拉伯NEOM设施预计要到2026年才能启动生产，Oman的ACME Duqm项目则计划于2028年启动。大型氢能源设施在沙漠环境中的历史维护和性能数据的缺失，导致在基础设施规划和拍卖决策中存在重大知识空白。鉴于这一数据缺口，环境条件成为预测基础设施维护压力的可及且可靠代理，因为诸如沙尘暴、极端温度和湿度波动等严酷的沙漠条件已被证实是可再生能源系统设备退化的良好记录驱动因素。为应对这一挑战，本文提出了一种人工智能决策支持系统，利用公开气象数据开发预测维护压力指数（MPI），以预测风险水平和未来氢能源基础设施的维护需求。该工具通过时间基准评估和验证性能声明，增强了监管预见性和运营决策，即使在缺乏历史运营基准的情况下，也可以将其用于拍卖评估标准中，纳入时间风险智能。 

---
# Quantizing Text-attributed Graphs for Semantic-Structural Integration 

**Title (ZH)**: 量化文本 Attribution 的图形以实现语义结构集成 

**Authors**: Jianyuan Bo, Hao Wu, Yuan Fang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19526)  

**Abstract**: Text-attributed graphs (TAGs) have emerged as a powerful representation for modeling complex relationships across diverse domains. With the rise of large language models (LLMs), there is growing interest in leveraging their capabilities for graph learning. However, current approaches face significant challenges in embedding structural information into LLM-compatible formats, requiring either computationally expensive alignment mechanisms or manual graph verbalization techniques that often lose critical structural details. Moreover, these methods typically require labeled data from source domains for effective transfer learning, significantly constraining their adaptability. We propose STAG, a novel self-supervised framework that directly quantizes graph structural information into discrete tokens using a frozen codebook. Unlike traditional quantization approaches, our method employs soft assignment and KL divergence guided quantization to address the unique challenges of graph data, which lacks natural tokenization structures. Our framework enables both LLM-based and traditional learning approaches, supporting true zero-shot transfer learning without requiring labeled data even in the source domain. Extensive experiments demonstrate state-of-the-art performance across multiple node classification benchmarks while maintaining compatibility with different LLM architectures, offering an elegant solution to bridging graph learning with LLMs. 

**Abstract (ZH)**: 基于文本标注的图形（Text-attributed Graphs, TAGs）已 emerges 作为描述跨多个领域复杂关系的强大表示形式。随着大规模语言模型（LLMs）的兴起，人们越来越有兴趣利用它们的能力进行图学习。然而，现有方法在将结构信息嵌入LLM兼容格式时面临着重大挑战，要么需要计算成本高昂的对齐机制，要么需要手动的图语义化技术，后者往往会丢失关键的结构细节。此外，这些方法通常需要来自源领域的标记数据以实现有效的迁移学习，这极大地限制了它们的适应性。我们提出了STAG，一种新颖的自监督框架，直接使用冻结的码本将图结构信息量化为离散的标记。不同于传统的量化方法，我们的方法采用软分配和基于KL散度的量化来解决图数据的独特挑战，而图数据缺乏自然的标记化结构。该框架支持基于LLM和传统学习方法，能够实现真正的零样本迁移学习，即使在源领域也没有需要标记数据。广泛实验表明，STAG在多个节点分类基准测试中取得了目前最佳性能，同时保持与不同LLM架构的兼容性，提供了一种优雅的解决方案，用于将图学习与LLM结合。 

---
# MMCircuitEval: A Comprehensive Multimodal Circuit-Focused Benchmark for Evaluating LLMs 

**Title (ZH)**: MMCircuitEval: 一个全面的多模态电路集中基准，用于评估LLM 

**Authors**: Chenchen Zhao, Zhengyuan Shi, Xiangyu Wen, Chengjie Liu, Yi Liu, Yunhao Zhou, Yuxiang Zhao, Hefei Feng, Yinan Zhu, Gwok-Waa Wan, Xin Cheng, Weiyu Chen, Yongqi Fu, Chujie Chen, Chenhao Xue, Guangyu Sun, Ying Wang, Yibo Lin, Jun Yang, Ning Xu, Xi Wang, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.19525)  

**Abstract**: The emergence of multimodal large language models (MLLMs) presents promising opportunities for automation and enhancement in Electronic Design Automation (EDA). However, comprehensively evaluating these models in circuit design remains challenging due to the narrow scope of existing benchmarks. To bridge this gap, we introduce MMCircuitEval, the first multimodal benchmark specifically designed to assess MLLM performance comprehensively across diverse EDA tasks. MMCircuitEval comprises 3614 meticulously curated question-answer (QA) pairs spanning digital and analog circuits across critical EDA stages - ranging from general knowledge and specifications to front-end and back-end design. Derived from textbooks, technical question banks, datasheets, and real-world documentation, each QA pair undergoes rigorous expert review for accuracy and relevance. Our benchmark uniquely categorizes questions by design stage, circuit type, tested abilities (knowledge, comprehension, reasoning, computation), and difficulty level, enabling detailed analysis of model capabilities and limitations. Extensive evaluations reveal significant performance gaps among existing LLMs, particularly in back-end design and complex computations, highlighting the critical need for targeted training datasets and modeling approaches. MMCircuitEval provides a foundational resource for advancing MLLMs in EDA, facilitating their integration into real-world circuit design workflows. Our benchmark is available at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型在电子设计自动化中的涌现为自动化和增强带来了 promising 的机会。然而，由于现有基准的范围狭窄，在电路设计中全面评估这些模型仍然具有挑战性。为解决这一问题，我们引入了 MMCircuitEval，这是首个专门设计用于全面评估多模态大型语言模型在各种电子设计自动化任务中的性能的多模态基准。MMCircuitEval 包含 3614 个经过精心策划的问答（QA）对，覆盖了从数字电路到模拟电路的关键电子设计自动化阶段，题目涉及从基础知识和规格到前端和后端设计的各种方面。这些 QA 对来源于教材、技术问题集、数据表和实际文档，并经过严格的专家审查以确保准确性和相关性。该基准独树一帜地根据设计阶段、电路类型、测试能力（包括知识、理解、推理和计算）以及难度级别对问题进行分类，从而能够详细分析模型的能力和局限性。广泛的研究揭示了现有语言模型在后端设计和复杂计算方面存在显著的性能差距，突显了构建针对特定训练数据集和建模方法的迫切需求。MMCircuitEval 为推进多模态大型语言模型在电子设计自动化中的应用提供了基础资源，有助于其实现实际电路设计工作流程的整合。该基准可通过以下链接获取：this https URL。 

---
# Language Models for Controllable DNA Sequence Design 

**Title (ZH)**: 可控DNA序列设计的语言模型 

**Authors**: Xingyu Su, Xiner Li, Yuchao Lin, Ziqian Xie, Degui Zhi, Shuiwang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2507.19523)  

**Abstract**: We consider controllable DNA sequence design, where sequences are generated by conditioning on specific biological properties. While language models (LMs) such as GPT and BERT have achieved remarkable success in natural language generation, their application to DNA sequence generation remains largely underexplored. In this work, we introduce ATGC-Gen, an Automated Transformer Generator for Controllable Generation, which leverages cross-modal encoding to integrate diverse biological signals. ATGC-Gen is instantiated with both decoder-only and encoder-only transformer architectures, allowing flexible training and generation under either autoregressive or masked recovery objectives. We evaluate ATGC-Gen on representative tasks including promoter and enhancer sequence design, and further introduce a new dataset based on ChIP-Seq experiments for modeling protein binding specificity. Our experiments demonstrate that ATGC-Gen can generate fluent, diverse, and biologically relevant sequences aligned with the desired properties. Compared to prior methods, our model achieves notable improvements in controllability and functional relevance, highlighting the potential of language models in advancing programmable genomic design. The source code is released at (this https URL). 

**Abstract (ZH)**: 可控DNA序列设计中的自动Transformer生成器ATGC-Gen及其应用 

---
# Exoplanet Detection Using Machine Learning Models Trained on Synthetic Light Curves 

**Title (ZH)**: 基于合成光曲数据训练的机器学习模型系外行星检测 

**Authors**: Ethan Lo, Dan C. Lo  

**Link**: [PDF](https://arxiv.org/pdf/2507.19520)  

**Abstract**: With manual searching processes, the rate at which scientists and astronomers discover exoplanets is slow because of inefficiencies that require an extensive time of laborious inspections. In fact, as of now there have been about only 5,000 confirmed exoplanets since the late 1900s. Recently, machine learning (ML) has proven to be extremely valuable and efficient in various fields, capable of processing massive amounts of data in addition to increasing its accuracy by learning. Though ML models for discovering exoplanets owned by large corporations (e.g. NASA) exist already, they largely depend on complex algorithms and supercomputers. In an effort to reduce such complexities, in this paper, we report the results and potential benefits of various, well-known ML models in the discovery and validation of extrasolar planets. The ML models that are examined in this study include logistic regression, k-nearest neighbors, and random forest. The dataset on which the models train and predict is acquired from NASA's Kepler space telescope. The initial results show promising scores for each model. However, potential biases and dataset imbalances necessitate the use of data augmentation techniques to further ensure fairer predictions and improved generalization. This study concludes that, in the context of searching for exoplanets, data augmentation techniques significantly improve the recall and precision, while the accuracy varies for each model. 

**Abstract (ZH)**: 基于机器学习模型在寻找和验证系外行星中的应用与改进 

---
# Physics-informed transfer learning for SHM via feature selection 

**Title (ZH)**: 基于特征选择的物理信息迁移学习在结构健康监测中的应用 

**Authors**: J. Poole, P. Gardner, A. J. Hughes, N. Dervilis, R. S. Mills, T. A. Dardeno, K. Worden  

**Link**: [PDF](https://arxiv.org/pdf/2507.19519)  

**Abstract**: Data used for training structural health monitoring (SHM) systems are expensive and often impractical to obtain, particularly labelled data. Population-based SHM presents a potential solution to this issue by considering the available data across a population of structures. However, differences between structures will mean the training and testing distributions will differ; thus, conventional machine learning methods cannot be expected to generalise between structures. To address this issue, transfer learning (TL), can be used to leverage information across related domains. An important consideration is that the lack of labels in the target domain limits data-based metrics to quantifying the discrepancy between the marginal distributions. Thus, a prerequisite for the application of typical unsupervised TL methods is to identify suitable source structures (domains), and a set of features, for which the conditional distributions are related to the target structure. Generally, the selection of domains and features is reliant on domain expertise; however, for complex mechanisms, such as the influence of damage on the dynamic response of a structure, this task is not trivial. In this paper, knowledge of physics is leveraged to select more similar features, the modal assurance criterion (MAC) is used to quantify the correspondence between the modes of healthy structures. The MAC is shown to have high correspondence with a supervised metric that measures joint-distribution similarity, which is the primary indicator of whether a classifier will generalise between domains. The MAC is proposed as a measure for selecting a set of features that behave consistently across domains when subjected to damage, i.e. features with invariance in the conditional distributions. This approach is demonstrated on numerical and experimental case studies to verify its effectiveness in various applications. 

**Abstract (ZH)**: 基于群体的结构健康监测中的迁移学习方法研究 

---
# Target Circuit Matching in Large-Scale Netlists using GNN-Based Region Prediction 

**Title (ZH)**: 基于GNN的区域预测在大规模网表中目标电路匹配 

**Authors**: Sangwoo Seo, Jimin Seo, Yoonho Lee, Donghyeon Kim, Hyejin Shin, Banghyun Sung, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2507.19518)  

**Abstract**: Subgraph matching plays an important role in electronic design automation (EDA) and circuit verification. Traditional rule-based methods have limitations in generalizing to arbitrary target circuits. Furthermore, node-to-node matching approaches tend to be computationally inefficient, particularly for large-scale circuits. Deep learning methods have emerged as a potential solution to address these challenges, but existing models fail to efficiently capture global subgraph embeddings or rely on inefficient matching matrices, which limits their effectiveness for large circuits. In this paper, we propose an efficient graph matching approach that utilizes Graph Neural Networks (GNNs) to predict regions of high probability for containing the target circuit. Specifically, we construct various negative samples to enable GNNs to accurately learn the presence of target circuits and develop an approach to directly extracting subgraph embeddings from the entire circuit, which captures global subgraph information and addresses the inefficiency of applying GNNs to all candidate subgraphs. Extensive experiments demonstrate that our approach significantly outperforms existing methods in terms of time efficiency and target region prediction, offering a scalable and effective solution for subgraph matching in large-scale circuits. 

**Abstract (ZH)**: 基于图神经网络的高效子图匹配方法 

---
# BikeVAE-GNN: A Variational Autoencoder-Augmented Hybrid Graph Neural Network for Sparse Bicycle Volume Estimation 

**Title (ZH)**: BikeVAE-GNN：一种用于稀疏自行车流量估计的变分自编码器增强混合图神经网络 

**Authors**: Mohit Gupta, Debjit Bhowmick, Ben Beck  

**Link**: [PDF](https://arxiv.org/pdf/2507.19517)  

**Abstract**: Accurate link-level bicycle volume estimation is essential for informed urban and transport planning but it is challenged by extremely sparse count data in urban bicycling networks worldwide. We propose BikeVAE-GNN, a novel dual-task framework augmenting a Hybrid Graph Neural Network (GNN) with Variational Autoencoder (VAE) to estimate Average Daily Bicycle (ADB) counts, addressing sparse bicycle networks. The Hybrid-GNN combines Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), and GraphSAGE to effectively model intricate spatial relationships in sparse networks while VAE generates synthetic nodes and edges to enrich the graph structure and enhance the estimation performance. BikeVAE-GNN simultaneously performs - regression for bicycling volume estimation and classification for bicycling traffic level categorization. We demonstrate the effectiveness of BikeVAE-GNN using OpenStreetMap data and publicly available bicycle count data within the City of Melbourne - where only 141 of 15,933 road segments have labeled counts (resulting in 99% count data sparsity). Our experiments show that BikeVAE-GNN outperforms machine learning and baseline GNN models, achieving a mean absolute error (MAE) of 30.82 bicycles per day, accuracy of 99% and F1-score of 0.99. Ablation studies further validate the effective role of Hybrid-GNN and VAE components. Our research advances bicycling volume estimation in sparse networks using novel and state-of-the-art approaches, providing insights for sustainable bicycling infrastructures. 

**Abstract (ZH)**: 准确的自行车流量链路级估计对于城市和交通规划至关重要，但全球城市自行车网络中的计数数据极为稀疏，对自行车流量估计构成了挑战。我们提出了一种名为BikeVAE-GNN的新型双重任务框架，该框架通过将混合图神经网络（GNN）与变分自编码器（VAE）结合，以估计平均每日自行车（ADB）计数，解决稀疏自行车网络问题。混合GNN结合了图卷积网络（GCN）、图注意网络（GAT）和GraphSAGE，有效建模稀疏网络中的复杂空间关系，而VAE生成合成节点和边以丰富图结构并增强估计性能。BikeVAE-GNN同时进行自行车流量估计的回归和骑行交通等级分类的分类。我们使用OpenStreetMap数据和墨尔本市公开可获取的自行车计数数据（其中只有141条道路路段有标注计数，导致99%的计数数据稀疏性）验证了BikeVAE-GNN的有效性。实验结果表明，BikeVAE-GNN优于机器学习和基础GNN模型，平均绝对误差（MAE）为每天30.82辆自行车，准确率为99%，F1分数为0.99。消融研究表明，混合GNN和VAE组件的有效性得到进一步验证。我们的研究采用新颖和最先进的方法，促进了稀疏网络中的自行车流量估计，为可持续的自行车基础设施提供了见解。 

---
# Enhancing Spatiotemporal Networks with xLSTM: A Scalar LSTM Approach for Cellular Traffic Forecasting 

**Title (ZH)**: 基于标量LSTM的时空网络增强：细胞级交通预测方法 

**Authors**: Khalid Ali, Zineddine Bettouche, Andreas Kassler, Andreas Fischer  

**Link**: [PDF](https://arxiv.org/pdf/2507.19513)  

**Abstract**: Accurate spatiotemporal traffic forecasting is vital for intelligent resource management in 5G and beyond. However, conventional AI approaches often fail to capture the intricate spatial and temporal patterns that exist, due to e.g., the mobility of users. We introduce a lightweight, dual-path Spatiotemporal Network that leverages a Scalar LSTM (sLSTM) for efficient temporal modeling and a three-layer Conv3D module for spatial feature extraction. A fusion layer integrates both streams into a cohesive representation, enabling robust forecasting. Our design improves gradient stability and convergence speed while reducing prediction error. Evaluations on real-world datasets show superior forecast performance over ConvLSTM baselines and strong generalization to unseen regions, making it well-suited for large-scale, next-generation network deployments. Experimental evaluation shows a 23% MAE reduction over ConvLSTM, with a 30% improvement in model generalization. 

**Abstract (ZH)**: 准确的空间时间交通预测对于5G及更高级别的智能资源管理至关重要。然而，传统的AI方法往往难以捕捉到由于用户移动等原因存在的复杂的空间和时间模式。我们提出了一种轻量化、双路径的空间时间网络，该网络利用标量LSTM（sLSTM）进行高效的时间建模，并使用三层Conv3D模块进行空间特征提取。融合层将这两种流集成到一个统一的表示中，从而实现稳健的预测。我们的设计提高了梯度稳定性和收敛速度，并减少了预测误差。在实际数据集上的评估显示，该方法在预测性能上优于ConvLSTM基线，并且能够很好地泛化到未见过的区域，使其适用于大规模、下一代网络的部署。实验评估表明，与ConvLSTM相比，预测误差降低了23%，模型泛化能力提高了30%。 

---
# Beyond 9-to-5: A Generative Model for Augmenting Mobility Data of Underrepresented Shift Workers 

**Title (ZH)**: 超越朝九晚五：一种增强少代表性轮班工作者出行数据的生成模型 

**Authors**: Haoxuan Ma, Xishun Liao, Yifan Liu, Chris Stanford, Jiaqi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.19510)  

**Abstract**: This paper addresses a critical gap in urban mobility modeling by focusing on shift workers, a population segment comprising 15-20% of the workforce in industrialized societies yet systematically underrepresented in traditional transportation surveys and planning. This underrepresentation is revealed in this study by a comparative analysis of GPS and survey data, highlighting stark differences between the bimodal temporal patterns of shift workers and the conventional 9-to-5 schedules recorded in surveys. To address this bias, we introduce a novel transformer-based approach that leverages fragmented GPS trajectory data to generate complete, behaviorally valid activity patterns for individuals working non-standard hours. Our method employs periodaware temporal embeddings and a transition-focused loss function specifically designed to capture the unique activity rhythms of shift workers and mitigate the inherent biases in conventional transportation datasets. Evaluation shows that the generated data achieves remarkable distributional alignment with GPS data from Los Angeles County (Average JSD < 0.02 for all evaluation metrics). By transforming incomplete GPS traces into complete, representative activity patterns, our approach provides transportation planners with a powerful data augmentation tool to fill critical gaps in understanding the 24/7 mobility needs of urban populations, enabling precise and inclusive transportation planning. 

**Abstract (ZH)**: 本文通过聚焦于夜班工作者这一在工业化社会中占15-20% workforce但在传统交通调查和规划中系统性地代表性不足的人群群体，填补了城市移动建模中的一个关键空白。通过比较分析GPS数据和问卷数据，本研究揭示了夜班工作者的双峰时间模式与传统9至5工作时间记录在问卷中的显著差异。为解决这一偏差，本文引入了一种基于变压器的新颖方法，利用碎片化的GPS轨迹数据生成完整且行为上有效的活动模式，适用于非标准工作时间的个人。该方法采用了时间段感知的时间嵌入，并设计了一种侧重于转换的损失函数，专门用于捕捉夜班工作者的独特活动节奏并减轻传统交通数据集中的固有偏差。评估结果显示，生成的数据在所有评估指标上的分布与洛杉矶县的GPS数据实现了显著对齐（平均JSD<0.02）。通过将不完整的GPS轨迹转化为完整的具有代表性的活动模式，本文的方法为交通规划者提供了一种强大的数据增强工具，以便填补对城市人口24/7移动需求理解的关键空白，有助于实现精确和平等的交通规划。 

---
# Gaze-Aware AI: Mathematical modeling of epistemic experience of the Marginalized for Human-Computer Interaction & AI Systems 

**Title (ZH)**: 面向注视的AI：边缘化群体认知体验的数学建模用于人机交互与AI系统 

**Authors**: Omkar Suresh Hatti  

**Link**: [PDF](https://arxiv.org/pdf/2507.19500)  

**Abstract**: The proliferation of artificial intelligence provides an opportunity to create psychological spaciousness in society. Spaciousness is defined as the ability to hold diverse interpersonal interactions and forms the basis for vulnerability that leads to authenticity that leads to prosocial behaviors and thus to societal harmony. This paper demonstrates an attempt to quantify, the human conditioning to subconsciously modify authentic self-expression to fit the norms of the dominant culture. Gaze is explored across various marginalized and intersectional groups, using concepts from postmodern philosophy and psychology. The effects of gaze are studied through analyzing a few redacted Reddit posts, only to be discussed in discourse and not endorsement. A mathematical formulation for the Gaze Pressure Index (GPI)-Diff Composite Metric is presented to model the analysis of two sets of conversational spaces in relation to one another. The outcome includes an equation to train Large Language Models (LLMs) - the working mechanism of AI products such as Chat-GPT; and an argument for affirming and inclusive HCI, based on the equation, is presented. The argument is supported by a few principles of Neuro-plasticity, The brain's lifelong capacity to rewire. 

**Abstract (ZH)**: 人工智能的普及为在社会中创造心理空间提供了机会。心理空间被定义为容纳多样的人际交往的能力，这是走向脆弱、真诚、亲社会行为以及社会和谐的基础。本文旨在量化人类无意识地将其真实自我表达适配于主导文化规范的条件作用。通过后现代哲学和心理学的概念，研究了凝视在不同边缘化和交叉群体中的作用，仅通过分析少量隐去内容的Reddit帖子进行探讨，而不作推荐。提出了凝视压力指数（GPI）-差异复合指标的数学公式，以模型化两个对话空间之间的分析。结果包括一个用于训练大规模语言模型（LLMs）的方程，以及基于该方程的论据支持包容性的人机交互。这一论据得到了神经可塑性原则的支持，大脑终生重塑的能力。 

---
# ChatMyopia: An AI Agent for Pre-consultation Education in Primary Eye Care Settings 

**Title (ZH)**: 聚焦聊天：初级眼科护理环境中的人工智能预咨询教育代理 

**Authors**: Yue Wu, Xiaolan Chen, Weiyi Zhang, Shunming Liu, Wing Man Rita Sum, Xinyuan Wu, Xianwen Shang, Chea-su Kee, Mingguang He, Danli Shi  

**Link**: [PDF](https://arxiv.org/pdf/2507.19498)  

**Abstract**: Large language models (LLMs) show promise for tailored healthcare communication but face challenges in interpretability and multi-task integration particularly for domain-specific needs like myopia, and their real-world effectiveness as patient education tools has yet to be demonstrated. Here, we introduce ChatMyopia, an LLM-based AI agent designed to address text and image-based inquiries related to myopia. To achieve this, ChatMyopia integrates an image classification tool and a retrieval-augmented knowledge base built from literature, expert consensus, and clinical guidelines. Myopic maculopathy grading task, single question examination and human evaluations validated its ability to deliver personalized, accurate, and safe responses to myopia-related inquiries with high scalability and interpretability. In a randomized controlled trial (n=70, NCT06607822), ChatMyopia significantly improved patient satisfaction compared to traditional leaflets, enhancing patient education in accuracy, empathy, disease awareness, and patient-eyecare practitioner communication. These findings highlight ChatMyopia's potential as a valuable supplement to enhance patient education and improve satisfaction with medical services in primary eye care settings. 

**Abstract (ZH)**: 大型语言模型（LLMs）在个性化医疗沟通方面显示出潜力，但在可解释性和多任务集成方面，尤其是在像近视这样的特定领域需求方面面临着挑战，它们作为患者教育工具的真实世界有效性尚未得到证明。为此，我们介绍了基于LLM的AI代理ChatMyopia，旨在处理与近视相关的内容和图像查询。ChatMyopia集成了图像分类工具和从文献、专家共识和临床指南构建的检索增强知识库。针对近视黄斑病变分级任务、单问题评估和人类评估，验证了ChatMyopia能够提供个性化、准确且安全的近视相关查询响应，并具备高可扩展性和可解释性。在一项随机对照试验（n=70，NCT06607822）中，与传统小册子相比，ChatMyopia显著提高了患者满意度，提升了患者教育的准确性、同理心、疾病意识以及患者与眼科保健提供者之间的沟通。这些发现突显了ChatMyopia作为提高初级眼科护理中患者教育和医疗服务满意度有价值的补充的潜力。 

---
# Unlimited Editions: Documenting Human Style in AI Art Generation 

**Title (ZH)**: 无限版次：记录AI艺术生成中的人类风格 

**Authors**: Alex Leitch, Celia Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.19497)  

**Abstract**: As AI art generation becomes increasingly sophisticated, HCI research has focused primarily on questions of detection, authenticity, and automation. This paper argues that such approaches fundamentally misunderstand how artistic value emerges from the concerns that drive human image production. Through examination of historical precedents, we demonstrate that artistic style is not only visual appearance but the resolution of creative struggle, as artists wrestle with influence and technical constraints to develop unique ways of seeing. Current AI systems flatten these human choices into reproducible patterns without preserving their provenance. We propose that HCI's role lies not only in perfecting visual output, but in developing means to document the origins and evolution of artistic style as it appears within generated visual traces. This reframing suggests new technical directions for HCI research in generative AI, focused on automatic documentation of stylistic lineage and creative choice rather than simple reproduction of aesthetic effects. 

**Abstract (ZH)**: 随着AI艺术生成技术日益 sophistication，人机交互研究主要集中在检测、 authenticity 和自动化等方面。本文认为，此类方法从根本上误解了艺术价值如何源自驱动人类图像创作的关切。通过考察历史先例，我们表明，艺术风格不仅是视觉表现，更是创造性的抗争结果，艺术家们在应对影响与技术限制的同时，发展出独特的视角。当前的AI系统将这些人类选择简化为可复制的模式，而不保留其起源。我们提议，人机交互的角色不仅在于完善视觉输出，还在于开发方法记录艺术风格在生成视觉痕迹中的起源和演变。这种重新定位为生成AI的人机交互研究指出了新的技术方向，重点在于自动记录风格谱系和创造性选择，而非仅仅是简单地复制美学效果。 

---
# Simulating Human Behavior with the Psychological-mechanism Agent: Integrating Feeling, Thought, and Action 

**Title (ZH)**: 基于心理机制代理模拟人类行为：融合感觉、思考与行动 

**Authors**: Qing Dong, Pengyuan Liu, Dong Yu, Chen Kang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19495)  

**Abstract**: Generative agents have made significant progress in simulating human behavior, but existing frameworks often simplify emotional modeling and focus primarily on specific tasks, limiting the authenticity of the simulation. Our work proposes the Psychological-mechanism Agent (PSYA) framework, based on the Cognitive Triangle (Feeling-Thought-Action), designed to more accurately simulate human behavior. The PSYA consists of three core modules: the Feeling module (using a layer model of affect to simulate changes in short-term, medium-term, and long-term emotions), the Thought module (based on the Triple Network Model to support goal-directed and spontaneous thinking), and the Action module (optimizing agent behavior through the integration of emotions, needs and plans). To evaluate the framework's effectiveness, we conducted daily life simulations and extended the evaluation metrics to self-influence, one-influence, and group-influence, selection five classic psychological experiments for simulation. The results show that the PSYA framework generates more natural, consistent, diverse, and credible behaviors, successfully replicating human experimental outcomes. Our work provides a richer and more accurate emotional and cognitive modeling approach for generative agents and offers an alternative to human participants in psychological experiments. 

**Abstract (ZH)**: 心理机制代理（PSYA）框架：基于认知三角形的生成代理情感与认知建模 

---
# ChartGen: Scaling Chart Understanding Via Code-Guided Synthetic Chart Generation 

**Title (ZH)**: ChartGen：通过代码导向的合成图表生成扩展图表理解 

**Authors**: Jovana Kondic, Pengyuan Li, Dhiraj Joshi, Zexue He, Shafiq Abedin, Jennifer Sun, Ben Wiesel, Eli Schwartz, Ahmed Nassar, Bo Wu, Assaf Arbelle, Aude Oliva, Dan Gutfreund, Leonid Karlinsky, Rogerio Feris  

**Link**: [PDF](https://arxiv.org/pdf/2507.19492)  

**Abstract**: Chart-to-code reconstruction -- the task of recovering executable plotting scripts from chart images -- provides important insights into a model's ability to ground data visualizations in precise, machine-readable form. Yet many existing multimodal benchmarks largely focus primarily on answering questions about charts or summarizing them. To bridge this gap, we present ChartGen, a fully-automated pipeline for code-guided synthetic chart generation. Starting from seed chart images, ChartGen (i) prompts a vision-language model (VLM) to reconstruct each image into a python script, and (ii) iteratively augments that script with a code-oriented large language model (LLM). Using ChartGen, we create 222.5K unique chart-image code pairs from 13K seed chart images, and present an open-source synthetic chart dataset covering 27 chart types, 11 plotting libraries, and multiple data modalities (image, code, text, CSV, DocTags). From this corpus, we curate a held-out chart-to-code evaluation subset of 4.3K chart image-code pairs, and evaluate six open-weight VLMs (3B - 26B parameters), highlighting substantial room for progress. We release the pipeline, prompts, and the dataset to help accelerate efforts towards robust chart understanding and vision-conditioned code generation: this https URL 

**Abstract (ZH)**: 图到代码重构——从图表图像恢复可执行绘图脚本的任务，为理解模型将数据可视化精确转化为机器可读形式的能力提供了重要见解。然而，现有许多多模态基准主要侧重于回答关于图表的问题或对图表进行总结。为弥合这一差距，我们提出了ChartGen，这是一种完全自动化的代码引导合成图表生成流水线。从种子图表图像开始，ChartGen (i) 启动一个视觉-语言模型（VLM）将每个图像重构为一个Python脚本，(ii) 并迭代地使用代码导向的大型语言模型（LLM）扩充该脚本。使用ChartGen，我们从13K种子图表图像中创建了222,500个唯一的图表图像-代码对，并提出了一个包含27种图表类型、11种绘图库和多种数据模态（图像、代码、文本、CSV、DocTags）的开源合成图表数据集。从该语料库中，我们精心挑选了一个保留的图表到代码评估子集，包含4,300个图表图像-代码对，并对六种开源预训练VLM（3B - 26B参数）进行了评估，突显了显著的改进空间。我们发布了该流水线、提示和数据集，以帮助加速对稳健的图表理解和视觉条件下的代码生成的研究进展：https://link.to.dataset.com 

---
# Does AI and Human Advice Mitigate Punishment for Selfish Behavior? An Experiment on AI ethics From a Psychological Perspective 

**Title (ZH)**: AI和人类建议能否减轻自私行为的惩罚？从心理学视角探究AI伦理的一项实验 

**Authors**: Margarita Leib, Nils Köbis, Ivan Soraperra  

**Link**: [PDF](https://arxiv.org/pdf/2507.19487)  

**Abstract**: People increasingly rely on AI-advice when making decisions. At times, such advice can promote selfish behavior. When individuals abide by selfishness-promoting AI advice, how are they perceived and punished? To study this question, we build on theories from social psychology and combine machine-behavior and behavioral economic approaches. In a pre-registered, financially-incentivized experiment, evaluators could punish real decision-makers who (i) received AI, human, or no advice. The advice (ii) encouraged selfish or prosocial behavior, and decision-makers (iii) behaved selfishly or, in a control condition, behaved prosocially. Evaluators further assigned responsibility to decision-makers and their advisors. Results revealed that (i) prosocial behavior was punished very little, whereas selfish behavior was punished much more. Focusing on selfish behavior, (ii) compared to receiving no advice, selfish behavior was penalized more harshly after prosocial advice and more leniently after selfish advice. Lastly, (iii) whereas selfish decision-makers were seen as more responsible when they followed AI compared to human advice, punishment between the two advice sources did not vary. Overall, behavior and advice content shape punishment, whereas the advice source does not. 

**Abstract (ZH)**: 当人们遵循促进自私行为的AI建议时，他们是如何被评价和惩罚的？：一项结合机器行为和行为经济学方法的预注册实验研究 

---
# Confirmation bias: A challenge for scalable oversight 

**Title (ZH)**: 确认偏差：大规模监督的挑战 

**Authors**: Gabriel Recchia, Chatrik Singh Mangat, Jinu Nyachhyon, Mridul Sharma, Callum Canavan, Dylan Epstein-Gross, Muhammed Abdulbari  

**Link**: [PDF](https://arxiv.org/pdf/2507.19486)  

**Abstract**: Scalable oversight protocols aim to empower evaluators to accurately verify AI models more capable than themselves. However, human evaluators are subject to biases that can lead to systematic errors. We conduct two studies examining the performance of simple oversight protocols where evaluators know that the model is "correct most of the time, but not all of the time". We find no overall advantage for the tested protocols, although in Study 1, showing arguments in favor of both answers improves accuracy in cases where the model is incorrect. In Study 2, participants in both groups become more confident in the system's answers after conducting online research, even when those answers are incorrect. We also reanalyze data from prior work that was more optimistic about simple protocols, finding that human evaluators possessing knowledge absent from models likely contributed to their positive results--an advantage that diminishes as models continue to scale in capability. These findings underscore the importance of testing the degree to which oversight protocols are robust to evaluator biases, whether they outperform simple deference to the model under evaluation, and whether their performance scales with increasing problem difficulty and model capability. 

**Abstract (ZH)**: 可扩展的监督协议旨在赋予评估者准确验证超越他们能力的人工智能模型的权力。然而，人类评估者可能会受到偏见的影响，导致系统性错误。我们进行了两项研究，探讨了评估者知道模型“大部分时间正确，但并不总是正确”的简单监督协议的表现。我们没有发现测试协议的整体优势，但在研究1中，显示支持两个答案的理由可以提高模型错误时的准确性。在研究2中，即使答案是错误的，两组参与者在进行在线研究后对系统的答案也变得更加自信。我们还重新分析了更为乐观的先前工作中关于简单协议的数据，发现拥有模型缺乏的知识的人类评估者可能是其正面结果的原因——这种优势随着模型能力的不断提升而减弱。这些发现强调了测试监督协议对评估者偏见的鲁棒性、其是否在超越评估模型的情况下表现出优势以及其性能随问题难度和模型能力增加是否可扩展的重要性。 

---
# Creativity as a Human Right: Design Considerations for Computational Creativity Systems 

**Title (ZH)**: 创造力作为一种人权：计算创造力系统的设计考量 

**Authors**: Alayt Issak  

**Link**: [PDF](https://arxiv.org/pdf/2507.19485)  

**Abstract**: We investigate creativity that is underlined in the Universal Declaration of Human Rights (UDHR) to present design considerations for Computational Creativity (CC) systems. We find this declaration to describe creativity in salient aspects and bring to light creativity as a Human Right attributed to the Fourth Generation of such rights. This generation of rights attributes CC systems and the evolving nature of interaction with entities of shared intelligence. Our methodology examines five of thirty articles from the UDHR and demonstrates each article with actualizations concluding with design considerations for each. We contribute our findings to ground the relationship between creativity and CC systems. 

**Abstract (ZH)**: 我们在《世界人权宣言》中探讨创造性的内涵，以提出计算创造力（CC）系统的设计 considerations。我们发现此宣言在显著方面描述了创造性，并将其视为第四代人权之一。这一代人权将计算创造力系统及其与共享智能实体互动性质的发展视为重要组成部分。我们的方法研究《世界人权宣言》中三十篇文章中的五篇，并通过实际应用每篇文章，最终提出每篇文章的设计 considerations。我们贡献我们的研究成果，以确立创造力与计算创造力系统之间的关系。 

---
# The Architecture of Cognitive Amplification: Enhanced Cognitive Scaffolding as a Resolution to the Comfort-Growth Paradox in Human-AI Cognitive Integration 

**Title (ZH)**: 认知放大架构：增强认知支架作为人类-人工智能认知整合中舒适与成长悖论的解决方案 

**Authors**: Giuseppe Riva  

**Link**: [PDF](https://arxiv.org/pdf/2507.19483)  

**Abstract**: AI systems now function as cognitive extensions, evolving from tools to active cognitive collaborators within human-AI integrated systems. While these systems can amplify cognition - enhancing problem-solving, learning, and creativity - they present a fundamental "comfort-growth paradox": AI's user-friendly nature may foster intellectual stagnation by minimizing cognitive friction necessary for development. As AI aligns with user preferences and provides frictionless assistance, it risks inducing cognitive complacency rather than promoting growth. We introduce Enhanced Cognitive Scaffolding to resolve this paradox - reconceptualizing AI from convenient assistant to dynamic mentor. Drawing from Vygotskian theories, educational scaffolding principles, and AI ethics, our framework integrates three dimensions: (1) Progressive Autonomy, where AI support gradually fades as user competence increases; (2) Adaptive Personalization, tailoring assistance to individual needs and learning trajectories; and (3) Cognitive Load Optimization, balancing mental effort to maximize learning while minimizing unnecessary complexity. Research across educational, workplace, creative, and healthcare domains supports this approach, demonstrating accelerated skill acquisition, improved self-regulation, and enhanced higher-order thinking. The framework includes safeguards against risks like dependency, skill atrophy, and bias amplification. By prioritizing cognitive development over convenience in human-AI interaction, Enhanced Cognitive Scaffolding offers a pathway toward genuinely amplified cognition while safeguarding autonomous thought and continuous learning. 

**Abstract (ZH)**: AI系统现在作为认知扩展发挥作用，从工具演变为人在环AI系统中的主动认知合作者。虽然这些系统能够增强认知，提升解决问题、学习和创新的能力，但它们提出了一个根本的“舒适-成长悖论”：AI友好的特性可能会通过减少必要的认知摩擦，导致认知停滞不前，从而阻碍发展。随着AI倾向于满足用户偏好并提供无摩擦帮助，它可能会引发认知上的安逸而非增长。我们引入了增强的认知支架来解决这一悖论——将AI重新概念化为动态导师，而非方便的助手。我们的框架借鉴了维果茨基理论、教育支架原则和AI伦理，整合了三个维度：（1）逐步自主，随着用户能力的提高，AI支持逐渐减少；（2）适应性个性化，根据个人需求和学习轨迹定制帮助；（3）认知负载优化，平衡心理努力以最大化学习效果并减少不必要的复杂性。在教育、工作场所、创造性活动和医疗保健等多个领域进行的研究支持这一方法，显示加速了技能获取、提高了自我调节能力以及增强了高层次思维。该框架包括了对依赖性、技能萎缩和偏见放大的防范措施。通过在人机交互中优先考虑认知发展而非便捷性，增强的认知支架提供了实现真正增强认知的途径，同时保护自主思考和持续学习。 

---
# Transfer or Self-Supervised? Bridging the Performance Gap in Medical Imaging 

**Title (ZH)**: 迁移学习或自我监督？缩小医疗成像性能差距的方法 

**Authors**: Zehui Zhao, Laith Alzubaidi, Jinglan Zhang, Ye Duan, Usman Naseem, Yuantong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2407.05592)  

**Abstract**: Recently, transfer learning and self-supervised learning have gained significant attention within the medical field due to their ability to mitigate the challenges posed by limited data availability, improve model generalisation, and reduce computational expenses. Transfer learning and self-supervised learning hold immense potential for advancing medical research. However, it is crucial to recognise that transfer learning and self-supervised learning architectures exhibit distinct advantages and limitations, manifesting variations in accuracy, training speed, and robustness. This paper compares the performance and robustness of transfer learning and self-supervised learning in the medical field. Specifically, we pre-trained two models using the same source domain datasets with different pre-training methods and evaluated them on small-sized medical datasets to identify the factors influencing their final performance. We tested data with several common issues in medical domains, such as data imbalance, data scarcity, and domain mismatch, through comparison experiments to understand their impact on specific pre-trained models. Finally, we provide recommendations to help users apply transfer learning and self-supervised learning methods in medical areas, and build more convenient and efficient deployment strategies. 

**Abstract (ZH)**: 近年来，转让学习和自监督学习在医疗领域受到了广泛关注，因其能够缓解数据稀缺带来的挑战、提升模型泛化能力并减少计算成本。转让学习和自监督学习在推动医疗研究方面具有巨大潜力。然而，重要的是要认识到，转让学习和自监督学习架构各自具有不同的优势和局限性，这些差异表现在准确性、训练速度和鲁棒性等方面。本文比较了转让学习和自监督学习在医疗领域的性能和鲁棒性。具体地，我们使用相同的源域数据集和不同的预训练方法预训练了两个模型，并在小规模的医疗数据集上评估它们，以确定影响其最终性能的因素。通过比较实验，我们测试了在医疗领域常见的数据不平衡、数据稀缺和领域不匹配等问题，以了解这些问题对特定预训练模型的影响。最后，我们提供建议以帮助用户在医疗领域应用转让学习和自监督学习方法，并建立更便捷和高效的部署策略。 

---
