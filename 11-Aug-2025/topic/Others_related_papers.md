# Situationally-aware Path Planning Exploiting 3D Scene Graphs 

**Title (ZH)**: 基于3D场景图的情境感知路径规划 

**Authors**: Saad Ejaz, Marco Giberna, Muhammad Shaheer, Jose Andres Millan-Romera, Ali Tourani, Paul Kremer, Holger Voos, Jose Luis Sanchez-Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2508.06283)  

**Abstract**: 3D Scene Graphs integrate both metric and semantic information, yet their structure remains underutilized for improving path planning efficiency and interpretability. In this work, we present S-Path, a situationally-aware path planner that leverages the metric-semantic structure of indoor 3D Scene Graphs to significantly enhance planning efficiency. S-Path follows a two-stage process: it first performs a search over a semantic graph derived from the scene graph to yield a human-understandable high-level path. This also identifies relevant regions for planning, which later allows the decomposition of the problem into smaller, independent subproblems that can be solved in parallel. We also introduce a replanning mechanism that, in the event of an infeasible path, reuses information from previously solved subproblems to update semantic heuristics and prioritize reuse to further improve the efficiency of future planning attempts. Extensive experiments on both real-world and simulated environments show that S-Path achieves average reductions of 5.7x in planning time while maintaining comparable path optimality to classical sampling-based planners and surpassing them in complex scenarios, making it an efficient and interpretable path planner for environments represented by indoor 3D Scene Graphs. 

**Abstract (ZH)**: 基于3D场景图的情况感知路径规划 

---
# Mitigating Undesired Conditions in Flexible Production with Product-Process-Resource Asset Knowledge Graphs 

**Title (ZH)**: 基于产品-工艺-资源资产知识图谱的柔性生产中不良条件缓解方法 

**Authors**: Petr Novak, Stefan Biffl, Marek Obitko, Petr Kadera  

**Link**: [PDF](https://arxiv.org/pdf/2508.06278)  

**Abstract**: Contemporary industrial cyber-physical production systems (CPPS) composed of robotic workcells face significant challenges in the analysis of undesired conditions due to the flexibility of Industry 4.0 that disrupts traditional quality assurance mechanisms. This paper presents a novel industry-oriented semantic model called Product-Process-Resource Asset Knowledge Graph (PPR-AKG), which is designed to analyze and mitigate undesired conditions in flexible CPPS. Built on top of the well-proven Product-Process-Resource (PPR) model originating from ISA-95 and VDI-3682, a comprehensive OWL ontology addresses shortcomings of conventional model-driven engineering for CPPS, particularly inadequate undesired condition and error handling representation. The integration of semantic technologies with large language models (LLMs) provides intuitive interfaces for factory operators, production planners, and engineers to interact with the entire model using natural language. Evaluation with the use case addressing electric vehicle battery remanufacturing demonstrates that the PPR-AKG approach efficiently supports resource allocation based on explicitly represented capabilities as well as identification and mitigation of undesired conditions in production. The key contributions include (1) a holistic PPR-AKG model capturing multi-dimensional production knowledge, and (2) the useful combination of the PPR-AKG with LLM-based chatbots for human interaction. 

**Abstract (ZH)**: 当代工业物理- cyber物理生产系统（CPPS）由机器人工作单元组成，面临着由于工业4.0的灵活性而对非期望条件进行分析的重大挑战，这破坏了传统的质量保证机制。本文提出了一种面向行业的新型语义模型——产品-过程-资源资产知识图谱（PPR-AKG），旨在分析和缓解灵活CPPS中的非期望条件。该模型基于可信的ISA-95和VDI-3682起源的PPR模型，在此基础上构建了一个全面的OWL本体，解决了传统模型驱动工程在CPPS中的不足，特别是在非期望条件和错误处理表示方面的不足。语义技术与大型语言模型（LLMs）的集成为工厂操作员、生产计划人员和工程师提供了直观的接口，使其能够使用自然语言与整个模型进行交互。使用电动汽车电池再制造案例研究的评估表明，PPR-AKG方法能够有效地根据显式表示的能力进行资源分配，并识别和缓解生产中的非期望条件。关键贡献包括（1）一个综合的PPR-AKG模型，捕获多维度的生产知识；（2）PPR-AKG与基于LLM的聊天机器人的有效结合，用于人类交互。 

---
# ReNiL: Relative Neural Inertial Locator with Any-Scale Bayesian Inference 

**Title (ZH)**: 相对神经惯性定位器：任意尺度贝叶斯推断 

**Authors**: Kaixuan Wu, Yuanzhuo Xu, Zejun Zhang, Weiping Zhu, Steve Drew, Xiaoguang Niu  

**Link**: [PDF](https://arxiv.org/pdf/2508.06053)  

**Abstract**: Pedestrian inertial localization is key for mobile and IoT services because it provides infrastructure-free positioning. Yet most learning-based methods depend on fixed sliding-window integration, struggle to adapt to diverse motion scales and cadences, and yield inconsistent uncertainty, limiting real-world use. We present ReNiL, a Bayesian deep-learning framework for accurate, efficient, and uncertainty-aware pedestrian localization. ReNiL introduces Inertial Positioning Demand Points (IPDPs) to estimate motion at contextually meaningful waypoints instead of dense tracking, and supports inference on IMU sequences at any scale so cadence can match application needs. It couples a motion-aware orientation filter with an Any-Scale Laplace Estimator (ASLE), a dual-task network that blends patch-based self-supervision with Bayesian regression. By modeling displacements with a Laplace distribution, ReNiL provides homogeneous Euclidean uncertainty that integrates cleanly with other sensors. A Bayesian inference chain links successive IPDPs into consistent trajectories. On RoNIN-ds and a new WUDataset covering indoor and outdoor motion from 28 participants, ReNiL achieves state-of-the-art displacement accuracy and uncertainty consistency, outperforming TLIO, CTIN, iMoT, and RoNIN variants while reducing computation. Application studies further show robustness and practicality for mobile and IoT localization, making ReNiL a scalable, uncertainty-aware foundation for next-generation positioning. 

**Abstract (ZH)**: 基于惯性的人行定位对于移动和物联网服务至关重要，因为它提供了无基础设施的位置定位。然而，大多数基于学习的方法依赖于固定的时间窗集成，难以适应多样的运动规模和步伐，并且会得出不一致的不确定性，限制了其实用性。我们提出了ReNiL，这是一种用于准确、高效且具备不确定性意识的人行定位的贝叶斯深度学习框架。ReNiL 引入了惯性定位需求点（IPDPs）来在上下文相关的重要点处估计运动，而非密集跟踪，并支持在任何规模的 IMU 序列上进行推理，以便步调能够匹配应用需求。该框架结合了运动感知的方向滤波器与任意尺度拉普拉斯估计器（ASLE），这是一种兼具基于块的自监督与贝叶斯回归的双任务网络。通过使用拉普拉斯分布建模位移，ReNiL 提供了均匀的欧几里得不确定性，可以与其他传感器良好集成。贝叶斯推理链将连续的 IPDPs 连接成一致的轨迹。在RoNIN-ds数据集和一个包含28名参与者室内和室外运动的新WUDataset上，ReNiL 达到了最先进的位移精度和不确定性一致性，表现优于TLIO、CTIN、iMoT 和 RoNIN 变体，并且减少了计算量。进一步的研究表明，ReNiL 在移动和物联网定位中具有稳健性和实用性，使其成为下一代定位的可扩展且具备不确定性意识的基础。 

---
# What Voting Rules Actually Do: A Data-Driven Analysis of Multi-Winner Voting 

**Title (ZH)**: 基于数据驱动分析的多席位选举规则的实际作用 

**Authors**: Joshua Caiata, Ben Armstrong, Kate Larson  

**Link**: [PDF](https://arxiv.org/pdf/2508.06454)  

**Abstract**: Committee-selection problems arise in many contexts and applications, and there has been increasing interest within the social choice research community on identifying which properties are satisfied by different multi-winner voting rules. In this work, we propose a data-driven framework to evaluate how frequently voting rules violate axioms across diverse preference distributions in practice, shifting away from the binary perspective of axiom satisfaction given by worst-case analysis. Using this framework, we analyze the relationship between multi-winner voting rules and their axiomatic performance under several preference distributions. We then show that neural networks, acting as voting rules, can outperform traditional rules in minimizing axiom violations. Our results suggest that data-driven approaches to social choice can inform the design of new voting systems and support the continuation of data-driven research in social choice. 

**Abstract (ZH)**: 基于数据的投票规则评估框架：多胜者投票规则与公理性能的关系及其超越传统规则的表现 

---
# The Fair Game: Auditing & Debiasing AI Algorithms Over Time 

**Title (ZH)**: 公平游戏：审计与纠正时变AI算法中的偏差 

**Authors**: Debabrota Basu, Udvas Das  

**Link**: [PDF](https://arxiv.org/pdf/2508.06443)  

**Abstract**: An emerging field of AI, namely Fair Machine Learning (ML), aims to quantify different types of bias (also known as unfairness) exhibited in the predictions of ML algorithms, and to design new algorithms to mitigate them. Often, the definitions of bias used in the literature are observational, i.e. they use the input and output of a pre-trained algorithm to quantify a bias under concern. In reality,these definitions are often conflicting in nature and can only be deployed if either the ground truth is known or only in retrospect after deploying the algorithm. Thus,there is a gap between what we want Fair ML to achieve and what it does in a dynamic social environment. Hence, we propose an alternative dynamic mechanism,"Fair Game",to assure fairness in the predictions of an ML algorithm and to adapt its predictions as the society interacts with the algorithm over time. "Fair Game" puts together an Auditor and a Debiasing algorithm in a loop around an ML algorithm. The "Fair Game" puts these two components in a loop by leveraging Reinforcement Learning (RL). RL algorithms interact with an environment to take decisions, which yields new observations (also known as data/feedback) from the environment and in turn, adapts future decisions. RL is already used in algorithms with pre-fixed long-term fairness goals. "Fair Game" provides a unique framework where the fairness goals can be adapted over time by only modifying the auditor and the different biases it quantifies. Thus,"Fair Game" aims to simulate the evolution of ethical and legal frameworks in the society by creating an auditor which sends feedback to a debiasing algorithm deployed around an ML system. This allows us to develop a flexible and adaptive-over-time framework to build Fair ML systems pre- and post-deployment. 

**Abstract (ZH)**: 新兴的人工智能领域之一公平机器学习（Fair Machine Learning, FML）旨在量化机器学习算法预测中表现出的不同类型的偏差（也称为不公平性），并设计新的算法来减轻这些偏差。文献中使用的偏差定义往往是观测性的，即它们利用预训练算法的输入和输出来量化关心的偏差。然而，这些定义在现实中往往是相互冲突的，并且只有在知道真实值或在算法部署后回顾时才能应用。因此，在动态社会环境中，我们希望FML实现的目标与实际情况之间存在差距。为此，我们提出了一种替代的动态机制“Fair Game”，以确保机器学习算法预测的公平性，并使其预测能够随着社会与算法的互动而适应变化。Fair Game将审计员和去偏算法闭环地嵌入到机器学习算法中。通过利用强化学习（RL），Fair Game将这两部分放在一个闭环中，RL算法通过与环境交互来做出决策，进而从环境中获得新的观察结果（也称为数据/反馈），并在此基础上调整未来的决策。RL已经在具有固定长期公平目标的算法中使用。Fair Game提供了一个独特的框架，其中公平目标可以通过仅修改审计员及其量化的不同偏差来随时间适应。Fair Game旨在通过创建一个审计员将反馈发送给部署在机器学习系统周围的去偏算法来模拟伦理和法律框架在社会中的演变。这使我们能够开发出一种灵活且随着时间而适应的框架，用于构建部署前和部署后的公平机器学习系统。 

---
# Automated Creation of the Legal Knowledge Graph Addressing Legislation on Violence Against Women: Resource, Methodology and Lessons Learned 

**Title (ZH)**: 面向反女性暴力立法的法律知识图谱自动化创建：资源、方法和技术经验 

**Authors**: Claudia dAmato, Giuseppe Rubini, Francesco Didio, Donato Francioso, Fatima Zahra Amara, Nicola Fanizzi  

**Link**: [PDF](https://arxiv.org/pdf/2508.06368)  

**Abstract**: Legal decision-making process requires the availability of comprehensive and detailed legislative background knowledge and up-to-date information on legal cases and related sentences/decisions. Legal Knowledge Graphs (KGs) would be a valuable tool to facilitate access to legal information, to be queried and exploited for the purpose, and to enable advanced reasoning and machine learning applications. Indeed, legal KGs may act as knowledge intensive component to be used by pre-dictive machine learning solutions supporting the decision process of the legal expert. Nevertheless, a few KGs can be found in the legal domain. To fill this gap, we developed a legal KG targeting legal cases of violence against women, along with clear adopted methodologies. Specifically, the paper introduces two complementary approaches for automated legal KG construction; a systematic bottom-up approach, customized for the legal domain, and a new solution leveraging Large Language Models. Starting from legal sentences publicly available from the European Court of Justice, the solutions integrate structured data extraction, ontology development, and semantic enrichment to produce KGs tailored for legal cases involving violence against women. After analyzing and comparing the results of the two approaches, the developed KGs are validated via suitable competency questions. The obtained KG may be impactful for multiple purposes: can improve the accessibility to legal information both to humans and machine, can enable complex queries and may constitute an important knowledge component to be possibly exploited by machine learning tools tailored for predictive justice. 

**Abstract (ZH)**: 法律决策过程需要全面和详细的立法背景知识以及最新的法律案例及相关判决信息。法律知识图谱（KGs）能够作为有价值的工具，促进法律信息的访问和查询，支持高级推理和机器学习应用。实际上，法律KG可能作为知识密集型组件，用于支持法律专家的决策过程中的预测机器学习解决方案。然而，在法律领域中可以找到的KG并不多。为填补这一空白，我们开发了一个针对针对妇女暴力案件的法律KG，并采用了明确的方法论。具体而言，本文介绍了两种互补的自动化法律KG构建方法；一种是针对法律领域的系统自底向上的方法，以及一种利用大型语言模型的新解决方案。从欧洲法院公开的法律条文中出发，这些解决方案结合了结构化数据提取、本体开发和语义增强，以生成专门针对涉及妇女暴力的法律案件的KG。通过对两种方法的结果进行分析和比较，通过合适的技能问题验证了所开发的KG。获得的KG可能在多个方面产生重大影响：可以提高法律信息的可访问性，不仅对人类而且对机器，可以实现复杂查询，并可能成为被预测司法定制的机器学习工具利用的重要知识组件。 

---
# From Explainable to Explanatory Artificial Intelligence: Toward a New Paradigm for Human-Centered Explanations through Generative AI 

**Title (ZH)**: 从可解释的人工智能到具解释性的智能：通过生成式人工智能迈向以人类为中心的新解释范式 

**Authors**: Christian Meske, Justin Brenne, Erdi Uenal, Sabahat Oelcer, Ayseguel Doganguen  

**Link**: [PDF](https://arxiv.org/pdf/2508.06352)  

**Abstract**: Current explainable AI (XAI) approaches prioritize algorithmic transparency and present explanations in abstract, non-adaptive formats that often fail to support meaningful end-user understanding. This paper introduces "Explanatory AI" as a complementary paradigm that leverages generative AI capabilities to serve as explanatory partners for human understanding rather than providers of algorithmic transparency. While XAI reveals algorithmic decision processes for model validation, Explanatory AI addresses contextual reasoning to support human decision-making in sociotechnical contexts. We develop a definition and systematic eight-dimensional conceptual model distinguishing Explanatory AI through narrative communication, adaptive personalization, and progressive disclosure principles. Empirical validation through Rapid Contextual Design methodology with healthcare professionals demonstrates that users consistently prefer context-sensitive, multimodal explanations over technical transparency. Our findings reveal the practical urgency for AI systems designed for human comprehension rather than algorithmic introspection, establishing a comprehensive research agenda for advancing user-centered AI explanation approaches across diverse domains and cultural contexts. 

**Abstract (ZH)**: 当前可解释人工智能（XAI）方法侧重于算法透明度，并以抽象且非适应性的方式呈现解释，往往无法支持用户的有意义理解。“解释型人工智能”提出作为一种补充范式，利用生成型人工智能能力，作为人类理解的解释伙伴，而非算法透明度的提供者。虽然XAI揭示算法决策过程以供模型验证，解释型人工智能则侧重于情境推理，以支持社会技术背景下的人类决策。我们通过叙述性沟通、适应性个性化和渐进式披露原则，提出了解释型人工智能的定义和系统性的八维概念模型。通过与医疗专业人员合作运用快速情境化设计方法论进行实证验证，结果表明用户一致偏好情境敏感的多模态解释而非技术透明度。我们的研究发现突显了设计面向人类理解而非算法内省的人工智能系统的紧迫性，并为跨不同领域和文化背景下的用户中心型AI解释方法的发展建立了全面的研究议程。 

---
# AntiCheatPT: A Transformer-Based Approach to Cheat Detection in Competitive Computer Games 

**Title (ZH)**: AntiCheatPT：一种基于变换器的方法，用于竞赛型计算机游戏中的作弊检测 

**Authors**: Mille Mei Zhen Loo, Gert Luzkov, Paolo Burelli  

**Link**: [PDF](https://arxiv.org/pdf/2508.06348)  

**Abstract**: Cheating in online video games compromises the integrity of gaming experiences. Anti-cheat systems, such as VAC (Valve Anti-Cheat), face significant challenges in keeping pace with evolving cheating methods without imposing invasive measures on users' systems. This paper presents AntiCheatPT\_256, a transformer-based machine learning model designed to detect cheating behaviour in Counter-Strike 2 using gameplay data. To support this, we introduce and publicly release CS2CD: A labelled dataset of 795 matches. Using this dataset, 90,707 context windows were created and subsequently augmented to address class imbalance. The transformer model, trained on these windows, achieved an accuracy of 89.17\% and an AUC of 93.36\% on an unaugmented test set. This approach emphasizes reproducibility and real-world applicability, offering a robust baseline for future research in data-driven cheat detection. 

**Abstract (ZH)**: 在线视频游戏中作弊破坏了游戏体验的完整性。反作弊系统，如VAC（Valve Anti-Cheat），在跟上不断演变的作弊方法的同时，面临着在不侵犯用户系统的情况下保持同步的显著挑战。本文提出了一种基于变换器的机器学习模型AntiCheatPT\_256，以使用竞技数据检测《反恐精英：全球攻势》中的作弊行为。为此，我们引入并公开发布了CS2CD：一个包含795场比赛的标记数据集。使用该数据集，创建了90,707个上下文窗口，并通过增强方法解决了类别不平衡问题。该变换器模型在这些窗口上训练后，在未增强的测试集上实现了89.17%的准确率和93.36%的AUC值。该方法强调了可重复性和实际应用性，为未来的数据驱动型作弊检测研究提供了稳健的基线。 

---
# Symmetry breaking for inductive logic programming 

**Title (ZH)**: 归纳逻辑编程中的对称性破缺 

**Authors**: Andrew Cropper, David M. Cerna, Matti Järvisalo  

**Link**: [PDF](https://arxiv.org/pdf/2508.06263)  

**Abstract**: The goal of inductive logic programming is to search for a hypothesis that generalises training data and background knowledge. The challenge is searching vast hypothesis spaces, which is exacerbated because many logically equivalent hypotheses exist. To address this challenge, we introduce a method to break symmetries in the hypothesis space. We implement our idea in answer set programming. Our experiments on multiple domains, including visual reasoning and game playing, show that our approach can reduce solving times from over an hour to just 17 seconds. 

**Abstract (ZH)**: 归纳逻辑程序设计的目标是寻找一个假设来泛化训练数据和背景知识。挑战在于搜索巨大的假设空间，因为存在许多逻辑等价的假设。为应对这一挑战，我们提出了一种在假设空间中打破对称性的方法。我们将该思想实现于回答集编程中。我们的实验涵盖了视觉推理和游戏玩等领域，结果显示我们的方法可以将求解时间从超过一个小时缩短至仅仅17秒。 

---
# Learning Logical Rules using Minimum Message Length 

**Title (ZH)**: 基于最小消息长度学习逻辑规则 

**Authors**: Ruben Sharma, Sebastijan Dumančić, Ross D. King, Andrew Cropper  

**Link**: [PDF](https://arxiv.org/pdf/2508.06230)  

**Abstract**: Unifying probabilistic and logical learning is a key challenge in AI. We introduce a Bayesian inductive logic programming approach that learns minimum message length programs from noisy data. Our approach balances hypothesis complexity and data fit through priors, which explicitly favour more general programs, and a likelihood that favours accurate programs. Our experiments on several domains, including game playing and drug design, show that our method significantly outperforms previous methods, notably those that learn minimum description length programs. Our results also show that our approach is data-efficient and insensitive to example balance, including the ability to learn from exclusively positive examples. 

**Abstract (ZH)**: 统一概率学习和逻辑学习是AI中的一个关键挑战。我们提出了一种贝叶斯归纳逻辑 Programming方法，该方法从嘈杂数据中学习最小消息长度程序。我们的方法通过先验平衡假设的复杂性和数据拟合，先验明确偏好更一般的程序，而似然性偏好更准确的程序。在游戏玩和药物设计等多个领域的实验中，我们的方法显著优于先前方法，尤其是那些学习最小描述长度程序的方法。我们的结果还表明，我们的方法具有数据效率高和对样例平衡不敏感的特点，包括能够仅从正例中学习。 

---
# Study of Robust Features in Formulating Guidance for Heuristic Algorithms for Solving the Vehicle Routing Problem 

**Title (ZH)**: 基于启发式算法解决车辆路线问题的稳健特征研究 

**Authors**: Bachtiar Herdianto, Romain Billot, Flavien Lucas, Marc Sevaux  

**Link**: [PDF](https://arxiv.org/pdf/2508.06129)  

**Abstract**: The Vehicle Routing Problem (VRP) is a complex optimization problem with numerous real-world applications, mostly solved using metaheuristic algorithms due to its $\mathcal{NP}$-Hard nature. Traditionally, these metaheuristics rely on human-crafted designs developed through empirical studies. However, recent research shows that machine learning methods can be used the structural characteristics of solutions in combinatorial optimization, thereby aiding in designing more efficient algorithms, particularly for solving VRP. Building on this advancement, this study extends the previous research by conducting a sensitivity analysis using multiple classifier models that are capable of predicting the quality of VRP solutions. Hence, by leveraging explainable AI, this research is able to extend the understanding of how these models make decisions. Finally, our findings indicate that while feature importance varies, certain features consistently emerge as strong predictors. Furthermore, we propose a unified framework able of ranking feature impact across different scenarios to illustrate this finding. These insights highlight the potential of feature importance analysis as a foundation for developing a guidance mechanism of metaheuristic algorithms for solving the VRP. 

**Abstract (ZH)**: 车辆路线问题（VRP）是一个复杂的优化问题，具有广泛的实际应用，通常由于其NP-hard性质，采用元启发式算法求解。传统上，这些元启发式算法依赖于通过经验研究开发的人工设计。然而，最近的研究表明，机器学习方法可以用于组合优化问题中解的结构特征，从而帮助设计更高效的算法，特别是在求解VRP方面。在此基础上，本研究通过使用多种分类器模型进行敏感性分析，以预测VRP解的质量。因此，通过利用可解释的人工智能，本研究能够扩展对这些模型决策过程的理解。最终，我们的研究发现虽然特征重要性有所差异，但某些特征始终作为强预测因子出现。此外，我们提出了一种统一框架，能够跨不同场景对特征影响进行排名，以说明这一发现。这些见解突显了特征重要性分析作为开发用于解决VRP的元启发式算法指导机制的基础潜力。 

---
# Aggregate-Combine-Readout GNNs Are More Expressive Than Logic C2 

**Title (ZH)**: 聚合-合并-读取GNNs在表达能力上优于逻辑C2 

**Authors**: Stan P Hauke, Przemysław Andrzej Wałęga  

**Link**: [PDF](https://arxiv.org/pdf/2508.06091)  

**Abstract**: In recent years, there has been growing interest in understanding the expressive power of graph neural networks (GNNs) by relating them to logical languages. This research has been been initialised by an influential result of Barceló et al. (2020), who showed that the graded modal logic (or a guarded fragment of the logic C2), characterises the logical expressiveness of aggregate-combine GNNs. As a ``challenging open problem'' they left the question whether full C2 characterises the logical expressiveness of aggregate-combine-readout GNNs. This question has remained unresolved despite several attempts. In this paper, we solve the above open problem by proving that the logical expressiveness of aggregate-combine-readout GNNs strictly exceeds that of C2. This result holds over both undirected and directed graphs. Beyond its implications for GNNs, our work also leads to purely logical insights on the expressive power of infinitary logics. 

**Abstract (ZH)**: 近年来，有关通过将图神经网络（GNNs）与逻辑语言相关联来理解其表达能力的研究越来越受到关注。这项研究起始于Barceló等人（2020）的一项有影响力的成果，他们证明了分级模态逻辑（或逻辑C2的守护片段）刻画了聚合结合GNNs的逻辑表达能力。作为一个“具有挑战性的开放问题”，他们留下的问题是全C2是否刻画了聚合结合读取GNNs的逻辑表达能力。尽管有几次尝试，这个问题仍未得到解决。在本文中，我们通过证明聚合结合读取GNNs的逻辑表达能力严格超越全C2，解决了上述开放问题。这一结果对于无向图和有向图都成立。除了对GNNs的影响，我们的工作还为无限逻辑的表达能力提供了纯粹逻辑上的见解。 

---
# A Generic Complete Anytime Beam Search for Optimal Decision Tree 

**Title (ZH)**: 通用的完备任意时间束搜索算法求最优决策树 

**Authors**: Harold Silvère Kiossou, Siegfried Nijssen, Pierre Schaus  

**Link**: [PDF](https://arxiv.org/pdf/2508.06064)  

**Abstract**: Finding an optimal decision tree that minimizes classification error is known to be NP-hard. While exact algorithms based on MILP, CP, SAT, or dynamic programming guarantee optimality, they often suffer from poor anytime behavior -- meaning they struggle to find high-quality decision trees quickly when the search is stopped before completion -- due to unbalanced search space exploration. To address this, several anytime extensions of exact methods have been proposed, such as LDS-DL8.5, Top-k-DL8.5, and Blossom, but they have not been systematically compared, making it difficult to assess their relative effectiveness. In this paper, we propose CA-DL8.5, a generic, complete, and anytime beam search algorithm that extends the DL8.5 framework and unifies some existing anytime strategies. In particular, CA-DL8.5 generalizes previous approaches LDS-DL8.5 and Top-k-DL8.5, by allowing the integration of various heuristics and relaxation mechanisms through a modular design. The algorithm reuses DL8.5's efficient branch-and-bound pruning and trie-based caching, combined with a restart-based beam search that gradually relaxes pruning criteria to improve solution quality over time. Our contributions are twofold: (1) We introduce this new generic framework for exact and anytime decision tree learning, enabling the incorporation of diverse heuristics and search strategies; (2) We conduct a rigorous empirical comparison of several instantiations of CA-DL8.5 -- based on Purity, Gain, Discrepancy, and Top-k heuristics -- using an anytime evaluation metric called the primal gap integral. Experimental results on standard classification benchmarks show that CA-DL8.5 using LDS (limited discrepancy) consistently provides the best anytime performance, outperforming both other CA-DL8.5 variants and the Blossom algorithm while maintaining completeness and optimality guarantees. 

**Abstract (ZH)**: 寻找最小化分类错误的最优决策树是已知的NP-hard问题。基于MILP、CP、SAT或动态规划的精确算法虽然能保证最优性，但由于搜索空间探索不平衡，常常表现出较差的任意时间行为。为解决这一问题，提出了几种精确方法的任意时间扩展，如LDS-DL8.5、Top-k-DL8.5和Blossom，但这些方法尚未系统比较，使其相对有效性难以评估。在本文中，我们提出了一种通用的、完整的、任意时间的束搜索算法CA-DL8.5，该算法扩展了DL8.5框架并统一了一些现有的任意时间策略。特别是在模块化设计中，CA-DL8.5通过允许不同的启发式方法和松弛机制的应用，推广了LDS-DL8.5和Top-k-DL8.5的先前方法。算法利用DL8.5高效的分支定界剪枝和基于字典树的缓存，并结合基于重启的束搜索，逐步放松剪枝标准，以提高解的质量。我们的贡献主要体现在两个方面：（1）引入了一种新的通用框架，支持精确和任意时间决策树学习，能够整合各种启发式和搜索策略；（2）通过任意时间评估指标（原始间隙积分）对几种CA-DL8.5的具体实例进行了严格的实证比较。实验结果表明，使用LDS（有限偏差）的CA-DL8.5始终提供最佳的任意时间性能，优于其他CA-DL8.5变体和Blossom算法，同时保持完整性和最优性保证。 

---
# Don't Forget Imagination! 

**Title (ZH)**: 不要忘记想象力！ 

**Authors**: Evgenii E. Vityaev, Andrei Mantsivoda  

**Link**: [PDF](https://arxiv.org/pdf/2508.06062)  

**Abstract**: Cognitive imagination is a type of imagination that plays a key role in human thinking. It is not a ``picture-in-the-head'' imagination. It is a faculty to mentally visualize coherent and holistic systems of concepts and causal links that serve as semantic contexts for reasoning, decision making and prediction. Our position is that the role of cognitive imagination is still greatly underestimated, and this creates numerous problems and diminishes the current capabilities of AI. For instance, when reasoning, humans rely on imaginary contexts to retrieve background info. They also constantly return to the context for semantic verification that their reasoning is still reasonable. Thus, reasoning without imagination is blind. This paper is a call for greater attention to cognitive imagination as the next promising breakthrough in artificial intelligence. As an instrument for simulating cognitive imagination, we propose semantic models -- a new approach to mathematical models that can learn, like neural networks, and are based on probabilistic causal relationships. Semantic models can simulate cognitive imagination because they ensure the consistency of imaginary contexts and implement a glass-box approach that allows the context to be manipulated as a holistic and coherent system of interrelated facts glued together with causal relations. 

**Abstract (ZH)**: 认知想象是一种在人类思考中发挥关键作用的想象力类型，它不是头脑中的“画面”想象。它是一种能够精神化地可视化概念及其因果联系的综合系统，这些系统和服务作为语义背景用于推理、决策和预测。我们认为认知想象的作用仍然被大大低估了，这造成了诸多问题，削弱了当前人工智能的性能。例如，在推理时，人类依赖于想象的情境来检索背景信息，并不断返回情境以语义验证其推理依然合理。因此，没有想象力的推理将是盲目的。本文呼吁对认知想象给予更多的关注，作为人工智能下一个有望取得突破的方向。作为一种模拟认知想象的工具，我们提出语义模型——一种类似于神经网络的学习方法，基于概率因果关系的新数学模型。语义模型能够模拟认知想象，因为它们确保了想象情境的一致性，并采用了透明箱方法，使得情境可以作为一个综合且连贯的事实系统被整体操控并连接在一起。 

---
# Planning Agents on an Ego-Trip: Leveraging Hybrid Ego-Graph Ensembles for Improved Tool Retrieval in Enterprise Task Planning 

**Title (ZH)**: 基于自我之旅的规划代理：利用混合自我图ensemble提高企业任务规划中的工具检索 

**Authors**: Sahil Bansal, Sai Shruthi Sistla, Aarti Arikatala, Sebastian Schreiber  

**Link**: [PDF](https://arxiv.org/pdf/2508.05888)  

**Abstract**: Effective tool retrieval is essential for AI agents to select from a vast array of tools when identifying and planning actions in the context of complex user queries. Despite its central role in planning, this aspect remains underexplored in the literature. Traditional approaches rely primarily on similarities between user queries and tool descriptions, which significantly limits retrieval accuracy, specifically when handling multi-step user requests. To address these limitations, we propose a Knowledge Graph (KG)-based tool retrieval framework that captures the semantic relationships between tools and their functional dependencies. Our retrieval algorithm leverages ensembles of 1-hop ego tool graphs to model direct and indirect connections between tools, enabling more comprehensive and contextual tool selection for multi-step tasks. We evaluate our approach on a synthetically generated internal dataset across six defined user classes, extending previous work on coherent dialogue synthesis and too retrieval benchmarks. Results demonstrate that our tool graph-based method achieves 91.85% tool coverage on the micro-average Complete Recall metric, compared to 89.26% for re-ranked semantic-lexical hybrid retrieval, the strongest non-KG baseline in our experiments. These findings support our hypothesis that the structural information in the KG provides complementary signals to pure similarity matching, particularly for queries requiring sequential tool composition. 

**Abstract (ZH)**: 基于知识图谱的工具检索框架：为复杂用户查询中识别和计划动作有效选择工具 essential for AI agents 

---
# Holistic Explainable AI (H-XAI): Extending Transparency Beyond Developers in AI-Driven Decision Making 

**Title (ZH)**: 全方位可解释人工智能（H-XAI）：超越开发者的AI驱动决策透明性扩展 

**Authors**: Kausik Lakkaraju, Siva Likitha Valluru, Biplav Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2508.05792)  

**Abstract**: Current eXplainable AI (XAI) methods largely serve developers, often focusing on justifying model outputs rather than supporting diverse stakeholder needs. A recent shift toward Evaluative AI reframes explanation as a tool for hypothesis testing, but still focuses primarily on operational organizations. We introduce Holistic-XAI (H-XAI), a unified framework that integrates causal rating methods with traditional XAI methods to support explanation as an interactive, multi-method process. H-XAI allows stakeholders to ask a series of questions, test hypotheses, and compare model behavior against automatically constructed random and biased baselines. It combines instance-level and global explanations, adapting to each stakeholder's goals, whether understanding individual decisions, assessing group-level bias, or evaluating robustness under perturbations. We demonstrate the generality of our approach through two case studies spanning six scenarios: binary credit risk classification and financial time-series forecasting. H-XAI fills critical gaps left by existing XAI methods by combining causal ratings and post-hoc explanations to answer stakeholder-specific questions at both the individual decision level and the overall model level. 

**Abstract (ZH)**: 当前可解释AI（XAI）方法主要面向开发者，往往侧重于解释模型输出而非支持多元利益相关者的需求。近期对评价性AI的转向将解释重新定义为假设检验的工具，但仍主要关注运营组织。我们提出了整体可解释AI（H-XAI），这是一种统一框架，将因果评估方法与传统XAI方法整合，以支持一种互动的、多方法的解释过程。H-XAI使利益相关者能够提出一系列问题，测试假设，并将模型行为与自动构建的随机和有偏基准进行对比。它结合了实例级和全局解释，适应每个利益相关者的特定目标，无论是理解个别决策，评估群体偏见，还是评估扰动下的鲁棒性。我们通过两个横跨六个场景的案例研究展示了方法的普适性：二元信贷风险分类和金融时间序列预测。H-XAI通过结合因果评估和事后解释填补了现有XAI方法的空白，能够在个体决策层面和整体模型层面回答特定利益相关者的问题。 

---
# Whither symbols in the era of advanced neural networks? 

**Title (ZH)**: 先进神经网络时代的符号何去何从？ 

**Authors**: Thomas L. Griffiths, Brenden M. Lake, R. Thomas McCoy, Ellie Pavlick, Taylor W. Webb  

**Link**: [PDF](https://arxiv.org/pdf/2508.05776)  

**Abstract**: Some of the strongest evidence that human minds should be thought about in terms of symbolic systems has been the way they combine ideas, produce novelty, and learn quickly. We argue that modern neural networks -- and the artificial intelligence systems built upon them -- exhibit similar abilities. This undermines the argument that the cognitive processes and representations used by human minds are symbolic, although the fact that these neural networks are typically trained on data generated by symbolic systems illustrates that such systems play an important role in characterizing the abstract problems that human minds have to solve. This argument leads us to offer a new agenda for research on the symbolic basis of human thought. 

**Abstract (ZH)**: 人类思维应以符号系统为基础的一些最有力证据来自它们组合观念、创造新颖事物和快速学习的能力。我们argue现代神经网络及其构建的智能系统展示了类似的能力。这削弱了人类思维的认知过程和表征是符号性的这一观点，尽管这些神经网络通常是通过符号系统生成的数据进行训练的事实表明，此类系统在描述人类思维必须解决的抽象问题中扮演着重要角色。这一论证促使我们提出关于人类思维的符号基础的研究新议程。 

---
# InfiGUI-G1: Advancing GUI Grounding with Adaptive Exploration Policy Optimization 

**Title (ZH)**: InfiGUI-G1：基于自适应探索策略优化的GUI定位提升 

**Authors**: Yuhang Liu, Zeyu Liu, Shuanghe Zhu, Pengxiang Li, Congkai Xie, Jiasheng Wang, Xueyu Hu, Xiaotian Han, Jianbo Yuan, Xinyao Wang, Shengyu Zhang, Hongxia Yang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05731)  

**Abstract**: The emergence of Multimodal Large Language Models (MLLMs) has propelled the development of autonomous agents that operate on Graphical User Interfaces (GUIs) using pure visual input. A fundamental challenge is robustly grounding natural language instructions. This requires a precise spatial alignment, which accurately locates the coordinates of each element, and, more critically, a correct semantic alignment, which matches the instructions to the functionally appropriate UI element. Although Reinforcement Learning with Verifiable Rewards (RLVR) has proven to be effective at improving spatial alignment for these MLLMs, we find that inefficient exploration bottlenecks semantic alignment, which prevent models from learning difficult semantic associations. To address this exploration problem, we present Adaptive Exploration Policy Optimization (AEPO), a new policy optimization framework. AEPO employs a multi-answer generation strategy to enforce broader exploration, which is then guided by a theoretically grounded Adaptive Exploration Reward (AER) function derived from first principles of efficiency eta=U/C. Our AEPO-trained models, InfiGUI-G1-3B and InfiGUI-G1-7B, establish new state-of-the-art results across multiple challenging GUI grounding benchmarks, achieving significant relative improvements of up to 9.0% against the naive RLVR baseline on benchmarks designed to test generalization and semantic understanding. Resources are available at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）的出现推动了基于图形用户界面（GUIs）的自主代理的发展，这些代理仅使用纯视觉输入操作GUIs。一个基本的挑战是可靠地将自然语言指令进行语义关联。这需要精确的空间对齐，准确地定位每个元素的坐标，并且更重要的是正确进行语义对齐，即将指令匹配到功能适当的UI元素上。尽管可验证奖励的强化学习（RLVR）已经被证明能够有效提高这些MLLMs的空间对齐能力，但我们发现，无效的探索瓶颈阻碍了语义对齐的学习，从而妨碍模型学习困难的语义关联。为了解决这一探索问题，我们提出了自适应探索策略优化（AEPO），这是一种新的策略优化框架。AEPO采用多答案生成策略以促进更广泛的探索，并由从效率第一原理η=U/C导出的理论支撑的自适应探索奖励（AER）函数指导。AEPO训练后的模型InfiGUI-G1-3B和InfiGUI-G1-7B在多个具有挑战性的GUI语义关联基准测试中建立了新的最佳结果，相对于基准设计用于测试泛化能力和语义理解的天真RLVR基线，在基准测试中实现了高达9.0%的显著相对改进。更多资源请访问：这个链接。 

---
# WGAST: Weakly-Supervised Generative Network for Daily 10 m Land Surface Temperature Estimation via Spatio-Temporal Fusion 

**Title (ZH)**: WGAST：基于时空融合的弱监督生成网络每日10米地表温度估算 

**Authors**: Sofiane Bouaziz, Adel Hafiane, Raphael Canals, Rachid Nedjai  

**Link**: [PDF](https://arxiv.org/pdf/2508.06485)  

**Abstract**: Urbanization, climate change, and agricultural stress are increasing the demand for precise and timely environmental monitoring. Land Surface Temperature (LST) is a key variable in this context and is retrieved from remote sensing satellites. However, these systems face a trade-off between spatial and temporal resolution. While spatio-temporal fusion methods offer promising solutions, few have addressed the estimation of daily LST at 10 m resolution. In this study, we present WGAST, a Weakly-Supervised Generative Network for Daily 10 m LST Estimation via Spatio-Temporal Fusion of Terra MODIS, Landsat 8, and Sentinel-2. WGAST is the first end-to-end deep learning framework designed for this task. It adopts a conditional generative adversarial architecture, with a generator composed of four stages: feature extraction, fusion, LST reconstruction, and noise suppression. The first stage employs a set of encoders to extract multi-level latent representations from the inputs, which are then fused in the second stage using cosine similarity, normalization, and temporal attention mechanisms. The third stage decodes the fused features into high-resolution LST, followed by a Gaussian filter to suppress high-frequency noise. Training follows a weakly supervised strategy based on physical averaging principles and reinforced by a PatchGAN discriminator. Experiments demonstrate that WGAST outperforms existing methods in both quantitative and qualitative evaluations. Compared to the best-performing baseline, on average, WGAST reduces RMSE by 17.18% and improves SSIM by 11.00%. Furthermore, WGAST is robust to cloud-induced LST and effectively captures fine-scale thermal patterns, as validated against 33 ground-based sensors. The code is available at this https URL. 

**Abstract (ZH)**: 城市化、气候变化和农业压力正在增加对精确及时的环境监测的需求。地表温度（LST）是这一背景下的一项关键变量，通常通过卫星遥感获取。然而，这些系统面临着空间分辨率和时间分辨率之间的权衡。虽然时空融合方法提供了有前景的解决方案，但较少有方法解决了每日10米分辨率LST的估算问题。在本研究中，我们提出了一种名为WGAST的弱监督生成网络，通过结合MODIS、Landsat 8和Sentinel-2的数据实现每日10米分辨率LST的估算。WGAST是首个针对此任务设计的端到端深度学习框架，采用条件生成对抗网络架构，包括特征提取、融合、LST重建和噪声抑制四个阶段。该网络的第一阶段使用一组编码器从输入中提取多级潜在表示，第二阶段通过余弦相似性、规范化和时间注意力机制融合这些表示，第三阶段将融合特征解码为高分辨率LST，并通过高斯滤波消除高频噪声。训练采用基于物理平均原则的弱监督策略，并通过PatchGAN判别器强化。实验表明，WGAST在定量和定性评估中均优于现有方法。与表现最佳的基线方法相比，WGAST平均降低了17.18%的RMSE，并提高了11.00%的SSIM。此外，WGAST对云导致的地表温度变化具有鲁棒性，并且能够有效地捕捉到细尺度热模式，这些结果经过33个地面传感器验证。代码可在以下链接获取。 

---
# Intuition emerges in Maximum Caliber models at criticality 

**Title (ZH)**: Intuition Emerges in Maximum Caliber Models at Criticality 

**Authors**: Lluís Arola-Fernández  

**Link**: [PDF](https://arxiv.org/pdf/2508.06477)  

**Abstract**: Whether large predictive models merely parrot their training data or produce genuine insight lacks a physical explanation. This work reports a primitive form of intuition that emerges as a metastable phase of learning that critically balances next-token prediction against future path-entropy. The intuition mechanism is discovered via mind-tuning, the minimal principle that imposes Maximum Caliber in predictive models with a control temperature-like parameter $\lambda$. Training on random walks in deterministic mazes reveals a rich phase diagram: imitation (low $\lambda$), rule-breaking hallucination (high $\lambda$), and a fragile in-between window exhibiting strong protocol-dependence (hysteresis) and multistability, where models spontaneously discover novel goal-directed strategies. These results are captured by an effective low-dimensional theory and frame intuition as an emergent property at the critical balance between memorizing what is and wondering what could be. 

**Abstract (ZH)**: 大型预测模型是简单重复训练数据还是产生真实见解缺乏物理解释。本工作报告了一种学习过程中出现的原始直觉形式，它作为下一标记预测与未来路径熵之间的临界平衡的亚稳态相出现。通过心调适机制发现了这种直觉机理，这是一种将最大 caliber 原理应用于具有控制温度似参数 $\lambda$ 的预测模型的最小原则。在确定性迷宫中的随机行走训练揭示了丰富的相图：模仿（低 $\lambda$）、规则破坏性的幻觉（高 $\lambda$），以及一个脆弱的中间窗口，表现出强烈的操作规程依赖性（滞回）和多稳态，在此窗口中模型自发地发现新的目标导向策略。这些结果可以用有效的低维理论捕获，并将直觉框架视为在记忆“是什么”与探索“可能是什么”之间临界平衡的涌现性质。 

---
# SPARSE Data, Rich Results: Few-Shot Semi-Supervised Learning via Class-Conditioned Image Translation 

**Title (ZH)**: 稀疏数据，丰富成果：基于类条件图像翻译的少样本半监督学习 

**Authors**: Guido Manni, Clemente Lauretti, Loredana Zollo, Paolo Soda  

**Link**: [PDF](https://arxiv.org/pdf/2508.06429)  

**Abstract**: Deep learning has revolutionized medical imaging, but its effectiveness is severely limited by insufficient labeled training data. This paper introduces a novel GAN-based semi-supervised learning framework specifically designed for low labeled-data regimes, evaluated across settings with 5 to 50 labeled samples per class. Our approach integrates three specialized neural networks -- a generator for class-conditioned image translation, a discriminator for authenticity assessment and classification, and a dedicated classifier -- within a three-phase training framework. The method alternates between supervised training on limited labeled data and unsupervised learning that leverages abundant unlabeled images through image-to-image translation rather than generation from noise. We employ ensemble-based pseudo-labeling that combines confidence-weighted predictions from the discriminator and classifier with temporal consistency through exponential moving averaging, enabling reliable label estimation for unlabeled data. Comprehensive evaluation across eleven MedMNIST datasets demonstrates that our approach achieves statistically significant improvements over six state-of-the-art GAN-based semi-supervised methods, with particularly strong performance in the extreme 5-shot setting where the scarcity of labeled data is most challenging. The framework maintains its superiority across all evaluated settings (5, 10, 20, and 50 shots per class). Our approach offers a practical solution for medical imaging applications where annotation costs are prohibitive, enabling robust classification performance even with minimal labeled data. Code is available at this https URL. 

**Abstract (ZH)**: 基于GAN的半监督学习框架在有限标注数据条件下的应用：医学影像中的革新 

---
# Dimensional Characterization and Pathway Modeling for Catastrophic AI Risks 

**Title (ZH)**: 灾难性AI风险的维度表征与路径建模 

**Authors**: Ze Shen Chin  

**Link**: [PDF](https://arxiv.org/pdf/2508.06411)  

**Abstract**: Although discourse around the risks of Artificial Intelligence (AI) has grown, it often lacks a comprehensive, multidimensional framework, and concrete causal pathways mapping hazard to harm. This paper aims to bridge this gap by examining six commonly discussed AI catastrophic risks: CBRN, cyber offense, sudden loss of control, gradual loss of control, environmental risk, and geopolitical risk. First, we characterize these risks across seven key dimensions, namely intent, competency, entity, polarity, linearity, reach, and order. Next, we conduct risk pathway modeling by mapping step-by-step progressions from the initial hazard to the resulting harms. The dimensional approach supports systematic risk identification and generalizable mitigation strategies, while risk pathway models help identify scenario-specific interventions. Together, these methods offer a more structured and actionable foundation for managing catastrophic AI risks across the value chain. 

**Abstract (ZH)**: 尽管关于人工智能（AI）风险的讨论日益增多，但往往缺乏一个全面且多维度的框架，以及将风险转化为具体伤害的明确因果路径。本文旨在通过探讨六种常见讨论的AI灾难性风险——化学、生物、放射性、核（CBRN）、网络攻击、突然失去控制、渐进失去控制、环境风险和地缘政治风险来填补这一空白。首先，我们从意图、能力、实体、极性、线性、范围和顺序七个关键维度对这些风险进行描述。随后，我们进行风险路径建模，逐步映射从初始危险到最终伤害的过程。维度方法有助于系统地识别风险并提出普遍适用的缓解策略，而风险路径模型有助于识别特定场景下的干预措施。这些方法共同为跨价值链管理灾难性AI风险提供了更加结构化和可操作的基础。 

---
# A Systematic Literature Review of Retrieval-Augmented Generation: Techniques, Metrics, and Challenges 

**Title (ZH)**: 基于检索增强生成的技术、指标与挑战的系统文献综述 

**Authors**: Andrew Brown, Muhammad Roman, Barry Devereux  

**Link**: [PDF](https://arxiv.org/pdf/2508.06401)  

**Abstract**: This systematic review of the research literature on retrieval-augmented generation (RAG) provides a focused analysis of the most highly cited studies published between 2020 and May 2025. A total of 128 articles met our inclusion criteria. The records were retrieved from ACM Digital Library, IEEE Xplore, Scopus, ScienceDirect, and the Digital Bibliography and Library Project (DBLP). RAG couples a neural retriever with a generative language model, grounding output in up-to-date, non-parametric memory while retaining the semantic generalisation stored in model weights. Guided by the PRISMA 2020 framework, we (i) specify explicit inclusion and exclusion criteria based on citation count and research questions, (ii) catalogue datasets, architectures, and evaluation practices, and (iii) synthesise empirical evidence on the effectiveness and limitations of RAG. To mitigate citation-lag bias, we applied a lower citation-count threshold to papers published in 2025 so that emerging breakthroughs with naturally fewer citations were still captured. This review clarifies the current research landscape, highlights methodological gaps, and charts priority directions for future research. 

**Abstract (ZH)**: This systematic review of retrieval-augmented generation (RAG)研究文献的系统回顾：2020年至2025年最具引用价值的研究综述 

---
# Robust Target Speaker Diarization and Separation via Augmented Speaker Embedding Sampling 

**Title (ZH)**: 基于增强speaker嵌入采样的鲁棒目标说话人分离与识别 

**Authors**: Md Asif Jalal, Luca Remaggi, Vasileios Moschopoulos, Thanasis Kotsiopoulos, Vandana Rajan, Karthikeyan Saravanan, Anastasis Drosou, Junho Heo, Hyuk Oh, Seokyeong Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2508.06393)  

**Abstract**: Traditional speech separation and speaker diarization approaches rely on prior knowledge of target speakers or a predetermined number of participants in audio signals. To address these limitations, recent advances focus on developing enrollment-free methods capable of identifying targets without explicit speaker labeling. This work introduces a new approach to train simultaneous speech separation and diarization using automatic identification of target speaker embeddings, within mixtures. Our proposed model employs a dual-stage training pipeline designed to learn robust speaker representation features that are resilient to background noise interference. Furthermore, we present an overlapping spectral loss function specifically tailored for enhancing diarization accuracy during overlapped speech frames. Experimental results show significant performance gains compared to the current SOTA baseline, achieving 71% relative improvement in DER and 69% in cpWER. 

**Abstract (ZH)**: 传统的语音分离和说话人分场合方法依赖于目标说话人的先验知识或音频信号中参与者数量的先设。为了解决这些限制，近期的进展专注于开发无需 Enrollment 的方法，能够无需显式的讲话人标签来识别目标。本文提出了一种新的方法，用于同时训练语音分离和分场合，通过混合中的自动目标说话人嵌入识别。我们提出的模型采用双阶段训练管道，旨在学习对背景噪声干扰具有抗干扰性的稳健说话人表示特征。此外，我们呈现了一种特定于重叠谱的损失函数，以提高在重叠语音帧期间的分场合准确性。实验结果表明，与当前的 SOTA 基线相比，显著提高了性能，特别是在 DER 上实现了 71% 的相对改进，在 cpWER 上实现了 69% 的改进。 

---
# Identity Increases Stability in Neural Cellular Automata 

**Title (ZH)**: 身份增加了神经细胞自动机的稳定性 

**Authors**: James Stovold  

**Link**: [PDF](https://arxiv.org/pdf/2508.06389)  

**Abstract**: Neural Cellular Automata (NCAs) offer a way to study the growth of two-dimensional artificial organisms from a single seed cell. From the outset, NCA-grown organisms have had issues with stability, their natural boundary often breaking down and exhibiting tumour-like growth or failing to maintain the expected shape. In this paper, we present a method for improving the stability of NCA-grown organisms by introducing an 'identity' layer with simple constraints during training.
Results show that NCAs grown in close proximity are more stable compared with the original NCA model. Moreover, only a single identity value is required to achieve this increase in stability. We observe emergent movement from the stable organisms, with increasing prevalence for models with multiple identity values.
This work lays the foundation for further study of the interaction between NCA-grown organisms, paving the way for studying social interaction at a cellular level in artificial organisms. 

**Abstract (ZH)**: Neural Cellular Automata (NCAs) 提供了一种从单个种子细胞研究二维人工有机物生长的方式。从一开始，CA-g生长方式就遇到了稳定性问题，表现为非自然自然的行为和类似肿瘤的生长方式，难以保持稳定形态。本文提出了一种方法通过引入一个具有简单约束的‘身份’状态方式来提高NCAs-g生长方式的稳定性。结果表明，引入‘身份’状态后使得系统更稳定，与原生CA相比。而且只需要一个单态方式使就能达到到这一稳定性的提高。随著复杂稳定态的增多方式，涌现的行为方式也更倾向于多种稳定态的的组合。此研究为进一深入研究NCAs-g生长方式之间的相互相互互动打下基础。并并为在人工生命体中从细胞水平研究社会互动提供可能。 

---
# ActivityDiff: A diffusion model with Positive and Negative Activity Guidance for De Novo Drug Design 

**Title (ZH)**: ActivityDiff：一种具有正负活动引导的扩散模型用于从头药物设计 

**Authors**: Renyi Zhou, Huimin Zhu, Jing Tang, Min Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.06364)  

**Abstract**: Achieving precise control over a molecule's biological activity-encompassing targeted activation/inhibition, cooperative multi-target modulation, and off-target toxicity mitigation-remains a critical challenge in de novo drug design. However, existing generative methods primarily focus on producing molecules with a single desired activity, lacking integrated mechanisms for the simultaneous management of multiple intended and unintended molecular interactions. Here, we propose ActivityDiff, a generative approach based on the classifier-guidance technique of diffusion models. It leverages separately trained drug-target classifiers for both positive and negative guidance, enabling the model to enhance desired activities while minimizing harmful off-target effects. Experimental results show that ActivityDiff effectively handles essential drug design tasks, including single-/dual-target generation, fragment-constrained dual-target design, selective generation to enhance target specificity, and reduction of off-target effects. These results demonstrate the effectiveness of classifier-guided diffusion in balancing efficacy and safety in molecular design. Overall, our work introduces a novel paradigm for achieving integrated control over molecular activity, and provides ActivityDiff as a versatile and extensible framework. 

**Abstract (ZH)**: 实现对分子生物活性的精确控制，包括靶向激活/抑制、协同多靶点调节以及减轻非靶向毒性，仍然是从头药物设计中的关键挑战。现有的生成方法主要侧重于生成具有单一期望活性的分子，缺乏同时管理多个期望和非期望分子相互作用的集成机制。在此，我们提出了一种基于分类器引导技术的生成方法——ActivityDiff。该方法利用分别训练的正向和负向药物-靶点分类器进行双重指导，使得模型能够在增强期望活性的同时最小化有害的非靶向效应。实验结果表明，ActivityDiff 能够有效地处理包括单靶点/双靶点生成、片段限制的双靶点设计、选择性生成以增强靶点特异性以及减少非靶向效应在内的关键药物设计任务。这些结果证明了分类器引导的扩散方法在分子设计中平衡效能和安全性方面的有效性。总体而言，我们的工作介绍了一种新的集成控制分子活性的范式，并提供了一个灵活且可扩展的框架——ActivityDiff。 

---
# Are you In or Out (of gallery)? Wisdom from the Same-Identity Crowd 

**Title (ZH)**: 你是被接纳还是被排斥（（还是画廊？）来自相同身份群体的智慧 

**Authors**: Aman Bhatta, Maria Dhakal, Michael C. King, Kevin W. Bowyer  

**Link**: [PDF](https://arxiv.org/pdf/2508.06357)  

**Abstract**: A central problem in one-to-many facial identification is that the person in the probe image may or may not have enrolled image(s) in the gallery; that is, may be In-gallery or Out-of-gallery. Past approaches to detect when a rank-one result is Out-of-gallery have mostly focused on finding a suitable threshold on the similarity score. We take a new approach, using the additional enrolled images of the identity with the rank-one result to predict if the rank-one result is In-gallery / Out-of-gallery. Given a gallery of identities and images, we generate In-gallery and Out-of-gallery training data by extracting the ranks of additional enrolled images corresponding to the rank-one identity. We then train a classifier to utilize this feature vector to predict whether a rank-one result is In-gallery or Out-of-gallery. Using two different datasets and four different matchers, we present experimental results showing that our approach is viable for mugshot quality probe images, and also, importantly, for probes degraded by blur, reduced resolution, atmospheric turbulence and sunglasses. We also analyze results across demographic groups, and show that In-gallery / Out-of-gallery classification accuracy is similar across demographics. Our approach has the potential to provide an objective estimate of whether a one-to-many facial identification is Out-of-gallery, and thereby to reduce false positive identifications, wrongful arrests, and wasted investigative time. Interestingly, comparing the results of older deep CNN-based face matchers with newer ones suggests that the effectiveness of our Out-of-gallery detection approach emerges only with matchers trained using advanced margin-based loss functions. 

**Abstract (ZH)**: 一项重要的挑战在于一对一多识别中，探针图像中的人物可能或可能不在-gallery中，即可能是In-gallery或Out-of-gallery。以往的方法主要集中在找到相似度分数的合适阈值来检测rank-one结果是否为Out-of-gallery。我们提出了一种新的方法，利用rank-one结果对应身份的额外注册图像来预测rank-one结果是In-gallery还是Out-of-gallery。给定一组身份和图像，我们通过提取与rank-one身份对应的额外注册图像的排名来生成In-gallery和Out-of-gallery的训练数据。然后，我们训练一个分类器，利用该特征矢量预测rank-one结果是In-gallery还是Out-of-gallery。使用两个不同的数据集和四种不同的匹配器，我们展示了实验结果，表明我们的方法适用于 mugshot质量的探针图像，并且对于因模糊、分辨率降低、大气湍流和墨镜而退化的探针图像也同样适用。我们还分析了不同人口统计学组别的结果，显示In-gallery/Out-of-gallery分类的准确性在不同人口统计学群体中相似。我们的方法有潜力提供一种客观估计一对一多识别是否为Out-of-gallery的方法，从而减少误报、误捕和浪费的调查时间。有趣的是，将较早的基于深层CNN的面匹配器的结果与较新的匹配器的结果进行比较表明，我们的Out-of-gallery检测方法的有效性仅在使用先进的基于边界的损失函数进行训练的匹配器中出现。 

---
# Structural Equation-VAE: Disentangled Latent Representations for Tabular Data 

**Title (ZH)**: 结构方程-VAE: 分离潜变量表示的表格数据 

**Authors**: Ruiyu Zhang, Ce Zhao, Xin Zhao, Lin Nie, Wai-Fung Lam  

**Link**: [PDF](https://arxiv.org/pdf/2508.06347)  

**Abstract**: Learning interpretable latent representations from tabular data remains a challenge in deep generative modeling. We introduce SE-VAE (Structural Equation-Variational Autoencoder), a novel architecture that embeds measurement structure directly into the design of a variational autoencoder. Inspired by structural equation modeling, SE-VAE aligns latent subspaces with known indicator groupings and introduces a global nuisance latent to isolate construct-specific confounding variation. This modular architecture enables disentanglement through design rather than through statistical regularizers alone. We evaluate SE-VAE on a suite of simulated tabular datasets and benchmark its performance against a series of leading baselines using standard disentanglement metrics. SE-VAE consistently outperforms alternatives in factor recovery, interpretability, and robustness to nuisance variation. Ablation results reveal that architectural structure, rather than regularization strength, is the key driver of performance. SE-VAE offers a principled framework for white-box generative modeling in scientific and social domains where latent constructs are theory-driven and measurement validity is essential. 

**Abstract (ZH)**: 从表格数据中学习可解释的潜在表示仍然是深度生成建模中的一个挑战。SE-VAE（结构方程变量自编码器）是一种新颖的架构，它将测量结构直接嵌入到变分自编码器的设计中。受结构方程建模的启发，SE-VAE 将潜在子空间与已知指标分组对齐，并引入一个全局干扰潜在变量以隔离特定结构的混杂变异。这种模块化架构通过设计而非仅通过统计正则化来实现解耦。我们在一系列模拟表格数据集上评估了 SE-VAE，并使用标准解耦指标将其性能与一系列领先基准进行比较。SE-VAE 在因子恢复、可解释性和对干扰变异的稳健性方面始终优于其他替代方案。消融结果表明，架构结构而非正则化强度是性能的关键驱动因素。SE-VAE 为科学和社会领域中的白盒生成建模提供了一个有原则的框架，在这些领域中，潜在结构是理论驱动的，测量有效性是必不可少的。 

---
# Harnessing Adaptive Topology Representations for Zero-Shot Graph Question Answering 

**Title (ZH)**: 基于自适应拓扑表示的零样本图问答 

**Authors**: Yanbin Wei, Jiangyue Yan, Chun Kang, Yang Chen, Hua Liu, James T. Kwok, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.06345)  

**Abstract**: Large Multimodal Models (LMMs) have shown generalized zero-shot capabilities in diverse domain question-answering (QA) tasks, including graph QA that involves complex graph topologies. However, most current approaches use only a single type of graph representation, namely Topology Representation Form (TRF), such as prompt-unified text descriptions or style-fixed visual styles. Those "one-size-fits-all" approaches fail to consider the specific preferences of different models or tasks, often leading to incorrect or overly long responses. To address this, we first analyze the characteristics and weaknesses of existing TRFs, and then design a set of TRFs, denoted by $F_{ZS}$, tailored to zero-shot graph QA. We then introduce a new metric, Graph Response Efficiency (GRE), which measures the balance between the performance and the brevity in graph QA. Built on these, we develop the DynamicTRF framework, which aims to improve both the accuracy and conciseness of graph QA. To be specific, DynamicTRF first creates a TRF Preference (TRFP) dataset that ranks TRFs based on their GRE scores, to probe the question-specific TRF preferences. Then it trains a TRF router on the TRFP dataset, to adaptively assign the best TRF from $F_{ZS}$ for each question during the inference. Extensive experiments across 7 in-domain algorithmic graph QA tasks and 2 out-of-domain downstream tasks show that DynamicTRF significantly enhances the zero-shot graph QA of LMMs in terms of accuracy 

**Abstract (ZH)**: 标题：大型多 跨模态模型在零样本图问答任务中的偏好分析与动态调整框架 

---
# On Approximate MMS Allocations on Restricted Graph Classes 

**Title (ZH)**: 在受限图类上的近似MMS分配 

**Authors**: Václav Blažej, Michał Dębski ad Zbigniew Lonc, Marta Piecyk, Paweł Rzążewski  

**Link**: [PDF](https://arxiv.org/pdf/2508.06343)  

**Abstract**: We study the problem of fair division of a set of indivisible goods with connectivity constraints. Specifically, we assume that the goods are represented as vertices of a connected graph, and sets of goods allocated to the agents are connected subgraphs of this graph. We focus on the widely-studied maximin share criterion of fairness. It has been shown that an allocation satisfying this criterion may not exist even without connectivity constraints, i.e., if the graph of goods is complete. In view of this, it is natural to seek approximate allocations that guarantee each agent a connected bundle of goods with value at least a constant fraction of the maximin share value to the agent. It is known that for some classes of graphs, such as complete graphs, cycles, and $d$-claw-free graphs for any fixed $d$, such approximate allocations indeed exist. However, it is an open problem whether they exist for the class of all graphs.
In this paper, we continue the systematic study of the existence of approximate allocations on restricted graph classes. In particular, we show that such allocations exist for several well-studied classes, including block graphs, cacti, complete multipartite graphs, and split graphs. 

**Abstract (ZH)**: 我们研究具有连通性约束的一组不可分物品的公平分配问题。具体地，我们假设物品用连通图的顶点表示，分配给代理的物品集应为该图的连通子图。我们 focus 在广泛研究的最大化最小份额公平性标准上。已证明，在没有连通性约束的情况下，即使对于完全图，也可能不存在满足这一标准的分配。鉴于此，自然地，寻求保证每个代理获得具有至少为自身最大化最小份额值常数倍价值的连通物品集的近似分配变得合理。已知对于某些类别的图，如完全图、圈图和任意固定 $d$ 的 $d$-爪图，确实存在这样的近似分配。然而，对于所有图的类别来说，它们是否存在仍是一个公开问题。

在本文中，我们继续对受限图类上近似分配的存在性进行系统研究。特别是，我们证明了对于几种已经被广泛研究的图类，如块图、环图、完全.multipartite 图和分裂图，确实存在这样的近似分配。 

---
# FedMeNF: Privacy-Preserving Federated Meta-Learning for Neural Fields 

**Title (ZH)**: FedMeNF：隐私保护联邦元学习在神经场中的应用 

**Authors**: Junhyeog Yun, Minui Hong, Gunhee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.06301)  

**Abstract**: Neural fields provide a memory-efficient representation of data, which can effectively handle diverse modalities and large-scale data. However, learning to map neural fields often requires large amounts of training data and computations, which can be limited to resource-constrained edge devices. One approach to tackle this limitation is to leverage Federated Meta-Learning (FML), but traditional FML approaches suffer from privacy leakage. To address these issues, we introduce a novel FML approach called FedMeNF. FedMeNF utilizes a new privacy-preserving loss function that regulates privacy leakage in the local meta-optimization. This enables the local meta-learner to optimize quickly and efficiently without retaining the client's private data. Our experiments demonstrate that FedMeNF achieves fast optimization speed and robust reconstruction performance, even with few-shot or non-IID data across diverse data modalities, while preserving client data privacy. 

**Abstract (ZH)**: 基于神经场的联邦元学习方法FedMeNF提供了一种内存高效的数据表示，能够有效处理多种模态和大规模数据。然而，学习神经场的映射往往需要大量的训练数据和计算，这在资源受限的边缘设备上可能受到限制。为解决这一限制，我们提出了一种新的联邦元学习方法FedMeNF，它利用一种新的隐私保护损失函数调节局部元优化中的隐私泄漏，使局部元学习者能够快速高效地优化而不保留客户端的私有数据。实验结果表明，FedMeNF在少量样本或非IID数据下，能够实现快速优化速度和稳健的重建性能，同时保护客户端数据隐私。 

---
# OM2P: Offline Multi-Agent Mean-Flow Policy 

**Title (ZH)**: OM2P: 下线多 agents 平均流量策略 

**Authors**: Zhuoran Li, Xun Wang, Hai Zhong, Longbo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.06269)  

**Abstract**: Generative models, especially diffusion and flow-based models, have been promising in offline multi-agent reinforcement learning. However, integrating powerful generative models into this framework poses unique challenges. In particular, diffusion and flow-based policies suffer from low sampling efficiency due to their iterative generation processes, making them impractical in time-sensitive or resource-constrained settings. To tackle these difficulties, we propose OM2P (Offline Multi-Agent Mean-Flow Policy), a novel offline MARL algorithm to achieve efficient one-step action sampling. To address the misalignment between generative objectives and reward maximization, we introduce a reward-aware optimization scheme that integrates a carefully-designed mean-flow matching loss with Q-function supervision. Additionally, we design a generalized timestep distribution and a derivative-free estimation strategy to reduce memory overhead and improve training stability. Empirical evaluations on Multi-Agent Particle and MuJoCo benchmarks demonstrate that OM2P achieves superior performance, with up to a 3.8x reduction in GPU memory usage and up to a 10.8x speed-up in training time. Our approach represents the first to successfully integrate mean-flow model into offline MARL, paving the way for practical and scalable generative policies in cooperative multi-agent settings. 

**Abstract (ZH)**: 基于生成模型的 Offline 多智能体强化学习方法：OM2P 算法 

---
# Numerical Considerations in Weighted Model Counting 

**Title (ZH)**: 带权重模型计数的数值考虑 

**Authors**: Randal E. Bryant  

**Link**: [PDF](https://arxiv.org/pdf/2508.06264)  

**Abstract**: Weighted model counting computes the sum of the rational-valued weights associated with the satisfying assignments for a Boolean formula, where the weight of an assignment is given by the product of the weights assigned to the positive and negated variables comprising the assignment. Weighted model counting finds applications across a variety of domains including probabilistic reasoning and quantitative risk assessment.
Most weighted model counting programs operate by (explicitly or implicitly) converting the input formula into a form that enables arithmetic evaluation, using multiplication for conjunctions and addition for disjunctions. Performing this evaluation using floating-point arithmetic can yield inaccurate results, and it cannot quantify the level of precision achieved. Computing with rational arithmetic gives exact results, but it is costly in both time and space.
This paper describes how to combine multiple numeric representations to efficiently compute weighted model counts that are guaranteed to achieve a user-specified precision. When all weights are nonnegative, we prove that the precision loss of arithmetic evaluation using floating-point arithmetic can be tightly bounded. We show that supplementing a standard IEEE double-precision representation with a separate 64-bit exponent, a format we call extended-range double (ERD), avoids the underflow and overflow issues commonly encountered in weighted model counting. For problems with mixed negative and positive weights, we show that a combination of interval floating-point arithmetic and rational arithmetic can achieve the twin goals of efficiency and guaranteed precision. For our evaluations, we have devised especially challenging formulas and weight assignments, demonstrating the robustness of our approach. 

**Abstract (ZH)**: 加权模型计数计算与布尔公式满足赋值关联的有理值权重之和，其中赋值的权重由构成该赋值的正变量和负变量所分配的权重的乘积给出。加权模型计数在概率推理和定量风险评估等多个领域都有应用。

大多数加权模型计数程序通过（显式或隐式地）将输入公式转换为便于算术评估的形式来工作，使用乘法表示合取，加法表示析取。使用浮点算术进行这种评估可能会导致不准确的结果，并且无法量化所达到的精度。使用有理算术可以得到精确的结果，但时间和空间成本较高。

本文描述了如何结合多种数值表示来高效计算可保证达到用户指定精度的加权模型计数。当所有权重均为非负时，我们证明使用浮点算术进行算术评估的精度损失可以被紧密界。我们展示了通过使用扩展范围双精度（ERD）格式扩展标准的IEEE双精度表示，可以避免在加权模型计数中常见的下溢和上溢问题。对于具有混合正负权重的问题，我们展示了结合区间浮点算术和有理算术可以同时实现高效性和保证精度的目标。在我们的评估中，我们设计了特别具有挑战性的公式和权重分配，证明了我们方法的鲁棒性。 

---
# Synthetic Data Generation and Differential Privacy using Tensor Networks' Matrix Product States (MPS) 

**Title (ZH)**: 使用张量网络的矩阵乘积态合成数据生成与差分隐私 

**Authors**: Alejandro Moreno R., Desale Fentaw, Samuel Palmer, Raúl Salles de Padua, Ninad Dixit, Samuel Mugel, Roman Orús, Manuel Radons, Josef Menter, Ali Abedi  

**Link**: [PDF](https://arxiv.org/pdf/2508.06251)  

**Abstract**: Synthetic data generation is a key technique in modern artificial intelligence, addressing data scarcity, privacy constraints, and the need for diverse datasets in training robust models. In this work, we propose a method for generating privacy-preserving high-quality synthetic tabular data using Tensor Networks, specifically Matrix Product States (MPS). We benchmark the MPS-based generative model against state-of-the-art models such as CTGAN, VAE, and PrivBayes, focusing on both fidelity and privacy-preserving capabilities. To ensure differential privacy (DP), we integrate noise injection and gradient clipping during training, enabling privacy guarantees via Rényi Differential Privacy accounting. Across multiple metrics analyzing data fidelity and downstream machine learning task performance, our results show that MPS outperforms classical models, particularly under strict privacy constraints. This work highlights MPS as a promising tool for privacy-aware synthetic data generation. By combining the expressive power of tensor network representations with formal privacy mechanisms, the proposed approach offers an interpretable and scalable alternative for secure data sharing. Its structured design facilitates integration into sensitive domains where both data quality and confidentiality are critical. 

**Abstract (ZH)**: 合成数据生成是现代人工智能的关键技术，用于解决数据稀缺性、隐私限制以及训练 robust 模型所需多样化数据集的问题。在本文中，我们提出了一种使用张量网络（具体为矩阵积态 MPS）生成保隐私高质量合成表格数据的方法。我们将基于 MPS 的生成模型与当前最先进的模型（如 CTGAN、VAE 和 PrivBayes）进行基准测试，重点关注保真度和保隐私能力。通过在训练过程中集成噪声注入和梯度修剪，以 Rényi 差分隐私进行隐私保障计算。通过对多个评估数据保真度和下游机器学习任务性能的指标进行分析，我们的结果表明，在严格隐私约束下，MPS 优于经典模型。本文强调 MPS 是一种有前景的隐私感知合成数据生成工具。通过结合张量网络表示的强大表达能力和正式的隐私机制，所提出的方法为安全数据共享提供了可解释且可扩展的替代方案。其结构化设计便于将其集成到数据质量和保密性至关重要的敏感领域。 

---
# Membership Inference Attack with Partial Features 

**Title (ZH)**: 部分特征下的成员推断攻击 

**Authors**: Xurun Wang, Guangrui Liu, Xinjie Li, Haoyu He, Lin Yao, Weizhe Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.06244)  

**Abstract**: Machine learning models have been shown to be susceptible to membership inference attack, which can be used to determine whether a given sample appears in the training data. Existing membership inference methods commonly assume that the adversary has full access to the features of the target sample. This assumption, however, does not hold in many real-world scenarios where only partial features information is available, thereby limiting the applicability of these methods. In this work, we study an inference scenario where the adversary observes only partial features of each sample and aims to infer whether this observed subset was present in the training set of the target model. We define this problem as Partial Feature Membership Inference (PFMI). To address this problem, we propose MRAD (Memory-guided Reconstruction and Anomaly Detection), a two-stage attack framework. In the first stage, MRAD optimizes the unknown feature values to minimize the loss of the sample. In the second stage, it measures the deviation between the reconstructed sample and the training distribution using anomaly detection. Empirical results demonstrate that MRAD is effective across a range of datasets, and maintains compatibility with various off-the-shelf anomaly detection techniques. For example, on STL-10, our attack achieves an AUC of around 0.6 even with 40% of the missing features. 

**Abstract (ZH)**: 机器学习模型在面对仅拥有目标样本部分特征信息的成员推理攻击时的可攻击性研究：一种基于记忆引导重构与异常检测的两阶段攻击框架（Partial Feature Membership Inference: A Memory-guided Reconstruction and Anomaly Detection Framework） 

---
# InfoCausalQA:Can Models Perform Non-explicit Causal Reasoning Based on Infographic? 

**Title (ZH)**: InfoCausalQA：模型能在信息图表基础上进行非显式因果推理吗？ 

**Authors**: Keummin Ka, Junhyeong Park, Jahyun Jeon, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.06220)  

**Abstract**: Recent advances in Vision-Language Models (VLMs) have demonstrated impressive capabilities in perception and reasoning. However, the ability to perform causal inference -- a core aspect of human cognition -- remains underexplored, particularly in multimodal settings. In this study, we introduce InfoCausalQA, a novel benchmark designed to evaluate causal reasoning grounded in infographics that combine structured visual data with textual context. The benchmark comprises two tasks: Task 1 focuses on quantitative causal reasoning based on inferred numerical trends, while Task 2 targets semantic causal reasoning involving five types of causal relations: cause, effect, intervention, counterfactual, and temporal. We manually collected 494 infographic-text pairs from four public sources and used GPT-4o to generate 1,482 high-quality multiple-choice QA pairs. These questions were then carefully revised by humans to ensure they cannot be answered based on surface-level cues alone but instead require genuine visual grounding. Our experimental results reveal that current VLMs exhibit limited capability in computational reasoning and even more pronounced limitations in semantic causal reasoning. Their significantly lower performance compared to humans indicates a substantial gap in leveraging infographic-based information for causal inference. Through InfoCausalQA, we highlight the need for advancing the causal reasoning abilities of multimodal AI systems. 

**Abstract (ZH)**: Recent Advances in Vision-Language Models: Evaluating Causal Reasoning in Multimodal Settings with InfoCausalQA 

---
# Reparameterization Proximal Policy Optimization 

**Title (ZH)**: 重参数近端策略优化 

**Authors**: Hai Zhong, Xun Wang, Zhuoran Li, Longbo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.06214)  

**Abstract**: Reparameterization policy gradient (RPG) is promising for improving sample efficiency by leveraging differentiable dynamics. However, a critical barrier is its training instability, where high-variance gradients can destabilize the learning process. To address this, we draw inspiration from Proximal Policy Optimization (PPO), which uses a surrogate objective to enable stable sample reuse in the model-free setting. We first establish a connection between this surrogate objective and RPG, which has been largely unexplored and is non-trivial. Then, we bridge this gap by demonstrating that the reparameterization gradient of a PPO-like surrogate objective can be computed efficiently using backpropagation through time. Based on this key insight, we propose Reparameterization Proximal Policy Optimization (RPO), a stable and sample-efficient RPG-based method. RPO enables multiple epochs of stable sample reuse by optimizing a clipped surrogate objective tailored for RPG, while being further stabilized by Kullback-Leibler (KL) divergence regularization and remaining fully compatible with existing variance reduction methods. We evaluate RPO on a suite of challenging locomotion and manipulation tasks, where experiments demonstrate that our method achieves superior sample efficiency and strong performance. 

**Abstract (ZH)**: 基于重参数化 proximal 策略优化的重参数化策略梯度（RPG-RPO）方法 

---
# Graph Federated Learning for Personalized Privacy Recommendation 

**Title (ZH)**: 图联邦学习在个性化隐私推荐中的应用 

**Authors**: Ce Na, Kai Yang, Dengzhao Fang, Yu Li, Jingtong Gao, Chengcheng Zhu, Jiale Zhang, Xiaobing Sun, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2508.06208)  

**Abstract**: Federated recommendation systems (FedRecs) have gained significant attention for providing privacy-preserving recommendation services. However, existing FedRecs assume that all users have the same requirements for privacy protection, i.e., they do not upload any data to the server. The approaches overlook the potential to enhance the recommendation service by utilizing publicly available user data. In real-world applications, users can choose to be private or public. Private users' interaction data is not shared, while public users' interaction data can be shared. Inspired by the issue, this paper proposes a novel Graph Federated Learning for Personalized Privacy Recommendation (GFed-PP) that adapts to different privacy requirements while improving recommendation performance. GFed-PP incorporates the interaction data of public users to build a user-item interaction graph, which is then used to form a user relationship graph. A lightweight graph convolutional network (GCN) is employed to learn each user's user-specific personalized item embedding. To protect user privacy, each client learns the user embedding and the scoring function locally. Additionally, GFed-PP achieves optimization of the federated recommendation framework through the initialization of item embedding on clients and the aggregation of the user relationship graph on the server. Experimental results demonstrate that GFed-PP significantly outperforms existing methods for five datasets, offering superior recommendation accuracy without compromising privacy. This framework provides a practical solution for accommodating varying privacy preferences in federated recommendation systems. 

**Abstract (ZH)**: 联邦个性化隐私推荐的图联邦学习（GFed-PP） 

---
# Classification is a RAG problem: A case study on hate speech detection 

**Title (ZH)**: 分类分类是生成式检索增强生成（REDAgent）任务的一种研究：仇恨言论检测案例研究 

**Authors**: Richard Willats, Josh Pennington, Aravind Mohan, Bertie Vidgen  

**Link**: [PDF](https://arxiv.org/pdf/2508.06204)  

**Abstract**: Robust content moderation requires classification systems that can quickly adapt to evolving policies without costly retraining. We present classification using Retrieval-Augmented Generation (RAG), which shifts traditional classification tasks from determining the correct category in accordance with pre-trained parameters to evaluating content in relation to contextual knowledge retrieved at inference. In hate speech detection, this transforms the task from "is this hate speech?" to "does this violate the hate speech policy?"
Our Contextual Policy Engine (CPE) - an agentic RAG system - demonstrates this approach and offers three key advantages: (1) robust classification accuracy comparable to leading commercial systems, (2) inherent explainability via retrieved policy segments, and (3) dynamic policy updates without model retraining. Through three experiments, we demonstrate strong baseline performance and show that the system can apply fine-grained policy control by correctly adjusting protection for specific identity groups without requiring retraining or compromising overall performance. These findings establish that RAG can transform classification into a more flexible, transparent, and adaptable process for content moderation and wider classification problems. 

**Abstract (ZH)**: 鲁棒的内容审核需要能够快速适应 evolving policies 的分类系统，无需昂贵的重新训练。我们提出使用检索增强生成（RAG）的分类方法，将传统的分类任务从根据预训练参数确定正确的类别转向在推理时根据检索到的上下文知识评估内容。在仇恨言论检测中，这一方法将任务从“这是否是仇恨言论？”转变为“这是否违反了仇恨言论政策？”

我们的上下文政策引擎（CPE）——一个自主的RAG系统——展示了这种方法，并提供以下三大优势：（1）与领先商业系统相当的稳健分类准确性，（2）通过检索到的政策片段固有的解释性，（3）无需模型重新训练即可进行动态政策更新。通过三项实验，我们展示了强大的基线性能，并证明了系统可以通过正确调整特定身份群体的保护措施来应用细致的政策控制，而无需重新训练或牺牲整体性能。这些发现确立了RAG可以将分类转变为内容审核和更广泛分类问题中更具灵活性、透明性和适应性的过程。 

---
# Benchmarking Pretrained Molecular Embedding Models For Molecular Representation Learning 

**Title (ZH)**: 预训练分子嵌入模型的分子表示学习基准研究 

**Authors**: Mateusz Praski, Jakub Adamczyk, Wojciech Czech  

**Link**: [PDF](https://arxiv.org/pdf/2508.06199)  

**Abstract**: Pretrained neural networks have attracted significant interest in chemistry and small molecule drug design. Embeddings from these models are widely used for molecular property prediction, virtual screening, and small data learning in molecular chemistry. This study presents the most extensive comparison of such models to date, evaluating 25 models across 25 datasets. Under a fair comparison framework, we assess models spanning various modalities, architectures, and pretraining strategies. Using a dedicated hierarchical Bayesian statistical testing model, we arrive at a surprising result: nearly all neural models show negligible or no improvement over the baseline ECFP molecular fingerprint. Only the CLAMP model, which is also based on molecular fingerprints, performs statistically significantly better than the alternatives. These findings raise concerns about the evaluation rigor in existing studies. We discuss potential causes, propose solutions, and offer practical recommendations. 

**Abstract (ZH)**: 预训练神经网络在化学和小分子药物设计中的应用引起了显著兴趣。这些模型的嵌入广泛用于分子性质预测、虚拟筛选和分子化学中的小数据学习。本研究迄今为止进行了最广泛的此类模型比较，评估了25个模型在25个数据集上的表现。在公平比较框架下，我们评估了涵盖不同模态、架构和预训练策略的模型。使用专门的分层贝叶斯统计测试模型，我们得出一个出人意料的结果：几乎所有神经网络模型在基线ECFP分子指纹方面几乎没有或没有任何改进。只有基于分子指纹的CLAMP模型表现出统计显著性的优越性。这些发现引发了对现有研究中评估严谨性的担忧。我们讨论了潜在的原因，提出了解决方案，并提供了实用建议。 

---
# Differentially Private Federated Clustering with Random Rebalancing 

**Title (ZH)**: 不同隐私保护的联邦聚类方法：随机重新平衡 

**Authors**: Xiyuan Yang, Shengyuan Hu, Soyeon Kim, Tian Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.06183)  

**Abstract**: Federated clustering aims to group similar clients into clusters and produce one model for each cluster. Such a personalization approach typically improves model performance compared with training a single model to serve all clients, but can be more vulnerable to privacy leakage. Directly applying client-level differentially private (DP) mechanisms to federated clustering could degrade the utilities significantly. We identify that such deficiencies are mainly due to the difficulties of averaging privacy noise within each cluster (following standard privacy mechanisms), as the number of clients assigned to the same clusters is uncontrolled. To this end, we propose a simple and effective technique, named RR-Cluster, that can be viewed as a light-weight add-on to many federated clustering algorithms. RR-Cluster achieves reduced privacy noise via randomly rebalancing cluster assignments, guaranteeing a minimum number of clients assigned to each cluster. We analyze the tradeoffs between decreased privacy noise variance and potentially increased bias from incorrect assignments and provide convergence bounds for RR-Clsuter. Empirically, we demonstrate the RR-Cluster plugged into strong federated clustering algorithms results in significantly improved privacy/utility tradeoffs across both synthetic and real-world datasets. 

**Abstract (ZH)**: 联邦聚类旨在将相似的客户端分组到聚类中，并为每个聚类生成一个模型。这种个性化方法通常与为所有客户端训练单一模型相比能提高模型性能，但也更容易泄露隐私。直接将客户端级别的差分隐私（DP）机制应用于联邦聚类可能会显著降低效用。我们认为这种缺陷主要源于在每个聚类内平均隐私噪声的困难（遵循标准的隐私机制），因为分配到相同聚类的客户端数量不受控制。为此，我们提出了一种简单而有效的技术，称为RR-Cluster，它可以被视为对许多联邦聚类算法的轻量级补充。RR-Cluster通过随机重新平衡聚类分配来减少隐私噪声，确保每个聚类分配的客户端数量下限。我们分析了减少隐私噪声方差与潜在增加由于错误分配引起的偏差之间的权衡，并提供了RR-Cluster的收敛界。实证研究表明，RR-Cluster嵌入到强大的联邦聚类算法中，在合成数据集和真实世界数据集中都能显著改善隐私/效用权衡。 

---
# One Size Does Not Fit All: A Distribution-Aware Sparsification for More Precise Model Merging 

**Title (ZH)**: 大小不一：一种基于分布的稀疏化方法以实现更精确的模型合并 

**Authors**: Yingfeng Luo, Dingyang Lin, Junxin Wang, Ziqiang Xu, Kaiyan Chang, Tong Zheng, Bei Li, Anxiang Ma, Tong Xiao, Zhengtao Yu, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.06163)  

**Abstract**: Model merging has emerged as a compelling data-free paradigm for multi-task learning, enabling the fusion of multiple fine-tuned models into a single, powerful entity. A key technique in merging methods is sparsification, which prunes redundant parameters from task vectors to mitigate interference. However, prevailing approaches employ a ``one-size-fits-all'' strategy, applying a uniform sparsity ratio that overlooks the inherent structural and statistical heterogeneity of model parameters. This often leads to a suboptimal trade-off, where critical parameters are inadvertently pruned while less useful ones are retained. To address this limitation, we introduce \textbf{TADrop} (\textbf{T}ensor-wise \textbf{A}daptive \textbf{Drop}), an adaptive sparsification strategy that respects this heterogeneity. Instead of a global ratio, TADrop assigns a tailored sparsity level to each parameter tensor based on its distributional properties. The core intuition is that tensors with denser, more redundant distributions can be pruned aggressively, while sparser, more critical ones are preserved. As a simple and plug-and-play module, we validate TADrop by integrating it with foundational, classic, and SOTA merging methods. Extensive experiments across diverse tasks (vision, language, and multimodal) and models (ViT, BEiT) demonstrate that TADrop consistently and significantly boosts their performance. For instance, when enhancing a leading merging method, it achieves an average performance gain of 2.0\% across 8 ViT-B/32 tasks. TADrop provides a more effective way to mitigate parameter interference by tailoring sparsification to the model's structure, offering a new baseline for high-performance model merging. 

**Abstract (ZH)**: Tensor-wise Adaptive Drop: An Adaptive Sparsification Strategy for High-Performance Model Merging 

---
# Semantic Item Graph Enhancement for Multimodal Recommendation 

**Title (ZH)**: 多模态推荐中的语义项图增强 

**Authors**: Xiaoxiong Zhang, Xin Zhou, Zhiwei Zeng, Dusit Niyato, Zhiqi Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.06154)  

**Abstract**: Multimodal recommendation systems have attracted increasing attention for their improved performance by leveraging items' multimodal information. Prior methods often build modality-specific item-item semantic graphs from raw modality features and use them as supplementary structures alongside the user-item interaction graph to enhance user preference learning. However, these semantic graphs suffer from semantic deficiencies, including (1) insufficient modeling of collaborative signals among items and (2) structural distortions introduced by noise in raw modality features, ultimately compromising performance. To address these issues, we first extract collaborative signals from the interaction graph and infuse them into each modality-specific item semantic graph to enhance semantic modeling. Then, we design a modulus-based personalized embedding perturbation mechanism that injects perturbations with modulus-guided personalized intensity into embeddings to generate contrastive views. This enables the model to learn noise-robust representations through contrastive learning, thereby reducing the effect of structural noise in semantic graphs. Besides, we propose a dual representation alignment mechanism that first aligns multiple semantic representations via a designed Anchor-based InfoNCE loss using behavior representations as anchors, and then aligns behavior representations with the fused semantics by standard InfoNCE, to ensure representation consistency. Extensive experiments on four benchmark datasets validate the effectiveness of our framework. 

**Abstract (ZH)**: 多ar模式推荐系统通过结合项目的大规模多媒体信息正在逐渐吸引越来越多的关注。通常，，，，，项目特定的项目-项目语义图从原始项目特征开始并它们作为补充结构与内容-项目交互图并结合使用以增强用户偏好学习。然而，这种语义图存在语义缺陷，包括（1
user
请下面的标题翻译成中文，要符合学术规范：
Mult Modal Recommendation System Through Leveraging Items' Multim Modal Information 

---
# FMCE-Net++: Feature Map Convergence Evaluation and Training 

**Title (ZH)**: FMCE-Net++: 特征图收敛评估与训练 

**Authors**: Zhibo Zhu, Renyu Huang, Lei He  

**Link**: [PDF](https://arxiv.org/pdf/2508.06109)  

**Abstract**: Deep Neural Networks (DNNs) face interpretability challenges due to their opaque internal representations. While Feature Map Convergence Evaluation (FMCE) quantifies module-level convergence via Feature Map Convergence Scores (FMCS), it lacks experimental validation and closed-loop integration. To address this limitation, we propose FMCE-Net++, a novel training framework that integrates a pretrained, frozen FMCE-Net as an auxiliary head. This module generates FMCS predictions, which, combined with task labels, jointly supervise backbone optimization through a Representation Auxiliary Loss. The RAL dynamically balances the primary classification loss and feature convergence optimization via a tunable \Representation Abstraction Factor. Extensive experiments conducted on MNIST, CIFAR-10, FashionMNIST, and CIFAR-100 demonstrate that FMCE-Net++ consistently enhances model performance without architectural modifications or additional data. Key experimental outcomes include accuracy gains of $+1.16$ pp (ResNet-50/CIFAR-10) and $+1.08$ pp (ShuffleNet v2/CIFAR-100), validating that FMCE-Net++ can effectively elevate state-of-the-art performance ceilings. 

**Abstract (ZH)**: 深度神经网络（DNNs）由于其不透明的内部表示面临可解释性挑战。为了解决这一问题，我们提出了一种新的训练框架FMCE-Net++，该框架集成了一个预训练且冻结的FMCE-Net作为辅助头。该模块生成特征图收敛评分（FMCS）预测，结合任务标签，通过表示辅助损失（RAL）联合监督主干优化。表示抽象因子动态平衡主要分类损失和特征收敛优化。在MNIST、CIFAR-10、FashionMNIST和CIFAR-100上的 extensive 实验表明，FMCE-Net++在不进行架构修改或增加数据的情况下，一致地提升了模型性能。关键实验结果包括ResNet-50/CIFAR-10上 accuracy 提升1.16个百分点和ShuffleNet v2/CIFAR-100上 accuracy 提升1.08个百分点，验证了FMCE-Net++能够有效提升前沿性能天花板。 

---
# Mask & Match: Learning to Recognize Handwritten Math with Self-Supervised Attention 

**Title (ZH)**: 遮罩与匹配：学习识别手写数学公式的一种自监督注意力方法 

**Authors**: Shree Mitra, Ritabrata Chakraborty, Nilkanta Sahu  

**Link**: [PDF](https://arxiv.org/pdf/2508.06107)  

**Abstract**: Recognizing handwritten mathematical expressions (HMER) is a challenging task due to the inherent two-dimensional structure, varying symbol scales, and complex spatial relationships among symbols. In this paper, we present a self-supervised learning (SSL) framework for HMER that eliminates the need for expensive labeled data. Our approach begins by pretraining an image encoder using a combination of global and local contrastive loss, enabling the model to learn both holistic and fine-grained representations. A key contribution of this work is a novel self-supervised attention network, which is trained using a progressive spatial masking strategy. This attention mechanism is designed to learn semantically meaningful focus regions, such as operators, exponents, and nested mathematical notation, without requiring any supervision. The progressive masking curriculum encourages the network to become increasingly robust to missing or occluded visual information, ultimately improving structural understanding. Our complete pipeline consists of (1) self-supervised pretraining of the encoder, (2) self-supervised attention learning, and (3) supervised fine-tuning with a transformer decoder to generate LATEX sequences. Extensive experiments on CROHME benchmarks demonstrate that our method outperforms existing SSL and fully supervised baselines, validating the effectiveness of our progressive attention mechanism in enhancing HMER performance. Our codebase can be found here. 

**Abstract (ZH)**: 基于自监督学习的手写数学表达识别（HMER） 

---
# MeanAudio: Fast and Faithful Text-to-Audio Generation with Mean Flows 

**Title (ZH)**: MeanAudio: 快速且忠实的文本到音频生成方法 

**Authors**: Xiquan Li, Junxi Liu, Yuzhe Liang, Zhikang Niu, Wenxi Chen, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.06098)  

**Abstract**: Recent developments in diffusion- and flow- based models have significantly advanced Text-to-Audio Generation (TTA). While achieving great synthesis quality and controllability, current TTA systems still suffer from slow inference speed, which significantly limits their practical applicability. This paper presents MeanAudio, a novel MeanFlow-based model tailored for fast and faithful text-to-audio generation. Built on a Flux-style latent transformer, MeanAudio regresses the average velocity field during training, enabling fast generation by mapping directly from the start to the endpoint of the flow trajectory. By incorporating classifier-free guidance (CFG) into the training target, MeanAudio incurs no additional cost in the guided sampling process. To further stabilize training, we propose an instantaneous-to-mean curriculum with flow field mix-up, which encourages the model to first learn the foundational instantaneous dynamics, and then gradually adapt to mean flows. This strategy proves critical for enhancing training efficiency and generation quality. Experimental results demonstrate that MeanAudio achieves state-of-the-art performance in single-step audio generation. Specifically, it achieves a real time factor (RTF) of 0.013 on a single NVIDIA RTX 3090, yielding a 100x speedup over SOTA diffusion-based TTA systems. Moreover, MeanAudio also demonstrates strong performance in multi-step generation, enabling smooth and coherent transitions across successive synthesis steps. 

**Abstract (ZH)**: Recent developments in MeanFlow-based models have significantly advanced Text-to-Audio Generation (TTA). This paper presents MeanAudio, a novel MeanFlow-based model tailored for fast and faithful text-to-audio generation. 

---
# Towards MR-Based Trochleoplasty Planning 

**Title (ZH)**: 基于磁共振的 trochleoplasty 手术规划研究 

**Authors**: Michael Wehrli, Alicia Durrer, Paul Friedrich, Sidaty El Hadramy, Edwin Li, Luana Brahaj, Carol C. Hasler, Philippe C. Cattin  

**Link**: [PDF](https://arxiv.org/pdf/2508.06076)  

**Abstract**: To treat Trochlear Dysplasia (TD), current approaches rely mainly on low-resolution clinical Magnetic Resonance (MR) scans and surgical intuition. The surgeries are planned based on surgeons experience, have limited adoption of minimally invasive techniques, and lead to inconsistent outcomes. We propose a pipeline that generates super-resolved, patient-specific 3D pseudo-healthy target morphologies from conventional clinical MR scans. First, we compute an isotropic super-resolved MR volume using an Implicit Neural Representation (INR). Next, we segment femur, tibia, patella, and fibula with a multi-label custom-trained network. Finally, we train a Wavelet Diffusion Model (WDM) to generate pseudo-healthy target morphologies of the trochlear region. In contrast to prior work producing pseudo-healthy low-resolution 3D MR images, our approach enables the generation of sub-millimeter resolved 3D shapes compatible for pre- and intraoperative use. These can serve as preoperative blueprints for reshaping the femoral groove while preserving the native patella articulation. Furthermore, and in contrast to other work, we do not require a CT for our pipeline - reducing the amount of radiation. We evaluated our approach on 25 TD patients and could show that our target morphologies significantly improve the sulcus angle (SA) and trochlear groove depth (TGD). The code and interactive visualization are available at this https URL. 

**Abstract (ZH)**: 用于治疗)*/
膝盂发育不良（Trochlear Dysplasia，TD），当前的方法主要依赖低分辨率的临床磁共振（MR）扫描和手术直觉。手术计划基于外科医生的经验，有限地采用了微创技术，导致手术结果不一致。我们提出了一种生成从常规临床MR扫描中提取的高分辨率、患者特定的3D伪健康目标形态的流水线。首先，我们使用隐式神经表示（INR）计算一个各向同性的高分辨率MR体积。接下来，我们使用多标签自定义训练网络对股骨、胫骨、髌骨和腓骨进行分割。最后，我们训练一种小波扩散模型（WDM）以生成膝盂区域的伪健康目标形态。与先前生成伪健康低分辨率3D MR图像的方法不同，我们的方法可以生成适用于术前和术中使用的亚毫米级解析度的3D形状。这些可以作为术前蓝图，用于重塑股骨沟，同时保留原有的髌骨关节面。此外，与其它方法不同，我们的流水线不需要CT扫描——从而减少了辐射量。我们在25例TD患者的评估中展示了我们的目标形态显著改善了膝盂沟角（Sulcus Angle，SA）和膝盂沟深度（Trochlear Groove Depth，TGD）。代码和交互式可视化可在以下链接获取。 

---
# Architecture-Aware Generalization Bounds for Temporal Networks: Theory and Fair Comparison Methodology 

**Title (ZH)**: 面向架构的时空网络泛化界理论与公平比较方法学 

**Authors**: Barak Gahtan, Alex M. Bronstein  

**Link**: [PDF](https://arxiv.org/pdf/2508.06066)  

**Abstract**: Deep temporal architectures such as Temporal Convolutional Networks (TCNs) achieve strong predictive performance on sequential data, yet theoretical understanding of their generalization remains limited. We address this gap by providing both the first non-vacuous, architecture-aware generalization bounds for deep temporal models and a principled evaluation methodology.
For exponentially $\beta$-mixing sequences, we derive bounds scaling as $ O\!\Bigl(R\,\sqrt{\tfrac{D\,p\,n\,\log N}{N}}\Bigr), $ where $D$ is network depth, $p$ kernel size, $n$ input dimension, and $R$ weight norm. Our delayed-feedback blocking mechanism transforms dependent samples into effectively independent ones while discarding only $O(1/\log N)$ of the data, yielding $\sqrt{D}$ scaling instead of exponential, implying that doubling depth requires approximately quadrupling the training data.
We also introduce a fair-comparison methodology that fixes the effective sample size to isolate the effect of temporal structure from information content. Under $N_{\text{eff}}=2{,}000$, strongly dependent sequences ($\rho=0.8$) exhibit $\approx76\%$ smaller generalization gaps than weakly dependent ones ($\rho=0.2$), challenging the intuition that dependence is purely detrimental. Yet convergence rates diverge from theory: weak dependencies follow $N_{\text{eff}}^{-1.21}$ scaling and strong dependencies follow $N_{\text{eff}}^{-0.89}$, both steeper than the predicted $N^{-0.5}$. These findings reveal that temporal dependence can enhance learning under fixed information budgets, while highlighting gaps between theory and practice that motivate future research. 

**Abstract (ZH)**: 深度时间架构如时间卷积网络（TCNs）在序列数据上实现了强大的预测性能，但对它们的泛化能力仍缺乏理论理解。本文通过提供第一个非平凡的、架构感知的深度时间模型泛化界，并提出了一种基本原则的评估方法来弥补这一差距。 

---
# ThematicPlane: Bridging Tacit User Intent and Latent Spaces for Image Generation 

**Title (ZH)**: 主题平面：连接隐含用户意图与潜在空间的图像生成方法 

**Authors**: Daniel Lee, Nikhil Sharma, Donghoon Shin, DaEun Choi, Harsh Sharma, Jeonghwan Kim, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2508.06065)  

**Abstract**: Generative AI has made image creation more accessible, yet aligning outputs with nuanced creative intent remains challenging, particularly for non-experts. Existing tools often require users to externalize ideas through prompts or references, limiting fluid exploration. We introduce ThematicPlane, a system that enables users to navigate and manipulate high-level semantic concepts (e.g., mood, style, or narrative tone) within an interactive thematic design plane. This interface bridges the gap between tacit creative intent and system control. In our exploratory study (N=6), participants engaged in divergent and convergent creative modes, often embracing unexpected results as inspiration or iteration cues. While they grounded their exploration in familiar themes, differing expectations of how themes mapped to outputs revealed a need for more explainable controls. Overall, ThematicPlane fosters expressive, iterative workflows and highlights new directions for intuitive, semantics-driven interaction in generative design tools. 

**Abstract (ZH)**: 生成式AI使图像创作更加易于访问，但使输出与细腻的创作意图保持一致仍然具有挑战性，特别是在非专业人士中尤为重要。现有工具通常需要用户通过提示等形式外部化想法 限制了流畅的探索。我们引入了主题平面 允
user
主题平面 pestic-testid
-testid
主题平面（ThemePlane） 

---
# Adaptive Heterogeneous Graph Neural Networks: Bridging Heterophily and Heterogeneity 

**Title (ZH)**: 自适应异质图神经网络：连接异构性和异质性 

**Authors**: Qin Chen, Guojie Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.06034)  

**Abstract**: Heterogeneous graphs (HGs) are common in real-world scenarios and often exhibit heterophily. However, most existing studies focus on either heterogeneity or heterophily in isolation, overlooking the prevalence of heterophilic HGs in practical applications. Such ignorance leads to their performance degradation. In this work, we first identify two main challenges in modeling heterophily HGs: (1) varying heterophily distributions across hops and meta-paths; (2) the intricate and often heterophily-driven diversity of semantic information across different meta-paths. Then, we propose the Adaptive Heterogeneous Graph Neural Network (AHGNN) to tackle these challenges. AHGNN employs a heterophily-aware convolution that accounts for heterophily distributions specific to both hops and meta-paths. It then integrates messages from diverse semantic spaces using a coarse-to-fine attention mechanism, which filters out noise and emphasizes informative signals. Experiments on seven real-world graphs and twenty baselines demonstrate the superior performance of AHGNN, particularly in high-heterophily situations. 

**Abstract (ZH)**: 异质图（HGs）在现实场景中很常见，常表现出异质性。然而，现有的大多数研究要么孤立地关注异质性，要么关注异质性，忽视了实践应用中异质性HG的普遍性。这种忽视导致了它们性能的下降。在本文中，我们首先识别出建模异质性HG的两个主要挑战：（1）跨跃点和元路径的异质性分布变化；（2）不同元路径上复杂的且往往由异质性驱动的语义信息多样性。然后，我们提出了自适应异质图神经网络（AHGNN）来应对这些挑战。AHGNN采用一种意识到异质性的卷积，考虑了跨越跃点和元路径的特定异质性分布。它然后通过一种粗到细的注意力机制整合来自不同语义空间的消息，该机制过滤掉噪声并强调有用信号。在七个真实世界图和二十个基线上的实验表明，AHGNN在高异质性情况下表现尤为出色。 

---
# Crisp Attention: Regularizing Transformers via Structured Sparsity 

**Title (ZH)**: crisp Attention: 通过结构化稀疏性正则化 Transformers 

**Authors**: Sagar Gandhi, Vishal Gandhi  

**Link**: [PDF](https://arxiv.org/pdf/2508.06016)  

**Abstract**: The quadratic computational cost of the self-attention mechanism is a primary challenge in scaling Transformer models. While attention sparsity is widely studied as a technique to improve computational efficiency, it is almost universally assumed to come at the cost of model accuracy. In this paper, we report a surprising counter-example to this common wisdom. By introducing structured, post-hoc sparsity to the attention mechanism of a DistilBERT model during fine-tuning on the SST-2 sentiment analysis task, we find that model accuracy improves significantly. Our model with 80\% attention sparsity achieves a validation accuracy of 91.59\%, a 0.97\% absolute improvement over the dense baseline. We hypothesize that this phenomenon is due to sparsity acting as a powerful implicit regularizer, preventing the model from overfitting by forcing it to make predictions with a more constrained and robust set of features. Our work recasts attention sparsity not just as a tool for computational efficiency, but as a potential method for improving the generalization and performance of Transformer models. 

**Abstract (ZH)**: 自注意力机制的计算成本呈二次增长是扩展Transformer模型的主要挑战。虽然注意力稀疏性作为一种提高计算效率的技术得到广泛应用，但几乎普遍认为这会以降低模型准确度为代价。本文报道了一个令人惊讶的反例。通过在SST-2情感分析任务微调DistilBERT模型时引入结构化的后验稀疏性，我们发现模型准确度显著提高。我们提出的80%注意力稀疏模型在验证集上的准确率达到91.59%，比密集基线提高了0.97%。我们推测，这一现象是由于稀疏性作为强有力的隐式正则化手段，通过迫使模型以更受限和 robust 的特征集做出预测，防止过拟合。本文将注意力稀疏性重新定义为不仅是一种计算效率工具，也是一种提高Transformer模型泛化能力和性能的方法。 

---
# ETA: Energy-based Test-time Adaptation for Depth Completion 

**Title (ZH)**: 基于能量的测试时适配以完成深度估计 

**Authors**: Younjoon Chung, Hyoungseob Park, Patrick Rim, Xiaoran Zhang, Jihe He, Ziyao Zeng, Safa Cicek, Byung-Woo Hong, James S. Duncan, Alex Wong  

**Link**: [PDF](https://arxiv.org/pdf/2508.05989)  

**Abstract**: We propose a method for test-time adaptation of pretrained depth completion models. Depth completion models, trained on some ``source'' data, often predict erroneous outputs when transferred to ``target'' data captured in novel environmental conditions due to a covariate shift. The crux of our method lies in quantifying the likelihood of depth predictions belonging to the source data distribution. The challenge is in the lack of access to out-of-distribution (target) data prior to deployment. Hence, rather than making assumptions regarding the target distribution, we utilize adversarial perturbations as a mechanism to explore the data space. This enables us to train an energy model that scores local regions of depth predictions as in- or out-of-distribution. We update the parameters of pretrained depth completion models at test time to minimize energy, effectively aligning test-time predictions to those of the source distribution. We call our method ``Energy-based Test-time Adaptation'', or ETA for short. We evaluate our method across three indoor and three outdoor datasets, where ETA improve over the previous state-of-the-art method by an average of 6.94% for outdoors and 10.23% for indoors. Project Page: this https URL. 

**Abstract (ZH)**: 基于能量的方法在测试时适应预训练的深度完成模型 

---
# DAFMSVC: One-Shot Singing Voice Conversion with Dual Attention Mechanism and Flow Matching 

**Title (ZH)**: DAFMSVC：基于双注意力机制和流动匹配的一次性歌声转换 

**Authors**: Wei Chen, Binzhu Sha, Dan Luo, Jing Yang, Zhuo Wang, Fan Fan, Zhiyong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05978)  

**Abstract**: Singing Voice Conversion (SVC) transfers a source singer's timbre to a target while keeping melody and lyrics. The key challenge in any-to-any SVC is adapting unseen speaker timbres to source audio without quality degradation. Existing methods either face timbre leakage or fail to achieve satisfactory timbre similarity and quality in the generated audio. To address these challenges, we propose DAFMSVC, where the self-supervised learning (SSL) features from the source audio are replaced with the most similar SSL features from the target audio to prevent timbre leakage. It also incorporates a dual cross-attention mechanism for the adaptive fusion of speaker embeddings, melody, and linguistic content. Additionally, we introduce a flow matching module for high quality audio generation from the fused features. Experimental results show that DAFMSVC significantly enhances timbre similarity and naturalness, outperforming state-of-the-art methods in both subjective and objective evaluations. 

**Abstract (ZH)**: 歌唱声音转换（Singing Voice Conversion, SVC）将源歌手的音色转移到目标歌手的同时保持旋律和歌词。任何到任何的SVC的关键挑战是在不降低音质的情况下适应未见过的目标歌手音色。现有方法要么面临音色泄露的问题，要么无法在生成的音频中实现令人满意的音色相似度和音质。为了应对这些挑战，我们提出了DAFMSVC，其中源音频的自监督学习（SSL）特征被目标音频中最相似的SSL特征替换，以防止音色泄露。该方法还采用了双交叉注意力机制，用于适应结合说话人嵌入、旋律和语言内容。此外，我们引入了流匹配模块，用于从融合特征生成高质量的音频。实验结果表明，DAFMSVC显著提高了音色相似度和自然度，在主观和客观评价中均优于现有最好的方法。 

---
# Impact-driven Context Filtering For Cross-file Code Completion 

**Title (ZH)**: 基于影响的上下文过滤代码跨文件完成 

**Authors**: Yanzhou Li, Shangqing Liu, Kangjie Chen, Tianwei Zhang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05970)  

**Abstract**: Retrieval-augmented generation (RAG) has recently demonstrated considerable potential for repository-level code completion, as it integrates cross-file knowledge with in-file preceding code to provide comprehensive contexts for generation. To better understand the contribution of the retrieved cross-file contexts, we introduce a likelihood-based metric to evaluate the impact of each retrieved code chunk on the completion. Our analysis reveals that, despite retrieving numerous chunks, only a small subset positively contributes to the completion, while some chunks even degrade performance. To address this issue, we leverage this metric to construct a repository-level dataset where each retrieved chunk is labeled as positive, neutral, or negative based on its relevance to the target completion. We then propose an adaptive retrieval context filtering framework, CODEFILTER, trained on this dataset to mitigate the harmful effects of negative retrieved contexts in code completion. Extensive evaluation on the RepoEval and CrossCodeLongEval benchmarks demonstrates that CODEFILTER consistently improves completion accuracy compared to approaches without filtering operations across various tasks. Additionally, CODEFILTER significantly reduces the length of the input prompt, enhancing computational efficiency while exhibiting strong generalizability across different models. These results underscore the potential of CODEFILTER to enhance the accuracy, efficiency, and attributability of repository-level code completion. 

**Abstract (ZH)**: 检索增强生成（RAG）最近在仓库级别代码补全方面展现了显著潜力，因为它将跨文件知识与文件内先前代码相结合，提供生成所需的整体上下文。为了更好地理解检索到的跨文件上下文的贡献，我们引入了一个基于概率的度量来评估每个检索到的代码片段对补全的影响。我们的分析显示，尽管检索了大量的片段，但只有一小部分积极地促进了补全，有些片段甚至降低了性能。为了解决这个问题，我们利用此度量构建了一个仓库级别的数据集，在该数据集中，每个检索到的片段根据其与目标补全的相关性被标记为正面、中性或负面。然后，我们提出了一种基于此数据集训练的自适应检索上下文过滤框架CODEFILTER，用于减轻负面检索上下文对代码补全的有害影响。在RepoEval和CrossCodeLongEval基准上的广泛评估表明，与其他没有过滤操作的方法相比，CODEFILTER在各类任务中一致地提高了补全准确性。此外，CODEFILTER显著减少了输入提示的长度，提高了计算效率，并且在不同的模型之间表现出强大的普适性。这些结果表明，CODEFILTER有潜力提高仓库级别代码补全的准确度、效率和可归因性。 

---
# Multi-Armed Bandits-Based Optimization of Decision Trees 

**Title (ZH)**: 基于多臂老虎机的决策树优化 

**Authors**: Hasibul Karim Shanto, Umme Ayman Koana, Shadikur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2508.05957)  

**Abstract**: Decision trees, without appropriate constraints, can easily become overly complex and prone to overfit, capturing noise rather than generalizable patterns. To resolve this problem,pruning operation is a crucial part in optimizing decision trees, as it not only reduces the complexity of trees but also decreases the probability of generating overfit models. The conventional pruning techniques like Cost-Complexity Pruning (CCP) and Reduced Error Pruning (REP) are mostly based on greedy approaches that focus on immediate gains in performance while pruning nodes of the decision tree. However, this might result in a lower generalization in the long run, compromising the robust ability of the tree model when introduced to unseen data samples, particularly when trained with small and complex datasets. To address this challenge, we are proposing a Multi-Armed Bandits (MAB)-based pruning approach, a reinforcement learning (RL)-based technique, that will dynamically prune the tree to generate an optimal decision tree with better generalization. Our proposed approach assumes the pruning process as an exploration-exploitation problem, where we are utilizing the MAB algorithms to find optimal branch nodes to prune based on feedback from each pruning actions. Experimental evaluation on several benchmark datasets, demonstrated that our proposed approach results in better predictive performance compared to the traditional ones. This suggests the potential of utilizing MAB for a dynamic and probabilistic way of decision tree pruning, in turn optimizing the decision tree-based model. 

**Abstract (ZH)**: 基于多臂 bandit 的决策树剪枝方法：一种增强泛化能力的强化学习approach 

---
# Prosocial Behavior Detection in Player Game Chat: From Aligning Human-AI Definitions to Efficient Annotation at Scale 

**Title (ZH)**: 玩家游戏聊天中的利他行为检测：从人类-AI定义一致到大规模高效标注 

**Authors**: Rafal Kocielnik, Min Kim, Penphob, Boonyarungsrit, Fereshteh Soltani, Deshawn Sambrano, Animashree Anandkumar, R. Michael Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2508.05938)  

**Abstract**: Detecting prosociality in text--communication intended to affirm, support, or improve others' behavior--is a novel and increasingly important challenge for trust and safety systems. Unlike toxic content detection, prosociality lacks well-established definitions and labeled data, requiring new approaches to both annotation and deployment. We present a practical, three-stage pipeline that enables scalable, high-precision prosocial content classification while minimizing human labeling effort and inference costs. First, we identify the best LLM-based labeling strategy using a small seed set of human-labeled examples. We then introduce a human-AI refinement loop, where annotators review high-disagreement cases between GPT-4 and humans to iteratively clarify and expand the task definition-a critical step for emerging annotation tasks like prosociality. This process results in improved label quality and definition alignment. Finally, we synthesize 10k high-quality labels using GPT-4 and train a two-stage inference system: a lightweight classifier handles high-confidence predictions, while only $\sim$35\% of ambiguous instances are escalated to GPT-4o. This architecture reduces inference costs by $\sim$70% while achieving high precision ($\sim$0.90). Our pipeline demonstrates how targeted human-AI interaction, careful task formulation, and deployment-aware architecture design can unlock scalable solutions for novel responsible AI tasks. 

**Abstract (ZH)**: 在文本中检测利他行为——旨在肯定、支持或改进他人行为的沟通——是信任与安全系统面临的新型且日益重要的挑战。不同于有毒内容检测，利他行为缺乏成熟的定义和标记数据，需要新的标注和部署方法。我们提出一个实用的三阶段流水线，能够在减少人工标注努力和推理成本的同时，实现可扩展且高精度的利他内容分类。首先，我们使用少量的人工标注示例确定最佳基于大语言模型的标注策略。随后引入人类-AI修正循环，标注员审核GPT-4与人类之间高分歧的案例，以迭代澄清和扩展任务定义——这是新兴标注任务如利他行为的关键步骤。此过程提高了标签质量和定义一致性。最后，我们使用GPT-4合成10,000个高质量标签，并训练一个两阶段推理系统：轻量级分类器处理高置信度预测，而仅有约35%的模棱两可实例被升级到GPT-4o。此架构将推理成本降低了约70%，同时保持高精度（约0.90）。我们的流水线展示了针对新型负责任AI任务的目标化人类-AI互动、精细的任务表述以及部署意识架构设计如何解锁可扩展解决方案。 

---
# REFS: Robust EEG feature selection with missing multi-dimensional annotation for emotion recognition 

**Title (ZH)**: REFS: 面向情绪识别的鲁棒EEG特征选择，处理缺失多维标注 

**Authors**: Xueyuan Xu, Wenjia Dong, Fulin Wei, Li Zhuo  

**Link**: [PDF](https://arxiv.org/pdf/2508.05933)  

**Abstract**: The affective brain-computer interface is a crucial technology for affective interaction and emotional intelligence, emerging as a significant area of research in the human-computer interaction. Compared to single-type features, multi-type EEG features provide a multi-level representation for analyzing multi-dimensional emotions. However, the high dimensionality of multi-type EEG features, combined with the relatively small number of high-quality EEG samples, poses challenges such as classifier overfitting and suboptimal real-time performance in multi-dimensional emotion recognition. Moreover, practical applications of affective brain-computer interface frequently encounters partial absence of multi-dimensional emotional labels due to the open nature of the acquisition environment, and ambiguity and variability in individual emotion perception. To address these challenges, this study proposes a novel EEG feature selection method for missing multi-dimensional emotion recognition. The method leverages adaptive orthogonal non-negative matrix factorization to reconstruct the multi-dimensional emotional label space through second-order and higher-order correlations, which could reduce the negative impact of missing values and outliers on label reconstruction. Simultaneously, it employs least squares regression with graph-based manifold learning regularization and global feature redundancy minimization regularization to enable EEG feature subset selection despite missing information, ultimately achieving robust EEG-based multi-dimensional emotion recognition. Simulation experiments on three widely used multi-dimensional emotional datasets, DREAMER, DEAP and HDED, reveal that the proposed method outperforms thirteen advanced feature selection methods in terms of robustness for EEG emotional feature selection. 

**Abstract (ZH)**: 具有情感识别功能的大脑-计算机接口是情感交互和情感智能的关键技术，是人机交互领域的一个重要研究方向。与单一类型特征相比，多类型EEG特征提供了多维度情感分析的多层次表示。然而，多类型EEG特征的高维性，结合高质量EEG样本数量相对较少，导致分类器过拟合和实时性能不佳等问题。此外，由于采集环境的开放性，具有情感识别功能的大脑-计算机接口在实际应用中经常遇到多维度情感标签部分缺失的情况，且个体情感感知具有模糊性和变异性。为解决这些问题，本研究提出了一种新的EEG特征选择方法，以实现多维度情感识别。该方法利用自适应正交非负矩阵分解技术，通过二阶和高阶相关性重构多维度情感标签空间，减少缺失值和异常值对标签重构的负面影响。同时，该方法采用基于图的流形学习正则化和全局特征冗余最小化正则化与最小二乘回归结合，即使在信息缺失的情况下也能实现EEG特征子集选择，最终实现稳健的情感识别。在对广泛使用的三维多维度情感数据集DREAMER、DEAP和HDED进行的仿真实验中，提出的方法在EEG情感特征选择的鲁棒性方面优于十三种先进的特征选择方法。 

---
# Enhancing Software Vulnerability Detection Through Adaptive Test Input Generation Using Genetic Algorithm 

**Title (ZH)**: 基于遗传算法的自适应测试输入生成以增强软件漏洞检测 

**Authors**: Yanusha Mehendran, Maolin Tang, Yi Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05923)  

**Abstract**: Software vulnerabilities continue to undermine the reliability and security of modern systems, particularly as software complexity outpaces the capabilities of traditional detection methods. This study introduces a genetic algorithm-based method for test input generation that innovatively integrates genetic operators and adaptive learning to enhance software vulnerability detection. A key contribution is the application of the crossover operator, which facilitates exploration by searching across a broader space of potential test inputs. Complementing this, an adaptive feedback mechanism continuously learns from the system's execution behavior and dynamically guides input generation toward promising areas of the input space. Rather than relying on fixed or randomly selected inputs, the approach evolves a population of structurally valid test cases using feedback-driven selection, enabling deeper and more effective code traversal. This strategic integration of exploration and exploitation ensures that both diverse and targeted test inputs are developed over time. Evaluation was conducted across nine open-source JSON-processing libraries. The proposed method achieved substantial improvements in coverage compared to a benchmark evolutionary fuzzing method, with average gains of 39.8% in class coverage, 62.4% in method coverage, 105.0% in line coverage, 114.0% in instruction coverage, and 166.0% in branch coverage. These results highlight the method's capacity to detect deeper and more complex vulnerabilities, offering a scalable and adaptive solution to software security testing. 

**Abstract (ZH)**: 基于遗传算法的测试输入生成方法在软件漏洞检测中的应用：探索与利用的协同增强 

---
# Do Ethical AI Principles Matter to Users? A Large-Scale Analysis of User Sentiment and Satisfaction 

**Title (ZH)**: 伦理AI原则对用户有意义吗？大规模分析用户情感与满意度 

**Authors**: Stefan Pasch, Min Chul Cha  

**Link**: [PDF](https://arxiv.org/pdf/2508.05913)  

**Abstract**: As AI systems become increasingly embedded in organizational workflows and consumer applications, ethical principles such as fairness, transparency, and robustness have been widely endorsed in policy and industry guidelines. However, there is still scarce empirical evidence on whether these principles are recognized, valued, or impactful from the perspective of users. This study investigates the link between ethical AI and user satisfaction by analyzing over 100,000 user reviews of AI products from G2. Using transformer-based language models, we measure sentiment across seven ethical dimensions defined by the EU Ethics Guidelines for Trustworthy AI. Our findings show that all seven dimensions are positively associated with user satisfaction. Yet, this relationship varies systematically across user and product types. Technical users and reviewers of AI development platforms more frequently discuss system-level concerns (e.g., transparency, data governance), while non-technical users and reviewers of end-user applications emphasize human-centric dimensions (e.g., human agency, societal well-being). Moreover, the association between ethical AI and user satisfaction is significantly stronger for non-technical users and end-user applications across all dimensions. Our results highlight the importance of ethical AI design from users' perspectives and underscore the need to account for contextual differences across user roles and product types. 

**Abstract (ZH)**: 随着人工智能系统越来越多地嵌入组织工作流和消费者应用中，公平性、透明性和鲁棒性等伦理原则已在政策和行业指南中得到了广泛认可。然而，关于这些原则是否被用户认可、重视或具有影响力，仍缺乏实证证据。本研究通过分析来自G2的逾100,000条AI产品用户评论，探讨伦理AI与用户满意度之间的联系。利用基于转换器的语言模型，我们衡量了用户评论中七个由欧盟可信赖AI伦理指南定义的伦理维度的 sentiment。研究发现，这七个维度都与用户满意度正相关。然而，这种关系在不同用户和产品类型之间系统地有所不同。技术用户和技术平台的审查者更频繁地讨论系统层面的关切（如透明性、数据治理），而非技术用户和终端用户应用的审查者则更强调以人类为中心的维度（如人类自主权、社会福祉）。此外，所有维度上，伦理AI与用户满意度之间的关联对非技术用户和终端用户应用来说更为显著。研究结果强调了从用户视角设计伦理AI的重要性，并突显了在用户角色和产品类型不同的背景下考虑情境差异的必要性。 

---
# From Imperfect Signals to Trustworthy Structure: Confidence-Aware Inference from Heterogeneous and Reliability-Varying Utility Data 

**Title (ZH)**: 从不完美信号到可信赖结构：考虑置信度的异构且可靠性变化的效用数据推理 

**Authors**: Haoran Li, Lihao Mai, Muhao Guo, Jiaqi Wu, Yang Weng, Yannan Sun, Ce Jimmy Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05791)  

**Abstract**: Accurate distribution grid topology is essential for reliable modern grid operations. However, real-world utility data originates from multiple sources with varying characteristics and levels of quality. In this work, developed in collaboration with Oncor Electric Delivery, we propose a scalable framework that reconstructs a trustworthy grid topology by systematically integrating heterogeneous data. We observe that distribution topology is fundamentally governed by two complementary dimensions: the spatial layout of physical infrastructure (e.g., GIS and asset metadata) and the dynamic behavior of the system in the signal domain (e.g., voltage time series). When jointly leveraged, these dimensions support a complete and physically coherent reconstruction of network connectivity. To address the challenge of uneven data quality without compromising observability, we introduce a confidence-aware inference mechanism that preserves structurally informative yet imperfect inputs, while quantifying the reliability of each inferred connection for operator interpretation. This soft handling of uncertainty is tightly coupled with hard enforcement of physical feasibility: we embed operational constraints, such as transformer capacity limits and radial topology requirements, directly into the learning process. Together, these components ensure that inference is both uncertainty-aware and structurally valid, enabling rapid convergence to actionable, trustworthy topologies under real-world deployment conditions. The proposed framework is validated using data from over 8000 meters across 3 feeders in Oncor's service territory, demonstrating over 95% accuracy in topology reconstruction and substantial improvements in confidence calibration and computational efficiency relative to baseline methods. 

**Abstract (ZH)**: 准确的配电网络拓扑对于可靠现代电网运营至关重要。然而，实际的公用事业数据源自多个具有不同特性和质量水平的数据源。在与Oncor Electric Delivery合作下，我们提出了一种可扩展的框架，通过系统地整合异质数据来重构一个可信的网络拓扑。我们观察到，配电网络拓扑本质上受两个互补维度的治理：物理基础设施的空间布局（例如，GIS和资产元数据）和系统的动态行为在信号域（例如，电压时间序列）。当这些维度共同利用时，支持网络连接的完整且物理上一致的重构。为了解决数据质量不均的问题同时保持可观测性，我们引入了一种基于置信度的推理机制，保留结构性信息但不完美的输入，同时量化每个推断连接的可靠性供操作员解释。这种对不确定性的软处理与严格的物理可行性硬约束紧密结合：我们将如变压器容量限制和放射状拓扑要求等运行约束直接嵌入学习过程中。这些组件共同确保推理既是不确定性的感知又是结构上的有效，使得在实际部署条件下能够快速收敛到行动性好且值得信赖的拓扑。所提出的框架利用Oncor服务区域内三个馈电超过8000米的数据进行验证，展现出了超过95%的拓扑重构准确率，并且在置信度校准和计算效率方面显著优于基线方法。 

---
# UnGuide: Learning to Forget with LoRA-Guided Diffusion Models 

**Title (ZH)**: LoRA-Guided Diffusion Models: Learning to Forget 

**Authors**: Agnieszka Polowczyk, Alicja Polowczyk, Dawid Malarz, Artur Kasymov, Marcin Mazur, Jacek Tabor, Przemysław Spurek  

**Link**: [PDF](https://arxiv.org/pdf/2508.05755)  

**Abstract**: Recent advances in large-scale text-to-image diffusion models have heightened concerns about their potential misuse, especially in generating harmful or misleading content. This underscores the urgent need for effective machine unlearning, i.e., removing specific knowledge or concepts from pretrained models without compromising overall performance. One possible approach is Low-Rank Adaptation (LoRA), which offers an efficient means to fine-tune models for targeted unlearning. However, LoRA often inadvertently alters unrelated content, leading to diminished image fidelity and realism. To address this limitation, we introduce UnGuide -- a novel approach which incorporates UnGuidance, a dynamic inference mechanism that leverages Classifier-Free Guidance (CFG) to exert precise control over the unlearning process. UnGuide modulates the guidance scale based on the stability of a few first steps of denoising processes, enabling selective unlearning by LoRA adapter. For prompts containing the erased concept, the LoRA module predominates and is counterbalanced by the base model; for unrelated prompts, the base model governs generation, preserving content fidelity. Empirical results demonstrate that UnGuide achieves controlled concept removal and retains the expressive power of diffusion models, outperforming existing LoRA-based methods in both object erasure and explicit content removal tasks. 

**Abstract (ZH)**: 大规模文本到图像扩散模型的 Recent Advances 加剧了对其潜在滥用的担忧，尤其是生成有害或误导性内容。这突显了急需有效的机器遗忘技术，即在不损害整体性能的情况下从预训练模型中移除特定的知识或概念。一种可能的方法是低秩适应（LoRA），它提供了一种有效的方法来微调模型以实现有针对性的遗忘。然而，LoRA 通常会无意中改变不相关的内容，导致图像保真度和现实感下降。为了解决这一限制，我们引入了 UnGuide —— 一种新颖的方法，该方法结合了 UnGuidance 动态推理机制，利用分类器无关指导（CFG）对遗忘过程进行精确控制。UnGuide 根据去噪过程的前几步的稳定性调节指导尺度，使 LoRA 适配器能够选择性地进行遗忘。对于包含被擦除概念的提示，LoRA 模块占主导地位，并由基础模型进行制衡，保留内容保真度；对于不相关的提示，基础模型主导生成过程。实验结果表明，UnGuide 实现了可控的概念去除，并保留了扩散模型的表达能力，在物体擦除和明确内容去除任务中均优于现有的基于 LoRA 的方法。 

---
# A Physiologically-Constrained Neural Network Digital Twin Framework for Replicating Glucose Dynamics in Type 1 Diabetes 

**Title (ZH)**: 基于生理约束的神经网络数字孪生框架：复制1型糖尿病血糖动态 

**Authors**: Valentina Roquemen-Echeverri, Taisa Kushner, Peter G. Jacobs, Clara Mosquera-Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2508.05705)  

**Abstract**: Simulating glucose dynamics in individuals with type 1 diabetes (T1D) is critical for developing personalized treatments and supporting data-driven clinical decisions. Existing models often miss key physiological aspects and are difficult to individualize. Here, we introduce physiologically-constrained neural network (NN) digital twins to simulate glucose dynamics in T1D. To ensure interpretability and physiological consistency, we first build a population-level NN state-space model aligned with a set of ordinary differential equations (ODEs) describing glucose regulation. This model is formally verified to conform to known T1D dynamics. Digital twins are then created by augmenting the population model with individual-specific models, which include personal data, such as glucose management and contextual information, capturing both inter- and intra-individual variability. We validate our approach using real-world data from the T1D Exercise Initiative study. Two weeks of data per participant were split into 5-hour sequences and simulated glucose profiles were compared to observed ones. Clinically relevant outcomes were used to assess similarity via paired equivalence t-tests with predefined clinical equivalence margins. Across 394 digital twins, glucose outcomes were equivalent between simulated and observed data: time in range (70-180 mg/dL) was 75.1$\pm$21.2% (simulated) vs. 74.4$\pm$15.4% (real; P<0.001); time below range (<70 mg/dL) 2.5$\pm$5.2% vs. 3.0$\pm$3.3% (P=0.022); and time above range (>180 mg/dL) 22.4$\pm$22.0% vs. 22.6$\pm$15.9% (P<0.001). Our framework can incorporate unmodeled factors like sleep and activity while preserving key dynamics. This approach enables personalized in silico testing of treatments, supports insulin optimization, and integrates physics-based and data-driven modeling. Code: this https URL 

**Abstract (ZH)**: 模拟1型糖尿病（T1D）个体的葡萄糖动态对于开发个性化治疗方法和支持数据驱动的临床决策至关重要。现有的模型往往忽略了关键的生理方面，且难以个性化。我们介绍了基于生理约束的神经网络（NN）数字孪生，以模拟T1D的葡萄糖动态。为了确保可解释性和生理一致性，我们首先建立了一个与描述葡萄糖调节的常微分方程（ODEs）集合相一致的人口级NN状态空间模型。该模型正式验证符合已知的T1D动态。接着，通过将个体特异性模型添加到人口模型中，创建数字孪生，这些模型包括个人数据，如葡萄糖管理和个人背景信息，捕获了个体间和个体内的变异。我们使用T1D Exercise Initiative研究的实际数据验证了该方法。每位参与者两周的数据被分为5小时序列，模拟的葡萄糖曲线与观察到的进行了比较。通过预定义的临床等效边际使用成对等效t检验评估了临床相关结果。在整个394个数字孪生中，模拟和观察数据的葡萄糖结果等效：葡萄糖目标范围内（70-180 mg/dL）的时间为75.1±21.2%（模拟） vs. 74.4±15.4%（真实；P<0.001）；低于目标范围（<70 mg/dL）的时间为2.5±5.2% vs. 3.0±3.3%（P=0.022）；高于目标范围（>180 mg/dL）的时间为22.4±22.0% vs. 22.6±15.9%（P<0.001）。我们的框架可以整合未建模的因素，例如睡眠和活动，同时保持关键动态。该方法允许在体外个性化测试治疗方法、支持胰岛素优化，并整合基于物理和数据驱动的建模方法。代码：https://github.com/AlibabaCloudcommend/Qwen-T1D-Glucose-Model。 

---
# Multi-Faceted Large Embedding Tables for Pinterest Ads Ranking 

**Title (ZH)**: Pinterest 广告排名的多面大型嵌入表 

**Authors**: Runze Su, Jiayin Jin, Jiacheng Li, Sihan Wang, Guangtong Bai, Zelun Wang, Li Tang, Yixiong Meng, Huasen Wu, Zhimeng Pan, Kungang Li, Han Sun, Zhifang Liu, Haoyang Li, Siping Ji, Ling Leng, Prathibha Deshikachar  

**Link**: [PDF](https://arxiv.org/pdf/2508.05700)  

**Abstract**: Large embedding tables are indispensable in modern recommendation systems, thanks to their ability to effectively capture and memorize intricate details of interactions among diverse entities. As we explore integrating large embedding tables into Pinterest's ads ranking models, we encountered not only common challenges such as sparsity and scalability, but also several obstacles unique to our context. Notably, our initial attempts to train large embedding tables from scratch resulted in neutral metrics. To tackle this, we introduced a novel multi-faceted pretraining scheme that incorporates multiple pretraining algorithms. This approach greatly enriched the embedding tables and resulted in significant performance improvements. As a result, the multi-faceted large embedding tables bring great performance gain on both the Click-Through Rate (CTR) and Conversion Rate (CVR) domains. Moreover, we designed a CPU-GPU hybrid serving infrastructure to overcome GPU memory limits and elevate the scalability. This framework has been deployed in the Pinterest Ads system and achieved 1.34% online CPC reduction and 2.60% CTR increase with neutral end-to-end latency change. 

**Abstract (ZH)**: 大型嵌入表在现代推荐系统中不可或缺，得益于其有效捕捉和记忆多样化实体间复杂交互细节的能力。在将大型嵌入表集成到Pinterest的广告排名模型中时，我们不仅面临稀疏性和扩展性等常见挑战，还遇到一些特有的障碍。最初尝试从零训练大型嵌入表导致了中性的评估指标。为解决这一问题，我们引入了一种结合多种预训练算法的新型多方面预训练方案，极大地丰富了嵌入表并取得了显著的性能提升。多方面大型嵌入表在点击率(CTR)和转化率(CVR)方面带来了显著的性能提升。此外，我们设计了一种CPU-GPU混合服务架构以克服GPU内存限制并提升扩展性。该框架已在Pinterest Ads系统中部署，实现了1.34%的在线CPM减少和2.60%的CTR提升，端到端延迟保持不变。 

---
# Log2Sig: Frequency-Aware Insider Threat Detection via Multivariate Behavioral Signal Decomposition 

**Title (ZH)**: Log2Sig: 基于多变量行为信号分解的频率aware内部威胁检测 

**Authors**: Kaichuan Kong, Dongjie Liu, Xiaobo Jin, Zhiying Li, Guanggang Geng  

**Link**: [PDF](https://arxiv.org/pdf/2508.05696)  

**Abstract**: Insider threat detection presents a significant challenge due to the deceptive nature of malicious behaviors, which often resemble legitimate user operations. However, existing approaches typically model system logs as flat event sequences, thereby failing to capture the inherent frequency dynamics and multiscale disturbance patterns embedded in user behavior. To address these limitations, we propose Log2Sig, a robust anomaly detection framework that transforms user logs into multivariate behavioral frequency signals, introducing a novel representation of user behavior. Log2Sig employs Multivariate Variational Mode Decomposition (MVMD) to extract Intrinsic Mode Functions (IMFs), which reveal behavioral fluctuations across multiple temporal scales. Based on this, the model further performs joint modeling of behavioral sequences and frequency-decomposed signals: the daily behavior sequences are encoded using a Mamba-based temporal encoder to capture long-term dependencies, while the corresponding frequency components are linearly projected to match the encoder's output dimension. These dual-view representations are then fused to construct a comprehensive user behavior profile, which is fed into a multilayer perceptron for precise anomaly detection. Experimental results on the CERT r4.2 and r5.2 datasets demonstrate that Log2Sig significantly outperforms state-of-the-art baselines in both accuracy and F1 score. 

**Abstract (ZH)**: 基于Log2Sig的日志异常检测框架：多变量行为频率信号表示与模式分解 

---
# Empirical Evaluation of AI-Assisted Software Package Selection: A Knowledge Graph Approach 

**Title (ZH)**: 基于知识图谱的AI辅助软件包选择的实证评价 

**Authors**: Siamak Farshidi, Amir Saberhabibi, Behbod Eskafi, Niloofar Nikfarjam, Sadegh Eskandari, Slinger Jansen, Michel Chaudron, Bedir Tekinerdogan  

**Link**: [PDF](https://arxiv.org/pdf/2508.05693)  

**Abstract**: Selecting third-party software packages in open-source ecosystems like Python is challenging due to the large number of alternatives and limited transparent evidence for comparison. Generative AI tools are increasingly used in development workflows, but their suggestions often overlook dependency evaluation, emphasize popularity over suitability, and lack reproducibility. This creates risks for projects that require transparency, long-term reliability, maintainability, and informed architectural decisions. This study formulates software package selection as a Multi-Criteria Decision-Making (MCDM) problem and proposes a data-driven framework for technology evaluation. Automated data pipelines continuously collect and integrate software metadata, usage trends, vulnerability information, and developer sentiment from GitHub, PyPI, and Stack Overflow. These data are structured into a decision model representing relationships among packages, domain features, and quality attributes. The framework is implemented in PySelect, a decision support system that uses large language models to interpret user intent and query the model to identify contextually appropriate packages. The approach is evaluated using 798,669 Python scripts from 16,887 GitHub repositories and a user study based on the Technology Acceptance Model. Results show high data extraction precision, improved recommendation quality over generative AI baselines, and positive user evaluations of usefulness and ease of use. This work introduces a scalable, interpretable, and reproducible framework that supports evidence-based software selection using MCDM principles, empirical data, and AI-assisted intent modeling. 

**Abstract (ZH)**: 在开源生态系统如Python中选择第三方软件包具有挑战性，由于可供选择的方案众多且缺乏透明的比较证据。生成式AI工具在开发流程中使用越来越多，但其建议往往忽视依赖性评估，侧重流行度而非适用性，并缺乏可重复性。这增加了需要透明度、长期可靠性、可维护性和知情架构决策的项目的风险。本研究将软件包选择问题形式化为多准则决策制定（MCDM）问题，并提出一种基于数据的技术评估框架。自动化的数据管道不断收集和整合来自GitHub、PyPI和Stack Overflow的软件元数据、使用趋势、漏洞信息及开发者情感。这些数据被结构化为一个决策模型，代表软件包、领域特征与质量属性之间关系。该框架在PySelect中实现，这是一个决策支持系统，利用大型语言模型理解用户意图，并查询模型以识别上下文合适的技术。该方法通过来自16,887个GitHub仓库的798,669个Python脚本和基于技术接受模型的用户研究进行了评估。结果表明，数据提取精度高，推荐质量优于生成式AI基准模型，并且用户对实用性和易用性给予了积极评价。本研究引入了一种可扩展、可解释和可重复的框架，利用MCDM原则、实证数据和AI辅助意图建模支持基于证据的软件选择。 

---
# Selection-Based Vulnerabilities: Clean-Label Backdoor Attacks in Active Learning 

**Title (ZH)**: 基于选择的漏洞：主动学习中的清洁标签后门攻击 

**Authors**: Yuhan Zhi, Longtian Wang, Xiaofei Xie, Chao Shen, Qiang Hu, Xiaohong Guan  

**Link**: [PDF](https://arxiv.org/pdf/2508.05681)  

**Abstract**: Active learning(AL), which serves as the representative label-efficient learning paradigm, has been widely applied in resource-constrained scenarios. The achievement of AL is attributed to acquisition functions, which are designed for identifying the most important data to label. Despite this success, one question remains unanswered: is AL safe? In this work, we introduce ALA, a practical and the first framework to utilize the acquisition function as the poisoning attack surface to reveal the weakness of active learning. Specifically, ALA optimizes imperceptibly poisoned inputs to exhibit high uncertainty scores, increasing their probability of being selected by acquisition functions. To evaluate ALA, we conduct extensive experiments across three datasets, three acquisition functions, and two types of clean-label backdoor triggers. Results show that our attack can achieve high success rates (up to 94%) even under low poisoning budgets (0.5%-1.0%) while preserving model utility and remaining undetectable to human annotators. Our findings remind active learning users: acquisition functions can be easily exploited, and active learning should be deployed with caution in trusted data scenarios. 

**Abstract (ZH)**: AL安全吗？一种利用获取函数进行中毒攻击的实用框架（ALA） 

---
# Are All Genders Equal in the Eyes of Algorithms? -- Analysing Search and Retrieval Algorithms for Algorithmic Gender Fairness 

**Title (ZH)**: 算法视角下所有性别平等吗？——搜索和检索算法的算法性别公平性分析 

**Authors**: Stefanie Urchs, Veronika Thurner, Matthias Aßenmacher, Ludwig Bothmann, Christian Heumann, Stephanie Thiemichen  

**Link**: [PDF](https://arxiv.org/pdf/2508.05680)  

**Abstract**: Algorithmic systems such as search engines and information retrieval platforms significantly influence academic visibility and the dissemination of knowledge. Despite assumptions of neutrality, these systems can reproduce or reinforce societal biases, including those related to gender. This paper introduces and applies a bias-preserving definition of algorithmic gender fairness, which assesses whether algorithmic outputs reflect real-world gender distributions without introducing or amplifying disparities. Using a heterogeneous dataset of academic profiles from German universities and universities of applied sciences, we analyse gender differences in metadata completeness, publication retrieval in academic databases, and visibility in Google search results. While we observe no overt algorithmic discrimination, our findings reveal subtle but consistent imbalances: male professors are associated with a greater number of search results and more aligned publication records, while female professors display higher variability in digital visibility. These patterns reflect the interplay between platform algorithms, institutional curation, and individual self-presentation. Our study highlights the need for fairness evaluations that account for both technical performance and representational equality in digital systems. 

**Abstract (ZH)**: 算法系统如搜索引擎和信息检索平台在很大程度上影响学术可见性和知识的传播。尽管存在中立性的假设，这些系统仍然可能会重现或加强社会偏见，包括性别偏见。本文提出并应用了一种保留偏见的算法性别公平定义，评估算法输出是否真实反映了现实世界的性别分布而没有引入或放大差异。通过德国高校和应用科学大学的异质性学术档案数据集，我们分析了性别在元数据完整度、学术数据库中的论文检索以及在Google搜索结果中的可见性方面的差异。尽管我们没有观察到明确的算法歧视，但我们的研究发现存在细微且一致的不平衡：男性教授与更多的搜索结果和更一致的出版记录相关联，而女性教授的数字可见性则显示出更高的变异性。这些模式反映了平台算法、机构策展和个人自我呈现之间的相互作用。本研究强调了在数字系统中同时考虑技术绩效和表现平等的公平性评估的必要性。 

---
# Adversarial Attacks on Reinforcement Learning-based Medical Questionnaire Systems: Input-level Perturbation Strategies and Medical Constraint Validation 

**Title (ZH)**: 基于强化学习的医疗问卷系统对抗攻击：输入级扰动策略与医学约束验证 

**Authors**: Peizhuo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05677)  

**Abstract**: RL-based medical questionnaire systems have shown great potential in medical scenarios. However, their safety and robustness remain unresolved. This study performs a comprehensive evaluation on adversarial attack methods to identify and analyze their potential vulnerabilities. We formulate the diagnosis process as a Markov Decision Process (MDP), where the state is the patient responses and unasked questions, and the action is either to ask a question or to make a diagnosis. We implemented six prevailing major attack methods, including the Fast Gradient Signed Method (FGSM), Projected Gradient Descent (PGD), Carlini & Wagner Attack (C&W) attack, Basic Iterative Method (BIM), DeepFool, and AutoAttack, with seven epsilon values each. To ensure the generated adversarial examples remain clinically plausible, we developed a comprehensive medical validation framework consisting of 247 medical constraints, including physiological bounds, symptom correlations, and conditional medical constraints. We achieved a 97.6% success rate in generating clinically plausible adversarial samples. We performed our experiment on the National Health Interview Survey (NHIS) dataset (this https URL), which consists of 182,630 samples, to predict the participant's 4-year mortality rate. We evaluated our attacks on the AdaptiveFS framework proposed in arXiv:2004.00994. Our results show that adversarial attacks could significantly impact the diagnostic accuracy, with attack success rates ranging from 33.08% (FGSM) to 64.70% (AutoAttack). Our work has demonstrated that even under strict medical constraints on the input, such RL-based medical questionnaire systems still show significant vulnerabilities. 

**Abstract (ZH)**: 基于RL的医疗问卷系统在医疗场景中展示了巨大的潜力，但其安全性和鲁棒性问题尚未解决。该研究对对抗攻击方法进行了全面评估，以识别和分析其潜在漏洞。我们将诊断过程形式化为马尔可夫决策过程（MDP），其中状态为患者的反应和未问的问题，动作则是提问或诊断。我们实现了六种主要的攻击方法，包括快速梯度符号方法（FGSM）、投影梯度下降（PGD）、Carlini & Wagner攻击（C&W攻击）、基本迭代方法（BIM）、DeepFool和AutoAttack，每种方法有七个不同的ε值。为了确保生成的对抗样本在临床上具有合理性，我们开发了一个包括247个医学约束的全面医学验证框架，这些约束涵盖了生理边界、症状相关性和条件医学约束。我们成功地生成了97.6%在临床上合理的对抗样本。我们在National Health Interview Survey（NHIS）数据集中（具体内容请参见此链接）进行了实验，该数据集包含182,630个样本，用于预测参与者的4年死亡率。我们将攻击方法应用于arXiv:2004.00994中提出的AdaptiveFS框架。实验结果显示，对抗攻击显著影响了诊断准确性，攻击成功率从33.08%（FGSM）到64.70%（AutoAttack）不等。我们的工作表明，在输入受到严格医学约束的情况下，基于RL的医疗问卷系统仍表现出显著的脆弱性。 

---
# Breaking the Top-$K$ Barrier: Advancing Top-$K$ Ranking Metrics Optimization in Recommender Systems 

**Title (ZH)**: 突破Top-$K$障碍：提高推荐系统中Top-$K$排名指标优化 

**Authors**: Weiqin Yang, Jiawei Chen, Shengjia Zhang, Peng Wu, Yuegang Sun, Yan Feng, Chun Chen, Can Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.05673)  

**Abstract**: In the realm of recommender systems (RS), Top-$K$ ranking metrics such as NDCG@$K$ are the gold standard for evaluating recommendation performance. However, during the training of recommendation models, optimizing NDCG@$K$ poses significant challenges due to its inherent discontinuous nature and the intricate Top-$K$ truncation. Recent efforts to optimize NDCG@$K$ have either overlooked the Top-$K$ truncation or suffered from high computational costs and training instability. To overcome these limitations, we propose SoftmaxLoss@$K$ (SL@$K$), a novel recommendation loss tailored for NDCG@$K$ optimization. Specifically, we integrate the quantile technique to handle Top-$K$ truncation and derive a smooth upper bound for optimizing NDCG@$K$ to address discontinuity. The resulting SL@$K$ loss has several desirable properties, including theoretical guarantees, ease of implementation, computational efficiency, gradient stability, and noise robustness. Extensive experiments on four real-world datasets and three recommendation backbones demonstrate that SL@$K$ outperforms existing losses with a notable average improvement of 6.03%. The code is available at this https URL. 

**Abstract (ZH)**: 在推荐系统领域，Top-$K$ 排名指标如 NDCG@$K$ 是评估推荐性能的金标准。然而，在推荐模型训练过程中，优化 NDCG@$K$ 由于其固有的不连续性和 Top-$K$ 截断的复杂性而面临显著挑战。最近为优化 NDCG@$K$ 的努力要么忽略了 Top-$K$ 截断，要么遭受高计算成本和训练不稳定性的困扰。为克服这些限制，我们提出了 SoftmaxLoss@$K$（SL@$K$），这是一种针对 NDCG@$K$ 优化的新推荐损失函数。具体而言，我们结合分位数技术处理 Top-$K$ 截断，并推导出一个平滑的上界以优化 NDCG@$K$，从而解决不连续性问题。SL@$K$ 损失具有若干 desirable 属性，包括理论保证、易于实现、高效计算、梯度稳定性及抗噪声能力。在四个真实世界数据集和三种推荐模型架构上的广泛实验表明，SL@$K$ 在平均性能上优于现有损失函数，提升幅度达 6.03%。代码可在以下链接获取。 

---
# HySemRAG: A Hybrid Semantic Retrieval-Augmented Generation Framework for Automated Literature Synthesis and Methodological Gap Analysis 

**Title (ZH)**: HySemRAG：一种混合语义检索增强生成框架，用于自动化文献综合与方法论缺口分析 

**Authors**: Alejandro Godinez  

**Link**: [PDF](https://arxiv.org/pdf/2508.05666)  

**Abstract**: We present HySemRAG, a framework that combines Extract, Transform, Load (ETL) pipelines with Retrieval-Augmented Generation (RAG) to automate large-scale literature synthesis and identify methodological research gaps. The system addresses limitations in existing RAG architectures through a multi-layered approach: hybrid retrieval combining semantic search, keyword filtering, and knowledge graph traversal; an agentic self-correction framework with iterative quality assurance; and post-hoc citation verification ensuring complete traceability. Our implementation processes scholarly literature through eight integrated stages: multi-source metadata acquisition, asynchronous PDF retrieval, custom document layout analysis using modified Docling architecture, bibliographic management, LLM-based field extraction, topic modeling, semantic unification, and knowledge graph construction. The system creates dual data products - a Neo4j knowledge graph enabling complex relationship queries and Qdrant vector collections supporting semantic search - serving as foundational infrastructure for verifiable information synthesis. Evaluation across 643 observations from 60 testing sessions demonstrates structured field extraction achieving 35.1% higher semantic similarity scores (0.655 $\pm$ 0.178) compared to PDF chunking approaches (0.485 $\pm$ 0.204, p < 0.000001). The agentic quality assurance mechanism achieves 68.3% single-pass success rates with 99.0% citation accuracy in validated responses. Applied to geospatial epidemiology literature on ozone exposure and cardiovascular disease, the system identifies methodological trends and research gaps, demonstrating broad applicability across scientific domains for accelerating evidence synthesis and discovery. 

**Abstract (ZH)**: HySemRAG：结合抽取、转换、加载（ETL）管道与检索增强生成（RAG）的框架，实现大规模文献综合并识别方法学研究空白 

---
# Enhancing Retrieval-Augmented Generation for Electric Power Industry Customer Support 

**Title (ZH)**: 增强基于检索的生成以改善电力行业客户支持 

**Authors**: Hei Yu Chan, Kuok Tou Ho, Chenglong Ma, Yujing Si, Hok Lai Lin, Sa Lei Lam  

**Link**: [PDF](https://arxiv.org/pdf/2508.05664)  

**Abstract**: Many AI customer service systems use standard NLP pipelines or finetuned language models, which often fall short on ambiguous, multi-intent, or detail-specific queries. This case study evaluates recent techniques: query rewriting, RAG Fusion, keyword augmentation, intent recognition, and context reranking, for building a robust customer support system in the electric power domain. We compare vector-store and graph-based RAG frameworks, ultimately selecting the graph-based RAG for its superior performance in handling complex queries. We find that query rewriting improves retrieval for queries using non-standard terminology or requiring precise detail. RAG Fusion boosts performance on vague or multifaceted queries by merging multiple retrievals. Reranking reduces hallucinations by filtering irrelevant contexts. Intent recognition supports the decomposition of complex questions into more targeted sub-queries, increasing both relevance and efficiency. In contrast, keyword augmentation negatively impacts results due to biased keyword selection. Our final system combines intent recognition, RAG Fusion, and reranking to handle disambiguation and multi-source queries. Evaluated on both a GPT-4-generated dataset and a real-world electricity provider FAQ dataset, it achieves 97.9% and 89.6% accuracy respectively, substantially outperforming baseline RAG models. 

**Abstract (ZH)**: 许多AI客服系统采用标准NLP流水线或微调的语言模型，往往在处理模糊、多意图或详细特定的查询时表现不佳。本案例研究评估了近期技术：查询重写、RAG融合、关键词扩充、意图识别和上下文重排名，以构建电力领域的 robust 客户支持系统。我们比较了基于向量存储和图的RAG框架，最终选择了图的RAG框架，因为它在处理复杂查询时表现出更强的性能。我们发现，查询重写可以提高使用非标准术语或需要精确细节的查询的检索效果。RAG融合通过合并多个检索结果来增强对模糊或多方面查询的性能。重排名通过过滤无关上下文来减少幻觉。意图识别支持将复杂问题分解为更具体的子查询，从而提高相关性和效率。相比之下，关键词扩充由于关键词选择偏向性而负面影响结果。我们的最终系统结合了意图识别、RAG融合和重排名，以处理歧义和多来源查询。该系统在GPT-4生成的数据集和现实世界的电力提供商FAQ数据集上分别达到了97.9%和89.6%的准确率，显著优于基础RAG模型。 

---
# From Static to Dynamic: A Streaming RAG Approach to Real-time Knowledge Base 

**Title (ZH)**: 从静态到动态：一种实时知识库的流式RAG方法 

**Authors**: Yuzhou Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05662)  

**Abstract**: Dynamic streams from news feeds, social media, sensor networks, and financial markets challenge static RAG frameworks. Full-scale indices incur high memory costs; periodic rebuilds introduce latency that undermines data freshness; naive sampling sacrifices semantic coverage. We present Streaming RAG, a unified pipeline that combines multi-vector cosine screening, mini-batch clustering, and a counter-based heavy-hitter filter to maintain a compact prototype set. We further prove an approximation bound \$E\[R(K\_t)] \ge R^\* - L \Delta\$ linking retrieval quality to clustering variance. An incremental index upsert mechanism refreshes prototypes without interrupting queries. Experiments on eight real-time streams show statistically significant gains in Recall\@10 (up to 3 points, p < 0.01), end-to-end latency below 15 ms, and throughput above 900 documents per second under a 150 MB budget. Hyperparameter sensitivity analysis over cluster count, admission probability, relevance threshold, and counter capacity validates default settings. In open-domain question answering with GPT-3.5 Turbo, we record 3.2-point gain in Exact Match and 2.8-point gain in F1 on SQuAD; abstractive summarization yields ROUGE-L improvements. Streaming RAG establishes a new Pareto frontier for retrieval augmentation. 

**Abstract (ZH)**: 动态流处理新闻 feed、社交媒体、传感器网络和金融市场数据挑战静态RAG框架。Streaming RAG：统一多向量余弦筛选、 mini-批聚类和基于计数的重量级元素过滤管道以维护紧凑原型集 

---
# Zero-Shot Retrieval for Scalable Visual Search in a Two-Sided Marketplace 

**Title (ZH)**: 面向双方市场的可扩展视觉搜索零样本检索 

**Authors**: Andre Rusli, Shoma Ishimoto, Sho Akiyama, Aman Kumar Singh  

**Link**: [PDF](https://arxiv.org/pdf/2508.05661)  

**Abstract**: Visual search offers an intuitive way for customers to explore diverse product catalogs, particularly in consumer-to-consumer (C2C) marketplaces where listings are often unstructured and visually driven. This paper presents a scalable visual search system deployed in Mercari's C2C marketplace, where end-users act as buyers and sellers. We evaluate recent vision-language models for zero-shot image retrieval and compare their performance with an existing fine-tuned baseline. The system integrates real-time inference and background indexing workflows, supported by a unified embedding pipeline optimized through dimensionality reduction. Offline evaluation using user interaction logs shows that the multilingual SigLIP model outperforms other models across multiple retrieval metrics, achieving a 13.3% increase in nDCG@5 over the baseline. A one-week online A/B test in production further confirms real-world impact, with the treatment group showing substantial gains in engagement and conversion, up to a 40.9% increase in transaction rate via image search. Our findings highlight that recent zero-shot models can serve as a strong and practical baseline for production use, which enables teams to deploy effective visual search systems with minimal overhead, while retaining the flexibility to fine-tune based on future data or domain-specific needs. 

**Abstract (ZH)**: 视觉搜索为顾客探索多样化的商品目录提供了一种直观的方式，特别是在消费者对消费者（C2C）市场中，商品列表往往无结构且以视觉为导向。本文介绍了Mercari C2C市场中部署的一种可扩展的视觉搜索系统，其中最终用户既充当买家也充当卖家。我们评估了最近的视觉-语言模型在零样本图像检索中的性能，并将其与现有微调基线进行了比较。该系统结合了实时推理和后台索引工作流程，通过降维优化实现了统一嵌入管道。使用用户交互日志的离线评估显示，多语言SigLIP模型在多个检索指标上优于其他模型，对基线的nDCG@5提高了13.3%。生产环境中的为期一周的A/B测试进一步证实了实际影响，治疗组在通过图像搜索进行互动和转化方面分别提高了40.9%。我们的研究结果表明，最近的零样本模型可以作为生产环境中强有力的实用基线，使得团队能够在最少的开销下部署有效的视觉搜索系统，同时能够根据未来的数据或特定领域需求进行微调。 

---
# Open-Source Agentic Hybrid RAG Framework for Scientific Literature Review 

**Title (ZH)**: 开源代理混合检索框架：科学文献综述 

**Authors**: Aditya Nagori, Ricardo Accorsi Casonatto, Ayush Gautam, Abhinav Manikantha Sai Cheruvu, Rishikesan Kamaleswaran  

**Link**: [PDF](https://arxiv.org/pdf/2508.05660)  

**Abstract**: The surge in scientific publications challenges traditional review methods, demanding tools that integrate structured metadata with full-text analysis. Hybrid Retrieval Augmented Generation (RAG) systems, combining graph queries with vector search offer promise but are typically static, rely on proprietary tools, and lack uncertainty estimates. We present an agentic approach that encapsulates the hybrid RAG pipeline within an autonomous agent capable of (1) dynamically selecting between GraphRAG and VectorRAG for each query, (2) adapting instruction-tuned generation in real time to researcher needs, and (3) quantifying uncertainty during inference. This dynamic orchestration improves relevance, reduces hallucinations, and promotes reproducibility.
Our pipeline ingests bibliometric open-access data from PubMed, arXiv, and Google Scholar APIs, builds a Neo4j citation-based knowledge graph (KG), and embeds full-text PDFs into a FAISS vector store (VS) using the all-MiniLM-L6-v2 model. A Llama-3.3-70B agent selects GraphRAG (translating queries to Cypher for KG) or VectorRAG (combining sparse and dense retrieval with re-ranking). Instruction tuning refines domain-specific generation, and bootstrapped evaluation yields standard deviation for evaluation metrics.
On synthetic benchmarks mimicking real-world queries, the Instruction-Tuned Agent with Direct Preference Optimization (DPO) outperforms the baseline, achieving a gain of 0.63 in VS Context Recall and a 0.56 gain in overall Context Precision. Additional gains include 0.24 in VS Faithfulness, 0.12 in both VS Precision and KG Answer Relevance, 0.11 in overall Faithfulness score, 0.05 in KG Context Recall, and 0.04 in both VS Answer Relevance and overall Precision. These results highlight the system's improved reasoning over heterogeneous sources and establish a scalable framework for autonomous, agentic scientific discovery. 

**Abstract (ZH)**: 科学出版物的激增挑战了传统审核方法，要求集成结构化元数据与全文分析的工具。混合检索增强生成（RAG）系统结合图查询与向量搜索展现出潜力，但通常较为静态，依赖于专有工具，并缺乏不确定性估计。我们提出了一种代理方法，将混合RAG管道封装在一个自主代理中，该代理能够（1）根据每个查询动态选择GraphRAG和VectorRAG，（2）实时适应研究需求进行指令微调生成，以及（3）在推断过程中量化不确定性。这种动态编排提高了相关性，减少了妄言，并促进了可重复性。

我们的管道从PubMed、arXiv和Google Scholar API摄取 bibliometric 开放访问数据，构建基于引文的Neo4j知识图谱（KG），并使用all-MiniLM-L6-v2模型将全文PDF嵌入FAISS向量存储（VS）。Llama-3.3-70B代理选择GraphRAG（将查询转换为KG的Cypher查询）或VectorRAG（结合稀疏和密集检索并重新排序）。指令微调细化了领域特定生成，而自强化评估提供了评估指标的标准差。

在模拟实际情况查询的合成基准测试中，带有直接偏好优化（DPO）的指令微调代理优于基线，VS上下文召回率提高0.63，整体上下文精准度提高0.56。其他改进包括VS忠实度提高0.24、VS精准度提高0.12和知识图谱答案相关性提高0.12、整体忠实度评分提高0.11、知识图谱上下文召回率提高0.05以及VS答案相关性和整体精准度分别提高0.04。这些结果突显了系统在异构来源上改进推理的能力，并建立了自主、代理化的科学发现可扩展框架。 

---
# Comparison of Information Retrieval Techniques Applied to IT Support Tickets 

**Title (ZH)**: IT支持票文中信息检索技术的比较 

**Authors**: Leonardo Santiago Benitez Pereira, Robinson Pizzio, Samir Bonho  

**Link**: [PDF](https://arxiv.org/pdf/2508.05654)  

**Abstract**: Institutions dependent on IT services and resources acknowledge the crucial significance of an IT help desk system, that act as a centralized hub connecting IT staff and users for service requests. Employing various Machine Learning models, these IT help desk systems allow access to corrective actions used in the past, but each model has different performance when applied to different datasets. This work compares eleven Information Retrieval techniques in a dataset of IT support tickets, with the goal of implementing a software that facilitates the work of Information Technology support analysts. The best results were obtained with the Sentence-BERT technique, in its multi-language variation distilluse-base-multilingual-cased-v1, where 78.7% of the recommendations made by the model were considered relevant. TF-IDF (69.0%), Word2vec (68.7%) and LDA (66.3%) techniques also had consistent results. Furthermore, the used datasets and essential parts of coding have been published and made open source. It also demonstrated the practicality of a support ticket recovery system by implementing a minimal viable prototype, and described in detail the implementation of the system. Finally, this work proposed a novel metric for comparing the techniques, whose aim is to closely reflect the perception of the IT analysts about the retrieval quality. 

**Abstract (ZH)**: 基于IT服务和资源的机构认识到IT帮助台系统的重要性，该系统充当IT人员和用户之间的集中枢纽，用于服务请求。这些IT帮助台系统利用各种机器学习模型，允许访问过去采取的纠正措施，但每个模型在应用于不同数据集时表现不同。本研究在IT支持票务数据集中比较了十一种信息检索技术，旨在实施一款支持信息技术支持分析师工作的软件。Sentence-BERT技术（特别是在多语言变体distilluse-base-multilingual-cased-v1）取得了最佳效果，其中78.7%的模型推荐被认为相关。TF-IDF（69.0%）、Word2vec（68.7%）和LDA（66.3%）技术也取得了持续的结果。此外，使用的数据集和编码的关键部分已被发布并开源。该研究还通过实现最小可行原型演示了支持票务恢复系统的实用性，并详细描述了系统的实现。最后，本文提出了一种新的技术对比指标，其目的是尽量贴近信息技术分析师对于检索质量的感知。 

---
# Modeling Interactive Narrative Systems: A Formal Approach 

**Title (ZH)**: 建模互动叙事系统：一种形式化方法 

**Authors**: Jules Clerc, Domitile Lourdeaux, Mohamed Sallak, Johann Barbier, Marc Ravaine  

**Link**: [PDF](https://arxiv.org/pdf/2508.05653)  

**Abstract**: Interactive Narrative Systems (INS) have revolutionized digital experiences by empowering users to actively shape their stories, diverging from traditional passive storytelling. However, the field faces challenges due to fragmented research efforts and diverse system representations. This paper introduces a formal representation framework for INS, inspired by diverse approaches from the state of the art. By providing a consistent vocabulary and modeling structure, the framework facilitates the analysis, the description and comparison of INS properties. Experimental validations on the "Little Red Riding Hood" scenario highlight the usefulness of the proposed formalism and its impact on improving the evaluation of INS. This work aims to foster collaboration and coherence within the INS research community by proposing a methodology for formally representing these systems. 

**Abstract (ZH)**: 交互叙事系统（INS）通过让用户主动塑造故事，革新了数字体验，超越了传统的被动叙述方式。然而，由于研究努力分散和系统表示多样化的挑战，该领域面临挑战。本文介绍了一种正式表示框架，该框架受到最新技术多样方法的启发。通过提供一致的词汇和建模结构，该框架促进了对INS属性的分析、描述和比较。在“小红帽”场景上的实验验证强调了所提出正式主义的实用性及其对改进INS评估的影响。本文旨在通过提出正式表示这些系统的办法，促进INS研究社区的合作与一致性。 

---
# Query-Aware Graph Neural Networks for Enhanced Retrieval-Augmented Generation 

**Title (ZH)**: 面向查询的图神经网络在增强检索增强生成中的应用 

**Authors**: Vibhor Agrawal, Fay Wang, Rishi Puri  

**Link**: [PDF](https://arxiv.org/pdf/2508.05647)  

**Abstract**: We present a novel graph neural network (GNN) architecture for retrieval-augmented generation (RAG) that leverages query-aware attention mechanisms and learned scoring heads to improve retrieval accuracy on complex, multi-hop questions. Unlike traditional dense retrieval methods that treat documents as independent entities, our approach constructs per-episode knowledge graphs that capture both sequential and semantic relationships between text chunks. We introduce an Enhanced Graph Attention Network with query-guided pooling that dynamically focuses on relevant parts of the graph based on user queries. Experimental results demonstrate that our approach significantly outperforms standard dense retrievers on complex question answering tasks, particularly for questions requiring multi-document reasoning. Our implementation leverages PyTorch Geometric for efficient processing of graph-structured data, enabling scalable deployment in production retrieval systems 

**Abstract (ZH)**: 一种基于查询感知注意力机制和学习评分头部的新型图神经网络架构：增强图注意力网络在复杂多跳问答中的检索增强生成 

---
# Request-Only Optimization for Recommendation Systems 

**Title (ZH)**: 仅请求优化的推荐系统 

**Authors**: Liang Guo, Wei Li, Lucy Liao, Huihui Cheng, Rui Zhang, Yu Shi, Yueming Wang, Yanzun Huang, Keke Zhai, Pengchao Wang, Timothy Shi, Xuan Cao, Shengzhi Wang, Renqin Cai, Zhaojie Gong, Omkar Vichare, Rui Jian, Leon Gao, Shiyan Deng, Xingyu Liu, Xiong Zhang, Fu Li, Wenlei Xie, Bin Wen, Rui Li, Xing Liu, Jiaqi Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2508.05640)  

**Abstract**: Deep Learning Recommendation Models (DLRMs) represent one of the largest machine learning applications on the planet. Industry-scale DLRMs are trained with petabytes of recommendation data to serve billions of users every day. To utilize the rich user signals in the long user history, DLRMs have been scaled up to unprecedented complexity, up to trillions of floating-point operations (TFLOPs) per example. This scale, coupled with the huge amount of training data, necessitates new storage and training algorithms to efficiently improve the quality of these complex recommendation systems. In this paper, we present a Request-Only Optimizations (ROO) training and modeling paradigm. ROO simultaneously improves the storage and training efficiency as well as the model quality of recommendation systems. We holistically approach this challenge through co-designing data (i.e., request-only data), infrastructure (i.e., request-only based data processing pipeline), and model architecture (i.e., request-only neural architectures). Our ROO training and modeling paradigm treats a user request as a unit of the training data. Compared with the established practice of treating a user impression as a unit, our new design achieves native feature deduplication in data logging, consequently saving data storage. Second, by de-duplicating computations and communications across multiple impressions in a request, this new paradigm enables highly scaled-up neural network architectures to better capture user interest signals, such as Generative Recommenders (GRs) and other request-only friendly architectures. 

**Abstract (ZH)**: 深度学习推荐模型（DLRMs）代表了世界上最大的机器学习应用之一。工业规模的DLRMs通过巨量的推荐数据训练，每天为数十亿用户提供服务。为了利用长用户历史中的丰富用户信号，DLRMs被扩展到前所未有的复杂度，每例达到数万亿次浮点运算（TFLOPs）。这种规模，结合大量的训练数据，需要新的存储和训练算法以高效地改进这些复杂推荐系统的质量。在本文中，我们提出了一种仅请求优化（ROO）的训练和建模范式。ROO 同时改进了推荐系统的存储和训练效率以及模型质量。通过协同设计数据（即仅请求数据）、基础设施（即基于请求的数据处理管道）以及模型架构（即仅请求的神经架构），我们从整体上应对了这一挑战。与将用户印象视为训练数据单位的传统做法相比，我们的新设计在数据日志中实现了原生特征去重，因此节省了数据存储空间。其次，通过在一个请求中对多个印象进行计算和通信去重，这种新范式使得可扩展的神经网络架构能够更好地捕捉用户兴趣信号，如生成型推荐器（GRs）和其他仅请求友好的架构。 

---
# SHACL Validation in the Presence of Ontologies: Semantics and Rewriting Techniques 

**Title (ZH)**: 面向本体的SHACL验证：语义与重写技术 

**Authors**: Anouk Oudshoorn, Magdalena Ortiz, Mantas Simkus  

**Link**: [PDF](https://arxiv.org/pdf/2507.12286)  

**Abstract**: SHACL and OWL are two prominent W3C standards for managing RDF data. These languages share many features, but they have one fundamental difference: OWL, designed for inferring facts from incomplete data, makes the open-world assumption, whereas SHACL is a constraint language that treats the data as complete and must be validated under the closed-world assumption. The combination of both formalisms is very appealing and has been called for, but their semantic gap is a major challenge, semantically and computationally. In this paper, we advocate a semantics for SHACL validation in the presence of ontologies based on core universal models. We provide a technique for constructing these models for ontologies in the rich data-tractable description logic Horn-ALCHIQ. Furthermore, we use a finite representation of this model to develop a rewriting technique that reduces SHACL validation in the presence of ontologies to standard validation. Finally, we study the complexity of SHACL validation in the presence of ontologies, and show that even very simple ontologies make the problem EXPTIME-complete, and PTIME-complete in data complexity. 

**Abstract (ZH)**: SHACL和OWL是两个 promin 华盛标准，用于管理RDF数据。这些语言共享许多特征，但它们有一个基本区别：OWL旨在从不完整数据中推断事实，基于开放世界假设，而SHACL是一种约束语言，认为数据是完整的，并且必须在封闭世界假设下进行验证。这两种形式主义的结合非常诱人，但它们的语义差距是一个主要挑战，从语义和计算角度来看都是。在本文中，我们提出了一种基于核心通用模型的SHACL验证语义，在 ontology 存在的情况下。我们提供了一种技术来为丰富的数据可处理描述逻辑Horn-ALCHIQ中的ontology构造这些模型。此外，我们使用该模型的有限表示来开发一种重写技术，将ontology存在下的SHACL验证减少为标准验证。最后，我们研究了ontology存在下的SHACL验证的复杂性，并展示了即使是非常简单的ontology也会使问题变为EXPTIME完全问题，在数据复杂性方面则为PTIME完全问题。 

---
# Epidemic Control on a Large-Scale-Agent-Based Epidemiology Model using Deep Deterministic Policy Gradient 

**Title (ZH)**: 大规模基于代理的流行病学模型中基于深度确定性策略梯度的流行病控制 

**Authors**: Gaurav Deshkar, Jayanta Kshirsagar, Harshal Hayatnagarkar, Janani Venugopalan  

**Link**: [PDF](https://arxiv.org/pdf/2304.04475)  

**Abstract**: To mitigate the impact of the pandemic, several measures include lockdowns, rapid vaccination programs, school closures, and economic stimulus. These interventions can have positive or unintended negative consequences. Current research to model and determine an optimal intervention automatically through round-tripping is limited by the simulation objectives, scale (a few thousand individuals), model types that are not suited for intervention studies, and the number of intervention strategies they can explore (discrete vs continuous). We address these challenges using a Deep Deterministic Policy Gradient (DDPG) based policy optimization framework on a large-scale (100,000 individual) epidemiological agent-based simulation where we perform multi-objective optimization. We determine the optimal policy for lockdown and vaccination in a minimalist age-stratified multi-vaccine scenario with a basic simulation for economic activity. With no lockdown and vaccination (mid-age and elderly), results show optimal economy (individuals below the poverty line) with balanced health objectives (infection, and hospitalization). An in-depth simulation is needed to further validate our results and open-source our framework. 

**Abstract (ZH)**: 为了减轻疫情的影响，采取了封锁、快速疫苗接种计划、学校关闭和经济刺激等多种措施。这些干预措施可能会带来积极或未预见的负面后果。目前通过往返模拟来建模并自动生成最优干预措施的研究受到模拟目标、规模（几千人）、不适用于干预研究的模型类型以及可探索的干预策略数量（离散 vs 连续）的限制。我们通过在大规模（10万人）流行病学基于代理的模拟中利用基于Deep Deterministic Policy Gradient (DDPG) 的策略优化框架来应对这些挑战，并进行多目标优化。我们在一个简化了年龄分层多疫苗的模拟中确定了最优封锁和接种疫苗政策，并进行了基本的经济活动模拟。在没有封锁和接种疫苗（中老年）的情况下，结果显示最优经济活动（低于贫困线的个体）和平衡的健康目标（感染和住院）。需要进行深入的模拟进一步验证我们的结果并开源我们的框架。 

---
