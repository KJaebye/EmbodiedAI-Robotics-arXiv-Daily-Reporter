# Nominal Evaluation Of Automatic Multi-Sections Control Potential In Comparison To A Simpler One- Or Two-Sections Alternative With Predictive Spray Switching 

**Title (ZH)**: 名义评估自动多段控制潜力与简单的一段或多段替代方案（带有预测性喷雾切换）的比较 

**Authors**: Mogens Plessen  

**Link**: [PDF](https://arxiv.org/pdf/2508.11573)  

**Abstract**: Automatic Section Control (ASC) is a long-standing trend for spraying in agriculture. It promises to minimise spray overlap areas. The core idea is to (i) switch off spray nozzles on areas that have already been sprayed, and (ii) to dynamically adjust nozzle flow rates along the boom bar that holds the spray nozzles when velocities of boom sections vary during turn maneuvers. ASC is not possible without sensors, in particular for accurate positioning data. Spraying and the movement of modern wide boom bars are highly dynamic processes. In addition, many uncertainty factors have an effect such as cross wind drift, boom height, nozzle clogging in open-field conditions, and so forth. In view of this complexity, the natural question arises if a simpler alternative exist. Therefore, an Automatic Multi-Sections Control method is compared to a proposed simpler one- or two-sections alternative that uses predictive spray switching. The comparison is provided under nominal conditions. Agricultural spraying is intrinsically linked to area coverage path planning and spray switching logic. Combinations of two area coverage path planning and switching logics as well as three sections-setups are compared. The three sections-setups differ by controlling 48 sections, 2 sections or controlling all nozzles uniformly with the same control signal as one single section. Methods are evaluated on 10 diverse real-world field examples, including non-convex field contours, freeform mainfield lanes and multiple obstacle areas. A preferred method is suggested that (i) minimises area coverage pathlength, (ii) offers intermediate overlap, (iii) is suitable for manual driving by following a pre-planned predictive spray switching logic for an area coverage path plan, and (iv) and in contrast to ASC can be implemented sensor-free and therefore at low cost. 

**Abstract (ZH)**: 自动多区域控制（AMC）方法及其预测喷洒切换的简单替代方案比较 

---
# A Recursive Total Least Squares Solution for Bearing-Only Target Motion Analysis and Circumnavigation 

**Title (ZH)**: 基于轴承数据的运动分析与环绕解算的递归最小二乘解法 

**Authors**: Lin Li, Xueming Liu, Zhoujingzi Qiu, Tianjiang Hu, Qingrui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11289)  

**Abstract**: Bearing-only Target Motion Analysis (TMA) is a promising technique for passive tracking in various applications as a bearing angle is easy to measure. Despite its advantages, bearing-only TMA is challenging due to the nonlinearity of the bearing measurement model and the lack of range information, which impairs observability and estimator convergence. This paper addresses these issues by proposing a Recursive Total Least Squares (RTLS) method for online target localization and tracking using mobile observers. The RTLS approach, inspired by previous results on Total Least Squares (TLS), mitigates biases in position estimation and improves computational efficiency compared to pseudo-linear Kalman filter (PLKF) methods. Additionally, we propose a circumnavigation controller to enhance system observability and estimator convergence by guiding the mobile observer in orbit around the target. Extensive simulations and experiments are performed to demonstrate the effectiveness and robustness of the proposed method. The proposed algorithm is also compared with the state-of-the-art approaches, which confirms its superior performance in terms of both accuracy and stability. 

**Abstract (ZH)**: 仅轴承目标运动分析（TMA）是各种应用中一种有前途的被动跟踪技术，因为轴承角容易测量。尽管具有优势，但仅轴承TMA由于航向测量模型的非线性及缺乏距离信息，影响了可观测性和估计器的收敛性。本文通过提出一种递归最小总平方（RTLS）方法，解决在线目标定位与跟踪问题，利用移动观测者。RTLS方法借鉴了最小总平方（TLS）的先前结果，减少了位置估计的偏倚并提高了计算效率，相较于伪线性卡尔曼滤波（PLKF）方法。此外，我们提出了一种循航控制器，通过引导移动观测者围绕目标进行循航，增强系统的可观测性和估计器的收敛性。进行了广泛的仿真和实验，以证明所提方法的有效性和鲁棒性。将所提算法与最新方法进行了比较，证实了其在准确性和稳定性方面的优越性能。 

---
# Landmark-Assisted Monte Carlo Planning 

**Title (ZH)**: 地标辅助蒙特卡洛规划 

**Authors**: David H. Chan, Mark Roberts, Dana S. Nau  

**Link**: [PDF](https://arxiv.org/pdf/2508.11493)  

**Abstract**: Landmarks$\unicode{x2013}$conditions that must be satisfied at some point in every solution plan$\unicode{x2013}$have contributed to major advancements in classical planning, but they have seldom been used in stochastic domains. We formalize probabilistic landmarks and adapt the UCT algorithm to leverage them as subgoals to decompose MDPs; core to the adaptation is balancing between greedy landmark achievement and final goal achievement. Our results in benchmark domains show that well-chosen landmarks can significantly improve the performance of UCT in online probabilistic planning, while the best balance of greedy versus long-term goal achievement is problem-dependent. The results suggest that landmarks can provide helpful guidance for anytime algorithms solving MDPs. 

**Abstract (ZH)**: 地标：条件必须在每个解决方案计划中的某个点被满足，它们在经典规划中取得了重大进展，但很少被用于随机领域。我们形式化了概率地标，并适应了UCT算法以利用它们作为子目标来分解MDPs；适应的核心在于权衡贪婪地标达成与最终目标达成之间的平衡。基准领域中的实验结果显示，精心选择的地标可以显著提高UCT在在线概率规划中的性能，而贪婪与长期目标达成的最佳平衡依赖于具体问题。这些结果表明，地标可以为解决MDPs的随时算法提供有益的指导。 

---
# SAGE: Scale-Aware Gradual Evolution for Continual Knowledge Graph Embedding 

**Title (ZH)**: SAGE: 意识到规模的逐步演化持续知识图嵌入 

**Authors**: Yifei Li, Lingling Zhang, Hang Yan, Tianzhe Zhao, Zihan Ma, Muye Huang, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.11347)  

**Abstract**: Traditional knowledge graph (KG) embedding methods aim to represent entities and relations in a low-dimensional space, primarily focusing on static graphs. However, real-world KGs are dynamically evolving with the constant addition of entities, relations and facts. To address such dynamic nature of KGs, several continual knowledge graph embedding (CKGE) methods have been developed to efficiently update KG embeddings to accommodate new facts while maintaining learned knowledge. As KGs grow at different rates and scales in real-world scenarios, existing CKGE methods often fail to consider the varying scales of updates and lack systematic evaluation throughout the entire update process. In this paper, we propose SAGE, a scale-aware gradual evolution framework for CKGE. Specifically, SAGE firstly determine the embedding dimensions based on the update scales and expand the embedding space accordingly. The Dynamic Distillation mechanism is further employed to balance the preservation of learned knowledge and the incorporation of new facts. We conduct extensive experiments on seven benchmarks, and the results show that SAGE consistently outperforms existing baselines, with a notable improvement of 1.38% in MRR, 1.25% in H@1 and 1.6% in H@10. Furthermore, experiments comparing SAGE with methods using fixed embedding dimensions show that SAGE achieves optimal performance on every snapshot, demonstrating the importance of adaptive embedding dimensions in CKGE. The codes of SAGE are publicly available at: this https URL. 

**Abstract (ZH)**: 面向规模感知渐进演化的持续知识图嵌入框架SAGE 

---
# On Strong and Weak Admissibility in Non-Flat Assumption-Based Argumentation 

**Title (ZH)**: 非平坦假设论辩中的强可接受性和弱可接受性研究 

**Authors**: Matti Berthold, Lydia Blümel, Anna Rapberger  

**Link**: [PDF](https://arxiv.org/pdf/2508.11182)  

**Abstract**: In this work, we broaden the investigation of admissibility notions in the context of assumption-based argumentation (ABA). More specifically, we study two prominent alternatives to the standard notion of admissibility from abstract argumentation, namely strong and weak admissibility, and introduce the respective preferred, complete and grounded semantics for general (sometimes called non-flat) ABA. To do so, we use abstract bipolar set-based argumentation frameworks (BSAFs) as formal playground since they concisely capture the relations between assumptions and are expressive enough to represent general non-flat ABA frameworks, as recently shown. While weak admissibility has been recently investigated for a restricted fragment of ABA in which assumptions cannot be derived (flat ABA), strong admissibility has not been investigated for ABA so far. We introduce strong admissibility for ABA and investigate desirable properties. We furthermore extend the recent investigations of weak admissibility in the flat ABA fragment to the non-flat case. We show that the central modularization property is maintained under classical, strong, and weak admissibility. We also show that strong and weakly admissible semantics in non-flat ABA share some of the shortcomings of standard admissible semantics and discuss ways to address these. 

**Abstract (ZH)**: 在这项工作中，我们扩展了对假设基于论辩论中容许性概念的研究。具体地，我们研究了抽象论辩中标准容许性概念的两种主要替代方案，即强容许性和弱容许性，并引入了适用于一般（有时称为非扁平）假设基于论辩的相应优先级、完备性和基础语义。为此，我们使用抽象双极集合论辩框架（BSAFs）作为形式化的研究平台，因为它们简洁地捕捉了假设之间的关系，并且能够表达一般非扁平假设基于论辩框架，这是最近的研究成果。虽然弱容许性最近在假设不可推导的扁平假设基于论辩片段中进行了研究，但强容许性尚未在假设基于论辩中进行研究。我们为假设基于论辩引入了强容许性并研究了其 desirable 属性。我们还扩展了在扁平假设基于论辩片段中对弱容许性的最近研究，将其推广到非扁平情况。我们证明了在经典、强和弱容许性下保持了核心模块化特性。我们还展示了非扁平假设基于论辩中的强容许性和弱容许性语义与标准容许性语义共享的一些缺点，并讨论了解决这些问题的方法。 

---
# Learn to optimize for automatic proton PBS treatment planning for H&N cancers 

**Title (ZH)**: 自动质子治疗计划优化学习：针对头颈癌的H&N癌症质子束治疗计划自动化优化 

**Authors**: Qingqing Wang, Liqiang Xiao, Chang Chang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11085)  

**Abstract**: Proton PBS treatment planning for H&N cancers involves numerous conflicting objectives, requiring significant effort from human planners to balance and satisfy multiple clinical goals during planning. To achieve this, experience-demanding objective parameter adjustment and computationally expensive inverse optimization are performed iteratively. Extensive efforts have been made to automatically adjust objective parameters, but the most time-consuming component, i.e., inverse optimization, still relies heavily on theory-driven approaches. We propose a data-driven inverse optimizer and integrate it into a PPO-based automatic treatment planning framework to automatically generate high-quality plans within a clinical acceptable planning time. The inverse optimizer is a L2O method that predicts update steps by learning from the task-specific data distribution. For the first time, we integrate techniques designed for long-context processing, originally developed for LLMs, into a Transformer-based L2O framework to address the scalability issue of existing L2O methods. The PPO framework functions as an outer-loop virtual planner, autonomously adjusting objective parameters through a policy network, and the dose predictor is used to initialize objective parameters. The inner-loop L2O inverse optimizer computes machine-deliverable MU values based on objectives refined by the PPO policy network. 97 patients are collected in this study, and compared with L-BFGSB, our L2O-based inverse optimizer improves the effectiveness and efficiency by 22.97% and 36.41%, respectively. In conjunction with the PPO-based learned virtual planner, plans generated by our framework within an average of 2.55 hours show improved or comparable OAR sparing with superior target coverage for patients with different prescription dose levels, number of target volumes, beam angles, etc., compared with human-generated plans. 

**Abstract (ZH)**: 数据驱动的质子PBS治疗计划方法及其在H&N癌症治疗中的应用：基于PPO的方法 

---
# From Individual to Multi-Agent Algorithmic Recourse: Minimizing the Welfare Gap via Capacitated Bipartite Matching 

**Title (ZH)**: 从个体到多智能体算法可问责性：通过容量受限二部图匹配最小化福利差距 

**Authors**: Zahra Khotanlou, Kate Larson, Amir-Hossein Karimi  

**Link**: [PDF](https://arxiv.org/pdf/2508.11070)  

**Abstract**: Decision makers are increasingly relying on machine learning in sensitive situations. In such settings, algorithmic recourse aims to provide individuals with actionable and minimally costly steps to reverse unfavorable AI-driven decisions. While existing research predominantly focuses on single-individual (i.e., seeker) and single-model (i.e., provider) scenarios, real-world applications often involve multiple interacting stakeholders. Optimizing outcomes for seekers under an individual welfare approach overlooks the inherently multi-agent nature of real-world systems, where individuals interact and compete for limited resources. To address this, we introduce a novel framework for multi-agent algorithmic recourse that accounts for multiple recourse seekers and recourse providers. We model this many-to-many interaction as a capacitated weighted bipartite matching problem, where matches are guided by both recourse cost and provider capacity. Edge weights, reflecting recourse costs, are optimized for social welfare while quantifying the welfare gap between individual welfare and this collectively feasible outcome. We propose a three-layer optimization framework: (1) basic capacitated matching, (2) optimal capacity redistribution to minimize the welfare gap, and (3) cost-aware optimization balancing welfare maximization with capacity adjustment costs. Experimental validation on synthetic and real-world datasets demonstrates that our framework enables the many-to-many algorithmic recourse to achieve near-optimal welfare with minimum modification in system settings. This work extends algorithmic recourse from individual recommendations to system-level design, providing a tractable path toward higher social welfare while maintaining individual actionability. 

**Abstract (ZH)**: 多代理算法救济框架：从个体推荐到系统级设计 

---
# Grounding Rule-Based Argumentation Using Datalog 

**Title (ZH)**: 基于Datalog的规则推理论辩 grounding 

**Authors**: Martin Diller, Sarah Alice Gaggl, Philipp Hanisch, Giuseppina Monterosso, Fritz Rauschenbach  

**Link**: [PDF](https://arxiv.org/pdf/2508.10976)  

**Abstract**: ASPIC+ is one of the main general frameworks for rule-based argumentation for AI. Although first-order rules are commonly used in ASPIC+ examples, most existing approaches to reason over rule-based argumentation only support propositional rules. To enable reasoning over first-order instances, a preliminary grounding step is required. As groundings can lead to an exponential increase in the size of the input theories, intelligent procedures are needed. However, there is a lack of dedicated solutions for ASPIC+. Therefore, we propose an intelligent grounding procedure that keeps the size of the grounding manageable while preserving the correctness of the reasoning process. To this end, we translate the first-order ASPIC+ instance into a Datalog program and query a Datalog engine to obtain ground substitutions to perform the grounding of rules and contraries. Additionally, we propose simplifications specific to the ASPIC+ formalism to avoid grounding of rules that have no influence on the reasoning process. Finally, we performed an empirical evaluation of a prototypical implementation to show scalability. 

**Abstract (ZH)**: ASPIC+是一种基于规则的论证主要通用框架之一。虽然ASPIC+示例中常用一阶规则，但大多数基于规则的论证推理方法只支持命题规则。为了在一阶实例上进行推理，需要一个初步的实例化步骤。由于实例化可能导致输入理论的大小指数级增加，因此需要智能过程。然而，缺少针对ASPIC+的专用解决方案。因此，我们提出了一种智能实例化程序，该程序能够在保持实例化规模可控的同时，保持推理过程的正确性。为此，我们将一阶ASPIC+实例转换为Datalog程序，并查询Datalog引擎以获取地面置换，进行规则和反例的实例化。此外，我们针对ASPIC+形式主义提出了一种特定的简化方法，以避免实例化对推理过程没有影响的规则。最后，我们对一个原型实现进行了实证评估，以展示其实现的可扩展性。 

---
# A Comprehensive Perspective on Explainable AI across the Machine Learning Workflow 

**Title (ZH)**: 全面视角下的可解释人工智能在整个机器学习工作流中的应用 

**Authors**: George Paterakis, Andrea Castellani, George Papoutsoglou, Tobias Rodemann, Ioannis Tsamardinos  

**Link**: [PDF](https://arxiv.org/pdf/2508.11529)  

**Abstract**: Artificial intelligence is reshaping science and industry, yet many users still regard its models as opaque "black boxes". Conventional explainable artificial-intelligence methods clarify individual predictions but overlook the upstream decisions and downstream quality checks that determine whether insights can be trusted. In this work, we present Holistic Explainable Artificial Intelligence (HXAI), a user-centric framework that embeds explanation into every stage of the data-analysis workflow and tailors those explanations to users. HXAI unifies six components (data, analysis set-up, learning process, model output, model quality, communication channel) into a single taxonomy and aligns each component with the needs of domain experts, data analysts and data scientists. A 112-item question bank covers these needs; our survey of contemporary tools highlights critical coverage gaps. Grounded in theories of human explanation, principles from human-computer interaction and findings from empirical user studies, HXAI identifies the characteristics that make explanations clear, actionable and cognitively manageable. A comprehensive taxonomy operationalises these insights, reducing terminological ambiguity and enabling rigorous coverage analysis of existing toolchains. We further demonstrate how AI agents that embed large-language models can orchestrate diverse explanation techniques, translating technical artifacts into stakeholder-specific narratives that bridge the gap between AI developers and domain experts. Departing from traditional surveys or perspective articles, this work melds concepts from multiple disciplines, lessons from real-world projects and a critical synthesis of the literature to advance a novel, end-to-end viewpoint on transparency, trustworthiness and responsible AI deployment. 

**Abstract (ZH)**: 人工智能重塑科学与产业，但许多用户仍视其模型为不透明的“黑盒”。传统的可解释人工智能方法虽能阐明个体预测，却忽视了决定洞察是否可信赖的上游决策和下游质量检查。本文提出了综合可解释人工智能（HXAI）框架，该框架以用户为中心，在数据分析工作流的每个阶段嵌入解释，并根据领域专家、数据分析师和数据科学家的需求进行定制。HXAI将六个组件（数据、分析设置、学习过程、模型输出、模型质量、沟通渠道）统一到一个分类体系中，并与这些用户的需要对齐。包含112个问题的问卷涵盖了这些需求；我们对当前工具的调研突显了关键的覆盖缺口。基于人类解释理论、人机交互原则及实证用户研究的发现，HXAI识别出使解释清晰、可操作且认知上易于管理的特征。一个全面的分类体系将这些见解具体化，减少了术语上的歧义，并使现有工具链的严格覆盖分析成为可能。我们进一步展示了嵌入大语言模型的AI代理如何协调多样的解释技术，将技术成果转化为利益相关者特定的叙述，从而弥合AI开发者与领域专家之间的差距。本文综合了来自多个学科的概念、实际项目的经验教训以及文献的批判性综合，提出了关于透明度、可靠性和负责任AI部署的端到端新视角。 

---
# Weighted First Order Model Counting for Two-variable Logic with Axioms on Two Relations 

**Title (ZH)**: 带公理的双关系两变量逻辑的加权一阶模型计数 

**Authors**: Qipeng Kuang, Václav Kůla, Ondřej Kuželka, Yuanhong Wang, Yuyi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11515)  

**Abstract**: The Weighted First-Order Model Counting Problem (WFOMC) asks to compute the weighted sum of models of a given first-order logic sentence over a given domain. The boundary between fragments for which WFOMC can be computed in polynomial time relative to the domain size lies between the two-variable fragment ($\text{FO}^2$) and the three-variable fragment ($\text{FO}^3$). It is known that WFOMC for \FOthree{} is $\mathsf{\#P_1}$-hard while polynomial-time algorithms exist for computing WFOMC for $\text{FO}^2$ and $\text{C}^2$, possibly extended by certain axioms such as the linear order axiom, the acyclicity axiom, and the connectedness axiom. All existing research has concentrated on extending the fragment with axioms on a single distinguished relation, leaving a gap in understanding the complexity boundary of axioms on multiple relations. In this study, we explore the extension of the two-variable fragment by axioms on two relations, presenting both negative and positive results. We show that WFOMC for $\text{FO}^2$ with two linear order relations and $\text{FO}^2$ with two acyclic relations are $\mathsf{\#P_1}$-hard. Conversely, we provide an algorithm in time polynomial in the domain size for WFOMC of $\text{C}^2$ with a linear order relation, its successor relation and another successor relation. 

**Abstract (ZH)**: 加权一阶模型计数问题（WFOMC）要求计算给定的一阶逻辑句子在给定领域中的加权模型之和。能够在领域大小相对多项式时间内计算WFOMC的片段边界位于两变量片段（FO²）和三变量片段（FO³）之间。已知WFOMC对于FO³来说是P₁#-难的，而对于FO²和C²（可能扩展了一些特定的公理如线性序公理、无环公理和连通性公理）可以在多项式时间内计算。所有现有的研究都集中在使用针对单一特殊关系的公理扩展片段上，留下了关于涉及多个关系的公理复杂性边界的理解缺口。在这项研究中，我们探讨了使用两个关系的公理扩展两变量片段，提供了既有负面结果也有正面结果。我们证明了具有两个线性序关系的FO²和具有两个无环关系的FO²的WFOMC都是P₁#-难的。反过来，我们提供了在领域大小多项式时间内计算包含线性序关系、其后继关系和另一个后继关系的C²的WFOMC的算法。 

---
# Towards Faithful Class-level Self-explainability in Graph Neural Networks by Subgraph Dependencies 

**Title (ZH)**: 基于子图依赖的图神经网络阶层忠实自我解释研究 

**Authors**: Fanzhen Liu, Xiaoxiao Ma, Jian Yang, Alsharif Abuadbba, Kristen Moore, Surya Nepal, Cecile Paris, Quan Z. Sheng, Jia Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.11513)  

**Abstract**: Enhancing the interpretability of graph neural networks (GNNs) is crucial to ensure their safe and fair deployment. Recent work has introduced self-explainable GNNs that generate explanations as part of training, improving both faithfulness and efficiency. Some of these models, such as ProtGNN and PGIB, learn class-specific prototypes, offering a potential pathway toward class-level explanations. However, their evaluations focus solely on instance-level explanations, leaving open the question of whether these prototypes meaningfully generalize across instances of the same class. In this paper, we introduce GraphOracle, a novel self-explainable GNN framework designed to generate and evaluate class-level explanations for GNNs. Our model jointly learns a GNN classifier and a set of structured, sparse subgraphs that are discriminative for each class. We propose a novel integrated training that captures graph$\unicode{x2013}$subgraph$\unicode{x2013}$prediction dependencies efficiently and faithfully, validated through a masking-based evaluation strategy. This strategy enables us to retroactively assess whether prior methods like ProtGNN and PGIB deliver effective class-level explanations. Our results show that they do not. In contrast, GraphOracle achieves superior fidelity, explainability, and scalability across a range of graph classification tasks. We further demonstrate that GraphOracle avoids the computational bottlenecks of previous methods$\unicode{x2014}$like Monte Carlo Tree Search$\unicode{x2014}$by using entropy-regularized subgraph selection and lightweight random walk extraction, enabling faster and more scalable training. These findings position GraphOracle as a practical and principled solution for faithful class-level self-explainability in GNNs. 

**Abstract (ZH)**: 增强图神经网络（GNN）的可解释性对于确保其安全和公平部署至关重要。最近的工作引入了自解释的GNN，这些模型在训练过程中生成解释，提高了忠实度和效率。其中一些模型，如ProtGNN和PGIB，学习类特定的原型，为类水平的解释提供了一种潜在途径。然而，这些模型的评估仅关注实例水平的解释，留下了这样一个问题：这些原型在相同类别的不同实例之间是否真正具有泛化能力。在本文中，我们提出了GraphOracle，一种新颖的自解释GNN框架，旨在为GNN生成和评估类水平的解释。我们的模型联合学习一个GNN分类器和一组结构化、稀疏的子图，这些子图对每个类具有鉴别作用。我们提出了一种新颖的集成训练方法，能够高效且真实地捕捉图-子图-预测的依赖关系，并通过对比蒙特卡洛树搜索等方法的掩蔽评估策略得到验证。该策略使我们能够回顾性地评估先前方法（如ProtGNN和PGIB）是否提供了有效的类水平解释。我们的结果表明，它们未能实现这一点。相比之下，GraphOracle在多种图分类任务中实现了更高的忠实度、可解释性和可扩展性。此外，我们证明GraphOracle通过使用熵正则化的子图选择和轻量级随机游走提取避免了先前方法（如蒙特卡洛树搜索）的计算瓶颈，从而实现了更快和更可扩展的训练。这些发现将GraphOracle定位为图神经网络中真实可靠的类水平自解释性的实用且原理性的解决方案。 

---
# Handwritten Text Recognition of Historical Manuscripts Using Transformer-Based Models 

**Title (ZH)**: 基于变压器模型的历史手稿手写文本识别 

**Authors**: Erez Meoded  

**Link**: [PDF](https://arxiv.org/pdf/2508.11499)  

**Abstract**: Historical handwritten text recognition (HTR) is essential for unlocking the cultural and scholarly value of archival documents, yet digitization is often hindered by scarce transcriptions, linguistic variation, and highly diverse handwriting styles. In this study, we apply TrOCR, a state-of-the-art transformer-based HTR model, to 16th-century Latin manuscripts authored by Rudolf Gwalther. We investigate targeted image preprocessing and a broad suite of data augmentation techniques, introducing four novel augmentation methods designed specifically for historical handwriting characteristics. We also evaluate ensemble learning approaches to leverage the complementary strengths of augmentation-trained models. On the Gwalther dataset, our best single-model augmentation (Elastic) achieves a Character Error Rate (CER) of 1.86, while a top-5 voting ensemble achieves a CER of 1.60 - representing a 50% relative improvement over the best reported TrOCR_BASE result and a 42% improvement over the previous state of the art. These results highlight the impact of domain-specific augmentations and ensemble strategies in advancing HTR performance for historical manuscripts. 

**Abstract (ZH)**: 的历史手写文本识别（HTR）对于解锁档案文件的文化和学术价值至关重要，但由于缺乏转录、语言变异和高度多样的手写风格，数字化往往受到阻碍。本研究将最先进的变压器基线HTR模型TrOCR应用于鲁道夫·格瓦尔瑟撰写的16世纪拉丁手稿。我们研究了针对特定图像预处理方法和一系列广泛的数据增强技术，引入了四种专为历史手写特征设计的新型增强方法。我们还评估了集成学习方法以充分利用增强训练模型的互补优势。在格瓦尔瑟数据集中，我们最佳单一模型增强（Elastic）的字符错误率（CER）为1.86，而前五票投票集成则达到CER 1.60，分别比最佳报告的TrOCR_BASE结果提高了50%，比之前最先进的技术提高了42%。这些结果突显了领域特定增强和集成策略对提高历史手稿HTR性能的影响。 

---
# RMSL: Weakly-Supervised Insider Threat Detection with Robust Multi-sphere Learning 

**Title (ZH)**: RMSL：稳健多球学习驱动的弱监督内部威胁检测 

**Authors**: Yang Wang, Yaxin Zhao, Xinyu Jiao, Sihan Xu, Xiangrui Cai, Ying Zhang, Xiaojie Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2508.11472)  

**Abstract**: Insider threat detection aims to identify malicious user behavior by analyzing logs that record user interactions. Due to the lack of fine-grained behavior-level annotations, detecting specific behavior-level anomalies within user behavior sequences is challenging. Unsupervised methods face high false positive rates and miss rates due to the inherent ambiguity between normal and anomalous behaviors. In this work, we instead introduce weak labels of behavior sequences, which have lower annotation costs, i.e., the training labels (anomalous or normal) are at sequence-level instead of behavior-level, to enhance the detection capability for behavior-level anomalies by learning discriminative features. To achieve this, we propose a novel framework called Robust Multi-sphere Learning (RMSL). RMSL uses multiple hyper-spheres to represent the normal patterns of behaviors. Initially, a one-class classifier is constructed as a good anomaly-supervision-free starting point. Building on this, using multiple instance learning and adaptive behavior-level self-training debiasing based on model prediction confidence, the framework further refines hyper-spheres and feature representations using weak sequence-level labels. This approach enhances the model's ability to distinguish between normal and anomalous behaviors. Extensive experiments demonstrate that RMSL significantly improves the performance of behavior-level insider threat detection. 

**Abstract (ZH)**: 基于弱标签的稳健多球学习方法在用户行为序列中检测行为级异常以识别内部威胁 

---
# Informative Post-Hoc Explanations Only Exist for Simple Functions 

**Title (ZH)**: 仅存在简单的函数上的有说服力的后验解释。 

**Authors**: Eric Günther, Balázs Szabados, Robi Bhattacharjee, Sebastian Bordt, Ulrike von Luxburg  

**Link**: [PDF](https://arxiv.org/pdf/2508.11441)  

**Abstract**: Many researchers have suggested that local post-hoc explanation algorithms can be used to gain insights into the behavior of complex machine learning models. However, theoretical guarantees about such algorithms only exist for simple decision functions, and it is unclear whether and under which assumptions similar results might exist for complex models. In this paper, we introduce a general, learning-theory-based framework for what it means for an explanation to provide information about a decision function. We call an explanation informative if it serves to reduce the complexity of the space of plausible decision functions. With this approach, we show that many popular explanation algorithms are not informative when applied to complex decision functions, providing a rigorous mathematical rejection of the idea that it should be possible to explain any model. We then derive conditions under which different explanation algorithms become informative. These are often stronger than what one might expect. For example, gradient explanations and counterfactual explanations are non-informative with respect to the space of differentiable functions, and SHAP and anchor explanations are not informative with respect to the space of decision trees. Based on these results, we discuss how explanation algorithms can be modified to become informative. While the proposed analysis of explanation algorithms is mathematical, we argue that it holds strong implications for the practical applicability of these algorithms, particularly for auditing, regulation, and high-risk applications of AI. 

**Abstract (ZH)**: 一种基于学习理论的解释框架：关于决策函数信息性的新视角 

---
# Minimizing Surrogate Losses for Decision-Focused Learning using Differentiable Optimization 

**Title (ZH)**: 最小化决策导向学习的代理损失函数优化 

**Authors**: Jayanta Mandi, Ali İrfan Mahmutoğulları, Senne Berden, Tias Guns  

**Link**: [PDF](https://arxiv.org/pdf/2508.11365)  

**Abstract**: Decision-focused learning (DFL) trains a machine learning (ML) model to predict parameters of an optimization problem, to directly minimize decision regret, i.e., maximize decision quality. Gradient-based DFL requires computing the derivative of the solution to the optimization problem with respect to the predicted parameters. However, for many optimization problems, such as linear programs (LPs), the gradient of the regret with respect to the predicted parameters is zero almost everywhere. Existing gradient-based DFL approaches for LPs try to circumvent this issue in one of two ways: (a) smoothing the LP into a differentiable optimization problem by adding a quadratic regularizer and then minimizing the regret directly or (b) minimizing surrogate losses that have informative (sub)gradients. In this paper, we show that the former approach still results in zero gradients, because even after smoothing the regret remains constant across large regions of the parameter space. To address this, we propose minimizing surrogate losses -- even when a differentiable optimization layer is used and regret can be minimized directly. Our experiments demonstrate that minimizing surrogate losses allows differentiable optimization layers to achieve regret comparable to or better than surrogate-loss based DFL methods. Further, we demonstrate that this also holds for DYS-Net, a recently proposed differentiable optimization technique for LPs, that computes approximate solutions and gradients through operations that can be performed using feedforward neural network layers. Because DYS-Net executes the forward and the backward pass very efficiently, by minimizing surrogate losses using DYS-Net, we are able to attain regret on par with the state-of-the-art while reducing training time by a significant margin. 

**Abstract (ZH)**: 决策导向的学习（DFL）训练机器学习（ML）模型以预测优化问题的参数，直接最小化决策遗憾，即最大化决策质量。基于梯度的DFL需要计算优化问题的解关于预测参数的导数。然而，对于许多优化问题，如线性规划（LPs），遗憾关于预测参数的梯度几乎处处为零。现有基于梯度的DFL方法针对LPs试图通过两种方式之一来克服这一问题：（a）通过添加二次正则化项将LP平滑为可微优化问题，然后直接最小化遗憾，或（b）最小化具有信息性（次）梯度的替代损失函数。在这篇文章中，我们展示了第一种方法仍然会导致梯度为零，因为在平滑后，遗憾在参数空间的大量区域内仍然是恒定的。为此，我们提出即使使用可微优化层且可以直接最小化遗憾时，仍最小化替代损失函数。我们的实验表明，最小化替代损失函数使可微优化层能够实现与基于替代损失的DFL方法相当甚至更好的遗憾结果。此外，我们展示了这一点也适用于DYS-Net，这是一种最近提出的LP可微优化技术，通过可以使用前向神经网络层执行的操作计算近似解和梯度。由于DYS-Net非常高效地执行前向和反向传递，通过使用DYS-Net最小化替代损失函数，我们能够在大幅减少训练时间的同时达到与当前最佳方法相当的遗憾结果。 

---
# PTSM: Physiology-aware and Task-invariant Spatio-temporal Modeling for Cross-Subject EEG Decoding 

**Title (ZH)**: PTSM：生理aware且任务不变的时空建模用于跨被试EEG解码 

**Authors**: Changhong Jing, Yan Liu, Shuqiang Wang, Bruce X.B. Yu, Gong Chen, Zhejing Hu, Zhi Zhang, Yanyan Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.11357)  

**Abstract**: Cross-subject electroencephalography (EEG) decoding remains a fundamental challenge in brain-computer interface (BCI) research due to substantial inter-subject variability and the scarcity of subject-invariant representations. This paper proposed PTSM (Physiology-aware and Task-invariant Spatio-temporal Modeling), a novel framework for interpretable and robust EEG decoding across unseen subjects. PTSM employs a dual-branch masking mechanism that independently learns personalized and shared spatio-temporal patterns, enabling the model to preserve individual-specific neural characteristics while extracting task-relevant, population-shared features. The masks are factorized across temporal and spatial dimensions, allowing fine-grained modulation of dynamic EEG patterns with low computational overhead. To further address representational entanglement, PTSM enforces information-theoretic constraints that decompose latent embeddings into orthogonal task-related and subject-related subspaces. The model is trained end-to-end via a multi-objective loss integrating classification, contrastive, and disentanglement objectives. Extensive experiments on cross-subject motor imagery datasets demonstrate that PTSM achieves strong zero-shot generalization, outperforming state-of-the-art baselines without subject-specific calibration. Results highlight the efficacy of disentangled neural representations for achieving both personalized and transferable decoding in non-stationary neurophysiological settings. 

**Abstract (ZH)**: 生理aware和任务不变时空建模：跨受试者脑电图（EEG）解码的新框架 

---
# NeMo: A Neuron-Level Modularizing-While-Training Approach for Decomposing DNN Models 

**Title (ZH)**: NeMo: 一种在训练过程中按神经元模块化分解DNN模型的方法 

**Authors**: Xiaohan Bi, Binhang Qi, Hailong Sun, Xiang Gao, Yue Yu, Xiaojun Liang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11348)  

**Abstract**: With the growing incorporation of deep neural network (DNN) models into modern software systems, the prohibitive construction costs have become a significant challenge. Model reuse has been widely applied to reduce training costs, but indiscriminately reusing entire models may incur significant inference overhead. Consequently, DNN modularization has gained attention, enabling module reuse by decomposing DNN models. The emerging modularizing-while-training (MwT) paradigm, which incorporates modularization into training, outperforms modularizing-after-training approaches. However, existing MwT methods focus on small-scale CNN models at the convolutional kernel level and struggle with diverse DNNs and large-scale models, particularly Transformer-based models. To address these limitations, we propose NeMo, a scalable and generalizable MwT approach. NeMo operates at the neuron level fundamental component common to all DNNs-ensuring applicability to Transformers and various architectures. We design a contrastive learning-based modular training method with an effective composite loss function, enabling scalability to large-scale models. Comprehensive experiments on two Transformer-based models and four CNN models across two classification datasets demonstrate NeMo's superiority over state-of-the-art MwT methods. Results show average gains of 1.72% in module classification accuracy and 58.10% reduction in module size, demonstrating efficacy across both CNN and large-scale Transformer-based models. A case study on open-source projects shows NeMo's potential benefits in practical scenarios, offering a promising approach for scalable and generalizable DNN modularization. 

**Abstract (ZH)**: 随着深度神经网络（DNN）模型在现代软件系统中的广泛应用， prohibitive construction costs已成为重大挑战。模型重用已被广泛应用于降低训练成本，但随意地重用整个模型可能会引起显著的推理开销。因此，DNN模块化受到了关注，通过将DNN模型分解以实现模块重用。新兴的训练时模块化（MwT）范式，将模块化整合到训练过程中，优于训练后模块化方法。然而，现有MwT方法主要针对卷积核级别的小规模CNN模型，并且难以处理多样化的DNN和大规模模型，特别是Transformer模型。为了解决这些局限性，我们提出了NeMo，一种可扩展且通用的MwT方法。NeMo在所有DNNs中都具有通用性的神经元级别基本组件上操作，确保适用于Transformer和其他架构。我们设计了一种基于对比学习的模块化训练方法，并采用有效的复合损失函数，使其能够扩展到大规模模型。在两个Transformer模型和四个CNN模型上的两个分类数据集上的全面实验表明，NeMo在最先进的MwT方法中具有优势。结果显示，模块分类准确率平均提高1.72%，模块大小减少58.10%，显示出其在CNN和大规模Transformer模型中的有效性。开源项目案例研究显示，NeMo在实际场景中具有潜在益处，为可扩展且通用的DNN模块化提供了前景广阔的方法。 

---
# RegimeNAS: Regime-Aware Differentiable Architecture Search With Theoretical Guarantees for Financial Trading 

**Title (ZH)**: Regime-Aware Differentiable Architecture Search with Theoretical Guarantees for Financial Trading 

**Authors**: Prathamesh Devadiga, Yashmitha Shailesh  

**Link**: [PDF](https://arxiv.org/pdf/2508.11338)  

**Abstract**: We introduce RegimeNAS, a novel differentiable architecture search framework specifically designed to enhance cryptocurrency trading performance by explicitly integrating market regime awareness. Addressing the limitations of static deep learning models in highly dynamic financial environments, RegimeNAS features three core innovations: (1) a theoretically grounded Bayesian search space optimizing architectures with provable convergence properties; (2) specialized, dynamically activated neural modules (Volatility, Trend, and Range blocks) tailored for distinct market conditions; and (3) a multi-objective loss function incorporating market-specific penalties (e.g., volatility matching, transition smoothness) alongside mathematically enforced Lipschitz stability constraints. Regime identification leverages multi-head attention across multiple timeframes for improved accuracy and uncertainty estimation. Rigorous empirical evaluation on extensive real-world cryptocurrency data demonstrates that RegimeNAS significantly outperforms state-of-the-art benchmarks, achieving an 80.3% Mean Absolute Error reduction compared to the best traditional recurrent baseline and converging substantially faster (9 vs. 50+ epochs). Ablation studies and regime-specific analysis confirm the critical contribution of each component, particularly the regime-aware adaptation mechanism. This work underscores the imperative of embedding domain-specific knowledge, such as market regimes, directly within the NAS process to develop robust and adaptive models for challenging financial applications. 

**Abstract (ZH)**: 介绍RegimeTypeNAS框架，该框架通过引入一种带有理论依据的贝叶斯优化方法来优化架构，并具备证明性的收敛性质；特别设计了能够动态激活的神经模块（波动性趋势和范围模块），以适应不同的市场条件；以及引入多目标损失函数，同时考虑了市场特定的惩罚项（如：波动匹配匹配平滑度）并通过数学手段保证Lipschitz稳定性。RegiseumIdentification利用多头注意力机制在多个时间框架上进行交互以提高准确性和不确定性估计。严格的实证研究表明，相对于传统的循环序基线中模型，RegiseumNAS在现实世界中的加密货币数据上中显著优于最新的基准模型，平均绝对误差降低8.3个百分点，且收敛速度快得多（几轮 vs. 十数个时期）。进一步研究表明和市场特定分析证实了每部分成分的的作用、特别是市场意识适应机制的重要性。该研究强调了直接嵌入同期特定知识（如：市场周期条件）到架构搜索NAS过程中开发稳健和适应性强的模型的必要性。 

---
# SGSimEval: A Comprehensive Multifaceted and Similarity-Enhanced Benchmark for Automatic Survey Generation Systems 

**Title (ZH)**: SGSimEval: 一种全面的多维度且相似度增强的自动问卷生成系统基准测试 

**Authors**: Beichen Guo, Zhiyuan Wen, Yu Yang, Peng Gao, Ruosong Yang, Jiaxing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.11310)  

**Abstract**: The growing interest in automatic survey generation (ASG), a task that traditionally required considerable time and effort, has been spurred by recent advances in large language models (LLMs). With advancements in retrieval-augmented generation (RAG) and the rising popularity of multi-agent systems (MASs), synthesizing academic surveys using LLMs has become a viable approach, thereby elevating the need for robust evaluation methods in this domain. However, existing evaluation methods suffer from several limitations, including biased metrics, a lack of human preference, and an over-reliance on LLMs-as-judges. To address these challenges, we propose SGSimEval, a comprehensive benchmark for Survey Generation with Similarity-Enhanced Evaluation that evaluates automatic survey generation systems by integrating assessments of the outline, content, and references, and also combines LLM-based scoring with quantitative metrics to provide a multifaceted evaluation framework. In SGSimEval, we also introduce human preference metrics that emphasize both inherent quality and similarity to humans. Extensive experiments reveal that current ASG systems demonstrate human-comparable superiority in outline generation, while showing significant room for improvement in content and reference generation, and our evaluation metrics maintain strong consistency with human assessments. 

**Abstract (ZH)**: 自动调查生成中的相似性增强评估基准（基于自动调查生成系统的综合评估框架） 

---
# Is General-Purpose AI Reasoning Sensitive to Data-Induced Cognitive Biases? Dynamic Benchmarking on Typical Software Engineering Dilemmas 

**Title (ZH)**: 通用人工智能是否 数据诱导的认知偏差 是否具有理性敏感性？典型软件工程悖论的动态基准测试 

**Authors**: Francesco Sovrano, Gabriele Dominici, Rita Sevastjanova, Alessandra Stramiglio, Alberto Bacchelli  

**Link**: [PDF](https://arxiv.org/pdf/2508.11278)  

**Abstract**: Human cognitive biases in software engineering can lead to costly errors. While general-purpose AI (GPAI) systems may help mitigate these biases due to their non-human nature, their training on human-generated data raises a critical question: Do GPAI systems themselves exhibit cognitive biases?
To investigate this, we present the first dynamic benchmarking framework to evaluate data-induced cognitive biases in GPAI within software engineering workflows. Starting with a seed set of 16 hand-crafted realistic tasks, each featuring one of 8 cognitive biases (e.g., anchoring, framing) and corresponding unbiased variants, we test whether bias-inducing linguistic cues unrelated to task logic can lead GPAI systems from correct to incorrect conclusions.
To scale the benchmark and ensure realism, we develop an on-demand augmentation pipeline relying on GPAI systems to generate task variants that preserve bias-inducing cues while varying surface details. This pipeline ensures correctness (88--99% on average, according to human evaluation), promotes diversity, and controls reasoning complexity by leveraging Prolog-based reasoning and LLM-as-a-judge validation. It also verifies that the embedded biases are both harmful and undetectable by logic-based, unbiased reasoners.
We evaluate leading GPAI systems (GPT, LLaMA, DeepSeek) and find a consistent tendency to rely on shallow linguistic heuristics over deep reasoning. All systems exhibit cognitive biases (ranging from 5.9% to 35% across types), with bias sensitivity increasing sharply with task complexity (up to 49%), highlighting critical risks in real-world software engineering deployments. 

**Abstract (ZH)**: 软件工程中的人类认知偏差可能导致昂贵的错误。通用人工智能系统因其非人类特性可能帮助减轻这些偏差，但它们依赖于人类生成的数据进行训练，提出了一个关键问题：通用人工智能系统本身是否也会表现出认知偏差？

为探讨这一问题，我们提出了首个动态基准框架，用于评估数据诱发的认知偏差在软件工程工作流中的通用人工智能系统中。以16个手工构建的现实任务为种子集，每个任务包含8种认知偏差（例如，锚定偏差、框架效应）及其相应的无偏变体，我们测试无关任务逻辑的语言提示是否能够引导通用人工智能系统从正确结论转向错误结论。

为了扩大基准的规模并确保现实性，我们开发了一个按需扩充管道，依赖通用人工智能系统生成保留偏差诱导提示而表面细节各异的任务变体。该管道确保了正确性（根据人类评估平均为88%至99%）、促进了多样性和通过基于Prolog的推理和LLM作为裁判进行验证来控制推理复杂性。此外，该管道验证了嵌入的偏差既有害且逻辑无偏的推理器不可检测。

我们评估了领先的通用人工智能系统（GPT、LLaMA、DeepSeek），发现它们倾向于依赖浅层语言启发式而非深层推理的一致倾向。所有系统都表现出认知偏差（不同类型从5.9%到35%不等），任务复杂性增加时，认知偏差敏感性急剧上升（高达49%），突显了实际软件工程部署中的关键风险。 

---
# Graph Neural Diffusion via Generalized Opinion Dynamics 

**Title (ZH)**: 图神经扩散通过广义意见动力学 

**Authors**: Asela Hevapathige, Asiri Wijesinghe, Ahad N. Zehmakan  

**Link**: [PDF](https://arxiv.org/pdf/2508.11249)  

**Abstract**: There has been a growing interest in developing diffusion-based Graph Neural Networks (GNNs), building on the connections between message passing mechanisms in GNNs and physical diffusion processes. However, existing methods suffer from three critical limitations: (1) they rely on homogeneous diffusion with static dynamics, limiting adaptability to diverse graph structures; (2) their depth is constrained by computational overhead and diminishing interpretability; and (3) theoretical understanding of their convergence behavior remains limited. To address these challenges, we propose GODNF, a Generalized Opinion Dynamics Neural Framework, which unifies multiple opinion dynamics models into a principled, trainable diffusion mechanism. Our framework captures heterogeneous diffusion patterns and temporal dynamics via node-specific behavior modeling and dynamic neighborhood influence, while ensuring efficient and interpretable message propagation even at deep layers. We provide a rigorous theoretical analysis demonstrating GODNF's ability to model diverse convergence configurations. Extensive empirical evaluations of node classification and influence estimation tasks confirm GODNF's superiority over state-of-the-art GNNs. 

**Abstract (ZH)**: 基于意见动力学的通用扩散神经框架：统一多种意见动力学模型以克服图神经网络的三个关键限制 

---
# Cross-Granularity Hypergraph Retrieval-Augmented Generation for Multi-hop Question Answering 

**Title (ZH)**: 跨粒度超图检索增强生成用于多跳问答 

**Authors**: Changjian Wang, Weihong Deng, Weili Guan, Quan Lu, Ning Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11247)  

**Abstract**: Multi-hop question answering (MHQA) requires integrating knowledge scattered across multiple passages to derive the correct answer. Traditional retrieval-augmented generation (RAG) methods primarily focus on coarse-grained textual semantic similarity and ignore structural associations among dispersed knowledge, which limits their effectiveness in MHQA tasks. GraphRAG methods address this by leveraging knowledge graphs (KGs) to capture structural associations, but they tend to overly rely on structural information and fine-grained word- or phrase-level retrieval, resulting in an underutilization of textual semantics. In this paper, we propose a novel RAG approach called HGRAG for MHQA that achieves cross-granularity integration of structural and semantic information via hypergraphs. Structurally, we construct an entity hypergraph where fine-grained entities serve as nodes and coarse-grained passages as hyperedges, and establish knowledge association through shared entities. Semantically, we design a hypergraph retrieval method that integrates fine-grained entity similarity and coarse-grained passage similarity via hypergraph diffusion. Finally, we employ a retrieval enhancement module, which further refines the retrieved results both semantically and structurally, to obtain the most relevant passages as context for answer generation with the LLM. Experimental results on benchmark datasets demonstrate that our approach outperforms state-of-the-art methods in QA performance, and achieves a 6$\times$ speedup in retrieval efficiency. 

**Abstract (ZH)**: 基于超图的多跳问答（HGRAG）：结构与语义信息的跨粒度整合 

---
# How Causal Abstraction Underpins Computational Explanation 

**Title (ZH)**: 因果抽象如何支撑计算解释 

**Authors**: Atticus Geiger, Jacqueline Harding, Thomas Icard  

**Link**: [PDF](https://arxiv.org/pdf/2508.11214)  

**Abstract**: Explanations of cognitive behavior often appeal to computations over representations. What does it take for a system to implement a given computation over suitable representational vehicles within that system? We argue that the language of causality -- and specifically the theory of causal abstraction -- provides a fruitful lens on this topic. Drawing on current discussions in deep learning with artificial neural networks, we illustrate how classical themes in the philosophy of computation and cognition resurface in contemporary machine learning. We offer an account of computational implementation grounded in causal abstraction, and examine the role for representation in the resulting picture. We argue that these issues are most profitably explored in connection with generalization and prediction. 

**Abstract (ZH)**: 认知行为的解释通常依赖于对表示的计算。要使一个系统在一个系统中的适当表征载体上实现给定的计算，需要什么条件？我们认为，因果语言——特别是因果抽象的理论——为探讨这一问题提供了富有成果的观点。借鉴深度学习中人工神经网络的当前讨论，我们展示了计算和认知中经典主题如何在现代机器学习中重新出现。我们提供了一个基于因果抽象的计算实现的说明，并探讨了表征在所得图景中的作用。我们认为，这些议题与泛化和预测的关系最值得深入探讨。 

---
# Quantum-Boosted High-Fidelity Deep Learning 

**Title (ZH)**: 量子增强高保真深度学习 

**Authors**: Feng-ao Wang, Shaobo Chen, Yao Xuan, Junwei Liu, Qi Gao, Hongdong Zhu, Junjie Hou, Lixin Yuan, Jinyu Cheng, Chenxin Yi, Hai Wei, Yin Ma, Tao Xu, Kai Wen, Yixue Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.11190)  

**Abstract**: A fundamental limitation of probabilistic deep learning is its predominant reliance on Gaussian priors. This simplistic assumption prevents models from accurately capturing the complex, non-Gaussian landscapes of natural data, particularly in demanding domains like complex biological data, severely hindering the fidelity of the model for scientific discovery. The physically-grounded Boltzmann distribution offers a more expressive alternative, but it is computationally intractable on classical computers. To date, quantum approaches have been hampered by the insufficient qubit scale and operational stability required for the iterative demands of deep learning. Here, we bridge this gap by introducing the Quantum Boltzmann Machine-Variational Autoencoder (QBM-VAE), a large-scale and long-time stable hybrid quantum-classical architecture. Our framework leverages a quantum processor for efficient sampling from the Boltzmann distribution, enabling its use as a powerful prior within a deep generative model. Applied to million-scale single-cell datasets from multiple sources, the QBM-VAE generates a latent space that better preserves complex biological structures, consistently outperforming conventional Gaussian-based deep learning models like VAE and SCVI in essential tasks such as omics data integration, cell-type classification, and trajectory inference. It also provides a typical example of introducing a physics priori into deep learning to drive the model to acquire scientific discovery capabilities that breaks through data limitations. This work provides the demonstration of a practical quantum advantage in deep learning on a large-scale scientific problem and offers a transferable blueprint for developing hybrid quantum AI models. 

**Abstract (ZH)**: 一种概率深度学习的基本局限是其主要依赖于高斯先验。这种简单的假设使得模型难以准确捕捉自然数据中复杂的、非高斯的空间特征，尤其是在复杂的生物数据等挑战性领域中，严重影响了模型的科学发现能力。基于物理原理的玻尔兹曼分布提供了更具表达力的替代方案，但其在经典计算机上的计算不可行。迄今为止，量子方法受限于无法满足深度学习迭代需求所需的足够量子位规模和操作稳定性。在这里，我们通过引入量子玻尔兹曼机-变分自编码器（QBM-VAE）来弥补这一差距，这是一种大规模且长时间稳定的量子-经典混合架构。我们的框架利用量子处理器进行玻尔兹曼分布高效采样，使它可以作为深度生成模型中的强大先验。应用于来自多个来源的百万规模单细胞数据集，QBM-VAE 生成的隐空间更好地保留了复杂的生物结构，在诸如组学数据整合、细胞类型分类和轨迹推断等关键任务中，始终优于传统的基于高斯的深度学习模型（如 VAE 和 SCVI）。此外，它还提供了一个将物理先验引入深度学习以驱动模型获得突破数据限制的科学发现能力的典型示例。这项工作展示了在大规模科学问题上实现实用的量子优势，并为开发混合量子人工智能模型提供了可转移的设计蓝图。 

---
# A Semi-supervised Generative Model for Incomplete Multi-view Data Integration with Missing Labels 

**Title (ZH)**: 一种半监督生成模型用于 incomplete 多-view 数据集成的缺失标签处理 

**Authors**: Yiyang Shen, Weiran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11180)  

**Abstract**: Multi-view learning is widely applied to real-life datasets, such as multiple omics biological data, but it often suffers from both missing views and missing labels. Prior probabilistic approaches addressed the missing view problem by using a product-of-experts scheme to aggregate representations from present views and achieved superior performance over deterministic classifiers, using the information bottleneck (IB) principle. However, the IB framework is inherently fully supervised and cannot leverage unlabeled data. In this work, we propose a semi-supervised generative model that utilizes both labeled and unlabeled samples in a unified framework. Our method maximizes the likelihood of unlabeled samples to learn a latent space shared with the IB on labeled data. We also perform cross-view mutual information maximization in the latent space to enhance the extraction of shared information across views. Compared to existing approaches, our model achieves better predictive and imputation performance on both image and multi-omics data with missing views and limited labeled samples. 

**Abstract (ZH)**: 多视图学习广泛应用于实际生活数据集，如多组学生物数据，但 often 患有缺失视图和缺失标签的问题。先前的概率方法通过使用专家产品的方案聚合现有视图的表示，并通过信息瓶颈原则优于确定性分类器解决了缺失视图问题，取得了更好的性能。然而，信息瓶颈框架本质上是全监督的，不能利用未标记的数据。在本文中，我们提出了一种半监督生成模型，该模型在统一框架中使用标记和未标记样本。我们的方法通过最大化未标记样本的似然性，在共享的潜在空间中学习IB机制。我们还在潜在空间中进行跨视图互信息最大化，以增强视图间共享信息的提取。与现有方法相比，我们的模型在具有缺失视图和有限标记样本的图像和多组学数据上实现了更好的预测和缺失值填充性能。 

---
# Better Supervised Fine-tuning for VQA: Integer-Only Loss 

**Title (ZH)**: 更好的监督微调方法：仅整数损失的VQA 

**Authors**: Baihong Qian, Haotian Fan, Wenjie Liao, Yunqiu Wang, Tao Li, Junhui Cui  

**Link**: [PDF](https://arxiv.org/pdf/2508.11170)  

**Abstract**: With the rapid advancement of vision language models(VLM), their ability to assess visual content based on specific criteria and dimensions has become increasingly critical for applications such as video-theme consistency assessment and visual quality scoring. However, existing methods often suffer from imprecise results and inefficient loss calculation, which limit the focus of the model on key evaluation indicators. To address this, we propose IOVQA(Integer-only VQA), a novel fine-tuning approach tailored for VLMs to enhance their performance in video quality assessment tasks. The key innovation of IOVQA lies in its label construction and its targeted loss calculation mechanism. Specifically, during dataset curation, we constrain the model's output to integers within the range of [10,50], ensuring numerical stability, and convert decimal Overall_MOS to integer before using them as labels. We also introduce a target-mask strategy: when computing the loss, only the first two-digit-integer of the label is unmasked, forcing the model to learn the critical components of the numerical evaluation. After fine-tuning the Qwen2.5-VL model using the constructed dataset, experimental results demonstrate that the proposed method significantly improves the model's accuracy and consistency in the VQA task, ranking 3rd in VQualA 2025 GenAI-Bench AIGC Video Quality Assessment Challenge -- Track I. Our work highlights the effectiveness of merely leaving integer labels during fine-tuning, providing an effective idea for optimizing VLMs in quantitative evaluation scenarios. 

**Abstract (ZH)**: 基于整数的视觉问答方法（IOVQA）：一种针对视觉语言模型的细调方法以提升视频质量评估性能 

---
# Tabularis Formatus: Predictive Formatting for Tables 

**Title (ZH)**: Tabularis Formatus: 表格的预测性格式化 

**Authors**: Mukul Singh, José Cambronero, Sumit Gulwani, Vu Le, Gust Verbruggen  

**Link**: [PDF](https://arxiv.org/pdf/2508.11121)  

**Abstract**: Spreadsheet manipulation software are widely used for data management and analysis of tabular data, yet the creation of conditional formatting (CF) rules remains a complex task requiring technical knowledge and experience with specific platforms. In this paper we present TaFo, a neuro-symbolic approach to generating CF suggestions for tables, addressing common challenges such as user unawareness, difficulty in rule creation, and inadequate user interfaces. TaFo takes inspiration from component based synthesis systems and extends them with semantic knowledge of language models and a diversity preserving rule this http URL previous methods focused on structural formatting, TaFo uniquely incorporates value-based formatting, automatically learning both the rule trigger and the associated visual formatting properties for CF rules. By removing the dependency on user specification used by existing techniques in the form of formatted examples or natural language instruction, TaFo makes formatting completely predictive and automated for the user. To evaluate TaFo, we use a corpus of 1.8 Million public workbooks with CF and manual formatting. We compare TaFo against a diverse set of symbolic and neural systems designed for or adapted for the task of table formatting. Our results show that TaFo generates more accurate, diverse and complete formatting suggestions than current systems and outperforms these by 15.6\%--26.5\% on matching user added ground truth rules in tables. 

**Abstract (ZH)**: 基于神经符号方法的表格条件格式建议生成系统：TaFo 

---
# Quantization through Piecewise-Affine Regularization: Optimization and Statistical Guarantees 

**Title (ZH)**: 分段线性正则化下的量化：优化与统计保证 

**Authors**: Jianhao Ma, Lin Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.11112)  

**Abstract**: Optimization problems over discrete or quantized variables are very challenging in general due to the combinatorial nature of their search space. Piecewise-affine regularization (PAR) provides a flexible modeling and computational framework for quantization based on continuous optimization. In this work, we focus on the setting of supervised learning and investigate the theoretical foundations of PAR from optimization and statistical perspectives. First, we show that in the overparameterized regime, where the number of parameters exceeds the number of samples, every critical point of the PAR-regularized loss function exhibits a high degree of quantization. Second, we derive closed-form proximal mappings for various (convex, quasi-convex, and non-convex) PARs and show how to solve PAR-regularized problems using the proximal gradient method, its accelerated variant, and the Alternating Direction Method of Multipliers. Third, we study statistical guarantees of PAR-regularized linear regression problems; specifically, we can approximate classical formulations of $\ell_1$-, squared $\ell_2$-, and nonconvex regularizations using PAR and obtain similar statistical guarantees with quantized solutions. 

**Abstract (ZH)**: 离散或量化变量上的优化问题由于其搜索空间的组合性质通常具有很大的挑战性。分段线性正则化（PAR）为基于连续优化的量化提供了一种灵活的建模和计算框架。在本文中，我们专注于监督学习的设置，并从优化和统计的角度研究PAR的理论基础。首先，我们证明在参数过参数化的情况下，即参数数量超过样本数量时，PAR正则化损失函数的每个临界点都表现出高度的量化。其次，我们推导出各种（凸的、拟凸的和非凸的）PAR的闭形式近邻映射，并展示如何使用近邻梯度方法、其加速变体以及交替方向乘子法来求解PAR正则化问题。第三，我们研究PAR正则化线性回归问题的统计保证；具体而言，我们可以使用PAR近似经典的$\ell_1$、平方$\ell_2$和非凸正则化，并获得与量化解决方案类似的统计保证。 

---
# Diffusion is a code repair operator and generator 

**Title (ZH)**: 扩散是一种代码修复操作符和生成器。 

**Authors**: Mukul Singh, Gust Verbruggen, Vu Le, Sumit Gulwani  

**Link**: [PDF](https://arxiv.org/pdf/2508.11110)  

**Abstract**: Code diffusion models generate code by iteratively removing noise from the latent representation of a code snippet. During later steps of the diffusion process, when the code snippet has almost converged, differences between discrete representations of these snippets look like last-mile repairs applied to broken or incomplete code. We evaluate the extent to which this resemblance can be exploited to leverage pre-trained code diffusion models for the problem of last-mile repair by considering two applications with significant potential. First, we can leverage the diffusion model for last-mile repair by adding noise to a broken code snippet and resuming the diffusion process. Second, we can leverage the diffusion model to generate arbitrary amount of training data for last-mile repair tasks (that are computationally more efficient) by sampling an intermediate program (input) and the final program (output) from the diffusion process. We perform experiments on 3 domains (Python, Excel and PowerShell) to evaluate applications, as well as analyze properties. 

**Abstract (ZH)**: 代码扩散模型通过迭代去除代码片段潜在表示中的噪音来生成代码。在扩散过程的后期步骤中，当代码片段几乎收敛时，这些片段的离散表示之间的差异看起来像是对损坏或不完整代码所做的最后一英里修复。我们通过考虑两种具有重大潜力的应用来评估这种相似性可以被利用的程度。首先，可以通过向损坏的代码片段添加噪音并继续扩散过程来利用扩散模型进行最后一英里的修复。其次，可以通过从扩散过程中采样中间程序（输入）和最终程序（输出）来利用扩散模型生成任意数量的最后一英里修复任务的训练数据（这些任务在计算上更高效）。我们在3个领域（Python、Excel和PowerShell）上进行了实验，以评估应用并分析其属性。 

---
# Compressive Meta-Learning 

**Title (ZH)**: 压缩元学习 

**Authors**: Daniel Mas Montserrat, David Bonet, Maria Perera, Xavier Giró-i-Nieto, Alexander G. Ioannidis  

**Link**: [PDF](https://arxiv.org/pdf/2508.11090)  

**Abstract**: The rapid expansion in the size of new datasets has created a need for fast and efficient parameter-learning techniques. Compressive learning is a framework that enables efficient processing by using random, non-linear features to project large-scale databases onto compact, information-preserving representations whose dimensionality is independent of the number of samples and can be easily stored, transferred, and processed. These database-level summaries are then used to decode parameters of interest from the underlying data distribution without requiring access to the original samples, offering an efficient and privacy-friendly learning framework. However, both the encoding and decoding techniques are typically randomized and data-independent, failing to exploit the underlying structure of the data. In this work, we propose a framework that meta-learns both the encoding and decoding stages of compressive learning methods by using neural networks that provide faster and more accurate systems than the current state-of-the-art approaches. To demonstrate the potential of the presented Compressive Meta-Learning framework, we explore multiple applications -- including neural network-based compressive PCA, compressive ridge regression, compressive k-means, and autoencoders. 

**Abstract (ZH)**: 新的数据集规模迅速扩张催生了快速高效参数学习技术的需求。压缩学习是一种框架，通过使用随机的非线性特征将大规模数据库投影到维度与样本数量无关且可轻松存储、传输和处理的信息保留表示上。这些数据库级别的摘要随后用于从潜在的数据分布中解码感兴趣的参数，而无需访问原始样本，从而提供了一个高效且隐私友好的学习框架。然而，编码和解码技术通常随机化且与数据独立，未能利用数据的潜在结构。在本文中，我们提出了一种框架，通过使用神经网络来元学习压缩学习方法的编码和解码阶段，从而比当前最先进的方法提供更快更准确的系统。为了展示所提出的压缩元学习框架的潜力，我们探索了多个应用，包括基于神经网络的压缩PCA、压缩岭回归、压缩K-means和自编码器。 

---
# Learning with Confidence 

**Title (ZH)**: 具有信心的學習 

**Authors**: Oliver Ethan Richardson  

**Link**: [PDF](https://arxiv.org/pdf/2508.11037)  

**Abstract**: We characterize a notion of confidence that arises in learning or updating beliefs: the amount of trust one has in incoming information and its impact on the belief state. This learner's confidence can be used alongside (and is easily mistaken for) probability or likelihood, but it is fundamentally a different concept -- one that captures many familiar concepts in the literature, including learning rates and number of training epochs, Shafer's weight of evidence, and Kalman gain. We formally axiomatize what it means to learn with confidence, give two canonical ways of measuring confidence on a continuum, and prove that confidence can always be represented in this way. Under additional assumptions, we derive more compact representations of confidence-based learning in terms of vector fields and loss functions. These representations induce an extended language of compound "parallel" observations. We characterize Bayes Rule as the special case of an optimizing learner whose loss representation is a linear expectation. 

**Abstract (ZH)**: 我们刻画一种在学习或更新信念中出现的信心概念：对 incoming 信息的信任程度及其对信念状态的影响。这种学习者的信心可以与概率或似然性一起使用（并且容易被混淆），但本质上是不同的概念——它涵盖了文献中许多熟悉的概念，包括学习速率、训练周期数、Shafers 的证据权重以及卡尔曼增益。我们形式化地公理化了信心学习的意义，给出了信心在连续统一体上度量的两种典型方式，并证明信心总可以如此表示。在附加假设下，我们推导出基于信心学习的更紧凑表示，这些表示诱导了一个扩展的“并行”观测语言。我们将贝叶斯规则视为使损失表示为线性期望的优化学习者的特殊情况。 

---
# Note on Selection Bias in Observational Estimates of Algorithmic Progress 

**Title (ZH)**: 关于观察估计算法进展中的选择偏差笔记 

**Authors**: Parker Whitfill  

**Link**: [PDF](https://arxiv.org/pdf/2508.11033)  

**Abstract**: Ho et. al (2024) is an interesting paper that attempts to estimate the degree of algorithmic progress from language models. They collect observational data on language models' loss and compute over time, and argue that as time has passed, language models' algorithmic efficiency has been rising. That is, the loss achieved for fixed compute has been dropping over time. In this note, I want to raise one potential methodological problem with the estimation strategy. Intuitively, if part of algorithmic quality is latent, and compute choices are endogenous to algorithmic quality, then resulting estimates of algorithmic quality will be biased. 

**Abstract (ZH)**: Ho等（2024）：算法进步度估算的潜在方法论问题 

---
# Risk-Based Prognostics and Health Management 

**Title (ZH)**: 基于风险的预测性维护与健康管理 

**Authors**: John W. Sheppard  

**Link**: [PDF](https://arxiv.org/pdf/2508.11031)  

**Abstract**: It is often the case that risk assessment and prognostics are viewed as related but separate tasks. This chapter describes a risk-based approach to prognostics that seeks to provide a tighter coupling between risk assessment and fault prediction. We show how this can be achieved using the continuous-time Bayesian network as the underlying modeling framework. Furthermore, we provide an overview of the techniques that are available to derive these models from data and show how they might be used in practice to achieve tasks like decision support and performance-based logistics. This work is intended to provide an overview of the recent developments related to risk-based prognostics, and we hope that it will serve as a tutorial of sorts that will assist others in adopting these techniques. 

**Abstract (ZH)**: 风险评估导向的 prognostics 方法：风险评估与故障预测的紧密耦合及其在决策支持和绩效物流中的应用 

---
# Zono-Conformal Prediction: Zonotope-Based Uncertainty Quantification for Regression and Classification Tasks 

**Title (ZH)**: zonotope- conformal预测：基于zonotope的回归和分类任务中的不确定性量化 

**Authors**: Laura Lützow, Michael Eichelbeck, Mykel J. Kochenderfer, Matthias Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2508.11025)  

**Abstract**: Conformal prediction is a popular uncertainty quantification method that augments a base predictor with prediction sets with statistically valid coverage guarantees. However, current methods are often computationally expensive and data-intensive, as they require constructing an uncertainty model before calibration. Moreover, existing approaches typically represent the prediction sets with intervals, which limits their ability to capture dependencies in multi-dimensional outputs. We address these limitations by introducing zono-conformal prediction, a novel approach inspired by interval predictor models and reachset-conformant identification that constructs prediction zonotopes with assured coverage. By placing zonotopic uncertainty sets directly into the model of the base predictor, zono-conformal predictors can be identified via a single, data-efficient linear program. While we can apply zono-conformal prediction to arbitrary nonlinear base predictors, we focus on feed-forward neural networks in this work. Aside from regression tasks, we also construct optimal zono-conformal predictors in classification settings where the output of an uncertain predictor is a set of possible classes. We provide probabilistic coverage guarantees and present methods for detecting outliers in the identification data. In extensive numerical experiments, we show that zono-conformal predictors are less conservative than interval predictor models and standard conformal prediction methods, while achieving a similar coverage over the test data. 

**Abstract (ZH)**: 自适应预测是一种流行的方法，用于扩展基础预测器以提供具有统计上 合有效覆盖保证的预测集。尽管当前方法通常计算密集且耗时，而且需要构建预测误差的校准过程。此外，现有的方法通常用使用区间来表示预测集，这 限制了对高维度输出中依赖性的捕捉能力。我们通过引入 zono-conformal 预测解决了这些限制，这是一种受基于区间预测模型和区域识别方法启发的方法。通过将 zon 型型不确定性集直接集成到基础预测器模型中，可 zono-con former 预测器可以通过单个的数据高效线性程序来识别。虽然我们可以将 zono-con caled 预测应用于任意非线性基础预测器上在本文中我们专注于将之应用于前馈神经网络。除了在现有的任务中我们还在分类设置中构建制最优的 zono-con 得预测器其中输出是一个可能的分类。我们给出了概率性的覆盖保证并 幯示了检测识别中的异常值的方法。在广泛的数值实验中我们表明 zono-con 络预测器比更保守于基于区间预测模型和 标准的预测误差方法 on 在优点是具有相近的覆盖能力 on on。 

---
# Match & Choose: Model Selection Framework for Fine-tuning Text-to-Image Diffusion Models 

**Title (ZH)**: 匹配与选择：文本到图像扩散模型微调的模型选择框架 

**Authors**: Basile Lewandowski, Robert Birke, Lydia Y. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10993)  

**Abstract**: Text-to-image (T2I) models based on diffusion and transformer architectures advance rapidly. They are often pretrained on large corpora, and openly shared on a model platform, such as HuggingFace. Users can then build up AI applications, e.g., generating media contents, by adopting pretrained T2I models and fine-tuning them on the target dataset. While public pretrained T2I models facilitate the democratization of the models, users face a new challenge: which model can be best fine-tuned based on the target data domain? Model selection is well addressed in classification tasks, but little is known in (pretrained) T2I models and their performance indication on the target domain. In this paper, we propose the first model selection framework, M&C, which enables users to efficiently choose a pretrained T2I model from a model platform without exhaustively fine-tuning them all on the target dataset. The core of M&C is a matching graph, which consists of: (i) nodes of available models and profiled datasets, and (ii) edges of model-data and data-data pairs capturing the fine-tuning performance and data similarity, respectively. We then build a model that, based on the inputs of model/data feature, and, critically, the graph embedding feature, extracted from the matching graph, predicts the model achieving the best quality after fine-tuning for the target domain. We evaluate M&C on choosing across ten T2I models for 32 datasets against three baselines. Our results show that M&C successfully predicts the best model for fine-tuning in 61.3% of the cases and a closely performing model for the rest. 

**Abstract (ZH)**: 基于扩散和变换器架构的文本到图像（T2I）模型快速发展。它们通常在大型语料库上进行预训练，并且在诸如HuggingFace的模型平台上公开共享。用户可以采用这些预训练的T2I模型并在目标数据集上进行微调来构建AI应用程序，例如生成媒体内容。虽然公开的预训练T2I模型促进了模型的民主化使用，但用户面临新的挑战：在目标数据域中哪些模型最适合进行微调？在分类任务中，模型选择问题得到了很好的解决，但在（预训练的）T2I模型及其在目标域上的性能指示方面还知之甚少。在本文中，我们提出了第一个模型选择框架M&C，该框架使用户能够在不全部在目标数据集上进行耗时微调的情况下，从模型平台上高效地选择一个预训练的T2I模型。M&C的核心是一个匹配图，该图包括：（i）可用模型和配置数据集的节点，以及（ii）模型-数据和数据-数据配对边，分别捕捉微调性能和数据相似性。然后构建一个模型，基于输入的模型/数据特征以及从匹配图中提取的图嵌入特征，预测在目标域上微调后的最佳性能模型。我们针对十个T2I模型和32个数据集与三个基线进行选择评估。结果表明，M&C在61.3%的情况下成功预测了最适合微调的模型，而对于其余情况则预测了表现相近的模型。 

---
# Retro-Expert: Collaborative Reasoning for Interpretable Retrosynthesis 

**Title (ZH)**: Retro-Expert: 共享推理实现可解释逆合成反应 

**Authors**: Xinyi Li, Sai Wang, Yutian Lin, Yu Wu, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10967)  

**Abstract**: Retrosynthesis prediction aims to infer the reactant molecule based on a given product molecule, which is a fundamental task in chemical synthesis. However, existing models rely on static pattern-matching paradigm, which limits their ability to perform effective logic decision-making, leading to black-box decision-making. Building on this, we propose Retro-Expert, an interpretable retrosynthesis framework that performs collaborative reasoning by combining the complementary reasoning strengths of Large Language Models and specialized models via reinforcement learning. It outputs natural language explanations grounded in chemical logic through three components: (1) specialized models perform shallow reasoning to construct high-quality chemical decision space, (2) LLM-driven critical reasoning to generate predictions and corresponding interpretable reasoning path, and (3) reinforcement learning optimizing interpretable decision policy. Experiments show that Retro-Expert not only surpasses both LLM-based and specialized models across different metrics but also provides expert-aligned explanations that bridge the gap between AI predictions and actionable chemical insights. 

**Abstract (ZH)**: retrosynthesis 预测旨在根据给定的目标分子推断反应物分子，这是化学合成中的一个基本任务。然而，现有的模型依赖于静态模式匹配范式，这限制了它们进行有效的逻辑决策能力，导致黑盒决策。在此基础上，我们提出了一种可解释的 retrosynthesis 框架 Retro-Expert，通过结合大型语言模型和专门模型的互补推理优势，并利用强化学习进行协作推理。该框架通过三个组成部分输出基于化学逻辑的自然语言解释：（1）专门模型执行浅层推理以构建高质量的化学决策空间，（2）由大型语言模型驱动的关键推理以生成预测及其相应的可解释推理路径，（3）利用强化学习优化可解释的决策策略。实验证明，Retro-Expert 不仅在不同指标上超越了基于大型语言模型和专门模型，还提供了与专家一致的解释，填补了人工智能预测与可操作化学洞察之间的差距。 

---
# Towards Efficient Prompt-based Continual Learning in Distributed Medical AI 

**Title (ZH)**: 基于提示的分布式医疗AI高效连续学习研究 

**Authors**: Gyutae Oh, Jitae Shin  

**Link**: [PDF](https://arxiv.org/pdf/2508.10954)  

**Abstract**: Modern AI models achieve state-of-the-art performance with large-scale, high-quality datasets; however, ethical, social, and institutional constraints in the medical domain severely restrict data sharing, rendering centralized learning nearly impossible. Each institution must incrementally update models using only local data. Traditional training overfits new samples and suffers from catastrophic forgetting, losing previously acquired knowledge. Medical data distributions also shift due to varying diagnostic equipment and demographics. Although continual learning (CL) has advanced, most methods address natural images, leaving medical-domain-specific CL underexplored. We propose a prompt-based continual learning (PCL) approach featuring a unified prompt pool with a minimal expansion strategy: by expanding and freezing a subset of prompts, our method reduces computational overhead, and a novel regularization term balances retention and adaptation. Experiments on three diabetic retinopathy datasets Aptos2019, LI2019, and Diabetic Retinopathy Detection show our model improves final classification accuracy by at least 10% and F1-score by 9 points over state-of-the-art approaches while lowering inference cost. We anticipate this study will drive sustainable medical AI advances, enabling real-time diagnosis, patient monitoring, and telemedicine applications in distributed healthcare. Code will be released upon acceptance 

**Abstract (ZH)**: 基于提示的持续学习在医疗领域的研究：糖尿病视网膜病变数据集上的应用 

---
# Modeling and Detecting Company Risks from News: A Case Study in Bloomberg News 

**Title (ZH)**: 基于 Bloomberg 新闻的公司风险建模与检测：案例研究넣﹁
user
A Multi-Task Learning Framework for Server Failure Prediction: A Case Study on Alibaba Cloud。保持原句式， the titleeline。 

**Authors**: Jiaxin Pei, Soumya Vadlamannati, Liang-Kang Huang, Daniel Preotiuc-Pietro, Xinyu Hua  

**Link**: [PDF](https://arxiv.org/pdf/2508.10927)  

**Abstract**: Identifying risks associated with a company is important to investors and the well-being of the overall financial market. In this study, we build a computational framework to automatically extract company risk factors from news articles. Our newly proposed schema comprises seven distinct aspects, such as supply chain, regulations, and competitions. We sample and annotate 744 news articles and benchmark various machine learning models. While large language models have achieved huge progress in various types of NLP tasks, our experiment shows that zero-shot and few-shot prompting state-of-the-art LLMs (e.g. LLaMA-2) can only achieve moderate to low performances in identifying risk factors. And fine-tuned pre-trained language models are performing better on most of the risk factors. Using this model, we analyze over 277K Bloomberg news articles and demonstrate that identifying risk factors from news could provide extensive insight into the operations of companies and industries. 

**Abstract (ZH)**: 识别与公司相关的风险对于投资者和整体金融市场福祉至关重要。本研究构建了一个计算框架，以自动从新闻文章中提取公司风险因素。我们提出的新方案包括七个不同的方面，如供应链、监管和竞争等。我们采样并标注了744篇新闻文章，并比较了多种机器学习模型。尽管大型语言模型在各种NLP任务中取得了巨大进展，但我们的实验显示，零样本和少样本提示最新的大规模语言模型（如LLaMA-2）在识别风险因素方面的表现仅达到中等到较低的水平。而微调的预训练语言模型在多数风险因素上的表现更好。通过该模型，我们分析了超过27.7万篇彭博新闻文章，并证明从新闻中识别风险因素能够为公司和行业的运营提供广泛洞见。 

---
# Managing the unexpected: Operator behavioural data and its value in predicting correct alarm responses 

**Title (ZH)**: 管理意外情况：操作员行为数据及其在预测正确报警响应中的价值 

**Authors**: Chidera W. Amazu, Joseph Mietkiewicz, Ammar N. Abbas, Gabriele Baldissone, Davide Fissore, Micaela Demichela, Anders L. Madsen, Maria Chiara Leva  

**Link**: [PDF](https://arxiv.org/pdf/2508.10917)  

**Abstract**: Data from psychophysiological measures can offer new insight into control room operators' behaviour, cognition, and mental workload status. This can be particularly helpful when combined with appraisal of capacity to respond to possible critical plant conditions (i.e. critical alarms response scenarios). However, wearable physiological measurement tools such as eye tracking and EEG caps can be perceived as intrusive and not suitable for usage in daily operations. Therefore, this article examines the potential of using real-time data from process and operator-system interactions during abnormal scenarios that can be recorded and retrieved from the distributed control system's historian or process log, and their capacity to provide insight into operator behavior and predict their response outcomes, without intruding on daily tasks. Data for this study were obtained from a design of experiment using a formaldehyde production plant simulator and four human-in-the-loop experimental support configurations. A comparison between the different configurations in terms of both behaviour and performance is presented in this paper. A step-wise logistic regression and a Bayesian network models were used to achieve this objective. The results identified some predictive metrics and the paper discuss their value as precursor or predictor of overall system performance in alarm response scenarios. Knowledge of relevant and predictive behavioural metrics accessible in real time can better equip decision-makers to predict outcomes and provide timely support measures for operators. 

**Abstract (ZH)**: 基于异常场景中操作员-系统交互实时数据的生理psychophysiological测量在报警响应中的潜力及预测价值 

---
# SDSNN: A Single-Timestep Spiking Neural Network with Self-Dropping Neuron and Bayesian Optimization 

**Title (ZH)**: SDSN: 基于自消除神经元和贝叶斯优化的一时时间发放神经网络 

**Authors**: Changqing Xu, Buxuan Song, Yi Liu, Xinfang Liao, Wenbin Zheng, Yintang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10913)  

**Abstract**: Spiking Neural Networks (SNNs), as an emerging biologically inspired computational model, demonstrate significant energy efficiency advantages due to their event-driven information processing mechanism. Compared to traditional Artificial Neural Networks (ANNs), SNNs transmit information through discrete spike signals, which substantially reduces computational energy consumption through their sparse encoding approach. However, the multi-timestep computation model significantly increases inference latency and energy, limiting the applicability of SNNs in edge computing scenarios. We propose a single-timestep SNN, which enhances accuracy and reduces computational energy consumption in a single timestep by optimizing spike generation and temporal parameters. We design a Self-Dropping Neuron mechanism, which enhances information-carrying capacity through dynamic threshold adjustment and selective spike suppression. Furthermore, we employ Bayesian optimization to globally search for time parameters and obtain an efficient inference mode with a single time step. Experimental results on the Fashion-MNIST, CIFAR-10, and CIFAR-100 datasets demonstrate that, compared to traditional multi-timestep SNNs employing the Leaky Integrate-and-Fire (LIF) model, our method achieves classification accuracies of 93.72%, 92.20%, and 69.45%, respectively, using only single-timestep spikes, while maintaining comparable or even superior accuracy. Additionally, it reduces energy consumption by 56%, 21%, and 22%, respectively. 

**Abstract (ZH)**: 基于单时步模型的自适应阈值神经元刺激发SNNs 

---
# Generalized Similarity U: A Non-parametric Test of Association Based on Similarity 

**Title (ZH)**: 广义相似性U：基于相似性的非参数关联检验 

**Authors**: Changshuai Wei, Qing Lu  

**Link**: [PDF](https://arxiv.org/pdf/1801.01220)  

**Abstract**: Second generation sequencing technologies are being increasingly used for genetic association studies, where the main research interest is to identify sets of genetic variants that contribute to various phenotype. The phenotype can be univariate disease status, multivariate responses and even high-dimensional outcomes. Considering the genotype and phenotype as two complex objects, this also poses a general statistical problem of testing association between complex objects. We here proposed a similarity-based test, generalized similarity U (GSU), that can test the association between complex objects. We first studied the theoretical properties of the test in a general setting and then focused on the application of the test to sequencing association studies. Based on theoretical analysis, we proposed to use Laplacian kernel based similarity for GSU to boost power and enhance robustness. Through simulation, we found that GSU did have advantages over existing methods in terms of power and robustness. We further performed a whole genome sequencing (WGS) scan for Alzherimer Disease Neuroimaging Initiative (ADNI) data, identifying three genes, APOE, APOC1 and TOMM40, associated with imaging phenotype. We developed a C++ package for analysis of whole genome sequencing data using GSU. The source codes can be downloaded at this https URL. 

**Abstract (ZH)**: 二代测序技术在基因关联研究中的应用：基于相似性的测试方法GSU及其在阿尔茨海默病神经影像学倡议数据中的应用 

---
# Trees Assembling Mann Whitney Approach for Detecting Genome-wide Joint Association among Low Marginal Effect loci 

**Title (ZH)**: 树木组装曼尼 Whitney 方法检测低边际效应位点的全基因组联合关联 

**Authors**: Changshuai Wei, Daniel J. Schaid, Qing Lu  

**Link**: [PDF](https://arxiv.org/pdf/1505.01206)  

**Abstract**: Common complex diseases are likely influenced by the interplay of hundreds, or even thousands, of genetic variants. Converging evidence shows that genetic variants with low marginal effects (LME) play an important role in disease development. Despite their potential significance, discovering LME genetic variants and assessing their joint association on high dimensional data (e.g., genome wide association studies) remain a great challenge. To facilitate joint association analysis among a large ensemble of LME genetic variants, we proposed a computationally efficient and powerful approach, which we call Trees Assembling Mann whitney (TAMW). Through simulation studies and an empirical data application, we found that TAMW outperformed multifactor dimensionality reduction (MDR) and the likelihood ratio based Mann whitney approach (LRMW) when the underlying complex disease involves multiple LME loci and their interactions. For instance, in a simulation with 20 interacting LME loci, TAMW attained a higher power (power=0.931) than both MDR (power=0.599) and LRMW (power=0.704). In an empirical study of 29 known Crohn's disease (CD) loci, TAMW also identified a stronger joint association with CD than those detected by MDR and LRMW. Finally, we applied TAMW to Wellcome Trust CD GWAS to conduct a genome wide analysis. The analysis of 459K single nucleotide polymorphisms was completed in 40 hours using parallel computing, and revealed a joint association predisposing to CD (p-value=2.763e-19). Further analysis of the newly discovered association suggested that 13 genes, such as ATG16L1 and LACC1, may play an important role in CD pathophysiological and etiological processes. 

**Abstract (ZH)**: 遗传变异的共同复杂性疾病联合关联分析：Trees Assembling Mann Whitney (TAMW) 方法的研究 

---
# A Weighted U Statistic for Genetic Association Analyses of Sequencing Data 

**Title (ZH)**: 加权U统计量在序列数据遗传关联分析中的应用 

**Authors**: Changshuai Wei, Ming Li, Zihuai He, Olga Vsevolozhskaya, Daniel J. Schaid, Qing Lu  

**Link**: [PDF](https://arxiv.org/pdf/1505.01204)  

**Abstract**: With advancements in next generation sequencing technology, a massive amount of sequencing data are generated, offering a great opportunity to comprehensively investigate the role of rare variants in the genetic etiology of complex diseases. Nevertheless, this poses a great challenge for the statistical analysis of high-dimensional sequencing data. The association analyses based on traditional statistical methods suffer substantial power loss because of the low frequency of genetic variants and the extremely high dimensionality of the data. We developed a weighted U statistic, referred to as WU-seq, for the high-dimensional association analysis of sequencing data. Based on a non-parametric U statistic, WU-SEQ makes no assumption of the underlying disease model and phenotype distribution, and can be applied to a variety of phenotypes. Through simulation studies and an empirical study, we showed that WU-SEQ outperformed a commonly used SKAT method when the underlying assumptions were violated (e.g., the phenotype followed a heavy-tailed distribution). Even when the assumptions were satisfied, WU-SEQ still attained comparable performance to SKAT. Finally, we applied WU-seq to sequencing data from the Dallas Heart Study (DHS), and detected an association between ANGPTL 4 and very low density lipoprotein cholesterol. 

**Abstract (ZH)**: 随着下一代测序技术的发展，产生了大量的测序数据，为全面探索罕见变异在复杂疾病遗传病因中的作用提供了巨大机会。然而，这给高维测序数据的统计分析带来了巨大挑战。基于传统统计方法的关联分析因遗传变异频率低和数据的极高维度而遭受严重效能损失。我们开发了一种加权U统计量，称为WU-seq，用于测序数据的高维关联分析。基于非参数U统计量，WU-SEQ不假设底层疾病模型和表型分布，并可用于多种表型。通过模拟研究和实证研究，我们表明，当底层假设被违反时（例如，表型遵循重尾分布），WU-SEQ优于常用的SKAT方法。即使在假设成立时，WU-SEQ也能达到与SKAT相当的性能。最终，我们应用WU-seq分析了达拉斯心脏研究（DHS）的测序数据，并检测到ANGPTL 4与极低密度脂蛋白胆固醇之间的关联。 

---
# A Generalized Similarity U Test for Multivariate Analysis of Sequencing Data 

**Title (ZH)**: 泛化相似性U检验在序列数据多元分析中的应用 

**Authors**: Changshuai Wei, Qing Lu  

**Link**: [PDF](https://arxiv.org/pdf/1505.01179)  

**Abstract**: Sequencing-based studies are emerging as a major tool for genetic association studies of complex diseases. These studies pose great challenges to the traditional statistical methods (e.g., single-locus analyses based on regression methods) because of the high-dimensionality of data and the low frequency of genetic variants. In addition, there is a great interest in biology and epidemiology to identify genetic risk factors contributed to multiple disease phenotypes. The multiple phenotypes can often follow different distributions, which violates the assumptions of most current methods. In this paper, we propose a generalized similarity U test, referred to as GSU. GSU is a similarity-based test and can handle high-dimensional genotypes and phenotypes. We studied the theoretical properties of GSU, and provided the efficient p-value calculation for association test as well as the sample size and power calculation for the study design. Through simulation, we found that GSU had advantages over existing methods in terms of power and robustness to phenotype distributions. Finally, we used GSU to perform a multivariate analysis of sequencing data in the Dallas Heart Study and identified a joint association of 4 genes with 5 metabolic related phenotypes. 

**Abstract (ZH)**: 基于测序的研究正逐渐成为复杂疾病遗传关联研究的重要工具。这些研究给传统的统计方法（如基于回归方法的单核苷酸分析）带来了巨大挑战，因为数据的高维度性和遗传变异的低频性。此外，在生物学和流行病学中，人们对识别 Contribution to 多个疾病表型的遗传风险因素表现出极大的兴趣。这些表型往往遵循不同的分布，这违反了当前大多数方法的基本假设。在本文中，我们提出了一种广义相似性U检验，称为GSU。GSU 是一种基于相似性的检验，可以处理高维度的基因型和表型。我们研究了 GSU 的理论性质，并提供了关联检验的高效p值计算方法以及研究设计中的样本量和功效计算。通过模拟，我们发现 GSU 在功效和表型分布稳健性方面优于现有方法。最后，我们在达拉斯心脏研究中使用 GSU 进行了多变量测序数据分析，并识别出4个基因与5个代谢相关表型的联合关联。 

---
# A weighted U statistic for association analysis considering genetic heterogeneity 

**Title (ZH)**: 考虑遗传异质性的加权U统计量用于关联分析 

**Authors**: Changshuai Wei, Robert C. Elston, Qing Lu  

**Link**: [PDF](https://arxiv.org/pdf/1504.08319)  

**Abstract**: Converging evidence suggests that common complex diseases with the same or similar clinical manifestations could have different underlying genetic etiologies. While current research interests have shifted toward uncovering rare variants and structural variations predisposing to human diseases, the impact of heterogeneity in genetic studies of complex diseases has been largely overlooked. Most of the existing statistical methods assume the disease under investigation has a homogeneous genetic effect and could, therefore, have low power if the disease undergoes heterogeneous pathophysiological and etiological processes. In this paper, we propose a heterogeneity weighted U (HWU) method for association analyses considering genetic heterogeneity. HWU can be applied to various types of phenotypes (e.g., binary and continuous) and is computationally effcient for high- dimensional genetic data. Through simulations, we showed the advantage of HWU when the underlying genetic etiology of a disease was heterogeneous, as well as the robustness of HWU against different model assumptions (e.g., phenotype distributions). Using HWU, we conducted a genome-wide analysis of nicotine dependence from the Study of Addiction: Genetics and Environments (SAGE) dataset. The genome-wide analysis of nearly one million genetic markers took 7 hours, identifying heterogeneous effects of two new genes (i.e., CYP3A5 and IKBKB) on nicotine dependence. 

**Abstract (ZH)**: 不同临床表现的常见复杂疾病的遗传异质性证据正在不断汇聚。当前研究兴趣已转向揭示诱发人类疾病的风险罕见变异和结构变异，但遗传研究中的异质性影响已被广泛关注不足。现有的大多数统计方法假设所研究的疾病具有均质的遗传效应，因此在疾病经历异质的病理生理学和病因学过程时，可能会导致较低的统计功效。本文提出了一种考虑遗传异质性的异质性加权U（HWU）方法，用于关联分析。HWU可以应用于各种类型的表型（如二元和连续型），并且对于高维遗传数据具有计算效率。通过模拟实验，我们展示了当疾病的根本遗传病因异质时，HWU的优势，以及HWU在不同模型假设（如表型分布）下的稳健性。使用HWU，我们对来自Addiction: Genetics and Environments (AGE) 数据集的尼古丁依赖进行了全基因组分析。全基因组分析近一百万个遗传标记耗时7小时，鉴定出两种新的基因（即CYP3A5和IKBKB）在尼古丁依赖中的异质性效应。 

---
