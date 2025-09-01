# Automated Clinical Problem Detection from SOAP Notes using a Collaborative Multi-Agent LLM Architecture 

**Title (ZH)**: 基于协作多智能体LLM架构的SOAP笔记临床问题自动检测 

**Authors**: Yeawon Lee, Xiaoyang Wang, Christopher C. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.21803)  

**Abstract**: Accurate interpretation of clinical narratives is critical for patient care, but the complexity of these notes makes automation challenging. While Large Language Models (LLMs) show promise, single-model approaches can lack the robustness required for high-stakes clinical tasks. We introduce a collaborative multi-agent system (MAS) that models a clinical consultation team to address this gap. The system is tasked with identifying clinical problems by analyzing only the Subjective (S) and Objective (O) sections of SOAP notes, simulating the diagnostic reasoning process of synthesizing raw data into an assessment. A Manager agent orchestrates a dynamically assigned team of specialist agents who engage in a hierarchical, iterative debate to reach a consensus. We evaluated our MAS against a single-agent baseline on a curated dataset of 420 MIMIC-III notes. The dynamic multi-agent configuration demonstrated consistently improved performance in identifying congestive heart failure, acute kidney injury, and sepsis. Qualitative analysis of the agent debates reveals that this structure effectively surfaces and weighs conflicting evidence, though it can occasionally be susceptible to groupthink. By modeling a clinical team's reasoning process, our system offers a promising path toward more accurate, robust, and interpretable clinical decision support tools. 

**Abstract (ZH)**: 准确解释临床叙事对于患者护理至关重要，但这些笔记的复杂性使自动化任务面临挑战。虽然大规模语言模型(Large Language Models, LLMs)前景广阔，但单一模型的方法可能无法满足高风险临床任务所需的稳健性。我们提出了一种协作多智能体系统(collaborative multi-agent system, MAS)，该系统旨在通过分析SOAP笔记中仅有的主诉(S)和体征(O)部分来模拟诊断推理过程，从而识别临床问题。管理(agent Manager)智能体协调一个动态分配的专业智能体团队，进行层次化的迭代辩论，以达成共识。我们在一个包含420份MIMIC-III笔记的精心策划数据集上将我们的MAS与单一智能体基线进行评估。动态多智能体配置在识别充血性心力衰竭、急性肾损伤和败血症方面表现出一致的性能提升。对智能体辩论的定性分析表明，这种结构有效地突显并权衡了相冲突的证据，尽管偶尔会受到群体思维的影响。通过模拟临床团队的推理过程，我们的系统为更加准确、稳健和可解释的临床决策支持工具开辟了有前景的道路。 

---
# Tree-Guided Diffusion Planner 

**Title (ZH)**: 树引导扩散规划者 

**Authors**: Hyeonseong Jeon, Cheolhong Min, Jaesik Park  

**Link**: [PDF](https://arxiv.org/pdf/2508.21800)  

**Abstract**: Planning with pretrained diffusion models has emerged as a promising approach for solving test-time guided control problems. However, standard gradient guidance typically performs optimally under convex and differentiable reward landscapes, showing substantially reduced effectiveness in real-world scenarios involving non-convex objectives, non-differentiable constraints, and multi-reward structures. Furthermore, recent supervised planning approaches require task-specific training or value estimators, which limits test-time flexibility and zero-shot generalization. We propose a Tree-guided Diffusion Planner (TDP), a zero-shot test-time planning framework that balances exploration and exploitation through structured trajectory generation. We frame test-time planning as a tree search problem using a bi-level sampling process: (1) diverse parent trajectories are produced via training-free particle guidance to encourage broad exploration, and (2) sub-trajectories are refined through fast conditional denoising guided by task objectives. TDP addresses the limitations of gradient guidance by exploring diverse trajectory regions and harnessing gradient information across this expanded solution space using only pretrained models and test-time reward signals. We evaluate TDP on three diverse tasks: maze gold-picking, robot arm block manipulation, and AntMaze multi-goal exploration. TDP consistently outperforms state-of-the-art approaches on all tasks. The project page can be found at: this http URL. 

**Abstract (ZH)**: 基于树引导的扩散计划器：一种零样本测试时规划框架 

---
# Orientability of Causal Relations in Time Series using Summary Causal Graphs and Faithful Distributions 

**Title (ZH)**: 时间序列中因果关系的可定向性：基于摘要因果图和忠实分布的研究 

**Authors**: Timothée Loranchet, Charles K. Assaad  

**Link**: [PDF](https://arxiv.org/pdf/2508.21742)  

**Abstract**: Understanding causal relations between temporal variables is a central challenge in time series analysis, particularly when the full causal structure is unknown. Even when the full causal structure cannot be fully specified, experts often succeed in providing a high-level abstraction of the causal graph, known as a summary causal graph, which captures the main causal relations between different time series while abstracting away micro-level details. In this work, we present conditions that guarantee the orientability of micro-level edges between temporal variables given the background knowledge encoded in a summary causal graph and assuming having access to a faithful and causally sufficient distribution with respect to the true unknown graph. Our results provide theoretical guarantees for edge orientation at the micro-level, even in the presence of cycles or bidirected edges at the macro-level. These findings offer practical guidance for leveraging SCGs to inform causal discovery in complex temporal systems and highlight the value of incorporating expert knowledge to improve causal inference from observational time series data. 

**Abstract (ZH)**: 理解时间变量之间的因果关系是时间序列分析中的一个核心挑战，特别是在因果结构完全未知的情况下。即使无法完全指定因果结构，专家通常能够提供一个高层次的因果图的抽象，称为汇总因果图，该图捕获了不同时间序列之间的主要因果关系，同时忽略了微观细节。在本文中，我们提出了在汇总因果图的背景知识下，假设可以访问与真实未知图忠实且因果充分的分布时，保证时间变量之间微观层面边的方向性的条件。我们的结果为在宏观层面存在循环或双箭头边的情况下，提供边缘定向的理论保证。这些发现为利用汇总因果图指导复杂时序系统中的因果发现提供了实践指导，并强调了整合专家知识以改进从观察时序数据中进行因果推断的价值。 

---
# Freeze and Conquer: Reusable Ansatz for Solving the Traveling Salesman Problem 

**Title (ZH)**: 冻结并征服：解决旅行商问题的可重用Ansatz 

**Authors**: Fabrizio Fagiolo, Nicolo' Vescera  

**Link**: [PDF](https://arxiv.org/pdf/2508.21730)  

**Abstract**: In this paper we present a variational algorithm for the Traveling Salesman Problem (TSP) that combines (i) a compact encoding of permutations, which reduces the qubit requirement too, (ii) an optimize-freeze-reuse strategy: where the circuit topology (``Ansatz'') is first optimized on a training instance by Simulated Annealing (SA), then ``frozen'' and re-used on novel instances, limited to a rapid re-optimization of only the circuit parameters. This pipeline eliminates costly structural research in testing, making the procedure immediately implementable on NISQ hardware.
On a set of $40$ randomly generated symmetric instances that span $4 - 7$ cities, the resulting Ansatz achieves an average optimal trip sampling probability of $100\%$ for 4 city cases, $90\%$ for 5 city cases and $80\%$ for 6 city cases. With 7 cities the success rate drops markedly to an average of $\sim 20\%$, revealing the onset of scalability limitations of the proposed method.
The results show robust generalization ability for moderate problem sizes and indicate how freezing the Ansatz can dramatically reduce time-to-solution without degrading solution quality. The paper also discusses scalability limitations, the impact of ``warm-start'' initialization of parameters, and prospects for extension to more complex problems, such as Vehicle Routing and Job-Shop Scheduling. 

**Abstract (ZH)**: 一种结合紧凑排列编码和优化冻结重用策略的变分算法求解旅行商问题 

---
# PosterForest: Hierarchical Multi-Agent Collaboration for Scientific Poster Generation 

**Title (ZH)**: PosterForest: 分层多智能体协作生成科学poster 

**Authors**: Jiho Choi, Seojeong Park, Seongjong Song, Hyunjung Shim  

**Link**: [PDF](https://arxiv.org/pdf/2508.21720)  

**Abstract**: We present a novel training-free framework, \textit{PosterForest}, for automated scientific poster generation. Unlike prior approaches, which largely neglect the hierarchical structure of scientific documents and the semantic integration of textual and visual elements, our method addresses both challenges directly. We introduce the \textit{Poster Tree}, a hierarchical intermediate representation that jointly encodes document structure and visual-textual relationships at multiple levels. Our framework employs a multi-agent collaboration strategy, where agents specializing in content summarization and layout planning iteratively coordinate and provide mutual feedback. This approach enables the joint optimization of logical consistency, content fidelity, and visual coherence. Extensive experiments on multiple academic domains show that our method outperforms existing baselines in both qualitative and quantitative evaluations. The resulting posters achieve quality closest to expert-designed ground truth and deliver superior information preservation, structural clarity, and user preference. 

**Abstract (ZH)**: 我们提出了一种无需训练的新型框架 \textit{PosterForest}，用于自动化科学海报生成。与之前的 approaches 不同，我们的方法直接解决了科学文档的层次结构及其文本和视觉元素语义集成这两个挑战。我们引入了 \textit{Poster Tree}，这是一种层次化的中间表示，能够同时在多个层次上联合编码文档结构和视觉-文本关系。我们的框架采用多智能体协作策略，其中专门负责内容总结和布局规划的智能体相互协作并提供相互反馈。这种方法使得逻辑一致性、内容保真度和视觉连贯性的联合优化成为可能。在多个学术领域的广泛实验表明，我们的方法在定性和定量评估中均优于现有基线。生成的海报的质量接近专家设计的真实标杆，并且在信息保存、结构清晰度和用户偏好方面表现更优。 

---
# Leveraging Imperfection with MEDLEY A Multi-Model Approach Harnessing Bias in Medical AI 

**Title (ZH)**: 利用 Imperfection 机遇：一种利用医疗AI偏见的多模型方法（MEDLEY） 

**Authors**: Farhad Abtahi, Mehdi Astaraki, Fernando Seoane  

**Link**: [PDF](https://arxiv.org/pdf/2508.21648)  

**Abstract**: Bias in medical artificial intelligence is conventionally viewed as a defect requiring elimination. However, human reasoning inherently incorporates biases shaped by education, culture, and experience, suggesting their presence may be inevitable and potentially valuable. We propose MEDLEY (Medical Ensemble Diagnostic system with Leveraged diversitY), a conceptual framework that orchestrates multiple AI models while preserving their diverse outputs rather than collapsing them into a consensus. Unlike traditional approaches that suppress disagreement, MEDLEY documents model-specific biases as potential strengths and treats hallucinations as provisional hypotheses for clinician verification. A proof-of-concept demonstrator was developed using over 30 large language models, creating a minimum viable product that preserved both consensus and minority views in synthetic cases, making diagnostic uncertainty and latent biases transparent for clinical oversight. While not yet a validated clinical tool, the demonstration illustrates how structured diversity can enhance medical reasoning under clinician supervision. By reframing AI imperfection as a resource, MEDLEY offers a paradigm shift that opens new regulatory, ethical, and innovation pathways for developing trustworthy medical AI systems. 

**Abstract (ZH)**: 医学人工智能中的偏差通常被视为需要消除的缺陷。然而，人类推理本质上融合了由教育、文化及经验形成的偏差，这表明其存在可能是不可避免且可能有价值的。我们提出了一种MEDLEY（Medical Ensemble Diagnostic system with Leveraged diversitY）概念框架，该框架协调多个AI模型并保留其多样化的输出，而非将其压缩为一致意见。与传统的抑制分歧的方法不同，MEDLEY记录模型特定的偏差作为潜在的优势，并将幻觉视为供临床医生验证的暂定假设。该概念验证演示器使用了超过30个大型语言模型，开发了一个最小可行产品，该产品在合成案例中既保留了共识意见也保留了少数派观点，使诊断不确定性及潜在偏差透明化，便于临床监督。尽管尚未被验证为临床工具，该演示展示了在临床监督下结构化多样性的增强效果。通过将AI不完美之处重新定义为资源，MEDLEY提供了一种范式转变，开启了开发可信的医疗AI系统的新型监管、伦理和创新途径。 

---
# A-MHA*: Anytime Multi-Heuristic A* 

**Title (ZH)**: A-MHA*: 任何时间多启发式A* 

**Authors**: Ramkumar Natarajan, Muhammad Suhail Saleem, William Xiao, Sandip Aine, Howie Choset, Maxim Likhachev  

**Link**: [PDF](https://arxiv.org/pdf/2508.21637)  

**Abstract**: Designing good heuristic functions for graph search requires adequate domain knowledge. It is often easy to design heuristics that perform well and correlate with the underlying true cost-to-go values in certain parts of the search space but these may not be admissible throughout the domain thereby affecting the optimality guarantees of the search. Bounded suboptimal search using several such partially good but inadmissible heuristics was developed in Multi-Heuristic A* (MHA*). Although MHA* leverages multiple inadmissible heuristics to potentially generate a faster suboptimal solution, the original version does not improve the solution over time. It is a one shot algorithm that requires careful setting of inflation factors to obtain a desired one time solution. In this work, we tackle this issue by extending MHA* to an anytime version that finds a feasible suboptimal solution quickly and continually improves it until time runs out. Our work is inspired from the Anytime Repairing A* (ARA*) algorithm. We prove that our precise adaptation of ARA* concepts in the MHA* framework preserves the original suboptimal and completeness guarantees and enhances MHA* to perform in an anytime fashion. Furthermore, we report the performance of A-MHA* in 3-D path planning domain and sliding tiles puzzle and compare against MHA* and other anytime algorithms. 

**Abstract (ZH)**: 设计好的图搜索启发函数需要足够的领域知识。虽然在搜索空间的某些部分容易设计出性能良好且与实际剩余代价高度相关的启发函数，但这些启发函数在整个领域内可能不具备可接纳性，从而影响搜索的最优性保证。Multi-Heuristic A* (MHA*) 开发了使用多个部分良好但不可接纳的启发函数来进行有界次优搜索的方法，尽管 MHA* 可以利用多个启发函数来潜在地生成更快的次优解，但其原始版本不会随时间改进解的质量。它是一个一次性算法，需要精细设置放缩因子来获得一次满意的解。本文通过将 MHA* 扩展为一种任意时间版本来解决这一问题，该版本可以快速找到一个可行的次优解，并在时间允许的情况下不断改进该解。我们的工作受到了 Anytime Repairing A* (ARA*) 算法的启发。我们证明，MHA* 框架中我们精确适应 ARA* 的概念保留了原始的次优和完备性保证，并使 MHA* 能够以任意时间的方式运行。此外，我们在 3D 路径规划领域和滑动方块谜题中报告了 A-MHA* 的性能，并将其与其他任意时间算法进行了比较。 

---
# Integrating Large Language Models with Network Optimization for Interactive and Explainable Supply Chain Planning: A Real-World Case Study 

**Title (ZH)**: 将大规模语言模型与网络优化集成以实现交互式和可解释的供应链规划：一个实际案例研究 

**Authors**: Saravanan Venkatachalam  

**Link**: [PDF](https://arxiv.org/pdf/2508.21622)  

**Abstract**: This paper presents an integrated framework that combines traditional network optimization models with large language models (LLMs) to deliver interactive, explainable, and role-aware decision support for supply chain planning. The proposed system bridges the gap between complex operations research outputs and business stakeholder understanding by generating natural language summaries, contextual visualizations, and tailored key performance indicators (KPIs). The core optimization model addresses tactical inventory redistribution across a network of distribution centers for multi-period and multi-item, using a mixed-integer formulation. The technical architecture incorporates AI agents, RESTful APIs, and a dynamic user interface to support real-time interaction, configuration updates, and simulation-based insights. A case study demonstrates how the system improves planning outcomes by preventing stockouts, reducing costs, and maintaining service levels. Future extensions include integrating private LLMs, transfer learning, reinforcement learning, and Bayesian neural networks to enhance explainability, adaptability, and real-time decision-making. 

**Abstract (ZH)**: 本文提出了一种结合传统网络优化模型与大规模语言模型（LLMs）的集成框架，以提供供应链计划中的互动、可解释和角色感知的决策支持。该提出的系统通过生成自然语言摘要、情境可视化和定制的关键绩效指标（KPIs），弥合了复杂运作研究输出与业务相关方理解之间的差距。核心优化模型采用混合整数规划形式，针对多个时期和多项目的分配中心网络，解决战略库存再分配问题。技术架构包括AI代理、RESTful API和动态用户界面，以支持实时交互、配置更新和基于模拟的洞察。案例研究展示了该系统通过防止缺货、降低费用和维持服务级别来改善规划结果。未来扩展包括集成私人LLMs、迁移学习、强化学习和贝叶斯神经网络，以提高可解释性、适应性和实时决策能力。 

---
# Scalable Solution Methods for Dec-POMDPs with Deterministic Dynamics 

**Title (ZH)**: 分布确定动力学条件下扩展解决方法的Dec-POMDP研究 

**Authors**: Yang You, Alex Schutz, Zhikun Li, Bruno Lacerda, Robert Skilton, Nick Hawes  

**Link**: [PDF](https://arxiv.org/pdf/2508.21595)  

**Abstract**: Many high-level multi-agent planning problems, including multi-robot navigation and path planning, can be effectively modeled using deterministic actions and observations.
In this work, we focus on such domains and introduce the class of Deterministic Decentralized POMDPs (Det-Dec-POMDPs). This is a subclass of Dec-POMDPs characterized by deterministic transitions and observations conditioned on the state and joint actions.
We then propose a practical solver called Iterative Deterministic POMDP Planning (IDPP). This method builds on the classic Joint Equilibrium Search for Policies framework and is specifically optimized to handle large-scale Det-Dec-POMDPs that current Dec-POMDP solvers are unable to address efficiently. 

**Abstract (ZH)**: 许多高级多智能体规划问题，包括多机器人导航和路径规划，可以用确定性动作和观测有效建模。
在本文中，我们关注此类领域，并介绍了确定性分散部分可观测马尔可夫决策过程（Det-Dec-POMDPs）这一类。这是Dec-POMDP的一种子类，其特征是基于状态和联合动作的确定性转换和观测。
然后我们提出了一个实用的解算器，称为迭代确定性POMDP规划（IDPP）。该方法基于经典的联合均衡搜索策略框架，并特别优化以处理当前Dec-POMDP解算器无法高效处理的大规模Det-Dec-POMDP。 

---
# Revisiting Landmarks: Learning from Previous Plans to Generalize over Problem Instances 

**Title (ZH)**: 重新审视地标：从先前计划中学习以在问题实例间泛化 

**Authors**: Issa Hanou, Sebastijan Dumančić, Mathijs de Weerdt  

**Link**: [PDF](https://arxiv.org/pdf/2508.21564)  

**Abstract**: We propose a new framework for discovering landmarks that automatically generalize across a domain. These generalized landmarks are learned from a set of solved instances and describe intermediate goals for planning problems where traditional landmark extraction algorithms fall short. Our generalized landmarks extend beyond the predicates of a domain by using state functions that are independent of the objects of a specific problem and apply to all similar objects, thus capturing repetition. Based on these functions, we construct a directed generalized landmark graph that defines the landmark progression, including loop possibilities for repetitive subplans. We show how to use this graph in a heuristic to solve new problem instances of the same domain. Our results show that the generalized landmark graphs learned from a few small instances are also effective for larger instances in the same domain. If a loop that indicates repetition is identified, we see a significant improvement in heuristic performance over the baseline. Generalized landmarks capture domain information that is interpretable and useful to an automated planner. This information can be discovered from a small set of plans for the same domain. 

**Abstract (ZH)**: 我们提出了一种新的框架，用于发现能够在整个领域中自动泛化的地标。这些泛化的地标是从一组已解决的实例中学习得到的，并描述了解决传统地标提取算法无法处理的规划问题时的中间目标。这些泛化的地标通过使用独立于特定问题对象的状态函数，并适用于所有类似对象的函数来扩展领域谓词，从而捕获重复性。基于这些函数，我们构建了一个有向泛化地标图，定义了地标演变，包括重复子计划的循环可能性。我们展示了如何使用该图在启发式算法中解决同领域的新实例问题。我们的结果表明，从少数小实例中学习得到的泛化地标图对于同一领域中的较大实例同样有效。如果识别出表明重复性的循环，启发式的性能将显著优于基线。泛化的地标捕获了可解释且对自动规划器有用的领域信息，这些信息可以从同一个领域中的一组计划中发现。 

---
# HealthProcessAI: A Technical Framework and Proof-of-Concept for LLM-Enhanced Healthcare Process Mining 

**Title (ZH)**: 健康过程AI：增强型LLM辅助医疗过程挖掘的技术框架与概念验证 

**Authors**: Eduardo Illueca-Fernandez, Kaile Chen, Fernando Seoane, Farhad Abtahi  

**Link**: [PDF](https://arxiv.org/pdf/2508.21540)  

**Abstract**: Process mining has emerged as a powerful analytical technique for understanding complex healthcare workflows. However, its application faces significant barriers, including technical complexity, a lack of standardized approaches, and limited access to practical training resources. We introduce HealthProcessAI, a GenAI framework designed to simplify process mining applications in healthcare and epidemiology by providing a comprehensive wrapper around existing Python (PM4PY) and R (bupaR) libraries. To address unfamiliarity and improve accessibility, the framework integrates multiple Large Language Models (LLMs) for automated process map interpretation and report generation, helping translate technical analyses into outputs that diverse users can readily understand. We validated the framework using sepsis progression data as a proof-of-concept example and compared the outputs of five state-of-the-art LLM models through the OpenRouter platform. To test its functionality, the framework successfully processed sepsis data across four proof-of-concept scenarios, demonstrating robust technical performance and its capability to generate reports through automated LLM analysis. LLM evaluation using five independent LLMs as automated evaluators revealed distinct model strengths: Claude Sonnet-4 and Gemini 2.5-Pro achieved the highest consistency scores (3.79/4.0 and 3.65/4.0) when evaluated by automated LLM assessors. By integrating multiple Large Language Models (LLMs) for automated interpretation and report generation, the framework addresses widespread unfamiliarity with process mining outputs, making them more accessible to clinicians, data scientists, and researchers. This structured analytics and AI-driven interpretation combination represents a novel methodological advance in translating complex process mining results into potentially actionable insights for healthcare applications. 

**Abstract (ZH)**: HealthProcessAI：一种集成大型语言模型的GenAI框架，用于简化医疗和流行病学中的过程挖掘应用 

---
# Counterfactual Scenarios for Automated Planning 

**Title (ZH)**: .Counterfactual 情景下的自动化规划 

**Authors**: Nicola Gigante, Francesco Leofante, Andrea Micheli  

**Link**: [PDF](https://arxiv.org/pdf/2508.21521)  

**Abstract**: Counterfactual Explanations (CEs) are a powerful technique used to explain Machine Learning models by showing how the input to a model should be minimally changed for the model to produce a different output. Similar proposals have been made in the context of Automated Planning, where CEs have been characterised in terms of minimal modifications to an existing plan that would result in the satisfaction of a different goal. While such explanations may help diagnose faults and reason about the characteristics of a plan, they fail to capture higher-level properties of the problem being solved. To address this limitation, we propose a novel explanation paradigm that is based on counterfactual scenarios. In particular, given a planning problem $P$ and an \ltlf formula $\psi$ defining desired properties of a plan, counterfactual scenarios identify minimal modifications to $P$ such that it admits plans that comply with $\psi$. In this paper, we present two qualitative instantiations of counterfactual scenarios based on an explicit quantification over plans that must satisfy $\psi$. We then characterise the computational complexity of generating such counterfactual scenarios when different types of changes are allowed on $P$. We show that producing counterfactual scenarios is often only as expensive as computing a plan for $P$, thus demonstrating the practical viability of our proposal and ultimately providing a framework to construct practical algorithms in this area. 

**Abstract (ZH)**: 基于反事实情景的解释范式 

---
# Modeling Wise Decision Making: A Z-Number Fuzzy Framework Inspired by Phronesis 

**Title (ZH)**: 基于贤政治国理念的Z-数模糊框架：明智决策建模 

**Authors**: Sweta Kaman, Ankita Sharma, Romi Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2508.21517)  

**Abstract**: Background: Wisdom is a superordinate construct that embraces perspective taking, reflectiveness, prosocial orientation, reflective empathetic action, and intellectual humility. Unlike conventional models of reasoning that are rigidly bound by binary thinking, wisdom unfolds in shades of ambiguity, requiring both graded evaluation and self-reflective humility. Current measures depend on self-reports and seldom reflect the humility and uncertainty inherent in wise reasoning. A computational framework that takes into account both multidimensionality and confidence has the potential to improve psychological science and allow humane AI. Method: We present a fuzzy inference system with Z numbers, each of the decisions being expressed in terms of a wisdom score (restriction) and confidence score (certainty). As part of this study, participants (N = 100) were exposed to culturally neutral pictorial moral dilemma tasks to which they generated think-aloud linguistic responses, which were mapped into five theoretically based components of wisdom. The scores of each individual component were combined using a base of 21 rules, with membership functions tuned via Gaussian kernel density estimation. Results: In a proof of concept study, the system produced dual attribute wisdom representations that correlated modestly but significantly with established scales while showing negligible relations with unrelated traits, supporting convergent and divergent validity. Contribution: The contribution is to formalize wisdom as a multidimensional, uncertainty-conscious construct, operationalized in the form of Z-numbers. In addition to progressing measurement in psychology, it calculates how fuzzy Z numbers can provide AI systems with interpretable, confidence-sensitive reasoning that affords a safe, middle ground between rigorous computation and human-like judgment. 

**Abstract (ZH)**: 背景：智慧是一种超阶构造，包括视角转换、反思性、利他倾向、反思性同理行动以及智力谦逊。与传统二元推理模型的刚性思维不同，智慧在灰色地带展开，要求进行逐步评估和自我反思的谦逊。现有测度依赖于自我报告，很少反映智慧推理中的谦逊和不确定性。一种考虑多维性和置信度的计算框架有可能提高心理学研究并允许人性化的人工智能。方法：我们提出了一种基于Z数的模糊推理系统，每个决策以智慧评分（限制）和置信评分（确定性）的形式表达。作为本研究的一部分，参与者（N=100）完成了文化中立的图示道德困境任务，并生成了自我陈述语言响应，这些响应被映射到五个理论基础上的智慧组成部分。每个组成部分的得分使用21条基本规则结合，通过高斯核密度估计调整隶属度函数。结果：在概念验证研究中，该系统生成了双重属性的智慧表示，这些表示与现有量表相关适度但显著，与无关特质几乎没有关系，支持了一致效度和区分效度。贡献：贡献在于将智慧形式化为一个多维、注意不确定性的构造，并以Z数的形式实现。除了推进心理学中的测量方法外，它计算了模糊Z数如何为人工智能系统提供解释性强、置信度敏感的推理，从而提供严格计算与类人判断之间的安全、中间地带。 

---
# MMSearch-Plus: A Simple Yet Challenging Benchmark for Multimodal Browsing Agents 

**Title (ZH)**: MMSearch-Plus: 一个简单而具有挑战性的多模态浏览代理基准 

**Authors**: Xijia Tao, Yihua Teng, Xinxing Su, Xinyu Fu, Jihao Wu, Chaofan Tao, Ziru Liu, Haoli Bai, Rui Liu, Lingpeng Kong  

**Link**: [PDF](https://arxiv.org/pdf/2508.21475)  

**Abstract**: Large multimodal language models (MLLMs) are increasingly deployed as web agents, yet many multimodal browsing benchmarks can be solved by shallow, fixed workflows that lean on high-recall image search and nearby text-masking the genuinely multimodal challenges of fine-grained visual reasoning, provenance verification, and long-horizon tool use. We introduce MMSearch-Plus, a benchmark of 311 tasks that highly demand multimodal understanding while preserving the difficulty profile of strong text-only browsing suites. Each item is constructed to contain multiple weak, localized visual signals that must be extracted, propagated through iterative text-image search, and cross-validated under retrieval noise before answering. Our curation procedure, Spatial-Temporal Extrapolation, seeds questions whose answers require extrapolating from spatial cues (micro-text, part-level appearance, layouts, signage) and temporal traces (broadcast overlays, seasonal context) to out-of-image facts such as events, dates, and venues. We provide a model-agnostic agent framework with browsing tools and evaluate a range of closed and open MLLMs. The strongest agent (o3) attains 15.1% without search and 36.0% accuracy with rollout under our framework, while a strong open-source model (Qwen-2.5-VL-72B-Instruct) achieves 0.0% without search and 6.9% after 20 rounds of search. Beyond answer accuracy, we assess bounding-box production and cropped-image search, and conduct an error analysis that surfaces failures in source verification, part-based reasoning, and long-horizon planning. 

**Abstract (ZH)**: 大规模多模态语言模型（MLLMs）正越来越多地作为网络代理部署，然而许多多模态浏览基准可以通过依赖高召回率图像搜索和附近文本遮罩的方法来解决，这些方法忽视了真正需要精细视觉推理、来源验证和长期工具使用等多模态挑战。我们引入了MMSearch-Plus，这是一个包含311项任务的基准测试，这些任务高度要求多模态理解同时保持强大的文本-only 浏览套件的难度特征。每项任务都构建为包含多个弱局部视觉信号，这些信号必须被提取并通过迭代的文本-图像搜索传播，在检索噪声下进行交叉验证后才能回答。我们的编纂程序，时空外推，通过从空间线索（微观文本、部件级外观、布局、标识）和时间踪迹（广播覆盖、季节背景）推断出图像外的事实（事件、日期、场地）来提出问题。我们提供了一个模型无关的代理框架，配备了浏览工具，并评估了多种闭源和开源MMLMs。在我们的框架下，最强的代理（o3）在无搜索情况下达到15.1%的准确率，并在进行展开后达到36.0%的准确率，而在无搜索情况下，一个强大的开源模型（Qwen-2.5-VL-72B-Instruct）达到0.0%的准确率，并在20轮搜索后达到6.9%的准确率。除了答案准确性，我们还评估了边界框生成和裁剪图像搜索，并进行了一项错误分析，揭示了来源验证、部件级推理和长期规划等方面的失败。 

---
# Learning Lifted Action Models From Traces of Incomplete Actions and States 

**Title (ZH)**: 从不完整动作和状态轨迹中学习提升的动作模型 

**Authors**: Niklas Jansen, Jonas Gösgens, Hector Geffner  

**Link**: [PDF](https://arxiv.org/pdf/2508.21449)  

**Abstract**: Consider the problem of learning a lifted STRIPS model of the sliding-tile puzzle from random state-action traces where the states represent the location of the tiles only, and the actions are the labels up, down, left, and right, with no arguments. Two challenges are involved in this problem. First, the states are not full STRIPS states, as some predicates are missing, like the atoms representing the position of the ``blank''. Second, the actions are not full STRIPS either, as they do not reveal all the objects involved in the actions effects and preconditions. Previous approaches have addressed different versions of this model learning problem, but most assume that actions in the traces are full STRIPS actions or that the domain predicates are all observable. The new setting considered in this work is more ``realistic'', as the atoms observed convey the state of the world but not full STRIPS states, and the actions reveal the arguments needed for selecting the action but not the ones needed for modeling it in STRIPS. For formulating and addressing the learning problem, we introduce a variant of STRIPS, which we call STRIPS+, where certain STRIPS action arguments can be left implicit in preconditions which can also involve a limited form of existential quantification. The learning problem becomes the problem of learning STRIPS+ models from STRIPS+ state-action traces. For this, the proposed learning algorithm, called SYNTH, constructs a stratified sequence (conjunction) of precondition expressions or ``queries'' for each action, that denote unique objects in the state and ground the implicit action arguments in STRIPS+. The correctness and completeness of SYNTH is established, and its scalability is tested on state-action traces obtained from STRIPS+ models derived from existing STRIPS domains. 

**Abstract (ZH)**: 学习仅基于瓷砖位置的状态-动作踪迹的提升STRIPS模型问题 

---
# A General Framework of Epistemic Forgetting and its Instantiation by Ranking Functions 

**Title (ZH)**: 知识遗忘的一般框架及其通过排名函数的具体实现 

**Authors**: Christoph Beierle, Alexander Hahn, Diana Howey, Gabriele Kern-Isberner, Kai Sauerwald  

**Link**: [PDF](https://arxiv.org/pdf/2508.21441)  

**Abstract**: Forgetting as a knowledge management operation deliberately ignores parts of the knowledge and beliefs of an agent, for various reasons. Forgetting has many facets, one may want to forget parts of the syntax, a proposition, or a conditional. In the literature, two main operators suitable for performing forgetting have been proposed and investigated in depth: First, variable elimination is a syntactical method that blends out certain atomic variables to focus on the rest of the language. It has been mainly used in the area of logic programming and answer set programming. Second, contraction in AGM belief revision theory effectively removes propositions from belief sets under logical deduction. Both operations rely mainly on classical logics. In this article, we take an epistemic perspective and study forgetting operations in epistemic states with richer semantic structures, but with clear links to propositional logic. This allows us to investigate what forgetting in the epistemic background means, thereby lifting well-known and novel forgetting operations to the epistemic level. We present five general types of epistemic forgetting and instantiate them with seven concrete forgetting operations for Spohn's ranking functions. We take inspiration from postulates of forgetting both from logic programming and AGM theory to propose a rich landscape of axioms for evaluating forgetting operations. Finally, we evaluate all concrete forgetting operations according to all postulates, leading to a novel comprehensive overview highlighting differences and commonalities among the forgetting operators. 

**Abstract (ZH)**: 作为知识管理操作的遗忘故意忽略代理的部分知识和信念，出于各种原因。遗忘有多个层面，人们可能想遗忘语法规则的一部分、一个命题或条件。在文献中，两种主要的操作符被提出并深入研究用于执行遗忘：首先，变量消除是一种语法方法，通过消除某些原子变量来集中于语言的其余部分。它主要在逻辑编程和回答集编程领域使用。其次，在AGM信念修订理论中，收缩有效地从信念集合中逻辑推导地移除命题。这两种操作主要依赖经典逻辑。在本文中，我们从认识论视角出发，研究具有更丰富语义结构的认识状态下的遗忘操作，但同时保有关于命题逻辑的清晰联系。这使我们能够探究认识论背景下遗忘的意义，并将广为人知和新颖的遗忘操作提升到认识论层面。我们提出了五种一般类型的认识论遗忘，并具体实例化了Spohn排序函数的七种遗忘操作。我们从逻辑编程和AGM理论的遗忘公理中汲取灵感，提出了一个丰富的公理景观，用于评估遗忘操作。最后，我们根据所有公理评估所有具体遗忘操作，从而提出了一个新颖的综合概述，强调不同遗忘操作之间的差异和共同点。 

---
# CARJAN: Agent-Based Generation and Simulation of Traffic Scenarios with AJAN 

**Title (ZH)**: CARJAN: 基于代理的交通场景生成与模拟AJAN 

**Authors**: Leonard Frank Neis, Andre Antakli, Matthias Klusch  

**Link**: [PDF](https://arxiv.org/pdf/2508.21411)  

**Abstract**: User-friendly modeling and virtual simulation of urban traffic scenarios with different types of interacting agents such as pedestrians, cyclists and autonomous vehicles remains a challenge. We present CARJAN, a novel tool for semi-automated generation and simulation of such scenarios based on the multi-agent engineering framework AJAN and the driving simulator CARLA. CARJAN provides a visual user interface for the modeling, storage and maintenance of traffic scenario layouts, and leverages SPARQL Behavior Tree-based decision-making and interactions for agents in dynamic scenario simulations in CARLA. CARJAN provides a first integrated approach for interactive, intelligent agent-based generation and simulation of virtual traffic scenarios in CARLA. 

**Abstract (ZH)**: 基于不同的交互代理类型（如行人、骑行者和自动驾驶车辆）的城市交通场景友好建模与虚拟仿真仍然是一个挑战。我们提出了CARJAN，一种基于多代理工程框架AJAN和驾驶模拟器CARLA的半自动化生成与仿真工具。CARJAN提供了一个可视化的用户界面用于交通场景布局的建模、存储与维护，并利用SPARQL行为树驱动的决策与交互，在CARLA的动态场景仿真中实现代理的交互与智能行为。CARJAN提供了在CARLA中交互式、智能代理生成与仿真虚拟交通场景的首个集成方法。 

---
# AI Compute Architecture and Evolution Trends 

**Title (ZH)**: AI计算架构与 evolution trends 演化趋势 

**Authors**: Bor-Sung Liang  

**Link**: [PDF](https://arxiv.org/pdf/2508.21394)  

**Abstract**: The focus of AI development has shifted from academic research to practical applications. However, AI development faces numerous challenges at various levels. This article will attempt to analyze the opportunities and challenges of AI from several different perspectives using a structured approach. This article proposes a seven-layer model for AI compute architecture, including Physical Layer, Link Layer, Neural Network Layer, Context Layer, Agent Layer, Orchestrator Layer, and Application Layer, from bottom to top. It also explains how AI computing has evolved into this 7-layer architecture through the three-stage evolution on large-scale language models (LLMs). For each layer, we describe the development trajectory and key technologies. In Layers 1 and 2 we discuss AI computing issues and the impact of Scale-Up and Scale-Out strategies on computing architecture. In Layer 3 we explore two different development paths for LLMs. In Layer 4 we discuss the impact of contextual memory on LLMs and compares it to traditional processor memory. In Layers 5 to 7 we discuss the trends of AI agents and explore the issues in evolution from a single AI agent to an AI-based ecosystem, and their impact on the AI industry. Furthermore, AI development involves not only technical challenges but also the economic issues to build self-sustainable ecosystem. This article analyzes the internet industry to provide predictions on the future trajectory of AI development. 

**Abstract (ZH)**: AI发展从学术研究转向实际应用：七层架构下的机遇与挑战 

---
# AHELM: A Holistic Evaluation of Audio-Language Models 

**Title (ZH)**: AHELM：音频-语言模型的整体评估 

**Authors**: Tony Lee, Haoqin Tu, Chi Heem Wong, Zijun Wang, Siwei Yang, Yifan Mai, Yuyin Zhou, Cihang Xie, Percy Liang  

**Link**: [PDF](https://arxiv.org/pdf/2508.21376)  

**Abstract**: Evaluations of audio-language models (ALMs) -- multimodal models that take interleaved audio and text as input and output text -- are hindered by the lack of standardized benchmarks; most benchmarks measure only one or two capabilities and omit evaluative aspects such as fairness or safety. Furthermore, comparison across models is difficult as separate evaluations test a limited number of models and use different prompting methods and inference parameters. To address these shortfalls, we introduce AHELM, a benchmark that aggregates various datasets -- including 2 new synthetic audio-text datasets called PARADE, which evaluates the ALMs on avoiding stereotypes, and CoRe-Bench, which measures reasoning over conversational audio through inferential multi-turn question answering -- to holistically measure the performance of ALMs across 10 aspects we have identified as important to the development and usage of ALMs: audio perception, knowledge, reasoning, emotion detection, bias, fairness, multilinguality, robustness, toxicity, and safety. We also standardize the prompts, inference parameters, and evaluation metrics to ensure equitable comparisons across models. We test 14 open-weight and closed-API ALMs from 3 developers and 3 additional simple baseline systems each consisting of an automatic speech recognizer and a language model. Our results show that while Gemini 2.5 Pro ranks top in 5 out of 10 aspects, it exhibits group unfairness ($p=0.01$) on ASR tasks whereas most of the other models do not. We also find that the baseline systems perform reasonably well on AHELM, with one ranking 5th overall despite having only speech-to-text capabilities. For transparency, all raw prompts, model generations, and outputs are available on our website at this https URL. AHELM is intended to be a living benchmark and new datasets and models will be added over time. 

**Abstract (ZH)**: 音语言模型（ALMs）的评估——接受交错音频和文本作为输入并输出文本的多模态模型——受到缺乏标准化基准的阻碍；大多数基准仅衡量一两种能力，并省略了诸如公平性或安全性之类的评估方面。此外，模型之间的比较困难，因为单独的评估测试的模型有限且使用不同的提示方法和推理参数。为了解决这些不足，我们引入了AHELM基准，整合了多种数据集，包括两个新的合成音频-文本数据集PARADE，用于评估ALMs避免刻板印象的能力，以及CoRe-Bench，用于通过推理多轮问答来衡量对话音频上的推理能力，以全面衡量ALMs在我们识别的十个重要方面（音频感知、知识、推理、情感检测、偏差、公平性、多语言性、鲁棒性、有毒内容和安全性）上的性能。我们还标准化了提示、推理参数和评估指标，以确保模型之间的公平比较。我们测试了来自3个开发者的14个开放权重和闭合API音语言模型，以及3个附加的简单基线系统，每个系统由自动语音识别器和语言模型组成。结果显示，虽然Gemini 2.5 Pro在10个方面中的5个方面排名第一，但在ASR任务中表现出组不公平性（$p=0.01$），而大多数其他模型则没有。我们还发现，基线系统在AHELM上的表现相当不错，其中一个系统排名第五，尽管只有语音到文本的能力。为透明起见，所有原始提示、模型生成和输出均在我们的网站上提供。AHELM旨在成为一种活基准，将来会不断添加新的数据集和模型。 

---
# Think in Games: Learning to Reason in Games via Reinforcement Learning with Large Language Models 

**Title (ZH)**: 在游戏中思考：通过大规模语言模型强化学习进行游戏推理学习 

**Authors**: Yi Liao, Yu Gu, Yuan Sui, Zining Zhu, Yifan Lu, Guohua Tang, Zhongqian Sun, Wei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.21365)  

**Abstract**: Large language models (LLMs) excel at complex reasoning tasks such as mathematics and coding, yet they frequently struggle with simple interactive tasks that young children perform effortlessly. This discrepancy highlights a critical gap between declarative knowledge (knowing about something) and procedural knowledge (knowing how to do something). Although traditional reinforcement learning (RL) agents can acquire procedural knowledge through environmental interaction, they often operate as black boxes and require substantial training data. In contrast, LLMs possess extensive world knowledge and reasoning capabilities, but are unable to effectively convert this static knowledge into dynamic decision-making in interactive settings. To address this challenge, we propose Think in Games (TiG), a novel framework that empowers LLMs to develop procedural understanding through direct interaction with game environments, while retaining their inherent reasoning and explanatory abilities. Specifically, TiG reformulates RL-based decision-making as a language modeling task: LLMs generate language-guided policies, which are refined iteratively through online reinforcement learning based on environmental feedback. Our experimental results show that TiG successfully bridges the gap between declarative and procedural knowledge, achieving competitive performance with dramatically lower data and computational demands compared to conventional RL methods. Moreover, TiG provides step-by-step natural language explanations for its decisions, greatly improving transparency and interpretability in complex interactive tasks. 

**Abstract (ZH)**: 大型语言模型通过游戏增强程序性理解：Think in Games框架 

---
# Multi-Ontology Integration with Dual-Axis Propagation for Medical Concept Representation 

**Title (ZH)**: 基于双轴传播的多本体集成与医学概念表示 

**Authors**: Mohsen Nayebi Kerdabadi, Arya Hadizadeh Moghaddam, Dongjie Wang, Zijun Yao  

**Link**: [PDF](https://arxiv.org/pdf/2508.21320)  

**Abstract**: Medical ontology graphs map external knowledge to medical codes in electronic health records via structured relationships. By leveraging domain-approved connections (e.g., parent-child), predictive models can generate richer medical concept representations by incorporating contextual information from related concepts. However, existing literature primarily focuses on incorporating domain knowledge from a single ontology system, or from multiple ontology systems (e.g., diseases, drugs, and procedures) in isolation, without integrating them into a unified learning structure. Consequently, concept representation learning often remains limited to intra-ontology relationships, overlooking cross-ontology connections. In this paper, we propose LINKO, a large language model (LLM)-augmented integrative ontology learning framework that leverages multiple ontology graphs simultaneously by enabling dual-axis knowledge propagation both within and across heterogeneous ontology systems to enhance medical concept representation learning. Specifically, LINKO first employs LLMs to provide a graph-retrieval-augmented initialization for ontology concept embedding, through an engineered prompt that includes concept descriptions, and is further augmented with ontology context. Second, our method jointly learns the medical concepts in diverse ontology graphs by performing knowledge propagation in two axes: (1) intra-ontology vertical propagation across hierarchical ontology levels and (2) inter-ontology horizontal propagation within every level in parallel. Last, through extensive experiments on two public datasets, we validate the superior performance of LINKO over state-of-the-art baselines. As a plug-in encoder compatible with existing EHR predictive models, LINKO further demonstrates enhanced robustness in scenarios involving limited data availability and rare disease prediction. 

**Abstract (ZH)**: 多 ontology 图辅助的大型语言模型增强集成本体学习框架：提升医疗概念表示学习 

---
# MultiFluxAI Enhancing Platform Engineering with Advanced Agent-Orchestrated Retrieval Systems 

**Title (ZH)**: MultiFluxAI 提升平台工程的高级代理 orchestrated 检索系统增强平台 

**Authors**: Sri Ram Macharla, Sridhar Murthy J, Anjaneyulu Pasala  

**Link**: [PDF](https://arxiv.org/pdf/2508.21307)  

**Abstract**: MultiFluxAI is an innovative AI platform developed to address the challenges of managing and integrating vast, disparate data sources in product engineering across application domains. It addresses both current and new service related queries that enhance user engagement in the digital ecosystem. This platform leverages advanced AI techniques, such as Generative AI, vectorization, and agentic orchestration to provide dynamic and context-aware responses to complex user queries. 

**Abstract (ZH)**: MultiFluxAI是一个创新的AI平台，旨在应对产品工程跨应用领域中管理与整合大量异构数据源的挑战。该平台通过利用生成式AI、向量化和智能orchestration等先进技术，提供动态且上下文相关的复杂用户查询响应，增强数字生态系统中的用户参与。 

---
# Addressing accuracy and hallucination of LLMs in Alzheimer's disease research through knowledge graphs 

**Title (ZH)**: 通过知识图谱解决阿尔茨海默病研究中大语言模型的准确性和幻觉问题 

**Authors**: Tingxuan Xu, Jiarui Feng, Justin Melendez, Kaleigh Roberts, Donghong Cai, Mingfang Zhu, Donald Elbert, Yixin Chen, Randall J. Bateman  

**Link**: [PDF](https://arxiv.org/pdf/2508.21238)  

**Abstract**: In the past two years, large language model (LLM)-based chatbots, such as ChatGPT, have revolutionized various domains by enabling diverse task completion and question-answering capabilities. However, their application in scientific research remains constrained by challenges such as hallucinations, limited domain-specific knowledge, and lack of explainability or traceability for the response. Graph-based Retrieval-Augmented Generation (GraphRAG) has emerged as a promising approach to improving chatbot reliability by integrating domain-specific contextual information before response generation, addressing some limitations of standard LLMs. Despite its potential, there are only limited studies that evaluate GraphRAG on specific domains that require intensive knowledge, like Alzheimer's disease or other biomedical domains. In this paper, we assess the quality and traceability of two popular GraphRAG systems. We compile a database of 50 papers and 70 expert questions related to Alzheimer's disease, construct a GraphRAG knowledge base, and employ GPT-4o as the LLM for answering queries. We then compare the quality of responses generated by GraphRAG with those from a standard GPT-4o model. Additionally, we discuss and evaluate the traceability of several Retrieval-Augmented Generation (RAG) and GraphRAG systems. Finally, we provide an easy-to-use interface with a pre-built Alzheimer's disease database for researchers to test the performance of both standard RAG and GraphRAG. 

**Abstract (ZH)**: 过去两年，基于大型语言模型（LLM）的聊天机器人，如ChatGPT，通过实现多样化的任务完成和问答能力，颠覆了多个领域。然而，它们在科学研究中的应用仍受到幻觉、领域特定知识有限以及响应缺乏可解释性和可追溯性等挑战的制约。基于图的检索增强生成（GraphRAG）作为一种方法，通过在响应生成前整合领域特定的上下文信息，有望改善聊天机器人的可靠性，从而解决标准LLM的一些局限性。尽管具有潜力，但在需要密集专业知识的具体领域（如阿尔茨海默病或其它生物医药领域）中，仅有限的研究对GraphRAG进行了评估。在本文中，我们评估了两种流行GraphRAG系统的质量和可追溯性。我们收集了50篇与阿尔茨海默病相关的论文和70个专家问题，构建了一个GraphRAG知识库，并使用GPT-4o作为LLM以回答查询。随后，我们将GraphRAG生成的响应质量与标准GPT-4o模型的响应质量进行了比较。此外，我们讨论并评估了几种检索增强生成（RAG）和GraphRAG系统的可追溯性。最后，我们提供了一个易于使用的界面，内置了阿尔茨海默病数据库，以供研究人员测试标准RAG和GraphRAG的性能。 

---
# Fuzzy, Symbolic, and Contextual: Enhancing LLM Instruction via Cognitive Scaffolding 

**Title (ZH)**: 模糊性、符号性与情境性：通过认知支架增强大语言模型指令训练 

**Authors**: Vanessa Figueiredo  

**Link**: [PDF](https://arxiv.org/pdf/2508.21204)  

**Abstract**: We study how architectural inductive biases influence the cognitive behavior of large language models (LLMs) in instructional dialogue. We introduce a symbolic scaffolding mechanism paired with a short-term memory schema designed to promote adaptive, structured reasoning in Socratic tutoring. Using controlled ablation across five system variants, we evaluate model outputs via expert-designed rubrics covering scaffolding, responsiveness, symbolic reasoning, and conversational memory. We present preliminary results using an LLM-based evaluation framework aligned to a cognitively grounded rubric. This enables scalable, systematic comparisons across architectural variants in early-stage experimentation. The preliminary results show that our full system consistently outperforms baseline variants. Analysis reveals that removing memory or symbolic structure degrades key cognitive behaviors, including abstraction, adaptive probing, and conceptual continuity. These findings support a processing-level account in which architectural scaffolds can reliably shape emergent instructional strategies in LLMs. 

**Abstract (ZH)**: 我们研究了建筑学诱导偏见如何影响大型语言模型（LLM）在教学对话中的认知行为。我们引入了一种符号支撑机制，结合一种短期记忆架构，旨在促进苏格拉底式辅导中的适应性和结构化推理。通过跨越五种系统变体的受控剥离分析，我们使用由专家设计的标准来评估模型输出，涵盖支撑、响应性、符号推理和对话记忆。我们采用与认知基础标准相一致的LLM评估框架呈现初步结果。这使得在早期实验中能够进行可扩展和系统的架构变体比较。初步结果表明，我们的完整系统始终优于基线变体。分析表明，移除记忆或符号结构会降低抽象、适应性探查和概念连续性等关键认知行为。这些发现支持一个处理层面的解释，即架构支撑可以可靠地塑造LLM中的 emergent 教学策略。 

---
# The Demon is in Ambiguity: Revisiting Situation Recognition with Single Positive Multi-Label Learning 

**Title (ZH)**: demon存在于不确定性中：重新审视基于单一正样本多标签学习的情境识别 

**Authors**: Yiming Lin, Yuchen Niu, Shang Wang, Kaizhu Huang, Qiufeng Wang, Xiao-Bo Jin  

**Link**: [PDF](https://arxiv.org/pdf/2508.21816)  

**Abstract**: Context recognition (SR) is a fundamental task in computer vision that aims to extract structured semantic summaries from images by identifying key events and their associated entities. Specifically, given an input image, the model must first classify the main visual events (verb classification), then identify the participating entities and their semantic roles (semantic role labeling), and finally localize these entities in the image (semantic role localization). Existing methods treat verb classification as a single-label problem, but we show through a comprehensive analysis that this formulation fails to address the inherent ambiguity in visual event recognition, as multiple verb categories may reasonably describe the same image. This paper makes three key contributions: First, we reveal through empirical analysis that verb classification is inherently a multi-label problem due to the ubiquitous semantic overlap between verb categories. Second, given the impracticality of fully annotating large-scale datasets with multiple labels, we propose to reformulate verb classification as a single positive multi-label learning (SPMLL) problem - a novel perspective in SR research. Third, we design a comprehensive multi-label evaluation benchmark for SR that is carefully designed to fairly evaluate model performance in a multi-label setting. To address the challenges of SPMLL, we futher develop the Graph Enhanced Verb Multilayer Perceptron (GE-VerbMLP), which combines graph neural networks to capture label correlations and adversarial training to optimize decision boundaries. Extensive experiments on real-world datasets show that our approach achieves more than 3\% MAP improvement while remaining competitive on traditional top-1 and top-5 accuracy metrics. 

**Abstract (ZH)**: 基于上下文识别的动词分类（多标签问题）：一种新的研究视角 

---
# DynaMark: A Reinforcement Learning Framework for Dynamic Watermarking in Industrial Machine Tool Controllers 

**Title (ZH)**: DynaMark：工业机床控制器中动态水印的强化学习框架 

**Authors**: Navid Aftabi, Abhishek Hanchate, Satish Bukkapatnam, Dan Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.21797)  

**Abstract**: Industry 4.0's highly networked Machine Tool Controllers (MTCs) are prime targets for replay attacks that use outdated sensor data to manipulate actuators. Dynamic watermarking can reveal such tampering, but current schemes assume linear-Gaussian dynamics and use constant watermark statistics, making them vulnerable to the time-varying, partly proprietary behavior of MTCs. We close this gap with DynaMark, a reinforcement learning framework that models dynamic watermarking as a Markov decision process (MDP). It learns an adaptive policy online that dynamically adapts the covariance of a zero-mean Gaussian watermark using available measurements and detector feedback, without needing system knowledge. DynaMark maximizes a unique reward function balancing control performance, energy consumption, and detection confidence dynamically. We develop a Bayesian belief updating mechanism for real-time detection confidence in linear systems. This approach, independent of specific system assumptions, underpins the MDP for systems with linear dynamics. On a Siemens Sinumerik 828D controller digital twin, DynaMark achieves a reduction in watermark energy by 70% while preserving the nominal trajectory, compared to constant variance baselines. It also maintains an average detection delay equivalent to one sampling interval. A physical stepper-motor testbed validates these findings, rapidly triggering alarms with less control performance decline and exceeding existing benchmarks. 

**Abstract (ZH)**: Industry 4.0中高度网络化的机床控制器是重放攻击的主要目标，这些攻击利用过时的传感器数据操控执行器。动态水印可以揭示此类篡改，但现有方案假设线性高斯动态，并采用恒定的水印统计值，使其容易受到机床控制器时间变化且部分专有的行为影响。我们通过DynaMark填补这一空白，DynaMark是一个基于强化学习的框架，将其动态水印建模为马尔可夫决策过程（MDP）。该框架在线学习自适应策略，动态调整零均值高斯水印的协方差，无需系统知识。DynaMark最大化一个独特的奖励函数，该函数平衡控制性能、能量消耗和检测置信度，使其能够动态变化。我们为线性系统开发了一种贝叶斯信念更新机制，以实现实时检测置信度。这种方法不依赖于特定系统假设，为具有线性动力学的系统提供了MDP基础。在西门子Sinumerik 828D控制器数字孪生中，DynaMark在保持名义轨迹的同时将水印能量降低了70%，与恒定方差基线相比。它还维持了相当于一个采样间隔的平均检测延迟。物理步进电机测试平台验证了这些发现，能够在控制性能下降更少的情况下迅速触发警报，超过了现有基准。 

---
# TMUAD: Enhancing Logical Capabilities in Unified Anomaly Detection Models with a Text Memory Bank 

**Title (ZH)**: TMUAD：通过文本记忆库增强统一异常检测模型的逻辑能力 

**Authors**: Jiawei Liu, Jiahe Hou, Wei Wang, Jinsong Du, Yang Cong, Huijie Fan  

**Link**: [PDF](https://arxiv.org/pdf/2508.21795)  

**Abstract**: Anomaly detection, which aims to identify anomalies deviating from normal patterns, is challenging due to the limited amount of normal data available. Unlike most existing unified methods that rely on carefully designed image feature extractors and memory banks to capture logical relationships between objects, we introduce a text memory bank to enhance the detection of logical anomalies. Specifically, we propose a Three-Memory framework for Unified structural and logical Anomaly Detection (TMUAD). First, we build a class-level text memory bank for logical anomaly detection by the proposed logic-aware text extractor, which can capture rich logical descriptions of objects from input images. Second, we construct an object-level image memory bank that preserves complete object contours by extracting features from segmented objects. Third, we employ visual encoders to extract patch-level image features for constructing a patch-level memory bank for structural anomaly detection. These three complementary memory banks are used to retrieve and compare normal images that are most similar to the query image, compute anomaly scores at multiple levels, and fuse them into a final anomaly score. By unifying structural and logical anomaly detection through collaborative memory banks, TMUAD achieves state-of-the-art performance across seven publicly available datasets involving industrial and medical domains. The model and code are available at this https URL. 

**Abstract (ZH)**: 统一结构与逻辑异常检测的三记忆库框架（TMUAD） 

---
# MoE-Health: A Mixture of Experts Framework for Robust Multimodal Healthcare Prediction 

**Title (ZH)**: MoE-Health：一种混合专家框架以实现稳健的多模态医疗预测 

**Authors**: Xiaoyang Wang, Christopher C. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.21793)  

**Abstract**: Healthcare systems generate diverse multimodal data, including Electronic Health Records (EHR), clinical notes, and medical images. Effectively leveraging this data for clinical prediction is challenging, particularly as real-world samples often present with varied or incomplete modalities. Existing approaches typically require complete modality data or rely on manual selection strategies, limiting their applicability in real-world clinical settings where data availability varies across patients and institutions. To address these limitations, we propose MoE-Health, a novel Mixture of Experts framework designed for robust multimodal fusion in healthcare prediction. MoE-Health architecture is specifically developed to handle samples with differing modalities and improve performance on critical clinical tasks. By leveraging specialized expert networks and a dynamic gating mechanism, our approach dynamically selects and combines relevant experts based on available data modalities, enabling flexible adaptation to varying data availability scenarios. We evaluate MoE-Health on the MIMIC-IV dataset across three critical clinical prediction tasks: in-hospital mortality prediction, long length of stay, and hospital readmission prediction. Experimental results demonstrate that MoE-Health achieves superior performance compared to existing multimodal fusion methods while maintaining robustness across different modality availability patterns. The framework effectively integrates multimodal information, offering improved predictive performance and robustness in handling heterogeneous and incomplete healthcare data, making it particularly suitable for deployment in diverse healthcare environments with heterogeneous data availability. 

**Abstract (ZH)**: MoE-Health: 一种用于医疗预测的新型专家混合框架 

---
# Going over Fine Web with a Fine-Tooth Comb: Technical Report of Indexing Fine Web for Problematic Content Search and Retrieval 

**Title (ZH)**: 细梳详查细网：问题内容搜索与检索的索引技术报告 

**Authors**: Inés Altemir Marinas, Anastasiia Kucherenko, Andrei Kucharavy  

**Link**: [PDF](https://arxiv.org/pdf/2508.21788)  

**Abstract**: Large language models (LLMs) rely heavily on web-scale datasets like Common Crawl, which provides over 80\% of training data for some modern models. However, the indiscriminate nature of web crawling raises challenges in data quality, safety, and ethics. Despite the critical importance of training data quality, prior research on harmful content has been limited to small samples due to computational constraints. This project presents a framework for indexing and analyzing LLM training datasets using an ElasticSearch-based pipeline. We apply it to SwissAI's FineWeb-2 corpus (1.5TB, four languages), achieving fast query performance--most searches in milliseconds, all under 2 seconds. Our work demonstrates real-time dataset analysis, offering practical tools for safer, more accountable AI systems. 

**Abstract (ZH)**: 大规模语言模型（LLMs） heavily依赖像Common Crawl这样的Web规模数据集，为某些现代模型提供了超过80%的训练数据。然而，无差别的Web抓取性质在数据质量、安全性和伦理方面提出了挑战。尽管训练数据质量至关重要，但由于计算限制，前期研究针对有害内容的样本有限。本项目提出了一种使用基于ElasticSearch的管道进行索引和分析LLMs训练数据集的框架。我们将该框架应用于SwissAI的FineWeb-2语料库（1.5TB，四种语言），实现了快速查询性能——大多数查询在毫秒级，所有查询在2秒以内。我们的工作展示了实时数据集分析，提供了更安全、更负责任的AI系统的实用工具。 

---
# PiCSAR: Probabilistic Confidence Selection And Ranking 

**Title (ZH)**: PiCSAR: 概率置信度选择和排序 

**Authors**: Joshua Ong Jun Leang, Zheng Zhao, Aryo Pradipta Gema, Sohee Yang, Wai-Chung Kwan, Xuanli He, Wenda Li, Pasquale Minervini, Eleonora Giunchiglia, Shay B. Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2508.21787)  

**Abstract**: Best-of-n sampling improves the accuracy of large language models (LLMs) and large reasoning models (LRMs) by generating multiple candidate solutions and selecting the one with the highest reward. The key challenge for reasoning tasks is designing a scoring function that can identify correct reasoning chains without access to ground-truth answers. We propose Probabilistic Confidence Selection And Ranking (PiCSAR): a simple, training-free method that scores each candidate generation using the joint log-likelihood of the reasoning and final answer. The joint log-likelihood of the reasoning and final answer naturally decomposes into reasoning confidence and answer confidence. PiCSAR achieves substantial gains across diverse benchmarks (+10.18 on MATH500, +9.81 on AIME2025), outperforming baselines with at least 2x fewer samples in 16 out of 20 comparisons. Our analysis reveals that correct reasoning chains exhibit significantly higher reasoning and answer confidence, justifying the effectiveness of PiCSAR. 

**Abstract (ZH)**: Best-of-n 抽样通过生成多个候选解决方案并选择得分最高的一个，提高了大规模语言模型（LLMs）和大规模推理模型（LRMs）的准确性。对于推理任务的关键挑战是设计一个评分函数，能够在不访问真实答案的情况下识别正确的推理链。我们提出了一种无需训练的简单方法 Probabilistic Confidence Selection And Ranking (PiCSAR)：该方法使用推理和最终答案的联合对数似然性对每个候选生成进行评分。推理和最终答案的联合对数似然性自然地分解为推理置信度和答案置信度。PiCSAR 在多种基准测试中取得了显著提升（MATH500 上提升了 10.18，AIME2025 上提升了 9.81），在 20 次比较中有 16 次超过了基线模型，且使用样本的数量至少减少了 2 倍。我们的分析表明，正确的推理链显示出显著更高的推理和答案置信度，这证明了 PiCSAR 的有效性。 

---
# Benchmarking GPT-5 in Radiation Oncology: Measurable Gains, but Persistent Need for Expert Oversight 

**Title (ZH)**: 在放射肿瘤学中基准测试GPT-5：可测量的收益，但仍需持续的专业监督 

**Authors**: Ugur Dinc, Jibak Sarkar, Philipp Schubert, Sabine Semrau, Thomas Weissmann, Andre Karius, Johann Brand, Bernd-Niklas Axer, Ahmed Gomaa, Pluvio Stephan, Ishita Sheth, Sogand Beirami, Annette Schwarz, Udo Gaipl, Benjamin Frey, Christoph Bert, Stefanie Corradini, Rainer Fietkau, Florian Putz  

**Link**: [PDF](https://arxiv.org/pdf/2508.21777)  

**Abstract**: Introduction: Large language models (LLM) have shown great potential in clinical decision support. GPT-5 is a novel LLM system that has been specifically marketed towards oncology use.
Methods: Performance was assessed using two complementary benchmarks: (i) the ACR Radiation Oncology In-Training Examination (TXIT, 2021), comprising 300 multiple-choice items, and (ii) a curated set of 60 authentic radiation oncologic vignettes representing diverse disease sites and treatment indications. For the vignette evaluation, GPT-5 was instructed to generate concise therapeutic plans. Four board-certified radiation oncologists rated correctness, comprehensiveness, and hallucinations. Inter-rater reliability was quantified using Fleiss' \k{appa}.
Results: On the TXIT benchmark, GPT-5 achieved a mean accuracy of 92.8%, outperforming GPT-4 (78.8%) and GPT-3.5 (62.1%). Domain-specific gains were most pronounced in Dose and Diagnosis. In the vignette evaluation, GPT-5's treatment recommendations were rated highly for correctness (mean 3.24/4, 95% CI: 3.11-3.38) and comprehensiveness (3.59/4, 95% CI: 3.49-3.69). Hallucinations were rare with no case reaching majority consensus for their presence. Inter-rater agreement was low (Fleiss' \k{appa} 0.083 for correctness), reflecting inherent variability in clinical judgment. Errors clustered in complex scenarios requiring precise trial knowledge or detailed clinical adaptation.
Discussion: GPT-5 clearly outperformed prior model variants on the radiation oncology multiple-choice benchmark. Although GPT-5 exhibited favorable performance in generating real-world radiation oncology treatment recommendations, correctness ratings indicate room for further improvement. While hallucinations were infrequent, the presence of substantive errors underscores that GPT-5-generated recommendations require rigorous expert oversight before clinical implementation. 

**Abstract (ZH)**: Introduction: 大型语言模型(LLM)在临床决策支持方面显示出巨大潜力。GPT-5是一种专门针对肿瘤学使用的新颖LLM系统。

Methods: 性能评估使用了两个互补的标准：(i) ACR放射肿瘤学在职考试(TXIT, 2021)，包含300个多选题；(ii) 一个精心挑选的包含60个真实放射肿瘤学案例集，涵盖了多种疾病部位和治疗适应症。对于案例评估，GPT-5被指示生成简明的治疗方案。四位放射肿瘤学专科医生评估了治疗方案的正确性、全面性和虚构内容。信度使用Fleiss的\kappa进行量化。

Results: 在TXIT基准测试中，GPT-5的平均正确率为92.8%，Performance优于GPT-4（78.8%）和GPT-3.5（62.1%）。在专业领域内，收益最显著的是剂量和诊断。在案例评估中，GPT-5的治疗建议在正确性（平均3.24/4，95% CI: 3.11-3.38）和全面性（3.59/4，95% CI: 3.49-3.69）方面得到了高度评价。虚构内容很少见，没有一例达到了多数共识。正确性评定的信度较低（Fleiss的\kappa为0.083），反映出临床判断的固有变异性。错误多出现在需要精准实验知识或详细临床适应的复杂场景中。

Discussion: GPT-5在放射肿瘤学多项选择基准测试中的表现明显优于先前的模型版本。尽管GPT-5在生成真实世界放射肿瘤学治疗建议方面的表现优异，但正确性评分表明有进一步改进的空间。虽然虚构内容很少见，实质性错误的存在表明，在临床实施前，GPT-5生成的建议需要严格的专家监督。 

---
# Unsupervised Video Continual Learning via Non-Parametric Deep Embedded Clustering 

**Title (ZH)**: 无监督视频连续学习 via 非参数深嵌入聚类 

**Authors**: Nattapong Kurpukdee, Adrian G. Bors  

**Link**: [PDF](https://arxiv.org/pdf/2508.21773)  

**Abstract**: We propose a realistic scenario for the unsupervised video learning where neither task boundaries nor labels are provided when learning a succession of tasks. We also provide a non-parametric learning solution for the under-explored problem of unsupervised video continual learning. Videos represent a complex and rich spatio-temporal media information, widely used in many applications, but which have not been sufficiently explored in unsupervised continual learning. Prior studies have only focused on supervised continual learning, relying on the knowledge of labels and task boundaries, while having labeled data is costly and not practical. To address this gap, we study the unsupervised video continual learning (uVCL). uVCL raises more challenges due to the additional computational and memory requirements of processing videos when compared to images. We introduce a general benchmark experimental protocol for uVCL by considering the learning of unstructured video data categories during each task. We propose to use the Kernel Density Estimation (KDE) of deep embedded video features extracted by unsupervised video transformer networks as a non-parametric probabilistic representation of the data. We introduce a novelty detection criterion for the incoming new task data, dynamically enabling the expansion of memory clusters, aiming to capture new knowledge when learning a succession of tasks. We leverage the use of transfer learning from the previous tasks as an initial state for the knowledge transfer to the current learning task. We found that the proposed methodology substantially enhances the performance of the model when successively learning many tasks. We perform in-depth evaluations on three standard video action recognition datasets, including UCF101, HMDB51, and Something-to-Something V2, without using any labels or class boundaries. 

**Abstract (ZH)**: 我们提出了一种无监督视频学习的现实场景，在学习一系列任务时，既不提供任务边界也不提供标签。我们还提供了一种非参数学习解决方案，解决了无监督视频连续学习这一未充分探索的问题。视频代表了一种复杂且丰富的时空媒体信息，在许多应用中被广泛使用，但在无监督连续学习中尚未得到充分探索。 prior研究仅关注有监督连续学习，依赖于标签和任务边界的知识，而标注数据的成本高昂且不切实际。为了解决这一问题，我们研究了无监督视频连续学习(uVCL)。uVCL由于处理视频时所需额外的计算和内存资源，相较于图像具有更大的挑战性。我们通过在每次任务中考虑无结构视频数据类别的学习，引入了一种通用的基准实验协议。我们提议使用无监督视频转换器网络提取的深度嵌入视频特征的核密度估计(KDE)作为数据的非参数概率表示。我们引入了一种新任务数据的新颖性检测标准，动态扩展记忆簇，旨在学习一系列任务时捕获新知识。我们利用从先前任务的迁移学习作为当前学习任务知识传递的初始状态。我们发现所提出的方法在相继学习多个任务时显著提高了模型的性能。我们在三个标准视频动作识别数据集上进行了深入评估，包括UCF101、HMDB51和Something-to-Something V2，且未使用任何标签或类边界。 

---
# Reasoning-Intensive Regression 

**Title (ZH)**: 密集推理回归 

**Authors**: Diane Tchuindjo, Omar Khattab  

**Link**: [PDF](https://arxiv.org/pdf/2508.21762)  

**Abstract**: AI researchers and practitioners increasingly apply large language models (LLMs) to what we call reasoning-intensive regression (RiR), i.e. deducing subtle numerical properties from text. Unlike standard language regression tasks, e.g. for sentiment or similarity, RiR often appears instead in ad-hoc problems like rubric-based scoring or domain-specific retrieval, where much deeper analysis of text is required while only limited task-specific training data and computation are available. We cast three realistic problems as RiR tasks to establish an initial benchmark, and use that to test our hypothesis that prompting frozen LLMs and finetuning Transformer encoders via gradient descent will both often struggle in RiR. We then propose MENTAT, a simple and lightweight method that combines batch-reflective prompt optimization with neural ensemble learning. MENTAT achieves up to 65% improvement over both baselines, though substantial room remains for future advances in RiR. 

**Abstract (ZH)**: AI研究人员和 practitioners 逐渐将大规模语言模型 (LLMs) 应用于我们称之为推理密集型回归 (RiR) 的任务中，即从文本中推导出细微的数值属性。与标准语言回归任务，如情感分析或相似度分析不同，RiR 经常出现在如评分表评分或领域特定检索等即兴问题中，这些任务需要更深入的文本分析，但仅有有限的任务特定训练数据和计算资源可用。我们将三个现实问题作为 RiR 任务以建立初始基准，并通过此基准测试我们的假设：冻结的 LLMs 在提示调优和通过梯度下降微调 Transformer 编码器时通常都会在 RiR 中遇到困难。随后，我们提出了 MENTAT，一种简单且轻量的方法，结合了批量反思提示优化与神经集成学习。尽管如此，RiR 仍然有巨大的未来进步空间，MENTAT 相较于基准方法实现了高达 65% 的性能提升。 

---
# Neural Network Acceleration on MPSoC board: Integrating SLAC's SNL, Rogue Software and Auto-SNL 

**Title (ZH)**: 基于MPSoC板的神经网络加速：整合SLAC的SNL、Rogue Software和Auto-SNL 

**Authors**: Hamza Ezzaoui Rahali, Abhilasha Dave, Larry Ruckman, Mohammad Mehdi Rahimifar, Audrey C. Therrien, James J. Russel, Ryan T. Herbst  

**Link**: [PDF](https://arxiv.org/pdf/2508.21739)  

**Abstract**: The LCLS-II Free Electron Laser (FEL) will generate X-ray pulses for beamline experiments at rates of up to 1~MHz, with detectors producing data throughputs exceeding 1 TB/s. Managing such massive data streams presents significant challenges, as transmission and storage infrastructures become prohibitively expensive. Machine learning (ML) offers a promising solution for real-time data reduction, but conventional implementations introduce excessive latency, making them unsuitable for high-speed experimental environments. To address these challenges, SLAC developed the SLAC Neural Network Library (SNL), a specialized framework designed to deploy real-time ML inference models on Field-Programmable Gate Arrays (FPGA). SNL's key feature is the ability to dynamically update model weights without requiring FPGA resynthesis, enhancing flexibility for adaptive learning applications. To further enhance usability and accessibility, we introduce Auto-SNL, a Python extension that streamlines the process of converting Python-based neural network models into SNL-compatible high-level synthesis code. This paper presents a benchmark comparison against hls4ml, the current state-of-the-art tool, across multiple neural network architectures, fixed-point precisions, and synthesis configurations targeting a Xilinx ZCU102 FPGA. The results showed that SNL achieves competitive or superior latency in most tested architectures, while in some cases also offering FPGA resource savings. This adaptation demonstrates SNL's versatility, opening new opportunities for researchers and academics in fields such as high-energy physics, medical imaging, robotics, and many more. 

**Abstract (ZH)**: SLAC自由电子激光器（FEL）LCLS-II将生成高达1~MHz的X射线脉冲用于束线实验，伴随而来的数据量超过1 TB/s，这给数据传输和存储基础设施带来了巨大挑战。机器学习（ML）提供了一种实时数据减量的前景方案，但传统的实现方式引入了过高的延迟，使之不适合高速实验环境。为应对这些挑战，SLAC开发了SLAC神经网络库（SLAC Neural Network Library, SNL），这是一种专门框架，旨在将实时ML推理模型部署在现场可编程门阵列（FPGA）上。SNL的关键特性是能够在不requiring FPGA重新综合的情况下动态更新模型权重，增强自适应学习应用的灵活性。为了进一步提高使用性和可访问性，我们引入了Auto-SNL，这是一种Python扩展，简化了将基于Python的神经网络模型转换为SNL兼容的高级综合代码的过程。本文在Xilinx ZCU102 FPGA上针对多种神经网络架构、定点精度和综合配置，将SNL与当前最先进的工具hls4ml进行了基准比较。结果表明，在大多数测试架构中，SNL达到了具有竞争力或更优的延迟，而在某些情况下还提供了FPGA资源节省。这一适应性展示了SNL的灵活性，为高能物理、医学成像、机器人学等领域的研究人员和学者开辟了新的机遇。 

---
# Developer Insights into Designing AI-Based Computer Perception Tools 

**Title (ZH)**: 开发者对设计基于人工智能的计算机感知工具的见解 

**Authors**: Maya Guhan, Meghan E. Hurley, Eric A. Storch, John Herrington, Casey Zampella, Julia Parish-Morris, Gabriel Lázaro-Muñoz, Kristin Kostick-Quenet  

**Link**: [PDF](https://arxiv.org/pdf/2508.21733)  

**Abstract**: Artificial intelligence (AI)-based computer perception (CP) technologies use mobile sensors to collect behavioral and physiological data for clinical decision-making. These tools can reshape how clinical knowledge is generated and interpreted. However, effective integration of these tools into clinical workflows depends on how developers balance clinical utility with user acceptability and trustworthiness. Our study presents findings from 20 in-depth interviews with developers of AI-based CP tools. Interviews were transcribed and inductive, thematic analysis was performed to identify 4 key design priorities: 1) to account for context and ensure explainability for both patients and clinicians; 2) align tools with existing clinical workflows; 3) appropriately customize to relevant stakeholders for usability and acceptability; and 4) push the boundaries of innovation while aligning with established paradigms. Our findings highlight that developers view themselves as not merely technical architects but also ethical stewards, designing tools that are both acceptable by users and epistemically responsible (prioritizing objectivity and pushing clinical knowledge forward). We offer the following suggestions to help achieve this balance: documenting how design choices around customization are made, defining limits for customization choices, transparently conveying information about outputs, and investing in user training. Achieving these goals will require interdisciplinary collaboration between developers, clinicians, and ethicists. 

**Abstract (ZH)**: 基于人工智能的计算机感知技术：移动传感器收集生物行为和生理数据以促进临床决策。这些工具能够重塑临床知识的生成与解释方式。然而，这些工具的有效集成取决于开发者如何平衡临床效用、用户接受度和可信度。本研究基于对20名开发基于人工智能的计算机感知工具的开发者的深度访谈结果，识别出4项关键设计优先事项：1）考虑上下文并确保对患者和临床医生都具有解释性；2）使工具与现有的临床工作流程相契合；3）适当定制以适应相关利益相关者，提高可用性和接受度；4）创新边界同时与传统范式保持一致。研究结果表明，开发者不仅被视为技术架构师，也是伦理监护人，致力于设计既能被用户接受又能负责任地推进临床知识发展的工具。为此，我们提出以下建议：记录定制设计选择的过程、定义定制选择的限制、透明地传达输出信息以及投资用户培训。实现这些目标需要开发者、临床医生和伦理学家之间的跨学科合作。 

---
# CAD2DMD-SET: Synthetic Generation Tool of Digital Measurement Device CAD Model Datasets for fine-tuning Large Vision-Language Models 

**Title (ZH)**: CAD2DMD-SET：合成生成数字测量设备CAD模型数据集的工具，用于大型视觉-语言模型的微调 

**Authors**: João Valente, Atabak Dehban, Rodrigo Ventura  

**Link**: [PDF](https://arxiv.org/pdf/2508.21732)  

**Abstract**: Recent advancements in Large Vision-Language Models (LVLMs) have demonstrated impressive capabilities across various multimodal tasks. They continue, however, to struggle with trivial scenarios such as reading values from Digital Measurement Devices (DMDs), particularly in real-world conditions involving clutter, occlusions, extreme viewpoints, and motion blur; common in head-mounted cameras and Augmented Reality (AR) applications. Motivated by these limitations, this work introduces CAD2DMD-SET, a synthetic data generation tool designed to support visual question answering (VQA) tasks involving DMDs. By leveraging 3D CAD models, advanced rendering, and high-fidelity image composition, our tool produces diverse, VQA-labelled synthetic DMD datasets suitable for fine-tuning LVLMs. Additionally, we present DMDBench, a curated validation set of 1,000 annotated real-world images designed to evaluate model performance under practical constraints. Benchmarking three state-of-the-art LVLMs using Average Normalised Levenshtein Similarity (ANLS) and further fine-tuning LoRA's of these models with CAD2DMD-SET's generated dataset yielded substantial improvements, with InternVL showcasing a score increase of 200% without degrading on other tasks. This demonstrates that the CAD2DMD-SET training dataset substantially improves the robustness and performance of LVLMs when operating under the previously stated challenging conditions. The CAD2DMD-SET tool is expected to be released as open-source once the final version of this manuscript is prepared, allowing the community to add different measurement devices and generate their own datasets. 

**Abstract (ZH)**: 最近在大型视觉-语言模型（LVLMs）方面取得的进展展示了其在多种多模态任务中的出色能力。然而，这些模型在处理数字测量设备（DMDs）读取数值等简单场景时仍存在挑战，特别是在包含杂乱环境、遮挡、极端视角和运动模糊等现实条件下，这些情况在头戴式相机和增强现实（AR）应用中尤为常见。受这些限制的启发，本文提出了CAD2DMD-SET，一种合成数据生成工具，旨在支持涉及DMDs的视觉问答（VQA）任务。通过利用3D CAD模型、高级渲染和高保真图像合成，我们的工具生成了多样化的、带有VQA标签的合成DMD数据集，适用于LVLMs的微调。此外，我们还介绍了DMDBench，一个包含1,000张带有标注的现实世界图像的精炼验证集，用于在实际约束条件下评估模型性能。使用平均归一化莱文相似度（ANLS）对三种最先进的LVLMs进行基准测试，并进一步使用CAD2DMD-SET生成的数据集微调这些模型的LoRA，结果显示显著的性能提升，其中InternVL的得分提高了200%，且未在其他任务上退步。这表明CAD2DMD-SET训练数据集在上述挑战条件下大大提高了LVLMs的鲁棒性和性能。CAD2DMD-SET工具计划在最终论文版本准备好后作为开源发布，允许社区添加不同的测量设备并生成自己的数据集。 

---
# OptMark: Robust Multi-bit Diffusion Watermarking via Inference Time Optimization 

**Title (ZH)**: OptMark：通过推理时间优化的稳健多比特扩散水标记技术 

**Authors**: Jiazheng Xing, Hai Ci, Hongbin Xu, Hangjie Yuan, Yong Liu, Mike Zheng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2508.21727)  

**Abstract**: Watermarking diffusion-generated images is crucial for copyright protection and user tracking. However, current diffusion watermarking methods face significant limitations: zero-bit watermarking systems lack the capacity for large-scale user tracking, while multi-bit methods are highly sensitive to certain image transformations or generative attacks, resulting in a lack of comprehensive robustness. In this paper, we propose OptMark, an optimization-based approach that embeds a robust multi-bit watermark into the intermediate latents of the diffusion denoising process. OptMark strategically inserts a structural watermark early to resist generative attacks and a detail watermark late to withstand image transformations, with tailored regularization terms to preserve image quality and ensure imperceptibility. To address the challenge of memory consumption growing linearly with the number of denoising steps during optimization, OptMark incorporates adjoint gradient methods, reducing memory usage from O(N) to O(1). Experimental results demonstrate that OptMark achieves invisible multi-bit watermarking while ensuring robust resilience against valuemetric transformations, geometric transformations, editing, and regeneration attacks. 

**Abstract (ZH)**: 基于优化的Diffusion生成图像水印方法OptMark对于版权保护和用户追踪至关重要。然而，当前的Diffusion水印方法面临显著限制：零比特水印系统缺乏大规模用户追踪能力，而多比特方法对某些图像变换或生成攻击极为敏感，导致缺乏全面的鲁棒性。在本文中，我们提出OptMark，一种基于优化的方法，将鲁棒的多比特水印嵌入Diffusion去噪过程的中间潜在特征中。OptMark在早期战略性地插入结构水印以抵抗生成攻击，在后期插入细节水印以抵御图像变换，通过定制化的正则化项来保留图像质量并确保不可感知性。为了解决优化过程中内存消耗线性增长的挑战，OptMark引入了伴随梯度方法，将内存使用从O(N)降低到O(1)。实验结果表明，OptMark实现了不可感知的多比特水印，同时确保了对估值变换、几何变换、编辑和再生攻击的鲁棒性。 

---
# Entropy-Based Non-Invasive Reliability Monitoring of Convolutional Neural Networks 

**Title (ZH)**: 基于熵的非侵入式卷积神经网络可靠性监控 

**Authors**: Amirhossein Nazeri, Wael Hafez  

**Link**: [PDF](https://arxiv.org/pdf/2508.21715)  

**Abstract**: Convolutional Neural Networks (CNNs) have become the foundation of modern computer vision, achieving unprecedented accuracy across diverse image recognition tasks. While these networks excel on in-distribution data, they remain vulnerable to adversarial perturbations imperceptible input modifications that cause misclassification with high confidence. However, existing detection methods either require expensive retraining, modify network architecture, or degrade performance on clean inputs. Here we show that adversarial perturbations create immediate, detectable entropy signatures in CNN activations that can be monitored without any model modification. Using parallel entropy monitoring on VGG-16, we demonstrate that adversarial inputs consistently shift activation entropy by 7% in early convolutional layers, enabling 90% detection accuracy with false positives and false negative rates below 20%. The complete separation between clean and adversarial entropy distributions reveals that CNNs inherently encode distribution shifts in their activation patterns. This work establishes that CNN reliability can be assessed through activation entropy alone, enabling practical deployment of self-diagnostic vision systems that detect adversarial inputs in real-time without compromising original model performance. 

**Abstract (ZH)**: 卷积神经网络（CNNs）已成为现代计算机视觉的基石，实现了各种图像识别任务前所未有的准确性。虽然这些网络在内部数据分布上表现出色，但它们仍然容易受到不可感知的输入修改（即对抗性扰动）的影响，这些扰动会导致模型以高置信度错误分类。然而，现有的检测方法要么需要昂贵的重新训练，要么修改网络架构，要么在干净输入上降低性能。在这里，我们展示了对抗性扰动会在CNN激活中立即产生可检测的熵特征签名，这些特征可以通过不修改任何模型的情况下进行监控。通过对VGG-16进行并行熵监控，我们证明了对抗性输入始终在早期卷积层中将激活熵偏移7%，从而实现90%的检测准确率，并保持假正例和假负例率低于20%。干净和对抗性熵分布的完全分离揭示了CNNs在其激活模式中固有地编码分布偏移。这项工作表明，仅通过激活熵就可以评估CNN的可靠性，从而实现实时检测对抗性输入的自诊断视觉系统的实际部署，而不影响原始模型的性能。 

---
# Why Stop at Words? Unveiling the Bigger Picture through Line-Level OCR 

**Title (ZH)**: 为什么止于词组？通过行级OCR揭示更大的图景 

**Authors**: Shashank Vempati, Nishit Anand, Gaurav Talebailkar, Arpan Garai, Chetan Arora  

**Link**: [PDF](https://arxiv.org/pdf/2508.21693)  

**Abstract**: Conventional optical character recognition (OCR) techniques segmented each character and then recognized. This made them prone to error in character segmentation, and devoid of context to exploit language models. Advances in sequence to sequence translation in last decade led to modern techniques first detecting words and then inputting one word at a time to a model to directly output full words as sequence of characters. This allowed better utilization of language models and bypass error-prone character segmentation step. We observe that the above transition in style has moved the bottleneck in accuracy to word segmentation. Hence, in this paper, we propose a natural and logical progression from word level OCR to line-level OCR. The proposal allows to bypass errors in word detection, and provides larger sentence context for better utilization of language models. We show that the proposed technique not only improves the accuracy but also efficiency of OCR. Despite our thorough literature survey, we did not find any public dataset to train and benchmark such shift from word to line-level OCR. Hence, we also contribute a meticulously curated dataset of 251 English page images with line-level annotations. Our experimentation revealed a notable end-to-end accuracy improvement of 5.4%, underscoring the potential benefits of transitioning towards line-level OCR, especially for document images. We also report a 4 times improvement in efficiency compared to word-based pipelines. With continuous improvements in large language models, our methodology also holds potential to exploit such advances. Project Website: this https URL 

**Abstract (ZH)**: 传统的光学字符识别（OCR）技术将每个字符进行分割然后识别，这使得它们容易在字符分割时出错，并且无法利用语言模型的上下文信息。近年来序列到序列翻译的进步导致现代技术首先检测单词，然后每次向模型输入一个单词，直接输出完整的单词序列，从而更好地利用语言模型并跳过易出错的字符分割步骤。我们观察到，这种风格的变化将准确率瓶颈移至了单词分割。因此，在本文中，我们提出了一种从单词级OCR自然过渡到行级OCR的方案。该方案能够绕过单词检测错误，并提供较大的句子上下文，从而更好地利用语言模型。我们证明，所提出的技术不仅提高了OCR的准确率，还提高了其效率。尽管我们进行了广泛的文献调研，但仍未发现公开的用于训练和基准测试此类从单词级到行级OCR转变的数据集。因此，我们还贡献了一个精心编纂的包含251张英文页面图像和行级注释的数据集。实验结果显示，该技术的端到端准确率提高了5.4%，突显了向行级OCR转变的潜在益处，尤其是在文档图像中。此外，我们还报告了与基于单词的管道相比，效率提高了4倍。随着大规模语言模型的持续改进，我们的方法也有可能利用这些进步。项目网址: this https URL 

---
# Harnessing IoT and Generative AI for Weather-Adaptive Learning in Climate Resilience Education 

**Title (ZH)**: 利用物联网和生成式人工智能进行气候适应性学习的weather适应性教育 

**Authors**: Imran S. A. Khan, Emmanuel G. Blanchard, Sébastien George  

**Link**: [PDF](https://arxiv.org/pdf/2508.21666)  

**Abstract**: This paper introduces the Future Atmospheric Conditions Training System (FACTS), a novel platform that advances climate resilience education through place-based, adaptive learning experiences. FACTS combines real-time atmospheric data collected by IoT sensors with curated resources from a Knowledge Base to dynamically generate localized learning challenges. Learner responses are analyzed by a Generative AI powered server, which delivers personalized feedback and adaptive support. Results from a user evaluation indicate that participants found the system both easy to use and effective for building knowledge related to climate resilience. These findings suggest that integrating IoT and Generative AI into atmospherically adaptive learning technologies holds significant promise for enhancing educational engagement and fostering climate awareness. 

**Abstract (ZH)**: 面向未来大气条件的培训系统（FACTS）：一种基于-place-的自适应学习平台，促进气候韧性教育 

---
# QZhou-Embedding Technical Report 

**Title (ZH)**: 周求真-嵌入技术报告 

**Authors**: Peng Yu, En Xu, Bin Chen, Haibiao Chen, Yinfei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21632)  

**Abstract**: We present QZhou-Embedding, a general-purpose contextual text embedding model with exceptional text representation capabilities. Built upon the Qwen2.5-7B-Instruct foundation model, we designed a unified multi-task framework comprising specialized data transformation and training strategies. The data transformation scheme enables the incorporation of more diverse textual training datasets, while the task-specific training strategies enhance model learning efficiency. We developed a data synthesis pipeline leveraging LLM API, incorporating techniques such as paraphrasing, augmentation, and hard negative example generation to improve the semantic richness and sample difficulty of the training set. Additionally, we employ a two-stage training strategy, comprising initial retrieval-focused pretraining followed by full-task fine-tuning, enabling the embedding model to extend its capabilities based on robust retrieval performance. Our model achieves state-of-the-art results on the MTEB and CMTEB benchmarks, ranking first on both leaderboards (August 27 2025), and simultaneously achieves state-of-the-art performance on tasks including reranking, clustering, etc. Our findings demonstrate that higher-quality, more diverse data is crucial for advancing retrieval model performance, and that leveraging LLMs generative capabilities can further optimize data quality for embedding model breakthroughs. Our model weights are released on HuggingFace under Apache 2.0 license. For reproducibility, we provide evaluation code and instructions on GitHub. 

**Abstract (ZH)**: Qwen-Embedding：一种通用上下文文本嵌入模型及其卓越的文本表示能力 

---
# Physics-Informed Spectral Modeling for Hyperspectral Imaging 

**Title (ZH)**: 物理约束光谱建模在超光谱成像中的应用 

**Authors**: Zuzanna Gawrysiak, Krzysztof Krawiec  

**Link**: [PDF](https://arxiv.org/pdf/2508.21618)  

**Abstract**: We present PhISM, a physics-informed deep learning architecture that learns without supervision to explicitly disentangle hyperspectral observations and model them with continuous basis functions. \mname outperforms prior methods on several classification and regression benchmarks, requires limited labeled data, and provides additional insights thanks to interpretable latent representation. 

**Abstract (ZH)**: PhISM：一种物理知情的深度学习架构，无需监督地显式地分离超光谱观测并用连续基函数建模，且在多个分类和回归基准测试中表现更优，需要的标记数据较少，并提供可解释的潜在表示以供进一步洞察。 

---
# Middo: Model-Informed Dynamic Data Optimization for Enhanced LLM Fine-Tuning via Closed-Loop Learning 

**Title (ZH)**: Middo: 基于模型的动态数据优化以通过闭环学习增强LLM微调 

**Authors**: Zinan Tang, Xin Gao, Qizhi Pei, Zhuoshi Pan, Mengzhang Cai, Jiang Wu, Conghui He, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21589)  

**Abstract**: Supervised Fine-Tuning (SFT) Large Language Models (LLM) fundamentally rely on high-quality training data. While data selection and data synthesis are two common strategies to improve data quality, existing approaches often face limitations in static dataset curation that fail to adapt to evolving model capabilities. In this paper, we introduce Middo, a self-evolving Model-informed dynamic data optimization framework that uses model-aware data selection and context-preserving data refinement. Unlike conventional one-off filtering/synthesis methods, our framework establishes a closed-loop optimization system: (1) A self-referential diagnostic module proactively identifies suboptimal samples through tri-axial model signals - loss patterns (complexity), embedding cluster dynamics (diversity), and self-alignment scores (quality); (2) An adaptive optimization engine then transforms suboptimal samples into pedagogically valuable training points while preserving semantic integrity; (3) This optimization process continuously evolves with model capability through dynamic learning principles. Experiments on multiple benchmarks demonstrate that our \method consistently enhances the quality of seed data and boosts LLM's performance with improving accuracy by 7.15% on average while maintaining the original dataset scale. This work establishes a new paradigm for sustainable LLM training through dynamic human-AI co-evolution of data and models. Our datasets, models, and code are coming soon. 

**Abstract (ZH)**: 监督微调（SFT）大型语言模型（LLM）从根本上依赖高质量的训练数据。Middo：一种基于模型的自适应动态数据优化框架，通过模型意识的数据选择和语境保留的数据精炼实现自我演进。 

---
# A Survey on Current Trends and Recent Advances in Text Anonymization 

**Title (ZH)**: 当前文本匿名化趋势与recent advances综述 

**Authors**: Tobias Deußer, Lorenz Sparrenberg, Armin Berger, Max Hahnbück, Christian Bauckhage, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2508.21587)  

**Abstract**: The proliferation of textual data containing sensitive personal information across various domains requires robust anonymization techniques to protect privacy and comply with regulations, while preserving data usability for diverse and crucial downstream tasks. This survey provides a comprehensive overview of current trends and recent advances in text anonymization techniques. We begin by discussing foundational approaches, primarily centered on Named Entity Recognition, before examining the transformative impact of Large Language Models, detailing their dual role as sophisticated anonymizers and potent de-anonymization threats. The survey further explores domain-specific challenges and tailored solutions in critical sectors such as healthcare, law, finance, and education. We investigate advanced methodologies incorporating formal privacy models and risk-aware frameworks, and address the specialized subfield of authorship anonymization. Additionally, we review evaluation frameworks, comprehensive metrics, benchmarks, and practical toolkits for real-world deployment of anonymization solutions. This review consolidates current knowledge, identifies emerging trends and persistent challenges, including the evolving privacy-utility trade-off, the need to address quasi-identifiers, and the implications of LLM capabilities, and aims to guide future research directions for both academics and practitioners in this field. 

**Abstract (ZH)**: 文本数据中包含敏感个人信息的扩散要求采用 robust 匿名化技术以保护隐私和遵守法规，同时保留数据对多样化和关键下游任务的可用性。本文综述了当前文本匿名化技术的趋势和 recent 进展，讨论了基础方法，主要集中在命名实体识别，随后探讨了大型语言模型的变革性影响及其作为高级匿名化工具和强大去匿名化威胁的双重角色。此外，本文还探索了医疗、法律、金融和教育等关键领域特有的挑战和定制化解决方案。综述中探讨了结合形式隐私模型和风险感知框架的高级方法，以及作者匿名化这一专门子领域的进展。同时，本文还回顾了评估框架、全面指标、基准和实际部署匿名化解决方案的工具包。本文总结了当前的知识，指出了新兴趋势和持续挑战，包括隐私-效用权衡的演变、必须应对的准标识符问题，以及大型语言模型能力的影响，并旨在为学术界和该领域的实践者指引未来的研究方向。 

---
# NSPDI-SNN: An efficient lightweight SNN based on nonlinear synaptic pruning and dendritic integration 

**Title (ZH)**: NSPDI-SNN: 一种基于非线性突触修剪和树突整合的高效轻量级SNN 

**Authors**: Wuque Cai, Hongze Sun, Jiayi He, Qianqian Liao, Yunliang Zang, Duo Chen, Dezhong Yao, Daqing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2508.21566)  

**Abstract**: Spiking neural networks (SNNs) are artificial neural networks based on simulated biological neurons and have attracted much attention in recent artificial intelligence technology studies. The dendrites in biological neurons have efficient information processing ability and computational power; however, the neurons of SNNs rarely match the complex structure of the dendrites. Inspired by the nonlinear structure and highly sparse properties of neuronal dendrites, in this study, we propose an efficient, lightweight SNN method with nonlinear pruning and dendritic integration (NSPDI-SNN). In this method, we introduce nonlinear dendritic integration (NDI) to improve the representation of the spatiotemporal information of neurons. We implement heterogeneous state transition ratios of dendritic spines and construct a new and flexible nonlinear synaptic pruning (NSP) method to achieve the high sparsity of SNN. We conducted systematic experiments on three benchmark datasets (DVS128 Gesture, CIFAR10-DVS, and CIFAR10) and extended the evaluation to two complex tasks (speech recognition and reinforcement learning-based maze navigation task). Across all tasks, NSPDI-SNN consistently achieved high sparsity with minimal performance degradation. In particular, our method achieved the best experimental results on all three event stream datasets. Further analysis showed that NSPDI significantly improved the efficiency of synaptic information transfer as sparsity increased. In conclusion, our results indicate that the complex structure and nonlinear computation of neuronal dendrites provide a promising approach for developing efficient SNN methods. 

**Abstract (ZH)**: 基于非线性修剪和树突整合的高效轻量级脉冲神经网络（NSPDI-SNN） 

---
# Limitations of Physics-Informed Neural Networks: a Study on Smart Grid Surrogation 

**Title (ZH)**: 物理学知情神经网络的局限性：智能电网代理研究 

**Authors**: Julen Cestero, Carmine Delle Femine, Kenji S. Muro, Marco Quartulli, Marcello Restelli  

**Link**: [PDF](https://arxiv.org/pdf/2508.21559)  

**Abstract**: Physics-Informed Neural Networks (PINNs) present a transformative approach for smart grid modeling by integrating physical laws directly into learning frameworks, addressing critical challenges of data scarcity and physical consistency in conventional data-driven methods. This paper evaluates PINNs' capabilities as surrogate models for smart grid dynamics, comparing their performance against XGBoost, Random Forest, and Linear Regression across three key experiments: interpolation, cross-validation, and episodic trajectory prediction. By training PINNs exclusively through physics-based loss functions (enforcing power balance, operational constraints, and grid stability) we demonstrate their superior generalization, outperforming data-driven models in error reduction. Notably, PINNs maintain comparatively lower MAE in dynamic grid operations, reliably capturing state transitions in both random and expert-driven control scenarios, while traditional models exhibit erratic performance. Despite slight degradation in extreme operational regimes, PINNs consistently enforce physical feasibility, proving vital for safety-critical applications. Our results contribute to establishing PINNs as a paradigm-shifting tool for smart grid surrogation, bridging data-driven flexibility with first-principles rigor. This work advances real-time grid control and scalable digital twins, emphasizing the necessity of physics-aware architectures in mission-critical energy systems. 

**Abstract (ZH)**: 物理学知情神经网络（PINNs）通过将物理定律直接集成到学习框架中，为智能电网建模提供了一种变革性的方法，解决了常规数据驱动方法中数据稀缺性和物理一致性的关键挑战。本文评估了PINNs作为智能电网动力学代理模型的能力，比较了其在三次关键实验（插值、交叉验证和事件轨迹预测）中与XGBoost、随机森林和线性回归模型的性能差异。通过仅使用基于物理的损失函数（强制功率平衡、运行约束和电网稳定性）来训练PINNs，我们展示了其卓越的泛化能力，优于数据驱动模型的误差减少。值得注意的是，PINNs在动态电网操作中保持相对较低的MAE，可靠地捕捉随机和专家驱动控制场景下的状态转换，而传统模型则表现出不稳定的性能。尽管在极端运行条件下略有下降，但PINNs始终确保物理可行性，证明其在关键安全应用中的重要性。我们的研究成果表明，PINNs 是一种变革性的工具，可以实现智能电网的代理建模，将数据驱动的灵活性与基于原理的严谨性相结合，进一步推进了实时电网控制和可扩展的数字孪生技术，并强调了在关键能源系统中使用物理感知架构的必要性。 

---
# EZ-Sort: Efficient Pairwise Comparison via Zero-Shot CLIP-Based Pre-Ordering and Human-in-the-Loop Sorting 

**Title (ZH)**: EZ-Sort: 高效的零样本CLIP指导的先序对比与人力参与排序 

**Authors**: Yujin Park, Haejun Chung, Ikbeom Jang  

**Link**: [PDF](https://arxiv.org/pdf/2508.21550)  

**Abstract**: Pairwise comparison is often favored over absolute rating or ordinal classification in subjective or difficult annotation tasks due to its improved reliability. However, exhaustive comparisons require a massive number of annotations (O(n^2)). Recent work has greatly reduced the annotation burden (O(n log n)) by actively sampling pairwise comparisons using a sorting algorithm. We further improve annotation efficiency by (1) roughly pre-ordering items using the Contrastive Language-Image Pre-training (CLIP) model hierarchically without training, and (2) replacing easy, obvious human comparisons with automated comparisons. The proposed EZ-Sort first produces a CLIP-based zero-shot pre-ordering, then initializes bucket-aware Elo scores, and finally runs an uncertainty-guided human-in-the-loop MergeSort. Validation was conducted using various datasets: face-age estimation (FGNET), historical image chronology (DHCI), and retinal image quality assessment (EyePACS). It showed that EZ-Sort reduced human annotation cost by 90.5% compared to exhaustive pairwise comparisons and by 19.8% compared to prior work (when n = 100), while improving or maintaining inter-rater reliability. These results demonstrate that combining CLIP-based priors with uncertainty-aware sampling yields an efficient and scalable solution for pairwise ranking. 

**Abstract (ZH)**: 基于CLIP的对比排序：一种高效可扩展的成对排序方法 

---
# What Data is Really Necessary? A Feasibility Study of Inference Data Minimization for Recommender Systems 

**Title (ZH)**: 真正需要哪些数据？推荐系统中推理数据最小化可行性的研究 

**Authors**: Jens Leysen, Marco Favier, Bart Goethals  

**Link**: [PDF](https://arxiv.org/pdf/2508.21547)  

**Abstract**: Data minimization is a legal principle requiring personal data processing to be limited to what is necessary for a specified purpose. Operationalizing this principle for recommender systems, which rely on extensive personal data, remains a significant challenge. This paper conducts a feasibility study on minimizing implicit feedback inference data for such systems. We propose a novel problem formulation, analyze various minimization techniques, and investigate key factors influencing their effectiveness. We demonstrate that substantial inference data reduction is technically feasible without significant performance loss. However, its practicality is critically determined by two factors: the technical setting (e.g., performance targets, choice of model) and user characteristics (e.g., history size, preference complexity). Thus, while we establish its technical feasibility, we conclude that data minimization remains practically challenging and its dependence on the technical and user context makes a universal standard for data `necessity' difficult to implement. 

**Abstract (ZH)**: 数据最小化是法律原则，要求个人信息处理仅限于为特定目的所必需的数据。将这一原则应用于依赖大量个人信息的推荐系统中仍然是一个重大挑战。本文开展了减少隐式反馈推理数据可行性的研究。我们提出了一个新的问题表述，分析了各种最小化技术，并调查了影响其有效性的关键因素。我们证明，在不显著牺牲性能的情况下，大幅度减少推理数据是技术上可行的。然而，其实用性强烈取决于两个因素：技术设置（例如，性能目标、所选模型）和用户特征（例如，历史大小、偏好复杂性）。因此，尽管我们证明了其技术可行性，但得出结论认为，数据最小化在实际应用中仍具有挑战性，并且其依赖于技术和用户上下文使得制定数据“必要性”的通用标准难以实施。 

---
# Complete Gaussian Splats from a Single Image with Denoising Diffusion Models 

**Title (ZH)**: 使用去噪扩散模型从单张图像生成完整高斯点云 

**Authors**: Ziwei Liao, Mohamed Sayed, Steven L. Waslander, Sara Vicente, Daniyar Turmukhambetov, Michael Firman  

**Link**: [PDF](https://arxiv.org/pdf/2508.21542)  

**Abstract**: Gaussian splatting typically requires dense observations of the scene and can fail to reconstruct occluded and unobserved areas. We propose a latent diffusion model to reconstruct a complete 3D scene with Gaussian splats, including the occluded parts, from only a single image during inference. Completing the unobserved surfaces of a scene is challenging due to the ambiguity of the plausible surfaces. Conventional methods use a regression-based formulation to predict a single "mode" for occluded and out-of-frustum surfaces, leading to blurriness, implausibility, and failure to capture multiple possible explanations. Thus, they often address this problem partially, focusing either on objects isolated from the background, reconstructing only visible surfaces, or failing to extrapolate far from the input views. In contrast, we propose a generative formulation to learn a distribution of 3D representations of Gaussian splats conditioned on a single input image. To address the lack of ground-truth training data, we propose a Variational AutoReconstructor to learn a latent space only from 2D images in a self-supervised manner, over which a diffusion model is trained. Our method generates faithful reconstructions and diverse samples with the ability to complete the occluded surfaces for high-quality 360-degree renderings. 

**Abstract (ZH)**: 基于拉普拉斯扩散模型的单图全场景高保真重建 

---
# On the Hardness of Learning GNN-based SAT Solvers: The Role of Graph Ricci Curvature 

**Title (ZH)**: 基于图神经网络的SAT求解器学习的难题：图 Ricci 曲率的作用 

**Authors**: Geri Skenderi  

**Link**: [PDF](https://arxiv.org/pdf/2508.21513)  

**Abstract**: Graph Neural Networks (GNNs) have recently shown promise as solvers for Boolean Satisfiability Problems (SATs) by operating on graph representations of logical formulas. However, their performance degrades sharply on harder instances, raising the question of whether this reflects fundamental architectural limitations. In this work, we provide a geometric explanation through the lens of graph Ricci Curvature (RC), which quantifies local connectivity bottlenecks. We prove that bipartite graphs derived from random k-SAT formulas are inherently negatively curved, and that this curvature decreases with instance difficulty. Building on this, we show that GNN-based SAT solvers are affected by oversquashing, a phenomenon where long-range dependencies become impossible to compress into fixed-length representations. We validate our claims empirically across different SAT benchmarks and confirm that curvature is both a strong indicator of problem complexity and can be used to predict performance. Finally, we connect our findings to design principles of existing solvers and outline promising directions for future work. 

**Abstract (ZH)**: 图神经网络（GNNs）通过操作逻辑公式的图表示，在解决布尔可满足性问题（SATs）方面展现出了潜力。然而，它们在较难实例上的性能急剧下降，引发了是否反映了基本架构限制的疑问。在这项工作中，我们通过图里奇曲率（RC）的几何视角提供了解释，RC量化了局部连接瓶颈。我们证明来自随机k-SAT公式的双部图本质上是负曲率的，并且这种曲率随着实例难度的增加而降低。在此基础上，我们展示了基于GNN的SAT求解器受到过度挤压现象的影响，该现象使得长距离依赖关系无法压缩到固定长度表示中。我们通过不同SAT基准的实证研究验证了这些观点，并确认曲率既是问题复杂性的强指标，也可以用来预测性能。最后，我们将我们的发现与现有求解器的设计原则联系起来，并提出了未来工作的有希望的方向。 

---
# ELV-Halluc: Benchmarking Semantic Aggregation Hallucinations in Long Video Understanding 

**Title (ZH)**: ELV-Halluc: 长视频理解中语义聚合幻象的基准测试 

**Authors**: Hao Lu, Jiahao Wang, Yaolun Zhang, Ruohui Wang, Xuanyu Zheng, Yepeng Tang, Dahua Lin, Lewei Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21496)  

**Abstract**: Video multimodal large language models (Video-MLLMs) have achieved remarkable progress in video understanding. However, they remain vulnerable to hallucination-producing content inconsistent with or unrelated to video inputs. Previous video hallucination benchmarks primarily focus on short-videos. They attribute hallucinations to factors such as strong language priors, missing frames, or vision-language biases introduced by the visual encoder. While these causes indeed account for most hallucinations in short videos, they still oversimplify the cause of hallucinations. Sometimes, models generate incorrect outputs but with correct frame-level semantics. We refer to this type of hallucination as Semantic Aggregation Hallucination (SAH), which arises during the process of aggregating frame-level semantics into event-level semantic groups. Given that SAH becomes particularly critical in long videos due to increased semantic complexity across multiple events, it is essential to separate and thoroughly investigate the causes of this type of hallucination. To address the above issues, we introduce ELV-Halluc, the first benchmark dedicated to long-video hallucination, enabling a systematic investigation of SAH. Our experiments confirm the existence of SAH and show that it increases with semantic complexity. Additionally, we find that models are more prone to SAH on rapidly changing semantics. Moreover, we discuss potential approaches to mitigate SAH. We demonstrate that positional encoding strategy contributes to alleviating SAH, and further adopt DPO strategy to enhance the model's ability to distinguish semantics within and across events. To support this, we curate a dataset of 8K adversarial data pairs and achieve improvements on both ELV-Halluc and Video-MME, including a substantial 27.7% reduction in SAH ratio. 

**Abstract (ZH)**: Video多模态大型语言模型（Video-MLLMs）在视频理解领域取得了显著进展，然而，它们仍然容易产生与视频输入内容不一致或无关的幻觉。现有的视频幻觉基准主要关注短视频，将幻觉归因于强大的语言先验、缺失的帧或由视觉编码器引入的视觉-语言偏见。虽然这些因素确实解释了短视频中大部分的幻觉，但它们仍然过于简化了幻觉的原因。有时，模型生成错误的输出，但帧级语义是正确的。我们将这种类型的幻觉称为语义聚合幻觉（SAH），它发生在将帧级语义聚合到事件级语义组的过程中。由于长视频中多事件间的语义复杂性增加，SAH变得尤为重要。因此，有必要分离和深入研究这种幻觉的原因。为了解决上述问题，我们引入了ELV-Halluc，这是第一个专门关注长视频幻觉的基准，使得系统地研究SAH成为可能。我们的实验验证了SAH的存在，并表明其随语义复杂性的增加而增加。此外，我们发现模型在快速变化的语义上更容易产生SAH。我们还讨论了减轻SAH的潜在方法。我们证明了位置编码策略有助于缓解SAH，并进一步采用了DPO策略以增强模型在事件内外区分语义的能力。为此，我们构建了一个包含8K对抗数据对的数据集，并在ELV-Halluc和Video-MME上取得了改进，包括SAH比显著减少了27.7%。 

---
# Priors Matter: Addressing Misspecification in Bayesian Deep Q-Learning 

**Title (ZH)**: 先验信息很重要：解决贝叶斯深度Q学习中的模型设定错误问题 

**Authors**: Pascal R. van der Vaart, Neil Yorke-Smith, Matthijs T.J. Spaan  

**Link**: [PDF](https://arxiv.org/pdf/2508.21488)  

**Abstract**: Uncertainty quantification in reinforcement learning can greatly improve exploration and robustness. Approximate Bayesian approaches have recently been popularized to quantify uncertainty in model-free algorithms. However, so far the focus has been on improving the accuracy of the posterior approximation, instead of studying the accuracy of the prior and likelihood assumptions underlying the posterior. In this work, we demonstrate that there is a cold posterior effect in Bayesian deep Q-learning, where contrary to theory, performance increases when reducing the temperature of the posterior. To identify and overcome likely causes, we challenge common assumptions made on the likelihood and priors in Bayesian model-free algorithms. We empirically study prior distributions and show through statistical tests that the common Gaussian likelihood assumption is frequently violated. We argue that developing more suitable likelihoods and priors should be a key focus in future Bayesian reinforcement learning research and we offer simple, implementable solutions for better priors in deep Q-learning that lead to more performant Bayesian algorithms. 

**Abstract (ZH)**: 不确定性量化在强化学习中的应用可以显著提高探索能力和鲁棒性。最近，近似贝叶斯方法被广泛用于量化模型自由算法中的不确定性。然而，至今为止的研究主要集中在提高后验近似的准确性，而不是研究支撑后验的先验和似然假设的准确性。在本工作中，我们展示了在贝叶斯深度Q学习中存在一个后验冷却效应，与理论相反，降低后验温度可以提高性能。为了识别并克服可能的原因，我们挑战了在贝叶斯模型自由算法中常见的似然和先验假设。我们通过实证研究先验分布，并通过统计测试表明常见的高斯似然假设经常被违反。我们认为，在未来的研究中，开发更合适的似然和先验应成为重点，并提出了一些简单可行的解决方案，以改进深度Q学习中的先验，从而提高贝叶斯算法的性能。 

---
# HSFN: Hierarchical Selection for Fake News Detection building Heterogeneous Ensemble 

**Title (ZH)**: HSFN：分层选择的虚假新闻检测构建异构集成 

**Authors**: Sara B. Coutinho, Rafael M.O. Cruz, Francimaria R. S. Nascimento, George D. C. Cavalcanti  

**Link**: [PDF](https://arxiv.org/pdf/2508.21482)  

**Abstract**: Psychological biases, such as confirmation bias, make individuals particularly vulnerable to believing and spreading fake news on social media, leading to significant consequences in domains such as public health and politics. Machine learning-based fact-checking systems have been widely studied to mitigate this problem. Among them, ensemble methods are particularly effective in combining multiple classifiers to improve robustness. However, their performance heavily depends on the diversity of the constituent classifiers-selecting genuinely diverse models remains a key challenge, especially when models tend to learn redundant patterns. In this work, we propose a novel automatic classifier selection approach that prioritizes diversity, also extended by performance. The method first computes pairwise diversity between classifiers and applies hierarchical clustering to organize them into groups at different levels of granularity. A HierarchySelect then explores these hierarchical levels to select one pool of classifiers per level, each representing a distinct intra-pool diversity. The most diverse pool is identified and selected for ensemble construction from these. The selection process incorporates an evaluation metric reflecting each classifiers's performance to ensure the ensemble also generalises well. We conduct experiments with 40 heterogeneous classifiers across six datasets from different application domains and with varying numbers of classes. Our method is compared against the Elbow heuristic and state-of-the-art baselines. Results show that our approach achieves the highest accuracy on two of six datasets. The implementation details are available on the project's repository: this https URL . 

**Abstract (ZH)**: 心理学上的偏见，如确认偏见，使个体特别容易相信和传播社交媒体上的假新闻，对公共健康和政治等领域产生了重大影响。基于机器学习的事实核查系统广泛研究以减轻这一问题。其中，集成方法特别有效，可以通过结合多个分类器来提高鲁棒性。然而，其性能 heavily 取决于构成分类器的多样性——选择真正多样化的模型仍然是一个关键挑战，尤其是在模型倾向于学习冗余模式时。在本工作中，我们提出了一种新的自动分类器选择方法，该方法优先考虑多样性和性能。该方法首先计算分类器之间的成对多样性，并应用层次聚类来按不同程度的粒度组织它们。然后，HierarchySelect 探索这些层次结构，在每一级选择一个池的分类器，每个池代表一种独特的 intra-pool 多样性。最多样化的池被识别并选择用于集成构建。选择过程结合了个别分类器性能的评估指标，以确保集成也能很好地泛化。我们在六个不同应用领域的四个零分类器上进行了实验，数据集具有不同的类数。我们的方法与 Elbow 偏差和最先进的基准进行了比较。结果显示，我们的方法在六个数据集中有两个实现了最高的准确性。项目实现细节可在项目仓库中找到：this https URL。 

---
# Igniting Creative Writing in Small Language Models: LLM-as-a-Judge versus Multi-Agent Refined Rewards 

**Title (ZH)**: 在小型语言模型中激发创造性写作：LLM作为裁判与多智能体精炼奖励的对比 

**Authors**: Xiaolong Wei, Bo Lu, Xingyu Zhang, Zhejun Zhao, Dongdong Shen, Long Xia, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.21476)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable creative writing capabilities, yet their substantial computational demands hinder widespread use. Enhancing Small Language Models (SLMs) offers a promising alternative, but current methods like Supervised Fine-Tuning (SFT) struggle with novelty, and Reinforcement Learning from Human Feedback (RLHF) is costly. This paper explores two distinct AI-driven reward strategies within a Reinforcement Learning from AI Feedback (RLAIF) framework to ignite the creative writing of a 7B-parameter SLM, specifically for generating Chinese greetings. The first strategy employs a RM trained on high-quality preference data curated by a novel multi-agent rejection sampling framework designed for creative tasks. The second, more novel strategy utilizes a principle-guided LLM-as-a-Judge, whose reward function is optimized via an adversarial training scheme with a reflection mechanism, to directly provide reward signals. Comprehensive experiments reveal that while both approaches significantly enhance creative output over baselines, the principle-guided LLM-as-a-Judge demonstrably yields superior generation quality. Furthermore, it offers notable advantages in training efficiency and reduced dependency on human-annotated data, presenting a more scalable and effective path towards creative SLMs. Our automated evaluation methods also exhibit strong alignment with human judgments. Our code and data are publicly available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）展示了卓越的创造性写作能力，但其巨大的计算需求阻碍了其广泛应用。增强小型语言模型（SLMs）提供了一种有希望的替代方案，但当前的方法如监督微调（SFT）在新颖性方面存在不足，强化学习从人类反馈（RLHF）则成本较高。本文在AI反馈强化学习（RLAIF）框架内探索了两种独特的AI驱动的奖励策略，以激发一个包含70亿参数的小型语言模型的创造性写作能力，特别用于生成中文问候语。第一种策略采用了一种基于新设计的多智能体拒绝采样框架训练的偏好模型，该框架旨在针对创造任务。第二种更为新颖的策略则利用了一个原理引导下的LLM作为裁判，其奖励函数通过与反向训练方案结合的反思机制进行优化，直接提供奖励信号。全面的实验表明，虽然两种方法在基线之上显著提升了创造性输出，但原理引导下的LLM作为裁判明显产生了更高的生成质量。此外，其在训练效率和减少对人工标注数据的依赖方面具有显著优势，为创造性SLM的发展提供了更加可扩展和有效的方法。我们的自动评估方法也与人类判断表现出较强的一致性。我们的代码和数据已在此网页公开。 

---
# Controllable 3D Molecular Generation for Structure-Based Drug Design Through Bayesian Flow Networks and Gradient Integration 

**Title (ZH)**: 基于结构的药物设计中通过贝叶斯流网络和梯度集成的可控3D分子生成 

**Authors**: Seungyeon Choi, Hwanhee Kim, Chihyun Park, Dahyeon Lee, Seungyong Lee, Yoonju Kim, Hyoungjoon Park, Sein Kwon, Youngwan Jo, Sanghyun Park  

**Link**: [PDF](https://arxiv.org/pdf/2508.21468)  

**Abstract**: Recent advances in Structure-based Drug Design (SBDD) have leveraged generative models for 3D molecular generation, predominantly evaluating model performance by binding affinity to target proteins. However, practical drug discovery necessitates high binding affinity along with synthetic feasibility and selectivity, critical properties that were largely neglected in previous evaluations. To address this gap, we identify fundamental limitations of conventional diffusion-based generative models in effectively guiding molecule generation toward these diverse pharmacological properties. We propose CByG, a novel framework extending Bayesian Flow Network into a gradient-based conditional generative model that robustly integrates property-specific guidance. Additionally, we introduce a comprehensive evaluation scheme incorporating practical benchmarks for binding affinity, synthetic feasibility, and selectivity, overcoming the limitations of conventional evaluation methods. Extensive experiments demonstrate that our proposed CByG framework significantly outperforms baseline models across multiple essential evaluation criteria, highlighting its effectiveness and practicality for real-world drug discovery applications. 

**Abstract (ZH)**: 基于结构的药物设计（SBDD）的 Recent 进展通过生成模型进行三维分子生成，并主要通过靶蛋白的结合亲和力评价模型性能。然而，实际的药物发现不仅需要高结合亲和力，还需要合成可行性及选择性等关键属性，而这些属性在之前的研究中被很大程度上忽视。为解决这一问题，我们指出了常规基于扩散的生成模型在有效引导分子生成以获得这些多样的药理学属性方面的根本局限。我们提出了一种新颖的 CByG 框架，将贝叶斯流网络扩展为基于梯度的条件生成模型，以稳健地整合属性特定的引导。此外，我们引入了一套综合的评估方案，结合实际基准评估结合亲和力、合成可行性和选择性，克服了传统评估方法的局限性。广泛的实验表明，我们提出的 CByG 框架在多个关键评估标准上显著优于基线模型，突显了其在真实世界药物发现应用中的有效性和实用性。 

---
# Diffusion-based Multi-modal Synergy Interest Network for Click-through Rate Prediction 

**Title (ZH)**: 基于扩散的多模态协同兴趣网络点击率预测 

**Authors**: Xiaoxi Cui, Weihai Lu, Yu Tong, Yiheng Li, Zhejun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.21460)  

**Abstract**: In click-through rate prediction, click-through rate prediction is used to model users' interests. However, most of the existing CTR prediction methods are mainly based on the ID modality. As a result, they are unable to comprehensively model users' multi-modal preferences. Therefore, it is necessary to introduce multi-modal CTR prediction. Although it seems appealing to directly apply the existing multi-modal fusion methods to click-through rate prediction models, these methods (1) fail to effectively disentangle commonalities and specificities across different modalities; (2) fail to consider the synergistic effects between modalities and model the complex interactions between modalities.
To address the above issues, this paper proposes the Diffusion-based Multi-modal Synergy Interest Network (Diff-MSIN) framework for click-through prediction. This framework introduces three innovative modules: the Multi-modal Feature Enhancement (MFE) Module Synergistic Relationship Capture (SRC) Module, and the Feature Dynamic Adaptive Fusion (FDAF) Module. The MFE Module and SRC Module extract synergistic, common, and special information among different modalities. They effectively enhances the representation of the modalities, improving the overall quality of the fusion. To encourage distinctiveness among different features, we design a Knowledge Decoupling method. Additionally, the FDAF Module focuses on capturing user preferences and reducing fusion noise. To validate the effectiveness of the Diff-MSIN framework, we conducted extensive experiments using the Rec-Tmall and three Amazon datasets. The results demonstrate that our approach yields a significant improvement of at least 1.67% compared to the baseline, highlighting its potential for enhancing multi-modal recommendation systems. Our code is available at the following link: this https URL. 

**Abstract (ZH)**: 基于扩散过程的多模态协同兴趣网络在点击率预测中的应用 

---
# MedShift: Implicit Conditional Transport for X-Ray Domain Adaptation 

**Title (ZH)**: MedShift: 隐式条件传输在X射线领域适应中应用 

**Authors**: Francisco Caetano, Christiaan Viviers, Peter H.H. de With, Fons van der Sommen  

**Link**: [PDF](https://arxiv.org/pdf/2508.21435)  

**Abstract**: Synthetic medical data offers a scalable solution for training robust models, but significant domain gaps limit its generalizability to real-world clinical settings. This paper addresses the challenge of cross-domain translation between synthetic and real X-ray images of the head, focusing on bridging discrepancies in attenuation behavior, noise characteristics, and soft tissue representation. We propose MedShift, a unified class-conditional generative model based on Flow Matching and Schrodinger Bridges, which enables high-fidelity, unpaired image translation across multiple domains. Unlike prior approaches that require domain-specific training or rely on paired data, MedShift learns a shared domain-agnostic latent space and supports seamless translation between any pair of domains seen during training. We introduce X-DigiSkull, a new dataset comprising aligned synthetic and real skull X-rays under varying radiation doses, to benchmark domain translation models. Experimental results demonstrate that, despite its smaller model size compared to diffusion-based approaches, MedShift offers strong performance and remains flexible at inference time, as it can be tuned to prioritize either perceptual fidelity or structural consistency, making it a scalable and generalizable solution for domain adaptation in medical imaging. The code and dataset are available at this https URL 

**Abstract (ZH)**: 合成医学数据提供了一种可扩展的解决方案，用于训练 robust 模型，但显著的领域差异限制了其在实际临床环境中的泛化能力。本文解决了合成和真实头部 X 光图像之间跨领域转换的挑战，专注于弥合衰减行为、噪声特性和软组织表示的差异。我们提出了基于 Flow Matching 和 Schrodinger Bridges 的统一类条件生成模型 MedShift，该模型可在多个领域之间实现高保真、非配对的图像转换。与需要领域特定训练或依赖配对数据的先前方法不同，MedShift 学习到一个共享的领域无关潜在空间，并支持在训练过程中看到的任意领域对之间的无缝转换。我们提出了一个新的数据集 X-DigiSkull，其中包含在不同辐射剂量下对齐的合成和真实头骨 X 光图像，用于评估领域转换模型。实验结果表明，尽管其模型大小较小，与基于扩散的方法相比，MedShift 具有较强的性能，并且在推理时保持灵活，可以调整以优先考虑感知保真度或结构一致性，从而使它成为医疗成像中领域适应的可扩展和通用解决方案。代码和数据集可以在以下链接获取。 

---
# The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management 

**Title (ZH)**: 简单观察掩蔽与大规模语言模型总结一样高效，用于代理情境管理的复杂性陷阱 

**Authors**: Tobias Lindenbauer, Igor Slinko, Ludwig Felder, Egor Bogomolov, Yaroslav Zharov  

**Link**: [PDF](https://arxiv.org/pdf/2508.21433)  

**Abstract**: Large Language Model (LLM)-based agents solve complex tasks through iterative reasoning, exploration, and tool-use, a process that can result in long, expensive context histories. While state-of-the-art Software Engineering ( SE) agents like OpenHands or Cursor use LLM-based summarization to tackle this issue, it is unclear whether the increased complexity offers tangible performance benefits compared to simply omitting older observations. We present a systematic comparison of these strategies within SWE-agent on SWE-bench Verified across five diverse model configurations. We find that a simple observation-masking strategy halves cost relative to a raw agent while matching, and sometimes slightly exceeding, the solve rate of LLM summarization. For example, with Qwen3-Coder 480B, masking improves solve rate from 53.8% (raw agent) to 54.8%, while remaining competitive with summarization at a lower cost. These results suggest that, at least within SWE-agent on SWE-bench Verified, the most effective and efficient context management can be the simplest. We release code and data for reproducibility 

**Abstract (ZH)**: 基于大型语言模型的代理通过迭代推理、探索和工具使用来解决复杂任务，这一过程可能会产生长且昂贵的历史上下文。虽然像OpenHands或Cursor这样的最新软件工程代理使用基于大型语言模型的总结来应对这一问题，但尚不清楚增加的复杂性是否能提供实际的性能优势，相比直接省略较旧的观察结果。在SWE-agent上，我们在SWE-bench Verified的五个不同模型配置中系统比较了这些策略。我们发现，简单的观察屏蔽策略使成本减半，同时与基于大型语言模型的总结匹配，有时甚至稍微超过总结的解题率。例如，使用Qwen3-Coder 480B时，屏蔽提高了解题率从53.8%（原始代理）到54.8%，同时以较低成本保持与总结的竞争力。这些结果表明，在SWE-agent上，SWE-bench Verified至少在最有效和高效的上下文管理上，最简单的做法可能是最好的。我们公布代码和数据以确保可再现性。 

---
# Med-RewardBench: Benchmarking Reward Models and Judges for Medical Multimodal Large Language Models 

**Title (ZH)**: Med-RewardBench: 评价医学多模态大型语言模型奖励模型和评估者的基准 

**Authors**: Meidan Ding, Jipeng Zhang, Wenxuan Wang, Cheng-Yi Li, Wei-Chieh Fang, Hsin-Yu Wu, Haiqin Zhong, Wenting Chen, Linlin Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.21430)  

**Abstract**: Multimodal large language models (MLLMs) hold significant potential in medical applications, including disease diagnosis and clinical decision-making. However, these tasks require highly accurate, context-sensitive, and professionally aligned responses, making reliable reward models and judges critical. Despite their importance, medical reward models (MRMs) and judges remain underexplored, with no dedicated benchmarks addressing clinical requirements. Existing benchmarks focus on general MLLM capabilities or evaluate models as solvers, neglecting essential evaluation dimensions like diagnostic accuracy and clinical relevance. To address this, we introduce Med-RewardBench, the first benchmark specifically designed to evaluate MRMs and judges in medical scenarios. Med-RewardBench features a multimodal dataset spanning 13 organ systems and 8 clinical departments, with 1,026 expert-annotated cases. A rigorous three-step process ensures high-quality evaluation data across six clinically critical dimensions. We evaluate 32 state-of-the-art MLLMs, including open-source, proprietary, and medical-specific models, revealing substantial challenges in aligning outputs with expert judgment. Additionally, we develop baseline models that demonstrate substantial performance improvements through fine-tuning. 

**Abstract (ZH)**: 多模态大型语言模型在医疗应用中的潜力包括疾病诊断和临床决策，然而这些任务需要高度准确、上下文敏感和专业对齐的响应，因此可靠的奖励模型和评判者至关重要。尽管如此，医疗奖励模型（MRMs）和评判者仍未得到充分探索，缺少专门针对临床需求的基准测试。现有基准测试主要关注通用的MLLM能力或评估模型作为求解器，忽视了诊断准确性等关键评估维度。为解决这一问题，我们介绍了Med-RewardBench，这是首个专门用于评估MRMs和评判者在医疗场景中的基准测试。Med-RewardBench 包含涵盖13个器官系统和8个临床部门的多模态数据集，共有1026个专家标注的案例。严谨的三步过程确保了六个临床关键维度的高质量评估数据。我们评估了32个最先进的MLLMs，包括开源、专有和医疗专用模型，揭示了与专家判断对齐的巨大挑战。此外，我们还开发了基准模型，通过微调实现了显著的性能提升。 

---
# Benchmarking the State of Networks with a Low-Cost Method Based on Reservoir Computing 

**Title (ZH)**: 基于水库计算的低成本方法评估网络状态 

**Authors**: Felix Simon Reimers, Carl-Hendrik Peters, Stefano Nichele  

**Link**: [PDF](https://arxiv.org/pdf/2508.21420)  

**Abstract**: Using data from mobile network utilization in Norway, we showcase the possibility of monitoring the state of communication and mobility networks with a non-invasive, low-cost method. This method transforms the network data into a model within the framework of reservoir computing and then measures the model's performance on proxy tasks. Experimentally, we show how the performance on these proxies relates to the state of the network. A key advantage of this approach is that it uses readily available data sets and leverages the reservoir computing framework for an inexpensive and largely agnostic method. Data from mobile network utilization is available in an anonymous, aggregated form with multiple snapshots per day. This data can be treated like a weighted network. Reservoir computing allows the use of weighted, but untrained networks as a machine learning tool. The network, initialized as a so-called echo state network (ESN), projects incoming signals into a higher dimensional space, on which a single trained layer operates. This consumes less energy than deep neural networks in which every weight of the network is trained. We use neuroscience inspired tasks and trained our ESN model to solve them. We then show how the performance depends on certain network configurations and also how it visibly decreases when perturbing the network. While this work serves as proof of concept, we believe it can be elevated to be used for near-real-time monitoring as well as the identification of possible weak spots of both mobile communication networks as well as transportation networks. 

**Abstract (ZH)**: 利用挪威移动网络利用数据，展示一种无侵入、低成本监控通信和移动网络状态的方法 

---
# DRASP: A Dual-Resolution Attentive Statistics Pooling Framework for Automatic MOS Prediction 

**Title (ZH)**: DRASP：一种双分辨率注意统计池化框架用于自动MOS预测 

**Authors**: Cheng-Yeh Yang, Kuan-Tang Huang, Chien-Chun Wang, Hung-Shin Lee, Hsin-Min Wang, Berlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.21407)  

**Abstract**: A pooling mechanism is essential for mean opinion score (MOS) prediction, facilitating the transformation of variable-length audio features into a concise fixed-size representation that effectively encodes speech quality. Existing pooling methods typically operate at a singular granularity, concentrating either on a comprehensive global perspective or a detailed frame-level analysis, which may overlook complementary perceptual insights. To address this limitation, we introduce the Dual-Resolution Attentive Statistics Pooling (DRASP) framework. DRASP integrates both coarse-grained, global statistical summaries and fine-grained, attentive analyses of perceptually significant segments. This dual-view architecture empowers our model to formulate a more thorough and robust representation, capturing both the overarching structural context and salient local details concurrently. Extensive experiments validate the effectiveness and strong generalization ability of the proposed framework. It consistently outperforms various baseline methods across diverse datasets (MusicEval and AES-Natural), MOS prediction backbones (including a CLAP-based model and AudioBox-Aesthetics), and different audio generation systems, achieving a relative improvement of 10.39% in system-level Spearman's rank correlation coefficient (SRCC) over the widely-used average pooling approach. 

**Abstract (ZH)**: 一种双分辨率注意统计聚合（DRASP）框架在语音质量均方意见评分（MOS）预测中的应用 

---
# zkLoRA: Fine-Tuning Large Language Models with Verifiable Security via Zero-Knowledge Proofs 

**Title (ZH)**: zkLoRA: 通过零知识证明实现可验证安全的大规模语言模型微调 

**Authors**: Guofu Liao, Taotao Wang, Shengli Zhang, Jiqun Zhang, Shi Long, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2508.21393)  

**Abstract**: Fine-tuning large language models (LLMs) is crucial for adapting them to specific tasks, yet it remains computationally demanding and raises concerns about correctness and privacy, particularly in untrusted environments. Although parameter-efficient methods like Low-Rank Adaptation (LoRA) significantly reduce resource requirements, ensuring the security and verifiability of fine-tuning under zero-knowledge constraints remains an unresolved challenge. To address this, we introduce zkLoRA, the first framework to integrate LoRA fine-tuning with zero-knowledge proofs (ZKPs), achieving provable security and correctness. zkLoRA employs advanced cryptographic techniques -- such as lookup arguments, sumcheck protocols, and polynomial commitments -- to verify both arithmetic and non-arithmetic operations in Transformer-based architectures. The framework provides end-to-end verifiability for forward propagation, backward propagation, and parameter updates during LoRA fine-tuning, while safeguarding the privacy of model parameters and training data. Leveraging GPU-based implementations, zkLoRA demonstrates practicality and efficiency through experimental validation on open-source LLMs like LLaMA, scaling up to 13 billion parameters. By combining parameter-efficient fine-tuning with ZKPs, zkLoRA bridges a critical gap, enabling secure and trustworthy deployment of LLMs in sensitive or untrusted environments. 

**Abstract (ZH)**: 基于零知识证明的LoRA细调框架：实现可验证的安全与正确性 

---
# AllSummedUp: un framework open-source pour comparer les metriques d'evaluation de resume 

**Title (ZH)**: AllSummedUp: 一个开源框架，用于比较摘要评估指标 

**Authors**: Tanguy Herserant, Vincent Guigue  

**Link**: [PDF](https://arxiv.org/pdf/2508.21389)  

**Abstract**: This paper investigates reproducibility challenges in automatic text summarization evaluation. Based on experiments conducted across six representative metrics ranging from classical approaches like ROUGE to recent LLM-based methods (G-Eval, SEval-Ex), we highlight significant discrepancies between reported performances in the literature and those observed in our experimental setting. We introduce a unified, open-source framework, applied to the SummEval dataset and designed to support fair and transparent comparison of evaluation metrics. Our results reveal a structural trade-off: metrics with the highest alignment with human judgments tend to be computationally intensive and less stable across runs. Beyond comparative analysis, this study highlights key concerns about relying on LLMs for evaluation, stressing their randomness, technical dependencies, and limited reproducibility. We advocate for more robust evaluation protocols including exhaustive documentation and methodological standardization to ensure greater reliability in automatic summarization assessment. 

**Abstract (ZH)**: 本文探讨了自动文本摘要评估中的重现性挑战。基于在六个代表性指标上进行的实验（包括ROUGE等经典方法和G-Eval、SEval-Ex等 recent LLM-based方法），我们指出了文献中报告的性能与我们在实验设置中观察到的显著差异。我们引入了一个统一的开源框架，应用于SummEval数据集，旨在支持评估指标的公平和透明比较。研究结果揭示了一个结构性权衡：与人类判断最一致的指标往往是计算密集型且在不同运行中不够稳定的。除了比较分析，本研究还强调了依赖LLMs进行评估的关键问题，突出了它们的随机性、技术依赖性和有限的重现性。我们倡导更 robust 的评估协议，包括详尽的文档记录和方法标准化，以确保自动摘要评估的更高可靠性。 

---
# Normality and the Turing Test 

**Title (ZH)**: 正常性与图灵测试 

**Authors**: Alexandre Kabbach  

**Link**: [PDF](https://arxiv.org/pdf/2508.21382)  

**Abstract**: This paper proposes to revisit the Turing test through the concept of normality. Its core argument is that the statistical interpretation of the normal--understood as the average both in the normative and mathematical sense of the term--proves useful for understanding the Turing test in at least two ways. First, in the sense that the Turing test targets normal/average rather than exceptional human intelligence, so that successfully passing the test requires building machines that "make mistakes" and display imperfect behavior just like normal/average humans. Second, in the sense that the Turing test is a statistical test where judgments of intelligence are never carried out by a single "average" judge (understood as non-expert) but always by a full jury. As such, the notion of "average human interrogator" that Turing talks about in his original paper should be understood primarily as referring to a mathematical abstraction made of the normalized aggregate of individual judgments of multiple judges. In short, this paper argues that the Turing test is a test of normal intelligence as assessed by a normal judge characterizing the average judgment of a pool of human interrogators. Its conclusions are twofold. First, it argues that large language models such as ChatGPT are unlikely to pass the Turing test as those models precisely target exceptional rather than normal/average human intelligence. As such, they constitute models of what it proposes to call artificial smartness rather than artificial intelligence per se. Second, it argues that the core question of whether the Turing test can contribute anything to the understanding of human cognition is that of whether the human mind is really reducible to the normal/average mind--a question which largely extends beyond the Turing test itself and questions the conceptual underpinnings of the normalist paradigm it belongs to. 

**Abstract (ZH)**: 通过正态性的概念重访图灵测试：图灵测试是对正态智能的统计评估 

---
# Iterative Inference in a Chess-Playing Neural Network 

**Title (ZH)**: 棋弈神经网络中的迭代推理 

**Authors**: Elias Sandmann, Sebastian Lapuschkin, Wojciech Samek  

**Link**: [PDF](https://arxiv.org/pdf/2508.21380)  

**Abstract**: Do neural networks build their representations through smooth, gradual refinement, or via more complex computational processes? We investigate this by extending the logit lens to analyze the policy network of Leela Chess Zero, a superhuman chess engine. We find strong monotonic trends in playing strength and puzzle-solving ability across layers, yet policy distributions frequently follow non-smooth trajectories. Evidence for this includes correct puzzle solutions that are discovered early but subsequently discarded, move rankings that remain poorly correlated with final outputs, and high policy divergence until late in the network. These findings contrast with the smooth distributional convergence typically observed in language models. 

**Abstract (ZH)**: 神经网络是通过平滑、渐进的细化还是通过更为复杂的计算过程构建其表示？我们通过将logit镜头扩展应用于分析超人棋弈引擎Leela Chess Zero的策略网络，来探讨这一问题。我们在各层中发现了较强的单调趋势，但在策略分布上却经常跟随非平滑的轨迹。这包括早期发现正确的解谜方案但随后被丢弃、移动排名与最终输出的相关性较差，以及网络较晚阶段政策 divergence 较高。这些发现与语言模型中通常观察到的平滑的分布收敛形成对比。 

---
# RoboInspector: Unveiling the Unreliability of Policy Code for LLM-enabled Robotic Manipulation 

**Title (ZH)**: RoboInspector: 揭示基于LLM的机器人操作中策略代码的不可靠性 

**Authors**: Chenduo Ying, Linkang Du, Peng Cheng, Yuanchao Shu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21378)  

**Abstract**: Large language models (LLMs) demonstrate remarkable capabilities in reasoning and code generation, enabling robotic manipulation to be initiated with just a single instruction. The LLM carries out various tasks by generating policy code required to control the robot. Despite advances in LLMs, achieving reliable policy code generation remains a significant challenge due to the diverse requirements of real-world tasks and the inherent complexity of user instructions. In practice, different users may provide distinct instructions to drive the robot for the same task, which may cause the unreliability of policy code generation. To bridge this gap, we design RoboInspector, a pipeline to unveil and characterize the unreliability of the policy code for LLM-enabled robotic manipulation from two perspectives: the complexity of the manipulation task and the granularity of the instruction. We perform comprehensive experiments with 168 distinct combinations of tasks, instructions, and LLMs in two prominent frameworks. The RoboInspector identifies four main unreliable behaviors that lead to manipulation failure. We provide a detailed characterization of these behaviors and their underlying causes, giving insight for practical development to reduce unreliability. Furthermore, we introduce a refinement approach guided by failure policy code feedback that improves the reliability of policy code generation by up to 35% in LLM-enabled robotic manipulation, evaluated in both simulation and real-world environments. 

**Abstract (ZH)**: 大型语言模型在机器人 manipulation 中的政策代码生成可靠性研究：从任务复杂性和指令粒度视角探索不确定行为及其改进方法 

---
# Challenges and Applications of Large Language Models: A Comparison of GPT and DeepSeek family of models 

**Title (ZH)**: 大型语言模型的挑战与应用：GPT与DeepSeek家族模型的比较 

**Authors**: Shubham Sharma, Sneha Tuli, Narendra Badam  

**Link**: [PDF](https://arxiv.org/pdf/2508.21377)  

**Abstract**: Large Language Models (LLMs) are transforming AI across industries, but their development and deployment remain complex. This survey reviews 16 key challenges in building and using LLMs and examines how these challenges are addressed by two state-of-the-art models with unique approaches: OpenAI's closed source GPT-4o (May 2024 update) and DeepSeek-V3-0324 (March 2025), a large open source Mixture-of-Experts model. Through this comparison, we showcase the trade-offs between closed source models (robust safety, fine-tuned reliability) and open source models (efficiency, adaptability). We also explore LLM applications across different domains (from chatbots and coding tools to healthcare and education), highlighting which model attributes are best suited for each use case. This article aims to guide AI researchers, developers, and decision-makers in understanding current LLM capabilities, limitations, and best practices. 

**Abstract (ZH)**: 大规模语言模型（LLMs）正在跨行业transform AI，但其研发和部署仍然复杂。本文回顾了构建和使用LLMs的16个关键挑战，并探讨了两个采用独特方法的前沿模型OpenAI的GPT-4o（2024年5月更新）和DeepSeek-V3-0324（2025年3月）是如何应对这些挑战的。通过比较这两种模型，我们展示了闭源模型（稳健的安全性，微调的可靠性）与开源模型（效率，适应性）之间的权衡。同时，我们探讨了LLMs在不同领域的应用（从聊天机器人和编码工具到医疗和教育），指出了哪些模型特性最适合每种应用场景。本文旨在指导AI研究人员、开发人员和决策者了解当前LLM的能力、局限性和最佳实践。 

---
# EconAgentic in DePIN Markets: A Large Language Model Approach to the Sharing Economy of Decentralized Physical Infrastructure 

**Title (ZH)**: 基于大语言模型的DePIN市场经济代理研究：去中心化物理基础设施共享经济探索 

**Authors**: Yulin Liu, Mocca Schweitzer  

**Link**: [PDF](https://arxiv.org/pdf/2508.21368)  

**Abstract**: The Decentralized Physical Infrastructure (DePIN) market is revolutionizing the sharing economy through token-based economics and smart contracts that govern decentralized operations. By 2024, DePIN projects have exceeded \$10 billion in market capitalization, underscoring their rapid growth. However, the unregulated nature of these markets, coupled with the autonomous deployment of AI agents in smart contracts, introduces risks such as inefficiencies and potential misalignment with human values. To address these concerns, we introduce EconAgentic, a Large Language Model (LLM)-powered framework designed to mitigate these challenges. Our research focuses on three key areas: 1) modeling the dynamic evolution of DePIN markets, 2) evaluating stakeholders' actions and their economic impacts, and 3) analyzing macroeconomic indicators to align market outcomes with societal goals. Through EconAgentic, we simulate how AI agents respond to token incentives, invest in infrastructure, and adapt to market conditions, comparing AI-driven decisions with human heuristic benchmarks. Our results show that EconAgentic provides valuable insights into the efficiency, inclusion, and stability of DePIN markets, contributing to both academic understanding and practical improvements in the design and governance of decentralized, tokenized economies. 

**Abstract (ZH)**: 基于代币经济和智能合约的去中心化物理基础设施（DePIN）市场正在通过去中心化运营革新共享经济。到2024年，DePIN项目市场资本已超过100亿美元，凸显其迅猛增长。然而，这些市场的无监管性质以及智能合约中自主部署的人工智能代理带来的潜在风险，如低效率和与人类价值取向的偏差，需要加以应对。为此，我们提出了EconAgentic框架，这是一种以大规模语言模型为动力的框架，旨在缓解这些挑战。我们的研究集中在三个方面：1） modeling the dynamic evolution of DePIN markets（建模DePIN市场的动态演变），2） evaluating stakeholders' actions and their economic impacts（评估相关方行为及其经济影响），3） analyzing macroeconomic indicators to align market outcomes with societal goals（分析宏观经济指标以使市场结果与社会目标保持一致）。通过EconAgentic，我们模拟了人工智能代理如何响应代币激励、投资基础设施并适应市场条件，并将AI驱动的决策与人类启发式基准进行了对比。结果表明，EconAgentic为理解和改进去中心化代币化经济的设计与治理提供了有价值的见解。 

---
# Adaptive Heavy-Tailed Stochastic Gradient Descent 

**Title (ZH)**: 自适应重尾随机梯度下降 

**Authors**: Bodu Gong, Gustavo Enrique Batista, Pierre Lafaye de Micheaux  

**Link**: [PDF](https://arxiv.org/pdf/2508.21353)  

**Abstract**: In the era of large-scale neural network models, optimization algorithms often struggle with generalization due to an overreliance on training loss. One key insight widely accepted in the machine learning community is the idea that wide basins (regions around a local minimum where the loss increases gradually) promote better generalization by offering greater stability to small changes in input data or model parameters. In contrast, sharp minima are typically more sensitive and less stable. Motivated by two key empirical observations - the inherent heavy-tailed distribution of gradient noise in stochastic gradient descent and the Edge of Stability phenomenon during neural network training, in which curvature grows before settling at a plateau, we introduce Adaptive Heavy Tailed Stochastic Gradient Descent (AHTSGD). The algorithm injects heavier-tailed noise into the optimizer during the early stages of training to enhance exploration and gradually transitions to lighter-tailed noise as sharpness stabilizes. By dynamically adapting to the sharpness of the loss landscape throughout training, AHTSGD promotes accelerated convergence to wide basins. AHTSGD is the first algorithm to adjust the nature of injected noise into an optimizer based on the Edge of Stability phenomenon. AHTSGD consistently outperforms SGD and other noise-based methods on benchmarks like MNIST and CIFAR-10, with marked gains on noisy datasets such as SVHN. It ultimately accelerates early training from poor initializations and improves generalization across clean and noisy settings, remaining robust to learning rate choices. 

**Abstract (ZH)**: 在大规模神经网络模型时代，优化算法往往因过度依赖训练损失而难以实现泛化。机器学习社区广泛接受的一个关键见解是，宽盆地（围绕局部极小值且损失逐渐增加的区域）通过提供对输入数据或模型参数微小变化的更大稳定性来促进更好的泛化。相比之下，尖锐极小值通常更为敏感且不稳定。基于两个关键的经验观察——随机梯度下降中的固有重尾梯度噪声分布以及神经网络训练中的临界稳定现象，在该现象中曲率增长后再趋于平台，我们引入了自适应重尾随机梯度下降（AHTSGD）。该算法在训练初期向优化器注入更重尾的噪声以增强探索，并随着尖锐度的稳定逐步过渡到较轻尾的噪声。通过在整个训练过程中动态适应损失景观的尖锐度，AHTSGD 加速了对宽盆地的收敛。AHTSGD 是首个基于临界稳定现象调整注入噪声性质的算法。在 MNIST 和 CIFAR-10 等基准测试中，AHTSGD 一致优于 SGD 和其他基于噪声的方法，并在 SVHN 等噪声数据集上表现出显著改进。它加速了从较差初始条件的早期训练，并在干净和噪声环境中改善了泛化能力，且对学习率的选择具有鲁棒性。 

---
# DLGAN : Time Series Synthesis Based on Dual-Layer Generative Adversarial Networks 

**Title (ZH)**: DLGAN：基于双层生成对抗网络的时间序列合成 

**Authors**: Xuan Hou, Shuhan Liu, Zhaohui Peng, Yaohui Chu, Yue Zhang, Yining Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.21340)  

**Abstract**: Time series synthesis is an effective approach to ensuring the secure circulation of time series data. Existing time series synthesis methods typically perform temporal modeling based on random sequences to generate target sequences, which often struggle to ensure the temporal dependencies in the generated time series. Additionally, directly modeling temporal features on random sequences makes it challenging to accurately capture the feature information of the original time series. To address the above issues, we propose a simple but effective generative model \textbf{D}ual-\textbf{L}ayer \textbf{G}enerative \textbf{A}dversarial \textbf{N}etworks, named \textbf{DLGAN}. The model decomposes the time series generation process into two stages: sequence feature extraction and sequence reconstruction. First, these two stages form a complete time series autoencoder, enabling supervised learning on the original time series to ensure that the reconstruction process can restore the temporal dependencies of the sequence. Second, a Generative Adversarial Network (GAN) is used to generate synthetic feature vectors that align with the real-time sequence feature vectors, ensuring that the generator can capture the temporal features from real time series. Extensive experiments on four public datasets demonstrate the superiority of this model across various evaluation metrics. 

**Abstract (ZH)**: 双层生成对抗网络DLGAN：一种有效的时序数据合成方法 

---
# Stairway to Fairness: Connecting Group and Individual Fairness 

**Title (ZH)**: 阶梯通往公平：群体公平与个体公平的连接 

**Authors**: Theresia Veronika Rampisela, Maria Maistro, Tuukka Ruotsalo, Falk Scholer, Christina Lioma  

**Link**: [PDF](https://arxiv.org/pdf/2508.21334)  

**Abstract**: Fairness in recommender systems (RSs) is commonly categorised into group fairness and individual fairness. However, there is no established scientific understanding of the relationship between the two fairness types, as prior work on both types has used different evaluation measures or evaluation objectives for each fairness type, thereby not allowing for a proper comparison of the two. As a result, it is currently not known how increasing one type of fairness may affect the other. To fill this gap, we study the relationship of group and individual fairness through a comprehensive comparison of evaluation measures that can be used for both fairness types. Our experiments with 8 runs across 3 datasets show that recommendations that are highly fair for groups can be very unfair for individuals. Our finding is novel and useful for RS practitioners aiming to improve the fairness of their systems. Our code is available at: this https URL. 

**Abstract (ZH)**: 推荐系统中的公平性：集体公平性和个体公平性的关系研究 

---
# Stage-Diff: Stage-wise Long-Term Time Series Generation Based on Diffusion Models 

**Title (ZH)**: 阶段差分：基于扩散模型的分阶段长期时间序列生成 

**Authors**: Xuan Hou, Shuhan Liu, Zhaohui Peng, Yaohui Chu, Yue Zhang, Yining Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.21330)  

**Abstract**: Generative models have been successfully used in the field of time series generation. However, when dealing with long-term time series, which span over extended periods and exhibit more complex long-term temporal patterns, the task of generation becomes significantly more challenging. Long-term time series exhibit long-range temporal dependencies, but their data distribution also undergoes gradual changes over time. Finding a balance between these long-term dependencies and the drift in data distribution is a key challenge. On the other hand, long-term time series contain more complex interrelationships between different feature sequences, making the task of effectively capturing both intra-sequence and inter-sequence dependencies another important challenge. To address these issues, we propose Stage-Diff, a staged generative model for long-term time series based on diffusion models. First, through stage-wise sequence generation and inter-stage information transfer, the model preserves long-term sequence dependencies while enabling the modeling of data distribution shifts. Second, within each stage, progressive sequence decomposition is applied to perform channel-independent modeling at different time scales, while inter-stage information transfer utilizes multi-channel fusion modeling. This approach combines the robustness of channel-independent modeling with the information fusion advantages of multi-channel modeling, effectively balancing the intra-sequence and inter-sequence dependencies of long-term time series. Extensive experiments on multiple real-world datasets validate the effectiveness of Stage-Diff in long-term time series generation tasks. 

**Abstract (ZH)**: 基于扩散模型的分阶段生成模型Stage-Diff在长时序生成中的应用 

---
# Locus: Agentic Predicate Synthesis for Directed Fuzzing 

**Title (ZH)**: Locus: 主动谓词综合用于定向模糊测试 

**Authors**: Jie Zhu, Chihao Shen, Ziyang Li, Jiahao Yu, Yizheng Chen, Kexin Pei  

**Link**: [PDF](https://arxiv.org/pdf/2508.21302)  

**Abstract**: Directed fuzzing aims to find program inputs that lead to specified target program states. It has broad applications, such as debugging system crashes, confirming reported bugs, and generating exploits for potential vulnerabilities. This task is inherently challenging because target states are often deeply nested in the program, while the search space manifested by numerous possible program inputs is prohibitively large. Existing approaches rely on branch distances or manually-specified constraints to guide the search; however, the branches alone are often insufficient to precisely characterize progress toward reaching the target states, while the manually specified constraints are often tailored for specific bug types and thus difficult to generalize to diverse target states and programs.
We present Locus, a novel framework to improve the efficiency of directed fuzzing. Our key insight is to synthesize predicates to capture fuzzing progress as semantically meaningful intermediate states, serving as milestones towards reaching the target states. When used to instrument the program under fuzzing, they can reject executions unlikely to reach the target states, while providing additional coverage guidance. To automate this task and generalize to diverse programs, Locus features an agentic framework with program analysis tools to synthesize and iteratively refine the candidate predicates, while ensuring the predicates strictly relax the target states to prevent false rejections via symbolic execution. Our evaluation shows that Locus substantially improves the efficiency of eight state-of-the-art fuzzers in discovering real-world vulnerabilities, achieving an average speedup of 41.6x. So far, Locus has found eight previously unpatched bugs, with one already acknowledged with a draft patch. 

**Abstract (ZH)**: 面向目标状态的定向 fuzzing 旨在找到导致指定目标程序状态的输入。该方法在很多领域有着广泛的应用，如调试系统崩溃、验证报告的错误、生成利用潜在漏洞的攻击手段。然而，目标状态往往深藏于程序内部，而由大量可能的程序输入表现出来的搜索空间又极其庞大，使得这项任务具有根本性挑战。现有方法依赖分支距离或手工指定约束来引导搜索；然而，单独的分支往往不足以精确地描述接近目标状态的进展，而手工指定的约束往往针对特定的错误类型，难以泛化到不同的目标状态和程序中。

我们提出了 Locus，一种改进定向 fuzzing 效率的新型框架。我们的核心洞察是生成谓词以捕获 fuzzing 进度作为语义上有意义的中间状态，作为达到目标状态的里程碑。当用于 fuzzing 程序的仪器化时，它们可以拒绝那些不可能达到目标状态的执行过程，同时提供额外的覆盖指导。为了使这一任务自动化并适应多种程序，Locus 特设了一个包含程序分析工具的代理框架，用于合成和迭代细化候选谓词，同时确保通过符号执行防止谓词产生误拒绝，从而严格放宽目标状态。我们的评估显示，Locus 显著提高了八种最先进的 fuzzing 工具发现真实漏洞的效率，平均加速了 41.6 倍。截至目前，Locus 已发现八个新的未修补错误，其中一个已被承认并附有草稿补丁。 

---
# MyGO: Memory Yielding Generative Offline-consolidation for Lifelong Learning Systems 

**Title (ZH)**: MyGO: 内存优化生成离线整合机制用于终身学习系统 

**Authors**: Shihao Ji, Zihui Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.21296)  

**Abstract**: Continual or Lifelong Learning aims to develop models capable of acquiring new knowledge from a sequence of tasks without catastrophically forgetting what has been learned before. Existing approaches often rely on storing samples from previous tasks (experience replay) or employing complex regularization terms to protect learned weights. However, these methods face challenges related to data privacy, storage limitations, and performance degradation when tasks are dissimilar. To address these challenges, we introduce MyGO (Memory Yielding Generative Offline-consolidation), a novel lifelong learning framework inspired by the biological wake-sleep cycle. During the "wake" phase, the system rapidly learns a new task and trains a compact generative model (Generative Memory, G-mem) to capture its data distribution. During the "sleep" phase, the system enters an offline state, using all learned G-mem models to generate pseudo-data ("dreams") and consolidate new and old knowledge into a core feature extractor via knowledge distillation. This approach obviates the need to store any raw data, retaining only compact generative models, which offers significant advantages in privacy and storage efficiency. We evaluate MyGO on computer vision (Split-MNIST) and natural language processing (Split-AG News) benchmarks, comparing it against a sequential fine-tuning baseline. The results demonstrate that MyGO significantly mitigates catastrophic forgetting and maintains high average accuracy across tasks, proving the framework's effectiveness and domain-generality. 

**Abstract (ZH)**: 持续学习或终身学习旨在开发能够在执行序列任务时获得新知识而不 catastrophically 忘记之前学习内容的模型。现有方法往往依赖于存储之前任务的样本（经验回放）或采用复杂的正则化项来保护学习到的权重。然而，这些方法面临数据隐私、存储限制以及任务不相似时性能下降的挑战。为解决这些挑战，我们引入了 MyGO（Memory Yielding Generative Offline-consolidation）这一新的终身学习框架，该框架受生物唤醒-睡眠循环的启发。在“唤醒”阶段，系统快速学习新任务并训练一个紧凑的生成模型（生成记忆，G-mem）来捕获其数据分布。在“睡眠”阶段，系统进入离线状态，利用所有已学习的 G-mem 模型生成伪数据（“梦境”）并通过知识蒸馏将新旧知识整合到核心特征提取器中。这种方法消除了存储任何原始数据的需求，仅保留紧凑的生成模型，从而在隐私和存储效率方面具有显著优势。我们在计算机视觉（Split-MNIST）和自然语言处理（Split-AG News）基准上评估了 MyGO，并将其与顺序微调基线进行对比。结果表明，MyGO 显著减少了 catastrophic 忘记现象，并在整个任务中保持了较高的平均准确性，证明了该框架的有效性和适用性。 

---
# BLUEX Revisited: Enhancing Benchmark Coverage with Automatic Captioning 

**Title (ZH)**: BLUEX 重访：通过自动字幕增强基准覆盖范围 

**Authors**: João Guilherme Alves Santos, Giovana Kerche Bonás, Thales Sales Almeida  

**Link**: [PDF](https://arxiv.org/pdf/2508.21294)  

**Abstract**: With the growing capabilities of Large Language Models (LLMs), there is an increasing need for robust evaluation methods, especially in multilingual and non-English contexts. We present an updated version of the BLUEX dataset, now including 2024-2025 exams and automatically generated image captions using state-of-the-art models, enhancing its relevance for data contamination studies in LLM pretraining. Captioning strategies increase accessibility to text-only models by more than 40%, producing 1,422 usable questions, more than doubling the number in the original BLUEX. We evaluated commercial and open-source LLMs and their ability to leverage visual context through captions. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）能力的不断增强，尤其是在多语言和非英语环境中，对 robust 评估方法的需求也在增加。我们更新了 BLUEX 数据集，其中包括 2024-2025 年的考试和使用先进模型自动生成的图像描述，增强了其在 LLM 预训练数据污染研究方面的相关性。图像描述策略将仅文本模型的访问性提高了超过 40%，生成了 1,422 个可用问题，数量是原始 BLUEX 的两倍多。我们评估了商业和开源 LLM 以及它们通过图像描述利用视觉上下文的能力。 

---
# Efficient Code Embeddings from Code Generation Models 

**Title (ZH)**: 来自代码生成模型的高效代码嵌入 

**Authors**: Daria Kryvosheieva, Saba Sturua, Michael Günther, Scott Martens, Han Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.21290)  

**Abstract**: jina-code-embeddings is a novel code embedding model suite designed to retrieve code from natural language queries, perform technical question-answering, and identify semantically similar code snippets across programming languages. It makes innovative use of an autoregressive backbone pre-trained on both text and code, generating embeddings via last-token pooling. We outline the training recipe and demonstrate state-of-the-art performance despite the relatively small size of the models, validating this approach to code embedding model construction. 

**Abstract (ZH)**: Jina-Code-Embeddings是一种新型代码嵌入模型套件，旨在从自然语言查询中检索代码、执行技术问答，并在不同编程语言中识别语义相似的代码片段。该模型利用预训练于文本和代码的自回归主干，通过最后一个词池化生成嵌入。我们详细阐述了训练方法，并展示了尽管模型相对较小仍能达到最先进的性能，验证了这种代码嵌入模型构建方法的有效性。 

---
# A Financial Brain Scan of the LLM 

**Title (ZH)**: LLM的财务脑扫描 

**Authors**: Hui Chen, Antoine Didisheim, Luciano Somoza, Hanqing Tian  

**Link**: [PDF](https://arxiv.org/pdf/2508.21285)  

**Abstract**: Emerging techniques in computer science make it possible to "brain scan" large language models (LLMs), identify the plain-English concepts that guide their reasoning, and steer them while holding other factors constant. We show that this approach can map LLM-generated economic forecasts to concepts such as sentiment, technical analysis, and timing, and compute their relative importance without reducing performance. We also show that models can be steered to be more or less risk-averse, optimistic, or pessimistic, which allows researchers to correct or simulate biases. The method is transparent, lightweight, and replicable for empirical research in the social sciences. 

**Abstract (ZH)**: 新兴的计算机科学技术使人们能够“扫描”大型语言模型（LLMs），识别指导其推理的简单英语概念，并在保持其他因素不变的情况下引导它们。我们展示了这种方法可以将LLM生成的经济预测映射到情感、技术分析和时间性等概念，并在不降低性能的情况下计算它们的相对重要性。我们还展示了可以通过引导模型变得更加或不那么风险averse、乐观或悲观，从而使研究人员能够纠正或模拟偏见。该方法具有透明性、轻量级和可重复性，适用于社会科学中的实证研究。 

---
# Deep Active Learning for Lung Disease Severity Classification from Chest X-rays: Learning with Less Data in the Presence of Class Imbalance 

**Title (ZH)**: 基于胸片的肺部疾病严重程度分类的深度主动学习：面向类别不平衡情况下的少数据学习 

**Authors**: Roy M. Gabriel, Mohammadreza Zandehshahvar, Marly van Assen, Nattakorn Kittisut, Kyle Peters, Carlo N. De Cecco, Ali Adibi  

**Link**: [PDF](https://arxiv.org/pdf/2508.21263)  

**Abstract**: To reduce the amount of required labeled data for lung disease severity classification from chest X-rays (CXRs) under class imbalance, this study applied deep active learning with a Bayesian Neural Network (BNN) approximation and weighted loss function. This retrospective study collected 2,319 CXRs from 963 patients (mean age, 59.2 $\pm$ 16.6 years; 481 female) at Emory Healthcare affiliated hospitals between January and November 2020. All patients had clinically confirmed COVID-19. Each CXR was independently labeled by 3 to 6 board-certified radiologists as normal, moderate, or severe. A deep neural network with Monte Carlo Dropout was trained using active learning to classify disease severity. Various acquisition functions were used to iteratively select the most informative samples from an unlabeled pool. Performance was evaluated using accuracy, area under the receiver operating characteristic curve (AU ROC), and area under the precision-recall curve (AU PRC). Training time and acquisition time were recorded. Statistical analysis included descriptive metrics and performance comparisons across acquisition strategies. Entropy Sampling achieved 93.7% accuracy (AU ROC, 0.91) in binary classification (normal vs. diseased) using 15.4% of the training data. In the multi-class setting, Mean STD sampling achieved 70.3% accuracy (AU ROC, 0.86) using 23.1% of the labeled data. These methods outperformed more complex and computationally expensive acquisition functions and significantly reduced labeling needs. Deep active learning with BNN approximation and weighted loss effectively reduces labeled data requirements while addressing class imbalance, maintaining or exceeding diagnostic performance. 

**Abstract (ZH)**: 基于贝叶斯神经网络近似和加权损失函数的深度主动学习在缓解类不平衡下减少胸片肺疾病严重程度分类所需标注数据量的研究 

---
# Breaking the Cold-Start Barrier: Reinforcement Learning with Double and Dueling DQNs 

**Title (ZH)**: 打破冷启动障碍：基于双Q值和 Dueling DQNs 的强化学习 

**Authors**: Minda Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.21259)  

**Abstract**: Recommender systems struggle to provide accurate suggestions to new users with limited interaction history, a challenge known as the cold-user problem. This paper proposes a reinforcement learning approach using Double and Dueling Deep Q-Networks (DQN) to dynamically learn user preferences from sparse feedback, enhancing recommendation accuracy without relying on sensitive demographic data. By integrating these advanced DQN variants with a matrix factorization model, we achieve superior performance on a large e-commerce dataset compared to traditional methods like popularity-based and active learning strategies. Experimental results show that our method, particularly Dueling DQN, reduces Root Mean Square Error (RMSE) for cold users, offering an effective solution for privacy-constrained environments. 

**Abstract (ZH)**: 基于双重视忆和 Dueling DQN 的强化学习推荐方法应对冷启动用户问题 

---
# Reinforcement Learning for Optimizing Large Qubit Array based Quantum Sensor Circuits 

**Title (ZH)**: 基于大型量子比特阵列的量子传感器电路优化的强化学习方法 

**Authors**: Laxmisha Ashok Attisara, Sathish Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.21253)  

**Abstract**: As the number of qubits in a sensor increases, the complexity of designing and controlling the quantum circuits grows exponentially. Manually optimizing these circuits becomes infeasible. Optimizing entanglement distribution in large-scale quantum circuits is critical for enhancing the sensitivity and efficiency of quantum sensors [5], [6]. This paper presents an engineering integration of reinforcement learning with tensor-network-based simulation (MPS) for scalable circuit optimization for optimizing quantum sensor circuits with up to 60 qubits. To enable efficient simulation and scalability, we adopt tensor network methods, specifically the Matrix Product State (MPS) representation, instead of traditional state vector or density matrix approaches. Our reinforcement learning agent learns to restructure circuits to maximize Quantum Fisher Information (QFI) and entanglement entropy while reducing gate counts and circuit depth. Experimental results show consistent improvements, with QFI values approaching 1, entanglement entropy in the 0.8-1.0 range, and up to 90% reduction in depth and gate count. These results highlight the potential of combining quantum machine learning and tensor networks to optimize complex quantum circuits under realistic constraints. 

**Abstract (ZH)**: 量子传感器电路的可扩展优化：基于强化学习和 tensor 网络方法的工程集成 

---
# Quantum Machine Learning for Optimizing Entanglement Distribution in Quantum Sensor Circuits 

**Title (ZH)**: 量子机器学习在量子传感器电路中优化纠缠分布中的应用 

**Authors**: Laxmisha Ashok Attisara, Sathish Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.21252)  

**Abstract**: In the rapidly evolving field of quantum computing, optimizing quantum circuits for specific tasks is crucial for enhancing performance and efficiency. More recently, quantum sensing has become a distinct and rapidly growing branch of research within the area of quantum science and technology. The field is expected to provide new opportunities, especially regarding high sensitivity and precision. Entanglement is one of the key factors in achieving high sensitivity and measurement precision [3]. This paper presents a novel approach utilizing quantum machine learning techniques to optimize entanglement distribution in quantum sensor circuits. By leveraging reinforcement learning within a quantum environment, we aim to optimize the entanglement layout to maximize Quantum Fisher Information (QFI) and entanglement entropy, which are key indicators of a quantum system's sensitivity and coherence, while minimizing circuit depth and gate counts. Our implementation, based on Qiskit, integrates noise models and error mitigation strategies to simulate realistic quantum environments. The results demonstrate significant improvements in circuit performance and sensitivity, highlighting the potential of machine learning in quantum circuit optimization by measuring high QFI and entropy in the range of 0.84-1.0 with depth and gate count reduction by 20-86%. 

**Abstract (ZH)**: 在量子计算快速发展的领域中，针对特定任务优化量子电路对于提升性能和效率至关重要。近年来，量子传感已成为量子科学和技术领域的一个 distinct 和迅速增长的研究分支。该领域预计将提供新的机会，特别是在高灵敏度和精度方面。缠结是实现高灵敏度和测量精度的关键因素之一 [3]。本文提出了一种新的方法，利用量子机器学习技术优化量子传感器电路中的纠缠分布。通过在量子环境中利用强化学习，我们旨在优化纠缠布局，以最大化量子费雪信息 (QFI) 和纠缠熵——这两者是量子系统灵敏度和相干性的关键指标——同时减少电路深度和门的操作次数。基于 Qiskit 的实现结合了噪声模型和错误缓解策略，以模拟现实的量子环境。结果表明，在减少 20-86% 的电路深度和门操作次数的情况下，可以显著提高电路性能和灵敏度，突显了机器学习在量子电路优化中的潜力，特别是在 QFI 和熵的范围为 0.84-1.0 的测量值方面。 

---
# A Mixture of Experts Gating Network for Enhanced Surrogate Modeling in External Aerodynamics 

**Title (ZH)**: 专家门控网络在外部流体力学代理模型中的增强建模 

**Authors**: Mohammad Amin Nabian, Sanjay Choudhry  

**Link**: [PDF](https://arxiv.org/pdf/2508.21249)  

**Abstract**: The computational cost associated with high-fidelity CFD simulations remains a significant bottleneck in the automotive design and optimization cycle. While ML-based surrogate models have emerged as a promising alternative to accelerate aerodynamic predictions, the field is characterized by a diverse and rapidly evolving landscape of specialized neural network architectures, with no single model demonstrating universal superiority. This paper introduces a novel meta-learning framework that leverages this architectural diversity as a strength. We propose a Mixture of Experts (MoE) model that employs a dedicated gating network to dynamically and optimally combine the predictions from three heterogeneous, state-of-the-art surrogate models: DoMINO, a decomposable multi-scale neural operator; X-MeshGraphNet, a scalable multi-scale graph neural network; and FigConvNet, a factorized implicit global convolution network. The gating network learns a spatially-variant weighting strategy, assigning credibility to each expert based on its localized performance in predicting surface pressure and wall shear stress fields. To prevent model collapse and encourage balanced expert contributions, we integrate an entropy regularization term into the training loss function. The entire system is trained and validated on the DrivAerML dataset, a large-scale, public benchmark of high-fidelity CFD simulations for automotive aerodynamics. Quantitative results demonstrate that the MoE model achieves a significant reduction in L-2 prediction error, outperforming not only the ensemble average but also the most accurate individual expert model across all evaluated physical quantities. This work establishes the MoE framework as a powerful and effective strategy for creating more robust and accurate composite surrogate models by synergistically combining the complementary strengths of specialized architectures. 

**Abstract (ZH)**: 基于高保真CFD模拟的计算成本仍然是汽车设计和优化周期中的一个重要瓶颈。虽然基于机器学习的代理模型已成为加速气动预测的有前途的替代方案，但该领域的特点是专业神经网络架构的多样性和快速演变，没有单一模型表现出普遍的优越性。本文引入了一种利用这种架构多样性作为优势的元学习框架。我们提出了一种专家混合模型（MoE），该模型使用专门的门控网络动态地最优地结合来自三个异构的最先进的代理模型：DoMINO（可分解的多尺度神经算子）、X-MeshGraphNet（可扩展的多尺度图神经网络）和FigConvNet（因子化的隐式全局卷积网络）的预测。门控网络学习一种空间变权重策略，基于每个专家在预测表面压力和壁面剪切应力场方面的局部性能赋予其可信度。为了防止模型退化并鼓励专家贡献的均衡，我们在训练损失函数中引入了一个熵正则化项。整个系统在DrivAerML数据集上进行训练和验证，该数据集是一个大规模的公共基准数据集，包含用于汽车气动学的高保真CFD模拟。定量结果表明，MoE模型在L-2预测误差上实现了显著的减少，不仅优于各专家模型的平均值，而且在所有评估的物理量中也优于最准确的单个专家模型。本文建立了MoE框架作为一种策略，通过协同结合特定架构的互补优势来创建更稳健和准确的复合代理模型。 

---
# Zero-Shot KWS for Children's Speech using Layer-Wise Features from SSL Models 

**Title (ZH)**: 基于SSL模型层wise特征的零样本儿童语音识别 

**Authors**: Subham Kutum, Abhijit Sinha, Hemant Kumar Kathania, Sudarsana Reddy Kadiri, Mahesh Chandra Govil  

**Link**: [PDF](https://arxiv.org/pdf/2508.21248)  

**Abstract**: Numerous methods have been proposed to enhance Keyword Spotting (KWS) in adult speech, but children's speech presents unique challenges for KWS systems due to its distinct acoustic and linguistic characteristics. This paper introduces a zero-shot KWS approach that leverages state-of-the-art self-supervised learning (SSL) models, including Wav2Vec2, HuBERT and Data2Vec. Features are extracted layer-wise from these SSL models and used to train a Kaldi-based DNN KWS system. The WSJCAM0 adult speech dataset was used for training, while the PFSTAR children's speech dataset was used for testing, demonstrating the zero-shot capability of our method. Our approach achieved state-of-the-art results across all keyword sets for children's speech. Notably, the Wav2Vec2 model, particularly layer 22, performed the best, delivering an ATWV score of 0.691, a MTWV score of 0.7003 and probability of false alarm and probability of miss of 0.0164 and 0.0547 respectively, for a set of 30 keywords. Furthermore, age-specific performance evaluation confirmed the system's effectiveness across different age groups of children. To assess the system's robustness against noise, additional experiments were conducted using the best-performing layer of the best-performing Wav2Vec2 model. The results demonstrated a significant improvement over traditional MFCC-based baseline, emphasizing the potential of SSL embeddings even in noisy conditions. To further generalize the KWS framework, the experiments were repeated for an additional CMU dataset. Overall the results highlight the significant contribution of SSL features in enhancing Zero-Shot KWS performance for children's speech, effectively addressing the challenges associated with the distinct characteristics of child speakers. 

**Abstract (ZH)**: 一种利用最先进的半监督学习模型提高儿童语音关键词识别的方法 

---
# HCQA: Hybrid Classical-Quantum Agent for Generating Optimal Quantum Sensor Circuits 

**Title (ZH)**: HCQA: 综合经典-量子代理生成最优量子传感器电路 

**Authors**: Ahmad Alomari, Sathish A. P. Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.21246)  

**Abstract**: This study proposes an HCQA for designing optimal Quantum Sensor Circuits (QSCs) to address complex quantum physics problems. The HCQA integrates computational intelligence techniques by leveraging a Deep Q-Network (DQN) for learning and policy optimization, enhanced by a quantum-based action selection mechanism based on the Q-values. A quantum circuit encodes the agent current state using Ry gates, and then creates a superposition of possible actions. Measurement of the circuit results in probabilistic action outcomes, allowing the agent to generate optimal QSCs by selecting sequences of gates that maximize the Quantum Fisher Information (QFI) while minimizing the number of gates. This computational intelligence-driven HCQA enables the automated generation of entangled quantum states, specifically the squeezed states, with high QFI sensitivity for quantum state estimation and control. Evaluation of the HCQA on a QSC that consists of two qubits and a sequence of Rx, Ry, and S gates demonstrates its efficiency in generating optimal QSCs with a QFI of 1. This work highlights the synergy between AI-driven learning and quantum computation, illustrating how intelligent agents can autonomously discover optimal quantum circuit designs for enhanced sensing and estimation tasks. 

**Abstract (ZH)**: 本研究提出了一种HCQA，用于设计最优量子传感器电路（QSCs），以解决复杂的量子物理问题。该HCQA通过利用深度Q网络（DQN）进行学习和策略优化，并结合基于Q值的量子动作选择机制，实现了计算智能技术的集成。量子电路使用Ry门将代理的当前状态编码，并创建可能动作的叠加态。测量电路结果生成概率性的动作结果，使代理能够通过选择最大化量子费舍尔信息（QFI）并最小化门的数量的门序列来生成最优QSCs。这种基于计算智能的HCQA能够自动生成纠缠量子态，特别是压缩态，这些态对量子态估计和控制具有高QFI敏感性。对由两个量子位和Rx、Ry、S门序列组成的QSC进行评估表明，该方法在生成具有QFI为1的最优QSCs方面表现出高效性。本研究突显了AI驱动学习与量子计算之间的协同作用，展示了智能代理如何自主发现用于增强传感和估计任务的最优量子电路设计。 

---
# Full-Frequency Temporal Patching and Structured Masking for Enhanced Audio Classification 

**Title (ZH)**: 全频段时间patches提取和结构化遮罩增强音频分类 

**Authors**: Aditya Makineni, Baocheng Geng, Qing Tian  

**Link**: [PDF](https://arxiv.org/pdf/2508.21243)  

**Abstract**: Transformers and State-Space Models (SSMs) have advanced audio classification by modeling spectrograms as sequences of patches. However, existing models such as the Audio Spectrogram Transformer (AST) and Audio Mamba (AuM) adopt square patching from computer vision, which disrupts continuous frequency patterns and produces an excessive number of patches, slowing training, and increasing computation. We propose Full-Frequency Temporal Patching (FFTP), a patching strategy that better matches the time-frequency asymmetry of spectrograms by spanning full frequency bands with localized temporal context, preserving harmonic structure, and significantly reducing patch count and computation. We also introduce SpecMask, a patch-aligned spectrogram augmentation that combines full-frequency and localized time-frequency masks under a fixed masking budget, enhancing temporal robustness while preserving spectral continuity. When applied on both AST and AuM, our patching method with SpecMask improves mAP by up to +6.76 on AudioSet-18k and accuracy by up to +8.46 on SpeechCommandsV2, while reducing computation by up to 83.26%, demonstrating both performance and efficiency gains. 

**Abstract (ZH)**: Transformer和状态空间模型通过将谱图建模为补丁序列推动了音频分类的发展。然而，现有的模型如音频谱图变换器（AST）和音频曼巴（AuM）采用源自计算机视觉的平方补丁策略，这破坏了连续的频率模式，产生了过多的补丁，从而减慢了训练速度并增加了计算量。我们提出了全频段时域补丁策略（FFTP），该策略通过局部时间上下文覆盖完整频率带宽更好地匹配谱图的时间-频率不对称性，保留谐波结构，并显著减少了补丁数量和计算量。我们还引入了SpecMask补丁对齐的谱图增强方法，在固定遮罩预算下结合了全频段和局部时频遮罩，增强了时间稳健性的同时保留了光谱连续性。在AST和AuM上应用我们的补丁方法和SpecMask在AudioSet-18k上的mAP最高提高了6.76%，在SpeechCommandsV2上的准确率最高提高了8.46%，同时计算量最高降低了83.26%，同時展示了性能和效率的双重提升。 

---
# Decoding Memories: An Efficient Pipeline for Self-Consistency Hallucination Detection 

**Title (ZH)**: 解码记忆：一种高效的一致性幻觉检测管道 

**Authors**: Weizhi Gao, Xiaorui Liu, Feiyi Wang, Dan Lu, Junqi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.21228)  

**Abstract**: Large language models (LLMs) have demonstrated impressive performance in both research and real-world applications, but they still struggle with hallucination. Existing hallucination detection methods often perform poorly on sentence-level generation or rely heavily on domain-specific knowledge. While self-consistency approaches help address these limitations, they incur high computational costs due to repeated generation. In this paper, we conduct the first study on identifying redundancy in self-consistency methods, manifested as shared prefix tokens across generations, and observe that non-exact-answer tokens contribute minimally to the semantic content. Based on these insights, we propose a novel Decoding Memory Pipeline (DMP) that accelerates generation through selective inference and annealed decoding. Being orthogonal to the model, dataset, decoding strategy, and self-consistency baseline, our DMP consistently improves the efficiency of multi-response generation and holds promise for extension to alignment and reasoning tasks. Extensive experiments show that our method achieves up to a 3x speedup without sacrificing AUROC performance. 

**Abstract (ZH)**: 大型语言模型在研究和实际应用中展现出了令人印象深刻的性能，但仍存在幻觉问题。现有的幻觉检测方法在句子级生成上表现不佳，或严重依赖领域特定知识。虽然自一致性方法有助于解决这些问题，但由于重复生成引起的高计算成本，限制了其应用。本文首次研究了自一致性方法中的冗余性，表现为生成过程中的共享前缀令牌，并观察到非精确答案令牌对语义内容的贡献极小。基于这些洞察，我们提出了一种新型解码记忆管道（DMP），通过选择性推理和退火解码加速生成过程。DMP与模型、数据集、解码策略和自一致性基线无关，能一致地提高多响应生成的效率，并有望扩展到对齐和推理任务中。广泛实验证明，我们的方法在不牺牲AUROC性能的情况下最多可实现3倍的加速。 

---
# Can Layer-wise SSL Features Improve Zero-Shot ASR Performance for Children's Speech? 

**Title (ZH)**: 层wise SSL特征能否提高儿童语音零样本ASR性能？ 

**Authors**: Abhijit Sinha, Hemant Kumar Kathania, Sudarsana Reddy Kadiri, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2508.21225)  

**Abstract**: Automatic Speech Recognition (ASR) systems often struggle to accurately process children's speech due to its distinct and highly variable acoustic and linguistic characteristics. While recent advancements in self-supervised learning (SSL) models have greatly enhanced the transcription of adult speech, accurately transcribing children's speech remains a significant challenge. This study investigates the effectiveness of layer-wise features extracted from state-of-the-art SSL pre-trained models - specifically, Wav2Vec2, HuBERT, Data2Vec, and WavLM in improving the performance of ASR for children's speech in zero-shot scenarios. A detailed analysis of features extracted from these models was conducted, integrating them into a simplified DNN-based ASR system using the Kaldi toolkit. The analysis identified the most effective layers for enhancing ASR performance on children's speech in a zero-shot scenario, where WSJCAM0 adult speech was used for training and PFSTAR children speech for testing. Experimental results indicated that Layer 22 of the Wav2Vec2 model achieved the lowest Word Error Rate (WER) of 5.15%, representing a 51.64% relative improvement over the direct zero-shot decoding using Wav2Vec2 (WER of 10.65%). Additionally, age group-wise analysis demonstrated consistent performance improvements with increasing age, along with significant gains observed even in younger age groups using the SSL features. Further experiments on the CMU Kids dataset confirmed similar trends, highlighting the generalizability of the proposed approach. 

**Abstract (ZH)**: 自动语音识别（ASR）系统往往难以准确处理儿童语音，因为儿童的语音具有独特的且高度变异性 acoustic 和 语言 特征。尽管最近自监督学习（SSL）模型在成人类语音转录方面取得了显著进步，但准确转录儿童语音仍然是一个重大挑战。本研究探讨了从最先进的 SSL 预训练模型（具体为 Wav2Vec2、HuBERT、Data2Vec 和 WavLM）提取的逐层特征在零样本场景中提高儿童语音 ASR 性能的有效性。通过 Kaldi 工具包将这些模型提取的特征整合到简化型 DNN 基础 ASR 系统中，详细分析了特征提取结果，并研究了 Wav2Vec2 模型第 22 层在零样本场景中的最佳性能，使用 WSJCAM0 成人语音进行训练，使用 PFSTAR 儿童语音进行测试。实验结果表明，Wav2Vec2 模型第 22 层实现了最低的词错误率（WER）5.15%，相对改进百分比为 51.64%，超过直接使用 Wav2Vec2 的零样本解码（WER 为 10.65%）。此外，按年龄组的分析显示随着年龄的增长持续性能提升，并且即使在较年轻年龄组中也观察到显著收益。进一步在 CMU Kids 数据集上的实验验证了类似趋势，突出了所提方法的一致性。 

---
# Generalizable Object Re-Identification via Visual In-Context Prompting 

**Title (ZH)**: 基于视觉情境提示的可迁移对象重识别 

**Authors**: Zhizhong Huang, Xiaoming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21222)  

**Abstract**: Current object re-identification (ReID) methods train domain-specific models (e.g., for persons or vehicles), which lack generalization and demand costly labeled data for new categories. While self-supervised learning reduces annotation needs by learning instance-wise invariance, it struggles to capture \textit{identity-sensitive} features critical for ReID. This paper proposes Visual In-Context Prompting~(VICP), a novel framework where models trained on seen categories can directly generalize to unseen novel categories using only \textit{in-context examples} as prompts, without requiring parameter adaptation. VICP synergizes LLMs and vision foundation models~(VFM): LLMs infer semantic identity rules from few-shot positive/negative pairs through task-specific prompting, which then guides a VFM (\eg, DINO) to extract ID-discriminative features via \textit{dynamic visual prompts}. By aligning LLM-derived semantic concepts with the VFM's pre-trained prior, VICP enables generalization to novel categories, eliminating the need for dataset-specific retraining. To support evaluation, we introduce ShopID10K, a dataset of 10K object instances from e-commerce platforms, featuring multi-view images and cross-domain testing. Experiments on ShopID10K and diverse ReID benchmarks demonstrate that VICP outperforms baselines by a clear margin on unseen categories. Code is available at this https URL. 

**Abstract (ZH)**: 当前目标再识别（ReID）方法训练领域特定模型（例如，针对人员或车辆），这些模型缺乏泛化能力，并且需要为新类别标注大量数据。虽然自我监督学习通过学习实例不变性减少了标注需求，但难以捕捉到再识别任务中至关重要的身份敏感特征。本文提出了一种新颖的视觉上下文提示框架（VICP），该框架使在已见类别上训练的模型可以直接通过仅使用上下文示例作为提示来泛化到未见的新类别，而不需要参数调整。VICP 结合了语言大模型（LLMs）和视觉基础模型（VFM）：LLMs 通过特定任务的提示从少量正负配对中推断语义身份规则，然后指导 VFM（例如，DINO）通过动态视觉提示提取身份鉴别特征。通过将从LLM推断出的语义概念与VFM的先验对齐，VICP使得模型能够泛化到新类别，从而消除了针对特定数据集的重新训练需求。为了支持评估，我们引入了包含10K个电子商务平台对象实例的ShopID10K数据集，该数据集包含多视角图像和跨域测试。在ShopID10K和多样化的ReID基准测试上进行的实验表明，VICP在未见类别上的性能明显优于基线方法。代码可在以下链接获取。 

---
# Enhancing Robustness of Autoregressive Language Models against Orthographic Attacks via Pixel-based Approach 

**Title (ZH)**: 通过基于像素的方法增强自回归语言模型对抗拼写攻击的鲁棒性 

**Authors**: Han Yang, Jian Lan, Yihong Liu, Hinrich Schütze, Thomas Seidl  

**Link**: [PDF](https://arxiv.org/pdf/2508.21206)  

**Abstract**: Autoregressive language models are vulnerable to orthographic attacks, where input text is perturbed with characters from multilingual alphabets, leading to substantial performance degradation. This vulnerability primarily stems from the out-of-vocabulary issue inherent in subword tokenizers and their embeddings. To address this limitation, we propose a pixel-based generative language model that replaces the text-based embeddings with pixel-based representations by rendering words as individual images. This design provides stronger robustness to noisy inputs, while an extension of compatibility to multilingual text across diverse writing systems. We evaluate the proposed method on the multilingual LAMBADA dataset, WMT24 dataset and the SST-2 benchmark, demonstrating both its resilience to orthographic noise and its effectiveness in multilingual settings. 

**Abstract (ZH)**: 自回归语言模型易受拼写攻击的影响，输入文本通过加入多语言字母表中的字符进行扰动，导致性能显著下降。这一弱点主要源于亚词分词器及其嵌入中存在的未登录词问题。为解决这一限制，我们提出了一种基于像素的生成语言模型，通过将单词渲染为单独的图像来替代基于文本的嵌入，从而提供更强的对噪声输入的鲁棒性，并扩展其在多种书写系统中的兼容性。我们在多语言LAMBADA数据集、WMT24数据集和SST-2基准上评估了所提出的方法，证明了其对拼写噪声的抗御能力和在多语言环境中的有效性。 

---
# Improving Aviation Safety Analysis: Automated HFACS Classification Using Reinforcement Learning with Group Relative Policy Optimization 

**Title (ZH)**: 改进航空安全分析：基于群体相对策略优化的强化学习自动HFACS分类 

**Authors**: Arash Ahmadi, Sarah Sharif, Yaser Banad  

**Link**: [PDF](https://arxiv.org/pdf/2508.21201)  

**Abstract**: Analyzing the human factors behind aviation accidents is crucial for preventing future incidents, yet traditional methods using the Human Factors Analysis and Classification System (HFACS) are limited by scalability and consistency. To address this, we introduce an automated HFACS classification framework for aviation safety analysis that utilizes Reinforcement Learning with Group Relative Policy Optimization (GRPO) to fine-tune a Llama-3.1 8B language model. Our approach incorporates a multi-component reward system tailored for aviation safety analysis and integrates synthetic data generation to overcome class imbalance in accident datasets. The resulting GRPO-optimized model achieved noticeable performance gains, including a 350% increase in exact match accuracy (from 0.0400 to 0.1800) and an improved partial match accuracy of 0.8800. Significantly, our specialized model outperforms state-of-the-art LLMs (Large Language Models), including GPT-5-mini and Gemini-2.5-fiash, on key metrics. This research also proposes exact match accuracy in multi-label HFACS classification problem as a new benchmarking methodology to evaluate the advanced reasoning capabilities of language models. Ultimately, our work validates that smaller, domain-optimized models can provide a computationally efficient and better solution for critical safety analysis. This approach makes powerful, low-latency deployment on resource-constrained edge devices feasible. 

**Abstract (ZH)**: 基于强化学习与组相对策略优化的航空安全分析自动化HFACS分类框架 

---
# Manifold Trajectories in Next-Token Prediction: From Replicator Dynamics to Softmax Equilibrium 

**Title (ZH)**: 流形轨迹在下一个词预测中的建模：从复制动态到Softmax平衡 

**Authors**: Christopher R. Lee-Jenkins  

**Link**: [PDF](https://arxiv.org/pdf/2508.21186)  

**Abstract**: Decoding in large language models is often described as scoring tokens and normalizing with softmax. We give a minimal, self-contained account of this step as a constrained variational principle on the probability simplex. The discrete, normalization-respecting ascent is the classical multiplicative-weights (entropic mirror) update; its continuous-time limit is the replicator flow. From these ingredients we prove that, for a fixed context and temperature, the next-token distribution follows a smooth trajectory inside the simplex and converges to the softmax equilibrium. This formalizes the common ``manifold traversal'' intuition at the output-distribution level. The analysis yields precise, practice-facing consequences: temperature acts as an exact rescaling of time along the same trajectory, while top-k and nucleus sampling restrict the flow to a face with identical guarantees. We also outline a controlled account of path-dependent score adjustments and their connection to loop-like, hallucination-style behavior. We make no claims about training dynamics or internal representations; those are deferred to future work. 

**Abstract (ZH)**: 大型语言模型中的解码通常被描述为对令牌进行评分并使用softmax进行规范化。我们以约束变分原理在概率单纯形上的形式给出这一步骤的 minimalist、自包含说明。离散的、遵守规范化准则的上升过程是经典的乘子权重（对数镜像）更新；其连续时间极限是复制流。从这些成分出发，我们证明，在固定上下文和温度的情况下，下一个令牌的分布沿单纯形内的平滑轨迹移动，并最终收敛到softmax稳态。这在输出分布层面上正式化了常见的“流形遍历”直觉。分析给出了精确的、面向实践的结果：温度作为时间沿相同轨迹的精确缩放，而top-k和nucleus采样将流限制在具有相同保证的面上。我们还概述了路径依赖得分调整的受控说明及其与环形幻觉行为的联系。我们不对训练动力学或内部表示发表任何声明；这些内容将留待未来工作。 

---
# BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design 

**Title (ZH)**: BED-LLM：基于LLM和贝叶斯实验设计的智能信息收集方法 

**Authors**: Deepro Choudhury, Sinead Williamson, Adam Goliński, Ning Miao, Freddie Bickford Smith, Michael Kirchhof, Yizhe Zhang, Tom Rainforth  

**Link**: [PDF](https://arxiv.org/pdf/2508.21184)  

**Abstract**: We propose a general-purpose approach for improving the ability of Large Language Models (LLMs) to intelligently and adaptively gather information from a user or other external source using the framework of sequential Bayesian experimental design (BED). This enables LLMs to act as effective multi-turn conversational agents and interactively interface with external environments. Our approach, which we call BED-LLM (Bayesian Experimental Design with Large Language Models), is based on iteratively choosing questions or queries that maximize the expected information gain (EIG) about the task of interest given the responses gathered previously. We show how this EIG can be formulated in a principled way using a probabilistic model derived from the LLM's belief distribution and provide detailed insights into key decisions in its construction. Further key to the success of BED-LLM are a number of specific innovations, such as a carefully designed estimator for the EIG, not solely relying on in-context updates for conditioning on previous responses, and a targeted strategy for proposing candidate queries. We find that BED-LLM achieves substantial gains in performance across a wide range of tests based on the 20-questions game and using the LLM to actively infer user preferences, compared to direct prompting of the LLM and other adaptive design strategies. 

**Abstract (ZH)**: 基于贝叶斯实验设计的大语言模型通用增强方法 

---
# FUTURE: Flexible Unlearning for Tree Ensemble 

**Title (ZH)**: 未来：树ensemble的灵活遗忘技术 

**Authors**: Ziheng Chen, Jin Huang, Jiali Cheng, Yuchan Guo, Mengjie Wang, Lalitesh Morishetti, Kaushiki Nag, Hadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2508.21181)  

**Abstract**: Tree ensembles are widely recognized for their effectiveness in classification tasks, achieving state-of-the-art performance across diverse domains, including bioinformatics, finance, and medical diagnosis. With increasing emphasis on data privacy and the \textit{right to be forgotten}, several unlearning algorithms have been proposed to enable tree ensembles to forget sensitive information. However, existing methods are often tailored to a particular model or rely on the discrete tree structure, making them difficult to generalize to complex ensembles and inefficient for large-scale datasets. To address these limitations, we propose FUTURE, a novel unlearning algorithm for tree ensembles. Specifically, we formulate the problem of forgetting samples as a gradient-based optimization task. In order to accommodate non-differentiability of tree ensembles, we adopt the probabilistic model approximations within the optimization framework. This enables end-to-end unlearning in an effective and efficient manner. Extensive experiments on real-world datasets show that FUTURE yields significant and successful unlearning performance. 

**Abstract (ZH)**: 树ensemble在分类任务中广受认可，能够在生物信息学、金融和医疗诊断等领域达到最先进性能。随着对数据隐私和“被遗忘的权利”重视的增加，已经提出了多种遗忘算法，使树ensemble能够遗忘敏感信息。然而，现有方法通常针对特定模型或依赖于离散的树结构，难以泛化到复杂的ensemble并对于大规模数据集效率低下。为解决这些限制，我们提出了FUTURE，一种新的树ensemble遗忘算法。具体而言，我们将遗忘样本的问题形式化为基于梯度的优化任务。为了适应树ensemble的非可微性，我们在优化框架中采用了概率模型近似。这使得端到端的遗忘在有效且高效的方式下实现。在真正的数据集上的广泛实验表明，FUTURE能够在显著且成功地实现遗忘方面表现出色。 

---
# Deep Residual Echo State Networks: exploring residual orthogonal connections in untrained Recurrent Neural Networks 

**Title (ZH)**: 深度残余回声状态网络：探究未训练递归神经网络中的残余正交连接 

**Authors**: Matteo Pinna, Andrea Ceni, Claudio Gallicchio  

**Link**: [PDF](https://arxiv.org/pdf/2508.21172)  

**Abstract**: Echo State Networks (ESNs) are a particular type of untrained Recurrent Neural Networks (RNNs) within the Reservoir Computing (RC) framework, popular for their fast and efficient learning. However, traditional ESNs often struggle with long-term information processing. In this paper, we introduce a novel class of deep untrained RNNs based on temporal residual connections, called Deep Residual Echo State Networks (DeepResESNs). We show that leveraging a hierarchy of untrained residual recurrent layers significantly boosts memory capacity and long-term temporal modeling. For the temporal residual connections, we consider different orthogonal configurations, including randomly generated and fixed-structure configurations, and we study their effect on network dynamics. A thorough mathematical analysis outlines necessary and sufficient conditions to ensure stable dynamics within DeepResESN. Our experiments on a variety of time series tasks showcase the advantages of the proposed approach over traditional shallow and deep RC. 

**Abstract (ZH)**: 基于时间剩余连接的深度未训练循环神经网络：深残差回声状态网络（DeepResESNs） 

---
# Quantifying Label-Induced Bias in Large Language Model Self- and Cross-Evaluations 

**Title (ZH)**: 量化标签诱导偏见在大规模语言模型自我评估和交叉评估中的影响 

**Authors**: Muskan Saraf, Sajjad Rezvani Boroujeni, Justin Beaudry, Hossein Abedi, Tom Bush  

**Link**: [PDF](https://arxiv.org/pdf/2508.21164)  

**Abstract**: Large language models (LLMs) are increasingly used to evaluate outputs, yet their judgments may be influenced. This study examines bias in self- and cross-model evaluations by ChatGPT, Gemini, and Claude under four conditions: no labels, true labels, and two false-label scenarios. Blog posts authored by each model were evaluated by all three using both overall preference voting and quality ratings for Coherence, Informativeness, and Conciseness, with all scores expressed as percentages for direct comparison. Results reveal striking asymmetries: the "Claude" label consistently boosts scores, while the "Gemini" label consistently depresses them, regardless of actual content. False labels frequently reversed rankings, producing shifts of up to 50 percentage points in preference votes and up to 12 percentage points in converted quality ratings. Gemini's self-scores collapsed under true labels, while Claude's self-preference intensified. These findings show that perceived model identity can heavily distort high-level judgments and subtly influence detailed quality ratings, underscoring the need for blind or multimodel evaluation protocols to ensure fairness in LLM benchmarking. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地用于评估输出，但其判断可能会受到偏见的影响。本研究考察了ChatGPT、Gemini和Claude在四种条件下的自我评估和相互评估中的偏见：无标签、真实标签以及两种虚假标签情景。每种模型撰写的博客帖子分别由这三种模型使用整体偏好投票和一致性、信息性和简洁性评分进行评估，所有评分均以百分比形式表示，以便直接比较。结果显示存在明显的不对称性：“Claude”标签始终提升评分，而“Gemini”标签始终降低评分，与实际内容无关。虚假标签经常导致排名反转，评分偏好投票可上下波动50个百分点，转换后的质量评分可上下波动12个百分点。Gemini的自我评分在真实标签下崩溃，而Claude的自我偏好则加剧。这些发现表明，模型的身份感知可以严重失真高层次判断，并微妙地影响详细的质量评分，突显了在LLM基准测试中需要采用盲评估或多种模型评估协议以确保公平性的必要性。 

---
# RadGS-Reg: Registering Spine CT with Biplanar X-rays via Joint 3D Radiative Gaussians Reconstruction and 3D/3D Registration 

**Title (ZH)**: RadGS-Reg: 通过联合3D辐射高斯重建和3D/3D配准实现脊柱CT与双平面X射线的注册 

**Authors**: Ao Shen, Xueming Fu, Junfeng Jiang, Qiang Zeng, Ye Tang, Zhengming Chen, Luming Nong, Feng Wang, S. Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.21154)  

**Abstract**: Computed Tomography (CT)/X-ray registration in image-guided navigation remains challenging because of its stringent requirements for high accuracy and real-time performance. Traditional "render and compare" methods, relying on iterative projection and comparison, suffer from spatial information loss and domain gap. 3D reconstruction from biplanar X-rays supplements spatial and shape information for 2D/3D registration, but current methods are limited by dense-view requirements and struggles with noisy X-rays. To address these limitations, we introduce RadGS-Reg, a novel framework for vertebral-level CT/X-ray registration through joint 3D Radiative Gaussians (RadGS) reconstruction and 3D/3D registration. Specifically, our biplanar X-rays vertebral RadGS reconstruction module explores learning-based RadGS reconstruction method with a Counterfactual Attention Learning (CAL) mechanism, focusing on vertebral regions in noisy X-rays. Additionally, a patient-specific pre-training strategy progressively adapts the RadGS-Reg from simulated to real data while simultaneously learning vertebral shape prior knowledge. Experiments on in-house datasets demonstrate the state-of-the-art performance for both tasks, surpassing existing methods. The code is available at: this https URL. 

**Abstract (ZH)**: 基于联合3D辐射高斯重建与3D/3D配准的椎体级CT/X射线注册 

---
# WaveLLDM: Design and Development of a Lightweight Latent Diffusion Model for Speech Enhancement and Restoration 

**Title (ZH)**: WaveLLDM：轻量级潜在扩散模型的设计与开发，用于语音增强与恢复 

**Authors**: Kevin Putra Santoso, Rizka Wakhidatus Sholikah, Raden Venantius Hari Ginardi  

**Link**: [PDF](https://arxiv.org/pdf/2508.21153)  

**Abstract**: High-quality audio is essential in a wide range of applications, including online communication, virtual assistants, and the multimedia industry. However, degradation caused by noise, compression, and transmission artifacts remains a major challenge. While diffusion models have proven effective for audio restoration, they typically require significant computational resources and struggle to handle longer missing segments. This study introduces WaveLLDM (Wave Lightweight Latent Diffusion Model), an architecture that integrates an efficient neural audio codec with latent diffusion for audio restoration and denoising. Unlike conventional approaches that operate in the time or spectral domain, WaveLLDM processes audio in a compressed latent space, reducing computational complexity while preserving reconstruction quality. Empirical evaluations on the Voicebank+DEMAND test set demonstrate that WaveLLDM achieves accurate spectral reconstruction with low Log-Spectral Distance (LSD) scores (0.48 to 0.60) and good adaptability to unseen data. However, it still underperforms compared to state-of-the-art methods in terms of perceptual quality and speech clarity, with WB-PESQ scores ranging from 1.62 to 1.71 and STOI scores between 0.76 and 0.78. These limitations are attributed to suboptimal architectural tuning, the absence of fine-tuning, and insufficient training duration. Nevertheless, the flexible architecture that combines a neural audio codec and latent diffusion model provides a strong foundation for future development. 

**Abstract (ZH)**: 高质量音频在在线通信、虚拟助手和多媒体行业中至关重要，但由噪声、压缩和传输产生的降级仍然是一个主要挑战。尽管扩散模型在音频修复方面表现出有效性，但通常需要大量的计算资源，并且难以处理较长的缺失段落。本研究引入了WaveLLDM（波浪轻量级潜变量扩散模型），该架构将高效神经音频编解码器与潜变量扩散相结合，用于音频修复和去噪。与传统的在时域或谱域操作的方法不同，WaveLLDM在压缩的潜在空间中处理音频，从而降低计算复杂度同时保持重建质量。在Voicebank+DEMAND测试集上的实证评估表明，WaveLLDM实现了准确的谱重建，低对数谱距离（LSD）分数（0.48至0.60），并且对未见过的数据具有良好的适应性。然而，与最先进的方法相比，WaveLLDM在感知质量和语音清晰度方面表现仍然较差，WB-PESQ分数范围为1.62至1.71，STOI分数范围为0.76至0.78。这些局限性归因于架构调优不完善、缺乏微调以及训练时间不足。然而，结合神经音频编解码器和潜变量扩散模型的灵活架构为未来的发展提供了坚实的基础。 

---
# A Survey of Scientific Large Language Models: From Data Foundations to Agent Frontiers 

**Title (ZH)**: 科学大型语言模型综述：从数据基础到代理前沿 

**Authors**: Ming Hu, Chenglong Ma, Wei Li, Wanghan Xu, Jiamin Wu, Jucheng Hu, Tianbin Li, Guohang Zhuang, Jiaqi Liu, Yingzhou Lu, Ying Chen, Chaoyang Zhang, Cheng Tan, Jie Ying, Guocheng Wu, Shujian Gao, Pengcheng Chen, Jiashi Lin, Haitao Wu, Lulu Chen, Fengxiang Wang, Yuanyuan Zhang, Xiangyu Zhao, Feilong Tang, Encheng Su, Junzhi Ning, Xinyao Liu, Ye Du, Changkai Ji, Cheng Tang, Huihui Xu, Ziyang Chen, Ziyan Huang, Jiyao Liu, Pengfei Jiang, Yizhou Wang, Chen Tang, Jianyu Wu, Yuchen Ren, Siyuan Yan, Zhonghua Wang, Zhongxing Xu, Shiyan Su, Shangquan Sun, Runkai Zhao, Zhisheng Zhang, Yu Liu, Fudi Wang, Yuanfeng Ji, Yanzhou Su, Hongming Shan, Chunmei Feng, Jiahao Xu, Jiangtao Yan, Wenhao Tang, Diping Song, Lihao Liu, Yanyan Huang, Lequan Yu, Bin Fu, Shujun Wang, Xiaomeng Li, Xiaowei Hu, Yun Gu, Ben Fei, Zhongying Deng, Benyou Wang, Yuewen Cao, Minjie Shen, Haodong Duan, Jie Xu, Yirong Chen, Fang Yan, Hongxia Hao, Jielan Li, Jiajun Du, Yanbo Wang, Imran Razzak, Chi Zhang, Lijun Wu, Conghui He, Zhaohui Lu, Jinhai Huang, Yihao Liu, Fenghua Ling, Yuqiang Li, Aoran Wang, Qihao Zheng, Nanqing Dong, Tianfan Fu, Dongzhan Zhou, Yan Lu, Wenlong Zhang, Jin Ye, Jianfei Cai, Wanli Ouyang, Yu Qiao, Zongyuan Ge, Shixiang Tang, Junjun He  

**Link**: [PDF](https://arxiv.org/pdf/2508.21148)  

**Abstract**: Scientific Large Language Models (Sci-LLMs) are transforming how knowledge is represented, integrated, and applied in scientific research, yet their progress is shaped by the complex nature of scientific data. This survey presents a comprehensive, data-centric synthesis that reframes the development of Sci-LLMs as a co-evolution between models and their underlying data substrate. We formulate a unified taxonomy of scientific data and a hierarchical model of scientific knowledge, emphasizing the multimodal, cross-scale, and domain-specific challenges that differentiate scientific corpora from general natural language processing datasets. We systematically review recent Sci-LLMs, from general-purpose foundations to specialized models across diverse scientific disciplines, alongside an extensive analysis of over 270 pre-/post-training datasets, showing why Sci-LLMs pose distinct demands -- heterogeneous, multi-scale, uncertainty-laden corpora that require representations preserving domain invariance and enabling cross-modal reasoning. On evaluation, we examine over 190 benchmark datasets and trace a shift from static exams toward process- and discovery-oriented assessments with advanced evaluation protocols. These data-centric analyses highlight persistent issues in scientific data development and discuss emerging solutions involving semi-automated annotation pipelines and expert validation. Finally, we outline a paradigm shift toward closed-loop systems where autonomous agents based on Sci-LLMs actively experiment, validate, and contribute to a living, evolving knowledge base. Collectively, this work provides a roadmap for building trustworthy, continually evolving artificial intelligence (AI) systems that function as a true partner in accelerating scientific discovery. 

**Abstract (ZH)**: 科学大型语言模型（Sci-LLMs）正在改变知识在科学研究中的表示、整合和应用方式，但其进步受到科学数据复杂性的制约。本文综述从数据-centric 角度出发，重新构架 Sci-LLMs 的发展为模型与其底层数据基础的共生进化。我们制定了统一的科学数据分类体系和层次化的科学知识模型，强调了多模态、跨尺度和领域特定的挑战，这些挑战将科学语料库区分为一般的自然语言处理数据集。系统回顾了从通用基础模型到跨多个科学学科的专业化模型，并对超过270个预训练和后训练数据集进行了详尽分析，展示了Sci-LLMs独有的需求——异质性、多尺度、充满不确定性的语料库，这些语料库需要能够保持领域不变性和支持跨模态推理的表示。在评估中，我们检验了超过190个基准数据集，并跟踪了从静态考试转向过程导向和发现导向评估的转变，同时采用先进的评估协议。这些数据-centric 分析突出了科学数据发展中持续存在的问题，并讨论了涉及半自动化注释流水线和专家验证的新兴解决方案。最终，我们概述了一种范式转变，即基于Sci-LLMs的自主代理能够积极实验、验证，并为持续演进的知识库贡献内容。本文共同提供了一条路线图，用于构建可信赖的、持续演进的人工智能系统，使其作为真正合作伙伴加速科学发现。 

---
# HiddenObject: Modality-Agnostic Fusion for Multimodal Hidden Object Detection 

**Title (ZH)**: 隐含对象检测：模态无关的多模态隐含对象融合 

**Authors**: Harris Song, Tuan-Anh Vu, Sanjith Menon, Sriram Narasimhan, M. Khalid Jawed  

**Link**: [PDF](https://arxiv.org/pdf/2508.21135)  

**Abstract**: Detecting hidden or partially concealed objects remains a fundamental challenge in multimodal environments, where factors like occlusion, camouflage, and lighting variations significantly hinder performance. Traditional RGB-based detection methods often fail under such adverse conditions, motivating the need for more robust, modality-agnostic approaches. In this work, we present HiddenObject, a fusion framework that integrates RGB, thermal, and depth data using a Mamba-based fusion mechanism. Our method captures complementary signals across modalities, enabling enhanced detection of obscured or camouflaged targets. Specifically, the proposed approach identifies modality-specific features and fuses them in a unified representation that generalizes well across challenging scenarios. We validate HiddenObject across multiple benchmark datasets, demonstrating state-of-the-art or competitive performance compared to existing methods. These results highlight the efficacy of our fusion design and expose key limitations in current unimodal and naïve fusion strategies. More broadly, our findings suggest that Mamba-based fusion architectures can significantly advance the field of multimodal object detection, especially under visually degraded or complex conditions. 

**Abstract (ZH)**: 检测隐藏或部分遮蔽的物体仍然是多模态环境中的一项基本挑战，遮挡、迷彩和光照变化等因素显著阻碍了性能的提升。传统的基于RGB的检测方法在这些不利条件下往往失效，推动了对更为 robust、模态无关的方法的需求。在本文中，我们提出了一种融合框架HiddenObject，该框架使用Mamba机制融合RGB、热成像和深度数据。我们的方法在不同模态中捕捉互补信号，增强了对被遮挡或迷彩目标的检测能力。具体来说，所提出的方法识别出特定模态的特征，并在统一的表示中融合这些特征，以在复杂场景中泛化良好。我们在多个基准数据集中验证了HiddenObject，结果显示其性能与现有方法相当或优于现有方法。这些结果突显了我们融合设计的有效性，并揭示了当前单模态和简单的融合策略的关键局限性。更广泛地说，我们的发现表明，基于Mamba的融合架构能够显著推动多模态目标检测领域的进展，尤其是在视觉退化或复杂条件下。 

---
# R-4B: Incentivizing General-Purpose Auto-Thinking Capability in MLLMs via Bi-Mode Annealing and Reinforce Learning 

**Title (ZH)**: R-4B: 通过双模式退火和强化学习激励通用自思考能力在MLLMs中的应用 

**Authors**: Jie Jiang, Qi Yang, Bolin Ni, Shiming Xiang, Han Hu, Houwen Peng  

**Link**: [PDF](https://arxiv.org/pdf/2508.21113)  

**Abstract**: Multimodal Large Language Models (MLLMs) equipped with step-by-step thinking capabilities have demonstrated remarkable performance on complex reasoning problems. However, this thinking process is redundant for simple problems solvable without complex reasoning. To address this inefficiency, we propose R-4B, an auto-thinking MLLM, which can adaptively decide when to think based on problem complexity. The central idea of R-4B is to empower the model with both thinking and non-thinking capabilities using bi-mode annealing, and apply Bi-mode Policy Optimization~(BPO) to improve the model's accuracy in determining whether to activate the thinking process. Specifically, we first train the model on a carefully curated dataset spanning various topics, which contains samples from both thinking and non-thinking modes. Then it undergoes a second phase of training under an improved GRPO framework, where the policy model is forced to generate responses from both modes for each input query. Experimental results show that R-4B achieves state-of-the-art performance across 25 challenging benchmarks. It outperforms Qwen2.5-VL-7B in most tasks and achieves performance comparable to larger models such as Kimi-VL-A3B-Thinking-2506 (16B) on reasoning-intensive benchmarks with lower computational cost. 

**Abstract (ZH)**: 具有自适应思考能力的Multimodal Large Language模型R-4B：基于问题复杂度的自动思考机制 

---
# EmbodiedOneVision: Interleaved Vision-Text-Action Pretraining for General Robot Control 

**Title (ZH)**: EmbodiedOneVision: 交错的视觉-文本-行动预训练面向通用机器人控制 

**Authors**: Delin Qu, Haoming Song, Qizhi Chen, Zhaoqing Chen, Xianqiang Gao, Xinyi Ye, Qi Lv, Modi Shi, Guanghui Ren, Cheng Ruan, Maoqing Yao, Haoran Yang, Jiacheng Bao, Bin Zhao, Dong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.21112)  

**Abstract**: The human ability to seamlessly perform multimodal reasoning and physical interaction in the open world is a core goal for general-purpose embodied intelligent systems. Recent vision-language-action (VLA) models, which are co-trained on large-scale robot and visual-text data, have demonstrated notable progress in general robot control. However, they still fail to achieve human-level flexibility in interleaved reasoning and interaction. In this work, introduce EO-Robotics, consists of EO-1 model and EO-Data1.5M dataset. EO-1 is a unified embodied foundation model that achieves superior performance in multimodal embodied reasoning and robot control through interleaved vision-text-action pre-training. The development of EO-1 is based on two key pillars: (i) a unified architecture that processes multimodal inputs indiscriminately (image, text, video, and action), and (ii) a massive, high-quality multimodal embodied reasoning dataset, EO-Data1.5M, which contains over 1.5 million samples with emphasis on interleaved vision-text-action comprehension. EO-1 is trained through synergies between auto-regressive decoding and flow matching denoising on EO-Data1.5M, enabling seamless robot action generation and multimodal embodied reasoning. Extensive experiments demonstrate the effectiveness of interleaved vision-text-action learning for open-world understanding and generalization, validated through a variety of long-horizon, dexterous manipulation tasks across multiple embodiments. This paper details the architecture of EO-1, the data construction strategy of EO-Data1.5M, and the training methodology, offering valuable insights for developing advanced embodied foundation models. 

**Abstract (ZH)**: 人类能够在开放世界中无缝进行多模态推理和物理交互的能力是通用体态智能系统的核心目标。近期的视觉-语言-动作（VLA）模型在大规模机器人和视觉-文本数据上的共同训练中，展示了在通用机器人控制方面显著的进步。然而，它们仍然无法达到人类级别的灵活的交互与推理能力。在这项工作中，我们引入了EO-Robotics，包括EO-1模型和EO-Data1.5M数据集。EO-1是一种统一的体态基础模型，通过交织的视觉-文本-动作预训练，在多模态体态推理和机器人控制方面表现出优越性能。EO-1的发展基于两大关键支柱：（i）一种可以无差别处理多模态输入（图像、文本、视频和动作）的统一架构，以及（ii）一个大规模的高质量多模态体态推理数据集EO-Data1.5M，该数据集包含超过150万个样本，重点放在交织的视觉-文本-动作理解上。EO-1通过在EO-Data1.5M上的自回归解码和流匹配去噪之间的协同作用进行训练，从而实现无缝的机器人动作生成和多模态体态推理。大量实验证明了交织的视觉-文本-动作学习在开放世界理解和泛化方面的有效性，并通过多个体态下的多种长时延灵巧操作任务得到了验证。本论文详细介绍了EO-1的架构、EO-Data1.5M的数据构建策略以及训练方法，为开发高级体态基础模型提供了宝贵见解。 

---
# Automating the Deep Space Network Data Systems; A Case Study in Adaptive Anomaly Detection through Agentic AI 

**Title (ZH)**: 基于代理人工智能的自适应异常检测案研究：自动化深空网络数据系统 

**Authors**: Evan J. Chou, Lisa S. Locke, Harvey M. Soldan  

**Link**: [PDF](https://arxiv.org/pdf/2508.21111)  

**Abstract**: The Deep Space Network (DSN) is NASA's largest network of antenna facilities that generate a large volume of multivariate time-series data. These facilities contain DSN antennas and transmitters that undergo degradation over long periods of time, which may cause costly disruptions to the data flow and threaten the earth-connection of dozens of spacecraft that rely on the Deep Space Network for their lifeline. The purpose of this study was to experiment with different methods that would be able to assist JPL engineers with directly pinpointing anomalies and equipment degradation through collected data, and continue conducting maintenance and operations of the DSN for future space missions around our universe. As such, we have researched various machine learning techniques that can fully reconstruct data through predictive analysis, and determine anomalous data entries within real-time datasets through statistical computations and thresholds. On top of the fully trained and tested machine learning models, we have also integrated the use of a reinforcement learning subsystem that classifies identified anomalies based on severity level and a Large Language Model that labels an explanation for each anomalous data entry, all of which can be improved and fine-tuned over time through human feedback/input. Specifically, for the DSN transmitters, we have also implemented a full data pipeline system that connects the data extraction, parsing, and processing workflow all together as there was no coherent program or script for performing these tasks before. Using this data pipeline system, we were able to then also connect the models trained from DSN antenna data, completing the data workflow for DSN anomaly detection. This was all wrapped around and further connected by an agentic AI system, where complex reasoning was utilized to determine the classifications and predictions of anomalous data. 

**Abstract (ZH)**: 深空网络（DSN）是NASA最大的天线设施网络，生成大量多变量时间序列数据。这些设施中的DSN天线和发射机在长时间内会逐渐退化，可能导致数据流中断并威胁到依赖DSN的数十个航天器的地面连接。本研究旨在通过采用不同的方法，帮助JPL工程师直接定位异常和设备退化，并继续进行DSN的维护和操作，为未来围绕我们宇宙的空间任务提供保障。为此，我们研究了各种可用于完全重建数据并通过预测分析确定实时数据集中异常数据条目的机器学习技术，并结合使用基于严重程度分类识别出的异常的强化学习子系统和为每个异常数据条目提供解释的大型语言模型，这些模型可以通过人类反馈/输入进行改进和微调。特别地，对于DSN发射机，我们还实现了一个完整的数据管道系统，将数据提取、解析和处理工作流程整合在一起，此前没有相应的程序或脚本执行这些任务。使用此数据管道系统，我们能够连接从DSN天线数据训练的模型，完成DSN异常检测的数据工作流程。所有这些都由一个代理AI系统进一步整合，利用复杂推理来确定异常数据的分类和预测。 

---
# An Explainable, Attention-Enhanced, Bidirectional Long Short-Term Memory Neural Network for Joint 48-Hour Forecasting of Temperature, Irradiance, and Relative Humidity 

**Title (ZH)**: 具有解释性的注意力增强双向长短期记忆神经网络用于联合预报未来48小时的温度、辐照度和相对湿度 

**Authors**: Georgios Vamvouras, Konstantinos Braimakis, Christos Tzivanidis  

**Link**: [PDF](https://arxiv.org/pdf/2508.21109)  

**Abstract**: This paper presents a Deep Learning (DL) framework for 48-hour forecasting of temperature, solar irradiance, and relative humidity to support Model Predictive Control (MPC) in smart HVAC systems. The approach employs a stacked Bidirectional Long Short-Term Memory (BiLSTM) network with attention, capturing temporal and cross-feature dependencies by jointly predicting all three variables. Historical meteorological data (2019-2022) with encoded cyclical time features were used for training, while 2023 data evaluated generalization. The model achieved Mean Absolute Errors of 1.3 degrees Celsius (temperature), 31 W/m2 (irradiance), and 6.7 percentage points (humidity), outperforming state-of-the-art numerical weather prediction and machine learning benchmarks. Integrated Gradients quantified feature contributions, and attention weights revealed temporal patterns, enhancing interpretability. By combining multivariate forecasting, attention-based DL, and explainability, this work advances data-driven weather prediction. The demonstrated accuracy and transparency highlight the framework's potential for energy-efficient building control through reliable short-term meteorological forecasting. 

**Abstract (ZH)**: 基于深度学习的48小时温度、太阳辐照度和相对湿度预测框架：支持智能HVAC系统的模型预测控制 

---
# Learning to Generate Unit Test via Adversarial Reinforcement Learning 

**Title (ZH)**: 基于对抗强化学习的单元测试生成学习 

**Authors**: Dongjun Lee, Changho Hwang, Kimin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.21107)  

**Abstract**: Unit testing is a core practice in programming, enabling systematic evaluation of programs produced by human developers or large language models (LLMs). Given the challenges in writing comprehensive unit tests, LLMs have been employed to automate test generation, yet methods for training LLMs to produce high-quality tests remain underexplored. In this work, we propose UTRL, a novel reinforcement learning framework that trains an LLM to generate high-quality unit tests given a programming instruction. Our key idea is to iteratively train two LLMs, the unit test generator and the code generator, in an adversarial manner via reinforcement learning. The unit test generator is trained to maximize a discrimination reward, which reflects its ability to produce tests that expose faults in the code generator's solutions, and the code generator is trained to maximize a code reward, which reflects its ability to produce solutions that pass the unit tests generated by the test generator. In our experiments, we demonstrate that unit tests generated by Qwen3-4B trained via UTRL show higher quality compared to unit tests generated by the same model trained via supervised fine-tuning on human-written ground-truth unit tests, yielding code evaluations that more closely align with those induced by the ground-truth tests. Moreover, Qwen3-4B trained with UTRL outperforms frontier models such as GPT-4.1 in generating high-quality unit tests, highlighting the effectiveness of UTRL in training LLMs for this task. 

**Abstract (ZH)**: 基于强化学习的UTRL框架：训练大型语言模型生成高质量单元测试 

---
# Dynamic Low-rank Approximation of Full-Matrix Preconditioner for Training Generalized Linear Models 

**Title (ZH)**: 全矩阵预处理的动态低秩逼近在训练广义线性模型中的应用 

**Authors**: Tatyana Matveeva, Aleksandr Katrutsa, Evgeny Frolov  

**Link**: [PDF](https://arxiv.org/pdf/2508.21106)  

**Abstract**: Adaptive gradient methods like Adagrad and its variants are widespread in large-scale optimization. However, their use of diagonal preconditioning matrices limits the ability to capture parameter correlations. Full-matrix adaptive methods, approximating the exact Hessian, can model these correlations and may enable faster convergence. At the same time, their computational and memory costs are often prohibitive for large-scale models. To address this limitation, we propose AdaGram, an optimizer that enables efficient full-matrix adaptive gradient updates. To reduce memory and computational overhead, we utilize fast symmetric factorization for computing the preconditioned update direction at each iteration. Additionally, we maintain the low-rank structure of a preconditioner along the optimization trajectory using matrix integrator methods. Numerical experiments on standard machine learning tasks show that AdaGram converges faster or matches the performance of diagonal adaptive optimizers when using rank five and smaller rank approximations. This demonstrates AdaGram's potential as a scalable solution for adaptive optimization in large models. 

**Abstract (ZH)**: AdaGram: 一种高效的全矩阵自适应梯度优化器 

---
# PVPO: Pre-Estimated Value-Based Policy Optimization for Agentic Reasoning 

**Title (ZH)**: PVPO：基于预估价值的策略优化方法及其在代理推理中的应用 

**Authors**: Wenfeng Feng, Penghong Zhao, Guochao Jiang, Chuzhan Hao, Yuewei Zhang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.21104)  

**Abstract**: Critic-free reinforcement learning methods, particularly group policies, have attracted considerable attention for their efficiency in complex tasks. However, these methods rely heavily on multiple sampling and comparisons within the policy to estimate advantage, which may cause the policy to fall into local optimum and increase computational cost. To address these issues, we propose PVPO, an efficient reinforcement learning method enhanced by an advantage reference anchor and data pre-sampling. Specifically, we use the reference model to rollout in advance and employ the calculated reward score as a reference anchor. Our approach effectively corrects the cumulative bias introduced by intra-group comparisons and significantly reduces reliance on the number of rollouts. Meanwhile, the reference model can assess sample difficulty during data pre-sampling, enabling effective selection of high-gain data to improve training efficiency. Experiments conducted on nine datasets across two domains demonstrate that PVPO achieves State-Of-The-Art (SOTA) performance. Our approach not only demonstrates robust generalization across multiple tasks, but also exhibits scalable performance across models of varying scales. 

**Abstract (ZH)**: 无评判的强化学习方法，尤其是群体策略，由于在复杂任务中的高效性而受到广泛关注。然而，这些方法在策略内多次采样和比较以估计优势时依赖性很强，可能会导致策略陷入局部最优并增加计算成本。为解决这些问题，我们提出了一种名为PVPO的有效强化学习方法，该方法通过优势参考锚点和数据预采样进行增强。具体来说，我们利用参考模型提前展开并使用计算得到的奖励分数作为参考锚点。我们的方法有效纠正了组内比较引入的累积偏差，并显著减少了对展开次数的依赖。同时，在数据预采样过程中，参考模型可以评估样本难度，从而有效选择高收益数据以提高训练效率。我们在两个领域的九个数据集上进行的实验表明，PVPO实现了最先进的性能。我们的方法不仅在多个任务上展示了鲁棒的泛化能力，还在不同规模的模型上展示了可扩展的性能。 

---
# Spatiotemporal EEG-Based Emotion Recognition Using SAM Ratings from Serious Games with Hybrid Deep Learning 

**Title (ZH)**: 基于混合深度学习的时空EEG情绪识别：来自严肃游戏的SAM评分 

**Authors**: Abdul Rehman, Ilona Heldal, Jerry Chun-Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.21103)  

**Abstract**: Recent advancements in EEG-based emotion recognition have shown promising outcomes using both deep learning and classical machine learning approaches; however, most existing studies focus narrowly on binary valence prediction or subject-specific classification, which limits generalizability and deployment in real-world affective computing systems. To address this gap, this paper presents a unified, multigranularity EEG emotion classification framework built on the GAMEEMO dataset, which consists of 14-channel EEG recordings and continuous self-reported emotion ratings (boring, horrible, calm, and funny) from 28 subjects across four emotion-inducing gameplay scenarios. Our pipeline employs a structured preprocessing strategy that comprises temporal window segmentation, hybrid statistical and frequency-domain feature extraction, and z-score normalization to convert raw EEG signals into robust, discriminative input vectors. Emotion labels are derived and encoded across three complementary axes: (i) binary valence classification based on the averaged polarity of positive and negative emotion ratings, and (ii) Multi-class emotion classification, where the presence of the most affective state is predicted. (iii) Fine-grained multi-label representation via binning each emotion into 10 ordinal classes. We evaluate a broad spectrum of models, including Random Forest, XGBoost, and SVM, alongside deep neural architectures such as LSTM, LSTM-GRU, and CNN-LSTM. Among these, the LSTM-GRU model consistently outperforms the others, achieving an F1-score of 0.932 in the binary valence task and 94.5% and 90.6% in both multi-class and Multi-Label emotion classification. 

**Abstract (ZH)**: 基于EEG的情感识别近期取得了显著进展，使用深度学习和经典机器学习方法均显示出有希望的结果；然而，大多数现有研究集中在二元情感正 valence 预测或被试特异性分类上，这限制了其在真实世界情感计算系统中的泛化能力和部署。为解决这一差距，本文基于GAMEEMO数据集提出了一种统一的多粒度EEG情感分类框架，该数据集包含28名被试在四种游戏诱导情感场景中采集的14导联EEG记录和连续自我报告的情感评分（无聊、糟糕、平静、有趣）。我们的流水线采用了一种结构化的预处理策略，包括时间窗口分割、混合统计和频域特征提取以及z-score归一化，将原始EEG信号转化为稳健、有区别的输入向量。情感标签在三个互补轴上进行衍生和编码：(i) 基于正负情感评分平均极性的二元情感分类，(ii) 多类情感分类，预测最情感状态的存在，(iii) 细分多标签表示，将每种情感细分为10个序数类别。我们评估了包括随机森林、XGBoost、SVM以及深度神经架构如LSTM、LSTM-GRU和CNN-LSTM在内的多种模型。其中，LSTM-GRU模型在这三项任务中持续表现出最佳性能，在二元情感分类任务中F1分数为0.932，在多类和多标签情感分类中分别达到94.5%和90.6%。 

---
# Beyond Prediction: Reinforcement Learning as the Defining Leap in Healthcare AI 

**Title (ZH)**: 超越预测：强化学习在医疗人工智能领域的定义性突破 

**Authors**: Dilruk Perera, Gousia Habib, Qianyi Xu, Daniel J. Tan, Kai He, Erik Cambria, Mengling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2508.21101)  

**Abstract**: Reinforcement learning (RL) marks a fundamental shift in how artificial intelligence is applied in healthcare. Instead of merely predicting outcomes, RL actively decides interventions with long term goals. Unlike traditional models that operate on fixed associations, RL systems learn through trial, feedback, and long-term reward optimization, introducing transformative possibilities and new risks. From an information fusion lens, healthcare RL typically integrates multi-source signals such as vitals, labs clinical notes, imaging and device telemetry using temporal and decision-level mechanisms. These systems can operate within centralized, federated, or edge architectures to meet real-time clinical constraints, and naturally span data, features and decision fusion levels. This survey explore RL's rise in healthcare as more than a set of tools, rather a shift toward agentive intelligence in clinical environments. We first structure the landscape of RL techniques including model-based and model-free methods, offline and batch-constrained approaches, and emerging strategies for reward specification and uncertainty calibration through the lens of healthcare constraints. We then comprehensively analyze RL applications spanning critical care, chronic disease, mental health, diagnostics, and robotic assistance, identifying their trends, gaps, and translational bottlenecks. In contrast to prior reviews, we critically analyze RL's ethical, deployment, and reward design challenges, and synthesize lessons for safe, human-aligned policy learning. This paper serves as both a a technical roadmap and a critical reflection of RL's emerging transformative role in healthcare AI not as prediction machinery, but as agentive clinical intelligence. 

**Abstract (ZH)**: 强化学习（RL）在医疗健康领域的应用标志着人工智能应用方式的根本转变。不同于传统模型基于固定关联的工作方式，RL 系统通过试验、反馈和长期奖励优化来学习，引入了变革性的可能性和新的风险。从信息融合的视角来看，医疗健康的 RL 通常通过时间序列和决策层面的机制整合多源信号，如生命体征、实验室检查、临床记录、影像和设备遥测数据。这些系统可以在集中式、联邦式或边缘架构中运行，以满足临床实时约束，并自然跨越数据、特征和决策融合层次。本文探讨了 RL 在医疗健康中的崛起，不仅作为一种工具集，更是一种向临床环境中的自主智能转变。我们首先从医疗健康约束的角度结构化 RL 技术的景观，包括基于模型和无模型方法、离线和批量约束方法，以及用于奖励规范和不确定性校准的新兴策略。然后，我们全面分析了 RL 在重症监护、慢性病、心理健康、诊断和机器人辅助等领域中的应用，指出其趋势、差距和转化瓶颈。相对于先前的综述，我们深入分析了 RL 在伦理、部署和奖励设计方面的挑战，并综合了安全、以人为本政策学习的教训。本文既作为技术路线图，也作为对 RL 在医疗 AI 中新兴变革性角色的批判性反思，这种角色不仅是预测机器，更是自主临床智能。 

---
# Safe-Control: A Safety Patch for Mitigating Unsafe Content in Text-to-Image Generation Models 

**Title (ZH)**: Safe-Control: 一种缓解文本到图像生成模型中不安全内容的安全补丁 

**Authors**: Xiangtao Meng, Yingkai Dong, Ning Yu, Li Wang, Zheng Li, Shanqing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2508.21099)  

**Abstract**: Despite the advancements in Text-to-Image (T2I) generation models, their potential for misuse or even abuse raises serious safety concerns. Model developers have made tremendous efforts to introduce safety mechanisms that can address these concerns in T2I models. However, the existing safety mechanisms, whether external or internal, either remain susceptible to evasion under distribution shifts or require extensive model-specific adjustments. To address these limitations, we introduce Safe-Control, an innovative plug-and-play safety patch designed to mitigate unsafe content generation in T2I models. Using data-driven strategies and safety-aware conditions, Safe-Control injects safety control signals into the locked T2I model, acting as an update in a patch-like manner. Model developers can also construct various safety patches to meet the evolving safety requirements, which can be flexibly merged into a single, unified patch. Its plug-and-play design further ensures adaptability, making it compatible with other T2I models of similar denoising architecture. We conduct extensive evaluations on six diverse and public T2I models. Empirical results highlight that Safe-Control is effective in reducing unsafe content generation across six diverse T2I models with similar generative architectures, yet it successfully maintains the quality and text alignment of benign images. Compared to seven state-of-the-art safety mechanisms, including both external and internal defenses, Safe-Control significantly outperforms all baselines in reducing unsafe content generation. For example, it reduces the probability of unsafe content generation to 7%, compared to approximately 20% for most baseline methods, under both unsafe prompts and the latest adversarial attacks. 

**Abstract (ZH)**: 尽管文本到图像（T2I）生成模型取得了进展，但它们被滥用的可能性引发严重的安全担忧。模型开发者已付出巨大努力，引入了能够解决这些担忧的安全机制。然而，现有的安全机制，无论是外部的还是内部的，要么在分布变化下仍然容易被绕过，要么需要大量的模型特定调整。为解决这些局限性，我们引入了Safe-Control，这是一种创新的即插即用安全补丁，旨在减轻T2I模型中不安全内容的生成。通过数据驱动的策略和安全感知的条件，Safe-Control将安全控制信号注入锁定的T2I模型中，以补丁方式作为更新。模型开发者还可以构建各种安全补丁以适应不断变化的安全需求，并且可以灵活合并到一个统一的补丁中。其即插即用的设计进一步确保了其适应性，使它兼容具有类似去噪架构的其他T2I模型。我们在六个不同的公开T2I模型上进行了广泛评估。实证结果表明，Safe-Control在六种具有类似生成架构的T2I模型中均有效减少了不安全内容的生成，同时仍能够保持良性图像的质量和文本对齐。与包括外部和内部防御在内的七种最先进的安全机制相比，Safe-Control在减少不安全内容生成方面显著优于所有基线。例如，在不安全提示和最新的对抗攻击下，它将不安全内容生成的概率降低到7%，而大多数基线方法这一概率约为20%。 

---
# TrInk: Ink Generation with Transformer Network 

**Title (ZH)**: TrInk: 基于变压器网络的墨水生成 

**Authors**: Zezhong Jin, Shubhang Desai, Xu Chen, Biyi Fang, Zhuoyi Huang, Zhe Li, Chong-Xin Gan, Xiao Tu, Man-Wai Mak, Yan Lu, Shujie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21098)  

**Abstract**: In this paper, we propose TrInk, a Transformer-based model for ink generation, which effectively captures global dependencies. To better facilitate the alignment between the input text and generated stroke points, we introduce scaled positional embeddings and a Gaussian memory mask in the cross-attention module. Additionally, we design both subjective and objective evaluation pipelines to comprehensively assess the legibility and style consistency of the generated handwriting. Experiments demonstrate that our Transformer-based model achieves a 35.56\% reduction in character error rate (CER) and an 29.66% reduction in word error rate (WER) on the IAM-OnDB dataset compared to previous methods. We provide an demo page with handwriting samples from TrInk and baseline models at: this https URL 

**Abstract (ZH)**: 本文提出了基于Transformer的TrInk模型，有效捕捉全局依赖关系。为了更好地使输入文本与生成的笔画点对齐，在交叉注意力模块中引入了缩放位置嵌入和高斯记忆遮罩。此外，我们设计了主观和客观评估管道，全面评估生成手写体的可读性和风格一致性。实验结果表明，与先前方法相比，我们的Transformer基模型在IAM-OnDB数据集上的字符错误率（CER）降低了35.56%，单词错误率（WER）降低了29.66%。我们提供了一个演示页面，包含TrInk和基线模型的手写样本：this https URL。 

---
# Model-Driven Quantum Code Generation Using Large Language Models and Retrieval-Augmented Generation 

**Title (ZH)**: 基于大型语言模型和检索增强生成的模型驱动量子代码生成 

**Authors**: Nazanin Siavash, Armin Moin  

**Link**: [PDF](https://arxiv.org/pdf/2508.21097)  

**Abstract**: This paper introduces a novel research direction for model-to-text/code transformations by leveraging Large Language Models (LLMs) that can be enhanced with Retrieval-Augmented Generation (RAG) pipelines. The focus is on quantum and hybrid quantum-classical software systems, where model-driven approaches can help reduce the costs and mitigate the risks associated with the heterogeneous platform landscape and lack of developers' skills. We validate one of the proposed ideas regarding generating code out of UML model instances of software systems. This Python code uses a well-established library, called Qiskit, to execute on gate-based or circuit-based quantum computers. The RAG pipeline that we deploy incorporates sample Qiskit code from public GitHub repositories. Experimental results show that well-engineered prompts can improve CodeBLEU scores by up to a factor of four, yielding more accurate and consistent quantum code. However, the proposed research direction can go beyond this through further investigation in the future by conducting experiments to address our other research questions and ideas proposed here, such as deploying software system model instances as the source of information in the RAG pipelines, or deploying LLMs for code-to-code transformations, for instance, for transpilation use cases. 

**Abstract (ZH)**: 本文通过利用增强检索增强生成（RAG）管道的大语言模型（LLMs），提出了模型到文本/代码转换的一种新颖研究方向，重点关注量子和混合量子- classical软件系统，其中模型驱动的方法可以帮助减少异构平台landscape带来的成本和风险，以及缺乏开发者技能的问题。我们验证了将软件系统的UML模型实例生成代码的其中一个提案。该Python代码使用了名为Qiskit的成熟库，在基于门的或量子电路的量子计算机上执行。我们部署的RAG管道包含来自公共GitHub仓库的Qiskit代码样例。实验结果显示，精心设计的提示可以将CodeBLEU得分提高多达四倍，从而生成更准确和一致的量子代码。然而，通过在未来进行实验来进一步研究我们的其他研究问题和提出的其他想法，该研究方向可以取得更进一步的成果，例如将软件系统模型实例用作RAG管道的信息源，或部署LLMs进行代码到代码的转换，例如移植使用案例。 

---
# CoBA: Counterbias Text Augmentation for Mitigating Various Spurious Correlations via Semantic Triples 

**Title (ZH)**: CoBA: 反偏见文本增强以通过语义三元组减轻各种虚假相关性 

**Authors**: Kyohoon Jin, Juhwan Choi, Jungmin Yun, Junho Lee, Soojin Jang, Youngbin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.21083)  

**Abstract**: Deep learning models often learn and exploit spurious correlations in training data, using these non-target features to inform their predictions. Such reliance leads to performance degradation and poor generalization on unseen data. To address these limitations, we introduce a more general form of counterfactual data augmentation, termed counterbias data augmentation, which simultaneously tackles multiple biases (e.g., gender bias, simplicity bias) and enhances out-of-distribution robustness. We present CoBA: CounterBias Augmentation, a unified framework that operates at the semantic triple level: first decomposing text into subject-predicate-object triples, then selectively modifying these triples to disrupt spurious correlations. By reconstructing the text from these adjusted triples, CoBA generates counterbias data that mitigates spurious patterns. Through extensive experiments, we demonstrate that CoBA not only improves downstream task performance, but also effectively reduces biases and strengthens out-of-distribution resilience, offering a versatile and robust solution to the challenges posed by spurious correlations. 

**Abstract (ZH)**: 深度学习模型往往会在训练数据中学习和利用虚假相关性，并使用这些非目标特征来指导其预测。这种依赖导致在未见数据上的性能下降和泛化能力差。为解决这些问题，我们引入了一种更通用形式的反事实数据增强方法，称为反偏差数据增强，它可以同时解决多种偏差（如性别偏差、简单性偏差），并提高分布外鲁棒性。我们提出了一种统一体义三元组级别的反偏差增强框架CoBA：首先将文本分解为主题-谓词-宾语三元组，然后选择性地修改这些三元组以破坏虚假相关性。通过从这些调整的三元组重构文本，CoBA生成反偏差数据，以减轻虚假模式。通过广泛的实验，我们证明CoBA不仅能提高下游任务性能，还能有效减少偏差并增强分布外鲁棒性，提供了一种灵活且稳健的解决方案来应对虚假相关带来的挑战。 

---
# Pep2Prob Benchmark: Predicting Fragment Ion Probability for MS$^2$-based Proteomics 

**Title (ZH)**: Pep2Prob 基准：基于 MS$^2$ 的蛋白质组学中肽片段离子概率预测 

**Authors**: Hao Xu, Zhichao Wang, Shengqi Sang, Pisit Wajanasara, Nuno Bandeira  

**Link**: [PDF](https://arxiv.org/pdf/2508.21076)  

**Abstract**: Proteins perform nearly all cellular functions and constitute most drug targets, making their analysis fundamental to understanding human biology in health and disease. Tandem mass spectrometry (MS$^2$) is the major analytical technique in proteomics that identifies peptides by ionizing them, fragmenting them, and using the resulting mass spectra to identify and quantify proteins in biological samples. In MS$^2$ analysis, peptide fragment ion probability prediction plays a critical role, enhancing the accuracy of peptide identification from mass spectra as a complement to the intensity information. Current approaches rely on global statistics of fragmentation, which assumes that a fragment's probability is uniform across all peptides. Nevertheless, this assumption is oversimplified from a biochemical principle point of view and limits accurate prediction. To address this gap, we present Pep2Prob, the first comprehensive dataset and benchmark designed for peptide-specific fragment ion probability prediction. The proposed dataset contains fragment ion probability statistics for 608,780 unique precursors (each precursor is a pair of peptide sequence and charge state), summarized from more than 183 million high-quality, high-resolution, HCD MS$^2$ spectra with validated peptide assignments and fragmentation annotations. We establish baseline performance using simple statistical rules and learning-based methods, and find that models leveraging peptide-specific information significantly outperform previous methods using only global fragmentation statistics. Furthermore, performance across benchmark models with increasing capacities suggests that the peptide-fragmentation relationship exhibits complex nonlinearities requiring sophisticated machine learning approaches. 

**Abstract (ZH)**: 蛋白质几乎执行所有细胞功能，并构成大多数药物靶标，因此对其分析对于理解健康和疾病中的人类生物学至关重要。串联质谱（MS$^2$）是蛋白质组学中主要的分析技术，通过电离、裂解肽并使用生成的质谱来识别和定量生物样品中的蛋白质。在MS$^2$分析中，肽碎片离子概率预测起到关键作用，作为强度信息的补充，增强从质谱中识别肽的准确性。当前的方法依赖于裂解的全局统计信息，假设一个碎片的概率在所有肽中均匀分布。然而，从生物化学原理的角度来看，这一假设过于简化，限制了准确预测的能力。为解决这一问题，我们提出了Pep2Prob，这是首个用于肽特异性碎片离子概率预测的综合数据集和基准。该数据集包含来自超过1.83亿个高质量、高分辨率HCD MS$^2$谱图的608,780个独特前体（每个前体是一对肽序列和电荷状态）的碎片离子概率统计信息，这些谱图已验证了肽的分配和裂解注解。我们使用简单的统计规则和基于学习的方法建立了基线性能，并发现利用肽特异性信息的模型显著优于仅使用全局裂解统计数据的先前方法。此外，基准模型性能随容量增加的趋势表明，肽-裂解关系表现出复杂的非线性关系，需要 sophisticated 的机器学习方法。 

---
# QuadKAN: KAN-Enhanced Quadruped Motion Control via End-to-End Reinforcement Learning 

**Title (ZH)**: QuadKAN: 通过端到端强化学习增强的四足运动控制 

**Authors**: Allen Wang, Gavin Tao  

**Link**: [PDF](https://arxiv.org/pdf/2508.19153)  

**Abstract**: We address vision-guided quadruped motion control with reinforcement learning (RL) and highlight the necessity of combining proprioception with vision for robust control. We propose QuadKAN, a spline-parameterized cross-modal policy instantiated with Kolmogorov-Arnold Networks (KANs). The framework incorporates a spline encoder for proprioception and a spline fusion head for proprioception-vision inputs. This structured function class aligns the state-to-action mapping with the piecewise-smooth nature of gait, improving sample efficiency, reducing action jitter and energy consumption, and providing interpretable posture-action sensitivities. We adopt Multi-Modal Delay Randomization (MMDR) and perform end-to-end training with Proximal Policy Optimization (PPO). Evaluations across diverse terrains, including both even and uneven surfaces and scenarios with static or dynamic obstacles, demonstrate that QuadKAN achieves consistently higher returns, greater distances, and fewer collisions than state-of-the-art (SOTA) baselines. These results show that spline-parameterized policies offer a simple, effective, and interpretable alternative for robust vision-guided locomotion. A repository will be made available upon acceptance. 

**Abstract (ZH)**: 基于视觉的四足运动控制：结合知觉与强化学习的方法及其实现 

---
