# Dynamic Initialization for LiDAR-inertial SLAM 

**Title (ZH)**: LiDAR-惯性SLAM的动态初始化方法 

**Authors**: Jie Xu, Yongxin Ma, Yixuan Li, Xuanxuan Zhang, Jun Zhou, Shenghai Yuan, Lihua Xie  

**Link**: [PDF](https://arxiv.org/pdf/2504.01451)  

**Abstract**: The accuracy of the initial state, including initial velocity, gravity direction, and IMU biases, is critical for the initialization of LiDAR-inertial SLAM systems. Inaccurate initial values can reduce initialization speed or lead to failure. When the system faces urgent tasks, robust and fast initialization is required while the robot is moving, such as during the swift assessment of rescue environments after natural disasters, bomb disposal, and restarting LiDAR-inertial SLAM in rescue missions. However, existing initialization methods usually require the platform to remain stationary, which is ineffective when the robot is in motion. To address this issue, this paper introduces a robust and fast dynamic initialization method for LiDAR-inertial systems (D-LI-Init). This method iteratively aligns LiDAR-based odometry with IMU measurements to achieve system initialization. To enhance the reliability of the LiDAR odometry module, the LiDAR and gyroscope are tightly integrated within the ESIKF framework. The gyroscope compensates for rotational distortion in the point cloud. Translational distortion compensation occurs during the iterative update phase, resulting in the output of LiDAR-gyroscope odometry. The proposed method can initialize the system no matter the robot is moving or stationary. Experiments on public datasets and real-world environments demonstrate that the D-LI-Init algorithm can effectively serve various platforms, including vehicles, handheld devices, and UAVs. D-LI-Init completes dynamic initialization regardless of specific motion patterns. To benefit the research community, we have open-sourced our code and test datasets on GitHub. 

**Abstract (ZH)**: 基于激光雷达-惯性SLAM系统的初始状态准确性对于系统初始化至关重要，包括初始速度、重力方向和IMU偏差。不准确的初始值会降低初始化速度或导致初始化失败。当系统面临紧急任务时，特别是在机器人移动过程中，需要快速且鲁棒的初始化方法，如自然灾难后的救援环境快速评估、排爆任务以及救援行动中激光雷达-惯性SLAM的重启。然而，现有的初始化方法通常要求平台保持静止，这在机器人移动时是无效的。为解决这一问题，本文提出了一种基于激光雷达-惯性系统的鲁棒快速动态初始化方法（D-LI-Init）。该方法通过迭代对准激光雷达里程计与IMU测量值来实现系统初始化。为了增强激光雷达里程计模块的可靠性，激光雷达与陀螺仪在ESIKF框架下紧密集成，陀螺仪补偿点云中的旋转失真，平移失真补偿发生在迭代更新阶段，从而得到激光雷达-陀螺仪里程计输出。所提出的方法可以在机器人移动或静止时实现系统初始化。在公共数据集和实际环境上的实验表明，D-LI-Init算法可以有效服务于包括车辆、手持设备和无人机在内的各种平台，并且可以根据特定运动模式的差异完成动态初始化。为促进科研社区的发展，我们已在GitHub上开源了我们的代码和测试数据集。 

---
# Overcoming Deceptiveness in Fitness Optimization with Unsupervised Quality-Diversity 

**Title (ZH)**: 克服适应度优化中的误导性问题以实现无监督的质量多样性 

**Authors**: Lisa Coiffard, Paul Templier, Antoine Cully  

**Link**: [PDF](https://arxiv.org/pdf/2504.01915)  

**Abstract**: Policy optimization seeks the best solution to a control problem according to an objective or fitness function, serving as a fundamental field of engineering and research with applications in robotics. Traditional optimization methods like reinforcement learning and evolutionary algorithms struggle with deceptive fitness landscapes, where following immediate improvements leads to suboptimal solutions. Quality-diversity (QD) algorithms offer a promising approach by maintaining diverse intermediate solutions as stepping stones for escaping local optima. However, QD algorithms require domain expertise to define hand-crafted features, limiting their applicability where characterizing solution diversity remains unclear. In this paper, we show that unsupervised QD algorithms - specifically the AURORA framework, which learns features from sensory data - efficiently solve deceptive optimization problems without domain expertise. By enhancing AURORA with contrastive learning and periodic extinction events, we propose AURORA-XCon, which outperforms all traditional optimization baselines and matches, in some cases even improving by up to 34%, the best QD baseline with domain-specific hand-crafted features. This work establishes a novel application of unsupervised QD algorithms, shifting their focus from discovering novel solutions toward traditional optimization and expanding their potential to domains where defining feature spaces poses challenges. 

**Abstract (ZH)**: 无监督质量多样性算法AURORA-XCon在欺骗性优化问题中的应用 

---
# CoRAG: Collaborative Retrieval-Augmented Generation 

**Title (ZH)**: CoRAG：协作检索增强生成 

**Authors**: Aashiq Muhamed, Mona Diab, Virginia Smith  

**Link**: [PDF](https://arxiv.org/pdf/2504.01883)  

**Abstract**: Retrieval-Augmented Generation (RAG) models excel in knowledge-intensive tasks, especially under few-shot learning constraints. We introduce CoRAG, a framework extending RAG to collaborative settings, where clients jointly train a shared model using a collaborative passage store. To evaluate CoRAG, we introduce CRAB, a benchmark for collaborative homogeneous open-domain question answering. Our experiments demonstrate that CoRAG consistently outperforms both parametric collaborative learning methods and locally trained RAG models in low-resource scenarios. Further analysis reveals the critical importance of relevant passages within the shared store, the surprising benefits of incorporating irrelevant passages, and the potential for hard negatives to negatively impact performance. This introduces a novel consideration in collaborative RAG: the trade-off between leveraging a collectively enriched knowledge base and the potential risk of incorporating detrimental passages from other clients. Our findings underscore the viability of CoRAG, while also highlighting key design challenges and promising avenues for future research. 

**Abstract (ZH)**: CoRAG：扩展RAG到协作场景的框架 

---
# An Approach to Technical AGI Safety and Security 

**Title (ZH)**: 技术AGI安全与安全措施方法 

**Authors**: Rohin Shah, Alex Irpan, Alexander Matt Turner, Anna Wang, Arthur Conmy, David Lindner, Jonah Brown-Cohen, Lewis Ho, Neel Nanda, Raluca Ada Popa, Rishub Jain, Rory Greig, Samuel Albanie, Scott Emmons, Sebastian Farquhar, Sébastien Krier, Senthooran Rajamanoharan, Sophie Bridgers, Tobi Ijitoye, Tom Everitt, Victoria Krakovna, Vikrant Varma, Vladimir Mikulik, Zachary Kenton, Dave Orr, Shane Legg, Noah Goodman, Allan Dafoe, Four Flynn, Anca Dragan  

**Link**: [PDF](https://arxiv.org/pdf/2504.01849)  

**Abstract**: Artificial General Intelligence (AGI) promises transformative benefits but also presents significant risks. We develop an approach to address the risk of harms consequential enough to significantly harm humanity. We identify four areas of risk: misuse, misalignment, mistakes, and structural risks. Of these, we focus on technical approaches to misuse and misalignment. For misuse, our strategy aims to prevent threat actors from accessing dangerous capabilities, by proactively identifying dangerous capabilities, and implementing robust security, access restrictions, monitoring, and model safety mitigations. To address misalignment, we outline two lines of defense. First, model-level mitigations such as amplified oversight and robust training can help to build an aligned model. Second, system-level security measures such as monitoring and access control can mitigate harm even if the model is misaligned. Techniques from interpretability, uncertainty estimation, and safer design patterns can enhance the effectiveness of these mitigations. Finally, we briefly outline how these ingredients could be combined to produce safety cases for AGI systems. 

**Abstract (ZH)**: 人工通用智能（AGI）承诺带来变革性的好处，但也带来了重大风险。我们提出了一种方法来应对可能严重危害人类的潜在危害风险。我们识别了四个风险领域：滥用、错配、错误和结构风险。在这四个领域中，我们重点关注技术手段以应对滥用和错配风险。对于滥用风险，我们的策略旨在通过主动识别危险能力、实施稳健的安全措施、访问限制、监控和模型安全性缓解措施，防止威胁行为者获取危险能力。为了应对错配风险，我们概述了两条防线。首先，模型层面的缓解措施，如增强监督和稳健训练，有助于构建对齐的模型。其次，系统层面的安全措施，如监控和访问控制，即使在模型错配的情况下也能减轻损害。来自可解释性、不确定估计和更安全的设计模式的技术可以增强这些缓解措施的有效性。最后，我们简要概述了这些要素如何结合以生成AGI系统的安全性案例。 

---
# PaperBench: Evaluating AI's Ability to Replicate AI Research 

**Title (ZH)**: PaperBench: 评估AI复制AI研究的能力 

**Authors**: Giulio Starace, Oliver Jaffe, Dane Sherburn, James Aung, Jun Shern Chan, Leon Maksin, Rachel Dias, Evan Mays, Benjamin Kinsella, Wyatt Thompson, Johannes Heidecke, Amelia Glaese, Tejal Patwardhan  

**Link**: [PDF](https://arxiv.org/pdf/2504.01848)  

**Abstract**: We introduce PaperBench, a benchmark evaluating the ability of AI agents to replicate state-of-the-art AI research. Agents must replicate 20 ICML 2024 Spotlight and Oral papers from scratch, including understanding paper contributions, developing a codebase, and successfully executing experiments. For objective evaluation, we develop rubrics that hierarchically decompose each replication task into smaller sub-tasks with clear grading criteria. In total, PaperBench contains 8,316 individually gradable tasks. Rubrics are co-developed with the author(s) of each ICML paper for accuracy and realism. To enable scalable evaluation, we also develop an LLM-based judge to automatically grade replication attempts against rubrics, and assess our judge's performance by creating a separate benchmark for judges. We evaluate several frontier models on PaperBench, finding that the best-performing tested agent, Claude 3.5 Sonnet (New) with open-source scaffolding, achieves an average replication score of 21.0\%. Finally, we recruit top ML PhDs to attempt a subset of PaperBench, finding that models do not yet outperform the human baseline. We \href{this https URL}{open-source our code} to facilitate future research in understanding the AI engineering capabilities of AI agents. 

**Abstract (ZH)**: 我们介绍PaperBench：一个评估AI代理再现顶级AI研究成果能力的基准测试。 

---
# A Novel Approach To Implementing Knowledge Distillation In Tsetlin Machines 

**Title (ZH)**: 一种在Tsetlin机器中实现知识蒸馏的新方法 

**Authors**: Calvin Kinateder  

**Link**: [PDF](https://arxiv.org/pdf/2504.01798)  

**Abstract**: The Tsetlin Machine (TM) is a propositional logic based model that uses conjunctive clauses to learn patterns from data. As with typical neural networks, the performance of a Tsetlin Machine is largely dependent on its parameter count, with a larger number of parameters producing higher accuracy but slower execution. Knowledge distillation in neural networks transfers information from an already-trained teacher model to a smaller student model to increase accuracy in the student without increasing execution time. We propose a novel approach to implementing knowledge distillation in Tsetlin Machines by utilizing the probability distributions of each output sample in the teacher to provide additional context to the student. Additionally, we propose a novel clause-transfer algorithm that weighs the importance of each clause in the teacher and initializes the student with only the most essential data. We find that our algorithm can significantly improve performance in the student model without negatively impacting latency in the tested domains of image recognition and text classification. 

**Abstract (ZH)**: 基于命题逻辑的特林机（Tsetlin Machine）通过使用合取_clause_学习数据模式。特林机的性能主要取决于其参数数量，更多参数可提高准确性但会减慢执行速度。神经网络中的知识蒸馏将已训练的教师模型的信息传递给较小的学生模型，以在不增加执行时间的情况下提高学生模型的准确性。我们提出了一种在特林机中实施知识蒸馏的新型方法，通过利用教师每个输出样本的概率分布为学生提供额外上下文。此外，我们还提出了一种新型的_clause_转移算法，根据教师中每个_clause_的重要性对学生进行初始化，仅使用最关键的数据。我们发现该算法能够在不负面影响测试领域（图像识别和文本分类）中延迟的情况下显著提高学生模型的性能。 

---
# Enhancing Interpretability in Generative AI Through Search-Based Data Influence Analysis 

**Title (ZH)**: 基于搜索的数据影响分析以提高生成AI的可解释性 

**Authors**: Theodoros Aivalis, Iraklis A. Klampanos, Antonis Troumpoukis, Joemon M. Jose  

**Link**: [PDF](https://arxiv.org/pdf/2504.01771)  

**Abstract**: Generative AI models offer powerful capabilities but often lack transparency, making it difficult to interpret their output. This is critical in cases involving artistic or copyrighted content. This work introduces a search-inspired approach to improve the interpretability of these models by analysing the influence of training data on their outputs. Our method provides observational interpretability by focusing on a model's output rather than on its internal state. We consider both raw data and latent-space embeddings when searching for the influence of data items in generated content. We evaluate our method by retraining models locally and by demonstrating the method's ability to uncover influential subsets in the training data. This work lays the groundwork for future extensions, including user-based evaluations with domain experts, which is expected to improve observational interpretability further. 

**Abstract (ZH)**: 生成式AI模型提供了强大的功能，但往往缺乏透明度，使其输出难以解释。这在涉及艺术或受版权保护内容的情况下尤其关键。本文介绍了一种基于搜索的方法，通过分析训练数据对模型输出的影响来提高模型的可解释性。我们的方法通过关注模型的输出而非其内部状态，提供了一种观察性的可解释性。我们在生成内容中考虑了原始数据和潜在空间嵌入来搜索数据项的影响。我们通过局部重新训练模型对方法进行了评估，并展示了该方法在揭示训练数据中具有影响力的子集方面的能力。这项工作为未来的扩展奠定了基础，包括与领域专家进行基于用户的研究评估，预期将进一步提高观察性的可解释性。 

---
# Epistemic Skills: Reasoning about Knowledge and Oblivion 

**Title (ZH)**: 知识与遗忘的推理技能 

**Authors**: Xiaolong Liang, Yì N. Wáng  

**Link**: [PDF](https://arxiv.org/pdf/2504.01733)  

**Abstract**: This paper presents a class of epistemic logics that captures the dynamics of acquiring knowledge and descending into oblivion, while incorporating concepts of group knowledge. The approach is grounded in a system of weighted models, introducing an ``epistemic skills'' metric to represent the epistemic capacities tied to knowledge updates. Within this framework, knowledge acquisition is modeled as a process of upskilling, whereas oblivion is represented as a consequence of downskilling. The framework further enables exploration of ``knowability'' and ``forgettability,'' defined as the potential to gain knowledge through upskilling and to lapse into oblivion through downskilling, respectively. Additionally, it supports a detailed analysis of the distinctions between epistemic de re and de dicto expressions. The computational complexity of the model checking and satisfiability problems is examined, offering insights into their theoretical foundations and practical implications. 

**Abstract (ZH)**: 本文提出了一类知识逻辑，capture了知识获取和知识遗忘动态过程，并融入了群体知识的概念。该方法基于加权模型系统，引入了“知识技能”度量来表示与知识更新相关的认识能力。在此框架下，知识获取被建模为技能提升的过程，而遗忘则被表示为技能下降的后果。该框架还允许探索“可得知性”和“易遗忘性”，分别定义为通过技能提升获得知识和通过技能下降陷入遗忘的可能性。此外，该框架还支持对epistemic de re和de dicto表达式的区别进行详细分析。研究了模型检查和可满足性问题的计算复杂性，提供了理论基础和实际应用的见解。 

---
# Identifying Macro Causal Effects in C-DMGs 

**Title (ZH)**: 识别C-DMGs中的宏观因果效应 

**Authors**: Simon Ferreira, Charles K. Assaad  

**Link**: [PDF](https://arxiv.org/pdf/2504.01551)  

**Abstract**: Causal effect identification using causal graphs is a fundamental challenge in causal inference. While extensive research has been conducted in this area, most existing methods assume the availability of fully specified causal graphs. However, in complex domains such as medicine and epidemiology, complete causal knowledge is often unavailable, and only partial information about the system is accessible. This paper focuses on causal effect identification within partially specified causal graphs, with particular emphasis on cluster-directed mixed graphs (C-DMGs). These graphs provide a higher-level representation of causal relationships by grouping variables into clusters, offering a more practical approach for handling complex systems. Unlike fully specified causal graphs, C-DMGs can contain cycles, which complicate their analysis and interpretation. Furthermore, their cluster-based nature introduces new challenges, as it gives rise to two distinct types of causal effects, macro causal effects and micro causal effects, with different properties. In this work, we focus on macro causal effects, which describe the effects of entire clusters on other clusters. We establish that the do-calculus is both sound and complete for identifying these effects in C-DMGs. Additionally, we provide a graphical characterization of non-identifiability for macro causal effects in these graphs. 

**Abstract (ZH)**: 使用因果图识别因果效应是因果推断中的基础挑战。尽管在此领域进行了广泛的研究，但大多数现有方法假定可以获取完全指定的因果图。然而，在医学和流行病学等复杂领域中，完整的因果知识往往不可用，只能访问系统的部分信息。本文专注于部分指定的因果图中的因果效应识别，特别着重于聚类导向混合图（C-DMGs）。这些图通过将变量分组为聚类，提供了更高层次的因果关系表示，为处理复杂系统提供了更实用的方法。与完全指定的因果图不同，C-DMGs 可包含循环，这使得它们的分析和解释更加复杂。此外，基于聚类的性质引入了新的挑战，产生了两种不同类型的原因效应：宏观因果效应和微观因果效应，它们具有不同的属性。在这项工作中，我们重点关注宏观因果效应，这些效应描述了整个聚类对其他聚类的影响。我们建立了 do- calculus 在 C-DMGs 中识别这些效应的完备性。此外，我们还提供了宏因果效应在这些图中非识别性的图形特征。 

---
# AI-Newton: A Concept-Driven Physical Law Discovery System without Prior Physical Knowledge 

**Title (ZH)**: AI-Newton: 一种无需先验物理知识的基于概念的物理定律发现系统 

**Authors**: You-Le Fang, Dong-Shan Jian, Xiang Li, Yan-Qing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.01538)  

**Abstract**: Current limitations in human scientific discovery necessitate a new research paradigm. While advances in artificial intelligence (AI) offer a highly promising solution, enabling AI to emulate human-like scientific discovery remains an open challenge. To address this, we propose AI-Newton, a concept-driven discovery system capable of autonomously deriving physical laws from raw data -- without supervision or prior physical knowledge. The system integrates a knowledge base and knowledge representation centered on physical concepts, along with an autonomous discovery workflow. As a proof of concept, we apply AI-Newton to a large set of Newtonian mechanics problems. Given experimental data with noise, the system successfully rediscovers fundamental laws, including Newton's second law, energy conservation and law of gravitation, using autonomously defined concepts. This achievement marks a significant step toward AI-driven autonomous scientific discovery. 

**Abstract (ZH)**: 当前人类科学研究的限制 necessitate 一种新的研究范式。尽管人工智能（AI）的进步提供了高度有希望的解决方案，但使AI能够模仿人类的科学研究仍是一项开放的挑战。为了解决这一问题，我们提出AI-Newton，一种基于概念的发现系统，能够在没有监督或先验物理知识的情况下自主从原始数据中推导出物理定律。该系统结合了以物理概念为中心的知识库和知识表示，以及一个自主发现的工作流程。作为概念验证，我们将AI-Newton 应用于大量的牛顿力学问题。给定包含噪声的实验数据，系统成功地重新发现了基本定律，包括牛顿第二定律、能量守恒定律和万有引力定律，使用了自主定义的概念。这一成就标志着向AI驱动的自主科学研究迈出了一大步。 

---
# An Explainable Reconfiguration-Based Optimization Algorithm for Industrial and Reliability-Redundancy Allocation Problems 

**Title (ZH)**: 可解释的基于重组的优化算法及其在工业可靠性冗余分配问题中的应用 

**Authors**: Dikshit Chauhan, Nitin Gupta, Anupam Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2504.01331)  

**Abstract**: Industrial and reliability optimization problems often involve complex constraints and require efficient, interpretable solutions. This paper presents AI-AEFA, an advanced parameter reconfiguration-based metaheuristic algorithm designed to address large-scale industrial and reliability-redundancy allocation problems. AI-AEFA enhances search space exploration and convergence efficiency through a novel log-sigmoid-based parameter adaptation and chaotic mapping mechanism. The algorithm is validated across twenty-eight IEEE CEC 2017 constrained benchmark problems, fifteen large-scale industrial optimization problems, and seven reliability-redundancy allocation problems, consistently outperforming state-of-the-art optimization techniques in terms of feasibility, computational efficiency, and convergence speed. The additional key contribution of this work is the integration of SHAP (Shapley Additive Explanations) to enhance the interpretability of AI-AEFA, providing insights into the impact of key parameters such as Coulomb's constant, charge, acceleration, and electrostatic force. This explainability feature enables a deeper understanding of decision-making within the AI-AEFA framework during the optimization processes. The findings confirm AI-AEFA as a robust, scalable, and interpretable optimization tool with significant real-world applications. 

**Abstract (ZH)**: 工业和可靠性优化问题通常包含复杂约束，需要高效的可解释解决方案。本文介绍了一种先进的参数重构元启发式算法AI-AEFA，用于解决大规模工业和可靠性冗余分配问题。AI-AEFA通过一种新颖的基于对数sigmoid的参数自适应和混沌映射机制来增强搜索空间探索和收敛效率。该算法在28个IEEE CEC 2017约束测试问题、15个大规模工业优化问题以及7个可靠性冗余分配问题上进行了验证，始终在可实现性、计算效率和收敛速度方面优于现有的优化技术。本文的另一项关键贡献是将SHAP（Shapley值解释）集成到AI-AEFA中，以增强其可解释性，提供对关键参数如库仑常数、电荷、加速度和静电作用力的影响洞察。这种可解释性功能使得在优化过程中的决策制定具有更深层次的理解。研究结果证实，AI-AEFA是一种稳健、可扩展并具有很强可解释性的优化工具，具备显著的实际应用价值。 

---
# Off-Policy Evaluation for Sequential Persuasion Process with Unobserved Confounding 

**Title (ZH)**: 未观察到混杂因素下的序贯说服过程离策策略评估 

**Authors**: Nishanth Venkatesh S., Heeseung Bang, Andreas A. Malikopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2504.01211)  

**Abstract**: In this paper, we expand the Bayesian persuasion framework to account for unobserved confounding variables in sender-receiver interactions. While traditional models assume that belief updates follow Bayesian principles, real-world scenarios often involve hidden variables that impact the receiver's belief formation and decision-making. We conceptualize this as a sequential decision-making problem, where the sender and receiver interact over multiple rounds. In each round, the sender communicates with the receiver, who also interacts with the environment. Crucially, the receiver's belief update is affected by an unobserved confounding variable. By reformulating this scenario as a Partially Observable Markov Decision Process (POMDP), we capture the sender's incomplete information regarding both the dynamics of the receiver's beliefs and the unobserved confounder. We prove that finding an optimal observation-based policy in this POMDP is equivalent to solving for an optimal signaling strategy in the original persuasion framework. Furthermore, we demonstrate how this reformulation facilitates the application of proximal learning for off-policy evaluation in the persuasion process. This advancement enables the sender to evaluate alternative signaling strategies using only observational data from a behavioral policy, thus eliminating the necessity for costly new experiments. 

**Abstract (ZH)**: 本文扩展了Bayesian说服框架，以考虑发送者-接收者交互中的未观察到的混淆变量。传统模型假设信念更新遵循Bayesian原则，但现实世界的情景通常涉及影响接收者信念形成和决策的隐藏变量。我们将此概念化为一个序列决策问题，其中发送者和接收者在多轮交互中进行互动。在每一轮中，发送者与接收者沟通，接收者也与环境互动。关键的是，接收者的信念更新受到未观察到的混淆变量的影响。通过将此场景重新表述为部分可观测马尔可夫决策过程（POMDP），我们捕捉到了发送者关于接收者信念动态及其未观察到的混杂因素的不完整信息。我们证明，在这个POMDP中寻找最优观测策略等同于在原始说服框架中寻找最优信号策略。此外，我们展示了这种重新表述如何促进在说服过程中使用近邻学习进行离策评估的应用。这一进步使发送者能够仅使用行为策略的观察数据来评估替代信号策略，从而避免了昂贵的新实验的必要性。 

---
# Remember, but also, Forget: Bridging Myopic and Perfect Recall Fairness with Past-Discounting 

**Title (ZH)**: 记住，但也忘却：过去折扣下的短视与完美回忆公平性桥梁 

**Authors**: Ashwin Kumar, William Yeoh  

**Link**: [PDF](https://arxiv.org/pdf/2504.01154)  

**Abstract**: Dynamic resource allocation in multi-agent settings often requires balancing efficiency with fairness over time--a challenge inadequately addressed by conventional, myopic fairness measures. Motivated by behavioral insights that human judgments of fairness evolve with temporal distance, we introduce a novel framework for temporal fairness that incorporates past-discounting mechanisms. By applying a tunable discount factor to historical utilities, our approach interpolates between instantaneous and perfect-recall fairness, thereby capturing both immediate outcomes and long-term equity considerations. Beyond aligning more closely with human perceptions of fairness, this past-discounting method ensures that the augmented state space remains bounded, significantly improving computational tractability in sequential decision-making settings. We detail the formulation of discounted-recall fairness in both additive and averaged utility contexts, illustrate its benefits through practical examples, and discuss its implications for designing balanced, scalable resource allocation strategies. 

**Abstract (ZH)**: 动态多代理环境中资源分配往往需要在效率与时间维度上的公平性之间取得平衡——这超出了传统短视公平性度量的处理能力。受人类公平判断随时间距离发生变化的行为洞察启发，我们提出了一种新的时序公平性框架，该框架包含过去的折扣机制。通过调整历史效用的折扣因子，我们的方法在瞬时公平性和完美的记忆公平性之间进行插值，从而同时考虑即时结果和长期公平性考量。除了更贴近人类对公平性的感知之外，这种方法还确保扩充后的状态空间保持有界，从而在顺序决策中极大地提高了计算上的可处理性。我们详细阐述了折扣记忆公平性的形式化定义，通过实用示例展示了其优势，并讨论了其对设计平衡且可扩展的资源分配策略的影响。 

---
# Benchmarking Synthetic Tabular Data: A Multi-Dimensional Evaluation Framework 

**Title (ZH)**: 合成表格数据的基准测试：一个多维度评估框架 

**Authors**: Andrey Sidorenko, Michael Platzer, Mario Scriminaci, Paul Tiwald  

**Link**: [PDF](https://arxiv.org/pdf/2504.01908)  

**Abstract**: Evaluating the quality of synthetic data remains a key challenge for ensuring privacy and utility in data-driven research. In this work, we present an evaluation framework that quantifies how well synthetic data replicates original distributional properties while ensuring privacy. The proposed approach employs a holdout-based benchmarking strategy that facilitates quantitative assessment through low- and high-dimensional distribution comparisons, embedding-based similarity measures, and nearest-neighbor distance metrics. The framework supports various data types and structures, including sequential and contextual information, and enables interpretable quality diagnostics through a set of standardized metrics. These contributions aim to support reproducibility and methodological consistency in benchmarking of synthetic data generation techniques. The code of the framework is available at this https URL. 

**Abstract (ZH)**: 评价合成数据的质量仍然是确保数据驱动研究中隐私性和效用的关键挑战。本文提出了一种评价框架，该框架量化合成数据复制原始分布特性的能力，同时确保隐私性。所提出的方法采用了一种基于保留集的基准策略，通过低维和高维分布比较、嵌入式相似性度量和最近邻距离度量实现定量评估。该框架支持各种数据类型和结构，包括序列和上下文信息，并通过一组标准化指标提供可解释的质量诊断。这些贡献旨在支持合成数据生成技术基准测试中的重现性和方法一致性。该框架的代码可在此处访问：this https URL。 

---
# Accelerating IoV Intrusion Detection: Benchmarking GPU-Accelerated vs CPU-Based ML Libraries 

**Title (ZH)**: 加速IoV入侵检测：GPU加速机器学习库与CPU基机器学习库的基准测试 

**Authors**: Furkan Çolhak, Hasan Coşkun, Tsafac Nkombong Regine Cyrille, Tedi Hoxa, Mert İlhan Ecevit, Mehmet Nafiz Aydın  

**Link**: [PDF](https://arxiv.org/pdf/2504.01905)  

**Abstract**: The Internet of Vehicles (IoV) may face challenging cybersecurity attacks that may require sophisticated intrusion detection systems, necessitating a rapid development and response system. This research investigates the performance advantages of GPU-accelerated libraries (cuML) compared to traditional CPU-based implementations (scikit-learn), focusing on the speed and efficiency required for machine learning models used in IoV threat detection environments. The comprehensive evaluations conducted employ four machine learning approaches (Random Forest, KNN, Logistic Regression, XGBoost) across three distinct IoV security datasets (OTIDS, GIDS, CICIoV2024). Our findings demonstrate that GPU-accelerated implementations dramatically improved computational efficiency, with training times reduced by a factor of up to 159 and prediction speeds accelerated by up to 95 times compared to traditional CPU processing, all while preserving detection accuracy. This remarkable performance breakthrough empowers researchers and security specialists to harness GPU acceleration for creating faster, more effective threat detection systems that meet the urgent real-time security demands of today's connected vehicle networks. 

**Abstract (ZH)**: 车辆网络（IoV）可能面临严峻的网络安全攻击，这可能需要复杂的入侵检测系统，从而需要快速的发展和响应系统。本研究探讨了GPU加速库（cuML）与传统基于CPU的实现（scikit-learn）相比的性能优势，重点关注用于IoV威胁检测环境中所需的机器学习模型的速度和效率。全面的评估采用四种机器学习方法（随机森林、KNN、逻辑回归、XGBoost）和三个不同的IoV安全数据集（OTIDS、GIDS、CICIoV2024）。研究发现，GPU加速实现显著提高了计算效率，与传统CPU处理相比，训练时间最多减少了159倍，预测速度加快了95倍，同时保持了检测准确性。这一卓越的性能突破使研究人员和安全专家能够利用GPU加速来创建更快、更有效的威胁检测系统，以满足当今联网车辆网络的紧急实时安全需求。 

---
# Graphically Speaking: Unmasking Abuse in Social Media with Conversation Insights 

**Title (ZH)**: 图形说话：借助对话洞察揭露社交媒体中的虐待行为 

**Authors**: Célia Nouri, Jean-Philippe Cointet, Chloé Clavel  

**Link**: [PDF](https://arxiv.org/pdf/2504.01902)  

**Abstract**: Detecting abusive language in social media conversations poses significant challenges, as identifying abusiveness often depends on the conversational context, characterized by the content and topology of preceding comments. Traditional Abusive Language Detection (ALD) models often overlook this context, which can lead to unreliable performance metrics. Recent Natural Language Processing (NLP) methods that integrate conversational context often depend on limited and simplified representations, and report inconsistent results. In this paper, we propose a novel approach that utilize graph neural networks (GNNs) to model social media conversations as graphs, where nodes represent comments, and edges capture reply structures. We systematically investigate various graph representations and context windows to identify the optimal configuration for ALD. Our GNN model outperform both context-agnostic baselines and linear context-aware methods, achieving significant improvements in F1 scores. These findings demonstrate the critical role of structured conversational context and establish GNNs as a robust framework for advancing context-aware abusive language detection. 

**Abstract (ZH)**: 利用图神经网络建模社交媒体对话以检测网络欺凌语言：系统探究最优配置 

---
# Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions 

**Title (ZH)**: 多重ODE解辅助外推的增强扩散采样 

**Authors**: Jinyoung Choi, Junoh Kang, Bohyung Han  

**Link**: [PDF](https://arxiv.org/pdf/2504.01855)  

**Abstract**: Diffusion probabilistic models (DPMs), while effective in generating high-quality samples, often suffer from high computational costs due to their iterative sampling process. To address this, we propose an enhanced ODE-based sampling method for DPMs inspired by Richardson extrapolation, which reduces numerical error and improves convergence rates. Our method, RX-DPM, leverages multiple ODE solutions at intermediate time steps to extrapolate the denoised prediction in DPMs. This significantly enhances the accuracy of estimations for the final sample while maintaining the number of function evaluations (NFEs). Unlike standard Richardson extrapolation, which assumes uniform discretization of the time grid, we develop a more general formulation tailored to arbitrary time step scheduling, guided by local truncation error derived from a baseline sampling method. The simplicity of our approach facilitates accurate estimation of numerical solutions without significant computational overhead, and allows for seamless and convenient integration into various DPMs and solvers. Additionally, RX-DPM provides explicit error estimates, effectively demonstrating the faster convergence as the leading error term's order increases. Through a series of experiments, we show that the proposed method improves the quality of generated samples without requiring additional sampling iterations. 

**Abstract (ZH)**: 基于Richardson外推的改进ODE采样方法增强DPM 

---
# Rethinking industrial artificial intelligence: a unified foundation framework 

**Title (ZH)**: 重新思考工业人工智能：统一基础框架 

**Authors**: Jay Lee, Hanqi Su  

**Link**: [PDF](https://arxiv.org/pdf/2504.01797)  

**Abstract**: Recent advancement in industrial artificial intelligence (AI) is reshaping the industry, driving smarter manufacturing, predictive maintenance, and intelligent decision-making. However, existing approaches often focus primarily on algorithms and models, overlooking the importance of systematically integrating domain knowledge, data, and models to ensure more comprehensive and effective AI solutions. Therefore, the effective development and deployment of Industrial AI solutions require a more comprehensive and systematic approach. To address this gap, this paper summarizes previous research and rethinks the role of industrial AI and presents a unified industrial AI foundation framework comprising three core modules: knowledge module, data module, and model module. These modules help to extend and enhance the industrial AI methodology platform, supporting various industrial applications. In addition, a case study on rotating machinery diagnosis demonstrates the framework's effectiveness, and several future directions are highlighted for the development of the industrial AI foundation framework. 

**Abstract (ZH)**: Recent advancement in industrial artificial intelligence (AI)正在重塑工业领域，推动更智能的制造、预测性维护和智能决策。然而，现有方法通常主要集中在算法和模型上，忽视了系统地整合领域知识、数据和模型以确保更全面和有效的AI解决方案的重要性。因此，工业AI解决方案的有效开发和部署需要一种更全面和系统的方法。为解决这一差距，本文总结了先前的研究，重新思考工业AI的作用，并提出了一种统一的工业AI基础框架，包括三个核心模块：知识模块、数据模块和模型模块。这些模块有助于扩展和增强工业AI方法平台，支持各种工业应用。此外，旋转机械诊断案例研究证明了该框架的有效性，并提出了工业AI基础框架发展的若干未来方向。 

---
# CLaP -- State Detection from Time Series 

**Title (ZH)**: CLaP -- 时间序列中的状态检测 

**Authors**: Arik Ermshaus, Patrick Schäfer, Ulf Leser  

**Link**: [PDF](https://arxiv.org/pdf/2504.01783)  

**Abstract**: The ever-growing amount of sensor data from machines, smart devices, and the environment leads to an abundance of high-resolution, unannotated time series (TS). These recordings encode the recognizable properties of latent states and transitions from physical phenomena that can be modelled as abstract processes. The unsupervised localization and identification of these states and their transitions is the task of time series state detection (TSSD). We introduce CLaP, a new, highly accurate and efficient algorithm for TSSD. It leverages the predictive power of time series classification for TSSD in an unsupervised setting by applying novel self-supervision techniques to detect whether data segments emerge from the same state or not. To this end, CLaP cross-validates a classifier with segment-labelled subsequences to quantify confusion between segments. It merges labels from segments with high confusion, representing the same latent state, if this leads to an increase in overall classification quality. We conducted an experimental evaluation using 391 TS from four benchmarks and found CLaP to be significantly more precise in detecting states than five state-of-the-art competitors. It achieves the best accuracy-runtime tradeoff and is scalable to large TS. We provide a Python implementation of CLaP, which can be deployed in TS analysis workflows. 

**Abstract (ZH)**: CLaP：一种新的高效高精度的无监督时间序列状态检测算法 

---
# Sky of Unlearning (SoUL): Rewiring Federated Machine Unlearning via Selective Pruning 

**Title (ZH)**: 解learn天空(SoUL):基于选择性剪枝的 federated 机器卸载重构 

**Authors**: Md Mahabub Uz Zaman, Xiang Sun, Jingjing Yao  

**Link**: [PDF](https://arxiv.org/pdf/2504.01705)  

**Abstract**: The Internet of Drones (IoD), where drones collaborate in data collection and analysis, has become essential for applications such as surveillance and environmental monitoring. Federated learning (FL) enables drones to train machine learning models in a decentralized manner while preserving data privacy. However, FL in IoD networks is susceptible to attacks like data poisoning and model inversion. Federated unlearning (FU) mitigates these risks by eliminating adversarial data contributions, preventing their influence on the model. This paper proposes sky of unlearning (SoUL), a federated unlearning framework that efficiently removes the influence of unlearned data while maintaining model performance. A selective pruning algorithm is designed to identify and remove neurons influential in unlearning but minimally impact the overall performance of the model. Simulations demonstrate that SoUL outperforms existing unlearning methods, achieves accuracy comparable to full retraining, and reduces computation and communication overhead, making it a scalable and efficient solution for resource-constrained IoD networks. 

**Abstract (ZH)**: 无人机物联网中的去学习（SoUL）框架：高效去除未学习数据影响的同时保持模型性能 

---
# Segmentation variability and radiomics stability for predicting Triple-Negative Breast Cancer subtype using Magnetic Resonance Imaging 

**Title (ZH)**: 基于磁共振成像预测三阴性乳腺癌亚型的分割变异性和 Radiomics 稳定性研究 

**Authors**: Isabella Cama, Alejandro Guzmán, Cristina Campi, Michele Piana, Karim Lekadir, Sara Garbarino, Oliver Díaz  

**Link**: [PDF](https://arxiv.org/pdf/2504.01692)  

**Abstract**: Most papers caution against using predictive models for disease stratification based on unselected radiomic features, as these features are affected by contouring variability. Instead, they advocate for the use of the Intraclass Correlation Coefficient (ICC) as a measure of stability for feature selection. However, the direct effect of segmentation variability on the predictive models is rarely studied. This study investigates the impact of segmentation variability on feature stability and predictive performance in radiomics-based prediction of Triple-Negative Breast Cancer (TNBC) subtype using Magnetic Resonance Imaging. A total of 244 images from the Duke dataset were used, with segmentation variability introduced through modifications of manual segmentations. For each mask, explainable radiomic features were selected using the Shapley Additive exPlanations method and used to train logistic regression models. Feature stability across segmentations was assessed via ICC, Pearson's correlation, and reliability scores quantifying the relationship between feature stability and segmentation variability. Results indicate that segmentation accuracy does not significantly impact predictive performance. While incorporating peritumoral information may reduce feature reproducibility, it does not diminish feature predictive capability. Moreover, feature selection in predictive models is not inherently tied to feature stability with respect to segmentation, suggesting that an overreliance on ICC or reliability scores for feature selection might exclude valuable predictive features. 

**Abstract (ZH)**: 基于分割变异性的放射omics模型在三阴性乳腺癌亚型预测中的影响： Magnetic Resonance Imaging 证据 

---
# K-P Quantum Neural Networks 

**Title (ZH)**: K-P量子神经网络 

**Authors**: Elija Perrier  

**Link**: [PDF](https://arxiv.org/pdf/2504.01673)  

**Abstract**: We present an extension of K-P time-optimal quantum control solutions using global Cartan $KAK$ decompositions for geodesic-based solutions. Extending recent time-optimal \emph{constant-$\theta$} control results, we integrate Cartan methods into equivariant quantum neural network (EQNN) for quantum control tasks. We show that a finite-depth limited EQNN ansatz equipped with Cartan layers can replicate the constant-$\theta$ sub-Riemannian geodesics for K-P problems. We demonstrate how for certain classes of control problem on Riemannian symmetric spaces, gradient-based training using an appropriate cost function converges to certain global time-optimal solutions when satisfying simple regularity conditions. This generalises prior geometric control theory methods and clarifies how optimal geodesic estimation can be performed in quantum machine learning contexts. 

**Abstract (ZH)**: 我们提出了一种基于全局Cartan $KAK$分解的地节线解法扩展K-P时间最优量子控制方法，将其应用于equivariant量子神经网络（EQNN）中的量子控制任务。我们将Cartan方法集成到equivariant量子神经网络中，展示了一种具有Cartan层的有限深度限定EQNN范式可以复制K-P问题的常数-$\theta$次优地节线。我们证明，在满足简单正则条件时，针对Riemannian对称空间中某些控制问题的梯度训练可以通过合适的成本函数收敛到某些全局时间最优解。这一成果扩展了先前的几何控制理论方法，并明确了在量子机器学习背景下最优地节线估计的实现方式。 

---
# Anomaly Detection for Hybrid Butterfly Subspecies via Probability Filtering 

**Title (ZH)**: 混合蝴蝶亚种的异常检测通过概率过滤 

**Authors**: Bo-Kai Ruan, Yi-Zeng Fang, Hong-Han Shuai, Juinn-Dar Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01671)  

**Abstract**: Detecting butterfly hybrids requires knowledge of the parent subspecies, and the process can be tedious when encountering a new subspecies. This study focuses on a specific scenario where a model trained to recognize hybrid species A can generalize to species B when B biologically mimics A. Since species A and B share similar patterns, we leverage BioCLIP as our feature extractor to capture features based on their taxonomy. Consequently, the algorithm designed for species A can be transferred to B, as their hybrid and non-hybrid patterns exhibit similar relationships. To determine whether a butterfly is a hybrid, we adopt proposed probability filtering and color jittering to augment and simulate the mimicry. With these approaches, we achieve second place in the official development phase. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 检测蝴蝶杂交体需要了解其亲本亚种的信息，当遇到新的亚种时，这一过程可能会很繁琐。本研究集中于一种特定场景，即训练用于识别杂交种群A的模型能够泛化到生物模拟A的种群B。由于种群A和B具有相似的模式，我们利用BioCLIP作为特征提取器，基于它们的分类学特征来提取特征。因此，用于识别种群A的算法可以转移到种群B，因为它们的杂交和非杂交模式表现出相似的关系。为了确定一只蝴蝶是否为杂交体，我们采用了提议的概率过滤和颜色抖动方法来增强和模拟模拟现象。通过这些方法，我们在官方开发阶段取得了第二名的成绩。我们的代码可在以下网址公开获取：这个https URL。 

---
# Market-Oriented Flow Allocation for Thermal Solar Plants: An Auction-Based Methodology with Artificial Intelligence 

**Title (ZH)**: 面向市场的热太阳能电站流量分配方法：基于拍卖的人工智能技术 

**Authors**: Sara Ruiz-Moreno, Antonio J. Gallego, Antonio J. Gallego, Antonio J. Gallego  

**Link**: [PDF](https://arxiv.org/pdf/2504.01652)  

**Abstract**: This paper presents a novel method to optimize thermal balance in parabolic trough collector (PTC) plants. It uses a market-based system to distribute flow among loops combined with an artificial neural network (ANN) to reduce computation and data requirements. This auction-based approach balances loop temperatures, accommodating varying thermal losses and collector efficiencies. Validation across different thermal losses, optical efficiencies, and irradiance conditions-sunny, partially cloudy, and cloudy-show improved thermal power output and intercept factors compared to a no-allocation system. It demonstrates scalability and practicality for large solar thermal plants, enhancing overall performance. The method was first validated through simulations on a realistic solar plant model, then adapted and successfully tested in a 50 MW solar trough plant, demonstrating its advantages. Furthermore, the algorithms have been implemented, commissioned, and are currently operating in 13 commercial solar trough plants. 

**Abstract (ZH)**: 本文提出了一种新型方法以优化抛物柱型聚光_collectors(PTC)系统中的热平衡。该方法结合了一个基于市场的流分配系统和人工神经网络(ANN)，以减少计算和数据需求。基于拍卖的方法平衡了环路温度，适应不同的热损失和集热器效率变化。在不同的热损失、光学效率和光照条件（晴天、部分阴天和阴天）下验证，显示出相比于无分配系统的改进热功率输出和截距因子。该方法证明了在大型太阳能热电站中的可扩展性和实用性，提升整体性能。该方法首先通过现实的太阳能电站模型仿真得到了验证，随后在50 MW太阳能槽式电站中进行了调整和成功测试，证实了其优势。此外，相关算法已在13个商业槽式太阳能电站中实现、调试并运行。 

---
# Benchmarking the Spatial Robustness of DNNs via Natural and Adversarial Localized Corruptions 

**Title (ZH)**: 基于自然和 adversarial 局部扰动的空间 robustness 基准测试：DNNs 的空间鲁棒性评估 

**Authors**: Giulia Marchiori Pietrosanti, Giulio Rossolini, Alessandro Biondi, Giorgio Buttazzo  

**Link**: [PDF](https://arxiv.org/pdf/2504.01632)  

**Abstract**: The robustness of DNNs is a crucial factor in safety-critical applications, particularly in complex and dynamic environments where localized corruptions can arise. While previous studies have evaluated the robustness of semantic segmentation (SS) models under whole-image natural or adversarial corruptions, a comprehensive investigation into the spatial robustness of dense vision models under localized corruptions remained underexplored. This paper fills this gap by introducing specialized metrics for benchmarking the spatial robustness of segmentation models, alongside with an evaluation framework to assess the impact of localized corruptions. Furthermore, we uncover the inherent complexity of characterizing worst-case robustness using a single localized adversarial perturbation. To address this, we propose region-aware multi-attack adversarial analysis, a method that enables a deeper understanding of model robustness against adversarial perturbations applied to specific regions. The proposed metrics and analysis were evaluated on 15 segmentation models in driving scenarios, uncovering key insights into the effects of localized corruption in both natural and adversarial forms. The results reveal that models respond to these two types of threats differently; for instance, transformer-based segmentation models demonstrate notable robustness to localized natural corruptions but are highly vulnerable to adversarial ones and vice-versa for CNN-based models. Consequently, we also address the challenge of balancing robustness to both natural and adversarial localized corruptions by means of ensemble models, thereby achieving a broader threat coverage and improved reliability for dense vision tasks. 

**Abstract (ZH)**: Dense视觉模型在局部扰动下的空间鲁棒性分析与评估 

---
# Horizon Scans can be accelerated using novel information retrieval and artificial intelligence tools 

**Title (ZH)**: 使用新的信息检索和人工智能工具可以加速地平线扫描。 

**Authors**: Lena Schmidt, Oshin Sharma, Chris Marshall, Sonia Garcia Gonzalez Moral  

**Link**: [PDF](https://arxiv.org/pdf/2504.01627)  

**Abstract**: Introduction: Horizon scanning in healthcare assesses early signals of innovation, crucial for timely adoption. Current horizon scanning faces challenges in efficient information retrieval and analysis, especially from unstructured sources like news, presenting a need for innovative tools. Methodology: The study introduces SCANAR and AIDOC, open-source Python-based tools designed to improve horizon scanning. SCANAR automates the retrieval and processing of news articles, offering functionalities such as de-duplication and unsupervised relevancy ranking. AIDOC aids filtration by leveraging AI to reorder textual data based on relevancy, employing neural networks for semantic similarity, and subsequently prioritizing likely relevant entries for human review. Results: Twelve internal datasets from horizon scans and four external benchmarking datasets were used. SCANAR improved retrieval efficiency by automating processes previously dependent on manual labour. AIDOC displayed work-saving potential, achieving around 62% reduction in manual review efforts at 95% recall. Comparative analysis with benchmarking data showed AIDOC's performance was similar to existing systematic review automation tools, though performance varied depending on dataset characteristics. A smaller case-study on our news datasets shows the potential of ensembling large language models within the active-learning process for faster detection of relevant articles across news datasets. Conclusion: The validation indicates that SCANAR and AIDOC show potential to enhance horizon scanning efficiency by streamlining data retrieval and prioritisation. These tools may alleviate methodological limitations and allow broader, swifter horizon scans. Further studies are suggested to optimize these models and to design new workflows and validation processes that integrate large language models. 

**Abstract (ZH)**: 介绍：医疗卫生领域的前瞻扫描评估早期创新信号，对于及时采纳至关重要。当前前瞻扫描面临高效信息检索和分析的挑战，尤其是在处理新闻等非结构化来源时，需要创新工具。方法：研究引入了SCANAR和AIDOC，这两种开源的Python工具旨在提高前瞻扫描的效率。SCANAR自动化新闻文章的检索和处理，提供去重和无监督相关性排名等功能。AIDOC通过利用AI重新排序文本数据以提高相关性，并使用神经网络计算语义相似性，进而优先处理需要人工审查的相关条目。结果：使用了12个内部前瞻扫描数据集和4个外部基准数据集。SCANAR通过自动化以往依赖人工劳动的过程提高了检索效率。AIDOC展示了节省人工工作时间的潜力，实现了约62%的降低手动审查工作量，同时在95%的召回率下。与基准数据的比较分析显示，AIDOC的性能与现有的系统评价自动化工具相似，但性能取决于数据集的特性。一个小规模案例研究表明，在积极学习过程中集成大型语言模型有潜力加快新闻数据集中相关文章的检测速度。结论：验证表明，SCANAR和AIDOC有潜力通过简化数据检索和优先级排序提高前瞻扫描的效率。这些工具可能缓解方法论限制，并允许更广泛、更快的前瞻扫描。建议进一步研究以优化这些模型，并设计结合大型语言模型的新工作流程和验证过程。 

---
# Pro-DG: Procedural Diffusion Guidance for Architectural Facade Generation 

**Title (ZH)**: Pro-DG: 建筑 facades 生成的程序化扩散引导方法 

**Authors**: Aleksander Plocharski, Jan Swidzinski, Przemyslaw Musialski  

**Link**: [PDF](https://arxiv.org/pdf/2504.01571)  

**Abstract**: We present Pro-DG, a framework for procedurally controllable photo-realistic facade generation that combines a procedural shape grammar with diffusion-based image synthesis. Starting from a single input image, we reconstruct its facade layout using grammar rules, then edit that structure through user-defined transformations. As facades are inherently multi-hierarchical structures, we introduce hierarchical matching procedure that aligns facade structures at different levels which is used to introduce control maps to guide a generative diffusion pipeline. This approach retains local appearance fidelity while accommodating large-scale edits such as floor duplication or window rearrangement. We provide a thorough evaluation, comparing Pro-DG against inpainting-based baselines and synthetic ground truths. Our user study and quantitative measurements indicate improved preservation of architectural identity and higher edit accuracy. Our novel method is the first to integrate neuro-symbolically derived shape-grammars for modeling with modern generative model and highlights the broader potential of such approaches for precise and controllable image manipulation. 

**Abstract (ZH)**: Pro-DG：一种结合过程化形状语法和扩散基础图像合成的程序可控逼真外立面生成框架 

---
# Optimizing Package Delivery with Quantum Annealers: Addressing Time-Windows and Simultaneous Pickup and Delivery 

**Title (ZH)**: 使用量子退火优化包裹配送：解决时间窗口和同时 Pickup 与 Delivery 问题 

**Authors**: Eneko Osaba, Esther Villar-Rodriguez, Pablo Miranda-Rodriguez, Antón Asla  

**Link**: [PDF](https://arxiv.org/pdf/2504.01560)  

**Abstract**: Recent research at the intersection of quantum computing and routing problems has been highly prolific. Much of this work focuses on classical problems such as the Traveling Salesman Problem and the Vehicle Routing Problem. The practical applicability of these problems depends on the specific objectives and constraints considered. However, it is undeniable that translating complex real-world requirements into these classical formulations often proves challenging. In this paper, we resort to our previously published quantum-classical technique for addressing real-world-oriented routing problems, known as Quantum for Real Package Delivery (Q4RPD), and elaborate on solving additional realistic problem instances. Accordingly, this paper emphasizes the following characteristics: i) simultaneous pickup and deliveries, ii) time-windows, and iii) mobility restrictions by vehicle type. To illustrate the application of Q4RPD, we have conducted an experimentation comprising seven instances, serving as a demonstration of the newly developed features. 

**Abstract (ZH)**: 近期，量子计算与路由问题交叉领域的研究极为活跃。许多研究集中在诸如旅行商问题和车辆路由问题等经典问题上。这些经典问题的实际适用性取决于所考虑的具体目标和约束条件。然而，将复杂的现实世界需求转化为这些经典形式通常极具挑战性。本文采用我们之前发表的适用于现实导向路由问题的量子-经典技术，即Quantum for Real Package Delivery (Q4RPD)，并探讨了解决更多现实问题实例的方法。相应地，本文强调了以下特点：i) 同时进行取货和配送，ii) 时间窗，以及 iii) 车辆类型引起的移动性限制。为了说明Q4RPD的应用，我们进行了包括七种实例的实验，作为新开发功能的演示。 

---
# Hyperbolic Diffusion Recommender Model 

**Title (ZH)**: 双曲扩散推荐模型 

**Authors**: Meng Yuan, Yutian Xiao, Wei Chen, Chu Zhao, Deqing Wang, Fuzhen Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01541)  

**Abstract**: Diffusion models (DMs) have emerged as the new state-of-the-art family of deep generative models. To gain deeper insights into the limitations of diffusion models in recommender systems, we investigate the fundamental structural disparities between images and items. Consequently, items often exhibit distinct anisotropic and directional structures that are less prevalent in images. However, the traditional forward diffusion process continuously adds isotropic Gaussian noise, causing anisotropic signals to degrade into noise, which impairs the semantically meaningful representations in recommender systems.
Inspired by the advancements in hyperbolic spaces, we propose a novel \textit{\textbf{H}yperbolic} \textit{\textbf{D}iffusion} \textit{\textbf{R}ecommender} \textit{\textbf{M}odel} (named HDRM). Unlike existing directional diffusion methods based on Euclidean space, the intrinsic non-Euclidean structure of hyperbolic space makes it particularly well-adapted for handling anisotropic diffusion processes. In particular, we begin by formulating concepts to characterize latent directed diffusion processes within a geometrically grounded hyperbolic space. Subsequently, we propose a novel hyperbolic latent diffusion process specifically tailored for users and items. Drawing upon the natural geometric attributes of hyperbolic spaces, we impose structural restrictions on the space to enhance hyperbolic diffusion propagation, thereby ensuring the preservation of the intrinsic topology of user-item graphs. Extensive experiments on three benchmark datasets demonstrate the effectiveness of HDRM. 

**Abstract (ZH)**: Hyperbolic Diffusion Recommender Model 

---
# Domain Guidance: A Simple Transfer Approach for a Pre-trained Diffusion Model 

**Title (ZH)**: 领域引导：一种预训练扩散模型的简单转移方法 

**Authors**: Jincheng Zhong, Xiangcheng Zhang, Jianmin Wang, Mingsheng Long  

**Link**: [PDF](https://arxiv.org/pdf/2504.01521)  

**Abstract**: Recent advancements in diffusion models have revolutionized generative modeling. However, the impressive and vivid outputs they produce often come at the cost of significant model scaling and increased computational demands. Consequently, building personalized diffusion models based on off-the-shelf models has emerged as an appealing alternative. In this paper, we introduce a novel perspective on conditional generation for transferring a pre-trained model. From this viewpoint, we propose *Domain Guidance*, a straightforward transfer approach that leverages pre-trained knowledge to guide the sampling process toward the target domain. Domain Guidance shares a formulation similar to advanced classifier-free guidance, facilitating better domain alignment and higher-quality generations. We provide both empirical and theoretical analyses of the mechanisms behind Domain Guidance. Our experimental results demonstrate its substantial effectiveness across various transfer benchmarks, achieving over a 19.6% improvement in FID and a 23.4% improvement in FD$_\text{DINOv2}$ compared to standard fine-tuning. Notably, existing fine-tuned models can seamlessly integrate Domain Guidance to leverage these benefits, without additional training. 

**Abstract (ZH)**: 最近在扩散模型方面的进步已经革新了生成模型。然而，它们所产生的令人印象深刻且生动的输出往往伴随着模型规模的大幅增加和更高的计算需求。因此，基于现成模型构建个性化的扩散模型已成为一种有吸引力的替代方案。在本文中，我们提出了一种新的视角来实现条件生成，并介绍了用于迁移预训练模型的Domain Guidance方法。Domain Guidance利用预训练知识引导采样过程向目标领域靠拢，其形式类似于高级的无分类器引导，从而实现更好的领域对齐和更高的生成质量。我们从实证和理论两个方面分析了Domain Guidance的机制。实验结果表明，与标准微调相比，Domain Guidance在多种迁移基准测试中表现出显著的有效性，FID和FD$_\text{DINOv2}$分别提高了19.6%和23.4%，且现有的微调模型可以无缝集成Domain Guidance，无需额外训练。 

---
# HH-PIM: Dynamic Optimization of Power and Performance with Heterogeneous-Hybrid PIM for Edge AI Devices 

**Title (ZH)**: HH-PIM：面向边缘AI设备的异构混合PIM的功率和性能动态优化 

**Authors**: Sangmin Jeon, Kangju Lee, Kyeongwon Lee, Woojoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.01468)  

**Abstract**: Processing-in-Memory (PIM) architectures offer promising solutions for efficiently handling AI applications in energy-constrained edge environments. While traditional PIM designs enhance performance and energy efficiency by reducing data movement between memory and processing units, they are limited in edge devices due to continuous power demands and the storage requirements of large neural network weights in SRAM and DRAM. Hybrid PIM architectures, incorporating non-volatile memories like MRAM and ReRAM, mitigate these limitations but struggle with a mismatch between fixed computing resources and dynamically changing inference workloads. To address these challenges, this study introduces a Heterogeneous-Hybrid PIM (HH-PIM) architecture, comprising high-performance MRAM-SRAM PIM modules and low-power MRAM-SRAM PIM modules. We further propose a data placement optimization algorithm that dynamically allocates data based on computational demand, maximizing energy efficiency. FPGA prototyping and power simulations with processors featuring HH-PIM and other PIM types demonstrate that the proposed HH-PIM achieves up to $60.43$ percent average energy savings over conventional PIMs while meeting application latency requirements. These results confirm the suitability of HH-PIM for adaptive, energy-efficient AI processing in edge devices. 

**Abstract (ZH)**: 基于内存的处理（PIM）架构为在能量受限的边缘环境中高效处理AI应用提供了有 promise 的解决方案。该研究提出了一种异构混合PIM（HH-PIM）架构，结合了高性能的MRAM-SRAM PIM模块和低功耗的MRAM-SRAM PIM模块，并提出了一种数据放置优化算法，动态分配数据以最大化能效。FPGA原型验证和功率仿真结果显示，提出的HH-PIM相比于传统PIM，平均能效提高了60.43%，同时满足应用延迟要求。这些结果证实HH-PIM适用于边缘设备中的自适应、高效能AI处理。 

---
# Generative Retrieval and Alignment Model: A New Paradigm for E-commerce Retrieval 

**Title (ZH)**: 生成式检索和对齐模型：电子商务检索的新范式 

**Authors**: Ming Pang, Chunyuan Yuan, Xiaoyu He, Zheng Fang, Donghao Xie, Fanyi Qu, Xue Jiang, Changping Peng, Zhangang Lin, Zheng Luo, Jingping Shao  

**Link**: [PDF](https://arxiv.org/pdf/2504.01403)  

**Abstract**: Traditional sparse and dense retrieval methods struggle to leverage general world knowledge and often fail to capture the nuanced features of queries and products. With the advent of large language models (LLMs), industrial search systems have started to employ LLMs to generate identifiers for product retrieval. Commonly used identifiers include (1) static/semantic IDs and (2) product term sets. The first approach requires creating a product ID system from scratch, missing out on the world knowledge embedded within LLMs. While the second approach leverages this general knowledge, the significant difference in word distribution between queries and products means that product-based identifiers often do not align well with user search queries, leading to missed product recalls. Furthermore, when queries contain numerous attributes, these algorithms generate a large number of identifiers, making it difficult to assess their quality, which results in low overall recall efficiency.
To address these challenges, this paper introduces a novel e-commerce retrieval paradigm: the Generative Retrieval and Alignment Model (GRAM). GRAM employs joint training on text information from both queries and products to generate shared text identifier codes, effectively bridging the gap between queries and products. This approach not only enhances the connection between queries and products but also improves inference efficiency. The model uses a co-alignment strategy to generate codes optimized for maximizing retrieval efficiency. Additionally, it introduces a query-product scoring mechanism to compare product values across different codes, further boosting retrieval efficiency. Extensive offline and online A/B testing demonstrates that GRAM significantly outperforms traditional models and the latest generative retrieval models, confirming its effectiveness and practicality. 

**Abstract (ZH)**: 一种生成式检索与对齐模型：电子商务检索的新范式 

---
# Virtual Reality and Artificial Intelligence as Psychological Countermeasures in Space and Other Isolated and Confined Environments: A Scoping Review 

**Title (ZH)**: 虚拟现实与人工智能在太空及其他孤立受限环境中的心理应对措施：一种范围性综述 

**Authors**: Jennifer Sharp, Joshua Kelson, Daryl South, Anthony Saliba, Muhammad Ashad Kabir  

**Link**: [PDF](https://arxiv.org/pdf/2504.01366)  

**Abstract**: Spaceflight is an isolated and confined environment (ICE) that exposes astronauts to psychological hazards, such as stress, danger, and monotony. Virtual reality (VR) and artificial intelligence (AI) technologies can serve as psychological countermeasures as they can digitally simulate immersive environments, interactive companions, and therapeutic experiences. Our study employs a scoping literature review approach to identify what is currently known about the use and effectiveness of VR and AI-based interventions as psychological countermeasures to improve mood or emotional states in adults in space or other ICEs. Additionally, this review aimed to identify gaps in the knowledge base and whether a systematic review with meta-analysis was warranted. The review included studies where the intervention was used or intended for use in space or other extraterrestrial environments (ICE). Our search strategy yielded 19 studies from 3390 records across seven major databases. All studies focused on VR-based interventions, with no eligible AI-based intervention studies found. VR interventions were found to be effective for relaxation and improving mood, emergency training, as an interactive communication platform, for comparing interior designs, and for enhancing exercise. There were improvements for measures of mood and emotion\n (e.g., anxiety and stress); however, user preferences varied, and some instances of cybersickness were reported. A systematic review with meta-analysis is not recommended due to the heterogeneity of results. There is significant scope for further research into the use of VR for a wider range of mood and emotion variables using standardised assessment instruments. Additionally, the potential application of AI as a psychological countermeasure warrants further investigation. 

**Abstract (ZH)**: 空间飞行是一种孤立和受限的环境（ICE），会给宇航员带来心理危害，如压力、危险和单调。虚拟现实（VR）和人工智能（AI）技术可以作为心理对策，因为它们可以数字模拟沉浸式环境、互动伙伴和治疗体验。本研究采用范围性文献综述的方法，旨在识别目前关于VR和AI基干预措施作为心理对策，在空间或其他ICE中改善成人情绪或情感状态的认知情况。此外，本综述还旨在识别知识空白，并确定是否需要进行系统性综述和元分析。综述包括在空间或其它外星球环境（ICE）中使用或计划使用的干预措施的研究。检索策略共从七个主要数据库中获取了3390条记录中的19项研究。所有研究均集中在VR干预措施上，未发现符合条件的AI基干预措施研究。研究结果发现，VR干预措施对于放松、改善情绪、应急培训、作为互动沟通平台、室内设计比较以及增强锻炼等方面均有效。情绪和情感指标（如焦虑和压力）有所改善，但用户偏好各异，且有部分出现虚拟现实病的报告。由于结果的异质性，不建议进行系统性综述和元分析。对于使用标准化评估工具来探索更广泛范围的情绪和情感变量VR应用具有巨大的研究潜力。此外，作为心理对策的AI潜在应用也需要进一步研究。 

---
# Biomedical Question Answering via Multi-Level Summarization on a Local Knowledge Graph 

**Title (ZH)**: 基于本地知识图谱的多层级总结 biomedical 问题解答 

**Authors**: Lingxiao Guan, Yuanhao Huang, Jie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.01309)  

**Abstract**: In Question Answering (QA), Retrieval Augmented Generation (RAG) has revolutionized performance in various domains. However, how to effectively capture multi-document relationships, particularly critical for biomedical tasks, remains an open question. In this work, we propose a novel method that utilizes propositional claims to construct a local knowledge graph from retrieved documents. Summaries are then derived via layerwise summarization from the knowledge graph to contextualize a small language model to perform QA. We achieved comparable or superior performance with our method over RAG baselines on several biomedical QA benchmarks. We also evaluated each individual step of our methodology over a targeted set of metrics, demonstrating its effectiveness. 

**Abstract (ZH)**: 在问答（QA）中，检索增强生成（RAG）已在多个领域革新了性能。然而，如何有效地捕获多文档关系，特别是在生物医学任务中尤为关键，仍然是一个开放的问题。在本文中，我们提出了一种新颖的方法，利用命题主张从检索到的文档构建局部知识图谱。然后通过层级摘要从知识图谱中提取总结，以对小型语言模型进行上下文化，使其执行问答任务。我们在多个生物医学问答基准上实现了与RAG基线相当或更优的性能。我们还通过对目标指标集中的每一步方法进行了评估，证明了其有效性。 

---
# Dynamic Graph Structure Estimation for Learning Multivariate Point Process using Spiking Neural Networks 

**Title (ZH)**: 基于突触神经网络学习多元点过程的动态图结构估计 

**Authors**: Biswadeep Chakraborty, Hemant Kumawat, Beomseok Kang, Saibal Mukhopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2504.01246)  

**Abstract**: Modeling and predicting temporal point processes (TPPs) is critical in domains such as neuroscience, epidemiology, finance, and social sciences. We introduce the Spiking Dynamic Graph Network (SDGN), a novel framework that leverages the temporal processing capabilities of spiking neural networks (SNNs) and spike-timing-dependent plasticity (STDP) to dynamically estimate underlying spatio-temporal functional graphs. Unlike existing methods that rely on predefined or static graph structures, SDGN adapts to any dataset by learning dynamic spatio-temporal dependencies directly from the event data, enhancing generalizability and robustness. While SDGN offers significant improvements over prior methods, we acknowledge its limitations in handling dense graphs and certain non-Gaussian dependencies, providing opportunities for future refinement. Our evaluations, conducted on both synthetic and real-world datasets including NYC Taxi, 911, Reddit, and Stack Overflow, demonstrate that SDGN achieves superior predictive accuracy while maintaining computational efficiency. Furthermore, we include ablation studies to highlight the contributions of its core components. 

**Abstract (ZH)**: 基于尖峰神经网络和尖峰时间依赖可塑性的动态图网络（SDGN）：时空点过程建模与预测 

---
# A Conformal Risk Control Framework for Granular Word Assessment and Uncertainty Calibration of CLIPScore Quality Estimates 

**Title (ZH)**: 粒度词评估和CLIPScore质量估计不确定性校准的同构风险管理框架 

**Authors**: Gonçalo Gomes, Chrysoula Zerva, Bruno Martins  

**Link**: [PDF](https://arxiv.org/pdf/2504.01225)  

**Abstract**: This study explores current limitations of learned image captioning evaluation metrics, specifically the lack of granular assessment for individual word misalignments within captions, and the reliance on single-point quality estimates without considering uncertainty. To address these limitations, we propose a simple yet effective strategy for generating and calibrating CLIPScore distributions. Leveraging a model-agnostic conformal risk control framework, we calibrate CLIPScore values for task-specific control variables, to tackle the aforementioned two limitations. Experimental results demonstrate that using conformal risk control, over the distributions produced with simple methods such as input masking, can achieve competitive performance compared to more complex approaches. Our method effectively detects misaligned words, while providing formal guarantees aligned with desired risk levels, and improving the correlation between uncertainty estimations and prediction errors, thus enhancing the overall reliability of caption evaluation metrics. 

**Abstract (ZH)**: 本研究探讨了学习图像字幕评价指标目前的局限性，特别是在个体单词对齐错误的精细评估方面的不足，以及对单一质量估计的依赖而不考虑不确定性。为了解决这些局限性，我们提出了一种简单而有效的策略，用于生成和校准CLIPScore分布。利用一个模型无关的自适应风险控制框架，我们校准CLIPScore值以针对特定任务进行控制，以解决上述两个局限性。实验结果表明，使用自适应风险控制，相较于简单方法（如输入遮掩）生成的分布，可以获得竞争力的表现。我们的方法能够有效检测对齐错误的单词，同时提供与期望风险水平对齐的正式保证，并且提高了不确定性估计与预测误差之间的相关性，从而增强了字幕评价指标的整体可靠性。 

---
# Neural Approaches to SAT Solving: Design Choices and Interpretability 

**Title (ZH)**: 神经网络方法在SAT求解中的设计选择与可解释性 

**Authors**: David Mojžíšek, Jan Hůla, Ziwei Li, Ziyu Zhou, Mikoláš Janota  

**Link**: [PDF](https://arxiv.org/pdf/2504.01173)  

**Abstract**: In this contribution, we provide a comprehensive evaluation of graph neural networks applied to Boolean satisfiability problems, accompanied by an intuitive explanation of the mechanisms enabling the model to generalize to different instances. We introduce several training improvements, particularly a novel closest assignment supervision method that dynamically adapts to the model's current state, significantly enhancing performance on problems with larger solution spaces. Our experiments demonstrate the suitability of variable-clause graph representations with recurrent neural network updates, which achieve good accuracy on SAT assignment prediction while reducing computational demands. We extend the base graph neural network into a diffusion model that facilitates incremental sampling and can be effectively combined with classical techniques like unit propagation. Through analysis of embedding space patterns and optimization trajectories, we show how these networks implicitly perform a process very similar to continuous relaxations of MaxSAT, offering an interpretable view of their reasoning process. This understanding guides our design choices and explains the ability of recurrent architectures to scale effectively at inference time beyond their training distribution, which we demonstrate with test-time scaling experiments. 

**Abstract (ZH)**: 本研究提供了图神经网络在布尔可满足性问题上的全面评估，并直观解释了模型是如何泛化到不同实例的机制。我们引入了几种训练改进，特别是新型的最邻近指派监督方法，该方法能够动态适应模型的当前状态，显著提高了大规模解空间问题上的性能。我们的实验表明，基于变量-子句图表示并使用循环神经网络更新的方法，在SAT赋值预测中实现了良好的准确性，同时降低了计算需求。我们扩展了基础图神经网络为扩散模型，以促进增量采样，并能够与诸如单元传播的经典技术有效地结合。通过嵌入空间模式分析和优化轨迹，我们展示了这些网络如何隐式执行类似于MaxSAT连续放松的过程，提供了对它们推理过程的可解释视角。这种理解指导了我们的设计选择，并解释了循环架构在推理时如何有效地扩展到训练分布之外的规模，我们通过测试时的扩展实验进行演示。 

---
# Catch Me if You Search: When Contextual Web Search Results Affect the Detection of Hallucinations 

**Title (ZH)**: 当基于上下文的网页搜索结果影响幻觉检测时：Catch Me if You Search 

**Authors**: Mahjabin Nahar, Eun-Ju Lee, Jin Won Park, Dongwon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.01153)  

**Abstract**: While we increasingly rely on large language models (LLMs) for various tasks, these models are known to produce inaccurate content or 'hallucinations' with potentially disastrous consequences. The recent integration of web search results into LLMs prompts the question of whether people utilize them to verify the generated content, thereby avoiding falling victim to hallucinations. This study (N = 560) investigated how the provision of search results, either static (fixed search results) or dynamic (participant-driven searches), affect participants' perceived accuracy and confidence in evaluating LLM-generated content (i.e., genuine, minor hallucination, major hallucination), compared to the control condition (no search results). Findings indicate that participants in both static and dynamic conditions (vs. control) rated hallucinated content to be less accurate. However, those in the dynamic condition rated genuine content as more accurate and demonstrated greater overall confidence in their assessments than those in the static or control conditions. In addition, those higher in need for cognition (NFC) rated major hallucinations to be less accurate than low NFC participants, with no corresponding difference for genuine content or minor hallucinations. These results underscore the potential benefits of integrating web search results into LLMs for the detection of hallucinations, as well as the need for a more nuanced approach when developing human-centered systems, taking user characteristics into account. 

**Abstract (ZH)**: 随着我们越来越多地依赖大型语言模型（LLMs）来完成各种任务，这些模型被Known to produce inaccurate content or 'hallucinations' with potentially disastrous consequences.近年来，将网络搜索结果集成到LLMs中引发了人们对人们是否会利用这些结果来验证生成的内容，从而避免落入“幻觉”的陷阱的兴趣。本研究（N = 560）探讨了提供静态（固定搜索结果）或动态（参与者驱动的搜索）搜索结果，如何影响参与者对LLM生成内容的准确性和信心感知（即真实、轻微幻觉、重大幻觉）与没有搜索结果的控制条件相比。研究发现，与控制组相比，静态和动态条件下参与者的幻觉内容被评为更不准确。然而，动态条件下参与者的真内容被评为更准确，并且展示了总体上更高的评估信心。此外，高认知需要（NFC）的参与者比低NFC的参与者更认为重大幻觉不准确，但对真实内容或轻微幻觉没有相应的差异。这些结果强调了将网络搜索结果集成到LLMs中以检测幻觉的潜在益处，以及在开发以用户为中心的系统时需要采取更细致的方法，考虑用户特征。 

---
# ffstruc2vec: Flat, Flexible and Scalable Learning of Node Representations from Structural Identities 

**Title (ZH)**: ffstruc2vec：扁平、灵活且可扩展的结构身份节点表示学习 

**Authors**: Mario Heidrich, Jeffrey Heidemann, Rüdiger Buchkremer, Gonzalo Wandosell Fernández de Bobadilla  

**Link**: [PDF](https://arxiv.org/pdf/2504.01122)  

**Abstract**: Node embedding refers to techniques that generate low-dimensional vector representations of nodes in a graph while preserving specific properties of the nodes. A key challenge in the field is developing scalable methods that can preserve structural properties suitable for the required types of structural patterns of a given downstream application task. While most existing methods focus on preserving node proximity, those that do preserve structural properties often lack the flexibility to preserve various types of structural patterns required by downstream application tasks. This paper introduces ffstruc2vec, a scalable deep-learning framework for learning node embedding vectors that preserve structural identities. Its flat, efficient architecture allows high flexibility in capturing diverse types of structural patterns, enabling broad adaptability to various downstream application tasks. The proposed framework significantly outperforms existing approaches across diverse unsupervised and supervised tasks in practical applications. Moreover, ffstruc2vec enables explainability by quantifying how individual structural patterns influence task outcomes, providing actionable interpretation. To our knowledge, no existing framework combines this level of flexibility, scalability, and structural interpretability, underscoring its unique capabilities. 

**Abstract (ZH)**: 节点嵌入指的是生成图中节点的低维度向量表示以保留节点特定属性的技术。该领域的一个关键挑战是开发可扩展的方法，以保留适合给定下游应用任务所需结构模式的结构属性。虽然现有的大多数方法侧重于保留节点邻近性，但能够保留各种结构模式的方法往往缺乏灵活性以满足下游应用任务的需求。本文提出了ffstruc2vec，这是一种可扩展的深度学习框架，用于学习保留结构身份的节点嵌入向量。其扁平、高效的设计架构允许在捕获各种类型的结构模式方面具有高度灵活性，从而广泛适应各种下游应用任务。所提出的框架在各种实际应用中的无监督和监督任务中显著优于现有方法。此外，ffstruc2vec 通过量化个别结构模式对任务结果的影响来实现可解释性，提供了可操作的解释。据我们所知，没有现有的框架能够结合这种灵活性、可扩展性和结构解释性，突显了其独特的功能。 

---
# Hard-constraining Neumann boundary conditions in physics-informed neural networks via Fourier feature embeddings 

**Title (ZH)**: 通过傅里叶特征嵌入在物理指导神经网络中硬约束纽曼边界条件 

**Authors**: Christopher Straub, Philipp Brendel, Vlad Medvedev, Andreas Rosskopf  

**Link**: [PDF](https://arxiv.org/pdf/2504.01093)  

**Abstract**: We present a novel approach to hard-constrain Neumann boundary conditions in physics-informed neural networks (PINNs) using Fourier feature embeddings. Neumann boundary conditions are used to described critical processes in various application, yet they are more challenging to hard-constrain in PINNs than Dirichlet conditions. Our method employs specific Fourier feature embeddings to directly incorporate Neumann boundary conditions into the neural network's architecture instead of learning them. The embedding can be naturally extended by high frequency modes to better capture high frequency phenomena. We demonstrate the efficacy of our approach through experiments on a diffusion problem, for which our method outperforms existing hard-constraining methods and classical PINNs, particularly in multiscale and high frequency scenarios. 

**Abstract (ZH)**: 一种用于物理引导神经网络中严格约束诺伊曼边界条件的傅里叶特征嵌入方法 

---
# Predicting Movie Production Years through Facial Recognition of Actors with Machine Learning 

**Title (ZH)**: 通过面部识别技术基于演员的脸部特征使用机器学习预测电影拍摄年份 

**Authors**: Asraa Muayed Abdalah, Noor Redha Alkazaz  

**Link**: [PDF](https://arxiv.org/pdf/2504.01047)  

**Abstract**: This study used machine learning algorithms to identify actors and extract the age of actors from images taken randomly from movies. The use of images taken from Arab movies includes challenges such as non-uniform lighting, different and multiple poses for the actors and multiple elements with the actor or a group of actors. Additionally, the use of make-up, wigs, beards, and wearing different accessories and costumes made it difficult for the system to identify the personality of the same actor. The Arab Actors Dataset-AAD comprises 574 images sourced from various movies, encompassing both black and white as well as color compositions. The images depict complete scenes or fragments thereof. Multiple models were employed for feature extraction, and diverse machine learning algorithms were utilized during the classification and prediction stages to determine the most effective algorithm for handling such image types. The study demonstrated the effectiveness of the Logistic Regression model exhibited the best performance compared to other models in the training phase, as evidenced by its AUC, precision, CA and F1score values of 99%, 86%, 85.5% and 84.2% respectively. The findings of this study can be used to improve the precision and reliability of facial recognition technology for various uses as with movies search services, movie suggestion algorithms, and genre classification of movies. 

**Abstract (ZH)**: 本研究使用机器学习算法识别演员并从随机选取的电影图像中抽取演员的年龄。用于分析的阿拉伯电影图像包括非均匀光照、演员的不同及多重姿态、演员或一组演员周围的多个元素等挑战。化妆、假发、胡子以及不同的配饰和服装使得系统难以识别同一演员的身份特征。阿拉伯演员数据集-AAD包含574张来自不同电影的图像，涵盖黑白和彩色两种类型，图像显示的是完整场景或其中片段。多种模型用于特征提取，在分类和预测阶段采用了多种机器学习算法，以确定处理此类图像类型的最佳算法。研究表明，在训练阶段，逻辑回归模型的表现最优，其AUC、精确度、类别准确率和F1分数分别为99%、86%、85.5%和84.2%。本研究的成果可应用于提高面部识别技术的准确性和可靠性，应用于电影搜索服务、电影推荐算法和电影类型分类等方面。 

---
# Are clinicians ethically obligated to disclose their use of medical machine learning systems to patients? 

**Title (ZH)**: 医生是否有伦理义务向患者披露其使用医疗机器学习系统的情况？ 

**Authors**: Joshua Hatherley  

**Link**: [PDF](https://arxiv.org/pdf/2504.01043)  

**Abstract**: It is commonly accepted that clinicians are ethically obligated to disclose their use of medical machine learning systems to patients, and that failure to do so would amount to a moral fault for which clinicians ought to be held accountable. Call this "the disclosure thesis." Four main arguments have been, or could be, given to support the disclosure thesis in the ethics literature: the risk-based argument, the rights-based argument, the materiality argument, and the autonomy argument. In this article, I argue that each of these four arguments are unconvincing, and therefore, that the disclosure thesis ought to be rejected. I suggest that mandating disclosure may also even risk harming patients by providing stakeholders with a way to avoid accountability for harm that results from improper applications or uses of these systems. 

**Abstract (ZH)**: 公认的伦理观点是临床医生有义务向患者披露使用医疗机器学习系统的事实，否则将构成一种临床医生应当承担责任的道德过错。称这一观点为“披露命题”。伦理学文献中提供了或可以提供的支持披露命题的主要论据有：基于风险的论据、基于权利的论据、重要性论据和自主性论据。本文argument认为这四个论据都不令人信服，因此应当拒绝披露命题。我建议，强制披露甚至可能通过为相关方提供逃避责任的方式损害患者。 

---
# Artificial intelligence and democracy: Towards digital authoritarianism or a democratic upgrade? 

**Title (ZH)**: 人工智能与民主：迈向数字专制主义还是民主升级？ 

**Authors**: Fereniki Panagopoulou  

**Link**: [PDF](https://arxiv.org/pdf/2504.01034)  

**Abstract**: Do robots vote? Do machines make decisions instead of us? No, (at least not yet), but this is something that could happen. The impact of Artificial Intelligence (AI) on democracy is a complex issue that requires thorough research and careful regulation. At the most important level, that of the electoral process, it is noted that it is not determined by the AI, but it is greatly impacted by its multiple applications. New types of online campaigns, driven by AI applications, are replacing traditional ones. The potential for manipulating voters and indirectly influencing the electoral outcome should not be underestimated. Certainly, instances of voter manipulation are not absent from traditional political campaigns, with the only difference being that digital manipulation is often carried out without our knowledge, e.g. by monitoring our behavior on social media. Nevertheless, we should not overlook the positive impact that AI has in the upgrading of democratic institutions by providing a forum for participation in decision-making. In this context, as a first step, we look into the potential jeopardization of democratic processes posed by the use of AI tools. Secondly, we consider the possibility of strengthening democratic processes by using AI, as well as the democratization of AI itself through the possibilities it offers. And thirdly, the impact of AI on the representative system is also discussed. The paper is concluded with recommendations and conclusions. 

**Abstract (ZH)**: 机器人投票吗？机器代替我们做决定吗？不，至少目前还没有，但这种情况可能会发生。人工智能（AI）对民主的影响是一个复杂的问题，需要深入研究和谨慎监管。在选举过程这一最关键层面，人们注意到选举并未由AI决定，但受到了其多种应用的重大影响。由AI驱动的新类型在线竞选活动正在取代传统竞选活动。不可忽视利用AI进行选民操纵和间接影响选举结果的潜在风险。当然，在传统政治竞选中也有选民操纵的情况，只是数字操纵往往是在我们不知情的情况下进行的，例如通过监控我们在社交媒体的行为。然而，我们不应忽视AI在提升民主制度方面带来的积极影响，通过提供参与决策的论坛。在这个背景下，首先，我们探讨AI工具使用对民主过程可能构成的威胁；其次，考虑利用AI增强民主过程的可能性，以及通过AI普及化带来的民主化进程；最后，讨论AI对代表制度的影响。论文以建议和结论作结。 

---
# Who is Responsible When AI Fails? Mapping Causes, Entities, and Consequences of AI Privacy and Ethical Incidents 

**Title (ZH)**: 当AI失败时谁来负责？映射AI隐私和伦理事件的原因、主体与后果 

**Authors**: Hilda Hadan, Reza Hadi Mogavi, Leah Zhang-Kennedy, Lennart E. Nacke  

**Link**: [PDF](https://arxiv.org/pdf/2504.01029)  

**Abstract**: The rapid growth of artificial intelligence (AI) technologies has changed decision-making in many fields. But, it has also raised major privacy and ethical concerns. However, many AI incidents taxonomies and guidelines for academia, industry, and government lack grounding in real-world incidents. We analyzed 202 real-world AI privacy and ethical incidents. This produced a taxonomy that classifies incident types across AI lifecycle stages. It accounts for contextual factors such as causes, responsible entities, disclosure sources, and impacts. Our findings show insufficient incident reporting from AI developers and users. Many incidents are caused by poor organizational decisions and legal non-compliance. Only a few legal actions and corrective measures exist, while risk-mitigation efforts are limited. Our taxonomy contributes a structured approach in reporting of future AI incidents. Our findings demonstrate that current AI governance frameworks are inadequate. We urgently need child-specific protections and AI policies on social media. They must moderate and reduce the spread of harmful AI-generated content. Our research provides insights for policymakers and practitioners, which lets them design ethical AI. It also support AI incident detection and risk management. Finally, it guides AI policy development. Improved policies will protect people from harmful AI applications and support innovation in AI systems. 

**Abstract (ZH)**: 人工智能技术的迅速发展改变了许多领域的决策，但也引发了重大隐私和伦理问题。然而，许多面向学术界、工业界和政府的AI事件分类和指南缺乏对实际事件的依据。我们分析了202个真实的AI隐私和伦理事件，产生了涵盖AI生命周期各阶段事件类型的分类体系，考虑了原因、责任主体、披露来源和影响等背景因素。我们的研究发现，AI开发者和用户对事件报告不足，许多事件由组织决策不当和法律违规引起。当前仅有一少量的法律行动和纠正措施，风险缓解努力也有限。我们的分类体系为未来AI事件的报告提供了一种结构化的方法。我们的研究结果表明，当前的AI治理体系是不充分的，迫切需要针对儿童的具体保护措施和社交媒体上的AI政策，以适度和减少有害AI生成内容的传播。我们的研究为政策制定者和从业人员提供了见解，帮助他们设计伦理的AI，支持AI事件的检测和风险管理，并指导AI政策的制定。改进的政策将保护人们免受有害AI应用的影响，并支持AI系统的创新。 

---
# Diagnosis of Pulmonary Hypertension by Integrating Multimodal Data with a Hybrid Graph Convolutional and Transformer Network 

**Title (ZH)**: 基于混合图卷积和变压器网络的多模态数据整合在肺动脉高压诊断中的应用 

**Authors**: Fubao Zhu, Yang Zhang, Gengmin Liang, Jiaofen Nan, Yanting Li, Chuang Han, Danyang Sun, Zhiguo Wang, Chen Zhao, Wenxuan Zhou, Jian He, Yi Xu, Iokfai Cheang, Xu Zhu, Yanli Zhou, Weihua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.01025)  

**Abstract**: Early and accurate diagnosis of pulmonary hypertension (PH) is essential for optimal patient management. Differentiating between pre-capillary and post-capillary PH is critical for guiding treatment decisions. This study develops and validates a deep learning-based diagnostic model for PH, designed to classify patients as non-PH, pre-capillary PH, or post-capillary PH. This retrospective study analyzed data from 204 patients (112 with pre-capillary PH, 32 with post-capillary PH, and 60 non-PH controls) at the First Affiliated Hospital of Nanjing Medical University. Diagnoses were confirmed through right heart catheterization. We selected 6 samples from each category for the test set (18 samples, 10%), with the remaining 186 samples used for the training set. This process was repeated 35 times for testing. This paper proposes a deep learning model that combines Graph convolutional networks (GCN), Convolutional neural networks (CNN), and Transformers. The model was developed to process multimodal data, including short-axis (SAX) sequences, four-chamber (4CH) sequences, and clinical parameters. Our model achieved a performance of Area under the receiver operating characteristic curve (AUC) = 0.81 +- 0.06(standard deviation) and Accuracy (ACC) = 0.73 +- 0.06 on the test set. The discriminative abilities were as follows: non-PH subjects (AUC = 0.74 +- 0.11), pre-capillary PH (AUC = 0.86 +- 0.06), and post-capillary PH (AUC = 0.83 +- 0.10). It has the potential to support clinical decision-making by effectively integrating multimodal data to assist physicians in making accurate and timely diagnoses. 

**Abstract (ZH)**: 早期和准确诊断肺动脉高压（PH）对于患者管理至关重要。鉴别预毛细血管型和post毛细血管型PH对于指导治疗决策至关重要。本研究开发并验证了一种基于深度学习的诊断模型，旨在将患者分类为非PH、预毛细血管型PH或post毛细血管型PH。该回顾性研究分析了南京医科大学第一附属医院204名患者的数据（其中112名患者为预毛细血管型PH，32名患者为post毛细血管型PH，60名非PH对照），通过右心导管检查确认诊断。我们从每类中选择了6个样本进行测试集（18个样本，占10%），其余186个样本用于训练集。此过程重复测试35次。本文提出了一种结合图卷积网络（GCN）、卷积神经网络（CNN）和变换器的深度学习模型。该模型用于处理包括短轴位（SAX）序列、四腔位（4CH）序列和临床参数在内的多模态数据。该模型在测试集上的性能为受试者操作特征曲线下面积（AUC）= 0.81 ± 0.06（标准差），准确率（ACC）= 0.73 ± 0.06。鉴别能力分别为：非PH患者（AUC = 0.74 ± 0.11）、预毛细血管型PH（AUC = 0.86 ± 0.06）和post毛细血管型PH（AUC = 0.83 ± 0.10）。该模型有望通过有效整合多模态数据来支持临床决策，帮助医生进行准确及时的诊断。 

---
