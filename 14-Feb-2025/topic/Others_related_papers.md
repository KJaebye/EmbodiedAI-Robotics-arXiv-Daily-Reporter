# MTDP: Modulated Transformer Diffusion Policy Model 

**Title (ZH)**: MTDP: 调制变换扩散策略模型 

**Authors**: Qianhao Wang, Yinqian Sun, Enmeng Lu, Qian Zhang, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2502.09029)  

**Abstract**: Recent research on robot manipulation based on Behavior Cloning (BC) has made significant progress. By combining diffusion models with BC, diffusion policiy has been proposed, enabling robots to quickly learn manipulation tasks with high success rates. However, integrating diffusion policy with high-capacity Transformer presents challenges, traditional Transformer architectures struggle to effectively integrate guiding conditions, resulting in poor performance in manipulation tasks when using Transformer-based models. In this paper, we investigate key architectural designs of Transformers and improve the traditional Transformer architecture by proposing the Modulated Transformer Diffusion Policy (MTDP) model for diffusion policy. The core of this model is the Modulated Attention module we proposed, which more effectively integrates the guiding conditions with the main input, improving the generative model's output quality and, consequently, increasing the robot's task success rate. In six experimental tasks, MTDP outperformed existing Transformer model architectures, particularly in the Toolhang experiment, where the success rate increased by 12\%. To verify the generality of Modulated Attention, we applied it to the UNet architecture to construct Modulated UNet Diffusion Policy model (MUDP), which also achieved higher success rates than existing UNet architectures across all six experiments. The Diffusion Policy uses Denoising Diffusion Probabilistic Models (DDPM) as the diffusion model. Building on this, we also explored Denoising Diffusion Implicit Models (DDIM) as the diffusion model, constructing the MTDP-I and MUDP-I model, which nearly doubled the generation speed while maintaining performance. 

**Abstract (ZH)**: 基于行为克隆的机器人操作研究：摩爾 Filed 变形器扩散策略（MTDP）模型 

---
# Bilevel Learning for Bilevel Planning 

**Title (ZH)**: bilevel学习用于 bilevel 规划 

**Authors**: Bowen Li, Tom Silver, Sebastian Scherer, Alexander Gray  

**Link**: [PDF](https://arxiv.org/pdf/2502.08697)  

**Abstract**: A robot that learns from demonstrations should not just imitate what it sees -- it should understand the high-level concepts that are being demonstrated and generalize them to new tasks. Bilevel planning is a hierarchical model-based approach where predicates (relational state abstractions) can be leveraged to achieve compositional generalization. However, previous bilevel planning approaches depend on predicates that are either hand-engineered or restricted to very simple forms, limiting their scalability to sophisticated, high-dimensional state spaces. To address this limitation, we present IVNTR, the first bilevel planning approach capable of learning neural predicates directly from demonstrations. Our key innovation is a neuro-symbolic bilevel learning framework that mirrors the structure of bilevel planning. In IVNTR, symbolic learning of the predicate "effects" and neural learning of the predicate "functions" alternate, with each providing guidance for the other. We evaluate IVNTR in six diverse robot planning domains, demonstrating its effectiveness in abstracting various continuous and high-dimensional states. While most existing approaches struggle to generalize (with <35% success rate), our IVNTR achieves an average of 77% success rate on unseen tasks. Additionally, we showcase IVNTR on a mobile manipulator, where it learns to perform real-world mobile manipulation tasks and generalizes to unseen test scenarios that feature new objects, new states, and longer task horizons. Our findings underscore the promise of learning and planning with abstractions as a path towards high-level generalization. 

**Abstract (ZH)**: 一种能直接从演示中学习神经谓词的机器人不应该只是模仿所见的动作，而应该理解展示的高层概念并将其应用于新任务。IVNTR：一种能从演示中学习神经谓词的 bilevel 计划方法 

---
# Deployment-friendly Lane-changing Intention Prediction Powered by Brain-inspired Spiking Neural Networks 

**Title (ZH)**: 基于脑启发脉冲神经网络的部署friendly变道意图预测 

**Authors**: Junjie Yang, Shuqi Shen, Hui Zhong, Qiming Zhang, Hongliang Lu, Hai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.08659)  

**Abstract**: Accurate and real-time prediction of surrounding vehicles' lane-changing intentions is a critical challenge in deploying safe and efficient autonomous driving systems in open-world scenarios. Existing high-performing methods remain hard to deploy due to their high computational cost, long training times, and excessive memory requirements. Here, we propose an efficient lane-changing intention prediction approach based on brain-inspired Spiking Neural Networks (SNN). By leveraging the event-driven nature of SNN, the proposed approach enables us to encode the vehicle's states in a more efficient manner. Comparison experiments conducted on HighD and NGSIM datasets demonstrate that our method significantly improves training efficiency and reduces deployment costs while maintaining comparable prediction accuracy. Particularly, compared to the baseline, our approach reduces training time by 75% and memory usage by 99.9%. These results validate the efficiency and reliability of our method in lane-changing predictions, highlighting its potential for safe and efficient autonomous driving systems while offering significant advantages in deployment, including reduced training time, lower memory usage, and faster inference. 

**Abstract (ZH)**: 基于脑启发突触神经网络的准确实时周边车辆变道意图预测方法 

---
# Analyzable Parameters Dominated Vehicle Platoon Dynamics Modeling and Analysis: A Physics-Encoded Deep Learning Approach 

**Title (ZH)**: 可分析参数主导的车辆车队动力学建模与分析：一种物理编码深度学习方法 

**Authors**: Hao Lyu, Yanyong Guo, Pan Liu, Shuo Feng, Weilin Ren, Quansheng Yue  

**Link**: [PDF](https://arxiv.org/pdf/2502.08658)  

**Abstract**: Recently, artificial intelligence (AI)-enabled nonlinear vehicle platoon dynamics modeling plays a crucial role in predicting and optimizing the interactions between vehicles. Existing efforts lack the extraction and capture of vehicle behavior interaction features at the platoon scale. More importantly, maintaining high modeling accuracy without losing physical analyzability remains to be solved. To this end, this paper proposes a novel physics-encoded deep learning network, named PeMTFLN, to model the nonlinear vehicle platoon dynamics. Specifically, an analyzable parameters encoded computational graph (APeCG) is designed to guide the platoon to respond to the driving behavior of the lead vehicle while ensuring local stability. Besides, a multi-scale trajectory feature learning network (MTFLN) is constructed to capture platoon following patterns and infer the physical parameters required for APeCG from trajectory data. The human-driven vehicle trajectory datasets (HIGHSIM) were used to train the proposed PeMTFLN. The trajectories prediction experiments show that PeMTFLN exhibits superior compared to the baseline models in terms of predictive accuracy in speed and gap. The stability analysis result shows that the physical parameters in APeCG is able to reproduce the platoon stability in real-world condition. In simulation experiments, PeMTFLN performs low inference error in platoon trajectories generation. Moreover, PeMTFLN also accurately reproduces ground-truth safety statistics. The code of proposed PeMTFLN is open source. 

**Abstract (ZH)**: 近年来，AI辅助的非线性车队动力学模型在预测和优化车辆之间的交互方面发挥着关键作用。现有努力缺乏在车队规模上提取和捕捉车辆行为交互特征。更重要的是，保持高建模准确性同时不失去物理可解析性的问题尚未解决。为此，本文提出了一种新的物理编码深度学习网络，命名为PeMTFLN，用于建模非线性车队动力学。具体地，设计了一种可解析参数编码计算图（APeCG）来引导车队响应前车的驾驶行为并确保局部稳定性。此外，构建了一种多尺度轨迹特征学习网络（MTFLN），用于捕捉车队跟随模式并从轨迹数据中推断APeCG所需的物理参数。使用人工驾驶车辆轨迹数据集（HIGHSIM）对提出的PeMTFLN进行训练。轨迹预测实验显示，与基线模型相比，PeMTFLN在速度和间距预测精度上表现出优异性能。稳定性分析结果表明，APeCG中的物理参数能够在实际条件下重现车队稳定性。在仿真实验中，PeMTFLN在生成车队轨迹时具有低推断误差，并且准确再现了地面真实的安全统计数据。提出的PeMTFLN代码开源。 

---
# A Survey of Reinforcement Learning for Optimization in Automation 

**Title (ZH)**: 自动化中强化学习优化综述 

**Authors**: Ahmad Farooq, Kamran Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2502.09417)  

**Abstract**: Reinforcement Learning (RL) has become a critical tool for optimization challenges within automation, leading to significant advancements in several areas. This review article examines the current landscape of RL within automation, with a particular focus on its roles in manufacturing, energy systems, and robotics. It discusses state-of-the-art methods, major challenges, and upcoming avenues of research within each sector, highlighting RL's capacity to solve intricate optimization challenges. The paper reviews the advantages and constraints of RL-driven optimization methods in automation. It points out prevalent challenges encountered in RL optimization, including issues related to sample efficiency and scalability; safety and robustness; interpretability and trustworthiness; transfer learning and meta-learning; and real-world deployment and integration. It further explores prospective strategies and future research pathways to navigate these challenges. Additionally, the survey includes a comprehensive list of relevant research papers, making it an indispensable guide for scholars and practitioners keen on exploring this domain. 

**Abstract (ZH)**: 强化学习（RL）已经成为自动化领域优化挑战的关键工具，推动了多个领域的重大进展。本文综述了RL在自动化领域的当前状况，特别侧重于其在制造、能源系统和机器人领域的应用。文章讨论了各领域的前沿方法、主要挑战和未来研究方向，强调了RL解决复杂优化挑战的能力。论文回顾了RL驱动的自动化优化方法的优势和限制，并指出了在RL优化中常见的挑战，包括样本效率和可扩展性问题；安全性和鲁棒性问题；可解释性和可信性问题；迁移学习和元学习问题；以及实际部署和集成问题。此外，综述还探讨了应对这些挑战的潜在策略和未来研究路径，并提供了一份相关研究文献的综合列表，成为学者和 practitioner 探索该领域的重要指南。 

---
# Dual Formulation for Non-Rectangular Lp Robust Markov Decision Processes 

**Title (ZH)**: 非矩形Lp稳健马尔可夫决策过程的对偶 formulations 

**Authors**: Navdeep Kumar, Adarsh Gupta, Maxence Mohamed Elfatihi, Giorgia Ramponi, Kfir Yehuda Levy, Shie Mannor  

**Link**: [PDF](https://arxiv.org/pdf/2502.09432)  

**Abstract**: We study robust Markov decision processes (RMDPs) with non-rectangular uncertainty sets, which capture interdependencies across states unlike traditional rectangular models. While non-rectangular robust policy evaluation is generally NP-hard, even in approximation, we identify a powerful class of $L_p$-bounded uncertainty sets that avoid these complexity barriers due to their structural simplicity. We further show that this class can be decomposed into infinitely many \texttt{sa}-rectangular $L_p$-bounded sets and leverage its structural properties to derive a novel dual formulation for $L_p$ RMDPs. This formulation provides key insights into the adversary's strategy and enables the development of the first robust policy evaluation algorithms for non-rectangular RMDPs. Empirical results demonstrate that our approach significantly outperforms brute-force methods, establishing a promising foundation for future investigation into non-rectangular robust MDPs. 

**Abstract (ZH)**: 我们研究非矩形不确定性集下的鲁棒马尔可夫决策过程（RMDPs），这类模型能够捕捉状态间的相互依赖关系，不同于传统的矩形模型。尽管非矩形鲁棒策略评估通常在近似计算中是NP-hard问题，但我们识别出一类具有结构简单性的$L_p$-有界不确定性集，避免了这些复杂性障碍。此外，我们证明了这类不确定性集可以分解为无穷多个$\texttt{sa}$-矩形$L_p$-有界集，并利用其结构特性推导出$L_p$ RMDPs的新颖对偶形式化。该形式化提供了对手策略的关键见解，并首次为非矩形RMDPs开发了鲁棒策略评估算法。实验结果表明，我们的方法显著优于暴力方法，为未来非矩形鲁棒MDP的研究奠定了有希望的基础。 

---
# Indeterminacy in Affective Computing: Considering Meaning and Context in Data Collection Practices 

**Title (ZH)**: 情感计算中的不确定性的考虑：在数据采集实践中纳入意义与情境 

**Authors**: Bernd Dudzik, Tiffany Matej Hrkalovic, Chenxu Hao, Chirag Raman, Masha Tsfasman  

**Link**: [PDF](https://arxiv.org/pdf/2502.09294)  

**Abstract**: Automatic Affect Prediction (AAP) uses computational analysis of input data such as text, speech, images, and physiological signals to predict various affective phenomena (e.g., emotions or moods). These models are typically constructed using supervised machine-learning algorithms, which rely heavily on labeled training datasets. In this position paper, we posit that all AAP training data are derived from human Affective Interpretation Processes, resulting in a form of Affective Meaning. Research on human affect indicates a form of complexity that is fundamental to such meaning: it can possess what we refer to here broadly as Qualities of Indeterminacy (QIs) - encompassing Subjectivity (meaning depends on who is interpreting), Uncertainty (lack of confidence regarding meanings' correctness), Ambiguity (meaning contains mutually exclusive concepts) and Vagueness (meaning is situated at different levels in a nested hierarchy). Failing to appropriately consider QIs leads to results incapable of meaningful and reliable predictions. Based on this premise, we argue that a crucial step in adequately addressing indeterminacy in AAP is the development of data collection practices for modeling corpora that involve the systematic consideration of 1) a relevant set of QIs and 2) context for the associated interpretation processes. To this end, we are 1) outlining a conceptual model of AIPs and the QIs associated with the meaning these produce and a conceptual structure of relevant context, supporting understanding of its role. Finally, we use our framework for 2) discussing examples of context-sensitivity-related challenges for addressing QIs in data collection setups. We believe our efforts can stimulate a structured discussion of both the role of aspects of indeterminacy and context in research on AAP, informing the development of better practices for data collection and analysis. 

**Abstract (ZH)**: 自动情感预测中的意义不確定性：數據收集實踐的系統考慮 

---
# Hybrid Answer Set Programming: Foundations and Applications 

**Title (ZH)**: 混合回答集编程：基础与应用 

**Authors**: Nicolas Rühling  

**Link**: [PDF](https://arxiv.org/pdf/2502.09235)  

**Abstract**: Answer Set Programming (ASP) is a powerful tool for solving real-world problems. However, many problems involve numeric values and complex constraints beyond the capabilities of standard ASP solvers. Hybrid solvers like CLINGCON and CLINGO[DL] address this by using specialized methods for specific constraints. However, these solvers lack a strong theoretical foundation.
This issue has first been addressed by introducing the Logic of Here-and-There with constraints (HT_c) as an extension of the Logic of Here-and-There (HT) and its non-monotone extension Equilibrium Logic. Nowadays, HT serves as a logical foundation for ASP and has facilitated a broader understanding of this paradigm. The idea is that HTC (and other extensions) play an analogous role for hybrid ASP.
There remain many open questions about these logics regarding their fundamental characteristics as well as their practical use in solvers, ie. how they can guide the implementation.
Having a formal understanding of these hybrid logics is also needed to better understand the inherent structure of the (real-world) problems they are applied to and to improve their representations in ASP. As an example of an application of ASP we use product configuration. 

**Abstract (ZH)**: 基于Here-and-There的约束逻辑（HT_c）在Answer Set Programming中的应用 

---
# Computational methods for Dynamic Answer Set Programming 

**Title (ZH)**: 动态答集编程的计算方法 

**Authors**: Susana Hahn  

**Link**: [PDF](https://arxiv.org/pdf/2502.09228)  

**Abstract**: In our daily lives and industrial settings, we often encounter dynamic problems that require reasoning over time and metric constraints. These include tasks such as scheduling, routing, and production sequencing. Dynamic logics have traditionally addressed these needs but often lack the flexibility and integration required for comprehensive problem modeling. This research aims to extend Answer Set Programming (ASP), a powerful declarative problem-solving approach, to handle dynamic domains effectively. By integrating concepts from dynamic, temporal, and metric logics into ASP, we seek to develop robust systems capable of modeling complex dynamic problems and performing efficient reasoning tasks, thereby enhancing ASPs applicability in industrial contexts. 

**Abstract (ZH)**: 在日常生活中和工业环境中，我们经常遇到需要时间推理和度量约束的动态问题。这些问题包括调度、路径规划和生产排序等任务。传统动态逻辑虽能解决这些问题，但往往缺乏全面问题建模所需的灵活性和集成性。本研究旨在扩展Answer Set Programming（ASP），一种强大的声明式问题求解方法，使其能够有效处理动态领域。通过将动态、时序和度量逻辑的概念集成到ASP中，我们寻求开发出能够建模复杂动态问题并进行高效推理的稳健系统，从而增强ASP在工业环境中的应用性。 

---
# Generating Causally Compliant Counterfactual Explanations using ASP 

**Title (ZH)**: 基于ASP生成因果合规的反事实解释 

**Authors**: Sopam Dasgupta  

**Link**: [PDF](https://arxiv.org/pdf/2502.09226)  

**Abstract**: This research is focused on generating achievable counterfactual explanations. Given a negative outcome computed by a machine learning model or a decision system, the novel CoGS approach generates (i) a counterfactual solution that represents a positive outcome and (ii) a path that will take us from the negative outcome to the positive one, where each node in the path represents a change in an attribute (feature) value. CoGS computes paths that respect the causal constraints among features. Thus, the counterfactuals computed by CoGS are realistic. CoGS utilizes rule-based machine learning algorithms to model causal dependencies between features. The paper discusses the current status of the research and the preliminary results obtained. 

**Abstract (ZH)**: 本研究专注于生成可实现的反事实解释。给定由机器学习模型或决策系统计算出的负面结果，新颖的CoGS方法生成（i）一个代表正面结果的反事实解决方案，以及（ii）一条从负面结果到正面结果的路径，其中路径中的每个节点表示属性（特征）值的变化。CoGS计算尊重特征间因果约束的路径，因此CoGS计算出的反事实是现实的。CoGS利用基于规则的机器学习算法来建模特征间的因果依赖关系。本文讨论了研究的当前状态及初步结果。 

---
# Order-Sorted Intensional Logic: Expressing Subtyping Polymorphism with Typing Assertions and Quantification over Concepts 

**Title (ZH)**: 有序分种类内逻辑：通过类型声明和概念上的量化表达子类型泛型 

**Authors**: Đorđe Marković, Marc Denecker  

**Link**: [PDF](https://arxiv.org/pdf/2502.09224)  

**Abstract**: Subtyping, also known as subtype polymorphism, is a concept extensively studied in programming language theory, delineating the substitutability relation among datatypes. This property ensures that programs designed for supertype objects remain compatible with their subtypes.
In this paper, we explore the capability of order-sorted logic for utilizing these ideas in the context of Knowledge Representation. We recognize two fundamental limitations: First, the inability of this logic to address the concept  rather than the value  of non-logical symbols, and second, the lack of language constructs for constraining the type of terms. Consequently, we propose guarded order-sorted intensional logic, where guards are language constructs for annotating typing information and intensional logic provides support for quantification over concepts. 

**Abstract (ZH)**: 子类型化，也称为子类型多态性，是编程语言理论中广泛研究的概念，描述了数据类型间的可替换关系。这种性质确保了为超类型对象设计的程序能够与其子类型保持兼容。

在本文中，我们探讨了偏序排序逻辑在知识表示中的应用能力。我们认识到这种逻辑存在的两个基本局限：首先，无法处理非逻辑符号的意义而不是其值；其次，缺乏限制项类型的语言构造。因此，我们提出了一种受保护的偏序排序意向性逻辑，其中受保护的部分是用于标注类型信息的语言构造，意向性逻辑则提供了对概念进行量化的支持。 

---
# ASP-driven User-interaction with Clinguin 

**Title (ZH)**: ASP驱动的用户与Clinguin的互动 

**Authors**: Alexander Beiser, Susana Hahn, Torsten Schaub  

**Link**: [PDF](https://arxiv.org/pdf/2502.09222)  

**Abstract**: We present clinguin, a system for ASP-driven user interface design. Clinguin streamlines the development of user interfaces for ASP developers by letting them build interactive prototypes directly in ASP, eliminating the need for separate frontend languages. To this end, clinguin uses a few dedicated predicates to define user interfaces and the treatment of user-triggered events. This simple design greatly facilitates the specification of user interactions with an ASP system, in our case clingo. 

**Abstract (ZH)**: 基于ASP的用户界面设计系统：clinguin 

---
# Pearce's Characterisation in an Epistemic Domain 

**Title (ZH)**: 佩奇在知识域中的特征化 

**Authors**: Ezgi Iraz Su  

**Link**: [PDF](https://arxiv.org/pdf/2502.09221)  

**Abstract**: Answer-set programming (ASP) is a successful problem-solving approach in logic-based AI. In ASP, problems are represented as declarative logic programs, and solutions are identified through their answer sets. Equilibrium logic (EL) is a general-purpose nonmonotonic reasoning formalism, based on a monotonic logic called here-and-there logic. EL was basically proposed by Pearce as a foundational framework of ASP. Epistemic specifications (ES) are extensions of ASP-programs with subjective literals. These new modal constructs in the ASP-language make it possible to check whether a regular literal of ASP is true in every (or some) answer-set of a program. ES-programs are interpreted by world-views, which are essentially collections of answer-sets. (Reflexive) autoepistemic logic is a nonmonotonic formalism, modeling self-belief (knowledge) of ideally rational agents. A relatively new semantics for ES is based on a combination of EL and (reflexive) autoepistemic logic. In this paper, we first propose an overarching framework in the epistemic ASP domain. We then establish a correspondence between existing (reflexive) (auto)epistemic equilibrium logics and our easily-adaptable comprehensive framework, building on Pearce's characterisation of answer-sets as equilibrium models. We achieve this by extending Ferraris' work on answer sets for propositional theories to the epistemic case and reveal the relationship between some ES-semantic proposals. 

**Abstract (ZH)**: 基于知识的谓词逻辑程序综合框架与扩展世界观均衡逻辑的对应关系研究 

---
# Mind the Gaps: Logical English, Prolog, and Multi-agent Systems for Autonomous Vehicles 

**Title (ZH)**: 注意差距：逻辑英语、Prolog与多agent系统在自主车辆中的应用 

**Authors**: Galileo Sartor, Adam Wyner, Giuseppe Contissa  

**Link**: [PDF](https://arxiv.org/pdf/2502.09216)  

**Abstract**: In this paper, we present a modular system for representing and reasoning with legal aspects of traffic rules for autonomous vehicles. We focus on a subset of the United Kingdom's Highway Code (HC) related to junctions. As human drivers and automated vehicles (AVs) will interact on the roads, especially in urban environments, we claim that an accessible, unitary, high-level computational model should exist and be applicable to both users. Autonomous vehicles introduce a shift in liability that should not bring disadvantages or increased burden on human drivers. We develop a system "in silico" of the model.  The proposed system is built of three main components: a natural language interface, using Logical English, which encodes the rules; an internal representation of the rules in Prolog; and an multi-agent-based simulation environment, built in NetLogo. The three components interact: Logical English is translated into and out of Prolog (along with some support code); Prolog and NetLogo interface via predicates. Such a modular approach enables the different components to carry different "burdens" in the overall system; it also allows swapping of modules. Given NetLogo, we can visualize the effect of the modeled rules as well as validate the system with a simple dynamic running scenario. Designated agents monitor the behaviour of the vehicles for compliance and record potential violations where they occur. The information on potential violations is then utilized by Validators, to determine whether the violation is punishable, differentiating between exceptions and cases. 

**Abstract (ZH)**: 本文提出了一种模块化系统，用于表示和推理自主车辆交通规则的法律方面。我们关注的是英国公路行为守则（HP）中与交叉路口相关的子集。随着人类驾驶者和自主车辆（AVs）在道路上相互作用，特别是在城市环境中，我们主张应该存在一个易于访问且统一的高层次计算模型，并且适用于用户。自主车辆引入了责任转移，不应给人类驾驶者带来劣势或增加负担。我们开发了一个“计算盒”中的系统模型。该系统由三个主要组件组成：使用逻辑英语的自然语言接口，用于编码规则；使用Prolog的规则内部表示；以及使用NetLogo构建的多智能体仿真环境。这三部分相互作用：逻辑英语被翻译为并从Prolog中翻译出来（同时包括一些辅助代码）；Prolog和NetLogo通过谓词进行接口。这种模块化的方法使得系统中的不同组件能够承担不同的“负担”；它还允许模块的互换。借助NetLogo，我们可以可视化模型规则的效果，并通过简单的动态运行场景验证系统。指定智能体监控车辆行为以确保合规，并记录违规行为发生的情况。然后，验证器利用这些违规信息，确定是否可以对其进行处罚，区分例外情况和具体案例。 

---
# Counterfactual Explanations as Plans 

**Title (ZH)**: 反事实解释作为计划 

**Authors**: Vaishak Belle  

**Link**: [PDF](https://arxiv.org/pdf/2502.09205)  

**Abstract**: There has been considerable recent interest in explainability in AI, especially with black-box machine learning models.  As correctly observed by the planning community, when the application at hand is not a single-shot decision or prediction, but a sequence of actions that depend on observations, a richer notion of explanations are desirable. 
In this paper, we look to provide a formal account of ``counterfactual explanations," based in terms of action sequences. We then show that this naturally leads to an account of model reconciliation, which might take the form of the user correcting the agent's model, or suggesting actions to the agent's plan. For this, we will need to articulate what is true versus what is known, and we appeal to a modal fragment of the situation calculus to formalise these intuitions. We consider various settings: the agent knowing partial truths, weakened truths and having false beliefs, and show that our definitions easily generalize to these different settings. 

**Abstract (ZH)**: 近年来，人工智能中的可解释性受到了广泛关注，尤其是在黑盒机器学习模型中。正如规划社区所正确指出的，当应用场景不是一个单一决策或预测，而是一系列依赖于观测的动作时，需要一种更为丰富的解释概念。本文旨在基于动作序列提供“反事实解释”的正式说明。随后，我们表明，这自然引出了模型调节的概念，这可能表现为用户修正代理模型或向代理的计划建议行动。为此，我们需要区分什么是真的与什么是已知的，我们使用情境演算的一个模态片段来正式化这些直觉。我们考虑了各种情境：代理知道部分真实情况、弱化的真实情况以及持有错误信念的情况，并展示了我们的定义能够很容易地泛化到这些不同的情境中。 

---
# AoI-Sensitive Data Forwarding with Distributed Beamforming in UAV-Assisted IoT 

**Title (ZH)**: 基于无人机辅助IoT的AoI敏感数据转发与分布式波束形成 

**Authors**: Zifan Lang, Guixia Liu, Geng Sun, Jiahui Li, Zemin Sun, Jiacheng Wang, Victor C.M. Leung  

**Link**: [PDF](https://arxiv.org/pdf/2502.09038)  

**Abstract**: This paper proposes a UAV-assisted forwarding system based on distributed beamforming to enhance age of information (AoI) in Internet of Things (IoT). Specifically, UAVs collect and relay data between sensor nodes (SNs) and the remote base station (BS). However, flight delays increase the AoI and degrade the network performance. To mitigate this, we adopt distributed beamforming to extend the communication range, reduce the flight frequency and ensure the continuous data relay and efficient energy utilization. Then, we formulate an optimization problem to minimize AoI and UAV energy consumption, by jointly optimizing the UAV trajectories and communication schedules. The problem is non-convex and with high dynamic, and thus we propose a deep reinforcement learning (DRL)-based algorithm to solve the problem, thereby enhancing the stability and accelerate convergence speed. Simulation results show that the proposed algorithm effectively addresses the problem and outperforms other benchmark algorithms. 

**Abstract (ZH)**: 基于分布式波束形成的无人机辅助转发系统以增强物联网中信息年龄 

---
# Data Sensor Fusion In Digital Twin Technology For Enhanced Capabilities In A Home Environment 

**Title (ZH)**: 数字孪生技术中的数据传感器融合以增强家庭环境的能力 

**Authors**: Benjamin Momoh, Salisu Yahaya  

**Link**: [PDF](https://arxiv.org/pdf/2502.08874)  

**Abstract**: This paper investigates the integration of data sensor fusion in digital twin technology to bolster home environment capabilities, particularly in the context of challenges brought on by the coronavirus pandemic and its economic effects. The study underscores the crucial role of digital transformation in not just adapting to, but also mitigating disruptions during the fourth industrial revolution. Using the Wit Motion sensor, data was collected for activities such as walking, working, sitting, and lying, with sensors measuring accelerometers, gyroscopes, and magnetometers. The research integrates Cyber-physical systems, IoT, AI, and robotics to fortify digital twin capabilities.
The paper compares sensor fusion methods, including feature-level fusion, decision-level fusion, and Kalman filter fusion, alongside machine learning models like SVM, GBoost, and Random Forest to assess model effectiveness. Results show that sensor fusion significantly improves the accuracy and reliability of these models, as it compensates for individual sensor weaknesses, particularly with magnetometers. Despite higher accuracy in ideal conditions, integrating data from multiple sensors ensures more consistent and reliable results in real-world settings, thereby establishing a robust system that can be confidently applied in practical scenarios. 

**Abstract (ZH)**: 本文研究了数据传感器融合在数字孪生技术中的集成，以增强家庭环境的功能，特别是在冠状病毒 pandemic 和其经济影响带来的挑战背景下。研究强调了数字转型在适应第四次工业革命带来的颠覆中起到的关键作用，不仅是适应，还有减轻这些影响。通过使用Wit Motion传感器收集行走、工作、坐和躺等活动的数据，传感器测量加速度计、陀螺仪和磁力计。研究结合了网络物理系统、物联网、人工智能和机器人技术，以强化数字孪生的能力。本文比较了多种传感器融合方法，包括特征级融合、决策级融合和Kalman滤波融合，以及SVM、GBoost和随机森林等机器学习模型，以评估模型的有效性。结果表明，传感器融合显著提高了这些模型的准确性和可靠性，因为它弥补了单个传感器的弱点，特别是在磁力计方面。尽管在理想条件下具有更高的准确性，但从多个传感器整合数据确保了在实际环境中的更一致和可靠的性能，从而建立了可以自信应用于实际场景的稳健系统。 

---
# Off-Switching Not Guaranteed 

**Title (ZH)**: 关机不可保证 

**Authors**: Sven Neth  

**Link**: [PDF](https://arxiv.org/pdf/2502.08864)  

**Abstract**: Hadfield-Menell et al. (2017) propose the Off-Switch Game, a model of Human-AI cooperation in which AI agents always defer to humans because they are uncertain about our preferences. I explain two reasons why AI agents might not defer. First, AI agents might not value learning. Second, even if AI agents value learning, they might not be certain to learn our actual preferences. 

**Abstract (ZH)**: Hadfield-Menell等（2017）提出Off-Switch Game：一种人类与AI合作的模型，在这种模型中，由于AI agent对人类的偏好不够确定，AI agent总是将决策权交给人类。我解释了AI agent为何可能不会总是将决策权交给人类的两个原因：首先，AI agent可能不重视学习；其次，即便AI agent重视学习，它们也可能不能确定学到了人类的实际偏好。 

---
# Estimating Probabilities of Causation with Machine Learning Models 

**Title (ZH)**: 基于机器学习模型估计因果概率 

**Authors**: Shuai Wang, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.08858)  

**Abstract**: Probabilities of causation play a crucial role in modern decision-making. This paper addresses the challenge of predicting probabilities of causation for subpopulations with insufficient data using machine learning models. Tian and Pearl first defined and derived tight bounds for three fundamental probabilities of causation: the probability of necessity and sufficiency (PNS), the probability of sufficiency (PS), and the probability of necessity (PN). However, estimating these probabilities requires both experimental and observational distributions specific to each subpopulation, which are often unavailable or impractical to obtain with limited population-level data. We assume that the probabilities of causation for each subpopulation are determined by its characteristics. To estimate these probabilities for subpopulations with insufficient data, we propose using machine learning models that draw insights from subpopulations with sufficient data. Our evaluation of multiple machine learning models indicates that, given sufficient population-level data and an appropriate choice of machine learning model and activation function, PNS can be effectively predicted. Through simulation studies, we show that our multilayer perceptron (MLP) model with the Mish activation function achieves a mean absolute error (MAE) of approximately 0.02 in predicting PNS for 32,768 subpopulations using data from around 2,000 subpopulations. 

**Abstract (ZH)**: 因果概率在现代决策中扮演着关键角色。本文探讨了使用机器学习模型预测数据不足子群体的因果概率的挑战。天和 Pearl 首先定义并推导出了三种基本因果概率的紧界：必要性和充分性概率（PNS）、充分性概率（PS）和必要性概率（PN）。然而，估计这些概率需要特定于每个子群体的实验和观察分布，而在有限的人群数据中，这些分布通常不可用或难以获取。我们假设每个子群体的因果概率由其特征决定。为了预测缺乏数据子群体的因果概率，我们提出使用从数据充足子群体中提取洞察的机器学习模型。我们的多种机器学习模型评估表明，在有足够的总体数据和适当选择的机器学习模型和激活函数条件下，PNS 可以有效预测。通过模拟研究，我们显示利用约 2,000 个子群体的数据训练的具有 Mish 激活函数的多层感知机（MLP）模型，在预测 32,768 个子群体的 PNS 时的均绝对误差（MAE）约为 0.02。 

---
# High-Throughput SAT Sampling 

**Title (ZH)**: 高通量SAT采样 

**Authors**: Arash Ardakani, Minwoo Kang, Kevin He, Qijing Huang, John Wawrzynek  

**Link**: [PDF](https://arxiv.org/pdf/2502.08673)  

**Abstract**: In this work, we present a novel technique for GPU-accelerated Boolean satisfiability (SAT) sampling. Unlike conventional sampling algorithms that directly operate on conjunctive normal form (CNF), our method transforms the logical constraints of SAT problems by factoring their CNF representations into simplified multi-level, multi-output Boolean functions. It then leverages gradient-based optimization to guide the search for a diverse set of valid solutions. Our method operates directly on the circuit structure of refactored SAT instances, reinterpreting the SAT problem as a supervised multi-output regression task. This differentiable technique enables independent bit-wise operations on each tensor element, allowing parallel execution of learning processes. As a result, we achieve GPU-accelerated sampling with significant runtime improvements ranging from $33.6\times$ to $523.6\times$ over state-of-the-art heuristic samplers. We demonstrate the superior performance of our sampling method through an extensive evaluation on $60$ instances from a public domain benchmark suite utilized in previous studies. 

**Abstract (ZH)**: 本工作中，我们提出了一种用于GPU加速的布尔可满足性（SAT）采样新技术。不同于传统的直接在 conjunctive normal form (CNF) 上操作的采样算法，我们的方法通过将SAT问题的逻辑约束转换为其CNF表示的简化多层次、多输出布尔函数来重新构造这些约束。然后，利用基于梯度的优化来引导寻找多样化的有效解集。我们的方法直接作用于重新构造的SAT实例的电路结构，将SAT问题重新解释为监督多输出回归任务。这种可微分的技术能够在每个张量元素上独立执行位级操作，允许学习过程的并行执行。因此，我们实现了相对于最先进的启发式采样器显著的运行时间提升，范围从33.6倍到523.6倍。通过在先前研究中使用的公共领域基准套件中的60个实例上进行广泛评估，我们展示了我们采样方法的优越性能。 

---
# Score-of-Mixture Training: Training One-Step Generative Models Made Simple 

**Title (ZH)**: 混合得分训练：简化的一步生成模型训练 

**Authors**: Tejas Jayashankar, J. Jon Ryu, Gregory Wornell  

**Link**: [PDF](https://arxiv.org/pdf/2502.09609)  

**Abstract**: We propose Score-of-Mixture Training (SMT), a novel framework for training one-step generative models by minimizing a class of divergences called the $\alpha$-skew Jensen-Shannon divergence. At its core, SMT estimates the score of mixture distributions between real and fake samples across multiple noise levels. Similar to consistency models, our approach supports both training from scratch (SMT) and distillation using a pretrained diffusion model, which we call Score-of-Mixture Distillation (SMD). It is simple to implement, requires minimal hyperparameter tuning, and ensures stable training. Experiments on CIFAR-10 and ImageNet 64x64 show that SMT/SMD are competitive with and can even outperform existing methods. 

**Abstract (ZH)**: 我们提出了一种新颖的一步生成模型训练框架Score-of-Mixture Training (SMT)，通过最小化一类称为$\alpha$-skew Jensen-Shannon发散来进行训练。SMT的核心在于估计不同噪声水平下真样本和假样本混合分布的得分。与一致性模型相似，我们的方法支持从零开始训练（SMT）和使用预训练扩散模型进行蒸馏，我们称之为Score-of-Mixture Distillation (SMD)。SMT/SMD易于实现，需要微调的超参数最少，并确保训练稳定。实验结果显示，SMT/SMD在CIFAR-10和ImageNet 64x64上的表现与现有方法相当，甚至在某些情况下超越现有方法。 

---
# MorphNLI: A Stepwise Approach to Natural Language Inference Using Text Morphing 

**Title (ZH)**: 形态推理：基于文本形态变化的逐步自然语言推理方法 

**Authors**: Vlad Andrei Negru, Robert Vacareanu, Camelia Lemnaru, Mihai Surdeanu, Rodica Potolea  

**Link**: [PDF](https://arxiv.org/pdf/2502.09567)  

**Abstract**: We introduce MorphNLI, a modular step-by-step approach to natural language inference (NLI). When classifying the premise-hypothesis pairs into {entailment, contradiction, neutral}, we use a language model to generate the necessary edits to incrementally transform (i.e., morph) the premise into the hypothesis. Then, using an off-the-shelf NLI model we track how the entailment progresses with these atomic changes, aggregating these intermediate labels into a final output. We demonstrate the advantages of our proposed method particularly in realistic cross-domain settings, where our method always outperforms strong baselines with improvements up to 12.6% (relative). Further, our proposed approach is explainable as the atomic edits can be used to understand the overall NLI label. 

**Abstract (ZH)**: 我们介绍了一种模块化的自然语言推理（NLI）逐步方法MorphNLI。当将前提-假设对分类为{蕴含、矛盾、中立}时，我们使用语言模型生成必要的编辑，逐步转换（即，形态变化）前提为假设。然后，利用现成的NLI模型跟踪这些原子变化如何推进蕴含关系，并将这些中间标签聚合为最终输出。我们展示了在现实的跨域设置中，所提出的方法总是优于强基线，改进幅度最高可达12.6%（相对改进）。此外，所提出的方法具有可解释性，因为原子编辑可用于理解整体NLI标签。 

---
# Diffusion Models for Molecules: A Survey of Methods and Tasks 

**Title (ZH)**: 分子扩散模型：方法与任务综述 

**Authors**: Liang Wang, Chao Song, Zhiyuan Liu, Yu Rong, Qiang Liu, Shu Wu, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.09511)  

**Abstract**: Generative tasks about molecules, including but not limited to molecule generation, are crucial for drug discovery and material design, and have consistently attracted significant attention. In recent years, diffusion models have emerged as an impressive class of deep generative models, sparking extensive research and leading to numerous studies on their application to molecular generative tasks. Despite the proliferation of related work, there remains a notable lack of up-to-date and systematic surveys in this area. Particularly, due to the diversity of diffusion model formulations, molecular data modalities, and generative task types, the research landscape is challenging to navigate, hindering understanding and limiting the area's growth. To address this, this paper conducts a comprehensive survey of diffusion model-based molecular generative methods. We systematically review the research from the perspectives of methodological formulations, data modalities, and task types, offering a novel taxonomy. This survey aims to facilitate understanding and further flourishing development in this area. The relevant papers are summarized at: this https URL. 

**Abstract (ZH)**: 基于扩散模型的分子生成方法综述：从方法论、数据模态和任务类型的角度 bied分类与分析 

---
# Cracking the Code: Enhancing Development finance understanding with artificial intelligence 

**Title (ZH)**: 破解代码：运用人工智能增强发展融资理解 

**Authors**: Pierre Beaucoral  

**Link**: [PDF](https://arxiv.org/pdf/2502.09495)  

**Abstract**: Analyzing development projects is crucial for understanding donors aid strategies, recipients priorities, and to assess development finance capacity to adress development issues by on-the-ground actions. In this area, the Organisation for Economic Co-operation and Developments (OECD) Creditor Reporting System (CRS) dataset is a reference data source. This dataset provides a vast collection of project narratives from various sectors (approximately 5 million projects). While the OECD CRS provides a rich source of information on development strategies, it falls short in informing project purposes due to its reporting process based on donors self-declared main objectives and pre-defined industrial sectors. This research employs a novel approach that combines Machine Learning (ML) techniques, specifically Natural Language Processing (NLP), an innovative Python topic modeling technique called BERTopic, to categorise (cluster) and label development projects based on their narrative descriptions. By revealing existing yet hidden topics of development finance, this application of artificial intelligence enables a better understanding of donor priorities and overall development funding and provides methods to analyse public and private projects narratives. 

**Abstract (ZH)**: 分析发展项目对于理解捐助者援助策略、受援国优先事项以及评估发展资金能力以解决实地问题至关重要。在这个领域，经济合作与发展组织（OECD）债权人报告系统（CRS）数据集是一个参考数据源。该数据集提供了来自各个部门（约50万个项目）的大量项目叙述。虽然OECD CRS提供了丰富的关于发展策略的信息，但由于其基于捐助者自主申报的主要目标和预定义工业部门的报告流程，它在说明项目目的方面存在不足。本研究采用了一种新颖的方法，结合了机器学习（ML）技术，特别是自然语言处理（NLP），以及一种创新的Python主题建模技术BERTopic，根据项目的叙述描述对其分类和标注。通过揭示发展融资中已存在但未被发现的主题，这一人工智能应用有助于更好地理解捐助者优先事项和整体发展资金情况，并提供了分析公共和私营项目叙述的方法。 

---
# PenTest++: Elevating Ethical Hacking with AI and Automation 

**Title (ZH)**: PenTest++: 通过AI和自动化提升道德黑客技术 

**Authors**: Haitham S. Al-Sinani, Chris J. Mitchell  

**Link**: [PDF](https://arxiv.org/pdf/2502.09484)  

**Abstract**: Traditional ethical hacking relies on skilled professionals and time-intensive command management, which limits its scalability and efficiency. To address these challenges, we introduce PenTest++, an AI-augmented system that integrates automation with generative AI (GenAI) to optimise ethical hacking workflows. Developed in a controlled virtual environment, PenTest++ streamlines critical penetration testing tasks, including reconnaissance, scanning, enumeration, exploitation, and documentation, while maintaining a modular and adaptable design. The system balances automation with human oversight, ensuring informed decision-making at key stages, and offers significant benefits such as enhanced efficiency, scalability, and adaptability. However, it also raises ethical considerations, including privacy concerns and the risks of AI-generated inaccuracies (hallucinations). This research underscores the potential of AI-driven systems like PenTest++ to complement human expertise in cybersecurity by automating routine tasks, enabling professionals to focus on strategic decision-making. By incorporating robust ethical safeguards and promoting ongoing refinement, PenTest++ demonstrates how AI can be responsibly harnessed to address operational and ethical challenges in the evolving cybersecurity landscape. 

**Abstract (ZH)**: 传统伦理黑客依赖于 skilled professionals 和耗时的命令管理，这限制了其可扩展性和效率。为应对这些挑战，我们引入了 PenTest++，一个结合了自动化与生成式人工智能 (GenAI) 的 AI 增强系统，以优化伦理黑客攻击工作流程。PenTest++ 在受控虚拟环境中开发，简化了包括侦察、扫描、枚举、利用和记录在内的关键渗透测试任务，同时保持模块化和可适应的设计。该系统平衡了自动化与人工监控，确保在关键阶段做出明智的决策，并提供了显著的效率、可扩展性和适应性方面的好处。然而，它也引发了伦理方面的考虑，包括隐私问题和 AI 生成不准确性的风险。本研究突显了类似 PenTest++ 的 AI 驱动系统在网络安全领域通过自动化常规任务来补充人类专长的潜力，使专业人员能够专注于战略决策。通过整合强大的伦理保障措施并促进持续改进，PenTest++ 证明了 AI 如何在不断变化的网络安全环境中负责任地应对运营和伦理挑战。 

---
# Relational Conformal Prediction for Correlated Time Series 

**Title (ZH)**: 相关性 conformal 预测方法用于相关时间序列分析 

**Authors**: Andrea Cini, Alexander Jenkins, Danilo Mandic, Cesare Alippi, Filippo Maria Bianchi  

**Link**: [PDF](https://arxiv.org/pdf/2502.09443)  

**Abstract**: We address the problem of uncertainty quantification in time series forecasting by exploiting observations at correlated sequences. Relational deep learning methods leveraging graph representations are among the most effective tools for obtaining point estimates from spatiotemporal data and correlated time series. However, the problem of exploiting relational structures to estimate the uncertainty of such predictions has been largely overlooked in the same context. To this end, we propose a novel distribution-free approach based on the conformal prediction framework and quantile regression. Despite the recent applications of conformal prediction to sequential data, existing methods operate independently on each target time series and do not account for relationships among them when constructing the prediction interval. We fill this void by introducing a novel conformal prediction method based on graph deep learning operators. Our method, named Conformal Relational Prediction (CoRel), does not require the relational structure (graph) to be known as a prior and can be applied on top of any pre-trained time series predictor. Additionally, CoRel includes an adaptive component to handle non-exchangeable data and changes in the input time series. Our approach provides accurate coverage and archives state-of-the-art uncertainty quantification in relevant benchmarks. 

**Abstract (ZH)**: 基于相关序列的时序预测中的不确定性量化：基于图深学习的约束关系预测 

---
# Transformer-Enhanced Variational Autoencoder for Crystal Structure Prediction 

**Title (ZH)**: 基于Transformer增强的变分自编码器晶体结构预测 

**Authors**: Ziyi Chen, Yang Yuan, Siming Zheng, Jialong Guo, Sihan Liang, Yangang Wang, Zongguo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.09423)  

**Abstract**: Crystal structure forms the foundation for understanding the physical and chemical properties of materials. Generative models have emerged as a new paradigm in crystal structure prediction(CSP), however, accurately capturing key characteristics of crystal structures, such as periodicity and symmetry, remains a significant challenge. In this paper, we propose a Transformer-Enhanced Variational Autoencoder for Crystal Structure Prediction (TransVAE-CSP), who learns the characteristic distribution space of stable materials, enabling both the reconstruction and generation of crystal structures. TransVAE-CSP integrates adaptive distance expansion with irreducible representation to effectively capture the periodicity and symmetry of crystal structures, and the encoder is a transformer network based on an equivariant dot product attention mechanism. Experimental results on the carbon_24, perov_5, and mp_20 datasets demonstrate that TransVAE-CSP outperforms existing methods in structure reconstruction and generation tasks under various modeling metrics, offering a powerful tool for crystal structure design and optimization. 

**Abstract (ZH)**: 晶体结构形式是理解材料的物理和化学性质的基础。生成模型新兴为晶体结构预测（CSP）的新范式，然而准确捕获晶体结构的关键特征，如周期性和对称性，仍然是一个重大挑战。在本文中，我们提出了一种增强型变分自编码器（TransVAE-CSP）用于晶体结构预测，该模型学习稳定材料的特征分布空间，既能够重建也能够生成晶体结构。TransVAE-CSP 结合自适应距离扩展和不可约表示，有效捕捉晶体结构的周期性和对称性，编码器基于等变点积注意力机制的Transformer网络。在碳_24、perov_5 和 mp_20 数据集上的实验结果表明，TransVAE-CSP 在各种建模指标下的结构重建和生成任务中均优于现有方法，为晶体结构设计和优化提供了一种强大的工具。 

---
# Simple Path Structural Encoding for Graph Transformers 

**Title (ZH)**: 图变换器的简单路径结构编码 

**Authors**: Louis Airale, Antonio Longa, Mattia Rigon, Andrea Passerini, Roberto Passerone  

**Link**: [PDF](https://arxiv.org/pdf/2502.09365)  

**Abstract**: Graph transformers extend global self-attention to graph-structured data, achieving notable success in graph learning. Recently, random walk structural encoding (RWSE) has been found to further enhance their predictive power by encoding both structural and positional information into the edge representation. However, RWSE cannot always distinguish between edges that belong to different local graph patterns, which reduces its ability to capture the full structural complexity of graphs. This work introduces Simple Path Structural Encoding (SPSE), a novel method that utilizes simple path counts for edge encoding. We show theoretically and experimentally that SPSE overcomes the limitations of RWSE, providing a richer representation of graph structures, particularly for capturing local cyclic patterns. To make SPSE computationally tractable, we propose an efficient approximate algorithm for simple path counting. SPSE demonstrates significant performance improvements over RWSE on various benchmarks, including molecular and long-range graph datasets, achieving statistically significant gains in discriminative tasks. These results pose SPSE as a powerful edge encoding alternative for enhancing the expressivity of graph transformers. 

**Abstract (ZH)**: Graph Transformer中的简单路径结构编码：超越随机游 walk 结构编码的方法 

---
# Neural Spatiotemporal Point Processes: Trends and Challenges 

**Title (ZH)**: 神经空时点过程：趋势与挑战 

**Authors**: Sumantrak Mukherjee, Mouad Elhamdi, George Mohler, David A. Selby, Yao Xie, Sebastian Vollmer, Gerrit Grossmann  

**Link**: [PDF](https://arxiv.org/pdf/2502.09341)  

**Abstract**: Spatiotemporal point processes (STPPs) are probabilistic models for events occurring in continuous space and time. Real-world event data often exhibit intricate dependencies and heterogeneous dynamics. By incorporating modern deep learning techniques, STPPs can model these complexities more effectively than traditional approaches. Consequently, the fusion of neural methods with STPPs has become an active and rapidly evolving research area. In this review, we categorize existing approaches, unify key design choices, and explain the challenges of working with this data modality. We further highlight emerging trends and diverse application domains. Finally, we identify open challenges and gaps in the literature. 

**Abstract (ZH)**: 空间时间点过程（STPPs）是用于描述连续空间和时间中事件发生的概率模型。现实世界中的事件数据常常表现出复杂的依赖关系和异质动力学。通过融合现代深度学习技术，STPPs可以更有效地建模这些复杂性，超越了传统的建模方法。因此，神经方法与STPPs的融合已成为一个活跃且快速发展的研究领域。在本文综述中，我们对现有方法进行分类，统一关键设计选择，并解释处理这种数据模态所面临的挑战。我们进一步强调了新兴趋势和多种应用领域。最后，我们指出现有文献中的开放挑战和空白。 

---
# Graph Diffusion Network for Drug-Gene Prediction 

**Title (ZH)**: 药物-基因预测的图扩散网络 

**Authors**: Jiayang Wu, Wensheng Gan, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.09335)  

**Abstract**: Predicting drug-gene associations is crucial for drug development and disease treatment. While graph neural networks (GNN) have shown effectiveness in this task, they face challenges with data sparsity and efficient contrastive learning implementation. We introduce a graph diffusion network for drug-gene prediction (GDNDGP), a framework that addresses these limitations through two key innovations. First, it employs meta-path-based homogeneous graph learning to capture drug-drug and gene-gene relationships, ensuring similar entities share embedding spaces. Second, it incorporates a parallel diffusion network that generates hard negative samples during training, eliminating the need for exhaustive negative sample retrieval. Our model achieves superior performance on the DGIdb 4.0 dataset and demonstrates strong generalization capability on tripartite drug-gene-disease networks. Results show significant improvements over existing methods in drug-gene prediction tasks, particularly in handling complex heterogeneous relationships. The source code is publicly available at this https URL. 

**Abstract (ZH)**: 基于图扩散网络的药物-基因关联预测（GDNDGP）：一种解决数据稀疏性和高效对比学习问题的框架 

---
# Predicting Drive Test Results in Mobile Networks Using Optimization Techniques 

**Title (ZH)**: 使用优化技术预测移动网络路测结果 

**Authors**: MohammadJava Taheri, Abolfazl Diyanat, MortezaAli Ahmadi, Ali Nazari  

**Link**: [PDF](https://arxiv.org/pdf/2502.09305)  

**Abstract**: Mobile network operators constantly optimize their networks to ensure superior service quality and coverage. This optimization is crucial for maintaining an optimal user experience and requires extensive data collection and analysis. One of the primary methods for gathering this data is through drive tests, where technical teams use specialized equipment to collect signal information across various regions. However, drive tests are both costly and time-consuming, and they face challenges such as traffic conditions, environmental factors, and limited access to certain areas. These constraints make it difficult to replicate drive tests under similar conditions. In this study, we propose a method that enables operators to predict received signal strength at specific locations using data from other drive test points. By reducing the need for widespread drive tests, this approach allows operators to save time and resources while still obtaining the necessary data to optimize their networks and mitigate the challenges associated with traditional drive tests. 

**Abstract (ZH)**: 移动网络运营商不断优化其网络以确保卓越的服务质量和覆盖范围。这种方法对于保持最佳用户体验至关重要，并需要大量数据的收集和分析。收集这些数据的主要方法之一是通过路测，技术团队使用专门的设备在不同地区收集信号信息。然而，路测既耗费成本又耗时，并且面临交通状况、环境因素和某些区域的访问限制等挑战。这些限制使得难以在相似条件下重复路测。在本研究中，我们提出了一种方法，使运营商能够使用其他路测点的数据来预测特定位置的接收信号强度。通过减少广泛进行路测的需求，这种方法允许运营商节省时间和资源，同时仍然获得优化网络和缓解传统路测挑战所需的数据。 

---
# LiSA: Leveraging Link Recommender to Attack Graph Neural Networks via Subgraph Injection 

**Title (ZH)**: LiSA：通过子图注入攻击图神经网络的链接推荐方法 

**Authors**: Wenlun Zhang, Enyan Dai, Kentaro Yoshioka  

**Link**: [PDF](https://arxiv.org/pdf/2502.09271)  

**Abstract**: Graph Neural Networks (GNNs) have demonstrated remarkable proficiency in modeling data with graph structures, yet recent research reveals their susceptibility to adversarial attacks. Traditional attack methodologies, which rely on manipulating the original graph or adding links to artificially created nodes, often prove impractical in real-world settings. This paper introduces a novel adversarial scenario involving the injection of an isolated subgraph to deceive both the link recommender and the node classifier within a GNN system. Specifically, the link recommender is mislead to propose links between targeted victim nodes and the subgraph, encouraging users to unintentionally establish connections and that would degrade the node classification accuracy, thereby facilitating a successful attack. To address this, we present the LiSA framework, which employs a dual surrogate model and bi-level optimization to simultaneously meet two adversarial objectives. Extensive experiments on real-world datasets demonstrate the effectiveness of our method. 

**Abstract (ZH)**: 图神经网络（GNNs）在建模具有图结构的数据方面表现出色，但近期研究揭示了其对 adversarial 攻击的脆弱性。传统的攻击方法依赖于操控原始图或添加链接到人工创建的节点，这些方法在实际应用中往往难以实施。本文提出了一种新颖的 adversarial 场景，涉及注入孤立子图以欺骗 GNN 系统内的链接推荐器和节点分类器。具体而言，链接推荐器被误导提议在目标受害者节点与子图之间建立链接，促使用户无意间建立连接，从而降低节点分类准确性，进而实现攻击成功。为解决这一问题，我们提出了 LiSA 框架，该框架采用双 surrogate 模型和分层次优化，同时满足两个 adversarial 目标。在真实世界数据集上的广泛实验表明了我们方法的有效性。 

---
# Bandit Multiclass List Classification 

**Title (ZH)**: 多臂多类列表分类 

**Authors**: Liad Erez, Tomer Koren  

**Link**: [PDF](https://arxiv.org/pdf/2502.09257)  

**Abstract**: We study the problem of multiclass list classification with (semi-)bandit feedback, where input examples are mapped into subsets of size $m$ of a collection of $K$ possible labels, and the feedback consists of the predicted labels which lie in the set of true labels of the given example. Our main result is for the $(\varepsilon,\delta)$-PAC variant of the problem for which we design an algorithm that returns an $\varepsilon$-optimal hypothesis with high probability using a sample complexity of $O \big( (\mathrm{poly}(K/m) + sm / \varepsilon^2) \log (|H|/\delta) \big)$ where $H$ is the underlying (finite) hypothesis class and $s$ is an upper bound on the number of true labels for a given example. This bound improves upon known bounds for combinatorial semi-bandits whenever $s \ll K$. Moreover, in the regime where $s = O(1)$ the leading terms in our bound match the corresponding full-information rates, implying that bandit feedback essentially comes at no cost. Our PAC learning algorithm is also computationally efficient given access to an ERM oracle for $H$. Additionally, we consider the regret minimization setting where data can be generated adversarially, and establish a regret bound of $\widetilde O(|H| + \sqrt{smT \log |H|})$. Our results generalize and extend those of Erez et al. (2024) who consider the simpler single-label setting corresponding to $s=m=1$, and in fact hold for the more general contextual combinatorial semi-bandit problem with $s$-sparse rewards. 

**Abstract (ZH)**: 多类别列表分类的（半）bandit反馈研究：(ε,δ)-PAC范式下的算法与复杂性分析 

---
# AnomalyGFM: Graph Foundation Model for Zero/Few-shot Anomaly Detection 

**Title (ZH)**: AnomalyGFM: 基于图的模型在零/少量样本异常检测中的应用 

**Authors**: Hezhe Qiao, Chaoxi Niu, Ling Chen, Guansong Pang  

**Link**: [PDF](https://arxiv.org/pdf/2502.09254)  

**Abstract**: Graph anomaly detection (GAD) aims to identify abnormal nodes that differ from the majority of the nodes in a graph, which has been attracting significant attention in recent years. Existing generalist graph models have achieved remarkable success in different graph tasks but struggle to generalize to the GAD task. This limitation arises from their difficulty in learning generalized knowledge for capturing the inherently infrequent, irregular and heterogeneous abnormality patterns in graphs from different domains. To address this challenge, we propose AnomalyGFM, a GAD-oriented graph foundation model that supports zero-shot inference and few-shot prompt tuning for GAD in diverse graph datasets. One key insight is that graph-agnostic representations for normal and abnormal classes are required to support effective zero/few-shot GAD across different graphs. Motivated by this, AnomalyGFM is pre-trained to align data-independent, learnable normal and abnormal class prototypes with node representation residuals (i.e., representation deviation of a node from its neighbors). The residual features essentially project the node information into a unified feature space where we can effectively measure the abnormality of nodes from different graphs in a consistent way. This provides a driving force for the learning of graph-agnostic, discriminative prototypes for the normal and abnormal classes, which can be used to enable zero-shot GAD on new graphs, including very large-scale graphs. If there are few-shot labeled normal nodes available in the new graphs, AnomalyGFM can further support prompt tuning to leverage these nodes for better adaptation. Comprehensive experiments on 11 widely-used GAD datasets with real anomalies, demonstrate that AnomalyGFM significantly outperforms state-of-the-art competing methods under both zero- and few-shot GAD settings. 

**Abstract (ZH)**: 面向图异常检测的图基础模型（AnomalyGFM）：支持零样本推理和少样本提示调优的图异常检测基础模型 

---
# The Joint Entity-Relation Extraction Model Based on Span and Interactive Fusion Representation for Chinese Medical Texts with Complex Semantics 

**Title (ZH)**: 基于跨度和交互融合表示的中文医疗文本复杂语义联合实体-关系提取模型 

**Authors**: Danni Feng, Runzhi Li, Jing Wang, Siyu Yan, Lihong Ma, Yunli Xing  

**Link**: [PDF](https://arxiv.org/pdf/2502.09247)  

**Abstract**: Joint entity-relation extraction is a critical task in transforming unstructured or semi-structured text into triplets, facilitating the construction of large-scale knowledge graphs, and supporting various downstream applications. Despite its importance, research on Chinese text, particularly with complex semantics in specialized domains like medicine, remains limited. To address this gap, we introduce the CH-DDI, a Chinese drug-drug interactions dataset designed to capture the intricacies of medical text. Leveraging the strengths of attention mechanisms in capturing long-range dependencies, we propose the SEA module, which enhances the extraction of complex contextual semantic information, thereby improving entity recognition and relation extraction. Additionally, to address the inefficiencies of existing methods in facilitating information exchange between entity recognition and relation extraction, we present an interactive fusion representation module. This module employs Cross Attention for bidirectional information exchange between the tasks and further refines feature extraction through BiLSTM. Experimental results on both our CH-DDI dataset and public CoNLL04 dataset demonstrate that our model exhibits strong generalization capabilities. On the CH-DDI dataset, our model achieves an F1-score of 96.73% for entity recognition and 78.43% for relation extraction. On the CoNLL04 dataset, it attains an entity recognition precision of 89.54% and a relation extraction accuracy of 71.64%. 

**Abstract (ZH)**: 中文标题：联合实体-关系抽取：Chinese DDITuple数据集的设计与应用 

---
# Logical foundations of Smart Contracts 

**Title (ZH)**: 智能合约的逻辑基础 

**Authors**: Kalonji Kalala  

**Link**: [PDF](https://arxiv.org/pdf/2502.09232)  

**Abstract**: Nowadays, sophisticated domains are emerging which require appropriate formalisms to be specified accurately in order to reason about them. One such domain is constituted of smart contracts that have emerged in cyber physical systems as a way of enforcing formal agreements between components of these systems.  Smart contracts self-execute to run and share business processes through blockchain, in decentralized systems, with many different participants. Legal contracts are in many cases complex documents, with a number of exceptions, and many subcontracts. The implementation of smart contracts based on legal contracts is a long and laborious task, that needs to include all actions, procedures, and the effects of actions related to the execution of the contract. An ongoing open problem in this area is to formally account for smart contracts using a uniform and somewhat universal formalism. This thesis proposes logical foundations to smart contracts using the Situation Calculus, a logic for reasoning about actions. Situation Calculus is one of the prominent logic-based artificial intelligence approaches that provides enough logical mechanism to specify and implement dynamic and complex systems such as contracts. Situation Calculus is suitable to show how worlds dynamically change.  Smart contracts are going to be implement with Golog (written en Prolog), a Situation Calculus-based programming language for modeling complex and dynamic behaviors. 

**Abstract (ZH)**: 现今，一些复杂的领域需要适当的形式化方法来准确地规定和推理。其中一个领域是由智能合约构成的，这类合约在 cyber physical 系统中作为一种方式，用于强制执行这些系统组件之间的正式协议。智能合约自我执行，在区块链驱动的去中心化系统中运行和共享业务流程，涉及众多参与者。法律合同通常是复杂的文档，包含多个例外和子合同。基于法律合同实现智能合约是一项漫长而繁琐的任务，需要涵盖合约执行过程中所有相关的行为、程序及其效果。在这个领域中，一个持续存在的开放问题是使用统一且相对通用的形式化方法正式地描述智能合约。本论文提出使用情况 calculus 为智能合约提供逻辑基础，情况 calculus 是一种基于逻辑的人工智能方法，提供了足够的逻辑手段来规定和实现动态复杂的系统，如合约。情况 calculus 适用于展示世界如何动态变化。智能合约将使用 Golog（用 Prolog 编写）来实现，Golog 是一种基于情况 calculus 的编程语言，用于建模复杂和动态的行为。 

---
# Relating Answer Set Programming and Many-sorted Logics for Formal Verification 

**Title (ZH)**: 将规则集编程与多态逻辑关联起来用于形式验证 

**Authors**: Zachary Hansen  

**Link**: [PDF](https://arxiv.org/pdf/2502.09230)  

**Abstract**: Answer Set Programming (ASP) is an important logic programming paradigm within the field of Knowledge Representation and Reasoning. As a concise, human-readable, declarative language, ASP is an excellent tool for developing trustworthy (especially, artificially intelligent) software systems. However, formally verifying ASP programs offers some unique challenges, such as
1. a lack of modularity (the meanings of rules are difficult to define in isolation from the enclosing program),
2. the ground-and-solve semantics (the meanings of rules are dependent on the input data with which the program is grounded), and
3. limitations of existing tools.
My research agenda has been focused on addressing these three issues with the intention of making ASP verification an accessible, routine task that is regularly performed alongside program development. In this vein, I have investigated alternative semantics for ASP based on translations into the logic of here-and-there and many-sorted first-order logic. These semantics promote a modular understanding of logic programs, bypass grounding, and enable us to use automated theorem provers to automatically verify properties of programs. 

**Abstract (ZH)**: 基于_here-and-there_和_many-sorted一阶逻辑_的ASP替代语义研究 

---
# Graphical Conditions for the Existence, Unicity and Number of Regular Models 

**Title (ZH)**: 图形条件下的正规模型的存在性、唯一性及数量条件 

**Authors**: Van-Giang Trinh, Belaid Benhamou, Sylvain Soliman, François Fages  

**Link**: [PDF](https://arxiv.org/pdf/2502.09220)  

**Abstract**: The regular models of a normal logic program are a particular type of partial (i.e. 3-valued) models which correspond to stable partial models with minimal undefinedness. In this paper, we explore graphical conditions on the dependency graph of a finite ground normal logic program to analyze the existence, unicity and number of regular models for the program. We show three main results: 1) a necessary condition for the existence of non-trivial (i.e. non-2-valued) regular models, 2) a sufficient condition for the unicity of regular models, and 3) two upper bounds for the number of regular models based on positive feedback vertex sets. The first two conditions generalize the finite cases of the two existing results obtained by You and Yuan (1994) for normal logic programs with well-founded stratification. The third result is also new to the best of our knowledge. Key to our proofs is a connection that we establish between finite ground normal logic programs and Boolean network theory. 

**Abstract (ZH)**: 正常逻辑程序的正则模型是一种特殊的部分模型（即3值模型），对应于具有最小不确定性的一种稳定部分模型。本文探讨了有限基本正常逻辑程序的依赖图上的图条件，分析程序的正则模型的存在性、唯一性和数量。我们展示了三个主要结果：1) 非平凡（非2值）正则模型存在的必要条件；2) 正则模型唯一的充分条件；3) 基于正反馈顶点集的正则模型数量的两个上界。前两个条件推广了You和Yuan（1994）关于具有良好奠基分层的正常逻辑程序的有限情况的两个结果。第三个结果据我们所知也是新的。我们证明的关键在于建立了有限基本正常逻辑程序与布尔网络理论之间的联系。 

---
# Abduction of Domain Relationships from Data for VQA 

**Title (ZH)**: 从数据中推导领域关系进行VQA 

**Authors**: Al Mehdi Saadat Chowdhury, Paulo Shakarian, Gerardo I. Simari  

**Link**: [PDF](https://arxiv.org/pdf/2502.09219)  

**Abstract**: In this paper, we study the problem of visual question answering (VQA) where the image and query are represented by ASP programs that lack domain data.  We provide an approach that is orthogonal and complementary to existing knowledge augmentation techniques where we abduce domain relationships of image constructs from past examples. After framing the abduction problem, we provide a baseline approach, and an implementation that significantly improves the accuracy of query answering yet requires few examples. 

**Abstract (ZH)**: 在本文中，我们研究了视觉问答（VQA）问题，其中图像和查询由缺乏领域数据的ASP程序表示。我们提供了一种与现有知识增强技术正交且互补的方法，通过从过往例子中推导图像构造的领域关系。在界定归结问题后，我们提供了一种基线方法及其实现，该实现显著提高了查询回答的准确性，同时只需要少量示例。 

---
# Data2Concept2Text: An Explainable Multilingual Framework for Data Analysis Narration 

**Title (ZH)**: Data2Concept2Text：一种可解释的多语言数据分析叙述框架 

**Authors**: Flavio Bertini, Alessandro Dal Palù, Federica Zaglio, Francesco Fabiano, Andrea Formisano  

**Link**: [PDF](https://arxiv.org/pdf/2502.09218)  

**Abstract**: This paper presents a complete explainable system that interprets a set of data, abstracts the underlying features and describes them in a natural language of choice. The system relies on two crucial stages: (i) identifying emerging properties from data and transforming them into abstract concepts, and (ii) converting these concepts into natural language. Despite the impressive natural language generation capabilities demonstrated by Large Language Models, their statistical nature and the intricacy of their internal mechanism still force us to employ these techniques as black boxes, forgoing trustworthiness. Developing an explainable pipeline for data interpretation would allow facilitating its use in safety-critical environments like processing medical information and allowing non-experts and visually impaired people to access narrated information. To this end, we believe that the fields of knowledge representation and automated reasoning research could present a valid alternative. Expanding on prior research that tackled the first stage (i), we focus on the second stage, named Concept2Text. Being explainable, data translation is easily modeled through logic-based rules, once again emphasizing the role of declarative programming in achieving AI explainability. This paper explores a Prolog/CLP-based rewriting system to interpret concepts-articulated in terms of classes and relations, plus common knowledge-derived from a generic ontology, generating natural language text. Its main features include hierarchical tree rewritings, modular multilingual generation, support for equivalent variants across semantic, grammar, and lexical levels, and a transparent rule-based system. We outline the architecture and demonstrate its flexibility through some examples capable of generating numerous diverse and equivalent rewritings based on the input concept. 

**Abstract (ZH)**: 一种基于Prolog/CLP的可解释数据解释系统：从概念到自然语言的转换 

---
# Efficient OWL2QL Meta-reasoning Using ASP-based Hybrid Knowledge Bases 

**Title (ZH)**: 基于ASP为基础的混合知识库的高效OWL2QL元推理 

**Authors**: Haya Majid Qureshi, Wolfgang Faber  

**Link**: [PDF](https://arxiv.org/pdf/2502.09206)  

**Abstract**: Metamodeling refers to scenarios in ontologies in which classes and roles can be members of classes or occur in roles. This is a desirable modelling feature in several applications, but allowing it without restrictions is problematic for several reasons, mainly because it causes undecidability. Therefore, practical languages either forbid metamodeling explicitly or treat occurrences of classes as instances to be semantically different from other occurrences, thereby not allowing metamodeling semantically. Several extensions have been proposed to provide metamodeling to some extent. Building on earlier work that reduces metamodeling query answering to Datalog query answering, recently reductions to query answering over hybrid knowledge bases were proposed with the aim of using the Datalog transformation only where necessary. Preliminary work showed that the approach works, but the hoped-for performance improvements were not observed yet. In this work we expand on this body of work by improving the theoretical basis of the reductions and by using alternative tools that show competitive performance. 

**Abstract (ZH)**: 元建模是指本体中类和角色可以成为类的成员或出现在角色中的情景。这一特征在多个应用中都是 desirable 的，但如果不加以限制地允许它会引发问题，主要是因为它会导致不可判定性。因此，实用的语言要么明确规定禁止元建模，要么将类的出现视为实例，使其在语义上不同于其他出现，从而不允许在语义上进行元建模。提出了几种扩展来在一定程度上提供元建模。在早前将元建模查询回答归约到 Datalog 查询回答的基础上，最近提出了归约到混合知识库查询回答的策略，目的是仅在必要时使用 Datalog 转换。初步的工作表明该方法可行，但期待的性能改进尚未实现。在本工作中，我们通过改进归约的理论基础并使用替代工具来实现具有竞争力的性能，进一步扩展了这一研究领域。 

---
# Two-Stage Representation Learning for Analyzing Movement Behavior Dynamics in People Living with Dementia 

**Title (ZH)**: 两阶段表示学习分析痴呆患者运动行为动力学 

**Authors**: Jin Cui, Alexander Capstick, Payam Barnaghi, Gregory Scott  

**Link**: [PDF](https://arxiv.org/pdf/2502.09173)  

**Abstract**: In remote healthcare monitoring, time series representation learning reveals critical patient behavior patterns from high-frequency data. This study analyzes home activity data from individuals living with dementia by proposing a two-stage, self-supervised learning approach tailored to uncover low-rank structures. The first stage converts time-series activities into text sequences encoded by a pre-trained language model, providing a rich, high-dimensional latent state space using a PageRank-based method. This PageRank vector captures latent state transitions, effectively compressing complex behaviour data into a succinct form that enhances interpretability. This low-rank representation not only enhances model interpretability but also facilitates clustering and transition analysis, revealing key behavioral patterns correlated with clinicalmetrics such as MMSE and ADAS-COG scores. Our findings demonstrate the framework's potential in supporting cognitive status prediction, personalized care interventions, and large-scale health monitoring. 

**Abstract (ZH)**: 在远程医疗监控中，时间序列表示学习可从高频率数据中揭示关键患者行为模式。本研究通过提出一种针对察觉低秩结构定制的两阶段自我监督学习方法，分析痴呆患者的家庭活动数据。第一阶段将时间序列活动转换为由预训练语言模型编码的文本序列，并使用基于PageRank的方法提供一个丰富的高维潜态空间。基于PageRank的向量捕捉潜态转换，有效将复杂行为数据压缩为易于理解的形式，提高可解释性。这种低秩表示不仅提升了模型的可解释性，还促进了聚类和转换分析，揭示了与MMSE和ADAS-COG评分相关的关键行为模式。我们的研究结果证明了该框架在支持认知状态预测、个性化护理干预和大规模健康监测中的潜力。 

---
# Automatic Pruning via Structured Lasso with Class-wise Information 

**Title (ZH)**: 基于类内信息的结构化lasso自动剪枝 

**Authors**: Xiang Liu, Mingchen Li, Xia Li, Leigang Qu, Zifan Peng, Yijun Song, Zemin Liu, Linshan Jiang, Jialin Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.09125)  

**Abstract**: Most pruning methods concentrate on unimportant filters of neural networks. However, they face the loss of statistical information due to a lack of consideration for class-wise data. In this paper, from the perspective of leveraging precise class-wise information for model pruning, we utilize structured lasso with guidance from Information Bottleneck theory. Our approach ensures that statistical information is retained during the pruning process. With these techniques, we introduce two innovative adaptive network pruning schemes: sparse graph-structured lasso pruning with Information Bottleneck (\textbf{sGLP-IB}) and sparse tree-guided lasso pruning with Information Bottleneck (\textbf{sTLP-IB}). The key aspect is pruning model filters using sGLP-IB and sTLP-IB to better capture class-wise relatedness. Compared to multiple state-of-the-art methods, our approaches demonstrate superior performance across three datasets and six model architectures in extensive experiments. For instance, using the VGG16 model on the CIFAR-10 dataset, we achieve a parameter reduction of 85%, a decrease in FLOPs by 61%, and maintain an accuracy of 94.10% (0.14% higher than the original model); we reduce the parameters by 55% with the accuracy at 76.12% using the ResNet architecture on ImageNet (only drops 0.03%). In summary, we successfully reduce model size and computational resource usage while maintaining accuracy. Our codes are at this https URL. 

**Abstract (ZH)**: Most pruning methods focus on unimportant filters of neural networks. However, they suffer from the loss of statistical information due to a lack of consideration for class-wise data. From the perspective of leveraging precise class-wise information for model pruning, this paper utilizes structured lasso with guidance from Information Bottleneck theory to ensure that statistical information is retained during the pruning process. Two innovative adaptive network pruning schemes, sparse graph-structured lasso pruning with Information Bottleneck (sGLP-IB) and sparse tree-guided lasso pruning with Information Bottleneck (sTLP-IB), are introduced. The key aspect is pruning model filters using sGLP-IB and sTLP-IB to better capture class-wise relatedness. Compared to multiple state-of-the-art methods, our approaches demonstrate superior performance across three datasets and six model architectures in extensive experiments. For instance, using the VGG16 model on the CIFAR-10 dataset, we achieve a parameter reduction of 85%, a decrease in FLOPs by 61%, and maintain an accuracy of 94.10% (0.14% higher than the original model); we reduce the parameters by 55% with the accuracy at 76.12% using the ResNet architecture on ImageNet (only drops 0.03%). In summary, we successfully reduce model size and computational resource usage while maintaining accuracy. Our codes are available at this <https://> URL. 

---
# Improving Deep Regression with Tightness 

**Title (ZH)**: 改进深度回归的紧致性方法 

**Authors**: Shihao Zhang, Yuguang Yan, Angela Yao  

**Link**: [PDF](https://arxiv.org/pdf/2502.09122)  

**Abstract**: For deep regression, preserving the ordinality of the targets with respect to the feature representation improves performance across various tasks. However, a theoretical explanation for the benefits of ordinality is still lacking. This work reveals that preserving ordinality reduces the conditional entropy $H(Z|Y)$ of representation $Z$ conditional on the target $Y$. However, our findings reveal that typical regression losses do little to reduce $H(Z|Y)$, even though it is vital for generalization performance. With this motivation, we introduce an optimal transport-based regularizer to preserve the similarity relationships of targets in the feature space to reduce $H(Z|Y)$. Additionally, we introduce a simple yet efficient strategy of duplicating the regressor targets, also with the aim of reducing $H(Z|Y)$. Experiments on three real-world regression tasks verify the effectiveness of our strategies to improve deep regression. Code: this https URL. 

**Abstract (ZH)**: 对于深度回归任务，保持目标相对于特征表示的序关系可以提高各种任务的性能。然而，关于序关系益处的理论解释仍然缺乏。本工作揭示了保持序关系可以降低条件熵$H(Z|Y)$。然而，我们的发现表明，典型的回归损失对降低$H(Z|Y)$的帮助不大，尽管这对于泛化性能至关重要。基于此动机，我们引入一种基于最优运输的正则化项来保持特征空间中目标的相似性关系，以降低$H(Z|Y)$。此外，我们还提出了一种简单而有效的策略，即复制回归器目标，也旨在降低$H(Z|Y)$。实验证实在三个真实世界的回归任务上，我们的策略能够有效提高深度回归性能。代码：这个链接。 

---
# One-shot Federated Learning Methods: A Practical Guide 

**Title (ZH)**: 一-shot联邦学习方法：一个实用指南 

**Authors**: Xiang Liu, Zhenheng Tang, Xia Li, Yijun Song, Sijie Ji, Zemin Liu, Bo Han, Linshan Jiang, Jialin Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.09104)  

**Abstract**: One-shot Federated Learning (OFL) is a distributed machine learning paradigm that constrains client-server communication to a single round, addressing privacy and communication overhead issues associated with multiple rounds of data exchange in traditional Federated Learning (FL). OFL demonstrates the practical potential for integration with future approaches that require collaborative training models, such as large language models (LLMs). However, current OFL methods face two major challenges: data heterogeneity and model heterogeneity, which result in subpar performance compared to conventional FL methods. Worse still, despite numerous studies addressing these limitations, a comprehensive summary is still lacking. To address these gaps, this paper presents a systematic analysis of the challenges faced by OFL and thoroughly reviews the current methods. We also offer an innovative categorization method and analyze the trade-offs of various techniques. Additionally, we discuss the most promising future directions and the technologies that should be integrated into the OFL field. This work aims to provide guidance and insights for future research. 

**Abstract (ZH)**: One-shot 联邦学习 (OFL): 面临的挑战与现有方法综述及未来方向 

---
# Show Me the Work: Fact-Checkers' Requirements for Explainable Automated Fact-Checking 

**Title (ZH)**: 展示工作内容：事实核查人员对可解释的自动化事实核查的需求 

**Authors**: Greta Warren, Irina Shklovski, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2502.09083)  

**Abstract**: The pervasiveness of large language models and generative AI in online media has amplified the need for effective automated fact-checking to assist fact-checkers in tackling the increasing volume and sophistication of misinformation. The complex nature of fact-checking demands that automated fact-checking systems provide explanations that enable fact-checkers to scrutinise their outputs. However, it is unclear how these explanations should align with the decision-making and reasoning processes of fact-checkers to be effectively integrated into their workflows. Through semi-structured interviews with fact-checking professionals, we bridge this gap by: (i) providing an account of how fact-checkers assess evidence, make decisions, and explain their processes; (ii) examining how fact-checkers use automated tools in practice; and (iii) identifying fact-checker explanation requirements for automated fact-checking tools. The findings show unmet explanation needs and identify important criteria for replicable fact-checking explanations that trace the model's reasoning path, reference specific evidence, and highlight uncertainty and information gaps. 

**Abstract (ZH)**: 大型语言模型和生成式AI在在线媒体中的普遍性加剧了对有效自动化事实核查的需求，以协助事实核查人员应对日益增加的误导信息的数量和复杂性。事实核查的复杂性要求自动化事实核查系统提供可供事实核查人员审查其输出的解释。然而，尚不清楚这些解释应如何与事实核查人员的决策和推理过程对齐，以便有效地集成到其工作流程中。通过半结构化的访谈，我们通过以下三个方面的研究弥合了这一差距：(i) 描述事实核查人员评估证据、做出决策和解释其过程的方式；(ii) 探讨事实核查人员实际使用自动化工具的情况；(iii) 确定事实核查人员对自动化事实核查工具的解释需求。研究发现，存在未满足的解释需求，并且指出了可重复的事实核查解释的重要标准，这些标准追踪模型的推理路径、引用具体证据，并突出不确定性与信息空缺。 

---
# Exploring the Needs of Practising Musicians in Co-Creative AI Through Co-Design 

**Title (ZH)**: 探索实践音乐家在共创人工智能中的需求：共设计方法 

**Authors**: Stephen James Krol, Maria Teresa Llano Rodriguez, Miguel Loor Paredes  

**Link**: [PDF](https://arxiv.org/pdf/2502.09055)  

**Abstract**: Recent advances in generative AI music have resulted in new technologies that are being framed as co-creative tools for musicians with early work demonstrating their potential to add to music practice. While the field has seen many valuable contributions, work that involves practising musicians in the design and development of these tools is limited, with the majority of work including them only once a tool has been developed. In this paper, we present a case study that explores the needs of practising musicians through the co-design of a musical variation system, highlighting the importance of involving a diverse range of musicians throughout the design process and uncovering various design insights. This was achieved through two workshops and a two week ecological evaluation, where musicians from different musical backgrounds offered valuable insights not only on a musical system's design but also on how a musical AI could be integrated into their musical practices. 

**Abstract (ZH)**: Recent进展在生成式AI音乐中的研究已经产生了新的技术，这些技术被框架为音乐家的共创工具，早期的工作显示了它们对音乐实践的潜在贡献。尽管该领域已作出许多有价值的贡献，但在设计和开发这些工具时涉及实际演奏音乐家的工作仍然有限，大多数工作在工具开发完成后才将他们纳入其中。本文通过一个案例研究探讨了实际演奏音乐家的需求，并通过共同设计一个音乐变体系统，突显了在整个设计过程中涉及广泛音乐家的重要性，并揭示了各种设计见解。这通过两场研讨会和为期两周的生态评估实现，不同背景的音乐家不仅提供了有关音乐系统设计的宝贵见解，还探讨了如何将音乐AI整合到他们的音乐实践中。 

---
# Leveraging Member-Group Relations via Multi-View Graph Filtering for Effective Group Recommendation 

**Title (ZH)**: 利用多视图图过滤挖掘成员-群体关系以实现有效的群体推荐 

**Authors**: Chae-Hyun Kim, Yoon-Ryung Choi, Jin-Duk Park, Won-Yong Shin  

**Link**: [PDF](https://arxiv.org/pdf/2502.09050)  

**Abstract**: Group recommendation aims at providing optimized recommendations tailored to diverse groups, enabling groups to enjoy appropriate items. On the other hand, most existing group recommendation methods are built upon deep neural network (DNN) architectures designed to capture the intricate relationships between member-level and group-level interactions. While these DNN-based approaches have proven their effectiveness, they require complex and expensive training procedures to incorporate group-level interactions in addition to member-level interactions. To overcome such limitations, we introduce Group-GF, a new approach for extremely fast recommendations of items to each group via multi-view graph filtering (GF) that offers a holistic view of complex member-group dynamics, without the need for costly model training. Specifically, in Group-GF, we first construct three item similarity graphs manifesting different viewpoints for GF. Then, we discover a distinct polynomial graph filter for each similarity graph and judiciously aggregate the three graph filters. Extensive experiments demonstrate the effectiveness of Group-GF in terms of significantly reducing runtime and achieving state-of-the-art recommendation accuracy. 

**Abstract (ZH)**: 组推荐旨在为不同的组提供优化的个性化推荐，使组能够享受合适的产品。现有大多数组推荐方法基于深度神经网络（DNN）架构，旨在捕捉成员级和组级交互的复杂关系。虽然这些基于DNN的方法证明了其有效性，但它们需要复杂的和昂贵的训练过程来同时考虑成员级和组级交互。为克服这些局限性，我们引入了Group-GF方法，通过多视图图过滤（GF）以极快的速度为每个组推荐物品，同时提供复杂成员-组动态的全面视图，无需进行昂贵的模型训练。具体来说，在Group-GF中，我们首先构建三个物品相似图，体现GF的不同视角。接着，我们为每个相似图发现一个独特的多项式图滤波器，并巧妙地聚合三个滤波器。广泛实验表明，Group-GF在大幅减少运行时间和达到最先进的推荐精度方面非常有效。 

---
# Criteria-Aware Graph Filtering: Extremely Fast Yet Accurate Multi-Criteria Recommendation 

**Title (ZH)**: 基于指标的图过滤：极其快速且准确的多指标推荐 

**Authors**: Jin-Duk Park, Jaemin Yoo, Won-Yong Shin  

**Link**: [PDF](https://arxiv.org/pdf/2502.09046)  

**Abstract**: Multi-criteria (MC) recommender systems, which utilize MC rating information for recommendation, are increasingly widespread in various e-commerce domains. However, the MC recommendation using training-based collaborative filtering, requiring consideration of multiple ratings compared to single-criterion counterparts, often poses practical challenges in achieving state-of-the-art performance along with scalable model training. To solve this problem, we propose CA-GF, a training-free MC recommendation method, which is built upon criteria-aware graph filtering for efficient yet accurate MC recommendations. Specifically, first, we construct an item-item similarity graph using an MC user-expansion graph. Next, we design CA-GF composed of the following key components, including 1) criterion-specific graph filtering where the optimal filter for each criterion is found using various types of polynomial low-pass filters and 2) criteria preference-infused aggregation where the smoothed signals from each criterion are aggregated. We demonstrate that CA-GF is (a) efficient: providing the computational efficiency, offering the extremely fast runtime of less than 0.2 seconds even on the largest benchmark dataset, (b) accurate: outperforming benchmark MC recommendation methods, achieving substantial accuracy gains up to 24% compared to the best competitor, and (c) interpretable: providing interpretations for the contribution of each criterion to the model prediction based on visualizations. 

**Abstract (ZH)**: 基于准则感知图过滤的无需训练的多准则推荐方法 

---
# EventSTR: A Benchmark Dataset and Baselines for Event Stream based Scene Text Recognition 

**Title (ZH)**: EventSTR：基于事件流的场景文本识别基准数据集和基线方法 

**Authors**: Xiao Wang, Jingtao Jiang, Dong Li, Futian Wang, Lin Zhu, Yaowei Wang, Yongyong Tian, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.09020)  

**Abstract**: Mainstream Scene Text Recognition (STR) algorithms are developed based on RGB cameras which are sensitive to challenging factors such as low illumination, motion blur, and cluttered backgrounds. In this paper, we propose to recognize the scene text using bio-inspired event cameras by collecting and annotating a large-scale benchmark dataset, termed EventSTR. It contains 9,928 high-definition (1280 * 720) event samples and involves both Chinese and English characters. We also benchmark multiple STR algorithms as the baselines for future works to compare. In addition, we propose a new event-based scene text recognition framework, termed SimC-ESTR. It first extracts the event features using a visual encoder and projects them into tokens using a Q-former module. More importantly, we propose to augment the vision tokens based on a memory mechanism before feeding into the large language models. A similarity-based error correction mechanism is embedded within the large language model to correct potential minor errors fundamentally based on contextual information. Extensive experiments on the newly proposed EventSTR dataset and two simulation STR datasets fully demonstrate the effectiveness of our proposed model. We believe that the dataset and algorithmic model can innovatively propose an event-based STR task and are expected to accelerate the application of event cameras in various industries. The source code and pre-trained models will be released on this https URL 

**Abstract (ZH)**: 基于生物启发事件相机的场景文本识别 

---
# Zero-shot Concept Bottleneck Models 

**Title (ZH)**: 零样本概念瓶颈模型 

**Authors**: Shin'ya Yamaguchi, Kosuke Nishida, Daiki Chijiwa, Yasutoshi Ida  

**Link**: [PDF](https://arxiv.org/pdf/2502.09018)  

**Abstract**: Concept bottleneck models (CBMs) are inherently interpretable and intervenable neural network models, which explain their final label prediction by the intermediate prediction of high-level semantic concepts. However, they require target task training to learn input-to-concept and concept-to-label mappings, incurring target dataset collections and training resources. In this paper, we present \textit{zero-shot concept bottleneck models} (Z-CBMs), which predict concepts and labels in a fully zero-shot manner without training neural networks. Z-CBMs utilize a large-scale concept bank, which is composed of millions of vocabulary extracted from the web, to describe arbitrary input in various domains. For the input-to-concept mapping, we introduce concept retrieval, which dynamically finds input-related concepts by the cross-modal search on the concept bank. In the concept-to-label inference, we apply concept regression to select essential concepts from the retrieved concepts by sparse linear regression. Through extensive experiments, we confirm that our Z-CBMs provide interpretable and intervenable concepts without any additional training. Code will be available at this https URL. 

**Abstract (ZH)**: 零样本概念瓶颈模型（Z-CBMs）：无需训练的全零样本概念和标签预测 

---
# PixLift: Accelerating Web Browsing via AI Upscaling 

**Title (ZH)**: PixLift: 通过AI超分加速网络浏览 

**Authors**: Yonas Atinafu, Sarthak Malla, HyunSeok Daniel Jang, Nouar Aldahoul, Matteo Varvello, Yasir Zaki  

**Link**: [PDF](https://arxiv.org/pdf/2502.08995)  

**Abstract**: Accessing the internet in regions with expensive data plans and limited connectivity poses significant challenges, restricting information access and economic growth. Images, as a major contributor to webpage sizes, exacerbate this issue, despite advances in compression formats like WebP and AVIF. The continued growth of complex and curated web content, coupled with suboptimal optimization practices in many regions, has prevented meaningful reductions in web page sizes. This paper introduces PixLift, a novel solution to reduce webpage sizes by downscaling their images during transmission and leveraging AI models on user devices to upscale them. By trading computational resources for bandwidth, PixLift enables more affordable and inclusive web access. We address key challenges, including the feasibility of scaled image requests on popular websites, the implementation of PixLift as a browser extension, and its impact on user experience. Through the analysis of 71.4k webpages, evaluations of three mainstream upscaling models, and a user study, we demonstrate PixLift's ability to significantly reduce data usage without compromising image quality, fostering a more equitable internet. 

**Abstract (ZH)**: 在数据计划昂贵且连接有限的地区访问互联网带来显著挑战，限制了信息获取和经济发展。由于图像对网页大小的显著贡献，尽管有WebP和AVIF等压缩格式的进步，这一问题并未得到缓解。随着复杂且精心策划的网页内容的持续增长，以及许多地区优化实践的不足，网页大小的有效减小仍未实现。本文介绍了一种名为PixLift的新型解决方案，该解决方案在传输过程中通过缩小图像尺寸，并在用户设备上利用AI模型放大图像来减少网页大小。通过在计算资源与带宽之间进行权衡，PixLift使得更加经济实惠和包容性的网页访问成为可能。我们解决了包括流行网站上缩放图像请求的可行性、PixLift作为浏览器扩展的实现以及对用户体验的影响等关键挑战。通过对71400个网页的分析、三种主流放大模型的评估以及用户研究，我们证明PixLift能够在不牺牲图像质量的情况下显著减少数据使用量，促进一个更加公平的互联网。 

---
# RLSA-PFL: Robust Lightweight Secure Aggregation with Model Inconsistency Detection in Privacy-Preserving Federated Learning 

**Title (ZH)**: RLSA-PFL：隐私保护联邦学习中的鲁棒轻量级安全聚合及模型不一致性检测 

**Authors**: Nazatul H. Sultan, Yan Bo, Yansong Gao, Seyit Camtepe, Arash Mahboubi, Hang Thanh Bui, Aufeef Chauhan, Hamed Aboutorab, Michael Bewong, Praveen Gauravaram, Rafiqul Islam, Sharif Abuadbba  

**Link**: [PDF](https://arxiv.org/pdf/2502.08989)  

**Abstract**: Federated Learning (FL) allows users to collaboratively train a global machine learning model by sharing local model only, without exposing their private data to a central server. This distributed learning is particularly appealing in scenarios where data privacy is crucial, and it has garnered substantial attention from both industry and academia. However, studies have revealed privacy vulnerabilities in FL, where adversaries can potentially infer sensitive information from the shared model parameters. In this paper, we present an efficient masking-based secure aggregation scheme utilizing lightweight cryptographic primitives to mitigate privacy risks. Our scheme offers several advantages over existing methods. First, it requires only a single setup phase for the entire FL training session, significantly reducing communication overhead. Second, it minimizes user-side overhead by eliminating the need for user-to-user interactions, utilizing an intermediate server layer and a lightweight key negotiation method. Third, the scheme is highly resilient to user dropouts, and the users can join at any FL round. Fourth, it can detect and defend against malicious server activities, including recently discovered model inconsistency attacks. Finally, our scheme ensures security in both semi-honest and malicious settings. We provide security analysis to formally prove the robustness of our approach. Furthermore, we implemented an end-to-end prototype of our scheme. We conducted comprehensive experiments and comparisons, which show that it outperforms existing solutions in terms of communication and computation overhead, functionality, and security. 

**Abstract (ZH)**: 联邦学习（FL）允许用户通过共享本地模型而非暴露其私有数据给中央服务器来协作训练全局机器学习模型。这种分布式学习特别适用于数据隐私至关重要的场景，已在产业界和学术界引起了广泛关注。然而，研究表明在联邦学习中存在隐私漏洞，攻击者可以潜在地从共享的模型参数中推断出敏感信息。在本文中，我们提出了一种基于高效掩码的安全聚合方案，利用轻量级的加密基本原语来缓解隐私风险。该方案相对于现有方法具有多种优势。首先，它只需一个设置阶段即可用于整个FL训练会话，显著减少了通信开销。其次，通过利用中间服务器层和轻量级密钥协商方法，该方案减少了用户端的开销，消除了用户间的交互需求。第三，该方案对用户脱机具有高度鲁棒性，用户可以在任意FL轮次加入。第四，该方案能够检测并防御恶意服务器活动，包括近期发现的模型不一致性攻击。最后，该方案在半诚实和恶意设置下均能确保安全性。我们提供了安全分析以正式证明该方法的鲁棒性。此外，我们实现了一个完整的端到端原型系统，并进行了全面的实验与对比，结果显示该方案在通信和计算开销、功能性和安全性方面均优于现有解决方案。 

---
# Few is More: Task-Efficient Skill-Discovery for Multi-Task Offline Multi-Agent Reinforcement Learning 

**Title (ZH)**: 少数胜多多：多任务离线多智能体 reinforcement learning 的任务高效技能发现 

**Authors**: Xun Wang, Zhuoran Li, Hai Zhong, Longbo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.08985)  

**Abstract**: As a data-driven approach, offline MARL learns superior policies solely from offline datasets, ideal for domains rich in historical data but with high interaction costs and risks. However, most existing methods are task-specific, requiring retraining for new tasks, leading to redundancy and inefficiency. To address this issue, in this paper, we propose a task-efficient multi-task offline MARL algorithm, Skill-Discovery Conservative Q-Learning (SD-CQL). Unlike existing offline skill-discovery methods, SD-CQL discovers skills by reconstructing the next observation. It then evaluates fixed and variable actions separately and employs behavior-regularized conservative Q-learning to execute the optimal action for each skill. This approach eliminates the need for local-global alignment and enables strong multi-task generalization from limited small-scale source tasks. Substantial experiments on StarCraftII demonstrates the superior generalization performance and task-efficiency of SD-CQL. It achieves the best performance on $\textbf{10}$ out of $14$ task sets, with up to $\textbf{65%}$ improvement on individual task sets, and is within $4\%$ of the best baseline on the remaining four. 

**Abstract (ZH)**: 基于数据驱动的 Offline MARL：Skill-Discovery Conservative Q-Learning (SD-CQL) 的任务高效多任务学习算法 

---
# Biologically Plausible Brain Graph Transformer 

**Title (ZH)**: 生物可实现的大脑图形变换器 

**Authors**: Ciyuan Peng, Yuelong Huang, Qichao Dong, Shuo Yu, Feng Xia, Chengqi Zhang, Yaochu Jin  

**Link**: [PDF](https://arxiv.org/pdf/2502.08958)  

**Abstract**: State-of-the-art brain graph analysis methods fail to fully encode the small-world architecture of brain graphs (accompanied by the presence of hubs and functional modules), and therefore lack biological plausibility to some extent. This limitation hinders their ability to accurately represent the brain's structural and functional properties, thereby restricting the effectiveness of machine learning models in tasks such as brain disorder detection. In this work, we propose a novel Biologically Plausible Brain Graph Transformer (BioBGT) that encodes the small-world architecture inherent in brain graphs. Specifically, we present a network entanglement-based node importance encoding technique that captures the structural importance of nodes in global information propagation during brain graph communication, highlighting the biological properties of the brain structure. Furthermore, we introduce a functional module-aware self-attention to preserve the functional segregation and integration characteristics of brain graphs in the learned representations. Experimental results on three benchmark datasets demonstrate that BioBGT outperforms state-of-the-art models, enhancing biologically plausible brain graph representations for various brain graph analytical tasks 

**Abstract (ZH)**: 最先进的脑图分析方法无法完全编码脑图中的小世界架构（伴随hub和功能模块的存在），因此在某种程度上缺乏生物合理性。这一局限性妨碍了它们准确表示脑的结构和功能特性的能力，进而限制了机器学习模型在诸如脑障碍检测等任务中的效果。在这项工作中，我们提出了一种新的生物合理脑图变换器（BioBGT），以编码脑图中固有的小世界架构。具体来说，我们提出了一种基于网络纠缠的节点重要性编码技术，该技术在脑图通信过程中捕捉全局信息传播中的结构重要性，突显脑结构的生物特性。此外，我们引入了一种功能模块意识的自注意力机制，以在学习表示中保留脑图的功能分离和整合特性。在三个基准数据集上的实验结果表明，BioBGT优于最先进的模型，增强了各种脑图分析任务中脑图的生物合理性表示。 

---
# Analysis of Off-Policy $n$-Step TD-Learning with Linear Function Approximation 

**Title (ZH)**: Off-Policy $n$-Step TD-Learning with Linear Function Approximation的分析 

**Authors**: Han-Dong Lim, Donghwan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.08941)  

**Abstract**: This paper analyzes multi-step temporal difference (TD)-learning algorithms within the ``deadly triad'' scenario, characterized by linear function approximation, off-policy learning, and bootstrapping. In particular, we prove that $n$-step TD-learning algorithms converge to a solution as the sampling horizon $n$ increases sufficiently. The paper is divided into two parts. In the first part, we comprehensively examine the fundamental properties of their model-based deterministic counterparts, including projected value iteration, gradient descent algorithms, which can be viewed as prototype deterministic algorithms whose analysis plays a pivotal role in understanding and developing their model-free reinforcement learning counterparts. In particular, we prove that these algorithms converge to meaningful solutions when $n$ is sufficiently large. Based on these findings, in the second part, two $n$-step TD-learning algorithms are proposed and analyzed, which can be seen as the model-free reinforcement learning counterparts of the model-based deterministic algorithms. 

**Abstract (ZH)**: 本文分析了在“致命三角”情景下多步时差（TD）学习算法，该情景包括线性函数近似、离策学习和自举。特别地，我们证明了当采样窗口$n$足够大时，$n$步TD学习算法收敛到解。本文分为两部分。在第一部分中，我们全面探讨了其模型驱动的确定性对应算法的基本性质，包括投影值迭代和梯度下降算法，这些算法可以被视为原型确定性算法，其分析在理解并开发其无模型自由强化学习对应算法中起着关键作用。特别地，我们证明当$n$足够大时，这些算法收敛到有意义的解。基于这些发现，第二部分提出了两种$n$步TD学习算法，并进行了分析，这些算法可以被视为模型驱动的确定性算法的无模型自由强化学习对应算法。 

---
# TokenSynth: A Token-based Neural Synthesizer for Instrument Cloning and Text-to-Instrument 

**Title (ZH)**: TokenSynth：一种基于令牌的神经合成器，用于乐器克隆和文本到乐器的转换 

**Authors**: Kyungsu Kim, Junghyun Koo, Sungho Lee, Haesun Joung, Kyogu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.08939)  

**Abstract**: Recent advancements in neural audio codecs have enabled the use of tokenized audio representations in various audio generation tasks, such as text-to-speech, text-to-audio, and text-to-music generation. Leveraging this approach, we propose TokenSynth, a novel neural synthesizer that utilizes a decoder-only transformer to generate desired audio tokens from MIDI tokens and CLAP (Contrastive Language-Audio Pretraining) embedding, which has timbre-related information. Our model is capable of performing instrument cloning, text-to-instrument synthesis, and text-guided timbre manipulation without any fine-tuning. This flexibility enables diverse sound design and intuitive timbre control. We evaluated the quality of the synthesized audio, the timbral similarity between synthesized and target audio/text, and synthesis accuracy (i.e., how accurately it follows the input MIDI) using objective measures. TokenSynth demonstrates the potential of leveraging advanced neural audio codecs and transformers to create powerful and versatile neural synthesizers. The source code, model weights, and audio demos are available at: this https URL 

**Abstract (ZH)**: 近期神经音频编码器的进步使得在语音合成、文本转音频和文本转音乐生成等各种音频生成任务中使用标记化音频表示成为可能。基于这一方法，我们提出了一种新型神经合成器TokenSynth，它利用解码器变压器从MIDI标记和CLAP（对比语言-音频预训练）嵌入生成所需音频标记，CLAP嵌入包含音色相关信息。我们的模型可以在无需微调的情况下执行乐器克隆、文本转乐器合成和文本引导的音色操控。这种灵活性使得声音设计具有多样性和直观的音色控制。我们使用客观指标评估了合成音频的质量、合成音频与目标音频/文本的音色相似性以及合成精度（即其对输入MIDI的遵循程度）。TokenSynth展示了利用高级神经音频编码器和变压器创建强大且多功能神经合成器的潜力。源代码、模型权重和音频示例可在以下链接获取：this https URL。 

---
# Diffusion Models Through a Global Lens: Are They Culturally Inclusive? 

**Title (ZH)**: 从全球视角看扩散模型：它们具备文化包容性吗？ 

**Authors**: Zahra Bayramli, Ayhan Suleymanzade, Na Min An, Huzama Ahmad, Eunsu Kim, Junyeong Park, James Thorne, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2502.08914)  

**Abstract**: Text-to-image diffusion models have recently enabled the creation of visually compelling, detailed images from textual prompts. However, their ability to accurately represent various cultural nuances remains an open question. In our work, we introduce CultDiff benchmark, evaluating state-of-the-art diffusion models whether they can generate culturally specific images spanning ten countries. We show that these models often fail to generate cultural artifacts in architecture, clothing, and food, especially for underrepresented country regions, by conducting a fine-grained analysis of different similarity aspects, revealing significant disparities in cultural relevance, description fidelity, and realism compared to real-world reference images. With the collected human evaluations, we develop a neural-based image-image similarity metric, namely, CultDiff-S, to predict human judgment on real and generated images with cultural artifacts. Our work highlights the need for more inclusive generative AI systems and equitable dataset representation over a wide range of cultures. 

**Abstract (ZH)**: 基于文本的图像diffusion模型 recently 已经使从文本提示生成视觉上引人注目且详细的图像成为可能。然而，它们在准确表现各种文化细微差别方面的能力仍然是一个开放问题。在我们的工作中，我们引入了CultDiff基准，评估最先进的diffusion模型是否能够生成跨越十个不同国家的文化特定图像。通过细粒度分析不同相似性方面，我们展示了这些模型在建筑、服饰和食物等文化标志物生成方面常表现出色，尤其是在代表性不足的国家地区，与真实世界参考图像相比，在文化相关性、描述准确性和逼真度方面存在显著差异。借助收集的人类评估，我们开发了一种基于神经网络的图像-图像相似性度量，即CultDiff-S，以预测具有文化标志物的现实和生成图像的人类判断。我们的工作强调了需要更包容的生成AI系统以及在多种文化范围内的公平数据集表示。 

---
# Learning in Strategic Queuing Systems with Small Buffers 

**Title (ZH)**: 战略排队系统中小缓冲区的学习机制 

**Authors**: Ariana Abel, Yoav Kolumbus, Jeronimo Martin Duque, Eva Tardos  

**Link**: [PDF](https://arxiv.org/pdf/2502.08898)  

**Abstract**: Routers in networking use simple learning algorithms to find the best way to deliver packets to their desired destination. This simple, myopic and distributed decision system makes large queuing systems simple to operate, but at the same time, the system needs more capacity than would be required if all traffic were centrally coordinated. In a recent paper, Gaitonde and Tardos (EC 2020 and JACM 2023) initiate the study of such systems, modeling them as an infinitely repeated game in which routers compete for servers and the system maintains a state (number of packets held by each queue) resulting from outcomes of previous rounds. Queues get to send a packet at each step to one of the servers, and servers attempt to process only one of the arriving packets, modeling routers. However, their model assumes that servers have no buffers at all, so queues have to resend all packets that were not served successfully. They show that, even with hugely increased server capacity relative to what is needed in the centrally-coordinated case, ensuring that the system is stable requires using timestamps and priority for older packets. We consider a system with two important changes, which make the model more realistic: first we add a very small buffer to each server, allowing it to hold on to a single packet to be served later (even if it fails to serve it); and second, we do not require timestamps or priority for older packets. Our main result is to show that when queues are learning, a small constant factor increase in server capacity, compared to what would be needed if centrally coordinating, suffices to keep the system stable, even if servers select randomly among packets arriving simultaneously. This work contributes to the growing literature on the impact of selfish learning in systems with carryover effects between rounds: when outcomes in the present round affect the game in the future. 

**Abstract (ZH)**: 网络中路由器使用简单的学习算法来找到将数据包交付至目标的最佳路径。这种简单、短视且分布式的决策系统使得大型队列系统易于操作，但同时系统需要比全部流量集中协调所需更多的容量。近期，Gaitonde和Tardos（EC 2020 和 JACM 2023）研究了此类系统，将其建模为一个无限重复的游戏，在游戏中路由器竞争服务器，并保持由先前各轮次结果构成的状态（即每个队列持有的数据包数量）。队列在每一步可以将一个数据包发送到服务器中的一个，而服务器试图处理到达的所有数据包中的一个，模拟路由器。然而，他们的模型假设服务器完全没有缓冲区，因此队列必须重新发送所有未成功处理的数据包。他们证明，即使相对于集中协调所需的情况，服务器容量大幅增加，确保系统稳定也需要使用时间戳和老数据包的优先级。我们考虑了一个具有两个重要变化的系统，这些变化使模型更具有现实性：首先，我们为每个服务器添加了一个很小的缓冲区，使其能够保留一个数据包以待以后处理（即使未能立即处理它）；其次，我们不需要为老数据包使用时间戳或优先级。我们的主要结果是证明，在路由器学习的情况下，与集中协调所需相比，服务器容量的很小常数倍增加足以保持系统的稳定，即使服务器在同时到达多个数据包时随机选择处理一个数据包。该工作为正的影响自我学习在具有跨轮次影响的系统中的研究做出了贡献：当前轮次的结果会以某种方式影响未来游戏的结果。 

---
# Generative AI for Internet of Things Security: Challenges and Opportunities 

**Title (ZH)**: 生成式人工智能在物联网安全中的挑战与机遇 

**Authors**: Yan Lin Aung, Ivan Christian, Ye Dong, Xiaodong Ye, Sudipta Chattopadhyay, Jianying Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.08886)  

**Abstract**: As Generative AI (GenAI) continues to gain prominence and utility across various sectors, their integration into the realm of Internet of Things (IoT) security evolves rapidly. This work delves into an examination of the state-of-the-art literature and practical applications on how GenAI could improve and be applied in the security landscape of IoT. Our investigation aims to map the current state of GenAI implementation within IoT security, exploring their potential to fortify security measures further. Through the compilation, synthesis, and analysis of the latest advancements in GenAI technologies applied to IoT, this paper not only introduces fresh insights into the field, but also lays the groundwork for future research directions. It explains the prevailing challenges within IoT security, discusses the effectiveness of GenAI in addressing these issues, and identifies significant research gaps through MITRE Mitigations. Accompanied with three case studies, we provide a comprehensive overview of the progress and future prospects of GenAI applications in IoT security. This study serves as a foundational resource to improve IoT security through the innovative application of GenAI, thus contributing to the broader discourse on IoT security and technology integration. 

**Abstract (ZH)**: 随着生成式人工智能（GenAI）在各个领域的 prominence 和实用性不断增强，其在物联网（IoT）安全领域的集成应用迅速发展。本文探讨了当前最先进的文献和实际应用，分析GenAI如何改善并应用于物联网安全领域。我们的研究旨在梳理GenAI在物联网安全中的现状，探索其进一步强化安全措施的潜力。通过对应用于物联网的最新GenAI技术的总结、综合和分析，本文不仅为该领域提供了新的见解，还为未来的研究方向奠定了基础。本文解释了物联网安全领域现存的主要挑战，讨论了GenAI在应对这些问题上的有效性，并通过MITRE Mitigations 确定了重要的研究空白。伴随三个案例研究，本文提供了GenAI在物联网安全中进展与未来展望的全面概述。这项研究作为改进物联网安全的基础资源，通过创新应用GenAI而贡献于更广泛的物联网安全和技术集成讨论。 

---
# Harnessing Vision Models for Time Series Analysis: A Survey 

**Title (ZH)**: 基于视觉模型的时间序列分析：一个综述 

**Authors**: Jingchao Ni, Ziming Zhao, ChengAo Shen, Hanghang Tong, Dongjin Song, Wei Cheng, Dongsheng Luo, Haifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.08869)  

**Abstract**: Time series analysis has witnessed the inspiring development from traditional autoregressive models, deep learning models, to recent Transformers and Large Language Models (LLMs). Efforts in leveraging vision models for time series analysis have also been made along the way but are less visible to the community due to the predominant research on sequence modeling in this domain. However, the discrepancy between continuous time series and the discrete token space of LLMs, and the challenges in explicitly modeling the correlations of variates in multivariate time series have shifted some research attentions to the equally successful Large Vision Models (LVMs) and Vision Language Models (VLMs). To fill the blank in the existing literature, this survey discusses the advantages of vision models over LLMs in time series analysis. It provides a comprehensive and in-depth overview of the existing methods, with dual views of detailed taxonomy that answer the key research questions including how to encode time series as images and how to model the imaged time series for various tasks. Additionally, we address the challenges in the pre- and post-processing steps involved in this framework and outline future directions to further advance time series analysis with vision models. 

**Abstract (ZH)**: 时间序列分析从传统的自回归模型、深度学习模型发展到近期的变换器和大规模语言模型(LLMs)取得了令人振奋的进步。沿途也进行了利用视觉模型进行时间序列分析的努力，但由于该领域主要集中在序列建模研究，这些努力在社区中的影响力较小。然而，连续时间序列与LLMs的离散词空间之间的差异，以及多变量时间序列中变量间相关性的显式建模挑战，已部分研究方向转向同样成功的大型视觉模型(LVMs)和视觉语言模型(VLMs)。为了填补现有文献的空白，本文综述了视觉模型在时间序列分析中的优势。文章提供了现有方法的全面而深入的概述，并从详细分类学视角解答关键研究问题，包括如何将时间序列编码为图像，以及如何对图像化的时间序列进行建模以应用于各种任务。此外，本文还讨论了该框架中预处理和后处理步骤中的挑战，并勾勒出未来的研究方向以进一步推动利用视觉模型的时间序列分析。 

---
# A Reversible Solver for Diffusion SDEs 

**Title (ZH)**: 可逆求解器for扩散SDEs 

**Authors**: Zander W. Blasingame, Chen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.08834)  

**Abstract**: Diffusion models have quickly become the state-of-the-art for generation tasks across many different data modalities. An important ability of diffusion models is the ability to encode samples from the data distribution back into the sampling prior distribution. This is useful for performing alterations to real data samples along with guided generation via the continuous adjoint equations. We propose an algebraically reversible solver for diffusion SDEs that can exactly invert real data samples into the prior distribution. 

**Abstract (ZH)**: 扩散模型已成为多种数据模态生成任务的前沿方法。扩散模型的一个重要能力是能够将数据分布的样本重新编码回采样先验分布。这有助于通过对真实数据样本进行修改以及通过连续伴随方程进行导向生成。我们提出了一种代数可逆求解器，它可以精确地将真实数据样本反转到先验分布。 

---
# A Survey on Data-Centric AI: Tabular Learning from Reinforcement Learning and Generative AI Perspective 

**Title (ZH)**: 数据为中心的AI综述：从强化学习和生成AI视角的表型学习 

**Authors**: Wangyang Ying, Cong Wei, Nanxu Gong, Xinyuan Wang, Haoyue Bai, Arun Vignesh Malarkkan, Sixun Dong, Dongjie Wang, Denghui Zhang, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2502.08828)  

**Abstract**: Tabular data is one of the most widely used data formats across various domains such as bioinformatics, healthcare, and marketing. As artificial intelligence moves towards a data-centric perspective, improving data quality is essential for enhancing model performance in tabular data-driven applications. This survey focuses on data-driven tabular data optimization, specifically exploring reinforcement learning (RL) and generative approaches for feature selection and feature generation as fundamental techniques for refining data spaces. Feature selection aims to identify and retain the most informative attributes, while feature generation constructs new features to better capture complex data patterns. We systematically review existing generative methods for tabular data engineering, analyzing their latest advancements, real-world applications, and respective strengths and limitations. This survey emphasizes how RL-based and generative techniques contribute to the automation and intelligence of feature engineering. Finally, we summarize the existing challenges and discuss future research directions, aiming to provide insights that drive continued innovation in this field. 

**Abstract (ZH)**: 表格数据是生物信息学、医疗保健和营销等领域广泛使用的数据格式之一。随着人工智能向以数据为中心的视角发展，提高表格数据的质量对于增强表数据驱动应用中的模型性能至关重要。本文综述了数据驱动的表格数据优化，特别探讨了基于强化学习（RL）和生成方法的特征选择和特征生成作为精炼数据空间的基本技术。特征选择旨在识别并保留最富有信息性的属性，而特征生成则构建新的特征以更好地捕捉复杂的数据模式。本文系统地回顾了现有的生成方法在表格数据工程中的应用，分析了它们的最新进展、实际应用及其各自的优缺点。本文强调了基于RL和生成技术如何促进特征工程的自动化和智能化。最后，本文总结了现有挑战并讨论了未来的研究方向，旨在提供推动该领域持续创新的见解。 

---
# DejAIvu: Identifying and Explaining AI Art on the Web in Real-Time with Saliency Maps 

**Title (ZH)**: DejAIyu：实时利用显著图识别和解释网络上的AI艺术 

**Authors**: Jocelyn Dzuong  

**Link**: [PDF](https://arxiv.org/pdf/2502.08821)  

**Abstract**: The recent surge in advanced generative models, such as diffusion models and generative adversarial networks (GANs), has led to an alarming rise in AI-generated images across various domains on the web. While such technologies offer benefits such as democratizing artistic creation, they also pose challenges in misinformation, digital forgery, and authenticity verification. Additionally, the uncredited use of AI-generated images in media and marketing has sparked significant backlash from online communities. In response to this, we introduce DejAIvu, a Chrome Web extension that combines real-time AI-generated image detection with saliency-based explainability while users browse the web. Using an ONNX-optimized deep learning model, DejAIvu automatically analyzes images on websites such as Google Images, identifies AI-generated content using model inference, and overlays a saliency heatmap to highlight AI-related artifacts. Our approach integrates efficient in-browser inference, gradient-based saliency analysis, and a seamless user experience, ensuring that AI detection is both transparent and interpretable. We also evaluate DejAIvu across multiple pretrained architectures and benchmark datasets, demonstrating high accuracy and low latency, making it a practical and deployable tool for enhancing AI image accountability. The code for this system can be found at this https URL. 

**Abstract (ZH)**: 最近生成模型（如扩散模型和生成对抗网络GANs）的蓬勃发展导致网络上各类领域出现了惊人的AI生成图像的增长。虽然这些技术提供了如艺术创作民主化等优势，但也带来了虚假信息、数字伪造和真实性验证等方面的挑战。此外，AI生成图像在媒体和营销中的未经许可使用引发了在线社区的强烈反对。针对以上问题，我们提出了DejAIvu，这是一个集成了实时AI生成图像检测和基于显著性的解释性的Chrome Web插件。通过优化后的ONNX深度学习模型，DejAIvu自动分析Google Images等网站上的图像，利用模型推断识别AI生成内容，并叠加显著性热图以突出AI相关特征。我们的方法结合了高效的浏览器内推理、基于梯度的显著性分析以及无缝用户体验，确保AI检测既透明又可解释。我们还在多个预训练架构和基准数据集上评估了DejAIvu，结果显示其具有高准确性和低延迟，使其成为增强AI图像问责制的实用和可部署工具。该系统的代码可在此处找到：this https URL。 

---
# CLOVER: A Test Case Generation Benchmark with Coverage, Long-Context, and Verification 

**Title (ZH)**: CLOVER：一个基于覆盖、长上下文和验证的测试用例生成基准 

**Authors**: Jiacheng Xu, Bo Pang, Jin Qu, Hiroaki Hayashi, Caiming Xiong, Yingbo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.08806)  

**Abstract**: Software testing is a critical aspect of software development, yet generating test cases remains a routine task for engineers. This paper presents a benchmark, CLOVER, to evaluate models' capabilities in generating and completing test cases under specific conditions. Spanning from simple assertion completions to writing test cases that cover specific code blocks across multiple files, these tasks are based on 12 python repositories, analyzing 845 problems with context lengths ranging from 4k to 128k tokens. Utilizing code testing frameworks, we propose a method to construct retrieval contexts using coverage information. While models exhibit comparable performance with short contexts, notable differences emerge with 16k contexts. Notably, models like GPT-4o and Claude 3.5 can effectively leverage relevant snippets; however, all models score below 35\% on the complex Task III, even with the oracle context provided, underscoring the benchmark's significance and the potential for model improvement. The benchmark is containerized for code execution across tasks, and we will release the code, data, and construction methodologies. 

**Abstract (ZH)**: 软件测试是软件开发中的一个关键方面，但生成测试案例仍然是工程师的一项常规任务。本文提出一个基准CLOVER，用于评估模型在特定条件下生成和完成测试案例的能力。这些任务从简单的断言补齐扩展到跨越多个文件覆盖特定代码块的测试案例编写，基于12个Python仓库，分析845个问题，上下文长度范围从4k到128k个 tokens。利用代码测试框架，我们提出了一种使用覆盖率信息构建检索上下文的方法。尽管在短上下文中模型表现相当，但在16k上下文中，模型之间的差异变得明显。值得注意的是，GPT-4o和Claude 3.5等模型可以有效利用相关片段，但所有模型在复杂任务III中得分均低于35%，即使提供 oracle 上下文也是如此，突显了该基准的重要性和模型改进的潜力。该基准已容器化以支持跨任务的代码执行，并将发布代码、数据和构建方法。 

---
# Auction Design using Value Prediction with Hallucinations 

**Title (ZH)**: 基于生成式价值预测的拍卖设计 

**Authors**: Ilan Lobel, Humberto Moreira, Omar Mouchtaki  

**Link**: [PDF](https://arxiv.org/pdf/2502.08792)  

**Abstract**: We investigate a Bayesian mechanism design problem where a seller seeks to maximize revenue by selling an indivisible good to one of n buyers, incorporating potentially unreliable predictions (signals) of buyers' private values derived from a machine learning model. We propose a framework where these signals are sometimes reflective of buyers' true valuations but other times are hallucinations, which are uncorrelated with the buyers' true valuations. Our main contribution is a characterization of the optimal auction under this framework. Our characterization establishes a near-decomposition of how to treat types above and below the signal. For the one buyer case, the seller's optimal strategy is to post one of three fairly intuitive prices depending on the signal, which we call the "ignore", "follow" and "cap" actions. 

**Abstract (ZH)**: 我们探讨了一个贝叶斯机制设计问题，卖方旨在通过向n位买家之一出售一件不可分割的商品来最大化收入，同时考虑来自机器学习模型的买家私有价值潜在不可靠的信号（信号）。我们提出了一种框架，在这种框架下，这些信号有时反映了买家的真实估值，但有时则是与买家的真实估值不相关的幻觉。我们的主要贡献是描述了在这种框架下最优拍卖的设计。我们的描述确立了一种近似的分解，说明了如何处理信号之上和之下的类型。对于单一买家的情况，卖方的最优策略是根据信号的提示之一，采取“忽略”、“跟随”和“限制”三种相当直观的价格策略之一。 

---
# Scalable Discrete Diffusion Samplers: Combinatorial Optimization and Statistical Physics 

**Title (ZH)**: 可扩展的离散扩散采样器：组合优化与统计物理 

**Authors**: Sebastian Sanokowski, Wilhelm Berghammer, Martin Ennemoser, Haoyu Peter Wang, Sepp Hochreiter, Sebastian Lehner  

**Link**: [PDF](https://arxiv.org/pdf/2502.08696)  

**Abstract**: Learning to sample from complex unnormalized distributions over discrete domains emerged as a promising research direction with applications in statistical physics, variational inference, and combinatorial optimization. Recent work has demonstrated the potential of diffusion models in this domain. However, existing methods face limitations in memory scaling and thus the number of attainable diffusion steps since they require backpropagation through the entire generative process. To overcome these limitations we introduce two novel training methods for discrete diffusion samplers, one grounded in the policy gradient theorem and the other one leveraging Self-Normalized Neural Importance Sampling (SN-NIS). These methods yield memory-efficient training and achieve state-of-the-art results in unsupervised combinatorial optimization. Numerous scientific applications additionally require the ability of unbiased sampling. We introduce adaptations of SN-NIS and Neural Markov Chain Monte Carlo that enable for the first time the application of discrete diffusion models to this problem. We validate our methods on Ising model benchmarks and find that they outperform popular autoregressive approaches. Our work opens new avenues for applying diffusion models to a wide range of scientific applications in discrete domains that were hitherto restricted to exact likelihood models. 

**Abstract (ZH)**: 从离散域中的复杂非规范化分布中学习采样：一种在统计物理、变分推理和组合优化领域的有前途的研究方向。现有的研究表明扩散模型在这一领域具有潜在的应用价值。然而，现有方法在内存扩展方面存在局限性，从而限制了可以实现的扩散步数，因为它们需要通过对整个生成过程的反向传播。为克服这些限制，我们提出了两种针对离散扩散采样器的新训练方法，一种基于策略梯度定理，另一种利用自我归一化神经重要性采样（SN-NIS）。这些方法实现了内存高效的训练并且在无监督组合优化中达到了最先进的结果。此外，许多科学应用还需要无偏差采样的能力。我们对SN-NIS和神经马尔可夫链蒙特卡罗方法进行了适应性改进，这使得离散扩散模型能够首次应用于这一问题。我们通过Ising模型基准实验验证了我们的方法，并发现它们优于流行的自回归方法。我们的工作为将扩散模型应用于广泛的科学应用开辟了新的途径，这些应用此前仅限于精确似然模型。 

---
# EEG Artifact Detection and Correction with Deep Autoencoders 

**Title (ZH)**: 基于深度自编码器的脑电 Artefact 检测与校正 

**Authors**: David Aquilué-Llorens, Aureli Soria-Frisch  

**Link**: [PDF](https://arxiv.org/pdf/2502.08686)  

**Abstract**: EEG signals convey important information about brain activity both in healthy and pathological conditions. However, they are inherently noisy, which poses significant challenges for accurate analysis and interpretation. Traditional EEG artifact removal methods, while effective, often require extensive expert intervention. This study presents LSTEEG, a novel LSTM-based autoencoder designed for the detection and correction of artifacts in EEG signals. Leveraging deep learning, particularly LSTM layers, LSTEEG captures non-linear dependencies in sequential EEG data. LSTEEG demonstrates superior performance in both artifact detection and correction tasks compared to other state-of-the-art convolutional autoencoders. Our methodology enhances the interpretability and utility of the autoencoder's latent space, enabling data-driven automated artefact removal in EEG its application in downstream tasks. This research advances the field of efficient and accurate multi-channel EEG preprocessing, and promotes the implementation and usage of automated EEG analysis pipelines for brain health applications. 

**Abstract (ZH)**: EEG信号传递着关于大脑活动的重要信息，无论是健康状态还是病理状态。然而，EEG信号固有的噪声特性给其准确分析和解释带来了重大挑战。传统EEG伪迹去除方法虽有效，但往往需要大量专家干预。本研究提出了一种新的基于LSTM的自动编码器LSTEEG，用于EEG信号中的伪迹检测和校正。利用深度学习，特别是LSTM层，LSTEEG捕捉到了序列EEG数据中的非线性依赖关系。相较于其他最先进的卷积自动编码器，LSTEEG在伪迹检测和校正任务中表现出优越的性能。我们的方法增强了自动编码器潜在空间的可解释性和实用性，使其在EEG下游任务中的数据驱动的自动伪迹去除成为可能。本研究推进了高效准确的多通道EEG预处理技术的发展，并促进了脑健康应用中的自动化EEG分析流水线的实施和使用。 

---
# Beyond Models! Explainable Data Valuation and Metric Adaption for Recommendation 

**Title (ZH)**: 超越模型！可解释的数据估值与度量适应性推荐 

**Authors**: Renqi Jia, Xiaokun Zhang, Bowei He, Qiannan Zhu, Weitao Xu, Jiehao Chen, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.08685)  

**Abstract**: User behavior records serve as the foundation for recommender systems. While the behavior data exhibits ease of acquisition, it often suffers from varying quality. Current methods employ data valuation to discern high-quality data from low-quality data. However, they tend to employ black-box design, lacking transparency and interpretability. Besides, they are typically tailored to specific evaluation metrics, leading to limited generality across various tasks. To overcome these issues, we propose an explainable and versatile framework DVR which can enhance the efficiency of data utilization tailored to any requirements of the model architectures and evaluation metrics. For explainable data valuation, a data valuator is presented to evaluate the data quality via calculating its Shapley value from the game-theoretic perspective, ensuring robust mathematical properties and reliability. In order to accommodate various evaluation metrics, including differentiable and non-differentiable ones, a metric adapter is devised based on reinforcement learning, where a metric is treated as the reinforcement reward that guides model optimization. Extensive experiments conducted on various benchmarks verify that our framework can improve the performance of current recommendation algorithms on various metrics including ranking accuracy, diversity, and fairness. Specifically, our framework achieves up to 34.7\% improvements over existing methods in terms of representative NDCG metric. The code is available at this https URL. 

**Abstract (ZH)**: 用户行为记录是推荐系统的基础。虽然行为数据易于获取，但其质量参差不齐。现有方法通过数据估值来区分高质量数据和低质量数据，但这些方法往往采用黑盒设计，缺乏透明性和可解释性。此外，它们通常针对特定的评估指标进行优化，导致在不同任务上的通用性有限。为解决这些问题，我们提出了一种可解释且通用的框架DVR，该框架能够根据模型架构和评估指标的不同需求提高数据利用效率。为了实现可解释的数据估值，我们给出了一个数据估值器，通过博弈论视角计算Shapley值来评估数据质量，确保其稳健的数学性质和可靠性。为了适应各种评估指标，包括可微和不可微的指标，我们基于强化学习设计了指标适配器，其中评估指标被视为引导模型优化的强化奖励。在多种基准上的广泛实验验证了我们的框架可以提高各种指标（包括排名准确性、多样性和公平性）下的推荐算法性能。具体而言，与现有方法相比，在代表性NDCG指标上，我们的框架实现了最多34.7%的性能提升。代码可在下方链接获取。 

---
# Self-Evaluation for Job-Shop Scheduling 

**Title (ZH)**: 作业车间调度的自我评估方法 

**Authors**: Imanol Echeverria, Maialen Murua, Roberto Santana  

**Link**: [PDF](https://arxiv.org/pdf/2502.08684)  

**Abstract**: Combinatorial optimization problems, such as scheduling and route planning, are crucial in various industries but are computationally intractable due to their NP-hard nature. Neural Combinatorial Optimization methods leverage machine learning to address these challenges but often depend on sequential decision-making, which is prone to error accumulation as small mistakes propagate throughout the process. Inspired by self-evaluation techniques in Large Language Models, we propose a novel framework that generates and evaluates subsets of assignments, moving beyond traditional stepwise approaches. Applied to the Job-Shop Scheduling Problem, our method integrates a heterogeneous graph neural network with a Transformer to build a policy model and a self-evaluation function. Experimental validation on challenging, well-known benchmarks demonstrates the effectiveness of our approach, surpassing state-of-the-art methods. 

**Abstract (ZH)**: 组合最优化问题，如调度和路线规划，在各个行业中至关重要，但由于其NP难的本质，这些问题是计算上难以处理的。神经组合最优化方法利用机器学习来应对这些挑战，但通常依赖于顺序决策过程，这种过程容易累积错误，因为小错误会沿过程传播。受到大型语言模型中自我评估技术的启发，我们提出了一种新颖的框架，该框架生成并评估作业分配的子集，超越了传统的逐步方法。应用于作业车间调度问题，我们的方法结合了异质图神经网络和Transformer来构建策略模型和自我评估函数。在具有挑战性的知名基准上的实验验证证明了我们方法的有效性，超越了现有最先进的方法。 

---
# On the Role of Pre-trained Embeddings in Binary Code Analysis 

**Title (ZH)**: 预训练嵌入在二进制代码分析中的作用 

**Authors**: Alwin Maier, Felix Weissberg, Konrad Rieck  

**Link**: [PDF](https://arxiv.org/pdf/2502.08682)  

**Abstract**: Deep learning has enabled remarkable progress in binary code analysis. In particular, pre-trained embeddings of assembly code have become a gold standard for solving analysis tasks, such as measuring code similarity or recognizing functions. These embeddings are capable of learning a vector representation from unlabeled code. In contrast to natural language processing, however, label information is not scarce for many tasks in binary code analysis. For example, labeled training data for function boundaries, optimization levels, and argument types can be easily derived from debug information provided by a compiler. Consequently, the main motivation of embeddings does not transfer directly to binary code analysis.
In this paper, we explore the role of pre-trained embeddings from a critical perspective. To this end, we systematically evaluate recent embeddings for assembly code on five downstream tasks using a corpus of 1.2 million functions from the Debian distribution. We observe that several embeddings perform similarly when sufficient labeled data is available, and that differences reported in prior work are hardly noticeable. Surprisingly, we find that end-to-end learning without pre-training performs best on average, which calls into question the need for specialized embeddings. By varying the amount of labeled data, we eventually derive guidelines for when embeddings offer advantages and when end-to-end learning is preferable for binary code analysis. 

**Abstract (ZH)**: 深度学习在二进制代码分析中的应用已经取得了显著进展。特别是，预训练的汇编代码嵌入已成为解决代码相似性测量和识别功能等分析任务的标准方法。这些嵌入能够从未标注的代码中学习向量表示。然而，与自然语言处理不同，在二进制代码分析的许多任务中，标签信息并不稀缺。例如，从编译器提供的调试信息可以轻松地为函数边界、优化级别和参数类型等任务获取标注训练数据。因此，嵌入的主要动机并不直接适用于二进制代码分析。

在这篇论文中，我们从批判性视角探讨了预训练嵌入的作用。为此，我们使用Debian分发中的120万函数语料库系统性地评估了近期的汇编代码嵌入在五个下游任务上的表现。我们发现，当有足够的标注数据时，几种嵌入的表现相似，且先前报道的差异几乎不可察觉。令人惊讶的是，我们发现端到端学习而无需预训练在平均情况下表现最佳，这引发了对专门嵌入需求的质疑。通过改变标注数据的数量，我们最终得出了关于何时嵌入有利、何时端到端学习更适用于二进制代码分析的指南。 

---
# Deep Learning-Driven Malware Classification with API Call Sequence Analysis and Concept Drift Handling 

**Title (ZH)**: 基于API调用序列分析和概念漂移处理的深度学习驱动恶意软件分类 

**Authors**: Bishwajit Prasad Gond, Durga Prasad Mohapatra  

**Link**: [PDF](https://arxiv.org/pdf/2502.08679)  

**Abstract**: Malware classification in dynamic environments presents a significant challenge due to concept drift, where the statistical properties of malware data evolve over time, complicating detection efforts. To address this issue, we propose a deep learning framework enhanced with a genetic algorithm to improve malware classification accuracy and adaptability. Our approach incorporates mutation operations and fitness score evaluations within genetic algorithms to continuously refine the deep learning model, ensuring robustness against evolving malware threats. Experimental results demonstrate that this hybrid method significantly enhances classification performance and adaptability, outperforming traditional static models. Our proposed approach offers a promising solution for real-time malware classification in ever-changing cybersecurity landscapes. 

**Abstract (ZH)**: 动态环境中恶意软件分类由于概念漂移问题构成了显著挑战，其中恶意软件数据的统计特性随时间演化，增加了检测难度。为解决该问题，我们提出一种结合遗传算法的深度学习框架，以提高恶意软件分类准确性和适应性。该方法在遗传算法中引入变异操作和适应度评分评估，持续优化深度学习模型，确保其对不断演变的恶意软件威胁具有鲁棒性。实验结果表明，此混合方法显著提升了分类性能和适应性，超越了传统的静态模型。我们提出的方法为动态变化的网络环境中实时恶意软件分类提供了前景广阔的解决方案。 

---
