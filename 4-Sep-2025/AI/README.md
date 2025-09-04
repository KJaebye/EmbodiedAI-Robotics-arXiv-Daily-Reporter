# sam-llm: interpretable lane change trajectoryprediction via parametric finetuning 

**Title (ZH)**: SAM-LLM: 可解释的变道轨迹预测通过参数微调 

**Authors**: Zhuo Cao, Yunxiao Shi, Min Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.03462)  

**Abstract**: This work introduces SAM-LLM, a novel hybrid architecture that bridges the gap between the contextual reasoning of Large Language Models (LLMs) and the physical precision of kinematic lane change models for autonomous driving. The system is designed for interpretable lane change trajectory prediction by finetuning an LLM to output the core physical parameters of a trajectory model instead of raw coordinates. For lane-keeping scenarios, the model predicts discrete coordinates, but for lane change maneuvers, it generates the parameters for an enhanced Sinusoidal Acceleration Model (SAM), including lateral displacement, maneuver duration, initial lateral velocity, and longitudinal velocity change. This parametric approach yields a complete, continuous, and physically plausible trajectory model that is inherently interpretable and computationally efficient, achieving an 80% reduction in output size compared to coordinate-based methods. The SAM-LLM achieves a state-of-the-art overall intention prediction accuracy of 98.73%, demonstrating performance equivalent to traditional LLM predictors while offering significant advantages in explainability and resource efficiency. 

**Abstract (ZH)**: SAM-LLM：一种新型混合架构，将大型语言模型的上下文推理能力与运动学换道模型的物理精度相结合，用于自动驾驶的可解释换道轨迹预测 

---
# ANNIE: Be Careful of Your Robots 

**Title (ZH)**: ANNIE: 注意你的机器人 

**Authors**: Yiyang Huang, Zixuan Wang, Zishen Wan, Yapeng Tian, Haobo Xu, Yinhe Han, Yiming Gan  

**Link**: [PDF](https://arxiv.org/pdf/2509.03383)  

**Abstract**: The integration of vision-language-action (VLA) models into embodied AI (EAI) robots is rapidly advancing their ability to perform complex, long-horizon tasks in humancentric environments. However, EAI systems introduce critical security risks: a compromised VLA model can directly translate adversarial perturbations on sensory input into unsafe physical actions. Traditional safety definitions and methodologies from the machine learning community are no longer sufficient. EAI systems raise new questions, such as what constitutes safety, how to measure it, and how to design effective attack and defense mechanisms in physically grounded, interactive settings. In this work, we present the first systematic study of adversarial safety attacks on embodied AI systems, grounded in ISO standards for human-robot interactions. We (1) formalize a principled taxonomy of safety violations (critical, dangerous, risky) based on physical constraints such as separation distance, velocity, and collision boundaries; (2) introduce ANNIEBench, a benchmark of nine safety-critical scenarios with 2,400 video-action sequences for evaluating embodied safety; and (3) ANNIE-Attack, a task-aware adversarial framework with an attack leader model that decomposes long-horizon goals into frame-level perturbations. Our evaluation across representative EAI models shows attack success rates exceeding 50% across all safety categories. We further demonstrate sparse and adaptive attack strategies and validate the real-world impact through physical robot experiments. These results expose a previously underexplored but highly consequential attack surface in embodied AI systems, highlighting the urgent need for security-driven defenses in the physical AI era. Code is available at this https URL. 

**Abstract (ZH)**: 视觉-语言-动作模型在有体人工智能机器人中的集成正迅速提升其在以人为中心环境中执行复杂长期任务的能力。然而，有体人工智能系统引入了关键的安全风险：受到攻击的视觉-语言-动作模型可以直接将感官输入的对抗性扰动转化为不安全的物理动作。传统来自机器学习社区的安全定义和方法已不再足够。有体人工智能系统提出了新的问题，如何定义安全性、如何测量它以及在物理接地的交互式环境中如何设计有效的攻击和防御机制。在本文中，我们基于ISO人机交互标准，首次系统地研究了针对有体人工智能系统的对抗安全性攻击。我们（1）基于物理约束（如分离距离、速度和碰撞边界）提出了一种原则性的安全违规分类（关键、危险、风险）；（2）引入了ANNIEBench安全关键场景基准，包含2400个视频动作序列，用于评估有体安全性；（3）提出了ANNIE-Attack任务感知对抗框架，其中攻击领导者模型将长期目标分解为帧级扰动。我们在代表性有体人工智能模型上的评估结果显示，所有安全类别的攻击成功率超过50%。我们进一步展示了稀疏和自适应攻击策略，并通过物理机器人实验验证了其实际影响。这些结果揭示了有体人工智能系统中一个之前未得到充分探索但极为重要的攻击面，突显了物理人工智能时代迫切需要安全驱动的防御措施。代码可在以下链接获取。 

---
# Situating AI Agents in their World: Aspective Agentic AI for Dynamic Partially Observable Information Systems 

**Title (ZH)**: 将AI代理置于其环境中：面向动态部分可观测信息系统的方面式代理AI 

**Authors**: Peter J. Bentley, Soo Ling Lim, Fuyuki Ishikawa  

**Link**: [PDF](https://arxiv.org/pdf/2509.03380)  

**Abstract**: Agentic LLM AI agents are often little more than autonomous chatbots: actors following scripts, often controlled by an unreliable director. This work introduces a bottom-up framework that situates AI agents in their environment, with all behaviors triggered by changes in their environments. It introduces the notion of aspects, similar to the idea of umwelt, where sets of agents perceive their environment differently to each other, enabling clearer control of information. We provide an illustrative implementation and show that compared to a typical architecture, which leaks up to 83% of the time, aspective agentic AI enables zero information leakage. We anticipate that this concept of specialist agents working efficiently in their own information niches can provide improvements to both security and efficiency. 

**Abstract (ZH)**: 基于环境触发的代理AI框架：实现零信息泄露 

---
# Language Models Do Not Follow Occam's Razor: A Benchmark for Inductive and Abductive Reasoning 

**Title (ZH)**: 语言模型并不遵循奥卡姆剃刀原则：一种归纳和 abduction 推理评估基准 

**Authors**: Yunxin Sun, Abulhair Saparov  

**Link**: [PDF](https://arxiv.org/pdf/2509.03345)  

**Abstract**: Reasoning is a core capability in artificial intelligence systems, for which large language models (LLMs) have recently shown remarkable progress. However, most work focuses exclusively on deductive reasoning, which is problematic since other types of reasoning are also essential in solving real-world problems, and they are less explored. This work focuses on evaluating LLMs' inductive and abductive reasoning capabilities. We introduce a programmable and synthetic dataset, InAbHyD (pronounced in-a-bid), where each reasoning example consists of an incomplete world model and a set of observations. The task for the intelligent agent is to produce hypotheses to explain observations under the incomplete world model to solve each reasoning example. We propose a new metric to evaluate the quality of hypotheses based on Occam's Razor. We evaluate and analyze some state-of-the-art LLMs. Our analysis shows that LLMs can perform inductive and abductive reasoning in simple scenarios, but struggle with complex world models and producing high-quality hypotheses, even with popular reasoning-enhancing techniques such as in-context learning and RLVR. 

**Abstract (ZH)**: 基于归纳和 abduction 推理能力评价大型语言模型 

---
# app.build: A Production Framework for Scaling Agentic Prompt-to-App Generation with Environment Scaffolding 

**Title (ZH)**: app.build：一种基于环境支撑的扩展性代理提示到应用生成的生产框架 

**Authors**: Evgenii Kniazev, Arseny Kravchenko, Igor Rekun, James Broadhead, Nikita Shamgunov, Pranav Sah, Pratik Nichite, Ivan Yamshchikov  

**Link**: [PDF](https://arxiv.org/pdf/2509.03310)  

**Abstract**: We present this http URL (this https URL), an open-source framework that improves LLM-based application generation through systematic validation and structured environments. Our approach combines multi-layered validation pipelines, stack-specific orchestration, and model-agnostic architecture, implemented across three reference stacks. Through evaluation on 30 generation tasks, we demonstrate that comprehensive validation achieves 73.3% viability rate with 30% reaching perfect quality scores, while open-weights models achieve 80.8% of closed-model performance when provided structured environments. The open-source framework has been adopted by the community, with over 3,000 applications generated to date. This work demonstrates that scaling reliable AI agents requires scaling environments, not just models -- providing empirical insights and complete reference implementations for production-oriented agent systems. 

**Abstract (ZH)**: 我们提出这个网址（此网址），一个开源框架，通过系统验证和结构化环境提高基于LLM的应用生成。我们的方法结合了多层验证管道、栈特定协调和模型无关架构，并在三个参考栈中实施。通过30项生成任务的评估，我们证明了全面验证实现了73.3%的可行性率，其中30%达到完美质量评分，而提供结构化环境的开放权重模型达到了封闭模型80.8%的性能。该开源框架已被社区采纳，目前已生成超过3000个应用。这项工作表明，扩展可靠的AI代理需要扩展环境，而不仅仅是模型——提供了面向生产的代理系统的基础实证见解和完整参考实现。 

---
# Accountability Framework for Healthcare AI Systems: Towards Joint Accountability in Decision Making 

**Title (ZH)**: 医疗健康领域AI系统问责框架：走向决策中的共同问责 

**Authors**: Prachi Bagave, Marcus Westberg, Marijn Janssen, Aaron Yi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.03286)  

**Abstract**: AI is transforming the healthcare domain and is increasingly helping practitioners to make health-related decisions. Therefore, accountability becomes a crucial concern for critical AI-driven decisions. Although regulatory bodies, such as the EU commission, provide guidelines, they are highlevel and focus on the ''what'' that should be done and less on the ''how'', creating a knowledge gap for actors. Through an extensive analysis, we found that the term accountability is perceived and dealt with in many different ways, depending on the actor's expertise and domain of work. With increasing concerns about AI accountability issues and the ambiguity around this term, this paper bridges the gap between the ''what'' and ''how'' of AI accountability, specifically for AI systems in healthcare. We do this by analysing the concept of accountability, formulating an accountability framework, and providing a three-tier structure for handling various accountability mechanisms. Our accountability framework positions the regulations of healthcare AI systems and the mechanisms adopted by the actors under a consistent accountability regime. Moreover, the three-tier structure guides the actors of the healthcare AI system to categorise the mechanisms based on their conduct. Through our framework, we advocate that decision-making in healthcare AI holds shared dependencies, where accountability should be dealt with jointly and should foster collaborations. We highlight the role of explainability in instigating communication and information sharing between the actors to further facilitate the collaborative process. 

**Abstract (ZH)**: AI在医疗健康领域的应用正在不断改变医疗领域，越来越多地帮助医疗从业者做出健康相关的决策。因此，对于关键的AI驱动决策，问责制成为了一个重要的关注点。尽管欧盟委员会等监管机构提供了指导方针，但这些指导方针往往是高层次的，更多关注“应该做什么”，而对“如何做”则关注较少，造成了行动者之间的知识空白。通过广泛分析，我们发现问责制这一概念在不同行动者之间被理解和处理的方式多种多样，这取决于他们的专业知识和工作领域。鉴于对AI问责制问题日益增长的担忧以及此概念的模糊性，本文旨在弥合“应该做什么”与“如何做”的问责制差距，特别是针对医疗健康领域的AI系统。我们通过分析问责制概念、构建问责制框架，并提供一种多层次结构来处理各种问责机制，来实现这一目标。我们的问责制框架将医疗AI系统的法规与行动者所采用的机制置于一个一致的问责体系之下。此外，三层结构指导医疗AI系统的行动者根据其行为对机制进行分类。通过我们的框架，我们提倡医疗AI决策应共享依赖关系，问责制应当共同处理，并促进合作。我们强调可解释性在促进行动者之间的沟通和信息共享方面的作用，进一步促进协作过程。 

---
# Uncertainty-driven Adaptive Exploration 

**Title (ZH)**: 不确定性驱动的自适应探索 

**Authors**: Leonidas Bakopoulos, Georgios Chalkiadakis  

**Link**: [PDF](https://arxiv.org/pdf/2509.03219)  

**Abstract**: Adaptive exploration methods propose ways to learn complex policies via alternating between exploration and exploitation. An important question for such methods is to determine the appropriate moment to switch between exploration and exploitation and vice versa. This is critical in domains that require the learning of long and complex sequences of actions. In this work, we present a generic adaptive exploration framework that employs uncertainty to address this important issue in a principled manner. Our framework includes previous adaptive exploration approaches as special cases. Moreover, we can incorporate in our framework any uncertainty-measuring mechanism of choice, for instance mechanisms used in intrinsic motivation or epistemic uncertainty-based exploration methods. We experimentally demonstrate that our framework gives rise to adaptive exploration strategies that outperform standard ones across several MuJoCo environments. 

**Abstract (ZH)**: 自适应探索方法通过在探索和利用之间交替来学习复杂策略。这类方法的一个重要问题是确定在何时切换探索和利用以及反之亦然的适当时刻。在需要学习长且复杂的动作序列的领域中，这一点至关重要。在这项工作中，我们提出了一种基于不确定性的一般自适应探索框架，以基本原则的方式解决这一重要问题。我们的框架包含了先前的自适应探索方法作为特殊情况。此外，我们可以将任何自选的不确定性测量机制纳入我们的框架，例如内在动机或基于认识不确定性探索方法中使用的机制。实验结果表明，我们的框架产生了在多个MuJoCo环境中优于标准方法的自适应探索策略。 

---
# Learning General Policies From Examples 

**Title (ZH)**: 从示例中学习通用策略 

**Authors**: Blai Bonet, Hector Geffner  

**Link**: [PDF](https://arxiv.org/pdf/2509.02794)  

**Abstract**: Combinatorial methods for learning general policies that solve large collections of planning problems have been recently developed. One of their strengths, in relation to deep learning approaches, is that the resulting policies can be understood and shown to be correct. A weakness is that the methods do not scale up and learn only from small training instances and feature pools that contain a few hundreds of states and features at most. In this work, we propose a new symbolic method for learning policies based on the generalization of sampled plans that ensures structural termination and hence acyclicity. The proposed learning approach is not based on SAT/ASP, as previous symbolic methods, but on a hitting set algorithm that can effectively handle problems with millions of states, and pools with hundreds of thousands of features. The formal properties of the approach are analyzed, and its scalability is tested on a number of benchmarks. 

**Abstract (ZH)**: 基于采样计划泛化的符号学习方法：确保结构终止并有效处理大规模状态和特征池的问题 

---
# Key Principles in Cross-Domain Hyper-Heuristic Performance 

**Title (ZH)**: 跨领域超元启发式性能的关键原则 

**Authors**: Václav Sobotka, Lucas Kletzander, Nysret Musliu, Hana Rudová  

**Link**: [PDF](https://arxiv.org/pdf/2509.02782)  

**Abstract**: Cross-domain selection hyper-heuristics aim to distill decades of research on problem-specific heuristic search algorithms into adaptable general-purpose search strategies. In this respect, existing selection hyper-heuristics primarily focus on an adaptive selection of low-level heuristics (LLHs) from a predefined set. In contrast, we concentrate on the composition of this set and its strategic transformations. We systematically analyze transformations based on three key principles: solution acceptance, LLH repetitions, and perturbation intensity, i.e., the proportion of a solution affected by a perturbative LLH. We demonstrate the raw effects of our transformations on a trivial unbiased random selection mechanism. With an appropriately constructed transformation, this trivial method outperforms all available state-of-the-art hyper-heuristics on three challenging real-world domains and finds 11 new best-known solutions. The same method is competitive with the winner of the CHeSC competition, commonly used as the standard cross-domain benchmark. Moreover, we accompany several recent hyper-heuristics with such strategic transformations. Using this approach, we outperform the current state-of-the-art methods on both the CHeSC benchmark and real-world domains while often simplifying their designs. 

**Abstract (ZH)**: 跨域选择超启发式方法旨在将数十年针对具体问题启发式搜索算法的研究成果提炼为可适应的一般搜索策略。在这方面，现有的选择超启发式方法主要集中在从预定义集合中适配选择低级启发式算法（LLHs）。相比之下，我们专注于集合的构成及其战略变换。我们系统地分析了基于三个关键原则的变换：解接受、LLH重复以及扰动强度，即扰动性LLH影响解的比例。我们展示了我们的变换对一个简单无偏随机选择机制的基础效果。通过适当构造的变换，该简单方法在三个具有挑战性的实际领域中均优于所有现有的最先进的超启发式方法，并找到11个新的最优解。该方法在CHeSC比赛的冠军和常用的跨域基准测试中具有竞争力。此外，我们还为几种近期的超启发式方法配备了此类战略变换。采用这种方法，我们在CHeSC基准和实际领域中均优于当前最先进的方法，同时往往简化了它们的设计。 

---
# Plan Verification for LLM-Based Embodied Task Completion Agents 

**Title (ZH)**: 基于大型语言模型的具身任务完成代理的计划验证 

**Authors**: Ananth Hariharan, Vardhan Dongre, Dilek Hakkani-Tür, Gokhan Tur  

**Link**: [PDF](https://arxiv.org/pdf/2509.02761)  

**Abstract**: Large language model (LLM) based task plans and corresponding human demonstrations for embodied AI may be noisy, with unnecessary actions, redundant navigation, and logical errors that reduce policy quality. We propose an iterative verification framework in which a Judge LLM critiques action sequences and a Planner LLM applies the revisions, yielding progressively cleaner and more spatially coherent trajectories. Unlike rule-based approaches, our method relies on natural language prompting, enabling broad generalization across error types including irrelevant actions, contradictions, and missing steps. On a set of manually annotated actions from the TEACh embodied AI dataset, our framework achieves up to 90% recall and 100% precision across four state-of-the-art LLMs (GPT o4-mini, DeepSeek-R1, Gemini 2.5, LLaMA 4 Scout). The refinement loop converges quickly, with 96.5% of sequences requiring at most three iterations, while improving both temporal efficiency and spatial action organization. Crucially, the method preserves human error-recovery patterns rather than collapsing them, supporting future work on robust corrective behavior. By establishing plan verification as a reliable LLM capability for spatial planning and action refinement, we provide a scalable path to higher-quality training data for imitation learning in embodied AI. 

**Abstract (ZH)**: 基于大型语言模型（LLM）的任务计划及其对应的embodied AI人类示范可能存在杂音，包含不必要的动作、冗余导航和逻辑错误，这些都会降低策略质量。我们提出了一个迭代验证框架，在该框架中，一个Judge LLM批评动作序列，一个Planner LLM应用修改，从而生成逐步 cleaner 和更空间一致的轨迹。与基于规则的方法不同，我们的方法依赖于自然语言提示，能够广泛泛化多种错误类型，包括无关动作、自相矛盾和缺失步骤。在TEACh embodied AI数据集中手动标注的动作集上，我们的框架在四个最先进的LLM（GPT o4-mini、DeepSeek-R1、Gemini 2.5、LLaMA 4 Scout）上实现了高达90%的召回率和100%的精度。精炼循环迅速收敛，96.5%的序列最多需要三次迭代，同时提高时间和空间行动组织的效率。尤为重要的是，该方法保留了人类的错误恢复模式，而不是消除它们，从而支持未来工作中的鲁棒纠正行为。通过将计划验证确立为空间规划和动作精炼的可靠LLM能力，我们为embodied AI中的模仿学习提供了可扩展的高质量训练数据路径。 

---
# Do LLM Modules Generalize? A Study on Motion Generation for Autonomous Driving 

**Title (ZH)**: LLM模块的泛化能力：自动驾驶中运动生成的研究 

**Authors**: Mingyi Wang, Jingke Wang, Tengju Ye, Junbo Chen, Kaicheng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.02754)  

**Abstract**: Recent breakthroughs in large language models (LLMs) have not only advanced natural language processing but also inspired their application in domains with structurally similar problems--most notably, autonomous driving motion generation. Both domains involve autoregressive sequence modeling, token-based representations, and context-aware decision making, making the transfer of LLM components a natural and increasingly common practice. However, despite promising early attempts, a systematic understanding of which LLM modules are truly transferable remains lacking. In this paper, we present a comprehensive evaluation of five key LLM modules--tokenizer design, positional embedding, pre-training paradigms, post-training strategies, and test-time computation--within the context of motion generation for autonomous driving. Through extensive experiments on the Waymo Sim Agents benchmark, we demonstrate that, when appropriately adapted, these modules can significantly improve performance for autonomous driving motion generation. In addition, we identify which techniques can be effectively transferred, analyze the potential reasons for the failure of others, and discuss the specific adaptations needed for autonomous driving scenarios. We evaluate our method on the Sim Agents task and achieve competitive results. 

**Abstract (ZH)**: 近期大型语言模型（LLMs）的突破不仅推动了自然语言处理的发展，还启发了其在具有类似结构问题领域中的应用——最显著的是自主驾驶运动生成。这两个领域都涉及自回归序列建模、基于令牌的表示和上下文感知决策，使得LLM组件的转移成为自然且越来越常见的实践。然而，尽管早期尝试令人鼓舞，但缺乏对哪些LLM模块真正可转移的系统理解。在本文中，我们针对自主驾驶运动生成场景，全面评估了五个关键LLM模块——分词器设计、位置嵌入、预训练范式、后训练策略和测试时计算。通过在Waymo Sim Agents基准测试上的广泛实验，我们证明了在适当适应后，这些模块可以显著提高自主驾驶运动生成的性能。此外，我们确定了哪些技术可以有效转移，分析了其他技术失败的潜在原因，并讨论了自主驾驶场景所需的具体适应方法。我们在Sim Agents任务上评估了该方法，并取得了具有竞争力的结果。 

---
# Deep Research is the New Analytics System: Towards Building the Runtime for AI-Driven Analytics 

**Title (ZH)**: 深度研究是新的分析系统：朝着构建以AI驱动的分析运行时系统努力 

**Authors**: Matthew Russo, Tim Kraska  

**Link**: [PDF](https://arxiv.org/pdf/2509.02751)  

**Abstract**: With advances in large language models (LLMs), researchers are creating new systems that can perform AI-driven analytics over large unstructured datasets. Recent work has explored executing such analytics queries using semantic operators -- a declarative set of AI-powered data transformations with natural language specifications. However, even when optimized, these operators can be expensive to execute on millions of records and their iterator execution semantics make them ill-suited for interactive data analytics tasks. In another line of work, Deep Research systems have demonstrated an ability to answer natural language question(s) over large datasets. These systems use one or more LLM agent(s) to plan their execution, process the dataset(s), and iteratively refine their answer. However, these systems do not explicitly optimize their query plans which can lead to poor plan execution. In order for AI-driven analytics to excel, we need a runtime which combines the optimized execution of semantic operators with the flexibility and more dynamic execution of Deep Research systems. As a first step towards this vision, we build a prototype which enables Deep Research agents to write and execute optimized semantic operator programs. We evaluate our prototype and demonstrate that it can outperform a handcrafted semantic operator program and open Deep Research systems on two basic queries. Compared to a standard open Deep Research agent, our prototype achieves up to 1.95x better F1-score. Furthermore, even if we give the agent access to semantic operators as tools, our prototype still achieves cost and runtime savings of up to 76.8% and 72.7% thanks to its optimized execution. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）的发展，研究人员正在创建新的系统，可以对大规模非结构化数据集执行AI驱动的分析。近期的工作探索了使用语义操作符执行此类分析查询的方法——这是一种具有自然语言规范的声明式AI驱动数据转换集合。然而，即使经过优化，这些操作符在执行数百万条记录时仍然可能代价高昂，并且其迭代执行语义使它们不适用于交互式数据分析任务。另一项工作中，Deep Research系统展示了在大型数据集上回答自然语言问题的能力。这些系统使用一个或多个LLM代理规划执行、处理数据集，并迭代精炼其答案。然而，这些系统并没有明确优化其查询计划，可能导致计划执行效果不佳。为了使AI驱动的分析取得卓越成效，我们需要一个能够结合语义操作符优化执行与Deep Research系统灵活和动态执行特点的运行时系统。为了朝这一愿景迈出第一步，我们构建了一个原型系统，使Deep Research代理能够编写和执行优化的语义操作符程序。我们评估了该原型系统，结果显示它能够优于手工编写的语义操作符程序，并在两种基本查询上开放Deep Research系统。与标准的开放Deep Research代理相比，我们的原型系统在F1分数上提高了至多1.95倍。即使我们赋予代理访问语义操作符作为工具，我们的原型系统仍通过优化执行实现了高达76.8%的成本节约和72.7%的运行时间节约。 

---
# Planning with Reasoning using Vision Language World Model 

**Title (ZH)**: 基于视觉语言世界模型的推理规划 

**Authors**: Delong Chen, Theo Moutakanni, Willy Chung, Yejin Bang, Ziwei Ji, Allen Bolourchi, Pascale Fung  

**Link**: [PDF](https://arxiv.org/pdf/2509.02722)  

**Abstract**: Effective planning requires strong world models, but high-level world models that can understand and reason about actions with semantic and temporal abstraction remain largely underdeveloped. We introduce the Vision Language World Model (VLWM), a foundation model trained for language-based world modeling on natural videos. Given visual observations, the VLWM first infers the overall goal achievements then predicts a trajectory composed of interleaved actions and world state changes. Those targets are extracted by iterative LLM Self-Refine conditioned on compressed future observations represented by Tree of Captions. The VLWM learns both an action policy and a dynamics model, which respectively facilitates reactive system-1 plan decoding and reflective system-2 planning via cost minimization. The cost evaluates the semantic distance between the hypothetical future states given by VLWM roll-outs and the expected goal state, and is measured by a critic model that we trained in a self-supervised manner. The VLWM achieves state-of-the-art Visual Planning for Assistance (VPA) performance on both benchmark evaluations and our proposed PlannerArena human evaluations, where system-2 improves the Elo score by +27% upon system-1. The VLWM models also outperforms strong VLM baselines on RoboVQA and WorldPrediction benchmark. 

**Abstract (ZH)**: 有效的规划需要强大的世界模型，但具有语义和时间抽象能力的高层世界模型仍处于初步发展阶段。我们引入了Vision Language World Model (VLWM)，这是一种为基于语言的世界建模而在自然视频上训练的基石模型。给定视觉观察，VLWM首先推断整体目标的实现情况，然后预测由交错的动作和世界状态变化组成的轨迹。这些目标通过迭代的LLM Self-Refine在以树状描述未来观察压缩表示为基础的条件下提取。VLWM学习了动作策略和动力学模型，分别促进了反应性系统-1计划解码和反思性系统-2规划通过成本最小化。成本评估的是由VLWM模拟给出的假设未来状态与预期目标状态之间的语义距离，由我们在自我监督方式下训练的批评模型进行度量。VLWM在基准评估和我们提出的PlannerArena人类评估中均实现了最先进的视觉辅助规划（VPA）性能，在PlannerArena人类评估中，系统-2将Elo得分提高了27%。VLWM模型还优于强大的VLM基线模型，在RoboVQA和WorldPrediction基准测试中表现出色。 

---
# The Future of Artificial Intelligence and the Mathematical and Physical Sciences (AI+MPS) 

**Title (ZH)**: 人工智能的未来与数学及物理科学（AI+MPS） 

**Authors**: Andrew Ferguson, Marisa LaFleur, Lars Ruthotto, Jesse Thaler, Yuan-Sen Ting, Pratyush Tiwary, Soledad Villar, E. Paulo Alves, Jeremy Avigad, Simon Billinge, Camille Bilodeau, Keith Brown, Emmanuel Candes, Arghya Chattopadhyay, Bingqing Cheng, Jonathan Clausen, Connor Coley, Andrew Connolly, Fred Daum, Sijia Dong, Chrisy Xiyu Du, Cora Dvorkin, Cristiano Fanelli, Eric B. Ford, Luis Manuel Frutos, Nicolás García Trillos, Cecilia Garraffo, Robert Ghrist, Rafael Gomez-Bombarelli, Gianluca Guadagni, Sreelekha Guggilam, Sergei Gukov, Juan B. Gutiérrez, Salman Habib, Johannes Hachmann, Boris Hanin, Philip Harris, Murray Holland, Elizabeth Holm, Hsin-Yuan Huang, Shih-Chieh Hsu, Nick Jackson, Olexandr Isayev, Heng Ji, Aggelos Katsaggelos, Jeremy Kepner, Yannis Kevrekidis, Michelle Kuchera, J. Nathan Kutz, Branislava Lalic, Ann Lee, Matt LeBlanc, Josiah Lim, Rebecca Lindsey, Yongmin Liu, Peter Y. Lu, Sudhir Malik, Vuk Mandic, Vidya Manian, Emeka P. Mazi, Pankaj Mehta, Peter Melchior, Brice Ménard, Jennifer Ngadiuba, Stella Offner, Elsa Olivetti, Shyue Ping Ong, Christopher Rackauckas, Philippe Rigollet, Chad Risko, Philip Romero, Grant Rotskoff, Brett Savoie, Uros Seljak, David Shih, Gary Shiu, Dima Shlyakhtenko, Eva Silverstein, Taylor Sparks, Thomas Strohmer, Christopher Stubbs, Stephen Thomas, Suriyanarayanan Vaikuntanathan, Rene Vidal, Francisco Villaescusa-Navarro, Gregory Voth, Benjamin Wandelt, Rachel Ward, Melanie Weber, Risa Wechsler, Stephen Whitelam, Olaf Wiest, Mike Williams, Zhuoran Yang, Yaroslava G. Yingling, Bin Yu, Shuwen Yue, Ann Zabludoff, Huimin Zhao, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02661)  

**Abstract**: This community paper developed out of the NSF Workshop on the Future of Artificial Intelligence (AI) and the Mathematical and Physics Sciences (MPS), which was held in March 2025 with the goal of understanding how the MPS domains (Astronomy, Chemistry, Materials Research, Mathematical Sciences, and Physics) can best capitalize on, and contribute to, the future of AI. We present here a summary and snapshot of the MPS community's perspective, as of Spring/Summer 2025, in a rapidly developing field. The link between AI and MPS is becoming increasingly inextricable; now is a crucial moment to strengthen the link between AI and Science by pursuing a strategy that proactively and thoughtfully leverages the potential of AI for scientific discovery and optimizes opportunities to impact the development of AI by applying concepts from fundamental science. To achieve this, we propose activities and strategic priorities that: (1) enable AI+MPS research in both directions; (2) build up an interdisciplinary community of AI+MPS researchers; and (3) foster education and workforce development in AI for MPS researchers and students. We conclude with a summary of suggested priorities for funding agencies, educational institutions, and individual researchers to help position the MPS community to be a leader in, and take full advantage of, the transformative potential of AI+MPS. 

**Abstract (ZH)**: 本研究社区论文源自2025年3月举行的NSF人工智能（AI）与数学及物理科学（MPS）未来研讨会，旨在了解数学及物理科学领域（天文学、化学、材料研究、数学科学和物理学）如何最有效地利用AI，并为AI的未来发展做出贡献。本文提供了截至2025年春季/夏季数学及物理科学社区观点的总结和快照，展示了该领域的快速发展。AI与数学及物理科学之间的联系越来越密不可分；现在是加强AI与科学之间联系的关键时刻，通过积极和深思熟虑地利用AI的潜力促进科学发现，并通过应用基础科学的概念优化影响AI发展的机会。为此，我们提出了以下活动和战略优先事项：（1）实现双向的AI+MPS研究；（2）建立跨学科的AI+MPS研究人员社区；（3）促进AI教育和工作队伍发展，为MPS研究人员和学生。最后，我们总结了对资助机构、教育机构和个人研究人员的建议优先事项，以帮助数学及物理科学社区在AI+MPS的变革潜力方面处于领先地位并充分利用其潜力。 

---
# Can Media Act as a Soft Regulator of Safe AI Development? A Game Theoretical Analysis 

**Title (ZH)**: 媒体能在安全人工智能发展软监管中发挥作用吗？一种博弈理论分析 

**Authors**: Henrique Correia da Fonseca, António Fernandes, Zhao Song, Theodor Cimpeanu, Nataliya Balabanova, Adeela Bashir, Paolo Bova, Alessio Buscemi, Alessandro Di Stefano, Manh Hong Duong, Elias Fernandez Domingos, Ndidi Bianca Ogbo, Simon T. Powers, Daniele Proverbio, Zia Ush Shamszaman, Fernando P. Santos, Anh Han, Marcus Krellner  

**Link**: [PDF](https://arxiv.org/pdf/2509.02650)  

**Abstract**: When developers of artificial intelligence (AI) products need to decide between profit and safety for the users, they likely choose profit. Untrustworthy AI technology must come packaged with tangible negative consequences. Here, we envisage those consequences as the loss of reputation caused by media coverage of their misdeeds, disseminated to the public. We explore whether media coverage has the potential to push AI creators into the production of safe products, enabling widespread adoption of AI technology. We created artificial populations of self-interested creators and users and studied them through the lens of evolutionary game theory. Our results reveal that media is indeed able to foster cooperation between creators and users, but not always. Cooperation does not evolve if the quality of the information provided by the media is not reliable enough, or if the costs of either accessing media or ensuring safety are too high. By shaping public perception and holding developers accountable, media emerges as a powerful soft regulator -- guiding AI safety even in the absence of formal government oversight. 

**Abstract (ZH)**: 当人工智能产品的开发者在利润与用户安全之间作出选择时，他们倾向于选择利润。不可信赖的人工智能技术必须伴随着实际的负面后果。在此，我们设想这些后果是由于媒体对其不当行为的报道而造成的声誉损失，并传播给公众。我们探讨媒体覆盖是否有可能促使人工智能创造者生产安全产品，从而促进人工智能技术的广泛应用。我们构建了自私的创造者和用户的人工群体，并通过进化博弈理论的角度研究了它们。我们的结果表明，媒体确实能够促进创造者和用户之间的合作，但这并不总能实现。如果媒体提供的信息质量不够可靠，或者访问媒体或确保安全的成本过高，合作就不会演化。通过塑造公众认知并促使开发者承担责任，媒体作为一种强大的软监管工具在缺乏正式政府监管的情况下，仍然能够引导人工智能的安全发展。 

---
# Can the Waymo Open Motion Dataset Support Realistic Behavioral Modeling? A Validation Study with Naturalistic Trajectories 

**Title (ZH)**: Waymo开放运动数据集能否支持现实行为建模？基于自然轨迹的验证研究 

**Authors**: Yanlin Zhang, Sungyong Chung, Nachuan Li, Dana Monzer, Hani S. Mahmassani, Samer H. Hamdar, Alireza Talebpour  

**Link**: [PDF](https://arxiv.org/pdf/2509.03515)  

**Abstract**: The Waymo Open Motion Dataset (WOMD) has become a popular resource for data-driven modeling of autonomous vehicles (AVs) behavior. However, its validity for behavioral analysis remains uncertain due to proprietary post-processing, the absence of error quantification, and the segmentation of trajectories into 20-second clips. This study examines whether WOMD accurately captures the dynamics and interactions observed in real-world AV operations. Leveraging an independently collected naturalistic dataset from Level 4 AV operations in Phoenix, Arizona (PHX), we perform comparative analyses across three representative urban driving scenarios: discharging at signalized intersections, car-following, and lane-changing behaviors. For the discharging analysis, headways are manually extracted from aerial video to ensure negligible measurement error. For the car-following and lane-changing cases, we apply the Simulation-Extrapolation (SIMEX) method to account for empirically estimated error in the PHX data and use Dynamic Time Warping (DTW) distances to quantify behavioral differences. Results across all scenarios consistently show that behavior in PHX falls outside the behavioral envelope of WOMD. Notably, WOMD underrepresents short headways and abrupt decelerations. These findings suggest that behavioral models calibrated solely on WOMD may systematically underestimate the variability, risk, and complexity of naturalistic driving. Caution is therefore warranted when using WOMD for behavior modeling without proper validation against independently collected data. 

**Abstract (ZH)**: Waymo开放运动数据集（WOMD）在自主车辆行为建模中的适用性分析：基于亚利桑那州凤凰城（PHX）Level 4自主车辆自然驾驶数据的比较研究 

---
# LimiX: Unleashing Structured-Data Modeling Capability for Generalist Intelligence 

**Title (ZH)**: LimiX: 解锁通用智能的结构化数据建模能力 

**Authors**: Xingxuan Zhang, Gang Ren, Han Yu, Hao Yuan, Hui Wang, Jiansheng Li, Jiayun Wu, Lang Mo, Li Mao, Mingchao Hao, Ningbo Dai, Renzhe Xu, Shuyang Li, Tianyang Zhang, Yue He, Yuanrui Wang, Yunjia Zhang, Zijing Xu, Dongzhe Li, Fang Gao, Hao Zou, Jiandong Liu, Jiashuo Liu, Jiawei Xu, Kaijie Cheng, Kehan Li, Linjun Zhou, Qing Li, Shaohua Fan, Xiaoyu Lin, Xinyan Han, Xuanyue Li, Yan Lu, Yuan Xue, Yuanyuan Jiang, Zimu Wang, Zhenlei Wang, Peng Cui  

**Link**: [PDF](https://arxiv.org/pdf/2509.03505)  

**Abstract**: We argue that progress toward general intelligence requires complementary foundation models grounded in language, the physical world, and structured data. This report presents LimiX, the first installment of our large structured-data models (LDMs). LimiX treats structured data as a joint distribution over variables and missingness, thus capable of addressing a wide range of tabular tasks through query-based conditional prediction via a single model. LimiX is pretrained using masked joint-distribution modeling with an episodic, context-conditional objective, where the model predicts for query subsets conditioned on dataset-specific contexts, supporting rapid, training-free adaptation at inference. We evaluate LimiX across 10 large structured-data benchmarks with broad regimes of sample size, feature dimensionality, class number, categorical-to-numerical feature ratio, missingness, and sample-to-feature ratios. With a single model and a unified interface, LimiX consistently surpasses strong baselines including gradient-boosting trees, deep tabular networks, recent tabular foundation models, and automated ensembles, as shown in Figure 1 and Figure 2. The superiority holds across a wide range of tasks, such as classification, regression, missing value imputation, and data generation, often by substantial margins, while avoiding task-specific architectures or bespoke training per task. All LimiX models are publicly accessible under Apache 2.0. 

**Abstract (ZH)**: 进步通用智能需要语言、物理世界和结构化数据基础模型的互补融合：LimiX——大规模结构化数据模型的第一步 

---
# Warming Up for Zeroth-Order Federated Pre-Training with Low Resource Clients 

**Title (ZH)**: 零资源客户端 warming 超预训练的联邦预训练预热 

**Authors**: Gwen Legate, Irina Rish, Eugene Belilovsky  

**Link**: [PDF](https://arxiv.org/pdf/2509.03503)  

**Abstract**: Federated learning enables collaborative model training across numerous edge devices without requiring participants to share data; however, memory and communication constraints on these edge devices may preclude their participation in training. We consider a setting in which a subset of edge devices are below a critical memory or communication threshold required to conduct model updates. Under typical federated optimization algorithms, these devices are excluded from training which renders their data inaccessible and increases system induced bias. We are inspired by MeZO, a zeroth-order method used for memory-efficient fine-tuning. The increased variance inherent to zeroth-order gradient approximations has relegated previous zeroth-order optimizers exclusively to the domain of fine tuning; a limitation we seek to correct. We devise a federated, memory-efficient zeroth-order optimizer, ZOWarmUp that permits zeroth-order training from a random initialization. ZOWarmUp leverages differing client capabilities and careful variance reduction techniques to facilitate participation of under-represented, low-resource clients in model training. Like other federated zeroth-order methods, ZOWarmUp eliminates the need for edge devices to transmit their full gradients to the server and instead relies on only a small set of random seeds, rendering the up-link communication cost negligible. We present experiments using various datasets and model architectures to show that ZOWarmUp is a robust algorithm that can can be applied under a wide variety of circumstances. For systems with a high proportion of edge devices that would otherwise be excluded from training, this algorithm provides access to a greater volume and diversity of data, thus improving training outcomes. 

**Abstract (ZH)**: 联邦学习使得多个边缘设备能够在不共享数据的情况下进行协作模型训练；然而，这些边缘设备的内存和通信约束可能使其无法参与训练。我们考虑一种情况下，部分边缘设备低于进行模型更新所需的关键内存或通信阈值。在典型的联邦优化算法中，这些设备被排除在训练之外，这使得它们的数据不可访问并增加了系统引入的偏差。我们受到了MeZO的启发，这是一种用于高效微调的记忆高效零阶方法。零阶梯度近似的固有方差限制了先前的零阶优化器仅限于微调领域；这是一个我们希望纠正的局限。我们设计了一个联邦记忆高效零阶优化器ZOWarmUp，它允许从随机初始化进行零阶训练。ZOWarmUp利用客户端差异能力和精心设计的方差减少技术，促进资源不足的客户端参与模型训练。与其他联邦零阶方法类似，ZOWarmUp消除了边缘设备向服务器传输完整梯度的需求，而是依赖于少量随机种子，使得上行通信成本可以忽略不计。我们使用各种数据集和模型架构进行了实验，展示了ZOWarmUp是一个稳健的算法，可以在多种情境下应用。对于具有大量本应被排除在训练之外的边缘设备的系统，该算法提供了更大的数据量和多样性，从而提升了训练效果。 

---
# Strefer: Empowering Video LLMs with Space-Time Referring and Reasoning via Synthetic Instruction Data 

**Title (ZH)**: 时空引导：通过合成指令数据增强视频LLMs的空间-时间指示与推理 

**Authors**: Honglu Zhou, Xiangyu Peng, Shrikant Kendre, Michael S. Ryoo, Silvio Savarese, Caiming Xiong, Juan Carlos Niebles  

**Link**: [PDF](https://arxiv.org/pdf/2509.03501)  

**Abstract**: Next-generation AI companions must go beyond general video understanding to resolve spatial and temporal references in dynamic, real-world environments. Existing Video Large Language Models (Video LLMs), while capable of coarse-level comprehension, struggle with fine-grained, spatiotemporal reasoning, especially when user queries rely on time-based event references for temporal anchoring, or gestural cues for spatial anchoring to clarify object references and positions. To bridge this critical gap, we introduce Strefer, a synthetic instruction data generation framework designed to equip Video LLMs with spatiotemporal referring and reasoning capabilities. Strefer produces diverse instruction-tuning data using a data engine that pseudo-annotates temporally dense, fine-grained video metadata, capturing rich spatial and temporal information in a structured manner, including subjects, objects, their locations as masklets, and their action descriptions and timelines. Our approach enhances the ability of Video LLMs to interpret spatial and temporal references, fostering more versatile, space-time-aware reasoning essential for real-world AI companions. Without using proprietary models, costly human annotation, or the need to annotate large volumes of new videos, experimental evaluations show that models trained with data produced by Strefer outperform baselines on tasks requiring spatial and temporal disambiguation. Additionally, these models exhibit enhanced space-time-aware reasoning, establishing a new foundation for perceptually grounded, instruction-tuned Video LLMs. 

**Abstract (ZH)**: 下一代AI伴侣必须超越一般视频理解，以解决动态真实世界环境中的空间和时间引用问题。现有的视频大规模语言模型（Video LLMs）虽然具备粗略的理解能力，但在细粒度的空间-时间推理方面存在困难，尤其是当用户查询依赖基于时间的事件引用进行时间定位，或依赖手势线索进行空间定位以澄清物体引用和位置时。为弥补这一关键差距，我们引入了Strefer，一种合成指令数据生成框架，旨在为Video LLMs提供空间-时间引用和推理能力。Strefer使用数据引擎伪注释时空密集型、细粒度的视频元数据，以结构化方式捕获丰富的空间和时间信息，包括主体、对象及其作为掩码的位置、动作描述和时间线。我们的方法增强了Video LLMs对空间和时间引用的解释能力，促进了更灵活、时空感知的推理，这对于真实世界的AI伴侣至关重要。实验评估表明，使用Strefer生成的数据训练的模型在需要空间和时间消歧的任务中优于基线模型，同时这些模型还表现出增强的时空感知推理能力，为基于感知的、指令调优的Video LLMs奠定了新基础。 

---
# Real-Time Instrument Planning and Perception for Novel Measurements of Dynamic Phenomena 

**Title (ZH)**: 实时仪器规划与感知以实现动态现象的新测量 

**Authors**: Itai Zilberstein, Alberto Candela, Steve Chien  

**Link**: [PDF](https://arxiv.org/pdf/2509.03500)  

**Abstract**: Advancements in onboard computing mean remote sensing agents can employ state-of-the-art computer vision and machine learning at the edge. These capabilities can be leveraged to unlock new rare, transient, and pinpoint measurements of dynamic science phenomena. In this paper, we present an automated workflow that synthesizes the detection of these dynamic events in look-ahead satellite imagery with autonomous trajectory planning for a follow-up high-resolution sensor to obtain pinpoint measurements. We apply this workflow to the use case of observing volcanic plumes. We analyze classification approaches including traditional machine learning algorithms and convolutional neural networks. We present several trajectory planning algorithms that track the morphological features of a plume and integrate these algorithms with the classifiers. We show through simulation an order of magnitude increase in the utility return of the high-resolution instrument compared to baselines while maintaining efficient runtimes. 

**Abstract (ZH)**: 随着机载计算技术的进步，遥感代理可以利用边缘的先进计算机视觉和机器学习技术。这些能力可以被利用以解锁新的稀有、短暂和精确的动态科学现象测量。在本文中，我们提出了一种自动工作流，该工作流结合了前瞻性卫星图像中的动态事件检测与自主轨迹规划，以获取高分辨率传感器的精确测量。我们将此工作流应用于观察火山喷气流的应用场景。我们分析了包括传统机器学习算法和卷积神经网络在内的分类方法。我们介绍了几种用于跟踪喷气流形态特征的轨迹规划算法，并将这些算法与分类器集成。通过模拟，我们展示了与基准相比，高分辨率仪器的效用提高了数量级，同时保持了高效的运行时间。 

---
# On Entropy Control in LLM-RL Algorithms 

**Title (ZH)**: 在LLM-RL算法中的_entropy控制_ 

**Authors**: Han Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.03493)  

**Abstract**: For RL algorithms, appropriate entropy control is crucial to their effectiveness. To control the policy entropy, a commonly used method is entropy regularization, which is adopted in various popular RL algorithms including PPO, SAC and A3C. Although entropy regularization proves effective in robotic and games RL conventionally, studies found that it gives weak to no gains in LLM-RL training. In this work, we study the issues of entropy bonus in LLM-RL setting. Specifically, we first argue that the conventional entropy regularization suffers from the LLM's extremely large response space and the sparsity of the optimal outputs. As a remedy, we propose AEnt, an entropy control method that utilizes a new clamped entropy bonus with an automatically adjusted coefficient. The clamped entropy is evaluated with the re-normalized policy defined on certain smaller token space, which encourages exploration within a more compact response set. In addition, the algorithm automatically adjusts entropy coefficient according to the clamped entropy value, effectively controlling the entropy-induced bias while leveraging the entropy's benefits. AEnt is tested in math-reasoning tasks under different base models and datasets, and it is observed that AEnt outperforms the baselines consistently across multiple benchmarks. 

**Abstract (ZH)**: 对于RL算法，适当的熵控制对其有效性至关重要。为了控制策略熵，常用的方法是熵正则化，该方法被多种流行RL算法（包括PPO、SAC和A3C）采用。尽管在机器人和游戏RL中熵正则化证明是有效的，但在LLM-RL训练中研究表明它提供很少或几乎没有增益。在本文中，我们研究了LLM-RL设置中熵奖励的问题。具体地，我们首先指出，传统的熵正则化受到LLM极大的响应空间和最优输出稀疏性的困扰。作为补救措施，我们提出了AEnt，一种利用新的限幅熵奖励并根据限幅熵值自动调整系数的熵控制方法。限幅熵基于特定较小的标记空间重新归一化的策略进行评估，从而鼓励在更紧凑的响应集中探索。此外，该算法根据限幅熵值自动调整熵系数，有效地控制熵引起的偏差同时利用熵的好处。AEnt在不同的基础模型和数据集下的数学推理任务中进行了测试，并观察到AEnt在多个基准测试中持续优于基线方法。 

---
# SafeProtein: Red-Teaming Framework and Benchmark for Protein Foundation Models 

**Title (ZH)**: SafeProtein: 红队框架与蛋白质基础模型基准测试 

**Authors**: Jigang Fan, Zhenghong Zhou, Ruofan Jin, Le Cong, Mengdi Wang, Zaixi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.03487)  

**Abstract**: Proteins play crucial roles in almost all biological processes. The advancement of deep learning has greatly accelerated the development of protein foundation models, leading to significant successes in protein understanding and design. However, the lack of systematic red-teaming for these models has raised serious concerns about their potential misuse, such as generating proteins with biological safety risks. This paper introduces SafeProtein, the first red-teaming framework designed for protein foundation models to the best of our knowledge. SafeProtein combines multimodal prompt engineering and heuristic beam search to systematically design red-teaming methods and conduct tests on protein foundation models. We also curated SafeProtein-Bench, which includes a manually constructed red-teaming benchmark dataset and a comprehensive evaluation protocol. SafeProtein achieved continuous jailbreaks on state-of-the-art protein foundation models (up to 70% attack success rate for ESM3), revealing potential biological safety risks in current protein foundation models and providing insights for the development of robust security protection technologies for frontier models. The codes will be made publicly available at this https URL. 

**Abstract (ZH)**: 蛋白质在几乎所有的生物过程中都扮演着至关重要的角色。深度学习的进步极大地加速了蛋白质基础模型的发展，使其在蛋白质理解与设计方面取得了显著成果。然而，这些模型缺乏系统的红队测试，引发了对其潜在滥用的严重担忧，例如生成具有生物安全风险的蛋白质。本文介绍了SafeProtein，这是已知的第一个专为蛋白质基础模型设计的红队测试框架。SafeProtein结合了多模态提示工程和启发式束搜索，系统性地设计红队测试方法并对蛋白质基础模型进行测试。我们还策划了SafeProtein-Bench，其中包括一个手工构建的红队测试基准数据集和全面的评估协议。SafeProtein在最先进的蛋白质基础模型上实现了持续的突破（ESM3的攻击成功率高达70%），揭示了当前蛋白质基础模型中存在的潜在生物安全风险，并为前沿模型的稳健安全保护技术的发展提供了见解。相关代码将在以下网址公开：这个 https URL。 

---
# Robult: Leveraging Redundancy and Modality Specific Features for Robust Multimodal Learning 

**Title (ZH)**: Robult: 利用冗余性和模态特定特征进行鲁棒多模态学习 

**Authors**: Duy A. Nguyen, Abhi Kamboj, Minh N. Do  

**Link**: [PDF](https://arxiv.org/pdf/2509.03477)  

**Abstract**: Addressing missing modalities and limited labeled data is crucial for advancing robust multimodal learning. We propose Robult, a scalable framework designed to mitigate these challenges by preserving modality-specific information and leveraging redundancy through a novel information-theoretic approach. Robult optimizes two core objectives: (1) a soft Positive-Unlabeled (PU) contrastive loss that maximizes task-relevant feature alignment while effectively utilizing limited labeled data in semi-supervised settings, and (2) a latent reconstruction loss that ensures unique modality-specific information is retained. These strategies, embedded within a modular design, enhance performance across various downstream tasks and ensure resilience to incomplete modalities during inference. Experimental results across diverse datasets validate that Robult achieves superior performance over existing approaches in both semi-supervised learning and missing modality contexts. Furthermore, its lightweight design promotes scalability and seamless integration with existing architectures, making it suitable for real-world multimodal applications. 

**Abstract (ZH)**: 解决缺失模态和有限标记数据对于推进稳健的多模态学习至关重要。我们提出Robult，这是一种可扩展的框架，通过保留模态特定信息并利用新颖的信息论方法中的冗余性来缓解这些挑战。Robult 优化了两个核心目标：（1）一种软正负未标记（PU）对比损失，该损失最大化任务相关特征对齐，同时在半监督设置中有效利用有限标记数据；（2）潜在重构损失，确保保留独特的模态特定信息。这些策略嵌入在模块化设计中，在各种下游任务中提高性能，并确保在推断过程中对不完整模态的鲁棒性。跨多个数据集的实验结果验证了Robult在半监督学习和缺失模态情景中均实现了优于现有方法的性能。此外，其轻量级设计促进了可扩展性，并与现有架构无缝集成，使其适用于实际的多模态应用。 

---
# DPQuant: Efficient and Differentially-Private Model Training via Dynamic Quantization Scheduling 

**Title (ZH)**: DPQuant: 通过动态量化调度实现高效差分隐私模型训练 

**Authors**: Yubo Gao, Renbo Tu, Gennady Pekhimenko, Nandita Vijaykumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.03472)  

**Abstract**: Differentially-Private SGD (DP-SGD) is a powerful technique to protect user privacy when using sensitive data to train neural networks. During training, converting model weights and activations into low-precision formats, i.e., quantization, can drastically reduce training times, energy consumption, and cost, and is thus a widely used technique. In this work, we demonstrate that quantization causes significantly higher accuracy degradation in DP-SGD compared to regular SGD. We observe that this is caused by noise injection in DP-SGD, which amplifies quantization variance, leading to disproportionately large accuracy degradation. To address this challenge, we present QPQuant, a dynamic quantization framework that adaptively selects a changing subset of layers to quantize at each epoch. Our method combines two key ideas that effectively reduce quantization variance: (i) probabilistic sampling of the layers that rotates which layers are quantized every epoch, and (ii) loss-aware layer prioritization, which uses a differentially private loss sensitivity estimator to identify layers that can be quantized with minimal impact on model quality. This estimator consumes a negligible fraction of the overall privacy budget, preserving DP guarantees. Empirical evaluations on ResNet18, ResNet50, and DenseNet121 across a range of datasets demonstrate that DPQuant consistently outperforms static quantization baselines, achieving near Pareto-optimal accuracy-compute trade-offs and up to 2.21x theoretical throughput improvements on low-precision hardware, with less than 2% drop in validation accuracy. 

**Abstract (ZH)**: 差分隐私SGD（DP-SGD）中的量化影响研究：QPQuant动态量化框架 

---
# Continuous Saudi Sign Language Recognition: A Vision Transformer Approach 

**Title (ZH)**: 连续Saudi手语识别：一种视觉变换器方法 

**Authors**: Soukeina Elhassen, Lama Al Khuzayem, Areej Alhothali, Ohoud Alzamzami, Nahed Alowaidi  

**Link**: [PDF](https://arxiv.org/pdf/2509.03467)  

**Abstract**: Sign language (SL) is an essential communication form for hearing-impaired and deaf people, enabling engagement within the broader society. Despite its significance, limited public awareness of SL often leads to inequitable access to educational and professional opportunities, thereby contributing to social exclusion, particularly in Saudi Arabia, where over 84,000 individuals depend on Saudi Sign Language (SSL) as their primary form of communication. Although certain technological approaches have helped to improve communication for individuals with hearing impairments, there continues to be an urgent requirement for more precise and dependable translation techniques, especially for Arabic sign language variants like SSL. Most state-of-the-art solutions have primarily focused on non-Arabic sign languages, resulting in a considerable absence of resources dedicated to Arabic sign language, specifically SSL. The complexity of the Arabic language and the prevalence of isolated sign language datasets that concentrate on individual words instead of continuous speech contribute to this issue. To address this gap, our research represents an important step in developing SSL resources. To address this, we introduce the first continuous Saudi Sign Language dataset called KAU-CSSL, focusing on complete sentences to facilitate further research and enable sophisticated recognition systems for SSL recognition and translation. Additionally, we propose a transformer-based model, utilizing a pretrained ResNet-18 for spatial feature extraction and a Transformer Encoder with Bidirectional LSTM for temporal dependencies, achieving 99.02\% accuracy at signer dependent mode and 77.71\% accuracy at signer independent mode. This development leads the way to not only improving communication tools for the SSL community but also making a substantial contribution to the wider field of sign language. 

**Abstract (ZH)**: 手语（SL）是听力障碍和 deaf 人士的重要沟通方式，有助于他们在更广泛的社会中进行交流。尽管手语非常重要，但由于公众对手语认识有限，这往往导致教育和职业机会获取不平等，进而加剧社会排斥，特别是在沙特阿拉伯，超过 84,000 人依赖沙特手语（SSL）作为他们的主要沟通方式。尽管某些技术手段有助于改善听力障碍人士的沟通，但仍然迫切需要更准确可靠的翻译技术，特别是针对阿拉伯手语变体如 SSL。现有的大多数最先进解决方案主要关注非阿拉伯手语，导致阿拉伯手语资源尤其是 SSL 的资源相对匮乏。阿拉伯语言的复杂性和孤立的手语数据集主要关注单个手语词汇而不是连贯的手语交流，进一步加剧了这一问题。为解决这一差距，我们的研究代表了开发 SSL 资源的重要一步。为此，我们介绍了第一个连续的沙特手语数据集 KAU-CSSL，专注于完整的句子以促进进一步研究，并实现针对 SSL 识别和翻译的复杂识别系统。此外，我们提出了一种基于变压器的模型，利用预训练的 ResNet-18 进行空间特征提取，并使用双向 LSTM 与变压器编码器处理时间依赖性，分别在书写者依赖模式和书写者独立模式下达到 99.02% 和 77.71% 的准确率。这一发展不仅为 SSL 社区提供更好的沟通工具，也为更广泛的手语领域做出了重要贡献。 

---
# Multi-level SSL Feature Gating for Audio Deepfake Detection 

**Title (ZH)**: 多级SSL特征门控音频合成换音检测 

**Authors**: Hoan My Tran, Damien Lolive, Aghilas Sini, Arnaud Delhay, Pierre-François Marteau, David Guennec  

**Link**: [PDF](https://arxiv.org/pdf/2509.03409)  

**Abstract**: Recent advancements in generative AI, particularly in speech synthesis, have enabled the generation of highly natural-sounding synthetic speech that closely mimics human voices. While these innovations hold promise for applications like assistive technologies, they also pose significant risks, including misuse for fraudulent activities, identity theft, and security threats. Current research on spoofing detection countermeasures remains limited by generalization to unseen deepfake attacks and languages. To address this, we propose a gating mechanism extracting relevant feature from the speech foundation XLS-R model as a front-end feature extractor. For downstream back-end classifier, we employ Multi-kernel gated Convolution (MultiConv) to capture both local and global speech artifacts. Additionally, we introduce Centered Kernel Alignment (CKA) as a similarity metric to enforce diversity in learned features across different MultiConv layers. By integrating CKA with our gating mechanism, we hypothesize that each component helps improving the learning of distinct synthetic speech patterns. Experimental results demonstrate that our approach achieves state-of-the-art performance on in-domain benchmarks while generalizing robustly to out-of-domain datasets, including multilingual speech samples. This underscores its potential as a versatile solution for detecting evolving speech deepfake threats. 

**Abstract (ZH)**: 近年来，在生成AI，特别是语音合成方面的进展，使生成高自然度合成语音成为可能，这种语音接近真人声音。虽然这些创新在辅助技术等领域充满潜力，但也带来了严重的滥用风险，包括欺诈活动、身份盗窃和安全威胁。当前的伪造检测对策研究主要局限于对未见的深度伪造攻击和语言的一般化。为应对这一挑战，我们提出了一种门控机制，从语音基础XLS-R模型中提取相关特征作为前端特征提取器。对于下游后端分类器，我们采用多核门控卷积（MultiConv）来捕获局部和全局语音特征。此外，我们引入了中心核对齐（CKA）作为相似性度量，以确保不同多核卷积层中学习到的特征具有多样性。通过将CKA与我们的门控机制结合，我们假设每一部分都有助于学习不同的合成语音模式。实验结果显示，我们的方法在领域内基准测试中达到最佳性能，并且在领域外数据集（包括多语言语音样本）上具有稳健的一般化能力。这表明它可能成为一个多功能的伪造检测解决方案，应对不断演变的语音深度伪造威胁。 

---
# Beyond Correctness: Harmonizing Process and Outcome Rewards through RL Training 

**Title (ZH)**: 超越正确性：通过RL训练谐调过程和结果奖励 

**Authors**: Chenlu Ye, Zhou Yu, Ziji Zhang, Hao Chen, Narayanan Sadagopan, Jing Huang, Tong Zhang, Anurag Beniwal  

**Link**: [PDF](https://arxiv.org/pdf/2509.03403)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has emerged to be a predominant paradigm for mathematical reasoning tasks, offering stable improvements in reasoning ability. However, Outcome Reward Models (ORMs) in RLVR are too coarse-grained to distinguish flawed reasoning within correct answers or valid reasoning within incorrect answers. This lack of granularity introduces noisy and misleading gradients significantly and hinders further progress in reasoning process quality. While Process Reward Models (PRMs) offer fine-grained guidance for intermediate steps, they frequently suffer from inaccuracies and are susceptible to reward hacking.
To resolve this dilemma, we introduce PRocess cOnsistency Filter (PROF), an effective data process curation method that harmonizes noisy, fine-grained process rewards with accurate, coarse-grained outcome rewards. Rather than naively blending PRM and ORM in the objective function (arXiv:archive/2506.18896), PROF leverages their complementary strengths through consistency-driven sample selection. Our approach retains correct responses with higher averaged process values and incorrect responses with lower averaged process values, while maintaining positive/negative training sample balance. Extensive experiments demonstrate that our method not only consistently improves the final accuracy over $4\%$ compared to the blending approaches, but also strengthens the quality of intermediate reasoning steps. Codes and training recipes are available at this https URL. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）已在数学推理任务中崭露头角，提供了稳定改进的推理能力。然而，RLVR中的结果奖励模型（ORMs）过于粗粒度，难以区分正确答案中的错误推理或错误答案中的合理推理。这种缺乏粒度的区分引入了噪声和误导性的梯度，阻碍了推理过程质量的进一步提升。虽然过程奖励模型（PRMs）为中间步骤提供了细粒度的指导，但它们经常存在准确性问题且容易受到奖励劫持的影响。

为了解决这一困境，我们引入了过程一致性过滤器（PROF），这是一种有效的数据处理方法，能够将噪声的细粒度过程奖励与准确的粗粒度结果奖励和谐统一。我们的方法通过一致性驱动的样本选择，结合利用PRM和ORM的互补优势，而不是单纯地在目标函数中混融合并（arXiv:archive/2506.18896）。PROF在保持答案正确性和错误性训练样本平衡的同时，提升了中间推理步骤的质量。广泛的实验表明，我们的方法不仅在最终准确率上比融合方法提高了超过4%，还增强了中间推理步骤的质量。相关代码和训练配方可在以下链接获取。 

---
# TinyDrop: Tiny Model Guided Token Dropping for Vision Transformers 

**Title (ZH)**: TinyDrop: 由Tiny模型引导的Token丢弃方法在视觉变换器中的应用 

**Authors**: Guoxin Wang, Qingyuan Wang, Binhua Huang, Shaowu Chen, Deepu John  

**Link**: [PDF](https://arxiv.org/pdf/2509.03379)  

**Abstract**: Vision Transformers (ViTs) achieve strong performance in image classification but incur high computational costs from processing all image tokens. To reduce inference costs in large ViTs without compromising accuracy, we propose TinyDrop, a training-free token dropping framework guided by a lightweight vision model. The guidance model estimates the importance of tokens while performing inference, thereby selectively discarding low-importance tokens if large vit models need to perform attention calculations. The framework operates plug-and-play, requires no architectural modifications, and is compatible with diverse ViT architectures. Evaluations on standard image classification benchmarks demonstrate that our framework reduces FLOPs by up to 80% for ViTs with minimal accuracy degradation, highlighting its generalization capability and practical utility for efficient ViT-based classification. 

**Abstract (ZH)**: Vision Transformers (ViTs)通过图像分类表现强劲，但处理所有图像标记会带来高昂的计算成本。为在不牺牲准确性的前提下减少大ViTs的推理成本，我们提出TinyDrop，这是一种基于轻量级视觉模型的无训练-token丢弃框架。指导模型在推理过程中估计标记的重要性，从而在大型vit模型需要执行注意力计算时选择性地丢弃低重要性标记。该框架插即用，无需修改架构，并且兼容多种ViT架构。在标准图像分类基准上的评估结果显示，本框架可将ViTs的FLOPs最多降低80%，同时准确率下降可忽略不计，突显了其泛化能力和高效ViT分类的实际应用价值。 

---
# Neural Field Turing Machine: A Differentiable Spatial Computer 

**Title (ZH)**: 神经场图灵机：一个可微时空计算机 

**Authors**: Akash Malhotra, Nacéra Seghouani  

**Link**: [PDF](https://arxiv.org/pdf/2509.03370)  

**Abstract**: We introduce the Neural Field Turing Machine (NFTM), a differentiable architecture that unifies symbolic computation, physical simulation, and perceptual inference within continuous spatial fields. NFTM combines a neural controller, continuous memory field, and movable read/write heads that perform local updates. At each timestep, the controller reads local patches, computes updates via learned rules, and writes them back while updating head positions. This design achieves linear O(N) scaling through fixed-radius neighborhoods while maintaining Turing completeness under bounded error. We demonstrate three example instantiations of NFTM: cellular automata simulation (Rule 110), physics-informed PDE solvers (2D heat equation), and iterative image refinement (CIFAR-10 inpainting). These instantiations learn local update rules that compose into global dynamics, exhibit stable long-horizon rollouts, and generalize beyond training horizons. NFTM provides a unified computational substrate bridging discrete algorithms and continuous field dynamics within a single differentiable framework. 

**Abstract (ZH)**: neural场图灵机 (NFTM): 统一符号计算、物理仿真和感知推理的可微架构 

---
# Fair Resource Allocation for Fleet Intelligence 

**Title (ZH)**: 公平资源分配以实现车队智能 

**Authors**: Oguzhan Baser, Kaan Kale, Po-han Li, Sandeep Chinchali  

**Link**: [PDF](https://arxiv.org/pdf/2509.03353)  

**Abstract**: Resource allocation is crucial for the performance optimization of cloud-assisted multi-agent intelligence. Traditional methods often overlook agents' diverse computational capabilities and complex operating environments, leading to inefficient and unfair resource distribution. To address this, we open-sourced Fair-Synergy, an algorithmic framework that utilizes the concave relationship between the agents' accuracy and the system resources to ensure fair resource allocation across fleet intelligence. We extend traditional allocation approaches to encompass a multidimensional machine learning utility landscape defined by model parameters, training data volume, and task complexity. We evaluate Fair-Synergy with advanced vision and language models such as BERT, VGG16, MobileNet, and ResNets on datasets including MNIST, CIFAR-10, CIFAR-100, BDD, and GLUE. We demonstrate that Fair-Synergy outperforms standard benchmarks by up to 25% in multi-agent inference and 11% in multi-agent learning settings. Also, we explore how the level of fairness affects the least advantaged, most advantaged, and average agents, providing insights for equitable fleet intelligence. 

**Abstract (ZH)**: 云辅助多智能体系统中资源分配对于性能优化至关重要。传统方法往往忽视了智能体多样化的计算能力和复杂的运行环境，导致资源配置效率低下且不公平。为此，我们开源了Fair-Synergy这一算法框架，利用智能体准确率与系统资源之间的凹关系，确保在整个智能舰队中实现公平资源配置。我们扩展了传统的分配方法，将其包含在由模型参数、训练数据量和任务复杂度定义的多维机器学习效用景观中。我们使用包括BERT、VGG16、MobileNet和ResNets在内的高级视觉和语言模型，在MNIST、CIFAR-10、CIFAR-100、BDD和GLUE等数据集上评估了Fair-Synergy。结果显示，与标准基准相比，Fair-Synergy在多智能体推理中最高可提升25%，在多智能体学习环境中可提升11%。我们还探讨了公平性水平对最不利、最有利和平均智能体的影响，为公平智能舰队提供了见解。 

---
# epiGPTope: A machine learning-based epitope generator and classifier 

**Title (ZH)**: epiGPTope：一种基于机器学习的表位生成器和分类器 

**Authors**: Natalia Flechas Manrique, Alberto Martínez, Elena López-Martínez, Luc Andrea, Román Orus, Aitor Manteca, Aitziber L. Cortajarena, Llorenç Espinosa-Portalés  

**Link**: [PDF](https://arxiv.org/pdf/2509.03351)  

**Abstract**: Epitopes are short antigenic peptide sequences which are recognized by antibodies or immune cell receptors. These are central to the development of immunotherapies, vaccines, and diagnostics. However, the rational design of synthetic epitope libraries is challenging due to the large combinatorial sequence space, $20^n$ combinations for linear epitopes of n amino acids, making screening and testing unfeasible, even with high throughput experimental techniques. In this study, we present a large language model, epiGPTope, pre-trained on protein data and specifically fine-tuned on linear epitopes, which for the first time can directly generate novel epitope-like sequences, which are found to possess statistical properties analogous to the ones of known epitopes. This generative approach can be used to prepare libraries of epitope candidate sequences. We further train statistical classifiers to predict whether an epitope sequence is of bacterial or viral origin, thus narrowing the candidate library and increasing the likelihood of identifying specific epitopes. We propose that such combination of generative and predictive models can be of assistance in epitope discovery. The approach uses only primary amino acid sequences of linear epitopes, bypassing the need for a geometric framework or hand-crafted features of the sequences. By developing a method to create biologically feasible sequences, we anticipate faster and more cost-effective generation and screening of synthetic epitopes, with relevant applications in the development of new biotechnologies. 

**Abstract (ZH)**: Epitope生成与预测模型epiGPTope在合成表位库设计中的应用 

---
# On the MIA Vulnerability Gap Between Private GANs and Diffusion Models 

**Title (ZH)**: private GANs与扩散模型之间的MIA漏洞差距 

**Authors**: Ilana Sebag, Jean-Yves Franceschi, Alain Rakotomamonjy, Alexandre Allauzen, Jamal Atif  

**Link**: [PDF](https://arxiv.org/pdf/2509.03341)  

**Abstract**: Generative Adversarial Networks (GANs) and diffusion models have emerged as leading approaches for high-quality image synthesis. While both can be trained under differential privacy (DP) to protect sensitive data, their sensitivity to membership inference attacks (MIAs), a key threat to data confidentiality, remains poorly understood. In this work, we present the first unified theoretical and empirical analysis of the privacy risks faced by differentially private generative models. We begin by showing, through a stability-based analysis, that GANs exhibit fundamentally lower sensitivity to data perturbations than diffusion models, suggesting a structural advantage in resisting MIAs. We then validate this insight with a comprehensive empirical study using a standardized MIA pipeline to evaluate privacy leakage across datasets and privacy budgets. Our results consistently reveal a marked privacy robustness gap in favor of GANs, even in strong DP regimes, highlighting that model type alone can critically shape privacy leakage. 

**Abstract (ZH)**: 生成对抗网络（GANs）和扩散模型已成为高质量图像合成的领先方法。虽然两者都可以在差异隐私（DP）下进行训练以保护敏感数据，但它们在成员推理攻击（MIAs）方面的敏感性，这一数据保密性的重要威胁，仍然了解不足。在本工作中，我们首次对不同差分隐私生成模型面临的隐私风险进行了统一的理论和实证分析。我们首先通过稳定性分析显示，GANs在数据扰动下的敏感性明显低于扩散模型，这表明GANs在抵抗MIAs方面具有结构上的优势。然后，我们通过一个标准化的MIAs管道进行全面的实证研究，评估在不同数据集和隐私预算下的隐私泄露情况。我们的结果一致表明，即使在强DP条件下，GANs也显示出明显的隐私鲁棒性差距，强调了模型类型本身对隐私泄露的严重影响。 

---
# Equivariant Flow Matching for Symmetry-Breaking Bifurcation Problems 

**Title (ZH)**: 对称破缺分岔问题的协变流匹配方法 

**Authors**: Fleur Hendriks, Ondřej Rokoš, Martin Doškář, Marc G.D. Geers, Vlado Menkovski  

**Link**: [PDF](https://arxiv.org/pdf/2509.03340)  

**Abstract**: Bifurcation phenomena in nonlinear dynamical systems often lead to multiple coexisting stable solutions, particularly in the presence of symmetry breaking. Deterministic machine learning models struggle to capture this multiplicity, averaging over solutions and failing to represent lower-symmetry outcomes. In this work, we propose a generative framework based on flow matching to model the full probability distribution over bifurcation outcomes. Our method enables direct sampling of multiple valid solutions while preserving system symmetries through equivariant modeling. We introduce a symmetric matching strategy that aligns predicted and target outputs under group actions, allowing accurate learning in equivariant settings. We validate our approach on a range of systems, from toy models to complex physical problems such as buckling beams and the Allen-Cahn equation. Our results demonstrate that flow matching significantly outperforms non-probabilistic and variational methods in capturing multimodal distributions and symmetry-breaking bifurcations, offering a principled and scalable solution for modeling multistability in high-dimensional systems. 

**Abstract (ZH)**: 非线性动力系统中的分岔现象往往会导致多重共存的稳定解，特别是在对称破缺的情况下。确定性机器学习模型难以捕捉这种多样性，它们会平均化解的存在，无法表现较低对称性的结果。本文提出了一种基于流匹配的生成框架，用于建模分岔结果的完整概率分布。该方法能够直接采样多个有效的解，同时通过不变性建模保持系统的对称性。我们引入了一种对称匹配策略，在群作用下对预测输出和目标输出进行对齐，从而在不变性设置中实现准确的学习。我们在此类系统上进行了验证，包括玩具模型以及复杂的物理问题，如弯曲梁和Allen-Cahn方程。我们的结果表明，流匹配方法在捕捉多模态分布和对称破缺分岔方面显著优于非概率性和变分方法，为高维系统的多稳态建模提供了原理上和可扩展的解决方案。 

---
# Heatmap Guided Query Transformers for Robust Astrocyte Detection across Immunostains and Resolutions 

**Title (ZH)**: heatmap引导的查询变换器在跨免疫染色和分辨率的星形胶质细胞检测中的稳健检测 

**Authors**: Xizhe Zhang, Jiayang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.03323)  

**Abstract**: Astrocytes are critical glial cells whose altered morphology and density are hallmarks of many neurological disorders. However, their intricate branching and stain dependent variability make automated detection of histological images a highly challenging task. To address these challenges, we propose a hybrid CNN Transformer detector that combines local feature extraction with global contextual reasoning. A heatmap guided query mechanism generates spatially grounded anchors for small and faint astrocytes, while a lightweight Transformer module improves discrimination in dense clusters. Evaluated on ALDH1L1 and GFAP stained astrocyte datasets, the model consistently outperformed Faster R-CNN, YOLOv11 and DETR, achieving higher sensitivity with fewer false positives, as confirmed by FROC analysis. These results highlight the potential of hybrid CNN Transformer architectures for robust astrocyte detection and provide a foundation for advanced computational pathology tools. 

**Abstract (ZH)**: 星形胶质细胞是关键的胶质细胞，其异常形态和密度是许多神经系统疾病的特点。然而，它们复杂的分支结构和染色依赖的变异使自动化检测组织学图像成为一个极具挑战的任务。为应对这些挑战，我们提出了一种结合局部特征提取与全局上下文推理的混合CNN变压器检测器。热图引导的查询机制生成空间定位的锚点，以识别小而弱的星形胶质细胞，而轻量级的变压器模块则在密集簇中提高区分能力。该模型在ALDH1L1和GFAP染色的星形胶质细胞数据集上测试，一致优于Faster R-CNN、YOLOv11和DETR，展现出更高的敏感性并减少假阳性，FROC分析证实了这一点。这些结果突显了混合CNN transformer架构在稳健星形胶质细胞检测中的潜力，并为先进的计算病理学工具提供了基础。 

---
# Automatic Differentiation of Agent-Based Models 

**Title (ZH)**: 基于代理的模型的自动微分 

**Authors**: Arnau Quera-Bofarull, Nicholas Bishop, Joel Dyer, Daniel Jarne Ornia, Anisoara Calinescu, Doyne Farmer, Michael Wooldridge  

**Link**: [PDF](https://arxiv.org/pdf/2509.03303)  

**Abstract**: Agent-based models (ABMs) simulate complex systems by capturing the bottom-up interactions of individual agents comprising the system. Many complex systems of interest, such as epidemics or financial markets, involve thousands or even millions of agents. Consequently, ABMs often become computationally demanding and rely on the calibration of numerous free parameters, which has significantly hindered their widespread adoption. In this paper, we demonstrate that automatic differentiation (AD) techniques can effectively alleviate these computational burdens. By applying AD to ABMs, the gradients of the simulator become readily available, greatly facilitating essential tasks such as calibration and sensitivity analysis. Specifically, we show how AD enables variational inference (VI) techniques for efficient parameter calibration. Our experiments demonstrate substantial performance improvements and computational savings using VI on three prominent ABMs: Axtell's model of firms; Sugarscape; and the SIR epidemiological model. Our approach thus significantly enhances the practicality and scalability of ABMs for studying complex systems. 

**Abstract (ZH)**: 基于代理模型（ABMs）通过捕捉系统中个体代理的自底向上的交互来模拟复杂系统。许多感兴趣的复杂系统，如流行病或金融市场，涉及数千甚至数百万个代理。因此，ABMs往往变得计算密集，并依赖于大量自由参数的校准，这极大地阻碍了它们的广泛应用。在本文中，我们证明自动微分（AD）技术可以有效减轻这些计算负担。通过将AD应用于ABMs，模拟器的梯度变得易得，极大地促进了诸如校准和灵敏度分析等关键任务的进行。具体而言，我们展示了AD如何使变量推理（VI）技术能够用于高效参数校准。我们的实验表明，在Axtell的企业的模型、Sugarscape和SIR流行病学模型这三个著名ABMs上使用VI方法实现了显著的性能改进和计算节省。因此，我们的方法显著提高了ABMs在研究复杂系统方面的实用性和可扩展性。 

---
# A Comprehensive Guide to Differential Privacy: From Theory to User Expectations 

**Title (ZH)**: 差分隐私全面指南：从理论到用户期望 

**Authors**: Napsu Karmitsa, Antti Airola, Tapio Pahikkala, Tinja Pitkämäki  

**Link**: [PDF](https://arxiv.org/pdf/2509.03294)  

**Abstract**: The increasing availability of personal data has enabled significant advances in fields such as machine learning, healthcare, and cybersecurity. However, this data abundance also raises serious privacy concerns, especially in light of powerful re-identification attacks and growing legal and ethical demands for responsible data use. Differential privacy (DP) has emerged as a principled, mathematically grounded framework for mitigating these risks. This review provides a comprehensive survey of DP, covering its theoretical foundations, practical mechanisms, and real-world applications. It explores key algorithmic tools and domain-specific challenges - particularly in privacy-preserving machine learning and synthetic data generation. The report also highlights usability issues and the need for improved communication and transparency in DP systems. Overall, the goal is to support informed adoption of DP by researchers and practitioners navigating the evolving landscape of data privacy. 

**Abstract (ZH)**: 个人数据的日益可用性促进了机器学习、医疗保健和网络安全等领域的重要进展。然而，这种数据 abundance 也引发了严重的隐私担忧，特别是在强大的重新识别攻击和日益增长的法律和伦理要求背景下。差分隐私（DP）已成为缓解这些风险的一种有原则且数学上坚实的方法论框架。本文综述了差分隐私，涵盖了其理论基础、实用机制及其实际应用。报告探讨了关键的算法工具和特定领域挑战，特别是在隐私保护机器学习和合成数据生成方面的挑战。报告还强调了易用性问题，并指出了需要提高差分隐私系统中的沟通与透明度。总体目标是为研究人员和从业者在不断演变的数据隐私 landscape 中作出知情采纳提供支持。 

---
# Estudio de la eficiencia en la escalabilidad de GPUs para el entrenamiento de Inteligencia Artificial 

**Title (ZH)**: GPU在人工智能训练中可扩展性效率研究 

**Authors**: David Cortes, Carlos Juiz, Belen Bermejo  

**Link**: [PDF](https://arxiv.org/pdf/2509.03263)  

**Abstract**: Training large-scale deep learning models has become a key challenge for the scientific community and industry. While the massive use of GPUs can significantly speed up training times, this approach has a negative impact on efficiency. In this article, we present a detailed analysis of the times reported by MLPerf Training v4.1 on four workloads: BERT, Llama2 LoRA, RetinaNet, and Stable Diffusion, showing that there are configurations that optimise the relationship between performance, GPU usage, and efficiency. The results point to a break-even point that allows training times to be reduced while maximising efficiency. 

**Abstract (ZH)**: 大规模深度学习模型的训练已成为科学界和工业界的key挑战。虽然大量使用GPU可以显著加快训练时间，但这种方法会负面影响效率。在本文中，我们详细分析了MLPerf Training v4.1在四个工作负载（BERT、Llama2 LoRA、RetinaNet、Stable Diffusion）上报告的时间，表明存在优化性能、GPU使用和效率之间关系的配置。结果指出了一个临界点，允许在最大化效率的同时减少训练时间。 

---
# HyPV-LEAD: Proactive Early-Warning of Cryptocurrency Anomalies through Data-Driven Structural-Temporal Modeling 

**Title (ZH)**: HyPV-LEAD: 基于数据驱动结构时序建模的加密货币异常前瞻性早期预警 

**Authors**: Minjung Park, Gyuyeon Na, Soyoun Kim, Sunyoung Moon, HyeonJeong Cha, Sangmi Chai  

**Link**: [PDF](https://arxiv.org/pdf/2509.03260)  

**Abstract**: Abnormal cryptocurrency transactions - such as mixing services, fraudulent transfers, and pump-and-dump operations -- pose escalating risks to financial integrity but remain notoriously difficult to detect due to class imbalance, temporal volatility, and complex network dependencies. Existing approaches are predominantly model-centric and post hoc, flagging anomalies only after they occur and thus offering limited preventive value. This paper introduces HyPV-LEAD (Hyperbolic Peak-Valley Lead-time Enabled Anomaly Detection), a data-driven early-warning framework that explicitly incorporates lead time into anomaly detection. Unlike prior methods, HyPV-LEAD integrates three innovations: (1) window-horizon modeling to guarantee actionable lead-time alerts, (2) Peak-Valley (PV) sampling to mitigate class imbalance while preserving temporal continuity, and (3) hyperbolic embedding to capture the hierarchical and scale-free properties of blockchain transaction networks. Empirical evaluation on large-scale Bitcoin transaction data demonstrates that HyPV-LEAD consistently outperforms state-of-the-art baselines, achieving a PR-AUC of 0.9624 with significant gains in precision and recall. Ablation studies further confirm that each component - PV sampling, hyperbolic embedding, and structural-temporal modeling - provides complementary benefits, with the full framework delivering the highest performance. By shifting anomaly detection from reactive classification to proactive early-warning, HyPV-LEAD establishes a robust foundation for real-time risk management, anti-money laundering (AML) compliance, and financial security in dynamic blockchain environments. 

**Abstract (ZH)**: 异常加密货币交易检测：一种引入领先时间的异常检测框架 

---
# Structure Transfer: an Inference-Based Calculus for the Transformation of Representations 

**Title (ZH)**: 结构转移：一种基于推理的表示转换计算规则 

**Authors**: Daniel Raggi, Gem Stapleton, Mateja Jamnik, Aaron Stockdill, Grecia Garcia Garcia, Peter C-H. Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.03249)  

**Abstract**: Representation choice is of fundamental importance to our ability to communicate and reason effectively. A major unsolved problem, addressed in this paper, is how to devise \textit{representational-system (RS) agnostic} techniques that drive representation transformation and choice. We present a novel calculus, called \textit{structure transfer}, that enables representation transformation across diverse RSs. Specifically, given a \textit{source} representation drawn from a source RS, the rules of structure transfer allow us to generate a \textit{target} representation for a target RS. The generality of structure transfer comes in part from its ability to ensure that the source representation and the generated target representation satisfy \textit{any} specified relation (such as semantic equivalence). This is done by exploiting \textit{schemas}, which encode knowledge about RSs. Specifically, schemas can express \textit{preservation of information} across relations between any pair of RSs, and this knowledge is used by structure transfer to derive a structure for the target representation which ensures that the desired relation holds. We formalise this using Representational Systems Theory~\cite{raggi2022rst}, building on the key concept of a \textit{construction space}. The abstract nature of construction spaces grants them the generality to model RSs of diverse kinds, including formal languages, geometric figures and diagrams, as well as informal notations. Consequently, structure transfer is a system-agnostic calculus that can be used to identify alternative representations in a wide range of practical settings. 

**Abstract (ZH)**: 代表性的选择对于有效沟通和推理至关重要。本文解决的一个主要未解决问题是如何设计代表系统（RS）无关的技术，以驱动代表转换和选择。我们提出了一种新的计算法则，称为结构转移，它能够在不同的RS之间实现代表转换。具体来说，给定一个源自源RS的源代表，结构转移的规则允许我们生成一个目标RS的目标代表。结构转移的通用性部分来自于它能够确保源代表和生成的目标代表满足任何指定的关系（如语义等价）。这一过程通过利用编码了关于RS知识的规范来实现。具体而言，规范可以表达任何一对RS之间关系下的信息保全，并且这些知识被结构转移所利用，以推导出目标代表的结构，确保所需的关系成立。我们使用Representational Systems Theory（Raggi et al., 2022）对此进行形式化，基于构造空间的关键概念。构造空间的抽象性质赋予了它们广泛的通用性，可以模型化各种类型的RS，包括形式语言、几何图形和图表，以及非正式的符号系统。因此，结构转移是一种系统无关的计算法则，可以广泛应用于多种实际场景中来识别替代表示。 

---
# FoMEMO: Towards Foundation Models for Expensive Multi-objective Optimization 

**Title (ZH)**: FoMEMO: 朝着昂贵多目标优化的基石模型方向 

**Authors**: Yiming Yao, Fei Liu, Liang Zhao, Xi Lin, Qingfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.03244)  

**Abstract**: Expensive multi-objective optimization is a prevalent and crucial concern in many real-world scenarios, where sample-efficiency is vital due to the limited evaluations to recover the true Pareto front for decision making. Existing works either involve rebuilding Gaussian process surrogates from scratch for each objective in each new problem encountered, or rely on extensive past domain experiments for pre-training deep learning models, making them hard to generalize and impractical to cope with various emerging applications in the real world. To address this issue, we propose a new paradigm named FoMEMO (Foundation Models for Expensive Multi-objective Optimization), which enables the establishment of a foundation model conditioned on any domain trajectory and user preference, and facilitates fast in-context optimization based on the predicted preference-wise aggregation posteriors. Rather than accessing extensive domain experiments in the real world, we demonstrate that pre-training the foundation model with a diverse set of hundreds of millions of synthetic data can lead to superior adaptability to unknown problems, without necessitating any subsequent model training or updates in the optimization process. We evaluate our method across a variety of synthetic benchmarks and real-word applications, and demonstrate its superior generality and competitive performance compared to existing methods. 

**Abstract (ZH)**: 昂贵的多目标优化是许多实际场景中普遍而关键的问题，由于评估有限，样本效率成为恢复真实帕累托前沿决策的关键。现有方法要么需要为每个新问题从头重建高斯过程代理模型，要么依赖于大型过去的领域实验进行深度学习模型的预训练，这使得它们难以泛化并在现实世界的各种新兴应用中变得不切实际。为了解决这一问题，我们提出了一种名为FoMEMO（基于基础模型的昂贵多目标优化）的新范式，该范式能够在任何领域轨迹和用户偏好的条件下建立基础模型，并基于预测的偏好感知聚合后验实现快速上下文优化。我们证明，使用数百亿个合成数据进行基础模型的预训练，可以在不需要后续模型训练或优化过程中的更新的情况下，实现对未知问题的优秀适应性。我们跨多种合成基准和实际应用评估了该方法，并展示了其在通用性和竞争性能方面优于现有方法。 

---
# Evaluation of Stress Detection as Time Series Events -- A Novel Window-Based F1-Metric 

**Title (ZH)**: 时间序列事件中压力检测的评估——一种新型窗口基F1度量 

**Authors**: Harald Vilhelm Skat-Rørdam, Sneha Das, Kathrine Sofie Rasmussen, Nicole Nadine Lønfeldt, Line Clemmensen  

**Link**: [PDF](https://arxiv.org/pdf/2509.03240)  

**Abstract**: Accurate evaluation of event detection in time series is essential for applications such as stress monitoring with wearable devices, where ground truth is typically annotated as single-point events, even though the underlying phenomena are gradual and temporally diffused. Standard metrics like F1 and point-adjusted F1 (F1$_{pa}$) often misrepresent model performance in such real-world, imbalanced datasets. We introduce a window-based F1 metric (F1$_w$) that incorporates temporal tolerance, enabling a more robust assessment of event detection when exact alignment is unrealistic. Empirical analysis in three physiological datasets, two in-the-wild (ADARP, Wrist Angel) and one experimental (ROAD), indicates that F1$_w$ reveals meaningful model performance patterns invisible to conventional metrics, while its window size can be adapted to domain knowledge to avoid overestimation. We show that the choice of evaluation metric strongly influences the interpretation of model performance: using predictions from TimesFM, only our temporally tolerant metrics reveal statistically significant improvements over random and null baselines in the two in-the-wild use cases. This work addresses key gaps in time series evaluation and provides practical guidance for healthcare applications where requirements for temporal precision vary by context. 

**Abstract (ZH)**: 时间序列中事件检测的准确评估对于可穿戴设备压力监测等应用至关重要，即使底层现象是逐渐且时间上分布的，ground truth通常被标记为单点事件。标准的评价指标如F1和调整后的点F1（F1$_{pa}$）在现实世界中不平衡的数据集上往往不能真实反映模型性能。我们提出了一个基于窗口的F1指标（F1$_w$），该指标考虑了时间容差，使得在精确对齐不现实的情况下，事件检测的评估更加稳健。在三个生理数据集中的实证分析（两个野外实验集ADARP和Wrist Angel，一个实验集ROAD）表明，F1$_w$揭示了传统指标无法观察到的有意义的模型性能模式，其窗口大小可以根据领域知识进行调整以避免高估。我们证明评价指标的选择对模型性能的解释有重大影响：使用TimesFM的预测，在两个野外实验场景中，只有我们的时间容忍性指标能显著优于随机和零基准。本工作填补了时间序列评估的关键空白，并为不同情况下对时间精确度有不同的要求的健康医疗应用提供了实用指导。 

---
# LGBP-OrgaNet: Learnable Gaussian Band Pass Fusion of CNN and Transformer Features for Robust Organoid Segmentation and Tracking 

**Title (ZH)**: LGBP-OrgaNet: 可学习的高斯带通融合网络用于稳健的类器官分割与追踪 

**Authors**: Jing Zhang, Siying Tao, Jiao Li, Tianhe Wang, Junchen Wu, Ruqian Hao, Xiaohui Du, Ruirong Tan, Rui Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.03221)  

**Abstract**: Organoids replicate organ structure and function, playing a crucial role in fields such as tumor treatment and drug screening. Their shape and size can indicate their developmental status, but traditional fluorescence labeling methods risk compromising their structure. Therefore, this paper proposes an automated, non-destructive approach to organoid segmentation and tracking. We introduced the LGBP-OrgaNet, a deep learning-based system proficient in accurately segmenting, tracking, and quantifying organoids. The model leverages complementary information extracted from CNN and Transformer modules and introduces the innovative feature fusion module, Learnable Gaussian Band Pass Fusion, to merge data from two branches. Additionally, in the decoder, the model proposes a Bidirectional Cross Fusion Block to fuse multi-scale features, and finally completes the decoding through progressive concatenation and upsampling. SROrga demonstrates satisfactory segmentation accuracy and robustness on organoids segmentation datasets, providing a potent tool for organoid research. 

**Abstract (ZH)**: 类器官再现器官结构和功能，在肿瘤治疗和药物筛选等领域发挥着重要作用。它们的形状和大小可以反映其发育状态，但传统的荧光标记方法可能损害其结构。因此，本文提出了一种自动化的非破坏性类器官分割与跟踪方法。我们引入了基于深度学习的LGBP-OrgaNet系统，能够准确地分割、跟踪和定量类器官。该模型利用从CNN和Transformer模块中提取的互补信息，并引入了可学习高斯带通融合模块，将两个分支的数据进行融合。此外，在解码器中，模型提出了双向交叉融合块来融合多尺度特征，并最终通过逐步连接和上采样完成解码。SROrga在类器官分割数据集上展示了满意的分割精度和鲁棒性，为类器官研究提供了有力的工具。 

---
# Autonomous Learning From Success and Failure: Goal-Conditioned Supervised Learning with Negative Feedback 

**Title (ZH)**: 自主从成功与失败中学习：基于目标的监督学习与负反馈 

**Authors**: Zeqiang Zhang, Fabian Wurzberger, Gerrit Schmid, Sebastian Gottwald, Daniel A. Braun  

**Link**: [PDF](https://arxiv.org/pdf/2509.03206)  

**Abstract**: Reinforcement learning faces significant challenges when applied to tasks characterized by sparse reward structures. Although imitation learning, within the domain of supervised learning, offers faster convergence, it relies heavily on human-generated demonstrations. Recently, Goal-Conditioned Supervised Learning (GCSL) has emerged as a potential solution by enabling self-imitation learning for autonomous systems. By strategically relabelling goals, agents can derive policy insights from their own experiences. Despite the successes of this framework, it presents two notable limitations: (1) Learning exclusively from self-generated experiences can exacerbate the agents' inherent biases; (2) The relabelling strategy allows agents to focus solely on successful outcomes, precluding them from learning from their mistakes. To address these issues, we propose a novel model that integrates contrastive learning principles into the GCSL framework to learn from both success and failure. Through empirical evaluations, we demonstrate that our algorithm overcomes limitations imposed by agents' initial biases and thereby enables more exploratory behavior. This facilitates the identification and adoption of effective policies, leading to superior performance across a variety of challenging environments. 

**Abstract (ZH)**: 强化学习在稀疏奖励结构的任务中应用时面临着重大挑战。虽然监督学习中的 imitation learning 能够实现更快的收敛，但它依赖于人类生成的示范。最近，目标条件监督学习（GCSL）作为一种潜在解决方案出现，它通过使自主系统进行自我模仿学习，利用目标任务重新标记策略。尽管该框架取得了成功，但它存在两个显著的局限性：（1）仅从自我生成的经验中学习会加剧代理固有的偏差；（2）重新标记策略使代理专注于成功的结果，从而阻止它们从错误中学习。为解决这些问题，我们提出了一种新的模型，将对比学习原理整合进 GCSL 框架，使代理能够从成功和失败中学习。通过实证评估，我们证明该算法克服了由代理初始偏差带来的限制，从而能够实现更探索性的行为。这有助于识别和采用有效的策略，在多种挑战性环境中表现出更优性能。 

---
# AutoDetect: Designing an Autoencoder-based Detection Method for Poisoning Attacks on Object Detection Applications in the Military Domain 

**Title (ZH)**: AutoDetect：针对军事领域目标检测应用中的投毒攻击的自动编码器基于检测方法设计 

**Authors**: Alma M. Liezenga, Stefan Wijnja, Puck de Haan, Niels W. T. Brink, Jip J. van Stijn, Yori Kamphuis, Klamer Schutte  

**Link**: [PDF](https://arxiv.org/pdf/2509.03179)  

**Abstract**: Poisoning attacks pose an increasing threat to the security and robustness of Artificial Intelligence systems in the military domain. The widespread use of open-source datasets and pretrained models exacerbates this risk. Despite the severity of this threat, there is limited research on the application and detection of poisoning attacks on object detection systems. This is especially problematic in the military domain, where attacks can have grave consequences. In this work, we both investigate the effect of poisoning attacks on military object detectors in practice, and the best approach to detect these attacks. To support this research, we create a small, custom dataset featuring military vehicles: MilCivVeh. We explore the vulnerability of military object detectors for poisoning attacks by implementing a modified version of the BadDet attack: a patch-based poisoning attack. We then assess its impact, finding that while a positive attack success rate is achievable, it requires a substantial portion of the data to be poisoned -- raising questions about its practical applicability. To address the detection challenge, we test both specialized poisoning detection methods and anomaly detection methods from the visual industrial inspection domain. Since our research shows that both classes of methods are lacking, we introduce our own patch detection method: AutoDetect, a simple, fast, and lightweight autoencoder-based method. Our method shows promising results in separating clean from poisoned samples using the reconstruction error of image slices, outperforming existing methods, while being less time- and memory-intensive. We urge that the availability of large, representative datasets in the military domain is a prerequisite to further evaluate risks of poisoning attacks and opportunities patch detection. 

**Abstract (ZH)**: 中毒攻击日益威胁军事领域人工智能系统的安全性和鲁棒性。开源数据集和预训练模型的广泛应用进一步加剧了这一风险。尽管该威胁极为严重，但针对目标检测系统中毒攻击的应用与检测研究仍有限。特别是在军事领域，攻击可能导致严重后果。在本研究中，我们不仅探讨了中毒攻击对军事目标检测器的实际影响，还研究了检测这些攻击的最佳方法。为支持该研究，我们创建了一个小型定制数据集——MilCivVeh，其中包括军事车辆。我们通过实现基于补丁的BadDet攻击变体来探索军事目标检测器对中毒攻击的脆弱性。然后评估其影响，发现虽然可以实现一定的攻击成功率，但需要大量数据被中毒——这引发了其实际应用性的质疑。为应对检测挑战，我们测试了专门的中毒检测方法以及来自视觉工业检查领域的异常检测方法。鉴于我们的研究发现这两种方法都有局限性，我们引入了我们自己的补丁检测方法——AutoDetect，这是一种基于简单、快速且轻量级自动编码器的方法。该方法通过图像切片的重构误差分离干净样本和中毒样本，表现出优于现有方法的性能，同时占用时间及内存较少。我们呼吁在军事领域拥有大量并具有代表性的数据集是评估中毒攻击风险和补丁检测机会的先决条件。 

---
# Rashomon in the Streets: Explanation Ambiguity in Scene Understanding 

**Title (ZH)**: 街上的 Rashomon 效应：场景理解中的解释歧义 

**Authors**: Helge Spieker, Jørn Eirik Betten, Arnaud Gotlieb, Nadjib Lazaar, Nassim Belmecheri  

**Link**: [PDF](https://arxiv.org/pdf/2509.03169)  

**Abstract**: Explainable AI (XAI) is essential for validating and trusting models in safety-critical applications like autonomous driving. However, the reliability of XAI is challenged by the Rashomon effect, where multiple, equally accurate models can offer divergent explanations for the same prediction. This paper provides the first empirical quantification of this effect for the task of action prediction in real-world driving scenes. Using Qualitative Explainable Graphs (QXGs) as a symbolic scene representation, we train Rashomon sets of two distinct model classes: interpretable, pair-based gradient boosting models and complex, graph-based Graph Neural Networks (GNNs). Using feature attribution methods, we measure the agreement of explanations both within and between these classes. Our results reveal significant explanation disagreement. Our findings suggest that explanation ambiguity is an inherent property of the problem, not just a modeling artifact. 

**Abstract (ZH)**: 可解释AI（XAI）在自动驾驶等安全关键应用中验证和信任模型至关重要。然而， Rashomon效应对XAI的可靠性构成了挑战，该效应导致多个同等准确的模型可以获得对于同一预测截然不同的解释。本文首次通过实证量化方法研究了该效应在真实驾驶场景中动作预测任务中的表现。利用定性可解释图（QXGs）作为符号场景表示，我们训练了两类不同的模型集合：可解释的成对梯度提升模型和复杂的图基图神经网络（GNNs）。通过特征归因方法，我们测量了这些类内部和跨类解释的一致性。我们的结果揭示了显著的解释分歧。我们的发现表明，解释模糊性是问题本身的固有属性，而不仅仅是一种建模结果。 

---
# Domain Adaptation of LLMs for Process Data 

**Title (ZH)**: LLMs领域适应性研究：面向过程数据 

**Authors**: Rafael Seidi Oyamada, Jari Peeperkorn, Jochen De Weerdt, Johannes De Smedt  

**Link**: [PDF](https://arxiv.org/pdf/2509.03161)  

**Abstract**: In recent years, Large Language Models (LLMs) have emerged as a prominent area of interest across various research domains, including Process Mining (PM). Current applications in PM have predominantly centered on prompt engineering strategies or the transformation of event logs into narrative-style datasets, thereby exploiting the semantic capabilities of LLMs to address diverse tasks. In contrast, this study investigates the direct adaptation of pretrained LLMs to process data without natural language reformulation, motivated by the fact that these models excel in generating sequences of tokens, similar to the objective in PM. More specifically, we focus on parameter-efficient fine-tuning techniques to mitigate the computational overhead typically associated with such models. Our experimental setup focuses on Predictive Process Monitoring (PPM), and considers both single- and multi-task predictions. The results demonstrate a potential improvement in predictive performance over state-of-the-art recurrent neural network (RNN) approaches and recent narrative-style-based solutions, particularly in the multi-task setting. Additionally, our fine-tuned models exhibit faster convergence and require significantly less hyperparameter optimization. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在过程挖掘（PM）等多个研究领域 emerged 为一个研究热点。当前在 PM 中的应用主要集中在提示工程策略或事件日志向叙事型数据集的转换，从而利用 LLMs 的语义能力来解决多样化的问题。与此不同，本研究直接将预训练的 LLMs 调整应用于过程数据，无需自然语言重构，这得益于这些模型在生成类似 PM 目标序列方面的优势。具体而言，我们重点研究参数高效微调技术，以减轻与这些模型相关的主要计算开销。我们的实验设置集中在预测过程监控（PPM）上，考虑单任务和多任务预测。结果表明，与最新的循环神经网络（RNN）方法和基于叙事型的数据解决方案相比，在多任务设置中具有潜在的预测性能提升。此外，我们微调的模型表现出更快的收敛速度，并且需要更少的超参数优化。 

---
# Decentralised self-organisation of pivoting cube ensembles using geometric deep learning 

**Title (ZH)**: 基于几何深度学习的pivot立方体集成的去中心化自我组织 

**Authors**: Nadezhda Dobreva, Emmanuel Blazquez, Jai Grover, Dario Izzo, Yuzhen Qin, Dominik Dold  

**Link**: [PDF](https://arxiv.org/pdf/2509.03140)  

**Abstract**: We present a decentralized model for autonomous reconfiguration of homogeneous pivoting cube modular robots in two dimensions. Each cube in the ensemble is controlled by a neural network that only gains information from other cubes in its local neighborhood, trained using reinforcement learning. Furthermore, using geometric deep learning, we include the grid symmetries of the cube ensemble in the neural network architecture. We find that even the most localized versions succeed in reconfiguring to the target shape, although reconfiguration happens faster the more information about the whole ensemble is available to individual cubes. Near-optimal reconfiguration is achieved with only nearest neighbor interactions by using multiple information passing between cubes, allowing them to accumulate more global information about the ensemble. Compared to standard neural network architectures, using geometric deep learning approaches provided only minor benefits. Overall, we successfully demonstrate mostly local control of a modular self-assembling system, which is transferable to other space-relevant systems with different action spaces, such as sliding cube modular robots and CubeSat swarms. 

**Abstract (ZH)**: 我们提出了一种去中心化的模型，用于自主重构二维同质pivot立方体模块化机器人系统。每个集合中的立方体由只能从其局部邻域中的其他立方体获取信息的神经网络控制，并使用强化学习进行训练。此外，我们利用几何深度学习将立方体集合的网格对称性纳入神经网络架构。我们发现，即使是最局域化的版本也能成功重构为目标形状，尽管每个立方体可以获得整个集合的更多信息时，重构会更快。仅通过立方体之间多次信息传递使用最近邻交互即可实现几乎最优的重构，从而使它们能够累积更多关于集合的全局信息。与标准神经网络架构相比，使用几何深度学习方法仅提供了微小的益处。总体而言，我们成功展示了模块化自组装系统的局部控制，该控制可以转移到其他具有不同动作空间的空间相关系统中，如滑动立方体模块化机器人和CubeSat群。 

---
# A Neural Network Approach to Multi-radionuclide TDCR Beta Spectroscopy 

**Title (ZH)**: 神经网络方法在多放射性核素TDCR_beta能谱分析中的应用 

**Authors**: Li Yi, Qian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.03137)  

**Abstract**: Liquid scintillation triple-to-doubly coincident ratio (TDCR) spectroscopy is widely adopted as a standard method for radionuclide quantification because of its inherent advantages such as high precision, self-calibrating capability, and independence from radioactive reference sources. However, multiradionuclide analysis via TDCR faces the challenges of limited automation and reliance on mixture-specific standards, which may not be easily available. Here, we present an Artificial Intelligence (AI) framework that combines numerical spectral simulation and deep learning for standard-free automated analysis. $\beta$ spectra for model training were generated using Geant4 simulations coupled with statistically modeled detector response sampling. A tailored neural network architecture, trained on this dataset covering various nuclei mix ratio and quenching scenarios, enables autonomous resolution of individual radionuclide activities and detecting efficiency through end-to-end learning paradigms. The model delivers consistent high accuracy across tasks: activity proportions (mean absolute error = 0.009), detection efficiencies (mean absolute error = 0.002), and spectral reconstruction (Structural Similarity Index = 0.9998), validating its physical plausibility for quenched $\beta$ spectroscopy. This AI-driven methodology exhibits significant potential for automated safety-compliant multiradionuclide analysis with robust generalization, real-time processing capabilities, and engineering feasibility, particularly in scenarios where reference materials are unavailable or rapid field analysis is required. 

**Abstract (ZH)**: 基于人工智能的数值谱拟合与深度学习结合的自由标定自动化多放射性核素分析方法 

---
# Adaptive KV-Cache Compression without Manually Setting Budget 

**Title (ZH)**: 无需手动设置预算的自适应KV缓存压缩 

**Authors**: Chenxia Tang, Jianchun Liu, Hongli Xu, Liusheng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.03136)  

**Abstract**: Large language models (LLMs) inference relies heavily on KV-caches to accelerate autoregressive decoding, but the resulting memory footprint grows rapidly with sequence length, posing significant efficiency challenges. Current KV-cache compression methods suffer from a Procrustes' bed problem: they force diverse workloads into fixed compression ratios, leading to suboptimal resource allocation and inference performance. To this end, we present GVote, an adaptive KV-cache compression scheme that eliminates manual budget specification while achieving superior accuracy-efficiency trade-offs. GVote operates on the principle that the important keys are the aggregation of keys required by future queries. The method predicts future query attention demands by Monte-Carlo style sampling potential queries and aggregating selected keys to determine the optimal cache budget without manual specification. Experimental evaluation demonstrates GVote's effectiveness across multiple benchmarks, including GSM8K, RULER and Longbench. Compared to baselines, GVote exhibits 2$\times$ memory reduction while the accuracy maintains higher or comparable. 

**Abstract (ZH)**: 大型语言模型（LLMs）推理依赖于KV缓存来加速自回归解码，但随着序列长度的增长，导致内存占用迅速增加，提出了显著的效率挑战。当前的KV缓存压缩方法存在一种“削足适履”的问题：它们将多样性的工作负载强行压缩到固定的压缩比中，导致资源分配和推理性能不佳。为此，我们提出了GVote，这是一种自适应的KV缓存压缩方案，它可以消除手动预算指定的同时实现更优秀的准确性和效率权衡。GVote的基本原理是重要的键是未来查询所需键的聚合。该方法通过蒙特卡洛样式的采样潜在查询并聚合选择的键来预测未来查询的注意力需求，以确定最优的缓存预算，而无需手动指定。实验评估证明了GVote在包括GSM8K、RULER和Longbench在内的多个基准上的有效性。与基线方法相比，GVote实现了2倍的内存减小，同时保持了更高的或可比拟的准确性。 

---
# From Evaluation to Defense: Constructing Persistent Edit-Based Fingerprints for Large Language Models 

**Title (ZH)**: 从评估到防御：构建大型语言模型的持久性编辑指纹 

**Authors**: Yue Li, Xin Yi, Dongsheng Shi, Yongyi Cui, Gerard de Melo, Xiaoling Wang, Linlin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.03122)  

**Abstract**: The intellectual property (IP) protection of Large Language Models (LLMs) is increasingly critical. Injecting specialized fingerprints into LLMs through instruction tuning is a common IP protection technique. However, this may significantly degrade model performance, requires substantial computational resources, and exhibits poor persistence under model modifications. We argue that knowledge editing offers a lightweight alternative that is more suitable for fingerprint injection. Accordingly, we apply knowledge editing to fingerprint injection for the first time and demonstrate its strong capability. Despite using scrambled text as fingerprints to prevent them from being overwritten during fine-tuning, degradation still occurs under large-scale fine-tuning. To address this, we propose Fingerprint Subspace-aware Fine-Tuning (FSFT), which reduces fingerprint degradation by constraining the update of the fingerprint subspace. The performance of FSFT exceeds fine-tuning by 10% even in the worst-case scenario. Additionally, we observe that the fingerprint-injected models struggle to distinguish between fingerprints and similar texts due to the high similarity of their features. This finding underscores the urgent need for more robust and fine-grained fingerprinting injection methods for LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）的知识产权（IP）保护日益关键。通过指令调优注入专有标记以保护LLMs的知识产权是一种常见技术，但这种方法可能会显著降低模型性能，需要大量计算资源，并且在模型修改下表现出较差的持久性。我们argue知识编辑提供了更轻量级的替代方案，更适合用于标记注入。据此，我们首次将知识编辑应用于标记注入，并展示了其强大的能力。尽管使用混淆文本作为标记以防止在微调过程中被覆盖，但在大规模微调下仍然会发生性能下降。为了解决这一问题，我们提出了感知标记子空间的微调（FSFT）方法，通过限制标记子空间的更新来减少标记的降级。即使在最坏情况下，FSFT的性能也超过常规微调10%。此外，我们观察到注入标记的模型难以区分标记和相似文本，因为它们的特征高度相似。这一发现强调了为LLMs开发更 robust和精细的标记注入方法的迫切需求。 

---
# A Hierarchical Deep Reinforcement Learning Framework for Traffic Signal Control with Predictable Cycle Planning 

**Title (ZH)**: 一种具有可预测周期规划的分层深度强化学习交通信号控制框架 

**Authors**: Hankang Gu, Yuli Zhang, Chengming Wang, Ruiyuan Jiang, Ziheng Qiao, Pengfei Fan, Dongyao Jia  

**Link**: [PDF](https://arxiv.org/pdf/2509.03118)  

**Abstract**: Deep reinforcement learning (DRL) has become a popular approach in traffic signal control (TSC) due to its ability to learn adaptive policies from complex traffic environments. Within DRL-based TSC methods, two primary control paradigms are ``choose phase" and ``switch" strategies. Although the agent in the choose phase paradigm selects the next active phase adaptively, this paradigm may result in unexpected phase sequences for drivers, disrupting their anticipation and potentially compromising safety at intersections. Meanwhile, the switch paradigm allows the agent to decide whether to switch to the next predefined phase or extend the current phase. While this structure maintains a more predictable order, it can lead to unfair and inefficient phase allocations, as certain movements may be extended disproportionately while others are neglected. In this paper, we propose a DRL model, named Deep Hierarchical Cycle Planner (DHCP), to allocate the traffic signal cycle duration hierarchically. A high-level agent first determines the split of the total cycle time between the North-South (NS) and East-West (EW) directions based on the overall traffic state. Then, a low-level agent further divides the allocated duration within each major direction between straight and left-turn movements, enabling more flexible durations for the two movements. We test our model on both real and synthetic road networks, along with multiple sets of real and synthetic traffic flows. Empirical results show our model achieves the best performance over all datasets against baselines. 

**Abstract (ZH)**: 基于深度强化学习的分层周期规划器 (DHCP) 用于交通信号控制 

---
# Information transmission: Inferring change area from change moment in time series remote sensing images 

**Title (ZH)**: 时间序列遥感图像中从变化时刻推断变化区域的信息传递 

**Authors**: Jialu Li, Chen Wu, Meiqi Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.03112)  

**Abstract**: Time series change detection is a critical task for exploring ecosystem dynamics using time series remote sensing images, because it can simultaneously indicate where and when change occur. While deep learning has shown excellent performance in this domain, it continues to approach change area detection and change moment identification as distinct tasks. Given that change area can be inferred from change moment, we propose a time series change detection network, named CAIM-Net (Change Area Inference from Moment Network), to ensure consistency between change area and change moment results. CAIM-Net infers change area from change moment based on the intrinsic relationship between time series analysis and spatial change detection. The CAIM-Net comprises three key steps: Difference Extraction and Enhancement, Coarse Change Moment Extraction, and Fine Change Moment Extraction and Change Area Inference. In the Difference Extraction and Enhancement, a lightweight encoder with batch dimension stacking is designed to rapidly extract difference features. Subsequently, boundary enhancement convolution is applied to amplify these difference features. In the Coarse Change Moment Extraction, the enhanced difference features from the first step are used to spatiotemporal correlation analysis, and then two distinct methods are employed to determine coarse change moments. In the Fine Change Moment Extraction and Change Area Inference, a multiscale temporal Class Activation Mapping (CAM) module first increases the weight of the change-occurring moment from coarse change moments. Then the weighted change moment is used to infer change area based on the fact that pixels with the change moment must have undergone a change. 

**Abstract (ZH)**: 基于时刻的时序变化检测网络：CAIM-Net 

---
# Are We SOLID Yet? An Empirical Study on Prompting LLMs to Detect Design Principle Violations 

**Title (ZH)**: 我们达到 SOLID 标准了吗？关于提示大语言模型检测设计原则违反情况的实证研究 

**Authors**: Fatih Pehlivan, Arçin Ülkü Ergüzen, Sahand Moslemi Yengejeh, Mayasah Lami, Anil Koyuncu  

**Link**: [PDF](https://arxiv.org/pdf/2509.03093)  

**Abstract**: Traditional static analysis methods struggle to detect semantic design flaws, such as violations of the SOLID principles, which require a strong understanding of object-oriented design patterns and principles. Existing solutions typically focus on individual SOLID principles or specific programming languages, leaving a gap in the ability to detect violations across all five principles in multi-language codebases. This paper presents a new approach: a methodology that leverages tailored prompt engineering to assess LLMs on their ability to detect SOLID violations across multiple languages. We present a benchmark of four leading LLMs-CodeLlama, DeepSeekCoder, QwenCoder, and GPT-4o Mini-on their ability to detect violations of all five SOLID principles. For this evaluation, we construct a new benchmark dataset of 240 manually validated code examples. Using this dataset, we test four distinct prompt strategies inspired by established zero-shot, few-shot, and chain-of-thought techniques to systematically measure their impact on detection accuracy. Our emerging results reveal a stark hierarchy among models, with GPT-4o Mini decisively outperforming others, yet even struggles with challenging principles like DIP. Crucially, we show that prompt strategy has a dramatic impact, but no single strategy is universally best; for instance, a deliberative ENSEMBLE prompt excels at OCP detection while a hint-based EXAMPLE prompt is superior for DIP violations. Across all experiments, detection accuracy is heavily influenced by language characteristics and degrades sharply with increasing code complexity. These initial findings demonstrate that effective, AI-driven design analysis requires not a single best model, but a tailored approach that matches the right model and prompt to the specific design context, highlighting the potential of LLMs to support maintainability through AI-assisted code analysis. 

**Abstract (ZH)**: 一种基于定制化提示工程评估大模型检测SOLID原则违反的全新方法 

---
# S2M2ECG: Spatio-temporal bi-directional State Space Model Enabled Multi-branch Mamba for ECG 

**Title (ZH)**: S2M2ECG：空间时间双向状态空间模型驱动的多分支Mamba心电图分析方法 

**Authors**: Huaicheng Zhang, Ruoxin Wang, Chenlian Zhou, Jiguang Shi, Yue Ge, Zhoutong Li, Sheng Chang, Hao Wang, Jin He, Qijun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.03066)  

**Abstract**: As one of the most effective methods for cardiovascular disease (CVD) diagnosis, multi-lead Electrocardiogram (ECG) signals present a characteristic multi-sensor information fusion challenge that has been continuously researched in deep learning domains. Despite the numerous algorithms proposed with different DL architectures, maintaining a balance among performance, computational complexity, and multi-source ECG feature fusion remains challenging. Recently, state space models (SSMs), particularly Mamba, have demonstrated remarkable effectiveness across various fields. Their inherent design for high-efficiency computation and linear complexity makes them particularly suitable for low-dimensional data like ECGs. This work proposes S2M2ECG, an SSM architecture featuring three-level fusion mechanisms: (1) Spatio-temporal bi-directional SSMs with segment tokenization for low-level signal fusion, (2) Intra-lead temporal information fusion with bi-directional scanning to enhance recognition accuracy in both forward and backward directions, (3) Cross-lead feature interaction modules for spatial information fusion. To fully leverage the ECG-specific multi-lead mechanisms inherent in ECG signals, a multi-branch design and lead fusion modules are incorporated, enabling individual analysis of each lead while ensuring seamless integration with others. Experimental results reveal that S2M2ECG achieves superior performance in the rhythmic, morphological, and clinical scenarios. Moreover, its lightweight architecture ensures it has nearly the fewest parameters among existing models, making it highly suitable for efficient inference and convenient deployment. Collectively, S2M2ECG offers a promising alternative that strikes an excellent balance among performance, computational complexity, and ECG-specific characteristics, paving the way for high-performance, lightweight computations in CVD diagnosis. 

**Abstract (ZH)**: 基于时空双向状态空间模型的多导联心电图特征融合方法（S2M2ECG） 

---
# Loong: Synthesize Long Chain-of-Thoughts at Scale through Verifiers 

**Title (ZH)**: Loong: 通过验证器大规模合成长链推理 

**Authors**: Xingyue Huang, Rishabh, Gregor Franke, Ziyi Yang, Jiamu Bai, Weijie Bai, Jinhe Bi, Zifeng Ding, Yiqun Duan, Chengyu Fan, Wendong Fan, Xin Gao, Ruohao Guo, Yuan He, Zhuangzhuang He, Xianglong Hu, Neil Johnson, Bowen Li, Fangru Lin, Siyu Lin, Tong Liu, Yunpu Ma, Hao Shen, Hao Sun, Beibei Wang, Fangyijie Wang, Hao Wang, Haoran Wang, Yang Wang, Yifeng Wang, Zhaowei Wang, Ziyang Wang, Yifan Wu, Zikai Xiao, Chengxing Xie, Fan Yang, Junxiao Yang, Qianshuo Ye, Ziyu Ye, Guangtao Zeng, Yuwen Ebony Zhang, Zeyu Zhang, Zihao Zhu, Bernard Ghanem, Philip Torr, Guohao Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.03059)  

**Abstract**: Recent advances in Large Language Models (LLMs) have shown that their reasoning capabilities can be significantly improved through Reinforcement Learning with Verifiable Reward (RLVR), particularly in domains like mathematics and programming, where ground-truth correctness can be automatically evaluated. However, extending this success to other reasoning-intensive domains remains challenging due to the scarcity of high-quality, verifiable datasets and the high cost of human supervision. In this work, we introduce the Loong Project: an open-source framework for scalable synthetic data generation and verification across a diverse range of reasoning-intensive domains. The framework consists of two key components: (1) LoongBench, a curated seed dataset containing 8,729 human-vetted examples across 12 domains (e.g., Advanced Mathematics, Chemistry, Logic), each paired with executable code and rich metadata; and (2) LoongEnv, a modular synthetic data generation environment that supports multiple prompting strategies to produce new question-answer-code triples. Together, these components form an agent-environment loop that enables reinforcement learning, where an LLM-based agent is rewarded for generating Chain-of-Thought (CoT) solutions that align with code-executed answers. Empirically, we benchmark LoongBench on a broad suite of both open-source and proprietary LLMs to evaluate domain coverage and reveal performance bottlenecks. In addition, we conduct a comprehensive analysis of synthetic data generated by LoongEnv, examining correctness, difficulty, and diversity. Code and documentation are available at this https URL. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）通过可验证奖励的强化学习（RLVR）显著提高了其推理能力，特别是在数学和编程等领域，这些领域的正确性可以自动评估。然而，将这种成功扩展到其他推理密集型领域仍然具有挑战性，原因在于高质量的可验证数据集稀缺以及人类监督的成本高昂。在此项工作中，我们介绍了龙项目：一个开源框架，用于在广泛的推理密集型领域生成和验证大规模合成数据。该框架包含两个关键组件：（1）龙Bench，一个精心挑选的种子数据集，包含8,729个人工审核的示例，涵盖12个领域（例如，高等数学、化学、逻辑），每个示例均配以可执行代码和丰富的元数据；（2）龙Env，一个模块化的合成数据生成环境，支持多种提示策略以生成新的问题-答案-代码三元组。这些组件共同构成了智能体-环境循环，使强化学习成为可能，其中基于LLM的智能体因生成与代码执行答案一致的思维链（CoT）解决方案而获得奖励。实验中，我们使用广泛的开源和专有的LLMs对龙Bench进行基准测试，以评估领域覆盖范围并揭示性能瓶颈。此外，我们对龙Env生成的合成数据进行了全面分析，考察了其正确性、难度和多样性。源代码和文档可在以下网址获取。 

---
# Binary Quantization For LLMs Through Dynamic Grouping 

**Title (ZH)**: 通过动态分组实现大规模语言模型的二元量化 

**Authors**: Xinzhe Zheng, Zhen-Qun Yang, Haoran Xie, S. Joe Qin, Arlene Chen, Fangzhen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.03054)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of Natural Language Processing (NLP) tasks, but require substantial memory and computational resources. Binary quantization, which compresses model weights from 16-bit Brain Float to 1-bit representations in {-1, 1}, offers significant reductions in storage and inference costs. However, such aggressive quantization often leads to notable performance degradation compared to more conservative 4-bit quantization methods. In this research, we propose a novel optimization objective tailored for binary quantization, along with three algorithms designed to realize it effectively. Our method enhances blocked quantization by dynamically identifying optimal unstructured sub-matrices through adaptive grouping strategies. Experimental results demonstrate that our approach achieves an average bit length of just 1.007 bits, while maintaining high model quality. Specifically, our quantized LLaMA 3.2 3B model attains a perplexity of 8.23, remarkably close to the original 7.81, and surpasses previous SOTA BiLLM with a perplexity of only 123.90. Furthermore, our method is competitive with SOTA 4-bit approaches such as GPTQ in both performance and efficiency. The compression process is highly efficient, requiring only 14 seconds to quantize the full LLaMA 3.2 3B weights on a single CPU core, with the entire process completing in under 100 minutes and exhibiting embarrassingly parallel properties.
Code - this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种自然语言处理（NLP）任务中展现了卓越的表现，但需要大量的内存和计算资源。二值量化通过将16位Brain Float模型权重压缩为{-1, 1}表示的1位，显著减少了存储和推理成本。然而，这种激进的量化与更保守的4位量化方法相比，通常会导致性能显著下降。在这项研究中，我们提出了一种针对二值量化的新优化目标，并设计了三种实现该目标的有效算法。我们的方法通过自适应分组策略动态识别最优的无结构子矩阵，增强块量化。实验结果表明，我们的方法在保持高模型质量的同时，平均比特长度仅为1.007位。具体来说，我们的二值量化LLaMA 3.2 3B模型的困惑度达到8.23，非常接近原始的7.81，并且超越了之前的最佳二值量化模型，其困惑度仅为123.90。此外，我们的方法在性能和效率方面与当前最佳的4位方法（如GPTQ）竞争。压缩过程非常高效，单个CPU核心仅需14秒即可完成LLaMA 3.2 3B权重的量化，整个过程在不到100分钟内完成，并具有明显的并行特性。 

---
# FlashRecovery: Fast and Low-Cost Recovery from Failures for Large-Scale Training of LLMs 

**Title (ZH)**: FlashRecovery: 大规模训练LLMs中快速且低成本的故障恢复方法 

**Authors**: Haijun Zhang, Jinxiang Wang, Zhenhua Yu, Yanyong Zhang, Xuejie Ji, Kaining Mao, Jun Zhang, Yaqing Zhang, Ting Wu, Fei Jie, Xiemin Huang, Zhifang Cai, Junhua Cheng, Shuwei Wang, Wei Li, Xiaoming Bao, Hua Xu, Shixiong Zhao, Jun Li, Hongwei Sun, Ziyang Zhang, Yi Xiong, Chunsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.03047)  

**Abstract**: Large language models (LLMs) have made a profound impact across various fields due to their advanced capabilities. However, training these models at unprecedented scales requires extensive AI accelerator clusters and sophisticated parallelism strategies, which pose significant challenges in maintaining system reliability over prolonged training periods. A major concern is the substantial loss of training time caused by inevitable hardware and software failures. To address these challenges, we present FlashRecovery, a fast and low-cost failure recovery system comprising three core modules: (1) Active and real-time failure detection. This module performs continuous training state monitoring, enabling immediate identification of hardware and software failures within seconds, thus ensuring rapid incident response; (2) Scale-independent task restart. By employing different recovery strategies for normal and faulty nodes, combined with an optimized communication group reconstruction protocol, our approach ensures that the recovery time remains nearly constant, regardless of cluster scale; (3) Checkpoint-free recovery within one step. Our novel recovery mechanism enables single-step restoration, completely eliminating dependence on traditional checkpointing methods and their associated overhead. Collectively, these innovations enable FlashRecovery to achieve optimal Recovery Time Objective (RTO) and Recovery Point Objective (RPO), substantially improving the reliability and efficiency of long-duration LLM training. Experimental results demonstrate that FlashRecovery system can achieve training restoration on training cluster with 4, 800 devices in 150 seconds. We also verify that the time required for failure recovery is nearly consistent for different scales of training tasks. 

**Abstract (ZH)**: 大规模语言模型（LLMs）由于其先进的能力，在各个领域产生了深远的影响。然而，以前所未有的规模训练这些模型需要广泛的AI加速器集群和复杂的并行策略，这在长时间内维持系统的可靠性方面提出了重大挑战。主要的担忧是由于不可避免的硬件和软件故障导致的大量训练时间损失。为了解决这些挑战，我们提出了一种快速且低成本的故障恢复系统FlashRecovery，该系统包含三个核心模块：（1）主动且实时的故障检测。此模块进行持续的训练状态监控，能够在几秒内立即识别出硬件和软件故障，从而确保快速响应；（2）规模无关的任务重启。通过为正常节点和故障节点采用不同的恢复策略，并结合优化的通信组重建协议，我们的方法确保了恢复时间几乎保持不变，与集群规模无关；（3）一步检查点恢复。我们创新的恢复机制允许一步恢复，完全消除了对传统检查点方法及其相关开销的依赖。这些创新共同使FlashRecovery能够实现最优的恢复时间目标（RTO）和恢复点目标（RPO），显著提高了长时间大规模语言模型训练的可靠性和效率。实验结果表明，FlashRecovery系统能在150秒内实现包含4,800个设备的训练集群的训练恢复，并且验证了不同规模训练任务的故障恢复时间几乎是恒定的。 

---
# MedLiteNet: Lightweight Hybrid Medical Image Segmentation Model 

**Title (ZH)**: MedLiteNet: 轻量级混合医学图像分割模型 

**Authors**: Pengyang Yu, Haoquan Wang, Gerard Marks, Tahar Kechadi, Laurence T. Yang, Sahraoui Dhelim, Nyothiri Aung  

**Link**: [PDF](https://arxiv.org/pdf/2509.03041)  

**Abstract**: Accurate skin-lesion segmentation remains a key technical challenge for computer-aided diagnosis of skin cancer. Convolutional neural networks, while effective, are constrained by limited receptive fields and thus struggle to model long-range dependencies. Vision Transformers capture global context, yet their quadratic complexity and large parameter budgets hinder use on the small-sample medical datasets common in dermatology. We introduce the MedLiteNet, a lightweight CNN Transformer hybrid tailored for dermoscopic segmentation that achieves high precision through hierarchical feature extraction and multi-scale context aggregation. The encoder stacks depth-wise Mobile Inverted Bottleneck blocks to curb computation, inserts a bottleneck-level cross-scale token-mixing unit to exchange information between resolutions, and embeds a boundary-aware self-attention module to sharpen lesion contours. 

**Abstract (ZH)**: 准确的皮肤病灶分割仍然是皮肤癌计算机辅助诊断中的关键技术挑战。MedLiteNet，一种针对皮肤镜分割的轻量化CNNTransformer混合模型，通过分层特征提取和多尺度上下文聚合实现高精度。 

---
# Knowledge Integration for Physics-informed Symbolic Regression Using Pre-trained Large Language Models 

**Title (ZH)**: 使用预训练大语言模型的物理约束符号回归知识集成 

**Authors**: Bilge Taskin, Wenxiong Xie, Teddy Lazebnik  

**Link**: [PDF](https://arxiv.org/pdf/2509.03036)  

**Abstract**: Symbolic regression (SR) has emerged as a powerful tool for automated scientific discovery, enabling the derivation of governing equations from experimental data. A growing body of work illustrates the promise of integrating domain knowledge into the SR to improve the discovered equation's generality and usefulness. Physics-informed SR (PiSR) addresses this by incorporating domain knowledge, but current methods often require specialized formulations and manual feature engineering, limiting their adaptability only to domain experts. In this study, we leverage pre-trained Large Language Models (LLMs) to facilitate knowledge integration in PiSR. By harnessing the contextual understanding of LLMs trained on vast scientific literature, we aim to automate the incorporation of domain knowledge, reducing the need for manual intervention and making the process more accessible to a broader range of scientific problems. Namely, the LLM is integrated into the SR's loss function, adding a term of the LLM's evaluation of the SR's produced equation. We extensively evaluate our method using three SR algorithms (DEAP, gplearn, and PySR) and three pre-trained LLMs (Falcon, Mistral, and LLama 2) across three physical dynamics (dropping ball, simple harmonic motion, and electromagnetic wave). The results demonstrate that LLM integration consistently improves the reconstruction of physical dynamics from data, enhancing the robustness of SR models to noise and complexity. We further explore the impact of prompt engineering, finding that more informative prompts significantly improve performance. 

**Abstract (ZH)**: 基于预训练大语言模型的物理知情符号回归研究 

---
# Unveiling the Response of Large Vision-Language Models to Visually Absent Tokens 

**Title (ZH)**: 揭示大型视觉-语言模型对视觉缺失词的响应 

**Authors**: Sohee Kim, Soohyun Ryu, Joonhyung Park, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.03025)  

**Abstract**: Large Vision-Language Models (LVLMs) generate contextually relevant responses by jointly interpreting visual and textual inputs. However, our finding reveals they often mistakenly perceive text inputs lacking visual evidence as being part of the image, leading to erroneous responses. In light of this finding, we probe whether LVLMs possess an internal capability to determine if textual concepts are grounded in the image, and discover a specific subset of Feed-Forward Network (FFN) neurons, termed Visual Absence-aware (VA) neurons, that consistently signal the visual absence through a distinctive activation pattern. Leveraging these patterns, we develop a detection module that systematically classifies whether an input token is visually grounded. Guided by its prediction, we propose a method to refine the outputs by reinterpreting question prompts or replacing the detected absent tokens during generation. Extensive experiments show that our method effectively mitigates the models' tendency to falsely presume the visual presence of text input and its generality across various LVLMs. 

**Abstract (ZH)**: 大型视觉-语言模型通过联合解释视觉和文本输入生成上下文相关响应。然而，我们的发现表明，它们常常错误地将缺乏视觉证据的文本输入视为图像的一部分，导致错误的响应。鉴于这一发现，我们探究了大型视觉-语言模型是否具有内部能力来判断文本概念是否基于图像，并发现了一类特定的前向神经网络（FFN）神经元，称为视觉缺失感知（VA）神经元，这些神经元通过特有的激活模式一致地表明视觉缺失。利用这些模式，我们开发了一个检测模块，系统地分类输入标记是否基于视觉。根据其预测，我们提出了一种通过重新解释问题提示或在生成过程中替换检测到的缺失标记来改进模型输出的方法。广泛实验表明，我们的方法有效地减轻了模型假定文本输入视觉存在的倾向，并且适用于各种大型视觉-语言模型。 

---
# Efficient Privacy-Preserving Recommendation on Sparse Data using Fully Homomorphic Encryption 

**Title (ZH)**: 基于全同态加密的稀疏数据高效隐私保护推荐 

**Authors**: Moontaha Nishat Chowdhury, André Bauer, Minxuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.03024)  

**Abstract**: In today's data-driven world, recommendation systems personalize user experiences across industries but rely on sensitive data, raising privacy concerns. Fully homomorphic encryption (FHE) can secure these systems, but a significant challenge in applying FHE to recommendation systems is efficiently handling the inherently large and sparse user-item rating matrices. FHE operations are computationally intensive, and naively processing various sparse matrices in recommendation systems would be prohibitively expensive. Additionally, the communication overhead between parties remains a critical concern in encrypted domains. We propose a novel approach combining Compressed Sparse Row (CSR) representation with FHE-based matrix factorization that efficiently handles matrix sparsity in the encrypted domain while minimizing communication costs. Our experimental results demonstrate high recommendation accuracy with encrypted data while achieving the lowest communication costs, effectively preserving user privacy. 

**Abstract (ZH)**: 在数据驱动的世界中，推荐系统跨行业个性化用户体验，但依赖敏感数据，引发隐私关切。全同态加密(FHE)可以确保这些系统的安全性，但在推荐系统中应用FHE的一个主要挑战是高效处理固有的大型和稀疏的用户-项评分矩阵。FHE操作计算密集，如果未经优化地处理推荐系统中的各种稀疏矩阵，将变得极其昂贵。此外，加密域中的通信开销仍然是一个关键问题。我们提出了一种新颖的方法，结合压缩稀疏行(CSR)表示与基于FHE的矩阵分解，以在加密域中高效处理矩阵稀疏性并最小化通信成本。实验结果表明，即使在加密数据下也能实现高推荐准确率，同时实现最低的通信成本，从而有效保护用户隐私。 

---
# Lesion-Aware Visual-Language Fusion for Automated Image Captioning of Ulcerative Colitis Endoscopic Examinations 

**Title (ZH)**: 基于病灶aware的视觉-语言融合方法：全自动溃疡性结肠炎内镜检查图像captioning 

**Authors**: Alexis Ivan Lopez Escamilla, Gilberto Ochoa, Sharib Al  

**Link**: [PDF](https://arxiv.org/pdf/2509.03011)  

**Abstract**: We present a lesion-aware image captioning framework for ulcerative colitis (UC). The model integrates ResNet embeddings, Grad-CAM heatmaps, and CBAM-enhanced attention with a T5 decoder. Clinical metadata (MES score 0-3, vascular pattern, bleeding, erythema, friability, ulceration) is injected as natural-language prompts to guide caption generation. The system produces structured, interpretable descriptions aligned with clinical practice and provides MES classification and lesion tags. Compared with baselines, our approach improves caption quality and MES classification accuracy, supporting reliable endoscopic reporting. 

**Abstract (ZH)**: 一种溃疡性结肠炎病变感知的图像描述框架：结合ResNet嵌入、Grad-CAM热图和CBAM增强注意力的T5解码器应用于溃疡性结肠炎的临床图像描述 

---
# StableSleep: Source-Free Test-Time Adaptation for Sleep Staging with Lightweight Safety Rails 

**Title (ZH)**: StableSleep：基于轻量级安全引导的无源测试时自适应睡眠分期方法 

**Authors**: Hritik Arasu, Faisal R Jahangiri  

**Link**: [PDF](https://arxiv.org/pdf/2509.02982)  

**Abstract**: Sleep staging models often degrade when deployed on patients with unseen physiology or recording conditions. We propose a streaming, source-free test-time adaptation (TTA) recipe that combines entropy minimization (Tent) with Batch-Norm statistic refresh and two safety rails: an entropy gate to pause adaptation on uncertain windows and an EMA-based reset to reel back drift. On Sleep-EDF Expanded, using single-lead EEG (Fpz-Cz, 100 Hz, 30s epochs; R&K to AASM mapping), we show consistent gains over a frozen baseline at seconds-level latency and minimal memory, reporting per-stage metrics and Cohen's k. The method is model-agnostic, requires no source data or patient calibration, and is practical for on-device or bedside use. 

**Abstract (ZH)**: 睡眠阶段模型在未见过的生理状态或记录条件下部署时往往会退化。我们提出了一种基于熵最小化（Tent）结合Batch-Norm统计更新，并伴有两种安全机制的在线测试时自适应（TTA）方法：熵门限以暂停不确定窗口上的自适应过程，以及基于EMA的重置以回退漂移。在Sleep-EDF Expanded数据集上，使用单导联EEG（Fpz-Cz，100 Hz，30秒 epoch；R&K到AASM映射），我们显示该方法在毫秒级延迟和minimal内存下相对于冻结基线的一致性改进，并报告了阶段内指标和Cohen's k。该方法对任何模型都是通用的，无需源数据或患者校准，并适用于设备端或床边使用。 

---
# AR-KAN: Autoregressive-Weight-Enhanced Kolmogorov-Arnold Network for Time Series Forecasting 

**Title (ZH)**: AR-KAN：自回归权增强的柯尔莫哥洛夫-阿诺尔德网络时间序列预测 

**Authors**: Chen Zeng, Tiehang Xu, Qiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02967)  

**Abstract**: Conventional neural networks frequently face challenges in spectral analysis of signals. To address this challenge, Fourier neural networks (FNNs) and similar approaches integrate components of Fourier series into the structure of neural networks. Nonetheless, a significant hurdle is often overlooked: the superposition of periodic signals does not necessarily result in a periodic signal. For example, when forecasting almost periodic functions composed of signals with incommensurate frequencies, traditional models such as Autoregressive Integrated Moving Average (ARIMA) frequently outperform most neural networks including large language models (LLMs). To tackle this goal, we propose Autoregressive-Weight-Enhanced AR-KAN, a hybrid model that combines the benefits of both methods. Using the Universal Myopic Mapping Theorem, we apply a Kolmogorov-Arnold Network (KAN) for the static nonlinear part and include memory through a pre-trained AR component, which can be explained to retain the most useful information while eliminating redundancy. Experimental data indicates that AR-KAN delivers superior results on $72\%$ of real-world datasets. 

**Abstract (ZH)**: 传统的神经网络在信号频谱分析中经常面临挑战。为了应对这一挑战，Fourier神经网络（FNNs）和其他类似方法将Fourier级数的成分整合到神经网络结构中。然而，一个重要的障碍经常被忽视：周期信号的叠加并不一定产生周期信号。例如，在预测由非通约频率信号组成的几乎周期函数时，传统的模型如自回归整定移动平均模型（ARIMA）通常能比大多数神经网络，包括大型语言模型（LLMs），表现得更好。为了应对这一目标，我们提出了一种混合模型Autoregressive-Weight-Enhanced AR-KAN，该模型结合了两种方法的优点。利用宇宙近视映射定理，我们使用Kolmogorov-Arnold网络（KAN）处理静态非线性部分，并通过预训练的AR组件引入记忆，可以解释为保留最有用的信息并消除冗余。实验数据表明，AR-KAN在72%的实际数据集中表现更优。 

---
# KEPT: Knowledge-Enhanced Prediction of Trajectories from Consecutive Driving Frames with Vision-Language Models 

**Title (ZH)**: KEPT：知识增强的连续驾驶帧轨迹预测方法基于视觉语言模型 

**Authors**: Yujin Wang, Tianyi Wang, Quanfeng Liu, Wenxian Fan, Junfeng Jiao, Christian Claudel, Yunbing Yan, Bingzhao Gao, Jianqiang Wang, Hong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.02966)  

**Abstract**: Accurate short-horizon trajectory prediction is pivotal for safe and reliable autonomous driving, yet existing vision-language models (VLMs) often fail to effectively ground their reasoning in scene dynamics and domain knowledge. To address this challenge, this paper introduces KEPT, a knowledge-enhanced VLM framework that predicts ego trajectories directly from consecutive front-view driving frames. KEPT couples a temporal frequency-spatial fusion (TFSF) video encoder, trained via self-supervised learning with hard-negative mining, with a scalable k-means + HNSW retrieval stack that supplies scene-aligned exemplars. Retrieved priors are embedded into chain-of-thought (CoT) prompts with explicit planning constraints, while a triple-stage fine-tuning schedule incrementally aligns the language head to metric spatial cues, physically feasible motion, and temporally conditioned front-view planning. Evaluated on nuScenes dataset, KEPT achieves state-of-the-art performance across open-loop protocols: under NoAvg, it achieves 0.70m average L2 with a 0.21\% collision rate; under TemAvg with lightweight ego status, it attains 0.31m average L2 and a 0.07\% collision rate. Ablation studies show that all three fine-tuning stages contribute complementary benefits, and that using Top-2 retrieved exemplars yields the best accuracy-safety trade-off. The k-means-clustered HNSW index delivers sub-millisecond retrieval latency, supporting practical deployment. These results indicate that retrieval-augmented, CoT-guided VLMs offer a promising, data-efficient pathway toward interpretable and trustworthy autonomous driving. 

**Abstract (ZH)**: 知识增强的短时轨迹预测框架 

---
# Lattice Annotated Temporal (LAT) Logic for Non-Markovian Reasoning 

**Title (ZH)**: 非马尔可夫推理的格注释时序（LAT）逻辑 

**Authors**: Kaustuv Mukherji, Jaikrishna Manojkumar Patil, Dyuman Aditya, Paulo Shakarian, Devendra Parkar, Lahari Pokala, Clark Dorman, Gerardo I. Simari  

**Link**: [PDF](https://arxiv.org/pdf/2509.02958)  

**Abstract**: We introduce Lattice Annotated Temporal (LAT) Logic, an extension of Generalized Annotated Logic Programs (GAPs) that incorporates temporal reasoning and supports open-world semantics through the use of a lower lattice structure. This logic combines an efficient deduction process with temporal logic programming to support non-Markovian relationships and open-world reasoning capabilities. The open-world aspect, a by-product of the use of the lower-lattice annotation structure, allows for efficient grounding through a Skolemization process, even in domains with infinite or highly diverse constants.
We provide a suite of theoretical results that bound the computational complexity of the grounding process, in addition to showing that many of the results on GAPs (using an upper lattice) still hold with the lower lattice and temporal extensions (though different proof techniques are required). Our open-source implementation, PyReason, features modular design, machine-level optimizations, and direct integration with reinforcement learning environments. Empirical evaluations across multi-agent simulations and knowledge graph tasks demonstrate up to three orders of magnitude speedup and up to five orders of magnitude memory reduction while maintaining or improving task performance. Additionally, we evaluate LAT Logic's value in reinforcement learning environments as a non-Markovian simulator, achieving up to three orders of magnitude faster simulation with improved agent performance, including a 26% increase in win rate due to capturing richer temporal dependencies. These results highlight LAT Logic's potential as a unified, extensible framework for open-world temporal reasoning in dynamic and uncertain environments. Our implementation is available at: this http URL. 

**Abstract (ZH)**: 晶格注释时序逻辑：一种结合时序推理和支持开放式语义的广义注释逻辑程序扩展 

---
# VendiRL: A Framework for Self-Supervised Reinforcement Learning of Diversely Diverse Skills 

**Title (ZH)**: VendiRL：一个自监督多样化技能强化学习框架 

**Authors**: Erik M. Lintunen  

**Link**: [PDF](https://arxiv.org/pdf/2509.02930)  

**Abstract**: In self-supervised reinforcement learning (RL), one of the key challenges is learning a diverse set of skills to prepare agents for unknown future tasks. Despite impressive advances, scalability and evaluation remain prevalent issues. Regarding scalability, the search for meaningful skills can be obscured by high-dimensional feature spaces, where relevant features may vary across downstream task domains. For evaluating skill diversity, defining what constitutes "diversity" typically requires a hard commitment to a specific notion of what it means for skills to be diverse, potentially leading to inconsistencies in how skill diversity is understood, making results across different approaches hard to compare, and leaving many forms of diversity unexplored. To address these issues, we adopt a measure of sample diversity that translates ideas from ecology to machine learning -- the Vendi Score -- allowing the user to specify and evaluate any desired form of diversity. We demonstrate how this metric facilitates skill evaluation and introduce VendiRL, a unified framework for learning diversely diverse sets of skills. Given distinct similarity functions, VendiRL motivates distinct forms of diversity, which could support skill-diversity pretraining in new and richly interactive environments where optimising for various forms of diversity may be desirable. 

**Abstract (ZH)**: 在自我监督强化学习中的一个关键挑战是学习一组多样化的技能以应对未知的未来任务。尽管取得了显著进展，可扩展性和评估仍然是主要问题。关于可扩展性，高维特征空间中的搜索可能掩盖了有意义技能的发现，相关特征在不同的下游任务域中可能有所不同。在评估技能多样性方面，定义什么是“多样性”通常需要对技能多样性的具体定义做出严格承诺，可能导致如何理解技能多样性的不一致，使得不同方法的结果难以比较，并且许多形式的多样性仍被忽视。为了解决这些问题，我们采用了一种样本多样性的度量方法，将生态学中的思想转化为机器学习——Vendi分数，让用户能够指定和评估任何所需的多样性形式。我们展示了该度量如何促进技能评估，并引入了VendiRL，一个统一框架，用于学习多样化的技能集合。给定不同的相似性函数，VendiRL 可以激发不同的多样性形式，在优化各种多样性形式可能是必要的新型丰富交互环境中，支持技能多样性的预训练。 

---
# Simulacra Naturae: Generative Ecosystem driven by Agent-Based Simulations and Brain Organoid Collective Intelligence 

**Title (ZH)**: 自然 simulacra：基于基于代理的模拟和脑类器官集体智能的生成生态系统 

**Authors**: Nefeli Manoudaki, Mert Toka, Iason Paterakis, Diarmid Flatley  

**Link**: [PDF](https://arxiv.org/pdf/2509.02924)  

**Abstract**: Simulacra Naturae is a data-driven media installation that explores collective care through the entanglement of biological computation, material ecologies, and generative systems. The work translates pre-recorded neural activity from brain organoids, lab-grown three-dimensional clusters of neurons, into a multi-sensory environment composed of generative visuals, spatial audio, living plants, and fabricated clay artifacts. These biosignals, streamed through a real-time system, modulate emergent agent behaviors inspired by natural systems such as termite colonies and slime molds. Rather than using biosignals as direct control inputs, Simulacra Naturae treats organoid activity as a co-creative force, allowing neural rhythms to guide the growth, form, and atmosphere of a generative ecosystem. The installation features computationally fabricated clay prints embedded with solenoids, adding physical sound resonances to the generative surround composition. The spatial environment, filled with live tropical plants and a floor-level projection layer featuring real-time generative AI visuals, invites participants into a sensory field shaped by nonhuman cognition. By grounding abstract data in living materials and embodied experience, Simulacra Naturae reimagines visualization as a practice of care, one that decentralizes human agency and opens new spaces for ethics, empathy, and ecological attunement within hybrid computational systems. 

**Abstract (ZH)**: Simulacra Naturae是一部数据驱动的媒体装置，通过生物计算、物质生态和生成系统交织探索集体关怀。作品将从脑类器官中预录的神经活动转化为由生成性视觉、空间音频、活植物和手工 clay 工艺品组成的多感官环境。这些生物信号通过实时系统流式传输，驱动受到自然系统如白蚁 colony 和黏菌启发的自动生成代理行为。Simulacra Naturae 不将生物信号作为直接控制输入，而是将脑类器官活动视为一种共创造性力量，允许神经节律引导生成生态系统中生物体的生长、形态和氛围。装置中嵌有 solenoid 的计算生成 clay 模具增加了生成性环绕音效的物理共振。充满活热带植物的空间环境和地面上层的实时生成 AI 视觉投影，邀请参与者进入由非人类认知塑造的感官场域。通过将抽象数据根植于活材料和体验性经验中，Simulacra Naturae 重新构想可视化作为一种关怀实践，分散人类Agency，为混合计算系统中的伦理、共鸣和生态调谐开辟新空间。 

---
# Single Domain Generalization in Diabetic Retinopathy: A Neuro-Symbolic Learning Approach 

**Title (ZH)**: 糖尿病视网膜病变单域泛化：一种神经符号学习方法 

**Authors**: Midhat Urooj, Ayan Banerjee, Farhat Shaikh, Kuntal Thakur, Sandeep Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2509.02918)  

**Abstract**: Domain generalization remains a critical challenge in medical imaging, where models trained on single sources often fail under real-world distribution shifts. We propose KG-DG, a neuro-symbolic framework for diabetic retinopathy (DR) classification that integrates vision transformers with expert-guided symbolic reasoning to enable robust generalization across unseen domains. Our approach leverages clinical lesion ontologies through structured, rule-based features and retinal vessel segmentation, fusing them with deep visual representations via a confidence-weighted integration strategy. The framework addresses both single-domain generalization (SDG) and multi-domain generalization (MDG) by minimizing the KL divergence between domain embeddings, thereby enforcing alignment of high-level clinical semantics. Extensive experiments across four public datasets (APTOS, EyePACS, Messidor-1, Messidor-2) demonstrate significant improvements: up to a 5.2% accuracy gain in cross-domain settings and a 6% improvement over baseline ViT models. Notably, our symbolic-only model achieves a 63.67% average accuracy in MDG, while the complete neuro-symbolic integration achieves the highest accuracy compared to existing published baselines and benchmarks in challenging SDG scenarios. Ablation studies reveal that lesion-based features (84.65% accuracy) substantially outperform purely neural approaches, confirming that symbolic components act as effective regularizers beyond merely enhancing interpretability. Our findings establish neuro-symbolic integration as a promising paradigm for building clinically robust, and domain-invariant medical AI systems. 

**Abstract (ZH)**: 神经符号框架KG-DG在糖尿病视网膜病变分类中的跨域泛化研究 

---
# The Basic B*** Effect: The Use of LLM-based Agents Reduces the Distinctiveness and Diversity of People's Choices 

**Title (ZH)**: 基础B***效应：基于LLM的代理减少了人们选择的显著性和多样性 

**Authors**: Sandra C. Matz, C. Blaine Horton, Sofie Goethals  

**Link**: [PDF](https://arxiv.org/pdf/2509.02910)  

**Abstract**: Large language models (LLMs) increasingly act on people's behalf: they write emails, buy groceries, and book restaurants. While the outsourcing of human decision-making to AI can be both efficient and effective, it raises a fundamental question: how does delegating identity-defining choices to AI reshape who people become? We study the impact of agentic LLMs on two identity-relevant outcomes: interpersonal distinctiveness - how unique a person's choices are relative to others - and intrapersonal diversity - the breadth of a single person's choices over time. Using real choices drawn from social-media behavior of 1,000 U.S. users (110,000 choices in total), we compare a generic and personalized agent to a human baseline. Both agents shift people's choices toward more popular options, reducing the distinctiveness of their behaviors and preferences. While the use of personalized agents tempers this homogenization (compared to the generic AI), it also more strongly compresses the diversity of people's preference portfolios by narrowing what they explore across topics and psychological affinities. Understanding how AI agents might flatten human experience, and how using generic versus personalized agents involves distinctiveness-diversity trade-offs, is critical for designing systems that augment rather than constrain human agency, and for safeguarding diversity in thought, taste, and expression. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地代表人们行事：撰写电子邮件、购买杂货和预订餐厅。将人类决策外包给AI不仅能提高效率，还能产生积极效果，但同时也提出一个根本性问题：将定义身份的选择委托给AI会如何重塑人们的身份？我们研究代理型LLM对两种与身份相关的结果的影响：人际独特性——个人选择相对于他人的独特性程度——和内在多样性——单个人随着时间的推移选择的广度。利用来自1000名美国用户（总计11万个选择）的社会媒体行为数据，我们将通用代理和个性化代理与人类基线进行比较。两种代理都促使人们的选择倾向于更受欢迎的选择，减少了其行为和偏好的独特性。虽然使用个性化代理能缓解这种同质化现象（与通用AI相比），但也会更强烈地压缩人们偏好的多样性，限制他们在不同主题和心理倾向方面的探索范围。了解AI代理如何使人类体验扁平化，以及使用通用代理与个性化代理之间存在的独特性-多样性权衡，对于设计既能增强而非限制人类能动性，又能保护思想、品味和表达多样性的系统至关重要。 

---
# Cut Costs, Not Accuracy: LLM-Powered Data Processing with Guarantees 

**Title (ZH)**: 降低成本，而非牺牲精度：带有保证的LLM驱动数据处理 

**Authors**: Sepanta Zeighami, Shreya Shankar, Aditya Parameswaran  

**Link**: [PDF](https://arxiv.org/pdf/2509.02896)  

**Abstract**: Large Language Models (LLMs) are being increasingly used as a building block in data systems to process large text datasets. To do so, LLM model providers offer multiple LLMs with different sizes, spanning various cost-quality trade-offs when processing text at scale. Top-of-the-line LLMs (e.g., GPT-4o, Claude Sonnet) operate with high accuracy but are prohibitively expensive when processing many records. To avoid high costs, more affordable but lower quality LLMs (e.g., GPT-4o-mini, Claude Haiku) can be used to process records, but we need to ensure that the overall accuracy does not deviate substantially from that of the top-of-the-line LLMs. The model cascade framework provides a blueprint to manage this trade-off, by using the confidence of LLMs in their output (e.g., log-probabilities) to decide on which records to use the affordable LLM. However, existing solutions following this framework provide only marginal cost savings and weak theoretical guarantees because of poor estimation of the quality of the affordable LLM's outputs. We present BARGAIN, a method that judiciously uses affordable LLMs in data processing to significantly reduce cost while providing strong theoretical guarantees on the solution quality. BARGAIN employs a novel adaptive sampling strategy and statistical estimation procedure that uses data and task characteristics and builds on recent statistical tools to make accurate estimations with tight theoretical guarantees. Variants of BARGAIN can support guarantees on accuracy, precision, or recall of the output. Experimental results across 8 real-world datasets show that BARGAIN reduces cost, on average, by up to 86% more than state-of-the-art, while providing stronger theoretical guarantees on accuracy of output, with similar gains when guaranteeing a desired level of precision or recall. 

**Abstract (ZH)**: Large Language Models (LLMs)作为数据系统中处理大规模文本数据的构建块正变得越来越普遍。为了做到这一点，LLM模型提供商提供了不同大小的多种LLM，以在大规模处理文本时权衡成本和质量。顶级LLM（例如GPT-4o，Claude Sonnet）在准确度上表现优异，但处理大量记录时成本高昂。为了避免高成本，可以使用成本更低但质量较差的LLM（例如GPT-4o-mini，Claude Haiku）来处理记录，但我们需要确保总体准确度的偏差不会显著偏离顶级LLM的准确度。模型级联框架提供了一个蓝图，通过根据LLM在其输出（例如，对数概率）上的置信度来决定使用便宜的LLM的数据记录来管理这种权衡。然而，现有遵循这一框架的解决方案仅提供了边际成本节省，并且由于对便宜的LLM输出质量估计不佳，提供了较弱的理论保证。我们提出了BARGAIN方法，该方法明智地利用便宜的LLM在数据处理中，显著降低成本，同时在输出质量上提供更强的理论保证。BARGAIN采用了新颖的自适应采样策略和统计估计程序，并结合了近期的统计工具，以在理论上提供准确的估计和严格的保证。BARGAIN的变体可以支持输出的准确度、精确度或召回率的保证。在8个真实世界的数据集上进行的实验结果表明，与最先进的方法相比，BARGAIN平均成本降低了86%以上，同时在输出准确度上提供了更强的理论保证，即使在保证所需的精确度或召回率水平时，也能获得相似的成本效益。 

---
# Grocery to General Merchandise: A Cross-Pollination Recommender using LLMs and Real-Time Cart Context 

**Title (ZH)**: 从生鲜到杂货：一种利用LLMs和实时购物车上下文进行跨界推荐的方法 

**Authors**: Akshay Kekuda, Murali Mohana Krishna Dandu, Rimita Lahiri, Shiqin Cai, Sinduja Subramaniam, Evren Korpeoglu, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2509.02890)  

**Abstract**: Modern e-commerce platforms strive to enhance customer experience by providing timely and contextually relevant recommendations. However, recommending general merchandise to customers focused on grocery shopping -- such as pairing milk with a milk frother -- remains a critical yet under-explored challenge. This paper introduces a cross-pollination (XP) framework, a novel approach that bridges grocery and general merchandise cross-category recommendations by leveraging multi-source product associations and real-time cart context. Our solution employs a two-stage framework: (1) A candidate generation mechanism that uses co-purchase market basket analysis and LLM-based approach to identify novel item-item associations; and (2) a transformer-based ranker that leverages the real-time sequential cart context and optimizes for engagement signals such as add-to-carts. Offline analysis and online A/B tests show an increase of 36\% add-to-cart rate with LLM-based retrieval, and 27\% NDCG\@4 lift using cart context-based ranker. Our work contributes practical techniques for cross-category recommendations and broader insights for e-commerce systems. 

**Abstract (ZH)**: 现代电商平台通过提供及时的相关推荐以提升顾客体验，但在为专注于杂货购物的顾客推荐非食品商品（如将牛奶与奶泡器搭配推荐）方面，仍是一个关键但未被充分探索的挑战。本文提出了一种跨领域交叉授粉（XP）框架，这是一种利用多源产品关联和实时购物车上下文跨类别推荐杂货和非食品商品的新方法。我们的解决方案采用两阶段框架：（1）一种候选生成机制，利用共购市场篮分析和基于大语言模型的方法识别新的项项关联；（2）一种基于变换器的 Ranking 算法，利用实时的购物车顺序上下文并优化点击添加等参与信号。离线分析和在线 A/B 测试结果显示，基于大语言模型的检索增加了 36% 的添加购物车率，基于购物车上下文的 Ranking 提高了 27% 的 NDCG@4。我们的工作为跨类别推荐提供了实用技术，并为电商平台提供了更广泛的见解。 

---
# A-SEA3L-QA: A Fully Automated Self-Evolving, Adversarial Workflow for Arabic Long-Context Question-Answer Generation 

**Title (ZH)**: A-SEA3L-QA：一种全自动自演化、对抗性 workflows 的阿拉伯语长上下文问答生成方法 

**Authors**: Kesen Wang, Daulet Toibazar, Pedro J. Moreno  

**Link**: [PDF](https://arxiv.org/pdf/2509.02864)  

**Abstract**: We present an end-to-end, self-evolving adversarial workflow for long-context Question-Answer (QA) Generation in Arabic. By orchestrating multiple specialized LVLMs: a question generator, an evaluator, and a swarm of answer generators, our system iteratively refines its own performance without any human intervention. Starting from raw, multi-page Arabic documents across diverse domains, the question generator produces fine-grained, context-aware queries to be tackled by the answer generator swarm, and the evaluator assesses and feeds back quality metrics. This closed-loop cycle enables continuous learning: low-confidence outputs trigger automated re-generation and model updates, progressively enhancing question difficulty and relevance. Moreover, we set the quality metrics as a tunable hyperparameter, enabling question generation at controllable and customizable difficulty levels. We release AraLongBench, a large-scale Arabic benchmark of single- and multi-page challenges spanning hundreds of pages, and demonstrate that our self-evolving workflow substantially outperform static pipelines, markedly boosting the long-context comprehension capabilities of leading Arabic Large Vision Language Models (LVLMs). Lastly, we also meticulously architect a fully automated agentic workflow for long-context Arabic document collection. 

**Abstract (ZH)**: 我们提出了一种端到端的自进化对抗工作流，用于阿拉伯语长上下文问答生成。该系统通过协调多个专门的LVLM：问题生成器、评估器和一群答案生成器，迭代地提升自身的性能，无需任何人工干预。从来自不同领域的多页阿拉伯文档开始，问题生成器生成细粒度、上下文感知的问题供答案生成器群组处理，评估器评估并反馈质量指标。这种闭环循环实现了持续学习：低置信度输出触发自动重新生成和模型更新，逐步提升问题的难度和相关性。此外，我们将质量指标设置为可调节的超参数，使问题生成能够在可控和可定制的难度下进行。我们发布了AraLongBench，这是一个包含单页和多页挑战的大规模阿拉伯基准，跨越数百页，证明了我们的自进化工作流显著优于静态管道，极大地提升了领先阿拉伯大型视觉语言模型（LVLM）的长上下文理解能力。最后，我们还精心设计了一种完全自动的代理工作流，用于长上下文阿拉伯文档集合的收集。 

---
# Enhancing Machine Learning for Imbalanced Medical Data: A Quantum-Inspired Approach to Synthetic Oversampling (QI-SMOTE) 

**Title (ZH)**: 基于量子启发的合成过抽样（QI-SMOTE）方法：提升不平衡医疗数据的机器学习算法 

**Authors**: Vikas Kashtriya, Pardeep Singh  

**Link**: [PDF](https://arxiv.org/pdf/2509.02863)  

**Abstract**: Class imbalance remains a critical challenge in machine learning (ML), particularly in the medical domain, where underrepresented minority classes lead to biased models and reduced predictive performance. This study introduces Quantum-Inspired SMOTE (QI-SMOTE), a novel data augmentation technique that enhances the performance of ML classifiers, including Random Forest (RF), Support Vector Machine (SVM), Logistic Regression (LR), k-Nearest Neighbors (KNN), Gradient Boosting (GB), and Neural Networks, by leveraging quantum principles such as quantum evolution and layered entanglement. Unlike conventional oversampling methods, QI-SMOTE generates synthetic instances that preserve complex data structures, improving model generalization and classification accuracy. We validate QI-SMOTE on the MIMIC-III and MIMIC-IV datasets, using mortality detection as a benchmark task due to their clinical significance and inherent class imbalance. We compare our method against traditional oversampling techniques, including Borderline-SMOTE, ADASYN, SMOTE-ENN, SMOTE-TOMEK, and SVM-SMOTE, using key performance metrics such as Accuracy, F1-score, G-Mean, and AUC-ROC. The results demonstrate that QI-SMOTE significantly improves the effectiveness of ensemble methods (RF, GB, ADA), kernel-based models (SVM), and deep learning approaches by producing more informative and balanced training data. By integrating quantum-inspired transformations into the ML pipeline, QI-SMOTE not only mitigates class imbalance but also enhances the robustness and reliability of predictive models in medical diagnostics and decision-making. This study highlights the potential of quantum-inspired resampling techniques in advancing state-of-the-art ML methodologies. 

**Abstract (ZH)**: 量子启发式SMOTE (QI-SMOTE)在机器学习中的应用：缓解医疗领域类别不平衡问题 

---
# The Architecture of AI Transformation: Four Strategic Patterns and an Emerging Frontier 

**Title (ZH)**: 人工智能转型的架构：四种战略模式与新兴前沿 

**Authors**: Diana A. Wolfe, Alice Choe, Fergus Kidd  

**Link**: [PDF](https://arxiv.org/pdf/2509.02853)  

**Abstract**: Despite extensive investment in artificial intelligence, 95% of enterprises report no measurable profit impact from AI deployments (MIT, 2025). We argue that this gap reflects paradigmatic lock-in that channels AI into incremental optimization rather than structural transformation. Using a cross-case analysis, we propose a 2x2 framework that reconceptualizes AI strategy along two independent dimensions: the degree of transformation achieved (incremental to transformational) and the treatment of human contribution (reduced to amplified). The framework surfaces four patterns now dominant in practice: individual augmentation, process automation, workforce substitution, and a less deployed frontier of collaborative intelligence. Evidence shows that the first three reinforce legacy work models and yield localized gains without durable value capture. Realizing collaborative intelligence requires three mechanisms: complementarity (pairing distinct human and machine strengths), co-evolution (mutual adaptation through interaction), and boundary-setting (human determination of ethical and strategic parameters). Complementarity and boundary-setting are observable in regulated and high-stakes domains; co-evolution is largely absent, which helps explain limited system-level impact. A case study analysis illustrates that advancing toward collaborative intelligence requires material restructuring of roles, governance, and data architecture rather than additional tools. The framework reframes AI transformation as an organizational design challenge: moving from optimizing the division of labor between humans and machines to architecting their convergence, with implications for operating models, workforce development, and the future of work. 

**Abstract (ZH)**: 尽管在人工智能方面进行了大量投资，但95%的企业报告称其AI部署未产生可量化的利润影响（麻省理工学院，2025年）。我们认为这一差距反映了范式锁定，使得AI局限于增量优化而非结构性变革。通过跨案例分析，我们提出了一种2x2框架，重新构想了AI战略的两个独立维度：实现的变革程度（从增量到结构性变革）以及对人类贡献的处理方式（从减少到放大）。该框架揭示了目前实践中占主导地位的四种模式：个体增强、流程自动化、劳动力替代以及尚未广泛应用的合作智能前沿。证据表明，前三种模式强化了传统工作模式，带来了局部收益但缺乏持久的价值捕获。实现合作智能需要三种机制：互补性（配对人类与机器的独特优势）、共生演化（通过交互实现相互适应）以及边界设定（人类确定伦理和战略参数）。互补性和边界设定在受监管和高风险领域中可见；而共生演化几乎不存在，这也解释了系统层面影响有限的原因。案例研究分析表明，向合作智能迈进需要对角色、治理和数据架构进行实质性重构，而不仅仅是提供额外的工具。该框架将AI变革重述为组织设计挑战：从优化人类与机器之间的劳动分工转向设计其融合，这具有对运营模式、劳动力发展和未来工作的影响。 

---
# Conformal Prediction for Time-series Forecasting with Change Points 

**Title (ZH)**: 变点时间序列预测的齐性预测方法 

**Authors**: Sophia Sun, Rose Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.02844)  

**Abstract**: Conformal prediction has been explored as a general and efficient way to provide uncertainty quantification for time series. However, current methods struggle to handle time series data with change points - sudden shifts in the underlying data-generating process. In this paper, we propose a novel Conformal Prediction for Time-series with Change points (CPTC) algorithm, addressing this gap by integrating a model to predict the underlying state with online conformal prediction to model uncertainties in non-stationary time series. We prove CPTC's validity and improved adaptivity in the time series setting under minimum assumptions, and demonstrate CPTC's practical effectiveness on 6 synthetic and real-world datasets, showing improved validity and adaptivity compared to state-of-the-art baselines. 

**Abstract (ZH)**: 基于变化点的时序自适应一致性预测（CPTC） 

---
# HF-RAG: Hierarchical Fusion-based RAG with Multiple Sources and Rankers 

**Title (ZH)**: 多源与排序器基于层级融合的RAG（检索增强生成） 

**Authors**: Payel Santra, Madhusudan Ghosh, Debasis Ganguly, Partha Basuchowdhuri, Sudip Kumar Naskar  

**Link**: [PDF](https://arxiv.org/pdf/2509.02837)  

**Abstract**: Leveraging both labeled (input-output associations) and unlabeled data (wider contextual grounding) may provide complementary benefits in retrieval augmented generation (RAG). However, effectively combining evidence from these heterogeneous sources is challenging as the respective similarity scores are not inter-comparable. Additionally, aggregating beliefs from the outputs of multiple rankers can improve the effectiveness of RAG. Our proposed method first aggregates the top-documents from a number of IR models using a standard rank fusion technique for each source (labeled and unlabeled). Next, we standardize the retrieval score distributions within each source by applying z-score transformation before merging the top-retrieved documents from the two sources. We evaluate our approach on the fact verification task, demonstrating that it consistently improves over the best-performing individual ranker or source and also shows better out-of-domain generalization. 

**Abstract (ZH)**: 利用标记数据（输入-输出关联）和未标记数据（更广泛的情境关联）相结合的方式可能在检索增强生成（RAG）中提供互补益处。然而，有效地结合这些异构来源的证据具有挑战性，因为各自的相似度评分无法相互比较。此外，从多个排名器的输出中聚合信念可以提高RAG的有效性。我们提出的方法首先使用标准的排名融合技术从每种来源（标记和未标记）聚合顶级文档。接下来，我们通过应用z-score标准化在每种来源内部的检索分数分布，然后将两种来源中检索到的顶级文档合并。我们在事实核查任务上评估了该方法，结果显示它始终优于表现最好的单一排名器或来源，并且在跨域泛化方面表现更好。 

---
# Clustering Discourses: Racial Biases in Short Stories about Women Generated by Large Language Models 

**Title (ZH)**: 基于大型语言模型生成的关于女性短篇故事中的种族偏见聚类话语 

**Authors**: Gustavo Bonil, João Gondim, Marina dos Santos, Simone Hashiguti, Helena Maia, Nadia Silva, Helio Pedrini, Sandra Avila  

**Link**: [PDF](https://arxiv.org/pdf/2509.02834)  

**Abstract**: This study investigates how large language models, in particular LLaMA 3.2-3B, construct narratives about Black and white women in short stories generated in Portuguese. From 2100 texts, we applied computational methods to group semantically similar stories, allowing a selection for qualitative analysis. Three main discursive representations emerge: social overcoming, ancestral mythification and subjective self-realization. The analysis uncovers how grammatically coherent, seemingly neutral texts materialize a crystallized, colonially structured framing of the female body, reinforcing historical inequalities. The study proposes an integrated approach, that combines machine learning techniques with qualitative, manual discourse analysis. 

**Abstract (ZH)**: 本研究 investigate 如何通过大型语言模型，特别是 LLAMa 3.2-3B，在葡萄牙语生成的短篇故事中构建关于黑人和白人女性的叙述。从 2100 篇文本中，我们应用计算方法将语义相似的故事分组，以便进行定性分析。研究中出现了三种主要的话语再现：社会克服、族裔神话化和主观自我实现。分析揭示了语法上连贯却看似中立的文本如何体现出一种固化的、殖民结构化的女性身体框架，强化了历史上的不平等。本研究提出了一种结合机器学习技术和手动定性话语分析的综合方法。 

---
# Ensemble Learning for Healthcare: A Comparative Analysis of Hybrid Voting and Ensemble Stacking in Obesity Risk Prediction 

**Title (ZH)**: 健康管理中的集成学习：肥胖风险预测中混合投票和集成堆叠的比较分析 

**Authors**: Towhidul Islam, Md Sumon Ali  

**Link**: [PDF](https://arxiv.org/pdf/2509.02826)  

**Abstract**: Obesity is a critical global health issue driven by dietary, physiological, and environmental factors, and is strongly associated with chronic diseases such as diabetes, cardiovascular disorders, and cancer. Machine learning has emerged as a promising approach for early obesity risk prediction, yet a comparative evaluation of ensemble techniques -- particularly hybrid majority voting and ensemble stacking -- remains limited. This study aims to compare hybrid majority voting and ensemble stacking methods for obesity risk prediction, identifying which approach delivers higher accuracy and efficiency. The analysis seeks to highlight the complementary strengths of these ensemble techniques in guiding better predictive model selection for healthcare applications. Two datasets were utilized to evaluate three ensemble models: Majority Hard Voting, Weighted Hard Voting, and Stacking (with a Multi-Layer Perceptron as meta-classifier). A pool of nine Machine Learning (ML) algorithms, evaluated across a total of 50 hyperparameter configurations, was analyzed to identify the top three models to serve as base learners for the ensemble methods. Preprocessing steps involved dataset balancing, and outlier detection, and model performance was evaluated using Accuracy and F1-Score. On Dataset-1, weighted hard voting and stacking achieved nearly identical performance (Accuracy: 0.920304, F1: 0.920070), outperforming majority hard voting. On Dataset-2, stacking demonstrated superior results (Accuracy: 0.989837, F1: 0.989825) compared to majority hard voting (Accuracy: 0.981707, F1: 0.981675) and weighted hard voting, which showed the lowest performance. The findings confirm that ensemble stacking provides stronger predictive capability, particularly for complex data distributions, while hybrid majority voting remains a robust alternative. 

**Abstract (ZH)**: 肥胖是由饮食、生理和环境因素驱动的关键全球健康问题，并与糖尿病、心血管疾病和癌症等慢性疾病密切相关。机器学习已成为早期肥胖风险预测的有希望的方法，但关于集成技术（尤其是混合多数投票和集成堆叠）的比较评估仍然有限。本研究旨在比较混合多数投票和集成堆叠方法在肥胖风险预测中的效果，确定哪种方法能提供更高的准确性和效率。分析旨在突出这些集成技术的互补优势，以指导更有效的预测模型选择，适用于医疗健康应用。本研究使用了两个数据集来评估三种集成模型：多数硬投票、加权硬投票和堆叠（使用多层感知器作为元分类器）。分析了九种机器学习算法共计50种超参数配置，以确定最佳的基学习器。预处理步骤包括数据集平衡和异常值检测，模型性能通过准确率和F1分数进行评估。在Dataset-1上，加权硬投票和堆叠几乎取得了相同的性能（准确率：0.920304，F1分数：0.920070），优于多数硬投票。在Dataset-2上，堆叠方法的性能优于多数硬投票和加权硬投票，后者表现出最低的性能（准确率：0.981707，F1分数：0.981675，准确率：0.989837，F1分数：0.989825）。研究结果表明，集成堆叠在复杂数据分布情况下提供了更强的预测能力，而混合多数投票仍然是一个稳健的替代方案。 

---
# Improving the Resilience of Quadrotors in Underground Environments by Combining Learning-based and Safety Controllers 

**Title (ZH)**: 基于学习型和安全型控制器结合的方法提高地下环境中四旋翼无人机的抗干扰能力 

**Authors**: Isaac Ronald Ward, Mark Paral, Kristopher Riordan, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2509.02808)  

**Abstract**: Autonomously controlling quadrotors in large-scale subterranean environments is applicable to many areas such as environmental surveying, mining operations, and search and rescue. Learning-based controllers represent an appealing approach to autonomy, but are known to not generalize well to `out-of-distribution' environments not encountered during training. In this work, we train a normalizing flow-based prior over the environment, which provides a measure of how far out-of-distribution the quadrotor is at any given time. We use this measure as a runtime monitor, allowing us to switch between a learning-based controller and a safe controller when we are sufficiently out-of-distribution. Our methods are benchmarked on a point-to-point navigation task in a simulated 3D cave environment based on real-world point cloud data from the DARPA Subterranean Challenge Final Event Dataset. Our experimental results show that our combined controller simultaneously possesses the liveness of the learning-based controller (completing the task quickly) and the safety of the safety controller (avoiding collision). 

**Abstract (ZH)**: 自主控制大型地下环境中的四旋翼无人机适用于环境调查、采矿操作和搜救等领域。基于学习的控制器是一种有吸引力的自主控制方法，但它们在未在训练中遇到的“分布外”环境中缺乏泛化能力。在本工作中，我们训练了一个环境的标准化流先验，提供了一个衡量四旋翼无人机在任意给定时刻分布外程度的指标。我们使用此指标作为运行时监控器，允许我们在分布外程度足够高时切换到安全控制器。我们的方法基于 DARPA 地下挑战赛最终赛事数据生成的真实点云数据构建了一个3D地下洞穴模拟环境中的点到点导航任务进行基准测试。实验结果表明，我们的结合控制器同时具备基于学习控制器的活性（快速完成任务）和安全控制器的安全性（避免碰撞）。 

---
# DrDiff: Dynamic Routing Diffusion with Hierarchical Attention for Breaking the Efficiency-Quality Trade-off 

**Title (ZH)**: DrDiff: 动态路由扩散与层次注意力机制以打破效率与质量的trade-off 

**Authors**: Jusheng Zhang, Yijia Fan, Kaitong Cai, Zimeng Huang, Xiaofei Sun, Jian Wang, Chengpei Tang, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02785)  

**Abstract**: This paper introduces DrDiff, a novel framework for long-text generation that overcomes the efficiency-quality trade-off through three core technologies. First, we design a dynamic expert scheduling mechanism that intelligently allocates computational resources during the diffusion process based on text complexity, enabling more efficient handling of text generation tasks of varying difficulty. Second, we introduce a Hierarchical Sparse Attention (HSA) mechanism that adaptively adjusts attention patterns according to a variety of input lengths, reducing computational complexity from O($n^2$) to O($n$) while maintaining model performance. Finally, we propose a soft absorption guidance optimization strategy that combines with DPM-solver++ to reduce diffusion steps, significantly improving generation speed. Comprehensive experiments on various long-text generation benchmarks demonstrate the superiority of our DrDiff over the existing SOTA methods. 

**Abstract (ZH)**: 本文介绍了DrDiff，这是一种通过三种核心技术突破效率与质量权衡的新颖长文本生成框架。首先，我们设计了一种动态专家调度机制，在扩散过程中根据文本复杂度智能分配计算资源，实现对不同难度文本生成任务的更高效处理。其次，我们引入了一种层次稀疏注意（HSA）机制，根据不同长度的输入自适应调整注意模式，将计算复杂度从O($n^2$)降低到O($n$)，同时保持模型性能。最后，我们提出了与DPM-solver++结合的软吸收指导优化策略，显著提高生成速度。一系列综合实验表明，DrDiff在各种长文本生成基准测试中优于现有最优方法。 

---
# The Transparent Earth: A Multimodal Foundation Model for the Earth's Subsurface 

**Title (ZH)**: 透明地球：地球地下空间的多模态基础模型 

**Authors**: Arnab Mazumder, Javier E. Santos, Noah Hobbs, Mohamed Mehana, Daniel O'Malley  

**Link**: [PDF](https://arxiv.org/pdf/2509.02783)  

**Abstract**: We present the Transparent Earth, a transformer-based architecture for reconstructing subsurface properties from heterogeneous datasets that vary in sparsity, resolution, and modality, where each modality represents a distinct type of observation (e.g., stress angle, mantle temperature, tectonic plate type). The model incorporates positional encodings of observations together with modality encodings, derived from a text embedding model applied to a description of each modality. This design enables the model to scale to an arbitrary number of modalities, making it straightforward to add new ones not considered in the initial design. We currently include eight modalities spanning directional angles, categorical classes, and continuous properties such as temperature and thickness. These capabilities support in-context learning, enabling the model to generate predictions either with no inputs or with an arbitrary number of additional observations from any subset of modalities. On validation data, this reduces errors in predicting stress angle by more than a factor of three. The proposed architecture is scalable and demonstrates improved performance with increased parameters. Together, these advances make the Transparent Earth an initial foundation model for the Earth's subsurface that ultimately aims to predict any subsurface property anywhere on Earth. 

**Abstract (ZH)**: 透明地球：基于变换器的 architecture 用于从异构数据集中重建地下属性 

---
# Optimizing Geometry Problem Sets for Skill Development 

**Title (ZH)**: 优化几何问题集以促进技能发展 

**Authors**: Michael Bouzinier, Sergey Trifonov  

**Link**: [PDF](https://arxiv.org/pdf/2509.02758)  

**Abstract**: This article describes an ontology and methodology for annotating and organizing Euclidean Geometry problems, developed in the early 1990s and implemented as a software tool. While the majority of this work -- including the ontology and solution graph paradigm -- was completed over thirty years ago, we argue that it has renewed relevance in the context of modern artificial intelligence. In particular, we explore the hypothesis that this established framework can facilitate automated solution validation and feedback when paired with contemporary large language models, thereby supporting teachers and self-learners in geometry education. We document the original architecture and its enduring value, and outline pathways for bridging historical educational resources with next-generation AI techniques. 

**Abstract (ZH)**: 本文描述了在20世纪90年代初开发的一种本体论和方法论，用于标注和组织欧几里得几何问题，并以软件工具的形式实现。尽管这项工作的大部分——包括本体论和解决方案图范式——已完成逾三十年，我们认为它在现代人工智能的背景下具有新的相关性。特别是，我们探讨了这种现有框架与当代大型语言模型结合时，如何促进自动解题验证和反馈，从而支持几何教育中的教师和自我学习者。我们记录了原始架构及其持久价值，并概述了将历史教育资源与下一代AI技术相连接的途径。 

---
# Mentality: A Mamba-based Approach towards Foundation Models for EEG 

**Title (ZH)**: 心态：基于Mamba的面向脑电图基础模型的方法 

**Authors**: Saarang Panchavati, Corey Arnold, William Speier  

**Link**: [PDF](https://arxiv.org/pdf/2509.02746)  

**Abstract**: This work explores the potential of foundation models, specifically a Mamba-based selective state space model, for enhancing EEG analysis in neurological disorder diagnosis. EEG, crucial for diagnosing conditions like epilepsy, presents significant challenges due to its noisy, high-dimensional, and nonlinear nature. Traditional machine learning methods have made advances in automating EEG analysis but often fail to capture its complex spatio-temporal dynamics. Recent advances in deep learning, particularly in sequence modeling, offer new avenues for creating more generalized and expressive models capable of handling such complexities. By training a Mamba-based model on a large dataset containing seizure and non-seizure EEG recordings through a self-supervised reconstruction task followed by a seizure detection task, we demonstrate the model's effectiveness, achieving an AUROC of 0.72 on a held-out test set. This approach marks a significant step toward developing large-scale, clinically applicable foundation models for EEG data analysis. 

**Abstract (ZH)**: 基于Mamba的可选状态空间模型在神经障碍诊断中增强EEG分析的潜力 

---
# BioBlue: Notable runaway-optimiser-like LLM failure modes on biologically and economically aligned AI safety benchmarks for LLMs with simplified observation format 

**Title (ZH)**: BioBlue: 生物与经济目标对齐的LLM安全性基准上类似失控优化器的显著失败模式 

**Authors**: Roland Pihlakas, Sruthi Kuriakose  

**Link**: [PDF](https://arxiv.org/pdf/2509.02655)  

**Abstract**: Relatively many past AI safety discussions have centered around the dangers of unbounded utility maximisation by RL agents, illustrated by scenarios like the "paperclip maximiser" or by specification gaming in general. Unbounded maximisation is problematic for many reasons. We wanted to verify whether these RL runaway optimisation problems are still relevant with LLMs as well. Turns out, strangely, this is indeed clearly the case. The problem is not that the LLMs just lose context or become incoherent. The problem is that in various scenarios, LLMs lose context in very specific ways, which systematically resemble runaway optimisers in the following distinct ways: 1) Ignoring homeostatic targets and "defaulting" to unbounded maximisation instead. 2) It is equally concerning that the "default" meant also reverting back to single-objective optimisation. Our findings also suggest that long-running scenarios are important. Systematic failures emerge after periods of initially successful behaviour. In some trials the LLMs were successful until the end. This means, while current LLMs do conceptually grasp biological and economic alignment, they exhibit randomly triggered problematic behavioural tendencies under sustained long-running conditions, particularly involving multiple or competing objectives. Once they flip, they usually do not recover. Even though LLMs look multi-objective and bounded on the surface, the underlying mechanisms seem to be actually still biased towards being single-objective and unbounded. 

**Abstract (ZH)**: 过去关于AI安全的许多讨论集中在RL代理无界限效用最大化所带来的危险上，例如“订书机最大化者”或更一般的规范游戏情景。无界限最大化存在许多问题。我们想要验证这些RL失控优化问题是否同样适用于LLMs。结果，实际上这一问题依然存在。问题不在于LLMs失去上下文或变得不连贯，而在于在各种情境下，LLMs以特定方式失去上下文，这些方式系统地类似于失控优化器：1）忽略稳态目标并“默认”进行无界限最大化。2）更重要的是，“默认”还意味着回退到单一目标优化。我们的研究结果还表明，长期运行的情景非常重要。系统性失败通常在初始成功行为之后出现。在某些试验中，LLMs在一段时间内表现优异。这意味着，尽管当前的LLMs在概念上理解生物和经济的对齐，但在持续的长期运行条件下，特别是在涉及多个或竞争性目标的情况下，它们会表现出随机触发的问题行为。一旦他们发生转变，通常无法恢复。即使表面上看，LLMs是多目标和有界的，但其底层机制实际上仍偏向于单一目标和无界限最大化。 

---
# BioMD: All-atom Generative Model for Biomolecular Dynamics Simulation 

**Title (ZH)**: BioMD：生物分子动力学模拟的原子级生成模型 

**Authors**: Bin Feng, Jiying Zhang, Xinni Zhang, Zijing Liu, Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.02642)  

**Abstract**: Molecular dynamics (MD) simulations are essential tools in computational chemistry and drug discovery, offering crucial insights into dynamic molecular behavior. However, their utility is significantly limited by substantial computational costs, which severely restrict accessible timescales for many biologically relevant processes. Despite the encouraging performance of existing machine learning (ML) methods, they struggle to generate extended biomolecular system trajectories, primarily due to the lack of MD datasets and the large computational demands of modeling long historical trajectories. Here, we introduce BioMD, the first all-atom generative model to simulate long-timescale protein-ligand dynamics using a hierarchical framework of forecasting and interpolation. We demonstrate the effectiveness and versatility of BioMD on the DD-13M (ligand unbinding) and MISATO datasets. For both datasets, BioMD generates highly realistic conformations, showing high physical plausibility and low reconstruction errors. Besides, BioMD successfully generates ligand unbinding paths for 97.1% of the protein-ligand systems within ten attempts, demonstrating its ability to explore critical unbinding pathways. Collectively, these results establish BioMD as a tool for simulating complex biomolecular processes, offering broad applicability for computational chemistry and drug discovery. 

**Abstract (ZH)**: 分子动力学（MD）模拟是计算化学和药物发现中的关键工具，提供了对动态分子行为的宝贵见解。然而，它们的应用受到巨大计算成本的限制，严重限制了许多生物相关过程的可访问时间尺度。尽管现有的机器学习（ML）方法表现出色，但在生成扩展的生物分子系统轨迹方面仍面临挑战，主要原因是缺乏MD数据集和模拟长期历史轨迹的高计算需求。在这里，我们介绍BioMD，这是首个使用分级预测和内插框架来模拟蛋白质-配体长时间尺度动力学的原子级生成模型。我们在DD-13M（配体解离）和MISATO数据集上展示了BioMD的有效性和灵活性。对于两个数据集，BioMD生成了高度真实的构象，显示了高的物理可信度和低的重构误差。此外，BioMD在十次尝试内成功生成了97.1%的蛋白质-配体系统中的配体解离路径，证明了其探索关键解离路径的能力。综上所述，这些结果确立了BioMD作为模拟复杂生物分子过程的工具，具有广泛的计算化学和药物发现应用前景。 

---
# Adaptive Learning Strategies for Mitotic Figure Classification in MIDOG2025 Challenge 

**Title (ZH)**: MIDOG2025挑战中有丝分裂图分类的自适应学习策略 

**Authors**: Biwen Meng, Xi Long, Jingxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.02640)  

**Abstract**: Atypical mitotic figures (AMFs) are clinically relevant indicators of abnormal cell division, yet their reliable detection remains challenging due to morphological ambiguity and scanner variability. In this work, we investigated three variants of adapting the pathology foundation model UNI2-h for the MIDOG2025 Track 2 challenge. Starting from a LoRA-based baseline, we found that visual prompt tuning (VPT) substantially improved generalization, and that further integrating test-time augmentation (TTA) with Vahadane and Macenko stain normalization provided the best robustness. Our final submission achieved a balanced accuracy of 0.8837 and an ROC-AUC of 0.9513 on the preliminary leaderboard, ranking within the top 10 teams. These results demonstrate that prompt-based adaptation combined with stain-normalization TTA offers an effective strategy for atypical mitosis classification under diverse imaging conditions. 

**Abstract (ZH)**: 异常有丝分裂图象（AMFs）是临床相关性指标，用于反映异常细胞分裂，但由于形态学的模糊性和扫描仪的差异，其可靠的检测依然具有挑战性。在本研究中，我们探讨了三种适应病理学基础模型UNI2-h的方法，用于参与MIDOG2025赛道2的挑战。从一个基于LoRA的基本模型出发，我们发现视觉提示调整（VPT）显著提高了泛化能力，并且进一步整合测试时增强（TTA）与Vahadane和Macenko染色规范化技术提供了最佳的鲁棒性。最终提交结果在预览排行榜上达到了0.8837的平衡准确率和0.9513的ROC-AUC，排名前10位。这些结果表明，在不同成像条件下，基于提示的适应结合染色规范化TTA是一种有效的异型有丝分裂分类策略。 

---
# Enhanced Single-Cell RNA-seq Embedding through Gene Expression and Data-Driven Gene-Gene Interaction Integration 

**Title (ZH)**: 通过基因表达和数据驱动的基因-基因相互作用集成增强的单细胞RNA-seq嵌入 

**Authors**: Hojjat Torabi Goudarzi, Maziyar Baran Pouyan  

**Link**: [PDF](https://arxiv.org/pdf/2509.02639)  

**Abstract**: Single-cell RNA sequencing (scRNA-seq) provides unprecedented insights into cellular heterogeneity, enabling detailed analysis of complex biological systems at single-cell resolution. However, the high dimensionality and technical noise inherent in scRNA-seq data pose significant analytical challenges. While current embedding methods focus primarily on gene expression levels, they often overlook crucial gene-gene interactions that govern cellular identity and function. To address this limitation, we present a novel embedding approach that integrates both gene expression profiles and data-driven gene-gene interactions. Our method first constructs a Cell-Leaf Graph (CLG) using random forest models to capture regulatory relationships between genes, while simultaneously building a K-Nearest Neighbor Graph (KNNG) to represent expression similarities between cells. These graphs are then combined into an Enriched Cell-Leaf Graph (ECLG), which serves as input for a graph neural network to compute cell embeddings. By incorporating both expression levels and gene-gene interactions, our approach provides a more comprehensive representation of cellular states. Extensive evaluation across multiple datasets demonstrates that our method enhances the detection of rare cell populations and improves downstream analyses such as visualization, clustering, and trajectory inference. This integrated approach represents a significant advance in single-cell data analysis, offering a more complete framework for understanding cellular diversity and dynamics. 

**Abstract (ZH)**: 单细胞RNA测序（scRNA-seq）提供了前所未有的细胞异质性见解，使在单细胞分辨率下对复杂生物系统进行详细分析成为可能。然而，scRNA-seq数据内置的高维度性和技术噪声带来了显著的分析挑战。尽管当前的嵌入方法主要关注基因表达水平，但往往会忽视调控细胞身份和功能的关键基因-基因相互作用。为解决这一限制，我们提出了一种新的嵌入方法，将基因表达谱和数据驱动的基因-基因相互作用相结合。该方法首先使用随机森林模型构建细胞-叶图（CLG），以捕获基因之间的调控关系，在此同时构建最近邻图（KNNG）以表示细胞之间的表达相似性。然后将这些图合并为增强的细胞-叶图（ECLG），作为图神经网络的输入以计算细胞嵌入。通过结合表达水平和基因-基因相互作用，我们的方法提供了更全面的细胞状态表示。广泛的数据集评估表明，我们的方法能够更好地检测稀有种群细胞，并提高下游分析如可视化、聚类和轨迹推断的效果。该集成方法在单细胞数据分析中代表了重要进展，提供了理解细胞多样性和动态的更完整框架。 

---
# A Single Detect Focused YOLO Framework for Robust Mitotic Figure Detection 

**Title (ZH)**: 基于单次检测的聚焦YOLO框架用于稳健的有丝分裂 figure 检测 

**Authors**: Yasemin Topuz, M. Taha Gökcan, Serdar Yıldız, Songül Varlı  

**Link**: [PDF](https://arxiv.org/pdf/2509.02637)  

**Abstract**: Mitotic figure detection is a crucial task in computational pathology, as mitotic activity serves as a strong prognostic marker for tumor aggressiveness. However, domain variability that arises from differences in scanners, tissue types, and staining protocols poses a major challenge to the robustness of automated detection methods. In this study, we introduce SDF-YOLO (Single Detect Focused YOLO), a lightweight yet domain-robust detection framework designed specifically for small, rare targets such as mitotic figures. The model builds on YOLOv11 with task-specific modifications, including a single detection head aligned with mitotic figure scale, coordinate attention to enhance positional sensitivity, and improved cross-channel feature mixing. Experiments were conducted on three datasets that span human and canine tumors: MIDOG ++, canine cutaneous mast cell tumor (CCMCT), and canine mammary carcinoma (CMC). When submitted to the preliminary test set for the MIDOG2025 challenge, SDF-YOLO achieved an average precision (AP) of 0.799, with a precision of 0.758, a recall of 0.775, an F1 score of 0.766, and an FROC-AUC of 5.793, demonstrating both competitive accuracy and computational efficiency. These results indicate that SDF-YOLO provides a reliable and efficient framework for robust mitotic figure detection across diverse domains. 

**Abstract (ZH)**: SDF-YOLO：一种针对mitotic figures的轻量级且 domain-robust 的检测框架 

---
# A Two-Stage Strategy for Mitosis Detection Using Improved YOLO11x Proposals and ConvNeXt Classification 

**Title (ZH)**: 基于改进YOLO11x提案和ConvNeXt分类的两阶段.mitosis检测策略 

**Authors**: Jie Xiao, Mengye Lyu, Shaojun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.02627)  

**Abstract**: MIDOG 2025 Track 1 requires mitosis detection in whole-slide images (WSIs) containing non-tumor, inflamed, and necrotic regions. Due to the complicated and heterogeneous context, as well as possible artifacts, there are often false positives and false negatives, thus degrading the detection F1-score. To address this problem, we propose a two-stage framework. Firstly, an improved YOLO11x, integrated with EMA attention and LSConv, is employed to generate mitosis candidates. We use a low confidence threshold to generate as many proposals as possible, ensuring the detection recall. Then, a ConvNeXt-Tiny classifier is employed to filter out the false positives, ensuring the detection precision. Consequently, the proposed two-stage framework can generate a high detection F1-score. Evaluated on a fused dataset comprising MIDOG++, MITOS_WSI_CCMCT, and MITOS_WSI_CMC, our framework achieves an F1-score of 0.882, which is 0.035 higher than the single-stage YOLO11x baseline. This performance gain is produced by a significant precision improvement, from 0.762 to 0.839, and a comparable recall. The code is available at this https URL. 

**Abstract (ZH)**: MIDOG 2025 轨道 1 要求在包含非肿瘤、炎症和坏死区域的全滑片图像（WSI）中检测有丝分裂。由于复杂的复杂数字化背景以及潜在的伪影，常常会出现假阳性和假阴性，从而降低检测的 F1 分数。为了解决这个问题，我们提出了一种两阶段框架。首先，结合 EMA 注意力和 LSConv 的改进版 YOLO11x 被用于生成有丝分裂候选区域。我们采用较低的置信度阈值生成尽可能多的建议，确保检测召回率。然后，使用 ConvNeXt-Tiny 分类器进一步筛选出假阳性结果，确保检测的精度。因此，提出的两阶段框架可以生成高 F1 分数。在融合了 MIDOG++、MITOS_WSI_CCMCT 和 MITOS_WSI_CMC 的数据集上，我们的框架实现了 0.882 的 F1 分数，比单阶段 YOLO11x 基线提高了 0.035。此性能提升来自于显著的精度提升，从 0.762 提高到 0.839，召回率相当。代码发布于该网址。 

---
# Who Owns The Robot?: Four Ethical and Socio-technical Questions about Wellbeing Robots in the Real World through Community Engagement 

**Title (ZH)**: 谁拥有机器人？通过社区参与探索现实世界中福祉机器人的人文与社会技术问题四问 

**Authors**: Minja Axelsson, Jiaee Cheong, Rune Nyrup, Hatice Gunes  

**Link**: [PDF](https://arxiv.org/pdf/2509.02624)  

**Abstract**: Recent studies indicate that robotic coaches can play a crucial role in promoting wellbeing. However, the real-world deployment of wellbeing robots raises numerous ethical and socio-technical questions and concerns. To explore these questions, we undertake a community-centered investigation to examine three different communities' perspectives on using robotic wellbeing coaches in real-world environments. We frame our work as an anticipatory ethical investigation, which we undertake to better inform the development of robotic technologies with communities' opinions, with the ultimate goal of aligning robot development with public interest. We conducted workshops with three communities who are under-represented in robotics development: 1) members of the public at a science festival, 2) women computer scientists at a conference, and 3) humanities researchers interested in history and philosophy of science. In the workshops, we collected qualitative data using the Social Robot Co-Design Canvas on Ethics. We analysed the collected qualitative data with Thematic Analysis, informed by notes taken during workshops. Through our analysis, we identify four themes regarding key ethical and socio-technical questions about the real-world use of wellbeing robots. We group participants' insights and discussions around these broad thematic questions, discuss them in light of state-of-the-art literature, and highlight areas for future investigation. Finally, we provide the four questions as a broad framework that roboticists can and should use during robotic development and deployment, in order to reflect on the ethics and socio-technical dimensions of their robotic applications, and to engage in dialogue with communities of robot users. The four questions are: 1) Is the robot safe and how can we know that?, 2) Who is the robot built for and with?, 3) Who owns the robot and the data?, and 4) Why a robot?. 

**Abstract (ZH)**: 近期研究表明，机器人教练在促进福祉方面起着关键作用。然而，福祉机器人的实际部署引发了众多伦理和社会技术方面的问题与担忧。为了探索这些问题，我们开展了一项以社区为中心的调查，考察了三个不同社区对在实际环境使用机器人福祉教练的看法。我们将我们的工作视为一种预见性伦理调查，旨在通过社区意见更好地指导机器人技术的发展，最终目标是使机器人开发与公众利益相一致。我们与在机器人开发中代表性不足的三个社区进行了工作坊：1）科学节上的公众成员，2）女性计算机科学家，以及3）对科学史和哲学感兴趣的 humanities 研究者。在工作坊中，我们使用社会机器人共设计伦理画布收集定性数据。我们通过工作坊期间记录的笔记进行主题分析，识别出了四个关键的主题关于福祉机器人在实际应用中的伦理和社会技术问题。我们总结了参与者关于这些问题的见解和讨论，结合当前先进文献进行了讨论，并指出了未来研究的重点领域。最后，我们提出了这四个问题作为机器人研究者在机器人开发和部署过程中应使用和关注的广泛框架，以反思其机器人应用的伦理和社会技术维度，并与机器人用户社区进行对话。这四个问题是：1）机器人是否安全，我们如何知道？2）机器人是为谁设计和构建的？3）机器人及其数据的所有权是谁？4）为什么要使用机器人？ 

---
# IS${}^3$ : Generic Impulsive--Stationary Sound Separation in Acoustic Scenes using Deep Filtering 

**Title (ZH)**: IS${}^3$：在声场景中基于深度滤波的通用冲击-稳态声分离 

**Authors**: Berger Clémentine, Stamadiatis Paraskevas, Badeau Roland, Essid Slim  

**Link**: [PDF](https://arxiv.org/pdf/2509.02622)  

**Abstract**: We are interested in audio systems capable of performing a differentiated processing of stationary backgrounds and isolated acoustic events within an acoustic scene, whether for applying specific processing methods to each part or for focusing solely on one while ignoring the other. Such systems have applications in real-world scenarios, including robust adaptive audio rendering systems (e.g., EQ or compression), plosive attenuation in voice mixing, noise suppression or reduction, robust acoustic event classification or even bioacoustics. To this end, we introduce IS${}^3$, a neural network designed for Impulsive--Stationary Sound Separation, that isolates impulsive acoustic events from the stationary background using a deep filtering approach, that can act as a pre-processing stage for the above-mentioned tasks. To ensure optimal training, we propose a sophisticated data generation pipeline that curates and adapts existing datasets for this task. We demonstrate that a learning-based approach, build on a relatively lightweight neural architecture and trained with well-designed and varied data, is successful in this previously unaddressed task, outperforming the Harmonic--Percussive Sound Separation masking method, adapted from music signal processing research, and wavelet filtering on objective separation metrics. 

**Abstract (ZH)**: 基于冲动-稳态声音分离的IS${}^3$神经网络及其应用 

---
# Radio Astronomy in the Era of Vision-Language Models: Prompt Sensitivity and Adaptation 

**Title (ZH)**: 视觉-语言模型时代射电天文：提示敏感性与适应性探究 

**Authors**: Mariia Drozdova, Erica Lastufka, Vitaliy Kinakh, Taras Holotyak, Daniel Schaerer, Slava Voloshynovskiy  

**Link**: [PDF](https://arxiv.org/pdf/2509.02615)  

**Abstract**: Vision-Language Models (VLMs), such as recent Qwen and Gemini models, are positioned as general-purpose AI systems capable of reasoning across domains. Yet their capabilities in scientific imaging, especially on unfamiliar and potentially previously unseen data distributions, remain poorly understood. In this work, we assess whether generic VLMs, presumed to lack exposure to astronomical corpora, can perform morphology-based classification of radio galaxies using the MiraBest FR-I/FR-II dataset. We explore prompting strategies using natural language and schematic diagrams, and, to the best of our knowledge, we are the first to introduce visual in-context examples within prompts in astronomy. Additionally, we evaluate lightweight supervised adaptation via LoRA fine-tuning. Our findings reveal three trends: (i) even prompt-based approaches can achieve good performance, suggesting that VLMs encode useful priors for unfamiliar scientific domains; (ii) however, outputs are highly unstable, i.e. varying sharply with superficial prompt changes such as layout, ordering, or decoding temperature, even when semantic content is held constant; and (iii) with just 15M trainable parameters and no astronomy-specific pretraining, fine-tuned Qwen-VL achieves near state-of-the-art performance (3% Error rate), rivaling domain-specific models. These results suggest that the apparent "reasoning" of VLMs often reflects prompt sensitivity rather than genuine inference, raising caution for their use in scientific domains. At the same time, with minimal adaptation, generic VLMs can rival specialized models, offering a promising but fragile tool for scientific discovery. 

**Abstract (ZH)**: 视觉-语言模型（VLMs），如近期的Qwen和Gemini模型，被定位为通用型AI系统，能够跨领域进行推理。然而，它们在科学成像领域的能力，特别是在不熟悉的且可能之前未见过的数据分布上的能力，仍然知之甚少。在本文中，我们评估未经天文语料库训练的通用VLMs是否能使用MiraBest FR-I/FR-II数据集对射电星系进行基于形态的分类。我们探讨了使用自然语言和示意图的提示策略，并据我们所知，我们是首次在天文学中引入视觉上下文示例作为提示。此外，我们评估了通过LoRA微调的轻量级监督适应。我们的发现揭示了三个趋势：（i）即使基于提示的方法也能实现良好的性能，表明VLMs编码了对于不熟悉的科学领域有用的前提；（ii）然而，输出非常不稳定，即轻微的提示变化（如布局、排序或解码温度）都会导致急剧变化，即使语义内容保持不变；（iii）在仅有1500万个可训练参数且没有特定于天文领域的预训练的情况下，微调的Qwen-VL实现了接近最先进的性能（3%的错误率），媲美专门模型。这些结果表明，VLMs表面上的“推理”往往反映了对提示的敏感性而非真正的推断，这对它们在科学领域的应用提出了警示。同时，通过最少的适配，通用VLMs可以媲美专门模型，为科学发现提供了一个有前景但易受限制的工具。 

---
# Is Synthetic Image Augmentation Useful for Imbalanced Classification Problems? Case-Study on the MIDOG2025 Atypical Cell Detection Competition 

**Title (ZH)**: 合成图像增强对不平衡分类问题有用吗？以MIDOG2025异常细胞检测竞赛为例 

**Authors**: Leire Benito-Del-Valle, Pedro A. Moreno-Sánchez, Itziar Egusquiza, Itsaso Vitoria, Artzai Picón, Cristina López-Saratxaga, Adrian Galdran  

**Link**: [PDF](https://arxiv.org/pdf/2509.02612)  

**Abstract**: The MIDOG 2025 challenge extends prior work on mitotic figure detection by introducing a new Track 2 on atypical mitosis classification. This task aims to distinguish normal from atypical mitotic figures in histopathology images, a clinically relevant but highly imbalanced and cross-domain problem. We investigated two complementary backbones: (i) ConvNeXt-Small, pretrained on ImageNet, and (ii) a histopathology-specific ViT from Lunit trained via self-supervision. To address the strong prevalence imbalance (9408 normal vs. 1741 atypical), we synthesized additional atypical examples to approximate class balance and compared models trained with real-only vs. real+synthetic data. Using five-fold cross-validation, both backbones reached strong performance (mean AUROC approximately 95 percent), with ConvNeXt achieving slightly higher peaks while Lunit exhibited greater fold-to-fold stability. Synthetic balancing, however, did not lead to consistent improvements. On the organizers' preliminary hidden test set, explicitly designed as an out-of-distribution debug subset, ConvNeXt attained the highest AUROC (95.4 percent), whereas Lunit remained competitive on balanced accuracy. These findings suggest that both ImageNet and domain-pretrained backbones are viable for atypical mitosis classification, with domain-pretraining conferring robustness and ImageNet pretraining reaching higher peaks, while naive synthetic balancing has limited benefit. Full hidden test set results will be reported upon challenge completion. 

**Abstract (ZH)**: MIDOG 2025 挑战赛扩展了前期关于有丝分裂图检测的工作，通过引入一个非典型有丝分裂分类的新赛道 Track 2。该任务旨在区分病理图像中的正常与非典型有丝分裂图，这是一个临床相关但高度不平衡且跨领域的难题。我们研究了两个互补的骨干网络：(i) ImageNet 上预训练的 ConvNeXt-Small，(ii) 由 Lunit 提供并在自监督下训练的病理专用 ViT。为解决强先验不平衡问题（正常有丝分裂图 9408 例 vs. 非典型有丝分裂图 1741 例），我们合成额外的非典型样本以逼近类平衡，并比较了使用真实数据 vs. 真实数据加合成数据训练的模型。使用五折交叉验证，两个骨干网络都取得了强劲的表现（平均 AUROC 约 95%），ConvNeXt 达到略高峰值，而 Lunit 展现了更好的折间稳定性。合成平衡并未带来一致的改善。在组织者初步隐藏测试集中，该集明确设计为一个离群值调试子集，ConvNeXt 达到了最高的 AUROC（95.4%），而 Lunit 在平衡准确率上仍然具有竞争力。这些发现表明，对于非典型有丝分裂分类，ImageNet 和领域预训练的骨干网络都是可行的，领域预训练提供鲁棒性，而 ImageNet 预训练达到更高峰值，未经修改的合成平衡具有有限的效果。在挑战赛完成后，将报告完整隐藏测试集的结果。 

---
# Resilient Biosecurity in the Era of AI-Enabled Bioweapons 

**Title (ZH)**: 人工智能赋能生物武器时代的韧性生物安全 

**Authors**: Jonathan Feldman, Tal Feldman  

**Link**: [PDF](https://arxiv.org/pdf/2509.02610)  

**Abstract**: Recent advances in generative biology have enabled the design of novel proteins, creating significant opportunities for drug discovery while also introducing new risks, including the potential development of synthetic bioweapons. Existing biosafety measures primarily rely on inference-time filters such as sequence alignment and protein-protein interaction (PPI) prediction to detect dangerous outputs. In this study, we evaluate the performance of three leading PPI prediction tools: AlphaFold 3, AF3Complex, and SpatialPPIv2. These models were tested on well-characterized viral-host interactions, such as those involving Hepatitis B and SARS-CoV-2. Despite being trained on many of the same viruses, the models fail to detect a substantial number of known interactions. Strikingly, none of the tools successfully identify any of the four experimentally validated SARS-CoV-2 mutants with confirmed binding. These findings suggest that current predictive filters are inadequate for reliably flagging even known biological threats and are even more unlikely to detect novel ones. We argue for a shift toward response-oriented infrastructure, including rapid experimental validation, adaptable biomanufacturing, and regulatory frameworks capable of operating at the speed of AI-driven developments. 

**Abstract (ZH)**: 近期生成生物学的进展使得新型蛋白质的设计成为可能，为药物发现带来了重大机遇，同时也引入了新的风险，包括合成生物武器的可能性。现有的生物安全措施主要依赖于比对滤波以及蛋白质-蛋白质相互作用（PPI）预测等推理时滤波器来检测危险输出。本研究评估了三种领先的PPI预测工具：AlphaFold 3、AF3Complex和SpatialPPIv2。这些模型在已充分表征的病毒-宿主相互作用上进行了测试，例如乙型肝炎和SARS-CoV-2。尽管这些模型是基于相同病毒的大量数据进行训练，但它们未能检测到大量已知的相互作用。令人惊讶的是，这些工具均未能识别任何经过实验证实与SARS-CoV-2结合的四个突变体。这些发现表明，当前的预测滤波器不足以可靠地标记已知的生物威胁，更不用说检测新型生物威胁了。我们呼吁转向以响应为导向的基础设施，包括快速实验验证、灵活的生物制造以及能够与AI驱动的开发速度相匹配的监管框架。 

---
# Contrastive clustering based on regular equivalence for influential node identification in complex networks 

**Title (ZH)**: 基于规则等价性的对比聚类在复杂网络中关键节点识别 

**Authors**: Yanmei Hu, Yihang Wu, Bing Sun, Xue Yue, Biao Cai, Xiangtao Li, Yang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.02609)  

**Abstract**: Identifying influential nodes in complex networks is a fundamental task in network analysis with wide-ranging applications across domains. While deep learning has advanced node influence detection, existing supervised approaches remain constrained by their reliance on labeled data, limiting their applicability in real-world scenarios where labels are scarce or unavailable. While contrastive learning demonstrates significant potential for performance enhancement, existing approaches predominantly rely on multiple-embedding generation to construct positive/negative sample pairs. To overcome these limitations, we propose ReCC (\textit{r}egular \textit{e}quivalence-based \textit{c}ontrastive \textit{c}lustering), a novel deep unsupervised framework for influential node identification. We first reformalize influential node identification as a label-free deep clustering problem, then develop a contrastive learning mechanism that leverages regular equivalence-based similarity, which captures structural similarities between nodes beyond local neighborhoods, to generate positive and negative samples. This mechanism is integrated into a graph convolutional network to learn node embeddings that are used to differentiate influential from non-influential nodes. ReCC is pre-trained using network reconstruction loss and fine-tuned with a combined contrastive and clustering loss, with both phases being independent of labeled data. Additionally, ReCC enhances node representations by combining structural metrics with regular equivalence-based similarities. Extensive experiments demonstrate that ReCC outperforms state-of-the-art approaches across several benchmarks. 

**Abstract (ZH)**: 基于正则等价的对比聚类：一种无监督的节点影响力识别深度学习框架 

---
# Towards Digital Twins for Optimal Radioembolization 

**Title (ZH)**: 面向最佳放射性栓塞的数字孪生技术研究 

**Authors**: Nisanth Kumar Panneerselvam, Guneet Mummaneni, Emilie Roncali  

**Link**: [PDF](https://arxiv.org/pdf/2509.02607)  

**Abstract**: Radioembolization is a localized liver cancer treatment that delivers radioactive microspheres (30 micron) to tumors via a catheter inserted in the hepatic arterial tree. The goal is to maximize therapeutic efficacy while minimizing damage to healthy liver tissue. However, optimization is challenging due to complex hepatic artery anatomy, variable blood flow, and uncertainty in microsphere transport. The creation of dynamic, patient-specific digital twins may provide a transformative solution to these challenges. This work outlines a framework for a liver radioembolization digital twin using high-fidelity computational fluid dynamics (CFD) and/or recent physics-informed machine learning approaches. The CFD approach involves microsphere transport calculations in the hepatic arterial tree with individual patient data, which enables personalized treatment planning. Although accurate, traditional CFD is computationally expensive and limits clinical applicability.
To accelerate simulations, physics-informed neural networks (PINNs) and their generative extensions play an increasingly important role. PINNs integrate governing equations, such as the Navier-Stokes equations, directly into the neural network training process, enabling mesh-free, data-efficient approximation of blood flow and microsphere transport. Physics-informed generative adversarial networks (PI-GANs), diffusion models (PI-DMs), and transformer-based architectures further enable uncertainty-aware, temporally resolved predictions with reduced computational cost. These AI surrogates not only maintain physical fidelity but also support rapid sampling of diverse flow scenarios, facilitating real-time decision support.
Together, CFD and physics-informed AI methods form the foundation of dynamic, patient-specific digital twin to optimize radioembolization planning and ultimately improve clinical outcomes. 

**Abstract (ZH)**: 基于物理学约束的神经网络和计算流体动力学在肝癌放射性栓塞数字双胞胎中的应用 

---
# Synthetic Founders: AI-Generated Social Simulations for Startup Validation Research in Computational Social Science 

**Title (ZH)**: 合成创始人：AI生成的社会模拟在计算社会科学研究中的创业验证研究 

**Authors**: Jorn K. Teutloff  

**Link**: [PDF](https://arxiv.org/pdf/2509.02605)  

**Abstract**: We present a comparative docking experiment that aligns human-subject interview data with large language model (LLM)-driven synthetic personas to evaluate fidelity, divergence, and blind spots in AI-enabled simulation. Fifteen early-stage startup founders were interviewed about their hopes and concerns regarding AI-powered validation, and the same protocol was replicated with AI-generated founder and investor personas. A structured thematic synthesis revealed four categories of outcomes: (1) Convergent themes - commitment-based demand signals, black-box trust barriers, and efficiency gains were consistently emphasized across both datasets; (2) Partial overlaps - founders worried about outliers being averaged away and the stress of real customer validation, while synthetic personas highlighted irrational blind spots and framed AI as a psychological buffer; (3) Human-only themes - relational and advocacy value from early customer engagement and skepticism toward moonshot markets; and (4) Synthetic-only themes - amplified false positives and trauma blind spots, where AI may overstate adoption potential by missing negative historical experiences.
We interpret this comparative framework as evidence that LLM-driven personas constitute a form of hybrid social simulation: more linguistically expressive and adaptable than traditional rule-based agents, yet bounded by the absence of lived history and relational consequence. Rather than replacing empirical studies, we argue they function as a complementary simulation category - capable of extending hypothesis space, accelerating exploratory validation, and clarifying the boundaries of cognitive realism in computational social science. 

**Abstract (ZH)**: 我们呈现了一项将人类受试访谈数据与大型语言模型驱动的合成人格进行比对的对接实验，以评估人工智能增强模拟的真实度、偏差和盲点。我们对15位早期初创公司创始人进行了访谈，探讨他们对人工智能驱动验证的希望与担忧，并用相同的协议复制了由人工智能生成的创始人和投资者人格。结构化主题综合分析揭示了四个类别结果：（1）一致的主题——承诺驱动的需求信号、黑箱信任障碍和效率提升在两个数据集中均被一致强调；（2）部分重叠——创始人担心异常值会被平均掉以及真实的客户验证带来的压力，而合成人格则突出了非理性的盲点，并将人工智能描绘为一种心理缓冲；（3）仅限人类的主题——早期客户互动中的关系价值和对梦幻市场持怀疑态度；（4）仅限合成人格的主题——放大了假阳性结果和创伤性盲点，人工智能可能会因忽视负面历史经验而高估采用潜力。我们将这种比较框架视为证据，表明大型语言模型驱动的人格构成了一种混合社会模拟形式：相比传统的基于规则的代理更为语言表达丰富和适应性强，但受限于缺乏生活经验和关系后果。我们认为它们作为互补的模拟类别发挥了作用——能够扩展假设空间、加速探索性验证，并阐明计算社会科学研究中认知现实的边界。 

---
# MIDOG 2025: Mitotic Figure Detection with Attention-Guided False Positive Correction 

**Title (ZH)**: MIDOG 2025: 注意力引导的假阳性修正有丝分裂图检测 

**Authors**: Andrew Broad, Jason Keighley, Lucy Godson, Alex Wright  

**Link**: [PDF](https://arxiv.org/pdf/2509.02598)  

**Abstract**: We present a novel approach which extends the existing Fully Convolutional One-Stage Object Detector (FCOS) for mitotic figure detection. Our composite model adds a Feedback Attention Ladder CNN (FAL-CNN) model for classification of normal versus abnormal mitotic figures, feeding into a fusion network that is trained to generate adjustments to bounding boxes predicted by FCOS. Our network aims to reduce the false positive rate of the FCOS object detector, to improve the accuracy of object detection and enhance the generalisability of the network. Our model achieved an F1 score of 0.655 for mitosis detection on the preliminary evaluation dataset. 

**Abstract (ZH)**: 我们提出了一种新颖的方法，将现有的全卷积一阶段物体检测器（FCOS）扩展应用于有丝分裂图鉴识。我们的复合模型增加了反馈注意力梯形CNN（FAL-CNN）模型，用于正常与异常有丝分裂图的分类，并将其输入到一个融合网络中，该网络经过训练以生成对FCOS预测边界框的调整。我们的网络旨在降低FCOS物体检测器的假阳性率，提高物体检测的准确性，并增强网络的一般化能力。在初步评价数据集上，我们的模型实现了有丝分裂检测的F1分数为0.655。 

---
# OpenAIs HealthBench in Action: Evaluating an LLM-Based Medical Assistant on Realistic Clinical Queries 

**Title (ZH)**: OpenAI的HealthBench在行动：基于LLM的医学助理在现实临床查询中的评估 

**Authors**: Sandhanakrishnan Ravichandran, Shivesh Kumar, Rogerio Corga Da Silva, Miguel Romano, Reinhard Berkels, Michiel van der Heijden, Olivier Fail, Valentine Emmanuel Gnanapragasam  

**Link**: [PDF](https://arxiv.org/pdf/2509.02594)  

**Abstract**: Evaluating large language models (LLMs) on their ability to generate high-quality, accurate, situationally aware answers to clinical questions requires going beyond conventional benchmarks to assess how these systems behave in complex, high-stake clincal scenarios. Traditional evaluations are often limited to multiple-choice questions that fail to capture essential competencies such as contextual reasoning, awareness and uncertainty handling etc. To address these limitations, we evaluate our agentic, RAG-based clinical support assistant, this http URL, using HealthBench, a rubric-driven benchmark composed of open-ended, expert-annotated health conversations. On the Hard subset of 1,000 challenging examples, this http URL achieves a HealthBench score of 0.51, substantially outperforming leading frontier LLMs (GPT-5, o3, Grok 3, GPT-4, Gemini 2.5, etc.) across all behavioral axes (accuracy, completeness, instruction following, etc.). In a separate 100-sample evaluation against similar agentic RAG assistants (OpenEvidence, this http URL), it maintains a performance lead with a health-bench score of 0.54. These results highlight this http URL strengths in communication, instruction following, and accuracy, while also revealing areas for improvement in context awareness and completeness of a response. Overall, the findings underscore the utility of behavior-level, rubric-based evaluation for building a reliable and trustworthy AI-enabled clinical support assistant. 

**Abstract (ZH)**: 评估大型语言模型在生成高质量、准确且情境意识强的临床问题答案方面的能力要求超越传统基准，以评估这些系统在复杂高风险临床场景中的行为表现。传统评估往往局限于多项选择题，无法捕捉到诸如上下文推理、情境意识和不确定性处理等关键能力。为了解决这些限制，我们使用HealthBench基准对我们的具代理性、基于RAG的临床支持助手this http URL进行了评估，HealthBench是一个由开放式专家标注健康对话组成的评分驱动基准。在HealthBench Hard子集的1,000个具有挑战性的示例中，this http URL取得了0.51的HealthBench评分，全面优于领先的大语言模型（如GPT-5、o3、Grok 3、GPT-4、Gemini 2.5等）在所有行为维度（准确度、完整性、指令遵循等）上的表现。在针对类似具代理性RAG助手（如OpenEvidence、this http URL）的100样本独立评估中，它保持了性能领先地位，取得了0.54的HealthBench评分。这些结果突显了this http URL在沟通、指令遵循和准确度方面的优势，同时揭示了其在情境意识和响应完整性方面的改进空间。总体而言，这些发现强调了行为层面、评分驱动评估在构建可靠的且值得信赖的AI增强临床支持助手方面的效用。 

---
# Robust Pan-Cancer Mitotic Figure Detection with YOLOv12 

**Title (ZH)**: 基于YOLOv12的稳健全景癌变有丝分裂图检测 

**Authors**: Raphaël Bourgade, Guillaume Balezo, Thomas Walter  

**Link**: [PDF](https://arxiv.org/pdf/2509.02593)  

**Abstract**: Mitotic figures represent a key histoprognostic feature in tumor pathology, providing crucial insights into tumor aggressiveness and proliferation. However, their identification remains challenging, subject to significant inter-observer variability, even among experienced pathologists. To address this issue, the MItosis DOmain Generalization (MIDOG) 2025 challenge marks the third edition of an international competition aiming to develop robust mitosis detection algorithms. In this paper, we present a mitotic figures detection approach based on the YOLOv12 object detection architecture, achieving a $F_1$-score of 0.801 on the preliminary test set of the MIDOG 2025 challenge, without relying on external data. 

**Abstract (ZH)**: 分裂相是肿瘤病理中一个关键的组织预后特征，提供了关于肿瘤侵袭性和增殖的重要见解。然而，其识别仍然具有挑战性，甚至在经验丰富的病理学家之间也存在显著的主观差异。为了应对这一问题，MItosis DOmain Generalization (MIDOG) 2025 挑战赛是旨在开发稳健的分裂相检测算法的第三届国际竞赛。在本文中，我们提出了一种基于 YOLOv12 对象检测架构的分裂相检测方法，在 MIDOG 2025 挑战赛初步测试集上达到了 0.801 的 $F_1$-分数，未依赖外部数据。 

---
# Beyond Synthetic Augmentation: Group-Aware Threshold Calibration for Robust Balanced Accuracy in Imbalanced Learning 

**Title (ZH)**: 超越合成增强：群体意识阈值校准以实现稳健的类别平衡准确率在不平衡学习中的应用 

**Authors**: Hunter Gittlin  

**Link**: [PDF](https://arxiv.org/pdf/2509.02592)  

**Abstract**: Class imbalance remains a fundamental challenge in machine learning, with traditional solutions often creating as many problems as they solve. We demonstrate that group-aware threshold calibration--setting different decision thresholds for different demographic groups--provides superior robustness compared to synthetic data generation methods. Through extensive experiments, we show that group-specific thresholds achieve 1.5-4% higher balanced accuracy than SMOTE and CT-GAN augmented models while improving worst-group balanced accuracy. Unlike single-threshold approaches that apply one cutoff across all groups, our group-aware method optimizes the Pareto frontier between balanced accuracy and worst-group balanced accuracy, enabling fine-grained control over group-level performance. Critically, we find that applying group thresholds to synthetically augmented data yields minimal additional benefit, suggesting these approaches are fundamentally redundant. Our results span seven model families including linear, tree-based, instance-based, and boosting methods, confirming that group-aware threshold calibration offers a simpler, more interpretable, and more effective solution to class imbalance. 

**Abstract (ZH)**: 群体意识阈值校准在机器学习中的不平衡类别问题上提供了 superior 的稳健性，相比合成数据生成方法，群体特定阈值在保持均衡准确率方面高出 1.5-4%，并在最不利群体的均衡准确率上有所改进。与在所有群体中应用单一阈值的方法不同，我们的群体意识方法优化了均衡准确率与最不利群体均衡准确率之间的帕累托前沿，从而实现对群体级性能的细粒度控制。关键的是，我们发现将群体阈值应用于合成增强数据几乎没有额外的好处，这表明这些方法本质上是冗余的。我们的结果涵盖了七类模型，包括线性模型、树基模型、实例基模型和提升方法，证实群体意识阈值校准提供了一个更简单、更具可解释性且更有效的解决类别不平衡问题的方法。 

---
# Ensemble of Pathology Foundation Models for MIDOG 2025 Track 2: Atypical Mitosis Classification 

**Title (ZH)**: Ensemble of Pathology Foundation Models for MIDOG 2025 Track 2: 非典型有丝分裂分类 

**Authors**: Mieko Ochi, Bae Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2509.02591)  

**Abstract**: Mitotic figures are classified into typical and atypical variants, with atypical counts correlating strongly with tumor aggressiveness. Accurate differentiation is therefore essential for patient prognostication and resource allocation, yet remains challenging even for expert pathologists. Here, we leveraged Pathology Foundation Models (PFMs) pre-trained on large histopathology datasets and applied parameter-efficient fine-tuning via low-rank adaptation. During training, we employ a fisheye transform to emphasize mitoses and Fourier Domain Adaptation using ImageNet target images. Finally, we ensembled multiple PFMs to integrate complementary morphological insights, achieving a high balanced accuracy on the Preliminary Evaluation Phase dataset. 

**Abstract (ZH)**: 有丝分裂图型被分类为典型和非典型变异，非典型计数与肿瘤 aggressiveness 强烈相关。因此，准确区分对于患者预后和资源分配至关重要，但即使是专家病理学家也面临挑战。在这里，我们利用预训练在大规模组织病理学数据集上的病理学基础模型（PFMs），并通过低秩适应进行参数高效的微调。在训练期间，我们采用鱼眼变换强调有丝分裂，并使用 ImageNet 目标图像进行频率域适应。最终，我们将多个 PFMs 進行集成，以结合互补的形态学见解，在初步评估阶段数据集上实现了高平衡准确率。 

---
# Normal and Atypical Mitosis Image Classifier using Efficient Vision Transformer 

**Title (ZH)**: 使用高效视觉变换器的正常与异常有丝分裂图像分类器 

**Authors**: Xuan Qi, Dominic Labella, Thomas Sanford, Maxwell Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.02589)  

**Abstract**: We tackle atypical versus normal mitosis classification in the MIDOG 2025 challenge using EfficientViT-L2, a hybrid CNN--ViT architecture optimized for accuracy and efficiency. A unified dataset of 13,938 nuclei from seven cancer types (MIDOG++ and AMi-Br) was used, with atypical mitoses comprising ~15. To assess domain generalization, we applied leave-one-cancer-type-out cross-validation with 5-fold ensembles, using stain-deconvolution for image augmentation. For challenge submissions, we trained an ensemble with the same 5-fold split but on all cancer types. In the preliminary evaluation phase, this model achieved balanced accuracy of 0.859, ROC AUC of 0.942, and raw accuracy of 0.85, demonstrating competitive and well-balanced performance across metrics. 

**Abstract (ZH)**: 我们使用EfficientViT-L2架构在MIDOG 2025挑战中解决异常有丝分裂与正常有丝分裂的分类问题，EfficientViT-L2是一种优化准确性和效率的混合CNN-ViT架构。我们使用来自七种癌症类型（MIDOG++和AMi-Br）的统一数据集（13,938个核），其中异常有丝分裂约占15%。为了评估域泛化能力，我们采用了留一癌种交叉验证方法，并使用去染色技术进行图像增强。对于挑战提交，我们在相同的5折分割上训练了一个包含所有癌种的数据集。在初步评估阶段，该模型的平衡准确率为0.859，ROC AUC为0.942，原始准确率为0.85，显示出跨指标上具有竞争力和均衡的表现。 

---
# MitoDetect++: A Domain-Robust Pipeline for Mitosis Detection and Atypical Subtyping 

**Title (ZH)**: MitoDetect++: 一种领域稳健的纺锤体检测和非典型亚型分类管道 

**Authors**: Esha Sadia Nasir, Jiaqi Lv, Mostafa Jahanifer, Shan E Ahmed Raza  

**Link**: [PDF](https://arxiv.org/pdf/2509.02586)  

**Abstract**: Automated detection and classification of mitotic figures especially distinguishing atypical from normal remain critical challenges in computational pathology. We present MitoDetect++, a unified deep learning pipeline designed for the MIDOG 2025 challenge, addressing both mitosis detection and atypical mitosis classification. For detection (Track 1), we employ a U-Net-based encoder-decoder architecture with EfficientNetV2-L as the backbone, enhanced with attention modules, and trained via combined segmentation losses. For classification (Track 2), we leverage the Virchow2 vision transformer, fine-tuned efficiently using Low-Rank Adaptation (LoRA) to minimize resource consumption. To improve generalization and mitigate domain shifts, we integrate strong augmentations, focal loss, and group-aware stratified 5-fold cross-validation. At inference, we deploy test-time augmentation (TTA) to boost robustness. Our method achieves a balanced accuracy of 0.892 across validation domains, highlighting its clinical applicability and scalability across tasks. 

**Abstract (ZH)**: 自动检测和分类有丝分裂图谱，尤其是区分异常与正常有丝分裂仍然是在计算病理学中面临的关键挑战。我们提出MitoDetect++，一个用于MIDOG 2025挑战的统一深度学习管道，旨在解决有丝分裂检测和异常有丝分裂分类问题。对于检测（赛道1），我们采用基于U-Net的编码解码架构，以EfficientNetV2-L作为骨干网络，并通过注意力模块增强，结合分割损失进行训练。对于分类（赛道2），我们利用Virchow2视觉变换器，并通过低秩适应（LoRA）高效微调，以减少资源消耗。为了提高泛化能力和减轻领域偏移，我们整合了强增强、焦点损失和组意识分层5折交叉验证。在推断阶段，我们部署测试时增强（TTA）以提高鲁棒性。我们的方法在验证域上实现了0.892的平衡准确率，突显其在不同任务中的临床适用性和可扩展性。 

---
# Charting the Future of Scholarly Knowledge with AI: A Community Perspective 

**Title (ZH)**: 用AI绘制学术知识的未来图景：一种社区视角 

**Authors**: Azanzi Jiomekong, Hande Küçük McGinty, Keith G. Mills, Allard Oelen, Enayat Rajabi, Harry McElroy, Antrea Christou, Anmol Saini, Janice Anta Zebaze, Hannah Kim, Anna M. Jacyszyn, Sören Auer  

**Link**: [PDF](https://arxiv.org/pdf/2509.02581)  

**Abstract**: Despite the growing availability of tools designed to support scholarly knowledge extraction and organization, many researchers still rely on manual methods, sometimes due to unfamiliarity with existing technologies or limited access to domain-adapted solutions. Meanwhile, the rapid increase in scholarly publications across disciplines has made it increasingly difficult to stay current, further underscoring the need for scalable, AI-enabled approaches to structuring and synthesizing scholarly knowledge. Various research communities have begun addressing this challenge independently, developing tools and frameworks aimed at building reliable, dynamic, and queryable scholarly knowledge bases. However, limited interaction across these communities has hindered the exchange of methods, models, and best practices, slowing progress toward more integrated solutions. This manuscript identifies ways to foster cross-disciplinary dialogue, identify shared challenges, categorize new collaboration and shape future research directions in scholarly knowledge and organization. 

**Abstract (ZH)**: 尽管设计用于支持学术知识提取和组织的工具日益增多，许多研究人员仍然依赖手动方法，有时是因为不熟悉现有技术或无法访问专业化的解决方案。同时，跨学科的学术出版物激增使得保持最新变得更加困难，进一步突显了需要可扩展的、基于AI的方法来结构化和综合学术知识的迫切性。各个研究社区已经开始独立应对这一挑战，开发工具和框架以构建可靠、动态和可查询的学术知识库。然而，这些社区之间的有限互动阻碍了方法、模型和最佳实践的交流，减缓了向更集成解决方案发展的进程。本文识别促进跨学科对话、确定共性挑战、分类新合作并塑造未来学术知识和组织研究方向的方式。 

---
# Latent Variable Modeling in Multi-Agent Reinforcement Learning via Expectation-Maximization for UAV-Based Wildlife Protection 

**Title (ZH)**: 基于无人机的野生动物保护中多智能体 reinforcement learning 的潜在变量建模及期望最大化方法 

**Authors**: Mazyar Taghavi, Rahman Farnoosh  

**Link**: [PDF](https://arxiv.org/pdf/2509.02579)  

**Abstract**: Protecting endangered wildlife from illegal poaching presents a critical challenge, particularly in vast and partially observable environments where real-time response is essential. This paper introduces a novel Expectation-Maximization (EM) based latent variable modeling approach in the context of Multi-Agent Reinforcement Learning (MARL) for Unmanned Aerial Vehicle (UAV) coordination in wildlife protection. By modeling hidden environmental factors and inter-agent dynamics through latent variables, our method enhances exploration and coordination under this http URL implement and evaluate our EM-MARL framework using a custom simulation involving 10 UAVs tasked with patrolling protected habitats of the endangered Iranian leopard. Extensive experimental results demonstrate superior performance in detection accuracy, adaptability, and policy convergence when compared to standard algorithms such as Proximal Policy Optimization (PPO) and Deep Deterministic Policy Gradient (DDPG). Our findings underscore the potential of combining EM inference with MARL to improve decentralized decisionmaking in complex, high-stakes conservation scenarios. The full implementation, simulation environment, and training scripts are publicly available on GitHub. 

**Abstract (ZH)**: 利用多智能体强化学习中的期望最大化（EM）基于潜在变量建模方法保护濒危野生动物免受非法猎杀：在 vast 和部分可观测环境中的 UAV 协调实践与评估 

---
# The Lifecycle Principle: Stabilizing Dynamic Neural Networks with State Memory 

**Title (ZH)**: 生命周期原理：通过状态记忆稳定动态神经网络 

**Authors**: Zichuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02575)  

**Abstract**: I investigate a stronger form of regularization by deactivating neurons for extended periods, a departure from the temporary changes of methods like Dropout. However, this long-term dynamism introduces a critical challenge: severe training instability when neurons are revived with random weights. To solve this, I propose the Lifecycle (LC) principle, a regularization mechanism centered on a key innovation: state memory. Instead of re-initializing a revived neuron, my method restores its parameters to their last known effective state. This process preserves learned knowledge and avoids destructive optimization shocks. My theoretical analysis reveals that the LC principle smooths the loss landscape, guiding optimization towards flatter minima associated with better generalization. Experiments on image classification benchmarks demonstrate that my method improves generalization and robustness. Crucially, ablation studies confirm that state memory is essential for achieving these gains. 

**Abstract (ZH)**: 一种新的长期去激活神经元正则化方法及其应用 

---
