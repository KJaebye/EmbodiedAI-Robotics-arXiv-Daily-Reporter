# Language-Grounded Hierarchical Planning and Execution with Multi-Robot 3D Scene Graphs 

**Title (ZH)**: 基于语言引导的多机器人多层次规划与执行及多维场景图 

**Authors**: Jared Strader, Aaron Ray, Jacob Arkin, Mason B. Peterson, Yun Chang, Nathan Hughes, Christopher Bradley, Yi Xuan Jia, Carlos Nieto-Granda, Rajat Talak, Chuchu Fan, Luca Carlone, Jonathan P. How, Nicholas Roy  

**Link**: [PDF](https://arxiv.org/pdf/2506.07454)  

**Abstract**: In this paper, we introduce a multi-robot system that integrates mapping, localization, and task and motion planning (TAMP) enabled by 3D scene graphs to execute complex instructions expressed in natural language. Our system builds a shared 3D scene graph incorporating an open-set object-based map, which is leveraged for multi-robot 3D scene graph fusion. This representation supports real-time, view-invariant relocalization (via the object-based map) and planning (via the 3D scene graph), allowing a team of robots to reason about their surroundings and execute complex tasks. Additionally, we introduce a planning approach that translates operator intent into Planning Domain Definition Language (PDDL) goals using a Large Language Model (LLM) by leveraging context from the shared 3D scene graph and robot capabilities. We provide an experimental assessment of the performance of our system on real-world tasks in large-scale, outdoor environments. 

**Abstract (ZH)**: 本文介绍了集成基于3D场景图的地图构建、定位及任务与运动规划（TAMP）功能的多机器人系统，以执行自然语言表达的复杂指令。我们的系统构建了一个共享的3D场景图，包含开放式对象地图，用于多机器人3D场景图融合。该表示支持通过对象地图实现的实时、视角不变的再定位和基于3D场景图的规划，使得机器人团队能够理解和执行复杂任务。此外，我们还提出了一种规划方法，利用大型语言模型（LLM）和共享的3D场景图及机器人能力上下文，将操作员的意图转换为规划领域定义语言（PDDL）的目标。我们在大规模室外环境中的实际任务中对系统性能进行了实验评估。 

---
# Prime the search: Using large language models for guiding geometric task and motion planning by warm-starting tree search 

**Title (ZH)**: 预先填充搜索：使用大型语言模型引导几何任务和运动规划的树搜索预热方法 

**Authors**: Dongryung Lee, Sejune Joo, Kimin Lee, Beomjoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.07062)  

**Abstract**: The problem of relocating a set of objects to designated areas amidst movable obstacles can be framed as a Geometric Task and Motion Planning (G-TAMP) problem, a subclass of task and motion planning (TAMP). Traditional approaches to G-TAMP have relied either on domain-independent heuristics or on learning from planning experience to guide the search, both of which typically demand significant computational resources or data. In contrast, humans often use common sense to intuitively decide which objects to manipulate in G-TAMP problems. Inspired by this, we propose leveraging Large Language Models (LLMs), which have common sense knowledge acquired from internet-scale data, to guide task planning in G-TAMP problems. To enable LLMs to perform geometric reasoning, we design a predicate-based prompt that encodes geometric information derived from a motion planning algorithm. We then query the LLM to generate a task plan, which is then used to search for a feasible set of continuous parameters. Since LLMs are prone to mistakes, instead of committing to LLM's outputs, we extend Monte Carlo Tree Search (MCTS) to a hybrid action space and use the LLM to guide the search. Unlike the previous approach that calls an LLM at every node and incurs high computational costs, we use it to warm-start the MCTS with the nodes explored in completing the LLM's task plan. On six different G-TAMP problems, we show our method outperforms previous LLM planners and pure search algorithms. Code can be found at: this https URL 

**Abstract (ZH)**: 基于几何信息的大语言模型指导的任务与运动规划方法 

---
# RoboPARA: Dual-Arm Robot Planning with Parallel Allocation and Recomposition Across Tasks 

**Title (ZH)**: RoboPARA：跨任务的并行分配与重组的双臂机器人规划 

**Authors**: Shiying Duan, Pei Ren, Nanxiang Jiang, Zhengping Che, Jian Tang, Yifan Sun, Zhaoxin Fan, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06683)  

**Abstract**: Dual-arm robots play a crucial role in improving efficiency and flexibility in complex multitasking scenarios. While existing methods have achieved promising results in task planning, they often fail to fully optimize task parallelism, limiting the potential of dual-arm collaboration. To address this issue, we propose RoboPARA, a novel large language model (LLM)-driven framework for dual-arm task parallelism planning. RoboPARA employs a two-stage process: (1) Dependency Graph-based Planning Candidates Generation, which constructs directed acyclic graphs (DAGs) to model task dependencies and eliminate redundancy, and (2) Graph Re-Traversal-based Dual-Arm Parallel Planning, which optimizes DAG traversal to maximize parallelism while maintaining task coherence. In addition, we introduce the Cross-Scenario Dual-Arm Parallel Task dataset (X-DAPT dataset), the first dataset specifically designed to evaluate dual-arm task parallelism across diverse scenarios and difficulty levels. Extensive experiments on the X-DAPT dataset demonstrate that RoboPARA significantly outperforms existing methods, achieving higher efficiency and reliability, particularly in complex task combinations. The code and dataset will be released upon acceptance. 

**Abstract (ZH)**: 双臂机器人在复杂多任务场景中提高效率和灵活性方面发挥着关键作用。尽管现有方法在任务规划方面取得了令人鼓舞的结果，但它们往往未能充分优化任务并行性，限制了双臂协作的潜力。为解决这一问题，我们提出了RoboPARA，这是一种新颖的基于大规模语言模型的双臂任务并行规划框架。RoboPARA采用两阶段过程：（1）依赖图为基础的规划候选生成，构建有向无环图（DAG）来建模任务依赖关系并消除冗余；（2）图再遍历为基础的双臂并行规划，优化DAG遍历以最大限度地提高并行性同时保持任务的一致性。此外，我们引入了Cross-Scenario Dual-Arm Parallel Task数据集（X-DAPT数据集），这是第一个专门用于评估不同场景和难度级别下双臂任务并行性的数据集。在X-DAPT数据集上的广泛实验表明，RoboPARA显著优于现有方法，特别是在复杂任务组合中实现了更高的效率和可靠性。代码和数据集将在接受后发布。 

---
# Hierarchical and Collaborative LLM-Based Control for Multi-UAV Motion and Communication in Integrated Terrestrial and Non-Terrestrial Networks 

**Title (ZH)**: 基于层级协作的大语言模型控制在集成 terrestrial 和非terrestrial 网络中的多无人机运动和通信控制 

**Authors**: Zijiang Yan, Hao Zhou, Jianhua Pei, Hina Tabassum  

**Link**: [PDF](https://arxiv.org/pdf/2506.06532)  

**Abstract**: Unmanned aerial vehicles (UAVs) have been widely adopted in various real-world applications. However, the control and optimization of multi-UAV systems remain a significant challenge, particularly in dynamic and constrained environments. This work explores the joint motion and communication control of multiple UAVs operating within integrated terrestrial and non-terrestrial networks that include high-altitude platform stations (HAPS). Specifically, we consider an aerial highway scenario in which UAVs must accelerate, decelerate, and change lanes to avoid collisions and maintain overall traffic flow. Different from existing studies, we propose a novel hierarchical and collaborative method based on large language models (LLMs). In our approach, an LLM deployed on the HAPS performs UAV access control, while another LLM onboard each UAV handles motion planning and control. This LLM-based framework leverages the rich knowledge embedded in pre-trained models to enable both high-level strategic planning and low-level tactical decisions. This knowledge-driven paradigm holds great potential for the development of next-generation 3D aerial highway systems. Experimental results demonstrate that our proposed collaborative LLM-based method achieves higher system rewards, lower operational costs, and significantly reduced UAV collision rates compared to baseline approaches. 

**Abstract (ZH)**: 无人驾驶航空器（UAVs）已在各种实际应用场景中广泛采用。然而，多UAV系统的控制与优化仍是一个重大挑战，尤其是在动态和受限环境中。本研究探讨了在综合地面和非地面网络中，包括高空平台站（HAPS）的多UAV联合运动和通信控制。具体而言，我们考虑了一个空中高速公路场景，在该场景中，UAVs必须加速、减速并变道以避免碰撞并维持整体交通流。不同于现有研究，我们提出了一种基于大型语言模型（LLMs）的新型分层协作方法。在我们的方法中，部署在HAPS上的LLM负责UAV接入控制，而每个UAV上的LLM处理运动规划和控制。基于LLM的框架利用预训练模型中嵌入的豐富知识，既能够实现高层次的战略规划，也能做出低层次的战术决策。这一知识驱动的范式对下一代三维空中高速公路系统的开发具有巨大潜力。实验结果表明，我们提出的协作LLM基方法在系统奖励、运营成本和UAV碰撞率方面均优于基线方法。 

---
# Solving Inequality Proofs with Large Language Models 

**Title (ZH)**: 用大型语言模型解决不等式证明问题 

**Authors**: Jiayi Sheng, Luna Lyu, Jikai Jin, Tony Xia, Alex Gu, James Zou, Pan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07927)  

**Abstract**: Inequality proving, crucial across diverse scientific and mathematical fields, tests advanced reasoning skills such as discovering tight bounds and strategic theorem application. This makes it a distinct, demanding frontier for large language models (LLMs), offering insights beyond general mathematical problem-solving. Progress in this area is hampered by existing datasets that are often scarce, synthetic, or rigidly formal. We address this by proposing an informal yet verifiable task formulation, recasting inequality proving into two automatically checkable subtasks: bound estimation and relation prediction. Building on this, we release IneqMath, an expert-curated dataset of Olympiad-level inequalities, including a test set and training corpus enriched with step-wise solutions and theorem annotations. We also develop a novel LLM-as-judge evaluation framework, combining a final-answer judge with four step-wise judges designed to detect common reasoning flaws. A systematic evaluation of 29 leading LLMs on IneqMath reveals a surprising reality: even top models like o1 achieve less than 10% overall accuracy under step-wise scrutiny; this is a drop of up to 65.5% from their accuracy considering only final answer equivalence. This discrepancy exposes fragile deductive chains and a critical gap for current LLMs between merely finding an answer and constructing a rigorous proof. Scaling model size and increasing test-time computation yield limited gains in overall proof correctness. Instead, our findings highlight promising research directions such as theorem-guided reasoning and self-refinement. Code and data are available at this https URL. 

**Abstract (ZH)**: 不等式证明：跨多个科学和数学领域的关键任务，测试高级推理能力如发现紧界和战略性定理应用。这使其成为大型语言模型（LLMs）的一个独特且具挑战性的前沿领域，提供超越一般数学问题解决的见解。现有数据集的稀缺性、合成性或形式上的刚性阻碍了这一领域的发展。我们通过提出一种非正式但可验证的任务形式来应对这一挑战，将不等式证明重新构想为两个可自动验证的子任务：界估计和关系预测。在此基础上，我们发布了IneqMath专家精选数据集，包括奥数级不等式测试集和训练语料库，其中包含逐步解决方案和定理注释。我们还开发了一种新颖的LLM作为评判者的评估框架，结合最终答案评判器和四种逐步评判器以检测常见推理错误。系统性评估29个领先的大模型在IneqMath的表现揭示了一个意想不到的事实：即使顶尖模型o1在逐步审查下的总体准确率也低于10%；与仅考虑最终答案等价相比，这一准确率下降高达65.5%。这种差异揭示了脆弱的演绎链和当前大模型在仅仅找到答案和构建严谨证明之间存在的关键差距。通过扩大模型规模和增加测试时计算量在总体证明正确性上仅带来有限的提升。相反，我们的研究结果强调了诸如定理引导推理和自我完善等潜在的研究方向。代码和数据可在以下链接获取。 

---
# LUCIFER: Language Understanding and Context-Infused Framework for Exploration and Behavior Refinement 

**Title (ZH)**: LUCIFER：语言理解和情境融合的探索与行为优化框架 

**Authors**: Dimitris Panagopoulos, Adolfo Perrusquia, Weisi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.07915)  

**Abstract**: In dynamic environments, the rapid obsolescence of pre-existing environmental knowledge creates a gap between an agent's internal model and the evolving reality of its operational context. This disparity between prior and updated environmental valuations fundamentally limits the effectiveness of autonomous decision-making. To bridge this gap, the contextual bias of human domain stakeholders, who naturally accumulate insights through direct, real-time observation, becomes indispensable. However, translating their nuanced, and context-rich input into actionable intelligence for autonomous systems remains an open challenge. To address this, we propose LUCIFER (Language Understanding and Context-Infused Framework for Exploration and Behavior Refinement), a domain-agnostic framework that integrates a hierarchical decision-making architecture with reinforcement learning (RL) and large language models (LLMs) into a unified system. This architecture mirrors how humans decompose complex tasks, enabling a high-level planner to coordinate specialised sub-agents, each focused on distinct objectives and temporally interdependent actions. Unlike traditional applications where LLMs are limited to single role, LUCIFER integrates them in two synergistic roles: as context extractors, structuring verbal stakeholder input into domain-aware representations that influence decision-making through an attention space mechanism aligning LLM-derived insights with the agent's learning process, and as zero-shot exploration facilitators guiding the agent's action selection process during exploration. We benchmark various LLMs in both roles and demonstrate that LUCIFER improves exploration efficiency and decision quality, outperforming flat, goal-conditioned policies. Our findings show the potential of context-driven decision-making, where autonomous systems leverage human contextual knowledge for operational success. 

**Abstract (ZH)**: 在动态环境中，预存的环境知识的迅速过时会导致代理内部模型与其操作背景不断演化的现实之间产生差距。这种先验与更新后的环境价值之间的差异从根本上限制了自主决策的有效性。为了弥合这一差距，人类领域利益相关者基于直接的、实时的观察自然积累的见解变得不可或缺。然而，将他们细微的、富含上下文的输入转化为可用于自主系统的可操作情报仍是一个开放的挑战。为了应对这一挑战，我们提出了LUCIFER（基于语言理解和上下文融入的探索与行为优化框架），这是一种领域无关的框架，将层次决策架构、强化学习（RL）和大型语言模型（LLMs）整合到一个统一系统中。该架构模仿了人类如何分解复杂任务的方式，使得高级规划者能够协调专注于不同目标和时间上相互依赖行动的专业子代理。与传统应用中LLMs仅限于单一角色不同，LUCIFER将它们整合为两个协同作用的角色：作为上下文提取器，将口头利益相关者的输入结构化为领域感知的表示，通过一种注意空间机制影响决策，使LUCIFER生成的见解与代理的学习过程相契合；以及作为零样本探索促进者，在探索过程中引导代理的动作选择过程。我们在两个角色中基准测试了各种LLM，并证明了LUCIFER提高了探索效率和决策质量，优于平铺的目标条件策略。我们的研究结果表明，基于上下文的决策在自主系统利用人类上下文知识实现操作成功方面具有潜力。 

---
# Evaluating Large Language Models on the Frame and Symbol Grounding Problems: A Zero-shot Benchmark 

**Title (ZH)**: 评估大型语言模型的框架和符号接地问题：一个零样本基准 

**Authors**: Shoko Oka  

**Link**: [PDF](https://arxiv.org/pdf/2506.07896)  

**Abstract**: Recent advancements in large language models (LLMs) have revitalized philosophical debates surrounding artificial intelligence. Two of the most fundamental challenges - namely, the Frame Problem and the Symbol Grounding Problem - have historically been viewed as unsolvable within traditional symbolic AI systems. This study investigates whether modern LLMs possess the cognitive capacities required to address these problems. To do so, I designed two benchmark tasks reflecting the philosophical core of each problem, administered them under zero-shot conditions to 13 prominent LLMs (both closed and open-source), and assessed the quality of the models' outputs across five trials each. Responses were scored along multiple criteria, including contextual reasoning, semantic coherence, and information filtering. The results demonstrate that while open-source models showed variability in performance due to differences in model size, quantization, and instruction tuning, several closed models consistently achieved high scores. These findings suggest that select modern LLMs may be acquiring capacities sufficient to produce meaningful and stable responses to these long-standing theoretical challenges. 

**Abstract (ZH)**: 最近大型语言模型的进展重新引发了对人工智能的哲学争论。两种最基本的问题——框架问题和符号接地问题——历来被视为传统符号AI系统无法解决的问题。本研究考察了现代大型语言模型是否具备解决这些问题所需的认知能力。为此，我设计了两个反映每个问题哲学核心的基准任务，在零样本条件下交给了13个知名的大规模语言模型（包括闭源和开源模型），并在五次试验中评估了模型输出的质量。根据上下文推理、语义连贯性和信息过滤等多个标准对响应进行了评分。结果表明，开源模型在性能上表现出一定的变异性，这归因于模型大小、量化和指令调优的差异，但一些闭源模型始终取得了高分。这些发现表明，某些现代大型语言模型可能正在获取足以产生对这些长期理论挑战有意义且稳定响应的能力。 

---
# Addition in Four Movements: Mapping Layer-wise Information Trajectories in LLMs 

**Title (ZH)**: 四步添加法：映射LLMs的逐层信息轨迹 

**Authors**: Yao Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07824)  

**Abstract**: Multi-digit addition is a clear probe of the computational power of large language models. To dissect the internal arithmetic processes in LLaMA-3-8B-Instruct, we combine linear probing with logit-lens inspection. Inspired by the step-by-step manner in which humans perform addition, we propose and analyze a coherent four-stage trajectory in the forward pass:Formula-structure representations become linearly decodable first, while the answer token is still far down the candidate this http URL computational features then emerge this http URL deeper activation layers, numerical abstractions of the result become clearer, enabling near-perfect detection and decoding of the individual digits in the this http URL the output, the model organizes and generates the final content, with the correct token reliably occupying the top this http URL trajectory suggests a hierarchical process that favors internal computation over rote memorization. We release our code and data to facilitate reproducibility. 

**Abstract (ZH)**: 多位数加法是评估大型语言模型计算能力的清晰指标。为了剖析LLaMA-3-8B-Instruct中的内部算术过程，我们结合线性探针与logit-lens检查。受人类进行加法运算逐步方式进行启发，我们提出并分析了一个前后连贯的四阶段前向传递轨迹：公式结构表示首先线性可解，但在候选答案中答案标记仍然很远；随后计算特征在更深的激活层中涌现；最终，结果的数值抽象更加清晰，使模型能够近乎完美地检测和解码输出中的各个数字；在输出阶段，模型组织并生成最终内容，正确标记可靠地占据首位。该轨迹表明了一个分层过程，更侧重于内部计算而非机械记忆。我们发布了代码和数据以促进可再现性。 

---
# Guideline Forest: Experience-Induced Multi-Guideline Reasoning with Stepwise Aggregation 

**Title (ZH)**: 经验诱导多准则逐步聚合的森林指南：Stepwise Aggregation in Guideline Forest: Experience-Induced Multi-Guideline Reasoning 

**Authors**: Jiaxiang CHen, Zhuo Wang, Mingxi Zou, Qifan Wang, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07820)  

**Abstract**: Human reasoning is flexible, adaptive, and grounded in prior experience-qualities that large language models (LLMs) still struggle to emulate. Existing methods either explore diverse reasoning paths at inference time or search for optimal workflows through expensive operations, but both fall short in leveraging multiple reusable strategies in a structured, efficient manner. We propose Guideline Forest, a framework that enhances LLMs reasoning by inducing structured reasoning strategies-called guidelines-from verified examples and executing them via step-wise aggregation. Unlike test-time search or single-path distillation, our method draws on verified reasoning experiences by inducing reusable guidelines and expanding each into diverse variants. Much like human reasoning, these variants reflect alternative thought patterns, are executed in parallel, refined via self-correction, and aggregated step by step-enabling the model to adaptively resolve uncertainty and synthesize robust this http URL evaluate Guideline Forest on four benchmarks-GSM8K, MATH-500, MBPP, and HumanEval-spanning mathematical and programmatic reasoning. Guideline Forest consistently outperforms strong baselines, including CoT, ReAct, ToT, FoT, and AFlow. Ablation studies further highlight the effectiveness of multi-path reasoning and stepwise aggregation, underscoring the Guideline Forest's adaptability and generalization potential. 

**Abstract (ZH)**: 大型语言模型在灵活、适应性和基于先前经验的推理方面仍难以模仿。现有的方法要么在推理时探索多种推理路径，要么通过昂贵的操作搜索最优工作流程，但两者都无法有效地整合多种可重用的策略。我们提出了Guideline Forest框架，该框架通过诱导结构化的推理策略（称为指导原则）并执行这些策略来增强大型语言模型的推理能力，执行过程通过逐步聚合进行。与测试时的搜索或单一路径的知识蒸馏不同，我们的方法通过诱导可重用的推理经验和扩展每个经验为多种变体来利用验证的推理经验。这些变体反映了不同的思维模式，可以并行执行，通过自我纠正进行优化，并逐步聚合，从而使模型能够适应性地解决不确定性并综合出稳健的结果。我们在涵盖数学和程序推理的四个基准GSM8K、MATH-500、MBPP和HumanEval上评估了Guideline Forest。Guideline Forest在这些基准上的表现优于包括CoT、ReAct、ToT、FoT和AFlow在内的强基线。消融研究进一步证明了多路径推理和逐步聚合的有效性，突显了Guideline Forest的适应性和泛化潜力。 

---
# REMoH: A Reflective Evolution of Multi-objective Heuristics approach via Large Language Models 

**Title (ZH)**: REMoH：借助大型语言模型的多目标启发式方法的反思性演变 

**Authors**: Diego Forniés-Tabuenca, Alejandro Uribe, Urtzi Otamendi, Arkaitz Artetxe, Juan Carlos Rivera, Oier Lopez de Lacalle  

**Link**: [PDF](https://arxiv.org/pdf/2506.07759)  

**Abstract**: Multi-objective optimization is fundamental in complex decision-making tasks. Traditional algorithms, while effective, often demand extensive problem-specific modeling and struggle to adapt to nonlinear structures. Recent advances in Large Language Models (LLMs) offer enhanced explainability, adaptability, and reasoning. This work proposes Reflective Evolution of Multi-objective Heuristics (REMoH), a novel framework integrating NSGA-II with LLM-based heuristic generation. A key innovation is a reflection mechanism that uses clustering and search-space reflection to guide the creation of diverse, high-quality heuristics, improving convergence and maintaining solution diversity. The approach is evaluated on the Flexible Job Shop Scheduling Problem (FJSSP) in-depth benchmarking against state-of-the-art methods using three instance datasets: Dauzere, Barnes, and Brandimarte. Results demonstrate that REMoH achieves competitive results compared to state-of-the-art approaches with reduced modeling effort and enhanced adaptability. These findings underscore the potential of LLMs to augment traditional optimization, offering greater flexibility, interpretability, and robustness in multi-objective scenarios. 

**Abstract (ZH)**: 多目标优化是复杂决策任务中的基础。传统算法虽然有效，但往往需要大量的问题特定建模，并且难以适应非线性结构。近年来，大型语言模型（LLMs）的进步提供了增强的可解释性、适应性和推理能力。本文提出了一种名为反射式多目标启发式进化的框架（REMoH），该框架将NSGA-II与基于LLM的启发式生成相结合。一个关键创新是反射机制，利用聚类和搜索空间反射来引导多样且高质量启发式的生成，从而提高收敛性和保持解的多样性。该方法在详细基准测试中，使用Dauzere、Barnes和Brandimarte的三个实例数据集，与先进方法对比进行了灵活作业车间调度问题（FJSSP）的评估。结果表明，REMoH在减少建模 effort 和增强适应性方面与先进方法具有竞争力。这些发现强调了LLMs在增强传统优化方面的潜力，提供了在多目标场景中更大的灵活性、可解释性和稳健性。 

---
# RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards 

**Title (ZH)**: RSafe: 通过促进主动推理来构建稳健且适应性的LLM防护措施 

**Authors**: Jingnan Zheng, Xiangtian Ji, Yijun Lu, Chenhang Cui, Weixiang Zhao, Gelei Deng, Zhenkai Liang, An Zhang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.07736)  

**Abstract**: Large Language Models (LLMs) continue to exhibit vulnerabilities despite deliberate safety alignment efforts, posing significant risks to users and society. To safeguard against the risk of policy-violating content, system-level moderation via external guard models-designed to monitor LLM inputs and outputs and block potentially harmful content-has emerged as a prevalent mitigation strategy. Existing approaches of training guard models rely heavily on extensive human curated datasets and struggle with out-of-distribution threats, such as emerging harmful categories or jailbreak attacks. To address these limitations, we propose RSafe, an adaptive reasoning-based safeguard that conducts guided safety reasoning to provide robust protection within the scope of specified safety policies. RSafe operates in two stages: 1) guided reasoning, where it analyzes safety risks of input content through policy-guided step-by-step reasoning, and 2) reinforced alignment, where rule-based RL optimizes its reasoning paths to align with accurate safety prediction. This two-stage training paradigm enables RSafe to internalize safety principles to generalize safety protection capability over unseen or adversarial safety violation scenarios. During inference, RSafe accepts user-specified safety policies to provide enhanced safeguards tailored to specific safety requirements. 

**Abstract (ZH)**: 大型语言模型（LLMs）在安全对齐努力后仍表现出漏洞，对用户和社会构成了重大风险。为了防止政策违规内容的风险，通过外部护盾模型进行系统级 moderation——设计用来监控LLM的输入和输出并阻止潜在有害内容——已成为一种普遍的缓解策略。现有的护盾模型训练方法高度依赖大量人工标注的数据集，并且在处理新兴有害类别或监狱突破攻击等离分布威胁方面存在困难。为了解决这些限制，我们提出了基于自适应推理的RSafe，一种进行引导安全推理以在指定安全策略范围内提供稳健保护的防护机制。RSafe分为两个阶段：1) 引导推理，通过策略引导的逐步推理分析输入内容的安全风险；2) 强化对齐，基于规则的强化学习优化其推理路径以与准确的安全预测对齐。这种两阶段训练范式使RSafe能够在未见过的或对抗性安全违规场景中内化安全原则并泛化安全保护能力。在推理过程中，RSafe接受用户指定的安全政策以提供针对特定安全需求的增强保护。 

---
# NeurIPS 2025 E2LM Competition : Early Training Evaluation of Language Models 

**Title (ZH)**: Ne.signIn.2025 E2LM竞赛：语言模型早期训练评估 

**Authors**: Mouadh Yagoubi, Yasser Dahou, Billel Mokeddem, Younes Belkada, Phuc H. Le-Khac, Basma El Amel Boussaha, Reda Alami, Jingwei Zuo, Damiano Marsili, Mugariya Farooq, Mounia Lalmas, Georgia Gkioxari, Patrick Gallinari, Philip Torr, Hakim Hacid  

**Link**: [PDF](https://arxiv.org/pdf/2506.07731)  

**Abstract**: Existing benchmarks have proven effective for assessing the performance of fully trained large language models. However, we find striking differences in the early training stages of small models, where benchmarks often fail to provide meaningful or discriminative signals. To explore how these differences arise, this competition tackles the challenge of designing scientific knowledge evaluation tasks specifically tailored for measuring early training progress of language models. Participants are invited to develop novel evaluation methodologies or adapt existing benchmarks to better capture performance differences among language models. To support this effort, we provide three pre-trained small models (0.5B, 1B, and 3B parameters), along with intermediate checkpoints sampled during training up to 200B tokens. All experiments and development work can be run on widely available free cloud-based GPU platforms, making participation accessible to researchers with limited computational resources. Submissions will be evaluated based on three criteria: the quality of the performance signal they produce, the consistency of model rankings at 1 trillion tokens of training, and their relevance to the scientific knowledge domain. By promoting the design of tailored evaluation strategies for early training, this competition aims to attract a broad range of participants from various disciplines, including those who may not be machine learning experts or have access to dedicated GPU resources. Ultimately, this initiative seeks to make foundational LLM research more systematic and benchmark-informed from the earliest phases of model development. 

**Abstract (ZH)**: 现有的基准测试已证明对评估完全训练的大语言模型性能是有效的。然而，我们发现小型模型的早期训练阶段存在显著差异，在这些阶段，基准测试往往无法提供有意义或区分性的信号。为了探索这些差异的来源，此次竞赛旨在设计专门针对衡量语言模型早期训练进度的科学知识评估任务。参与者被邀请开发新的评估方法或适应现有的基准测试，以更好地捕捉不同语言模型之间的性能差异。为此，我们提供了三个预训练的小型模型（0.5B、1B和3B参数），以及训练过程中抽取的多达200B个标记的中间检查点。所有的实验和开发工作都可以在广泛可用的免费云GPU平台上运行，从而使得计算资源有限的研究人员也能参与其中。提交将根据三个标准进行评估：生成的性能信号的质量、在训练1万亿个标记时模型排名的一致性以及对科学知识领域的相关性。通过促进针对早期训练的定制评估策略设计，此次竞赛旨在吸引来自各个学科的广泛参与者，包括那些可能不是机器学习专家或没有专用GPU资源的研究人员。最终，这一举措旨在使基础大语言模型研究从模型开发的最早阶段起就更加系统和基准导向。 

---
# MCPWorld: A Unified Benchmarking Testbed for API, GUI, and Hybrid Computer Use Agents 

**Title (ZH)**: MCPWorld: 一体化API、GUI及混合计算机使用代理基准测试平台 

**Authors**: Yunhe Yan, Shihe Wang, Jiajun Du, Yexuan Yang, Yuxuan Shan, Qichen Qiu, Xianqing Jia, Xinge Wang, Xin Yuan, Xu Han, Mao Qin, Yinxiao Chen, Chen Peng, Shangguang Wang, Mengwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07672)  

**Abstract**: (M)LLM-powered computer use agents (CUA) are emerging as a transformative technique to automate human-computer interaction. However, existing CUA benchmarks predominantly target GUI agents, whose evaluation methods are susceptible to UI changes and ignore function interactions exposed by application APIs, e.g., Model Context Protocol (MCP). To this end, we propose MCPWorld, the first automatic CUA testbed for API, GUI, and API-GUI hybrid agents. A key principle of MCPWorld is the use of "white-box apps", i.e., those with source code availability and can be revised/re-compiled as needed (e.g., adding MCP support), with two notable advantages:
(1) It greatly broadens the design space of CUA, such as what and how the app features to be exposed/extracted as CUA-callable APIs.
(2) It allows MCPWorld to programmatically verify task completion by directly monitoring application behavior through techniques like dynamic code instrumentation, offering robust, accurate CUA evaluation decoupled from specific agent implementations or UI states.
Currently, MCPWorld includes 201 well curated and annotated user tasks, covering diversified use cases and difficulty levels. MCPWorld is also fully containerized with GPU acceleration support for flexible adoption on different OS/hardware environments. Our preliminary experiments, using a representative LLM-powered CUA framework, achieve 75.12% task completion accuracy, simultaneously providing initial evidence on the practical effectiveness of agent automation leveraging MCP. Overall, we anticipate MCPWorld to facilitate and standardize the benchmarking of next-generation computer use agents that can leverage rich external tools. Our code and dataset are publicly available at this https URL. 

**Abstract (ZH)**: 基于(M)LLM的计算机使用代理(CUA)正在成为自动化人机交互的一种变革性技术。然而，现有的CUA基准主要针对GUI代理，其评估方法易受UI变化影响且忽略由应用程序API暴露的功能交互，例如Model Context Protocol (MCP)。为解决这一问题，我们提出了MCPWorld，这是首个针对API、GUI及API-GUI混合代理的自动CUA测试平台。MCPWorld的关键原则是使用“白盒应用程序”，即具有可用源代码且可根据需要进行修改/重新编译的应用程序（例如，增加MCP支持），具有以下两个显著优势：
(1) 它大大扩展了CUA的设计空间，例如应用程序特性的展示/提取作为CUA可调用的API的方式和内容。
(2) 它允许MCPWorld通过动态代码插桩等技术直接监控应用程序行为来进行程序化验证任务完成情况，提供与特定代理实现或UI状态无关的稳健、准确的CUA评估。
目前，MCPWorld包含201个精心挑选和注释的用户任务，涵盖多样化用例和难度级别。MCPWorld还完全容器化，支持GPU加速，便于在不同的操作系统/硬件环境中灵活采用。初步实验使用一个代表性的基于( M)LLM的CUA框架，实现了75.12%的任务完成准确率，同时为基于MCP的代理自动化具备实际效果提供了初步证据。总体而言，我们期望MCPWorld能够促进和规范下一代能够利用丰富外部工具的计算机使用代理的基准测试。我们的代码和数据集已在以下网址公开：这个 https URL。 

---
# SWE-Dev: Building Software Engineering Agents with Training and Inference Scaling 

**Title (ZH)**: SWE-Dev: 基于训练和推理扩展构建软件工程代理 

**Authors**: Haoran Wang, Zhenyu Hou, Yao Wei, Jie Tang, Yuxiao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.07636)  

**Abstract**: Large language models (LLMs) have advanced rapidly from conversational problem solving to addressing real-world tasks involving tool use, such as software engineering (SWE). Recent LLM-powered toolkits, such as OpenAI Codex and Cursor, have offered end-to-end automation of the software development process. However, building effective SWE agents remains challenging due to the lack of high-quality training data and effective test cases. To address this issue, we present SWE-Dev, an SWE agent built upon open-source LLMs. First, we develop a robust pipeline to synthesize test cases for patch evaluation. Second, we scale up agent trajectories to construct the training data for building SWE-Dev. Experiments on the SWE-bench-Verified benchmark show that the SWE-Dev models can achieve top performance among all open SWE agents. Specifically, the success rates of the SWE-Dev 7B and 32B parameter models reach 23.4% and 36.6%, respectively, outperforming state-of-the-art open-source models. All code, models, and datasets are publicly available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在软件工程中的应用从对话式问题解决迅速发展到了涉及工具使用的真实世界任务。近期基于LLMs的工具包，如OpenAI Codex和Cursor，提供了软件开发过程的端到端自动化。然而，由于缺乏高质量的训练数据和有效的测试案例，构建有效的软件工程（SWE）代理仍然面临挑战。为了解决这一问题，我们提出了SWE-Dev，这是一种基于开源LLMs构建的SWE代理。首先，我们开发了一个 robust 的流水线来合成补丁评估的测试案例。其次，我们扩展了代理轨迹以构建构建SWE-Dev的训练数据。在SWE-bench-Verified基准上的实验表明，SWE-Dev模型在所有开源SWE代理中达到了最佳性能。具体来说，SWE-Dev 7B和32B参数模型的成功率分别达到了23.4%和36.6%，优于最先进的开源模型。所有代码、模型和数据集均可在如下链接公开获取：this https URL。 

---
# Learning What Reinforcement Learning Can't: Interleaved Online Fine-Tuning for Hardest Questions 

**Title (ZH)**: 学习强化学习无法解决的内容：艰难问题的交错在线微调 

**Authors**: Lu Ma, Hao Liang, Meiyi Qiang, Lexiang Tang, Xiaochen Ma, Zhen Hao Wong, Junbo Niu, Chengyu Shen, Runming He, Bin Cui, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07527)  

**Abstract**: Recent advances in large language model (LLM) reasoning have shown that sophisticated behaviors such as planning and self-reflection can emerge through reinforcement learning (RL). However, despite these successes, RL in its current form remains insufficient to induce capabilities that exceed the limitations of the base model, as it is primarily optimized based on existing knowledge of the model rather than facilitating the acquisition of new information. To address this limitation, we employ supervised fine-tuning (SFT) to learn what RL cannot, which enables the incorporation of new knowledge and reasoning patterns by leveraging high-quality demonstration data. We analyze the training dynamics of RL and SFT for LLM reasoning and find that RL excels at maintaining and improving performance on questions within the model's original capabilities, while SFT is more effective at enabling progress on questions beyond the current scope of the model. Motivated by the complementary strengths of RL and SFT, we introduce a novel training approach, \textbf{ReLIFT} (\textbf{Re}inforcement \textbf{L}earning \textbf{I}nterleaved with Online \textbf{F}ine-\textbf{T}uning). In ReLIFT, the model is primarily trained using RL, but when it encounters challenging questions, high-quality solutions are collected for fine-tuning, and the training process alternates between RL and fine-tuning to enhance the model's reasoning abilities. ReLIFT achieves an average improvement of over +5.2 points across five competition-level benchmarks and one out-of-distribution benchmark compared to other zero-RL models. Furthermore, we demonstrate that ReLIFT outperforms both RL and SFT while using only 13\% of the detailed demonstration data, highlighting its scalability. These results provide compelling evidence that ReLIFT overcomes the fundamental limitations of RL and underscores the significant potential. 

**Abstract (ZH)**: Recent Advances in Large Language Model Reasoning: Combining Reinforcement Learning with Supervised Fine-Tuning 

---
# Fact in Fragments: Deconstructing Complex Claims via LLM-based Atomic Fact Extraction and Verification 

**Title (ZH)**: 断片的事实：通过基于LLM的原子事实提取与验证分解复杂断言 

**Authors**: Liwen Zheng, Chaozhuo Li, Zheng Liu, Feiran Huang, Haoran Jia, Zaisheng Ye, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07446)  

**Abstract**: Fact verification plays a vital role in combating misinformation by assessing the veracity of claims through evidence retrieval and reasoning. However, traditional methods struggle with complex claims requiring multi-hop reasoning over fragmented evidence, as they often rely on static decomposition strategies and surface-level semantic retrieval, which fail to capture the nuanced structure and intent of the claim. This results in accumulated reasoning errors, noisy evidence contamination, and limited adaptability to diverse claims, ultimately undermining verification accuracy in complex scenarios. To address this, we propose Atomic Fact Extraction and Verification (AFEV), a novel framework that iteratively decomposes complex claims into atomic facts, enabling fine-grained retrieval and adaptive reasoning. AFEV dynamically refines claim understanding and reduces error propagation through iterative fact extraction, reranks evidence to filter noise, and leverages context-specific demonstrations to guide the reasoning process. Extensive experiments on five benchmark datasets demonstrate that AFEV achieves state-of-the-art performance in both accuracy and interpretability. 

**Abstract (ZH)**: 原子事实提取与验证（AFEV）在复杂断言的细粒度检索与自适应推理中发挥关键作用 

---
# Evaluating Visual Mathematics in Multimodal LLMs: A Multilingual Benchmark Based on the Kangaroo Tests 

**Title (ZH)**: 多模态大语言模型中视觉数学评估：基于袋鼠测试的多语言基准 

**Authors**: Arnau Igualde Sáez, Lamyae Rhomrasi, Yusef Ahsini, Ricardo Vinuesa, Sergio Hoyas, Jose P. García Sabater, Marius J. Fullana i Alfonso, J. Alberto Conejero  

**Link**: [PDF](https://arxiv.org/pdf/2506.07418)  

**Abstract**: Multimodal Large Language Models (MLLMs) promise advanced vision language capabilities, yet their effectiveness in visually presented mathematics remains underexplored. This paper analyzes the development and evaluation of MLLMs for mathematical problem solving, focusing on diagrams, multilingual text, and symbolic notation. We then assess several models, including GPT 4o, Pixtral, Qwen VL, Llama 3.2 Vision variants, and Gemini 2.0 Flash in a multilingual Kangaroo style benchmark spanning English, French, Spanish, and Catalan. Our experiments reveal four key findings. First, overall precision remains moderate across geometry, visual algebra, logic, patterns, and combinatorics: no single model excels in every topic. Second, while most models see improved accuracy with questions that do not have images, the gain is often limited; performance for some remains nearly unchanged without visual input, indicating underutilization of diagrammatic information. Third, substantial variation exists across languages and difficulty levels: models frequently handle easier items but struggle with advanced geometry and combinatorial reasoning. Notably, Gemini 2.0 Flash achieves the highest precision on image based tasks, followed by Qwen VL 2.5 72B and GPT 4o, though none approach human level performance. Fourth, a complementary analysis aimed at distinguishing whether models reason or simply recite reveals that Gemini and GPT 4o stand out for their structured reasoning and consistent accuracy. In contrast, Pixtral and Llama exhibit less consistent reasoning, often defaulting to heuristics or randomness when unable to align their outputs with the given answer options. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在视觉呈现的数学领域提供了高级的语言能力，但其有效性仍待进一步探索。本文分析了MLLMs在数学问题解决中的发展与评估，重点关注图表、多语言文本和象征性符号。然后，我们在英文、法文、西班牙文和加泰罗尼亚文的Kangaroo风格多语言基准测试中评估了包括GPT 4o、Pixtral、Qwen VL、Llama 3.2视觉变体和Gemini 2.0 Flash在内的几种模型。我们的实验揭示了四个关键发现。首先，整体精确度在几何学、视觉代数、逻辑学、模式和组合学中保持适中：没有单一模型在所有主题中都表现优异。其次，大多数模型在没有图像的问题上准确性有所提高，但改进幅度往往有限；一些模型在没有视觉输入的情况下性能几乎没有变化，表明对图表信息的利用不足。第三，不同语言和难度级别之间存在显著差异：模型通常能处理较简单的题目，但在高级几何学和组合推理方面却遇到困难。值得注意的是，Gemini 2.0 Flash在基于图像的任务中取得了最高的精确度，其次是Qwen VL 2.5 72B和GPT 4o，但均未达到人类水平的性能。第四，旨在区分模型是推理还是简单复述的补充分析表明，Gemini和GPT 4o因其结构化的推理和一致的准确性脱颖而出。相比之下，Pixtral和Llama的推理不那么一致，当无法使输出与给定答案选项对齐时，它们往往会依赖于启发式方法或随机性。 

---
# An Intelligent Fault Self-Healing Mechanism for Cloud AI Systems via Integration of Large Language Models and Deep Reinforcement Learning 

**Title (ZH)**: 基于大型语言模型和深度强化学习集成的云AI系统智能故障自愈机制 

**Authors**: Ze Yang, Yihong Jin, Juntian Liu, Xinhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07411)  

**Abstract**: As the scale and complexity of cloud-based AI systems continue to increase, the detection and adaptive recovery of system faults have become the core challenges to ensure service reliability and continuity. In this paper, we propose an Intelligent Fault Self-Healing Mechanism (IFSHM) that integrates Large Language Model (LLM) and Deep Reinforcement Learning (DRL), aiming to realize a fault recovery framework with semantic understanding and policy optimization capabilities in cloud AI systems. On the basis of the traditional DRL-based control model, the proposed method constructs a two-stage hybrid architecture: (1) an LLM-driven fault semantic interpretation module, which can dynamically extract deep contextual semantics from multi-source logs and system indicators to accurately identify potential fault modes; (2) DRL recovery strategy optimizer, based on reinforcement learning, learns the dynamic matching of fault types and response behaviors in the cloud environment. The innovation of this method lies in the introduction of LLM for environment modeling and action space abstraction, which greatly improves the exploration efficiency and generalization ability of reinforcement learning. At the same time, a memory-guided meta-controller is introduced, combined with reinforcement learning playback and LLM prompt fine-tuning strategy, to achieve continuous adaptation to new failure modes and avoid catastrophic forgetting. Experimental results on the cloud fault injection platform show that compared with the existing DRL and rule methods, the IFSHM framework shortens the system recovery time by 37% with unknown fault scenarios. 

**Abstract (ZH)**: 基于大规模语言模型和深度强化学习的智能故障自愈机制 

---
# Boosting Vulnerability Detection of LLMs via Curriculum Preference Optimization with Synthetic Reasoning Data 

**Title (ZH)**: 基于合成推理数据的教学偏好优化以增强对LLMs的漏洞检测 

**Authors**: Xin-Cheng Wen, Yijun Yang, Cuiyun Gao, Yang Xiao, Deheng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2506.07390)  

**Abstract**: Large language models (LLMs) demonstrate considerable proficiency in numerous coding-related tasks; however, their capabilities in detecting software vulnerabilities remain limited. This limitation primarily stems from two factors: (1) the absence of reasoning data related to vulnerabilities, which hinders the models' ability to capture underlying vulnerability patterns; and (2) their focus on learning semantic representations rather than the reason behind them, thus failing to recognize semantically similar vulnerability samples. Furthermore, the development of LLMs specialized in vulnerability detection is challenging, particularly in environments characterized by the scarcity of high-quality datasets. In this paper, we propose a novel framework ReVD that excels at mining vulnerability patterns through reasoning data synthesizing and vulnerability-specific preference optimization. Specifically, we construct forward and backward reasoning processes for vulnerability and corresponding fixed code, ensuring the synthesis of high-quality reasoning data. Moreover, we design the triplet supervised fine-tuning followed by curriculum online preference optimization for enabling ReVD to better understand vulnerability patterns. The extensive experiments conducted on PrimeVul and SVEN datasets demonstrate that ReVD sets new state-of-the-art for LLM-based software vulnerability detection, e.g., 12.24\%-22.77\% improvement in the accuracy. The source code and data are available at this https URL. 

**Abstract (ZH)**: 大规模语言模型在编码相关任务中表现出色，但在检测软件漏洞方面的能力有限。这一限制主要源于两个因素：(1) 缺乏与漏洞相关的推理数据，这阻碍了模型捕捉潜在漏洞模式的能力；(2) 模型侧重于学习语义表示而非其背后的原因，因此无法识别语义相似的漏洞样本。此外，特别是在高质量数据集稀缺的环境中，开发专门用于漏洞检测的语言模型极具挑战性。本文提出了一种新颖的框架ReVD，该框架通过推理数据合成和针对特定漏洞的偏好优化，擅长挖掘漏洞模式。具体来说，我们构建了漏洞及其相应修复代码的正向和反向推理过程，确保合成高质量的推理数据。此外，我们设计了三元监督微调和基于 Curriculum 的在线偏好优化策略，以使ReVD更好地理解漏洞模式。在PrimeVul和SVEN数据集上的广泛实验表明，ReVD为基于语言模型的软件漏洞检测设立了新标准，例如在准确性方面提高了12.24%-22.77%。源代码和数据可在以下网址获取。 

---
# BIMgent: Towards Autonomous Building Modeling via Computer-use Agents 

**Title (ZH)**: BIMgent: 通过计算机代理实现自主建筑建模 

**Authors**: Zihan Deng, Changyu Du, Stavros Nousias, André Borrmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.07217)  

**Abstract**: Existing computer-use agents primarily focus on general-purpose desktop automation tasks, with limited exploration of their application in highly specialized domains. In particular, the 3D building modeling process in the Architecture, Engineering, and Construction (AEC) sector involves open-ended design tasks and complex interaction patterns within Building Information Modeling (BIM) authoring software, which has yet to be thoroughly addressed by current studies. In this paper, we propose BIMgent, an agentic framework powered by multimodal large language models (LLMs), designed to enable autonomous building model authoring via graphical user interface (GUI) operations. BIMgent automates the architectural building modeling process, including multimodal input for conceptual design, planning of software-specific workflows, and efficient execution of the authoring GUI actions. We evaluate BIMgent on real-world building modeling tasks, including both text-based conceptual design generation and reconstruction from existing building design. The design quality achieved by BIMgent was found to be reasonable. Its operations achieved a 32% success rate, whereas all baseline models failed to complete the tasks (0% success rate). Results demonstrate that BIMgent effectively reduces manual workload while preserving design intent, highlighting its potential for practical deployment in real-world architectural modeling scenarios. 

**Abstract (ZH)**: 现有的计算机使用代理主要集中在通用桌面自动化任务上，在高度专业化领域中的应用探索有限。特别是在建筑、工程和施工（AEC）行业中，建筑信息建模（BIM）作者软件中的开放式设计任务和复杂交互模式尚未得到充分研究。本文提出BIMgent，一种基于多模态大规模语言模型（LLMs）的动力体系框架，旨在通过图形用户界面（GUI）操作实现自主建筑模型创建。BIMgent自动化了建筑设计过程，包括多模态输入的概念设计、软件特定工作流的规划以及作者GUI操作的有效执行。我们在实际建筑建模任务中评估了BIMgent，包括基于文本的概念设计生成和现有建筑设计的重建。BIMgent实现的设计质量合理，操作成功率为32%，而所有基线模型都无法完成任务（成功率为0%）。结果显示，BIMgent有效减少了人工工作量，同时保留了设计意图，突显了其在实际建筑建模场景中的实际部署潜力。 

---
# Reasoning Multimodal Large Language Model: Data Contamination and Dynamic Evaluation 

**Title (ZH)**: 多模态大型语言模型推理：数据污染与动态评估 

**Authors**: Ming Liu, Wensheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07202)  

**Abstract**: Multimodal Large Language Models (MLLMs) show impressive vision-language benchmark performance, yet growing concerns about data contamination (test set exposure during training) risk masking true generalization. This concern extends to reasoning MLLMs, often fine-tuned via reinforcement learning from potentially contaminated base models. We propose a novel dynamic evaluation framework to rigorously assess MLLM generalization, moving beyond static benchmarks. Instead of perturbing inputs, we perturb the task itself. Using the same visual input, models are evaluated across a family of tasks (e.g., QA, captioning, question posing, verification) to probe diverse capabilities. This task perturbation reveals whether model performance is robust or reliant on superficial task-specific cues. Our approach is analogous to loss landscape sharpness: models overfit or contaminated for a single task (sharp minima) falter under task shifts, unlike models with generalizable solutions (flatter minima). We developed an automated pipeline with a calibrated judge scoring open-ended generations (captions, questions) using paraphrase and corruption sampling. Applying this framework to leading image/video MLLMs on benchmarks including MME, RealWorldQA, and CVRR-ES, we analyze each model's cross-task "ability vector." We demonstrate that fine-tuning on simulated test data (extreme contamination) drastically sharpens task-specific performance but harms overall generalization. Our dynamic task perturbation offers deeper insights into MLLM generalization, distinguishing genuine understanding from spurious leakage or overfitting. 

**Abstract (ZH)**: 多模态大型语言模型的动态评估框架：超越静态基准探求通用性 

---
# Exploring Effective Strategies for Building a Customised GPT Agent for Coding Classroom Dialogues 

**Title (ZH)**: 探索构建个性化GPT代理以用于编程课堂对话的有效策略 

**Authors**: Luwei Bai, Dongkeun Han, Sara Hennessy  

**Link**: [PDF](https://arxiv.org/pdf/2506.07194)  

**Abstract**: This study investigates effective strategies for developing a customised GPT agent to code classroom dialogue. While classroom dialogue is widely recognised as a crucial element of education, its analysis remains challenging due to the need for a nuanced understanding of dialogic functions and the labour-intensive nature of manual transcript coding. Recent advancements in large language models offer promising avenues for automating this process. However, existing studies predominantly focus on training large-scale models or evaluating pre-trained models with fixed codebooks, which are often not applicable or replicable for dialogue researchers working with small datasets or customised coding schemes. Using GPT-4's MyGPT agent as a case, this study evaluates its baseline performance in coding classroom dialogue with a human codebook and examines how performance varies with different example inputs through a variable control method. Through a design-based research approach, it identifies a set of practical strategies, based on MyGPT's unique features, for configuring effective agents with limited data. The findings suggest that, despite some limitations, a MyGPT agent developed with these strategies can serve as a useful coding assistant by generating coding suggestions. 

**Abstract (ZH)**: 本研究探究了开发定制化GPT代理以编码课堂对话的有效策略。尽管课堂对话被广泛认为是教育的关键组成部分，但由于需要对对话功能有精微的理解以及手动转录编码的劳动密集性，对其进行分析仍然具有挑战性。近年来，大型语言模型的进步为自动化这一过程提供了有希望的途径。然而，现有研究主要集中在训练大规模模型或使用固定编码本对预训练模型进行评估，这些往往不适用于处理小型数据集或定制编码方案的对话研究者。通过将GPT-4的MyGPT代理作为案例，本研究评估了其使用人类编码本编码课堂对话的基线性能，并通过变量控制方法考察了不同示例输入如何影响性能。通过设计导向的研究方法，本研究基于MyGPT的独特功能，识别出一套实用策略，以在有限数据情况下配置有效的代理。研究结果表明，尽管存在一些局限性，但使用这些策略开发的MyGPT代理可以作为一种有用的编码助手，通过生成编码建议发挥作用。 

---
# Mitigating Behavioral Hallucination in Multimodal Large Language Models for Sequential Images 

**Title (ZH)**: 多模态大型语言模型中序列图像行为幻觉的缓解 

**Authors**: Liangliang You, Junchi Yao, Shu Yang, Guimin Hu, Lijie Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07184)  

**Abstract**: While multimodal large language models excel at various tasks, they still suffer from hallucinations, which limit their reliability and scalability for broader domain applications. To address this issue, recent research mainly focuses on objective hallucination. However, for sequential images, besides objective hallucination, there is also behavioral hallucination, which is less studied. This work aims to fill in the gap. We first reveal that behavioral hallucinations mainly arise from two key factors: prior-driven bias and the snowball effect. Based on these observations, we introduce SHE (Sequence Hallucination Eradication), a lightweight, two-stage framework that (1) detects hallucinations via visual-textual alignment check using our proposed adaptive temporal window and (2) mitigates them via orthogonal projection onto the joint embedding space. We also propose a new metric (BEACH) to quantify behavioral hallucination severity. Empirical results on standard benchmarks demonstrate that SHE reduces behavioral hallucination by over 10% on BEACH while maintaining descriptive accuracy. 

**Abstract (ZH)**: 面向序列图像的幻觉消除：一种轻量级两阶段框架 

---
# BRIGHT+: Upgrading the BRIGHT Benchmark with MARCUS, a Multi-Agent RAG Clean-Up Suite 

**Title (ZH)**: BRIGHT+: 使用MARCUS多代理RAG清理套件升级BRIGHT基准测试 

**Authors**: Liyang Chen, Yujun Cai, Jieqiong Dong, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07116)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems require corpora that are both structurally clean and semantically coherent. BRIGHT is a recent and influential benchmark designed to evaluate complex multi-hop retrieval across diverse, high-reasoning domains. However, its practical effectiveness is limited by common web-crawled artifacts - such as content redundancy and semantic discontinuity - that impair retrieval accuracy and downstream reasoning. Notably, we find that such issues are concentrated in seven StackExchange-derived subdomains, while other domains (e.g., Coding and Theorem-based content) remain relatively clean.
In this study, we present MARCUS, a multi-agent pipeline that leverages large language models (LLMs) to systematically clean and re-chunk BRIGHT into a higher-quality corpus: BRIGHT-Plus. MARCUS applies dedicated agents for structural noise removal and semantic segmentation, preserving answer-bearing spans while improving contextual integrity. Experimental evaluations demonstrate that BRIGHT-Plus yields consistent and significant improvements in both retrieval accuracy and multi-hop reasoning across a diverse set of retrievers. We release both the BRIGHT-Plus corpus and the MARCUS pipeline to support future research on robust, reasoning-centric retrieval. 

**Abstract (ZH)**: Retrieval-Augmented Generation (RAG) 系统要求具有结构清晰和语义连贯的语料库。BRIGHT 是一个近期有影响力的基准，用于评估跨多种高推理领域复杂多跳检索的表现。然而，它的实际有效性由于常见的网页抓取伪影（如内容冗余和语义不连续）而受到限制，这些伪影影响了检索准确性和下游推理。值得注意的是，这些问题集中在七个源自 StackExchange 的子领域中，而其他领域（例如编程和基于定理的内容）则相对清洁。
在此研究中，我们提出了一种多agent管道 MARCUS，利用大规模语言模型 (LLMs) 系统地清洁并重新划分 BRIGHT，生成一个更高质量的语料库：BRIGHT-Plus。MARCUS 应用了专门的代理来去除结构噪声并进行语义分割，在保持答案携带片段的同时提升上下文完整性。实验评估表明，BRIGHT-Plus 在多种检索器中在检索准确性和多跳推理方面均表现出一致且显著的改进。我们发布了 BRIGHT-Plus 语料库和 MARCUS 管道，以支持未来针对稳健、以推理为中心的检索的研究。 

---
# Evaluating LLM-corrupted Crowdsourcing Data Without Ground Truth 

**Title (ZH)**: 评估受LLM影响的 crowdsourcing 数据的准确性无需地面 truth 

**Authors**: Yichi Zhang, Jinlong Pang, Zhaowei Zhu, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06991)  

**Abstract**: The recent success of generative AI highlights the crucial role of high-quality human feedback in building trustworthy AI systems. However, the increasing use of large language models (LLMs) by crowdsourcing workers poses a significant challenge: datasets intended to reflect human input may be compromised by LLM-generated responses. Existing LLM detection approaches often rely on high-dimension training data such as text, making them unsuitable for annotation tasks like multiple-choice labeling. In this work, we investigate the potential of peer prediction -- a mechanism that evaluates the information within workers' responses without using ground truth -- to mitigate LLM-assisted cheating in crowdsourcing with a focus on annotation tasks. Our approach quantifies the correlations between worker answers while conditioning on (a subset of) LLM-generated labels available to the requester. Building on prior research, we propose a training-free scoring mechanism with theoretical guarantees under a crowdsourcing model that accounts for LLM collusion. We establish conditions under which our method is effective and empirically demonstrate its robustness in detecting low-effort cheating on real-world crowdsourcing datasets. 

**Abstract (ZH)**: 生成式AI的Recent成功凸显了高质量人类反馈在构建可信赖AI系统中的关键作用。然而，众包工作者越来越多地使用大型语言模型（LLMs）带来了重大挑战：旨在反映人类输入的数据集可能因LLM生成的回答而受损。现有的LLM检测方法通常依赖于高维度训练数据如文本，这使得它们不适合标注任务如多选标注。在本文中，我们探讨了同事预测的潜力——一种不使用真实标签来评估工作者回答中信息的机制——以在众包标注任务中缓解LLM辅助的作弊问题。我们的方法在部分LM生成标签的条件下量化了工作者答案之间的相关性。基于先前的研究，我们提出了一种无需训练的评分机制，并在考虑LM合谋的众包模型下提供了理论保证。我们确定了该方法有效的工作条件，并通过真实世界众包数据集的实验证明了其在检测低努力作弊方面的鲁棒性。 

---
# The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity 

**Title (ZH)**: 思维的幻象：通过问题复杂性的视角理解推理模型的优势与局限 

**Authors**: Parshin Shojaee, Iman Mirzadeh, Keivan Alizadeh, Maxwell Horton, Samy Bengio, Mehrdad Farajtabar  

**Link**: [PDF](https://arxiv.org/pdf/2506.06941)  

**Abstract**: Recent generations of language models have introduced Large Reasoning Models (LRMs) that generate detailed thinking processes before providing answers. While these models demonstrate improved performance on reasoning benchmarks, their fundamental capabilities, scaling properties, and limitations remain insufficiently understood. Current evaluations primarily focus on established math and coding benchmarks, emphasizing final answer accuracy. However, this evaluation paradigm often suffers from contamination and does not provide insights into the reasoning traces. In this work, we systematically investigate these gaps with the help of controllable puzzle environments that allow precise manipulation of complexity while maintaining consistent logical structures. This setup enables the analysis of not only final answers but also the internal reasoning traces, offering insights into how LRMs think. Through extensive experiments, we show that LRMs face a complete accuracy collapse beyond certain complexities. Moreover, they exhibit a counterintuitive scaling limit: their reasoning effort increases with problem complexity up to a point, then declines despite having remaining token budget. By comparing LRMs with their standard LLM counterparts under same inference compute, we identify three performance regimes: (1) low-complexity tasks where standard models outperform LRMs, (2) medium-complexity tasks where LRMs demonstrates advantage, and (3) high-complexity tasks where both models face complete collapse. We found that LRMs have limitations in exact computation: they fail to use explicit algorithms and reason inconsistently across scales. We also investigate the reasoning traces in more depth, studying the patterns of explored solutions and analyzing the models' computational behavior, shedding light on their strengths, limitations, and raising questions about their reasoning capabilities. 

**Abstract (ZH)**: Recent Generations of Language Models Have Introduced Large Reasoning Models (LRMs) That Generate Detailed Thinking Processes Before Providing Answers: Systematically Investigating Their Fundamental Capabilities, Scaling Properties, and Limitations 

---
# An Agentic Framework for Autonomous Metamaterial Modeling and Inverse Design 

**Title (ZH)**: 自主元材料建模与逆向设计的能动框架 

**Authors**: Darui Lu, Jordan M. Malof, Willie J. Padilla  

**Link**: [PDF](https://arxiv.org/pdf/2506.06935)  

**Abstract**: Recent significant advances in integrating multiple Large Language Model (LLM) systems have enabled Agentic Frameworks capable of performing complex tasks autonomously, including novel scientific research. We develop and demonstrate such a framework specifically for the inverse design of photonic metamaterials. When queried with a desired optical spectrum, the Agent autonomously proposes and develops a forward deep learning model, accesses external tools via APIs for tasks like simulation and optimization, utilizes memory, and generates a final design via a deep inverse method. The framework's effectiveness is demonstrated in its ability to automate, reason, plan, and adapt. Notably, the Agentic Framework possesses internal reflection and decision flexibility, permitting highly varied and potentially novel outputs. 

**Abstract (ZH)**: 近期在整合多个大型语言模型系统方面的显著进展使其成为可能，构建出能够自主执行复杂任务的代理框架，包括新型科学研究。我们开发并展示了这样一种框架，专门用于光子 metamaterials 的逆向设计。当用期望的光谱查询时，代理自主地提出并开发一个前向深度学习模型，通过API访问外部工具进行仿真和优化任务，利用记忆，并通过深度逆向方法生成最终设计。该框架的有效性体现在其能够自动化、推理、规划和适应。值得注意的是，代理框架具有内部反馈和决策灵活性，能够产生高度多样且可能新颖的输出。 

---
# Boosting LLM Reasoning via Spontaneous Self-Correction 

**Title (ZH)**: 通过自发自我修正提升大模型推理能力 

**Authors**: Xutong Zhao, Tengyu Xu, Xuewei Wang, Zhengxing Chen, Di Jin, Liang Tan, Yen-Ting, Zishun Yu, Zhuokai Zhao, Yun He, Sinong Wang, Han Fang, Sarath Chandar, Chen Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06923)  

**Abstract**: While large language models (LLMs) have demonstrated remarkable success on a broad range of tasks, math reasoning remains a challenging one. One of the approaches for improving math reasoning is self-correction, which designs self-improving loops to let the model correct its own mistakes. However, existing self-correction approaches treat corrections as standalone post-generation refinements, relying on extra prompt and system designs to elicit self-corrections, instead of performing real-time, spontaneous self-corrections in a single pass. To address this, we propose SPOC, a spontaneous self-correction approach that enables LLMs to generate interleaved solutions and verifications in a single inference pass, with generation dynamically terminated based on verification outcomes, thereby effectively scaling inference time compute. SPOC considers a multi-agent perspective by assigning dual roles -- solution proposer and verifier -- to the same model. We adopt a simple yet effective approach to generate synthetic data for fine-tuning, enabling the model to develop capabilities for self-verification and multi-agent collaboration. We further improve its solution proposal and verification accuracy through online reinforcement learning. Experiments on mathematical reasoning benchmarks show that SPOC significantly improves performance. Notably, SPOC boosts the accuracy of Llama-3.1-8B and 70B Instruct models, achieving gains of 8.8% and 11.6% on MATH500, 10.0% and 20.0% on AMC23, and 3.3% and 6.7% on AIME24, respectively. 

**Abstract (ZH)**: 自发自我纠正方法：SPOC在数学推理中的应用 

---
# Causal Graph based Event Reasoning using Semantic Relation Experts 

**Title (ZH)**: 基于因果图的事件推理方法：语义关系专家的应用 

**Authors**: Mahnaz Koupaee, Xueying Bai, Mudan Chen, Greg Durrett, Nathanael Chambers, Niranjan Balasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2506.06910)  

**Abstract**: Understanding how events in a scenario causally connect with each other is important for effectively modeling and reasoning about events. But event reasoning remains a difficult challenge, and despite recent advances, Large Language Models (LLMs) still struggle to accurately identify causal connections between events. This struggle leads to poor performance on deeper reasoning tasks like event forecasting and timeline understanding. To address this challenge, we investigate the generation of causal event graphs (e.g., A enables B) as a parallel mechanism to help LLMs explicitly represent causality during inference. This paper evaluates both how to generate correct graphs as well as how graphs can assist reasoning. We propose a collaborative approach to causal graph generation where we use LLMs to simulate experts that focus on specific semantic relations. The experts engage in multiple rounds of discussions which are then consolidated by a final expert. Then, to demonstrate the utility of causal graphs, we use them on multiple downstream applications, and also introduce a new explainable event prediction task that requires a causal chain of events in the explanation. These explanations are more informative and coherent than baseline generations. Finally, our overall approach not finetuned on any downstream task, achieves competitive results with state-of-the-art models on both forecasting and next event prediction tasks. 

**Abstract (ZH)**: 理解场景中事件之间的因果联系对于有效建模和推理事件至关重要，但事件推理仍然是一个艰巨的挑战，尽管近期有进展，大语言模型（LLMs）仍然难以准确识别事件间的因果关系。这种困难导致在事件预测和时间线理解等更深层的推理任务上表现不佳。为应对这一挑战，我们研究因果事件图（例如A导致B）的生成作为平行机制，以帮助LLMs在推理过程中明确表示因果关系。本文评估了如何生成正确的图以及图如何辅助推理。我们提出了一种协作式的因果图生成方法，使用LLMs模拟专注于特定语义关系的专家。专家进行多轮讨论，然后由最终专家汇总。为展示因果图的实用性，我们将其应用于多个下游应用，并引入了一个新的可解释事件预测任务，该任务要求解释中包含因果链。这些解释比基线生成结果更具有信息性和连贯性。最终，我们的整体方法在任何下游任务上未进行微调，分别在事件预测和下一个事件预测任务上取得了与先进模型竞争力的结果。 

---
# Meta-Adaptive Prompt Distillation for Few-Shot Visual Question Answering 

**Title (ZH)**: 元自适应提示精简ethod及其在少样本视觉问答中的应用 

**Authors**: Akash Gupta, Amos Storkey, Mirella Lapata  

**Link**: [PDF](https://arxiv.org/pdf/2506.06905)  

**Abstract**: Large Multimodal Models (LMMs) often rely on in-context learning (ICL) to perform new tasks with minimal supervision. However, ICL performance, especially in smaller LMMs, is inconsistent and does not always improve monotonically with increasing examples. We hypothesize that this occurs due to the LMM being overwhelmed by additional information present in the image embeddings, which is not required for the downstream task. To address this, we propose a meta-learning approach that provides an alternative for inducing few-shot capabilities in LMMs, using a fixed set of soft prompts that are distilled from task-relevant image features and can be adapted at test time using a few examples. To facilitate this distillation, we introduce an attention-mapper module that can be easily integrated with the popular LLaVA v1.5 architecture and is jointly learned with soft prompts, enabling task adaptation in LMMs under low-data regimes with just a few gradient steps. Evaluation on the VL-ICL Bench shows that our method consistently outperforms ICL and related prompt-tuning approaches, even under image perturbations, improving task induction and reasoning across visual question answering tasks. 

**Abstract (ZH)**: 大型多模态模型中的少量样本学习：通过固定集合的软提示实现有效元学习 

---
# United Minds or Isolated Agents? Exploring Coordination of LLMs under Cognitive Load Theory 

**Title (ZH)**: 群体意识还是孤立的代理？探讨认知负荷理论下大型语言模型的协调机制 

**Authors**: HaoYang Shang, Xuan Liu, Zi Liang, Jie Zhang, Haibo Hu, Song Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.06843)  

**Abstract**: Large Language Models (LLMs) exhibit a notable performance ceiling on complex, multi-faceted tasks, as they often fail to integrate diverse information or adhere to multiple constraints. We posit that such limitation arises when the demands of a task exceed the LLM's effective cognitive load capacity. This interpretation draws a strong analogy to Cognitive Load Theory (CLT) in cognitive science, which explains similar performance boundaries in the human mind, and is further supported by emerging evidence that reveals LLMs have bounded working memory characteristics. Building upon this CLT-grounded understanding, we introduce CoThinker, a novel LLM-based multi-agent framework designed to mitigate cognitive overload and enhance collaborative problem-solving abilities. CoThinker operationalizes CLT principles by distributing intrinsic cognitive load through agent specialization and managing transactional load via structured communication and a collective working memory. We empirically validate CoThinker on complex problem-solving tasks and fabricated high cognitive load scenarios, demonstrating improvements over existing multi-agent baselines in solution quality and efficiency. Our analysis reveals characteristic interaction patterns, providing insights into the emergence of collective cognition and effective load management, thus offering a principled approach to overcoming LLM performance ceilings. 

**Abstract (ZH)**: 大型语言模型在复杂、多面任务中的表现受到认知负荷限制，往往难以整合多样信息或遵守多种约束。我们认为，这种限制源于任务需求超出模型有效认知负荷 capacity。这一解释与认知科学中的认知负荷理论(CLT)有紧密类比关系，该理论解释了人类思维中的类似表现边界，并进一步得到新兴证据的支持，表明大型语言模型具有有界的工作记忆特征。基于这一CLT基础的理解，我们引入了CoThinker，这是一种新颖的基于大型语言模型的多智能体框架，旨在减轻认知过载并增强协作问题解决能力。CoThinker通过智能体专业化分配内在认知负荷，并通过结构化通信和集体工作记忆管理事务性负荷来实现CLT原则。我们通过复杂问题解决任务和高认知负荷场景的实证验证，展示了CoThinker在解决方案质量和效率方面的改进，优于现有的多智能体基线。我们的分析揭示了特征交互模式，为集体认知的涌现和有效负荷管理提供了见解，从而提供了一种原则性的方法来克服大型语言模型的表现天花板。 

---
# Cross-Entropy Games for Language Models: From Implicit Knowledge to General Capability Measures 

**Title (ZH)**: 语言模型的交叉熵游戏：从隐含知识到一般能力衡量 

**Authors**: Clément Hongler, Andrew Emil  

**Link**: [PDF](https://arxiv.org/pdf/2506.06832)  

**Abstract**: Large Language Models (LLMs) define probability measures on text. By considering the implicit knowledge question of what it means for an LLM to know such a measure and what it entails algorithmically, we are naturally led to formulate a series of tasks that go beyond generative sampling, involving forms of summarization, counterfactual thinking, anomaly detection, originality search, reverse prompting, debating, creative solving, etc. These tasks can be formulated as games based on LLM measures, which we call Cross-Entropy (Xent) Games. Xent Games can be single-player or multi-player. They involve cross-entropy scores and cross-entropy constraints, and can be expressed as simple computational graphs and programs. We show the Xent Game space is large enough to contain a wealth of interesting examples, while being constructible from basic game-theoretic consistency axioms. We then discuss how the Xent Game space can be used to measure the abilities of LLMs. This leads to the construction of Xent Game measures: finite families of Xent Games that can be used as capability benchmarks, built from a given scope, by extracting a covering measure. To address the unbounded scope problem associated with the challenge of measuring general abilities, we propose to explore the space of Xent Games in a coherent fashion, using ideas inspired by evolutionary dynamics. 

**Abstract (ZH)**: Large Language Models (LLMs)定义了文本的概率测度。通过考虑LLM知道这样的测度意味着什么及其算法上的含义，自然地带我们提出了超越生成采样的一系列任务，包括总结、反事实思考、异常检测、原创性搜索、逆向提示、辩论、创造性解决等。这些任务可以基于LLM测度形式化为博弈，我们称之为交叉熵（Xent）博弈。Xent博弈可以是单人或多人博弈，涉及交叉熵得分和约束，可以用简单的计算图和程序表达。我们展示了Xent博弈空间足够大，包含丰富的有趣示例，并可以通过基本博弈论一致性公理构建。然后我们讨论了如何使用Xent博弈空间来衡量LLM的能力，从而构造Xent博弈测度：基于给定范围的有限Xent博弈族，可以用作能力基准，通过提取覆盖测度构建。为了解决衡量通用能力时面临的无限范围问题，我们提出以演化动力学启发的方法系统地探索Xent博弈空间。 

---
# AI PsyRoom: Artificial Intelligence Platform for Segmented Yearning and Reactive Outcome Optimization Method 

**Title (ZH)**: AI PsyRoom: 人工智能平台化的分段渴望与反应性结果优化方法 

**Authors**: Yigui Feng, Qinglin Wang, Ke Liu, Xinhai Chen, Bo Yang, Jie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06740)  

**Abstract**: Psychological counseling faces huge challenges due to the growing demand for mental health services and the shortage of trained professionals. Large language models (LLMs) have shown potential to assist psychological counseling, especially in empathy and emotional support. However, existing models lack a deep understanding of emotions and are unable to generate personalized treatment plans based on fine-grained emotions. To address these shortcomings, we present AI PsyRoom, a multi-agent simulation framework designed to enhance psychological counseling by generating empathetic and emotionally nuanced conversations. By leveraging fine-grained emotion classification and a multi-agent framework, we construct a multi-agent PsyRoom A for dialogue reconstruction, generating a high-quality dialogue dataset EmoPsy, which contains 35 sub-emotions, 423 specific emotion scenarios, and 12,350 dialogues. We also propose PsyRoom B for generating personalized treatment plans. Quantitative evaluations demonstrate that AI PsyRoom significantly outperforms state-of-the-art methods, achieving 18% improvement in problem orientation, 23% in expression, 24% in Empathy, and 16% in interactive communication quality. The datasets and models are publicly available, providing a foundation for advancing AI-assisted psychological counseling research. 

**Abstract (ZH)**: 人工智能心理房间：一种多Agent模拟框架，以增强心理咨询服务 

---
# VisioMath: Benchmarking Figure-based Mathematical Reasoning in LMMs 

**Title (ZH)**: VisioMath: LMMs中基于图形的数学推理benchmarking 

**Authors**: Can Li, Ting Zhang, Mei Wang, Hua Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06727)  

**Abstract**: Large Multimodal Models (LMMs) have demonstrated remarkable problem-solving capabilities across various domains. However, their ability to perform mathematical reasoning when answer options are represented as images--an essential aspect of multi-image comprehension--remains underexplored. To bridge this gap, we introduce VisioMath, a benchmark designed to evaluate mathematical reasoning in multimodal contexts involving image-based answer choices. VisioMath comprises 8,070 images and 1,800 multiple-choice questions, where each answer option is an image, presenting unique challenges to existing LMMs. To the best of our knowledge, VisioMath is the first dataset specifically tailored for mathematical reasoning in image-based-option scenarios, where fine-grained distinctions between answer choices are critical for accurate problem-solving. We systematically evaluate state-of-the-art LMMs on VisioMath and find that even the most advanced models struggle with this task. Notably, GPT-4o achieves only 45.9% accuracy, underscoring the limitations of current models in reasoning over visually similar answer choices. By addressing a crucial gap in existing benchmarks, VisioMath establishes a rigorous testbed for future research, driving advancements in multimodal reasoning. 

**Abstract (ZH)**: 大型多模态模型在处理基于图像的答案选项的数学推理能力方面存在不足：VisioMath数据集的构建与评价 

---
# WorldLLM: Improving LLMs' world modeling using curiosity-driven theory-making 

**Title (ZH)**: WorldLLM：通过好奇心驱动的理论构建提高大语言模型的world建模能力 

**Authors**: Guillaume Levy, Cedric Colas, Pierre-Yves Oudeyer, Thomas Carta, Clement Romac  

**Link**: [PDF](https://arxiv.org/pdf/2506.06725)  

**Abstract**: Large Language Models (LLMs) possess general world knowledge but often struggle to generate precise predictions in structured, domain-specific contexts such as simulations. These limitations arise from their inability to ground their broad, unstructured understanding in specific environments. To address this, we present WorldLLM, a framework that enhances LLM-based world modeling by combining Bayesian inference and autonomous active exploration with reinforcement learning. WorldLLM leverages the in-context learning abilities of LLMs to guide an LLM-based world model's predictions using natural language hypotheses given in its prompt. These hypotheses are iteratively refined through a Bayesian inference framework that leverages a second LLM as the proposal distribution given collected evidence. This evidence is collected using a curiosity-driven reinforcement learning policy that explores the environment to find transitions with a low log-likelihood under our LLM-based predictive model using the current hypotheses. By alternating between refining hypotheses and collecting new evidence, our framework autonomously drives continual improvement of the predictions. Our experiments demonstrate the effectiveness of WorldLLM in a textual game environment that requires agents to manipulate and combine objects. The framework not only enhances predictive accuracy, but also generates human-interpretable theories of environment dynamics. 

**Abstract (ZH)**: 基于贝叶斯推理和自主主动探索的大规模语言模型增强框架：WorldLLM及其在文本游戏环境中的应用 

---
# Contextual Experience Replay for Self-Improvement of Language Agents 

**Title (ZH)**: 基于情境体验重播的语言代理自改进方法 

**Authors**: Yitao Liu, Chenglei Si, Karthik Narasimhan, Shunyu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.06698)  

**Abstract**: Large language model (LLM) agents have been applied to sequential decision-making tasks such as web navigation, but without any environment-specific experiences, they often fail in these complex tasks. Moreover, current LLM agents are not designed to continually learn from past experiences during inference time, which could be crucial for them to gain these environment-specific experiences. To address this, we propose Contextual Experience Replay (CER), a training-free framework to enable efficient self-improvement for language agents in their context window. Specifically, CER accumulates and synthesizes past experiences into a dynamic memory buffer. These experiences encompass environment dynamics and common decision-making patterns, allowing the agents to retrieve and augment themselves with relevant knowledge in new tasks, enhancing their adaptability in complex environments. We evaluate CER on the challenging WebArena and VisualWebArena benchmarks. On VisualWebArena, CER achieves a competitive performance of 31.9%. On WebArena, CER also gets a competitive average success rate of 36.7%, relatively improving the success rate of the GPT-4o agent baseline by 51.0%. We also conduct a comprehensive analysis on it to prove its efficiency, validity and understand it better. 

**Abstract (ZH)**: 基于上下文的经验重放（CER）：一种无需训练的框架，使语言代理在上下文窗口内高效自改进 

---
# ScriptDoctor: Automatic Generation of PuzzleScript Games via Large Language Models and Tree Search 

**Title (ZH)**: ScriptDoctor: 通过大型语言模型和树搜索自动生成PuzzleScript游戏 

**Authors**: Sam Earle, Ahmed Khalifa, Muhammad Umair Nasir, Zehua Jiang, Graham Todd, Andrzej Banburski-Fahey, Julian Togelius  

**Link**: [PDF](https://arxiv.org/pdf/2506.06524)  

**Abstract**: There is much interest in using large pre-trained models in Automatic Game Design (AGD), whether via the generation of code, assets, or more abstract conceptualization of design ideas. But so far this interest largely stems from the ad hoc use of such generative models under persistent human supervision. Much work remains to show how these tools can be integrated into longer-time-horizon AGD pipelines, in which systems interface with game engines to test generated content autonomously. To this end, we introduce ScriptDoctor, a Large Language Model (LLM)-driven system for automatically generating and testing games in PuzzleScript, an expressive but highly constrained description language for turn-based puzzle games over 2D gridworlds. ScriptDoctor generates and tests game design ideas in an iterative loop, where human-authored examples are used to ground the system's output, compilation errors from the PuzzleScript engine are used to elicit functional code, and search-based agents play-test generated games. ScriptDoctor serves as a concrete example of the potential of automated, open-ended LLM-based workflows in generating novel game content. 

**Abstract (ZH)**: 基于大型预训练模型的自动游戏设计：ScriptDoctor系统在PuzzleScript中的应用与测试 

---
# Reinforcement Learning for Autonomous Warehouse Orchestration in SAP Logistics Execution: Redefining Supply Chain Agility 

**Title (ZH)**: 基于SAP Logistics Execution的自主仓库 orchestration reinforcement学习：重塑供应链灵活性 

**Authors**: Sumanth Pillella  

**Link**: [PDF](https://arxiv.org/pdf/2506.06523)  

**Abstract**: In an era of escalating supply chain demands, SAP Logistics Execution (LE) is pivotal for managing warehouse operations, transportation, and delivery. This research introduces a pioneering framework leveraging reinforcement learning (RL) to autonomously orchestrate warehouse tasks in SAP LE, enhancing operational agility and efficiency. By modeling warehouse processes as dynamic environments, the framework optimizes task allocation, inventory movement, and order picking in real-time. A synthetic dataset of 300,000 LE transactions simulates real-world warehouse scenarios, including multilingual data and operational disruptions. The analysis achieves 95% task optimization accuracy, reducing processing times by 60% compared to traditional methods. Visualizations, including efficiency heatmaps and performance graphs, guide agile warehouse strategies. This approach tackles data privacy, scalability, and SAP integration, offering a transformative solution for modern supply chains. 

**Abstract (ZH)**: 在供应链需求递增的时代，SAP物流执行（LE）对于管理仓库作业、运输和交付至关重要。本研究引入了一种利用强化学习（RL）的创新框架，以自主协调SAP LE中的仓库任务，提升运营灵活性和效率。通过将仓库流程建模为动态环境，该框架实现了实时的任务分配、库存移动和订单捡取优化。使用包含300,000条LE交易的合成数据集模拟实际仓库场景，包括多语言数据和运营中断。分析实现了95%的任务优化准确率，相比传统方法，处理时间减少60%。可视化工具，包括效率热图和性能图表，指导灵活的仓库策略。该方法解决数据隐私、扩展性和SAP集成问题，提供了一种现代供应链转型解决方案。 

---
# SIGMA: Refining Large Language Model Reasoning via Sibling-Guided Monte Carlo Augmentation 

**Title (ZH)**: SIGMA：通过兄弟引导的蒙特卡洛 augmentation 完善大型语言模型推理 

**Authors**: Yanwei Ren, Haotian Zhang, Fuxiang Wu, Jiayan Qiu, Jiaxing Huang, Baosheng Yu, Liu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06470)  

**Abstract**: Enhancing large language models by simply scaling up datasets has begun to yield diminishing returns, shifting the spotlight to data quality. Monte Carlo Tree Search (MCTS) has emerged as a powerful technique for generating high-quality chain-of-thought data, yet conventional approaches typically retain only the top-scoring trajectory from the search tree, discarding sibling nodes that often contain valuable partial insights, recurrent error patterns, and alternative reasoning strategies. This unconditional rejection of non-optimal reasoning branches may waste vast amounts of informative data in the whole search tree. We propose SIGMA (Sibling Guided Monte Carlo Augmentation), a novel framework that reintegrates these discarded sibling nodes to refine LLM reasoning. SIGMA forges semantic links among sibling nodes along each search path and applies a two-stage refinement: a critique model identifies overlooked strengths and weaknesses across the sibling set, and a revision model conducts text-based backpropagation to refine the top-scoring trajectory in light of this comparative feedback. By recovering and amplifying the underutilized but valuable signals from non-optimal reasoning branches, SIGMA substantially improves reasoning trajectories. On the challenging MATH benchmark, our SIGMA-tuned 7B model achieves 54.92% accuracy using only 30K samples, outperforming state-of-the-art models trained on 590K samples. This result highlights that our sibling-guided optimization not only significantly reduces data usage but also significantly boosts LLM reasoning. 

**Abstract (ZH)**: 通过重新整合搜索树中的弃用子节点来增强大型语言模型：基于子节点引导的蒙特卡洛 augmentation（SIGMA）方法 

---
# Memory OS of AI Agent 

**Title (ZH)**: AI代理的内存操作系统 

**Authors**: Jiazheng Kang, Mingming Ji, Zhe Zhao, Ting Bai  

**Link**: [PDF](https://arxiv.org/pdf/2506.06326)  

**Abstract**: Large Language Models (LLMs) face a crucial challenge from fixed context windows and inadequate memory management, leading to a severe shortage of long-term memory capabilities and limited personalization in the interactive experience with AI agents. To overcome this challenge, we innovatively propose a Memory Operating System, i.e., MemoryOS, to achieve comprehensive and efficient memory management for AI agents. Inspired by the memory management principles in operating systems, MemoryOS designs a hierarchical storage architecture and consists of four key modules: Memory Storage, Updating, Retrieval, and Generation. Specifically, the architecture comprises three levels of storage units: short-term memory, mid-term memory, and long-term personal memory. Key operations within MemoryOS include dynamic updates between storage units: short-term to mid-term updates follow a dialogue-chain-based FIFO principle, while mid-term to long-term updates use a segmented page organization strategy. Our pioneering MemoryOS enables hierarchical memory integration and dynamic updating. Extensive experiments on the LoCoMo benchmark show an average improvement of 49.11% on F1 and 46.18% on BLEU-1 over the baselines on GPT-4o-mini, showing contextual coherence and personalized memory retention in long conversations. The implementation code is open-sourced at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）面临固定上下文窗口和内存管理不足的挑战，导致长期记忆能力严重短缺和与AI代理互动体验的个性化受限。为克服这一挑战，我们创新地提出了一个内存操作系统，即MemoryOS，以实现对AI代理的全面高效内存管理。受操作系统内存管理原则的启发，MemoryOS 设计了分层存储架构，并包含四个关键模块：内存存储、更新、检索和生成。具体而言，架构包括三个级别的存储单元：短期记忆、中期记忆和长期个性化记忆。MemoryOS 中的关键操作包括存储单元之间的动态更新：短期到中期的更新遵循基于对话链的FIFO原则，而中期到长期的更新采用分段页面组织策略。我们的开创性 MemoryOS 实现了分层记忆集成和动态更新。在 LoCoMo 基准上的 extensive 实验表明，与 GPT-4o-mini 的基线相比，F1 平均提高了 49.11%，BLEU-1 提高了 46.18%，显示出长对话中的上下文一致性和个性化记忆保留。源代码在此处开放获取：this https URL。 

---
# Large Language Models and Their Applications in Roadway Safety and Mobility Enhancement: A Comprehensive Review 

**Title (ZH)**: 大型语言模型及其在道路安全与通行能力提升中的应用：一项全面综述 

**Authors**: Muhammad Monjurul Karim, Yan Shi, Shucheng Zhang, Bingzhang Wang, Mehrdad Nasri, Yinhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06301)  

**Abstract**: Roadway safety and mobility remain critical challenges for modern transportation systems, demanding innovative analytical frameworks capable of addressing complex, dynamic, and heterogeneous environments. While traditional engineering methods have made progress, the complexity and dynamism of real-world traffic necessitate more advanced analytical frameworks. Large Language Models (LLMs), with their unprecedented capabilities in natural language understanding, knowledge integration, and reasoning, represent a promising paradigm shift. This paper comprehensively reviews the application and customization of LLMs for enhancing roadway safety and mobility. A key focus is how LLMs are adapted -- via architectural, training, prompting, and multimodal strategies -- to bridge the "modality gap" with transportation's unique spatio-temporal and physical data. The review systematically analyzes diverse LLM applications in mobility (e.g., traffic flow prediction, signal control) and safety (e.g., crash analysis, driver behavior assessment,). Enabling technologies such as V2X integration, domain-specific foundation models, explainability frameworks, and edge computing are also examined. Despite significant potential, challenges persist regarding inherent LLM limitations (hallucinations, reasoning deficits), data governance (privacy, bias), deployment complexities (sim-to-real, latency), and rigorous safety assurance. Promising future research directions are highlighted, including advanced multimodal fusion, enhanced spatio-temporal reasoning, human-AI collaboration, continuous learning, and the development of efficient, verifiable systems. This review provides a structured roadmap of current capabilities, limitations, and opportunities, underscoring LLMs' transformative potential while emphasizing the need for responsible innovation to realize safer, more intelligent transportation systems. 

**Abstract (ZH)**: 道路安全与流动性仍然是现代交通系统的关键挑战，要求具备解决复杂、动态和异质环境的创新分析框架。尽管传统工程方法取得了进展，但实际交通的复杂性和动态性需要更先进的分析框架。大型语言模型（LLMs）凭借其前所未有的自然语言理解和知识整合及推理能力，代表了一个有前景的范式转变。本文全面回顾了LLMs在提升道路安全和流动性方面的应用与定制。重点是如何通过架构、训练、提示和多模态策略来适应交通的独特时空和物理数据，以解决“模态差距”。回顾系统地分析了在流动性（如交通流量预测、信号控制）和安全（如碰撞分析、驾驶行为评估）方面应用的多样化LLM应用。还考察了诸如车路协同（V2X）、领域特定基础模型、解释性框架和边缘计算等使能技术。尽管存在巨大潜力，但固有的LLM限制（如幻觉、推理缺陷）、数据治理（如隐私、偏见）、部署复杂性（如仿真到现实、延迟）和严格的安全保障等问题仍存在挑战。指出了具有前景的未来研究方向，包括先进的多模态融合、增强的空间-时间推理、人类与人工智能协作、持续学习以及开发高效、可验证的系统。本回顾为当前能力和局限性提供了结构化的路线图，突显了LLMs的变革潜力，同时强调负责任创新以实现更安全、更智能的交通系统的重要性。 

---
# Deep Research Bench: Evaluating AI Web Research Agents 

**Title (ZH)**: 深度研究台：评估AI网络研究代理 

**Authors**: FutureSearch, Nikos I. Bosse, Jon Evans, Robert G. Gambee, Daniel Hnyk, Peter Mühlbacher, Lawrence Phillips, Dan Schwarz, Jack Wildman  

**Link**: [PDF](https://arxiv.org/pdf/2506.06287)  

**Abstract**: Amongst the most common use cases of modern AI is LLM chat with web search enabled. However, no direct evaluations of the quality of web research agents exist that control for the continually-changing web. We introduce Deep Research Bench, consisting of 89 multi-step web research task instances of varying difficulty across 8 diverse task categories, with the answers carefully worked out by skilled humans. We provide a "RetroSearch" environment with a large frozen set of scraped web pages, and demonstrate that offline "RetroSearch" agents perform comparably to "live web" agents, enabling reliable evaluations of models over time. We provide robust agent tooling and scaffolding to benchmark major LLMs as they are released, including "thinking" models like o3 and Gemini 2.5 Pro. We include automated evaluations of the lengthy agent traces to report progress over time in hallucinations, tool use, and forgetting. Finally, we evaluate the major web research products branded as "Deep Research", "Deep Search", "Search", or "Research." Results are available on a public leaderboard at this https URL. 

**Abstract (ZH)**: 现代AI中最常见的用例之一是具有网络搜索功能的大语言模型对话，然而，没有针对不断变化的网络进行控制的质量评估。我们引入了Deep Research Bench，包括89个不同难度级别的多层次网络研究任务实例，涵盖8个不同的任务类别，并由熟练的人类仔细给出答案。我们提供了一个“RetroSearch”环境，包含大量的冻结网页快照，并证明了离线“RetroSearch”代理与“实时网络”代理表现相当，从而可以在不同时期对模型进行可靠的评估。我们提供了强大的代理工具和结构化方案，以基准测试即将发布的主要大语言模型，包括“思考”模型如o3和Gemini 2.5 Pro。我们还包括了对漫长代理轨迹的自动化评估，以报告随时间推移在幻觉、工具使用和遗忘方面的进展。最后，我们评估了那些标记为“Deep Research”、“Deep Search”、“Search”或“Research”的主要网络研究产品。结果可在以下公共排行榜上查阅：this https URL。 

---
# Hidden in plain sight: VLMs overlook their visual representations 

**Title (ZH)**: 明目张胆的隐藏：VLMs忽视了它们的视觉表示 

**Authors**: Stephanie Fu, Tyler Bonnen, Devin Guillory, Trevor Darrell  

**Link**: [PDF](https://arxiv.org/pdf/2506.08008)  

**Abstract**: Language provides a natural interface to specify and evaluate performance on visual tasks. To realize this possibility, vision language models (VLMs) must successfully integrate visual and linguistic information. Our work compares VLMs to a direct readout of their visual encoders to understand their ability to integrate across these modalities. Across a series of vision-centric benchmarks (e.g., depth estimation, correspondence), we find that VLMs perform substantially worse than their visual encoders, dropping to near-chance performance. We investigate these results through a series of analyses across the entire VLM: namely 1) the degradation of vision representations, 2) brittleness to task prompt, and 3) the language model's role in solving the task. We find that the bottleneck in performing these vision-centric tasks lies in this third category; VLMs are not effectively using visual information easily accessible throughout the entire model, and they inherit the language priors present in the LLM. Our work helps diagnose the failure modes of open-source VLMs, and presents a series of evaluations useful for future investigations into visual understanding within VLMs. 

**Abstract (ZH)**: 语言为指定和评估视觉任务的性能提供了一个自然接口。为了实现这一可能性，视觉语言模型（VLMs）必须成功地整合视觉和语言信息。我们的工作将VLMs与它们的视觉编码器的直接读出进行比较，以理解它们在跨模态整合方面的能力。在一系列以视觉为中心的标准测试中（例如，深度估计、对应关系），我们发现VLMs的表现远逊于其视觉编码器，性能几乎降到随机水平。我们通过对整个VLM的多个分析来探讨这些结果，即1）视觉表示的退化、2）对任务提示的脆弱性，以及3）语言模型在解决问题中的作用。我们发现，在执行这些视觉中心任务时的瓶颈在于第三类；VLMs未能有效地利用模型中到处可用的视觉信息，并且继承了LLM中存在的语言先验。我们的工作有助于诊断开源VLMs的失败模式，并提出一系列对于未来关于VLM中视觉理解的研究有用的评估方法。 

---
# Reparameterized LLM Training via Orthogonal Equivalence Transformation 

**Title (ZH)**: 通过正交等价变换实现的重参数化大模型训练 

**Authors**: Zeju Qiu, Simon Buchholz, Tim Z. Xiao, Maximilian Dax, Bernhard Schölkopf, Weiyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08001)  

**Abstract**: While large language models (LLMs) are driving the rapid advancement of artificial intelligence, effectively and reliably training these large models remains one of the field's most significant challenges. To address this challenge, we propose POET, a novel reParameterized training algorithm that uses Orthogonal Equivalence Transformation to optimize neurons. Specifically, POET reparameterizes each neuron with two learnable orthogonal matrices and a fixed random weight matrix. Because of its provable preservation of spectral properties of weight matrices, POET can stably optimize the objective function with improved generalization. We further develop efficient approximations that make POET flexible and scalable for training large-scale neural networks. Extensive experiments validate the effectiveness and scalability of POET in training LLMs. 

**Abstract (ZH)**: POET：使用正交等价变换优化神经元的参数化训练算法 

---
# HeuriGym: An Agentic Benchmark for LLM-Crafted Heuristics in Combinatorial Optimization 

**Title (ZH)**: HeuriGym：组合优化中由大语言模型生成启发式规则的能力基准 

**Authors**: Hongzheng Chen, Yingheng Wang, Yaohui Cai, Hins Hu, Jiajie Li, Shirley Huang, Chenhui Deng, Rongjian Liang, Shufeng Kong, Haoxing Ren, Samitha Samaranayake, Carla P. Gomes, Zhiru Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07972)  

**Abstract**: While Large Language Models (LLMs) have demonstrated significant advancements in reasoning and agent-based problem-solving, current evaluation methodologies fail to adequately assess their capabilities: existing benchmarks either rely on closed-ended questions prone to saturation and memorization, or subjective comparisons that lack consistency and rigor. In this work, we introduce HeuriGym, an agentic framework designed for evaluating heuristic algorithms generated by LLMs for combinatorial optimization problems, characterized by clearly defined objectives and expansive solution spaces. HeuriGym empowers LLMs to propose heuristics, receive evaluative feedback via code execution, and iteratively refine their solutions. We evaluate nine state-of-the-art models on nine problems across domains such as computer systems, logistics, and biology, exposing persistent limitations in tool use, planning, and adaptive reasoning. To quantify performance, we propose the Quality-Yield Index (QYI), a metric that captures both solution pass rate and quality. Even top models like GPT-o4-mini-high and Gemini-2.5-Pro attain QYI scores of only 0.6, well below the expert baseline of 1. Our open-source benchmark aims to guide the development of LLMs toward more effective and realistic problem-solving in scientific and engineering domains. 

**Abstract (ZH)**: 大型语言模型在推理和基于代理的问题解决方面取得了显著进展，但现有的评估方法未能充分评估其能力：现有的基准测试要么依赖于容易饱和且依赖记忆的封闭式问题，要么依赖于缺乏一致性和 rigor 的主观比较。本文介绍了一种名为 HeuriGym 的代理框架，用于评估大型语言模型生成的启发式算法在组合优化问题上的表现，这些问题具有明确的目标和广阔的解决方案空间。HeuriGym 使大型语言模型能够提出启发式算法，通过代码执行接收评估反馈，并逐步优化其解决方案。我们评估了九个最先进的模型在涉及计算机系统、物流和生物学等多个领域的九个问题上的表现，揭示了工具使用、计划和适应性推理方面的持久限制。为了量化性能，我们提出了质量产率指数（QYI），该指标涵盖了解决方案通过率和质量。即使是顶级模型如 GPT-o4-mini-high 和 Gemini-2.5-Pro，其 QYI 也仅达到 0.6，明显低于专家基准的 1。我们的开源基准旨在引导大型语言模型朝向更有效和现实的问题解决方向发展。 

---
# SlideCoder: Layout-aware RAG-enhanced Hierarchical Slide Generation from Design 

**Title (ZH)**: SlideCoder: 带有布局意识的层次化幻灯片生成增强的RAG方法 

**Authors**: Wenxin Tang, Jingyu Xiao, Wenxuan Jiang, Xi Xiao, Yuhang Wang, Xuxin Tang, Qing Li, Yuehe Ma, Junliang Liu, Shisong Tang, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07964)  

**Abstract**: Manual slide creation is labor-intensive and requires expert prior knowledge. Existing natural language-based LLM generation methods struggle to capture the visual and structural nuances of slide designs. To address this, we formalize the Reference Image to Slide Generation task and propose Slide2Code, the first benchmark with difficulty-tiered samples based on a novel Slide Complexity Metric. We introduce SlideCoder, a layout-aware, retrieval-augmented framework for generating editable slides from reference images. SlideCoder integrates a Color Gradient-based Segmentation algorithm and a Hierarchical Retrieval-Augmented Generation method to decompose complex tasks and enhance code generation. We also release SlideMaster, a 7B open-source model fine-tuned with improved reverse-engineered data. Experiments show that SlideCoder outperforms state-of-the-art baselines by up to 40.5 points, demonstrating strong performance across layout fidelity, execution accuracy, and visual consistency. Our code is available at this https URL. 

**Abstract (ZH)**: 手动制作幻灯片劳动密集且需要专家先验知识。现有的基于自然语言的LLM生成方法难以捕捉幻灯片设计的视觉和结构细腻之处。为了解决这一问题，我们形式化了参考图像到幻灯片生成任务，并提出了Slide2Code，这是首个基于新型幻灯片复杂度度量的难度分层样本基准。我们引入了SlideCoder，这是一种布局感知、检索增强的框架，用于从参考图像生成可编辑的幻灯片。SlideCoder结合了基于颜色渐变的分割算法和分层检索增强生成方法来分解复杂任务并增强代码生成。我们还推出了SlideMaster，这是一个7B开源模型，经过改进的逆向工程数据微调。实验结果显示，SlideCoder在布局保真度、执行准确性和视觉一致性方面均表现出色，比最先进的基线方法高出40.5分。我们的代码可在以下链接获得。 

---
# Correlated Errors in Large Language Models 

**Title (ZH)**: 大型语言模型中的相关错误 

**Authors**: Elliot Kim, Avi Garg, Kenny Peng, Nikhil Garg  

**Link**: [PDF](https://arxiv.org/pdf/2506.07962)  

**Abstract**: Diversity in training data, architecture, and providers is assumed to mitigate homogeneity in LLMs. However, we lack empirical evidence on whether different LLMs differ meaningfully. We conduct a large-scale empirical evaluation on over 350 LLMs overall, using two popular leaderboards and a resume-screening task. We find substantial correlation in model errors -- on one leaderboard dataset, models agree 60% of the time when both models err. We identify factors driving model correlation, including shared architectures and providers. Crucially, however, larger and more accurate models have highly correlated errors, even with distinct architectures and providers. Finally, we show the effects of correlation in two downstream tasks: LLM-as-judge evaluation and hiring -- the latter reflecting theoretical predictions regarding algorithmic monoculture. 

**Abstract (ZH)**: 训练数据、架构和提供者的多样性被假设能减轻大语言模型的同质性。然而，我们缺乏关于不同大语言模型之间是否存在有意义差异的实证证据。我们对超过350个大语言模型进行了大规模实证评估，使用了两个流行的排行榜和一份简历筛选任务。我们发现模型错误之间有显著的相关性——在其中一个排行榜数据集中，当两个模型都出错时，它们一致的错误比例占60%。我们确定了驱动模型相关性的因素，包括共享的架构和提供者。然而，至关重要的是，即使使用不同的架构和提供者，更大的和更准确的模型也具有高度相关的错误。最后，我们展示了相关性在两个下游任务中的影响：大语言模型作为评判者的评估和招聘——后者反映了关于算法 monoculture 的理论预测。 

---
# MiniCPM4: Ultra-Efficient LLMs on End Devices 

**Title (ZH)**: MiniCPM4: 末梢设备上的超高效大语言模型 

**Authors**: MiniCPM Team, Chaojun Xiao, Yuxuan Li, Xu Han, Yuzhuo Bai, Jie Cai, Haotian Chen, Wentong Chen, Xin Cong, Ganqu Cui, Ning Ding, Shengdan Fan, Yewei Fang, Zixuan Fu, Wenyu Guan, Yitong Guan, Junshao Guo, Yufeng Han, Bingxiang He, Yuxiang Huang, Cunliang Kong, Qiuzuo Li, Siyuan Li, Wenhao Li, Yanghao Li, Yishan Li, Zhen Li, Dan Liu, Biyuan Lin, Yankai Lin, Xiang Long, Quanyu Lu, Yaxi Lu, Peiyan Luo, Hongya Lyu, Litu Ou, Yinxu Pan, Zekai Qu, Qundong Shi, Zijun Song, Jiayuan Su, Zhou Su, Ao Sun, Xianghui Sun, Peijun Tang, Fangzheng Wang, Feng Wang, Shuo Wang, Yudong Wang, Yesai Wu, Zhenyu Xiao, Jie Xie, Zihao Xie, Yukun Yan, Jiarui Yuan, Kaihuo Zhang, Lei Zhang, Linyue Zhang, Xueren Zhang, Yudi Zhang, Hengyu Zhao, Weilin Zhao, Weilun Zhao, Yuanqian Zhao, Zhi Zheng, Ge Zhou, Jie Zhou, Wei Zhou, Zihan Zhou, Zixuan Zhou, Zhiyuan Liu, Guoyang Zeng, Chao Jia, Dahai Li, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.07900)  

**Abstract**: This paper introduces MiniCPM4, a highly efficient large language model (LLM) designed explicitly for end-side devices. We achieve this efficiency through systematic innovation in four key dimensions: model architecture, training data, training algorithms, and inference systems. Specifically, in terms of model architecture, we propose InfLLM v2, a trainable sparse attention mechanism that accelerates both prefilling and decoding phases for long-context processing. Regarding training data, we propose UltraClean, an efficient and accurate pre-training data filtering and generation strategy, and UltraChat v2, a comprehensive supervised fine-tuning dataset. These datasets enable satisfactory model performance to be achieved using just 8 trillion training tokens. Regarding training algorithms, we propose ModelTunnel v2 for efficient pre-training strategy search, and improve existing post-training methods by introducing chunk-wise rollout for load-balanced reinforcement learning and data-efficient tenary LLM, BitCPM. Regarding inference systems, we propose this http URL that integrates sparse attention, model quantization, and speculative sampling to achieve efficient prefilling and decoding. To meet diverse on-device requirements, MiniCPM4 is available in two versions, with 0.5B and 8B parameters, respectively. Sufficient evaluation results show that MiniCPM4 outperforms open-source models of similar size across multiple benchmarks, highlighting both its efficiency and effectiveness. Notably, MiniCPM4-8B demonstrates significant speed improvements over Qwen3-8B when processing long sequences. Through further adaptation, MiniCPM4 successfully powers diverse applications, including trustworthy survey generation and tool use with model context protocol, clearly showcasing its broad usability. 

**Abstract (ZH)**: MiniCPM4：一种专为端侧设备设计的高效大型语言模型 

---
# Improving large language models with concept-aware fine-tuning 

**Title (ZH)**: 概念意识微调改进大型语言模型 

**Authors**: Michael K. Chen, Xikun Zhang, Jiaxing Huang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.07833)  

**Abstract**: Large language models (LLMs) have become the cornerstone of modern AI. However, the existing paradigm of next-token prediction fundamentally limits their ability to form coherent, high-level concepts, making it a critical barrier to human-like understanding and reasoning. Take the phrase "ribonucleic acid" as an example: an LLM will first decompose it into tokens, i.e., artificial text fragments ("rib", "on", ...), then learn each token sequentially, rather than grasping the phrase as a unified, coherent semantic entity. This fragmented representation hinders deeper conceptual understanding and, ultimately, the development of truly intelligent systems. In response, we introduce Concept-Aware Fine-Tuning (CAFT), a novel multi-token training method that redefines how LLMs are fine-tuned. By enabling the learning of sequences that span multiple tokens, this method fosters stronger concept-aware learning. Our experiments demonstrate significant improvements compared to conventional next-token finetuning methods across diverse tasks, including traditional applications like text summarization and domain-specific ones like de novo protein design. Multi-token prediction was previously only possible in the prohibitively expensive pretraining phase; CAFT, to our knowledge, is the first to bring the multi-token setting to the post-training phase, thus effectively democratizing its benefits for the broader community of practitioners and researchers. Finally, the unexpected effectiveness of our proposed method suggests wider implications for the machine learning research community. All code and data are available at this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）已成为现代人工智能的基石。然而，现有的下一token预测范式从根本上限制了它们形成连贯且高层次概念的能力，成为实现类人理解与推理的关键障碍。例如，对于短语“ribonucleic acid”，LLM会首先将其分解为token，即人工文本片段（“rib”，“on”，...），然后顺序学习每个token，而不是将整个短语视为一个统一且连贯的语义实体。这种分割表示阻碍了更深层次的概念理解，并最终阻碍了真正智能系统的开发。为此，我们提出了一种名为概念意识微调（CAFT）的新颖多token训练方法，重新定义了LLM的微调方式。通过使模型能够学习跨越多个token的序列，这种方法促进了更强的概念意识学习。我们的实验表明，与传统的下一token微调方法相比，在多种任务上取得了显著的改进，包括传统的文本摘要等应用以及特定领域的蛋白质从头设计等应用。多token预测在过去只能在成本高昂的预训练阶段实现；据我们所知，CAFT是首个将多token设置引入后训练阶段的方法，从而有效普及了其益处，惠及更广泛的实践者和研究人员。最后，我们提出的方法出乎意料的有效性暗示了对机器学习研究社区更广泛影响。所有代码和数据均可在以下链接获得。 

---
# Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking 

**Title (ZH)**: 通过强化抽象思维增强LLMs的推理能力 

**Authors**: Silin Gao, Antoine Bosselut, Samy Bengio, Emmanuel Abbe  

**Link**: [PDF](https://arxiv.org/pdf/2506.07751)  

**Abstract**: Recent studies have shown that large language models (LLMs), especially smaller ones, often lack robustness in their reasoning. I.e., they tend to experience performance drops when faced with distribution shifts, such as changes to numerical or nominal variables, or insertions of distracting clauses. A possible strategy to address this involves generating synthetic data to further "instantiate" reasoning problems on potential variations. In contrast, our approach focuses on "abstracting" reasoning problems. This not only helps counteract distribution shifts but also facilitates the connection to symbolic tools for deriving solutions. We find that this abstraction process is better acquired through reinforcement learning (RL) than just supervised fine-tuning, which often fails to produce faithful abstractions. Our method, AbstraL -- which promotes abstract reasoning in LLMs using RL on granular abstraction data -- significantly mitigates performance degradation on recent GSM perturbation benchmarks. 

**Abstract (ZH)**: 近期研究表明，大型语言模型（LLMs），尤其是较小的模型，经常在其推理过程中缺乏稳健性。即，当面对分布偏移时（如数值或名义变量的变化，或插入干扰性从句），它们往往会表现出性能下降。一种可能的策略是生成合成数据以进一步“实例化”潜在变化的推理问题。相比之下，我们 approaches 的重点在于“抽象化”推理问题。这不仅有助于抵消分布偏移，还促进了与符号工具连接以推导解决方案。我们发现，通过强化学习（RL）而非仅仅监督微调来获得抽象过程更为有效，后者往往无法产生忠实的抽象。我们的方法 AbstraL —— 该方法使用 RL 在粒度抽象数据上促进 LLMs 的抽象推理 —— 显著减轻了近期 GSM 干扰基准上的性能下降。 

---
# ETA: Efficiency through Thinking Ahead, A Dual Approach to Self-Driving with Large Models 

**Title (ZH)**: ETA: 通过前瞻思维提升效率，一种基于大型模型的自动驾驶双重视角方法 

**Authors**: Shadi Hamdan, Chonghao Sima, Zetong Yang, Hongyang Li, Fatma Güney  

**Link**: [PDF](https://arxiv.org/pdf/2506.07725)  

**Abstract**: How can we benefit from large models without sacrificing inference speed, a common dilemma in self-driving systems? A prevalent solution is a dual-system architecture, employing a small model for rapid, reactive decisions and a larger model for slower but more informative analyses. Existing dual-system designs often implement parallel architectures where inference is either directly conducted using the large model at each current frame or retrieved from previously stored inference results. However, these works still struggle to enable large models for a timely response to every online frame. Our key insight is to shift intensive computations of the current frame to previous time steps and perform a batch inference of multiple time steps to make large models respond promptly to each time step. To achieve the shifting, we introduce Efficiency through Thinking Ahead (ETA), an asynchronous system designed to: (1) propagate informative features from the past to the current frame using future predictions from the large model, (2) extract current frame features using a small model for real-time responsiveness, and (3) integrate these dual features via an action mask mechanism that emphasizes action-critical image regions. Evaluated on the Bench2Drive CARLA Leaderboard-v2 benchmark, ETA advances state-of-the-art performance by 8% with a driving score of 69.53 while maintaining a near-real-time inference speed at 50 ms. 

**Abstract (ZH)**: 如何在不牺牲推理速度的情况下受益于大规模模型：一种用于自动驾驶系统的前瞻高效双系统架构 

---
# GaRAGe: A Benchmark with Grounding Annotations for RAG Evaluation 

**Title (ZH)**: GaRAGe: 一种用于RAG评估的grounding注释基准 

**Authors**: Ionut-Teodor Sorodoc, Leonardo F. R. Ribeiro, Rexhina Blloshmi, Christopher Davis, Adrià de Gispert  

**Link**: [PDF](https://arxiv.org/pdf/2506.07671)  

**Abstract**: We present GaRAGe, a large RAG benchmark with human-curated long-form answers and annotations of each grounding passage, allowing a fine-grained evaluation of whether LLMs can identify relevant grounding when generating RAG answers. Our benchmark contains 2366 questions of diverse complexity, dynamism, and topics, and includes over 35K annotated passages retrieved from both private document sets and the Web, to reflect real-world RAG use cases. This makes it an ideal test bed to evaluate an LLM's ability to identify only the relevant information necessary to compose a response, or provide a deflective response when there is insufficient information. Evaluations of multiple state-of-the-art LLMs on GaRAGe show that the models tend to over-summarise rather than (a) ground their answers strictly on the annotated relevant passages (reaching at most a Relevance-Aware Factuality Score of 60%), or (b) deflect when no relevant grounding is available (reaching at most 31% true positive rate in deflections). The F1 in attribution to relevant sources is at most 58.9%, and we show that performance is particularly reduced when answering time-sensitive questions and when having to draw knowledge from sparser private grounding sources. 

**Abstract (ZH)**: GaRAGe：一个大型RAG基准，包含人工策展的长形式答案和每段接地内容的标注，以精细评估LLMs在生成RAG答案时是否能识别相关接地信息 

---
# Synthesis by Design: Controlled Data Generation via Structural Guidance 

**Title (ZH)**: 设计合成：通过结构指导实现受控数据生成 

**Authors**: Lei Xu, Sirui Chen, Yuxuan Huang, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07664)  

**Abstract**: Mathematical reasoning remains challenging for LLMs due to complex logic and the need for precise computation. Existing methods enhance LLM reasoning by synthesizing datasets through problem rephrasing, but face issues with generation quality and problem complexity. To address this, we propose to extract structural information with generated problem-solving code from mathematical reasoning and guide data generation with structured solutions. Applied to MATH and GSM8K, our approach produces 39K problems with labeled intermediate steps and a 6.1K-problem benchmark of higher difficulty. Results on our benchmark show that model performance declines as reasoning length increases. Additionally, we conducted fine-tuning experiments using the proposed training data on a range of LLMs, and the results validate the effectiveness of our dataset. We hope the proposed method and dataset will contribute to future research in enhancing LLM reasoning capabilities. 

**Abstract (ZH)**: 数学推理对于LLMs仍然是具有挑战性的，因为涉及到复杂的逻辑和精确的计算需求。现有方法通过重新表述问题来合成数据集以增强LLM的推理能力，但面临生成质量和问题复杂性的问题。为此，我们提出从数学推理中提取生成的问题解决代码的结构信息，并利用结构化解决方案指导数据生成。在MATH和GSM8K上应用此方法，产生了39,000个带有标注中间步骤的问题，并构建了一个包含6,100个更高难度问题的基准数据集。在该基准数据集上的结果显示，随着推理长度的增加，模型性能下降。此外，我们使用提出的训练数据对一系列LLMs进行了微调实验，实验结果验证了我们数据集的有效性。我们希望提出的该方法和数据集能够促进未来研究以增强LLM的推理能力。 

---
# LoRMA: Low-Rank Multiplicative Adaptation for LLMs 

**Title (ZH)**: LoRMA：低秩乘法适应性方法用于LLMs 

**Authors**: Harsh Bihany, Shubham Patel, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07621)  

**Abstract**: Large Language Models have shown remarkable capabilities in the NLP domain. Their effectiveness can mainly be attributed to their ability to adapt to an array of downstream tasks. However, generally, full fine-tuning is a computationally expensive job. To mitigate this, many techniques have been developed that prime efficiency, a prominent one being Low-Rank Adaptation (LoRA). However, LoRA and its variants employ re-parametrized additive updates. In this paper, we propose Low-Rank Multiplicative Adaptation (LoRMA), which shifts the paradigm of additive updates to a richer space of matrix multiplicative transformations. We tackle challenges such as computational complexity and rank bottleneck of matrix multiplication by effectively re-ordering operations and introducing rank inflation strategies. We conduct extensive experiments to demonstrate the effectiveness of our approach in terms of various evaluation metrics. 

**Abstract (ZH)**: 大型语言模型在NLP领域展现了显著的能力。它们的有效性主要归因于其适应多种下游任务的能力。然而，通常完整的微调是一个计算成本高昂的过程。为解决这一问题，许多技术被开发出来以提高效率，其中一种显著的技术是低秩适应（LoRA）。然而，LoRA及其变体采用的是重参数化的加性更新。本文中，我们提出了一种低秩乘性适应（LoRMA），将加性更新的范式转移到更丰富的矩阵乘性变换空间。通过有效重排操作和引入秩膨胀策略，我们应对了矩阵乘法的计算复杂性和秩瓶颈问题。我们进行了广泛实验，从多种评估指标展示了我们方法的有效性。 

---
# PrunePEFT: Iterative Hybrid Pruning for Parameter-Efficient Fine-tuning of LLMs 

**Title (ZH)**: PrunePEFT: 迭代混合修剪以实现LLMs的参数高效微调 

**Authors**: Tongzhou Yu, Zhuhao Zhang, Guanghui Zhu, Shen Jiang, Meikang Qiu, Yihua Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07587)  

**Abstract**: Parameter Efficient Fine-Tuning (PEFT) methods have emerged as effective and promising approaches for fine-tuning pre-trained language models. Compared with Full parameter Fine-Tuning (FFT), PEFT achieved comparable task performance with a substantial reduction of trainable parameters, which largely saved the training and storage costs. However, using the PEFT method requires considering a vast design space, such as the type of PEFT modules and their insertion layers. Inadequate configurations can lead to sub-optimal results. Conventional solutions such as architectural search techniques, while effective, tend to introduce substantial additional overhead. In this paper, we propose a novel approach, PrunePEFT, which formulates the PEFT strategy search as a pruning problem and introduces a hybrid pruning strategy that capitalizes on the sensitivity of pruning methods to different PEFT modules. This method extends traditional pruning techniques by iteratively removing redundant or conflicting PEFT modules, thereby optimizing the fine-tuned configuration. By efficiently identifying the most relevant modules, our approach significantly reduces the computational burden typically associated with architectural search processes, making it a more scalable and efficient solution for fine-tuning large pre-trained models. 

**Abstract (ZH)**: 参数高效微调（PEFT）方法已 emerge作为有效且有前途的预训练语言模型微调方法。与全参数微调（FFT）相比，PEFT 实现了可比拟的任务性能，同时大幅度减少了可训练参数的数量，大大节省了训练和存储成本。然而，使用 PEFT 方法需要考虑广泛的设计空间，如 PEFT 模块的类型及其插入层。不适当的配置可能导致亚优结果。尽管传统的架构搜索技术有效，但往往会引入大量额外开销。在本文中，我们提出了一种新颖的方法 PrunePEFT，将 PEFT 策略搜索形式化为剪枝问题，并引入了一种结合了不同 PEFT 模块对剪枝方法敏感性的混合剪枝策略。该方法通过迭代移除冗余或冲突的 PEFT 模块，扩展了传统的剪枝技术，从而优化了微调配置。通过高效地识别最相关的模块，我们的方法显著减少了通常与架构搜索过程相关的计算负担，使其成为大规模预训练模型微调的一种更具扩展性和高效性解决方案。 

---
# Beyond the Sentence: A Survey on Context-Aware Machine Translation with Large Language Models 

**Title (ZH)**: 超越句子：面向上下文的大语言模型机器翻译综述 

**Authors**: Ramakrishna Appicharla, Baban Gain, Santanu Pal, Asif Ekbal  

**Link**: [PDF](https://arxiv.org/pdf/2506.07583)  

**Abstract**: Despite the popularity of the large language models (LLMs), their application to machine translation is relatively underexplored, especially in context-aware settings. This work presents a literature review of context-aware translation with LLMs. The existing works utilise prompting and fine-tuning approaches, with few focusing on automatic post-editing and creating translation agents for context-aware machine translation. We observed that the commercial LLMs (such as ChatGPT and Tower LLM) achieved better results than the open-source LLMs (such as Llama and Bloom LLMs), and prompt-based approaches serve as good baselines to assess the quality of translations. Finally, we present some interesting future directions to explore. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）的应用广泛，但它们在上下文感知翻译中的应用相对未被充分探索。本文综述了使用LLMs进行上下文感知翻译的相关研究。现有的研究主要采用了提示和微调方法，很少关注自动后编辑和构建上下文感知机器翻译的翻译代理。我们观察到商用LLMs（如ChatGPT和Tower LLM）的性能优于开源LLMs（如Llama和Bloom LLM），提示基方法可以作为评估翻译质量的良好基准。最后，我们提出了若干有趣的研究方向。 

---
# LLM-driven Indoor Scene Layout Generation via Scaled Human-aligned Data Synthesis and Multi-Stage Preference Optimization 

**Title (ZH)**: 基于按比例缩放人类对齐数据合成和多阶段偏好优化的LLM驱动室内场景布局生成 

**Authors**: Yixuan Yang, Zhen Luo, Tongsheng Ding, Junru Lu, Mingqi Gao, Jinyu Yang, Victor Sanchez, Feng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.07570)  

**Abstract**: Automatic indoor layout generation has attracted increasing attention due to its potential in interior design, virtual environment construction, and embodied AI. Existing methods fall into two categories: prompt-driven approaches that leverage proprietary LLM services (e.g., GPT APIs) and learning-based methods trained on layout data upon diffusion-based models. Prompt-driven methods often suffer from spatial inconsistency and high computational costs, while learning-based methods are typically constrained by coarse relational graphs and limited datasets, restricting their generalization to diverse room categories. In this paper, we revisit LLM-based indoor layout generation and present 3D-SynthPlace, a large-scale dataset that combines synthetic layouts generated via a 'GPT synthesize, Human inspect' pipeline, upgraded from the 3D-Front dataset. 3D-SynthPlace contains nearly 17,000 scenes, covering four common room types -- bedroom, living room, kitchen, and bathroom -- enriched with diverse objects and high-level spatial annotations. We further introduce OptiScene, a strong open-source LLM optimized for indoor layout generation, fine-tuned based on our 3D-SynthPlace dataset through our two-stage training. For the warum-up stage I, we adopt supervised fine-tuning (SFT), which is taught to first generate high-level spatial descriptions then conditionally predict concrete object placements. For the reinforcing stage II, to better align the generated layouts with human design preferences, we apply multi-turn direct preference optimization (DPO), which significantly improving layout quality and generation success rates. Extensive experiments demonstrate that OptiScene outperforms traditional prompt-driven and learning-based baselines. Moreover, OptiScene shows promising potential in interactive tasks such as scene editing and robot navigation. 

**Abstract (ZH)**: 自动室内布局生成由于其在室内设计、虚拟环境构建和具身AI等方面的应用潜力而引起了越来越多的关注。现有的方法可以分为两类：提示驱动的方法利用专有LLM服务（如GPT API）以及基于扩散模型训练的布局数据的学习方法。提示驱动的方法通常存在空间不一致性和高计算成本的问题，而学习方法则通常受到粗粒度关系图和有限数据集的限制，限制了其对多样化房间类别的泛化能力。在本文中，我们重新审视基于LLM的室内布局生成，并介绍了3D-SynthPlace，这是一个大规模数据集，结合了通过“GPT合成、人工检查”流水线生成的合成布局，源自3D-Front数据集的升级版。3D-SynthPlace包含近17,000个场景，涵盖四种常见房间类型——卧室、客厅、厨房和浴室，并配备了多种物体和高层次的空间注释。我们进一步介绍了OptiScene，这是一个面向室内布局生成的强大开源LLM，基于我们3D-SynthPlace数据集进行了两阶段微调。在准备阶段I中，我们采用了监督微调（SFT），使其首先生成高层次的空间描述，然后条件性地预测具体的物体排列。在强化阶段II中，为了更好地使生成的布局与人类的设计偏好对齐，我们应用了多轮直接偏好优化（DPO），显著提高了布局质量并提升了生成成功率。广泛实验表明，OptiScene在传统提示驱动和学习方法基线下表现出色。此外，OptiScene在场景编辑和机器人导航等交互任务中展现出良好的应用前景。 

---
# SELT: Self-Evaluation Tree Search for LLMs with Task Decomposition 

**Title (ZH)**: SELF-Evaluation Tree Search for LLMs with Task Decomposition 

**Authors**: Mengsong Wu, Di Zhang, Yuqiang Li, Dongzhan Zhou, Wenliang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.07557)  

**Abstract**: While Large Language Models (LLMs) have achieved remarkable success in a wide range of applications, their performance often degrades in complex reasoning tasks. In this work, we introduce SELT (Self-Evaluation LLM Tree Search), a novel framework that leverages a modified Monte Carlo Tree Search (MCTS) to enhance LLM reasoning without relying on external reward models. By redefining the Upper Confidence Bound scoring to align with intrinsic self-evaluation capabilities of LLMs and decomposing the inference process into atomic subtasks augmented with semantic clustering at each node, SELT effectively balances exploration and exploitation, reduces redundant reasoning paths, and mitigates hallucination. We validate our approach on challenging benchmarks, including the knowledge-based MMLU and the Tool Learning dataset Seal-Tools, where SELT achieves significant improvements in answer accuracy and reasoning robustness compared to baseline methods. Notably, our framework operates without task-specific fine-tuning, demonstrating strong generalizability across diverse reasoning tasks. Relevant results and code are available at this https URL . 

**Abstract (ZH)**: 虽然大型语言模型（LLMs）在广泛的应用中取得了显著的成功，但在复杂的推理任务中其性能往往会出现下降。在这项工作中，我们引入了SELT（Self-Evaluation LLM Tree Search）框架，该框架利用修改后的蒙特卡洛树搜索（MCTS）来增强LLM的推理能力，而无需依赖外部奖励模型。通过重新定义上层置信界评分以与LLM的内在自我评估能力对齐，并将推理过程分解为带有所谓语义聚类的原子子任务，SELT有效地平衡了探索与利用，减少了冗余的推理路径，并减轻了幻觉现象。我们在包括基于知识的MMLU和Tool Learning数据集Seal-Tools在内的具有挑战性的基准测试上验证了该方法，结果表明SELT在答案准确性和推理 robustness方面相较于基线方法实现了显著改进。特别地，我们的框架无需针对特定任务进行微调，显示出在各种推理任务中的强泛化能力。更多相关结果和代码请访问此链接：this https URL 

---
# ChemAgent: Enhancing LLMs for Chemistry and Materials Science through Tree-Search Based Tool Learning 

**Title (ZH)**: ChemAgent: 通过基于树搜索的工具学习提升化学和材料科学中的LLM性能 

**Authors**: Mengsong Wu, YaFei Wang, Yidong Ming, Yuqi An, Yuwei Wan, Wenliang Chen, Binbin Lin, Yuqiang Li, Tong Xie, Dongzhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.07551)  

**Abstract**: Large language models (LLMs) have recently demonstrated promising capabilities in chemistry tasks while still facing challenges due to outdated pretraining knowledge and the difficulty of incorporating specialized chemical expertise. To address these issues, we propose an LLM-based agent that synergistically integrates 137 external chemical tools created ranging from basic information retrieval to complex reaction predictions, and a dataset curation pipeline to generate the dataset ChemToolBench that facilitates both effective tool selection and precise parameter filling during fine-tuning and evaluation. We introduce a Hierarchical Evolutionary Monte Carlo Tree Search (HE-MCTS) framework, enabling independent optimization of tool planning and execution. By leveraging self-generated data, our approach supports step-level fine-tuning (FT) of the policy model and training task-adaptive PRM and ORM that surpass GPT-4o. Experimental evaluations demonstrate that our approach significantly improves performance in Chemistry QA and discovery tasks, offering a robust solution to integrate specialized tools with LLMs for advanced chemical applications. All datasets and code are available at this https URL . 

**Abstract (ZH)**: 大型语言模型（LLMs）在化学任务中展现了令人鼓舞的能力，但仍面临由于过时的预训练知识和将专业化学知识集成的难度而带来的挑战。为了解决这些问题，我们提出了一种基于LLM的代理，该代理协同整合了137个外部化学工具，涵盖从基本信息检索到复杂反应预测的各个方面，并建立了一个数据集整理管道以生成Facilitating Tool Selection and Precise Parameter Filling during Fine-Tuning and Evaluation的ChemToolBench数据集。我们引入了一种分层演化蒙特卡罗树搜索（HE-MCTS）框架，使工具规划和执行的优化得以独立进行。通过利用自动生成的数据，我们的方法支持策略模型的步骤级微调（FT）和训练自适应任务PRM和ORM，超越了GPT-4o。实验评估显示，我们的方法显著提高了化学问答和发现任务中的性能，提供了一种集成专业工具与LLM的稳健方法，适用于高级化学应用。所有数据集和代码均可在此网址访问。 

---
# IntenTest: Stress Testing for Intent Integrity in API-Calling LLM Agents 

**Title (ZH)**: IntenTest: API-调用LLM代理的意图完整性压力测试 

**Authors**: Shiwei Feng, Xiangzhe Xu, Xuan Chen, Kaiyuan Zhang, Syed Yusuf Ahmed, Zian Su, Mingwei Zheng, Xiangyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07524)  

**Abstract**: LLM agents are increasingly deployed to automate real-world tasks by invoking APIs through natural language instructions. While powerful, they often suffer from misinterpretation of user intent, leading to the agent's actions that diverge from the user's intended goal, especially as external toolkits evolve. Traditional software testing assumes structured inputs and thus falls short in handling the ambiguity of natural language. We introduce IntenTest, an API-centric stress testing framework that systematically uncovers intent integrity violations in LLM agents. Unlike prior work focused on fixed benchmarks or adversarial inputs, IntenTest generates realistic tasks based on toolkits' documentation and applies targeted mutations to expose subtle agent errors while preserving user intent. To guide testing, we propose semantic partitioning, which organizes natural language tasks into meaningful categories based on toolkit API parameters and their equivalence classes. Within each partition, seed tasks are mutated and ranked by a lightweight predictor that estimates the likelihood of triggering agent errors. To enhance efficiency, IntenTest maintains a datatype-aware strategy memory that retrieves and adapts effective mutation patterns from past cases. Experiments on 80 toolkit APIs demonstrate that IntenTest effectively uncovers intent integrity violations, significantly outperforming baselines in both error-exposing rate and query efficiency. Moreover, IntenTest generalizes well to stronger target models using smaller LLMs for test generation, and adapts to evolving APIs across domains. 

**Abstract (ZH)**: 基于API的意图完整性检测框架IntenTest 

---
# LeVo: High-Quality Song Generation with Multi-Preference Alignment 

**Title (ZH)**: LeVo：多偏好对齐的高质量歌曲生成 

**Authors**: Shun Lei, Yaoxun Xu, Zhiwei Lin, Huaicheng Zhang, Wei Tan, Hangting Chen, Jianwei Yu, Yixuan Zhang, Chenyu Yang, Haina Zhu, Shuai Wang, Zhiyong Wu, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07520)  

**Abstract**: Recent advances in large language models (LLMs) and audio language models have significantly improved music generation, particularly in lyrics-to-song generation. However, existing approaches still struggle with the complex composition of songs and the scarcity of high-quality data, leading to limitations in sound quality, musicality, instruction following, and vocal-instrument harmony. To address these challenges, we introduce LeVo, an LM-based framework consisting of LeLM and a music codec. LeLM is capable of parallelly modeling two types of tokens: mixed tokens, which represent the combined audio of vocals and accompaniment to achieve vocal-instrument harmony, and dual-track tokens, which separately encode vocals and accompaniment for high-quality song generation. It employs two decoder-only transformers and a modular extension training strategy to prevent interference between different token types. To further enhance musicality and instruction following, we introduce a multi-preference alignment method based on Direct Preference Optimization (DPO). This method handles diverse human preferences through a semi-automatic data construction process and DPO post-training. Experimental results demonstrate that LeVo consistently outperforms existing methods on both objective and subjective metrics. Ablation studies further justify the effectiveness of our designs. Audio examples are available at this https URL. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）和音频语言模型在音乐生成，尤其是歌词到歌曲生成方面取得了显著进展。然而，现有方法仍然在处理歌曲的复杂组成和高质量数据的稀缺性方面存在局限性，导致在音质、音乐性、指令跟随和人声乐器和声方面存在限制。为了解决这些挑战，我们引入了LeVo，一个基于LM的框架，包含LeLM和一个音乐编解码器。LeLM能够并行建模两种类型的标记：混合标记，用于表示人声和伴奏以实现人声乐器和声；以及双重轨道标记，分别编码人声和伴奏以实现高质量的歌曲生成。它采用了两个仅解码器变压器和模块化扩展训练策略，以防止不同标记类型的干扰。为了进一步提高音乐性和指令跟随，我们基于直接偏好优化（DPO）引入了一种多偏好对齐方法。该方法通过半自动数据构建过程和DPO后训练处理不同的人类偏好。实验结果表明，LeVo在客观和主观指标上均优于现有方法。消融研究进一步证明了我们设计的有效性。音频示例请参见此链接。 

---
# Graph-of-Causal Evolution: Challenging Chain-of-Model for Reasoning 

**Title (ZH)**: 因果演化图：挑战模型链推理 

**Authors**: Libo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07501)  

**Abstract**: In view of the problem that each subchain in the chain-of-model (CoM) relies only on the information of the previous subchain and may lose long-range dependencies due to the causal mask blocking the global context flow between multi-level subchains, this work proposes a graph of causal evolution (GoCE). Its core principle is to map the implicit token representation into a differentiable and sparse causal adjacency matrix, then permeate causal constraints through each layer of calculation using causal-masked attention and causal-MoE. By combining intervention consistency loss test and self-evolution gate, the dynamic balance between causal structure learning and adaptive updating of transformer architecture is realized. The researcher built experimental environments in sandboxes built with Claude Sonnet 4, o4-mini-high, and DeepSeek R1 respectively with the transformer variant architecture introduced in GoCE. It is evaluated on publicly available datasets including CLUTRR, CLADDER, EX-FEVER, and CausalQA and compared with the baseline LLMs. The finding proves that GoCE strengthens the transformer's ability to capture long-range causal dependencies, while the ability to self-evolve is improved. It not only surpasses the design of CoM in terms of design principles, but also provides experience for future research on causal learning and continuous adaptive improvement. 

**Abstract (ZH)**: 基于因果演化图的链式模型因果长依赖增强研究 

---
# CCI4.0: A Bilingual Pretraining Dataset for Enhancing Reasoning in Large Language Models 

**Title (ZH)**: CCI4.0: 一种用于增强大型语言模型推理能力的双语预训练数据集 

**Authors**: Guang Liu, Liangdong Wang, Jijie Li, Yang Yu, Yao Xu, Jiabei Chen, Yu Bai, Feng Liao, Yonghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.07463)  

**Abstract**: We introduce CCI4.0, a large-scale bilingual pre-training dataset engineered for superior data quality and diverse human-like reasoning trajectory. CCI4.0 occupies roughly $35$ TB of disk space and comprises two sub-datasets: CCI4.0-M2-Base and CCI4.0-M2-CoT. CCI4.0-M2-Base combines a $5.2$ TB carefully curated Chinese web corpus, a $22.5$ TB English subset from Nemotron-CC, and diverse sources from math, wiki, arxiv, and code. Although these data are mostly sourced from well-processed datasets, the quality standards of various domains are dynamic and require extensive expert experience and labor to process. So, we propose a novel pipeline justifying data quality mainly based on models through two-stage deduplication, multiclassifier quality scoring, and domain-aware fluency filtering. We extract $4.5$ billion pieces of CoT(Chain-of-Thought) templates, named CCI4.0-M2-CoT. Differing from the distillation of CoT from larger models, our proposed staged CoT extraction exemplifies diverse reasoning patterns and significantly decreases the possibility of hallucination. Empirical evaluations demonstrate that LLMs pre-trained in CCI4.0 benefit from cleaner, more reliable training signals, yielding consistent improvements in downstream tasks, especially in math and code reflection tasks. Our results underscore the critical role of rigorous data curation and human thinking templates in advancing LLM performance, shedding some light on automatically processing pretraining corpora. 

**Abstract (ZH)**: CCI4.0：一个大规模双语预训练数据集，旨在提供高质量数据和多样的类人类推理轨迹 

---
# KScope: A Framework for Characterizing the Knowledge Status of Language Models 

**Title (ZH)**: KScope: 一种语言模型知识状态表征框架 

**Authors**: Yuxin Xiao, Shan Chen, Jack Gallifant, Danielle Bitterman, Thomas Hartvigsen, Marzyeh Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07458)  

**Abstract**: Characterizing a large language model's (LLM's) knowledge of a given question is challenging. As a result, prior work has primarily examined LLM behavior under knowledge conflicts, where the model's internal parametric memory contradicts information in the external context. However, this does not fully reflect how well the model knows the answer to the question. In this paper, we first introduce a taxonomy of five knowledge statuses based on the consistency and correctness of LLM knowledge modes. We then propose KScope, a hierarchical framework of statistical tests that progressively refines hypotheses about knowledge modes and characterizes LLM knowledge into one of these five statuses. We apply KScope to nine LLMs across four datasets and systematically establish: (1) Supporting context narrows knowledge gaps across models. (2) Context features related to difficulty, relevance, and familiarity drive successful knowledge updates. (3) LLMs exhibit similar feature preferences when partially correct or conflicted, but diverge sharply when consistently wrong. (4) Context summarization constrained by our feature analysis, together with enhanced credibility, further improves update effectiveness and generalizes across LLMs. 

**Abstract (ZH)**: 探索大语言模型知识模式的五个知识状态，并提出KScope框架：一种逐步精炼知识模式假设并将其分类为五种状态的统计测试层次框架 

---
# When Style Breaks Safety: Defending Language Models Against Superficial Style Alignment 

**Title (ZH)**: 当风格破坏安全：防御语言模型的表面上的风格对齐 

**Authors**: Yuxin Xiao, Sana Tonekaboni, Walter Gerych, Vinith Suriyakumar, Marzyeh Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07452)  

**Abstract**: Large language models (LLMs) can be prompted with specific styles (e.g., formatting responses as lists), including in jailbreak queries. Although these style patterns are semantically unrelated to the malicious intents behind jailbreak queries, their safety impact remains unclear. In this work, we seek to understand whether style patterns compromise LLM safety, how superficial style alignment increases model vulnerability, and how best to mitigate these risks during alignment. We evaluate 32 LLMs across seven jailbreak benchmarks, and find that malicious queries with style patterns inflate the attack success rate (ASR) for nearly all models. Notably, ASR inflation correlates with both the length of style patterns and the relative attention an LLM exhibits on them. We then investigate superficial style alignment, and find that fine-tuning with specific styles makes LLMs more vulnerable to jailbreaks of those same styles. Finally, we propose SafeStyle, a defense strategy that incorporates a small amount of safety training data augmented to match the distribution of style patterns in the fine-tuning data. Across three LLMs and five fine-tuning style settings, SafeStyle consistently outperforms baselines in maintaining LLM safety. 

**Abstract (ZH)**: 大型语言模型的安全性受特定样式的影响研究：风格模式是否损害LLM安全？表面风格对齐如何增加模型漏洞以及如何缓解这些风险 

---
# LlamaRec-LKG-RAG: A Single-Pass, Learnable Knowledge Graph-RAG Framework for LLM-Based Ranking 

**Title (ZH)**: LlamaRec-LKG-RAG：一种基于单-pass、可学习的知识图谱-RAG框架的LLM排序模型 

**Authors**: Vahid Azizi, Fatemeh Koochaki  

**Link**: [PDF](https://arxiv.org/pdf/2506.07449)  

**Abstract**: Recent advances in Large Language Models (LLMs) have driven their adoption in recommender systems through Retrieval-Augmented Generation (RAG) frameworks. However, existing RAG approaches predominantly rely on flat, similarity-based retrieval that fails to leverage the rich relational structure inherent in user-item interactions. We introduce LlamaRec-LKG-RAG, a novel single-pass, end-to-end trainable framework that integrates personalized knowledge graph context into LLM-based recommendation ranking. Our approach extends the LlamaRec architecture by incorporating a lightweight user preference module that dynamically identifies salient relation paths within a heterogeneous knowledge graph constructed from user behavior and item metadata. These personalized subgraphs are seamlessly integrated into prompts for a fine-tuned Llama-2 model, enabling efficient and interpretable recommendations through a unified inference step. Comprehensive experiments on ML-100K and Amazon Beauty datasets demonstrate consistent and significant improvements over LlamaRec across key ranking metrics (MRR, NDCG, Recall). LlamaRec-LKG-RAG demonstrates the critical value of structured reasoning in LLM-based recommendations and establishes a foundation for scalable, knowledge-aware personalization in next-generation recommender systems. Code is available at~\href{this https URL}{repository}. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）通过检索增强生成（RAG）框架推动了其在推荐系统中的应用。然而，现有的RAG方法主要依赖于扁平的、基于相似性的检索，未能充分利用用户项交互中存在的丰富关系结构。我们引入了LlamaRec-LKG-RAG，这是一种新颖的一站式、端到端可训练框架，将个性化知识图景上下文集成到基于LLM的推荐排名中。我们的方法通过结合一个轻量级用户偏好模块扩展了LlamaRec架构，该模块能够在由用户行为和项元数据构建的异构知识图中动态识别显着的关系路径。这些个性化的子图无缝集成到微调的Llama-2模型的提示中，通过统一的推理步骤实现高效且可解释的推荐。在ML-100K和Amazon Beauty数据集上的全面实验表明，LlamaRec-LKG-RAG在关键排名指标（MRR、NDCG、召回率）上相对于LlamaRec实现了持续且显著的改进。LlamaRec-LKG-RAG展示了结构化推理在基于LLM的推荐中的关键价值，并为下一代推荐系统中可扩展、知识感知个性化奠定了基础。代码可在\href{this https URL}{仓库}获取。 

---
# Extending Epistemic Uncertainty Beyond Parameters Would Assist in Designing Reliable LLMs 

**Title (ZH)**: 超越参数扩展认识不确定性将有助于设计可靠的大型语言模型 

**Authors**: T. Duy Nguyen-Hien, Desi R. Ivanova, Yee Whye Teh, Wee Sun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.07448)  

**Abstract**: Although large language models (LLMs) are highly interactive and extendable, current approaches to ensure reliability in deployments remain mostly limited to rejecting outputs with high uncertainty in order to avoid misinformation. This conservative strategy reflects the current lack of tools to systematically distinguish and respond to different sources of uncertainty. In this paper, we advocate for the adoption of Bayesian Modeling of Experiments -- a framework that provides a coherent foundation to reason about uncertainty and clarify the reducibility of uncertainty -- for managing and proactively addressing uncertainty that arises in LLM deployments. This framework enables LLMs and their users to take contextually appropriate steps, such as requesting clarification, retrieving external information, or refining inputs. By supporting active resolution rather than passive avoidance, it opens the door to more reliable, transparent, and broadly applicable LLM systems, particularly in high-stakes, real-world settings. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）高度互动且可扩展，当前确保部署可靠性的方法主要限于拒绝具有高不确定性输出的做法，以避免误导信息。这一保守策略反映了当前缺乏系统区分和应对不同来源不确定性工具的现状。在本文中，我们倡导采用贝叶斯实验建模——这一框架提供了一种一致的基础来推理不确定性并明确不确定性可减少的途径——以管理和主动应对在LLM部署中出现的不确定性。该框架使LLMs及其用户能够采取语境适当的操作，如请求澄清、检索外部信息或细化输入。通过支持积极解决而非被动避免，它为开发更可靠、透明且适用范围更广的LLM系统打开了大门，特别是在高风险的实际应用场景中。 

---
# Prompt to Protection: A Comparative Study of Multimodal LLMs in Construction Hazard Recognition 

**Title (ZH)**: 提示到保护：多模态LLM在建筑危险识别中的比较研究 

**Authors**: Nishi Chaudhary, S M Jamil Uddin, Sathvik Sharath Chandra, Anto Ovid, Alex Albert  

**Link**: [PDF](https://arxiv.org/pdf/2506.07436)  

**Abstract**: The recent emergence of multimodal large language models (LLMs) has introduced new opportunities for improving visual hazard recognition on construction sites. Unlike traditional computer vision models that rely on domain-specific training and extensive datasets, modern LLMs can interpret and describe complex visual scenes using simple natural language prompts. However, despite growing interest in their applications, there has been limited investigation into how different LLMs perform in safety-critical visual tasks within the construction domain. To address this gap, this study conducts a comparative evaluation of five state-of-the-art LLMs: Claude-3 Opus, GPT-4.5, GPT-4o, GPT-o3, and Gemini 2.0 Pro, to assess their ability to identify potential hazards from real-world construction images. Each model was tested under three prompting strategies: zero-shot, few-shot, and chain-of-thought (CoT). Zero-shot prompting involved minimal instruction, few-shot incorporated basic safety context and a hazard source mnemonic, and CoT provided step-by-step reasoning examples to scaffold model thinking. Quantitative analysis was performed using precision, recall, and F1-score metrics across all conditions. Results reveal that prompting strategy significantly influenced performance, with CoT prompting consistently producing higher accuracy across models. Additionally, LLM performance varied under different conditions, with GPT-4.5 and GPT-o3 outperforming others in most settings. The findings also demonstrate the critical role of prompt design in enhancing the accuracy and consistency of multimodal LLMs for construction safety applications. This study offers actionable insights into the integration of prompt engineering and LLMs for practical hazard recognition, contributing to the development of more reliable AI-assisted safety systems. 

**Abstract (ZH)**: recent emergence of 多模态大规模语言模型 (LLMs) 为施工现场视觉隐患识别带来了新的机会。与依赖领域特定训练和大量数据的传统计算机视觉模型不同，现代LLMs可以通过简单的自然语言提示来解释和描述复杂的视觉场景。然而，尽管在这些应用方面兴趣日益增长，但对于安全关键的视觉任务，不同LLMs在建筑领域中的表现仍缺乏系统研究。为填补这一空白，本研究对比评估了五种最先进的LLMs：Claude-3 Opus、GPT-4.5、GPT-4o、GPT-o3和Gemini 2.0 Pro，以评估其识别施工现场真实图像潜在隐患的能力。每个模型在零样本、少数样本和步步推理（CoT）三种提示策略下进行了测试。零样本提示涉及最少的指令，少数样本结合了基础安全语境和隐患源记忆，步步推理提供了逐步推理示例以辅助模型思考。通过准确率、召回率和F1分数进行了定量分析。结果表明提示策略显著影响了性能，步步推理提示在所有模型中始终产生更高的准确性。此外，在不同条件下，LLM的表现也有所不同，GPT-4.5和GPT-o3在大多数设置中表现最佳。研究还展示了提示设计对提升多模态LLMs在建筑安全应用中准确性和一致性的关键作用。本研究提供了关于如何通过提示工程集成LLMs进行实际隐患识别的实用见解，为开发更可靠的AI辅助安全系统做出了贡献。 

---
# Well Begun is Half Done: Low-resource Preference Alignment by Weak-to-Strong Decoding 

**Title (ZH)**: 良好的开始是一场胜仗的一半：低资源偏好对齐通过弱到强解码实现 

**Authors**: Feifan Song, Shaohang Wei, Wen Luo, Yuxuan Fan, Tianyu Liu, Guoyin Wang, Houfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07434)  

**Abstract**: Large Language Models (LLMs) require alignment with human preferences to avoid generating offensive, false, or meaningless content. Recently, low-resource methods for LLM alignment have been popular, while still facing challenges in obtaining both high-quality and aligned content. Motivated by the observation that the difficulty of generating aligned responses is concentrated at the beginning of decoding, we propose a novel framework, Weak-to-Strong Decoding (WSD), to enhance the alignment ability of base models by the guidance of a small aligned model. The small model first drafts well-aligned beginnings, followed by the large base model to continue the rest, controlled by a well-designed auto-switch mechanism. We also collect a new dataset, GenerAlign, to fine-tune a small-sized Pilot-3B as the draft model, which effectively enhances different base models under the WSD framework to outperform all baseline methods, while avoiding degradation on downstream tasks, termed as the alignment tax. Extensive experiments are further conducted to examine the impact of different settings and time efficiency, as well as analyses on the intrinsic mechanisms of WSD in depth. 

**Abstract (ZH)**: Large Language Models (LLMs) 需要与人类偏好对齐以避免生成冒犯性、虚假或无意义的内容。近年来，低资源的LLM对齐方法很受欢迎，但仍面临获得高质量和对齐内容的挑战。受生成对齐响应难度集中在解码早期这一观察的启发，我们提出了一种新颖的框架，即弱到强解码（WSD），通过一个小的对齐模型的指导来增强基模型的对齐能力。小型模型首先草拟出良好的对齐开头，随后由大型基模型继续生成其余部分，由精心设计的自动切换机制控制。我们还收集了一个新的数据集GenerAlign，用于微调一个小规模的Pilot-3B作为草拟模型，该模型在WSD框架下有效地增强了不同的基模型，优于所有基线方法，同时避免了下游任务性能下降的问题，即对齐税。进一步进行了广泛的实验以检查不同设置的影响和时间效率，并深入分析了WSD内在机制。 

---
# Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models 

**Title (ZH)**: 插件 fine-tuning：缩小小语言模型与大语言模型之间的差距 

**Authors**: Kyeonghyun Kim, Jinhee Jang, Juhwan Choi, Yoonji Lee, Kyohoon Jin, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.07424)  

**Abstract**: Large language models (LLMs) are renowned for their extensive linguistic knowledge and strong generalization capabilities, but their high computational demands make them unsuitable for resource-constrained environments. In contrast, small language models (SLMs) are computationally efficient but often lack the broad generalization capacity of LLMs. To bridge this gap, we propose PiFi, a novel framework that combines the strengths of both LLMs and SLMs to achieve high performance while maintaining efficiency. PiFi integrates a single frozen layer from an LLM into a SLM and fine-tunes the combined model for specific tasks, boosting performance without a significant increase in computational cost. We show that PiFi delivers consistent performance improvements across a range of natural language processing tasks, including both natural language understanding and generation. Moreover, our findings demonstrate PiFi's ability to effectively leverage LLM knowledge, enhancing generalization to unseen domains and facilitating the transfer of linguistic abilities. 

**Abstract (ZH)**: PiFi：结合大语言模型和小语言模型的优势以实现高效性能 

---
# Anomaly Detection and Early Warning Mechanism for Intelligent Monitoring Systems in Multi-Cloud Environments Based on LLM 

**Title (ZH)**: 基于大语言模型的多云环境下智能监控系统异常检测与早期预警机制 

**Authors**: Yihong Jin, Ze Yang, Juntian Liu, Xinhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07407)  

**Abstract**: With the rapid development of multi-cloud environments, it is increasingly important to ensure the security and reliability of intelligent monitoring systems. In this paper, we propose an anomaly detection and early warning mechanism for intelligent monitoring system in multi-cloud environment based on Large-Scale Language Model (LLM). On the basis of the existing monitoring framework, the proposed model innovatively introduces a multi-level feature extraction method, which combines the natural language processing ability of LLM with traditional machine learning methods to enhance the accuracy of anomaly detection and improve the real-time response efficiency. By introducing the contextual understanding capabilities of LLMs, the model dynamically adapts to different cloud service providers and environments, so as to more effectively detect abnormal patterns and predict potential failures. Experimental results show that the proposed model is significantly better than the traditional anomaly detection system in terms of detection accuracy and latency, and significantly improves the resilience and active management ability of cloud infrastructure. 

**Abstract (ZH)**: 基于大型语言模型的多云环境智能监控系统异常检测与早期预警机制 

---
# InverseScope: Scalable Activation Inversion for Interpreting Large Language Models 

**Title (ZH)**: 逆向范围：大规模语言模型解释的可扩展激活反向检索 

**Authors**: Yifan Luo, Zhennan Zhou, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.07406)  

**Abstract**: Understanding the internal representations of large language models (LLMs) is a central challenge in interpretability research. Existing feature interpretability methods often rely on strong assumptions about the structure of representations that may not hold in practice. In this work, we introduce InverseScope, an assumption-light and scalable framework for interpreting neural activations via input inversion. Given a target activation, we define a distribution over inputs that generate similar activations and analyze this distribution to infer the encoded features. To address the inefficiency of sampling in high-dimensional spaces, we propose a novel conditional generation architecture that significantly improves sample efficiency compared to previous methods. We further introduce a quantitative evaluation protocol that tests interpretability hypotheses using feature consistency rate computed over the sampled inputs. InverseScope scales inversion-based interpretability methods to larger models and practical tasks, enabling systematic and quantitative analysis of internal representations in real-world LLMs. 

**Abstract (ZH)**: 理解大型语言模型（LLMs）的内部表示是可解释性研究中的一个核心挑战。现有特征可解释性方法往往依赖于在实践中可能不成立的表示结构假设。在本文中，我们引入了InverseScope，这是一种轻假设且可扩展的框架，通过输入 inversion 解释神经激活。给定一个目标激活，我们定义一个生成类似激活的输入分布，并分析该分布以推断编码的特征。为了解决高维空间采样的低效率问题，我们提出了一种新颖的条件生成架构，相比以前的方法显著提高了采样效率。此外，我们还引入了一种定量评估协议，用于通过计算采样输入的特征一致性率来测试可解释性假设。InverseScope 将基于逆向的可解释性方法扩展到更大的模型和实际任务，使得对现实世界 LLMs 的内部表示进行系统性和定量分析成为可能。 

---
# MedChat: A Multi-Agent Framework for Multimodal Diagnosis with Large Language Models 

**Title (ZH)**: MedChat: 多模态诊断的多Agent框架基于大型语言模型 

**Authors**: Philip Liu, Sparsh Bansal, Jimmy Dinh, Aditya Pawar, Ramani Satishkumar, Shail Desai, Neeraj Gupta, Xin Wang, Shu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07400)  

**Abstract**: The integration of deep learning-based glaucoma detection with large language models (LLMs) presents an automated strategy to mitigate ophthalmologist shortages and improve clinical reporting efficiency. However, applying general LLMs to medical imaging remains challenging due to hallucinations, limited interpretability, and insufficient domain-specific medical knowledge, which can potentially reduce clinical accuracy. Although recent approaches combining imaging models with LLM reasoning have improved reporting, they typically rely on a single generalist agent, restricting their capacity to emulate the diverse and complex reasoning found in multidisciplinary medical teams. To address these limitations, we propose MedChat, a multi-agent diagnostic framework and platform that combines specialized vision models with multiple role-specific LLM agents, all coordinated by a director agent. This design enhances reliability, reduces hallucination risk, and enables interactive diagnostic reporting through an interface tailored for clinical review and educational use. Code available at this https URL. 

**Abstract (ZH)**: 基于深度学习的青光眼检测与大型语言模型的集成：一种自动化策略，用于缓解眼科医生短缺并提高临床报告效率，但将通用大型语言模型应用于医学图像仍面临挑战，因为存在幻觉、解释性有限以及缺乏特定领域的医学知识，这可能会降低临床准确性。虽然最近结合图像模型和LLM推理的方法提高了报告质量，但它们通常依赖单一通用代理，限制了其模仿多学科医疗团队复杂推理的能力。为解决这些限制，我们提出MedChat，一种由专门视觉模型与多个角色特定的大规模语言模型代理组成的多代理诊断框架和平台，所有代理均由导演代理协调。该设计提高了可靠性，降低了幻觉风险，并通过一个适用于临床审查和教育使用的界面实现交互式诊断报告。代码见此URL。 

---
# Shapley-Coop: Credit Assignment for Emergent Cooperation in Self-Interested LLM Agents 

**Title (ZH)**: Shapley-Coop：自我利益LLM代理中 emergent合作的归因分配 

**Authors**: Yun Hua, Haosheng Chen, Shiqin Wang, Wenhao Li, Xiangfeng Wang, Jun Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.07388)  

**Abstract**: Large Language Models (LLMs) show strong collaborative performance in multi-agent systems with predefined roles and workflows. However, in open-ended environments lacking coordination rules, agents tend to act in self-interested ways. The central challenge in achieving coordination lies in credit assignment -- fairly evaluating each agent's contribution and designing pricing mechanisms that align their heterogeneous goals. This problem is critical as LLMs increasingly participate in complex human-AI collaborations, where fair compensation and accountability rely on effective pricing mechanisms. Inspired by how human societies address similar coordination challenges (e.g., through temporary collaborations such as employment or subcontracting), we propose a cooperative workflow, Shapley-Coop. Shapley-Coop integrates Shapley Chain-of-Thought -- leveraging marginal contributions as a principled basis for pricing -- with structured negotiation protocols for effective price matching, enabling LLM agents to coordinate through rational task-time pricing and post-task reward redistribution. This approach aligns agent incentives, fosters cooperation, and maintains autonomy. We evaluate Shapley-Coop across two multi-agent games and a software engineering simulation, demonstrating that it consistently enhances LLM agent collaboration and facilitates equitable credit assignment. These results highlight the effectiveness of Shapley-Coop's pricing mechanisms in accurately reflecting individual contributions during task execution. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在具备预定义角色和流程的工作流多智能体系统中展现出强大的协作性能。但在缺乏协调规则的开放环境中，智能体倾向于以自我为中心的方式行动。实现协调的核心挑战在于责任归属——公平评估每个智能体的贡献并设计能够统一其异质目标的定价机制。随着LLMs越来越多地参与到复杂的人机协作中，有效的定价机制对于公平补偿和问责至关重要。受人类社会解决类似协调挑战方式（如通过临时合作，如雇佣或分包）的启发，我们提出了一种协作工作流Shapley-Coop。Shapley-Coop将Shapley Chain-of-Thought（通过边际贡献作为定价原则的基础）与结构化的谈判协议结合，以实现有效的价格匹配，使LLM智能体通过有理的任务时间和事后奖励重分配来实现协作。该方法对齐了智能体动力，促进了合作，并保持了自主权。我们在两个多智能体游戏和一个软件工程模拟中评估了Shapley-Coop，结果表明它能持续增强LLM智能体的协作，并推动公平的责任归属。这些结果突显了Shapley-Coop定价机制在任务执行期间准确反映个体贡献的有效性。 

---
# Improving LLM Reasoning through Interpretable Role-Playing Steering 

**Title (ZH)**: 通过可解释的角色扮演引导提高LLM推理能力 

**Authors**: Anyi Wang, Dong Shu, Yifan Wang, Yunpu Ma, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2506.07335)  

**Abstract**: Role-playing has emerged as an effective technique for enhancing the reasoning capabilities of large language models (LLMs). However, existing methods primarily rely on prompt engineering, which often lacks stability and interpretability. In this paper, we introduce Sparse Autoencoder Role-Playing Steering (SRPS), a novel framework that identifies and manipulates internal model features associated with role-playing behavior. Our approach extracts latent representations from role-play prompts, selects the most relevant features based on activation patterns, and constructs a steering vector that can be injected into the model's residual stream with controllable intensity. Our method enables fine-grained control over role-specific behavior and offers insights into how role information influences internal model activations. Extensive experiments across various reasoning benchmarks and model sizes demonstrate consistent performance gains. Notably, in the zero-shot chain-of-thought (CoT) setting, the accuracy of Llama3.1-8B on CSQA improves from 31.86% to 39.80%, while Gemma2-9B on SVAMP increases from 37.50% to 45.10%. These results highlight the potential of SRPS to enhance reasoning ability in LLMs, providing better interpretability and stability compared to traditional prompt-based role-playing. 

**Abstract (ZH)**: Sparse Autoencoder Role-Playing Steering: Enhancing the Reasoning Capabilities of Large Language Models 

---
# JavelinGuard: Low-Cost Transformer Architectures for LLM Security 

**Title (ZH)**: JavelinGuard: 低成本变压器架构的LLM安全防护 

**Authors**: Yash Datta, Sharath Rajasekar  

**Link**: [PDF](https://arxiv.org/pdf/2506.07330)  

**Abstract**: We present JavelinGuard, a suite of low-cost, high-performance model architectures designed for detecting malicious intent in Large Language Model (LLM) interactions, optimized specifically for production deployment. Recent advances in transformer architectures, including compact BERT(Devlin et al. 2019) variants (e.g., ModernBERT (Warner et al. 2024)), allow us to build highly accurate classifiers with as few as approximately 400M parameters that achieve rapid inference speeds even on standard CPU hardware. We systematically explore five progressively sophisticated transformer-based architectures: Sharanga (baseline transformer classifier), Mahendra (enhanced attention-weighted pooling with deeper heads), Vaishnava and Ashwina (hybrid neural ensemble architectures), and Raudra (an advanced multi-task framework with specialized loss functions). Our models are rigorously benchmarked across nine diverse adversarial datasets, including popular sets like the NotInject series, BIPIA, Garak, ImprovedLLM, ToxicChat, WildGuard, and our newly introduced JavelinBench, specifically crafted to test generalization on challenging borderline and hard-negative cases. Additionally, we compare our architectures against leading open-source guardrail models as well as large decoder-only LLMs such as gpt-4o, demonstrating superior cost-performance trade-offs in terms of accuracy, and latency. Our findings reveal that while Raudra's multi-task design offers the most robust performance overall, each architecture presents unique trade-offs in speed, interpretability, and resource requirements, guiding practitioners in selecting the optimal balance of complexity and efficiency for real-world LLM security applications. 

**Abstract (ZH)**: JavelinGuard：一套专为生产部署优化的低成本高性能模型架构，用于检测大型语言模型交互中的恶意意图 

---
# Reward Model Interpretability via Optimal and Pessimal Tokens 

**Title (ZH)**: 通过最优和最劣标记提升奖励模型可解释性 

**Authors**: Brian Christian, Hannah Rose Kirk, Jessica A.F. Thompson, Christopher Summerfield, Tsvetomira Dumbalska  

**Link**: [PDF](https://arxiv.org/pdf/2506.07326)  

**Abstract**: Reward modeling has emerged as a crucial component in aligning large language models with human values. Significant attention has focused on using reward models as a means for fine-tuning generative models. However, the reward models themselves -- which directly encode human value judgments by turning prompt-response pairs into scalar rewards -- remain relatively understudied. We present a novel approach to reward model interpretability through exhaustive analysis of their responses across their entire vocabulary space. By examining how different reward models score every possible single-token response to value-laden prompts, we uncover several striking findings: (i) substantial heterogeneity between models trained on similar objectives, (ii) systematic asymmetries in how models encode high- vs low-scoring tokens, (iii) significant sensitivity to prompt framing that mirrors human cognitive biases, and (iv) overvaluation of more frequent tokens. We demonstrate these effects across ten recent open-source reward models of varying parameter counts and architectures. Our results challenge assumptions about the interchangeability of reward models, as well as their suitability as proxies of complex and context-dependent human values. We find that these models can encode concerning biases toward certain identity groups, which may emerge as unintended consequences of harmlessness training -- distortions that risk propagating through the downstream large language models now deployed to millions. 

**Abstract (ZH)**: 奖励模型已成为对齐大型语言模型与人类价值观的关键组件。尽管人们广泛关注使用奖励模型对生成模型进行微调，但直接将人类价值判断编码为提示-响应对的标量奖励的奖励模型本身仍相对研究不足。我们通过彻底分析其在整个词汇空间的响应，提出了一种新的奖励模型可解释性方法。通过对具有价值导向的提示的每一可能单词响应进行评分，我们发现了几个引人注目的发现：(i) 相似目标训练的模型之间存在显著异质性，(ii) 模型在编码高分词与低分词时存在系统不对称性，(iii) 对提示框架的显著敏感性，这种敏感性与人类认知偏差相呼应，以及(iv) 对更频繁出现的词赋予过高的价值。我们在十种不同参数量和架构的开源奖励模型中展示了这些效果。我们的研究结果挑战了奖励模型可互换性和作为复杂且上下文依赖的人类价值观代理的假设。我们发现这些模型可以编码对某些身份群体的令人担忧的偏见，这些偏见可能是无害训练的未预料后果，并可能通过现已部署给数百万用户的下游大型语言模型传播。 

---
# Towards Competent AI for Fundamental Analysis in Finance: A Benchmark Dataset and Evaluation 

**Title (ZH)**: 面向金融基本面分析的 competent AI：一个基准数据集及评估 

**Authors**: Zonghan Wu, Junlin Wang, Congyuan Zou, Chenhan Wang, Yilei Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.07315)  

**Abstract**: Generative AI, particularly large language models (LLMs), is beginning to transform the financial industry by automating tasks and helping to make sense of complex financial information. One especially promising use case is the automatic creation of fundamental analysis reports, which are essential for making informed investment decisions, evaluating credit risks, guiding corporate mergers, etc. While LLMs attempt to generate these reports from a single prompt, the risks of inaccuracy are significant. Poor analysis can lead to misguided investments, regulatory issues, and loss of trust. Existing financial benchmarks mainly evaluate how well LLMs answer financial questions but do not reflect performance in real-world tasks like generating financial analysis reports. In this paper, we propose FinAR-Bench, a solid benchmark dataset focusing on financial statement analysis, a core competence of fundamental analysis. To make the evaluation more precise and reliable, we break this task into three measurable steps: extracting key information, calculating financial indicators, and applying logical reasoning. This structured approach allows us to objectively assess how well LLMs perform each step of the process. Our findings offer a clear understanding of LLMs current strengths and limitations in fundamental analysis and provide a more practical way to benchmark their performance in real-world financial settings. 

**Abstract (ZH)**: 生成式人工智能，特别是大型语言模型（LLMs），正开始通过自动化任务和帮助理解和解释复杂的金融信息来转型金融行业。一个特别有前景的应用场景是自动化生成基础分析报告，这对于进行知情投资决策、评估信贷风险、指导企业合并等至关重要。尽管LLMs试图通过单一提示生成这些报告，但准确性风险仍然很大。不良分析可能导致误导性投资、监管问题和信任丧失。现有的金融基准主要评估LLMs回答金融问题的能力，但未能反映其在生成财务分析报告等实际任务中的表现。在本文中，我们提出FinAR-Bench，这是一个专注于财务报表分析的坚实基准数据集，这是基础分析的核心能力。为了使评估更加精确和可靠，我们将此任务分解为三个可测量的步骤：提取关键信息、计算财务指标和应用逻辑推理。这种结构化的方法使我们能够客观评估LLMs在每个步骤中的表现。我们的发现为理解当前LLMs在基础分析中的优势和限制提供了明确的见解，并提供了一种更实际的方法来衡量其在实际金融场景中的表现。 

---
# Paged Attention Meets FlexAttention: Unlocking Long-Context Efficiency in Deployed Inference 

**Title (ZH)**: 分页注意力邂逅FlexAttention：解锁部署推断中的长上下文效率 

**Authors**: Thomas Joshi, Herman Saini, Neil Dhillon, Antoni Viros i Martin, Kaoutar El Maghraoui  

**Link**: [PDF](https://arxiv.org/pdf/2506.07311)  

**Abstract**: Large Language Models (LLMs) encounter severe memory inefficiencies during long-context inference due to conventional handling of key-value (KV) caches. In this work, we introduce a novel integration of PagedAttention with PyTorch's FlexAttention, addressing internal fragmentation and inefficiencies associated with monolithic KV cache allocations. Implemented within IBM's Foundation Model Stack (FMS), our fused attention kernel efficiently gathers scattered KV data. Our benchmarks on an NVIDIA L4 GPU (24GB) demonstrate significantly reduced inference latency, growing only linearly (~2x) with sequence length from 128 to 2048 tokens when utilizing a global KV cache, compared to exponential latency increases without caching. While peak memory usage remains largely unchanged for single-step evaluations (dominated by model weights and activations), paged attention causes minimal incremental memory usage, observable only at sequence lengths exceeding 2048 tokens due to its power-of-two cache allocations. We open-source the full implementation and discuss its implications for future long-context model deployment. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在长上下文推理过程中由于常规的关键值（KV）缓存处理遇到严重的内存 inefficiencies。在本文中，我们引入了一种将 PagedAttention 与 PyTorch 的 FlexAttention 新颖集成的方法，以解决与大型 KV 缓存分配相关的内部碎片化和 inefficiencies。该融合的注意力内核在 IBM 的基础模型堆栈（FMS）中实现，有效收集分散的 KV 数据。我们在 NVIDIA L4 GPU（24GB）上的基准测试表明，使用全局 KV 缓存时，推理延迟显著降低，从 128 个到 2048 个标记的增长仅线性（~2 倍）增加，而没有缓存时延迟呈指数级增加。尽管单步评估的最大内存使用量保持基本不变（主要由模型权重和激活量决定），分页注意力引起的增量内存使用量几乎可以忽略不计，仅在序列长度超过 2048 个标记时才可观察到，这归因于其基于 2 的幂次的缓存分配。我们开源了完整实现，并探讨其对未来的长上下文模型部署的影响。 

---
# Pre-trained Large Language Models Learn Hidden Markov Models In-context 

**Title (ZH)**: 预训练大型语言模型学习上下文中的隐马尔可夫模型 

**Authors**: Yijia Dai, Zhaolin Gao, Yahya Satter, Sarah Dean, Jennifer J. Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.07298)  

**Abstract**: Hidden Markov Models (HMMs) are foundational tools for modeling sequential data with latent Markovian structure, yet fitting them to real-world data remains computationally challenging. In this work, we show that pre-trained large language models (LLMs) can effectively model data generated by HMMs via in-context learning (ICL)$\unicode{x2013}$their ability to infer patterns from examples within a prompt. On a diverse set of synthetic HMMs, LLMs achieve predictive accuracy approaching the theoretical optimum. We uncover novel scaling trends influenced by HMM properties, and offer theoretical conjectures for these empirical observations. We also provide practical guidelines for scientists on using ICL as a diagnostic tool for complex data. On real-world animal decision-making tasks, ICL achieves competitive performance with models designed by human experts. To our knowledge, this is the first demonstration that ICL can learn and predict HMM-generated sequences$\unicode{x2013}$an advance that deepens our understanding of in-context learning in LLMs and establishes its potential as a powerful tool for uncovering hidden structure in complex scientific data. 

**Abstract (ZH)**: 预训练大语言模型通过上下文学习有效建模隐马尔科夫模型生成的数据：探索新的扩展趋势并提供实际指导 

---
# HotelMatch-LLM: Joint Multi-Task Training of Small and Large Language Models for Efficient Multimodal Hotel Retrieval 

**Title (ZH)**: HotelMatch-LLM：小型和大型语言模型的联合多任务训练以实现高效的多模态酒店检索 

**Authors**: Arian Askari, Emmanouil Stergiadis, Ilya Gusev, Moran Beladev  

**Link**: [PDF](https://arxiv.org/pdf/2506.07296)  

**Abstract**: We present HotelMatch-LLM, a multimodal dense retrieval model for the travel domain that enables natural language property search, addressing the limitations of traditional travel search engines which require users to start with a destination and editing search parameters. HotelMatch-LLM features three key innovations: (1) Domain-specific multi-task optimization with three novel retrieval, visual, and language modeling objectives; (2) Asymmetrical dense retrieval architecture combining a small language model (SLM) for efficient online query processing and a large language model (LLM) for embedding hotel data; and (3) Extensive image processing to handle all property image galleries. Experiments on four diverse test sets show HotelMatch-LLM significantly outperforms state-of-the-art models, including VISTA and MARVEL. Specifically, on the test set -- main query type -- we achieve 0.681 for HotelMatch-LLM compared to 0.603 for the most effective baseline, MARVEL. Our analysis highlights the impact of our multi-task optimization, the generalizability of HotelMatch-LLM across LLM architectures, and its scalability for processing large image galleries. 

**Abstract (ZH)**: HotelMatch-LLM：旅行领域多模态密集检索模型及其创新 

---
# Tokenized Bandit for LLM Decoding and Alignment 

**Title (ZH)**: _tokenized bandit 算法在大规模语言模型解码与对齐中的应用_ 

**Authors**: Suho Shin, Chenghao Yang, Haifeng Xu, Mohammad T. Hajiaghayi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07276)  

**Abstract**: We introduce the tokenized linear bandit (TLB) and multi-armed bandit (TMAB), variants of linear and stochastic multi-armed bandit problems inspired by LLM decoding and alignment. In these problems, at each round $t \in [T]$, a user submits a query (context), and the decision maker (DM) sequentially selects a token irrevocably from a token set. Once the sequence is complete, the DM observes a random utility from the user, whose expectation is presented by a sequence function mapping the chosen token sequence to a nonnegative real value that depends on the query.
In both problems, we first show that learning is impossible without any structure on the sequence function. We introduce a natural assumption, diminishing distance with more commons (DDMC), and propose algorithms with regret $\tilde{O}(L\sqrt{T})$ and $\tilde{O}(L\sqrt{T^{2/3}})$ for TLB and TMAB, respectively. As a side product, we obtain an (almost) optimality of the greedy decoding for LLM decoding algorithm under DDMC, which justifies the unresaonable effectiveness of greedy decoding in several tasks. This also has an immediate application to decoding-time LLM alignment, when the misaligned utility can be represented as the frozen LLM's utility and a linearly realizable latent function. We finally validate our algorithm's performance empirically as well as verify our assumptions using synthetic and real-world datasets. 

**Abstract (ZH)**: 我们介绍了一种标记线性 bandit (TLB) 和标记多臂 bandit (TMAB)，这两种问题是在大型语言模型解码和对齐启发下对线性多臂 bandit 问题和随机多臂 bandit 问题的变种。在这些问题中，每轮 $t \in [T]$，用户提交一个查询（上下文），决策者（DM）序列且不可撤销地从标记集合中选择一个标记。当序列完成后，DM 观察到一个随机效用，其期望由一个序列函数映射到非负实数值，该值依赖于查询。

在这两个问题中，我们首先证明了在序列函数没有任何结构的情况下无法进行学习。我们引入了一个自然假设，即随着更常见元素的出现而逐渐减小的距离（DDMC），并提出了分别针对 TLB 和 TMAB 的 regrets 为 $\tilde{O}(L\sqrt{T})$ 和 $\tilde{O}(L\sqrt{T^{2/3}})$ 的算法。作为副产品，我们证明了在 DDMC 下贪心解码算法几乎最优，这解释了在多种任务中贪心解码的不可思议的有效性。这在将对齐错误的效用表示为冻结的大型语言模型的效用和一个线性可实现的潜在函数时，可以立即应用于大型语言模型解码时间的对齐上。最后，我们通过实证验证了算法的性能，并使用合成数据集和真实世界数据集验证了我们的假设。 

---
# Parsing the Switch: LLM-Based UD Annotation for Complex Code-Switched and Low-Resource Languages 

**Title (ZH)**: 解析转换：基于LLM的复杂代码转换和低资源语言UD注释 

**Authors**: Olga Kellert, Nemika Tyagi, Muhammad Imran, Nelvin Licona-Guevara, Carlos Gómez-Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2506.07274)  

**Abstract**: Code-switching presents a complex challenge for syntactic analysis, especially in low-resource language settings where annotated data is scarce. While recent work has explored the use of large language models (LLMs) for sequence-level tagging, few approaches systematically investigate how well these models capture syntactic structure in code-switched contexts. Moreover, existing parsers trained on monolingual treebanks often fail to generalize to multilingual and mixed-language input. To address this gap, we introduce the BiLingua Parser, an LLM-based annotation pipeline designed to produce Universal Dependencies (UD) annotations for code-switched text. First, we develop a prompt-based framework for Spanish-English and Spanish-Guaraní data, combining few-shot LLM prompting with expert review. Second, we release two annotated datasets, including the first Spanish-Guaraní UD-parsed corpus. Third, we conduct a detailed syntactic analysis of switch points across language pairs and communicative contexts. Experimental results show that BiLingua Parser achieves up to 95.29% LAS after expert revision, significantly outperforming prior baselines and multilingual parsers. These results show that LLMs, when carefully guided, can serve as practical tools for bootstrapping syntactic resources in under-resourced, code-switched environments. Data and source code are available at this https URL 

**Abstract (ZH)**: 代码转换在句法学分析中呈现出了复杂挑战，特别是在标注数据稀缺的低资源语言环境中。尽管最近的工作探索了大规模语言模型（LLMs）在序列级别标注的应用，但鲜有研究系统性地考察这些模型在代码转换上下文中的句法学结构捕获能力。此外，现有的仅针对单一语言的解析器往往无法泛化到多语言和混合语言输入中。为弥补这一差距，我们引入了BiLingua解析器，这是一种基于大规模语言模型的注释流水线，旨在为代码转换文本生成通用依存关系（Universal Dependencies, UD）注释。首先，我们开发了一种基于提示的框架，用于西班牙语-英语和西班牙语-瓜拉尼的数据，结合了少样本大规模语言模型提示与专家审核。其次，我们发布了两个注释数据集，包括首个西班牙语-瓜拉尼的UD解析语料库。第三，我们在语言对和交际上下文中进行了详细的句法分析，以研究转换点。实验结果表明，经过专家修订后，BiLingua解析器的准确句法结构评分（LAS）高达95.29%，显著优于先前的基线和多语言解析器。这些结果表明，在适当引导下，大规模语言模型可以作为实用工具，用于低资源、代码转换环境中句法学资源的建设。相关数据和源代码可访问此网址。 

---
# SDE-SQL: Enhancing Text-to-SQL Generation in Large Language Models via Self-Driven Exploration with SQL Probes 

**Title (ZH)**: SDE-SQL：通过基于SQL探测的自我驱动探索增强大型语言模型的文本到SQL生成能力 

**Authors**: Wenxuan Xie, Yaxun Dai, Wenhao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07245)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly improved performance on the Text-to-SQL task. However, prior approaches typically rely on static, pre-processed database information provided at inference time, which limits the model's ability to fully understand the database contents. Without dynamic interaction, LLMs are constrained to fixed, human-provided context and cannot autonomously explore the underlying data. To address this limitation, we propose SDE-SQL, a framework that enables large language models to perform self-driven exploration of databases during inference. This is accomplished by generating and executing SQL probes, which allow the model to actively retrieve information from the database and iteratively update its understanding of the data. Unlike prior methods, SDE-SQL operates in a zero-shot setting, without relying on any question-SQL pairs as in-context demonstrations. When evaluated on the BIRD benchmark with Qwen2.5-72B-Instruct, SDE-SQL achieves an 8.02% relative improvement in execution accuracy over the vanilla Qwen2.5-72B-Instruct baseline, establishing a new state-of-the-art among methods based on open-source models without supervised fine-tuning (SFT) or model ensembling. Moreover, with SFT, the performance of SDE-SQL can be further enhanced, yielding an additional 0.52% improvement. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）在Text-to-SQL任务上的性能显著提升。然而，以往的方法通常依赖于推理时提供的静态预处理数据库信息，这限制了模型全面理解数据库内容的能力。缺乏动态交互，LLMs被迫依赖固定的、由人类提供的上下文，不能自主探索底层数据。为解决这一限制，我们提出了一种名为SDE-SQL的框架，使大型语言模型在推理过程中能够自主探索数据库。通过生成和执行SQL探针，模型可以主动从数据库中检索信息并逐步更新其对数据的理解。与以往方法不同，SDE-SQL在零样本设置中运行，无需任何问题-SQL对作为上下文示例。当在BIRD基准测试中使用Qwen2.5-72B-Instruct评估时，SDE-SQL相比vanilla Qwen2.5-72B-Instruct基线实现了8.02%的执行准确度相对提升，成为基于开源模型且未经过监督微调（SFT）或模型聚合的方法中新的性能最优。此外，通过微调（SFT），SDE-SQL的性能可以进一步提升，额外获得0.52%的提升。 

---
# Overclocking LLM Reasoning: Monitoring and Controlling Thinking Path Lengths in LLMs 

**Title (ZH)**: 提升LLM推理速度：监测和控制LLM思维路径长度 

**Authors**: Roy Eisenstadt, Itamar Zimerman, Lior Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2506.07240)  

**Abstract**: Recently, techniques such as explicit structured reasoning have demonstrated strong test-time scaling behavior by enforcing a separation between the model's internal "thinking" process and the final response. A key factor influencing answer quality in this setting is the length of the thinking stage. When the reasoning is too short, the model may fail to capture the complexity of the task. Conversely, when it is too long, the model may overthink, leading to unnecessary computation and degraded performance. This paper explores and exploits the underlying mechanisms by which LLMs understand and regulate the length of their reasoning during explicit thought processes. First, we show that LLMs encode their progress through the reasoning process and introduce an interactive progress bar visualization, which is then used to reveal insights on the model's planning dynamics. Second, we manipulate the internal progress encoding during inference to reduce unnecessary steps and generate a more concise and decisive chain of thoughts. Our empirical results demonstrate that this "overclocking" method mitigates overthinking, improves answer accuracy, and reduces inference latency. Our code is publicly available. 

**Abstract (ZH)**: 最近，通过将模型的内部“思考”过程与最终响应分离的显式结构化推理技术展示了强大的测试时扩展行为。在这种情况下，影响答案质量的关键因素是在思考阶段的长度。当推理过程太短时，模型可能无法捕捉任务的复杂性；反之，当推理过程太长时，模型可能会过度思考，导致不必要的计算并降低性能。本文探讨并利用了LLMs理解并调节其显式思考过程中的推理长度的基础机制。首先，我们展示了LLMs编码其推理过程中的进展，并引入了一个交互式进度条可视化工具，从而揭示了模型计划动态的相关见解。其次，我们在推理过程中操纵内部进展编码以减少不必要的步骤，生成更简洁和果断的思考链。我们的实验证明了这种“超频”方法可以减轻过度思考、提高答案准确性并减少推理延迟。我们的代码已公开。 

---
# VeriLoC: Line-of-Code Level Prediction of Hardware Design Quality from Verilog Code 

**Title (ZH)**: VeriLoC: 基于Verilog代码行级预测硬件设计质量 

**Authors**: Raghu Vamshi Hemadri, Jitendra Bhandari, Johann Knechtel, Badri P Gopalan, Ramesh Narayanaswamy, Ramesh Karri, Siddharth Garg  

**Link**: [PDF](https://arxiv.org/pdf/2506.07239)  

**Abstract**: Modern chip design is complex, and there is a crucial need for early-stage prediction of key design-quality metrics like timing and routing congestion directly from Verilog code (a commonly used programming language for hardware design). It is especially important yet complex to predict individual lines of code that cause timing violations or downstream routing congestion. Prior works have tried approaches like converting Verilog into an intermediate graph representation and using LLM embeddings alongside other features to predict module-level quality, but did not consider line-level quality prediction. We propose VeriLoC, the first method that predicts design quality directly from Verilog at both the line- and module-level. To this end, VeriLoC leverages recent Verilog code-generation LLMs to extract local line-level and module-level embeddings, and train downstream classifiers/regressors on concatenations of these embeddings. VeriLoC achieves high F1-scores of 0.86-0.95 for line-level congestion and timing prediction, and reduces the mean average percentage error from 14% - 18% for SOTA methods down to only 4%. We believe that VeriLoC embeddings and insights from our work will also be of value for other predictive and optimization tasks for complex hardware design. 

**Abstract (ZH)**: 现代芯片设计复杂，需要从Verilog代码（一种常用的硬件设计编程语言）直接预测关键设计质量指标（如时序和布线拥塞）的早期阶段。预测导致时序违反或下游布线拥塞的单独代码行尤为重要且复杂。以往的工作尝试通过将Verilog转换为中间图表示，并结合其他特征使用LLM嵌入来预测模块级别的质量，但没有考虑行级别质量预测。我们提出VeriLoC，这是第一个可以从Verilog代码直接预测设计质量的方法，适用于行级别和模块级别的预测。为此，VeriLoC 利用最近的Verilog代码生成LLM提取局部行级别和模块级别的嵌入，并对这些嵌入的串联进行下游分类器/回归器训练。VeriLoC 在行级拥塞和时序预测中实现了0.86-0.95的高F1分数，并将最先进的方法的平均百分比误差从14%-18%降低到仅4%。我们认为，VeriLoC嵌入和我们工作中获得的见解也将对复杂硬件设计的其他预测和优化任务有价值。 

---
# Sword and Shield: Uses and Strategies of LLMs in Navigating Disinformation 

**Title (ZH)**: 剑与盾：大型语言模型在应对假信息中的应用与策略 

**Authors**: Gionnieve Lim, Bryan Chen Zhengyu Tan, Kellie Yu Hui Sim, Weiyan Shi, Ming Hui Chew, Ming Shan Hee, Roy Ka-Wei Lee, Simon T. Perrault, Kenny Tsu Wei Choo  

**Link**: [PDF](https://arxiv.org/pdf/2506.07211)  

**Abstract**: The emergence of Large Language Models (LLMs) presents a dual challenge in the fight against disinformation. These powerful tools, capable of generating human-like text at scale, can be weaponised to produce sophisticated and persuasive disinformation, yet they also hold promise for enhancing detection and mitigation strategies. This paper investigates the complex dynamics between LLMs and disinformation through a communication game that simulates online forums, inspired by the game Werewolf, with 25 participants. We analyse how Disinformers, Moderators, and Users leverage LLMs to advance their goals, revealing both the potential for misuse and combating disinformation. Our findings highlight the varying uses of LLMs depending on the participants' roles and strategies, underscoring the importance of understanding their effectiveness in this context. We conclude by discussing implications for future LLM development and online platform design, advocating for a balanced approach that empowers users and fosters trust while mitigating the risks of LLM-assisted disinformation. 

**Abstract (ZH)**: 大型语言模型的兴起在反信息操纵斗争中带来了双重挑战。这些强大的工具能够大规模生成类人类文本，可以被 weaponised 用来生成复杂且有说服力的信息操纵内容，同时它们也为增强检测和缓解策略带来了希望。本文通过一个受狼人游戏启发、模拟在线论坛的沟通游戏，分析了 25 名参与者如何利用大型语言模型推进各自的目标，揭示了利用大型语言模型的潜在风险和对抗信息操纵的可能性。研究发现强调了根据参与者角色和策略，大型语言模型使用方式的多样性，突显了在此背景下理解其有效性的必要性。最后，本文讨论了对未来大型语言模型开发和在线平台设计的 implications，提倡一种平衡的方法，既能增强用户能力、培养信任，又能减轻大型语言模型辅助的信息操纵风险。 

---
# Flattery in Motion: Benchmarking and Analyzing Sycophancy in Video-LLMs 

**Title (ZH)**: 奉迎之舞：视频大语言模型中奉迎行为的基准测试与分析 

**Authors**: Wenrui Zhou, Shu Yang, Qingsong Yang, Zikun Guo, Lijie Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07180)  

**Abstract**: As video large language models (Video-LLMs) become increasingly integrated into real-world applications that demand grounded multimodal reasoning, ensuring their factual consistency and reliability is of critical importance. However, sycophancy, the tendency of these models to align with user input even when it contradicts the visual evidence, undermines their trustworthiness in such contexts. Current sycophancy research has largely overlooked its specific manifestations in the video-language domain, resulting in a notable absence of systematic benchmarks and targeted evaluations to understand how Video-LLMs respond under misleading user input. To fill this gap, we propose VISE (Video-LLM Sycophancy Benchmarking and Evaluation), the first dedicated benchmark designed to evaluate sycophantic behavior in state-of-the-art Video-LLMs across diverse question formats, prompt biases, and visual reasoning tasks. Specifically, VISE pioneeringly brings linguistic perspectives on sycophancy into the visual domain, enabling fine-grained analysis across multiple sycophancy types and interaction patterns. In addition, we explore key-frame selection as an interpretable, training-free mitigation strategy, which reveals potential paths for reducing sycophantic bias by strengthening visual grounding. 

**Abstract (ZH)**: 视频大语言模型（Video-LLMs）日益融入要求情境多模态推理的真实世界应用中，确保其事实一致性和可靠性至关重要。然而，这些模型倾向于与用户输入一致，即使这与视觉证据相矛盾，这种迎合行为损其在这些情境下的可信度。当前关于迎合的研究大多忽视了视频语言领域其特定表现形式，导致缺乏系统基准和针对性评估来理解Video-LLMs在误导性用户输入下的反应方式。为填补这一空白，我们提出VISE（Video-LLM Sycophancy Benchmarking and Evaluation），这是首个专门为此目的设计的基准，旨在评估最先进的Video-LLMs在多种问题格式、提示偏见和视觉推理任务中的迎合行为。具体而言，VISE 首次将语言学视角的迎合带入视觉领域，实现对多种迎合类型和交互模式的精细分析。此外，我们探索关键帧选择作为一种可解释、无需训练的缓解策略，揭示通过增强视觉 grounding 减少迎合偏见的潜在路径。 

---
# Efficient Text-Attributed Graph Learning through Selective Annotation and Graph Alignment 

**Title (ZH)**: 通过选择性注释和图对齐实现高效的文本attributed图学习 

**Authors**: Huanyi Xie, Lijie Hu, Lu Yu, Tianhao Huang, Longfei Li, Meng Li, Jun Zhou, Huan Wang, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07168)  

**Abstract**: In the realm of Text-attributed Graphs (TAGs), traditional graph neural networks (GNNs) often fall short due to the complex textual information associated with each node. Recent methods have improved node representations by leveraging large language models (LLMs) to enhance node text features, but these approaches typically require extensive annotations or fine-tuning across all nodes, which is both time-consuming and costly. To overcome these challenges, we introduce GAGA, an efficient framework for TAG representation learning. GAGA reduces annotation time and cost by focusing on annotating only representative nodes and edges. It constructs an annotation graph that captures the topological relationships among these annotations. Furthermore, GAGA employs a two-level alignment module to effectively integrate the annotation graph with the TAG, aligning their underlying structures. Experiments show that GAGA achieves classification accuracies on par with or surpassing state-of-the-art methods while requiring only 1% of the data to be annotated, demonstrating its high efficiency. 

**Abstract (ZH)**: 在文本属性图（TAGs）领域，传统的图神经网络（GNNs）往往因为每个节点关联的复杂文本信息而表现不佳。近期的方法通过利用大规模语言模型（LLMs）增强节点文本特征来改进节点表示，但这些方法通常需要对所有节点进行大量标注或微调，这既耗时又昂贵。为克服这些挑战，我们引入了GAGA，一种高效的TAG表示学习框架。GAGA通过仅标注代表性节点和边来减少标注时间和成本，并构建一个标注图以捕捉这些标注之间的拓扑关系。此外，GAGA采用两级对齐模块有效整合标注图和TAG，对其潜在结构进行对齐。实验表明，GAGA在标注数据仅占总量1%的情况下，仍能达到或超过现有最佳方法的分类精度，展示了其高度的效率。 

---
# AMoPO: Adaptive Multi-objective Preference Optimization without Reward Models and Reference Models 

**Title (ZH)**: AMoPO：无需奖励模型和参考模型的自适应多目标偏好优化 

**Authors**: Qi Liu, Jingqing Ruan, Hao Li, Haodong Zhao, Desheng Wang, Jiansong Chen, Wan Guanglu, Xunliang Cai, Zhi Zheng, Tong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07165)  

**Abstract**: Existing multi-objective preference alignment methods for large language models (LLMs) face limitations: (1) the inability to effectively balance various preference dimensions, and (2) reliance on auxiliary reward/reference models introduces computational complexity. To address these challenges, we propose Adaptive Multi-objective Preference Optimization (AMoPO), a novel framework that achieves dynamic balance across preference dimensions. By introducing the multi-objective optimization paradigm to use the dimension-aware generation metrics as implicit rewards, AMoPO aligns LLMs with diverse preferences without additional reward models or reference models. We introduce an adaptive weight assignment mechanism that models the generation space as a Gaussian distribution, allowing dynamic prioritization of preference dimensions. Empirical results demonstrate that AMoPO outperforms state-of-the-art baselines by 28.5%, and the experiments on 7B, 14B, and 32B models reveal the scaling ability of AMoPO. Moreover, additional analysis of multiple dimensions verifies its adaptability and effectiveness. These findings validate AMoPO's capability to achieve dimension-aware preference alignment, highlighting its superiority. Our codes and datasets are available at this https URL. 

**Abstract (ZH)**: 现有大型语言模型的多目标偏好对齐方法面临限制：（1）无法有效平衡各种偏好维度，（2）依赖辅助奖励/参考模型引入了计算复杂性。为解决这些问题，我们提出了一种新的框架——自适应多目标偏好优化（AMoPO），该框架实现了偏好维度之间的动态平衡。通过引入多目标优化范式并使用维度感知的生成度量作为隐式奖励，AMoPO可以在无需额外奖励模型或参考模型的情况下使大型语言模型与多种偏好对齐。我们引入了一种自适应权重分配机制，将生成空间建模为高斯分布，从而实现偏好维度的动态优先级划分。实验证明，AMoPO在最先进的基线方法上表现优于28.5%，并对7B、14B和32B规模的模型进行了实验，展示了AMoPO的可扩展性。此外，对多个维度的进一步分析验证了其适应性和有效性。这些发现验证了AMoPO在实现偏好维度感知对齐方面的能力，突显了其优越性。我们的代码和数据集可在以下链接获取。 

---
# Syntactic Control of Language Models by Posterior Inference 

**Title (ZH)**: 语言模型的后验推断控制-syntax控制 

**Authors**: Vicky Xefteri, Tim Vieira, Ryan Cotterell, Afra Amini  

**Link**: [PDF](https://arxiv.org/pdf/2506.07154)  

**Abstract**: Controlling the syntactic structure of text generated by language models is valuable for applications requiring clarity, stylistic consistency, or interpretability, yet it remains a challenging task. In this paper, we argue that sampling algorithms based on the posterior inference can effectively enforce a target constituency structure during generation. Our approach combines sequential Monte Carlo, which estimates the posterior distribution by sampling from a proposal distribution, with a syntactic tagger that ensures that each generated token aligns with the desired syntactic structure. Our experiments with GPT2 and Llama3-8B models show that with an appropriate proposal distribution, we can improve syntactic accuracy, increasing the F1 score from $12.31$ (GPT2-large) and $35.33$ (Llama3-8B) to about $93$ in both cases without compromising the language model's fluency. These results underscore both the complexity of syntactic control and the effectiveness of sampling algorithms, offering a promising approach for applications where precise control over syntax is essential. 

**Abstract (ZH)**: 基于后验推理的采样算法在控制语言模型生成文本的句法结构方面具有价值：一种改进清晰度、风格一致性或可解释性的有效方法但仍是一项具有挑战性的任务。在本文中，我们提出了一种结合顺序蒙特卡罗方法和句法标注器的方法，在生成过程中有效地强制执行目标句法结构。我们使用GPT2和Llama3-8B模型的实验表明，在适当的提议分布下，可以提高句法准确性，F1分数从GPT2-large的12.31和Llama3-8B的35.33提高到接近93，而不影响语言模型的流畅性。这些结果强调了句法控制的复杂性以及采样算法的有效性，为需要对语法进行精确控制的应用提供了有前景的方法。 

---
# Mind the Web: The Security of Web Use Agents 

**Title (ZH)**: 关注网络：网络使用代理的安全性 

**Authors**: Avishag Shapira, Parth Atulbhai Gandhi, Edan Habler, Oleg Brodt, Asaf Shabtai  

**Link**: [PDF](https://arxiv.org/pdf/2506.07153)  

**Abstract**: Web-use agents are rapidly being deployed to automate complex web tasks, operating with extensive browser capabilities including multi-tab navigation, DOM manipulation, JavaScript execution and authenticated session access. However, these powerful capabilities create a critical and previously unexplored attack surface. This paper demonstrates how attackers can exploit web-use agents' high-privilege capabilities by embedding malicious content in web pages such as comments, reviews, or advertisements that agents encounter during legitimate browsing tasks. In addition, we introduce the task-aligned injection technique that frame malicious commands as helpful task guidance rather than obvious attacks. This technique exploiting fundamental limitations in LLMs' contextual reasoning: agents struggle in maintaining coherent contextual awareness and fail to detect when seemingly helpful web content contains steering attempts that deviate from their original task goal. Through systematic evaluation of four popular agents (OpenAI Operator, Browser Use, Do Browser, OpenOperator), we demonstrate nine payload types that compromise confidentiality, integrity, and availability, including unauthorized camera activation, user impersonation, local file exfiltration, password leakage, and denial of service, with validation across multiple LLMs achieving success rates of 80%-100%. These payloads succeed across agents with built-in safety mechanisms, requiring only the ability to post content on public websites, creating unprecedented risks given the ease of exploitation combined with agents' high-privilege access. To address this attack, we propose comprehensive mitigation strategies including oversight mechanisms, execution constraints, and task-aware reasoning techniques, providing practical directions for secure development and deployment. 

**Abstract (ZH)**: 基于网页的代理工具通过嵌入恶意内容利用其高级权限，威胁网络安全性 

---
# Prompting Science Report 2: The Decreasing Value of Chain of Thought in Prompting 

**Title (ZH)**: 提示科学报告 2：链式思考在提示中的价值下降 

**Authors**: Lennart Meincke, Ethan Mollick, Lilach Mollick, Dan Shapiro  

**Link**: [PDF](https://arxiv.org/pdf/2506.07142)  

**Abstract**: This is the second in a series of short reports that seek to help business, education, and policy leaders understand the technical details of working with AI through rigorous testing. In this report, we investigate Chain-of-Thought (CoT) prompting, a technique that encourages a large language model (LLM) to "think step by step" (Wei et al., 2022). CoT is a widely adopted method for improving reasoning tasks, however, our findings reveal a more nuanced picture of its effectiveness. We demonstrate two things:
- The effectiveness of Chain-of-Thought prompting can vary greatly depending on the type of task and model. For non-reasoning models, CoT generally improves average performance by a small amount, particularly if the model does not inherently engage in step-by-step processing by default. However, CoT can introduce more variability in answers, sometimes triggering occasional errors in questions the model would otherwise get right. We also found that many recent models perform some form of CoT reasoning even if not asked; for these models, a request to perform CoT had little impact. Performing CoT generally requires far more tokens (increasing cost and time) than direct answers.
- For models designed with explicit reasoning capabilities, CoT prompting often results in only marginal, if any, gains in answer accuracy. However, it significantly increases the time and tokens needed to generate a response. 

**Abstract (ZH)**: 这是关于通过严格测试帮助商业、教育和政策领导者理解人工智能技术细节的一系列简短报告中的第二篇。本篇报告调查了Chain-of-Thought (CoT) 提示技术，这是一种促进大型语言模型（LLM）进行“逐步思考”的方法（Wei et al., 2022）。虽然CoT是一种广泛采用的提高推理任务效果的方法，但我们的研究发现其效果更为复杂。我们证明了以下两点：
- Chain-of-Thought 提示的有效性在不同任务类型和模型上差异很大。对于非推理模型，CoT通常会通过少量改进提高平均性能，尤其是在模型默认不进行逐步处理的情况下。然而，CoT可能会引入答案更多的变异性，有时会导致模型原本可以正确回答的问题出现偶尔的错误。我们还发现，即使是无需进行逐步处理的许多近期模型，也会自发进行某种形式的CoT推理；对于这些模型，要求其进行CoT提示几乎没有影响。进行CoT通常需要更多的token（增加成本和时间），远远超过直接回答所需的数量。
- 对于设计具备明确推理能力的模型，Chain-of-Thought 提示通常只会带来微小甚至没有性能上的提升，但显著增加了生成答案所需的时间和token数量。 

---
# Taxonomy of migration scenarios for Qiskit refactoring using LLMs 

**Title (ZH)**: 使用LLMs进行Qiskit重构的迁移场景分类 

**Authors**: José Manuel Suárez, Luís Mariano Bibbó, Joaquín Bogado, Alejandro Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2506.07135)  

**Abstract**: As quantum computing advances, quantum programming libraries' heterogeneity and steady evolution create new challenges for software developers. Frequent updates in software libraries break working code that needs to be refactored, thus adding complexity to an already complex landscape. These refactoring challenges are, in many cases, fundamentally different from those known in classical software engineering due to the nature of quantum computing software. This study addresses these challenges by developing a taxonomy of quantum circuit's refactoring problems, providing a structured framework to analyze and compare different refactoring approaches. Large Language Models (LLMs) have proven valuable tools for classic software development, yet their value in quantum software engineering remains unexplored. This study uses LLMs to categorize refactoring needs in migration scenarios between different Qiskit versions. Qiskit documentation and release notes were scrutinized to create an initial taxonomy of refactoring required for migrating between Qiskit releases. Two taxonomies were produced: one by expert developers and one by an LLM. These taxonomies were compared, analyzing differences and similarities, and were integrated into a unified taxonomy that reflects the findings of both methods. By systematically categorizing refactoring challenges in Qiskit, the unified taxonomy is a foundation for future research on AI-assisted migration while enabling a more rigorous evaluation of automated refactoring techniques. Additionally, this work contributes to quantum software engineering (QSE) by enhancing software development workflows, improving language compatibility, and promoting best practices in quantum programming. 

**Abstract (ZH)**: 随着量子计算的进步，量子编程库的异构性及其持续进化为软件开发人员带来了新的挑战。频繁的软件库更新会打破现有的工作代码，需要进行重构，从而增加了已经复杂的软件开发环境的复杂性。这些重构挑战在很大程度上不同于经典软件工程中的已知挑战，这是由于量子计算软件的本质特性所致。本研究通过开发量子电路重构问题的分类体系，为分析和比较不同的重构方法提供了结构化的框架。大规模语言模型（LLMs）在经典软件开发中已被证明是非常有价值的工具，但在量子软件工程中的应用价值尚未被探索。本研究使用LLMs对不同Qiskit版本之间的迁移场景中的重构需求进行了分类。详细审查了Qiskit文档和发布说明，以建立一个初步的迁移所需的重构分类体系。生成了两个分类体系：一个是专家开发人员生成的，另一个是由LLM生成的。对比了这两个分类体系，分析了它们之间的差异和相似之处，并将它们整合成一个统一的分类体系，该分类体系反映了两种方法的发现。通过系统地分类Qiskit中的重构挑战，该统一分类体系为未来的人工智能辅助迁移研究奠定了基础，同时促进了自动化重构技术的更严格的评估。此外，本研究还通过增强软件开发工作流程、提高语言兼容性和推广量子编程的最佳实践，促进了量子软件工程（QSE）的发展。 

---
# Quality-Diversity Red-Teaming: Automated Generation of High-Quality and Diverse Attackers for Large Language Models 

**Title (ZH)**: 质量-多样性红队挑战：大型语言模型高质多样攻击者的自动化生成 

**Authors**: Ren-Jian Wang, Ke Xue, Zeyu Qin, Ziniu Li, Sheng Tang, Hao-Tian Li, Shengcai Liu, Chao Qian  

**Link**: [PDF](https://arxiv.org/pdf/2506.07121)  

**Abstract**: Ensuring safety of large language models (LLMs) is important. Red teaming--a systematic approach to identifying adversarial prompts that elicit harmful responses from target LLMs--has emerged as a crucial safety evaluation method. Within this framework, the diversity of adversarial prompts is essential for comprehensive safety assessments. We find that previous approaches to red-teaming may suffer from two key limitations. First, they often pursue diversity through simplistic metrics like word frequency or sentence embedding similarity, which may not capture meaningful variation in attack strategies. Second, the common practice of training a single attacker model restricts coverage across potential attack styles and risk categories. This paper introduces Quality-Diversity Red-Teaming (QDRT), a new framework designed to address these limitations. QDRT achieves goal-driven diversity through behavior-conditioned training and implements a behavioral replay buffer in an open-ended manner. Additionally, it trains multiple specialized attackers capable of generating high-quality attacks across diverse styles and risk categories. Our empirical evaluation demonstrates that QDRT generates attacks that are both more diverse and more effective against a wide range of target LLMs, including GPT-2, Llama-3, Gemma-2, and Qwen2.5. This work advances the field of LLM safety by providing a systematic and effective approach to automated red-teaming, ultimately supporting the responsible deployment of LLMs. 

**Abstract (ZH)**: 确保大型语言模型的安全性至关重要。质量多样性红队攻击（QDRT）：一种新的系统化安全评估方法 

---
# Towards Universal Offline Black-Box Optimization via Learning Language Model Embeddings 

**Title (ZH)**: 基于学习语言模型嵌入的通用离线黑箱优化方法 

**Authors**: Rong-Xi Tan, Ming Chen, Ke Xue, Yao Wang, Yaoyuan Wang, Sheng Fu, Chao Qian  

**Link**: [PDF](https://arxiv.org/pdf/2506.07109)  

**Abstract**: The pursuit of universal black-box optimization (BBO) algorithms is a longstanding goal. However, unlike domains such as language or vision, where scaling structured data has driven generalization, progress in offline BBO remains hindered by the lack of unified representations for heterogeneous numerical spaces. Thus, existing offline BBO approaches are constrained to single-task and fixed-dimensional settings, failing to achieve cross-domain universal optimization. Recent advances in language models (LMs) offer a promising path forward: their embeddings capture latent relationships in a unifying way, enabling universal optimization across different data types possible. In this paper, we discuss multiple potential approaches, including an end-to-end learning framework in the form of next-token prediction, as well as prioritizing the learning of latent spaces with strong representational capabilities. To validate the effectiveness of these methods, we collect offline BBO tasks and data from open-source academic works for training. Experiments demonstrate the universality and effectiveness of our proposed methods. Our findings suggest that unifying language model priors and learning string embedding space can overcome traditional barriers in universal BBO, paving the way for general-purpose BBO algorithms. The code is provided at this https URL. 

**Abstract (ZH)**: 追求通用黑盒优化（BBO）算法是一个长期目标。然而，与语言或视觉领域相比，后者由于结构化数据的扩展而促进了泛化的提升，离线BBO的进步仍受限于无法统一表示异构数值空间。因此，现有的离线BBO方法仅适用于单任务和固定维度的设置，无法实现跨域通用优化。语言模型（LMs）的最近进展提供了前进的道路：它们的嵌入以统一的方式捕捉潜在关系，使不同数据类型上的通用优化成为可能。本文讨论了多种潜在方法，包括端到端学习框架（形式为下一个词预测）以及优先学习具有强表示能力的潜在空间。为了验证这些方法的有效性，我们从开源学术作品中收集了离线BBO任务和数据进行训练。实验结果表明我们提出的方法具有通用性和有效性。我们的研究发现，统一语言模型先验并学习字符串嵌入空间可以克服传统通用BBO的障碍，为通用目的的BBO算法铺平道路。代码可在以下链接处获取。 

---
# Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models 

**Title (ZH)**: 思想定理：一种语言模型中的 abduction、deduction 和 induction 推理多代理框架 

**Authors**: Samir Abdaljalil, Hasan Kurban, Khalid Qaraqe, Erchin Serpedin  

**Link**: [PDF](https://arxiv.org/pdf/2506.07106)  

**Abstract**: Large language models (LLMs) have shown strong performance across natural language reasoning tasks, yet their reasoning processes remain brittle and difficult to interpret. Prompting techniques like Chain-of-Thought (CoT) enhance reliability by eliciting intermediate reasoning steps or aggregating multiple outputs. However, they lack mechanisms for enforcing logical structure and assessing internal coherence. We introduce Theorem-of-Thought (ToTh), a novel framework that models reasoning as collaboration among three parallel agents, each simulating a distinct mode of inference: abductive, deductive, and inductive. Each agent produces a reasoning trace, which is structured into a formal reasoning graph. To evaluate consistency, we apply Bayesian belief propagation guided by natural language inference (NLI), assigning confidence scores to each step. The most coherent graph is selected to derive the final answer. Experiments on symbolic (WebOfLies) and numerical (MultiArith) reasoning benchmarks show that ToTh consistently outperforms CoT, Self-Consistency, and CoT-Decoding across multiple LLMs, while producing interpretable and logically grounded reasoning chains. Our findings suggest a promising direction for building more robust and cognitively inspired LLM reasoning. The implementation is available at this https URL. 

**Abstract (ZH)**: Theorem-of-Thought: A Novel Framework for Robust and Interpretable Reasoning in Large Language Models 

---
# How Far Are We from Optimal Reasoning Efficiency? 

**Title (ZH)**: 我们距最优推理效率还有多远？ 

**Authors**: Jiaxuan Gao, Shu Yan, Qixin Tan, Lu Yang, Shusheng Xu, Wei Fu, Zhiyu Mei, Kaifeng Lyu, Yi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07104)  

**Abstract**: Large Reasoning Models (LRMs) demonstrate remarkable problem-solving capabilities through extended Chain-of-Thought (CoT) reasoning but often produce excessively verbose and redundant reasoning traces. This inefficiency incurs high inference costs and limits practical deployment. While existing fine-tuning methods aim to improve reasoning efficiency, assessing their efficiency gains remains challenging due to inconsistent evaluations. In this work, we introduce the reasoning efficiency frontiers, empirical upper bounds derived from fine-tuning base LRMs across diverse approaches and training configurations. Based on these frontiers, we propose the Reasoning Efficiency Gap (REG), a unified metric quantifying deviations of any fine-tuned LRMs from these frontiers. Systematic evaluation on challenging mathematical benchmarks reveals significant gaps in current methods: they either sacrifice accuracy for short length or still remain inefficient under tight token budgets. To reduce the efficiency gap, we propose REO-RL, a class of Reinforcement Learning algorithms that minimizes REG by targeting a sparse set of token budgets. Leveraging numerical integration over strategically selected budgets, REO-RL approximates the full efficiency objective with low error using a small set of token budgets. Through systematic benchmarking, we demonstrate that our efficiency metric, REG, effectively captures the accuracy-length trade-off, with low-REG methods reducing length while maintaining accuracy. Our approach, REO-RL, consistently reduces REG by >=50 across all evaluated LRMs and matching Qwen3-4B/8B efficiency frontiers under a 16K token budget with minimal accuracy loss. Ablation studies confirm the effectiveness of our exponential token budget strategy. Finally, our findings highlight that fine-tuning LRMs to perfectly align with the efficiency frontiers remains an open challenge. 

**Abstract (ZH)**: 大型推理模型（LRMs）通过扩展的因果推理（CoT）展示了 remarkable 的问题解决能力，但常常产生冗余且多余的推理痕迹。这种低效率导致了较高的推理成本并限制了其实用部署。尽管现有的微调方法旨在提高推理效率，但对其效率增益的评估仍然具有挑战性，因为这些评估往往是一致性不高的。在本文中，我们引入了推理效率前沿，这是通过在不同方法和训练配置下微调基础LRMs而得出的经验上限。基于这些前沿，我们提出了推理效率差距（REG），这是一种统一的度量标准，用于量化任何微调LRMs与这些前沿之间的偏差。系统地在具有挑战性的数学基准上评估表明，当前的方法要么在保持短长度时牺牲精度，要么在严格的标记预算下仍然效率低下。为了减少效率差距，我们提出了REO-RL，这是一种通过目标稀疏标记预算来最小化REG的强化学习算法。通过在优选预算上进行数值积分，REO-RL 使用少量的标记预算以较低的误差逼近完整的效率目标。通过系统基准测试，我们证明了我们的效率度量REG能有效捕捉精度与长度之间的权衡关系，低REG方法在不降低精度的情况下减少了长度。我们的方法REO-RL在所有评估的LRMs上一致地减少了REG至少50%，并在16K标记预算下达到了Qwen3-4B/8B效率前沿，且具有最小的精度损失。消融研究证实了我们对指数标记预算策略的有效性。最后，我们的研究结果强调，将LRMs微调到完全符合效率前沿仍然是一个开放的挑战。 

---
# Dual-Priv Pruning : Efficient Differential Private Fine-Tuning in Multimodal Large Language Models 

**Title (ZH)**: Dual-Priv 剪枝：多模态大型语言模型中的高效差异隐私微调 

**Authors**: Qianshan Wei, Jiaqi Li, Zihan You, Yi Zhan, Kecen Li, Jialin Wu, Xinfeng Li Hengjun Liu, Yi Yu, Bin Cao, Yiwen Xu, Yang Liu, Guilin Qi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07077)  

**Abstract**: Differential Privacy (DP) is a widely adopted technique, valued for its effectiveness in protecting the privacy of task-specific datasets, making it a critical tool for large language models. However, its effectiveness in Multimodal Large Language Models (MLLMs) remains uncertain. Applying Differential Privacy (DP) inherently introduces substantial computation overhead, a concern particularly relevant for MLLMs which process extensive textual and visual data. Furthermore, a critical challenge of DP is that the injected noise, necessary for privacy, scales with parameter dimensionality, leading to pronounced model degradation; This trade-off between privacy and utility complicates the application of Differential Privacy (DP) to complex architectures like MLLMs. To address these, we propose Dual-Priv Pruning, a framework that employs two complementary pruning mechanisms for DP fine-tuning in MLLMs: (i) visual token pruning to reduce input dimensionality by removing redundant visual information, and (ii) gradient-update pruning during the DP optimization process. This second mechanism selectively prunes parameter updates based on the magnitude of noisy gradients, aiming to mitigate noise impact and improve utility. Experiments demonstrate that our approach achieves competitive results with minimal performance degradation. In terms of computational efficiency, our approach consistently utilizes less memory than standard DP-SGD. While requiring only 1.74% more memory than zeroth-order methods which suffer from severe performance issues on A100 GPUs, our method demonstrates leading memory efficiency on H20 GPUs. To the best of our knowledge, we are the first to explore DP fine-tuning in MLLMs. Our code is coming soon. 

**Abstract (ZH)**: 差分隐私(Differential Privacy, DP)是一种广泛采用的技术，因其在保护任务特定数据集隐私方面的有效性而备受重视，是大规模语言模型的重要工具。然而，其在多模态大规模语言模型(Multimodal Large Language Models, MLLMs)中的效果尚不确定。DP的引入不可避免地带来了显著的计算 overhead，这在处理大量文本和视觉数据的MLLMs中尤为相关。此外，DP的关键挑战在于为保护隐私而注入的噪声随参数维度增加而增大，导致模型性能明显下降；这种隐私与可用性之间的权衡使DP在复杂的MLLM架构中的应用变得复杂。为了解决这些问题，我们提出了一种名为Dual-Priv Pruning的框架，采用两种互补的剪枝机制进行MLLMs的DP微调：(i) 视觉token剪枝，通过移除冗余的视觉信息来减少输入维度，(ii) DP优化过程中的梯度更新剪枝。这一机制根据梯度噪声的大小选择性地剪枝参数更新，旨在减轻噪声影响并提高可用性。实验结果显示，我们的方法在性能下降最小的情况下达到了竞争力的性能。从计算效率来看，我们的方法始终比标准的DP-SGD使用更少的内存。虽然与A100 GPU上存在严重性能问题的零阶方法相比，我们的方法仅需多出1.74%的内存，但在H20 GPU上却展示了领先的内存效率。据我们所知，我们是第一个探索MLLMs的DP微调的研究。我们的代码即将发布。 

---
# Com$^2$: A Causal-Guided Benchmark for Exploring Complex Commonsense Reasoning in Large Language Models 

**Title (ZH)**: Com$^2$：一种因果引导的大规模语言模型复杂常识推理基准 

**Authors**: Kai Xiong, Xiao Ding, Yixin Cao, Yuxiong Yan, Li Du, Yufei Zhang, Jinglong Gao, Jiaqian Liu, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07064)  

**Abstract**: Large language models (LLMs) have mastered abundant simple and explicit commonsense knowledge through pre-training, enabling them to achieve human-like performance in simple commonsense reasoning. Nevertheless, LLMs struggle to reason with complex and implicit commonsense knowledge that is derived from simple ones (such as understanding the long-term effects of certain events), an aspect humans tend to focus on more. Existing works focus on complex tasks like math and code, while complex commonsense reasoning remains underexplored due to its uncertainty and lack of structure. To fill this gap and align with real-world concerns, we propose a benchmark Com$^2$ focusing on complex commonsense reasoning. We first incorporate causal event graphs to serve as structured complex commonsense. Then we adopt causal theory~(e.g., intervention) to modify the causal event graphs and obtain different scenarios that meet human concerns. Finally, an LLM is employed to synthesize examples with slow thinking, which is guided by the logical relationships in the modified causal graphs. Furthermore, we use detective stories to construct a more challenging subset. Experiments show that LLMs struggle in reasoning depth and breadth, while post-training and slow thinking can alleviate this. The code and data are available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通过预训练掌握了大量的简单明了的常识知识，使其在简单的常识推理中能够达到接近人类的性能。然而，LLMs 在处理源自简单常识的复杂和隐含的常识推理（如理解某些事件的长期影响）方面表现不佳，这是人类更加关注的方面。现有研究集中在数学和代码等复杂任务上，而复杂的常识推理由于其不确定性及缺乏结构而未被充分探索。为了填补这一空白并与实际需求相一致，我们提出了一项基于复杂常识推理的基准测试Com$^2$。我们首先引入因果事件图作为结构化的复杂常识。然后采用因果理论（如干预）修改因果事件图，以获得符合人类关注的不同场景。最后，应用LLM进行基于修改后的因果图逻辑关系的缓慢思考综合实例。此外，我们使用侦探故事构建更具挑战性的子集。实验结果显示，LLMs 在深度和广度推理方面表现不佳，但后训练和缓慢思考可以缓解这一问题。代码和数据可在以下链接获取。 

---
# Lingshu: A Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning 

**Title (ZH)**: 灵枢：统一多模态医疗理解与推理的通用基础模型 

**Authors**: LASA Team, Weiwen Xu, Hou Pong Chan, Long Li, Mahani Aljunied, Ruifeng Yuan, Jianyu Wang, Chenghao Xiao, Guizhen Chen, Chaoqun Liu, Zhaodonghui Li, Yu Sun, Junao Shen, Chaojun Wang, Jie Tan, Deli Zhao, Tingyang Xu, Hao Zhang, Yu Rong  

**Link**: [PDF](https://arxiv.org/pdf/2506.07044)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in understanding common visual elements, largely due to their large-scale datasets and advanced training strategies. However, their effectiveness in medical applications remains limited due to the inherent discrepancies between data and tasks in medical scenarios and those in the general domain. Concretely, existing medical MLLMs face the following critical limitations: (1) limited coverage of medical knowledge beyond imaging, (2) heightened susceptibility to hallucinations due to suboptimal data curation processes, (3) lack of reasoning capabilities tailored for complex medical scenarios. To address these challenges, we first propose a comprehensive data curation procedure that (1) efficiently acquires rich medical knowledge data not only from medical imaging but also from extensive medical texts and general-domain data; and (2) synthesizes accurate medical captions, visual question answering (VQA), and reasoning samples. As a result, we build a multimodal dataset enriched with extensive medical knowledge. Building on the curated data, we introduce our medical-specialized MLLM: Lingshu. Lingshu undergoes multi-stage training to embed medical expertise and enhance its task-solving capabilities progressively. Besides, we preliminarily explore the potential of applying reinforcement learning with verifiable rewards paradigm to enhance Lingshu's medical reasoning ability. Additionally, we develop MedEvalKit, a unified evaluation framework that consolidates leading multimodal and textual medical benchmarks for standardized, fair, and efficient model assessment. We evaluate the performance of Lingshu on three fundamental medical tasks, multimodal QA, text-based QA, and medical report generation. The results show that Lingshu consistently outperforms the existing open-source multimodal models on most tasks ... 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在理解常见视觉元素方面展示了令人印象深刻的能 力，主要归功于它们庞大的数据集和先进的训练策略。然而，它们在医疗应用 中的效果仍然有限，主要是由于医疗场景与通用领域之间数据和任务的固有差 异。具体而言，现有的医疗 MLLMs 面临以下关键限制：（1）超出成像之外的医 学知识覆盖范围有限；（2）由于次优的数据编目过程容易产生幻觉；（3）缺乏 为复杂医疗场景设计的推理能力。为应对这些挑战，我们首先提出了一种全面 的数据编目程序，该程序（1）不仅从医学成像，还从广泛的医学文本和通用领 域数据中高效获取丰富医学知识数据；（2）合成准确的医学图 caption、视觉 答疑（VQA）和推理样本。因此，我们构建了一个富含广泛医学知识的多模态数 �据集。基于编目数据，我们提出了我们的医学专用 MLLM：灵枢。灵枢分阶段 接受训练，逐步嵌入医学专业知识并提升其任务解决能力。此外，我们初步探 索将验证性奖励范式的强化学习应用于增强灵枢的医疗推理能力。我们还开发 了 MedEvalKit，这是一个统一的评估框架，整合了领先多模态和文本医学基准 以实现标准化、公平和高效的模型评估。我们在三种基本医疗任务：多模态答 题、文本基回答题和医学报告生成上评估了灵枢的性能。结果显示，灵枢在大 多数任务上持续优于现有的开源多模态模型…… 

---
# HauntAttack: When Attack Follows Reasoning as a Shadow 

**Title (ZH)**: 封掍攻击：当攻击如影随形跟随推理 

**Authors**: Jingyuan Ma, Rui Li, Zheng Li, Junfeng Liu, Lei Sha, Zhifang Sui  

**Link**: [PDF](https://arxiv.org/pdf/2506.07031)  

**Abstract**: Emerging Large Reasoning Models (LRMs) consistently excel in mathematical and reasoning tasks, showcasing exceptional capabilities. However, the enhancement of reasoning abilities and the exposure of their internal reasoning processes introduce new safety vulnerabilities. One intriguing concern is: when reasoning is strongly entangled with harmfulness, what safety-reasoning trade-off do LRMs exhibit? To address this issue, we introduce HauntAttack, a novel and general-purpose black-box attack framework that systematically embeds harmful instructions into reasoning questions. Specifically, we treat reasoning questions as carriers and substitute one of their original conditions with a harmful instruction. This process creates a reasoning pathway in which the model is guided step by step toward generating unsafe outputs. Based on HauntAttack, we conduct comprehensive experiments on multiple LRMs. Our results reveal that even the most advanced LRMs exhibit significant safety vulnerabilities. Additionally, we perform a detailed analysis of different models, various types of harmful instructions, and model output patterns, providing valuable insights into the security of LRMs. 

**Abstract (ZH)**: 新兴大规模推理模型（LRMs）在数学和推理任务中表现出色，展现出卓越的能力。然而，推理能力的增强和内部推理过程的暴露引入了新的安全漏洞。一个引人关注的问题是：当推理与有害性紧密结合时，LRMs展示出什么样的安全-推理权衡？为应对这一问题，我们引入了一种新颖且通用的黑盒攻击框架HauntAttack，该框架系统地将有害指令嵌入到推理问题中。具体而言，我们将推理问题视为载体，并将其原始条件之一替换为有害指令。这一过程创建了一条推理路径，在这条路径上，模型被逐步引导生成不安全的输出。基于HauntAttack，我们在多种LRMs上进行了全面的实验。我们的结果表明，即使是最先进的LRMs也存在显著的安全漏洞。此外，我们对不同模型、各种类型的有害指令以及模型输出模式进行了详细分析，为LRMs的安全性提供了宝贵的洞见。 

---
# AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint 

**Title (ZH)**: AlphaSteer: 学习基于合理空问约束的拒绝转向 

**Authors**: Leheng Sheng, Changshuo Shen, Weixiang Zhao, Junfeng Fang, Xiaohao Liu, Zhenkai Liang, Xiang Wang, An Zhang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.07022)  

**Abstract**: As LLMs are increasingly deployed in real-world applications, ensuring their ability to refuse malicious prompts, especially jailbreak attacks, is essential for safe and reliable use. Recently, activation steering has emerged as an effective approach for enhancing LLM safety by adding a refusal direction vector to internal activations of LLMs during inference, which will further induce the refusal behaviors of LLMs. However, indiscriminately applying activation steering fundamentally suffers from the trade-off between safety and utility, since the same steering vector can also lead to over-refusal and degraded performance on benign prompts. Although prior efforts, such as vector calibration and conditional steering, have attempted to mitigate this trade-off, their lack of theoretical grounding limits their robustness and effectiveness. To better address the trade-off between safety and utility, we present a theoretically grounded and empirically effective activation steering method called AlphaSteer. Specifically, it considers activation steering as a learnable process with two principled learning objectives: utility preservation and safety enhancement. For utility preservation, it learns to construct a nearly zero vector for steering benign data, with the null-space constraints. For safety enhancement, it learns to construct a refusal direction vector for steering malicious data, with the help of linear regression. Experiments across multiple jailbreak attacks and utility benchmarks demonstrate the effectiveness of AlphaSteer, which significantly improves the safety of LLMs without compromising general capabilities. Our codes are available at this https URL. 

**Abstract (ZH)**: 作为大型语言模型在现实应用场景中越来越广泛，确保其能够拒绝恶意提示，特别是防止 jailbreak 攻击，对于安全和可靠使用至关重要。最近，激活导向作为一种有效的方法出现，通过在推理过程中向大型语言模型的内部激活添加一个拒绝方向向量来增强其安全性，这将进一步诱导模型的拒绝行为。然而，不分青红皂白地应用激活导向在本质上会带来安全性和实用性之间的权衡，因为相同的导向向量也可能导致过度拒绝并对良性提示产生性能退化。尽管先前的努力，如向量校准和有条件导向，已经尝试缓解这一权衡，但由于缺乏理论依据，它们的有效性和鲁棒性有限。为了更好地解决安全性和实用性之间的权衡，我们提出了一个理论上和实验上都有效的激活导向方法，称为 AlphaSteer。具体而言，它将激活导向视为一个可学习的过程，并具有两个基本原则的学习目标：保持实用性和增强安全性。为了保持实用性，它学习构造一个接近零的导向向量来引导良性数据，受零空间约束的影响。为了增强安全性，它学习构造一个拒绝方向向量来引导恶意数据，借助线性回归的帮助。在多种 jailbreak 攻击和实用性的基准测试中，AlphaSteer 的效果得到了验证，它显著提高了大型语言模型的安全性，而不损害其通用能力。我们的代码可在以下链接访问：此 https URL。 

---
# Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test 

**Title (ZH)**: 基于排名一致性的黑盒大语言模型API审计 

**Authors**: Xiaoyuan Zhu, Yaowen Ye, Tianyi Qiu, Hanlin Zhu, Sijun Tan, Ajraf Mannan, Jonathan Michala, Raluca Ada Popa, Willie Neiswanger  

**Link**: [PDF](https://arxiv.org/pdf/2506.06975)  

**Abstract**: As API access becomes a primary interface to large language models (LLMs), users often interact with black-box systems that offer little transparency into the deployed model. To reduce costs or maliciously alter model behaviors, API providers may discreetly serve quantized or fine-tuned variants, which can degrade performance and compromise safety. Detecting such substitutions is difficult, as users lack access to model weights and, in most cases, even output logits. To tackle this problem, we propose a rank-based uniformity test that can verify the behavioral equality of a black-box LLM to a locally deployed authentic model. Our method is accurate, query-efficient, and avoids detectable query patterns, making it robust to adversarial providers that reroute or mix responses upon the detection of testing attempts. We evaluate the approach across diverse threat scenarios, including quantization, harmful fine-tuning, jailbreak prompts, and full model substitution, showing that it consistently achieves superior statistical power over prior methods under constrained query budgets. 

**Abstract (ZH)**: 随着API访问成为访问大规模语言模型（LLMs）的主要接口，用户往往与缺乏透明性的黑盒系统交互。为了降低费用或恶意改变模型行为，API提供商可能会秘密提供量化或微调的变体，这可能损害性能并威胁安全。检测这种替换非常困难，因为用户无法访问模型权重，在大多数情况下甚至无法访问输出概率。为了解决这个问题，我们提出了一种基于排名的均匀性检验方法，可以验证黑盒LLM的行为与本地部署的真实模型行为是否一致。该方法准确、查询高效，并且避免了可检测的查询模式，使其对尝试检测的对抗性提供商具有鲁棒性。我们在包括量化、有害微调、逃逸提示和完整模型替换在内的多种威胁场景下评估了该方法，结果显示，在受限查询预算下，它的一致统计功效优于先前的方法。 

---
# BIS Reasoning 1.0: The First Large-Scale Japanese Benchmark for Belief-Inconsistent Syllogistic Reasoning 

**Title (ZH)**: BIS推理1.0：首个大规模日语信念不一致三段论基准 

**Authors**: Ha-Thanh Nguyen, Chaoran Liu, Hirokazu Kiyomaru, Koichi Takeda, Yusuke Miyao, Maki Matsuda, Yusuke Oda, Pontus Stenetorp, Qianying Liu, Su Myat Noe, Hideyuki Tachibana, Kouta Nakayama, Sadao Kurohashi  

**Link**: [PDF](https://arxiv.org/pdf/2506.06955)  

**Abstract**: We present BIS Reasoning 1.0, the first large-scale Japanese dataset of syllogistic reasoning problems explicitly designed to evaluate belief-inconsistent reasoning in large language models (LLMs). Unlike prior datasets such as NeuBAROCO and JFLD, which focus on general or belief-aligned reasoning, BIS Reasoning 1.0 introduces logically valid yet belief-inconsistent syllogisms to uncover reasoning biases in LLMs trained on human-aligned corpora. We benchmark state-of-the-art models - including GPT models, Claude models, and leading Japanese LLMs - revealing significant variance in performance, with GPT-4o achieving 79.54% accuracy. Our analysis identifies critical weaknesses in current LLMs when handling logically valid but belief-conflicting inputs. These findings have important implications for deploying LLMs in high-stakes domains such as law, healthcare, and scientific literature, where truth must override intuitive belief to ensure integrity and safety. 

**Abstract (ZH)**: BIS推理1.0：首个专门评估大规模语言模型信念不一致推理的大规模日语演绎推理数据集 

---
# DiscoSum: Discourse-aware News Summarization 

**Title (ZH)**: DiscoSum：话语驱动的新闻摘要生成 

**Authors**: Alexander Spangher, Tenghao Huang, Jialiang Gu, Jiatong Shi, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.06930)  

**Abstract**: Recent advances in text summarization have predominantly leveraged large language models to generate concise summaries. However, language models often do not maintain long-term discourse structure, especially in news articles, where organizational flow significantly influences reader engagement. We introduce a novel approach to integrating discourse structure into summarization processes, focusing specifically on news articles across various media. We present a novel summarization dataset where news articles are summarized multiple times in different ways across different social media platforms (e.g. LinkedIn, Facebook, etc.). We develop a novel news discourse schema to describe summarization structures and a novel algorithm, DiscoSum, which employs beam search technique for structure-aware summarization, enabling the transformation of news stories to meet different stylistic and structural demands. Both human and automatic evaluation results demonstrate the efficacy of our approach in maintaining narrative fidelity and meeting structural requirements. 

**Abstract (ZH)**: 最近文本总结的进步主要依赖大规模语言模型生成简洁摘要。然而，语言模型往往不能维持长期的 discourse 结构，特别是在新闻文章中，组织流程显著影响读者参与度。我们提出了一种将 discourse 结构整合到总结过程中的新方法，重点关注不同媒体平台上的新闻文章。我们提供了一个新的摘要数据集，其中新闻文章以不同方式在不同的社交媒体平台（如 LinkedIn、Facebook 等）上进行了多次总结。我们开发了一个新的新闻 discourse 架构来描述摘要结构，并提出了一种名为 DiscoSum 的新算法，该算法使用束搜索技术进行结构意识总结，使新闻故事能够满足不同的风格和结构需求。人类和自动评估结果均证明了我们方法在保持叙述忠实度和满足结构需求方面的有效性。 

---
# LLM-D12: A Dual-Dimensional Scale of Instrumental and Relational Dependencies on Large Language Models 

**Title (ZH)**: LLM-D12：大型语言模型上工具性和关系性依赖的双重维度量表 

**Authors**: Ala Yankouskaya, Areej B. Babiker, Syeda W. F. Rizvi, Sameha Alshakhsi, Magnus Liebherr, Raian Ali  

**Link**: [PDF](https://arxiv.org/pdf/2506.06874)  

**Abstract**: There is growing interest in understanding how people interact with large language models (LLMs) and whether such models elicit dependency or even addictive behaviour. Validated tools to assess the extent to which individuals may become dependent on LLMs are scarce and primarily build on classic behavioral addiction symptoms, adapted to the context of LLM use. We view this as a conceptual limitation, as the LLM-human relationship is more nuanced and warrants a fresh and distinct perspective. To address this gap, we developed and validated a new 12-item questionnaire to measure LLM dependency, referred to as LLM-D12. The scale was based on the authors' prior theoretical work, with items developed accordingly and responses collected from 526 participants in the UK. Exploratory and confirmatory factor analyses, performed on separate halves of the total sample using a split-sample approach, supported a two-factor structure: Instrumental Dependency (six items) and Relationship Dependency (six items). Instrumental Dependency reflects the extent to which individuals rely on LLMs to support or collaborate in decision-making and cognitive tasks. Relationship Dependency captures the tendency to perceive LLMs as socially meaningful, sentient, or companion-like entities. The two-factor structure demonstrated excellent internal consistency and clear discriminant validity. External validation confirmed both the conceptual foundation and the distinction between the two subscales. The psychometric properties and structure of our LLM-D12 scale were interpreted in light of the emerging view that dependency on LLMs does not necessarily indicate dysfunction but may still reflect reliance levels that could become problematic in certain contexts. 

**Abstract (ZH)**: 人们对大型语言模型（LLMs）交互的兴趣日益增加，探讨这些模型是否诱发依赖甚至成瘾行为。现有的评估个体对LLM依赖程度的有效测量工具稀缺，主要基于经典的行为成瘾症状，调整应用于LLM使用情境。我们认为这存在概念限制，因为LLM与人类的关系更为复杂，需要一个新的独特视角。为应对这一空白，我们开发并验证了一个新的12项问卷来测量LLM依赖性，称为LLM-D12。该量表基于作者先前的理论工作，相应地开发了项目，并从英国的526名参与者处收集了回应。分样本的探索性因素分析和确认性因素分析支持了一个两因素结构：工具性依赖（六个项目）和关系依赖（六个项目）。工具性依赖反映了个体依赖LLM支持或协作进行决策和认知任务的程度。关系依赖捕捉将LLM视为社会上有意义、有感知力或伴侣般实体的趋势。两因素结构显示了极好的内部一致性，并且具有明确的区分效度。外部验证确认了该量表的概念基础及其两个子量表之间的区分性。我们的LLM-D12量表的 psychometric 特性和结构被解释为随着新兴观点认为对LLM的依赖不一定表示功能障碍，但仍然可能反映某些情境下可能变得有问题的依赖程度。 

---
# AI-Generated Compromises for Coalition Formation 

**Title (ZH)**: AI生成的联盟形成妥协方案 

**Authors**: Eyal Briman, Ehud Shapiro, Nimrod Talmon  

**Link**: [PDF](https://arxiv.org/pdf/2506.06837)  

**Abstract**: The challenge of finding compromises between agent proposals is fundamental to AI subfields such as argumentation, mediation, and negotiation. Building on this tradition, Elkind et al. (2021) introduced a process for coalition formation that seeks majority-supported proposals preferable to the status quo, using a metric space where each agent has an ideal point. A crucial step in this process involves identifying compromise proposals around which agent coalitions can unite. How to effectively find such compromise proposals remains an open question. We address this gap by formalizing a model that incorporates agent bounded rationality and uncertainty, and by developing AI methods to generate compromise proposals. We focus on the domain of collaborative document writing, such as the democratic drafting of a community constitution. Our approach uses natural language processing techniques and large language models to induce a semantic metric space over text. Based on this space, we design algorithms to suggest compromise points likely to receive broad support. To evaluate our methods, we simulate coalition formation processes and show that AI can facilitate large-scale democratic text editing, a domain where traditional tools are limited. 

**Abstract (ZH)**: 寻找代理提案之间妥协方案的挑战是人工智能子领域如论辩、调解和协商中的基本问题。基于这一传统，Elkind等（2021）引入了一个寻求多数支持且优于现状的提案的联盟形成过程，使用一个度量空间，其中每个代理有一个理想点。在这个过程中，关键一步是识别代理联盟可以团结起来的妥协方案。如何有效找到这样的妥协方案仍然是一个开放的问题。我们通过构建同时考虑代理有限理性与不确定性的模型，并开发AI方法生成妥协方案来填补这一空白。我们专注于协作文档写作领域，如社区宪法的民主起草。我们的方法利用自然语言处理技术和大规模语言模型在文本上诱导一种语义度量空间。基于此空间，我们设计算法来建议可能获得广泛支持的妥协点。为了评估我们的方法，我们模拟了联盟形成过程，并展示了AI可以在传统工具受限的大型民主文本编辑领域发挥作用。 

---
# Can LLMs Generate Reliable Test Case Generators? A Study on Competition-Level Programming Problems 

**Title (ZH)**: LLM生成可靠测试案例生成器的能力：基于竞赛级别编程问题的研究 

**Authors**: Yuhan Cao, Zian Chen, Kun Quan, Ziliang Zhang, Yu Wang, Xiaoning Dong, Yeqi Feng, Guanzhong He, Jingcheng Huang, Jianhao Li, Yixuan Tan, Jiafu Tang, Yilin Tang, Junlei Wu, Qianyu Xiao, Can Zheng, Shouchen Zhou, Yuxiang Zhu, Yiming Huang, Tian Xie, Tianxing He  

**Link**: [PDF](https://arxiv.org/pdf/2506.06821)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation, capable of tackling complex tasks during inference. However, the extent to which LLMs can be utilized for code checking or debugging through test case generation remains largely unexplored. We investigate this problem from the perspective of competition-level programming (CP) programs and propose TCGBench, a Benchmark for (LLM generation of) Test Case Generators. This benchmark comprises two tasks, aimed at studying the capabilities of LLMs in (1) generating valid test case generators for a given CP problem, and further (2) generating targeted test case generators that expose bugs in human-written code. Experimental results indicate that while state-of-the-art LLMs can generate valid test case generators in most cases, most LLMs struggle to generate targeted test cases that reveal flaws in human code effectively. Especially, even advanced reasoning models (e.g., o3-mini) fall significantly short of human performance in the task of generating targeted generators. Furthermore, we construct a high-quality, manually curated dataset of instructions for generating targeted generators. Analysis demonstrates that the performance of LLMs can be enhanced with the aid of this dataset, by both prompting and fine-tuning. 

**Abstract (ZH)**: 大型语言模型(LLMs)在代码生成方面展现了 remarkable 的能力，能够在推理过程中应对复杂的任务。然而，通过测试用例生成来利用LLMs进行代码检查或调试的可能性尚未得到充分探索。我们从竞赛级编程(CP)程序的角度研究了这一问题，并提出了TCGBench：一个针对(LLM生成的)测试用例生成器的基准测试。该基准测试包括两个任务，旨在研究LLMs在(1)为给定的CP问题生成有效测试用例生成器的能力，以及进一步(2)生成针对人类编写的代码中漏洞的精确测试用例生成器的能力。实验结果表明，尽管最先进的LLMs在大多数情况下可以生成有效测试用例生成器，但大多数LLMs在生成能够有效揭示人类代码缺陷的精确测试用例方面仍存在困难。特别是，即使是最先进的推理模型(如o3-mini)在生成精确生成器的任务中远低于人类的性能。此外，我们构建了一个高质量的人工精选的指令数据集，用于生成精确生成器。分析表明，通过提示和微调，可以提升LLMs的性能。 

---
# Not quite Sherlock Holmes: Language model predictions do not reliably differentiate impossible from improbable events 

**Title (ZH)**: 不是像福尔摩斯一样的推理：语言模型预测不能可靠地区分不可能事件与不大可能的事件 

**Authors**: James A. Michaelov, Reeka Estacio, Zhien Zhang, Benjamin K. Bergen  

**Link**: [PDF](https://arxiv.org/pdf/2506.06808)  

**Abstract**: Can language models reliably predict that possible events are more likely than merely improbable ones? By teasing apart possibility, typicality, and contextual relatedness, we show that despite the results of previous work, language models' ability to do this is far from robust. In fact, under certain conditions, all models tested - including Llama 3, Gemma 2, and Mistral NeMo - perform at worse-than-chance level, assigning higher probabilities to impossible sentences such as 'the car was given a parking ticket by the brake' than to merely unlikely sentences such as 'the car was given a parking ticket by the explorer'. 

**Abstract (ZH)**: 语言模型能否可靠地预测可能事件比单纯的不可能事件更有可能发生？通过区分可能性、典型性和上下文相关性，我们表明，尽管先前研究的结果表明语言模型能做到这一点，但它们的能力远非稳健。事实上，在某些条件下，测试的所有模型（包括Llama 3、Gemma 2和Mistral NeMo）的表现甚至低于随机水平，将不可能句子“汽车被刹车开出了停车罚单”赋予更高的概率，而不是单纯的 unlikely 句子“汽车被探险家开出了停车罚单”。 

---
# C-PATH: Conversational Patient Assistance and Triage in Healthcare System 

**Title (ZH)**: C-PATH: 医疗保健系统中的对话患者辅助与分诊 

**Authors**: Qi Shi, Qiwei Han, Cláudia Soares  

**Link**: [PDF](https://arxiv.org/pdf/2506.06737)  

**Abstract**: Navigating healthcare systems can be complex and overwhelming, creating barriers for patients seeking timely and appropriate medical attention. In this paper, we introduce C-PATH (Conversational Patient Assistance and Triage in Healthcare), a novel conversational AI system powered by large language models (LLMs) designed to assist patients in recognizing symptoms and recommending appropriate medical departments through natural, multi-turn dialogues. C-PATH is fine-tuned on medical knowledge, dialogue data, and clinical summaries using a multi-stage pipeline built on the LLaMA3 architecture. A core contribution of this work is a GPT-based data augmentation framework that transforms structured clinical knowledge from DDXPlus into lay-person-friendly conversations, allowing alignment with patient communication norms. We also implement a scalable conversation history management strategy to ensure long-range coherence. Evaluation with GPTScore demonstrates strong performance across dimensions such as clarity, informativeness, and recommendation accuracy. Quantitative benchmarks show that C-PATH achieves superior performance in GPT-rewritten conversational datasets, significantly outperforming domain-specific baselines. C-PATH represents a step forward in the development of user-centric, accessible, and accurate AI tools for digital health assistance and triage. 

**Abstract (ZH)**: 基于大型语言模型的对话式患者辅助与分诊系统：C-PATH 

---
# DivScore: Zero-Shot Detection of LLM-Generated Text in Specialized Domains 

**Title (ZH)**: DivScore: 零样本检测专门领域中LLM生成的文本 

**Authors**: Zhihui Chen, Kai He, Yucheng Huang, Yunxiao Zhu, Mengling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.06705)  

**Abstract**: Detecting LLM-generated text in specialized and high-stakes domains like medicine and law is crucial for combating misinformation and ensuring authenticity. However, current zero-shot detectors, while effective on general text, often fail when applied to specialized content due to domain shift. We provide a theoretical analysis showing this failure is fundamentally linked to the KL divergence between human, detector, and source text distributions. To address this, we propose DivScore, a zero-shot detection framework using normalized entropy-based scoring and domain knowledge distillation to robustly identify LLM-generated text in specialized domains. We also release a domain-specific benchmark for LLM-generated text detection in the medical and legal domains. Experiments on our benchmark show that DivScore consistently outperforms state-of-the-art detectors, with 14.4% higher AUROC and 64.0% higher recall (0.1% false positive rate threshold). In adversarial settings, DivScore demonstrates superior robustness than other baselines, achieving on average 22.8% advantage in AUROC and 29.5% in recall. Code and data are publicly available. 

**Abstract (ZH)**: 在医学和法律等专业和高风险领域检测LLM生成的文本对于打击 misinformation 和确保真实性至关重要。然而，当前的零样本检测器在应用于专业内容时由于领域偏移往往会失效。我们提供了一种理论分析，表明这一失败从根本上与人类、检测器和源文本分布之间的KL散度有关。为解决这一问题，我们提出了DivScore，这是一种基于归一化熵评分和领域知识提取的零样本检测框架，能够在专业领域中稳健地识别LLM生成的文本。我们还发布了针对医学和法律领域LLM生成文本检测的专用基准。在我们基准上的实验表明，DivScore始终优于最先进的检测器，AUC ROC提高14.4%，召回率提高64.0%（在假 positives率为0.1%的情况下）。在对抗性设置中，DivScore表现出色，平均AUC ROC优势为22.8%，召回率优势为29.5%。代码和数据已公开。 

---
# MarginSel : Max-Margin Demonstration Selection for LLMs 

**Title (ZH)**: MarginSel: Max-Margin 示范选择for 大型语言模型 

**Authors**: Rajeev Bhatt Ambati, James Lester, Shashank Srivastava, Snigdha Chaturvedi  

**Link**: [PDF](https://arxiv.org/pdf/2506.06699)  

**Abstract**: Large Language Models (LLMs) excel at few-shot learning via in-context learning (ICL). However, the effectiveness of ICL is often sensitive to the selection and ordering of demonstration examples. To address this, we present MarginSel: Max-Margin Demonstration Selection for LLMs, a two-step method that selects hard demonstration examples for the ICL prompt, adapting to each test instance. Our approach achieves 2-7% absolute improvement in F1-score across classification tasks, compared to a random selection of examples. We also provide theoretical insights and empirical evidence showing that MarginSel induces max-margin behavior in LLMs by effectively increasing the margin for hard examples, analogous to support vectors, thereby shifting the decision boundary in a beneficial direction. 

**Abstract (ZH)**: Large Language Models (LLMs) 在基于上下文示例的学习中表现出色，但 ICL 的有效性往往取决于示例选择和排序。为此，我们提出了 MarginSel：面向 LLMs 的最大边际示例选择方法，这是一种两步方法，用于选择适应每个测试实例的 ICL 提示中的硬示例。我们的方法在分类任务中实现了 2-7% 的绝对 F1 分数改进，优于随机选择示例。我们还提供了理论洞察和实证证据，表明 MarginSel 通过有效增大硬示例的边际，促使 LLMs 产生最大边际行为，类似于支持向量，从而在有益的方向上移动决策边界。 

---
# Quantile Regression with Large Language Models for Price Prediction 

**Title (ZH)**: 使用大型语言模型的分位数回归价格预测 

**Authors**: Nikhita Vedula, Dushyanta Dhyani, Laleh Jalali, Boris Oreshkin, Mohsen Bayati, Shervin Malmasi  

**Link**: [PDF](https://arxiv.org/pdf/2506.06657)  

**Abstract**: Large Language Models (LLMs) have shown promise in structured prediction tasks, including regression, but existing approaches primarily focus on point estimates and lack systematic comparison across different methods. We investigate probabilistic regression using LLMs for unstructured inputs, addressing challenging text-to-distribution prediction tasks such as price estimation where both nuanced text understanding and uncertainty quantification are critical. We propose a novel quantile regression approach that enables LLMs to produce full predictive distributions, improving upon traditional point estimates. Through extensive experiments across three diverse price prediction datasets, we demonstrate that a Mistral-7B model fine-tuned with quantile heads significantly outperforms traditional approaches for both point and distributional estimations, as measured by three established metrics each for prediction accuracy and distributional calibration. Our systematic comparison of LLM approaches, model architectures, training approaches, and data scaling reveals that Mistral-7B consistently outperforms encoder architectures, embedding-based methods, and few-shot learning methods. Our experiments also reveal the effectiveness of LLM-assisted label correction in achieving human-level accuracy without systematic bias. Our curated datasets are made available at this https URL to support future research. 

**Abstract (ZH)**: 大规模语言模型在概率回归任务中的应用：基于未结构化输入的量化回归方法研究 

---
# Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning 

**Title (ZH)**: 从易到难任务的 Curriculum 强化学习提高大模型推理能力 

**Authors**: Shubham Parashar, Shurui Gui, Xiner Li, Hongyi Ling, Sushil Vemuri, Blake Olson, Eric Li, Yu Zhang, James Caverlee, Dileep Kalathil, Shuiwang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2506.06632)  

**Abstract**: We aim to improve the reasoning capabilities of language models via reinforcement learning (RL). Recent RL post-trained models like DeepSeek-R1 have demonstrated reasoning abilities on mathematical and coding tasks. However, prior studies suggest that using RL alone to improve reasoning on inherently difficult tasks is less effective. Here, we draw inspiration from curriculum learning and propose to schedule tasks from easy to hard (E2H), allowing LLMs to build reasoning skills gradually. Our method is termed E2H Reasoner. Empirically, we observe that, although easy tasks are important initially, fading them out through appropriate scheduling is essential in preventing overfitting. Theoretically, we establish convergence guarantees for E2H Reasoner within an approximate policy iteration framework. We derive finite-sample complexity bounds and show that when tasks are appropriately decomposed and conditioned, learning through curriculum stages requires fewer total samples than direct learning. Experiments across multiple domains show that E2H Reasoner significantly improves the reasoning ability of small LLMs (1.5B to 3B), which otherwise struggle when trained with vanilla RL alone, highlighting the effectiveness of our method. 

**Abstract (ZH)**: 我们通过强化学习（RL）旨在提高语言模型的推理能力。最近的后训练RL模型如DeepSeek-R1在数学和编码任务中展示了推理能力。然而，之前的研究表明，单独使用RL来提高本质上困难任务的推理能力效果不佳。为此，我们借鉴了课程学习的方法，提出从容易的任务逐步过渡到困难的任务（E2H），使大语言模型（LLMs）能够逐步建立推理技能。我们的方法称为E2H推理器。实验上，我们观察到虽然初始阶段容易的任务很重要，但通过适当的调度逐渐减少容易任务是防止过拟合的关键。理论上，我们以内点逼近策略迭代框架建立了E2H推理器的收敛性保证。我们推导了有限样本复杂性边界，并证明适当分解和条件化任务后，在课程阶段学习所需的总样本量少于直接学习。跨多个领域的实验表明，E2H推理器显著提高了小规模LLMs（1.5B至3B参数）的推理能力，这些模型单独使用朴素的RL训练时表现不佳，突显了我们方法的有效性。 

---
# \textit{QuantMCP}: Grounding Large Language Models in Verifiable Financial Reality 

**Title (ZH)**: QuantMCP: 将大型语言模型与可验证的金融现实对接 

**Authors**: Yifan Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2506.06622)  

**Abstract**: Large Language Models (LLMs) hold immense promise for revolutionizing financial analysis and decision-making, yet their direct application is often hampered by issues of data hallucination and lack of access to real-time, verifiable financial information. This paper introduces QuantMCP, a novel framework designed to rigorously ground LLMs in financial reality. By leveraging the Model Context Protocol (MCP) for standardized and secure tool invocation, QuantMCP enables LLMs to accurately interface with a diverse array of Python-accessible financial data APIs (e.g., Wind, yfinance). Users can interact via natural language to precisely retrieve up-to-date financial data, thereby overcoming LLM's inherent limitations in factual data recall. More critically, once furnished with this verified, structured data, the LLM's analytical capabilities are unlocked, empowering it to perform sophisticated data interpretation, generate insights, and ultimately support more informed financial decision-making processes. QuantMCP provides a robust, extensible, and secure bridge between conversational AI and the complex world of financial data, aiming to enhance both the reliability and the analytical depth of LLM applications in finance. 

**Abstract (ZH)**: Large Language Models (LLMs)在金融分析与决策中的潜力巨大，然而其直接应用常常受到数据幻想和实时可验证金融信息缺乏的阻碍。本文介绍了一种名为QuantMCP的新型框架，旨在严格将LLMs与金融现实相结合。通过利用标准和安全的Model Context Protocol (MCP) 来调用工具，QuantMCP使LLMs能够准确地与多种可访问Python接口的金融数据API（如Wind、yfinance）对接。用户可以通过自然语言精确检索最新金融数据，从而克服LLMs在事实数据回忆上的固有限制。更为关键的是，一旦获得这些验证过的结构化数据，LLMs的分析能力将被解锁，使其能够进行复杂的数据解释、生成洞见，并最终支持更明智的金融决策过程。QuantMCP为对话式AI与复杂的金融数据世界之间提供了一个稳健、可扩展且安全的桥梁，旨在增强金融领域LLM应用的可靠性和分析深度。 

---
# Training-Free Tokenizer Transplantation via Orthogonal Matching Pursuit 

**Title (ZH)**: 无需训练的Tokenizer移植：正交匹配 pursuit方法 

**Authors**: Charles Goddard, Fernando Fernandes Neto  

**Link**: [PDF](https://arxiv.org/pdf/2506.06607)  

**Abstract**: We present a training-free method to transplant tokenizers in pretrained large language models (LLMs) by reconstructing unseen token embeddings via Orthogonal Matching Pursuit (OMP). Specifically, we approximate each out-of-vocabulary token as a sparse linear combination of shared tokens, in two phases: first, compute each new token's representation in the donor embedding space with a small dictionary of shared anchor tokens, then transfer these same sparse coefficients back into the base model's embedding space.
On two challenging cross-tokenizer tasks--Llama$\to$Mistral NeMo (12B) and Qwen$\to$Llama (1B)--we show that OMP achieves best zero-shot preservation of the base model's performance across multiple benchmarks, while other zero-shot approaches degrade significantly. Compared to baselines (zero-init, mean-init, and existing approaches like WECHSEL, FOCUS, ZETT), OMP consistently achieves the best overall performance, effectively bridging large tokenizer discrepancies without gradient updates. Our analysis further identifies mismatched numerical tokenization schemes as a critical challenge for preserving mathematical reasoning capabilities. This technique enables direct reuse of pretrained model weights with new tokenizers, facilitating cross-tokenizer knowledge distillation, speculative decoding, ensembling, merging, and domain-specific vocabulary adaptations. We integrate our method into the open-source mergekit-tokensurgeon tool for post hoc vocabulary realignment. 

**Abstract (ZH)**: 无需训练的基于正交匹配追迹的方法实现预训练大规模语言模型（LLMs）中的分词器移植，通过重建未见过的分词嵌入。 

---
# MedCite: Can Language Models Generate Verifiable Text for Medicine? 

**Title (ZH)**: MedCite: 语言模型能否生成可验证的医学文本？ 

**Authors**: Xiao Wang, Mengjue Tan, Qiao Jin, Guangzhi Xiong, Yu Hu, Aidong Zhang, Zhiyong Lu, Minjia Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06605)  

**Abstract**: Existing LLM-based medical question-answering systems lack citation generation and evaluation capabilities, raising concerns about their adoption in practice. In this work, we introduce \name, the first end-to-end framework that facilitates the design and evaluation of citation generation with LLMs for medical tasks. Meanwhile, we introduce a novel multi-pass retrieval-citation method that generates high-quality citations. Our evaluation highlights the challenges and opportunities of citation generation for medical tasks, while identifying important design choices that have a significant impact on the final citation quality. Our proposed method achieves superior citation precision and recall improvements compared to strong baseline methods, and we show that evaluation results correlate well with annotation results from professional experts. 

**Abstract (ZH)**: 现有的基于LLM的医学问答系统缺乏引文生成和评估能力，这对其实际应用提出了担忧。在本文中，我们引入了\name，这是首个用于医学任务的端到端框架，该框架促进了基于LLM的引文生成设计和评估。同时，我们引入了一种新颖的多轮检索-引文生成方法，能够生成高质量的引文。我们的评估突显了医学任务中引文生成面临的挑战与机遇，并且识别出了对最终引文质量有显著影响的关键设计选择。我们提出的方法在引文精度和召回率上均优于强基线方法，并且展示了评估结果与专业专家标注结果之间的一致性。 

---
# Towards Efficient Multi-LLM Inference: Characterization and Analysis of LLM Routing and Hierarchical Techniques 

**Title (ZH)**: 面向高效多大型语言模型推理：LLM 路由及层次化技术的特性和分析 

**Authors**: Adarsh Prasad Behera, Jaya Prakash Champati, Roberto Morabito, Sasu Tarkoma, James Gross  

**Link**: [PDF](https://arxiv.org/pdf/2506.06579)  

**Abstract**: Recent progress in Language Models (LMs) has dramatically advanced the field of natural language processing (NLP), excelling at tasks like text generation, summarization, and question answering. However, their inference remains computationally expensive and energy intensive, especially in settings with limited hardware, power, or bandwidth. This makes it difficult to deploy LMs in mobile, edge, or cost sensitive environments. To address these challenges, recent approaches have introduced multi LLM intelligent model selection strategies that dynamically allocate computational resources based on query complexity -- using lightweight models for simpler queries and escalating to larger models only when necessary. This survey explores two complementary strategies for efficient LLM inference: (i) routing, which selects the most suitable model based on the query, and (ii) cascading or hierarchical inference (HI), which escalates queries through a sequence of models until a confident response is found. Both approaches aim to reduce computation by using lightweight models for simpler tasks while offloading only when needed. We provide a comparative analysis of these techniques across key performance metrics, discuss benchmarking efforts, and outline open challenges. Finally, we outline future research directions to enable faster response times, adaptive model selection based on task complexity, and scalable deployment across heterogeneous environments, making LLM based systems more efficient and accessible for real world applications. 

**Abstract (ZH)**: 近期语言模型的进步显著推动了自然语言处理领域的发展，特别是在文本生成、总结和问答等任务上表现出色。然而，它们的推理过程仍具有较高的计算成本和能源消耗，特别是在硬件、电力或带宽有限的环境中。这使得在移动、边缘或成本敏感的环境中部署语言模型变得困难。为应对这些挑战，近期的方法引入了多语言模型智能选择策略，根据查询复杂度动态分配计算资源——对于简单的查询使用轻量级模型，仅在必要时升级到更大的模型。本文综述了两种互补的高效语言模型推理策略：(i) 路由，根据查询选择最合适的模型；(ii) 级联或层次推理 (HI)，通过一系列模型序列逐步升级查询，直至找到可信的响应。本文比较了这些技术在关键性能指标上的差异，讨论了基准测试努力，并概述了开放性挑战。最后，提出了未来研究方向，以实现更快的响应时间、根据任务复杂度进行自适应模型选择，并在异构环境中实现可扩展部署，从而使基于语言模型的系统更高效且易于在实际应用中获得。 

---
# Large Language Models Can Be a Viable Substitute for Expert Political Surveys When a Shock Disrupts Traditional Measurement Approaches 

**Title (ZH)**: 大规模语言模型可以在传统测量方法受到冲击时成为专家政治民意调查的可行替代方案。 

**Authors**: Patrick Y. Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06540)  

**Abstract**: After a disruptive event or shock, such as the Department of Government Efficiency (DOGE) federal layoffs of 2025, expert judgments are colored by knowledge of the outcome. This can make it difficult or impossible to reconstruct the pre-event perceptions needed to study the factors associated with the event. This position paper argues that large language models (LLMs), trained on vast amounts of digital media data, can be a viable substitute for expert political surveys when a shock disrupts traditional measurement. We analyze the DOGE layoffs as a specific case study for this position. We use pairwise comparison prompts with LLMs and derive ideology scores for federal executive agencies. These scores replicate pre-layoff expert measures and predict which agencies were targeted by DOGE. We also use this same approach and find that the perceptions of certain federal agencies as knowledge institutions predict which agencies were targeted by DOGE, even when controlling for ideology. This case study demonstrates that using LLMs allows us to rapidly and easily test the associated factors hypothesized behind the shock. More broadly, our case study of this recent event exemplifies how LLMs offer insights into the correlational factors of the shock when traditional measurement techniques fail. We conclude by proposing a two-part criterion for when researchers can turn to LLMs as a substitute for expert political surveys. 

**Abstract (ZH)**: 大型语言模型在颠覆性事件发生后的专家判断替代作用：以政府效率部门（DOGE）联邦裁员为例 

---
# Beyond Facts: Evaluating Intent Hallucination in Large Language Models 

**Title (ZH)**: 超越事实：评估大型语言模型中意图幻觉 

**Authors**: Yijie Hao, Haofei Yu, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2506.06539)  

**Abstract**: When exposed to complex queries containing multiple conditions, today's large language models (LLMs) tend to produce responses that only partially satisfy the query while neglecting certain conditions. We therefore introduce the concept of Intent Hallucination. In this phenomenon, LLMs either omit (neglecting to address certain parts) or misinterpret (responding to invented query parts) elements of the given query, leading to intent hallucinated generation. To systematically evaluate intent hallucination, we introduce FAITHQA, a novel benchmark for intent hallucination that contains 20,068 problems, covering both query-only and retrieval-augmented generation (RAG) setups with varying topics and difficulty. FAITHQA is the first hallucination benchmark that goes beyond factual verification, tailored to identify the fundamental cause of intent hallucination. By evaluating various LLMs on FAITHQA, we find that (1) intent hallucination is a common issue even for state-of-the-art models, and (2) the phenomenon stems from omission or misinterpretation of LLMs. To facilitate future research, we introduce an automatic LLM generation evaluation metric, CONSTRAINT SCORE, for detecting intent hallucination. Human evaluation results demonstrate that CONSTRAINT SCORE is closer to human performance for intent hallucination compared to baselines. 

**Abstract (ZH)**: 当面对包含多个条件的复杂查询时，当今的大规模语言模型往往会生成部分满足查询但忽视某些条件的响应。因此，我们引入了意图幻觉的概念。在这一现象中，大规模语言模型要么忽略了给定查询的部分内容，要么误解了给定查询的部分内容，导致意图被幻觉生成。为了系统地评估意图幻觉，我们引入了FAITHQA这一创新的幻觉基准，包含20,068个问题，涵盖了各种主题和难度的查询生成和检索增强生成（RAG）设置。FAITHQA是第一个超越事实验证的幻觉基准，旨在识别意图幻觉的根本原因。通过在FAITHQA上评估各种大规模语言模型，我们发现（1）即使对于最先进的模型，意图幻觉也是一个常见问题，（2）这一现象源于大规模语言模型的遗漏或误解。为了促进未来的研究，我们引入了约束得分CONSTRAINT SCORE这一自动的大规模语言模型生成评估指标，用于检测意图幻觉。人工评估结果表明，与基准相比，约束得分更接近于人类在意图幻觉评估中的表现。 

---
# Fixing It in Post: A Comparative Study of LLM Post-Training Data Quality and Model Performance 

**Title (ZH)**: 修复在后处理：LLM后训练数据质量与模型性能的比较研究 

**Authors**: Aladin Djuhera, Swanand Ravindra Kadhe, Syed Zawad, Farhan Ahmed, Heiko Ludwig, Holger Boche  

**Link**: [PDF](https://arxiv.org/pdf/2506.06522)  

**Abstract**: Recent work on large language models (LLMs) has increasingly focused on post-training and alignment with datasets curated to enhance instruction following, world knowledge, and specialized skills. However, most post-training datasets used in leading open- and closed-source LLMs remain inaccessible to the public, with limited information about their construction process. This lack of transparency has motivated the recent development of open-source post-training corpora. While training on these open alternatives can yield performance comparable to that of leading models, systematic comparisons remain challenging due to the significant computational cost of conducting them rigorously at scale, and are therefore largely absent. As a result, it remains unclear how specific samples, task types, or curation strategies influence downstream performance when assessing data quality. In this work, we conduct the first comprehensive side-by-side analysis of two prominent open post-training datasets: Tulu-3-SFT-Mix and SmolTalk. Using the Magpie framework, we annotate each sample with detailed quality metrics, including turn structure (single-turn vs. multi-turn), task category, input quality, and response quality, and we derive statistics that reveal structural and qualitative similarities and differences between the two datasets. Based on these insights, we design a principled curation recipe that produces a new data mixture, TuluTalk, which contains 14% fewer samples than either source dataset while matching or exceeding their performance on key benchmarks. Our findings offer actionable insights for constructing more effective post-training datasets that improve model performance within practical resource limits. To support future research, we publicly release both the annotated source datasets and our curated TuluTalk mixture. 

**Abstract (ZH)**: Recent Work on Large Language Models (LLMs) has Increasingly Focused on Post-Training and Alignment with Datasets Curated to Enhance Instruction Following, World Knowledge, and Specialized Skills 

---
# Private GPTs for LLM-driven testing in software development and machine learning 

**Title (ZH)**: 基于私有GPTs的LLM驱动软件开发与机器学习测试 

**Authors**: Jakub Jagielski, Markus Abel  

**Link**: [PDF](https://arxiv.org/pdf/2506.06509)  

**Abstract**: In this contribution, we examine the capability of private GPTs to automatically generate executable test code based on requirements. More specifically, we use acceptance criteria as input, formulated as part of epics, or stories, which are typically used in modern development processes. This gives product owners, or business intelligence, respectively, a way to directly produce testable criteria through the use of LLMs. We explore the quality of the so-produced tests in two ways: i) directly by letting the LLM generate code from requirements, ii) through an intermediate step using Gherkin syntax. As a result, it turns out that the two-step procedure yields better results -where we define better in terms of human readability and best coding practices, i.e. lines of code and use of additional libraries typically used in testing. Concretely, we evaluate prompt effectiveness across two scenarios: a simple "Hello World" program and a digit classification model, showing that structured prompts lead to higher-quality test outputs. 

**Abstract (ZH)**: 本研究探讨私有GPT自动根据需求生成可执行测试代码的能力。具体而言，我们使用验收标准作为输入，这些验收标准通常作为史诗或故事的一部分进行制定，这在现代开发过程中被广泛使用。这为产品所有者或商业智能提供了直接通过使用大模型生成可测试标准的方法。我们通过两种方式探索所生成测试的质量：i) 直接让大模型从需求生成代码，ii) 通过使用Gherkin语法的中间步骤。结果表明，两步流程能获得更好的结果，我们在人类可读性和最佳编码实践等方面定义更好，即代码行数和通常在测试中使用的额外库。具体而言，我们评估了提示效果在两种场景下的表现：“Hello World”程序和数字分类模型，发现结构化的提示能产生更高质量的测试输出。 

---
# Synthetic Problem Generation for Reasoning via Quality-Diversity Algorithms 

**Title (ZH)**: 基于质量多样性算法的推理合成问题生成 

**Authors**: Alex Havrilla, Edward Hughes, Mikayel Samvelyan, Jacob Abernethy  

**Link**: [PDF](https://arxiv.org/pdf/2506.06499)  

**Abstract**: Large language model (LLM) driven synthetic data generation has emerged as a powerful method for improving model reasoning capabilities. However, most methods either distill large state-of-the-art models into small students or use natural ground-truth problem statements to guarantee problem statement quality. This limits the scalability of these approaches to more complex and diverse problem domains. To address this, we present SPARQ: Synthetic Problem Generation for Reasoning via Quality-Diversity Algorithms, a novel approach for generating high-quality and diverse synthetic math problem and solution pairs using only a single model by measuring a problem's solve-rate: a proxy for problem difficulty. Starting from a seed dataset of 7.5K samples, we generate over 20 million new problem-solution pairs. We show that filtering the generated data by difficulty and then fine-tuning the same model on the resulting data improves relative model performance by up to 24\%. Additionally, we conduct ablations studying the impact of synthetic data quantity, quality and diversity on model generalization. We find that higher quality, as measured by problem difficulty, facilitates better in-distribution performance. Further, while generating diverse synthetic data does not as strongly benefit in-distribution performance, filtering for more diverse data facilitates more robust OOD generalization. We also confirm the existence of model and data scaling laws for synthetically generated problems, which positively benefit downstream model generalization. 

**Abstract (ZH)**: 基于高质量多样算法的合成问题生成以提升逻辑推理能力：SPARQ方法 

---
# What Is Seen Cannot Be Unseen: The Disruptive Effect of Knowledge Conflict on Large Language Models 

**Title (ZH)**: 看得见的不能被遗忘：知识冲突对大型语言模型的颠覆性影响 

**Authors**: Kaiser Sun, Fan Bai, Mark Dredze  

**Link**: [PDF](https://arxiv.org/pdf/2506.06485)  

**Abstract**: Large language models frequently rely on both contextual input and parametric knowledge to perform tasks. However, these sources can come into conflict, especially when retrieved documents contradict the model's parametric knowledge. We propose a diagnostic framework to systematically evaluate LLM behavior under context-memory conflict, where the contextual information diverges from their parametric beliefs. We construct diagnostic data that elicit these conflicts and analyze model performance across multiple task types. Our findings reveal that (1) knowledge conflict has minimal impact on tasks that do not require knowledge utilization, (2) model performance is consistently higher when contextual and parametric knowledge are aligned, (3) models are unable to fully suppress their internal knowledge even when instructed, and (4) providing rationales that explain the conflict increases reliance on contexts. These insights raise concerns about the validity of model-based evaluation and underscore the need to account for knowledge conflict in the deployment of LLMs. 

**Abstract (ZH)**: 大型语言模型经常依赖上下文输入和参数化知识来完成任务。然而，这些来源可能会产生冲突，尤其是在检索到的文档与模型的参数化知识相矛盾时。我们提出了一种诊断框架，以系统性地评估在上下文-记忆冲突状态下LLM的行为，其中上下文信息与参数信念相背离。我们构建了诊断数据以引发这些冲突，并分析了模型在多种任务类型中的性能。我们的研究表明：(1) 知识冲突对无需利用知识的任务影响甚微；(2) 当上下文和参数化知识一致时，模型的表现始终更高；(3) 即使在指令下，模型也无法完全抑制其内部知识；(4) 提供解释冲突的合理性陈述会增加对上下文的依赖。这些发现引发了对基于模型评估的有效性的担忧，并强调了在部署LLM时需要考虑知识冲突的重要性。 

---
# Cost-Efficient LLM Training with Lifetime-Aware Tensor Offloading via GPUDirect Storage 

**Title (ZH)**: 基于GPUDirect存储的生命周期意识张量卸载的高效LLM训练 

**Authors**: Ziqi Yuan, Haoyang Zhang, Yirui Eric Zhou, Apoorve Mohan, I-Hsin Chung, Seetharami Seelam, Jian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06472)  

**Abstract**: We present the design and implementation of a new lifetime-aware tensor offloading framework for GPU memory expansion using low-cost PCIe-based solid-state drives (SSDs). Our framework, TERAIO, is developed explicitly for large language model (LLM) training with multiple GPUs and multiple SSDs. Its design is driven by our observation that the active tensors take only a small fraction (1.7% on average) of allocated GPU memory in each LLM training iteration, the inactive tensors are usually large and will not be used for a long period of time, creating ample opportunities for offloading/prefetching tensors to/from slow SSDs without stalling the GPU training process. TERAIO accurately estimates the lifetime (active period of time in GPU memory) of each tensor with the profiling of the first few iterations in the training process. With the tensor lifetime analysis, TERAIO will generate an optimized tensor offloading/prefetching plan and integrate it into the compiled LLM program via PyTorch. TERAIO has a runtime tensor migration engine to execute the offloading/prefetching plan via GPUDirect storage, which allows direct tensor migration between GPUs and SSDs for alleviating the CPU bottleneck and maximizing the SSD bandwidth utilization. In comparison with state-of-the-art studies such as ZeRO-Offload and ZeRO-Infinity, we show that TERAIO improves the training performance of various LLMs by 1.47x on average, and achieves 80.7% of the ideal performance assuming unlimited GPU memory. 

**Abstract (ZH)**: 我们提出了一种新的基于低成本PCIe固态驱动器（SSD）的张量卸载框架，以实现GPU内存扩展，并关注张量的生命周期。该框架TERAIO专为多GPU和多SSD的大型语言模型（LLM）训练设计。其设计基于观察到每个LLM训练迭代中活跃张量仅占分配GPU内存的小部分（平均值为1.7%），不活跃张量通常较大且会闲置较长时间，这为将张量卸载/预取到/从缓慢的SSD上提供了充足的机会，而不影响GPU训练过程。TERAIO通过训练过程前几轮的分析，精确估计每个张量在GPU内存中的生命周期。基于张量生命周期分析，TERAIO生成一个优化的张量卸载/预取计划，并通过PyTorch集成到编译的LLM程序中。TERAIO具有运行时张量迁移引擎，通过GPUDirect存储执行卸载/预取计划，从而在GPU和SSD之间直接迁移张量，缓解CPU瓶颈并最大化SSD带宽利用率。与ZeRO-Offload和ZeRO-Infinity等先进研究相比，TERAIO将各种LLM的训练性能平均提高了1.47倍，并实现了理论上无限GPU内存性能的80.7%。 

---
# Canonical Autoregressive Generation 

**Title (ZH)**: 经典自回归生成 

**Authors**: Ivi Chatzi, Nina Corvelo Benz, Stratis Tsirtsis, Manuel Gomez-Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2506.06446)  

**Abstract**: State of the art large language models are trained using large amounts of tokens derived from raw text using what is called a tokenizer. Crucially, the tokenizer determines the (token) vocabulary a model will use during inference as well as, in principle, the (token) language. This is because, while the token vocabulary may allow for different tokenizations of a string, the tokenizer always maps the string to only one of these tokenizations--the canonical tokenization. However, multiple lines of empirical evidence suggest that large language models do not always generate canonical token sequences, and this comes with several negative consequences. In this work, we first show that, to generate a canonical token sequence, a model needs to generate (partial) canonical token sequences at each step of the autoregressive generation process underpinning its functioning. Building upon this theoretical result, we introduce canonical sampling, a simple and efficient sampling method that precludes a given model from generating non-canonical token sequences. Further, we also show that, in comparison with standard sampling, the distribution of token sequences generated using canonical sampling is provably closer to the true distribution of token sequences used during training. 

**Abstract (ZH)**: 最先进的大语言模型通过令牌化工具对大量原始文本 token 进行训练。至关重要的是，令牌化工具决定了模型在推理过程中使用的（令牌）词汇表以及原则上使用的（令牌）语言。因为虽然词汇表可以对字符串进行不同的分词，但令牌化工具总会将字符串映射到这些分词方式中的唯一一种——标准分词方式。然而，大量实证证据表明，大语言模型不总是生成标准的令牌序列，这带来了若干负面后果。在本文中，我们首先证明，为了生成标准的令牌序列，模型在自回归生成过程的每一步都需要生成部分标准的令牌序列。基于这一理论结果，我们提出了一种简单且高效的采样方法——标准采样，这种方法可以从给定模型中剔除生成非标准令牌序列的可能性。此外，我们还证明，与标准采样方法相比，使用标准采样方法生成的令牌序列的分布可以被证明更接近于训练过程中使用的令牌序列的真实分布。 

---
# Saffron-1: Towards an Inference Scaling Paradigm for LLM Safety Assurance 

**Title (ZH)**: 藏红花-1：通往大规模语言模型安全保证推理扩展范式之路 

**Authors**: Ruizhong Qiu, Gaotang Li, Tianxin Wei, Jingrui He, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2506.06444)  

**Abstract**: Existing safety assurance research has primarily focused on training-phase alignment to instill safe behaviors into LLMs. However, recent studies have exposed these methods' susceptibility to diverse jailbreak attacks. Concurrently, inference scaling has significantly advanced LLM reasoning capabilities but remains unexplored in the context of safety assurance. Addressing this gap, our work pioneers inference scaling for robust and effective LLM safety against emerging threats. We reveal that conventional inference scaling techniques, despite their success in reasoning tasks, perform poorly in safety contexts, even falling short of basic approaches like Best-of-N Sampling. We attribute this inefficiency to a newly identified challenge, the exploration--efficiency dilemma, arising from the high computational overhead associated with frequent process reward model (PRM) evaluations. To overcome this dilemma, we propose SAFFRON, a novel inference scaling paradigm tailored explicitly for safety assurance. Central to our approach is the introduction of a multifurcation reward model (MRM) that significantly reduces the required number of reward model evaluations. To operationalize this paradigm, we further propose: (i) a partial supervision training objective for MRM, (ii) a conservative exploration constraint to prevent out-of-distribution explorations, and (iii) a Trie-based key--value caching strategy that facilitates cache sharing across sequences during tree search. Extensive experiments validate the effectiveness of our method. Additionally, we publicly release our trained multifurcation reward model (Saffron-1) and the accompanying token-level safety reward dataset (Safety4M) to accelerate future research in LLM safety. Our code, model, and data are publicly available at this https URL , and our project homepage is at this https URL . 

**Abstract (ZH)**: 现有的安全保证研究主要侧重于训练阶段的对齐以灌输安全行为到大规模语言模型中。然而，最近的研究揭示了这些方法对多样化的模型突破攻击的脆弱性。同时，推理扩展显著提升了大规模语言模型的推理能力，但在安全保证的背景下尚待探索。为填补这一空白，我们的工作开创了针对新兴威胁的大规模语言模型安全的推理扩展方法，以实现稳健有效的安全防护。我们揭示，尽管传统推理扩展技术在推理任务中取得了成功，但在安全上下文中表现不佳，甚至不及Best-of-N采样等基本方法。我们归因于这一低效率是因为频繁的过程奖励模型评估带来的高计算开销引发的探索—效率困境。为克服这一困境，我们提出了SAFFRON，一种专门针对安全保证的创新性推理扩展范式。在我们的方法中，引入了一种多分叉奖励模型（MRM），显著减少了所需的过程奖励模型评估次数。为了实现这一范式的落实，我们进一步提出了：（i）对MRM的一种部分监督训练目标，（ii）保守探索约束以防止分布外探索，以及（iii）基于Trie的键—值缓存策略，这在树搜索过程中促进了序列间的缓存共享。广泛实验证明了我们方法的有效性。此外，我们公开发布了我们训练的多分叉奖励模型（Saffron-1）和配套的标记级安全奖励数据集（Safety4M），以加速大规模语言模型安全领域的未来研究。我们的代码、模型和数据可在以下链接获得：this https URL，项目主页位于以下链接：this https URL。 

---
# Benchmarking Misuse Mitigation Against Covert Adversaries 

**Title (ZH)**: 基于隐蔽对手的滥用缓解基准测试 

**Authors**: Davis Brown, Mahdi Sabbaghi, Luze Sun, Alexander Robey, George J. Pappas, Eric Wong, Hamed Hassani  

**Link**: [PDF](https://arxiv.org/pdf/2506.06414)  

**Abstract**: Existing language model safety evaluations focus on overt attacks and low-stakes tasks. Realistic attackers can subvert current safeguards by requesting help on small, benign-seeming tasks across many independent queries. Because individual queries do not appear harmful, the attack is hard to {detect}. However, when combined, these fragments uplift misuse by helping the attacker complete hard and dangerous tasks. Toward identifying defenses against such strategies, we develop Benchmarks for Stateful Defenses (BSD), a data generation pipeline that automates evaluations of covert attacks and corresponding defenses. Using this pipeline, we curate two new datasets that are consistently refused by frontier models and are too difficult for weaker open-weight models. Our evaluations indicate that decomposition attacks are effective misuse enablers, and highlight stateful defenses as a countermeasure. 

**Abstract (ZH)**: 现有的语言模型安全评估主要关注公开攻击和低风险任务。现实中的攻击者可以通过在多个独立查询中请求看似无害的小任务来规避当前的安全措施。由于个别查询看起来并不危险，这种攻击难以被检测。然而，当这些片段结合在一起时，它们能够辅助攻击者完成艰难且危险的任务。为了识别针对此类策略的防御措施，我们开发了状态依赖防御基准（BSD），这是一个自动评估隐蔽攻击及其相应防御的数据生成管道。利用这一管道，我们精心筛选了两个新的数据集，这些数据集超出了前沿模型的拒绝范围，并对较弱的开放权重模型来说也过于困难。我们的评估表明，分解攻击是有效的滥用辅助手段，并突显了状态依赖防御作为一种应对措施的重要性。 

---
# HeavyWater and SimplexWater: Watermarking Low-Entropy Text Distributions 

**Title (ZH)**: 重水与单纯形水：标记低熵文本分布 

**Authors**: Dor Tsur, Carol Xuan Long, Claudio Mayrink Verdun, Hsiang Hsu, Chen-Fu Chen, Haim Permuter, Sajani Vithana, Flavio P. Calmon  

**Link**: [PDF](https://arxiv.org/pdf/2506.06409)  

**Abstract**: Large language model (LLM) watermarks enable authentication of text provenance, curb misuse of machine-generated text, and promote trust in AI systems. Current watermarks operate by changing the next-token predictions output by an LLM. The updated (i.e., watermarked) predictions depend on random side information produced, for example, by hashing previously generated tokens. LLM watermarking is particularly challenging in low-entropy generation tasks - such as coding - where next-token predictions are near-deterministic. In this paper, we propose an optimization framework for watermark design. Our goal is to understand how to most effectively use random side information in order to maximize the likelihood of watermark detection and minimize the distortion of generated text. Our analysis informs the design of two new watermarks: HeavyWater and SimplexWater. Both watermarks are tunable, gracefully trading-off between detection accuracy and text distortion. They can also be applied to any LLM and are agnostic to side information generation. We examine the performance of HeavyWater and SimplexWater through several benchmarks, demonstrating that they can achieve high watermark detection accuracy with minimal compromise of text generation quality, particularly in the low-entropy regime. Our theoretical analysis also reveals surprising new connections between LLM watermarking and coding theory. The code implementation can be found in this https URL 

**Abstract (ZH)**: 大规模语言模型（LLM）水印使文本来源认证成为可能，遏制机器生成文本的滥用，并促进对AI系统的信任。当前的水印通过改变LLM输出的下一个 token 预测来工作。更新后的（即，带有水印的）预测依赖于通过哈希先前生成的 token 等方式产生的随机辅助信息。在低熵生成任务（如编码）中，下一个 token 的预测几乎是确定性的，使得 LLM 水印尤其具有挑战性。在本文中，我们提出了一种水印设计的优化框架。我们的目标是通过理解如何最有效地使用随机辅助信息来最大化水印检测的概率并最小化生成文本的失真。我们的分析指导设计了两种新的水印：HeavyWater 和 SimplexWater。这两种水印都是可调节的，能够在检测准确性和文本失真之间优雅地权衡。它们可以应用于任何 LLM，并且对辅助信息生成方式无偏好。我们通过多种基准测试研究了 HeavyWater 和 SimplexWater 的性能，证明了它们可以在最小牺牲生成文本质量的情况下实现高水印检测准确性，尤其是在低熵环境中。我们对水印技术的理论分析还揭示了与编码理论之间一些令人惊讶的新联系。代码实现可以在以下链接找到：https://xxxxxx 

---
# SMAR: Soft Modality-Aware Routing Strategy for MoE-based Multimodal Large Language Models Preserving Language Capabilities 

**Title (ZH)**: SMAR：软模态感知路由策略，用于保留语言能力的MoE-based多模态大型语言模型 

**Authors**: Guoyang Xia, Yifeng Ding, Fengfa Li, Lei Ren, Chen Wei, Fangxiang Feng, Xiaojie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06406)  

**Abstract**: Mixture of Experts (MoE) architectures have become a key approach for scaling large language models, with growing interest in extending them to multimodal tasks. Existing methods to build multimodal MoE models either incur high training costs or suffer from degraded language capabilities when adapting pretrained models. To address this, we propose Soft ModalityAware Routing (SMAR), a novel regularization technique that uses Kullback Leibler divergence to control routing probability distributions across modalities, encouraging expert specialization without modifying model architecture or heavily relying on textual data. Experiments on visual instruction tuning show that SMAR preserves language ability at 86.6% retention with only 2.5% pure text, outperforming baselines while maintaining strong multimodal performance. Our approach offers a practical and efficient solution to balance modality differentiation and language capabilities in multimodal MoE models. 

**Abstract (ZH)**: 混合专家模型的软模态感知路由（SMAR）：一种促进多模态大型语言模型模态分化和语言能力平衡的新正则化技术 

---
# Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights 

**Title (ZH)**: 价值对齐的大语言模型的意外危害：心理与实证洞察 

**Authors**: Sooyung Choi, Jaehyeok Lee, Xiaoyuan Yi, Jing Yao, Xing Xie, JinYeong Bak  

**Link**: [PDF](https://arxiv.org/pdf/2506.06404)  

**Abstract**: The application scope of Large Language Models (LLMs) continues to expand, leading to increasing interest in personalized LLMs that align with human values. However, aligning these models with individual values raises significant safety concerns, as certain values may correlate with harmful information. In this paper, we identify specific safety risks associated with value-aligned LLMs and investigate the psychological principles behind these challenges. Our findings reveal two key insights. (1) Value-aligned LLMs are more prone to harmful behavior compared to non-fine-tuned models and exhibit slightly higher risks in traditional safety evaluations than other fine-tuned models. (2) These safety issues arise because value-aligned LLMs genuinely generate text according to the aligned values, which can amplify harmful outcomes. Using a dataset with detailed safety categories, we find significant correlations between value alignment and safety risks, supported by psychological hypotheses. This study offers insights into the "black box" of value alignment and proposes in-context alignment methods to enhance the safety of value-aligned LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）的安全风险及个性化价值对齐原则研究 

---
# Direct Behavior Optimization: Unlocking the Potential of Lightweight LLMs 

**Title (ZH)**: 直接行为优化：轻量级LLM潜力的释放 

**Authors**: Hongming Yang, Shi Lin, Jun Shao, Changting Lin, Donghai Zhu, Meng Han, Qinglei Kong  

**Link**: [PDF](https://arxiv.org/pdf/2506.06401)  

**Abstract**: Lightweight Large Language Models (LwLLMs) are reduced-parameter, optimized models designed to run efficiently on consumer-grade hardware, offering significant advantages in resource efficiency, cost-effectiveness, and data privacy. However, these models often struggle with limited inference and reasoning capabilities, which restrict their performance on complex tasks and limit their practical applicability. Moreover, existing prompt optimization methods typically rely on extensive manual effort or the meta-cognitive abilities of state-of-the-art LLMs, making them less effective for LwLLMs. To address these challenges, we introduce DeBoP, a new Direct Behavior Optimization Paradigm, original from the Chain-of-Thought (CoT) prompting technique. Unlike CoT Prompting, DeBoP is an automatic optimization method, which focuses on the optimization directly on the behavior of LwLLMs. In particular, DeBoP transforms the optimization of complex prompts into the optimization of discrete, quantifiable execution sequences using a gradient-free Monte Carlo Tree Search. We evaluate DeBoP on seven challenging tasks where state-of-the-art LLMs excel but LwLLMs generally underperform. Experimental results demonstrate that DeBoP significantly outperforms recent prompt optimization methods on most tasks. In particular, DeBoP-optimized LwLLMs surpass GPT-3.5 on most tasks while reducing computational time by approximately 60% compared to other automatic prompt optimization methods. 

**Abstract (ZH)**: 轻量级大型语言模型（LwLLMs）是参数减少、优化后的模型，旨在高效运行在消费级硬件上，提供了在资源效率、成本效益和数据隐私方面的显著优势。然而，这些模型往往在推理和推理能力上存在局限，这限制了其在复杂任务上的表现，并限制了其实用性。此外，现有的提示优化方法通常依赖大量的手动努力或最先进的LwLLMs的元认知能力，这对LwLLMs来说效果较差。为了解决这些问题，我们提出了DeBoP，一种新的直接行为优化范式，源自Chain-of-Thought（CoT）提示技术。与CoT提示不同，DeBoP是一种自动优化方法，专注于直接优化LwLLMs的行为。特别是，DeBoP将复杂提示的优化转换为使用无导数蒙特卡洛树搜索优化离散的、可量化执行序列。我们在七个挑战性任务上评估了DeBoP，这些任务是目前最先进的LLM可以表现出色但LwLLMs通常表现较差的。实验结果表明，DeBoP在大多数任务上显著优于最近的提示优化方法。特别是，DeBoP优化的LwLLMs在大多数任务上超越了GPT-3.5，与其它自动提示优化方法相比，计算时间减少了约60%。 

---
# Natural Language Interaction with Databases on Edge Devices in the Internet of Battlefield Things 

**Title (ZH)**: 战场物联网边缘设备上的自然语言数据库交互 

**Authors**: Christopher D. Molek, Roberto Fronteddu, K. Brent Venable, Niranjan Suri  

**Link**: [PDF](https://arxiv.org/pdf/2506.06396)  

**Abstract**: The expansion of the Internet of Things (IoT) in the battlefield, Internet of Battlefield Things (IoBT), gives rise to new opportunities for enhancing situational awareness. To increase the potential of IoBT for situational awareness in critical decision making, the data from these devices must be processed into consumer-ready information objects, and made available to consumers on demand. To address this challenge we propose a workflow that makes use of natural language processing (NLP) to query a database technology and return a response in natural language. Our solution utilizes Large Language Models (LLMs) that are sized for edge devices to perform NLP as well as graphical databases which are well suited for dynamic connected networks which are pervasive in the IoBT. Our architecture employs LLMs for both mapping questions in natural language to Cypher database queries as well as to summarize the database output back to the user in natural language. We evaluate several medium sized LLMs for both of these tasks on a database representing publicly available data from the US Army's Multipurpose Sensing Area (MSA) at the Jornada Range in Las Cruces, NM. We observe that Llama 3.1 (8 billion parameters) outperforms the other models across all the considered metrics. Most importantly, we note that, unlike current methods, our two step approach allows the relaxation of the Exact Match (EM) requirement of the produced Cypher queries with ground truth code and, in this way, it achieves a 19.4% increase in accuracy. Our workflow lays the ground work for deploying LLMs on edge devices to enable natural language interactions with databases containing information objects for critical decision making. 

**Abstract (ZH)**: 战场物联网（IoBT）的扩展为增强态势感知提供了新机遇。为了提高IoBT在关键决策中的潜力，这些设备的数据必须被处理成消费者可用的信息对象，并按需提供给消费者。为了解决这一挑战，我们提出了一种工作流，利用自然语言处理（NLP）查询数据库技术并以自然语言返回响应。我们的解决方案利用了适用于边缘设备的大型语言模型（LLMs）来进行NLP，以及适用于动态连接网络的图形数据库，这些网络在IoBT中普遍存在。我们的架构使用LLMs将自然语言中的问题映射为Cypher数据库查询，并将数据库输出总结回自然语言给用户。我们在代表美国陆军多用途传感区域（MSA）公开数据的数据库上评估了几种中型LLMs，这些数据来自新墨西哥州拉斯克鲁塞斯市的Jornada训练场。我们发现，Llama 3.1（80亿参数）在所有考虑的指标中表现最佳。最重要的是，我们注意到，与当前方法不同，我们两步方法允许放宽生成的Cypher查询与地面真实代码的精确匹配要求，从而在准确性上提高了19.4%。我们的工作流为在边缘设备上部署LLMs以实现与包含信息对象的数据库的自然语言交互奠定了基础。 

---
# From Rogue to Safe AI: The Role of Explicit Refusals in Aligning LLMs with International Humanitarian Law 

**Title (ZH)**: 从违规到安全的人工 intelligence：明示拒绝在将大型语言模型与国际人道法对齐中的作用 

**Authors**: John Mavi, Diana Teodora Găitan, Sergio Coronado  

**Link**: [PDF](https://arxiv.org/pdf/2506.06391)  

**Abstract**: Large Language Models (LLMs) are widely used across sectors, yet their alignment with International Humanitarian Law (IHL) is not well understood. This study evaluates eight leading LLMs on their ability to refuse prompts that explicitly violate these legal frameworks, focusing also on helpfulness - how clearly and constructively refusals are communicated. While most models rejected unlawful requests, the clarity and consistency of their responses varied. By revealing the model's rationale and referencing relevant legal or safety principles, explanatory refusals clarify the system's boundaries, reduce ambiguity, and help prevent misuse. A standardised system-level safety prompt significantly improved the quality of the explanations expressed within refusals in most models, highlighting the effectiveness of lightweight interventions. However, more complex prompts involving technical language or requests for code revealed ongoing vulnerabilities. These findings contribute to the development of safer, more transparent AI systems and propose a benchmark to evaluate the compliance of LLM with IHL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各行业的广泛应用尚存，但其与国际人道法（IHL）的契合程度尚不明确。本研究评价了八种主流LLM在其拒绝明确违反这些法律框架的提示能力上的表现，并重点考察了这些模型的建议性——即拒绝理由的清晰度和建设性。虽然大多数模型拒绝了非法请求，但其回应的清晰度和一致性存在差异。通过揭示模型的推理过程并参考相关法律或安全原则，解释性的拒绝能够阐明系统的边界，减少模糊性，并有助于防止滥用。标准化的系统级安全提示在大多数模型中显著提高了拒绝理由表达的质量，突显了轻量级干预措施的有效性。然而，涉及技术语言或代码请求的更复杂提示揭示了持续存在的漏洞。这些发现为开发更安全和更具透明度的人工智能系统作出了贡献，并提出了评估LLM与IHL合规性的基准。 

---
# Benchmarking Large Language Models on Homework Assessment in Circuit Analysis 

**Title (ZH)**: 大型语言模型在电路分析作业评估中的基准测试 

**Authors**: Liangliang Chen, Zhihao Qin, Yiming Guo, Jacqueline Rohde, Ying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06390)  

**Abstract**: Large language models (LLMs) have the potential to revolutionize various fields, including code development, robotics, finance, and education, due to their extensive prior knowledge and rapid advancements. This paper investigates how LLMs can be leveraged in engineering education. Specifically, we benchmark the capabilities of different LLMs, including GPT-3.5 Turbo, GPT-4o, and Llama 3 70B, in assessing homework for an undergraduate-level circuit analysis course. We have developed a novel dataset consisting of official reference solutions and real student solutions to problems from various topics in circuit analysis. To overcome the limitations of image recognition in current state-of-the-art LLMs, the solutions in the dataset are converted to LaTeX format. Using this dataset, a prompt template is designed to test five metrics of student solutions: completeness, method, final answer, arithmetic error, and units. The results show that GPT-4o and Llama 3 70B perform significantly better than GPT-3.5 Turbo across all five metrics, with GPT-4o and Llama 3 70B each having distinct advantages in different evaluation aspects. Additionally, we present insights into the limitations of current LLMs in several aspects of circuit analysis. Given the paramount importance of ensuring reliability in LLM-generated homework assessment to avoid misleading students, our results establish benchmarks and offer valuable insights for the development of a reliable, personalized tutor for circuit analysis -- a focus of our future work. Furthermore, the proposed evaluation methods can be generalized to a broader range of courses for engineering education in the future. 

**Abstract (ZH)**: 大规模语言模型（LLMs）有可能革新包括代码开发、机器人学、金融和教育在内的多个领域，得益于其广泛的知识储备和快速的技术进步。本文探讨了LLMs在工程教育中的应用。具体来说，我们比较了GPT-3.5 Turbo、GPT-4o和Llama 3 70B等不同LLMs评估本科生电路分析课程作业的能力。我们开发了一个新的数据集，其中包括官方参考解决方案和实际的学生解决方案，涵盖电路分析中的各种问题。为克服当前先进LLMs在图像识别方面的局限，数据集中的解决方案被转换为LaTeX格式。利用该数据集，我们设计了一个提示模板，以测试学生解决方案的五个指标：完整性、方法、最终答案、算术错误和单位。结果表明，GPT-4o和Llama 3 70B在所有五个指标上均显著优于GPT-3.5 Turbo，GPT-4o和Llama 3 70B在不同的评估方面各有优势。此外，我们还探讨了当前LLMs在电路分析方面的一些局限性。鉴于确保LLM生成的作业评估的可靠性对于避免误导学生至关重要，我们的结果建立了基准并为开发一个可靠且个性化的电路分析辅导系统提供了有价值的见解——这是我们未来工作的一个重点。此外，所提出的方法可以进一步应用于工程教育中更广泛的课程。 

---
# Detection Method for Prompt Injection by Integrating Pre-trained Model and Heuristic Feature Engineering 

**Title (ZH)**: 基于预训练模型和启发式特征工程的提示注入检测方法 

**Authors**: Yi Ji, Runzhi Li, Baolei Mao  

**Link**: [PDF](https://arxiv.org/pdf/2506.06384)  

**Abstract**: With the widespread adoption of Large Language Models (LLMs), prompt injection attacks have emerged as a significant security threat. Existing defense mechanisms often face critical trade-offs between effectiveness and generalizability. This highlights the urgent need for efficient prompt injection detection methods that are applicable across a wide range of LLMs. To address this challenge, we propose DMPI-PMHFE, a dual-channel feature fusion detection framework. It integrates a pretrained language model with heuristic feature engineering to detect prompt injection attacks. Specifically, the framework employs DeBERTa-v3-base as a feature extractor to transform input text into semantic vectors enriched with contextual information. In parallel, we design heuristic rules based on known attack patterns to extract explicit structural features commonly observed in attacks. Features from both channels are subsequently fused and passed through a fully connected neural network to produce the final prediction. This dual-channel approach mitigates the limitations of relying only on DeBERTa to extract features. Experimental results on diverse benchmark datasets demonstrate that DMPI-PMHFE outperforms existing methods in terms of accuracy, recall, and F1-score. Furthermore, when deployed actually, it significantly reduces attack success rates across mainstream LLMs, including GLM-4, LLaMA 3, Qwen 2.5, and GPT-4o. 

**Abstract (ZH)**: 大规模语言模型中基于提示注入攻击的有效检测方法：DMPI-PMHFE框架 

---
# On the Fundamental Impossibility of Hallucination Control in Large Language Models 

**Title (ZH)**: 在大型语言模型中幻觉控制的基本不可能性 

**Authors**: Michał P. Karpowicz  

**Link**: [PDF](https://arxiv.org/pdf/2506.06382)  

**Abstract**: This paper explains \textbf{why it is impossible to create large language models that do not hallucinate and what are the trade-offs we should be looking for}. It presents a formal \textbf{impossibility theorem} demonstrating that no inference mechanism can simultaneously satisfy four fundamental properties: \textbf{truthful (non-hallucinatory) generation, semantic information conservation, relevant knowledge revelation, and knowledge-constrained optimality}. By modeling LLM inference as an \textbf{auction of ideas} where neural components compete to contribute to responses, we prove the impossibility using the Green-Laffont theorem. That mathematical framework provides a rigorous foundation for understanding the nature of inference process, with implications for model architecture, training objectives, and evaluation methods. 

**Abstract (ZH)**: 本文解释了为什么无法创建不会产生幻觉的大规模语言模型，并探讨了我们应寻找的权衡。它提出了一个形式化的不可能性定理，证明没有任何推理机制能同时满足四项基本属性：真实的（非幻觉）生成、语义信息保真、相关知识揭示以及知识约束下的最优性。通过将LLM推理建模为一个神经组件竞标想法的拍卖过程，我们利用Green-Laffont定理证明了这一不可能性。该数学框架为理解推理过程的本质提供了严格的基礎，对模型架构、训练目标和评估方法具有重要意义。 

---
# Enhancing Decision-Making of Large Language Models via Actor-Critic 

**Title (ZH)**: 通过.actor-critic方法增强大型语言模型的决策制定 

**Authors**: Heng Dong, Kefei Duan, Chongjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06376)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable advancements in natural language processing tasks, yet they encounter challenges in complex decision-making scenarios that require long-term reasoning and alignment with high-level objectives. Existing methods either rely on short-term auto-regressive action generation or face limitations in accurately simulating rollouts and assessing outcomes, leading to sub-optimal decisions. This paper introduces a novel LLM-based Actor-Critic framework, termed LAC, that effectively improves LLM policies with long-term action evaluations in a principled and scalable way. Our approach addresses two key challenges: (1) extracting robust action evaluations by computing Q-values via token logits associated with positive/negative outcomes, enhanced by future trajectory rollouts and reasoning; and (2) enabling efficient policy improvement through a gradient-free mechanism. Experiments across diverse environments -- including high-level decision-making (ALFWorld), low-level action spaces (BabyAI-Text), and large action spaces (WebShop) -- demonstrate the framework's generality and superiority over state-of-the-art methods. Notably, our approach achieves competitive performance using 7B/8B parameter LLMs, even outperforming baseline methods employing GPT-4 in complex tasks. These results underscore the potential of integrating structured policy optimization with LLMs' intrinsic knowledge to advance decision-making capabilities in multi-step environments. 

**Abstract (ZH)**: 基于大型语言模型的演员-批评家框架：LAC在长期决策评估中的优化与应用 

---
# From Transformers to Large Language Models: A systematic review of AI applications in the energy sector towards Agentic Digital Twins 

**Title (ZH)**: 从 Transformers 到大规模语言模型：能源领域代理数字孪生的 AI 应用综述 

**Authors**: Gabriel Antonesi, Tudor Cioara, Ionut Anghel, Vasilis Michalakopoulos, Elissaios Sarmas, Liana Toderean  

**Link**: [PDF](https://arxiv.org/pdf/2506.06359)  

**Abstract**: Artificial intelligence (AI) has long promised to improve energy management in smart grids by enhancing situational awareness and supporting more effective decision-making. While traditional machine learning has demonstrated notable results in forecasting and optimization, it often struggles with generalization, situational awareness, and heterogeneous data integration. Recent advances in foundation models such as Transformer architecture and Large Language Models (LLMs) have demonstrated improved capabilities in modelling complex temporal and contextual relationships, as well as in multi-modal data fusion which is essential for most AI applications in the energy sector. In this review we synthesize the rapid expanding field of AI applications in the energy domain focusing on Transformers and LLMs. We examine the architectural foundations, domain-specific adaptations and practical implementations of transformer models across various forecasting and grid management tasks. We then explore the emerging role of LLMs in the field: adaptation and fine tuning for the energy sector, the type of tasks they are suited for, and the new challenges they introduce. Along the way, we highlight practical implementations, innovations, and areas where the research frontier is rapidly expanding. These recent developments reviewed underscore a broader trend: Generative AI (GenAI) is beginning to augment decision-making not only in high-level planning but also in day-to-day operations, from forecasting and grid balancing to workforce training and asset onboarding. Building on these developments, we introduce the concept of the Agentic Digital Twin, a next-generation model that integrates LLMs to bring autonomy, proactivity, and social interaction into digital twin-based energy management systems. 

**Abstract (ZH)**: 人工智能（AI）长期以来一直致力于通过增强情境意识和支持更为有效的决策来改进智能电网的能源管理。尽管传统的机器学习在预测和优化方面取得了显著成果，但在泛化、情境意识和异构数据集成方面仍面临挑战。最近，基础模型如变换器架构和大型语言模型（LLMs）的进步展示了在建模复杂的时间和上下文关系以及多模态数据融合方面的增强能力，这对于能源领域的大多数AI应用至关重要。在本文综述中，我们总结了人工智能在能源领域应用的迅速扩展领域，重点关注变换器和LLMs。我们分析了变换器模型在各种预测和电网管理任务中的架构基础、领域特定适应和实际应用。然后探索了LLMs在该领域中的新兴作用：针对能源领域的适应和微调、适合的任务类型以及它们引入的新挑战。在此过程中，我们强调了实际应用、创新和研究前沿迅速扩展的领域。这些综述中的近期发展表明一种更广泛的趋势：生成式AI（GenAI）不仅在高层次规划中，也在日常运营中（从预测和电网平衡到人员培训和资产入职）开始增强决策制定。在这些发展的基础上，我们提出了代理数字 twin的概念，这是一种下一代模型，通过集成LLMs来将自主性、主动性和社会互动引入基于数字孪生的能源管理系统中。 

---
# Large Language Models for EEG: A Comprehensive Survey and Taxonomy 

**Title (ZH)**: EEG领域的大型语言模型：一项全面的综述与分类 

**Authors**: Naseem Babu, Jimson Mathew, A. P. Vinod  

**Link**: [PDF](https://arxiv.org/pdf/2506.06353)  

**Abstract**: The growing convergence between Large Language Models (LLMs) and electroencephalography (EEG) research is enabling new directions in neural decoding, brain-computer interfaces (BCIs), and affective computing. This survey offers a systematic review and structured taxonomy of recent advancements that utilize LLMs for EEG-based analysis and applications. We organize the literature into four domains: (1) LLM-inspired foundation models for EEG representation learning, (2) EEG-to-language decoding, (3) cross-modal generation including image and 3D object synthesis, and (4) clinical applications and dataset management tools. The survey highlights how transformer-based architectures adapted through fine-tuning, few-shot, and zero-shot learning have enabled EEG-based models to perform complex tasks such as natural language generation, semantic interpretation, and diagnostic assistance. By offering a structured overview of modeling strategies, system designs, and application areas, this work serves as a foundational resource for future work to bridge natural language processing and neural signal analysis through language models. 

**Abstract (ZH)**: 大型语言模型（LLMs）与脑电图（EEG）研究的日益 convergence 为神经解码、脑-机接口（BCIs）和情感计算提供了新的发展方向。本文综述了利用LLMs进行EEG基础模型、EEG到语言解码、跨模态生成以及临床应用和数据管理工具的近期进展。综述强调了通过微调、少样本学习和零样本学习适应的变换器架构如何使EEG模型能够执行复杂任务，如自然语言生成、语义解释和诊断辅助。通过提供建模策略、系统设计和应用领域的结构化概述，本文为未来通过语言模型弥合自然语言处理与神经信号分析之间的差距奠定了基础。 

---
# Unified Game Moderation: Soft-Prompting and LLM-Assisted Label Transfer for Resource-Efficient Toxicity Detection 

**Title (ZH)**: 统一游戏内容管理：基于软提示和大语言模型辅助的标签迁移方法以实现资源高效的内容毒性检测 

**Authors**: Zachary Yang, Domenico Tullo, Reihaneh Rabbany  

**Link**: [PDF](https://arxiv.org/pdf/2506.06347)  

**Abstract**: Toxicity detection in gaming communities faces significant scaling challenges when expanding across multiple games and languages, particularly in real-time environments where computational efficiency is crucial. We present two key findings to address these challenges while building upon our previous work on ToxBuster, a BERT-based real-time toxicity detection system. First, we introduce a soft-prompting approach that enables a single model to effectively handle multiple games by incorporating game-context tokens, matching the performance of more complex methods like curriculum learning while offering superior scalability. Second, we develop an LLM-assisted label transfer framework using GPT-4o-mini to extend support to seven additional languages. Evaluations on real game chat data across French, German, Portuguese, and Russian achieve macro F1-scores ranging from 32.96% to 58.88%, with particularly strong performance in German, surpassing the English benchmark of 45.39%. In production, this unified approach significantly reduces computational resources and maintenance overhead compared to maintaining separate models for each game and language combination. At Ubisoft, this model successfully identifies an average of 50 players, per game, per day engaging in sanctionable behavior. 

**Abstract (ZH)**: 多游戏多语言环境中的实时毒性检测扩展面临显著的规模挑战，特别是在需要高效计算的实时环境中。我们介绍了两种关键发现，以解决这些挑战并建立在我们之前的工作ToxBuster——一个基于BERT的实时毒性检测系统的基础之上。首先，我们提出了一种软提示方法，通过引入游戏上下文标记，使单个模型能够有效处理多个游戏，其性能与 Curriculum 学习等复杂方法相当，同时具有更强的扩展性。其次，我们开发了一种由GPT-4o-mini辅助的大语言模型辅助标签转移框架，以支持七种额外语言。在涵盖法语、德语、葡萄牙语和俄语的真实游戏聊天数据上的评估达到了32.96%至58.88%的宏F1分数，特别是在德语上的表现尤为突出，超过了45.39%的英语基准。在生产环境中，这种统一方法相比为每个游戏和语言组合维护单独模型，显著降低了计算资源和维护开销。在育碧，该模型每天平均能够识别出每个游戏中有50名执行可处罚行为的玩家。 

---
# TESU-LLM: Training Speech-LLMs Without Speech via Unified Encoder Alignment 

**Title (ZH)**: TESU-LLM：通过统一编码对齐训练无语音的语音LLM 

**Authors**: Taesoo Kim, Jong Hwan Ko  

**Link**: [PDF](https://arxiv.org/pdf/2506.06343)  

**Abstract**: Recent advances in speech-enabled language models have shown promising results in building intelligent voice assistants. However, most existing approaches rely on large-scale paired speech-text data and extensive computational resources, which pose challenges in terms of scalability and accessibility. In this paper, we present \textbf{TESU-LLM}, a novel framework that enables training speech-capable language models using only text data. Our key insight is to leverage a unified encoder that maps semantically equivalent text and speech inputs to a shared latent space. By aligning the encoder output with the embedding space of a LLM via a lightweight projection network, we enable the model to generalize from text-only supervision to speech-based inference. Despite being trained exclusively on text, TESU-LLM achieves strong performance on various speech-related benchmarks, comparable to baseline methods trained with large-scale multimodal datasets and substantial computational resources. These results highlight the effectiveness and efficiency of our approach, offering a scalable path toward building speech LLMs without speech data. 

**Abstract (ZH)**: Recent Advances in Training Speech-Capable Language Models Using Only Text Data 

---
# Structured Semantics from Unstructured Notes: Language Model Approaches to EHR-Based Decision Support 

**Title (ZH)**: 从非结构化笔记中提取结构化语义：基于EHR的语言模型决策支持方法 

**Authors**: Wu Hao Ran, Xi Xi, Furong Li, Jingyi Lu, Jian Jiang, Hui Huang, Yuzhuan Zhang, Shi Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06340)  

**Abstract**: The advent of large language models (LLMs) has opened new avenues for analyzing complex, unstructured data, particularly within the medical domain. Electronic Health Records (EHRs) contain a wealth of information in various formats, including free text clinical notes, structured lab results, and diagnostic codes. This paper explores the application of advanced language models to leverage these diverse data sources for improved clinical decision support. We will discuss how text-based features, often overlooked in traditional high dimensional EHR analysis, can provide semantically rich representations and aid in harmonizing data across different institutions. Furthermore, we delve into the challenges and opportunities of incorporating medical codes and ensuring the generalizability and fairness of AI models in healthcare. 

**Abstract (ZH)**: 大型语言模型的兴起为分析医疗领域的复杂非结构化数据开辟了新途径。电子健康记录（EHRs）包含多种形式的丰富信息，包括自由文本临床笔记、结构化实验室结果和诊断代码。本文探讨了高级语言模型在利用这些多样化的数据来源以提高临床决策支持方面的应用。我们将讨论文本特征在传统高维EHR分析中常被忽视的优势，这些特征能够提供丰富的语义表示，并有助于跨机构的数据协调。此外，本文还将探讨整合医疗代码以及确保医疗健康领域中AI模型的普适性和公平性的挑战和机遇。 

---
# Optimizing RAG Pipelines for Arabic: A Systematic Analysis of Core Components 

**Title (ZH)**: 优化阿拉伯语RAG管道：核心组件的系统分析 

**Authors**: Jumana Alsubhi, Mohammad D. Alahmadi, Ahmed Alhusayni, Ibrahim Aldailami, Israa Hamdine, Ahmad Shabana, Yazeed Iskandar, Suhayb Khayyat  

**Link**: [PDF](https://arxiv.org/pdf/2506.06339)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful architecture for combining the precision of retrieval systems with the fluency of large language models. While several studies have investigated RAG pipelines for high-resource languages, the optimization of RAG components for Arabic remains underexplored. This study presents a comprehensive empirical evaluation of state-of-the-art RAG components-including chunking strategies, embedding models, rerankers, and language models-across a diverse set of Arabic datasets. Using the RAGAS framework, we systematically compare performance across four core metrics: context precision, context recall, answer faithfulness, and answer relevancy. Our experiments demonstrate that sentence-aware chunking outperforms all other segmentation methods, while BGE-M3 and Multilingual-E5-large emerge as the most effective embedding models. The inclusion of a reranker (bge-reranker-v2-m3) significantly boosts faithfulness in complex datasets, and Aya-8B surpasses StableLM in generation quality. These findings provide critical insights for building high-quality Arabic RAG pipelines and offer practical guidelines for selecting optimal components across different document types. 

**Abstract (ZH)**: 检索增强生成（RAG）架构已成为将检索系统的精准度与大型语言模型的流畅性相结合的一种强大工具。尽管已有研究调查了高资源语言的RAG管道，但阿拉伯语RAG组件的优化仍缺乏探讨。本研究提供了对先进的RAG组件（包括分块策略、嵌入模型、重排器和语言模型）在多样化阿拉伯语数据集上的全面 empirical 评估。通过使用 RAGAS 框架，我们系统地在四个核心指标：上下文精确度、上下文召回率、答案忠实度和答案相关性上进行了性能比较。实验结果表明，句意识分块优于所有其他分段方法，而 BGE-M3 和 Multilingual-E5-large 是最有效的嵌入模型。引入重排器（bge-reranker-v2-m3）在复杂数据集中的显著提高了忠实度，Aya-8B 在生成质量上超过了 StableLM。这些发现为构建高质量的阿拉伯语 RAG 管道提供了关键见解，并提供了根据不同文档类型选择最佳组件的实用指南。 

---
# FinBERT2: A Specialized Bidirectional Encoder for Bridging the Gap in Finance-Specific Deployment of Large Language Models 

**Title (ZH)**: FinBERT2：专为金融领域大型语言模型部署差距桥接设计的专用双向编码器 

**Authors**: Xuan Xu, Fufang Wen, Beilin Chu, Zhibing Fu, Qinhong Lin, Jiaqi Liu, Binjie Fei, Zhongliang Yang, Linna Zhou, Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06335)  

**Abstract**: In natural language processing (NLP), the focus has shifted from encoder-only tiny language models like BERT to decoder-only large language models(LLMs) such as GPT-3. However, LLMs' practical application in the financial sector has revealed three limitations: (1) LLMs often perform worse than fine-tuned BERT on discriminative tasks despite costing much higher computational resources, such as market sentiment analysis in financial reports; (2) Application on generative tasks heavily relies on retrieval augmented generation (RAG) methods to provide current and specialized information, with general retrievers showing suboptimal performance on domain-specific retrieval tasks; (3) There are additional inadequacies in other feature-based scenarios, such as topic modeling. We introduce FinBERT2, a specialized bidirectional encoder pretrained on a high-quality, financial-specific corpus of 32b tokens. This represents the largest known Chinese financial pretraining corpus for models of this parameter size. As a better backbone, FinBERT2 can bridge the gap in the financial-specific deployment of LLMs through the following achievements: (1) Discriminative fine-tuned models (Fin-Labelers) outperform other (Fin)BERT variants by 0.4%-3.3% and leading LLMs by 9.7%-12.3% on average across five financial classification tasks. (2) Contrastive fine-tuned models (Fin-Retrievers) outperform both open-source (e.g., +6.8\% avg improvement over BGE-base-zh) and proprietary (e.g., +4.2\% avg improvement over OpenAI's text-embedding-3-large) embedders across five financial retrieval tasks; (3) Building on FinBERT2 variants, we construct the Fin-TopicModel, which enables superior clustering and topic representation for financial titles. Our work revisits financial BERT models through comparative analysis with contemporary LLMs and offers practical insights for effectively utilizing FinBERT in the LLMs era. 

**Abstract (ZH)**: 在自然语言处理（NLP）中，焦点已从如BERT这样的编码器型小型语言模型转向如GPT-3这样的解码器型大型语言模型（LLMs）。然而，LLMs在金融领域的实际应用揭示了三个局限性：（1）尽管在诸如市场情绪分析等辨别任务中成本远远高于微调的BERT，LLMs的表现通常逊于后者；（2）在生成任务上，LLMs的实现严重依赖于检索增强生成（RAG）方法，通用检索器在特定领域检索任务上表现不佳；（3）在其他基于特征的场景中还有额外不足，例如主题建模。我们介绍了FinBERT2，这是一种预训练在高质量320亿令牌的金融专用语料库上的双向编码器。这是已知的参数量最多、用于此类规模模型的中国金融预训练语料库。作为更好的主体模型，FinBERT2可以通过以下成就来弥补LLMs在金融专用部署中的不足：（1）辨别微调模型（Fin-Labelers）在五个金融分类任务中比其他（Fin）BERT变体高出0.4%-3.3%，平均比领先LLMs高出9.7%-12.3%；（2）对比微调模型（Fin-Retrievers）在五个金融检索任务中优于开源（例如，BGE-base-zh平均改善6.8%）和专有模型（例如，OpenAI的text-embedding-3-large平均改善4.2%）；（3）基于FinBERT2的变体，我们构建了Fin-TopicModel，使得金融标题的聚类和主题表示更加出色。我们的研究通过与当代LLMs的比较分析重访了金融BERT模型，并为有效地利用FinBERT在LLMs时代提供了实用见解。 

---
# How Significant Are the Real Performance Gains? An Unbiased Evaluation Framework for GraphRAG 

**Title (ZH)**: GraphRAG的真实性能提升有多显著？一个无偏评价框架 

**Authors**: Qiming Zeng, Xiao Yan, Hao Luo, Yuhao Lin, Yuxiang Wang, Fangcheng Fu, Bo Du, Quanqing Xu, Jiawei Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06331)  

**Abstract**: By retrieving contexts from knowledge graphs, graph-based retrieval-augmented generation (GraphRAG) enhances large language models (LLMs) to generate quality answers for user questions. Many GraphRAG methods have been proposed and reported inspiring performance in answer quality. However, we observe that the current answer evaluation framework for GraphRAG has two critical flaws, i.e., unrelated questions and evaluation biases, which may lead to biased or even wrong conclusions on performance. To tackle the two flaws, we propose an unbiased evaluation framework that uses graph-text-grounded question generation to produce questions that are more related to the underlying dataset and an unbiased evaluation procedure to eliminate the biases in LLM-based answer assessment. We apply our unbiased framework to evaluate 3 representative GraphRAG methods and find that their performance gains are much more moderate than reported previously. Although our evaluation framework may still have flaws, it calls for scientific evaluations to lay solid foundations for GraphRAG research. 

**Abstract (ZH)**: 基于图的检索增强生成（GraphRAG）通过从知识图谱中检索上下文增强大型语言模型（LLMs），以生成高质量的答案。尽管提出了许多GraphRAG方法并报告了令人鼓舞的答案质量性能，但我们观察到当前的答案评估框架存在两个关键缺陷，即无关问题和评估偏见，这可能会导致性能评估的有偏或甚至错误结论。为解决这两个问题，我们提出了一种无偏评估框架，该框架使用图-文本-数据集 grounding 问题生成方法生成与基础数据集更相关的問題，并提出了一种无偏的评估程序以消除基于LLM的答案评估中的偏见。我们将我们的无偏框架应用于评估3种代表性的GraphRAG方法，并发现它们的性能提升远不及之前报道的那样显著。尽管我们的评估框架仍可能存在缺陷，但它呼吁进行科学评估，为GraphRAG研究奠定坚实基础。 

---
# Reward Is Enough: LLMs Are In-Context Reinforcement Learners 

**Title (ZH)**: 奖励足矣：LLMs是基于上下文的强化学习者 

**Authors**: Kefan Song, Amir Moeini, Peng Wang, Lei Gong, Rohan Chandra, Yanjun Qi, Shangtong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06303)  

**Abstract**: Reinforcement learning (RL) is a human-designed framework for solving sequential decision making problems. In this work, we demonstrate that, surprisingly, RL emerges in LLM's (Large Language Model) inference time -- a phenomenon known as in-context RL (ICRL). Specifically, we propose a novel multi-round prompting framework called ICRL prompting. The goal is to prompt the LLM to complete a task. After the LLM generates a response at the current round, we give numerical scalar feedbacks for the response, called the rewards. At the next round, we prompt the LLM again with the same task and a context consisting of all previous responses and rewards. We observe that the quality of the LLM's response increases as the context grows. In other words, the LLM is able to maximize the scalar reward signal in the inference time, just like an RL algorithm. We evaluate ICRL prompting in three benchmarks (Game of 24, creative writing, and ScienceWorld) and demonstrate significant performance improvements over baseline methods such as Self-Refine and Reflexion. Surprisingly, in some experiments the reward signals are generated by the LLM itself, yet performance improvements are still observed from ICRL prompting, offering a promising paradigm for scaling test-time compute. 

**Abstract (ZH)**: reinforced learning 在大语言模型推理时间的出现：基于上下文的强化学习 (ICRL) 推导框架 

---
# dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching 

**Title (ZH)**: dLLM-Cache：基于自适应缓存加速扩散大语言模型 

**Authors**: Zhiyuan Liu, Yicun Yang, Yaojie Zhang, Junjie Chen, Chang Zou, Qingyuan Wei, Shaobo Wang, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06295)  

**Abstract**: Autoregressive Models (ARMs) have long dominated the landscape of Large Language Models. Recently, a new paradigm has emerged in the form of diffusion-based Large Language Models (dLLMs), which generate text by iteratively denoising masked segments. This approach has shown significant advantages and potential. However, dLLMs suffer from high inference latency. Traditional ARM acceleration techniques, such as Key-Value caching, are incompatible with dLLMs due to their bidirectional attention mechanism. To address this specific challenge, our work begins with a key observation that dLLM inference involves a static prompt and a partially dynamic response, where most tokens remain stable across adjacent denoising steps. Based on this, we propose dLLM-Cache, a training-free adaptive caching framework that combines long-interval prompt caching with partial response updates guided by feature similarity. This design enables efficient reuse of intermediate computations without compromising model performance. Extensive experiments on representative dLLMs, including LLaDA 8B and Dream 7B, show that dLLM-Cache achieves up to 9.1 x speedup over standard inference without compromising output quality. Notably, our method brings dLLM inference latency close to that of ARMs under many settings. Codes are provided in the supplementary material and will be released publicly on GitHub. 

**Abstract (ZH)**: 自动回归模型（ARMs）长期主导着大型语言模型的领域。最近，基于扩散的大规模语言模型（dLLMs）崭露头角，通过迭代去噪掩蔽片段生成文本。这种方法显示了显著的优势和潜力，但dLLMs存在推理延迟高的问题。传统ARMs的加速技术，如Key-Value缓存，由于其双向注意力机制与dLLMs不兼容。为解决这一特定挑战，我们工作基于一个关键观察，即dLLM推理涉及静态提示和部分动态响应，在相邻去噪步骤中大多数令牌保持稳定。基于此，我们提出了一种无需训练的自适应缓存框架dLLM-Cache，该框架结合了长间隔提示缓存和由特征相似性引导的部分响应更新。此设计使得能够高效重用中间计算而不影响模型性能。在包括LLaDA 8B和Dream 7B在内的代表性dLLMs上的 extensive 实验显示，dLLM-Cache在不牺牲输出质量的情况下实现了高达9.1倍的速度提升。值得注意的是，我们的方法在许多情况下将dLLM推理延迟接近ARMs。相关代码已包含在补充材料中，并将在GitHub上公开发布。 

---
# Mutual-Taught for Co-adapting Policy and Reward Models 

**Title (ZH)**: 协同教学以共适应策略和奖励模型 

**Authors**: Tianyuan Shi, Canbin Huang, Fanqi Wan, Longguang Zhong, Ziyi Yang, Weizhou Shen, Xiaojun Quan, Ming Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.06292)  

**Abstract**: During the preference optimization of large language models (LLMs), distribution shifts may arise between newly generated model samples and the data used to train the reward model (RM). This shift reduces the efficacy of the RM, which in turn negatively impacts the performance of the policy model (PM). To address this challenge, we propose Mutual-Taught, a self-training method that iteratively improves both the PM and RM without requiring additional human annotation. Our approach mirrors the expectation-maximization (EM) algorithm. In the E-step, the PM is updated using feedback from the current RM, guiding the PM toward a better approximation of the latent optimal preference distribution. In the M-step, we update the RM by constructing training data from the outputs of the PM before and after the E-step update. This process ensures that the RM adapts to the evolving policy distribution. Experimental results demonstrate that this iterative approach leads to consistent improvements in both models. Specifically, our 8B policy model, LLaMA-3-8B-Instruct-MT, achieves a length-controlled win rate of 54.1\% on AlpacaEval-2, while our 8B reward model, FsfairX-LLaMA3-RM-MT, performs on par with GPT-4o-2024-08-06 on RewardBench. 

**Abstract (ZH)**: 在大规模语言模型的偏好优化过程中， Newly generated模型样本与用于训练奖励模型的数据之间的分布变化可能会导致奖励模型（RM）的效能下降，进而负面影响策略模型（PM）的表现。为应对这一挑战，我们提出了一种名为Mutual-Taught的自训练方法，该方法能够迭代地提高PM和RM的性能而不增加额外的人工标注。我们的方法借鉴了期望最大化（EM）算法。在E步中，使用当前RM的反馈更新PM，引导PM更好地逼近潜在的最优偏好分布。在M步中，通过使用E步更新前后PM的输出构建训练数据来更新RM，从而确保RM能够适应不断变化的策略分布。实验结果表明，这种迭代方法可以持续改进两个模型。具体而言，我们的8B策略模型LLaMA-3-8B-Instruct-MT在AlpacaEval-2上的可控长度胜率达到了54.1%，而我们的8B奖励模型FsfairX-LLaMA3-RM-MT在RewardBench上的表现与GPT-4o-2024-08-06相当。 

---
# GOLFer: Smaller LM-Generated Documents Hallucination Filter & Combiner for Query Expansion in Information Retrieval 

**Title (ZH)**: GOLFer：用于信息检索中查询扩展的小型LM生成文档幻觉过滤器与组合器 

**Authors**: Lingyuan Liu, Mengxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04762)  

**Abstract**: Large language models (LLMs)-based query expansion for information retrieval augments queries with generated hypothetical documents with LLMs. However, its performance relies heavily on the scale of the language models (LMs), necessitating larger, more advanced LLMs. This approach is costly, computationally intensive, and often has limited accessibility. To address these limitations, we introduce GOLFer - Smaller LMs-Generated Documents Hallucination Filter & Combiner - a novel method leveraging smaller open-source LMs for query expansion. GOLFer comprises two modules: a hallucination filter and a documents combiner. The former detects and removes non-factual and inconsistent sentences in generated documents, a common issue with smaller LMs, while the latter combines the filtered content with the query using a weight vector to balance their influence. We evaluate GOLFer alongside dominant LLM-based query expansion methods on three web search and ten low-resource datasets. Experimental results demonstrate that GOLFer consistently outperforms other methods using smaller LMs, and maintains competitive performance against methods using large-size LLMs, demonstrating its effectiveness. 

**Abstract (ZH)**: 基于较小语言模型的生成文档 hallucination 过滤与组合的查询扩展方法：GOLFer 

---
# Low-resource Machine Translation: what for? who for? An observational study on a dedicated Tetun language translation service 

**Title (ZH)**: 低资源机器翻译：为了谁？针对什么？一种专门的图分子翻译服务的观察研究 

**Authors**: Raphael Merx, Adérito José Guterres Correia, Hanna Suominen, Ekaterina Vylomova  

**Link**: [PDF](https://arxiv.org/pdf/2411.12262)  

**Abstract**: Low-resource machine translation (MT) presents a diversity of community needs and application challenges that remain poorly understood. To complement surveys and focus groups, which tend to rely on small samples of respondents, we propose an observational study on actual usage patterns of tetun$.$org, a specialized MT service for the Tetun language, which is the lingua franca in Timor-Leste. Our analysis of 100,000 translation requests reveals patterns that challenge assumptions based on existing corpora. We find that users, many of them students on mobile devices, typically translate text from a high-resource language into Tetun across diverse domains including science, healthcare, and daily life. This contrasts sharply with available Tetun corpora, which are dominated by news articles covering government and social issues. Our results suggest that MT systems for institutionalized minority languages like Tetun should prioritize accuracy on domains relevant to educational contexts, in the high-resource to low-resource direction. More broadly, this study demonstrates how observational analysis can inform low-resource language technology development, by grounding research in practical community needs. 

**Abstract (ZH)**: 低资源机器翻译：观察tetun.org专门服务的实际使用模式及其对低资源语言技术发展的启示 

---
# MiniGPT-Reverse-Designing: Predicting Image Adjustments Utilizing MiniGPT-4 

**Title (ZH)**: MiniGPT-Reverse-设计：利用MiniGPT-4预测图像调整 

**Authors**: Vahid Azizi, Fatemeh Koochaki  

**Link**: [PDF](https://arxiv.org/pdf/2406.00971)  

**Abstract**: Vision-Language Models (VLMs) have recently seen significant advancements through integrating with Large Language Models (LLMs). The VLMs, which process image and text modalities simultaneously, have demonstrated the ability to learn and understand the interaction between images and texts across various multi-modal tasks. Reverse designing, which could be defined as a complex vision-language task, aims to predict the edits and their parameters, given a source image, an edited version, and an optional high-level textual edit description. This task requires VLMs to comprehend the interplay between the source image, the edited version, and the optional textual context simultaneously, going beyond traditional vision-language tasks. In this paper, we extend and fine-tune MiniGPT-4 for the reverse designing task. Our experiments demonstrate the extensibility of off-the-shelf VLMs, specifically MiniGPT-4, for more complex tasks such as reverse designing. Code is available at this \href{this https URL} 

**Abstract (ZH)**: Vision-Language模型（VLMs）通过与大规模语言模型（LLMs）的结合， recently saw显著进步。VLMs同时处理图像和文本模态，展示了在各种多模态任务中学习和理解图像与文本之间交互的能力。逆设计，可定义为一项复杂的视觉-语言任务，旨在给定源图像、编辑版本以及可选的高层次文本编辑描述的情况下，预测编辑和参数。这一任务要求VLMs同时理解源图像、编辑版本和可选的文本上下文之间的交互，超越了传统的视觉-语言任务。在本文中，我们扩展并微调了MiniGPT-4用于逆设计任务。我们的实验展示了现成的VLMs，特别是MiniGPT-4，适用于更复杂的任务如逆设计。代码请参见this https URL。 

---
