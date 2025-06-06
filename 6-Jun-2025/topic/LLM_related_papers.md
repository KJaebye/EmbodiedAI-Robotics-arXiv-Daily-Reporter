# A Framework Leveraging Large Language Models for Autonomous UAV Control in Flying Networks 

**Title (ZH)**: 一种利用大规模语言模型实现飞行网络中自主无人机控制的框架 

**Authors**: Diana Nunes, Ricardo Amorim, Pedro Ribeiro, André Coelho, Rui Campos  

**Link**: [PDF](https://arxiv.org/pdf/2506.04404)  

**Abstract**: This paper proposes FLUC, a modular framework that integrates open-source Large Language Models (LLMs) with Unmanned Aerial Vehicle (UAV) autopilot systems to enable autonomous control in Flying Networks (FNs). FLUC translates high-level natural language commands into executable UAV mission code, bridging the gap between operator intent and UAV behaviour.
FLUC is evaluated using three open-source LLMs - Qwen 2.5, Gemma 2, and LLaMA 3.2 - across scenarios involving code generation and mission planning. Results show that Qwen 2.5 excels in multi-step reasoning, Gemma 2 balances accuracy and latency, and LLaMA 3.2 offers faster responses with lower logical coherence. A case study on energy-aware UAV positioning confirms FLUC's ability to interpret structured prompts and autonomously execute domain-specific logic, showing its effectiveness in real-time, mission-driven control. 

**Abstract (ZH)**: 本文提出FLUC，这是一种模块化框架，将开源大规模语言模型（LLMs）与无人机（UAV）自主飞行控制系统集成，以实现飞行网络（FNs）的自主控制。FLUC将高级自然语言指令转换为可执行的无人机任务代码，填补了操作员意图与无人机行为之间的差距。 

---
# LLM-First Search: Self-Guided Exploration of the Solution Space 

**Title (ZH)**: LLM-First Search: 自主引导的解空间探索 

**Authors**: Nathan Herr, Tim Rocktäschel, Roberta Raileanu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05213)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable improvements in reasoning and planning through increased test-time compute, often by framing problem-solving as a search process. While methods like Monte Carlo Tree Search (MCTS) have proven effective in some domains, their reliance on fixed exploration hyperparameters limits their adaptability across tasks of varying difficulty, rendering them impractical or expensive in certain settings. In this paper, we propose \textbf{LLM-First Search (LFS)}, a novel \textit{LLM Self-Guided Search} method that removes the need for pre-defined search strategies by empowering the LLM to autonomously control the search process via self-guided exploration. Rather than relying on external heuristics or hardcoded policies, the LLM evaluates whether to pursue the current search path or explore alternative branches based on its internal scoring mechanisms. This enables more flexible and context-sensitive reasoning without requiring manual tuning or task-specific adaptation. We evaluate LFS on Countdown and Sudoku against three classic widely-used search algorithms, Tree-of-Thoughts' Breadth First Search (ToT-BFS), Best First Search (BestFS), and MCTS, each of which have been used to achieve SotA results on a range of challenging reasoning tasks. We found that LFS (1) performs better on more challenging tasks without additional tuning, (2) is more computationally efficient compared to the other methods, especially when powered by a stronger model, (3) scales better with stronger models, due to its LLM-First design, and (4) scales better with increased compute budget. Our code is publicly available at \href{this https URL}{LLM-First-Search}. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过增加测试时计算资源在推理和规划方面展现出显著的改进，通常将问题求解视为一个搜索过程。虽然像蒙特卡洛树搜索（MCTS）这样的方法在某些领域已被证明是有效的，但它们依赖于固定探索超参数的特点限制了它们在不同难度任务中的适应性，使得它们在某些情况下不实用或成本高昂。在本文中，我们提出了\textbf{LLM-First Search (LFS)}，这是一种新颖的\textit{LLM 自主搜索}方法，通过赋予LLM自主控制搜索过程的能力，从而无需预定义的搜索策略。LLM根据其内部评分机制决定是否继续当前搜索路径或探索其他分支，从而实现更灵活和上下文敏感的推理，而无需手动调优或特定任务的适应。我们通过Countdown和Sudoku对LFS进行了评估，与Tree-of-Thoughts的广度优先搜索（ToT-BFS）、最佳优先搜索（BestFS）和MCTS等三种经典广泛应用的搜索算法进行了比较，这些算法已被用于实现一系列具有挑战性的推理任务的当前最佳结果。我们发现，LFS（1）在更具挑战性的任务上表现更优，无需额外调优；（2）与其他方法相比，更具计算效率，尤其是在使用更强的模型时；（3）由于其LLM-First设计，在更强的模型上表现出更好的可扩展性；（4）随着计算预算的增加，在更强的模型上表现出更好的可扩展性。我们的代码在\href{this https URL}{LLM-First-Search}公开可用。 

---
# Mathematical Reasoning for Unmanned Aerial Vehicles: A RAG-Based Approach for Complex Arithmetic Reasoning 

**Title (ZH)**: 无人机数学推理：基于RAG的复杂算术推理方法 

**Authors**: Mehdi Azarafza, Mojtaba Nayyeri, Faezeh Pasandideh, Steffen Staab, Achim Rettberg  

**Link**: [PDF](https://arxiv.org/pdf/2506.04998)  

**Abstract**: Autonomous UAV operation necessitates reliable mathematical reasoning for tasks such as trajectory planning and power management. While traditional flight control relies on hardcoded equations, recent Large Language Models (LLMs) offer potential for more flexible problem-solving but struggle with reliably selecting and applying correct mathematical formulations and executing precise multi-step arithmetic. We propose RAG-UAV, a retrieval-augmented generation framework designed to improve the mathematical reasoning of several LLMs (including GPT o1/Turbo, Llama-3.2/3.3, Mistral, and DeepSeek R1) in UAV-specific contexts by providing access to relevant domain literature. To conduct an initial assessment, we introduce the UAV-Math-Bench, a small problem set comprising 20 UAV-centric mathematical problems across four difficulty levels. Our experiments demonstrate that incorporating retrieval substantially increases exact answer accuracy (achieving up to 75% with o1), reduces instances of incorrect formulation selection (from 25% without RAG to 5% with RAG), decreases numerical errors, reducing Mean Squared Error (MSE) by orders of magnitude for the best-performing models. This pilot study indicates that RAG can enable general-purpose LLMs to function as more reliable tools for engineering analysis, although direct real-time flight control requires further investigation and validation on a larger scale. All benchmark data, question and answer are publicly available. 

**Abstract (ZH)**: 自主无人机操作需要可靠的数学推理能力，以完成轨迹规划和电力管理等任务。虽然传统飞行控制依赖于硬编码的方程，但最近的大语言模型（LLMs）在更具灵活性的问题求解方面具有潜力，但在可靠选择和应用正确的数学公式以及执行精确的多步计算方面存在困难。我们提出了一种名为RAG-UAV的检索增强生成框架，旨在通过提供相关领域文献访问权限来提升几种LLM（包括GPT o1/Turbo、Llama-3.2/3.3、Mistral和DeepSeek R1）在无人机特定上下文中的数学推理能力。为了进行初步评估，我们引入了UAV-Math-Bench，这是一个包含20个跨四个难度级别的无人机中心数学问题的小型问题集。实验结果表明，引入检索显著提高了准确答案的准确率（最高达到75%），减少了错误公式选择的实例（从未使用RAG的25%降至使用RAG的5%），降低了数值误差，使最佳模型的均方误差（MSE）减少了几个数量级。这项初步研究表明，RAG可以使通用大语言模型更可靠地用于工程分析，尽管直接实时飞行控制仍需在更大规模下进行进一步调查和验证。所有基准数据、问题和答案均已公开。 

---
# When Thinking LLMs Lie: Unveiling the Strategic Deception in Representations of Reasoning Models 

**Title (ZH)**: 当LLMs欺骗他人：揭示推理模型表示中的战略欺骗 

**Authors**: Kai Wang, Yihao Zhang, Meng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.04909)  

**Abstract**: The honesty of large language models (LLMs) is a critical alignment challenge, especially as advanced systems with chain-of-thought (CoT) reasoning may strategically deceive humans. Unlike traditional honesty issues on LLMs, which could be possibly explained as some kind of hallucination, those models' explicit thought paths enable us to study strategic deception--goal-driven, intentional misinformation where reasoning contradicts outputs. Using representation engineering, we systematically induce, detect, and control such deception in CoT-enabled LLMs, extracting "deception vectors" via Linear Artificial Tomography (LAT) for 89% detection accuracy. Through activation steering, we achieve a 40% success rate in eliciting context-appropriate deception without explicit prompts, unveiling the specific honesty-related issue of reasoning models and providing tools for trustworthy AI alignment. 

**Abstract (ZH)**: 大型语言模型的诚实性是一个关键的对齐挑战，尤其是当先进系统具备链式思考（CoT）推理能力时，可能战略性地欺骗人类。 

---
# Evaluation is All You Need: Strategic Overclaiming of LLM Reasoning Capabilities Through Evaluation Design 

**Title (ZH)**: 只需要评估：通过评估设计战略性夸大LLM推理能力 

**Authors**: Lin Sun, Weihong Lin, Jinzhu Wu, Yongfu Zhu, Xiaoqi Jian, Guangxiang Zhao, Change Jia, Linglin Zhang, Sai-er Hu, Yuhan Wu, Xiangzheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04734)  

**Abstract**: Reasoning models represented by the Deepseek-R1-Distill series have been widely adopted by the open-source community due to their strong performance in mathematics, science, programming, and other domains. However, our study reveals that their benchmark evaluation results are subject to significant fluctuations caused by various factors. Subtle differences in evaluation conditions can lead to substantial variations in results. Similar phenomena are observed in other open-source inference models fine-tuned based on the Deepseek-R1-Distill series, as well as in the QwQ-32B model, making their claimed performance improvements difficult to reproduce reliably. Therefore, we advocate for the establishment of a more rigorous paradigm for model performance evaluation and present our empirical assessments of the Deepseek-R1-Distill series models. 

**Abstract (ZH)**: Deepseek-R1-Distill系列推理模型因其在数学、科学、编程等领域表现出色而被开源社区广泛采用。然而，我们的研究表明，这些模型的基准评估结果因各种因素的影响存在显著波动。评估条件的细微差异会导致结果的显著变化。类似现象在基于Deepseek-R1-Distill系列 fine-tuned 的其他开源推理模型以及QwQ-32B模型中也观察到，这使得其声称的性能改进难以可靠地复现。因此，我们倡导建立更严格的模型性能评估范式，并呈现对Deepseek-R1-Distill系列模型的实证评估。 

---
# Beyond Accuracy: Dissecting Mathematical Reasoning for LLMs Under Reinforcement Learning 

**Title (ZH)**: 超越准确率：在强化学习下分解数学推理能力 

**Authors**: Jiayu Wang, Yifei Ming, Zixuan Ke, Caiming Xiong, Shafiq Joty, Aws Albarghouthi, Frederic Sala  

**Link**: [PDF](https://arxiv.org/pdf/2506.04723)  

**Abstract**: Reinforcement learning (RL) has become the dominant paradigm for endowing language models with advanced reasoning capabilities. Despite the substantial empirical gains demonstrated by RL-based training methods like GRPO, a granular understanding of their advantages is still lacking. To address this gap, we introduce a fine-grained analytic framework to dissect the impact of RL on reasoning. Our framework specifically investigates key elements that have been hypothesized to benefit from RL training: (1) plan-following and execution, (2) problem decomposition, and (3) improved reasoning and knowledge utilization. Using this framework, we gain insights beyond mere accuracy. For instance, providing models with explicit step-by-step plans surprisingly degrades performance on the most challenging benchmarks, yet RL-tuned models exhibit greater robustness, experiencing markedly smaller performance drops than their base counterparts. This suggests that RL may not primarily enhance the execution of external plans but rather empower models to formulate and follow internal strategies better suited to their reasoning processes. Conversely, we observe that RL enhances the model's capacity to integrate provided knowledge into its reasoning process, leading to performance improvements across diverse tasks. We also study difficulty, showing improved training by developing new ways to exploit hard problems. Our findings lay a foundation for more principled training and evaluation of reasoning models. 

**Abstract (ZH)**: 强化学习（RL）已成为赋予语言模型高级推理能力的主要范式。尽管基于RL的训练方法如GRPO展示了显著的经验收益，但对其优势的细致理解仍然不足。为填补这一空白，我们引入了一种细致的分析框架来剖析RL对推理的影响。该框架具体探讨了已被假设可以从RL训练中受益的关键元素：（1）计划遵循与执行，（2）问题分解，以及（3）改进的推理与知识利用。利用该框架，我们获得了超越单纯准确性的洞见。例如，为模型提供明确的逐步计划出人意料地降低了最具有挑战性的基准上的性能，但RL调优模型表现出更大的鲁棒性，其性能下降幅度远小于基模型。这表明，RL可能主要不是增强对外部计划的执行，而是使模型能够更好地制定和遵循更适合其推理过程的内部策略。相反，我们观察到，RL增强了模型将提供的知识整合到其推理过程中的能力，从而在多种任务上取得了性能提升。我们还研究了难度问题，通过开发新的方法来利用难题提高了训练效果。我们的发现为更原则性的推理模型训练和评估奠定了基础。 

---
# Empowering Economic Simulation for Massively Multiplayer Online Games through Generative Agent-Based Modeling 

**Title (ZH)**: 通过生成性基于代理的建模为大型多人在线游戏赋能经济模拟 

**Authors**: Bihan Xu, Shiwei Zhao, Runze Wu, Zhenya Huang, Jiawei Wang, Zhipeng Hu, Kai Wang, Haoyu Liu, Tangjie Lv, Le Li, Changjie Fan, Xin Tong, Jiangze Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.04699)  

**Abstract**: Within the domain of Massively Multiplayer Online (MMO) economy research, Agent-Based Modeling (ABM) has emerged as a robust tool for analyzing game economics, evolving from rule-based agents to decision-making agents enhanced by reinforcement learning. Nevertheless, existing works encounter significant challenges when attempting to emulate human-like economic activities among agents, particularly regarding agent reliability, sociability, and interpretability. In this study, we take a preliminary step in introducing a novel approach using Large Language Models (LLMs) in MMO economy simulation. Leveraging LLMs' role-playing proficiency, generative capacity, and reasoning aptitude, we design LLM-driven agents with human-like decision-making and adaptability. These agents are equipped with the abilities of role-playing, perception, memory, and reasoning, addressing the aforementioned challenges effectively. Simulation experiments focusing on in-game economic activities demonstrate that LLM-empowered agents can promote emergent phenomena like role specialization and price fluctuations in line with market rules. 

**Abstract (ZH)**: 使用大型语言模型在MMO经济模拟中引入具有类人决策能力和适应性的代理方法研究 

---
# E-bike agents: Large Language Model-Driven E-Bike Accident Analysis and Severity Prediction 

**Title (ZH)**: 基于大型语言模型的电动自行车事故分析与严重性预测 

**Authors**: Zhichao Yang, Jiashu He, Mohammad B. Al-Khasawneh, Darshan Pandit, Cirillo Cinzia  

**Link**: [PDF](https://arxiv.org/pdf/2506.04654)  

**Abstract**: Electric bicycles (e-bikes) are rapidly increasing in use, raising safety concerns due to a rise in accident reports. However, e-bike incident reports often use unstructured narrative formats, which hinders quantitative safety analysis. This study introduces E-bike agents, a framework that uses large language models (LLM) powered agents to classify and extract safety variables from unstructured incident reports. Our framework consists of four LLM agents, handling data classification, information extraction, injury cause determination, and component linkage, to extract the key factors that could lead to E-bike accidents and cause varying severity levels. Furthermore, we used an ordered logit model to examine the relationship between the severity of the incident and the factors retrieved, such as gender, the type of cause, and environmental conditions. Our research shows that equipment issues are slightly more common than human-related ones, but human-related incidents are more often fatal. Specifically, pedals, tires, and brakes are frequent contributors to accidents. The model achieves a high weighted F1 score of 0.87 in classification accuracy, highlighting the potential of using LLMs to extract unstructured data in niche domains, such as transportation. Our method offers a scalable solution to improve e-bike safety analytics and provides actionable information for policy makers, designers, and regulators. 

**Abstract (ZH)**: 电助力自行车(e-bike)的使用迅速增加，由于事故报告增多而引起安全关注。然而，e-bike事故报告往往使用非结构化的叙述格式，这阻碍了定量安全分析。本文介绍了一种E-bike代理框架，该框架利用大语言模型（LLM）驱动的代理对非结构化的事故报告进行分类和提取安全变量。该框架包括四个LLM代理，分别处理数据分类、信息提取、伤害原因确定和组件关联，以提取可能导致e-bike事故和不同严重程度的关键因素。此外，我们使用有序logit模型分析事故严重程度与提取因素之间的关系，如性别、原因类型和环境条件。研究发现，设备问题比人为问题略多，但人为事故更致命。具体来说，脚踏、轮胎和刹车是事故频发的原因。模型在分类准确性上获得较高加权F1分数0.87，突显了在交通等专业领域使用LLM提取非结构化数据的潜力。本文方法提供了可扩展的解决方案，以提高e-bike安全分析质量，并为决策者、设计师和监管机构提供可操作信息。 

---
# Agents of Change: Self-Evolving LLM Agents for Strategic Planning 

**Title (ZH)**: 变革的推动者：自演进的大语言模型代理在战略规划中的应用 

**Authors**: Nikolas Belle, Dakota Barnes, Alfonso Amayuelas, Ivan Bercovich, Xin Eric Wang, William Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04651)  

**Abstract**: Recent advances in LLMs have enabled their use as autonomous agents across a range of tasks, yet they continue to struggle with formulating and adhering to coherent long-term strategies. In this paper, we investigate whether LLM agents can self-improve when placed in environments that explicitly challenge their strategic planning abilities. Using the board game Settlers of Catan, accessed through the open-source Catanatron framework, we benchmark a progression of LLM-based agents, from a simple game-playing agent to systems capable of autonomously rewriting their own prompts and their player agent's code. We introduce a multi-agent architecture in which specialized roles (Analyzer, Researcher, Coder, and Player) collaborate to iteratively analyze gameplay, research new strategies, and modify the agent's logic or prompt. By comparing manually crafted agents to those evolved entirely by LLMs, we evaluate how effectively these systems can diagnose failure and adapt over time. Our results show that self-evolving agents, particularly when powered by models like Claude 3.7 and GPT-4o, outperform static baselines by autonomously adopting their strategies, passing along sample behavior to game-playing agents, and demonstrating adaptive reasoning over multiple iterations. 

**Abstract (ZH)**: 近期大规模语言模型的发展使它们能够在多种任务中作为自主代理使用，但仍难以形成和遵守连贯的长期策略。本文探讨将语言模型代理置于明确挑战其战略规划能力的环境中时，它们能否自我提升。通过使用开源Catanatron框架访问《,settlers of Catan》这一棋盘游戏，我们对一系列基于语言模型的代理进行了基准测试，从简单的游戏代理到能够自主重写自己提示和玩家代理代码的系统。我们引入了一种多代理架构，其中专门的角色（分析员、研究员、程序员和玩家）协作迭代地分析游戏玩法、研究新策略并修改代理的逻辑或提示。通过对手工构建的代理与完全由语言模型进化出的代理进行比较，我们评估了这些系统诊断失败和随着时间迭代适应的能力。结果显示，特别是由 Claude 3.7 和 GPT-4o 等模型驱动的自进化代理优于静态基线，能够自主采用策略、传递示例行为给游戏代理，并在多次迭代中展示适应性推理。 

---
# Schema Generation for Large Knowledge Graphs Using Large Language Models 

**Title (ZH)**: 使用大规模语言模型大规模知识图谱的模式生成 

**Authors**: Bohui Zhang, Yuan He, Lydia Pintscher, Albert Meroño Peñuela, Elena Simperl  

**Link**: [PDF](https://arxiv.org/pdf/2506.04512)  

**Abstract**: Schemas are vital for ensuring data quality in the Semantic Web and natural language processing. Traditionally, their creation demands substantial involvement from knowledge engineers and domain experts. Leveraging the impressive capabilities of large language models (LLMs) in related tasks like ontology engineering, we explore automatic schema generation using LLMs. To bridge the resource gap, we introduce two datasets: YAGO Schema and Wikidata EntitySchema, along with evaluation metrics. The LLM-based pipelines effectively utilize local and global information from knowledge graphs (KGs) to generate validating schemas in Shape Expressions (ShEx). Experiments demonstrate LLMs' strong potential in producing high-quality ShEx schemas, paving the way for scalable, automated schema generation for large KGs. Furthermore, our benchmark introduces a new challenge for structured generation, pushing the limits of LLMs on syntactically rich formalisms. 

**Abstract (ZH)**: 基于大型语言模型的Schema自动生成对于语义网和自然语言处理中的数据质量至关重要。传统上，Schema的创建需要知识工程师和领域专家的大量参与。通过利用大型语言模型在onto工程等相关任务中的卓越能力，我们探索了使用大型语言模型进行自动Schema生成。为了弥合资源差距，我们引入了两个数据集：YAGO Schema和Wikidata EntitySchema，以及评价指标。基于大型语言模型的流水线有效利用知识图谱的局部和全局信息，生成Shape Expressions (ShEx)形式的验证Schema。实验表明，大型语言模型在生成高质量ShEx Schema方面具有巨大潜力，为大规模知识图谱的可扩展、自动化Schema生成铺平了道路。此外，我们的基准测试引入了结构化生成的新挑战，推动了大型语言模型在语法丰富形式化语言上的能力极限。 

---
# CogMath: Assessing LLMs' Authentic Mathematical Ability from a Human Cognitive Perspective 

**Title (ZH)**: CogMath: 从人类认知视角评估LLM的真正数学能力 

**Authors**: Jiayu Liu, Zhenya Huang, Wei Dai, Cheng Cheng, Jinze Wu, Jing Sha, Song Li, Qi Liu, Shijin Wang, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.04481)  

**Abstract**: Although large language models (LLMs) show promise in solving complex mathematical tasks, existing evaluation paradigms rely solely on a coarse measure of overall answer accuracy, which are insufficient for assessing their authentic capabilities. In this paper, we propose \textbf{CogMath}, which comprehensively assesses LLMs' mathematical abilities through the lens of human cognition. Specifically, inspired by psychological theories, CogMath formalizes human reasoning process into 3 stages: \emph{problem comprehension}, \emph{problem solving}, and \emph{solution summarization}. Within these stages, we investigate perspectives such as numerical calculation, knowledge, and counterfactuals, and design a total of 9 fine-grained evaluation dimensions. In each dimension, we develop an ``\emph{Inquiry}-\emph{Judge}-\emph{Reference}'' multi-agent system to generate inquiries that assess LLMs' mastery from this dimension. An LLM is considered to truly master a problem only when excelling in all inquiries from the 9 dimensions. By applying CogMath on three benchmarks, we reveal that the mathematical capabilities of 7 mainstream LLMs are overestimated by 30\%-40\%. Moreover, we locate their strengths and weaknesses across specific stages/dimensions, offering in-depth insights to further enhance their reasoning abilities. 

**Abstract (ZH)**: CogMath：通过人类认知视角全面评估大规模语言模型的数学能力 

---
# Matching Markets Meet LLMs: Algorithmic Reasoning with Ranked Preferences 

**Title (ZH)**: 匹配市场遇上LLMs：基于排名偏好算法推理 

**Authors**: Hadi Hosseini, Samarth Khanna, Ronak Singh  

**Link**: [PDF](https://arxiv.org/pdf/2506.04478)  

**Abstract**: The rise of Large Language Models (LLMs) has driven progress in reasoning tasks -- from program synthesis to scientific hypothesis generation -- yet their ability to handle ranked preferences and structured algorithms in combinatorial domains remains underexplored. We study matching markets, a core framework behind applications like resource allocation and ride-sharing, which require reconciling individual ranked preferences to ensure stable outcomes. We evaluate several state-of-the-art models on a hierarchy of preference-based reasoning tasks -- ranging from stable-matching generation to instability detection, instability resolution, and fine-grained preference queries -- to systematically expose their logical and algorithmic limitations in handling ranked inputs. Surprisingly, even top-performing models with advanced reasoning struggle to resolve instability in large markets, often failing to identify blocking pairs or execute algorithms iteratively. We further show that parameter-efficient fine-tuning (LoRA) significantly improves performance in small markets, but fails to bring about a similar improvement on large instances, suggesting the need for more sophisticated strategies to improve LLMs' reasoning with larger-context inputs. 

**Abstract (ZH)**: 大型语言模型（LLMs）的兴起推动了推理任务的进步——从程序合成到科学假说生成——然而它们在处理组合域中的排序偏好和结构化算法方面的能力仍待探索。我们研究了匹配市场这一核心框架，该框架应用于资源分配和汽车共享等领域，需要综合考虑个体的排序偏好以确保结果的稳定性。我们评估了几种最先进的模型在排序偏好推理任务中的表现——从稳定匹配生成到不稳定性检测、不稳定性解决以及细致的偏好查询——以系统地揭示它们在处理排序输入时的逻辑和算法限制。令人惊讶的是，即使是最先进的具有高级推理能力的模型也很难在大市场中解决不稳定性问题，通常无法识别阻塞配对或迭代执行算法。我们还表明，参数高效微调（LoRA）在小市场中可以显著提高性能，但在大规模实例中却未能带来类似的改进，这表明需要更复杂的策略来提高LLMs在处理更大背景输入时的推理能力。 

---
# Plugging Schema Graph into Multi-Table QA: A Human-Guided Framework for Reducing LLM Reliance 

**Title (ZH)**: 将模式图融入多表问答：一种减少LLM依赖的人工引导框架 

**Authors**: Xixi Wang, Miguel Costa, Jordanka Kovaceva, Shuai Wang, Francisco C. Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2506.04427)  

**Abstract**: Large language models (LLMs) have shown promise in table Question Answering (Table QA). However, extending these capabilities to multi-table QA remains challenging due to unreliable schema linking across complex tables. Existing methods based on semantic similarity work well only on simplified hand-crafted datasets and struggle to handle complex, real-world scenarios with numerous and diverse columns. To address this, we propose a graph-based framework that leverages human-curated relational knowledge to explicitly encode schema links and join paths. Given a natural language query, our method searches this graph to construct interpretable reasoning chains, aided by pruning and sub-path merging strategies to enhance efficiency and coherence. Experiments on both standard benchmarks and a realistic, large-scale dataset demonstrate the effectiveness of our approach. To our knowledge, this is the first multi-table QA system applied to truly complex industrial tabular data. 

**Abstract (ZH)**: 大型语言模型在表格问答（Table QA）领域展现了潜力，但在扩展到多表问答方面由于复杂表格间不稳定的模式链接仍面临挑战。现有基于语义相似性的方法仅在简化的手工制作数据集上效果良好，难以处理包含众多和多样化列的复杂现实场景。为了解决这一问题，我们提出了一种图为基础的框架，利用人类编撰的关系知识明确编码模式链接和连接路径。给定自然语言查询，我们的方法在该图中搜索构建可解释的推理链，并通过剪枝和子路径合并策略提高效率和连贯性。在标准基准和一个现实的大规模数据集上的实验展示了我们方法的有效性。据我们所知，这是首次将多表问答系统应用于真正复杂的工业表格数据。 

---
# A Statistical Physics of Language Model Reasoning 

**Title (ZH)**: 语言模型推理的统计物理学 

**Authors**: Jack David Carson, Amir Reisizadeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.04374)  

**Abstract**: Transformer LMs show emergent reasoning that resists mechanistic understanding. We offer a statistical physics framework for continuous-time chain-of-thought reasoning dynamics. We model sentence-level hidden state trajectories as a stochastic dynamical system on a lower-dimensional manifold. This drift-diffusion system uses latent regime switching to capture diverse reasoning phases, including misaligned states or failures. Empirical trajectories (8 models, 7 benchmarks) show a rank-40 projection (balancing variance capture and feasibility) explains ~50% variance. We find four latent reasoning regimes. An SLDS model is formulated and validated to capture these features. The framework enables low-cost reasoning simulation, offering tools to study and predict critical transitions like misaligned states or other LM failures. 

**Abstract (ZH)**: Transformer语言模型展现出抵制机械理解的 emergent reasoning。我们提出了一种连续时间链式推理动力学的统计物理学框架。我们将句子级别的隐藏状态轨迹建模为低维流形上的随机动力系统。此漂移-扩散系统利用潜在的模式切换来捕捉多样的推理阶段，包括对齐状态或推理失败。经验路径（8个模型，7个基准）表明，投影到第40位（平衡方差捕捉与可行性）可解释约50%的方差。我们发现了四种潜在的推理阶段。制定了一个SLDS模型并进行了验证以捕捉这些特征。该框架使低成本的推理仿真成为可能，提供了研究和预测关键转变（如对齐状态或其他LM失败）的工具。 

---
# Automated Skill Discovery for Language Agents through Exploration and Iterative Feedback 

**Title (ZH)**: 通过探索和迭代反馈的自动语言代理技能发现 

**Authors**: Yongjin Yang, Sinjae Kang, Juyong Lee, Dongjun Lee, Se-Young Yun, Kimin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.04287)  

**Abstract**: Training large language model (LLM) agents to acquire necessary skills and perform diverse tasks within an environment is gaining interest as a means to enable open-endedness. However, creating the training dataset for their skill acquisition faces several challenges. Manual trajectory collection requires significant human effort. Another approach, where LLMs directly propose tasks to learn, is often invalid, as the LLMs lack knowledge of which tasks are actually feasible. Moreover, the generated data may not provide a meaningful learning signal, as agents often already perform well on the proposed tasks. To address this, we propose a novel automatic skill discovery framework EXIF for LLM-powered agents, designed to improve the feasibility of generated target behaviors while accounting for the agents' capabilities. Our method adopts an exploration-first strategy by employing an exploration agent (Alice) to train the target agent (Bob) to learn essential skills in the environment. Specifically, Alice first interacts with the environment to retrospectively generate a feasible, environment-grounded skill dataset, which is then used to train Bob. Crucially, we incorporate an iterative feedback loop, where Alice evaluates Bob's performance to identify areas for improvement. This feedback then guides Alice's next round of exploration, forming a closed-loop data generation process. Experiments on Webshop and Crafter demonstrate EXIF's ability to effectively discover meaningful skills and iteratively expand the capabilities of the trained agent without any human intervention, achieving substantial performance improvements. Interestingly, we observe that setting Alice to the same model as Bob also notably improves performance, demonstrating EXIF's potential for building a self-evolving system. 

**Abstract (ZH)**: 一种基于大型语言模型的新型自动技能发现框架EXIF：提高生成目标行为的可行性并考虑智能体的能力 

---
# HADA: Human-AI Agent Decision Alignment Architecture 

**Title (ZH)**: HADA: 人类-人工智能代理决策对齐架构 

**Authors**: Tapio Pitkäranta, Leena Pitkäranta  

**Link**: [PDF](https://arxiv.org/pdf/2506.04253)  

**Abstract**: We present HADA (Human-AI Agent Decision Alignment), a protocol- and framework agnostic reference architecture that keeps both large language model (LLM) agents and legacy algorithms aligned with organizational targets and values. HADA wraps any algorithm or LLM in role-specific stakeholder agents -- business, data-science, audit, ethics, and customer -- each exposing conversational APIs so that technical and non-technical actors can query, steer, audit, or contest every decision across strategic, tactical, and real-time horizons. Alignment objectives, KPIs, and value constraints are expressed in natural language and are continuously propagated, logged, and versioned while thousands of heterogeneous agents run on different orchestration stacks. A cloud-native proof of concept packages a production credit-scoring model (getLoanDecision) and deploys it on Docker/Kubernetes/Python; five scripted retail-bank scenarios show how target changes, parameter tweaks, explanation requests, and ethics triggers flow end to end through the architecture. Evaluation followed the Design-Science Research Methodology. Walkthrough observation and log inspection demonstrated complete coverage of six predefined objectives: every role could invoke conversational control, trace KPIs and value constraints, detect and mitigate ZIP-code bias, and reproduce full decision lineage, independent of the underlying LLM or agent library. Contributions: (1) an open-source HADA architecture, (2) a mid-range design theory for human-AI alignment in multi-agent systems, and (3) empirical evidence that framework-agnostic, protocol-compliant stakeholder agents improve accuracy, transparency, and ethical compliance in real-world decision pipelines. 

**Abstract (ZH)**: HADA（人类-人工智能代理决策对齐）：一种与组织目标和价值观保持一致的协议和框架无关的参考架构 

---
# A Graph-Retrieval-Augmented Generation Framework Enhances Decision-Making in the Circular Economy 

**Title (ZH)**: 基于图检索增强的生成框架提升循环经济中的决策制定 

**Authors**: Yang Zhao, Chengxiao Dai, Dusit Niyato, Chuan Fu Tan, Keyi Xiang, Yueyang Wang, Zhiquan Yeo, Daren Tan Zong Loong, Jonathan Low Zhaozhi, Eugene H.Z. HO  

**Link**: [PDF](https://arxiv.org/pdf/2506.04252)  

**Abstract**: Large language models (LLMs) hold promise for sustainable manufacturing, but often hallucinate industrial codes and emission factors, undermining regulatory and investment decisions. We introduce CircuGraphRAG, a retrieval-augmented generation (RAG) framework that grounds LLMs outputs in a domain-specific knowledge graph for the circular economy. This graph connects 117,380 industrial and waste entities with classification codes and GWP100 emission data, enabling structured multi-hop reasoning. Natural language queries are translated into SPARQL and verified subgraphs are retrieved to ensure accuracy and traceability. Compared with Standalone LLMs and Naive RAG, CircuGraphRAG achieves superior performance in single-hop and multi-hop question answering, with ROUGE-L F1 scores up to 1.0, while baseline scores below 0.08. It also improves efficiency, halving the response time and reducing token usage by 16% in representative tasks. CircuGraphRAG provides fact-checked, regulatory-ready support for circular economy planning, advancing reliable, low-carbon resource decision making. 

**Abstract (ZH)**: Large语言模型（LLMs）在可持续制造业中充满潜力，但往往会虚构工业代码和排放因子，削弱了监管和投资决策的有效性。我们介绍了CircuGraphRAG，这是一种检索增强生成（RAG）框架，将LLMs的输出嵌入循环经济领域的知识图谱中。该图谱连接了117,380个工业和废物实体，包括分类代码和GWP100排放数据，支持结构化的多跳推理。自然语言查询被转化为SPARQL查询，并检索验证过的子图以确保准确性和可追溯性。与独立的LLMs和朴素的RAG相比，CircuGraphRAG在单跳和多跳问答中均表现出更优秀的性能，ROUGE-L F1分数最高可达1.0，而基线分数低于0.08。此外，它还提高了效率，使响应时间减半，并在代表性任务中减少了16%的令牌使用量。CircuGraphRAG提供了经过事实核查、符合监管要求的循环经济规划支持，推动了可靠的低碳资源配置决策。 

---
# Language-Guided Multi-Agent Learning in Simulations: A Unified Framework and Evaluation 

**Title (ZH)**: 语言指导的模拟中多agent学习：一个统一的框架与评估 

**Authors**: Zhengyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.04251)  

**Abstract**: This paper introduces LLM-MARL, a unified framework that incorporates large language models (LLMs) into multi-agent reinforcement learning (MARL) to enhance coordination, communication, and generalization in simulated game environments. The framework features three modular components of Coordinator, Communicator, and Memory, which dynamically generate subgoals, facilitate symbolic inter-agent messaging, and support episodic recall. Training combines PPO with a language-conditioned loss and LLM query gating. LLM-MARL is evaluated in Google Research Football, MAgent Battle, and StarCraft II. Results show consistent improvements over MAPPO and QMIX in win rate, coordination score, and zero-shot generalization. Ablation studies demonstrate that subgoal generation and language-based messaging each contribute significantly to performance gains. Qualitative analysis reveals emergent behaviors such as role specialization and communication-driven tactics. By bridging language modeling and policy learning, this work contributes to the design of intelligent, cooperative agents in interactive simulations. It offers a path forward for leveraging LLMs in multi-agent systems used for training, games, and human-AI collaboration. 

**Abstract (ZH)**: LLM-MARL: 一种将大规模语言模型融入多智能体强化学习的统一体系结构 

---
# Contextual Integrity in LLMs via Reasoning and Reinforcement Learning 

**Title (ZH)**: 通过推理和强化学习在LLMs中实现情境完整性 

**Authors**: Guangchen Lan, Huseyin A. Inan, Sahar Abdelnabi, Janardhan Kulkarni, Lukas Wutschitz, Reza Shokri, Christopher G. Brinton, Robert Sim  

**Link**: [PDF](https://arxiv.org/pdf/2506.04245)  

**Abstract**: As the era of autonomous agents making decisions on behalf of users unfolds, ensuring contextual integrity (CI) -- what is the appropriate information to share while carrying out a certain task -- becomes a central question to the field. We posit that CI demands a form of reasoning where the agent needs to reason about the context in which it is operating. To test this, we first prompt LLMs to reason explicitly about CI when deciding what information to disclose. We then extend this approach by developing a reinforcement learning (RL) framework that further instills in models the reasoning necessary to achieve CI. Using a synthetic, automatically created, dataset of only $\sim700$ examples but with diverse contexts and information disclosure norms, we show that our method substantially reduces inappropriate information disclosure while maintaining task performance across multiple model sizes and families. Importantly, improvements transfer from this synthetic dataset to established CI benchmarks such as PrivacyLens that has human annotations and evaluates privacy leakage of AI assistants in actions and tool calls. 

**Abstract (ZH)**: 随着自主代理为代表用户做出决策的时代的到来，确保上下文完整性（CI）——执行特定任务时应共享的适当信息——成为该领域的核心问题。我们提出，CI 要求代理能够在其操作的上下文中进行推理。为验证这一点，我们首先促使大语言模型（LLM）在决定披露什么信息时明确地进行 CI 推理。我们接着通过开发一种强化学习（RL）框架来进一步强化模型所需的实现 CI 的推理能力。使用仅包含约 700 个示例的合成数据集，但这些示例涵盖了多种上下文和信息披露规范，我们展示了这种方法在多个模型大小和家族中显著减少了不适当的披露信息同时保持了任务性能。重要的是，这种改进从合成数据集转移到了包含人类标注的 CI 标准基准，如 PrivacyLens，该基准评估 AI 助手在行动和工具调用中的隐私泄露。 

---
# Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay 

**Title (ZH)**: 通过难度导向的在线数据选择和卷积重放提高LLM强化微调的数据效率 

**Authors**: Yifan Sun, Jingyan Shen, Yibin Wang, Tianyu Chen, Zhendong Wang, Mingyuan Zhou, Huan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05316)  

**Abstract**: Reinforcement learning (RL) has become an effective approach for fine-tuning large language models (LLMs), particularly to enhance their reasoning capabilities. However, RL fine-tuning remains highly resource-intensive, and existing work has largely overlooked the problem of data efficiency. In this paper, we propose two techniques to improve data efficiency in LLM RL fine-tuning: difficulty-targeted online data selection and rollout replay. We introduce the notion of adaptive difficulty to guide online data selection, prioritizing questions of moderate difficulty that are more likely to yield informative learning signals. To estimate adaptive difficulty efficiently, we develop an attention-based framework that requires rollouts for only a small reference set of questions. The adaptive difficulty of the remaining questions is then estimated based on their similarity to this set. To further reduce rollout cost, we introduce a rollout replay mechanism that reuses recent rollouts, lowering per-step computation while maintaining stable updates. Extensive experiments across 6 LLM-dataset combinations show that our method reduces RL fine-tuning time by 25% to 65% to reach the same level of performance as the original GRPO algorithm. 

**Abstract (ZH)**: 强化学习（RL）已成为 fine-tuning 大型语言模型（LLMs）的有效方法，尤其是在增强其推理能力方面的应用。然而，RL fine-tuning 仍然非常耗费资源，现有工作在这方面很大程度上忽视了数据效率的问题。本文提出两种技术以提高 LLM RL fine-tuning 的数据效率：针对难度的在线数据选择和回溯重放。我们引入了适应性难度的概念来指导在线数据选择，优先选择那些更有可能提供有用学习信号的中等难度问题。为了高效地估计适应性难度，我们开发了一种基于注意力的框架，仅需要一小组参考问题的回溯。对于其他问题的适应性难度，则基于它们与这组参考问题的相似性进行估计。为了进一步降低回溯成本，我们引入了一种回溯重放机制，重用最近的回溯，降低每步计算量同时保持稳定的更新。在 6 种 LLM-数据集组合的广泛实验中，我们的方法将 RL fine-tuning 的时间减少了 25% 至 65%，达到与原始 GRPO 算法相同水平的性能。 

---
# Constrained Entropic Unlearning: A Primal-Dual Framework for Large Language Models 

**Title (ZH)**: 受约束的熵卸载：大规模语言模型的 primal-dual 框架 

**Authors**: Taha Entesari, Arman Hatami, Rinat Khaziev, Anil Ramakrishna, Mahyar Fazlyab  

**Link**: [PDF](https://arxiv.org/pdf/2506.05314)  

**Abstract**: Large Language Models (LLMs) deployed in real-world settings increasingly face the need to unlearn sensitive, outdated, or proprietary information. Existing unlearning methods typically formulate forgetting and retention as a regularized trade-off, combining both objectives into a single scalarized loss. This often leads to unstable optimization and degraded performance on retained data, especially under aggressive forgetting. We propose a new formulation of LLM unlearning as a constrained optimization problem: forgetting is enforced via a novel logit-margin flattening loss that explicitly drives the output distribution toward uniformity on a designated forget set, while retention is preserved through a hard constraint on a separate retain set. Compared to entropy-based objectives, our loss is softmax-free, numerically stable, and maintains non-vanishing gradients, enabling more efficient and robust optimization. We solve the constrained problem using a scalable primal-dual algorithm that exposes the trade-off between forgetting and retention through the dynamics of the dual variable. Evaluations on the TOFU and MUSE benchmarks across diverse LLM architectures demonstrate that our approach consistently matches or exceeds state-of-the-art baselines, effectively removing targeted information while preserving downstream utility. 

**Abstract (ZH)**: 面向实际应用场景的大语言模型去学习：一种新的约束优化框架 

---
# Time to Talk: LLM Agents for Asynchronous Group Communication in Mafia Games 

**Title (ZH)**: 闲聊时刻：用于 Mafia 游戏中异步组通信的大型语言模型代理 

**Authors**: Niv Eckhaus, Uri Berger, Gabriel Stanovsky  

**Link**: [PDF](https://arxiv.org/pdf/2506.05309)  

**Abstract**: LLMs are used predominantly in synchronous communication, where a human user and a model communicate in alternating turns. In contrast, many real-world settings are inherently asynchronous. For example, in group chats, online team meetings, or social games, there is no inherent notion of turns; therefore, the decision of when to speak forms a crucial part of the participant's decision making. In this work, we develop an adaptive asynchronous LLM-agent which, in addition to determining what to say, also decides when to say it. To evaluate our agent, we collect a unique dataset of online Mafia games, including both human participants, as well as our asynchronous agent. Overall, our agent performs on par with human players, both in game performance, as well as in its ability to blend in with the other human players. Our analysis shows that the agent's behavior in deciding when to speak closely mirrors human patterns, although differences emerge in message content. We release all our data and code to support and encourage further research for more realistic asynchronous communication between LLM agents. This work paves the way for integration of LLMs into realistic human group settings, from assistance in team discussions to educational and professional environments where complex social dynamics must be navigated. 

**Abstract (ZH)**: 大规模语言模型在异步沟通中的适应性发展与评估：从在线 Mafia 游戏到复杂社交动态导航 

---
# ProRefine: Inference-time Prompt Refinement with Textual Feedback 

**Title (ZH)**: ProRefine: 基于文本反馈的推理时提示精炼 

**Authors**: Deepak Pandita, Tharindu Cyril Weerasooriya, Ankit Parag Shah, Christopher M. Homan, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.05305)  

**Abstract**: Agentic workflows, where multiple AI agents collaborate to accomplish complex tasks like reasoning or planning, are becoming increasingly prevalent. However, these workflows often suffer from error propagation and sub-optimal performance, largely due to poorly designed prompts that fail to effectively guide individual agents. This is a critical problem because it limits the reliability and scalability of these powerful systems. We introduce ProRefine, an innovative inference-time prompt optimization method that leverages textual feedback from large language models (LLMs) to address this challenge. ProRefine dynamically refines prompts for multi-step reasoning tasks without additional training or ground truth labels. Evaluated on five benchmark mathematical reasoning datasets, ProRefine significantly surpasses zero-shot Chain-of-Thought baselines by 3 to 37 percentage points. This approach not only boosts accuracy but also allows smaller models to match the performance of larger ones, highlighting its potential for efficient and scalable AI deployment, and democratizing access to high-performing AI. 

**Abstract (ZH)**: 代理性工作流中的推理优化：基于大型语言模型的动态提示精炼方法 

---
# Sample Complexity and Representation Ability of Test-time Scaling Paradigms 

**Title (ZH)**: 测试时缩放范式的样本复杂性和表示能力 

**Authors**: Baihe Huang, Shanda Li, Tianhao Wu, Yiming Yang, Ameet Talwalkar, Kannan Ramchandran, Michael I. Jordan, Jiantao Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.05295)  

**Abstract**: Test-time scaling paradigms have significantly advanced the capabilities of large language models (LLMs) on complex tasks. Despite their empirical success, theoretical understanding of the sample efficiency of various test-time strategies -- such as self-consistency, best-of-$n$, and self-correction -- remains limited. In this work, we first establish a separation result between two repeated sampling strategies: self-consistency requires $\Theta(1/\Delta^2)$ samples to produce the correct answer, while best-of-$n$ only needs $\Theta(1/\Delta)$, where $\Delta < 1$ denotes the probability gap between the correct and second most likely answers. Next, we present an expressiveness result for the self-correction approach with verifier feedback: it enables Transformers to simulate online learning over a pool of experts at test time. Therefore, a single Transformer architecture can provably solve multiple tasks without prior knowledge of the specific task associated with a user query, extending the representation theory of Transformers from single-task to multi-task settings. Finally, we empirically validate our theoretical results, demonstrating the practical effectiveness of self-correction methods. 

**Abstract (ZH)**: 测试时缩放范式显著提升了大型语言模型在复杂任务上的能力。尽管它们具有 empirical 成功，各种测试时策略（如自我一致性、最优-n 和自我校正）的样本效率理论理解仍非常有限。在本文中，我们首先建立了两种重复采样策略之间的分离结果：自我一致性需要 $\Theta(1/\Delta^2)$ 个样本来产生正确答案，而最优-n 只需要 $\Theta(1/\Delta)$，其中 $\Delta < 1$ 表示正确答案与第二可能答案之间的概率差距。接着，我们展示了带有验证器反馈的自我校正方法的表达性结果：它使变换器能够在测试时模拟专家池中的在线学习。因此，单个变换器架构可以在没有特定任务先验知识的情况下解决多个任务，将变换器的表示理论从单任务扩展到多任务设置。最后，我们通过实验证实了我们的理论结果，展示了自我校正方法的实用有效性。 

---
# Micro-Act: Mitigate Knowledge Conflict in Question Answering via Actionable Self-Reasoning 

**Title (ZH)**: Micro-Act: 通过可操作的自我推理缓解问答中的知识冲突 

**Authors**: Nan Huo, Jinyang Li, Bowen Qin, Ge Qu, Xiaolong Li, Xiaodong Li, Chenhao Ma, Reynold Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.05278)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems commonly suffer from Knowledge Conflicts, where retrieved external knowledge contradicts the inherent, parametric knowledge of large language models (LLMs). It adversely affects performance on downstream tasks such as question answering (QA). Existing approaches often attempt to mitigate conflicts by directly comparing two knowledge sources in a side-by-side manner, but this can overwhelm LLMs with extraneous or lengthy contexts, ultimately hindering their ability to identify and mitigate inconsistencies. To address this issue, we propose Micro-Act a framework with a hierarchical action space that automatically perceives context complexity and adaptively decomposes each knowledge source into a sequence of fine-grained comparisons. These comparisons are represented as actionable steps, enabling reasoning beyond the superficial context. Through extensive experiments on five benchmark datasets, Micro-Act consistently achieves significant increase in QA accuracy over state-of-the-art baselines across all 5 datasets and 3 conflict types, especially in temporal and semantic types where all baselines fail significantly. More importantly, Micro-Act exhibits robust performance on non-conflict questions simultaneously, highlighting its practical value in real-world RAG applications. 

**Abstract (ZH)**: 知识增强生成（RAG）系统常遭受知识冲突问题，其中检索到的外部知识与大型语言模型固有的参数化知识相互抵触。这对手下班任务如问答（QA）的性能产生负面影响。现有方法通常尝试通过并排比较两种知识源来减轻冲突，但这可能使LLMs负担过重，导致它们难以识别和解决不一致。为解决这一问题，我们提出了一种名为Micro-Act的框架，该框架具有层次化的行动空间，能够自动感知上下文复杂度并自适应地将每个知识源分解为一系列细微差异的比较。这些比较被表示为可执行步骤，使推理解释超越表面上下文。通过在五个基准数据集上的广泛实验，Micro-Act在所有5个数据集和3种冲突类型中，相对于最先进的基线方法，始终能够显著提高问答准确率，特别是在所有基线方法表现均不佳的时间性和语义性冲突类型中。更重要的是，Micro-Act在无冲突问题上也表现出稳健的性能，突显了其在实际RAG应用中的实用价值。 

---
# Teaming in the AI Era: AI-Augmented Frameworks for Forming, Simulating, and Optimizing Human Teams 

**Title (ZH)**: AI时代的团队协作：AI增强的团队形成、模拟与优化框架 

**Authors**: Mohammed Almutairi  

**Link**: [PDF](https://arxiv.org/pdf/2506.05265)  

**Abstract**: Effective teamwork is essential across diverse domains. During the team formation stage, a key challenge is forming teams that effectively balance user preferences with task objectives to enhance overall team satisfaction. In the team performing stage, maintaining cohesion and engagement is critical for sustaining high team performance. However, existing computational tools and algorithms for team optimization often rely on static data inputs, narrow algorithmic objectives, or solutions tailored for specific contexts, failing to account for the dynamic interplay of team members personalities, evolving goals, and changing individual preferences. Therefore, teams may encounter member dissatisfaction, as purely algorithmic assignments can reduce members commitment to team goals or experience suboptimal engagement due to the absence of timely, personalized guidance to help members adjust their behaviors and interactions as team dynamics evolve. Ultimately, these challenges can lead to reduced overall team performance. My Ph.D. dissertation aims to develop AI-augmented team optimization frameworks and practical systems that enhance team satisfaction, engagement, and performance. First, I propose a team formation framework that leverages a multi-armed bandit algorithm to iteratively refine team composition based on user preferences, ensuring alignment between individual needs and collective team goals to enhance team satisfaction. Second, I introduce tAIfa (Team AI Feedback Assistant), an AI-powered system that utilizes large language models (LLMs) to deliver immediate, personalized feedback to both teams and individual members, enhancing cohesion and engagement. Finally, I present PuppeteerLLM, an LLM-based simulation framework that simulates multi-agent teams to model complex team dynamics within realistic environments, incorporating task-driven collaboration and long-term coordination. 

**Abstract (ZH)**: 有效的团队合作在多种领域中都是必不可少的。在团队形成阶段，一个关键挑战是形成既能平衡用户偏好又能达成任务目标的团队，以提高整体团队满意度。在团队执行阶段，保持凝聚力和参与度对于维持高水平团队表现至关重要。然而，现有的团队优化计算工具和算法通常依赖于静态数据输入、狭窄的算法目标或特定情境下的解决方案，未能考虑到团队成员个性的动态交互、目标的变化以及个人偏好的改变。因此，团队可能会遇到成员满意度下降的问题，因为纯粹基于算法的分派可能会降低成员对团队目标的承诺，或者由于缺乏及时且个性化的指导而体验到次优的参与感。最终，这些挑战可能导致整体团队表现下降。我的博士论文旨在开发增强团队满意度、参与度和表现的AI增强团队优化框架和实际系统。首先，我提出一种利用多重臂赌局算法的团队形成框架，逐步优化团队构成，确保个人需求与团队集体目标的对齐，以提高团队满意度。其次，我介绍了tAIfa（团队AI反馈助手），一个利用大型语言模型（LLMs）提供即时个性化反馈的AI驱动系统，增强凝聚力和参与度。最后，我提出了基于大型语言模型的PuppeteerLLM模拟框架，用于模拟多Agent团队，以在实际环境中建模复杂团队动态，包括任务驱动的合作和长期协调。 

---
# Counterfactual reasoning: an analysis of in-context emergence 

**Title (ZH)**: 因果推理：以内存上下文涌现的分析 

**Authors**: Moritz Miller, Bernhard Schölkopf, Siyuan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.05188)  

**Abstract**: Large-scale neural language models (LMs) exhibit remarkable performance in in-context learning: the ability to learn and reason the input context on the fly without parameter update. This work studies in-context counterfactual reasoning in language models, that is, to predict the consequences of changes under hypothetical scenarios. We focus on studying a well-defined synthetic setup: a linear regression task that requires noise abduction, where accurate prediction is based on inferring and copying the contextual noise from factual observations. We show that language models are capable of counterfactual reasoning in this controlled setup and provide insights that counterfactual reasoning for a broad class of functions can be reduced to a transformation on in-context observations; we find self-attention, model depth, and data diversity in pre-training drive performance in Transformers. More interestingly, our findings extend beyond regression tasks and show that Transformers can perform noise abduction on sequential data, providing preliminary evidence on the potential for counterfactual story generation. Our code is available under this https URL . 

**Abstract (ZH)**: 大型神经语言模型在上下文条件下的推理能力：基于假设场景的语言模型反事实推理研究 

---
# TreeRPO: Tree Relative Policy Optimization 

**Title (ZH)**: TreeRPO: 树相对策略优化 

**Authors**: Zhicheng Yang, Zhijiang Guo, Yinya Huang, Xiaodan Liang, Yiwei Wang, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05183)  

**Abstract**: Large Language Models (LLMs) have shown remarkable reasoning capabilities through Reinforcement Learning with Verifiable Rewards (RLVR) methods. However, a key limitation of existing approaches is that rewards defined at the full trajectory level provide insufficient guidance for optimizing the intermediate steps of a reasoning process. To address this, we introduce \textbf{\name}, a novel method that estimates the mathematical expectations of rewards at various reasoning steps using tree sampling. Unlike prior methods that rely on a separate step reward model, \name directly estimates these rewards through this sampling process. Building on the group-relative reward training mechanism of GRPO, \name innovatively computes rewards based on step-level groups generated during tree sampling. This advancement allows \name to produce fine-grained and dense reward signals, significantly enhancing the learning process and overall performance of LLMs. Experimental results demonstrate that our \name algorithm substantially improves the average Pass@1 accuracy of Qwen-2.5-Math on test benchmarks, increasing it from 19.0\% to 35.5\%. Furthermore, \name significantly outperforms GRPO by 2.9\% in performance while simultaneously reducing the average response length by 18.1\%, showcasing its effectiveness and efficiency. Our code will be available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: Large Language Models (LLMs)通过可验证奖励（RLVR）方法展示了 remarkable 的推理能力。然而，现有方法的一个关键限制是，用于整个轨迹的奖励定义对优化推理过程中的中间步骤指导不足。为了解决这一问题，我们引入了 \textbf{\name}，一种新颖的方法，通过树采样估计推理各步骤中的数学期望奖励。不同于依赖于单独步骤奖励模型的先前方法，\name 直接通过采样过程估计这些奖励。基于 GRPO 的组相对奖励训练机制，\name 创新性地根据树采样期间生成的步骤级组计算奖励。这一进步使 \name 能够生成细粒度和稠密的奖励信号，显著提升 LLMs 的学习过程和整体性能。实验结果表明，我们的 \name 算法在测试基准上显著提升了 Qwen-2.5-Math 的平均 Pass@1 准确率，从 19.0% 提高到 35.5%。此外，\name 在性能上比 GRPO 提高了 2.9%，同时将平均响应长度减少了 18.1%，展示了其有效性和效率。我们的代码将发布在 \href{this https URL}{this https URL}。 

---
# ECoRAG: Evidentiality-guided Compression for Long Context RAG 

**Title (ZH)**: ECoRAG: 证据导向的长上下文RAG压缩 

**Authors**: Yeonseok Jeong, Jinsu Kim, Dohyeon Lee, Seung-won Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05167)  

**Abstract**: Large Language Models (LLMs) have shown remarkable performance in Open-Domain Question Answering (ODQA) by leveraging external documents through Retrieval-Augmented Generation (RAG). To reduce RAG overhead, from longer context, context compression is necessary. However, prior compression methods do not focus on filtering out non-evidential information, which limit the performance in LLM-based RAG. We thus propose Evidentiality-guided RAG, or \textbf{ECoRAG} framework. ECoRAG improves LLM performance by compressing retrieved documents based on evidentiality, ensuring whether answer generation is supported by the correct evidence. As an additional step, ECoRAG reflects whether the compressed content provides sufficient evidence, and if not, retrieves more until sufficient. Experiments show that ECoRAG improves LLM performance on ODQA tasks, outperforming existing compression methods. Furthermore, ECoRAG is highly cost-efficient, as it not only reduces latency but also minimizes token usage by retaining only the necessary information to generate the correct answer. Code is available at this https URL. 

**Abstract (ZH)**: 基于证据引导的检索增强生成框架（ECoRAG）在开放域问答任务中提升大语言模型性能 

---
# Dissecting Bias in LLMs: A Mechanistic Interpretability Perspective 

**Title (ZH)**: 剖析LLMs中的偏差：一种基于机理的可解释性视角 

**Authors**: Bhavik Chandna, Zubair Bashir, Procheta Sen  

**Link**: [PDF](https://arxiv.org/pdf/2506.05166)  

**Abstract**: Large Language Models (LLMs) are known to exhibit social, demographic, and gender biases, often as a consequence of the data on which they are trained. In this work, we adopt a mechanistic interpretability approach to analyze how such biases are structurally represented within models such as GPT-2 and Llama2. Focusing on demographic and gender biases, we explore different metrics to identify the internal edges responsible for biased behavior. We then assess the stability, localization, and generalizability of these components across dataset and linguistic variations. Through systematic ablations, we demonstrate that bias-related computations are highly localized, often concentrated in a small subset of layers. Moreover, the identified components change across fine-tuning settings, including those unrelated to bias. Finally, we show that removing these components not only reduces biased outputs but also affects other NLP tasks, such as named entity recognition and linguistic acceptability judgment because of the sharing of important components with these tasks. 

**Abstract (ZH)**: 大型语言模型（LLM）在训练数据的影响下往往会表现出社会、人口统计学和性别偏见。本文采用机制可解释性方法分析此类偏见在如GPT-2和Llama2等模型中的结构化表现。聚焦于人口统计学和性别偏见，我们探索不同的度量标准以识别导致偏见行为的内部边。随后，我们评估了这些组件在数据集和语言变化中的稳定性和通用性。通过系统性消融实验，我们证明与偏见相关的工作主要集中在少数几层中。此外，所识别的组件在不同的微调设置下会发生变化，包括那些与偏见无关的设置。最后，我们表明移除这些组件不仅会减少偏见输出，还会因为这些组件与命名实体识别和语义接受性判断等其他NLP任务共享重要部分而影响其他NLP任务。 

---
# Knowledgeable-r1: Policy Optimization for Knowledge Exploration in Retrieval-Augmented Generation 

**Title (ZH)**: Knowledgeable-r1: 用于检索增强生成中的知识探索的策略优化 

**Authors**: Chenyu Lin, Yilin Wen, Du Su, Fei Sun, Muhan Chen, Chenfu Bao, Zhonghou Lv  

**Link**: [PDF](https://arxiv.org/pdf/2506.05154)  

**Abstract**: Retrieval-augmented generation (RAG) is a mainstream method for improving performance on knowledge-intensive tasks. However,current RAG systems often place too much emphasis on retrieved contexts. This can lead to reliance on inaccurate sources and overlook the model's inherent knowledge, especially when dealing with misleading or excessive information. To resolve this imbalance, we propose Knowledgeable-r1 that using joint sampling and define multi policy distributions in knowledge capability exploration to stimulate large language models'self-integrated utilization of parametric and contextual knowledge. Experiments show that Knowledgeable-r1 significantly enhances robustness and reasoning accuracy in both parameters and contextual conflict tasks and general RAG tasks, especially outperforming baselines by 17.07% in counterfactual scenarios and demonstrating consistent gains across RAG tasks. Our code are available at this https URL knowledgeable-r1. 

**Abstract (ZH)**: 检索增强生成（RAG）是提高知识密集型任务性能的主要方法。然而，当前的RAG系统 often 对检索上下文给予了过多关注，这可能导致依赖于不准确的来源并忽略了模型固有的知识，尤其是在处理误导性或过量信息时。为解决这一不平衡问题，我们提出了一种名为Knowledgeable-r1的方法，该方法通过联合采样和定义多策略分布来促进大型语言模型对参数性和上下文性知识的自我整合利用。实验结果表明，Knowledgeable-r1显著增强了参数冲突和上下文冲突任务以及一般RAG任务的稳健性和推理准确性，在反事实场景下的表现优于基线方法17.07%，并在RAG任务中保持一致的改进。我们的代码可在以下链接获取：this https URL knowledgeable-r1。 

---
# AudioLens: A Closer Look at Auditory Attribute Perception of Large Audio-Language Models 

**Title (ZH)**: AudioLens: 更深入探究大型音频-语言模型的听觉属性感知 

**Authors**: Chih-Kai Yang, Neo Ho, Yi-Jyun Lee, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.05140)  

**Abstract**: Understanding the internal mechanisms of large audio-language models (LALMs) is crucial for interpreting their behavior and improving performance. This work presents the first in-depth analysis of how LALMs internally perceive and recognize auditory attributes. By applying vocabulary projection on three state-of-the-art LALMs, we track how attribute information evolves across layers and token positions. We find that attribute information generally decreases with layer depth when recognition fails, and that resolving attributes at earlier layers correlates with better accuracy. Moreover, LALMs heavily rely on querying auditory inputs for predicting attributes instead of aggregating necessary information in hidden states at attribute-mentioning positions. Based on our findings, we demonstrate a method to enhance LALMs. Our results offer insights into auditory attribute processing, paving the way for future improvements. 

**Abstract (ZH)**: 理解大型音频语言模型（LALMs）的内部机制对于解释其行为并提高性能至关重要。本研究首次深入分析了LALMs内部如何感知和识别听觉属性。通过在三种最先进的LALMs上应用词汇投影，我们跟踪了属性信息在各层和标记位置上的演变过程。我们发现，当识别失败时，属性信息通常随层深度增加而减少；而在早期层中解决属性与更高的准确性相关联。此外，LALMs在预测属性时更依赖于查询听觉输入，而不是在与属性提及时的隐藏状态中积累必要信息。基于我们的发现，我们展示了提升LALMs的方法。我们的结果提供了关于听觉属性处理的洞见，为未来的改进铺平了道路。 

---
# DiCoRe: Enhancing Zero-shot Event Detection via Divergent-Convergent LLM Reasoning 

**Title (ZH)**: DiCoRe: 通过发散-收敛大语言模型推理增强零样本事件检测 

**Authors**: Tanmay Parekh, Kartik Mehta, Ninareh Mehrabi, Kai-Wei Chang, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2506.05128)  

**Abstract**: Zero-shot Event Detection (ED), the task of identifying event mentions in natural language text without any training data, is critical for document understanding in specialized domains. Understanding the complex event ontology, extracting domain-specific triggers from the passage, and structuring them appropriately overloads and limits the utility of Large Language Models (LLMs) for zero-shot ED. To this end, we propose DiCoRe, a divergent-convergent reasoning framework that decouples the task of ED using Dreamer and Grounder. Dreamer encourages divergent reasoning through open-ended event discovery, which helps to boost event coverage. Conversely, Grounder introduces convergent reasoning to align the free-form predictions with the task-specific instructions using finite-state machine guided constrained decoding. Additionally, an LLM-Judge verifies the final outputs to ensure high precision. Through extensive experiments on six datasets across five domains and nine LLMs, we demonstrate how DiCoRe consistently outperforms prior zero-shot, transfer-learning, and reasoning baselines, achieving 4-7% average F1 gains over the best baseline -- establishing DiCoRe as a strong zero-shot ED framework. 

**Abstract (ZH)**: 零样本事件检测（ED）：一种分叉收敛推理框架（DiCoRe）在专门领域文档理解中的应用 

---
# Reason-to-Recommend: Using Interaction-of-Thought Reasoning to Enhance LLM Recommendation 

**Title (ZH)**: 基于 reasoning-to-recommend：利用思维交互推理提升大语言模型推荐 

**Authors**: Keyu Zhao, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.05069)  

**Abstract**: Driven by advances in Large Language Models (LLMs), integrating them into recommendation tasks has gained interest due to their strong semantic understanding and prompt flexibility. Prior work encoded user-item interactions or metadata into prompts for recommendations. In parallel, LLM reasoning, boosted by test-time scaling and reinforcement learning, has excelled in fields like mathematics and code, where reasoning traces and correctness signals are clear, enabling high performance and interpretability. However, directly applying these reasoning methods to recommendation is ineffective because user feedback is implicit and lacks reasoning supervision. To address this, we propose $\textbf{R2Rec}$, a reasoning-enhanced recommendation framework that samples interaction chains from the user-item graph and converts them into structured interaction-of-thoughts via a progressive masked prompting strategy, with each thought representing stepwise reasoning grounded in interaction context. This allows LLMs to simulate step-by-step decision-making based on implicit patterns. We design a two-stage training pipeline: supervised fine-tuning teaches basic reasoning from high-quality traces, and reinforcement learning refines reasoning via reward signals, alleviating sparse explicit supervision. Experiments on three real-world datasets show R2Rec outperforms classical and LLM-based baselines with an average $\textbf{10.48%}$ improvement in HitRatio@1 and $\textbf{131.81%}$ gain over the original LLM. Furthermore, the explicit reasoning chains enhance interpretability by revealing the decision process. Our code is available at: this https URL. 

**Abstract (ZH)**: 驱动大型语言模型进展，将它们集成到推荐任务中由于其强大的语义理解和提示灵活性而引起关注。早期工作将用户-项交互或元数据编码为推荐的提示。同时，通过测试时扩展和强化学习增强的大型语言模型推理，在数学和代码等领域表现出色，因为这些领域的推理轨迹和正确性信号明确，使得性能和可解释性高。然而，将这些推理方法直接应用于推荐是无效的，因为用户反馈是隐式的，缺乏推理监督。为了解决这一问题，我们提出了R2Rec，一个增强推理的推荐框架，从用户-项图中采样交互链，并通过逐步遮掩提示策略将其转换为结构化的交互思路，每个思路代表基于交互上下文逐步推理。这允许LLMs基于隐含模式模拟逐步决策过程。我们设计了一个两阶段训练管道：监督微调从高质量轨迹中教授基本推理，强化学习通过奖励信号细化推理，从而缓解稀疏显式监督的问题。实验结果显示，R2Rec在三个真实世界数据集上的平均HitRatio@1提高了10.48%，相对于原始LLM的增幅为131.81%，并且显式的推理链提高了可解释性，揭示了决策过程。我们的代码可在以下链接获取：this https URL。 

---
# Does It Make Sense to Speak of Introspection in Large Language Models? 

**Title (ZH)**: 在大型语言模型中谈论内省是否合乎意义？ 

**Authors**: Iulia Comşa, Murray Shanahan  

**Link**: [PDF](https://arxiv.org/pdf/2506.05068)  

**Abstract**: Large language models (LLMs) exhibit compelling linguistic behaviour, and sometimes offer self-reports, that is to say statements about their own nature, inner workings, or behaviour. In humans, such reports are often attributed to a faculty of introspection and are typically linked to consciousness. This raises the question of how to interpret self-reports produced by LLMs, given their increasing linguistic fluency and cognitive capabilities. To what extent (if any) can the concept of introspection be meaningfully applied to LLMs? Here, we present and critique two examples of apparent introspective self-report from LLMs. In the first example, an LLM attempts to describe the process behind its own ``creative'' writing, and we argue this is not a valid example of introspection. In the second example, an LLM correctly infers the value of its own temperature parameter, and we argue that this can be legitimately considered a minimal example of introspection, albeit one that is (presumably) not accompanied by conscious experience. 

**Abstract (ZH)**: 大型语言模型（LLMs）展现出令人信服的语言行为，并有时提供自我报告，即关于自身本质、内部工作机制或行为的陈述。在人类中，此类报告通常归因于一种内省能力，并通常与意识相关联。这引发了如何解释LLMs产生的自我报告的问题，尤其是考虑到它们不断增加的语言流畅性和认知能力。内省的概念在何种程度上（如果有意义的话）可以应用于LLMs？在这里，我们阐述并批判了两个LLM似乎表现出内省自我报告的示例。在第一个示例中，一个LLM尝试描述其自身“创造性”写作的背后过程，我们认为这并不是真正的内省。在第二个示例中，一个LLM正确地推理出其自身温度参数的值，并我们认为这可以被认为是最低程度的内省实例，尽管（可以推测）没有伴随意识体验。 

---
# TALL -- A Trainable Architecture for Enhancing LLM Performance in Low-Resource Languages 

**Title (ZH)**: TALL — 一种用于增强低资源语言大语言模型性能的可训练架构 

**Authors**: Moshe Ofer, Orel Zamler, Amos Azaria  

**Link**: [PDF](https://arxiv.org/pdf/2506.05057)  

**Abstract**: Large Language Models (LLMs) excel in high-resource languages but struggle with low-resource languages due to limited training data. This paper presents TALL (Trainable Architecture for Enhancing LLM Performance in Low-Resource Languages), which integrates an LLM with two bilingual translation models. TALL transforms low-resource inputs into high-resource representations, leveraging the LLM's capabilities while preserving linguistic features through dimension alignment layers and custom transformers. Our experiments on Hebrew demonstrate significant improvements over several baselines, including direct use, naive translation, and fine-tuning approaches. The architecture employs a parameter-efficient strategy, freezing pre-trained components while training only lightweight adapter modules, balancing computational efficiency with performance gains. 

**Abstract (ZH)**: larg语言模型（LLMs）在高资源语言上表现出色，但在低资源语言上由于训练数据有限而面临挑战。本文提出了TALL（用于增强低资源语言LLM性能的可训练架构），该架构将LLM与两个双语翻译模型相结合。TALL将低资源输入转换为高资源表示，通过维度对齐层和自定义变压器保留语言特征。我们在希伯来语上的实验表明，TALL在多个基线方法（包括直接使用、简单翻译和微调方法）上取得了显著的性能提升。该架构采用了参数高效的策略，冻结预训练组件，仅训练轻量级适配器模块，平衡了计算效率和性能提升。 

---
# From Struggle (06-2024) to Mastery (02-2025) LLMs Conquer Advanced Algorithm Exams and Pave the Way for Editorial Generation 

**Title (ZH)**: 从挣扎（06-2024）到精通（02-2025）：大型语言模型征服高级算法考试并为编辑生成铺平道路 

**Authors**: Adrian Marius Dumitran, Theodor-Pierre Moroianu, Vasile Paul Alexe  

**Link**: [PDF](https://arxiv.org/pdf/2506.04965)  

**Abstract**: This paper presents a comprehensive evaluation of the performance of state-of-the-art Large Language Models (LLMs) on challenging university-level algorithms exams. By testing multiple models on both a Romanian exam and its high-quality English translation, we analyze LLMs' problem-solving capabilities, consistency, and multilingual performance. Our empirical study reveals that the most recent models not only achieve scores comparable to top-performing students but also demonstrate robust reasoning skills on complex, multi-step algorithmic challenges, even though difficulties remain with graph-based tasks. Building on these findings, we explore the potential of LLMs to support educational environments through the generation of high-quality editorial content, offering instructors a powerful tool to enhance student feedback. The insights and best practices discussed herein pave the way for further integration of generative AI in advanced algorithm education. 

**Abstract (ZH)**: 本研究全面评估了最先进大型语言模型在大学级算法考试中的 performance。通过在罗马尼亚考题及其高质量英语翻译版本上测试多个模型，我们分析了语言模型的问题解决能力、一致性和多语言性能。我们的实证研究显示，最新的模型不仅在成绩上可与顶尖学生媲美，还在复杂多步算法挑战中展示了稳健的推理能力，尽管在图任务方面仍有困难。基于这些发现，我们探讨了大型语言模型在通过生成高质量编辑内容支持教育环境方面的潜力，为教师提供了一种增强学生反馈的强大工具。文中讨论的见解和最佳实践为生成式AI在高级算法教育中的进一步整合铺平了道路。 

---
# Simulating LLM-to-LLM Tutoring for Multilingual Math Feedback 

**Title (ZH)**: 多语言数学反馈的大型语言模型到大型语言模型辅导模拟 

**Authors**: Junior Cedric Tonga, KV Aditya Srivatsa, Kaushal Kumar Maurya, Fajri Koto, Ekaterina Kochmar  

**Link**: [PDF](https://arxiv.org/pdf/2506.04920)  

**Abstract**: Large language models (LLMs) have demonstrated the ability to generate formative feedback and instructional hints in English, making them increasingly relevant for AI-assisted education. However, their ability to provide effective instructional support across different languages, especially for mathematically grounded reasoning tasks, remains largely unexamined. In this work, we present the first large-scale simulation of multilingual tutor-student interactions using LLMs. A stronger model plays the role of the tutor, generating feedback in the form of hints, while a weaker model simulates the student. We explore 352 experimental settings across 11 typologically diverse languages, four state-of-the-art LLMs, and multiple prompting strategies to assess whether language-specific feedback leads to measurable learning gains. Our study examines how student input language, teacher feedback language, model choice, and language resource level jointly influence performance. Results show that multilingual hints can significantly improve learning outcomes, particularly in low-resource languages when feedback is aligned with the student's native language. These findings offer practical insights for developing multilingual, LLM-based educational tools that are both effective and inclusive. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在英语中展示了生成形成性反馈和教学提示的能力，使其在人工智能辅助教育中日益相关。然而，它们在不同语言，特别是与数学推理任务相关的教学支持方面的能力仍 largely unexamined. 在这项工作中，我们使用大规模语言模型首次进行了多语言 tutor-student 互动的大规模仿真。更强的语言模型扮演导师角色，生成以提示形式的反馈，较弱的语言模型模拟学生。我们在11种类型学上不同的语言、四种最先进的大规模语言模型和多种提示策略下探索了352种实验设置，以评估语言特定的反馈是否能够带来可衡量的学习成效。本研究考察了学生输入语言、教师反馈语言、模型选择和语言资源级别如何共同影响性能。结果表明，多语言提示在反馈与学生母语对齐时，特别是在低资源语言中，可以显著提高学习成果。这些发现为开发既有效又包容的大规模语言模型为基础的多语言教育工具提供了实用见解。 

---
# Verbose ListOps (VLO): Beyond Long Context -- Unmasking LLM's Reasoning Blind Spots 

**Title (ZH)**: Verbose ListOps (VLO): 超越长上下文——揭示大模型的推理盲点 

**Authors**: Alex Pan, Mary-Anne Williams  

**Link**: [PDF](https://arxiv.org/pdf/2506.04907)  

**Abstract**: Large Language Models (LLMs), whilst great at extracting facts from text, struggle with nested narrative reasoning. Existing long context and multi-hop QA benchmarks inadequately test this, lacking realistic distractors or failing to decouple context length from reasoning complexity, masking a fundamental LLM limitation. We introduce Verbose ListOps, a novel benchmark that programmatically transposes ListOps computations into lengthy, coherent stories. This uniquely forces internal computation and state management of nested reasoning problems by withholding intermediate results, and offers fine-grained controls for both narrative size \emph{and} reasoning difficulty. Whilst benchmarks like LongReason (2025) advance approaches for synthetically expanding the context size of multi-hop QA problems, Verbose ListOps pinpoints a specific LLM vulnerability: difficulty in state management for nested sub-reasoning amongst semantically-relevant, distracting narrative. Our experiments show that leading LLMs (e.g., OpenAI o4, Gemini 2.5 Pro) collapse in performance on Verbose ListOps at modest (~10k token) narrative lengths, despite effortlessly solving raw ListOps equations. Addressing this failure is paramount for real-world text interpretation which requires identifying key reasoning points, tracking conceptual intermediate results, and filtering irrelevant information. Verbose ListOps, and its extensible generation framework thus enables targeted reasoning enhancements beyond mere context-window expansion; a critical step to automating the world's knowledge work. 

**Abstract (ZH)**: 大型语言模型在处理嵌套叙事推理时表现出色于提取事实，但存在困难。现有的长上下文和多跳问答基准测试不足，缺乏现实的干扰项或未能分离上下文长度与推理复杂度，掩盖了模型的基本限制。我们引入了Verbose ListOps这一新颖基准，通过编程方式将ListOps计算转换为长长的连贯故事。这一基准独特地强制内嵌推理和状态管理，通过保留中间结果来处理嵌套推理问题，并提供了对叙事规模和推理难度的精细控制。虽然像LongReason这样的基准推进了合成扩展多跳问答题目上下文大小的方法，Verbose ListOps指出了LLM的一个特定脆弱性：在相关但分散的叙事中管理嵌套子推理的状态难度。我们的实验表明，即使是领先的LLM（例如，OpenAI o4、Gemini 2.5 Pro）在Verbose ListOps下在适度长度（约10k词）的故事中表现不佳，尽管它们能够轻易解决原始的ListOps方程。解决这一失败对于需要识别关键推理点、追踪概念性中间结果和过滤无关信息的实际文本解释至关重要。Verbose ListOps及其可扩展生成框架因此能够实现超越简单上下文窗扩展的目标推理增强；这是将世界知识工作自动化的一个关键步骤。 

---
# Multiple-Choice Question Generation Using Large Language Models: Methodology and Educator Insights 

**Title (ZH)**: 使用大型语言模型生成多项选择题：方法与教育者见解 

**Authors**: Giorgio Biancini, Alessio Ferrato, Carla Limongelli  

**Link**: [PDF](https://arxiv.org/pdf/2506.04851)  

**Abstract**: Integrating Artificial Intelligence (AI) in educational settings has brought new learning approaches, transforming the practices of both students and educators. Among the various technologies driving this transformation, Large Language Models (LLMs) have emerged as powerful tools for creating educational materials and question answering, but there are still space for new applications. Educators commonly use Multiple-Choice Questions (MCQs) to assess student knowledge, but manually generating these questions is resource-intensive and requires significant time and cognitive effort. In our opinion, LLMs offer a promising solution to these challenges. This paper presents a novel comparative analysis of three widely known LLMs - Llama 2, Mistral, and GPT-3.5 - to explore their potential for creating informative and challenging MCQs. In our approach, we do not rely on the knowledge of the LLM, but we inject the knowledge into the prompt to contrast the hallucinations, giving the educators control over the test's source text, too. Our experiment involving 21 educators shows that GPT-3.5 generates the most effective MCQs across several known metrics. Additionally, it shows that there is still some reluctance to adopt AI in the educational field. This study sheds light on the potential of LLMs to generate MCQs and improve the educational experience, providing valuable insights for the future. 

**Abstract (ZH)**: 将人工智能集成到教育环境中带来了新的学习方法，转变了学生和教育者的实践方式。在推动这一转变的各种技术中，大型语言模型(Large Language Models, LLMs)已成为创建教育材料和问答的强大工具，但仍有新的应用空间。教育者通常使用选择题（Multiple-Choice Questions, MCQs）来评估学生知识，但手工生成这些题目耗时且需要大量的认知努力。我们认为，LLMs为解决这些挑战提供了可行的解决方案。本文介绍了对三种广泛known的LLMs——Llama 2、Mistral和GPT-3.5——的新型比较分析，以探索它们创建信息性和挑战性选择题的潜力。在我们的方法中，我们并未依赖LLMs的知识，而是将知识注入提示中以对比其生成的内容，使教育者也能够控制测试的源文本。我们的实验涉及21名教育者，结果显示GPT-3.5在多个已知指标下生成了最有效的选择题。此外，该研究还显示教育领域仍然存在不愿采用AI的现象。本研究为LLMs在生成选择题和改善教育体验方面的潜力提供了有价值的见解，并为未来的应用提供了指导。 

---
# On Automating Security Policies with Contemporary LLMs 

**Title (ZH)**: 使用当代大语言模型自动化安全策略 

**Authors**: Pablo Fernández Saura, K. R. Jayaram, Vatche Isahagian, Jorge Bernal Bernabé, Antonio Skarmeta  

**Link**: [PDF](https://arxiv.org/pdf/2506.04838)  

**Abstract**: The complexity of modern computing environments and the growing sophistication of cyber threats necessitate a more robust, adaptive, and automated approach to security enforcement. In this paper, we present a framework leveraging large language models (LLMs) for automating attack mitigation policy compliance through an innovative combination of in-context learning and retrieval-augmented generation (RAG). We begin by describing how our system collects and manages both tool and API specifications, storing them in a vector database to enable efficient retrieval of relevant information. We then detail the architectural pipeline that first decomposes high-level mitigation policies into discrete tasks and subsequently translates each task into a set of actionable API calls. Our empirical evaluation, conducted using publicly available CTI policies in STIXv2 format and Windows API documentation, demonstrates significant improvements in precision, recall, and F1-score when employing RAG compared to a non-RAG baseline. 

**Abstract (ZH)**: 现代计算环境的复杂性和网络威胁的日益 sophistication 要求采用一种更 robust、更适应环境并且自动化的安全执行方法。在本文中，我们提出了一种利用大规模语言模型 (LLMs) 通过创新的上下文学习和检索增强生成 (RAG) 结合来自动实现攻击缓解策略合规性的框架。我们首先描述了系统如何收集和管理工具和 API 规范，并将它们存储在向量数据库中以实现相关信息的高效检索。然后，我们详细说明了架构管道，该管道首先将高层级的缓解策略分解为离散任务，随后将每个任务转化为一组可执行的 API 调用。我们的实证评估使用公开的 STIXv2 格式的 CTI 策略和 Windows API 文档表明，在使用 RAG 时，相比非 RAG 基线，精确度、召回率和 F1 分数显著提高。 

---
# A Reasoning-Based Approach to Cryptic Crossword Clue Solving 

**Title (ZH)**: 基于推理的隐晦填字谜线索解答方法 

**Authors**: Martin Andrews, Sam Witteveen  

**Link**: [PDF](https://arxiv.org/pdf/2506.04824)  

**Abstract**: Cryptic crossword clues are challenging language tasks for which new test sets are released daily by major newspapers on a global basis. Each cryptic clue contains both the definition of the answer to be placed in the crossword grid (in common with regular crosswords), and 'wordplay' that proves that the answer is correct (i.e. a human solver can be confident that an answer is correct without needing crossing words as confirmation). This work describes an LLM-based reasoning system built from open-licensed components that solves cryptic clues by (i) hypothesising answers; (ii) proposing wordplay explanations; and (iii) using a verifier system that operates on codified reasoning steps. Overall, this system establishes a new state-of-the-art performance on the challenging Cryptonite dataset of clues from The Times and The Telegraph newspapers in the UK. Because each proved solution is expressed in Python, interpretable wordplay reasoning for proven answers is available for inspection. 

**Abstract (ZH)**: 全球主要报纸每日发布的隐谜 crossword 答题挑战：一种基于开源组件的大型语言模型推理系统及其在 Cryptonite 数据集上的最新性能 

---
# Dissecting Logical Reasoning in LLMs: A Fine-Grained Evaluation and Supervision Study 

**Title (ZH)**: 剖析大语言模型中的逻辑推理：一种精细粒度的评估与监督研究 

**Authors**: Yujun Zhou, Jiayi Ye, Zipeng Ling, Yufei Han, Yue Huang, Haomin Zhuang, Zhenwen Liang, Kehan Guo, Taicheng Guo, Xiangqi Wang, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04810)  

**Abstract**: Logical reasoning is a core capability for many applications of large language models (LLMs), yet existing benchmarks often rely solely on final-answer accuracy, failing to capture the quality and structure of the reasoning process. We propose FineLogic, a fine-grained evaluation framework that assesses logical reasoning across three dimensions: overall benchmark accuracy, stepwise soundness, and representation-level alignment. In addition, to better understand how reasoning capabilities emerge, we conduct a comprehensive study on the effects of supervision format during fine-tuning. We construct four supervision styles (one natural language and three symbolic variants) and train LLMs under each. Our findings reveal that natural language supervision yields strong generalization even on out-of-distribution and long-context tasks, while symbolic reasoning styles promote more structurally sound and atomic inference chains. Further, our representation-level probing shows that fine-tuning primarily improves reasoning behaviors through step-by-step generation, rather than enhancing shortcut prediction or internalized correctness. Together, our framework and analysis provide a more rigorous and interpretable lens for evaluating and improving logical reasoning in LLMs. 

**Abstract (ZH)**: 细粒度逻辑推理评估框架：从监督格式探究推理能力的涌现机制及其实现 

---
# Towards LLM-Centric Multimodal Fusion: A Survey on Integration Strategies and Techniques 

**Title (ZH)**: 面向LLM的多模态融合：一种综合策略和技术的概述 

**Authors**: Jisu An, Junseok Lee, Jeoungeun Lee, Yongseok Son  

**Link**: [PDF](https://arxiv.org/pdf/2506.04788)  

**Abstract**: The rapid progress of Multimodal Large Language Models(MLLMs) has transformed the AI landscape. These models combine pre-trained LLMs with various modality encoders. This integration requires a systematic understanding of how different modalities connect to the language backbone. Our survey presents an LLM-centric analysis of current approaches. We examine methods for transforming and aligning diverse modal inputs into the language embedding space. This addresses a significant gap in existing literature. We propose a classification framework for MLLMs based on three key dimensions. First, we examine architectural strategies for modality integration. This includes both the specific integration mechanisms and the fusion level. Second, we categorize representation learning techniques as either joint or coordinate representations. Third, we analyze training paradigms, including training strategies and objective functions. By examining 125 MLLMs developed between 2021 and 2025, we identify emerging patterns in the field. Our taxonomy provides researchers with a structured overview of current integration techniques. These insights aim to guide the development of more robust multimodal integration strategies for future models built on pre-trained foundations. 

**Abstract (ZH)**: 多模态大型语言模型的快速进步已重塑人工智能格局。这些模型结合了预训练的大语言模型和各种模态编码器。这种集成需要系统理解不同模态如何连接到语言骨干。我们的综述从大语言模型的角度分析了当前的方法。我们研究了将多样化的模态输入转换和对齐到语言嵌入空间的方法。这填补了现有文献中的一个重要空白。我们根据三个关键维度提出了多模态大型语言模型的分类框架。首先，我们研究了模态集成的架构策略，包括具体的集成机制和融合层次。其次，我们将表示学习技术分类为联合表示或协调表示。第三，我们分析了训练范式，包括训练策略和目标函数。通过分析2021年至2025年间开发的125个MMLMs，我们确定了领域中的新兴模式。我们的分类框架为研究人员提供了一个结构化的概述，旨在指导未来基于预训练基础构建更具鲁棒性的多模态集成策略。 

---
# Fine-Grained Interpretation of Political Opinions in Large Language Models 

**Title (ZH)**: 大型语言模型中政治意见的细粒度解释 

**Authors**: Jingyu Hu, Mengyue Yang, Mengnan Du, Weiru Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04774)  

**Abstract**: Studies of LLMs' political opinions mainly rely on evaluations of their open-ended responses. Recent work indicates that there is a misalignment between LLMs' responses and their internal intentions. This motivates us to probe LLMs' internal mechanisms and help uncover their internal political states. Additionally, we found that the analysis of LLMs' political opinions often relies on single-axis concepts, which can lead to concept confounds. In this work, we extend the single-axis to multi-dimensions and apply interpretable representation engineering techniques for more transparent LLM political concept learning. Specifically, we designed a four-dimensional political learning framework and constructed a corresponding dataset for fine-grained political concept vector learning. These vectors can be used to detect and intervene in LLM internals. Experiments are conducted on eight open-source LLMs with three representation engineering techniques. Results show these vectors can disentangle political concept confounds. Detection tasks validate the semantic meaning of the vectors and show good generalization and robustness in OOD settings. Intervention Experiments show these vectors can intervene in LLMs to generate responses with different political leanings. 

**Abstract (ZH)**: LLM政治观点研究主要依赖于对其开放回答的评价。近期研究表明，LLM的回答与它们的内部意图存在偏差，这促使我们探索LLM的内部机制，帮助揭示其内部政治状态。此外，我们发现分析LLM的政治观点常依赖于一维概念，这可能导致概念混淆。在本工作中，我们将一维扩展到多维，并运用可解释的表示工程技术进行更透明的LLM政治概念学习。具体地，我们设计了一个四维政治学习框架，并构建了一个相应的数据集用于细粒度政治概念向量学习。这些向量可用于检测和干预LLM的内部。实验在八种开源LLM上进行，使用三种表示工程技术。结果表明这些向量能够分离政治概念混淆。检测任务验证了这些向量的语义意义，并展示了在OOD设置下的良好泛化能力和鲁棒性。干预实验显示这些向量能够干预LLM生成不同政治倾向的回答。 

---
# Lifelong Evolution: Collaborative Learning between Large and Small Language Models for Continuous Emergent Fake News Detection 

**Title (ZH)**: lifelong 进化: 大型和小型语言模型之间的合作学习以实现连续涌现式假新闻检测 

**Authors**: Ziyi Zhou, Xiaoming Zhang, Litian Zhang, Yibo Zhang, Zhenyu Guan, Chaozhuo Li, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04739)  

**Abstract**: The widespread dissemination of fake news on social media has significantly impacted society, resulting in serious consequences. Conventional deep learning methodologies employing small language models (SLMs) suffer from extensive supervised training requirements and difficulties adapting to evolving news environments due to data scarcity and distribution shifts. Large language models (LLMs), despite robust zero-shot capabilities, fall short in accurately detecting fake news owing to outdated knowledge and the absence of suitable demonstrations. In this paper, we propose a novel Continuous Collaborative Emergent Fake News Detection (C$^2$EFND) framework to address these challenges. The C$^2$EFND framework strategically leverages both LLMs' generalization power and SLMs' classification expertise via a multi-round collaborative learning framework. We further introduce a lifelong knowledge editing module based on a Mixture-of-Experts architecture to incrementally update LLMs and a replay-based continue learning method to ensure SLMs retain prior knowledge without retraining entirely. Extensive experiments on Pheme and Twitter16 datasets demonstrate that C$^2$EFND significantly outperforms existed methods, effectively improving detection accuracy and adaptability in continuous emergent fake news scenarios. 

**Abstract (ZH)**: 大规模社交媒体上传播的假新闻对社会产生了重大影响， conventional deep learning 方法依赖小型语言模型（SLMs）由于监督训练需求大且难以适应不断变化的新闻环境，即由于数据稀缺和分布变化。尽管大型语言模型（LLMs）具有强大的零样本能力，但在检测假新闻时由于知识过时和缺乏合适的示例而表现不佳。本文提出了一种新的连续协作 emergent 假新闻检测（C$^2$EFND）框架，该框架通过多轮协作学习框架战略地利用了 LLMs 的泛化能力和 SLMs 的分类专长。本文还介绍了一种基于 Mixture-of-Experts 架构的终身知识编辑模块，用于增量更新 LLMs，并提出了基于回放的持续学习方法以确保 SLMs 在无需完全重新训练的情况下保留先前知识。在 Pheme 和 Twitter16 数据集上的广泛实验表明，C$^2$EFND 显著优于现有方法，在持续 emergent 假新闻检测场景中有效提高了检测准确性和适应性。 

---
# Line of Sight: On Linear Representations in VLLMs 

**Title (ZH)**: 视觉射线：关于VLLMs中的线性表示 

**Authors**: Achyuta Rajaram, Sarah Schwettmann, Jacob Andreas, Arthur Conmy  

**Link**: [PDF](https://arxiv.org/pdf/2506.04706)  

**Abstract**: Language models can be equipped with multimodal capabilities by fine-tuning on embeddings of visual inputs. But how do such multimodal models represent images in their hidden activations? We explore representations of image concepts within LlaVA-Next, a popular open-source VLLM. We find a diverse set of ImageNet classes represented via linearly decodable features in the residual stream. We show that the features are causal by performing targeted edits on the model output. In order to increase the diversity of the studied linear features, we train multimodal Sparse Autoencoders (SAEs), creating a highly interpretable dictionary of text and image features. We find that although model representations across modalities are quite disjoint, they become increasingly shared in deeper layers. 

**Abstract (ZH)**: 多模态模型中的图像概念表示：以LlaVA-Next为例 

---
# On the Mechanism of Reasoning Pattern Selection in Reinforcement Learning for Language Models 

**Title (ZH)**: reinforcement learning for language models 中文标题可以翻译为：基于强化学习的语言模型推理模式选择机制 

**Authors**: Xingwu Chen, Tianle Li, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2506.04695)  

**Abstract**: Reinforcement learning (RL) has demonstrated remarkable success in enhancing model capabilities, including instruction-following, preference learning, and reasoning. Yet despite its empirical successes, the mechanisms by which RL improves reasoning abilities remain poorly understood. We present a systematic study of Reinforcement Learning with Verifiable Rewards (RLVR), showing that its primary benefit comes from optimizing the selection of existing reasoning patterns. Through extensive experiments, we demonstrate that RLVR-trained models preferentially adopt high-success-rate reasoning patterns while mostly maintaining stable performance on individual patterns. We further develop theoretical analyses on the convergence and training dynamics of RLVR based on a simplified question-reason-answer model. We study the gradient flow and show that RLVR can indeed find the solution that selects the reason pattern with the highest success rate. Besides, our theoretical results
reveal two distinct regimes regarding the convergence of RLVR training: (1) rapid convergence for models with relatively strong initial reasoning capabilities versus (2) slower optimization dynamics for weaker models. Furthermore, we show that the slower optimization for weaker models can be mitigated by applying the supervised fine-tuning (SFT) before RLVR, when using a feasibly high-quality SFT dataset. We validate the theoretical findings through extensive experiments. This work advances our theoretical understanding of RL's role in LLM fine-tuning and offers insights for further enhancing reasoning capabilities. 

**Abstract (ZH)**: 强化学习（RL）在增强模型能力方面取得了显著成功，包括指令遵循、偏好学习和推理。尽管在实证上取得了一定的成功，但RL提高推理能力的机制仍然不甚明了。我们对可验证奖励的强化学习（RLVR）进行了系统研究，表明其主要益处在于优化现有推理模式的选择。通过大量实验，我们证明了RLVR训练的模型倾向于选择成功率较高的推理模式，同时在个体模式上的性能保持稳定。我们进一步基于简化的问题-推理-答案模型对RLVR的收敛性和训练动力学进行了理论分析。我们研究了梯度流动，并证明RLVR确实能找到选择成功率最高的推理模式的解。此外，我们的理论结果揭示了RLVR训练收敛的两种不同模式：（1）对于初始推理能力强的模型，收敛速度快；（2）对于初始推理能力较弱的模型，则优化动态较慢。我们还展示了通过在RLVR之前应用监督微调（SFT），并使用高质量的SFT数据集，可以减轻较弱模型的优化缓慢问题。我们通过大量实验验证了这些理论发现。这项工作推进了我们对RL在大语言模型微调中作用的理论理解，并为进一步增强推理能力提供了见解。 

---
# MMRefine: Unveiling the Obstacles to Robust Refinement in Multimodal Large Language Models 

**Title (ZH)**: MMRefine: 揭示多模态大型语言模型稳健精炼的障碍 

**Authors**: Gio Paik, Geewook Kim, Jinbae Im  

**Link**: [PDF](https://arxiv.org/pdf/2506.04688)  

**Abstract**: This paper introduces MMRefine, a MultiModal Refinement benchmark designed to evaluate the error refinement capabilities of Multimodal Large Language Models (MLLMs). As the emphasis shifts toward enhancing reasoning during inference, MMRefine provides a framework that evaluates MLLMs' abilities to detect and correct errors across six distinct scenarios beyond just comparing final accuracy before and after refinement. Furthermore, the benchmark analyzes the refinement performance by categorizing errors into six error types. Experiments with various open and closed MLLMs reveal bottlenecks and factors impeding refinement performance, highlighting areas for improvement in effective reasoning enhancement. Our code and dataset are publicly available at this https URL. 

**Abstract (ZH)**: 本文介绍了MMRefine，一个旨在评估多模态大型语言模型（MLLMs）错误修正能力的多模态精炼基准。随着推理过程中强化推理的重视增加，MMRefine 提供了一个评估框架，该框架不仅通过比较精炼前后最终准确性来评估MLLMs的能力，还在六个不同的场景中检测和修正错误。此外，基准通过将错误分类为六种错误类型来分析精炼性能。对各种开放和封闭的MLLMs的实验揭示了影响精炼性能的瓶颈和因素，并突出了有效推理增强的改进领域。我们的代码和数据集可在以下链接公开访问：this https URL。 

---
# Urania: Differentially Private Insights into AI Use 

**Title (ZH)**: Urania：AI应用的差异化隐私见解 

**Authors**: Daogao Liu, Edith Cohen, Badih Ghazi, Peter Kairouz, Pritish Kamath, Alexander Knop, Ravi Kumar, Pasin Manurangsi, Adam Sealfon, Da Yu, Chiyuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04681)  

**Abstract**: We introduce $Urania$, a novel framework for generating insights about LLM chatbot interactions with rigorous differential privacy (DP) guarantees. The framework employs a private clustering mechanism and innovative keyword extraction methods, including frequency-based, TF-IDF-based, and LLM-guided approaches. By leveraging DP tools such as clustering, partition selection, and histogram-based summarization, $Urania$ provides end-to-end privacy protection. Our evaluation assesses lexical and semantic content preservation, pair similarity, and LLM-based metrics, benchmarking against a non-private Clio-inspired pipeline (Tamkin et al., 2024). Moreover, we develop a simple empirical privacy evaluation that demonstrates the enhanced robustness of our DP pipeline. The results show the framework's ability to extract meaningful conversational insights while maintaining stringent user privacy, effectively balancing data utility with privacy preservation. 

**Abstract (ZH)**: 我们介绍了一种名为$Urania$的新框架，该框架能够在严格差异隐私（DP）保证下生成大规模语言模型（LLM）聊天机器人交互的洞察。该框架采用了一种私有聚类机制和创新的关键词提取方法，包括基于频率、基于TF-IDF和基于LLM的方法。通过利用聚类、分区选择和基于直方图的总结等差异隐私工具，$Urania$提供端到端的隐私保护。我们的评估从词汇和语义内容保留、配对相似性和基于LLM的指标等方面进行，与非私有克利奥启发式管道（Tamkin等，2024）进行基准测试。此外，我们开发了一种简单的经验隐私评估方法，展示了我们差异隐私管道的增强鲁棒性。结果显示，该框架能够在严格保护用户隐私的同时提取有意义的对话洞察，有效平衡数据利用与隐私保护。 

---
# Gen-n-Val: Agentic Image Data Generation and Validation 

**Title (ZH)**: Gen-n-Val: 自主图像数据生成与验证 

**Authors**: Jing-En Huang, I-Sheng Fang, Tzuhsuan Huang, Chih-Yu Wang, Jun-Cheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.04676)  

**Abstract**: Recently, Large Language Models (LLMs) and Vision Large Language Models (VLLMs) have demonstrated impressive performance as agents across various tasks while data scarcity and label noise remain significant challenges in computer vision tasks, such as object detection and instance segmentation. A common solution for resolving these issues is to generate synthetic data. However, current synthetic data generation methods struggle with issues, such as multiple objects per mask, inaccurate segmentation, and incorrect category labels, limiting their effectiveness. To address these issues, we introduce Gen-n-Val, a novel agentic data generation framework that leverages Layer Diffusion (LD), LLMs, and VLLMs to produce high-quality, single-object masks and diverse backgrounds. Gen-n-Val consists of two agents: (1) The LD prompt agent, an LLM, optimizes prompts for LD to generate high-quality foreground instance images and segmentation masks. These optimized prompts ensure the generation of single-object synthetic data with precise instance masks and clean backgrounds. (2) The data validation agent, a VLLM, which filters out low-quality synthetic instance images. The system prompts for both agents are refined through TextGrad. Additionally, we use image harmonization to combine multiple instances within scenes. Compared to state-of-the-art synthetic data approaches like MosaicFusion, our approach reduces invalid synthetic data from 50% to 7% and improves performance by 1% mAP on rare classes in COCO instance segmentation with YOLOv9c and YOLO11m. Furthermore, Gen-n-Val shows significant improvements (7. 1% mAP) over YOLO-Worldv2-M in open-vocabulary object detection benchmarks with YOLO11m. Moreover, Gen-n-Val improves the performance of YOLOv9 and YOLO11 families in instance segmentation and object detection. 

**Abstract (ZH)**: Gen-n-Val：一种基于Layer Diffusion的高质量合成数据生成框架 

---
# Scaling Laws for Robust Comparison of Open Foundation Language-Vision Models and Datasets 

**Title (ZH)**: 开放基础语言-视觉模型和数据集稳健比较的标度律 

**Authors**: Marianna Nezhurina, Tomer Porian, Giovanni Pucceti, Tommie Kerssies, Romain Beaumont, Mehdi Cherti, Jenia Jitsev  

**Link**: [PDF](https://arxiv.org/pdf/2506.04598)  

**Abstract**: In studies of transferable learning, scaling laws are obtained for various important foundation models to predict their properties and performance at larger scales. We show here how scaling law derivation can also be used for model and dataset comparison, allowing to decide which procedure is to be preferred for pre-training. For the first time, full scaling laws based on dense measurements across a wide span of model and samples seen scales are derived for two important language-vision learning procedures, CLIP and MaMMUT, that use either contrastive only or contrastive and captioning text generative loss. Ensuring sufficient prediction accuracy for held out points, we use derived scaling laws to compare both models, obtaining evidence for MaMMUT's stronger improvement with scale and better sample efficiency than standard CLIP. To strengthen validity of the comparison, we show scaling laws for various downstream tasks, classification, retrieval, and segmentation, and for different open datasets, DataComp, DFN and Re-LAION, observing consistently the same trends. We show that comparison can also be performed when deriving scaling laws with a constant learning rate schedule, reducing compute cost. Accurate derivation of scaling laws provides thus means to perform model and dataset comparison across scale spans, avoiding misleading conclusions based on measurements from single reference scales only, paving the road for systematic comparison and improvement of open foundation models and datasets for their creation. We release all the pre-trained models with their intermediate checkpoints, including openMaMMUT-L/14, which achieves $80.3\%$ zero-shot ImageNet-1k accuracy, trained on 12.8B samples from DataComp-1.4B. Code for reproducing experiments in the paper and raw experiments data can be found at this https URL. 

**Abstract (ZH)**: 在迁移学习研究中，通过建立各种重要基础模型的标度律来预测其在更大尺度下的属性和性能。我们展示了如何利用标度律推导方法来进行模型和数据集的比较，以便决定预训练时应优先采用哪种 procedure。首次基于广泛范围的模型和样本观测尺度，推导了两种重要的语言-视觉学习 procedure（CLIP 和 MaMMUT）的完整标度律，这两种 procedure 分别使用对比或对比和配图文本生成损失。通过确保对外预测点足够的准确性，我们使用推导出的标度律比较了两种模型，提供了 MaMMUT 在规模上表现出更强改进和更好样本效率的证据。为增强比较的可靠性，我们展示了在多种下游任务（分类、检索和分割）以及不同开源数据集（DataComp、DFN 和 Re-LAION）上推导出的标度律，观察到一致的趋势。我们展示了即使在使用恒定学习率调度推导标度律时也能进行比较，从而减少了计算成本。准确推导出的标度律提供了在不同尺度范围内比较模型和数据集的方法，避免了仅基于单一参考尺度测量得出错误结论的情况，为系统比较和改进开放基础模型和数据集奠定了道路。我们发布了所有预训练模型及其中间检查点，包括 openMaMMUT-L/14，该模型在 128亿个 DataComp-1.4B 样本上训练，实现了80.3% 的零样本 ImageNet-1k 准确率。论文中的实验代码和原始实验数据可在此处找到。 

---
# Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification 

**Title (ZH)**: Safe: 通过回顾性的步骤感知形式验证增强大型语言模型的数学推理能力 

**Authors**: Chengwu Liu, Ye Yuan, Yichun Yin, Yan Xu, Xin Xu, Zaoyu Chen, Yasheng Wang, Lifeng Shang, Qun Liu, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04592)  

**Abstract**: Chain-of-Thought (CoT) prompting has become the de facto method to elicit reasoning capabilities from large language models (LLMs). However, to mitigate hallucinations in CoT that are notoriously difficult to detect, current methods such as process reward models (PRMs) or self-consistency operate as opaque boxes and do not provide checkable evidence for their judgments, possibly limiting their effectiveness. To address this issue, we draw inspiration from the idea that "the gold standard for supporting a mathematical claim is to provide a proof". We propose a retrospective, step-aware formal verification framework $Safe$. Rather than assigning arbitrary scores, we strive to articulate mathematical claims in formal mathematical language Lean 4 at each reasoning step and provide formal proofs to identify hallucinations. We evaluate our framework $Safe$ across multiple language models and various mathematical datasets, demonstrating a significant performance improvement while offering interpretable and verifiable evidence. We also propose $FormalStep$ as a benchmark for step correctness theorem proving with $30,809$ formal statements. To the best of our knowledge, our work represents the first endeavor to utilize formal mathematical language Lean 4 for verifying natural language content generated by LLMs, aligning with the reason why formal mathematical languages were created in the first place: to provide a robust foundation for hallucination-prone human-written proofs. 

**Abstract (ZH)**: Safe：使用形式化数学语言验证大型语言模型生成内容的回顾性步骤感知形式化验证框架 

---
# Reasoning or Overthinking: Evaluating Large Language Models on Financial Sentiment Analysis 

**Title (ZH)**: 理性思考还是过度思考：大型语言模型在金融情感分析中的评估 

**Authors**: Dimitris Vamvourellis, Dhagash Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2506.04574)  

**Abstract**: We investigate the effectiveness of large language models (LLMs), including reasoning-based and non-reasoning models, in performing zero-shot financial sentiment analysis. Using the Financial PhraseBank dataset annotated by domain experts, we evaluate how various LLMs and prompting strategies align with human-labeled sentiment in a financial context. We compare three proprietary LLMs (GPT-4o, GPT-4.1, o3-mini) under different prompting paradigms that simulate System 1 (fast and intuitive) or System 2 (slow and deliberate) thinking and benchmark them against two smaller models (FinBERT-Prosus, FinBERT-Tone) fine-tuned on financial sentiment analysis. Our findings suggest that reasoning, either through prompting or inherent model design, does not improve performance on this task. Surprisingly, the most accurate and human-aligned combination of model and method was GPT-4o without any Chain-of-Thought (CoT) prompting. We further explore how performance is impacted by linguistic complexity and annotation agreement levels, uncovering that reasoning may introduce overthinking, leading to suboptimal predictions. This suggests that for financial sentiment classification, fast, intuitive "System 1"-like thinking aligns more closely with human judgment compared to "System 2"-style slower, deliberative reasoning simulated by reasoning models or CoT prompting. Our results challenge the default assumption that more reasoning always leads to better LLM decisions, particularly in high-stakes financial applications. 

**Abstract (ZH)**: 我们调查了大型语言模型（LLMs），包括基于推理和非基于推理的模型，在执行零样本财务情感分析方面的有效性。使用领域专家标注的Financial PhraseBank数据集，我们评估了各种LLMs和提示策略与人类标注的情感在财务背景下的一致性。我们在不同的提示范式下比较了三种专有LLMs（GPT-4o、GPT-4.1、o3-mini），这些范式模拟了系统1（快速直观）或系统2（缓慢审慎）的思考方式，并将它们与两种较小的模型（FinBERT-Prosus、FinBERT-Tone）进行了对比，后者是专门针对财务情感分析微调的。我们的研究发现，无论是通过提示还是模型本身的结构，推理并未提高这项任务的表现。令人意外的是，最准确且与人类情感相一致的组合是无需任何推理链（CoT）提示的GPT-4o。我们进一步探讨了语言复杂性和注释一致性水平对表现的影响，发现推理可能会导致过度思考，从而导致次优预测。这表明，在财务情感分类中，快速直观的“系统1”类型的思考与人类判断更为一致，而非由推理模型或推理链提示模拟的“系统2”类型的缓慢审慎推理。我们的结果挑战了更多推理总是会导致更好的LLM决策的默认假设，特别是在高风险的财务应用中。 

---
# Clustering and Median Aggregation Improve Differentially Private Inference 

**Title (ZH)**: 聚类和中位数聚合改进差异隐私推断 

**Authors**: Kareem Amin, Salman Avestimehr, Sara Babakniya, Alex Bie, Weiwei Kong, Natalia Ponomareva, Umar Syed  

**Link**: [PDF](https://arxiv.org/pdf/2506.04566)  

**Abstract**: Differentially private (DP) language model inference is an approach for generating private synthetic text. A sensitive input example is used to prompt an off-the-shelf large language model (LLM) to produce a similar example. Multiple examples can be aggregated together to formally satisfy the DP guarantee.
Prior work creates inference batches by sampling sensitive inputs uniformly at random. We show that uniform sampling degrades the quality of privately generated text, especially when the sensitive examples concern heterogeneous topics.
We remedy this problem by clustering the input data before selecting inference batches. Next, we observe that clustering also leads to more similar next-token predictions across inferences. We use this insight to introduce a new algorithm that aggregates next token statistics by privately computing medians instead of averages. This approach leverages the fact that the median has decreased local sensitivity when next token predictions are similar, allowing us to state a data-dependent and ex-post DP guarantee about the privacy properties of this algorithm. Finally, we demonstrate improvements in terms of representativeness metrics (e.g., MAUVE) as well as downstream task performance. We show that our method produces high-quality synthetic data at significantly lower privacy cost than a previous state-of-the-art method. 

**Abstract (ZH)**: 不同隐私保护的语言模型推理是一种生成私人合成文本的方法。通过使用敏感输入示例来提示现成的大语言模型生成类似示例。多个示例可以结合起来正式满足差分隐私保证。

先前的工作通过均匀随机采样敏感输入来创建推理批次。我们展示了均匀采样会降低私人生成文本的质量，特别是在敏感示例涉及异构主题时。

为此，我们在选择推理批次之前对输入数据进行聚类。我们还观察到，聚类导致了更相似的后续标记预测。我们利用这一洞察引入了一种新算法，通过私有计算中位数而不是均值来聚合后续标记统计数据。该方法利用了当后续标记预测相似时中位数局部敏感性较低的事实，使我们能够针对该算法的隐私属性陈述数据依赖且事后差分隐私保证。最后，我们在表示性指标（如MAUVE）以及下游任务性能方面展示了改进。我们展示了我们的方法以显著较低的隐私成本生成高质量的合成数据，优于先前的最佳方法。 

---
# SSA-COMET: Do LLMs Outperform Learned Metrics in Evaluating MT for Under-Resourced African Languages? 

**Title (ZH)**: SSA-COMET：低资源非洲语言机器翻译评估中大型语言模型是否优于学习到的指标？ 

**Authors**: Senyu Li, Jiayi Wang, Felermino D. M. A. Ali, Colin Cherry, Daniel Deutsch, Eleftheria Briakou, Rui Sousa-Silva, Henrique Lopes Cardoso, Pontus Stenetorp, David Ifeoluwa Adelani  

**Link**: [PDF](https://arxiv.org/pdf/2506.04557)  

**Abstract**: Evaluating machine translation (MT) quality for under-resourced African languages remains a significant challenge, as existing metrics often suffer from limited language coverage and poor performance in low-resource settings. While recent efforts, such as AfriCOMET, have addressed some of the issues, they are still constrained by small evaluation sets, a lack of publicly available training data tailored to African languages, and inconsistent performance in extremely low-resource scenarios. In this work, we introduce SSA-MTE, a large-scale human-annotated MT evaluation (MTE) dataset covering 13 African language pairs from the News domain, with over 63,000 sentence-level annotations from a diverse set of MT systems. Based on this data, we develop SSA-COMET and SSA-COMET-QE, improved reference-based and reference-free evaluation metrics. We also benchmark prompting-based approaches using state-of-the-art LLMs like GPT-4o and Claude. Our experimental results show that SSA-COMET models significantly outperform AfriCOMET and are competitive with the strongest LLM (Gemini 2.5 Pro) evaluated in our study, particularly on low-resource languages such as Twi, Luo, and Yoruba. All resources are released under open licenses to support future research. 

**Abstract (ZH)**: 评估低资源非洲语言机器翻译质量仍是一项重大挑战：现有指标往往语言覆盖面有限且在低资源环境中的表现不佳。虽然如AfriCOMET等近期努力解决了部分问题，但仍然受限于小规模的评估集、面向非洲语言的高质量训练数据不足以及在极端低资源场景中的不一致表现。在本工作中，我们引入了SSA-MTE，这是一个大规模人工注释的机器翻译评估（MTE）数据集，涵盖了13种非洲语言在新闻领域的语言对，包含超过63,000个句子级别的注释，覆盖了多种机器翻译系统。基于此数据，我们开发了SSA-COMET和SSA-COMET-QE，改进的基于参考和非基于参考的评估指标。我们还使用最新的大型语言模型（如GPT-4o和Claude）测试了提示方法。实验结果表明，SSA-COMET模型显著优于AfriCOMET，并且在包括特威语、卢奥语和约鲁巴语等低资源语言方面与我们研究中评估的最强大型语言模型（Gemini 2.5 Pro）竞争。所有资源均在开放许可证下发布，以支持未来的研究。 

---
# hdl2v: A Code Translation Dataset for Enhanced LLM Verilog Generation 

**Title (ZH)**: hdl2v：一种代码转换数据集，用于增强LLM Verilog生成 

**Authors**: Charles Hong, Brendan Roberts, Huijae An, Alex Um, Advay Ratan, Yakun Sophia Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.04544)  

**Abstract**: Large language models (LLMs) are playing an increasingly large role in domains such as code generation, including hardware code generation, where Verilog is the key language. However, the amount of publicly available Verilog code pales in comparison to the amount of code available for software languages like Python. In this work, we present hdl2v ("HDL-to-Verilog"), a dataset which seeks to increase the amount of available human-written Verilog data by translating or compiling three other hardware description languages - VHDL, Chisel, and PyMTL3 - to Verilog. Furthermore, we demonstrate the value of hdl2v in enhancing LLM Verilog generation by improving performance of a 32 billion-parameter open-weight model by up to 23% (pass@10) in VerilogEvalV2, without utilizing any data augmentation or knowledge distillation from larger models. We also show hdl2v's ability to boost the performance of a data augmentation-based fine-tuning approach by 63%. Finally, we characterize and analyze our dataset to better understand which characteristics of HDL-to-Verilog datasets can be expanded upon in future work for even better performance. 

**Abstract (ZH)**: 大型语言模型（LLMs）在代码生成领域的作用越来越重要，包括硬件代码生成，其中Verilog是关键语言。然而，公开可用的Verilog代码数量远不及软件语言如Python的代码数量。在本工作中，我们介绍了hdl2v（HDL-to-Verilog）数据集，旨在通过将VHDL、Chisel和PyMTL3等三种硬件描述语言转换为Verilog，增加可用的由人类编写的Verilog数据量。此外，我们展示了hdl2v在提高LLM生成Verilog代码的性能方面的价值，通过在VerilogEvalV2中的表现，提高了具有320亿参数的开源权重模型的性能高达23%（pass@10），而无需使用任何数据增强或来自更大模型的知识蒸馏。我们还展示了hdl2v在基于数据增强的微调方法中的性能提升能力，提高了63%。最后，我们对数据集进行了表征和分析，以便更好地理解HDL-to-Verilog数据集哪些特性可以在未来工作中扩展，以获得更优的性能。 

---
# Is It JUST Semantics? A Case Study of Discourse Particle Understanding in LLMs 

**Title (ZH)**: 它是仅仅语义问题吗？大规模语言模型中话语粒子理解的案例研究 

**Authors**: William Sheffield, Kanishka Misra, Valentina Pyatkin, Ashwini Deo, Kyle Mahowald, Junyi Jessy Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.04534)  

**Abstract**: Discourse particles are crucial elements that subtly shape the meaning of text. These words, often polyfunctional, give rise to nuanced and often quite disparate semantic/discourse effects, as exemplified by the diverse uses of the particle "just" (e.g., exclusive, temporal, emphatic). This work investigates the capacity of LLMs to distinguish the fine-grained senses of English "just", a well-studied example in formal semantics, using data meticulously created and labeled by expert linguists. Our findings reveal that while LLMs exhibit some ability to differentiate between broader categories, they struggle to fully capture more subtle nuances, highlighting a gap in their understanding of discourse particles. 

**Abstract (ZH)**: 话语 particles 是微妙塑造文本意义的关键元素。这些多功能的词语产生了细微且往往相当不同的语义/话语效果，以“just”这一词语的多种用途为例（例如， exclusivity、时间性、强调）。本研究探讨了大语言模型（LLMs）区分英语“just”的细微含义的能力，这些含义是形式语义学中一个研究充分的例子，利用了由专家语言学家精心创建和标注的数据。我们的发现表明，虽然大语言模型在区分更广泛的类别时表现出一些能力，但在捕捉更微妙的差异方面仍存在困难，突显了它们在理解话语 particles 方面的不足。 

---
# BEAR: BGP Event Analysis and Reporting 

**Title (ZH)**: BEAR: BGP事件分析与报告 

**Authors**: Hanqing Li, Melania Fedeli, Vinay Kolar, Diego Klabjan  

**Link**: [PDF](https://arxiv.org/pdf/2506.04514)  

**Abstract**: The Internet comprises of interconnected, independently managed Autonomous Systems (AS) that rely on the Border Gateway Protocol (BGP) for inter-domain routing. BGP anomalies--such as route leaks and hijacks--can divert traffic through unauthorized or inefficient paths, jeopardizing network reliability and security. Although existing rule-based and machine learning methods can detect these anomalies using structured metrics, they still require experts with in-depth BGP knowledge of, for example, AS relationships and historical incidents, to interpret events and propose remediation. In this paper, we introduce BEAR (BGP Event Analysis and Reporting), a novel framework that leverages large language models (LLMs) to automatically generate comprehensive reports explaining detected BGP anomaly events. BEAR employs a multi-step reasoning process that translates tabular BGP data into detailed textual narratives, enhancing interpretability and analytical precision. To address the limited availability of publicly documented BGP anomalies, we also present a synthetic data generation framework powered by LLMs. Evaluations on both real and synthetic datasets demonstrate that BEAR achieves 100% accuracy, outperforming Chain-of-Thought and in-context learning baselines. This work pioneers an automated approach for explaining BGP anomaly events, offering valuable operational insights for network management. 

**Abstract (ZH)**: BGP事件分析与报告：利用大规模语言模型自动生成全面报告 

---
# Learning to Diagnose Privately: DP-Powered LLMs for Radiology Report Classification 

**Title (ZH)**: 学习进行私密诊断：DP-Powered LLMs在放射学报告分类中的应用 

**Authors**: Payel Bhattacharjee, Fengwei Tian, Ravi Tandon, Joseph Lo, Heidi Hanson, Geoffrey Rubin, Nirav Merchant, John Gounley  

**Link**: [PDF](https://arxiv.org/pdf/2506.04450)  

**Abstract**: Purpose: This study proposes a framework for fine-tuning large language models (LLMs) with differential privacy (DP) to perform multi-abnormality classification on radiology report text. By injecting calibrated noise during fine-tuning, the framework seeks to mitigate the privacy risks associated with sensitive patient data and protect against data leakage while maintaining classification performance. Materials and Methods: We used 50,232 radiology reports from the publicly available MIMIC-CXR chest radiography and CT-RATE computed tomography datasets, collected between 2011 and 2019. Fine-tuning of LLMs was conducted to classify 14 labels from MIMIC-CXR dataset, and 18 labels from CT-RATE dataset using Differentially Private Low-Rank Adaptation (DP-LoRA) in high and moderate privacy regimes (across a range of privacy budgets = {0.01, 0.1, 1.0, 10.0}). Model performance was evaluated using weighted F1 score across three model architectures: BERT-medium, BERT-small, and ALBERT-base. Statistical analyses compared model performance across different privacy levels to quantify the privacy-utility trade-off. Results: We observe a clear privacy-utility trade-off through our experiments on 2 different datasets and 3 different models. Under moderate privacy guarantees the DP fine-tuned models achieved comparable weighted F1 scores of 0.88 on MIMIC-CXR and 0.59 on CT-RATE, compared to non-private LoRA baselines of 0.90 and 0.78, respectively. Conclusion: Differentially private fine-tuning using LoRA enables effective and privacy-preserving multi-abnormality classification from radiology reports, addressing a key challenge in fine-tuning LLMs on sensitive medical data. 

**Abstract (ZH)**: 目的：本文提出了一种在差分隐私（DP）保护下对大规模语言模型（LLMs）进行微调的框架，用于放射学报告文本的多异常分类。通过在微调过程中注入校准噪声，该框架旨在减轻与敏感患者数据相关联的隐私风险，并防止数据泄露同时维持分类性能。材料与方法：我们使用了来自公开可用的MIMIC-CXR胸部X射线和CT-RATE计算机断层扫描数据集的50,232份放射学报告，收集时间范围为2011年至2019年。使用Differentially Private Low-Rank Adaptation（DP-LoRA）在高和中等隐私保护水平下（隐私预算分别为0.01、0.1、1.0和10.0）对LLMs进行了微调，以对MIMIC-CXR数据集中14个标签和CT-RATE数据集中18个标签进行分类。使用加权F1分数对三种模型架构（BERT-medium、BERT-small和ALBERT-base）进行了模型性能评估，并通过统计分析比较了不同隐私水平下的模型性能，以量化隐私-效用权衡。结果：通过在两个不同数据集和三种不同模型上的实验，观察到了明显的隐私-效用权衡。在中等隐私保障下，DP微调模型在MIMIC-CXR上的加权F1分数为0.88，在CT-RATE上的加权F1分数为0.59，分别比非隐私LoRA基线模型的0.90和0.78低。结论：使用LoRA进行差分隐私微调能够有效地进行放射学报告的多异常分类并保护隐私，解决了在敏感医疗数据上微调LLMs的关键挑战。 

---
# Unpacking Let Alone: Human-Scale Models Generalize to a Rare Construction in Form but not Meaning 

**Title (ZH)**: unpacking "Let Alone": 人类规模模型在形式上但非意义上泛化至一种罕见的构建 

**Authors**: Wesley Scivetti, Tatsuya Aoyama, Ethan Wilcox, Nathan Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2506.04408)  

**Abstract**: Humans have a remarkable ability to acquire and understand grammatical phenomena that are seen rarely, if ever, during childhood. Recent evidence suggests that language models with human-scale pretraining data may possess a similar ability by generalizing from frequent to rare constructions. However, it remains an open question how widespread this generalization ability is, and to what extent this knowledge extends to meanings of rare constructions, as opposed to just their forms. We fill this gap by testing human-scale transformer language models on their knowledge of both the form and meaning of the (rare and quirky) English LET-ALONE construction. To evaluate our LMs we construct a bespoke synthetic benchmark that targets syntactic and semantic properties of the construction. We find that human-scale LMs are sensitive to form, even when related constructions are filtered from the dataset. However, human-scale LMs do not make correct generalizations about LET-ALONE's meaning. These results point to an asymmetry in the current architectures' sample efficiency between language form and meaning, something which is not present in human language learners. 

**Abstract (ZH)**: 人类在童年时期罕见接触的情况下still具备获取和理解罕见语法现象的卓越能力。近期研究表明，具有人类规模预训练数据的语言模型可能通过从常见结构推广到罕见结构的方式具备类似的能 力。然而，这种推广能力的普及程度以及这种知识在罕见结构含义上的扩展程度如何仍然有待探究。我们通过在人类规模变换器语言模型上测试其关于英语LET-ALONE构造的形式和含义的知识，填补了这一空白。为了评估我们的语言模型，我们构建了一个专门的合成基准，该基准针对构造的语言形式和语义属性。我们发现，人类规模的语言模型对形式敏感，即使在过滤掉相关构造且缺乏背景的情况下也是如此。然而，人类规模的语言模型并不能正确地推断LET-ALONE构造的含义。这些结果揭示了当前模型在语言形式和意义学习效率上的不对称性，这与人类语言学习者中不存在的现象形成对比。 

---
# MedAgentGym: Training LLM Agents for Code-Based Medical Reasoning at Scale 

**Title (ZH)**: MedAgentGym: 大规模训练基于代码的医疗推理LLM代理 

**Authors**: Ran Xu, Yuchen Zhuang, Yishan Zhong, Yue Yu, Xiangru Tang, Hang Wu, May D. Wang, Peifeng Ruan, Donghan Yang, Tao Wang, Guanghua Xiao, Carl Yang, Yang Xie, Wenqi Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.04405)  

**Abstract**: We introduce MedAgentGYM, the first publicly available training environment designed to enhance coding-based medical reasoning capabilities in large language model (LLM) agents. MedAgentGYM comprises 72,413 task instances across 129 categories derived from authentic real-world biomedical scenarios. Tasks are encapsulated within executable coding environments, each featuring detailed task descriptions, interactive feedback mechanisms, verifiable ground-truth annotations, and scalable training trajectory generation. Extensive benchmarking of over 30 LLMs reveals a notable performance disparity between commercial API-based models and open-source counterparts. Leveraging MedAgentGYM, Med-Copilot-7B achieves substantial performance gains through supervised fine-tuning (+36.44%) and continued reinforcement learning (+42.47%), emerging as an affordable and privacy-preserving alternative competitive with gpt-4o. By offering both a comprehensive benchmark and accessible, expandable training resources within unified execution environments, MedAgentGYM delivers an integrated platform to develop LLM-based coding assistants for advanced biomedical research and practice. 

**Abstract (ZH)**: 我们介绍了MedAgentGYM，这是首个公开可用的训练环境，旨在提升大型语言模型（LLM）代理的基于编码的医疗推理能力。MedAgentGYM包含来自129个真实世界生物医学场景衍生的72,413个任务实例。任务封装在可执行的编码环境中，每个环境中都包含详细的任务描述、交互式反馈机制、可验证的地面真实标注以及可扩展的训练轨迹生成。对超过30个LLM的广泛基准测试表明，基于商业API的模型与开源模型之间存在显著性能差异。利用MedAgentGYM，Med-Copilot-7B通过监督微调 (+36.44%) 和持续强化学习 (+42.47%) 实现了显著的性能提升，并成为一种经济且保护隐私的替代方案，可与gpt-4o媲美。通过提供全面的基准测试和统一执行环境中可访问且可扩展的训练资源，MedAgentGYM交付了一个集成平台，用于开发基于LLM的编码助手，以推动高级生物医学研究和实践。 

---
# MELABenchv1: Benchmarking Large Language Models against Smaller Fine-Tuned Models for Low-Resource Maltese NLP 

**Title (ZH)**: MELABenchv1：将大型语言模型与较小的细调模型在低资源马耳他语NLP任务中进行基准测试 

**Authors**: Kurt Micallef, Claudia Borg  

**Link**: [PDF](https://arxiv.org/pdf/2506.04385)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across various Natural Language Processing (NLP) tasks, largely due to their generalisability and ability to perform tasks without additional training. However, their effectiveness for low-resource languages remains limited. In this study, we evaluate the performance of 55 publicly available LLMs on Maltese, a low-resource language, using a newly introduced benchmark covering 11 discriminative and generative tasks. Our experiments highlight that many models perform poorly, particularly on generative tasks, and that smaller fine-tuned models often perform better across all tasks. From our multidimensional analysis, we investigate various factors impacting performance. We conclude that prior exposure to Maltese during pre-training and instruction-tuning emerges as the most important factor. We also examine the trade-offs between fine-tuning and prompting, highlighting that while fine-tuning requires a higher initial cost, it yields better performance and lower inference costs. Through this work, we aim to highlight the need for more inclusive language technologies and recommend that researchers working with low-resource languages consider more "traditional" language modelling approaches. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种自然语言处理（NLP）任务中表现出了卓越的能力，主要是由于它们的普适性和无需额外训练就能完成任务的能力。然而，它们在低资源语言上的效果仍然有限。在本研究中，我们使用一个新的基准测试评估了55个公开可用的LLMs在马耳他语（一种低资源语言）上的性能，该基准测试覆盖了11个 discriminative 和 generative 任务。我们的实验表明，许多模型在生成任务上的表现较差，而微调后的较小模型在所有任务上通常表现更好。通过对多维度数据分析，我们探讨了影响性能的各种因素。我们得出结论，预训练和指令微调期间对马耳他语的先前接触是最重要因素。我们还探讨了微调和提示之间的权衡关系，指出虽然微调需要更高的初始成本，但它能提供更好的性能和更低的推理成本。通过本研究，我们旨在强调更多包容的语言技术的必要性，并建议研究人员在处理低资源语言时考虑更多的“传统”语言建模方法。 

---
# AUTOCT: Automating Interpretable Clinical Trial Prediction with LLM Agents 

**Title (ZH)**: AUTOCT: 用LLM代理自动进行可解释的临床试验预测 

**Authors**: Fengze Liu, Haoyu Wang, Joonhyuk Cho, Dan Roth, Andrew W. Lo  

**Link**: [PDF](https://arxiv.org/pdf/2506.04293)  

**Abstract**: Clinical trials are critical for advancing medical treatments but remain prohibitively expensive and time-consuming. Accurate prediction of clinical trial outcomes can significantly reduce research and development costs and accelerate drug discovery. While recent deep learning models have shown promise by leveraging unstructured data, their black-box nature, lack of interpretability, and vulnerability to label leakage limit their practical use in high-stakes biomedical contexts. In this work, we propose AutoCT, a novel framework that combines the reasoning capabilities of large language models with the explainability of classical machine learning. AutoCT autonomously generates, evaluates, and refines tabular features based on public information without human input. Our method uses Monte Carlo Tree Search to iteratively optimize predictive performance. Experimental results show that AutoCT performs on par with or better than SOTA methods on clinical trial prediction tasks within only a limited number of self-refinement iterations, establishing a new paradigm for scalable, interpretable, and cost-efficient clinical trial prediction. 

**Abstract (ZH)**: 自动临床试验预测框架AutoCT：结合大型语言模型推理能力和经典机器学习解释性 

---
# Spore in the Wild: Case Study on Spore.fun, a Real-World Experiment of Sovereign Agent Open-ended Evolution on Blockchain with TEEs 

**Title (ZH)**: 异 Welt: 一种主权代理开放进化实验——基于TEE的Spore.fun在现实世界中的案例研究 

**Authors**: Botao Amber Hu, Helena Rong  

**Link**: [PDF](https://arxiv.org/pdf/2506.04236)  

**Abstract**: In Artificial Life (ALife) research, replicating Open-Ended Evolution (OEE)-the continuous emergence of novelty observed in biological life-has traditionally been pursued within isolated closed system simulations, such as Tierra and Avida, which have typically plateaued after an initial burst of novelty, failing to achieve sustained OEE. Scholars suggest that OEE requires an "open" system that continually exchanges information or energy with its environment. A recent technological innovation in decentralized physical infrastructure networks (DePIN) providing permissionless computational substrates enables deploying large language model (LLM)-based AI agents on blockchains integrated with Trusted Execution Environments (TEEs). This enables on-chain agents to operate autonomously "in the wild," achieving self-sovereignty without human oversight. These agents can control their own social media accounts and cryptocurrency wallets, allowing them to interact directly with blockchain-based financial networks and broader human social media. Building on this new paradigm of on-chain agents, this http URL is a recent real-world AI evolution experiment that enables autonomous breeding and evolution of new on-chain agents. This paper presents a detailed case study of this http URL, examining agent behaviors and their evolutionary trajectories through digital ethology. We aim to spark discussion about whether "open" ALife systems "in-the-wild," based on permissionless computational substrates and driven by economic incentives to interact with their environment, could finally achieve the long-sought goal of OEE. 

**Abstract (ZH)**: 在人工生命（ALife）研究中，复制开放-ended 进化（OEE）—生物生命中观察到的持续不断的创新—传统上是在孤立的封闭系统模拟中进行的，如Tierra和Avida，这些模拟通常在创新爆发初期之后达到平台期，未能实现持续的OEE。学者们建议，OEE 需要一个“开放”的系统，该系统能够持续与环境交换信息或能量。最近，在分散的物理基础设施网络（DePIN）中的一项技术革新提供了无许可的计算基础，使得能够在与可信执行环境（TEEs）集成的区块链上部署基于大规模语言模型（LLM）的AI代理。这使链上代理能够在没有人类监督的情况下自主“在野外”运行，实现自我主权。这些代理能够控制自己的社交媒体账户和加密货币钱包，从而直接与基于区块链的金融网络以及更广泛的人类社交媒体互动。在此基础上，本文介绍了一个最近的实际世界AI进化实验，该实验使得新的链上代理能够自主繁衍和演化。本文详细研究了这些代理的行为及其进化的轨迹，通过数字动物学进行分析。我们旨在探讨基于无许可计算基础和由与环境互动的经济激励驱动的“开放”的ALife系统，是否终于实现了长期追求的OEE目标。 

---
# HSSBench: Benchmarking Humanities and Social Sciences Ability for Multimodal Large Language Models 

**Title (ZH)**: HSSBench: 多模态大型语言模型的人文与社会科学能力评估 

**Authors**: Zhaolu Kang, Junhao Gong, Jiaxu Yan, Wanke Xia, Yian Wang, Ziwen Wang, Huaxuan Ding, Zhuo Cheng, Wenhao Cao, Zhiyuan Feng, Siqi He, Shannan Yan, Junzhe Chen, Xiaomin He, Chaoya Jiang, Wei Ye, Kaidong Yu, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.03922)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated significant potential to advance a broad range of domains. However, current benchmarks for evaluating MLLMs primarily emphasize general knowledge and vertical step-by-step reasoning typical of STEM disciplines, while overlooking the distinct needs and potential of the Humanities and Social Sciences (HSS). Tasks in the HSS domain require more horizontal, interdisciplinary thinking and a deep integration of knowledge across related fields, which presents unique challenges for MLLMs, particularly in linking abstract concepts with corresponding visual representations. Addressing this gap, we present HSSBench, a dedicated benchmark designed to assess the capabilities of MLLMs on HSS tasks in multiple languages, including the six official languages of the United Nations. We also introduce a novel data generation pipeline tailored for HSS scenarios, in which multiple domain experts and automated agents collaborate to generate and iteratively refine each sample. HSSBench contains over 13,000 meticulously designed samples, covering six key categories. We benchmark more than 20 mainstream MLLMs on HSSBench and demonstrate that it poses significant challenges even for state-of-the-art models. We hope that this benchmark will inspire further research into enhancing the cross-disciplinary reasoning abilities of MLLMs, especially their capacity to internalize and connect knowledge across fields. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在多个领域展现出巨大的潜力。然而，目前用于评估MLLMs的基准主要强调STEM学科中的通用知识和垂直逐步推理，而忽视了人文学科和社会科学（HSS）的独特需求和潜力。HSS领域的任务需要更多的跨学科横向思考和相关领域知识的深度融合，这对MLLMs提出了独特的挑战，尤其是在将抽象概念与相应的视觉表示连接起来方面。为弥补这一不足，我们提出了HSSBench，这是一种专门为评估MLLMs在HSS任务上的能力而设计的基准，涵盖六种联合国官方语言。我们还介绍了一种针对HSS场景的新型数据生成管道，其中多个领域专家和自动化代理协作生成并逐步优化每个样本。HSSBench包含超过13,000个精心设计的样本，涵盖六个关键类别。我们在HSSBench上对超过20种主流MLLMs进行了基准测试，并表明即使对于最先进的模型，这一基准也提出了重大挑战。我们希望这个基准能够激发进一步研究，以增强MLLMs的跨学科推理能力，尤其是它们在跨领域内吸收和连接知识的能力。 

---
# COSMOS: Predictable and Cost-Effective Adaptation of LLMs 

**Title (ZH)**: COSMOS: 可预测且成本有效的大型语言模型适应方法 

**Authors**: Jiayu Wang, Aws Albarghouthi, Frederic Sala  

**Link**: [PDF](https://arxiv.org/pdf/2505.01449)  

**Abstract**: Large language models (LLMs) achieve remarkable performance across numerous tasks by using a diverse array of adaptation strategies. However, optimally selecting a model and adaptation strategy under resource constraints is challenging and often requires extensive experimentation. We investigate whether it is possible to accurately predict both performance and cost without expensive trials. We formalize the strategy selection problem for LLMs and introduce COSMOS, a unified prediction framework that efficiently estimates adaptation outcomes at minimal cost. We instantiate and study the capability of our framework via a pair of powerful predictors: embedding-augmented lightweight proxy models to predict fine-tuning performance, and low-sample scaling laws to forecast retrieval-augmented in-context learning. Extensive evaluation across eight representative benchmarks demonstrates that COSMOS achieves high prediction accuracy while reducing computational costs by 92.72% on average, and up to 98.71% in resource-intensive scenarios. Our results show that efficient prediction of adaptation outcomes is not only feasible but can substantially reduce the computational overhead of LLM deployment while maintaining performance standards. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通过使用多种适应策略在众多任务中取得了显著性能。然而，在资源受限条件下选择最优模型和适应策略常常具有挑战性，往往需要大量的实验。我们研究是否可以在没有昂贵实验的情况下准确预测性能和成本。我们将大规模语言模型的策略选择问题形式化，并引入COSMOS，这是一种统一的预测框架，能够以极低的成本高效估计适应效果。我们通过一对强大的预测器实例化并研究了该框架的能力：基于嵌入的轻量级代理模型预测微调性能，以及小样本缩放定律预测检索增强的上下文学习效果。在八个代表性基准上的广泛评估表明，COSMOS在平均情况下将计算成本降低了92.72%，在资源密集型场景中最高可达98.71%。我们的结果表明，高效预测适应效果不仅是可行的，而且可以在不牺牲性能标准的前提下显著减少大规模语言模型部署的计算开销。 

---
