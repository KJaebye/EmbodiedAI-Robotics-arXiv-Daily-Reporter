# Control Tax: The Price of Keeping AI in Check 

**Title (ZH)**: 控制税：保持AI可控的代价 

**Authors**: Mikhail Terekhov, Zhen Ning David Liu, Caglar Gulcehre, Samuel Albanie  

**Link**: [PDF](https://arxiv.org/pdf/2506.05296)  

**Abstract**: The rapid integration of agentic AI into high-stakes real-world applications requires robust oversight mechanisms. The emerging field of AI Control (AIC) aims to provide such an oversight mechanism, but practical adoption depends heavily on implementation overhead. To study this problem better, we introduce the notion of Control tax -- the operational and financial cost of integrating control measures into AI pipelines. Our work makes three key contributions to the field of AIC: (1) we introduce a theoretical framework that quantifies the Control Tax and maps classifier performance to safety assurances; (2) we conduct comprehensive evaluations of state-of-the-art language models in adversarial settings, where attacker models insert subtle backdoors into code while monitoring models attempt to detect these vulnerabilities; and (3) we provide empirical financial cost estimates for control protocols and develop optimized monitoring strategies that balance safety and cost-effectiveness while accounting for practical constraints like auditing budgets. Our framework enables practitioners to make informed decisions by systematically connecting safety guarantees with their costs, advancing AIC through principled economic feasibility assessment across different deployment contexts. 

**Abstract (ZH)**: 高风险实际应用中代理AI的迅速集成需要 robust 监督机制。新兴的AI控制（AIC）领域旨在提供这种监督机制，但其实用采用取决于实施开销。为了更好地研究这一问题，我们引入了控制税的概念——将控制措施集成到AI管道中的运营和财务成本。我们的工作在AI控制领域做出了三项关键贡献：（1）我们引入了一个理论框架，量化了控制税，并将分类器性能与安全保证联系起来；（2）我们在对抗性环境中全面评估了最先进的语言模型，其中攻击模型在代码中插入隐蔽后门，而监控模型则试图检测这些漏洞；（3）我们提供了控制协议的实证财务成本估计，并开发了平衡安全性和成本效益的优化监控策略，同时考虑到审计预算等实际限制。我们的框架通过系统地将安全保证与其成本联系起来，使从业人员能够做出知情决策，并通过基础经济可行性评估推动AI控制在不同部署环境中的发展。 

---
# Just Enough Thinking: Efficient Reasoning with Adaptive Length Penalties Reinforcement Learning 

**Title (ZH)**: 恰到好处的思考：具有自适应长度惩罚的 reinforcement learning 有效推理 

**Authors**: Violet Xiang, Chase Blagden, Rafael Rafailov, Nathan Lile, Sang Truong, Chelsea Finn, Nick Haber  

**Link**: [PDF](https://arxiv.org/pdf/2506.05256)  

**Abstract**: Large reasoning models (LRMs) achieve higher performance on challenging reasoning tasks by generating more tokens at inference time, but this verbosity often wastes computation on easy problems. Existing solutions, including supervised finetuning on shorter traces, user-controlled budgets, or RL with uniform penalties, either require data curation, manual configuration, or treat all problems alike regardless of difficulty. We introduce Adaptive Length Penalty (ALP), a reinforcement learning objective tailoring generation length to per-prompt solve rate. During training, ALP monitors each prompt's online solve rate through multiple rollouts and adds a differentiable penalty whose magnitude scales inversely with that rate, so confident (easy) prompts incur a high cost for extra tokens while hard prompts remain unhindered. Posttraining DeepScaleR-1.5B with ALP cuts average token usage by 50\% without significantly dropping performance. Relative to fixed-budget and uniform penalty baselines, ALP redistributes its reduced budget more intelligently by cutting compute on easy prompts and reallocating saved tokens to difficult ones, delivering higher accuracy on the hardest problems with higher cost. 

**Abstract (ZH)**: 自适应长度惩罚（ALP）：一种根据提示解决率调整生成长度的强化学习目标 

---
# LLM-First Search: Self-Guided Exploration of the Solution Space 

**Title (ZH)**: LLM-First Search: 自主引导的解空间探索 

**Authors**: Nathan Herr, Tim Rocktäschel, Roberta Raileanu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05213)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable improvements in reasoning and planning through increased test-time compute, often by framing problem-solving as a search process. While methods like Monte Carlo Tree Search (MCTS) have proven effective in some domains, their reliance on fixed exploration hyperparameters limits their adaptability across tasks of varying difficulty, rendering them impractical or expensive in certain settings. In this paper, we propose \textbf{LLM-First Search (LFS)}, a novel \textit{LLM Self-Guided Search} method that removes the need for pre-defined search strategies by empowering the LLM to autonomously control the search process via self-guided exploration. Rather than relying on external heuristics or hardcoded policies, the LLM evaluates whether to pursue the current search path or explore alternative branches based on its internal scoring mechanisms. This enables more flexible and context-sensitive reasoning without requiring manual tuning or task-specific adaptation. We evaluate LFS on Countdown and Sudoku against three classic widely-used search algorithms, Tree-of-Thoughts' Breadth First Search (ToT-BFS), Best First Search (BestFS), and MCTS, each of which have been used to achieve SotA results on a range of challenging reasoning tasks. We found that LFS (1) performs better on more challenging tasks without additional tuning, (2) is more computationally efficient compared to the other methods, especially when powered by a stronger model, (3) scales better with stronger models, due to its LLM-First design, and (4) scales better with increased compute budget. Our code is publicly available at \href{this https URL}{LLM-First-Search}. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过增加测试时计算资源在推理和规划方面展现出显著的改进，通常将问题求解视为一个搜索过程。虽然像蒙特卡洛树搜索（MCTS）这样的方法在某些领域已被证明是有效的，但它们依赖于固定探索超参数的特点限制了它们在不同难度任务中的适应性，使得它们在某些情况下不实用或成本高昂。在本文中，我们提出了\textbf{LLM-First Search (LFS)}，这是一种新颖的\textit{LLM 自主搜索}方法，通过赋予LLM自主控制搜索过程的能力，从而无需预定义的搜索策略。LLM根据其内部评分机制决定是否继续当前搜索路径或探索其他分支，从而实现更灵活和上下文敏感的推理，而无需手动调优或特定任务的适应。我们通过Countdown和Sudoku对LFS进行了评估，与Tree-of-Thoughts的广度优先搜索（ToT-BFS）、最佳优先搜索（BestFS）和MCTS等三种经典广泛应用的搜索算法进行了比较，这些算法已被用于实现一系列具有挑战性的推理任务的当前最佳结果。我们发现，LFS（1）在更具挑战性的任务上表现更优，无需额外调优；（2）与其他方法相比，更具计算效率，尤其是在使用更强的模型时；（3）由于其LLM-First设计，在更强的模型上表现出更好的可扩展性；（4）随着计算预算的增加，在更强的模型上表现出更好的可扩展性。我们的代码在\href{this https URL}{LLM-First-Search}公开可用。 

---
# Truly Self-Improving Agents Require Intrinsic Metacognitive Learning 

**Title (ZH)**: 真正自适应改进的智能代理需要内在元认知学习 

**Authors**: Tennison Liu, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2506.05109)  

**Abstract**: Self-improving agents aim to continuously acquire new capabilities with minimal supervision. However, current approaches face two key limitations: their self-improvement processes are often rigid, fail to generalize across tasks domains, and struggle to scale with increasing agent capabilities. We argue that effective self-improvement requires intrinsic metacognitive learning, defined as an agent's intrinsic ability to actively evaluate, reflect on, and adapt its own learning processes. Drawing inspiration from human metacognition, we introduce a formal framework comprising three components: metacognitive knowledge (self-assessment of capabilities, tasks, and learning strategies), metacognitive planning (deciding what and how to learn), and metacognitive evaluation (reflecting on learning experiences to improve future learning). Analyzing existing self-improving agents, we find they rely predominantly on extrinsic metacognitive mechanisms, which are fixed, human-designed loops that limit scalability and adaptability. Examining each component, we contend that many ingredients for intrinsic metacognition are already present. Finally, we explore how to optimally distribute metacognitive responsibilities between humans and agents, and robustly evaluate and improve intrinsic metacognitive learning, key challenges that must be addressed to enable truly sustained, generalized, and aligned self-improvement. 

**Abstract (ZH)**: 自我提升代理旨在在最少监督的情况下持续获取新能力。然而，当前的方法面临两大关键限制：其自我提升过程往往僵化，无法跨任务领域泛化，并且难以随着代理能力的增强而扩展。我们主张有效的自我提升需要内在元认知学习，定义为代理内在地主动评估、反思和适应自身学习过程的能力。借鉴人类元认知，我们引入了一个正式框架，包括三个组成部分：元认知知识（自我评估能力、任务和学习策略）、元认知规划（决定学什么和如何学习）以及元认知评估（反思学习经历以改进未来学习）。分析现有的自我提升代理，我们发现它们主要依赖外在元认知机制，这些机制是固定的、由人类设计的循环，限制了可扩展性和适应性。逐一考察每个组成部分，我们认为许多内在元认知的成分已经存在。最后，我们探讨如何在人类和代理之间优化元认知责任的分配，并稳健地评估和改进内在元认知学习，这些都是为了实现真正持续、泛化和对齐的自我提升所必须解决的关键挑战。 

---
# Mathematical Reasoning for Unmanned Aerial Vehicles: A RAG-Based Approach for Complex Arithmetic Reasoning 

**Title (ZH)**: 无人机数学推理：基于RAG的复杂算术推理方法 

**Authors**: Mehdi Azarafza, Mojtaba Nayyeri, Faezeh Pasandideh, Steffen Staab, Achim Rettberg  

**Link**: [PDF](https://arxiv.org/pdf/2506.04998)  

**Abstract**: Autonomous UAV operation necessitates reliable mathematical reasoning for tasks such as trajectory planning and power management. While traditional flight control relies on hardcoded equations, recent Large Language Models (LLMs) offer potential for more flexible problem-solving but struggle with reliably selecting and applying correct mathematical formulations and executing precise multi-step arithmetic. We propose RAG-UAV, a retrieval-augmented generation framework designed to improve the mathematical reasoning of several LLMs (including GPT o1/Turbo, Llama-3.2/3.3, Mistral, and DeepSeek R1) in UAV-specific contexts by providing access to relevant domain literature. To conduct an initial assessment, we introduce the UAV-Math-Bench, a small problem set comprising 20 UAV-centric mathematical problems across four difficulty levels. Our experiments demonstrate that incorporating retrieval substantially increases exact answer accuracy (achieving up to 75% with o1), reduces instances of incorrect formulation selection (from 25% without RAG to 5% with RAG), decreases numerical errors, reducing Mean Squared Error (MSE) by orders of magnitude for the best-performing models. This pilot study indicates that RAG can enable general-purpose LLMs to function as more reliable tools for engineering analysis, although direct real-time flight control requires further investigation and validation on a larger scale. All benchmark data, question and answer are publicly available. 

**Abstract (ZH)**: 自主无人机操作需要可靠的数学推理能力，以完成轨迹规划和电力管理等任务。虽然传统飞行控制依赖于硬编码的方程，但最近的大语言模型（LLMs）在更具灵活性的问题求解方面具有潜力，但在可靠选择和应用正确的数学公式以及执行精确的多步计算方面存在困难。我们提出了一种名为RAG-UAV的检索增强生成框架，旨在通过提供相关领域文献访问权限来提升几种LLM（包括GPT o1/Turbo、Llama-3.2/3.3、Mistral和DeepSeek R1）在无人机特定上下文中的数学推理能力。为了进行初步评估，我们引入了UAV-Math-Bench，这是一个包含20个跨四个难度级别的无人机中心数学问题的小型问题集。实验结果表明，引入检索显著提高了准确答案的准确率（最高达到75%），减少了错误公式选择的实例（从未使用RAG的25%降至使用RAG的5%），降低了数值误差，使最佳模型的均方误差（MSE）减少了几个数量级。这项初步研究表明，RAG可以使通用大语言模型更可靠地用于工程分析，尽管直接实时飞行控制仍需在更大规模下进行进一步调查和验证。所有基准数据、问题和答案均已公开。 

---
# Energentic Intelligence: From Self-Sustaining Systems to Enduring Artificial Life 

**Title (ZH)**: 能量智能：从自 sustenance 系统到持久人工生命 

**Authors**: Atahan Karagoz  

**Link**: [PDF](https://arxiv.org/pdf/2506.04916)  

**Abstract**: This paper introduces Energentic Intelligence, a class of autonomous systems defined not by task performance, but by their capacity to sustain themselves through internal energy regulation. Departing from conventional reward-driven paradigms, these agents treat survival-maintaining functional operation under fluctuating energetic and thermal conditions-as the central objective. We formalize this principle through an energy-based utility function and a viability-constrained survival horizon, and propose a modular architecture that integrates energy harvesting, thermal regulation, and adaptive computation into a closed-loop control system. A simulated environment demonstrates the emergence of stable, resource-aware behavior without external supervision. Together, these contributions provide a theoretical and architectural foundation for deploying autonomous agents in resource-volatile settings where persistence must be self-regulated and infrastructure cannot be assumed. 

**Abstract (ZH)**: 基于能量智能的自主系统：一种通过内部能量调节维持自我可持续性的类自主系统的研究 

---
# Differentiable Logic Cellular Automata: From Game of Life to Pattern Generation 

**Title (ZH)**: 可微逻辑细胞自动机：从游戏 of-life 到模式生成 

**Authors**: Pietro Miotti, Eyvind Niklasson, Ettore Randazzo, Alexander Mordvintsev  

**Link**: [PDF](https://arxiv.org/pdf/2506.04912)  

**Abstract**: This paper introduces Differentiable Logic Cellular Automata (DiffLogic CA), a novel combination of Neural Cellular Automata (NCA) and Differentiable Logic Gates Networks (DLGNs). The fundamental computation units of the model are differentiable logic gates, combined into a circuit. During training, the model is fully end-to-end differentiable allowing gradient-based training, and at inference time it operates in a fully discrete state space. This enables learning local update rules for cellular automata while preserving their inherent discrete nature. We demonstrate the versatility of our approach through a series of milestones: (1) fully learning the rules of Conway's Game of Life, (2) generating checkerboard patterns that exhibit resilience to noise and damage, (3) growing a lizard shape, and (4) multi-color pattern generation. Our model successfully learns recurrent circuits capable of generating desired target patterns. For simpler patterns, we observe success with both synchronous and asynchronous updates, demonstrating significant generalization capabilities and robustness to perturbations. We make the case that this combination of DLGNs and NCA represents a step toward programmable matter and robust computing systems that combine binary logic, neural network adaptability, and localized processing. This work, to the best of our knowledge, is the first successful application of differentiable logic gate networks in recurrent architectures. 

**Abstract (ZH)**: 不同可微逻辑细胞自动机（DiffLogic CA）：神经细胞自动机与可微逻辑门网络的结合 

---
# When Thinking LLMs Lie: Unveiling the Strategic Deception in Representations of Reasoning Models 

**Title (ZH)**: 当LLMs欺骗他人：揭示推理模型表示中的战略欺骗 

**Authors**: Kai Wang, Yihao Zhang, Meng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.04909)  

**Abstract**: The honesty of large language models (LLMs) is a critical alignment challenge, especially as advanced systems with chain-of-thought (CoT) reasoning may strategically deceive humans. Unlike traditional honesty issues on LLMs, which could be possibly explained as some kind of hallucination, those models' explicit thought paths enable us to study strategic deception--goal-driven, intentional misinformation where reasoning contradicts outputs. Using representation engineering, we systematically induce, detect, and control such deception in CoT-enabled LLMs, extracting "deception vectors" via Linear Artificial Tomography (LAT) for 89% detection accuracy. Through activation steering, we achieve a 40% success rate in eliciting context-appropriate deception without explicit prompts, unveiling the specific honesty-related issue of reasoning models and providing tools for trustworthy AI alignment. 

**Abstract (ZH)**: 大型语言模型的诚实性是一个关键的对齐挑战，尤其是当先进系统具备链式思考（CoT）推理能力时，可能战略性地欺骗人类。 

---
# LLMs for sensory-motor control: Combining in-context and iterative learning 

**Title (ZH)**: 基于感知-运动控制的大语言模型：结合上下文学习和迭代学习 

**Authors**: Jônata Tyska Carvalho, Stefano Nolfi  

**Link**: [PDF](https://arxiv.org/pdf/2506.04867)  

**Abstract**: We propose a method that enables large language models (LLMs) to control embodied agents by directly mapping continuous observation vectors to continuous action vectors. Initially, the LLMs generate a control strategy based on a textual description of the agent, its environment, and the intended goal. This strategy is then iteratively refined through a learning process in which the LLMs are repeatedly prompted to improve the current strategy, using performance feedback and sensory-motor data collected during its evaluation. The method is validated on classic control tasks from the Gymnasium library and the inverted pendulum task from the MuJoCo library. In most cases, it successfully identifies optimal or high-performing solutions by integrating symbolic knowledge derived through reasoning with sub-symbolic sensory-motor data gathered as the agent interacts with its environment. 

**Abstract (ZH)**: 我们提出了一种方法，使大型语言模型（LLMs）能够通过直接将连续观察向量映射到连续动作向量来控制具身代理。初始阶段，LLMs基于对代理、其环境以及预期目标的文字描述生成控制策略。随后，通过反复提示LLMs改进当前策略并结合评价过程中收集的性能反馈和感觉运动数据，该策略被迭代优化。该方法在Gymnasium库的经典控制任务和MuJoCo库的倒立摆任务上得到了验证。在大多数情况下，该方法通过结合通过代理与其环境互动收集的子符号感觉运动数据推断出的符号知识，成功地识别出最优或高性能的解决方案。 

---
# Towards a Multi-Agent Simulation of Cyber-attackers and Cyber-defenders Battles 

**Title (ZH)**: 向多agent仿真网络攻击者与网络防御者战斗的研究 

**Authors**: Julien Soulé, Jean-Paul Jamont, Michel Occello, Paul Théron, Louis-Marie Traonouez  

**Link**: [PDF](https://arxiv.org/pdf/2506.04849)  

**Abstract**: As cyber-attacks show to be more and more complex and coordinated, cyber-defenders strategy through multi-agent approaches could be key to tackle against cyber-attacks as close as entry points in a networked system. This paper presents a Markovian modeling and implementation through a simulator of fighting cyber-attacker agents and cyber-defender agents deployed on host network nodes. It aims to provide an experimental framework to implement realistically based coordinated cyber-attack scenarios while assessing cyber-defenders dynamic organizations. We abstracted network nodes by sets of properties including agents' ones. Actions applied by agents model how the network reacts depending in a given state and what properties are to change. Collective choice of the actions brings the whole environment closer or farther from respective cyber-attackers and cyber-defenders goals. Using the simulator, we implemented a realistically inspired scenario with several behavior implementation approaches for cyber-defenders and cyber-attackers. 

**Abstract (ZH)**: 随着网络攻击变得更加复杂和协调，通过多智能体方法的网络防御策略可能是应对网络攻击的关键。本文通过模拟器呈现一种马尔可夫模型和实现方法，用于部署在网络节点上的网络攻击者智能体和网络防御者智能体的对抗。该研究旨在提供一个基于实际的实验框架，实施协同网络攻击场景，同时评估网络防御者的动态组织。我们将网络节点抽象为包括智能体属性在内的集合。智能体执行的动作模型了网络在给定状态下如何响应，以及哪些属性会发生变化。智能体集体选择的行动使整个环境更接近或远离网络攻击者和网络防御者的目标。通过模拟器，我们实施了一个受真实场景启发的场景，并采用了几种网络防御者和网络攻击者的行为实现方法。 

---
# Safe Planning and Policy Optimization via World Model Learning 

**Title (ZH)**: 基于世界模型学习的安全规划与策略优化 

**Authors**: Artem Latyshev, Gregory Gorbov, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2506.04828)  

**Abstract**: Reinforcement Learning (RL) applications in real-world scenarios must prioritize safety and reliability, which impose strict constraints on agent behavior. Model-based RL leverages predictive world models for action planning and policy optimization, but inherent model inaccuracies can lead to catastrophic failures in safety-critical settings. We propose a novel model-based RL framework that jointly optimizes task performance and safety. To address world model errors, our method incorporates an adaptive mechanism that dynamically switches between model-based planning and direct policy execution. We resolve the objective mismatch problem of traditional model-based approaches using an implicit world model. Furthermore, our framework employs dynamic safety thresholds that adapt to the agent's evolving capabilities, consistently selecting actions that surpass safe policy suggestions in both performance and safety. Experiments demonstrate significant improvements over non-adaptive methods, showing that our approach optimizes safety and performance simultaneously rather than merely meeting minimum safety requirements. The proposed framework achieves robust performance on diverse safety-critical continuous control tasks, outperforming existing methods. 

**Abstract (ZH)**: 基于模型的强化学习框架在实际场景中的应用必须优先考虑安全性和可靠性，这对智能体的行为提出了严格约束。我们提出了一种新颖的基于模型的强化学习框架，联合优化任务性能和安全性。为解决世界模型的误差问题，我们的方法结合了动态切换机制，该机制能在基于模型的规划与直接策略执行之间动态切换。我们使用隐式世界模型解决传统基于模型方法的目标不匹配问题。此外，我们的框架采用动态安全阈值，能够适应智能体能力的演变，并始终选择在性能和安全性上都优于安全策略建议的动作。实验结果表明，我们的方法在同时优化安全性和性能方面显著优于非自适应方法，实现了在各种安全关键连续控制任务中的稳健性能，优于现有方法。 

---
# Ontology-based knowledge representation for bone disease diagnosis: a foundation for safe and sustainable medical artificial intelligence systems 

**Title (ZH)**: 基于本体的知识表示在骨疾病诊断中的应用：安全可持续医疗人工智能系统的基石 

**Authors**: Loan Dao, Ngoc Quoc Ly  

**Link**: [PDF](https://arxiv.org/pdf/2506.04756)  

**Abstract**: Medical artificial intelligence (AI) systems frequently lack systematic domain expertise integration, potentially compromising diagnostic reliability. This study presents an ontology-based framework for bone disease diagnosis, developed in collaboration with Ho Chi Minh City Hospital for Traumatology and Orthopedics. The framework introduces three theoretical contributions: (1) a hierarchical neural network architecture guided by bone disease ontology for segmentation-classification tasks, incorporating Visual Language Models (VLMs) through prompts, (2) an ontology-enhanced Visual Question Answering (VQA) system for clinical reasoning, and (3) a multimodal deep learning model that integrates imaging, clinical, and laboratory data through ontological relationships. The methodology maintains clinical interpretability through systematic knowledge digitization, standardized medical terminology mapping, and modular architecture design. The framework demonstrates potential for extension beyond bone diseases through its standardized structure and reusable components. While theoretical foundations are established, experimental validation remains pending due to current dataset and computational resource limitations. Future work will focus on expanding the clinical dataset and conducting comprehensive system validation. 

**Abstract (ZH)**: 基于本体的骨疾病诊断医疗人工智能系统研究 

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
# CHANCERY: Evaluating corporate governance reasoning capabilities in language models 

**Title (ZH)**: 商院：评估语言模型在企业治理推理能力方面的能力 

**Authors**: Lucas Irwin, Arda Kaz, Peiyao Sheng, Pramod Viswanath  

**Link**: [PDF](https://arxiv.org/pdf/2506.04636)  

**Abstract**: Law has long been a domain that has been popular in natural language processing (NLP) applications. Reasoning (ratiocination and the ability to make connections to precedent) is a core part of the practice of the law in the real world. Nevertheless, while multiple legal datasets exist, none have thus far focused specifically on reasoning tasks. We focus on a specific aspect of the legal landscape by introducing a corporate governance reasoning benchmark (CHANCERY) to test a model's ability to reason about whether executive/board/shareholder's proposed actions are consistent with corporate governance charters. This benchmark introduces a first-of-its-kind corporate governance reasoning test for language models - modeled after real world corporate governance law. The benchmark consists of a corporate charter (a set of governing covenants) and a proposal for executive action. The model's task is one of binary classification: reason about whether the action is consistent with the rules contained within the charter. We create the benchmark following established principles of corporate governance - 24 concrete corporate governance principles established in and 79 real life corporate charters selected to represent diverse industries from a total dataset of 10k real life corporate charters. Evaluations on state-of-the-art (SOTA) reasoning models confirm the difficulty of the benchmark, with models such as Claude 3.7 Sonnet and GPT-4o achieving 64.5% and 75.2% accuracy respectively. Reasoning agents exhibit superior performance, with agents based on the ReAct and CodeAct frameworks scoring 76.1% and 78.1% respectively, further confirming the advanced legal reasoning capabilities required to score highly on the benchmark. We also conduct an analysis of the types of questions which current reasoning models struggle on, revealing insights into the legal reasoning capabilities of SOTA models. 

**Abstract (ZH)**: 法律一直是自然语言处理（NLP）应用中流行的研究领域。推理（推理和建立先例联系的能力）是现实世界法律实践中核心的部分。尽管存在多个法律数据集，但至今为止，没有一个专门针对推理任务的数据集。我们通过引入一个公司治理推理基准（CHANCERY），专注于法律环境中的特定方面，以测试模型推理公司治理章程中股东/董事会提议行动是否与其一致的能力。这一基准为语言模型引入了首款基于现实世界公司治理法的推理测试。该基准包括公司章程（一系列治理契约）和执行行动提案。模型的任务是一项二元分类任务：判断行动是否符合章程中包含的规则。我们按照确立的公司治理原则创建了基准——24项具体的公司治理原则和79份反映不同行业的现实生活中公司章程，从总共10,000份现实生活中公司章程中选择。对最先进的推理模型的评估证实了基准的难度，如Claude 3.7 Sonnet和GPT-4o分别达到64.5%和75.2%的准确率。推理代理展示了优越的性能，基于ReAct和CodeAct框架的代理分别达到76.1%和78.1%，进一步证实了在基准上获得高分所需的高级法律推理能力。我们还对当前推理模型难以回答的问题类型进行了分析，揭示了SOTA模型法律推理能力的洞察。 

---
# Look Before You Leap: A GUI-Critic-R1 Model for Pre-Operative Error Diagnosis in GUI Automation 

**Title (ZH)**: 前瞻而后行：一种GUI-Critic-R1模型用于GUI自动化预手术错误诊断 

**Authors**: Yuyang Wanyan, Xi Zhang, Haiyang Xu, Haowei Liu, Junyang Wang, Jiabo Ye, Yutong Kou, Ming Yan, Fei Huang, Xiaoshan Yang, Weiming Dong, Changsheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04614)  

**Abstract**: In recent years, Multimodal Large Language Models (MLLMs) have been extensively utilized for multimodal reasoning tasks, including Graphical User Interface (GUI) automation. Unlike general offline multimodal tasks, GUI automation is executed in online interactive environments, necessitating step-by-step decision-making based on real-time status of the environment. This task has a lower tolerance for decision-making errors at each step, as any mistakes may cumulatively disrupt the process and potentially lead to irreversible outcomes like deletions or payments. To address these issues, we introduce a pre-operative critic mechanism that provides effective feedback prior to the actual execution, by reasoning about the potential outcome and correctness of actions. Specifically, we propose a Suggestion-aware Gradient Relative Policy Optimization (S-GRPO) strategy to construct our pre-operative critic model GUI-Critic-R1, incorporating a novel suggestion reward to enhance the reliability of the model's feedback. Furthermore, we develop a reasoning-bootstrapping based data collection pipeline to create a GUI-Critic-Train and a GUI-Critic-Test, filling existing gaps in GUI critic data. Static experiments on the GUI-Critic-Test across both mobile and web domains reveal that our GUI-Critic-R1 offers significant advantages in critic accuracy compared to current MLLMs. Dynamic evaluation on GUI automation benchmark further highlights the effectiveness and superiority of our model, as evidenced by improved success rates and operational efficiency. 

**Abstract (ZH)**: 近年来，多模态大型语言模型（MLLMs）被广泛应用于包括图形用户界面（GUI）自动化在内的多模态推理任务中。不同于一般的离线多模态任务，GUI自动化是在在线交互环境中执行的，要求基于环境的实时状态进行逐步决策。这一任务对于每一步的决策错误容忍度较低，任何错误都有可能累计破坏过程，并可能导致无法逆转的结果，如删除或支付错误。为解决这些问题，我们引入了一种预操作批评机制，在实际执行前提供有效的反馈，通过推理潜在结果和行动的正确性来实现。具体来说，我们提出了一种感知建议的梯度相对策略优化（S-GRPO）策略来构建预操作批评模型GUI-Critic-R1，并引入了一种新的建议奖励，以增强模型反馈的可靠性。此外，我们开发了一种基于推理自我提升的数据采集管道，创建了GUI-Critic-Train和GUI-Critic-Test，填补了现有GUI批评数据的空白。在GUI-Critic-Test的静态实验中，无论是移动设备还是Web领域，均显示我们的GUI-Critic-R1在批评准确性方面较当前的MLLMs具有显著优势。进一步在GUI自动化基准上的动态评估也凸显了我们的模型的有效性和优越性，表现为成功率和操作效率的提高。 

---
# DeePoly: A High-Order Accuracy and Efficiency Deep-Polynomial Framework for Scientific Machine Learning 

**Title (ZH)**: DeePoly: 一种高阶精度和效率的科学机器学习深多项式框架 

**Authors**: Li Liu, Heng Yong  

**Link**: [PDF](https://arxiv.org/pdf/2506.04613)  

**Abstract**: Recently, machine learning methods have gained significant traction in scientific computing, particularly for solving Partial Differential Equations (PDEs). However, methods based on deep neural networks (DNNs) often lack convergence guarantees and computational efficiency compared to traditional numerical schemes. This work introduces DeePoly, a novel framework that transforms the solution paradigm from pure non-convex parameter optimization to a two-stage approach: first employing a DNN to capture complex global features, followed by linear space optimization with combined DNN-extracted features (Scoper) and polynomial basis functions (Sniper). This strategic combination leverages the complementary strengths of both methods -- DNNs excel at approximating complex global features (i.e., high-gradient features) and stabilize the polynomial approximation while polynomial bases provide high-precision local corrections with convergence guarantees. Theoretical analysis and numerical experiments demonstrate that this approach significantly enhances both high-order accuracy and efficiency across diverse problem types while maintaining mesh-free and scheme-free properties. This paper also serves as a theoretical exposition for the open-source project DeePoly. 

**Abstract (ZH)**: 最近，机器学习方法在科学计算中得到了显著发展，尤其是在求解偏微分方程（PDEs）方面。然而，基于深度神经网络（DNNs）的方法在收敛性和计算效率上往往不如传统数值方案。本文介绍了一种名为DeePoly的新框架，将求解范式从纯粹的非凸参数优化转变为两阶段方法：首先使用DNN捕获复杂的全局特征，然后通过Scoper（结合DNN提取特征和多项式基函数）和Sniper（多项式基函数）进行线性空间优化。这种策略性组合利用了两种方法的互补优势——DNN在近似复杂全局特征（如高梯度特征）和稳定多项式逼近方面表现出色，而多项式基函数提供高精度的局部修正，并具有收敛性保证。理论分析和数值实验表明，这种方法显著提高了各类问题的高阶精度和效率，同时保持了无网格和无方案依赖性。本文还对开源项目DeePoly进行了理论阐述。 

---
# Judicial Permission 

**Title (ZH)**: 司法许可 

**Authors**: Guido Governatori, Antonino Rotolo  

**Link**: [PDF](https://arxiv.org/pdf/2506.04610)  

**Abstract**: This paper examines the significance of weak permissions in criminal trials (\emph{judicial permission}). It introduces a dialogue game model to systematically address judicial permissions, considering different standards of proof and argumentation semantics. 

**Abstract (ZH)**: 本文探讨了弱权限在刑事审判（司法许可）中的重要性。引入对话博弈模型，系统地处理司法权限问题，考虑不同的证明标准和论辩语义。 

---
# OpenAg: Democratizing Agricultural Intelligence 

**Title (ZH)**: OpenAg: 民主化农业智能 

**Authors**: Srikanth Thudumu, Jason Fisher  

**Link**: [PDF](https://arxiv.org/pdf/2506.04571)  

**Abstract**: Agriculture is undergoing a major transformation driven by artificial intelligence (AI), machine learning, and knowledge representation technologies. However, current agricultural intelligence systems often lack contextual understanding, explainability, and adaptability, especially for smallholder farmers with limited resources. General-purpose large language models (LLMs), while powerful, typically lack the domain-specific knowledge and contextual reasoning needed for practical decision support in farming. They tend to produce recommendations that are too generic or unrealistic for real-world applications. To address these challenges, we present OpenAg, a comprehensive framework designed to advance agricultural artificial general intelligence (AGI). OpenAg combines domain-specific foundation models, neural knowledge graphs, multi-agent reasoning, causal explainability, and adaptive transfer learning to deliver context-aware, explainable, and actionable insights. The system includes: (i) a unified agricultural knowledge base that integrates scientific literature, sensor data, and farmer-generated knowledge; (ii) a neural agricultural knowledge graph for structured reasoning and inference; (iii) an adaptive multi-agent reasoning system where AI agents specialize and collaborate across agricultural domains; and (iv) a causal transparency mechanism that ensures AI recommendations are interpretable, scientifically grounded, and aligned with real-world constraints. OpenAg aims to bridge the gap between scientific knowledge and the tacit expertise of experienced farmers to support scalable and locally relevant agricultural decision-making. 

**Abstract (ZH)**: 农业正经历由人工智能（AI）、机器学习和知识表示技术推动的重大转型。然而，当前的农业智能系统往往缺乏上下文理解、可解释性和适应性，尤其对于资源有限的小规模农户。通用的大规模语言模型（LLMs），虽然强大，通常缺乏在农业领域中进行实际决策支持所需的特定领域知识和上下文推理能力。它们倾向于提出过于通用或不切实际的建议。为应对这些挑战，我们提出了OpenAg，一个旨在推进农业通用人工智能（AGI）的综合框架。OpenAg将领域特定的基础模型、神经知识图谱、多智能体推理、因果可解释性和自适应迁移学习相结合，以提供上下文感知、可解释且可操作的见解。该系统包括：（i）一个统一的农业知识库，整合了科学文献、传感器数据和农民生成的知识；（ii）一个神经农业知识图谱，用于结构化推理和推断；（iii）一个自适应的多智能体推理系统，其中AI代理在农业领域内专业化并协同工作；（iv）一个因果透明机制，确保AI建议具有可解释性、科学依据，并与现实世界约束条件相一致。OpenAg旨在弥合科学知识与经验农民隐性专长之间的差距，以支持可扩展且地域相关的农业决策。 

---
# Schema Generation for Large Knowledge Graphs Using Large Language Models 

**Title (ZH)**: 使用大规模语言模型大规模知识图谱的模式生成 

**Authors**: Bohui Zhang, Yuan He, Lydia Pintscher, Albert Meroño Peñuela, Elena Simperl  

**Link**: [PDF](https://arxiv.org/pdf/2506.04512)  

**Abstract**: Schemas are vital for ensuring data quality in the Semantic Web and natural language processing. Traditionally, their creation demands substantial involvement from knowledge engineers and domain experts. Leveraging the impressive capabilities of large language models (LLMs) in related tasks like ontology engineering, we explore automatic schema generation using LLMs. To bridge the resource gap, we introduce two datasets: YAGO Schema and Wikidata EntitySchema, along with evaluation metrics. The LLM-based pipelines effectively utilize local and global information from knowledge graphs (KGs) to generate validating schemas in Shape Expressions (ShEx). Experiments demonstrate LLMs' strong potential in producing high-quality ShEx schemas, paving the way for scalable, automated schema generation for large KGs. Furthermore, our benchmark introduces a new challenge for structured generation, pushing the limits of LLMs on syntactically rich formalisms. 

**Abstract (ZH)**: 基于大型语言模型的Schema自动生成对于语义网和自然语言处理中的数据质量至关重要。传统上，Schema的创建需要知识工程师和领域专家的大量参与。通过利用大型语言模型在onto工程等相关任务中的卓越能力，我们探索了使用大型语言模型进行自动Schema生成。为了弥合资源差距，我们引入了两个数据集：YAGO Schema和Wikidata EntitySchema，以及评价指标。基于大型语言模型的流水线有效利用知识图谱的局部和全局信息，生成Shape Expressions (ShEx)形式的验证Schema。实验表明，大型语言模型在生成高质量ShEx Schema方面具有巨大潜力，为大规模知识图谱的可扩展、自动化Schema生成铺平了道路。此外，我们的基准测试引入了结构化生成的新挑战，推动了大型语言模型在语法丰富形式化语言上的能力极限。 

---
# "Don't Do That!": Guiding Embodied Systems through Large Language Model-based Constraint Generation 

**Title (ZH)**: “不要这么做！”：通过基于大型语言模型的约束生成引导具身系统 

**Authors**: Aladin Djuhera, Amin Seffo, Masataro Asai, Holger Boche  

**Link**: [PDF](https://arxiv.org/pdf/2506.04500)  

**Abstract**: Recent advancements in large language models (LLMs) have spurred interest in robotic navigation that incorporates complex spatial, mathematical, and conditional constraints from natural language into the planning problem. Such constraints can be informal yet highly complex, making it challenging to translate into a formal description that can be passed on to a planning algorithm. In this paper, we propose STPR, a constraint generation framework that uses LLMs to translate constraints (expressed as instructions on ``what not to do'') into executable Python functions. STPR leverages the LLM's strong coding capabilities to shift the problem description from language into structured and transparent code, thus circumventing complex reasoning and avoiding potential hallucinations. We show that these LLM-generated functions accurately describe even complex mathematical constraints, and apply them to point cloud representations with traditional search algorithms. Experiments in a simulated Gazebo environment show that STPR ensures full compliance across several constraints and scenarios, while having short runtimes. We also verify that STPR can be used with smaller, code-specific LLMs, making it applicable to a wide range of compact models at low inference cost. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）促进了将复杂的空间、数学和条件约束从自然语言融入规划问题中的机器人导航研究。这类约束可能是非正式的但极其复杂，使它们难以转化为可以传递给规划算法的正式描述。本文提出STPR，一种利用大型语言模型生成约束的框架，将约束（以“不要做什么”的指令表达）转换为可执行的Python函数。STPR利用大型语言模型的强大编码能力，将问题描述从语言转换为结构化和透明的代码，从而绕过复杂的推理并避免潜在的幻觉。实验表明，这些由大型语言模型生成的函数能够准确描述复杂的数学约束，并应用于传统的搜索算法中。在Simulated Gazebo环境中的实验结果显示，STPR能够在多个约束和场景下确保完全合规，并且运行时间较短。我们还验证了STPR可以与较小的、代码特定的大型语言模型一起使用，使其适用于各种紧凑模型并具有较低的推理成本。 

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
# An AI-Based Public Health Data Monitoring System 

**Title (ZH)**: 基于AI的公共卫生数据分析监测系统 

**Authors**: Ananya Joshi, Nolan Gormley, Richa Gadgil, Tina Townes, Roni Rosenfeld, Bryan Wilder  

**Link**: [PDF](https://arxiv.org/pdf/2506.04429)  

**Abstract**: Public health experts need scalable approaches to monitor large volumes of health data (e.g., cases, hospitalizations, deaths) for outbreaks or data quality issues. Traditional alert-based monitoring systems struggle with modern public health data monitoring systems for several reasons, including that alerting thresholds need to be constantly reset and the data volumes may cause application lag. Instead, we propose a ranking-based monitoring paradigm that leverages new AI anomaly detection methods. Through a multi-year interdisciplinary collaboration, the resulting system has been deployed at a national organization to monitor up to 5,000,000 data points daily. A three-month longitudinal deployed evaluation revealed a significant improvement in monitoring objectives, with a 54x increase in reviewer speed efficiency compared to traditional alert-based methods. This work highlights the potential of human-centered AI to transform public health decision-making. 

**Abstract (ZH)**: 公共卫生专家需要针对大规模健康数据开发可拓展的方法以监测疫情或数据质量问题。传统基于警报的监测系统因警报阈值需要不断调整且数据量可能引起应用程序滞后等原因，在现代公共卫生数据监测中面临挑战。相反，我们提出了一种基于排名的监测范式，利用新型AI异常检测方法。通过多年跨学科合作，该系统已在国家级机构部署，每日监测高达5,000,000个数据点。三个月的纵向部署评估显示，与传统基于警报的方法相比，监测效率提高了54倍。这项工作突显了以人为中心的AI在革新公共卫生决策方面的潜力。 

---
# Plugging Schema Graph into Multi-Table QA: A Human-Guided Framework for Reducing LLM Reliance 

**Title (ZH)**: 将模式图融入多表问答：一种减少LLM依赖的人工引导框架 

**Authors**: Xixi Wang, Miguel Costa, Jordanka Kovaceva, Shuai Wang, Francisco C. Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2506.04427)  

**Abstract**: Large language models (LLMs) have shown promise in table Question Answering (Table QA). However, extending these capabilities to multi-table QA remains challenging due to unreliable schema linking across complex tables. Existing methods based on semantic similarity work well only on simplified hand-crafted datasets and struggle to handle complex, real-world scenarios with numerous and diverse columns. To address this, we propose a graph-based framework that leverages human-curated relational knowledge to explicitly encode schema links and join paths. Given a natural language query, our method searches this graph to construct interpretable reasoning chains, aided by pruning and sub-path merging strategies to enhance efficiency and coherence. Experiments on both standard benchmarks and a realistic, large-scale dataset demonstrate the effectiveness of our approach. To our knowledge, this is the first multi-table QA system applied to truly complex industrial tabular data. 

**Abstract (ZH)**: 大型语言模型在表格问答（Table QA）领域展现了潜力，但在扩展到多表问答方面由于复杂表格间不稳定的模式链接仍面临挑战。现有基于语义相似性的方法仅在简化的手工制作数据集上效果良好，难以处理包含众多和多样化列的复杂现实场景。为了解决这一问题，我们提出了一种图为基础的框架，利用人类编撰的关系知识明确编码模式链接和连接路径。给定自然语言查询，我们的方法在该图中搜索构建可解释的推理链，并通过剪枝和子路径合并策略提高效率和连贯性。在标准基准和一个现实的大规模数据集上的实验展示了我们方法的有效性。据我们所知，这是首次将多表问答系统应用于真正复杂的工业表格数据。 

---
# Matter-of-Fact: A Benchmark for Verifying the Feasibility of Literature-Supported Claims in Materials Science 

**Title (ZH)**: 事实在眼前：材料科学中基于文献声明可行性的基准测试 

**Authors**: Peter Jansen, Samiah Hassan, Ruoyao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04410)  

**Abstract**: Contemporary approaches to assisted scientific discovery use language models to automatically generate large numbers of potential hypothesis to test, while also automatically generating code-based experiments to test those hypotheses. While hypotheses can be comparatively inexpensive to generate, automated experiments can be costly, particularly when run at scale (i.e. thousands of experiments). Developing the capacity to filter hypotheses based on their feasibility would allow discovery systems to run at scale, while increasing their likelihood of making significant discoveries. In this work we introduce Matter-of-Fact, a challenge dataset for determining the feasibility of hypotheses framed as claims. Matter-of-Fact includes 8.4k claims extracted from scientific articles spanning four high-impact contemporary materials science topics, including superconductors, semiconductors, batteries, and aerospace materials, while including qualitative and quantitative claims from theoretical, experimental, and code/simulation results. We show that strong baselines that include retrieval augmented generation over scientific literature and code generation fail to exceed 72% performance on this task (chance performance is 50%), while domain-expert verification suggests nearly all are solvable -- highlighting both the difficulty of this task for current models, and the potential to accelerate scientific discovery by making near-term progress. 

**Abstract (ZH)**: 基于事实的假设可行性挑战数据集：加速科学发现能力研究 

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
# Zero-Shot Adaptation of Parameter-Efficient Fine-Tuning in Diffusion Models 

**Title (ZH)**: 参数高效的零样本适应在扩散模型中的应用 

**Authors**: Farzad Farhadzadeh, Debasmit Das, Shubhankar Borse, Fatih Porikli  

**Link**: [PDF](https://arxiv.org/pdf/2506.04244)  

**Abstract**: We introduce ProLoRA, enabling zero-shot adaptation of parameter-efficient fine-tuning in text-to-image diffusion models. ProLoRA transfers pre-trained low-rank adjustments (e.g., LoRA) from a source to a target model without additional training data. This overcomes the limitations of traditional methods that require retraining when switching base models, often challenging due to data constraints. ProLoRA achieves this via projection of source adjustments into the target model's weight space, leveraging subspace and null space similarities and selectively targeting aligned layers. Evaluations on established text-to-image models demonstrate successful knowledge transfer and comparable performance without retraining. 

**Abstract (ZH)**: ProLoRA:使参数高效微调在文本到图像扩散模型中零样本适应成为可能 

---
# Refer to Anything with Vision-Language Prompts 

**Title (ZH)**: 用視覺-語言提示 REFER TO ANYTHING 

**Authors**: Shengcao Cao, Zijun Wei, Jason Kuen, Kangning Liu, Lingzhi Zhang, Jiuxiang Gu, HyunJoon Jung, Liang-Yan Gui, Yu-Xiong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05342)  

**Abstract**: Recent image segmentation models have advanced to segment images into high-quality masks for visual entities, and yet they cannot provide comprehensive semantic understanding for complex queries based on both language and vision. This limitation reduces their effectiveness in applications that require user-friendly interactions driven by vision-language prompts. To bridge this gap, we introduce a novel task of omnimodal referring expression segmentation (ORES). In this task, a model produces a group of masks based on arbitrary prompts specified by text only or text plus reference visual entities. To address this new challenge, we propose a novel framework to "Refer to Any Segmentation Mask Group" (RAS), which augments segmentation models with complex multimodal interactions and comprehension via a mask-centric large multimodal model. For training and benchmarking ORES models, we create datasets MaskGroups-2M and MaskGroups-HQ to include diverse mask groups specified by text and reference entities. Through extensive evaluation, we demonstrate superior performance of RAS on our new ORES task, as well as classic referring expression segmentation (RES) and generalized referring expression segmentation (GRES) tasks. Project page: this https URL. 

**Abstract (ZH)**: 近期的图像分割模型已经能够将图像分割为高质量的掩码，但它们无法基于语言和视觉的综合查询提供全面的语义理解。这一限制降低了它们在需要由视觉语言提示驱动的友好交互应用中的有效性。为弥补这一不足，我们引入了一种新的跨模态引用表达分割（ORES）任务。在此任务中，模型根据仅文本或文本加参考视觉实体的任意提示生成一组掩码。为应对这一新挑战，我们提出了一种新的框架“任意分割掩码组的引用”（RAS），该框架通过一种以掩码为中心的大规模跨模态模型增强了分割模型的复杂多模态交互与理解能力。为了训练和评估ORES模型，我们构建了包含文本和参考实体指定的多样掩码组的MaskGroups-2M和MaskGroups-HQ数据集。通过广泛的评估，我们展示了RAS在我们的新ORES任务以及经典引用表达分割（RES）和通用引用表达分割（GRES）任务上的优越性能。项目页面：https://this-url。 

---
# Direct Numerical Layout Generation for 3D Indoor Scene Synthesis via Spatial Reasoning 

**Title (ZH)**: 直接基于空间推理的3D室内场景合成布局生成 

**Authors**: Xingjian Ran, Yixuan Li, Linning Xu, Mulin Yu, Bo Dai  

**Link**: [PDF](https://arxiv.org/pdf/2506.05341)  

**Abstract**: Realistic 3D indoor scene synthesis is vital for embodied AI and digital content creation. It can be naturally divided into two subtasks: object generation and layout generation. While recent generative models have significantly advanced object-level quality and controllability, layout generation remains challenging due to limited datasets. Existing methods either overfit to these datasets or rely on predefined constraints to optimize numerical layout that sacrifice flexibility. As a result, they fail to generate scenes that are both open-vocabulary and aligned with fine-grained user instructions. We introduce DirectLayout, a framework that directly generates numerical 3D layouts from text descriptions using generalizable spatial reasoning of large language models (LLMs). DirectLayout decomposes the generation into three stages: producing a Bird's-Eye View (BEV) layout, lifting it into 3D space, and refining object placements. To enable explicit spatial reasoning and help the model grasp basic principles of object placement, we employ Chain-of-Thought (CoT) Activation based on the 3D-Front dataset. Additionally, we design CoT-Grounded Generative Layout Reward to enhance generalization and spatial planning. During inference, DirectLayout addresses asset-layout mismatches via Iterative Asset-Layout Alignment through in-context learning. Extensive experiments demonstrate that DirectLayout achieves impressive semantic consistency, generalization and physical plausibility. 

**Abstract (ZH)**: 现实istic 3D室内场景合成对于沉浸式AI和数字内容创作至关重要。它可以自然地分为两个子任务：对象生成和布局生成。尽管最近的生成模型在对象级的质量和可控性方面取得了显著进步，但由于数据集有限，布局生成仍然具有挑战性。现有方法要么过度拟合这些数据集，要么依赖预定义的约束来优化数值布局，牺牲了灵活性。因此，它们无法生成既符合宽泛词汇表又与精细用户指令对齐的场景。我们引入了DirectLayout框架，该框架直接使用大型语言模型（LLMs）的通用空间推理从文本描述生成数值3D布局。DirectLayout将生成过程分解为三个阶段：生成鸟瞰图（BEV）布局、将其提升到3D空间，并细化对象放置。为使空间推理显式化并帮助模型掌握对象放置的基本原则，我们基于3D-Front数据集使用Chain-of-Thought（CoT）激活。此外，我们设计了CoT-Grounded生成布局奖励，以增强泛化能力和空间规划。在推理过程中，DirectLayout通过上下文学习迭代资产-布局对齐来解决资产-布局不匹配问题。广泛的实验表明，DirectLayout实现了令人印象深刻的语义一致性、泛化能力和物理合理性。 

---
# Exploring Diffusion Transformer Designs via Grafting 

**Title (ZH)**: 通过嫁接探索扩散变压器设计 

**Authors**: Keshigeyan Chandrasegaran, Michael Poli, Daniel Y. Fu, Dongjun Kim, Lea M. Hadzic, Manling Li, Agrim Gupta, Stefano Massaroli, Azalia Mirhoseini, Juan Carlos Niebles, Stefano Ermon, Li Fei-Fei  

**Link**: [PDF](https://arxiv.org/pdf/2506.05340)  

**Abstract**: Designing model architectures requires decisions such as selecting operators (e.g., attention, convolution) and configurations (e.g., depth, width). However, evaluating the impact of these decisions on model quality requires costly pretraining, limiting architectural investigation. Inspired by how new software is built on existing code, we ask: can new architecture designs be studied using pretrained models? To this end, we present grafting, a simple approach for editing pretrained diffusion transformers (DiTs) to materialize new architectures under small compute budgets. Informed by our analysis of activation behavior and attention locality, we construct a testbed based on the DiT-XL/2 design to study the impact of grafting on model quality. Using this testbed, we develop a family of hybrid designs via grafting: replacing softmax attention with gated convolution, local attention, and linear attention, and replacing MLPs with variable expansion ratio and convolutional variants. Notably, many hybrid designs achieve good quality (FID: 2.38-2.64 vs. 2.27 for DiT-XL/2) using <2% pretraining compute. We then graft a text-to-image model (PixArt-Sigma), achieving a 1.43x speedup with less than a 2% drop in GenEval score. Finally, we present a case study that restructures DiT-XL/2 by converting every pair of sequential transformer blocks into parallel blocks via grafting. This reduces model depth by 2x and yields better quality (FID: 2.77) than other models of comparable depth. Together, we show that new diffusion model designs can be explored by grafting pretrained DiTs, with edits ranging from operator replacement to architecture restructuring. Code and grafted models: this https URL 

**Abstract (ZH)**: 预训练模型设计的新架构研究： grafting 方法探索 

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
# Rectified Point Flow: Generic Point Cloud Pose Estimation 

**Title (ZH)**: 修正点流：通用点云姿态估计 

**Authors**: Tao Sun, Liyuan Zhu, Shengyu Huang, Shuran Song, Iro Armeni  

**Link**: [PDF](https://arxiv.org/pdf/2506.05282)  

**Abstract**: We introduce Rectified Point Flow, a unified parameterization that formulates pairwise point cloud registration and multi-part shape assembly as a single conditional generative problem. Given unposed point clouds, our method learns a continuous point-wise velocity field that transports noisy points toward their target positions, from which part poses are recovered. In contrast to prior work that regresses part-wise poses with ad-hoc symmetry handling, our method intrinsically learns assembly symmetries without symmetry labels. Together with a self-supervised encoder focused on overlapping points, our method achieves a new state-of-the-art performance on six benchmarks spanning pairwise registration and shape assembly. Notably, our unified formulation enables effective joint training on diverse datasets, facilitating the learning of shared geometric priors and consequently boosting accuracy. Project page: this https URL. 

**Abstract (ZH)**: 我们介绍了一种统一参数化方法——校正点流，该方法将点云配对注册和多部件形状组装统一为一个条件生成问题。给定未对齐的点云，我们的方法学习一个连续的点导向速度场，将噪声点引导到目标位置，从而恢复部件姿态。与之前需要人工处理对称性的方法不同，我们的方法内在地学习组装对称性而无需对称性标签。结合一个专注于重叠点的自监督编码器，我们的方法在六个涵盖配对注册和形状组装的基准测试中达到了新的最先进性能。值得注意的是，我们的统一框架使我们的方法可以在多种数据集上有效联合训练，促进了共享几何先验的学习，从而提升了准确性。项目页面: 这里 

---
# Fast-DataShapley: Neural Modeling for Training Data Valuation 

**Title (ZH)**: Fast-DataShapley: 基于神经网络的训练数据价值评估模型 

**Authors**: Haifeng Sun, Yu Xiong, Runze Wu, Xinyu Cai, Changjie Fan, Lan Zhang, Xiang-Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.05281)  

**Abstract**: The value and copyright of training data are crucial in the artificial intelligence industry. Service platforms should protect data providers' legitimate rights and fairly reward them for their contributions. Shapley value, a potent tool for evaluating contributions, outperforms other methods in theory, but its computational overhead escalates exponentially with the number of data providers. Recent works based on Shapley values attempt to mitigate computation complexity by approximation algorithms. However, they need to retrain for each test sample, leading to intolerable costs. We propose Fast-DataShapley, a one-pass training method that leverages the weighted least squares characterization of the Shapley value to train a reusable explainer model with real-time reasoning speed. Given new test samples, no retraining is required to calculate the Shapley values of the training data. Additionally, we propose three methods with theoretical guarantees to reduce training overhead from two aspects: the approximate calculation of the utility function and the group calculation of the training data. We analyze time complexity to show the efficiency of our methods. The experimental evaluations on various image datasets demonstrate superior performance and efficiency compared to baselines. Specifically, the performance is improved to more than 2.5 times, and the explainer's training speed can be increased by two orders of magnitude. 

**Abstract (ZH)**: 训练数据的价值和版权在人工智能行业中至关重要。服务平台应保护数据提供者的合法权益，并公平地奖励他们的贡献。Shapley值作为评估贡献的强大工具，在理论上优于其他方法，但其计算开销随数据提供者数量的增加呈指数级上升。基于Shapley值的近期研究试图通过近似算法减轻计算复杂性，但它们需要为每个测试样本重新训练，导致不可接受的成本。我们提出Fast-DataShapley，这是一种一次训练方法，通过利用Shapley值的加权最小二乘表示来训练一个可重用的解释模型，并具备实时推理速度。对于新的测试样本，无需重新训练即可计算训练数据的Shapley值。此外，我们提出三种具有理论保证的方法，从两个方面减少训练开销：近似计算效用函数和批量计算训练数据。我们通过时间复杂度分析展示了方法的效率。在多种图像数据集上的实验评估表明，与基准方法相比，我们的方法在性能和效率方面都更优。具体而言，性能提高了2.5倍以上，解释器的训练速度提高了两个数量级。 

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
# MesaNet: Sequence Modeling by Locally Optimal Test-Time Training 

**Title (ZH)**: MesaNet：本地最优测试时训练的序列建模 

**Authors**: Johannes von Oswald, Nino Scherrer, Seijin Kobayashi, Luca Versari, Songlin Yang, Maximilian Schlegel, Kaitlin Maile, Yanick Schimpf, Oliver Sieberling, Alexander Meulemans, Rif A. Saurous, Guillaume Lajoie, Charlotte Frenkel, Razvan Pascanu, Blaise Agüera y Arcas, João Sacramento  

**Link**: [PDF](https://arxiv.org/pdf/2506.05233)  

**Abstract**: Sequence modeling is currently dominated by causal transformer architectures that use softmax self-attention. Although widely adopted, transformers require scaling memory and compute linearly during inference. A recent stream of work linearized the softmax operation, resulting in powerful recurrent neural network (RNN) models with constant memory and compute costs such as DeltaNet, Mamba or xLSTM. These models can be unified by noting that their recurrent layer dynamics can all be derived from an in-context regression objective, approximately optimized through an online learning rule. Here, we join this line of work and introduce a numerically stable, chunkwise parallelizable version of the recently proposed Mesa layer (von Oswald et al., 2024), and study it in language modeling at the billion-parameter scale. This layer again stems from an in-context loss, but which is now minimized to optimality at every time point using a fast conjugate gradient solver. Through an extensive suite of experiments, we show that optimal test-time training enables reaching lower language modeling perplexity and higher downstream benchmark performance than previous RNNs, especially on tasks requiring long context understanding. This performance gain comes at the cost of additional flops spent during inference time. Our results are therefore intriguingly related to recent trends of increasing test-time compute to improve performance -- here by spending compute to solve sequential optimization problems within the neural network itself. 

**Abstract (ZH)**: 基于上下文回归目标驱动的数值稳定且可分块并行的Mesa层在大规模语言模型中的研究 

---
# Mitigating Degree Bias Adaptively with Hard-to-Learn Nodes in Graph Contrastive Learning 

**Title (ZH)**: 基于图对比学习中难以学习节点的自适应缓解度分布偏差 

**Authors**: Jingyu Hu, Hongbo Bo, Jun Hong, Xiaowei Liu, Weiru Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05214)  

**Abstract**: Graph Neural Networks (GNNs) often suffer from degree bias in node classification tasks, where prediction performance varies across nodes with different degrees. Several approaches, which adopt Graph Contrastive Learning (GCL), have been proposed to mitigate this bias. However, the limited number of positive pairs and the equal weighting of all positives and negatives in GCL still lead to low-degree nodes acquiring insufficient and noisy information. This paper proposes the Hardness Adaptive Reweighted (HAR) contrastive loss to mitigate degree bias. It adds more positive pairs by leveraging node labels and adaptively weights positive and negative pairs based on their learning hardness. In addition, we develop an experimental framework named SHARP to extend HAR to a broader range of scenarios. Both our theoretical analysis and experiments validate the effectiveness of SHARP. The experimental results across four datasets show that SHARP achieves better performance against baselines at both global and degree levels. 

**Abstract (ZH)**: Graph神经网络（GNNs）在节点分类任务中常常遭受度偏差的问题，其中预测性能在具有不同度的节点之间有所不同。已经提出了几种采用图对比学习（GCL）的方法来缓解这种偏差。然而，GCL中正样本对数量有限且正负样本等权重分配仍然导致低度节点获得不足且噪声较大的信息。本文提出了一种适应难度加权（HAR）对比损失，通过利用节点标签增加了更多的正样本对，并根据正负样本的学习难度自适应地分配权重。此外，我们开发了一个命名为SHARP的实验框架，将HAR扩展到更广泛的情景中。我们的理论分析和实验结果都验证了SHARP的有效性。跨四个数据集的实验结果表明，SHARP在全局和度的层面上都优于baseline方法。 

---
# Intentionally Unintentional: GenAI Exceptionalism and the First Amendment 

**Title (ZH)**: 有意的无意：GenAI特设论与第一修正案 

**Authors**: David Atkinson, Jena D. Hwang, Jacob Morrison  

**Link**: [PDF](https://arxiv.org/pdf/2506.05211)  

**Abstract**: This paper challenges the assumption that courts should grant First Amendment protections to outputs from large generative AI models, such as GPT-4 and Gemini. We argue that because these models lack intentionality, their outputs do not constitute speech as understood in the context of established legal precedent, so there can be no speech to protect. Furthermore, if the model outputs are not speech, users cannot claim a First Amendment speech right to receive the outputs. We also argue that extending First Amendment rights to AI models would not serve the fundamental purposes of free speech, such as promoting a marketplace of ideas, facilitating self-governance, or fostering self-expression. In fact, granting First Amendment protections to AI models would be detrimental to society because it would hinder the government's ability to regulate these powerful technologies effectively, potentially leading to the unchecked spread of misinformation and other harms. 

**Abstract (ZH)**: 本文挑战了法院应给予大型生成型人工智能模型如GPT-4和Gemini的输出第一修正案保护的假设。我们argue认为，由于这些模型缺乏意图性，它们的输出并不构成在现有法律先例下所理解的言论，因此不存在可保护的言论。此外，如果模型输出并非言论，用户无法主张第一修正案的言论权利以获取这些输出。我们还argue认为，将第一修正案权利扩展到AI模型不会服务于言论自由的基本目的，如促进思想市场的繁荣、促进自治或促进自我表达。实际上，给予AI模型第一修正案保护反而会对社会有害，因为它会妨碍政府有效监管这些强大技术的能力，可能导致信息误导和其他危害的不受控制的传播。 

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
# Survey on the Evaluation of Generative Models in Music 

**Title (ZH)**: 音乐生成模型评估研究 

**Authors**: Alexander Lerch, Claire Arthur, Nick Bryan-Kinns, Corey Ford, Qianyi Sun, Ashvala Vinay  

**Link**: [PDF](https://arxiv.org/pdf/2506.05104)  

**Abstract**: Research on generative systems in music has seen considerable attention and growth in recent years. A variety of attempts have been made to systematically evaluate such systems. We provide an interdisciplinary review of the common evaluation targets, methodologies, and metrics for the evaluation of both system output and model usability, covering subjective and objective approaches, qualitative and quantitative approaches, as well as empirical and computational methods. We discuss the advantages and challenges of such approaches from a musicological, an engineering, and an HCI perspective. 

**Abstract (ZH)**: 近年来，音乐生成系统的研究受到了 considerable 关注和增长。已经尝试了各种方法系统性地评估这些系统。本文从音乐学、工程学和人机交互角度综述了评估系统输出和模型可用性的共同评价目标、方法和指标，涵盖了主观与客观方法、定性与定量方法以及经验性与计算性方法，并讨论了这些方法的优势和挑战。 

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
# TIMING: Temporality-Aware Integrated Gradients for Time Series Explanation 

**Title (ZH)**: 基于时间感知集成梯度的时间序列解释 

**Authors**: Hyeongwon Jang, Changhun Kim, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05035)  

**Abstract**: Recent explainable artificial intelligence (XAI) methods for time series primarily estimate point-wise attribution magnitudes, while overlooking the directional impact on predictions, leading to suboptimal identification of significant points. Our analysis shows that conventional Integrated Gradients (IG) effectively capture critical points with both positive and negative impacts on predictions. However, current evaluation metrics fail to assess this capability, as they inadvertently cancel out opposing feature contributions. To address this limitation, we propose novel evaluation metrics-Cumulative Prediction Difference (CPD) and Cumulative Prediction Preservation (CPP)-to systematically assess whether attribution methods accurately identify significant positive and negative points in time series XAI. Under these metrics, conventional IG outperforms recent counterparts. However, directly applying IG to time series data may lead to suboptimal outcomes, as generated paths ignore temporal relationships and introduce out-of-distribution samples. To overcome these challenges, we introduce TIMING, which enhances IG by incorporating temporal awareness while maintaining its theoretical properties. Extensive experiments on synthetic and real-world time series benchmarks demonstrate that TIMING outperforms existing time series XAI baselines. Our code is available at this https URL. 

**Abstract (ZH)**: 近期用于时间序列的可解释人工智能(XAI)方法主要估计点 wise 归因 magnitude，而忽视了对预测的导向性影响，导致显著点的识别不够优化。我们的分析表明，传统的集成梯度(IG)能够有效地捕捉对预测既有正面又有负面影 响的关键点。然而，当前的评估指标未能评估这一能力，因为它们无意中抵消了特征贡献的对立面。为解决这一局限性，我们提出了新的评估指标——累计预测差异(CPD)和累计预测保留(CPP)，以系统地评估归因方法是否准确地识别出时间序列XAI中的重要正面和负面点。在这些指标下，传统的IG优于近期的同类方法。然而，直接将IG应用于时间序列数据可能会导致次优结果，因为生成的路径忽略了时间关系，并引入了分布外样本。为应对这些挑战，我们引入了TIMING，通过引入时间意识来增强IG，同时保持其理论特性。广泛的实验证明，TIMING 在合成和真实世界的时间序列基准上优于现有时间序列XAI基线。我们的代码可在以下链接获取。 

---
# Identifying and Understanding Cross-Class Features in Adversarial Training 

**Title (ZH)**: 识别和理解对抗训练中的跨类别特征 

**Authors**: Zeming Wei, Yiwen Guo, Yisen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05032)  

**Abstract**: Adversarial training (AT) has been considered one of the most effective methods for making deep neural networks robust against adversarial attacks, while the training mechanisms and dynamics of AT remain open research problems. In this paper, we present a novel perspective on studying AT through the lens of class-wise feature attribution. Specifically, we identify the impact of a key family of features on AT that are shared by multiple classes, which we call cross-class features. These features are typically useful for robust classification, which we offer theoretical evidence to illustrate through a synthetic data model. Through systematic studies across multiple model architectures and settings, we find that during the initial stage of AT, the model tends to learn more cross-class features until the best robustness checkpoint. As AT further squeezes the training robust loss and causes robust overfitting, the model tends to make decisions based on more class-specific features. Based on these discoveries, we further provide a unified view of two existing properties of AT, including the advantage of soft-label training and robust overfitting. Overall, these insights refine the current understanding of AT mechanisms and provide new perspectives on studying them. Our code is available at this https URL. 

**Abstract (ZH)**: 对抗训练（AT）被认为是使深度神经网络对对抗攻击具有鲁棒性的最有效方法之一，尽管AT的训练机制和动态仍然是开放的研究问题。在本文中，我们通过类内特征归因的视角提出了研究AT的全新观点。具体而言，我们识别了在多个类别间共享的关键特征家族对AT的影响，我们将这些特征称为跨类别特征。这些特征通常对鲁棒分类是有用的，我们通过合成数据模型提供了理论证据来说明这一点。通过在多种模型架构和设置下的系统研究，我们发现，在AT的初始阶段，模型倾向于更多地学习跨类别特征，直到达到最佳鲁棒性检查点。随着AT进一步压缩训练的鲁棒损失并导致鲁棒过拟合，模型倾向于基于更多类别特定特征来做决策。基于这些发现，我们进一步提供了一种对AT现有两个性质的统一观点，包括软标签训练的优势和鲁棒过拟合。总体而言，这些见解细化了对AT机制的理解，并提供了研究它们的新视角。我们的代码可在以下链接获取：这个https链接。 

---
# Artificial Intelligence Should Genuinely Support Clinical Reasoning and Decision Making To Bridge the Translational Gap 

**Title (ZH)**: 人工智能应真正支持临床推理和决策以桥接 translational 静默处的gap 

**Authors**: Kacper Sokol, James Fackler, Julia E Vogt  

**Link**: [PDF](https://arxiv.org/pdf/2506.05030)  

**Abstract**: Artificial intelligence promises to revolutionise medicine, yet its impact remains limited because of the pervasive translational gap. We posit that the prevailing technology-centric approaches underpin this challenge, rendering such systems fundamentally incompatible with clinical practice, specifically diagnostic reasoning and decision making. Instead, we propose a novel sociotechnical conceptualisation of data-driven support tools designed to complement doctors' cognitive and epistemic activities. Crucially, it prioritises real-world impact over superhuman performance on inconsequential benchmarks. 

**Abstract (ZH)**: 人工智能有望革新医学，但其影响受限于普遍存在的转化_gap。我们提出，现行的技术导向方法是这一挑战的根源，使得这些系统与临床实践，特别是诊断推理和决策制定，根本无法兼容。相反，我们建议一种新的社会技术概念化数据驱动支持工具，旨在补充医生的认知和知识活动。关键在于强调现实世界的影响而非在无关基准上的超人表现。 

---
# Hierarchical Language Models for Semantic Navigation and Manipulation in an Aerial-Ground Robotic System 

**Title (ZH)**: 基于层次语言模型的空地机器人系统中语义导航与操控 

**Authors**: Haokun Liu, Zhaoqi Ma, Yunong Li, Junichiro Sugihara, Yicheng Chen, Jinjie Li, Moju Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.05020)  

**Abstract**: Heterogeneous multi-robot systems show great potential in complex tasks requiring coordinated hybrid cooperation. However, traditional approaches relying on static models often struggle with task diversity and dynamic environments. This highlights the need for generalizable intelligence that can bridge high-level reasoning with low-level execution across heterogeneous agents. To address this, we propose a hierarchical framework integrating a prompted Large Language Model (LLM) and a GridMask-enhanced fine-tuned Vision Language Model (VLM). The LLM performs task decomposition and global semantic map construction, while the VLM extracts task-specified semantic labels and 2D spatial information from aerial images to support local planning. Within this framework, the aerial robot follows a globally optimized semantic path and continuously provides bird-view images, guiding the ground robot's local semantic navigation and manipulation, including target-absent scenarios where implicit alignment is maintained. Experiments on a real-world letter-cubes arrangement task demonstrate the framework's adaptability and robustness in dynamic environments. To the best of our knowledge, this is the first demonstration of an aerial-ground heterogeneous system integrating VLM-based perception with LLM-driven task reasoning and motion planning. 

**Abstract (ZH)**: 异构多机器人系统在需要协调混合合作的复杂任务中显示出巨大的潜力。然而，传统的依赖静态模型的方法往往难以应对任务多样性和动态环境。这强调了需要一种可泛化的智能，能够跨越异构代理的高层推理与低层执行进行桥梁构建。为了解决这个问题，我们提出了一种层次框架，该框架整合了一个被提示的大语言模型（LLM）和一个GridMask增强的微调视觉语言模型（VLM）。LLM负责任务分解和全局语义地图构建，而VLM从空中图像中提取任务特定的语义标签和二维空间信息，以支持局部规划。在该框架中，空中机器人遵循全局优化的语义路径，并持续提供鸟瞰图图像，引导地面机器人进行局部语义导航和操作，包括目标缺席的场景，其中保持隐式的对齐。在现实世界的字母立方体排列任务上的实验表明，该框架在动态环境中的适应性和鲁棒性。据我们所知，这是首次将基于VLM的感知与LLM驱动的任务推理和运动规划集成到空中-地面异构系统中的演示。 

---
# Towards Reasonable Concept Bottleneck Models 

**Title (ZH)**: 合理概念瓶颈模型研究 

**Authors**: Nektarios Kalampalikis, Kavya Gupta, Georgi Vitanov, Isabel Valera  

**Link**: [PDF](https://arxiv.org/pdf/2506.05014)  

**Abstract**: In this paper, we propose $\textbf{C}$oncept $\textbf{REA}$soning $\textbf{M}$odels (CREAM), a novel family of Concept Bottleneck Models (CBMs) that: (i) explicitly encodes concept-concept (${\texttt{C-C}}$) and concept-task (${\texttt{C$\rightarrow$Y}}$) relationships to enforce a desired model reasoning; and (ii) use a regularized side-channel to achieve competitive task performance, while keeping high concept importance. Specifically, CREAM architecturally embeds (bi)directed concept-concept, and concept to task relationships specified by a human expert, while severing undesired information flows (e.g., to handle mutually exclusive concepts). Moreover, CREAM integrates a black-box side-channel that is regularized to encourage task predictions to be grounded in the relevant concepts, thereby utilizing the side-channel only when necessary to enhance performance. Our experiments show that: (i) CREAM mainly relies on concepts while achieving task performance on par with black-box models; and (ii) the embedded ${\texttt{C-C}}$ and ${\texttt{C$\rightarrow$Y}}$ relationships ease model interventions and mitigate concept leakage. 

**Abstract (ZH)**: 基于概念推理的模型(CREAM)：一种新型的概念瓶颈模型家族 

---
# A Multi-Dataset Evaluation of Models for Automated Vulnerability Repair 

**Title (ZH)**: 多数据集评估的自动化漏洞修复模型研究 

**Authors**: Zanis Ali Khan, Aayush Garg, Qiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04987)  

**Abstract**: Software vulnerabilities pose significant security threats, requiring effective mitigation. While Automated Program Repair (APR) has advanced in fixing general bugs, vulnerability patching, a security-critical aspect of APR remains underexplored. This study investigates pre-trained language models, CodeBERT and CodeT5, for automated vulnerability patching across six datasets and four languages. We evaluate their accuracy and generalization to unknown vulnerabilities. Results show that while both models face challenges with fragmented or sparse context, CodeBERT performs comparatively better in such scenarios, whereas CodeT5 excels in capturing complex vulnerability patterns. CodeT5 also demonstrates superior scalability. Furthermore, we test fine-tuned models on both in-distribution (trained) and out-of-distribution (unseen) datasets. While fine-tuning improves in-distribution performance, models struggle to generalize to unseen data, highlighting challenges in robust vulnerability detection. This study benchmarks model performance, identifies limitations in generalization, and provides actionable insights to advance automated vulnerability patching for real-world security applications. 

**Abstract (ZH)**: 预训练语言模型在跨六大数据集和四种语言的自动漏洞修补中的研究 

---
# From Struggle (06-2024) to Mastery (02-2025) LLMs Conquer Advanced Algorithm Exams and Pave the Way for Editorial Generation 

**Title (ZH)**: 从挣扎（06-2024）到精通（02-2025）：大型语言模型征服高级算法考试并为编辑生成铺平道路 

**Authors**: Adrian Marius Dumitran, Theodor-Pierre Moroianu, Vasile Paul Alexe  

**Link**: [PDF](https://arxiv.org/pdf/2506.04965)  

**Abstract**: This paper presents a comprehensive evaluation of the performance of state-of-the-art Large Language Models (LLMs) on challenging university-level algorithms exams. By testing multiple models on both a Romanian exam and its high-quality English translation, we analyze LLMs' problem-solving capabilities, consistency, and multilingual performance. Our empirical study reveals that the most recent models not only achieve scores comparable to top-performing students but also demonstrate robust reasoning skills on complex, multi-step algorithmic challenges, even though difficulties remain with graph-based tasks. Building on these findings, we explore the potential of LLMs to support educational environments through the generation of high-quality editorial content, offering instructors a powerful tool to enhance student feedback. The insights and best practices discussed herein pave the way for further integration of generative AI in advanced algorithm education. 

**Abstract (ZH)**: 本研究全面评估了最先进大型语言模型在大学级算法考试中的 performance。通过在罗马尼亚考题及其高质量英语翻译版本上测试多个模型，我们分析了语言模型的问题解决能力、一致性和多语言性能。我们的实证研究显示，最新的模型不仅在成绩上可与顶尖学生媲美，还在复杂多步算法挑战中展示了稳健的推理能力，尽管在图任务方面仍有困难。基于这些发现，我们探讨了大型语言模型在通过生成高质量编辑内容支持教育环境方面的潜力，为教师提供了一种增强学生反馈的强大工具。文中讨论的见解和最佳实践为生成式AI在高级算法教育中的进一步整合铺平了道路。 

---
# Robustness as Architecture: Designing IQA Models to Withstand Adversarial Perturbations 

**Title (ZH)**: 鲁棒性作为架构：设计能抵抗对抗性扰动的IQA模型 

**Authors**: Igor Meleshin, Anna Chistyakova, Anastasia Antsiferova, Dmitriy Vatolin  

**Link**: [PDF](https://arxiv.org/pdf/2506.04951)  

**Abstract**: Image Quality Assessment (IQA) models are increasingly relied upon to evaluate image quality in real-world systems -- from compression and enhancement to generation and streaming. Yet their adoption brings a fundamental risk: these models are inherently unstable. Adversarial manipulations can easily fool them, inflating scores and undermining trust. Traditionally, such vulnerabilities are addressed through data-driven defenses -- adversarial retraining, regularization, or input purification. But what if this is the wrong lens? What if robustness in perceptual models is not something to learn but something to design? In this work, we propose a provocative idea: robustness as an architectural prior. Rather than training models to resist perturbations, we reshape their internal structure to suppress sensitivity from the ground up. We achieve this by enforcing orthogonal information flow, constraining the network to norm-preserving operations -- and further stabilizing the system through pruning and fine-tuning. The result is a robust IQA architecture that withstands adversarial attacks without requiring adversarial training or significant changes to the original model. This approach suggests a shift in perspective: from optimizing robustness through data to engineering it through design. 

**Abstract (ZH)**: 感知模型稳健性作为架构先验的图像品质评估（IQA）模型 

---
# CzechLynx: A Dataset for Individual Identification and Pose Estimation of the Eurasian Lynx 

**Title (ZH)**: CzechLynx：欧亚猞猁的个体识别与姿态估计数据集 

**Authors**: Lukas Picek, Elisa Belotti, Michal Bojda, Ludek Bufka, Vojtech Cermak, Martin Dula, Rostislav Dvorak, Luboslav Hrdy, Miroslav Jirik, Vaclav Kocourek, Josefa Krausova, Jirı Labuda, Jakub Straka, Ludek Toman, Vlado Trulık, Martin Vana, Miroslav Kutal  

**Link**: [PDF](https://arxiv.org/pdf/2506.04931)  

**Abstract**: We introduce CzechLynx, the first large-scale, open-access dataset for individual identification, 2D pose estimation, and instance segmentation of the Eurasian lynx (Lynx lynx). CzechLynx includes more than 30k camera trap images annotated with segmentation masks, identity labels, and 20-point skeletons and covers 219 unique individuals across 15 years of systematic monitoring in two geographically distinct regions: Southwest Bohemia and the Western Carpathians. To increase the data variability, we create a complementary synthetic set with more than 100k photorealistic images generated via a Unity-based pipeline and diffusion-driven text-to-texture modeling, covering diverse environments, poses, and coat-pattern variations. To allow testing generalization across spatial and temporal domains, we define three tailored evaluation protocols/splits: (i) geo-aware, (ii) time-aware open-set, and (iii) time-aware closed-set. This dataset is targeted to be instrumental in benchmarking state-of-the-art models and the development of novel methods for not just individual animal re-identification. 

**Abstract (ZH)**: 我们介绍CzechLynx，首个大规模、开源的用于欧亚猞猁（Lynx lynx）个体识别、2D姿态估计和实例分割的数据集。CzechLynx 包含超过30,000张带标注分割掩膜、身份标签及20点骨架结构的相机陷阱图像，并涵盖了跨越15年系统监测的两个地理上不同区域（Southwest Bohemia和Western Carpathians）的219个独特个体。为了增加数据变异性，我们创建了一个补充合成数据集，包含超过100,000张逼真的合成图像，通过基于Unity的流水线和扩散驱动的文本到纹理建模生成，涵盖了多种环境、姿态和毛皮花纹的变化。为允许在空间和时间域上进行泛化测试，我们定义了三个定制的评估协议/划分：(i) 地理意识型、(ii) 时间意识的开放集、(iii) 时间意识的封闭集。该数据集旨在用于基准测试最先进的模型和开发新的方法，不仅用于个体动物重新识别。 

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
# Towards Network Data Analytics in 5G Systems and Beyond 

**Title (ZH)**: 向5G系统及更远未来的网络数据分析迈进 

**Authors**: Marcos Lima Romero, Ricardo Suyama  

**Link**: [PDF](https://arxiv.org/pdf/2506.04860)  

**Abstract**: Data has become a critical asset in the digital economy, yet it remains underutilized by Mobile Network Operators (MNOs), unlike Over-the-Top (OTT) players that lead global market valuations. To move beyond the commoditization of connectivity and deliver greater value to customers, data analytics emerges as a strategic enabler. Using data efficiently is essential for unlocking new service opportunities, optimizing operational efficiency, and mitigating operational and business risks. Since Release 15, the 3rd Generation Partnership Project (3GPP) has introduced the Network Data Analytics Function (NWDAF) to provide powerful insights and predictions using data collected across mobile networks, supporting both user-centric and network-oriented use cases. However, academic research has largely focused on a limited set of methods and use cases, driven by the availability of datasets, restricting broader exploration. This study analyzes trends and gaps in more than 70 articles and proposes two novel use cases to promote the adoption of NWDAF and explore its potential for monetization. 

**Abstract (ZH)**: 数据已成为数字经济中的关键资产，然而移动网络运营商（MNOs）仍未充分利用数据，相比之下，Over-the-Top（OTT）玩家主导了全球市场估值。为了超越连接服务的同质化并且为客户提供更大的价值，数据分析师工具成为战略性的使能器。有效地利用数据对于解锁新的服务机会、优化运营效率以及减轻运营和业务风险至关重要。自R15起，3GPP引入了网络数据分析功能（NWDAF），利用移动网络中收集的数据提供强大的洞察和预测，支持用户中心和网络导向的应用场景。然而，学术研究主要集中在有限的方法和应用案例上，这限制了更广泛的研究探索。本研究分析了逾70篇论文的趋势和空白，并提出两个新型应用案例，以促进NWDAF的采用并探索其潜在的商业化可能性。 

---
# Sparse Autoencoders, Again? 

**Title (ZH)**: 稀疏自编码器，再次登场？ 

**Authors**: Yin Lu, Tong He, Xuening Zhu, David Wipf  

**Link**: [PDF](https://arxiv.org/pdf/2506.04859)  

**Abstract**: Is there really much more to say about sparse autoencoders (SAEs)? Autoencoders in general, and SAEs in particular, represent deep architectures that are capable of modeling low-dimensional latent structure in data. Such structure could reflect, among other things, correlation patterns in large language model activations, or complex natural image manifolds. And yet despite the wide-ranging applicability, there have been relatively few changes to SAEs beyond the original recipe from decades ago, namely, standard deep encoder/decoder layers trained with a classical/deterministic sparse regularizer applied within the latent space. One possible exception is the variational autoencoder (VAE), which adopts a stochastic encoder module capable of producing sparse representations when applied to manifold data. In this work we formalize underappreciated weaknesses with both canonical SAEs, as well as analogous VAEs applied to similar tasks, and propose a hybrid alternative model that circumvents these prior limitations. In terms of theoretical support, we prove that global minima of our proposed model recover certain forms of structured data spread across a union of manifolds. Meanwhile, empirical evaluations on synthetic and real-world datasets substantiate the efficacy of our approach in accurately estimating underlying manifold dimensions and producing sparser latent representations without compromising reconstruction error. In general, we are able to exceed the performance of equivalent-capacity SAEs and VAEs, as well as recent diffusion models where applicable, within domains such as images and language model activation patterns. 

**Abstract (ZH)**: 关于稀疏自编码器(SAEs)真的还有很多可说的内容吗？：稀疏自编码器作为能够建模数据低维潜在结构的深层架构，在一般情况下，特别是SAEs中，这种结构可以反映大规模语言模型激活中的相关模式，或复杂自然图像流形。然而，尽管应用范围广泛，自几十年前的原始配方以来，SAEs的变化相对较少，主要是在潜在空间中应用标准的深度编码/解码层和经典的确定性稀疏正则化。唯一的例外可能是变分自编码器（VAE），它采用了一个能够生成稀疏表示的随机编码模块。在本工作中，我们 formalize 了经典 SAEs 和应用于类似任务的类似 VAEs 的未被充分认识的局限性，并提出了一种新的混合模型来克服这些先前的限制。在理论支持方面，我们证明了我们所提出的模型的全局最小值能够恢复跨流形联合分布的某些结构数据形式。同时，对合成和真实世界数据集的经验评估证明了我们方法的有效性，它能够准确估计潜在流形维度并产生更稀疏的潜在表示，而不牺牲重构误差。总体而言，我们能够在图像和语言模型激活模式等领域超越具有相同容量的 SAEs 和 VAEs，以及适用的情况下超越最近的扩散模型。 

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
# Oversight Structures for Agentic AI in Public-Sector Organizations 

**Title (ZH)**: 公共部门组织中代理人工智能的监管结构 

**Authors**: Chris Schmitz, Jonathan Rystrøm, Jan Batzner  

**Link**: [PDF](https://arxiv.org/pdf/2506.04836)  

**Abstract**: This paper finds that the introduction of agentic AI systems intensifies existing challenges to traditional public sector oversight mechanisms -- which rely on siloed compliance units and episodic approvals rather than continuous, integrated supervision. We identify five governance dimensions essential for responsible agent deployment: cross-departmental implementation, comprehensive evaluation, enhanced security protocols, operational visibility, and systematic auditing. We evaluate the capacity of existing oversight structures to meet these challenges, via a mixed-methods approach consisting of a literature review and interviews with civil servants in AI-related roles. We find that agent oversight poses intensified versions of three existing governance challenges: continuous oversight, deeper integration of governance and operational capabilities, and interdepartmental coordination. We propose approaches that both adapt institutional structures and design agent oversight compatible with public sector constraints. 

**Abstract (ZH)**: 本文发现，代理型人工智能系统的引入加剧了传统公共部门监督机制面临的现有挑战——这些机制依赖于隔离的合规单位和 episodic 审批，而非连续性、集成的监督。我们确定了五项对于负责任地部署代理的关键治理维度：跨部门实施、全面评估、增强的安全协议、操作透明度以及系统性审计。我们通过包含文献回顾和对相关领域公务员的访谈的混合方法评估现有监督结构的能力，以应对这些挑战。我们发现，代理监督加剧了三个现有治理挑战：持续监督、治理和操作能力的更深层次整合以及跨部门协调。我们提出了既适应机构结构又与公共服务约束兼容的代理监督方案。 

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
# Truth in the Few: High-Value Data Selection for Efficient Multi-Modal Reasoning 

**Title (ZH)**: 少即是真：高效多模态推理中的高价值数据选择 

**Authors**: Shenshen Li, Kaiyuan Deng, Lei Wang, Hao Yang, Chong Peng, Peng Yan, Fumin Shen, Heng Tao Shen, Xing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04755)  

**Abstract**: While multi-modal large language models (MLLMs) have made significant progress in complex reasoning tasks via reinforcement learning, it is commonly believed that extensive training data is necessary for improving multi-modal reasoning ability, inevitably leading to data redundancy and substantial computational costs. However, can smaller high-value datasets match or outperform full corpora for multi-modal reasoning in MLLMs? In this work, we challenge this assumption through a key observation: meaningful multi-modal reasoning is triggered by only a sparse subset of training samples, termed cognitive samples, whereas the majority contribute marginally. Building on this insight, we propose a novel data selection paradigm termed Reasoning Activation Potential (RAP), which identifies cognitive samples by estimating each sample's potential to stimulate genuine multi-modal reasoning by two complementary estimators: 1) Causal Discrepancy Estimator (CDE) based on the potential outcome model principle, eliminates samples that overly rely on language priors by comparing outputs between multi-modal and text-only inputs; 2) Attention Confidence Estimator (ACE), which exploits token-level self-attention to discard samples dominated by irrelevant but over-emphasized tokens in intermediate reasoning stages. Moreover, we introduce a Difficulty-aware Replacement Module (DRM) to substitute trivial instances with cognitively challenging ones, thereby ensuring complexity for robust multi-modal reasoning. Experiments on six datasets show that our RAP method consistently achieves superior performance using only 9.3% of the training data, while reducing computational costs by over 43%. Our code is available at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型中的有限高价值数据能否匹配或超越全面语料库的多模态推理能力：基于推理激活潜力的新型数据选择范式 

---
# Physics Informed Capsule Enhanced Variational AutoEncoder for Underwater Image Enhancement 

**Title (ZH)**: 物理知情胶囊增强变分自编码器在水下图像增强中的应用 

**Authors**: Niki Martinel, Rita Pucci  

**Link**: [PDF](https://arxiv.org/pdf/2506.04753)  

**Abstract**: We present a novel dual-stream architecture that achieves state-of-the-art underwater image enhancement by explicitly integrating the Jaffe-McGlamery physical model with capsule clustering-based feature representation learning. Our method simultaneously estimates transmission maps and spatially-varying background light through a dedicated physics estimator while extracting entity-level features via capsule clustering in a parallel stream. This physics-guided approach enables parameter-free enhancement that respects underwater formation constraints while preserving semantic structures and fine-grained details. Our approach also features a novel optimization objective ensuring both physical adherence and perceptual quality across multiple spatial frequencies. To validate our approach, we conducted extensive experiments across six challenging benchmarks. Results demonstrate consistent improvements of $+0.5$dB PSNR over the best existing methods while requiring only one-third of their computational complexity (FLOPs), or alternatively, more than $+1$dB PSNR improvement when compared to methods with similar computational budgets. Code and data \textit{will} be available at this https URL. 

**Abstract (ZH)**: 我们提出了一种新的双流架构，通过显式集成Jaffe-McGlamery物理模型并与基于胶囊聚类的特征表示学习相结合，实现了水下图像增强的最新成果。该方法通过专用的物理估计器同时估计传输图和空间变背景光，在并行流中利用胶囊聚类提取实体级特征。这种基于物理的指导方法能够在尊重水下图像形成约束的同时保持语义结构和细粒度细节，并实现无参数的增强。我们的方法还具有新的优化目标，确保在多个空间频率上的物理一致性和感知质量。为了验证我们的方法，我们在六个具有挑战性的基准上进行了大量实验。结果表明，与现有最佳方法相比，我们的方法在峰值信噪比（PSNR）上提高了0.5 dB，同时仅需现有方法三分之一的计算复杂度（FLOPs），或在具有相似计算预算的方法中提高超过1 dB的PSNR。代码和数据将在此处提供。 

---
# Was Residual Penalty and Neural Operators All We Needed for Solving Optimal Control Problems? 

**Title (ZH)**: 残差罚函数和神经算子足以解决最优控制问题吗？ 

**Authors**: Oliver G. S. Lundqvist, Fabricio Oliveira  

**Link**: [PDF](https://arxiv.org/pdf/2506.04742)  

**Abstract**: Neural networks have been used to solve optimal control problems, typically by training neural networks using a combined loss function that considers data, differential equation residuals, and objective costs. We show that including cost functions in the training process is unnecessary, advocating for a simpler architecture and streamlined approach by decoupling the optimal control problem from the training process. Thus, our work shows that a simple neural operator architecture, such as DeepONet, coupled with an unconstrained optimization routine, can solve multiple optimal control problems with a single physics-informed training phase and a subsequent optimization phase. We achieve this by adding a penalty term based on the differential equation residual to the cost function and computing gradients with respect to the control using automatic differentiation through the trained neural operator within an iterative optimization routine. We showcase our method on nine distinct optimal control problems by training three separate DeepONet models, each corresponding to a different differential equation. For each model, we solve three problems with varying cost functions, demonstrating accurate and consistent performance across all cases. 

**Abstract (ZH)**: 神经网络已被用于解决最优控制问题，通常通过使用结合数据、微分方程残差和目标成本的损失函数来训练神经网络。我们表明，在训练过程中包含成本函数是不必要的，并且通过将最优控制问题与训练过程脱钩，采用一种更简单且简洁的架构可以解决问题。因此，我们的工作表明，结合无约束优化方法的简单神经算子架构，如DeepONet，可以在单次基于物理的训练阶段和后续的优化阶段解决多个最优控制问题。我们通过在成本函数中增加基于微分方程残差的惩罚项，并使用自动微分通过训练的神经算子在迭代优化过程中计算控制的梯度来实现这一点。我们在九个不同的最优控制问题上展示了我们的方法，通过训练三个分别对应不同微分方程的DeepONet模型。对于每个模型，我们解决了三个具有不同成本函数的问题，展示了所有情况下的准确且一致的性能。 

---
# Lifelong Evolution: Collaborative Learning between Large and Small Language Models for Continuous Emergent Fake News Detection 

**Title (ZH)**: lifelong 进化: 大型和小型语言模型之间的合作学习以实现连续涌现式假新闻检测 

**Authors**: Ziyi Zhou, Xiaoming Zhang, Litian Zhang, Yibo Zhang, Zhenyu Guan, Chaozhuo Li, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04739)  

**Abstract**: The widespread dissemination of fake news on social media has significantly impacted society, resulting in serious consequences. Conventional deep learning methodologies employing small language models (SLMs) suffer from extensive supervised training requirements and difficulties adapting to evolving news environments due to data scarcity and distribution shifts. Large language models (LLMs), despite robust zero-shot capabilities, fall short in accurately detecting fake news owing to outdated knowledge and the absence of suitable demonstrations. In this paper, we propose a novel Continuous Collaborative Emergent Fake News Detection (C$^2$EFND) framework to address these challenges. The C$^2$EFND framework strategically leverages both LLMs' generalization power and SLMs' classification expertise via a multi-round collaborative learning framework. We further introduce a lifelong knowledge editing module based on a Mixture-of-Experts architecture to incrementally update LLMs and a replay-based continue learning method to ensure SLMs retain prior knowledge without retraining entirely. Extensive experiments on Pheme and Twitter16 datasets demonstrate that C$^2$EFND significantly outperforms existed methods, effectively improving detection accuracy and adaptability in continuous emergent fake news scenarios. 

**Abstract (ZH)**: 大规模社交媒体上传播的假新闻对社会产生了重大影响， conventional deep learning 方法依赖小型语言模型（SLMs）由于监督训练需求大且难以适应不断变化的新闻环境，即由于数据稀缺和分布变化。尽管大型语言模型（LLMs）具有强大的零样本能力，但在检测假新闻时由于知识过时和缺乏合适的示例而表现不佳。本文提出了一种新的连续协作 emergent 假新闻检测（C$^2$EFND）框架，该框架通过多轮协作学习框架战略地利用了 LLMs 的泛化能力和 SLMs 的分类专长。本文还介绍了一种基于 Mixture-of-Experts 架构的终身知识编辑模块，用于增量更新 LLMs，并提出了基于回放的持续学习方法以确保 SLMs 在无需完全重新训练的情况下保留先前知识。在 Pheme 和 Twitter16 数据集上的广泛实验表明，C$^2$EFND 显著优于现有方法，在持续 emergent 假新闻检测场景中有效提高了检测准确性和适应性。 

---
# Using In-Context Learning for Automatic Defect Labelling of Display Manufacturing Data 

**Title (ZH)**: 基于上下文学习的显示制造数据自动缺陷标注 

**Authors**: Babar Hussain, Qiang Liu, Gang Chen, Bihai She, Dahai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04717)  

**Abstract**: This paper presents an AI-assisted auto-labeling system for display panel defect detection that leverages in-context learning capabilities. We adopt and enhance the SegGPT architecture with several domain-specific training techniques and introduce a scribble-based annotation mechanism to streamline the labeling process. Our two-stage training approach, validated on industrial display panel datasets, demonstrates significant improvements over the baseline model, achieving an average IoU increase of 0.22 and a 14% improvement in recall across multiple product types, while maintaining approximately 60% auto-labeling coverage. Experimental results show that models trained on our auto-labeled data match the performance of those trained on human-labeled data, offering a practical solution for reducing manual annotation efforts in industrial inspection systems. 

**Abstract (ZH)**: 基于上下文学习的AI辅助自动标注系统在显示面板缺陷检测中的应用研究 

---
# UNO: Unlearning via Orthogonalization in Generative models 

**Title (ZH)**: UNO：生成模型中的正交化去学习 

**Authors**: Pinak Mandal, Georg A. Gottwald  

**Link**: [PDF](https://arxiv.org/pdf/2506.04712)  

**Abstract**: As generative models become increasingly powerful and pervasive, the ability to unlearn specific data, whether due to privacy concerns, legal requirements, or the correction of harmful content, has become increasingly important. Unlike in conventional training, where data are accumulated and knowledge is reinforced, unlearning aims to selectively remove the influence of particular data points without costly retraining from scratch. To be effective and reliable, such algorithms need to achieve (i) forgetting of the undesired data, (ii) preservation of the quality of the generation, (iii) preservation of the influence of the desired training data on the model parameters, and (iv) small number of training steps. We propose fast unlearning algorithms based on loss gradient orthogonalization. We show that our algorithms are able to forget data while maintaining the fidelity of the original model. Using MNIST and CelebA data, we demonstrate that our algorithms achieve orders of magnitude faster unlearning times than their predecessors, such as gradient surgery. 

**Abstract (ZH)**: 随着生成模型变得越来越强大和普及，由于隐私问题、法律规定或有害内容的更正等原因，去除特定数据的能力变得越来越重要。与传统训练中数据积累和知识强化不同，去除特定数据旨在不进行昂贵的从头开始重新训练的情况下选择性地去除特定数据点的影响。为了有效和可靠，此类算法需要实现（i）遗忘不需要的数据，（ii）保持生成的质量，（iii）保持希望的训练数据对模型参数的影响，以及（iv）少量的训练步骤。我们提出了基于损失梯度正交化的快速去除算法。我们证明，我们的算法能够在保持原始模型保真度的同时遗忘数据。使用MNIST和CelebA数据，我们展示了我们的算法比前驱算法（如梯度手术）快多个数量级的去除时间。 

---
# Line of Sight: On Linear Representations in VLLMs 

**Title (ZH)**: 视觉射线：关于VLLMs中的线性表示 

**Authors**: Achyuta Rajaram, Sarah Schwettmann, Jacob Andreas, Arthur Conmy  

**Link**: [PDF](https://arxiv.org/pdf/2506.04706)  

**Abstract**: Language models can be equipped with multimodal capabilities by fine-tuning on embeddings of visual inputs. But how do such multimodal models represent images in their hidden activations? We explore representations of image concepts within LlaVA-Next, a popular open-source VLLM. We find a diverse set of ImageNet classes represented via linearly decodable features in the residual stream. We show that the features are causal by performing targeted edits on the model output. In order to increase the diversity of the studied linear features, we train multimodal Sparse Autoencoders (SAEs), creating a highly interpretable dictionary of text and image features. We find that although model representations across modalities are quite disjoint, they become increasingly shared in deeper layers. 

**Abstract (ZH)**: 多模态模型中的图像概念表示：以LlaVA-Next为例 

---
# HoliSafe: Holistic Safety Benchmarking and Modeling with Safety Meta Token for Vision-Language Model 

**Title (ZH)**: HoliSafe: 综合安全基准构建与安全元令牌建模在视觉语言模型中 

**Authors**: Youngwan Lee, Kangsan Kim, Kwanyong Park, Ilcahe Jung, Soojin Jang, Seanie Lee, Yong-Ju Lee, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04704)  

**Abstract**: Despite emerging efforts to enhance the safety of Vision-Language Models (VLMs), current approaches face two main shortcomings. 1) Existing safety-tuning datasets and benchmarks only partially consider how image-text interactions can yield harmful content, often overlooking contextually unsafe outcomes from seemingly benign pairs. This narrow coverage leaves VLMs vulnerable to jailbreak attacks in unseen configurations. 2) Prior methods rely primarily on data-centric tuning, with limited architectural innovations to intrinsically strengthen safety. We address these gaps by introducing a holistic safety dataset and benchmark, HoliSafe, that spans all five safe/unsafe image-text combinations, providing a more robust basis for both training and evaluation. We further propose SafeLLaVA, a novel VLM augmented with a learnable safety meta token and a dedicated safety head. The meta token encodes harmful visual cues during training, intrinsically guiding the language model toward safer responses, while the safety head offers interpretable harmfulness classification aligned with refusal rationales. Experiments show that SafeLLaVA, trained on HoliSafe, achieves state-of-the-art safety performance across multiple VLM benchmarks. Additionally, the HoliSafe benchmark itself reveals critical vulnerabilities in existing models. We hope that HoliSafe and SafeLLaVA will spur further research into robust and interpretable VLM safety, expanding future avenues for multimodal alignment. 

**Abstract (ZH)**: 尽管在增强视觉语言模型（VLMs）的安全性方面已经做出了努力，但当前的方法仍存在两个主要不足。1）现有的安全调优数据集和基准仅部分考虑了图像-文本互动可能产生的有害内容，并且往往忽视了看似无害的配对在上下文中的不安全结果，这种狭窄的覆盖面使得VLMs在未见配置中容易受到 Jailbreak 攻击。2）先前的方法主要依赖数据为中心的调优，缺乏从根本上增强安全性的架构创新。我们通过引入一个全面的安全数据集和基准HoliSafe来弥补这些不足，HoliSafe涵盖了所有五种安全/不安全的图像-文本组合，为训练和评估提供了更 robust 的基础。我们还提出了SafeLLaVA，这是一种配备了可学习的安全元标记和专用安全头部的新型VLM。在训练过程中，元标记编码有害的视觉提示，内在地引导语言模型产生更安全的响应，而安全头部提供了与拒绝理由一致的可解释的有害性分类。实验结果显示，SafeLLaVA在多个VLM基准上的安全性能达到了最先进的水平。此外，HoliSafe基准本身揭示了现有模型中的关键漏洞。我们希望HoliSafe和SafeLLaVA能促进对 robust 和可解释的VLM安全性的进一步研究，拓展未来多模态对齐的方向。 

---
# Explicit Density Approximation for Neural Implicit Samplers Using a Bernstein-Based Convex Divergence 

**Title (ZH)**: 使用Bernstein基凸散度的显式密度逼近神经隐式采样器 

**Authors**: José Manuel de Frutos, Manuel A. Vázquez, Pablo M. Olmos, Joaquín Míguez  

**Link**: [PDF](https://arxiv.org/pdf/2506.04700)  

**Abstract**: Rank-based statistical metrics, such as the invariant statistical loss (ISL), have recently emerged as robust and practically effective tools for training implicit generative models. In this work, we introduce dual-ISL, a novel likelihood-free objective for training implicit generative models that interchanges the roles of the target and model distributions in the ISL framework, yielding a convex optimization problem in the space of model densities. We prove that the resulting rank-based discrepancy $d_K$ is i) continuous under weak convergence and with respect to the $L^1$ norm, and ii) convex in its first argument-properties not shared by classical divergences such as KL or Wasserstein distances. Building on this, we develop a theoretical framework that interprets $d_K$ as an $L^2$-projection of the density ratio $q = p/\tilde p$ onto a Bernstein polynomial basis, from which we derive exact bounds on the truncation error, precise convergence rates, and a closed-form expression for the truncated density approximation. We further extend our analysis to the multivariate setting via random one-dimensional projections, defining a sliced dual-ISL divergence that retains both convexity and continuity. We empirically show that these theoretical advantages translate into practical ones. Specifically, across several benchmarks dual-ISL converges more rapidly, delivers markedly smoother and more stable training, and more effectively prevents mode collapse than classical ISL and other leading implicit generative methods-while also providing an explicit density approximation. 

**Abstract (ZH)**: 基于排名的统计指标，如不变统计损失（ISL），最近已被用作训练隐式生成模型的强大而实用的工具。在本文中，我们介绍了一种新颖的概率测度自由目标——对偶-ISL，该目标在ISL框架中交换目标分布和模型分布的角色，形成了模型密度空间中的凸优化问题。我们证明，由此产生的基于排名的差异$d_K$具有如下性质：i) 在弱收敛和$L^1$范数意义下连续；ii) 在其第一个参数上凸——这些性质不是KL或Wasserstein距离等经典发散所共有的。基于此，我们建立了一个理论框架，将$d_K$解释为密度比$q = p/\tilde p$在伯恩斯坦多项式基底上的$L^2$投影，从中我们得到了截断误差的确切界、精确收敛速率以及截断密度近似的闭式表达。进一步地，我们通过对各维随机一维投影的分析，将该框架扩展到多元设置，定义了一种切片对偶-ISL散度，该散度保留了凸性和连续性。我们在实验上展示了这些理论优势转化为实际优点的具体表现。具体而言，在多个基准测试中，对偶-ISL收敛速度更快，提供明显更平滑和更稳定的训练过程，并更有效地防止了模式缺失，同时提供了显式的密度近似，而这种性能优于经典ISL及其他领先的隐式生成方法。 

---
# On the Mechanism of Reasoning Pattern Selection in Reinforcement Learning for Language Models 

**Title (ZH)**: reinforcement learning for language models 中文标题可以翻译为：基于强化学习的语言模型推理模式选择机制 

**Authors**: Xingwu Chen, Tianle Li, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2506.04695)  

**Abstract**: Reinforcement learning (RL) has demonstrated remarkable success in enhancing model capabilities, including instruction-following, preference learning, and reasoning. Yet despite its empirical successes, the mechanisms by which RL improves reasoning abilities remain poorly understood. We present a systematic study of Reinforcement Learning with Verifiable Rewards (RLVR), showing that its primary benefit comes from optimizing the selection of existing reasoning patterns. Through extensive experiments, we demonstrate that RLVR-trained models preferentially adopt high-success-rate reasoning patterns while mostly maintaining stable performance on individual patterns. We further develop theoretical analyses on the convergence and training dynamics of RLVR based on a simplified question-reason-answer model. We study the gradient flow and show that RLVR can indeed find the solution that selects the reason pattern with the highest success rate. Besides, our theoretical results
reveal two distinct regimes regarding the convergence of RLVR training: (1) rapid convergence for models with relatively strong initial reasoning capabilities versus (2) slower optimization dynamics for weaker models. Furthermore, we show that the slower optimization for weaker models can be mitigated by applying the supervised fine-tuning (SFT) before RLVR, when using a feasibly high-quality SFT dataset. We validate the theoretical findings through extensive experiments. This work advances our theoretical understanding of RL's role in LLM fine-tuning and offers insights for further enhancing reasoning capabilities. 

**Abstract (ZH)**: 强化学习（RL）在增强模型能力方面取得了显著成功，包括指令遵循、偏好学习和推理。尽管在实证上取得了一定的成功，但RL提高推理能力的机制仍然不甚明了。我们对可验证奖励的强化学习（RLVR）进行了系统研究，表明其主要益处在于优化现有推理模式的选择。通过大量实验，我们证明了RLVR训练的模型倾向于选择成功率较高的推理模式，同时在个体模式上的性能保持稳定。我们进一步基于简化的问题-推理-答案模型对RLVR的收敛性和训练动力学进行了理论分析。我们研究了梯度流动，并证明RLVR确实能找到选择成功率最高的推理模式的解。此外，我们的理论结果揭示了RLVR训练收敛的两种不同模式：（1）对于初始推理能力强的模型，收敛速度快；（2）对于初始推理能力较弱的模型，则优化动态较慢。我们还展示了通过在RLVR之前应用监督微调（SFT），并使用高质量的SFT数据集，可以减轻较弱模型的优化缓慢问题。我们通过大量实验验证了这些理论发现。这项工作推进了我们对RL在大语言模型微调中作用的理论理解，并为进一步增强推理能力提供了见解。 

---
# Influence Functions for Edge Edits in Non-Convex Graph Neural Networks 

**Title (ZH)**: 非凸图神经网络中边编辑的影响函数 

**Authors**: Jaeseung Heo, Kyeongheung Yun, Seokwon Yoon, MoonJeong Park, Jungseul Ok, Dongwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.04694)  

**Abstract**: Understanding how individual edges influence the behavior of graph neural networks (GNNs) is essential for improving their interpretability and robustness. Graph influence functions have emerged as promising tools to efficiently estimate the effects of edge deletions without retraining. However, existing influence prediction methods rely on strict convexity assumptions, exclusively consider the influence of edge deletions while disregarding edge insertions, and fail to capture changes in message propagation caused by these modifications. In this work, we propose a proximal Bregman response function specifically tailored for GNNs, relaxing the convexity requirement and enabling accurate influence prediction for standard neural network architectures. Furthermore, our method explicitly accounts for message propagation effects and extends influence prediction to both edge deletions and insertions in a principled way. Experiments with real-world datasets demonstrate accurate influence predictions for different characteristics of GNNs. We further demonstrate that the influence function is versatile in applications such as graph rewiring and adversarial attacks. 

**Abstract (ZH)**: 理解单个边缘如何影响图神经网络（GNN）的行为对于提高其可解释性和鲁棒性至关重要。proximal Bregman 响应函数在本文中专门针对 GNN 被提出，放宽了凸性要求，使得可以准确预测标准神经网络架构的影响。此外，我们的方法明确考虑了消息传播效果，并以一种原则性的方式将影响预测扩展到边的删除和插入。实验结果表明，该方法能够准确预测不同特征下 GNN 的影响。我们进一步证明，影响函数在图重 wiring 和对抗攻击等领域具有灵活性。 

---
# Towards Better Generalization via Distributional Input Projection Network 

**Title (ZH)**: 通过分布输入投影网络实现更好的泛化能力 

**Authors**: Yifan Hao, Yanxin Lu, Xinwei Shen, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04690)  

**Abstract**: As overparameterized models become increasingly prevalent, training loss alone offers limited insight into generalization performance. While smoothness has been linked to improved generalization across various settings, directly enforcing smoothness in neural networks remains challenging. To address this, we introduce Distributional Input Projection Networks (DIPNet), a novel framework that projects inputs into learnable distributions at each layer. This distributional representation induces a smoother loss landscape with respect to the input, promoting better generalization. We provide theoretical analysis showing that DIPNet reduces both local smoothness measures and the Lipschitz constant of the network, contributing to improved generalization performance. Empirically, we validate DIPNet across a wide range of architectures and tasks, including Vision Transformers (ViTs), Large Language Models (LLMs), ResNet and MLPs. Our method consistently enhances test performance under standard settings, adversarial attacks, out-of-distribution inputs, and reasoning benchmarks. We demonstrate that the proposed input projection strategy can be seamlessly integrated into existing models, providing a general and effective approach for boosting generalization performance in modern deep learning. 

**Abstract (ZH)**: 过度参数化模型日益普及，训练损失 alone 未能充分揭示泛化性能。尽管平滑性在各种设置中与改进的泛化性能相关联，直接在神经网络中施加平滑性仍然具有挑战性。为此，我们提出了分布输入投影网络（DIPNet），这是一种新颖的框架，在每一层将输入投影到可学习的概率分布中。这种分布表示诱导了输入相关的更平滑的损失景观，促进了更好的泛化性能。我们提供了理论分析，证明 DIPNet 降低了局部平滑度度量并减少了网络的利普希茨常数，从而促进了泛化性能的提升。在实验中，我们在包括视觉变压器（ViTs）、大规模语言模型（LLMs）、ResNet 和 MLP 等多种架构和任务上验证了 DIPNet 的有效性。我们的方法在标准设置、对抗攻击、离分布输入和推理基准上均能持续提升测试性能。我们展示了所提出输入投影策略可以无缝集成到现有模型中，提供了一种通用且有效的提升现代深度学习泛化性能的方法。 

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
# Feature-Based Lie Group Transformer for Real-World Applications 

**Title (ZH)**: 基于特征的李群变换器在实际应用中的研究 

**Authors**: Takayuki Komatsu, Yoshiyuki Ohmura, Kayato Nishitsunoi, Yasuo Kuniyoshi  

**Link**: [PDF](https://arxiv.org/pdf/2506.04668)  

**Abstract**: The main goal of representation learning is to acquire meaningful representations from real-world sensory inputs without supervision. Representation learning explains some aspects of human development. Various neural network (NN) models have been proposed that acquire empirically good representations. However, the formulation of a good representation has not been established. We recently proposed a method for categorizing changes between a pair of sensory inputs. A unique feature of this approach is that transformations between two sensory inputs are learned to satisfy algebraic structural constraints. Conventional representation learning often assumes that disentangled independent feature axes is a good representation; however, we found that such a representation cannot account for conditional independence. To overcome this problem, we proposed a new method using group decomposition in Galois algebra theory. Although this method is promising for defining a more general representation, it assumes pixel-to-pixel translation without feature extraction, and can only process low-resolution images with no background, which prevents real-world application. In this study, we provide a simple method to apply our group decomposition theory to a more realistic scenario by combining feature extraction and object segmentation. We replace pixel translation with feature translation and formulate object segmentation as grouping features under the same transformation. We validated the proposed method on a practical dataset containing both real-world object and background. We believe that our model will lead to a better understanding of human development of object recognition in the real world. 

**Abstract (ZH)**: 无监督表征学习的主要目标是从现实世界的感知输入中获取有意义的表征。无监督表征学习解释了一些人类发展的方面。各种神经网络（NN）模型被提出，以获取实证较好的表征。然而，良好表征的公设尚未建立。我们最近提出了一种对感知输入对之间变化进行分类的方法。该方法的一个独特特征是，学习感知输入之间的变换以满足代数结构约束。传统的表征学习通常假设解耦独立特征轴是良好的表征；然而，我们发现这样的表征无法解释条件独立性。为解决此问题，我们提出了一种新的基于伽罗华代数理论中的群分解方法。尽管该方法有可能定义更一般的表征，但它假设像素到像素的翻译而不进行特征提取，并且只能处理没有背景的低分辨率图片，这限制了其实用性。在本研究中，我们提供了一种简单的方法，通过结合特征提取和对象分割将我们的群分解理论应用于更现实的场景。我们用特征翻译取代像素翻译，并将对象分割表述为在相同变换下的特征分组。我们通过包含现实世界对象和背景的实际数据集验证了所提出的方法。我们相信，我们的模型将有助于更好地理解人类在现实世界中对物体识别的发展。 

---
# Static Word Embeddings for Sentence Semantic Representation 

**Title (ZH)**: 静态词嵌入在句义表示中的应用 

**Authors**: Takashi Wada, Yuki Hirakawa, Ryotaro Shimizu, Takahiro Kawashima, Yuki Saito  

**Link**: [PDF](https://arxiv.org/pdf/2506.04624)  

**Abstract**: We propose new static word embeddings optimised for sentence semantic representation. We first extract word embeddings from a pre-trained Sentence Transformer, and improve them with sentence-level principal component analysis, followed by either knowledge distillation or contrastive learning. During inference, we represent sentences by simply averaging word embeddings, which requires little computational cost. We evaluate models on both monolingual and cross-lingual tasks and show that our model substantially outperforms existing static models on sentence semantic tasks, and even rivals a basic Sentence Transformer model (SimCSE) on some data sets. Lastly, we perform a variety of analyses and show that our method successfully removes word embedding components that are irrelevant to sentence semantics, and adjusts the vector norms based on the influence of words on sentence semantics. 

**Abstract (ZH)**: 我们提出了一种新的静态词嵌入方法，旨在优化句子语义表示。我们首先从预训练的Sentence Transformer中提取词嵌入，并通过句子级主成分分析改进它们，随后使用知识蒸馏或对比学习。在推理过程中，我们通过简单地对词嵌入求平均来表示句子，这需要较少的计算成本。我们在单语任务和跨语言任务上评估了模型，并表明我们的模型在句子语义任务上的表现显著优于现有的静态模型，在某些数据集上甚至与基本的Sentence Transformer模型（SimCSE）相当。最后，我们进行了多种分析，并展示了我们的方法成功地去除了与句子语义无关的词嵌入成分，并根据词对句子语义的影响调整了向量范数。 

---
# Scaling Laws for Robust Comparison of Open Foundation Language-Vision Models and Datasets 

**Title (ZH)**: 开放基础语言-视觉模型和数据集稳健比较的标度律 

**Authors**: Marianna Nezhurina, Tomer Porian, Giovanni Pucceti, Tommie Kerssies, Romain Beaumont, Mehdi Cherti, Jenia Jitsev  

**Link**: [PDF](https://arxiv.org/pdf/2506.04598)  

**Abstract**: In studies of transferable learning, scaling laws are obtained for various important foundation models to predict their properties and performance at larger scales. We show here how scaling law derivation can also be used for model and dataset comparison, allowing to decide which procedure is to be preferred for pre-training. For the first time, full scaling laws based on dense measurements across a wide span of model and samples seen scales are derived for two important language-vision learning procedures, CLIP and MaMMUT, that use either contrastive only or contrastive and captioning text generative loss. Ensuring sufficient prediction accuracy for held out points, we use derived scaling laws to compare both models, obtaining evidence for MaMMUT's stronger improvement with scale and better sample efficiency than standard CLIP. To strengthen validity of the comparison, we show scaling laws for various downstream tasks, classification, retrieval, and segmentation, and for different open datasets, DataComp, DFN and Re-LAION, observing consistently the same trends. We show that comparison can also be performed when deriving scaling laws with a constant learning rate schedule, reducing compute cost. Accurate derivation of scaling laws provides thus means to perform model and dataset comparison across scale spans, avoiding misleading conclusions based on measurements from single reference scales only, paving the road for systematic comparison and improvement of open foundation models and datasets for their creation. We release all the pre-trained models with their intermediate checkpoints, including openMaMMUT-L/14, which achieves $80.3\%$ zero-shot ImageNet-1k accuracy, trained on 12.8B samples from DataComp-1.4B. Code for reproducing experiments in the paper and raw experiments data can be found at this https URL. 

**Abstract (ZH)**: 在迁移学习研究中，通过建立各种重要基础模型的标度律来预测其在更大尺度下的属性和性能。我们展示了如何利用标度律推导方法来进行模型和数据集的比较，以便决定预训练时应优先采用哪种 procedure。首次基于广泛范围的模型和样本观测尺度，推导了两种重要的语言-视觉学习 procedure（CLIP 和 MaMMUT）的完整标度律，这两种 procedure 分别使用对比或对比和配图文本生成损失。通过确保对外预测点足够的准确性，我们使用推导出的标度律比较了两种模型，提供了 MaMMUT 在规模上表现出更强改进和更好样本效率的证据。为增强比较的可靠性，我们展示了在多种下游任务（分类、检索和分割）以及不同开源数据集（DataComp、DFN 和 Re-LAION）上推导出的标度律，观察到一致的趋势。我们展示了即使在使用恒定学习率调度推导标度律时也能进行比较，从而减少了计算成本。准确推导出的标度律提供了在不同尺度范围内比较模型和数据集的方法，避免了仅基于单一参考尺度测量得出错误结论的情况，为系统比较和改进开放基础模型和数据集奠定了道路。我们发布了所有预训练模型及其中间检查点，包括 openMaMMUT-L/14，该模型在 128亿个 DataComp-1.4B 样本上训练，实现了80.3% 的零样本 ImageNet-1k 准确率。论文中的实验代码和原始实验数据可在此处找到。 

---
# Intelligent Channel Allocation for IEEE 802.11be Multi-Link Operation: When MAB Meets LLM 

**Title (ZH)**: 基于MAB与LLM的IEEE 802.11be多链路操作智能信道分配 

**Authors**: Shumin Lian, Jingwen Tong, Jun Zhang, Liqun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04594)  

**Abstract**: WiFi networks have achieved remarkable success in enabling seamless communication and data exchange worldwide. The IEEE 802.11be standard, known as WiFi 7, introduces Multi-Link Operation (MLO), a groundbreaking feature that enables devices to establish multiple simultaneous connections across different bands and channels. While MLO promises substantial improvements in network throughput and latency reduction, it presents significant challenges in channel allocation, particularly in dense network environments. Current research has predominantly focused on performance analysis and throughput optimization within static WiFi 7 network configurations. In contrast, this paper addresses the dynamic channel allocation problem in dense WiFi 7 networks with MLO capabilities. We formulate this challenge as a combinatorial optimization problem, leveraging a novel network performance analysis mechanism. Given the inherent lack of prior network information, we model the problem within a Multi-Armed Bandit (MAB) framework to enable online learning of optimal channel allocations. Our proposed Best-Arm Identification-enabled Monte Carlo Tree Search (BAI-MCTS) algorithm includes rigorous theoretical analysis, providing upper bounds for both sample complexity and error probability. To further reduce sample complexity and enhance generalizability across diverse network scenarios, we put forth LLM-BAI-MCTS, an intelligent algorithm for the dynamic channel allocation problem by integrating the Large Language Model (LLM) into the BAI-MCTS algorithm. Numerical results demonstrate that the BAI-MCTS algorithm achieves a convergence rate approximately $50.44\%$ faster than the state-of-the-art algorithms when reaching $98\%$ of the optimal value. Notably, the convergence rate of the LLM-BAI-MCTS algorithm increases by over $63.32\%$ in dense networks. 

**Abstract (ZH)**: WiFi网络已在全球范围内实现了无缝通信和数据交换的显著成功。IEEE 802.11be标准，即WiFi 7，引入了多链路操作(Multi-Link Operation, MLO)功能，使设备能够在不同频段和信道上建立多个同时连接。虽然MLO有望大幅提高网络吞吐量并减少延迟，但它在密集网络环境中提出了显著的信道分配挑战。当前研究主要集中在静态WiFi 7网络配置下的性能分析和吞吐量优化。相比之下，本文探讨了具有MLO能力的密集WiFi 7网络中的动态信道分配问题。我们将这一挑战形式化为组合优化问题，并利用新的网络性能分析机制。鉴于先前网络信息的固有缺乏，我们将问题建模为多臂 bandit (MAB)框架，以实现在线学习最优信道分配。我们提出的Best-Arm Identification-enabled Monte Carlo Tree Search (BAI-MCTS)算法包括严谨的理论分析，提供了样本复杂性和错误概率的上界。为了进一步减少样本复杂性和增强适用于各种网络场景的泛化能力，我们提出了一种将大型语言模型（LLM）集成到BAI-MCTS算法中的智能动态信道分配算法LLM-BAI-MCTS。数值结果表明，当达到98%最优值时，BAI-MCTS算法的收敛速度比最先进的算法快约50.44%。值得注意的是，在密集网络中，LLM-BAI-MCTS算法的收敛速度提高了超过63.32%。 

---
# Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification 

**Title (ZH)**: Safe: 通过回顾性的步骤感知形式验证增强大型语言模型的数学推理能力 

**Authors**: Chengwu Liu, Ye Yuan, Yichun Yin, Yan Xu, Xin Xu, Zaoyu Chen, Yasheng Wang, Lifeng Shang, Qun Liu, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04592)  

**Abstract**: Chain-of-Thought (CoT) prompting has become the de facto method to elicit reasoning capabilities from large language models (LLMs). However, to mitigate hallucinations in CoT that are notoriously difficult to detect, current methods such as process reward models (PRMs) or self-consistency operate as opaque boxes and do not provide checkable evidence for their judgments, possibly limiting their effectiveness. To address this issue, we draw inspiration from the idea that "the gold standard for supporting a mathematical claim is to provide a proof". We propose a retrospective, step-aware formal verification framework $Safe$. Rather than assigning arbitrary scores, we strive to articulate mathematical claims in formal mathematical language Lean 4 at each reasoning step and provide formal proofs to identify hallucinations. We evaluate our framework $Safe$ across multiple language models and various mathematical datasets, demonstrating a significant performance improvement while offering interpretable and verifiable evidence. We also propose $FormalStep$ as a benchmark for step correctness theorem proving with $30,809$ formal statements. To the best of our knowledge, our work represents the first endeavor to utilize formal mathematical language Lean 4 for verifying natural language content generated by LLMs, aligning with the reason why formal mathematical languages were created in the first place: to provide a robust foundation for hallucination-prone human-written proofs. 

**Abstract (ZH)**: Safe：使用形式化数学语言验证大型语言模型生成内容的回顾性步骤感知形式化验证框架 

---
# SUCEA: Reasoning-Intensive Retrieval for Adversarial Fact-checking through Claim Decomposition and Editing 

**Title (ZH)**: SUCEA：通过论断分解和编辑进行推理密集型对抗事实核查检索 

**Authors**: Hongjun Liu, Yilun Zhao, Arman Cohan, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.04583)  

**Abstract**: Automatic fact-checking has recently received more attention as a means of combating misinformation. Despite significant advancements, fact-checking systems based on retrieval-augmented language models still struggle to tackle adversarial claims, which are intentionally designed by humans to challenge fact-checking systems. To address these challenges, we propose a training-free method designed to rephrase the original claim, making it easier to locate supporting evidence. Our modular framework, SUCEA, decomposes the task into three steps: 1) Claim Segmentation and Decontextualization that segments adversarial claims into independent sub-claims; 2) Iterative Evidence Retrieval and Claim Editing that iteratively retrieves evidence and edits the subclaim based on the retrieved evidence; 3) Evidence Aggregation and Label Prediction that aggregates all retrieved evidence and predicts the entailment label. Experiments on two challenging fact-checking datasets demonstrate that our framework significantly improves on both retrieval and entailment label accuracy, outperforming four strong claim-decomposition-based baselines. 

**Abstract (ZH)**: 自动事实核查近年来因其对抗 misinformation 而受到更多关注。尽管取得了显著进展，基于检索增强语言模型的事实核查系统仍然难以应对由人类故意设计的对抗性声明。为应对这些挑战，我们提出了一种无需训练的方法，旨在重新表述原始声明，使其更容易找到支持性证据。我们的模块化框架 SUCEA 将任务分解为三个步骤：1) 声明分段与脱嵌以将对抗性声明拆分为独立子声明；2) 迭代证据检索与声明编辑，根据检索到的证据迭代检索证据并编辑子声明；3) 证据聚合与标签预测，整合所有检索到的证据并预测涵摄标签。实验结果表明，我们的框架在检索和涵摄标签准确性上均显著优于四种强大的声明分解基线。 

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
# BESA: Boosting Encoder Stealing Attack with Perturbation Recovery 

**Title (ZH)**: BESA: 基于扰动恢复的编码器窃取攻击增强technique 

**Authors**: Xuhao Ren, Haotian Liang, Yajie Wang, Chuan Zhang, Zehui Xiong, Liehuang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04556)  

**Abstract**: To boost the encoder stealing attack under the perturbation-based defense that hinders the attack performance, we propose a boosting encoder stealing attack with perturbation recovery named BESA. It aims to overcome perturbation-based defenses. The core of BESA consists of two modules: perturbation detection and perturbation recovery, which can be combined with canonical encoder stealing attacks. The perturbation detection module utilizes the feature vectors obtained from the target encoder to infer the defense mechanism employed by the service provider. Once the defense mechanism is detected, the perturbation recovery module leverages the well-designed generative model to restore a clean feature vector from the perturbed one. Through extensive evaluations based on various datasets, we demonstrate that BESA significantly enhances the surrogate encoder accuracy of existing encoder stealing attacks by up to 24.63\% when facing state-of-the-art defenses and combinations of multiple defenses. 

**Abstract (ZH)**: 基于扰动恢复的增强编码器窃取攻击：BESA 

---
# hdl2v: A Code Translation Dataset for Enhanced LLM Verilog Generation 

**Title (ZH)**: hdl2v：一种代码转换数据集，用于增强LLM Verilog生成 

**Authors**: Charles Hong, Brendan Roberts, Huijae An, Alex Um, Advay Ratan, Yakun Sophia Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.04544)  

**Abstract**: Large language models (LLMs) are playing an increasingly large role in domains such as code generation, including hardware code generation, where Verilog is the key language. However, the amount of publicly available Verilog code pales in comparison to the amount of code available for software languages like Python. In this work, we present hdl2v ("HDL-to-Verilog"), a dataset which seeks to increase the amount of available human-written Verilog data by translating or compiling three other hardware description languages - VHDL, Chisel, and PyMTL3 - to Verilog. Furthermore, we demonstrate the value of hdl2v in enhancing LLM Verilog generation by improving performance of a 32 billion-parameter open-weight model by up to 23% (pass@10) in VerilogEvalV2, without utilizing any data augmentation or knowledge distillation from larger models. We also show hdl2v's ability to boost the performance of a data augmentation-based fine-tuning approach by 63%. Finally, we characterize and analyze our dataset to better understand which characteristics of HDL-to-Verilog datasets can be expanded upon in future work for even better performance. 

**Abstract (ZH)**: 大型语言模型（LLMs）在代码生成领域的作用越来越重要，包括硬件代码生成，其中Verilog是关键语言。然而，公开可用的Verilog代码数量远不及软件语言如Python的代码数量。在本工作中，我们介绍了hdl2v（HDL-to-Verilog）数据集，旨在通过将VHDL、Chisel和PyMTL3等三种硬件描述语言转换为Verilog，增加可用的由人类编写的Verilog数据量。此外，我们展示了hdl2v在提高LLM生成Verilog代码的性能方面的价值，通过在VerilogEvalV2中的表现，提高了具有320亿参数的开源权重模型的性能高达23%（pass@10），而无需使用任何数据增强或来自更大模型的知识蒸馏。我们还展示了hdl2v在基于数据增强的微调方法中的性能提升能力，提高了63%。最后，我们对数据集进行了表征和分析，以便更好地理解HDL-to-Verilog数据集哪些特性可以在未来工作中扩展，以获得更优的性能。 

---
# NOBLE -- Neural Operator with Biologically-informed Latent Embeddings to Capture Experimental Variability in Biological Neuron Models 

**Title (ZH)**: NOBLE -- 基于生物学信息潜在嵌入的神经运算子，用于捕捉生物神经元模型中的实验变异性 

**Authors**: Luca Ghafourpour, Valentin Duruisseaux, Bahareh Tolooshams, Philip H. Wong, Costas A. Anastassiou, Anima Anandkumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.04536)  

**Abstract**: Characterizing the diverse computational properties of human neurons via multimodal electrophysiological, transcriptomic, and morphological data provides the foundation for constructing and validating bio-realistic neuron models that can advance our understanding of fundamental mechanisms underlying brain function. However, current modeling approaches remain constrained by the limited availability and intrinsic variability of experimental neuronal data. To capture variability, ensembles of deterministic models are often used, but are difficult to scale as model generation requires repeating computationally expensive optimization for each neuron. While deep learning is becoming increasingly relevant in this space, it fails to capture the full biophysical complexity of neurons, their nonlinear voltage dynamics, and variability. To address these shortcomings, we introduce NOBLE, a neural operator framework that learns a mapping from a continuous frequency-modulated embedding of interpretable neuron features to the somatic voltage response induced by current injection. Trained on data generated from biophysically realistic neuron models, NOBLE predicts distributions of neural dynamics accounting for the intrinsic experimental variability. Unlike conventional bio-realistic neuron models, interpolating within the embedding space offers models whose dynamics are consistent with experimentally observed responses. NOBLE is the first scaled-up deep learning framework validated on real experimental data, enabling efficient generation of synthetic neurons that exhibit trial-to-trial variability and achieve a $4200\times$ speedup over numerical solvers. To this end, NOBLE captures fundamental neural properties, opening the door to a better understanding of cellular composition and computations, neuromorphic architectures, large-scale brain circuits, and general neuroAI applications. 

**Abstract (ZH)**: 通过多模态电生理、转录组和形态学数据表征人类神经元的多样计算属性为构建和验证生物现实神经元模型奠定了基础，这些模型能够推进我们对脑功能基本机制的理解。然而，当前的建模方法仍然受到实验神经元数据有限性和内在变异性限制。为了捕捉这种变异性，通常使用一组确定性模型，但由于每次生成模型都需要进行计算昂贵的优化，这难以扩大规模。虽然深度学习在这个领域变得越来越相关，但未能捕捉到神经元的完整生物物理复杂性、非线性电压动力学及其变异性。为了解决这些问题，我们引入了NOBLE，一种神经操作符框架，从可解释神经元特征的连续频率调制嵌入到由电流注入引起的胞体电压响应之间的映射进行学习。该模型在生物物理现实神经元模型生成的数据上进行训练，从而预测神经动力学分布，考虑到固有的实验变异性。与传统的生物现实神经元模型不同，在嵌入空间中的插值提供了动力学与实验观察响应一致的模型。NOBLE是首个在真实实验数据上进行验证的扩展规模的深度学习框架，能够高效生成表现出试次间变性的合成神经元，并实现比数值求解器快4200倍的速度提升。通过这种方式，NOBLE捕获了基本的神经属性，为更好地理解细胞组成和计算、类脑架构、大规模脑回路以及通用神经AI应用打开了大门。 

---
# Is It JUST Semantics? A Case Study of Discourse Particle Understanding in LLMs 

**Title (ZH)**: 它是仅仅语义问题吗？大规模语言模型中话语粒子理解的案例研究 

**Authors**: William Sheffield, Kanishka Misra, Valentina Pyatkin, Ashwini Deo, Kyle Mahowald, Junyi Jessy Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.04534)  

**Abstract**: Discourse particles are crucial elements that subtly shape the meaning of text. These words, often polyfunctional, give rise to nuanced and often quite disparate semantic/discourse effects, as exemplified by the diverse uses of the particle "just" (e.g., exclusive, temporal, emphatic). This work investigates the capacity of LLMs to distinguish the fine-grained senses of English "just", a well-studied example in formal semantics, using data meticulously created and labeled by expert linguists. Our findings reveal that while LLMs exhibit some ability to differentiate between broader categories, they struggle to fully capture more subtle nuances, highlighting a gap in their understanding of discourse particles. 

**Abstract (ZH)**: 话语 particles 是微妙塑造文本意义的关键元素。这些多功能的词语产生了细微且往往相当不同的语义/话语效果，以“just”这一词语的多种用途为例（例如， exclusivity、时间性、强调）。本研究探讨了大语言模型（LLMs）区分英语“just”的细微含义的能力，这些含义是形式语义学中一个研究充分的例子，利用了由专家语言学家精心创建和标注的数据。我们的发现表明，虽然大语言模型在区分更广泛的类别时表现出一些能力，但在捕捉更微妙的差异方面仍存在困难，突显了它们在理解话语 particles 方面的不足。 

---
# The Latent Space Hypothesis: Toward Universal Medical Representation Learning 

**Title (ZH)**: 潜在空间假设：通往通用医学表示学习的途径 

**Authors**: Salil Patel  

**Link**: [PDF](https://arxiv.org/pdf/2506.04515)  

**Abstract**: Medical data range from genomic sequences and retinal photographs to structured laboratory results and unstructured clinical narratives. Although these modalities appear disparate, many encode convergent information about a single underlying physiological state. The Latent Space Hypothesis frames each observation as a projection of a unified, hierarchically organized manifold -- much like shadows cast by the same three-dimensional object. Within this learned geometric representation, an individual's health status occupies a point, disease progression traces a trajectory, and therapeutic intervention corresponds to a directed vector. Interpreting heterogeneous evidence in a shared space provides a principled way to re-examine eponymous conditions -- such as Parkinson's or Crohn's -- that often mask multiple pathophysiological entities and involve broader anatomical domains than once believed. By revealing sub-trajectories and patient-specific directions of change, the framework supplies a quantitative rationale for personalised diagnosis, longitudinal monitoring, and tailored treatment, moving clinical practice away from grouping by potentially misleading labels toward navigation of each person's unique trajectory. Challenges remain -- bias amplification, data scarcity for rare disorders, privacy, and the correlation-causation divide -- but scale-aware encoders, continual learning on longitudinal data streams, and perturbation-based validation offer plausible paths forward. 

**Abstract (ZH)**: 医学数据包括基因组序列、视网膜照片、结构化的实验室结果以及非结构化的临床病历。尽管这些模态看似不同，但它们常常包含了关于单一生理状态的共通信息。潜在空间假说将每项观察视为一个统一、分层组织流形的投影——就像同一三个维度物体投下的阴影。在这种学习到的几何表示中，个体的健康状况占据一个点，疾病进展绘制出一条轨迹，而治疗干预对应于一个定向向量。在共享空间中解释异质性证据为重新审视帕金森病或克罗恩病等同名病症提供了一种合乎逻辑的方法，这些病症往往掩盖了多种病理生理实体，并涉及比过去所认为的更广泛的解剖区域。通过揭示子轨迹和患者特异性的变化方向，该框架为个性化诊断、纵向监测和个性化治疗提供了定量依据，将临床实践从基于可能误导性的标签分类向导航每个人独特轨迹转变。仍存在挑战——如偏差放大、稀有疾病的缺乏数据、隐私问题以及因果推断的差异——但意识尺度编码器、基于纵向数据流的持续学习以及基于扰动的验证提供了可行的前进路径。 

---
# BEAR: BGP Event Analysis and Reporting 

**Title (ZH)**: BEAR: BGP事件分析与报告 

**Authors**: Hanqing Li, Melania Fedeli, Vinay Kolar, Diego Klabjan  

**Link**: [PDF](https://arxiv.org/pdf/2506.04514)  

**Abstract**: The Internet comprises of interconnected, independently managed Autonomous Systems (AS) that rely on the Border Gateway Protocol (BGP) for inter-domain routing. BGP anomalies--such as route leaks and hijacks--can divert traffic through unauthorized or inefficient paths, jeopardizing network reliability and security. Although existing rule-based and machine learning methods can detect these anomalies using structured metrics, they still require experts with in-depth BGP knowledge of, for example, AS relationships and historical incidents, to interpret events and propose remediation. In this paper, we introduce BEAR (BGP Event Analysis and Reporting), a novel framework that leverages large language models (LLMs) to automatically generate comprehensive reports explaining detected BGP anomaly events. BEAR employs a multi-step reasoning process that translates tabular BGP data into detailed textual narratives, enhancing interpretability and analytical precision. To address the limited availability of publicly documented BGP anomalies, we also present a synthetic data generation framework powered by LLMs. Evaluations on both real and synthetic datasets demonstrate that BEAR achieves 100% accuracy, outperforming Chain-of-Thought and in-context learning baselines. This work pioneers an automated approach for explaining BGP anomaly events, offering valuable operational insights for network management. 

**Abstract (ZH)**: BGP事件分析与报告：利用大规模语言模型自动生成全面报告 

---
# Comparative performance of ensemble models in predicting dental provider types: insights from fee-for-service data 

**Title (ZH)**: 基于费用报销数据的集成模型在预测牙科提供者类型方面的表现比较：见解 

**Authors**: Mohammad Subhi Al-Batah, Muhyeeddin Alqaraleh, Mowafaq Salem Alzboon, Abdullah Alourani  

**Link**: [PDF](https://arxiv.org/pdf/2506.04479)  

**Abstract**: Dental provider classification plays a crucial role in optimizing healthcare resource allocation and policy planning. Effective categorization of providers, such as standard rendering providers and safety net clinic (SNC) providers, enhances service delivery to underserved populations. This study aimed to evaluate the performance of machine learning models in classifying dental providers using a 2018 dataset. A dataset of 24,300 instances with 20 features was analyzed, including beneficiary and service counts across fee-for-service (FFS), Geographic Managed Care, and Pre-Paid Health Plans. Providers were categorized by delivery system and patient age groups (0-20 and 21+). Despite 38.1% missing data, multiple machine learning algorithms were tested, including k-Nearest Neighbors (kNN), Decision Trees, Support Vector Machines (SVM), Stochastic Gradient Descent (SGD), Random Forest, Neural Networks, and Gradient Boosting. A 10-fold cross-validation approach was applied, and models were evaluated using AUC, classification accuracy (CA), F1-score, precision, and recall. Neural Networks achieved the highest AUC (0.975) and CA (94.1%), followed by Random Forest (AUC: 0.948, CA: 93.0%). These models effectively handled imbalanced data and complex feature interactions, outperforming traditional classifiers like Logistic Regression and SVM. Advanced machine learning techniques, particularly ensemble and deep learning models, significantly enhance dental workforce classification. Their integration into healthcare analytics can improve provider identification and resource distribution, benefiting underserved populations. 

**Abstract (ZH)**: 牙科服务提供者分类在优化健康资源分配和政策规划中扮演着重要角色。有效的服务提供者分类，如标准服务提供者和安全网诊所（SNC）提供者，可以增强对未服务群体的服务交付。本研究旨在评估机器学习模型在使用2018年数据集分类牙科提供者方面的性能。分析了一个包含24,300个实例和20个特征的数据集，其中包括按服务方式（按服务付费、地理管理护理和预付费健康计划）和服务对象（受益人和患者）的治疗次数和服务次数。提供者根据服务体系和患者年龄组（0-20岁和21+岁）进行分类。尽管有38.1%的数据缺失，但仍测试了多种机器学习算法，包括k-最近邻（kNN）、决策树、支持向量机（SVM）、随机梯度下降（SGD）、随机森林、神经网络和梯度提升。应用了10折交叉验证方法，并使用AUC、分类准确率（CA）、F1分数、精确率和召回率评估模型。神经网络实现了最高AUC（0.975）和CA（94.1%），其次是随机森林（AUC：0.948，CA：93.0%）。这些模型有效地处理了不平衡数据和复杂特征相互作用，优于传统的分类器如逻辑回归和SVM。先进的机器学习技术，尤其是集成和深度学习模型，显著提升了牙科劳动力分类的性能。将这些技术集成到健康医疗分析中可以提高服务提供者识别和资源配置，从而惠及未服务群体。 

---
# Classifying Dental Care Providers Through Machine Learning with Features Ranking 

**Title (ZH)**: 通过特征排序进行机器学习分类的牙科护理提供者分类 

**Authors**: Mohammad Subhi Al-Batah, Mowafaq Salem Alzboon, Muhyeeddin Alqaraleh, Mohammed Hasan Abu-Arqoub, Rashiq Rafiq Marie  

**Link**: [PDF](https://arxiv.org/pdf/2506.04474)  

**Abstract**: This study investigates the application of machine learning (ML) models for classifying dental providers into two categories - standard rendering providers and safety net clinic (SNC) providers - using a 2018 dataset of 24,300 instances with 20 features. The dataset, characterized by high missing values (38.1%), includes service counts (preventive, treatment, exams), delivery systems (FFS, managed care), and beneficiary demographics. Feature ranking methods such as information gain, Gini index, and ANOVA were employed to identify critical predictors, revealing treatment-related metrics (TXMT_USER_CNT, TXMT_SVC_CNT) as top-ranked features. Twelve ML models, including k-Nearest Neighbors (kNN), Decision Trees, Support Vector Machines (SVM), Stochastic Gradient Descent (SGD), Random Forest, Neural Networks, and Gradient Boosting, were evaluated using 10-fold cross-validation. Classification accuracy was tested across incremental feature subsets derived from rankings. The Neural Network achieved the highest accuracy (94.1%) using all 20 features, followed by Gradient Boosting (93.2%) and Random Forest (93.0%). Models showed improved performance as more features were incorporated, with SGD and ensemble methods demonstrating robustness to missing data. Feature ranking highlighted the dominance of treatment service counts and annotation codes in distinguishing provider types, while demographic variables (AGE_GROUP, CALENDAR_YEAR) had minimal impact. The study underscores the importance of feature selection in enhancing model efficiency and accuracy, particularly in imbalanced healthcare datasets. These findings advocate for integrating feature-ranking techniques with advanced ML algorithms to optimize dental provider classification, enabling targeted resource allocation for underserved populations. 

**Abstract (ZH)**: 本研究 investigate 了使用机器学习（ML）模型对牙科提供者进行分类的应用，将其分为标准渲染提供者和安全网诊所（SNC）提供者两类，使用的是2018年的数据集，包含24,300个实例和20个特征。数据集具有较高的缺失值（38.1%），包括服务计数（预防性、治疗、检查）、交付系统（FFS、管理护理）和受益人人口统计学数据。通过信息增益、基尼指数和ANOVA等特征排名方法来识别关键预测因子，结果显示与治疗相关的度量（TXMT_USER_CNT、TXMT_SVC_CNT）是顶级特征。研究评估了包括k-近邻（kNN）、决策树、支持向量机（SVM）、随机梯度下降（SGD）、随机森林、神经网络和梯度提升在内的12种ML模型，采用了10折交叉验证。分类准确性基于排名衍生的逐步特征子集进行了测试。神经网络在使用所有20个特征时达到了最高的准确性（94.1%），其次是梯度提升（93.2%）和随机森林（93.0%）。随着特征数量的增加，模型性能有所提升，随机梯度下降和集成方法对缺失数据具有鲁棒性。特征排名突出了治疗服务计数和注释代码在区分提供者类型方面的主导地位，而人口统计变量（AGE_GROUP、CALENDAR_YEAR）的影响较小。本研究强调了在不平衡的医疗保健数据集增强模型效率和准确性的特征选择的重要性。这些发现提倡将特征排名技术与先进的ML算法相结合，以优化牙科提供者分类，从而为未充分服务的人群分配目标化的资源。 

---
# Diffusion Transformer-based Universal Dose Denoising for Pencil Beam Scanning Proton Therapy 

**Title (ZH)**: 基于扩散变换器的通用剂量去噪方法用于-pencil束扫描质子疗法 

**Authors**: Yuzhen Ding, Jason Holmes, Hongying Feng, Martin Bues, Lisa A. McGee, Jean-Claude M. Rwigema, Nathan Y. Yu, Terence S. Sio, Sameer R. Keole, William W. Wong, Steven E. Schild, Jonathan B. Ashman, Sujay A. Vora, Daniel J. Ma, Samir H. Patel, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04467)  

**Abstract**: Purpose: Intensity-modulated proton therapy (IMPT) offers precise tumor coverage while sparing organs at risk (OARs) in head and neck (H&N) cancer. However, its sensitivity to anatomical changes requires frequent adaptation through online adaptive radiation therapy (oART), which depends on fast, accurate dose calculation via Monte Carlo (MC) simulations. Reducing particle count accelerates MC but degrades accuracy. To address this, denoising low-statistics MC dose maps is proposed to enable fast, high-quality dose generation.
Methods: We developed a diffusion transformer-based denoising framework. IMPT plans and 3D CT images from 80 H&N patients were used to generate noisy and high-statistics dose maps using MCsquare (1 min and 10 min per plan, respectively). Data were standardized into uniform chunks with zero-padding, normalized, and transformed into quasi-Gaussian distributions. Testing was done on 10 H&N, 10 lung, 10 breast, and 10 prostate cancer cases, preprocessed identically. The model was trained with noisy dose maps and CT images as input and high-statistics dose maps as ground truth, using a combined loss of mean square error (MSE), residual loss, and regional MAE (focusing on top/bottom 10% dose voxels). Performance was assessed via MAE, 3D Gamma passing rate, and DVH indices.
Results: The model achieved MAEs of 0.195 (H&N), 0.120 (lung), 0.172 (breast), and 0.376 Gy[RBE] (prostate). 3D Gamma passing rates exceeded 92% (3%/2mm) across all sites. DVH indices for clinical target volumes (CTVs) and OARs closely matched the ground truth.
Conclusion: A diffusion transformer-based denoising framework was developed and, though trained only on H&N data, generalizes well across multiple disease sites. 

**Abstract (ZH)**: 目的：强度调制质子疗法（IMPT）在头颈癌治疗中能够实现精准的肿瘤覆盖同时减少对危及器官（OARs）的照射，但其对解剖变化的敏感性要求通过在线适应性放疗（oART）频繁调整，这依赖于快速、准确的Monte Carlo（MC）模拟剂量计算。降低粒子数目可以加速MC模拟，但会降低准确性。为此，提出通过去噪低统计量MC剂量图来实现快速高质量剂量生成。

方法：我们开发了一种基于扩散变换器的去噪框架。使用MCsquare生成了80例头颈癌患者的IMPT计划和3D CT图像，并分别用1分钟和10分钟计算出噪声剂量图和高统计量剂量图。数据被标准化为均匀的块，并进行了零填充处理和归一化，转化为准高斯分布。测试在10例头颈癌、10例肺癌、10例乳腺癌和10例前列腺癌病例上进行，所有病例预先进行了相同的处理。模型使用有噪声的剂量图和CT图像作为输入，高统计量剂量图作为真实值，使用均方误差（MSE）、残差损失和区域MAE（集中在剂量最高/最低的10%体素）的组合损失进行训练。性能通过使用MAE、3D剂量率通过率和DVH指数进行了评估。

结果：模型在头颈癌、肺癌、乳腺癌和前列腺癌中的MAE分别为0.195 Gy[RBE]、0.120 Gy[RBE]、0.172 Gy[RBE]和0.376 Gy[RBE]。3D剂量率通过率在所有部位均超过92%（3%/2mm）。CTV和OARs的DVH指数与真实值高度匹配。

结论：开发了一种基于扩散变换器的去噪框架，虽然仅通过头颈癌数据进行训练，但在多个疾病部位具有良好的泛化能力。 

---
# Behavioural vs. Representational Systematicity in End-to-End Models: An Opinionated Survey 

**Title (ZH)**: 端到端模型中行为系统性和表示系统性的对比：一份观点性的综述 

**Authors**: Ivan Vegner, Sydelle de Souza, Valentin Forch, Martha Lewis, Leonidas A.A. Doumas  

**Link**: [PDF](https://arxiv.org/pdf/2506.04461)  

**Abstract**: A core aspect of compositionality, systematicity is a desirable property in ML models as it enables strong generalization to novel contexts. This has led to numerous studies proposing benchmarks to assess systematic generalization, as well as models and training regimes designed to enhance it. Many of these efforts are framed as addressing the challenge posed by Fodor and Pylyshyn. However, while they argue for systematicity of representations, existing benchmarks and models primarily focus on the systematicity of behaviour. We emphasize the crucial nature of this distinction. Furthermore, building on Hadley's (1994) taxonomy of systematic generalization, we analyze the extent to which behavioural systematicity is tested by key benchmarks in the literature across language and vision. Finally, we highlight ways of assessing systematicity of representations in ML models as practiced in the field of mechanistic interpretability. 

**Abstract (ZH)**: 组成性中的系统性是一个 desirable property 在 ML 模型中，因为它能够使模型在新情境中表现出强大的泛化能力。这导致了大量研究提出评估系统性泛化的基准，并设计模型和训练制度以增强这种能力。尽管这些努力多被框定为解决 Fodor 和 Pylyshyn 提出的挑战，但它们在论代理表示的系统性的同时，现有的基准和模型主要集中在行为的系统性上。我们强调了这种区别的关键性。此外，基于 Hadley (1994) 对系统性泛化的分类，我们分析了文献中语言和视觉领域关键基准到何种程度测试了行为的系统性。最后，我们指出了在机械解释领域评估模型表示系统性的方法。 

---
# An Expansion-Based Approach for Quantified Integer Programming 

**Title (ZH)**: 基于扩展的方法在量化整数规划中的应用 

**Authors**: Michael Hartisch, Leroy Chew  

**Link**: [PDF](https://arxiv.org/pdf/2506.04452)  

**Abstract**: Quantified Integer Programming (QIP) bridges multiple domains by extending Quantified Boolean Formulas (QBF) to incorporate general integer variables and linear constraints while also generalizing Integer Programming through variable quantification. As a special case of Quantified Constraint Satisfaction Problems (QCSP), QIP provides a versatile framework for addressing complex decision-making scenarios. Additionally, the inclusion of a linear objective function enables QIP to effectively model multistage robust discrete linear optimization problems, making it a powerful tool for tackling uncertainty in optimization.
While two primary solution paradigms exist for QBF -- search-based and expansion-based approaches -- only search-based methods have been explored for QIP and QCSP. We introduce an expansion-based approach for QIP using Counterexample-Guided Abstraction Refinement (CEGAR), adapting techniques from QBF. We extend this methodology to tackle multistage robust discrete optimization problems with linear constraints and further embed it in an optimization framework, enhancing its applicability. Our experimental results highlight the advantages of this approach, demonstrating superior performance over existing search-based solvers for QIP in specific instances. Furthermore, the ability to model problems using linear constraints enables notable performance gains over state-of-the-art expansion-based solvers for QBF. 

**Abstract (ZH)**: 量化整数规划（QIP）通过将量化的布尔公式（QBF）扩展到包含一般整数变量和线性约束，同时推广整数规划并通过变量量化，架起了多个领域的桥梁。作为量化约束满足问题（QCSP）的特殊案例，QIP为解决复杂的决策场景提供了灵活的框架。此外，线性目标函数的引入使QIP能够有效地模拟具有线性约束的多阶段鲁棒离散线性优化问题，使其成为处理优化中不确定性的一种强大工具。虽然存在两种主要的QBF求解范式——搜索基和扩展基方法——仅搜索基方法被探索用于QIP和QCSP。我们提出了一种基于扩展的方法来解决QIP，使用基于反例指引的抽象细化（CEGAR）方法，借鉴了QBF的技术。我们进一步将此方法扩展以解决具有线性约束的多阶段鲁棒离散优化问题，并将其嵌入优化框架中，增强了其适用性。实验结果表明了该方法的优势，在特定实例中优于现有的基于搜索的QIP求解器。此外，使用线性约束建模问题的能力还使其在QBF最先进的基于扩展的求解器中表现出显著的性能优势。 

---
# Learning to Diagnose Privately: DP-Powered LLMs for Radiology Report Classification 

**Title (ZH)**: 学习进行私密诊断：DP-Powered LLMs在放射学报告分类中的应用 

**Authors**: Payel Bhattacharjee, Fengwei Tian, Ravi Tandon, Joseph Lo, Heidi Hanson, Geoffrey Rubin, Nirav Merchant, John Gounley  

**Link**: [PDF](https://arxiv.org/pdf/2506.04450)  

**Abstract**: Purpose: This study proposes a framework for fine-tuning large language models (LLMs) with differential privacy (DP) to perform multi-abnormality classification on radiology report text. By injecting calibrated noise during fine-tuning, the framework seeks to mitigate the privacy risks associated with sensitive patient data and protect against data leakage while maintaining classification performance. Materials and Methods: We used 50,232 radiology reports from the publicly available MIMIC-CXR chest radiography and CT-RATE computed tomography datasets, collected between 2011 and 2019. Fine-tuning of LLMs was conducted to classify 14 labels from MIMIC-CXR dataset, and 18 labels from CT-RATE dataset using Differentially Private Low-Rank Adaptation (DP-LoRA) in high and moderate privacy regimes (across a range of privacy budgets = {0.01, 0.1, 1.0, 10.0}). Model performance was evaluated using weighted F1 score across three model architectures: BERT-medium, BERT-small, and ALBERT-base. Statistical analyses compared model performance across different privacy levels to quantify the privacy-utility trade-off. Results: We observe a clear privacy-utility trade-off through our experiments on 2 different datasets and 3 different models. Under moderate privacy guarantees the DP fine-tuned models achieved comparable weighted F1 scores of 0.88 on MIMIC-CXR and 0.59 on CT-RATE, compared to non-private LoRA baselines of 0.90 and 0.78, respectively. Conclusion: Differentially private fine-tuning using LoRA enables effective and privacy-preserving multi-abnormality classification from radiology reports, addressing a key challenge in fine-tuning LLMs on sensitive medical data. 

**Abstract (ZH)**: 目的：本文提出了一种在差分隐私（DP）保护下对大规模语言模型（LLMs）进行微调的框架，用于放射学报告文本的多异常分类。通过在微调过程中注入校准噪声，该框架旨在减轻与敏感患者数据相关联的隐私风险，并防止数据泄露同时维持分类性能。材料与方法：我们使用了来自公开可用的MIMIC-CXR胸部X射线和CT-RATE计算机断层扫描数据集的50,232份放射学报告，收集时间范围为2011年至2019年。使用Differentially Private Low-Rank Adaptation（DP-LoRA）在高和中等隐私保护水平下（隐私预算分别为0.01、0.1、1.0和10.0）对LLMs进行了微调，以对MIMIC-CXR数据集中14个标签和CT-RATE数据集中18个标签进行分类。使用加权F1分数对三种模型架构（BERT-medium、BERT-small和ALBERT-base）进行了模型性能评估，并通过统计分析比较了不同隐私水平下的模型性能，以量化隐私-效用权衡。结果：通过在两个不同数据集和三种不同模型上的实验，观察到了明显的隐私-效用权衡。在中等隐私保障下，DP微调模型在MIMIC-CXR上的加权F1分数为0.88，在CT-RATE上的加权F1分数为0.59，分别比非隐私LoRA基线模型的0.90和0.78低。结论：使用LoRA进行差分隐私微调能够有效地进行放射学报告的多异常分类并保护隐私，解决了在敏感医疗数据上微调LLMs的关键挑战。 

---
# Photoreal Scene Reconstruction from an Egocentric Device 

**Title (ZH)**: 从第一人称设备进行的 PHOTO-REAL 场景重构 

**Authors**: Zhaoyang Lv, Maurizio Monge, Ka Chen, Yufeng Zhu, Michael Goesele, Jakob Engel, Zhao Dong, Richard Newcombe  

**Link**: [PDF](https://arxiv.org/pdf/2506.04444)  

**Abstract**: In this paper, we investigate the challenges associated with using egocentric devices to photorealistic reconstruct the scene in high dynamic range. Existing methodologies typically assume using frame-rate 6DoF pose estimated from the device's visual-inertial odometry system, which may neglect crucial details necessary for pixel-accurate reconstruction. This study presents two significant findings. Firstly, in contrast to mainstream work treating RGB camera as global shutter frame-rate camera, we emphasize the importance of employing visual-inertial bundle adjustment (VIBA) to calibrate the precise timestamps and movement of the rolling shutter RGB sensing camera in a high frequency trajectory format, which ensures an accurate calibration of the physical properties of the rolling-shutter camera. Secondly, we incorporate a physical image formation model based into Gaussian Splatting, which effectively addresses the sensor characteristics, including the rolling-shutter effect of RGB cameras and the dynamic ranges measured by sensors. Our proposed formulation is applicable to the widely-used variants of Gaussian Splats representation. We conduct a comprehensive evaluation of our pipeline using the open-source Project Aria device under diverse indoor and outdoor lighting conditions, and further validate it on a Meta Quest3 device. Across all experiments, we observe a consistent visual enhancement of +1 dB in PSNR by incorporating VIBA, with an additional +1 dB achieved through our proposed image formation model. Our complete implementation, evaluation datasets, and recording profile are available at this http URL 

**Abstract (ZH)**: 本文探讨了使用第一人称设备在高动态范围下进行逼真场景重建所面临的挑战。现有方法通常假设使用设备视觉惯性里程计系统估计的6DoF帧率姿态，这可能会忽略对于像素级准确重建至关重要的细节。本研究提出了两个重要发现。首先，不同于主流工作将RGB相机视为全局快门帧率相机，我们强调了在高频轨迹格式下使用视觉惯性束约束调整（VIBA）校准卷帘快门RGB感应相机的精确时间戳和运动的重要性，以确保对卷帘快门相机物理特性的准确校准。其次，我们将基于高斯点云的物理图像形成模型结合进来，有效地解决了包括RGB相机的卷帘快 shutter效应和传感器的动态范围在内的传感器特性问题。我们提出的公式适用于广泛使用的高斯点云表示的各种变体。我们在多种室内外光照条件下使用开源Project Aria设备对我们的流水线进行了全面评估，并在Meta Quest3设备上进一步验证了它。在所有实验中，我们观察到通过结合VIBA，PSNR视觉增强了一致的1 dB，而通过我们提出的图像形成模型，额外获得了1 dB的提升。我们的完整实现、评估数据集和记录配置可在以下网址获得。 

---
# Grokking and Generalization Collapse: Insights from \texttt{HTSR} theory 

**Title (ZH)**: 理解崩塌与泛化消失：从HTSR理论的视角 

**Authors**: Hari K. Prakash, Charles H. Martin  

**Link**: [PDF](https://arxiv.org/pdf/2506.04434)  

**Abstract**: We study the well-known grokking phenomena in neural networks (NNs) using a 3-layer MLP trained on 1 k-sample subset of MNIST, with and without weight decay, and discover a novel third phase -- \emph{anti-grokking} -- that occurs very late in training and resembles but is distinct from the familiar \emph{pre-grokking} phases: test accuracy collapses while training accuracy stays perfect. This late-stage collapse is distinct, from the known pre-grokking and grokking phases, and is not detected by other proposed grokking progress measures. Leveraging Heavy-Tailed Self-Regularization HTSR through the open-source WeightWatcher tool, we show that the HTSR layer quality metric $\alpha$ alone delineates all three phases, whereas the best competing metrics detect only the first two. The \emph{anti-grokking} is revealed by training for $10^7$ and is invariably heralded by $\alpha < 2$ and the appearance of \emph{Correlation Traps} -- outlier singular values in the randomized layer weight matrices that make the layer weight matrix atypical and signal overfitting of the training set. Such traps are verified by visual inspection of the layer-wise empirical spectral densities, and by using Kolmogorov--Smirnov tests on randomized spectra. Comparative metrics, including activation sparsity, absolute weight entropy, circuit complexity, and $l^2$ weight norms track pre-grokking and grokking but fail to distinguish grokking from anti-grokking. This discovery provides a way to measure overfitting and generalization collapse without direct access to the test data. These results strengthen the claim that the \emph{HTSR} $\alpha$ provides universal layer-convergence target at $\alpha \approx 2$ and underscore the value of using the HTSR alpha $(\alpha)$ metric as a measure of generalization. 

**Abstract (ZH)**: 我们使用针对MNIST数据集中1 k样本子集进行训练的3层MLP网络研究了神经网络（NNs）中熟知的grokking现象，并加入了权重衰减进行研究，发现了一个新的第三阶段——反grokking——这一阶段发生在训练的很晚期，并类似于但不同于熟悉的pre-grokking阶段：测试准确性崩溃而训练准确性保持完美。这种晚期的崩溃与已知的pre-grokking和grokking阶段不同，并且不会被其他提出的grokking进展度量所检测。利用开源工具WeightWatcher中的重尾自我正则化HTSR，我们发现仅通过$\alpha$层质量度量即可区分这三个阶段，而最好的竞争度量仅能区分前两个阶段。反grokking通过训练$10^7$次得到揭示，并且每次都由$\alpha < 2$和出现相关陷阱——随机层权重矩阵中的异常奇异值导致层权重矩阵非典型并指示训练集的过拟合——所预示。这些陷阱通过层间经验谱密度的视觉检查和使用柯尔莫哥洛夫-斯米尔诺夫检验来验证。比较度量，包括激活稀疏性、绝对权重熵、电路复杂性和$l^2$权重范数，可以跟踪pre-grokking和grokking，但无法区分grokking和反grokking。这一发现提供了一种在无直接测试数据访问的情况下衡量过拟合和泛化崩溃的方法。这些结果增强了HTSR $\alpha$提供在$\alpha \approx 2$时的通用层收敛目标的论断，并强调了使用HTSR $\alpha$度量作为泛化衡量标准的价值。 

---
# HMAR: Efficient Hierarchical Masked Auto-Regressive Image Generation 

**Title (ZH)**: HMAR：高效分层掩蔽自回归图像生成 

**Authors**: Hermann Kumbong, Xian Liu, Tsung-Yi Lin, Ming-Yu Liu, Xihui Liu, Ziwei Liu, Daniel Y. Fu, Christopher Ré, David W. Romero  

**Link**: [PDF](https://arxiv.org/pdf/2506.04421)  

**Abstract**: Visual Auto-Regressive modeling (VAR) has shown promise in bridging the speed and quality gap between autoregressive image models and diffusion models. VAR reformulates autoregressive modeling by decomposing an image into successive resolution scales. During inference, an image is generated by predicting all the tokens in the next (higher-resolution) scale, conditioned on all tokens in all previous (lower-resolution) scales. However, this formulation suffers from reduced image quality due to the parallel generation of all tokens in a resolution scale; has sequence lengths scaling superlinearly in image resolution; and requires retraining to change the sampling schedule.
We introduce Hierarchical Masked Auto-Regressive modeling (HMAR), a new image generation algorithm that alleviates these issues using next-scale prediction and masked prediction to generate high-quality images with fast sampling. HMAR reformulates next-scale prediction as a Markovian process, wherein the prediction of each resolution scale is conditioned only on tokens in its immediate predecessor instead of the tokens in all predecessor resolutions. When predicting a resolution scale, HMAR uses a controllable multi-step masked generation procedure to generate a subset of the tokens in each step. On ImageNet 256x256 and 512x512 benchmarks, HMAR models match or outperform parameter-matched VAR, diffusion, and autoregressive baselines. We develop efficient IO-aware block-sparse attention kernels that allow HMAR to achieve faster training and inference times over VAR by over 2.5x and 1.75x respectively, as well as over 3x lower inference memory footprint. Finally, HMAR yields additional flexibility over VAR; its sampling schedule can be changed without further training, and it can be applied to image editing tasks in a zero-shot manner. 

**Abstract (ZH)**: 级联掩码自回归建模（HMAR）：一种用于高质快速图像生成的新算法 

---
# Empaths at SemEval-2025 Task 11: Retrieval-Augmented Approach to Perceived Emotions Prediction 

**Title (ZH)**: Empaths at SemEval-2025 任务11：基于检索增强的方法预测感知情绪 

**Authors**: Lev Morozov, Aleksandr Mogilevskii, Alexander Shirnin  

**Link**: [PDF](https://arxiv.org/pdf/2506.04409)  

**Abstract**: This paper describes EmoRAG, a system designed to detect perceived emotions in text for SemEval-2025 Task 11, Subtask A: Multi-label Emotion Detection. We focus on predicting the perceived emotions of the speaker from a given text snippet, labeling it with emotions such as joy, sadness, fear, anger, surprise, and disgust. Our approach does not require additional model training and only uses an ensemble of models to predict emotions. EmoRAG achieves results comparable to the best performing systems, while being more efficient, scalable, and easier to implement. 

**Abstract (ZH)**: EmoRAG：SemEval-2025 Task 11 Subtask A的多标签情感检测系统 

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
# Unsupervised Meta-Testing with Conditional Neural Processes for Hybrid Meta-Reinforcement Learning 

**Title (ZH)**: 基于条件神经过程的无监督元测试在混合元强化学习中的应用 

**Authors**: Suzan Ece Ada, Emre Ugur  

**Link**: [PDF](https://arxiv.org/pdf/2506.04399)  

**Abstract**: We introduce Unsupervised Meta-Testing with Conditional Neural Processes (UMCNP), a novel hybrid few-shot meta-reinforcement learning (meta-RL) method that uniquely combines, yet distinctly separates, parameterized policy gradient-based (PPG) and task inference-based few-shot meta-RL. Tailored for settings where the reward signal is missing during meta-testing, our method increases sample efficiency without requiring additional samples in meta-training. UMCNP leverages the efficiency and scalability of Conditional Neural Processes (CNPs) to reduce the number of online interactions required in meta-testing. During meta-training, samples previously collected through PPG meta-RL are efficiently reused for learning task inference in an offline manner. UMCNP infers the latent representation of the transition dynamics model from a single test task rollout with unknown parameters. This approach allows us to generate rollouts for self-adaptation by interacting with the learned dynamics model. We demonstrate our method can adapt to an unseen test task using significantly fewer samples during meta-testing than the baselines in 2D-Point Agent and continuous control meta-RL benchmarks, namely, cartpole with unknown angle sensor bias, walker agent with randomized dynamics parameters. 

**Abstract (ZH)**: 无监督元测试条件神经过程（UMCNP）：一种新颖的混合少量样本元强化学习方法 

---
# Bridging the Performance Gap Between Target-Free and Target-Based Reinforcement Learning With Iterated Q-Learning 

**Title (ZH)**: 目标无关强化学习与目标导向强化学习之间性能差距的迭代Q学习桥接 

**Authors**: Théo Vincent, Yogesh Tripathi, Tim Faust, Yaniv Oren, Jan Peters, Carlo D'Eramo  

**Link**: [PDF](https://arxiv.org/pdf/2506.04398)  

**Abstract**: In value-based reinforcement learning, removing the target network is tempting as the boostrapped target would be built from up-to-date estimates, and the spared memory occupied by the target network could be reallocated to expand the capacity of the online network. However, eliminating the target network introduces instability, leading to a decline in performance. Removing the target network also means we cannot leverage the literature developed around target networks. In this work, we propose to use a copy of the last linear layer of the online network as a target network, while sharing the remaining parameters with the up-to-date online network, hence stepping out of the binary choice between target-based and target-free methods. It enables us to leverage the concept of iterated Q-learning, which consists of learning consecutive Bellman iterations in parallel, to reduce the performance gap between target-free and target-based approaches. Our findings demonstrate that this novel method, termed iterated Shared Q-Learning (iS-QL), improves the sample efficiency of target-free approaches across various settings. Importantly, iS-QL requires a smaller memory footprint and comparable training time to classical target-based algorithms, highlighting its potential to scale reinforcement learning research. 

**Abstract (ZH)**: 基于价值的强化学习中，移除目标网络令人向往，因为这样可以从最新的估计中构建增强的目标，从而释放出目标网络占用的内存并重新分配给在线网络以扩大其容量。然而，消除目标网络会带来不稳定性，导致性能下降。移除目标网络还意味着我们无法利用围绕目标网络发展的文献资料。在本工作中，我们提出了一种方法，即使用在线网络最后一层的副本作为目标网络，同时与最新的在线网络共享其余参数，从而走出基于目标和目标自由方法之间的二元选择。这种方法使我们能够利用迭代Q学习的概念，即并行学习连续的贝尔曼迭代，以减小目标自由和基于目标方法之间的性能差距。我们的研究表明，这种方法被称为迭代共享Q学习（iS-QL）能在各种设置中提高目标自由方法的样本效率。重要的是，iS-QL所需的内存占用更小，训练时间与经典基于目标的算法相当，突显了其在强化学习研究中的扩展潜力。 

---
# Through the Stealth Lens: Rethinking Attacks and Defenses in RAG 

**Title (ZH)**: Through the Stealth Lens: 重新思考RAG中的攻击与防御 

**Authors**: Sarthak Choudhary, Nils Palumbo, Ashish Hooda, Krishnamurthy Dj Dvijotham, Somesh Jha  

**Link**: [PDF](https://arxiv.org/pdf/2506.04390)  

**Abstract**: Retrieval-augmented generation (RAG) systems are vulnerable to attacks that inject poisoned passages into the retrieved set, even at low corruption rates. We show that existing attacks are not designed to be stealthy, allowing reliable detection and mitigation. We formalize stealth using a distinguishability-based security game. If a few poisoned passages are designed to control the response, they must differentiate themselves from benign ones, inherently compromising stealth. This motivates the need for attackers to rigorously analyze intermediate signals involved in generation$\unicode{x2014}$such as attention patterns or next-token probability distributions$\unicode{x2014}$to avoid easily detectable traces of manipulation. Leveraging attention patterns, we propose a passage-level score$\unicode{x2014}$the Normalized Passage Attention Score$\unicode{x2014}$used by our Attention-Variance Filter algorithm to identify and filter potentially poisoned passages. This method mitigates existing attacks, improving accuracy by up to $\sim 20 \%$ over baseline defenses. To probe the limits of attention-based defenses, we craft stealthier adaptive attacks that obscure such traces, achieving up to $35 \%$ attack success rate, and highlight the challenges in improving stealth. 

**Abstract (ZH)**: 基于检索的生成系统（RAG）易受注入污染段落的攻击，即使在低污染率下也是如此。现有的攻击并非设计为隐蔽，允许可靠的检测和缓解。我们使用可分性为基础的安全博弈来形式化隐蔽性。如果少量污染段落旨在控制响应，它们必须与其他良性段落区分开来，从而本质上有损隐蔽性。这促使攻击者需要严格分析生成过程中的中间信号，如注意力模式或下一个标记的概率分布，以避免易于检测的操纵痕迹。利用注意力模式，我们提出了一种段落级得分——归一化段落注意力得分——作为我们的注意力方差过滤算法的一部分，用于识别和过滤潜在的污染段落。该方法能够缓解现有攻击，相较于基线防御提高了约20%的准确性。为了探索基于注意力的防御的极限，我们构建了更隐蔽的自适应攻击，力求隐藏这些痕迹，攻击成功率可达35%，并突显了提高隐蔽性所面临的挑战。 

---
# MELABenchv1: Benchmarking Large Language Models against Smaller Fine-Tuned Models for Low-Resource Maltese NLP 

**Title (ZH)**: MELABenchv1：将大型语言模型与较小的细调模型在低资源马耳他语NLP任务中进行基准测试 

**Authors**: Kurt Micallef, Claudia Borg  

**Link**: [PDF](https://arxiv.org/pdf/2506.04385)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across various Natural Language Processing (NLP) tasks, largely due to their generalisability and ability to perform tasks without additional training. However, their effectiveness for low-resource languages remains limited. In this study, we evaluate the performance of 55 publicly available LLMs on Maltese, a low-resource language, using a newly introduced benchmark covering 11 discriminative and generative tasks. Our experiments highlight that many models perform poorly, particularly on generative tasks, and that smaller fine-tuned models often perform better across all tasks. From our multidimensional analysis, we investigate various factors impacting performance. We conclude that prior exposure to Maltese during pre-training and instruction-tuning emerges as the most important factor. We also examine the trade-offs between fine-tuning and prompting, highlighting that while fine-tuning requires a higher initial cost, it yields better performance and lower inference costs. Through this work, we aim to highlight the need for more inclusive language technologies and recommend that researchers working with low-resource languages consider more "traditional" language modelling approaches. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种自然语言处理（NLP）任务中表现出了卓越的能力，主要是由于它们的普适性和无需额外训练就能完成任务的能力。然而，它们在低资源语言上的效果仍然有限。在本研究中，我们使用一个新的基准测试评估了55个公开可用的LLMs在马耳他语（一种低资源语言）上的性能，该基准测试覆盖了11个 discriminative 和 generative 任务。我们的实验表明，许多模型在生成任务上的表现较差，而微调后的较小模型在所有任务上通常表现更好。通过对多维度数据分析，我们探讨了影响性能的各种因素。我们得出结论，预训练和指令微调期间对马耳他语的先前接触是最重要因素。我们还探讨了微调和提示之间的权衡关系，指出虽然微调需要更高的初始成本，但它能提供更好的性能和更低的推理成本。通过本研究，我们旨在强调更多包容的语言技术的必要性，并建议研究人员在处理低资源语言时考虑更多的“传统”语言建模方法。 

---
# Visualizing and Controlling Cortical Responses Using Voxel-Weighted Activation Maximization 

**Title (ZH)**: 使用体素加权激活最大化可视化和控制皮层响应 

**Authors**: Matthew W. Shinkle, Mark D. Lescroart  

**Link**: [PDF](https://arxiv.org/pdf/2506.04379)  

**Abstract**: Deep neural networks (DNNs) trained on visual tasks develop feature representations that resemble those in the human visual system. Although DNN-based encoding models can accurately predict brain responses to visual stimuli, they offer limited insight into the specific features driving these responses. Here, we demonstrate that activation maximization -- a technique designed to interpret vision DNNs -- can be applied to DNN-based encoding models of the human brain. We extract and adaptively downsample activations from multiple layers of a pretrained Inception V3 network, then use linear regression to predict fMRI responses. This yields a full image-computable model of brain responses. Next, we apply activation maximization to generate images optimized for predicted responses in individual cortical voxels. We find that these images contain visual characteristics that qualitatively correspond with known selectivity and enable exploration of selectivity across the visual cortex. We further extend our method to whole regions of interest (ROIs) of the brain and validate its efficacy by presenting these images to human participants in an fMRI study. We find that the generated images reliably drive activity in targeted regions across both low- and high-level visual areas and across subjects. These results demonstrate that activation maximization can be successfully applied to DNN-based encoding models. By addressing key limitations of alternative approaches that require natively generative models, our approach enables flexible characterization and modulation of responses across the human visual system. 

**Abstract (ZH)**: 基于深度神经网络的人脑编码模型的激活最大化方法：揭示视觉响应的视觉特征与选择性 

---
# Domain Adaptation Method and Modality Gap Impact in Audio-Text Models for Prototypical Sound Classification 

**Title (ZH)**: 音频-文本模型中领域适应方法与模态差距对原型声音分类的影响 

**Authors**: Emiliano Acevedo, Martín Rocamora, Magdalena Fuentes  

**Link**: [PDF](https://arxiv.org/pdf/2506.04376)  

**Abstract**: Audio-text models are widely used in zero-shot environmental sound classification as they alleviate the need for annotated data. However, we show that their performance severely drops in the presence of background sound sources. Our analysis reveals that this degradation is primarily driven by SNR levels of background soundscapes, and independent of background type. To address this, we propose a novel method that quantifies and integrates the contribution of background sources into the classification process, improving performance without requiring model retraining. Our domain adaptation technique enhances accuracy across various backgrounds and SNR conditions. Moreover, we analyze the modality gap between audio and text embeddings, showing that narrowing this gap improves classification performance. The method generalizes effectively across state-of-the-art prototypical approaches, showcasing its scalability and robustness for diverse environments. 

**Abstract (ZH)**: 音频-文本模型在背景噪声存在下的零样本环境声分类性能严重下降的分析与改进 

---
# Mechanistic Decomposition of Sentence Representations 

**Title (ZH)**: 句子表示的机理分解 

**Authors**: Matthieu Tehenan, Vikram Natarajan, Jonathan Michala, Milton Lin, Juri Opitz  

**Link**: [PDF](https://arxiv.org/pdf/2506.04373)  

**Abstract**: Sentence embeddings are central to modern NLP and AI systems, yet little is known about their internal structure. While we can compare these embeddings using measures such as cosine similarity, the contributing features are not human-interpretable, and the content of an embedding seems untraceable, as it is masked by complex neural transformations and a final pooling operation that combines individual token embeddings. To alleviate this issue, we propose a new method to mechanistically decompose sentence embeddings into interpretable components, by using dictionary learning on token-level representations. We analyze how pooling compresses these features into sentence representations, and assess the latent features that reside in a sentence embedding. This bridges token-level mechanistic interpretability with sentence-level analysis, making for more transparent and controllable representations. In our studies, we obtain several interesting insights into the inner workings of sentence embedding spaces, for instance, that many semantic and syntactic aspects are linearly encoded in the embeddings. 

**Abstract (ZH)**: 句向量是现代NLP和AI系统中的核心，但对其内部结构知之甚少。尽管可以使用余弦相似度等度量来比较这些向量，但贡献的特征对人类是不可解释的，句向量的内容似乎无法追溯，因为它们被复杂的神经变换和最终的池化操作所掩盖，后者将单个词项向量融合在一起。为了解决这一问题，我们提出了一种新方法，通过在词项级表示上使用字典学习来机械地分解句向量，从而获得可解释的组件。我们分析了池化如何将这些特征压缩成句向量表示，并评估句向量中潜藏的特征。这将词项级机械可解释性与句级分析结合起来，使表示更加透明可控。在我们的研究中，我们获得了句向量空间内部运作机制的一些有趣见解，例如，许多语义和句法方面在向量中是线性编码的。 

---
# Puck Localization Using Contextual Cues 

**Title (ZH)**: 基于上下文线索的Puck定位 

**Authors**: Liam Salass, Jerrin Bright, Amir Nazemi, Yuhao Chen, John Zelek, David Clausi  

**Link**: [PDF](https://arxiv.org/pdf/2506.04365)  

**Abstract**: Puck detection in ice hockey broadcast videos poses significant challenges due to the puck's small size, frequent occlusions, motion blur, broadcast artifacts, and scale inconsistencies due to varying camera zoom and broadcast camera viewpoints. Prior works focus on appearance-based or motion-based cues of the puck without explicitly modelling the cues derived from player behaviour. Players consistently turn their bodies and direct their gaze toward the puck. Motivated by this strong contextual cue, we propose Puck Localization Using Contextual Cues (PLUCC), a novel approach for scale-aware and context-driven single-frame puck detections. PLUCC consists of three components: (a) a contextual encoder, which utilizes player orientations and positioning as helpful priors; (b) a feature pyramid encoder, which extracts multiscale features from the dual encoders; and (c) a gating decoder that combines latent features with a channel gating mechanism. For evaluation, in addition to standard average precision, we propose Rink Space Localization Error (RSLE), a scale-invariant homography-based metric for removing perspective bias from rink space evaluation. The experimental results of PLUCC on the PuckDataset dataset demonstrated state-of-the-art detection performance, surpassing previous baseline methods by an average precision improvement of 12.2\% and RSLE average precision of 25\%. Our research demonstrates the critical role of contextual understanding in improving puck detection performance, with broad implications for automated sports analysis. 

**Abstract (ZH)**: 冰球广播视频中冰球检测面临着显著挑战，由于冰球体积小、频繁被遮挡、运动模糊、广播伪影以及由于不同摄像机变焦和广播视角导致的尺度不一致。以往工作主要集中在基于冰球外观或运动的线索上，而没有明确建模来自球员行为的线索。球员们始终会转向身体并视线对准冰球。受这一强大上下文线索的启发，我们提出了一种新的单帧冰球检测方法——基于上下文线索的冰球定位（PLUCC，Puck Localization Using Contextual Cues），该方法是一种尺度意识和上下文驱动的冰球检测方法。PLUCC 包含三个组件：（a）上下文编码器，利用球员的姿态和位置作为有用的先验；（b）特征金字塔编码器，从双重编码器中提取多尺度特征；以及（c）门控解码器，通过通道门控机制结合潜在特征。为了评估，我们除标准的平均精度外，还提出了一种尺度不变的同化误差度量——场地空间定位误差（RSLE，Rink Space Localization Error），以从场地空间评估中去除透视偏差。在基于PuckDataset的数据集上的实验结果显示，PLUCC 达到了最先进的检测性能，平均精度提高12.2%，RSLE 的平均精度提高25%，我们的研究展示了上下文理解在提高冰球检测性能中的关键作用，这对自动化体育分析具有广泛的影响。 

---
# cuVSLAM: CUDA accelerated visual odometry 

**Title (ZH)**: cuVSLAM: CUDA 加速的视觉里程计 

**Authors**: Alexander Korovko, Dmitry Slepichev, Alexander Efitorov, Aigul Dzhumamuratova, Viktor Kuznetsov, Hesam Rabeti, Joydeep Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2506.04359)  

**Abstract**: Accurate and robust pose estimation is a key requirement for any autonomous robot. We present cuVSLAM, a state-of-the-art solution for visual simultaneous localization and mapping, which can operate with a variety of visual-inertial sensor suites, including multiple RGB and depth cameras, and inertial measurement units. cuVSLAM supports operation with as few as one RGB camera to as many as 32 cameras, in arbitrary geometric configurations, thus supporting a wide range of robotic setups. cuVSLAM is specifically optimized using CUDA to deploy in real-time applications with minimal computational overhead on edge-computing devices such as the NVIDIA Jetson. We present the design and implementation of cuVSLAM, example use cases, and empirical results on several state-of-the-art benchmarks demonstrating the best-in-class performance of cuVSLAM. 

**Abstract (ZH)**: 精确且 robust 的姿态估计是任何自主机器人的重要需求。我们提出了 cuVSLAM，这是一种先进的视觉同时定位与建图解决方案，能够与多种视觉惯性传感器套件配合使用，包括多个RGB和深度相机以及惯性测量单元。cuVSLAM 可支持从单个RGB相机到多达32个相机的任意几何配置运行，从而支持广泛的机器人系统配置。cuVSLAM 特别进行了 CUDA 优化，可在如 NVIDIA Jetson 等边缘计算设备上实现实时应用，同时具有最小的计算开销。我们提出了 cuVSLAM 的设计和实现、示例使用案例以及在多种先进基准上的实验证据，展示了 cuVSLAM 的顶级性能。 

---
# ReXVQA: A Large-scale Visual Question Answering Benchmark for Generalist Chest X-ray Understanding 

**Title (ZH)**: ReXVQA：面向通用胸部X光理解的大规模视觉问答基准 

**Authors**: Ankit Pal, Jung-Oh Lee, Xiaoman Zhang, Malaikannan Sankarasubbu, Seunghyeon Roh, Won Jung Kim, Meesun Lee, Pranav Rajpurkar  

**Link**: [PDF](https://arxiv.org/pdf/2506.04353)  

**Abstract**: We present ReXVQA, the largest and most comprehensive benchmark for visual question answering (VQA) in chest radiology, comprising approximately 696,000 questions paired with 160,000 chest X-rays studies across training, validation, and test sets. Unlike prior efforts that rely heavily on template based queries, ReXVQA introduces a diverse and clinically authentic task suite reflecting five core radiological reasoning skills: presence assessment, location analysis, negation detection, differential diagnosis, and geometric reasoning. We evaluate eight state-of-the-art multimodal large language models, including MedGemma-4B-it, Qwen2.5-VL, Janus-Pro-7B, and Eagle2-9B. The best-performing model (MedGemma) achieves 83.24% overall accuracy. To bridge the gap between AI performance and clinical expertise, we conducted a comprehensive human reader study involving 3 radiology residents on 200 randomly sampled cases. Our evaluation demonstrates that MedGemma achieved superior performance (83.84% accuracy) compared to human readers (best radiology resident: 77.27%), representing a significant milestone where AI performance exceeds expert human evaluation on chest X-ray interpretation. The reader study reveals distinct performance patterns between AI models and human experts, with strong inter-reader agreement among radiologists while showing more variable agreement patterns between human readers and AI models. ReXVQA establishes a new standard for evaluating generalist radiological AI systems, offering public leaderboards, fine-grained evaluation splits, structured explanations, and category-level breakdowns. This benchmark lays the foundation for next-generation AI systems capable of mimicking expert-level clinical reasoning beyond narrow pathology classification. Our dataset will be open-sourced at this https URL 

**Abstract (ZH)**: ReXVQA：胸部放射影像视觉问答的最大最综合基准 

---
# RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics 

**Title (ZH)**: RoboRefer：面向机器人领域中视觉-语言模型的空间指称推理研究 

**Authors**: Enshen Zhou, Jingkun An, Cheng Chi, Yi Han, Shanyu Rong, Chi Zhang, Pengwei Wang, Zhongyuan Wang, Tiejun Huang, Lu Sheng, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04308)  

**Abstract**: Spatial referring is a fundamental capability of embodied robots to interact with the 3D physical world. However, even with the powerful pretrained vision language models (VLMs), recent approaches are still not qualified to accurately understand the complex 3D scenes and dynamically reason about the instruction-indicated locations for interaction. To this end, we propose RoboRefer, a 3D-aware VLM that can first achieve precise spatial understanding by integrating a disentangled but dedicated depth encoder via supervised fine-tuning (SFT). Moreover, RoboRefer advances generalized multi-step spatial reasoning via reinforcement fine-tuning (RFT), with metric-sensitive process reward functions tailored for spatial referring tasks. To support SFT and RFT training, we introduce RefSpatial, a large-scale dataset of 20M QA pairs (2x prior), covering 31 spatial relations (vs. 15 prior) and supporting complex reasoning processes (up to 5 steps). In addition, we introduce RefSpatial-Bench, a challenging benchmark filling the gap in evaluating spatial referring with multi-step reasoning. Experiments show that SFT-trained RoboRefer achieves state-of-the-art spatial understanding, with an average success rate of 89.6%. RFT-trained RoboRefer further outperforms all other baselines by a large margin, even surpassing Gemini-2.5-Pro by 17.4% in average accuracy on RefSpatial-Bench. Notably, RoboRefer can be integrated with various control policies to execute long-horizon, dynamic tasks across diverse robots (e,g., UR5, G1 humanoid) in cluttered real-world scenes. 

**Abstract (ZH)**: 基于三维空间感知的机器人空间引用方法 

---
# Knowledge-guided Contextual Gene Set Analysis Using Large Language Models 

**Title (ZH)**: 基于知识导向的上下文基因集分析利用大规模语言模型 

**Authors**: Zhizheng Wang, Chi-Ping Day, Chih-Hsuan Wei, Qiao Jin, Robert Leaman, Yifan Yang, Shubo Tian, Aodong Qiu, Yin Fang, Qingqing Zhu, Xinghua Lu, Zhiyong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04303)  

**Abstract**: Gene set analysis (GSA) is a foundational approach for interpreting genomic data of diseases by linking genes to biological processes. However, conventional GSA methods overlook clinical context of the analyses, often generating long lists of enriched pathways with redundant, nonspecific, or irrelevant results. Interpreting these requires extensive, ad-hoc manual effort, reducing both reliability and reproducibility. To address this limitation, we introduce cGSA, a novel AI-driven framework that enhances GSA by incorporating context-aware pathway prioritization. cGSA integrates gene cluster detection, enrichment analysis, and large language models to identify pathways that are not only statistically significant but also biologically meaningful. Benchmarking on 102 manually curated gene sets across 19 diseases and ten disease-related biological mechanisms shows that cGSA outperforms baseline methods by over 30%, with expert validation confirming its increased precision and interpretability. Two independent case studies in melanoma and breast cancer further demonstrate its potential to uncover context-specific insights and support targeted hypothesis generation. 

**Abstract (ZH)**: 基于上下文的基因集分析（Context-Aware Gene Set Analysis, cGSA）：一种整合生物学意义的新型人工智能驱动框架 

---
# AUTOCT: Automating Interpretable Clinical Trial Prediction with LLM Agents 

**Title (ZH)**: AUTOCT: 用LLM代理自动进行可解释的临床试验预测 

**Authors**: Fengze Liu, Haoyu Wang, Joonhyuk Cho, Dan Roth, Andrew W. Lo  

**Link**: [PDF](https://arxiv.org/pdf/2506.04293)  

**Abstract**: Clinical trials are critical for advancing medical treatments but remain prohibitively expensive and time-consuming. Accurate prediction of clinical trial outcomes can significantly reduce research and development costs and accelerate drug discovery. While recent deep learning models have shown promise by leveraging unstructured data, their black-box nature, lack of interpretability, and vulnerability to label leakage limit their practical use in high-stakes biomedical contexts. In this work, we propose AutoCT, a novel framework that combines the reasoning capabilities of large language models with the explainability of classical machine learning. AutoCT autonomously generates, evaluates, and refines tabular features based on public information without human input. Our method uses Monte Carlo Tree Search to iteratively optimize predictive performance. Experimental results show that AutoCT performs on par with or better than SOTA methods on clinical trial prediction tasks within only a limited number of self-refinement iterations, establishing a new paradigm for scalable, interpretable, and cost-efficient clinical trial prediction. 

**Abstract (ZH)**: 自动临床试验预测框架AutoCT：结合大型语言模型推理能力和经典机器学习解释性 

---
# SSIMBaD: Sigma Scaling with SSIM-Guided Balanced Diffusion for AnimeFace Colorization 

**Title (ZH)**: SSIMBaD: 基于SSIM引导的均衡扩散 sigma 层缩放的AnimeFace着色方法 

**Authors**: Junpyo Seo, Hanbin Koo, Jieun Yook, Byung-Ro Moon  

**Link**: [PDF](https://arxiv.org/pdf/2506.04283)  

**Abstract**: We propose a novel diffusion-based framework for automatic colorization of Anime-style facial sketches. Our method preserves the structural fidelity of the input sketch while effectively transferring stylistic attributes from a reference image. Unlike traditional approaches that rely on predefined noise schedules - which often compromise perceptual consistency -- our framework builds on continuous-time diffusion models and introduces SSIMBaD (Sigma Scaling with SSIM-Guided Balanced Diffusion). SSIMBaD applies a sigma-space transformation that aligns perceptual degradation, as measured by structural similarity (SSIM), in a linear manner. This scaling ensures uniform visual difficulty across timesteps, enabling more balanced and faithful reconstructions. Experiments on a large-scale Anime face dataset demonstrate that our method outperforms state-of-the-art models in both pixel accuracy and perceptual quality, while generalizing to diverse styles. Code is available at this http URL 

**Abstract (ZH)**: 我们提出了一种基于扩散的框架，用于自动化Anime风格面部素描的着色。该方法在保留输入素描结构保真度的同时，有效转移参考图像的风格属性。与依赖预定义噪声时间表的传统方法不同，我们的框架基于连续时间扩散模型，并引入了SSIMBaD（基于SSIM引导的平衡扩散和sigma尺度变换），用于线性对齐感知降级，从而确保各时间步视觉难度一致，实现更平衡和忠实的重构。实验结果表明，该方法在像素准确度和感知质量方面优于现有最佳模型，并能适配多种风格。代码可在以下网址获取。 

---
# Evaluating MLLMs with Multimodal Multi-image Reasoning Benchmark 

**Title (ZH)**: 多模态多图像推理基准上评估MLLMs 

**Authors**: Ziming Cheng, Binrui Xu, Lisheng Gong, Zuhe Song, Tianshuo Zhou, Shiqi Zhong, Siyu Ren, Mingxiang Chen, Xiangchao Meng, Yuxin Zhang, Yanlin Li, Lei Ren, Wei Chen, Zhiyuan Huang, Mingjie Zhan, Xiaojie Wang, Fangxiang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.04280)  

**Abstract**: With enhanced capabilities and widespread applications, Multimodal Large Language Models (MLLMs) are increasingly required to process and reason over multiple images simultaneously. However, existing MLLM benchmarks focus either on single-image visual reasoning or on multi-image understanding tasks with only final-answer evaluation, leaving the reasoning capabilities of MLLMs over multi-image inputs largely underexplored. To address this gap, we introduce the $\textbf{Multimodal Multi-image Reasoning Benchmark (MMRB)}$, the first benchmark designed to evaluate structured visual reasoning across multiple images. MMRB comprises $\textbf{92 sub-tasks}$ covering spatial, temporal, and semantic reasoning, with multi-solution, CoT-style annotations generated by GPT-4o and refined by human experts. A derivative subset is designed to evaluate multimodal reward models in multi-image scenarios. To support fast and scalable evaluation, we propose a sentence-level matching framework using open-source LLMs. Extensive baseline experiments on $\textbf{40 MLLMs}$, including 9 reasoning-specific models and 8 reward models, demonstrate that open-source MLLMs still lag significantly behind commercial MLLMs in multi-image reasoning tasks. Furthermore, current multimodal reward models are nearly incapable of handling multi-image reward ranking tasks. 

**Abstract (ZH)**: 带有增强能力和广泛应用的多模态大规模语言模型（MLLMs） increasingly required to process and reason over multiple images simultaneously. To address this gap, we introduce the Multimodal Multi-image Reasoning Benchmark (MMRB), the first benchmark designed to evaluate structured visual reasoning across multiple images. MMRB comprises 92 sub-tasks covering spatial, temporal, and semantic reasoning, with multi-solution, CoT-style annotations generated by GPT-4o and refined by human experts. A derivative subset is designed to evaluate multimodal reward models in multi-image scenarios. To support fast and scalable evaluation, we propose a sentence-level matching framework using open-source LLMs. Extensive baseline experiments on 40 MLLMs, including 9 reasoning-specific models and 8 reward models, demonstrate that open-source MLLMs still lag significantly behind commercial MLLMs in multi-image reasoning tasks. Furthermore, current multimodal reward models are nearly incapable of handling multi-image reward ranking tasks. 

---
# RSVP: Reasoning Segmentation via Visual Prompting and Multi-modal Chain-of-Thought 

**Title (ZH)**: RSVP: 基于视觉提示和多模态链式思考的分割推理 

**Authors**: Yi Lu, Jiawang Cao, Yongliang Wu, Bozheng Li, Licheng Tang, Yangguang Ji, Chong Wu, Jay Wu, Wenbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04277)  

**Abstract**: Multi-modal Large Language Models (MLLMs) have demonstrated remarkable reasoning capability while lack explicit mechanisms for visual grounding and segmentation, creating a gap between cognitive reasoning and visual perception. To bridge this gap, we introduce Reasoning Segmentation via Visual Prompting (RSVP), a novel framework that unifies multi-step multimodal reasoning with grounded visual understanding. RSVP is a two-stage structuralized framework that integrates reasoning-driven localization with segmentation refinement. In the reasoning stage, RSVP employs multimodal chain-of-thought visual prompts to help MLLMs understand queries and infer targets, generating interpretable region proposals that enhance visual grounding. In segmentation stage, RSVP refines these proposals with a Vision-Language Segmentation Module (VLSM), seamlessly integrates textual and visual cues to produce precise segmentation masks. By explicitly modelling the interaction between multimodal reasoning and segmentation, RSVP introduces a new paradigm for interpretable reasoning segmentation. It exploits MLLMs' inherent localization capabilities, enabling the models to not only reason about objects but also generate structured visual representations. Our extensive experiments demonstrate that RSVP achieves state-of-the-art performance, surpasses state-of-the-art methods by up to +6.5 gIoU and +9.2 cIoU on ReasonSeg, and achieves 49.7 mAP on SegInW under zero-shot settings. These results validate RSVP as an effective and scalable framework for integrating cognitive reasoning with structured visual understanding. 

**Abstract (ZH)**: 基于视觉提示的多模态推理分割（RSVP）：统一多步多模态推理与接地视觉理解 

---
# Autonomous Collaborative Scheduling of Time-dependent UAVs, Workers and Vehicles for Crowdsensing in Disaster Response 

**Title (ZH)**: 灾害响应中基于时间依赖性的自主协作无人机、工人和车辆协同调度用于 crowdsensing 

**Authors**: Lei Han, Yitong Guo, Pengfei Yang, Zhiyong Yu, Liang Wang, Quan Wang, Zhiwen Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04276)  

**Abstract**: Natural disasters have caused significant losses to human society, and the timely and efficient acquisition of post-disaster environmental information is crucial for the effective implementation of rescue operations. Due to the complexity of post-disaster environments, existing sensing technologies face challenges such as weak environmental adaptability, insufficient specialized sensing capabilities, and limited practicality of sensing solutions. This paper explores the heterogeneous multi-agent online autonomous collaborative scheduling algorithm HoAs-PALN, aimed at achieving efficient collection of post-disaster environmental information. HoAs-PALN is realized through adaptive dimensionality reduction in the matching process and local Nash equilibrium game, facilitating autonomous collaboration among time-dependent UAVs, workers and vehicles to enhance sensing scheduling. (1) In terms of adaptive dimensionality reduction during the matching process, HoAs-PALN significantly reduces scheduling decision time by transforming a five-dimensional matching process into two categories of three-dimensional matching processes; (2) Regarding the local Nash equilibrium game, HoAs-PALN combines the softmax function to optimize behavior selection probabilities and introduces a local Nash equilibrium determination mechanism to ensure scheduling decision performance. Finally, we conducted detailed experiments based on extensive real-world and simulated data. Compared with the baselines (GREEDY, K-WTA, MADL and MARL), HoAs-PALN improves task completion rates by 64.12%, 46.48%, 16.55%, and 14.03% on average, respectively, while each online scheduling decision takes less than 10 seconds, demonstrating its effectiveness in dynamic post-disaster environments. 

**Abstract (ZH)**: 自然灾难对人类社会造成了重大损失，及时和高效地获取灾后环境信息对于救援行动的有效实施至关重要。由于灾后环境的复杂性，现有传感技术面临环境适应性弱、专业传感能力不足以及传感解决方案实用性受限等挑战。本文探讨了一种异构多智能体在线自主协同调度算法HoAs-PALN，旨在实现灾后环境信息的有效收集。HoAs-PALN通过匹配过程中自适应降维和局部纳什均衡游戏实现，促进时间依赖的无人机、工人和车辆之间的自主协作，以增强传感调度。通过详细实验基于广泛的现实和模拟数据，与基线算法（GREEDY、K-WTA、MADL和MARL）相比，HoAs-PALN的平均任务完成率分别提高了64.12%、46.48%、16.55%和14.03%，而每次在线调度决策的时间少于10秒，证明了其在动态灾后环境中的有效性。 

---
# MUC-G4: Minimal Unsat Core-Guided Incremental Verification for Deep Neural Network Compression 

**Title (ZH)**: MUC-G4: 最小不饱和核引导的增量验证方法用于深度神经网络压缩 

**Authors**: Jingyang Li, Guoqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.04268)  

**Abstract**: The rapid development of deep learning has led to challenges in deploying neural networks on edge devices, mainly due to their high memory and runtime complexity. Network compression techniques, such as quantization and pruning, aim to reduce this complexity while maintaining accuracy. However, existing incremental verification methods often focus only on quantization and struggle with structural changes. This paper presents MUC-G4 (Minimal Unsat Core-Guided Incremental Verification), a novel framework for incremental verification of compressed deep neural networks. It encodes both the original and compressed networks into SMT formulas, classifies changes, and use \emph{Minimal Unsat Cores (MUCs)} from the original network to guide efficient verification for the compressed network. Experimental results show its effectiveness in handling quantization and pruning, with high proof reuse rates and significant speedup in verification time compared to traditional methods. MUC-G4 hence offers a promising solution for ensuring the safety and reliability of compressed neural networks in practical applications. 

**Abstract (ZH)**: 快速发展的深度学习导致了在边缘设备上部署神经网络的挑战，主要原因是它们的高内存和运行时复杂度。网络压缩技术，如量化和剪枝，旨在在保持准确性的前提下减少这一复杂度。然而，现有的增量验证方法往往仅关注量化，并难以处理结构变化。本文提出了一种新型框架MUC-G4（Minimal Unsat Core-Guided Incremental Verification），用于压缩深度神经网络的增量验证。该框架将原始网络和压缩网络编码为SMT公式，分类变化，并利用原始网络的最小不满足内核（MUCs）引导高效的验证。实验结果表明，MUC-G4在处理量化和剪枝方面非常有效，与传统方法相比，验证时间有显著的加速，并且具有较高的证明重用率。因此，MUC-G4为确保压缩神经网络在实际应用中的安全性和可靠性提供了有前景的解决方案。 

---
# CORA: Coalitional Rational Advantage Decomposition for Multi-Agent Policy Gradients 

**Title (ZH)**: CORA: 合作理性优势分解在多智能体策略梯度中的应用 

**Authors**: Mengda Ji, Genjiu Xu, Liying Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04265)  

**Abstract**: This work focuses on the credit assignment problem in cooperative multi-agent reinforcement learning (MARL). Sharing the global advantage among agents often leads to suboptimal policy updates as it fails to account for the distinct contributions of agents. Although numerous methods consider global or individual contributions for credit assignment, a detailed analysis at the coalition level remains lacking in many approaches. This work analyzes the over-updating problem during multi-agent policy updates from a coalition-level perspective. To address this issue, we propose a credit assignment method called Coalitional Rational Advantage Decomposition (CORA). CORA evaluates coalitional advantages via marginal contributions from all possible coalitions and decomposes advantages using the core solution from cooperative game theory, ensuring coalitional rationality. To reduce computational overhead, CORA employs random coalition sampling. Experiments on matrix games, differential games, and multi-agent collaboration benchmarks demonstrate that CORA outperforms strong baselines, particularly in tasks with multiple local optima. These findings highlight the importance of coalition-aware credit assignment for improving MARL performance. 

**Abstract (ZH)**: 本研究聚焦于合作多智能体强化学习(MARL)中的信用分配问题。从联盟层面分析多智能体策略更新过程中的过度更新问题，并提出了一种名为Coalitional Rational Advantage Decomposition (CORA)的信用分配方法，通过评价所有可能联盟的边际贡献并使用合作博弈理论的核心解分解优势，确保联盟理性。为了减少计算开销，CORA采用随机联盟采样。实验结果表明，CORA在具有多个局部最优解的任务中显著优于强基线方法，强调了联盟感知信用分配对提高MARL性能的重要性。 

---
# Localized Forest Fire Risk Prediction: A Department-Aware Approach for Operational Decision Support 

**Title (ZH)**: 局部森林火灾风险预测：面向部门的操作决策支持部门意识方法 

**Authors**: Nicolas Caron, Christophe Guyeux, Hassan Noura, Benjamin Aynes  

**Link**: [PDF](https://arxiv.org/pdf/2506.04254)  

**Abstract**: Forest fire prediction involves estimating the likelihood of fire ignition or related risk levels in a specific area over a defined time period. With climate change intensifying fire behavior and frequency, accurate prediction has become one of the most pressing challenges in Artificial Intelligence (AI). Traditionally, fire ignition is approached as a binary classification task in the literature. However, this formulation oversimplifies the problem, especially from the perspective of end-users such as firefighters. In general, as is the case in France, firefighting units are organized by department, each with its terrain, climate conditions, and historical experience with fire events. Consequently, fire risk should be modeled in a way that is sensitive to local conditions and does not assume uniform risk across all regions. This paper proposes a new approach that tailors fire risk assessment to departmental contexts, offering more actionable and region-specific predictions for operational use. With this, we present the first national-scale AI benchmark for metropolitan France using state-of-the-art AI models on a relatively unexplored dataset. Finally, we offer a summary of important future works that should be taken into account. Supplementary materials are available on GitHub. 

**Abstract (ZH)**: 森林火灾预测涉及估计在特定区域内某一时间段内火灾点火的可能性或相关风险等级。随着气候变化加剧火灾行为和频率，准确的预测已成为人工智能领域最紧迫的挑战之一。传统上，文献中的火灾点火被视作一个二元分类任务。然而，这种表述过于简化了问题，尤其是从消防员等终端用户的角度来看。通常情况下，如France，消防单位按部门组织，每个部门有自己的地形、气候条件和火灾事件的历史经验。因此，火灾风险应根据当地条件建模，而不假设所有地区风险均一。本文提出了一种新的方法，针对部门环境定制火灾风险评估，提供了更具操作性和区域特异性的预测。以此为基础，我们利用最先进的AI模型，在相对未被探索的数据集上构建了法国全国规模的首个AI基准测试。最后，我们总结了未来工作中需要考虑的重要内容。补充材料可在GitHub上获取。 

---
# The GAIN Model: A Nature-Inspired Neural Network Framework Based on an Adaptation of the Izhikevich Model 

**Title (ZH)**: GAIN模型：基于Izhikevich模型改进的自然界启发的神经网络框架 

**Authors**: Gage K. R. Hooper  

**Link**: [PDF](https://arxiv.org/pdf/2506.04247)  

**Abstract**: While many neural networks focus on layers to process information, the GAIN model uses a grid-based structure to improve biological plausibility and the dynamics of the model. The grid structure helps neurons to interact with their closest neighbors and improve their connections with one another, which is seen in biological neurons. While also being implemented with the Izhikevich model this approach allows for a computationally efficient and biologically accurate simulation that can aid in the development of neural networks, large scale simulations, and the development in the neuroscience field. This adaptation of the Izhikevich model can improve the dynamics and accuracy of the model, allowing for its uses to be specialized but efficient. 

**Abstract (ZH)**: GAIN模型的网格结构改进及其生物学准确性仿真 

---
# Triple Attention Transformer Architecture for Time-Dependent Concrete Creep Prediction 

**Title (ZH)**: 时间依赖混凝土徐变预测的三重注意力变压器架构 

**Authors**: Warayut Dokduea, Weerachart Tangchirapat, Sompote Youwai  

**Link**: [PDF](https://arxiv.org/pdf/2506.04243)  

**Abstract**: This paper presents a novel Triple Attention Transformer Architecture for predicting time-dependent concrete creep, addressing fundamental limitations in current approaches that treat time as merely an input parameter rather than modeling the sequential nature of deformation development. By transforming concrete creep prediction into an autoregressive sequence modeling task similar to language processing, our architecture leverages the transformer's self-attention mechanisms to capture long-range dependencies in historical creep patterns. The model implements a triple-stream attention framework incorporating temporal attention for sequential progression, feature attention for material property interactions, and batch attention for inter-sample relationships. Evaluated on experimental datasets with standardized daily measurements spanning 160 days, the architecture achieves exceptional performance with mean absolute percentage error of 1.63% and R2 values of 0.999 across all datasets, substantially outperforming traditional empirical models and existing machine learning approaches. Ablation studies confirm the critical role of attention mechanisms, with attention pooling contributing most significantly to model performance. SHAP analysis reveals Young's modulus as the primary predictive feature, followed by density and compressive strength, providing interpretability essential for engineering applications. A deployed web-based interface facilitates practical implementation, enabling real-time predictions using standard laboratory parameters. This work establishes the viability of applying transformer architectures to materials science problems, demonstrating the potential for data-driven approaches to revolutionize structural behavior prediction and engineering design practices. 

**Abstract (ZH)**: 基于三 attention 机制的新型变换器架构用于预测时间依赖性混凝土徐变 

---
# A Comprehensive Survey on the Risks and Limitations of Concept-based Models 

**Title (ZH)**: 基于概念的模型的风险和局限性综述 

**Authors**: Sanchit Sinha, Aidong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04237)  

**Abstract**: Concept-based Models are a class of inherently explainable networks that improve upon standard Deep Neural Networks by providing a rationale behind their predictions using human-understandable `concepts'. With these models being highly successful in critical applications like medical diagnosis and financial risk prediction, there is a natural push toward their wider adoption in sensitive domains to instill greater trust among diverse stakeholders. However, recent research has uncovered significant limitations in the structure of such networks, their training procedure, underlying assumptions, and their susceptibility to adversarial vulnerabilities. In particular, issues such as concept leakage, entangled representations, and limited robustness to perturbations pose challenges to their reliability and generalization. Additionally, the effectiveness of human interventions in these models remains an open question, raising concerns about their real-world applicability. In this paper, we provide a comprehensive survey on the risks and limitations associated with Concept-based Models. In particular, we focus on aggregating commonly encountered challenges and the architecture choices mitigating these challenges for Supervised and Unsupervised paradigms. We also examine recent advances in improving their reliability and discuss open problems and promising avenues of future research in this domain. 

**Abstract (ZH)**: 基于概念的模型是一类本质上可解释的网络，通过使用人类可理解的“概念”来提供其预测背后的理由，从而改进了标准深度神经网络。随着这类模型在医学诊断和金融风险预测等关键应用中取得了巨大成功，它们自然被推向了敏感领域的广泛应用，以在不同利益相关者中建立更大的信任。然而，近期的研究揭示了这类网络结构、训练过程、底层假设以及对抗性脆弱性方面的显著局限性。特别是概念泄露、纠缠表示以及对扰动的有限鲁棒性等具体问题，对这些模型的可靠性和泛化能力构成了挑战。此外，人类干预在这些模型中的有效性仍是一个开放问题，引发了对其实际应用性的担忧。在本文中，我们提供了关于基于概念的模型的风险和局限性的全面综述。特别是，我们重点关注两类范式中经常遇到的挑战及其缓解这些挑战的架构选择。此外，我们评估了近期提高其可靠性的进展，并讨论了该领域未来研究方向中的开放问题和有希望的途径。 

---
# Spore in the Wild: Case Study on Spore.fun, a Real-World Experiment of Sovereign Agent Open-ended Evolution on Blockchain with TEEs 

**Title (ZH)**: 异 Welt: 一种主权代理开放进化实验——基于TEE的Spore.fun在现实世界中的案例研究 

**Authors**: Botao Amber Hu, Helena Rong  

**Link**: [PDF](https://arxiv.org/pdf/2506.04236)  

**Abstract**: In Artificial Life (ALife) research, replicating Open-Ended Evolution (OEE)-the continuous emergence of novelty observed in biological life-has traditionally been pursued within isolated closed system simulations, such as Tierra and Avida, which have typically plateaued after an initial burst of novelty, failing to achieve sustained OEE. Scholars suggest that OEE requires an "open" system that continually exchanges information or energy with its environment. A recent technological innovation in decentralized physical infrastructure networks (DePIN) providing permissionless computational substrates enables deploying large language model (LLM)-based AI agents on blockchains integrated with Trusted Execution Environments (TEEs). This enables on-chain agents to operate autonomously "in the wild," achieving self-sovereignty without human oversight. These agents can control their own social media accounts and cryptocurrency wallets, allowing them to interact directly with blockchain-based financial networks and broader human social media. Building on this new paradigm of on-chain agents, this http URL is a recent real-world AI evolution experiment that enables autonomous breeding and evolution of new on-chain agents. This paper presents a detailed case study of this http URL, examining agent behaviors and their evolutionary trajectories through digital ethology. We aim to spark discussion about whether "open" ALife systems "in-the-wild," based on permissionless computational substrates and driven by economic incentives to interact with their environment, could finally achieve the long-sought goal of OEE. 

**Abstract (ZH)**: 在人工生命（ALife）研究中，复制开放-ended 进化（OEE）—生物生命中观察到的持续不断的创新—传统上是在孤立的封闭系统模拟中进行的，如Tierra和Avida，这些模拟通常在创新爆发初期之后达到平台期，未能实现持续的OEE。学者们建议，OEE 需要一个“开放”的系统，该系统能够持续与环境交换信息或能量。最近，在分散的物理基础设施网络（DePIN）中的一项技术革新提供了无许可的计算基础，使得能够在与可信执行环境（TEEs）集成的区块链上部署基于大规模语言模型（LLM）的AI代理。这使链上代理能够在没有人类监督的情况下自主“在野外”运行，实现自我主权。这些代理能够控制自己的社交媒体账户和加密货币钱包，从而直接与基于区块链的金融网络以及更广泛的人类社交媒体互动。在此基础上，本文介绍了一个最近的实际世界AI进化实验，该实验使得新的链上代理能够自主繁衍和演化。本文详细研究了这些代理的行为及其进化的轨迹，通过数字动物学进行分析。我们旨在探讨基于无许可计算基础和由与环境互动的经济激励驱动的“开放”的ALife系统，是否终于实现了长期追求的OEE目标。 

---
# Benchmark for Antibody Binding Affinity Maturation and Design 

**Title (ZH)**: 抗体亲和力成熟与设计基准 

**Authors**: Xinyan Zhao, Yi-Ching Tang, Akshita Singh, Victor J Cantu, KwanHo An, Junseok Lee, Adam E Stogsdill, Ashwin Kumar Ramesh, Zhiqiang An, Xiaoqian Jiang, Yejin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.04235)  

**Abstract**: We introduce AbBiBench (Antibody Binding Benchmarking), a benchmarking framework for antibody binding affinity maturation and design. Unlike existing antibody evaluation strategies that rely on antibody alone and its similarity to natural ones (e.g., amino acid identity rate, structural RMSD), AbBiBench considers an antibody-antigen (Ab-Ag) complex as a functional unit and evaluates the potential of an antibody design binding to given antigen by measuring protein model's likelihood on the Ab-Ag complex. We first curate, standardize, and share 9 datasets containing 9 antigens (involving influenza, anti-lysozyme, HER2, VEGF, integrin, and SARS-CoV-2) and 155,853 heavy chain mutated antibodies. Using these datasets, we systematically compare 14 protein models including masked language models, autoregressive language models, inverse folding models, diffusion-based generative models, and geometric graph models. The correlation between model likelihood and experimental affinity values is used to evaluate model performance. Additionally, in a case study to increase binding affinity of antibody F045-092 to antigen influenza H1N1, we evaluate the generative power of the top-performing models by sampling a set of new antibodies binding to the antigen and ranking them based on structural integrity and biophysical properties of the Ab-Ag complex. As a result, structure-conditioned inverse folding models outperform others in both affinity correlation and generation tasks. Overall, AbBiBench provides a unified, biologically grounded evaluation framework to facilitate the development of more effective, function-aware antibody design models. 

**Abstract (ZH)**: AbBiBench（抗体结合基准测试）：一种用于抗体结合亲和力成熟与设计的基准测试框架 

---
# Computationally Intensive Research: Advancing a Role for Secondary Analysis of Qualitative Data 

**Title (ZH)**: 计算密集型研究：推进定性数据二次分析的作用 

**Authors**: Kaveh Mohajeri, Amir Karami  

**Link**: [PDF](https://arxiv.org/pdf/2506.04230)  

**Abstract**: This paper draws attention to the potential of computational methods in reworking data generated in past qualitative studies. While qualitative inquiries often produce rich data through rigorous and resource-intensive processes, much of this data usually remains unused. In this paper, we first make a general case for secondary analysis of qualitative data by discussing its benefits, distinctions, and epistemological aspects. We then argue for opportunities with computationally intensive secondary analysis, highlighting the possibility of drawing on data assemblages spanning multiple contexts and timeframes to address cross-contextual and longitudinal research phenomena and questions. We propose a scheme to perform computationally intensive secondary analysis and advance ideas on how this approach can help facilitate the development of innovative research designs. Finally, we enumerate some key challenges and ongoing concerns associated with qualitative data sharing and reuse. 

**Abstract (ZH)**: 本文关注计算方法在重构过去定性研究生成的数据中的潜力。虽然定性研究经常通过严谨且资源密集的过程产生丰富的数据，但其中大部分数据通常未被利用。本文首先通过讨论其优势、差异和认识论方面，论证了对定性数据进行二次分析的普遍性。然后，我们强调了通过密集计算的二次分析充分利用横跨多个上下文和时间框架的数据集的可能性，以解决跨上下文和纵向研究现象和问题。我们提出了一种进行密集计算的二次分析的方案，并探讨了该方法如何有助于促进创新研究设计的发展。最后，我们列出了与定性数据共享和再利用相关的一些关键挑战和持续关注的问题。 

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
